//! Per-pixel develop point-operation chain, mirroring `mock/process.rs`.
//!
//! The kernel owns only point maths: white balance/gain, dehaze, HDR/exposure,
//! the C1 camera colour stage, the display tone/colour stages, output banding
//! and the gamma encode. Geometry, sampling, clarity and layer compositing stay
//! on the CPU and hand this kernel their already-sampled linear RGB.
//!
//! Compiled with `-fmad=false` so every multiply/add stays in the CPU order.

#include <math.h>
#include <float.h>

#define FILM_MAX 16
#define CLUT_N 49
#define CLUT_NODES (49 * 49 * 49)
#define GAMMA_SIZE 8192
#define GAMMA_RANGE 4.0f

// ---------------------------------------------------------------------------
// Parameter layout. Keep byte-for-byte in sync with develop_gpu.rs.
// ---------------------------------------------------------------------------
#define P_W 0
#define P_H 1
#define P_RX 2
#define P_RY 3
#define P_RW 4
#define P_RH 5
#define P_FINE 6
#define P_BYPASS 7
#define P_HAS_GAIN 8
#define P_WB 9
#define P_EXPOSURE 12
#define P_C1 13
#define P_PREMAT 14
#define P_FILM_N 23
#define P_BANDING 24
#define P_BG_GAIN 25
#define P_BG_LIMIT 26
#define P_BG_STRENGTH 27
#define P_BG_FW 28
#define P_BG_FH 29
#define P_DEHAZE_OMEGA 30
#define P_DEHAZE_C 31
#define P_DEHAZE_LO 34
#define P_DEHAZE_HI 37
#define P_HDR 40
#define P_BRIGHT_GAMMA 44
#define P_CONTRAST 45
#define P_LEVELS 46
#define P_LV_LO 47
#define P_LV_SPAN 50
#define P_LV_INV 53
#define P_LV_MLO 56
#define P_LV_MSPAN 59
#define P_LV_MINV 62
#define P_LV_OUT_GAIN 65
#define P_LV_OUT_SHADOW 66
#define P_CURVE 67
#define P_SAT 72
#define P_CB 73
#define P_CB_OFF 74
#define P_CBG 86
#define P_CBG_GAIN 87
#define P_BW 99
#define P_BW_WMIN 100
#define P_BW_WMAX 103
#define P_BW_SHIFT 106
#define P_BW_USE_LUMA 107
#define P_BW_TINT_LIGHT 108
#define P_BW_TINT_DARK 111
#define P_BW_DARK_HI 114
#define P_BW_LIGHT_LO 115
#define P_CE 116
#define P_CE_EXTRA 117
#define P_N_SKIN 118
#define P_N_SHIFT 119
#define P_BW_SPLIT 120
#define P_BANDING_OUTER 121
#define P_SIZE 122

// Curve LUT order inside `luts`: rgb, red, green, blue, luma.
#define LUT_RGB 0
#define LUT_R 256
#define LUT_G 512
#define LUT_B 768
#define LUT_LUMA 1024
#define GAMMA_OFF 1280

// Editor descriptor layout.
#define E_GLOBAL_ENABLED 0
#define E_GLOBAL_BAND 1
#define E_GLOBAL_SHIFT 13
#define E_BASIC_EXTRA 16
#define E_BASIC 17
#define E_BASIC_STRIDE 16
#define E_SKIN_BASE 145
#define E_SKIN_STRIDE 24
#define E_SHIFT_STRIDE 22

static __device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

static __device__ __forceinline__ float madd(float a, float b, float c) {
    return a * b + c;
}

static __device__ __forceinline__ float rem_euclid(float x, float m) {
    float r = fmodf(x, m);
    return r < 0.0f ? r + m : r;
}

static __device__ __forceinline__ float luma709(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

static __device__ __forceinline__ float luma601(float r, float g, float b) {
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

static __device__ __forceinline__ float sstep(float a, float b, float x) {
    if (b <= a) {
        return x < a ? 0.0f : 1.0f;
    }
    float t = clampf((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// ---------------------------------------------------------------------------
// Gamma / LUT helpers
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float linear_to_srgb(float v) {
    if (v <= 0.0031308f) {
        return v * 12.92f;
    }
    return 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
}

static __device__ __forceinline__ float gamma_lookup(const float* gamma, float v) {
    int i = (int)(fmaxf(v, 0.0f) / GAMMA_RANGE * (float)(GAMMA_SIZE - 1));
    if (i < 0) i = 0;
    if (i > GAMMA_SIZE - 1) i = GAMMA_SIZE - 1;
    return gamma[i];
}

static __device__ __forceinline__ float encode_srgb(
    const float* P, const float* luts, float c) {
    if (P[P_FINE] != 0.0f) {
        return linear_to_srgb(fmaxf(c, 0.0f));
    }
    return gamma_lookup(luts + GAMMA_OFF, c);
}

static __device__ __forceinline__ float lut_get(const float* lut, float v) {
    float x = clampf(v, 0.0f, 1.0f) * 255.0f;
    int i = (int)x;
    if (i > 254) i = 254;
    float t = x - (float)i;
    return lut[i] + (lut[i + 1] - lut[i]) * t;
}

// ---------------------------------------------------------------------------
// Scene-linear edits: WB, gain plane, dehaze, HDR, exposure.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float dehaze_soft(float x, float a) {
    if (a <= 0.0f) {
        return fmaxf(x, 0.0f);
    }
    float t = fmaxf(a - fabsf(x), 0.0f);
    return fmaxf(x, 0.0f) + t * t * t / (6.0f * a * a);
}

static __device__ __forceinline__ float3 apply_dehaze(float3 g, const float* P) {
    const float omega = P[P_DEHAZE_OMEGA];
    if (omega <= 0.0f || !isfinite(omega)) {
        return g;
    }
    float c[3] = {
        fmaxf(P[P_DEHAZE_C + 0], 0.0f),
        fmaxf(P[P_DEHAZE_C + 1], 0.0f),
        fmaxf(P[P_DEHAZE_C + 2], 0.0f),
    };
    float a[3] = {c[0] * 65535.0f, c[1] * 65535.0f, c[2] * 65535.0f};
    if (a[0] <= FLT_EPSILON && a[1] <= FLT_EPSILON && a[2] <= FLT_EPSILON) {
        return g;
    }
    bool use_level[3];
    float lo[3];
    float span[3];
    for (int k = 0; k < 3; ++k) {
        lo[k] = P[P_DEHAZE_LO + k];
        span[k] = P[P_DEHAZE_HI + k] - lo[k];
        use_level[k] = span[k] > 1e-3f;
    }
    float in[3];
    for (int k = 0; k < 3; ++k) {
        float v = k == 0 ? g.x : (k == 1 ? g.y : g.z);
        in[k] = use_level[k] ? (v - lo[k]) / span[k] * 65535.0f : v * 65535.0f;
    }
    const float eps = 1e-6f;
    const float delta = fminf(2.0f * omega, 1.0f);
    float ah[3] = {c[0] * delta, c[1] * delta, c[2] * delta};
    float i1[3];
    float a1[3];
    for (int k = 0; k < 3; ++k) {
        i1[k] = dehaze_soft(in[k] - ah[k], ah[k]);
        a1[k] = dehaze_soft(a[k] - ah[k], ah[k]);
    }
    float recovered[3];
    float li = luma709(i1[0], i1[1], i1[2]);
    float la = luma709(a1[0], a1[1], a1[2]);
    float keep = la / fmaxf(li + la, eps);
    for (int k = 0; k < 3; ++k) {
        recovered[k] = (1.0f - keep) * i1[k];
        i1[k] *= keep;
    }
    float m_raw = fminf(fminf(i1[0] / fmaxf(a1[0], eps), i1[1] / fmaxf(a1[1], eps)),
                        i1[2] / fmaxf(a1[2], eps));
    float m0 = clampf(m_raw, 0.0f, 1.0f);
    float mag = sqrtf(i1[0] * i1[0] + i1[1] * i1[1] + i1[2] * i1[2]);
    float dvec[3] = {m0 * a1[0], m0 * a1[1], m0 * a1[2]};
    float dx = i1[0] - dvec[0];
    float dy = i1[1] - dvec[1];
    float dz = i1[2] - dvec[2];
    float mag_d = sqrtf(dx * dx + dy * dy + dz * dz);
    float ratio = fmaxf(mag + eps, eps);
    float val = (ratio * 0.375f - (mag_d + eps) * (4.0f / 7.0f)) / ratio;
    float t = 1.0f / (1.0f + expf((0.5f - val) * 4.0f));
    float m = m0 * t;
    for (int k = 0; k < 3; ++k) {
        recovered[k] += (1.0f - t) * i1[k];
        i1[k] *= t;
    }
    float t0 = fmaxf(1.0f - omega * m, 0.05f);
    float out[3];
    for (int k = 0; k < 3; ++k) {
        out[k] = (i1[k] - a1[k]) / t0 + a1[k] + recovered[k];
    }
    float max_i = fmaxf(fmaxf(i1[0], i1[1]), i1[2]);
    if (max_i > eps) {
        float sat_i = (max_i - fminf(fminf(i1[0], i1[1]), i1[2])) / max_i;
        float limit = sat_i * sqrtf(1.0f + (0.5f + 2.0f * m_raw) * (0.5f + 2.0f * m_raw));
        float max_o = fmaxf(fmaxf(out[0], out[1]), out[2]);
        if (max_o > eps) {
            float sat_o = (max_o - fminf(fminf(out[0], out[1]), out[2])) / max_o;
            if (sat_o > limit) {
                float k = limit / sat_o;
                float l = luma709(out[0], out[1], out[2]);
                out[0] = k * out[0] + (1.0f - k) * l;
                out[1] = k * out[1] + (1.0f - k) * l;
                out[2] = k * out[2] + (1.0f - k) * l;
            }
        }
    }
    float result[3];
    for (int k = 0; k < 3; ++k) {
        result[k] = use_level[k]
            ? fmaxf(lo[k] + out[k] * span[k] / 65535.0f, 0.0f)
            : fmaxf(out[k] / 65535.0f, 0.0f);
    }
    return make_float3(result[0], result[1], result[2]);
}

static __device__ __forceinline__ float hdr_toe(float x, float k) {
    float u = x * (2.0f * 0.9f);
    float v = fabsf(u) + 1.0f;
    float r = (u * v) / (v + u * u) * 0.5f + 0.5f;
    r = fmaxf(r, u - r + 1.0f) * (1.0f / (2.0f * k));
    float s = r * r;
    float num = (1.0f + r + s) * (r + s * s);
    return num / (1.0f + num) * k;
}

static __device__ __forceinline__ float hdr_inv_toe(float x, float k) {
    return hdr_toe(-x, k);
}

#define HDR_K 1.0f

static __device__ __forceinline__ float3 apply_hdr(float3 g, const float* P) {
    const float highlights = P[P_HDR + 0];
    const float shadows = P[P_HDR + 1];
    const float whites = P[P_HDR + 2];
    const float blacks = P[P_HDR + 3];
    float x = luma709(g.x, g.y, g.z);
    float shift = 0.0f;
    if (whites != 0.0f) {
        shift += whites / 100.0f * 0.2f * sstep(0.65f, 1.0f, x);
    }
    if (blacks != 0.0f) {
        shift += blacks / 100.0f * 0.12f * (1.0f - sstep(0.0f, 0.25f, x));
    }
    if (fabsf(shift) > 1e-5f) {
        float k = fmaxf(x + shift, 0.0f) / fmaxf(x, 1e-4f);
        g.x *= k;
        g.y *= k;
        g.z *= k;
    }
    if (highlights == 0.0f && shadows == 0.0f) {
        return g;
    }
    x = luma709(g.x, g.y, g.z);
    float m = clampf(highlights / 100.0f, -1.0f, 1.0f);
    float b = 1.0f;
    if (m != 0.0f) {
        float t = hdr_toe(x, HDR_K);
        b = exp2f(m < 0.0f ? -t : t) * fabsf(m) + (1.0f - fabsf(m));
    }
    m = clampf(shadows / 100.0f, -1.0f, 1.0f);
    float a = 1.0f;
    if (m != 0.0f) {
        float t = hdr_inv_toe(x, HDR_K);
        a = exp2f(m < 0.0f ? -t : t) * fabsf(m) + (1.0f - fabsf(m));
    }
    float gain = a / b;
    g.x *= gain;
    g.y *= gain;
    g.z *= gain;
    return g;
}

static __device__ __forceinline__ float3 linear_edits(
    float3 lin, const float* P, const float* gain, int index, float fx, float fy) {
    float3 c = make_float3(lin.x * P[P_WB + 0], lin.y * P[P_WB + 1], lin.z * P[P_WB + 2]);
    if (P[P_HAS_GAIN] != 0.0f && gain != 0) {
        float g = gain[index];
        c.x *= g;
        c.y *= g;
        c.z *= g;
    }
    c = apply_dehaze(c, P);
    c = apply_hdr(c, P);
    float e = P[P_EXPOSURE];
    c.x *= e;
    c.y *= e;
    c.z *= e;
    (void)fx;
    (void)fy;
    return c;
}

// ---------------------------------------------------------------------------
// C1 camera colour stage
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float film_at(
    const float* film, int count, float code) {
    if (count <= 0) {
        return code;
    }
    int right = 1;
    while (right < count && film[right * 3] <= code) {
        ++right;
    }
    if (right < 1) right = 1;
    if (right > count - 1) right = count - 1;
    float x0 = film[(right - 1) * 3];
    float y0 = film[(right - 1) * 3 + 1];
    float m0 = film[(right - 1) * 3 + 2];
    float x1 = film[right * 3];
    float y1 = film[right * 3 + 1];
    float m1 = film[right * 3 + 2];
    float dx = x1 - x0;
    if (fabsf(dx) < 0.0001f) {
        return madd(code - x0, m0 + m1, y0 + y1) * 0.5f;
    }
    float t = (code - x0) / dx;
    float s = t - 1.0f;
    float base = madd(t * t * (y1 - y0), madd(-2.0f, t, 3.0f), y0);
    return madd(madd(m0, s, m1 * t) * t * s, dx, base);
}

static __device__ __forceinline__ float couple_encode(float code) {
    const float K = 4.0f / 256.0f;
    const float SLOPE = 1.0f / (1.0f - 0.5f * K);
    const float TOE = SLOPE / (2.0f * K);
    const float INTERCEPT = 1.0f - SLOPE;
    return code > K * K ? madd(SLOPE, sqrtf(code), INTERCEPT) : TOE * code;
}

static __device__ __forceinline__ float couple_luma(float r, float g, float b) {
    return fabsf(madd(b, 0.114f, madd(r, 0.299f, g * 0.587f))) + 1e-10f;
}

static __device__ __forceinline__ float couple_range(float r, float g, float b) {
    return fmaxf(r, fmaxf(g, b)) - fminf(r, fminf(g, b));
}

static __device__ __forceinline__ float3 couple_float(float3 before, float3 curved) {
    const float K = 4.0f / 256.0f;
    const float SLOPE = 1.0f / (1.0f - 0.5f * K);
    const float TOE = SLOPE / (2.0f * K);
    const float INTERCEPT = 1.0f - SLOPE;
    float src[3] = {
        couple_encode(before.x), couple_encode(before.y), couple_encode(before.z)};
    float tgt[3] = {
        couple_encode(curved.x), couple_encode(curved.y), couple_encode(curved.z)};
    float y_target = couple_luma(tgt[0], tgt[1], tgt[2]);
    if (tgt[0] > 1.0f || tgt[1] > 1.0f || tgt[2] > 1.0f) {
        float limit = couple_luma(fminf(tgt[0], 1.0f), fminf(tgt[1], 1.0f), fminf(tgt[2], 1.0f));
        tgt[0] *= limit / y_target;
        tgt[1] *= limit / y_target;
        tgt[2] *= limit / y_target;
        y_target = limit;
    }
    float src_luma = couple_luma(src[0], src[1], src[2]);
    src[0] *= y_target / src_luma;
    src[1] *= y_target / src_luma;
    src[2] *= y_target / src_luma;
    float gain = madd((couple_range(tgt[0], tgt[1], tgt[2]) + 1e-6f)
                          / (couple_range(src[0], src[1], src[2]) + 1e-6f),
                      0.5f, 0.5f);
    float result[3] = {
        madd(src[0] - y_target, gain, y_target),
        madd(src[1] - y_target, gain, y_target),
        madd(src[2] - y_target, gain, y_target),
    };
    float distance = fmaxf(fmaxf(result[0], result[1]), result[2]) - y_target;
    if (distance > 0.0f && 1.0f - y_target < 2.0f * distance) {
        float ratio = (1.0f - y_target) / distance;
        float edge = fmaxf(1.0f - fabsf(ratio - 1.0f), 0.0f);
        float smooth = madd(edge * edge * edge, -1.0f / 6.0f, fminf(ratio, 1.0f));
        result[0] = madd(result[0] - y_target, smooth, y_target);
        result[1] = madd(result[1] - y_target, smooth, y_target);
        result[2] = madd(result[2] - y_target, smooth, y_target);
    }
    float curved_arr[3] = {curved.x, curved.y, curved.z};
    float out[3];
    for (int c = 0; c < 3; ++c) {
        float value = result[c];
        float decoded;
        if (value > K * K * TOE) {
            float z = (value - INTERCEPT) * (1.0f / SLOPE);
            decoded = z * z;
        } else {
            decoded = (1.0f / TOE) * value;
        }
        out[c] = madd(curved_arr[c], 0.35f, decoded * (1.0f - 0.35f));
    }
    return make_float3(out[0], out[1], out[2]);
}

static __device__ __forceinline__ float encode_camera(float sample) {
    if (sample > 0.01947f) {
        return madd(1.075f, sqrtf(sample), 1.0f - 1.075f);
    }
    return 3.8521f * fmaxf(sample, 0.0f);
}

static __device__ __forceinline__ float3 clut_sample(const unsigned short* nodes, float3 rgb) {
    float peak = 1.0f;
    peak = fmaxf(peak, rgb.x);
    peak = fmaxf(peak, rgb.y);
    peak = fmaxf(peak, rgb.z);
    float scale = 1.0f / peak;
    float components[3] = {rgb.x, rgb.y, rgb.z};
    int base_index[3];
    float fractions[3];
    for (int c = 0; c < 3; ++c) {
        float position = clampf(components[c] * scale, 0.0f, 1.0f) * (float)(CLUT_N - 1);
        base_index[c] = (int)position;
        fractions[c] = position - (float)base_index[c];
    }
    int order[3] = {1, 0, 2};
    if (fractions[0] > fractions[1]) {
        int tmp = order[0];
        order[0] = order[1];
        order[1] = tmp;
    }
    if (fractions[2] > fractions[order[0]]) {
        int tmp = order[2];
        order[2] = order[1];
        order[1] = order[0];
        order[0] = tmp;
    } else if (fractions[2] >= fractions[order[1]]) {
        int tmp = order[1];
        order[1] = order[2];
        order[2] = tmp;
    }
    float largest = fractions[order[0]];
    float middle = fractions[order[1]];
    float smallest = fractions[order[2]];
    float weights[4] = {1.0f - largest, largest - middle, middle - smallest, smallest};
    int corner[3] = {base_index[0], base_index[1], base_index[2]};
    float vertices[4][3];
    {
        int index = (corner[0] * CLUT_N + corner[1]) * CLUT_N + corner[2];
        vertices[0][0] = (float)nodes[index * 3 + 0];
        vertices[0][1] = (float)nodes[index * 3 + 1];
        vertices[0][2] = (float)nodes[index * 3 + 2];
    }
    for (int slot = 1; slot < 4; ++slot) {
        int axis = order[slot - 1];
        corner[axis] = min(corner[axis] + 1, CLUT_N - 1);
        int index = (corner[0] * CLUT_N + corner[1]) * CLUT_N + corner[2];
        vertices[slot][0] = (float)nodes[index * 3 + 0];
        vertices[slot][1] = (float)nodes[index * 3 + 1];
        vertices[slot][2] = (float)nodes[index * 3 + 2];
    }
    float out[3];
    for (int c = 0; c < 3; ++c) {
        float code = ((vertices[0][c] * weights[0] + vertices[1][c] * weights[1])
                      + vertices[2][c] * weights[2]) + vertices[3][c] * weights[3];
        out[c] = (code * peak) * (1.0f / 65536.0f);
    }
    return make_float3(out[0], out[1], out[2]);
}

static __device__ __forceinline__ float3 c1_develop(
    const float* P, const float* c1, float3 rgb) {
    const float* film = c1;
    const unsigned short* nodes = (const unsigned short*)(c1 + FILM_MAX * 3);
    float linear[3] = {fmaxf(rgb.x, 0.0f), fmaxf(rgb.y, 0.0f), fmaxf(rgb.z, 0.0f)};
    float matrix[3];
    for (int row = 0; row < 3; ++row) {
        const float* m = P + P_PREMAT + row * 3;
        float v = m[0] * linear[0] + m[1] * linear[1] + m[2] * linear[2];
        matrix[row] = fmaxf(v, 0.0f);
    }
    int film_count = (int)P[P_FILM_N];
    float curved[3];
    for (int c = 0; c < 3; ++c) {
        curved[c] = film_at(film, film_count, matrix[c]);
    }
    float3 coupled = couple_float(
        make_float3(matrix[0], matrix[1], matrix[2]),
        make_float3(curved[0], curved[1], curved[2]));
    float encoded[3] = {
        encode_camera(coupled.x), encode_camera(coupled.y), encode_camera(coupled.z)};
    float3 camera = make_float3(encoded[0], encoded[1], encoded[2]);
    return clut_sample(nodes, camera);
}

// ---------------------------------------------------------------------------
// Display stages
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float tone_contrast(float g, float amount) {
    if (fabsf(amount) < 1e-4f) {
        return g;
    }
    float k = 1.0f + amount;
    if (k >= 1.0f) {
        float x = (g - 0.5f) * k;
        return 0.5f + x / (1.0f + fabsf(x) * (k - 1.0f) * 0.8f);
    }
    return 0.5f + (g - 0.5f) * k;
}

static __device__ __forceinline__ float levels_input(float v, float lo, float span, float inv_gamma) {
    float t = clampf((v - lo) / span, 0.0f, 1.0f);
    return powf(t, inv_gamma);
}

static __device__ __forceinline__ float3 color_balance(float3 g, const float* P) {
    float l = clampf(luma709(g.x, g.y, g.z), 0.0f, 1.0f);
    float ws = (1.0f - l) * (1.0f - l);
    float wh = l * l;
    float wm = fmaxf(1.0f - ws - wh, 0.0f);
    float w[4] = {1.0f, ws, wm, wh};
    float out[3] = {g.x, g.y, g.z};
    for (int wheel = 0; wheel < 4; ++wheel) {
        for (int c = 0; c < 3; ++c) {
            out[c] += P[P_CB_OFF + wheel * 3 + c] * w[wheel];
        }
    }
    return make_float3(out[0], out[1], out[2]);
}

static __device__ __forceinline__ float3 color_balance_gains(float3 g, const float* P) {
    float l_in = luma709(g.x, g.y, g.z);
    float out[3] = {
        g.x * P[P_CBG_GAIN + 0], g.y * P[P_CBG_GAIN + 1], g.z * P[P_CBG_GAIN + 2]};
    float maxc = fmaxf(fmaxf(out[0], out[1]), out[2]);
    if (maxc > 1e-6f && isfinite(maxc)) {
        float protect = maxc <= 1.0f ? 1.0f : clampf(2.0f - maxc, 0.0f, 1.0f);
        float bell = 4.0f * maxc * (1.0f - maxc);
        float wh = maxc * maxc * protect;
        float wm = bell * bell * bell * 0.5f;
        float ws = (1.0f - maxc) * (1.0f - maxc) * 0.5f;
        float zone_w[3] = {wh, wm, ws};
        float norm[3] = {out[0] / maxc, out[1] / maxc, out[2] / maxc};
        // Zone order in the CPU: highlight, midtone, shadow.
        const int zone_offset[3] = {9, 6, 3};
        for (int zone = 0; zone < 3; ++zone) {
            int param = P_CBG_GAIN + zone_offset[zone];
            float z[3] = {P[param + 0], P[param + 1], P[param + 2]};
            float gmin = fminf(fminf(z[0], z[1]), z[2]);
            float gmax = fmaxf(fmaxf(z[0], z[1]), z[2]);
            float w = zone_w[zone];
            if (!isfinite(w) || fabsf(w) <= 1e-6f || !(gmin > 1e-6f) || gmax - gmin <= 1e-3f) {
                continue;
            }
            for (int c = 0; c < 3; ++c) {
                float step = (z[c] - gmin) / gmin;
                if (isfinite(step)) {
                    out[c] += w * step * norm[c];
                }
            }
        }
    }
    float l_out = luma709(out[0], out[1], out[2]);
    if (fabsf(l_out) > 1e-6f && l_in > 0.0f && isfinite(l_out)) {
        float k = clampf(l_in / l_out, 0.0f, 8.0f);
        out[0] *= k;
        out[1] *= k;
        out[2] *= k;
    }
    for (int c = 0; c < 3; ++c) {
        out[c] = isfinite(out[c]) ? fmaxf(out[c], 0.0f) : 0.0f;
    }
    return make_float3(out[0], out[1], out[2]);
}

// B&W ----------------------------------------------------------------------
static __device__ __forceinline__ float bw_sector_y(
    float3 g, const float* P) {
    int q[3];
    float clamped[3] = {clampf(g.x, 0.0f, 1.0f), clampf(g.y, 0.0f, 1.0f), clampf(g.z, 0.0f, 1.0f)};
    for (int c = 0; c < 3; ++c) {
        q[c] = (int)roundf(clamped[c] * 65535.0f);
    }
    int w_min[3] = {(int)P[P_BW_WMIN + 0], (int)P[P_BW_WMIN + 1], (int)P[P_BW_WMIN + 2]};
    int w_max[3] = {(int)P[P_BW_WMAX + 0], (int)P[P_BW_WMAX + 1], (int)P[P_BW_WMAX + 2]};
    int i_max = q[0] > q[2] ? 0 : 2;
    int max_v = q[0] > q[2] ? q[0] : q[2];
    if (q[1] > max_v) {
        i_max = 1;
        max_v = q[1];
    }
    int i_min = q[1] < q[0] ? 1 : 0;
    int min_v = q[1] < q[0] ? q[1] : q[0];
    if (q[2] < min_v) {
        i_min = 2;
        min_v = q[2];
    }
    int mid = q[3 - i_max - i_min];
    int t = min_v + ((mid - min_v) * w_min[i_min] + (max_v - mid) * w_max[i_max]) / 1024;
    int y = (q[0] * 340 + q[1] * 626 + q[2] * 56) / 1024;
    if (t * 8 < y) {
        if (t <= 0) {
            t = 0;
        } else {
            unsigned int yu = (unsigned int)y;
            unsigned int a = ((yu * yu) >> 5) * 25u;
            unsigned int b = 1u + ((yu * 288u - (unsigned int)t * 1024u) >> 5);
            int floor_value = (int)(a / b) - (int)(yu / 32u);
            t = floor_value > 0 ? floor_value : 0;
        }
    }
    int shift = (int)P[P_BW_SHIFT];
    int rounded = (t >> shift) << shift;
    if (rounded < 0) rounded = 0;
    if (rounded > 65535) rounded = 65535;
    return (float)rounded / 65535.0f;
}

static __device__ __forceinline__ float3 bw_split_tone(float y, const float* P) {
    float light[3] = {P[P_BW_TINT_LIGHT + 0], P[P_BW_TINT_LIGHT + 1], P[P_BW_TINT_LIGHT + 2]};
    float dark[3] = {P[P_BW_TINT_DARK + 0], P[P_BW_TINT_DARK + 1], P[P_BW_TINT_DARK + 2]};
    float dark_hi = P[P_BW_DARK_HI];
    float light_lo = P[P_BW_LIGHT_LO];
    if (light_lo <= dark_hi) {
        return make_float3(y, y, y);
    }
    float dark_ramp[3];
    float light_ramp[3];
    for (int c = 0; c < 3; ++c) {
        dark_ramp[c] = dark[c] * y / light_lo;
        light_ramp[c] = light[c] * y / (light_lo - dark_hi);
    }
    float out[3];
    for (int c = 0; c < 3; ++c) {
        if (y < dark_hi) {
            out[c] = dark_ramp[c];
        } else if (y > light_lo) {
            out[c] = light_ramp[c];
        } else {
            float t = (y - dark_hi) / (light_lo - dark_hi);
            out[c] = dark_ramp[c] + (light_ramp[c] - dark_ramp[c]) * t;
        }
        out[c] = clampf(out[c], 0.0f, 1.0f);
    }
    return make_float3(out[0], out[1], out[2]);
}

static __device__ __forceinline__ float3 black_white(float3 g, const float* P) {
    float y = P[P_BW_USE_LUMA] != 0.0f
        ? luma709(g.x, g.y, g.z)
        : bw_sector_y(g, P);
    y = fmaxf(y, 0.0f);
    if (P[P_BW_SPLIT] == 0.0f) {
        return make_float3(y, y, y);
    }
    return bw_split_tone(y, P);
}

// Color editor -------------------------------------------------------------
static __device__ __forceinline__ void rgb_to_hsv(float3 c, float* h, float* s, float* v) {
    float mx = fmaxf(fmaxf(c.x, c.y), c.z);
    float mn = fminf(fminf(c.x, c.y), c.z);
    float d = mx - mn;
    float hue;
    if (d < 1e-6f) {
        hue = 0.0f;
    } else if (mx == c.x) {
        hue = 60.0f * rem_euclid((c.y - c.z) / d, 6.0f);
    } else if (mx == c.y) {
        hue = 60.0f * ((c.z - c.x) / d + 2.0f);
    } else {
        hue = 60.0f * ((c.x - c.y) / d + 4.0f);
    }
    *h = hue;
    *s = mx < 1e-6f ? 0.0f : d / mx;
    *v = mx;
}

static __device__ __forceinline__ float3 hue_rgb(float hue) {
    float h = rem_euclid(hue, 360.0f) / 60.0f;
    float x = 1.0f - fabsf(fmodf(h, 2.0f) - 1.0f);
    int sector = (int)h;
    switch (sector) {
        case 0: return make_float3(1.0f, x, 0.0f);
        case 1: return make_float3(x, 1.0f, 0.0f);
        case 2: return make_float3(0.0f, 1.0f, x);
        case 3: return make_float3(0.0f, x, 1.0f);
        case 4: return make_float3(x, 0.0f, 1.0f);
        default: return make_float3(1.0f, 0.0f, x);
    }
}

static __device__ __forceinline__ float3 hsv_to_rgb(float h, float s, float v) {
    float3 base = hue_rgb(h);
    return make_float3(
        v * (1.0f - s + s * base.x),
        v * (1.0f - s + s * base.y),
        v * (1.0f - s + s * base.z));
}

static __device__ __forceinline__ float signed_hue_delta(float h, float center) {
    float d = rem_euclid(h - center, 360.0f);
    return d > 180.0f ? d - 360.0f : d;
}

static __device__ __forceinline__ float hue_distance(float a, float b) {
    float d = rem_euclid(a - b, 360.0f);
    return fminf(d, 360.0f - d);
}

static __device__ __forceinline__ float band_weight(
    const float* b, float extra_feather, float h, float s, float v) {
    float d = signed_hue_delta(h, b[0]);
    float feather = fmaxf(b[9], 0.1f) + extra_feather;
    float hue_w;
    if (d < b[3]) {
        hue_w = sstep(b[3] - feather, b[3], d);
    } else if (d > b[4]) {
        hue_w = 1.0f - sstep(b[4], b[4] + feather, d);
    } else {
        hue_w = 1.0f;
    }
    float ds = (s - b[1]) * 100.0f;
    float sat_feather = fmaxf(10.0f, b[9]);
    float sat_w;
    if (ds < b[5]) {
        sat_w = sstep(b[5] - sat_feather, b[5], ds);
    } else if (ds > b[6]) {
        sat_w = 1.0f - sstep(b[6], b[6] + sat_feather, ds);
    } else {
        sat_w = 1.0f;
    }
    float vv = clampf(v, 0.0f, 1.0f);
    float value_w;
    if (vv < b[7]) {
        value_w = sstep(b[7] - 0.05f, b[7], vv);
    } else if (vv > b[8]) {
        value_w = 1.0f - sstep(b[8], b[8] + 0.05f, vv);
    } else {
        value_w = 1.0f;
    }
    return clampf(hue_w * sat_w * value_w, 0.0f, 1.0f);
}

static __device__ __forceinline__ float color_range_weight(
    const float* band, float hue, float width, float smoothness, float min_saturation,
    float3 c) {
    float h, s, v;
    rgb_to_hsv(c, &h, &s, &v);
    float half = width / 2.0f;
    float d = hue_distance(h, hue);
    float hue_w = 1.0f - sstep(half, half + fmaxf(smoothness, 0.1f), d);
    return hue_w * sstep(min_saturation * 0.5f, min_saturation + 0.05f, s);
}

static __device__ __forceinline__ void apply_homogeneity(
    const float* band, const float* homogeneity, float w, float* h, float* s, float* v) {
    float hue = homogeneity[0] / 100.0f * w;
    if (hue != 0.0f) {
        *h += signed_hue_delta(band[0], *h) * hue;
    }
    float sat = homogeneity[1] / 100.0f * w;
    if (sat != 0.0f) {
        *s += (band[1] - *s) * sat;
    }
    float light = homogeneity[2] / 100.0f * w;
    if (light != 0.0f) {
        *v += (band[2] - *v) * light;
    }
}

static __device__ __forceinline__ float3 color_editor(float3 g, const float* P, const float* E) {
    float h0, s0, v0;
    rgb_to_hsv(make_float3(fmaxf(g.x, 0.0f), fmaxf(g.y, 0.0f), fmaxf(g.z, 0.0f)),
               &h0, &s0, &v0);
    if (s0 < 0.02f) {
        return g;
    }
    float extra = fmaxf(E[E_BASIC_EXTRA], 0.0f);
    float h = h0;
    float s = s0;
    float v = v0;
    float dh = 0.0f;
    float ds = 0.0f;
    float dl = 0.0f;

    if (E[E_GLOBAL_ENABLED] != 0.0f) {
        const float* band = E + E_GLOBAL_BAND;
        float w = band_weight(band, 0.0f, h0, s0, v0);
        if (band[10] != 0.0f) w = 1.0f - w;
        dh += E[E_GLOBAL_SHIFT + 0] * w;
        ds += E[E_GLOBAL_SHIFT + 1] * w;
        dl += E[E_GLOBAL_SHIFT + 2] * w;
    }

    int n_skin = (int)P[P_N_SKIN];
    for (int i = 0; i < n_skin; ++i) {
        const float* e = E + E_SKIN_BASE + i * E_SKIN_STRIDE;
        int mode = (int)e[0];
        if (mode == 1) {
            const float* band = e + 1;
            float w = band_weight(band, 0.0f, h0, s0, v0);
            if (band[10] != 0.0f) w = 1.0f - w;
            apply_homogeneity(band, e + 13, w, &h, &s, &v);
        } else if (mode == 2) {
            float w = color_range_weight(e + 16, e[16], e[17], e[18], e[19], g);
            h += signed_hue_delta(e[16], h) * (e[20] / 100.0f * w);
            if (e[21] > 0.0f) {
                float target_s = clampf(e[19] + 0.35f, 0.15f, 0.55f);
                s += (target_s - s) * (e[21] / 100.0f * w);
            }
            if (e[22] > 0.0f) {
                float target_v = 0.65f;
                v += (target_v - v) * (e[22] / 100.0f * w * 0.5f);
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        const float* e = E + E_BASIC + i * E_BASIC_STRIDE;
        if (e[15] != 0.0f) {
            continue;
        }
        const float* band = e;
        float w = band_weight(band, extra, h, s, v);
        if (band[10] != 0.0f) w = 1.0f - w;
        dh += e[12] * w;
        ds += e[13] * w;
        dl += e[14] * w;
    }

    int n_shift = (int)P[P_N_SHIFT];
    for (int i = 0; i < n_shift; ++i) {
        const float* e = E + E_SKIN_BASE + n_skin * E_SKIN_STRIDE + i * E_SHIFT_STRIDE;
        int mode = (int)e[0];
        if (mode == 1) {
            continue;
        }
        float w;
        if (mode == 0) {
            const float* band = e + 2;
            w = band_weight(band, e[1], h0, s0, v0);
            if (band[10] != 0.0f) w = 1.0f - w;
        } else {
            w = color_range_weight(e + 17, e[17], e[18], e[19], e[20], g);
        }
        dh += e[14] * w;
        ds += e[15] * w;
        dl += e[16] * w;
    }

    dh = clampf(dh, -180.0f, 180.0f);
    ds = clampf(ds, -100.0f, 100.0f);
    dl = clampf(dl, -100.0f, 100.0f);
    float s2 = clampf(s * (1.0f + ds / 100.0f), 0.0f, 1.0f);
    float v2 = fmaxf(v * (1.0f + dl / 100.0f * 0.5f * s), 0.0f);
    return hsv_to_rgb(h + dh, s2, v2);
}

// Output banding -----------------------------------------------------------
static __device__ __forceinline__ float3 output_banding(
    float3 rgb, const float* P, const unsigned int* words,
    const unsigned char* left, const unsigned char* right, float fx, float fy) {
    if (P[P_BANDING] == 0.0f || words == 0 || left == 0 || right == 0) {
        return rgb;
    }
    int gain = (int)P[P_BG_GAIN];
    int limit = (int)P[P_BG_LIMIT];
    int strength = (int)P[P_BG_STRENGTH];
    if (limit <= 0 || strength <= 0) {
        return rgb;
    }
    int x = (int)(fx * P[P_BG_FW]);
    int y = (int)(fy * P[P_BG_FH]);
    int cell = ((((x & ~1) & 0x0f) << 4) | (y & 0x0f));
    int block = (((x & 0xf0) >> 4) | (y & 0xf0));
    unsigned int pair = words[(block + (int)left[cell]) & 255]
        - words[(block + (int)right[cell]) & 255];
    unsigned int half = (x & 1) != 0 ? (pair >> 16) : (pair & 0xffffu);
    int high = (((int)(half >> 8)) * gain) >> 16;
    int low = (((int)(half & 255u)) * gain) >> 16;
    float comps[3] = {rgb.x, rgb.y, rgb.z};
    float out[3];
    for (int c = 0; c < 3; ++c) {
        int value = (int)(clampf(comps[c], 0.0f, 1.0f) * 65535.0f + 0.5f);
        unsigned short code = (unsigned short)value;
        int headroom = min(limit, 65535 - (int)code);
        int adjusted = (int)code + (((high - low) * headroom * strength) >> 16);
        if (adjusted < 0) adjusted = 0;
        if (adjusted > 65535) adjusted = 65535;
        out[c] = (float)adjusted / 65535.0f;
    }
    return make_float3(out[0], out[1], out[2]);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
extern "C" __global__ void develop_kernel(
    float4* io,
    const float* P,
    const float* luts,
    const float* gain,
    const float* c1,
    const float* editor,
    const unsigned int* band_words,
    const unsigned char* band_left,
    const unsigned char* band_right) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = (int)P[P_W];
    int h = (int)P[P_H];
    if (x >= w || y >= h) {
        return;
    }
    int index = y * w + x;
    float4 sample = io[index];
    if (sample.w <= 0.5f) {
        io[index] = make_float4(0.11f, 0.11f, 0.11f, 1.0f);
        return;
    }
    float fx = P[P_RX] + ((float)x + 0.5f) / (float)w * P[P_RW];
    float fy = P[P_RY] + ((float)y + 0.5f) / (float)h * P[P_RH];
    float3 lin = make_float3(sample.x, sample.y, sample.z);

    float3 display;
    if (P[P_BYPASS] != 0.0f) {
        if (P[P_C1] != 0.0f) {
            display = c1_develop(P, c1, lin);
            display = output_banding(display, P, band_words, band_left, band_right, fx, fy);
        } else {
            display = make_float3(
                encode_srgb(P, luts, lin.x),
                encode_srgb(P, luts, lin.y),
                encode_srgb(P, luts, lin.z));
        }
    } else {
        float3 c = linear_edits(lin, P, gain, index, fx, fy);
        if (P[P_C1] != 0.0f) {
            display = c1_develop(P, c1, c);
        } else {
            display = make_float3(
                encode_srgb(P, luts, c.x),
                encode_srgb(P, luts, c.y),
                encode_srgb(P, luts, c.z));
        }
    }

    // develop_display() ------------------------------------------------------
    if (P[P_BRIGHT_GAMMA] != 1.0f) {
        float gamma = P[P_BRIGHT_GAMMA];
        display.x = powf(fmaxf(display.x, 0.0f), gamma);
        display.y = powf(fmaxf(display.y, 0.0f), gamma);
        display.z = powf(fmaxf(display.z, 0.0f), gamma);
    }
    float contrast_amount = P[P_CONTRAST];  // already `contrast/50*0.8`
    display.x = tone_contrast(display.x, contrast_amount);
    display.y = tone_contrast(display.y, contrast_amount);
    display.z = tone_contrast(display.z, contrast_amount);

    if (P[P_LEVELS] != 0.0f) {
        display.x = levels_input(display.x, P[P_LV_LO + 0], P[P_LV_SPAN + 0], P[P_LV_INV + 0]);
        display.y = levels_input(display.y, P[P_LV_LO + 1], P[P_LV_SPAN + 1], P[P_LV_INV + 1]);
        display.z = levels_input(display.z, P[P_LV_LO + 2], P[P_LV_SPAN + 2], P[P_LV_INV + 2]);
        display.x = levels_input(display.x, P[P_LV_MLO + 0], P[P_LV_MSPAN + 0], P[P_LV_MINV + 0]);
        display.y = levels_input(display.y, P[P_LV_MLO + 1], P[P_LV_MSPAN + 1], P[P_LV_MINV + 1]);
        display.z = levels_input(display.z, P[P_LV_MLO + 2], P[P_LV_MSPAN + 2], P[P_LV_MINV + 2]);
    }
    if (P[P_CURVE + 0] != 0.0f) {
        const float* lut = luts + LUT_RGB;
        display.x = lut_get(lut, display.x);
        display.y = lut_get(lut, display.y);
        display.z = lut_get(lut, display.z);
    }
    if (P[P_CURVE + 1] != 0.0f) display.x = lut_get(luts + LUT_R, display.x);
    if (P[P_CURVE + 2] != 0.0f) display.y = lut_get(luts + LUT_G, display.y);
    if (P[P_CURVE + 3] != 0.0f) display.z = lut_get(luts + LUT_B, display.z);
    if (P[P_CURVE + 4] != 0.0f) {
        float l = fmaxf(luma601(display.x, display.y, display.z), 1e-4f);
        float k = lut_get(luts + LUT_LUMA, l) / l;
        display.x *= k;
        display.y *= k;
        display.z *= k;
    }
    if (P[P_LEVELS] != 0.0f) {
        float gain_lv = P[P_LV_OUT_GAIN];
        float shadow = P[P_LV_OUT_SHADOW];
        display.x = display.x * gain_lv + shadow;
        display.y = display.y * gain_lv + shadow;
        display.z = display.z * gain_lv + shadow;
    }

    float sat = P[P_SAT];
    if (fabsf(sat - 1.0f) > 1e-4f) {
        float l = luma709(display.x, display.y, display.z);
        display.x = l + (display.x - l) * sat;
        display.y = l + (display.y - l) * sat;
        display.z = l + (display.z - l) * sat;
    }
    if (P[P_CB] != 0.0f) {
        display = color_balance(display, P);
    }
    if (P[P_CBG] != 0.0f) {
        display = color_balance_gains(display, P);
    }
    if (P[P_CE] != 0.0f) {
        display = color_editor(display, P, editor);
    }
    if (P[P_BW] != 0.0f) {
        display = black_white(display, P);
    }

    if (P[P_BANDING_OUTER] != 0.0f) {
        display = output_banding(display, P, band_words, band_left, band_right, fx, fy);
    }
    io[index] = make_float4(display.x, display.y, display.z, 1.0f);
}
