// CUDA twins of the production display-sRGB f32 noise stages:
// `stage_luminance_nr_f32` (C1 LumaNoise2018) and `stage_color_nr_f32`
// (C1 ColorNoise2012 / LABRangeFilter2012).
//
// Compile with -fmad=false: the CPU reference is plain f32 without fused
// multiply-add, so the same expression order has to survive codegen.

#include <cuda_runtime.h>

__device__ __forceinline__ float luma255(float3 p) {
    return (0.2126f * p.x + 0.7152f * p.y + 0.0722f * p.z) * 255.0f;
}

__device__ __forceinline__ float clamp01(float v) {
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

// Replicated-border luminance plane, identical to `padded_luma_f32`.
extern "C" __global__ void nr_luma_pad_kernel(
    const float3* rgb, float* pad, int w, int h
) {
    int pw = w + 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= h + 2) return;
    int sx = min(max(x - 1, 0), w - 1);
    int sy = min(max(y - 1, 0), h - 1);
    pad[y * pw + x] = luma255(rgb[sy * w + sx]);
}

// Radius-1 bilateral luma NR, identical to `stage_luminance_nr_f32`.
extern "C" __global__ void nr_luma_apply_kernel(
    float3* rgb, const float* pad, const float* lut, int w, int h, float strength
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int OFF[8][2] = {
        {-1, -1}, {0, -1}, {1, -1}, {-1, 0},
        {1, 0}, {-1, 1}, {0, 1}, {1, 1}
    };
    const float SPATIAL[8] = {
        1.0f / 3.0f, 0.5f, 1.0f / 3.0f, 0.5f,
        0.5f, 1.0f / 3.0f, 0.5f, 1.0f / 3.0f
    };
    int pw = w + 2;
    float c = pad[(y + 1) * pw + x + 1];
    float num = c;
    float den = 1.0f;
    for (int k = 0; k < 8; k++) {
        float l = pad[(y + 1 + OFF[k][1]) * pw + (x + 1 + OFF[k][0])];
        float ad = fabsf(l - c) * 4.0f;
        int i0 = min((int)ad, 1023);
        float t = ad - (float)i0;
        float range = lut[i0] + (lut[i0 + 1] - lut[i0]) * t;
        float wt = range * SPATIAL[k];
        num += wt * l;
        den += wt;
    }
    float target = den > 1e-6f ? num / den : c;
    target = c + (target - c) * strength;
    float new_l = fminf(fmaxf(target, 0.0f), 255.0f);
    if (fabsf(new_l - c) >= 1e-3f) {
        float f = c > 1.0f ? new_l / c : 1.0f;
        int i = y * w + x;
        float3 p = rgb[i];
        p.x = clamp01(p.x * f);
        p.y = clamp01(p.y * f);
        p.z = clamp01(p.z * f);
        rgb[i] = p;
    }
}

__device__ __forceinline__ float srgb_to_linear(float v) {
    v = clamp01(v);
    return v <= 0.04045f ? v / 12.92f : powf((v + 0.055f) / 1.055f, 2.4f);
}

extern "C" __global__ void nr_color_encode_kernel(
    const float3* rgb, float* pl, float* pu, float* pv, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float3 p = rgb[i];
    float r = srgb_to_linear(p.x) * 65535.0f;
    float g = srgb_to_linear(p.y) * 65535.0f;
    float b = srgb_to_linear(p.z) * 65535.0f;
    float l = fmaxf(0.299f * r + 0.587f * g + 0.114f * b, 0.0f) + 35.759254f;
    float inv_l = 1.0f / l;
    pl[i] = l;
    pu[i] = (r - g) * inv_l;
    pv[i] = (b - g) * inv_l;
}

// Horizontal range-filter pass. `L` is unchanged, so only U/V are rewritten.
extern "C" __global__ void nr_color_h_kernel(
    const float* pl, const float* pu, const float* pv,
    float* hpu, float* hpv, int w, int h,
    int radius, int step, float pz, float pw, float pz_chroma, float pw_chroma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int row = y * w;
    int i = row + x;
    float lc = pl[i];
    float uc = pu[i];
    float vc = pv[i];
    float t_root = fmaxf(pz + pw * lc, 1.0f);
    float t_l2 = t_root * t_root;
    float t_c = fmaxf(pz_chroma + pw_chroma * lc, 1.0f);
    float u_lc = uc * lc;
    float v_lc = vc * lc;
    float sum_u = 0.0f;
    float sum_v = 0.0f;
    float sum_w = 0.0f;
    for (int off = -radius; off <= radius; off += step) {
        int nx = max(0, min(x + off, w - 1));
        int j = row + nx;
        float li = pl[j];
        float ui = pu[j];
        float vi = pv[j];
        float dl = lc - li;
        float du = u_lc - ui * li;
        float dv = v_lc - vi * li;
        float dl2 = dl * dl;
        float dc = fmaxf(0.17887762f * du * du, 0.3776723f * dv * dv);
        float w_l = t_l2 / (dl2 * dl2 + t_l2);
        float w_c = t_c / (dc + t_c);
        float wt = fmaxf(w_l * w_c - 0.01f, 0.0f);
        sum_u += ui * wt;
        sum_v += vi * wt;
        sum_w += wt;
    }
    if (sum_w > 1e-6f) {
        float inv_w = 1.0f / sum_w;
        hpu[i] = sum_u * inv_w;
        hpv[i] = sum_v * inv_w;
    } else {
        hpu[i] = uc;
        hpv[i] = vc;
    }
}

__device__ __forceinline__ float linear_to_srgb(float v) {
    return v <= 0.0031308f ? v * 12.92f : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
}

// Vertical range-filter pass plus linear-RGB reconstruction.
extern "C" __global__ void nr_color_v_kernel(
    const float* pl, const float* hpu, const float* hpv, float3* rgb,
    int w, int h, int radius, int step,
    float pz, float pw, float pz_chroma, float pw_chroma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    float lc = pl[i];
    float uc = hpu[i];
    float vc = hpv[i];
    float t_root = fmaxf(pz + pw * lc, 1.0f);
    float t_l2 = t_root * t_root;
    float t_c = fmaxf(pz_chroma + pw_chroma * lc, 1.0f);
    float u_lc = uc * lc;
    float v_lc = vc * lc;
    float sum_u = 0.0f;
    float sum_v = 0.0f;
    float sum_w = 0.0f;
    for (int off = -radius; off <= radius; off += step) {
        int ny = max(0, min(y + off, h - 1));
        int j = ny * w + x;
        float li = pl[j];
        float ui = hpu[j];
        float vi = hpv[j];
        float dl = lc - li;
        float du = u_lc - ui * li;
        float dv = v_lc - vi * li;
        float dl2 = dl * dl;
        float dc = fmaxf(0.17887762f * du * du, 0.3776723f * dv * dv);
        float w_l = t_l2 / (dl2 * dl2 + t_l2);
        float w_c = t_c / (dc + t_c);
        float wt = fmaxf(w_l * w_c - 0.01f, 0.0f);
        sum_u += ui * wt;
        sum_v += vi * wt;
        sum_w += wt;
    }
    float u = uc;
    float v = vc;
    if (sum_w > 1e-6f) {
        float inv_w = 1.0f / sum_w;
        u = sum_u * inv_w;
        v = sum_v * inv_w;
    }
    float ul = u * lc;
    float vl = v * lc;
    float g = lc - ul * 0.299f - vl * 0.114f - 35.759254f;
    float r0 = fmaxf(ul + g, 1.0f);
    float g0 = fmaxf(g, 1.0f);
    float b0 = fmaxf(vl + g, 1.0f);
    float sc = fminf(
        (lc - (35.759254f - 1.0f))
            / (fmaxf(0.299f * r0 + 0.587f * g0 + 0.114f * b0, 0.0f) + 1.0f),
        1.5f
    );
    float3 p;
    p.x = linear_to_srgb(clamp01(r0 * sc / 65535.0f));
    p.y = linear_to_srgb(clamp01(g0 * sc / 65535.0f));
    p.z = linear_to_srgb(clamp01(b0 * sc / 65535.0f));
    rgb[i] = p;
}
