// CUDA twins of the three light per-pixel operators: manual vignette, the LCC
// Uniform Light gain plane and film grain. Compile with -fmad=false so the
// float order matches the CPU reference in `src/mock/pixel_gpu.rs`.

#include <cuda_runtime.h>

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ unsigned char to_u8(float v) {
    return (unsigned char)fminf(fmaxf(roundf(v), 0.0f), 255.0f);
}

// ───────────────────────── vignette gain plane ─────────────────────────
//
// Mirrors `vignette_gain` in `mock/process.rs`:
//   elliptic : rho2 = 2 * ((x-cx)^2 + (y-cy)^2)
//   circular : rho2 = 4 * ((x-cx)^2*a^2 + (y-cy)^2) / (a^2 + 1)
//   gain     = exp2(clamp(amount * rho2, -3, 3))

extern "C" __global__ void vignette_gain_kernel(
    float* __restrict__ out, int w, int h,
    float origin_x, float span_x, float origin_y, float span_y,
    float amount, float aspect, float cx, float cy, int method
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float fx = origin_x + ((float)x + 0.5f) / (float)w * span_x;
    float fy = origin_y + ((float)y + 0.5f) / (float)h * span_y;
    float dx = fx - cx;
    float dy = fy - cy;
    float a2 = aspect * aspect;
    float rho2;
    if (method == 0) {
        rho2 = 2.0f * (dx * dx + dy * dy);
    } else {
        rho2 = 4.0f * (dx * dx * a2 + dy * dy) / (a2 + 1.0f);
    }
    float ev = clampf(amount * rho2, -3.0f, 3.0f);
    out[y * w + x] = exp2f(ev);
}

// ─────────────────────────── LCC gain plane ────────────────────────────
//
// One texel per source pixel, at integer source coordinates, mirroring
// `UniformLight::source_gain`: node luminance plus the dx/dy slope, bilinear
// across the grid cell, then `exp2((max_level - field) / 8192 * strength)`
// through the soft/hard/uncapped limiter.

extern "C" __global__ void lcc_gain_kernel(
    float* __restrict__ out, int w, int h,
    int columns, int rows, float origin_x, float origin_y,
    float step_x, float step_y, float max_level, float strength, int limit,
    const float* __restrict__ lum, const float* __restrict__ slope_x,
    const float* __restrict__ slope_y
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float px = (float)x - origin_x;
    float py = (float)y - origin_y;
    float qx = px / step_x;
    float qy = py / step_y;
    if (!isfinite(qx) || !isfinite(qy)) {
        out[y * w + x] = 1.0f;
        return;
    }
    float fx = clampf(truncf(qx + 100.0f) - 100.0f, 0.0f, (float)(columns - 2));
    float fy = clampf(truncf(qy + 100.0f) - 100.0f, 0.0f, (float)(rows - 2));
    int ix = (int)fx;
    int iy = (int)fy;
    float tx = qx - (float)ix;
    float ty = qy - (float)iy;
    float node[4];
    int nx[4] = {ix, ix + 1, ix, ix + 1};
    int ny[4] = {iy, iy, iy + 1, iy + 1};
    for (int k = 0; k < 4; k++) {
        int node_index = ny[k] * columns + nx[k];
        float centre_x = (float)nx[k] * step_x;
        float centre_y = (float)ny[k] * step_y;
        node[k] = lum[node_index]
            + slope_x[node_index] * (px - centre_x) / (2.0f * step_x)
            + slope_y[node_index] * (py - centre_y) / (2.0f * step_y);
    }
    float top = node[0] * (1.0f - tx) + node[1] * tx;
    float bottom = node[2] * (1.0f - tx) + node[3] * tx;
    float field = top * (1.0f - ty) + bottom * ty;
    float ev = ((max_level - field) / 8192.0f) * strength;
    float limited = ev;
    if (ev > 3.0f) {
        if (limit == 0) {
            limited = ev > 5.0f ? 4.0f : ev - 0.25f * (ev - 3.0f) * (ev - 3.0f);
        } else if (limit == 1) {
            limited = 3.0f;
        }
    }
    out[y * w + x] = exp2f(limited);
}

// ───────────────────────────── film grain ──────────────────────────────
//
// The integer hash family is copied from `mock/grain.rs`, which is itself the
// recovered `grains.cl` sequence. Everything is wrapping u32 arithmetic.

__device__ __forceinline__ unsigned int g_mix(unsigned int x) {
    x = x - (x << 6);
    x ^= x >> 17;
    x = x - (x << 9);
    x ^= x << 4;
    x = x - (x << 3);
    x ^= x << 10;
    x ^= x >> 15;
    return x;
}

__device__ __forceinline__ unsigned int g_axis(unsigned int x) {
    x ^= x >> 4;
    x = x * 0x27d4eb2du;
    return x ^ (x >> 15);
}

__device__ __forceinline__ unsigned int g_lcg(unsigned int x) {
    return 69069u * x + 1234567u;
}

__device__ __forceinline__ float gaussian(int x, int y) {
    unsigned int a = g_mix((unsigned int)x + g_axis((unsigned int)y));
    unsigned int b = g_mix((unsigned int)y + g_lcg((unsigned int)x));
    int sum = 0;
    for (int round = 0; round < 2; round++) {
        a = 36969u * (a & 0xFFFFu) + (a >> 16);
        b = 18000u * (b & 0xFFFFu) + (b >> 16);
        unsigned int packed = (a << 16) + b;
        sum += (int)(packed & 0x3FFu);
        sum += (int)((packed >> 10) & 0x3FFu);
        sum += (int)((packed >> 20) & 0x3FFu);
    }
    sum -= 6 << 9;
    sum *= 90;
    sum += (int)(a & 0x3Fu) - 32;
    return (float)sum * (1.0f / 65536.0f);
}

extern "C" __global__ void grain_plane_kernel(
    float* __restrict__ plane, int w, int h, int ox, int oy
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    plane[y * w + x] = gaussian(x + ox, y + oy);
}

// v1: the image is multiplied by the raw noise plane (no visibility weight).
extern "C" __global__ void grain_v1_kernel(
    uchar4* rgba, int w, int h, float gain, int ox, int oy
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    float n = gaussian(x + ox, y + oy);
    float k = 1.0f + n * gain;
    uchar4 p = rgba[i];
    p.x = to_u8((float)p.x * k);
    p.y = to_u8((float)p.y * k);
    p.z = to_u8((float)p.z * k);
    rgba[i] = p;
}

extern "C" __global__ void grain_blur_h_kernel(
    const float* __restrict__ in, float* __restrict__ out,
    int w, int h, int r, const float* __restrict__ taps
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float acc = 0.0f;
    for (int k = 0; k <= 2 * r; k++) {
        int sx = clampi(x + k - r, 0, w - 1);
        acc += in[y * w + sx] * taps[k];
    }
    out[y * w + x] = acc;
}

extern "C" __global__ void grain_blur_v_kernel(
    const float* __restrict__ in, float* __restrict__ out,
    int w, int h, int r, const float* __restrict__ taps
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float acc = 0.0f;
    for (int k = 0; k <= 2 * r; k++) {
        int sy = clampi(y + k - r, 0, h - 1);
        acc += in[sy * w + x] * taps[k];
    }
    out[y * w + x] = acc;
}

// Deterministic plane statistics for the v8 sigma normalisation. Each row is
// reduced by one block in a fixed tree, then the row totals are reduced in a
// single block, so the same frame always lands on the same deviation.
extern "C" __global__ void grain_row_stats_kernel(
    const float* __restrict__ in, float* __restrict__ row_sum,
    float* __restrict__ row_sq, int w, int h
) {
    int y = blockIdx.x;
    if (y >= h) return;
    int t = threadIdx.x;
    float s = 0.0f;
    float q = 0.0f;
    for (int x = t; x < w; x += blockDim.x) {
        float v = in[y * w + x];
        s += v;
        q += v * v;
    }
    __shared__ float sh_s[256];
    __shared__ float sh_q[256];
    sh_s[t] = s;
    sh_q[t] = q;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) {
            sh_s[t] += sh_s[t + stride];
            sh_q[t] += sh_q[t + stride];
        }
        __syncthreads();
    }
    if (t == 0) {
        row_sum[y] = sh_s[0];
        row_sq[y] = sh_q[0];
    }
}

extern "C" __global__ void grain_reduce_stats_kernel(
    const float* __restrict__ row_sum, const float* __restrict__ row_sq,
    int h, float inv_n, float* __restrict__ out
) {
    int t = threadIdx.x;
    float s = 0.0f;
    float q = 0.0f;
    for (int y = t; y < h; y += blockDim.x) {
        s += row_sum[y];
        q += row_sq[y];
    }
    __shared__ float sh_s[256];
    __shared__ float sh_q[256];
    sh_s[t] = s;
    sh_q[t] = q;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) {
            sh_s[t] += sh_s[t + stride];
            sh_q[t] += sh_q[t + stride];
        }
        __syncthreads();
    }
    if (t == 0) {
        float mean = sh_s[0] * inv_n;
        float var = sh_q[0] * inv_n - mean * mean;
        out[0] = sqrtf(var < 0.0f ? 0.0f : var);
    }
}

__device__ __constant__ unsigned short V8_LUT[257] = {
    32, 509, 684, 810, 908, 990, 1060, 1120, 1173, 1220, 1262, 1300, 1334, 1364, 1392, 1417,
    1440, 1461, 1479, 1496, 1512, 1526, 1538, 1550, 1560, 1569, 1577, 1584, 1590, 1595, 1599, 1603,
    1606, 1608, 1610, 1611, 1611, 1611, 1611, 1610, 1608, 1606, 1604, 1601, 1598, 1595, 1591, 1587,
    1582, 1578, 1573, 1567, 1562, 1556, 1550, 1544, 1537, 1531, 1524, 1517, 1510, 1503, 1495, 1487,
    1480, 1472, 1464, 1456, 1447, 1439, 1430, 1422, 1413, 1404, 1396, 1387, 1378, 1369, 1359, 1350,
    1341, 1332, 1322, 1313, 1303, 1294, 1284, 1275, 1265, 1256, 1246, 1236, 1227, 1217, 1207, 1197,
    1188, 1178, 1168, 1158, 1149, 1139, 1129, 1119, 1110, 1100, 1090, 1080, 1071, 1061, 1051, 1041,
    1032, 1022, 1012, 1003, 993, 984, 974, 965, 955, 945, 936, 927, 917, 908, 898, 889,
    880, 871, 861, 852, 843, 834, 825, 816, 807, 798, 789, 780, 771, 762, 753, 744,
    736, 727, 718, 710, 701, 693, 684, 676, 667, 659, 651, 642, 634, 626, 618, 610,
    602, 594, 586, 578, 570, 562, 554, 547, 539, 531, 524, 516, 509, 501, 494, 486,
    479, 472, 464, 457, 450, 443, 436, 429, 422, 415, 408, 401, 394, 388, 381, 374,
    368, 361, 354, 348, 341, 335, 329, 322, 316, 310, 304, 298, 292, 285, 279, 273,
    268, 262, 256, 250, 244, 239, 233, 227, 222, 216, 211, 205, 200, 194, 189, 184,
    179, 173, 168, 163, 158, 153, 148, 143, 138, 133, 128, 124, 119, 114, 109, 105,
    100, 96, 91, 87, 82, 78, 73, 69, 65, 60, 56, 52, 48, 44, 40, 36,
    32
};

// v8: weight the blurred plane by the visibility LUT, then multiply.
extern "C" __global__ void grain_v8_kernel(
    uchar4* rgba, const float* __restrict__ plane, int w, int h, float gain
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    uchar4 p = rgba[i];
    unsigned int r = (unsigned int)p.x * 257u;
    unsigned int g = (unsigned int)p.y * 257u;
    unsigned int b = (unsigned int)p.z * 257u;
    unsigned int mx = r > g ? (r > b ? r : b) : (g > b ? g : b);
    unsigned int luma16 = (r + b + 2u * mx) / 4u;
    unsigned int index = luma16 >> 8;
    if (index > 255u) index = 255u;
    float weight = fminf((float)V8_LUT[index] / 1611.0f, 1.0f);
    float k = 1.0f + weight * plane[i] * gain;
    p.x = to_u8((float)p.x * k);
    p.y = to_u8((float)p.y * k);
    p.z = to_u8((float)p.z * k);
    rgba[i] = p;
}
