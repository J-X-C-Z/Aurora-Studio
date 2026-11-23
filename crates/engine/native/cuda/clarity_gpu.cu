// CUDA twins of the CPU Clarity/Structure kernels in `src/mock/clarity.rs`.
// Compile with -fmad=false so the float order matches the CPU reference.

#include <cuda_runtime.h>

__device__ __forceinline__ float luma601f(float r, float g, float b) {
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/// Bilinear sample of a grid plane at output pixel `(x, y)`.
/// Mirrors `clarity::PreCurve::sample` / the full-resolution grid lookup.
__device__ __forceinline__ float sample_plane(
    const float* plane, int gw, int gh, float step, int x, int y
) {
    float fx = clampf(((float)x + 0.5f) / step - 0.5f, 0.0f, (float)(gw - 1));
    float fy = clampf(((float)y + 0.5f) / step - 0.5f, 0.0f, (float)(gh - 1));
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = x0 + 1 < gw - 1 ? x0 + 1 : gw - 1;
    int y1 = y0 + 1 < gh - 1 ? y0 + 1 : gh - 1;
    float tx = fx - (float)x0;
    float ty = fy - (float)y0;
    float a = plane[y0 * gw + x0] * (1.0f - tx) + plane[y0 * gw + x1] * tx;
    float b = plane[y1 * gw + x0] * (1.0f - tx) + plane[y1 * gw + x1] * tx;
    return a * (1.0f - ty) + b * ty;
}

/// `(p + d) * s`, then C1's Punch chroma push around the result's luma.
__device__ __forceinline__ void shade3(
    float in_r, float in_g, float in_b, float d, float s, float chroma,
    float* out_r, float* out_g, float* out_b
) {
    float r = (in_r + d) * s;
    float g = (in_g + d) * s;
    float b = (in_b + d) * s;
    if (chroma != 1.0f) {
        float l = luma601f(r, g, b);
        r = l + (r - l) * chroma;
        g = l + (g - l) * chroma;
        b = l + (b - l) * chroma;
    }
    *out_r = r;
    *out_g = g;
    *out_b = b;
}

/// Block average into `cols x rows` cells, matching `clarity::plane_low`.
/// Runs one thread per output cell; the source grid is small enough that the
/// serial inner loop is cheaper than a second reduction pass.
extern "C" __global__ void clarity_blockavg_kernel(
    const float* src, float* small, int gw, int gh, int cols, int rows,
    int compensate
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cols) return;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= rows) return;
    int y0 = j * gh / rows;
    int y1 = (j + 1) * gh / rows;
    if (y1 <= y0) y1 = y0 + 1;
    if (y1 > gh) y1 = gh;
    int x0 = i * gw / cols;
    int x1 = (i + 1) * gw / cols;
    if (x1 <= x0) x1 = x0 + 1;
    if (x1 > gw) x1 = gw;
    // The CPU sums in f64. A plain f32 sum over a full-resolution cell drifts
    // enough to move 16-bit output by tens of codes, so 16-bit and pre-curve
    // grids use Kahan; the 8-bit path stays on the plain sum, which already
    // quantises to the same byte and is faster.
    float sum = 0.0f;
    float carry = 0.0f;
    int n = 0;
    if (compensate == 2) {
        // Exact match for the CPU's f64 block sum; used where a 16-bit output
        // must not cross a code boundary.
        double exact = 0.0;
        for (int yy = y0; yy < y1; yy++) {
            for (int xx = x0; xx < x1; xx++) {
                exact += (double)src[yy * gw + xx];
                n++;
            }
        }
        small[j * cols + i] = (float)(exact / (double)n);
        return;
    }
    if (compensate != 0) {
        for (int yy = y0; yy < y1; yy++) {
            for (int xx = x0; xx < x1; xx++) {
                float value = src[yy * gw + xx] - carry;
                float next = sum + value;
                carry = (next - sum) - value;
                sum = next;
                n++;
            }
        }
    } else {
        for (int yy = y0; yy < y1; yy++) {
            for (int xx = x0; xx < x1; xx++) {
                sum += src[yy * gw + xx];
                n++;
            }
        }
    }
    small[j * cols + i] = sum / (float)n;
}

/// Bilinear upsample of the cell plane back to `gw x gh`.
extern "C" __global__ void clarity_upsample_kernel(
    const float* small, float* out, int gw, int gh, int cols, int rows
) {
    int xx = blockIdx.x * blockDim.x + threadIdx.x;
    if (xx >= gw) return;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    if (yy >= gh) return;
    float fy = clampf(
        ((float)yy + 0.5f) * (float)rows / (float)gh - 0.5f, 0.0f, (float)(rows - 1));
    int j0 = (int)floorf(fy);
    int j1 = j0 + 1 < rows - 1 ? j0 + 1 : rows - 1;
    float ty = fy - (float)j0;
    float fx = clampf(
        ((float)xx + 0.5f) * (float)cols / (float)gw - 0.5f, 0.0f, (float)(cols - 1));
    int i0 = (int)floorf(fx);
    int i1 = i0 + 1 < cols - 1 ? i0 + 1 : cols - 1;
    float tx = fx - (float)i0;
    float a = small[j0 * cols + i0] * (1.0f - tx) + small[j0 * cols + i1] * tx;
    float b = small[j1 * cols + i0] * (1.0f - tx) + small[j1 * cols + i1] * tx;
    out[yy * gw + xx] = a * (1.0f - ty) + b * ty;
}

/// Clarity gain on a pre-curve f32 RGB frame (`Vec<[f32; 3]>` layout).
extern "C" __global__ void clarity_gain_f32_kernel(
    float* rgb, const float* fine, const float* coarse,
    int w, int h, int gw, int gh, float step,
    float k_big, float k_fine, float chroma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= w) return;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= h) return;
    int i = y * w + x;
    float r = rgb[i * 3 + 0];
    float g = rgb[i * 3 + 1];
    float b = rgb[i * 3 + 2];
    float l = luma601f(r, g, b);
    float big = sample_plane(coarse, gw, gh, step, x, y);
    float small = sample_plane(fine, gw, gh, step, x, y);
    float d = k_big * (l - big) + k_fine * (l - small);
    if (d < 0.0f) d *= 0.6f;
    shade3(r, g, b, d, 1.0f, chroma,
           &rgb[i * 3 + 0], &rgb[i * 3 + 1], &rgb[i * 3 + 2]);
}

/// Clarity gain on an 8-bit RGBA frame.
extern "C" __global__ void clarity_gain_rgba8_kernel(
    uchar4* rgba, const float* fine, const float* coarse,
    int w, int h, int gw, int gh, float step,
    float k_big, float k_fine, float chroma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= w) return;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= h) return;
    int i = y * w + x;
    uchar4 p = rgba[i];
    float r = (float)p.x * (1.0f / 255.0f);
    float g = (float)p.y * (1.0f / 255.0f);
    float b = (float)p.z * (1.0f / 255.0f);
    float l = luma601f(r, g, b);
    float big = sample_plane(coarse, gw, gh, step, x, y);
    float small = sample_plane(fine, gw, gh, step, x, y);
    float d = k_big * (l - big) + k_fine * (l - small);
    if (d < 0.0f) d *= 0.6f;
    float qr, qg, qb;
    shade3(r, g, b, d, 1.0f, chroma, &qr, &qg, &qb);
    p.x = (unsigned char)(clampf(qr, 0.0f, 1.0f) * 255.0f + 0.5f);
    p.y = (unsigned char)(clampf(qg, 0.0f, 1.0f) * 255.0f + 0.5f);
    p.z = (unsigned char)(clampf(qb, 0.0f, 1.0f) * 255.0f + 0.5f);
    rgba[i] = p;
}

/// Clarity gain on a 16-bit linear RGBA frame; keeps the 8-bit preview twin in
/// step, exactly like `clarity::shade_frame` does for Structure.
extern "C" __global__ void clarity_gain_rgba16_kernel(
    ushort4* rgba16, uchar4* rgba8, const float* fine, const float* coarse,
    int w, int h, int gw, int gh, float step,
    float k_big, float k_fine, float chroma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= w) return;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= h) return;
    int i = y * w + x;
    ushort4 p16 = rgba16[i];
    float r = (float)p16.x * (1.0f / 65535.0f);
    float g = (float)p16.y * (1.0f / 65535.0f);
    float b = (float)p16.z * (1.0f / 65535.0f);
    float l = luma601f(r, g, b);
    float big = sample_plane(coarse, gw, gh, step, x, y);
    float small = sample_plane(fine, gw, gh, step, x, y);
    float d = k_big * (l - big) + k_fine * (l - small);
    if (d < 0.0f) d *= 0.6f;
    float qr, qg, qb;
    shade3(r, g, b, d, 1.0f, chroma, &qr, &qg, &qb);
    float vr = clampf(qr, 0.0f, 1.0f);
    float vg = clampf(qg, 0.0f, 1.0f);
    float vb = clampf(qb, 0.0f, 1.0f);
    p16.x = (unsigned short)(vr * 65535.0f + 0.5f);
    p16.y = (unsigned short)(vg * 65535.0f + 0.5f);
    p16.z = (unsigned short)(vb * 65535.0f + 0.5f);
    rgba16[i] = p16;
    uchar4 p8 = rgba8[i];
    p8.x = (unsigned char)(vr * 255.0f + 0.5f);
    p8.y = (unsigned char)(vg * 255.0f + 0.5f);
    p8.z = (unsigned char)(vb * 255.0f + 0.5f);
    rgba8[i] = p8;
}

/// Luma of an 8-bit RGBA frame, in 0..1.
extern "C" __global__ void clarity_luma8_kernel(
    const uchar4* rgba, float* luma, int w, int h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= w) return;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= h) return;
    uchar4 p = rgba[y * w + x];
    luma[y * w + x] =
        luma601f((float)p.x, (float)p.y, (float)p.z) * (1.0f / 255.0f);
}

/// Luma of a 16-bit linear RGBA frame, in 0..1.
extern "C" __global__ void clarity_luma16_kernel(
    const ushort4* rgba, float* luma, int w, int h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= w) return;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= h) return;
    ushort4 p = rgba[y * w + x];
    luma[y * w + x] =
        luma601f((float)p.x, (float)p.y, (float)p.z) * (1.0f / 65535.0f);
}

/// Horizontal separable box blur with clamped edges; mirrors `clarity::box_blur`.
extern "C" __global__ void clarity_box_h_kernel(
    const float* src, float* dst, int w, int h, int r
) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= h) return;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= w) return;
    if (r == 0) {
        dst[y * w + x] = src[y * w + x];
        return;
    }
    float inv = 1.0f / (float)(2 * r + 1);
    float acc = 0.0f;
    const float* row = src + y * w;
    for (int k = -r; k <= r; k++) {
        acc += row[clampi(x + k, 0, w - 1)];
    }
    dst[y * w + x] = acc * inv;
}

/// Vertical separable box blur with clamped edges; mirrors `clarity::box_blur`.
extern "C" __global__ void clarity_box_v_kernel(
    const float* src, float* dst, int w, int h, int r
) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= h) return;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= w) return;
    if (r == 0) {
        dst[y * w + x] = src[y * w + x];
        return;
    }
    float inv = 1.0f / (float)(2 * r + 1);
    float acc = 0.0f;
    for (int k = -r; k <= r; k++) {
        int yy = clampi(y + k, 0, h - 1);
        acc += src[yy * w + x];
    }
    dst[y * w + x] = acc * inv;
}

/// Structure gain on an 8-bit RGBA frame; mirrors `clarity::shade_frame`.
extern "C" __global__ void clarity_structure8_kernel(
    uchar4* rgba, const float* luma, const float* low, int w, int h, float k
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= w) return;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= h) return;
    int i = y * w + x;
    float t = 1.0f + (luma[i] - low[i]) * (k / fmaxf(low[i], 1e-3f));
    if (t < 0.0f) t = 0.0f;
    float s = t * t;
    uchar4 p = rgba[i];
    p.x = (unsigned char)(
        clampf((float)p.x * (1.0f / 255.0f) * s, 0.0f, 1.0f) * 255.0f + 0.5f);
    p.y = (unsigned char)(
        clampf((float)p.y * (1.0f / 255.0f) * s, 0.0f, 1.0f) * 255.0f + 0.5f);
    p.z = (unsigned char)(
        clampf((float)p.z * (1.0f / 255.0f) * s, 0.0f, 1.0f) * 255.0f + 0.5f);
    rgba[i] = p;
}

/// Structure gain on a 16-bit linear RGBA frame plus its 8-bit twin.
extern "C" __global__ void clarity_structure16_kernel(
    ushort4* rgba16, uchar4* rgba8, const float* luma, const float* low,
    int w, int h, float k
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= w) return;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= h) return;
    int i = y * w + x;
    float t = 1.0f + (luma[i] - low[i]) * (k / fmaxf(low[i], 1e-3f));
    if (t < 0.0f) t = 0.0f;
    float s = t * t;
    ushort4 p16 = rgba16[i];
    float vr = clampf((float)p16.x * (1.0f / 65535.0f) * s, 0.0f, 1.0f);
    float vg = clampf((float)p16.y * (1.0f / 65535.0f) * s, 0.0f, 1.0f);
    float vb = clampf((float)p16.z * (1.0f / 65535.0f) * s, 0.0f, 1.0f);
    p16.x = (unsigned short)(vr * 65535.0f + 0.5f);
    p16.y = (unsigned short)(vg * 65535.0f + 0.5f);
    p16.z = (unsigned short)(vb * 65535.0f + 0.5f);
    rgba16[i] = p16;
    uchar4 p8 = rgba8[i];
    p8.x = (unsigned char)(vr * 255.0f + 0.5f);
    p8.y = (unsigned char)(vg * 255.0f + 0.5f);
    p8.z = (unsigned char)(vb * 255.0f + 0.5f);
    rgba8[i] = p8;
}
