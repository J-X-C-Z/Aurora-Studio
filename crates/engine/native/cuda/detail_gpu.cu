// CUDA twins of the CPU detail stages in `src/mock/detail.rs`.
// Compile with -fmad=false so the float order matches the CPU reference.

#include <cuda_runtime.h>

__device__ __forceinline__ float luma8(unsigned char r, unsigned char g, unsigned char b) {
    return 0.2126f * (float)r + 0.7152f * (float)g + 0.0722f * (float)b;
}

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ unsigned char to_u8(float v) {
    return (unsigned char)fminf(fmaxf(roundf(v), 0.0f), 255.0f);
}

extern "C" __global__ void luma_plane_kernel(
    const uchar4* rgba, float* luma, int w, int h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    uchar4 p = rgba[y * w + x];
    luma[y * w + x] = luma8(p.x, p.y, p.z);
}

extern "C" __global__ void luminance_nr_kernel(
    uchar4* rgba, const float* luma, const float* lut, int w, int h,
    float strength, float single
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
    int i = y * w + x;
    uchar4 p = rgba[i];
    float c = luma[i];
    float num = 0.0f;
    float den = 0.0f;
    float neighbours[8];
    for (int k = 0; k < 8; k++) {
        int xx = clampi(x + OFF[k][0], 0, w - 1);
        int yy = clampi(y + OFF[k][1], 0, h - 1);
        float l = luma[yy * w + xx];
        neighbours[k] = l;
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
    if (single > 0.0f) {
        bool isolated = true;
        for (int k = 0; k < 8; k++) {
            if (fabsf(neighbours[k] - c) <= 6.0f) {
                isolated = false;
                break;
            }
        }
        if (isolated) {
            float sorted[8];
            for (int k = 0; k < 8; k++) sorted[k] = neighbours[k];
            for (int a = 1; a < 8; a++) {
                float v = sorted[a];
                int b = a - 1;
                while (b >= 0 && sorted[b] > v) {
                    sorted[b + 1] = sorted[b];
                    b--;
                }
                sorted[b + 1] = v;
            }
            float median = (sorted[3] + sorted[4]) * 0.5f;
            target = target * (1.0f - single) + median * single;
        }
    }
    float new_l = fminf(fmaxf(target, 0.0f), 255.0f);
    if (fabsf(new_l - c) >= 0.05f) {
        float f = c > 1.0f ? new_l / c : 1.0f;
        p.x = to_u8((float)p.x * f);
        p.y = to_u8((float)p.y * f);
        p.z = to_u8((float)p.z * f);
        rgba[i] = p;
    }
}

extern "C" __global__ void ycbcr_kernel(
    const uchar4* rgba, float* cb, float* cr, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uchar4 p = rgba[i];
    float yy = 0.2126f * (float)p.x + 0.7152f * (float)p.y + 0.0722f * (float)p.z;
    cb[i] = ((float)p.z - yy) / 1.8556f;
    cr[i] = ((float)p.x - yy) / 1.5748f;
}

extern "C" __global__ void box_h_kernel(
    const float* src, float* dst, int w, int h, int r
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float inv = 1.0f / (float)(2 * r + 1);
    float sum = 0.0f;
    int base = y * w;
    for (int k = -r; k <= r; k++) {
        sum += src[base + clampi(x + k, 0, w - 1)];
    }
    dst[base + x] = sum * inv;
}

extern "C" __global__ void box_v_kernel(
    const float* src, float* dst, int w, int h, int r
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float inv = 1.0f / (float)(2 * r + 1);
    float sum = 0.0f;
    for (int k = -r; k <= r; k++) {
        int yy = clampi(y + k, 0, h - 1);
        sum += src[yy * w + x];
    }
    dst[y * w + x] = sum * inv;
}

extern "C" __global__ void ycbcr_back_kernel(
    uchar4* rgba, const float* cb, const float* cr, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uchar4 p = rgba[i];
    float yy = 0.2126f * (float)p.x + 0.7152f * (float)p.y + 0.0722f * (float)p.z;
    float cbl = cb[i];
    float crl = cr[i];
    float rr = yy + 1.5748f * crl;
    float bb = yy + 1.8556f * cbl;
    float gg = (yy - 0.2126f * rr - 0.0722f * bb) / 0.7152f;
    p.x = to_u8(rr);
    p.y = to_u8(gg);
    p.z = to_u8(bb);
    rgba[i] = p;
}

extern "C" __global__ void moire_kernel(
    uchar4* rgba, const float* luma, int w, int h, float weight
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float low = 0.0f;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int xx = clampi(x + dx, 0, w - 1);
            int yy = clampi(y + dy, 0, h - 1);
            low += luma[yy * w + xx];
        }
    }
    low /= 9.0f;
    int i = y * w + x;
    float c = luma[i];
    float blend = weight * fminf(fmaxf(fabsf(c - low) / 10.0f, 0.0f), 1.0f);
    float target = c + (low - c) * blend;
    if (fabsf(target - c) < 0.05f) return;
    float f = c > 1.0f ? target / c : 1.0f;
    uchar4 p = rgba[i];
    p.x = to_u8((float)p.x * f);
    p.y = to_u8((float)p.y * f);
    p.z = to_u8((float)p.z * f);
    rgba[i] = p;
}

__device__ __forceinline__ float3 sample_bilinear(
    const uchar4* src, int w, int h, float x, float y
) {
    float fx = x - 0.5f;
    float fy = y - 0.5f;
    float x0f = floorf(fx);
    float y0f = floorf(fy);
    float tx = fx - x0f;
    float ty = fy - y0f;
    int x0 = (int)x0f;
    int y0 = (int)y0f;
    float3 acc = make_float3(0.0f, 0.0f, 0.0f);
    for (int dy = 0; dy < 2; dy++) {
        float wy = dy == 0 ? 1.0f - ty : ty;
        for (int dx = 0; dx < 2; dx++) {
            float wx = dx == 0 ? 1.0f - tx : tx;
            int sx = clampi(x0 + dx, 0, w - 1);
            int sy = clampi(y0 + dy, 0, h - 1);
            uchar4 p = src[sy * w + sx];
            float weight = wx * wy;
            acc.x += (float)p.x * weight;
            acc.y += (float)p.y * weight;
            acc.z += (float)p.z * weight;
        }
    }
    return acc;
}

__device__ __forceinline__ float3 remap_lens(
    const uchar4* src, int w, int h, int x, int y,
    float distortion, float falloff, int ca, int hide
) {
    float cx = (float)w * 0.5f;
    float cy = (float)h * 0.5f;
    float dx = ((float)x + 0.5f - cx) / cx;
    float dy = ((float)y + 0.5f - cy) / cy;
    float r2 = dx * dx + dy * dy;
    float radial = 1.0f + 0.30f * distortion * r2;
    float sx = cx + dx * cx * radial;
    float sy = cy + dy * cy * radial;
    bool outside = sx < 0.0f || sy < 0.0f || sx > (float)w || sy > (float)h;
    if (outside && hide) return make_float3(0.0f, 0.0f, 0.0f);
    float3 rgb = sample_bilinear(src, w, h, sx, sy);
    if (ca) {
        float amount = 0.004f * r2;
        float3 red = sample_bilinear(
            src, w, h, cx + (sx - cx) * (1.0f + amount),
            cy + (sy - cy) * (1.0f + amount)
        );
        float3 blue = sample_bilinear(
            src, w, h, cx + (sx - cx) * (1.0f - amount),
            cy + (sy - cy) * (1.0f - amount)
        );
        rgb.x = red.x;
        rgb.z = blue.z;
    }
    if (falloff > 0.0f) {
        float gain = 1.0f + falloff * 0.35f * r2;
        rgb.x *= gain;
        rgb.y *= gain;
        rgb.z *= gain;
    }
    return rgb;
}

__device__ __forceinline__ uchar4 remap_lens_u8(
    const uchar4* src, int w, int h, int x, int y,
    float distortion, float falloff, int ca, int hide
) {
    float3 rgb = remap_lens(src, w, h, x, y, distortion, falloff, ca, hide);
    uchar4 p;
    p.x = to_u8(rgb.x);
    p.y = to_u8(rgb.y);
    p.z = to_u8(rgb.z);
    p.w = 255;
    return p;
}

extern "C" __global__ void copy_rgba_kernel(
    const uchar4* src, uchar4* dst, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

extern "C" __global__ void lens_kernel(
    const uchar4* src, uchar4* dst, int w, int h,
    float distortion, float falloff, int ca, int hide, float purple
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    uchar4 p = remap_lens_u8(src, w, h, x, y, distortion, falloff, ca, hide);
    if (purple > 0.0f) {
        float pn = fminf(fmaxf(
            (fminf((float)p.x, (float)p.z) - (float)p.y) / 255.0f, 0.0f
        ), 1.0f);
        if (pn > 0.0f) {
            float lo = 255.0f;
            float hi = 0.0f;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int xx = clampi(x + dx, 0, w - 1);
                    int yy = clampi(y + dy, 0, h - 1);
                    uchar4 n = remap_lens_u8(src, w, h, xx, yy, distortion, falloff, ca, hide);
                    float l = luma8(n.x, n.y, n.z);
                    lo = fminf(lo, l);
                    hi = fmaxf(hi, l);
                }
            }
            float edge = fminf(fmaxf((hi - lo) / 40.0f, 0.0f), 1.0f);
            float amount = purple * pn * edge;
            if (amount > 0.0f) {
                float gray = luma8(p.x, p.y, p.z);
                p.x = to_u8((float)p.x * (1.0f - amount) + gray * amount);
                p.y = to_u8((float)p.y * (1.0f - amount) + gray * amount);
                p.z = to_u8((float)p.z * (1.0f - amount) + gray * amount);
            }
        }
    }
    dst[y * w + x] = p;
}

extern "C" __global__ void sharpen_kernel(
    uchar4* rgba, const float* luma, const float* blur, int w, int h,
    float gain, float threshold, float halo
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    float detail = luma[i] - blur[i];
    if (fabsf(detail) < threshold) detail = 0.0f;
    float v = luma[i] + detail * gain;
    if (halo > 0.0f) {
        float lo = 3.402823466e+38f;
        float hi = -3.402823466e+38f;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int xx = clampi(x + dx, 0, w - 1);
                int yy = clampi(y + dy, 0, h - 1);
                float l = luma[yy * w + xx];
                lo = fminf(lo, l);
                hi = fmaxf(hi, l);
            }
        }
        float clamped = fminf(fmaxf(v, lo), hi);
        v = v * (1.0f - halo) + clamped * halo;
    }
    float target = fminf(fmaxf(v, 0.0f), 255.0f);
    float c = luma[i];
    if (fabsf(target - c) < 0.05f) return;
    float f = c > 1.0f ? target / c : 1.0f;
    uchar4 p = rgba[i];
    p.x = to_u8((float)p.x * f);
    p.y = to_u8((float)p.y * f);
    p.z = to_u8((float)p.z * f);
    rgba[i] = p;
}
