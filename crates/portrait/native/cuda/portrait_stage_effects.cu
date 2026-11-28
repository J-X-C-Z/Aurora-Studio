#include <math.h>

__device__ __forceinline__ unsigned char clamp_u8(float v) {
    if (v <= 0.0f) return 0;
    if (v >= 255.0f) return 255;
    return (unsigned char)rintf(v);
}

__device__ __forceinline__ void copy_rgba(
    const unsigned char* src,
    unsigned char* dst,
    unsigned int idx
) {
    unsigned int off = idx * 4u;
    dst[off]     = src[off];
    dst[off + 1] = src[off + 1];
    dst[off + 2] = src[off + 2];
    dst[off + 3] = src[off + 3];
}

__device__ __forceinline__ void box_blur_rgb_norm(
    const unsigned char* src,
    int width,
    int height,
    int x,
    int y,
    int radius,
    float* out_r,
    float* out_g,
    float* out_b
) {
    float sr = 0.0f;
    float sg = 0.0f;
    float sb = 0.0f;
    int count_1d = 2 * radius + 1;
    float inv_row = 1.0f / (float)count_1d;
    for (int dy = -radius; dy <= radius; ++dy) {
        int sy = y + dy;
        if (sy < 0) sy = 0;
        if (sy >= height) sy = height - 1;
        float hr = 0.0f;
        float hg = 0.0f;
        float hb = 0.0f;
        for (int dx = -radius; dx <= radius; ++dx) {
            int sx = x + dx;
            if (sx < 0) sx = 0;
            if (sx >= width) sx = width - 1;
            unsigned int off = (unsigned int)(sy * width + sx) * 4u;
            hr += (float)src[off];
            hg += (float)src[off + 1];
            hb += (float)src[off + 2];
        }
        sr += hr * inv_row;
        sg += hg * inv_row;
        sb += hb * inv_row;
    }
    float inv_total = inv_row / 255.0f;
    *out_r = sr * inv_total;
    *out_g = sg * inv_total;
    *out_b = sb * inv_total;
}

extern "C" __global__ void pf90001_texture_smooth_kernel(
    const unsigned char* src_rgba,
    const unsigned char* skin_mask,
    unsigned char* dst_rgba,
    unsigned int width,
    unsigned int height,
    float strength
) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned int idx = y * width + x;
    float m = (float)skin_mask[idx] / 255.0f;
    if (m <= 0.0f) {
        copy_rgba(src_rgba, dst_rgba, idx);
        return;
    }

    float br = 0.0f, bg = 0.0f, bb = 0.0f;
    box_blur_rgb_norm(src_rgba, (int)width, (int)height, (int)x, (int)y, 4, &br, &bg, &bb);

    unsigned int off = idx * 4u;
    float sr = (float)src_rgba[off] / 255.0f;
    float sg = (float)src_rgba[off + 1] / 255.0f;
    float sb = (float)src_rgba[off + 2] / 255.0f;
    float dr = br - sr;
    float dg = bg - sg;
    float db = bb - sb;
    float diff2 = dr * dr + dg * dg + db * db;
    float edge_keep = expf(-diff2 / (2.0f * 0.085f * 0.085f));
    float w = strength * m * edge_keep;

    dst_rgba[off]     = clamp_u8(roundf((sr + dr * w) * 255.0f));
    dst_rgba[off + 1] = clamp_u8(roundf((sg + dg * w) * 255.0f));
    dst_rgba[off + 2] = clamp_u8(roundf((sb + db * w) * 255.0f));
    dst_rgba[off + 3] = src_rgba[off + 3];
}

extern "C" __global__ void pf90003_body_skin_uniform_kernel(
    const unsigned char* src_rgba,
    const unsigned char* skin_mask,
    const unsigned char* face_mask,
    unsigned char* dst_rgba,
    unsigned int width,
    unsigned int height,
    unsigned int has_face_mask,
    float strength,
    float target_r,
    float target_g,
    float target_b,
    float target_luma
) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned int idx = y * width + x;
    float s = (float)skin_mask[idx] / 255.0f;
    float f = has_face_mask ? ((float)face_mask[idx] / 255.0f) : 0.0f;
    float body = s * (1.0f - f * 0.85f);
    float bw = (body > 0.01f) ? body : 0.0f;
    if (bw <= 0.0f) {
        copy_rgba(src_rgba, dst_rgba, idx);
        return;
    }

    unsigned int off = idx * 4u;
    float r = (float)src_rgba[off];
    float g = (float)src_rgba[off + 1];
    float b = (float)src_rgba[off + 2];
    float luma = fmaxf(0.299f * r + 0.587f * g + 0.114f * b, 1.0f);
    float ratio = fminf(fmaxf(luma / target_luma, 0.6f), 1.5f);
    float ur = target_r * ratio;
    float ug = target_g * ratio;
    float ub = target_b * ratio;
    float blend = strength * bw * 0.55f;

    dst_rgba[off]     = clamp_u8(roundf(r + (ur - r) * blend));
    dst_rgba[off + 1] = clamp_u8(roundf(g + (ug - g) * blend));
    dst_rgba[off + 2] = clamp_u8(roundf(b + (ub - b) * blend));
    dst_rgba[off + 3] = src_rgba[off + 3];
}

extern "C" __global__ void pf100005_eye_white_purify_kernel(
    const unsigned char* src_rgba,
    const unsigned char* face_mask,
    unsigned char* dst_rgba,
    unsigned int width,
    unsigned int height,
    unsigned int x0,
    unsigned int y0,
    unsigned int x1,
    unsigned int eye_y_end,
    float strength
) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned int idx = y * width + x;
    if (x < x0 || x > x1 || y < y0 || y > eye_y_end) {
        copy_rgba(src_rgba, dst_rgba, idx);
        return;
    }
    float fm = (float)face_mask[idx] / 255.0f;
    if (fm <= 0.0f) {
        copy_rgba(src_rgba, dst_rgba, idx);
        return;
    }

    unsigned int off = idx * 4u;
    float r = (float)src_rgba[off] / 255.0f;
    float g = (float)src_rgba[off + 1] / 255.0f;
    float b = (float)src_rgba[off + 2] / 255.0f;
    float max_c = fmaxf(r, fmaxf(g, b));
    float min_c = fminf(r, fminf(g, b));
    float luma = 0.299f * r + 0.587f * g + 0.114f * b;
    float sat = (max_c > 1e-4f) ? ((max_c - min_c) / max_c) : 0.0f;
    if (luma > 0.35f && sat < 0.35f) {
        float luma_w = fminf(fmaxf((luma - 0.35f) / 0.35f, 0.0f), 1.0f);
        float sat_w = fminf(fmaxf((0.35f - sat) / 0.35f, 0.0f), 1.0f);
        float w_pix = strength * fm * luma_w * sat_w * 0.65f;
        float neutral = fminf(fmaxf(luma + 0.08f * w_pix, 0.0f), 1.0f);
        dst_rgba[off]     = clamp_u8(roundf((r + (neutral - r) * w_pix) * 255.0f));
        dst_rgba[off + 1] = clamp_u8(roundf((g + (neutral - g) * w_pix) * 255.0f));
        dst_rgba[off + 2] = clamp_u8(roundf((b + (neutral - b) * w_pix) * 255.0f));
        dst_rgba[off + 3] = src_rgba[off + 3];
    } else {
        copy_rgba(src_rgba, dst_rgba, idx);
    }
}

extern "C" __global__ void pf30189_double_chin_shadow_kernel(
    const unsigned char* src_rgba,
    unsigned char* dst_rgba,
    unsigned int width,
    unsigned int height,
    float cx,
    float cy,
    float rx,
    float ry,
    float strength
) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned int idx = y * width + x;
    float dx = ((float)x - cx) / rx;
    float dy = ((float)y - cy) / ry;
    float r2 = dx * dx + dy * dy;
    if (r2 < 1.0f) {
        float bell = (1.0f - r2) * strength * 0.32f;
        unsigned int off = idx * 4u;
        float r = (float)src_rgba[off];
        float g = (float)src_rgba[off + 1];
        float b = (float)src_rgba[off + 2];
        dst_rgba[off]     = clamp_u8(roundf(r + (255.0f - r) * bell));
        dst_rgba[off + 1] = clamp_u8(roundf(g + (255.0f - g) * bell));
        dst_rgba[off + 2] = clamp_u8(roundf(b + (255.0f - b) * bell));
        dst_rgba[off + 3] = src_rgba[off + 3];
    } else {
        copy_rgba(src_rgba, dst_rgba, idx);
    }
}

extern "C" __global__ void pf30209_jawline_enhance_kernel(
    const unsigned char* src_rgba,
    const unsigned char* mask,
    unsigned char* dst_rgba,
    unsigned int width,
    unsigned int height,
    unsigned int x0,
    unsigned int jaw_y_min,
    unsigned int x1,
    unsigned int y1,
    float strength
) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned int idx = y * width + x;
    if (x < x0 || x > x1 || y < jaw_y_min || y > y1) {
        copy_rgba(src_rgba, dst_rgba, idx);
        return;
    }
    float m = (float)mask[idx] / 255.0f;
    if (m <= 0.0f) {
        copy_rgba(src_rgba, dst_rgba, idx);
        return;
    }

    float br = 0.0f, bg = 0.0f, bb = 0.0f;
    box_blur_rgb_norm(src_rgba, (int)width, (int)height, (int)x, (int)y, 3, &br, &bg, &bb);

    unsigned int denom_u = (y1 > jaw_y_min) ? (y1 - jaw_y_min) : 1u;
    float y_w = fminf(fmaxf((float)(y - jaw_y_min) / (float)denom_u, 0.0f), 1.0f);
    float amount = strength * m * y_w * 0.65f;

    unsigned int off = idx * 4u;
    float sr = (float)src_rgba[off] / 255.0f;
    float sg = (float)src_rgba[off + 1] / 255.0f;
    float sb = (float)src_rgba[off + 2] / 255.0f;

    dst_rgba[off]     = clamp_u8(roundf((sr + (sr - br) * amount) * 255.0f));
    dst_rgba[off + 1] = clamp_u8(roundf((sg + (sg - bg) * amount) * 255.0f));
    dst_rgba[off + 2] = clamp_u8(roundf((sb + (sb - bb) * amount) * 255.0f));
    dst_rgba[off + 3] = src_rgba[off + 3];
}
