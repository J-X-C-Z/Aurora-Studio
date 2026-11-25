// Generic Bayer / X-Trans linear RAW post-processing CUDA kernels.
//
// Compiled with:
//   nvcc --ptx -O3 --fmad=false -arch=compute_75 -o raw_linear_post.ptx raw_linear_post.cu
//
// Kernels:
//   1. pack_rgb3_to_f4_kernel       : tightly-packed [f32; 3] -> float4
//   2. unpack_f4_to_rgb3_kernel     : float4 -> tightly-packed [f32; 3]
//   3. highlight_recovery_kernel    : 1:1 with z6iii/reconstruct/highlight.cl
//   4. anti_alias_clamped_kernel    : 1:1 with z6iii/reconstruct/reconstruct.cl (program 4),
//                                     with in-kernel [0, W-1]x[0, H-1] clamping (no CPU halo)
//   5. color_noise_2012_encode_kernel : 1:1 with z6iii/render.rs apply_linear_color_noise_2012
//   6. color_noise_2012_pass_h_kernel : horizontal range filter (stage 1 & ISO>=400 stage 2)
//   7. color_noise_2012_pass_v_kernel : vertical range filter (stage 1 & ISO>=400 stage 2)
//   8. color_noise_2012_decode_kernel : LUV -> linear RGB reconstruction

// Built-in CUDA vector/math intrinsics (float4, make_float4, fminf, fmaxf, fmaf)
// are provided directly by nvcc without pulling in host CRT/runtime headers.

// ============================================================================
// 0. RGB3 <-> float4 pack / unpack helpers (keeps CPU free of staging copies)
// ============================================================================

extern "C" __global__ void pack_rgb3_to_f4_kernel(
    const float* __restrict__ src_rgb3,
    float4* __restrict__ dst_f4,
    int total_pixels
) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_pixels) return;
    const int base = idx * 3;
    dst_f4[idx] = make_float4(src_rgb3[base], src_rgb3[base + 1], src_rgb3[base + 2], 0.0f);
}

extern "C" __global__ void unpack_f4_to_rgb3_kernel(
    const float4* __restrict__ src_f4,
    float* __restrict__ dst_rgb3,
    int total_pixels
) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_pixels) return;
    const float4 px = src_f4[idx];
    const int base = idx * 3;
    dst_rgb3[base]     = px.x;
    dst_rgb3[base + 1] = px.y;
    dst_rgb3[base + 2] = px.z;
}

// ============================================================================
// 1. Highlight recovery (1:1 with z6iii/reconstruct/highlight.cl)
// ============================================================================

struct HighlightParams {
    float p[16];
};

__device__ __forceinline__ float recover_channel(
    float value,
    float n0,
    float n1,
    float threshold,
    float strength
) {
    if (!(value > threshold)) {
        return value;
    }
    const float excess = (value - threshold) * (strength / threshold);
    const float balance = fminf(n0, n1) / value;
    const float weight = fminf(excess * balance, 1.0f);
    const float mean = (n0 + n1) * 0.5f;
    return mean * weight + value * (1.0f - weight);
}

__device__ __forceinline__ float normalized_excess(float value, float threshold, float span) {
    return value > threshold ? (value - threshold) * (1.0f / span) : 0.0f;
}

extern "C" __global__ void highlight_recovery_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    int total_pixels,
    HighlightParams params
) {
    const int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_pixels) return;

    const float4 pixel = src[index];
    const float tx = params.p[0];
    const float ty = params.p[1];
    const float tz = params.p[2];
    const float strength = params.p[3];

    float rx = recover_channel(pixel.x, pixel.y, pixel.z, tx, strength);
    float ry = recover_channel(pixel.y, pixel.x, pixel.z, ty, strength);
    float rz = recover_channel(pixel.z, pixel.y, pixel.x, tz, strength);

    if (rx > tx || ry > ty || rz > tz) {
        const float red_excess   = normalized_excess(rx, tx, params.p[4]);
        const float green_excess = normalized_excess(ry, ty, params.p[5]);
        const float blue_excess  = normalized_excess(rz, tz, params.p[6]);
        const float nx = rx * params.p[7];
        const float ny = ry * params.p[8];
        const float nz = rz * params.p[9];
        const float floor_level = fminf(fminf(nx, ny), nz) * params.p[10];
        const float fade = ((red_excess + green_excess) + blue_excess) * floor_level;
        const float weight = fminf(1.0f, fade * fade);
        const float luma = (rx * 0.299f + ry * 0.587f) + rz * 0.114f;
        const float one_minus_w = 1.0f - weight;
        rx = rx * one_minus_w + luma * weight;
        ry = ry * one_minus_w + luma * weight;
        rz = rz * one_minus_w + luma * weight;
    }

    dst[index] = make_float4(rx, ry, rz, pixel.w);
}

// ============================================================================
// 2. Anti-alias (1:1 with reconstruct.cl program 4, in-kernel edge clamp)
// ============================================================================

__device__ __forceinline__ int clamp_i(int v, int hi) {
    return v < 0 ? 0 : (v > hi ? hi : v);
}

__device__ __forceinline__ float median_chroma_clamped(
    const float4* __restrict__ input,
    int width,
    int height,
    int col,
    int row,
    bool blue
) {
    float values[25];
    int count = 0;
    const int max_x = width - 1;
    const int max_y = height - 1;
    for (int dy = -2; dy <= 2; ++dy) {
        const int sy = clamp_i(row + dy, max_y);
        const int row_base = sy * width;
        for (int dx = -2; dx <= 2; ++dx) {
            const int sx = clamp_i(col + dx, max_x);
            const float4 pixel = input[row_base + sx];
            const float chroma = (blue ? pixel.z : pixel.x) - pixel.y;
            int index = count;
            while (index > 0 && values[index - 1] > chroma) {
                values[index] = values[index - 1];
                --index;
            }
            values[index] = chroma;
            ++count;
        }
    }
    return values[12];
}

__device__ __forceinline__ float aa_blend(
    float original,
    float green,
    float chroma,
    float factor,
    float gain
) {
    const float z = chroma * factor;
    const float gate = fmaxf(0.0f, fmaf(-z, z, 1.0f));
    const float weight = (gate * gate) * gain;
    return fmaf(weight, chroma + green, (1.0f - weight) * original);
}

extern "C" __global__ void anti_alias_clamped_kernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int width,
    int height,
    float param_x,
    float param_y
) {
    const int column = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int row    = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (column >= width || row >= height) return;

    const int idx = row * width + column;
    const float4 pixel = input[idx];
    const float factor = param_x / fmaxf(1.0f, pixel.y + 1.0f);
    const float r = aa_blend(
        pixel.x,
        pixel.y,
        median_chroma_clamped(input, width, height, column, row, false),
        factor,
        param_y
    );
    const float b = aa_blend(
        pixel.z,
        pixel.y,
        median_chroma_clamped(input, width, height, column, row, true),
        factor,
        param_y
    );
    output[idx] = make_float4(r, pixel.y, b, pixel.w);
}

// ============================================================================
// 3. Linear ColorNoise2012 (1:1 with z6iii/render.rs apply_linear_color_noise_2012)
// ============================================================================

static constexpr float CN_K0 = 0.299f;
static constexpr float CN_K1 = 0.587f;
static constexpr float CN_K2 = 0.114f;
static constexpr float CN_K3 = 35.759254f;
static constexpr float CN_PX = 0.17887762f;
static constexpr float CN_PY = 0.3776723f;

struct ColorNoisePassParams {
    int width;
    int height;
    int step;
    int radius;
    float pz_luma;
    float pw_luma;
    float pz_chroma;
    float pw_chroma;
    float blend;
    int use_blend;
};

extern "C" __global__ void color_noise_2012_encode_kernel(
    const float4* __restrict__ src_rgb,
    float4* __restrict__ dst_luv,
    int total_pixels
) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_pixels) return;

    const float4 px = src_rgb[idx];
    const float r = px.x;
    const float g = px.y;
    const float b = px.z;
    const float l = fmaxf(CN_K0 * r + CN_K1 * g + CN_K2 * b, 0.0f) + CN_K3;
    const float inv_l = 1.0f / l;
    dst_luv[idx] = make_float4(l, (r - g) * inv_l, (b - g) * inv_l, 0.0f);
}

extern "C" __global__ void color_noise_2012_pass_h_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    ColorNoisePassParams p
) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= p.width || y >= p.height) return;

    const int row_base = y * p.width;
    const float4 c = src[row_base + x];
    const float lc = c.x;
    const float uc = c.y;
    const float vc = c.z;
    const float t_l = fmaxf(p.pz_luma + p.pw_luma * lc, 1.0f);
    const float t_l2 = t_l * t_l;
    const float t_c = fmaxf(p.pz_chroma + p.pw_chroma * lc, 1.0f);
    const float u_lc = uc * lc;
    const float v_lc = vc * lc;
    const int max_x = p.width - 1;

    float sum_u = 0.0f;
    float sum_v = 0.0f;
    float sum_w = 0.0f;
    for (int off = -p.radius; off <= p.radius; off += p.step) {
        const int nx = clamp_i(x + off, max_x);
        const float4 nb = src[row_base + nx];
        const float li = nb.x;
        const float ui = nb.y;
        const float vi = nb.z;
        const float dl = lc - li;
        const float du = u_lc - ui * li;
        const float dv = v_lc - vi * li;
        const float dl2 = dl * dl;
        const float dc = fmaxf(CN_PX * du * du, CN_PY * dv * dv);
        const float w_l = t_l2 / (dl2 * dl2 + t_l2);
        const float w_c = t_c / (dc + t_c);
        const float wt = fmaxf(w_l * w_c - 0.01f, 0.0f);
        sum_u += ui * wt;
        sum_v += vi * wt;
        sum_w += wt;
    }

    float out_u = uc;
    float out_v = vc;
    if (sum_w > 1e-6f) {
        const float inv_w = 1.0f / sum_w;
        if (p.use_blend != 0) {
            out_u = uc + (sum_u * inv_w - uc) * p.blend;
            out_v = vc + (sum_v * inv_w - vc) * p.blend;
        } else {
            out_u = sum_u * inv_w;
            out_v = sum_v * inv_w;
        }
    }
    dst[row_base + x] = make_float4(lc, out_u, out_v, 0.0f);
}

extern "C" __global__ void color_noise_2012_pass_v_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    ColorNoisePassParams p
) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= p.width || y >= p.height) return;

    const float4 c = src[y * p.width + x];
    const float lc = c.x;
    const float uc = c.y;
    const float vc = c.z;
    const float t_l = fmaxf(p.pz_luma + p.pw_luma * lc, 1.0f);
    const float t_l2 = t_l * t_l;
    const float t_c = fmaxf(p.pz_chroma + p.pw_chroma * lc, 1.0f);
    const float u_lc = uc * lc;
    const float v_lc = vc * lc;
    const int max_y = p.height - 1;

    float sum_u = 0.0f;
    float sum_v = 0.0f;
    float sum_w = 0.0f;
    for (int off = -p.radius; off <= p.radius; off += p.step) {
        const int ny = clamp_i(y + off, max_y);
        const float4 nb = src[ny * p.width + x];
        const float li = nb.x;
        const float ui = nb.y;
        const float vi = nb.z;
        const float dl = lc - li;
        const float du = u_lc - ui * li;
        const float dv = v_lc - vi * li;
        const float dl2 = dl * dl;
        const float dc = fmaxf(CN_PX * du * du, CN_PY * dv * dv);
        const float w_l = t_l2 / (dl2 * dl2 + t_l2);
        const float w_c = t_c / (dc + t_c);
        const float wt = fmaxf(w_l * w_c - 0.01f, 0.0f);
        sum_u += ui * wt;
        sum_v += vi * wt;
        sum_w += wt;
    }

    float out_u = uc;
    float out_v = vc;
    if (sum_w > 1e-6f) {
        const float inv_w = 1.0f / sum_w;
        if (p.use_blend != 0) {
            out_u = uc + (sum_u * inv_w - uc) * p.blend;
            out_v = vc + (sum_v * inv_w - vc) * p.blend;
        } else {
            out_u = sum_u * inv_w;
            out_v = sum_v * inv_w;
        }
    }
    dst[y * p.width + x] = make_float4(lc, out_u, out_v, 0.0f);
}

extern "C" __global__ void color_noise_2012_decode_kernel(
    const float4* __restrict__ src_luv,
    float4* __restrict__ dst_rgb,
    int total_pixels
) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_pixels) return;

    const float4 px = src_luv[idx];
    const float lc = px.x;
    const float u  = px.y;
    const float v  = px.z;
    const float ul = u * lc;
    const float vl = v * lc;
    const float g  = lc - ul * CN_K0 - vl * CN_K2 - CN_K3;
    const float r0 = fmaxf(ul + g, 1.0f);
    const float g0 = fmaxf(g, 1.0f);
    const float b0 = fmaxf(vl + g, 1.0f);
    const float sc = fminf(
        (lc - (CN_K3 - 1.0f)) / (fmaxf(CN_K0 * r0 + CN_K1 * g0 + CN_K2 * b0, 0.0f) + 1.0f),
        1.5f
    );
    dst_rgb[idx] = make_float4(r0 * sc, g0 * sc, b0 * sc, 0.0f);
}
