// Native CUDA GPU-resident reconstruction pipeline for Nikon Z6III HE RAW.
//
// Compiled with:
//   nvcc -ptx --gpu-architecture=compute_75 -O3 --fmad=false --prec-div=true --prec-sqrt=true
//        z6iii_raw.cu -o z6iii_raw.ptx
//
// Single H2D upload (raw 14-bit Bayer) -> 7 GPU-resident stages -> single D2H
// download (oriented RGBA16). Every floating-point operation matches the
// OpenCL + CPU C1v2012 baseline (`reconstruct.cl`, `highlight.cl`,
// `c1_stage2_basic.rs`, `apply_linear_chromatic_aberration`, `orient_rgba16`).

#include <stdint.h>

// ============================================================================
// Stage 0: Mirror active Bayer crop with HALO=20 and left-shift by 2 bits.
// ============================================================================

__device__ __forceinline__ int mirror_bayer_pairs(int coord, int size) {
    // Arithmetic right shift by 1 equals Euclidean division by 2 for 2's complement.
    int pair = coord >> 1;
    if (pair < 0) {
        pair = -pair;
    } else if (pair >= (size >> 1)) {
        pair = size - 1 - pair;
    }
    return (pair << 1) + (coord & 1);
}

__device__ __forceinline__ int mirror_once(int coord, int size) {
    int end = size - 1;
    int r = (coord < 0) ? -coord : (coord > end ? 2 * end - coord : coord);
    if (r < 0) r = 0;
    else if (r > end) r = end;
    return r;
}

extern "C" __global__ void mirror_bayer_halo_shift2(
    const uint16_t* __restrict__ src,
    uint16_t* __restrict__ dst,
    int full_w,
    int active_left,
    int active_top,
    int active_w,
    int active_h,
    int halo
) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int halo_w = active_w + 2 * halo;
    const int halo_h = active_h + 2 * halo;
    if (x >= halo_w || y >= halo_h) return;

    const int sx = active_left + mirror_bayer_pairs(x - halo, active_w);
    const int sy = active_top + mirror_once(y - halo, active_h);
    dst[y * halo_w + x] = (uint16_t)(src[sy * full_w + sx] << 2);
}

// ============================================================================
// Stage 0b: AdaptiveGG (reconstruct.cl program 0)
// ============================================================================

extern "C" __global__ void adaptive_gg(
    const uint16_t* __restrict__ source,
    uint16_t* __restrict__ destination,
    int width,
    int height,
    float threshold,
    float s1,
    float black
) {
    const int col = 2 * (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (col + 1 >= width || row >= height) return;

    const int point = row * width + col;
    // Preserve 2px border untouched, matching `output = input.clone()` + ROI [2, 2, w-4, h-4].
    if (row < 2 || row >= height - 2 || col < 2 || col >= width - 2) {
        destination[point] = source[point];
        destination[point + 1] = source[point + 1];
        return;
    }

    const int green = 1 - (row & 1);
    const float original = (float)source[point + green];
    const float signal = fmaxf(original - black, 0.0f);
    uint16_t corrected = source[point + green];
    if (signal < 4.0f * threshold) {
        const int center = point + green;
        const float northwest = (float)source[center - width - 1];
        const float northeast = (float)source[center - width + 1];
        const float southwest = (float)source[center + width - 1];
        const float southeast = (float)source[center + width + 1];
        const float base = fmaf(signal, 0.25f, s1);
        const float w_nw = 1.0f / (base + fabsf(original - northwest));
        const float w_ne = 1.0f / (base + fabsf(original - northeast));
        const float w_sw = 1.0f / (base + fabsf(original - southwest));
        const float w_se = 1.0f / (base + fabsf(original - southeast));
        float color = fmaf(northwest, w_nw, northeast * w_ne);
        color = fmaf(southwest, w_sw, color);
        color = fmaf(southeast, w_se, color);
        const float sum = ((w_nw + w_ne) + w_sw) + w_se;
        const float dark = fmaf(color / sum, 0.5f, original * 0.5f);
        float output = dark;
        if (signal >= threshold) {
            const float mix = (signal - threshold) / (3.0f * threshold);
            output = fmaf(dark, 1.0f - mix, original * mix);
        }
        corrected = (uint16_t)__float2uint_rz(output);
    }
    destination[point + green] = corrected;
    destination[point + 1 - green] = source[point + 1 - green];
}

// ============================================================================
// Stage 1: Bayer directional green interpolation (reconstruct.cl program 1)
// ============================================================================

__device__ __forceinline__ int tap(const uint16_t* center, int stride, int row, int column) {
    return (int)center[row * stride + column];
}

__device__ __forceinline__ int4 residual(const uint16_t* center, int stride, int y, int x, int predicted) {
    return make_int4(
        tap(center, stride, y, x - 1) - predicted,
        tap(center, stride, y, x + 1) - predicted,
        tap(center, stride, y - 1, x) - predicted,
        tap(center, stride, y + 1, x) - predicted
    );
}

__device__ __forceinline__ uint32_t uabs_i(int v) {
    return (uint32_t)(v < 0 ? -v : v);
}

__device__ __forceinline__ int choose_dir(const uint16_t* center, int stride, float gain, float black) {
    const int qgain = __float2int_rz(gain * 1024.0f);
    const uint32_t curvature_gain = __float2uint_rz((gain * 1024.0f) * 1.6f);
    const int bias = __float2int_rz(fmaf(-black, gain, black));
    const int center_prediction = __float2int_rd((float)center[0] * gain) + bias;
    const int4 reference = residual(center, stride, 0, 0, center_prediction);

    uint32_t horizontal = (uabs_i(tap(center, stride, 0, -2) + tap(center, stride, 0, 2) - 2 * (int)center[0]) * curvature_gain) >> 10;
    uint32_t vertical   = (uabs_i(tap(center, stride, -2, 0) + tap(center, stride, 2, 0) - 2 * (int)center[0]) * curvature_gain) >> 10;

    for (int sign = -1; sign <= 1; sign += 2) {
        for (int side = -2; side <= 2; side += 4) {
            horizontal += uabs_i(tap(center, stride, sign, 0) - tap(center, stride, sign, side));
            vertical   += uabs_i(tap(center, stride, 0, sign) - tap(center, stride, side, sign));
        }
    }

    uint32_t cx = horizontal, cy = horizontal, cz = vertical, cw = vertical;
    for (int offset = -4; offset <= 4; offset += 8) {
        int hprediction = ((tap(center, stride, 0, offset) * qgain) >> 10) + bias;
        int vprediction = ((tap(center, stride, offset, 0) * qgain) >> 10) + bias;
        int4 across = residual(center, stride, 0, offset, hprediction);
        int4 along  = residual(center, stride, offset, 0, vprediction);
        cx += uabs_i(across.x - reference.x);
        cy += uabs_i(across.y - reference.y);
        cz += uabs_i(along.z  - reference.z);
        cw += uabs_i(along.w  - reference.w);
        horizontal += uabs_i(across.x - across.y);
        vertical   += uabs_i(along.z  - along.w);
    }
    for (int y = -2; y <= 2; y += 2) {
        for (int x = -2; x <= 2; x += 2) {
            const int predicted = ((tap(center, stride, y, x) * qgain) >> 10) + bias;
            const int4 nearby = residual(center, stride, y, x, predicted);
            cx += uabs_i(nearby.x - reference.x);
            cy += uabs_i(nearby.y - reference.y);
            cz += uabs_i(nearby.z - reference.z);
            cw += uabs_i(nearby.w - reference.w);
            horizontal += uabs_i(nearby.x - nearby.y);
            vertical   += uabs_i(nearby.z - nearby.w);
        }
    }
    const uint32_t max_xy = cx > cy ? cx : cy;
    const uint32_t min_xy = cx < cy ? cx : cy;
    const uint32_t max_zw = cz > cw ? cz : cw;
    const uint32_t min_zw = cz < cw ? cz : cw;
    if (max_xy < min_zw && horizontal < vertical) return 1;
    if (max_zw < min_xy && vertical < horizontal) return 2;

    const uint32_t curvature = uabs_i(reference.x + reference.y - reference.z - reference.w) / 2u;
    const int signal = tap(center, stride, 0, -1) + tap(center, stride, 0, 1)
                     + tap(center, stride, -1, 0) + tap(center, stride, 1, 0)
                     - __float2int_rz(black * 4.0f);
    if ((int)(curvature * curvature) < signal) return 3;

    cx = 0; cy = 0; cz = 0; cw = 0;
    for (int y = -4; y <= 4; y += 2) {
        for (int x = -4; x <= 4; x += 2) {
            const int predicted = ((tap(center, stride, y, x) * qgain) >> 10) + bias;
            const int4 res = residual(center, stride, y, x, predicted);
            cx += uabs_i(res.x - reference.x);
            cy += uabs_i(res.y - reference.y);
            cz += uabs_i(res.z - reference.z);
            cw += uabs_i(res.w - reference.w);
        }
    }
    const uint32_t max_xy2 = cx > cy ? cx : cy;
    const uint32_t min_xy2 = cx < cy ? cx : cy;
    const uint32_t max_zw2 = cz > cw ? cz : cw;
    const uint32_t min_zw2 = cz < cw ? cz : cw;
    if (max_xy2 < min_zw2) return 1;
    if (max_zw2 < min_xy2) return 2;
    return 3;
}

__device__ __forceinline__ float axis_weight(const uint16_t* center, int along, int across, float gain, bool scaled) {
    const uint32_t neighbors =
        uabs_i((int)center[-along] - (int)center[along]) +
        uabs_i((int)center[-across + 2 * along] - (int)center[-across]) +
        uabs_i((int)center[across + 2 * along] - (int)center[across]) +
        uabs_i((int)center[3 * along] - (int)center[along]);
    const uint32_t difference = uabs_i((int)center[2 * along] - (int)center[0]);
    if (scaled) return (float)(1u + neighbors + difference) * gain;
    return fmaf((float)difference, gain, (float)(1u + neighbors));
}

__device__ __forceinline__ float midpoint(const uint16_t* center, int offset) {
    return (float)(((int)center[0] + (int)center[offset] + 1) / 2);
}

__device__ __forceinline__ float green_value(
    const uint16_t* center,
    int stride,
    float gain,
    float floor_code,
    float black,
    int direction
) {
    float weight, numer, denom;
    if (direction == 3) {
        const float neg_x = axis_weight(center, -1, stride, gain, false);
        const float pos_x = axis_weight(center,  1, stride, gain, false);
        const float neg_y = axis_weight(center, -stride, 1, gain, false);
        const float pos_y = axis_weight(center,  stride, 1, gain, false);
        const float pair_x = pos_x * neg_x;
        const float pair_y = pos_y * neg_y;
        const float down  = neg_y * pair_x;
        const float up    = pos_y * pair_x;
        const float right = pair_y * neg_x;
        const float left  = pair_y * pos_x;
        weight = down + ((right + left) + up);
        numer = fmaf(right, (float)center[1],
                fmaf(left,  (float)center[-1],
                fmaf(up,    (float)center[-stride],
                     down * (float)center[stride])));
        denom = fmaf(right, midpoint(center, 2),
                fmaf(left,  midpoint(center, -2),
                fmaf(up,    midpoint(center, -2 * stride),
                     down * midpoint(center, 2 * stride))));
    } else {
        const int along  = direction == 1 ? 1 : stride;
        const int across = direction == 1 ? stride : 1;
        const float neg = axis_weight(center, -along, across, gain, false);
        const float pos = axis_weight(center,  along, across, gain, direction == 1);
        weight = pos + neg;
        numer = fmaf(pos, (float)center[-along], neg * (float)center[along]);
        denom = fmaf(pos, midpoint(center, -2 * along), neg * midpoint(center, 2 * along));
    }
    const float floor_value = weight * floor_code;
    const float top    = fmaxf(fmaf(-weight, black, numer + floor_value), floor_value);
    const float bottom = fmaxf(fmaf(-weight, black, denom + floor_value), floor_value);
    const float ratio  = top / bottom;
    return (ratio * (((float)center[0] - black) + floor_code) - floor_code) + black;
}

extern "C" __global__ void bayer_stage1(
    const uint16_t* __restrict__ source,
    float4* __restrict__ output,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    int stride,
    float red_gain,
    float blue_gain,
    float floor_code,
    float black
) {
    const int x = 2 * (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x + 1 >= roi_w || y >= roi_h) return;

    const int odd = y & 1;
    const int start = (roi_y + y) * stride + roi_x + x;
    const uint16_t* center = source + start + odd;
    const float gain = odd ? blue_gain : red_gain;
    const int direction = choose_dir(center, stride, gain, black);
    const float green = green_value(center, stride, gain, floor_code, black, direction);

    if (odd) {
        output[start]     = make_float4(0.0f, (float)source[start], 0.0f, 0.0f);
        output[start + 1] = make_float4(0.0f, green, (float)center[0], 0.0f);
    } else {
        output[start]     = make_float4((float)center[0], green, 0.0f, 0.0f);
        output[start + 1] = make_float4(0.0f, (float)source[start + 1], 0.0f, 0.0f);
    }
}

// ============================================================================
// Stage 2: Basic green refinement + black subtraction + WB scaling
//          (matches `c1_stage2_basic.rs` + `subtract_black_rgb32f` + `* wb`)
// ============================================================================

__device__ __forceinline__ float s2_sample(
    const float4* __restrict__ s1,
    int stride,
    int hx,
    int hy,
    int dy,
    int dx,
    int ch,
    float neg_black
) {
    const float4 p = s1[(hy + dy) * stride + (hx + dx)];
    const float v = (ch == 0) ? p.x : ((ch == 1) ? p.y : p.z);
    return v + neg_black;
}

extern "C" __global__ void stage2_refine_green_and_scale(
    const float4* __restrict__ stage1,
    float4* __restrict__ stage2,
    int halo_w,
    int support_left,
    int support_top,
    int support_w,
    int support_h,
    float neg_black,
    float wb_r,
    float wb_g,
    float wb_b
) {
    const int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (col >= support_w || row >= support_h) return;

    const int hx = support_left + col;
    const int hy = support_top + row;
    const float4 raw_p = stage1[hy * halo_w + hx];
    float r = raw_p.x + neg_black;
    float g = raw_p.y + neg_black;
    float b = raw_p.z + neg_black;

    const bool odd_row = (hy & 1) != 0;
    const bool odd_col = (hx & 1) != 0;
    if (odd_row == odd_col) {
        const int pri = odd_row ? 2 : 0;
        const int opp = odd_row ? 0 : 2;
        const float center = g;
        const float up    = s2_sample(stage1, halo_w, hx, hy, -1,  0, 1, neg_black);
        const float down  = s2_sample(stage1, halo_w, hx, hy,  1,  0, 1, neg_black);
        const float left  = s2_sample(stage1, halo_w, hx, hy,  0, -1, 1, neg_black);
        const float right = s2_sample(stage1, halo_w, hx, hy,  0,  1, 1, neg_black);
        const float ul    = s2_sample(stage1, halo_w, hx, hy, -1, -1, 1, neg_black);
        const float ur    = s2_sample(stage1, halo_w, hx, hy, -1,  1, 1, neg_black);
        const float dl    = s2_sample(stage1, halo_w, hx, hy,  1, -1, 1, neg_black);
        const float dr    = s2_sample(stage1, halo_w, hx, hy,  1,  1, 1, neg_black);
        const float uul   = s2_sample(stage1, halo_w, hx, hy, -2, -1, 1, neg_black);
        const float uur   = s2_sample(stage1, halo_w, hx, hy, -2,  1, 1, neg_black);
        const float ddl   = s2_sample(stage1, halo_w, hx, hy,  2, -1, 1, neg_black);
        const float ddr   = s2_sample(stage1, halo_w, hx, hy,  2,  1, 1, neg_black);
        const float ull   = s2_sample(stage1, halo_w, hx, hy, -1, -2, 1, neg_black);
        const float urr   = s2_sample(stage1, halo_w, hx, hy, -1,  2, 1, neg_black);
        const float dll   = s2_sample(stage1, halo_w, hx, hy,  1, -2, 1, neg_black);
        const float drr   = s2_sample(stage1, halo_w, hx, hy,  1,  2, 1, neg_black);
        const float chroma = odd_row ? b : r;

        const float vertical   = fabsf(down - center) + fabsf(up - center);
        const float horizontal = fabsf(right - center) + fabsf(left - center);

        const float diagonal_back =
            (((fabsf(s2_sample(stage1, halo_w, hx, hy,  2, -2, pri, neg_black) - chroma)
             + fabsf(s2_sample(stage1, halo_w, hx, hy, -2,  2, pri, neg_black) - chroma))
             + fabsf(s2_sample(stage1, halo_w, hx, hy,  1, -1, opp, neg_black)
                   - s2_sample(stage1, halo_w, hx, hy, -1,  1, opp, neg_black)))
             + fabsf(up - left))
             + fabsf(down - right);

        const float diagonal_slash =
            (((fabsf(s2_sample(stage1, halo_w, hx, hy,  2,  2, pri, neg_black) - chroma)
             + fabsf(s2_sample(stage1, halo_w, hx, hy, -2, -2, pri, neg_black) - chroma))
             + fabsf(s2_sample(stage1, halo_w, hx, hy, -1, -1, opp, neg_black)
                   - s2_sample(stage1, halo_w, hx, hy,  1,  1, opp, neg_black)))
             + fabsf(up - right))
             + fabsf(down - left);

        const float g0 = diagonal_back;
        const float g1 = diagonal_slash;
        const float g2 = ((fabsf(ddl - center) + fabsf(uur - center)) + fabsf(up - dl)) + fabsf(down - ur);
        const float g3 = ((fabsf(dll - center) + fabsf(urr - center)) + fabsf(left - ur)) + fabsf(right - dl);
        const float g4 = ((fabsf(ddr - center) + fabsf(uul - center)) + fabsf(down - ul)) + fabsf(up - dr);
        const float g5 = ((fabsf(drr - center) + fabsf(ull - center)) + fabsf(right - ul)) + fabsf(left - dr);

        const float axial = fminf(vertical, horizontal);
        float minimum = axial;
        minimum = fminf(minimum, g0);
        minimum = fminf(minimum, g1);
        minimum = fminf(minimum, g2);
        minimum = fminf(minimum, g3);
        minimum = fminf(minimum, g4);
        minimum = fminf(minimum, g5);

        if (!(axial <= minimum)) {
            const float one_third = __uint_as_float(0x3eaaaaabu);
            const float axial_average = (vertical < horizontal)
                ? ((center + up) + down) * one_third
                : ((center + left) + right) * one_third;
            const float oblique_third = __uint_as_float(0x3eaaaaaau);
            float cand;
            if (g0 <= minimum) {
                cand = ((2.0f * center + ur) + dl) * 0.25f;
            } else if (g1 <= minimum) {
                cand = ((2.0f * center + ul) + dr) * 0.25f;
            } else if (g2 <= minimum) {
                cand = ((ddl + center) + uur) * oblique_third;
            } else if (g3 <= minimum) {
                cand = ((dll + center) + urr) * oblique_third;
            } else if (g4 <= minimum) {
                cand = ((center + uul) + ddr) * oblique_third;
            } else {
                cand = ((center + ull) + drr) * oblique_third;
            }
            g = (cand + axial_average) * 0.5f;
        }
    }

    stage2[row * support_w + col] = make_float4(r * wb_r, g * wb_g, b * wb_b, 0.0f);
}

// ============================================================================
// Stage 3: Planar R/B reconstruction (reconstruct.cl program 2)
// ============================================================================

__device__ __forceinline__ float channel_f4(float4 pixel, int component) {
    return component == 0 ? pixel.x : pixel.z;
}

__device__ __forceinline__ float color_cost(float difference, float denominator) {
    // OpenCL emits `div.full.f32` + `add.f32 0.03f`, which ptxas fuses into
    // `MUFU.RCP` + `FFMA(rcp, |diff|, 0.03f)`. Since `d >= 1.0f`, `div.full`
    // takes the direct `rcp.approx.f32` path without exponent rescaling.
    const float d = fmaxf(1.0f, denominator);
    float rcp;
    asm("rcp.approx.f32 %0, %1;" : "=f"(rcp) : "f"(d));
    return fmaf(rcp, fabsf(difference), 0.03f);
}

extern "C" __global__ void reconstruct_diagonal(
    const float4* __restrict__ source,
    float4* __restrict__ target,
    int width,
    int height,
    float floor_code
) {
    const int column = 2 * (int)(blockIdx.x * blockDim.x + threadIdx.x) + 2;
    const int row = (int)(blockIdx.y * blockDim.y + threadIdx.y) + 2;
    if (column + 1 >= width - 2 || row >= height - 2) return;

    const int odd = row & 1;
    const int position = row * width + column + odd;
    const float4* center = source + position;
    const int missing = odd ? 0 : 2;
    const int native  = 2 - missing;

    const float4 nw = center[-width - 1], ne = center[-width + 1];
    const float4 sw = center[ width - 1], se = center[ width + 1];
    const float cnw = channel_f4(nw, missing), cne = channel_f4(ne, missing);
    const float csw = channel_f4(sw, missing), cse = channel_f4(se, missing);

    const bool main_diag = fabsf(cnw - cse) < fabsf(csw - cne);
    const bool chroma = (((cnw + cne) + csw) + cse) * 0.75f < (((nw.y + se.y) + ne.y) + sw.y);
    const int first  = main_diag ? -width - 1 : width - 1;
    const int second = -first;
    const float c_first  = channel_f4(center[first], missing);
    const float c_second = channel_f4(center[second], missing);
    const float g_first  = center[first].y;
    const float g_second = center[second].y;
    const float4 c0 = center[0];
    const float native_center = channel_f4(c0, native);

    float w_first  = (1.0f + fabsf(g_first  - c0.y)) + fabsf(channel_f4(center[2 * first],  native) - native_center);
    float w_second = (1.0f + fabsf(g_second - c0.y)) + fabsf(channel_f4(center[2 * second], native) - native_center);
    if (chroma) {
        w_first  *= color_cost(c_first  - g_first,  (1.0f + c_first)  + g_first);
        w_second *= color_cost(c_second - g_second, (1.0f + c_second) + g_second);
    }
    const float weight      = w_first + w_second;
    const float numerator   = fmaf(weight, floor_code, fmaf(w_second, c_first, w_first * c_second));
    const float denominator = fmaf(weight, floor_code, fmaf(w_second, g_first, w_first * g_second));
    const float ratio       = numerator / fmaxf(weight * floor_code, denominator);
    const float value       = ratio * (floor_code + c0.y) - floor_code;

    float4 result = c0;
    if (odd) result.x = value;
    else     result.z = value;
    target[position] = result;
    target[position + (odd ? -1 : 1)] = center[odd ? -1 : 1];
}

__device__ __forceinline__ float at_green(const float4* center, int width, int component, float floor_code) {
    const float4 up = center[-width], down = center[width], left = center[-1], right = center[1];
    const float cu = channel_f4(up, component),   cd = channel_f4(down, component);
    const float cl = channel_f4(left, component), cr = channel_f4(right, component);
    const bool chroma = (((cu + cl) + cr) + cd) * 0.75f < (((up.y + down.y) + right.y) + left.y);
    const float vertical   = 1.0f + fabsf(up.y - down.y);
    const float horizontal = 1.0f + fabsf(left.y - right.y);
    const float green = center[0].y;
    float wu = vertical   + fabsf(center[-2 * width].y - green);
    float wd = vertical   + fabsf(center[ 2 * width].y - green);
    float wl = horizontal + fabsf(center[-2].y - green);
    float wr = horizontal + fabsf(center[ 2].y - green);
    if (chroma) {
        wu *= color_cost(up.y    - cu, (1.0f + up.y)    + cu);
        wd *= color_cost(down.y  - cd, (1.0f + down.y)  + cd);
        wl *= color_cost(left.y  - cl, (1.0f + left.y)  + cl);
        wr *= color_cost(right.y - cr, (1.0f + right.y) + cr);
    }
    wu = 1.0f / wu; wd = 1.0f / wd; wl = 1.0f / wl; wr = 1.0f / wr;
    const float weight      = ((wu + wd) + wr) + wl;
    const float numerator   = fmaf(weight, floor_code, fmaf(wr, cr,      fmaf(wl, cl,     fmaf(wu, cu,   wd * cd))));
    const float denominator = fmaf(weight, floor_code, fmaf(wr, right.y, fmaf(wl, left.y, fmaf(wu, up.y, wd * down.y))));
    return fmaf(numerator / fmaxf(weight * floor_code, denominator), floor_code + green, -floor_code);
}

extern "C" __global__ void reconstruct_green(
    const float4* __restrict__ source,
    float4* __restrict__ target,
    int width,
    int height,
    float floor_code
) {
    const int column = 2 * (int)(blockIdx.x * blockDim.x + threadIdx.x) + 4;
    const int row = (int)(blockIdx.y * blockDim.y + threadIdx.y) + 4;
    if (column + 1 >= width - 4 || row >= height - 4) return;

    const int odd = row & 1;
    const int position = row * width + column + (1 - odd);
    const float4* center = source + position;
    const float4 c0 = center[0];
    target[position] = make_float4(
        at_green(center, width, 0, floor_code),
        c0.y,
        at_green(center, width, 2, floor_code),
        c0.w
    );
    target[position + (odd ? 1 : -1)] = center[odd ? 1 : -1];
}

// ============================================================================
// Stage 4: GG blur compensation (reconstruct.cl program 3)
// ============================================================================

__device__ __forceinline__ float guide_value(float4 pixel, float4 weights) {
    const float green = (pixel.y + pixel.y) * weights.y;
    return fmaxf(1.0f, fmaf(pixel.z, weights.z, fmaf(pixel.x, weights.x, green)) + 1.0f);
}

extern "C" __global__ void gg_compensate(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int width,
    int margin,
    int output_width,
    int output_height,
    int phase,
    float wx,
    float wy,
    float wz,
    float ww,
    float param_noise,
    float param_threshold
) {
    const int ox = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int oy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (ox >= output_width || oy >= output_height) return;

    const int column = ox + margin;
    const int row = oy + margin;
    const float4 weights = make_float4(wx, wy, wz, ww);
    const float4* center = input + row * width + column;

    float peak = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if ((row + dy + column + dx + phase) & 1) {
                peak = fmaxf(peak, center[dy * width + dx].y * weights.y);
            }
        }
    }

    float4 result = *center;
    if (peak < 4.0f * param_threshold) {
        float rolloff = 1.0f;
        if (peak >= param_threshold) {
            rolloff = fmaf(param_threshold, 4.0f, -peak) * (1.0f / (3.0f * param_threshold));
        }
        const float middle = guide_value(result, weights);
        const float nw = guide_value(center[-width - 1], weights);
        const float ne = guide_value(center[-width + 1], weights);
        const float sw = guide_value(center[ width - 1], weights);
        const float se = guide_value(center[ width + 1], weights);
        const bool main_diag = fabsf(fmaf(middle, 2.0f, -nw) - se) < fabsf(fmaf(middle, 2.0f, -ne) - sw);
        const float diag = main_diag ? (nw + se) : (ne + sw);
        const float delta = fmaf(fmaf(middle, 2.0f, diag), -0.25f, middle) * 20.0f;
        const float limit = fminf(middle, param_noise);
        const float limited = (delta * limit) / (limit + fabsf(delta));
        const float scale = fmaf(limited * rolloff, 0.75f, middle) / middle;
        result.x *= scale;
        result.y *= scale;
        result.z *= scale;
        result.w *= scale;
    }
    output[oy * output_width + ox] = result;
}

// ============================================================================
// Stage 5: Optional linear chromatic aberration warp
//          (matches `render.rs::apply_linear_chromatic_aberration`)
// ============================================================================

__device__ __forceinline__ float sample_lut(const float2* table, int count, float t) {
    if (count <= 0) return 0.0f;
    float r0 = table[0].x;
    float v0 = table[0].y;
    if (!isfinite(t) || t <= r0) return v0;
    float prev_r = r0;
    float prev_v = v0;
    for (int i = 1; i < count; ++i) {
        float r = table[i].x;
        float v = table[i].y;
        if (t <= r) {
            float span = r - prev_r;
            return (span <= 0.0f) ? v : (prev_v + __fdiv_rn((v - prev_v) * (t - prev_r), span));
        }
        prev_r = r;
        prev_v = v;
    }
    return prev_v;
}

__device__ __forceinline__ float sample_bilinear_ch(
    const float4* __restrict__ src,
    int tile_w,
    int tile_h,
    float fx,
    float fy,
    int ch
) {
    const float max_x = (float)(tile_w - 1);
    const float max_y = (float)(tile_h - 1);
    const float sx = fminf(fmaxf(fx, 0.0f), max_x);
    const float sy = fminf(fmaxf(fy, 0.0f), max_y);
    const int x0 = (int)floorf(sx);
    const int y0 = (int)floorf(sy);
    const int x1 = (x0 + 1 < tile_w) ? (x0 + 1) : (tile_w - 1);
    const int y1 = (y0 + 1 < tile_h) ? (y0 + 1) : (tile_h - 1);
    const float tx = sx - (float)x0;
    const float ty = sy - (float)y0;
    const float4 q00 = src[y0 * tile_w + x0];
    const float4 q10 = src[y0 * tile_w + x1];
    const float4 q01 = src[y1 * tile_w + x0];
    const float4 q11 = src[y1 * tile_w + x1];
    const float p00 = (ch == 0) ? q00.x : q00.z;
    const float p10 = (ch == 0) ? q10.x : q10.z;
    const float p01 = (ch == 0) ? q01.x : q01.z;
    const float p11 = (ch == 0) ? q11.x : q11.z;
    const float top = p00 + (p10 - p00) * tx;
    const float bot = p01 + (p11 - p01) * tx;
    return top + (bot - top) * ty;
}

extern "C" __global__ void linear_ca_warp(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    int tile_w,
    int tile_h,
    int origin_x,
    int origin_y,
    int full_w,
    int full_h,
    const float2* __restrict__ red_lut,
    int red_count,
    const float2* __restrict__ blue_lut,
    int blue_count
) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= tile_w || y >= tile_h) return;

    float4 pixel = src[y * tile_w + x];
    const float cx = (float)full_w * 0.5f;
    const float cy = (float)full_h * 0.5f;
    const float half_diag = fmaxf(__fsqrt_rn(cx * cx + cy * cy), 1.0f);
    const float gx = (float)(origin_x + x);
    const float gy = (float)(origin_y + y);
    const float dx = gx - cx;
    const float dy = gy - cy;
    const float r = __fsqrt_rn(dx * dx + dy * dy);
    if (r >= 1e-4f) {
        const float inv_r = __fdiv_rn(1.0f, r);
        const float dir_x = dx * inv_r;
        const float dir_y = dy * inv_r;
        const float t = fminf(fmaxf(__fdiv_rn(r, half_diag), 0.0f), 1.0f);
        if (red_count > 0) {
            const float shift = fminf(fmaxf(sample_lut(red_lut, red_count, t) * half_diag, -3.0f), 3.0f);
            pixel.x = sample_bilinear_ch(src, tile_w, tile_h, (float)x + dir_x * shift, (float)y + dir_y * shift, 0);
        }
        if (blue_count > 0) {
            const float shift = fminf(fmaxf(sample_lut(blue_lut, blue_count, t) * half_diag, -3.0f), 3.0f);
            pixel.z = sample_bilinear_ch(src, tile_w, tile_h, (float)x + dir_x * shift, (float)y + dir_y * shift, 2);
        }
    }
    dst[y * tile_w + x] = pixel;
}

// ============================================================================
// Stage 6: Highlight recovery (highlight.cl)
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
    return fmaf(mean, weight, value * (1.0f - weight));
}

__device__ __forceinline__ float normalized_excess(float value, float threshold, float span) {
    return value > threshold ? (value - threshold) * (1.0f / span) : 0.0f;
}

extern "C" __global__ void highlight_recover(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    int total_pixels,
    HighlightParams params
) {
    const int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_pixels) return;

    const float4 pixel = src[index];
    const float tx = params.p[0], ty = params.p[1], tz = params.p[2];
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
        const float luma = fmaf(rz, 0.114f, fmaf(rx, 0.299f, ry * 0.587f));
        const float one_minus_w = 1.0f - weight;
        const float luma_w = luma * weight;
        rx = fmaf(rx, one_minus_w, luma_w);
        ry = fmaf(ry, one_minus_w, luma_w);
        rz = fmaf(rz, one_minus_w, luma_w);
    }

    dst[index] = make_float4(rx, ry, rz, pixel.w);
}

// ============================================================================
// Stage 7: 5x5 median chroma anti-alias + clamp u16 + EXIF orient
//          (matches `reconstruct.cl` program 4 + `pack_rgba16` + `orient_rgba16`)
// ============================================================================

__device__ __forceinline__ float median_chroma(const float4* center, int width, bool blue) {
    float values[25];
    int count = 0;
    for (int row = -2; row <= 2; ++row) {
        for (int col = -2; col <= 2; ++col) {
            const float4 pixel = center[row * width + col];
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

__device__ __forceinline__ float aa_blend(float original, float green, float chroma, float factor, float gain) {
    const float z = chroma * factor;
    const float gate = fmaxf(0.0f, fmaf(-z, z, 1.0f));
    const float weight = (gate * gate) * gain;
    return fmaf(weight, chroma + green, (1.0f - weight) * original);
}

__device__ __forceinline__ uint16_t float_to_u16_clamp(float v) {
    if (!(v > 0.0f)) return 0u;
    if (v >= 65535.0f) return 65535u;
    return (uint16_t)__float2uint_rz(v);
}

extern "C" __global__ void anti_alias_and_orient_rgba16(
    const float4* __restrict__ input,
    ushort4* __restrict__ output,
    int halo_w,
    int out_w,
    int out_h,
    float param_x,
    float param_y,
    int orientation
) {
    const int column = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (column >= out_w || row >= out_h) return;

    const float4* center = input + (row + 2) * halo_w + (column + 2);
    const float4 pixel = *center;
    const float factor = param_x / fmaxf(1.0f, pixel.y + 1.0f);
    const float r = aa_blend(pixel.x, pixel.y, median_chroma(center, halo_w, false), factor, param_y);
    const float g = pixel.y;
    const float b = aa_blend(pixel.z, pixel.y, median_chroma(center, halo_w, true),  factor, param_y);

    const ushort4 packed = make_ushort4(
        float_to_u16_clamp(r),
        float_to_u16_clamp(g),
        float_to_u16_clamp(b),
        65535u
    );

    int dst_idx;
    if (orientation == 3) {
        dst_idx = (out_h - 1 - row) * out_w + (out_w - 1 - column);
    } else if (orientation == 6) {
        // dst_w = out_h, dx = out_h - 1 - row, dy = column
        dst_idx = column * out_h + (out_h - 1 - row);
    } else if (orientation == 8) {
        // dst_w = out_h, dx = row, dy = out_w - 1 - column
        dst_idx = (out_w - 1 - column) * out_h + row;
    } else {
        dst_idx = row * out_w + column;
    }
    output[dst_idx] = packed;
}
