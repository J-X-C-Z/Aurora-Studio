// Generic 2x2 Bayer (C1 Stage 1/2/3) and Fujifilm X-Trans 6x6 CUDA demosaic kernels.
//
// Compiled with:
//   nvcc --ptx -O3 --fmad=false --prec-div=true --prec-sqrt=true -arch=compute_75 -Xcompiler /wd4819 -o raw_demosaic.ptx raw_demosaic.cu
//
// Kernels:
//   Generic 2x2 Bayer (1:1 with aurora_raw::c1_raw_he Stage 1/2/3 + active mirror):
//     1. bayer_stage1_kernel              : shared-memory 11x11 directional green + black offset
//     2. bayer_stage2_scale_kernel        : 5x5 green refinement on R/B sites + WB channel scale
//     3. bayer_stage3_diagonal_kernel     : diagonal R/B reconstruction on B/R sites (in-place)
//     4. bayer_stage3_green_pack_kernel   : cross R/B reconstruction on G sites + pack [f32; 3]
//
//   Fujifilm X-Trans 6x6 (1:1 with aurora_raw::xtrans::demosaic_xtrans):
//     5. xtrans_normalize_kernel          : active crop + optional bit-shift + black/WB -> 16-bit
//     6. xtrans_green_interp_kernel       : directional green interpolation on non-G sites
//     7. xtrans_fix_green_kernel          : 3x3 cross min/max clamp on interpolated G sites
//     8. xtrans_chroma_rebuild_kernel     : 5x5 R-G / B-G chroma rebuild + pack [f32; 3]

// ============================================================================
// Common helpers
// ============================================================================

__device__ __forceinline__ int clamp_i(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ int abs_i(int v) {
    return v < 0 ? -v : v;
}

__device__ __forceinline__ int mirror_once(int coord, int size) {
    const int end = size - 1;
    int reflected = coord;
    if (coord < 0) {
        reflected = -coord;
    } else if (coord > end) {
        reflected = 2 * end - coord;
    }
    return clamp_i(reflected, 0, end);
}

__device__ __forceinline__ unsigned short normalize_to_14bit(
    unsigned short v,
    int shift_left,
    int shift_right
) {
    if (shift_left > 0) {
        unsigned int shifted = ((unsigned int)v) << shift_left;
        return (unsigned short)(shifted > 16383u ? 16383u : shifted);
    }
    if (shift_right > 0) {
        return (unsigned short)(v >> shift_right);
    }
    return v;
}

// ============================================================================
// Part A: Generic 2x2 Bayer C1 Stage 1/2/3 (RGGB-aligned active crop)
// ============================================================================

struct BayerDemosaicParams {
    int raw_stride;
    int src_left;
    int src_top;
    int width;
    int height;
    int shift_left;
    int shift_right;
    int rgb_black;
    float red_gain;
    float blue_gain;
    float noise_floor;
    float black_pivot;
    float wb_r;
    float wb_g;
    float wb_b;
    float stage3_offset;
};

__device__ __forceinline__ int stage1_predict(int word, int gain_q10, int bias) {
    return (int)(((unsigned int)(word * gain_q10)) >> 10) + bias;
}

__device__ __forceinline__ float stage1_interpolate_green(
    const unsigned short tile[26][26],
    int cy,
    int cx,
    int direction,
    float noise_floor,
    float black,
    float gain
) {
    const int center = (int)tile[cy][cx];

    float weight = 0.0f;
    float numerator = 0.0f;
    float denominator = 0.0f;

    if (direction == 1 || direction == 2) {
        const bool vertical = (direction == 2);
        const int near0 = vertical ? (int)tile[cy - 1][cx] : (int)tile[cy][cx - 1];
        const int near1 = vertical ? (int)tile[cy + 1][cx] : (int)tile[cy][cx + 1];
        const int far0  = vertical ? (int)tile[cy - 2][cx] : (int)tile[cy][cx - 2];
        const int far1  = vertical ? (int)tile[cy + 2][cx] : (int)tile[cy][cx + 2];

        const int dn = abs_i(near0 - near1);

        int w_neg_int, w_pos_int;
        int far_neg_diff, far_pos_diff;
        if (vertical) {
            w_neg_int = 1 + dn
                + abs_i((int)tile[cy - 2][cx - 1] - (int)tile[cy][cx - 1])
                + abs_i((int)tile[cy - 2][cx + 1] - (int)tile[cy][cx + 1])
                + abs_i((int)tile[cy - 3][cx] - (int)tile[cy - 1][cx]);
            w_pos_int = 1 + dn
                + abs_i((int)tile[cy + 2][cx - 1] - (int)tile[cy][cx - 1])
                + abs_i((int)tile[cy + 2][cx + 1] - (int)tile[cy][cx + 1])
                + abs_i((int)tile[cy + 3][cx] - (int)tile[cy + 1][cx]);
            far_neg_diff = abs_i(far0 - center);
            far_pos_diff = abs_i(far1 - center);
        } else {
            w_neg_int = 1 + dn
                + abs_i((int)tile[cy - 1][cx - 2] - (int)tile[cy - 1][cx])
                + abs_i((int)tile[cy + 1][cx - 2] - (int)tile[cy + 1][cx])
                + abs_i((int)tile[cy][cx - 3] - (int)tile[cy][cx - 1]);
            w_pos_int = 1 + dn
                + abs_i((int)tile[cy - 1][cx + 2] - (int)tile[cy - 1][cx])
                + abs_i((int)tile[cy + 1][cx + 2] - (int)tile[cy + 1][cx])
                + abs_i((int)tile[cy][cx + 3] - (int)tile[cy][cx + 1]);
            far_neg_diff = abs_i(far0 - center);
            far_pos_diff = abs_i(far1 - center);
        }

        const float negative = (float)w_neg_int + (float)far_neg_diff * gain;
        const float positive = (float)w_pos_int + (float)far_pos_diff * gain;
        const float mid0 = (float)((far0 + center) >> 1);
        const float mid1 = (float)((far1 + center) >> 1);

        weight = positive + negative;
        numerator = (float)near0 * positive + (float)near1 * negative;
        denominator = mid1 * negative + mid0 * positive;
    } else {
        // Horizontal axis
        const int x_near0 = (int)tile[cy][cx - 1];
        const int x_near1 = (int)tile[cy][cx + 1];
        const int x_far0  = (int)tile[cy][cx - 2];
        const int x_far1  = (int)tile[cy][cx + 2];
        const int x_dn    = abs_i(x_near0 - x_near1);
        const int x_neg_int = 1 + x_dn
            + abs_i((int)tile[cy - 1][cx - 2] - (int)tile[cy - 1][cx])
            + abs_i((int)tile[cy + 1][cx - 2] - (int)tile[cy + 1][cx])
            + abs_i((int)tile[cy][cx - 3] - (int)tile[cy][cx - 1]);
        const int x_pos_int = 1 + x_dn
            + abs_i((int)tile[cy - 1][cx + 2] - (int)tile[cy - 1][cx])
            + abs_i((int)tile[cy + 1][cx + 2] - (int)tile[cy + 1][cx])
            + abs_i((int)tile[cy][cx + 3] - (int)tile[cy][cx + 1]);
        const float x_neg = (float)x_neg_int + (float)abs_i(x_far0 - center) * gain;
        const float x_pos = (float)x_pos_int + (float)abs_i(x_far1 - center) * gain;
        const float x_mid0 = (float)((x_far0 + center) >> 1);
        const float x_mid1 = (float)((x_far1 + center) >> 1);

        // Vertical axis
        const int y_near0 = (int)tile[cy - 1][cx];
        const int y_near1 = (int)tile[cy + 1][cx];
        const int y_far0  = (int)tile[cy - 2][cx];
        const int y_far1  = (int)tile[cy + 2][cx];
        const int y_dn    = abs_i(y_near0 - y_near1);
        const int y_neg_int = 1 + y_dn
            + abs_i((int)tile[cy - 2][cx - 1] - (int)tile[cy][cx - 1])
            + abs_i((int)tile[cy - 2][cx + 1] - (int)tile[cy][cx + 1])
            + abs_i((int)tile[cy - 3][cx] - (int)tile[cy - 1][cx]);
        const int y_pos_int = 1 + y_dn
            + abs_i((int)tile[cy + 2][cx - 1] - (int)tile[cy][cx - 1])
            + abs_i((int)tile[cy + 2][cx + 1] - (int)tile[cy][cx + 1])
            + abs_i((int)tile[cy + 3][cx] - (int)tile[cy + 1][cx]);
        const float y_neg = (float)y_neg_int + (float)abs_i(y_far0 - center) * gain;
        const float y_pos = (float)y_pos_int + (float)abs_i(y_far1 - center) * gain;
        const float y_mid0 = (float)((y_far0 + center) >> 1);
        const float y_mid1 = (float)((y_far1 + center) >> 1);

        const float x_pair = x_pos * x_neg;
        const float y_pair = y_pos * y_neg;
        const float left  = y_pair * x_pos;
        const float right = y_pair * x_neg;
        const float up    = x_pair * y_pos;
        const float down  = x_pair * y_neg;

        weight = left + right + up + down;
        numerator = (float)y_near1 * down + (float)y_near0 * up
            + (float)x_near0 * left + (float)x_near1 * right;
        denominator = y_mid1 * down + y_mid0 * up + x_mid0 * left + x_mid1 * right;
    }

    const float floor_v = weight * noise_floor;
    const float black_weight = weight * black;
    const float num_c = fmaxf(floor_v, numerator + floor_v - black_weight);
    const float den_c = fmaxf(floor_v, denominator + floor_v - black_weight);
    return (num_c / den_c) * ((float)center - black + noise_floor) - noise_floor + black;
}

// Kernel 1: Stage 1 directional green + RGB32F black subtraction.
// Grid covers (width + 16) x (height + 16), corresponding to active crop coords [-8, width+8) x [-8, height+8).
extern "C" __global__ void bayer_stage1_kernel(
    const unsigned short* __restrict__ raw,
    float4* __restrict__ dst_stage1,
    BayerDemosaicParams p
) {
    __shared__ unsigned short tile[26][26];

    const int stage1_w = p.width + 16;
    const int stage1_h = p.height + 16;

    const int block_gx0 = (int)(blockIdx.x * blockDim.x);
    const int block_gy0 = (int)(blockIdx.y * blockDim.y);

    // Cooperative load of 26x26 Bayer patch (16x16 block + 5-pixel halo on each side)
    const int tid = (int)(threadIdx.y * blockDim.x + threadIdx.x);
    const int total_tile = 26 * 26;
    for (int i = tid; i < total_tile; i += 256) {
        const int ty = i / 26;
        const int tx = i - ty * 26;
        // Active crop coordinate: gx - 8, gy - 8; tile offset -5
        const int crop_x = block_gx0 + tx - 13;
        const int crop_y = block_gy0 + ty - 13;
        const int sx = mirror_once(crop_x, p.width);
        const int sy = mirror_once(crop_y, p.height);
        const unsigned short s14 = normalize_to_14bit(
            raw[(p.src_top + sy) * p.raw_stride + p.src_left + sx],
            p.shift_left,
            p.shift_right
        );
        tile[ty][tx] = (unsigned short)(s14 << 2);
    }
    __syncthreads();

    const int gx = block_gx0 + (int)threadIdx.x;
    const int gy = block_gy0 + (int)threadIdx.y;
    if (gx >= stage1_w || gy >= stage1_h) return;

    const int cx = (int)threadIdx.x + 5;
    const int cy = (int)threadIdx.y + 5;
    const bool odd_row = (gy & 1) != 0;
    const bool odd_col = (gx & 1) != 0;

    const float center_f = (float)tile[cy][cx];
    const float black_offset = -(float)p.rgb_black;

    if (odd_row != odd_col) {
        dst_stage1[gy * stage1_w + gx] = make_float4(
            0.0f + black_offset,
            center_f + black_offset,
            0.0f + black_offset,
            0.0f
        );
        return;
    }

    const float gain = odd_row ? p.blue_gain : p.red_gain;
    const float black = p.black_pivot;

    const int center = (int)tile[cy][cx];
    const float gain_scaled = gain * 1024.0f;
    const int gain_q10 = (int)gain_scaled;
    const int bias = (int)(black - black * gain);
    const int curvature_gain = (int)(gain_scaled * 1.6f);

    const int pred_center = stage1_predict(center, gain_q10, bias);
    const int ref0 = (int)tile[cy][cx - 1] - pred_center;
    const int ref1 = (int)tile[cy][cx + 1] - pred_center;
    const int ref2 = (int)tile[cy - 1][cx] - pred_center;
    const int ref3 = (int)tile[cy + 1][cx] - pred_center;

    const unsigned int h_init =
        ((unsigned int)(abs_i((int)tile[cy][cx - 2] + (int)tile[cy][cx + 2] - 2 * center) * curvature_gain) >> 10)
        + (unsigned int)abs_i((int)tile[cy - 1][cx] - (int)tile[cy - 1][cx - 2])
        + (unsigned int)abs_i((int)tile[cy - 1][cx] - (int)tile[cy - 1][cx + 2])
        + (unsigned int)abs_i((int)tile[cy + 1][cx] - (int)tile[cy + 1][cx - 2])
        + (unsigned int)abs_i((int)tile[cy + 1][cx] - (int)tile[cy + 1][cx + 2]);

    const unsigned int v_init =
        ((unsigned int)(abs_i((int)tile[cy - 2][cx] + (int)tile[cy + 2][cx] - 2 * center) * curvature_gain) >> 10)
        + (unsigned int)abs_i((int)tile[cy][cx - 1] - (int)tile[cy - 2][cx - 1])
        + (unsigned int)abs_i((int)tile[cy][cx - 1] - (int)tile[cy + 2][cx - 1])
        + (unsigned int)abs_i((int)tile[cy][cx + 1] - (int)tile[cy - 2][cx + 1])
        + (unsigned int)abs_i((int)tile[cy][cx + 1] - (int)tile[cy + 2][cx + 1]);

    unsigned int c0 = h_init;
    unsigned int c1 = h_init;
    unsigned int c2 = v_init;
    unsigned int c3 = v_init;
    unsigned int c4 = h_init;
    unsigned int c5 = v_init;

    // offset in {-4, 4}
    #pragma unroll
    for (int oi = 0; oi < 2; ++oi) {
        const int off = (oi == 0) ? -4 : 4;
        // (base_row = 0, base_col = off, first = 0)
        {
            const int target = stage1_predict((int)tile[cy][cx + off], gain_q10, bias);
            const int s0 = (int)tile[cy][cx + off - 1];
            const int s1 = (int)tile[cy][cx + off + 1];
            c0 += (unsigned int)abs_i(s0 - target - ref0);
            c1 += (unsigned int)abs_i(s1 - target - ref1);
            c4 += (unsigned int)abs_i(s0 - s1);
        }
        // (base_row = off, base_col = 0, first = 2)
        {
            const int target = stage1_predict((int)tile[cy + off][cx], gain_q10, bias);
            const int s0 = (int)tile[cy + off - 1][cx];
            const int s1 = (int)tile[cy + off + 1][cx];
            c2 += (unsigned int)abs_i(s0 - target - ref2);
            c3 += (unsigned int)abs_i(s1 - target - ref3);
            c5 += (unsigned int)abs_i(s0 - s1);
        }
    }

    // (row, col) in {-2, 0, 2} x {-2, 0, 2}
    #pragma unroll
    for (int dr = -2; dr <= 2; dr += 2) {
        #pragma unroll
        for (int dc = -2; dc <= 2; dc += 2) {
            const int target = stage1_predict((int)tile[cy + dr][cx + dc], gain_q10, bias);
            const int n0 = (int)tile[cy + dr][cx + dc - 1];
            const int n1 = (int)tile[cy + dr][cx + dc + 1];
            const int n2 = (int)tile[cy + dr - 1][cx + dc];
            const int n3 = (int)tile[cy + dr + 1][cx + dc];
            c0 += (unsigned int)abs_i(n0 - target - ref0);
            c1 += (unsigned int)abs_i(n1 - target - ref1);
            c2 += (unsigned int)abs_i(n2 - target - ref2);
            c3 += (unsigned int)abs_i(n3 - target - ref3);
            c4 += (unsigned int)abs_i(n0 - n1);
            c5 += (unsigned int)abs_i(n2 - n3);
        }
    }

    // early_direction (auxiliary_difference == 0.0 < 0.1)
    const unsigned int h_min = c0 < c1 ? c0 : c1;
    const unsigned int h_max = c0 > c1 ? c0 : c1;
    const unsigned int v_min = c2 < c3 ? c2 : c3;
    const unsigned int v_max = c2 > c3 ? c2 : c3;

    int direction = 0;
    if (h_max < v_min && c4 < c5) {
        direction = 1;
    } else if (v_max < h_min && c5 < c4) {
        direction = 2;
    }

    // low_curvature_direction
    if (direction == 0) {
        const int left  = (int)tile[cy][cx - 1];
        const int right = (int)tile[cy][cx + 1];
        const int up    = (int)tile[cy - 1][cx];
        const int down  = (int)tile[cy + 1][cx];
        const int curvature = abs_i(right + left - down - up) / 2;
        const int signal = right + left + down + up - (int)(black * 4.0f);
        if (curvature * curvature < signal) {
            direction = 3;
        }
    }

    // extended_costs + extended_direction
    if (direction == 0) {
        unsigned int e0 = 0u, e1 = 0u, e2 = 0u, e3 = 0u;
        for (int dr = -4; dr <= 4; dr += 2) {
            for (int dc = -4; dc <= 4; dc += 2) {
                const int target = stage1_predict((int)tile[cy + dr][cx + dc], gain_q10, bias);
                e0 += (unsigned int)abs_i((int)tile[cy + dr][cx + dc - 1] - target - ref0);
                e1 += (unsigned int)abs_i((int)tile[cy + dr][cx + dc + 1] - target - ref1);
                e2 += (unsigned int)abs_i((int)tile[cy + dr - 1][cx + dc] - target - ref2);
                e3 += (unsigned int)abs_i((int)tile[cy + dr + 1][cx + dc] - target - ref3);
            }
        }
        const unsigned int eh_min = e0 < e1 ? e0 : e1;
        const unsigned int eh_max = e0 > e1 ? e0 : e1;
        const unsigned int ev_min = e2 < e3 ? e2 : e3;
        const unsigned int ev_max = e2 > e3 ? e2 : e3;
        if (eh_max < ev_min) {
            direction = 1;
        } else if (ev_max < eh_min) {
            direction = 2;
        } else {
            direction = 3;
        }
    }

    const float green = stage1_interpolate_green(tile, cy, cx, direction, p.noise_floor, black, gain);
    const float r = (odd_row ? 0.0f : center_f) + black_offset;
    const float g = green + black_offset;
    const float b = (odd_row ? center_f : 0.0f) + black_offset;

    dst_stage1[gy * stage1_w + gx] = make_float4(r, g, b, 0.0f);
}

// Kernel 2: Stage 2 basic green refinement on R/B sites + Stage 3 channel WB scaling.
// Grid covers (width + 12) x (height + 12), reading stage1 at (gx + 2, gy + 2).
extern "C" __global__ void bayer_stage2_scale_kernel(
    const float4* __restrict__ src_stage1,
    float4* __restrict__ dst_stage2,
    BayerDemosaicParams p
) {
    const int stage2_w = p.width + 12;
    const int stage2_h = p.height + 12;
    const int gx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int gy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (gx >= stage2_w || gy >= stage2_h) return;

    const int s1_stride = p.width + 16;
    const int sx = gx + 2;
    const int sy = gy + 2;

    float4 pixel = src_stage1[sy * s1_stride + sx];
    const bool odd_row = (gy & 1) != 0;
    const bool odd_col = (gx & 1) != 0;

    if (odd_row == odd_col) {
        auto green_at = [&](int dr, int dc) -> float {
            return src_stage1[(sy + dr) * s1_stride + (sx + dc)].y;
        };
        auto primary_at = [&](int dr, int dc) -> float {
            const float4 px = src_stage1[(sy + dr) * s1_stride + (sx + dc)];
            return odd_row ? px.z : px.x;
        };
        auto opposite_at = [&](int dr, int dc) -> float {
            const float4 px = src_stage1[(sy + dr) * s1_stride + (sx + dc)];
            return odd_row ? px.x : px.z;
        };

        const float center = pixel.y;
        const float up    = green_at(-1, 0);
        const float down  = green_at( 1, 0);
        const float left  = green_at( 0, -1);
        const float right = green_at( 0,  1);
        const float ul    = green_at(-1, -1);
        const float ur    = green_at(-1,  1);
        const float dl    = green_at( 1, -1);
        const float dr    = green_at( 1,  1);
        const float uul   = green_at(-2, -1);
        const float uur   = green_at(-2,  1);
        const float ddl   = green_at( 2, -1);
        const float ddr   = green_at( 2,  1);
        const float ull   = green_at(-1, -2);
        const float urr   = green_at(-1,  2);
        const float dll   = green_at( 1, -2);
        const float drr   = green_at( 1,  2);

        const float chroma = odd_row ? pixel.z : pixel.x;
        const float vertical = fabsf(down - center) + fabsf(up - center);
        const float horizontal = fabsf(right - center) + fabsf(left - center);

        const float diagonal_back =
            fabsf(primary_at(2, -2) - chroma) + fabsf(primary_at(-2, 2) - chroma)
            + fabsf(opposite_at(1, -1) - opposite_at(-1, 1))
            + fabsf(up - left) + fabsf(down - right);
        const float diagonal_slash =
            fabsf(primary_at(2, 2) - chroma) + fabsf(primary_at(-2, -2) - chroma)
            + fabsf(opposite_at(-1, -1) - opposite_at(1, 1))
            + fabsf(up - right) + fabsf(down - left);

        const float g0 = diagonal_back;
        const float g1 = diagonal_slash;
        const float g2 = fabsf(ddl - center) + fabsf(uur - center) + fabsf(up - dl) + fabsf(down - ur);
        const float g3 = fabsf(dll - center) + fabsf(urr - center) + fabsf(left - ur) + fabsf(right - dl);
        const float g4 = fabsf(ddr - center) + fabsf(uul - center) + fabsf(down - ul) + fabsf(up - dr);
        const float g5 = fabsf(drr - center) + fabsf(ull - center) + fabsf(right - ul) + fabsf(left - dr);

        const float axial = fminf(vertical, horizontal);
        float minimum = axial;
        minimum = fminf(minimum, g0);
        minimum = fminf(minimum, g1);
        minimum = fminf(minimum, g2);
        minimum = fminf(minimum, g3);
        minimum = fminf(minimum, g4);
        minimum = fminf(minimum, g5);

        if (!(axial <= minimum)) {
            const float one_third = __int_as_float(0x3eaaaaab);
            const float axial_average = (vertical < horizontal)
                ? (((center + up) + down) * one_third)
                : (((center + left) + right) * one_third);

            float cand;
            if (g5 <= minimum) {
                cand = ((center + ull) + drr) * one_third;
            } else if (g4 <= minimum) {
                cand = ((center + uul) + ddr) * one_third;
            } else if (g3 <= minimum) {
                cand = ((dll + center) + urr) * one_third;
            } else if (g2 <= minimum) {
                cand = ((ddl + center) + uur) * one_third;
            } else if (g1 <= minimum) {
                cand = ((2.0f * center + ul) + dr) * 0.25f;
            } else {
                cand = ((2.0f * center + ur) + dl) * 0.25f;
            }
            pixel.y = (cand + axial_average) * 0.5f;
        }
    }

    dst_stage2[gy * stage2_w + gx] = make_float4(
        pixel.x * p.wb_r,
        pixel.y * p.wb_g,
        pixel.z * p.wb_b,
        0.0f
    );
}

__device__ __forceinline__ float chroma_cost(float color, float green, bool enabled) {
    if (enabled) {
        const float denom = fmaxf((color + green) + 1.0f, 1.0f);
        return fabsf(color - green) * (1.0f / denom) + 0.03f;
    }
    return 1.0f;
}

// Kernel 3: Stage 3 diagonal R/B reconstruction on B/R sites (in-place on stage2_buf).
// Grid covers (width + 8) x (height + 8), corresponding to stage2 coords [2..width+10) x [2..height+10).
// Writes only to .z on even/even (R) sites and .x on odd/odd (B) sites; reads only opposite parity,
// so in-place update has zero data hazards.
extern "C" __global__ void bayer_stage3_diagonal_kernel(
    float4* __restrict__ buf_stage2,
    BayerDemosaicParams p
) {
    const int diag_w = p.width + 8;
    const int diag_h = p.height + 8;
    const int gx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int gy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (gx >= diag_w || gy >= diag_h) return;

    const int row = gy + 2;
    const int col = gx + 2;
    const int parity = row & 1;
    if ((col & 1) != parity) return;

    const int stride = p.width + 12;
    const float4 ul = buf_stage2[(row - 1) * stride + (col - 1)];
    const float4 ur = buf_stage2[(row - 1) * stride + (col + 1)];
    const float4 dl = buf_stage2[(row + 1) * stride + (col - 1)];
    const float4 dr = buf_stage2[(row + 1) * stride + (col + 1)];
    const float4 ctr = buf_stage2[row * stride + col];

    // parity == 0 (R site): interpolate B (.z), guide is R (.x)
    // parity == 1 (B site): interpolate R (.x), guide is B (.z)
    const float pul = (parity == 0) ? ul.z : ul.x;
    const float pur = (parity == 0) ? ur.z : ur.x;
    const float pdl = (parity == 0) ? dl.z : dl.x;
    const float pdr = (parity == 0) ? dr.z : dr.x;

    const float gul = ul.y;
    const float gur = ur.y;
    const float gdl = dl.y;
    const float gdr = dr.y;

    const bool selected = fabsf(pul - pdr) < fabsf(pdl - pur);
    const bool enabled = ((pdr + pdl) + (pur + pul)) * 0.75f < (gdr + gdl) + (gur + gul);

    const float center_g = ctr.y;
    const float guide_center = (parity == 0) ? ctr.x : ctr.z;

    float c_first, c_second;
    float g_first, g_second;
    float gf_first, gf_second;

    if (selected) {
        // [0, 3] -> (-1, -1) and (+1, +1)
        c_first = pul;
        c_second = pdr;
        g_first = gul;
        g_second = gdr;
        const float4 far0 = buf_stage2[(row - 2) * stride + (col - 2)];
        const float4 far3 = buf_stage2[(row + 2) * stride + (col + 2)];
        gf_first  = (parity == 0) ? far0.x : far0.z;
        gf_second = (parity == 0) ? far3.x : far3.z;
    } else {
        // [2, 1] -> (+1, -1) and (-1, +1)
        c_first = pdl;
        c_second = pur;
        g_first = gdl;
        g_second = gur;
        const float4 far2 = buf_stage2[(row + 2) * stride + (col - 2)];
        const float4 far1 = buf_stage2[(row - 2) * stride + (col + 2)];
        gf_first  = (parity == 0) ? far2.x : far2.z;
        gf_second = (parity == 0) ? far1.x : far1.z;
    }

    const float norm_first  = fabsf(gf_first - guide_center) + (fabsf(g_first - center_g) + 1.0f);
    const float norm_second = fabsf(gf_second - guide_center) + (fabsf(g_second - center_g) + 1.0f);
    const float first_cost  = chroma_cost(c_first, g_first, enabled) * norm_first;
    const float second_cost = chroma_cost(c_second, g_second, enabled) * norm_second;

    const float offset = p.stage3_offset;
    const float floor_v = (second_cost + first_cost) * offset;
    const float denominator = (g_second * first_cost + second_cost * g_first) + floor_v;
    const float numerator   = (c_second * first_cost + second_cost * c_first) + floor_v;
    const float output = ((1.0f / fmaxf(denominator, floor_v)) * numerator) * (offset + center_g) - offset;

    if (parity == 0) {
        buf_stage2[row * stride + col].z = output;
    } else {
        buf_stage2[row * stride + col].x = output;
    }
}

// Kernel 4: Stage 3 cross R/B reconstruction on G sites + pack tightly to [f32; 3].
// Grid covers width x height, reading stage2 at (x + 6, y + 6).
extern "C" __global__ void bayer_stage3_green_pack_kernel(
    const float4* __restrict__ buf_stage2,
    float* __restrict__ dst_rgb3,
    BayerDemosaicParams p
) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= p.width || y >= p.height) return;

    const int stride = p.width + 12;
    const int row = y + 6;
    const int col = x + 6;
    const float4 ctr = buf_stage2[row * stride + col];
    const int out_base = (y * p.width + x) * 3;

    if ((x & 1) == (y & 1)) {
        dst_rgb3[out_base]     = ctr.x;
        dst_rgb3[out_base + 1] = ctr.y;
        dst_rgb3[out_base + 2] = ctr.z;
        return;
    }

    const float4 u  = buf_stage2[(row - 1) * stride + col];
    const float4 d  = buf_stage2[(row + 1) * stride + col];
    const float4 l  = buf_stage2[row * stride + (col - 1)];
    const float4 r  = buf_stage2[row * stride + (col + 1)];

    const float gu = u.y;
    const float gd = d.y;
    const float gl = l.y;
    const float gr = r.y;
    const float center = ctr.y;

    const float vertical   = fabsf(gu - gd) + 1.0f;
    const float horizontal = fabsf(gr - gl) + 1.0f;

    const float gu2 = buf_stage2[(row - 2) * stride + col].y;
    const float gd2 = buf_stage2[(row + 2) * stride + col].y;
    const float gl2 = buf_stage2[row * stride + (col - 2)].y;
    const float gr2 = buf_stage2[row * stride + (col + 2)].y;

    const float norm_u = fabsf(gu2 - center) + vertical;
    const float norm_d = fabsf(gd2 - center) + vertical;
    const float norm_l = fabsf(gl2 - center) + horizontal;
    const float norm_r = fabsf(gr2 - center) + horizontal;

    const float green_sum = (gu + gd) + (gr + gl);
    const float offset = p.stage3_offset;

    auto eval_channel = [&](float pu, float pd, float pl, float pr) -> float {
        const bool enabled = ((pr + pl) + (pd + pu)) * 0.75f < green_sum;
        const float cu = chroma_cost(pu, gu, enabled) * norm_u;
        const float cd = chroma_cost(pd, gd, enabled) * norm_d;
        const float cl = chroma_cost(pl, gl, enabled) * norm_l;
        const float cr = chroma_cost(pr, gr, enabled) * norm_r;
        const float wu = cd * (cl * cr);
        const float wd = cu * (cl * cr);
        const float wl = cr * (cd * cu);
        const float wr = cl * (cd * cu);
        const float floor_v = ((wl + wr) + (wd + wu)) * offset;
        const float numerator = ((wd * pd + wu * pu) + floor_v) + (wr * pr + wl * pl);
        const float denominator = ((gu * wu + wd * gd) + floor_v) + (wr * gr + wl * gl);
        return ((1.0f / fmaxf(denominator, floor_v)) * numerator) * (center + offset) - offset;
    };

    const float out_r = eval_channel(u.x, d.x, l.x, r.x);
    const float out_b = eval_channel(u.z, d.z, l.z, r.z);

    dst_rgb3[out_base]     = out_r;
    dst_rgb3[out_base + 1] = center;
    dst_rgb3[out_base + 2] = out_b;
}

// ============================================================================
// Part B: Fujifilm X-Trans 6x6 CFA Demosaic (1:1 with aurora_raw::xtrans)
// ============================================================================

struct XTransDemosaicParams {
    int raw_stride;
    int left;
    int top;
    int width;
    int height;
    int shift_left;
    int shift_right;
    float black;
    float wb_r;
    float wb_g;
    float wb_b;
};

__device__ __forceinline__ int xtrans_color(int x, int y) {
    // Standard Fujifilm X-Trans 6x6 pattern (0=R, 1=G, 2=B):
    //   [1, 2, 1, 1, 0, 1]
    //   [0, 1, 0, 2, 1, 2]
    //   [1, 2, 1, 1, 0, 1]
    //   [1, 0, 1, 1, 2, 1]
    //   [2, 1, 2, 0, 1, 0]
    //   [1, 0, 1, 1, 2, 1]
    static const unsigned char lut[36] = {
        1, 2, 1, 1, 0, 1,
        0, 1, 0, 2, 1, 2,
        1, 2, 1, 1, 0, 1,
        1, 0, 1, 1, 2, 1,
        2, 1, 2, 0, 1, 0,
        1, 0, 1, 1, 2, 1
    };
    return (int)lut[(y % 6) * 6 + (x % 6)];
}

// Kernel 5: X-Trans Stage 0 (black subtraction + 14->16 bit scale + WB + clamp 65535)
extern "C" __global__ void xtrans_normalize_kernel(
    const unsigned short* __restrict__ raw,
    float* __restrict__ dst_plane,
    XTransDemosaicParams p
) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= p.width || y >= p.height) return;

    const unsigned short s14 = normalize_to_14bit(
        raw[(p.top + y) * p.raw_stride + p.left + x],
        p.shift_left,
        p.shift_right
    );
    const int c = xtrans_color(x, y);
    const float wb = (c == 0) ? p.wb_r : ((c == 1) ? p.wb_g : p.wb_b);
    const float v = fmaxf((float)s14 - p.black, 0.0f) * 4.0f * wb;
    dst_plane[y * p.width + x] = fminf(v, 65535.0f);
}

// Kernel 6: X-Trans Stage 1 (directional green interpolation on non-G sites)
extern "C" __global__ void xtrans_green_interp_kernel(
    const float* __restrict__ plane,
    float* __restrict__ dst_green,
    int width,
    int height
) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    if (xtrans_color(x, y) == 1) {
        dst_green[idx] = plane[idx];
        return;
    }

    const int max_x = width - 1;
    const int max_y = height - 1;
    auto at = [&](int sx, int sy) -> float {
        return plane[clamp_i(sy, 0, max_y) * width + clamp_i(sx, 0, max_x)];
    };

    float sum_h = 0.0f, cnt_h = 0.0f;
    float sum_v = 0.0f, cnt_v = 0.0f;
    float sum_d1 = 0.0f, cnt_d1 = 0.0f;
    float sum_d2 = 0.0f, cnt_d2 = 0.0f;

    #pragma unroll
    for (int di = 0; di < 2; ++di) {
        const int d = (di == 0) ? -1 : 1;
        const int nx = clamp_i(x + d, 0, max_x);
        const int ny = clamp_i(y + d, 0, max_y);
        if (xtrans_color(nx, y) == 1) {
            sum_h += plane[y * width + nx];
            cnt_h += 1.0f;
        }
        if (xtrans_color(x, ny) == 1) {
            sum_v += plane[ny * width + x];
            cnt_v += 1.0f;
        }
        if (xtrans_color(nx, ny) == 1) {
            sum_d1 += plane[ny * width + nx];
            cnt_d1 += 1.0f;
        }
        const int ny2 = clamp_i(y - d, 0, max_y);
        if (xtrans_color(nx, ny2) == 1) {
            sum_d2 += plane[ny2 * width + nx];
            cnt_d2 += 1.0f;
        }
    }

    const float gh  = fabsf(at(x - 1, y) - at(x + 1, y)) + 1.0f;
    const float gv  = fabsf(at(x, y - 1) - at(x, y + 1)) + 1.0f;
    const float gd1 = fabsf(at(x - 1, y - 1) - at(x + 1, y + 1)) + 1.0f;
    const float gd2 = fabsf(at(x - 1, y + 1) - at(x + 1, y - 1)) + 1.0f;

    float num = 0.0f;
    float den = 0.0f;
    if (cnt_h > 0.0f) {
        const float w = 1.0f / gh;
        num += (sum_h / cnt_h) * w;
        den += w;
    }
    if (cnt_v > 0.0f) {
        const float w = 1.0f / gv;
        num += (sum_v / cnt_v) * w;
        den += w;
    }
    if (cnt_d1 > 0.0f) {
        const float w = 0.8f / gd1;
        num += (sum_d1 / cnt_d1) * w;
        den += w;
    }
    if (cnt_d2 > 0.0f) {
        const float w = 0.8f / gd2;
        num += (sum_d2 / cnt_d2) * w;
        den += w;
    }

    dst_green[idx] = (den > 0.0f) ? (num / den) : plane[idx];
}

// Kernel 7: X-Trans Stage 2 (fixGreen: 3x3 cross min/max clamp on interpolated G sites)
extern "C" __global__ void xtrans_fix_green_kernel(
    const float* __restrict__ green,
    float* __restrict__ dst_green_fixed,
    int width,
    int height
) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    const float c = green[idx];
    if (xtrans_color(x, y) == 1) {
        dst_green_fixed[idx] = c;
        return;
    }

    const int max_x = width - 1;
    const int max_y = height - 1;
    const float n0 = green[y * width + clamp_i(x - 1, 0, max_x)];
    const float n1 = green[y * width + clamp_i(x + 1, 0, max_x)];
    const float n2 = green[clamp_i(y - 1, 0, max_y) * width + x];
    const float n3 = green[clamp_i(y + 1, 0, max_y) * width + x];
    const float lo = fminf(fminf(n0, n1), fminf(n2, n3));
    const float hi = fmaxf(fmaxf(n0, n1), fmaxf(n2, n3));
    dst_green_fixed[idx] = fminf(fmaxf(c, lo), hi);
}

// Kernel 8: X-Trans Stage 3 (5x5 R-G and B-G chroma rebuild + pack [f32; 3])
extern "C" __global__ void xtrans_chroma_rebuild_kernel(
    const float* __restrict__ plane,
    const float* __restrict__ green_fixed,
    float* __restrict__ dst_rgb3,
    int width,
    int height
) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    const int max_x = width - 1;
    const int max_y = height - 1;
    const float g = green_fixed[idx];

    float sum_rg = 0.0f, wt_r = 0.0f;
    float sum_bg = 0.0f, wt_b = 0.0f;

    #pragma unroll
    for (int dy = -2; dy <= 2; ++dy) {
        const int ny = clamp_i(y + dy, 0, max_y);
        const int row_base = ny * width;
        #pragma unroll
        for (int dx = -2; dx <= 2; ++dx) {
            const int nx = clamp_i(x + dx, 0, max_x);
            const int c = xtrans_color(nx, ny);
            if (c == 1) continue;
            const float dist2 = (float)(dx * dx + dy * dy);
            const float w = 1.0f / (1.0f + dist2);
            const float diff = plane[row_base + nx] - green_fixed[row_base + nx];
            if (c == 0) {
                sum_rg += diff * w;
                wt_r += w;
            } else {
                sum_bg += diff * w;
                wt_b += w;
            }
        }
    }

    const int self_c = xtrans_color(x, y);
    float r;
    if (self_c == 0) {
        r = plane[idx];
    } else if (wt_r > 0.0f) {
        r = fminf(fmaxf(g + sum_rg / wt_r, 0.0f), 65535.0f);
    } else {
        r = g;
    }

    float b;
    if (self_c == 2) {
        b = plane[idx];
    } else if (wt_b > 0.0f) {
        b = fminf(fmaxf(g + sum_bg / wt_b, 0.0f), 65535.0f);
    } else {
        b = g;
    }

    const int out_base = idx * 3;
    dst_rgb3[out_base]     = r;
    dst_rgb3[out_base + 1] = fminf(fmaxf(g, 0.0f), 65535.0f);
    dst_rgb3[out_base + 2] = b;
}
