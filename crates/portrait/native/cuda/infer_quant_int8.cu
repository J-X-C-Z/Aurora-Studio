#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// Aurora Native Inference Engine — UInt8 Dynamic Quantization CUDA Kernels
//
// Targets eSamEncoderPreUInt8.onnx:
//   1. dynamic_quantize_linear_minmax_f32
//   2. dynamic_quantize_linear_apply_f32_to_u8
//   3. matmul_integer_u8_to_i32
//   4. conv_integer_u8_to_i32
//
// All kernels strictly use 2D grid (blockIdx.x, blockIdx.y) and 2D block
// (threadIdx.x, threadIdx.y) to match crate::cuda::Nvidia::launch_raw.
//
// Build: nvcc --ptx -O3 -arch=compute_75 (PTX only; the driver JITs it for
// Turing/Ampere/Ada/Blackwell). Equivalent NVRTC invocation when cl.exe is not
// installed on the build host:
//   nvrtc --gpu-architecture=compute_75 --std=c++14 --dopt=on -I<cuda headers>
// ============================================================================

namespace {

constexpr int DQL_REDUCE_BLOCK = 256;

// ---- MatMulInteger tile geometry (block = (16, 16) = 256 threads) -----------
// Each block owns one 64x64 output tile and streams K in 32-wide steps.
// Each thread computes a 4x4 register tile via unsigned byte dot-products
// (`dp4a.u32.u32`), so one instruction retires 4 int8 multiply-accumulates.
constexpr int MM_TILE = 64;       // tile edge (M and N)
constexpr int MM_TK = 32;         // K step
constexpr int MM_LD = MM_TK + 4;  // padded shared row stride in bytes (bank skew)
constexpr int MM_WORDS = MM_TILE * (MM_TK / 4);  // u32 words per staged tile

__device__ __forceinline__ unsigned char clamp_u8_f32(float v) {
    if (!(v > 0.0f)) {
        return 0;
    }
    if (v >= 255.0f) {
        return 255;
    }
    return static_cast<unsigned char>(v);
}

__device__ __forceinline__ int load_zp(
    const unsigned char* __restrict__ zp_ptr,
    int mode,
    int batch_idx,
    int axis_idx,
    int axis_len) {
    if (mode == 0 || zp_ptr == nullptr) {
        return 0;
    }
    if (mode == 1) {
        return static_cast<int>(zp_ptr[0]);
    }
    if (mode == 2) {
        return static_cast<int>(zp_ptr[axis_idx]);
    }
    // mode == 3: batched per-axis [batch, axis_len]
    return static_cast<int>(zp_ptr[static_cast<size_t>(batch_idx) * static_cast<size_t>(axis_len) +
                                   static_cast<size_t>(axis_idx)]);
}

// 4x unsigned byte dot-product with int32 accumulate (sm_61+).
// Inputs are raw byte quads; zero-point correction is applied by the caller as
//   sum((a-za)*(b-zb)) = sum(a*b) - za*sum(b) - zb*sum(a) + K*za*zb
// which keeps the inner loop a single `dp4a` instruction.
__device__ __forceinline__ int dp4a_u8(unsigned a, unsigned b, int c) {
    int d;
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ int byte_quad_sum(unsigned w) {
    return static_cast<int>(w & 0xFFu) + static_cast<int>((w >> 8) & 0xFFu) +
           static_cast<int>((w >> 16) & 0xFFu) + static_cast<int>(w >> 24);
}

}  // namespace

// ----------------------------------------------------------------------------
// 1a. DynamicQuantizeLinear — Stage 1: min/max reduction + scale & zero_point
//
// ONNX specification:
//   min = min(0.0f, min(x))
//   max = max(0.0f, max(x))
//   y_scale = (max - min) / 255.0f
//   y_zero_point = (uint8_t)clamp(rintf(-min / y_scale), 0, 255)
//
// Launch with grid=(1, 1), block=(256, 1).
// ----------------------------------------------------------------------------
extern "C" __global__ void dynamic_quantize_linear_minmax_f32(
    const float* __restrict__ x,
    int n,
    float* __restrict__ y_scale,
    unsigned char* __restrict__ y_zero_point) {
    __shared__ float s_min[DQL_REDUCE_BLOCK];
    __shared__ float s_max[DQL_REDUCE_BLOCK];

    const int tid = threadIdx.x;
    float local_min = 0.0f;
    float local_max = 0.0f;

    for (int i = tid; i < n; i += DQL_REDUCE_BLOCK) {
        const float v = x[i];
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
    }

    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();

    for (int stride = DQL_REDUCE_BLOCK >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + stride]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float min_val = fminf(0.0f, s_min[0]);
        const float max_val = fmaxf(0.0f, s_max[0]);
        const float scale = (max_val - min_val) / 255.0f;
        unsigned char zp = 0;
        if (scale > 0.0f) {
            zp = clamp_u8_f32(rintf(-min_val / scale));
        }
        *y_scale = scale;
        *y_zero_point = zp;
    }
}

// ----------------------------------------------------------------------------
// 1b. DynamicQuantizeLinear — Stage 2: quantize F32 -> U8
//
// ONNX specification:
//   y[i] = (uint8_t)clamp(rintf(x[i] / y_scale) + y_zero_point, 0, 255)
//
// Launch with grid=(min((n + 255) / 256, 65535), 1), block=(256, 1).
// ----------------------------------------------------------------------------
extern "C" __global__ void dynamic_quantize_linear_apply_f32_to_u8(
    const float* __restrict__ x,
    int n,
    const float* __restrict__ y_scale,
    const unsigned char* __restrict__ y_zero_point,
    unsigned char* __restrict__ y) {
    const float scale = *y_scale;
    const unsigned char zp_u8 = *y_zero_point;
    const float zp_f = static_cast<float>(zp_u8);
    // Use one correctly rounded division. Multiplying by a rounded reciprocal
    // changes the uint8 at half-integer boundaries and drifts in the SAM encoder.

    const int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        if (scale > 0.0f) {
            const float q = rintf(__fdiv_rn(x[idx], scale)) + zp_f;
            y[idx] = clamp_u8_f32(q);
        } else {
            y[idx] = zp_u8;
        }
    }
}

// ----------------------------------------------------------------------------
// 2. MatMulInteger — 2D / 3D / 4D Batched U8 x U8 -> I32 with optional ZP
//
// Computes:
//   Y[b, i, j] = sum_{k=0..K-1} ((int)A[ba, i, k] - zp_a) * ((int)B[bb, k, j] - zp_b)
//
// dp4a formulation (exact, integer):
//   S[i,j] = sum_k raw_a[i,k] * raw_b[k,j]        (dp4a, plain i32 accumulate)
//   Y[i,j] = S[i,j] - za[i]*sumB[j] - zb[j]*sumA[i] + K*za[i]*zb[j]
// `sumA` (per row, over all K) and `sumB` (per column, over all K) are reduced
// inside the same kernel while the tiles are staged into shared memory, so the
// decomposition needs no extra kernel launch and no extra VRAM.
//
// Batch indexing supports arbitrary 2D/3D/4D NumPy broadcast via (d0, d1):
//   b = blockIdx.y in [0, d0 * d1)
//   b0 = b / d1, b1 = b % d1
//   a_base = b0 * a_stride0 + b1 * a_stride1
//   b_base = b0 * b_stride0 + b1 * b_stride1
//   y_base = b * M * N
//
// Zero-point modes (a_zp_mode / b_zp_mode):
//   0 = none (default 0)
//   1 = scalar [1]
//   2 = per-row [M] (for A) / per-column [N] (for B)
//   3 = batched per-row [batch, M] (for A) / batched per-column [batch, N] (for B)
//
// Launch with:
//   block = (16, 16)
//   grid  = (ceil(M / 64) * ceil(N / 64), d0 * d1)
//
// The K-tile walk uses a grid-stride loop over 64x64 tiles, so a caller that
// still publishes the legacy 32x32-tile grid (4x more blocks) stays correct:
// surplus blocks retire immediately instead of re-computing.
// ----------------------------------------------------------------------------
extern "C" __global__ void matmul_integer_u8_to_i32(
    const unsigned char* __restrict__ a,
    const unsigned char* __restrict__ b,
    const unsigned char* __restrict__ a_zp,
    const unsigned char* __restrict__ b_zp,
    int* __restrict__ y,
    int m,
    int n,
    int k,
    int d0,
    int d1,
    long long a_stride0,
    long long a_stride1,
    long long b_stride0,
    long long b_stride1,
    int a_zp_mode,
    int b_zp_mode) {
    __shared__ __align__(16) unsigned char s_a[MM_TILE][MM_LD];
    __shared__ __align__(16) unsigned char s_b[MM_TILE][MM_LD];
    __shared__ int s_sum_a[MM_TILE];
    __shared__ int s_sum_b[MM_TILE];

    const int batch_idx = blockIdx.y;
    if (batch_idx >= d0 * d1) {
        return;
    }

    const int tiles_n = (n + MM_TILE - 1) / MM_TILE;
    const int tiles_m = (m + MM_TILE - 1) / MM_TILE;
    const int tile_total = tiles_m * tiles_n;

    const int b0 = batch_idx / d1;
    const int b1 = batch_idx - b0 * d1;
    const long long a_base = static_cast<long long>(b0) * a_stride0 +
                             static_cast<long long>(b1) * a_stride1;
    const long long b_base = static_cast<long long>(b0) * b_stride0 +
                             static_cast<long long>(b1) * b_stride1;
    const long long y_base = static_cast<long long>(batch_idx) *
                             static_cast<long long>(m) * static_cast<long long>(n);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;

    // Only reduce the cross terms that the zero-point modes actually require.
    const bool need_sum_a = (b_zp_mode != 0);  // for zb[j] * sumA[i]
    const bool need_sum_b = (a_zp_mode != 0);  // for za[i] * sumB[j]
    const bool need_cross = need_sum_a || need_sum_b;

    // 4-byte vector loads are legal when the innermost K run is 4-aligned.
    const bool a_vec = ((k & 3) == 0) && ((a_base & 3) == 0);

    for (int tile_linear = blockIdx.x; tile_linear < tile_total; tile_linear += gridDim.x) {
        const int tile_m = tile_linear / tiles_n;
        const int tile_n = tile_linear - tile_m * tiles_n;
        const int row_base = tile_m * MM_TILE;
        const int col_base = tile_n * MM_TILE;

        const unsigned char* __restrict__ a_b = a + a_base;
        const unsigned char* __restrict__ b_b = b + b_base;

        const int rows[4] = {4 * ty + 0, 4 * ty + 1, 4 * ty + 2, 4 * ty + 3};
        const int cols[4] = {4 * tx + 0, 4 * tx + 1, 4 * tx + 2, 4 * tx + 3};

        int za[4];
        int zb[4];
        for (int i = 0; i < 4; ++i) {
            const int r = row_base + rows[i];
            za[i] = (r < m) ? load_zp(a_zp, a_zp_mode, batch_idx, r, m) : 0;
            const int c = col_base + cols[i];
            zb[i] = (c < n) ? load_zp(b_zp, b_zp_mode, batch_idx, c, n) : 0;
        }

        int acc[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[i][j] = 0;
            }
        }

        if (need_cross) {
            if (tid < MM_TILE) {
                s_sum_a[tid] = 0;
                s_sum_b[tid] = 0;
            }
            __syncthreads();
        }

        for (int k0 = 0; k0 < k; k0 += MM_TK) {
            // ---- stage A tile: rows [row_base, +64), K [k0, k0+32) -------------
            // Word w = 4 consecutive K bytes of one row (contiguous in memory).
            for (int t = tid; t < MM_WORDS; t += 256) {
                const int row = t >> 3;          // 0..63
                const int seg = (t & 7) << 2;    // 0,4,...,28
                const int g_row = row_base + row;
                const int g_k = k0 + seg;
                unsigned word = 0;
                if (g_row < m) {
                    if (a_vec && (g_k + 4) <= k) {
                        word = *reinterpret_cast<const unsigned*>(
                            a_b + static_cast<long long>(g_row) * k + g_k);
                    } else {
                        for (int j = 0; j < 4; ++j) {
                            const int gk = g_k + j;
                            if (gk < k) {
                                word |= static_cast<unsigned>(
                                            a_b[static_cast<long long>(g_row) * k + gk])
                                        << (8 * j);
                            }
                        }
                    }
                }
                *reinterpret_cast<unsigned*>(&s_a[row][seg]) = word;
                if (need_sum_a) {
                    const int s = byte_quad_sum(word);
                    if (s != 0) {
                        atomicAdd(&s_sum_a[row], s);
                    }
                }
            }

            // ---- stage B tile transposed: cols [col_base, +64), K [k0, k0+32) --
            // Word w = 4 consecutive K bytes of one column (K-major stride n).
            for (int t = tid; t < MM_WORDS; t += 256) {
                const int col = t & 63;         // coalesced over columns
                const int seg = (t >> 6) << 2;  // 0,4,8,12
                const int g_col = col_base + col;
                const int g_k = k0 + seg;
                unsigned word = 0;
                if (g_col < n) {
                    if ((g_k + 4) <= k) {
                        const long long p = static_cast<long long>(g_k) *
                                                static_cast<long long>(n) + g_col;
                        const long long nn = static_cast<long long>(n);
                        word = static_cast<unsigned>(b_b[p]) |
                               (static_cast<unsigned>(b_b[p + nn]) << 8) |
                               (static_cast<unsigned>(b_b[p + 2 * nn]) << 16) |
                               (static_cast<unsigned>(b_b[p + 3 * nn]) << 24);
                    } else {
                        for (int j = 0; j < 4; ++j) {
                            const int gk = g_k + j;
                            if (gk < k) {
                                word |= static_cast<unsigned>(
                                            b_b[static_cast<long long>(gk) * n + g_col])
                                        << (8 * j);
                            }
                        }
                    }
                }
                *reinterpret_cast<unsigned*>(&s_b[col][seg]) = word;
                if (need_sum_b) {
                    const int s = byte_quad_sum(word);
                    if (s != 0) {
                        atomicAdd(&s_sum_b[col], s);
                    }
                }
            }

            __syncthreads();

            // ---- 4x4 register tile x 4-byte K quads, one dp4a per 4 MACs ------
            const int r0 = 4 * ty;
            const int c0 = 4 * tx;
#pragma unroll
            for (int kk = 0; kk < MM_TK; kk += 4) {
                unsigned aw[4];
                unsigned bw[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    aw[i] = *reinterpret_cast<const unsigned*>(&s_a[r0 + i][kk]);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    bw[j] = *reinterpret_cast<const unsigned*>(&s_b[c0 + j][kk]);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[i][j] = dp4a_u8(aw[i], bw[j], acc[i][j]);
                    }
                }
            }

            __syncthreads();
        }

        // ---- epilogue: zero-point correction + store -------------------------
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = rows[i];
            const int g_row = row_base + row;
            if (g_row >= m) {
                continue;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = cols[j];
                const int g_col = col_base + col;
                if (g_col >= n) {
                    continue;
                }
                long long v = acc[i][j];
                if (need_cross) {
                    if (b_zp_mode != 0) {
                        v -= static_cast<long long>(zb[j]) * s_sum_a[row];
                    }
                    if (a_zp_mode != 0) {
                        v -= static_cast<long long>(za[i]) * s_sum_b[col];
                    }
                    if (a_zp_mode != 0 && b_zp_mode != 0) {
                        v += static_cast<long long>(k) * za[i] * zb[j];
                    }
                }
                y[y_base + static_cast<long long>(g_row) * n + g_col] =
                    static_cast<int>(v);
            }
        }
        // s_sum_a / s_sum_b are re-initialised by the next tile iteration.
        __syncthreads();
    }
}

// ----------------------------------------------------------------------------
// 3. ConvInteger — General 2D NCHW U8 x U8 -> I32 with optional ZP
//
// Covers eSamEncoderPreUInt8.onnx:
//   - k=16, s=16, p=0     (patch embed, c_in=3, c_out=768)
//   - k=1,  s=1,  p=0     (pointwise, c_in=c_out=768)
//   - k=3,  s=1,  p=1
// as well as arbitrary kernel_shape, strides, pads, dilations, and groups.
//
// Per ONNX ConvInteger spec, out-of-bounds padded pixels equal x_zero_point, so
// ((int)x_pad - zp_x) == 0 and out-of-bounds taps contribute 0.
//
// dp4a formulation (exact, integer): raw byte products accumulate in i32 and the
// zero points are folded in afterwards over the taps that were actually inside:
//   Y = S - zp_x*Wsum - zp_w*Xsum + (taps*c_in_g)*zp_x*zp_w
// which keeps the hot loop free of per-byte subtraction/branching.
//
// Two concerns drive the schedule:
//   * taps are consumed 4 at a time: `wide` path packs 4 consecutive k_w taps of
//     one channel (needs dilation_w == 1 and 4-byte alignment on both sides);
//     the default path packs 4 consecutive input channels of one tap, with the
//     weights transposed into shared memory so the operand is one shared u32.
//   * input traffic: one block serves up to 4 output channels, so each input
//     byte is fetched once per 4 channels instead of once per channel. The
//     channel count per block is derived from gridDim.y, so a caller that still
//     launches the legacy one-channel-per-block grid keeps working unchanged.
//     Grouping is *per group* (ceil(C_out/group/4) blocks per group): a 4-channel
//     window must never straddle a group boundary, because every block derives
//     its input-channel window `ic_start` from the group of its first output
//     channel. C_out=6, group=2 therefore uses windows [0..3), [3..6).
//
// Weights are staged per block into shared memory when the slice fits
// CV_SHARED_BYTES; otherwise words are assembled from cached global reads.
//
// Launch with:
//   block = (256, 1)
//   grid  = (ceil_div(H_out * W_out, 256),
//            N * group * ceil_div(C_out / group, 4))              [4 oc / block]
//   grid  = (ceil_div(H_out * W_out, 256), N * C_out)             [legacy, 1 oc]
// ----------------------------------------------------------------------------
namespace {

constexpr int CV_SHARED_BYTES = 12 * 1024;
constexpr int CV_THREADS = 256;

// Build a u32 of 4 consecutive input channels for one tap, zero-filling the
// padding channels so partial channel groups still feed a whole dp4a.
__device__ __forceinline__ unsigned gather_x_quad(
    const unsigned char* __restrict__ x,
    long long base,
    int c_in_g,
    int ic,
    long long hw_in) {
    unsigned w = 0;
    for (int j = 0; j < 4; ++j) {
        const int c = ic + j;
        if (c < c_in_g) {
            w |= static_cast<unsigned>(x[base + static_cast<long long>(c) * hw_in]) << (8 * j);
        }
    }
    return w;
}

// s_w[o][tap][ic] with 4 consecutive channels of one tap packed into one u32.
template <int OCS>
__device__ __forceinline__ void cv_stage_channel(
    unsigned char* __restrict__ s_w,
    const unsigned char* __restrict__ w,
    long long w_oc_base,
    int c_in_g,
    long long kh_kw,
    int c_out,
    int oc0,
    int taps,
    int icp,
    int tid) {
    const int stride = taps * icp;
    const int total = OCS * stride;
    for (int idx = tid; idx < total; idx += CV_THREADS) {
        const int o = idx / stride;
        const int rem = idx - o * stride;
        const int tp = rem / icp;
        const int ic = rem - tp * icp;
        unsigned char v = 0;
        if ((oc0 + o) < c_out && ic < c_in_g) {
            v = w[w_oc_base + static_cast<long long>(o) * c_in_g * kh_kw +
                  static_cast<long long>(ic) * kh_kw + tp];
        }
        s_w[idx] = v;
    }
}

// s_w[o][ic][tap4] with 4 consecutive k_w taps of one channel packed per u32.
template <int OCS>
__device__ __forceinline__ void cv_stage_wide(
    unsigned char* __restrict__ s_w,
    const unsigned char* __restrict__ w,
    long long w_oc_base,
    int c_in_g,
    long long kh_kw,
    int c_out,
    int oc0,
    int taps,
    int taps4,
    int tid) {
    const int stride = c_in_g * taps4;
    const int total = OCS * stride;
    for (int idx = tid; idx < total; idx += CV_THREADS) {
        const int o = idx / stride;
        const int rem = idx - o * stride;
        const int ic = rem / taps4;
        const int tp = rem - ic * taps4;
        unsigned char v = 0;
        if ((oc0 + o) < c_out && tp < taps) {
            v = w[w_oc_base + static_cast<long long>(o) * c_in_g * kh_kw +
                  static_cast<long long>(ic) * kh_kw + tp];
        }
        s_w[idx] = v;
    }
}

// Accumulate OCS output channels of one output pixel.
//   WIDE = 4 consecutive k_w taps per dp4a; otherwise 4 consecutive channels.
template <int OCS, bool WIDE>
__device__ __forceinline__ void cv_body(
    const unsigned char* __restrict__ x,
    const unsigned char* __restrict__ w,
    const unsigned char* __restrict__ s_w,
    bool staged,
    int ocs,
    int c_in_g,
    int ic_start,
    int h_in,
    int w_in,
    long long hw_in,
    long long w_oc_base,
    long long kh_kw,
    int k_h,
    int k_w,
    int dilation_h,
    int dilation_w,
    int ih_base,
    int iw_base,
    int taps,
    int taps4,
    int icp,
    int* acc,
    int* sum_w,
    int& sum_x,
    int& n_incl) {
    const long long oc_step = static_cast<long long>(c_in_g) * kh_kw;
    // Tail block of a group covers fewer than OCS channels; clamping the loop
    // keeps the weight reads of the unused lanes out of the next group / past
    // the end of `w`.
    const int o_end = (ocs < OCS) ? ocs : OCS;

    if (WIDE) {
        const int full_chunks = k_w >> 2;
        for (int kh_i = 0; kh_i < k_h; ++kh_i) {
            const int ih = ih_base + kh_i * dilation_h;
            if (ih < 0 || ih >= h_in) {
                continue;
            }
            const long long x_row = static_cast<long long>(ih) * w_in;
            const bool row_inside =
                (iw_base >= 0) && (iw_base + k_w <= w_in) && ((iw_base & 3) == 0);
            for (int ic = 0; ic < c_in_g; ++ic) {
                const long long x_c = x_row + static_cast<long long>(ic_start + ic) * hw_in;
                const long long w_tap0 = static_cast<long long>(kh_i) * k_w;
                int kw_i = 0;
                if (row_inside) {
                    for (int ch = 0; ch < full_chunks; ++ch) {
                        const int iw = iw_base + 4 * ch;
                        const unsigned xw = *reinterpret_cast<const unsigned*>(&x[x_c + iw]);
                        sum_x += byte_quad_sum(xw);
                        const long long rel = w_tap0 + 4 * ch;
#pragma unroll
                        for (int o = 0; o < o_end; ++o) {
                            unsigned ww;
                            if (staged) {
                                ww = *reinterpret_cast<const unsigned*>(
                                    &s_w[(static_cast<long long>(o) * c_in_g + ic) * taps4 + rel]);
                            } else {
                                ww = *reinterpret_cast<const unsigned*>(
                                    &w[w_oc_base + static_cast<long long>(o) * oc_step +
                                       static_cast<long long>(ic) * kh_kw + rel]);
                            }
                            acc[o] = dp4a_u8(xw, ww, acc[o]);
                            sum_w[o] += byte_quad_sum(ww);
                        }
                    }
                    n_incl += full_chunks << 2;
                    kw_i = full_chunks << 2;
                }
                for (; kw_i < k_w; ++kw_i) {
                    const int iw = iw_base + kw_i;
                    if (iw < 0 || iw >= w_in) {
                        continue;
                    }
                    const int xv = static_cast<int>(x[x_c + iw]);
                    sum_x += xv;
#pragma unroll
                    for (int o = 0; o < o_end; ++o) {
                        const int wv = static_cast<int>(
                            w[w_oc_base + static_cast<long long>(o) * oc_step +
                              static_cast<long long>(ic) * kh_kw + w_tap0 + kw_i]);
                        acc[o] += xv * wv;
                        sum_w[o] += wv;
                    }
                    ++n_incl;
                }
            }
        }
    } else {
        const int ic_lim = c_in_g & ~3;
        for (int kh_i = 0; kh_i < k_h; ++kh_i) {
            const int ih = ih_base + kh_i * dilation_h;
            if (ih < 0 || ih >= h_in) {
                continue;
            }
            const long long x_row = static_cast<long long>(ih) * w_in;
            for (int kw_i = 0; kw_i < k_w; ++kw_i) {
                const int iw = iw_base + kw_i * dilation_w;
                if (iw < 0 || iw >= w_in) {
                    continue;
                }
                const long long tap = static_cast<long long>(kh_i) * k_w + kw_i;
                const long long x_tap = x_row + iw + static_cast<long long>(ic_start) * hw_in;
                int ic = 0;
                for (; ic < ic_lim; ic += 4) {
                    const unsigned xw = gather_x_quad(x, x_tap, c_in_g, ic, hw_in);
                    sum_x += byte_quad_sum(xw);
#pragma unroll
                    for (int o = 0; o < o_end; ++o) {
                        unsigned ww;
                        if (staged) {
                            ww = *reinterpret_cast<const unsigned*>(
                                &s_w[(static_cast<long long>(o) * taps + tap) * icp + ic]);
                        } else {
                            unsigned gw = 0;
                            for (int j = 0; j < 4; ++j) {
                                gw |= static_cast<unsigned>(
                                          w[w_oc_base + static_cast<long long>(o) * oc_step +
                                            static_cast<long long>(ic + j) * kh_kw + tap])
                                      << (8 * j);
                            }
                            ww = gw;
                        }
                        acc[o] = dp4a_u8(xw, ww, acc[o]);
                        sum_w[o] += byte_quad_sum(ww);
                    }
                }
                for (; ic < c_in_g; ++ic) {
                    const int xv = static_cast<int>(x[x_tap + static_cast<long long>(ic) * hw_in]);
                    sum_x += xv;
#pragma unroll
                    for (int o = 0; o < o_end; ++o) {
                        const int wv = static_cast<int>(
                            w[w_oc_base + static_cast<long long>(o) * oc_step +
                              static_cast<long long>(ic) * kh_kw + tap]);
                        acc[o] += xv * wv;
                        sum_w[o] += wv;
                    }
                    ++n_incl;
                }
                if (ic_lim > 0) {
                    n_incl += ic_lim;
                }
            }
        }
    }
}

}  // namespace

extern "C" __global__ void conv_integer_u8_to_i32(
    const unsigned char* __restrict__ x,
    const unsigned char* __restrict__ w,
    const unsigned char* __restrict__ x_zp,
    const unsigned char* __restrict__ w_zp,
    int* __restrict__ y,
    int batch_n,
    int c_in,
    int h_in,
    int w_in,
    int c_out,
    int k_h,
    int k_w,
    int h_out,
    int w_out,
    int stride_h,
    int stride_w,
    int pad_top,
    int pad_left,
    int dilation_h,
    int dilation_w,
    int group,
    int x_zp_mode,
    int w_zp_mode) {
    __shared__ __align__(16) unsigned char s_w[CV_SHARED_BYTES];

    const int hw_out = h_out * w_out;
    const int spatial_idx = blockIdx.x * CV_THREADS + threadIdx.x;
    const int noc = blockIdx.y;

    if (noc >= batch_n * c_out) {
        return;  // block-uniform, so retiring here is barrier-safe
    }

    // 4 output channels per block when the caller publishes that grid, blocked
    // *inside* each group so one block never mixes two groups' input channels.
    const int c_in_per_group = c_in / group;
    const int c_out_per_group = c_out / group;
    const int oc_blocks = (c_out_per_group + 3) >> 2;
    const bool oc4 = (static_cast<long long>(gridDim.y) ==
                      static_cast<long long>(batch_n) * static_cast<long long>(group) *
                          static_cast<long long>(oc_blocks));
    const int n_idx = oc4 ? (noc / (group * oc_blocks)) : (noc / c_out);
    const int lane = oc4 ? (noc - n_idx * group * oc_blocks) : (noc - n_idx * c_out);
    const int g = oc4 ? (lane / oc_blocks) : (lane / c_out_per_group);
    const int oc_local = oc4 ? ((lane - g * oc_blocks) << 2) : (lane - g * c_out_per_group);
    const int oc0 = g * c_out_per_group + oc_local;
    const int oc_left = c_out_per_group - oc_local;
    const int ocs = oc4 ? (oc_left < 4 ? oc_left : 4) : 1;

    const bool active = spatial_idx < hw_out;
    const int oh = active ? (spatial_idx / w_out) : 0;
    const int ow = active ? (spatial_idx - oh * w_out) : 0;

    const int ic_start = g * c_in_per_group;
    const long long hw_in = static_cast<long long>(h_in) * static_cast<long long>(w_in);

    const long long x_batch_base =
        static_cast<long long>(n_idx) * static_cast<long long>(c_in) * hw_in;
    const long long w_oc_base = static_cast<long long>(oc0) *
                                static_cast<long long>(c_in_per_group) *
                                static_cast<long long>(k_h) * static_cast<long long>(k_w);
    const long long kh_kw = static_cast<long long>(k_h) * static_cast<long long>(k_w);

    const int ih_base = oh * stride_h - pad_top;
    const int iw_base = ow * stride_w - pad_left;

    const int taps = k_h * k_w;
    const int taps4 = (taps + 3) & ~3;
    const int icp = (c_in_per_group + 3) & ~3;

    // 4-byte tap packing needs both sides naturally aligned.
    const bool wide_align = ((k_w & 3) == 0) && ((hw_in & 3) == 0) &&
                            ((static_cast<long long>(w_in) & 3) == 0) && ((kh_kw & 3) == 0) &&
                            ((reinterpret_cast<uintptr_t>(x) & 3u) == 0) &&
                            ((reinterpret_cast<uintptr_t>(w) & 3u) == 0);
    const bool wide = (dilation_w == 1) && (k_w >= 4) && wide_align;

    const long long need = wide
                               ? static_cast<long long>(ocs) * c_in_per_group * taps4
                               : static_cast<long long>(ocs) * taps * icp;
    const bool staged = need <= static_cast<long long>(CV_SHARED_BYTES);

    // Cooperative staging: every thread takes part, including the ones that will
    // retire below, because the barrier and the fill are block-wide.
    if (staged) {
        const int tid = threadIdx.x;
        if (oc4) {
            if (wide) {
                cv_stage_wide<4>(s_w, w, w_oc_base, c_in_per_group, kh_kw, c_out, oc0, taps,
                                 taps4, tid);
            } else {
                cv_stage_channel<4>(s_w, w, w_oc_base, c_in_per_group, kh_kw, c_out, oc0, taps,
                                    icp, tid);
            }
        } else {
            if (wide) {
                cv_stage_wide<1>(s_w, w, w_oc_base, c_in_per_group, kh_kw, c_out, oc0, taps,
                                 taps4, tid);
            } else {
                cv_stage_channel<1>(s_w, w, w_oc_base, c_in_per_group, kh_kw, c_out, oc0, taps,
                                    icp, tid);
            }
        }
        __syncthreads();
    }

    if (!active) {
        return;
    }

    const unsigned char* __restrict__ x_pix = x + x_batch_base;

    int acc[4] = {0, 0, 0, 0};
    int sum_w[4] = {0, 0, 0, 0};
    int sum_x = 0;
    int n_incl = 0;

    if (oc4) {
        if (wide) {
            cv_body<4, true>(x_pix, w, s_w, staged, ocs, c_in_per_group, ic_start, h_in, w_in, hw_in,
                             w_oc_base, kh_kw, k_h, k_w, dilation_h, dilation_w, ih_base, iw_base,
                             taps, taps4, icp, acc, sum_w, sum_x, n_incl);
        } else {
            cv_body<4, false>(x_pix, w, s_w, staged, ocs, c_in_per_group, ic_start, h_in, w_in, hw_in,
                              w_oc_base, kh_kw, k_h, k_w, dilation_h, dilation_w, ih_base, iw_base,
                              taps, taps4, icp, acc, sum_w, sum_x, n_incl);
        }
    } else {
        if (wide) {
            cv_body<1, true>(x_pix, w, s_w, staged, ocs, c_in_per_group, ic_start, h_in, w_in, hw_in,
                             w_oc_base, kh_kw, k_h, k_w, dilation_h, dilation_w, ih_base, iw_base,
                             taps, taps4, icp, acc, sum_w, sum_x, n_incl);
        } else {
            cv_body<1, false>(x_pix, w, s_w, staged, ocs, c_in_per_group, ic_start, h_in, w_in, hw_in,
                              w_oc_base, kh_kw, k_h, k_w, dilation_h, dilation_w, ih_base, iw_base,
                              taps, taps4, icp, acc, sum_w, sum_x, n_incl);
        }
    }

    const int zp_x = (x_zp_mode != 0 && x_zp != nullptr) ? static_cast<int>(x_zp[0]) : 0;

#pragma unroll
    for (int o = 0; o < 4; ++o) {
        if (o >= ocs) {
            break;
        }
        const int oc = oc0 + o;
        if (oc >= c_out) {
            break;
        }
        int zp_w = 0;
        if (w_zp_mode == 1 && w_zp != nullptr) {
            zp_w = static_cast<int>(w_zp[0]);
        } else if (w_zp_mode == 2 && w_zp != nullptr) {
            zp_w = static_cast<int>(w_zp[oc]);
        }

        long long v = acc[o];
        v -= static_cast<long long>(zp_x) * sum_w[o];
        v -= static_cast<long long>(zp_w) * sum_x;
        v += static_cast<long long>(n_incl) * zp_x * zp_w;

        const long long y_idx = (static_cast<long long>(n_idx) * c_out + oc) *
                                    static_cast<long long>(hw_out) +
                                spatial_idx;
        y[y_idx] = static_cast<int>(v);
    }
}
