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
// ============================================================================

namespace {

constexpr int DQL_REDUCE_BLOCK = 256;
constexpr int MM_TILE = 16;

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

    const int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        if (scale > 0.0f) {
            const float q = rintf(x[idx] / scale) + zp_f;
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
//   grid  = (ceil_div(M, 16) * ceil_div(N, 16), d0 * d1)
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
    __shared__ int s_a[MM_TILE][MM_TILE];
    __shared__ int s_b[MM_TILE][MM_TILE];

    const int tiles_n = (n + MM_TILE - 1) / MM_TILE;
    const int tile_m = blockIdx.x / tiles_n;
    const int tile_n = blockIdx.x - tile_m * tiles_n;
    const int batch_idx = blockIdx.y;

    if (batch_idx >= d0 * d1) {
        return;
    }

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
    const int row = tile_m * MM_TILE + ty;
    const int col = tile_n * MM_TILE + tx;

    const int zp_a = (row < m) ? load_zp(a_zp, a_zp_mode, batch_idx, row, m) : 0;
    const int zp_b = (col < n) ? load_zp(b_zp, b_zp_mode, batch_idx, col, n) : 0;

    int acc = 0;
    const int num_k_tiles = (k + MM_TILE - 1) / MM_TILE;

    for (int t = 0; t < num_k_tiles; ++t) {
        const int a_col = t * MM_TILE + tx;
        const int b_row = t * MM_TILE + ty;

        if (row < m && a_col < k) {
            const unsigned char av =
                a[a_base + static_cast<long long>(row) * static_cast<long long>(k) + a_col];
            s_a[ty][tx] = static_cast<int>(av) - zp_a;
        } else {
            s_a[ty][tx] = 0;
        }

        if (b_row < k && col < n) {
            const unsigned char bv =
                b[b_base + static_cast<long long>(b_row) * static_cast<long long>(n) + col];
            s_b[ty][tx] = static_cast<int>(bv) - zp_b;
        } else {
            s_b[ty][tx] = 0;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MM_TILE; ++kk) {
            acc += s_a[ty][kk] * s_b[kk][tx];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        y[y_base + static_cast<long long>(row) * static_cast<long long>(n) + col] = acc;
    }
}

// ----------------------------------------------------------------------------
// 3. ConvInteger — General 2D NCHW U8 x U8 -> I32 with optional ZP
//
// Covers eSamEncoderPreUInt8.onnx:
//   - k=16, s=16, p=0
//   - k=1,  s=1,  p=0
//   - k=3,  s=1,  p=1
// as well as arbitrary kernel_shape, strides, pads, dilations, and groups.
//
// Per ONNX ConvInteger spec, out-of-bounds padded pixels equal x_zero_point,
// so ((int)x_pad - zp_x) == 0 and out-of-bounds positions contribute 0.
//
// Zero-point modes:
//   x_zp_mode: 0 = none (0), 1 = scalar [1]
//   w_zp_mode: 0 = none (0), 1 = scalar [1], 2 = per-channel [C_out]
//
// Launch with:
//   block = (256, 1)
//   grid  = (ceil_div(H_out * W_out, 256), N * C_out)
// ----------------------------------------------------------------------------
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
    const int hw_out = h_out * w_out;
    const int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int noc = blockIdx.y;

    if (spatial_idx >= hw_out || noc >= batch_n * c_out) {
        return;
    }

    const int n_idx = noc / c_out;
    const int oc = noc - n_idx * c_out;
    const int oh = spatial_idx / w_out;
    const int ow = spatial_idx - oh * w_out;

    const int c_in_per_group = c_in / group;
    const int c_out_per_group = c_out / group;
    const int g = oc / c_out_per_group;
    const int ic_start = g * c_in_per_group;

    const int zp_x = (x_zp_mode != 0 && x_zp != nullptr) ? static_cast<int>(x_zp[0]) : 0;
    int zp_w = 0;
    if (w_zp_mode == 1 && w_zp != nullptr) {
        zp_w = static_cast<int>(w_zp[0]);
    } else if (w_zp_mode == 2 && w_zp != nullptr) {
        zp_w = static_cast<int>(w_zp[oc]);
    }

    const long long x_batch_base =
        static_cast<long long>(n_idx) * static_cast<long long>(c_in) *
        static_cast<long long>(h_in) * static_cast<long long>(w_in);
    const long long w_oc_base =
        static_cast<long long>(oc) * static_cast<long long>(c_in_per_group) *
        static_cast<long long>(k_h) * static_cast<long long>(k_w);
    const long long hw_in = static_cast<long long>(h_in) * static_cast<long long>(w_in);
    const long long kh_kw = static_cast<long long>(k_h) * static_cast<long long>(k_w);

    const int ih_base = oh * stride_h - pad_top;
    const int iw_base = ow * stride_w - pad_left;

    int acc = 0;

    if (k_h == 1 && k_w == 1 && pad_top == 0 && pad_left == 0) {
        // Fast path for 1x1 pointwise convolution (frequent in eSamEncoderPreUInt8).
        const long long x_spatial_offset =
            static_cast<long long>(ih_base) * static_cast<long long>(w_in) + iw_base;
        if (ih_base >= 0 && ih_base < h_in && iw_base >= 0 && iw_base < w_in) {
            for (int ic = 0; ic < c_in_per_group; ++ic) {
                const long long x_idx =
                    x_batch_base + static_cast<long long>(ic_start + ic) * hw_in + x_spatial_offset;
                const int xv = static_cast<int>(x[x_idx]) - zp_x;
                const int wv = static_cast<int>(w[w_oc_base + ic]) - zp_w;
                acc += xv * wv;
            }
        }
    } else {
        for (int ic = 0; ic < c_in_per_group; ++ic) {
            const long long x_c_base =
                x_batch_base + static_cast<long long>(ic_start + ic) * hw_in;
            const long long w_c_base = w_oc_base + static_cast<long long>(ic) * kh_kw;

            for (int kh = 0; kh < k_h; ++kh) {
                const int ih = ih_base + kh * dilation_h;
                if (ih < 0 || ih >= h_in) {
                    continue;
                }
                const long long x_row_base = x_c_base + static_cast<long long>(ih) * w_in;
                const long long w_row_base = w_c_base + static_cast<long long>(kh) * k_w;

                for (int kw = 0; kw < k_w; ++kw) {
                    const int iw = iw_base + kw * dilation_w;
                    if (iw < 0 || iw >= w_in) {
                        continue;
                    }
                    const int xv = static_cast<int>(x[x_row_base + iw]) - zp_x;
                    const int wv = static_cast<int>(w[w_row_base + kw]) - zp_w;
                    acc += xv * wv;
                }
            }
        }
    }

    const long long y_idx =
        static_cast<long long>(noc) * static_cast<long long>(hw_out) + spatial_idx;
    y[y_idx] = acc;
}
