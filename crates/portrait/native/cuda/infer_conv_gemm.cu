// Aurora Native Inference Engine — Phase 2 Core Conv / ConvTranspose / MatMul / GEMM Kernels.
//
// Target: sm_75+ PTX compiled with:
//   nvcc --ptx -O3 --prec-div=true --prec-sqrt=true -arch=compute_75 -Xcompiler /wd4819
//
// Design invariants:
// 1. All kernels use 1D or 2D grids (gridDim.z == 1, blockDim.z == 1) to match
//    `crate::cuda::Nvidia::launch_raw`.
// 2. All tensor indices and byte/element strides use 64-bit offsets where needed
//    to avoid 32-bit overflow on large feature maps.
// 3. Zero external headers or DLL dependencies.

#include <math.h>
#include <stddef.h>

// Activation codes shared with `ops_conv_gemm.rs` (`FusedActivation`).
// 0 = None, 1 = ReLU, 2 = SiLU (x * sigmoid(x))
__device__ __forceinline__ float apply_activation(float v, unsigned int act_mode) {
  if (act_mode == 1u) {
    return v > 0.0f ? v : 0.0f;
  }
  if (act_mode == 2u) {
    return v / (1.0f + expf(-v));
  }
  return v;
}

// ---------------------------------------------------------------------------
// 1. im2col_nchw_f32
// ---------------------------------------------------------------------------
// Unfolds one NCHW sample [C_in, H_in, W_in] into a 2D column matrix
// [C_in * kh * kw, out_h * out_w] in row-major order.
// For grouped convolutions, each group's slice [g * c_in_g * kh * kw ..] is
// contiguous along the row dimension, allowing a single batched GEMM across
// all groups.
extern "C" __global__ void im2col_nchw_f32(
    const float* __restrict__ x,
    float* __restrict__ col,
    unsigned int c_in,
    unsigned int h_in,
    unsigned int w_in,
    unsigned int kh,
    unsigned int kw,
    unsigned int sh,
    unsigned int sw,
    unsigned int dh,
    unsigned int dw,
    unsigned int pad_t,
    unsigned int pad_l,
    unsigned int out_h,
    unsigned int out_w) {
  const size_t out_hw = (size_t)out_h * (size_t)out_w;
  const size_t col_rows = (size_t)c_in * (size_t)kh * (size_t)kw;
  const size_t total = col_rows * out_hw;
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (idx >= total) return;

  const size_t spatial = idx % out_hw;
  const size_t col_row = idx / out_hw;

  const unsigned int oh = (unsigned int)(spatial / (size_t)out_w);
  const unsigned int ow = (unsigned int)(spatial - (size_t)oh * (size_t)out_w);

  const unsigned int kw_i = (unsigned int)(col_row % (size_t)kw);
  const size_t tmp = col_row / (size_t)kw;
  const unsigned int kh_i = (unsigned int)(tmp % (size_t)kh);
  const unsigned int c = (unsigned int)(tmp / (size_t)kh);

  const int ih = (int)(oh * sh) - (int)pad_t + (int)(kh_i * dh);
  const int iw = (int)(ow * sw) - (int)pad_l + (int)(kw_i * dw);

  float val = 0.0f;
  if (ih >= 0 && ih < (int)h_in && iw >= 0 && iw < (int)w_in) {
    const size_t x_idx = ((size_t)c * (size_t)h_in + (size_t)ih) * (size_t)w_in + (size_t)iw;
    val = x[x_idx];
  }
  col[idx] = val;
}

// ---------------------------------------------------------------------------
// 2. sgemm_tiled_f32
// ---------------------------------------------------------------------------
// 32x32 Shared-Memory Tiled Strided-Batched SGEMM with 2x2 register blocking
// per thread (blockDim = (16, 16), 256 threads/block).
//
// Computes for each batch b in [0, batch_count):
//   C_b = act(alpha * op(A_b) * op(B_b) + beta_term)
//
// Matrix layouts (row-major):
//   op(A) is [M, K]: if trans_a == 0, A is [M, K]; if trans_a == 1, A is [K, M].
//   op(B) is [K, N]: if trans_b == 0, B is [K, N]; if trans_b == 1, B is [N, K].
//   C     is [M, N].
//
// bias_mode:
//   0 = No external bias; if beta != 0, adds beta * C_b[row, col].
//   1 = Row broadcast:    adds beta * bias_b[row]         (Conv NCHW bias or Gemm [M, 1]).
//   2 = Col broadcast:    adds beta * bias_b[col]         (Gemm [N] or [1, N]).
//   3 = Scalar broadcast: adds beta * bias_b[0]           (Gemm [1]).
//   4 = Full 2D:          adds beta * bias_b[row * N + col] (Gemm [M, N]).
//
// Launch geometry:
//   grid  = (batch_count * ceil(M / 32) * ceil(N / 32), 1)
//   block = (16, 16)
extern "C" __global__ void sgemm_tiled_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    const float* __restrict__ bias,
    unsigned int m,
    unsigned int n,
    unsigned int k,
    unsigned int trans_a,
    unsigned int trans_b,
    float alpha,
    float beta,
    unsigned int bias_mode,
    unsigned int act_mode,
    unsigned int batch_count,
    unsigned long long stride_a,
    unsigned long long stride_b,
    unsigned long long stride_c,
    unsigned long long stride_bias) {
  const unsigned int tiles_m = (m + 31u) >> 5;
  const unsigned int tiles_n = (n + 31u) >> 5;
  const unsigned int tiles_per_batch = tiles_m * tiles_n;
  const unsigned int tile_linear = blockIdx.x;
  const unsigned int batch_idx = tile_linear / tiles_per_batch;
  if (batch_idx >= batch_count) return;

  const unsigned int rem = tile_linear - batch_idx * tiles_per_batch;
  const unsigned int tile_m = rem / tiles_n;
  const unsigned int tile_n = rem - tile_m * tiles_n;

  const unsigned int tx = threadIdx.x; // 0..15
  const unsigned int ty = threadIdx.y; // 0..15
  const unsigned int row_base = tile_m * 32u;
  const unsigned int col_base = tile_n * 32u;

  const float* __restrict__ a_batch = a + (size_t)batch_idx * (size_t)stride_a;
  const float* __restrict__ b_batch = b + (size_t)batch_idx * (size_t)stride_b;
  float* __restrict__ c_batch = c + (size_t)batch_idx * (size_t)stride_c;

  // +1 column padding eliminates shared memory bank conflicts.
  __shared__ float As[32][33];
  __shared__ float Bs[32][33];

  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;

  const unsigned int row0 = row_base + ty;
  const unsigned int row1 = row0 + 16u;
  const unsigned int col0 = col_base + tx;
  const unsigned int col1 = col0 + 16u;

  for (unsigned int k0 = 0u; k0 < k; k0 += 32u) {
    // Load 32x32 tile of op(A): rows [row_base..+32), cols [k0..+32)
    #pragma unroll
    for (unsigned int dr = 0u; dr < 32u; dr += 16u) {
      const unsigned int r = ty + dr;
      const unsigned int g_row = row_base + r;
      #pragma unroll
      for (unsigned int dc = 0u; dc < 32u; dc += 16u) {
        const unsigned int col_k = tx + dc;
        const unsigned int g_k = k0 + col_k;
        float val = 0.0f;
        if (g_row < m && g_k < k) {
          if (trans_a == 0u) {
            val = a_batch[(size_t)g_row * (size_t)k + (size_t)g_k];
          } else {
            val = a_batch[(size_t)g_k * (size_t)m + (size_t)g_row];
          }
        }
        As[r][col_k] = val;
      }
    }

    // Load 32x32 tile of op(B): rows [k0..+32), cols [col_base..+32)
    #pragma unroll
    for (unsigned int dr = 0u; dr < 32u; dr += 16u) {
      const unsigned int row_k = ty + dr;
      const unsigned int g_k = k0 + row_k;
      #pragma unroll
      for (unsigned int dc = 0u; dc < 32u; dc += 16u) {
        const unsigned int c_idx = tx + dc;
        const unsigned int g_col = col_base + c_idx;
        float val = 0.0f;
        if (g_k < k && g_col < n) {
          if (trans_b == 0u) {
            val = b_batch[(size_t)g_k * (size_t)n + (size_t)g_col];
          } else {
            val = b_batch[(size_t)g_col * (size_t)k + (size_t)g_k];
          }
        }
        Bs[row_k][c_idx] = val;
      }
    }

    __syncthreads();

    #pragma unroll
    for (unsigned int kk = 0u; kk < 32u; ++kk) {
      const float a0 = As[ty][kk];
      const float a1 = As[ty + 16u][kk];
      const float b0 = Bs[kk][tx];
      const float b1 = Bs[kk][tx + 16u];
      acc00 = fmaf(a0, b0, acc00);
      acc01 = fmaf(a0, b1, acc01);
      acc10 = fmaf(a1, b0, acc10);
      acc11 = fmaf(a1, b1, acc11);
    }

    __syncthreads();
  }

  const float* __restrict__ bias_batch =
      (bias != nullptr) ? (bias + (size_t)batch_idx * (size_t)stride_bias) : nullptr;

  const unsigned int rows[2] = {row0, row1};
  const unsigned int cols[2] = {col0, col1};
  const float accs[2][2] = {{acc00, acc01}, {acc10, acc11}};

  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    const unsigned int r = rows[i];
    if (r >= m) continue;
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
      const unsigned int col_out = cols[j];
      if (col_out >= n) continue;

      float val = alpha * accs[i][j];
      const size_t c_idx = (size_t)r * (size_t)n + (size_t)col_out;

      if (bias_mode == 0u) {
        if (beta != 0.0f) {
          val += beta * c_batch[c_idx];
        }
      } else if (bias_batch != nullptr && beta != 0.0f) {
        if (bias_mode == 1u) {
          val += beta * bias_batch[r];
        } else if (bias_mode == 2u) {
          val += beta * bias_batch[col_out];
        } else if (bias_mode == 3u) {
          val += beta * bias_batch[0];
        } else if (bias_mode == 4u) {
          val += beta * bias_batch[c_idx];
        }
      }

      c_batch[c_idx] = apply_activation(val, act_mode);
    }
  }
}

// ---------------------------------------------------------------------------
// 3. add_bias_nchw_f32
// ---------------------------------------------------------------------------
// In-place NCHW channel bias add + optional fused activation.
extern "C" __global__ void add_bias_nchw_f32(
    float* __restrict__ y,
    const float* __restrict__ bias,
    unsigned int batch,
    unsigned int channels,
    unsigned int spatial,
    unsigned int act_mode) {
  const size_t total = (size_t)batch * (size_t)channels * (size_t)spatial;
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (idx >= total) return;

  const unsigned int c = (unsigned int)((idx / (size_t)spatial) % (size_t)channels);
  float val = y[idx];
  if (bias != nullptr) {
    val += bias[c];
  }
  y[idx] = apply_activation(val, act_mode);
}

// ---------------------------------------------------------------------------
// 4. depthwise_conv2d_nchw_f32
// ---------------------------------------------------------------------------
// Direct Depthwise Conv2D kernel (group == C_in == C_out), zero im2col scratch.
// x:    [N, C, H_in, W_in]
// w:    [C, 1, kh, kw]
// bias: [C] or nullptr
// y:    [N, C, out_h, out_w]
extern "C" __global__ void depthwise_conv2d_nchw_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    unsigned int n_batch,
    unsigned int channels,
    unsigned int h_in,
    unsigned int w_in,
    unsigned int kh,
    unsigned int kw,
    unsigned int sh,
    unsigned int sw,
    unsigned int dh,
    unsigned int dw,
    unsigned int pad_t,
    unsigned int pad_l,
    unsigned int out_h,
    unsigned int out_w,
    unsigned int act_mode) {
  const size_t out_hw = (size_t)out_h * (size_t)out_w;
  const size_t total = (size_t)n_batch * (size_t)channels * out_hw;
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (idx >= total) return;

  const size_t spatial = idx % out_hw;
  const size_t nc = idx / out_hw;
  const unsigned int oh = (unsigned int)(spatial / (size_t)out_w);
  const unsigned int ow = (unsigned int)(spatial - (size_t)oh * (size_t)out_w);
  const unsigned int c = (unsigned int)(nc % (size_t)channels);
  const unsigned int ni = (unsigned int)(nc / (size_t)channels);

  const size_t x_base = ((size_t)ni * (size_t)channels + (size_t)c) * (size_t)h_in * (size_t)w_in;
  const size_t w_base = (size_t)c * (size_t)kh * (size_t)kw;

  const int ih_origin = (int)(oh * sh) - (int)pad_t;
  const int iw_origin = (int)(ow * sw) - (int)pad_l;

  float acc = (bias != nullptr) ? bias[c] : 0.0f;

  for (unsigned int kh_i = 0u; kh_i < kh; ++kh_i) {
    const int ih = ih_origin + (int)(kh_i * dh);
    if (ih < 0 || ih >= (int)h_in) continue;
    const size_t x_row = x_base + (size_t)ih * (size_t)w_in;
    const size_t w_row = w_base + (size_t)kh_i * (size_t)kw;
    for (unsigned int kw_i = 0u; kw_i < kw; ++kw_i) {
      const int iw = iw_origin + (int)(kw_i * dw);
      if (iw < 0 || iw >= (int)w_in) continue;
      acc = fmaf(x[x_row + (size_t)iw], w[w_row + (size_t)kw_i], acc);
    }
  }

  y[idx] = apply_activation(acc, act_mode);
}

// ---------------------------------------------------------------------------
// 5. conv_transpose2d_nchw_f32
// ---------------------------------------------------------------------------
// Gather-based NCHW ConvTranspose2D kernel (deterministic, zero atomicAdd).
// Supports k=2x2 s=2x2 p=0 as well as general stride/pad/dilation/group.
//
// x:    [N, C_in, H_in, W_in]
// w:    [C_in, C_out / group, kh, kw]  (ONNX ConvTranspose weight layout)
// bias: [C_out] or nullptr
// y:    [N, C_out, out_h, out_w]
extern "C" __global__ void conv_transpose2d_nchw_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    unsigned int n_batch,
    unsigned int c_in,
    unsigned int h_in,
    unsigned int w_in,
    unsigned int c_out,
    unsigned int kh,
    unsigned int kw,
    unsigned int sh,
    unsigned int sw,
    unsigned int dh,
    unsigned int dw,
    unsigned int pad_t,
    unsigned int pad_l,
    unsigned int out_h,
    unsigned int out_w,
    unsigned int group,
    unsigned int act_mode) {
  const size_t out_hw = (size_t)out_h * (size_t)out_w;
  const size_t total = (size_t)n_batch * (size_t)c_out * out_hw;
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (idx >= total) return;

  const size_t spatial = idx % out_hw;
  const size_t nc = idx / out_hw;
  const unsigned int oh = (unsigned int)(spatial / (size_t)out_w);
  const unsigned int ow = (unsigned int)(spatial - (size_t)oh * (size_t)out_w);
  const unsigned int co = (unsigned int)(nc % (size_t)c_out);
  const unsigned int ni = (unsigned int)(nc / (size_t)c_out);

  const unsigned int c_in_g = c_in / group;
  const unsigned int c_out_g = c_out / group;
  const unsigned int g = co / c_out_g;
  const unsigned int co_g = co - g * c_out_g;
  const unsigned int ci_start = g * c_in_g;

  const size_t in_hw = (size_t)h_in * (size_t)w_in;
  const size_t x_batch_base = (size_t)ni * (size_t)c_in * in_hw;
  const size_t kh_kw = (size_t)kh * (size_t)kw;
  const size_t w_ci_stride = (size_t)c_out_g * kh_kw;

  float acc = (bias != nullptr) ? bias[co] : 0.0f;

  for (unsigned int kh_i = 0u; kh_i < kh; ++kh_i) {
    const int h_num = (int)oh + (int)pad_t - (int)(kh_i * dh);
    if (h_num < 0) continue;
    if ((h_num % (int)sh) != 0) continue;
    const int ih = h_num / (int)sh;
    if (ih >= (int)h_in) continue;

    for (unsigned int kw_i = 0u; kw_i < kw; ++kw_i) {
      const int w_num = (int)ow + (int)pad_l - (int)(kw_i * dw);
      if (w_num < 0) continue;
      if ((w_num % (int)sw) != 0) continue;
      const int iw = w_num / (int)sw;
      if (iw >= (int)w_in) continue;

      const size_t x_sp = (size_t)ih * (size_t)w_in + (size_t)iw;
      const size_t w_k_off = (size_t)co_g * kh_kw + (size_t)kh_i * (size_t)kw + (size_t)kw_i;

      for (unsigned int ci_i = 0u; ci_i < c_in_g; ++ci_i) {
        const unsigned int ci = ci_start + ci_i;
        const float xv = x[x_batch_base + (size_t)ci * in_hw + x_sp];
        const float wv = w[(size_t)ci * w_ci_stride + w_k_off];
        acc = fmaf(xv, wv, acc);
      }
    }
  }

  y[idx] = apply_activation(acc, act_mode);
}

// ---------------------------------------------------------------------------
// 6. batched_matmul_f32
// ---------------------------------------------------------------------------
// 32x32 Shared-Memory Tiled Batched MatMul with up to 4 broadcast batch dims
// (supports 2D, 3D, 4D, 5D, and 6D ONNX MatMul with NumPy broadcasting).
//
// For each broadcasted batch coordinate (i0, i1, i2, i3):
//   A_b = a + i0*a_s0 + i1*a_s1 + i2*a_s2 + i3*a_s3   ([M, K] row-major)
//   B_b = b + i0*b_s0 + i1*b_s1 + i2*b_s2 + i3*b_s3   ([K, N] row-major)
//   C_b = c + batch_idx * (M * N)                     ([M, N] row-major)
//
// Broadcasted batch dimensions simply pass stride 0 for the corresponding dim.
//
// Launch geometry:
//   grid  = (batch_total * ceil(M / 32) * ceil(N / 32), 1)
//   block = (16, 16)
extern "C" __global__ void batched_matmul_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    unsigned int m,
    unsigned int n,
    unsigned int k,
    unsigned int d0,
    unsigned int d1,
    unsigned int d2,
    unsigned int d3,
    unsigned long long a_s0,
    unsigned long long a_s1,
    unsigned long long a_s2,
    unsigned long long a_s3,
    unsigned long long b_s0,
    unsigned long long b_s1,
    unsigned long long b_s2,
    unsigned long long b_s3) {
  const unsigned int batch_total = d0 * d1 * d2 * d3;
  const unsigned int tiles_m = (m + 31u) >> 5;
  const unsigned int tiles_n = (n + 31u) >> 5;
  const unsigned int tiles_per_batch = tiles_m * tiles_n;
  const unsigned int tile_linear = blockIdx.x;
  const unsigned int batch_idx = tile_linear / tiles_per_batch;
  if (batch_idx >= batch_total) return;

  const unsigned int rem = tile_linear - batch_idx * tiles_per_batch;
  const unsigned int tile_m = rem / tiles_n;
  const unsigned int tile_n = rem - tile_m * tiles_n;

  // Decompose batch_idx into (i0, i1, i2, i3).
  unsigned int b_rem = batch_idx;
  const unsigned int i3 = b_rem % d3;
  b_rem /= d3;
  const unsigned int i2 = b_rem % d2;
  b_rem /= d2;
  const unsigned int i1 = b_rem % d1;
  const unsigned int i0 = b_rem / d1;

  const size_t off_a = (size_t)i0 * (size_t)a_s0 + (size_t)i1 * (size_t)a_s1 +
                       (size_t)i2 * (size_t)a_s2 + (size_t)i3 * (size_t)a_s3;
  const size_t off_b = (size_t)i0 * (size_t)b_s0 + (size_t)i1 * (size_t)b_s1 +
                       (size_t)i2 * (size_t)b_s2 + (size_t)i3 * (size_t)b_s3;
  const size_t off_c = (size_t)batch_idx * (size_t)m * (size_t)n;

  const float* __restrict__ a_batch = a + off_a;
  const float* __restrict__ b_batch = b + off_b;
  float* __restrict__ c_batch = c + off_c;

  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;
  const unsigned int row_base = tile_m * 32u;
  const unsigned int col_base = tile_n * 32u;

  __shared__ float As[32][33];
  __shared__ float Bs[32][33];

  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;

  for (unsigned int k0 = 0u; k0 < k; k0 += 32u) {
    #pragma unroll
    for (unsigned int dr = 0u; dr < 32u; dr += 16u) {
      const unsigned int r = ty + dr;
      const unsigned int g_row = row_base + r;
      #pragma unroll
      for (unsigned int dc = 0u; dc < 32u; dc += 16u) {
        const unsigned int col_k = tx + dc;
        const unsigned int g_k = k0 + col_k;
        As[r][col_k] = (g_row < m && g_k < k)
                           ? a_batch[(size_t)g_row * (size_t)k + (size_t)g_k]
                           : 0.0f;
      }
    }

    #pragma unroll
    for (unsigned int dr = 0u; dr < 32u; dr += 16u) {
      const unsigned int row_k = ty + dr;
      const unsigned int g_k = k0 + row_k;
      #pragma unroll
      for (unsigned int dc = 0u; dc < 32u; dc += 16u) {
        const unsigned int c_idx = tx + dc;
        const unsigned int g_col = col_base + c_idx;
        Bs[row_k][c_idx] = (g_k < k && g_col < n)
                               ? b_batch[(size_t)g_k * (size_t)n + (size_t)g_col]
                               : 0.0f;
      }
    }

    __syncthreads();

    #pragma unroll
    for (unsigned int kk = 0u; kk < 32u; ++kk) {
      const float a0 = As[ty][kk];
      const float a1 = As[ty + 16u][kk];
      const float b0 = Bs[kk][tx];
      const float b1 = Bs[kk][tx + 16u];
      acc00 = fmaf(a0, b0, acc00);
      acc01 = fmaf(a0, b1, acc01);
      acc10 = fmaf(a1, b0, acc10);
      acc11 = fmaf(a1, b1, acc11);
    }

    __syncthreads();
  }

  const unsigned int r0 = row_base + ty;
  const unsigned int r1 = r0 + 16u;
  const unsigned int c0 = col_base + tx;
  const unsigned int c1 = c0 + 16u;

  if (r0 < m) {
    if (c0 < n) c_batch[(size_t)r0 * (size_t)n + (size_t)c0] = acc00;
    if (c1 < n) c_batch[(size_t)r0 * (size_t)n + (size_t)c1] = acc01;
  }
  if (r1 < m) {
    if (c0 < n) c_batch[(size_t)r1 * (size_t)n + (size_t)c0] = acc10;
    if (c1 < n) c_batch[(size_t)r1 * (size_t)n + (size_t)c1] = acc11;
  }
}
