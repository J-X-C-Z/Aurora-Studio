// CUDA operator kernels for `native_infer_graph::Graph` (FP32 NCHW).
//
// Each entry point is the GPU twin of one `OpKind` arm of
// `native_infer_graph::execute_node`, and the FP32 CPU executor stays the
// oracle: `native_infer_cuda` runs both and compares. Every expression mirrors
// the CPU statement order so `-fmad=false` alone is enough to keep results
// within 1e-4; group statistics are the one place where the reduction order
// differs (parallel two-pass instead of Welford), which is why the end-to-end
// comparison is a tolerance test, not a bit test.
//
// Build (CUDA 13.4; same flag set as work/backend-r11/o1i/build-ptx.cmd):
//   nvcc -std=c++17 -O3 -fmad=false -gencode arch=compute_89,code=sm_89 -ptx ^
//        native_infer_ops.cu -o native_infer_ops.ptx
//
// Tensor layouts are contiguous NCHW, i.e. `TensorShape::index(n, c, h, w)` ==
// `((n * C + c) * H + h) * W + w`. Element counts above 2^31 are refused by the
// host before a launch, so `int` indexing is safe here.
//
// Pointer arguments for optional operands are passed as 0 and guarded by a
// uniform flag; the kernel never dereferences a null argument.

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// ---- shared float helpers -------------------------------------------------

__device__ __forceinline__ float ni_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float ni_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float ni_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float ni_div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float ni_sqrt(float a) { return __fsqrt_rn(a); }

// ---- uniform blocks (mirrored by #[repr(C)] structs in native_infer_cuda.rs)

typedef struct NiConvUniforms {
  int n;
  int c_in;
  int in_h;
  int in_w;
  int c_out;
  int kh;
  int kw;
  int stride_h;
  int stride_w;
  int pad_t;
  int pad_l;
  int dil_h;
  int dil_w;
  int out_h;
  int out_w;
  int has_bias;  /* 0/1 */
  int depthwise; /* 0/1 */
} NiConvUniforms;

typedef struct NiNormUniforms {
  int count;    /* n * c * spatial */
  int channels;
  int spatial; /* h * w */
  float eps;
} NiNormUniforms;

typedef struct NiGroupNormUniforms {
  int n;
  int channels;
  int spatial;
  int groups;
  float eps;
} NiGroupNormUniforms;

typedef struct NiBroadcastUniforms {
  int out_n, out_c, out_h, out_w;
  int a_n, a_c, a_h, a_w;
  int b_n, b_c, b_h, b_w;
} NiBroadcastUniforms;

typedef struct NiResizeUniforms {
  int in_n, in_c, in_h, in_w;
  int out_n, out_c, out_h, out_w;
  float scale_h;
  float scale_w;
  int align_corners; /* 0/1 */
} NiResizeUniforms;

typedef struct NiGridSampleUniforms {
  int n;
  int c;
  int in_h, in_w;
  int grid_h, grid_w;
} NiGridSampleUniforms;

typedef struct NiMatMulUniforms {
  int n;
  int rows;       /* a.c */
  int k;          /* a.w == b.h */
  int cols;       /* b.c */
  int a_n_stride; /* a.c * a.h * a.w: elements per batch of A */
  int a_row_stride; /* a.h * a.w: elements per row of A */
  int b_n_stride; /* b.c * b.h * b.w: elements per batch of B */
  int b_j_stride; /* b.h * b.w: elements per output column of B */
  int b_k_stride; /* b.w: elements per contraction step of B */
} NiMatMulUniforms;

#ifdef __cplusplus
static_assert(sizeof(NiConvUniforms) == 68, "NiConvUniforms");
static_assert(sizeof(NiNormUniforms) == 16, "NiNormUniforms");
static_assert(sizeof(NiGroupNormUniforms) == 20, "NiGroupNormUniforms");
static_assert(sizeof(NiBroadcastUniforms) == 48, "NiBroadcastUniforms");
static_assert(sizeof(NiResizeUniforms) == 44, "NiResizeUniforms");
static_assert(sizeof(NiGridSampleUniforms) == 24, "NiGridSampleUniforms");
static_assert(sizeof(NiMatMulUniforms) == 36, "NiMatMulUniforms");
#endif

__device__ __forceinline__ int ni_linear_id() {
  return (int)(blockIdx.x * blockDim.x + threadIdx.x);
}

// ---- elementwise unary ----------------------------------------------------

extern "C" __global__ void ni_copy_bytes_kernel(unsigned char* __restrict__ dst,
                                               const unsigned char* __restrict__ src,
                                               long long count) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  dst[i] = src[i];
}

extern "C" __global__ void ni_relu_kernel(const float* __restrict__ x,
                                         float* __restrict__ out, int count) {
  const int i = ni_linear_id();
  if (i >= count) return;
  out[i] = fmaxf(x[i], 0.0f); /* CPU: `v.max(0.0)` */
}

extern "C" __global__ void ni_silu_kernel(const float* __restrict__ x,
                                         float* __restrict__ out, int count) {
  const int i = ni_linear_id();
  if (i >= count) return;
  const float v = x[i];
  /* CPU: `v / (1.0 + (-v).exp())` */
  out[i] = ni_div(v, ni_add(1.0f, expf(-v)));
}

extern "C" __global__ void ni_sigmoid_kernel(const float* __restrict__ x,
                                            float* __restrict__ out, int count) {
  const int i = ni_linear_id();
  if (i >= count) return;
  const float v = x[i];
  /* CPU: `1.0 / (1.0 + (-v).exp())` */
  out[i] = ni_div(1.0f, ni_add(1.0f, expf(-v)));
}

// ---- convolution ----------------------------------------------------------

// One thread per output element; accumulation order is the CPU's
// `for ic { for kh { for kw { acc += x * w } } }`.
__device__ __forceinline__ void ni_conv_element(const float* x, const float* w,
                                                const float* bias, float* out,
                                                const NiConvUniforms& u, int i) {
  const int ow = i % u.out_w;
  int rest = i / u.out_w;
  const int oh = rest % u.out_h;
  rest /= u.out_h;
  const int oc = rest % u.c_out;
  const int n = rest / u.c_out;

  const int ic_start = u.depthwise ? oc : 0;
  const int ic_end = u.depthwise ? (oc + 1) : u.c_in;
  const int oc_stride = (u.depthwise ? 1 : u.c_in) * u.kh * u.kw;

  float acc = u.has_bias ? bias[oc] : 0.0f;
  for (int ic = ic_start; ic < ic_end; ++ic) {
    const float* xp = x + ((size_t)(n * u.c_in + ic) * u.in_h) * u.in_w;
    const float* wp = w + oc * oc_stride + (u.depthwise ? 0 : ic) * u.kh * u.kw;
    for (int kh = 0; kh < u.kh; ++kh) {
      int ih = oh * u.stride_h + kh * u.dil_h;
      if (ih < u.pad_t) continue;
      ih -= u.pad_t;
      if (ih >= u.in_h) continue;
      for (int kw = 0; kw < u.kw; ++kw) {
        int iw = ow * u.stride_w + kw * u.dil_w;
        if (iw < u.pad_l) continue;
        iw -= u.pad_l;
        if (iw >= u.in_w) continue;
        acc = ni_add(acc, ni_mul(xp[ih * u.in_w + iw], wp[kh * u.kw + kw]));
      }
    }
  }
  out[i] = acc;
}

extern "C" __global__ void ni_conv2d_kernel(const float* __restrict__ x,
                                            const float* __restrict__ w,
                                            const float* __restrict__ bias,
                                            float* __restrict__ out,
                                            NiConvUniforms u) {
  const int i = ni_linear_id();
  if (i >= u.n * u.out_h * u.out_w * u.c_out) return;
  ni_conv_element(x, w, bias, out, u, i);
}

extern "C" __global__ void ni_depthwise_conv2d_kernel(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ bias, float* __restrict__ out,
    NiConvUniforms u) {
  const int i = ni_linear_id();
  if (i >= u.n * u.out_h * u.out_w * u.c_out) return;
  ni_conv_element(x, w, bias, out, u, i);
}

// ---- normalization --------------------------------------------------------

extern "C" __global__ void ni_batchnorm_kernel(
    const float* __restrict__ x, const float* __restrict__ scale,
    const float* __restrict__ offset, const float* __restrict__ mean,
    const float* __restrict__ var, float* __restrict__ out, NiNormUniforms u) {
  const int i = ni_linear_id();
  if (i >= u.count) return;
  const int c = (i / u.spatial) % u.channels;
  const float inv_std = ni_div(1.0f, ni_sqrt(ni_add(var[c], u.eps)));
  /* CPU: `(x - mean) * inv_std * scale + offset` */
  out[i] = ni_add(ni_mul(ni_mul(ni_sub(x[i], mean[c]), inv_std), scale[c]),
                  offset[c]);
}

// One block per (sample, group); block reduction over the group's elements.
extern "C" __global__ void ni_groupnorm_kernel(
    const float* __restrict__ x, const float* __restrict__ scale,
    const float* __restrict__ offset, float* __restrict__ out,
    NiGroupNormUniforms u) {
  const int group_id = (int)blockIdx.x;
  const int n = group_id / u.groups;
  const int g = group_id % u.groups;
  const int per_group = u.channels / u.groups;
  const int count = per_group * u.spatial;
  if (count <= 0) return;

  const size_t base = (size_t)n * u.channels * u.spatial;
  float* outp = out + base;

  __shared__ float sh_sum[256];
  __shared__ float sh_sq[256];
  const int tid = (int)threadIdx.x;
  const int lanes = (int)blockDim.x;

  float sum = 0.0f;
  float sq = 0.0f;
  for (int k = tid; k < count; k += lanes) {
    const int c = g * per_group + k / u.spatial;
    const float v = x[base + (size_t)c * u.spatial + (k % u.spatial)];
    sum = ni_add(sum, v);
    sq = ni_add(sq, ni_mul(v, v));
  }
  sh_sum[tid] = sum;
  sh_sq[tid] = sq;
  __syncthreads();
  for (int stride = lanes >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sh_sum[tid] = ni_add(sh_sum[tid], sh_sum[tid + stride]);
      sh_sq[tid] = ni_add(sh_sq[tid], sh_sq[tid + stride]);
    }
    __syncthreads();
  }

  const float mean = ni_div(sh_sum[0], (float)count);
  const float variance = ni_sub(ni_div(sh_sq[0], (float)count), ni_mul(mean, mean));
  const float inv_std = ni_div(1.0f, ni_sqrt(ni_add(variance, u.eps)));

  for (int k = tid; k < count; k += lanes) {
    const int c = g * per_group + k / u.spatial;
    const size_t idx = (size_t)c * u.spatial + (k % u.spatial);
    /* CPU: `(x - mean) * inv_std * scale + offset` */
    outp[idx] = ni_add(ni_mul(ni_mul(ni_sub(x[base + idx], mean), inv_std),
                              scale[c]),
                       offset[c]);
  }
}

// ---- broadcast binary -----------------------------------------------------

__device__ __forceinline__ int ni_broadcast_index(const NiBroadcastUniforms& u,
                                                  int n, int c, int h, int w) {
  return ((((u.a_n == 1 ? 0 : n) * u.a_c) + (u.a_c == 1 ? 0 : c)) * u.a_h +
          (u.a_h == 1 ? 0 : h)) *
             u.a_w +
         (u.a_w == 1 ? 0 : w);
}

__device__ __forceinline__ int ni_secondary_index(const NiBroadcastUniforms& u,
                                                  int n, int c, int h, int w) {
  return ((((u.b_n == 1 ? 0 : n) * u.b_c) + (u.b_c == 1 ? 0 : c)) * u.b_h +
          (u.b_h == 1 ? 0 : h)) *
             u.b_w +
         (u.b_w == 1 ? 0 : w);
}

__device__ __forceinline__ int ni_broadcast_offset(const NiBroadcastUniforms& u,
                                                   int i, int* n, int* c, int* h,
                                                   int* w) {
  *w = i % u.out_w;
  int rest = i / u.out_w;
  *h = rest % u.out_h;
  rest /= u.out_h;
  *c = rest % u.out_c;
  *n = rest / u.out_c;
  return (((*n * u.out_c + *c) * u.out_h + *h) * u.out_w + *w);
}

extern "C" __global__ void ni_add_kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ out,
                                         NiBroadcastUniforms u) {
  const int i = ni_linear_id();
  if (i >= u.out_n * u.out_c * u.out_h * u.out_w) return;
  int n, c, h, w;
  const int o = ni_broadcast_offset(u, i, &n, &c, &h, &w);
  out[o] = ni_add(a[ni_broadcast_index(u, n, c, h, w)],
                  b[ni_secondary_index(u, n, c, h, w)]);
}

extern "C" __global__ void ni_mul_kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ out,
                                         NiBroadcastUniforms u) {
  const int i = ni_linear_id();
  if (i >= u.out_n * u.out_c * u.out_h * u.out_w) return;
  int n, c, h, w;
  const int o = ni_broadcast_offset(u, i, &n, &c, &h, &w);
  out[o] = ni_mul(a[ni_broadcast_index(u, n, c, h, w)],
                  b[ni_secondary_index(u, n, c, h, w)]);
}

// ---- resampling -----------------------------------------------------------

extern "C" __global__ void ni_resize_bilinear_kernel(
    const float* __restrict__ x, float* __restrict__ out, NiResizeUniforms u) {
  const int i = ni_linear_id();
  if (i >= u.out_n * u.out_c * u.out_h * u.out_w) return;

  const int ow = i % u.out_w;
  int rest = i / u.out_w;
  const int oh = rest % u.out_h;
  rest /= u.out_h;
  const int c = rest % u.out_c;
  const int n = rest / u.out_c;

  const float in_h = (float)u.in_h;
  const float in_w = (float)u.in_w;
  float src_y;
  if (u.align_corners && u.out_h > 1) {
    src_y = ni_div(ni_mul((float)oh, ni_sub(in_h, 1.0f)), (float)(u.out_h - 1));
  } else {
    src_y = ni_sub(ni_div(ni_add((float)oh, 0.5f), u.scale_h), 0.5f);
  }
  const float src_x = (u.align_corners && u.out_w > 1)
                          ? ni_div(ni_mul((float)ow, ni_sub(in_w, 1.0f)),
                                   (float)(u.out_w - 1))
                          : ni_sub(ni_div(ni_add((float)ow, 0.5f), u.scale_w), 0.5f);

  const int y0 = (int)fmaxf(floorf(src_y), 0.0f);
  const int y1 = (y0 + 1) < (u.in_h - 1) ? (y0 + 1) : (u.in_h - 1);
  const float ay = fminf(fmaxf(ni_sub(src_y, (float)y0), 0.0f), 1.0f);
  const int x0 = (int)fmaxf(floorf(src_x), 0.0f);
  const int x1 = (x0 + 1) < (u.in_w - 1) ? (x0 + 1) : (u.in_w - 1);
  const float ax = fminf(fmaxf(ni_sub(src_x, (float)x0), 0.0f), 1.0f);

  const size_t plane = (size_t)u.in_h * u.in_w;
  const float* xp = x + ((size_t)n * u.in_c + c) * plane;
  const float v00 = xp[(size_t)y0 * u.in_w + x0];
  const float v01 = xp[(size_t)y0 * u.in_w + x1];
  const float v10 = xp[(size_t)y1 * u.in_w + x0];
  const float v11 = xp[(size_t)y1 * u.in_w + x1];

  const float top = ni_add(ni_mul(ni_sub(1.0f, ax), v00), ni_mul(ax, v01));
  const float bottom = ni_add(ni_mul(ni_sub(1.0f, ax), v10), ni_mul(ax, v11));
  out[i] = ni_add(ni_mul(ni_sub(1.0f, ay), top), ni_mul(ay, bottom));
}

extern "C" __global__ void ni_grid_sample_kernel(
    const float* __restrict__ x, const float* __restrict__ grid,
    float* __restrict__ out, NiGridSampleUniforms u) {
  const int total = u.n * u.c * u.grid_h * u.grid_w;
  const int i = ni_linear_id();
  if (i >= total) return;

  const int gw = i % u.grid_w;
  int rest = i / u.grid_w;
  const int gh = rest % u.grid_h;
  rest /= u.grid_h;
  const int c = rest % u.c;
  const int n = rest / u.c;

  const size_t grid_plane = (size_t)u.grid_h * u.grid_w;
  const float gx = grid[((size_t)n * 2 + 0) * grid_plane + (size_t)gh * u.grid_w + gw];
  const float gy = grid[((size_t)n * 2 + 1) * grid_plane + (size_t)gh * u.grid_w + gw];

  const float ix = ni_sub(ni_mul(ni_mul(ni_add(gx, 1.0f), 0.5f), (float)u.in_w), 0.5f);
  const float iy = ni_sub(ni_mul(ni_mul(ni_add(gy, 1.0f), 0.5f), (float)u.in_h), 0.5f);
  const int x0 = (int)floorf(ix);
  const int y0 = (int)floorf(iy);
  const float ax = ni_sub(ix, (float)x0);
  const float ay = ni_sub(iy, (float)y0);

  const size_t plane = (size_t)u.in_h * u.in_w;
  const float* xp = x + ((size_t)n * u.c + c) * plane;

  float acc = 0.0f;
  for (int dy = 0; dy < 2; ++dy) {
    const float wy = dy == 0 ? ni_sub(1.0f, ay) : ay;
    const int sy = y0 + dy;
    if (sy < 0 || sy >= u.in_h) continue;
    for (int dx = 0; dx < 2; ++dx) {
      const float wx = dx == 0 ? ni_sub(1.0f, ax) : ax;
      const int sx = x0 + dx;
      if (sx < 0 || sx >= u.in_w) continue;
      /* CPU: `acc += wx * wy * x[...]` (left-associative product, zero weights
         included, out-of-bounds taps skipped). */
      acc = ni_add(acc, ni_mul(ni_mul(wx, wy), xp[(size_t)sy * u.in_w + sx]));
    }
  }
  out[i] = acc;
}

// ---- matmul ---------------------------------------------------------------

extern "C" __global__ void ni_matmul_kernel(const float* __restrict__ a,
                                            const float* __restrict__ b,
                                            float* __restrict__ out,
                                            NiMatMulUniforms u) {
  const int total = u.n * u.rows * u.cols;
  const int i = ni_linear_id();
  if (i >= total) return;

  const int j = i % u.cols;
  int rest = i / u.cols;
  const int row = rest % u.rows;
  const int n = rest / u.rows;

  const float* ap = a + (size_t)n * u.a_n_stride + (size_t)row * u.a_row_stride;
  const float* bp = b + (size_t)n * u.b_n_stride + (size_t)j * u.b_j_stride;
  float acc = 0.0f;
  for (int k = 0; k < u.k; ++k) {
    acc = ni_add(acc, ni_mul(ap[k], bp[(size_t)k * u.b_k_stride]));
  }
  out[i] = acc;
}

// ---- quantization ---------------------------------------------------------

extern "C" __global__ void ni_quantize_kernel(const float* __restrict__ x,
                                              signed char* __restrict__ out,
                                              float scale, int zero_point,
                                              int count) {
  const int i = ni_linear_id();
  if (i >= count) return;
  /* CPU: `(v / scale).round() as i32 + zero_point`, then clamp to i8.
     `roundf` is ties away from zero, matching Rust's `f32::round`. */
  float rounded = roundf(ni_div(x[i], scale));
  rounded = fminf(fmaxf(rounded, -1.0e9f), 1.0e9f);
  int q = (int)rounded + zero_point;
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  out[i] = (signed char)q;
}

extern "C" __global__ void ni_dequantize_kernel(const signed char* __restrict__ x,
                                                float* __restrict__ out, float scale,
                                                int zero_point, int count) {
  const int i = ni_linear_id();
  if (i >= count) return;
  out[i] = ni_mul((float)((int)x[i] - zero_point), scale);
}
