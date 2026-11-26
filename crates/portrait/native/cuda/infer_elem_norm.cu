// Aurora Native Inference Engine — Elementwise, Activation, Comparison,
// Type-Cast, Softmax, and Normalization CUDA Kernels.
//
// Compiled to PTX with:
//   nvcc --ptx -O3 --fmad=false --prec-div=true --prec-sqrt=true \
//        -arch=compute_75 -Xcompiler /wd4819 infer_elem_norm.cu -o infer_elem_norm.ptx
//
// All reduction kernels use warp-shuffle + static shared memory (zero dynamic
// shared memory required) so they launch cleanly via `cuLaunchKernel(..., 0, ...)`.

#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Broadcast descriptors (passed by value, <= 5D NumPy broadcasting)
// ---------------------------------------------------------------------------

struct Broadcast5D {
    uint32_t total;
    uint32_t ndim;
    uint32_t out_shape[5];
    uint32_t a_strides[5];
    uint32_t b_strides[5];
};

struct Broadcast5DTernary {
    uint32_t total;
    uint32_t ndim;
    uint32_t out_shape[5];
    uint32_t c_strides[5];
    uint32_t x_strides[5];
    uint32_t y_strides[5];
};

// ---------------------------------------------------------------------------
// Warp / Block Reduction helpers (static shared memory, up to 32 warps)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_max_f32(float val) {
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFFu, val, 16));
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFFu, val, 8));
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFFu, val, 4));
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFFu, val, 2));
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFFu, val, 1));
    return val;
}

__device__ __forceinline__ double warp_reduce_sum_f64(double val) {
    val += __shfl_down_sync(0xFFFFFFFFu, val, 16);
    val += __shfl_down_sync(0xFFFFFFFFu, val, 8);
    val += __shfl_down_sync(0xFFFFFFFFu, val, 4);
    val += __shfl_down_sync(0xFFFFFFFFu, val, 2);
    val += __shfl_down_sync(0xFFFFFFFFu, val, 1);
    return val;
}

__device__ __forceinline__ float block_reduce_max_f32(float val, float* smem) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wid  = threadIdx.x >> 5;
    const uint32_t num_warps = (blockDim.x + 31u) >> 5;

    val = warp_reduce_max_f32(val);
    if (lane == 0u) {
        smem[wid] = val;
    }
    __syncthreads();

    if (wid == 0u) {
        float wv = (lane < num_warps) ? smem[lane] : -FLT_MAX;
        wv = warp_reduce_max_f32(wv);
        if (lane == 0u) {
            smem[0] = wv;
        }
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ double block_reduce_sum_f64(double val, double* smem) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wid  = threadIdx.x >> 5;
    const uint32_t num_warps = (blockDim.x + 31u) >> 5;

    val = warp_reduce_sum_f64(val);
    if (lane == 0u) {
        smem[wid] = val;
    }
    __syncthreads();

    if (wid == 0u) {
        double wv = (lane < num_warps) ? smem[lane] : 0.0;
        wv = warp_reduce_sum_f64(wv);
        if (lane == 0u) {
            smem[0] = wv;
        }
    }
    __syncthreads();
    return smem[0];
}

// ---------------------------------------------------------------------------
// 1. Generic <= 5D NumPy Broadcast Binary / Compare / Logic / Where Kernels
// ---------------------------------------------------------------------------

// op: 0=Add, 1=Sub, 2=Mul, 3=Div, 4=Pow
extern "C" __global__ void binary_broadcast_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    Broadcast5D bc,
    uint32_t op
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bc.total) return;

    uint32_t rem = idx;
    uint32_t off_a = 0u;
    uint32_t off_b = 0u;
    for (int d = static_cast<int>(bc.ndim) - 1; d >= 0; --d) {
        const uint32_t dim = bc.out_shape[d];
        const uint32_t coord = rem % dim;
        rem /= dim;
        off_a += coord * bc.a_strides[d];
        off_b += coord * bc.b_strides[d];
    }

    const float va = a[off_a];
    const float vb = b[off_b];
    float res = 0.0f;
    switch (op) {
        case 0u: res = va + vb; break;
        case 1u: res = va - vb; break;
        case 2u: res = va * vb; break;
        case 3u: res = va / vb; break;
        case 4u: res = powf(va, vb); break;
        default: res = 0.0f; break;
    }
    out[idx] = res;
}

// op: 0=Equal, 1=Greater, 2=GreaterOrEqual, 3=Less
extern "C" __global__ void compare_broadcast_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    uint8_t* __restrict__ out,
    Broadcast5D bc,
    uint32_t op
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bc.total) return;

    uint32_t rem = idx;
    uint32_t off_a = 0u;
    uint32_t off_b = 0u;
    for (int d = static_cast<int>(bc.ndim) - 1; d >= 0; --d) {
        const uint32_t dim = bc.out_shape[d];
        const uint32_t coord = rem % dim;
        rem /= dim;
        off_a += coord * bc.a_strides[d];
        off_b += coord * bc.b_strides[d];
    }

    const float va = a[off_a];
    const float vb = b[off_b];
    bool res = false;
    switch (op) {
        case 0u: res = (va == vb); break;
        case 1u: res = (va > vb);  break;
        case 2u: res = (va >= vb); break;
        case 3u: res = (va < vb);  break;
        default: res = false;      break;
    }
    out[idx] = res ? 1u : 0u;
}

extern "C" __global__ void and_broadcast_u8(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    uint8_t* __restrict__ out,
    Broadcast5D bc
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bc.total) return;

    uint32_t rem = idx;
    uint32_t off_a = 0u;
    uint32_t off_b = 0u;
    for (int d = static_cast<int>(bc.ndim) - 1; d >= 0; --d) {
        const uint32_t dim = bc.out_shape[d];
        const uint32_t coord = rem % dim;
        rem /= dim;
        off_a += coord * bc.a_strides[d];
        off_b += coord * bc.b_strides[d];
    }

    out[idx] = (a[off_a] != 0u && b[off_b] != 0u) ? 1u : 0u;
}

extern "C" __global__ void where_broadcast_f32(
    const uint8_t* __restrict__ cond,
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    Broadcast5DTernary bc
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bc.total) return;

    uint32_t rem = idx;
    uint32_t off_c = 0u;
    uint32_t off_x = 0u;
    uint32_t off_y = 0u;
    for (int d = static_cast<int>(bc.ndim) - 1; d >= 0; --d) {
        const uint32_t dim = bc.out_shape[d];
        const uint32_t coord = rem % dim;
        rem /= dim;
        off_c += coord * bc.c_strides[d];
        off_x += coord * bc.x_strides[d];
        off_y += coord * bc.y_strides[d];
    }

    out[idx] = (cond[off_c] != 0u) ? x[off_x] : y[off_y];
}

// ---------------------------------------------------------------------------
// 2. Unary Activations & Math Kernels + Broadcast PRelu
// ---------------------------------------------------------------------------

// op:
//  0=Relu, 1=Sigmoid, 2=SiLU, 3=GELU(erf), 4=Erf, 5=Sqrt, 6=Reciprocal,
//  7=Abs, 8=Sin, 9=Cos, 10=HardSwish, 11=HardSigmoid(p0=alpha, p1=beta),
//  12=LeakyRelu(p0=alpha), 13=Clip(p0=lo, p1=hi)
extern "C" __global__ void unary_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    uint32_t total,
    uint32_t op,
    float p0,
    float p1
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const float v = x[idx];
    float res = v;
    switch (op) {
        case 0u:  // Relu
            res = (v > 0.0f) ? v : 0.0f;
            break;
        case 1u:  // Sigmoid
            res = 1.0f / (1.0f + expf(-v));
            break;
        case 2u:  // SiLU
            res = v / (1.0f + expf(-v));
            break;
        case 3u:  // GELU (exact erf)
            res = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
            break;
        case 4u:  // Erf
            res = erff(v);
            break;
        case 5u:  // Sqrt
            res = sqrtf(v);
            break;
        case 6u:  // Reciprocal
            res = 1.0f / v;
            break;
        case 7u:  // Abs
            res = fabsf(v);
            break;
        case 8u:  // Sin
            res = sinf(v);
            break;
        case 9u:  // Cos
            res = cosf(v);
            break;
        case 10u: { // HardSwish: x * clamp(x / 6 + 0.5, 0, 1)
            const float hs = fminf(fmaxf(v * (1.0f / 6.0f) + 0.5f, 0.0f), 1.0f);
            res = v * hs;
            break;
        }
        case 11u: // HardSigmoid(alpha, beta): clamp(alpha * x + beta, 0, 1)
            res = fminf(fmaxf(p0 * v + p1, 0.0f), 1.0f);
            break;
        case 12u: // LeakyRelu(alpha)
            res = (v >= 0.0f) ? v : (p0 * v);
            break;
        case 13u: // Clip(lo, hi)
            res = fminf(fmaxf(v, p0), p1);
            break;
        default:
            res = v;
            break;
    }
    out[idx] = res;
}

extern "C" __global__ void prelu_broadcast_f32(
    const float* __restrict__ x,
    const float* __restrict__ slope,
    float* __restrict__ out,
    Broadcast5D bc
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bc.total) return;

    uint32_t rem = idx;
    uint32_t off_x = 0u;
    uint32_t off_s = 0u;
    for (int d = static_cast<int>(bc.ndim) - 1; d >= 0; --d) {
        const uint32_t dim = bc.out_shape[d];
        const uint32_t coord = rem % dim;
        rem /= dim;
        off_x += coord * bc.a_strides[d];
        off_s += coord * bc.b_strides[d];
    }

    const float v = x[off_x];
    const float s = slope[off_s];
    out[idx] = (v >= 0.0f) ? v : (v * s);
}

// ---------------------------------------------------------------------------
// 3. Type Cast Kernels (FP16 <-> FP32, INT64 <-> FP32)
// ---------------------------------------------------------------------------

extern "C" __global__ void cast_f16_to_f32(
    const uint16_t* __restrict__ src,
    float* __restrict__ dst,
    uint32_t total
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    dst[idx] = __half2float(__ushort_as_half(src[idx]));
}

extern "C" __global__ void cast_f32_to_f16(
    const float* __restrict__ src,
    uint16_t* __restrict__ dst,
    uint32_t total
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    dst[idx] = __half_as_ushort(__float2half_rn(src[idx]));
}

extern "C" __global__ void cast_i64_to_f32(
    const int64_t* __restrict__ src,
    float* __restrict__ dst,
    uint32_t total
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    dst[idx] = static_cast<float>(src[idx]);
}

extern "C" __global__ void cast_f32_to_i64(
    const float* __restrict__ src,
    int64_t* __restrict__ dst,
    uint32_t total
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    dst[idx] = static_cast<int64_t>(src[idx]);
}

// ---------------------------------------------------------------------------
// 4. Softmax & Normalization Kernels (Warp/Block Reduction, High Precision)
// ---------------------------------------------------------------------------

// General axis Softmax over tensor reshaped to [outer, axis_dim, inner].
// Launch with grid = (outer * inner, 1), block = (256, 1).
extern "C" __global__ void softmax_axis_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    uint32_t outer,
    uint32_t axis_dim,
    uint32_t inner
) {
    const uint32_t row = blockIdx.x;
    if (row >= outer * inner) return;

    const uint32_t o = row / inner;
    const uint32_t i = row % inner;
    const uint32_t base = o * axis_dim * inner + i;

    __shared__ float smem_max[32];
    __shared__ double smem_sum[32];

    float local_max = -FLT_MAX;
    for (uint32_t a = threadIdx.x; a < axis_dim; a += blockDim.x) {
        const float v = x[base + a * inner];
        local_max = fmaxf(local_max, v);
    }
    const float row_max = block_reduce_max_f32(local_max, smem_max);

    double local_sum = 0.0;
    for (uint32_t a = threadIdx.x; a < axis_dim; a += blockDim.x) {
        const float e = expf(x[base + a * inner] - row_max);
        local_sum += static_cast<double>(e);
    }
    const double row_sum = block_reduce_sum_f64(local_sum, smem_sum);
    const float inv_sum = (row_sum > 0.0) ? static_cast<float>(1.0 / row_sum) : 0.0f;

    for (uint32_t a = threadIdx.x; a < axis_dim; a += blockDim.x) {
        const uint32_t pos = base + a * inner;
        out[pos] = expf(x[pos] - row_max) * inv_sum;
    }
}

// LayerNormalization along the last dimension: input reshaped to [rows, norm_dim].
// `scale` and `bias` are optional (pass nullptr / 0 to omit).
// Launch with grid = (rows, 1), block = (256, 1).
extern "C" __global__ void layer_norm_last_f32(
    const float* __restrict__ x,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ out,
    uint32_t rows,
    uint32_t norm_dim,
    float eps
) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;

    const float* row_x = x + static_cast<size_t>(row) * norm_dim;
    float* row_out = out + static_cast<size_t>(row) * norm_dim;

    __shared__ double smem_sum[32];

    double local_sum = 0.0;
    for (uint32_t c = threadIdx.x; c < norm_dim; c += blockDim.x) {
        local_sum += static_cast<double>(row_x[c]);
    }
    const double mean = block_reduce_sum_f64(local_sum, smem_sum) / static_cast<double>(norm_dim);

    double local_var = 0.0;
    for (uint32_t c = threadIdx.x; c < norm_dim; c += blockDim.x) {
        const double diff = static_cast<double>(row_x[c]) - mean;
        local_var += diff * diff;
    }
    const double var = block_reduce_sum_f64(local_var, smem_sum) / static_cast<double>(norm_dim);
    const float mean_f = static_cast<float>(mean);
    const float inv_std = 1.0f / sqrtf(static_cast<float>(var) + eps);

    for (uint32_t c = threadIdx.x; c < norm_dim; c += blockDim.x) {
        float y = (row_x[c] - mean_f) * inv_std;
        if (scale != nullptr) {
            y *= scale[c];
        }
        if (bias != nullptr) {
            y += bias[c];
        }
        row_out[c] = y;
    }
}

// InstanceNormalization on NCHW [n, c, h, w] with hw = h * w.
// `scale` and `bias` have length `c` (optional; pass nullptr / 0 to omit).
// Launch with grid = (n * c, 1), block = (256, 1).
extern "C" __global__ void instance_norm_nchw_f32(
    const float* __restrict__ x,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ out,
    uint32_t n,
    uint32_t c,
    uint32_t hw,
    float eps
) {
    const uint32_t slice = blockIdx.x;
    if (slice >= n * c) return;

    const uint32_t ci = slice % c;
    const float* slice_x = x + static_cast<size_t>(slice) * hw;
    float* slice_out = out + static_cast<size_t>(slice) * hw;

    __shared__ double smem_sum[32];

    double local_sum = 0.0;
    for (uint32_t p = threadIdx.x; p < hw; p += blockDim.x) {
        local_sum += static_cast<double>(slice_x[p]);
    }
    const double mean = block_reduce_sum_f64(local_sum, smem_sum) / static_cast<double>(hw);

    double local_var = 0.0;
    for (uint32_t p = threadIdx.x; p < hw; p += blockDim.x) {
        const double diff = static_cast<double>(slice_x[p]) - mean;
        local_var += diff * diff;
    }
    const double var = block_reduce_sum_f64(local_var, smem_sum) / static_cast<double>(hw);
    const float mean_f = static_cast<float>(mean);
    const float inv_std = 1.0f / sqrtf(static_cast<float>(var) + eps);
    const float s = (scale != nullptr) ? scale[ci] : 1.0f;
    const float b = (bias != nullptr) ? bias[ci] : 0.0f;

    for (uint32_t p = threadIdx.x; p < hw; p += blockDim.x) {
        slice_out[p] = (slice_x[p] - mean_f) * inv_std * s + b;
    }
}

// Pre-fused BatchNormalization on NCHW [n, c, h, w] with hw = h * w.
// Consumes (scale_eff, bias_eff) produced by `bn_fold.rs`:
//   y = x * scale_eff[c] + bias_eff[c]
extern "C" __global__ void batch_norm_fused_nchw_f32(
    const float* __restrict__ x,
    const float* __restrict__ scale_eff,
    const float* __restrict__ bias_eff,
    float* __restrict__ out,
    uint32_t n,
    uint32_t c,
    uint32_t hw
) {
    const uint32_t total = n * c * hw;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const uint32_t ci = (idx / hw) % c;
    out[idx] = x[idx] * scale_eff[ci] + bias_eff[ci];
}
