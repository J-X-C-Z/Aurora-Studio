#include "eye_bag.h"

// PF90116 recovered pixel law, identical to the R3 CPU oracle:
//   q = saturate_u8(cvRound(255 * response))
//   m = q * (1/255) * degree
//   out = clamp(cvttss2si(color*m + src*(1-m)), 0, 255)
//   m == 0 or NaN -> copy; alpha follows src.
// Explicit binary32 intrinsics preserve the native separate mul/add rounding.

__device__ __forceinline__ float e_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float e_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float e_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float e_div(float a, float b) { return __fdiv_rn(a, b); }

// cvttss2si: truncate toward zero.
__device__ __forceinline__ int e_trunc_to_int(float value) {
  if (!isfinite(value) || value >= 2147483648.0f || value < -2147483648.0f)
    return (-2147483647 - 1);
  return __float2int_rz(value);
}

extern "C" __global__ void eye_bag_compose_kernel(
    const unsigned char* source_rgba,
    const float* color_rgb,
    const float* response,
    unsigned char* out_rgba,
    EyeBagUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;

  const size_t pixel = (size_t)y * (size_t)uniforms.width + (size_t)x;
  const size_t index = (size_t)y * (size_t)uniforms.stride + (size_t)x * 4u;

  out_rgba[index + 0u] = source_rgba[index + 0u];
  out_rgba[index + 1u] = source_rgba[index + 1u];
  out_rgba[index + 2u] = source_rgba[index + 2u];
  out_rgba[index + 3u] = source_rgba[index + 3u];

  const float scaled = e_mul(255.0f, response[pixel]);
  const int rounded = (!isfinite(scaled) || scaled >= 2147483648.0f || scaled < -2147483648.0f)
      ? (-2147483647 - 1) : __float2int_rn(scaled);
  const float q = (float)(rounded < 0 ? 0 : (rounded > 255 ? 255 : rounded));
  const float m = e_mul(e_mul(q, 0.00392156862745098f), uniforms.degree);
  if (m == 0.0f || isnan(m)) return;

  const float one_minus = e_sub(1.0f, m);
  for (int channel = 0; channel < 3; ++channel) {
    const float color = color_rgb[pixel * 3u + (size_t)channel];
    const float source = (float)source_rgba[index + (size_t)channel];
    const float blended = e_add(e_mul(color, m), e_mul(source, one_minus));
    int value = e_trunc_to_int(blended);
    value = value < 0 ? 0 : (value > 255 ? 255 : value);
    out_rgba[index + (size_t)channel] = (unsigned char)value;
  }
}
