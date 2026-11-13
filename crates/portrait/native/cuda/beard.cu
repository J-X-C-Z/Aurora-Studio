#include "beard.h"

// Same order as kernels::beard::pixel: mixed = (w*t + (255-w)*b)/255, then
// base + strength*(mixed-base), nearest-even, clamp. No FMA contraction.

__device__ __forceinline__ float b_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float b_mul(float a, float b) { return __fmul_rn(a, b); }

__device__ __forceinline__ unsigned char b_quantize(float value) {
  if (!(value > 0.0f)) return 0;
  if (value >= 255.0f) return 255;
  return (unsigned char)rintf(value);
}

extern "C" __global__ void beard_blend_kernel(
    const unsigned char* source,
    const unsigned char* repair,
    const unsigned char* mask_a,
    const unsigned char* mask_b,
    unsigned char* out,
    BeardUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;

  const size_t pixel = (size_t)y * (size_t)uniforms.width + (size_t)x;
  const size_t index = (size_t)y * (size_t)uniforms.stride + (size_t)x * 4u;
  const float weight = (float)(mask_a[pixel] < mask_b[pixel] ? mask_a[pixel] : mask_b[pixel]);
  const float rest = b_add(255.0f, -weight);
  const float scale = 0.00392156862745098f;

  for (int channel = 0; channel < 3; ++channel) {
    const float base = (float)source[index + (size_t)channel];
    const float target = (float)repair[index + (size_t)channel];
    const float mixed = b_mul(b_add(b_mul(weight, target), b_mul(rest, base)), scale);
    const float value = b_add(base, b_mul(uniforms.strength, b_add(mixed, -base)));
    out[index + (size_t)channel] = b_quantize(value);
  }
  out[index + 3u] = source[index + 3u];
}
