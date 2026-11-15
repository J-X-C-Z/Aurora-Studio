#include "teeth_blend.h"

// Source-order translation of kernels::teeth_blend::apply.
// The vendor constant is bit-exact: 0x3B807BBC. Compile with -fmad=false and use
// explicit intrinsics so mul+add cannot contract into fma.

__device__ __forceinline__ float t_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float t_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float t_sub(float a, float b) { return __fsub_rn(a, b); }

extern "C" __global__ void teeth_blend_kernel(
    const unsigned char* original_rgba,
    const unsigned char* effect_rgba,
    const unsigned char* mask,
    unsigned char* out_rgba,
    TeethBlendUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;

  const size_t pixel = (size_t)y * (size_t)uniforms.width + (size_t)x;
  const size_t index = (size_t)y * (size_t)uniforms.stride + (size_t)x * 4u;
  const unsigned int m8 = mask[pixel];
  const float scale = __uint_as_float(0x3B807BBCu);

  if (m8 == 255u) {
    out_rgba[index + 0u] = effect_rgba[index + 0u];
    out_rgba[index + 1u] = effect_rgba[index + 1u];
    out_rgba[index + 2u] = effect_rgba[index + 2u];
    out_rgba[index + 3u] = original_rgba[index + 3u];
    return;
  }

  const float m = t_mul((float)m8, scale);
  const float one_minus = t_sub(1.0f, m);
  for (int channel = 0; channel < 3; ++channel) {
    const float high = t_mul(m, (float)effect_rgba[index + (size_t)channel]);
    const float low = t_mul((float)original_rgba[index + (size_t)channel], one_minus);
    const int truncated = (int)(t_add(high, low));
    out_rgba[index + (size_t)channel] = (unsigned char)(truncated < 0 ? 0 : (truncated > 255 ? 255 : truncated));
  }
  out_rgba[index + 3u] = original_rgba[index + 3u];
}
