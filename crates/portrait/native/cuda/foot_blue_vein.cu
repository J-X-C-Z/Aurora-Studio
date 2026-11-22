#include "foot_blue_vein.h"

// Same order as kernels::hair_body::foot_blue_vein's composite loop.
// Build with -fmad=false. Explicit binary32 intrinsics keep the CPU parity
// contract from changing if compiler contraction settings drift.

__device__ __forceinline__ float fbv_add(float a, float b) {
  return __fadd_rn(a, b);
}

__device__ __forceinline__ float fbv_sub(float a, float b) {
  return __fsub_rn(a, b);
}

__device__ __forceinline__ float fbv_mul(float a, float b) {
  return __fmul_rn(a, b);
}

__device__ __forceinline__ float fbv_div(float a, float b) {
  return __fdiv_rn(a, b);
}

// Rust f32::round: nearest integer, midpoint away from zero.
__device__ __forceinline__ unsigned char fbv_to_byte(float value) {
  if (!(value > 0.0f)) return 0;
  if (value >= 255.0f) return 255;
  return (unsigned char)roundf(value);
}

extern "C" __global__ void foot_blue_vein_blend_kernel(
    const unsigned char* source,
    const unsigned char* repair,
    const unsigned char* feathered_mask,
    unsigned char* out,
    FootBlueVeinUniforms uniforms) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < uniforms.roi_x || y < uniforms.roi_y) return;
  if (x >= uniforms.roi_x + uniforms.roi_w ||
      y >= uniforms.roi_y + uniforms.roi_h) {
    return;
  }

  const size_t pixel =
      (size_t)y * (size_t)uniforms.width + (size_t)x;
  const size_t index =
      (size_t)y * (size_t)uniforms.stride + (size_t)x * 4u;
  const float weight = fbv_mul(
      fbv_div((float)feathered_mask[pixel], 255.0f), uniforms.degree);
  if (!(weight > 0.0f)) return;

  const float rest = fbv_sub(1.0f, weight);
  for (unsigned int channel = 0; channel < 3u; ++channel) {
    const float src = (float)source[index + channel];
    const float dst = (float)repair[index + channel];
    const float value = fbv_add(
        fbv_mul(src, rest), fbv_mul(dst, weight));
    out[index + channel] = fbv_to_byte(value);
  }
  out[index + 3u] = source[index + 3u];
}
