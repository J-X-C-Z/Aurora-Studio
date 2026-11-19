#include "pf30025_tooth_repair.h"

// Same byte law as kernels::face_detail::tooth_repair. Compile with
// -fmad=false; the explicit binary32 intrinsics pin the CPU operation order.

__device__ __forceinline__ float p25_add(float a, float b) {
  return __fadd_rn(a, b);
}

__device__ __forceinline__ float p25_sub(float a, float b) {
  return __fsub_rn(a, b);
}

__device__ __forceinline__ float p25_mul(float a, float b) {
  return __fmul_rn(a, b);
}

// Mirrors Rust byte_away: non-finite -> 0, midpoint away from zero, clamp.
__device__ __forceinline__ unsigned char p25_byte_away(float value) {
  if (!isfinite(value)) return 0;
  const float adjusted = value >= 0.0f
      ? p25_add(value, 0.5f)
      : p25_sub(value, 0.5f);
  if (!(adjusted > 0.0f)) return 0;
  if (adjusted >= 255.0f) return 255;
  return static_cast<unsigned char>(adjusted);
}

extern "C" __global__ void pf30025_tooth_repair_kernel(
    const unsigned char* source_rgba,
    const unsigned char* repair_bgra,
    const float* teeth_mask,
    unsigned char* out_rgba,
    Pf30025ToothRepairUniforms uniforms) {
  const unsigned int x =
      uniforms.roi_x + blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y =
      uniforms.roi_y + blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= uniforms.roi_x + uniforms.roi_w ||
      y >= uniforms.roi_y + uniforms.roi_h) {
    return;
  }

  const size_t pixel = (size_t)y * (size_t)uniforms.width + (size_t)x;
  const float weight = p25_mul(teeth_mask[pixel], uniforms.degree);
  if (!(weight > 0.0f)) return;

  const size_t index = (size_t)y * (size_t)uniforms.stride + (size_t)x * 4u;
  if (weight >= 1.0f) {
    out_rgba[index] = repair_bgra[index + 2u];
    out_rgba[index + 1u] = repair_bgra[index + 1u];
    out_rgba[index + 2u] = repair_bgra[index];
    return;
  }

  for (unsigned int channel = 0; channel < 3u; ++channel) {
    const size_t repair_channel = index + (channel == 0u ? 2u : (channel == 2u ? 0u : 1u));
    const float src = (float)source_rgba[index + channel];
    const float repair = (float)repair_bgra[repair_channel];
    const float delta = p25_sub(repair, src);
    out_rgba[index + channel] = p25_byte_away(p25_add(src, p25_mul(delta, weight)));
  }
}
