#include "pf30025_tooth_repair.h"

// Same byte law as kernels::face_detail::tooth_repair. Compile with
// -fmad=false; the explicit binary32 intrinsics pin the vendor operation order.

__device__ __forceinline__ float p25_add(float a, float b) {
  return __fadd_rn(a, b);
}

__device__ __forceinline__ float p25_sub(float a, float b) {
  return __fsub_rn(a, b);
}

__device__ __forceinline__ float p25_mul(float a, float b) {
  return __fmul_rn(a, b);
}

// Mirrors vendor cvttss2si: truncate toward zero, then clamp to a byte.
__device__ __forceinline__ unsigned char p25_trunc_byte(float value) {
  if (!isfinite(value)) return 0;
  const float truncated = truncf(value);
  if (!(truncated > 0.0f)) return 0;
  if (truncated >= 255.0f) return 255;
  return static_cast<unsigned char>(truncated);
}

extern "C" __global__ void pf30025_tooth_repair_kernel(
    const unsigned char* source_rgba,
    const unsigned char* repair_bgra,
    const float* gate,
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
  const float weight = p25_mul(gate[pixel], uniforms.degree);
  if (weight == 0.0f) return;

  const size_t index = (size_t)y * (size_t)uniforms.stride + (size_t)x * 4u;
  const float source_weight = p25_sub(1.0f, weight);
  for (unsigned int channel = 0; channel < 3u; ++channel) {
    const size_t repair_channel = index + (channel == 0u ? 2u : (channel == 2u ? 0u : 1u));
    const float src = (float)source_rgba[index + channel];
    const float repair = (float)repair_bgra[repair_channel];
    const float source_term = p25_mul(src, source_weight);
    const float repair_term = p25_mul(repair, weight);
    out_rgba[index + channel] =
        p25_trunc_byte(p25_add(source_term, repair_term));
  }
}
