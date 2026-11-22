#include "stray_hair.h"

// Same operation order as kernels::hair_body::stray_hair. Explicit rounded
// intrinsics prevent FMA contraction; roundf is half-away from zero.

__device__ __forceinline__ float sh_add(float a, float b) {
  return __fadd_rn(a, b);
}

__device__ __forceinline__ float sh_mul(float a, float b) {
  return __fmul_rn(a, b);
}

__device__ __forceinline__ float sh_div(float a, float b) {
  return __fdiv_rn(a, b);
}

__device__ __forceinline__ unsigned char sh_round_byte(float value) {
  const float clamped = fminf(fmaxf(value, 0.0f), 255.0f);
  return static_cast<unsigned char>(roundf(clamped));
}

extern "C" __global__ void pf90088_stray_hair_kernel(
    const unsigned char* source,
    const unsigned char* repair,
    const unsigned char* raw_mask,
    const unsigned char* person_mask,
    const unsigned char* background_mask,
    unsigned char* out,
    StrayHairUniforms uniforms) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= static_cast<int>(uniforms.width) ||
      y >= static_cast<int>(uniforms.height)) {
    return;
  }

  const size_t pixel =
      static_cast<size_t>(y) * static_cast<size_t>(uniforms.width) +
      static_cast<size_t>(x);
  const size_t offset =
      static_cast<size_t>(y) * static_cast<size_t>(uniforms.stride) +
      static_cast<size_t>(x) * 4u;
  const float weight = sh_div(static_cast<float>(raw_mask[pixel]), 255.0f);
  const float person = uniforms.has_person != 0
      ? sh_div(static_cast<float>(person_mask[pixel]), 255.0f)
      : 1.0f;
  const float background = uniforms.has_background != 0
      ? sh_div(static_cast<float>(background_mask[pixel]), 255.0f)
      : 1.0f;
  const float person_region = sh_mul(person, uniforms.person_area);
  const float background_region =
      sh_mul(background, uniforms.background_area);
  const float region = fmaxf(person_region, background_region);
  const float alpha = fminf(
      fmaxf(sh_mul(sh_mul(weight, uniforms.degree), region), 0.0f),
      1.0f);

  for (int channel = 0; channel < 3; ++channel) {
    const float src =
        static_cast<float>(source[offset + static_cast<size_t>(channel)]);
    const float dst =
        static_cast<float>(repair[offset + static_cast<size_t>(channel)]);
    const float mixed = sh_add(src, sh_mul(sh_add(dst, -src), alpha));
    out[offset + static_cast<size_t>(channel)] = sh_round_byte(mixed);
  }
  out[offset + 3u] = source[offset + 3u];
}
