/*
 * PF90080 final-consumer candidate.
 *
 * out.rgb = nearest-even(src + (mask/255 * degree) * (effect - src))
 * out.a   = src.a
 *
 * Effect RGB and mask are same-job full-frame producer planes. This kernel
 * performs no lane selection, ROI crop, upscale, colour conversion, or
 * framebuffer blend.
 */

#include "oil_shine_blend.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

__device__ __forceinline__ uint8_t QuantizeNearestEven(float value) {
  const float clamped = fminf(fmaxf(value, 0.0f), 255.0f);
  return static_cast<uint8_t>(nearbyintf(clamped));
}

}  // namespace

extern "C" __global__ void oil_shine_mask_blend_kernel(
    const uint8_t* source,
    const uint8_t* effect_rgb,
    const uint8_t* mask,
    uint8_t* out,
    OilShineBlendUniforms u) {
  const int32_t x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int32_t y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= u.width || y >= u.height) {
    return;
  }

  const size_t index = static_cast<size_t>(y) * static_cast<size_t>(u.width) +
      static_cast<size_t>(x);
  const size_t rgba = index * 4u;
  const size_t rgb = index * 3u;
  const float mask_unit = __fdiv_rn(static_cast<float>(mask[index]), 255.0f);
  const float weight = __fmul_rn(mask_unit, u.degree);

  for (int32_t c = 0; c < 3; ++c) {
    const float src = static_cast<float>(source[rgba + static_cast<size_t>(c)]);
    const float effect = static_cast<float>(effect_rgb[rgb + static_cast<size_t>(c)]);
    const float mixed = __fadd_rn(src, __fmul_rn(__fsub_rn(effect, src), weight));
    out[rgba + static_cast<size_t>(c)] = QuantizeNearestEven(mixed);
  }
  out[rgba + 3u] = source[rgba + 3u];
}
