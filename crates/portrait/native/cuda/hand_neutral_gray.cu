/*
 * PF30187 final consumer.
 *
 * weight = hand_mask/255 * model_alpha/255 * degree
 * target = explicit BGRA neutral, or round(src + 255*diff)
 * out.rgb = round_away(src*(1-weight) + target*weight)
 * out.a = src.a
 */

#include "hand_neutral_gray.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ uint8_t QuantizeAwayFromZero(float value) {
  const float clamped = fminf(fmaxf(value, 0.0f), 255.0f);
  return static_cast<uint8_t>(__float2uint_rz(__fadd_rn(clamped, 0.5f)));
}

}  // namespace

extern "C" __global__ void hand_neutral_gray_kernel(
    const uint8_t* source,
    const uint8_t* hand_mask,
    const uint8_t* model_alpha,
    const uint8_t* neutral_bgra,
    const float* model_diff_bgr,
    uint8_t* out,
    HandNeutralGrayUniforms u) {
  const int32_t x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int32_t y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= u.width || y >= u.height) {
    return;
  }

  const size_t index = static_cast<size_t>(y) * static_cast<size_t>(u.width) +
      static_cast<size_t>(x);
  const size_t rgba = index * 4u;
  const size_t rgb = index * 3u;
  const float mask_unit = __fdiv_rn(static_cast<float>(hand_mask[index]), 255.0f);
  const float alpha_unit = __fdiv_rn(static_cast<float>(model_alpha[index]), 255.0f);
  const float weight = __fmul_rn(__fmul_rn(mask_unit, alpha_unit), u.degree);
  const float inverse = __fsub_rn(1.0f, weight);

  for (int32_t channel = 0; channel < 3; ++channel) {
    const float src = static_cast<float>(source[rgba + static_cast<size_t>(channel)]);
    uint8_t target;
    if (u.target_kind == 0) {
      target = neutral_bgra[rgba + static_cast<size_t>(channel)];
    } else {
      const float delta = model_diff_bgr[rgb + static_cast<size_t>(channel)];
      target = QuantizeAwayFromZero(__fadd_rn(src, __fmul_rn(delta, 255.0f)));
    }
    const float mixed = __fadd_rn(
        __fmul_rn(src, inverse),
        __fmul_rn(static_cast<float>(target), weight));
    out[rgba + static_cast<size_t>(channel)] = QuantizeAwayFromZero(mixed);
  }
  out[rgba + 3u] = source[rgba + 3u];
}
