/*
 * PF90139/PF90140 hair smooth and PF90160 hair gloss consumer slices.
 *
 * PF90139 descriptor +0x80 is the PF90140 mix strength; descriptor +0x88 is
 * the PF90139 detail strength. The CUDA path uses the vendor SIMD conversion:
 * nearest-even for PF90139 and truncation toward zero for PF90160.
 */

#include "hair_detail.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr float kInv255 = 0.003921568859368563f;
constexpr float kDetailCenter = 127.5f;
constexpr float kDetailScale = 4.0f;
constexpr float kDetailGain = 0.8f;
constexpr float kGlossScale = 2.0f;

__device__ __forceinline__ float AddRn(const float a, const float b) {
  return __fadd_rn(a, b);
}

__device__ __forceinline__ float MulRn(const float a, const float b) {
  return __fmul_rn(a, b);
}

__device__ __forceinline__ float ClampByte(const float value) {
  if (!(value > 0.0f)) {
    return 0.0f;
  }
  return value >= 255.0f ? 255.0f : value;
}

__device__ __forceinline__ uint8_t RoundNearestEvenByte(const float value) {
  return static_cast<uint8_t>(__float2int_rn(ClampByte(value)));
}

__device__ __forceinline__ uint8_t TruncByte(const float value) {
  return static_cast<uint8_t>(__float2int_rz(ClampByte(value)));
}

}  // namespace

extern "C" __global__ void pf90139_hair_smooth_kernel(
    const uint8_t* source,
    const uint8_t* hair_mask,
    const uint8_t* mix_mask,
    const uint8_t* blend_rgb,
    const uint8_t* detail,
    uint8_t* output,
    HairSmoothUniforms u) {
  const uint32_t rx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t ry = blockIdx.y * blockDim.y + threadIdx.y;
  if (rx >= u.roi_width || ry >= u.roi_height) {
    return;
  }

  const uint32_t x = u.roi_x + rx;
  const uint32_t y = u.roi_y + ry;
  const size_t source_offset =
      static_cast<size_t>(y) * u.source_stride + static_cast<size_t>(x) * 4u;
  const size_t plane_offset = static_cast<size_t>(ry);
  const uint8_t hair_byte =
      hair_mask[plane_offset * u.hair_stride + rx];
  if (hair_byte == 0) {
    return;
  }

  const uint8_t mix_byte =
      mix_mask[plane_offset * u.mix_stride + rx];
  const uint8_t detail_byte =
      detail[plane_offset * u.detail_stride + rx];
  const size_t blend_offset =
      plane_offset * u.blend_stride + static_cast<size_t>(rx) * 3u;

  const float gate = MulRn(static_cast<float>(hair_byte), kInv255);
  const float mix_weight =
      MulRn(MulRn(MulRn(static_cast<float>(mix_byte), kInv255),
                  u.mix_strength),
            gate);
  const float detail_add =
      MulRn(MulRn(AddRn(static_cast<float>(detail_byte), -kDetailCenter),
                  kDetailScale),
            MulRn(MulRn(gate, u.detail_strength), kDetailGain));

#pragma unroll
  for (uint32_t channel = 0; channel < 3; ++channel) {
    const float src = static_cast<float>(source[source_offset + channel]);
    float base = src;
    if (mix_weight > 0.0f) {
      const float blend = static_cast<float>(blend_rgb[blend_offset + channel]);
      base = AddRn(MulRn(src, AddRn(1.0f, -mix_weight)),
                   MulRn(blend, mix_weight));
    }
    output[source_offset + channel] =
        RoundNearestEvenByte(AddRn(base, detail_add));
  }
}

extern "C" __global__ void pf90160_hair_gloss_kernel(
    const uint8_t* source,
    const uint8_t* hair_mask,
    const uint8_t* model_detail,
    uint8_t* output,
    HairGlossUniforms u) {
  const uint32_t rx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t ry = blockIdx.y * blockDim.y + threadIdx.y;
  if (rx >= u.roi_width || ry >= u.roi_height) {
    return;
  }

  const uint32_t x = u.roi_x + rx;
  const uint32_t y = u.roi_y + ry;
  const size_t source_offset =
      static_cast<size_t>(y) * u.source_stride + static_cast<size_t>(x) * 4u;
  const size_t plane_offset = static_cast<size_t>(ry);
  const uint8_t mask_byte =
      hair_mask[plane_offset * u.hair_stride + rx];
  if (mask_byte == 0) {
    return;
  }
  const uint8_t detail_byte =
      model_detail[plane_offset * u.detail_stride + rx];

  const float centered =
      MulRn(AddRn(static_cast<float>(detail_byte), -kDetailCenter),
            kGlossScale);
  const float delta =
      MulRn(MulRn(MulRn(centered, static_cast<float>(mask_byte)), kInv255),
            u.scale);

#pragma unroll
  for (uint32_t channel = 0; channel < 3; ++channel) {
    const float sum =
        AddRn(static_cast<float>(source[source_offset + channel]), delta);
    output[source_offset + channel] = TruncByte(sum);
  }
}
