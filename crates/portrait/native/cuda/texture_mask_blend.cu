/*
 * PF30009 texture_mask_blend on NVIDIA.
 *
 * Law (identical to `kernels::texture_mask_blend`):
 *   out.rgb = round(original + (effect - original) * (mask/255 * degree))
 *   out.a   = original.a
 * The effect plane is sampled through its own UV stream with clamp-to-edge
 * bilinear taps; the sample is quantized to bytes before the mix, exactly like
 * the CPU reference.
 */

#include "texture_mask_blend.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

__device__ __forceinline__ float AddRn(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float MulRn(float a, float b) { return __fmul_rn(a, b); }

__device__ __forceinline__ uint8_t Quantize(float value) {
  const float clamped = fminf(fmaxf(value, 0.0f), 255.0f);
  return static_cast<uint8_t>(roundf(clamped));
}

__device__ __forceinline__ float Texel(
    const uint8_t* image,
    int32_t width,
    int32_t x,
    int32_t y,
    int32_t channel) {
  const size_t offset =
      (static_cast<size_t>(y) * static_cast<size_t>(width) +
       static_cast<size_t>(x)) * 4u +
      static_cast<size_t>(channel);
  return static_cast<float>(image[offset]);
}

__device__ __forceinline__ void Sample(
    const uint8_t* effect,
    int32_t width,
    int32_t height,
    float u,
    float v,
    float* out) {
  const float cu = fminf(fmaxf(u, 0.0f), 1.0f);
  const float cv = fminf(fmaxf(v, 0.0f), 1.0f);
  const float px = AddRn(MulRn(cu, static_cast<float>(width)), -0.5f);
  const float py = AddRn(MulRn(cv, static_cast<float>(height)), -0.5f);
  const float fx0 = floorf(px);
  const float fy0 = floorf(py);
  const float tx = AddRn(px, -fx0);
  const float ty = AddRn(py, -fy0);
  const int32_t x0 = static_cast<int32_t>(fx0);
  const int32_t y0 = static_cast<int32_t>(fy0);
  const int32_t x0c = x0 < 0 ? 0 : (x0 > width - 1 ? width - 1 : x0);
  const int32_t y0c = y0 < 0 ? 0 : (y0 > height - 1 ? height - 1 : y0);
  const int32_t x1c = (x0c + 1) > width - 1 ? width - 1 : (x0c + 1);
  const int32_t y1c = (y0c + 1) > height - 1 ? height - 1 : (y0c + 1);
  for (int32_t c = 0; c < 4; ++c) {
    const float p00 = Texel(effect, width, x0c, y0c, c);
    const float p10 = Texel(effect, width, x1c, y0c, c);
    const float p01 = Texel(effect, width, x0c, y1c, c);
    const float p11 = Texel(effect, width, x1c, y1c, c);
    const float top = AddRn(p00, MulRn(AddRn(p10, -p00), tx));
    const float bottom = AddRn(p01, MulRn(AddRn(p11, -p01), tx));
    out[c] = static_cast<float>(Quantize(AddRn(top, MulRn(AddRn(bottom, -top), ty))));
  }
}

}  // namespace

extern "C" __global__ void texture_mask_blend_kernel(
    const uint8_t* original,
    const uint8_t* effect,
    const uint8_t* mask,
    const float* effect_uv,
    uint8_t* out,
    TextureMaskBlendUniforms u) {
  const int32_t x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int32_t y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= u.width || y >= u.height) {
    return;
  }
  const size_t index = static_cast<size_t>(y) * static_cast<size_t>(u.width) +
      static_cast<size_t>(x);

  float uu;
  float vv;
  if (u.has_uv != 0) {
    uu = effect_uv[index * 2u];
    vv = effect_uv[index * 2u + 1u];
  } else {
    uu = AddRn(static_cast<float>(x), 0.5f) / static_cast<float>(u.width);
    vv = AddRn(static_cast<float>(y), 0.5f) / static_cast<float>(u.height);
  }

  float sampled[4];
  Sample(effect, u.effect_width, u.effect_height, uu, vv, sampled);

  const float mask_unit = static_cast<float>(mask[index]) / 255.0f;
  const float m = u.reverse_alpha != 0 ? AddRn(1.0f, -mask_unit) : mask_unit;
  const float weight = MulRn(m, u.degree);

  uint8_t* target = out + index * 4u;
  for (int32_t c = 0; c < 3; ++c) {
    const float src = static_cast<float>(original[index * 4u + static_cast<size_t>(c)]);
    const float mixed = AddRn(src, MulRn(AddRn(sampled[c], -src), weight));
    target[c] = Quantize(mixed);
  }
  target[3] = original[index * 4u + 3u];
}
