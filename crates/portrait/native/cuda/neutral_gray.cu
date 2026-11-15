/*
 * PF30013 neutral-gray flattening and PF30014 neutral-gray 3D on NVIDIA.
 *
 * Both entries mirror kernels::skin_adjust. They consume only same-job inputs:
 * original/current/mask for PF30013, and source/light/dark/alpha/mask for
 * PF30014. A missing 3D asset or ROI is refused by the Rust adapter before
 * either kernel is launched.
 */

#include "neutral_gray.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr float kInv255 = 1.0f / 255.0f;
constexpr float kHslEpsilon = 0.001f;

__device__ __forceinline__ float Add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float Sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float Mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float Div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float Clamp01(float value) {
  return fminf(fmaxf(value, 0.0f), 1.0f);
}

struct Hsl {
  float h;
  float s;
  float l;
};

__device__ __forceinline__ Hsl RgbToHsl(float r, float g, float b) {
  const float base_max = g < b ? b : g;
  const float maximum = fmaxf(r, base_max);
  const float minimum = fminf(r, g < b ? g : b);
  const float chroma = Sub(maximum, minimum);
  const float w = r < base_max ? r : base_max;
  const float offset = r < base_max
      ? (g < b ? 2.0f / 3.0f : -1.0f / 3.0f)
      : (g < b ? -1.0f : 0.0f);
  const float hue_denominator =
      Add(Mul(6.0f, chroma), kHslEpsilon);
  const float h = fabsf(Add(Div(Sub(w, minimum), hue_denominator), offset));
  const float l = Sub(maximum, Mul(chroma, 0.5f));
  const float s = Div(
      chroma,
      Add(Sub(1.0f, fabsf(Sub(Mul(l, 2.0f), 1.0f))), kHslEpsilon));
  return Hsl{h, s, l};
}

__device__ __forceinline__ float Axis(float value, float chroma, float l) {
  return Clamp01(Add(Mul(Sub(value, 0.5f), chroma), l));
}

__device__ __forceinline__ void HslToRgb(
    float h,
    float s,
    float l,
    float* rgb) {
  const float x = Mul(h, 6.0f);
  const float chroma =
      Mul(Sub(1.0f, fabsf(Sub(Mul(l, 2.0f), 1.0f))), s);
  rgb[0] = Axis(Clamp01(Sub(fabsf(Sub(x, 3.0f)), 1.0f)), chroma, l);
  rgb[1] = Axis(Clamp01(Sub(2.0f, fabsf(Sub(x, 2.0f)))), chroma, l);
  rgb[2] = Axis(Clamp01(Sub(2.0f, fabsf(Sub(x, 4.0f)))), chroma, l);
}

__device__ __forceinline__ uint8_t RoundByte(float value) {
  return static_cast<uint8_t>(roundf(fminf(fmaxf(value, 0.0f), 1.0f) * 255.0f));
}

__device__ __forceinline__ uint8_t RoundEvenByte(float value) {
  return static_cast<uint8_t>(
      rintf(fminf(fmaxf(value, 0.0f), 255.0f)));
}

}  // namespace

extern "C" __global__ void skin_grey_kernel(
    const uint8_t* original,
    const uint8_t* current,
    const uint8_t* mask,
    uint8_t* out,
    SkinGreyUniforms u) {
  const int32_t x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int32_t y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= u.width || y >= u.height) {
    return;
  }

  const size_t index =
      static_cast<size_t>(y) * static_cast<size_t>(u.width) +
      static_cast<size_t>(x);
  const size_t offset = index * 4u;
  uint8_t* target = out + offset;
  target[0] = current[offset];
  target[1] = current[offset + 1u];
  target[2] = current[offset + 2u];
  target[3] = current[offset + 3u];

  const float weight =
      Clamp01(Mul(Mul(static_cast<float>(mask[index]), kInv255), u.alpha));
  if (weight <= kHslEpsilon) {
    return;
  }

  const Hsl source = RgbToHsl(
      static_cast<float>(original[offset]) * kInv255,
      static_cast<float>(original[offset + 1u]) * kInv255,
      static_cast<float>(original[offset + 2u]) * kInv255);
  const Hsl current_hsl = RgbToHsl(
      static_cast<float>(current[offset]) * kInv255,
      static_cast<float>(current[offset + 1u]) * kInv255,
      static_cast<float>(current[offset + 2u]) * kInv255);
  float projected[3];
  HslToRgb(source.h, source.s, current_hsl.l, projected);

  for (int32_t channel = 0; channel < 3; ++channel) {
    const float now =
        static_cast<float>(current[offset + static_cast<size_t>(channel)]) *
        kInv255;
    const float mixed =
        Add(now, Mul(Sub(projected[channel], now), weight));
    target[channel] = RoundByte(mixed);
  }
}

extern "C" __global__ void neutral_gray_3d_kernel(
    const uint8_t* source,
    const float* light,
    const float* dark,
    const float* alpha,
    const float* mask,
    uint8_t* out,
    NeutralGray3dUniforms u) {
  const int32_t x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int32_t y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= 512 || y >= 512) {
    return;
  }

  const size_t pixel =
      static_cast<size_t>(y) * 512u + static_cast<size_t>(x);
  const size_t offset = pixel * 3u;
  if (u.active == 0) {
    out[offset] = source[offset];
    out[offset + 1u] = source[offset + 1u];
    out[offset + 2u] = source[offset + 2u];
    return;
  }

  const float weight = Mul(alpha[pixel], mask[pixel]);
  for (int32_t channel = 0; channel < 3; ++channel) {
    const size_t at = offset + static_cast<size_t>(channel);
    const float delta = Mul(Mul(Sub(light[at], dark[at]), 127.5f), weight);
    out[at] = RoundEvenByte(Add(static_cast<float>(source[at]), delta));
  }
}
