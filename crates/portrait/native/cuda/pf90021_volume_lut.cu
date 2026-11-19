#include <cuda_runtime.h>
#include <stdint.h>

#include "pf90021_volume_lut.h"

namespace {

constexpr int kLutEdge = 64;
constexpr float kLutSpan = 63.0f;

__device__ __forceinline__ float clamp01(float value) {
  return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ __forceinline__ float div255(uint8_t value) {
  // Match the CPU oracle at one-LSB rounding boundaries.
  return __fdiv_rn(static_cast<float>(value), 255.0f);
}

__device__ __forceinline__ float lerp_rn(float a, float b, float t) {
  return __fadd_rn(a, __fmul_rn(__fsub_rn(b, a), t));
}

__device__ __forceinline__ float3 lerp3(float3 a, float3 b, float t) {
  return make_float3(lerp_rn(a.x, b.x, t), lerp_rn(a.y, b.y, t),
                     lerp_rn(a.z, b.z, t));
}

__device__ __forceinline__ float3 load_node(const float* volume, int x, int y,
                                            int z) {
  const size_t index =
      ((static_cast<size_t>(z) * kLutEdge + y) * kLutEdge + x) * 3;
  return make_float3(volume[index], volume[index + 1], volume[index + 2]);
}

__device__ __forceinline__ float3 sample_lut64(const float* volume,
                                               float3 rgb) {
  const float xf = __fmul_rn(clamp01(rgb.x), kLutSpan);
  const float yf = __fmul_rn(clamp01(rgb.y), kLutSpan);
  const float zf = __fmul_rn(clamp01(rgb.z), kLutSpan);

  const int x0 = static_cast<int>(floorf(xf));
  const int y0 = static_cast<int>(floorf(yf));
  const int z0 = static_cast<int>(floorf(zf));
  const int x1 = min(x0 + 1, kLutEdge - 1);
  const int y1 = min(y0 + 1, kLutEdge - 1);
  const int z1 = min(z0 + 1, kLutEdge - 1);

  const float tx = __fsub_rn(xf, static_cast<float>(x0));
  const float ty = __fsub_rn(yf, static_cast<float>(y0));
  const float tz = __fsub_rn(zf, static_cast<float>(z0));

  const float3 c000 = load_node(volume, x0, y0, z0);
  const float3 c100 = load_node(volume, x1, y0, z0);
  const float3 c010 = load_node(volume, x0, y1, z0);
  const float3 c110 = load_node(volume, x1, y1, z0);
  const float3 c001 = load_node(volume, x0, y0, z1);
  const float3 c101 = load_node(volume, x1, y0, z1);
  const float3 c011 = load_node(volume, x0, y1, z1);
  const float3 c111 = load_node(volume, x1, y1, z1);

  const float3 c00 = lerp3(c000, c100, tx);
  const float3 c01 = lerp3(c010, c110, tx);
  const float3 c10 = lerp3(c001, c101, tx);
  const float3 c11 = lerp3(c011, c111, tx);
  const float3 c0 = lerp3(c00, c01, ty);
  const float3 c1 = lerp3(c10, c11, ty);
  return lerp3(c0, c1, tz);
}

__device__ __forceinline__ uint8_t half_up_u8(float value) {
  const float scaled = __fadd_rn(__fmul_rn(clamp01(value), 255.0f), 0.5f);
  return static_cast<uint8_t>(
      min(255, static_cast<int>(floorf(scaled))));
}

__device__ __forceinline__ int pixel_index(uint32_t width, uint32_t height,
                                           uint32_t stride, uint32_t bpp) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return -1;
  }
  return static_cast<int>(y * stride + x * bpp);
}

}  // namespace

extern "C" __global__ void pf90021_lut64_pass_kernel(
    const uint8_t* source, const float* volume, float* output,
    const Pf90021LutPassParams params) {
  const int source_index =
      pixel_index(params.width, params.height, params.source_stride, 4);
  if (source_index < 0) {
    return;
  }

  const uint8_t* src = source + source_index;
  const float3 rgb = make_float3(
      div255(src[0]), div255(src[1]), div255(src[2]));
  const float3 mapped = sample_lut64(volume, rgb);
  const int output_index =
      pixel_index(params.width, params.height, params.output_stride, 16);
  uint8_t* output_bytes = reinterpret_cast<uint8_t*>(output);
  float4* dst = reinterpret_cast<float4*>(output_bytes + output_index);
  *dst = make_float4(clamp01(mapped.x), clamp01(mapped.y), clamp01(mapped.z),
                     div255(src[3]));
}

extern "C" __global__ void pf90021_mask_composite_kernel(
    const uint8_t* source, const float* effect, const uint8_t* mask,
    uint8_t* output, const Pf90021CompositeParams params) {
  const int source_index =
      pixel_index(params.width, params.height, params.source_stride, 4);
  if (source_index < 0) {
    return;
  }

  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  const int effect_index =
      static_cast<int>(y * params.effect_stride + x * 16);
  const int mask_index = static_cast<int>(y * params.mask_stride + x);
  const int output_index =
      static_cast<int>(y * params.output_stride + x * 4);

  const uint8_t* src = source + source_index;
  const uint8_t* effect_bytes = reinterpret_cast<const uint8_t*>(effect);
  const float4* mapped =
      reinterpret_cast<const float4*>(effect_bytes + effect_index);
  const float mask_weight = div255(mask[mask_index]);
  const float alpha = __fmul_rn(clamp01(params.u_alpha), mask_weight);

  const float src_r = div255(src[0]);
  const float src_g = div255(src[1]);
  const float src_b = div255(src[2]);
  uint8_t* dst = output + output_index;
  dst[0] = half_up_u8(lerp_rn(src_r, mapped->x, alpha));
  dst[1] = half_up_u8(lerp_rn(src_g, mapped->y, alpha));
  dst[2] = half_up_u8(lerp_rn(src_b, mapped->z, alpha));
  dst[3] = src[3];
}
