/*
 * PF8025/PF8026 V2FuseOffset.
 *
 * The recovered pass samples an optional encoded base target and Grid_2005,
 * decodes both to signed offsets, adds the left/right degree-scaled Grid term,
 * then writes the four-channel positive/negative encoding. The source target is
 * copied outside the configured 255x171 placement rectangle.
 */

#include "v2fuseoffset.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

__device__ __forceinline__ float AddRn(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float MulRn(float a, float b) { return __fmul_rn(a, b); }

__device__ __forceinline__ float Lerp(float a, float b, float t) {
  return AddRn(a, MulRn(AddRn(b, -a), t));
}

__device__ __forceinline__ int32_t ClampIndex(float value, int32_t limit) {
  const int32_t index = static_cast<int32_t>(value);
  return index < 0 ? 0 : (index > limit ? limit : index);
}

__device__ __forceinline__ void SampleF32(
    const float* source,
    int32_t width,
    int32_t height,
    float u,
    float v,
    float* out) {
  const float w = static_cast<float>(width);
  const float h = static_cast<float>(height);
  const float x = fminf(fmaxf(AddRn(MulRn(u, w), -0.5f), 0.0f), w - 1.0f);
  const float y = fminf(fmaxf(AddRn(MulRn(v, h), -0.5f), 0.0f), h - 1.0f);
  const float x0f = floorf(x);
  const float y0f = floorf(y);
  const float x1f = fminf(AddRn(x0f, 1.0f), w - 1.0f);
  const float y1f = fminf(AddRn(y0f, 1.0f), h - 1.0f);
  const float fx = AddRn(x, -x0f);
  const float fy = AddRn(y, -y0f);
  const int32_t x0 = ClampIndex(x0f, width - 1);
  const int32_t y0 = ClampIndex(y0f, height - 1);
  const int32_t x1 = ClampIndex(x1f, width - 1);
  const int32_t y1 = ClampIndex(y1f, height - 1);
  for (int32_t channel = 0; channel < 4; ++channel) {
    const size_t p00 =
        (static_cast<size_t>(y0) * static_cast<size_t>(width) +
         static_cast<size_t>(x0)) * 4u +
        static_cast<size_t>(channel);
    const size_t p10 =
        (static_cast<size_t>(y0) * static_cast<size_t>(width) +
         static_cast<size_t>(x1)) * 4u +
        static_cast<size_t>(channel);
    const size_t p01 =
        (static_cast<size_t>(y1) * static_cast<size_t>(width) +
         static_cast<size_t>(x0)) * 4u +
        static_cast<size_t>(channel);
    const size_t p11 =
        (static_cast<size_t>(y1) * static_cast<size_t>(width) +
         static_cast<size_t>(x1)) * 4u +
        static_cast<size_t>(channel);
    const float top = Lerp(source[p00], source[p10], fx);
    const float bottom = Lerp(source[p01], source[p11], fx);
    out[channel] = Lerp(top, bottom, fy);
  }
}

__device__ __forceinline__ void SampleU8(
    const uint8_t* source,
    int32_t width,
    int32_t height,
    float u,
    float v,
    float* out) {
  const float w = static_cast<float>(width);
  const float h = static_cast<float>(height);
  const float x = fminf(fmaxf(AddRn(MulRn(u, w), -0.5f), 0.0f), w - 1.0f);
  const float y = fminf(fmaxf(AddRn(MulRn(v, h), -0.5f), 0.0f), h - 1.0f);
  const float x0f = floorf(x);
  const float y0f = floorf(y);
  const float x1f = fminf(AddRn(x0f, 1.0f), w - 1.0f);
  const float y1f = fminf(AddRn(y0f, 1.0f), h - 1.0f);
  const float fx = AddRn(x, -x0f);
  const float fy = AddRn(y, -y0f);
  const int32_t x0 = ClampIndex(x0f, width - 1);
  const int32_t y0 = ClampIndex(y0f, height - 1);
  const int32_t x1 = ClampIndex(x1f, width - 1);
  const int32_t y1 = ClampIndex(y1f, height - 1);
  for (int32_t channel = 0; channel < 4; ++channel) {
    const size_t p00 =
        (static_cast<size_t>(y0) * static_cast<size_t>(width) +
         static_cast<size_t>(x0)) * 4u +
        static_cast<size_t>(channel);
    const size_t p10 =
        (static_cast<size_t>(y0) * static_cast<size_t>(width) +
         static_cast<size_t>(x1)) * 4u +
        static_cast<size_t>(channel);
    const size_t p01 =
        (static_cast<size_t>(y1) * static_cast<size_t>(width) +
         static_cast<size_t>(x0)) * 4u +
        static_cast<size_t>(channel);
    const size_t p11 =
        (static_cast<size_t>(y1) * static_cast<size_t>(width) +
         static_cast<size_t>(x1)) * 4u +
        static_cast<size_t>(channel);
    const float top =
        Lerp(static_cast<float>(source[p00]) / 255.0f,
             static_cast<float>(source[p10]) / 255.0f,
             fx);
    const float bottom =
        Lerp(static_cast<float>(source[p01]) / 255.0f,
             static_cast<float>(source[p11]) / 255.0f,
             fx);
    out[channel] = Lerp(top, bottom, fy);
  }
}

}  // namespace

extern "C" __global__ void v2fuseoffset_kernel(
    const float* base,
    const uint8_t* grid,
    float* out,
    V2FuseUniforms uniforms) {
  const int32_t pixel =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  constexpr int32_t kPixelCount = V2FUSE_WIDTH * V2FUSE_HEIGHT;
  if (pixel >= kPixelCount) {
    return;
  }
  const int32_t px = pixel % V2FUSE_WIDTH;
  const int32_t py = pixel / V2FUSE_WIDTH;
  float* dst = out + static_cast<size_t>(pixel) * 4u;
  const float* src = base + static_cast<size_t>(pixel) * 4u;
  for (int32_t channel = 0; channel < 4; ++channel) {
    dst[channel] = src[channel];
  }
  if (px < V2FUSE_RECT_X ||
      px >= V2FUSE_RECT_X + V2FUSE_RECT_WIDTH ||
      py < V2FUSE_RECT_Y ||
      py >= V2FUSE_RECT_Y + V2FUSE_RECT_HEIGHT) {
    return;
  }

  const float target_u =
      MulRn(AddRn(static_cast<float>(px), 0.5f), 1.0f / 280.0f);
  const float target_v =
      MulRn(AddRn(static_cast<float>(py), 0.5f), 1.0f / 280.0f);
  const float grid_u =
      MulRn(AddRn(static_cast<float>(px - V2FUSE_RECT_X), 0.5f), 1.0f / 255.0f);
  const float grid_v =
      MulRn(AddRn(static_cast<float>(py - V2FUSE_RECT_Y), 0.5f), 1.0f / 171.0f);

  float base_rgba[4];
  float grid_rgba[4];
  SampleF32(base, V2FUSE_WIDTH, V2FUSE_HEIGHT, target_u, target_v, base_rgba);
  SampleU8(grid, V2FUSE_GRID_WIDTH, V2FUSE_GRID_HEIGHT, grid_u, grid_v, grid_rgba);

  const float factor =
      grid_u >= 0.5f ? uniforms.right_degree : uniforms.left_degree;
  const float base_x = MulRn(AddRn(base_rgba[0], -base_rgba[1]), 0.25f);
  const float base_y = MulRn(AddRn(base_rgba[2], -base_rgba[3]), 0.25f);
  const float grid_x = MulRn(AddRn(grid_rgba[0], -grid_rgba[1]), 0.25f);
  const float grid_y = MulRn(AddRn(grid_rgba[2], -grid_rgba[3]), 0.25f);
  const float dx = AddRn(base_x, MulRn(grid_x, factor));
  const float dy = AddRn(base_y, MulRn(grid_y, factor));

  dst[0] = MulRn(fmaxf(dx, 0.0f), 4.0f);
  dst[1] = MulRn(fmaxf(AddRn(-dx, 0.0f), 0.0f), 4.0f);
  dst[2] = MulRn(fmaxf(dy, 0.0f), 4.0f);
  dst[3] = MulRn(fmaxf(AddRn(-dy, 0.0f), 0.0f), 4.0f);
}
