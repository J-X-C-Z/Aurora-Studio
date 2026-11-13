/*
 * O3b CUDA mesh rasterizer.
 *
 * Reference-grade candidate: rule parity with crates/pixcake/src/raster.rs
 * first, throughput later. Two kernels:
 *   mesh_raster_select  one thread per triangle, bbox scan, per-pixel resolve
 *   mesh_raster_emit    one thread per pixel, re-evaluate, bilinear sample
 *
 * Arithmetic goes through __fadd_rn/__fmul_rn so the compiler cannot contract
 * mul+add into fma and drift from the Rust f32 sequence.
 */

#include "mesh_raster.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr float kDegenerateArea2 = 1e-12f;
constexpr uint32_t kUncovered = 0xFFFFFFFFu;
constexpr int32_t kBlock = 128;

struct Vec2f {
  float x;
  float y;
};

__device__ __forceinline__ float AddRn(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float MulRn(float a, float b) { return __fmul_rn(a, b); }

__device__ __forceinline__ Vec2f Sub(Vec2f a, Vec2f b) {
  return Vec2f{AddRn(a.x, -b.x), AddRn(a.y, -b.y)};
}

__device__ __forceinline__ float Cross(Vec2f a, Vec2f b) {
  return AddRn(MulRn(a.x, b.y), -MulRn(a.y, b.x));
}

__device__ __forceinline__ Vec2f ReadVec2(const float* values, uint16_t index) {
  const size_t offset = static_cast<size_t>(index) * 2u;
  return Vec2f{values[offset], values[offset + 1u]};
}

__device__ __forceinline__ bool IsFront(float area2, int32_t winding) {
  const bool positive = area2 > 0.0f;
  return winding == MESH_RASTER_WINDING_POSITIVE_IS_FRONT ? positive : !positive;
}

/*
 * Signed-area weights normalized by area2, edge test, then clamp to
 * non-negative and renormalize exactly like the Rust reference.
 * Returns false when the sample is not covered.
 */
__device__ __forceinline__ bool Weights(
    Vec2f p0,
    Vec2f p1,
    Vec2f p2,
    float area2,
    float sx,
    float sy,
    int32_t edge,
    float* b0,
    float* b1,
    float* b2) {
  const Vec2f s{sx, sy};
  float w0 = Cross(Sub(p1, s), Sub(p2, s)) / area2;
  float w1 = Cross(Sub(p2, s), Sub(p0, s)) / area2;
  float w2 = Cross(Sub(p0, s), Sub(p1, s)) / area2;
  const bool inside = edge == MESH_RASTER_EDGE_INCLUSIVE
      ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
      : (w0 > 0.0f && w1 > 0.0f && w2 > 0.0f);
  if (!inside) {
    return false;
  }
  w0 = fmaxf(w0, 0.0f);
  w1 = fmaxf(w1, 0.0f);
  w2 = fmaxf(w2, 0.0f);
  const float sum = AddRn(AddRn(w0, w1), w2);
  if (!(sum > 0.0f)) {
    return false;
  }
  *b0 = w0 / sum;
  *b1 = w1 / sum;
  *b2 = w2 / sum;
  return true;
}

/* p = uv * size - 0.5 texel-center rule; clamp or repeat per axis. */
__device__ __forceinline__ void Axis(
    float pos,
    int32_t size,
    int32_t wrap,
    int32_t* i0,
    int32_t* i1,
    float* frac) {
  if (wrap == MESH_RASTER_WRAP_REPEAT) {
    const float floor_pos = floorf(pos);
    const int32_t raw = static_cast<int32_t>(floor_pos);
    const int32_t wrapped = ((raw % size) + size) % size;
    *i0 = wrapped;
    *i1 = (wrapped + 1) % size;
    *frac = AddRn(pos, -floor_pos);
    return;
  }
  const float limit = static_cast<float>(size - 1);
  const float clamped = fminf(fmaxf(pos, 0.0f), limit);
  const float floor_pos = floorf(clamped);
  const int32_t i = static_cast<int32_t>(floor_pos);
  *i0 = i;
  *i1 = i + 1 < size ? i + 1 : size - 1;
  *frac = AddRn(clamped, -floor_pos);
}

__device__ __forceinline__ float Texel(
    const uint8_t* source,
    int32_t width,
    int32_t x,
    int32_t y,
    int32_t channel) {
  const size_t offset =
      (static_cast<size_t>(y) * static_cast<size_t>(width) +
       static_cast<size_t>(x)) * 4u +
      static_cast<size_t>(channel);
  return static_cast<float>(source[offset]) / 255.0f;
}

__device__ __forceinline__ float Lerp(float a, float b, float t) {
  return AddRn(a, MulRn(AddRn(b, -a), t));
}

/* Straight-alpha bilinear tap; RGBA8 quantized with ties-to-even. */
__device__ __forceinline__ void Sample(
    const uint8_t* source,
    int32_t width,
    int32_t height,
    float u,
    float v,
    int32_t wrap,
    uint8_t* out) {
  int32_t x0, x1, y0, y1;
  float fx, fy;
  Axis(AddRn(MulRn(u, static_cast<float>(width)), -0.5f), width, wrap, &x0, &x1, &fx);
  Axis(AddRn(MulRn(v, static_cast<float>(height)), -0.5f), height, wrap, &y0, &y1, &fy);
  for (int32_t c = 0; c < 4; ++c) {
    const float top = Lerp(
        Texel(source, width, x0, y0, c), Texel(source, width, x1, y0, c), fx);
    const float bottom = Lerp(
        Texel(source, width, x0, y1, c), Texel(source, width, x1, y1, c), fx);
    const float value = Lerp(top, bottom, fy);
    const int32_t q = __float2int_rn(MulRn(value, 255.0f));
    out[c] = static_cast<uint8_t>(q < 0 ? 0 : (q > 255 ? 255 : q));
  }
}

__device__ __forceinline__ void Bounds(
    Vec2f p0,
    Vec2f p1,
    Vec2f p2,
    int32_t width,
    int32_t height,
    int32_t* x0,
    int32_t* x1,
    int32_t* y0,
    int32_t* y1) {
  const float min_x = fminf(p0.x, fminf(p1.x, p2.x));
  const float max_x = fmaxf(p0.x, fmaxf(p1.x, p2.x));
  const float min_y = fminf(p0.y, fminf(p1.y, p2.y));
  const float max_y = fmaxf(p0.y, fmaxf(p1.y, p2.y));
  // Clamp in float first so the int cast is always defined and the range
  // matches the Rust bbox exactly; extra columns are rejected by Weights.
  const float limit_x = static_cast<float>(width - 1);
  const float limit_y = static_cast<float>(height - 1);
  *x0 = static_cast<int32_t>(ceilf(fminf(fmaxf(AddRn(min_x, -0.5f), 0.0f), limit_x)));
  *x1 = static_cast<int32_t>(floorf(fminf(fmaxf(AddRn(max_x, -0.5f), 0.0f), limit_x)));
  *y0 = static_cast<int32_t>(ceilf(fminf(fmaxf(AddRn(min_y, -0.5f), 0.0f), limit_y)));
  *y1 = static_cast<int32_t>(floorf(fminf(fmaxf(AddRn(max_y, -0.5f), 0.0f), limit_y)));
}

__global__ void mesh_raster_select(
    const float* dest,
    const uint16_t* ebo,
    int32_t width,
    int32_t height,
    int32_t triangle_count,
    MeshRasterRules rules,
    uint32_t* selected,
    uint32_t* coverage) {
  const int32_t triangle = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (triangle >= triangle_count) {
    return;
  }

  const int32_t base = triangle * 3;
  const uint16_t i0 = ebo[base];
  const uint16_t i1 = ebo[base + 1];
  const uint16_t i2 = ebo[base + 2];
  if (i0 == i1 || i1 == i2 || i0 == i2) {
    return;
  }
  const Vec2f p0 = ReadVec2(dest, i0);
  const Vec2f p1 = ReadVec2(dest, i1);
  const Vec2f p2 = ReadVec2(dest, i2);
  if (!isfinite(p0.x) || !isfinite(p0.y) || !isfinite(p1.x) || !isfinite(p1.y) ||
      !isfinite(p2.x) || !isfinite(p2.y)) {
    return;
  }
  const float area2 = Cross(Sub(p1, p0), Sub(p2, p0));
  if (fabsf(area2) <= kDegenerateArea2) {
    return;
  }

  const bool front = IsFront(area2, rules.winding);
  if (rules.cull == MESH_RASTER_CULL_FRONT && front) {
    return;
  }
  if (rules.cull == MESH_RASTER_CULL_BACK && !front) {
    return;
  }

  int32_t x0, x1, y0, y1;
  Bounds(p0, p1, p2, width, height, &x0, &x1, &y0, &y1);
  if (x0 > x1 || y0 > y1) {
    return;
  }

  for (int32_t py = y0; py <= y1; ++py) {
    const float sy = AddRn(static_cast<float>(py), 0.5f);
    for (int32_t px = x0; px <= x1; ++px) {
      const float sx = AddRn(static_cast<float>(px), 0.5f);
      float b0, b1, b2;
      if (!Weights(p0, p1, p2, area2, sx, sy, rules.edge, &b0, &b1, &b2)) {
        continue;
      }
      const size_t pixel =
          static_cast<size_t>(py) * static_cast<size_t>(width) +
          static_cast<size_t>(px);
      atomicAdd(&coverage[pixel], 1u);
      if (rules.overlap == MESH_RASTER_OVERLAP_FIRST) {
        atomicMin(&selected[pixel], static_cast<uint32_t>(triangle));
      } else if (rules.overlap == MESH_RASTER_OVERLAP_LAST) {
        atomicMax(&selected[pixel], static_cast<uint32_t>(triangle));
      } else {
        // Reject: selected value is unused, emit drops multi-face pixels.
        atomicMin(&selected[pixel], static_cast<uint32_t>(triangle));
      }
    }
  }
}

__global__ void mesh_raster_emit(
    const float* dest,
    const float* source_uv,
    const uint16_t* ebo,
    const uint8_t* source,
    uint8_t* out,
    int32_t width,
    int32_t height,
    int32_t wrap,
    MeshRasterRules rules,
    const uint32_t* selected,
    const uint32_t* coverage) {
  const size_t pixel =
      static_cast<size_t>(blockIdx.x) * blockDim.x + static_cast<size_t>(threadIdx.x);
  const size_t pixel_count =
      static_cast<size_t>(width) * static_cast<size_t>(height);
  if (pixel >= pixel_count) {
    return;
  }
  if (coverage[pixel] == 0u) {
    return;
  }
  const uint32_t triangle = selected[pixel];
  if (triangle == kUncovered) {
    return;
  }
  if (coverage[pixel] > 1u && rules.overlap == MESH_RASTER_OVERLAP_REJECT) {
    return;
  }

  const int32_t px = static_cast<int32_t>(pixel % static_cast<size_t>(width));
  const int32_t py = static_cast<int32_t>(pixel / static_cast<size_t>(width));
  const int32_t base = static_cast<int32_t>(triangle) * 3;
  const uint16_t i0 = ebo[base];
  const uint16_t i1 = ebo[base + 1];
  const uint16_t i2 = ebo[base + 2];
  const Vec2f p0 = ReadVec2(dest, i0);
  const Vec2f p1 = ReadVec2(dest, i1);
  const Vec2f p2 = ReadVec2(dest, i2);
  const float area2 = Cross(Sub(p1, p0), Sub(p2, p0));
  if (fabsf(area2) <= kDegenerateArea2) {
    return;
  }

  float b0, b1, b2;
  if (!Weights(
          p0,
          p1,
          p2,
          area2,
          AddRn(static_cast<float>(px), 0.5f),
          AddRn(static_cast<float>(py), 0.5f),
          rules.edge,
          &b0,
          &b1,
          &b2)) {
    return;
  }

  const Vec2f uv0 = ReadVec2(source_uv, i0);
  const Vec2f uv1 = ReadVec2(source_uv, i1);
  const Vec2f uv2 = ReadVec2(source_uv, i2);
  const float u = AddRn(AddRn(MulRn(b0, uv0.x), MulRn(b1, uv1.x)), MulRn(b2, uv2.x));
  const float v = AddRn(AddRn(MulRn(b0, uv0.y), MulRn(b1, uv1.y)), MulRn(b2, uv2.y));
  Sample(source, width, height, u, v, wrap, out + pixel * 4u);
}

bool ValidRules(const MeshRasterRules& rules) {
  const bool winding = rules.winding == MESH_RASTER_WINDING_POSITIVE_IS_FRONT ||
      rules.winding == MESH_RASTER_WINDING_NEGATIVE_IS_FRONT;
  const bool cull = rules.cull == MESH_RASTER_CULL_NONE ||
      rules.cull == MESH_RASTER_CULL_FRONT || rules.cull == MESH_RASTER_CULL_BACK;
  const bool edge = rules.edge == MESH_RASTER_EDGE_EXCLUSIVE ||
      rules.edge == MESH_RASTER_EDGE_INCLUSIVE;
  const bool overlap = rules.overlap == MESH_RASTER_OVERLAP_REJECT ||
      rules.overlap == MESH_RASTER_OVERLAP_FIRST ||
      rules.overlap == MESH_RASTER_OVERLAP_LAST;
  return winding && cull && edge && overlap;
}

}  // namespace

extern "C" MeshRasterStatus mesh_raster_run(
    const MeshRasterInput* input,
    const MeshRasterScratch* scratch,
    void* stream) {
  if (input == nullptr || scratch == nullptr) {
    return MESH_RASTER_INVALID_ARGUMENT;
  }
  if (input->dest == nullptr || input->source_uv == nullptr || input->ebo == nullptr ||
      input->source == nullptr || input->out == nullptr ||
      scratch->selected == nullptr || scratch->coverage == nullptr) {
    return MESH_RASTER_INVALID_ARGUMENT;
  }
  if (input->width <= 0 || input->height <= 0) {
    return MESH_RASTER_INVALID_ARGUMENT;
  }
  if (input->vertex_count < 3 || input->triangle_count <= 0) {
    return MESH_RASTER_INVALID_ARGUMENT;
  }
  if (input->wrap != MESH_RASTER_WRAP_CLAMP && input->wrap != MESH_RASTER_WRAP_REPEAT) {
    return MESH_RASTER_RULES_REQUIRED;
  }
  if (!ValidRules(input->rules)) {
    return MESH_RASTER_RULES_REQUIRED;
  }
  for (int32_t i = 0; i < input->triangle_count * 3; ++i) {
    if (static_cast<int32_t>(input->ebo[i]) >= input->vertex_count) {
      return MESH_RASTER_OUT_OF_RANGE;
    }
  }

  const size_t pixel_count =
      static_cast<size_t>(input->width) * static_cast<size_t>(input->height);
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  if (cudaMemsetAsync(
          scratch->selected, 0xFF, pixel_count * sizeof(uint32_t), cuda_stream) !=
      cudaSuccess) {
    return MESH_RASTER_CUDA_ERROR;
  }
  if (cudaMemsetAsync(
          scratch->coverage, 0, pixel_count * sizeof(uint32_t), cuda_stream) !=
      cudaSuccess) {
    return MESH_RASTER_CUDA_ERROR;
  }

  const int32_t triangle_grid = (input->triangle_count + kBlock - 1) / kBlock;
  mesh_raster_select<<<triangle_grid, kBlock, 0, cuda_stream>>>(
      input->dest,
      input->ebo,
      input->width,
      input->height,
      input->triangle_count,
      input->rules,
      scratch->selected,
      scratch->coverage);

  const int32_t pixel_grid =
      static_cast<int32_t>((pixel_count + kBlock - 1) / static_cast<size_t>(kBlock));
  mesh_raster_emit<<<pixel_grid, kBlock, 0, cuda_stream>>>(
      input->dest,
      input->source_uv,
      input->ebo,
      input->source,
      input->out,
      input->width,
      input->height,
      input->wrap,
      input->rules,
      scratch->selected,
      scratch->coverage);

  const cudaError_t launch_status = cudaGetLastError();
  const cudaError_t sync_status = cudaStreamSynchronize(cuda_stream);
  if (launch_status != cudaSuccess || sync_status != cudaSuccess) {
    return MESH_RASTER_CUDA_ERROR;
  }
  return MESH_RASTER_OK;
}

extern "C" const char* mesh_raster_status_name(MeshRasterStatus status) {
  switch (status) {
    case MESH_RASTER_OK:
      return "OK";
    case MESH_RASTER_INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case MESH_RASTER_RULES_REQUIRED:
      return "RULES_REQUIRED";
    case MESH_RASTER_OUT_OF_RANGE:
      return "OUT_OF_RANGE";
    case MESH_RASTER_CUDA_ERROR:
      return "CUDA_ERROR";
  }
  return "UNKNOWN";
}
