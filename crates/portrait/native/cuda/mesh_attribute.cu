#include "mesh_attribute.h"

// Source-order translation of kernels::mesh_attribute::rasterize.
// Pixel centres, signed cross area, |area| < 1e-8 skip, edge tolerance -1e-6,
// and triangle-order last-writer coverage. The write is resolved per pixel from
// the highest covering triangle index, which is exactly what the CPU's
// sequential overwrite leaves behind.

namespace {

struct Vec2f {
  float x;
  float y;
};

__device__ __forceinline__ float a_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float a_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float a_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float a_div(float a, float b) { return __fdiv_rn(a, b); }

__device__ __forceinline__ Vec2f read_vec2(const float* values, unsigned int index) {
  const size_t offset = (size_t)index * 2u;
  return Vec2f{values[offset], values[offset + 1u]};
}

__device__ __forceinline__ bool covered(
    Vec2f p0,
    Vec2f p1,
    Vec2f p2,
    float inverse,
    float sx,
    float sy,
    float* b0,
    float* b1,
    float* b2) {
  float l0 = a_mul(
      a_sub(a_mul(a_sub(p1.x, sx), a_sub(p2.y, sy)), a_mul(a_sub(p1.y, sy), a_sub(p2.x, sx))),
      inverse);
  float l1 = a_mul(
      a_sub(a_mul(a_sub(p2.x, sx), a_sub(p0.y, sy)), a_mul(a_sub(p2.y, sy), a_sub(p0.x, sx))),
      inverse);
  float l2 = a_sub(a_sub(1.0f, l0), l1);
  if (l0 < -1e-6f || l1 < -1e-6f || l2 < -1e-6f) {
    return false;
  }
  const float sum = a_add(a_add(l0, l1), l2);
  if (!(sum > 0.0f)) {
    return false;
  }
  *b0 = a_div(fmaxf(l0, 0.0f), sum);
  *b1 = a_div(fmaxf(l1, 0.0f), sum);
  *b2 = a_div(fmaxf(l2, 0.0f), sum);
  return true;
}

// Shared bbox rule: floor/ceil clamped to the image, empty range reported as
// min > max.
__device__ __forceinline__ bool bounds(
    Vec2f p0,
    Vec2f p1,
    Vec2f p2,
    int width,
    int height,
    int* min_x,
    int* max_x,
    int* min_y,
    int* max_y) {
  const float fx = fminf(p0.x, fminf(p1.x, p2.x));
  const float cx = fmaxf(p0.x, fmaxf(p1.x, p2.x));
  const float fy = fminf(p0.y, fminf(p1.y, p2.y));
  const float cy = fmaxf(p0.y, fmaxf(p1.y, p2.y));
  *min_x = (int)fmaxf(0.0f, floorf(fx));
  *max_x = (int)fminf((float)(width - 1), ceilf(cx));
  *min_y = (int)fmaxf(0.0f, floorf(fy));
  *max_y = (int)fminf((float)(height - 1), ceilf(cy));
  if (*max_x < 0) *max_x = 0;
  if (*max_y < 0) *max_y = 0;
  return *min_x <= *max_x && *min_y <= *max_y;
}

}  // namespace

extern "C" __global__ void mesh_attribute_select_kernel(
    const float* dest,
    const unsigned int* faces,
    unsigned int* last_triangle,
    unsigned int* write_count,
    MeshAttributeUniforms uniforms) {
  const int triangle = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (triangle >= (int)uniforms.triangle_count) return;

  const size_t base = (size_t)triangle * 3u;
  const unsigned int i0 = faces[base + 0u];
  const unsigned int i1 = faces[base + 1u];
  const unsigned int i2 = faces[base + 2u];
  const Vec2f p0 = read_vec2(dest, i0);
  const Vec2f p1 = read_vec2(dest, i1);
  const Vec2f p2 = read_vec2(dest, i2);
  const float area = a_sub(
      a_mul(a_sub(p1.x, p0.x), a_sub(p2.y, p0.y)),
      a_mul(a_sub(p1.y, p0.y), a_sub(p2.x, p0.x)));
  if (fabsf(area) < 1e-8f) return;
  const float inverse = a_div(1.0f, area);

  int min_x, max_x, min_y, max_y;
  if (!bounds(p0, p1, p2, (int)uniforms.width, (int)uniforms.height, &min_x, &max_x, &min_y, &max_y)) {
    return;
  }
  for (int py = min_y; py <= max_y; ++py) {
    const float sy = a_add((float)py, 0.5f);
    for (int px = min_x; px <= max_x; ++px) {
      const float sx = a_add((float)px, 0.5f);
      float b0, b1, b2;
      if (!covered(p0, p1, p2, inverse, sx, sy, &b0, &b1, &b2)) {
        continue;
      }
      const size_t pixel = (size_t)py * uniforms.width + (size_t)px;
      atomicMax(&last_triangle[pixel], (unsigned int)triangle + 1u);
      atomicAdd(&write_count[pixel], 1u);
    }
  }
}

extern "C" __global__ void mesh_attribute_resolve_kernel(
    const float* dest,
    const float* attr,
    const unsigned int* faces,
    const unsigned int* last_triangle,
    float* values,
    MeshAttributeUniforms uniforms) {
  const size_t pixel = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t pixel_count = (size_t)uniforms.width * (size_t)uniforms.height;
  if (pixel >= pixel_count) return;
  const unsigned int stored = last_triangle[pixel];
  if (stored == 0u) return;

  const int triangle = (int)(stored - 1u);
  const size_t base = (size_t)triangle * 3u;
  const unsigned int i0 = faces[base + 0u];
  const unsigned int i1 = faces[base + 1u];
  const unsigned int i2 = faces[base + 2u];
  const Vec2f p0 = read_vec2(dest, i0);
  const Vec2f p1 = read_vec2(dest, i1);
  const Vec2f p2 = read_vec2(dest, i2);
  const float area = a_sub(
      a_mul(a_sub(p1.x, p0.x), a_sub(p2.y, p0.y)),
      a_mul(a_sub(p1.y, p0.y), a_sub(p2.x, p0.x)));
  if (fabsf(area) < 1e-8f) return;
  const float inverse = a_div(1.0f, area);

  const int px = (int)(pixel % uniforms.width);
  const int py = (int)(pixel / uniforms.width);
  float b0, b1, b2;
  if (!covered(p0, p1, p2, inverse, a_add((float)px, 0.5f), a_add((float)py, 0.5f), &b0, &b1, &b2)) {
    return;
  }
  const Vec2f a0 = read_vec2(attr, i0);
  const Vec2f a1 = read_vec2(attr, i1);
  const Vec2f a2 = read_vec2(attr, i2);
  values[pixel * 2u + 0u] =
      a_add(a_add(a_mul(b0, a0.x), a_mul(b1, a1.x)), a_mul(b2, a2.x));
  values[pixel * 2u + 1u] =
      a_add(a_add(a_mul(b0, a0.y), a_mul(b1, a1.y)), a_mul(b2, a2.y));
}
