/*
 * PF8025/PF8026 V2Warp forward-mesh candidate.
 *
 * The select kernel mirrors the project forward-raster coverage contract. The
 * eval kernel consumes the selected triangle and applies the recovered V2Warp
 * fragment order: offset decode, face scales, row-vector mat2, edge, then the
 * nine-tap Catmull-Rom source sample.
 *
 * This is not a claim of pixel equality with the vendor GL/FBO path.
 */

#include "v2warp.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstddef>

namespace {

constexpr float kDegenerateArea2 = 1e-12f;
constexpr uint32_t kUncovered = 0xFFFFFFFFu;

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
  return winding == V2WARP_WINDING_POSITIVE_IS_FRONT ? positive : !positive;
}

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
  const bool inside = edge == V2WARP_EDGE_INCLUSIVE
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
  const float limit_x = static_cast<float>(width - 1);
  const float limit_y = static_cast<float>(height - 1);
  *x0 = static_cast<int32_t>(ceilf(fminf(fmaxf(AddRn(min_x, -0.5f), 0.0f), limit_x)));
  *x1 = static_cast<int32_t>(floorf(fminf(fmaxf(AddRn(max_x, -0.5f), 0.0f), limit_x)));
  *y0 = static_cast<int32_t>(ceilf(fminf(fmaxf(AddRn(min_y, -0.5f), 0.0f), limit_y)));
  *y1 = static_cast<int32_t>(floorf(fminf(fmaxf(AddRn(max_y, -0.5f), 0.0f), limit_y)));
}

__device__ __forceinline__ Vec2f Bary2(
    Vec2f a, Vec2f b, Vec2f c, float wa, float wb, float wc) {
  return Vec2f{
      AddRn(AddRn(MulRn(wa, a.x), MulRn(wb, b.x)), MulRn(wc, c.x)),
      AddRn(AddRn(MulRn(wa, a.y), MulRn(wb, b.y)), MulRn(wc, c.y))};
}

__device__ __forceinline__ float Lerp(float a, float b, float t) {
  return AddRn(a, MulRn(AddRn(b, -a), t));
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

/* texture2D with GL_LINEAR and clamp-to-edge, texel-center rule p=uv*size-.5. */
__device__ __forceinline__ Vec2f SampleOffset(
    const float* offset,
    int32_t width,
    int32_t height,
    Vec2f uv) {
  const float w = static_cast<float>(width);
  const float h = static_cast<float>(height);
  const float x = fminf(fmaxf(AddRn(MulRn(uv.x, w), -0.5f), 0.0f), w - 1.0f);
  const float y = fminf(fmaxf(AddRn(MulRn(uv.y, h), -0.5f), 0.0f), h - 1.0f);
  const float x0f = floorf(x);
  const float y0f = floorf(y);
  const float x1f = fminf(AddRn(x0f, 1.0f), w - 1.0f);
  const float y1f = fminf(AddRn(y0f, 1.0f), h - 1.0f);
  const float fx = AddRn(x, -x0f);
  const float fy = AddRn(y, -y0f);
  const int32_t x0 = static_cast<int32_t>(x0f);
  const int32_t y0 = static_cast<int32_t>(y0f);
  const int32_t x1 = static_cast<int32_t>(x1f);
  const int32_t y1 = static_cast<int32_t>(y1f);
  float rgba[4];
  for (int32_t c = 0; c < 4; ++c) {
    const size_t p00 =
        (static_cast<size_t>(y0) * static_cast<size_t>(width) +
         static_cast<size_t>(x0)) * 4u +
        static_cast<size_t>(c);
    const size_t p10 =
        (static_cast<size_t>(y0) * static_cast<size_t>(width) +
         static_cast<size_t>(x1)) * 4u +
        static_cast<size_t>(c);
    const size_t p01 =
        (static_cast<size_t>(y1) * static_cast<size_t>(width) +
         static_cast<size_t>(x0)) * 4u +
        static_cast<size_t>(c);
    const size_t p11 =
        (static_cast<size_t>(y1) * static_cast<size_t>(width) +
         static_cast<size_t>(x1)) * 4u +
        static_cast<size_t>(c);
    const float top = Lerp(offset[p00], offset[p10], fx);
    const float bottom = Lerp(offset[p01], offset[p11], fx);
    rgba[c] = Lerp(top, bottom, fy);
  }
  return Vec2f{
      MulRn(AddRn(rgba[0], -rgba[1]), 0.25f),
      MulRn(AddRn(rgba[2], -rgba[3]), 0.25f)};
}

__device__ __forceinline__ float Edge(Vec2f uv) {
  const float ex =
      AddRn(2.49f, -fabsf(MulRn(4.98f, AddRn(uv.x, -0.5f))));
  const float ey =
      AddRn(2.49f, -fabsf(MulRn(4.98f, AddRn(uv.y, -0.5f))));
  return fminf(fmaxf(fminf(ex, ey), 0.0f), 1.0f);
}

__device__ __forceinline__ float Weight0(float f) {
  return MulRn(f, AddRn(-0.5f, MulRn(f, AddRn(1.0f, MulRn(-0.5f, f)))));
}

__device__ __forceinline__ float Weight1(float f) {
  return AddRn(1.0f, MulRn(MulRn(f, f), AddRn(-2.5f, MulRn(1.5f, f))));
}

__device__ __forceinline__ float Weight2(float f) {
  return MulRn(f, AddRn(0.5f, MulRn(f, AddRn(2.0f, MulRn(-1.5f, f)))));
}

__device__ __forceinline__ float Weight3(float f) {
  return MulRn(MulRn(f, f), AddRn(-0.5f, MulRn(0.5f, f)));
}

__device__ __forceinline__ void BilinearSource(
    const uint8_t* source,
    int32_t width,
    int32_t height,
    Vec2f uv,
    float* out) {
  const float w = static_cast<float>(width);
  const float h = static_cast<float>(height);
  const float x = fminf(fmaxf(AddRn(MulRn(uv.x, w), -0.5f), 0.0f), w - 1.0f);
  const float y = fminf(fmaxf(AddRn(MulRn(uv.y, h), -0.5f), 0.0f), h - 1.0f);
  const float x0f = floorf(x);
  const float y0f = floorf(y);
  const float x1f = fminf(AddRn(x0f, 1.0f), w - 1.0f);
  const float y1f = fminf(AddRn(y0f, 1.0f), h - 1.0f);
  const float fx = AddRn(x, -x0f);
  const float fy = AddRn(y, -y0f);
  const int32_t x0 = static_cast<int32_t>(x0f);
  const int32_t y0 = static_cast<int32_t>(y0f);
  const int32_t x1 = static_cast<int32_t>(x1f);
  const int32_t y1 = static_cast<int32_t>(y1f);
  for (int32_t c = 0; c < 4; ++c) {
    const float top = Lerp(
        Texel(source, width, x0, y0, c),
        Texel(source, width, x1, y0, c),
        fx);
    const float bottom = Lerp(
        Texel(source, width, x0, y1, c),
        Texel(source, width, x1, y1, c),
        fx);
    out[c] = Lerp(top, bottom, fy);
  }
}

__device__ __forceinline__ void CatmullRom(
    const uint8_t* source,
    int32_t width,
    int32_t height,
    Vec2f uv,
    float tex_w,
    float tex_h,
    float* out) {
  const float sx = MulRn(uv.x, tex_w);
  const float sy = MulRn(uv.y, tex_h);
  const float px = AddRn(floorf(AddRn(sx, -0.5f)), 0.5f);
  const float py = AddRn(floorf(AddRn(sy, -0.5f)), 0.5f);
  const float fx = AddRn(sx, -px);
  const float fy = AddRn(sy, -py);

  const float w0x = Weight0(fx);
  const float w1x = Weight1(fx);
  const float w2x = Weight2(fx);
  const float w3x = Weight3(fx);
  const float w0y = Weight0(fy);
  const float w1y = Weight1(fy);
  const float w2y = Weight2(fy);
  const float w3y = Weight3(fy);
  const float w12x = AddRn(w1x, w2x);
  const float w12y = AddRn(w1y, w2y);
  const float t12x = AddRn(px, w2x / w12x);
  const float t12y = AddRn(py, w2y / w12y);

  const float tap_x[9] = {AddRn(px, -1.0f), t12x, AddRn(px, 2.0f),
                          AddRn(px, -1.0f), t12x, AddRn(px, 2.0f),
                          AddRn(px, -1.0f), t12x, AddRn(px, 2.0f)};
  const float tap_y[9] = {AddRn(py, -1.0f), AddRn(py, -1.0f), AddRn(py, -1.0f),
                          t12y, t12y, t12y,
                          AddRn(py, 2.0f), AddRn(py, 2.0f), AddRn(py, 2.0f)};
  const float weight[9] = {
      MulRn(w0x, w0y), MulRn(w12x, w0y), MulRn(w3x, w0y),
      MulRn(w0x, w12y), MulRn(w12x, w12y), MulRn(w3x, w12y),
      MulRn(w0x, w3y), MulRn(w12x, w3y), MulRn(w3x, w3y)};

  for (int32_t c = 0; c < 4; ++c) {
    out[c] = 0.0f;
  }
  for (int32_t tap = 0; tap < 9; ++tap) {
    const Vec2f uv_tap{tap_x[tap] / tex_w, tap_y[tap] / tex_h};
    float sample[4];
    BilinearSource(source, width, height, uv_tap, sample);
    for (int32_t c = 0; c < 4; ++c) {
      out[c] = AddRn(out[c], MulRn(sample[c], weight[tap]));
    }
  }
}

__device__ __forceinline__ uint8_t Quantize(float value) {
  const int32_t q = __float2int_rn(MulRn(value, 255.0f));
  return static_cast<uint8_t>(q < 0 ? 0 : (q > 255 ? 255 : q));
}

__device__ __forceinline__ void v2warp_select_body(
    const float* positions,
    const uint16_t* ebo,
    int32_t width,
    int32_t height,
    int32_t triangle_count,
    V2WarpRules rules,
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
  const Vec2f p0 = ReadVec2(positions, i0);
  const Vec2f p1 = ReadVec2(positions, i1);
  const Vec2f p2 = ReadVec2(positions, i2);
  const float area2 = Cross(Sub(p1, p0), Sub(p2, p0));
  if (fabsf(area2) <= kDegenerateArea2) {
    return;
  }
  const bool front = IsFront(area2, rules.winding);
  if (rules.cull == V2WARP_CULL_FRONT && front) {
    return;
  }
  if (rules.cull == V2WARP_CULL_BACK && !front) {
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
      if (rules.overlap == V2WARP_OVERLAP_LAST) {
        atomicMax(&selected[pixel], static_cast<uint32_t>(triangle));
      } else {
        atomicMin(&selected[pixel], static_cast<uint32_t>(triangle));
      }
    }
  }
}

__device__ __forceinline__ void v2warp_eval_body(
    const uint8_t* source,
    const float* offset,
    uint8_t* out,
    const float* positions,
    const float* texcoord,
    const float* offset_texcoord,
    const float* face_scales,
    const uint16_t* ebo,
    const uint32_t* selected,
    const uint32_t* coverage,
    V2WarpUniforms uniforms) {
  const size_t pixel =
      static_cast<size_t>(blockIdx.x) * blockDim.x + static_cast<size_t>(threadIdx.x);
  const size_t pixel_count =
      static_cast<size_t>(uniforms.width) * static_cast<size_t>(uniforms.height);
  if (pixel >= pixel_count || coverage[pixel] == 0u) {
    return;
  }
  const uint32_t triangle = selected[pixel];
  if (triangle == kUncovered) {
    return;
  }
  if (coverage[pixel] > 1u && uniforms.overlap == V2WARP_OVERLAP_REJECT) {
    return;
  }

  const int32_t px = static_cast<int32_t>(pixel % static_cast<size_t>(uniforms.width));
  const int32_t py = static_cast<int32_t>(pixel / static_cast<size_t>(uniforms.width));
  const int32_t base = static_cast<int32_t>(triangle) * 3;
  const uint16_t i0 = ebo[base];
  const uint16_t i1 = ebo[base + 1];
  const uint16_t i2 = ebo[base + 2];
  const Vec2f p0 = ReadVec2(positions, i0);
  const Vec2f p1 = ReadVec2(positions, i1);
  const Vec2f p2 = ReadVec2(positions, i2);
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
          uniforms.edge,
          &b0,
          &b1,
          &b2)) {
    return;
  }
  const Vec2f uv = Bary2(
      ReadVec2(texcoord, i0), ReadVec2(texcoord, i1), ReadVec2(texcoord, i2),
      b0, b1, b2);
  const Vec2f offset_uv = Bary2(
      ReadVec2(offset_texcoord, i0),
      ReadVec2(offset_texcoord, i1),
      ReadVec2(offset_texcoord, i2),
      b0,
      b1,
      b2);
  const Vec2f scales = Bary2(
      ReadVec2(face_scales, i0),
      ReadVec2(face_scales, i1),
      ReadVec2(face_scales, i2),
      b0,
      b1,
      b2);

  Vec2f d = SampleOffset(
      offset, uniforms.offset_width, uniforms.offset_height, offset_uv);
  d.x = MulRn(d.x, scales.x);
  d.y = MulRn(d.y, scales.y);
  const float input_x = d.x;
  const float input_y = d.y;
  d.x = AddRn(
      MulRn(input_x, uniforms.mat2_raw0),
      MulRn(input_y, uniforms.mat2_raw2));
  d.y = AddRn(
      MulRn(input_x, uniforms.mat2_raw1),
      MulRn(input_y, uniforms.mat2_raw3));
  const float edge = Edge(uv);
  d.x = MulRn(d.x, edge);
  d.y = MulRn(d.y, edge);

  float color[4];
  CatmullRom(
      source,
      uniforms.width,
      uniforms.height,
      Vec2f{AddRn(uv.x, d.x), AddRn(uv.y, d.y)},
      uniforms.filter_width,
      uniforms.filter_height,
      color);
  uint8_t* dst = out + pixel * 4u;
  for (int32_t c = 0; c < 4; ++c) {
    dst[c] = Quantize(color[c]);
  }
}

}  // namespace

extern "C" __global__ void v2warp_select_kernel(
    const float* positions,
    const uint16_t* ebo,
    int32_t width,
    int32_t height,
    int32_t triangle_count,
    V2WarpRules rules,
    uint32_t* selected,
    uint32_t* coverage) {
  v2warp_select_body(
      positions, ebo, width, height, triangle_count, rules, selected, coverage);
}

extern "C" __global__ void v2warp_eval_kernel(
    const uint8_t* source,
    const float* offset,
    uint8_t* out,
    const float* positions,
    const float* texcoord,
    const float* offset_texcoord,
    const float* face_scales,
    const uint16_t* ebo,
    const uint32_t* selected,
    const uint32_t* coverage,
    V2WarpUniforms uniforms) {
  v2warp_eval_body(
      source,
      offset,
      out,
      positions,
      texcoord,
      offset_texcoord,
      face_scales,
      ebo,
      selected,
      coverage,
      uniforms);
}
