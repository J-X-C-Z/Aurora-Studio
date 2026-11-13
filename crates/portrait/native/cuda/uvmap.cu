#include "uvmap.h"

// Source-order translation of kernels::uvmap_color::{apply, sample_bilinear,
// sample_catmull_rom}. Compile with -fmad=false; every float op that can change
// a byte goes through an explicit binary32 intrinsic.

__device__ __forceinline__ float u_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float u_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float u_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float u_div(float a, float b) { return __fdiv_rn(a, b); }

__device__ __forceinline__ float u_lerp(float a, float b, float t) {
  return u_add(a, u_mul(u_sub(b, a), t));
}

__device__ __forceinline__ float u_texel(
    const unsigned char* source,
    int width,
    int x,
    int y,
    int channel) {
  return u_div((float)source[((size_t)y * (size_t)width + (size_t)x) * 4u + (size_t)channel], 255.0f);
}

// texture2D with clamp-to-edge and the texel-centre rule p = uv * size - 0.5.
__device__ __forceinline__ void u_bilinear(
    const unsigned char* source,
    int width,
    int height,
    float u,
    float v,
    float* out) {
  float w = (float)width;
  float h = (float)height;
  float x = fminf(fmaxf(u_sub(u_mul(u, w), 0.5f), 0.0f), u_sub(w, 1.0f));
  float y = fminf(fmaxf(u_sub(u_mul(v, h), 0.5f), 0.0f), u_sub(h, 1.0f));
  float x0 = floorf(x);
  float y0 = floorf(y);
  float x1 = fminf(u_add(x0, 1.0f), u_sub(w, 1.0f));
  float y1 = fminf(u_add(y0, 1.0f), u_sub(h, 1.0f));
  float fx = u_sub(x, x0);
  float fy = u_sub(y, y0);
  for (int channel = 0; channel < 4; ++channel) {
    float top = u_lerp(
        u_texel(source, width, (int)x0, (int)y0, channel),
        u_texel(source, width, (int)x1, (int)y0, channel),
        fx);
    float bottom = u_lerp(
        u_texel(source, width, (int)x0, (int)y1, channel),
        u_texel(source, width, (int)x1, (int)y1, channel),
        fx);
    out[channel] = u_lerp(top, bottom, fy);
  }
}

// texture2D(mo, uv) for the float4 displacement field. Only the sampling
// coordinate is clamped; the signed offsets themselves stay untouched.
__device__ __forceinline__ void u_field_bilinear(
    const float* field,
    int width,
    int height,
    float u,
    float v,
    float* out) {
  float w = (float)width;
  float h = (float)height;
  float x = fminf(fmaxf(u_sub(u_mul(u, w), 0.5f), 0.0f), u_sub(w, 1.0f));
  float y = fminf(fmaxf(u_sub(u_mul(v, h), 0.5f), 0.0f), u_sub(h, 1.0f));
  float x0 = floorf(x);
  float y0 = floorf(y);
  float x1 = fminf(u_add(x0, 1.0f), u_sub(w, 1.0f));
  float y1 = fminf(u_add(y0, 1.0f), u_sub(h, 1.0f));
  float fx = u_sub(x, x0);
  float fy = u_sub(y, y0);
  int ix0 = (int)x0;
  int iy0 = (int)y0;
  int ix1 = (int)x1;
  int iy1 = (int)y1;
  for (int channel = 0; channel < 4; ++channel) {
    float p00 = field[((size_t)iy0 * (size_t)width + (size_t)ix0) * 4u + (size_t)channel];
    float p10 = field[((size_t)iy0 * (size_t)width + (size_t)ix1) * 4u + (size_t)channel];
    float p01 = field[((size_t)iy1 * (size_t)width + (size_t)ix0) * 4u + (size_t)channel];
    float p11 = field[((size_t)iy1 * (size_t)width + (size_t)ix1) * 4u + (size_t)channel];
    float top = u_lerp(p00, p10, fx);
    float bottom = u_lerp(p01, p11, fx);
    out[channel] = u_lerp(top, bottom, fy);
  }
}

__device__ __forceinline__ void u_catmull_weights(float f, float* w) {
  w[0] = u_mul(f, u_add(-0.5f, u_mul(f, u_sub(1.0f, u_mul(0.5f, f)))));
  w[1] = u_add(1.0f, u_mul(u_mul(f, f), u_add(-2.5f, u_mul(1.5f, f))));
  w[2] = u_mul(f, u_add(0.5f, u_mul(f, u_sub(2.0f, u_mul(1.5f, f)))));
  w[3] = u_mul(u_mul(f, f), u_add(-0.5f, u_mul(0.5f, f)));
}

// Shipped 9-tap Catmull-Rom; `tex_w`/`tex_h` are the shader's u_Width/u_Height.
__device__ __forceinline__ void u_catmull(
    const unsigned char* source,
    int width,
    int height,
    float u,
    float v,
    float tex_w,
    float tex_h,
    float* out) {
  float sx = u_mul(u, tex_w);
  float sy = u_mul(v, tex_h);
  float px = u_add(floorf(u_sub(sx, 0.5f)), 0.5f);
  float py = u_add(floorf(u_sub(sy, 0.5f)), 0.5f);
  float fx = u_sub(sx, px);
  float fy = u_sub(sy, py);
  float wx[4];
  float wy[4];
  u_catmull_weights(fx, wx);
  u_catmull_weights(fy, wy);
  float w12x = u_add(wx[1], wx[2]);
  float w12y = u_add(wy[1], wy[2]);
  float t12x = u_add(px, u_div(wx[2], w12x));
  float t12y = u_add(py, u_div(wy[2], w12y));

  float taps_x[9] = {u_sub(px, 1.0f), t12x, u_add(px, 2.0f),
                     u_sub(px, 1.0f), t12x, u_add(px, 2.0f),
                     u_sub(px, 1.0f), t12x, u_add(px, 2.0f)};
  float taps_y[9] = {u_sub(py, 1.0f), u_sub(py, 1.0f), u_sub(py, 1.0f),
                     t12y, t12y, t12y,
                     u_add(py, 2.0f), u_add(py, 2.0f), u_add(py, 2.0f)};
  float weights[9] = {u_mul(wx[0], wy[0]), u_mul(w12x, wy[0]), u_mul(wx[3], wy[0]),
                      u_mul(wx[0], w12y), u_mul(w12x, w12y), u_mul(wx[3], w12y),
                      u_mul(wx[0], wy[3]), u_mul(w12x, wy[3]), u_mul(wx[3], wy[3])};

  for (int channel = 0; channel < 4; ++channel) {
    out[channel] = 0.0f;
  }
  for (int tap = 0; tap < 9; ++tap) {
    float sample[4];
    u_bilinear(
        source,
        width,
        height,
        u_div(taps_x[tap], tex_w),
        u_div(taps_y[tap], tex_h),
        sample);
    for (int channel = 0; channel < 4; ++channel) {
      out[channel] = u_add(out[channel], u_mul(sample[channel], weights[tap]));
    }
  }
}

__device__ __forceinline__ unsigned char u_quantize(float value) {
  float scaled = u_mul(value, 255.0f);
  scaled = fminf(fmaxf(scaled, 0.0f), 255.0f);
  int q = (int)rintf(scaled);
  return (unsigned char)(q < 0 ? 0 : (q > 255 ? 255 : q));
}

extern "C" __global__ void uvmap_color_kernel(
    const unsigned char* source_rgba,
    const float* field_xy,
    unsigned char* out_rgba,
    UvmapUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;

  unsigned int source_index =
      (unsigned int)y * uniforms.stride + (unsigned int)x * 4u;

  float u = u_div(u_add((float)x, 0.5f), (float)uniforms.width);
  float v = u_div(u_add((float)y, 0.5f), (float)uniforms.height);
  // Four floats per pixel: .xy is the RG pair, .zw the BA pair. Match
  // texture2D(mo, ae): sample the field bilinearly with clamp-to-edge, then add
  // the selected offset. The offsets stay signed and are never quantised.
  float field[4];
  u_field_bilinear(
      field_xy, (int)uniforms.width, (int)uniforms.height, u, v, field);
  float offset_x = uniforms.channel == 1u ? field[2] : field[0];
  float offset_y = uniforms.channel == 1u ? field[3] : field[1];
  float target_u = u_add(u, offset_x);
  float target_v = u_add(v, offset_y);

  float color[4];
  if (uniforms.catmull != 0u) {
    u_catmull(
        source_rgba,
        (int)uniforms.width,
        (int)uniforms.height,
        target_u,
        target_v,
        uniforms.catmull_w,
        uniforms.catmull_h,
        color);
  } else {
    u_bilinear(source_rgba, (int)uniforms.width, (int)uniforms.height, target_u, target_v, color);
  }

  out_rgba[source_index + 0u] = u_quantize(color[0]);
  out_rgba[source_index + 1u] = u_quantize(color[1]);
  out_rgba[source_index + 2u] = u_quantize(color[2]);
  out_rgba[source_index + 3u] = u_quantize(color[3]);
}
