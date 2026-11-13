#include "moisture.h"

// Source-order translation of kernels::moisture::{face_apply, body_apply}.
// Compile with -fmad=false; every float op that can change a byte is an
// explicit binary32 intrinsic, and every expression keeps the CPU's grouping.

__device__ __forceinline__ float m_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float m_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float m_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float m_div(float a, float b) { return __fdiv_rn(a, b); }

__device__ __forceinline__ float m_clamp01(float value) {
  return fminf(fmaxf(value, 0.0f), 1.0f);
}

// Rust mix(a, b, t) = a * (1 - t) + b * t.
__device__ __forceinline__ float m_mix(float from, float to, float amount) {
  return m_add(m_mul(from, m_sub(1.0f, amount)), m_mul(to, amount));
}

__device__ __forceinline__ float m_forward_y(float r, float g, float b) {
  return m_add(m_add(m_mul(0.299f, r), m_mul(0.587f, g)), m_mul(0.114f, b));
}

__device__ __forceinline__ float m_forward_u(float r, float g, float b) {
  return m_add(m_add(m_mul(-0.169f, r), m_mul(-0.331f, g)), m_mul(0.500f, b));
}

__device__ __forceinline__ float m_forward_v(float r, float g, float b) {
  return m_add(m_add(m_mul(0.500f, r), m_mul(-0.419f, g)), m_mul(-0.081f, b));
}

// GL LINEAR + CLAMP_TO_EDGE over a f32 ramp: x = t*N - 0.5.
__device__ __forceinline__ float m_lut(const float* samples, unsigned int len, float t) {
  if (len == 1u) return samples[0];
  float x = fmaxf(m_sub(m_mul(m_clamp01(t), (float)len), 0.5f), 0.0f);
  float floor_x = floorf(x);
  float fraction = m_sub(x, floor_x);
  unsigned int first = (unsigned int)floor_x;
  if (first > len - 1u) first = len - 1u;
  unsigned int second = first + 1u < len ? first + 1u : len - 1u;
  return m_mix(samples[first], samples[second], fraction);
}

// Skin map: bilinear with x = blue, y = red.
__device__ __forceinline__ float m_skin_map(
    const float* samples,
    unsigned int width,
    unsigned int height,
    float blue,
    float red) {
  float x = fmaxf(m_sub(m_mul(m_clamp01(blue), (float)width), 0.5f), 0.0f);
  float y = fmaxf(m_sub(m_mul(m_clamp01(red), (float)height), 0.5f), 0.0f);
  float fx = floorf(x);
  float fy = floorf(y);
  float ax = m_sub(x, fx);
  float ay = m_sub(y, fy);
  unsigned int x0 = (unsigned int)fx;
  if (x0 > width - 1u) x0 = width - 1u;
  unsigned int y0 = (unsigned int)fy;
  if (y0 > height - 1u) y0 = height - 1u;
  unsigned int x1 = x0 + 1u < width ? x0 + 1u : width - 1u;
  unsigned int y1 = y0 + 1u < height ? y0 + 1u : height - 1u;
  float top = m_mix(samples[y0 * width + x0], samples[y0 * width + x1], ax);
  float bottom = m_mix(samples[y1 * width + x0], samples[y1 * width + x1], ax);
  return m_mix(top, bottom, ay);
}

__device__ __forceinline__ float m_sum2(float a, float b) {
  return m_clamp01(m_add(a, b));
}

__device__ __forceinline__ float m_sum3(float a, float b, float c) {
  return m_clamp01(m_add(m_add(a, b), c));
}

// MaskChannel order matches kernels::moisture::MaskChannel.
__device__ __forceinline__ float m_select_mask(
    float r, float g, float b, float a, unsigned int channel) {
  switch (channel) {
    case 0u: return r;
    case 1u: return g;
    case 2u: return b;
    case 3u: return a;
    case 4u: return m_sum2(r, g);
    case 5u: return m_sum2(r, b);
    case 6u: return m_sum2(r, a);
    case 7u: return m_sum2(g, b);
    case 8u: return m_sum2(g, a);
    case 9u: return m_sum2(b, a);
    case 10u: return m_sum3(r, g, b);
    case 11u: return m_clamp01(m_add(m_add(r, g), a));
    case 12u: return m_clamp01(m_add(m_add(r, b), a));
    case 13u: return m_clamp01(m_add(m_add(g, b), a));
    default: return m_clamp01(m_add(m_add(r, g), m_add(b, a)));
  }
}

__device__ __forceinline__ unsigned char m_byte(float value) {
  float scaled = fminf(fmaxf(m_mul(value, 255.0f), 0.0f), 255.0f);
  int q = (int)rintf(scaled);
  return (unsigned char)(q < 0 ? 0 : (q > 255 ? 255 : q));
}

extern "C" __global__ void moisture_face_kernel(
    const unsigned char* source_rgba,
    const unsigned char* mask_rgba,
    const unsigned char* blur_y,
    const float* skin_map,
    const float* hardlight,
    const float* beauty,
    unsigned char* out_rgba,
    MoistureFaceUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;

  unsigned int pixel = (unsigned int)y * uniforms.width + (unsigned int)x;
  unsigned int index = (unsigned int)y * uniforms.stride + (unsigned int)x * 4u;
  unsigned char r8 = source_rgba[index + 0u];
  unsigned char g8 = source_rgba[index + 1u];
  unsigned char b8 = source_rgba[index + 2u];
  unsigned char alpha8 = source_rgba[index + 3u];
  out_rgba[index + 0u] = r8;
  out_rgba[index + 1u] = g8;
  out_rgba[index + 2u] = b8;
  out_rgba[index + 3u] = alpha8;

  float r = m_div((float)r8, 255.0f);
  float g = m_div((float)g8, 255.0f);
  float b = m_div((float)b8, 255.0f);
  float blur = m_div((float)blur_y[pixel], 255.0f);

  float source_y = m_forward_y(r, g, b);
  float skin = m_skin_map(skin_map, uniforms.skin_width, uniforms.skin_height, b, r);
  float dog = 0.0f;
  if (skin > 0.0f) {
    float q1 = m_sub(1.0f, blur);
    float reference = m_clamp01(m_sub(m_add(source_y, m_mul(2.0f, q1)), 1.0f));
    reference = m_mul(m_add(reference, source_y), 0.5f);
    dog = m_sub(1.0f, m_lut(hardlight, uniforms.hardlight_len, reference));
  }

  float mask_r = m_div((float)mask_rgba[pixel * 4u + 0u], 255.0f);
  float mask_g = m_div((float)mask_rgba[pixel * 4u + 1u], 255.0f);
  float mask_b = m_div((float)mask_rgba[pixel * 4u + 2u], 255.0f);
  float mask_a = m_div((float)mask_rgba[pixel * 4u + 3u], 255.0f);
  float mask = m_select_mask(mask_r, mask_g, mask_b, mask_a, uniforms.channel);
  float weight = m_mul(m_mul(skin, uniforms.degree), mask);
  if (!(weight > 0.0f)) return;

  float yv = m_forward_y(r, g, b);
  float u = m_forward_u(r, g, b);
  float v = m_forward_v(r, g, b);
  float curve = m_lut(beauty, uniforms.beauty_len, yv);
  float result_y = m_mix(yv, m_mix(yv, curve, dog), weight);
  float out_r = m_add(m_add(result_y, m_mul(-0.00093f, u)), m_mul(1.401687f, v));
  float out_g = m_add(m_add(result_y, m_mul(-0.3437f, u)), m_mul(-0.71417f, v));
  float out_b = m_add(m_add(result_y, m_mul(1.77216f, u)), m_mul(0.00099f, v));
  out_rgba[index + 0u] = m_byte(out_r);
  out_rgba[index + 1u] = m_byte(out_g);
  out_rgba[index + 2u] = m_byte(out_b);
}

extern "C" __global__ void moisture_body_kernel(
    const unsigned char* source_rgba,
    const float* weight_skin,
    const float* weight_dog,
    const float* beauty,
    unsigned char* out_rgba,
    MoistureBodyUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.roi_w || y >= (int)uniforms.roi_h) return;

  unsigned int px = uniforms.roi_x + (unsigned int)x;
  unsigned int py = uniforms.roi_y + (unsigned int)y;
  unsigned int pixel = py * uniforms.width + px;
  unsigned int index = py * uniforms.stride + px * 4u;
  unsigned char r8 = source_rgba[index + 0u];
  unsigned char g8 = source_rgba[index + 1u];
  unsigned char b8 = source_rgba[index + 2u];
  unsigned char alpha8 = source_rgba[index + 3u];
  out_rgba[index + 0u] = r8;
  out_rgba[index + 1u] = g8;
  out_rgba[index + 2u] = b8;
  out_rgba[index + 3u] = alpha8;

  float alpha = m_mul(weight_skin[pixel], uniforms.degree);
  if (!(alpha > 0.0f)) return;

  float r = m_div((float)r8, 255.0f);
  float g = m_div((float)g8, 255.0f);
  float b = m_div((float)b8, 255.0f);
  float yv = m_forward_y(r, g, b);
  float u = m_forward_u(r, g, b);
  float v = m_forward_v(r, g, b);
  float curve = m_lut(beauty, uniforms.beauty_len, yv);
  float inner = m_mix(yv, curve, weight_dog[pixel]);
  float result = m_mix(yv, inner, alpha);
  float out_r = m_add(m_add(result, m_mul(-0.00093f, u)), m_mul(1.401687f, v));
  float out_g = m_add(m_add(result, m_mul(-0.3437f, u)), m_mul(-0.71417f, v));
  float out_b = m_add(m_add(result, m_mul(1.77216f, u)), m_mul(0.00099f, v));
  out_rgba[index + 0u] = m_byte(out_r);
  out_rgba[index + 1u] = m_byte(out_g);
  out_rgba[index + 2u] = m_byte(out_b);
}
