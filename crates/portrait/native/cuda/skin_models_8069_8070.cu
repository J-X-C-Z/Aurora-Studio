#include "skin_models_8069_8070.h"

// Source-order translations of kernels::skin_models::{surface_pass, gauss_pass,
// texture_surface_blur, clarity_apply}. Compile with -fmad=false. Each filter
// pass stores RGBA8, preserving every intermediate quantisation point.

__device__ __forceinline__ float sm_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float sm_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float sm_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float sm_div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float sm_neg(float a) { return -a; }

__device__ __forceinline__ float sm_clamp(float value, float low, float high) {
  return fminf(fmaxf(value, low), high);
}

__device__ __forceinline__ float sm_lerp(float from, float to, float amount) {
  return sm_add(from, sm_mul(sm_sub(to, from), amount));
}

__device__ __forceinline__ float sm_unit(unsigned char value) {
  return sm_div((float)value, 255.0f);
}

__device__ __forceinline__ unsigned char sm_to_byte(float value) {
  float encoded = sm_mul(value, 255.0f);
  if (encoded <= 0.0f) return 0;
  if (encoded >= 255.0f) return 255;
  return (unsigned char)rintf(encoded);
}

__device__ __forceinline__ unsigned char sm_quantize(float value) {
  float rounded = rintf(sm_mul(value, 255.0f));
  return (unsigned char)(int)sm_clamp(rounded, 0.0f, 255.0f);
}

__device__ __forceinline__ int sm_clamp_index(int value, unsigned int size) {
  if (value < 0) return 0;
  if (value >= (int)size) return (int)size - 1;
  return value;
}

__device__ __forceinline__ float sm_bilinear_channel(
    const unsigned char* rgba,
    unsigned int a,
    unsigned int b,
    unsigned int c,
    unsigned int d,
    float fx,
    float fy,
    unsigned int channel) {
  float top = sm_lerp(sm_unit(rgba[a + channel]), sm_unit(rgba[b + channel]), fx);
  float bottom = sm_lerp(sm_unit(rgba[c + channel]), sm_unit(rgba[d + channel]), fx);
  return sm_lerp(top, bottom, fy);
}

__device__ __forceinline__ float4 sm_bilinear(
    const unsigned char* rgba,
    unsigned int width,
    unsigned int height,
    unsigned int stride,
    float u,
    float v) {
  float px = sm_sub(sm_mul(u, (float)width), 0.5f);
  float py = sm_sub(sm_mul(v, (float)height), 0.5f);
  float x0f = floorf(px);
  float y0f = floorf(py);
  float fx = sm_sub(px, x0f);
  float fy = sm_sub(py, y0f);
  int x0 = sm_clamp_index((int)x0f, width);
  int x1 = sm_clamp_index((int)x0f + 1, width);
  int y0 = sm_clamp_index((int)y0f, height);
  int y1 = sm_clamp_index((int)y0f + 1, height);

  unsigned int a = (unsigned int)y0 * stride + (unsigned int)x0 * 4u;
  unsigned int b = (unsigned int)y0 * stride + (unsigned int)x1 * 4u;
  unsigned int c = (unsigned int)y1 * stride + (unsigned int)x0 * 4u;
  unsigned int d = (unsigned int)y1 * stride + (unsigned int)x1 * 4u;
  return make_float4(
      sm_bilinear_channel(rgba, a, b, c, d, fx, fy, 0u),
      sm_bilinear_channel(rgba, a, b, c, d, fx, fy, 1u),
      sm_bilinear_channel(rgba, a, b, c, d, fx, fy, 2u),
      sm_bilinear_channel(rgba, a, b, c, d, fx, fy, 3u));
}

__device__ __forceinline__ float sm_smooth_step(float edge0, float edge1, float value) {
  float t = sm_clamp(sm_div(sm_sub(value, edge0), sm_sub(edge1, edge0)), 0.0f, 1.0f);
  return sm_mul(sm_mul(t, t), sm_sub(3.0f, sm_mul(2.0f, t)));
}

__device__ __forceinline__ float4 sm_sample_offset(
    const unsigned char* source,
    const SkinModelsFilterUniforms& uniforms,
    float u,
    float v,
    float sign,
    int distance) {
  float offset_u = sm_mul(sm_mul(sign, (float)distance), uniforms.step_u);
  float offset_v = sm_mul(sm_mul(sign, (float)distance), uniforms.step_v);
  return sm_bilinear(
      source,
      uniforms.width,
      uniforms.height,
      uniforms.stride,
      sm_add(u, offset_u),
      sm_add(v, offset_v));
}

extern "C" __global__ void pf8069_surface_kernel(
    const unsigned char* source_rgba,
    unsigned char* out_rgba,
    SkinModelsFilterUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;

  float u = sm_div(sm_add((float)x, 0.5f), (float)uniforms.width);
  float v = sm_div(sm_add((float)y, 0.5f), (float)uniforms.height);
  float4 center = sm_bilinear(
      source_rgba, uniforms.width, uniforms.height, uniforms.stride, u, v);
  float sum_r = center.x;
  float sum_g = center.y;
  float sum_b = center.z;
  float total_weight = 1.0f;

  for (int distance = 1; distance <= 4; ++distance) {
    for (int tap = 0; tap < 2; ++tap) {
      float sign = tap == 0 ? -1.0f : 1.0f;
      float4 sample = sm_sample_offset(
          source_rgba, uniforms, u, v, sign, distance);
      float dr = sm_sub(sample.x, center.x);
      float dg = sm_sub(sample.y, center.y);
      float db = sm_sub(sample.z, center.z);
      float squared = sm_add(
          sm_add(sm_mul(dr, dr), sm_mul(dg, dg)), sm_mul(db, db));
      float difference = sqrtf(squared);
      float weight = sm_sub(
          1.0f, sm_smooth_step(0.0f, uniforms.hk, difference));
      sum_r = sm_add(sum_r, sm_mul(sample.x, weight));
      sum_g = sm_add(sum_g, sm_mul(sample.y, weight));
      sum_b = sm_add(sum_b, sm_mul(sample.z, weight));
      total_weight = sm_add(total_weight, weight);
    }
  }

  unsigned int index =
      (unsigned int)y * uniforms.stride + (unsigned int)x * 4u;
  out_rgba[index + 0u] = sm_to_byte(sm_div(sum_r, total_weight));
  out_rgba[index + 1u] = sm_to_byte(sm_div(sum_g, total_weight));
  out_rgba[index + 2u] = sm_to_byte(sm_div(sum_b, total_weight));
  out_rgba[index + 3u] = sm_to_byte(center.w);
}

extern "C" __global__ void pf8069_gauss_kernel(
    const unsigned char* source_rgba,
    unsigned char* out_rgba,
    SkinModelsFilterUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;

  const float weights[4] = {0.15f, 0.12f, 0.09f, 0.05f};
  float u = sm_div(sm_add((float)x, 0.5f), (float)uniforms.width);
  float v = sm_div(sm_add((float)y, 0.5f), (float)uniforms.height);
  float4 center = sm_bilinear(
      source_rgba, uniforms.width, uniforms.height, uniforms.stride, u, v);
  float sum_r = sm_mul(0.18f, center.x);
  float sum_g = sm_mul(0.18f, center.y);
  float sum_b = sm_mul(0.18f, center.z);
  float sum_a = sm_mul(0.18f, center.w);

  for (int distance = 1; distance <= 4; ++distance) {
    float weight = weights[distance - 1];
    for (int tap = 0; tap < 2; ++tap) {
      float sign = tap == 0 ? -1.0f : 1.0f;
      float4 sample = sm_sample_offset(
          source_rgba, uniforms, u, v, sign, distance);
      sum_r = sm_add(sum_r, sm_mul(sample.x, weight));
      sum_g = sm_add(sum_g, sm_mul(sample.y, weight));
      sum_b = sm_add(sum_b, sm_mul(sample.z, weight));
      sum_a = sm_add(sum_a, sm_mul(sample.w, weight));
    }
  }

  unsigned int index =
      (unsigned int)y * uniforms.stride + (unsigned int)x * 4u;
  out_rgba[index + 0u] = sm_to_byte(sum_r);
  out_rgba[index + 1u] = sm_to_byte(sum_g);
  out_rgba[index + 2u] = sm_to_byte(sum_b);
  out_rgba[index + 3u] = sm_to_byte(sum_a);
}

extern "C" __global__ void pf8069_blend_kernel(
    const unsigned char* source_rgba,
    const unsigned char* layer_rgba,
    const unsigned char* mask_r,
    const unsigned char* extra_mask_r,
    unsigned char* out_rgba,
    SkinModelsBlendUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;

  unsigned int index =
      (unsigned int)y * uniforms.stride + (unsigned int)x * 4u;
  unsigned int mask_index =
      (unsigned int)y * uniforms.mask_row_bytes + (unsigned int)x;
  out_rgba[index + 0u] = source_rgba[index + 0u];
  out_rgba[index + 1u] = source_rgba[index + 1u];
  out_rgba[index + 2u] = source_rgba[index + 2u];
  out_rgba[index + 3u] = source_rgba[index + 3u];

  float mix = sm_mul(sm_unit(layer_rgba[index + 3u]), uniforms.degree);
  mix = sm_mul(mix, sm_unit(mask_r[mask_index]));
  mix = sm_mul(mix, sm_unit(extra_mask_r[mask_index]));
  if (!(mix > 0.01f)) return;

  for (int channel = 0; channel < 3; ++channel) {
    float src = sm_unit(source_rgba[index + (unsigned int)channel]);
    float dst = sm_unit(layer_rgba[index + (unsigned int)channel]);
    out_rgba[index + (unsigned int)channel] =
        sm_to_byte(sm_lerp(src, dst, mix));
  }
}

extern "C" __global__ void pf8070_clarity_kernel(
    const unsigned char* source_rgba,
    const unsigned char* base_rgba,
    const unsigned char* blur_rgba,
    const unsigned char* mask_r,
    unsigned char* out_rgba,
    SkinModelsClarityUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.roi_w || y >= (int)uniforms.roi_h) return;

  unsigned int local =
      (unsigned int)y * uniforms.base_width + (unsigned int)x;
  float amount = sm_mul(uniforms.degree, sm_unit(mask_r[local]));
  if (!(amount > 0.0f)) return;

  unsigned int source_index =
      (uniforms.roi_y + (unsigned int)y) * uniforms.stride
      + (uniforms.roi_x + (unsigned int)x) * 4u;
  unsigned int local_index = local * 4u;
  for (int channel = 0; channel < 3; ++channel) {
    float src = sm_unit(source_rgba[source_index + (unsigned int)channel]);
    float base_value = sm_unit(base_rgba[local_index + (unsigned int)channel]);
    float blur_value = sm_unit(blur_rgba[local_index + (unsigned int)channel]);
    float diff = sm_sub(base_value, blur_value);
    float negative = fmaxf(sm_sub(sm_neg(diff), uniforms.threshold), 0.0f);
    float positive = fmaxf(sm_sub(diff, uniforms.threshold), 0.0f);
    float edge = sm_sub(positive, negative);
    float reference = sm_clamp(
        sm_add(src, sm_mul(uniforms.factor, edge)), 0.0f, 1.0f);
    out_rgba[source_index + (unsigned int)channel] =
        sm_quantize(sm_lerp(src, reference, amount));
  }
}
