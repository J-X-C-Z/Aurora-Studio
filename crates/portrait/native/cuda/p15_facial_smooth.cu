#include "p15_facial_smooth_public.h"

#include <cuda_runtime.h>
#include <cmath>
#include <stddef.h>

// P15 source-order translation. Compile with -fmad=false. All float operations
// that can affect the recovered expression are explicit binary32 intrinsics.

__device__ __forceinline__ float p15_add(float a, float b) {
  return __fadd_rn(a, b);
}

__device__ __forceinline__ float p15_sub(float a, float b) {
  return __fsub_rn(a, b);
}

__device__ __forceinline__ float p15_mul(float a, float b) {
  return __fmul_rn(a, b);
}

__device__ __forceinline__ float p15_div(float a, float b) {
  return __fdiv_rn(a, b);
}

__device__ __forceinline__ float p15_max(float a, float b) {
  return b > a ? b : a;
}

__device__ __forceinline__ float p15_min(float a, float b) {
  return b < a ? b : a;
}

__device__ __forceinline__ float p15_clamp(float v, float lo, float hi) {
  return p15_min(p15_max(v, lo), hi);
}

// GLSL mix(x,y,a) is x*(1-a)+y*a, not a fused operation.
__device__ __forceinline__ float p15_mix(float x, float y, float a) {
  return p15_add(p15_mul(x, p15_sub(1.0f, a)), p15_mul(y, a));
}

__device__ __forceinline__ float4 p15_load(const float* p, size_t offset) {
  return make_float4(p[offset], p[offset + 1], p[offset + 2], p[offset + 3]);
}

__device__ __forceinline__ float4 p15_bilinear(
    const float* rgba,
    const P15FacialSmoothUniforms& u,
    float tex_u,
    float tex_v) {
  float x = p15_sub(p15_mul(tex_u, (float)u.width), 0.5f);
  float y = p15_sub(p15_mul(tex_v, (float)u.height), 0.5f);
  float x0f = floorf(x);
  float y0f = floorf(y);
  float fx = p15_sub(x, x0f);
  float fy = p15_sub(y, y0f);
  int x0 = (int)p15_clamp(x0f, 0.0f, (float)(u.width - 1));
  int y0 = (int)p15_clamp(y0f, 0.0f, (float)(u.height - 1));
  int x1 = min(x0 + 1, (int)u.width - 1);
  int y1 = min(y0 + 1, (int)u.height - 1);
  size_t row0 = (size_t)y0 * u.stride;
  size_t row1 = (size_t)y1 * u.stride;
  size_t a = ((row0 + x0) * 4u);
  size_t b = ((row0 + x1) * 4u);
  size_t c = ((row1 + x0) * 4u);
  size_t d = ((row1 + x1) * 4u);
  float4 top = make_float4(
      p15_mix(rgba[a], rgba[b], fx),
      p15_mix(rgba[a + 1], rgba[b + 1], fx),
      p15_mix(rgba[a + 2], rgba[b + 2], fx),
      p15_mix(rgba[a + 3], rgba[b + 3], fx));
  float4 bottom = make_float4(
      p15_mix(rgba[c], rgba[d], fx),
      p15_mix(rgba[c + 1], rgba[d + 1], fx),
      p15_mix(rgba[c + 2], rgba[d + 2], fx),
      p15_mix(rgba[c + 3], rgba[d + 3], fx));
  return make_float4(
      p15_mix(top.x, bottom.x, fy),
      p15_mix(top.y, bottom.y, fy),
      p15_mix(top.z, bottom.z, fy),
      p15_mix(top.w, bottom.w, fy));
}

__device__ __forceinline__ float p15_smoothstep(float edge0, float edge1, float value) {
  float t = p15_clamp(
      p15_div(p15_sub(value, edge0), p15_sub(edge1, edge0)), 0.0f, 1.0f);
  return p15_mul(p15_mul(t, t), p15_sub(3.0f, p15_mul(2.0f, t)));
}

// SurfaceBlurBasic AlgoType=0: sample -1,+1,-2,+2,-3,+3,-4,+4 in order.
__device__ __forceinline__ float4 p15_surface_sample(
    const float* source,
    const P15FacialSmoothUniforms& u,
    float2 center,
    float2 step) {
  float4 center_rgba = p15_bilinear(source, u, center.x, center.y);
  float sum_r = center_rgba.x;
  float sum_g = center_rgba.y;
  float sum_b = center_rgba.z;
  float total_weight = 1.0f;
  const int distances[8] = {-1, 1, -2, 2, -3, 3, -4, 4};
  for (int i = 0; i < 8; ++i) {
    float4 sample = p15_bilinear(source, u,
        p15_add(center.x, p15_mul(step.x, (float)distances[i])),
        p15_add(center.y, p15_mul(step.y, (float)distances[i])));
    float dr = p15_sub(sample.x, center_rgba.x);
    float dg = p15_sub(sample.y, center_rgba.y);
    float db = p15_sub(sample.z, center_rgba.z);
    float squared = p15_add(
        p15_add(p15_mul(dr, dr), p15_mul(dg, dg)), p15_mul(db, db));
    float difference = sqrtf(squared);
    float weight = p15_sub(1.0f, p15_smoothstep(
        0.0f, p15_div(u.surface_threshold, 255.0f), difference));
    sum_r = p15_add(sum_r, p15_mul(sample.x, weight));
    sum_g = p15_add(sum_g, p15_mul(sample.y, weight));
    sum_b = p15_add(sum_b, p15_mul(sample.z, weight));
    total_weight = p15_add(total_weight, weight);
  }
  return make_float4(
      p15_div(sum_r, total_weight),
      p15_div(sum_g, total_weight),
      p15_div(sum_b, total_weight),
      center_rgba.w);
}

// 1DGaussBlur: center * 0.18 then eight weights in original order.
__device__ __forceinline__ float4 p15_gauss_sample(
    const float* source,
    const P15FacialSmoothUniforms& u,
    float2 center,
    float2 step) {
  const int distances[8] = {-4, -3, -2, -1, 1, 2, 3, 4};
  const float weights[8] = {0.05f, 0.09f, 0.12f, 0.15f,
                            0.15f, 0.12f, 0.09f, 0.05f};
  float4 center_rgba = p15_bilinear(source, u, center.x, center.y);
  float4 sum = make_float4(
      p15_mul(center_rgba.x, 0.18f),
      p15_mul(center_rgba.y, 0.18f),
      p15_mul(center_rgba.z, 0.18f),
      p15_mul(center_rgba.w, 0.18f));
  for (int i = 0; i < 8; ++i) {
    float4 sample = p15_bilinear(source, u,
        p15_add(center.x, p15_mul(step.x, (float)distances[i])),
        p15_add(center.y, p15_mul(step.y, (float)distances[i])));
    sum.x = p15_add(sum.x, p15_mul(sample.x, weights[i]));
    sum.y = p15_add(sum.y, p15_mul(sample.y, weights[i]));
    sum.z = p15_add(sum.z, p15_mul(sample.z, weights[i]));
    sum.w = p15_add(sum.w, p15_mul(sample.w, weights[i]));
  }
  return sum;
}

// FacialSmoothBoxBlur: 3x3 mean, source alpha preserved.
__device__ __forceinline__ float4 p15_box_sample(
    const float* source,
    const P15FacialSmoothUniforms& u,
    float2 center) {
  float3 sum = make_float3(0.0f, 0.0f, 0.0f);
  float step_u = p15_div(1.0f, (float)u.width);
  float step_v = p15_div(1.0f, (float)u.height);
  for (int j = -1; j <= 1; ++j) {
    for (int i = -1; i <= 1; ++i) {
      float4 sample = p15_bilinear(source, u,
          p15_add(center.x, p15_mul(step_u, (float)i)),
          p15_add(center.y, p15_mul(step_v, (float)j)));
      sum.x = p15_add(sum.x, sample.x);
      sum.y = p15_add(sum.y, sample.y);
      sum.z = p15_add(sum.z, sample.z);
    }
  }
  float4 center_rgba = p15_bilinear(source, u, center.x, center.y);
  return make_float4(
      p15_div(sum.x, 9.0f),
      p15_div(sum.y, 9.0f),
      p15_div(sum.z, 9.0f),
      center_rgba.w);
}

__device__ __forceinline__ float p15_composite_channel(
    float source, float base_value, float residual, float alpha,
    const P15FacialSmoothUniforms& u) {
  float high = p15_sub(source, base_value);
  float diff_left = p15_mul(p15_sub(residual, base_value), u.diff_factor);
  float diff_right = p15_mul(residual, u.diff_factor2);
  float diff = p15_clamp(p15_add(diff_left, diff_right),
                         u.diff_min_threshold, u.diff_max_threshold);
  float ref_threshold = p15_mix(
      p15_mul(residual, u.color_threshold_factor), residual, u.mix_alpha);
  float maximum = p15_max(ref_threshold, p15_sub(source, high));
  float before_clamp = p15_sub(p15_add(source, diff), high);
  float reference = p15_add(p15_clamp(before_clamp, 0.0f, maximum), high);
  float mixed = p15_mix(source, reference, alpha);
  float high_frequency = p15_clamp(p15_sub(mixed, source),
      u.high_frequency_min_threshold, u.high_frequency_max_threshold);
  return p15_add(source, high_frequency);
}

extern "C" __global__ void p15_facial_smooth_composite_kernel(
    const float* source_rgba,
    const float* mask_rgba,
    const float* skin_mask_r,
    const float* base_rgba,
    const float* residual_base_rgba,
    float* output_rgba,
    P15FacialSmoothUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;
  int pixel = y * (int)uniforms.stride + x;
  size_t offset = (size_t)pixel * 4u;
  float4 source = p15_load(source_rgba, offset);
  float4 mask = p15_load(mask_rgba, offset);
  float4 base_color = p15_load(base_rgba, offset);
  float4 residual = p15_load(residual_base_rgba, offset);
  float alpha = p15_mul(
      p15_max(p15_max(
          p15_mul(uniforms.dark_circle_alpha, mask.x),
          p15_mul(uniforms.legal_alpha, mask.z)), 0.0f),
      skin_mask_r[pixel]);
  float4 result = source;
  if (alpha > 0.0f) {
    result.x = p15_composite_channel(
        source.x, base_color.x, residual.x, alpha, uniforms);
    result.y = p15_composite_channel(
        source.y, base_color.y, residual.y, alpha, uniforms);
    result.z = p15_composite_channel(
        source.z, base_color.z, residual.z, alpha, uniforms);
  }
  result.w = source.w;
  output_rgba[offset] = result.x;
  output_rgba[offset + 1] = result.y;
  output_rgba[offset + 2] = result.z;
  output_rgba[offset + 3] = result.w;
}

extern "C" __global__ void p15_surface_pass_kernel(
    const float* source_rgba,
    float* output_rgba,
    P15FacialSmoothUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;
  float2 center = make_float2(
      p15_div((float)x + 0.5f, (float)uniforms.width),
      p15_div((float)y + 0.5f, (float)uniforms.height));
  float2 step = uniforms.box_blur_stage == 1u
      ? make_float2(p15_div(uniforms.surface_radius, (float)uniforms.width), 0.0f)
      : make_float2(0.0f, p15_div(uniforms.surface_radius, (float)uniforms.height));
  float4 result = p15_surface_sample(source_rgba, uniforms, center, step);
  size_t offset = ((size_t)y * uniforms.stride + x) * 4u;
  output_rgba[offset] = result.x;
  output_rgba[offset + 1] = result.y;
  output_rgba[offset + 2] = result.z;
  output_rgba[offset + 3] = result.w;
}

extern "C" __global__ void p15_gauss_pass_kernel(
    const float* source_rgba,
    float* output_rgba,
    P15FacialSmoothUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;
  float2 center = make_float2(
      p15_div((float)x + 0.5f, (float)uniforms.width),
      p15_div((float)y + 0.5f, (float)uniforms.height));
  float2 step = uniforms.box_blur_stage == 1u
      ? make_float2(p15_div(uniforms.gauss_offset, (float)uniforms.width), 0.0f)
      : make_float2(0.0f, p15_div(uniforms.gauss_offset, (float)uniforms.height));
  float4 result = p15_gauss_sample(source_rgba, uniforms, center, step);
  size_t offset = ((size_t)y * uniforms.stride + x) * 4u;
  output_rgba[offset] = result.x;
  output_rgba[offset + 1] = result.y;
  output_rgba[offset + 2] = result.z;
  output_rgba[offset + 3] = result.w;
}

extern "C" __global__ void p15_box_blur_kernel(
    const float* source_rgba,
    float* output_rgba,
    P15FacialSmoothUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;
  float2 center = make_float2(
      p15_div((float)x + 0.5f, (float)uniforms.width),
      p15_div((float)y + 0.5f, (float)uniforms.height));
  float4 result = p15_box_sample(source_rgba, uniforms, center);
  size_t offset = ((size_t)y * uniforms.stride + x) * 4u;
  output_rgba[offset] = result.x;
  output_rgba[offset + 1] = result.y;
  output_rgba[offset + 2] = result.z;
  output_rgba[offset + 3] = result.w;
}

extern "C" int32_t p15_validate_launch(const P15LiveInputs* inputs) {
  if (inputs == nullptr || inputs->uniforms == nullptr) {
    return P15_STATUS_INVALID_UNIFORMS;
  }
  const P15FacialSmoothUniforms& u = *inputs->uniforms;
  if (u.width == 0 || u.height == 0 || u.stride < u.width ||
      u.roi_size != u.width || u.roi_size != u.height ||
      !std::isfinite(u.pf_value) || !std::isfinite(u.surface_radius) ||
      !std::isfinite(u.surface_threshold) || !std::isfinite(u.gauss_offset)) {
    return P15_STATUS_INVALID_LAYOUT;
  }
  if (u.pf != 90023u && u.pf != 90024u && u.pf != 21024u) {
    return P15_STATUS_INVALID_UNIFORMS;
  }
  if (!(u.pf_value > u.gate) || u.gate != 0.001f) {
    return P15_STATUS_INVALID_GATE;
  }
  if (u.algo_type != 0u) {
    return P15_STATUS_UNSUPPORTED_ALGO;
  }
  if (inputs->source_rgba == nullptr || inputs->mask_rgba == nullptr ||
      inputs->skin_mask_r == nullptr || inputs->base_rgba == nullptr ||
      inputs->residual_base_rgba == nullptr || inputs->output_rgba == nullptr) {
    return P15_STATUS_MISSING_TEXTURE;
  }
  if (!inputs->gl_state_captured) {
    return P15_STATUS_MISSING_GL_STATE;
  }
  return P15_STATUS_OK;
}
