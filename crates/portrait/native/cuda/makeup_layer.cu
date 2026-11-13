#include "makeup_layer.h"

// Source-order translation of kernels::makeup_material::apply_image_layer_dual.
// Compile with -fmad=false; every float step is an explicit binary32 intrinsic.

__device__ __forceinline__ float k_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float k_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float k_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float k_div(float a, float b) { return __fdiv_rn(a, b); }

__device__ __forceinline__ float k_clamp01(float value) {
  if (isnan(value)) return 0.0f;
  return fminf(fmaxf(value, 0.0f), 1.0f);
}

// Rust byte_even: clamp, scale, round halves to even.
__device__ __forceinline__ unsigned char k_byte_even(float value) {
  float v = k_mul(fminf(fmaxf(value, 0.0f), 1.0f), 255.0f);
  int q = (int)rintf(v);
  return (unsigned char)(q < 0 ? 0 : (q > 255 ? 255 : q));
}

// Clamped bilinear with the texel-centre rule, matching `sample_rgba`.
__device__ __forceinline__ void k_sample(
    const unsigned char* image,
    unsigned int width,
    unsigned int height,
    float u,
    float v,
    float* out) {
  u = k_clamp01(u);
  v = k_clamp01(v);
  float x = k_sub(k_mul(u, (float)width), 0.5f);
  float y = k_sub(k_mul(v, (float)height), 0.5f);
  float x0f = floorf(x);
  float y0f = floorf(y);
  float tx = k_sub(x, x0f);
  float ty = k_sub(y, y0f);
  int x0 = (int)x0f;
  int y0 = (int)y0f;
  x0 = x0 < 0 ? 0 : (x0 > (int)width - 1 ? (int)width - 1 : x0);
  int x1 = ((int)x0f) + 1;
  x1 = x1 < 0 ? 0 : (x1 > (int)width - 1 ? (int)width - 1 : x1);
  y0 = y0 < 0 ? 0 : (y0 > (int)height - 1 ? (int)height - 1 : y0);
  int y1 = ((int)y0f) + 1;
  y1 = y1 < 0 ? 0 : (y1 > (int)height - 1 ? (int)height - 1 : y1);
  for (int channel = 0; channel < 4; ++channel) {
    float p00 = k_div((float)image[((size_t)y0 * width + x0) * 4u + channel], 255.0f);
    float p10 = k_div((float)image[((size_t)y0 * width + x1) * 4u + channel], 255.0f);
    float p01 = k_div((float)image[((size_t)y1 * width + x0) * 4u + channel], 255.0f);
    float p11 = k_div((float)image[((size_t)y1 * width + x1) * 4u + channel], 255.0f);
    float top = k_add(k_mul(p00, k_sub(1.0f, tx)), k_mul(p10, tx));
    float bottom = k_add(k_mul(p01, k_sub(1.0f, tx)), k_mul(p11, tx));
    out[channel] = k_add(k_mul(top, k_sub(1.0f, ty)), k_mul(bottom, ty));
  }
}

__device__ __forceinline__ float k_mask_weight(const float* mask, unsigned int channel) {
  float value;
  switch (channel) {
    case 0u: value = mask[0]; break;
    case 1u: value = mask[1]; break;
    case 2u: value = mask[2]; break;
    case 3u: value = mask[3]; break;
    case 4u: value = k_add(mask[0], mask[1]); break;
    case 5u: value = k_add(mask[0], mask[2]); break;
    case 6u: value = k_add(mask[0], mask[3]); break;
    case 7u: value = k_add(mask[1], mask[2]); break;
    case 8u: value = k_add(mask[1], mask[3]); break;
    case 9u: value = k_add(mask[2], mask[3]); break;
    case 10u: value = k_add(k_add(mask[0], mask[1]), mask[2]); break;
    case 11u: value = k_add(k_add(mask[0], mask[1]), mask[3]); break;
    case 12u: value = k_add(k_add(mask[0], mask[2]), mask[3]); break;
    case 13u: value = k_add(k_add(mask[1], mask[2]), mask[3]); break;
    default: value = k_add(k_add(mask[0], mask[1]), k_add(mask[2], mask[3])); break;
  }
  return k_clamp01(value);
}

__device__ __forceinline__ float k_soft_light(float base, float blend) {
  if (blend < 0.5f) {
    return k_add(k_mul(k_mul(2.0f, base), blend), k_mul(k_mul(base, base), k_sub(1.0f, k_mul(2.0f, blend))));
  }
  return k_add(k_mul(sqrtf(base), k_sub(k_mul(2.0f, blend), 1.0f)), k_mul(k_mul(2.0f, base), k_sub(1.0f, blend)));
}

__device__ __forceinline__ float k_pin_light(float base, float blend) {
  if (blend < 0.5f) {
    return fminf(base, k_mul(2.0f, blend));
  }
  return fmaxf(base, k_mul(2.0f, k_sub(blend, 0.5f)));
}

__device__ __forceinline__ float k_blend_channel(float base, float blend, unsigned int mode) {
  switch (mode) {
    case 0u: return blend;
    case 1u: return k_mul(base, blend);
    case 2u: return k_soft_light(base, blend);
    default: return k_pin_light(base, blend);
  }
}

extern "C" __global__ void makeup_layer_kernel(
    const unsigned char* source_rgba,
    const unsigned char* material_rgba,
    const unsigned char* owner_mask_rgba,
    const float* mask_uv,
    const float* material_uv,
    unsigned char* out_rgba,
    MakeupLayerUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;

  const size_t pixel = (size_t)y * (size_t)uniforms.width + (size_t)x;
  const size_t index = (size_t)y * (size_t)uniforms.stride + (size_t)x * 4u;
  out_rgba[index + 0u] = source_rgba[index + 0u];
  out_rgba[index + 1u] = source_rgba[index + 1u];
  out_rgba[index + 2u] = source_rgba[index + 2u];
  out_rgba[index + 3u] = source_rgba[index + 3u];

  float mask_sample[4];
  k_sample(owner_mask_rgba, uniforms.mask_width, uniforms.mask_height,
           mask_uv[pixel * 2u + 0u], mask_uv[pixel * 2u + 1u], mask_sample);
  float material_sample[4];
  k_sample(material_rgba, uniforms.material_width, uniforms.material_height,
           material_uv[pixel * 2u + 0u], material_uv[pixel * 2u + 1u], material_sample);

  const float factor = k_clamp01(
      k_mul(k_mul(material_sample[3], uniforms.mix_alpha),
            k_mask_weight(mask_sample, uniforms.mask_channel)));
  if (factor <= 0.01f) return;

  const float source[3] = {
      k_div((float)source_rgba[index + 0u], 255.0f),
      k_div((float)source_rgba[index + 1u], 255.0f),
      k_div((float)source_rgba[index + 2u], 255.0f)};
  const unsigned char source_alpha = source_rgba[index + 3u];
  for (int channel = 0; channel < 3; ++channel) {
    const float reference = k_blend_channel(source[channel], material_sample[channel], uniforms.mode);
    if (uniforms.blend_alpha != 0u) {
      out_rgba[index + (size_t)channel] = k_byte_even(reference);
    } else {
      out_rgba[index + (size_t)channel] =
          k_byte_even(k_add(k_mul(source[channel], k_sub(1.0f, factor)), k_mul(reference, factor)));
    }
  }
  out_rgba[index + 3u] = uniforms.blend_alpha != 0u
      ? k_byte_even(k_mul(factor, k_div((float)source_alpha, 255.0f)))
      : source_alpha;
}
