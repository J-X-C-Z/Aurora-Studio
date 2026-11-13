#include "gloss_lut.h"

// Source-order translation of kernels::skin_models::{gloss_apply, gloss_pass}.
// Compile with -fmad=false; every float op that can change a byte goes through
// an explicit binary32 intrinsic. Every LUT is an explicit caller-owned input.

__device__ __forceinline__ float g_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float g_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float g_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float g_div(float a, float b) { return __fdiv_rn(a, b); }

// position = coded/255*256 - 0.5; edge taps clamp independently.
__device__ __forceinline__ float gloss_lut_sample(const unsigned char* lut, unsigned char coded) {
  float position = g_sub(g_mul(g_div((float)coded, 255.0f), 256.0f), 0.5f);
  float lower = floorf(position);
  float fraction = g_sub(position, lower);
  int index = (int)lower;
  int right_raw = index + 1;
  int left = index < 0 ? 0 : (index > 255 ? 255 : index);
  int right = right_raw < 0 ? 0 : (right_raw > 255 ? 255 : right_raw);
  float low = g_div((float)lut[left], 255.0f);
  float high = g_div((float)lut[right], 255.0f);
  return g_add(g_mul(low, g_sub(1.0f, fraction)), g_mul(high, fraction));
}

// Rust to_byte: <=0 -> 0, >=255 -> 255, else round ties to even.
__device__ __forceinline__ unsigned char gloss_to_byte(float value) {
  float encoded = g_mul(value, 255.0f);
  if (encoded <= 0.0f) return 0;
  if (encoded >= 255.0f) return 255;
  return (unsigned char)rintf(encoded);
}

extern "C" __global__ void gloss_pass_kernel(
    const unsigned char* source_rgba,
    const unsigned char* mask_r,
    const unsigned char* extra_mask_r,
    const unsigned char* lut,
    unsigned char* out_rgba,
    GlossUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;

  unsigned int index = (unsigned int)y * uniforms.stride + (unsigned int)x * 4u;
  unsigned int mask_index = (unsigned int)y * uniforms.mask_row_bytes + (unsigned int)x;

  unsigned char px0 = source_rgba[index + 0u];
  unsigned char px1 = source_rgba[index + 1u];
  unsigned char px2 = source_rgba[index + 2u];
  unsigned char alpha = source_rgba[index + 3u];

  float mask = g_mul(
      g_div((float)mask_r[mask_index], 255.0f),
      g_div((float)extra_mask_r[mask_index], 255.0f));


  unsigned char coded[3] = {px0, px1, px2};
  unsigned char out_coded[3];
  for (int channel = 0; channel < 3; ++channel) {
    float source = g_div((float)coded[channel], 255.0f);
    float mapped = gloss_lut_sample(lut, coded[channel]);
    float value = g_add(source, g_mul(g_sub(mapped, source), uniforms.degree));
    value = fminf(fmaxf(value, 0.0f), 1.0f);
    float result = g_add(source, g_mul(g_sub(value, source), mask));
    out_coded[channel] = gloss_to_byte(result);
  }

  out_rgba[index + 0u] = out_coded[0];
  out_rgba[index + 1u] = out_coded[1];
  out_rgba[index + 2u] = out_coded[2];
  out_rgba[index + 3u] = alpha;
}
