#include "skin_lut.h"

// Source-order translation of `kernels::makeup::skin_adjust_lut`.
// Compile with -fmad=false; every float op that can change a pixel is an
// explicit binary32 intrinsic. The tap is integer fixed point (8-bit weights,
// (q*257+128)>>8), the blend is mul-then-add (no fma, matching the Rust lane),
// and the store is 12-bit quantised: ((floor(v*4096)*255 + 2047) >> 12).

__device__ __forceinline__ float sl_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float sl_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float sl_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float sl_div(float a, float b) { return __fdiv_rn(a, b); }

__device__ __forceinline__ int sl_min(int a, int b) { return b < a ? b : a; }
__device__ __forceinline__ int sl_max(int a, int b) { return b > a ? b : a; }

__device__ __forceinline__ float sl_floor(float value) { return floorf(value); }

__device__ __forceinline__ int sl_lut_byte(
    const unsigned char* lut,
    unsigned int x,
    unsigned int y,
    unsigned int channel) {
  return (int)lut[(y * 512u + x) * 3u + channel];
}

// Bilinear tap with the recovered 8-bit weight fixed point.
__device__ __forceinline__ float sl_tap(
    const unsigned char* lut,
    unsigned int channel,
    float px,
    float py) {
  float u = sl_sub(sl_mul(px, 512.0f), 0.5f);
  float v = sl_sub(sl_mul(py, 512.0f), 0.5f);
  float i0f = sl_floor(u);
  float j0f = sl_floor(v);
  int i0 = (int)i0f;
  int j0 = (int)j0f;
  int kx = (int)sl_floor(sl_add(sl_mul(sl_sub(u, i0f), 256.0f), 0.5f));
  int ky = (int)sl_floor(sl_add(sl_mul(sl_sub(v, j0f), 256.0f), 0.5f));
  int t = (kx * ky + 128) >> 8;
  int w00 = 256 - kx - ky + t;
  int w10 = kx - t;
  int w01 = ky - t;
  int w11 = t;
  unsigned int x0 = (unsigned int)sl_min(sl_max(i0, 0), 511);
  unsigned int x1 = (unsigned int)sl_min(sl_max(i0 + 1, 0), 511);
  unsigned int y0 = (unsigned int)sl_min(sl_max(j0, 0), 511);
  unsigned int y1 = (unsigned int)sl_min(sl_max(j0 + 1, 0), 511);
  int q = sl_lut_byte(lut, x0, y0, channel) * w00 +
          sl_lut_byte(lut, x1, y0, channel) * w10 +
          sl_lut_byte(lut, x0, y1, channel) * w01 +
          sl_lut_byte(lut, x1, y1, channel) * w11;
  q = q * 257;
  return sl_div((float)((q + 128) >> 8), 65535.0f);
}

// Blue picks the 64x64 tile; r/g pick the position inside it.
__device__ __forceinline__ float sl_lut(
    const unsigned char* lut,
    float r,
    float g,
    float b,
    unsigned int channel) {
  float blue = sl_mul(b, 63.0f);
  float bf = sl_floor(blue);
  float q1y = sl_floor(sl_mul(bf, 0.125f));
  float q1x = sl_sub(bf, sl_mul(q1y, 8.0f));
  float bc = ceilf(blue);
  float q2y = sl_floor(sl_mul(bc, 0.125f));
  float q2x = sl_sub(bc, sl_mul(q2y, 8.0f));
  const float k = 0.0009765625f;
  float span = __uint_as_float(0x3dfc0001u);
  float x1 = sl_add(sl_mul(span, r), sl_add(sl_mul(q1x, 0.125f), k));
  float y1 = sl_add(sl_mul(span, g), sl_add(sl_mul(q1y, 0.125f), k));
  float x2 = sl_add(sl_add(sl_mul(q2x, 0.125f), sl_mul(span, r)), k);
  float y2 = sl_add(sl_add(sl_mul(q2y, 0.125f), sl_mul(span, g)), k);
  float low = sl_tap(lut, channel, x1, y1);
  float high = sl_tap(lut, channel, x2, y2);
  return sl_add(low, sl_mul(sl_sub(blue, bf), sl_sub(high, low)));
}

// 12-bit quantisation of the blended channel.
__device__ __forceinline__ unsigned char sl_quantize(float value) {
  int value12 = (int)sl_floor(sl_mul(value, 4096.0f));
  int q = (value12 * 255 + 2047) >> 12;
  return (unsigned char)sl_min(sl_max(q, 0), 255);
}

extern "C" __global__ void skin_adjust_lut_kernel(
    const unsigned char* source_rgba,
    const unsigned char* mask_r8,
    const unsigned char* lut_rgb512,
    unsigned char* output_rgba,
    SkinLutUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;

  unsigned int source_index = (unsigned int)y * uniforms.stride + (unsigned int)x * 4u;
  unsigned int mask_index = (unsigned int)y * uniforms.mask_row_bytes + (unsigned int)x;

  unsigned char source_r = source_rgba[source_index + 0u];
  unsigned char source_g = source_rgba[source_index + 1u];
  unsigned char source_b = source_rgba[source_index + 2u];
  unsigned char source_a = source_rgba[source_index + 3u];

  output_rgba[source_index + 0u] = source_r;
  output_rgba[source_index + 1u] = source_g;
  output_rgba[source_index + 2u] = source_b;
  output_rgba[source_index + 3u] = source_a;

  float mask = sl_div((float)mask_r8[mask_index], 255.0f);
  float weight = sl_mul(mask, uniforms.mix);
  if (!(weight > uniforms.gate)) return;

  float r = sl_div((float)source_r, 255.0f);
  float g = sl_div((float)source_g, 255.0f);
  float b = sl_div((float)source_b, 255.0f);
  float lut_r = sl_lut(lut_rgb512, r, g, b, 0u);
  float lut_g = sl_lut(lut_rgb512, r, g, b, 1u);
  float lut_b = sl_lut(lut_rgb512, r, g, b, 2u);

  float blended_r = sl_add(r, sl_mul(weight, sl_sub(lut_r, r)));
  float blended_g = sl_add(g, sl_mul(weight, sl_sub(lut_g, g)));
  float blended_b = sl_add(b, sl_mul(weight, sl_sub(lut_b, b)));

  output_rgba[source_index + 0u] = sl_quantize(blended_r);
  output_rgba[source_index + 1u] = sl_quantize(blended_g);
  output_rgba[source_index + 2u] = sl_quantize(blended_b);
  output_rgba[source_index + 3u] = source_a;
}
