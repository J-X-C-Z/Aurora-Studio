#include "smile_warp.h"

// Source-order translation of kernels::smile_warp::pf100007_apply.
// Pixel space, byte-space bilinear, truncating store, alpha copied.
// Compile with -fmad=false; the lerps use explicit binary32 intrinsics so the
// compiler cannot contract them into fma.

__device__ __forceinline__ float s_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float s_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float s_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float s_div(float a, float b) { return __fdiv_rn(a, b); }

// Bilinear tap in byte space with the CPU's truncating store. Non-finite
// coordinates collapse to the same zero the Rust saturating cast produces.
__device__ __forceinline__ void smile_bilinear(
    const unsigned char* source,
    unsigned int width,
    unsigned int height,
    float x,
    float y,
    unsigned char* rgb) {
  const float limit_x = (float)(width - 1u);
  const float limit_y = (float)(height - 1u);
  float cx = fminf(fmaxf(x, 0.0f), limit_x);
  float cy = fminf(fmaxf(y, 0.0f), limit_y);
  if (isnan(x) || isnan(y)) {
    rgb[0] = 0;
    rgb[1] = 0;
    rgb[2] = 0;
    return;
  }
  float fx0 = floorf(cx);
  float fy0 = floorf(cy);
  unsigned int x0 = (unsigned int)fx0;
  unsigned int y0 = (unsigned int)fy0;
  unsigned int x1 = x0 + 1u < width ? x0 + 1u : width - 1u;
  unsigned int y1 = y0 + 1u < height ? y0 + 1u : height - 1u;
  float fx = s_sub(cx, fx0);
  float fy = s_sub(cy, fy0);
  for (int channel = 0; channel < 3; ++channel) {
    float p00 = (float)source[((size_t)y0 * width + x0) * 4u + (size_t)channel];
    float p10 = (float)source[((size_t)y0 * width + x1) * 4u + (size_t)channel];
    float p01 = (float)source[((size_t)y1 * width + x0) * 4u + (size_t)channel];
    float p11 = (float)source[((size_t)y1 * width + x1) * 4u + (size_t)channel];
    float top = s_add(p00, s_mul(s_sub(p10, p00), fx));
    float bottom = s_add(p01, s_mul(s_sub(p11, p01), fx));
    float value = s_add(top, s_mul(s_sub(bottom, top), fy));
    value = fminf(fmaxf(value, 0.0f), 255.0f);
    rgb[channel] = (unsigned char)(int)value;
  }
}

extern "C" __global__ void smile_warp_kernel(
    const unsigned char* source_bgra,
    const unsigned char* lip_mask,
    const unsigned char* mouth_mask,
    const float* model_dx,
    const float* model_dy,
    unsigned char* out_bgra,
    SmileUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;

  // Full-image dispatch: every pixel is copied first, so anything outside the
  // ROI keeps the source bytes exactly like the CPU consumer.
  const size_t pixel = (size_t)y * (size_t)uniforms.width + (size_t)x;
  const size_t index = (size_t)y * (size_t)uniforms.stride + (size_t)x * 4u;
  out_bgra[index + 0u] = source_bgra[index + 0u];
  out_bgra[index + 1u] = source_bgra[index + 1u];
  out_bgra[index + 2u] = source_bgra[index + 2u];
  out_bgra[index + 3u] = source_bgra[index + 3u];

  const size_t roi_right = (size_t)uniforms.roi_x + (size_t)uniforms.roi_w;
  const size_t roi_bottom = (size_t)uniforms.roi_y + (size_t)uniforms.roi_h;
  if ((size_t)x < (size_t)uniforms.roi_x || (size_t)y < (size_t)uniforms.roi_y ||
      (size_t)x >= roi_right || (size_t)y >= roi_bottom) {
    return;
  }

  float support = s_mul(s_div((float)lip_mask[pixel], 255.0f), s_div((float)mouth_mask[pixel], 255.0f));
  float weight = s_mul(uniforms.degree, support);
  if (support <= 0.001f || weight <= 0.0f) return;

  float sx = s_add((float)x, s_mul(weight, model_dx[pixel]));
  float sy = s_add((float)y, s_mul(weight, model_dy[pixel]));
  unsigned char rgb[3];
  smile_bilinear(source_bgra, uniforms.width, uniforms.height, sx, sy, rgb);
  out_bgra[index + 0u] = rgb[0];
  out_bgra[index + 1u] = rgb[1];
  out_bgra[index + 2u] = rgb[2];
}

// PF8033 entry: same body, but the mouth plane is the constant 255, so the
// support term is exactly `mask/255` (multiplying by 1.0 changes no bit).
// The caller therefore skips a whole 1-byte-per-pixel plane: no host fill, no
// device allocation, no upload.
extern "C" __global__ void smile_warp_mask_kernel(
    const unsigned char* source_bgra,
    const unsigned char* mask,
    const float* model_dx,
    const float* model_dy,
    unsigned char* out_bgra,
    SmileUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;

  const size_t pixel = (size_t)y * (size_t)uniforms.width + (size_t)x;
  const size_t index = (size_t)y * (size_t)uniforms.stride + (size_t)x * 4u;
  out_bgra[index + 0u] = source_bgra[index + 0u];
  out_bgra[index + 1u] = source_bgra[index + 1u];
  out_bgra[index + 2u] = source_bgra[index + 2u];
  out_bgra[index + 3u] = source_bgra[index + 3u];

  const size_t roi_right = (size_t)uniforms.roi_x + (size_t)uniforms.roi_w;
  const size_t roi_bottom = (size_t)uniforms.roi_y + (size_t)uniforms.roi_h;
  if ((size_t)x < (size_t)uniforms.roi_x || (size_t)y < (size_t)uniforms.roi_y ||
      (size_t)x >= roi_right || (size_t)y >= roi_bottom) {
    return;
  }

  float support = s_div((float)mask[pixel], 255.0f);
  float weight = s_mul(uniforms.degree, support);
  if (support <= 0.001f || weight <= 0.0f) return;

  float sx = s_add((float)x, s_mul(weight, model_dx[pixel]));
  float sy = s_add((float)y, s_mul(weight, model_dy[pixel]));
  unsigned char rgb[3];
  smile_bilinear(source_bgra, uniforms.width, uniforms.height, sx, sy, rgb);
  out_bgra[index + 0u] = rgb[0];
  out_bgra[index + 1u] = rgb[1];
  out_bgra[index + 2u] = rgb[2];
}
