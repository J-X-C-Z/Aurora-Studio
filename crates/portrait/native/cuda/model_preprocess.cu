#include "model_preprocess.h"

// CPU twins: crates/portrait/src/cuda_model_preprocess.rs
// Compile with -fmad=false. The explicit binary32 intrinsics pin the CPU
// operation order; '/' keeps its default -prec-div=true so both sides run the
// same correctly rounded division.

__device__ __forceinline__ float mpp_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float mpp_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float mpp_mul(float a, float b) { return __fmul_rn(a, b); }

// Mirrors Rust byte_away in kernels::face_detail: non-finite -> 0, truncate
// toward zero, then clamp to 0..255.
__device__ __forceinline__ unsigned char mpp_byte_away(float value) {
  if (!isfinite(value)) return 0;
  const float adjusted = value >= 0.0f ? mpp_add(value, 0.5f) : mpp_sub(value, 0.5f);
  int rounded;
  if (adjusted >= 2147483648.0f) rounded = 2147483647;
  else if (adjusted <= -2147483649.0f) rounded = -2147483647 - 1;
  else rounded = static_cast<int>(adjusted);
  if (rounded < 0) return 0;
  if (rounded > 255) return 255;
  return static_cast<unsigned char>(rounded);
}

// Maps one output coordinate to (i0, i1, frac) with the CPU's half-pixel
// centers. The multiply-then-divide order matches `(i + 0.5) * src / out - 0.5`
// in the engine's CPU preprocess, so the rounding is identical.
__device__ __forceinline__ void mpp_axis(float out_index, float src_count,
                                         float out_count, unsigned int src_last,
                                         unsigned int* i0, unsigned int* i1,
                                         float* frac) {
  const float scaled =
      mpp_sub((mpp_mul(mpp_add(out_index, 0.5f), src_count) / out_count), 0.5f);
  float floored = floorf(scaled);
  if (!(floored > 0.0f)) floored = 0.0f;
  if (floored > (float)src_last) floored = (float)src_last;
  unsigned int x0 = static_cast<unsigned int>(floored);
  unsigned int x1 = x0 + 1u;
  if (x1 > src_last) x1 = src_last;
  float f = mpp_sub(scaled, floored);
  if (!(f > 0.0f)) f = 0.0f;
  if (f > 1.0f) f = 1.0f;
  *i0 = x0;
  *i1 = x1;
  *frac = f;
}

extern "C" __global__ void mpp_rgba8_to_nchw_kernel(
    const unsigned char* src_rgba, float* out_nchw, MppNchwUniforms uniforms) {
  const unsigned int total = uniforms.out_w * uniforms.out_h;
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const unsigned int x = index % uniforms.out_w;
  const unsigned int y = index / uniforms.out_w;

  unsigned int x0, x1, y0, y1;
  float fx, fy;
  // The ROI rectangle is the sampling window; the CPU reference passes the ROI
  // width/height as the source extent.
  mpp_axis((float)x, (float)uniforms.roi_w, (float)uniforms.out_w,
           uniforms.roi_w - 1u, &x0, &x1, &fx);
  mpp_axis((float)y, (float)uniforms.roi_h, (float)uniforms.out_h,
           uniforms.roi_h - 1u, &y0, &y1, &fy);
  x0 += uniforms.roi_x;
  x1 += uniforms.roi_x;
  y0 += uniforms.roi_y;
  y1 += uniforms.roi_y;

  const size_t plane = (size_t)uniforms.out_w * (size_t)uniforms.out_h;
  const float one = 1.0f;
  for (unsigned int channel = 0; channel < uniforms.channels; ++channel) {
    // Byte order: identity for RGB, reversed for the BGRA model sources.
    unsigned int byte;
    if (channel > 2u) byte = 3u;
    else if (uniforms.swap_rb) byte = (channel == 0u) ? 2u : (channel == 2u ? 0u : 1u);
    else byte = channel;

    const size_t row0 = (size_t)y0 * uniforms.src_stride;
    const size_t row1 = (size_t)y1 * uniforms.src_stride;
    const float v00 = (float)src_rgba[(row0 + x0) * 4u + byte] / uniforms.divisor;
    const float v10 = (float)src_rgba[(row0 + x1) * 4u + byte] / uniforms.divisor;
    const float v01 = (float)src_rgba[(row1 + x0) * 4u + byte] / uniforms.divisor;
    const float v11 = (float)src_rgba[(row1 + x1) * 4u + byte] / uniforms.divisor;
    const float top = mpp_add(mpp_mul(v00, mpp_sub(one, fx)), mpp_mul(v10, fx));
    const float bottom = mpp_add(mpp_mul(v01, mpp_sub(one, fx)), mpp_mul(v11, fx));
    const float sample = mpp_add(mpp_mul(top, mpp_sub(one, fy)), mpp_mul(bottom, fy));
    out_nchw[(size_t)channel * plane + index] =
        (mpp_sub(sample, uniforms.mean[channel]) / uniforms.std_dev[channel]);
  }
}

extern "C" __global__ void mpp_nhwc4_to_planar_kernel(
    const float* src_nhwc4, float* out_planes, MppPlanarUniforms uniforms) {
  const unsigned int total = uniforms.width * uniforms.height;
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const unsigned int x = index % uniforms.width;
  const unsigned int y = index / uniforms.width;
  const size_t source = ((size_t)y * uniforms.stride + x) * 4u;
  const size_t plane = (size_t)total;
  for (unsigned int channel = 0; channel < uniforms.channels; ++channel) {
    out_planes[(size_t)channel * plane + index] = src_nhwc4[source + channel];
  }
}

extern "C" __global__ void mpp_f32_to_u8_kernel(
    const float* src_f32, unsigned char* out_u8, MppQuantUniforms uniforms) {
  const unsigned int total = uniforms.width * uniforms.height;
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const unsigned int x = index % uniforms.width;
  const unsigned int y = index / uniforms.width;
  const size_t source = ((size_t)y * uniforms.stride + x) * uniforms.channels;
  const size_t dest = (size_t)index * uniforms.channels;
  for (unsigned int channel = 0; channel < uniforms.channels; ++channel) {
    const float scaled = mpp_add(
        mpp_mul(src_f32[source + channel], uniforms.scale), uniforms.offset);
    out_u8[dest + channel] = mpp_byte_away(scaled);
  }
}
