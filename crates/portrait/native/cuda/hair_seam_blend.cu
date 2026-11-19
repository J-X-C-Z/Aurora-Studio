#include "hair_seam_blend.h"

// Source-order translation of kernels::face_detail::hair_seam_blend.
// Compile with -fmad=false; every arithmetic step is an explicit binary32
// intrinsic. The store reproduces Rust byte_away, not ties-to-even roundf.

__device__ __forceinline__ float hs_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float hs_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float hs_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float hs_div(float a, float b) { return __fdiv_rn(a, b); }

__device__ __forceinline__ unsigned char hs_byte_away(float value) {
  if (!isfinite(value)) return 0u;
  const float adjusted = value >= 0.0f ? hs_add(value, 0.5f) : hs_sub(value, -0.5f);
  const int truncated = (int)adjusted;
  return (unsigned char)(truncated < 0 ? 0 : (truncated > 255 ? 255 : truncated));
}

extern "C" __global__ void pf90087_hair_seam_blend_kernel(
    const unsigned char* source_rgba,
    const unsigned char* part_line_mask,
    const unsigned char* repair_rgb,
    const unsigned char* repair_mask,
    unsigned char* out_rgba,
    HairSeamBlendUniforms uniforms) {
  const unsigned int dx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= uniforms.roi_w || dy >= uniforms.roi_h) return;

  const unsigned int x = uniforms.roi_x + dx;
  const unsigned int y = uniforms.roi_y + dy;
  if (x >= uniforms.width || y >= uniforms.height) return;

  const size_t pixel =
      (size_t)y * (size_t)uniforms.width + (size_t)x;
  const size_t rgba = pixel * 4u;
  const size_t repair = pixel * 3u;

  const unsigned int seam8 = part_line_mask[pixel];
  if (seam8 != 0u && repair_mask[pixel] != 0u) {
    const float seam = hs_div((float)seam8, 255.0f);
    const float unclamped = hs_mul(uniforms.degree, seam);
    const float weight = fminf(fmaxf(unclamped, 0.0f), 1.0f);
    for (unsigned int channel = 0; channel < 3u; ++channel) {
      const float src = (float)source_rgba[rgba + channel];
      const float rep = (float)repair_rgb[repair + channel];
      const float low = hs_mul(src, hs_sub(1.0f, weight));
      const float high = hs_mul(rep, weight);
      out_rgba[rgba + channel] = hs_byte_away(hs_add(low, high));
    }
  } else {
    out_rgba[rgba + 0u] = source_rgba[rgba + 0u];
    out_rgba[rgba + 1u] = source_rgba[rgba + 1u];
    out_rgba[rgba + 2u] = source_rgba[rgba + 2u];
  }
  out_rgba[rgba + 3u] = source_rgba[rgba + 3u];
}
