#include "pf30030_blue_vein.h"

// The correction image is authored by the vendor model with neutral byte 80.
// Integer rounding is deliberate: round half away from zero, then clamp.
__device__ __forceinline__ int scale_correction(int delta, unsigned int mask) {
  const int product = delta * (int)mask;
  return product >= 0 ? (product + 127) / 255 : -((-product + 127) / 255);
}

extern "C" __global__ void pf30030_blue_vein_kernel(
    const unsigned char* source_bgra,
    const unsigned char* feathered_mask,
    const unsigned char* correction_rgb,
    unsigned char* out_bgra,
    Pf30030BlueVeinUniforms uniforms) {
  const unsigned int local_x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int local_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (local_x >= uniforms.roi_width || local_y >= uniforms.roi_height) return;

  const unsigned int x = uniforms.roi_x + local_x;
  const unsigned int y = uniforms.roi_y + local_y;
  const size_t pixel = (size_t)y * (size_t)uniforms.width + (size_t)x;
  const size_t bgra = (size_t)y * (size_t)uniforms.stride + (size_t)x * 4u;
  const unsigned int mask = feathered_mask[pixel];
  if (mask == 0u) return;

  for (unsigned int channel = 0; channel < 3u; ++channel) {
    const int delta = (int)correction_rgb[pixel * 3u + (size_t)channel] - 80;
    const int scaled = scale_correction(delta, mask);
    int value = (int)source_bgra[bgra + (size_t)channel] + scaled;
    value = value < 0 ? 0 : (value > 255 ? 255 : value);
    out_bgra[bgra + (size_t)channel] = (unsigned char)value;
  }
}
