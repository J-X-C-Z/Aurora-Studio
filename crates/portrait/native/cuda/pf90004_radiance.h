#ifndef AURORA_PF90004_RADIANCE_H
#define AURORA_PF90004_RADIANCE_H

#include <stdint.h>

/*
 * PF90004 fixed LUT/gamma/matrix pixel chain.
 *
 * Buffers are RGBA8. Brightness and contrast are the selected 256-byte native
 * rows. Hue maps are 222 float32 values; gamma is 4096 RGBA32F entries and the
 * matrices are 18 float32 values.
 */
typedef struct Pf90004Uniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;
  uint32_t mask_row_bytes;
  uint32_t parsing_row_bytes;
  uint32_t face_row_bytes;
  uint32_t use_brightness;
  uint32_t use_contrast;
} Pf90004Uniforms;

#ifdef __cplusplus
static_assert(sizeof(Pf90004Uniforms) == 32,
              "Pf90004Uniforms must stay 32 bytes");
#endif

#endif
