#ifndef PIXCAKE_MOISTURE_PUBLIC_H
#define PIXCAKE_MOISTURE_PUBLIC_H

#include <stdint.h>

/*
 * PF90002 (face) / PF90010 (body) moisturizing: one thread per pixel.
 *
 * Face inputs: source RGBA8, mask RGBA8, blur Y (u8, one per pixel), skin map
 * (f32 width*height, sampled as x=blue, y=red), hardlight LUT and beauty LUT
 * (f32, GL LINEAR + CLAMP_TO_EDGE). `channel` is the MaskChannel index.
 *
 * Body inputs: source RGBA8, weight_skin and weight_dog (f32 per pixel), beauty
 * LUT, and a degree already gated by the caller.
 */
typedef struct MoistureFaceUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;
  uint32_t channel;     /* 0..14, MaskChannel order */
  float degree;
  uint32_t skin_width;
  uint32_t skin_height;
  uint32_t hardlight_len;
  uint32_t beauty_len;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
} MoistureFaceUniforms;

typedef struct MoistureBodyUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;
  uint32_t beauty_len;
  float degree;
  uint32_t roi_x;
  uint32_t roi_y;
  uint32_t roi_w;
  uint32_t roi_h;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
} MoistureBodyUniforms;

#ifdef __cplusplus
static_assert(sizeof(MoistureFaceUniforms) == 48, "MoistureFaceUniforms must stay 48 bytes");
static_assert(sizeof(MoistureBodyUniforms) == 48, "MoistureBodyUniforms must stay 48 bytes");
#endif

#endif
