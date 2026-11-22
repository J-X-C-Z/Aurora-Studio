#ifndef PIXCAKE_EYE_BAG_PUBLIC_H
#define PIXCAKE_EYE_BAG_PUBLIC_H

#include <stdint.h>

/*
 * PF90116 eye-bag composite, one thread per pixel.
 *
 *   q = saturate_u8(round(255 * response)) (nearest, ties to even, like cvRound)
 *   m = q * (1/255) * degree
 *   out[c] = clamp(trunc(color[c] * m + src[c] * (1 - m)), 0, 255)
 *   m == 0 or NaN( m ) -> the source pixel is copied
 *   alpha is copied from the source
 */
typedef struct EyeBagUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;   /* bytes per source row, >= width * 4 */
  float degree;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  uint32_t reserved3;
} EyeBagUniforms;

#ifdef __cplusplus
static_assert(sizeof(EyeBagUniforms) == 32, "EyeBagUniforms must stay 32 bytes");
#endif

#endif
