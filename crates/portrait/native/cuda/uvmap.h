#ifndef PIXCAKE_UVMAP_PUBLIC_H
#define PIXCAKE_UVMAP_PUBLIC_H

#include <stdint.h>

/*
 * PF407 type-143 UV-map apply, one thread per pixel.
 *
 * `field` is four floats per pixel (`.xy` RG pair, `.zw` BA pair), not bytes:
 * the recovered displacement is signed and negative offsets must survive
 * transport, so nothing is packed into u8 on the way in.
 *
 * channel 0 reads field.rg, channel 1 reads field.ba.
 * catmull 0 uses the 3-arg texture tap, 1 uses the 9-tap Catmull-Rom sample
 * with `catmull_w`/`catmull_h` as the shader's u_Width/u_Height.
 */
typedef struct UvmapUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;   /* bytes per source row, >= width * 4 */
  uint32_t channel;  /* 0 = RG, 1 = BA */
  uint32_t catmull;  /* 0 = bilinear, 1 = Catmull-Rom */
  float catmull_w;
  float catmull_h;
  uint32_t reserved0;
} UvmapUniforms;

#ifdef __cplusplus
static_assert(sizeof(UvmapUniforms) == 32, "UvmapUniforms must stay 32 bytes");
#endif

#endif
