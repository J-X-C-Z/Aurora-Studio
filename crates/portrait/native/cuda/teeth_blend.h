#ifndef PIXCAKE_TEETH_BLEND_PUBLIC_H
#define PIXCAKE_TEETH_BLEND_PUBLIC_H

#include <stdint.h>

/*
 * PF30026 final mask blend, one thread per pixel.
 *
 *   m = mask_u8 * 0x3B807BBC
 *   out = trunc(m*effect + original*(1-m))   // cvtt
 *   mask == 255 copies the effect RGB
 *   alpha is always the original
 */
typedef struct TeethBlendUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  uint32_t reserved3;
  uint32_t reserved4;
} TeethBlendUniforms;

#ifdef __cplusplus
static_assert(sizeof(TeethBlendUniforms) == 32, "TeethBlendUniforms must stay 32 bytes");
#endif

#endif
