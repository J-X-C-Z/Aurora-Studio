#ifndef PIXCAKE_TEETH_LUT_PUBLIC_H
#define PIXCAKE_TEETH_LUT_PUBLIC_H

#include <stdint.h>

/*
 * PF108-PF110 teeth whitening: one pass that runs whiten then bright.
 *
 * `source_rgba` / `out_rgba` are RGBA8 like `aurora_pixcake::Image`; alpha is
 * copied. `mask_g` is one byte per pixel (mask id2.G). Both atlases are packed
 * 512x512x3 (786,432 bytes).
 *
 * `dz` = common*dz, `ea` = common*bright, already normalised to [0,1]; the
 * per-pixel weights are `store_unorm8(mask*dz)/255` and `store_unorm8(mask*ea)/255`.
 */
typedef struct TeethUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;         /* bytes per source row, >= width * 4 */
  uint32_t mask_row_bytes; /* bytes per mask row, >= width */
  float dz;
  float ea;
  uint32_t reserved0;
  uint32_t reserved1;
} TeethUniforms;

#ifdef __cplusplus
static_assert(sizeof(TeethUniforms) == 32, "TeethUniforms must stay 32 bytes");
#endif

#endif
