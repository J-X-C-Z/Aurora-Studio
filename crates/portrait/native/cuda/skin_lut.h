#ifndef PIXCAKE_SKIN_LUT_PUBLIC_H
#define PIXCAKE_SKIN_LUT_PUBLIC_H

#include <stdint.h>

/*
 * PF90005 SKIN_WHITEN / PF90007 SKIN_ROSY SkinAdjustV2 stage.
 *
 * Buffers are RGBA8 like `aurora_pixcake::Image`: r at +0, g at +1, b at +2,
 * a at +3. The recovered Metal reference reads a BGRA texture; this port keeps
 * the Rust oracle's RGBA packing, which is the accepted lane.
 *
 * `lut_rgb512` is the packed 512x512x3 atlas (8x8 tiles of 64x64), not a
 * texture array.
 */
typedef struct SkinLutUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;        /* bytes per source row, >= width * 4 */
  uint32_t mask_row_bytes;/* bytes per mask row, >= width */
  float mix;              /* uniform degree, [0,1] */
  float gate;             /* must be exactly 0.001 */
  uint32_t lut_width;     /* must be 512 */
  uint32_t lut_height;    /* must be 512 */
} SkinLutUniforms;

#ifdef __cplusplus
static_assert(sizeof(SkinLutUniforms) == 32, "SkinLutUniforms must stay 32 bytes");
#endif

#endif
