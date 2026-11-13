#ifndef PIXCAKE_GLOSS_LUT_PUBLIC_H
#define PIXCAKE_GLOSS_LUT_PUBLIC_H

#include <stdint.h>

/*
 * PF8068 SKIN_GLOSS one pass of the three-pass chain.
 *
 * Buffers are RGBA8 like `aurora_pixcake::Image`; the kernel only reads and
 * writes r/g/b and copies alpha through untouched. `mask_r` and `extra_mask_r`
 * are one byte per pixel; their product (each /255) is the blend weight.
 *
 * The explicit `lut` pointer supplies 256 caller-owned bytes.
 * `pass` identifies the stage: 0 positive1, 1 positive2,
 * 2 negative. `degree` is the already-normalised positive/negative value.
 */
typedef struct GlossUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;         /* bytes per source row, >= width * 4 */
  uint32_t mask_row_bytes; /* bytes per mask row, >= width */
  uint32_t pass;           /* 0, 1 or 2 */
  float degree;            /* [0,1]; the host skips a pass when it is 0 */
  uint32_t reserved0;
  uint32_t reserved1;
} GlossUniforms;

#ifdef __cplusplus
static_assert(sizeof(GlossUniforms) == 32, "GlossUniforms must stay 32 bytes");
#endif

#endif
