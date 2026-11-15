#ifndef PIXCAKE_SKIN_MODELS_8069_8070_PUBLIC_H
#define PIXCAKE_SKIN_MODELS_8069_8070_PUBLIC_H

#include <stdint.h>

/*
 * PF8069 negative FaceSurfaceBlur branch.
 *
 * `filter` runs one quantised directional pass. `vertical` selects the second
 * of the CPU's H/V pairs; surface and gauss use separate entries because their
 * kernels differ. `blend` consumes the gauss result's alpha and writes the
 * recovered type10 mix. All masks are R8, one byte per full-frame pixel.
 */
typedef struct SkinModelsFilterUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;       /* bytes per RGBA8 row, >= width * 4 */
  uint32_t vertical;
  float step_u;
  float step_v;
  float hk;              /* surface threshold, 10.45 / 255 */
  uint32_t reserved;
} SkinModelsFilterUniforms;

typedef struct SkinModelsBlendUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;       /* bytes per RGBA8 row */
  uint32_t mask_row_bytes;
  float degree;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
} SkinModelsBlendUniforms;

/*
 * PF8070 type136 SharpnessWithMask over one ROI. `source_rgba` is full-frame;
 * `base_rgba`, `blur_rgba` and `mask_r` are ROI-sized. The host seeds the
 * output from the source, so outside the ROI and gated pixels keep source bytes.
 */
typedef struct SkinModelsClarityUniforms {
  uint32_t source_width;
  uint32_t stride;       /* bytes per full-frame RGBA8 row */
  uint32_t roi_x;
  uint32_t roi_y;
  uint32_t roi_w;
  uint32_t roi_h;
  uint32_t base_width;
  uint32_t base_height;
  float degree;
  float threshold;
  float factor;
  uint32_t reserved;
} SkinModelsClarityUniforms;

#ifdef __cplusplus
static_assert(sizeof(SkinModelsFilterUniforms) == 32,
              "SkinModelsFilterUniforms must stay 32 bytes");
static_assert(sizeof(SkinModelsBlendUniforms) == 32,
              "SkinModelsBlendUniforms must stay 32 bytes");
static_assert(sizeof(SkinModelsClarityUniforms) == 48,
              "SkinModelsClarityUniforms must stay 48 bytes");
#endif

#endif
