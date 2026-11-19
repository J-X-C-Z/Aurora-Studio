#ifndef PIXCAKE_HAIR_SEAM_BLEND_H
#define PIXCAKE_HAIR_SEAM_BLEND_H

#include <stdint.h>

/*
 * PF90087 final blend. The CPU producer supplies a tight RGB8 repair plane and
 * a compacted U8 coverage mask; out-of-ROI and uncovered pixels keep source.
 */
typedef struct HairSeamBlendUniforms {
  uint32_t width;
  uint32_t height;
  float degree;
  uint32_t roi_x;
  uint32_t roi_y;
  uint32_t roi_w;
  uint32_t roi_h;
  uint32_t reserved;
} HairSeamBlendUniforms;

#define HAIR_SEAM_BLEND_UNIFORM_BYTES 32u

#if defined(__cplusplus)
static_assert(sizeof(HairSeamBlendUniforms) == HAIR_SEAM_BLEND_UNIFORM_BYTES,
              "HairSeamBlendUniforms ABI changed");
#endif

#endif /* PIXCAKE_HAIR_SEAM_BLEND_H */
