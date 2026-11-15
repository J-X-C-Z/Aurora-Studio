#ifndef PIXCAKE_FOOT_BLUE_VEIN_PUBLIC_H
#define PIXCAKE_FOOT_BLUE_VEIN_PUBLIC_H

#include <stdint.h>

/*
 * PF44297 foot blue-vein final composite.
 *
 * The host supplies a tight W*H feathered mask and the same-job model repair
 * image. The kernel writes only RGB inside the ROI and always preserves the
 * source alpha. The current Rust adapter accepts only the full-frame ROI;
 * local ROI/affine remains unresolved rather than inferred here.
 *
 * weight = (feathered_mask/255) * degree
 * out.rgb = round_away_from_zero(src.rgb*(1-weight) + repair.rgb*weight)
 */
typedef struct FootBlueVeinUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;
  float degree;
  uint32_t roi_x;
  uint32_t roi_y;
  uint32_t roi_w;
  uint32_t roi_h;
} FootBlueVeinUniforms;

#ifdef __cplusplus
static_assert(sizeof(FootBlueVeinUniforms) == 32,
              "FootBlueVeinUniforms must stay 32 bytes");
#endif

#endif
