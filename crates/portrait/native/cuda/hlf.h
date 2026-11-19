#ifndef PIXCAKE_HLF_PUBLIC_H
#define PIXCAKE_HLF_PUBLIC_H

#include <stdint.h>

/*
 * PF30021 consumes caller-supplied full-frame W*H*3 float planes. The vendor
 * fragment evaluates every pixel; ROI is validated by the host but does not
 * crop this pass.
 */
typedef struct HlfFaceUniforms {
  uint32_t width;
  uint32_t height;
  float degree;
  uint32_t reserved;
} HlfFaceUniforms;

/*
 * PF30022 WithRoi. Median, gauss, and mask planes are full-frame. The adjust
 * LUT is optional at the ABI level and required when spotless_degree > 0:
 * one L8 row, clamp-to-edge linear x sampling, fixed logical y=0.5.
 * The kernel writes RGB only inside roi and preserves source alpha.
 */
typedef struct HlfBodyUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t roi_x;
  uint32_t roi_y;
  uint32_t roi_w;
  uint32_t roi_h;
  float degree;
  float retouch_degree;
  float retouch_high_freq_factor;
  float diff_min_range;
  float diff_max_range;
  uint32_t lut_width;
  uint32_t lut_height;
  float spotless_degree;
  float spotless_acne_threshold;
  float spotless_high_freq_factor;
  float spotless_low_freq_alpha;
  uint32_t reserved;
} HlfBodyUniforms;

/*
 * Candidate HLF preprocessing. These values are not vendor_equivalent:
 * the vendor interval values and sampler border state were not recovered.
 * Coordinates use clamp-to-edge; the Gaussian is the recovered fixed 9-tap
 * kernel applied as a horizontal then vertical sample-domain convolution.
 */
typedef struct HlfPreprocessUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t interval;
  uint32_t reserved;
} HlfPreprocessUniforms;

#ifdef __cplusplus
static_assert(sizeof(HlfFaceUniforms) == 16,
              "HlfFaceUniforms must stay 16 bytes");
static_assert(sizeof(HlfBodyUniforms) == 72,
              "HlfBodyUniforms must stay 72 bytes");
static_assert(sizeof(HlfPreprocessUniforms) == 16,
              "HlfPreprocessUniforms must stay 16 bytes");
#endif

#endif
