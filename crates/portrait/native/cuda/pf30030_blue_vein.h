#ifndef PIXCAKE_PF30030_BLUE_VEIN_PUBLIC_H
#define PIXCAKE_PF30030_BLUE_VEIN_PUBLIC_H

#include <stdint.h>

/*
 * PF30030 hand blue-vein correction, one thread per ROI pixel.
 *
 * The caller supplies the already-authored correction RGB image and the
 * feathered hand-vein mask. The output is seeded from the source, this kernel
 * writes RGB only inside the ROI, and alpha is never written.
 *
 *   delta = correction_rgb - 80
 *   scaled = delta * mask / 255, rounded away from zero
 *   out.rgb = clamp(source.rgb + scaled, 0, 255)
 */
typedef struct Pf30030BlueVeinUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;
  uint32_t roi_x;
  uint32_t roi_y;
  uint32_t roi_width;
  uint32_t roi_height;
  uint32_t reserved;
} Pf30030BlueVeinUniforms;

#ifdef __cplusplus
static_assert(sizeof(Pf30030BlueVeinUniforms) == 32,
              "Pf30030BlueVeinUniforms must stay 32 bytes");
#endif

#endif
