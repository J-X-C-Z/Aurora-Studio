#ifndef PIXCAKE_SMILE_WARP_PUBLIC_H
#define PIXCAKE_SMILE_WARP_PUBLIC_H

#include <stdint.h>

/*
 * PF100007 SmileV1 pixel consumer, one thread per ROI pixel.
 *
 * Pixel space, not UV: `sx = x + weight*dx`, `sy = y + weight*dy`, bilinear in
 * byte space, truncating store, alpha copied from the source — exactly the
 * `kernels::smile_warp::pf100007_apply` arithmetic.
 */
typedef struct SmileUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;   /* bytes per source row, >= width * 4 */
  float degree;
  uint32_t roi_x;
  uint32_t roi_y;
  uint32_t roi_w;
  uint32_t roi_h;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  uint32_t reserved3;
} SmileUniforms;

#ifdef __cplusplus
static_assert(sizeof(SmileUniforms) == 48, "SmileUniforms must stay 48 bytes");
#endif

#endif
