#ifndef AURORA_V2FUSEOFFSET_H
#define AURORA_V2FUSEOFFSET_H

#include <stdint.h>

/*
 * PF8025/PF8026 V2FuseOffset stage.
 *
 * Inputs are the same-job 255x171 Grid_2005 RGBA8 texture and either the
 * previous 280x280 encoded RGBA32F target or a zero-filled first target.
 * The output is a 280x280 RGBA32F target in the layout consumed by V2Warp.
 */

#define V2FUSE_WIDTH 280
#define V2FUSE_HEIGHT 280
#define V2FUSE_GRID_WIDTH 255
#define V2FUSE_GRID_HEIGHT 171
#define V2FUSE_RECT_X 12
#define V2FUSE_RECT_Y 86
#define V2FUSE_RECT_WIDTH 255
#define V2FUSE_RECT_HEIGHT 171

typedef struct V2FuseUniforms {
  float left_degree;
  float right_degree;
} V2FuseUniforms;

#if defined(__cplusplus)
static_assert(sizeof(V2FuseUniforms) == 8, "V2FuseUniforms ABI changed");
#endif

#endif /* AURORA_V2FUSEOFFSET_H */
