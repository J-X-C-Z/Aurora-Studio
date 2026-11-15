#ifndef AURORA_V2WARP_H
#define AURORA_V2WARP_H

#include <stdint.h>

/*
 * PF8025/PF8026 V2Warp candidate.
 *
 * The pinned V2 matrix reduces a_position to framebuffer pixels with w=1.
 * This ABI therefore takes already-projected float2 positions. Four independent
 * 526xfloat2 attributes and one 2982xuint16 EBO are required. The encoded
 * offset texture is the recovered RGBA32F pass-1 target; source is RGBA8.
 */

#define V2WARP_VERTEX_COUNT 526
#define V2WARP_INDEX_COUNT 2982
#define V2WARP_TRIANGLE_COUNT 994

typedef enum V2WarpStatus {
  V2WARP_OK = 0,
  V2WARP_INVALID_ARGUMENT = 1,
  V2WARP_RULES_REQUIRED = 2,
  V2WARP_OUT_OF_RANGE = 3,
  V2WARP_CUDA_ERROR = 4
} V2WarpStatus;

typedef enum V2WarpWinding {
  V2WARP_WINDING_POSITIVE_IS_FRONT = 1,
  V2WARP_WINDING_NEGATIVE_IS_FRONT = 2
} V2WarpWinding;

typedef enum V2WarpCull {
  V2WARP_CULL_NONE = 1,
  V2WARP_CULL_FRONT = 2,
  V2WARP_CULL_BACK = 3
} V2WarpCull;

typedef enum V2WarpEdge {
  V2WARP_EDGE_EXCLUSIVE = 1,
  V2WARP_EDGE_INCLUSIVE = 2
} V2WarpEdge;

typedef enum V2WarpOverlap {
  V2WARP_OVERLAP_REJECT = 1,
  V2WARP_OVERLAP_FIRST = 2,
  V2WARP_OVERLAP_LAST = 3
} V2WarpOverlap;

typedef struct V2WarpRules {
  int32_t winding;
  int32_t cull;
  int32_t edge;
  int32_t overlap;
} V2WarpRules;

typedef struct V2WarpUniforms {
  int32_t width;
  int32_t height;
  int32_t offset_width;
  int32_t offset_height;
  int32_t triangle_count;
  int32_t overlap;
  int32_t edge;
  float mat2_raw0;
  float mat2_raw1;
  float mat2_raw2;
  float mat2_raw3;
  float filter_width;
  float filter_height;
} V2WarpUniforms;

#if defined(__cplusplus)
static_assert(sizeof(V2WarpRules) == 16, "V2WarpRules ABI changed");
static_assert(sizeof(V2WarpUniforms) == 52, "V2WarpUniforms ABI changed");
#endif

#endif /* AURORA_V2WARP_H */
