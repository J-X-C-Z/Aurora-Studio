#ifndef AURORA_MESH_RASTER_H
#define AURORA_MESH_RASTER_H

#include <stddef.h>
#include <stdint.h>

/*
 * O3b CUDA mesh rasterizer ABI.
 *
 * Geometry contract, identical to crates/pixcake/src/raster.rs:
 * - dest: destination geometry in framebuffer pixels, origin top-left,
 *   x right, y down. Pixel (x, y) is sampled at (x + 0.5, y + 0.5).
 * - source_uv: normalized [0, 1]; row 0 of the source image is v = 0.
 * - ebo: uint16 triangle list in original index order; that order resolves
 *   overlap.
 * - area2 = cross(p1 - p0, p2 - p0); |area2| <= 1e-12 is degenerate.
 * - CULL_FRONT discards front-facing triangles (same value meaning as the
 *   captured GL state: glCullFace(GL_FRONT) -> CULL_FRONT).
 * - Alpha is straight, sampled and quantized like RGB, never blended.
 * - Uncovered pixels keep the bytes already in `out`, which the caller
 *   prefills with the source image.
 *
 * This header is independent of the historical C1 tile contract; it does not
 * replace or modify v2warp_tile_rasterizer_contract.h.
 */

#define MESH_RASTER_ABI_VERSION 1u

typedef enum MeshRasterStatus {
  MESH_RASTER_OK = 0,
  MESH_RASTER_INVALID_ARGUMENT = 1,
  MESH_RASTER_RULES_REQUIRED = 2,
  MESH_RASTER_OUT_OF_RANGE = 3,
  MESH_RASTER_CUDA_ERROR = 4
} MeshRasterStatus;

typedef enum MeshRasterWinding {
  MESH_RASTER_WINDING_POSITIVE_IS_FRONT = 1,
  MESH_RASTER_WINDING_NEGATIVE_IS_FRONT = 2
} MeshRasterWinding;

typedef enum MeshRasterCull {
  MESH_RASTER_CULL_NONE = 1,
  MESH_RASTER_CULL_FRONT = 2,
  MESH_RASTER_CULL_BACK = 3
} MeshRasterCull;

typedef enum MeshRasterEdge {
  MESH_RASTER_EDGE_EXCLUSIVE = 1,
  MESH_RASTER_EDGE_INCLUSIVE = 2
} MeshRasterEdge;

typedef enum MeshRasterOverlap {
  MESH_RASTER_OVERLAP_REJECT = 1,
  MESH_RASTER_OVERLAP_FIRST = 2,
  MESH_RASTER_OVERLAP_LAST = 3
} MeshRasterOverlap;

typedef enum MeshRasterWrap {
  MESH_RASTER_WRAP_CLAMP = 0,
  MESH_RASTER_WRAP_REPEAT = 1
} MeshRasterWrap;

/* 16 bytes; every field is explicit, there are no defaults. */
typedef struct MeshRasterRules {
  int32_t winding;
  int32_t cull;
  int32_t edge;
  int32_t overlap;
} MeshRasterRules;

/*
 * Caller-owned per-pixel scratch, width * height entries each:
 *   selected  uint32_t, initialized by the launcher, 0xFFFFFFFF = uncovered
 *   coverage  uint32_t, initialized by the launcher
 * Allocate with the byte counts below; one allocation of both is fine.
 */
#define MESH_RASTER_SELECTED_BYTES(width, height) \
  ((size_t)(width) * (size_t)(height) * sizeof(uint32_t))
#define MESH_RASTER_COVERAGE_BYTES(width, height) \
  ((size_t)(width) * (size_t)(height) * sizeof(uint32_t))

typedef struct MeshRasterScratch {
  uint32_t* selected;
  uint32_t* coverage;
} MeshRasterScratch;

typedef struct MeshRasterInput {
  /* vertex_count interleaved float2 values. */
  const float* dest;
  /* vertex_count interleaved float2 values. */
  const float* source_uv;
  /* triangle_count * 3 uint16 indices. */
  const uint16_t* ebo;
  /* width * height * 4 straight-alpha RGBA8 source bytes. */
  const uint8_t* source;
  /* width * height * 4 RGBA8 destination, prefilled with the source image. */
  uint8_t* out;
  int32_t width;
  int32_t height;
  int32_t vertex_count;
  int32_t triangle_count;
  int32_t wrap;
  MeshRasterRules rules;
} MeshRasterInput;

#if defined(__cplusplus)
static_assert(sizeof(MeshRasterRules) == 16, "MeshRasterRules ABI changed");
extern "C" {
#endif

/*
 * Validates the descriptor and launches two kernels on `stream`
 * (nullptr = default stream). Nothing is allocated; scratch and out are
 * caller owned. The call synchronizes to report launch errors.
 *
 * Finite vertex values are the caller's contract (the Rust layer refuses
 * them); non-finite triangles are dropped per triangle rather than sampled.
 */
MeshRasterStatus mesh_raster_run(
    const MeshRasterInput* input,
    const MeshRasterScratch* scratch,
    void* stream);

const char* mesh_raster_status_name(MeshRasterStatus status);

#if defined(__cplusplus)
}
#endif

#endif /* AURORA_MESH_RASTER_H */
