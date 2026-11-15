#ifndef AURORA_MESH_FRAGMENT_H
#define AURORA_MESH_FRAGMENT_H

#include <stdint.h>

/*
 * O3d fragment ABI. Mirrors crates/pixcake/src/raster_fragment.rs exactly:
 * same parameters, same evaluation order, same quantization.
 *
 * One thread per point. The caller supplies the source RGBA8 image plus one
 * (src_uv, dst_uv) pair per point; the kernel writes one RGBA8 per point.
 * Geometry (mesh, EBO, barycentrics, coverage) is out of scope here.
 */

#define MESH_FRAGMENT_ABI_VERSION 1u

typedef enum MeshFragmentStatus {
  MESH_FRAGMENT_OK = 0,
  MESH_FRAGMENT_INVALID_ARGUMENT = 1,
  MESH_FRAGMENT_OUT_OF_RANGE = 2,
  MESH_FRAGMENT_CUDA_ERROR = 3
} MeshFragmentStatus;

/* Which smoothcoord() the pass uses. */
typedef enum MeshFragmentEdge {
  MESH_FRAGMENT_EDGE_RESHAPE_NORMAL = 0,
  MESH_FRAGMENT_EDGE_ADJ_CLAMP = 1
} MeshFragmentEdge;

/* i_ / jb / ic. A negative factor forces smoothcoord() == 1. */
typedef struct MeshFragmentClamp {
  float factor;
  int32_t protect_x;
  int32_t protect_y;
} MeshFragmentClamp;

/* The four define keys from 0x611B460, as explicit 0/1 flags. */
typedef struct MeshFragmentDefines {
  int32_t clamp_edge;
  int32_t uv_map;
  int32_t uv_map_ba;
  int32_t catmull_rom;
} MeshFragmentDefines;

typedef struct MeshFragmentInput {
  /* width * height * 4 straight-alpha RGBA8. */
  const uint8_t* source;
  /* count * 4 RGBA8 output. */
  uint8_t* out;
  /* count interleaved float2, normalized. */
  const float* src_uv;
  /* count interleaved float2, normalized. */
  const float* dst_uv;
  int32_t width;
  int32_t height;
  int32_t count;
  int32_t edge;
  MeshFragmentClamp clamp;
  MeshFragmentDefines defines;
  /* Shader u_Width / u_Height for Catmull-Rom; 0 means "use the image size". */
  float catmull_width;
  float catmull_height;
} MeshFragmentInput;

#if defined(__cplusplus)
static_assert(sizeof(MeshFragmentClamp) == 12, "MeshFragmentClamp ABI changed");
static_assert(sizeof(MeshFragmentDefines) == 16, "MeshFragmentDefines ABI changed");
static_assert(sizeof(MeshFragmentInput) == 88, "MeshFragmentInput ABI changed");
extern "C" {
#endif

/*
 * Validates the descriptor, launches one kernel on `stream` (nullptr = default
 * stream) and synchronizes to report launch errors. No allocation.
 */
MeshFragmentStatus mesh_fragment_run(const MeshFragmentInput* input, void* stream);

const char* mesh_fragment_status_name(MeshFragmentStatus status);

#if defined(__cplusplus)
}
#endif

#endif /* AURORA_MESH_FRAGMENT_H */
