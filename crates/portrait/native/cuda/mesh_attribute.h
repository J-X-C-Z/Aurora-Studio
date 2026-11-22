#ifndef PIXCAKE_MESH_ATTRIBUTE_PUBLIC_H
#define PIXCAKE_MESH_ATTRIBUTE_PUBLIC_H

#include <stdint.h>

/*
 * PF8033 attribute rasterizer, two passes:
 *   select   one thread per triangle: bbox scan, record max triangle index and
 *            the covering count per pixel
 *   resolve  one thread per pixel: re-evaluate the winning triangle exactly
 *
 * Last-writer order is reproduced by "highest triangle index wins", which is
 * what the CPU's sequential overwrite means.
 */
typedef struct MeshAttributeUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t vertex_count;
  uint32_t triangle_count;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  uint32_t reserved3;
} MeshAttributeUniforms;

#ifdef __cplusplus
static_assert(sizeof(MeshAttributeUniforms) == 32, "MeshAttributeUniforms must stay 32 bytes");
#endif

#endif
