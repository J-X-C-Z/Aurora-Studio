#ifndef PIXCAKE_OIL_SHINE_BLEND_H
#define PIXCAKE_OIL_SHINE_BLEND_H

#include <stdint.h>

typedef struct OilShineBlendUniforms {
  int32_t width;
  int32_t height;
  float degree;
  int32_t reserved;
} OilShineBlendUniforms;

#define OIL_SHINE_BLEND_UNIFORM_BYTES 16u

#if defined(__cplusplus)
static_assert(sizeof(OilShineBlendUniforms) == OIL_SHINE_BLEND_UNIFORM_BYTES,
              "OilShineBlendUniforms ABI changed");
#endif

#endif
