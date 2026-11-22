#ifndef AURORA_HAIR_DETAIL_H
#define AURORA_HAIR_DETAIL_H

#include <stdint.h>

typedef struct HairSmoothUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t roi_x;
  uint32_t roi_y;
  uint32_t roi_width;
  uint32_t roi_height;
  uint32_t source_stride;
  uint32_t hair_stride;
  uint32_t mix_stride;
  uint32_t blend_stride;
  uint32_t detail_stride;
  float mix_strength;
  float detail_strength;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
} HairSmoothUniforms;

typedef struct HairGlossUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t roi_x;
  uint32_t roi_y;
  uint32_t roi_width;
  uint32_t roi_height;
  uint32_t source_stride;
  uint32_t hair_stride;
  uint32_t detail_stride;
  float scale;
  uint32_t reserved0;
  uint32_t reserved1;
} HairGlossUniforms;

#ifdef __cplusplus
static_assert(sizeof(HairSmoothUniforms) == 64, "HairSmoothUniforms ABI");
static_assert(sizeof(HairGlossUniforms) == 48, "HairGlossUniforms ABI");
#endif

#endif
