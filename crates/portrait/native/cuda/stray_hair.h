#ifndef PIXCAKE_STRAY_HAIR_PUBLIC_H
#define PIXCAKE_STRAY_HAIR_PUBLIC_H

#include <stdint.h>

/*
 * PF90088 stray-hair final composite. Both same-job producers are required:
 * HAIRFLY_V1 (RGB repair image) and RAW_STRAY_HAIR_MASK. Person and background
 * masks are optional area gates; absent gates behave as all-area 1.0.
 */
typedef struct StrayHairUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;
  float degree;
  float person_area;
  float background_area;
  int32_t has_person;
  int32_t has_background;
} StrayHairUniforms;

#if defined(__cplusplus)
static_assert(sizeof(StrayHairUniforms) == 32,
              "StrayHairUniforms must stay 32 bytes");
#endif

#endif
