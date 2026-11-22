#ifndef PIXCAKE_NEUTRAL_GRAY_H
#define PIXCAKE_NEUTRAL_GRAY_H

#include <stdint.h>

/* PF30013 neutral-gray flattening. */
typedef struct SkinGreyUniforms {
  int32_t width;
  int32_t height;
  float alpha;
} SkinGreyUniforms;

#define SKIN_GREY_UNIFORM_BYTES 12u

/* PF30014 neutral-gray 3D. */
typedef struct NeutralGray3dUniforms {
  int32_t active;
} NeutralGray3dUniforms;

#define NEUTRAL_GRAY_3D_UNIFORM_BYTES 4u

#if defined(__cplusplus)
static_assert(sizeof(SkinGreyUniforms) == SKIN_GREY_UNIFORM_BYTES,
              "SkinGreyUniforms ABI changed");
static_assert(sizeof(NeutralGray3dUniforms) == NEUTRAL_GRAY_3D_UNIFORM_BYTES,
              "NeutralGray3dUniforms ABI changed");
#endif

#endif /* PIXCAKE_NEUTRAL_GRAY_H */
