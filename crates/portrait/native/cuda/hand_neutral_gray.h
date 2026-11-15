#ifndef PIXCAKE_HAND_NEUTRAL_GRAY_H
#define PIXCAKE_HAND_NEUTRAL_GRAY_H

#include <stdint.h>

typedef struct HandNeutralGrayUniforms {
  int32_t width;
  int32_t height;
  float degree;
  int32_t target_kind;
} HandNeutralGrayUniforms;

#define HAND_NEUTRAL_GRAY_UNIFORM_BYTES 16u

#if defined(__cplusplus)
static_assert(sizeof(HandNeutralGrayUniforms) == HAND_NEUTRAL_GRAY_UNIFORM_BYTES,
              "HandNeutralGrayUniforms ABI changed");
#endif

#endif
