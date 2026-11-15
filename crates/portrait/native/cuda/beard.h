#ifndef PIXCAKE_BEARD_PUBLIC_H
#define PIXCAKE_BEARD_PUBLIC_H

#include <stdint.h>

/*
 * PF21026 stubble removal: w = min(mask_a, mask_b), byte lerp of the repair
 * colour, nearest-even rounding, clamp 0..255, alpha copied from the source.
 * `strength` is the already-scaled caller value; the 0.75 small-frame rule is
 * applied on the host before launch.
 */
typedef struct BeardUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;
  float strength;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  uint32_t reserved3;
} BeardUniforms;

#ifdef __cplusplus
static_assert(sizeof(BeardUniforms) == 32, "BeardUniforms must stay 32 bytes");
#endif

#endif
