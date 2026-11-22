#ifndef AURORA_PF90021_VOLUME_LUT_PUBLIC_H
#define AURORA_PF90021_VOLUME_LUT_PUBLIC_H

#include <stdint.h>

typedef struct Pf90021LutPassParams {
  uint32_t width;
  uint32_t height;
  uint32_t source_stride;
  uint32_t output_stride;
} Pf90021LutPassParams;

typedef struct Pf90021CompositeParams {
  uint32_t width;
  uint32_t height;
  uint32_t source_stride;
  uint32_t effect_stride;
  uint32_t mask_stride;
  uint32_t output_stride;
  float u_alpha;
  uint32_t reserved;
} Pf90021CompositeParams;

#ifdef __cplusplus
static_assert(sizeof(Pf90021LutPassParams) == 16,
              "Pf90021LutPassParams must stay 16 bytes");
static_assert(sizeof(Pf90021CompositeParams) == 32,
              "Pf90021CompositeParams must stay 32 bytes");
#endif

#endif
