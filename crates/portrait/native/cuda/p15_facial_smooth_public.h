#ifndef P15_FACIAL_SMOOTH_PUBLIC_H
#define P15_FACIAL_SMOOTH_PUBLIC_H

#include <stdint.h>

#define P15_STATUS_OK 0
#define P15_STATUS_INVALID_UNIFORMS 1
#define P15_STATUS_INVALID_LAYOUT 2
#define P15_STATUS_MISSING_TEXTURE 3
#define P15_STATUS_MISSING_GL_STATE 4
#define P15_STATUS_INVALID_GATE 5
#define P15_STATUS_UNSUPPORTED_ALGO 6

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Explicit 88-byte ABI. All colors are float32 RGBA [0,1]. This is a source
 * contract, not a capture of product uniform values.
 */
typedef struct P15FacialSmoothUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;
  uint32_t pf;
  float pf_value;
  float gate;
  float legal_alpha;
  float dark_circle_alpha;
  float diff_factor;
  float diff_factor2;
  float color_threshold_factor;
  float mix_alpha;
  float diff_min_threshold;
  float diff_max_threshold;
  float high_frequency_min_threshold;
  float high_frequency_max_threshold;
  float surface_radius;
  float surface_threshold;
  float gauss_offset;
  uint32_t roi_size;
  uint32_t algo_type;
  uint32_t box_blur_stage;
} P15FacialSmoothUniforms;

/*
 * Explicit device-buffer pointers and live-input flags. gl_state_captured is
 * caller-supplied evidence; the CUDA adapter cannot infer the product FBO,
 * sampler, texture identity or render target.
 */
typedef struct P15LiveInputs {
  const float* source_rgba;
  const float* mask_rgba;
  const float* skin_mask_r;
  const float* base_rgba;
  const float* residual_base_rgba;
  float* output_rgba;
  const P15FacialSmoothUniforms* uniforms;
  int32_t gl_state_captured;
} P15LiveInputs;

int32_t p15_validate_launch(const P15LiveInputs* inputs);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
static_assert(sizeof(P15FacialSmoothUniforms) == 88,
              "P15FacialSmoothUniforms must stay 88 bytes");
#endif

#endif
