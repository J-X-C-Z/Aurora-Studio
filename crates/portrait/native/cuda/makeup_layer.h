#ifndef PIXCAKE_MAKEUP_LAYER_PUBLIC_H
#define PIXCAKE_MAKEUP_LAYER_PUBLIC_H

#include <stdint.h>

/*
 * `BlendImageLayerWithMask` pixel law, one thread per source pixel.
 *
 *   factor = clamp01(material.a * mix_alpha * mask_weight)
 *   factor <= 0.01            -> copy the source pixel
 *   out.rgb = byte_even(mix)  (mix depends on blend_alpha / blend mode)
 *   out.a   = source alpha, or byte_even(factor * source alpha) with BLEND_ALPHA
 *
 * Source, material and owner mask may have different sizes, so each carries its
 * own width/height; the two UV streams are per source pixel.
 */
typedef struct MakeupLayerUniforms {
  uint32_t width;         /* source */
  uint32_t height;
  uint32_t stride;        /* source bytes per row */
  uint32_t material_width;
  uint32_t material_height;
  uint32_t mask_width;
  uint32_t mask_height;
  float mix_alpha;
  uint32_t mode;          /* 0 normal, 1 multiply, 2 softlight, 3 pinlight */
  uint32_t mask_channel;  /* MaskChannel order, 0..14 */
  uint32_t blend_alpha;   /* 0 or 1 */
  uint32_t reserved0;
} MakeupLayerUniforms;

#ifdef __cplusplus
static_assert(sizeof(MakeupLayerUniforms) == 48, "MakeupLayerUniforms must stay 48 bytes");
#endif

#endif
