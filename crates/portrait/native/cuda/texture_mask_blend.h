#ifndef PIXCAKE_TEXTURE_MASK_BLEND_H
#define PIXCAKE_TEXTURE_MASK_BLEND_H

#include <stdint.h>

/*
 * PF30009 `texture_mask_blend` uniform block. Kernel arguments follow the
 * house `run_u8` shape: original, effect, mask, effect_uv, out, uniforms.
 */

typedef struct TextureMaskBlendUniforms {
  int32_t width;
  int32_t height;
  int32_t effect_width;
  int32_t effect_height;
  float degree;
  int32_t reverse_alpha;
  int32_t has_uv;
  int32_t reserved;
} TextureMaskBlendUniforms;

#define TEXTURE_MASK_BLEND_UNIFORM_BYTES 32u

#if defined(__cplusplus)
static_assert(sizeof(TextureMaskBlendUniforms) == TEXTURE_MASK_BLEND_UNIFORM_BYTES,
              "TextureMaskBlendUniforms ABI changed");
#endif

#endif /* PIXCAKE_TEXTURE_MASK_BLEND_H */
