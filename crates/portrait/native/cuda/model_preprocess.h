/*
 * WL46 model preprocessing: the three conversions every ONNX head repeats.
 *
 *   rgba8_to_nchw   RGBA8 (stride-padded, ROI, bilinear resize) -> NCHW f32,
 *                   per-sample divide, then (value - mean) / std_dev
 *   nhwc4_to_planar NHWC f32 (stride-padded) -> `channels` tight f32 planes
 *   f32_to_u8       interleaved f32 (stride-padded) -> tight u8 with the
 *                   Rust `byte_away` byte law
 *
 * Each kernel mirrors `crates/portrait/src/cuda_model_preprocess.rs`, which is
 * the CPU reference. Op order is pinned by -fmad=false plus the explicit
 * binary32 intrinsics, so CPU and GPU agree bit-for-bit.
 */
#ifndef AURORA_MODEL_PREPROCESS_PUBLIC_H
#define AURORA_MODEL_PREPROCESS_PUBLIC_H

#include <stdint.h>

/* RGBA8 -> NCHW f32. Samples are byte / divisor before the bilinear lerp,
   matching the CPU reference; mean/std_dev index the output channel. */
typedef struct MppNchwUniforms {
  uint32_t src_width;
  uint32_t src_height;
  uint32_t src_stride; /* pixels per source row, >= src_width */
  uint32_t roi_x;
  uint32_t roi_y;
  uint32_t roi_w;
  uint32_t roi_h;
  uint32_t out_w;
  uint32_t out_h;
  uint32_t channels;  /* 3 or 4 */
  float divisor;      /* per-sample divide, normally 255.0 */
  float mean[4];
  float std_dev[4];
  uint32_t swap_rb;   /* non-zero reads bytes b,g,r for channels 0..2 */
} MppNchwUniforms;

/* NHWC f32 -> planar f32. Output plane p starts at p * width * height. */
typedef struct MppPlanarUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride; /* pixels per source row, >= width */
  uint32_t channels; /* 1..4 */
} MppPlanarUniforms;

/* Interleaved f32 -> interleaved u8 through value * scale + offset. */
typedef struct MppQuantUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride; /* pixels per source row, >= width */
  uint32_t channels; /* 1..4 */
  float scale;
  float offset;
  uint32_t pad0;
  uint32_t pad1;
} MppQuantUniforms;

#ifdef __cplusplus
static_assert(sizeof(MppNchwUniforms) == 80, "MppNchwUniforms must stay 80 bytes");
static_assert(sizeof(MppPlanarUniforms) == 16, "MppPlanarUniforms must stay 16 bytes");
static_assert(sizeof(MppQuantUniforms) == 32, "MppQuantUniforms must stay 32 bytes");
#endif

#endif
