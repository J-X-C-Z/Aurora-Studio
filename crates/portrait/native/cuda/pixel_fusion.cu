// `PixelStageFusion` single-pass fusion kernel (RGBA8 in, RGBA8 out).
//
// This is the CUDA twin of `compute_backend::PixelStageFusion::execute_fused_into`
// + `PixelOp::apply_pixel`. Every stage is applied to one pixel in registers, on
// the already-quantised byte values, exactly like the Rust loop; the fused pass
// therefore has to reproduce the per-stage re-quantisation, not a single
// float accumulation of all stages.
//
// Build (CUDA 13.4; same flag set as work/backend-r11/o1i/build-ptx.cmd):
//   nvcc -std=c++17 -O3 -fmad=false -gencode arch=compute_89,code=sm_89 -ptx ^
//        pixel_fusion.cu -o pixel_fusion.ptx
//
// Do NOT add --use_fast_math and do not lower `-fmad`. `sqrt` must stay
// correctly rounded (`__fsqrt_rn`) and `roundf` (ties away from zero) is the
// rounding `quantize_u8` uses.
//
// Layout of the LUT blob (`lut_blob`):
//   * every `PF_LUT1D` op points at 768 bytes: r[256], g[256], b[256];
//   * every `PF_LUT3D` op points at `size^3` RGB float triples in
//     `[r + size * (g + size * b)]` order, addressed as a float index
//     (`arg0`), 4-byte aligned because every table starts at a multiple of 4.
// All offsets are computed by `cuda_backend::encode_stages`; the kernel never
// walks the blob on its own.

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#define PIXEL_FUSION_MAX_OPS 16

#define PF_BRIGHTNESS 0
#define PF_WHITEN 1
#define PF_ROSY 2
#define PF_TEMPERATURE 3
#define PF_SATURATION 4
#define PF_LUT1D 5
#define PF_LUT3D 6

typedef struct PixelFusionOp {
  int kind;   /* PF_* */
  int arg0;   /* LUT1D: byte offset; LUT3D: float offset */
  int arg1;   /* LUT3D: cube size, unused otherwise */
  float p0;   /* clamped parameter 0 */
  float p1;   /* clamped parameter 1 */
} PixelFusionOp;

typedef struct PixelFusionUniforms {
  int width;
  int height;
  int op_count;
  int vec_load; /* pixels per thread: 1, 2 or 4 */
  PixelFusionOp ops[PIXEL_FUSION_MAX_OPS];
} PixelFusionUniforms;

#ifdef __cplusplus
static_assert(sizeof(PixelFusionOp) == 20, "PixelFusionOp must stay 20 bytes");
static_assert(sizeof(PixelFusionUniforms) == 336,
              "PixelFusionUniforms must stay 336 bytes");
#endif

// Explicit binary32 ops: no contraction, no reciprocal approximation.
__device__ __forceinline__ float pf_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float pf_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float pf_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float pf_div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float pf_sqrt(float a) { return __fsqrt_rn(a); }

// Rust `v.clamp(0.0, 1.0)`: NaN falls through `min`/`max` the same way it does
// in the reference, and the byte inputs make a NaN unreachable in practice.
__device__ __forceinline__ float pf_clamp01(float v) { return fminf(fmaxf(v, 0.0f), 1.0f); }

// `quantize_u8`: (clamp(v,0,1) * 255).round() as u8. `roundf` is ties away from
// zero, which is what Rust's `f32::round` does (not `rintf`).
__device__ __forceinline__ unsigned char pf_to_byte(float value) {
  return (unsigned char)roundf(pf_mul(pf_clamp01(value), 255.0f));
}

// `sample_lut3d`: same index arithmetic, same tap order, same interpolation
// order as the CPU reference.
__device__ __forceinline__ const float* pf_lut3d_entry(
    const float* table, int size, int ri, int gi, int bi) {
  return table + 3 * (ri + size * (gi + size * bi));
}

__device__ __forceinline__ void pf_lut3d_sample(const float* table, int size,
                                                int r_u8, int g_u8, int b_u8,
                                                float* out) {
  const float max_idx = (float)(size - 1);
  const float rf = pf_mul(pf_div((float)r_u8, 255.0f), max_idx);
  const float gf = pf_mul(pf_div((float)g_u8, 255.0f), max_idx);
  const float bf = pf_mul(pf_div((float)b_u8, 255.0f), max_idx);

  int r0 = (int)floorf(rf);
  int g0 = (int)floorf(gf);
  int b0 = (int)floorf(bf);
  if (r0 > size - 1) r0 = size - 1;
  if (g0 > size - 1) g0 = size - 1;
  if (b0 > size - 1) b0 = size - 1;
  if (r0 < 0) r0 = 0;
  if (g0 < 0) g0 = 0;
  if (b0 < 0) b0 = 0;

  const int r1 = (r0 + 1) < (size - 1) ? (r0 + 1) : (size - 1);
  const int g1 = (g0 + 1) < (size - 1) ? (g0 + 1) : (size - 1);
  const int b1 = (b0 + 1) < (size - 1) ? (b0 + 1) : (size - 1);

  const float dr = pf_sub(rf, (float)r0);
  const float dg = pf_sub(gf, (float)g0);
  const float db = pf_sub(bf, (float)b0);

  const float* c000 = pf_lut3d_entry(table, size, r0, g0, b0);
  const float* c100 = pf_lut3d_entry(table, size, r1, g0, b0);
  const float* c010 = pf_lut3d_entry(table, size, r0, g1, b0);
  const float* c110 = pf_lut3d_entry(table, size, r1, g1, b0);
  const float* c001 = pf_lut3d_entry(table, size, r0, g0, b1);
  const float* c101 = pf_lut3d_entry(table, size, r1, g0, b1);
  const float* c011 = pf_lut3d_entry(table, size, r0, g1, b1);
  const float* c111 = pf_lut3d_entry(table, size, r1, g1, b1);

  const float one_minus_dr = pf_sub(1.0f, dr);
  const float one_minus_dg = pf_sub(1.0f, dg);
  const float one_minus_db = pf_sub(1.0f, db);

  for (int ch = 0; ch < 3; ++ch) {
    const float c00 = pf_add(pf_mul(c000[ch], one_minus_dr), pf_mul(c100[ch], dr));
    const float c10 = pf_add(pf_mul(c010[ch], one_minus_dr), pf_mul(c110[ch], dr));
    const float c01 = pf_add(pf_mul(c001[ch], one_minus_dr), pf_mul(c101[ch], dr));
    const float c11 = pf_add(pf_mul(c011[ch], one_minus_dr), pf_mul(c111[ch], dr));
    const float c0 = pf_add(pf_mul(c00, one_minus_dg), pf_mul(c10, dg));
    const float c1 = pf_add(pf_mul(c01, one_minus_dg), pf_mul(c11, dg));
    out[ch] = pf_add(pf_mul(c0, one_minus_db), pf_mul(c1, db));
  }
}

// One fused chain over one RGBA8 pixel. Alpha is carried through untouched.
__device__ __forceinline__ uchar4 pf_apply_chain(uchar4 px,
                                                const PixelFusionUniforms& u,
                                                const unsigned char* lut_blob) {
  for (int i = 0; i < u.op_count; ++i) {
    const PixelFusionOp op = u.ops[i];
    switch (op.kind) {
      case PF_BRIGHTNESS: {
        const float d = op.p0;
        px.x = pf_to_byte(pf_add(pf_div((float)px.x, 255.0f), d));
        px.y = pf_to_byte(pf_add(pf_div((float)px.y, 255.0f), d));
        px.z = pf_to_byte(pf_add(pf_div((float)px.z, 255.0f), d));
        break;
      }
      case PF_WHITEN: {
        const float s = op.p0;
        if (s == 0.0f) break; /* CPU: `if s == 0.0 { return px; }` */
        unsigned char in[3] = {px.x, px.y, px.z};
        for (int ch = 0; ch < 3; ++ch) {
          const float x = pf_div((float)in[ch], 255.0f);
          // x + s * 0.35 * (1 - x) * sqrt(x), left-associative like the CPU.
          const float lift = pf_add(
              x, pf_mul(pf_mul(pf_mul(s, 0.35f), pf_sub(1.0f, x)), pf_sqrt(x)));
          in[ch] = pf_to_byte(lift);
        }
        px.x = in[0];
        px.y = in[1];
        px.z = in[2];
        break;
      }
      case PF_ROSY: {
        const float s = op.p0;
        if (s == 0.0f) break;
        const float r0 = pf_div((float)px.x, 255.0f);
        const float g0 = pf_div((float)px.y, 255.0f);
        const float b0 = pf_div((float)px.z, 255.0f);
        const float r = pf_add(r0, pf_mul(pf_mul(s, 0.14f), pf_sub(1.0f, r0)));
        const float g = pf_sub(g0, pf_mul(pf_mul(s, 0.03f), g0));
        const float b = pf_add(b0, pf_mul(pf_mul(s, 0.02f), pf_sub(1.0f, b0)));
        px.x = pf_to_byte(r);
        px.y = pf_to_byte(g);
        px.z = pf_to_byte(b);
        break;
      }
      case PF_TEMPERATURE: {
        const float w = op.p0;
        const float t = op.p1;
        const float r = pf_add(pf_add(pf_div((float)px.x, 255.0f), pf_mul(w, 0.15f)),
                               pf_mul(t, 0.06f));
        const float g = pf_sub(pf_div((float)px.y, 255.0f), pf_mul(t, 0.10f));
        const float b = pf_add(pf_sub(pf_div((float)px.z, 255.0f), pf_mul(w, 0.15f)),
                               pf_mul(t, 0.06f));
        px.x = pf_to_byte(r);
        px.y = pf_to_byte(g);
        px.z = pf_to_byte(b);
        break;
      }
      case PF_SATURATION: {
        const float factor = pf_add(1.0f, op.p0);
        const float r0 = pf_div((float)px.x, 255.0f);
        const float g0 = pf_div((float)px.y, 255.0f);
        const float b0 = pf_div((float)px.z, 255.0f);
        const float luma = pf_add(
            pf_add(pf_mul(0.2126f, r0), pf_mul(0.7152f, g0)), pf_mul(0.0722f, b0));
        px.x = pf_to_byte(pf_add(luma, pf_mul(pf_sub(r0, luma), factor)));
        px.y = pf_to_byte(pf_add(luma, pf_mul(pf_sub(g0, luma), factor)));
        px.z = pf_to_byte(pf_add(luma, pf_mul(pf_sub(b0, luma), factor)));
        break;
      }
      case PF_LUT1D: {
        const unsigned char* table = lut_blob + op.arg0;
        px.x = table[px.x];
        px.y = table[256 + px.y];
        px.z = table[512 + px.z];
        break;
      }
      case PF_LUT3D: {
        const float* table = reinterpret_cast<const float*>(lut_blob) + op.arg0;
        float rgb[3];
        pf_lut3d_sample(table, op.arg1, px.x, px.y, px.z, rgb);
        px.x = pf_to_byte(rgb[0]);
        px.y = pf_to_byte(rgb[1]);
        px.z = pf_to_byte(rgb[2]);
        break;
      }
      default:
        break; /* unknown op: no-op, never silently guesses a law */
    }
  }
  return px;
}

union PfPacked4 {
  uint4 vector;
  unsigned char bytes[16];
};

__device__ __forceinline__ void pf_apply_chain_bytes(unsigned char* px,
                                                    const PixelFusionUniforms& u,
                                                    const unsigned char* lut_blob) {
  uchar4 value;
  value.x = px[0];
  value.y = px[1];
  value.z = px[2];
  value.w = px[3];
  value = pf_apply_chain(value, u, lut_blob);
  px[0] = value.x;
  px[1] = value.y;
  px[2] = value.z;
  px[3] = value.w;
}

extern "C" __global__ void pixel_fusion_kernel(
    const uchar4* __restrict__ src,
    const unsigned char* __restrict__ lut_blob,
    uchar4* __restrict__ dst,
    PixelFusionUniforms uniforms) {
  int vec = uniforms.vec_load;
  if (vec < 1) vec = 1;
  if (vec > 4) vec = 4;

  const int x0 = (int)(blockIdx.x * blockDim.x + threadIdx.x) * vec;
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (y >= uniforms.height || x0 >= uniforms.width) return;

  const int row = y * uniforms.width;

  // 16-byte path: four pixels per thread, only when the 4-pixel window stays
  // inside the row and the `uint4` load/store is 16-byte aligned.
  const bool vectorised =
      vec == 4 && (x0 + 4) <= uniforms.width && (((row + x0) & 3) == 0);
  if (vectorised) {
    PfPacked4 packed;
    packed.vector = *reinterpret_cast<const uint4*>(src + row + x0);
    pf_apply_chain_bytes(packed.bytes + 0, uniforms, lut_blob);
    pf_apply_chain_bytes(packed.bytes + 4, uniforms, lut_blob);
    pf_apply_chain_bytes(packed.bytes + 8, uniforms, lut_blob);
    pf_apply_chain_bytes(packed.bytes + 12, uniforms, lut_blob);
    *reinterpret_cast<uint4*>(dst + row + x0) = packed.vector;
    return;
  }

  for (int k = 0; k < vec; ++k) {
    const int x = x0 + k;
    if (x >= uniforms.width) return;
    const uchar4 px = src[row + x];
    dst[row + x] = pf_apply_chain(px, uniforms, lut_blob);
  }
}
