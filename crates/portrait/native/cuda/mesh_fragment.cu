/*
 * O3d CUDA candidate for crates/pixcake/src/raster_fragment.rs.
 *
 * Same parameters, same order:
 *   1. sample_uv = src_uv
 *   2. if clamp_edge: sample_uv = dst_uv + (src_uv - dst_uv) * smoothcoord(src_uv)
 *   3. if uv_map: two bilinear taps at sample_uv and dst_uv, fuse channels
 *      else: catmull_rom(sample_uv) or bilinear(sample_uv)
 *   4. quantize with ties-to-even
 *
 * Arithmetic uses __fadd_rn/__fmul_rn so nothing contracts into fma and drifts
 * from the Rust f32 sequence; division stays IEEE (`div.rn.f32`).
 */

#include "mesh_fragment.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

struct Vec2f {
  float x;
  float y;
};

__device__ __forceinline__ float AddRn(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float MulRn(float a, float b) { return __fmul_rn(a, b); }

/* Type 63 ReshapeNormal.frag */
__device__ __forceinline__ float SmoothCoordReshape(Vec2f uv) {
  const float x = AddRn(2.49f, -fabsf(MulRn(4.98f, AddRn(uv.x, -0.5f))));
  const float y = AddRn(2.49f, -fabsf(MulRn(4.98f, AddRn(uv.y, -0.5f))));
  return fminf(fmaxf(fminf(x, y), 0.0f), 1.0f);
}

/* Type 0x40 ReshapeNormalAdjClampEdge.frag */
__device__ __forceinline__ float SmoothCoordAdj(Vec2f uv, MeshFragmentClamp clamp) {
  if (clamp.factor < 0.0f) {
    return 1.0f;
  }
  float x = 1.0f;
  float y = 1.0f;
  const float half = MulRn(clamp.factor, 0.5f);
  if (clamp.protect_x == 1) {
    x = AddRn(half, -fabsf(MulRn(clamp.factor, fmaxf(AddRn(0.5f, -uv.x), 0.0f))));
  }
  if (clamp.protect_x == 2) {
    x = AddRn(half, -fabsf(MulRn(clamp.factor, fmaxf(AddRn(uv.x, -0.5f), 0.0f))));
  }
  if (clamp.protect_y == 1) {
    y = AddRn(half, -fabsf(MulRn(clamp.factor, fmaxf(AddRn(0.5f, -uv.y), 0.0f))));
  }
  if (clamp.protect_y == 2) {
    y = AddRn(half, -fabsf(MulRn(clamp.factor, fmaxf(AddRn(uv.y, -0.5f), 0.0f))));
  }
  return fminf(fmaxf(fminf(x, y), 0.0f), 1.0f);
}

__device__ __forceinline__ float SmoothCoord(
    Vec2f uv, int32_t edge, MeshFragmentClamp clamp) {
  return edge == MESH_FRAGMENT_EDGE_ADJ_CLAMP ? SmoothCoordAdj(uv, clamp)
                                              : SmoothCoordReshape(uv);
}

__device__ __forceinline__ float Texel(
    const uint8_t* source,
    int32_t width,
    int32_t x,
    int32_t y,
    int32_t channel) {
  const size_t offset =
      (static_cast<size_t>(y) * static_cast<size_t>(width) +
       static_cast<size_t>(x)) * 4u +
      static_cast<size_t>(channel);
  return static_cast<float>(source[offset]) / 255.0f;
}

__device__ __forceinline__ float Lerp(float a, float b, float t) {
  return AddRn(a, MulRn(AddRn(b, -a), t));
}

/* texture2D + GL_LINEAR + clamp-to-edge, p = uv * size - 0.5. */
__device__ __forceinline__ void Bilinear(
    const uint8_t* source,
    int32_t width,
    int32_t height,
    Vec2f uv,
    float* out) {
  const float w = static_cast<float>(width);
  const float h = static_cast<float>(height);
  const float x = fminf(fmaxf(AddRn(MulRn(uv.x, w), -0.5f), 0.0f), w - 1.0f);
  const float y = fminf(fmaxf(AddRn(MulRn(uv.y, h), -0.5f), 0.0f), h - 1.0f);
  const float x0f = floorf(x);
  const float y0f = floorf(y);
  const float x1f = fminf(AddRn(x0f, 1.0f), w - 1.0f);
  const float y1f = fminf(AddRn(y0f, 1.0f), h - 1.0f);
  const float fx = AddRn(x, -x0f);
  const float fy = AddRn(y, -y0f);
  const int32_t x0 = static_cast<int32_t>(x0f);
  const int32_t y0 = static_cast<int32_t>(y0f);
  const int32_t x1 = static_cast<int32_t>(x1f);
  const int32_t y1 = static_cast<int32_t>(y1f);
  for (int32_t c = 0; c < 4; ++c) {
    const float top = Lerp(
        Texel(source, width, x0, y0, c), Texel(source, width, x1, y0, c), fx);
    const float bottom = Lerp(
        Texel(source, width, x0, y1, c), Texel(source, width, x1, y1, c), fx);
    out[c] = Lerp(top, bottom, fy);
  }
}

__device__ __forceinline__ float Weight0(float f) {
  return MulRn(f, AddRn(-0.5f, MulRn(f, AddRn(1.0f, MulRn(-0.5f, f)))));
}

__device__ __forceinline__ float Weight1(float f) {
  return AddRn(1.0f, MulRn(MulRn(f, f), AddRn(-2.5f, MulRn(1.5f, f))));
}

__device__ __forceinline__ float Weight2(float f) {
  return MulRn(f, AddRn(0.5f, MulRn(f, AddRn(2.0f, MulRn(-1.5f, f)))));
}

__device__ __forceinline__ float Weight3(float f) {
  return MulRn(MulRn(f, f), AddRn(-0.5f, MulRn(0.5f, f)));
}

/* The shipped 9-tap Catmull-Rom with explicit u_Width / u_Height. */
__device__ __forceinline__ void CatmullRom(
    const uint8_t* source,
    int32_t width,
    int32_t height,
    Vec2f uv,
    float tex_w,
    float tex_h,
    float* out) {
  const float sx = MulRn(uv.x, tex_w);
  const float sy = MulRn(uv.y, tex_h);
  const float px = AddRn(floorf(AddRn(sx, -0.5f)), 0.5f);
  const float py = AddRn(floorf(AddRn(sy, -0.5f)), 0.5f);
  const float fx = AddRn(sx, -px);
  const float fy = AddRn(sy, -py);

  const float w0x = Weight0(fx);
  const float w1x = Weight1(fx);
  const float w2x = Weight2(fx);
  const float w3x = Weight3(fx);
  const float w0y = Weight0(fy);
  const float w1y = Weight1(fy);
  const float w2y = Weight2(fy);
  const float w3y = Weight3(fy);
  const float w12x = AddRn(w1x, w2x);
  const float w12y = AddRn(w1y, w2y);
  const float t12x = AddRn(px, w2x / w12x);
  const float t12y = AddRn(py, w2y / w12y);

  const float tap_x[9] = {AddRn(px, -1.0f), t12x, AddRn(px, 2.0f),
                          AddRn(px, -1.0f), t12x, AddRn(px, 2.0f),
                          AddRn(px, -1.0f), t12x, AddRn(px, 2.0f)};
  const float tap_y[9] = {AddRn(py, -1.0f), AddRn(py, -1.0f), AddRn(py, -1.0f),
                          t12y, t12y, t12y,
                          AddRn(py, 2.0f), AddRn(py, 2.0f), AddRn(py, 2.0f)};
  const float weight[9] = {MulRn(w0x, w0y), MulRn(w12x, w0y), MulRn(w3x, w0y),
                           MulRn(w0x, w12y), MulRn(w12x, w12y), MulRn(w3x, w12y),
                           MulRn(w0x, w3y), MulRn(w12x, w3y), MulRn(w3x, w3y)};

  for (int32_t c = 0; c < 4; ++c) {
    out[c] = 0.0f;
  }
  for (int32_t t = 0; t < 9; ++t) {
    Vec2f tap;
    tap.x = tap_x[t] / tex_w;
    tap.y = tap_y[t] / tex_h;
    float sample[4];
    Bilinear(source, width, height, tap, sample);
    for (int32_t c = 0; c < 4; ++c) {
      out[c] = AddRn(out[c], MulRn(sample[c], weight[t]));
    }
  }
}

__device__ __forceinline__ uint8_t Quantize(float value) {
  const int32_t q = __float2int_rn(MulRn(value, 255.0f));
  return static_cast<uint8_t>(q < 0 ? 0 : (q > 255 ? 255 : q));
}

__global__ void mesh_fragment_evaluate(MeshFragmentInput input) {
  const int32_t point = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (point >= input.count) {
    return;
  }

  Vec2f src_uv;
  Vec2f dst_uv;
  src_uv.x = input.src_uv[static_cast<size_t>(point) * 2u];
  src_uv.y = input.src_uv[static_cast<size_t>(point) * 2u + 1u];
  dst_uv.x = input.dst_uv[static_cast<size_t>(point) * 2u];
  dst_uv.y = input.dst_uv[static_cast<size_t>(point) * 2u + 1u];

  const float tex_w =
      input.catmull_width > 0.0f ? input.catmull_width : static_cast<float>(input.width);
  const float tex_h =
      input.catmull_height > 0.0f ? input.catmull_height : static_cast<float>(input.height);

  Vec2f sample_uv = src_uv;
  if (input.defines.clamp_edge) {
    const float factor = SmoothCoord(src_uv, input.edge, input.clamp);
    sample_uv.x = AddRn(dst_uv.x, MulRn(AddRn(src_uv.x, -dst_uv.x), factor));
    sample_uv.y = AddRn(dst_uv.y, MulRn(AddRn(src_uv.y, -dst_uv.y), factor));
  }

  float color[4];
  if (input.defines.uv_map) {
    float sampled[4];
    float current[4];
    Bilinear(input.source, input.width, input.height, sample_uv, sampled);
    Bilinear(input.source, input.width, input.height, dst_uv, current);
    const float dx = AddRn(sample_uv.x, -dst_uv.x);
    const float dy = AddRn(sample_uv.y, -dst_uv.y);
    if (input.defines.uv_map_ba) {
      color[0] = current[0];
      color[1] = current[1];
      color[2] = AddRn(sampled[2], dx);
      color[3] = AddRn(sampled[3], dy);
    } else {
      color[0] = AddRn(sampled[0], dx);
      color[1] = AddRn(sampled[1], dy);
      color[2] = current[2];
      color[3] = current[3];
    }
  } else if (input.defines.catmull_rom) {
    CatmullRom(
        input.source, input.width, input.height, sample_uv, tex_w, tex_h, color);
  } else {
    Bilinear(input.source, input.width, input.height, sample_uv, color);
  }

  uint8_t* out = input.out + static_cast<size_t>(point) * 4u;
  for (int32_t c = 0; c < 4; ++c) {
    out[c] = Quantize(color[c]);
  }
}

}  // namespace

extern "C" MeshFragmentStatus mesh_fragment_run(
    const MeshFragmentInput* input,
    void* stream) {
  if (input == nullptr) {
    return MESH_FRAGMENT_INVALID_ARGUMENT;
  }
  if (input->source == nullptr || input->out == nullptr || input->src_uv == nullptr ||
      input->dst_uv == nullptr) {
    return MESH_FRAGMENT_INVALID_ARGUMENT;
  }
  if (input->width <= 0 || input->height <= 0 || input->count <= 0) {
    return MESH_FRAGMENT_INVALID_ARGUMENT;
  }
  if (input->edge != MESH_FRAGMENT_EDGE_RESHAPE_NORMAL &&
      input->edge != MESH_FRAGMENT_EDGE_ADJ_CLAMP) {
    return MESH_FRAGMENT_OUT_OF_RANGE;
  }
  if (input->clamp.protect_x < 0 || input->clamp.protect_x > 2 ||
      input->clamp.protect_y < 0 || input->clamp.protect_y > 2 ||
      !(input->clamp.factor == input->clamp.factor)) {
    return MESH_FRAGMENT_OUT_OF_RANGE;
  }
  if ((input->defines.clamp_edge & ~1) || (input->defines.uv_map & ~1) ||
      (input->defines.uv_map_ba & ~1) || (input->defines.catmull_rom & ~1)) {
    return MESH_FRAGMENT_OUT_OF_RANGE;
  }

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const int32_t block = 128;
  const int32_t grid = (input->count + block - 1) / block;
  mesh_fragment_evaluate<<<grid, block, 0, cuda_stream>>>(*input);

  const cudaError_t launch_status = cudaGetLastError();
  const cudaError_t sync_status = cudaStreamSynchronize(cuda_stream);
  if (launch_status != cudaSuccess || sync_status != cudaSuccess) {
    return MESH_FRAGMENT_CUDA_ERROR;
  }
  return MESH_FRAGMENT_OK;
}

extern "C" const char* mesh_fragment_status_name(MeshFragmentStatus status) {
  switch (status) {
    case MESH_FRAGMENT_OK:
      return "OK";
    case MESH_FRAGMENT_INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case MESH_FRAGMENT_OUT_OF_RANGE:
      return "OUT_OF_RANGE";
    case MESH_FRAGMENT_CUDA_ERROR:
      return "CUDA_ERROR";
  }
  return "UNKNOWN";
}
