// Viewer scope maps on the GPU: one sample per thread, three density maps out.
//
// This is the CUDA twin of `crates/engine/src/scopes.rs::viewer_maps` — same
// sampling stride, same BT.709 luma, same top-down row order, same rounding — so
// a frame counted here and a frame counted on the CPU land in identical bins.
// Keep the bin geometry in step with the engine constants:
//   levels 160, columns 256, parade lanes 3 x 85, vectorscope 160 x 160.
//
// Compile with: nvcc -arch=compute_75 --ptx -O3 -fmad=false scope_gpu.cu
// `-fmad=false` matters: the luma and chroma chains must round exactly like the
// CPU reference, so no multiply-add contraction is allowed.

#include <cuda_runtime.h>

#define LEVELS 160
#define COLUMNS 256
#define LANE_WIDTH (COLUMNS / 3)
#define VECTOR 160
#define PARADE_BASE (LEVELS * COLUMNS)
#define VECTOR_BASE (2 * LEVELS * COLUMNS)

// Half away from zero as floor + a 0.5 comparison on the fraction. The PTX form
// is `cvt.rmi` + `sub.rn` + `setp.ge` + `selp`, which is why this is not written
// as `roundf` (that is ties-to-even and would drift against the CPU).
__device__ __forceinline__ int level_of(float q) {
    float floor_q = floorf(q);
    float frac = q - floor_q;
    int floored = (int)floor_q;
    return frac >= 0.5f ? floored + 1 : floored;
}

// Row index for a 0-255 value, top row white.
__device__ __forceinline__ int row_of(float value) {
    int level = level_of((value / 255.0f) * (float)(LEVELS - 1));
    return (LEVELS - 1) - min(max(level, 0), LEVELS - 1);
}

__device__ __forceinline__ int vector_cell(float q) {
    int cell = level_of(q);
    return min(max(cell, 0), VECTOR - 1);
}

extern "C" __global__ void zero_u32_kernel(unsigned int* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = 0u;
    }
}

extern "C" __global__ void scope_maps_kernel(
    const uchar4* pixels,
    unsigned int* maps,
    int w,
    int h,
    int step,
    int rshift
) {
    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;
    int x = sx * step;
    int y = sy * step;
    if (x >= w || y >= h) {
        return;
    }
    unsigned int packed = *(const unsigned int*)&pixels[y * w + x];
    int bshift = 16 - rshift;
    unsigned int r = (packed >> rshift) & 0xFFu;
    unsigned int g = (packed >> 8) & 0xFFu;
    unsigned int b = (packed >> bshift) & 0xFFu;

    float fr = (float)r;
    float fg = (float)g;
    float fb = (float)b;
    // BT.709 luma, multiplied and added left to right like the CPU.
    float luma = fr * 0.2126f + fg * 0.7152f + fb * 0.0722f;

    // Waveform: luma level, one column per output pixel.
    int wave_row = row_of(luma);
    int wave_col = min((x * COLUMNS) / w, COLUMNS - 1);
    atomicAdd(&maps[wave_row * COLUMNS + wave_col], 1u);

    // RGB parade: three lanes side by side.
    int lane_col = min((x * LANE_WIDTH) / w, LANE_WIDTH - 1);
    atomicAdd(&maps[PARADE_BASE + row_of(fr) * COLUMNS + lane_col], 1u);
    atomicAdd(&maps[PARADE_BASE + row_of(fg) * COLUMNS + LANE_WIDTH + lane_col], 1u);
    atomicAdd(&maps[PARADE_BASE + row_of(fb) * COLUMNS + 2 * LANE_WIDTH + lane_col], 1u);

    // Vectorscope: BT.709 chroma, neutral in the centre, Cb up and Cr right.
    float cb = (fb - luma) / 1.8556f / 255.0f;
    float cr = (fr - luma) / 1.5748f / 255.0f;
    int vx = vector_cell((0.5f + cb) * (float)(VECTOR - 1));
    int vy = vector_cell((0.5f - cr) * (float)(VECTOR - 1));
    atomicAdd(&maps[VECTOR_BASE + vy * VECTOR + vx], 1u);
}
