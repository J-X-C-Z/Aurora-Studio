// Cross-platform pixel-stage fusion kernel for AMD HIP / ROCm.
//
// This is a source-level candidate for `hipModuleLoadData` + `hipModuleLaunchKernel`
// in `src/hip_backend.rs`. It is NOT pre-compiled into a fatbin by this crate:
// no AMD SDK is required to build the Rust host. A real ROCm/HIP toolchain,
// device and parity test are needed before any DeviceOk/PerfOk claim.
//
// The per-pixel law mirrors `compute_backend::PixelOp::apply_pixel` so the CPU
// lane stays the byte-exact oracle. Alpha is copied unchanged.

#include <hip/hip_runtime.h>
#include <stdint.h>

#pragma clang fp contract(off)

struct PixelFusionParams {
    uint32_t width;
    uint32_t height;
    uint32_t stage_count;
    uint32_t stage_mask;
    uint32_t lut1d_size;
    uint32_t lut3d_size;
    uint32_t flags;
    float brightness_delta;
    float whiten_strength;
    float rosy_strength;
    float temperature_warmth;
    float temperature_tint;
    float saturation_amount;
};
static_assert(sizeof(PixelFusionParams) == 52, "PixelFusionParams must stay 52 bytes");

static constexpr uint32_t STAGE_BRIGHTNESS  = 1u << 0u;
static constexpr uint32_t STAGE_WHITEN      = 1u << 1u;
static constexpr uint32_t STAGE_ROSY        = 1u << 2u;
static constexpr uint32_t STAGE_TEMPERATURE = 1u << 3u;
static constexpr uint32_t STAGE_SATURATION  = 1u << 4u;
static constexpr uint32_t STAGE_LUT1D       = 1u << 5u;
static constexpr uint32_t STAGE_LUT3D       = 1u << 6u;

static constexpr float INV255 = 0.0039215686274509803f;

__device__ static inline uint8_t round_byte(float value) {
    float clamped = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
    return static_cast<uint8_t>(roundf(clamped * 255.0f));
}

__device__ static inline float sample_channel(const float4& value, uint32_t channel) {
    return channel == 0u ? value.x : (channel == 1u ? value.y : value.z);
}

__device__ static inline float3 sample_lut3d(
    uint32_t size, const float4* table, float r, float g, float b) {
    float max_index = static_cast<float>(size - 1u);
    float rf = fminf(fmaxf(r, 0.0f), 1.0f) * max_index;
    float gf = fminf(fmaxf(g, 0.0f), 1.0f) * max_index;
    float bf = fminf(fmaxf(b, 0.0f), 1.0f) * max_index;

    uint32_t r0 = static_cast<uint32_t>(rf);
    uint32_t g0 = static_cast<uint32_t>(gf);
    uint32_t b0 = static_cast<uint32_t>(bf);
    uint32_t r1 = r0 + 1u < size ? r0 + 1u : size - 1u;
    uint32_t g1 = g0 + 1u < size ? g0 + 1u : size - 1u;
    uint32_t b1 = b0 + 1u < size ? b0 + 1u : size - 1u;
    r0 = r0 < size ? r0 : size - 1u;
    g0 = g0 < size ? g0 : size - 1u;
    b0 = b0 < size ? b0 : size - 1u;

    float dr = rf - static_cast<float>(r0);
    float dg = gf - static_cast<float>(g0);
    float db = bf - static_cast<float>(b0);

    float4 c000 = table[r0 + size * (g0 + size * b0)];
    float4 c100 = table[r1 + size * (g0 + size * b0)];
    float4 c010 = table[r0 + size * (g1 + size * b0)];
    float4 c110 = table[r1 + size * (g1 + size * b0)];
    float4 c001 = table[r0 + size * (g0 + size * b1)];
    float4 c101 = table[r1 + size * (g0 + size * b1)];
    float4 c011 = table[r0 + size * (g1 + size * b1)];
    float4 c111 = table[r1 + size * (g1 + size * b1)];

    float3 result = {0.0f, 0.0f, 0.0f};
    for (uint32_t channel = 0u; channel < 3u; ++channel) {
        float c00 = sample_channel(c000, channel) * (1.0f - dr) + sample_channel(c100, channel) * dr;
        float c10 = sample_channel(c010, channel) * (1.0f - dr) + sample_channel(c110, channel) * dr;
        float c01 = sample_channel(c001, channel) * (1.0f - dr) + sample_channel(c101, channel) * dr;
        float c11 = sample_channel(c011, channel) * (1.0f - dr) + sample_channel(c111, channel) * dr;
        float c0 = c00 * (1.0f - dg) + c10 * dg;
        float c1 = c01 * (1.0f - dg) + c11 * dg;
        if (channel == 0u) { result.x = c0 * (1.0f - db) + c1 * db; }
        if (channel == 1u) { result.y = c0 * (1.0f - db) + c1 * db; }
        if (channel == 2u) { result.z = c0 * (1.0f - db) + c1 * db; }
    }
    return result;
}

// By-value params (52 bytes, 13 words) plus four device pointers, matching the
// host-side `[params..., src, dst, lut1d, lut3d]` argument array exactly.
extern "C" __global__ void pixel_fusion(
    PixelFusionParams p,
    uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const uint4* __restrict__ lut1d,
    const float4* __restrict__ lut3d) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pixel_count = p.width * p.height;
    if (gid >= pixel_count) {
        return;
    }

    uint8_t alpha = src[gid * 4u + 3u];
    float r = static_cast<float>(src[gid * 4u + 0u]) * INV255;
    float g = static_cast<float>(src[gid * 4u + 1u]) * INV255;
    float b = static_cast<float>(src[gid * 4u + 2u]) * INV255;

    if ((p.stage_mask & STAGE_BRIGHTNESS) != 0u) {
        float delta = fminf(fmaxf(p.brightness_delta, -1.0f), 1.0f);
        r = r + delta;
        g = g + delta;
        b = b + delta;
    }
    if ((p.stage_mask & STAGE_WHITEN) != 0u) {
        float strength = fminf(fmaxf(p.whiten_strength, 0.0f), 1.0f);
        if (strength > 0.0f) {
            r = r + strength * 0.35f * (1.0f - r) * sqrtf(r);
            g = g + strength * 0.35f * (1.0f - g) * sqrtf(g);
            b = b + strength * 0.35f * (1.0f - b) * sqrtf(b);
        }
    }
    if ((p.stage_mask & STAGE_ROSY) != 0u) {
        float strength = fminf(fmaxf(p.rosy_strength, 0.0f), 1.0f);
        if (strength > 0.0f) {
            r = r + strength * 0.14f * (1.0f - r);
            g = g - strength * 0.03f * g;
            b = b + strength * 0.02f * (1.0f - b);
        }
    }
    if ((p.stage_mask & STAGE_TEMPERATURE) != 0u) {
        float warmth = fminf(fmaxf(p.temperature_warmth, -1.0f), 1.0f);
        float tint = fminf(fmaxf(p.temperature_tint, -1.0f), 1.0f);
        r = r + warmth * 0.15f + tint * 0.06f;
        g = g - tint * 0.10f;
        b = b - warmth * 0.15f + tint * 0.06f;
    }
    if ((p.stage_mask & STAGE_SATURATION) != 0u) {
        float factor = 1.0f + fminf(fmaxf(p.saturation_amount, -1.0f), 1.0f);
        float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        r = luma + (r - luma) * factor;
        g = luma + (g - luma) * factor;
        b = luma + (b - luma) * factor;
    }
    if ((p.stage_mask & STAGE_LUT1D) != 0u) {
        float ri = fminf(fmaxf(r, 0.0f), 1.0f) * 255.0f;
        float gi = fminf(fmaxf(g, 0.0f), 1.0f) * 255.0f;
        float bi = fminf(fmaxf(b, 0.0f), 1.0f) * 255.0f;
        uint4 rv = lut1d[static_cast<uint32_t>(ri)];
        uint4 gv = lut1d[static_cast<uint32_t>(gi)];
        uint4 bv = lut1d[static_cast<uint32_t>(bi)];
        (void)rv.z; (void)rv.w; (void)gv.x; (void)gv.w; (void)bv.x; (void)bv.y;
        r = static_cast<float>(rv.x & 255u) * INV255;
        g = static_cast<float>(gv.y & 255u) * INV255;
        b = static_cast<float>(bv.z & 255u) * INV255;
    }
    if ((p.stage_mask & STAGE_LUT3D) != 0u && p.lut3d_size >= 2u) {
        float3 mapped = sample_lut3d(p.lut3d_size, lut3d, r, g, b);
        r = mapped.x;
        g = mapped.y;
        b = mapped.z;
    }

    dst[gid * 4u + 0u] = round_byte(r);
    dst[gid * 4u + 1u] = round_byte(g);
    dst[gid * 4u + 2u] = round_byte(b);
    dst[gid * 4u + 3u] = alpha;
}
