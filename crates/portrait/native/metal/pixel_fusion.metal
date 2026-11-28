// Cross-platform pixel-stage fusion kernel for Apple Metal (MSL).
//
// Single full-image pass: one RGBA8 read + one RGBA8 write, regardless of how
// many point operations the caller fused. The per-pixel law is a literal MSL
// mirror of `compute_backend::PixelOp::apply_pixel` so the CPU lane stays the
// byte-exact oracle. Alpha is copied from the source and is never modified.
//
// This source is embedded by `src/metal_backend.rs`. It is a source-level
// candidate: compiling it or obtaining device parity still requires a real
// macOS/iOS Metal runtime and is NOT claimed by the Windows host build.

#include <metal_stdlib>
using namespace metal;

// Fast-math contraction would change the mul-then-add order of the recovered
// portrait law. Keep the same ordering as the CPU/CUDA paths.
#pragma clang fp contract(off)

struct PixelFusionParams
{
    uint width;
    uint height;
    uint stage_count;
    uint stage_mask;
    uint lut1d_size;
    uint lut3d_size;
    uint flags;
    float brightness_delta;
    float whiten_strength;
    float rosy_strength;
    float temperature_warmth;
    float temperature_tint;
    float saturation_amount;
};
static_assert(sizeof(PixelFusionParams) == 52, "PixelFusionParams must stay 52 bytes");

constant uint STAGE_BRIGHTNESS  = 1u << 0u;
constant uint STAGE_WHITEN      = 1u << 1u;
constant uint STAGE_ROSY        = 1u << 2u;
constant uint STAGE_TEMPERATURE = 1u << 3u;
constant uint STAGE_SATURATION  = 1u << 4u;
constant uint STAGE_LUT1D       = 1u << 5u;
constant uint STAGE_LUT3D       = 1u << 6u;

constant float INV255 = 0.0039215686274509803f;

static inline uchar round_byte(float value)
{
    return uchar(clamp(round(clamp(value, 0.0f, 1.0f) * 255.0f), 0.0f, 255.0f));
}

static inline float3 sample_lut3d(uint size, device const float4* table, float r, float g, float b)
{
    float max_index = float(size - 1u);
    float rf = clamp(r, 0.0f, 1.0f) * max_index;
    float gf = clamp(g, 0.0f, 1.0f) * max_index;
    float bf = clamp(b, 0.0f, 1.0f) * max_index;

    uint r0 = uint(rf);
    uint g0 = uint(gf);
    uint b0 = uint(bf);
    uint r1 = min(r0 + 1u, size - 1u);
    uint g1 = min(g0 + 1u, size - 1u);
    uint b1 = min(b0 + 1u, size - 1u);
    r0 = min(r0, size - 1u);
    g0 = min(g0, size - 1u);
    b0 = min(b0, size - 1u);

    float dr = rf - float(r0);
    float dg = gf - float(g0);
    float db = bf - float(b0);

    float4 c000 = table[r0 + size * (g0 + size * b0)];
    float4 c100 = table[r1 + size * (g0 + size * b0)];
    float4 c010 = table[r0 + size * (g1 + size * b0)];
    float4 c110 = table[r1 + size * (g1 + size * b0)];
    float4 c001 = table[r0 + size * (g0 + size * b1)];
    float4 c101 = table[r1 + size * (g0 + size * b1)];
    float4 c011 = table[r0 + size * (g1 + size * b1)];
    float4 c111 = table[r1 + size * (g1 + size * b1)];

    float3 result = float3(0.0f);
    for (uint channel = 0u; channel < 3u; ++channel) {
        float c00 = c000[channel] * (1.0f - dr) + c100[channel] * dr;
        float c10 = c010[channel] * (1.0f - dr) + c110[channel] * dr;
        float c01 = c001[channel] * (1.0f - dr) + c101[channel] * dr;
        float c11 = c011[channel] * (1.0f - dr) + c111[channel] * dr;
        float c0 = c00 * (1.0f - dg) + c10 * dg;
        float c1 = c01 * (1.0f - dg) + c11 * dg;
        result[channel] = c0 * (1.0f - db) + c1 * db;
    }
    return result;
}

kernel void pixel_fusion(
    constant PixelFusionParams& p [[buffer(0)]],
    device const uchar4* src [[buffer(1)]],
    device uchar4* dst [[buffer(2)]],
    device const uchar4* lut1d [[buffer(3)]],
    device const float4* lut3d [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint pixel_count = p.width * p.height;
    if (gid >= pixel_count) {
        return;
    }

    uchar4 source = src[gid];
    float r = float(source.x) * INV255;
    float g = float(source.y) * INV255;
    float b = float(source.z) * INV255;

    if ((p.stage_mask & STAGE_BRIGHTNESS) != 0u) {
        float delta = clamp(p.brightness_delta, -1.0f, 1.0f);
        r = r + delta;
        g = g + delta;
        b = b + delta;
    }
    if ((p.stage_mask & STAGE_WHITEN) != 0u) {
        float strength = clamp(p.whiten_strength, 0.0f, 1.0f);
        if (strength > 0.0f) {
            r = r + strength * 0.35f * (1.0f - r) * sqrt(r);
            g = g + strength * 0.35f * (1.0f - g) * sqrt(g);
            b = b + strength * 0.35f * (1.0f - b) * sqrt(b);
        }
    }
    if ((p.stage_mask & STAGE_ROSY) != 0u) {
        float strength = clamp(p.rosy_strength, 0.0f, 1.0f);
        if (strength > 0.0f) {
            r = r + strength * 0.14f * (1.0f - r);
            g = g - strength * 0.03f * g;
            b = b + strength * 0.02f * (1.0f - b);
        }
    }
    if ((p.stage_mask & STAGE_TEMPERATURE) != 0u) {
        float warmth = clamp(p.temperature_warmth, -1.0f, 1.0f);
        float tint = clamp(p.temperature_tint, -1.0f, 1.0f);
        r = r + warmth * 0.15f + tint * 0.06f;
        g = g - tint * 0.10f;
        b = b - warmth * 0.15f + tint * 0.06f;
    }
    if ((p.stage_mask & STAGE_SATURATION) != 0u) {
        float factor = 1.0f + clamp(p.saturation_amount, -1.0f, 1.0f);
        float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        r = luma + (r - luma) * factor;
        g = luma + (g - luma) * factor;
        b = luma + (b - luma) * factor;
    }
    if ((p.stage_mask & STAGE_LUT1D) != 0u) {
        float3 mapped = float3(
            float(lut1d[uint(clamp(r, 0.0f, 1.0f) * 255.0f)].x) * INV255,
            float(lut1d[uint(clamp(g, 0.0f, 1.0f) * 255.0f)].y) * INV255,
            float(lut1d[uint(clamp(b, 0.0f, 1.0f) * 255.0f)].z) * INV255);
        r = mapped.x;
        g = mapped.y;
        b = mapped.z;
    }
    if ((p.stage_mask & STAGE_LUT3D) != 0u && p.lut3d_size >= 2u) {
        float3 mapped = sample_lut3d(p.lut3d_size, lut3d, r, g, b);
        r = mapped.x;
        g = mapped.y;
        b = mapped.z;
    }

    dst[gid] = uchar4(round_byte(r), round_byte(g), round_byte(b), source.w);
}
