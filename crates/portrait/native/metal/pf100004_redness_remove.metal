// PF100004 redness / red-vein removal (kernels/face_detail.rs red_vein).
// Metal mirror of native/wgsl/pf100004_redness_remove.wgsl.
#include <metal_stdlib>
using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

struct RednessRemoveParams
{
    uint width;
    uint height;
    float scale;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};
static_assert(sizeof(RednessRemoveParams) == 24, "RednessRemoveParams must stay 24 bytes");

static inline uint pack_rgba(uint r, uint g, uint b, uint a)
{
    return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

static inline uint round_even_byte(float value)
{
    float scaled = min(max(value, 0.0f), 1.0f) * 255.0f;
    float floor_v = floor(scaled);
    float frac = scaled - floor_v;
    float rounded = floor_v;
    if (frac > 0.5f) {
        rounded = floor_v + 1.0f;
    } else if (frac == 0.5f && ((uint(floor_v) & 1u) != 0u)) {
        rounded = floor_v + 1.0f;
    }
    return clamp(uint(rounded), 0u, 255u);
}

kernel void pf100004_redness_remove(
    constant RednessRemoveParams& p [[buffer(0)]],
    device const uint* src [[buffer(1)]],
    device const uint* effect_rgb [[buffer(2)]],
    device const uint* alpha_plane [[buffer(3)]],
    device uint* out [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    uint pixel = gid.y * p.width + gid.x;
    uint packed = src[pixel];
    float alpha = float(alpha_plane[pixel] & 255u) * (1.0f / 255.0f) * p.scale;
    if (alpha <= 0.0f) {
        out[pixel] = packed;
        return;
    }
    uint eff = effect_rgb[pixel];
    float sr = float(packed & 255u) * (1.0f / 255.0f);
    float sg = float((packed >> 8u) & 255u) * (1.0f / 255.0f);
    float sb = float((packed >> 16u) & 255u) * (1.0f / 255.0f);
    float er = float(eff & 255u) * (1.0f / 255.0f);
    float eg = float((eff >> 8u) & 255u) * (1.0f / 255.0f);
    float eb = float((eff >> 16u) & 255u) * (1.0f / 255.0f);
    uint r = round_even_byte(sr * (1.0f - alpha) + er * alpha);
    uint g = round_even_byte(sg * (1.0f - alpha) + eg * alpha);
    uint b = round_even_byte(sb * (1.0f - alpha) + eb * alpha);
    out[pixel] = pack_rgba(r, g, b, (packed >> 24u) & 255u);
}
