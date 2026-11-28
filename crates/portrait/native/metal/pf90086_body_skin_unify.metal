// PF90086 body skin-tone unify (kernels/makeup.rs body_tone_unify).
// Metal mirror of native/wgsl/pf90086_body_skin_unify.wgsl.
#include <metal_stdlib>
using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

struct BodyToneParams
{
    uint width;
    uint height;
    float w_m;
    float w_o;
    uint reserved0;
    uint reserved1;
};
static_assert(sizeof(BodyToneParams) == 24, "BodyToneParams must stay 24 bytes");

static inline uint pack_rgba(uint r, uint g, uint b, uint a)
{
    return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

static inline uint trunc_byte(float value)
{
    // C-like truncation toward zero, then clamp to byte.
    float truncated = trunc(value);
    return uint(clamp(truncated, 0.0f, 255.0f));
}

kernel void pf90086_body_skin_unify(
    constant BodyToneParams& p [[buffer(0)]],
    device const uint* src [[buffer(1)]],
    device const float* m_plane [[buffer(2)]],
    device const float* o_plane [[buffer(3)]],
    device uint* out [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    uint pixel = gid.y * p.width + gid.x;
    uint packed = src[pixel];
    uint base = pixel * 3u;
    float3 rgb = float3(float(packed & 255u), float((packed >> 8u) & 255u), float((packed >> 16u) & 255u));
    float3 m = float3(m_plane[base], m_plane[base + 1u], m_plane[base + 2u]);
    float3 o = float3(o_plane[base], o_plane[base + 1u], o_plane[base + 2u]);
    float3 value = rgb + 255.0f * (p.w_m * m * (rgb * (1.0f / 255.0f)) + p.w_o * o);
    uint r = trunc_byte(value.x);
    uint g = trunc_byte(value.y);
    uint b = trunc_byte(value.z);
    out[pixel] = pack_rgba(r, g, b, (packed >> 24u) & 255u);
}
