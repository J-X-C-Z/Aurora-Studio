// PF101 silkworm — recovered BlendImageLayerWithMask.frag pixel law.
// Metal mirror of native/wgsl/pf101_silkworm.wgsl.
#include <metal_stdlib>
using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

struct SilkwormParams
{
    uint width;
    uint height;
    float degree01;
    uint filter_type;   // 41 Multiply, 42 SoftLight, 43 Lighten, 44 Screen
    uint reserved0;
    uint reserved1;
    uint reserved2;
};
static_assert(sizeof(SilkwormParams) == 32, "SilkwormParams must stay 32 bytes");

constant float INV255 = 0.0039215686274509803f;

static inline uint pack_rgba(uint r, uint g, uint b, uint a)
{
    return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

static inline float4 sample_rgba(uint w, uint h, device const uint* data, float u, float v)
{
    float px = clamp(u * float(w) - 0.5f, 0.0f, float(w - 1u));
    float py = clamp(v * float(h) - 0.5f, 0.0f, float(h - 1u));
    uint x0 = uint(floor(px));
    uint y0 = uint(floor(py));
    uint x1 = min(x0 + 1u, w - 1u);
    uint y1 = min(y0 + 1u, h - 1u);
    float tx = px - float(x0);
    float ty = py - float(y0);
    uint a = data[y0 * w + x0];
    uint b = data[y0 * w + x1];
    uint c = data[y1 * w + x0];
    uint d = data[y1 * w + x1];
    float4 result;
    for (uint ch = 0u; ch < 4u; ++ch) {
        uint shift = ch * 8u;
        float av = float((a >> shift) & 255u);
        float bv = float((b >> shift) & 255u);
        float cv = float((c >> shift) & 255u);
        float dv = float((d >> shift) & 255u);
        float top = av * (1.0f - tx) + bv * tx;
        float bottom = cv * (1.0f - tx) + dv * tx;
        result[ch] = (top * (1.0f - ty) + bottom * ty) * INV255;
    }
    return result;
}

static inline float soft_light(float base, float blend)
{
    if (blend < 0.5f) {
        return 2.0f * base * blend + base * base * (1.0f - 2.0f * blend);
    }
    return sqrt(base) * (2.0f * blend - 1.0f) + 2.0f * base * (1.0f - blend);
}

static inline float blend_channel(float base, float blend, uint filter_type)
{
    if (filter_type == 41u) { return base * blend; }
    if (filter_type == 43u) { return fmax(base, blend); }
    if (filter_type == 44u) { return 1.0f - (1.0f - base) * (1.0f - blend); }
    return soft_light(base, blend);
}

static inline uint byte_even(float value)
{
    float scaled = clamp(value, 0.0f, 1.0f) * 255.0f;
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

kernel void pf101_silkworm(
    constant SilkwormParams& p [[buffer(0)]],
    device const uint* src [[buffer(1)]],
    device const uint* material [[buffer(2)]],
    device const uint* mask [[buffer(3)]],
    device const float2* mask_uv [[buffer(4)]],
    device const float2* material_uv [[buffer(5)]],
    device uint* out [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    uint pixel = gid.y * p.width + gid.x;
    uint packed = src[pixel];
    float4 mat = sample_rgba(p.width, p.height, material, material_uv[pixel].x, material_uv[pixel].y);
    float4 mask_sample = sample_rgba(p.width, p.height, mask, mask_uv[pixel].x, mask_uv[pixel].y);
    float factor = clamp(mat.w * p.degree01 * mask_sample.x, 0.0f, 1.0f);
    if (factor <= 0.01f) {
        out[pixel] = packed;
        return;
    }
    float3 base = float3(float(packed & 255u), float((packed >> 8u) & 255u), float((packed >> 16u) & 255u)) * INV255;
    float3 blended = float3(
        blend_channel(base.x, mat.x, p.filter_type),
        blend_channel(base.y, mat.y, p.filter_type),
        blend_channel(base.z, mat.z, p.filter_type));
    uint r = byte_even(base.x * (1.0f - factor) + blended.x * factor);
    uint g = byte_even(base.y * (1.0f - factor) + blended.y * factor);
    uint b = byte_even(base.z * (1.0f - factor) + blended.z * factor);
    out[pixel] = pack_rgba(r, g, b, (packed >> 24u) & 255u);
}
