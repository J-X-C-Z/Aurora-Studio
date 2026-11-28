// PF247 V-face — recovered 0x48 V2FuseOffset producer (cuda_vface.rs
// fuse_offset_cpu). Metal mirror of native/wgsl/pf247_vface.wgsl.
#include <metal_stdlib>
using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

constant uint OFFSET_EDGE = 280u;

struct VfaceParams
{
    uint width;
    uint height;
    uint roi_x;
    uint roi_y;
    uint roi_width;
    uint roi_height;
    uint atlas_width;
    uint atlas_height;
    float degree;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};
static_assert(sizeof(VfaceParams) == 48, "VfaceParams must stay 48 bytes");

static inline float4 unpack(uint packed)
{
    return float4(float(packed & 255u), float((packed >> 8u) & 255u),
                  float((packed >> 16u) & 255u), float((packed >> 24u) & 255u)) * (1.0f / 255.0f);
}

static inline float4 sample_rgba8(uint w, uint h, device const uint* data, float u, float v)
{
    float fx = clamp(floor(u * float(w)), 0.0f, float(w - 1u));
    float fy = clamp(floor(v * float(h)), 0.0f, float(h - 1u));
    return unpack(data[uint(fy) * w + uint(fx)]);
}

kernel void pf247_vface(
    constant VfaceParams& p [[buffer(0)]],
    device const uint* base [[buffer(1)]],
    device const uint* atlas [[buffer(2)]],
    device float4* out [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    uint pixel = gid.y * p.width + gid.x;
    if (gid.x < p.roi_x || gid.x >= p.roi_x + p.roi_width ||
        gid.y < p.roi_y || gid.y >= p.roi_y + p.roi_height) {
        out[pixel] = float4(0.0f);
        return;
    }
    float2 source_uv = (float2(gid) + 0.5f) / float(OFFSET_EDGE);
    float2 atlas_uv = float2(
        float(gid.x - p.roi_x) / float(p.roi_width) + 0.5f / float(p.roi_width),
        float(gid.y - p.roi_y) / float(p.roi_height) + 0.5f / float(p.roi_height));
    float4 src = sample_rgba8(p.width, p.height, base, source_uv.x, source_uv.y);
    float4 roi = sample_rgba8(p.atlas_width, p.atlas_height, atlas, atlas_uv.x, atlas_uv.y);
    float d0 = (src.x - src.y) * 0.25f + (roi.x - roi.y) * 0.25f * p.degree;
    float d1 = (src.z - src.w) * 0.25f + (roi.z - roi.w) * 0.25f * p.degree;
    out[pixel] = float4(max(d0, 0.0f) * 4.0f, max(-d0, 0.0f) * 4.0f,
                        max(d1, 0.0f) * 4.0f, max(-d1, 0.0f) * 4.0f);
}
