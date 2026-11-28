// PF21105 AI body skin unify (kernels/face_detail.rs body_skin_unify).
// Metal mirror of native/wgsl/pf21105_ai_body_skin_unify.wgsl.
#include <metal_stdlib>
using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

struct SkinUnify21105Params
{
    uint width;
    uint height;
    float degree01;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};
static_assert(sizeof(SkinUnify21105Params) == 24, "SkinUnify21105Params must stay 24 bytes");

static inline uint pack_rgba(uint r, uint g, uint b, uint a)
{
    return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

static inline uint trunc_byte(float value)
{
    return uint(clamp(trunc(value), 0.0f, 255.0f));
}

kernel void pf21105_ai_body_skin_unify(
    constant SkinUnify21105Params& p [[buffer(0)]],
    device const uint* current [[buffer(1)]],
    device const uint* mask [[buffer(2)]],
    device const uint* effect [[buffer(3)]],
    device uint* out [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    uint pixel = gid.y * p.width + gid.x;
    uint cur = current[pixel];
    float t = min(max(float(mask[pixel] & 255u) * (1.0f / 255.0f) * p.degree01, 0.0f), 1.0f);
    if (t <= 0.0f) {
        out[pixel] = cur;
        return;
    }
    uint eff = effect[pixel];
    float cr = float(cur & 255u), cg = float((cur >> 8u) & 255u), cb = float((cur >> 16u) & 255u);
    float er = float(eff & 255u), eg = float((eff >> 8u) & 255u), eb = float((eff >> 16u) & 255u);
    uint r = trunc_byte(cr * (1.0f - t) + er * t);
    uint g = trunc_byte(cg * (1.0f - t) + eg * t);
    uint b = trunc_byte(cb * (1.0f - t) + eb * t);
    out[pixel] = pack_rgba(r, g, b, (cur >> 24u) & 255u);
}
