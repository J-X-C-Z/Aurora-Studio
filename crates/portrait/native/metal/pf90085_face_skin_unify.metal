// PF90085 face skin-tone unify — reduced recovered pixel tail.
// Metal mirror of native/wgsl/pf90085_face_skin_unify.wgsl.
#include <metal_stdlib>
using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

struct SkinUnify90085Params
{
    uint width;
    uint height;
    float degree;
    uint flags;
    uint reserved0;
    uint reserved1;
    uint reserved2;
    float target_r;
    float target_g;
    float target_b;
    float reserved3;
};
static_assert(sizeof(SkinUnify90085Params) == 48, "SkinUnify90085Params must stay 48 bytes");

constant float FACE_PARSING_WEIGHT = 0.39f;
constant float MASK_GATE = 0.001f;

static inline float unpack_byte(uint v) { return float(v & 255u) * (1.0f / 255.0f); }

static inline uint quantise_even(float value)
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

static inline uint pack_rgba(uint r, uint g, uint b, uint a)
{
    return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

kernel void pf90085_face_skin_unify(
    constant SkinUnify90085Params& p [[buffer(0)]],
    device const uint* src [[buffer(1)]],
    device const uint* face_mask_g [[buffer(2)]],
    device const uint* skin_mask [[buffer(3)]],
    device const uint* parsing_a [[buffer(4)]],
    device const uint* parsing_g [[buffer(5)]],
    device const uint* parsing_b [[buffer(6)]],
    device const uint* protect [[buffer(7)]],
    device const uint* model_weight [[buffer(8)]],
    device const uint* model_target [[buffer(9)]],
    device const float* fallback_target [[buffer(10)]],
    device uint* out [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    uint pixel = gid.y * p.width + gid.x;
    uint src_packed = src[pixel];
    if (p.degree <= MASK_GATE) {
        out[pixel] = src_packed;
        return;
    }
    float face = unpack_byte(face_mask_g[pixel]);
    float skin = ((p.flags & 1u) != 0u) ? unpack_byte(skin_mask[pixel]) : 0.0f;
    float parsing = face * FACE_PARSING_WEIGHT;
    if ((p.flags & 2u) != 0u) {
        parsing = (unpack_byte(parsing_a[pixel]) + unpack_byte(parsing_g[pixel]) +
                   unpack_byte(parsing_b[pixel]) + face) * FACE_PARSING_WEIGHT;
    }
    float support = fmax(skin, parsing);
    if ((p.flags & 4u) != 0u) {
        support = support * (1.0f - unpack_byte(protect[pixel]));
    }
    if (support <= MASK_GATE) {
        out[pixel] = src_packed;
        return;
    }
    float mix_amount = support * p.degree;
    if ((p.flags & 8u) != 0u) {
        mix_amount = mix_amount * unpack_byte(model_weight[pixel]);
    }
    if (mix_amount <= 0.0f) {
        out[pixel] = src_packed;
        return;
    }
    float alpha = min(max(mix_amount, 0.0f), 1.0f);
    float3 target = float3(p.target_r, p.target_g, p.target_b);
    if ((p.flags & 16u) != 0u) {
        uint mp = model_target[pixel];
        target = float3(float(mp & 255u), float((mp >> 8u) & 255u), float((mp >> 16u) & 255u)) * (1.0f / 255.0f);
    } else if ((p.flags & 32u) == 0u) {
        target = float3(fallback_target[pixel * 3u], fallback_target[pixel * 3u + 1u], fallback_target[pixel * 3u + 2u]);
    }
    float sr = unpack_byte(src_packed);
    float sg = unpack_byte(src_packed >> 8u);
    float sb = unpack_byte(src_packed >> 16u);
    uint r = quantise_even(sr * (1.0f - alpha) + target.x * alpha);
    uint g = quantise_even(sg * (1.0f - alpha) + target.y * alpha);
    uint b = quantise_even(sb * (1.0f - alpha) + target.z * alpha);
    out[pixel] = pack_rgba(r, g, b, (src_packed >> 24u) & 255u);
}
