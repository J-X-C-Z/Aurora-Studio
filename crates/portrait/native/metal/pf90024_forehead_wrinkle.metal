// PF90024 forehead wrinkle — FacialSmoothProcessV2 composite.
// Metal mirror of native/wgsl/pf90023_eye_wrinkle.wgsl.
#include <metal_stdlib>
using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

struct WrinkleParams
{
    uint width;
    uint height;
    float degree;
    float gate;
    float legal_alpha;
    float dark_circle_alpha;
    float diff_factor;
    float diff_factor2;
    float color_threshold_factor;
    float mix_alpha;
    float diff_min_threshold;
    float diff_max_threshold;
    float high_freq_min_threshold;
    float high_freq_max_threshold;
};
static_assert(sizeof(WrinkleParams) == 56, "WrinkleParams must stay 56 bytes");

static inline float wr_mix(float x, float y, float a) { return x * (1.0f - a) + y * a; }

kernel void pf90024_forehead_wrinkle(
    constant WrinkleParams& p [[buffer(0)]],
    device const float4* source [[buffer(1)]],
    device const float4* base [[buffer(2)]],
    device const float4* residual [[buffer(3)]],
    device const float4* mask_rgba [[buffer(4)]],
    device const float* skin_mask [[buffer(5)]],
    device float4* out [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    uint pixel = gid.y * p.width + gid.x;
    float4 src = source[pixel];
    if (p.degree <= p.gate) {
        out[pixel] = src;
        return;
    }
    float4 mask = mask_rgba[pixel];
    float alpha = fmax(fmax(p.dark_circle_alpha * mask.x, p.legal_alpha * mask.z), 0.0f) * skin_mask[pixel];
    if (alpha <= 0.0f) {
        out[pixel] = src;
        return;
    }
    float4 b = base[pixel];
    float4 r = residual[pixel];
    float3 result = src.xyz;
    for (uint ch = 0u; ch < 3u; ++ch) {
        float high = src[ch] - b[ch];
        float diff = clamp((r[ch] - b[ch]) * p.diff_factor + r[ch] * p.diff_factor2,
                           p.diff_min_threshold, p.diff_max_threshold);
        float threshold = wr_mix(r[ch] * p.color_threshold_factor, r[ch], p.mix_alpha);
        float upper = fmax(threshold, src[ch] - high);
        float reference = clamp(src[ch] + diff - high, 0.0f, upper) + high;
        float delta = clamp(wr_mix(src[ch], reference, alpha) - src[ch],
                            p.high_freq_min_threshold, p.high_freq_max_threshold);
        result[ch] = src[ch] + delta;
    }
    out[pixel] = float4(result, src.w);
}
