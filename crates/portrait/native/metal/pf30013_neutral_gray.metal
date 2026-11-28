// PF30013 neutral gray skin flattening (cuda_neutral_gray.cu skin_grey_kernel).
// Metal mirror of native/wgsl/pf30013_neutral_gray.wgsl.
#include <metal_stdlib>
using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

struct NeutralGrayParams
{
    uint width;
    uint height;
    float alpha;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};
static_assert(sizeof(NeutralGrayParams) == 24, "NeutralGrayParams must stay 24 bytes");

constant float HSL_EPSILON = 0.001f;
constant float INV255 = 0.0039215686274509803f;

static inline uint pack_rgba(uint r, uint g, uint b, uint a)
{
    return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

struct Hsl { float h; float s; float l; };

static inline Hsl rgb_to_hsl(float r, float g, float b)
{
    float base_max = g < b ? b : g;
    float maximum = fmax(r, base_max);
    float minimum = fmin(r, fmin(g, b));
    float chroma = maximum - minimum;
    float w = fmin(r, base_max);
    float offset = r < base_max
        ? ((g < b) ? (2.0f / 3.0f) : (-1.0f / 3.0f))
        : ((g < b) ? -1.0f : 0.0f);
    float h = fabs((w - minimum) / (6.0f * chroma + HSL_EPSILON) + offset);
    float l = maximum - chroma * 0.5f;
    float s = chroma / (1.0f - fabs(l * 2.0f - 1.0f) + HSL_EPSILON);
    return Hsl{h, s, l};
}

static inline float hsl_axis(float value, float chroma, float l)
{
    return clamp((value - 0.5f) * chroma + l, 0.0f, 1.0f);
}

static inline float3 hsl_to_rgb(float h, float s, float l)
{
    float x = h * 6.0f;
    float chroma = (1.0f - fabs(l * 2.0f - 1.0f)) * s;
    return float3(
        hsl_axis(clamp(fabs(x - 3.0f) - 1.0f, 0.0f, 1.0f), chroma, l),
        hsl_axis(clamp(2.0f - fabs(x - 2.0f), 0.0f, 1.0f), chroma, l),
        hsl_axis(clamp(2.0f - fabs(x - 4.0f), 0.0f, 1.0f), chroma, l));
}

static inline uint round_byte(float value)
{
    return clamp(uint(round(clamp(value, 0.0f, 1.0f) * 255.0f)), 0u, 255u);
}

kernel void pf30013_neutral_gray(
    constant NeutralGrayParams& p [[buffer(0)]],
    device const uint* original [[buffer(1)]],
    device const uint* current [[buffer(2)]],
    device const uint* mask [[buffer(3)]],
    device uint* out [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    uint pixel = gid.y * p.width + gid.x;
    uint cur = current[pixel];
    float weight = clamp(float(mask[pixel] & 255u) * INV255 * p.alpha, 0.0f, 1.0f);
    if (weight <= HSL_EPSILON) {
        out[pixel] = cur;
        return;
    }
    uint orig = original[pixel];
    Hsl os = rgb_to_hsl(float(orig & 255u) * INV255, float((orig >> 8u) & 255u) * INV255, float((orig >> 16u) & 255u) * INV255);
    Hsl cs = rgb_to_hsl(float(cur & 255u) * INV255, float((cur >> 8u) & 255u) * INV255, float((cur >> 16u) & 255u) * INV255);
    float3 projected = hsl_to_rgb(os.h, os.s, cs.l);
    float3 now = float3(float(cur & 255u), float((cur >> 8u) & 255u), float((cur >> 16u) & 255u)) * INV255;
    float3 mixed = now + (projected - now) * weight;
    out[pixel] = pack_rgba(round_byte(mixed.x), round_byte(mixed.y), round_byte(mixed.z), (cur >> 24u) & 255u);
}
