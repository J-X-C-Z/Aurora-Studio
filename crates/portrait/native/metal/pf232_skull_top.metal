// PF232 skull top / skullEnhance — recovered 15-point radial warp.
// Metal mirror of native/wgsl/pf232_skull_top.wgsl.
#include <metal_stdlib>
using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

constant uint CONTOUR_POINTS = 15u;

struct SkullTopParams
{
    uint width;
    uint height;
    uint roi_x;
    uint roi_y;
    uint roi_width;
    uint roi_height;
    uint reserved0;
    uint reserved1;
};
static_assert(sizeof(SkullTopParams) == 32, "SkullTopParams must stay 32 bytes");

static inline uint pack_rgba(uint r, uint g, uint b, uint a)
{
    return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

static inline uint sample_bilinear(
    constant SkullTopParams& p,
    device const uint* source,
    float x, float y)
{
    float fx = clamp(x, 0.0f, float(p.width - 1u));
    float fy = clamp(y, 0.0f, float(p.height - 1u));
    uint x0 = uint(floor(fx));
    uint y0 = uint(floor(fy));
    uint x1 = min(x0 + 1u, p.width - 1u);
    uint y1 = min(y0 + 1u, p.height - 1u);
    float tx = fx - float(x0);
    float ty = fy - float(y0);
    uint a = source[y0 * p.width + x0];
    uint b = source[y0 * p.width + x1];
    uint c = source[y1 * p.width + x0];
    uint d = source[y1 * p.width + x1];
    uint rgb[3];
    for (uint ch = 0u; ch < 3u; ++ch) {
        uint shift = ch * 8u;
        float av = float((a >> shift) & 255u);
        float bv = float((b >> shift) & 255u);
        float cv = float((c >> shift) & 255u);
        float dv = float((d >> shift) & 255u);
        float top = av * (1.0f - tx) + bv * tx;
        float bottom = cv * (1.0f - tx) + dv * tx;
        rgb[ch] = clamp(uint(round(top * (1.0f - ty) + bottom * ty)), 0u, 255u);
    }
    return pack_rgba(rgb[0], rgb[1], rgb[2], (a >> 24u) & 255u);
}

kernel void pf232_skull_top(
    constant SkullTopParams& p [[buffer(0)]],
    device const uint* source [[buffer(1)]],
    constant float2& anchor [[buffer(2)]],
    constant float2* contour_source [[buffer(3)]],
    constant float2* contour_warped [[buffer(4)]],
    device uint* out [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    uint pixel = gid.y * p.width + gid.x;
    uint result = source[pixel];
    if (gid.x < p.roi_x || gid.x >= p.roi_x + p.roi_width ||
        gid.y < p.roi_y || gid.y >= p.roi_y + p.roi_height) {
        out[pixel] = result;
        return;
    }
    float px = float(gid.x) + 0.5f;
    float py = float(gid.y) + 0.5f;
    float dx = px - anchor.x;
    float dy = py - anchor.y;
    float radius = sqrt(dx * dx + dy * dy);
    if (radius < 1.0e-6f) {
        out[pixel] = result;
        return;
    }
    float2 dir = float2(dx / radius, dy / radius);
    uint best = 0u;
    float best_dot = -FLT_MAX;
    for (uint i = 0u; i < CONTOUR_POINTS; ++i) {
        float2 original = contour_source[i];
        float2 warped = contour_warped[i];
        float2 v = (warped + original) * 0.5f - anchor;
        float norm = sqrt(v.x * v.x + v.y * v.y);
        float dot = -FLT_MAX;
        if (norm > 1.0e-6f) {
            dot = (dir.x * v.x + dir.y * v.y) / norm;
        }
        if (dot > best_dot) {
            best_dot = dot;
            best = i;
        }
    }
    float2 warped_point = contour_warped[best];
    float2 source_point = contour_source[best];
    float warped_radius = max(dot(warped_point - anchor, dir), 1.0e-4f);
    float source_radius = max(dot(source_point - anchor, dir), 1.0e-4f);
    float mapped_radius = radius * source_radius / warped_radius;
    out[pixel] = sample_bilinear(p, source, anchor.x + dir.x * mapped_radius, anchor.y + dir.y * mapped_radius);
}
