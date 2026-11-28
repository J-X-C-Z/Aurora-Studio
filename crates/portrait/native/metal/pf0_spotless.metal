// PF0 spotless — recovered SpotlessProcessNormal fragment.
// Metal mirror of native/wgsl/pf0_spotless.wgsl.
// Source/blur are f32 RGBA/RGB normalized [0,1]; face/skin masks are f32.
// This is a source-level contract candidate; no production host path is wired.
#include <metal_stdlib>
using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

struct SpotlessParams
{
    uint width;
    uint height;
    float dist_factor;
    float dist_threshold;
    float dark_diff_min_threshold;
    float dark_diff_max_threshold;
    float white_diff_factor;
    float degree;
    uint flags;          // bit0: explicit skin mask present
    uint reserved0;
    uint reserved1;
    uint reserved2;
};
static_assert(sizeof(SpotlessParams) == 48, "SpotlessParams must stay 48 bytes");

static inline float sp_step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }
static inline float sp_mix(float x, float y, float a) { return x * (1.0f - a) + y * a; }
static inline float sp_length3(float x, float y, float z) { return sqrt(x * x + y * y + z * z); }

static inline float2 sp_rgb2hsv_hue_v(float r, float g, float b)
{
    float s = sp_step(b, g);
    float px = sp_mix(b, g, s);
    float py = sp_mix(g, b, s);
    float pz = sp_mix(-1.0f, 0.0f, s);
    float pw = sp_mix(2.0f / 3.0f, -1.0f / 3.0f, s);
    float t = sp_step(px, r);
    float qx = sp_mix(px, r, t);
    float qy = py;
    float qz = sp_mix(pw, pz, t);
    float qw = sp_mix(r, px, t);
    float d = qx - min(qw, qy);
    float hue = fabs(qz + (qw - qy) / (6.0f * d + 1.0e-10f));
    return float2(hue, qx);
}

static inline float sp_skin_value(float hue, float value)
{
    float skin = 1.0f;
    if ((0.18f <= hue && hue <= 0.89f) || value <= 0.2f) {
        skin = 0.0f;
    } else if (0.16f < hue && hue < 0.18f) {
        skin = min(skin, (0.18f - hue) * 50.0f);
    } else if (0.89f < hue && hue < 0.91f) {
        skin = min(skin, 1.0f - (0.91f - hue) * 50.0f);
    } else if (0.2f < value && hue < 0.3f) {
        skin = min(skin, 1.0f - (0.3f - value) * 10.0f);
    }
    return skin;
}

kernel void pf0_spotless(
    constant SpotlessParams& p [[buffer(0)]],
    device const float* src [[buffer(1)]],       // RGBA, stride width*4
    device const float* blur [[buffer(2)]],      // RGB,  stride width*3
    device const float* face_mask [[buffer(3)]],
    device const float* skin_mask [[buffer(4)]],
    device float* out [[buffer(5)]],             // RGBA
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    uint pixel = gid.y * p.width + gid.x;
    uint sidx = pixel * 4u;
    uint bidx = pixel * 3u;
    float sr = src[sidx], sg = src[sidx + 1u], sb = src[sidx + 2u], sa = src[sidx + 3u];
    float rr = sr, rg = sg, rb = sb;
    float fm = face_mask[pixel];
    if (fm > 0.0f) {
        float skin = skin_mask[pixel];
        if ((p.flags & 1u) == 0u) {
            float2 hv = sp_rgb2hsv_hue_v(sr, sg, sb);
            skin = sp_skin_value(hv.x, hv.y);
        }
        float factor = min(max(fm * skin * p.degree, 0.0f), 1.0f);
        if (factor > 0.0f) {
            float br = blur[bidx], bg = blur[bidx + 1u], bb = blur[bidx + 2u];
            float dist = sp_length3(max(br - sr, 0.0f), max(bg - sg, 0.0f), max(bb - sb, 0.0f)) * p.dist_factor;
            if (dist > p.dist_threshold) {
                float dr = min(max(br - sr, p.dark_diff_min_threshold), p.dark_diff_max_threshold);
                float dg = min(max(bg - sg, p.dark_diff_min_threshold), p.dark_diff_max_threshold);
                float db = min(max(bb - sb, p.dark_diff_min_threshold), p.dark_diff_max_threshold);
                rr = sp_mix(sr, min(sr + dr, 1.0f), factor);
                rg = sp_mix(sg, min(sg + dg, 1.0f), factor);
                rb = sp_mix(sb, min(sb + db, 1.0f), factor);
            } else {
                float inv_dist = sp_length3(max(sr - br, 0.0f), max(sg - bg, 0.0f), max(sb - bb, 0.0f)) * p.dist_factor;
                if (inv_dist > p.dist_threshold) {
                    rr = sp_mix(sr, br + (sr - br) * p.white_diff_factor, factor);
                    rg = sp_mix(sg, bg + (sg - bg) * p.white_diff_factor, factor);
                    rb = sp_mix(sb, bb + (sb - bb) * p.white_diff_factor, factor);
                }
            }
        }
    }
    out[sidx] = rr;
    out[sidx + 1u] = rg;
    out[sidx + 2u] = rb;
    out[sidx + 3u] = sa;
}
