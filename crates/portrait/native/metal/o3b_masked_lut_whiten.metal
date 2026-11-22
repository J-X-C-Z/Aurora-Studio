// o3 isolated Metal compute skeleton.
// This is a source-level contract only. It is not built or executed on Apple hardware.
#include <metal_stdlib>

using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

// All values are little-endian scalar fields with the offsets in contract/layout.json.
struct O3MaskProjectionParams {
    int source_mask_width;
    int source_mask_height;
    int surface_width;
    int surface_height;
    int destination_x;
    int destination_y;
    int destination_width;
    int destination_height;
    int source_rect_x;
    int source_rect_y;
    int source_rect_width;
    int source_rect_height;
    int domain_x0;
    int domain_y0;
    int roi_width;
    int roi_height;
    uint flags;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};
static_assert(sizeof(O3MaskProjectionParams) == 80, "O3 projection ABI must stay 80 bytes");

struct O3MaskedWhitenParams {
    uint surface_width;
    uint surface_height;
    uint mask_x;
    uint mask_y;
    uint mask_width;
    uint mask_height;
    uint pixel_count;
    uint flags;
    float alpha;
    float reserved0;
    float reserved1;
    float reserved2;
};
static_assert(sizeof(O3MaskedWhitenParams) == 48, "O3 whiten ABI must stay 48 bytes");

// Exact signed floor/ceil division for the integral projection case. The first
// implementation is deliberately restricted to integral ROI placement because
// the non-integral vendor rounding law is open.
static inline long o3_floor_div(long numerator, long denominator) {
    long quotient = numerator / denominator;
    if ((numerator % denominator) != 0 && ((numerator < 0) != (denominator < 0))) {
        quotient -= 1;
    }
    return quotient;
}

static inline long o3_ceil_div(long numerator, long denominator) {
    long quotient = numerator / denominator;
    if ((numerator % denominator) != 0 && ((numerator < 0) == (denominator < 0))) {
        quotient += 1;
    }
    return quotient;
}

static inline int o3_floor_edge(int edge, int domain0, int roi_extent, int surface_extent) {
    long numerator = (long)domain0 * (long)surface_extent
                   + (long)edge * (long)roi_extent;
    return (int)o3_floor_div(numerator, (long)surface_extent);
}

static inline int o3_ceil_edge(int edge, int domain0, int roi_extent, int surface_extent) {
    long numerator = (long)domain0 * (long)surface_extent
                   + (long)edge * (long)roi_extent;
    return (int)o3_ceil_div(numerator, (long)surface_extent);
}

static inline int o3_centre_edge(int edge, int domain0, int roi_extent, int surface_extent) {
    long numerator = 2L * (long)domain0 * (long)surface_extent
                   + (2L * (long)edge + 1L) * (long)roi_extent;
    return (int)o3_floor_div(numerator, 2L * (long)surface_extent);
}

// Coverage-max projection:
// one destination mask byte = max(source bytes intersecting its mapped footprint).
// destination_x/y are surface coordinates; the kernel writes only destination_width
// by destination_height bytes and never reads the source plane outside source_rect.
kernel void o3_coverage_max_mask_projection(
    device const uchar *source_mask [[buffer(0)]],
    device uchar *destination_mask [[buffer(1)]],
    constant O3MaskProjectionParams &p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= (uint)p.destination_width || gid.y >= (uint)p.destination_height) {
        return;
    }

    int gx = p.destination_x + (int)gid.x;
    int gy = p.destination_y + (int)gid.y;

    int vx_lo = o3_floor_edge(gx, p.domain_x0, p.roi_width, p.surface_width);
    int vx_hi = o3_ceil_edge(gx + 1, p.domain_x0, p.roi_width, p.surface_width) - 1;
    int vy_lo = o3_floor_edge(gy, p.domain_y0, p.roi_height, p.surface_height);
    int vy_hi = o3_ceil_edge(gy + 1, p.domain_y0, p.roi_height, p.surface_height) - 1;

    vx_lo = max(vx_lo, p.source_rect_x);
    vy_lo = max(vy_lo, p.source_rect_y);
    vx_hi = min(vx_hi, p.source_rect_x + p.source_rect_width - 1);
    vy_hi = min(vy_hi, p.source_rect_y + p.source_rect_height - 1);

    // An upscaled footprint can contain no source cell. Use the same
    // centre fallback as the CPU candidate instead of inventing coverage.
    if (vx_lo > vx_hi) {
        int centre = o3_centre_edge(gx, p.domain_x0, p.roi_width, p.surface_width);
        centre = clamp(centre, p.source_rect_x, p.source_rect_x + p.source_rect_width - 1);
        vx_lo = centre;
        vx_hi = centre;
    }
    if (vy_lo > vy_hi) {
        int centre = o3_centre_edge(gy, p.domain_y0, p.roi_height, p.surface_height);
        centre = clamp(centre, p.source_rect_y, p.source_rect_y + p.source_rect_height - 1);
        vy_lo = centre;
        vy_hi = centre;
    }

    uchar value = (uchar)0;
    for (int vy = vy_lo; vy <= vy_hi; ++vy) {
        int source_row = (vy - p.source_rect_y) * p.source_mask_width;
        for (int vx = vx_lo; vx <= vx_hi; ++vx) {
            uchar sample = source_mask[source_row + (vx - p.source_rect_x)];
            value = max(value, sample);
        }
    }

    destination_mask[(int)gid.y * p.destination_width + (int)gid.x] = value;
}

static inline float o3_clamp01(float value) {
    return value <= 0.0f ? 0.0f : (value >= 1.0f ? 1.0f : value);
}

static inline float3 o3_sample_lut(
    device const float4 *lut,
    float3 color)
{
    float pr = o3_clamp01(color.x) * 63.0f;
    float pg = o3_clamp01(color.y) * 63.0f;
    float pb = o3_clamp01(color.z) * 63.0f;
    int ir = (int)pr;
    int ig = (int)pg;
    int ib = (int)pb;
    float fr = pr - (float)ir;
    float fg = pg - (float)ig;
    float fb = pb - (float)ib;
    int jr = min(ir + 1, 63);
    int jg = min(ig + 1, 63);
    int jb = min(ib + 1, 63);
    float3 result = float3(0.0f, 0.0f, 0.0f);

    for (int bz = 0; bz < 2; ++bz) {
        int z = bz == 0 ? ib : jb;
        float wz = bz == 0 ? (1.0f - fb) : fb;
        int y_base = (z >> 3) * 64;
        int x_base = (z & 7) * 64;
        for (int gy = 0; gy < 2; ++gy) {
            int y = y_base + (gy == 0 ? ig : jg);
            float wy = gy == 0 ? (1.0f - fg) : fg;
            for (int gx = 0; gx < 2; ++gx) {
                int x = x_base + (gx == 0 ? ir : jr);
                float wx = gx == 0 ? (1.0f - fr) : fr;
                float weight = wz * wy * wx;
                float4 texel = lut[y * 512 + x];
                result += float3(texel.x, texel.y, texel.z) * weight;
            }
        }
    }
    return result;
}

static inline uchar o3_round_half_even(float value) {
    float scaled = value * 255.0f;
    if (!(scaled > 0.0f)) {
        return (uchar)0;
    }
    if (scaled >= 255.0f) {
        return (uchar)255;
    }
    float base_value = floor(scaled);
    uint base = (uint)base_value;
    float fraction = scaled - base_value;
    if (fraction > 0.5f || (fraction == 0.5f && ((base & 1u) != 0u))) {
        base += 1u;
    }
    return (uchar)min(base, 255u);
}

// Masked 3D-LUT whiten:
// BGRA8 source -> BGRA8 destination, with alpha copied byte-for-byte.
// The LUT is an RGBA float4 atlas: 512x512, 8x8 tiles of 64x64, R/G/B/A.
kernel void o3_masked_lut_whiten(
    device const uchar *source_bgra [[buffer(0)]],
    device uchar *destination_bgra [[buffer(1)]],
    device const uchar *mask_plane [[buffer(2)]],
    device const float4 *lut_atlas [[buffer(3)]],
    constant O3MaskedWhitenParams &p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.surface_width || gid.y >= p.surface_height) {
        return;
    }
    uint pixel = gid.y * p.surface_width + gid.x;
    if (pixel >= p.pixel_count) {
        return;
    }

    uint index = pixel * 4u;
    uchar b = source_bgra[index + 0u];
    uchar g = source_bgra[index + 1u];
    uchar r = source_bgra[index + 2u];
    uchar a = source_bgra[index + 3u];
    destination_bgra[index + 0u] = b;
    destination_bgra[index + 1u] = g;
    destination_bgra[index + 2u] = r;
    destination_bgra[index + 3u] = a;

    if (gid.x < p.mask_x || gid.y < p.mask_y
        || gid.x >= p.mask_x + p.mask_width
        || gid.y >= p.mask_y + p.mask_height) {
        return;
    }

    uint mask_index = (gid.y - p.mask_y) * p.mask_width + (gid.x - p.mask_x);
    float weight = (float)mask_plane[mask_index] / 255.0f;
    float alpha = weight * p.alpha;
    if (!(alpha > 0.001f)) {
        return;
    }

    float3 source_color = float3((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);
    float3 lut_color = o3_sample_lut(lut_atlas, source_color);
    float3 mixed = source_color * (1.0f - alpha) + lut_color * alpha;
    destination_bgra[index + 0u] = o3_round_half_even(mixed.z);
    destination_bgra[index + 1u] = o3_round_half_even(mixed.y);
    destination_bgra[index + 2u] = o3_round_half_even(mixed.x);
    // alpha at index + 3 is intentionally never rewritten here.
}
