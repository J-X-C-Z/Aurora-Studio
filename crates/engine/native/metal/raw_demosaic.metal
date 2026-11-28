//
//  raw_demosaic.metal
//  Aurora Studio — cross-platform RAW Bayer demosaic (Apple Metal MSL).
//
//  Twin of `crates/engine/native/wgsl/raw_demosaic.wgsl` and of
//  `demosaic_bayer_cpu_ref` in `crates/engine/src/gpu_crossplatform.rs`.
//  Malvar-He-Cutler (ICASSP 2004) gradient-corrected 5x5 linear interpolation
//  with mirrored borders. This is a high-quality linear reference, not a
//  reconstruction of any vendor demosaic.
//
//  Compiled on macOS only. Nothing on Windows or Linux builds or runs this
//  file, so it is shipped as source for the Metal lane; treat it as
//  IMPLEMENTED, not COMPILED, until a device test has run.
//
//  CFA codes: 0 = red, 1 = green, 2 = blue. The 2x2 phase is packed into one
//  uint32, low nibble first:
//    RGGB = 0x2110  BGGR = 0x0112  GRBG = 0x1201  GBRG = 0x1021
//  Bit-for-bit the same table as the WGSL twin and `CfaPattern::packed`.
//

#include <metal_stdlib>
using namespace metal;

struct DemosaicParams {
    uint2 size;
    uint pattern;
    uint pad;
};

static inline int mirror_index(int i, int n) {
    if (n <= 1) {
        return 0;
    }
    int v = i;
    if (v < 0) {
        v = -v;
    }
    if (v >= n) {
        v = 2 * n - 2 - v;
    }
    if (v < 0) {
        v = 0;
    }
    if (v >= n) {
        v = n - 1;
    }
    return v;
}

static inline uint cfa_packed(uint pattern) {
    if (pattern == 1u) {
        return 0x0112u; // BGGR
    }
    if (pattern == 2u) {
        return 0x1201u; // GRBG
    }
    if (pattern == 3u) {
        return 0x1021u; // GBRG
    }
    return 0x2110u;     // RGGB
}

static inline uint cfa_code(uint x, uint y, uint pattern) {
    uint index = (y & 1u) * 2u + (x & 1u);
    return (cfa_packed(pattern) >> (index * 4u)) & 0x0fu;
}

static inline float tap(device const float *plane, int x, int y, int w, int h) {
    int xx = mirror_index(x, w);
    int yy = mirror_index(y, h);
    return plane[(uint)yy * (uint)w + (uint)xx];
}

// Filter a: green at a red or blue site.
static inline float g_at_rb(device const float *p, int x, int y, int w, int h) {
    float c = tap(p, x, y, w, h);
    float n = tap(p, x, y - 1, w, h);
    float s = tap(p, x, y + 1, w, h);
    float e = tap(p, x + 1, y, w, h);
    float west = tap(p, x - 1, y, w, h);
    float n2 = tap(p, x, y - 2, w, h);
    float s2 = tap(p, x, y + 2, w, h);
    float e2 = tap(p, x + 2, y, w, h);
    float w2 = tap(p, x - 2, y, w, h);
    return (4.0f * c + 2.0f * (n + s + e + west) - (n2 + s2 + e2 + w2)) * 0.125f;
}

// Filter b: red/blue at a green site whose target colour lies horizontally.
static inline float rb_at_g_row(device const float *p, int x, int y, int w, int h) {
    float c = tap(p, x, y, w, h);
    float e = tap(p, x + 1, y, w, h);
    float west = tap(p, x - 1, y, w, h);
    float e2 = tap(p, x + 2, y, w, h);
    float w2 = tap(p, x - 2, y, w, h);
    float nw = tap(p, x - 1, y - 1, w, h);
    float ne = tap(p, x + 1, y - 1, w, h);
    float sw = tap(p, x - 1, y + 1, w, h);
    float se = tap(p, x + 1, y + 1, w, h);
    float n2 = tap(p, x, y - 2, w, h);
    float s2 = tap(p, x, y + 2, w, h);
    return (5.0f * c + 4.0f * (e + west) - (e2 + w2)
            - (nw + ne + sw + se) + 0.5f * (n2 + s2)) * 0.125f;
}

// Filter c: the transposed twin of filter b.
static inline float rb_at_g_col(device const float *p, int x, int y, int w, int h) {
    float c = tap(p, x, y, w, h);
    float n = tap(p, x, y - 1, w, h);
    float s = tap(p, x, y + 1, w, h);
    float n2 = tap(p, x, y - 2, w, h);
    float s2 = tap(p, x, y + 2, w, h);
    float nw = tap(p, x - 1, y - 1, w, h);
    float ne = tap(p, x + 1, y - 1, w, h);
    float sw = tap(p, x - 1, y + 1, w, h);
    float se = tap(p, x + 1, y + 1, w, h);
    float e2 = tap(p, x + 2, y, w, h);
    float w2 = tap(p, x - 2, y, w, h);
    return (5.0f * c + 4.0f * (n + s) - (n2 + s2)
            - (nw + ne + sw + se) + 0.5f * (e2 + w2)) * 0.125f;
}

// Filter d: red at blue and blue at red.
static inline float rb_at_opposite(device const float *p, int x, int y, int w, int h) {
    float c = tap(p, x, y, w, h);
    float nw = tap(p, x - 1, y - 1, w, h);
    float ne = tap(p, x + 1, y - 1, w, h);
    float sw = tap(p, x - 1, y + 1, w, h);
    float se = tap(p, x + 1, y + 1, w, h);
    float n2 = tap(p, x, y - 2, w, h);
    float s2 = tap(p, x, y + 2, w, h);
    float e2 = tap(p, x + 2, y, w, h);
    float w2 = tap(p, x - 2, y, w, h);
    return (6.0f * c + 2.0f * (nw + ne + sw + se) - 1.5f * (n2 + s2 + e2 + w2)) * 0.125f;
}

kernel void demosaic_bayer(device const float *bayer [[buffer(0)]],
                           device float4 *rgba [[buffer(1)]],
                           constant DemosaicParams &params [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    int w = (int)params.size.x;
    int h = (int)params.size.y;
    int x = (int)gid.x;
    int y = (int)gid.y;
    if (x >= w || y >= h) {
        return;
    }
    uint code = cfa_code(gid.x, gid.y, params.pattern);
    float center = tap(bayer, x, y, w, h);

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    if (code == 1u) {
        g = center;
        float across = rb_at_g_row(bayer, x, y, w, h);
        float down = rb_at_g_col(bayer, x, y, w, h);
        if (cfa_code(gid.x + 1u, gid.y, params.pattern) == 0u) {
            r = across;
            b = down;
        } else {
            b = across;
            r = down;
        }
    } else if (code == 0u) {
        r = center;
        g = g_at_rb(bayer, x, y, w, h);
        b = rb_at_opposite(bayer, x, y, w, h);
    } else {
        b = center;
        g = g_at_rb(bayer, x, y, w, h);
        r = rb_at_opposite(bayer, x, y, w, h);
    }

    uint index = gid.y * params.size.x + gid.x;
    rgba[index] = float4(r, g, b, 1.0f);
}
