// Aurora Studio — cross-platform RAW Bayer demosaic (WGSL).
//
// Twin of `demosaic_bayer_cpu_ref` in `crates/engine/src/gpu_crossplatform.rs`.
// Both are Malvar-He-Cutler (ICASSP 2004) gradient-corrected 5x5 linear
// interpolation with mirrored borders. Neither one is a claim about Capture
// One's own demosaic: this is a high-quality linear reference, not the vendor
// algorithm.
//
// Input is one linear f32 sample per photosite. Black-level subtraction and
// normalisation happen in the caller (`aurora_raw` hands over camera-linear
// samples); this shader only interpolates. Output is RGBA f32 with alpha 1.0.
//
// CFA codes: 0 = red, 1 = green, 2 = blue.
//
// The 2x2 phase of a pattern is packed into one u32, low nibble first:
//   nibble 0 = (x even, y even)   nibble 1 = (x odd, y even)
//   nibble 2 = (x even, y odd)    nibble 3 = (x odd, y odd)
//   RGGB = 0x2110  BGGR = 0x0112  GRBG = 0x1201  GBRG = 0x1021
// Keep in sync with `CfaPattern::packed` in the Rust twin.

struct DemosaicParams {
    size: vec2<u32>,
    pattern: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> demosaic_params: DemosaicParams;
@group(0) @binding(1) var<storage, read> bayer_plane: array<f32>;
@group(0) @binding(2) var<storage, read_write> rgba_out: array<vec4<f32>>;

// Mirror at the frame edge, the same rule the CPU reference uses: -1 -> 1,
// n -> n-2, then clamp. A single reflection covers the +/-2 taps of a 5x5
// kernel for every frame of width or height >= 2; a 1-pixel axis pins to 0.
fn mirror_index(i: i32, n: i32) -> i32 {
    if (n <= 1) {
        return 0;
    }
    var v = i;
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

fn cfa_packed(pattern: u32) -> u32 {
    if (pattern == 1u) {
        return 0x0112u; // BGGR
    }
    if (pattern == 2u) {
        return 0x1201u; // GRBG
    }
    if (pattern == 3u) {
        return 0x1021u; // GBRG
    }
    return 0x2110u; // RGGB, the default phase
}

fn cfa_code(x: u32, y: u32) -> u32 {
    let index = (y & 1u) * 2u + (x & 1u);
    return (cfa_packed(demosaic_params.pattern) >> (index * 4u)) & 0xfu;
}

fn tap(x: i32, y: i32, w: i32, h: i32) -> f32 {
    let xx = mirror_index(x, w);
    let yy = mirror_index(y, h);
    return bayer_plane[u32(yy) * u32(w) + u32(xx)];
}

// Green at a red or blue site (filter a):
//   4*C + 2*(N + S + E + W) - (N2 + S2 + E2 + W2), all over 8.
fn g_at_rb(x: i32, y: i32, w: i32, h: i32) -> f32 {
    let c = tap(x, y, w, h);
    let n = tap(x, y - 1, w, h);
    let s = tap(x, y + 1, w, h);
    let e = tap(x + 1, y, w, h);
    let west = tap(x - 1, y, w, h);
    let n2 = tap(x, y - 2, w, h);
    let s2 = tap(x, y + 2, w, h);
    let e2 = tap(x + 2, y, w, h);
    let w2 = tap(x - 2, y, w, h);
    return (4.0 * c + 2.0 * (n + s + e + west) - (n2 + s2 + e2 + w2)) * 0.125;
}

// Red/blue at a green site whose target colour lies horizontally (filter b):
//   5*C + 4*(E + W) - (E2 + W2) - (NW + NE + SW + SE) + 0.5*(N2 + S2).
fn rb_at_g_row(x: i32, y: i32, w: i32, h: i32) -> f32 {
    let c = tap(x, y, w, h);
    let e = tap(x + 1, y, w, h);
    let west = tap(x - 1, y, w, h);
    let e2 = tap(x + 2, y, w, h);
    let w2 = tap(x - 2, y, w, h);
    let nw = tap(x - 1, y - 1, w, h);
    let ne = tap(x + 1, y - 1, w, h);
    let sw = tap(x - 1, y + 1, w, h);
    let se = tap(x + 1, y + 1, w, h);
    let n2 = tap(x, y - 2, w, h);
    let s2 = tap(x, y + 2, w, h);
    return (5.0 * c + 4.0 * (e + west) - (e2 + w2)
        - (nw + ne + sw + se) + 0.5 * (n2 + s2)) * 0.125;
}

// The transposed twin of `rb_at_g_row` (filter c): the target colour lies
// vertically instead.
fn rb_at_g_col(x: i32, y: i32, w: i32, h: i32) -> f32 {
    let c = tap(x, y, w, h);
    let n = tap(x, y - 1, w, h);
    let s = tap(x, y + 1, w, h);
    let n2 = tap(x, y - 2, w, h);
    let s2 = tap(x, y + 2, w, h);
    let nw = tap(x - 1, y - 1, w, h);
    let ne = tap(x + 1, y - 1, w, h);
    let sw = tap(x - 1, y + 1, w, h);
    let se = tap(x + 1, y + 1, w, h);
    let e2 = tap(x + 2, y, w, h);
    let w2 = tap(x - 2, y, w, h);
    return (5.0 * c + 4.0 * (n + s) - (n2 + s2)
        - (nw + ne + sw + se) + 0.5 * (e2 + w2)) * 0.125;
}

// Red at blue and blue at red (filter d), symmetric in the two colours:
//   6*C + 2*(NW + NE + SW + SE) - 1.5*(N2 + S2 + E2 + W2).
fn rb_at_opposite(x: i32, y: i32, w: i32, h: i32) -> f32 {
    let c = tap(x, y, w, h);
    let nw = tap(x - 1, y - 1, w, h);
    let ne = tap(x + 1, y - 1, w, h);
    let sw = tap(x - 1, y + 1, w, h);
    let se = tap(x + 1, y + 1, w, h);
    let n2 = tap(x, y - 2, w, h);
    let s2 = tap(x, y + 2, w, h);
    let e2 = tap(x + 2, y, w, h);
    let w2 = tap(x - 2, y, w, h);
    return (6.0 * c + 2.0 * (nw + ne + sw + se) - 1.5 * (n2 + s2 + e2 + w2)) * 0.125;
}

@compute @workgroup_size(8, 8)
fn demosaic_bayer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = i32(demosaic_params.size.x);
    let h = i32(demosaic_params.size.y);
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= w || y >= h) {
        return;
    }
    let code = cfa_code(gid.x, gid.y);
    let center = tap(x, y, w, h);

    var r = 0.0;
    var g = 0.0;
    var b = 0.0;
    if (code == 1u) {
        // Green site: the row neighbours carry one colour, the column
        // neighbours the other. Only the CFA phase says which is which.
        g = center;
        let across = rb_at_g_row(x, y, w, h);
        let down = rb_at_g_col(x, y, w, h);
        if (cfa_code(gid.x + 1u, gid.y) == 0u) {
            r = across;
            b = down;
        } else {
            b = across;
            r = down;
        }
    } else if (code == 0u) {
        r = center;
        g = g_at_rb(x, y, w, h);
        b = rb_at_opposite(x, y, w, h);
    } else {
        b = center;
        g = g_at_rb(x, y, w, h);
        r = rb_at_opposite(x, y, w, h);
    }

    let index = gid.y * demosaic_params.size.x + gid.x;
    rgba_out[index] = vec4<f32>(r, g, b, 1.0);
}
