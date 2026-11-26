// Stage-fusion probe kernel.
//
// Two integer RGBA8 stages, both quantized back to u8 at every step:
//   A "exposure / white balance": per-channel Q8 gain + signed integer offset
//   B "tone / contrast":          Q8 contrast about mid-grey 128
//
// `op_a` then `op_b` is the serial pair (A writes `mid`, B reads it); `op_ab` is
// the fused single-dispatch version that produces the same bytes without ever
// touching `mid`.
//
// Every stage ends in `(t >> 8)` clamped to 0..255 on a NON-NEGATIVE integer, so
// there is no signed-shift or float-rounding ambiguity anywhere: the fused
// kernel re-quantizes between stages, which makes bit-exactness a property of
// the code, not of the GPU.

struct Packed {
    w: array<u32>,
};

@group(0) @binding(0) var<storage, read>       src : Packed;
@group(0) @binding(1) var<storage, read_write> dst : Packed;
@group(0) @binding(2) var<storage, read_write> mid : Packed;
@group(0) @binding(3) var<storage, read>       prm : Packed;

// prm words:
//   0 width   1 height
//   2 gain_r  3 gain_g  4 gain_b          (Q8: 256 == 1.0)
//   5 off_r   6 off_g   7 off_b          (signed u8 units, stored as bitcast i32)
//   8 contrast                            (Q8: 256 == 1.0)
//   9 strength_percent                    (0 / 50 / 100, informational)
//  10 alpha_policy                        (1 = leave fully transparent pixels untouched)

fn unpack_r(v: u32) -> u32 { return v & 0xffu; }
fn unpack_g(v: u32) -> u32 { return (v >> 8u) & 0xffu; }
fn unpack_b(v: u32) -> u32 { return (v >> 16u) & 0xffu; }
fn unpack_a(v: u32) -> u32 { return (v >> 24u) & 0xffu; }

fn pack_rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return (r & 0xffu) | ((g & 0xffu) << 8u) | ((b & 0xffu) << 16u) | ((a & 0xffu) << 24u);
}

// Stage A: value * gain / 256 + off, round-half-up, clamp to 8 bit.
fn stage_a(c: u32, gain: u32, off: u32) -> u32 {
    let t = i32(c) * i32(gain) + 128 + i32(off) * 256;
    return min(u32(max(t, 0)) >> 8u, 255u);
}

// Stage B: (value - 128) * contrast / 256 + 128, round-half-up, clamp to 8 bit.
fn stage_b(c: u32, contrast: u32) -> u32 {
    let t = (i32(c) - 128) * i32(contrast) + (128 * 256) + 128;
    return min(u32(max(t, 0)) >> 8u, 255u);
}

fn keep_alpha(p: u32) -> bool {
    return !(prm.w[10] == 1u && unpack_a(p) == 0u);
}

// --- unfused kernel 1: stage A only, intermediate image written to `mid` -----
@compute @workgroup_size(8, 8, 1)
fn op_a(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = prm.w[0];
    let h = prm.w[1];
    if (gid.x >= w || gid.y >= h) { return; }
    let i = gid.y * w + gid.x;
    let p = src.w[i];
    var q = p;
    if (keep_alpha(p)) {
        q = pack_rgba(
            stage_a(unpack_r(p), prm.w[2], prm.w[5]),
            stage_a(unpack_g(p), prm.w[3], prm.w[6]),
            stage_a(unpack_b(p), prm.w[4], prm.w[7]),
            unpack_a(p),
        );
    }
    mid.w[i] = q;
}

// --- unfused kernel 2: stage B only, reads the intermediate image ------------
@compute @workgroup_size(8, 8, 1)
fn op_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = prm.w[0];
    let h = prm.w[1];
    if (gid.x >= w || gid.y >= h) { return; }
    let i = gid.y * w + gid.x;
    let p = mid.w[i];
    var q = p;
    if (keep_alpha(p)) {
        q = pack_rgba(
            stage_b(unpack_r(p), prm.w[8]),
            stage_b(unpack_g(p), prm.w[8]),
            stage_b(unpack_b(p), prm.w[8]),
            unpack_a(p),
        );
    }
    dst.w[i] = q;
}

// --- fused kernel: stage A then stage B in ONE dispatch, quantizing between --
@compute @workgroup_size(8, 8, 1)
fn op_ab(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = prm.w[0];
    let h = prm.w[1];
    if (gid.x >= w || gid.y >= h) { return; }
    let i = gid.y * w + gid.x;
    let p = src.w[i];
    var q = p;
    if (keep_alpha(p)) {
        // stage A, quantized to u8 exactly as the standalone kernel would store it
        let a_r = stage_a(unpack_r(p), prm.w[2], prm.w[5]);
        let a_g = stage_a(unpack_g(p), prm.w[3], prm.w[6]);
        let a_b = stage_a(unpack_b(p), prm.w[4], prm.w[7]);
        // stage B applied to the already-quantized stage-A result
        q = pack_rgba(
            stage_b(a_r, prm.w[8]),
            stage_b(a_g, prm.w[8]),
            stage_b(a_b, prm.w[8]),
            unpack_a(p),
        );
    }
    dst.w[i] = q;
}
