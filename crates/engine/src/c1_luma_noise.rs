//! Capture One `CImgOpLumaNoise2018` + `CImgOpLumaTransfer` reference formulas.
//!
//! Evidence: current `ImgCoreDll.dll` SHA-256
//! `ca02b6ac2786e9a100f63221df7340e5df210416eb0bd53a4bb1bfc2306a44a4`,
//! OpenCL program `p_041588192.cl` (48,492 B, file offset 41,624,656), 14
//! variants; `BoxFilter` `p_040863312.cl` (5,771 B, file offset 40,900,016).
//!
//! Export queue order (DSC_7444, current DLL):
//! `0 -> 10 -> angleParamBuf upload -> 8 -> 11 -> BoxFilter h/v x4 ->
//!  4 -> 5 -> 6 -> LumaTransfer`
//!
//! `LumaTransfer` is a separate op, not variant 0. Its OpenCL source is a
//! 1,431-byte program at VA `0x1827bfe90` / file offset 41,675,408, SHA-256
//! `18d4138b1520ee318abee203e59a521500418f5d0a6f4b17716840958e01dba9`
//! (unique in the current DLL, referenced from the constructor at
//! `0x1817834a7`). The CL kernel, the CPU scalar fallback `0x181782d50` and
//! the two vector fallbacks `0x1817836b0` / `0x1817839a0` all implement the
//! same formula:
//!
//! ```text
//! L = dot(rgb, weights)
//! s = min(1.5, max(target, 1.0) / max(L, 1.0))
//! out.rgb = rgb * s + (target - L * s)
//! out.a   = 0
//! ```
//!
//! The three weights are read by the op wrapper from parameter-object fields
//! `+0x89c/+0x8a0/+0x8a4`; their upstream producer is not yet located.
//!
//! `LN_AngleParamBuf` is **not** a static vendor table. `CImgOpLumaNoise2018`
//! `SetParameters` at `0x181775100` appends eight 20-byte direction records
//! to `this+0xe0` and then generates the eight 24-byte entries
//! (`offsets[16] + mode + scale`) at `this+0x80`; `ProcessBlock` at
//! `0x18177bc50` copies them into the 0x310-byte OpenCL constant buffer and
//! fills `taps[64]` with the runtime list of valid block offsets (geometry
//! and mask dependent; the consumer only unrolls indices 0..49).
//! `angle_entries_vendor` below reproduces the exact generator rules.
//!
//! Not reimplemented here: the 8/16-point DCT helper bodies, the producer of
//! the per-pixel uchar angle-label plane, and the host constants that map ISO
//! to radius/threshold/strength.

/// Variant 0 (`LumaNoise2018`): `z = a*L + 3/8 a^2 + b^2 - a*c`;
/// `out = 2/a * sqrt(max(z,0))`.
///
/// `a` is the Poisson/gain term, `b` the read-noise floor, `c` the black
/// pivot. This is a variance-stabilising transform, not the `LumaTransfer`
/// op. The kernel does not guard `a == 0`; the guard below is a Rust safety
/// extension that returns `max(L, 0)` instead of dividing by zero.
pub fn luma_transfer_v0(luma: f32, a: f32, b: f32, c: f32) -> f32 {
    if a <= 0.0 || !a.is_finite() {
        return luma.max(0.0);
    }
    let z = a * luma + 0.375 * a * a + b * b - a * c;
    (2.0 / a) * z.max(0.0).sqrt()
}

/// Variant 5 (`LumaNoise2018`): inverse of variant 0.
///
/// ```text
/// z = max(value,0) * a/2
/// z = z^2 - (3/8 a^2 + b^2 - a*c)
/// out = max(0, z / a)
/// ```
///
/// The kernel has no `a == 0` guard; returning 0 is a Rust safety extension.
pub fn luma_transfer_v5_inverse(value: f32, a: f32, b: f32, c: f32) -> f32 {
    if a <= 0.0 || !a.is_finite() {
        return 0.0;
    }
    let z = value.max(0.0) * (a * 0.5);
    let z = z * z - (0.375 * a * a + b * b - a * c);
    (z * (1.0 / a)).max(0.0)
}

/// `LumaTransfer` kernel constants (CL passes `float4(1.0, w0, w1, w2)`).
pub const LUMATRANSFER_EPS: f32 = 1.0;
/// Maximum chroma gain (`const float zbfb6939efb = 1.5f`).
pub const LUMATRANSFER_MAX_GAIN: f32 = 1.5;

/// `LumaTransfer` op: transfer a target luma onto RGB while scaling chroma.
///
/// `rgb` is the source colour, `target` the fourth channel (the target luma),
/// `weights` the three luma coefficients, `eps` the `max(.,1.0)` floor and
/// `max_gain` the 1.5 chroma-gain cap. The CL kernel writes `out.a = 0`;
/// only the RGB result is returned here.
pub fn luma_transfer_op(
    rgb: [f32; 3],
    target: f32,
    weights: [f32; 3],
    eps: f32,
    max_gain: f32,
) -> [f32; 3] {
    let luma = rgb[0] * weights[0] + rgb[1] * weights[1] + rgb[2] * weights[2];
    let scale = (target.max(eps) / luma.max(eps)).min(max_gain);
    let offset = target - luma * scale;
    [
        rgb[0] * scale + offset,
        rgb[1] * scale + offset,
        rgb[2] * scale + offset,
    ]
}

/// `LumaTransfer` with the CL defaults `eps = 1.0`, `max_gain = 1.5`.
pub fn luma_transfer_op_default(rgb: [f32; 3], target: f32, weights: [f32; 3]) -> [f32; 3] {
    luma_transfer_op(rgb, target, weights, LUMATRANSFER_EPS, LUMATRANSFER_MAX_GAIN)
}

/// Variant 10: store `(luma*coef, coef)` for the weighted box filter.
pub fn value_weight_pair(luma: f32, coef: f32) -> (f32, f32) {
    (luma * coef, coef)
}

/// Variant 11: normalise the accumulated pair (`value / weight`).
///
/// The kernel divides unconditionally; the epsilon guard is a Rust safety
/// extension (the host's weight is positive).
pub fn normalise_pair(value: f32, weight: f32) -> f32 {
    if weight.abs() <= 1.0e-12 { 0.0 } else { value / weight }
}

/// `BoxFilter` kernel: `sum(src[-r..=r]) / (2r+1)` along one axis.
///
/// The kernel has five typed variants (ushort/ushort4/float/float4/float2)
/// with identical arithmetic. The active `zbc6516e8d5 == 0` path reads the
/// global plane directly and relies on the host to avoid out-of-range reads;
/// this reference clamps to the edge instead.
pub fn box_filter_1d(
    src: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    horizontal: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height];
    let count = (2 * radius + 1) as f32;
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            for d in -(radius as isize)..=radius as isize {
                let (xx, yy) = if horizontal {
                    ((x as isize + d).clamp(0, width as isize - 1) as usize, y)
                } else {
                    (x, (y as isize + d).clamp(0, height as isize - 1) as usize)
                };
                sum += src[yy * width + xx];
            }
            out[y * width + x] = sum / count;
        }
    }
    out
}

/// Four alternating box passes, matching the eight `BoxFilter` dispatches
/// (h, v, h, v, h, v, h, v) in the export queue.
pub fn box_filter_4(src: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut plane = src.to_vec();
    for pass in 0..4 {
        plane = box_filter_1d(&plane, width, height, radius, pass % 2 == 0);
    }
    plane
}

/// Variant 4 (`LumaNoise2018`): ratio transfer against the blurred plane.
///
/// ```text
/// out = src + src * ((max(src,1) / max(blur,1) - 1) * strength)
/// ```
///
/// This is *not* the USM threshold-excess kernel, which has the same variant
/// number in `UnsharpMaskLinear`.
pub fn ratio_apply_v4(src: &[f32], blur: &[f32], strength: f32) -> Vec<f32> {
    src.iter()
        .zip(blur.iter())
        .map(|(&s, &b)| {
            let ratio = s.max(1.0) / b.max(1.0) - 1.0;
            s + s * (ratio * strength)
        })
        .collect()
}

/// Variant 6 (`LumaNoise2018`): blend a plane with luma into the alpha slot.
///
/// ```text
/// luma = dot(src.rgb, weights)
/// out.rgb = src.rgb
/// out.a   = max(0, plane*mix + luma*(1-mix))
/// ```
///
/// `mix` is `args.w`, or the per-pixel mask value when a mask is bound. This
/// is *not* the USM combine kernel.
pub fn blend_to_alpha_v6(
    src: [f32; 4],
    plane: f32,
    weights: [f32; 3],
    mix: f32,
) -> [f32; 4] {
    let luma = src[0] * weights[0] + src[1] * weights[1] + src[2] * weights[2];
    let alpha = (plane * mix + luma * (1.0 - mix)).max(0.0);
    [src[0], src[1], src[2], alpha]
}

/// One `LN_AngleParamBuf` entry. The OpenCL struct is
/// `char offsets[16]; float scale; int mode; uint flag;` (host stride 32).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngleEntry {
    pub offsets: [i8; 16],
    pub scale: f32,
    pub mode: i32,
    pub flag: bool,
}

/// Eight 20-byte direction records appended by `SetParameters` (`0x181775100`).
///
/// Record layout: `{ int x; int y; u8 flag; f32 slope; f32 weight; }`.
/// Only `slope` and `flag` feed the 24-byte offset-entry generator below; the
/// `x/y/weight` fields are kept for documentation.
pub const VENDOR_DIRECTIONS: [(i32, i32, bool, f32, f32); 8] = [
    (-2, -2, false, 1.0, 1.125),
    (-1, -2, false, 0.5, 1.25),
    (0, -2, false, 0.0, 1.0),
    (1, -2, false, -0.5, 1.25),
    (2, -2, false, -1.0, 1.125),
    (2, -1, true, -0.5, 1.25),
    (2, 0, false, 0.0, 1.0),
    (2, 1, true, 0.5, 1.25),
];

/// Generate the eight `LN_AngleParamBuf` entries exactly as the vendor does.
///
/// `block_size` is `this+0xd8` (`16` when the parameter object's `+0xf8` is
/// 1, else `8`). For each direction slope `s`, with `h = block_size / 2`
/// (integer division) and `sign = -1` when `s < 0` else `+1`:
///
/// - `s == 0`:     `offsets[i] = 0`,                        `mode = 2`, `scale = 1`
/// - `|s| == 1`:   `offsets[i] = sign * (i - h)`,           `mode = 2`, `scale = 1`
/// - `|s| == 0.5`: `offsets[i] = h + sign * (i - h)`,       `mode = 1`, `scale = exp2(0.4)`
/// - otherwise:    `offsets[i] = round((i - h) * s)`,       `mode = 2`, `scale = 1`
///
/// Bytes `block_size..16` stay zero (the buffer is zeroed first).
pub fn angle_entries_vendor(block_size: usize) -> [AngleEntry; 8] {
    assert!(
        block_size == 8 || block_size == 16,
        "LumaNoise block size is 8 or 16"
    );
    let n = block_size as i32;
    let h = n / 2;
    let half_scale = 0.4f32.exp2();
    let mut out = [AngleEntry {
        offsets: [0; 16],
        scale: 1.0,
        mode: 2,
        flag: false,
    }; 8];
    for (a, entry) in out.iter_mut().enumerate() {
        let (_, _, flag, slope, _) = VENDOR_DIRECTIONS[a];
        entry.flag = flag;
        let sign = if slope < 0.0 { -1 } else { 1 };
        if slope == 0.0 {
            entry.mode = 2;
            entry.scale = 1.0;
        } else if slope.abs() == 1.0 {
            entry.mode = 2;
            entry.scale = 1.0;
            for i in 0..block_size {
                entry.offsets[i] = (sign * (i as i32 - h)) as i8;
            }
        } else if slope.abs() == 0.5 {
            entry.mode = 1;
            entry.scale = half_scale;
            for i in 0..block_size {
                entry.offsets[i] = (h + sign * (i as i32 - h)) as i8;
            }
        } else {
            entry.mode = 2;
            entry.scale = 1.0;
            for i in 0..block_size {
                entry.offsets[i] = ((i as i32 - h) as f32 * slope).round() as i8;
            }
        }
    }
    out
}

/// Variant 8/9 directional scatter (`za1b3317d00`), exact kernel arithmetic.
///
/// `local` is the filtered 8/16-tap profile, `y` the row inside the block,
/// `offsets` the entry's table, `mode` the entry mode (`1` halves the norm),
/// `origin` the block origin in `acc`, and `acc` the `(value, weight)` plane.
///
/// ```text
/// base = 256 / N
/// norm = base / (base + sum(|local[1..N-1]|))
/// if mode == 1: norm *= 0.5
/// w_y = norm * taper[y]
/// dir1 = flag ? 2 : mode;  dir2 = flag ? mode : 2
/// dir3 = flag ? 0 : mode;  dir4 = flag ? mode : 0
/// sx = (x*dir1 + offsets[y]*dir3 + 256) / 2 - 128
/// sy = (y*dir2 + offsets[x]*dir4 + 256) / 2 - 128
/// acc[origin + (sx,sy)] += (w_y * taper[x] * local[x], w_y * taper[x])
/// ```
///
/// The vendor kernel does not bounds-check `sx/sy` (the host guarantees the
/// block is in range); out-of-range samples are skipped here instead of
/// writing outside `acc`.
#[allow(clippy::too_many_arguments)]
pub fn direction_scatter(
    local: &[f32],
    y: usize,
    offsets: &[i8; 16],
    flag: bool,
    mode: i32,
    width: usize,
    height: usize,
    origin: (usize, usize),
    acc: &mut [[f32; 2]],
) {
    let n = local.len().min(16);
    if n < 3 || y >= n || (n != 8 && n != 16) {
        return;
    }
    let taper8 = [0.0f32, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0];
    let taper16 = [
        0.0f32, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0,
    ];
    let taper = if n == 8 { &taper8[..] } else { &taper16[..] };
    let base = 256.0 / n as f32;
    let local_sum: f32 = local[1..n - 1].iter().map(|v| v.abs()).sum();
    let mut norm = base / (base + local_sum);
    if mode == 1 {
        norm *= 0.5;
    }
    let (dir1, dir2, dir3, dir4) = if flag {
        (2, mode, 0, mode)
    } else {
        (mode, 2, mode, 0)
    };
    let w_y = norm * taper[y];
    if w_y == 0.0 {
        return;
    }
    let (ox, oy) = origin;
    for x in 1..n - 1 {
        let w = w_y * taper[x];
        if w.abs() <= 1.0e-4 {
            continue;
        }
        let sx = (x as i32 * dir1 + i32::from(offsets[y]) * dir3 + 256) / 2 - 128;
        let sy = (y as i32 * dir2 + i32::from(offsets[x]) * dir4 + 256) / 2 - 128;
        let px = ox as i32 + sx;
        let py = oy as i32 + sy;
        if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
            continue;
        }
        let idx = py as usize * width + px as usize;
        if idx < acc.len() {
            acc[idx][0] += w * local[x];
            acc[idx][1] += w;
        }
    }
}

/// ISO -> LumaNoise parameter defaults.
///
/// `floor` and `poisson` are the `camera_params.json` noise entries. Variant 0
/// uses `a` as the gain on luma and `b` as a floor; the variance-stabilising
/// assignment (`a = poisson`, `b = floor`) is natural but not a host-extracted
/// constant. The box radius/threshold/strength are inferred; the host
/// `SetParameters` computes its own values from camera data and has not been
/// mapped to ISO yet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LumaNoiseParams {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub box_radius: usize,
    pub threshold: f32,
    pub strength: f32,
    pub detail: f32,
}

pub fn iso_defaults(floor: f32, poisson: f32, iso: u32) -> LumaNoiseParams {
    let iso = iso.max(100) as f32;
    let radius = ((iso / 1600.0).round() as usize).clamp(1, 8);
    LumaNoiseParams {
        a: poisson.max(1.0e-6),
        b: floor.max(0.0),
        c: 0.0,
        box_radius: radius,
        threshold: (0.02 * (iso / 100.0).sqrt()).max(0.01),
        strength: (0.5 * (iso / 1600.0).min(1.5)).min(0.9),
        detail: 0.5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn transfer_v0_matches_the_kernel_formula() {
        // a=2, b=1, c=0.5, L=0.25:
        // z = 0.5 + 1.5 + 1 - 1 = 2.0; out = sqrt(2)
        let v = luma_transfer_v0(0.25, 2.0, 1.0, 0.5);
        assert!(close(v, 2.0f32.sqrt(), 1e-6), "{v}");
    }

    #[test]
    fn transfer_v0_black_and_guard() {
        let v = luma_transfer_v0(0.0, 2.0, 0.0, 0.0);
        assert!(close(v, (2.0f32 * 0.375 * 2.0).sqrt(), 1e-5));
        // Rust safety guard: the kernel would divide by zero.
        assert_eq!(luma_transfer_v0(0.7, 0.0, 0.0, 0.0), 0.7);
    }

    #[test]
    fn transfer_v5_inverts_v0() {
        let (a, b, c) = (1.5f32, 0.2f32, 0.1f32);
        for &l in &[0.0f32, 0.125, 0.25, 0.5, 0.75, 1.0] {
            let t = luma_transfer_v0(l, a, b, c);
            let back = luma_transfer_v5_inverse(t, a, b, c);
            assert!(close(back, l, 1e-5), "l={l} t={t} back={back}");
        }
    }

    #[test]
    fn luma_transfer_op_preserves_the_target_luma() {
        let w = [0.25f32, 0.5, 0.25];
        let rgb = [0.2f32, 0.4, 0.6];
        let out = luma_transfer_op_default(rgb, 0.4, w);
        let luma = out[0] * w[0] + out[1] * w[1] + out[2] * w[2];
        assert!(close(luma, 0.4, 1e-6), "{out:?} -> {luma}");
    }

    #[test]
    fn luma_transfer_op_clamps_the_chroma_gain() {
        let w = [0.25f32, 0.5, 0.25];
        let rgb = [0.2f32, 0.4, 0.6];
        // max(L,1)=1, max(target,1)=10 => raw gain 10, capped at 1.5.
        let out = luma_transfer_op_default(rgb, 10.0, w);
        let expected = [
            rgb[0] * 1.5 + (10.0 - 0.4 * 1.5),
            rgb[1] * 1.5 + (10.0 - 0.4 * 1.5),
            rgb[2] * 1.5 + (10.0 - 0.4 * 1.5),
        ];
        for i in 0..3 {
            assert!(close(out[i], expected[i], 1e-5), "{out:?}");
        }
    }

    #[test]
    fn luma_transfer_op_uses_the_eps_floor() {
        let w = [0.25f32, 0.5, 0.25];
        let rgb = [0.0001f32, 0.0001, 0.0001];
        // Both luma and target are below 1 => scale = 1, output luma = target.
        let out = luma_transfer_op_default(rgb, 0.5, w);
        let luma = out[0] * w[0] + out[1] * w[1] + out[2] * w[2];
        assert!(close(luma, 0.5, 1e-6), "{out:?} -> {luma}");
    }

    #[test]
    fn value_weight_pair_round_trips() {
        let (v, w) = value_weight_pair(0.7, 3.0);
        assert!(close(normalise_pair(v, w), 0.7, 1e-6));
        assert_eq!(normalise_pair(1.0, 0.0), 0.0);
        assert!(close(normalise_pair(2.0, 4.0), 0.5, 1e-6));
    }

    #[test]
    fn box_filter_preserves_a_flat_field() {
        let src = vec![0.3f32; 25];
        let out = box_filter_4(&src, 5, 5, 1);
        for v in out {
            assert!(close(v, 0.3, 1e-6));
        }
    }

    #[test]
    fn box_filter_1d_matches_the_box_kernel() {
        let src = [0.0f32, 0.0, 1.0, 0.0, 0.0];
        let out = box_filter_1d(&src, 5, 1, 1, true);
        let expected = [0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0];
        for i in 0..5 {
            assert!(close(out[i], expected[i], 1e-6), "{out:?}");
        }
    }

    #[test]
    fn ratio_apply_v4_matches_the_kernel() {
        // s=4, b=2, strength=0.25: ratio=1 => out = 4 + 4*0.25 = 5.
        let out = ratio_apply_v4(&[4.0f32], &[2.0f32], 0.25);
        assert!(close(out[0], 5.0, 1e-6), "{out:?}");
        // Both below 1 => max(.,1) makes the ratio zero.
        let out = ratio_apply_v4(&[0.5f32], &[0.1f32], 1.0);
        assert!(close(out[0], 0.5, 1e-6), "{out:?}");
    }

    #[test]
    fn blend_to_alpha_v6_matches_the_kernel() {
        let src = [0.2f32, 0.4, 0.6, 0.0];
        let w = [0.25f32, 0.5, 0.25];
        let plane = 0.9;
        let out = blend_to_alpha_v6(src, plane, w, 1.0);
        assert!(close(out[3], plane, 1e-6));
        assert_eq!(&out[..3], &src[..3]);
        let out = blend_to_alpha_v6(src, plane, w, 0.0);
        assert!(close(out[3], 0.4, 1e-6), "{out:?}");
        let out = blend_to_alpha_v6(src, -5.0, w, 1.0);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn vendor_angle_entries_16() {
        let e = angle_entries_vendor(16);
        assert_eq!(
            e[0].offsets,
            [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert_eq!(e[0].mode, 2);
        assert!(close(e[0].scale, 1.0, 0.0));
        assert!(!e[0].flag);
        assert_eq!(e[1].offsets, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(e[1].mode, 1);
        assert!(close(e[1].scale, 0.4f32.exp2(), 1e-6));
        assert_eq!(e[2].offsets, [0; 16]);
        assert_eq!(e[2].mode, 2);
        assert_eq!(e[3].offsets, [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(e[3].mode, 1);
        assert_eq!(e[4].offsets, [8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7]);
        assert_eq!(e[4].mode, 2);
        assert_eq!(e[5].offsets, e[3].offsets);
        assert_eq!(e[5].mode, 1);
        assert!(e[5].flag);
        assert_eq!(e[6].offsets, [0; 16]);
        assert!(!e[6].flag);
        assert_eq!(e[7].offsets, e[1].offsets);
        assert_eq!(e[7].mode, 1);
        assert!(e[7].flag);
    }

    #[test]
    fn vendor_angle_entries_8() {
        let e = angle_entries_vendor(8);
        assert_eq!(&e[0].offsets[..8], &[-4i8, -3, -2, -1, 0, 1, 2, 3]);
        assert_eq!(&e[0].offsets[8..], &[0i8; 8]);
        assert_eq!(&e[1].offsets[..8], &[0i8, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(&e[3].offsets[..8], &[8i8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(&e[4].offsets[..8], &[4i8, 3, 2, 1, 0, -1, -2, -3]);
    }

    /// Independent transcription of `za1b3317d00` used to cross-check the
    /// production `direction_scatter` (same loop order, f64 accumulation is
    /// not used; both sides are f32).
    fn scatter_reference(
        local: &[f32],
        y: usize,
        offsets: &[i8; 16],
        flag: bool,
        mode: i32,
        acc: &mut [[f32; 2]],
        width: usize,
        origin: (i32, i32),
    ) {
        let n = local.len();
        let taper8 = [0.0f32, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0];
        let taper16 = [
            0.0f32, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0,
        ];
        let taper: &[f32] = if n == 8 { &taper8 } else { &taper16 };
        let base = 256.0f32 / n as f32;
        let mut sum = 0.0f32;
        for v in &local[1..n - 1] {
            sum += v.abs();
        }
        let mut norm = base / (base + sum);
        if mode == 1 {
            norm *= 0.5;
        }
        let (d1, d2, d3, d4) = if flag {
            (2, mode, 0, mode)
        } else {
            (mode, 2, mode, 0)
        };
        let wy = norm * taper[y];
        for x in 1..n - 1 {
            let w = wy * taper[x];
            if w.abs() <= 1e-4 {
                continue;
            }
            let sx = (x as i32 * d1 + offsets[y] as i32 * d3 + 256) / 2 - 128;
            let sy = (y as i32 * d2 + offsets[x] as i32 * d4 + 256) / 2 - 128;
            let px = origin.0 + sx;
            let py = origin.1 + sy;
            if px < 0 || py < 0 {
                continue;
            }
            let idx = py as usize * width + px as usize;
            if idx < acc.len() {
                acc[idx][0] += w * local[x];
                acc[idx][1] += w;
            }
        }
    }

    #[test]
    fn direction_scatter_matches_the_kernel() {
        let mut local = [0.0f32; 16];
        for (i, v) in local.iter_mut().enumerate() {
            *v = (i as f32 - 7.5) * 0.125;
        }
        let entries = angle_entries_vendor(16);
        let (w, h) = (64usize, 64usize);
        for (idx, y) in [(0usize, 3usize), (1, 7), (5, 8), (7, 4)] {
            let e = entries[idx];
            let mut got = vec![[0.0f32; 2]; w * h];
            let mut want = vec![[0.0f32; 2]; w * h];
            direction_scatter(
                &local, y, &e.offsets, e.flag, e.mode, w, h, (24, 24), &mut got,
            );
            scatter_reference(
                &local, y, &e.offsets, e.flag, e.mode, &mut want, w, (24, 24),
            );
            for i in 0..got.len() {
                assert!(
                    close(got[i][0], want[i][0], 1e-6) && close(got[i][1], want[i][1], 1e-6),
                    "entry {idx} y {y} cell {i}: {:?} vs {:?}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn direction_scatter_mode_one_halves_the_norm() {
        let local = [0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let offsets = [0i8; 16];
        let mut a = vec![[0.0f32; 2]; 32 * 32];
        let mut b = vec![[0.0f32; 2]; 32 * 32];
        direction_scatter(&local, 1, &offsets, false, 2, 32, 32, (8, 8), &mut a);
        direction_scatter(&local, 1, &offsets, false, 1, 32, 32, (8, 8), &mut b);
        let wa: f32 = a.iter().map(|p| p[1]).sum();
        let wb: f32 = b.iter().map(|p| p[1]).sum();
        // Taper at x=2 makes the weight norm*1.0*1.0; mode=1 halves norm.
        assert!(close(wb, wa * 0.5, 1e-6), "{wa} vs {wb}");
    }
}
