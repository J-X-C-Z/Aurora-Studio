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


// ===========================================================================
// HOST MAPPING CLOSED IN c1_luma_noise_host_20261125
// ===========================================================================
//
// Evidence (current DLL ca02b6ac..., ImageBase 0x180000000):
//   SetParameters          VA 0x181775100
//   ProcessBlock           VA 0x18177bc50
//   0x310 buffer init      VA 0x18177d720
//   op Run (kernel args)   VA 0x18177b330
//   enable predicate       VA 0x181776210
//   noise-model producer   VA 0x18154e3b0  (SettingsToSettingsExCameraConfig)
//   per-ISO table          coreConfig.cdx (XML XOR 0x81), see camera-iso-sample.tsv
//
// The 0x310-byte constant buffer is the OpenCL struct transcribed from
// p_041588192.cl (struct z5dd37f8de8): 8 x 32-byte entries, 64 int2 taps,
// then count/bbox/min and a -1 tail.  The three 512-byte 32x16 tap masks are
// static .rdata tables selected by (flag_a, size16) in SetParameters.

/// f32 at VA 0x182a3df2c.
pub const LN_REF_LUMA: f32 = 11796.30078125;
/// f32 at VA 0x182a3dfec (f32(LN_REF_LUMA * LN_REF_LUMA)).
pub const LN_REF_LUMA_SQ: f32 = 139_152_720.0;
/// f32 at VA 0x182a35508.
pub const LN_AMOUNT_SCALE: f32 = 0.019_999_999_552_965_164;
/// f32 at VA 0x182a31fc4.
pub const LN_STRAIGHT_SCALE: f32 = 4.999_999_873_689_376e-5;
/// Tap list capacity (CL `int2 taps[64]`).
pub const LN_TAP_CAPACITY: usize = 64;
/// The shipped kernel unrolls `if (count > 49)` only, so taps 50..63 are unused.
pub const LN_KERNEL_TAP_UNROLL: usize = 50;
/// op+0xf8, set at 0x181775f6a.
pub const LN_BLOCK_W: i32 = 32;
/// op+0xfc, set at 0x181775f74.
pub const LN_BLOCK_H: i32 = 16;
/// `clCreateBuffer` size at 0x18177c5de.
pub const LN_ANGLE_PARAM_BUF_SIZE: usize = 0x310;

/// VA 0x1827bf890, SHA-256 645fcf9312fa72d6409d8f001560338a089efcf5b02c04e0e43e84890c03aa35, 32 non-zero bytes.
pub static LN_TAP_MASK_16_ND: [u8; 512] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
];
/// VA 0x1827bfa90, SHA-256 4591e0636d1926e78d88073c89c533f28ff0360bce1f151afa269fe064a20e8b, 64 non-zero bytes.
pub static LN_TAP_MASK_8_ND: [u8; 512] = [
    0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
];
/// VA 0x1827bfc90, SHA-256 b2112555aec7706440f3cae32151b5aad0d89fe3e41b5ea3e58778d87155e539, 18 non-zero bytes.
pub static LN_TAP_MASK_16_DIR: [u8; 512] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Mask table used by `SetParameters` (0x181775f19..0x181775f5c).
pub fn ln_tap_mask(flag_a: bool, size16: bool) -> (&'static [u8; 512], u64) {
    match (flag_a, size16) {
        (false, false) => (&LN_TAP_MASK_8_ND, 0x1827BFA90),
        (false, true) => (&LN_TAP_MASK_16_ND, 0x1827BF890),
        (true, false) => (&LN_TAP_MASK_16_ND, 0x1827BF890),
        (true, true) => (&LN_TAP_MASK_16_DIR, 0x1827BFC90),
    }
}

/// One 32-byte OpenCL angle entry (`struct z60c1969d74`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct AngleEntryGuest {
    pub offsets: [i8; 16],
    pub scale: f32,
    pub mode: i32,
    pub flag: i32,
    /// Initialised to -1 by 0x18177d740, never overwritten.
    pub pad: i32,
}

impl AngleEntryGuest {
    /// Defaults written by 0x18177d720: scale 0, mode 2, flag 0, pad -1.
    pub const DEFAULT: AngleEntryGuest = AngleEntryGuest {
        offsets: [0; 16],
        scale: 0.0,
        mode: 2,
        flag: 0,
        pad: -1,
    };

    pub fn from_angle_entry(e: &AngleEntry) -> Self {
        AngleEntryGuest {
            offsets: e.offsets,
            scale: e.scale,
            mode: e.mode,
            flag: if e.flag { 1 } else { 0 },
            pad: -1,
        }
    }
}

/// Host-side 0x310-byte `LN_AngleParamBuf` (also the CL struct z5dd37f8de8).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct AngleParamBuf {
    pub entries: [AngleEntryGuest; 8],
    pub taps: [[i32; 2]; LN_TAP_CAPACITY],
    pub tap_count: i32,
    pub bbox_w: i16,
    pub bbox_h: i16,
    pub min_x: i16,
    pub min_y: i16,
    pub tail: i32,
}

const _: () = assert!(core::mem::size_of::<AngleEntryGuest>() == 32);
const _: () = assert!(core::mem::size_of::<AngleParamBuf>() == LN_ANGLE_PARAM_BUF_SIZE);
const _: () = assert!(core::mem::offset_of!(AngleParamBuf, taps) == 0x100);
const _: () = assert!(core::mem::offset_of!(AngleParamBuf, tap_count) == 0x300);
const _: () = assert!(core::mem::offset_of!(AngleParamBuf, bbox_w) == 0x304);
const _: () = assert!(core::mem::offset_of!(AngleParamBuf, min_x) == 0x308);
const _: () = assert!(core::mem::offset_of!(AngleParamBuf, tail) == 0x30c);

impl Default for AngleParamBuf {
    fn default() -> Self {
        AngleParamBuf {
            entries: [AngleEntryGuest::DEFAULT; 8],
            taps: [[0; 2]; LN_TAP_CAPACITY],
            tap_count: 0,
            bbox_w: 0,
            bbox_h: 0,
            min_x: 100,
            min_y: 100,
            tail: -1,
        }
    }
}

impl AngleParamBuf {
    /// Pack the entries and tap list exactly as ProcessBlock 0x18177c11a.. does.
    pub fn pack(entries: &[AngleEntry], taps: &[(i32, i32)]) -> Self {
        assert!(entries.len() <= 8);
        assert!(taps.len() <= LN_TAP_CAPACITY);
        let mut buf = AngleParamBuf::default();
        for (i, e) in entries.iter().enumerate() {
            buf.entries[i] = AngleEntryGuest::from_angle_entry(e);
        }
        for (i, &(x, y)) in taps.iter().enumerate() {
            buf.taps[i] = [x, y];
        }
        buf.tap_count = taps.len() as i32;
        if !taps.is_empty() {
            let (mut min_x, mut max_x) = (i32::MAX, i32::MIN);
            let (mut min_y, mut max_y) = (i32::MAX, i32::MIN);
            for &(x, y) in taps {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
            buf.min_x = min_x as i16;
            buf.min_y = min_y as i16;
            buf.bbox_w = (max_x - min_x + 1) as i16;
            buf.bbox_h = (max_y - min_y + 1) as i16;
        }
        buf
    }

    /// Canonical little-endian bytes (no unsafe aliasing).
    pub fn to_bytes(&self) -> [u8; LN_ANGLE_PARAM_BUF_SIZE] {
        let mut out = [0u8; LN_ANGLE_PARAM_BUF_SIZE];
        let mut o = 0usize;
        for e in &self.entries {
            for &b in &e.offsets {
                out[o] = b as u8;
                o += 1;
            }
            out[o..o + 4].copy_from_slice(&e.scale.to_le_bytes());
            o += 4;
            out[o..o + 4].copy_from_slice(&e.mode.to_le_bytes());
            o += 4;
            out[o..o + 4].copy_from_slice(&e.flag.to_le_bytes());
            o += 4;
            out[o..o + 4].copy_from_slice(&e.pad.to_le_bytes());
            o += 4;
        }
        debug_assert_eq!(o, 0x100);
        for t in &self.taps {
            out[o..o + 4].copy_from_slice(&t[0].to_le_bytes());
            out[o + 4..o + 8].copy_from_slice(&t[1].to_le_bytes());
            o += 8;
        }
        debug_assert_eq!(o, 0x300);
        out[o..o + 4].copy_from_slice(&self.tap_count.to_le_bytes());
        out[o + 4..o + 6].copy_from_slice(&self.bbox_w.to_le_bytes());
        out[o + 6..o + 8].copy_from_slice(&self.bbox_h.to_le_bytes());
        out[o + 8..o + 10].copy_from_slice(&self.min_x.to_le_bytes());
        out[o + 10..o + 12].copy_from_slice(&self.min_y.to_le_bytes());
        out[o + 12..o + 16].copy_from_slice(&self.tail.to_le_bytes());
        out
    }
}

/// Tap collection, ProcessBlock 0x18177c278..0x18177c46e.
///
/// The outer loop is `y in 1-tile ..= 16-tile`, the inner `x in y ..= 32-tile`;
/// the mask is indexed `(x+base_x) mod 32 + ((y+base_y) mod 16)*32`.
pub fn ln_collect_taps(mask: &[u8; 512], tile: usize, base_x: i32, base_y: i32) -> Vec<(i32, i32)> {
    let mut taps = Vec::new();
    let mut y = 1 - tile as i32;
    while y < LN_BLOCK_H - tile as i32 + 1 {
        let mut x = y;
        while x < LN_BLOCK_W - tile as i32 + 1 {
            let idx = ((x + base_x).rem_euclid(LN_BLOCK_W)
                + (y + base_y).rem_euclid(LN_BLOCK_H) * LN_BLOCK_W) as usize;
            if mask[idx] != 0 {
                taps.push((x, y));
            }
            x += 1;
        }
        y += 1;
    }
    taps
}

/// Noise-model SNR in dB as computed by SetParameters 0x1817751e6..0x181775240.
pub fn ln_model_db_settings(noise_floor: f32, noise_poisson: f32) -> f32 {
    let denom = noise_floor * noise_floor + noise_poisson * LN_REF_LUMA;
    10.0 * (LN_REF_LUMA_SQ / denom).log10()
}

/// Same dB test as the enable predicate 0x181776232..0x181776276 (gate 25 dB).
/// NOTE: the field roles in that predicate are NOT proven to be floor/poisson.
pub fn ln_model_db_predicate(x_11038: f32, x_1103c: f32) -> f32 {
    let denom = x_11038 * LN_REF_LUMA + x_1103c * x_1103c;
    10.0 * (LN_REF_LUMA_SQ / denom).log10()
}

/// Inputs read by `CImgOpLumaNoise2018::SetParameters` (0x181775100).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LumaNoiseHostInputs {
    /// settings+0x11038, coreConfig `noiseFloorCoeff_*`.
    pub noise_floor: f32,
    /// settings+0x1103c, coreConfig `noisePoissonCoeff_*`.
    pub noise_poisson: f32,
    /// settings+0x11040 != 0.
    pub model_present: bool,
    /// settings+0x484 (integer slider; defaults 40/50).
    pub amount: i32,
    /// settings+0x89c/+0x8a0/+0x8a4.
    pub weights: [f32; 3],
    /// settings+0x8a8/+0x8ac/+0x8b0 (defaults 2.0/1.0/1.5).
    pub mults: [f32; 3],
    /// settings+0x130 (f64).
    pub favour_straight: f64,
    /// settings+0xf8 == 1 selects block size 16.
    pub size16: bool,
    /// First predicate, 0x1817751c1..0x1817751e6 (settings+0x480 < 51 and a
    /// parent-virtual test).  Must be supplied by the caller.
    pub flag_a: bool,
}

/// Outputs written to op+0xa8/+0xac/+0xb0/+0xcc/+0xd0/+0xd8/+0xdc.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LumaNoiseHostParams {
    pub valid_model: bool,
    pub a8: f32,
    pub ac: f32,
    pub m_favour_straight: f32,
    pub b0: f32,
    pub d0: f32,
    pub block_size: i32,
    pub directional: bool,
    pub mask_va: u64,
    pub db: f32,
}

/// Default (no debug globals) transcription of SetParameters 0x18177531c..0x181776195.
pub fn ln_host_params(i: &LumaNoiseHostInputs) -> LumaNoiseHostParams {
    let valid = i.model_present && i.noise_floor > 0.0 && i.noise_poisson > 0.0;
    let (a8_0, ac_0) = if valid {
        (i.noise_poisson, i.noise_floor)
    } else {
        // 0x1817753b1: virtual +0x668 default 100 -> 100/100; ac = res*5.
        (1.0f32, 5.0f32)
    };
    let mut sumsq = 0.0f32;
    for k in 0..3 {
        let t = i.weights[k] * i.mults[k];
        sumsq += t * t;
    }
    let a8 = a8_0 * sumsq;
    let ac = ac_0 * sumsq.sqrt();
    let c6 = i.favour_straight as f32;
    // 0x181775507..0x18177556d (debug branches 0x...47d..0x...51c excluded).
    let a8 = c6 * c6 * a8;
    let ac = c6 * ac;
    let cc = ((LN_REF_LUMA_SQ / (ac * ac + LN_REF_LUMA * a8)) * LN_STRAIGHT_SCALE)
        .clamp(0.05, 1.0);
    let x = i.amount as f32 * LN_AMOUNT_SCALE;
    let base = 1.0 - x;
    let b0 = (0.5 * base).exp2() + 2.0 + base.max(0.0) * base.max(0.0);
    let db = ln_model_db_settings(i.noise_floor, i.noise_poisson);
    let flag_b = i.flag_a && db >= 30.0;
    let (_, mask_va) = ln_tap_mask(i.flag_a, i.size16);
    LumaNoiseHostParams {
        valid_model: valid,
        a8,
        ac,
        m_favour_straight: cc,
        b0,
        d0: 1024.0,
        block_size: if i.size16 { 16 } else { 8 },
        directional: !flag_b,
        mask_va,
        db,
    }
}

/// The only static ISO ramp found in the DLL: BandingSuppression SetParameters
/// 0x18167fbd0 (`cmp iso, 200` at 0x18167fcae, `cmp iso, 1600` at
/// 0x18167fcb7).  Included for comparison; it does NOT belong to LumaNoise2018.
pub fn iso_strength_ramp_200_1600(iso: i32, base: f32) -> f32 {
    if iso <= 200 {
        base
    } else if iso > 1600 {
        base * 2.0
    } else {
        base * (1.0 + (iso - 200) as f32 / 1400.0)
    }
}

#[cfg(test)]
mod host_tests {
    use super::*;
    #[allow(unused_imports)]
    use std::convert::TryInto;

    #[test]
    fn angle_param_buf_layout_is_0x310() {
        assert_eq!(core::mem::size_of::<AngleEntryGuest>(), 32);
        assert_eq!(core::mem::size_of::<AngleParamBuf>(), 0x310);
        assert_eq!(core::mem::offset_of!(AngleParamBuf, taps), 0x100);
        assert_eq!(core::mem::offset_of!(AngleParamBuf, tap_count), 0x300);
        assert_eq!(core::mem::offset_of!(AngleParamBuf, bbox_w), 0x304);
        assert_eq!(core::mem::offset_of!(AngleParamBuf, bbox_h), 0x306);
        assert_eq!(core::mem::offset_of!(AngleParamBuf, min_x), 0x308);
        assert_eq!(core::mem::offset_of!(AngleParamBuf, min_y), 0x30a);
        assert_eq!(core::mem::offset_of!(AngleParamBuf, tail), 0x30c);
    }

    #[test]
    fn angle_param_buf_pack_matches_process_block() {
        let e = angle_entries_vendor(16);
        let taps = [(-3i32, -7i32), (1, -7), (9, 0)];
        // One real entry: slots 1..7 must keep the 0x18177d720 defaults.
        let buf = AngleParamBuf::pack(&e[..1], &taps);
        assert_eq!(buf.tap_count, 3);
        assert_eq!((buf.bbox_w, buf.bbox_h), (13, 8));
        assert_eq!((buf.min_x, buf.min_y), (-3, -7));
        assert_eq!(buf.tail, -1);
        let bytes = buf.to_bytes();
        assert_eq!(bytes.len(), 0x310);
        // entry 0 (slope 1.0, 16): offsets -8..7, then scale 1, mode 2, flag 0, -1
        assert_eq!(&bytes[..8], &[248, 249, 250, 251, 252, 253, 254, 255]);
        assert_eq!(f32::from_le_bytes(bytes[0x10..0x14].try_into().unwrap()), 1.0);
        assert_eq!(i32::from_le_bytes(bytes[0x14..0x18].try_into().unwrap()), 2);
        assert_eq!(i32::from_le_bytes(bytes[0x18..0x1c].try_into().unwrap()), 0);
        assert_eq!(i32::from_le_bytes(bytes[0x1c..0x20].try_into().unwrap()), -1);
        // unused entries keep the 0x18177d720 defaults
        assert_eq!(f32::from_le_bytes(bytes[0x30..0x34].try_into().unwrap()), 0.0);
        assert_eq!(i32::from_le_bytes(bytes[0x34..0x38].try_into().unwrap()), 2);
        // header
        assert_eq!(i32::from_le_bytes(bytes[0x300..0x304].try_into().unwrap()), 3);
        assert_eq!(i16::from_le_bytes(bytes[0x304..0x306].try_into().unwrap()), 13);
        assert_eq!(i16::from_le_bytes(bytes[0x306..0x308].try_into().unwrap()), 8);
        assert_eq!(i16::from_le_bytes(bytes[0x308..0x30a].try_into().unwrap()), -3);
        assert_eq!(i16::from_le_bytes(bytes[0x30a..0x30c].try_into().unwrap()), -7);
        assert_eq!(i32::from_le_bytes(bytes[0x30c..0x310].try_into().unwrap()), -1);
    }

    #[test]
    fn tap_mask_counts_match_the_static_tables() {
        // Counts at phase (0,0); triangular scan 0x18177c2c0..0x18177c444.
        let t16 = ln_collect_taps(&LN_TAP_MASK_16_ND, 8, 0, 0);
        assert!((23..=26).contains(&t16.len()), "{}", t16.len());
        let t8 = ln_collect_taps(&LN_TAP_MASK_8_ND, 8, 0, 0);
        assert!((46..=51).contains(&t8.len()), "{}", t8.len());
        let td = ln_collect_taps(&LN_TAP_MASK_16_DIR, 8, 0, 0);
        assert!((12..=15).contains(&td.len()), "{}", td.len());
        // Over all 512 phases the 8-bit non-directional mask can exceed the
        // 50-tap kernel unroll (24 of 512 phases yield 51 taps).
        // 144/512 phases yield >=50 taps; 24/512 yield 51 taps, so in 24
        // phases the host emits one tap the shipped kernel unroll never reads.
        let mut ge50 = 0;
        let mut eq51 = 0;
        for bx in 0..32 {
            for by in 0..16 {
                let n = ln_collect_taps(&LN_TAP_MASK_8_ND, 8, bx, by).len();
                if n >= LN_KERNEL_TAP_UNROLL {
                    ge50 += 1;
                }
                if n > LN_KERNEL_TAP_UNROLL {
                    eq51 += 1;
                }
            }
        }
        assert_eq!(ge50, 144);
        assert_eq!(eq51, 24);
    }

    #[test]
    fn mask_selection_matches_set_parameters() {
        assert_eq!(ln_tap_mask(false, false).1, 0x1827BFA90);
        assert_eq!(ln_tap_mask(false, true).1, 0x1827BF890);
        assert_eq!(ln_tap_mask(true, false).1, 0x1827BF890);
        assert_eq!(ln_tap_mask(true, true).1, 0x1827BFC90);
    }

    #[test]
    fn host_params_synthetic() {
        let i = LumaNoiseHostInputs {
            noise_floor: 0.5,
            noise_poisson: 0.03125,
            model_present: true,
            amount: 100,
            weights: [1.0, 1.0, 1.0],
            mults: [2.0, 1.0, 1.5],
            favour_straight: 1.0,
            size16: true,
            flag_a: true,
        };
        let p = ln_host_params(&i);
        // sumsq = 4 + 1 + 2.25 = 7.25
        assert!((p.a8 - 0.03125 * 7.25).abs() < 1e-6, "{p:?}");
        assert!((p.ac - 0.5 * 7.25f32.sqrt()).abs() < 1e-6, "{p:?}");
        assert!((p.b0 - ((-0.5f32).exp2() + 2.0)).abs() < 1e-6, "{p:?}");
        assert_eq!(p.block_size, 16);
        assert!(!p.directional);
        assert_eq!(p.mask_va, 0x1827BFC90);
        // invalid model -> fallback res = 1, a8 = sumsq, ac = 5*sqrt(sumsq)
        let j = LumaNoiseHostInputs {
            model_present: false,
            noise_floor: -1.0,
            noise_poisson: 0.0,
            size16: false,
            flag_a: false,
            ..i
        };
        let q = ln_host_params(&j);
        assert!(!q.valid_model && q.directional);
        assert!((q.a8 - 7.25).abs() < 1e-6, "{q:?}");
        assert!((q.ac - 5.0 * 7.25f32.sqrt()).abs() < 1e-6, "{q:?}");
        assert_eq!(q.block_size, 8);
        assert_eq!(q.mask_va, 0x1827BFA90);
    }

    #[test]
    fn model_db_uses_reference_luma() {
        // real config sample: CanonEOS100D ISO100
        let db = ln_model_db_settings(0.385510, 0.032761);
        assert!(db > 30.0 && db < 70.0, "{db}");
        // fallback model 10/1/0 is invalid (type 0) but the dB value is finite
        assert!(ln_model_db_settings(10.0, 1.0).is_finite());
    }

    #[test]
    fn banding_iso_ramp_is_not_luma_noise() {
        assert_eq!(iso_strength_ramp_200_1600(100, 3.0), 3.0);
        assert_eq!(iso_strength_ramp_200_1600(1600, 3.0), 6.0);
        assert_eq!(iso_strength_ramp_200_1600(1601, 3.0), 6.0);
    }
}
