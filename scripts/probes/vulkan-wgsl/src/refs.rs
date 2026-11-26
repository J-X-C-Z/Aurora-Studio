//! CPU reference for the module C probe.
//!
//! Everything here is integer (Q8 fixed point) or plain f32, mirroring the WGSL
//! kernels statement for statement. The GPU result must equal this byte for byte
//! on every channel, including alpha.
//!
//! The case matrix and the references live here (not in the binary) so the unit
//! tests below exercise exactly what a hardware run exercises, on CPU only.

/// Odd sizes first: a 1x1, a 3x5, two odd-by-odd, one odd-by-even, one even control.
pub const SIZES: &[(u32, u32)] = &[(37, 23), (1, 1), (3, 5), (63, 65), (127, 31), (64, 64)];
/// Strength levels: 0 % (strict identity), 50 %, 100 %.
pub const LEVELS: &[u32] = &[0, 50, 100];
/// FP32 matmul shapes as (m, k, n).
pub const MATMUL_SHAPES: &[(usize, usize, usize)] = &[(16, 32, 8), (64, 128, 32)];

/// Fixed-point parameters of the two-stage RGBA8 fusion.
#[derive(Clone, Copy, Debug)]
pub struct FusionParams {
    pub width: u32,
    pub height: u32,
    /// Q8 per-channel gain, 256 == 1.0
    pub gain: [u32; 3],
    /// signed per-channel offset in 8-bit units
    pub off: [i32; 3],
    /// Q8 contrast about mid-grey, 256 == 1.0
    pub contrast: u32,
    /// 0 / 50 / 100 — informational, folded into gain/off/contrast by the host
    pub strength_percent: u32,
    /// 1 = leave fully transparent pixels untouched
    pub alpha_policy: u32,
}

impl FusionParams {
    /// Host-side strength mapping: 0 % is exact identity, 100 % is the full look.
    pub fn for_level(width: u32, height: u32, level: u32) -> Self {
        const ID_GAIN: [i32; 3] = [256, 256, 256];
        const FULL_GAIN: [i32; 3] = [281, 256, 230]; // ~+9.8 % R, +0 % G, -10.2 % B
        const ID_OFF: [i32; 3] = [0, 0, 0];
        const FULL_OFF: [i32; 3] = [4, 0, -3];
        const ID_CONTRAST: i32 = 256;
        const FULL_CONTRAST: i32 = 320; // 1.25x

        let l = level.min(100) as i32;
        let lerp = |a: i32, b: i32| -> i32 { a + (b - a) * l / 100 };
        Self {
            width,
            height,
            gain: [
                lerp(ID_GAIN[0], FULL_GAIN[0]) as u32,
                lerp(ID_GAIN[1], FULL_GAIN[1]) as u32,
                lerp(ID_GAIN[2], FULL_GAIN[2]) as u32,
            ],
            off: [
                lerp(ID_OFF[0], FULL_OFF[0]),
                lerp(ID_OFF[1], FULL_OFF[1]),
                lerp(ID_OFF[2], FULL_OFF[2]),
            ],
            contrast: lerp(ID_CONTRAST, FULL_CONTRAST) as u32,
            strength_percent: level,
            alpha_policy: 1,
        }
    }

    /// 11 words, layout documented in `shaders/fusion.wgsl`.
    pub fn to_words(&self) -> Vec<u32> {
        vec![
            self.width,
            self.height,
            self.gain[0],
            self.gain[1],
            self.gain[2],
            self.off[0] as u32,
            self.off[1] as u32,
            self.off[2] as u32,
            self.contrast,
            self.strength_percent,
            self.alpha_policy,
        ]
    }
}

pub const PARAM_WORDS: usize = 11;
/// Poison planted in the intermediate image so a fused run can prove it never
/// touched it.
pub const MID_POISON: u32 = 0xDEAD_BEEF;

fn unpack_r(v: u32) -> u32 {
    v & 0xff
}
fn unpack_g(v: u32) -> u32 {
    (v >> 8) & 0xff
}
fn unpack_b(v: u32) -> u32 {
    (v >> 16) & 0xff
}
fn unpack_a(v: u32) -> u32 {
    (v >> 24) & 0xff
}
fn pack_rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
    (r & 0xff) | ((g & 0xff) << 8) | ((b & 0xff) << 16) | ((a & 0xff) << 24)
}

/// Stage A: `value * gain / 256 + off`, round-half-up, clamped to 8 bit.
fn stage_a(c: u32, gain: u32, off: i32) -> u32 {
    let t = (c as i32) * (gain as i32) + 128 + off * 256;
    ((t.max(0) as u32) >> 8).min(255)
}

/// Stage B: `(value - 128) * contrast / 256 + 128`, round-half-up, clamped to 8 bit.
fn stage_b(c: u32, contrast: u32) -> u32 {
    let t = (c as i32 - 128) * (contrast as i32) + (128 * 256) + 128;
    ((t.max(0) as u32) >> 8).min(255)
}

fn keep_alpha(p: u32, alpha_policy: u32) -> bool {
    !(alpha_policy == 1 && unpack_a(p) == 0)
}

/// Deterministic synthetic RGBA8 fixture: odd sizes, alpha variety, clipped edges.
pub fn make_fixture(width: u32, height: u32) -> Vec<u32> {
    let mut px = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = (x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13)).wrapping_add(11)) & 0xff;
            let g = ((x.wrapping_mul(31)) ^ (y.wrapping_mul(17)) ^ 0x5a) & 0xff;
            let b = ((x | y).wrapping_mul(5).wrapping_add(3)) & 0xff;
            let a = match (x + y) % 5 {
                0 => 0u32,     // fully transparent: must survive untouched
                1 => 128u32,   // half transparent
                2 => 1u32,     // near-transparent but not 0: must still be processed
                _ => 255u32,
            };
            px.push(pack_rgba(r, g, b, a));
        }
    }
    // Hard edges: clipping extremes on all four corners of an odd-sized buffer.
    if width >= 1 && height >= 1 {
        px[0] = pack_rgba(0, 0, 0, 0);
        px[(width - 1) as usize] = pack_rgba(255, 255, 255, 255);
        px[((height - 1) * width) as usize] = pack_rgba(255, 255, 255, 0);
        px[((height - 1) * width + width - 1) as usize] = pack_rgba(0, 0, 0, 255);
    }
    if width > 2 && height > 2 {
        let i = (height / 2 * width + width / 2) as usize;
        px[i] = pack_rgba(127, 128, 129, 254);
    }
    px
}

/// Serial A -> B with 8-bit quantization between the stages (the "unfused" truth).
pub fn reference_serial(src: &[u32], p: &FusionParams) -> (Vec<u32>, Vec<u32>) {
    let mut mid = Vec::with_capacity(src.len());
    let mut dst = Vec::with_capacity(src.len());
    for &s in src {
        let a = unpack_a(s);
        let m = if keep_alpha(s, p.alpha_policy) {
            pack_rgba(
                stage_a(unpack_r(s), p.gain[0], p.off[0]),
                stage_a(unpack_g(s), p.gain[1], p.off[1]),
                stage_a(unpack_b(s), p.gain[2], p.off[2]),
                a,
            )
        } else {
            s
        };
        mid.push(m);
        let d = if keep_alpha(m, p.alpha_policy) {
            pack_rgba(
                stage_b(unpack_r(m), p.contrast),
                stage_b(unpack_g(m), p.contrast),
                stage_b(unpack_b(m), p.contrast),
                unpack_a(m),
            )
        } else {
            m
        };
        dst.push(d);
    }
    (mid, dst)
}

/// Fused A then B in one pass, re-quantizing between the stages.
pub fn reference_fused(src: &[u32], p: &FusionParams) -> Vec<u32> {
    let mut dst = Vec::with_capacity(src.len());
    for &s in src {
        let a = unpack_a(s);
        let q = if keep_alpha(s, p.alpha_policy) {
            let ar = stage_a(unpack_r(s), p.gain[0], p.off[0]);
            let ag = stage_a(unpack_g(s), p.gain[1], p.off[1]);
            let ab = stage_a(unpack_b(s), p.gain[2], p.off[2]);
            pack_rgba(
                stage_b(ar, p.contrast),
                stage_b(ag, p.contrast),
                stage_b(ab, p.contrast),
                a,
            )
        } else {
            s
        };
        dst.push(q);
    }
    dst
}

/// Per-channel comparison of packed RGBA8 buffers.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rgba8Diff {
    pub mismatch_pixels: usize,
    pub first_mismatch: Option<(usize, u32, u32)>,
    pub max_abs: [u32; 4],
}

impl Rgba8Diff {
    pub fn exact(&self) -> bool {
        self.mismatch_pixels == 0
    }
}

pub fn diff_rgba8(got: &[u32], want: &[u32]) -> Rgba8Diff {
    let mut d = Rgba8Diff::default();
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        if g == w {
            continue;
        }
        d.mismatch_pixels += 1;
        if d.first_mismatch.is_none() {
            d.first_mismatch = Some((i, g, w));
        }
        let pairs = [
            (unpack_r(g), unpack_r(w)),
            (unpack_g(g), unpack_g(w)),
            (unpack_b(g), unpack_b(w)),
            (unpack_a(g), unpack_a(w)),
        ];
        for (ch, (a, b)) in pairs.iter().enumerate() {
            let diff = if a > b { a - b } else { b - a };
            if diff > d.max_abs[ch] {
                d.max_abs[ch] = diff;
            }
        }
    }
    d
}

// ---------------------------------------------------------------------------
// FP32 weighted matmul
// ---------------------------------------------------------------------------

pub struct MatMulCase {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub bias: Vec<f32>,
}

impl MatMulCase {
    /// Deterministic pseudo-random weights in [-1, 1) with no external RNG crate.
    pub fn new(m: usize, k: usize, n: usize) -> Self {
        let mut s = 0x9E37_79B9u32 ^ ((m as u32) << 16) ^ ((k as u32) << 8) ^ (n as u32);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s >> 8) as f32 / 8388608.0) - 1.0
        };
        let a = (0..m * k).map(|_| next()).collect();
        let b = (0..k * n).map(|_| next()).collect();
        let bias = (0..n).map(|_| next() * 0.25).collect();
        Self { m, k, n, a, b, bias }
    }

    /// FP32 reference, accumulation order identical to the kernel.
    pub fn reference_f32(&self) -> Vec<f32> {
        let mut c = vec![0f32; self.m * self.n];
        for row in 0..self.m {
            for col in 0..self.n {
                let mut acc = self.bias[col];
                for i in 0..self.k {
                    acc += self.a[row * self.k + i] * self.b[i * self.n + col];
                }
                c[row * self.n + col] = acc;
            }
        }
        c
    }

    /// FP32 reference with fused multiply-add, i.e. what a driver does when it
    /// contracts `acc + a*b`. If the GPU matches THIS one to 0, the difference
    /// against `reference_f32` was FMA contraction, not a wrong kernel.
    pub fn reference_fma_f32(&self) -> Vec<f32> {
        let mut c = vec![0f32; self.m * self.n];
        for row in 0..self.m {
            for col in 0..self.n {
                let mut acc = self.bias[col];
                for i in 0..self.k {
                    acc = self.a[row * self.k + i].mul_add(self.b[i * self.n + col], acc);
                }
                c[row * self.n + col] = acc;
            }
        }
        c
    }

    /// FP64 reference, to separate "kernel is wrong" from "f32 is f32".
    pub fn reference_f64(&self) -> Vec<f64> {
        let mut c = vec![0f64; self.m * self.n];
        for row in 0..self.m {
            for col in 0..self.n {
                let mut acc = self.bias[col] as f64;
                for i in 0..self.k {
                    acc += self.a[row * self.k + i] as f64 * self.b[i * self.n + col] as f64;
                }
                c[row * self.n + col] = acc;
            }
        }
        c
    }
}

pub fn max_abs_f32(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

pub fn max_abs_f32_f64(got: &[f32], want: &[f64]) -> f64 {
    got.iter()
        .zip(want.iter())
        .map(|(a, b)| (*a as f64 - b).abs())
        .fold(0f64, f64::max)
}

// ---------------------------------------------------------------------------
// CPU-only contract tests
//
// These need no GPU and no fixture files, so CI can run them anywhere. They
// cover the half of the claim that does not depend on hardware: the reference
// pair must agree, the strength levels must hit their documented endpoints, and
// the comparison helper must actually catch a one-LSB difference. The hardware
// half (GPU output == these references) can only be checked by running the
// binary on a Vulkan device.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn all_cases() -> impl Iterator<Item = (u32, u32, u32)> {
        SIZES
            .iter()
            .flat_map(|&(w, h)| LEVELS.iter().map(move |&level| (w, h, level)))
    }

    #[test]
    fn fused_reference_equals_serial_reference_on_every_case() {
        // This is the `cpu_fused_equals_cpu_serial` flag the binary records for
        // each case; here it is an assertion over the whole matrix.
        for (w, h, level) in all_cases() {
            let src = make_fixture(w, h);
            let prm = FusionParams::for_level(w, h, level);
            let (_, serial) = reference_serial(&src, &prm);
            let fused = reference_fused(&src, &prm);
            assert_eq!(serial, fused, "{w}x{h} at {level} %: fusing changed the result");
        }
    }

    #[test]
    fn level_zero_is_strict_identity() {
        for (w, h, level) in all_cases().filter(|case| case.2 == 0) {
            let src = make_fixture(w, h);
            let prm = FusionParams::for_level(w, h, level);
            let (mid, dst) = reference_serial(&src, &prm);
            assert_eq!(mid, src, "{w}x{h}: stage A is not identity at 0 %");
            assert_eq!(dst, src, "{w}x{h}: stage B is not identity at 0 %");
        }
    }

    #[test]
    fn fully_transparent_pixels_are_never_touched() {
        for (w, h, level) in all_cases() {
            let src = make_fixture(w, h);
            let prm = FusionParams::for_level(w, h, level);
            let (_, dst) = reference_serial(&src, &prm);
            for (i, (&s, &d)) in src.iter().zip(dst.iter()).enumerate() {
                if unpack_a(s) == 0 {
                    assert_eq!(s, d, "{w}x{h} at {level} %: alpha==0 pixel {i} changed");
                }
            }
        }
    }

    #[test]
    fn stage_quantization_rounds_half_up_and_clamps() {
        // Round-half-up: 1 * 128/256 = 0.5 rounds to 1, truncation would give 0.
        assert_eq!(stage_a(1, 128, 0), 1);
        assert_eq!(stage_a(0, 256, -1), 0, "negative intermediate must clamp to 0");
        assert_eq!(stage_a(255, 512, 0), 255, "510 must clamp to 255");
        assert_eq!(stage_a(255, 256, 0), 255);
        assert_eq!(stage_b(128, 512), 128, "mid-grey is the contrast pivot");
        assert_eq!(stage_b(0, 32), 112);
        assert_eq!(stage_b(255, 512), 255, "382 must clamp to 255");
        assert_eq!(stage_b(0, 4096), 0, "negative intermediate must clamp to 0");
    }

    #[test]
    fn diff_rgba8_catches_one_lsb_on_one_channel() {
        let want = [pack_rgba(10, 20, 30, 40), pack_rgba(1, 2, 3, 4)];
        let mut got = want;
        got[1] = pack_rgba(1, 2, 4, 3);

        let d = diff_rgba8(&got, &want);
        assert_eq!(d.mismatch_pixels, 1);
        assert_eq!(d.max_abs, [0, 0, 1, 1]);
        assert_eq!(d.first_mismatch, Some((1, got[1], want[1])));
        assert!(!d.exact());
        assert!(diff_rgba8(&want, &want).exact(), "identical buffers must compare exact");
    }

    #[test]
    fn matmul_references_agree_inside_the_contract() {
        for &(m, k, n) in MATMUL_SHAPES.iter().chain([(1, 1, 1)].iter()) {
            let case = MatMulCase::new(m, k, n);
            let f32_ref = case.reference_f32();
            let fma_ref = case.reference_fma_f32();
            let f64_ref = case.reference_f64();
            assert_eq!(f32_ref.len(), m * n);
            assert!(
                max_abs_f32(&fma_ref, &f32_ref) < 1e-4,
                "{m}x{k}x{n}: FMA and mul+add references diverge by more than the contract"
            );
            assert!(
                max_abs_f32_f64(&f32_ref, &f64_ref) < 1e-4,
                "{m}x{k}x{n}: f32 reference is outside 1e-4 of f64"
            );
        }
    }

    #[test]
    fn matmul_degenerate_shape_is_bias_plus_product() {
        let case = MatMulCase::new(1, 1, 1);
        assert_eq!(case.reference_f32()[0], case.a[0] * case.b[0] + case.bias[0]);
    }

    #[test]
    fn fixture_is_deterministic_and_hits_its_documented_corners() {
        let px = make_fixture(5, 4);
        assert_eq!(px, make_fixture(5, 4), "fixture must be reproducible");
        assert_eq!(px.len(), 20);
        assert_eq!(px[0], pack_rgba(0, 0, 0, 0));
        assert_eq!(px[4], pack_rgba(255, 255, 255, 255));
        assert_eq!(px[3 * 5], pack_rgba(255, 255, 255, 0));
        assert_eq!(px[3 * 5 + 4], pack_rgba(0, 0, 0, 255));
        assert_eq!(px[2 * 5 + 2], pack_rgba(127, 128, 129, 254));
    }
}
