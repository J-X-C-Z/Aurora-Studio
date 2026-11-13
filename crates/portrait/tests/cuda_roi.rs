//! Real NVIDIA coverage for the P15 composite kernel.
//!
//! Geometry that used to be untested — 1x1, non-square, non-multiples of the
//! 16x16 block, and rows padded to `stride > width` — is compared float-for-float
//! against `composite_pixel`, then the RGBA8 twin against `apply`.
//!
//! A host without a usable NVIDIA driver prints one `SKIP_NO_NVIDIA_DRIVER` line
//! per test and returns; set `PIXCAKE_CUDA_REQUIRED=1` to turn that into a
//! failure. A driver that is present but broken never skips — it panics.

#![cfg(windows)]

use aurora_pixcake::cuda::{CudaError, Nvidia, Uniforms};
use aurora_pixcake::kernels::facial_smooth as cpu;
use aurora_pixcake::{Image, Refusal};

#[derive(Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
    }

    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }

    fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit()
    }

    fn byte(&mut self) -> u8 {
        (self.next_u32() >> 24) as u8
    }
}

/// `None` only when this host has no driver or no device; that is a recorded
/// skip, never a silent pass.
fn gpu() -> Option<Nvidia> {
    match Nvidia::open() {
        Ok(gpu) => Some(gpu),
        Err(CudaError::Unavailable(reason)) => {
            eprintln!("SKIP_NO_NVIDIA_DRIVER: {reason}");
            assert!(
                std::env::var_os("PIXCAKE_CUDA_REQUIRED").is_none(),
                "CUDA required but unavailable: {reason}"
            );
            None
        }
        Err(broken) => panic!("NVIDIA path present but broken: {broken:?}"),
    }
}

fn ulp(left: f32, right: f32) -> u32 {
    fn key(value: f32) -> i64 {
        let bits = value.to_bits() as i32;
        if bits < 0 {
            (i32::MIN - bits) as i64
        } else {
            bits as i64
        }
    }
    (key(left) - key(right)).unsigned_abs() as u32
}

/// Random job for one geometry: `padding` fills the columns past `width`.
struct Job {
    width: u32,
    height: u32,
    stride: u32,
    source: Vec<f32>,
    base: Vec<f32>,
    residual: Vec<f32>,
    mask: Vec<f32>,
    skin: Vec<f32>,
    uniforms: cpu::Uniforms,
}

impl Job {
    fn new(rng: &mut Rng, width: u32, height: u32, stride: u32, padding: f32) -> Self {
        let len = (stride * height) as usize * 4;
        let mut source = vec![padding; len];
        let mut base = vec![padding; len];
        let mut residual = vec![padding; len];
        let mut mask = vec![padding; len];
        let mut skin = vec![padding; (stride * height) as usize];
        for row in 0..height {
            for column in 0..width {
                let pixel = (row * stride + column) as usize;
                for channel in 0..4 {
                    source[pixel * 4 + channel] = rng.unit();
                    base[pixel * 4 + channel] = rng.unit();
                    residual[pixel * 4 + channel] = rng.unit();
                    mask[pixel * 4 + channel] = rng.unit();
                }
                skin[pixel] = rng.unit();
            }
        }
        let low = rng.range(-1.0, -0.05);
        let high = rng.range(0.05, 1.0);
        Self {
            width,
            height,
            stride,
            source,
            base,
            residual,
            mask,
            skin,
            uniforms: cpu::Uniforms {
                legal_alpha: rng.range(0.2, 1.0),
                dark_circle_alpha: rng.range(0.2, 1.0),
                diff_factor: rng.range(0.0, 1.5),
                diff_factor2: rng.range(0.0, 1.5),
                color_threshold_factor: rng.range(0.0, 1.5),
                mix_alpha: rng.unit(),
                diff_min_threshold: low,
                diff_max_threshold: high,
                high_freq_min_threshold: rng.range(-1.0, -0.2),
                high_freq_max_threshold: rng.range(0.2, 1.0),
            },
        }
    }

    fn block(&self) -> Uniforms {
        Uniforms { stride: self.stride, ..Uniforms::new(self.width, self.height, 90024, 0.5, &self.uniforms) }
    }

    fn at(&self, row: u32, column: u32) -> usize {
        ((row * self.stride + column) as usize) * 4
    }

    /// Bit-exact comparison of every active pixel against `composite_pixel`.
    fn assert_matches_pixel_law(&self, output: &[f32]) -> u32 {
        assert_eq!(output.len(), self.source.len());
        let mut worst = 0u32;
        for row in 0..self.height {
            for column in 0..self.width {
                let at = self.at(row, column);
                let rgba = |values: &[f32]| [values[at], values[at + 1], values[at + 2], values[at + 3]];
                let expected = cpu::composite_pixel(
                    rgba(&self.source),
                    rgba(&self.base),
                    rgba(&self.residual),
                    self.mask[at],
                    self.mask[at + 2],
                    self.skin[at / 4],
                    self.uniforms,
                )
                .unwrap();
                for channel in 0..4 {
                    let delta = ulp(expected[channel], output[at + channel]);
                    assert!(
                        delta == 0,
                        "{}x{} stride {} pixel ({row},{column}) channel {channel}: cpu {} gpu {} ({delta} ulp)",
                        self.width,
                        self.height,
                        self.stride,
                        expected[channel],
                        output[at + channel],
                    );
                    worst = worst.max(delta);
                }
                assert_eq!(output[at + 3], self.source[at + 3], "alpha is copied, never computed");
            }
        }
        worst
    }

    /// Byte-exact comparison of the RGBA8 twin against `apply`.
    fn assert_matches_rgba8_twin(&self, gpu: &Nvidia) -> u8 {
        let pixels = (self.width * self.height) as usize;
        let active = |values: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(pixels * 4);
            for row in 0..self.height {
                for column in 0..self.width {
                    let at = self.at(row, column);
                    for channel in 0..4 {
                        out.push((values[at + channel].clamp(0.0, 1.0) * 255.0).round() as u8);
                    }
                }
            }
            out
        };
        let mask_r: Vec<u8> = (0..pixels).map(|i| (self.mask[i * 4].clamp(0.0, 1.0) * 255.0).round() as u8).collect();
        let mask_b: Vec<u8> = (0..pixels).map(|i| (self.mask[i * 4 + 2].clamp(0.0, 1.0) * 255.0).round() as u8).collect();
        let skin: Vec<u8> = (0..pixels).map(|i| (self.skin[i].clamp(0.0, 1.0) * 255.0).round() as u8).collect();
        let source = Image::new(self.width, self.height, active(&self.source)).unwrap();
        let base = Image::new(self.width, self.height, active(&self.base)).unwrap();
        let residual = Image::new(self.width, self.height, active(&self.residual)).unwrap();
        let expected = cpu::apply(&source, &base, &residual, &mask_r, &mask_b, &skin, self.uniforms).unwrap();
        let (actual, _) = gpu
            .facial_smooth(&source, &base, &residual, &mask_r, &mask_b, &skin, 90024, 0.5, &self.uniforms)
            .expect("real GPU dispatch");
        let mut worst = 0u8;
        for (index, (left, right)) in expected.pixels.iter().zip(&actual.pixels).enumerate() {
            let delta = left.abs_diff(*right);
            assert_eq!(
                delta, 0,
                "{}x{} byte {index} (pixel {}, channel {}): cpu {left} gpu {right}",
                self.width,
                self.height,
                index / 4,
                index % 4
            );
            worst = worst.max(delta);
        }
        worst
    }
}

#[test]
fn composite_matches_the_pixel_law_across_roi_geometry() {
    let Some(gpu) = gpu() else { return };
    // 1x1, single row/column, non-square, and both sides off the 16x16 block.
    for (width, height) in [(1, 1), (1, 7), (7, 1), (17, 13), (37, 19), (16, 16), (31, 32), (64, 64)] {
        let mut rng = Rng::new((width * 1000 + height) as u64);
        let job = Job::new(&mut rng, width, height, width, 0.0);
        let (output, dispatch) = gpu
            .composite_f32(&job.source, &job.mask, &job.skin, &job.base, &job.residual, &job.block())
            .expect("real GPU dispatch");
        job.assert_matches_pixel_law(&output);
        assert_eq!(dispatch.grid, (width.div_ceil(16).max(1), height.div_ceil(16).max(1)));
        eprintln!("{width}x{height} grid {:?} ok", dispatch.grid);
    }
}

#[test]
fn padded_rows_do_not_bleed_into_active_pixels() {
    let Some(gpu) = gpu() else { return };
    // NaN padding: any accidental read would poison the output instead of
    // hiding behind a plausible number.
    for (width, height, stride) in [(17, 13, 24), (37, 19, 48), (1, 1, 8), (16, 16, 17), (64, 64, 64)] {
        let mut rng = Rng::new((width * 7919 + stride) as u64);
        let job = Job::new(&mut rng, width, height, stride, f32::NAN);
        let (output, _) = gpu
            .composite_f32(&job.source, &job.mask, &job.skin, &job.base, &job.residual, &job.block())
            .expect("padding columns are never read");
        let worst = job.assert_matches_pixel_law(&output);
        assert_eq!(worst, 0);
        eprintln!("{width}x{height} stride {stride} ok");
    }
}

#[test]
fn legal_threshold_boundaries_match_the_pixel_law() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng::new(97);
    let base = Job::new(&mut rng, 21, 11, 21, 0.0);
    let mut cases = Vec::new();
    let mut degenerate = base.uniforms;
    degenerate.diff_min_threshold = 0.25;
    degenerate.diff_max_threshold = 0.25;
    cases.push(("degenerate diff clamp", degenerate));
    let mut frozen = base.uniforms;
    frozen.high_freq_min_threshold = 0.0;
    frozen.high_freq_max_threshold = 0.0;
    cases.push(("zero delta band", frozen));
    let mut wide = base.uniforms;
    wide.diff_min_threshold = -1.0;
    wide.diff_max_threshold = 1.0;
    wide.high_freq_min_threshold = -1.0;
    wide.high_freq_max_threshold = 1.0;
    cases.push(("full +-1 band", wide));
    let mut flat = base.uniforms;
    flat.mix_alpha = 0.0;
    flat.color_threshold_factor = 0.0;
    cases.push(("mix at 0", flat));
    let mut solid = base.uniforms;
    solid.mix_alpha = 1.0;
    solid.color_threshold_factor = 1.0;
    cases.push(("mix at 1", solid));

    for (label, uniforms) in cases {
        let job = Job { uniforms, ..Job::new(&mut rng.clone(), 21, 11, 21, 0.0) };
        let (output, _) = gpu
            .composite_f32(&job.source, &job.mask, &job.skin, &job.base, &job.residual, &job.block())
            .expect("legal boundaries must dispatch");
        assert_eq!(job.assert_matches_pixel_law(&output), 0, "{label}");
    }
}

#[test]
fn saturation_masks_are_honoured() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng::new(11);
    let job = Job::new(&mut rng, 19, 17, 19, 0.0);
    let mut zero = job.mask.clone();
    zero.iter_mut().for_each(|value| *value = 0.0);
    let mut skin_zero = job.skin.clone();
    skin_zero.iter_mut().for_each(|value| *value = 0.0);
    let (output, _) = gpu
        .composite_f32(&job.source, &zero, &skin_zero, &job.base, &job.residual, &job.block())
        .expect("zero masks dispatch");
    assert_eq!(output, job.source, "zero alpha is the identity");

    let mut one = job.mask.clone();
    one.iter_mut().for_each(|value| *value = 1.0);
    let mut skin_one = job.skin.clone();
    skin_one.iter_mut().for_each(|value| *value = 1.0);
    let full = Job { mask: one, skin: skin_one, ..Job::new(&mut Rng::new(11), 19, 17, 19, 0.0) };
    let (output, _) = gpu
        .composite_f32(&full.source, &full.mask, &full.skin, &full.base, &full.residual, &full.block())
        .expect("saturated masks dispatch");
    assert_eq!(full.assert_matches_pixel_law(&output), 0);
}

#[test]
fn rgba8_twin_is_byte_identical_on_odd_geometry() {
    let Some(gpu) = gpu() else { return };
    for (width, height) in [(1, 1), (17, 13), (37, 19), (63, 64)] {
        let mut rng = Rng::new((width * 131 + height) as u64);
        let job = Job::new(&mut rng, width, height, width, 0.0);
        assert_eq!(job.assert_matches_rgba8_twin(&gpu), 0, "{width}x{height}");
        eprintln!("rgba8 {width}x{height} byte-identical");
    }
}

#[test]
fn boundary_buffers_are_refused_like_the_cpu_lane() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng::new(5);
    let job = Job::new(&mut rng, 12, 9, 12, 0.0);
    let block = job.block();
    let run = |source: &[f32], mask: &[f32], skin: &[f32]| {
        gpu.composite_f32(source, mask, skin, &job.base, &job.residual, &block).err()
    };
    assert_eq!(run(&job.source, &job.mask, &job.skin), None);
    let mut nan = job.source.clone();
    nan[6] = f32::NAN;
    assert_eq!(run(&nan, &job.mask, &job.skin), Some(CudaError::Refused(Refusal::ValueOutOfRange("pixel"))));
    let mut hot = job.mask.clone();
    hot[2] = 1.5;
    assert_eq!(run(&job.source, &hot, &job.skin), Some(CudaError::Refused(Refusal::ValueOutOfRange("mask"))));
    let mut cold = job.skin.clone();
    cold[4] = -0.25;
    assert_eq!(run(&job.source, &job.mask, &cold), Some(CudaError::Refused(Refusal::ValueOutOfRange("mask"))));
}

#[test]
fn roi_size_gate_still_rejects_a_mismatched_side() {
    let uniforms = Uniforms::new(17, 13, 90024, 0.5, &cpu::Uniforms {
        legal_alpha: 1.0,
        dark_circle_alpha: 1.0,
        diff_factor: 1.0,
        diff_factor2: 0.0,
        color_threshold_factor: 0.0,
        mix_alpha: 1.0,
        diff_min_threshold: -1.0,
        diff_max_threshold: 1.0,
        high_freq_min_threshold: -1.0,
        high_freq_max_threshold: 1.0,
    });
    assert!(uniforms.validate().is_ok(), "non-square ROI is a legal launch");
    let moved = Uniforms { roi_size: 16, ..uniforms };
    assert_eq!(moved.validate(), Err(Refusal::ShapeMismatch("roi_size")));
}
