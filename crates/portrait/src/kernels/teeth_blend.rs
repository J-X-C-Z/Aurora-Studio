//! PF30026 final mask blend (pix-worker.exe `509D2293...6B84`).
//!
//! Recovered at `0x230EC90` / helper `0x230DF40`. Roles at `0x5615C74`:
//! output and original alias `ctx+0x90`, effect is `record+0x20`, mask is
//! `record+0xA8`, direction 0.
//!
//! Strict native order:
//!   m = mask_u8 * 0x3B807BBB
//!   high = m * effect_f
//!   low  = original_f * (1 - m)
//!   out  = cvtt(high + low)          // truncate toward zero
//!   mask == 255 copies effect RGB, alpha is always the original.
//!
//! The LUT operands are u8 and `m` is in `[0,1)`, so `high + low` stays in
//! `[0,256)`; `cvtt` cannot be negative or overflow. A violated domain is a
//! typed refusal, not a silent clamp.

use crate::{Image, Refusal};

/// Exact vendor constant: `0x8CEEF00` f32 bits `0x3B807BBB`.
pub const VENDOR_MASK_SCALE: f32 = f32::from_bits(0x3B80_7BBB);

fn blend_channel(m8: u8, original: u8, effect: u8) -> Result<u8, Refusal> {
    if m8 == 255 {
        return Ok(effect);
    }
    let m = m8 as f32 * VENDOR_MASK_SCALE;
    let high = m * effect as f32;
    let low = original as f32 * (1.0 - m);
    let sum = high + low;
    if !sum.is_finite() {
        return Err(Refusal::ValueOutOfRange("teeth blend sum"));
    }
    let truncated = sum.trunc();
    if !(0.0..=255.0).contains(&truncated) {
        return Err(Refusal::ValueOutOfRange("teeth blend sum"));
    }
    Ok(truncated as u8)
}

/// Explicit-input PF30026 final blend. This is the blend stage only; the
/// opaque model output and dense warp remain caller inputs.
pub fn apply(original: &Image, effect: &Image, mask: &[u8]) -> Result<Image, Refusal> {
    original
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("teeth blend original"))?;
    effect
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("teeth blend effect"))?;
    if original.width != effect.width || original.height != effect.height {
        return Err(Refusal::ShapeMismatch("teeth blend effect"));
    }
    if mask.len() != original.pixel_count() {
        return Err(Refusal::ShapeMismatch("teeth blend mask"));
    }

    let mut pixels = original.pixels.clone();
    for index in 0..mask.len() {
        let o = index * 4;
        for c in 0..3 {
            pixels[o + c] = blend_channel(mask[index], original.pixels[o + c], effect.pixels[o + c])?;
        }
        pixels[o + 3] = original.pixels[o + 3];
    }
    Image::new(original.width, original.height, pixels).ok_or(Refusal::ShapeMismatch("teeth blend output"))
}

/// Inputs produced by one PF30026 worker job.
///
/// This is the final mask blend only. The opaque model output and dense warp
/// are already inside `effect_rgb`.
#[derive(Clone, Copy, Debug, Default)]
pub struct AlignmentFinalBlendJob<'a> {
    pub original: Option<&'a Image>,
    /// RGBA8 container carrying the effect RGB; its alpha is ignored.
    pub effect_rgb: Option<&'a Image>,
    pub teeth_mask: Option<&'a [u8]>,
    /// First worker `CV_32FC1` edge plane.
    pub edge_plane_a: Option<&'a [f32]>,
    /// Second worker `CV_32FC1` edge plane.
    pub edge_plane_b: Option<&'a [f32]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AlignmentFinalBlendOutcome {
    Gated,
    Applied(Image),
}

pub(crate) struct ValidatedAlignmentFinalBlend<'a> {
    pub original: &'a Image,
    pub effect_rgb: &'a Image,
    pub teeth_mask: &'a [u8],
    pub gate_open: bool,
}

pub(crate) fn validate_alignment_final_blend(
    job: AlignmentFinalBlendJob<'_>,
) -> Result<ValidatedAlignmentFinalBlend<'_>, Refusal> {
    let original = job
        .original
        .ok_or(Refusal::MissingInput("PF30026 original"))?;
    let effect_rgb = job
        .effect_rgb
        .ok_or(Refusal::MissingInput("PF30026 effect RGB"))?;
    let teeth_mask = job
        .teeth_mask
        .ok_or(Refusal::MissingInput("PF30026 teeth mask"))?;
    let edge_plane_a = job
        .edge_plane_a
        .ok_or(Refusal::MissingInput("PF30026 edge plane A"))?;
    let edge_plane_b = job
        .edge_plane_b
        .ok_or(Refusal::MissingInput("PF30026 edge plane B"))?;

    original
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("PF30026 original"))?;
    effect_rgb
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("PF30026 effect RGB"))?;
    if (original.width, original.height) != (effect_rgb.width, effect_rgb.height) {
        return Err(Refusal::ShapeMismatch("PF30026 effect RGB"));
    }
    if teeth_mask.len() != original.pixel_count() {
        return Err(Refusal::ShapeMismatch("PF30026 teeth mask"));
    }
    let gate_open = crate::kernels::teeth_alignment::edge_gate(
        edge_plane_a,
        edge_plane_b,
        original.width,
        original.height,
    )?;
    Ok(ValidatedAlignmentFinalBlend {
        original,
        effect_rgb,
        teeth_mask,
        gate_open,
    })
}

/// Validate the complete same-job bundle and return the A/B edge gate without
/// blending pixels. Used by CUDA-only callers that must not report a CPU run.
pub fn alignment_final_blend_gate(job: AlignmentFinalBlendJob<'_>) -> Result<bool, Refusal> {
    validate_alignment_final_blend(job).map(|job| job.gate_open)
}

/// CPU oracle for the PF30026 alignment final blend stage.
pub fn apply_alignment_final_blend(
    job: AlignmentFinalBlendJob<'_>,
) -> Result<AlignmentFinalBlendOutcome, Refusal> {
    let job = validate_alignment_final_blend(job)?;
    if !job.gate_open {
        return Ok(AlignmentFinalBlendOutcome::Gated);
    }
    apply(job.original, job.effect_rgb, job.teeth_mask).map(AlignmentFinalBlendOutcome::Applied)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &[u8] = include_bytes!("../../../../work/backend-r27/r8/blend_fixture.bin");

    fn parts() -> (Image, Image, Vec<u8>) {
        let original = Image::new(2, 2, FIXTURE[..16].to_vec()).unwrap();
        let effect = Image::new(2, 2, FIXTURE[16..32].to_vec()).unwrap();
        (original, effect, FIXTURE[32..].to_vec())
    }

    #[test]
    fn vendor_scale_bits() {
        assert_eq!(VENDOR_MASK_SCALE.to_bits(), 0x3b807bbb);
    }

    #[test]
    fn fixture_matches_native_lut_math() {
        let (original, effect, mask) = parts();
        let out = apply(&original, &effect, &mask).unwrap();
        assert_eq!(out.pixel(0, 0), [105, 115, 125, 40]);
        assert_eq!(out.pixel(1, 0), [50, 60, 70, 80]);
        assert_eq!(out.pixel(0, 1), [255, 255, 255, 120]);
        assert_eq!(out.pixel(1, 1), [129, 120, 120, 160]);
    }

    #[test]
    fn core_contract_explicit_input() {
        let (original, effect, mask) = parts();
        let out = apply(&original, &effect, &mask).unwrap();
        for (index, pixel) in out.pixels.chunks_exact(4).enumerate() {
            assert_eq!(pixel[3], original.pixels[index * 4 + 3]);
        }
        assert_eq!(apply(&original, &effect, &[0; 4]).unwrap(), original);
        let full = apply(&original, &effect, &[255; 4]).unwrap();
        for (index, pixel) in full.pixels.chunks_exact(4).enumerate() {
            assert_eq!(pixel[..3], effect.pixels[index * 4..index * 4 + 3]);
            assert_eq!(pixel[3], original.pixels[index * 4 + 3]);
        }
    }

    #[test]
    fn mask_sweep_matches_f32_and_f64_oracles() {
        let pairs = [
            (0u8, 0u8),
            (0, 255),
            (255, 0),
            (255, 255),
            (1, 254),
            (254, 1),
            (10, 200),
            (200, 10),
            (128, 127),
            (127, 128),
            (37, 211),
            (211, 37),
            (64, 192),
            (192, 64),
            (5, 250),
            (250, 5),
        ];
        let mut f64_boundary_diffs = 0usize;
        for m8 in 0..=255u8 {
            for &(original, effect) in &pairs {
                let native = blend_channel(m8, original, effect).unwrap();
                let m32 = m8 as f32 * VENDOR_MASK_SCALE;
                let sum32 = m32 * effect as f32 + original as f32 * (1.0 - m32);
                assert!(sum32.is_finite());
                assert!(sum32 >= 0.0 && sum32 < 256.0, "domain {}", sum32);
                let expected32 = if m8 == 255 {
                    effect as i32
                } else {
                    sum32.trunc() as i32
                };
                assert_eq!(native as i32, expected32, "f32 m8={m8} {original}/{effect}");

                let m64 = m8 as f64 * VENDOR_MASK_SCALE as f64;
                let sum64 = m64 * effect as f64 + original as f64 * (1.0 - m64);
                let f64_i = if m8 == 255 {
                    effect as i32
                } else {
                    sum64.trunc() as i32
                };
                let native_i = native as i32;
                if f64_i != native_i {
                    // C++ uses float LUTs; f64 is a cross-check only. The
                    // only allowed difference is the f32 value sitting one
                    // ulp below/above the integer boundary.
                    f64_boundary_diffs += 1;
                    assert!((f64_i - native_i).abs() == 1, "f64 m8={m8} {original}/{effect}");
                    let boundary = f64_i.max(native_i) as f32;
                    assert!(
                        (sum32 - boundary).abs() < 1e-3,
                        "non-boundary f64 diff m8={m8} {original}/{effect} sum32={sum32}"
                    );
                }
            }
        }
        // Bounded: only f32-float rounding boundaries, not a semantic drift.
        assert!(f64_boundary_diffs <= 256 * pairs.len());
    }

    #[test]
    fn shape_and_length_are_refused() {
        let (original, effect, _) = parts();
        assert_eq!(
            apply(&original, &effect, &[0; 3]),
            Err(Refusal::ShapeMismatch("teeth blend mask"))
        );
        assert_eq!(
            apply(&original, &Image::solid(1, 1, [0; 4]), &[0; 4]),
            Err(Refusal::ShapeMismatch("teeth blend effect"))
        );
        let broken = Image {
            width: 2,
            height: 2,
            pixels: vec![0; 15],
        };
        assert_eq!(
            apply(&broken, &effect, &[0; 4]),
            Err(Refusal::ShapeMismatch("teeth blend original"))
        );
    }

    #[test]
    fn alignment_stage_requires_every_same_job_input() {
        let original = Image::solid(2, 2, [10, 20, 30, 40]);
        let effect = Image::solid(2, 2, [200, 210, 220, 0]);
        let mask = [128u8; 4];
        let edge = [0.5f32; 4];
        let full = AlignmentFinalBlendJob {
            original: Some(&original),
            effect_rgb: Some(&effect),
            teeth_mask: Some(&mask),
            edge_plane_a: Some(&edge),
            edge_plane_b: Some(&edge),
        };
        for (job, missing) in [
            (
                AlignmentFinalBlendJob {
                    original: None,
                    ..full
                },
                "PF30026 original",
            ),
            (
                AlignmentFinalBlendJob {
                    effect_rgb: None,
                    ..full
                },
                "PF30026 effect RGB",
            ),
            (
                AlignmentFinalBlendJob {
                    teeth_mask: None,
                    ..full
                },
                "PF30026 teeth mask",
            ),
            (
                AlignmentFinalBlendJob {
                    edge_plane_a: None,
                    ..full
                },
                "PF30026 edge plane A",
            ),
            (
                AlignmentFinalBlendJob {
                    edge_plane_b: None,
                    ..full
                },
                "PF30026 edge plane B",
            ),
        ] {
            assert_eq!(
                validate_alignment_final_blend(job).err(),
                Some(Refusal::MissingInput(missing))
            );
        }
    }

    #[test]
    fn alignment_stage_obeys_the_same_job_edge_gate() {
        let original = Image::solid(2, 2, [10, 20, 30, 40]);
        let effect = Image::solid(2, 2, [200, 210, 220, 0]);
        let mask = [128u8; 4];
        let open = [0.5f32; 4];
        let closed = [0.1f32; 4];
        let applied = apply_alignment_final_blend(AlignmentFinalBlendJob {
            original: Some(&original),
            effect_rgb: Some(&effect),
            teeth_mask: Some(&mask),
            edge_plane_a: Some(&open),
            edge_plane_b: Some(&open),
        })
        .unwrap();
        assert_eq!(
            applied,
            AlignmentFinalBlendOutcome::Applied(apply(&original, &effect, &mask).unwrap())
        );
        assert_eq!(
            apply_alignment_final_blend(AlignmentFinalBlendJob {
                original: Some(&original),
                effect_rgb: Some(&effect),
                teeth_mask: Some(&mask),
                edge_plane_a: Some(&closed),
                edge_plane_b: Some(&closed),
            }),
            Ok(AlignmentFinalBlendOutcome::Gated)
        );
    }
}
