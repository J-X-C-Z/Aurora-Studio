//! `FacialSmoothProcessV2` local pixel law (rows 36-38).
//!
//! `mask_r`, `mask_b` and `skin_mask` are one byte per pixel. Source, base and
//! residual share dimensions. Alpha is copied from the source, never filtered.

use crate::{Access, EffectMeta, Image, Refusal};

/// Local composites require caller-provided model outputs and uniforms.
pub const META: &[EffectMeta] = &[
    EffectMeta {
        pf: 90024,
        key: "removeForeheadLines",
        label: "Forehead wrinkles",
        access: Access::Cpu,
        inputs: &[
            "source",
            "base",
            "residual",
            "mask_r",
            "mask_b",
            "skin_mask",
            "uniforms",
        ],
    },
    EffectMeta {
        pf: 21024,
        key: "",
        label: "Glabella wrinkles",
        access: Access::Cpu,
        inputs: &[
            "source",
            "base",
            "residual",
            "mask_r",
            "mask_b",
            "skin_mask",
            "uniforms",
        ],
    },
    EffectMeta {
        pf: 90023,
        key: "removeEyeLines",
        label: "Eye wrinkles",
        access: Access::Cpu,
        inputs: &[
            "source",
            "base",
            "residual",
            "mask_r",
            "mask_b",
            "skin_mask",
            "uniforms",
        ],
    },
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub legal_alpha: f32,
    pub dark_circle_alpha: f32,
    pub diff_factor: f32,
    pub diff_factor2: f32,
    pub color_threshold_factor: f32,
    pub mix_alpha: f32,
    pub diff_min_threshold: f32,
    pub diff_max_threshold: f32,
    pub high_freq_min_threshold: f32,
    pub high_freq_max_threshold: f32,
}

impl Uniforms {
    fn all_finite(&self) -> bool {
        [
            self.legal_alpha,
            self.dark_circle_alpha,
            self.diff_factor,
            self.diff_factor2,
            self.color_threshold_factor,
            self.mix_alpha,
            self.diff_min_threshold,
            self.diff_max_threshold,
            self.high_freq_min_threshold,
            self.high_freq_max_threshold,
        ]
        .iter()
        .all(|value| value.is_finite())
    }

    fn validate(&self) -> Result<(), Refusal> {
        if !self.all_finite()
            || self.diff_min_threshold > self.diff_max_threshold
            || self.high_freq_min_threshold > self.high_freq_max_threshold
        {
            return Err(Refusal::ValueOutOfRange("uniforms"));
        }
        Ok(())
    }
}

fn mix(a: f32, b: f32, amount: f32) -> f32 {
    a * (1.0 - amount) + b * amount
}

/// Unquantized local shader output, for float textures and GPU comparison.
/// Uniforms and model outputs remain explicit caller inputs.
pub fn composite_pixel(
    source: [f32; 4],
    base: [f32; 4],
    residual: [f32; 4],
    mask_r: f32,
    mask_b: f32,
    skin_mask: f32,
    uniforms: Uniforms,
) -> Result<[f32; 4], Refusal> {
    uniforms.validate()?;
    if !source
        .iter()
        .chain(&base)
        .chain(&residual)
        .all(|v| v.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("pixel"));
    }
    if ![mask_r, mask_b, skin_mask]
        .iter()
        .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
    {
        return Err(Refusal::ValueOutOfRange("mask"));
    }
    let output = composite_unchecked(source, base, residual, mask_r, mask_b, skin_mask, uniforms);
    if !output.iter().all(|v| v.is_finite()) {
        return Err(Refusal::ValueOutOfRange("composite"));
    }
    Ok(output)
}

fn composite_unchecked(
    source: [f32; 4],
    base: [f32; 4],
    residual: [f32; 4],
    mask_r: f32,
    mask_b: f32,
    skin_mask: f32,
    uniforms: Uniforms,
) -> [f32; 4] {
    let alpha = (uniforms.dark_circle_alpha * mask_r)
        .max(uniforms.legal_alpha * mask_b)
        .max(0.0)
        * skin_mask;
    let mut output = source;
    if alpha <= 0.0 {
        return output;
    }
    for channel in 0..3 {
        let src = source[channel];
        let base_value = base[channel];
        let residual_value = residual[channel];
        let high = src - base_value;
        let diff = ((residual_value - base_value) * uniforms.diff_factor
            + residual_value * uniforms.diff_factor2)
            .clamp(uniforms.diff_min_threshold, uniforms.diff_max_threshold);
        let upper = mix(
            residual_value * uniforms.color_threshold_factor,
            residual_value,
            uniforms.mix_alpha,
        )
        .max(src - high);
        // GLSL clamp is min(max(x, lower), upper), including negative HDR bounds.
        let reference = (src + diff - high).max(0.0).min(upper) + high;
        let delta = (mix(src, reference, alpha) - src).clamp(
            uniforms.high_freq_min_threshold,
            uniforms.high_freq_max_threshold,
        );
        output[channel] = src + delta;
    }
    output
}

fn validate_inputs(
    source: &Image,
    base: &Image,
    residual: &Image,
    mask_r: &[u8],
    mask_b: &[u8],
    skin_mask: &[u8],
    uniforms: Uniforms,
) -> Result<(), Refusal> {
    source.validate()?;
    base.validate()?;
    residual.validate()?;
    if source.width != base.width
        || source.height != base.height
        || source.width != residual.width
        || source.height != residual.height
    {
        return Err(Refusal::ShapeMismatch("base/residual"));
    }
    let pixels = source.pixel_count();
    if mask_r.len() != pixels {
        return Err(Refusal::ShapeMismatch("mask_r"));
    }
    if mask_b.len() != pixels {
        return Err(Refusal::ShapeMismatch("mask_b"));
    }
    if skin_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("skin_mask"));
    }
    uniforms.validate()?;
    Ok(())
}

fn process_pixels(
    source: &[u8],
    base: &[u8],
    residual: &[u8],
    mask_r: &[u8],
    mask_b: &[u8],
    skin_mask: &[u8],
    uniforms: Uniforms,
    out: &mut [u8],
) -> Result<(), Refusal> {
    for (i, pixel) in out.chunks_exact_mut(4).enumerate() {
        let offset = i * 4;
        let src = std::array::from_fn(|c| source[offset + c] as f32 / 255.0);
        let base_value = std::array::from_fn(|c| base[offset + c] as f32 / 255.0);
        let residual_value = std::array::from_fn(|c| residual[offset + c] as f32 / 255.0);
        let output = composite_unchecked(
            src,
            base_value,
            residual_value,
            mask_r[i] as f32 / 255.0,
            mask_b[i] as f32 / 255.0,
            skin_mask[i] as f32 / 255.0,
            uniforms,
        );
        if !output.iter().all(|v| v.is_finite()) {
            return Err(Refusal::ValueOutOfRange("composite"));
        }
        for channel in 0..3 {
            pixel[channel] = (output[channel].clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    Ok(())
}

pub fn apply(
    source: &Image,
    base: &Image,
    residual: &Image,
    mask_r: &[u8],
    mask_b: &[u8],
    skin_mask: &[u8],
    uniforms: Uniforms,
) -> Result<Image, Refusal> {
    validate_inputs(source, base, residual, mask_r, mask_b, skin_mask, uniforms)?;
    let mut out = source.clone();
    process_pixels(&source.pixels, &base.pixels, &residual.pixels,
        mask_r, mask_b, skin_mask, uniforms, &mut out.pixels)?;
    Ok(out)
}

/// Opt-in CPU fallback using at most four row bands. Inputs are borrowed and
/// workers share the same pixel law; no image or mask bands are copied.
/// Scheduling remains caller-owned because small images may be slower.
pub fn apply_parallel(
    source: &Image,
    base: &Image,
    residual: &Image,
    mask_r: &[u8],
    mask_b: &[u8],
    skin_mask: &[u8],
    uniforms: Uniforms,
) -> Result<Image, Refusal> {
    validate_inputs(source, base, residual, mask_r, mask_b, skin_mask, uniforms)?;
    let rows_per_band = (source.height as usize).div_ceil(4);
    let pixels_per_band = rows_per_band * source.width as usize;
    let bytes_per_band = pixels_per_band * 4;
    let mut out = source.clone();
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(4);
        for (band, pixels) in out.pixels.chunks_mut(bytes_per_band).enumerate() {
            let first = band * pixels_per_band;
            let last = first + pixels.len() / 4;
            let source = &source.pixels[first * 4..last * 4];
            let base = &base.pixels[first * 4..last * 4];
            let residual = &residual.pixels[first * 4..last * 4];
            let mask_r = &mask_r[first..last];
            let mask_b = &mask_b[first..last];
            let skin_mask = &skin_mask[first..last];
            handles.push(scope.spawn(move || {
                process_pixels(source, base, residual, mask_r, mask_b, skin_mask, uniforms, pixels)
            }));
        }
        for handle in handles {
            handle.join().expect("facial smooth worker panicked")?;
        }
        Ok::<(), Refusal>(())
    })?;
    Ok(out)
}

#[cfg(test)]
fn test_uniforms() -> Uniforms {
    Uniforms {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel() -> Image {
        Image::solid(1, 1, [100, 150, 200, 77])
    }

    #[test]
    fn zero_alpha_is_identity() {
        let src = pixel();
        let out = apply(&src, &src, &src, &[0], &[0], &[255], test_uniforms()).unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn alpha_follows_source() {
        let src = pixel();
        let base = Image::solid(1, 1, [10, 20, 30, 0]);
        let out = apply(&src, &base, &src, &[255], &[255], &[255], test_uniforms()).unwrap();
        assert_eq!(out.pixel(0, 0)[3], 77);
    }

    #[test]
    fn mismatched_masks_are_refused() {
        let src = pixel();
        assert_eq!(
            apply(&src, &src, &src, &[], &[0], &[0], test_uniforms()),
            Err(Refusal::ShapeMismatch("mask_r"))
        );
    }

    #[test]
    fn non_finite_uniforms_are_refused() {
        let src = pixel();
        let mut uniforms = test_uniforms();
        uniforms.mix_alpha = f32::NAN;
        assert_eq!(
            apply(&src, &src, &src, &[0], &[0], &[0], uniforms),
            Err(Refusal::ValueOutOfRange("uniforms"))
        );
    }

    #[test]
    fn reversed_thresholds_are_refused_without_panicking() {
        let src = pixel();
        let mut uniforms = test_uniforms();
        uniforms.diff_min_threshold = 2.0;
        assert_eq!(
            apply(&src, &src, &src, &[255], &[255], &[255], uniforms),
            Err(Refusal::ValueOutOfRange("uniforms"))
        );
        uniforms = test_uniforms();
        uniforms.high_freq_max_threshold = -2.0;
        assert_eq!(
            apply(&src, &src, &src, &[255], &[255], &[255], uniforms),
            Err(Refusal::ValueOutOfRange("uniforms"))
        );
    }

    #[test]
    fn zero_alpha_bypasses_positive_delta_clamp() {
        let src = pixel();
        let mut uniforms = test_uniforms();
        uniforms.high_freq_min_threshold = 0.1;
        assert_eq!(
            apply(&src, &src, &src, &[0], &[0], &[255], uniforms).unwrap(),
            src
        );
    }

    #[test]
    fn recovered_composite_produces_non_identity_pixels() {
        let src = Image::solid(1, 1, [51, 51, 51, 77]);
        let base = Image::solid(1, 1, [26, 26, 26, 0]);
        let out = apply(&src, &base, &src, &[255], &[255], &[255], test_uniforms()).unwrap();
        assert_eq!(out.pixel(0, 0), [76, 76, 76, 77]);
    }

    #[test]
    fn float_composite_matches_recovered_reference_without_quantization() {
        let out = composite_pixel(
            [0.2, 0.2, 0.2, 0.75],
            [0.1, 0.1, 0.1, 0.0],
            [0.2, 0.2, 0.2, 0.0],
            0.0,
            1.0,
            1.0,
            test_uniforms(),
        )
        .unwrap();
        for channel in 0..3 {
            assert!((out[channel] - 0.3).abs() < 1e-7);
        }
        assert_eq!(out[3].to_bits(), 0.75f32.to_bits());
        assert!(composite_pixel(
            [f32::NAN; 4],
            [0.1; 4],
            [0.2; 4],
            0.0,
            1.0,
            1.0,
            test_uniforms()
        )
        .is_err());
        assert!(
            composite_pixel([0.2; 4], [0.1; 4], [0.2; 4], 0.0, 2.0, 1.0, test_uniforms()).is_err()
        );
    }

    #[test]
    fn parallel_matches_serial_on_small_and_odd_row_bands() {
        for (width, height) in [(1, 1), (3, 2), (7, 3), (17, 13), (33, 19)] {
            let bytes = (width * height * 4) as usize;
            let image = |seed| Image::new(width, height,
                (0..bytes).map(|i| ((i * 37 + seed) % 256) as u8).collect()).unwrap();
            let source = image(7);
            let base = image(29);
            let residual = image(71);
            let mask = |seed| (0..source.pixel_count())
                .map(|i| ((i * 13 + seed) % 256) as u8).collect::<Vec<_>>();
            let (mask_r, mask_b, skin) = (mask(2), mask(37), mask(59));
            let uniforms = Uniforms { legal_alpha: 0.37, diff_factor: -0.24,
                diff_factor2: 0.19, ..test_uniforms() };
            assert_eq!(
                apply_parallel(&source, &base, &residual, &mask_r, &mask_b, &skin, uniforms),
                apply(&source, &base, &residual, &mask_r, &mask_b, &skin, uniforms),
                "row bands changed pixels for {width}x{height}"
            );
        }
    }

    #[test]
    fn parallel_preserves_serial_refusals_before_spawning() {
        let source = pixel();
        let mut bad_uniforms = test_uniforms();
        bad_uniforms.mix_alpha = f32::NAN;
        assert_eq!(apply_parallel(&source, &source, &source, &[255], &[255], &[255], bad_uniforms),
            Err(Refusal::ValueOutOfRange("uniforms")));
        assert_eq!(apply_parallel(&source, &source, &source, &[], &[255], &[255], test_uniforms()),
            Err(Refusal::ShapeMismatch("mask_r")));
    }
}
