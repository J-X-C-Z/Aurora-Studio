//! PF0 "spotless" (祛斑祛痘) blur-difference local retouch: CPU reference.
//!
//! PROVEN vendor law, read-only sources:
//!   `04_reference/legacy-root/Aurora/逆向工程/像素蛋糕/glsl/Spotless/SpotlessProcessNormal.frag`
//!   SHA256 `d23e69c49295a40cb1f9f4f005d0ed78cb128a3f8f3eab11dc6b9fad0c22f36d`
//!   `04_reference/legacy-root/Aurora/逆向工程/像素蛋糕/glsl/Spotless/SpotlessProcessWithSkinMask.frag`
//!   SHA256 `b049130c80ffc3af87b8b516ffd6f37eac89b1af4e05a34020beacb84170f568`
//!
//! Both fragments implement the same per-pixel law; only the skin weight source
//! differs: an inline HSV heuristic (`Normal`) or `s_skinmask_texture`
//! (`WithSkinMask`). Per pixel:
//!
//!   factor = clamp(face_mask * skin_mask * degree, 0, 1)      (Normal 97-99)
//!   dark  : blur-src  → distance → if > threshold, add clamped diff, mix
//!   light : src-blur  → distance → if > threshold, pull towards blur, mix
//!   alpha = src.a                                              (Normal 125)
//!
//! This is NOT the 21-slot composition in `blemish.rs`: the Spotless shaders
//! read `s_texture`, `s_blur_texture` and `s_facemask_texture`, and no acne,
//! blemish or repair model output appears anywhere in either file.
//!
//! Numeric uniform values are NOT recovered (pf0-contract FINDINGS section 7),
//! so [`SpotlessParams`] deliberately has no `Default`; callers must pass the
//! values observed at the vendor draw call. The fragment declares `mediump` /
//! `lowp` inputs, so a real GPU may quantize; this CPU path is the f32 reference
//! and makes no claim of byte-identity with a specific GPU.
//!
//! Not exported through `kernels::registry()`: `blemish.rs` already owns PF0's
//! `spotless` row, and a second PF0 row would break the catalog's unique-PF
//! assertion. This module adds a new pixel kernel only and changes no routing.

use crate::Refusal;

/// The five vendor uniforms plus the effect degree.
///
/// Field names map one-to-one onto the GLSL uniform names:
/// `u_dist_factor`, `u_dist_threshold`, `u_dark_diff_min_threshold`,
/// `u_dark_diff_max_threshold`, `u_white_diff_factor`, `u_degree`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpotlessParams {
    pub dist_factor: f32,
    pub dist_threshold: f32,
    pub dark_diff_min_threshold: f32,
    pub dark_diff_max_threshold: f32,
    pub white_diff_factor: f32,
    pub degree: f32,
}

impl SpotlessParams {
    /// All uniforms must be finite so the CPU result is deterministic. `degree`
    /// is not range-checked: the shader clamps the product, not the degree.
    pub fn validate(self) -> Result<(), Refusal> {
        let values = [
            self.dist_factor,
            self.dist_threshold,
            self.dark_diff_min_threshold,
            self.dark_diff_max_threshold,
            self.white_diff_factor,
            self.degree,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("spotless_params"));
        }
        Ok(())
    }
}

/// Borrowed f32 RGBA image, row-major, four floats per pixel.
///
/// Values are expected in the shader's normalized range, but are not clamped on
/// input: the vendor fragment receives whatever the previous stage produced.
#[derive(Clone, Copy, Debug)]
pub struct ImageF32<'a> {
    pub width: u32,
    pub height: u32,
    pub pixels: &'a [f32],
}

fn expected_len(width: u32, height: u32) -> Result<usize, Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("image"));
    }
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(Refusal::ShapeMismatch("image"))
}

fn length3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// GLSL `step(edge, x)`.
#[inline]
fn step(edge: f32, x: f32) -> f32 {
    if x < edge {
        0.0
    } else {
        1.0
    }
}

/// GLSL `mix(x, y, a)` with the spec formula `x*(1-a) + y*a`. The branch factors
/// used here are exactly 0.0 or 1.0, so selection is bit-exact.
#[inline]
fn mix(x: f32, y: f32, a: f32) -> f32 {
    x * (1.0 - a) + y * a
}

/// `rgb2hsv` from `SpotlessProcessNormal.frag` lines 38-48.
///
/// Saturation is deliberately hard-coded to 0.0 in the vendor shader ("for
/// optimization"), so only hue and value carry information. Reproduced step by
/// step with the shader's `K = (0, -1/3, 2/3, -1)` and `e = 1e-10`.
pub fn rgb2hsv(rgb: [f32; 3]) -> [f32; 3] {
    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    let s = step(b, g);
    let (p_x, p_y, p_z, p_w) = (
        mix(b, g, s),
        mix(g, b, s),
        mix(-1.0, 0.0, s),
        mix(2.0 / 3.0, -1.0 / 3.0, s),
    );
    let t = step(p_x, r);
    let q_x = mix(p_x, r, t);
    let q_y = p_y;
    let q_z = mix(p_w, p_z, t);
    let q_w = mix(r, p_x, t);
    let d = q_x - q_w.min(q_y);
    let e = 1.0e-10f32;
    let hue = (q_z + (q_w - q_y) / (6.0 * d + e)).abs();
    [hue, 0.0, q_x]
}

/// Skin weight from an already computed hue/value pair.
///
/// `SpotlessProcessNormal.frag` lines 50-77. The hard band `0.18 <= hue <= 0.89`
/// and `value <= 0.2` returns 0; the three soft ramps are evaluated in vendor
/// order. Keeping this split lets tests hit the thresholds exactly.
pub fn skin_value_from_hue_value(hue: f32, value: f32) -> f32 {
    let mut skin_value = 1.0f32;
    if (0.18 <= hue && hue <= 0.89) || value <= 0.2 {
        skin_value = 0.0;
    } else if 0.16 < hue && hue < 0.18 {
        skin_value = skin_value.min((0.18 - hue) * 50.0); // 1.0 / 0.02
    } else if 0.89 < hue && hue < 0.91 {
        skin_value = skin_value.min(1.0 - (0.91 - hue) * 50.0); // 1.0 / 0.02
    } else if 0.2 < value && hue < 0.3 {
        skin_value = skin_value.min(1.0 - (0.3 - value) * 10.0); // 1.0 / 0.1
    }
    skin_value
}

/// Inline HSV skin heuristic for one source RGB.
pub fn skin_value(rgb: [f32; 3]) -> f32 {
    let hsv = rgb2hsv(rgb);
    skin_value_from_hue_value(hsv[0], hsv[2])
}

/// One pixel of the vendor fragment. `skin_mask` selects the
/// `WithSkinMask` path when `Some`; `None` uses the inline heuristic.
///
/// The `factor > 0.0` gate (Normal 101) is kept explicit: with `factor == 0`
/// the output is exactly the input, including subnormal/negative values.
pub fn shade(
    src: [f32; 4],
    blur: [f32; 3],
    face_mask: f32,
    skin_mask: Option<f32>,
    params: &SpotlessParams,
) -> [f32; 4] {
    let mut res = [src[0], src[1], src[2]];

    if face_mask > 0.0 {
        let skin = skin_mask.unwrap_or_else(|| skin_value([src[0], src[1], src[2]]));
        // `factor = min(1.0, max(0.0, face_mask * skin * degree))`.
        let factor = (face_mask * skin * params.degree).max(0.0).min(1.0);

        if factor > 0.0 {
            // Dark branch: `max(blur - src, 0)` (Normal 104).
            let dark = [
                (blur[0] - src[0]).max(0.0),
                (blur[1] - src[1]).max(0.0),
                (blur[2] - src[2]).max(0.0),
            ];
            let dist = length3(dark) * params.dist_factor;

            if dist > params.dist_threshold {
                // Normal 108-110: componentwise clamp, then add and cap at 1.
                for channel in 0..3 {
                    let diff = (blur[channel] - src[channel])
                        .max(params.dark_diff_min_threshold)
                        .min(params.dark_diff_max_threshold);
                    let target = (src[channel] + diff).min(1.0);
                    res[channel] = mix(src[channel], target, factor);
                }
            } else {
                // Light branch: `max(src - blur, 0)` (Normal 114).
                let light = [
                    (src[0] - blur[0]).max(0.0),
                    (src[1] - blur[1]).max(0.0),
                    (src[2] - blur[2]).max(0.0),
                ];
                let inv_dist = length3(light) * params.dist_factor;
                if inv_dist > params.dist_threshold {
                    // Normal 117-119: pull towards the blur, no clamp.
                    for channel in 0..3 {
                        let diff = src[channel] - blur[channel];
                        let target = blur[channel] + diff * params.white_diff_factor;
                        res[channel] = mix(src[channel], target, factor);
                    }
                }
            }
        }
    }

    // Normal 125: `gl_FragColor = vec4(res.rgb, src.a)`.
    [res[0], res[1], res[2], src[3]]
}

/// Apply the Spotless fragment to a whole image.
///
/// `skin_mask == Some` selects the `WithSkinMask` fragment; `None` selects the
/// inline HSV heuristic. All validation completes before `out` is touched, and
/// the hot loop allocates nothing. `out` must not alias `src.pixels` or
/// `blur.pixels` (Rust's borrow rules require a separate buffer anyway).
pub fn apply(
    src: ImageF32<'_>,
    blur: ImageF32<'_>,
    face_mask: &[f32],
    skin_mask: Option<&[f32]>,
    params: SpotlessParams,
    out: &mut [f32],
) -> Result<(), Refusal> {
    let bytes = expected_len(src.width, src.height)?;
    if src.pixels.len() != bytes {
        return Err(Refusal::ShapeMismatch("image"));
    }
    if blur.width != src.width || blur.height != src.height || blur.pixels.len() != bytes {
        return Err(Refusal::ShapeMismatch("blur"));
    }
    let pixels = bytes / 4;
    if face_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("face_mask"));
    }
    if let Some(mask) = skin_mask {
        if mask.len() != pixels {
            return Err(Refusal::ShapeMismatch("skin_mask"));
        }
    }
    if out.len() != bytes {
        return Err(Refusal::ShapeMismatch("output"));
    }
    params.validate()?;

    for pixel in 0..pixels {
        let index = pixel * 4;
        let src_rgba = [
            src.pixels[index],
            src.pixels[index + 1],
            src.pixels[index + 2],
            src.pixels[index + 3],
        ];
        let blur_rgb = [
            blur.pixels[index],
            blur.pixels[index + 1],
            blur.pixels[index + 2],
        ];
        let skin = skin_mask.map(|mask| mask[pixel]);
        let result = shade(src_rgba, blur_rgb, face_mask[pixel], skin, &params);
        out[index..index + 4].copy_from_slice(&result);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Fail-closed four-plane input contract (PF0 binding).
//
// The vendor `WithSkinMask` fragment samples exactly four planes:
//   d  = s_texture          (rgba, unit 0)
//   bZ = s_blur_texture     (rgb,  unit 1)
//   bk = s_facemask_texture (r,    unit 2)
//   bi = s_skinmask_texture (r,    unit 3)
// Each is bound through `GLSpotlessProcessBaseFilter::BindTextures`
// (pix-worker 0x57F53F0 / 0x57F55B0) and read with `.stp` / `.s` swizzles.
//
// The vendor draw path gates on a live client (`cell+0x48`), a non-empty face
// list, a positive ROI and a per-face degree above 0.01; it performs no
// job/ticket/hash credential check before draw (see `PF0-BINDING` FINDINGS
// section 4). This module therefore does not invent such a check either, but it
// does refuse any missing, mismatched or non-finite plane instead of silently
// substituting a constant plane. That silent fallback is exactly the behaviour
// this contract removes.
// ---------------------------------------------------------------------------

/// Channel layout a plane declares to match the vendor sampler swizzle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneChannels {
    /// One float per pixel; sampled as `.s` (red only).
    R,
    /// Three floats per pixel; sampled as `.stp` (rgb).
    Rgb,
    /// Four floats per pixel; rgb sampled as `.stp`, alpha copied to output.
    Rgba,
}

impl PlaneChannels {
    pub const fn count(self) -> usize {
        match self {
            PlaneChannels::R => 1,
            PlaneChannels::Rgb => 3,
            PlaneChannels::Rgba => 4,
        }
    }
}

/// One explicitly supplied same-job plane: extent, channel layout and f32 data.
///
/// Values are expected in the shader's normalized range but are not clamped on
/// input; only finiteness is enforced. A zero extent or empty payload means the
/// plane is missing and is a refusal, never a synthesized constant plane.
#[derive(Clone, Copy, Debug)]
pub struct Plane<'a> {
    pub width: u32,
    pub height: u32,
    pub channels: PlaneChannels,
    pub pixels: &'a [f32],
}

/// Named refusal tags for the four recovered PF0 samplers.
pub const PF0_SOURCE: &str = "PF0_s_texture";
pub const PF0_BLUR: &str = "PF0_s_blur_texture";
pub const PF0_FACE_MASK: &str = "PF0_s_facemask_texture";
pub const PF0_SKIN_MASK: &str = "PF0_s_skinmask_texture";
pub const PF0_SOURCE_CHANNELS: &str = "PF0_s_texture_channels";
pub const PF0_BLUR_CHANNELS: &str = "PF0_s_blur_texture_channels";
pub const PF0_FACE_MASK_CHANNELS: &str = "PF0_s_facemask_texture_channels";
pub const PF0_SKIN_MASK_CHANNELS: &str = "PF0_s_skinmask_texture_channels";
pub const PF0_OUTPUT: &str = "PF0_output";

/// The complete four-plane input for the vendor `WithSkinMask` fragment.
///
/// Every plane is mandatory: there is no `Option` and no default. Callers that
/// want the `Normal` fragment's inline HSV heuristic use [`apply`] with
/// `skin_mask == None`; callers that claim the skin-mask path must supply the
/// plane explicitly here.
#[derive(Clone, Copy, Debug)]
pub struct SpotlessInputs<'a> {
    pub source: Plane<'a>,
    pub blur: Plane<'a>,
    pub face_mask: Plane<'a>,
    pub skin_mask: Plane<'a>,
    pub params: SpotlessParams,
}

fn check_plane(
    plane: Plane<'_>,
    name: &'static str,
    channels_name: &'static str,
    expected: PlaneChannels,
) -> Result<usize, Refusal> {
    if plane.width == 0 || plane.height == 0 || plane.pixels.is_empty() {
        return Err(Refusal::MissingInput(name));
    }
    if plane.channels != expected {
        return Err(Refusal::ShapeMismatch(channels_name));
    }
    let expected_len = (plane.width as usize)
        .checked_mul(plane.height as usize)
        .and_then(|pixels| pixels.checked_mul(expected.count()))
        .ok_or(Refusal::ShapeMismatch(name))?;
    if plane.pixels.len() != expected_len {
        return Err(Refusal::ShapeMismatch(name));
    }
    if plane.pixels.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange(name));
    }
    Ok(expected_len / expected.count())
}

impl SpotlessInputs<'_> {
    /// Validate the four planes, their common extent and the uniforms.
    ///
    /// Returns the pixel count on success. Every refusal is named after the
    /// offending sampler so a missing or malformed plane cannot be confused
    /// with a pixel-law failure. The check runs to completion before any output
    /// buffer is touched.
    pub fn validate(&self) -> Result<usize, Refusal> {
        let pixels = check_plane(
            self.source,
            PF0_SOURCE,
            PF0_SOURCE_CHANNELS,
            PlaneChannels::Rgba,
        )?;
        check_plane(self.blur, PF0_BLUR, PF0_BLUR_CHANNELS, PlaneChannels::Rgb)?;
        check_plane(
            self.face_mask,
            PF0_FACE_MASK,
            PF0_FACE_MASK_CHANNELS,
            PlaneChannels::R,
        )?;
        check_plane(
            self.skin_mask,
            PF0_SKIN_MASK,
            PF0_SKIN_MASK_CHANNELS,
            PlaneChannels::R,
        )?;

        // All four planes share one logical extent. The vendor samples every
        // plane with the same `g` uv, so a differing extent is a producer bug,
        // not something to resample silently here.
        for (plane, name) in [
            (self.blur, PF0_BLUR),
            (self.face_mask, PF0_FACE_MASK),
            (self.skin_mask, PF0_SKIN_MASK),
        ] {
            if plane.width != self.source.width || plane.height != self.source.height {
                return Err(Refusal::ShapeMismatch(name));
            }
        }

        self.params.validate()?;
        Ok(pixels)
    }
}

/// Apply the `WithSkinMask` fragment to explicitly supplied four-plane inputs.
///
/// Unlike [`apply`], `skin_mask` is mandatory and accents the input contract:
/// a missing plane, a mismatched extent/channel layout, a non-finite value or a
/// non-finite uniform refuses before `out` is written. No plane is synthesized.
pub fn apply_inputs(inputs: SpotlessInputs<'_>, out: &mut [f32]) -> Result<(), Refusal> {
    let pixels = inputs.validate()?;
    if out.len() != pixels * 4 {
        return Err(Refusal::ShapeMismatch(PF0_OUTPUT));
    }

    for pixel in 0..pixels {
        let s = pixel * 4;
        let b = pixel * 3;
        let src_rgba = [
            inputs.source.pixels[s],
            inputs.source.pixels[s + 1],
            inputs.source.pixels[s + 2],
            inputs.source.pixels[s + 3],
        ];
        let blur_rgb = [
            inputs.blur.pixels[b],
            inputs.blur.pixels[b + 1],
            inputs.blur.pixels[b + 2],
        ];
        let face = inputs.face_mask.pixels[pixel];
        let skin = inputs.skin_mask.pixels[pixel];
        let result = shade(src_rgba, blur_rgb, face, Some(skin), &inputs.params);
        out[s..s + 4].copy_from_slice(&result);
    }
    Ok(())
}

/// Vendor per-face draw gate recovered from pix-worker `0x3F25661`
/// (`ucomiss xmm6, [0x8BAD294]`; `0x8BAD294` = 0.01f).
///
/// Faces whose computed degree is not strictly above this value are skipped
/// before the filter runs. Exposed so callers can reproduce the same gate
/// instead of drawing a zero-strength pass.
pub const DRAW_DEGREE_GATE: f32 = 0.01;

/// True when a per-face degree passes the recovered vendor gate.
pub fn passes_draw_gate(degree: f32) -> bool {
    degree > DRAW_DEGREE_GATE
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SpotlessParams {
        SpotlessParams {
            dist_factor: 2.0,
            dist_threshold: 0.1,
            dark_diff_min_threshold: 0.0,
            dark_diff_max_threshold: 1.0,
            white_diff_factor: 0.5,
            degree: 1.0,
        }
    }

    fn near(actual: f32, expected: f32) -> bool {
        (actual - expected).abs() <= 1e-5
    }

    #[test]
    fn hsv_matches_the_vendor_formula_and_forces_zero_saturation() {
        let cases = [
            ([1.0f32, 0.0, 0.0], 0.0f32, 1.0f32),          // red
            ([0.0, 1.0, 0.0], 1.0 / 3.0, 1.0),             // green
            ([0.0, 0.0, 1.0], 2.0 / 3.0, 1.0),             // blue
            ([1.0, 1.0, 0.0], 1.0 / 6.0, 1.0),             // yellow
            ([1.0, 0.0, 1.0], 5.0 / 6.0, 1.0),             // magenta
            ([0.5, 0.5, 0.5], 0.0, 0.5),                   // gray
            ([0.0, 0.0, 0.0], 0.0, 0.0),                   // black
        ];
        for (rgb, hue, value) in cases {
            let hsv = rgb2hsv(rgb);
            assert!(near(hsv[0], hue), "hue for {rgb:?}: {} != {hue}", hsv[0]);
            assert_eq!(hsv[1], 0.0, "vendor forces saturation to 0");
            assert!(near(hsv[2], value), "value for {rgb:?}: {} != {value}", hsv[2]);
        }
    }

    #[test]
    fn skin_heuristic_boundaries() {
        // Hard band, inclusive on both ends; dark pixels are never skin.
        assert_eq!(skin_value_from_hue_value(0.18, 0.9), 0.0);
        assert_eq!(skin_value_from_hue_value(0.89, 0.9), 0.0);
        assert_eq!(skin_value_from_hue_value(0.5, 0.9), 0.0);
        assert_eq!(skin_value_from_hue_value(0.5, 0.2), 0.0);
        assert_eq!(skin_value_from_hue_value(0.5, 0.0), 0.0);

        // Low-hue soft ramp: 0.16 and 0.18 are strict boundaries.
        assert_eq!(skin_value_from_hue_value(0.16, 0.9), 1.0);
        assert!(near(skin_value_from_hue_value(0.17, 0.9), 0.5));
        assert!(near(skin_value_from_hue_value(0.179, 0.9), 0.05));

        // High-hue soft ramp: 0.89 and 0.91 are strict boundaries.
        assert_eq!(skin_value_from_hue_value(0.91, 0.9), 1.0);
        assert!(near(skin_value_from_hue_value(0.90, 0.9), 0.5));
        assert!(near(skin_value_from_hue_value(0.901, 0.9), 0.55));

        // Value ramp only applies outside the hard hue band and hue < 0.3.
        assert_eq!(skin_value_from_hue_value(0.1, 0.2), 0.0);
        assert!(near(skin_value_from_hue_value(0.1, 0.25), 0.5));
        assert!(near(skin_value_from_hue_value(0.1, 0.21), 0.1));
        assert_eq!(skin_value_from_hue_value(0.95, 0.25), 1.0); // outside every ramp
        // Inside the hard band the value ramp must not apply.
        assert_eq!(skin_value_from_hue_value(0.2, 0.25), 0.0);
        assert_eq!(skin_value_from_hue_value(0.35, 0.25), 0.0);
    }

    #[test]
    fn skin_value_end_to_end() {
        assert_eq!(skin_value([1.0, 0.0, 0.0]), 1.0); // hue 0
        assert_eq!(skin_value([0.0, 1.0, 0.0]), 0.0); // hue 1/3
        assert_eq!(skin_value([0.0, 0.0, 1.0]), 0.0); // hue 2/3
        assert_eq!(skin_value([0.1, 0.1, 0.1]), 0.0); // value <= 0.2
        assert!(near(skin_value([1.0, 1.0, 0.0]), 0.666_666_7)); // hue 1/6 ramp
    }

    #[test]
    fn dark_branch_adds_clamped_blur_difference() {
        let p = params();
        let src = [0.25, 0.25, 0.25, 0.5];
        let blur = [0.5, 0.5, 0.5];
        let out = shade(src, blur, 1.0, Some(1.0), &p);
        assert_eq!([out[0], out[1], out[2]], [0.5, 0.5, 0.5]);
        assert_eq!(out[3], 0.5, "alpha comes from src, not blur");

        // dark_diff_min raises the clamped difference.
        let mut lower = p;
        lower.dark_diff_min_threshold = 0.4;
        let out = shade(src, blur, 1.0, Some(1.0), &lower);
        assert_eq!(out[0], 0.25f32 + 0.4f32);

        // dark_diff_max caps it.
        let mut upper = p;
        upper.dark_diff_max_threshold = 0.125;
        let out = shade(src, blur, 1.0, Some(1.0), &upper);
        assert_eq!(out[0], 0.25f32 + 0.125f32);

        // `min(src + diff, 1.0)` caps the result.
        let bright = shade([0.75, 0.5, 0.5, 1.0], [1.0, 0.5, 0.5], 1.0, Some(1.0), &p);
        assert_eq!(bright[0], 1.0);
    }

    #[test]
    fn light_branch_pulls_towards_blur() {
        let p = params();
        let src = [0.5, 0.5, 0.5, 0.25];
        let blur = [0.25, 0.25, 0.25];
        let out = shade(src, blur, 1.0, Some(1.0), &p);
        // ref = 0.25 + (0.5 - 0.25) * 0.5 = 0.375
        assert_eq!([out[0], out[1], out[2]], [0.375, 0.375, 0.375]);
        assert_eq!(out[3], 0.25);

        let mut full = p;
        full.white_diff_factor = 2.0;
        let out = shade(src, blur, 1.0, Some(1.0), &full);
        assert_eq!(out[0], 0.75);

        let mut zero = p;
        zero.white_diff_factor = 0.0;
        let out = shade(src, blur, 1.0, Some(1.0), &zero);
        assert_eq!(out[0], 0.25);
    }

    #[test]
    fn dark_branch_takes_precedence_over_light_branch() {
        let p = params();
        let src = [0.25, 0.4, 0.5, 1.0];
        let blur = [0.5, 0.0, 0.5];
        // Both distances exceed 0.1, but the vendor evaluates dark first.
        let out = shade(src, blur, 1.0, Some(1.0), &p);
        assert_eq!([out[0], out[1], out[2]], [0.5, 0.4, 0.5]);
        // The light answer would have been [0.5, 0.2, 0.5].
        assert_ne!(out[1], 0.2);
    }

    #[test]
    fn distance_threshold_is_strict() {
        let mut p = params();
        p.dist_threshold = 0.25;
        let src = [0.5, 0.5, 0.5, 1.0];
        let blur = [0.625, 0.5, 0.5];
        // dark dist = 0.125 * 2.0 == threshold, so `>` does not fire; the light
        // distance is zero. The pixel is unchanged.
        let out = shade(src, blur, 1.0, Some(1.0), &p);
        assert_eq!(out, src);
    }

    #[test]
    fn factor_is_clamped_and_gates_the_pixel() {
        let src = [0.25, 0.25, 0.25, 0.5];
        let blur = [0.5, 0.5, 0.5];

        // face * skin * degree above 1 clamps to a full application.
        let mut p = params();
        p.degree = 4.0;
        let out = shade(src, blur, 2.0, Some(2.0), &p);
        assert_eq!([out[0], out[1], out[2]], [0.5, 0.5, 0.5]);

        // Any non-positive product leaves the pixel untouched.
        let mut p = params();
        p.degree = -1.0;
        assert_eq!(shade(src, blur, 1.0, Some(1.0), &p), src);
        let p = params();
        assert_eq!(shade(src, blur, 0.0, Some(1.0), &p), src);
        assert_eq!(shade(src, blur, -1.0, Some(1.0), &p), src);
        assert_eq!(shade(src, blur, 1.0, Some(0.0), &p), src);
    }

    #[test]
    fn skin_mask_bypasses_or_overrides_the_inline_heuristic() {
        let p = params();
        // Green is not skin by the heuristic (hue 1/3), so nothing happens.
        let green = [0.0, 0.5, 0.0, 1.0];
        let blur = [0.25, 0.5, 0.25];
        let inline = shade(green, blur, 1.0, None, &p);
        assert_eq!(inline, green);

        // The WithSkinMask fragment trusts the supplied plane instead.
        let supplied = shade(green, blur, 1.0, Some(1.0), &p);
        assert_eq!([supplied[0], supplied[1], supplied[2]], [0.25, 0.5, 0.25]);

        // Conversely a red pixel the heuristic accepts stays put at skin 0.
        let red = [0.5, 0.0, 0.0, 1.0];
        let red_blur = [0.75, 0.0, 0.0];
        assert_ne!(shade(red, red_blur, 1.0, None, &p), red);
        assert_eq!(shade(red, red_blur, 1.0, Some(0.0), &p), red);
    }

    #[test]
    fn zero_degree_is_an_exact_identity() {
        let src_pixels = [
            0.1f32, 0.2, 0.3, 0.4,
            0.9, 0.8, 0.7, 0.6,
            0.0, 0.5, 1.0, 0.25,
            1.0, 0.0, 0.5, 0.75,
        ];
        let blur_pixels = [
            0.9f32, 0.9, 0.9, 1.0,
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.5,
        ];
        let src = ImageF32 { width: 2, height: 2, pixels: &src_pixels };
        let blur = ImageF32 { width: 2, height: 2, pixels: &blur_pixels };
        let face = [1.0f32; 4];
        let mut out = [f32::NAN; 16];

        let mut p = params();
        p.degree = 0.0;
        apply(src, blur, &face, None, p, &mut out).unwrap();
        assert_eq!(out, src_pixels);
    }

    #[test]
    fn apply_validates_every_input_before_writing() {
        let src_pixels = [0.0f32; 8];
        let blur_pixels = [0.0f32; 8];
        let face = [1.0f32, 1.0f32];
        let mut out = [9.0f32; 8];
        let src = ImageF32 { width: 2, height: 1, pixels: &src_pixels };
        let blur = ImageF32 { width: 2, height: 1, pixels: &blur_pixels };

        apply(src, blur, &face, None, params(), &mut out).unwrap();
        assert_eq!(out, [0.0; 8], "zero source, zero blur, alpha 0 is identity");

        out = [9.0; 8];
        assert_eq!(
            apply(src, blur, &[1.0], None, params(), &mut out),
            Err(Refusal::ShapeMismatch("face_mask"))
        );
        assert_eq!(
            apply(src, blur, &face, Some(&[1.0]), params(), &mut out),
            Err(Refusal::ShapeMismatch("skin_mask"))
        );
        assert_eq!(
            apply(src, blur, &face, None, params(), &mut out[..4]),
            Err(Refusal::ShapeMismatch("output"))
        );
        let wide = ImageF32 { width: 1, height: 2, pixels: &blur_pixels };
        assert_eq!(
            apply(src, wide, &face, None, params(), &mut out),
            Err(Refusal::ShapeMismatch("blur"))
        );
        let short = ImageF32 { width: 2, height: 1, pixels: &src_pixels[..4] };
        assert_eq!(
            apply(short, blur, &face, None, params(), &mut out),
            Err(Refusal::ShapeMismatch("image"))
        );
        let empty = ImageF32 { width: 0, height: 0, pixels: &[] };
        assert_eq!(
            apply(empty, empty, &[], None, params(), &mut out),
            Err(Refusal::ShapeMismatch("image"))
        );
        let mut bad = params();
        bad.degree = f32::NAN;
        assert_eq!(
            apply(src, blur, &face, None, bad, &mut out),
            Err(Refusal::ValueOutOfRange("spotless_params"))
        );
        assert_eq!(out, [9.0; 8], "refusals must not touch the output buffer");
    }

    #[test]
    fn apply_processes_each_pixel_with_its_own_masks() {
        let src_pixels = [
            0.25f32, 0.25, 0.25, 0.5,
            0.0, 0.5, 0.0, 1.0,
        ];
        let blur_pixels = [
            0.5f32, 0.5, 0.5, 0.9,
            0.25, 0.5, 0.25, 0.0,
        ];
        let src = ImageF32 { width: 2, height: 1, pixels: &src_pixels };
        let blur = ImageF32 { width: 2, height: 1, pixels: &blur_pixels };
        let face = [1.0f32, 1.0];
        let skin = [1.0f32, 1.0];
        let mut out = [0.0f32; 8];

        apply(src, blur, &face, Some(&skin), params(), &mut out).unwrap();
        assert_eq!(out, [0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.25, 1.0]);
    }

    #[test]
    fn params_reject_only_non_finite_values() {
        let mut p = params();
        p.dist_factor = f32::INFINITY;
        assert_eq!(p.validate(), Err(Refusal::ValueOutOfRange("spotless_params")));
        let mut p = params();
        p.white_diff_factor = f32::NEG_INFINITY;
        assert_eq!(p.validate(), Err(Refusal::ValueOutOfRange("spotless_params")));
        let mut p = params();
        p.dark_diff_min_threshold = f32::NAN;
        assert_eq!(p.validate(), Err(Refusal::ValueOutOfRange("spotless_params")));
        // Out-of-range but finite degrees are legal; the product is clamped.
        let mut p = params();
        p.degree = 1.0e9;
        assert_eq!(p.validate(), Ok(()));
    }

    // --- four-plane fail-closed contract (PF0 binding) ---------------------

    fn plane<'a>(width: u32, height: u32, channels: PlaneChannels, pixels: &'a [f32]) -> Plane<'a> {
        Plane { width, height, channels, pixels }
    }

    fn full_inputs<'a>(
        src: &'a [f32],
        blur: &'a [f32],
        face: &'a [f32],
        skin: &'a [f32],
    ) -> SpotlessInputs<'a> {
        SpotlessInputs {
            source: plane(2, 1, PlaneChannels::Rgba, src),
            blur: plane(2, 1, PlaneChannels::Rgb, blur),
            face_mask: plane(2, 1, PlaneChannels::R, face),
            skin_mask: plane(2, 1, PlaneChannels::R, skin),
            params: params(),
        }
    }

    #[test]
    fn four_planes_complete_match_the_reference_apply() {
        let src_pixels = [0.25f32, 0.25, 0.25, 0.5, 0.0, 0.5, 0.0, 1.0];
        // `apply` takes an RGBA `ImageF32` for the blur plane; the shader only
        // reads its rgb. The four-plane contract declares the true 3-channel
        // layout, so strip the unused alpha for `apply_inputs`.
        let blur_rgba = [0.5f32, 0.5, 0.5, 0.0, 0.25, 0.5, 0.25, 0.0];
        let blur_rgb = [0.5f32, 0.5, 0.5, 0.25, 0.5, 0.25];
        let face = [1.0f32, 1.0];
        let skin = [1.0f32, 1.0];

        let mut expected = [0.0f32; 8];
        apply(
            ImageF32 { width: 2, height: 1, pixels: &src_pixels },
            ImageF32 { width: 2, height: 1, pixels: &blur_rgba },
            &face,
            Some(&skin),
            params(),
            &mut expected,
        )
        .unwrap();

        let mut actual = [0.0f32; 8];
        apply_inputs(full_inputs(&src_pixels, &blur_rgb, &face, &skin), &mut actual).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual, [0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.25, 1.0]);
    }

    #[test]
    fn each_missing_plane_is_named_in_the_refusal() {
        let src_pixels = [0.25f32; 8];
        let blur_pixels = [0.25f32; 6];
        let mask = [1.0f32; 2];
        let hello = full_inputs(&src_pixels, &blur_pixels, &mask, &mask);
        let mut out = [9.0f32; 8];

        // source: empty payload / zero extent.
        let mut missing = hello;
        missing.source = plane(0, 0, PlaneChannels::Rgba, &[]);
        assert_eq!(apply_inputs(missing, &mut out), Err(Refusal::MissingInput(PF0_SOURCE)));

        let mut missing = hello;
        missing.blur = plane(2, 1, PlaneChannels::Rgb, &[]);
        assert_eq!(apply_inputs(missing, &mut out), Err(Refusal::MissingInput(PF0_BLUR)));

        let mut missing = hello;
        missing.face_mask = plane(2, 1, PlaneChannels::R, &[]);
        assert_eq!(
            apply_inputs(missing, &mut out),
            Err(Refusal::MissingInput(PF0_FACE_MASK))
        );

        let mut missing = hello;
        missing.skin_mask = plane(0, 1, PlaneChannels::R, &mask[..0]);
        assert_eq!(
            apply_inputs(missing, &mut out),
            Err(Refusal::MissingInput(PF0_SKIN_MASK))
        );
        assert_eq!(out, [9.0; 8], "missing-plane refusals must not touch output");
    }

    #[test]
    fn extent_mismatch_is_refused_by_plane_name() {
        let src_pixels = [0.25f32; 8];
        let blur_pixels = [0.25f32; 6];
        let mask = [1.0f32; 2];
        let hello = full_inputs(&src_pixels, &blur_pixels, &mask, &mask);
        let mut out = [9.0f32; 8];

        // Same pixel count but a different declared extent (1x2 vs 2x1).
        let mut bad = hello;
        bad.blur = plane(1, 2, PlaneChannels::Rgb, &blur_pixels);
        assert_eq!(apply_inputs(bad, &mut out), Err(Refusal::ShapeMismatch(PF0_BLUR)));

        // A plane whose extent differs and whose buffer is explicitly wrong.
        let mut bad = hello;
        bad.face_mask = plane(3, 1, PlaneChannels::R, &mask);
        assert_eq!(
            apply_inputs(bad, &mut out),
            Err(Refusal::ShapeMismatch(PF0_FACE_MASK))
        );

        let mut bad = hello;
        bad.skin_mask = plane(2, 2, PlaneChannels::R, &[1.0f32; 4]);
        assert_eq!(
            apply_inputs(bad, &mut out),
            Err(Refusal::ShapeMismatch(PF0_SKIN_MASK))
        );
        assert_eq!(out, [9.0; 8]);
    }

    #[test]
    fn channel_layout_mismatch_is_refused_by_plane_name() {
        let src_pixels = [0.25f32; 8];
        let blur_pixels = [0.25f32; 6];
        let mask = [1.0f32; 2];
        let hello = full_inputs(&src_pixels, &blur_pixels, &mask, &mask);
        let mut out = [9.0f32; 8];

        let mut bad = hello;
        bad.source = plane(2, 1, PlaneChannels::Rgb, &src_pixels);
        assert_eq!(
            apply_inputs(bad, &mut out),
            Err(Refusal::ShapeMismatch(PF0_SOURCE_CHANNELS))
        );

        let mut bad = hello;
        bad.blur = plane(2, 1, PlaneChannels::Rgba, &blur_pixels);
        assert_eq!(
            apply_inputs(bad, &mut out),
            Err(Refusal::ShapeMismatch(PF0_BLUR_CHANNELS))
        );

        let mut bad = hello;
        bad.face_mask = plane(2, 1, PlaneChannels::Rgb, &[1.0f32; 6]);
        assert_eq!(
            apply_inputs(bad, &mut out),
            Err(Refusal::ShapeMismatch(PF0_FACE_MASK_CHANNELS))
        );

        let mut bad = hello;
        bad.skin_mask = plane(2, 1, PlaneChannels::Rgb, &[1.0f32; 6]);
        assert_eq!(
            apply_inputs(bad, &mut out),
            Err(Refusal::ShapeMismatch(PF0_SKIN_MASK_CHANNELS))
        );
    }

    #[test]
    fn non_finite_plane_values_are_refused() {
        let src_pixels = [0.25f32; 8];
        let blur_pixels = [0.25f32; 6];
        let mask = [1.0f32; 2];
        let hello = full_inputs(&src_pixels, &blur_pixels, &mask, &mask);
        let mut out = [9.0f32; 8];

        let bad_blur = [0.25f32, f32::NAN, 0.5, 0.25, 0.5, 0.25];
        let mut bad = hello;
        bad.blur = plane(2, 1, PlaneChannels::Rgb, &bad_blur);
        assert_eq!(
            apply_inputs(bad, &mut out),
            Err(Refusal::ValueOutOfRange(PF0_BLUR))
        );

        let bad_skin = [1.0f32, f32::INFINITY];
        let mut bad = hello;
        bad.skin_mask = plane(2, 1, PlaneChannels::R, &bad_skin);
        assert_eq!(
            apply_inputs(bad, &mut out),
            Err(Refusal::ValueOutOfRange(PF0_SKIN_MASK))
        );
    }

    #[test]
    fn non_finite_uniforms_are_refused_after_planes_pass() {
        let src_pixels = [0.25f32; 8];
        let blur_pixels = [0.25f32; 6];
        let mask = [1.0f32; 2];
        let mut inputs = full_inputs(&src_pixels, &blur_pixels, &mask, &mask);
        inputs.params.degree = f32::NAN;
        let mut out = [9.0f32; 8];
        assert_eq!(
            apply_inputs(inputs, &mut out),
            Err(Refusal::ValueOutOfRange("spotless_params"))
        );
        assert_eq!(out, [9.0; 8]);
    }

    #[test]
    fn output_length_must_match_pixel_count() {
        let src_pixels = [0.25f32; 8];
        let blur_pixels = [0.25f32; 6];
        let mask = [1.0f32; 2];
        let hello = full_inputs(&src_pixels, &blur_pixels, &mask, &mask);
        let mut short = [0.0f32; 4];
        assert_eq!(
            apply_inputs(hello, &mut short),
            Err(Refusal::ShapeMismatch(PF0_OUTPUT))
        );
    }

    #[test]
    fn draw_gate_matches_the_recovered_vendor_threshold() {
        assert!(!passes_draw_gate(0.0));
        assert!(!passes_draw_gate(DRAW_DEGREE_GATE), "gate is strict `>`");
        assert!(!passes_draw_gate(f32::NEG_INFINITY));
        assert!(passes_draw_gate(DRAW_DEGREE_GATE + 1.0e-6));
        assert!(passes_draw_gate(1.0));
    }
}
