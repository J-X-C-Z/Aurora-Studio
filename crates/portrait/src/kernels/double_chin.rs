//! PF90034 double-chin warp: the AI state stores `std::vector<cv::Mat>` where
//! each Mat is a dense 2-channel float field (`0x4AB20F0` → `0x34B9B10` →
//! `0x4E69B90`, `cv::remap` with INTER_LINEAR + BORDER_REFLECT_101).
//!
//! Whether the stored channels are `(dx, dy)` or absolute `(x', y')`, and the
//! field scale, are not proven, so both are explicit caller inputs.
//!
//! The recovered parameter producer is per-face binary32 only:
//! `W_geometry = f32(PF30188 * PF90034)` and `W_shadow = f32(PF90034 * PF30189)`
//! (worker `0x7AEEA97`/`0x7AEEA9C`). There is no clamp, `/100`, or `0..1`
//! normalization in the fixed PE. A face absent from the vendor
//! `std::map<face_index, node>` yields `0.0f` (`0x7AEE963..0x7AEE987`).
//! PF30209 (jawline) is written to `stage+0x3A0/+0x3B8` but the stage copy at
//! `0x7AF2FD0` does not forward those slots, so it never enters this producer.

use crate::{Access, EffectMeta, Image, Refusal};

pub const META: &[EffectMeta] = &[EffectMeta {
    pf: 90034,
    key: "removeDoubleChin",
    label: "祛双下巴",
    access: Access::Cpu,
    inputs: &[
        "native PF90034 per-face value",
        "native PF30188 per-face value",
        "native PF30189 per-face value",
        "face parsing channels 10/11/14/6/8/9",
        "dense 2-channel shift field",
        "mode (Flow | Absolute)",
        "scale",
    ],
}];

/// Vendor execution gate at `0x8BFC350`, loaded with `movsd` as binary64
/// (`2D431CEBE2361A3F`). Values are widened from the f32 products before
/// `ucomisd` at `0x7E47324` / `0x7E47331`.
pub const WEIGHT_GATE: f64 = 1e-4;

/// One face's recovered binary32 weights, written to `node+0x7B0` (geometry)
/// and `node+0x7AC` (shadow).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weights {
    pub geometry: f32,
    pub shadow: f32,
}

impl Weights {
    /// The only SSE arithmetic in the recovered producer:
    /// `W_geometry = PF30188 * PF90034`, `W_shadow = PF90034 * PF30189`.
    /// The multiply is deliberately raw: no clamp, scale, or normalization.
    pub fn from_degrees(pf90034: f32, pf30188: f32, pf30189: f32) -> Self {
        Self {
            geometry: pf30188 * pf90034,
            shadow: pf90034 * pf30189,
        }
    }

    /// `W_shadow >= 1e-4 || W_geometry >= 1e-4` continues (`0x7E47335`).
    ///
    /// `ucomisd` reports unordered for NaN, so a NaN product also passes the
    /// machine gate. That is the recovered instruction semantics, not a
    /// product contract; callers that need finite values should validate
    /// first (as `per_face_weights` and `double_chin_weights` do).
    pub fn is_active(self) -> bool {
        let geometry = f64::from(self.geometry);
        let shadow = f64::from(self.shadow);
        geometry >= WEIGHT_GATE
            || shadow >= WEIGHT_GATE
            || geometry.is_nan()
            || shadow.is_nan()
    }
}

/// Vendor `std::map<face_index, node>` lookup for one face. A PF node that is
/// structurally absent is the recovered `0.0f` miss, not an error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceDegrees {
    pub face_id: u32,
    pub pf90034: Option<f32>,
    pub pf30188: Option<f32>,
    pub pf30189: Option<f32>,
}

impl FaceDegrees {
    pub fn weights(self) -> Weights {
        Weights::from_degrees(
            self.pf90034.unwrap_or(0.0),
            self.pf30188.unwrap_or(0.0),
            self.pf30189.unwrap_or(0.0),
        )
    }
}

/// Evaluate the recovered producer once per face. Each face keeps its own
/// binary32 pair: no broadcasting, cross-face accumulation, or normalization.
///
/// An empty packet is a missing same-job input. Duplicate face ids are a
/// malformed vendor map. Non-finite products are refused before any caller can
/// rely on the NaN-passes-gate machine quirk.
pub fn per_face_weights(faces: &[FaceDegrees]) -> Result<Vec<(u32, Weights)>, Refusal> {
    if faces.is_empty() {
        return Err(Refusal::MissingInput("PF90034 per-face packet"));
    }
    let mut seen = Vec::with_capacity(faces.len());
    let mut out = Vec::with_capacity(faces.len());
    for face in faces {
        if seen.contains(&face.face_id) {
            return Err(Refusal::ShapeMismatch("PF90034 duplicate face id"));
        }
        seen.push(face.face_id);
        let weights = face.weights();
        if !weights.geometry.is_finite() || !weights.shadow.is_finite() {
            return Err(Refusal::ValueOutOfRange("PF90034 per-face weights"));
        }
        out.push((face.face_id, weights));
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `(fx, fy)` are offsets: `map = p + field * scale`.
    Flow,
    /// `(fx, fy)` are destination coordinates: `map = field * scale`.
    Absolute,
}

fn reflect101(index: i64, len: i64) -> i64 {
    if len <= 1 {
        return 0;
    }
    let period = 2 * len - 2;
    let mut value = index % period;
    if value < 0 {
        value += period;
    }
    if value >= len {
        period - value
    } else {
        value
    }
}

fn sample(image: &Image, x: f32, y: f32) -> [u8; 3] {
    let width = image.width as i64;
    let x0f = x.floor();
    let y0f = y.floor();
    let tx = x - x0f;
    let ty = y - y0f;
    let x0 = reflect101(x0f as i64, width);
    let x1 = reflect101(x0f as i64 + 1, width);
    let y0 = reflect101(y0f as i64, image.height as i64);
    let y1 = reflect101(y0f as i64 + 1, image.height as i64);
    let at = |px: i64, py: i64, channel: usize| {
        image.pixels[((py * width + px) * 4) as usize + channel] as f32
    };
    let mut out = [0u8; 3];
    for (channel, value) in out.iter_mut().enumerate() {
        let top = at(x0, y0, channel) + (at(x1, y0, channel) - at(x0, y0, channel)) * tx;
        let bottom = at(x0, y1, channel) + (at(x1, y1, channel) - at(x0, y1, channel)) * tx;
        *value = (top + (bottom - top) * ty).round().clamp(0.0, 255.0) as u8;
    }
    out
}

pub fn warp(
    source: &Image,
    field: &[[f32; 2]],
    mode: Mode,
    scale: f32,
) -> Result<Image, Refusal> {
    source.validate()?;
    if field.len() != source.pixel_count() {
        return Err(Refusal::ShapeMismatch("shift field"));
    }
    if !scale.is_finite() {
        return Err(Refusal::ValueOutOfRange("shift scale"));
    }
    if field.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("shift field"));
    }
    let mut output = source.clone();
    for y in 0..source.height {
        for x in 0..source.width {
            let index = (y * source.width + x) as usize;
            let point = field[index];
            let (sx, sy) = match mode {
                Mode::Flow => (
                    x as f32 + point[0] * scale,
                    y as f32 + point[1] * scale,
                ),
                Mode::Absolute => (point[0] * scale, point[1] * scale),
            };
            let rgb = sample(source, sx, sy);
            let at = index * 4;
            output.pixels[at..at + 3].copy_from_slice(&rgb);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_flow_is_identity() {
        let source = Image::new(3, 2, (0..24).map(|i| (i * 7) as u8).collect()).unwrap();
        let out = warp(&source, &vec![[0.0, 0.0]; 6], Mode::Flow, 1.0).unwrap();
        assert_eq!(out, source);
    }

    #[test]
    fn absolute_identity_map_returns_the_source() {
        let source = Image::new(2, 2, (0..16).map(|i| (i * 5) as u8).collect()).unwrap();
        let field = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        assert_eq!(warp(&source, &field, Mode::Absolute, 1.0).unwrap(), source);
    }

    #[test]
    fn flow_shift_moves_pixels_and_reflects_edges() {
        let source =
            Image::new(3, 1, vec![10, 0, 0, 1, 20, 0, 0, 1, 30, 0, 0, 1]).unwrap();
        let out = warp(&source, &vec![[1.0, 0.0]; 3], Mode::Flow, 1.0).unwrap();
        assert_eq!(out.pixel(0, 0)[0], 20);
        // x=2 maps to 3, which BORDER_REFLECT_101 folds back to index 1.
        assert_eq!(out.pixel(2, 0)[0], 20);
    }

    #[test]
    fn malformed_fields_are_refused() {
        let source = Image::solid(2, 2, [0; 4]);
        assert_eq!(
            warp(&source, &[[0.0, 0.0]; 3], Mode::Flow, 1.0),
            Err(Refusal::ShapeMismatch("shift field"))
        );
        assert_eq!(
            warp(&source, &[[f32::NAN, 0.0]; 4], Mode::Flow, 1.0),
            Err(Refusal::ValueOutOfRange("shift field"))
        );
        assert_eq!(
            warp(&source, &[[0.0, 0.0]; 4], Mode::Flow, f32::INFINITY),
            Err(Refusal::ValueOutOfRange("shift scale"))
        );
    }

    #[test]
    fn weights_are_raw_binary32_products_without_scale_or_clamp() {
        // 200 * 3 would become 6 (or 0.06) under any /100 or 0..1 clamp.
        let large = Weights::from_degrees(200.0, 3.0, 4.0);
        assert_eq!(large.geometry, 600.0);
        assert_eq!(large.shadow, 800.0);
        // Negative products are not clamped to zero either.
        let negative = Weights::from_degrees(-1.0, 2.0, 3.0);
        assert_eq!(negative.geometry, -2.0);
        assert_eq!(negative.shadow, -3.0);
        // The recovered operands are `PF30188 * PF90034` and `PF90034 * PF30189`.
        assert_eq!(Weights::from_degrees(0.5, 0.4, 0.2), Weights {
            geometry: 0.4 * 0.5,
            shadow: 0.5 * 0.2,
        });
    }

    #[test]
    fn gate_uses_the_binary64_1e_4_threshold() {
        // Clearly above / below the recovered constant.
        assert!(Weights { geometry: 2e-4, shadow: 0.0 }.is_active());
        assert!(Weights { geometry: 0.0, shadow: 2e-4 }.is_active());
        assert!(!Weights { geometry: 5e-5, shadow: 5e-5 }.is_active());
        assert!(!Weights { geometry: 0.0, shadow: 0.0 }.is_active());
        // Negative degrees cannot pass the `>= 1e-4` comparison.
        assert!(!Weights { geometry: -1.0, shadow: -2.0 }.is_active());
    }

    #[test]
    fn per_face_weights_do_not_mix_or_accumulate_faces() {
        let faces = [
            FaceDegrees {
                face_id: 1,
                pf90034: Some(0.5),
                pf30188: Some(0.4),
                pf30189: Some(0.2),
            },
            FaceDegrees {
                face_id: 2,
                pf90034: Some(2.0),
                pf30188: Some(3.0),
                pf30189: Some(-1.0),
            },
        ];
        let out = per_face_weights(&faces).unwrap();
        assert_eq!(out[0], (1, Weights { geometry: 0.2, shadow: 0.1 }));
        assert_eq!(out[1], (2, Weights { geometry: 6.0, shadow: -2.0 }));
    }

    #[test]
    fn missing_face_nodes_follow_the_vendor_zero_miss() {
        let face = FaceDegrees {
            face_id: 7,
            pf90034: None,
            pf30188: Some(0.4),
            pf30189: Some(0.2),
        };
        assert_eq!(face.weights(), Weights { geometry: 0.0, shadow: 0.0 });
        assert!(!face.weights().is_active());
    }

    #[test]
    fn empty_packets_and_malformed_maps_are_refused_by_name() {
        assert_eq!(
            per_face_weights(&[]),
            Err(Refusal::MissingInput("PF90034 per-face packet"))
        );
        let duplicate = [
            FaceDegrees { face_id: 3, pf90034: Some(1.0), pf30188: Some(1.0), pf30189: Some(1.0) },
            FaceDegrees { face_id: 3, pf90034: Some(1.0), pf30188: Some(1.0), pf30189: Some(1.0) },
        ];
        assert_eq!(
            per_face_weights(&duplicate),
            Err(Refusal::ShapeMismatch("PF90034 duplicate face id"))
        );
        let nan = [FaceDegrees {
            face_id: 4,
            pf90034: Some(f32::NAN),
            pf30188: Some(1.0),
            pf30189: Some(1.0),
        }];
        assert_eq!(
            per_face_weights(&nan),
            Err(Refusal::ValueOutOfRange("PF90034 per-face weights"))
        );
    }

    #[test]
    fn nan_passes_the_recovered_machine_gate() {
        // `ucomisd` treats unordered as passing; this documents the recovered
        // instruction semantics even though callers refuse NaN upstream.
        assert!(Weights::from_degrees(f32::NAN, 1.0, 1.0).is_active());
    }
}
