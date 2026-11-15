//! PF90034 double-chin warp: the AI state stores `std::vector<cv::Mat>` where
//! each Mat is a dense 2-channel float field (`0x4AB20F0` → `0x34B9B10` →
//! `0x4E69B90`, `cv::remap` with INTER_LINEAR + BORDER_REFLECT_101).
//!
//! Whether the stored channels are `(dx, dy)` or absolute `(x', y')`, and the
//! field scale, are not proven, so both are explicit caller inputs.

use crate::{Access, EffectMeta, Image, Refusal};

pub const META: &[EffectMeta] = &[EffectMeta {
    pf: 90034,
    key: "",
    label: "祛双下巴",
    access: Access::Cpu,
    inputs: &["dense 2-channel shift field", "mode (Flow | Absolute)", "scale"],
}];

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
}
