//! Locally fitted match from our own rendering to Capture One's default look.
//!
//! ## Where the constants come from
//!
//! Fit pair (`tools/analyze2.c`, 640x424, 271 360 samples):
//! * target: `_DSC5673.cop`, the Capture One proxy, decoded with `tools/jxldec.c`
//!   (32-bit float JXL, linear-sRGB ICC, 2000-nit intensity target, SDR white at
//!   0.1015 of full scale -> multiplied by 2000/203 = 9.8522);
//! * source: the camera's own embedded JPEG of the same frame (`_DSC5673.NEF`
//!   preview, 640x424), sRGB-decoded to linear.
//!
//! Both sides are therefore *display-referred linear light*, 1.0 = display white.
//! The constants below are NOT valid on camera-space sensor data: LibRaw 0.21.5
//! cannot decode this NEF (Nikon HE compression, `nikon_he_load_raw_placeholder`),
//! so the camera-space fit could not be produced. Re-run `tools/analyze.c` on a
//! decodable 16-bit linear dump and replace the table when one exists.
//!
//! Pipeline: exposure * white balance -> 3x3 -> luminance-preserving tone curve.
//! Residual after the full stack: DeltaE76 mean 10.80, p95 38.73 (stage means:
//! exposure only 26.55, +WB 22.27, +matrix 15.38, +curve 10.80). The residual is
//! dominated by the 640x424 source (JPEG blocking, in-camera sharpening) and by
//! C1's local contrast, neither of which this global model can represent.
//!
//! ## Tone curve
//!
//! 32 control points, input linear luminance -> output linear luminance, fitted
//! as the median of each equal-count luminance bin. The shape is a shadow lift
//! with a soft shoulder: +1.09 EV at 0.0034, +1.34 EV at 0.0187, +0.80 EV at
//! 0.0585, +0.59 EV at 0.182, flat from 0.5, -0.08 EV at 1.08.

/// One fitted look: exposure, per-channel gains, colour matrix, tone curve.
#[derive(Debug, Clone, Copy)]
pub struct RawMatchCurve {
    /// Global exposure gain applied before the white balance.
    pub exposure: f32,
    /// Per-channel gains (r, g, b) that normalise the source's white point.
    pub white_balance: [f32; 3],
    /// Row-major 3x3 mapping the balanced RGB into the target's RGB.
    pub matrix: [[f32; 3]; 3],
    /// Tone curve control points, `(input, output)` in linear light, sorted.
    pub curve: &'static [(f32, f32)],
    /// Where the white balance gains came from, for the audit trail.
    pub wb_source: &'static str,
}

impl RawMatchCurve {
    /// Constants fitted from `_DSC5673` (see the module comment for the pair).
    pub const FITTED: RawMatchCurve = RawMatchCurve {
        exposure: 1.218596,
        white_balance: [1.0, 1.638580, 0.933690],
        matrix: [
            [0.482366, 0.513783, 0.218835],
            [-0.070536, 1.580662, 0.133321],
            [-0.541957, 1.198148, 0.650658],
        ],
        curve: &[
            (0.00336355, 0.00716208),
            (0.00511054, 0.01034763),
            (0.00611448, 0.01557020),
            (0.00773583, 0.02017170),
            (0.00963073, 0.02372377),
            (0.01867360, 0.04739704),
            (0.04008803, 0.07907377),
            (0.05853176, 0.10203423),
            (0.07292557, 0.12242568),
            (0.08497290, 0.14192876),
            (0.09578695, 0.16520008),
            (0.10781021, 0.18419117),
            (0.12236036, 0.20669879),
            (0.13492396, 0.22069420),
            (0.15271959, 0.23616299),
            (0.16736874, 0.25384867),
            (0.18208073, 0.27473542),
            (0.18880089, 0.29046115),
            (0.19661193, 0.30418023),
            (0.19944923, 0.30817983),
            (0.20451462, 0.31464016),
            (0.21357347, 0.32041711),
            (0.22043630, 0.33084729),
            (0.22621557, 0.34172300),
            (0.23263888, 0.35560009),
            (0.24370617, 0.36171144),
            (0.27372706, 0.36253679),
            (0.34805682, 0.41021043),
            (0.49120271, 0.49641487),
            (0.72524464, 0.72048664),
            (1.08259094, 1.02115607),
            (1.79946172, 1.42037868),
        ],
        // Least-squares per-channel gains between the two renders, normalised to
        // the red channel. The EXIF as-shot multipliers (1.625/1.000/1.412) are
        // NOT used: they describe sensor data, which this fit never sees.
        wb_source: "per-channel least squares (camera JPEG -> C1), normalised to red",
    };

    /// Sample the tone curve: log-log piecewise linear, clamped at both ends.
    pub fn tone(&self, x: f32) -> f32 {
        let p = &self.curve;
        if !(x > 0.0) {
            return p[0].1;
        }
        if x <= p[0].0 {
            return p[0].1 * x / p[0].0;
        }
        for w in p.windows(2) {
            let ((ax, ay), (bx, by)) = (w[0], w[1]);
            if x <= bx {
                if ax <= 0.0 {
                    return ay + (by - ay) * (x - ax) / (bx - ax);
                }
                let (la, lb, lx) = (ax.ln(), bx.ln(), x.ln());
                return (ay.ln() + (by.ln() - ay.ln()) * (lx - la) / (lb - la)).exp();
            }
        }
        p[p.len() - 1].1
    }
}

/// Applies the look in place to 8-bit RGBA, treating RGB as **linear light**
/// (0..255 == 0.0..1.0), the same convention as `raw::decode`. Alpha is copied.
pub fn apply(rgba: &mut [u8], width: u32, height: u32, curve: &RawMatchCurve) {
    let pixels = width as usize * height as usize;
    for px in rgba.chunks_exact_mut(4).take(pixels) {
        let mut c = [0.0f32; 3];
        for k in 0..3 {
            c[k] = px[k] as f32 / 255.0 * curve.exposure * curve.white_balance[k];
        }
        let m = &curve.matrix;
        let rgb = [
            m[0][0] * c[0] + m[0][1] * c[1] + m[0][2] * c[2],
            m[1][0] * c[0] + m[1][1] * c[1] + m[1][2] * c[2],
            m[2][0] * c[0] + m[2][1] * c[1] + m[2][2] * c[2],
        ];
        let y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        let gain = if y > 1e-6 { (curve.tone(y) / y).min(16.0) } else { 0.0 };
        for k in 0..3 {
            let v = rgb[k] * gain * 255.0;
            px[k] = if v.is_finite() { (v.clamp(0.0, 255.0) + 0.5) as u8 } else { 0 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> RawMatchCurve {
        RawMatchCurve {
            exposure: 1.0,
            white_balance: [1.0; 3],
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            curve: &[(0.0, 0.0), (1.0, 1.0)],
            wb_source: "identity",
        }
    }

    #[test]
    fn identity_is_a_byte_exact_no_op() {
        let mut px: Vec<u8> = (0..=255u8).flat_map(|v| [v, 255 - v, v / 2, 255]).collect();
        let before = px.clone();
        apply(&mut px, 256, 1, &identity());
        assert_eq!(px, before);
    }

    #[test]
    fn known_input_gives_known_output() {
        let mut curve = identity();
        curve.curve = &[(0.0, 0.0), (1.0, 0.25)];
        let mut px = vec![255, 255, 255, 255, 0, 0, 0, 255];
        apply(&mut px, 2, 1, &curve);
        assert_eq!(&px[..4], &[64, 64, 64, 255], "white maps to 0.25 linear");
        assert_eq!(&px[4..], &[0, 0, 0, 255], "black stays black");
    }

    #[test]
    fn extremes_stay_finite() {
        let mut black: Vec<u8> = (0..16).flat_map(|_| [0, 0, 0, 255]).collect();
        let mut white: Vec<u8> = (0..16).flat_map(|_| [255, 255, 255, 128]).collect();
        apply(&mut black, 4, 4, &RawMatchCurve::FITTED);
        apply(&mut white, 4, 4, &RawMatchCurve::FITTED);
        assert!(black.iter().step_by(4).all(|&v| v <= 2));
        assert!(white.iter().step_by(4).all(|&v| v >= 200));
        assert!(black.chunks(4).all(|p| p[3] == 255));
        assert!(white.chunks(4).all(|p| p[3] == 128), "alpha is copied verbatim");
    }

    #[test]
    fn fitted_curve_is_monotonic() {
        let curve = RawMatchCurve::FITTED;
        let mut prev = -1.0f32;
        for v in 0..=255u8 {
            let mut px = [v, v, v, 255];
            apply(&mut px, 1, 1, &curve);
            let y = px[1] as f32;
            assert!(y >= prev, "grey {v} renders darker than the one below");
            prev = y;
        }
    }
}
