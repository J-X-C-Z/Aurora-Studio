//! Encoded working RGB -> encoded output RGB (Adobe RGB 1998 and other matrix/TRC
//! profiles). Relative colorimetric, no white-point adaptation: the fixed DLL's
//! two-profile transform keeps neutral ramps neutral and matches this order.

use super::icc_grid::{quantize_output, saturate_word};
use super::icc_source::{be_word, curve_word, icc_tag};

const INVERSE_LEN: usize = 4096;

enum Decode {
    Gamma(f64),
    Table(Vec<u16>),
    Parametric3([f64; 5]),
}

impl Decode {
    fn parse(profile: &[u8], signature: &[u8; 4]) -> Option<Self> {
        let tag = icc_tag(profile, signature)?;
        if tag.get(..4)? == b"para" && tag.get(8..12)? == [0, 3, 0, 0] {
            let mut parameters = [0.0; 5];
            for (index, value) in parameters.iter_mut().enumerate() {
                let start = 12 + index * 4;
                *value = f64::from(i32::from_be_bytes(tag.get(start..start + 4)?.try_into().ok()?)) / 65536.0;
            }
            let [gamma, scale, offset, slope, threshold] = parameters;
            return (gamma > 0.0 && scale > 0.0 && slope > 0.0
                && (0.0..=1.0).contains(&threshold) && scale * threshold + offset >= 0.0)
                .then_some(Self::Parametric3(parameters));
        }
        if tag.get(..4)? != b"curv" {
            return None;
        }
        let count = u32::from_be_bytes(tag.get(8..12)?.try_into().ok()?) as usize;
        match count {
            0 => Some(Self::Gamma(1.0)),
            1 => Some(Self::Gamma(f64::from(u16::from_be_bytes([tag[12], tag[13]])) / 256.0)),
            _ => Some(Self::Table(
                tag.get(12..12 + 2 * count)?.chunks_exact(2).map(be_word).collect(),
            )),
        }
    }

    fn eval(&self, value: f64) -> f64 {
        let value = value.clamp(0.0, 1.0);
        match self {
            Self::Gamma(gamma) => value.powf(*gamma),
            Self::Parametric3([gamma, scale, offset, slope, threshold]) => {
                if value < *threshold { slope * value } else { (scale * value + offset).powf(*gamma) }
            }
            Self::Table(table) => {
                let span = table.len() - 1;
                let position = value * span as f64;
                let base = (position as usize).min(span - 1);
                let weight = position - base as f64;
                let low = f64::from(table[base]) / 65535.0;
                let high = f64::from(table[base + 1]) / 65535.0;
                low * (1.0 - weight) + high * weight
            }
        }
    }
}

enum Encode {
    Gamma(f64),
    Inverse(Vec<u16>),
}

impl Encode {
    fn build(curve: Decode) -> Option<Self> {
        Some(match curve {
            Decode::Gamma(gamma) => Self::Gamma(gamma),
            Decode::Parametric3(_) => return None,
            Decode::Table(forward) => {
                let span = (forward.len() - 1) as f64;
                Self::Inverse(
                    (0..INVERSE_LEN)
                        .map(|index| {
                            let code = index as f64 * 65535.0 / (INVERSE_LEN - 1) as f64;
                            let mut interval = 0;
                            while interval < forward.len() - 2
                                && f64::from(forward[interval + 1]) <= code
                            {
                                interval += 1;
                            }
                            let left = f64::from(forward[interval]);
                            let right = f64::from(forward[interval + 1]);
                            let low = interval as f64 * 65535.0 / span;
                            let high = (interval + 1) as f64 * 65535.0 / span;
                            if left == right {
                                saturate_word(high)
                            } else {
                                let slope = (high - low) / (right - left);
                                saturate_word(slope * code + (high - slope * right))
                            }
                        })
                        .collect(),
                )
            }
        })
    }

    fn apply(&self, linear: f64) -> u16 {
        if linear.is_nan() {
            return 0;
        }
        let linear = linear.clamp(0.0, 1.0);
        match self {
            Self::Gamma(gamma) => saturate_word(linear.powf(1.0 / gamma) * 65535.0),
            Self::Inverse(table) => curve_word(quantize_output(linear as f32), table),
        }
    }

    // Preserve float precision until the combined grid's final u16 quantization.
    fn encode_float(&self, linear: f32) -> f32 {
        match self {
            Self::Gamma(gamma) => f64::from(linear.clamp(0.0, 1.0)).powf(1.0 / gamma) as f32,
            Self::Inverse(_) => normalized(self.apply(f64::from(linear))),
        }
    }
}

fn colorants(profile: &[u8]) -> Option<[[f64; 3]; 3]> {
    let column = |signature: &[u8; 4]| -> Option<[f64; 3]> {
        let tag = icc_tag(profile, signature)?;
        let value = |index: usize| -> Option<f64> {
            let word = tag.get(8 + 4 * index..12 + 4 * index)?.try_into().ok()?;
            Some(f64::from(i32::from_be_bytes(word)) / 65536.0)
        };
        Some([value(0)?, value(1)?, value(2)?])
    };
    let (r, g, b) = (column(b"rXYZ")?, column(b"gXYZ")?, column(b"bXYZ")?);
    Some([[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]])
}

fn invert(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let [[a, b, c], [d, e, f], [g, h, i]] = matrix;
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    (determinant.abs() > 1e-12).then(|| {
        [
            [(e * i - f * h) / determinant, (c * h - b * i) / determinant, (b * f - c * e) / determinant],
            [(f * g - d * i) / determinant, (a * i - c * g) / determinant, (c * d - a * f) / determinant],
            [(d * h - e * g) / determinant, (b * g - a * h) / determinant, (a * e - b * d) / determinant],
        ]
    })
}

pub struct OutputStage {
    decode: [Decode; 3],
    to_xyz: [[f64; 3]; 3],
    to_linear: [[f64; 3]; 3],
    encode: [Encode; 3],
}

impl OutputStage {
    /// `input` is the profile whose encoding the chain emitted, `target` the output profile.
    pub fn new(input: &[u8], target: &[u8]) -> Option<Self> {
        let decode = [
            Decode::parse(input, b"rTRC")?,
            Decode::parse(input, b"gTRC")?,
            Decode::parse(input, b"bTRC")?,
        ];
        let encode = [
            Encode::build(Decode::parse(target, b"rTRC")?)?,
            Encode::build(Decode::parse(target, b"gTRC")?)?,
            Encode::build(Decode::parse(target, b"bTRC")?)?,
        ];
        Some(Self {
            decode,
            to_xyz: colorants(input)?,
            to_linear: invert(colorants(target)?)?,
            encode,
        })
    }

    /// Encoded working RGB in 0..1 to quantized output codes.
    pub fn apply(&self, rgb: [f32; 3]) -> Option<[u16; 3]> {
        let encoded = self.encode(rgb)?;
        Some(encoded.map(|value| saturate_word(f64::from(value) * 65535.0)))
    }

    /// Saturated encoded output in 0..1, before 16-bit quantization.
    pub fn encode(&self, rgb: [f32; 3]) -> Option<[f32; 3]> {
        if !rgb.into_iter().all(f32::is_finite) {
            return None;
        }
        let linear: [f64; 3] =
            std::array::from_fn(|channel| self.decode[channel].eval(f64::from(rgb[channel])));
        let xyz = self.to_xyz.map(|row| row[0] * linear[0] + row[1] * linear[1] + row[2] * linear[2]);
        let output = self.to_linear.map(|row| row[0] * xyz[0] + row[1] * xyz[1] + row[2] * xyz[2]);
        Some(std::array::from_fn(|channel| {
            normalized(self.encode[channel].apply(output[channel]))
        }))
    }
}

fn normalized(code: u16) -> f32 {
    (f64::from(code) / 65535.0) as f32
}

/// The fixed DLL's optimized two-profile transform: a 33^3 table sampled from
/// the same math and evaluated with type-1 tetrahedral interpolation.
pub struct OutputClut {
    sizes: [usize; 3],
    nodes: Vec<[u16; 3]>,
}

impl OutputClut {
    pub fn new(input: &[u8], target: &[u8], points: u8) -> Option<Self> {
        if !matches!(points, 17 | 33 | 49) {
            return None;
        }
        let stage = OutputStage::new(input, target)?;
        Self::sample(points, |rgb| stage.encode(rgb).unwrap_or([0.0; 3]))
    }

    pub fn from_camera49(input: &[u8], target: &[u8]) -> Option<Self> {
        let camera = super::icc_source::CameraPcs::from_prostandard(input)?;
        let matrix = invert(colorants(target)?)?.map(|row| row.map(|v| v * (65535.0 / 32768.0)));
        let curve = |tag| Encode::build(Decode::parse(target, tag)?);
        let trcs = [curve(b"rTRC")?, curve(b"gTRC")?, curve(b"bTRC")?];
        Self::sample(49, |rgb| {
            let [x,y,z] = camera.to_xyz(rgb).map(f64::from);
            let linear = matrix.map(|[a,b,c]| (a*x + b*y + c*z) as f32);
            std::array::from_fn(|i| trcs[i].encode_float(linear[i]))
        })
    }

    pub fn from_camera49_srgb(input: &[u8], srgb_target: Option<&[u8]>) -> Option<Self> {
        if let Some(clut) = srgb_target.and_then(|target| Self::from_camera49(input, target)) {
            return Some(clut);
        }
        let camera = super::icc_source::CameraPcs::from_prostandard(input)?;
        Self::sample(49, |rgb| {
            let xyz = camera.to_xyz(rgb).map(|value| value * (65535.0 / 32768.0));
            let linear = crate::icc::xyz_d50_to_linear_srgb(xyz);
            linear.map(|value| crate::mock::process::linear_to_srgb(value.clamp(0.0, 1.0)))
        })
    }

    fn sample(points: u8, evaluate: impl FnMut([f32; 3]) -> [f32; 3]) -> Option<Self> {
        let mut table = vec![0u16; 3 * usize::from(points).pow(3)];
        super::icc_grid::sample_rgb([points; 3], &mut table, evaluate).then_some(())?;
        let nodes = table.chunks_exact(3).map(|node| [node[0], node[1], node[2]]).collect();
        Some(Self { sizes: [usize::from(points); 3], nodes })
    }

    pub fn sample_srgb(&self, rgb: [f32; 3]) -> Option<[f32; 3]> {
        super::camera_clut::CameraClut::new(self.sizes, &self.nodes)?.sample_type1(rgb)
    }

    pub fn apply(&self, rgb: [f32; 3]) -> Option<[u16; 3]> {
        let sampled = super::camera_clut::CameraClut::new(self.sizes, &self.nodes)?.sample_type1(rgb)?;
        Some(sampled.map(|value| saturate_word(f64::from(value) * 65535.0)))
    }

    pub fn apply_float_export(&self, rgb: [f32; 3]) -> Option<[u16; 3]> {
        let sampled = super::camera_clut::CameraClut::new(self.sizes, &self.nodes)?
            .sample_type1(rgb)?;
        Some(sampled.map(|value| (value * 65535.0) as u16))
    }

    pub fn nodes(&self) -> &[[u16; 3]] {
        &self.nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bt2020_type3_decodes_toe_and_power_segments() {
        let curve = Decode::Parametric3([2.0, 65019.0 / 65536.0, 516.0 / 65536.0, 0.03125, 520.0 / 65536.0]);
        assert_eq!(curve.eval(0.0), 0.0);
        assert_eq!(curve.eval(1.0 / 256.0), 1.0 / 8192.0);
        assert!((curve.eval(0.5) - (66051.0_f64 / 131072.0).powi(2)).abs() < 1e-15);
        assert!(Encode::build(curve).is_none());
    }

    fn profile(matrix: [[f64; 3]; 3], gamma: u16) -> Vec<u8> {
        let mut tags: Vec<([u8; 4], Vec<u8>)> = Vec::new();
        for (signature, column) in [(b"rXYZ", 0usize), (b"gXYZ", 1), (b"bXYZ", 2)] {
            let mut payload = b"XYZ \0\0\0\0".to_vec();
            for row in 0..3 {
                payload.extend_from_slice(&((matrix[row][column] * 65536.0).round() as i32).to_be_bytes());
            }
            tags.push((*signature, payload));
        }
        for signature in [b"rTRC", b"gTRC", b"bTRC"] {
            let mut payload = b"curv\0\0\0\0".to_vec();
            payload.extend_from_slice(&1u32.to_be_bytes());
            payload.extend_from_slice(&gamma.to_be_bytes());
            tags.push((*signature, payload));
        }
        let start = 128 + 4 + 12 * tags.len() as u32;
        let mut bytes = vec![0u8; start as usize];
        let mut offset = start;
        for (index, (signature, payload)) in tags.iter().enumerate() {
            let entry = 132 + 12 * index;
            bytes[entry..entry + 4].copy_from_slice(signature);
            bytes[entry + 4..entry + 8].copy_from_slice(&offset.to_be_bytes());
            bytes[entry + 8..entry + 12].copy_from_slice(&(payload.len() as u32).to_be_bytes());
            offset += payload.len() as u32;
            bytes.extend_from_slice(payload);
        }
        bytes[128..132].copy_from_slice(&(tags.len() as u32).to_be_bytes());
        let length = bytes.len() as u32;
        bytes[..4].copy_from_slice(&length.to_be_bytes());
        bytes[36..40].copy_from_slice(b"acsp");
        bytes
    }

    #[test]
    fn identity_profiles_round_trip_and_reject_bad_input() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let stage = OutputStage::new(&profile(identity, 256), &profile(identity, 256)).unwrap();
        assert_eq!(stage.apply([0.0, 0.5, 1.0]), Some([0, 32768, 65535]));
        assert_eq!(stage.apply([f32::NAN, 0.0, 0.0]), None);
        assert!(OutputStage::new(&[], &[]).is_none());
        let clut = OutputClut::new(&profile(identity, 256), &profile(identity, 256), 17).unwrap();
        assert_eq!(clut.apply([0.0, 0.5, 1.0]), Some([0, 32768, 65535]));
        assert!(OutputClut::new(&profile(identity, 256), &profile(identity, 256), 8).is_none());
    }

}
