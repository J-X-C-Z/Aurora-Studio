//! Destination stages of the fixed C1 ICC float pipeline.

use super::icc_grid::{quantize_output, saturate_word};
use super::icc_source::{be_word, curve_word, icc_tag, normalized};

const PHOTO_RGB: [[f64; 3]; 3] = [
    [
        2.6918170275718314,
        -0.5112104015665965,
        -0.10216668384224432,
    ],
    [-1.0891781885296437, 3.016314465315685, 0.04102744925657062],
    [0.0, 0.0, 2.4244834538761766],
];
const BT2020: [[f64; 3]; 3] = [
    [
        3.2944091933831934,
        -0.7871861344459952,
        -0.47192340362138924,
    ],
    [
        -1.3651992002664817,
        3.2951852261821286,
        0.025630937251071094,
    ],
    [
        0.05938122622808974,
        -0.12590501337602014,
        2.5077496253694176,
    ],
];
const INVERSE_LEN: usize = 4096;

pub(crate) fn photo_inverse_table(profile: &[u8]) -> Option<Vec<u16>> {
    if profile.get(16..24)? != b"RGB XYZ " {
        return None;
    }
    let red = icc_tag(profile, b"rTRC")?;
    if red.len() < 12 + 2 * 513
        || &red[..4] != b"curv"
        || red[8..12] != 513u32.to_be_bytes()
        || icc_tag(profile, b"gTRC")? != red
        || icc_tag(profile, b"bTRC")? != red
    {
        return None;
    }
    let forward: Vec<_> = red[12..12+2*513].chunks_exact(2).map(be_word).collect();
    if forward[0] != 0 || forward[512] != u16::MAX || forward.windows(2).any(|pair| pair[0] > pair[1]) {
        return None;
    }
    let mut interval = 0;
    Some((0..INVERSE_LEN).map(|index| {
        let code = index as f64 * 65535.0 / (INVERSE_LEN - 1) as f64;
        while interval < 511 && f64::from(forward[interval+1]) <= code {
            interval += 1;
        }
        let left = f64::from(forward[interval]);
        let right = f64::from(forward[interval+1]);
        let low = interval as f64 * 65535.0 / 512.0;
        let high = (interval+1) as f64 * 65535.0 / 512.0;
        if left == right {
            saturate_word(high)
        } else {
            let slope = (high - low) / (right - left);
            saturate_word(slope * code + (high - slope * right))
        }
    }).collect())
}

/// Input is normalized ICC PCS XYZ as emitted by the float Lab-to-XYZ stage.
/// Unit Y in physical XYZ corresponds to 32768/65535 here.
/// Returns float RGB before the optimizer's u16 saturation and quantization.
pub(crate) fn pcs_xyz_to_encoded(photo_inverse: Option<&[u16]>, xyz: [f32; 3]) -> [f32; 3] {
    match photo_inverse {
        Some(inverse) => linear_rgb(&PHOTO_RGB, xyz).map(|value| photo_encode(value, inverse)),
        None => linear_rgb(&BT2020, xyz).map(bt2020_inverse),
    }
}

fn linear_rgb(matrix: &[[f64; 3]; 3], xyz: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = xyz.map(f64::from);
    matrix.map(|[a, b, c]| (0.0 + a * x + b * y + c * z) as f32)
}

fn photo_encode(value: f32, inverse: &[u16]) -> f32 {
    normalized(curve_word(quantize_output(value), inverse))
}
fn bt2020_inverse(value: f32) -> f32 {
    let x = f64::from(value);
    let toe_slope = 0.03125;
    let toe_end = 0.0079345703125;
    let scale = 0.9921112060546875;
    let offset = 0.00787353515625;
    let breakpoint = scale * toe_end + offset;
    if x < breakpoint * breakpoint {
        (x / toe_slope) as f32
    } else {
        ((x.sqrt() - offset) / scale) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_dispatch_and_range() {
        assert!(photo_inverse_table(&[]).is_none());
        assert_eq!(pcs_xyz_to_encoded(None, [0.0; 3]), [0.0; 3]);
        assert_eq!(photo_encode(-0.1, &[0, 65535]), 0.0);
        assert_eq!(photo_encode(1.1, &[0, 65535]), 1.0);
        assert_eq!(bt2020_inverse(-0.1), -3.2);
        assert!(bt2020_inverse(1.0) > 1.0);
    }

}
