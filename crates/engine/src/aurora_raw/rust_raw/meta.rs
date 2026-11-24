//! Metadata shared by the native decoders: CFA layout and the colour pipeline
//! LibRaw exposes as `rgb_cam`.
//!
//! `cam_xyz_coeff` below is a direct reimplementation of LibRaw's routine (same
//! operation order, double precision, `f32` only at the end) so the resulting
//! matrix is bit-identical when the camera matrix is known.

use super::{NativeError, NativeResult};

/// Builds LibRaw's `filters` word from a CFA pattern.
///
/// The pattern is repeated to sixteen 2-bit entries, least-significant first, so
/// that [`crate::raw::BayerFrame::color_at`] can index it.
pub(crate) fn filters_from_cfa(
    repeat: Option<(u16, u16)>,
    pattern: &[u8],
    plane_colors: Option<&[u8]>,
) -> NativeResult<(u32, u32)> {
    if repeat == Some((6, 6)) || pattern.len() >= 36 {
        // X-Trans: LibRaw uses the sentinel value 9 instead of a CFA word.
        return Ok((9, 3));
    }
    if pattern.is_empty() || pattern.len() > 16 {
        return Err(NativeError::corrupt("CFA pattern has an unsupported length"));
    }
    let mut unique = 0u32;
    for &entry in pattern {
        if entry > 31 {
            continue;
        }
        unique |= 1 << entry;
    }
    let colors = unique.count_ones().min(4).max(1);
    let default_planes = [0u8, 1, 2, 3];
    let mut planes = [0u8; 4];
    match plane_colors {
        Some(given) if given.len() >= colors as usize => {
            planes[..colors as usize].copy_from_slice(&given[..colors as usize]);
        }
        _ => {
            planes.copy_from_slice(&default_planes);
            // LibRaw's legacy CMY/GMCY plane maps.
            if unique == 0o70 {
                planes[..3].copy_from_slice(&[3, 4, 5]);
            } else if unique == 0o72 {
                planes.copy_from_slice(&[5, 3, 4, 1]);
            }
        }
    }
    let mut tab = [0u8; 256];
    for (color, &plane) in planes.iter().enumerate().take(colors as usize) {
        tab[usize::from(plane)] = color as u8;
    }
    let mut filters = 0u32;
    for index in (0..16).rev() {
        let value = pattern[index % pattern.len()];
        if value > 31 {
            return Err(NativeError::corrupt("CFA pattern value is out of range"));
        }
        filters = (filters << 2) | u32::from(tab[usize::from(value)]);
    }
    filters -= u32::from(filters == 0);
    Ok((filters, colors))
}

/// LibRaw upgrades a three-colour CFA word so the second green gets its own
/// code (0x94 -> 0xb4 for RGGB).
pub(crate) fn upgrade_filters(filters: u32, colors: u32) -> u32 {
    if filters > 999 && colors == 3 {
        filters | (((filters >> 2 & 0x2222_2222) | (filters << 2 & 0x8888_8888)) & (filters << 1))
    } else {
        filters
    }
}

/// Unpacks big-endian `bits`-wide samples (TIFF/DNG packed rows).
pub(crate) fn unpack_bits_msb(raw: &[u8], bits: u8, count: usize) -> NativeResult<Vec<u16>> {
    if bits == 0 || bits > 16 {
        return Err(NativeError::unsupported(format!("sample depth {bits}")));
    }
    let mut out = Vec::with_capacity(count);
    if bits == 16 {
        if raw.len() < count * 2 {
            return Err(NativeError::corrupt("raw strip is shorter than its samples"));
        }
        for chunk in raw.chunks_exact(2).take(count) {
            out.push(u16::from_be_bytes([chunk[0], chunk[1]]));
        }
        return Ok(out);
    }
    let total_bits = raw.len() * 8;
    if count.saturating_mul(usize::from(bits)) > total_bits {
        return Err(NativeError::corrupt("raw strip is shorter than its samples"));
    }
    let mut bit = 0usize;
    let mask = (1u32 << bits) - 1;
    let width = usize::from(bits);
    for _ in 0..count {
        let byte = bit / 8;
        let shift = bit % 8;
        // A 16-bit sample starting at bit 7 needs three bytes; use a 24-bit
        // window so every alignment is covered.
        let window = (u32::from(raw.get(byte).copied().unwrap_or(0)) << 16)
            | (u32::from(raw.get(byte + 1).copied().unwrap_or(0)) << 8)
            | u32::from(raw.get(byte + 2).copied().unwrap_or(0));
        let value = (window >> (24 - shift - width)) & mask;
        out.push(value as u16);
        bit += width;
    }
    Ok(out)
}

pub(crate) fn unpack_bits_le16(raw: &[u8], count: usize) -> NativeResult<Vec<u16>> {
    if raw.len() < count * 2 {
        return Err(NativeError::corrupt("raw strip is shorter than its samples"));
    }
    Ok(raw.chunks_exact(2).take(count).map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]])).collect())
}

// LibRaw_constants::xyz_rgb.
const XYZ_RGB: [[f64; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];

fn pseudoinverse(input: &[[f64; 3]; 4], size: usize) -> [[f64; 3]; 4] {
    let mut work = [[0.0f64; 6]; 3];
    for i in 0..3 {
        for j in 0..6 {
            work[i][j] = f64::from(u8::from(j == i + 3));
        }
        for j in 0..3 {
            for k in 0..size.min(4) {
                work[i][j] += input[k][i] * input[k][j];
            }
        }
    }
    for i in 0..3 {
        let num = work[i][i];
        if num.abs() > 0.000_01 {
            for j in 0..6 {
                work[i][j] /= num;
            }
        }
        for k in 0..3 {
            if k == i {
                continue;
            }
            let num = work[k][i];
            for j in 0..6 {
                work[k][j] -= work[i][j] * num;
            }
        }
    }
    let mut out = [[0.0f64; 3]; 4];
    for i in 0..size.min(4) {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += work[j][k + 3] * input[i][k];
            }
        }
    }
    out
}

/// LibRaw's `cam_xyz_coeff`: camera XYZ matrix to `rgb_cam`.
pub(crate) fn cam_xyz_to_rgb_cam(cam_xyz: &[[f64; 3]; 4], colors: usize) -> [[f32; 4]; 3] {
    let colors = colors.clamp(1, 4);
    let mut cam_rgb = [[0.0f64; 3]; 4];
    for i in 0..colors {
        for j in 0..3 {
            for k in 0..3 {
                cam_rgb[i][j] += cam_xyz[i][k] * XYZ_RGB[k][j];
            }
        }
        let sum: f64 = cam_rgb[i].iter().sum();
        if sum > 0.000_01 {
            for value in &mut cam_rgb[i] {
                *value /= sum;
            }
        } else {
            cam_rgb[i] = [0.0; 3];
        }
    }
    let inverse = pseudoinverse(&cam_rgb, colors);
    let mut rgb_cam = [[0.0f32; 4]; 3];
    for i in 0..3 {
        for j in 0..colors {
            rgb_cam[i][j] = inverse[j][i] as f32;
        }
    }
    rgb_cam
}

/// Camera matrix as `cam_xyz = trans / 10000.0`, mirroring LibRaw's Adobe table.
///
/// Only the Nikon entries the native path can currently validate are listed;
/// unknown models return `None` and the caller leaves `rgb_cam` zeroed rather
/// than guessing.
pub(crate) fn adobe_matrix(make: &str, model: &str) -> Option<[[f64; 3]; 4]> {
    let make = make.trim().to_ascii_uppercase();
    let model = normalize_model(make.as_str(), model);
    const NIKON: &[(&str, [i32; 9])] = &[
        ("Z 50", [11640, -4829, -1079, -5107, 13006, 2325, -972, 1711, 7380]),
        ("Z5_2", [9874, -3784, -823, -4728, 12673, 2286, -648, 1513, 6375]),
        ("Z 5", [8695, -2558, -648, -5015, 12711, 2575, -1279, 2215, 7514]),
        ("Z6_3", [11206, -4286, -941, -4879, 12847, 2251, -745, 1654, 7374]),
        ("Z 6", [9943, -3269, -839, -5323, 13269, 2259, -1198, 2083, 7557]),
        ("Z 7", [13705, -6004, -1400, -5464, 13568, 2062, -940, 1706, 7618]),
        ("Z 8", [11423, -4564, -1123, -4816, 12895, 2119, -210, 1061, 7282]),
        ("Z 9", [13389, -6049, -1441, -4544, 12757, 1969, 229, 498, 7390]),
        ("Z fc", [11640, -4829, -1079, -5107, 13006, 2325, -972, 1711, 7380]),
        ("ZR", [11206, -4286, -941, -4879, 12847, 2251, -745, 1654, 7374]),
    ];
    // LibRaw's table is scanned in order with prefix matching, so longer model
    // names must come first (ILCE-7RM5 before ILCE-7, ILCE-9M3 before ILCE-9).
    const SONY: &[(&str, [i32; 9])] = &[
        ("ILCE-7RM6", [11765, -5595, -1192, -3689, 11507, 2485, 51, 681, 5731]),
        ("ILCE-7RM5", [8200, -2976, -719, -4296, 12053, 2532, -429, 1282, 5774]),
        ("ILCE-7RM4", [7662, -2686, -660, -5240, 12965, 2530, -796, 1508, 6167]),
        ("ILCE-7RM3", [6640, -1847, -503, -5238, 13010, 2474, -993, 1673, 6527]),
        ("ILCE-7RM2", [6629, -1900, -483, -4618, 12349, 2550, -622, 1381, 6514]),
        ("ILCE-7SM3", [6912, -2127, -469, -4470, 12175, 2587, -398, 1478, 6492]),
        ("ILCE-7S", [5838, -1430, -246, -3497, 11477, 2297, -748, 1885, 5778]),
        ("ILCE-7CM2", [7460, -2365, -588, -5687, 13442, 2474, -624, 1156, 6584]),
        ("ILCE-7CR", [8200, -2976, -719, -4296, 12053, 2532, -429, 1282, 5774]),
        ("ILCE-7C", [7374, -2389, -551, -5435, 13162, 2519, -1006, 1795, 6552]),
        ("ILCE-7M5", [9089, -3577, -787, -3563, 11326, 2557, -114, 928, 5904]),
        ("ILCE-7M4", [7460, -2365, -588, -5687, 13442, 2474, -624, 1156, 6584]),
        ("ILCE-7M3", [7374, -2389, -551, -5435, 13162, 2519, -1006, 1795, 6552]),
        ("ILCE-7", [5271, -712, -347, -6153, 13653, 2763, -1601, 2366, 7242]),
        ("ILCE-9M3", [9811, -3908, -752, -3704, 11577, 2417, -73, 950, 5980]),
        ("ILCE-9", [6389, -1703, -378, -4562, 12265, 2587, -670, 1489, 6550]),
        ("ILCE-1", [8161, -2947, -739, -4811, 12668, 2389, -437, 1229, 6524]),
    ];
    const CANON: &[(&str, [i32; 9])] = &[
        ("EOS R5 Mark II", [9396, -2598, -1207, -4408, 12296, 2369, -505, 1575, 6077]),
        ("EOS R5", [9766, -2953, -1254, -4276, 12116, 2433, -437, 1336, 5131]),
        ("EOS R6 Mark III", [8309, -1786, -1095, -4851, 12736, 2353, -956, 1911, 5534]),
        ("EOS R6 Mark II", [9539, -2795, -1224, -4175, 11998, 2458, -465, 1755, 6048]),
        ("EOS R6", [8293, -1611, -1132, -4759, 12711, 2275, -1013, 2415, 5509]),
        ("EOS R7", [10424, -3138, -1300, -4221, 11938, 2584, -547, 1658, 6183]),
        ("EOS R8", [9539, -2795, -1224, -4175, 11998, 2458, -465, 1755, 6048]),
        ("EOS R10", [9269, -2012, -1107, -3990, 11762, 2527, -569, 2093, 4913]),
        ("EOS R50 V", [9089, -1810, -1209, -4294, 12116, 2454, -498, 1862, 4923]),
        ("EOS R50", [9269, -2012, -1107, -3990, 11762, 2527, -569, 2093, 4913]),
        ("EOS R100", [8230, -1515, -1032, -4179, 12005, 2454, -649, 2076, 4711]),
        ("EOS R3", [11004, -4202, -1177, -3774, 11646, 2429, -433, 1229, 6184]),
        ("EOS R1", [9020, -2275, -1209, -4916, 12856, 2284, -923, 1953, 5928]),
        ("EOS Ra", [22880, -11531, -2223, -2034, 10469, 1809, 316, 1401, 5769]),
        ("EOS R", [8293, -1789, -1094, -5025, 12925, 2327, -1199, 2769, 6108]),
    ];
    let table = if make.contains("NIKON") {
        NIKON
    } else if make.contains("SONY") {
        SONY
    } else if make.contains("CANON") {
        CANON
    } else {
        return None;
    };
    for (prefix, matrix) in table {
        if model.len() >= prefix.len() && model[..prefix.len()].eq_ignore_ascii_case(prefix) {
            // LibRaw stores `trans[j] / 10000.f` as `float`, then widens to double.
            let mut cam = [[0.0f64; 3]; 4];
            for index in 0..9 {
                cam[index / 3][index % 3] = f64::from(matrix[index] as f32 / 10000.0f32);
            }
            return Some(cam);
        }
    }
    None
}

fn normalize_model(make: &str, model: &str) -> String {
    let upper_model = model.trim().to_ascii_uppercase();
    let upper_make = make.trim().to_ascii_uppercase();
    let mut rest = upper_model.as_str();
    for prefix in [upper_make.as_str(), "NIKON CORPORATION", "NIKON", "SONY", "CANON"] {
        if let Some(stripped) = rest.strip_prefix(prefix) {
            rest = stripped;
            break;
        }
    }
    rest.trim_start_matches([' ', '_', '-']).trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rggb_pattern_matches_libraw_word() {
        let (filters, colors) = filters_from_cfa(Some((2, 2)), &[0, 1, 1, 2], None).unwrap();
        assert_eq!(filters, 0x9494_9494);
        assert_eq!(colors, 3);
    }

    #[test]
    fn bggr_pattern_matches_libraw_word() {
        let (filters, _) = filters_from_cfa(Some((2, 2)), &[2, 1, 1, 0], None).unwrap();
        assert_eq!(filters, 0x1616_1616);
    }

    #[test]
    fn unpacks_twelve_bit_rows() {
        // 0x123, 0x456, 0x789 packed big-endian into three bytes pairs.
        let raw = [0x12, 0x34, 0x56, 0x78, 0x90];
        let unpacked = unpack_bits_msb(&raw, 12, 3).unwrap();
        assert_eq!(unpacked, [0x123, 0x456, 0x789]);
    }

    #[test]
    fn z6_three_matrix_is_the_adobe_table_entry() {
        let cam = adobe_matrix("NIKON CORPORATION", "NIKON Z6_3").unwrap();
        assert_eq!(cam[0][0], f64::from(11206f32 / 10000.0f32));
        assert_eq!(cam[2][2], f64::from(7374f32 / 10000.0f32));
        let rgb = cam_xyz_to_rgb_cam(&cam, 3);
        // The identity check is only a sanity bound; the exact values are
        // compared against LibRaw by the ignored real-sample test.
        assert!((rgb[0][0] + rgb[0][1] + rgb[0][2] - 1.0).abs() < 0.2);
    }
}
