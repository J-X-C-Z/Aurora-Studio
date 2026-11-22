use super::camera_clut::lab_v4_to_icc_xyz;
use super::icc_grid::quantize_output;

const POINTS: usize = 33;
const CURVE_LEN: usize = 1025;

pub struct CameraPcs {
    curves: [Vec<u16>; 3],
    nodes: Vec<[u16; 3]>,
}

fn be_u32(bytes: &[u8], offset: usize) -> Option<usize> {
    Some(u32::from_be_bytes(bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?) as usize)
}

pub(crate) fn be_word(bytes: &[u8]) -> u16 {
    u16::from_be_bytes([bytes[0], bytes[1]])
}

pub(crate) fn normalized(word: u16) -> f32 {
    (f64::from(word) / 65535.0) as f32
}

fn coordinate(word: u16, domain: u32) -> (usize, i32) {
    let integer = u32::from(word) * domain;
    let fixed = integer + (integer + 0x7fff) / 0xffff;
    ((fixed >> 16) as usize, (fixed & 0xffff) as i32)
}

pub(crate) fn curve_word(word: u16, curve: &[u16]) -> u16 {
    let (lower, fraction) = coordinate(word, (curve.len() - 1) as u32);
    let left = i32::from(curve[lower]);
    let right = i32::from(curve[(lower + 1).min(curve.len() - 1)]);
    (left + ((right - left).wrapping_mul(fraction).wrapping_add(0x8000) >> 16)) as u16
}

pub(crate) fn icc_tag<'a>(profile: &'a [u8], signature: &[u8; 4]) -> Option<&'a [u8]> {
    if profile.get(36..40)? != b"acsp" {
        return None;
    }
    let count = be_u32(profile, 128)?;
    let directory = profile.get(132..132usize.checked_add(count.checked_mul(12)?)?)?;
    let entry = directory.chunks_exact(12).find(|entry| &entry[..4] == signature)?;
    let offset = be_u32(entry, 4)?;
    let length = be_u32(entry, 8)?;
    profile.get(offset..offset.checked_add(length)?)
}

impl CameraPcs {
    pub fn from_prostandard(profile: &[u8]) -> Option<Self> {
        if profile.get(16..24)? != b"RGB Lab " {
            return None;
        }
        let tag = icc_tag(profile, b"A2B0")?;
        let table_start = 52 + 3 * CURVE_LEN * 2;
        let table_end = table_start + POINTS.pow(3) * 6;
        if tag.len() != table_end + 12
            || &tag[..4] != b"mft2"
            || tag[8..11] != [3, 3, POINTS as u8]
            || tag[48..52] != [4, 1, 0, 2]
            || tag[12..48].chunks_exact(4).enumerate().any(|(index, word)| {
                be_u32(word, 0) != Some(if index % 4 == 0 { 65536 } else { 0 })
            })
            || tag[table_end..].chunks_exact(4).any(|curve| curve != [0, 0, 255, 255])
        {
            return None;
        }
        let curves = std::array::from_fn(|channel| {
            let start = 52 + channel * CURVE_LEN * 2;
            tag[start..start + CURVE_LEN * 2].chunks_exact(2).map(be_word).collect()
        });
        let nodes = tag[table_start..table_end].chunks_exact(6)
            .map(|words| [be_word(words), be_word(&words[2..]), be_word(&words[4..])]).collect();
        Some(Self { curves, nodes })
    }

    fn sample_lab(&self, rgb: [u16; 3]) -> [u16; 3] {
        let positions = rgb.map(|word| coordinate(word, (POINTS - 1) as u32));
        let strides = [POINTS * POINTS, POINTS, 1];
        let mut axes = [0, 1, 2];
        axes.sort_by_key(|&axis| (std::cmp::Reverse(positions[axis].1), axis));
        let mut index: usize = (0..3).map(|axis| positions[axis].0 * strides[axis]).sum();
        let base = self.nodes[index];
        let mut previous = base;
        let mut residual = [0x8001_i32; 3];
        for axis in axes {
            if rgb[axis] != u16::MAX {
                index += strides[axis];
            }
            let next = self.nodes[index];
            for channel in 0..3 {
                let delta = i32::from(next[channel]) - i32::from(previous[channel]);
                residual[channel] = residual[channel].wrapping_add(delta.wrapping_mul(positions[axis].1));
            }
            previous = next;
        }
        std::array::from_fn(|channel| {
            let value = residual[channel];
            base[channel].wrapping_add((value.wrapping_add(value >> 16) >> 16) as u16)
        })
    }

    fn stages(&self, rgb: [f32; 3]) -> [[f32; 3]; 5] {
        let shaped = std::array::from_fn(|channel| {
            normalized(curve_word(quantize_output(rgb[channel]), &self.curves[channel]))
        });
        let lab_table = self.sample_lab(shaped.map(quantize_output)).map(normalized);
        let lab_v2 = lab_table.map(|value| normalized(quantize_output(value)));
        let lab_v4 = lab_v2.map(|value| (f64::from(value) * (65535.0 / 65280.0)) as f32);
        [shaped, lab_table, lab_v2, lab_v4, lab_v4_to_icc_xyz(lab_v4)]
    }

    /// ICC normalized D50 XYZ (physical XYZ times 32768/65535).
    pub fn to_xyz(&self, rgb: [f32; 3]) -> [f32; 3] {
        self.stages(rgb)[4]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_profiles_are_rejected() {
        assert!(CameraPcs::from_prostandard(&[]).is_none());
        let mut profile = [0; 144];
        profile[16..24].copy_from_slice(b"RGB Lab ");
        profile[36..40].copy_from_slice(b"acsp");
        profile[128..132].copy_from_slice(&1u32.to_be_bytes());
        profile[132..136].copy_from_slice(b"A2B0");
        profile[136..144].fill(255);
        assert!(CameraPcs::from_prostandard(&profile).is_none());
    }

}
