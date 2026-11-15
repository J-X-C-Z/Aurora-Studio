//! Nikon NEF manufacturer lens-correction data.
//!
//! Capture One's "Manufacturer Profile" for Nikon Z bodies is read from the
//! NEF's `SubIFD1` tag `0xC7D5` (`NikonNEFInfo`). The nested TIFF contains
//! `0x0005 DistortionInfo` and `0x0006 VignetteInfo`, both plain (not
//! encrypted) rational coefficients.

use anyhow::{Context, Result};
use std::path::Path;

const TAG_SUB_IFDS: u16 = 0x014a;
const TAG_EXIF_IFD: u16 = 0x8769;
const TAG_FOCAL_LENGTH: u16 = 0x920a;
const TAG_NEF_INFO: u16 = 0xc7d5;
const TAG_DISTORTION_INFO: u16 = 0x0005;
const TAG_VIGNETTE_INFO: u16 = 0x0006;

pub const DISTORTION_INFO_LEN: usize = 84;
pub const VIGNETTE_INFO_LEN: usize = 116;

/// Meaning of `DistortionInfo[0x04]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistortionCorrection {
    /// 0: no lens attached; coefficients are normally zero.
    NoLensAttached,
    /// 1: vendor correction is optional.
    OnOptional,
    /// 2: vendor correction is off.
    Off,
    /// 3: vendor correction is required.
    OnRequired,
}

impl DistortionCorrection {
    fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::NoLensAttached),
            1 => Some(Self::OnOptional),
            2 => Some(Self::Off),
            3 => Some(Self::OnRequired),
            _ => None,
        }
    }

    /// True when a consumer should apply geometric correction by default.
    pub fn is_enabled(self) -> bool {
        matches!(self, Self::OnOptional | Self::OnRequired)
    }

    pub fn is_required(self) -> bool {
        self == Self::OnRequired
    }
}

/// Cubic radial distortion: `r' = r * (1 + k1*r^2 + k2*r^4 + k3*r^6)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManufacturerDistortion {
    pub k1: f32,
    pub k2: f32,
    pub k3: f32,
}

/// Sixth-order even vignetting polynomial coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManufacturerVignetting {
    pub v1: f32,
    pub v2: f32,
    pub v3: f32,
}

/// One NEF's manufacturer correction data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManufacturerLensData {
    pub distortion: Option<ManufacturerDistortion>,
    pub vignetting: Option<ManufacturerVignetting>,
    pub distortion_correction: Option<DistortionCorrection>,
    /// EXIF `0x920A`; `None` when absent or invalid.
    pub focal_length_mm: Option<f32>,
}

impl ManufacturerLensData {
    /// False for NaN/Inf coefficients, or when every correction coefficient is
    /// zero. Consumers should not build a correction from an unusable record.
    pub fn is_usable(&self) -> bool {
        let finite = self.distortion.map_or(true, |d| {
            d.k1.is_finite() && d.k2.is_finite() && d.k3.is_finite()
        }) && self.vignetting.map_or(true, |v| {
            v.v1.is_finite() && v.v2.is_finite() && v.v3.is_finite()
        });
        let any = self.distortion.map_or(false, |d| d.k1 != 0.0 || d.k2 != 0.0 || d.k3 != 0.0)
            || self.vignetting.map_or(false, |v| v.v1 != 0.0 || v.v2 != 0.0 || v.v3 != 0.0);
        finite && any
    }
}

/// Parse the first NEF IFD that carries `0xC7D5`.
///
/// Returns `None` for a missing tag, truncated/mismatched subtag, zero
/// coefficient set, zero denominator or non-finite value. It never panics.
pub fn parse_nef_lens_data(bytes: &[u8]) -> Option<ManufacturerLensData> {
    let tiff = Tiff::new(bytes)?;
    let ifd0 = tiff.u32_at(4)? as usize;
    let focal_length_mm = read_focal_length(&tiff, ifd0);
    let nef_info = find_nef_info(&tiff, ifd0)?;
    let nested = Tiff::new(nested_tiff(nef_info)?)?;
    let nested_ifd = nested.u32_at(4)? as usize;

    let distortion = nested
        .entry(nested_ifd, TAG_DISTORTION_INFO)
        .and_then(|entry| parse_distortion(entry.value));
    let vignetting = nested
        .entry(nested_ifd, TAG_VIGNETTE_INFO)
        .and_then(|entry| parse_vignetting(entry.value));

    let (distortion, distortion_correction) = match distortion {
        Some((correction, values)) => (Some(values), Some(correction)),
        None => (None, None),
    };
    if distortion.is_none() && vignetting.is_none() {
        return None;
    }
    let result = ManufacturerLensData {
        distortion,
        vignetting,
        distortion_correction,
        focal_length_mm,
    };
    result.is_usable().then_some(result)
}

/// Read a NEF from disk. A missing `0xC7D5` yields `Ok(None)`.
pub fn parse_nef_lens_data_file(path: &Path) -> Result<Option<ManufacturerLensData>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading NEF {}", path.display()))?;
    Ok(parse_nef_lens_data(&bytes))
}

fn parse_distortion(bytes: &[u8]) -> Option<(DistortionCorrection, ManufacturerDistortion)> {
    if bytes.len() != DISTORTION_INFO_LEN || &bytes[..4] != b"0100" {
        return None;
    }
    let correction = DistortionCorrection::from_code(bytes[4])?;
    let values = ManufacturerDistortion {
        k1: srational_at(bytes, 0x14)?,
        k2: srational_at(bytes, 0x1c)?,
        k3: srational_at(bytes, 0x24)?,
    };
    finite3(values.k1, values.k2, values.k3).then_some((correction, values))
}

fn parse_vignetting(bytes: &[u8]) -> Option<ManufacturerVignetting> {
    if bytes.len() != VIGNETTE_INFO_LEN || &bytes[..4] != b"0100" {
        return None;
    }
    let values = ManufacturerVignetting {
        v1: srational_at(bytes, 0x24)?,
        v2: srational_at(bytes, 0x34)?,
        v3: srational_at(bytes, 0x44)?,
    };
    finite3(values.v1, values.v2, values.v3).then_some(values)
}

fn finite3(a: f32, b: f32, c: f32) -> bool {
    a.is_finite() && b.is_finite() && c.is_finite()
}

fn read_focal_length(tiff: &Tiff<'_>, ifd0: usize) -> Option<f32> {
    let exif = tiff.entry(ifd0, TAG_EXIF_IFD)?;
    let exif_ifd = exif.first_u32()? as usize;
    let entry = tiff.entry(exif_ifd, TAG_FOCAL_LENGTH)?;
    let bytes = entry.value;
    let num = u32::from_le_bytes(bytes.get(0..4)?.try_into().ok()?) as f64;
    let den = u32::from_le_bytes(bytes.get(4..8)?.try_into().ok()?) as f64;
    if den == 0.0 {
        return None;
    }
    let value = (num / den) as f32;
    (value > 0.0 && value.is_finite()).then_some(value)
}

fn find_nef_info<'a>(tiff: &Tiff<'a>, ifd0: usize) -> Option<&'a [u8]> {
    if let Some(entry) = tiff.entry(ifd0, TAG_NEF_INFO) {
        return Some(entry.value);
    }
    let sub_ifds = tiff.entry(ifd0, TAG_SUB_IFDS)?;
    for offset in sub_ifds.u32_values() {
        if let Some(entry) = tiff.entry(offset as usize, TAG_NEF_INFO) {
            return Some(entry.value);
        }
    }
    None
}

fn nested_tiff(value: &[u8]) -> Option<&[u8]> {
    let start = value
        .windows(4)
        .take(24)
        .position(|w| w == b"II*\0" || w == b"MM\0*")?;
    value.get(start..)
}

fn srational_at(bytes: &[u8], offset: usize) -> Option<f32> {
    let num = i32::from_le_bytes(bytes.get(offset..offset + 4)?.try_into().ok()?);
    let den = i32::from_le_bytes(bytes.get(offset + 4..offset + 8)?.try_into().ok()?);
    if den == 0 {
        return None;
    }
    let value = (num as f64 / den as f64) as f32;
    value.is_finite().then_some(value)
}

fn type_size(typ: u16) -> Option<usize> {
    match typ {
        1 | 2 | 7 => Some(1),
        3 => Some(2),
        4 | 9 | 13 => Some(4),
        5 | 10 => Some(8),
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct Entry<'a> {
    typ: u16,
    count: u32,
    value: &'a [u8],
}

impl<'a> Entry<'a> {
    fn first_u32(self) -> Option<u32> {
        self.u32_values().into_iter().next()
    }

    fn u32_values(self) -> Vec<u32> {
        let mut out = Vec::new();
        let size = type_size(self.typ).unwrap_or(0);
        if size == 0 {
            return out;
        }
        let mut i = 0;
        while i + size <= self.value.len() {
            let v = match self.typ {
                3 => u16::from_le_bytes([self.value[i], self.value[i + 1]]) as u32,
                4 | 13 => u32::from_le_bytes([
                    self.value[i],
                    self.value[i + 1],
                    self.value[i + 2],
                    self.value[i + 3],
                ]),
                _ => return out,
            };
            out.push(v);
            i += size;
        }
        let _ = self.count;
        out
    }
}

struct Tiff<'a> {
    data: &'a [u8],
}

impl<'a> Tiff<'a> {
    fn new(data: &'a [u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }
        match &data[..4] {
            b"II*\0" | b"MM\0*" => Some(Self { data }),
            _ => None,
        }
    }

    fn u16_at(&self, offset: usize) -> Option<u16> {
        let bytes = self.data.get(offset..offset + 2)?;
        Some(if self.little() {
            u16::from_le_bytes(bytes.try_into().ok()?)
        } else {
            u16::from_be_bytes(bytes.try_into().ok()?)
        })
    }

    fn u32_at(&self, offset: usize) -> Option<u32> {
        let bytes = self.data.get(offset..offset + 4)?;
        Some(if self.little() {
            u32::from_le_bytes(bytes.try_into().ok()?)
        } else {
            u32::from_be_bytes(bytes.try_into().ok()?)
        })
    }

    fn little(&self) -> bool {
        self.data.starts_with(b"II")
    }

    fn entry(&self, ifd_offset: usize, tag: u16) -> Option<Entry<'a>> {
        let count = self.u16_at(ifd_offset)? as usize;
        for index in 0..count {
            let offset = ifd_offset.checked_add(2 + index * 12)?;
            let entry_tag = self.u16_at(offset)?;
            if entry_tag != tag {
                continue;
            }
            let typ = self.u16_at(offset + 2)?;
            let item_count = self.u32_at(offset + 4)?;
            let size = type_size(typ)?.checked_mul(item_count as usize)?;
            let value_offset = offset + 8;
            let value = if size <= 4 {
                self.data.get(value_offset..value_offset + size)?
            } else {
                let absolute = self.u32_at(value_offset)? as usize;
                self.data.get(absolute..absolute.checked_add(size)?)?
            };
            return Some(Entry {
                typ,
                count: item_count,
                value,
            });
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_srational(bytes: &mut [u8], offset: usize, num: i32, den: i32) {
        bytes[offset..offset + 4].copy_from_slice(&num.to_le_bytes());
        bytes[offset + 4..offset + 8].copy_from_slice(&den.to_le_bytes());
    }

    fn distortion(k1: i32, k2: i32, k3: i32, correction: u8) -> Vec<u8> {
        let mut bytes = vec![0u8; DISTORTION_INFO_LEN];
        bytes[..4].copy_from_slice(b"0100");
        bytes[4] = correction;
        put_srational(&mut bytes, 0x14, k1, 0x100000);
        put_srational(&mut bytes, 0x1c, k2, 0x100000);
        put_srational(&mut bytes, 0x24, k3, 0x100000);
        bytes
    }

    fn vignette(v1: i32, v2: i32, v3: i32) -> Vec<u8> {
        let mut bytes = vec![0u8; VIGNETTE_INFO_LEN];
        bytes[..4].copy_from_slice(b"0100");
        put_srational(&mut bytes, 0x24, v1, 0x100000);
        put_srational(&mut bytes, 0x34, v2, 0x100000);
        put_srational(&mut bytes, 0x44, v3, 0x100000);
        bytes
    }

    fn build_nef(dist: &[u8], vig: &[u8]) -> Vec<u8> {
        let mut nested = Vec::new();
        nested.extend_from_slice(b"II*\0");
        nested.extend_from_slice(&8u32.to_le_bytes());
        let dist_offset = 8 + 2 + 24 + 4;
        let vig_offset = dist_offset + dist.len();
        nested.extend_from_slice(&2u16.to_le_bytes());
        nested.extend_from_slice(&TAG_DISTORTION_INFO.to_le_bytes());
        nested.extend_from_slice(&7u16.to_le_bytes());
        nested.extend_from_slice(&(dist.len() as u32).to_le_bytes());
        nested.extend_from_slice(&(dist_offset as u32).to_le_bytes());
        nested.extend_from_slice(&TAG_VIGNETTE_INFO.to_le_bytes());
        nested.extend_from_slice(&7u16.to_le_bytes());
        nested.extend_from_slice(&(vig.len() as u32).to_le_bytes());
        nested.extend_from_slice(&(vig_offset as u32).to_le_bytes());
        nested.extend_from_slice(&0u32.to_le_bytes());
        nested.extend_from_slice(dist);
        nested.extend_from_slice(vig);

        let mut nef_info = Vec::new();
        nef_info.extend_from_slice(b"Nikon\0\x01\x03\0\0");
        nef_info.extend_from_slice(&nested);

        let ifd0_offset = 8usize;
        let sub_ifd_offset = ifd0_offset + 2 + 12 + 4;
        let nef_info_offset = sub_ifd_offset + 2 + 12 + 4;
        let mut nef = Vec::new();
        nef.extend_from_slice(b"II*\0");
        nef.extend_from_slice(&(ifd0_offset as u32).to_le_bytes());
        nef.extend_from_slice(&1u16.to_le_bytes());
        nef.extend_from_slice(&TAG_SUB_IFDS.to_le_bytes());
        nef.extend_from_slice(&4u16.to_le_bytes());
        nef.extend_from_slice(&1u32.to_le_bytes());
        nef.extend_from_slice(&(sub_ifd_offset as u32).to_le_bytes());
        nef.extend_from_slice(&0u32.to_le_bytes());
        nef.extend_from_slice(&1u16.to_le_bytes());
        nef.extend_from_slice(&TAG_NEF_INFO.to_le_bytes());
        nef.extend_from_slice(&7u16.to_le_bytes());
        nef.extend_from_slice(&(nef_info.len() as u32).to_le_bytes());
        nef.extend_from_slice(&(nef_info_offset as u32).to_le_bytes());
        nef.extend_from_slice(&0u32.to_le_bytes());
        nef.extend_from_slice(&nef_info);
        nef
    }

    #[test]
    fn parses_synthetic_nef_info() {
        let data = build_nef(
            &distortion(83_100, -38_835, -127_700, 3),
            &vignette(-5_430_000, 1_230_000, 599_000),
        );
        let parsed = parse_nef_lens_data(&data).expect("valid NEFInfo");
        let d = parsed.distortion.expect("distortion");
        assert!((d.k1 - 0.07923).abs() < 1e-4);
        assert!((d.k2 + 0.03703).abs() < 1e-4);
        assert!((d.k3 + 0.12178).abs() < 1e-4);
        assert_eq!(parsed.distortion_correction, Some(DistortionCorrection::OnRequired));
        let v = parsed.vignetting.expect("vignetting");
        assert!((v.v1 + 5.178).abs() < 1e-3);
    }

    #[test]
    fn truncated_distortion_is_rejected() {
        let data = build_nef(&distortion(1, 2, 3, 1)[..40], &vignette(1, 2, 3));
        let parsed = parse_nef_lens_data(&data).expect("vignetting still valid");
        assert!(parsed.distortion.is_none());
        assert!(parsed.vignetting.is_some());
    }

    #[test]
    fn wrong_vignette_length_is_rejected() {
        let data = build_nef(&distortion(1, 2, 3, 1), &vignette(1, 2, 3)[..100]);
        let parsed = parse_nef_lens_data(&data).expect("distortion still valid");
        assert!(parsed.vignetting.is_none());
        assert!(parsed.distortion.is_some());
    }

    #[test]
    fn zero_coefficients_are_rejected() {
        let data = build_nef(&distortion(0, 0, 0, 0), &vignette(0, 0, 0));
        assert!(parse_nef_lens_data(&data).is_none());
    }

    #[test]
    fn zero_denominator_is_rejected() {
        let mut dist = distortion(1, 2, 3, 1);
        put_srational(&mut dist, 0x14, 1, 0);
        let data = build_nef(&dist, &vignette(1, 2, 3));
        assert!(parse_nef_lens_data(&data).expect("vignetting").distortion.is_none());
    }

    #[test]
    fn unknown_distortion_flag_is_rejected() {
        let data = build_nef(&distortion(1, 2, 3, 9), &vignette(1, 2, 3));
        let parsed = parse_nef_lens_data(&data).expect("vignetting");
        assert!(parsed.distortion.is_none());
        assert!(parsed.distortion_correction.is_none());
    }

    #[test]
    fn truncated_file_does_not_panic() {
        let data = build_nef(&distortion(1, 2, 3, 1), &vignette(1, 2, 3));
        for len in 0..data.len() {
            let _ = parse_nef_lens_data(&data[..len]);
        }
    }

    #[test]
    fn non_finite_values_are_not_usable() {
        let mut data = ManufacturerLensData {
            distortion: Some(ManufacturerDistortion {
                k1: 1.0,
                k2: 0.0,
                k3: 0.0,
            }),
            vignetting: None,
            distortion_correction: Some(DistortionCorrection::OnRequired),
            focal_length_mm: Some(50.0),
        };
        assert!(data.is_usable());
        data.distortion.as_mut().unwrap().k1 = f32::NAN;
        assert!(!data.is_usable());
        data.distortion.as_mut().unwrap().k1 = f32::INFINITY;
        assert!(!data.is_usable());
    }

    #[test]
    #[ignore]
    fn real_nef_samples() {
        for (label, variable, expect) in [
            ("24mm", "AURORA_NEF_24", true),
            ("50mm", "AURORA_NEF_50", false),
            ("120mm", "AURORA_NEF_120", true),
        ] {
            let Some(path) = std::env::var_os(variable) else {
                continue;
            };
            let data = std::fs::read(&path).expect("read real NEF");
            let parsed = parse_nef_lens_data(&data);
            println!("{label} {}: {parsed:?}", std::path::Path::new(&path).display());
            assert_eq!(parsed.is_some(), expect, "{label}");
        }
    }
}
