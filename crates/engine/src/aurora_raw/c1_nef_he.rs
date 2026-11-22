//! Select the HE CFA strip by TIFF metadata, never by searching for a marker.
use std::io::{Error, ErrorKind, Result};

pub struct NikonHeStrip<'a> {
    pub bytes: &'a [u8],
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NikonBlackLevels(pub [u16; 4]);

impl NikonBlackLevels {
    pub fn capture_one_scalar(self) -> u32 {
        self.0.into_iter().map(u32::from).sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NikonCrop {
    pub left: usize,
    pub top: usize,
    pub width: usize,
    pub height: usize,
}

impl NikonCrop {
    pub fn cfa_phase(self) -> (usize, usize) {
        (self.left & 1, self.top & 1)
    }

    pub fn validate(self, frame_width: usize, frame_height: usize) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(Error::new(ErrorKind::InvalidData, "Nikon crop is empty"));
        }
        let right = self
            .left
            .checked_add(self.width)
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Nikon crop overflows"))?;
        let bottom = self
            .top
            .checked_add(self.height)
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Nikon crop overflows"))?;
        if right > frame_width || bottom > frame_height {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Nikon crop outside frame",
            ));
        }
        Ok(())
    }
}

struct Tiff<'a> {
    bytes: &'a [u8],
    big_endian: bool,
}

impl<'a> Tiff<'a> {
    fn from_bytes(bytes: &'a [u8]) -> Option<Self> {
        let big_endian = match bytes.get(..4) {
            Some(b"II\x2a\x00") => false,
            Some(b"MM\x00\x2a") => true,
            _ => return None,
        };
        Some(Self { bytes, big_endian })
    }

    fn root(&self) -> Result<&'a [u8]> {
        self.ifd(self.long(self.range(4, 4)?))
    }

    fn range(&self, offset: usize, length: usize) -> Result<&'a [u8]> {
        offset
            .checked_add(length)
            .and_then(|end| self.bytes.get(offset..end))
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "NEF offset outside file"))
    }

    fn short(&self, bytes: &[u8]) -> usize {
        let value = [bytes[0], bytes[1]];
        usize::from(if self.big_endian {
            u16::from_be_bytes(value)
        } else {
            u16::from_le_bytes(value)
        })
    }

    fn long(&self, bytes: &[u8]) -> usize {
        let value = [bytes[0], bytes[1], bytes[2], bytes[3]];
        (if self.big_endian {
            u32::from_be_bytes(value)
        } else {
            u32::from_le_bytes(value)
        }) as usize
    }

    fn ifd(&self, offset: usize) -> Result<&'a [u8]> {
        let entries = self.short(self.range(offset, 2)?);
        self.range(offset, 2 + entries * 12 + 4)
            .map(|ifd| &ifd[2..2 + entries * 12])
    }

    fn entry<'b>(&self, ifd: &'b [u8], tag: usize) -> Option<&'b [u8]> {
        ifd.chunks_exact(12).find(|entry| self.short(entry) == tag)
    }

    fn scalar(&self, ifd: &[u8], tag: usize) -> Option<usize> {
        let entry = self.entry(ifd, tag)?;
        if self.long(&entry[4..]) != 1 {
            return None;
        }
        match self.short(&entry[2..]) {
            3 => Some(self.short(&entry[8..])),
            4 => Some(self.long(&entry[8..])),
            _ => None,
        }
    }
}

fn nikon_maker_tiff(nef: &[u8]) -> Result<Option<Tiff<'_>>> {
    let Some(tiff) = Tiff::from_bytes(nef) else {
        return Ok(None);
    };
    let Some(exif_offset) = tiff.scalar(tiff.root()?, 0x8769) else {
        return Ok(None);
    };
    let Some(entry) = tiff.entry(tiff.ifd(exif_offset)?, 0x927c) else {
        return Ok(None);
    };
    let count = tiff.long(&entry[4..]);
    if tiff.short(&entry[2..]) != 7 || count < 18 {
        return Ok(None);
    }
    let note = tiff.range(tiff.long(&entry[8..]), count)?;
    if !note.starts_with(b"Nikon\0\x02") {
        return Ok(None);
    }
    Ok(Tiff::from_bytes(&note[10..]))
}

/// Read the EXIF sensitivity selection used before Nikon MakerNote fallback.
#[allow(dead_code)]
pub fn nikon_exif_sensitivity(nef: &[u8]) -> Result<Option<i32>> {
    let Some(tiff) = Tiff::from_bytes(nef) else { return Ok(None) };
    let Some(exif_offset) = tiff.scalar(tiff.root()?, 0x8769) else { return Ok(None) };
    let exif = tiff.ifd(exif_offset)?;
    let tag = match tiff.scalar(exif, 0x8830) {
        Some(1 | 4) => 0x8831,
        Some(2) => 0x8832,
        Some(3 | 5 | 6 | 7) => 0x8833,
        _ => return Ok(None),
    };
    Ok(tiff.scalar(exif, tag).and_then(|value| i32::try_from(value).ok()).filter(|&value| value > 0))
}

/// Returns tag 0x003D as stored. Missing/unsupported metadata is not a zero black level.
pub fn nikon_black_levels(nef: &[u8]) -> Result<Option<NikonBlackLevels>> {
    let Some(nikon) = nikon_maker_tiff(nef)? else {
        return Ok(None);
    };
    let Some(entry) = nikon.entry(nikon.root()?, 0x003d) else {
        return Ok(None);
    };
    if nikon.short(&entry[2..]) != 3 || nikon.long(&entry[4..]) != 4 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Nikon black level must contain four SHORT values",
        ));
    }
    let values = nikon.range(nikon.long(&entry[8..]), 8)?;
    Ok(Some(NikonBlackLevels(std::array::from_fn(|index| {
        nikon.short(&values[2 * index..]) as u16
    }))))
}

/// Reads the as-shot RGB multipliers from the verified four-rational Nikon layout.
pub fn nikon_as_shot_rgb(nef: &[u8]) -> Result<Option<[f32; 3]>> {
    let Some(nikon) = nikon_maker_tiff(nef)? else {
        return Ok(None);
    };
    let Some(entry) = nikon.entry(nikon.root()?, 0x000c) else {
        return Ok(None);
    };
    if nikon.short(&entry[2..]) != 5 || nikon.long(&entry[4..]) != 4 {
        return Err(Error::new(ErrorKind::InvalidData, "expected four Nikon WB rationals"));
    }
    let values = nikon.range(nikon.long(&entry[8..]), 32)?;
    let mut ratios = [0.0; 4];
    for (ratio, pair) in ratios.iter_mut().zip(values.chunks_exact(8)) {
        let numerator = nikon.long(pair);
        let denominator = nikon.long(&pair[4..]);
        if numerator == 0 || denominator == 0 {
            return Err(Error::new(ErrorKind::InvalidData, "invalid Nikon WB ratio"));
        }
        *ratio = (numerator as f64 / denominator as f64) as f32;
    }
    let [red, blue, green, second_green] = ratios;
    if green != second_green {
        return Err(Error::new(ErrorKind::Unsupported, "unequal Nikon green WB gains"));
    }
    Ok(Some([red, green, blue]))
}

/// Returns the Nikon MakerNote 0x0045 raw inset crop as stored: left, top, width, height.
/// Missing metadata is not replaced by another tag or by a full-frame fallback.
pub fn nikon_crop(nef: &[u8]) -> Result<Option<NikonCrop>> {
    let Some(nikon) = nikon_maker_tiff(nef)? else {
        return Ok(None);
    };
    let Some(entry) = nikon.entry(nikon.root()?, 0x0045) else {
        return Ok(None);
    };
    if nikon.short(&entry[2..]) != 3 || nikon.long(&entry[4..]) != 4 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Nikon crop must contain four SHORT values",
        ));
    }
    let values = nikon.range(nikon.long(&entry[8..]), 8)?;
    let crop = NikonCrop {
        left: nikon.short(&values[0..]),
        top: nikon.short(&values[2..]),
        width: nikon.short(&values[4..]),
        height: nikon.short(&values[6..]),
    };
    if crop.width == 0 || crop.height == 0 {
        return Err(Error::new(ErrorKind::InvalidData, "Nikon crop is empty"));
    }
    Ok(Some(crop))
}

/// Supports the single-strip, 14-bit RGGB Nikon HE variant verified on Z6III.
pub fn find_he_strip(nef: &[u8]) -> Result<Option<NikonHeStrip<'_>>> {
    let Some(tiff) = Tiff::from_bytes(nef) else {
        return Ok(None);
    };
    let root = tiff.root()?;
    let sub_ifds = tiff.entry(root, 330);
    let offsets = match sub_ifds {
        Some(entry) if matches!(tiff.short(&entry[2..]), 4 | 13) => {
            let count = tiff.long(&entry[4..]);
            match count {
                0 => &[][..],
                1 => &entry[8..12],
                2..=64 => tiff.range(tiff.long(&entry[8..]), count * 4)?,
                _ => return Err(Error::new(ErrorKind::InvalidData, "too many NEF SubIFDs")),
            }
        }
        _ => &[][..],
    };
    let mut candidate = None;
    for index in 0..=offsets.len() / 4 {
        let ifd = if index == 0 {
            root
        } else {
            tiff.ifd(tiff.long(&offsets[4 * (index - 1)..]))?
        };
        let value = |tag| tiff.scalar(ifd, tag);
        if (value(258), value(259), value(262), value(277))
            != (Some(14), Some(34713), Some(32803), Some(1))
        {
            continue;
        }
        let Some((width, height)) = value(256).zip(value(257)) else {
            continue;
        };
        if !(64..=65534).contains(&width)
            || !(4..=65535).contains(&height)
            || value(278) != Some(height)
        {
            continue;
        }
        let pattern = tiff.entry(ifd, 0x828e);
        if !pattern.is_some_and(|entry| {
            tiff.short(&entry[2..]) == 1
                && tiff.long(&entry[4..]) == 4
                && entry[8..] == [0, 1, 1, 2]
        }) {
            continue;
        }
        let Some((offset, length)) = value(273).zip(value(279)) else {
            continue;
        };
        let bytes = tiff.range(offset, length)?;
        if !bytes.starts_with(&[0xff, 0x10, 0xff, 0x50]) || !bytes.ends_with(&[0xff, 0x11]) {
            continue;
        }
        if candidate.is_some() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "multiple Nikon HE CFA strips",
            ));
        }
        candidate = Some(NikonHeStrip {
            bytes,
            width,
            height,
        });
    }
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exif_sensitivity_uses_capture_one_tag_precedence() {
        for (kind, expected) in [None, Some(100), Some(200), Some(400), Some(100), Some(400), Some(400), Some(400), None].into_iter().enumerate() {
            let mut nef = Vec::from(*b"II\x2a\x00\x08\x00\x00\x00");
            nef.extend([1, 0, 0x69, 0x87, 4, 0, 1, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0]);
            nef.extend(4u16.to_le_bytes());
            for (tag, value) in [(0x8830u16, kind as u32), (0x8831, 100), (0x8832, 200), (0x8833, 400)] {
                nef.extend(tag.to_le_bytes());
                nef.extend(4u16.to_le_bytes());
                nef.extend(1u32.to_le_bytes());
                nef.extend(value.to_le_bytes());
            }
            nef.extend([0; 4]);
            assert_eq!(nikon_exif_sensitivity(&nef).unwrap(), expected, "type {kind}");
        }
    }

    #[test]
    fn rejects_non_tiff_and_truncated_ifd() {
        assert!(find_he_strip(&[0xff, 0x10, 0xff, 0x50]).unwrap().is_none());
        assert!(find_he_strip(b"II\x2a\x00\xff\xff\xff\xff").is_err());
        assert!(find_he_strip(b"MM\x00\x2a\x00\x00\x00\x08\x00\x01").is_err());
    }

    #[test]
    fn nikon_note_uses_its_own_byte_order_and_bounds() {
        let mut nef = Vec::from(*b"II\x2a\x00\x08\x00\x00\x00");
        nef.extend([1, 0, 0x69, 0x87, 4, 0, 1, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0]);
        nef.extend([1, 0, 0x7c, 0x92, 7, 0, 44, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0]);
        nef.extend(*b"Nikon\0\x02\x11\0\0MM\0\x2a\0\0\0\x08");
        nef.extend([0, 1, 0, 0x3d, 0, 3, 0, 0, 0, 4, 0, 0, 0, 26, 0, 0, 0, 0]);
        nef.extend([0x03, 0xe8, 0x03, 0xe9, 0x03, 0xea, 0x03, 0xeb]);
        let levels = nikon_black_levels(&nef).unwrap().unwrap();
        assert_eq!(levels.0, [1000, 1001, 1002, 1003]);
        assert_eq!(levels.capture_one_scalar(), 4006);
        nef[75] = 27;
        nef.push(0);
        assert_eq!(
            nikon_black_levels(&nef).unwrap_err().kind(),
            ErrorKind::InvalidData
        );
        assert_eq!(nikon_black_levels(b"other file").unwrap(), None);
    }

    #[test]
    fn crop_bounds_are_checked_without_fallback() {
        let crop = NikonCrop {
            left: 8,
            top: 4,
            width: 6048,
            height: 4032,
        };
        crop.validate(6064, 4040).unwrap();
        assert_eq!(crop.cfa_phase(), (0, 0));
        assert!(crop.validate(6055, 4040).is_err());
        assert!(crop.validate(6064, 4035).is_err());
        assert!(NikonCrop { width: 0, ..crop }.validate(6064, 4040).is_err());
        assert_eq!(nikon_crop(b"other file").unwrap(), None);
    }

    #[test]
    fn as_shot_reads_inner_tiff_and_rejects_invalid_rationals() {
        let mut nef = Vec::from(*b"II\x2a\x00\x08\x00\x00\x00");
        nef.extend([1, 0, 0x69, 0x87, 4, 0, 1, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0]);
        nef.extend([1, 0, 0x7c, 0x92, 7, 0, 68, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0]);
        nef.extend(*b"Nikon\0\x02\x11\0\0MM\0\x2a\0\0\0\x08");
        nef.extend([0, 1, 0, 0x0c, 0, 5, 0, 0, 0, 4, 0, 0, 0, 26, 0, 0, 0, 0]);
        for value in [3_u32, 2, 5, 4, 2, 2, 4, 4] {
            nef.extend(value.to_be_bytes());
        }
        assert_eq!(nikon_as_shot_rgb(&nef).unwrap(), Some([1.5, 1.0, 1.25]));
        nef[107] = 3;
        assert_eq!(nikon_as_shot_rgb(&nef).unwrap_err().kind(), ErrorKind::Unsupported);
        nef[111] = 0;
        assert_eq!(nikon_as_shot_rgb(&nef).unwrap_err().kind(), ErrorKind::InvalidData);
    }
}
