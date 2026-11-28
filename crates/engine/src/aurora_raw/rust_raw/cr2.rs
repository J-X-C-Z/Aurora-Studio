//! Canon CR2 sensor decode: lossless JPEG strips plus the CR2 slice table
//! (`0xc640`) that maps the encoder's row-major stream back onto the sensor.

use super::tiff::{Ifd, Tiff};
use super::{cfa_filters, make_frame, meta, NativeError, NativeResult};

use crate::raw::BayerFrame;

pub(crate) fn decode(tiff: &Tiff<'_>, ifd0: &Ifd<'_>, _bytes: &[u8]) -> NativeResult<BayerFrame> {
    let ifd = super::raw_ifd(tiff)?;
    let raw_width = ifd.u32(0x0100).ok_or_else(|| NativeError::corrupt("CR2 has no image width"))? as usize;
    let raw_height = ifd.u32(0x0101).ok_or_else(|| NativeError::corrupt("CR2 has no image height"))? as usize;
    if raw_width == 0 || raw_height == 0 || raw_width > 100_000 || raw_height > 100_000 {
        return Err(NativeError::corrupt("CR2 dimensions are out of range"));
    }
    let bits = ifd.u16(0x0102).unwrap_or(14).clamp(8, 16) as u8;
    let compression = ifd.u16(0x0103).unwrap_or(7);
    let (filters, colors) = cfa_filters(&ifd)?;
    if filters == 9 {
        return Err(NativeError::unsupported("X-Trans sensor in a CR2 container"));
    }
    if compression != 7 {
        return Err(NativeError::unsupported(format!("Canon CR2 compression {compression}")));
    }
    let offset = ifd
        .u32s(0x0111)
        .and_then(|values| values.ok())
        .and_then(|values| values.first().copied())
        .ok_or_else(|| NativeError::corrupt("CR2 has no strip offset"))? as usize;
    let count = ifd
        .u32s(0x0117)
        .and_then(|values| values.ok())
        .and_then(|values| values.first().copied())
        .unwrap_or_else(|| (tiff.data().len().saturating_sub(offset)) as u32) as usize;
    let strip = tiff
        .data()
        .get(offset..offset.saturating_add(count))
        .ok_or_else(|| NativeError::corrupt("CR2 strip is outside the file"))?;

    let image = super::ljpeg::decode(strip, false)?;
    let jwide = image.width * image.components;
    if jwide == 0 || image.height == 0 {
        return Err(NativeError::corrupt("CR2 lossless JPEG has no samples"));
    }
    let slice: Option<[usize; 3]> = ifd.u16s(0xc640).and_then(|values| values.ok()).and_then(|values| {
        (values.len() >= 3).then(|| [values[0] as usize, values[1] as usize, values[2] as usize])
    });
    let total_samples = image.height.saturating_mul(jwide).min(image.data.len());
    let plane = if let Some(slice) = slice.filter(|slice| slice[0] != 0 && slice[1] != 0) {
        let mut plane = vec![0u16; raw_width * raw_height];
        let slice_samples = slice[1].saturating_mul(raw_height);
        for half in 0..=slice[0] {
            let src_base = half.saturating_mul(slice_samples);
            if src_base >= total_samples {
                break;
            }
            let width = if half < slice[0] { slice[1] } else { slice[2] };
            if width == 0 {
                return Err(NativeError::corrupt("CR2 slice table has a zero width"));
            }
            let x_base = half * slice[1];
            if x_base >= raw_width {
                continue;
            }
            let max_copy = width.min(raw_width - x_base);
            for r in 0..raw_height {
                let src_row = src_base + r * width;
                if src_row >= total_samples {
                    break;
                }
                let copy_len = max_copy.min(total_samples - src_row);
                let dst_row = r * raw_width + x_base;
                plane[dst_row..dst_row + copy_len]
                    .copy_from_slice(&image.data[src_row..src_row + copy_len]);
            }
        }
        plane
    } else if total_samples == raw_width * raw_height && image.data.len() == total_samples {
        image.data
    } else {
        let mut plane = vec![0u16; raw_width * raw_height];
        let copy_len = total_samples.min(plane.len());
        plane[..copy_len].copy_from_slice(&image.data[..copy_len]);
        plane
    };

    let (left, top, width, height) = super::active_area(&ifd, raw_width, raw_height);
    let black = ifd
        .reals(0xc61a)
        .and_then(|values| values.ok())
        .and_then(|values| (values.len() == 1 && values[0] > 0.0).then(|| values[0].round() as u32))
        .unwrap_or(0);
    let maximum = ifd.u32(0xc61d).unwrap_or_else(|| (1u32 << bits.min(16)) - 1);
    let make = ifd0.ascii(0x010f).unwrap_or_default();
    let model = ifd0.ascii(0x0110).unwrap_or_default();
    let rgb_cam = meta::adobe_matrix(&make, &model)
        .map(|matrix| meta::cam_xyz_to_rgb_cam(&matrix, colors as usize))
        .unwrap_or([[0.0f32; 4]; 3]);

    make_frame(
        raw_width,
        raw_height,
        left,
        top,
        width,
        height,
        bits,
        plane,
        filters,
        colors,
        black,
        maximum,
        [0.0; 4],
        rgb_cam,
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn slice_table_split_matches_libraw_formula() {
        // Two slices of 4 and 3 columns, 2 rows each: the first half of the
        // stream holds slice 0, the second half slice 1.
        let slice = [2usize, 4, 3];
        let raw_height = 2;
        let jwide = 8;
        let mut placed = vec![(usize::MAX, usize::MAX); jwide * 2];
        for jrow in 0..2 {
            for jcol in 0..jwide {
                let mut index = jrow * jwide + jcol;
                let mut half = index / (slice[1] * raw_height);
                let second = half >= slice[0];
                if second {
                    half = slice[0];
                }
                let width = slice[1 + usize::from(second)];
                index -= half * (slice[1] * raw_height);
                let row = index / width;
                let col = index % width + half * slice[1];
                placed[jrow * jwide + jcol] = (row, col);
            }
        }
        assert_eq!(placed[0], (0, 0));
        assert_eq!(placed[7], (1, 3));
        assert_eq!(placed[8], (0, 4));
        assert_eq!(placed[15], (1, 7));
    }
}
