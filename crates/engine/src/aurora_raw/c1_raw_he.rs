//! Independent Nikon HE Bayer decoding for the verified Z6III variant.

#[path = "c1_nef_he.rs"]
mod c1_nef_he;
#[path = "c1_nikon_he.rs"]
mod c1_nikon_he;
#[path = "c1_stage1_direction.rs"]
mod c1_stage1_direction;
#[path = "c1_stage1_green.rs"]
mod c1_stage1_green;
#[path = "c1_stage1_params.rs"]
mod c1_stage1_params;
#[path = "c1_stage2_basic.rs"]
mod c1_stage2_basic;
#[path = "c1_stage3_params.rs"]
mod c1_stage3_params;
#[path = "c1_stage3_planar.rs"]
mod c1_stage3_planar;
#[path = "nikon_he_bitstream.rs"]
#[allow(dead_code)]
mod nikon_he_bitstream;
#[cfg(test)]
#[path = "stage1_reference.rs"]
#[allow(dead_code)]
mod stage1_reference;

use std::io::{Error, ErrorKind, Result};
use std::sync::mpsc;

use c1_nef_he::find_he_strip;
#[allow(unused_imports)]
pub use c1_nef_he::{NikonBlackLevels, NikonCrop, nikon_as_shot_rgb, nikon_black_levels, nikon_crop, nikon_exif_sensitivity};
pub use c1_stage1_params::Stage1Params;
pub use c1_stage3_params::Z6iiiStage3;
use c1_nikon_he::{bayer_frame, bayer_prefix, dequantize, reconstruct_precinct};
use nikon_he_bitstream::{Decoder, Weights};

const PIH_PROFILE: [u8; 25] = [
    0, 0, 0, 16, 4, 4, 8, 18, 0x44, 0x10, 0x51, 0x14, 0x50, 0x88, 0x70, 0x83, 0xf0, 0x15, 0x23,
    0xd1, 0x49, 0xcd, 0x3f, 0x7f, 0x07,
];
const CDT_PROFILE: [u8; 12] = [14, 0x11, 0x51, 14, 0x11, 0x51, 14, 0x11, 0, 14, 0x11, 0x51];

fn marker<'a>(rest: &mut &'a [u8], id: u8) -> Result<&'a [u8]> {
    let header = rest
        .get(..4)
        .ok_or_else(|| Error::new(ErrorKind::UnexpectedEof, "truncated HE marker"))?;
    if header[..2] != [0xff, id] {
        return Err(Error::new(ErrorKind::Unsupported, "unexpected HE marker"));
    }
    let length = usize::from(u16::from_be_bytes([header[2], header[3]]));
    if length < 2 || rest.len() < length + 2 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "invalid HE marker length",
        ));
    }
    let (segment, remaining) = rest.split_at(length + 2);
    *rest = remaining;
    Ok(segment)
}

#[cfg(test)]
fn decode_first_slice(nef: &[u8]) -> Result<Option<(usize, Vec<u16>)>> {
    decode(nef, true)
}

/// Decodes the verified variant to uncropped, unnormalized RGGB Bayer samples.
pub fn decode_frame(nef: &[u8]) -> Result<Option<(usize, Vec<u16>)>> {
    decode(nef, false)
}

/// Decodes the verified 14-bit HE variant in the C1 wrapper's left-aligned sample units.
/// Does not crop, subtract black, or normalize the white level.
pub fn decode_frame_left_aligned(nef: &[u8]) -> Result<Option<(usize, Vec<u16>)>> {
    Ok(decode_frame(nef)?.map(|(width, mut samples)| {
        for sample in &mut samples {
            *sample <<= 2;
        }
        (width, samples)
    }))
}

/// CPU RGB32F black-offset stage, when selected by the C1v2012 recipe.
/// Takes packed RGB rows in left-aligned sample units and preserves negative values.
pub fn subtract_black_rgb32f(row: &mut [[f32; 3]], black: i32) {
    let offset = -(black as f32);
    for sample in row.iter_mut().flatten() {
        *sample += offset;
    }
}

/// A cropped view over a decoded full-frame Bayer plane.
/// The full-image API (`decode_frame`) is unchanged; this type only exposes the
/// MakerNote 0x0045 inset and keeps the full plane as its backing storage.
pub struct NikonHeCropFrame {
    crop: NikonCrop,
    full_width: usize,
    full_height: usize,
    samples: Vec<u16>,
}

impl NikonHeCropFrame {
    pub(crate) fn new(
        crop: NikonCrop,
        full_width: usize,
        full_height: usize,
        samples: Vec<u16>,
    ) -> Result<Self> {
        let expected = full_width
            .checked_mul(full_height)
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Bayer frame overflows"))?;
        if samples.len() != expected {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Bayer frame length does not match its dimensions",
            ));
        }
        crop.validate(full_width, full_height)?;
        Ok(Self {
            crop,
            full_width,
            full_height,
            samples,
        })
    }

    pub fn crop(&self) -> NikonCrop {
        self.crop
    }

    pub fn width(&self) -> usize {
        self.crop.width
    }

    pub fn height(&self) -> usize {
        self.crop.height
    }

    pub fn cfa_phase(&self) -> (usize, usize) {
        self.crop.cfa_phase()
    }

    pub fn full_width(&self) -> usize {
        self.full_width
    }

    pub fn full_height(&self) -> usize {
        self.full_height
    }

    pub fn full_samples(&self) -> &[u16] {
        &self.samples
    }

    pub fn row(&self, y: usize) -> Option<&[u16]> {
        if y >= self.crop.height {
            return None;
        }
        let row = self.crop.top.checked_add(y)?;
        let start = row
            .checked_mul(self.full_width)?
            .checked_add(self.crop.left)?;
        let end = start.checked_add(self.crop.width)?;
        self.samples.get(start..end)
    }

    pub fn rows(&self) -> impl Iterator<Item = &[u16]> {
        (0..self.crop.height).filter_map(|y| self.row(y))
    }

    pub fn sample(&self, x: usize, y: usize) -> Option<u16> {
        self.row(y)?.get(x).copied()
    }
}

/// Decodes the verified full frame, then exposes only the MakerNote 0x0045 inset.
/// A missing/unsupported crop returns `None`; it never falls back to the full frame.
pub fn decode_frame_cropped(nef: &[u8]) -> Result<Option<NikonHeCropFrame>> {
    let Some(crop) = nikon_crop(nef)? else {
        return Ok(None);
    };
    let Some((full_width, samples)) = decode_frame(nef)? else {
        return Ok(None);
    };
    if full_width == 0 || samples.len() % full_width != 0 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Bayer frame has an invalid width",
        ));
    }
    let full_height = samples.len() / full_width;
    NikonHeCropFrame::new(crop, full_width, full_height, samples).map(Some)
}

/// C1v2012 scalar `_1` and non-CA basic `_2`, on an interior rectangle in full-frame coordinates.
/// The caller selects a recipe and provides its parameters; output is sparse R/B, not demosaiced RGB.
pub fn reconstruct_basic_roi(
    frame: &NikonHeCropFrame,
    area: NikonCrop,
    params: Stage1Params,
    rgb_black: Option<i32>,
) -> Result<Vec<[f32; 3]>> {
    let invalid_halo = || Error::new(ErrorKind::InvalidInput, "stage 1/2 ROI requires seven real Bayer halo pixels");
    area.validate(frame.full_width, frame.full_height)?;
    let right = area.left + area.width;
    let bottom = area.top + area.height;
    if area.left < 7 || area.top < 7 || frame.full_width - right < 7 || frame.full_height - bottom < 7 {
        return Err(invalid_halo());
    }
    if ![params.red_gain, params.blue_gain, params.noise_floor].into_iter().all(|value| value.is_finite() && value > 0.0)
        || !params.black_pivot.is_finite() || params.black_pivot < 0.0 {
        return Err(Error::new(ErrorKind::InvalidInput, "non-finite or invalid stage 1 parameters"));
    }
    let stride = area.width + 4;
    let mut stage1 = Vec::with_capacity(stride * (area.height + 4));
    for y in area.top - 2..bottom + 2 {
        for x in area.left - 2..right + 2 {
            let patch = std::array::from_fn(|row| {
                let start = (y + row - 5) * frame.full_width + x - 5;
                std::array::from_fn(|col| frame.samples[start + col] << 2)
            });
            stage1.push(c1_stage1_green::stage1_rgb(&patch, y % 2 != 0, x % 2 != 0, params, 0.0).0);
        }
    }
    if let Some(black) = rgb_black {
        subtract_black_rgb32f(&mut stage1, black);
    }
    let mut stage2 = Vec::with_capacity(area.width * area.height);
    for row in 0..area.height {
        for col in 0..area.width {
            let y = row + 2;
            let x = col + 2;
            let mut pixel = stage1[y * stride + x];
            let odd_row = (area.top + row) % 2 != 0;
            if odd_row == ((area.left + col) % 2 != 0) {
                pixel[1] = c1_stage2_basic::refine_green_basic(&stage1, stride, x, y, odd_row);
            }
            stage2.push(pixel);
        }
    }
    Ok(stage2)
}

/// Interior non-CA basic demosaic with the planar `_3` branch before optional finishing.
/// Uses full-frame coordinates, an even RGGB origin and thirteen real Bayer halo pixels.
pub fn reconstruct_planar_roi(
    frame: &NikonHeCropFrame,
    area: NikonCrop,
    params: Stage1Params,
    rgb_black: Option<i32>,
    scale: [f32; 3],
    offset: f32,
) -> Result<Vec<[f32; 3]>> {
    reconstruct_planar_roi_with(frame, area, params, rgb_black, scale, offset, None)
}

fn reconstruct_planar_roi_with(
    frame: &NikonHeCropFrame,
    area: NikonCrop,
    params: Stage1Params,
    rgb_black: Option<i32>,
    scale: [f32; 3],
    offset: f32,
    finish: Option<(f32, i32)>,
) -> Result<Vec<[f32; 3]>> {
    let invalid_area = || Error::new(ErrorKind::InvalidInput, "planar ROI requires even RGGB origin and thirteen real Bayer halo pixels");
    area.validate(frame.full_width, frame.full_height)?;
    if !scale.into_iter().all(|value| value.is_finite() && value > 0.0) || !offset.is_finite() || offset <= 0.0 {
        return Err(Error::new(ErrorKind::InvalidInput, "invalid planar color scale or offset"));
    }
    if area.left < 13 || area.top < 13 || (area.left | area.top) & 1 != 0
        || frame.full_width - (area.left + area.width) < 13
        || frame.full_height - (area.top + area.height) < 13 {
        return Err(invalid_area());
    }
    let support = NikonCrop {
        left: area.left - 6,
        top: area.top - 6,
        width: area.width + 12,
        height: area.height + 12,
    };
    let stage2 = reconstruct_basic_roi(frame, support, params, rgb_black)?;
    let output = if let Some((noise, control)) = finish {
        c1_stage3_planar::planar_with_finish(&stage2, support.width, area.width, area.height, scale, offset, noise, control)
    } else {
        c1_stage3_planar::planar_core(&stage2, support.width, area.width, area.height, scale, offset)
    };
    output.ok_or_else(invalid_area)
}

fn mirror_once(coord: i64, size: usize) -> usize {
    let end = size as i64 - 1;
    let reflected = if coord < 0 { -coord } else if coord > end { 2 * end - coord } else { coord };
    reflected.clamp(0, end) as usize
}

pub fn reconstruct_planar_active_basic_roi(
    frame: &NikonHeCropFrame,
    area: NikonCrop,
    stage1: Stage1Params,
    rgb_black: Option<i32>,
    stage3: Z6iiiStage3,
) -> Result<Vec<[f32; 3]>> {
    let finish = stage3.inner_finish();
    if !stage3.supports_bare_planar() && finish.is_none() {
        return Err(Error::new(ErrorKind::Unsupported, "stage 3 uses an unimplemented fallback or finish pass"));
    }
    reconstruct_active_mirror_with(frame, area, stage1, rgb_black, stage3.gains(), stage3.offset(), finish)
}

/// Candidate for the native active-crop Bayer view followed by `Copy<Bayer16>` mode 3.
/// Output rectangles are in full-sensor coordinates and must lie inside the active crop.
pub fn reconstruct_planar_active_mirror_roi(
    frame: &NikonHeCropFrame,
    area: NikonCrop,
    params: Stage1Params,
    rgb_black: Option<i32>,
    scale: [f32; 3],
    offset: f32,
) -> Result<Vec<[f32; 3]>> {
    reconstruct_active_mirror_with(frame, area, params, rgb_black, scale, offset, None)
}

fn reconstruct_active_mirror_with(
    frame: &NikonHeCropFrame,
    area: NikonCrop,
    params: Stage1Params,
    rgb_black: Option<i32>,
    scale: [f32; 3],
    offset: f32,
    finish: Option<(f32, i32)>,
) -> Result<Vec<[f32; 3]>> {
    let crop = frame.crop();
    let invalid_area = || Error::new(ErrorKind::InvalidInput, "planar ROI must lie inside the active RGGB crop");
    let left = area.left.checked_sub(crop.left).ok_or_else(invalid_area)?;
    let top = area.top.checked_sub(crop.top).ok_or_else(invalid_area)?;
    let relative = NikonCrop { left, top, ..area };
    relative.validate(crop.width, crop.height)?;
    if (area.left | area.top) & 1 != 0 {
        return Err(Error::new(ErrorKind::InvalidInput, "planar ROI requires even RGGB origin"));
    }
    if left >= 13 && top >= 13
        && crop.width - (left + area.width) >= 13
        && crop.height - (top + area.height) >= 13
    {
        return reconstruct_planar_roi_with(frame, area, params, rgb_black, scale, offset, finish);
    }
    const HALO: usize = 14;
    let local = NikonCrop { left: HALO, top: HALO, ..area };
    let width = area.width + 2 * HALO;
    let height = area.height + 2 * HALO;
    let x_map: Vec<_> = (0..width)
        .map(|column| mirror_once(left as i64 + column as i64 - HALO as i64, crop.width))
        .collect();
    let mut samples = Vec::with_capacity(width * height);
    for row in 0..height {
        let start = (crop.top + mirror_once(top as i64 + row as i64 - HALO as i64, crop.height)) * frame.full_width + crop.left;
        samples.extend(x_map.iter().map(|column| frame.samples[start + column]));
    }
    let tile = NikonHeCropFrame::new(local, width, height, samples)?;
    reconstruct_planar_roi_with(&tile, local, params, rgb_black, scale, offset, finish)
}

/// One HE slice: a 0x20 marker resets the coefficient predictor, then 16 (or
/// fewer, for the tail) length-prefixed precincts. Each slice is therefore an
/// independent decode unit; only the byte offsets are discovered sequentially.
struct SlicePlan {
    /// Byte offset of each precinct's length field. Decoding is handed the
    /// payload from that offset onward, exactly as the sequential loop does, so
    /// a stream whose length field overruns its slice fails the same way.
    precincts: Vec<usize>,
}

fn coefficient_bands(precinct: nikon_he_bitstream::Precinct) -> Result<[Vec<Vec<i32>>; 8]> {
    let mut packets: [Vec<Vec<i32>>; 8] = std::array::from_fn(|_| Vec::new());
    for (destination, packet) in packets.iter_mut().zip(precinct.packets) {
        for band in packet.bands {
            let mut coefficient_band = Vec::with_capacity(band.groups.len() * 4);
            for group in band.groups {
                for coefficient in group.coeffs {
                    coefficient_band.push(dequantize(
                        coefficient.unsigned_abs(),
                        coefficient < 0,
                        group.nbits,
                        band.gtli,
                    )?);
                }
            }
            destination.push(coefficient_band);
        }
    }
    Ok(packets)
}

/// Walks the slice headers and precinct length fields without touching
/// coefficients. `None` means the payload is structurally irregular, and the
/// caller must use the sequential path so error selection stays identical.
fn plan_slices(payload: &[u8], precinct_count: usize) -> Option<Vec<SlicePlan>> {
    let mut plans: Vec<SlicePlan> = Vec::new();
    let mut at = 0usize;
    for index in 0..precinct_count {
        if index % 16 == 0 {
            let header = payload.get(at..at + 4)?;
            if header[..2] != [0xff, 0x20] {
                return None;
            }
            let length = usize::from(u16::from_be_bytes([header[2], header[3]]));
            if length < 2 || payload.len() < at + length + 2 {
                return None;
            }
            let segment = &payload[at..at + length + 2];
            if segment[4..] != ((index / 16) as u16).to_be_bytes() {
                return None;
            }
            at += length + 2;
            plans.push(SlicePlan { precincts: Vec::with_capacity(16) });
        }
        let header = payload.get(at..at + 12)?;
        let bytes_consumed = 12
            + (usize::from(header[0]) << 16)
            + (usize::from(header[1]) << 8)
            + usize::from(header[2]);
        if payload.len() < at.checked_add(bytes_consumed)? {
            return None;
        }
        plans.last_mut()?.precincts.push(at);
        at += bytes_consumed;
    }
    (at == payload.len()).then_some(plans)
}

fn decode_slice(
    decoder: &mut Decoder,
    payload: &[u8],
    plan: &SlicePlan,
    cell_width: usize,
) -> Result<Vec<c1_nikon_he::Precinct>> {
    decoder.reset_slice();
    let mut precincts = Vec::with_capacity(plan.precincts.len());
    for &start in &plan.precincts {
        let precinct = decoder.decode_precinct(&payload[start..])?;
        let packets = coefficient_bands(precinct)?;
        precincts.push(reconstruct_precinct(cell_width, &packets)?);
    }
    Ok(precincts)
}

/// Bounded worker count. One slice is one work item, so the pool never exceeds
/// the slice count and never assumes a particular machine.
fn worker_count(slices: usize) -> usize {
    let requested = std::env::var("AURORA_HE_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value >= 1)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .unwrap_or(1)
                .min(8)
        });
    requested.clamp(1, slices.max(1))
}

fn decode_slices_parallel(
    payload: &[u8],
    plans: &[SlicePlan],
    weights: Weights,
    cell_width: usize,
) -> Result<Vec<c1_nikon_he::Precinct>> {
    let workers = worker_count(plans.len());
    if workers <= 1 {
        let mut decoder = Decoder::new(weights);
        let mut precincts = Vec::new();
        for plan in plans {
            precincts.extend(decode_slice(&mut decoder, payload, plan, cell_width)?);
        }
        return Ok(precincts);
    }
    let mut collected: Vec<Option<Result<Vec<c1_nikon_he::Precinct>>>> =
        (0..plans.len()).map(|_| None).collect();
    let (sender, receiver) = mpsc::channel();
    let chunk = plans.len().div_ceil(workers);
    std::thread::scope(|scope| {
        for (worker, group) in plans.chunks(chunk).enumerate() {
            let sender = sender.clone();
            let first = worker * chunk;
            scope.spawn(move || {
                let mut decoder = Decoder::new(weights);
                for (offset, plan) in group.iter().enumerate() {
                    let result = decode_slice(&mut decoder, payload, plan, cell_width);
                    if sender.send((first + offset, result)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(sender);
        for (index, result) in receiver {
            collected[index] = Some(result);
        }
    });
    let mut precincts = Vec::with_capacity(plans.iter().map(|plan| plan.precincts.len()).sum());
    for slot in collected {
        match slot {
            Some(Ok(mut slice)) => precincts.append(&mut slice),
            Some(Err(error)) => return Err(error),
            None => {
                return Err(Error::new(
                    ErrorKind::Other,
                    "HE slice worker did not report a result",
                ));
            }
        }
    }
    Ok(precincts)
}

fn decode_precincts_sequential(
    rest: &mut &[u8],
    precinct_count: usize,
    weights: Weights,
    cell_width: usize,
) -> Result<Vec<c1_nikon_he::Precinct>> {
    let mut decoder = Decoder::new(weights);
    let mut precincts = Vec::with_capacity(precinct_count);
    for index in 0..precinct_count {
        if index % 16 == 0 {
            let slice = marker(rest, 0x20)?;
            if slice[4..] != ((index / 16) as u16).to_be_bytes() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "unexpected HE slice index",
                ));
            }
            decoder.reset_slice();
        }
        let precinct = decoder.decode_precinct(rest)?;
        *rest = &rest[precinct.bytes_consumed..];
        let packets = coefficient_bands(precinct)?;
        precincts.push(reconstruct_precinct(cell_width, &packets)?);
    }
    Ok(precincts)
}

fn decode(nef: &[u8], first_slice_only: bool) -> Result<Option<(usize, Vec<u16>)>> {
    let Some(strip) = find_he_strip(nef)? else {
        return Ok(None);
    };
    if (strip.width, strip.height) != (6064, 4040) {
        return Err(Error::new(ErrorKind::Unsupported, "untested HE geometry"));
    }
    let mut rest = &strip.bytes[2..strip.bytes.len() - 2];
    let capabilities = marker(&mut rest, 0x50)?;
    let picture = marker(&mut rest, 0x12)?;
    let components = marker(&mut rest, 0x13)?;
    let weights = marker(&mut rest, 0x14)?;
    if capabilities.get(4..22) != Some(b"CONTACT_INTOPIX_\xef\xc0")
        || picture.len() != 41
        || u32::from_be_bytes(picture[4..8].try_into().unwrap()) as usize != strip.bytes.len()
        || picture[8..16] != [0, 0, 0, 0, 0x17, 0xb0, 0x0f, 0xc8]
        || picture[16..] != PIH_PROFILE
        || components[4..] != CDT_PROFILE
    {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "untested HE codestream profile",
        ));
    }
    let weights = Weights::from_wgt(weights)?;
    let precinct_count = if first_slice_only {
        17
    } else {
        strip.height / 4
    };
    let cell_width = strip.width / 2;
    let precincts = match plan_slices(rest, precinct_count) {
        Some(plans) => {
            let precincts = decode_slices_parallel(rest, &plans, weights, cell_width)?;
            rest = &rest[rest.len()..];
            precincts
        }
        None => decode_precincts_sequential(&mut rest, precinct_count, weights, cell_width)?,
    };
    let bayer = if first_slice_only {
        bayer_prefix(&precincts, 64)?
    } else {
        if !rest.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "extra HE payload after final precinct",
            ));
        }
        bayer_frame(&precincts, strip.height)?
    };
    Ok(Some((strip.width, bayer)))
}

