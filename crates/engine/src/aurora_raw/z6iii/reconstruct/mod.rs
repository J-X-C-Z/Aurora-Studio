//! Nikon Z6III HE Bayer decoding and reconstruction.
//!
//! The decode and CPU reconstruction are platform-neutral. The System32
//! OpenCL loader and the GPU-only entry points below keep a Windows gate.

#![allow(dead_code)]

#[path = "c1_nef_he.rs"]
mod c1_nef_he;
#[path = "c1_adaptive_gg.rs"]
mod c1_adaptive_gg;
#[cfg(windows)]
#[path = "c1_adaptive_opencl.rs"]
mod c1_adaptive_opencl;
#[cfg(windows)]
#[path = "c1_cuda_pipeline.rs"]
mod c1_cuda_pipeline;
#[cfg(windows)]
#[allow(unused_imports)]
pub(crate) use c1_adaptive_opencl::anti_alias as anti_alias_opencl;
#[cfg(windows)]
pub(super) use c1_adaptive_opencl::{export_color, highlight_and_aa, recover_highlights, stage3_gg_highlight_aa};
#[cfg(windows)]
pub(super) use c1_cuda_pipeline::{
    available as cuda_available, enabled as cuda_enabled, reconstruct_rgba16 as cuda_reconstruct_rgba16,
    CudaPipelineInput,
};
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
#[path = "stage3/mod.rs"]
mod c1_stage3_planar;
#[path = "nikon_he_bitstream.rs"]
#[allow(dead_code)]
mod nikon_he_bitstream;
#[cfg(test)]
#[path = "stage1_reference.rs"]
#[allow(dead_code)]
mod stage1_reference;

use std::io::{Error, ErrorKind, Result};

pub(super) use c1_nef_he::find_he_strip;
use c1_nef_he::NikonHeStrip;
#[cfg(windows)]
pub(super) use c1_adaptive_opencl::available as gpu_available;
#[allow(unused_imports)]
pub use c1_nef_he::{NikonBlackLevels, NikonCrop, nikon_as_shot_rgb, nikon_black_levels, nikon_crop, nikon_exif_sensitivity};
#[allow(unused_imports)]
pub use c1_nef_he::is_z6iii_nef;
pub use c1_stage1_params::Stage1Params;
pub use c1_stage3_params::Z6iiiStage3;

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
    // Reconstruction-only u16 code plane. Public accessors always expose the decode.
    adaptive_bayer: Option<Vec<u16>>,
    stage1_plane: Option<(Stage1Params, Vec<[f32; 4]>)>,
}

impl NikonHeCropFrame {
    pub(in crate::aurora_raw) fn new(
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
            adaptive_bayer: None,
            stage1_plane: None,
        })
    }

    pub fn crop(&self) -> NikonCrop {
        self.crop
    }

    /// Prepare the full active crop once, before tile reconstruction. A zero threshold bypasses GG.
    pub fn with_adaptive_gg_variant2(mut self, threshold: f32, black: f32) -> Result<Self> {
        if !(0.0..=65535.0).contains(&threshold) || !(0.0..=65535.0).contains(&black) {
            return Err(Error::new(ErrorKind::InvalidInput, "invalid AdaptiveGG threshold or black level"));
        }
        self.adaptive_bayer = (threshold > 0.0).then(|| c1_adaptive_gg::prepare(
            &self.samples, self.full_width, self.full_height, self.crop, threshold, black,
        ));
        Ok(self)
    }

    /// GPU AdaptiveGG plus a mirrored halo; the reconstructed halo is the
    /// Windows OpenCL product route.
    #[cfg(windows)]
    pub fn with_control_bayer_halo(self, threshold: f32, black: f32) -> Result<Self> {
        const HALO: usize = 20;
        let active = self.crop;
        if active.width < 2 * HALO || active.width % 2 != 0 || active.height < HALO
            || active.cfa_phase() != (0, 0)
            || !(0.0..=65535.0).contains(&threshold) || !(0.0..=65535.0).contains(&black)
        {
            return Err(Error::new(ErrorKind::InvalidInput, "invalid active Bayer halo input"));
        }
        let crop = NikonCrop { left: HALO, top: HALO, ..active };
        let width = active.width + 2 * HALO;
        let height = active.height + 2 * HALO;
        let columns: Vec<_> = (0..width).map(|x| {
            active.left + mirror_bayer_pairs(x as i64 - HALO as i64, active.width)
        }).collect();
        let mut samples = Vec::with_capacity(width * height);
        for row in 0..height {
            let source_y = active.top + mirror_once(row as i64 - HALO as i64, active.height);
            samples.extend(columns.iter().map(|&x| self.samples[source_y * self.full_width + x]));
        }
        let mut frame = Self::new(crop, width, height, samples)?;
        let adaptive_roi = NikonCrop { left: 2, top: 2, width: width - 4, height: height - 4 };
        if threshold > 0.0 {
            frame.adaptive_bayer = Some(c1_adaptive_opencl::prepare(
                &frame.samples, width, height, adaptive_roi, threshold, black)?);
        }
        Ok(frame)
    }

    pub fn width(&self) -> usize {
        self.crop.width
    }

    #[cfg(windows)]
    pub fn with_stage1_opencl(mut self, params: Stage1Params) -> Result<Self> {
        let input = self.adaptive_bayer.as_deref().ok_or_else(||
            Error::new(ErrorKind::InvalidInput, "Stage1 OpenCL requires the prepared Bayer plane"))?;
        let stage1 = c1_adaptive_opencl::stage1(input, self.full_width, self.full_height, params)?;
        self.stage1_plane = Some((params, stage1));
        Ok(self)
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
    let cached = frame.stage1_plane.as_ref().filter(|(cached_params, _)| *cached_params == params
        && area.left >= 10 && area.top >= 10 && right + 10 <= frame.full_width && bottom + 10 <= frame.full_height);
    if let Some((_, cached_plane)) = cached {
        use rayon::prelude::*;
        let black_offset = rgb_black.map_or(0.0f32, |b| -(b as f32));
        let stage1_h = area.height + 4;
        let mut stage1 = vec![[0.0f32; 3]; stride * stage1_h];
        let full_w = frame.full_width;
        let left0 = area.left - 2;
        let top0 = area.top - 2;
        stage1
            .par_chunks_exact_mut(stride)
            .enumerate()
            .for_each(|(row_idx, dst_row)| {
                let src_row = &cached_plane[(top0 + row_idx) * full_w + left0..][..stride];
                for (dst, &[r, g, b, _]) in dst_row.iter_mut().zip(src_row) {
                    *dst = [r + black_offset, g + black_offset, b + black_offset];
                }
            });
        let mut stage2 = vec![[0.0f32; 3]; area.width * area.height];
        stage2
            .par_chunks_exact_mut(area.width)
            .enumerate()
            .for_each(|(row, dst_row)| {
                let y = row + 2;
                let odd_row = (area.top + row) % 2 != 0;
                for (col, dst) in dst_row.iter_mut().enumerate() {
                    let x = col + 2;
                    let mut pixel = stage1[y * stride + x];
                    if odd_row == ((area.left + col) % 2 != 0) {
                        pixel[1] = c1_stage2_basic::refine_green_basic(&stage1, stride, x, y, odd_row);
                    }
                    *dst = pixel;
                }
            });
        return Ok(stage2);
    }
    let mut stage1 = Vec::with_capacity(stride * (area.height + 4));
    let (plane, shift) = match &frame.adaptive_bayer {
        Some(filtered) => (filtered.as_slice(), 0),
        None => (frame.samples.as_slice(), 2),
    };
    for y in area.top - 2..bottom + 2 {
        for x in area.left - 2..right + 2 {
            let patch = std::array::from_fn(|row| {
                let start = (y + row - 5) * frame.full_width + x - 5;
                std::array::from_fn(|col| plane[start + col] << shift)
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

/// Parallel Stage 2 scaled directly into `[f32; 4]` for the fused GPU Stage3+GG+HL+AA strip path.
#[cfg(windows)]
pub(super) fn reconstruct_stage2_scaled_f4(
    frame: &NikonHeCropFrame,
    area: NikonCrop,
    params: Stage1Params,
    rgb_black: i32,
    scale: [f32; 3],
) -> Result<Vec<[f32; 4]>> {
    use rayon::prelude::*;
    let support = NikonCrop {
        left: area.left - 6,
        top: area.top - 6,
        width: area.width + 12,
        height: area.height + 12,
    };
    let stage2 = reconstruct_basic_roi(frame, support, params, Some(rgb_black))?;
    Ok(stage2
        .into_par_iter()
        .map(|p| [p[0] * scale[0], p[1] * scale[1], p[2] * scale[2], 0.0])
        .collect())
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
    finish: Option<c1_stage3_planar::Finish>,
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
    // Windows product route: the OpenCL stage 3 over even support widths, then
    // the same CPU finish. Elsewhere the CPU planar reconstruction below is the
    // platform-neutral equivalent.
    #[cfg(windows)]
    {
        if support.width.is_multiple_of(2) && !matches!(finish, Some(c1_stage3_planar::Finish::Diagnostic { .. })) {
            if let Some(c1_stage3_planar::Finish::GgBlurComp { noise }) = finish {
                return c1_adaptive_opencl::stage3_gg(&stage2, support.width, support.height, scale, offset, noise);
            }
            let reconstructed = c1_adaptive_opencl::stage3(&stage2, support.width, support.height, scale, offset)?;
            let planes = std::array::from_fn(|channel| reconstructed.iter().map(|pixel| pixel[channel]).collect());
            return Ok(c1_stage3_planar::finish_planar(planes, support.width, area.width, area.height, scale, offset, finish));
        }
    }
    let output = c1_stage3_planar::planar(&stage2, support.width, area.width, area.height, scale, offset, finish);
    output.ok_or_else(invalid_area)
}

fn mirror_once(coord: i64, size: usize) -> usize {
    let end = size as i64 - 1;
    let reflected = if coord < 0 { -coord } else if coord > end { 2 * end - coord } else { coord };
    reflected.clamp(0, end) as usize
}

fn mirror_bayer_pairs(coord: i64, size: usize) -> usize {
    let pair = coord.div_euclid(2);
    let pair = if pair < 0 { -pair } else if pair >= size as i64 / 2 { size as i64 - 1 - pair } else { pair };
    (2 * pair + coord.rem_euclid(2)) as usize
}

#[cfg(test)]
#[test]
fn control_bayer_halo_keeps_horizontal_phase() {
    assert_eq!((-4..2).map(|x| mirror_bayer_pairs(x, 48)).collect::<Vec<_>>(), [4, 5, 2, 3, 0, 1]);
    assert_eq!((46..52).map(|x| mirror_bayer_pairs(x, 48)).collect::<Vec<_>>(), [46, 47, 46, 47, 44, 45]);
}

pub fn reconstruct_planar_active_basic_roi(
    frame: &NikonHeCropFrame,
    area: NikonCrop,
    stage1: Stage1Params,
    rgb_black: Option<i32>,
    stage3: Z6iiiStage3,
) -> Result<Vec<[f32; 3]>> {
    let finish = stage3.inner_finish().map(|(noise, control)| c1_stage3_planar::Finish::Diagnostic { noise, control });
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

/// Ordinary Z6III stage 3 finishes its reconstructed planes before cropping the tile core.
pub fn reconstruct_planar_green_roi(
    frame: &NikonHeCropFrame, area: NikonCrop, stage1: Stage1Params,
    rgb_black: i32, gains: [f32; 3], threshold: f32, noise: f32,
) -> Result<Vec<[f32; 3]>> {
    reconstruct_active_mirror_with(frame, area, stage1, Some(rgb_black), gains, threshold,
        Some(c1_stage3_planar::Finish::GgBlurComp { noise }))
}

pub fn reconstruct_planar_green_halo_roi(
    frame: &NikonHeCropFrame, area: NikonCrop, stage1: Stage1Params,
    rgb_black: i32, gains: [f32; 3], threshold: f32, noise: f32,
) -> Result<Vec<[f32; 3]>> {
    reconstruct_planar_roi_with(frame, area, stage1, Some(rgb_black), gains, threshold,
        Some(c1_stage3_planar::Finish::GgBlurComp { noise }))
}

fn reconstruct_active_mirror_with(
    frame: &NikonHeCropFrame,
    area: NikonCrop,
    params: Stage1Params,
    rgb_black: Option<i32>,
    scale: [f32; 3],
    offset: f32,
    finish: Option<c1_stage3_planar::Finish>,
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
    let mirror_plane = |plane: &[u16]| {
        let mut samples = Vec::with_capacity(width * height);
        for row in 0..height {
            let start = (crop.top + mirror_once(top as i64 + row as i64 - HALO as i64, crop.height)) * frame.full_width + crop.left;
            samples.extend(x_map.iter().map(|column| plane[start + column]));
        }
        samples
    };
    let mut tile = NikonHeCropFrame::new(local, width, height, mirror_plane(&frame.samples))?;
    tile.adaptive_bayer = frame.adaptive_bayer.as_deref().map(mirror_plane);
    reconstruct_planar_roi_with(&tile, local, params, rgb_black, scale, offset, finish)
}

/// Validates the HE codestream header profile without decoding coefficients.
///
/// Returns the payload after the 0x14 marker and the weights marker. Every
/// caller that accepts a file for HE reconstruction goes through this, so the
/// cheap [`is_verified_he_nef`] gate and the real decode cannot disagree.
fn verified_profile<'a>(strip: &NikonHeStrip<'a>) -> Result<(&'a [u8], &'a [u8])> {
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
    Ok((rest, weights))
}

/// Whether `nef` is the HE profile [`decode_frame`] accepts, read from TIFF and
/// codestream headers only.
///
/// This is the pre-unpack backend gate: the verified profile is decoded from
/// the codestream itself without a LibRaw Bayer plane, so opening LibRaw first
/// would only pay for an unpack the other path does not use. A header mismatch
/// returns `false` and the caller keeps the LibRaw-first order.
pub(crate) fn is_verified_he_nef(nef: &[u8]) -> bool {
    match find_he_strip(nef) {
        Ok(Some(strip)) => verified_profile(&strip).is_ok(),
        _ => false,
    }
}

fn decode(nef: &[u8], first_slice_only: bool) -> Result<Option<(usize, Vec<u16>)>> {
    let Some(strip) = find_he_strip(nef)? else {
        return Ok(None);
    };
    let (rest, weights) = verified_profile(&strip)?;
    let bayer = c1_nikon_he::decode_he_bayer_parallel(
        rest,
        weights,
        strip.width,
        strip.height,
        first_slice_only,
    )?;
    Ok(Some((strip.width, bayer)))
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires an NVIDIA OpenCL GPU"]
    fn mirror_roi_preserves_phase_across_border_tiles() {
        let crop = NikonCrop { left: 8, top: 4, width: 40, height: 32 };
        let width = crop.width + 16;
        let height = crop.height + 8;
        let samples = (0..width * height).map(|index| (1200 + (index * 311) % 15000) as u16).collect();
        let frame = NikonHeCropFrame::new(crop, width, height, samples).unwrap();
        let params = Stage1Params::prepare([1.75, 1.0, 1.45], [12.8, 0.0], 4032, false);
        let reconstruct = |area| reconstruct_planar_active_mirror_roi(&frame, area, params, Some(4032), [1.25, 0.75, 1.5], 128.0).unwrap();
        let whole = reconstruct(crop);
        for y in (0..crop.height).step_by(8) {
            for x in (0..crop.width).step_by(8) {
                let tile = reconstruct(NikonCrop { left: crop.left + x, top: crop.top + y, width: 8, height: 8 });
                for row in 0..8 {
                    let start = (y + row) * crop.width + x;
                    assert!(whole[start..start+8].iter().flatten().zip(tile[row*8..(row+1)*8].iter().flatten())
                        .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
                }
            }
        }
        assert_eq!((0..9).map(|i| mirror_once(i - 3, 3)).collect::<Vec<_>>(), [2, 2, 1, 0, 1, 2, 1, 0, 0]);
    }

    #[test]
    #[ignore = "requires an NVIDIA OpenCL GPU"]
    fn active_mirror_roi_delegates_interior_and_rejects_invalid_areas() {
        let crop = NikonCrop { left: 8, top: 4, width: 64, height: 64 };
        let full_width = 80;
        let full_height = 72;
        let samples = (0..full_width * full_height)
            .map(|index| (1200 + (index * 311) % 15000) as u16)
            .collect();
        let frame = NikonHeCropFrame::new(crop, full_width, full_height, samples).unwrap();
        let params = Stage1Params::prepare([1.75, 1.0, 1.45], [12.8, 0.0], 4032, false);
        let scale = [1.25, 0.75, 1.5];

        let area = NikonCrop { left: crop.left + 16, top: crop.top + 16, width: 16, height: 16 };
        assert!(area.left - crop.left >= 13 && area.top - crop.top >= 13);
        assert!(crop.width - (area.left - crop.left + area.width) >= 13);
        assert!(crop.height - (area.top - crop.top + area.height) >= 13);
        assert_eq!(
            reconstruct_planar_active_mirror_roi(
                &frame,
                area,
                params,
                Some(4032),
                scale,
                128.0,
            )
            .unwrap(),
            reconstruct_planar_roi(&frame, area, params, Some(4032), scale, 128.0).unwrap(),
        );

        let left_of_crop = NikonCrop { left: crop.left - 1, ..area };
        assert_eq!(
            reconstruct_planar_active_mirror_roi(
                &frame,
                left_of_crop,
                params,
                Some(4032),
                scale,
                128.0,
            )
            .unwrap_err()
            .kind(),
            ErrorKind::InvalidInput,
        );

        let beyond_crop = NikonCrop {
            left: crop.left + crop.width - 4,
            ..area
        };
        assert_eq!(
            reconstruct_planar_active_mirror_roi(
                &frame,
                beyond_crop,
                params,
                Some(4032),
                scale,
                128.0,
            )
            .unwrap_err()
            .kind(),
            ErrorKind::InvalidData,
        );

        let odd_origin = NikonCrop { left: crop.left + 15, top: crop.top + 16, width: 8, height: 8 };
        assert_eq!(
            reconstruct_planar_active_mirror_roi(
                &frame,
                odd_origin,
                params,
                Some(4032),
                scale,
                128.0,
            )
            .unwrap_err()
            .kind(),
            ErrorKind::InvalidInput,
        );
    }

    #[test]
    #[ignore = "requires real HE NEFs and renders the active crop in bounded tiles"]
    fn original_bytes_to_active_crop_mirror_rgb() {
        for stem in ["_DSC4133", "_DSC4136", "_DSC5673"] {
            let nef = std::fs::read(std::env::var_os(stem).unwrap()).unwrap();
            let frame = decode_frame_cropped(&nef).unwrap().unwrap();
            let wb = nikon_as_shot_rgb(&nef).unwrap().unwrap();
            let black = nikon_black_levels(&nef).unwrap().unwrap().capture_one_scalar();
            let params = Stage1Params::prepare(wb, [12.8, 0.0], black, false);
            let stage3 = Z6iiiStage3::prepare(wb.map(|channel| channel / wb[1]), [12.8, 0.0],
                (40, 1.5), 0x514, 0, 0x7f, 0).unwrap();
            let crop = frame.crop();
            assert_eq!((crop.left, crop.top, crop.width, crop.height), (8, 4, 6048, 4032));
            let mut pixels = 0;
            let mut sum = [0.0f64; 3];
            for y in (0..crop.height).step_by(256) {
                for x in (0..crop.width).step_by(256) {
                    let area = NikonCrop {
                        left: crop.left+x, top: crop.top+y,
                        width: (crop.width-x).min(256), height: (crop.height-y).min(256),
                    };
                    let rgb = reconstruct_planar_active_basic_roi(&frame, area, params, Some(black as i32), stage3).unwrap();
                    assert_eq!(rgb.len(), area.width * area.height);
                    pixels += rgb.len();
                    for pixel in rgb {
                        assert!(pixel.iter().all(|value| value.is_finite()));
                        for channel in 0..3 {
                            sum[channel] += f64::from(pixel[channel]);
                        }
                    }
                }
            }
            assert_eq!(pixels, crop.width * crop.height);
            println!("{stem}: {pixels} finite candidate RGB pixels, channel sums {sum:?}");
        }
    }

    #[test]
    #[ignore = "requires an NVIDIA OpenCL GPU"]
    fn planar_roi_requires_real_halo_and_preserves_even_tile_phase() {
        let size = 54;
        let full = NikonCrop { left: 0, top: 0, width: size, height: size };
        let samples = (0..size * size).map(|index| (1200 + (index * 311) % 15000) as u16).collect();
        let frame = NikonHeCropFrame::new(full, size, size, samples).unwrap();
        let params = Stage1Params::prepare([1.75, 1.0, 1.45], [12.8, 0.0], 4032, false);
        let reconstruct = |area| reconstruct_planar_roi(&frame, area, params, Some(4032), [1.25, 0.75, 1.5], 128.0);
        let area = NikonCrop { left: 14, top: 14, width: 24, height: 24 };
        let whole = reconstruct(area).unwrap();
        for (left, top) in [(0, 0), (12, 0), (0, 12), (12, 12)] {
            let tile = reconstruct(NikonCrop { left: area.left + left, top: area.top + top, width: 12, height: 12 }).unwrap();
            for row in 0..12 {
                let start = (top + row) * area.width + left;
                assert_eq!(&whole[start..start + 12], &tile[row * 12..(row + 1) * 12]);
            }
        }
        assert_eq!(reconstruct(NikonCrop { top: 4, ..area }).unwrap_err().kind(), ErrorKind::InvalidInput);
        assert_eq!(reconstruct(NikonCrop { left: 15, ..area }).unwrap_err().kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn rgb32f_black_offset_preserves_signed_samples() {
        let mut row = [[0.0, 4031.75, 4032.0], [4032.25, 65535.0, 72000.0]];
        super::subtract_black_rgb32f(&mut row, 4032);
        assert_eq!(row, [[-4032.0, -0.25, 0.0], [0.25, 61503.0, 67968.0]]);
    }

    use super::*;

    #[test]
    #[ignore = "requires original NEFs and LibRaw Bayer fixtures"]
    fn original_bytes_to_bayer() {
        let fixture_dir = std::path::PathBuf::from(std::env::var_os("AURORA_HE_FIXTURES").unwrap());
        for stem in ["_DSC4133", "_DSC4136", "_DSC5673"] {
            let nef = std::fs::read(std::env::var_os(stem).unwrap()).unwrap();
            let levels = nikon_black_levels(&nef).unwrap().unwrap();
            assert_eq!(levels, NikonBlackLevels([1008; 4]), "{stem}");
            assert_eq!(levels.capture_one_scalar(), 4032, "{stem}");
            let (width, samples) = decode_first_slice(&nef).unwrap().unwrap();
            assert_eq!(width, 6064);
            let oracle = std::fs::read(fixture_dir.join(format!("{stem}.bayer.u16le"))).unwrap();
            assert_eq!(samples.len() * 2, oracle.len());
            for (index, (actual, expected)) in
                samples.iter().zip(oracle.chunks_exact(2)).enumerate()
            {
                assert_eq!(actual.to_le_bytes(), expected, "{stem} sample {index}");
            }
            println!(
                "{stem}: original NEF -> {} identical Bayer samples",
                samples.len()
            );
        }
    }

    #[test]
    #[ignore = "requires original NEFs and full-size LibRaw Bayer fixtures"]
    fn original_bytes_to_full_bayer() {
        let fixture_dir = std::path::PathBuf::from(std::env::var_os("AURORA_HE_FIXTURES").unwrap());
        for (stem, expected_gains, expected_predictions) in [
            ("_DSC4133", [1968, 1316], [14909, 11696]),
            ("_DSC4136", [1968, 1316], [39663, 28583]),
            ("_DSC5673", [1664, 1446], [4038, 4003]),
        ] {
            let nef = std::fs::read(std::env::var_os(stem).unwrap()).unwrap();
            let (width, samples) = decode_frame_left_aligned(&nef).unwrap().unwrap();
            assert_eq!(width, 6064);
            assert_eq!(samples.len(), 6064 * 4040);
            let oracle =
                std::fs::read(fixture_dir.join(format!("{stem}-full.bayer.u16le"))).unwrap();
            assert_eq!(samples.len() * 2, oracle.len());
            let mut mismatches = 0;
            let mut first_mismatch = None;
            let mut max_error = 0;
            for (index, (&actual, expected)) in
                samples.iter().zip(oracle.chunks_exact(2)).enumerate()
            {
                let expected = u16::from_le_bytes(expected.try_into().unwrap()) << 2;
                if actual != expected {
                    mismatches += 1;
                    max_error = max_error.max(actual.abs_diff(expected));
                    first_mismatch.get_or_insert((index / width, index % width, actual, expected));
                }
            }
            println!(
                "{stem}: {} left-aligned samples, {mismatches} mismatches, max diff {max_error}, first {first_mismatch:?}",
                samples.len()
            );
            assert_eq!(mismatches, 0, "{stem}");

            let [red, green, blue] = nikon_as_shot_rgb(&nef).unwrap().unwrap();
            let gains = [red, blue].map(|value| (value / green * 1024.0) as i64);
            assert_eq!(gains, expected_gains, "{stem}");
            let base = i64::from(nikon_black_levels(&nef).unwrap().unwrap().capture_one_scalar());
            let predictions = [66, 67].map(|position| {
                let center = i64::from(samples[position * width + position]);
                stage1_reference::pred(base, center, gains[position - 66])
            });
            assert_eq!(predictions, expected_predictions, "{stem}");
            println!("{stem}: as-shot Q10 {gains:?}, pivot {base}, CPU prediction {predictions:?}");
        }
    }

    #[test]
    #[ignore = "requires original NEFs"]
    fn original_bytes_crop_view() {
        for stem in ["_DSC4133", "_DSC4136", "_DSC5673"] {
            let nef = std::fs::read(std::env::var_os(stem).unwrap()).unwrap();
            let crop = nikon_crop(&nef).unwrap().unwrap();
            assert_eq!(
                crop,
                NikonCrop {
                    left: 8,
                    top: 4,
                    width: 6048,
                    height: 4032,
                },
                "{stem}"
            );
            assert_eq!(crop.cfa_phase(), (0, 0), "{stem}");

            let frame = decode_frame_cropped(&nef).unwrap().unwrap();
            assert_eq!(frame.crop(), crop, "{stem}");
            assert_eq!((frame.width(), frame.height()), (6048, 4032), "{stem}");
            assert_eq!(
                (frame.full_width(), frame.full_height()),
                (6064, 4040),
                "{stem}"
            );
            assert_eq!(frame.row(4032), None, "{stem}");

            let full = frame.full_samples();
            for y in [0, 1, 2016, 4031] {
                let row = frame.row(y).unwrap();
                let start = (4 + y) * 6064 + 8;
                assert_eq!(row, &full[start..start + 6048], "{stem} row {y}");
            }
            assert_eq!(frame.sample(0, 0), Some(full[4 * 6064 + 8]), "{stem}");
            assert_eq!(
                frame.sample(6047, 4031),
                Some(full[(4 + 4031) * 6064 + 8 + 6047]),
                "{stem}"
            );
            println!(
                "{stem}: crop 6048x4032 at (8,4), phase (0,0), rows match full frame"
            );
        }
    }

    #[test]
    fn cropped_api_does_not_fall_back_to_full_frame() {
        assert!(decode_frame_cropped(b"other file").unwrap().is_none());
    }
}
