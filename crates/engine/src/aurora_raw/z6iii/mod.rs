//! Neutral Z6III HE export, with a Windows desktop route for compatible recipes.
//!
//! Colour, CPU reconstruction, the 16-bit geometry stage and the linear CPU
//! finish helpers are platform-neutral and compile everywhere. The desktop
//! exporter and the System32 OpenCL pipeline keep their Windows/product gate.

#![cfg_attr(not(all(windows, feature = "z6iii-raw-export")), allow(dead_code, unused_imports))]

mod color;
#[cfg(all(windows, feature = "z6iii-raw-export"))]
pub(crate) mod desktop;
pub(crate) mod geometry16;
mod reconstruct;
mod render;

pub use color::{StageTrace, Z6iiiColorStage};
pub(in crate::aurora_raw) use render::{
    apply_linear_chromatic_aberration, apply_linear_color_noise_2012,
};
pub(crate) use reconstruct::is_verified_he_nef;
#[cfg(windows)]
pub(in crate::aurora_raw) use render::recover_and_antialias_frame;

use anyhow::Result;
#[cfg(windows)]
use anyhow::{Context, ensure};
use std::io::{Seek, Write};

pub struct ExportAssets<'a> {
    pub camera_icc: &'a [u8],
    pub curve: &'a [u8],
    pub linear_curve: &'a [u8],
    pub working_icc: &'a [u8],
    pub adobe_rgb_icc: &'a [u8],
    pub banding_tables: [&'a [u8]; 3],
}

pub struct Frame {
    width: u32,
    height: u32,
    pixels: Vec<u16>,
    icc: Vec<u8>,
}

impl Frame {
    pub fn write_tiff(&self, sink: impl Write + Seek) -> Result<()> {
        self.write_tiff_with_resolution(sink, None)
    }

    pub(crate) fn write_tiff_with_resolution(&self, sink: impl Write + Seek, ppi: Option<u32>) -> Result<()> {
        use tiff::{encoder::{Rational, TiffEncoder, colortype::RGB16}, tags::{ResolutionUnit, Tag}};
        let mut encoder = TiffEncoder::new(sink)?;
        let mut image = encoder.new_image::<RGB16>(self.width, self.height)?;
        image.encoder().write_tag(Tag::IccProfile, self.icc.as_slice())?;
        if let Some(ppi) = ppi {
            image.resolution(ResolutionUnit::Inch, Rational { n: ppi, d: 1 });
        }
        image.write_data(&self.pixels)?;
        Ok(())
    }
}

#[cfg(windows)]
pub fn render(nef: &[u8], assets: &ExportAssets<'_>) -> Result<Frame> {
    ensure!(reconstruct::is_z6iii_nef(nef), "expected a Nikon Z6III NEF");
    let orientation = crate::raw::exif_orientation_bytes(nef).unwrap_or(1);
    ensure!(matches!(orientation, 1 | 3 | 6 | 8), "unsupported NEF orientation {orientation}");
    let mut pixels = render::neutral(nef, assets).context("Z6III HE reconstruction")?;
    let (width, height) = orient(&mut pixels, 6048, 4032, orientation);
    Ok(Frame { width: width as u32, height: height as u32, pixels, icc: assets.adobe_rgb_icc.to_vec() })
}

/// Whether this machine exposes an OpenCL or CUDA device the Z6III pipeline can use.
#[cfg(windows)]
pub(crate) fn opencl_available() -> bool {
    reconstruct::cuda_available() || reconstruct::gpu_available()
}

/// Reconstructs linear camera-space RGBA16 via the Z6III GPU pipeline
/// (native CUDA GPU-resident path with OpenCL fallback: demosaic, linear CA,
/// highlight recovery and anti-aliasing) when a GPU is available.
///
/// The returned route distinguishes "not this file", "no GPU device" and
/// "device present but the reconstruction failed", so callers can record why a
/// frame did or did not come from the GPU pipeline.
#[cfg(windows)]
pub(in crate::aurora_raw) fn render_linear_rgba16_traced(
    nef: &[u8],
    bayer: Option<(usize, usize, Vec<u16>)>,
    ca_warp: Option<&crate::mock::lens_profile::LensWarp>,
) -> (Option<(usize, usize, Vec<u16>)>, super::OpenClRoute) {
    let (frame, route, _oriented) =
        render_linear_rgba16_oriented_traced(nef, bayer, ca_warp, 1);
    (frame, route)
}

/// Same as [`render_linear_rgba16_traced`], additionally allowing the GPU
/// pipeline to fuse EXIF `orientation` directly into the final RGBA16 write.
/// Returns `(frame, route, already_oriented)`.
#[cfg(windows)]
pub(in crate::aurora_raw) fn render_linear_rgba16_oriented_traced(
    nef: &[u8],
    bayer: Option<(usize, usize, Vec<u16>)>,
    ca_warp: Option<&crate::mock::lens_profile::LensWarp>,
    orientation: u16,
) -> (Option<(usize, usize, Vec<u16>)>, super::OpenClRoute, bool) {
    if !reconstruct::is_z6iii_nef(nef) {
        return (None, super::OpenClRoute::NotZ6iii, false);
    }
    if !reconstruct::cuda_available() && !reconstruct::gpu_available() {
        return (None, super::OpenClRoute::Unavailable, false);
    }
    match render::linear_rgba16_oriented(nef, bayer, ca_warp, orientation) {
        Ok((w, h, pixels, oriented)) => (Some((w, h, pixels)), super::OpenClRoute::Succeeded, oriented),
        Err(error) => {
            log::warn!("Z6III GPU linear decode fell back to CPU C1 chain: {error:#}");
            (None, super::OpenClRoute::Failed, false)
        }
    }
}

#[cfg(windows)]
pub(in crate::aurora_raw) fn render_linear_rgba16(
    nef: &[u8],
    bayer: Option<(usize, usize, Vec<u16>)>,
    ca_warp: Option<&crate::mock::lens_profile::LensWarp>,
) -> Option<(usize, usize, Vec<u16>)> {
    render_linear_rgba16_traced(nef, bayer, ca_warp).0
}

/// The active area used by both Z6III RAW reconstruction routes.
pub(crate) fn active_size(path: &std::path::Path) -> Option<(u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let nef = unsafe { memmap2::Mmap::map(&file).ok()? };
    if !reconstruct::is_z6iii_nef(&nef) {
        return None;
    }
    let crop = reconstruct::nikon_crop(&nef).ok()??;
    if (crop.width, crop.height) != (6048, 4032) {
        return None;
    }
    let size = (crop.width as u32, crop.height as u32);
    Some(match crate::raw::exif_orientation_bytes(&nef) {
        Some(6 | 8) => (size.1, size.0),
        _ => size,
    })
}

pub(crate) fn is_z6iii_nef_file(path: &std::path::Path) -> bool {
    let Ok(file) = std::fs::File::open(path) else { return false };
    let Ok(nef) = (unsafe { memmap2::Mmap::map(&file) }) else { return false };
    reconstruct::is_z6iii_nef(&nef)
}

fn orient(pixels: &mut Vec<u16>, width: usize, height: usize, orientation: u16) -> (usize, usize) {
    match orientation {
        3 => {
            for index in 0..width * height / 2 {
                for channel in 0..3 {
                    pixels.swap(index * 3 + channel, (width * height - 1 - index) * 3 + channel);
                }
            }
            (width, height)
        }
        6 | 8 => {
            let mut rotated = vec![0; pixels.len()];
            for (index, pixel) in pixels.chunks_exact(3).enumerate() {
                let (column, row) = (index % width, index / width);
                let dst = if orientation == 6 { column * height + height - 1 - row }
                    else { (width - 1 - column) * height + row };
                rotated[dst * 3..dst * 3 + 3].copy_from_slice(pixel);
            }
            *pixels = rotated;
            (height, width)
        }
        _ => (width, height),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_orientation_moves_whole_rgb16_pixels() {
        for (orientation, expected) in [(3, [6, 5, 4, 3, 2, 1]), (6, [4, 1, 5, 2, 6, 3]), (8, [3, 6, 2, 5, 1, 4])] {
            let mut pixels: Vec<_> = (1..=6).flat_map(|value| [value, value + 10, value + 20]).collect();
            assert_eq!(orient(&mut pixels, 3, 2, orientation), if orientation == 3 { (3, 2) } else { (2, 3) });
            assert!(pixels.into_iter().eq(expected.into_iter().flat_map(|value| [value, value + 10, value + 20])));
        }
    }
}
