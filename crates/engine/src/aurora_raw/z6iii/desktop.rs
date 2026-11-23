use super::{ExportAssets, reconstruct, render, render::check_assets};
use crate::{AdjustmentSet, ExportFormat, ImageInfo, OutputSharpening, ProcessRecipe, ScaleMode};
use anyhow::{Context, Result};
use std::{fs::File, io::{Read, Seek, SeekFrom}, path::{Path, PathBuf}};

const CURVES: [&str; 6] = [
    "NikonZ6III-Auto",
    "NikonZ6III-Film Standard",
    "NikonZ6III-Film Extra Shadow",
    "NikonZ6III-Film High Contrast",
    "NikonZ6III-Portrait",
    "NikonZ6III-Linear Response",
];

fn curve_stem(adjustments: &AdjustmentSet) -> &str {
    adjustments.base.film_curve.as_deref().unwrap_or(CURVES[0])
}

pub(crate) fn compatible(adjustments: &AdjustmentSet, recipe: &ProcessRecipe) -> bool {
    let mut neutral = AdjustmentSet::default();
    let profile = adjustments.base.icc_profile.as_deref();
    if !matches!(profile, None | Some("NikonZ6III-ProStandard"))
        || !CURVES.contains(&curve_stem(adjustments)) {
        return false;
    }
    neutral.base.icc_profile = adjustments.base.icc_profile.clone();
    neutral.base.film_curve = adjustments.base.film_curve.clone();
    neutral.sharpening.amount = 0.0;
    neutral.noise_reduction.luminance = 0.0;
    neutral.noise_reduction.color = 0.0;
    // Geometry is applied by the 16-bit stage, so it no longer disqualifies the
    // specialised export: only the non-geometry adjustments have to be neutral.
    let mut candidate = adjustments.clone();
    candidate.geometry = crate::Geometry::default();
    neutral.geometry = crate::Geometry::default();
    candidate == neutral && recipe.format == ExportFormat::Tiff && recipe.bit_depth == 16
        && recipe.icc_profile.trim().eq_ignore_ascii_case("Adobe RGB (1998)")
        && recipe.scale == ScaleMode::Fixed100 && recipe.watermark.is_none()
        && recipe.output_sharpening == OutputSharpening::Disabled
        && recipe.output_sharpening_settings.is_none()
}

struct Assets {
    camera: Vec<u8>,
    curve: Vec<u8>,
    adobe: Vec<u8>,
    banding: [[u8; 256]; 2],
    packed: [u8; 1024],
}

impl Assets {
    fn load(root: &Path, curve: &str) -> Result<Self> {
        let read = |path: &str| std::fs::read(root.join(path)).with_context(|| format!("read {path}"));
        let mut assets = Self {
            camera: read("Color Profiles/DSLR/NikonZ6III-ProStandard.icm")?,
            curve: read(&format!("Film Curves/{curve}.fcrv"))?,
            adobe: read("Color Profiles/Common/Adobe RGB (1998).icm")?,
            banding: [[0; 256]; 2],
            packed: [0; 1024],
        };
        let mut core = File::open(root.join("ImgCoreDll.dll"))?;
        let [left, right] = &mut assets.banding;
        for (offset, table) in [(40_613_632, &mut left[..]), (40_614_912, &mut right[..]), (40_613_888, &mut assets.packed[..])] {
            core.seek(SeekFrom::Start(offset))?;
            core.read_exact(table)?;
        }
        check_assets(&assets.borrow())?;
        Ok(assets)
    }

    fn borrow(&self) -> ExportAssets<'_> {
        ExportAssets {
            camera_icc: &self.camera, curve: &self.curve, linear_curve: &[], working_icc: &[],
            adobe_rgb_icc: &self.adobe, banding_tables: [&self.banding[0], &self.banding[1], &self.packed],
        }
    }
}

pub(crate) struct Export {
    nef: memmap2::Mmap,
    assets: Assets,
    geometry: crate::Geometry,
}

impl Export {
    pub(crate) fn prepare(path: &Path, adjustments: &AdjustmentSet, recipe: &ProcessRecipe) -> Result<Option<Self>> {
        if !compatible(adjustments, recipe)
            || !path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("nef")) {
            return Ok(None);
        }
        let file = File::open(path)?;
        let nef = unsafe { memmap2::Mmap::map(&file)? };
        if !reconstruct::is_z6iii_nef(&nef) || reconstruct::find_he_strip(&nef)?.is_none()
            || !matches!(crate::raw::exif_orientation_bytes(&nef).unwrap_or(1), 1 | 3 | 6 | 8)
            || !reconstruct::nikon_exif_sensitivity(&nef)?.is_some_and(|iso| super::color::z6iii_noise::z6iii_noise_for_iso(iso).is_some()) {
            return Ok(None);
        }
        if !reconstruct::gpu_available() {
            anyhow::bail!("NVIDIA OpenCL is unavailable");
        }
        let root = std::env::var_os("AURORA_Z6III_C1_DIR").map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\Program Files\Capture One\Capture One"));
        Ok(Some(Self {
            nef,
            assets: Assets::load(&root, curve_stem(adjustments))?,
            geometry: adjustments.geometry.clone(),
        }))
    }

    pub(crate) fn write(self, dir: &Path, info: &ImageInfo, recipe: &ProcessRecipe, cancelled: impl FnOnce() -> bool) -> Result<PathBuf> {
        let frame = render(&self.nef, &self.assets.borrow())?;
        let (frame, banding_origin) =
            super::geometry16::apply_frame_with_origin(frame, &self.geometry)?;
        // Phase source for the banding stage: the source pixel that lands on
        // output (0, 0). `banding_phase_dispatch_v2.patch` adds
        // `ExportAssets.banding_phase`; until that lands this value is carried
        // here so the wire is a one-line move at the call site.
        let _banding_phase = banding_origin;
        anyhow::ensure!(!cancelled(), "cancelled");
        crate::recipe::write_z6iii_tiff(dir, info, recipe, frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_neutral_full_size_adobe_rgb16_uses_the_specialized_engine() {
        let mut neutral = AdjustmentSet::default();
        let mut recipe = ProcessRecipe { format: ExportFormat::Tiff, bit_depth: 16,
            icc_profile: "Adobe RGB (1998)".into(), ..ProcessRecipe::default() };
        assert!(!compatible(&neutral, &recipe));
        neutral.sharpening.amount = 0.0;
        neutral.noise_reduction.luminance = 0.0;
        neutral.noise_reduction.color = 0.0;
        assert!(compatible(&neutral, &recipe));
        neutral.lens.distortion = 100.0;
        assert!(!compatible(&neutral, &recipe));
        neutral.lens.distortion = 0.0;
        neutral.base.film_curve = Some("NikonZ6III-Linear Response".into());
        assert!(compatible(&neutral, &recipe));
        neutral.base.film_curve = Some("NikonZ6III-Portrait".into());
        assert!(compatible(&neutral, &recipe));
        neutral.base.film_curve = Some("NikonZ6III-Unknown".into());
        assert!(!compatible(&neutral, &recipe));
        neutral.base.film_curve = Some("NikonZ6III-Auto".into());
        recipe.scale = ScaleMode::LongEdge;
        assert!(!compatible(&neutral, &recipe));
        recipe.scale = ScaleMode::Fixed100;
        recipe.icc_profile = "sRGB".into();
        assert!(!compatible(&neutral, &recipe));
    }
}
