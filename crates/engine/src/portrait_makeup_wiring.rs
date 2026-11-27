//! Same-frame wiring for the twelve 2D B3 makeup rows.
//!
//! `portrait_makeup_provider` requires exactly all fourteen `B3_MAKEUP_ROWS` and
//! a complete externally captured job. This module holds the assembly path for
//! the legitimately constructible 2D subset from one `FaceInputBundle`:
//!
//! * Face2D draw bytes and material UV come from `portrait_makeup_mesh2d`, which
//!   derives them from the bundle's 194 landmarks (never reimplemented here).
//! * A row's owner mask must arrive as same-job pixels through the owner-mask
//!   source. This module never synthesizes a mask and never substitutes a PNG.
//! * Material PNGs are decrypted vendor assets; their root is supplied by
//!   `AURORA_MAKEUP_ASSET_ROOT`. No B3 material PNG ships in this repository.
//!
//! ## Owner masks
//!
//! Owner masks are same-job pixels derived from the bundle's 194 landmarks: the
//! row's mesh triangles are rasterized into the recovered ROI, so no static
//! vendor template is substituted. This makes the seven mask-only rows
//! (100, 103, 105, 106, 117, 118, 30007) constructible without any file asset.
//!
//! Material-bearing rows (107 `LipStickImageMakeup`, 111, 115, 116, 90032)
//! remain blocked until `AURORA_MAKEUP_ASSET_ROOT` supplies decrypted PNGs; the
//! assembler skips them instead of failing the whole bundle.
//!
//! PF8 and PF9 are first-class FaceStereo preset rows in
//! `portrait_makeup_bundle::FACESTEREO_PRESET_ROWS`, but their owner mask comes
//! from a different captured pipeline. Requesting them through these 2D
//! landmark bindings is refused by name (`FACESTEREO_OWNER_MASK_UNSUPPORTED`);
//! no mask is synthesized. PF115
//! `3DMMFaceSticker` is refused upstream by `row_profile_for_effect`; this
//! module only selects the 2D `BlusherImageMakeup` branch.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use aurora_portrait::{Image, Refusal};

use crate::portrait_face_input_stage::FaceInputBundle;
use crate::portrait_makeup_bundle::{
    B3_MAKEUP_ROWS, MATERIAL_MISSING, OwnedFace2dDrawBytes, OwnedMakeupBundleInputRow,
    OwnedMakeupBundleInputs, produce_owned_makeup_bundle, row_profile_for_effect,
};
use crate::portrait_makeup_provider::{
    MATERIAL_PNG_DECODE, MATERIAL_PNG_MISSING, load_decrypted_material_png, recovered_filter_type,
};
use crate::portrait_stage::ProducerInputs;

/// Root of the decrypted vendor `effect/config` tree. It is deliberately
/// runtime-only: the decrypted assets are not part of this repository.
pub const MAKEUP_ASSET_ROOT_ENV: &str = "AURORA_MAKEUP_ASSET_ROOT";
pub const MAKEUP_ASSET_ROOT_MISSING: &str = "makeup asset root (AURORA_MAKEUP_ASSET_ROOT)";
pub const MAKEUP_2D_ROW_UNSUPPORTED: &str = "makeup 2D row (FaceStereo excluded)";
/// Named refusal: the 2D landmark wiring does not produce the FaceStereo
/// whole-face owner mask. It must arrive as a same-job capture instead.
pub const FACESTEREO_OWNER_MASK_UNSUPPORTED: &str = "FaceStereo owner mask (no 2D producer)";

/// One planned 2D row. `roi` is the recovered `[x, y, width, height]` region used
/// for draw-time mask UV. `material` is relative to `MAKEUP_ASSET_ROOT_ENV`.
#[derive(Clone, Copy, Debug)]
pub struct Makeup2dRowContract {
    pub pf: u32,
    pub source_effect: &'static str,
    pub roi: [i32; 4],
    pub mix_color: [u8; 4],
    pub opacity_percent: f32,
    pub material: &'static str,
    /// Named refusal when the row's material PNG is unavailable.
    pub material_missing: &'static str,
    /// Named refusal when no same-job owner mask was supplied. Empty for PF111,
    /// which is `MaterialOnly`.
    pub owner_mask_missing: &'static str,
}

pub static MAKEUP_2D_ROWS: [Makeup2dRowContract; 14] = [
    row(8, "BlusherImageMakeup", [95, 40, 901, 1081], [0, 0, 0, 255], 100.0, "FaceStereo/model_FaceStereoShadow_yy.png"),
    row(9, "BlusherImageMakeup", [95, 40, 901, 1081], [0, 0, 0, 255], 100.0, "FaceStereo/model_FaceStereoShadow_gg.png"),
    row(100, "NormalEyeBrowColorMakeup", [210, 410, 260, 110], [100, 82, 70, 255], 100.0, "Makeups/Eyebrow/MUEB00000010/mmz.png"),
    row(103, "SymEyeColorMakeup", [200, 400, 300, 300], [39, 15, 6, 255], 100.0, "Makeups/EyeLiner/MUELR00000001/yxz.png"),
    row(105, "SymEyeLashColorMakeup", [200, 400, 300, 300], [100, 45, 43, 255], 80.0, "Makeups/Eyelash/MUEL00000001/lt.png"),
    row(106, "BlusherColorMakeup", [95, 40, 901, 1081], [234, 112, 129, 255], 100.0, "Makeups/Blusher/MUBL00000001/model_Blusher_4001.png"),
    row(107, "LipStickImageMakeup", [350, 800, 300, 120], [0, 0, 0, 255], 100.0, "Makeups/Lipstick/MULS00000001/Satin/staintexture.png"),
    row(111, "EyePupilMakeup", [200, 400, 300, 300], [0, 0, 0, 255], 100.0, "Makeups/Eyepupil/MUEP00000010/mt.png"),
    row(115, "BlusherImageMakeup", [95, 40, 901, 1081], [0, 0, 0, 255], 100.0, "Makeups/FacialDecoration/MUFD00000004/z01.png"),
    row(116, "BlusherImageMakeup", [95, 40, 901, 1081], [0, 0, 0, 255], 90.0, "Makeups/Contour/MUCT00000001/01.png"),
    row(117, "NormalEyeBrowColorMakeup", [540, 410, 260, 110], [100, 82, 70, 255], 100.0, "Makeups/Eyebrow/MUEB00000010/mmy.png"),
    row(118, "SymEyeLashColorMakeup", [200, 400, 300, 300], [142, 71, 68, 255], 80.0, "Makeups/Eyelash/MUEL00000001/lb.png"),
    row(30007, "SymEyeColorMakeup", [500, 400, 300, 300], [39, 15, 6, 255], 100.0, "Makeups/EyeLiner/MUELR00000001/yxy.png"),
    row(90032, "SymEyeImageMakeup", [200, 400, 300, 300], [0, 0, 0, 255], 100.0, "Makeups/Eyemakeup/MUEM00000001/yyz.png"),
];

const fn row(
    pf: u32,
    source_effect: &'static str,
    roi: [i32; 4],
    mix_color: [u8; 4],
    opacity_percent: f32,
    material: &'static str,
) -> Makeup2dRowContract {
    // Static tags per PF: `Refusal` only carries `&'static str`.
    let (material_missing, owner_mask_missing) = match pf {
        8 => ("MAKEUP_MATERIAL_PNG_MISSING:8", "MAKEUP_SAME_JOB_OWNER_MASK:8"),
        9 => ("MAKEUP_MATERIAL_PNG_MISSING:9", "MAKEUP_SAME_JOB_OWNER_MASK:9"),
        100 => ("MAKEUP_MATERIAL_PNG_MISSING:100", "MAKEUP_SAME_JOB_OWNER_MASK:100"),
        103 => ("MAKEUP_MATERIAL_PNG_MISSING:103", "MAKEUP_SAME_JOB_OWNER_MASK:103"),
        105 => ("MAKEUP_MATERIAL_PNG_MISSING:105", "MAKEUP_SAME_JOB_OWNER_MASK:105"),
        106 => ("MAKEUP_MATERIAL_PNG_MISSING:106", "MAKEUP_SAME_JOB_OWNER_MASK:106"),
        107 => ("MAKEUP_MATERIAL_PNG_MISSING:107", "MAKEUP_SAME_JOB_OWNER_MASK:107"),
        111 => ("MAKEUP_MATERIAL_PNG_MISSING:111", ""),
        115 => ("MAKEUP_MATERIAL_PNG_MISSING:115", "MAKEUP_SAME_JOB_OWNER_MASK:115"),
        116 => ("MAKEUP_MATERIAL_PNG_MISSING:116", "MAKEUP_SAME_JOB_OWNER_MASK:116"),
        117 => ("MAKEUP_MATERIAL_PNG_MISSING:117", "MAKEUP_SAME_JOB_OWNER_MASK:117"),
        118 => ("MAKEUP_MATERIAL_PNG_MISSING:118", "MAKEUP_SAME_JOB_OWNER_MASK:118"),
        30007 => ("MAKEUP_MATERIAL_PNG_MISSING:30007", "MAKEUP_SAME_JOB_OWNER_MASK:30007"),
        90032 => ("MAKEUP_MATERIAL_PNG_MISSING:90032", "MAKEUP_SAME_JOB_OWNER_MASK:90032"),
        _ => ("", ""),
    };
    Makeup2dRowContract {
        pf,
        source_effect,
        roi,
        mix_color,
        opacity_percent,
        material,
        material_missing,
        owner_mask_missing,
    }
}

/// Rows that can be completed without any file asset.
///
/// Their owner masks are rasterized from the bundle's own 194 landmarks, so the
/// seven mask-only color rows run today. Material rows stay out until
/// `AURORA_MAKEUP_ASSET_ROOT` supplies decrypted PNGs.
#[allow(dead_code)]
pub fn makeup_rows_2d_available() -> &'static [u32] {
    &[100, 103, 105, 106, 117, 118, 30007]
}

/// Rows this wiring knows how to declare/assemble once the data-availability
/// blocker is cleared, in canonical B3 order. PF8/PF9 are declared but their
/// owner mask cannot be assembled from 2D landmarks (named refusal).
#[allow(dead_code)]
pub fn makeup_rows_2d_declared() -> &'static [u32] {
    &[8, 9, 100, 103, 105, 106, 107, 111, 115, 116, 117, 118, 30007, 90032]
}

fn row_contract(pf: u32) -> Option<&'static Makeup2dRowContract> {
    MAKEUP_2D_ROWS.iter().find(|row| row.pf == pf)
}

fn normalize_makeup_root(path: &Path) -> Option<PathBuf> {
    if path.join("Makeups").is_dir() {
        return Some(path.to_path_buf());
    }
    if path.file_name().is_some_and(|name| name == "Makeups") {
        return path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(Path::to_path_buf);
    }
    None
}

pub fn makeup_asset_root() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(dir) = std::env::var_os(MAKEUP_ASSET_ROOT_ENV) {
        candidates.push(PathBuf::from(dir));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for rel in ["assets/portrait/makeup", "assets/makeup", "makeup"] {
                candidates.push(dir.join(rel));
            }
            candidates.push(dir.join("../../assets/portrait/makeup"));
        }
    }
    if let Ok(dir) = std::env::current_dir() {
        for rel in ["assets/portrait/makeup", "assets/makeup", "makeup"] {
            candidates.push(dir.join(rel));
        }
    }
    candidates
        .into_iter()
        .find_map(|candidate| normalize_makeup_root(&candidate))
}

/// One discovered makeup style preset under `AURORA_MAKEUP_ASSET_ROOT/Makeups/`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MakeupPresetInfo {
    pub id: String,
    pub path: PathBuf,
    pub mirror_x: bool,
}

fn trailing_number(name: &str) -> u32 {
    let digits: String = name
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    digits.parse::<u32>().unwrap_or(u32::MAX)
}

fn find_png_recursive(dir: &Path, prefer: &[&str], exclude_sub: &[&str]) -> Option<PathBuf> {
    let mut all = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(cur) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&cur) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_dir() {
                stack.push(path);
            } else if (kind.is_file() || (kind.is_symlink() && path.is_file())) && path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("png"))
            {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                if exclude_sub.iter().any(|ex| name.contains(ex)) {
                    continue;
                }
                all.push((name, path));
            }
        }
    }
    all.sort(); // filename first, then path for reproducible ties between material variants
    for pat in prefer {
        let pat = pat.to_ascii_lowercase();
        if let Some((_, p)) = all.iter().find(|(n, _)| n == &pat) {
            return Some(p.clone());
        }
        let matches = |name: &str| match pat.strip_suffix(".png") {
            // yyz1.png is the same side as yyz.png; it must not fall back
            // to the alphabetically earlier opposite-side yyy1.png.
            Some(stem) => name.strip_suffix(".png").and_then(|n| n.strip_prefix(stem))
                .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())),
            None => name.starts_with(&pat),
        };
        if let Some((_, p)) = all.iter().find(|(n, _)| matches(n)) {
            return Some(p.clone());
        }
    }
    all.into_iter().next().map(|(_, p)| p)
}

/// Scans `makeup_asset_root()/Makeups/<category>/` and returns all available
/// style presets for `pf` in natural numeric order.
pub fn makeup_presets_for_pf(pf: u32) -> Vec<MakeupPresetInfo> {
    makeup_asset_root().map_or_else(Vec::new, |root| makeup_presets_at(&root, pf))
}

pub(super) fn makeup_presets_at(root: &Path, pf: u32) -> Vec<MakeupPresetInfo> {
    let (category, prefer, exclude): (&str, &[&str], &[&str]) = match pf {
        100 => ("Eyebrow", &["mmz.png", "mmy.png"], &["sideface"]),
        117 => ("Eyebrow", &["mmy.png", "mmz.png"], &["sideface"]),
        103 | 30007 => ("EyeLiner", &["yxz.png", "yxzs.png", "yxy.png"], &["sideface"]),
        105 => ("Eyelash", &["lt.png", "rt.png"], &["sideface"]),
        118 => ("Eyelash", &["lb.png", "rb.png"], &["sideface"]),
        106 => ("Blusher", &["model_blusher", "sh.png", "sh"], &["sideface"]),
        107 => ("Lipstick", &["staintexture.png"], &["sideface"]),
        111 => ("Eyepupil", &["mt.png"], &["eye_mask", "sideface"]),
        115 => ("FacialDecoration", &["z01.png"], &["sideface"]),
        116 => ("Contour", &[], &["sideface"]),
        90032 => ("Eyemakeup", &["yyz.png", "yyy.png"], &["sideface"]),
        101 => ("Silkworm", &["wcz", "wcy"], &["sideface"]),
        _ => return Vec::new(),
    };
    let cat_dir = root.join("Makeups").join(category);
    let Ok(entries) = std::fs::read_dir(&cat_dir) else {
        return Vec::new();
    };
    let mut dirs: Vec<(u32, String, PathBuf)> = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_dir() {
                return None;
            }
            let id = path.file_name()?.to_str()?.to_string();
            Some((trailing_number(&id), id, path))
        })
        .collect();
    dirs.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut presets = Vec::with_capacity(dirs.len());
    for (_, id, dir) in dirs {
        if let Some(png) = find_png_recursive(&dir, prefer, exclude) {
            let mirror_x = pf == 117
                && png
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.eq_ignore_ascii_case("mmz.png"));
            presets.push(MakeupPresetInfo {
                id,
                path: png,
                mirror_x,
            });
        }
    }
    presets
}

/// Recommended color swatches `(label_zh, rgba8)` for color-driven makeup rows.
pub fn makeup_palette_for_pf(pf: u32) -> &'static [(&'static str, [u8; 4])] {
    const BROW_PALETTE: &[(&str, [u8; 4])] = &[
        ("经典棕", [100, 82, 70, 255]),
        ("炭黑", [36, 34, 34, 255]),
        ("浅咖", [138, 108, 84, 255]),
        ("暖棕", [124, 78, 52, 255]),
        ("红棕", [128, 58, 48, 255]),
    ];
    const BLUSHER_PALETTE: &[(&str, [u8; 4])] = &[
        ("蜜桃粉", [234, 112, 129, 255]),
        ("珊瑚橘", [238, 132, 102, 255]),
        ("玫瑰红", [206, 74, 98, 255]),
        ("杏仁裸", [214, 148, 128, 255]),
        ("梅子粉", [188, 88, 126, 255]),
    ];
    const EYELINER_PALETTE: &[(&str, [u8; 4])] = &[
        ("深棕", [39, 15, 6, 255]),
        ("炭黑", [22, 20, 22, 255]),
        ("酒红", [112, 38, 48, 255]),
        ("大地咖", [96, 64, 48, 255]),
    ];
    const EYELASH_PALETTE: &[(&str, [u8; 4])] = &[
        ("自然棕", [100, 45, 43, 255]),
        ("浓黑", [24, 22, 24, 255]),
        ("焦糖", [132, 76, 56, 255]),
    ];
    match pf {
        100 | 117 => BROW_PALETTE,
        106 => BLUSHER_PALETTE,
        103 | 30007 => EYELINER_PALETTE,
        105 | 118 => EYELASH_PALETTE,
        _ => &[],
    }
}

fn resolve_style_selection(
    pf: u32,
    extra: Option<&BTreeMap<String, f32>>,
) -> Option<MakeupPresetInfo> {
    let extra = extra?;
    let key = format!("pc:{pf}:style");
    let raw = extra.get(&key).copied().or_else(|| {
        // Right brow follows left brow style when not explicitly overridden.
        (pf == 117).then(|| extra.get("pc:100:style").copied()).flatten()
    })?;
    let idx = raw.round() as isize;
    if idx <= 0 {
        return None;
    }
    let presets = makeup_presets_for_pf(pf);
    if presets.is_empty() {
        return None;
    }
    let clamped = (idx as usize - 1).min(presets.len() - 1);
    presets.into_iter().nth(clamped)
}

fn resolve_mix_color(pf: u32, default_color: [u8; 4], extra: Option<&BTreeMap<String, f32>>) -> [u8; 4] {
    let Some(extra) = extra else {
        return default_color;
    };
    let selected = |id| -> Option<[u8; 4]> {
        if let Some(&packed) = extra.get(&format!("pc:{id}:rgb")) {
            // Every RGB24 integer is exactly representable in f32. Absence, not
            // zero, means no custom color; zero is opaque black.
            if (0.0..=16_777_215.0).contains(&packed) && packed.fract() == 0.0 {
                let [_, r, g, b] = (packed as u32).to_be_bytes();
                return Some([r, g, b, 255]);
            }
        }
        let idx = extra.get(&format!("pc:{id}:color"))?.round() as isize;
        let palette = makeup_palette_for_pf(id);
        (idx > 0 && !palette.is_empty()).then(|| palette[(idx as usize - 1).min(palette.len() - 1)].1)
    };
    // An explicit right swatch wins even if the left uses a custom RGB.
    selected(pf).or_else(|| (pf == 117).then(|| selected(100)).flatten()).unwrap_or(default_color)
}

fn asset_root() -> Option<PathBuf> {
    makeup_asset_root()
}

/// Canonical B3 order, de-duplicated, unknown PFs refused. FaceStereo rows are
/// members of `B3_MAKEUP_ROWS` but not of this 2D wiring; the caller drops and
/// reports them.
fn selected_rows(requested: &[u32]) -> Result<Vec<u32>, Refusal> {
    let mut selected = Vec::with_capacity(requested.len());
    for pf in requested {
        if !B3_MAKEUP_ROWS.contains(pf) {
            return Err(Refusal::UnknownEffect(*pf));
        }
        if !selected.contains(pf) {
            selected.push(*pf);
        }
    }
    selected.sort_by_key(|pf| {
        B3_MAKEUP_ROWS
            .iter()
            .position(|row| row == pf)
            .expect("B3 membership checked")
    });
    Ok(selected)
}

impl FaceInputBundle {
    /// Assembles and attaches the owned 2D makeup bundle for `rows`.
    ///
    /// Owner masks are rasterized from this bundle's 194 landmarks into each
    /// row's recovered ROI, so the seven mask-only rows need no file asset.
    /// Material rows are read from `AURORA_MAKEUP_ASSET_ROOT`; rows whose
    /// material is unavailable are skipped, and only an all-skipped request is
    /// refused (`MAKEUP_ASSET_ROOT_MISSING` / `MAKEUP_MATERIAL_PNG_MISSING`).
    /// No static vendor template is ever substituted for an owner mask.
    pub fn attach_makeup_bundle(
        &self,
        source: &Image,
        rows: &[u32],
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.attach_makeup_bundle_with_face_mask(source, rows, None, producer)
    }

    pub fn attach_makeup_bundle_with_face_mask(
        &self,
        source: &Image,
        rows: &[u32],
        face_mask: Option<&[u8]>,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.attach_makeup_bundle_with_options(source, rows, face_mask, None, producer)
    }

    pub fn attach_makeup_bundle_with_options(
        &self,
        source: &Image,
        rows: &[u32],
        face_mask: Option<&[u8]>,
        extra: Option<&BTreeMap<String, f32>>,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        if rows.is_empty() {
            return Err(Refusal::MissingInput("makeup bundle rows"));
        }
        let requested = selected_rows(rows)?;
        let planned: Vec<u32> = requested
            .iter()
            .copied()
            .filter(|pf| row_contract(*pf).is_some())
            .collect();
        let unsupported: Vec<u32> = requested
            .iter()
            .copied()
            .filter(|pf| row_contract(*pf).is_none())
            .collect();
        if !unsupported.is_empty() {
            log::warn!("2D makeup wiring does not own rows {unsupported:?} (FaceStereo)");
        }
        if planned.is_empty() {
            return Err(Refusal::MissingInput(MAKEUP_2D_ROW_UNSUPPORTED));
        }

        let root = asset_root();
        let job_id = self.job_id;
        let root_ref = root.as_deref();
        let (input_rows, skipped_material) = assemble_constructible_rows_with_options(
            job_id,
            [source.width, source.height],
            &self.points194,
            &planned,
            face_mask,
            extra,
            |pf| {
                if let Some(preset) = resolve_style_selection(pf, extra) {
                    if let Some(img) = load_row_material(&preset.path, job_id, pf)? {
                        return Ok(Some((img, preset.mirror_x)));
                    }
                }
                let Some(root) = root_ref else {
                    return Ok(None);
                };
                let contract = row_contract(pf).ok_or(Refusal::UnknownEffect(pf))?;
                Ok(load_row_material(root.join(contract.material), job_id, pf)?
                    .map(|img| (img, false)))
            },
        )?;
        if !skipped_material.is_empty() {
            log::warn!("2D makeup rows skipped for missing material: {skipped_material:?}");
        }
        if input_rows.is_empty() {
            return Err(Refusal::MissingInput(if root.is_none() {
                MAKEUP_ASSET_ROOT_MISSING
            } else {
                MATERIAL_PNG_MISSING
            }));
        }

        let produced = produce_owned_makeup_bundle(Some(OwnedMakeupBundleInputs {
            job_id,
            ticket: Some(job_id),
            frame_size: [source.width, source.height],
            source_sha256: self.source_sha256,
            rows: &input_rows,
        }))?;
        producer.makeup_bundle = Some(produced);
        Ok(())
    }
}

fn load_row_material(
    path: impl AsRef<Path>,
    job_id: u64,
    pf: u32,
) -> Result<Option<Arc<Image>>, Refusal> {
    match load_decrypted_material_png(path, job_id, pf) {
        Ok(material) => {
            let texture = material.texture(job_id);
            let image = Image::new(texture.width, texture.height, texture.rgba8.to_vec())
                .ok_or(Refusal::ShapeMismatch(MATERIAL_MISSING))?;
            Ok(Some(Arc::new(image)))
        }
        Err(error) if error.code == MATERIAL_PNG_MISSING || error.code == MATERIAL_PNG_DECODE => {
            Ok(None)
        }
        Err(error) => Err(error
            .cause
            .unwrap_or(Refusal::MissingInput(error.code))),
    }
}

type OwnedMaterial = Arc<Image>;
type OwnedOwnerMask = Arc<Image>;

/// Assembles every requested row that can be completed right now. Material rows
/// whose PNG is unavailable are skipped and returned separately; owner-mask rows
/// are always constructed from the same-job landmarks.
#[cfg_attr(not(test), allow(dead_code))]
fn assemble_constructible_rows<M>(
    job_id: u64,
    frame_size: [u32; 2],
    points194: &[[f32; 2]],
    selected: &[u32],
    mut load_material: M,
) -> Result<(Vec<OwnedMakeupBundleInputRow>, Vec<u32>), Refusal>
where
    M: FnMut(u32) -> Result<Option<OwnedMaterial>, Refusal>,
{
    assemble_constructible_rows_with_options(
        job_id,
        frame_size,
        points194,
        selected,
        None,
        None,
        |pf| Ok(load_material(pf)?.map(|img| (img, false))),
    )
}

fn assemble_constructible_rows_with_options<M>(
    _job_id: u64,
    frame_size: [u32; 2],
    points194: &[[f32; 2]],
    selected: &[u32],
    face_mask: Option<&[u8]>,
    extra: Option<&BTreeMap<String, f32>>,
    mut load_material: M,
) -> Result<(Vec<OwnedMakeupBundleInputRow>, Vec<u32>), Refusal>
where
    M: FnMut(u32) -> Result<Option<(OwnedMaterial, bool)>, Refusal>,
{
    let mut rows = Vec::with_capacity(selected.len());
    let mut skipped_material = Vec::new();
    for &pf in selected {
        let contract = row_contract(pf).ok_or(Refusal::UnknownEffect(pf))?;
        let profile = row_profile_for_effect(pf, contract.source_effect)?;
        let (material, mask_texture, mirror_x) = if profile.requires_material {
            match load_material(pf)? {
                Some((material, mirror)) => (Some(material), None, mirror),
                None => {
                    skipped_material.push(pf);
                    continue;
                }
            }
        } else {
            let loaded = load_material(pf).ok().flatten();
            match loaded {
                Some((img, mirror)) => (None, Some(img), mirror),
                None => (None, None, false),
            }
        };
        let draw = crate::portrait_makeup_mesh2d::build_face2d_draw_bytes(
            points194,
            frame_size,
            &profile,
        )?;
        let positions = decode_owned_positions(&draw)?;
        let indices = decode_owned_indices(&draw)?;
        let (origin, size) = draw_origin_size(&positions, frame_size)?;
        let owner_mask: Option<OwnedOwnerMask> = if profile.requires_owner_mask() {
            Some(derive_owner_mask_textured(
                pf,
                &positions,
                &indices,
                origin,
                size,
                frame_size,
                mask_texture.as_deref(),
                mirror_x,
                face_mask,
            )?)
        } else {
            None
        };
        let material_uv = if profile.requires_material {
            Some(crate::portrait_makeup_mesh2d::build_region_uv(
                points194,
                frame_size,
                &profile,
            )?)
        } else {
            None
        };
        let filter_type = recovered_filter_type(pf, contract.source_effect)
            .ok_or(Refusal::MissingInput("makeup provider FilterType"))?;
        let mix_color = resolve_mix_color(pf, contract.mix_color, extra);
        rows.push(OwnedMakeupBundleInputRow {
            pf,
            source_effect: Some(contract.source_effect.to_owned()),
            face_id: Some(0),
            draw,
            origin,
            size,
            owner_mask,
            extra_owner_mask: None,
            material,
            material_uv,
            mix_color,
            opacity_percent: contract.opacity_percent,
            filter_type,
        });
    }
    Ok((rows, skipped_material))
}

/// Integer bounds of the row's draw, clipped to the frame. These become the
/// bundle `origin`/`size`, so `produce_face2d_mask_uv` normalizes every vertex
/// into `[0,1]`.
fn draw_origin_size(
    positions: &[[f32; 2]],
    frame_size: [u32; 2],
) -> Result<([i32; 2], [i32; 2]), Refusal> {
    if positions.is_empty() {
        return Err(Refusal::MissingInput("makeup draw positions"));
    }
    if frame_size[0] == 0 || frame_size[1] == 0 {
        return Err(Refusal::ShapeMismatch("makeup frame size"));
    }
    if positions.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("makeup draw positions"));
    }
    let min_x = positions
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, frame_size[0] as f32) as i32;
    let min_y = positions
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, frame_size[1] as f32) as i32;
    let max_x = positions
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, frame_size[0] as f32) as i32;
    let max_y = positions
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, frame_size[1] as f32) as i32;
    if max_x <= min_x || max_y <= min_y {
        return Err(Refusal::ShapeMismatch("makeup draw bounds"));
    }
    Ok(([min_x, min_y], [max_x - min_x, max_y - min_y]))
}

#[cfg_attr(not(test), allow(dead_code))]
fn derive_owner_mask(
    positions: &[[f32; 2]],
    indices: &[u16],
    origin: [i32; 2],
    size: [i32; 2],
) -> Result<OwnedOwnerMask, Refusal> {
    derive_owner_mask_textured(0, positions, indices, origin, size, [0, 0], None, false, None)
}

fn sample_mask_bilinear(tex: &Image, u: f32, v: f32) -> u8 {
    let w = tex.width as usize;
    let h = tex.height as usize;
    if w == 0 || h == 0 || tex.pixels.len() < w * h * 4 {
        return 0;
    }
    let fx = (u.clamp(0.0, 1.0) * (w.saturating_sub(1)) as f32).clamp(0.0, (w - 1) as f32);
    let fy = (v.clamp(0.0, 1.0) * (h.saturating_sub(1)) as f32).clamp(0.0, (h - 1) as f32);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let at = |x: usize, y: usize| -> f32 {
        let base = (y * w + x) * 4;
        // Material mask PNGs (mmz/mmy/yxz/lt/model_Blusher_4001) store soft
        // intensity in RGB (grayscale), with alpha=255.
        tex.pixels[base] as f32
    };
    let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
    let bot = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
    (top * (1.0 - ty) + bot * ty).round().clamp(0.0, 255.0) as u8
}

fn derive_owner_mask_textured(
    pf: u32,
    positions: &[[f32; 2]],
    indices: &[u16],
    origin: [i32; 2],
    size: [i32; 2],
    frame_size: [u32; 2],
    material_mask: Option<&Image>,
    mirror_x: bool,
    face_mask: Option<&[u8]>,
) -> Result<OwnedOwnerMask, Refusal> {
    if size[0] <= 0 || size[1] <= 0 {
        return Err(Refusal::ShapeMismatch("makeup owner mask bounds"));
    }
    let (w, h) = (size[0] as usize, size[1] as usize);
    let mut rgba = vec![0u8; w * h * 4];
    let origin_f = [origin[0] as f32, origin[1] as f32];
    let mut covered = false;
    for triangle in indices.chunks_exact(3) {
        let points = [
            positions
                .get(triangle[0] as usize)
                .copied()
                .ok_or(Refusal::ShapeMismatch("makeup owner mask vertex"))?,
            positions
                .get(triangle[1] as usize)
                .copied()
                .ok_or(Refusal::ShapeMismatch("makeup owner mask vertex"))?,
            positions
                .get(triangle[2] as usize)
                .copied()
                .ok_or(Refusal::ShapeMismatch("makeup owner mask vertex"))?,
        ];
        covered |= rasterize_triangle(&mut rgba, w, h, origin_f, points);
    }
    if !covered {
        return Err(Refusal::ShapeMismatch("makeup owner mask coverage"));
    }

    // When a real decrypted soft-gradient mask PNG is available for ColorMask /
    // HistoricalColor rows, modulate the rasterized polygon by the texture so
    // makeup follows the actual feathered brow/eyeliner/lash/blusher shape.
    if let Some(tex) = material_mask {
        let dual_eye_quads = matches!(pf, 103 | 105 | 118 | 30007) && positions.len() == 8;
        if dual_eye_quads {
            let left_box = (
                positions[0][0].min(positions[2][0]),
                positions[0][1].min(positions[2][1]),
                (positions[2][0] - positions[0][0]).abs().max(1.0),
                (positions[2][1] - positions[0][1]).abs().max(1.0),
            );
            let right_box = (
                positions[4][0].min(positions[6][0]),
                positions[4][1].min(positions[6][1]),
                (positions[6][0] - positions[4][0]).abs().max(1.0),
                (positions[6][1] - positions[4][1]).abs().max(1.0),
            );
            for y in 0..h {
                let fy = origin_f[1] + y as f32 + 0.5;
                for x in 0..w {
                    let at = (y * w + x) * 4;
                    if rgba[at] == 0 {
                        continue;
                    }
                    let fx = origin_f[0] + x as f32 + 0.5;
                    let mut val = 0u8;
                    if fx >= left_box.0
                        && fx <= left_box.0 + left_box.2
                        && fy >= left_box.1
                        && fy <= left_box.1 + left_box.3
                    {
                        let u = (fx - left_box.0) / left_box.2;
                        let v = (fy - left_box.1) / left_box.3;
                        val = val.max(sample_mask_bilinear(tex, u, v));
                    }
                    if fx >= right_box.0
                        && fx <= right_box.0 + right_box.2
                        && fy >= right_box.1
                        && fy <= right_box.1 + right_box.3
                    {
                        let u = 1.0 - (fx - right_box.0) / right_box.2;
                        let v = (fy - right_box.1) / right_box.3;
                        val = val.max(sample_mask_bilinear(tex, u, v));
                    }
                    rgba[at..at + 4].copy_from_slice(&[val, val, val, val]);
                }
            }
        } else {
            let wf = w as f32;
            let hf = h as f32;
            for y in 0..h {
                let v = (y as f32 + 0.5) / hf;
                for x in 0..w {
                    let at = (y * w + x) * 4;
                    if rgba[at] == 0 {
                        continue;
                    }
                    let raw_u = (x as f32 + 0.5) / wf;
                    let u = if mirror_x { 1.0 - raw_u } else { raw_u };
                    let val = sample_mask_bilinear(tex, u, v);
                    rgba[at..at + 4].copy_from_slice(&[val, val, val, val]);
                }
            }
        }
    }

    // Clip every owner mask against the same-job FaceMask so side-profile
    // polygons never spill onto background pixels outside the face.
    if let Some(fmask) = face_mask {
        let fw = frame_size[0] as usize;
        let fh = frame_size[1] as usize;
        if fw > 0 && fh > 0 && fmask.len() == fw * fh {
            for y in 0..h {
                let sy = origin[1] + y as i32;
                for x in 0..w {
                    let sx = origin[0] + x as i32;
                    let at = (y * w + x) * 4;
                    if rgba[at] == 0 && rgba[at + 1] == 0 && rgba[at + 2] == 0 && rgba[at + 3] == 0
                    {
                        continue;
                    }
                    let fm = if sx >= 0 && sy >= 0 && (sx as usize) < fw && (sy as usize) < fh {
                        fmask[sy as usize * fw + sx as usize] as u32
                    } else {
                        0
                    };
                    for c in 0..4 {
                        rgba[at + c] = ((rgba[at + c] as u32 * fm + 127) / 255) as u8;
                    }
                }
            }
        }
    }

    Image::new(size[0] as u32, size[1] as u32, rgba)
        .map(Arc::new)
        .ok_or(Refusal::ShapeMismatch("makeup owner mask image"))
}

fn decode_owned_positions(draw: &OwnedFace2dDrawBytes) -> Result<Vec<[f32; 2]>, Refusal> {
    let stride = draw.vertex_stride;
    if draw.vertices.is_empty()
        || stride < 8
        || draw.position_offset > stride - 8
        || draw.vertices.len() % stride != 0
    {
        return Err(Refusal::ShapeMismatch("makeup owner mask VBO"));
    }
    let offset = draw.position_offset;
    Ok(draw
        .vertices
        .chunks_exact(stride)
        .map(|vertex| {
            [
                f32::from_le_bytes(vertex[offset..offset + 4].try_into().expect("four-byte x")),
                f32::from_le_bytes(
                    vertex[offset + 4..offset + 8]
                        .try_into()
                        .expect("four-byte y"),
                ),
            ]
        })
        .collect())
}

fn decode_owned_indices(draw: &OwnedFace2dDrawBytes) -> Result<Vec<u16>, Refusal> {
    if draw.indices_u16.is_empty() || draw.draw_index_count == 0 || draw.draw_index_count % 3 != 0
    {
        return Err(Refusal::ShapeMismatch("makeup owner mask EBO"));
    }
    let start = draw
        .draw_first_index
        .checked_mul(2)
        .ok_or(Refusal::ShapeMismatch("makeup owner mask EBO"))?;
    let end = start
        .checked_add(draw.draw_index_count * 2)
        .ok_or(Refusal::ShapeMismatch("makeup owner mask EBO"))?;
    let bytes = draw
        .indices_u16
        .get(start..end)
        .ok_or(Refusal::ShapeMismatch("makeup owner mask EBO"))?;
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

fn rasterize_triangle(
    rgba: &mut [u8],
    width: usize,
    height: usize,
    origin: [f32; 2],
    triangle: [[f32; 2]; 3],
) -> bool {
    let points = [
        [triangle[0][0] - origin[0], triangle[0][1] - origin[1]],
        [triangle[1][0] - origin[0], triangle[1][1] - origin[1]],
        [triangle[2][0] - origin[0], triangle[2][1] - origin[1]],
    ];
    let edge = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let area = edge(points[0], points[1], points[2]);
    if !area.is_finite() || area == 0.0 {
        return false;
    }
    let min_x = (points
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as isize)
        .clamp(0, width as isize) as usize;
    let min_y = (points
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as isize)
        .clamp(0, height as isize) as usize;
    let max_x = (points
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil() as isize)
        .clamp(0, width as isize) as usize;
    let max_y = (points
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil() as isize)
        .clamp(0, height as isize) as usize;
    if min_x >= max_x || min_y >= max_y {
        return false;
    }
    let mut covered = false;
    for y in min_y..max_y {
        for x in min_x..max_x {
            let sample = [x as f32 + 0.5, y as f32 + 0.5];
            let w0 = edge(points[1], points[2], sample) / area;
            let w1 = edge(points[2], points[0], sample) / area;
            let w2 = edge(points[0], points[1], sample) / area;
            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                let at = (y * width + x) * 4;
                rgba[at..at + 4].copy_from_slice(&[255, 255, 255, 255]);
                covered = true;
            }
        }
    }
    covered
}

#[cfg(test)]
mod tests {
    use super::*;
    use aurora_portrait::kernels::makeup_mask_uv::color_mask_profile;
    use crate::portrait_makeup_bundle::row_profile;

    fn synthetic_points194() -> Vec<[f32; 2]> {
        (0..194)
            .map(|index| {
                let ring = (index % 3) as f32;
                let angle = index as f32 * 0.37;
                let rx = 330.0 - ring * 70.0;
                let ry = 380.0 - ring * 80.0;
                [450.0 + rx * angle.cos(), 520.0 + ry * angle.sin()]
            })
            .collect()
    }

    #[test]
    fn mask_only_rows_are_constructible_without_assets() {
        assert_eq!(
            makeup_rows_2d_available(),
            &[100, 103, 105, 106, 117, 118, 30007]
        );
        assert_eq!(
            makeup_rows_2d_declared(),
            &[8, 9, 100, 103, 105, 106, 107, 111, 115, 116, 117, 118, 30007, 90032]
        );
        assert_eq!(
            makeup_rows_2d_declared(),
            MAKEUP_2D_ROWS.map(|row| row.pf).as_slice()
        );
        for pf in makeup_rows_2d_declared() {
            assert!(B3_MAKEUP_ROWS.contains(pf), "PF{pf} is not a B3 row");
            assert!(row_contract(*pf).is_some(), "PF{pf} has no contract");
        }
        for pf in makeup_rows_2d_available() {
            let contract = row_contract(*pf).unwrap();
            let profile = row_profile_for_effect(*pf, contract.source_effect).unwrap();
            assert!(profile.requires_owner_mask());
            assert!(!profile.requires_material);
        }
    }

    #[test]
    fn owner_masks_are_rasterized_from_landmarks_into_each_roi() {
        let points = synthetic_points194();
        for pf in makeup_rows_2d_available() {
            let contract = row_contract(*pf).unwrap();
            let profile = row_profile_for_effect(*pf, contract.source_effect).unwrap();
            let draw = crate::portrait_makeup_mesh2d::build_face2d_draw_bytes(
                &points,
                [901, 1081],
                &profile,
            )
            .unwrap();
            let positions = decode_owned_positions(&draw).unwrap();
            let indices = decode_owned_indices(&draw).unwrap();
            let (origin, size) = draw_origin_size(&positions, [901, 1081]).unwrap();
            for point in &positions {
                let u = (point[0] - origin[0] as f32) / size[0] as f32;
                let v = (point[1] - origin[1] as f32) / size[1] as f32;
                assert!(
                    (-1.0e-6..=1.0 + 1.0e-6).contains(&u)
                        && (-1.0e-6..=1.0 + 1.0e-6).contains(&v),
                    "PF{pf} vertex outside draw bounds"
                );
            }
            let mask = derive_owner_mask(&positions, &indices, origin, size).unwrap();
            assert_eq!(mask.width, size[0] as u32, "PF{pf} mask width");
            assert_eq!(mask.height, size[1] as u32, "PF{pf} mask height");
            assert_eq!(
                mask.pixels.len(),
                (mask.width * mask.height * 4) as usize,
                "PF{pf} mask plane"
            );
            assert!(
                mask.pixels
                    .chunks_exact(4)
                    .any(|pixel| pixel[0] == 255 && pixel[1] == 255 && pixel[3] == 255),
                "PF{pf} mask has no coverage"
            );
        }
    }

    #[test]
    fn assembly_skips_material_rows_and_keeps_mask_rows() {
        let points = synthetic_points194();
        let selectable: Vec<u32> = makeup_rows_2d_declared()
            .iter()
            .copied()
            .filter(|pf| *pf != 8 && *pf != 9)
            .collect();
        let (rows, skipped) = assemble_constructible_rows(
            7,
            [901, 1081],
            &points,
            &selectable,
            |_| Ok(None),
        )
        .unwrap();
        assert_eq!(
            rows.iter().map(|row| row.pf).collect::<Vec<_>>(),
            vec![100, 103, 105, 106, 117, 118, 30007]
        );
        assert_eq!(skipped, vec![107, 111, 115, 116, 90032]);
        for row in rows {
            assert!(row.owner_mask.is_some(), "PF{} lost its owner mask", row.pf);
            assert!(row.material.is_none(), "PF{} gained a material", row.pf);
            assert!(row.material_uv.is_none(), "PF{} gained a material UV", row.pf);
            let mask = row.owner_mask.as_ref().unwrap();
            assert_eq!(mask.width, row.size[0] as u32, "PF{} mask width", row.pf);
            assert_eq!(mask.height, row.size[1] as u32, "PF{} mask height", row.pf);
        }
    }

    #[test]
    fn request_order_is_canonical_and_deduped() {
        assert_eq!(
            selected_rows(&[90032, 100, 115]).unwrap(),
            vec![100, 115, 90032],
            "2D order stays canonical and deduped"
        );
        assert_eq!(
            selected_rows(&[90032, 106]).unwrap(),
            vec![106, 90032],
            "B3 order is preserved, not request order"
        );
        assert_eq!(
            selected_rows(&[999]).unwrap_err(),
            Refusal::UnknownEffect(999)
        );
        for pf in [8, 9] {
            assert_eq!(selected_rows(&[pf]).unwrap(), vec![pf]);
        }
    }

    #[test]
    fn every_row_names_its_material_png() {
        for contract in MAKEUP_2D_ROWS {
            assert!(
                contract.material.starts_with("Makeups/")
                    || contract.material.starts_with("FaceStereo/"),
                "PF{} material path is neither Makeups/ nor FaceStereo/",
                contract.pf
            );
            assert!(contract.material.ends_with(".png"));
            assert!(
                contract
                    .material_missing
                    .starts_with("MAKEUP_MATERIAL_PNG_MISSING:")
            );
        }
    }

    #[test]
    fn rgb24_black_is_present_and_right_brow_overrides_left() {
        let default = [91, 94, 99, 255];
        let mut extra = BTreeMap::from([("pc:100:rgb".to_owned(), 0.0)]);
        assert_eq!(resolve_mix_color(100, default, Some(&extra)), [0, 0, 0, 255]);
        assert_eq!(resolve_mix_color(117, default, Some(&extra)), [0, 0, 0, 255]);
        extra.insert("pc:117:color".to_owned(), 2.0);
        assert_eq!(resolve_mix_color(117, default, Some(&extra)), makeup_palette_for_pf(117)[1].1);
        extra.insert("pc:117:rgb".to_owned(), 0xff0080 as f32);
        assert_eq!(resolve_mix_color(117, default, Some(&extra)), [255, 0, 128, 255]);
        extra.insert("pc:117:rgb".to_owned(), 16_777_216.0);
        assert_eq!(resolve_mix_color(117, default, Some(&extra)), makeup_palette_for_pf(117)[1].1);
    }

    #[test]
    fn roi_matches_the_recovered_color_profiles() {
        for pf in [100, 103, 105, 117, 118, 30007] {
            let recovered = color_mask_profile(pf).unwrap();
            let contract = row_contract(pf).unwrap();
            assert_eq!(
                contract.roi,
                [
                    recovered.roi.x,
                    recovered.roi.y,
                    recovered.roi.width,
                    recovered.roi.height
                ],
                "PF{pf} ROI drift"
            );
        }
    }

    #[test]
    fn filter_types_come_from_the_recovered_table() {
        for contract in MAKEUP_2D_ROWS {
            assert!(
                recovered_filter_type(contract.pf, contract.source_effect).is_some(),
                "PF{} has no recovered FilterType",
                contract.pf
            );
        }
    }

    #[test]
    fn owner_mask_tag_is_present_exactly_for_masked_rows() {
        for contract in MAKEUP_2D_ROWS {
            let profile = row_profile(contract.pf).unwrap();
            if profile.requires_owner_mask() {
                assert!(
                    contract
                        .owner_mask_missing
                        .starts_with("MAKEUP_SAME_JOB_OWNER_MASK:")
                );
            } else {
                assert!(contract.owner_mask_missing.is_empty());
            }
        }
    }
}

#[cfg(test)]
mod facestereo_wiring_tests {
    use super::*;

    #[test]
    fn facestereo_rows_are_declared_with_2d_hull_masks() {
        for pf in [8, 9] {
            assert_eq!(selected_rows(&[pf]).unwrap(), vec![pf]);
            assert!(makeup_rows_2d_declared().contains(&pf));
            let contract = row_contract(pf).unwrap();
            assert!(contract.material.starts_with("FaceStereo/"));
            let profile = row_profile_for_effect(pf, contract.source_effect).unwrap();
            assert!(profile.requires_owner_mask());
        }
    }

    #[test]
    fn no_landmark_based_owner_mask_is_generated_for_facestereo() {
        let profile = row_profile_for_effect(8, "BlusherImageMakeup").unwrap();
        assert!(profile.requires_owner_mask());
        assert!(profile.mask_semantic.contains("BlusherMaskType=15"));
        assert_eq!(
            profile.declared_mask_channel,
            Some(aurora_portrait::kernels::makeup_material::MaskChannel::Rgba)
        );
    }
}
