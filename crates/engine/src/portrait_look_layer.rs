//! Same-frame PF111 catchlight sub-layer: source selection, ticket binding and
//! the opt-in CPU composite.
//!
//! This layer is deliberately the *only* place that turns `pc:111` look data
//! into a catchlight. It does not touch the historical PF111 pupil/contact-lens
//! path, it never invents a vendor `ParamFlag`, and it never mutates a source
//! image: [`prepare`] only reads the bundle and the selected PNG, and [`apply`]
//! writes only the iris RGB of the caller-provided working frame after the
//! separate [`crate::portrait_look_catchlight`] compositor has validated both
//! eyes. A missing, ambiguous or malformed request is a typed [`Refusal`],
//! never a silent fallback to the last discovered material.
//!
//! The material is a real, human catchlight PNG discovered by
//! [`crate::portrait_makeup_wiring::human_catchlights_at`]; the animal
//! `AnimalCatchlights` tree is never searched. `pc:111:catchlight_style` is a
//! real 1-based ordinal into that discovered, naturally sorted list.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use aurora_portrait::{Image, Refusal};

use crate::portrait_face_input_stage::FaceInputBundle;
use crate::portrait_look_catchlight::{
    CATCHLIGHT_RING_POINTS, CatchlightEye, CatchlightMaterial, CatchlightParams,
    composite_catchlight_materials,
};

/// The owner look row. The catchlight child is only legal while this row is a
/// real, non-zero request.
pub(crate) const PC111_KEY: &str = "pc:111";
pub(crate) const CATCHLIGHT_AMOUNT_KEY: &str = "pc:111:catchlight_amount";
pub(crate) const CATCHLIGHT_STYLE_KEY: &str = "pc:111:catchlight_style";
pub(crate) const CATCHLIGHT_RGB_KEY: &str = "pc:111:catchlight_rgb";

/// Neutral RGB24: white, so the authored material colour is unchanged.
pub(crate) const CATCHLIGHT_RGB_WHITE: u32 = 0x00FF_FFFF;

/// Real same-frame iris rings: the 194-point template indices used by the
/// makeup mesh (`IRIS_LEFT` / `IRIS_RIGHT`), never a cross-image or fitted ring.
const LEFT_IRIS: [usize; CATCHLIGHT_RING_POINTS] = [27, 28, 29, 30, 31, 32, 33, 34];
const RIGHT_IRIS: [usize; CATCHLIGHT_RING_POINTS] = [45, 46, 47, 48, 49, 50, 51, 52];

/// Straight RGBA8 bounds for a real catchlight material. Authored materials are
/// 300x300; the upper bound only rejects obviously wrong/huge inputs.
const MATERIAL_MIN_SIDE: u32 = 2;
const MATERIAL_MAX_SIDE: u32 = 1024;
/// Upper bound on the whole PNG file, checked before reading it fully.
const MATERIAL_MAX_FILE_BYTES: u64 = 8 * 1024 * 1024;

/// Which source frame and which look material this layer was prepared from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HumanCatchlightTicket {
    pub(crate) job_id: u64,
    /// Digest of the original source at prepare time; provenance and
    /// fingerprint input only, never rechecked against the working frame.
    pub(crate) source_sha256: [u8; 32],
    pub(crate) source_size: [u32; 2],
    pub(crate) preset_id: String,
    /// 1-based ordinal into the discovered human catchlight list.
    pub(crate) style_index: u32,
}

/// A prepared, same-frame catchlight sub-layer.
///
/// The texture is decoded once in [`prepare`] and stored as straight RGBA8;
/// [`apply`] passes those bytes to the compositor, so no PNG is loaded or
/// decoded a second time. The field is crate-visible for the wiring owner's
/// fingerprint/attach code, not a public image API.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HumanCatchlight {
    pub(crate) material: Arc<Image>,
    pub(crate) rings: [[[f32; 2]; CATCHLIGHT_RING_POINTS]; 2],
    pub(crate) amount: f32,
    pub(crate) tint_rgb: [u8; 3],
    pub(crate) params: CatchlightParams,
    pub(crate) mirror_u: [bool; 2],
    pub(crate) ticket: HumanCatchlightTicket,
}

/// Explicit CPU gate for this sub-layer.
///
/// `auto`, an unset/empty value and `cpu` may use the CPU reference. Any
/// explicit device backend (for example `cuda`, `metal`, `wgpu`) is refused
/// here because no native catchlight sub-pass exists yet; returning `true`
/// would silently promote a device request to CPU. A present
/// `PORTRAIT_CUDA_REQUIRED` (the forced real-device QA switch) also refuses
/// CPU. Both `AURORA_GPU_BACKEND` and `AURORA_PORTRAIT_RENDER_BACKEND` are
/// checked; either one being explicit non-CPU is enough to refuse.
fn backend_allows_cpu(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        None => true,
        Some(value) => matches!(value.to_ascii_lowercase().as_str(), "" | "auto" | "cpu"),
    }
}

pub(crate) fn cpu_allowed() -> bool {
    // Forced real-device QA is never allowed to fall back to the CPU reference.
    if std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_some() {
        return false;
    }
    // Both the legacy and the real portrait render entry must agree. A value
    // outside {empty, auto, cpu} is an explicit non-CPU request.
    for key in ["AURORA_GPU_BACKEND", "AURORA_PORTRAIT_RENDER_BACKEND"] {
        let value = std::env::var(key).ok();
        if !backend_allows_cpu(value.as_deref()) {
            return false;
        }
    }
    true
}

fn has_child_keys(extra: &BTreeMap<String, f32>) -> bool {
    extra.contains_key(CATCHLIGHT_AMOUNT_KEY)
        || extra.contains_key(CATCHLIGHT_STYLE_KEY)
        || extra.contains_key(CATCHLIGHT_RGB_KEY)
}

fn amount_is_valid(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn style_is_valid(value: f32) -> bool {
    value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= 1_000_000.0
}

fn rgb_is_valid(value: f32) -> bool {
    value.is_finite()
        && value.fract() == 0.0
        && (0.0..=CATCHLIGHT_RGB_WHITE as f32).contains(&value)
}

/// True when this child request should be honored or explicitly refused.
///
/// Zero amount means "keep the selection but do not composite": that is *not*
/// a request, so the stage keeps the base PF111 iris path. Malformed child
/// values still return `true` so the caller reaches [`prepare`] and gets a
/// typed refusal instead of a silent drop. A missing/zero parent owner is not
/// a request either; an owner above 1 is malformed and is reported.
pub(crate) fn requested(extra: &BTreeMap<String, f32>) -> bool {
    if !has_child_keys(extra) {
        return false;
    }
    let amount = extra.get(CATCHLIGHT_AMOUNT_KEY).copied();
    let style = extra.get(CATCHLIGHT_STYLE_KEY).copied();
    let rgb = extra.get(CATCHLIGHT_RGB_KEY).copied();
    if amount.is_some_and(|value| !amount_is_valid(value))
        || style.is_some_and(|value| !style_is_valid(value))
        || rgb.is_some_and(|value| !rgb_is_valid(value))
    {
        return true;
    }
    let Some(owner) = extra.get(PC111_KEY).copied() else {
        return false;
    };
    if !owner.is_finite() || !(0.0..=1.0).contains(&owner) {
        return true;
    }
    owner > 0.0 && amount.unwrap_or(0.0) > 0.0
}

/// Selects and decodes the real material for an active request.
///
/// Fast paths come first: no child keys, a zero/none parent owner, or a zero
/// amount all return `Ok(None)` before the job ticket and iris rings are
/// touched, so the historical PF111 path is unchanged. A zero amount keeps the
/// material selection but does not request a composite. Malformed amount,
/// style or RGB is a [`Refusal`], including when the parent owner is off. The
/// style ordinal is used directly; there is no "take the last discovered file"
/// fallback. An owner above 1 is refused rather than treated as extra strength.
pub(crate) fn prepare(
    root: Option<&Path>,
    bundle: &FaceInputBundle,
    extra: Option<&BTreeMap<String, f32>>,
) -> Result<Option<HumanCatchlight>, Refusal> {
    let Some(extra) = extra else {
        return Ok(None);
    };
    if !has_child_keys(extra) {
        return Ok(None);
    }

    let amount = parse_amount(extra.get(CATCHLIGHT_AMOUNT_KEY).copied())?;
    let style = parse_style(extra.get(CATCHLIGHT_STYLE_KEY).copied())?;
    let tint_rgb = parse_rgb(extra.get(CATCHLIGHT_RGB_KEY).copied())?;

    let Some(owner) = extra.get(PC111_KEY).copied() else {
        return Ok(None);
    };
    if !owner.is_finite() || !(0.0..=1.0).contains(&owner) {
        return Err(Refusal::ValueOutOfRange("catchlight pc:111"));
    }
    if owner == 0.0 {
        return Ok(None);
    }
    if amount == 0.0 {
        return Ok(None);
    }
    if style == 0 {
        return Err(Refusal::ValueOutOfRange("catchlight style"));
    }

    if bundle.job_id == 0 {
        return Err(Refusal::ValueOutOfRange("catchlight job ticket"));
    }
    if bundle.source_size[0] == 0 || bundle.source_size[1] == 0 {
        return Err(Refusal::ShapeMismatch("catchlight source size"));
    }
    if bundle.points194.len() != 194 {
        return Err(Refusal::ShapeMismatch("catchlight iris ring"));
    }
    let rings: [[[f32; 2]; CATCHLIGHT_RING_POINTS]; 2] = [
        std::array::from_fn(|index| bundle.points194[LEFT_IRIS[index]]),
        std::array::from_fn(|index| bundle.points194[RIGHT_IRIS[index]]),
    ];
    if rings
        .iter()
        .flatten()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("catchlight iris ring"));
    }

    let root = root.ok_or(Refusal::MissingInput("catchlight asset root"))?;
    let presets = crate::portrait_makeup_wiring::human_catchlights_at(root);
    let index = (style - 1) as usize;
    let preset = presets
        .get(index)
        .ok_or(Refusal::ValueOutOfRange("catchlight style"))?;
    let material = load_material(root, &preset.path)?;

    Ok(Some(HumanCatchlight {
        material,
        rings,
        amount,
        tint_rgb,
        params: CatchlightParams::default(),
        mirror_u: [false, preset.mirror_x],
        ticket: HumanCatchlightTicket {
            job_id: bundle.job_id,
            source_sha256: bundle.source_sha256,
            source_size: bundle.source_size,
            preset_id: preset.id.clone(),
            style_index: style,
        },
    }))
}

impl HumanCatchlight {
    /// Composites the prepared material onto `image`.
    ///
    /// The job ticket and the source dimensions must match the bundle this
    /// layer was prepared from. The original source digest is provenance for
    /// the prepared layer, not a check on the working frame: by design this
    /// sub-layer runs after the PF111 base iris and other stages, so working
    /// pixels have legitimately diverged from the source. Both eyes are
    /// validated by the compositor before the first write; only RGB inside the
    /// iris rings changes, and the frame alpha is never written. The return
    /// value is the compositor's write count, not a unique-pixel count.
    pub(crate) fn apply(&self, image: &mut Image, job: u64) -> Result<usize, Refusal> {
        if job == 0 || job != self.ticket.job_id {
            return Err(Refusal::ValueOutOfRange("catchlight job ticket"));
        }
        image.validate()?;
        if [image.width, image.height] != self.ticket.source_size {
            return Err(Refusal::ShapeMismatch("catchlight source size"));
        }
        let material = || CatchlightMaterial::StraightRgba8 {
            rgba8: &self.material.pixels,
            width: self.material.width,
            height: self.material.height,
        };
        let eyes = [
            Some(CatchlightEye {
                iris_ring: &self.rings[0],
                material: material(),
                highlight_amount: self.amount,
                mirror_u: self.mirror_u[0],
                tint_rgb: self.tint_rgb,
            }),
            Some(CatchlightEye {
                iris_ring: &self.rings[1],
                material: material(),
                highlight_amount: self.amount,
                mirror_u: self.mirror_u[1],
                tint_rgb: self.tint_rgb,
            }),
        ];
        let receipt = composite_catchlight_materials(image, eyes, self.params)?;
        Ok(receipt.touched_pixels)
    }

    /// Stable in-memory fingerprint of the prepared layer, including the
    /// material bytes, rings, amount/tint, ticket and style.
    pub(crate) fn fingerprint(&self) -> u64 {
        let mut hash = 0xCBF2_9CE4_8422_2325u64;
        fn mix(hash: &mut u64, bytes: &[u8]) {
            for &byte in bytes {
                *hash ^= byte as u64;
                *hash = (*hash).wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        mix(&mut hash, &self.material.width.to_le_bytes());
        mix(&mut hash, &self.material.height.to_le_bytes());
        mix(&mut hash, &self.material.pixels);
        for point in self.rings.iter().flatten() {
            mix(&mut hash, &point[0].to_bits().to_le_bytes());
            mix(&mut hash, &point[1].to_bits().to_le_bytes());
        }
        mix(&mut hash, &self.amount.to_bits().to_le_bytes());
        mix(&mut hash, &self.tint_rgb);
        mix(&mut hash, &self.params.inset.to_bits().to_le_bytes());
        mix(&mut hash, &self.params.feather.to_bits().to_le_bytes());
        mix(&mut hash, &[self.mirror_u[0] as u8, self.mirror_u[1] as u8]);
        mix(&mut hash, &self.ticket.job_id.to_le_bytes());
        mix(&mut hash, &self.ticket.source_sha256);
        mix(&mut hash, &self.ticket.source_size[0].to_le_bytes());
        mix(&mut hash, &self.ticket.source_size[1].to_le_bytes());
        mix(&mut hash, self.ticket.preset_id.as_bytes());
        mix(&mut hash, &self.ticket.style_index.to_le_bytes());
        hash
    }

    /// True only when `extra` names exactly the same retained material and
    /// render parameters as this prepared layer.
    ///
    /// The stage uses this before reusing a cached bundle: an old layer must
    /// not survive a style/amount/tint change. The job/source ticket is not in
    /// `extra` (it is enforced by [`Self::apply`]); this check is the material
    /// selection plus amount/tint, not a substitute for the ticket check.
    pub(crate) fn matches_request(&self, extra: &BTreeMap<String, f32>) -> bool {
        if !requested(extra) {
            return false;
        }
        let Ok(amount) = parse_amount(extra.get(CATCHLIGHT_AMOUNT_KEY).copied()) else {
            return false;
        };
        let Ok(style) = parse_style(extra.get(CATCHLIGHT_STYLE_KEY).copied()) else {
            return false;
        };
        let Ok(tint_rgb) = parse_rgb(extra.get(CATCHLIGHT_RGB_KEY).copied()) else {
            return false;
        };
        amount.to_bits() == self.amount.to_bits()
            && style == self.ticket.style_index
            && tint_rgb == self.tint_rgb
    }
}

fn parse_amount(value: Option<f32>) -> Result<f32, Refusal> {
    let Some(value) = value else { return Ok(0.0) };
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Refusal::ValueOutOfRange("catchlight amount"));
    }
    Ok(value)
}

fn parse_style(value: Option<f32>) -> Result<u32, Refusal> {
    let Some(value) = value else { return Ok(0) };
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > 1_000_000.0 {
        return Err(Refusal::ValueOutOfRange("catchlight style"));
    }
    Ok(value as u32)
}

fn parse_rgb(value: Option<f32>) -> Result<[u8; 3], Refusal> {
    let Some(value) = value else {
        return Ok([255, 255, 255]);
    };
    if !value.is_finite()
        || value.fract() != 0.0
        || !(0.0..=CATCHLIGHT_RGB_WHITE as f32).contains(&value)
    {
        return Err(Refusal::ValueOutOfRange("catchlight rgb"));
    }
    let packed = value as u32;
    Ok([
        ((packed >> 16) & 0xFF) as u8,
        ((packed >> 8) & 0xFF) as u8,
        (packed & 0xFF) as u8,
    ])
}

/// Reads one real material path and keeps straight RGBA8 bytes.
///
/// The path must stay inside the supplied makeup root (a symlink that escapes
/// is refused), be an 8-bit RGBA PNG (no palette/RGB/synthesized alpha, no
/// 16-bit downcast), have a sane side length, and carry a non-empty alpha
/// signal. The image is decoded once; the result is crate-private.
fn load_material(root: &Path, path: &Path) -> Result<Arc<Image>, Refusal> {
    let canonical_root = root
        .canonicalize()
        .map_err(|_| Refusal::MissingInput("catchlight asset root"))?;
    let canonical_path = path
        .canonicalize()
        .map_err(|_| Refusal::MissingInput("catchlight material"))?;
    if !canonical_path.starts_with(&canonical_root) {
        return Err(Refusal::ValueOutOfRange("catchlight material path"));
    }
    if !canonical_path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
    {
        return Err(Refusal::ShapeMismatch("catchlight material type"));
    }
    // Bound the file and inspect the 33-byte IHDR before reading/decoding the
    // whole PNG, so a huge or malformed material cannot allocate first.
    let mut file = std::fs::File::open(&canonical_path)
        .map_err(|_| Refusal::MissingInput("catchlight material"))?;
    let metadata = file
        .metadata()
        .map_err(|_| Refusal::MissingInput("catchlight material"))?;
    if metadata.len() > MATERIAL_MAX_FILE_BYTES {
        return Err(Refusal::ValueOutOfRange("catchlight material size"));
    }
    let mut header = [0u8; 33];
    file.read_exact(&mut header)
        .map_err(|_| Refusal::ShapeMismatch("catchlight material png"))?;
    drop(file);
    if &header[..8] != b"\x89PNG\r\n\x1a\n" {
        return Err(Refusal::ShapeMismatch("catchlight material png"));
    }
    let file_width = u32::from_be_bytes([header[16], header[17], header[18], header[19]]);
    let file_height = u32::from_be_bytes([header[20], header[21], header[22], header[23]]);
    let bit_depth = header[24];
    let color_type = header[25];
    if bit_depth != 8 || color_type != 6 {
        return Err(Refusal::ShapeMismatch("catchlight material type"));
    }
    if !(MATERIAL_MIN_SIDE..=MATERIAL_MAX_SIDE).contains(&file_width)
        || !(MATERIAL_MIN_SIDE..=MATERIAL_MAX_SIDE).contains(&file_height)
    {
        return Err(Refusal::ShapeMismatch("catchlight material size"));
    }
    let bytes =
        std::fs::read(&canonical_path).map_err(|_| Refusal::MissingInput("catchlight material"))?;
    if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err(Refusal::ShapeMismatch("catchlight material png"));
    }
    let decoded = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
        .map_err(|_| Refusal::ShapeMismatch("catchlight material png"))?
        .to_rgba8();
    let (width, height) = decoded.dimensions();
    if !(MATERIAL_MIN_SIDE..=MATERIAL_MAX_SIDE).contains(&width)
        || !(MATERIAL_MIN_SIDE..=MATERIAL_MAX_SIDE).contains(&height)
    {
        return Err(Refusal::ShapeMismatch("catchlight material size"));
    }
    let pixels = decoded.into_raw();
    if !pixels.chunks_exact(4).any(|texel| texel[3] != 0) {
        return Err(Refusal::ValueOutOfRange("catchlight material signal"));
    }
    Ok(Arc::new(Image {
        width,
        height,
        pixels,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NormRect;
    use crate::portrait_face_input_stage::frame_digest;
    use crate::portrait_makeup_bundle::test_support;
    use crate::portrait_stage::{
        C1FaceDetection, C1FaceMesh, FaceLandmarkProducerInputs, FaceModelIdentity,
        FaceNormalization, PortraitBackend, PortraitJobInputs, PortraitMasks, ProducerInputs,
        apply_portrait_effects_with_job,
    };
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_root() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "aurora-catchlight-layer-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).expect("temp root");
        path
    }

    fn write_material(root: &Path, rgba: &[u8], width: u32, height: u32) {
        let directory = root.join("Makeups").join("Catchlights").join("AUCG0001");
        std::fs::create_dir_all(&directory).expect("material directory");
        let material = image::RgbaImage::from_raw(width, height, rgba.to_vec()).expect("rgba");
        material
            .save_with_format(directory.join("ysg.png"), image::ImageFormat::Png)
            .expect("write png");
    }

    fn opaque_material() -> (Vec<u8>, u32, u32) {
        let mut rgba = Vec::new();
        for _ in 0..4 * 4 {
            rgba.extend_from_slice(&[220, 40, 20, 255]);
        }
        (rgba, 4, 4)
    }

    fn write_header_only_png(root: &Path, width: u32, height: u32) {
        let directory = root.join("Makeups").join("Catchlights").join("AUCG0001");
        std::fs::create_dir_all(&directory).expect("material directory");
        let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
        bytes.extend_from_slice(&13u32.to_be_bytes());
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[8, 6, 0, 0, 0]);
        bytes.extend_from_slice(&[0u8; 4]);
        std::fs::write(directory.join("ysg.png"), bytes).expect("header png");
    }

    fn ring(cx: f32, cy: f32, radius: f32) -> [[f32; 2]; CATCHLIGHT_RING_POINTS] {
        std::array::from_fn(|index| {
            let angle = index as f32 * std::f32::consts::TAU / CATCHLIGHT_RING_POINTS as f32;
            [cx + radius * angle.cos(), cy + radius * angle.sin()]
        })
    }

    fn frame() -> (Image, [u8; 32], Vec<[f32; 2]>) {
        let mut pixels = vec![0u8; 64 * 48 * 4];
        for texel in pixels.chunks_exact_mut(4) {
            texel[0] = 16;
            texel[1] = 32;
            texel[2] = 48;
            texel[3] = 255;
        }
        let image = Image::new(64, 48, pixels).expect("frame");
        let digest = frame_digest(&image);
        let mut points = vec![[0.0f32; 2]; 194];
        let left = ring(16.0, 16.0, 5.0);
        let right = ring(48.0, 16.0, 5.0);
        for index in 0..CATCHLIGHT_RING_POINTS {
            points[LEFT_IRIS[index]] = left[index];
            points[RIGHT_IRIS[index]] = right[index];
        }
        (image, digest, points)
    }

    fn frame_32x24() -> (Image, Vec<[f32; 2]>) {
        let image = Image::solid(32, 24, [24, 36, 48, 255]);
        let mut points = vec![[0.0f32; 2]; 194];
        let left = ring(10.0, 12.0, 4.0);
        let right = ring(22.0, 12.0, 4.0);
        for index in 0..CATCHLIGHT_RING_POINTS {
            points[LEFT_IRIS[index]] = left[index];
            points[RIGHT_IRIS[index]] = right[index];
        }
        (image, points)
    }

    fn bundle(job: u64, image: &Image, digest: [u8; 32], points: Vec<[f32; 2]>) -> FaceInputBundle {
        let normalization = FaceNormalization {
            channel_order: "rgb",
            mean: [0.0; 3],
            scale: [1.0; 3],
        };
        let model = FaceModelIdentity {
            model_id: "test",
            sha256: [0u8; 32],
        };
        FaceInputBundle {
            job_id: job,
            source_size: [image.width, image.height],
            source_sha256: digest,
            detection: C1FaceDetection {
                score: 1.0,
                landmarks_source: [[0.0; 2]; 5],
                source_to_input: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                input_size: [1, 1],
                normalization,
                provider: "test",
                model,
            },
            mesh: C1FaceMesh {
                score: 1.0,
                landmarks_source: vec![[0.0; 3]],
                source_to_crop: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                crop_side: 1.0,
                input_size: [1, 1],
                normalization,
                provider: "test",
                model,
            },
            roi_xywh: [0.0, 0.0, image.width as f32, image.height as f32],
            roi_normalized_xywh: [0.0, 0.0, 1.0, 1.0],
            points17: Vec::new(),
            points142: Vec::new(),
            points194: points,
            points414: Vec::new(),
            landmark_producer: FaceLandmarkProducerInputs {
                resource_id: 0,
                provider: "test".to_string(),
                model_sha256: [0u8; 32],
                source_sha256: digest,
                source_size: [image.width, image.height],
                points142: Vec::new(),
                crop_bgr: Vec::new(),
                crop_sha256: [0u8; 32],
                raw52: Vec::new(),
                points52: Vec::new(),
                points194: Vec::new(),
                fit_pre: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                fit_inverse: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            },
            model_inputs: None,
            bound_digest: None,
        }
    }

    fn extras(pairs: &[(&str, f32)]) -> BTreeMap<String, f32> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_string(), *value))
            .collect()
    }

    #[test]
    fn missing_root_is_refused_when_the_request_is_active() {
        let (image, digest, points) = frame();
        let bundle = bundle(7, &image, digest, points);
        let extra = extras(&[
            (PC111_KEY, 1.0),
            (CATCHLIGHT_AMOUNT_KEY, 1.0),
            (CATCHLIGHT_STYLE_KEY, 1.0),
        ]);
        assert_eq!(
            prepare(None, &bundle, Some(&extra)),
            Err(Refusal::MissingInput("catchlight asset root"))
        );
    }

    #[test]
    fn neutral_and_absent_requests_return_none() {
        let (image, digest, points) = frame();
        let bundle = bundle(7, &image, digest, points);
        assert_eq!(prepare(None, &bundle, None), Ok(None));
        assert_eq!(
            prepare(None, &bundle, Some(&extras(&[(PC111_KEY, 1.0)]))),
            Ok(None)
        );
        assert_eq!(
            prepare(
                None,
                &bundle,
                Some(&extras(&[
                    (PC111_KEY, 1.0),
                    (CATCHLIGHT_AMOUNT_KEY, 0.0),
                    (CATCHLIGHT_STYLE_KEY, 0.0),
                    (CATCHLIGHT_RGB_KEY, CATCHLIGHT_RGB_WHITE as f32),
                ])),
            ),
            Ok(None)
        );
        assert_eq!(
            prepare(None, &bundle, Some(&extras(&[(PC111_KEY, 0.0)]))),
            Ok(None)
        );
        // Parent active + valid style/colour + amount 0 is a retained
        // selection, not a request: the base PF111 iris still runs.
        assert_eq!(
            prepare(
                None,
                &bundle,
                Some(&extras(&[
                    (PC111_KEY, 1.0),
                    (CATCHLIGHT_AMOUNT_KEY, 0.0),
                    (CATCHLIGHT_STYLE_KEY, 1.0),
                    (CATCHLIGHT_RGB_KEY, CATCHLIGHT_RGB_WHITE as f32),
                ])),
            ),
            Ok(None)
        );
    }

    #[test]
    fn child_data_without_an_active_owner_does_not_request() {
        let (image, digest, points) = frame();
        let bundle = bundle(7, &image, digest, points);
        for extra in [
            extras(&[(CATCHLIGHT_AMOUNT_KEY, 0.5)]),
            extras(&[(PC111_KEY, 0.0), (CATCHLIGHT_STYLE_KEY, 1.0)]),
        ] {
            assert_eq!(prepare(None, &bundle, Some(&extra)), Ok(None));
        }
        // Malformed child data is still explicit even when the owner is off.
        for extra in [
            extras(&[(CATCHLIGHT_STYLE_KEY, 1.5)]),
            extras(&[(PC111_KEY, 0.0), (CATCHLIGHT_AMOUNT_KEY, 2.0)]),
        ] {
            assert!(prepare(None, &bundle, Some(&extra)).is_err());
        }
    }

    #[test]
    fn invalid_style_is_refused_not_clamped() {
        let (image, digest, points) = frame();
        let bundle = bundle(7, &image, digest, points);
        let root = temp_root();
        let (rgba, width, height) = opaque_material();
        write_material(&root, &rgba, width, height);
        for style in [2.0, 1.5, -1.0, 1_000_001.0] {
            let extra = extras(&[
                (PC111_KEY, 1.0),
                (CATCHLIGHT_AMOUNT_KEY, 1.0),
                (CATCHLIGHT_STYLE_KEY, style),
            ]);
            assert_eq!(
                prepare(Some(&root), &bundle, Some(&extra)),
                Err(Refusal::ValueOutOfRange("catchlight style")),
                "style {style}"
            );
        }
        let no_style = extras(&[(PC111_KEY, 1.0), (CATCHLIGHT_AMOUNT_KEY, 1.0)]);
        assert_eq!(
            prepare(Some(&root), &bundle, Some(&no_style)),
            Err(Refusal::ValueOutOfRange("catchlight style"))
        );
        let owner_too_high = extras(&[
            (PC111_KEY, 1.5),
            (CATCHLIGHT_AMOUNT_KEY, 1.0),
            (CATCHLIGHT_STYLE_KEY, 1.0),
        ]);
        assert_eq!(
            prepare(Some(&root), &bundle, Some(&owner_too_high)),
            Err(Refusal::ValueOutOfRange("catchlight pc:111"))
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn invalid_materials_are_refused() {
        let (image, digest, points) = frame();
        let bundle = bundle(7, &image, digest, points);
        let extra = extras(&[
            (PC111_KEY, 1.0),
            (CATCHLIGHT_AMOUNT_KEY, 1.0),
            (CATCHLIGHT_STYLE_KEY, 1.0),
        ]);

        let root = temp_root();
        write_material(&root, &[0u8; 16], 2, 2);
        assert_eq!(
            prepare(Some(&root), &bundle, Some(&extra)),
            Err(Refusal::ValueOutOfRange("catchlight material signal"))
        );

        write_material(&root, &[10, 20, 30, 255], 1, 1);
        assert_eq!(
            prepare(Some(&root), &bundle, Some(&extra)),
            Err(Refusal::ShapeMismatch("catchlight material size"))
        );

        let directory = root.join("Makeups").join("Catchlights").join("AUCG0001");
        std::fs::write(directory.join("ysg.png"), b"not a png").expect("bad material");
        assert_eq!(
            prepare(Some(&root), &bundle, Some(&extra)),
            Err(Refusal::ShapeMismatch("catchlight material png"))
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn oversized_material_header_is_refused_before_full_read() {
        let (image, digest, points) = frame();
        let bundle = bundle(7, &image, digest, points);
        let root = temp_root();
        write_header_only_png(&root, 100_000, 100_000);
        let extra = extras(&[
            (PC111_KEY, 1.0),
            (CATCHLIGHT_AMOUNT_KEY, 1.0),
            (CATCHLIGHT_STYLE_KEY, 1.0),
        ]);
        assert_eq!(
            prepare(Some(&root), &bundle, Some(&extra)),
            Err(Refusal::ShapeMismatch("catchlight material size"))
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn apply_binds_ticket_and_source_size() {
        let (image, digest, points) = frame();
        let bundle = bundle(7, &image, digest, points);
        let root = temp_root();
        let (rgba, width, height) = opaque_material();
        write_material(&root, &rgba, width, height);
        let extra = extras(&[
            (PC111_KEY, 1.0),
            (CATCHLIGHT_AMOUNT_KEY, 1.0),
            (CATCHLIGHT_STYLE_KEY, 1.0),
        ]);
        let layer = prepare(Some(&root), &bundle, Some(&extra))
            .expect("prepare")
            .expect("active layer");

        let mut wrong_job = image.clone();
        let before_wrong_job = wrong_job.pixels.clone();
        assert_eq!(
            layer.apply(&mut wrong_job, 8),
            Err(Refusal::ValueOutOfRange("catchlight job ticket"))
        );
        assert_eq!(
            wrong_job.pixels, before_wrong_job,
            "wrong ticket must not write"
        );

        let mut wrong_size = Image::solid(32, 24, [0, 0, 0, 255]);
        assert_eq!(
            layer.apply(&mut wrong_size, 7),
            Err(Refusal::ShapeMismatch("catchlight source size"))
        );

        // This layer runs after PF111 base iris/other stages: the working frame
        // legitimately differs from the source. Ticket + dimensions still bind.
        let mut working = image.clone();
        working.pixels[0] ^= 1;
        assert!(
            layer.apply(&mut working, 7).is_ok(),
            "a working frame changed by earlier stages must still apply"
        );

        let mut target = image.clone();
        let touched = layer.apply(&mut target, 7).expect("apply");
        assert!(touched > 0);
        assert_eq!(target.pixel(0, 0), image.pixel(0, 0));
        for (before, after) in image
            .pixels
            .chunks_exact(4)
            .zip(target.pixels.chunks_exact(4))
        {
            assert_eq!(before[3], after[3], "frame alpha must not change");
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn fingerprint_is_stable_and_amount_sensitive() {
        let (image, digest, points) = frame();
        let bundle = bundle(7, &image, digest, points);
        let root = temp_root();
        let (rgba, width, height) = opaque_material();
        write_material(&root, &rgba, width, height);
        let active = |amount: f32| {
            extras(&[
                (PC111_KEY, 1.0),
                (CATCHLIGHT_AMOUNT_KEY, amount),
                (CATCHLIGHT_STYLE_KEY, 1.0),
            ])
        };
        let first = prepare(Some(&root), &bundle, Some(&active(1.0)))
            .expect("prepare")
            .expect("layer");
        let second = prepare(Some(&root), &bundle, Some(&active(1.0)))
            .expect("prepare")
            .expect("layer");
        let weaker = prepare(Some(&root), &bundle, Some(&active(0.5)))
            .expect("prepare")
            .expect("layer");
        assert_eq!(first.fingerprint(), second.fingerprint());
        assert_ne!(first.fingerprint(), weaker.fingerprint());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn cpu_gate_accepts_only_neutral_or_cpu_values() {
        assert!(backend_allows_cpu(None));
        assert!(backend_allows_cpu(Some("")));
        assert!(backend_allows_cpu(Some("auto")));
        assert!(backend_allows_cpu(Some(" CPU ")));
        for explicit in ["cuda", "metal", "wgpu", "gpu", "directml", "opencl"] {
            assert!(!backend_allows_cpu(Some(explicit)), "{explicit}");
        }
    }

    #[test]
    fn matches_request_rejects_stale_material_or_params() {
        let (image, digest, points) = frame();
        let bundle = bundle(7, &image, digest, points);
        let root = temp_root();
        let (rgba, width, height) = opaque_material();
        write_material(&root, &rgba, width, height);
        let active = extras(&[
            (PC111_KEY, 1.0),
            (CATCHLIGHT_AMOUNT_KEY, 1.0),
            (CATCHLIGHT_STYLE_KEY, 1.0),
        ]);
        let layer = prepare(Some(&root), &bundle, Some(&active))
            .expect("prepare")
            .expect("layer");
        assert!(layer.matches_request(&active));
        for stale in [
            extras(&[
                (PC111_KEY, 1.0),
                (CATCHLIGHT_AMOUNT_KEY, 0.5),
                (CATCHLIGHT_STYLE_KEY, 1.0),
            ]),
            extras(&[
                (PC111_KEY, 1.0),
                (CATCHLIGHT_AMOUNT_KEY, 1.0),
                (CATCHLIGHT_STYLE_KEY, 2.0),
            ]),
            extras(&[
                (PC111_KEY, 1.0),
                (CATCHLIGHT_AMOUNT_KEY, 1.0),
                (CATCHLIGHT_STYLE_KEY, 1.0),
                (CATCHLIGHT_RGB_KEY, 0.0),
            ]),
            extras(&[
                (PC111_KEY, 1.0),
                (CATCHLIGHT_AMOUNT_KEY, 0.0),
                (CATCHLIGHT_STYLE_KEY, 1.0),
            ]),
            extras(&[
                (PC111_KEY, 0.0),
                (CATCHLIGHT_AMOUNT_KEY, 1.0),
                (CATCHLIGHT_STYLE_KEY, 1.0),
            ]),
        ] {
            assert!(!layer.matches_request(&stale));
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn requested_tracks_the_same_contract_as_prepare() {
        assert!(!requested(&BTreeMap::new()));
        assert!(!requested(&extras(&[(PC111_KEY, 1.0)])));
        assert!(!requested(&extras(&[
            (PC111_KEY, 1.0),
            (CATCHLIGHT_AMOUNT_KEY, 0.0),
            (CATCHLIGHT_STYLE_KEY, 0.0),
            (CATCHLIGHT_RGB_KEY, CATCHLIGHT_RGB_WHITE as f32),
        ])));
        // Parent active + valid style/colour + amount 0: retained selection,
        // not a request; the stage must keep the base PF111 iris.
        assert!(!requested(&extras(&[
            (PC111_KEY, 1.0),
            (CATCHLIGHT_AMOUNT_KEY, 0.0),
            (CATCHLIGHT_STYLE_KEY, 1.0),
            (CATCHLIGHT_RGB_KEY, CATCHLIGHT_RGB_WHITE as f32),
        ])));
        // Parent off is not a request even when the child values are active.
        assert!(!requested(&extras(&[
            (PC111_KEY, 0.0),
            (CATCHLIGHT_AMOUNT_KEY, 0.5),
            (CATCHLIGHT_STYLE_KEY, 1.0),
        ])));
        assert!(requested(&extras(&[
            (PC111_KEY, 1.0),
            (CATCHLIGHT_AMOUNT_KEY, 0.5),
            (CATCHLIGHT_STYLE_KEY, 1.0),
        ])));
        // A positive amount without a style is a request that `prepare`
        // refuses, not a silent no-op.
        assert!(requested(&extras(&[
            (PC111_KEY, 1.0),
            (CATCHLIGHT_AMOUNT_KEY, 1.0),
        ])));
        assert!(requested(&extras(&[
            (PC111_KEY, 1.0),
            (CATCHLIGHT_STYLE_KEY, 1.5)
        ])));
        assert!(requested(&extras(&[
            (PC111_KEY, 1.0),
            (CATCHLIGHT_RGB_KEY, 16_777_216.0),
        ])));
        assert!(requested(&extras(&[
            (PC111_KEY, 1.0),
            (CATCHLIGHT_AMOUNT_KEY, f32::NAN),
        ])));
        // Parent above 1 is malformed, not extra strength.
        assert!(requested(&extras(&[
            (PC111_KEY, 1.5),
            (CATCHLIGHT_AMOUNT_KEY, 0.5),
            (CATCHLIGHT_STYLE_KEY, 1.0),
        ])));
        // Malformed child data is explicit even with an inactive owner.
        assert!(requested(&extras(&[(CATCHLIGHT_STYLE_KEY, 1.5)])));
    }

    /// Use generated geometry to exercise the real same-frame material producers
    /// and dispatcher without embedding a private face, model or original artwork.
    /// This is an integration oracle for the product, not an OEM comparison.
    #[test]
    #[ignore = "requires the full original look library and AURORA_PORTRAIT_RENDER_BACKEND=cpu"]
    fn installed_looks_render_all_selected_rows_on_a_controlled_face() {
        use std::collections::BTreeSet;
        assert_eq!(
            std::env::var("AURORA_PORTRAIT_RENDER_BACKEND").as_deref(),
            Ok("cpu")
        );
        let (w, h) = (256u32, 224u32);
        let source = Image::new(
            w,
            h,
            (0..w * h)
                .flat_map(|i| {
                    let (x, y) = (i % w, i / w);
                    [
                        110 + ((x + y) % 78) as u8,
                        85 + (x % 55) as u8,
                        80 + (y % 48) as u8,
                        255,
                    ]
                })
                .collect(),
        )
        .expect("generated chart");
        let mut points = vec![[128.0, 116.0]; 194];
        let mut ellipse = |start, len: usize, cx, cy, rx, ry| {
            for (index, p) in points[start..start + len].iter_mut().enumerate() {
                let a = index as f32 * std::f32::consts::TAU / len as f32;
                *p = [cx + rx * a.cos(), cy + ry * a.sin()];
            }
        };
        ellipse(0, 9, 85., 67., 20., 5.);
        ellipse(9, 9, 171., 67., 20., 5.);
        ellipse(18, 8, 85., 91., 18., 7.);
        ellipse(36, 8, 171., 91., 18., 7.);
        ellipse(27, 8, 85., 91., 6., 6.);
        ellipse(45, 8, 171., 91., 6., 6.);
        ellipse(73, 23, 128., 153., 25., 9.);
        ellipse(96, 11, 128., 153., 16., 3.);
        ellipse(109, 33, 128., 112., 80., 91.);
        let job = 73;
        let mut face = bundle(job, &source, frame_digest(&source), points);
        face.points414 = vec![[128., 112.]; 414];
        let plane: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                if ((x - 128.) / 81.).powi(2) + ((y - 112.) / 92.).powi(2) < 1.0 {
                    255
                } else {
                    0
                }
            })
            .collect();
        let suits = crate::makeup_suits();
        assert_eq!(suits.len(), 24);
        let masks = PortraitMasks::solid(1.0);
        let mut unique = BTreeSet::new();
        let mut changed_range = (usize::MAX, 0usize);
        let mut calls = 0;
        for suit in suits {
            let mut adj = crate::AdjustmentSet::default();
            crate::apply_makeup_suit(&suit, &mut adj)
                .unwrap_or_else(|e| panic!("{}: {e}", suit.id));
            let expected: BTreeSet<u32> = adj
                .extra
                .iter()
                .filter(|(_, v)| **v > 0.0)
                .filter_map(|(key, _)| key.strip_prefix("pc:")?.parse().ok())
                .collect();
            let b3: Vec<_> = expected.iter().copied().filter(|pf| *pf != 101).collect();
            let mut producer = ProducerInputs::default();
            face.attach_makeup_bundle_with_options(
                &source,
                &b3,
                Some(&plane),
                Some(&adj.extra),
                &mut producer,
            )
            .unwrap_or_else(|err| panic!("{} preparing material rows: {err:?}", suit.id));
            if expected.contains(&101) {
                face.attach_pf101_from_adjustments(
                    &source,
                    adj.extra["pc:101"],
                    &adj.extra,
                    &mut producer,
                )
                .unwrap_or_else(|err| panic!("{} preparing PF101: {err:?}", suit.id));
            }
            let mut inputs = PortraitJobInputs::from_renderer_job(job, &masks);
            inputs.fill_from_producer_inputs(&producer);
            let mut output = source.pixels.clone();
            let report = apply_portrait_effects_with_job(
                &mut output,
                w,
                h,
                NormRect::FULL,
                &adj.extra,
                &inputs,
            );
            assert_eq!(
                report.effects.iter().map(|r| r.pf).collect::<BTreeSet<_>>(),
                expected,
                "{}",
                suit.id
            );
            assert_eq!(
                report.effects.len(),
                expected.len(),
                "duplicate row: {}",
                suit.id
            );
            for r in report.effects {
                assert!(
                    r.executed && r.backend == PortraitBackend::Cpu && r.refusal.is_none(),
                    "{} PF{}: {:?}",
                    suit.id,
                    r.pf,
                    r
                );
                calls += 1;
            }
            let mut changed = 0;
            for (i, (a, b)) in source
                .pixels
                .chunks_exact(4)
                .zip(output.chunks_exact(4))
                .enumerate()
            {
                assert_eq!(a[3], b[3]);
                if a != b {
                    changed += 1;
                    assert_ne!(plane[i], 0, "{} changed the background", suit.id);
                }
            }
            assert!(changed > 0, "{} is a no-op", suit.id);
            changed_range = (changed_range.0.min(changed), changed_range.1.max(changed));
            assert!(unique.insert(output), "duplicate look output: {}", suit.id);
        }
        eprintln!(
            "installed-look render: {} distinct outputs; {calls} CPU stage rows; {}..{} changed pixels per 256x224 controlled frame",
            unique.len(),
            changed_range.0,
            changed_range.1
        );
    }

    #[test]
    fn pf111_catchlight_ab_changes_only_irises_and_reports_cpu() {
        // `complete_owned_bundle_for_job` is the existing fixed 32x24
        // same-job base fixture (explicitly a test fixture, not a production
        // attach): the production attach path reads AURORA_MAKEUP_ASSET_ROOT,
        // and this test intentionally avoids mutating process-global env.
        let (source, points) = frame_32x24();
        let root = temp_root();
        let (rgba, width, height) = opaque_material();
        write_material(&root, &rgba, width, height);
        let active = extras(&[
            (PC111_KEY, 0.8),
            (CATCHLIGHT_AMOUNT_KEY, 1.0),
            (CATCHLIGHT_STYLE_KEY, 1.0),
        ]);
        let bundle = bundle(7, &source, [0x51; 32], points);
        let layer = prepare(Some(&root), &bundle, Some(&active))
            .expect("prepare")
            .expect("active layer");
        let mut base = test_support::complete_owned_bundle_for_job(7);
        base.human_catchlight = Some(layer);
        let producer = ProducerInputs {
            makeup_bundle: Some(base),
            ..ProducerInputs::default()
        };
        let masks = PortraitMasks::solid(1.0);
        let mut inputs = PortraitJobInputs::from_renderer_job(7, &masks);
        inputs.fill_from_producer_inputs(&producer);

        let reference_extra = extras(&[(PC111_KEY, 0.8)]);
        let mut reference = source.pixels.clone();
        let _reference_report = apply_portrait_effects_with_job(
            &mut reference,
            32,
            24,
            NormRect::FULL,
            &reference_extra,
            &inputs,
        );
        let mut candidate = source.pixels.clone();
        let candidate_report = apply_portrait_effects_with_job(
            &mut candidate,
            32,
            24,
            NormRect::FULL,
            &active,
            &inputs,
        );
        let run = candidate_report
            .effects
            .iter()
            .find(|effect| effect.pf == 111)
            .expect("PF111 run");
        assert_eq!(run.backend, PortraitBackend::Cpu);
        assert!(run.executed, "PF111 catchlight row must execute");
        assert_eq!(run.refusal, None);
        assert_ne!(reference, candidate, "catchlight must change the frame");

        for index in (3..source.pixels.len()).step_by(4) {
            assert_eq!(
                reference[index], candidate[index],
                "catchlight must not change alpha relative to the base PF111 frame"
            );
        }
        let in_iris_box = |x: u32, y: u32| {
            let near =
                |cx: f32, cy: f32| (x as f32 - cx).abs() <= 5.0 && (y as f32 - cy).abs() <= 5.0;
            near(10.0, 12.0) || near(22.0, 12.0)
        };
        let mut changed = 0usize;
        for y in 0..24u32 {
            for x in 0..32u32 {
                let index = (y as usize * 32 + x as usize) * 4;
                if candidate[index..index + 4] != reference[index..index + 4] {
                    changed += 1;
                    assert!(
                        in_iris_box(x, y),
                        "catchlight changed a pixel outside both irises at {x},{y}"
                    );
                }
            }
        }
        assert!(changed > 0, "catchlight made no visible change");

        let mut wrong_inputs = PortraitJobInputs::from_renderer_job(8, &masks);
        wrong_inputs.fill_from_producer_inputs(&producer);
        let mut wrong_frame = source.pixels.clone();
        let before_wrong = wrong_frame.clone();
        let wrong_report = apply_portrait_effects_with_job(
            &mut wrong_frame,
            32,
            24,
            NormRect::FULL,
            &active,
            &wrong_inputs,
        );
        let wrong_run = wrong_report
            .effects
            .iter()
            .find(|effect| effect.pf == 111)
            .expect("wrong-ticket PF111 run");
        assert_eq!(wrong_frame, before_wrong, "wrong ticket must not write");
        assert!(wrong_run.refusal.is_some());
        std::fs::remove_dir_all(root).ok();
    }
}
