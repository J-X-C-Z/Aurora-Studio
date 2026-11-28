//! Local body-pose/keypoint ONNX discovery and inference.
//!
//! A model is accepted only when both the ONNX file and its expected SHA-256
//! are local and explicit. Missing models or digests are typed errors; this
//! module never substitutes landmarks, a mesh, or a zeroed pose.

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use aurora_portrait::kernels::body_pose_adapter::{
    adopt, AnchorSet, ModelLayout, ObservedKeypoint,
};
use aurora_portrait::{Image, Refusal};

use super::ort::{Runtime, Session, TensorMeta};

pub const BODY_POSE_ACQUISITION_REFUSAL: &str = "body pose model acquisition";
pub const BODY_POSE_SHA256_REFUSAL: &str = "body pose model SHA-256 manifest";
pub const BODY_POSE_TENSOR_CONTRACT_REFUSAL: &str = "body pose tensor contract";
/// The model's K has no recovered keypoint layout, so no semantic anchor may be
/// claimed for it. Refused as a whole instead of guessing a table.
pub const BODY_POSE_SEMANTIC_LAYOUT_REFUSAL: &str = "body pose semantic keypoint layout";
/// The native 69-slot model still has no recovered vendor slot order
/// (`SlotTable` is all `None`), so its raw output order must never be read as
/// semantics. Refused as a whole.
pub const BODY_POSE_NATIVE_SEMANTICS_REFUSAL: &str =
    "body pose 69-slot semantics (vendor order not recovered)";
/// A MediaPipe tensor without the visibility/presence channels cannot prove a
/// point is visible, so this module does not hand out anchors for it.
pub const BODY_POSE_VISIBILITY_REFUSAL: &str =
    "body pose mediapipe visibility (3-channel tensor carries no visibility/presence)";
pub const BODY_POSE_KEYPOINT_COUNT: usize = 69;
/// Smallest keypoint count this loader accepts from a non-native model.
/// The 69-slot ABI stays fail-closed: `BodyPoseInference::landmarks()` still
/// refuses anything other than exactly `BODY_POSE_KEYPOINT_COUNT`; an adapter
/// owns the K -> 69 mapping and must be identified in the inference receipt.
pub const BODY_POSE_MIN_KEYPOINTS: usize = 17;
/// MediaPipe Pose 的固定点数：只有这个 K 的 5 通道才按
/// `x,y,z,visibility,presence` 解析。
const MEDIAPIPE_POSE_KEYPOINTS: i64 = 33;
/// Upper bound accepted from a single pose tensor, so a malformed shape cannot
/// make the parser allocate an unreasonable keypoint vector.
pub const BODY_POSE_MAX_KEYPOINTS: usize = 256;

const MODEL_NAMES: [&str; 5] = [
    "bodyPose.onnx",
    "body_pose.onnx",
    "bodyKeypoints.onnx",
    "humanPose.onnx",
    "pose.onnx",
];

const MODEL_DIRS: [&str; 4] = [
    "models/AIModels/Common/BodyPose",
    "AIModels/Common/BodyPose",
    "models/BodyPose",
    "resources/body-pose",
];

/// Explicit model file. Highest priority; its `.sha256` sidecar is required.
pub const BODY_POSE_MODEL_ENV: &str = "AURORA_BODY_POSE_MODEL";
/// Directory holding one of [`MODEL_NAMES`] plus its `.sha256` sidecar.
pub const BODY_POSE_DIR_ENV: &str = "AURORA_BODY_POSE_DIR";
/// Executable-relative directories tried before the cwd-relative ones.
const EXE_MODEL_DIRS: [&str; 4] = [
    "models/bodypose",
    "assets/portrait/bodypose",
    "models/BodyPose",
    "bodypose",
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyKeypoint {
    pub x: f32,
    pub y: f32,
    pub score: f32,
}

/// What the third float of a keypoint tensor actually means.
///
/// The loader accepts `[1,K,2|3]` and MediaPipe's `[1,33,5]`, so a 3-channel
/// tensor can be either `(x, y, score)` — COCO/Halpe/OpenPose/WholeBody exports —
/// or MediaPipe Pose's `(x, y, z)`, where the third channel is **depth, not
/// confidence**. Reading z as a score would silently drop or keep the wrong
/// points, so the channel is decided by the model layout, never by the shape
/// alone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PoseThirdChannel {
    /// No third channel at all (`[1,K,2]`): the adapter's `1.0` convention
    /// applies and presence comes from the geometry only.
    #[default]
    Absent,
    /// Real per-keypoint confidence; usable for the `min_score` gate.
    Score,
    /// MediaPipe depth. Never used as confidence.
    Depth,
}

/// The confidence/visibility channels a model really emitted.
///
/// `visibility` and `presence` are MediaPipe Pose's fourth and fifth channels
/// (`x, y, z, visibility, presence`). They are kept as their own planes so the
/// semantic layer can require both instead of inventing a combined score, and so
/// a missing signal can never be mistaken for "fully visible".
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PoseChannels {
    pub third: PoseThirdChannel,
    /// Per-keypoint visibility probability, when the model emits it.
    pub visibility: Option<Vec<f32>>,
    /// Per-keypoint presence probability, when the model emits it.
    pub presence: Option<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BodyPoseInference {
    pub model_sha256: [u8; 32],
    pub provider: String,
    pub frame_size: (u32, u32),
    pub source_sha256: [u8; 32],
    pub input_size: (u32, u32),
    pub tensor_contract: BodyPoseTensorContract,
    pub keypoints: Vec<BodyKeypoint>,
    /// Channel planes as parsed from the model output, keyed by meaning.
    pub channels: PoseChannels,
}

impl BodyPoseInference {
    pub fn landmarks(
        &self,
    ) -> Result<Vec<aurora_portrait::kernels::body_geometry::Vec2>, Refusal> {
        if self.tensor_contract.keypoints != BODY_POSE_KEYPOINT_COUNT
            || self.keypoints.len() != BODY_POSE_KEYPOINT_COUNT
        {
            return Err(Refusal::ShapeMismatch("body pose 69-point ABI"));
        }
        Ok(self
            .keypoints
            .iter()
            .map(|point| aurora_portrait::kernels::body_geometry::Vec2::new(point.x, point.y))
            .collect())
    }

    /// The recovered keypoint layout this model's `K` belongs to.
    ///
    /// The layout is a property of the **model**, so it is decided here from the
    /// explicit tensor contract — never from a caller's guess and never from the
    /// channel count. `K` values with no recovered layout are refused as a whole
    /// rather than mapped through a neighbouring table.
    pub fn semantic_layout(&self) -> Result<ModelLayout, Refusal> {
        match self.tensor_contract.keypoints {
            17 => Ok(ModelLayout::Coco17),
            25 => Ok(ModelLayout::OpenPose25),
            26 => Ok(ModelLayout::Halpe26),
            33 => Ok(ModelLayout::MediaPipe33),
            133 => Ok(ModelLayout::WholeBody133Body),
            // Native 69: the vendor slot order is still unrecovered, so this
            // model's raw output order must not be read as semantics.
            BODY_POSE_KEYPOINT_COUNT => Err(Refusal::MissingInput(
                BODY_POSE_NATIVE_SEMANTICS_REFUSAL,
            )),
            _ => Err(Refusal::MissingInput(BODY_POSE_SEMANTIC_LAYOUT_REFUSAL)),
        }
    }

    /// What the parsed model tensor actually supplied.
    pub fn third_channel(&self) -> PoseThirdChannel {
        self.channels.third
    }

    /// Semantic anchors for the lead/mask producers.
    ///
    /// Frozen interface: `semantic_anchors(min_score) -> Result<AnchorSet, Refusal>`.
    /// Everything is delegated to the canonical
    /// [`aurora_portrait::kernels::body_pose_adapter::adopt`], so this module owns
    /// no keypoint table of its own. The admissibility rules are the adapter's:
    /// present, finite `x`/`y`/`score`, `x`/`y ∈ [0,1]`, `score ≥ min_score`;
    /// anything else becomes `None` — **no zero filling**.
    ///
    /// Channel semantics (see [`PoseChannels`]):
    /// * `visibility` + `presence` planes (MediaPipe `[1,33,5]`) — a point enters
    ///   only when both are finite and inside `[0,1]` and both are `≥ min_score`;
    ///   the reported score is `min(visibility, presence)`. Depth is never read as
    ///   confidence, and a missing signal is never treated as "visible".
    /// * a real `score` channel — gated on that score; `0` means absence.
    /// * neither (MediaPipe `[1,33,3]`) — refused with
    ///   [`BODY_POSE_VISIBILITY_REFUSAL`]: visibility cannot be proven from the
    ///   tensor, so the caller must supply an explicit observation instead.
    ///
    /// The model identity stays on the inference itself — `model_sha256`,
    /// `provider`, `tensor_contract` and `source_sha256` — so a receipt can name
    /// the exact model/provider that produced these anchors.
    pub fn semantic_anchors(&self, min_score: f32) -> Result<AnchorSet, Refusal> {
        let layout = self.semantic_layout()?;
        self.anchors_for(layout, min_score)
    }

    /// Same result, with the layout declared by the caller.
    ///
    /// Only for callers that hold real model evidence (a manifest naming the
    /// export); the declaration is still validated against the tensor's `K`, so a
    /// wrong claim is a `ShapeMismatch`, never a silent reinterpretation. This is
    /// the whole "trusted layout" escape hatch: no auto-detection, no extra table.
    pub fn semantic_anchors_with_layout(
        &self,
        layout: ModelLayout,
        min_score: f32,
    ) -> Result<AnchorSet, Refusal> {
        if layout.model_point_count() as usize != self.tensor_contract.keypoints {
            return Err(Refusal::ShapeMismatch("body pose layout override"));
        }
        self.anchors_for(layout, min_score)
    }

    fn anchors_for(&self, layout: ModelLayout, min_score: f32) -> Result<AnchorSet, Refusal> {
        if self.keypoints.len() != self.tensor_contract.keypoints {
            return Err(Refusal::ShapeMismatch("body pose keypoint vector"));
        }
        let third = self.third_channel();
        let visibility = self.channels.visibility.as_deref();
        let presence = self.channels.presence.as_deref();
        for plane in [visibility, presence].into_iter().flatten() {
            if plane.len() != self.keypoints.len() {
                return Err(Refusal::ShapeMismatch("body pose visibility plane"));
            }
        }
        // MediaPipe without visibility/presence: cannot prove any point is
        // visible, so it refuses rather than assuming everything is visible.
        if layout == ModelLayout::MediaPipe33 && visibility.is_none() {
            return Err(Refusal::MissingInput(BODY_POSE_VISIBILITY_REFUSAL));
        }
        // A half signal is not evidence either: presence without visibility (or
        // the other way round) must not silently degrade to "geometry only".
        if visibility.is_some() != presence.is_some() {
            return Err(Refusal::MissingInput(BODY_POSE_VISIBILITY_REFUSAL));
        }
        if visibility.is_none() && third != PoseThirdChannel::Score {
            return Err(Refusal::MissingInput("body pose landmark confidence"));
        }
        let observed: Vec<ObservedKeypoint> = self
            .keypoints
            .iter()
            .enumerate()
            .map(|(index, point)| {
                let in_frame = point.x.is_finite()
                    && point.y.is_finite()
                    && (0.0..=1.0).contains(&point.x)
                    && (0.0..=1.0).contains(&point.y);
                match (visibility, presence) {
                    (Some(visibility), Some(presence)) => {
                        let visible = visibility[index];
                        let present = presence[index];
                        // Out-of-range or non-finite probabilities are malformed
                        // channels, not evidence: that point is absent.
                        let real = visible.is_finite()
                            && present.is_finite()
                            && (0.0..=1.0).contains(&visible)
                            && (0.0..=1.0).contains(&present);
                        ObservedKeypoint {
                            x: point.x,
                            y: point.y,
                            score: if real { visible.min(present) } else { 0.0 },
                            present: in_frame && real,
                        }
                    }
                    _ => match third {
                        PoseThirdChannel::Score => ObservedKeypoint {
                            x: point.x,
                            y: point.y,
                            score: point.score,
                            // A zero score is how these exporters spell "not this
                            // instance"; it is absence, not a point at the origin.
                            present: in_frame && point.score.is_finite() && point.score > 0.0,
                        },
                        _ => ObservedKeypoint {
                            x: point.x,
                            y: point.y,
                            score: 1.0,
                            present: in_frame,
                        },
                    },
                }
            })
            .collect();
        adopt(layout, &observed, min_score)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BodyPoseModelSource {
    pub path: PathBuf,
    pub sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyPoseOutputLayout {
    KeypointXyz,
    XyzKeypoint,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BodyPoseTensorContract {
    pub input_name: String,
    pub input_dims: [i64; 4],
    pub input_element_type: i32,
    pub output_name: String,
    pub output_dims: Vec<i64>,
    pub output_element_type: i32,
    pub output_layout: BodyPoseOutputLayout,
    pub keypoints: usize,
}

pub struct BodyPoseModel {
    path: PathBuf,
    sha256: [u8; 32],
    session: Session,
    input_size: (u32, u32),
    contract: BodyPoseTensorContract,
}

impl BodyPoseModel {
    pub fn from_local() -> Result<Self> {
        let source = discover().map_err(|refusal| anyhow::anyhow!("{refusal:?}"))?;
        Self::load(source.path, source.sha256)
    }

    pub fn load(path: impl AsRef<Path>, expected_sha256: [u8; 32]) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        ensure!(
            path.is_file(),
            "body pose model is missing: {}",
            path.display()
        );
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read body pose model {}", path.display()))?;
        let sha256 = crate::sha256_32(&bytes);
        ensure!(
            sha256 == expected_sha256,
            "body pose model SHA-256 mismatch: got {}, expected {}",
            hex(&sha256),
            hex(&expected_sha256)
        );
        drop(bytes);

        let runtime = Runtime::get().context("ONNX Runtime is not available for body pose")?;
        let session = runtime
            .open_session(&path)
            .with_context(|| format!("open body pose model {}", path.display()))?;
        ensure!(
            session.input.dims.len() == 4
                && session.input.dims[0] == 1
                && session.input.dims[1] == 3
                && session.input.dims[2] > 0
                && session.input.dims[3] > 0
                && session.input.element_type == 1,
            "{BODY_POSE_TENSOR_CONTRACT_REFUSAL}: expects fixed float32 NCHW [1,3,H,W], got {:?} type {}",
            session.input.dims,
            session.input.element_type
        );
        let input_size = input_size(&session.input.dims)?;
        let (output, output_layout, keypoints) =
            keypoint_contract_any(&session.outputs).context(BODY_POSE_TENSOR_CONTRACT_REFUSAL)?;
        let contract = BodyPoseTensorContract {
            input_name: session.input.name.clone(),
            input_dims: session
                .input
                .dims
                .as_slice()
                .try_into()
                .expect("validated four input dimensions"),
            input_element_type: session.input.element_type,
            output_name: output.name.clone(),
            output_dims: output.dims.clone(),
            output_element_type: output.element_type,
            output_layout,
            keypoints,
        };

        Ok(Self {
            path,
            sha256,
            session,
            input_size,
            contract,
        })
    }

    pub fn model_path(&self) -> &Path {
        &self.path
    }

    pub fn model_sha256(&self) -> [u8; 32] {
        self.sha256
    }

    pub fn input_size(&self) -> (u32, u32) {
        self.input_size
    }

    pub fn tensor_contract(&self) -> &BodyPoseTensorContract {
        &self.contract
    }

    pub fn infer(&self, source: &Image) -> Result<BodyPoseInference> {
        source
            .validate()
            .map_err(|refusal| anyhow::anyhow!("body pose source: {refusal:?}"))?;
        let (input, transform) = prepare(source, self.input_size)?;
        let mut input = input;
        let mut outputs = Runtime::get()
            .context("ONNX Runtime is not available for body pose")?
            .run(
                &self.session,
                &mut input,
                &[1, 3, self.input_size.1 as i64, self.input_size.0 as i64],
            )?;

        let mut parsed = None;
        for (plane, meta) in outputs.drain(..).zip(&self.session.outputs) {
            if meta.name != self.contract.output_name || meta.element_type != 1 {
                continue;
            }
            if let Some(layout) =
                output_layout(meta.dims.as_slice(), plane.len(), self.contract.keypoints)
            {
                parsed = Some(parse_keypoints(
                    &plane,
                    layout,
                    self.contract.keypoints,
                    transform,
                    source.width,
                    source.height,
                )?);
                break;
            }
        }
        let (keypoints, channels) =
            parsed.ok_or_else(|| anyhow::anyhow!("body pose output shape mismatch"))?;
        Ok(BodyPoseInference {
            model_sha256: self.sha256,
            provider: self.session.provider().label().to_string(),
            frame_size: (source.width, source.height),
            source_sha256: crate::sha256_32(&source.pixels),
            input_size: self.input_size,
            tensor_contract: self.contract.clone(),
            keypoints,
            channels,
        })
    }
}

pub fn discover() -> std::result::Result<BodyPoseModelSource, Refusal> {
    let path = model_candidates()
        .into_iter()
        .find(|candidate| candidate.is_file())
        .ok_or(Refusal::MissingInput(BODY_POSE_ACQUISITION_REFUSAL))?;
    let sha256 = sha256_sidecar(&path).ok_or(Refusal::MissingInput(BODY_POSE_SHA256_REFUSAL))?;
    Ok(BodyPoseModelSource { path, sha256 })
}

fn model_candidates() -> Vec<PathBuf> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    model_candidates_from(
        exe_dir,
        std::env::var_os(BODY_POSE_MODEL_ENV).map(PathBuf::from),
        std::env::var_os(BODY_POSE_DIR_ENV).map(PathBuf::from),
        std::env::current_dir().ok(),
    )
}

/// Deterministic candidate order: explicit file, explicit directory,
/// executable-relative install layouts, dev `../../models`, then cwd-relative
/// historical directories. Duplicates collapse to their first position.
fn model_candidates_from(
    exe_dir: Option<PathBuf>,
    env_model: Option<PathBuf>,
    env_dir: Option<PathBuf>,
    cwd: Option<PathBuf>,
) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let mut push = |path: PathBuf| {
        if !candidates.iter().any(|existing| *existing == path) {
            candidates.push(path);
        }
    };
    if let Some(path) = env_model {
        push(path);
    }
    if let Some(dir) = env_dir {
        for name in MODEL_NAMES {
            push(dir.join(name));
        }
    }
    if let Some(dir) = exe_dir {
        for sub in EXE_MODEL_DIRS {
            for name in MODEL_NAMES {
                push(dir.join(sub).join(name));
            }
        }
        for sub in ["models/bodypose", "assets/portrait/bodypose"] {
            for name in MODEL_NAMES {
                push(dir.join("..").join("..").join(sub).join(name));
            }
        }
    }
    for root in MODEL_DIRS {
        let root = PathBuf::from(root);
        for name in MODEL_NAMES {
            push(root.join(name));
        }
    }
    if let Some(cwd) = cwd {
        for sub in ["models/bodypose", "assets/portrait/bodypose"] {
            for name in MODEL_NAMES {
                push(cwd.join(sub).join(name));
            }
        }
    }
    candidates
}

fn sha256_sidecar(path: &Path) -> Option<[u8; 32]> {
    let mut sidecar = path.as_os_str().to_os_string();
    sidecar.push(".sha256");
    let text = std::fs::read_to_string(PathBuf::from(sidecar)).ok()?;
    decode_hex(text.split_whitespace().next()?).ok()
}

fn input_size(dims: &[i64]) -> Result<(u32, u32)> {
    ensure!(
        dims.len() == 4 && dims[2] > 0 && dims[3] > 0,
        "{BODY_POSE_TENSOR_CONTRACT_REFUSAL}: input H/W must be fixed"
    );
    Ok((dims[3] as u32, dims[2] as u32))
}

#[derive(Clone, Copy)]
struct Letterbox {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    input_w: f32,
    input_h: f32,
}

fn prepare(source: &Image, target: (u32, u32)) -> Result<(Vec<f32>, Letterbox)> {
    if target.0 == 0 || target.1 == 0 {
        bail!("body pose input size must be nonzero");
    }
    let scale = (target.0 as f32 / source.width as f32).min(target.1 as f32 / source.height as f32);
    let resized_w = (source.width as f32 * scale).round() as u32;
    let resized_h = (source.height as f32 * scale).round() as u32;
    let pad_x = (target.0 - resized_w) / 2;
    let pad_y = (target.1 - resized_h) / 2;
    let plane = target.0 as usize * target.1 as usize;
    let mut tensor = vec![0.0f32; plane * 3];
    for y in 0..resized_h {
        let source_y = ((y as f32 + 0.5) / scale - 0.5).clamp(0.0, (source.height - 1) as f32);
        let y0 = source_y.floor() as u32;
        let y1 = (y0 + 1).min(source.height - 1);
        let ty = source_y - y0 as f32;
        for x in 0..resized_w {
            let source_x = ((x as f32 + 0.5) / scale - 0.5).clamp(0.0, (source.width - 1) as f32);
            let x0 = source_x.floor() as u32;
            let x1 = (x0 + 1).min(source.width - 1);
            let tx = source_x - x0 as f32;
            let target_x = x + pad_x;
            let target_y = y + pad_y;
            for channel in 0..3 {
                let at = |px: u32, py: u32| {
                    source.pixels[((py * source.width + px) * 4) as usize + channel] as f32 / 255.0
                };
                let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
                let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
                let value = top + (bottom - top) * ty;
                tensor
                    [channel * plane + target_y as usize * target.0 as usize + target_x as usize] =
                    value;
            }
        }
    }
    Ok((
        tensor,
        Letterbox {
            scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
            input_w: target.0 as f32,
            input_h: target.1 as f32,
        },
    ))
}

/// Accept any static `[1,K,2|3]` / `[1,2|3,K]` keypoint tensor whose `K` lies in
/// `BODY_POSE_MIN_KEYPOINTS..=BODY_POSE_MAX_KEYPOINTS`. `K` may differ from the
/// native 69: the caller is then responsible for an explicit K -> 69 adapter and
/// for carrying `keypoint_semantics_version` in its receipt. Nothing here
/// synthesizes missing slots.
fn keypoint_contract_any(
    outputs: &[TensorMeta],
) -> Option<(&TensorMeta, BodyPoseOutputLayout, usize)> {
    let min = BODY_POSE_MIN_KEYPOINTS as i64;
    let max = BODY_POSE_MAX_KEYPOINTS as i64;
    outputs.iter().find_map(|output| {
        if output.element_type != 1 || output.dims.len() != 3 || output.dims[0] != 1 {
            return None;
        }
        // MediaPipe Pose 的 5 通道 `x,y,z,visibility,presence`。只有 K=33 的 5 通道
        // 才按这个含义解析；其它 K 出现 5 通道属于未知契约，不猜。
        if output.dims[2] == 5 && output.dims[1] == MEDIAPIPE_POSE_KEYPOINTS {
            return Some((
                output,
                BodyPoseOutputLayout::KeypointXyz,
                output.dims[1] as usize,
            ));
        }
        if output.dims[1] == 5 && output.dims[2] == MEDIAPIPE_POSE_KEYPOINTS {
            return Some((
                output,
                BodyPoseOutputLayout::XyzKeypoint,
                output.dims[2] as usize,
            ));
        }
        if matches!(output.dims[2], 2 | 3) && (min..=max).contains(&output.dims[1]) {
            return Some((
                output,
                BodyPoseOutputLayout::KeypointXyz,
                output.dims[1] as usize,
            ));
        }
        if matches!(output.dims[1], 2 | 3) && (min..=max).contains(&output.dims[2]) {
            return Some((
                output,
                BodyPoseOutputLayout::XyzKeypoint,
                output.dims[2] as usize,
            ));
        }
        None
    })
}

fn keypoint_contract(outputs: &[TensorMeta]) -> Option<(&TensorMeta, BodyPoseOutputLayout, usize)> {
    outputs.iter().find_map(|output| {
        if output.element_type != 1 || output.dims.len() != 3 || output.dims[0] != 1 {
            return None;
        }
        if matches!(output.dims[2], 2 | 3) && output.dims[1] == BODY_POSE_KEYPOINT_COUNT as i64 {
            return Some((
                output,
                BodyPoseOutputLayout::KeypointXyz,
                output.dims[1] as usize,
            ));
        }
        if matches!(output.dims[1], 2 | 3) && output.dims[2] == BODY_POSE_KEYPOINT_COUNT as i64 {
            return Some((
                output,
                BodyPoseOutputLayout::XyzKeypoint,
                output.dims[2] as usize,
            ));
        }
        None
    })
}

fn output_layout(dims: &[i64], len: usize, keypoints: usize) -> Option<BodyPoseOutputLayout> {
    if len == keypoints * 2 {
        return Some(if dims.iter().rev().nth(1) == Some(&2) {
            BodyPoseOutputLayout::XyzKeypoint
        } else {
            BodyPoseOutputLayout::KeypointXyz
        });
    }
    // MediaPipe 的 5 通道：x,y,z,visibility,presence。
    if len == keypoints * 5 {
        if dims.last() == Some(&5) {
            return Some(BodyPoseOutputLayout::KeypointXyz);
        }
        if dims.iter().rev().nth(1) == Some(&5) {
            return Some(BodyPoseOutputLayout::XyzKeypoint);
        }
        return None;
    }
    if len != keypoints * 3 {
        return None;
    }
    if dims.last() == Some(&3) {
        Some(BodyPoseOutputLayout::KeypointXyz)
    } else if dims.iter().rev().nth(1) == Some(&3) {
        Some(BodyPoseOutputLayout::XyzKeypoint)
    } else {
        Some(BodyPoseOutputLayout::KeypointXyz)
    }
}

fn parse_keypoints(
    plane: &[f32],
    layout: BodyPoseOutputLayout,
    count: usize,
    transform: Letterbox,
    source_width: u32,
    source_height: u32,
) -> Result<(Vec<BodyKeypoint>, PoseChannels)> {
    let channels = plane.len() / count;
    ensure!(
        matches!(channels, 2 | 3 | 5) && channels * count == plane.len(),
        "body pose keypoint plane length mismatch"
    );
    let at = |channel: usize, index: usize| match layout {
        BodyPoseOutputLayout::KeypointXyz => plane[index * channels + channel],
        BodyPoseOutputLayout::XyzKeypoint => plane[channel * count + index],
    };
    // MediaPipe 的 4/5 通道是 visibility / presence，各自成平面保留：语义层要求
    // 两者同时成立，而不是把它们揉成一个分数或默认「全可见」。
    let visibility: Option<Vec<f32>> = (channels == 5)
        .then(|| (0..count).map(|index| at(3, index)).collect());
    let presence: Option<Vec<f32>> =
        (channels == 5).then(|| (0..count).map(|index| at(4, index)).collect());
    let third = match channels {
        // 5 通道的第三位是 z（深度）。
        5 => PoseThirdChannel::Depth,
        // 2/3 通道的第三位含义随布局，见 `BodyPoseInference::third_channel` 的兜底。
        _ => PoseThirdChannel::Absent,
    };
    let raw = (0..count)
        .map(|index| {
            let score = match channels {
                2 => 1.0,
                3 => at(2, index),
                // 五位里没有置信度：报两个真实概率的较小值，越界/非有限留原值由语义层拒绝。
                _ => at(3, index).min(at(4, index)),
            };
            BodyKeypoint {
                x: at(0, index),
                y: at(1, index),
                score,
            }
        })
        .collect::<Vec<_>>();
    ensure!(
        raw.iter()
            .all(|point| point.x.is_finite() && point.y.is_finite() && point.score.is_finite()),
        "body pose produced non-finite keypoints"
    );
    let normalized = raw
        .iter()
        .all(|point| point.x.abs() <= 2.0 && point.y.abs() <= 2.0);
    let points = raw
        .into_iter()
        .map(|point| {
            let input_x = if normalized {
                point.x * transform.input_w
            } else {
                point.x
            };
            let input_y = if normalized {
                point.y * transform.input_h
            } else {
                point.y
            };
            let x = (input_x - transform.pad_x) / transform.scale / source_width as f32;
            let y = (input_y - transform.pad_y) / transform.scale / source_height as f32;
            ensure!(
                (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y),
                "body pose keypoint outside source frame"
            );
            Ok(BodyKeypoint {
                x,
                y,
                score: point.score,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((
        points,
        PoseChannels {
            third,
            visibility,
            presence,
        },
    ))
}

fn decode_hex(value: &str) -> Result<[u8; 32]> {
    let value = value.trim();
    ensure!(value.len() == 64, "SHA-256 must contain 64 hex digits");
    let mut out = [0u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(chunk)?;
        out[index] = u8::from_str_radix(text, 16)?;
    }
    Ok(out)
}

pub fn hex(value: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(value.len() * 2);
    for byte in value {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use aurora_portrait::kernels::body_pose_adapter::Anchor;

    #[test]
    fn digest_and_size_parsing_are_strict() {
        let digits = "0123456789abcdef".repeat(4);
        assert_eq!(hex(&decode_hex(&digits).unwrap()), digits);
        assert!(decode_hex("00").is_err());
        assert_eq!(input_size(&[1, 3, 192, 256]).unwrap(), (256, 192));
        assert!(input_size(&[1, 3, -1, 256]).is_err());
    }

    #[test]
    fn keypoint_contract_accepts_only_static_pose_layouts() {
        let outputs = [TensorMeta {
            name: "pose".into(),
            dims: vec![1, 69, 3],
            element_type: 1,
        }];
        let (meta, layout, keypoints) = keypoint_contract(&outputs).unwrap();
        assert_eq!(meta.name, "pose");
        assert_eq!(layout, BodyPoseOutputLayout::KeypointXyz);
        assert_eq!(keypoints, BODY_POSE_KEYPOINT_COUNT);
        let wrong = [TensorMeta {
            name: "pose".into(),
            dims: vec![1, 17, 3],
            element_type: 1,
        }];
        assert!(keypoint_contract(&wrong).is_none());
    }

    #[test]
    fn any_layout_accepts_k_in_bounds_and_the_69_gate_stays_strict() {
        let meta = |dims: Vec<i64>| TensorMeta {
            name: "pose".into(),
            dims,
            element_type: 1,
        };
        for k in [17i64, 26, 33, 133] {
            let outputs = [meta(vec![1, k, 3])];
            let (_, layout, count) = keypoint_contract_any(&outputs).unwrap();
            assert_eq!(layout, BodyPoseOutputLayout::KeypointXyz);
            assert_eq!(count, k as usize);
            assert!(keypoint_contract(&outputs).is_none(), "K={k} must not pass the 69 gate");
        }
        let transposed = [meta(vec![1, 3, 133])];
        assert_eq!(
            keypoint_contract_any(&transposed).unwrap().1,
            BodyPoseOutputLayout::XyzKeypoint
        );
        assert!(keypoint_contract_any(&[meta(vec![1, 8, 3])]).is_none());
        assert!(keypoint_contract_any(&[meta(vec![1, 300, 3])]).is_none());
        let native = [meta(vec![1, 69, 3])];
        assert_eq!(keypoint_contract(&native).unwrap().2, 69);
        assert_eq!(
            keypoint_contract_any(&native).unwrap().1,
            BodyPoseOutputLayout::KeypointXyz
        );
    }

    #[test]
    fn candidates_put_explicit_env_first_and_dedupe() {
        let exe = PathBuf::from("C:/exe");
        let env_model = PathBuf::from("C:/models/pose.onnx");
        let env_dir = PathBuf::from("C:/models/dir");
        let cwd = PathBuf::from("C:/cwd");
        let candidates = model_candidates_from(
            Some(exe.clone()),
            Some(env_model.clone()),
            Some(env_dir.clone()),
            Some(cwd.clone()),
        );
        assert_eq!(candidates[0], env_model);
        assert_eq!(candidates[1], env_dir.join(MODEL_NAMES[0]));
        assert!(candidates.contains(&exe.join("models/bodypose").join(MODEL_NAMES[0])));
        assert!(candidates.contains(&cwd.join("models/bodypose").join(MODEL_NAMES[0])));

        let mut unique = candidates.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), candidates.len(), "candidates must not repeat");
    }

    #[test]
    fn discovery_without_any_local_model_is_named() {
        if model_candidates().iter().all(|path| !path.is_file()) {
            assert_eq!(
                discover(),
                Err(Refusal::MissingInput(BODY_POSE_ACQUISITION_REFUSAL))
            );
        }
    }

    // ── semantic anchors: channels, occlusion, Halpe-26 ──

    /// Build an inference for `count` keypoints with an explicit contract.
    fn inference_with(count: usize, keypoints: Vec<BodyKeypoint>) -> BodyPoseInference {
        BodyPoseInference {
            model_sha256: [7u8; 32],
            provider: "cuda".into(),
            frame_size: (1200, 800),
            source_sha256: [9u8; 32],
            input_size: (256, 192),
            tensor_contract: BodyPoseTensorContract {
                input_name: "input".into(),
                input_dims: [1, 3, 192, 256],
                input_element_type: 1,
                output_name: "pose".into(),
                output_dims: vec![1, count as i64, 3],
                output_element_type: 1,
                output_layout: BodyPoseOutputLayout::KeypointXyz,
                keypoints: count,
            },
            keypoints,
            channels: PoseChannels {
                third: if count == 33 { PoseThirdChannel::Depth } else { PoseThirdChannel::Score },
                ..PoseChannels::default()
            },
        }
    }

    /// MediaPipe `[1,33,5]`: `z` plus the two real probability planes.
    fn mp33_with(z: f32, visibility: Vec<f32>, presence: Vec<f32>) -> BodyPoseInference {
        let mut pose = inference_with(33, grid_points(33, z));
        pose.tensor_contract.output_dims = vec![1, 33, 5];
        pose.channels = PoseChannels {
            third: PoseThirdChannel::Depth,
            visibility: Some(visibility),
            presence: Some(presence),
        };
        pose
    }

    /// `count` points spread over the middle of the frame, all with `score`.
    fn grid_points(count: usize, score: f32) -> Vec<BodyKeypoint> {
        (0..count)
            .map(|index| BodyKeypoint {
                x: 0.2 + 0.5 * (index % 7) as f32 / 7.0,
                y: 0.2 + 0.5 * (index / 7) as f32 / 7.0,
                score,
            })
            .collect()
    }

    /// MediaPipe：z 永远不是置信；没有 visibility/presence 就不给点；
    /// 五通道时由两个真实概率决定。
    #[test]
    fn mediapipe_depth_is_never_confidence_and_visibility_is_required() {
        // ×3：第三通道是 z，证不了可见性 → 整体拒绝，不假设「全可见」。
        let depth_only = inference_with(33, grid_points(33, 0.0));
        assert_eq!(depth_only.third_channel(), PoseThirdChannel::Depth);
        assert_eq!(
            depth_only.semantic_anchors(0.5),
            Err(Refusal::MissingInput(BODY_POSE_VISIBILITY_REFUSAL))
        );

        // ×5：z 全 0（若被当 score 会清空全部），两个概率高 → 19 个锚点。
        let anchors = mp33_with(0.0, vec![0.9; 33], vec![0.9; 33])
            .semantic_anchors(0.5)
            .expect("mediapipe33 anchors");
        assert_eq!(anchors.present_count(), 19, "MediaPipe33 maps 19 anchors");

        // ×5：z 很大但 visibility 低 → 该点仍然缺失，深度救不回来。
        let mut visibility = vec![0.9f32; 33];
        visibility[27] = 0.05; // LeftAnkle
        let anchors = mp33_with(0.99, visibility, vec![0.9; 33])
            .semantic_anchors(0.5)
            .expect("mediapipe33 anchors");
        assert!(
            anchors.get(Anchor::LeftAnkle).is_none(),
            "a large z must not resurrect a hidden point"
        );

        // 对照组：真正的置信通道全 0 就是「这个实例不存在」，一个锚点都不给。
        let coco = inference_with(17, grid_points(17, 0.0));
        assert_eq!(coco.third_channel(), PoseThirdChannel::Score);
        assert_eq!(
            coco.semantic_anchors(0.0).expect("coco17 anchors").present_count(),
            0,
            "a zero score is absence, not a point"
        );
    }

    /// visibility 与 presence 是两个独立信号，任一不达标或缺信号都不给点。
    #[test]
    fn visibility_and_presence_are_independent_signals() {
        let mut visibility = vec![0.95f32; 33];
        let mut presence = vec![0.95f32; 33];
        visibility[27] = 0.9;
        presence[27] = 0.05; // LeftAnkle：可见性有，但不在画面
        visibility[28] = 0.05;
        presence[28] = 0.9; // RightAnkle：在画面，但被遮挡
        visibility[29] = 1.7; // LeftHeel：越界概率不是分数，也不是「可见」
        let anchors = mp33_with(0.0, visibility, presence)
            .semantic_anchors(0.5)
            .expect("mediapipe33 anchors");
        assert!(anchors.get(Anchor::LeftAnkle).is_none(), "presence failed");
        assert!(anchors.get(Anchor::RightAnkle).is_none(), "visibility failed");
        assert!(
            anchors.get(Anchor::LeftHeel).is_none(),
            "an out-of-range probability is not evidence"
        );
        assert!(anchors.get(Anchor::Nose).is_some());
        assert_eq!(anchors.present_count(), 16, "19 mapped anchors minus three");

        // 只剩一个概率平面同样是「信号不全」，不退回纯几何。
        let mut half = mp33_with(0.0, vec![0.9; 33], vec![0.9; 33]);
        half.channels.presence = None;
        assert_eq!(
            half.semantic_anchors(0.5),
            Err(Refusal::MissingInput(BODY_POSE_VISIBILITY_REFUSAL)),
            "presence without visibility must not default to fully visible"
        );
    }

    /// 受信布局覆盖仍要对着张量的 K 校验。
    #[test]
    fn a_declared_layout_is_validated_against_the_tensor_k() {
        let halpe = inference_with(26, grid_points(26, 0.9));
        assert!(halpe
            .semantic_anchors_with_layout(ModelLayout::Halpe26, 0.5)
            .is_ok());
        assert_eq!(
            halpe.semantic_anchors_with_layout(ModelLayout::Coco17, 0.5),
            Err(Refusal::ShapeMismatch("body pose layout override"))
        );
    }

    /// 遮挡/NaN/出画只让对应锚点缺失，绝不用零点补齐。
    #[test]
    fn occluded_or_invalid_points_become_absent_anchors() {
        let mut points = grid_points(26, 0.9);
        points[15] = BodyKeypoint {
            x: 0.4,
            y: 0.6,
            score: 0.0,
        }; // 遮挡
        points[16] = BodyKeypoint {
            x: f32::NAN,
            y: 0.6,
            score: 0.9,
        }; // 非有限
        points[24] = BodyKeypoint {
            x: 1.4,
            y: 0.6,
            score: 0.9,
        }; // 出画
        let anchors = inference_with(26, points)
            .semantic_anchors(0.5)
            .expect("halpe26 anchors");
        assert!(anchors.get(Anchor::LeftAnkle).is_none(), "occluded");
        assert!(anchors.get(Anchor::RightAnkle).is_none(), "non-finite");
        assert!(anchors.get(Anchor::LeftHeel).is_none(), "outside the frame");
        assert!(anchors.get(Anchor::LeftKnee).is_some(), "neighbours survive");
        assert!(anchors.require(Anchor::LeftHeel).is_err());
        assert_eq!(
            anchors.present_count(),
            19,
            "22 mapped anchors minus the three dropped ones"
        );
    }

    /// Halpe-26 的 17/18/19/20/21/24/25 必须落到头顶点/颈根/中髋/大趾/脚跟。
    #[test]
    fn halpe26_maps_head_top_neck_mid_hip_toes_and_heels() {
        let anchors = inference_with(26, grid_points(26, 0.9))
            .semantic_anchors(0.5)
            .expect("halpe26 anchors");
        for anchor in [
            Anchor::HeadTop,
            Anchor::NeckBase,
            Anchor::MidHip,
            Anchor::LeftToe,
            Anchor::RightToe,
            Anchor::LeftHeel,
            Anchor::RightHeel,
        ] {
            assert!(anchors.get(anchor).is_some(), "{anchor:?} must come from Halpe");
        }
        assert_eq!(anchors.present_count(), 22, "Halpe26 table maps 22 anchors");
        // 22/23（左右小趾）不映射：不该凭空多出锚点。
        assert!(anchors.get(Anchor::LeftKnee).is_some());
    }

    /// 无布局的 K 与原生 69 一样整体拒绝；语义结果不得喂进 legacy 69 槽。
    #[test]
    fn unknown_layouts_and_the_native_69_are_refused_not_guessed() {
        let native = inference_with(69, grid_points(69, 0.9));
        assert_eq!(native.semantic_layout(), Err(Refusal::MissingInput(BODY_POSE_NATIVE_SEMANTICS_REFUSAL)));
        assert_eq!(
            native.semantic_anchors(0.5),
            Err(Refusal::MissingInput(BODY_POSE_NATIVE_SEMANTICS_REFUSAL)),
            "the vendor slot order is unrecovered, so raw order is not semantics"
        );
        // 22 点这类没有回收布局的 K 也必须整体拒绝，而不是借用邻近表。
        let unknown = inference_with(22, grid_points(22, 0.9));
        assert_eq!(
            unknown.semantic_anchors(0.5),
            Err(Refusal::MissingInput(BODY_POSE_SEMANTIC_LAYOUT_REFUSAL))
        );
        // 语义路径与 legacy 69 槽 ABI 是两条路：K=26 的推断进不了 shader 槽表。
        let halpe = inference_with(26, grid_points(26, 0.9));
        assert!(crate::body_shader_abi::BodyKeypointSet::from_inference(&halpe).is_err());
        assert!(halpe.semantic_anchors(0.5).is_ok());
    }
}
