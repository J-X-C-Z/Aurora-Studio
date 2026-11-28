//! Local body-pose/keypoint ONNX discovery and inference.
//!
//! A model is accepted only when both the ONNX file and its expected SHA-256
//! are local and explicit. Missing models or digests are typed errors; this
//! module never substitutes landmarks, a mesh, or a zeroed pose.
//!
//! Model discovery order: `AURORA_BODY_POSE_MODEL` (explicit file), the folder
//! holding the running executable (`models/BodyPose/`, `AIModels/Common/BodyPose/`,
//! `models/`), then the CWD-relative and Capture One install roots below. The
//! `exe_dir` roots keep discovery working when a packaged `.bat` is launched
//! from a different working directory.
//!
//! Keypoint tensor contract (all exports must be float32):
//! * `[1,K,2]` / `[1,2,K]`: `(x, y)` only; the third channel is absent and
//!   presence comes from geometry.
//! * `[1,K,3]` / `[1,3,K]` (COCO-17, Halpe-26, OpenPose-25, WholeBody-133,
//!   and MediaPipe poses without visibility): `(x, y, score)` with `score` a
//!   real confidence in `[0,1]` (zero means absence), **not** MediaPipe's `z`.
//!   `x`/`y` are either normalized `[0,1]` or input-pixel `[0,input_w/h]`;
//!   the choice is per tensor, never per point.
//! * `[1,33,5]` / `[1,5,33]` (MediaPose Pose 33 only):
//!   `(x, y, z, visibility, presence)`. `z` is depth and is never a score;
//!   visibility and presence must both be probabilities in `[0,1]` — logits
//!   are not accepted. The postprocessing sigmoid used by common MediaPipe
//!   Python exports is therefore required.
//!
//! Channel order is `(x, y, …)` for the `[1,K,C]` layout and `(C, K)` for the
//! transposed `[1,C,K]` layout — never `(y, x, score)`.

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
pub const BODY_POSE_MODEL_ENV: &str = "AURORA_BODY_POSE_MODEL";
pub const BODY_POSE_KEYPOINT_COUNT: usize = 69;
/// Folders relative to the executable, tried before the CWD-relative/Capture One
/// roots so a packaged `.bat` finds its bundled model regardless of where it was
/// launched from.
const EXE_MODEL_DIRS: [&str; 3] = [
    "models/BodyPose",
    "AIModels/Common/BodyPose",
    "models",
];
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

const MODEL_DIRS: [&str; 10] = [
    r"C:\Program Files\Capture One\Capture One\AIModels\Common\BodyPose",
    r"C:\Program Files (x86)\Capture One\Capture One\AIModels\Common\BodyPose",
    r"C:\Program Files\Capture One\Capture One\AIModels\Common",
    r"C:\Program Files (x86)\Capture One\Capture One\AIModels\Common",
    "/Applications/Capture One.app/Contents/Resources/AIModels/Common/BodyPose",
    "/opt/captureone/AIModels/Common/BodyPose",
    "AIModels/Common/BodyPose",
    "BodyPose",
    "models/BodyPose",
    "resources/body-pose",
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
        semantic_layout_for(self.tensor_contract.keypoints)
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

/// Layout table used by [`BodyPoseInference::semantic_layout`]. Kept as a free
/// function so the loader can classify a 3-channel tensor's third channel while
/// parsing it, using the same authority as the semantic layer.
fn semantic_layout_for(keypoints: usize) -> Result<ModelLayout, Refusal> {
    match keypoints {
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
                    semantic_layout_for(self.contract.keypoints).ok(),
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

/// Candidate paths in discovery priority order.
///
/// The explicit env override is probed as a *file*, not joined with a model
/// name: it names the ONNX itself. A missing/invalid override is skipped rather
/// than fatal, so discovery still works when the variable is stale.
fn model_candidates() -> Vec<PathBuf> {
    body_pose_candidates(
        std::env::var_os(BODY_POSE_MODEL_ENV).map(PathBuf::from).as_deref(),
        std::env::current_exe().ok().and_then(|exe| exe.parent().map(Path::to_path_buf)).as_deref(),
    )
}

/// Pure form of [`model_candidates`] so discovery order can be tested without
/// touching the real environment or the real executable path.
fn body_pose_candidates(env_model: Option<&Path>, exe_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env_model {
        candidates.push(path.to_path_buf());
    }
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(dir) = exe_dir {
        roots.extend(EXE_MODEL_DIRS.iter().map(|relative| dir.join(relative)));
    }
    roots.extend(MODEL_DIRS.iter().map(PathBuf::from));
    for root in roots {
        for name in MODEL_NAMES {
            candidates.push(root.join(name));
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
    model_layout: Option<ModelLayout>,
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
        // MediaPipe 的 5 通道第三位是 z（深度），永远不作置信度。
        5 => PoseThirdChannel::Depth,
        // 2 通道没有第三位。
        2 => PoseThirdChannel::Absent,
        // 3 通道的第三位含义由模型布局决定，不能从张量形状猜：
        // COCO/Halpe/OpenPose/WholeBody 是 (x,y,score)，MediaPipe 是 (x,y,z)。
        3 => match model_layout {
            Some(ModelLayout::MediaPipe33) => PoseThirdChannel::Depth,
            Some(_) => PoseThirdChannel::Score,
            // 无已知语义布局：不把未知第三位当置信度。
            None => PoseThirdChannel::Absent,
        },
        _ => unreachable!("validated 2|3|5 channels"),
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
    // Only coordinates must be finite globally. An invalid confidence/visibility
    // channel is handled per point by `semantic_anchors` (that point becomes
    // absent) instead of throwing away an otherwise valid pose tensor.
    ensure!(
        raw.iter().all(|point| point.x.is_finite() && point.y.is_finite()),
        "body pose produced non-finite keypoints"
    );
    // Coordinate contract: x/y are either [0,1] (already normalized) or
    // [0,input_w]x[0,input_h] (pixel coordinates in the letterboxed input).
    // Decide per tensor, never per point — a single outlier must not flip the
    // interpretation of every other point. The 2.0 tolerance covers exporters
    // that emit slightly out-of-frame normalized keypoints (e.g. 1.02); any
    // coordinate beyond it means the whole plane is in input pixels.
    let normalized = raw.iter().all(|point| {
        point.x.abs() <= 2.0 && point.y.abs() <= 2.0
    });
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
    fn discovery_without_any_local_model_is_named() {
        if model_candidates().iter().all(|path| !path.is_file()) {
            assert_eq!(
                discover(),
                Err(Refusal::MissingInput(BODY_POSE_ACQUISITION_REFUSAL))
            );
        }
    }

    // ── model discovery: env override, exe_dir, relative fallbacks ──

    #[test]
    fn env_model_is_the_first_candidate() {
        let env = Path::new("D:/bundle/override/pose-133.onnx");
        let exe = Path::new("D:/bundle/bin/aurora-studio.exe");
        let candidates = body_pose_candidates(Some(env), Some(exe));
        assert_eq!(candidates.first().map(PathBuf::as_path), Some(env));
        // No other candidate may be allowed to shadow an explicit override.
        assert_eq!(candidates.iter().filter(|p| p.as_path() == env).count(), 1);
    }

    #[test]
    fn exe_dir_model_paths_are_searched_before_relative_and_c1_roots() {
        let exe = Path::new("D:/bundle/bin/aurora-studio.exe");
        let candidates = body_pose_candidates(None, exe.parent());
        let exe_dir = exe.parent().unwrap();
        let expected: Vec<PathBuf> = EXE_MODEL_DIRS
            .iter()
            .flat_map(|dir| {
                MODEL_NAMES
                    .iter()
                    .map(move |name| exe_dir.join(dir).join(name))
            })
            .collect();
        assert_eq!(&candidates[..expected.len()], &expected[..]);
        // All CWD-relative / Capture One roots stay as later candidates.
        for root in MODEL_DIRS {
            let name = MODEL_NAMES[0];
            assert!(candidates.contains(&PathBuf::from(root).join(name)));
        }
        assert!(
            candidates.iter().any(|p| p.ends_with("models/BodyPose/bodyPose.onnx")),
            "{candidates:?}"
        );
    }

    #[test]
    fn no_exe_dir_keeps_the_original_relative_and_c1_roots() {
        let candidates = body_pose_candidates(None, None);
        assert!(!candidates.iter().any(|p| p.as_os_str().is_empty()));
        assert!(candidates.contains(&PathBuf::from("models/BodyPose/bodyPose.onnx")));
        assert!(candidates.contains(&PathBuf::from(
            r"C:\Program Files\Capture One\Capture One\AIModels\Common\BodyPose\bodyPose.onnx"
        )));
    }

    #[test]
    fn env_model_without_exe_dir_is_still_a_candidate() {
        let env = Path::new("D:/bundle/override/pose-133.onnx");
        let candidates = body_pose_candidates(Some(env), None);
        assert_eq!(candidates.first().map(PathBuf::as_path), Some(env));
        assert!(candidates.contains(&PathBuf::from("models/BodyPose/bodyPose.onnx")));
    }

    // ── parse_keypoints: coordinate round-trips for COCO/MoveNet and MediaPipe ──

    /// Identity letterbox (input == source, no padding) for parse tests.
    fn identity_transform(w: u32, h: u32) -> Letterbox {
        Letterbox { scale: 1.0, pad_x: 0.0, pad_y: 0.0, input_w: w as f32, input_h: h as f32 }
    }

    fn parse_coco17(
        plane: &[f32],
        layout: BodyPoseOutputLayout,
        w: u32,
        h: u32,
    ) -> (Vec<BodyKeypoint>, PoseChannels) {
        parse_keypoints(plane, layout, 17, semantic_layout_for(17).ok(), identity_transform(w, h), w, h)
            .expect("coco17 parse")
    }

    /// COCO-17 / MoveNet `[1,17,3]`: `(x, y, score)` in `[0,1]` maps back to
    /// source pixels; the third channel is a real score, never a depth.
    #[test]
    fn parse_keypoints_coco17_normalized_coordinates_and_score() {
        let mut plane = vec![0.0f32; 17 * 3];
        plane[0] = 0.25;
        plane[1] = 0.75;
        plane[2] = 0.9;
        let (points, channels) = parse_coco17(&plane, BodyPoseOutputLayout::KeypointXyz, 640, 480);
        assert_eq!(channels.third, PoseThirdChannel::Score);
        assert_eq!(points[0].x, 0.25);
        assert_eq!(points[0].y, 0.75);
        assert_eq!(points[0].score, 0.9);
        // (x,y) order, never (y,x,score): verified by distinct x/y values.
        assert!((points[0].x - 0.75).abs() > 0.1);
    }

    /// COCO-17 input-pixel coordinates `[0,input_w/h]` with letterbox padding
    /// and a scale factor back to a differently sized source frame.
    #[test]
    fn parse_keypoints_coco17_pixel_coordinates_with_letterbox() {
        // source 800x400, input 320x320 -> scale = 0.4, resized 320x160,
        // pad_x = 0, pad_y = (320-160)/2 = 80.
        let transform = Letterbox { scale: 0.4, pad_x: 0.0, pad_y: 80.0, input_w: 320.0, input_h: 320.0 };
        let mut plane = vec![0.0f32; 17 * 3];
        // All points at input (160,160) map to source (0.5,0.5).
        for point in plane.chunks_exact_mut(3) {
            point[0] = 160.0;
            point[1] = 160.0;
            point[2] = 1.0;
        }
        plane[0] = 320.0; // input x -> 320/0.4 = 800 -> 800/800 = 1.0
        plane[1] = 240.0; // input y -> (240-80)/0.4 = 400 -> 400/400 = 1.0
        plane[2] = 0.8;
        let (points, _) = parse_keypoints(&plane, BodyPoseOutputLayout::KeypointXyz, 17, semantic_layout_for(17).ok(), transform, 800, 400).unwrap();
        assert!((points[0].x - 1.0).abs() < 1e-5, "{:?}", points[0]);
        assert!((points[0].y - 1.0).abs() < 1e-5, "{:?}", points[0]);
    }

    /// MediaPipe `[1,33,5]`: x/y in normalized [0,1] round-trip; z is depth;
    /// visibility/presence are separate probability planes in [0,1].
    #[test]
    fn parse_keypoints_mediapipe_33_five_channels() {
        let mut plane = vec![0.0f32; 33 * 5];
        // Every point at a valid in-frame position with high vis/pres.
        for (index, point) in plane.chunks_exact_mut(5).enumerate() {
            point[0] = 0.2 + 0.5 * (index % 7) as f32 / 7.0;
            point[1] = 0.2 + 0.5 * (index / 7) as f32 / 7.0;
            point[2] = 0.1; // z, never a score
            point[3] = 0.95;
            point[4] = 0.85;
        }
        let (points, channels) = parse_keypoints(&plane, BodyPoseOutputLayout::KeypointXyz, 33, semantic_layout_for(33).ok(), identity_transform(1200, 800), 1200, 800).unwrap();
        assert_eq!(channels.third, PoseThirdChannel::Depth);
        assert!((points[0].x - 0.2).abs() < 1e-6);
        assert_eq!(channels.visibility.as_ref().unwrap()[0], 0.95);
        assert_eq!(channels.presence.as_ref().unwrap()[0], 0.85);
        // A 5-channel plane must not accidentally use score gating:
        // all 19 mapped MediaPipe anchors are present.
        let pose = BodyPoseInference { keypoints: points, channels, ..inference_with(33, vec![]) };
        assert_eq!(pose.semantic_anchors(0.5).unwrap().present_count(), 19);
    }

    /// MediaPipe x/y in input pixels with letterbox, transposed `[1,5,33]`
    /// channel order; z and vis/pres are read from the right slots.
    #[test]
    fn parse_keypoints_mediapipe_33_pixel_coordinates_transposed() {
        // [1,5,33] channel-major: channel c starts at c*33.
        let mut plane = vec![0.0f32; 5 * 33];
        plane[0] = 120.0; // x(0)
        plane[33] = 60.0;  // y(0)
        plane[2 * 33] = -1.5; // z(0), never a score
        plane[3 * 33] = 0.7; // visibility(0)
        plane[4 * 33] = 0.7; // presence(0)
        let transform = Letterbox { scale: 1.0, pad_x: 0.0, pad_y: 0.0, input_w: 256.0, input_h: 256.0 };
        let (points, channels) = parse_keypoints(&plane, BodyPoseOutputLayout::XyzKeypoint, 33, semantic_layout_for(33).ok(), transform, 2560, 2560).unwrap();
        assert!((points[0].x - 120.0 / 2560.0).abs() < 1e-6);
        assert!((points[0].y - 60.0 / 2560.0).abs() < 1e-6);
        assert_eq!(channels.visibility.as_ref().unwrap()[0], 0.7);
        assert_eq!(channels.presence.as_ref().unwrap()[0], 0.7);
    }

    /// A point at x=1.5 in a normalized tensor stays normalized (within the
    /// 2.0 tolerance) instead of flipping the other 16 points into pixel mode.
    /// The out-of-frame point itself is rejected by the frame-range check.
    #[test]
    fn parse_keypoints_normalized_detection_tolerates_one_outlier() {
        let mut plane = vec![0.0f32; 17 * 3];
        plane[0] = 0.5; plane[1] = 0.5; plane[2] = 1.0;
        plane[3] = 1.5; plane[4] = 0.5; plane[5] = 1.0; // one point slightly outside [0,1]
        // The whole plane remains in [0,2] -> treated as normalized; x=1.5
        // resolves outside the frame, so the parse fails as a whole.
        assert!(parse_keypoints(&plane, BodyPoseOutputLayout::KeypointXyz, 17, semantic_layout_for(17).ok(), identity_transform(640, 480), 640, 480).is_err());
        // Change the outlier to a valid normalized coordinate: the other point
        // must still be interpreted as normalized (0.5), not as 0.5 input px.
        plane[3] = 0.9;
        let (points, _) = parse_coco17(&plane, BodyPoseOutputLayout::KeypointXyz, 640, 480);
        assert!((points[0].x - 0.5).abs() < 1e-6, "{:?}", points[0]);
        assert!((points[1].x - 0.9).abs() < 1e-6, "{:?}", points[1]);
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
