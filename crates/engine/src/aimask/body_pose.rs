//! Local body-pose/keypoint ONNX discovery and inference.
//!
//! A model is accepted only when both the ONNX file and its expected SHA-256
//! are local and explicit. Missing models or digests are typed errors; this
//! module never substitutes landmarks, a mesh, or a zeroed pose.

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use aurora_portrait::{Image, Refusal};

use super::ort::{Runtime, Session, TensorMeta};

pub const BODY_POSE_ACQUISITION_REFUSAL: &str = "body pose model acquisition";
pub const BODY_POSE_SHA256_REFUSAL: &str = "body pose model SHA-256 manifest";
pub const BODY_POSE_TENSOR_CONTRACT_REFUSAL: &str = "body pose tensor contract";
pub const BODY_POSE_KEYPOINT_COUNT: usize = 69;
/// Smallest keypoint count this loader accepts from a non-native model.
/// The 69-slot ABI stays fail-closed: `BodyPoseInference::landmarks()` still
/// refuses anything other than exactly `BODY_POSE_KEYPOINT_COUNT`; an adapter
/// owns the K -> 69 mapping and must be identified in the inference receipt.
pub const BODY_POSE_MIN_KEYPOINTS: usize = 17;
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyKeypoint {
    pub x: f32,
    pub y: f32,
    pub score: f32,
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
        let keypoints = parsed.ok_or_else(|| anyhow::anyhow!("body pose output shape mismatch"))?;
        Ok(BodyPoseInference {
            model_sha256: self.sha256,
            provider: self.session.provider().label().to_string(),
            frame_size: (source.width, source.height),
            source_sha256: crate::sha256_32(&source.pixels),
            input_size: self.input_size,
            tensor_contract: self.contract.clone(),
            keypoints,
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
    let mut candidates = Vec::new();
    for root in MODEL_DIRS {
        let root = PathBuf::from(root);
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
) -> Result<Vec<BodyKeypoint>> {
    let channels = plane.len() / count;
    ensure!(
        matches!(channels, 2 | 3) && channels * count == plane.len(),
        "body pose keypoint plane length mismatch"
    );
    let raw = (0..count)
        .map(|index| match layout {
            BodyPoseOutputLayout::KeypointXyz => BodyKeypoint {
                x: plane[index * channels],
                y: plane[index * channels + 1],
                score: if channels == 3 {
                    plane[index * channels + 2]
                } else {
                    1.0
                },
            },
            BodyPoseOutputLayout::XyzKeypoint => BodyKeypoint {
                x: plane[index],
                y: plane[count + index],
                score: if channels == 3 {
                    plane[count * 2 + index]
                } else {
                    1.0
                },
            },
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
    raw.into_iter()
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
        .collect()
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
}
