//! PF8033 CPU-forward receipt and artifact binding.
//!
//! A1f executes the recovered candidate networks with MNN CPU. This module
//! binds those outputs to the existing PF8033 Rust smile stage without
//! promoting the candidate channel split or scheduling to vendor parity.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use aurora_portrait::model_sha256::sha256;
use aurora_portrait::kernels::smile_warp;
use aurora_portrait::{Image, Refusal};
use serde_json::Value;

use super::portrait_face_stage::{
    apply_cpu, hash_f32, FaceGeometryDraw, FaceGeometryInputs, SmileHostSplitReceipt,
};
use super::portrait_face_geometry_provider::FaceGeometryProviderRow;
use super::{FaceGeometry, PortraitBackend};
use crate::NormRect;

pub const PF8033_CPU_FORWARD_RECEIPT_ENV: &str = "AURORA_PF8033_FORWARD_RECEIPT";
pub const PF8033_CPU_FORWARD_ARTIFACTS_ENV: &str = "AURORA_PF8033_ARTIFACTS";
pub const PF8033_MODEL_ROOT_ENV: &str = "AURORA_PF8033_MODEL_ROOT";

pub const PF8033_FINAL_TENSOR_ID: u32 = 780_929_437;
pub const PF8033_FINAL_TENSOR_FILE: &str = "pf8033-780929437.f32le";
pub const PF8033_DX_FILE: &str = "pf8033-dx.f32le";
pub const PF8033_DY_FILE: &str = "pf8033-dy.f32le";
pub const PF8033_MASK_FILE: &str = "pf8033-mask.u8";
pub const PF8033_ARTIFACT_CACHE: &str =
    r"C:\WorkSpace\Aurora Studio\90_work\build-cache\pf8033";
pub const PF8033_MODEL_WIDTH: u32 = 512;
pub const PF8033_MODEL_HEIGHT: u32 = 512;
pub const PF8033_FRAME_WIDTH: u32 = 1024;
pub const PF8033_FRAME_HEIGHT: u32 = 768;
pub const PF8033_MODEL_IDS: [u64; 3] = [806_921_480, 1_940_369_920, 3_157_078_880];
pub const PF8033_FINAL_MODEL_ID: u64 = 3_157_078_880;
pub const PF8033_RECEIPT_SCHEMA: &str = "aurora-pf8033-cpu-forward-receipt/1";
pub const PF8033_HOST_SPLIT_STATUS: &str =
    "EXECUTED_CANDIDATE__VENDOR_CHANNEL_SEMANTICS_UNPROVEN";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pf8033TensorReceipt {
    pub id: u64,
    pub shape: [u32; 4],
    pub sha256: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pf8033ModelReceipt {
    pub id: u64,
    pub source: PathBuf,
    pub raw_sha256: [u8; 32],
    pub decoded_sha256: [u8; 32],
    pub candidate: PathBuf,
    pub candidate_sha256: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pf8033NetworkReceipt {
    pub model_id: u64,
    pub inputs: Vec<Pf8033TensorReceipt>,
    pub outputs: Vec<Pf8033TensorReceipt>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pf8033HostSplitReceiptData {
    pub status: String,
    pub final_output_id: u32,
    pub channels: [u8; 3],
    pub center: String,
    pub scale_dx: f32,
    pub scale_dy: f32,
    pub scale_pixels: f32,
    pub model_output_sha256: [u8; 32],
    pub dx_sha256: [u8; 32],
    pub dy_sha256: [u8; 32],
    pub mask_sha256: [u8; 32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pf8033WarpReceipt {
    pub entry: String,
    pub degree: f32,
    pub changed_pixels: usize,
    pub output_sha256_rgba: [u8; 32],
    pub preview: PathBuf,
    pub preview_sha256: [u8; 32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pf8033CpuForwardReceipt {
    pub receipt_path: PathBuf,
    pub receipt_sha256: [u8; 32],
    pub job_id: u64,
    pub source_path: PathBuf,
    pub source_sha256: [u8; 32],
    pub width: u32,
    pub height: u32,
    pub models: Vec<Pf8033ModelReceipt>,
    pub networks: Vec<Pf8033NetworkReceipt>,
    pub host_split: Pf8033HostSplitReceiptData,
    pub warp: Pf8033WarpReceipt,
    pub cpu_forward: bool,
    pub vendor_golden_outputs: bool,
    pub host_proven_dx_dy_channels: bool,
}

impl Pf8033CpuForwardReceipt {
    pub fn parse(path: impl Into<PathBuf>, bytes: &[u8]) -> Result<Self, Refusal> {
        let path = path.into();
        let root: Value = serde_json::from_slice(bytes)
            .map_err(|_| Refusal::ValueOutOfRange("PF8033 CPU forward receipt JSON"))?;
        if text(&root, "schema", "PF8033 CPU forward receipt schema")? != PF8033_RECEIPT_SCHEMA {
            return Err(Refusal::ValueOutOfRange("PF8033 CPU forward receipt schema"));
        }
        let job_id = number(&root, "job_id", "PF8033 CPU forward job ticket")?;
        if job_id == 0 {
            return Err(Refusal::ValueOutOfRange("PF8033 CPU forward job ticket"));
        }

        let source = object(&root, "source", "PF8033 CPU forward source")?;
        let width = u32_number(source, "width")?;
        let height = u32_number(source, "height")?;
        if (width, height) != (PF8033_FRAME_WIDTH, PF8033_FRAME_HEIGHT) {
            return Err(Refusal::ShapeMismatch("PF8033 CPU forward source frame"));
        }

        let models = array(&root, "models", "PF8033 CPU forward models")?
            .iter()
            .map(parse_model)
            .collect::<Result<Vec<_>, _>>()?;
        if models.len() != PF8033_MODEL_IDS.len()
            || models
                .iter()
                .map(|model| model.id)
                .ne(PF8033_MODEL_IDS)
        {
            return Err(Refusal::ValueOutOfRange("PF8033 CPU forward model set"));
        }

        let networks = array(&root, "network_runs", "PF8033 CPU forward network runs")?
            .iter()
            .map(parse_network)
            .collect::<Result<Vec<_>, _>>()?;
        if networks.len() != 3
            || networks
                .iter()
                .map(|network| network.model_id)
                .ne(PF8033_MODEL_IDS.map(|id| id))
        {
            return Err(Refusal::ValueOutOfRange("PF8033 CPU forward network order"));
        }
        let final_tensor = networks[2]
            .outputs
            .iter()
            .find(|tensor| tensor.id == u64::from(PF8033_FINAL_TENSOR_ID))
            .ok_or(Refusal::MissingInput("PF8033 final 780929437 network output"))?;
        if final_tensor.shape != [1, 3, PF8033_MODEL_HEIGHT, PF8033_MODEL_WIDTH] {
            return Err(Refusal::ShapeMismatch("PF8033 final 780929437 network output"));
        }

        let split = object(&root, "host_split", "PF8033 host split receipt")?;
        let channels = object(split, "channels", "PF8033 host split channels")?;
        let channel = |name: &str| -> Result<u8, Refusal> {
            u64_number(
                split
                    .get("channels")
                    .and_then(|value| value.get(name))
                    .ok_or(Refusal::MissingInput("PF8033 host split channel"))?,
            )
            .and_then(|value| {
                u8::try_from(value)
                    .map_err(|_| Refusal::ValueOutOfRange("PF8033 host split channel"))
            })
        };
        if channels.as_object().map_or(true, |channels| channels.len() != 3) {
            return Err(Refusal::ShapeMismatch("PF8033 host split channels"));
        }
        let scale = object(split, "scale", "PF8033 host split scale")?;
        let host_split = Pf8033HostSplitReceiptData {
            status: text(split, "status", "PF8033 host split status")?.to_owned(),
            final_output_id: u32_from_u64(
                number(split, "final_output_id", "PF8033 host split output id")?,
                "PF8033 host split output id",
            )?,
            channels: [channel("dx")?, channel("dy")?, channel("mask")?],
            center: text(split, "center", "PF8033 host split center")?.to_owned(),
            scale_dx: f32_number(scale, "dx", "PF8033 host split scale")?,
            scale_dy: f32_number(scale, "dy", "PF8033 host split scale")?,
            scale_pixels: f32_number(scale, "pixels", "PF8033 host split scale")?,
            model_output_sha256: sha(split, "model_output_sha256")?,
            dx_sha256: sha(split, "dx_sha256")?,
            dy_sha256: sha(split, "dy_sha256")?,
            mask_sha256: sha(split, "mask_sha256")?,
        };
        if host_split.final_output_id != PF8033_FINAL_TENSOR_ID
            || host_split.channels != [0, 1, 2]
            || host_split.model_output_sha256 != final_tensor.sha256
        {
            return Err(Refusal::ValueOutOfRange("PF8033 host split output binding"));
        }
        if host_split.scale_dx <= 0.0
            || host_split.scale_dy <= 0.0
            || host_split.scale_pixels <= 0.0
            || !host_split.scale_dx.is_finite()
            || !host_split.scale_dy.is_finite()
            || !host_split.scale_pixels.is_finite()
        {
            return Err(Refusal::ValueOutOfRange("PF8033 host split scale"));
        }

        let warp = object(&root, "warp", "PF8033 warp receipt")?;
        let preview = PathBuf::from(text(warp, "preview", "PF8033 warp preview path")?);
        let preview_path = path
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
            .and_then(Path::parent)
            .map(|root| root.join(&preview))
            .unwrap_or_else(|| path.with_file_name(&preview));
        let preview_sha256 = fs::read(&preview_path)
            .map(|bytes| sha256(&bytes))
            .map_err(|_| Refusal::MissingInput("PF8033 warp preview artifact"))?;
        let changed = usize::try_from(number(
            warp,
            "changed_pixels",
            "PF8033 warp changed pixels",
        )?)
        .map_err(|_| Refusal::ValueOutOfRange("PF8033 warp changed pixels"))?;
        if changed == 0 {
            return Err(Refusal::ValueOutOfRange("PF8033 warp changed pixels"));
        }

        let evidence = object(&root, "evidence_level", "PF8033 forward evidence")?;
        let receipt = Self {
            receipt_path: path,
            receipt_sha256: sha256(bytes),
            job_id,
            source_path: PathBuf::from(text(source, "path", "PF8033 source path")?),
            source_sha256: sha(source, "sha256_rgba")?,
            width,
            height,
            models,
            networks,
            host_split,
            warp: Pf8033WarpReceipt {
                entry: text(warp, "entry", "PF8033 warp entry")?.to_owned(),
                degree: f32_number(warp, "degree", "PF8033 warp degree")?,
                changed_pixels: changed,
                output_sha256_rgba: sha(warp, "output_sha256_rgba")?,
                preview,
                preview_sha256,
            },
            cpu_forward: boolean(evidence, "cpu_forward")?,
            vendor_golden_outputs: boolean(evidence, "vendor_golden_outputs")?,
            host_proven_dx_dy_channels: boolean(evidence, "host_proven_dx_dy_channels")?,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn discover() -> Result<Self, Refusal> {
        let path = discover_receipt_path()?;
        let bytes = fs::read(&path)
            .map_err(|_| Refusal::MissingInput("PF8033 CPU forward receipt artifact"))?;
        Self::parse(path, &bytes)
    }

    fn validate(&self) -> Result<(), Refusal> {
        if !self.cpu_forward || self.vendor_golden_outputs || self.host_proven_dx_dy_channels {
            return Err(Refusal::ValueOutOfRange("PF8033 forward evidence level"));
        }
        if self.host_split.status != PF8033_HOST_SPLIT_STATUS
            || self.host_split.center != "per-channel median"
            || self.warp.entry != "pf8033_dense_warp equivalent"
            || !(0.0..=1.0).contains(&self.warp.degree)
        {
            return Err(Refusal::ValueOutOfRange("PF8033 candidate receipt semantics"));
        }
        Ok(())
    }

    pub fn discover_models(&self) -> Result<Vec<PathBuf>, Refusal> {
        let root = env::var_os(PF8033_MODEL_ROOT_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| self.models[0].source.parent().unwrap_or(Path::new(".")).into());
        let mut found = Vec::with_capacity(self.models.len());
        for model in &self.models {
            let path = root.join(format!("{}.bin", model.id));
            let missing = model_missing(model.id);
            let bytes = fs::read(&path).map_err(|_| Refusal::MissingInput(missing))?;
            if sha256(&bytes) != model.raw_sha256 {
                return Err(Refusal::ValueOutOfRange("PF8033 model raw SHA"));
            }
            found.push(path);
        }
        Ok(found)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pf8033CpuForwardArtifacts {
    pub receipt_sha256: [u8; 32],
    pub source_sha256: [u8; 32],
    pub final_field: super::SmileRgbReconstructionField,
    pub dx: Vec<f32>,
    pub dy: Vec<f32>,
    pub mask: Vec<u8>,
    pub host_split_receipt: SmileHostSplitReceipt,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pf8033CpuForwardRun {
    pub job_id: u64,
    pub receipt_sha256: [u8; 32],
    pub model_output_sha256: [u8; 32],
    pub dx_sha256: [u8; 32],
    pub dy_sha256: [u8; 32],
    pub mask_sha256: [u8; 32],
    pub output_sha256_rgba: [u8; 32],
    pub changed_pixels: usize,
    pub image: Image,
    pub backend: PortraitBackend,
}

pub fn load_pf8033_cpu_forward_artifacts(
    receipt: &Pf8033CpuForwardReceipt,
    source: &Image,
) -> Result<Pf8033CpuForwardArtifacts, Refusal> {
    source.validate()?;
    if (source.width, source.height) != (receipt.width, receipt.height) {
        return Err(Refusal::ShapeMismatch("PF8033 source frame"));
    }
    let source_sha256 = sha256(&source.pixels);
    if source_sha256 != receipt.source_sha256 {
        return Err(Refusal::ValueOutOfRange("PF8033 source RGBA SHA"));
    }
    let cache = PathBuf::from(PF8033_ARTIFACT_CACHE);
    let root = env::var_os(PF8033_CPU_FORWARD_ARTIFACTS_ENV)
        .map(PathBuf::from)
        .or_else(|| cache.is_dir().then_some(cache))
        .unwrap_or_else(|| receipt.receipt_path.parent().unwrap_or(Path::new(".")).into());
    let final_bytes = read_exact(
        &root.join(PF8033_FINAL_TENSOR_FILE),
        tensor_bytes(3, PF8033_MODEL_WIDTH, PF8033_MODEL_HEIGHT)?,
        "PF8033 final 780929437 tensor artifact",
        "PF8033 final 780929437 tensor bytes",
    )?;
    if sha256(&final_bytes) != receipt.host_split.model_output_sha256 {
        return Err(Refusal::ValueOutOfRange("PF8033 final 780929437 tensor SHA"));
    }
    let final_values = decode_f32(
        &final_bytes,
        "PF8033 final 780929437 tensor values",
    )?;
    let dx_bytes = read_exact(
        &root.join(PF8033_DX_FILE),
        tensor_bytes(1, PF8033_FRAME_WIDTH, PF8033_FRAME_HEIGHT)?,
        "PF8033 dx tensor artifact",
        "PF8033 dx tensor bytes",
    )?;
    let dy_bytes = read_exact(
        &root.join(PF8033_DY_FILE),
        tensor_bytes(1, PF8033_FRAME_WIDTH, PF8033_FRAME_HEIGHT)?,
        "PF8033 dy tensor artifact",
        "PF8033 dy tensor bytes",
    )?;
    let mask = read_exact(
        &root.join(PF8033_MASK_FILE),
        (PF8033_FRAME_WIDTH as usize) * (PF8033_FRAME_HEIGHT as usize),
        "PF8033 mask artifact",
        "PF8033 mask bytes",
    )?;
    let dx = decode_f32(&dx_bytes, "PF8033 dx tensor values")?;
    let dy = decode_f32(&dy_bytes, "PF8033 dy tensor values")?;
    let dx_sha256 = sha256(&dx_bytes);
    let dy_sha256 = sha256(&dy_bytes);
    let mask_sha256 = sha256(&mask);
    if dx_sha256 != receipt.host_split.dx_sha256
        || dy_sha256 != receipt.host_split.dy_sha256
        || mask_sha256 != receipt.host_split.mask_sha256
    {
        return Err(Refusal::ValueOutOfRange("PF8033 host split plane SHA"));
    }
    Ok(Pf8033CpuForwardArtifacts {
        receipt_sha256: receipt.receipt_sha256,
        source_sha256,
        final_field: super::SmileRgbReconstructionField {
            width: PF8033_MODEL_WIDTH,
            height: PF8033_MODEL_HEIGHT,
            values: final_values,
        },
        dx,
        dy,
        mask,
        host_split_receipt: SmileHostSplitReceipt {
            job_id: receipt.job_id,
            model_output_id: PF8033_FINAL_TENSOR_ID,
            model_output_sha256: receipt.host_split.model_output_sha256,
            dx_sha256,
            dy_sha256,
            mask_sha256,
        },
    })
}

pub fn build_pf8033_cpu_forward_row(
    receipt: &Pf8033CpuForwardReceipt,
    artifacts: &Pf8033CpuForwardArtifacts,
    controls: [f32; 2],
) -> Result<FaceGeometryProviderRow, Refusal> {
    if artifacts.receipt_sha256 != receipt.receipt_sha256
        || artifacts.source_sha256 != receipt.source_sha256
        || artifacts.host_split_receipt.job_id != receipt.job_id
    {
        return Err(Refusal::ValueOutOfRange("PF8033 CPU forward artifact receipt"));
    }
    if controls
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(Refusal::ValueOutOfRange("PF8033 CPU forward controls"));
    }
    Ok(FaceGeometryProviderRow {
        pf: 8033,
        controls,
        mesh_id: None,
        geometry: FaceGeometry::Smile {
            dx: artifacts.dx.clone(),
            dy: artifacts.dy.clone(),
            mask: artifacts.mask.clone(),
            host_split_receipt: Some(artifacts.host_split_receipt),
        },
    })
}

pub fn apply_pf8033_cpu_forward(
    source: &Image,
    receipt: &Pf8033CpuForwardReceipt,
    artifacts: &Pf8033CpuForwardArtifacts,
    controls: [f32; 2],
) -> Result<Pf8033CpuForwardRun, Refusal> {
    let row = build_pf8033_cpu_forward_row(receipt, artifacts, controls)?;
    let geometry = match row.geometry {
        FaceGeometry::Smile {
            dx,
            dy,
            mask,
            host_split_receipt,
        } => FaceGeometry::Smile {
            dx,
            dy,
            mask,
            host_split_receipt,
        },
        _ => return Err(Refusal::ValueOutOfRange("PF8033 CPU forward geometry")),
    };
    let input = FaceGeometryInputs {
        job_id: receipt.job_id,
        width: source.width,
        height: source.height,
        draws: vec![FaceGeometryDraw {
            pf: 8033,
            controls,
            source_sha256: artifacts.source_sha256,
            geometry,
        }],
    };
    let run = apply_cpu(
        source,
        NormRect::FULL,
        Some(receipt.job_id),
        Some(&input),
        None,
        8033,
        controls,
    )?;
    let image = run.image.unwrap_or_else(|| source.clone());
    let changed_pixels = image
        .pixels
        .chunks_exact(4)
        .zip(source.pixels.chunks_exact(4))
        .filter(|(after, before)| after != before)
        .count();
    let output_sha256_rgba = sha256(&image.pixels);
    if controls[0] == receipt.warp.degree
        && (changed_pixels != receipt.warp.changed_pixels
            || output_sha256_rgba != receipt.warp.output_sha256_rgba)
    {
        return Err(Refusal::ValueOutOfRange("PF8033 CPU warp receipt"));
    }
    Ok(Pf8033CpuForwardRun {
        job_id: receipt.job_id,
        receipt_sha256: receipt.receipt_sha256,
        model_output_sha256: artifacts.host_split_receipt.model_output_sha256,
        dx_sha256: artifacts.host_split_receipt.dx_sha256,
        dy_sha256: artifacts.host_split_receipt.dy_sha256,
        mask_sha256: artifacts.host_split_receipt.mask_sha256,
        output_sha256_rgba,
        changed_pixels,
        image,
        backend: PortraitBackend::Cpu,
    })
}

pub fn pf8033_cpu_forward_cpu_oracle(
    source: &Image,
    artifacts: &Pf8033CpuForwardArtifacts,
    degree: f32,
) -> Result<Vec<u8>, Refusal> {
    smile_warp::pf8033_dense_warp(
        &source.pixels,
        source.width,
        source.height,
        &artifacts.dx,
        &artifacts.dy,
        &artifacts.mask,
        degree,
    )
    .map_err(|error| match error {
        smile_warp::Refusal::ShapeMismatch(_) => Refusal::ShapeMismatch("PF8033 CPU oracle"),
        smile_warp::Refusal::MissingInput(_) => Refusal::MissingInput("PF8033 CPU oracle input"),
        smile_warp::Refusal::ValueOutOfRange(_) => {
            Refusal::ValueOutOfRange("PF8033 CPU oracle input")
        }
        smile_warp::Refusal::VendorConsumerOpen(_) => Refusal::NotImplemented(8033),
    })
}

fn discover_receipt_path() -> Result<PathBuf, Refusal> {
    if let Some(path) = env::var_os(PF8033_CPU_FORWARD_RECEIPT_ENV) {
        return Ok(path.into());
    }
    let candidates = [
        PathBuf::from(PF8033_ARTIFACT_CACHE).join("pf8033-cpu-forward.json"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/portrait/smile/pf8033-cpu-forward.json"),
        env::current_exe()
            .ok()
            .and_then(|path| path.parent().map(Path::to_path_buf))
            .unwrap_or_default()
            .join("assets/portrait/smile/pf8033-cpu-forward.json"),
    ];
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .ok_or(Refusal::MissingInput("PF8033 CPU forward receipt artifact"))
}

fn parse_model(value: &Value) -> Result<Pf8033ModelReceipt, Refusal> {
    let verifier = object(value, "official_verifier", "PF8033 model verifier")?;
    if number(verifier, "returncode", "PF8033 model verifier")? != 0
        || text(verifier, "stdout", "PF8033 model verifier")?.trim() != "PASS"
    {
        return Err(Refusal::ValueOutOfRange("PF8033 model verifier"));
    }
    Ok(Pf8033ModelReceipt {
        id: number(value, "id", "PF8033 model id")?,
        source: PathBuf::from(text(value, "source", "PF8033 model source")?),
        raw_sha256: sha(value, "raw_sha256")?,
        decoded_sha256: sha(value, "decoded_sha256")?,
        candidate: PathBuf::from(text(value, "candidate", "PF8033 candidate path")?),
        candidate_sha256: sha(value, "candidate_sha256")?,
    })
}

fn parse_network(value: &Value) -> Result<Pf8033NetworkReceipt, Refusal> {
    let model_id = number(value, "id", "PF8033 network model id")?;
    let inputs_object = object(value, "inputs", "PF8033 network inputs")?;
    let inputs = array(value, "external_inputs", "PF8033 network input order")?
        .iter()
        .map(|entry| {
            let name = text(entry, "name", "PF8033 network input name")?;
            let input = inputs_object
                .get(name)
                .ok_or(Refusal::MissingInput("PF8033 network input values"))?;
            let input_shape = shape(entry, "shape")?;
            if input_shape != shape(input, "shape")? {
                return Err(Refusal::ShapeMismatch("PF8033 network input shape"));
            }
            Ok::<_, Refusal>(Pf8033TensorReceipt {
                id: number(entry, "order", "PF8033 network input order")?,
                shape: input_shape,
                sha256: sha(input, "sha256")?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = array(value, "outputs", "PF8033 network outputs")?
        .iter()
        .map(|entry| {
            Ok::<_, Refusal>(Pf8033TensorReceipt {
                id: number(entry, "id", "PF8033 network output id")?,
                shape: shape(entry, "shape")?,
                sha256: sha(entry, "sha256")?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if outputs.is_empty() {
        return Err(Refusal::MissingInput("PF8033 network outputs"));
    }
    Ok(Pf8033NetworkReceipt {
        model_id,
        inputs,
        outputs,
    })
}

fn object<'a>(value: &'a Value, key: &str, error: &'static str) -> Result<&'a Value, Refusal> {
    value
        .get(key)
        .filter(|value| value.is_object())
        .ok_or(Refusal::MissingInput(error))
}

fn array<'a>(value: &'a Value, key: &str, error: &'static str) -> Result<&'a Vec<Value>, Refusal> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or(Refusal::MissingInput(error))
}

fn text<'a>(value: &'a Value, key: &str, error: &'static str) -> Result<&'a str, Refusal> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(Refusal::MissingInput(error))
}

fn number(value: &Value, key: &str, error: &'static str) -> Result<u64, Refusal> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or(Refusal::MissingInput(error))
}

fn u64_number(value: &Value) -> Result<u64, Refusal> {
    value
        .as_u64()
        .ok_or(Refusal::ValueOutOfRange("PF8033 receipt number"))
}

fn u32_number(value: &Value, key: &str) -> Result<u32, Refusal> {
    u32_from_u64(
        number(value, key, "PF8033 receipt dimension")?,
        "PF8033 receipt dimension",
    )
}

fn u32_from_u64(value: u64, error: &'static str) -> Result<u32, Refusal> {
    u32::try_from(value).map_err(|_| Refusal::ValueOutOfRange(error))
}

fn f32_number(value: &Value, key: &str, error: &'static str) -> Result<f32, Refusal> {
    let value = value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or(Refusal::MissingInput(error))?;
    let value = value as f32;
    if !value.is_finite() {
        return Err(Refusal::ValueOutOfRange(error));
    }
    Ok(value)
}

fn boolean(value: &Value, key: &str) -> Result<bool, Refusal> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or(Refusal::MissingInput("PF8033 evidence boolean"))
}

fn shape(value: &Value, key: &str) -> Result<[u32; 4], Refusal> {
    let values = array(value, key, "PF8033 tensor shape")?;
    if values.len() != 4 {
        return Err(Refusal::ShapeMismatch("PF8033 tensor shape"));
    }
    let mut shape = [0; 4];
    for (slot, value) in shape.iter_mut().zip(values) {
        *slot = u32_from_u64(u64_number(value)?, "PF8033 tensor shape")?;
    }
    Ok(shape)
}

fn sha(value: &Value, key: &str) -> Result<[u8; 32], Refusal> {
    let text = text(value, key, "PF8033 receipt SHA")?;
    parse_sha(text).ok_or(Refusal::ValueOutOfRange("PF8033 receipt SHA"))
}

fn parse_sha(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 {
        return None;
    }
    let mut bytes = [0u8; 32];
    for (index, slot) in bytes.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(bytes)
}

fn tensor_bytes(channels: usize, width: u32, height: u32) -> Result<usize, Refusal> {
    channels
        .checked_mul(width as usize)
        .and_then(|value| value.checked_mul(height as usize))
        .and_then(|value| value.checked_mul(4))
        .ok_or(Refusal::ShapeMismatch("PF8033 tensor bytes"))
}

fn read_exact(
    path: &Path,
    expected: usize,
    missing: &'static str,
    shape: &'static str,
) -> Result<Vec<u8>, Refusal> {
    let bytes = fs::read(path).map_err(|_| Refusal::MissingInput(missing))?;
    if bytes.len() != expected {
        return Err(Refusal::ShapeMismatch(shape));
    }
    Ok(bytes)
}

fn decode_f32(bytes: &[u8], error: &'static str) -> Result<Vec<f32>, Refusal> {
    if bytes.len() % 4 != 0 {
        return Err(Refusal::ShapeMismatch(error));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange(error));
    }
    Ok(values)
}

fn model_missing(id: u64) -> &'static str {
    match id {
        806_921_480 => "PF8033 model 806921480 artifact",
        1_940_369_920 => "PF8033 model 1940369920 artifact",
        3_157_078_880 => "PF8033 model 3157078880 artifact",
        _ => "PF8033 model artifact",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a1f_forward_receipt_pins_models_tensors_split_and_warp() {
        let bytes = include_bytes!("../../../assets/portrait/smile/pf8033-cpu-forward.json");
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/portrait/smile/pf8033-cpu-forward.json");
        let receipt = Pf8033CpuForwardReceipt::parse(path, bytes).unwrap();
        assert_eq!(receipt.job_id, 3_621_424_029_303_566_009);
        assert_eq!(
            receipt
                .models
                .iter()
                .map(|model| model.id)
                .collect::<Vec<_>>(),
            PF8033_MODEL_IDS
        );
        assert_eq!(
            receipt.networks[0].outputs[0].shape,
            [1, 1, PF8033_FRAME_HEIGHT, PF8033_FRAME_WIDTH]
        );
        assert_eq!(
            receipt.networks[2].outputs[0].shape,
            [1, 3, PF8033_MODEL_HEIGHT, PF8033_MODEL_WIDTH]
        );
        assert_eq!(receipt.host_split.channels, [0, 1, 2]);
        assert_eq!(receipt.host_split.scale_pixels, 16.0);
        assert_eq!(receipt.warp.changed_pixels, 762_262);
        assert_eq!(
            receipt.warp.output_sha256_rgba,
            parse_sha("764b6ee8cbb390d65acaf1fc808ddddddbe76d114027358a43c704ae6f5fa51c")
                .unwrap()
        );
        assert!(!receipt.vendor_golden_outputs);
        assert!(!receipt.host_proven_dx_dy_channels);
    }

    #[test]
    fn missing_tensor_artifact_has_an_exact_refusal() {
        let missing = std::env::temp_dir().join("aurora-pf8033-does-not-exist.f32le");
        assert_eq!(
            read_exact(
                &missing,
                4,
                "PF8033 final 780929437 tensor artifact",
                "PF8033 final 780929437 tensor bytes",
            ),
            Err(Refusal::MissingInput(
                "PF8033 final 780929437 tensor artifact"
            ))
        );
    }

    #[test]
    fn host_split_receipt_hashes_are_the_stage_contract() {
        let final_field = super::super::SmileRgbReconstructionField {
            width: PF8033_MODEL_WIDTH,
            height: PF8033_MODEL_HEIGHT,
            values: vec![0.25; 3 * 512 * 512],
        };
        let dx = vec![1.0; 1024 * 768];
        let dy = vec![-1.0; 1024 * 768];
        let mask = vec![255; 1024 * 768];
        let model_bytes = final_field
            .values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            hash_f32(&dx),
            sha256(&dx.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
        );
        assert_eq!(sha256(&model_bytes), sha256(&model_bytes));
        assert_eq!(mask.len(), dx.len());
    }
}
