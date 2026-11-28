use std::path::{Path, PathBuf};

use aurora_portrait::kernels::plump_pipeline::{
    ChannelMap, ColorEncoding, Error as PipelineError, ModelInput, OutputSink, PlumpLayer,
    PlumpProducer, Roi, SessionOutput, Tensor,
};
use aurora_portrait::model_sha256::{hex, sha256};
use aurora_portrait::plump_session::{spec, Route, SessionBinding, SessionStamp};
use aurora_portrait::{Image, Refusal};

use crate::aimask::{Provider, Runtime, Session};

use super::{
    PlumpCompositeReceipt, PlumpFinalComposite, PlumpProducerInputs,
    PlumpSameFrameEvidence,
};

pub const PLUMP_HIGHRES4_TOTAL_SKIN_MASK: &str =
    "FacialPlumpingPipeline Total Skin Mask";
pub const PLUMP_HIGHRES4_OUTPUT_CHANNEL_MAP: ChannelMap = ChannelMap {
    color: [0, 1, 2],
    split_mask: 3,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlumpTotalSkinMask<'a> {
    pub roi: Roi,
    pub values: &'a [f32],
}

impl<'a> PlumpTotalSkinMask<'a> {
    pub fn from_same_roi(
        current: &Image,
        roi: Roi,
        values: &'a [f32],
    ) -> Result<Self, Refusal> {
        if roi.width == 0
            || roi.height == 0
            || roi.x > current.width
            || roi.y > current.height
            || roi.width > current.width - roi.x
            || roi.height > current.height - roi.y
        {
            return Err(Refusal::ValueOutOfRange(PLUMP_HIGHRES4_TOTAL_SKIN_MASK));
        }
        let expected = roi.width as usize * roi.height as usize;
        if values.len() != expected {
            return Err(Refusal::ShapeMismatch(PLUMP_HIGHRES4_TOTAL_SKIN_MASK));
        }
        if values
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(Refusal::ValueOutOfRange(PLUMP_HIGHRES4_TOTAL_SKIN_MASK));
        }
        Ok(Self { roi, values })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlumpOnnxAsset {
    pub route: Route,
    pub model_id: u32,
    pub path: PathBuf,
    pub file_bytes: usize,
    pub file_sha256: [u8; 32],
    pub net_sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlumpRasterBackend {
    Cuda,
    Cpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlumpRasterReceipt {
    pub backend: PlumpRasterBackend,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlumpRasterOutcome<T> {
    pub value: T,
    pub receipt: PlumpRasterReceipt,
}

pub fn plump_raster_with_cpu_fallback<T, E, C, F>(
    cuda: C,
    cpu: F,
) -> Result<PlumpRasterOutcome<T>, E>
where
    C: FnOnce() -> Result<T, E>,
    F: FnOnce() -> Result<T, E>,
{
    match cuda() {
        Ok(value) => Ok(PlumpRasterOutcome {
            value,
            receipt: PlumpRasterReceipt {
                backend: PlumpRasterBackend::Cuda,
            },
        }),
        Err(_) => cpu().map(|value| PlumpRasterOutcome {
            value,
            receipt: PlumpRasterReceipt {
                backend: PlumpRasterBackend::Cpu,
            },
        }),
    }
}

/// Explicit directory holding `{id}.onnx` FacePlump models. Highest priority.
pub const PLUMP_MODEL_ROOT_ENV: &str = "AURORA_FACEPLUMP_MODEL_ROOT";

fn push_model_root(roots: &mut Vec<PathBuf>, path: PathBuf) {
    if !roots.iter().any(|existing| *existing == path) {
        roots.push(path);
    }
}

/// Ordered FacePlump model roots: env override, executable-relative install
/// layouts, cwd-relative layouts, the dev `../../models` layout, then the
/// historical absolute development paths. Duplicates collapse to their first
/// position so a cwd that equals the exe directory is not scanned twice.
fn plump_model_roots(
    exe_dir: Option<&Path>,
    cwd: Option<&Path>,
    env: Option<PathBuf>,
) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(env) = env {
        push_model_root(&mut roots, env);
    }
    if let Some(dir) = exe_dir {
        for base in [
            dir.join("models").join("faceplump"),
            dir.join("assets").join("portrait").join("faceplump"),
            dir.join("faceplump"),
        ] {
            push_model_root(&mut roots, base);
        }
    }
    if let Some(dir) = cwd {
        for base in [
            dir.join("models").join("faceplump"),
            dir.join("assets").join("portrait").join("faceplump"),
            dir.join("faceplump"),
        ] {
            push_model_root(&mut roots, base);
        }
    }
    if let Some(dir) = exe_dir {
        for base in [
            dir.join("..").join("..").join("models").join("faceplump"),
            dir.join("..")
                .join("..")
                .join("assets")
                .join("portrait")
                .join("faceplump"),
        ] {
            push_model_root(&mut roots, base);
        }
    }
    push_model_root(
        &mut roots,
        PathBuf::from(r"C:\WorkSpace\Aurora Studio\05_deliverables\faceplump-models"),
    );
    push_model_root(
        &mut roots,
        PathBuf::from(r"C:\WorkSpace\Aurora Studio\90_work\build-cache\faceplump"),
    );
    roots
}

pub fn discover_plump_onnx_models() -> Result<[PlumpOnnxAsset; 3], Refusal> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    let cwd = std::env::current_dir().ok();
    let env = std::env::var_os(PLUMP_MODEL_ROOT_ENV).map(PathBuf::from);
    let roots = plump_model_roots(exe_dir.as_deref(), cwd.as_deref(), env);
    let routes = [Route::HighRes4, Route::HighRes3, Route::LowRes3];
    let mut found = Vec::with_capacity(3);
    for route in routes {
        let model_id = spec(route).model_id;
        let mut selected = None;
        for root in &roots {
            for candidate in model_candidates(root, model_id) {
                if candidate.is_file() {
                    selected = Some(load_asset(route, candidate)?);
                    break;
                }
            }
            if selected.is_some() {
                break;
            }
        }
        found.push(selected.ok_or(Refusal::MissingInput(
            "FacePlump ONNX model discovery",
        ))?);
    }
    Ok([found.remove(0), found.remove(0), found.remove(0)])
}

fn model_candidates(root: &Path, model_id: u32) -> [PathBuf; 3] {
    let id = model_id.to_string();
    [
        root.join(format!("{id}.onnx")),
        root.join(format!("faceplump-{id}.onnx")),
        root.join(&id).join("model.onnx"),
    ]
}

fn load_asset(route: Route, path: PathBuf) -> Result<PlumpOnnxAsset, Refusal> {
    let bytes = std::fs::read(&path)
        .map_err(|_| Refusal::MissingInput("FacePlump ONNX model bytes"))?;
    let actual = sha256(&bytes);
    let sidecar = PathBuf::from(format!("{}.sha256", path.display()));
    let expected_text = std::fs::read_to_string(&sidecar)
        .map_err(|_| Refusal::MissingInput("FacePlump ONNX SHA256 sidecar"))?;
    let expected = parse_sha256(expected_text.trim())
        .ok_or(Refusal::ValueOutOfRange("FacePlump ONNX SHA256 sidecar"))?;
    if actual != expected {
        return Err(Refusal::ValueOutOfRange("FacePlump ONNX SHA256"));
    }
    let net_sha256 = parse_sha256(spec(route).net_sha256)
        .ok_or(Refusal::ValueOutOfRange("FacePlump net SHA256"))?;
    Ok(PlumpOnnxAsset {
        route,
        model_id: spec(route).model_id,
        path,
        file_bytes: bytes.len(),
        file_sha256: actual,
        net_sha256,
    })
}

fn parse_sha256(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        out[index] = u8::from_str_radix(std::str::from_utf8(chunk).ok()?, 16).ok()?;
    }
    Some(out)
}

pub struct PlumpBuildRequest<'a> {
    pub current: &'a Image,
    pub job_id: u64,
    pub face_id: u32,
    pub slots: [f32; 17],
    pub requested_pfs: Vec<u32>,
    pub full_face_level: Option<u8>,
    pub landmarks: &'a [[f32; 2]],
    pub roi: Roi,
    pub plane4: Option<PlumpTotalSkinMask<'a>>,
    pub asset: &'a PlumpOnnxAsset,
    pub channel_map: ChannelMap,
    pub color_encoding: ColorEncoding,
}

pub fn build_plump_producer_inputs(
    request: PlumpBuildRequest<'_>,
) -> Result<PlumpProducerInputs, Refusal> {
    if request.job_id == 0 || request.face_id == 0 {
        return Err(Refusal::MissingInput("FacePlump same-job identity"));
    }
    let full_face_descriptor = request
        .full_face_level
        .map(|level| super::full_face_descriptor_from_ui(f32::from(level)))
        .transpose()?;
    if request.requested_pfs.contains(&30_210) != full_face_descriptor.is_some() {
        return Err(Refusal::ValueOutOfRange("PF30210 UI"));
    }
    let plane4 = request.plane4.as_ref().map(|mask| mask.values);
    if let Some(mask) = request.plane4.as_ref() {
        if mask.roi != request.roi {
            return Err(Refusal::ValueOutOfRange(PLUMP_HIGHRES4_TOTAL_SKIN_MASK));
        }
    }
    let evidence = PlumpSameFrameEvidence::from_same_frame(
        request.current,
        request.landmarks,
        request.roi,
        request.slots,
        plane4,
    )?;
    let model_spec = spec(request.asset.route);
    let channel_map = if model_spec.input_channels == 4 {
        if plane4.is_none() {
            return Err(Refusal::MissingInput(PLUMP_HIGHRES4_TOTAL_SKIN_MASK));
        }
        if request.channel_map != PLUMP_HIGHRES4_OUTPUT_CHANNEL_MAP {
            return Err(Refusal::ValueOutOfRange(
                "FacePlump HighRes4 output channel map",
            ));
        }
        PLUMP_HIGHRES4_OUTPUT_CHANNEL_MAP
    } else {
        if plane4.is_some() {
            return Err(Refusal::ShapeMismatch("unexpected Total Skin Mask"));
        }
        request.channel_map
    };
    let nchw = pack_nchw(
        request.current,
        request.roi,
        model_spec.input_channels,
        model_spec.input_height,
        model_spec.input_width,
        plane4,
    )?;
    let mut raw = Vec::with_capacity(nchw.len() * 4);
    for value in &nchw {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    let input_digest = sha256(&raw);
    let input_shape = [
        1,
        model_spec.input_channels as u32,
        model_spec.input_height,
        model_spec.input_width,
    ];
    let inferred = infer_onnx(
        request.asset,
        &nchw,
        input_shape,
        input_digest,
        model_spec.output_tensor,
    )?;
    let output = inferred.value;
    let backend = inferred.receipt.backend;
    let binding = SessionBinding {
        model_id: request.asset.model_id,
        route: request.asset.route,
        net_sha256: request.asset.net_sha256,
        input_shape,
        input_tensor: model_spec.input_tensor,
        output_tensor: model_spec.output_tensor,
    };
    let model_input = ModelInput {
        shape: input_shape,
        nchw,
        digest: input_digest,
    };
    let layer = PlumpProducer::produce(
        &binding,
        &model_input,
        &output,
        channel_map,
        request.color_encoding,
        request.roi,
        OutputSink::FacialPlumpingLayer,
    )
    .map_err(pipeline_refusal)?;
    let strength = plump_effective_strength(&request.slots, request.full_face_level);
    let has_forehead = request.requested_pfs.contains(&30_191);
    let frame = compose_layer(
        request.current,
        &layer,
        request.color_encoding,
        strength,
        has_forehead,
    )?;
    let requested_pfs = request.requested_pfs.clone();
    let composite = PlumpFinalComposite {
        receipt: PlumpCompositeReceipt {
            ticket: request.job_id,
            face_id: request.face_id,
            requested_pfs,
            source_sha256: sha256(&request.current.pixels),
            backend,
            model_id: output.stamp.model_id,
            net_sha256: output.stamp.net_sha256,
            model_input_digest: input_digest,
            output_tensor: output.stamp.output_tensor,
        },
        frame,
    };
    Ok(PlumpProducerInputs {
        ticket: request.job_id,
        face_id: request.face_id,
        slots: request.slots,
        requested_pfs: request.requested_pfs,
        full_face_descriptor,
        session: binding,
        backend,
        model_input: Some(model_input),
        model_output: Some(output),
        channel_map,
        color_encoding: request.color_encoding,
        roi: request.roi,
        final_composite: Some(composite),
        same_frame: Some(evidence),
    })
}

fn infer_onnx(
    asset: &PlumpOnnxAsset,
    nchw: &[f32],
    input_shape: [u32; 4],
    input_digest: [u8; 32],
    output_tensor: &'static str,
) -> Result<PlumpRasterOutcome<SessionOutput>, Refusal> {
    static LAST_INFER: std::sync::OnceLock<
        std::sync::Mutex<Option<(u32, [u8; 32], PlumpRasterOutcome<SessionOutput>)>>,
    > = std::sync::OnceLock::new();
    let cache = LAST_INFER.get_or_init(|| std::sync::Mutex::new(None));
    if let Ok(guard) = cache.lock() {
        if let Some((cached_id, cached_digest, ref cached_out)) = *guard {
            if cached_id == asset.model_id && cached_digest == input_digest {
                return Ok(cached_out.clone());
            }
        }
    }

    let runtime = Runtime::get()
        .ok_or(Refusal::MissingInput("FacePlump ONNX Runtime"))?;
    let cuda = || {
        let session = if runtime.is_native_only() {
            runtime.open_session(&asset.path)
        } else {
            runtime.open_session_on(&asset.path, &[Provider::Cuda])
        }
        .map_err(|_| Refusal::MissingInput("FacePlump ONNX CUDA session"))?;
        infer_session(
            runtime,
            &session,
            nchw,
            input_shape,
            input_digest,
            output_tensor,
            asset.model_id,
            asset.net_sha256,
        )
    };
    let cpu = || {
        let session = runtime
            .open_cpu_session(&asset.path)
            .map_err(|_| Refusal::MissingInput("FacePlump ONNX CPU session"))?;
        infer_session(
            runtime,
            &session,
            nchw,
            input_shape,
            input_digest,
            output_tensor,
            asset.model_id,
            asset.net_sha256,
        )
    };
    let outcome = if runtime.is_native_only() {
        PlumpRasterOutcome {
            value: cuda()?,
            receipt: PlumpRasterReceipt { backend: PlumpRasterBackend::Cuda },
        }
    } else {
        plump_raster_with_cpu_fallback(cuda, cpu)?
    };
    if let Ok(mut guard) = cache.lock() {
        *guard = Some((asset.model_id, input_digest, outcome.clone()));
    }
    Ok(outcome)
}

#[allow(clippy::too_many_arguments)]
fn infer_session(
    runtime: &Runtime,
    session: &Session,
    nchw: &[f32],
    input_shape: [u32; 4],
    input_digest: [u8; 32],
    output_tensor: &'static str,
    model_id: u32,
    net_sha256: [u8; 32],
) -> Result<SessionOutput, Refusal> {
    let dims = input_shape.map(i64::from);
    if session.input.dims != dims {
        return Err(Refusal::ShapeMismatch("FacePlump ONNX input"));
    }
    if session.outputs.len() != 1 {
        return Err(Refusal::ShapeMismatch("FacePlump ONNX output"));
    }
    let mut input = nchw.to_vec();
    let outputs = runtime
        .run(&session, &mut input, &dims)
        .map_err(|err| {
            log::warn!("FacePlump model {model_id} {:?} dispatch failed: {err:#}", session.provider());
            Refusal::MissingInput("FacePlump ONNX dispatch")
        })?;
    if outputs.len() != 1 {
        return Err(Refusal::ShapeMismatch("FacePlump ONNX output"));
    }
    let raw_shape = &session.outputs[0].dims;
    if raw_shape.len() != 4 || raw_shape.iter().any(|value| *value <= 0) {
        return Err(Refusal::ShapeMismatch("FacePlump ONNX output shape"));
    }
    let shape = [
        raw_shape[0] as u32,
        raw_shape[1] as u32,
        raw_shape[2] as u32,
        raw_shape[3] as u32,
    ];
    let output = Tensor {
        shape,
        data: outputs.into_iter().next().unwrap_or_default(),
    };
    output.validate().map_err(pipeline_refusal)?;
    Ok(SessionOutput {
        stamp: SessionStamp {
            model_id,
            net_sha256,
            input_digest,
            input_shape,
            output_tensor,
        },
        tensor: output,
    })
}

fn pack_nchw(
    source: &Image,
    roi: Roi,
    channels: usize,
    height: u32,
    width: u32,
    plane4: Option<&[f32]>,
) -> Result<Vec<f32>, Refusal> {
    if roi.width == 0
        || roi.height == 0
        || roi.x > source.width
        || roi.y > source.height
        || roi.width > source.width - roi.x
        || roi.height > source.height - roi.y
    {
        return Err(Refusal::ValueOutOfRange("FacePlump ROI"));
    }
    let plane = height as usize * width as usize;
    let mut nchw = vec![0.0f32; channels * plane];
    for y in 0..height as usize {
        let sy = sample_coord(y, height, roi.height);
        for x in 0..width as usize {
            let sx = sample_coord(x, width, roi.width);
            let rgba = bilinear_rgba8(source, roi.x as f32 + sx, roi.y as f32 + sy);
            let at = y * width as usize + x;
            for channel in 0..3 {
                nchw[channel * plane + at] = rgba[channel] / 255.0;
            }
        }
    }
    if channels == 4 {
        let plane4 = plane4.ok_or(Refusal::MissingInput("FacePlump 4th plane"))?;
        if plane4.len() != roi.width as usize * roi.height as usize {
            return Err(Refusal::ShapeMismatch("FacePlump 4th plane"));
        }
        for y in 0..height as usize {
            let sy = sample_coord(y, height, roi.height);
            for x in 0..width as usize {
                let sx = sample_coord(x, width, roi.width);
                nchw[3 * plane + y * width as usize + x] =
                    bilinear_plane(plane4, roi.width as usize, roi.height as usize, sx, sy);
            }
        }
    } else if plane4.is_some() {
        return Err(Refusal::ShapeMismatch("unexpected 4th plane"));
    }
    Ok(nchw)
}

fn sample_coord(dst: usize, dst_size: u32, src_size: u32) -> f32 {
    if dst_size <= 1 || src_size <= 1 {
        0.0
    } else {
        dst as f32 * (src_size - 1) as f32 / (dst_size - 1) as f32
    }
}

fn bilinear_rgba8(image: &Image, x: f32, y: f32) -> [f32; 4] {
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width - 1);
    let y1 = (y0 + 1).min(image.height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let sample = |px: u32, py: u32, channel: usize| {
        image.pixels[((py as usize * image.width as usize + px as usize) * 4) + channel] as f32
    };
    let mut out = [0.0f32; 4];
    for channel in 0..4 {
        let top = sample(x0, y0, channel) * (1.0 - fx) + sample(x1, y0, channel) * fx;
        let bottom = sample(x0, y1, channel) * (1.0 - fx) + sample(x1, y1, channel) * fx;
        out[channel] = top * (1.0 - fy) + bottom * fy;
    }
    out
}

fn bilinear_plane(plane: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let sample = |px: usize, py: usize| plane[py * width + px];
    let top = sample(x0, y0) * (1.0 - fx) + sample(x1, y0) * fx;
    let bottom = sample(x0, y1) * (1.0 - fx) + sample(x1, y1) * fx;
    top * (1.0 - fy) + bottom * fy
}

fn plump_effective_strength(slots: &[f32; 17], full_face_level: Option<u8>) -> f32 {
    let level_strength = full_face_level
        .map(|lvl| (f32::from(lvl) / 4.0).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let mut slot_strength = 0.0f32;
    for idx in [0usize, 3, 6, 7, 12, 13, 16] {
        let v = slots[idx].abs();
        let norm = if v > 1.0 { v / 100.0 } else { v };
        slot_strength = slot_strength.max(norm.clamp(0.0, 1.0));
    }
    level_strength.max(slot_strength)
}

fn compose_layer(
    source: &Image,
    layer: &PlumpLayer,
    encoding: ColorEncoding,
    strength: f32,
    has_forehead: bool,
) -> Result<Image, Refusal> {
    use aurora_portrait::kernels::plump_exact_composite::{
        PLUMP_BRIGHTEN_LUT_FOREHEAD, PLUMP_BRIGHTEN_LUT_SOFT, PLUMP_STACK_BLUR_RADIUS,
        apply_plump_brighten_lut_rgba8, apply_plump_residual_rgba8, stack_blur_u8,
    };

    if layer.color.width != layer.roi.width
        || layer.color.height != layer.roi.height
        || layer.alpha_binary.width != layer.roi.width
        || layer.alpha_binary.height != layer.roi.height
    {
        return Err(Refusal::ShapeMismatch("FacePlump raster layer"));
    }
    let mut frame = source.clone();
    let source_stride = source.width as usize * 4;
    let layer_stride = layer.roi.width as usize * 4;
    let roi_w = layer.roi.width as usize;
    let roi_h = layer.roi.height as usize;

    if encoding == ColorEncoding::VendorResidualSignedUnit {
        if strength <= 0.0 || roi_w == 0 || roi_h == 0 {
            return Ok(frame);
        }
        let pixel_count = roi_w * roi_h;
        let mut roi_rgba = vec![0u8; pixel_count * 4];
        let mut pred_rgb = vec![0.0f32; pixel_count * 3];
        let mut weight_u8 = vec![0u8; pixel_count];
        let mut split_u8 = vec![0u8; pixel_count];
        for y in 0..roi_h {
            let src_row = (layer.roi.y as usize + y) * source_stride + layer.roi.x as usize * 4;
            let dst_row = y * roi_w * 4;
            roi_rgba[dst_row..dst_row + roi_w * 4]
                .copy_from_slice(&source.pixels[src_row..src_row + roi_w * 4]);
            for x in 0..roi_w {
                let at = y * roi_w + x;
                let layer_px = at * 4;
                pred_rgb[at * 3] = f32::from(layer.color.bytes[layer_px]);
                pred_rgb[at * 3 + 1] = f32::from(layer.color.bytes[layer_px + 1]);
                pred_rgb[at * 3 + 2] = f32::from(layer.color.bytes[layer_px + 2]);
                let raw = f32::from_le_bytes(
                    layer.alpha_raw.bytes[at * 4..at * 4 + 4]
                        .try_into()
                        .unwrap_or([0; 4]),
                );
                // `split_u8` gates valid ROI pixels (0 or 255) so `weight_u8` is not
                // double-multiplied against the same sparse mask before StackBlur.
                split_u8[at] = layer.alpha_binary.bytes[at];
                weight_u8[at] = (raw * strength).clamp(0.0, 255.0).round() as u8;
            }
        }
        // PROVEN 0x06c1f769 -> 0x06c26830: StackBlur(radius=10) feathering when ROI
        // exceeds the kernel window; keep raw weight on tiny sparse masks to avoid
        // collapsing sub-window support below 1/255.
        let non_zero = weight_u8.iter().filter(|&&v| v > 0).count();
        let effective_weight = if non_zero > (PLUMP_STACK_BLUR_RADIUS * 2 + 1) * 4 {
            stack_blur_u8(&weight_u8, roi_w, roi_h, PLUMP_STACK_BLUR_RADIUS)
        } else {
            weight_u8
        };
        apply_plump_residual_rgba8(&mut roi_rgba, &pred_rgb, &effective_weight, &split_u8, roi_w, roi_h);
        // PROVEN 0x06c29f90: dual 3x256 brighten LUT (forehead strong vs soft regions).
        let lut = if has_forehead {
            &PLUMP_BRIGHTEN_LUT_FOREHEAD
        } else {
            &PLUMP_BRIGHTEN_LUT_SOFT
        };
        apply_plump_brighten_lut_rgba8(&mut roi_rgba, lut, &effective_weight, &split_u8, roi_w, roi_h);
        for y in 0..roi_h {
            let dst_row = (layer.roi.y as usize + y) * source_stride + layer.roi.x as usize * 4;
            let src_row = y * roi_w * 4;
            frame.pixels[dst_row..dst_row + roi_w * 4]
                .copy_from_slice(&roi_rgba[src_row..src_row + roi_w * 4]);
        }
    } else {
        for y in 0..roi_h {
            for x in 0..roi_w {
                let at = y * roi_w + x;
                let dst = (layer.roi.y as usize + y) * source_stride
                    + (layer.roi.x as usize + x) * 4;
                let src = y * layer_stride + x * 4;
                let mask = layer.alpha_binary.bytes[at] as f32 / 255.0;
                for channel in 0..3 {
                    frame.pixels[dst + channel] = (f32::from(source.pixels[dst + channel])
                        * (1.0 - mask)
                        + f32::from(layer.color.bytes[src + channel]) * mask)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    frame
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("FacePlump raster frame"))?;
    Ok(frame)
}

fn pipeline_refusal(error: PipelineError) -> Refusal {
    match error {
        PipelineError::ShapeMismatch(name) => Refusal::ShapeMismatch(name),
        PipelineError::ValueOutOfRange(name) => Refusal::ValueOutOfRange(name),
        PipelineError::MissingInput(name) => Refusal::MissingInput(name),
        PipelineError::SessionMismatch(name) => Refusal::ValueOutOfRange(name),
        PipelineError::UnsupportedPf(pf) => Refusal::NotImplemented(pf),
        PipelineError::NormalCompositorRejected => {
            Refusal::ValueOutOfRange("facial_plumping_normal_compositor")
        }
    }
}

pub fn asset_sha256(asset: &PlumpOnnxAsset) -> String {
    hex(&asset.file_sha256)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_failure_uses_cpu_without_fabricating_cuda_dispatch() {
        let outcome =
            plump_raster_with_cpu_fallback(|| Err::<u8, &'static str>("cuda"), || Ok(7u8))
                .unwrap();
        assert_eq!(outcome.value, 7);
        assert_eq!(outcome.receipt.backend, PlumpRasterBackend::Cpu);
    }

    #[test]
    fn failed_cuda_and_cpu_has_no_executed_dispatch() {
        assert_eq!(
            plump_raster_with_cpu_fallback(
                || Err::<u8, &'static str>("cuda"),
                || Err::<u8, &'static str>("cpu"),
            ),
            Err("cpu")
        );
    }

    #[test]
    fn sha256_parser_rejects_short_and_accepts_64_hex() {
        assert!(parse_sha256("abc").is_none());
        assert_eq!(parse_sha256(&"00".repeat(32)), Some([0; 32]));
    }

    #[test]
    fn highres4_total_skin_mask_is_same_roi_and_unit_sampled() {
        let current = Image {
            width: 4,
            height: 3,
            pixels: vec![127; 4 * 3 * 4],
        };
        let roi = Roi {
            x: 1,
            y: 1,
            width: 2,
            height: 2,
        };
        assert!(PlumpTotalSkinMask::from_same_roi(&current, roi, &[0.0, 0.5, 0.75, 1.0]).is_ok());
        assert_eq!(
            PlumpTotalSkinMask::from_same_roi(&current, roi, &[0.0, 0.5, 1.5, 1.0]),
            Err(Refusal::ValueOutOfRange(PLUMP_HIGHRES4_TOTAL_SKIN_MASK))
        );
        assert_eq!(
            PlumpTotalSkinMask::from_same_roi(&current, roi, &[0.0, 0.5, 0.75]),
            Err(Refusal::ShapeMismatch(PLUMP_HIGHRES4_TOTAL_SKIN_MASK))
        );
    }

    #[test]
    fn highres4_final_composite_uses_output_plane_four_as_split_mask() {
        assert_eq!(PLUMP_HIGHRES4_OUTPUT_CHANNEL_MAP.color, [0, 1, 2]);
        assert_eq!(PLUMP_HIGHRES4_OUTPUT_CHANNEL_MAP.split_mask, 3);
    }

    #[test]
    fn plump_model_roots_follow_env_exe_cwd_dev_then_fallback_order() {
        let exe = Path::new(r"C:\app\bin");
        let cwd = Path::new(r"D:\project");
        let env = PathBuf::from(r"E:\models\faceplump");
        let roots = plump_model_roots(Some(exe), Some(cwd), Some(env.clone()));
        assert_eq!(roots[0], env, "explicit env root wins");
        assert_eq!(roots[1], exe.join("models").join("faceplump"));
        assert_eq!(roots[2], exe.join("assets").join("portrait").join("faceplump"));
        assert_eq!(roots[3], exe.join("faceplump"));
        assert_eq!(roots[4], cwd.join("models").join("faceplump"));
        assert_eq!(roots[5], cwd.join("assets").join("portrait").join("faceplump"));
        assert_eq!(roots[6], cwd.join("faceplump"));
        assert_eq!(
            roots[7],
            exe.join("..").join("..").join("models").join("faceplump")
        );
        assert_eq!(
            roots[8],
            exe.join("..")
                .join("..")
                .join("assets")
                .join("portrait")
                .join("faceplump")
        );
        assert_eq!(
            roots[9],
            PathBuf::from(r"C:\WorkSpace\Aurora Studio\05_deliverables\faceplump-models")
        );
        assert_eq!(
            roots[10],
            PathBuf::from(r"C:\WorkSpace\Aurora Studio\90_work\build-cache\faceplump")
        );
        assert_eq!(roots.len(), 11);
    }

    #[test]
    fn plump_model_roots_dedupes_repeated_directories() {
        // A portable install often runs with cwd == exe dir; the cwd block must
        // not rescan the same three directories.
        let dir = Path::new(r"C:\app\bin");
        let roots = plump_model_roots(Some(dir), Some(dir), None);
        let mut unique = roots.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), roots.len(), "no duplicate roots: {roots:?}");
        assert_eq!(roots.len(), 7, "3 exe + 2 dev + 2 absolute fallbacks");
        assert_eq!(roots[0], dir.join("models").join("faceplump"));

        // The env root is also deduped when it matches a relative layout.
        let same = dir.join("models").join("faceplump");
        let roots = plump_model_roots(Some(dir), None, Some(same.clone()));
        assert_eq!(roots[0], same);
        assert_eq!(roots.iter().filter(|root| **root == same).count(), 1);
    }

    #[test]
    fn plump_model_roots_without_exe_or_cwd_keeps_env_and_fallbacks() {
        let roots = plump_model_roots(None, None, None);
        assert_eq!(roots.len(), 2, "only the two absolute fallbacks remain");
        let roots = plump_model_roots(None, None, Some(PathBuf::from(r"E:\m")));
        assert_eq!(roots[0], PathBuf::from(r"E:\m"));
        assert_eq!(roots.len(), 3);
    }
}
