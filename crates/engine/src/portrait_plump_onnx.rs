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

pub fn discover_plump_onnx_models() -> Result<[PlumpOnnxAsset; 3], Refusal> {
    let mut roots = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            roots.push(parent.join("models").join("faceplump"));
            roots.push(parent.join("assets").join("portrait").join("faceplump"));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd.join("models").join("faceplump"));
        roots.push(cwd.join("assets").join("portrait").join("faceplump"));
    }
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
    let frame = compose_layer(request.current, &layer)?;
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
        let session = runtime
            .open_session_on(&asset.path, &[Provider::Cuda])
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
    let outcome = plump_raster_with_cpu_fallback(cuda, cpu)?;
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
        .map_err(|_| Refusal::MissingInput("FacePlump ONNX dispatch"))?;
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

fn compose_layer(source: &Image, layer: &PlumpLayer) -> Result<Image, Refusal> {
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
    for y in 0..layer.roi.height as usize {
        for x in 0..layer.roi.width as usize {
            let mask = layer.alpha_binary.bytes[y * layer.roi.width as usize + x] as f32 / 255.0;
            let dst = (layer.roi.y as usize + y) * source_stride
                + (layer.roi.x as usize + x) * 4;
            let src = y * layer_stride + x * 4;
            for channel in 0..3 {
                frame.pixels[dst + channel] = (f32::from(source.pixels[dst + channel])
                    * (1.0 - mask)
                    + f32::from(layer.color.bytes[src + channel]) * mask)
                    .round()
                    .clamp(0.0, 255.0) as u8;
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
}
