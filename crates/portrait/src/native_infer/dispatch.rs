//! Factories are model-aware: successfully enumerating a GPU does not make an ONNX graph executable.
use super::backend::{
    self, BackendKind, BackendRequest, Host, NamedInputs, NamedOutputs, NativeInferBackend,
    RouteError, ShapeOverrides,
};
use std::sync::Arc;

#[cfg(windows)]
impl NativeInferBackend for super::session::NativeCudaSession {
    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }
    fn device_detail(&self)->Option<String>{Some(self.actual_device_label())}
    fn arena_peak_bytes(&self) -> usize {
        super::session::NativeCudaSession::arena_peak_bytes(self)
    }
    fn run_f32(&self, inputs: &NamedInputs<'_>) -> Result<NamedOutputs, String> {
        super::session::NativeCudaSession::run_f32(self, inputs).map_err(|e| e.to_string())
    }
}

pub struct RoutedSession {
    pub session: Arc<dyn NativeInferBackend>,
    pub previous_attempts: Vec<(BackendKind, String)>,
}
pub fn open(
    request: BackendRequest,
    bytes: &[u8],
    shapes: &ShapeOverrides,
) -> Result<RoutedSession, RouteError> {
    let (session, _, previous_attempts) =
        backend::select(request, Host::current(), |k| open_one(k, bytes, shapes))?;
    Ok(RoutedSession {
        session,
        previous_attempts,
    })
}
/// Production first-use routing: an adapter with an incomplete graph may open but fail to
/// dispatch. Cache only a session whose first actual run and output validation succeeded.
pub fn open_and_run<T>(
    request: BackendRequest,
    bytes: &[u8],
    shapes: &ShapeOverrides,
    inputs: &NamedInputs<'_>,
    mut validate: impl FnMut(NamedOutputs) -> Result<T, String>,
) -> Result<(RoutedSession, T), RouteError> {
    let ((session, output), _, previous_attempts) =
        backend::select(request, Host::current(), |kind| {
            let session = open_one(kind, bytes, shapes)?;
            if session.kind() != kind {
                return Err("factory returned a different device".into());
            }
            let output = validate(session.run_f32(inputs)?)?;
            Ok((session, output))
        })?;
    Ok((
        RoutedSession {
            session,
            previous_attempts,
        },
        output,
    ))
}

fn open_one(
    kind: BackendKind,
    bytes: &[u8],
    shapes: &ShapeOverrides,
) -> Result<Arc<dyn NativeInferBackend>, String> {
    match kind {
        #[cfg(windows)]
        BackendKind::Cuda => Ok(Arc::new(
            super::session::NativeCudaSession::from_bytes(bytes, shapes)
                .map_err(|e| e.to_string())?,
        )),
        #[cfg(windows)]
        BackendKind::Dx12 => Ok(Arc::new(
            super::dx12_session::NativeDx12Session::from_bytes(bytes, shapes)?,
        )),
        #[cfg(target_os = "macos")]
        BackendKind::Metal => Ok(Arc::new(
            super::metal_session::NativeMetalSession::from_bytes(bytes, shapes)?,
        )),
        BackendKind::Cpu => Ok(Arc::new(super::cpu_session::NativeCpuSession::from_bytes(
            bytes, shapes,
        )?)),
        _ => Err(format!(
            "{} model executor is not available on this build or host",
            kind.label()
        )),
    }
}
