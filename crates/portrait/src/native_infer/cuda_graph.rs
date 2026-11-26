//! Static CUDA Graph capture, replay and conservative fallback for
//! [`NativeCudaSession`](super::session::NativeCudaSession).
//!
//! A static ONNX model is a fixed sequence of kernel launches over fixed device
//! pointers. That is exactly what a CUDA Graph records, so the whole graph can
//! be replayed with one `cuGraphLaunch` instead of one `cuLaunchKernel` per
//! node — 1,511 host-to-driver round trips for `323773030.onnx`.
//!
//! The capture contract has three hard requirements, and every one of them is
//! enforced here rather than assumed:
//!
//! 1. **One set of device pointers.** Inputs, outputs, weights and the arena are
//!    allocated once and owned by [`StaticGraph`]; the record pass never falls
//!    back to a per-run allocation.
//! 2. **One stream and one context.** Replay runs on a dedicated non-blocking
//!    stream inside the process-wide primary context, and the host waits for
//!    exactly that stream before reading back.
//! 3. **No host round trips inside the graph.** A readback, an upload or a
//!    device-to-device copy between two nodes has no replay equivalent, so the
//!    recorder flags it and the recording is discarded.
//!
//! When any requirement fails the session keeps its ordinary execution path and
//! the refusal is reported through [`GraphStatus`] instead of surfacing as an
//! inference error: a capture that cannot be proven safe costs one wasted pass,
//! never a wrong result.
//!
//! `AURORA_NATIVE_CUDA_GRAPH=0` disables the whole feature; the default is on.

use std::collections::HashMap;
use std::fmt;

use crate::native_infer::graph::{IrGraph, Op};

/// Environment switch: `0`, `false`, `off` or `no` disable graph replay.
pub const SWITCH_ENV: &str = "AURORA_NATIVE_CUDA_GRAPH";

/// Relative tolerance for the one-time capture self-check.
///
/// Replay runs the same kernels on the same pointers in the same order, so the
/// expected difference is zero; the slack only absorbs kernels that reduce with
/// a non-deterministic order (block-level atomics) and keeps the check from
/// turning a numerically identical model into a false negative.
pub const VERIFY_TOLERANCE: f32 = 1e-5;

/// Parses the switch. Only an explicit off value disables capture.
pub fn switch_enabled(raw: Option<&str>) -> bool {
    match raw {
        None => true,
        Some(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "" | "false" | "off" | "no"
        ),
    }
}

/// [`switch_enabled`] for the process environment.
pub fn enabled_from_env() -> bool {
    switch_enabled(std::env::var(SWITCH_ENV).ok().as_deref())
}

/// One inference call's input names and shapes, sorted by name.
pub type InputSignature = Vec<(String, Vec<usize>)>;

/// Why graph replay is unavailable. Every variant is recoverable: the session
/// falls back to the plain path and keeps working.
#[derive(Clone, Debug, PartialEq)]
pub enum Fallback {
    /// `AURORA_NATIVE_CUDA_GRAPH` asked for the plain path.
    EnvDisabled,
    /// `AURORA_NATIVE_NO_ARENA=1`: without the static arena every intermediate
    /// tensor is a per-run allocation, so there are no stable pointers.
    NoArena,
    /// An output (or, at capture time, an intermediate) has no static shape.
    DynamicShape(String),
    /// `Op::If` / a branch with its own tensors.
    ControlFlow(String),
    /// A host readback or upload appeared between two recorded nodes.
    Readback(&'static str),
    /// A device allocation appeared between two recorded nodes.
    Allocation(&'static str),
    /// A `cuMemcpyDtoD` appeared between two recorded nodes.
    DeviceCopy(&'static str),
    /// The driver does not export the CUDA Graph entry points.
    Driver(&'static str),
    /// A CUDA Graph driver call failed.
    DriverCall(&'static str, i32),
    /// The record pass itself failed.
    RecordFailed(String),
    /// The graph had no node, so there is nothing to replay.
    EmptyGraph,
    /// The first replay disagreed with the plain path.
    VerifyFailed { max_abs: f32, tolerance: f32 },
    /// The replay produced a different output set or shape.
    VerifyShape(String),
    /// The replay failed after it had been accepted.
    LaunchFailed(String),
    /// Anything else the capture contract does not cover.
    Unsupported(String),
}

impl Fallback {
    /// Maps a recorded host-side effect to the fallback it proves.
    pub(crate) fn for_violation(what: &'static str) -> Self {
        match what {
            "device allocation" => Self::Allocation(what),
            "host readback" | "host upload" => Self::Readback(what),
            "device-to-device copy" => Self::DeviceCopy(what),
            other => Self::Unsupported(other.to_string()),
        }
    }
}

impl fmt::Display for Fallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EnvDisabled => write!(f, "disabled by {SWITCH_ENV}"),
            Self::NoArena => write!(f, "arena disabled (AURORA_NATIVE_NO_ARENA=1)"),
            Self::DynamicShape(tensor) => write!(f, "dynamic shape for tensor '{tensor}'"),
            Self::ControlFlow(node) => write!(f, "control flow at {node}"),
            Self::Readback(what) => write!(f, "host round trip during capture: {what}"),
            Self::Allocation(what) => write!(f, "allocation during capture: {what}"),
            Self::DeviceCopy(what) => write!(f, "device copy during capture: {what}"),
            Self::Driver(what) => write!(f, "{what} unavailable"),
            Self::DriverCall(call, code) => write!(f, "{call} failed with CUresult {code}"),
            Self::RecordFailed(detail) => write!(f, "record pass failed: {detail}"),
            Self::EmptyGraph => write!(f, "recording produced no node"),
            Self::VerifyFailed {
                max_abs,
                tolerance,
            } => write!(
                f,
                "replay drifted from the plain path: max|diff| {max_abs:.6e} > {tolerance:.1e}"
            ),
            Self::VerifyShape(detail) => write!(f, "replay output mismatch: {detail}"),
            Self::LaunchFailed(detail) => write!(f, "replay failed: {detail}"),
            Self::Unsupported(detail) => write!(f, "not statically capturable: {detail}"),
        }
    }
}

impl From<crate::cuda::GraphRecordError> for Fallback {
    fn from(error: crate::cuda::GraphRecordError) -> Self {
        match error {
            crate::cuda::GraphRecordError::Unavailable(what) => Self::Driver(what),
            crate::cuda::GraphRecordError::Driver(call, code) => Self::DriverCall(call, code),
            crate::cuda::GraphRecordError::NotStatic(what) => Self::for_violation(what),
            crate::cuda::GraphRecordError::Empty => Self::EmptyGraph,
        }
    }
}

impl From<crate::cuda::CudaError> for Fallback {
    fn from(error: crate::cuda::CudaError) -> Self {
        match error {
            crate::cuda::CudaError::Unavailable(what) => Self::Driver(what),
            crate::cuda::CudaError::Driver(call, code) => Self::DriverCall(call, code),
            crate::cuda::CudaError::Refused(refusal) => Self::Unsupported(format!("{refusal:?}")),
        }
    }
}

/// Observable state of the graph path, for logs, A/B reports and tests.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphStatus {
    /// The switch allowed capture when the session was opened.
    pub enabled: bool,
    /// A graph is installed and replays this session's calls.
    pub active: bool,
    /// Kernel nodes in the installed graph.
    pub nodes: usize,
    /// Record passes started.
    pub capture_attempts: usize,
    /// Record passes that produced an installed graph.
    pub captures: usize,
    /// Replays that returned outputs.
    pub replays: usize,
    /// First reason the graph path stopped; `None` while it is still available.
    pub fallback: Option<Fallback>,
    /// `max|diff|` measured when the installed graph was accepted.
    pub last_verify_max_abs: Option<f32>,
}

impl GraphStatus {
    /// One line for a log or a benchmark table.
    pub fn summary(&self) -> String {
        match (&self.fallback, self.active) {
            (_, true) => format!(
                "cuda_graph active nodes={} captures={} replays={} verify_max_abs={:.3e}",
                self.nodes,
                self.captures,
                self.replays,
                self.last_verify_max_abs.unwrap_or(0.0),
            ),
            (Some(fallback), false) => format!(
                "cuda_graph inactive after {} attempt(s): {fallback}",
                self.capture_attempts
            ),
            (None, false) => format!(
                "cuda_graph pending (enabled={}, attempts={})",
                self.enabled, self.capture_attempts
            ),
        }
    }
}

/// Builds the call signature from the graph's declared inputs.
///
/// `shape_of` resolves the shape the caller is about to submit, which is how a
/// model whose dimensions were patched by `input_overrides` still gets a
/// matching signature.
pub fn signature(
    graph: &IrGraph,
    shape_of: impl Fn(&str) -> Option<Vec<usize>>,
) -> Option<InputSignature> {
    let mut inputs = Vec::with_capacity(graph.inputs.len());
    for &id in &graph.inputs {
        let info = &graph.tensors[id.0];
        inputs.push((info.name.clone(), shape_of(&info.name)?));
    }
    inputs.sort_by(|a, b| a.0.cmp(&b.0));
    Some(inputs)
}

/// Rejects the models that are known not to be statically replayable.
///
/// This is the cheap first gate. The authoritative gate is the record pass
/// itself, which observes readbacks and allocations directly; a model that
/// passes here can still be rejected there.
pub fn precheck(graph: &IrGraph) -> Result<(), Fallback> {
    if graph.outputs.is_empty() {
        return Err(Fallback::Unsupported("graph has no outputs".into()));
    }
    for node in &graph.nodes {
        if matches!(node.op, Op::If(_)) {
            return Err(Fallback::ControlFlow(format!(
                "op 'If' at node '{}'",
                node.name
            )));
        }
    }
    for &out in &graph.outputs {
        let info = &graph.tensors[out.0];
        if info.static_shape().is_none() {
            return Err(Fallback::DynamicShape(info.name.clone()));
        }
    }
    Ok(())
}

/// Compares a replay against the plain path, returning `max|diff|`.
pub fn compare_outputs(
    expected: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    actual: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    tolerance: f32,
) -> Result<f32, Fallback> {
    if expected.len() != actual.len() {
        return Err(Fallback::VerifyShape(format!(
            "{} outputs vs {} replayed",
            expected.len(),
            actual.len()
        )));
    }
    let mut max_abs = 0.0f32;
    for (name, (want, want_shape)) in expected {
        let (got, got_shape) = actual
            .get(name)
            .ok_or_else(|| Fallback::VerifyShape(format!("missing output '{name}'")))?;
        if want_shape != got_shape {
            return Err(Fallback::VerifyShape(format!(
                "output '{name}' shape {got_shape:?} != {want_shape:?}"
            )));
        }
        if want.len() != got.len() {
            return Err(Fallback::VerifyShape(format!(
                "output '{name}' length {} != {}",
                got.len(),
                want.len()
            )));
        }
        for (index, (want_value, got_value)) in want.iter().zip(got.iter()).enumerate() {
            if want_value.is_nan() || got_value.is_nan() {
                if want_value.is_nan() && got_value.is_nan() {
                    continue;
                }
                return Err(Fallback::VerifyShape(format!(
                    "output '{name}'[{index}] NaN mismatch ({want_value} vs {got_value})"
                )));
            }
            if !want_value.is_finite() || !got_value.is_finite() {
                if want_value == got_value { continue; }
                return Err(Fallback::VerifyShape(format!("output '{name}'[{index}] infinite mismatch")));
            }
            let difference = (want_value - got_value).abs();
            max_abs = max_abs.max(difference);
            if difference > tolerance * want_value.abs().max(1.0) {
                return Err(Fallback::VerifyFailed {
                    max_abs: difference,
                    tolerance,
                });
            }
        }
    }
    Ok(max_abs)
}

#[cfg(windows)]
mod replay {
    use super::*;
    use crate::cuda::{self, CudaError, DeviceBuffer, GraphExec, Nvidia, PinnedBuffer, Stream};
    use crate::Refusal;
    use std::sync::Mutex;

    #[inline]
    fn f32_as_bytes(values: &[f32]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }

    #[inline]
    fn f32_as_bytes_mut(values: &mut [f32]) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                values.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(values),
            )
        }
    }

    /// One fixed input buffer of the graph.
    pub(crate) struct GraphInput {
        pub(crate) name: String,
        pub(crate) shape: Vec<usize>,
        pub(crate) bytes: usize,
        /// Staging storage remains alive until this call's stream sync (or exec drop).
        pub(crate) upload: PinnedBuffer<'static>,
        pub(crate) buffer: DeviceBuffer<'static>,
    }

    /// One fixed output buffer of the graph.
    pub(crate) struct GraphOutput {
        pub(crate) name: String,
        pub(crate) shape: Vec<usize>,
        /// Elements read back as `f32` (the session's result type).
        pub(crate) values: usize,
        pub(crate) buffer: DeviceBuffer<'static>,
    }

    /// An installed graph plus every buffer it replays against.
    ///
    /// The buffers are owned here on purpose: a replay that outlived the arena
    /// (or an input buffer) it was recorded with would write into freed device
    /// memory. Dropping this value destroys the executable first, then frees the
    /// buffers, and the executable's `Drop` drains the context before that.
    pub(crate) struct StaticGraph {
        exec: GraphExec,
        stream: Stream<'static>,
        signature: InputSignature,
        inputs: Vec<GraphInput>,
        outputs: Vec<GraphOutput>,
    }

    // SAFETY: the opaque CUstream belongs to the shared retained domain, not
    // to the thread that created it. Slot's mutex serializes launch/destruction;
    // each driver operation binds that domain on its calling thread. Restrict
    // Send to this private aggregate, rather than all streams or host buffers.
    unsafe impl Send for StaticGraph {}

    impl StaticGraph {
        pub(crate) fn new(
            exec: GraphExec,
            stream: Stream<'static>,
            signature: InputSignature,
            inputs: Vec<GraphInput>,
            outputs: Vec<GraphOutput>,
        ) -> Self {
            Self {
                exec,
                stream,
                signature,
                inputs,
                outputs,
            }
        }

        pub(crate) fn signature(&self) -> &InputSignature {
            &self.signature
        }

        pub(crate) fn nodes(&self) -> usize {
            self.exec.node_count()
        }

        /// H2D inputs, one `cuGraphLaunch`, stream sync, D2H outputs.
        pub(crate) fn launch(
            &mut self,
            gpu: &'static Nvidia,
            inputs: &HashMap<String, (&[f32], Vec<usize>)>,
        ) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>, CudaError> {
            for input in &mut self.inputs {
                let (values, shape) = inputs
                    .get(&input.name)
                    .ok_or(CudaError::Refused(Refusal::MissingInput("graph input")))?;
                if shape.as_slice() != input.shape.as_slice()
                    || values.len() * 4 != input.bytes
                {
                    return Err(CudaError::Refused(Refusal::ShapeMismatch("graph input")));
                }
                if !values.is_empty() {
                    // Pageable legacy-stream cudaMemcpy may return after staging, before
                    // this non-blocking graph stream can see the fresh pixels.
                    input.upload.copy_from(f32_as_bytes(values));
                    gpu.upload_async(&input.buffer, &input.upload.as_slice()[..input.bytes], &self.stream)?;
                }
            }
            gpu.graph_launch(&self.exec, &self.stream)?;
            // The replay stream is non-blocking, so the legacy-stream copies
            // below are only ordered against it by this explicit wait.
            gpu.stream_sync(&self.stream)?;
            let mut results = HashMap::with_capacity(self.outputs.len());
            for output in &self.outputs {
                let mut values = vec![0.0f32; output.values];
                if output.values > 0 {
                    gpu.download_buffer(&output.buffer, f32_as_bytes_mut(&mut values))?;
                }
                results.insert(output.name.clone(), (values, output.shape.clone()));
            }
            Ok(results)
        }
    }

    /// RAII wrapper around one armed recording.
    ///
    /// Dropping without [`Recording::finish`] discards the recording, so an
    /// error path cannot leave the thread armed or leak the container graph.
    pub(crate) struct Recording {
        finished: bool,
    }

    impl Recording {
        pub(crate) fn begin() -> Result<Self, Fallback> {
            cuda::graph_record_begin()?;
            Ok(Self { finished: false })
        }

        /// First host-side effect that has no replay equivalent, if any.
        pub(crate) fn violation(&self) -> Option<&'static str> {
            cuda::graph_record_first_violation()
        }

        pub(crate) fn finish(mut self) -> Result<GraphExec, Fallback> {
            self.finished = true;
            if let Some(what) = cuda::graph_record_first_violation() {
                cuda::graph_record_abort();
                return Err(Fallback::for_violation(what));
            }
            cuda::graph_record_end().map_err(Fallback::from)
        }
    }

    impl Drop for Recording {
        fn drop(&mut self) {
            if !self.finished {
                cuda::graph_record_abort();
            }
        }
    }

    /// Outcome of trying to answer one call from the installed graph.
    pub(crate) enum SlotRun {
        /// The replay produced the outputs.
        Replay(HashMap<String, (Vec<f32>, Vec<usize>)>),
        /// No usable graph for this call: run the plain path.
        Plain,
    }

    struct Slot {
        enabled: bool,
        graph: Option<StaticGraph>,
        status: GraphStatus,
        /// Signature the last record pass was started for, so a rejected model
        /// is not re-recorded on every call.
        attempted: Option<InputSignature>,
    }

    /// Upper bound on record passes per session.
    ///
    /// A capture costs one extra plain pass plus a verified replay, so a model
    /// whose call shape keeps alternating must not turn every inference into a
    /// capture: after this many passes the session serves the plain path.
    const MAX_CAPTURE_ATTEMPTS: usize = 4;

    /// The session's graph slot: one installed graph plus the fallback state.
    pub(crate) struct GraphSlot {
        inner: Mutex<Slot>,
    }

    impl GraphSlot {
        pub(crate) fn new() -> Self {
            let enabled = enabled_from_env();
            Self {
                inner: Mutex::new(Slot {
                    enabled,
                    graph: None,
                    status: GraphStatus {
                        enabled,
                        ..GraphStatus::default()
                    },
                    attempted: None,
                }),
            }
        }

        pub(crate) fn status(&self) -> GraphStatus {
            self.lock().status.clone()
        }

        pub(crate) fn is_enabled(&self) -> bool {
            self.lock().enabled
        }

        /// Records why the graph path is off. The first reason wins: later
        /// probes would only overwrite the cause with its consequence.
        pub(crate) fn disable(&self, fallback: Fallback) {
            let mut slot = self.lock();
            if slot.status.fallback.is_none() {
                slot.status.fallback = Some(fallback);
            }
            slot.graph = None;
            slot.status.active = false;
            slot.status.nodes = 0;
        }

        /// Installs a verified graph, replacing any previous one.
        pub(crate) fn install(&self, graph: StaticGraph, verify_max_abs: f32) {
            let mut slot = self.lock();
            slot.status.nodes = graph.nodes();
            slot.graph = Some(graph);
            slot.status.active = true;
            slot.status.captures += 1;
            slot.status.last_verify_max_abs = Some(verify_max_abs);
        }

        /// Starts a record pass for `signature`; `false` means it was already
        /// tried for this call shape.
        pub(crate) fn begin_capture(&self, signature: &InputSignature) -> bool {
            let mut slot = self.lock();
            if !slot.enabled || slot.status.fallback.is_some() {
                return false;
            }
            if slot.status.capture_attempts >= MAX_CAPTURE_ATTEMPTS {
                return false;
            }
            if slot.attempted.as_ref() == Some(signature) {
                return false;
            }
            slot.attempted = Some(signature.clone());
            slot.status.capture_attempts += 1;
            true
        }

        /// Answers one call from the installed graph when the signature matches.
        pub(crate) fn run(
            &self,
            gpu: &'static Nvidia,
            inputs: &HashMap<String, (&[f32], Vec<usize>)>,
            signature: &InputSignature,
        ) -> SlotRun {
            let mut slot = self.lock();
            let Some(mut graph) = slot.graph.take() else {
                return SlotRun::Plain;
            };
            if graph.signature() != signature {
                // A plain pass may reuse the host-evaluated staging pool under a new shape.
                // Drop the old executable before any of its inputs can be overwritten.
                drop(graph);
                slot.status.active = false;
                slot.status.nodes = 0;
                slot.attempted = None;
                return SlotRun::Plain;
            }
            match graph.launch(gpu, inputs) {
                Ok(results) => {
                    slot.graph = Some(graph);
                    slot.status.replays += 1;
                    SlotRun::Replay(results)
                }
                Err(error) => {
                    // The graph is dropped here: its executable is destroyed
                    // before the plain path runs, so no half-applied replay can
                    // reach the next call.
                    drop(graph);
                    slot.status.active = false;
                    slot.status.nodes = 0;
                    slot.status.fallback = Some(Fallback::LaunchFailed(format!("{error:?}")));
                    SlotRun::Plain
                }
            }
        }

        /// Drops the installed graph while the session still owns its buffers.
        pub(crate) fn clear(&mut self) {
            if let Ok(slot) = self.inner.get_mut() {
                slot.graph = None;
                slot.status.active = false;
                slot.status.nodes = 0;
            }
        }

        fn lock(&self) -> std::sync::MutexGuard<'_, Slot> {
            self.inner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        }
    }
}

#[cfg(windows)]
pub(crate) use replay::{GraphInput, GraphOutput, GraphSlot, Recording, SlotRun, StaticGraph};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_infer::graph::{IfAttrs, IrNode, NodeId, TensorId, TensorInfo};
    use crate::native_infer::onnx_proto::DataType;

    fn tensor(id: usize, name: &str, shape: Option<Vec<Option<usize>>>) -> TensorInfo {
        TensorInfo {
            id: TensorId(id),
            name: name.into(),
            dtype: DataType::Float,
            shape,
            is_weight: false,
            is_graph_input: false,
            is_graph_output: false,
        }
    }

    fn empty_graph(name: &str) -> IrGraph {
        IrGraph {
            name: name.into(),
            tensors: Vec::new(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            weights: HashMap::new(),
            name_to_id: HashMap::new(),
        }
    }

    #[test]
    fn switch_defaults_on_and_only_an_explicit_off_disables_it() {
        assert!(switch_enabled(None));
        assert!(switch_enabled(Some("1")));
        assert!(switch_enabled(Some("  on ")));
        assert!(!switch_enabled(Some("0")));
        assert!(!switch_enabled(Some("off")));
        assert!(!switch_enabled(Some("FALSE")));
        assert!(!switch_enabled(Some("")));
    }

    #[test]
    fn precheck_rejects_control_flow() {
        let mut graph = empty_graph("if-branch");
        graph.tensors.push(tensor(0, "y", Some(vec![Some(1)])));
        graph.outputs.push(TensorId(0));
        graph.nodes.push(IrNode {
            id: NodeId(0),
            name: "gate".into(),
            op: Op::Identity,
            inputs: Vec::new(),
            outputs: vec![Some(TensorId(0))],
        });
        assert_eq!(precheck(&graph), Ok(()));

        let branch = empty_graph("branch");
        graph.nodes[0].op = Op::If(IfAttrs {
            then_branch: Box::new(branch.clone()),
            else_branch: Box::new(branch),
        });
        assert!(matches!(
            precheck(&graph),
            Err(Fallback::ControlFlow(node)) if node.contains("gate")
        ));
    }

    #[test]
    fn precheck_rejects_dynamic_outputs_and_empty_graphs() {
        let mut graph = empty_graph("dynamic");
        graph.tensors.push(tensor(0, "mask", Some(vec![Some(1), None])));
        graph.outputs.push(TensorId(0));
        assert_eq!(
            precheck(&graph),
            Err(Fallback::DynamicShape("mask".into()))
        );

        let empty = empty_graph("empty");
        assert!(matches!(precheck(&empty), Err(Fallback::Unsupported(_))));
    }

    #[test]
    fn signature_follows_graph_input_order_and_sorts_by_name() {
        let mut graph = empty_graph("two-inputs");
        graph.tensors.push(tensor(0, "b", Some(vec![Some(2)])));
        graph.tensors.push(tensor(1, "a", Some(vec![Some(3)])));
        graph.inputs = vec![TensorId(0), TensorId(1)];
        let sig = signature(&graph, |name| match name {
            "b" => Some(vec![2]),
            "a" => Some(vec![3]),
            _ => None,
        })
        .expect("signature");
        assert_eq!(
            sig,
            vec![("a".to_string(), vec![3]), ("b".to_string(), vec![2])]
        );
        // An input the caller does not supply cannot have a signature.
        assert!(signature(&graph, |_| None).is_none());
    }

    #[test]
    fn compare_outputs_accepts_identical_values_and_rejects_drift() {
        let plain = HashMap::from([(
            "out".to_string(),
            (vec![0.0f32, 1.0, -2.5], vec![3usize]),
        )]);
        let same = plain.clone();
        assert_eq!(compare_outputs(&plain, &same, VERIFY_TOLERANCE), Ok(0.0));

        let mut near = plain.clone();
        near.get_mut("out").unwrap().0[1] = 1.0 + 1e-9;
        assert!(compare_outputs(&plain, &near, VERIFY_TOLERANCE).is_ok());

        let mut drifted = plain.clone();
        drifted.get_mut("out").unwrap().0[1] = 1.5;
        assert!(matches!(
            compare_outputs(&plain, &drifted, VERIFY_TOLERANCE),
            Err(Fallback::VerifyFailed { .. })
        ));

        let mut reshaped = plain.clone();
        reshaped.get_mut("out").unwrap().1 = vec![1, 3];
        assert!(matches!(
            compare_outputs(&plain, &reshaped, VERIFY_TOLERANCE),
            Err(Fallback::VerifyShape(_))
        ));

        assert!(matches!(
            compare_outputs(&plain, &HashMap::new(), VERIFY_TOLERANCE),
            Err(Fallback::VerifyShape(_))
        ));
    }

    #[test]
    fn status_summary_names_the_fallback() {
        let status = GraphStatus {
            enabled: true,
            capture_attempts: 1,
            fallback: Some(Fallback::Readback("host readback")),
            ..GraphStatus::default()
        };
        assert!(status.summary().contains("host readback"));

        let active = GraphStatus {
            enabled: true,
            active: true,
            nodes: 42,
            captures: 1,
            replays: 7,
            last_verify_max_abs: Some(0.0),
            ..GraphStatus::default()
        };
        assert!(active.summary().contains("nodes=42"));
    }

    #[test]
    fn violation_mapping_keeps_the_cause() {
        assert_eq!(
            Fallback::for_violation("host readback"),
            Fallback::Readback("host readback")
        );
        assert_eq!(
            Fallback::for_violation("device allocation"),
            Fallback::Allocation("device allocation")
        );
        assert!(matches!(
            Fallback::for_violation("something new"),
            Fallback::Unsupported(_)
        ));
    }
}
