//! Native Direct3D 12 inference session: the AMD / Intel counterpart of
//! [`super::session::NativeCudaSession`].
//!
//! The graph walk is shared with the other backends through
//! [`super::load_and_plan`] (proto decode, IR, BatchNorm folding, static shape
//! inference, arena planning). What this module owns is the operator program:
//! every node becomes a recorded `Dispatch` into `native/hlsl/infer_core.hlsl`,
//! and every dispatch is submitted on one compute queue and waited on through a
//! fence before any value is read back.
//!
//! Three rules make the numbers trustworthy:
//!
//! 1. **A graph with any unsupported operator is rejected in
//!    [`NativeDx12Session::from_bytes`], before the first dispatch.** Nothing is
//!    computed on the CPU and reported as a DX12 result.
//! 2. **`kind()` reports [`BackendKind::Dx12`] only for a session that opened a
//!    real AMD/Intel device** through [`crate::dx12::Dx12Device::open`].
//! 3. **Tensors are stored in 32-bit slots.** FP16 weights are widened on upload,
//!    `Cast(to=Float16)` still rounds through IEEE half, and integer/bool tensors
//!    are carried as exactly-representable 32-bit floats. `Int64` values above
//!    2^24 are therefore not exact, which is why the integer-only operators
//!    (`Gather`, `ScatterND`, `Shape`, `Size`, `Split`, `If`, …) are rejected
//!    rather than approximated. `Pad` and `Slice` are planned only when their
//!    parameter tensors are static and exact as integers: they are read back
//!    through [`TensorProto::to_i64_vec`], never through the 32-bit slot
//!    representation, and a graph whose pads/starts are computed at run time is
//!    refused instead of being approximated.

use std::collections::HashMap;
use std::sync::Mutex;

use super::backend::{BackendKind, NamedInputs, NamedOutputs, NativeInferBackend, ShapeOverrides};
use super::graph::{GemmAttrs, IrGraph, Op, ReduceAttrs, TensorId};
use super::load_and_plan_portable as load_and_plan;
use super::onnx_proto::DataType;
use crate::dx12::{
    AdapterInfo, AdapterRequest, Bindings, BoundBuffer, Dx12Device, GpuBuffer, Kernel, Params,
    PARAM_SLOTS,
};

/// The HLSL source every kernel is compiled from.
pub const HLSL_SOURCE: &str = include_str!("../../native/hlsl/infer_core.hlsl");

/// Precompile only the common inference kernels. This is a PSO warmup, not a RAW
/// shader, a timed inference or proof of support for a specific ONNX graph.
pub fn prewarm_common_pipelines() -> Result<(usize, usize), String> {
    let (device, _) = Dx12Device::open(&AdapterRequest::from_env()).map_err(|e| e.to_string())?;
    let (mut loaded, mut cached) = (0, 0);
    for entry in ["cs_binary", "cs_unary", "cs_matmul", "cs_conv2d"] {
        let (_, created) = device.compile_cached(HLSL_SOURCE, entry).map_err(|e| e.to_string())?;
        if created { loaded += 1; } else { cached += 1; }
    }
    Ok((loaded, cached))
}

/// Threads per group; a multiple of RDNA Wave32 and GCN/Vega Wave64.
const GROUP: u32 = 64;
/// GEMM tile edge, matching `GEMM_TILE` in the HLSL.
const GEMM_TILE: u32 = 16;

// ---------------------------------------------------------------------------
// Operator support table
// ---------------------------------------------------------------------------

/// Operators this backend has an HLSL kernel (or a view-only mapping) for.
///
/// `Shape`/`Size`/`Gather`/`ScatterND`/`Split`/`Resize`/`ConvInteger`/
/// `MatMulInteger`/`DynamicQuantizeLinear`/`ConvTranspose`/`If`/
/// `ConstantOfShape` are deliberately absent: they need exact 64-bit integer
/// arithmetic or data-dependent control flow, and approximating them on a
/// different device would be a silent correctness regression.
///
/// `Pad` and `Slice` are listed because their HLSL kernels exist, but they are
/// only *planned* when the pads/starts/ends/steps tensors are static integers:
/// [`op_supported`] accepts them and [`plan_node`] refuses a graph whose
/// parameters would have to be computed on the device.
pub const SUPPORTED_OPS: &[&str] = &[
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Sqrt",
    "Reciprocal",
    "Abs",
    "Sin",
    "Cos",
    "Relu",
    "PRelu",
    "LeakyRelu",
    "Sigmoid",
    "HardSwish",
    "HardSigmoid",
    "Erf",
    "Clip",
    "Softmax",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "And",
    "Where",
    "Transpose",
    "Reshape",
    "Identity",
    "Squeeze",
    "Unsqueeze",
    "Concat",
    "Expand",
    "Tile",
    "Cast",
    "Constant",
    "Pad",
    "Slice",
    "Conv",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "ReduceMean",
    "BatchNormalization",
    "BatchNormFused",
    "InstanceNormalization",
    "LayerNormalization",
    "MatMul",
    "Gemm",
];

/// The shader stores numeric values in f32 slots. Only narrow integers can
/// be represented exactly; never advertise arbitrary INT64 or DOUBLE casts.
fn cast_mode(to: &DataType) -> Option<u32> {
    Some(match to {
        DataType::Float => 0,
        DataType::Float16 => 1,
        DataType::Int8 => 2,
        DataType::Uint8 => 3,
        DataType::Bool => 4,
        DataType::Int16 => 5,
        DataType::Uint16 => 6,
        _ => return None,
    })
}

fn op_supported(op: &Op) -> bool {
    match op {
        // View-only operators: no dispatch, the output reuses the input buffer.
        Op::Identity | Op::Reshape { .. } | Op::Squeeze { .. } | Op::Unsqueeze { .. } => true,
        Op::Constant { .. } => true,
        // Only `constant` padding has a kernel; `edge`/`reflect` would need a
        // different index walk and are refused before any dispatch.
        Op::Pad { mode } => mode == "constant",
        Op::Slice => true,
        Op::Cast { to } => cast_mode(to).is_some(),
        Op::Conv(attrs) => attrs.kernel_shape.as_ref().is_none_or(|k| k.len() == 2)
            && attrs.strides.len()==2 && attrs.dilations.len()==2 && !attrs.strides.contains(&0) && !attrs.dilations.contains(&0),
        Op::MaxPool(attrs) | Op::AveragePool(attrs) => attrs.kernel_shape.len() == 2,
        Op::ReduceMean(attrs) => reduce_mean_supported(attrs),
        Op::Softmax { .. }
        | Op::Add
        | Op::Sub
        | Op::Mul
        | Op::Div
        | Op::Pow
        | Op::PRelu
        | Op::Equal
        | Op::Greater
        | Op::GreaterOrEqual
        | Op::Less
        | Op::And
        | Op::Where
        | Op::Sqrt
        | Op::Reciprocal
        | Op::Abs
        | Op::Sin
        | Op::Cos
        | Op::Relu
        | Op::LeakyRelu { .. }
        | Op::Sigmoid
        | Op::HardSwish
        | Op::HardSigmoid { .. }
        | Op::Erf
        | Op::Clip
        | Op::Transpose { .. }
        | Op::Concat { .. }
        | Op::Expand
        | Op::Tile
        | Op::GlobalAveragePool
        | Op::BatchNormalization(_)
        | Op::BatchNormFused
        | Op::InstanceNormalization { .. }
        | Op::LayerNormalization(_)
        | Op::MatMul
        | Op::Gemm(_) => true,
        _ => false,
    }
}

/// `ReduceMean` is planned for the whole tensor or for the last axis only; any
/// other axis set (or a runtime-computed `axes` input) is refused in `plan_node`.
fn reduce_mean_supported(attrs: &ReduceAttrs) -> bool {
    let _ = attrs;
    true
}

// ---------------------------------------------------------------------------
// Program representation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Placement {
    buffer: usize,
    numel: usize,
    shape: Vec<usize>,
    /// Stored element slots; always 4-byte here (see the module docs).
    dtype: DataType,
}

impl Placement {
}

/// One recorded compute dispatch, fully resolved at plan time.
#[derive(Clone, Debug)]
struct Step {
    entry: &'static str,
    params: Params,
    srvs: Vec<(u32, usize)>,
    uavs: Vec<(u32, usize)>,
    groups: (u32, u32, u32),
    /// `"node_name:Op"`, so a failure names the operator that ran.
    label: String,
}

/// The device plus everything it owns.
struct Engine {
    device: Dx12Device,
    buffers: Vec<GpuBuffer>,
    kernels: HashMap<&'static str, Kernel>,
}

/// Timing and geometry of the last successful [`NativeDx12Session::run_f32`].
#[derive(Clone, Debug, Default)]
pub struct RunStats {
    pub dispatches: u64,
    pub elapsed_ms: f64,
    pub input_bytes: usize,
    pub output_bytes: usize,
}

/// A compiled ONNX graph resident on an AMD / Intel D3D12 device.
pub struct NativeDx12Session {
    graph: IrGraph,
    engine: Mutex<Engine>,
    steps: Vec<Step>,
    inputs: Vec<(String, TensorId, Placement)>,
    outputs: Vec<(String, TensorId, Placement)>,
    placements: HashMap<TensorId, Placement>,
    activation_bytes: usize,
    adapter: AdapterInfo,
    last_run: Mutex<Option<RunStats>>,
}

impl NativeDx12Session {
    /// Decodes, plans, allocates and compiles `onnx_bytes` onto the AMD/Intel GPU
    /// selected by [`AdapterRequest::from_env`] (`AURORA_DX12_ADAPTER`).
    ///
    /// Returns an error before any allocation when the graph holds an operator
    /// this backend cannot execute on the GPU.
    pub fn from_bytes(onnx_bytes: &[u8], overrides: &ShapeOverrides) -> Result<Self, String> {
        Self::open(onnx_bytes, overrides, &AdapterRequest::from_env())
    }

    /// Same as [`Self::from_bytes`] with an explicit adapter request.
    pub fn open(
        onnx_bytes: &[u8],
        overrides: &ShapeOverrides,
        request: &AdapterRequest,
    ) -> Result<Self, String> {
        let plan = load_and_plan(onnx_bytes, overrides).map_err(|e| e.to_string())?;
        let graph = plan.graph;

        // 1. Reject an unsupported graph before touching the GPU.
        let mut unsupported: Vec<String> = Vec::new();
        for node in &graph.nodes {
            if !op_supported(&node.op) {
                unsupported.push(format!("{}:{}", node.name, node.op.name()));
            }
        }
        if !unsupported.is_empty() {
            unsupported.sort();
            unsupported.dedup();
            return Err(format!(
                "dx12 backend cannot execute {} operator(s) on the GPU: {}; refusing before any dispatch",
                unsupported.len(),
                unsupported.join(", ")
            ));
        }

        // 2. Every *live* tensor needs a static shape. Tensors orphaned by BatchNorm
        //    folding (the folded node's input keeps no shape) are not referenced by
        //    any node and are skipped.
        let mut live: std::collections::HashSet<TensorId> = std::collections::HashSet::new();
        live.extend(graph.inputs.iter().copied());
        live.extend(graph.outputs.iter().copied());
        for node in &graph.nodes {
            live.extend(node.inputs.iter().flatten().copied());
            live.extend(node.outputs.iter().flatten().copied());
        }
        for tensor in &graph.tensors {
            if !live.contains(&tensor.id) {
                continue;
            }
            if let Some(shape)=tensor.static_shape() {
                let bytes=shape.iter().try_fold(4usize,|b,&d| b.checked_mul(d));
                if bytes.is_none_or(|n|n>u32::MAX as usize) {return Err(format!("DX12 byte-address range exceeded for {}",tensor.name));}
            }
            if tensor.static_shape().is_none() {
                return Err(format!(
                    "dx12 backend needs static shapes, but tensor `{}` is unresolved",
                    tensor.name
                ));
            }
        }

        // 3. Open the real device.
        let (mut device, adapter) = Dx12Device::open(request).map_err(|e| e.to_string())?;

        let mut buffers: Vec<GpuBuffer> = Vec::new();
        let mut placements: HashMap<TensorId, Placement> = HashMap::new();
        let mut uploads: Vec<(usize, Vec<f32>)> = Vec::new();

        // 4. Static weights.
        for (&tid, proto) in &graph.weights {
            let shape = graph.tensors[tid.0].static_shape().unwrap_or_default();
            let values = proto
                .to_f32_vec()
                .map_err(|e| format!("weight `{}` decode error: {e}", graph.tensors[tid.0].name))?;
            let numel = shape.iter().product::<usize>().max(1);
            let buffer = buffers.len();
            buffers.push(
                device
                    .alloc(format!("weight:{}", graph.tensors[tid.0].name), (numel * 4) as u64)
                    .map_err(|e| e.to_string())?,
            );
            uploads.push((buffer, values));
            placements.insert(tid, Placement { buffer, numel, shape, dtype: DataType::Float });
        }

        // 5. Graph inputs get their own buffers, filled on every run.
        let mut inputs: Vec<(String, TensorId, Placement)> = Vec::new();
        for &tid in &graph.inputs {
            let info = &graph.tensors[tid.0];
            let shape = info.static_shape().unwrap_or_default();
            let numel = shape.iter().product::<usize>().max(1);
            let buffer = buffers.len();
            buffers.push(
                device
                    .alloc(format!("input:{}", info.name), (numel * 4) as u64)
                    .map_err(|e| e.to_string())?,
            );
            placements.insert(tid, Placement { buffer, numel, shape, dtype: DataType::Float });
            inputs.push((info.name.clone(), tid, placements[&tid].clone()));
        }

        let mut steps: Vec<Step> = Vec::new();
        let mut activation_bytes = 0usize;
        let mut scratch: HashMap<String, usize> = HashMap::new();

        // 6. Walk the graph: allocate each node output, emit its dispatch steps.
        for node in &graph.nodes {
            let mut ctx = NodeCtx { graph: &graph, placements: &mut placements };
            let emitted = plan_node(
                node,
                &mut ctx,
                &mut device,
                &mut buffers,
                &mut activation_bytes,
                &mut scratch,
                &mut uploads,
            )?;
            steps.extend(emitted);
        }

        if steps.is_empty(){return Err("DX12 plan has no GPU dispatch; no DX12 execution will be claimed".into());}

        // 7. Graph outputs must exist.
        let mut outputs = Vec::new();
        for &tid in &graph.outputs {
            let info = &graph.tensors[tid.0];
            let placement = placements.get(&tid).cloned().ok_or_else(|| {
                format!("graph output `{}` was never produced", info.name)
            })?;
            outputs.push((info.name.clone(), tid, placement));
        }

        // 8. Upload weights once; the queue is drained before returning.
        for (buffer, values) in &uploads {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            device
                .upload(&buffers[*buffer], &bytes)
                .map_err(|e| e.to_string())?;
        }
        device.submit().map_err(|e| e.to_string())?;

        // 9. Compile every kernel the program needs.
        let mut kernels = HashMap::new();
        for step in &steps {
            if kernels.contains_key(step.entry) {
                continue;
            }
            let (kernel, _) = device.compile_cached(HLSL_SOURCE, step.entry).map_err(|e| e.to_string())?;
            kernels.insert(step.entry, kernel);
        }

        let summary_bytes = activation_bytes;
        Ok(Self {
            graph,
            engine: Mutex::new(Engine { device, buffers, kernels }),
            steps,
            inputs,
            outputs,
            placements,
            activation_bytes: summary_bytes,
            adapter,
            last_run: Mutex::new(None),
        })
    }

    /// Storage plan of every tensor in the compiled program:
    /// `(tensor name, shape, stored dtype, buffer id)`.
    ///
    /// Every tensor here is stored in 4-byte slots, so `dtype` reports
    /// [`DataType::Float`] even for `Int64`/`Bool` graph tensors; the module docs
    /// explain why the integer-only operators are refused instead.
    pub fn tensor_storage(&self) -> Vec<(String, Vec<usize>, DataType, usize)> {
        let mut out: Vec<(String, Vec<usize>, DataType, usize)> = self
            .placements
            .iter()
            .map(|(tid, p)| {
                (
                    self.graph.tensors[tid.0].name.clone(),
                    p.shape.clone(),
                    p.dtype,
                    p.buffer,
                )
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// The adapter this session actually opened.
    pub fn adapter(&self) -> &AdapterInfo {
        &self.adapter
    }

    /// Bytes of GPU memory the activations of this program occupy.
    pub fn activation_bytes(&self) -> usize {
        self.activation_bytes
    }

    /// Number of recorded dispatches, i.e. work per `run_f32` (barrier-free).
    pub fn dispatch_count(&self) -> usize {
        self.steps.len()
    }

    /// The compiled graph.
    pub fn graph(&self) -> &IrGraph {
        &self.graph
    }

    /// Statistics of the last successful run.
    pub fn last_run(&self) -> Option<RunStats> {
        self.last_run.lock().ok().and_then(|s| s.clone())
    }

    /// Dispatch entry points used by this program, in program order.
    pub fn kernel_entries(&self) -> Vec<&'static str> {
        let mut seen = Vec::new();
        for step in &self.steps {
            if !seen.contains(&step.entry) {
                seen.push(step.entry);
            }
        }
        seen
    }

    fn execute(&self, inputs: &NamedInputs<'_>) -> Result<NamedOutputs, String> {
        let started = std::time::Instant::now();
        let mut guard = self
            .engine
            .lock()
            .map_err(|_| "dx12 session lock poisoned".to_string())?;
        let Engine { device, buffers, kernels } = &mut *guard;

        if inputs.len()!=self.inputs.len() {return Err("DX12 input count mismatch".into());}
        let before=device.dispatch_count();
        // 1. Upload the caller's inputs.
        let mut input_bytes = 0usize;
        for (name, _tid, placement) in &self.inputs {
            let (data, shape) = inputs
                .get(name)
                .ok_or_else(|| format!("missing graph input `{name}`"))?;
            if shape!=&placement.shape {return Err(format!("wrong shape for {name}: expected {:?}, got {shape:?}",placement.shape));}
            let numel: usize = shape.iter().try_fold(1usize,|a,&b|a.checked_mul(b)).ok_or("input dimensions overflow")?;
            if data.len() != numel {
                return Err(format!(
                    "input `{name}` has {} values but shape {shape:?} needs {numel}",
                    data.len()
                ));
            }
            if numel > placement.numel {
                return Err(format!(
                    "input `{name}` shape {shape:?} exceeds its planned buffer ({} elements)",
                    placement.numel
                ));
            }
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            input_bytes += bytes.len();
            device
                .upload(&buffers[placement.buffer], &bytes)
                .map_err(|e| e.to_string())?;
        }

        // 2. Record the program.
        for step in &self.steps {
            let mut bindings = Bindings::default();
            for (slot, buffer) in &step.srvs {
                bindings
                    .srvs
                    .push((*slot, BoundBuffer { buffer: &buffers[*buffer], byte_offset: 0 }));
            }
            for (slot, buffer) in &step.uavs {
                bindings
                    .uavs
                    .push((*slot, BoundBuffer { buffer: &buffers[*buffer], byte_offset: 0 }));
            }
            let kernel = kernels
                .get(step.entry)
                .ok_or_else(|| format!("kernel `{}` for {} was not compiled", step.entry, step.label))?;
            let (x,y,z)=step.groups;
            if x<=65535 || step.entry=="cs_matmul" {
                device.dispatch(kernel,&step.params,&bindings,(x,y,z)).map_err(|e|format!("{}: {e}",step.label))?;
            } else {
                // D3D12 caps each dispatch dimension at 65535. Use an immutable
                // parameter slot per section; atomics or reads from another device are unnecessary.
                if y!=1 || z!=1 {return Err(format!("{}: unsupported multi-dimensional large dispatch",step.label));}
                let mut base=0u32;
                while base<x {
                    let chunk=(x-base).min(65535);let mut params=step.params;params[31]=base;
                    device.dispatch(kernel,&params,&bindings,(chunk,1,1)).map_err(|e|format!("{}: {e}",step.label))?;
                    base+=chunk;
                }
            }
        }

        // 3. Execute and wait, then read the outputs back.
        device.submit().map_err(|e| e.to_string())?;
        let mut results = HashMap::with_capacity(self.outputs.len());
        let mut output_bytes = 0usize;
        for (name, _tid, placement) in &self.outputs {
            let bytes = device
                .read(&buffers[placement.buffer], placement.numel * 4)
                .map_err(|e| format!("readback of `{name}`: {e}"))?;
            output_bytes += bytes.len();
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            results.insert(name.clone(), (values, placement.shape.clone()));
        }
        let dispatches = device.dispatch_count().saturating_sub(before);
        if let Ok(mut slot) = self.last_run.lock() {
            *slot = Some(RunStats {
                dispatches,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                input_bytes,
                output_bytes,
            });
        }
        Ok(results)
    }
}

impl NativeInferBackend for NativeDx12Session {
    fn kind(&self) -> BackendKind {
        BackendKind::Dx12
    }

    fn device_detail(&self) -> Option<String> {
        let a=self.adapter();Some(format!("{} · VID {:04x} · UMA {}",a.name,a.vendor_id,a.uma))
    }

    fn arena_peak_bytes(&self) -> usize {
        self.activation_bytes
    }

    fn run_f32(&self, inputs: &NamedInputs<'_>) -> Result<NamedOutputs, String> {
        self.execute(inputs)
    }
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

struct NodeCtx<'a> {
    graph: &'a IrGraph,
    placements: &'a mut HashMap<TensorId, Placement>,
}

impl NodeCtx<'_> {
    fn shape(&self, tid: TensorId) -> Result<Vec<usize>, String> {
        self.graph.tensors[tid.0]
            .static_shape()
            .ok_or_else(|| format!("tensor `{}` has no static shape", self.graph.tensors[tid.0].name))
    }

    fn numel(&self, tid: TensorId) -> Result<usize, String> {
        Ok(self.shape(tid)?.iter().product::<usize>().max(1))
    }

    fn placement(&self, tid: TensorId) -> Result<&Placement, String> {
        self.placements
            .get(&tid)
            .ok_or_else(|| format!("tensor `{}` has no placement", self.graph.tensors[tid.0].name))
    }

    fn input(&self, node: &super::graph::IrNode, index: usize) -> Result<TensorId, String> {
        node.inputs
            .get(index)
            .copied()
            .flatten()
            .ok_or_else(|| format!("{} is missing required input #{index}", node.op.name()))
    }

    fn set(&mut self, tid: TensorId, placement: Placement) {
        self.placements.insert(tid, placement);
    }

    /// Allocates the activation buffer for one node output and records it.
    fn place_output(
        &mut self,
        device: &mut Dx12Device,
        buffers: &mut Vec<GpuBuffer>,
        activation_bytes: &mut usize,
        tid: TensorId,
        shape: Vec<usize>,
        numel: usize,
    ) -> Result<usize, String> {
        let buffer = buffers.len();
        let name = format!("act:{}", self.graph.tensors[tid.0].name);
        buffers.push(
            device
                .alloc(name, (numel * 4) as u64)
                .map_err(|e| e.to_string())?,
        );
        *activation_bytes += numel * 4;
        self.placements.insert(
            tid,
            Placement { buffer, numel, shape, dtype: DataType::Float },
        );
        Ok(buffer)
    }
}

fn ceil_div(a: usize, b: usize) -> u32 {
    if b == 0 {
        0
    } else {
        a.div_ceil(b) as u32
    }
}

fn params_from(values: &[u32]) -> Params {
    let mut params: Params = [0; PARAM_SLOTS];
    for (i, v) in values.iter().enumerate().take(PARAM_SLOTS) {
        params[i] = *v;
    }
    params
}

fn f32_bits(v: f32) -> u32 {
    v.to_bits()
}

/// Right-aligned broadcast strides of `in_shape` against `out_shape`.
fn broadcast_strides(in_shape: &[usize], out_shape: &[usize]) -> Result<Vec<usize>, String> {
    if in_shape.len() > out_shape.len() {
        return Err(format!(
            "cannot broadcast rank {} onto rank {}",
            in_shape.len(),
            out_shape.len()
        ));
    }
    let pad = out_shape.len() - in_shape.len();
    let mut strides = vec![0usize; out_shape.len()];
    let mut acc = 1usize;
    for axis in (0..in_shape.len()).rev() {
        let out_axis = axis + pad;
        let dim = in_shape[axis];
        let out_dim = out_shape[out_axis];
        if dim == out_dim {
            strides[out_axis] = acc;
        } else if dim == 1 {
            strides[out_axis] = 0;
        } else {
            return Err(format!(
                "cannot broadcast dim {dim} onto {out_dim} at axis {out_axis}"
            ));
        }
        acc *= dim;
    }
    Ok(strides)
}

fn encode_shape_params(
    base: usize,
    shape: &[usize],
    strides_a: &[usize],
    strides_b: &[usize],
) -> Vec<u32> {
    let _ = base;
    let mut p = vec![0u32; PARAM_SLOTS];
    p[0] = shape.iter().product::<usize>().max(1) as u32;
    p[1] = shape.len() as u32;
    for (d, dim) in shape.iter().enumerate() {
        p[2 + d] = *dim as u32;
    }
    for (d, s) in strides_a.iter().enumerate() {
        p[8 + d] = *s as u32;
    }
    for (d, s) in strides_b.iter().enumerate() {
        p[14 + d] = *s as u32;
    }
    p
}

fn binary_opcode(op: &Op) -> Option<u32> {
    Some(match op {
        Op::Add => 0,
        Op::Sub => 1,
        Op::Mul => 2,
        Op::Div => 3,
        Op::Pow => 4,
        Op::PRelu => 5,
        Op::Equal => 7,
        Op::Greater => 8,
        Op::Less => 9,
        Op::GreaterOrEqual => 10,
        Op::And => 11,
        _ => return None,
    })
}

fn unary_opcode(op: &Op) -> Option<u32> {
    Some(match op {
        Op::Identity => 0,
        Op::Sqrt => 1,
        Op::Reciprocal => 2,
        Op::Abs => 3,
        Op::Sin => 4,
        Op::Cos => 5,
        Op::Relu => 6,
        Op::Sigmoid => 7,
        Op::Erf => 8,
        Op::LeakyRelu { .. } => 9,
        Op::HardSigmoid { .. } => 10,
        Op::HardSwish => 11,
        Op::Clip => 12,
        _ => return None,
    })
}

/// Plan one graph node into zero or more dispatches.
#[allow(clippy::too_many_arguments)]
fn plan_node(
    node: &super::graph::IrNode,
    ctx: &mut NodeCtx<'_>,
    device: &mut Dx12Device,
    buffers: &mut Vec<GpuBuffer>,
    activation_bytes: &mut usize,
    scratch: &mut HashMap<String, usize>,
    uploads: &mut Vec<(usize, Vec<f32>)>,
) -> Result<Vec<Step>, String> {
    let label = format!("{}:{}", node.name, node.op.name());
    let mut steps = Vec::new();

    let out_id = node
        .outputs
        .first()
        .copied()
        .flatten()
        .ok_or_else(|| format!("{label} has no output"))?;
    let out_shape = ctx.shape(out_id)?;
    let out_numel = out_shape.iter().product::<usize>().max(1);


    // ---- view-only operators ------------------------------------------------
    match &node.op {
        Op::Identity | Op::Reshape { .. } | Op::Squeeze { .. } | Op::Unsqueeze { .. } => {
            let src = ctx.input(node, 0)?;
            let src_placement = ctx.placement(src)?.clone();
            if src_placement.numel != out_numel {
                return Err(format!(
                    "{label} would reshape {} elements into {out_numel}",
                    src_placement.numel
                ));
            }
            ctx.set(
                out_id,
                Placement { shape: out_shape, dtype: DataType::Float, ..src_placement },
            );
            return Ok(steps);
        }
        Op::Constant { value } => {
            let values = value
                .to_f32_vec()
                .map_err(|e| format!("{label}: constant decode error {e}"))?;
            let buffer = buffers.len();
            buffers.push(
                device
                    .alloc(format!("const:{}", ctx.graph.tensors[out_id.0].name), (out_numel * 4) as u64)
                    .map_err(|e| e.to_string())?,
            );
            uploads.push((buffer, values));
            ctx.set(
                out_id,
                Placement { buffer, numel: out_numel, shape: out_shape, dtype: DataType::Float },
            );
            return Ok(steps);
        }
        _ => {}
    }

    // ---- element-wise -------------------------------------------------------
    if let Some(opcode) = binary_opcode(&node.op) {
        let a = ctx.input(node, 0)?;
        let b = ctx.input(node, 1)?;
        let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
        let strides_a = broadcast_strides(&ctx.shape(a)?, &out_shape)?;
        let strides_b = broadcast_strides(&ctx.shape(b)?, &out_shape)?;
        if out_shape.len() > 6 {
            return Err(format!("{label}: rank {} exceeds the 6 axes the HLSL walker supports", out_shape.len()));
        }
        let mut p = encode_shape_params(0, &out_shape, &strides_a, &strides_b);
        p[20] = opcode;
        steps.push(Step {
            entry: "cs_binary",
            params: params_from(&p),
            srvs: vec![(0, ctx.placement(a)?.buffer), (1, ctx.placement(b)?.buffer)],
            uavs: vec![(0, out_buffer)],
            groups: (ceil_div(out_numel, GROUP as usize), 1, 1),
            label: label.clone(),
        });
        return Ok(steps);
    }

    if node.op == Op::Where {
        let condition = ctx.input(node, 0)?;
        let then_branch = ctx.input(node, 1)?;
        let else_branch = ctx.input(node, 2)?;
        let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
        let strides_c = broadcast_strides(&ctx.shape(condition)?, &out_shape)?;
        let strides_t = broadcast_strides(&ctx.shape(then_branch)?, &out_shape)?;
        let strides_e = broadcast_strides(&ctx.shape(else_branch)?, &out_shape)?;
        let mut p = encode_shape_params(0, &out_shape, &strides_c, &strides_t);
        for (d, s) in strides_e.iter().enumerate() {
            p[22 + d] = *s as u32;
        }
        steps.push(Step {
            entry: "cs_where",
            params: params_from(&p),
            srvs: vec![
                (0, ctx.placement(condition)?.buffer),
                (1, ctx.placement(then_branch)?.buffer),
                (2, ctx.placement(else_branch)?.buffer),
            ],
            uavs: vec![(0, out_buffer)],
            groups: (ceil_div(out_numel, GROUP as usize), 1, 1),
            label: label.clone(),
        });
        return Ok(steps);
    }

    if let Some(opcode) = unary_opcode(&node.op) {
        if out_shape.len() > 6 {
            return Err(format!("{label}: rank {} exceeds 6", out_shape.len()));
        }
        let src = ctx.input(node, 0)?;
        let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
        let mut p = vec![0u32; PARAM_SLOTS];
        p[0] = out_numel as u32;
        p[20] = opcode;
        match &node.op {
            Op::LeakyRelu { alpha } => p[21] = f32_bits(*alpha),
            Op::HardSigmoid { alpha, beta } => {
                p[21] = f32_bits(*alpha);
                p[22] = f32_bits(*beta);
            }
            Op::Clip => {
                // ONNX Clip(min, max) takes its bounds as inputs 1 and 2; they must
                // be static for the bounds to be a constant.
                let min = static_scalar(ctx, node.inputs.get(1).copied().flatten())?.unwrap_or(f32::NEG_INFINITY);
                let max = static_scalar(ctx, node.inputs.get(2).copied().flatten())?.unwrap_or(f32::INFINITY);
                p[21] = f32_bits(min);
                p[22] = f32_bits(max);
            }
            _ => {}
        }
        steps.push(Step {
            entry: "cs_unary",
            params: params_from(&p),
            srvs: vec![(0, ctx.placement(src)?.buffer)],
            uavs: vec![(0, out_buffer)],
            groups: (ceil_div(out_numel, GROUP as usize), 1, 1),
            label: label.clone(),
        });
        return Ok(steps);
    }

    // ---- shape/layout operators --------------------------------------------
    match &node.op {
        Op::Transpose { perm } => {
            let src = ctx.input(node, 0)?;
            let in_shape = ctx.shape(src)?;
            if out_shape.len() > 6 {
                return Err(format!("{label}: rank {} exceeds 6", out_shape.len()));
            }
            let perm = if perm.is_empty() {
                (0..in_shape.len()).rev().collect::<Vec<_>>()
            } else {
                perm.clone()
            };
            let in_strides = contiguous_strides(&in_shape);
            let mut strides = vec![0usize; out_shape.len()];
            for (out_axis, &in_axis) in perm.iter().enumerate() {
                strides[out_axis] = in_strides[in_axis];
            }
            let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
            let mut p = vec![0u32; PARAM_SLOTS];
            p[0] = out_numel as u32;
            p[1] = out_shape.len() as u32;
            for (d, dim) in out_shape.iter().enumerate() {
                p[2 + d] = *dim as u32;
            }
            for (d, s) in strides.iter().enumerate() {
                p[8 + d] = *s as u32;
            }
            steps.push(Step {
                entry: "cs_transpose",
                params: params_from(&p),
                srvs: vec![(0, ctx.placement(src)?.buffer)],
                uavs: vec![(0, out_buffer)],
                groups: (ceil_div(out_numel, GROUP as usize), 1, 1),
                label: label.clone(),
            });
            return Ok(steps);
        }
        Op::Concat { axis } => {
            let rank = out_shape.len();
            let axis = normalize_axis(*axis, rank)?;
            let inner: usize = out_shape[axis + 1..].iter().product();
            let axis_out = out_shape[axis];
            let mut start = 0usize;
            let inputs: Vec<TensorId> = node.inputs.iter().filter_map(|t| *t).collect();
            let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
            for (index, src) in inputs.iter().enumerate() {
                let in_shape = ctx.shape(*src)?;
                let in_axis = in_shape[axis];
                let count: usize = in_shape.iter().product::<usize>().max(1);
                let mut p = vec![0u32; PARAM_SLOTS];
                p[0] = count as u32;
                p[1] = axis_out as u32;
                p[2] = start as u32;
                p[3] = in_axis as u32;
                p[4] = inner as u32;
                p[5] = 0;
                p[6] = 0;
                steps.push(Step {
                    entry: "cs_concat_copy",
                    params: params_from(&p),
                    srvs: vec![(0, ctx.placement(*src)?.buffer)],
                    uavs: vec![(0, out_buffer)],
                    groups: (ceil_div(count, GROUP as usize), 1, 1),
                    label: format!("{label}[{index}]"),
                });
                start += in_axis;
            }
            if start != axis_out {
                return Err(format!("{label}: concat axis total {start} != {axis_out}"));
            }
            return Ok(steps);
        }
        Op::Cast { to } => {
            let src = ctx.input(node, 0)?;
            let mode = cast_mode(to).ok_or_else(|| format!("{label}: Cast to {to:?} is not implemented"))?;
            let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
            let mut p = vec![0u32; PARAM_SLOTS];
            p[0] = out_numel as u32;
            p[20] = mode;
            steps.push(Step {
                entry: "cs_cast",
                params: params_from(&p),
                srvs: vec![(0, ctx.placement(src)?.buffer)],
                uavs: vec![(0, out_buffer)],
                groups: (ceil_div(out_numel, GROUP as usize), 1, 1),
                label: label.clone(),
            });
            return Ok(steps);
        }
        Op::Expand | Op::Tile => {
            let src = ctx.input(node, 0)?;
            let in_shape = ctx.shape(src)?;
            let (ax_in, ax_out) = match (in_shape.last(), out_shape.last()) {
                (Some(a), Some(b)) if *a > 0 && b % a == 0 => (*a, *b),
                _ => {
                    return Err(format!(
                        "{label}: only last-axis repetition is implemented (in {in_shape:?}, out {out_shape:?})"
                    ));
                }
            };
            if in_shape[..in_shape.len() - 1] != out_shape[..out_shape.len() - 1] {
                return Err(format!("{label}: only last-axis repetition is implemented"));
            }
            let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
            let p = params_from(&[out_numel as u32, ax_in as u32, (ax_out / ax_in) as u32, ax_out as u32]);
            steps.push(Step {
                entry: "cs_repeat_axis",
                params: p,
                srvs: vec![(0, ctx.placement(src)?.buffer)],
                uavs: vec![(0, out_buffer)],
                groups: (ceil_div(out_numel, GROUP as usize), 1, 1),
                label: label.clone(),
            });
            return Ok(steps);
        }
        Op::Softmax { axis } => {
            let rank = out_shape.len();
            let axis = normalize_axis(*axis, rank)?;
            if axis + 1 != rank {
                return Err(format!(
                    "{label}: Softmax is implemented for the last axis only (axis {axis} of rank {rank})"
                ));
            }
            let len = out_shape[rank - 1];
            let rows = out_numel / len.max(1);
            let src = ctx.input(node, 0)?;
            let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
            let p = params_from(&[rows as u32, len as u32, 0]);
            steps.push(Step {
                entry: "cs_softmax_last",
                params: p,
                srvs: vec![(0, ctx.placement(src)?.buffer)],
                uavs: vec![(0, out_buffer)],
                groups: (rows as u32, 1, 1),
                label: label.clone(),
            });
            return Ok(steps);
        }
        Op::ReduceMean(attrs) => {
            let src = ctx.input(node, 0)?;
            let in_shape = ctx.shape(src)?.to_vec();
            let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
            let inner = ctx.placement(src)?.numel;
            match plan_reduce_mean(ctx.graph, node, attrs, &in_shape, &label)? {
                ReduceMeanPlan::LastAxis { rows, len } => {
                    // Last-axis mean: one group per row, written as a per-row scalar.
                    let p = params_from(&[rows as u32, len as u32]);
                    steps.push(Step {
                        entry: "cs_reduce_rows",
                        params: p,
                        srvs: vec![(0, ctx.placement(src)?.buffer)],
                        uavs: vec![(0, out_buffer)],
                        groups: (rows as u32, 1, 1),
                        label: label.clone(),
                    });
                }
                ReduceMeanPlan::WholeTensor => {
                    // Whole-tensor mean in two passes through a partial-sum buffer.
                    let partials = ceil_div(inner, (GROUP as usize) * 4) as usize;
                    let key = format!("reduce_partials:{}", partials);
                    let scratch_buffer = match scratch.get(&key) {
                        Some(b) => *b,
                        None => {
                            let buffer = buffers.len();
                            buffers.push(
                                device
                                    .alloc(format!("scratch:{key}"), (partials * 4) as u64)
                                    .map_err(|e| e.to_string())?,
                            );
                            scratch.insert(key, buffer);
                            buffer
                        }
                    };
                    steps.push(Step {
                        entry: "cs_reduce_partial",
                        params: params_from(&[inner as u32, 0]),
                        srvs: vec![(0, ctx.placement(src)?.buffer)],
                        uavs: vec![(0, scratch_buffer)],
                        groups: (partials as u32, 1, 1),
                        label: format!("{label}:partial"),
                    });
                    steps.push(Step {
                        entry: "cs_reduce_finish",
                        params: params_from(&[partials as u32, f32_bits(inner as f32)]),
                        srvs: vec![(0, scratch_buffer)],
                        uavs: vec![(0, out_buffer)],
                        groups: (1, 1, 1),
                        label: format!("{label}:finish"),
                    });
                }
            }
            return Ok(steps);
        }
        Op::GlobalAveragePool => {
            let src = ctx.input(node, 0)?;
            let in_shape = ctx.shape(src)?;
            if in_shape.len() < 3 {
                return Err(format!("{label}: GlobalAveragePool needs rank >= 3"));
            }
            let planes: usize = in_shape[..2].iter().product();
            let len: usize = in_shape[2..].iter().product();
            let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
            let p = params_from(&[planes as u32, len as u32]);
            steps.push(Step {
                entry: "cs_global_avg_pool",
                params: p,
                srvs: vec![(0, ctx.placement(src)?.buffer)],
                uavs: vec![(0, out_buffer)],
                groups: (planes as u32, 1, 1),
                label: label.clone(),
            });
            return Ok(steps);
        }
        Op::MaxPool(attrs) | Op::AveragePool(attrs) => {
            let is_average = matches!(node.op, Op::AveragePool(_));
            let src = ctx.input(node, 0)?;
            let in_shape = ctx.shape(src)?;
            if in_shape.len() != 4 {
                return Err(format!("{label}: pooling is implemented for NCHW rank 4 only"));
            }
            if attrs.ceil_mode {return Err(format!("{label}: ceiling pooling edge windows are not implemented"));}
            if attrs.dilations.iter().any(|d| *d != 1) {
                return Err(format!("{label}: pooling dilation is not implemented"));
            }
            if is_average && attrs.count_include_pad {
                return Err(format!("{label}: count_include_pad=1 is not implemented"));
            }
            let (pad_top, pad_left) = pad_pair(&attrs.pads, 2)?;
            let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
            let p = params_from(&[
                out_numel as u32,
                in_shape[1] as u32,
                in_shape[2] as u32,
                in_shape[3] as u32,
                attrs.kernel_shape[0] as u32,
                attrs.kernel_shape[1] as u32,
                attrs.strides[0] as u32,
                attrs.strides[1] as u32,
                pad_top as u32,
                pad_left as u32,
                out_shape[2] as u32,
                out_shape[3] as u32,
                u32::from(is_average),
                0,
            ]);
            steps.push(Step {
                entry: "cs_pool",
                params: p,
                srvs: vec![(0, ctx.placement(src)?.buffer)],
                uavs: vec![(0, out_buffer)],
                groups: (ceil_div(out_numel, GROUP as usize), 1, 1),
                label: label.clone(),
            });
            return Ok(steps);
        }
        Op::Pad { mode } => {
            if mode != "constant" {
                return Err(format!(
                    "{label}: Pad mode `{mode}` has no DX12 kernel (constant only); refusing instead of \
                     approximating it"
                ));
            }
            let src = ctx.input(node, 0)?;
            let in_shape = ctx.shape(src)?;
            let rank = in_shape.len();
            if out_shape.len() != rank || rank > 6 {
                return Err(format!(
                    "{label}: Pad rank {rank} (output rank {}) is not supported (max 6 axes)",
                    out_shape.len()
                ));
            }
            let pads_id = node
                .inputs
                .get(1)
                .copied()
                .flatten()
                .ok_or_else(|| format!("{label}: Pad has no pads input"))?;
            let pads = static_i64_values(ctx, pads_id)?;
            if pads.len() != rank * 2 {
                return Err(format!(
                    "{label}: Pad needs {} pad values for rank {rank}, got {} (the opset-18 `axes` \
                     input is not implemented)",
                    rank * 2,
                    pads.len()
                ));
            }
            let mut begins = [0i32; 6];
            for axis in 0..rank {
                let begin = pads[axis];
                let end = pads[axis + rank];
                let dim = in_shape[axis] as i64;
                let expected = dim + begin + end;
                if expected != out_shape[axis] as i64 {
                    return Err(format!(
                        "{label}: Pad axis {axis}: {dim} {begin:+} {end:+} = {expected}, but the \
                         planned output axis is {}",
                        out_shape[axis]
                    ));
                }
                if out_shape[axis] == 0 {
                    return Err(format!("{label}: Pad produces an empty axis {axis}"));
                }
                if begin < i32::MIN as i64 || begin > i32::MAX as i64 {
                    return Err(format!("{label}: Pad begin {begin} does not fit a 32-bit index"));
                }
                begins[axis] = begin as i32;
            }
            let fill = match node.inputs.get(2).copied().flatten() {
                Some(id) => static_scalar(ctx, Some(id))?.ok_or_else(|| {
                    format!(
                        "{label}: Pad constant_value `{}` is not a static tensor",
                        ctx.graph.tensors[id.0].name
                    )
                })?,
                None => 0.0,
            };
            // The pads already reconcile the input and output shapes, so every
            // interior index is inside the source; the kernel still keeps the
            // load inside a branch for the border lanes.
            let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
            let mut p = vec![0u32; PARAM_SLOTS];
            p[0] = out_numel as u32;
            p[1] = rank as u32;
            for (d, dim) in out_shape.iter().enumerate() {
                p[2 + d] = *dim as u32;
            }
            for (d, dim) in in_shape.iter().enumerate() {
                p[8 + d] = *dim as u32;
            }
            for (d, s) in contiguous_strides(&in_shape).iter().enumerate() {
                p[14 + d] = *s as u32;
            }
            for axis in 0..rank {
                p[20 + axis] = begins[axis] as u32;
            }
            p[26] = f32_bits(fill);
            steps.push(Step {
                entry: "cs_pad",
                params: params_from(&p),
                srvs: vec![(0, ctx.placement(src)?.buffer)],
                uavs: vec![(0, out_buffer)],
                groups: (ceil_div(out_numel, GROUP as usize), 1, 1),
                label: label.clone(),
            });
            return Ok(steps);
        }
        Op::Slice => {
            let src = ctx.input(node, 0)?;
            let in_shape = ctx.shape(src)?;
            let rank = in_shape.len();
            if out_shape.len() != rank || rank > 6 {
                return Err(format!(
                    "{label}: Slice rank {rank} (output rank {}) is not supported (max 6 axes)",
                    out_shape.len()
                ));
            }
            let input_id = |index: usize, what: &str| -> Result<TensorId, String> {
                node.inputs
                    .get(index)
                    .copied()
                    .flatten()
                    .ok_or_else(|| format!("{label}: Slice has no {what} input"))
            };
            let starts = static_i64_values(ctx, input_id(1, "starts")?)?;
            let ends = static_i64_values(ctx, input_id(2, "ends")?)?;
            if starts.is_empty() || starts.len() != ends.len() {
                return Err(format!(
                    "{label}: Slice needs one end per start ({} starts, {} ends)",
                    starts.len(),
                    ends.len()
                ));
            }
            let axes = match node.inputs.get(3).copied().flatten() {
                Some(id) => static_i64_values(ctx, id)?,
                None => (0..starts.len() as i64).collect(),
            };
            let steps_in = match node.inputs.get(4).copied().flatten() {
                Some(id) => static_i64_values(ctx, id)?,
                None => vec![1i64; starts.len()],
            };
            if axes.len() < starts.len() || steps_in.len() < starts.len() {
                return Err(format!(
                    "{label}: Slice has {} starts but {} axes and {} steps",
                    starts.len(),
                    axes.len(),
                    steps_in.len()
                ));
            }
            let mut start_full = vec![0i32; rank];
            let mut step_full = vec![1i32; rank];
            let mut sliced = vec![false; rank];
            for i in 0..starts.len() {
                let axis = normalize_axis(axes[i], rank)?;
                if sliced[axis] {
                    return Err(format!("{label}: Slice names axis {axis} twice"));
                }
                sliced[axis] = true;
                let dim = in_shape[axis] as i64;
                let step = steps_in[i];
                if step == 0 {
                    return Err(format!("{label}: Slice step 0 is not a valid slice"));
                }
                if step < i32::MIN as i64 || step > i32::MAX as i64 {
                    return Err(format!("{label}: Slice step {step} does not fit a 32-bit index"));
                }
                let start = clamp_slice_index(starts[i], dim, step > 0);
                let end = clamp_slice_index(ends[i], dim, step > 0);
                let length = if step > 0 {
                    if end <= start { 0 } else { (end - start + step - 1) / step }
                } else if start <= end {
                    0
                } else {
                    (start - end + (-step) - 1) / (-step)
                };
                if length.max(0) as usize != out_shape[axis] {
                    return Err(format!(
                        "{label}: Slice axis {axis} would produce {} elements but the planned output \
                         axis is {}",
                        length.max(0),
                        out_shape[axis]
                    ));
                }
                let last = start + (length - 1) * step;
                if length == 0 {
                    return Err(format!("{label}: Slice produces an empty axis {axis}"));
                }
                if start < 0 || last < 0 || start >= dim || last >= dim {                    return Err(format!(
                        "{label}: Slice axis {axis} would read {start}..{last}, outside 0..{dim}"
                    ));
                }
                start_full[axis] = start as i32;
                step_full[axis] = step as i32;
            }
            for axis in 0..rank {
                if !sliced[axis] && out_shape[axis] != in_shape[axis] {
                    return Err(format!(
                        "{label}: Slice keeps axis {axis} whole but the shape changes {} -> {}",
                        in_shape[axis], out_shape[axis]
                    ));
                }
            }
            let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
            let mut p = vec![0u32; PARAM_SLOTS];
            p[0] = out_numel as u32;
            p[1] = rank as u32;
            for (d, dim) in out_shape.iter().enumerate() {
                p[2 + d] = *dim as u32;
            }
            for (d, s) in contiguous_strides(&in_shape).iter().enumerate() {
                p[8 + d] = *s as u32;
            }
            for axis in 0..rank {
                p[14 + axis] = start_full[axis] as u32;
                p[20 + axis] = step_full[axis] as u32;
            }
            steps.push(Step {
                entry: "cs_slice",
                params: params_from(&p),
                srvs: vec![(0, ctx.placement(src)?.buffer)],
                uavs: vec![(0, out_buffer)],
                groups: (ceil_div(out_numel, GROUP as usize), 1, 1),
                label: label.clone(),
            });
            return Ok(steps);
        }
        _ => {}
    }

    // ---- convolution -------------------------------------------------------
    if let Op::Conv(attrs) = &node.op {
        let x = ctx.input(node, 0)?;
        let w = ctx.input(node, 1)?;
        let x_shape = ctx.shape(x)?;
        let w_shape = ctx.shape(w)?;
        if x_shape.len() != 4 || w_shape.len() != 4 {
            return Err(format!("{label}: Conv is implemented for NCHW rank 4 only"));
        }
        if out_shape.len() != 4 {
            return Err(format!("{label}: Conv output rank {} is not 4", out_shape.len()));
        }
        let group = attrs.group.max(1);
        if x_shape[1]%group!=0 || w_shape[1]!=x_shape[1]/group || w_shape[0]!=out_shape[1] ||
            attrs.kernel_shape.as_ref().is_some_and(|k|k.as_slice()!=&w_shape[2..]) {
            return Err(format!("{label}: Conv weight shape, channel count, group or declared kernel disagree"));
        }
        if out_shape[1] % group != 0 {
            return Err(format!("{label}: group {group} does not divide cout {}", out_shape[1]));
        }
        let (pad_top, pad_left) = pad_pair(&attrs.pads, 2)?;
        let bias = node.inputs.get(2).copied().flatten();
        let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
        let mut p = vec![0u32; PARAM_SLOTS];
        p[0] = out_numel as u32;
        p[1] = x_shape[0] as u32;
        p[2] = x_shape[1] as u32;
        p[3] = x_shape[2] as u32;
        p[4] = x_shape[3] as u32;
        p[5] = out_shape[1] as u32;
        p[6] = w_shape[2] as u32;
        p[7] = w_shape[3] as u32;
        p[8] = attrs.strides[0] as u32;
        p[9] = attrs.strides[1] as u32;
        p[10] = pad_top as u32;
        p[11] = pad_left as u32;
        p[12] = attrs.dilations[0] as u32;
        p[13] = attrs.dilations[1] as u32;
        p[14] = group as u32;
        p[15] = (x_shape[1] / group) as u32;
        p[16] = out_shape[2] as u32;
        p[17] = out_shape[3] as u32;
        p[18] = 0;
        p[19] = 0;
        p[20] = 0;
        p[21] = 0;
        p[22] = 0;
        let mut srvs = vec![(0u32, ctx.placement(x)?.buffer), (1u32, ctx.placement(w)?.buffer)];
        p[20] = 0;
        if let Some(bias_id) = bias {
            p[21] = 1;
            srvs.push((2, ctx.placement(bias_id)?.buffer));
        }
        steps.push(Step {
            entry: "cs_conv2d",
            params: params_from(&p),
            srvs,
            uavs: vec![(0, out_buffer)],
            groups: (ceil_div(out_numel, GROUP as usize), 1, 1),
            label: label.clone(),
        });
        return Ok(steps);
    }

    // ---- normalisation -----------------------------------------------------
    if node.op == Op::BatchNormFused {
        let x = ctx.input(node, 0)?;
        let scale = ctx.input(node, 1)?;
        let bias = ctx.input(node, 2)?;
        let in_shape = ctx.shape(x)?;
        if in_shape.len() < 3 {
            return Err(format!("{label}: BatchNormFused needs rank >= 3"));
        }
        let channels = in_shape[1];
        let spatial: usize = in_shape[2..].iter().product();
        let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
        let p = params_from(&[out_numel as u32, spatial as u32, channels as u32]);
        steps.push(Step {
            entry: "cs_affine_channel",
            params: p,
            srvs: vec![(0, ctx.placement(x)?.buffer), (1, ctx.placement(scale)?.buffer), (2, ctx.placement(bias)?.buffer)],
            uavs: vec![(0, out_buffer)],
            groups: (ceil_div(out_numel, GROUP as usize), 1, 1),
            label: label.clone(),
        });
        return Ok(steps);
    }

    if let Op::BatchNormalization(attrs) = &node.op {
        // Only reachable when folding did not apply (dynamic scale/bias). The
        // per-channel affine is precomputed on the host from static weights.
        let x = ctx.input(node, 0)?;
        let scale = ctx.input(node, 1)?;
        let bias = ctx.input(node, 2)?;
        let mean = ctx.input(node, 3)?;
        let var = ctx.input(node, 4)?;
        let channels = ctx.shape(x)?[1];
        let (scale_v, bias_v) = static_channel_affine(ctx, scale, bias, mean, var, attrs.epsilon, channels)?;
        let scale_buffer = buffers.len();
        buffers.push(
            device
                .alloc(format!("bn_scale:{}", node.name), (channels * 4) as u64)
                .map_err(|e| e.to_string())?,
        );
        uploads.push((scale_buffer, scale_v));
        let bias_buffer = buffers.len();
        buffers.push(
            device
                .alloc(format!("bn_bias:{}", node.name), (channels * 4) as u64)
                .map_err(|e| e.to_string())?,
        );
        uploads.push((bias_buffer, bias_v));
        let in_shape = ctx.shape(x)?;
        let spatial: usize = in_shape[2..].iter().product();
        let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
        let p = params_from(&[out_numel as u32, spatial as u32, channels as u32]);
        steps.push(Step {
            entry: "cs_affine_channel",
            params: p,
            srvs: vec![(0, ctx.placement(x)?.buffer), (1, scale_buffer), (2, bias_buffer)],
            uavs: vec![(0, out_buffer)],
            groups: (ceil_div(out_numel, GROUP as usize), 1, 1),
            label: label.clone(),
        });
        return Ok(steps);
    }

    if let Op::InstanceNormalization { epsilon } = &node.op {
        let x = ctx.input(node, 0)?;
        let scale = ctx.input(node, 1)?;
        let bias = ctx.input(node, 2)?;
        let in_shape = ctx.shape(x)?;
        if in_shape.len() < 3 {
            return Err(format!("{label}: InstanceNormalization needs rank >= 3"));
        }
        let planes: usize = in_shape[..2].iter().product();
        let len: usize = in_shape[2..].iter().product();
        let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
        let p = params_from(&[planes as u32, len as u32, f32_bits(*epsilon), in_shape[1] as u32, 1]);
        steps.push(Step {
            entry: "cs_instance_norm",
            params: p,
            srvs: vec![(0, ctx.placement(x)?.buffer), (1, ctx.placement(scale)?.buffer), (2, ctx.placement(bias)?.buffer)],
            uavs: vec![(0, out_buffer)],
            groups: (planes as u32, 1, 1),
            label: label.clone(),
        });
        return Ok(steps);
    }

    if let Op::LayerNormalization(attrs) = &node.op {
        let x = ctx.input(node, 0)?;
        let in_shape = ctx.shape(x)?;
        let rank = in_shape.len();
        let axis = normalize_axis(attrs.axis, rank)?;
        if axis + 1 != rank {
            return Err(format!(
                "{label}: LayerNormalization is implemented for the last axis only (axis {axis} of rank {rank})"
            ));
        }
        let len = in_shape[rank - 1];
        let rows = ctx.numel(x)? / len.max(1);
        let scale = node.inputs.get(1).copied().flatten();
        let bias = node.inputs.get(2).copied().flatten();
        let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
        let affine = scale.is_some() && bias.is_some();
        let p = params_from(&[
            rows as u32,
            len as u32,
            f32_bits(attrs.epsilon),
            u32::from(affine),
            len as u32,
        ]);
        let mut srvs = vec![(0u32, ctx.placement(x)?.buffer)];
        if affine {
            srvs.push((1, ctx.placement(scale.unwrap())?.buffer));
            srvs.push((2, ctx.placement(bias.unwrap())?.buffer));
        }
        steps.push(Step {
            entry: "cs_layer_norm",
            params: p,
            srvs,
            uavs: vec![(0, out_buffer)],
            groups: (rows as u32, 1, 1),
            label: label.clone(),
        });
        return Ok(steps);
    }

    // ---- matrix multiplication --------------------------------------------
    if node.op == Op::MatMul || matches!(node.op, Op::Gemm(_)) {
        let (attrs, a, b, c) = match &node.op {
            Op::Gemm(attrs) => (
                Some(attrs.clone()),
                ctx.input(node, 0)?,
                ctx.input(node, 1)?,
                node.inputs.get(2).copied().flatten(),
            ),
            _ => (None, ctx.input(node, 0)?, ctx.input(node, 1)?, None),
        };
        let a_shape = ctx.shape(a)?;
        let b_shape = ctx.shape(b)?;
        let out_buffer = ctx.place_output(device, buffers, activation_bytes, out_id, out_shape.clone(), out_numel)?;
        let (p, groups) = matmul_params(&label, &a_shape, &b_shape, &out_shape, attrs.as_ref(), c.is_some())?;
        let mut srvs = vec![(0u32, ctx.placement(a)?.buffer), (1u32, ctx.placement(b)?.buffer)];
        if let Some(c_id) = c {
            srvs.push((2, ctx.placement(c_id)?.buffer));
        }
        steps.push(Step {
            entry: "cs_matmul",
            params: params_from(&p),
            srvs,
            uavs: vec![(0, out_buffer)],
            groups,
            label: label.clone(),
        });
        return Ok(steps);
    }

    Err(format!("{label}: no DX12 program was planned for this operator"))
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = acc;
        acc *= shape[axis].max(1);
    }
    strides
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize, String> {
    let rank_i = rank as i64;
    let normalized = if axis < 0 { axis + rank_i } else { axis };
    if normalized < 0 || normalized >= rank_i {
        return Err(format!("axis {axis} is out of range for rank {rank}"));
    }
    Ok(normalized as usize)
}

/// Shape inference uses all four pads for the output extent. Each kernel only
/// needs the top/left origin because its per-tap bounds check excludes padding.
fn pad_pair(pads: &[usize], spatial:usize) -> Result<(usize, usize), String> {
    if spatial!=2 {return Err("only NCHW 2D windows are implemented".into());}
    match pads { []=>Ok((0,0)), [top,left,_,_]=>Ok((*top,*left)), _=>Err(format!("expected four 2D pads, got {}",pads.len())) }
}

/// Reads a static scalar from an initializer or a `Constant` node output.
fn static_scalar(ctx: &NodeCtx<'_>, tid: Option<TensorId>) -> Result<Option<f32>, String> {
    let Some(tid) = tid else { return Ok(None) };
    let tensor = &ctx.graph.tensors[tid.0];
    if tensor.is_graph_input {
        return Err(format!(
            "tensor `{}` is a graph input, so it cannot be a static scalar",
            tensor.name
        ));
    }
    if let Some(proto) = ctx.graph.weights.get(&tid) {
        let values = proto.to_f32_vec().map_err(|e| e.to_string())?;
        return Ok(values.first().copied());
    }
    Ok(None)
}

fn static_values(ctx: &NodeCtx<'_>, tid: TensorId) -> Result<Vec<f32>, String> {
    if let Some(proto) = ctx.graph.weights.get(&tid) {
        return proto.to_f32_vec().map_err(|e| e.to_string());
    }
    for node in &ctx.graph.nodes {
        if let Op::Constant { value } = &node.op {
            if node.outputs.first().copied().flatten() == Some(tid) {
                return value.to_f32_vec().map_err(|e| e.to_string());
            }
        }
    }
    Err(format!(
        "tensor `{}` is not a static tensor",
        ctx.graph.tensors[tid.0].name
    ))
}

/// Reads an exact integer parameter tensor (pads, starts, ends, axes, steps)
/// from an initializer or a `Constant` node.
///
/// Deliberately *not* [`static_values`]: those bytes would be narrowed to the
/// 32-bit float tensor slots, which cannot represent every `Int64`. A tensor
/// that is computed at run time is an error, not a zero.
fn graph_static_i64_values(graph: &IrGraph, tid: TensorId) -> Result<Vec<i64>, String> {
    let name = || graph.tensors[tid.0].name.as_str();
    if let Some(proto) = graph.weights.get(&tid) {
        return proto
            .to_i64_vec()
            .map_err(|e| format!("`{}` is not a static integer tensor: {e}", name()));
    }
    for node in &graph.nodes {
        if let Op::Constant { value } = &node.op {
            if node.outputs.first().copied().flatten() == Some(tid) {
                return value
                    .to_i64_vec()
                    .map_err(|e| format!("`{}` is not a static integer tensor: {e}", name()));
            }
        }
    }
    Err(format!(
        "`{}` is computed at run time, so it cannot parameterise a static Pad/Slice/ReduceMean",
        name()
    ))
}

fn static_i64_values(ctx: &NodeCtx<'_>, tid: TensorId) -> Result<Vec<i64>, String> {
    graph_static_i64_values(ctx.graph, tid)
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ReduceMeanPlan {
    LastAxis { rows: usize, len: usize },
    WholeTensor,
}

/// Resolves `ReduceMean` axes from either legacy `attrs.axes` or opset-18+
/// constant `inputs[1]`, refusing unresolved or unsupported axis sets instead
/// of silently falling back to whole-tensor mean.
fn plan_reduce_mean(
    graph: &IrGraph,
    node: &super::graph::IrNode,
    attrs: &ReduceAttrs,
    in_shape: &[usize],
    label: &str,
) -> Result<ReduceMeanPlan, String> {
    let axes = match &attrs.axes {
        Some(axes) => axes.clone(),
        None => match node.inputs.get(1).copied().flatten() {
            Some(axes_tid) => graph_static_i64_values(graph, axes_tid)
                .map_err(|e| format!("{label}: ReduceMean requires static axes ({e})"))?,
            None => Vec::new(),
        },
    };
    if axes.is_empty() {
        if attrs.noop_with_empty_axes {
            return Err(format!(
                "{label}: ReduceMean with noop_with_empty_axes=1 and empty axes is not supported"
            ));
        }
        return Ok(ReduceMeanPlan::WholeTensor);
    }

    let rank = in_shape.len();
    let mut norm_axes = Vec::with_capacity(axes.len());
    for axis in axes {
        let norm = normalize_axis(axis, rank)?;
        if !norm_axes.contains(&norm) {
            norm_axes.push(norm);
        }
    }

    if rank > 0 && norm_axes.len() == 1 && norm_axes[0] == rank - 1 {
        let len = in_shape[rank - 1];
        let inner = in_shape.iter().product::<usize>().max(1);
        let rows = inner / len.max(1);
        Ok(ReduceMeanPlan::LastAxis { rows, len })
    } else if norm_axes.len() == rank {
        Ok(ReduceMeanPlan::WholeTensor)
    } else {
        Err(format!(
            "{label}: ReduceMean over axes {norm_axes:?} for shape {in_shape:?} is not supported (only whole-tensor or last-axis reduction is supported)"
        ))
    }
}

/// ONNX Slice index normalisation, identical to the static-shape inference in
/// [`super::graph`]: a negative index counts from the end, then a positive step
/// clamps to `0..=dim` while a negative step clamps to `-1..=dim-1`.
fn clamp_slice_index(index: i64, dim: i64, positive_step: bool) -> i64 {
    let value = if index < 0 { index + dim } else { index };
    if positive_step {
        value.clamp(0, dim)
    } else {
        value.clamp(-1, dim - 1)
    }
}

/// Folds `y = (x - mean) / sqrt(var + eps) * scale + bias` into a per-channel
/// affine `(scale_eff, bias_eff)` from static tensors.
fn static_channel_affine(
    ctx: &NodeCtx<'_>,
    scale: TensorId,
    bias: TensorId,
    mean: TensorId,
    var: TensorId,
    epsilon: f32,
    channels: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let scale_v = static_values(ctx, scale)?;
    let bias_v = static_values(ctx, bias)?;
    let mean_v = static_values(ctx, mean)?;
    let var_v = static_values(ctx, var)?;
    if scale_v.len() < channels || bias_v.len() < channels || mean_v.len() < channels || var_v.len() < channels {
        return Err(format!(
            "BatchNormalization channel count {channels} exceeds its parameter tensors"
        ));
    }
    let mut scale_eff = vec![0f32; channels];
    let mut bias_eff = vec![0f32; channels];
    for c in 0..channels {
        let inv = 1.0 / (var_v[c] + epsilon).sqrt();
        scale_eff[c] = scale_v[c] * inv;
        bias_eff[c] = bias_v[c] - mean_v[c] * scale_eff[c];
    }
    Ok((scale_eff, bias_eff))
}

fn matmul_params(
    label: &str,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    attrs: Option<&GemmAttrs>,
    has_c: bool,
) -> Result<(Vec<u32>, (u32, u32, u32)), String> {
    let alpha = attrs.map(|a| a.alpha).unwrap_or(1.0);
    let beta = attrs.map(|a| a.beta).unwrap_or(0.0);
    let trans_a = attrs.map(|a| a.trans_a).unwrap_or(false);
    let trans_b = attrs.map(|a| a.trans_b).unwrap_or(false);

    // Rank 2 (plain) or rank 3 with the batch on both sides.
    let (m, k, a_batch, a_stride, lda) = match a_shape.len() {
        2 => {
            let (m, k) = if trans_a { (a_shape[1], a_shape[0]) } else { (a_shape[0], a_shape[1]) };
            (m, k, 1usize, 0usize, a_shape[1])
        }
        3 => {
            let (m, k) = if trans_a { (a_shape[2], a_shape[1]) } else { (a_shape[1], a_shape[2]) };
            (m, k, a_shape[0], a_shape[1] * a_shape[2], a_shape[2])
        }
        other => return Err(format!("{label}: MatMul operand rank {other} is not implemented")),
    };
    let (n, b_batch, ldb) = match b_shape.len() {
        2 => {
            let (kk, n) = if trans_b { (b_shape[1], b_shape[0]) } else { (b_shape[0], b_shape[1]) };
            if kk != k {
                return Err(format!("{label}: inner dimensions differ ({k} vs {kk})"));
            }
            (n, 1usize, b_shape[1])
        }
        3 => {
            let (kk, n) = if trans_b { (b_shape[2], b_shape[1]) } else { (b_shape[1], b_shape[2]) };
            if kk != k {
                return Err(format!("{label}: inner dimensions differ ({k} vs {kk})"));
            }
            (n, b_shape[0], b_shape[2])
        }
        other => return Err(format!("{label}: MatMul operand rank {other} is not implemented")),
    };
    let batch = a_batch.max(b_batch);
    if a_batch != 1 && b_batch != 1 && a_batch != b_batch {
        return Err(format!("{label}: incompatible batch sizes {a_batch} and {b_batch}"));
    }
    if out_shape.last().copied() != Some(n) {
        return Err(format!("{label}: planned output {out_shape:?} does not end in {n}"));
    }
    if m == 0 || n == 0 || k == 0 {
        return Err(format!("{label}: empty matmul {m}x{k}x{n}"));
    }
    let b_stride = match b_shape.len() {
        2 => 0usize,
        3 => b_shape[1] * b_shape[2],
        _ => unreachable!(),
    };
    // p[0]=M p[1]=K p[2]=N p[3]=batch p[4]=lda p[5]=ldb p[6]=ldc p[7]=transA
    // p[8]=transB p[9]=aBatch p[10]=bBatch p[11]=cBatch p[12]=alpha p[13]=beta p[14]=hasC
    let mut p = vec![0u32; PARAM_SLOTS];
    p[0] = m as u32;
    p[1] = k as u32;
    p[2] = n as u32;
    p[3] = batch as u32;
    p[4] = lda as u32;
    p[5] = ldb as u32;
    p[6] = n as u32;
    p[7] = u32::from(trans_a);
    p[8] = u32::from(trans_b);
    p[9] = a_stride as u32;
    p[10] = b_stride as u32;
    p[11] = (m * n) as u32;
    p[12] = f32_bits(alpha);
    p[13] = f32_bits(beta);
    p[14] = u32::from(has_c);
    let groups = (
        m.div_ceil(GEMM_TILE as usize) as u32,
        n.div_ceil(GEMM_TILE as usize) as u32,
        batch as u32,
    );
    Ok((p, (groups.1, groups.0, groups.2)))
}

// ---------------------------------------------------------------------------
// Tests: real AMD/Intel device, real dispatches, A/B against a CPU reference
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- minimal protobuf writer, enough for synthetic ONNX models ---------

    fn varint(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                b |= 0x80;
            }
            out.push(b);
            if v == 0 {
                return;
            }
        }
    }

    fn tag(out: &mut Vec<u8>, field: u32, wire: u8) {
        varint(out, ((field as u64) << 3) | wire as u64);
    }

    fn bytes_field(out: &mut Vec<u8>, field: u32, data: &[u8]) {
        tag(out, field, 2);
        varint(out, data.len() as u64);
        out.extend_from_slice(data);
    }

    fn str_field(out: &mut Vec<u8>, field: u32, s: &str) {
        bytes_field(out, field, s.as_bytes());
    }

    fn value_info(name: &str, dims: &[i64]) -> Vec<u8> {
        let mut shape = Vec::new();
        for &d in dims {
            let mut dim = Vec::new();
            tag(&mut dim, 1, 0);
            varint(&mut dim, d as u64);
            bytes_field(&mut shape, 1, &dim);
        }
        let mut tensor_type = Vec::new();
        tag(&mut tensor_type, 1, 0);
        varint(&mut tensor_type, DataType::Float as u64);
        bytes_field(&mut tensor_type, 2, &shape);
        let mut type_proto = Vec::new();
        bytes_field(&mut type_proto, 1, &tensor_type);
        let mut vi = Vec::new();
        str_field(&mut vi, 1, name);
        bytes_field(&mut vi, 2, &type_proto);
        vi
    }

    fn tensor_f32(name: &str, dims: &[i64], values: &[f32]) -> Vec<u8> {
        let mut t = Vec::new();
        for &d in dims {
            tag(&mut t, 1, 0);
            varint(&mut t, d as u64);
        }
        tag(&mut t, 2, 0);
        varint(&mut t, DataType::Float as u64);
        str_field(&mut t, 8, name);
        let mut raw = Vec::with_capacity(values.len() * 4);
        for v in values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        bytes_field(&mut t, 9, &raw);
        t
    }

    fn attr_int(name: &str, value: i64) -> Vec<u8> {
        let mut a = Vec::new();
        str_field(&mut a, 1, name);
        tag(&mut a, 3, 0);
        varint(&mut a, value as u64);
        tag(&mut a, 20, 0);
        varint(&mut a, 2); // INT
        a
    }

    fn attr_ints(name: &str, values: &[i64]) -> Vec<u8> {
        let mut a = Vec::new();
        str_field(&mut a, 1, name);
        for &v in values {
            tag(&mut a, 8, 0);
            varint(&mut a, v as u64);
        }
        tag(&mut a, 20, 0);
        varint(&mut a, 7); // INTS
        a
    }

    fn attr_string(name: &str, value: &str) -> Vec<u8> {
        let mut a = Vec::new();
        str_field(&mut a, 1, name);
        str_field(&mut a, 4, value);
        tag(&mut a, 20, 0);
        varint(&mut a, 3); // STRING
        a
    }

    /// `AttributeProto` carrying a tensor, as `Constant` uses for its `value`.
    fn attr_tensor(name: &str, tensor: &[u8]) -> Vec<u8> {
        let mut a = Vec::new();
        str_field(&mut a, 1, name);
        bytes_field(&mut a, 5, tensor);
        tag(&mut a, 20, 0);
        varint(&mut a, 4); // TENSOR
        a
    }

    /// `TensorProto` with an exact `Int64` payload: pads, starts, ends, steps.
    fn tensor_i64(name: &str, dims: &[i64], values: &[i64]) -> Vec<u8> {
        let mut t = Vec::new();
        for &d in dims {
            tag(&mut t, 1, 0);
            varint(&mut t, d as u64);
        }
        tag(&mut t, 2, 0);
        varint(&mut t, DataType::Int64 as u64);
        str_field(&mut t, 8, name);
        let mut raw = Vec::with_capacity(values.len() * 8);
        for v in values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        bytes_field(&mut t, 9, &raw);
        t
    }

    fn node(op: &str, name: &str, inputs: &[&str], outputs: &[&str], attrs: &[Vec<u8>]) -> Vec<u8> {
        let mut n = Vec::new();
        for i in inputs {
            str_field(&mut n, 1, i);
        }
        for o in outputs {
            str_field(&mut n, 2, o);
        }
        str_field(&mut n, 3, name);
        str_field(&mut n, 4, op);
        for a in attrs {
            bytes_field(&mut n, 5, a);
        }
        n
    }

    struct Model {
        nodes: Vec<Vec<u8>>,
        inits: Vec<Vec<u8>>,
        inputs: Vec<Vec<u8>>,
        outputs: Vec<Vec<u8>>,
        value_infos: Vec<Vec<u8>>,
    }

    impl Model {
        fn new() -> Self {
            Self {
                nodes: Vec::new(),
                inits: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                value_infos: Vec::new(),
            }
        }

        fn node(mut self, op: &str, name: &str, inputs: &[&str], outputs: &[&str], attrs: &[Vec<u8>]) -> Self {
            self.nodes.push(node(op, name, inputs, outputs, attrs));
            self
        }

        fn init(mut self, name: &str, dims: &[i64], values: &[f32]) -> Self {
            self.inits.push(tensor_f32(name, dims, values));
            self
        }

        fn init_i64(mut self, name: &str, dims: &[i64], values: &[i64]) -> Self {
            self.inits.push(tensor_i64(name, dims, values));
            self
        }

        fn constant(mut self, name: &str, output: &str, tensor: &[u8]) -> Self {
            self.nodes
                .push(node("Constant", name, &[], &[output], &[attr_tensor("value", tensor)]));
            self
        }

        fn input(mut self, name: &str, dims: &[i64]) -> Self {
            self.inputs.push(value_info(name, dims));
            self
        }

        fn output(mut self, name: &str, dims: &[i64]) -> Self {
            self.outputs.push(value_info(name, dims));
            self
        }

        /// Declares an intermediate tensor's type, for graphs whose shape
        /// inference needs the extra information.
        fn value_info(mut self, name: &str, dims: &[i64]) -> Self {
            self.value_infos.push(super::tests::value_info(name, dims));
            self
        }

        fn encode(&self) -> Vec<u8> {
            let mut g = Vec::new();
            str_field(&mut g, 2, "dx12_synthetic");
            for n in &self.nodes {
                bytes_field(&mut g, 1, n);
            }
            for i in &self.inits {
                bytes_field(&mut g, 5, i);
            }
            for i in &self.inputs {
                bytes_field(&mut g, 11, i);
            }
            for o in &self.outputs {
                bytes_field(&mut g, 12, o);
            }
            for v in &self.value_infos {
                bytes_field(&mut g, 13, v);
            }
            let mut model = Vec::new();
            tag(&mut model, 1, 0);
            varint(&mut model, 8);
            bytes_field(&mut model, 7, &g);
            model
        }
    }

    /// Deterministic pseudo-random values, so the A/B is not a constant field.
    fn seeded(len: usize, seed: u32, scale: f32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let unit = ((state >> 8) as f32) / ((1u32 << 24) as f32);
                (unit * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn amd_request() -> AdapterRequest {
        AdapterRequest::parse("amd")
    }

    /// CPU `Pad(mode="constant")`: the interior copies the source, every border
    /// element reads `fill`. Written straight from the operator definition, with
    /// no shared code with the HLSL kernel.
    fn pad_reference(
        in_shape: &[usize],
        out_shape: &[usize],
        pads: &[i64],
        fill: f32,
        src: &[f32],
    ) -> Vec<f32> {
        let rank = in_shape.len();
        let in_strides = contiguous_strides(in_shape);
        let out_numel: usize = out_shape.iter().product();
        let mut out = vec![fill; out_numel];
        for (index, slot) in out.iter_mut().enumerate() {
            let mut rem = index;
            let mut source = 0usize;
            let mut interior = true;
            for axis in (0..rank).rev() {
                let coord = rem % out_shape[axis];
                rem /= out_shape[axis];
                let shifted = coord as i64 - pads[axis];
                if shifted < 0 || shifted >= in_shape[axis] as i64 {
                    interior = false;
                } else {
                    source += shifted as usize * in_strides[axis];
                }
            }
            if interior {
                *slot = src[source];
            }
        }
        out
    }

    /// CPU `Slice` from already-normalised per-axis starts and steps:
    /// `output[coord] = input[start + coord * step]`.
    fn slice_reference(
        in_shape: &[usize],
        out_shape: &[usize],
        starts: &[i64],
        steps: &[i64],
        src: &[f32],
    ) -> Vec<f32> {
        let rank = in_shape.len();
        let in_strides = contiguous_strides(in_shape);
        let out_numel: usize = out_shape.iter().product();
        let mut out = vec![0f32; out_numel];
        for (index, slot) in out.iter_mut().enumerate() {
            let mut rem = index;
            let mut source = 0i64;
            for axis in (0..rank).rev() {
                let coord = rem % out_shape[axis];
                rem /= out_shape[axis];
                source += (starts[axis] + coord as i64 * steps[axis]) * in_strides[axis] as i64;
            }
            *slot = src[source as usize];
        }
        out
    }

    /// ONNX Slice window for one axis: the normalised start and the element
    /// count, taken from the operator spec rather than from the planner.
    fn slice_window(dim: i64, start: i64, end: i64, step: i64) -> (i64, i64) {
        assert!(step != 0, "a slice step of 0 is never valid");
        let clamp = |v: i64| if step > 0 { v.clamp(0, dim) } else { v.clamp(-1, dim - 1) };
        let s = clamp(if start < 0 { start + dim } else { start });
        let e = clamp(if end < 0 { end + dim } else { end });
        let len = if step > 0 {
            if e <= s {
                0
            } else {
                (e - s + step - 1) / step
            }
        } else if s <= e {
            0
        } else {
            (s - e + (-step) - 1) / (-step)
        };
        (s, len)
    }

    /// Static `Pad` (constant mode): border fill, channel padding, a negative
    /// crop and a crop-plus-fill mix, each compared to the CPU definition above.
    #[test]
    #[ignore = "needs a Windows host with an AMD or Intel D3D12 adapter"]
    fn static_pad_constant_matches_the_operator_definition() {
        // (input shape, pads, constant_value input)
        let cases: &[(&[i64], &[i64], Option<f32>)] = &[
            (&[1, 2, 3, 4], &[0, 0, 0, 0, 0, 0, 1, 1], None),
            (&[1, 2, 3, 4], &[0, 1, 0, 0, 0, 2, 0, 0], Some(2.25)),
            (&[2, 3, 4], &[1, 0, 0, 0, 2, 0], Some(-1.5)),
            (&[1, 1, 5, 5], &[0, 0, -1, -1, 0, 0, 1, 1], Some(9.0)),
        ];
        for (index, (shape, pads, fill)) in cases.iter().enumerate() {
            let rank = shape.len();
            let fill = *fill;
            let out_dims: Vec<i64> = (0..rank)
                .map(|axis| shape[axis] + pads[axis] + pads[axis + rank])
                .collect();
            assert!(out_dims.iter().all(|d| *d > 0), "case {index}: padding produced an empty axis");

            let mut builder = Model::new()
                .input("x", shape)
                .init_i64("pads", &[(rank * 2) as i64], pads);
            let mut node_inputs = vec!["x", "pads"];
            if let Some(value) = fill {
                builder = builder.init("fill", &[1], &[value]);
                node_inputs.push("fill");
            }
            let model = builder
                .node("Pad", "pad0", &node_inputs, &["y"], &[])
                .output("y", &out_dims)
                .encode();

            let session = NativeDx12Session::open(&model, &ShapeOverrides::default(), &amd_request())
                .unwrap_or_else(|e| panic!("case {index}: open AMD dx12 session: {e}"));
            assert_eq!(session.kernel_entries(), vec!["cs_pad"], "case {index}");
            assert_eq!(session.dispatch_count(), 1, "case {index}: one dispatch per Pad");

            let in_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let out_shape: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
            let numel: usize = in_shape.iter().product();
            let x = seeded(numel, 0x5eed_0000 + index as u32, 3.0);
            let mut inputs = NamedInputs::new();
            inputs.insert("x".to_string(), (x.as_slice(), in_shape.clone()));
            let outs = session
                .run_f32(&inputs)
                .unwrap_or_else(|e| panic!("case {index}: dx12 run: {e}"));
            let (got, got_shape) = &outs["y"];
            assert_eq!(got_shape, &out_shape, "case {index}: output shape");
            let reference = pad_reference(&in_shape, &out_shape, pads, fill.unwrap_or(0.0), &x);
            let diff = max_abs_diff(got, &reference);
            eprintln!(
                "[pad] case {index} {shape:?} pads={pads:?} fill={fill:?} -> {out_shape:?} \
                 adapter={:?} max_abs_diff={diff:.3e}",
                session.adapter().name
            );
            assert_eq!(diff, 0.0, "case {index}: Pad is an exact copy, max diff {diff:.3e}");
        }
    }

    /// Static `Slice`: a single-axis crop, a strided window, a reversed window
    /// and a two-axis window, plus one case whose starts/ends/axes/steps arrive
    /// as `Constant` nodes instead of initializers.
    #[test]
    #[ignore = "needs a Windows host with an AMD or Intel D3D12 adapter"]
    fn static_slice_matches_the_operator_definition() {
        struct Case {
            shape: &'static [i64],
            starts: &'static [i64],
            ends: &'static [i64],
            axes: Option<&'static [i64]>,
            steps: Option<&'static [i64]>,
            constant_params: bool,
        }
        let cases = [
            // blemishRemoval's own shape: crop 32 rows, then 32 columns.
            Case {
                shape: &[1, 3, 12, 11],
                starts: &[2],
                ends: &[-2],
                axes: Some(&[2]),
                steps: None,
                constant_params: false,
            },
            Case {
                shape: &[1, 3, 12, 11],
                starts: &[1],
                ends: &[-1],
                axes: Some(&[3]),
                steps: Some(&[2]),
                constant_params: false,
            },
            // A reversed window on the last axis: 5, 3, 1.
            Case {
                shape: &[2, 3, 6],
                starts: &[-1],
                ends: &[-7],
                axes: Some(&[2]),
                steps: Some(&[-2]),
                constant_params: false,
            },
            // Two axes at once, one of them reversed.
            Case {
                shape: &[1, 3, 8, 8],
                starts: &[1, -1],
                ends: &[7, -8],
                axes: Some(&[2, -1]),
                steps: Some(&[1, -1]),
                constant_params: false,
            },
            // Parameters carried by `Constant` nodes.
            Case {
                shape: &[1, 2, 6, 6],
                starts: &[1],
                ends: &[5],
                axes: Some(&[2]),
                steps: Some(&[1]),
                constant_params: true,
            },
        ];
        for (index, case) in cases.iter().enumerate() {
            let shape = case.shape;
            let rank = shape.len();
            let axes: Vec<i64> = case
                .axes
                .map(|a| a.to_vec())
                .unwrap_or_else(|| (0..case.starts.len() as i64).collect());
            let steps: Vec<i64> = case
                .steps
                .map(|s| s.to_vec())
                .unwrap_or_else(|| vec![1i64; case.starts.len()]);
            assert_eq!(axes.len(), case.starts.len(), "case {index}: axes/starts");
            assert_eq!(steps.len(), case.ends.len(), "case {index}: steps/ends");

            let mut out_dims = shape.to_vec();
            let mut full_starts = vec![0i64; rank];
            let mut full_steps = vec![1i64; rank];
            for i in 0..case.starts.len() {
                let axis = if axes[i] < 0 { axes[i] + rank as i64 } else { axes[i] } as usize;
                let (start, len) =
                    slice_window(shape[axis], case.starts[i], case.ends[i], steps[i]);
                out_dims[axis] = len;
                full_starts[axis] = start;
                full_steps[axis] = steps[i];
            }
            assert!(out_dims.iter().all(|d| *d > 0), "case {index}: empty window");

            let mut builder = Model::new().input("x", shape);
            builder = if case.constant_params {
                builder
                    .constant("c_starts", "starts", &tensor_i64("starts", &[case.starts.len() as i64], case.starts))
                    .constant("c_ends", "ends", &tensor_i64("ends", &[case.ends.len() as i64], case.ends))
                    .constant("c_axes", "axes", &tensor_i64("axes", &[axes.len() as i64], &axes))
                    .constant("c_steps", "steps", &tensor_i64("steps", &[steps.len() as i64], &steps))
            } else {
                builder
                    .init_i64("starts", &[case.starts.len() as i64], case.starts)
                    .init_i64("ends", &[case.ends.len() as i64], case.ends)
                    .init_i64("axes", &[axes.len() as i64], &axes)
                    .init_i64("steps", &[steps.len() as i64], &steps)
            };
            let model = builder
                .node(
                    "Slice",
                    "slice0",
                    &["x", "starts", "ends", "axes", "steps"],
                    &["y"],
                    &[],
                )
                .output("y", &out_dims)
                .encode();

            let session = NativeDx12Session::open(&model, &ShapeOverrides::default(), &amd_request())
                .unwrap_or_else(|e| panic!("case {index}: open AMD dx12 session: {e}"));
            assert_eq!(session.kernel_entries(), vec!["cs_slice"], "case {index}");
            assert_eq!(session.dispatch_count(), 1, "case {index}: one dispatch per Slice");

            let in_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let out_shape: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
            let numel: usize = in_shape.iter().product();
            let x = seeded(numel, 0x51ce_0000 + index as u32, 3.0);
            let mut inputs = NamedInputs::new();
            inputs.insert("x".to_string(), (x.as_slice(), in_shape.clone()));
            let outs = session
                .run_f32(&inputs)
                .unwrap_or_else(|e| panic!("case {index}: dx12 run: {e}"));
            let (got, got_shape) = &outs["y"];
            assert_eq!(got_shape, &out_shape, "case {index}: output shape");
            let reference = slice_reference(&in_shape, &out_shape, &full_starts, &full_steps, &x);
            let diff = max_abs_diff(got, &reference);
            eprintln!(
                "[slice] case {index} {shape:?} starts={:?} ends={:?} steps={steps:?} -> \
                 {out_shape:?} adapter={:?} max_abs_diff={diff:.3e}",
                case.starts,
                case.ends,
                session.adapter().name
            );
            assert_eq!(diff, 0.0, "case {index}: Slice is an exact copy, max diff {diff:.3e}");
        }
    }

    /// `edge`/`reflect` padding needs a different index walk; the graph must be
    /// refused before the device is opened, never approximated with zeros.
    #[test]
    fn pad_mode_other_than_constant_is_refused_before_any_dispatch() {
        let model = Model::new()
            .input("x", &[1, 1, 4, 4])
            .init_i64("pads", &[8], &[0, 0, 1, 1, 0, 0, 1, 1])
            .node("Pad", "pad0", &["x", "pads"], &["y"], &[attr_string("mode", "edge")])
            .output("y", &[1, 1, 6, 6])
            .encode();
        let err = match NativeDx12Session::open(&model, &ShapeOverrides::default(), &amd_request()) {
            Ok(_) => panic!("edge padding has no DX12 kernel and must be refused"),
            Err(e) => e,
        };
        assert!(err.contains("Pad"), "error should name the operator: {err}");
        assert!(
            err.contains("before any dispatch"),
            "error should state the refusal point: {err}"
        );
        eprintln!("[reject] {err}");
    }

    /// Pads that the static inference can resolve but the planner cannot read as
    /// an initializer or a `Constant` must stop the plan: filling zeros there
    /// would be a silent correctness regression.
    #[test]
    #[ignore = "needs a Windows host with an AMD or Intel D3D12 adapter"]
    fn pad_parameters_computed_at_run_time_are_refused_not_zero_filled() {
        let model = Model::new()
            .input("x", &[1, 1, 4, 4])
            .init_i64("pads_2d", &[1, 8], &[0, 0, 1, 1, 0, 0, 1, 1])
            .node("Squeeze", "squeeze0", &["pads_2d"], &["pads"], &[attr_ints("axes", &[0])])
            .node("Pad", "pad0", &["x", "pads"], &["y"], &[])
            .output("y", &[1, 1, 6, 6])
            .encode();
        let err = match NativeDx12Session::open(&model, &ShapeOverrides::default(), &amd_request()) {
            Ok(_) => panic!("run-time pads must not be planned"),
            Err(e) => e,
        };
        assert!(
            err.contains("computed at run time"),
            "error should say the pads are not static: {err}"
        );
        eprintln!("[reject] {err}");
    }

    /// A graph with an operator this backend has no GPU kernel for must be
    /// rejected at construction time, before any dispatch happens.
    #[test]
    fn unsupported_operator_is_rejected_before_any_dispatch() {
        let model = Model::new()
            .input("x", &[1, 4])
            .init("idx", &[2], &[0.0, 1.0])
            .node("Gather", "gather0", &["x", "idx"], &["y"], &[attr_int("axis", 1)])
            .output("y", &[1, 2])
            .encode();
        let err = match NativeDx12Session::open(&model, &ShapeOverrides::default(), &amd_request()) {
            Ok(_) => panic!("Gather has no DX12 kernel and must be refused"),
            Err(e) => e,
        };
        assert!(err.contains("Gather"), "error should name the operator: {err}");
        assert!(err.contains("before any dispatch"), "error should state the refusal point: {err}");
        eprintln!("[reject] {err}");
    }

    /// Element-wise + softmax + matmul chain: every value comes from a recorded
    /// dispatch on the AMD device, compared to a CPU reference.
    #[test]
    #[ignore = "needs a Windows host with an AMD or Intel D3D12 adapter"]
    fn elementwise_softmax_matmul_match_cpu_reference() {
        let (rows, cols, out_cols) = (4usize, 8usize, 3usize);
        let w = seeded(cols, 0x51ed, 1.0);
        let b = seeded(cols, 0x9e37, 0.5);
        let m = seeded(cols * out_cols, 0x1234_5678, 0.75);
        let model = Model::new()
            .input("x", &[1, rows as i64, cols as i64])
            .init("w", &[cols as i64], &w)
            .init("b", &[cols as i64], &b)
            .init("m", &[cols as i64, out_cols as i64], &m)
            .node("Mul", "mul0", &["x", "w"], &["t1"], &[])
            .node("Add", "add0", &["t1", "b"], &["t2"], &[])
            .node("Relu", "relu0", &["t2"], &["t3"], &[])
            .node("Softmax", "softmax0", &["t3"], &["t4"], &[attr_int("axis", 2)])
            .node("MatMul", "matmul0", &["t4", "m"], &["y"], &[])
            .output("y", &[1, rows as i64, out_cols as i64])
            .encode();

        let session = NativeDx12Session::open(&model, &ShapeOverrides::default(), &amd_request())
            .expect("open AMD dx12 session");
        let adapter = session.adapter().clone();
        assert_eq!(session.kind(), BackendKind::Dx12);
        eprintln!(
            "[gpu] #{} {:?} vendor=0x{:04X} ({}) uma={} dispatches={} kernels={:?} activation_bytes={}",
            adapter.index,
            adapter.name,
            adapter.vendor_id,
            adapter.vendor_label(),
            adapter.uma,
            session.dispatch_count(),
            session.kernel_entries(),
            session.arena_peak_bytes()
        );

        let x = seeded(rows * cols, 0xbeef, 1.5);
        let mut inputs = NamedInputs::new();
        inputs.insert("x".to_string(), (x.as_slice(), vec![1, rows, cols]));
        let outs = session.run_f32(&inputs).expect("dx12 run");
        let (got, shape) = &outs["y"];
        assert_eq!(shape, &vec![1, rows, out_cols]);

        // CPU reference straight from the operator definitions.
        let mut reference = vec![0f32; rows * out_cols];
        for r in 0..rows {
            let mut t3 = vec![0f32; cols];
            for c in 0..cols {
                let v = x[r * cols + c] * w[c] + b[c];
                t3[c] = v.max(0.0);
            }
            let max = t3.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = t3.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let soft: Vec<f32> = exps.iter().map(|v| v / sum).collect();
            for oc in 0..out_cols {
                reference[r * out_cols + oc] =
                    (0..cols).map(|k| soft[k] * m[k * out_cols + oc]).sum();
            }
        }
        let diff = max_abs_diff(got, &reference);
        eprintln!(
            "[elementwise] gpu[0..6]={:?} cpu[0..6]={:?} max_abs_diff={:.3e}",
            &got[..6.min(got.len())],
            &reference[..6.min(reference.len())],
            diff
        );
        assert!(diff < 1e-4, "max abs diff {diff:.3e} exceeds 1e-4");
        let stats = session.last_run().expect("run stats");
        assert!(stats.dispatches >= 4, "expected at least 4 dispatches, got {}", stats.dispatches);
        eprintln!(
            "[elementwise] dispatches={} elapsed={:.2}ms upload={}B readback={}B",
            stats.dispatches, stats.elapsed_ms, stats.input_bytes, stats.output_bytes
        );
    }

    /// Convolution + activation + global average pool on the GPU.
    #[test]
    #[ignore = "needs a Windows host with an AMD or Intel D3D12 adapter"]
    fn conv_relu_global_avg_pool_matches_cpu_reference() {
        let (n, cin, h, w) = (1usize, 3usize, 8usize, 8usize);
        let (cout, kh, kw, pad) = (4usize, 3usize, 3usize, 1i64);
        let weights = seeded(cout * cin * kh * kw, 0xa5a5, 0.5);
        let bias = seeded(cout, 0x5a5a, 0.2);
        let model = Model::new()
            .input("x", &[n as i64, cin as i64, h as i64, w as i64])
            .init("w", &[cout as i64, cin as i64, kh as i64, kw as i64], &weights)
            .init("bias", &[cout as i64], &bias)
            .node(
                "Conv",
                "conv0",
                &["x", "w", "bias"],
                &["c1"],
                &[
                    attr_ints("kernel_shape", &[kh as i64, kw as i64]),
                    attr_ints("pads", &[pad, pad, pad, pad]),
                    attr_ints("strides", &[1, 1]),
                    attr_ints("dilations", &[1, 1]),
                    attr_int("group", 1),
                ],
            )
            .node("Relu", "relu0", &["c1"], &["r1"], &[])
            .node("GlobalAveragePool", "gap0", &["r1"], &["y"], &[])
            .output("y", &[n as i64, cout as i64, 1, 1])
            .encode();

        let session = NativeDx12Session::open(&model, &ShapeOverrides::default(), &amd_request())
            .expect("open AMD dx12 session");
        eprintln!(
            "[conv] adapter={:?} uma={} dispatches={} kernels={:?}",
            session.adapter().name,
            session.adapter().uma,
            session.dispatch_count(),
            session.kernel_entries()
        );
        let x = seeded(n * cin * h * w, 0x0dd0, 1.0);
        let mut inputs = NamedInputs::new();
        inputs.insert("x".to_string(), (x.as_slice(), vec![n, cin, h, w]));
        let outs = session.run_f32(&inputs).expect("dx12 run");
        let (got, shape) = &outs["y"];
        assert_eq!(shape, &vec![n, cout, 1, 1]);

        // CPU reference convolution (same padding, stride 1, dilation 1).
        let mut conv = vec![0f32; n * cout * h * w];
        for ni in 0..n {
            for oc in 0..cout {
                for y in 0..h {
                    for xx in 0..w {
                        let mut acc = bias[oc];
                        for ic in 0..cin {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = y as i64 + ky as i64 - pad;
                                    let ix = xx as i64 + kx as i64 - pad;
                                    if iy < 0 || ix < 0 || iy >= h as i64 || ix >= w as i64 {
                                        continue;
                                    }
                                    let xv = x[(ni * cin + ic) * h * w + iy as usize * w + ix as usize];
                                    let wv = weights[((oc * cin + ic) * kh + ky) * kw + kx];
                                    acc += xv * wv;
                                }
                            }
                        }
                        conv[(ni * cout + oc) * h * w + y * w + xx] = acc.max(0.0);
                    }
                }
            }
        }
        let mut reference = vec![0f32; n * cout];
        for ni in 0..n {
            for oc in 0..cout {
                let plane = &conv[(ni * cout + oc) * h * w..(ni * cout + oc + 1) * h * w];
                reference[ni * cout + oc] = plane.iter().sum::<f32>() / (h * w) as f32;
            }
        }
        let diff = max_abs_diff(got, &reference);
        eprintln!(
            "[conv] gpu={:?}\n       cpu={:?}\n       max_abs_diff={:.3e}",
            got, reference, diff
        );
        assert!(diff < 1e-4, "max abs diff {diff:.3e} exceeds 1e-4");
    }

    /// The session must report the adapter it opened, not a requested one.
    #[test]
    #[ignore = "needs a Windows host with an AMD or Intel D3D12 adapter"]
    fn adapter_report_and_backend_kind_are_the_real_device() {
        let adapters = crate::dx12::enumerate_adapters().expect("enumeration");
        for a in &adapters {
            eprintln!(
                "[enum] #{} {:?} vendor=0x{:04X} ({}) uma={} eligible={} {}",
                a.index,
                a.name,
                a.vendor_id,
                a.vendor_label(),
                a.uma,
                a.eligible,
                a.excluded_reason.unwrap_or("-")
            );
        }
        let model = Model::new()
            .input("x", &[1, 8])
            .init("w", &[8], &vec![2.0f32; 8])
            .node("Mul", "mul0", &["x", "w"], &["y"], &[])
            .output("y", &[1, 8])
            .encode();
        let session =
            NativeDx12Session::open(&model, &ShapeOverrides::default(), &amd_request()).expect("open");
        let adapter = session.adapter();
        assert!(matches!(adapter.vendor_id, crate::dx12::VENDOR_AMD | crate::dx12::VENDOR_INTEL));
        assert!(adapter.hardware, "software rasterizers are never selected");
        eprintln!(
            "[adapter] session opened #{} {:?} vendor=0x{:04X} uma={} cache_coherent_uma={}",
            adapter.index,
            adapter.name,
            adapter.vendor_id,
            adapter.uma,
            adapter.cache_coherent_uma
        );
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut inputs = NamedInputs::new();
        inputs.insert("x".to_string(), (x.as_slice(), vec![1, 8]));
        let outs = session.run_f32(&inputs).expect("run");
        assert_eq!(outs["y"].0, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }


    /// Transpose / Concat / AveragePool / MaxPool / ReduceMean / LayerNormalization
    /// / Cast: one synthetic graph per operator, each compared to a CPU reference.
    #[test]
    #[ignore = "needs a Windows host with an AMD or Intel D3D12 adapter"]
    fn extended_ops_match_cpu_reference() {
        let request = amd_request();

        // --- Transpose [1,4,3] with perm [0,2,1] -> [1,3,4]
        let (rows, cols) = (4usize, 3usize);
        let x = seeded(rows * cols, 0x11, 1.0);
        let model = Model::new()
            .input("x", &[1, rows as i64, cols as i64])
            .node("Transpose", "tr0", &["x"], &["y"], &[attr_ints("perm", &[0, 2, 1])])
            .output("y", &[1, cols as i64, rows as i64])
            .encode();
        let session = NativeDx12Session::open(&model, &ShapeOverrides::default(), &request).expect("open");
        let mut inputs = NamedInputs::new();
        inputs.insert("x".to_string(), (x.as_slice(), vec![1, rows, cols]));
        let outs = session.run_f32(&inputs).expect("run");
        let mut reference = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                reference[c * rows + r] = x[r * cols + c];
            }
        }
        let diff = max_abs_diff(&outs["y"].0, &reference);
        eprintln!("[transpose] diff={diff:.3e} kernels={:?}", session.kernel_entries());
        assert!(diff < 1e-6, "Transpose diff {diff:.3e}");

        // --- Concat along axis 1: [1,2,4] + [1,3,4] -> [1,5,4]
        let a = seeded(2 * 4, 0x22, 1.0);
        let b = seeded(3 * 4, 0x33, 1.0);
        let model = Model::new()
            .input("a", &[1, 2, 4])
            .input("b", &[1, 3, 4])
            .node("Concat", "cat0", &["a", "b"], &["y"], &[attr_int("axis", 1)])
            .output("y", &[1, 5, 4])
            .encode();
        let session = NativeDx12Session::open(&model, &ShapeOverrides::default(), &request).expect("open");
        let mut inputs = NamedInputs::new();
        inputs.insert("a".to_string(), (a.as_slice(), vec![1, 2, 4]));
        inputs.insert("b".to_string(), (b.as_slice(), vec![1, 3, 4]));
        let outs = session.run_f32(&inputs).expect("run");
        let mut reference = Vec::new();
        reference.extend_from_slice(&a);
        reference.extend_from_slice(&b);
        let diff = max_abs_diff(&outs["y"].0, &reference);
        eprintln!("[concat] diff={diff:.3e}");
        assert!(diff < 1e-6, "Concat diff {diff:.3e}");

        // --- AveragePool and MaxPool, 4x4 -> 2x2 with kernel 2 stride 2
        let plane: Vec<f32> = (0..16).map(|i| i as f32 - 7.0).collect();
        for (op, is_avg) in [("AveragePool", true), ("MaxPool", false)] {
            let model = Model::new()
                .input("x", &[1, 1, 4, 4])
                .node(
                    op,
                    "pool0",
                    &["x"],
                    &["y"],
                    &[
                        attr_ints("kernel_shape", &[2, 2]),
                        attr_ints("strides", &[2, 2]),
                        attr_ints("pads", &[0, 0, 0, 0]),
                    ],
                )
                .output("y", &[1, 1, 2, 2])
                .encode();
            let session =
                NativeDx12Session::open(&model, &ShapeOverrides::default(), &request).expect("open");
            let mut inputs = NamedInputs::new();
            inputs.insert("x".to_string(), (plane.as_slice(), vec![1, 1, 4, 4]));
            let outs = session.run_f32(&inputs).expect("run");
            let mut reference = vec![0f32; 4];
            for oy in 0..2 {
                for ox in 0..2 {
                    let mut acc = if is_avg { 0.0 } else { f32::NEG_INFINITY };
                    for ky in 0..2 {
                        for kx in 0..2 {
                            let v = plane[(oy * 2 + ky) * 4 + ox * 2 + kx];
                            acc = if is_avg { acc + v } else { acc.max(v) };
                        }
                    }
                    reference[oy * 2 + ox] = if is_avg { acc / 4.0 } else { acc };
                }
            }
            let diff = max_abs_diff(&outs["y"].0, &reference);
            eprintln!("[{op}] gpu={:?} cpu={reference:?} diff={diff:.3e}", outs["y"].0);
            assert!(diff < 1e-6, "{op} diff {diff:.3e}");
        }

        // --- ReduceMean over every axis, keepdims=1 -> [1,1,1]
        let values = seeded(2 * 3 * 4, 0x44, 2.0);
        let model = Model::new()
            .input("x", &[2, 3, 4])
            .node("ReduceMean", "mean0", &["x"], &["y"], &[attr_int("keepdims", 1)])
            .output("y", &[1, 1, 1])
            .encode();
        let session = NativeDx12Session::open(&model, &ShapeOverrides::default(), &request).expect("open");
        let mut inputs = NamedInputs::new();
        inputs.insert("x".to_string(), (values.as_slice(), vec![2, 3, 4]));
        let outs = session.run_f32(&inputs).expect("run");
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let diff = (outs["y"].0[0] - mean).abs();
        eprintln!("[reduce_mean] gpu={} cpu={mean} diff={diff:.3e}", outs["y"].0[0]);
        assert!(diff < 1e-6, "ReduceMean diff {diff:.3e}");

        // --- ReduceMean over last axis via legacy `axes` attr and opset-18 `axes` input
        let row_ref: Vec<f32> = values
            .chunks_exact(4)
            .map(|row| row.iter().sum::<f32>() / 4.0)
            .collect();
        let legacy_model = Model::new()
            .input("x", &[2, 3, 4])
            .node(
                "ReduceMean",
                "mean_attr",
                &["x"],
                &["y"],
                &[attr_ints("axes", &[-1]), attr_int("keepdims", 1)],
            )
            .output("y", &[2, 3, 1])
            .encode();
        let sess = NativeDx12Session::open(&legacy_model, &ShapeOverrides::default(), &request).expect("open legacy last-axis ReduceMean");
        let outs = sess.run_f32(&inputs).expect("run legacy last-axis ReduceMean");
        let diff = max_abs_diff(&outs["y"].0, &row_ref);
        assert!(diff < 1e-6, "ReduceMean legacy attr last-axis diff {diff:.3e}");

        let opset18_model = Model::new()
            .input("x", &[2, 3, 4])
            .constant("c_axes", "axes", &tensor_i64("axes", &[1], &[-1]))
            .node(
                "ReduceMean",
                "mean_const",
                &["x", "axes"],
                &["y"],
                &[attr_int("keepdims", 1)],
            )
            .output("y", &[2, 3, 1])
            .encode();
        let sess = NativeDx12Session::open(&opset18_model, &ShapeOverrides::default(), &request).expect("open opset18 last-axis ReduceMean");
        let outs = sess.run_f32(&inputs).expect("run opset18 last-axis ReduceMean");
        let diff = max_abs_diff(&outs["y"].0, &row_ref);
        assert!(diff < 1e-6, "ReduceMean opset18 const axes last-axis diff {diff:.3e}");

        // --- LayerNormalization over the last axis, with scale and bias
        let (rows, len) = (4usize, 6usize);
        let x = seeded(rows * len, 0x55, 2.0);
        let scale: Vec<f32> = seeded(len, 0x66, 0.5).iter().map(|v| 1.0 + v).collect();
        let bias: Vec<f32> = seeded(len, 0x77, 0.3);
        let epsilon = 1e-5f32;
        let model = Model::new()
            .input("x", &[1, rows as i64, len as i64])
            .init("scale", &[len as i64], &scale)
            .init("bias", &[len as i64], &bias)
            .node(
                "LayerNormalization",
                "ln0",
                &["x", "scale", "bias"],
                &["y"],
                &[attr_int("axis", -1)],
            )
            .output("y", &[1, rows as i64, len as i64])
            .encode();
        let session = NativeDx12Session::open(&model, &ShapeOverrides::default(), &request).expect("open");
        let mut inputs = NamedInputs::new();
        inputs.insert("x".to_string(), (x.as_slice(), vec![1, rows, len]));
        let outs = session.run_f32(&inputs).expect("run");
        let mut reference = vec![0f32; rows * len];
        for r in 0..rows {
            let row = &x[r * len..(r + 1) * len];
            let mean = row.iter().sum::<f32>() / len as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / len as f32;
            let inv = 1.0 / (var + epsilon).sqrt();
            for c in 0..len {
                reference[r * len + c] = (row[c] - mean) * inv * scale[c] + bias[c];
            }
        }
        let diff = max_abs_diff(&outs["y"].0, &reference);
        eprintln!("[layer_norm] diff={diff:.3e}");
        assert!(diff < 1e-5, "LayerNormalization diff {diff:.3e}");

        // --- Cast to Float16 must round through IEEE half
        let raw = [0.1f32, 1.0 / 3.0, -2.71828, 65504.0, 1e-5];
        let model = Model::new()
            .input("x", &[1, raw.len() as i64])
            .node("Cast", "cast0", &["x"], &["y"], &[attr_int("to", 10)])
            .output("y", &[1, raw.len() as i64])
            .encode();
        let session = NativeDx12Session::open(&model, &ShapeOverrides::default(), &request).expect("open");
        let mut inputs = NamedInputs::new();
        inputs.insert("x".to_string(), (raw.as_slice(), vec![1, raw.len()]));
        let outs = session.run_f32(&inputs).expect("run");
        let reference: Vec<f32> = raw
            .iter()
            .map(|v| super::super::onnx_proto::f16_to_f32(super::super::onnx_proto::f32_to_f16(*v)))
            .collect();
        let diff = max_abs_diff(&outs["y"].0, &reference);
        eprintln!("[cast_f16] gpu={:?} cpu={reference:?} diff={diff:.3e}", outs["y"].0);
        assert_eq!(diff, 0.0, "Cast to Float16 must match IEEE half rounding");
        // Exact bit patterns across half-way underflow, overflow, signed zero and NaN.
        // These landmarks are mathematical expectations, not the shared converter.
        let edge=[f32::from_bits(0x3300_0000), f32::from_bits(0x3300_0001), -0.0,
                  -65520.0, 65520.0, f32::MAX, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
        let m=Model::new().input("x",&[9]).node("Cast","edge",&["x"],&["y"],&[attr_int("to",10)]).output("y",&[9]).encode();
        let sess=NativeDx12Session::open(&m,&ShapeOverrides::default(),&request).expect("open edge cast");
        let result=sess.run_f32(&NamedInputs::from([("x".into(),(edge.as_slice(),vec![9]))])).expect("run edge cast");
        let v=&result["y"].0;
        let wanted=[0u32,0x3380_0000,0x8000_0000,0xff80_0000,0x7f80_0000,0x7f80_0000,0x7f80_0000,0xff80_0000];
        for (index,&bits) in wanted.iter().enumerate(){assert_eq!(v[index].to_bits(),bits,"FP16 cast edge {index}");}
        assert!(v[8].is_nan(),"NaN classification must survive");

    }

    #[test]
    #[ignore="requires a real AMD/Intel GPU with DX12 compute"]
    fn asymmetric_conv_uses_weight_shape_and_trailing_padding() {
        // x = [[1,2,3],[4,5,6]], W = [[1,2],[4,8]], pads=[top1,left0,bottom0,right1].
        // The two output rows see [zeros,top] and [top,bottom], including the right-hand edge.
        let m=Model::new().input("x",&[1,1,2,3]).init("w",&[1,1,2,2],&[1.,2.,4.,8.])
            .node("Conv","conv",&["x","w"],&["y"],&[attr_ints("pads",&[1,0,0,1])])
            .output("y",&[1,1,2,3]).encode();
        let x=[1.,2.,3.,4.,5.,6.];let sess=NativeDx12Session::open(&m,&ShapeOverrides::default(),&amd_request()).expect("open asymmetric conv");
        let output=sess.run_f32(&NamedInputs::from([("x".into(),(x.as_slice(),vec![1,1,2,3]))])).expect("run asymmetric conv");
        assert_eq!(output["y"].1,vec![1,1,2,3]);
        assert_eq!(output["y"].0,vec![20.,32.,12.,61.,76.,27.]);
    }

    #[test]
    #[ignore="requires a real AMD/Intel GPU with DX12 compute"]
    fn large_1d_dispatch_writes_across_the_65535_group_boundary(){
        let n=65535*64+3;let mut x:Vec<f32>=(0..n).map(|j|(j%257) as f32-100.).collect();
        // A never-written tail could read all zero; these sentinels must survive the second dispatch.
        x[n-3..].copy_from_slice(&[311.0, 312.0, 313.0]);
        let m=Model::new().input("x",&[n as i64]).node("Relu","relu",&["x"],&["y"],&[]).output("y",&[n as i64]).encode();
        let sess=NativeDx12Session::open(&m,&ShapeOverrides::default(),&amd_request()).expect("open large Relu");
        let r=sess.run_f32(&NamedInputs::from([("x".into(),(x.as_slice(),vec![n]))])).expect("dispatch split Relu");
        assert_eq!(r["y"].1,vec![n]);assert_eq!(r["y"].0.len(),n);
        for (j,(&a,&b)) in r["y"].0.iter().zip(&x).enumerate(){assert_eq!(a,b.max(0.),"incorrect element {j}");}
    }

    #[test]
    #[ignore="requires a real AMD/Intel GPU with DX12 compute"]
    fn pow_and_numeric_cast_regressions_on_device() {
        let test=|op: &str,x:&[f32], rhs: Option<&[f32]>, to: i64| {
            let dims=[x.len() as i64];
            let mut m=Model::new().input("x",&dims);
            let mut inputs=NamedInputs::from([("x".into(),(x,vec![x.len()]))]);
            if let Some(b)=rhs {
                m=m.input("b",&dims).node(op,"n",&["x","b"],&["y"],&[]);
                inputs.insert("b".into(),(b,vec![b.len()]));
            } else {
                m=m.node("Cast","n",&["x"],&["narrow"],&[attr_int("to",to)])
                    .node("Cast","back",&["narrow"],&["y"],&[attr_int("to",1)]);
            }
            let bin=m.output("y",&dims).encode();
            NativeDx12Session::open(&bin,&ShapeOverrides::default(),&amd_request()).unwrap()
                .run_f32(&inputs).unwrap().remove("y").unwrap().0
        };
        let x=[-2.,-2.,-2.,-0.,-0.,-1.,1.,f32::NAN,-2.,0.,f32::NEG_INFINITY];
        let b=[3.,2.,0.5,3.,-3.,f32::INFINITY,f32::NAN,0.,-3.,0.,0.5];
        for (j,got) in test("Pow",&x,Some(&b),0).into_iter().enumerate() {
            let want=x[j].powf(b[j]);
            if want.is_nan(){assert!(got.is_nan(),"power {j}");}
            else if want.is_finite() && want!=0. {assert!((got-want).abs()<1e-5,"power {j}: {got} != {want}");}
            else {assert_eq!(got.to_bits(),want.to_bits(),"power {j}");}
        }
        for (to,x,want) in [(2,vec![0.9,127.8,200.9,255.],vec![0.,127.,200.,255.]),
            (3,vec![-128.,-2.9,0.75,126.9],vec![-128.,-2.,0.,126.]),
            (4,vec![0.1,255.9,4097.9,65535.],vec![0.,255.,4097.,65535.]),
            (5,vec![-32768.,-3.9,256.8,32767.],vec![-32768.,-3.,256.,32767.])] {
            assert_eq!(test("Cast",&x,None,to),want,"narrow type {to}");
        }
    }

    /// The published support table must not list operators that have no kernel.
    #[test]
    fn supported_operator_table_is_honest() {
        assert!(SUPPORTED_OPS.len() >= 45, "table lists {}", SUPPORTED_OPS.len());
        assert!(SUPPORTED_OPS.contains(&"Pad") && SUPPORTED_OPS.contains(&"Slice"));
        assert!(op_supported(&Op::Pad { mode: "constant".into() }));
        assert!(!op_supported(&Op::Pad { mode: "reflect".into() }));
        assert!(op_supported(&Op::Slice));
        for to in [DataType::Float16, DataType::Uint8, DataType::Int16] {
            assert!(op_supported(&Op::Cast { to }));
        }
        for to in [DataType::Double, DataType::Int64, DataType::Int32] {
            assert!(!op_supported(&Op::Cast { to }));
        }
        for missing in [
            "ConvTranspose",
            "ConvInteger",
            "MatMulInteger",
            "Resize",
            "Gather",
            "ScatterND",
            "Split",
            "Shape",
            "Size",
            "DynamicQuantizeLinear",
            "If",
            "ConstantOfShape",
            "ExpandAll",
        ] {
            assert!(
                !SUPPORTED_OPS.contains(&missing),
                "{missing} has no DX12 kernel and must not be listed as supported"
            );
        }
    }

    #[test]
    fn reduce_mean_plans_legacy_attr_and_constant_input_axes() {
        let plan_for = |model_bytes: &[u8]| -> Result<ReduceMeanPlan, String> {
            let planned = load_and_plan(model_bytes, &ShapeOverrides::default()).map_err(|e| e.to_string())?;
            let node = planned
                .graph
                .nodes
                .iter()
                .find(|n| matches!(n.op, Op::ReduceMean(_)))
                .expect("ReduceMean node");
            let Op::ReduceMean(attrs) = &node.op else { unreachable!() };
            let src = node.inputs[0].expect("x");
            let in_shape = planned.graph.tensors[src.0].static_shape().expect("x shape");
            plan_reduce_mean(&planned.graph, node, attrs, &in_shape, "test:ReduceMean")
        };

        // Legacy attribute: axes = [-1] and positive last axis [2] on [2, 3, 4].
        for axes in [&[-1i64][..], &[2i64][..]] {
            let m = Model::new()
                .input("x", &[2, 3, 4])
                .node(
                    "ReduceMean",
                    "mean",
                    &["x"],
                    &["y"],
                    &[attr_ints("axes", axes), attr_int("keepdims", 1)],
                )
                .output("y", &[2, 3, 1])
                .encode();
            assert_eq!(plan_for(&m), Ok(ReduceMeanPlan::LastAxis { rows: 6, len: 4 }));
        }

        // Opset 18+ constant `axes` input tensor (via Constant node and via initializer).
        let m_const = Model::new()
            .input("x", &[2, 3, 4])
            .constant("c_axes", "axes", &tensor_i64("axes", &[1], &[-1]))
            .node("ReduceMean", "mean", &["x", "axes"], &["y"], &[attr_int("keepdims", 1)])
            .output("y", &[2, 3, 1])
            .encode();
        assert_eq!(plan_for(&m_const), Ok(ReduceMeanPlan::LastAxis { rows: 6, len: 4 }));

        let m_init = Model::new()
            .input("x", &[2, 3, 4])
            .init_i64("axes", &[1], &[2])
            .node("ReduceMean", "mean", &["x", "axes"], &["y"], &[attr_int("keepdims", 1)])
            .output("y", &[2, 3, 1])
            .encode();
        assert_eq!(plan_for(&m_init), Ok(ReduceMeanPlan::LastAxis { rows: 6, len: 4 }));

        // Opset 18+ constant `axes = [0, 1, 2]` -> whole-tensor reduction.
        let m_all = Model::new()
            .input("x", &[2, 3, 4])
            .constant("c_axes", "axes", &tensor_i64("axes", &[3], &[0, 1, 2]))
            .node("ReduceMean", "mean", &["x", "axes"], &["y"], &[attr_int("keepdims", 1)])
            .output("y", &[1, 1, 1])
            .encode();
        assert_eq!(plan_for(&m_all), Ok(ReduceMeanPlan::WholeTensor));

        // No axes attr and no axes input -> whole-tensor reduction.
        let m_empty = Model::new()
            .input("x", &[2, 3, 4])
            .node("ReduceMean", "mean", &["x"], &["y"], &[attr_int("keepdims", 1)])
            .output("y", &[1, 1, 1])
            .encode();
        assert_eq!(plan_for(&m_empty), Ok(ReduceMeanPlan::WholeTensor));
    }

    #[test]
    fn reduce_mean_rejects_unresolved_or_unsupported_axes() {
        let plan_for = |model_bytes: &[u8]| -> Result<ReduceMeanPlan, String> {
            let planned = load_and_plan(model_bytes, &ShapeOverrides::default()).map_err(|e| e.to_string())?;
            let node = planned
                .graph
                .nodes
                .iter()
                .find(|n| matches!(n.op, Op::ReduceMean(_)))
                .expect("ReduceMean node");
            let Op::ReduceMean(attrs) = &node.op else { unreachable!() };
            let src = node.inputs[0].expect("x");
            let in_shape = planned.graph.tensors[src.0].static_shape().expect("x shape");
            plan_reduce_mean(&planned.graph, node, attrs, &in_shape, "test:ReduceMean")
        };

        // Runtime-computed `axes` input must be rejected, never fall back to whole-tensor mean.
        let m_runtime = Model::new()
            .input("x", &[2, 3, 4])
            .input("axes", &[1])
            .node("ReduceMean", "mean", &["x", "axes"], &["y"], &[attr_int("keepdims", 1)])
            .output("y", &[2, 1, 4])
            .encode();
        let err = plan_for(&m_runtime).expect_err("runtime axes must be refused");
        assert!(err.contains("static axes"), "unexpected error: {err}");

        // Non-last axis via legacy attr and via constant input must both be rejected.
        let m_attr_mid = Model::new()
            .input("x", &[2, 3, 4])
            .node(
                "ReduceMean",
                "mean",
                &["x"],
                &["y"],
                &[attr_ints("axes", &[1]), attr_int("keepdims", 1)],
            )
            .output("y", &[2, 1, 4])
            .encode();
        assert!(plan_for(&m_attr_mid).is_err());

        let m_const_mid = Model::new()
            .input("x", &[2, 3, 4])
            .constant("c_axes", "axes", &tensor_i64("axes", &[1], &[1]))
            .node("ReduceMean", "mean", &["x", "axes"], &["y"], &[attr_int("keepdims", 1)])
            .output("y", &[2, 1, 4])
            .encode();
        assert!(plan_for(&m_const_mid).is_err());
    }

    /// Latency and throughput of the submitted GPU work, measured after warm-up.
    ///
    /// The wall clock covers upload + dispatch + fence wait + readback, i.e. what a
    /// caller of `run_f32` experiences on this AMD device.
    #[test]
    #[ignore = "needs a Windows host with an AMD or Intel D3D12 adapter"]
    fn dx12_latency_and_gemm_throughput_report() {
        let request = amd_request();

        // (1) 256x256x256 MatMul: 2*256^3 = 33.6 MFLOP of real GPU work.
        let (m, k, n) = (256usize, 256usize, 256usize);
        let a = seeded(m * k, 0xabcd, 1.0);
        let b = seeded(k * n, 0xdcba, 1.0);
        let model = Model::new()
            .input("a", &[m as i64, k as i64])
            .init("b", &[k as i64, n as i64], &b)
            .node("MatMul", "mm0", &["a", "b"], &["y"], &[])
            .output("y", &[m as i64, n as i64])
            .encode();
        let session = NativeDx12Session::open(&model, &ShapeOverrides::default(), &request).expect("open");
        let mut inputs = NamedInputs::new();
        inputs.insert("a".to_string(), (a.as_slice(), vec![m, k]));
        let mut samples = Vec::new();
        for _ in 0..12 {
            let started = std::time::Instant::now();
            let outs = session.run_f32(&inputs).expect("run");
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(outs["y"].0.len(), m * n);
        }
        samples.remove(0); // first run includes first-use pipeline warm-up
        samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let flops = 2.0 * (m * k * n) as f64;
        let best = samples[0];
        eprintln!(
            "[gemm {m}x{k}x{n}] adapter={:?} dispatch={:?} runs={} p50={:.3}ms best={:.3}ms -> {:.2} GFLOP/s (best)",
            session.adapter().name,
            session.kernel_entries(),
            samples.len(),
            samples[samples.len() / 2],
            best,
            flops / (best / 1000.0) / 1e9
        );

        // (2) The elementwise chain: submit/fence/readback overhead per run.
        let (rows, cols, out_cols) = (4usize, 8usize, 3usize);
        let w = seeded(cols, 0x51ed, 1.0);
        let bias = seeded(cols, 0x9e37, 0.5);
        let mat = seeded(cols * out_cols, 0x1234_5678, 0.75);
        let model = Model::new()
            .input("x", &[1, rows as i64, cols as i64])
            .init("w", &[cols as i64], &w)
            .init("b", &[cols as i64], &bias)
            .init("m", &[cols as i64, out_cols as i64], &mat)
            .node("Mul", "mul0", &["x", "w"], &["t1"], &[])
            .node("Add", "add0", &["t1", "b"], &["t2"], &[])
            .node("Relu", "relu0", &["t2"], &["t3"], &[])
            .node("Softmax", "softmax0", &["t3"], &["t4"], &[attr_int("axis", 2)])
            .node("MatMul", "matmul0", &["t4", "m"], &["y"], &[])
            .output("y", &[1, rows as i64, out_cols as i64])
            .encode();
        let session = NativeDx12Session::open(&model, &ShapeOverrides::default(), &request).expect("open");
        let x = seeded(rows * cols, 0xbeef, 1.5);
        let mut inputs = NamedInputs::new();
        inputs.insert("x".to_string(), (x.as_slice(), vec![1, rows, cols]));
        let mut samples = Vec::new();
        for _ in 0..30 {
            let started = std::time::Instant::now();
            session.run_f32(&inputs).expect("run");
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        samples.remove(0);
        samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let p50 = samples[samples.len() / 2];
        let p95 = samples[(samples.len() as f64 * 0.95) as usize - 1];
        eprintln!(
            "[elementwise chain] 5 dispatches, 1 submit: p50={p50:.3}ms p95={p95:.3}ms over {} runs, dispatches/run={}",
            samples.len(),
            session.dispatch_count()
        );
        assert!(p50 < 50.0, "per-run latency {p50:.3}ms looks wrong for a 32-element chain");
    }

    /// Kernels that had no A/B yet: `Where`, last-axis `Expand`, `InstanceNormalization`,
    /// and the folded `Conv`+`BatchNormalization` path.
    #[test]
    #[ignore = "needs a Windows host with an AMD or Intel D3D12 adapter"]
    fn where_expand_instance_norm_and_folded_batchnorm_match_cpu_reference() {
        let request = amd_request();

        // --- Where(Greater(x, 0), x, neg) on [1,8]
        let x = seeded(8, 0x91, 2.0);
        let zero = vec![0.0f32];
        let neg = seeded(8, 0x92, 1.0);
        let model = Model::new()
            .input("x", &[1, 8])
            .init("zero", &[1], &zero)
            .init("neg", &[8], &neg)
            .node("Greater", "gt0", &["x", "zero"], &["cond"], &[])
            .node("Where", "where0", &["cond", "x", "neg"], &["y"], &[])
            .output("y", &[1, 8])
            .encode();
        let session = NativeDx12Session::open(&model, &ShapeOverrides::default(), &request).expect("open");
        let mut inputs = NamedInputs::new();
        inputs.insert("x".to_string(), (x.as_slice(), vec![1, 8]));
        let outs = session.run_f32(&inputs).expect("run");
        let reference: Vec<f32> = x
            .iter()
            .zip(neg.iter())
            .map(|(v, n)| if *v > 0.0 { *v } else { *n })
            .collect();
        let diff = max_abs_diff(&outs["y"].0, &reference);
        eprintln!("[where] kernels={:?} diff={diff:.3e}", session.kernel_entries());
        assert!(diff < 1e-6, "Where diff {diff:.3e}");

        // --- Expand/Tile: last axis 3 -> 6 on [1,2,3]
        let src = seeded(6, 0x93, 1.0);
        let model = Model::new()
            .input("x", &[1, 2, 3])
            .init("shape", &[3], &[1.0, 2.0, 6.0])
            .node("Expand", "expand0", &["x", "shape"], &["y"], &[])
            .output("y", &[1, 2, 6])
            .encode();
        let session = NativeDx12Session::open(&model, &ShapeOverrides::default(), &request).expect("open");
        let mut inputs = NamedInputs::new();
        inputs.insert("x".to_string(), (src.as_slice(), vec![1, 2, 3]));
        let outs = session.run_f32(&inputs).expect("run");
        let mut reference = Vec::new();
        for row in 0..2 {
            for _repeat in 0..2 {
                reference.extend_from_slice(&src[row * 3..row * 3 + 3]);
            }
        }
        let diff = max_abs_diff(&outs["y"].0, &reference);
        eprintln!("[expand] gpu={:?} cpu={reference:?} diff={diff:.3e}", outs["y"].0);
        assert!(diff < 1e-6, "Expand diff {diff:.3e}");

        // --- InstanceNormalization on [1,2,3,3]
        let x = seeded(18, 0x94, 1.5);
        let scale = vec![1.5f32, 0.5];
        let bias = vec![0.1f32, -0.2];
        let epsilon = 1e-5f32;
        let model = Model::new()
            .input("x", &[1, 2, 3, 3])
            .init("scale", &[2], &scale)
            .init("bias", &[2], &bias)
            .node(
                "InstanceNormalization",
                "in0",
                &["x", "scale", "bias"],
                &["y"],
                &[attr_int("epsilon", 0)],
            )
            .output("y", &[1, 2, 3, 3])
            .encode();
        let session = NativeDx12Session::open(&model, &ShapeOverrides::default(), &request).expect("open");
        let mut inputs = NamedInputs::new();
        inputs.insert("x".to_string(), (x.as_slice(), vec![1, 2, 3, 3]));
        let outs = session.run_f32(&inputs).expect("run");
        let mut reference = vec![0f32; 18];
        for plane in 0..2 {
            let data = &x[plane * 9..plane * 9 + 9];
            let mean = data.iter().sum::<f32>() / 9.0;
            let var = data.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 9.0;
            let inv = 1.0 / (var + epsilon).sqrt();
            for i in 0..9 {
                reference[plane * 9 + i] = (data[i] - mean) * inv * scale[plane] + bias[plane];
            }
        }
        let diff = max_abs_diff(&outs["y"].0, &reference);
        eprintln!("[instance_norm] diff={diff:.3e} kernels={:?}", session.kernel_entries());
        assert!(diff < 1e-5, "InstanceNormalization diff {diff:.3e}");

        // --- Conv + BatchNormalization (folded) + Relu on [1,2,4,4]
        let (cin, cout, hw, pad) = (2usize, 3usize, 4usize, 1i64);
        let weights = seeded(cout * cin * 3 * 3, 0x95, 0.4);
        let conv_bias = seeded(cout, 0x96, 0.1);
        let bn_scale = seeded(cout, 0x97, 0.3).iter().map(|v| 1.0 + v).collect::<Vec<f32>>();
        let bn_bias = seeded(cout, 0x98, 0.1);
        let bn_mean = seeded(cout, 0x99, 0.2);
        let bn_var = seeded(cout, 0x9a, 0.2).iter().map(|v| 0.5 + v.abs()).collect::<Vec<f32>>();
        let bn_eps = 1e-5f32;
        let model = Model::new()
            .input("x", &[1, cin as i64, hw as i64, hw as i64])
            .init("w", &[cout as i64, cin as i64, 3, 3], &weights)
            .init("cb", &[cout as i64], &conv_bias)
            .init("bs", &[cout as i64], &bn_scale)
            .init("bb", &[cout as i64], &bn_bias)
            .init("bm", &[cout as i64], &bn_mean)
            .init("bv", &[cout as i64], &bn_var)
            .node(
                "Conv",
                "conv0",
                &["x", "w", "cb"],
                &["c1"],
                &[
                    attr_ints("kernel_shape", &[3, 3]),
                    attr_ints("pads", &[pad, pad, pad, pad]),
                    attr_ints("strides", &[1, 1]),
                ],
            )
            .node("BatchNormalization", "bn0", &["c1", "bs", "bb", "bm", "bv"], &["b1"], &[])
            .node("Relu", "relu0", &["b1"], &["y"], &[])
            .output("y", &[1, cout as i64, hw as i64, hw as i64])
            .value_info("c1", &[1, cout as i64, hw as i64, hw as i64])
            .value_info("b1", &[1, cout as i64, hw as i64, hw as i64])
            .encode();
        let session = NativeDx12Session::open(&model, &ShapeOverrides::default(), &request).expect("open");
        eprintln!("[conv+bn] kernels={:?} dispatches={}", session.kernel_entries(), session.dispatch_count());
        let x = seeded(cin * hw * hw, 0x9b, 1.0);
        let mut inputs = NamedInputs::new();
        inputs.insert("x".to_string(), (x.as_slice(), vec![1, cin, hw, hw]));
        let outs = session.run_f32(&inputs).expect("run");

        let mut reference = vec![0f32; cout * hw * hw];
        for oc in 0..cout {
            let s = bn_scale[oc] / (bn_var[oc] + bn_eps).sqrt();
            let shift = bn_bias[oc] - bn_mean[oc] * s;
            for y in 0..hw {
                for xx in 0..hw {
                    let mut acc = conv_bias[oc];
                    for ic in 0..cin {
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let iy = y as i64 + ky as i64 - pad;
                                let ix = xx as i64 + kx as i64 - pad;
                                if iy < 0 || ix < 0 || iy >= hw as i64 || ix >= hw as i64 {
                                    continue;
                                }
                                acc += x[ic * hw * hw + iy as usize * hw + ix as usize]
                                    * weights[((oc * cin + ic) * 3 + ky) * 3 + kx];
                            }
                        }
                    }
                    reference[oc * hw * hw + y * hw + xx] = (acc * s + shift).max(0.0);
                }
            }
        }
        let diff = max_abs_diff(&outs["y"].0, &reference);
        eprintln!("[conv+bn] diff={diff:.3e}");
        assert!(diff < 1e-4, "Conv+BatchNorm diff {diff:.3e}");

        // --- Standalone BatchNormalization with static parameters: folding turns it
        // into `BatchNormFused`, which the per-channel affine kernel executes.
        let (channels, hw2) = (3usize, 2usize);
        let x = seeded(channels * hw2 * hw2, 0x9c, 1.2);
        let scale = vec![1.3f32, 0.7, 1.0];
        let bias = vec![0.2f32, -0.1, 0.05];
        let mean = vec![0.3f32, -0.4, 0.0];
        let var = vec![1.2f32, 0.8, 1.5];
        let model = Model::new()
            .input("x", &[1, channels as i64, hw2 as i64, hw2 as i64])
            .init("bs", &[channels as i64], &scale)
            .init("bb", &[channels as i64], &bias)
            .init("bm", &[channels as i64], &mean)
            .init("bv", &[channels as i64], &var)
            .node("BatchNormalization", "bn0", &["x", "bs", "bb", "bm", "bv"], &["y"], &[])
            .output("y", &[1, channels as i64, hw2 as i64, hw2 as i64])
            .encode();
        let session = NativeDx12Session::open(&model, &ShapeOverrides::default(), &request).expect("open");
        eprintln!(
            "[batchnorm] kernels={:?} dispatches={}",
            session.kernel_entries(),
            session.dispatch_count()
        );
        let mut inputs = NamedInputs::new();
        inputs.insert("x".to_string(), (x.as_slice(), vec![1, channels, hw2, hw2]));
        let outs = session.run_f32(&inputs).expect("run");
        let mut reference = vec![0f32; channels * hw2 * hw2];
        for c in 0..channels {
            let eff = scale[c] / (var[c] + 1e-5).sqrt();
            let shift = bias[c] - mean[c] * eff;
            for i in 0..hw2 * hw2 {
                reference[c * hw2 * hw2 + i] = x[c * hw2 * hw2 + i] * eff + shift;
            }
        }
        let diff = max_abs_diff(&outs["y"].0, &reference);
        eprintln!("[batchnorm] diff={diff:.3e}");
        assert!(diff < 1e-6, "BatchNormFused diff {diff:.3e}");
    }

}
