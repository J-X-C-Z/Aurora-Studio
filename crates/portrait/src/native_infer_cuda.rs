//! CUDA executor for [`crate::native_infer_graph::Graph`].
//!
//! The FP32 CPU executor in [`crate::native_infer_graph::execute_graph`] stays
//! the oracle: this module lowers the same `OpKind` arms onto the kernels in
//! `native/cuda/native_infer_ops.cu` and the parity tests compare both paths.
//!
//! What is real here:
//!
//! * every one of the 14 `OpKind` variants has a kernel and a launch arm;
//! * device storage is the graph's own [`MemoryPlan`] slot layout, so tensors
//!   whose live ranges do not overlap share one allocation, exactly as the
//!   planner promises — `slot_of` / `capacity` below are derived from
//!   `plan_memory`, and a tensor is copied into its slot on its first read,
//!   which keeps a reused constant slot from clobbering a live value;
//! * all H2D/D2H traffic goes through one reusable staging buffer with an exact
//!   device-to-device copy, because the shared driver layer's upload/download
//!   helpers are whole-buffer operations;
//! * [`ExecutionMode::Strict`] fails when there is no device, and
//!   [`ExecutionMode::AutoFallback`] runs the CPU reference and reports
//!   `backend: Cpu` with the reason — a GPU success is never fabricated.
//!
//! What is not claimed: group statistics use a parallel two-pass reduction
//! instead of the CPU's Welford recurrence, so `GroupNorm` is compared with a
//! tolerance (the end-to-end assertion is `max_abs < 1e-4`), not bit-exactly.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::fmt;
use std::time::Instant;

use crate::autotune::{BackendTiming, ExecutionMode};
use crate::cuda::{CudaError, Dispatch, Nvidia};
use crate::device::Backend;
use crate::native_infer_graph::{
    execute_graph, plan_memory, DType, Graph, GraphError, GraphResult, MemoryPlan, OpKind,
    TensorBuffer, TensorShape,
};

/// JIT image holding every operator kernel.
pub const NATIVE_INFER_PTX: &str = include_str!("../native/cuda/native_infer_ops.ptx");

/// Marker the checked-in placeholder image carries. Never a success path.
pub const PTX_PLACEHOLDER_MARKER: &str = "AURORA-PTX-PLACEHOLDER";

/// Kernel entry points, one per lowered operation family.
pub const K_COPY: &str = "ni_copy_bytes_kernel";
pub const K_CONV2D: &str = "ni_conv2d_kernel";
pub const K_DEPTHWISE_CONV2D: &str = "ni_depthwise_conv2d_kernel";
pub const K_BATCHNORM: &str = "ni_batchnorm_kernel";
pub const K_GROUPNORM: &str = "ni_groupnorm_kernel";
pub const K_RELU: &str = "ni_relu_kernel";
pub const K_SILU: &str = "ni_silu_kernel";
pub const K_SIGMOID: &str = "ni_sigmoid_kernel";
pub const K_ADD: &str = "ni_add_kernel";
pub const K_MUL: &str = "ni_mul_kernel";
pub const K_RESIZE_BILINEAR: &str = "ni_resize_bilinear_kernel";
pub const K_GRID_SAMPLE: &str = "ni_grid_sample_kernel";
pub const K_MATMUL: &str = "ni_matmul_kernel";
pub const K_QUANTIZE: &str = "ni_quantize_kernel";
pub const K_DEQUANTIZE: &str = "ni_dequantize_kernel";

/// All lowered entry points, for reports and capability checks.
pub const KERNELS: [&str; 15] = [
    K_COPY,
    K_CONV2D,
    K_DEPTHWISE_CONV2D,
    K_BATCHNORM,
    K_GROUPNORM,
    K_RELU,
    K_SILU,
    K_SIGMOID,
    K_ADD,
    K_MUL,
    K_RESIZE_BILINEAR,
    K_GRID_SAMPLE,
    K_MATMUL,
    K_QUANTIZE,
    K_DEQUANTIZE,
];

/// 1D launch shape shared by every operator kernel.
const BLOCK_1D: (u32, u32) = (256, 1);

// ---------------------------------------------------------------------------
// Kernel argument blocks, field-for-field with `native/cuda/native_infer_ops.cu`
// ---------------------------------------------------------------------------

/// Mirrors `NiConvUniforms`: 68 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConvUniforms {
    pub n: i32,
    pub c_in: i32,
    pub in_h: i32,
    pub in_w: i32,
    pub c_out: i32,
    pub kh: i32,
    pub kw: i32,
    pub stride_h: i32,
    pub stride_w: i32,
    pub pad_t: i32,
    pub pad_l: i32,
    pub dil_h: i32,
    pub dil_w: i32,
    pub out_h: i32,
    pub out_w: i32,
    pub has_bias: i32,
    pub depthwise: i32,
}

/// Mirrors `NiNormUniforms`: 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NormUniforms {
    pub count: i32,
    pub channels: i32,
    pub spatial: i32,
    pub eps: f32,
}

/// Mirrors `NiGroupNormUniforms`: 20 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GroupNormUniforms {
    pub n: i32,
    pub channels: i32,
    pub spatial: i32,
    pub groups: i32,
    pub eps: f32,
}

/// Mirrors `NiBroadcastUniforms`: 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BroadcastUniforms {
    pub out_n: i32,
    pub out_c: i32,
    pub out_h: i32,
    pub out_w: i32,
    pub a_n: i32,
    pub a_c: i32,
    pub a_h: i32,
    pub a_w: i32,
    pub b_n: i32,
    pub b_c: i32,
    pub b_h: i32,
    pub b_w: i32,
}

/// Mirrors `NiResizeUniforms`: 44 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ResizeUniforms {
    pub in_n: i32,
    pub in_c: i32,
    pub in_h: i32,
    pub in_w: i32,
    pub out_n: i32,
    pub out_c: i32,
    pub out_h: i32,
    pub out_w: i32,
    pub scale_h: f32,
    pub scale_w: f32,
    pub align_corners: i32,
}

/// Mirrors `NiGridSampleUniforms`: 24 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GridSampleUniforms {
    pub n: i32,
    pub c: i32,
    pub in_h: i32,
    pub in_w: i32,
    pub grid_h: i32,
    pub grid_w: i32,
}

/// Mirrors `NiMatMulUniforms`: 36 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MatMulUniforms {
    pub n: i32,
    pub rows: i32,
    pub k: i32,
    pub cols: i32,
    pub a_n_stride: i32,
    pub a_row_stride: i32,
    pub b_n_stride: i32,
    pub b_j_stride: i32,
    pub b_k_stride: i32,
}

const _: () = assert!(std::mem::size_of::<ConvUniforms>() == 68);
const _: () = assert!(std::mem::size_of::<NormUniforms>() == 16);
const _: () = assert!(std::mem::size_of::<GroupNormUniforms>() == 20);
const _: () = assert!(std::mem::size_of::<BroadcastUniforms>() == 48);
const _: () = assert!(std::mem::size_of::<ResizeUniforms>() == 44);
const _: () = assert!(std::mem::size_of::<GridSampleUniforms>() == 24);
const _: () = assert!(std::mem::size_of::<MatMulUniforms>() == 36);

// ---------------------------------------------------------------------------
// Errors and results
// ---------------------------------------------------------------------------

/// Why a graph did not execute on the GPU.
#[derive(Clone, Debug, PartialEq)]
pub enum CudaGraphError {
    /// The graph itself is invalid (before any driver call).
    Graph(GraphError),
    /// A driver, module or launch failure.
    Cuda(CudaError),
}

impl From<GraphError> for CudaGraphError {
    fn from(error: GraphError) -> Self {
        Self::Graph(error)
    }
}

impl From<CudaError> for CudaGraphError {
    fn from(error: CudaError) -> Self {
        Self::Cuda(error)
    }
}

impl fmt::Display for CudaGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(error) => write!(f, "graph error: {error}"),
            Self::Cuda(error) => write!(f, "cuda error: {error:?}"),
        }
    }
}

impl std::error::Error for CudaGraphError {}

/// One graph execution (real CUDA, or the labelled CPU fallback).
#[derive(Clone, Debug)]
pub struct CudaGraphExecution {
    /// `Backend::Cuda` only when the kernels really ran.
    pub backend: Backend,
    /// Declared graph outputs, in declaration order.
    pub outputs: Vec<TensorBuffer>,
    /// The slot plan the executor used.
    pub plan: MemoryPlan,
    /// Every launch that happened, in schedule order (empty for a fallback).
    pub dispatches: Vec<Dispatch>,
    /// Host-precise timing for the whole execution.
    pub timing: BackendTiming,
    /// `Some(reason)` when [`ExecutionMode::AutoFallback`] used the CPU.
    pub fallback_reason: Option<&'static str>,
}

/// True for the checked-in placeholder, i.e. before `nvcc` produced the module.
pub fn ptx_is_placeholder(ptx: &str) -> bool {
    ptx.contains(PTX_PLACEHOLDER_MARKER)
}

/// Opens the operator module, refusing a placeholder image.
pub fn open_native_infer() -> Result<Nvidia, CudaError> {
    if ptx_is_placeholder(NATIVE_INFER_PTX) {
        return Err(CudaError::Unavailable(
            "native_infer_ops.ptx is a placeholder (run nvcc -ptx -fmad=false)",
        ));
    }
    Nvidia::open_ptx(NATIVE_INFER_PTX)
}

/// Whether a real graph execution can run on this host right now.
pub fn device_available() -> bool {
    open_native_infer().is_ok()
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

/// `plan_memory` for a graph whose runtime inputs are not constants.
///
/// The planner infers shapes from the graph alone, so runtime inputs are
/// registered on a clone as correctly shaped zeros: the lifetimes, slots and
/// byte peaks are identical to the real execution, only the pre-execution
/// values differ (and those are uploaded by the executor on first read).
pub fn plan_for_execution(
    graph: &Graph,
    runtime_inputs: &HashMap<usize, TensorBuffer>,
) -> GraphResult<MemoryPlan> {
    let mut planner = graph.clone();
    for (&id, tensor) in runtime_inputs {
        planner
            .constants
            .insert(id, TensorBuffer::zeros(tensor.dtype, tensor.shape));
    }
    plan_memory(&planner)
}

fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn grid_1d(count: usize) -> (u32, u32) {
    let blocks = (count as u64).div_ceil(BLOCK_1D.0 as u64).max(1);
    (blocks.min(u32::MAX as u64) as u32, 1)
}

fn launch(
    gpu: &Nvidia,
    kernel: &'static str,
    count: usize,
    params: &mut [*mut c_void],
) -> Result<Dispatch, CudaError> {
    gpu.launch_raw(kernel, grid_1d(count), BLOCK_1D, params)
}

fn shape_of(shapes: &HashMap<usize, TensorShape>, id: usize) -> GraphResult<TensorShape> {
    shapes
        .get(&id)
        .copied()
        .ok_or_else(|| GraphError::ShapeMismatch(format!("unresolved shape for tensor {id}")))
}

fn handle_for(
    slot_of: &HashMap<usize, usize>,
    handles: &[u64],
    tensor: usize,
) -> GraphResult<u64> {
    let slot = *slot_of.get(&tensor).ok_or_else(|| {
        GraphError::TensorMismatch(format!("tensor {tensor} has no planned memory slot"))
    })?;
    handles.get(slot).copied().ok_or_else(|| {
        GraphError::TensorMismatch(format!("slot {slot} for tensor {tensor} is unallocated"))
    })
}

/// `cuMemcpyDtoD` twin built from the byte-copy kernel (the shared driver
/// layer only exposes whole-buffer host copies).
fn device_copy(gpu: &Nvidia, dst: u64, src: u64, bytes: usize) -> Result<Dispatch, CudaError> {
    let mut dst_arg = dst;
    let mut src_arg = src;
    let mut count_arg = bytes as i64;
    let mut params: [*mut c_void; 3] = [
        &mut dst_arg as *mut u64 as *mut c_void,
        &mut src_arg as *mut u64 as *mut c_void,
        &mut count_arg as *mut i64 as *mut c_void,
    ];
    launch(gpu, K_COPY, bytes, &mut params)
}

fn dim(value: usize, what: &'static str) -> GraphResult<i32> {
    i32::try_from(value)
        .map_err(|_| GraphError::Unsupported(format!("{what} exceeds the 2^31 element limit")))
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// Executes `graph` on the GPU.
///
/// [`ExecutionMode::Strict`] returns [`CudaGraphError::Cuda`] when no device or
/// no compiled image is available. [`ExecutionMode::AutoFallback`] then runs
/// the FP32 CPU reference and returns `backend: Backend::Cpu` with
/// `fallback_reason` set. Driver errors *after* a device was opened are never
/// downgraded to a fallback: they are reported as errors.
pub fn execute_graph_cuda(
    graph: &Graph,
    runtime_inputs: &HashMap<usize, TensorBuffer>,
    mode: ExecutionMode,
) -> Result<CudaGraphExecution, CudaGraphError> {
    let plan = plan_for_execution(graph, runtime_inputs)?;
    let started = Instant::now();
    match run_on_device(graph, runtime_inputs, &plan) {
        Ok(mut execution) => {
            execution.timing.wall_ns = elapsed_ns(started);
            Ok(execution)
        }
        Err(CudaGraphError::Cuda(CudaError::Unavailable(reason)))
            if mode == ExecutionMode::AutoFallback =>
        {
            let reference = execute_graph(graph, runtime_inputs)?;
            Ok(CudaGraphExecution {
                backend: Backend::Cpu,
                outputs: reference.outputs,
                plan,
                dispatches: Vec::new(),
                timing: BackendTiming {
                    wall_ns: elapsed_ns(started),
                    ..BackendTiming::default()
                },
                fallback_reason: Some(reason),
            })
        }
        Err(error) => Err(error),
    }
}

fn run_on_device(
    graph: &Graph,
    runtime_inputs: &HashMap<usize, TensorBuffer>,
    plan: &MemoryPlan,
) -> Result<CudaGraphExecution, CudaGraphError> {
    let gpu = open_native_infer()?;
    let order = graph.topo_order()?;
    let extra_shapes: HashMap<usize, TensorShape> = runtime_inputs
        .iter()
        .map(|(&id, tensor)| (id, tensor.shape))
        .collect();
    let shapes = graph.infer_shapes_with(&extra_shapes)?;

    // Real dtype per tensor: `plan_memory` models node outputs as Fp32, while a
    // QuantizeLinear output is one byte per element. Sizing the slots from the
    // real dtype keeps the plan valid and never under-allocates.
    let mut dtypes: HashMap<usize, DType> = graph
        .constants
        .iter()
        .map(|(&id, tensor)| (id, tensor.dtype))
        .collect();
    for &index in &order {
        let node = &graph.nodes[index];
        let dtype = match node.op {
            OpKind::QuantizeLinear { .. } => DType::Int8,
            _ => DType::Fp32,
        };
        dtypes.insert(node.output, dtype);
    }
    for (&id, tensor) in runtime_inputs {
        dtypes.insert(id, tensor.dtype);
        if shapes.get(&id).copied() != Some(tensor.shape) {
            return Err(GraphError::TensorMismatch(format!(
                "runtime input {id} is {:?} but the graph infers {:?}",
                tensor.shape,
                shapes.get(&id)
            ))
            .into());
        }
    }

    // Every kernel indexes with 32-bit ints.
    for shape in shapes.values() {
        dim(shape.numel(), "tensor")?;
    }

    let mut slot_of: HashMap<usize, usize> = HashMap::new();
    let mut capacity: Vec<usize> = vec![0; plan.slot_count];
    for lifetime in &plan.lifetimes {
        if lifetime.slot >= capacity.len() {
            return Err(GraphError::ShapeMismatch(
                "memory plan slot index out of range".to_string(),
            )
            .into());
        }
        let bytes = lifetime.shape.byte_len(
            dtypes
                .get(&lifetime.tensor)
                .copied()
                .unwrap_or(lifetime.dtype),
        );
        slot_of.insert(lifetime.tensor, lifetime.slot);
        capacity[lifetime.slot] = capacity[lifetime.slot].max(bytes);
    }

    let mut max_bytes = capacity.iter().copied().max().unwrap_or(0);
    for tensor in graph.constants.values() {
        max_bytes = max_bytes.max(tensor.bytes.len());
    }
    for tensor in runtime_inputs.values() {
        max_bytes = max_bytes.max(tensor.bytes.len());
    }
    let max_bytes = max_bytes.max(4).min(i32::MAX as usize);

    // One device allocation per planned slot: tensors whose live ranges do not
    // overlap share it, which is exactly what `plan_memory` decided.
    let mut slot_buffers = Vec::with_capacity(capacity.len());
    for bytes in &capacity {
        slot_buffers.push(gpu.alloc_buffer((*bytes).max(4))?);
    }
    let handles: Vec<u64> = slot_buffers
        .iter()
        .map(|buffer| buffer.handle())
        .collect();
    let staging = gpu.alloc_buffer(max_bytes)?;
    let staging_handle = staging.handle();

    let mut dispatches: Vec<Dispatch> = Vec::with_capacity(order.len() + 1);
    let mut uploaded: HashSet<usize> = HashSet::new();
    let mut h2d_bytes: u64 = 0;

    // Uploads happen on a tensor's first read, never all up front: a slot may
    // be shared by two constants whose live ranges do not overlap, and an
    // eager upload of the later one would overwrite the earlier one's value.
    for &index in &order {
        let node = &graph.nodes[index];
        let mut needed: Vec<usize> = node.inputs.clone();
        match &node.op {
            OpKind::Conv2d { weight, bias, .. } | OpKind::DepthwiseConv2d { weight, bias, .. } => {
                needed.push(*weight);
                if let Some(b) = bias {
                    needed.push(*b);
                }
            }
            OpKind::BatchNorm {
                scale,
                offset,
                mean,
                var,
                ..
            } => {
                needed.extend([*scale, *offset, *mean, *var]);
            }
            OpKind::GroupNorm { scale, offset, .. } => {
                needed.extend([*scale, *offset]);
            }
            _ => {}
        }
        for input in needed {
            if uploaded.contains(&input) {
                continue;
            }
            let tensor = match runtime_inputs.get(&input) {
                Some(tensor) => Some(tensor),
                None => graph.constants.get(&input),
            };
            let Some(tensor) = tensor else {
                continue; // produced by an earlier node
            };
            let slot = *slot_of.get(&input).ok_or_else(|| {
                GraphError::TensorMismatch(format!("tensor {input} has no planned slot"))
            })?;
            if tensor.bytes.len() > max_bytes || tensor.bytes.len() > capacity[slot] {
                return Err(GraphError::TensorMismatch(format!(
                    "tensor {input} has {} bytes but its slot holds {}",
                    tensor.bytes.len(),
                    capacity[slot]
                ))
                .into());
            }
            gpu.upload_buffer(&staging, &tensor.bytes)?;
            let copied = device_copy(&gpu, handles[slot], staging_handle, tensor.bytes.len())?;
            dispatches.push(copied);
            h2d_bytes += tensor.bytes.len() as u64;
            uploaded.insert(input);
        }

        let out = handle_for(&slot_of, &handles, node.output)?;

        match &node.op {
            OpKind::Conv2d { .. } | OpKind::DepthwiseConv2d { .. } => {
                let depthwise = matches!(node.op, OpKind::DepthwiseConv2d { .. });
                let (stride, pad, dilation, weight, bias) = match &node.op {
                    OpKind::Conv2d {
                        stride,
                        pad,
                        dilation,
                        weight,
                        bias,
                    } => (*stride, *pad, *dilation, *weight, *bias),
                    OpKind::DepthwiseConv2d {
                        stride,
                        pad,
                        weight,
                        bias,
                    } => (*stride, *pad, (1usize, 1usize), *weight, *bias),
                    _ => unreachable!("matched a convolution above"),
                };
                let xs = shape_of(&shapes, node.inputs[0])?;
                let ws = shape_of(&shapes, weight)?;
                let os = shape_of(&shapes, node.output)?;
                let uniforms = ConvUniforms {
                    n: dim(xs.n, "conv batch")?,
                    c_in: dim(xs.c, "conv input channels")?,
                    in_h: dim(xs.h, "conv input height")?,
                    in_w: dim(xs.w, "conv input width")?,
                    c_out: dim(os.c, "conv output channels")?,
                    kh: dim(ws.h, "conv kernel height")?,
                    kw: dim(ws.w, "conv kernel width")?,
                    stride_h: dim(stride.0, "conv stride")?,
                    stride_w: dim(stride.1, "conv stride")?,
                    pad_t: dim(pad.0, "conv pad")?,
                    pad_l: dim(pad.2, "conv pad")?,
                    dil_h: dim(dilation.0, "conv dilation")?,
                    dil_w: dim(dilation.1, "conv dilation")?,
                    out_h: dim(os.h, "conv output height")?,
                    out_w: dim(os.w, "conv output width")?,
                    has_bias: i32::from(bias.is_some()),
                    depthwise: i32::from(depthwise),
                };
                let mut x_arg = handle_for(&slot_of, &handles, node.inputs[0])?;
                let mut w_arg = handle_for(&slot_of, &handles, weight)?;
                let mut b_arg = match bias {
                    Some(id) => handle_for(&slot_of, &handles, id)?,
                    None => 0,
                };
                let mut o_arg = out;
                let mut u_arg = uniforms;
                let mut params: [*mut c_void; 5] = [
                    &mut x_arg as *mut u64 as *mut c_void,
                    &mut w_arg as *mut u64 as *mut c_void,
                    &mut b_arg as *mut u64 as *mut c_void,
                    &mut o_arg as *mut u64 as *mut c_void,
                    &mut u_arg as *mut ConvUniforms as *mut c_void,
                ];
                let kernel = if depthwise {
                    K_DEPTHWISE_CONV2D
                } else {
                    K_CONV2D
                };
                dispatches.push(launch(&gpu, kernel, os.numel(), &mut params)?);
            }
            OpKind::BatchNorm {
                scale,
                offset,
                mean,
                var,
                epsilon,
            } => {
                let xs = shape_of(&shapes, node.inputs[0])?;
                let uniforms = NormUniforms {
                    count: dim(xs.numel(), "batchnorm elements")?,
                    channels: dim(xs.c, "batchnorm channels")?,
                    spatial: dim(xs.h * xs.w, "batchnorm spatial")?,
                    eps: *epsilon,
                };
                let mut x_arg = handle_for(&slot_of, &handles, node.inputs[0])?;
                let mut scale_arg = handle_for(&slot_of, &handles, *scale)?;
                let mut offset_arg = handle_for(&slot_of, &handles, *offset)?;
                let mut mean_arg = handle_for(&slot_of, &handles, *mean)?;
                let mut var_arg = handle_for(&slot_of, &handles, *var)?;
                let mut o_arg = out;
                let mut u_arg = uniforms;
                let mut params: [*mut c_void; 7] = [
                    &mut x_arg as *mut u64 as *mut c_void,
                    &mut scale_arg as *mut u64 as *mut c_void,
                    &mut offset_arg as *mut u64 as *mut c_void,
                    &mut mean_arg as *mut u64 as *mut c_void,
                    &mut var_arg as *mut u64 as *mut c_void,
                    &mut o_arg as *mut u64 as *mut c_void,
                    &mut u_arg as *mut NormUniforms as *mut c_void,
                ];
                dispatches.push(launch(&gpu, K_BATCHNORM, xs.numel(), &mut params)?);
            }
            OpKind::GroupNorm {
                groups,
                scale,
                offset,
                epsilon,
            } => {
                let xs = shape_of(&shapes, node.inputs[0])?;
                let uniforms = GroupNormUniforms {
                    n: dim(xs.n, "groupnorm batch")?,
                    channels: dim(xs.c, "groupnorm channels")?,
                    spatial: dim(xs.h * xs.w, "groupnorm spatial")?,
                    groups: dim(*groups, "groupnorm groups")?,
                    eps: *epsilon,
                };
                let mut x_arg = handle_for(&slot_of, &handles, node.inputs[0])?;
                let mut scale_arg = handle_for(&slot_of, &handles, *scale)?;
                let mut offset_arg = handle_for(&slot_of, &handles, *offset)?;
                let mut o_arg = out;
                let mut u_arg = uniforms;
                let mut params: [*mut c_void; 5] = [
                    &mut x_arg as *mut u64 as *mut c_void,
                    &mut scale_arg as *mut u64 as *mut c_void,
                    &mut offset_arg as *mut u64 as *mut c_void,
                    &mut o_arg as *mut u64 as *mut c_void,
                    &mut u_arg as *mut GroupNormUniforms as *mut c_void,
                ];
                // One block per (sample, group); the kernel reduces the group.
                let blocks = (xs.n * *groups).max(1).min(u32::MAX as usize) as u32;
                dispatches.push(gpu.launch_raw(
                    K_GROUPNORM,
                    (blocks, 1),
                    BLOCK_1D,
                    &mut params,
                )?);
            }
            OpKind::Relu | OpKind::SiLU | OpKind::Sigmoid => {
                let xs = shape_of(&shapes, node.inputs[0])?;
                let kernel = match node.op {
                    OpKind::Relu => K_RELU,
                    OpKind::SiLU => K_SILU,
                    _ => K_SIGMOID,
                };
                let mut x_arg = handle_for(&slot_of, &handles, node.inputs[0])?;
                let mut o_arg = out;
                let mut count = dim(xs.numel(), "elementwise elements")?;
                let mut params: [*mut c_void; 3] = [
                    &mut x_arg as *mut u64 as *mut c_void,
                    &mut o_arg as *mut u64 as *mut c_void,
                    &mut count as *mut i32 as *mut c_void,
                ];
                dispatches.push(launch(&gpu, kernel, xs.numel(), &mut params)?);
            }
            OpKind::Add | OpKind::Mul => {
                let as_ = shape_of(&shapes, node.inputs[0])?;
                let bs = shape_of(&shapes, node.inputs[1])?;
                let os = shape_of(&shapes, node.output)?;
                let uniforms = BroadcastUniforms {
                    out_n: dim(os.n, "broadcast")?,
                    out_c: dim(os.c, "broadcast")?,
                    out_h: dim(os.h, "broadcast")?,
                    out_w: dim(os.w, "broadcast")?,
                    a_n: dim(as_.n, "broadcast a")?,
                    a_c: dim(as_.c, "broadcast a")?,
                    a_h: dim(as_.h, "broadcast a")?,
                    a_w: dim(as_.w, "broadcast a")?,
                    b_n: dim(bs.n, "broadcast b")?,
                    b_c: dim(bs.c, "broadcast b")?,
                    b_h: dim(bs.h, "broadcast b")?,
                    b_w: dim(bs.w, "broadcast b")?,
                };
                let mut a_arg = handle_for(&slot_of, &handles, node.inputs[0])?;
                let mut b_arg = handle_for(&slot_of, &handles, node.inputs[1])?;
                let mut o_arg = out;
                let mut u_arg = uniforms;
                let mut params: [*mut c_void; 4] = [
                    &mut a_arg as *mut u64 as *mut c_void,
                    &mut b_arg as *mut u64 as *mut c_void,
                    &mut o_arg as *mut u64 as *mut c_void,
                    &mut u_arg as *mut BroadcastUniforms as *mut c_void,
                ];
                let kernel = if matches!(node.op, OpKind::Add) {
                    K_ADD
                } else {
                    K_MUL
                };
                dispatches.push(launch(&gpu, kernel, os.numel(), &mut params)?);
            }
            OpKind::ResizeBilinear {
                scale_h,
                scale_w,
                align_corners,
            } => {
                let xs = shape_of(&shapes, node.inputs[0])?;
                let os = shape_of(&shapes, node.output)?;
                let uniforms = ResizeUniforms {
                    in_n: dim(xs.n, "resize")?,
                    in_c: dim(xs.c, "resize")?,
                    in_h: dim(xs.h, "resize")?,
                    in_w: dim(xs.w, "resize")?,
                    out_n: dim(os.n, "resize")?,
                    out_c: dim(os.c, "resize")?,
                    out_h: dim(os.h, "resize")?,
                    out_w: dim(os.w, "resize")?,
                    scale_h: *scale_h,
                    scale_w: *scale_w,
                    align_corners: i32::from(*align_corners),
                };
                let mut x_arg = handle_for(&slot_of, &handles, node.inputs[0])?;
                let mut o_arg = out;
                let mut u_arg = uniforms;
                let mut params: [*mut c_void; 3] = [
                    &mut x_arg as *mut u64 as *mut c_void,
                    &mut o_arg as *mut u64 as *mut c_void,
                    &mut u_arg as *mut ResizeUniforms as *mut c_void,
                ];
                dispatches.push(launch(&gpu, K_RESIZE_BILINEAR, os.numel(), &mut params)?);
            }
            OpKind::GridSample => {
                let xs = shape_of(&shapes, node.inputs[0])?;
                let os = shape_of(&shapes, node.output)?;
                let uniforms = GridSampleUniforms {
                    n: dim(os.n, "gridsample")?,
                    c: dim(os.c, "gridsample")?,
                    in_h: dim(xs.h, "gridsample")?,
                    in_w: dim(xs.w, "gridsample")?,
                    grid_h: dim(os.h, "gridsample")?,
                    grid_w: dim(os.w, "gridsample")?,
                };
                let mut x_arg = handle_for(&slot_of, &handles, node.inputs[0])?;
                let mut grid_arg = handle_for(&slot_of, &handles, node.inputs[1])?;
                let mut o_arg = out;
                let mut u_arg = uniforms;
                let mut params: [*mut c_void; 4] = [
                    &mut x_arg as *mut u64 as *mut c_void,
                    &mut grid_arg as *mut u64 as *mut c_void,
                    &mut o_arg as *mut u64 as *mut c_void,
                    &mut u_arg as *mut GridSampleUniforms as *mut c_void,
                ];
                dispatches.push(launch(&gpu, K_GRID_SAMPLE, os.numel(), &mut params)?);
            }
            OpKind::MatMul => {
                let as_ = shape_of(&shapes, node.inputs[0])?;
                let bs = shape_of(&shapes, node.inputs[1])?;
                let os = shape_of(&shapes, node.output)?;
                let uniforms = MatMulUniforms {
                    n: dim(os.n, "matmul")?,
                    rows: dim(as_.c, "matmul rows")?,
                    k: dim(as_.w, "matmul inner dim")?,
                    cols: dim(bs.c, "matmul columns")?,
                    a_n_stride: dim(as_.c * as_.h * as_.w, "matmul stride")?,
                    a_row_stride: dim(as_.h * as_.w, "matmul stride")?,
                    b_n_stride: dim(bs.c * bs.h * bs.w, "matmul stride")?,
                    b_j_stride: dim(bs.h * bs.w, "matmul stride")?,
                    b_k_stride: dim(bs.w, "matmul stride")?,
                };
                let mut a_arg = handle_for(&slot_of, &handles, node.inputs[0])?;
                let mut b_arg = handle_for(&slot_of, &handles, node.inputs[1])?;
                let mut o_arg = out;
                let mut u_arg = uniforms;
                let mut params: [*mut c_void; 4] = [
                    &mut a_arg as *mut u64 as *mut c_void,
                    &mut b_arg as *mut u64 as *mut c_void,
                    &mut o_arg as *mut u64 as *mut c_void,
                    &mut u_arg as *mut MatMulUniforms as *mut c_void,
                ];
                dispatches.push(launch(&gpu, K_MATMUL, os.numel(), &mut params)?);
            }
            OpKind::QuantizeLinear { scale, zero_point } => {
                let xs = shape_of(&shapes, node.inputs[0])?;
                let mut x_arg = handle_for(&slot_of, &handles, node.inputs[0])?;
                let mut o_arg = out;
                let mut scale_arg = *scale;
                let mut zero_arg = i32::from(*zero_point);
                let mut count = dim(xs.numel(), "quantize elements")?;
                let mut params: [*mut c_void; 5] = [
                    &mut x_arg as *mut u64 as *mut c_void,
                    &mut o_arg as *mut u64 as *mut c_void,
                    &mut scale_arg as *mut f32 as *mut c_void,
                    &mut zero_arg as *mut i32 as *mut c_void,
                    &mut count as *mut i32 as *mut c_void,
                ];
                dispatches.push(launch(&gpu, K_QUANTIZE, xs.numel(), &mut params)?);
            }
            OpKind::DequantizeLinear { scale, zero_point } => {
                let xs = shape_of(&shapes, node.inputs[0])?;
                let mut x_arg = handle_for(&slot_of, &handles, node.inputs[0])?;
                let mut o_arg = out;
                let mut scale_arg = *scale;
                let mut zero_arg = i32::from(*zero_point);
                let mut count = dim(xs.numel(), "dequantize elements")?;
                let mut params: [*mut c_void; 5] = [
                    &mut x_arg as *mut u64 as *mut c_void,
                    &mut o_arg as *mut u64 as *mut c_void,
                    &mut scale_arg as *mut f32 as *mut c_void,
                    &mut zero_arg as *mut i32 as *mut c_void,
                    &mut count as *mut i32 as *mut c_void,
                ];
                dispatches.push(launch(&gpu, K_DEQUANTIZE, xs.numel(), &mut params)?);
            }
        }
    }

    let mut outputs: Vec<TensorBuffer> = Vec::with_capacity(graph.outputs.len());
    let mut d2h_bytes: u64 = 0;
    for &id in &graph.outputs {
        let shape = shape_of(&shapes, id)?;
        let dtype = dtypes
            .get(&id)
            .copied()
            .ok_or_else(|| GraphError::TensorMismatch(format!("tensor {id} has no dtype")))?;
        let bytes = shape.byte_len(dtype);
        if bytes > max_bytes {
            return Err(GraphError::TensorMismatch(format!(
                "output {id} needs {bytes} bytes, staging holds {max_bytes}"
            ))
            .into());
        }
        let source = handle_for(&slot_of, &handles, id)?;
        dispatches.push(device_copy(&gpu, staging_handle, source, bytes)?);
        let mut host = vec![0u8; bytes];
        gpu.download_buffer(&staging, &mut host)?;
        d2h_bytes += bytes as u64;
        outputs.push(TensorBuffer::from_bytes(dtype, shape, host)?);
    }

    Ok(CudaGraphExecution {
        backend: Backend::Cuda,
        outputs,
        plan: plan.clone(),
        dispatches,
        timing: BackendTiming {
            h2d_bytes,
            d2h_bytes,
            ..BackendTiming::default()
        },
        fallback_reason: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_infer_graph::{fold_conv_batchnorm, max_abs_diff, Node};

    fn param(len: usize, values: &[f32]) -> TensorBuffer {
        assert_eq!(values.len(), len);
        TensorBuffer::from_f32(TensorShape::vector(len), values).expect("parameter tensor")
    }

    fn one(shape: TensorShape, value: f32) -> TensorBuffer {
        TensorBuffer::from_f32(shape, &vec![value; shape.numel()]).expect("constant tensor")
    }

    /// A graph with one node of every `OpKind`, all outputs declared.
    fn every_op_graph() -> Graph {
        let mut graph = Graph::new();
        let xid = graph.add_constant(
            TensorBuffer::from_f32(
                TensorShape::new(1, 2, 4, 4),
                &(0..32).map(|i| (i as f32) * 0.031 - 0.4).collect::<Vec<f32>>(),
            )
            .unwrap(),
        );

        // Conv2d + bias.
        let w = graph.add_constant(
            TensorBuffer::from_f32(
                TensorShape::new(2, 2, 3, 3),
                &(0..36).map(|i| ((i % 7) as f32 - 3.0) * 0.09).collect::<Vec<f32>>(),
            )
            .unwrap(),
        );
        let b = graph.add_constant(param(2, &[0.11, -0.07]));
        let conv = graph.fresh_id();
        graph.add_node(Node::new(
            OpKind::Conv2d {
                stride: (1, 1),
                pad: (1, 1, 1, 1),
                dilation: (1, 1),
                weight: w,
                bias: Some(b),
            },
            vec![xid],
            conv,
            "conv2d",
        ));

        // BatchNorm.
        let bn_scale = graph.add_constant(param(2, &[1.3, 0.7]));
        let bn_offset = graph.add_constant(param(2, &[-0.2, 0.4]));
        let bn_mean = graph.add_constant(param(2, &[0.1, -0.15]));
        let bn_var = graph.add_constant(param(2, &[0.9, 1.21]));
        let bn = graph.fresh_id();
        graph.add_node(Node::new(
            OpKind::BatchNorm {
                scale: bn_scale,
                offset: bn_offset,
                mean: bn_mean,
                var: bn_var,
                epsilon: 1e-5,
            },
            vec![xid],
            bn,
            "batchnorm",
        ));

        // DepthwiseConv2d.
        let dw = graph.add_constant(
            TensorBuffer::from_f32(
                TensorShape::new(2, 1, 3, 3),
                &(0..18).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect::<Vec<f32>>(),
            )
            .unwrap(),
        );
        let dw_out = graph.fresh_id();
        graph.add_node(Node::new(
            OpKind::DepthwiseConv2d {
                stride: (1, 1),
                pad: (1, 1, 1, 1),
                weight: dw,
                bias: None,
            },
            vec![xid],
            dw_out,
            "depthwise",
        ));

        // GroupNorm.
        let gn_scale = graph.add_constant(param(2, &[1.0, 0.5]));
        let gn_offset = graph.add_constant(param(2, &[0.0, 0.25]));
        let gn = graph.fresh_id();
        graph.add_node(Node::new(
            OpKind::GroupNorm {
                groups: 2,
                scale: gn_scale,
                offset: gn_offset,
                epsilon: 1e-5,
            },
            vec![xid],
            gn,
            "groupnorm",
        ));

        let relu = graph.fresh_id();
        graph.add_node(Node::new(OpKind::Relu, vec![xid], relu, "relu"));
        let silu = graph.fresh_id();
        graph.add_node(Node::new(OpKind::SiLU, vec![xid], silu, "silu"));
        let sigmoid = graph.fresh_id();
        graph.add_node(Node::new(OpKind::Sigmoid, vec![xid], sigmoid, "sigmoid"));
        let add = graph.fresh_id();
        graph.add_node(Node::new(OpKind::Add, vec![relu, silu], add, "add"));
        let mul = graph.fresh_id();
        graph.add_node(Node::new(OpKind::Mul, vec![silu, sigmoid], mul, "mul"));

        let resize = graph.fresh_id();
        graph.add_node(Node::new(
            OpKind::ResizeBilinear {
                scale_h: 2.0,
                scale_w: 2.0,
                align_corners: false,
            },
            vec![xid],
            resize,
            "resize",
        ));

        let grid_values = [
            -0.75f32, 0.25, -0.25, 0.75, 0.5, -0.5, 0.0, 0.25, -0.9, 0.9, 0.1, -0.1, 0.35,
            -0.35, 0.6, -0.6, -0.75, 0.75, -0.5, 0.5, 0.0, 0.0, 0.25, -0.25, 0.9, -0.9, 0.4,
            -0.4, 0.15, -0.15, 0.7, -0.7,
        ];
        let grid = graph.add_constant(
            TensorBuffer::from_f32(TensorShape::new(1, 2, 4, 4), &grid_values).unwrap(),
        );
        let sampled = graph.fresh_id();
        graph.add_node(Node::new(
            OpKind::GridSample,
            vec![xid, grid],
            sampled,
            "gridsample",
        ));

        let a = graph.add_constant(
            TensorBuffer::from_f32(
                TensorShape::new(1, 2, 1, 4),
                &[1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 0.25, -0.75],
            )
            .unwrap(),
        );
        let bm = graph.add_constant(
            TensorBuffer::from_f32(
                TensorShape::new(1, 3, 4, 1),
                &[5.0, 7.0, 1.0, -1.0, 6.0, 8.0, 2.0, 0.5, 0.25, -0.5, 0.75, 1.5],
            )
            .unwrap(),
        );
        let mm = graph.fresh_id();
        graph.add_node(Node::new(OpKind::MatMul, vec![a, bm], mm, "matmul"));

        let quantized = graph.fresh_id();
        graph.add_node(Node::new(
            OpKind::QuantizeLinear {
                scale: 0.125,
                zero_point: 0,
            },
            vec![xid],
            quantized,
            "quantize",
        ));
        let dequantized = graph.fresh_id();
        graph.add_node(Node::new(
            OpKind::DequantizeLinear {
                scale: 0.125,
                zero_point: 0,
            },
            vec![quantized],
            dequantized,
            "dequantize",
        ));

        graph.set_outputs(vec![
            conv, bn, dw_out, gn, relu, silu, sigmoid, add, mul, resize, sampled, mm, quantized,
            dequantized,
        ]);
        graph
    }

    /// The `mini_portrait_residual_network` from `native_infer_graph` tests,
    /// with the stem supplied as a runtime input.
    fn mini_portrait_network() -> Graph {
        let mut graph = Graph::new();
        let input = graph.fresh_id();
        graph.declare_runtime_input(input);

        let w1 = graph.add_constant(
            TensorBuffer::from_f32(TensorShape::new(2, 2, 3, 3), &vec![0.1f32; 36]).unwrap(),
        );
        let b1 = graph.add_constant(param(2, &[0.0, 0.0]));
        let c1 = graph.fresh_id();
        graph.add_node(Node::new(
            OpKind::Conv2d {
                stride: (1, 1),
                pad: (1, 1, 1, 1),
                dilation: (1, 1),
                weight: w1,
                bias: Some(b1),
            },
            vec![input],
            c1,
            "conv1",
        ));
        let s1 = graph.add_constant(param(2, &[1.0, 1.0]));
        let o1 = graph.add_constant(param(2, &[0.0, 0.0]));
        let m1 = graph.add_constant(param(2, &[0.0, 0.0]));
        let v1 = graph.add_constant(param(2, &[1.0, 1.0]));
        let n1 = graph.fresh_id();
        graph.add_node(Node::new(
            OpKind::BatchNorm {
                scale: s1,
                offset: o1,
                mean: m1,
                var: v1,
                epsilon: 1e-5,
            },
            vec![c1],
            n1,
            "bn1",
        ));
        let a1 = graph.fresh_id();
        graph.add_node(Node::new(OpKind::SiLU, vec![n1], a1, "silu1"));

        let dw_w = graph.add_constant(
            TensorBuffer::from_f32(TensorShape::new(2, 1, 3, 3), &vec![0.05f32; 18]).unwrap(),
        );
        let dw = graph.fresh_id();
        graph.add_node(Node::new(
            OpKind::DepthwiseConv2d {
                stride: (1, 1),
                pad: (1, 1, 1, 1),
                weight: dw_w,
                bias: None,
            },
            vec![a1],
            dw,
            "dw",
        ));
        let gs = graph.add_constant(param(2, &[1.0, 1.0]));
        let go = graph.add_constant(param(2, &[0.0, 0.0]));
        let gn = graph.fresh_id();
        graph.add_node(Node::new(
            OpKind::GroupNorm {
                groups: 2,
                scale: gs,
                offset: go,
                epsilon: 1e-5,
            },
            vec![dw],
            gn,
            "gn",
        ));
        let skip_up = graph.fresh_id();
        graph.add_node(Node::new(
            OpKind::ResizeBilinear {
                scale_h: 2.0,
                scale_w: 2.0,
                align_corners: false,
            },
            vec![gn],
            skip_up,
            "skip_up",
        ));
        let up = graph.fresh_id();
        graph.add_node(Node::new(
            OpKind::ResizeBilinear {
                scale_h: 2.0,
                scale_w: 2.0,
                align_corners: false,
            },
            vec![a1],
            up,
            "up",
        ));
        let sum = graph.fresh_id();
        graph.add_node(Node::new(OpKind::Add, vec![up, skip_up], sum, "residual"));
        let out = graph.fresh_id();
        graph.add_node(Node::new(OpKind::Sigmoid, vec![sum], out, "sigmoid"));
        graph.set_outputs(vec![out]);
        graph
    }

    fn runtime_inputs_for(graph: &Graph) -> HashMap<usize, TensorBuffer> {
        let mut inputs = HashMap::new();
        for &id in &graph.runtime_inputs {
            inputs.insert(id, one(TensorShape::new(1, 2, 4, 4), 0.2));
        }
        inputs
    }

    /// Conv2d -> BatchNorm -> SiLU, i.e. exactly the pair
    /// `fold_conv_batchnorm` rewrites.
    fn conv_batchnorm_graph() -> Graph {
        let mut graph = Graph::new();
        let input = graph.fresh_id();
        graph.declare_runtime_input(input);

        let weight = graph.add_constant(
            TensorBuffer::from_f32(
                TensorShape::new(2, 2, 3, 3),
                &(0..36)
                    .map(|i| ((i % 7) as f32 - 3.0) * 0.11)
                    .collect::<Vec<f32>>(),
            )
            .unwrap(),
        );
        let bias = graph.add_constant(param(2, &[0.17, -0.23]));
        let conv = graph.fresh_id();
        graph.add_node(Node::new(
            OpKind::Conv2d {
                stride: (1, 1),
                pad: (1, 1, 1, 1),
                dilation: (1, 1),
                weight,
                bias: Some(bias),
            },
            vec![input],
            conv,
            "conv",
        ));

        let scale = graph.add_constant(param(2, &[1.3, 0.7]));
        let offset = graph.add_constant(param(2, &[-0.2, 0.4]));
        let mean = graph.add_constant(param(2, &[0.1, -0.15]));
        let var = graph.add_constant(param(2, &[0.9, 1.21]));
        let bn = graph.fresh_id();
        graph.add_node(Node::new(
            OpKind::BatchNorm {
                scale,
                offset,
                mean,
                var,
                epsilon: 1e-5,
            },
            vec![conv],
            bn,
            "bn",
        ));
        let activated = graph.fresh_id();
        graph.add_node(Node::new(OpKind::SiLU, vec![bn], activated, "silu"));
        graph.set_outputs(vec![activated]);
        graph
    }

    fn compare_outputs(
        cuda: &CudaGraphExecution,
        reference: &crate::native_infer_graph::ExecutionResult,
        tolerance: f32,
        label: &str,
    ) {
        assert_eq!(cuda.outputs.len(), reference.outputs.len());
        for (index, (got, want)) in cuda.outputs.iter().zip(&reference.outputs).enumerate() {
            assert_eq!(got.shape, want.shape, "{label} output {index} shape");
            assert_eq!(got.dtype, want.dtype, "{label} output {index} dtype");
            if got.dtype == DType::Int8 {
                assert_eq!(
                    got.bytes, want.bytes,
                    "{label} output {index} quantized bytes must match exactly"
                );
            } else {
                let diff = max_abs_diff(got, want).expect("comparable fp32 tensors");
                assert!(
                    diff < tolerance,
                    "{label} output {index} differs by {diff} (tolerance {tolerance})"
                );
            }
        }
    }

    #[test]
    fn uniform_blocks_match_the_cuda_structs() {
        assert_eq!(std::mem::size_of::<ConvUniforms>(), 68);
        assert_eq!(std::mem::size_of::<NormUniforms>(), 16);
        assert_eq!(std::mem::size_of::<GroupNormUniforms>(), 20);
        assert_eq!(std::mem::size_of::<BroadcastUniforms>(), 48);
        assert_eq!(std::mem::size_of::<ResizeUniforms>(), 44);
        assert_eq!(std::mem::size_of::<GridSampleUniforms>(), 24);
        assert_eq!(std::mem::size_of::<MatMulUniforms>(), 36);
        assert_eq!(std::mem::align_of::<ConvUniforms>(), 4);
        assert_eq!(std::mem::align_of::<ResizeUniforms>(), 4);
        assert_eq!(KERNELS.len(), 15);
    }

    #[test]
    fn planning_covers_runtime_inputs_that_are_not_constants() {
        let graph = mini_portrait_network();
        assert!(plan_memory(&graph).is_err(), "the raw planner needs shapes");
        let inputs = runtime_inputs_for(&graph);
        let plan = plan_for_execution(&graph, &inputs).expect("planned with runtime shapes");
        assert!(plan.slot_count >= 1);
        assert!(plan.slot_count <= plan.total_tensors);
        assert_eq!(
            plan.total_tensors,
            graph.constants.len() + inputs.len() + graph.nodes.len()
        );
    }

    #[test]
    fn unavailable_device_is_never_reported_as_a_gpu_success() {
        if device_available() {
            return; // the negative case only exists without a usable image
        }
        let graph = every_op_graph();
        match execute_graph_cuda(&graph, &HashMap::new(), ExecutionMode::Strict) {
            Err(CudaGraphError::Cuda(CudaError::Unavailable(_))) => {}
            other => panic!("strict mode must refuse without a device, got {other:?}"),
        }

        let fallback = execute_graph_cuda(&graph, &HashMap::new(), ExecutionMode::AutoFallback)
            .expect("auto fallback runs the cpu reference");
        assert_eq!(fallback.backend, Backend::Cpu);
        assert!(fallback.fallback_reason.is_some());
        assert!(fallback.dispatches.is_empty());
        let reference = execute_graph(&graph, &HashMap::new()).unwrap();
        compare_outputs(&fallback, &reference, 1e-6, "fallback");
    }

    #[test]
    fn every_op_kind_matches_the_cpu_reference_on_the_device() {
        if !device_available() {
            return; // documented: no driver, or the PTX image is still a placeholder
        }
        let graph = every_op_graph();
        let reference = execute_graph(&graph, &HashMap::new()).unwrap();
        let execution = execute_graph_cuda(&graph, &HashMap::new(), ExecutionMode::Strict)
            .expect("cuda graph execution");
        assert_eq!(execution.backend, Backend::Cuda);
        assert!(execution.fallback_reason.is_none());
        assert_eq!(execution.outputs.len(), 14);
        assert!(
            execution.dispatches.len() >= graph.nodes.len(),
            "every node must produce a real launch"
        );
        assert!(execution.timing.h2d_bytes > 0);
        assert!(execution.timing.d2h_bytes > 0);
        compare_outputs(&execution, &reference, 1e-4, "every-op");
    }

    #[test]
    fn mini_portrait_residual_network_matches_cpu_within_1e_4() {
        if !device_available() {
            return;
        }
        let graph = mini_portrait_network();
        let inputs = runtime_inputs_for(&graph);
        let reference = execute_graph(&graph, &inputs).unwrap();
        let execution = execute_graph_cuda(&graph, &inputs, ExecutionMode::Strict)
            .expect("cuda graph execution");
        assert_eq!(execution.outputs[0].shape, TensorShape::new(1, 2, 8, 8));
        compare_outputs(&execution, &reference, 1e-4, "mini-portrait");
        let values = execution.outputs[0].f32_values().unwrap();
        assert!(values.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
    }

    #[test]
    fn folded_conv_batchnorm_graph_matches_the_cpu_reference() {
        if !device_available() {
            return;
        }
        let graph = conv_batchnorm_graph();
        let mut folded = graph.clone();
        let report = fold_conv_batchnorm(&mut folded).expect("fold conv+bn");
        assert_eq!(report.fused_pairs(), 1);
        assert_eq!(folded.nodes.len(), graph.nodes.len() - 1);

        let inputs = runtime_inputs_for(&graph);
        let reference = execute_graph(&folded, &inputs).unwrap();
        let execution = execute_graph_cuda(&folded, &inputs, ExecutionMode::Strict)
            .expect("cuda graph execution");
        assert!(execution.dispatches.len() >= folded.nodes.len());
        compare_outputs(&execution, &reference, 1e-4, "folded");
    }
}
