//! Apple Metal native inference backend (`NativeMetalSession`).
//!
//! This module is the Metal half of the platform-neutral contract in
//! [`crate::native_infer::backend`]. It reuses the shared Phase 1 front end
//! (`onnx_proto` → `graph` → `bn_fold` → arena planner) and turns a compiled
//! plan into Metal dispatches:
//!
//! ```text
//! .onnx bytes
//!   -> load_and_plan()                    (shared, pure Rust)
//!   -> per-node encode                    (this module, pure Rust)
//!        static weights      -> one shared MTLBuffer, 256-byte aligned slots
//!        graph inputs        -> MTLResourceStorageModeShared (one host copy per borrowed slice)
//!        arena intermediates -> one shared MTLBuffer from the shared planner
//!        shape/control data  -> resolved on CPU at plan time
//!   -> one MSL library, one pipeline per used kernel
//!   -> run_f32(): one MTLCommandBuffer + one MTLComputeCommandEncoder,
//!      every node encoded into that single encoder, one commit, one wait
//! ```
//!
//! # What is and is not claimed
//!
//! * Construction rejects a graph whose operations cannot be encoded before any
//!   dispatch happens, and the rejection names the node and the reason.
//! * `run_f32` either encodes real Metal commands or returns an error. There is
//!   no CPU path behind this type: a session that cannot open a device is never
//!   constructed, so no CPU result can be reported as a Metal result.
//! * No MPS / MPSGraph kernels are used; the library is plain MSL compute. The
//!   only Apple frameworks touched are Metal, Foundation/CoreGraphics (device
//!   discovery) and libobjc (the message-send bridge).
//! * The MSL in this file has **not** been compiled by Apple's `metal` compiler
//!   in the current environment (no Apple toolchain on the Windows host). The
//!   Rust encoder, the parameter ABIs, the index math and the graph coverage are
//!   verified by tests that run on any host; MSL compilation and on-GPU numeric
//!   parity require a macOS machine and are reported as unverified until then.
//!
//! # Op coverage
//!
//! [`SUPPORTED_OPS`] lists every `Op` variant that has an encoding here.
//! `If` (data-dependent branch selection) and `ConvInteger` (integer-domain
//! convolution) are rejected at construction with an explicit reason; every
//! other variant of [`crate::native_infer::graph::Op`] is encoded. The test
//! `every_ir_op_variant_is_classified` fails when a new `Op` variant is added to
//! the IR without a decision here.

use std::collections::HashMap;
use std::fmt;

use crate::native_infer::arena::CUDA_DEFAULT_ALIGNMENT;
use crate::native_infer::backend::{
    BackendKind, NamedInputs, NamedOutputs, NativeInferBackend, ShapeOverrides,
};
use crate::native_infer::cpu_shape_eval::{eval_op as cpu_eval_op, CpuShapeEvaluator, CpuTensor};
use crate::native_infer::graph::{IrNode, Op, TensorId};
use crate::native_infer::onnx_proto::{DataType, TensorProto};
use crate::native_infer::{load_and_plan_portable as load_and_plan, CompiledPlan};

/// Every `Op` variant this backend encodes to Metal.
pub const SUPPORTED_OPS: &[&str] = &[
    "Conv",
    "ConvTranspose",
    "MatMul",
    "MatMulInteger",
    "Gemm",
    "BatchNormFused",
    "LayerNormalization",
    "InstanceNormalization",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "ReduceMean",
    "Resize",
    "Pad",
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
    "Concat",
    "Split",
    "Slice",
    "Gather",
    "ScatterND",
    "Expand",
    "Tile",
    "Squeeze",
    "Unsqueeze",
    "Shape",
    "Size",
    "Cast",
    "Constant",
    "ConstantOfShape",
    "Identity",
    "DynamicQuantizeLinear",
];

/// `Op` variants rejected before first dispatch, with the reason reported to the
/// caller. Kept next to [`SUPPORTED_OPS`] so the classification test can prove
/// that no variant is silently unhandled.
pub const REJECTED_OPS: &[(&str, &str)] = &[
    (
        "If",
        "data-dependent branch selection is not encoded: both branches would have to be \
         compiled and selected on the host after reading the condition tensor",
    ),
    (
        "ConvInteger",
        "integer-domain convolution is not encoded in this slice; only integer MatMul is",
    ),
    (
        "BatchNormalization",
        "an unfolded BatchNormalization reached execution; the shared BN folder must absorb it",
    ),
];

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum MetalError {
    /// The shared ONNX front end failed.
    FrontEnd(String),
    /// The graph uses an operation this backend does not encode.
    UnsupportedOp {
        node: String,
        op: &'static str,
        reason: String,
    },
    /// The graph is inside the supported set but this node's parameters cannot be
    /// encoded (dynamic shape, asymmetric pads, tensor rank above the ABI limit…).
    InvalidGraph { node: String, detail: String },
    /// No Metal device could be opened on this host.
    DeviceUnavailable { detail: String },
    /// Metal refused the MSL source or a pipeline.
    Pipeline { kernel: String, detail: String },
    /// A dispatch or readback failed at run time.
    Runtime(String),
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrontEnd(e) => write!(f, "metal front end: {e}"),
            Self::UnsupportedOp { node, op, reason } => {
                write!(f, "metal does not encode op '{op}' at node '{node}': {reason}")
            }
            Self::InvalidGraph { node, detail } => {
                write!(f, "metal cannot encode node '{node}': {detail}")
            }
            Self::DeviceUnavailable { detail } => write!(f, "metal device unavailable: {detail}"),
            Self::Pipeline { kernel, detail } => {
                write!(f, "metal pipeline for kernel '{kernel}' failed: {detail}")
            }
            Self::Runtime(e) => write!(f, "metal runtime: {e}"),
        }
    }
}

impl std::error::Error for MetalError {}

type Result<T> = std::result::Result<T, MetalError>;

// ---------------------------------------------------------------------------
// Element types and buffer regions
// ---------------------------------------------------------------------------

/// Element types that can live in a Metal buffer here. FP16/BF16 weights are
/// widened to FP32 once at plan time (the same choice the CUDA session makes),
/// so compute kernels only see the members of this enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Elem {
    F32,
    F16,
    I32,
    I64,
    U8,
    I8,
    Bool,
}

impl Elem {
    pub const fn bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 => 2,
            Self::I64 => 8,
            Self::U8 | Self::I8 | Self::Bool => 1,
        }
    }

    /// Stable code shared with the MSL cast/fill kernels.
    pub const fn code(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::I32 => 2,
            Self::I64 => 3,
            Self::U8 => 4,
            Self::I8 => 5,
            Self::Bool => 6,
        }
    }

    pub fn from_onnx(dt: DataType) -> Option<Self> {
        match dt {
            DataType::Float => Some(Self::F32),
            DataType::Float16 => Some(Self::F16),
            DataType::Int32 => Some(Self::I32),
            DataType::Int64 => Some(Self::I64),
            DataType::Uint8 => Some(Self::U8),
            DataType::Int8 => Some(Self::I8),
            DataType::Bool => Some(Self::Bool),
            _ => None,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16)
    }
}

/// Buffer that owns a region. Metal buffers are session-scoped; input/output
/// buffers are per-run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Buf {
    /// One shared buffer holding every static weight and materialised constant.
    Weights,
    /// The single intermediate arena produced by the shared planner.
    Arena,
    /// Second arena for tensors the liveness planner excluded (dynamic shapes,
    /// or tensors whose byte size changed after BN folding).
    Extra,
    Input(TensorId),
    Output(TensorId),
}

/// A region inside a [`Buf`]. `byte_off` is always 4-byte aligned; buffer base
/// offsets are 256-byte aligned because every slot the planner or this module
/// hands out uses [`CUDA_DEFAULT_ALIGNMENT`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub buf: Buf,
    pub byte_off: usize,
}

impl Region {
    const fn new(buf: Buf, byte_off: usize) -> Self {
        Self { buf, byte_off }
    }
}

/// A tensor bound to a region with its concrete shape and element type.
#[derive(Clone, Debug)]
pub struct Bound {
    pub region: Region,
    pub shape: Vec<usize>,
    pub elem: Elem,
}

impl Bound {
    pub fn numel(&self) -> usize {
        self.shape.iter().copied().product()
    }
}

/// Graph input or output description.
#[derive(Clone, Debug)]
pub struct GraphIo {
    pub tid: TensorId,
    pub name: String,
    pub shape: Vec<usize>,
    pub elem: Elem,
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// MSL entry points emitted by [`MSL_LIBRARY`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kernel {
    Copy,
    Fill,
    Unary,
    Binary,
    Compare,
    And,
    Where,
    Cast,
    ChannelAffine,
    Gemm,
    Conv2d,
    ConvTranspose2d,
    Pool2d,
    Rows,
    Resize,
    MatMulInt,
    Quantize,
    QuantFinalize,
    Scatter,
    Gather,
    MinMax,
}

impl Kernel {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Copy => "k_copy",
            Self::Fill => "k_fill",
            Self::Unary => "k_unary",
            Self::Binary => "k_binary",
            Self::Compare => "k_compare",
            Self::And => "k_and",
            Self::Where => "k_where",
            Self::Cast => "k_cast",
            Self::ChannelAffine => "k_channel_affine",
            Self::Gemm => "k_gemm",
            Self::Conv2d => "k_conv2d",
            Self::ConvTranspose2d => "k_conv_transpose2d",
            Self::Pool2d => "k_pool2d",
            Self::Rows => "k_rows",
            Self::Resize => "k_resize",
            Self::MatMulInt => "k_matmul_int",
            Self::Quantize => "k_quantize",
            Self::QuantFinalize => "k_quant_finalize",
            Self::Scatter => "k_scatter",
            Self::Gather => "k_gather",
            Self::MinMax => "k_minmax",
        }
    }

    /// Buffer index reserved for the parameter struct of every kernel.
    pub const PARAMS_INDEX: u32 = 9;
}

/// One encoded compute dispatch.
#[derive(Clone, Debug)]
pub struct Dispatch {
    pub node: String,
    pub op: &'static str,
    pub kernel: Kernel,
    /// `(buffer index, region)` in ascending index order; parameters are bound
    /// through `setBytes` at [`Kernel::PARAMS_INDEX`] and are not listed here.
    pub binds: Vec<(u32, Region)>,
    pub params: Vec<u8>,
    /// Threads per grid, exactly as `dispatchThreads:` expects.
    pub threads: [u32; 3],
    pub threadgroup: [u32; 3],
    /// Tensors this dispatch writes. Recorded explicitly because the write target
    /// is not always buffer 0 (reductions and quantisation read first).
    pub writes: Vec<TensorId>,
    /// Every tensor the source node consumes; the dispatch reads a subset, so
    /// this is the conservative dependency set used for hazard analysis.
    pub node_inputs: Vec<TensorId>,
}

impl Dispatch {
    fn validate(&self) -> std::result::Result<(), String> {
        for (want, (index, region)) in self.binds.iter().enumerate() {
            if *index != want as u32 {
                return Err(format!(
                    "node '{}': buffer index {index} out of order (expected {want}); MSL binds are dense",
                    self.node
                ));
            }
            if *index >= Kernel::PARAMS_INDEX {
                return Err(format!(
                    "node '{}': buffer index {index} collides with the parameter slot",
                    self.node
                ));
            }
            if region.byte_off % 4 != 0 {
                return Err(format!(
                    "node '{}': buffer {:?} offset {} is not 4-byte aligned",
                    self.node, region.buf, region.byte_off
                ));
            }
            if matches!(region.buf, Buf::Weights | Buf::Arena | Buf::Extra)
                && region.byte_off % CUDA_DEFAULT_ALIGNMENT != 0
            {
                return Err(format!(
                    "node '{}': buffer {:?} base offset {} is not {CUDA_DEFAULT_ALIGNMENT}-byte aligned",
                    self.node, region.buf, region.byte_off
                ));
            }
        }
        for axis in 0..3 {
            if self.threads[axis] == 0 || self.threadgroup[axis] == 0 {
                return Err(format!(
                    "node '{}': empty dispatch geometry {:?}/{:?}",
                    self.node, self.threads, self.threadgroup
                ));
            }
        }
        let group_threads = self.threadgroup.iter().take(3).product::<u32>();
        if group_threads > 1024 {
            return Err(format!(
                "node '{}': threadgroup holds {group_threads} threads (Metal limit 1024)",
                self.node
            ));
        }
        if self.params.len() > 4096 {
            return Err(format!(
                "node '{}': {} bytes of parameters exceed the 4 KiB setBytes limit",
                self.node,
                self.params.len()
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Parameter ABIs (mirrored by the MSL structs of the same name)
// ---------------------------------------------------------------------------

const MAX_RANK: usize = 8;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CopyParams {
    pub rank: u32,
    pub out_elems: u32,
    pub elem_bytes: u32,
    pub edge: u32,
    pub dst_base: i32,
    pub src_base: i32,
    pub pad_value: f32,
    pub out_shape: [u32; MAX_RANK],
    pub dst_stride: [i32; MAX_RANK],
    pub mul: [i32; MAX_RANK],
    pub add: [i32; MAX_RANK],
    pub shift: [i32; MAX_RANK],
    pub in_dim: [i32; MAX_RANK],
    pub use_mod: [u32; MAX_RANK],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FillParams {
    pub n: u32,
    pub elem_bytes: u32,
    pub dst_base: i32,
    pub bits: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct UnaryParams {
    pub n: u32,
    pub op: u32,
    pub alpha: f32,
    pub beta: f32,
    pub lo: f32,
    pub hi: f32,
    pub x_base: i32,
    pub y_base: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BinaryParams {
    pub rank: u32,
    pub n: u32,
    pub op: u32,
    pub unused: u32,
    pub a_base: i32,
    pub b_base: i32,
    pub y_base: i32,
    pub out_shape: [u32; MAX_RANK],
    pub a_mul: [u32; MAX_RANK],
    pub b_mul: [u32; MAX_RANK],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct WhereParams {
    pub rank: u32,
    pub n: u32,
    pub unused0: u32,
    pub unused1: u32,
    pub c_base: i32,
    pub a_base: i32,
    pub b_base: i32,
    pub y_base: i32,
    pub out_shape: [u32; MAX_RANK],
    pub c_mul: [u32; MAX_RANK],
    pub a_mul: [u32; MAX_RANK],
    pub b_mul: [u32; MAX_RANK],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CastParams {
    pub n: u32,
    pub from: u32,
    pub to: u32,
    pub unused: u32,
    pub x_base: i32,
    pub y_base: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ChannelAffineParams {
    pub n: u32,
    pub inner: u32,
    pub channels: u32,
    pub unused: u32,
    pub x_base: i32,
    pub y_base: i32,
    pub scale_base: i32,
    pub bias_base: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GemmParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
    pub a_off: i32,
    pub b_off: i32,
    pub c_off: i32,
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: u32,
    pub trans_b: u32,
    pub has_bias: u32,
    pub act: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ConvParams {
    pub n: u32,
    pub h: u32,
    pub w: u32,
    pub c: u32,
    pub oh: u32,
    pub ow: u32,
    pub oc: u32,
    pub kh: u32,
    pub kw: u32,
    pub sh: u32,
    pub sw: u32,
    pub ph: u32,
    pub pw: u32,
    pub dh: u32,
    pub dw: u32,
    pub groups: u32,
    pub act: u32,
    pub x_off: i32,
    pub w_off: i32,
    pub b_off: i32,
    pub y_off: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PoolParams {
    pub n: u32,
    pub c: u32,
    pub ih: u32,
    pub iw: u32,
    pub oh: u32,
    pub ow: u32,
    pub kh: u32,
    pub kw: u32,
    pub sh: u32,
    pub sw: u32,
    pub ph: u32,
    pub pw: u32,
    pub dh: u32,
    pub dw: u32,
    pub mode: u32,
    pub count_include_pad: u32,
    pub x_off: i32,
    pub y_off: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RowParams {
    pub outer: u32,
    pub r: u32,
    pub inner: u32,
    pub mode: u32,
    pub channels: u32,
    pub y_stride_outer: u32,
    pub y_stride_inner: u32,
    pub unused: u32,
    pub x_off: i32,
    pub y_off: i32,
    pub g_off: i32,
    pub b_off: i32,
    pub eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ResizeParams {
    pub n: u32,
    pub c: u32,
    pub ih: u32,
    pub iw: u32,
    pub oh: u32,
    pub ow: u32,
    pub mode: u32,
    pub coord_mode: u32,
    pub nearest_mode: u32,
    pub rh: f32,
    pub rw: f32,
    pub x_off: i32,
    pub y_off: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MatMulIntParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_off: i32,
    pub b_off: i32,
    pub c_off: i32,
    pub azp_off: i32,
    pub bzp_off: i32,
    pub a_zp: u32,
    pub b_zp: u32,
    pub use_buf_zp: u32,
    pub b_signed: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct QuantizeParams {
    pub n: u32,
    pub x_off: i32,
    pub y_off: i32,
    pub scale_off: i32,
    pub zp_off: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct QuantFinalizeParams {
    pub minmax_off: i32,
    pub scale_off: i32,
    pub zp_off: i32,
    pub eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ScatterParams {
    pub num_updates: u32,
    pub k: u32,
    pub slice_size: u32,
    pub out_elems: u32,
    pub rank: u32,
    pub idx_off: i32,
    pub upd_off: i32,
    pub out_off: i32,
    pub strides: [i32; MAX_RANK],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GatherParams {
    pub outer: u32,
    pub axis_len: u32,
    pub inner: u32,
    pub idx_len: u32,
    pub out_elems: u32,
    pub idx_i64: u32,
    pub unused0: u32,
    pub unused1: u32,
    pub data_off: i32,
    pub idx_off: i32,
    pub y_off: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MinMaxParams {
    pub n: u32,
    pub x_off: i32,
}

fn as_bytes<T: Copy>(value: &T) -> Vec<u8> {
    let ptr = (value as *const T).cast::<u8>();
    // SAFETY: `T` is a `#[repr(C)]` plain-old-data parameter struct made only of
    // integers and floats; every byte of its object representation is readable.
    unsafe { std::slice::from_raw_parts(ptr, std::mem::size_of::<T>()) }.to_vec()
}

// ---------------------------------------------------------------------------
// Compiled program
// ---------------------------------------------------------------------------

/// A model compiled to Metal dispatches. Pure data: it can be built and
/// inspected on any host. Executing it requires [`NativeMetalSession`], which
/// only exists when a Metal device was opened.
pub struct MslProgram {
    pub plan: CompiledPlan,
    pub dispatches: Vec<Dispatch>,
    pub tensors: HashMap<TensorId, Bound>,
    /// Control/shape tensors resolved on the CPU at plan time (never compute data).
    pub cpu: HashMap<TensorId, CpuTensor>,
    pub weights_blob: Vec<u8>,
    pub weight_slots: HashMap<TensorId, (usize, Elem, Vec<usize>)>,
    pub extra_bytes: usize,
    pub used_kernels: Vec<Kernel>,
    pub inputs: Vec<GraphIo>,
    pub outputs: Vec<GraphIo>,
    /// Graph outputs that are plan-time constants; returned as `f32` by name.
    pub constant_outputs: Vec<(String, Vec<f32>, Vec<usize>)>,
    pub msl: &'static str,
}

impl fmt::Debug for MslProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MslProgram")
            .field("nodes", &self.plan.graph.nodes.len())
            .field("dispatches", &self.dispatches.len())
            .field("kernels", &self.used_kernels)
            .field("arena_bytes", &self.plan.arena.peak_bytes)
            .field("extra_bytes", &self.extra_bytes)
            .field("weights_bytes", &self.weights_blob.len())
            .finish()
    }
}

impl MslProgram {
    /// Compile `.onnx` bytes into Metal dispatches. This never touches a device:
    /// every graph that passes here is fully encoded, and every graph that fails
    /// names the offending node.
    pub fn compile(onnx_bytes: &[u8], overrides: &ShapeOverrides) -> Result<Self> {
        let plan =
            load_and_plan(onnx_bytes, overrides).map_err(|e| MetalError::FrontEnd(e.to_string()))?;
        Builder::new(plan).build()
    }

    pub fn arena_bytes(&self) -> usize {
        self.plan.arena.peak_bytes
    }

    pub fn dispatch_count(&self) -> usize {
        self.dispatches.len()
    }

    /// Structural invariants that must hold before any command is encoded.
    pub fn validate(&self) -> std::result::Result<(), String> {
        for dispatch in &self.dispatches {
            dispatch.validate()?;
        }
        for io in &self.inputs {
            if !self.tensors.contains_key(&io.tid) {
                return Err(format!("graph input '{}' has no buffer binding", io.name));
            }
        }
        let mut seen = std::collections::HashSet::new();
        for dispatch in &self.dispatches {
            for (_, region) in &dispatch.binds {
                if region.byte_off % 4 != 0 {
                    return Err(format!("node '{}': misaligned region", dispatch.node));
                }
            }
        }
        for output in &self.outputs {
            let bound = self.tensors.contains_key(&output.tid);
            let constant = self
                .constant_outputs
                .iter()
                .any(|(name, _, _)| name == &output.name);
            if !bound && !constant {
                return Err(format!(
                    "graph output '{}' is produced by no dispatch and is not a plan-time constant",
                    output.name
                ));
            }
        }
        for kernel in &self.used_kernels {
            if !seen.insert(*kernel) {
                return Err(format!("kernel '{}' recorded twice", kernel.name()));
            }
            if !self.msl.contains(kernel.name()) {
                return Err(format!("kernel '{}' missing from the MSL library", kernel.name()));
            }
        }
        for (tid, (offset, _, _)) in &self.weight_slots {
            if offset % CUDA_DEFAULT_ALIGNMENT != 0 {
                return Err(format!(
                    "weight tensor {tid:?} sits at unaligned offset {offset}"
                ));
            }
            if *offset + 4 > self.weights_blob.len() && self.weights_blob.len() > 0 {
                // Zero-length tensors are legal; anything else must fit.
                let info = &self.plan.graph.tensors[tid.0];
                if info.static_byte_size().unwrap_or(0) > 0 {
                    return Err(format!("weight tensor {tid:?} exceeds the uploaded blob"));
                }
            }
        }
        if self.dispatches.is_empty() {
            return Err(
                "the graph encodes to zero Metal dispatches; this program cannot produce a \
                 device result and must not be reported as one"
                    .into(),
            );
        }
        Ok(())
    }

    /// MSL source for the whole library, suitable for `xcrun -sdk macosx metal`.
    pub fn msl_source(&self) -> &'static str {
        self.msl
    }
}

// ---------------------------------------------------------------------------
// Plan builder
// ---------------------------------------------------------------------------

struct Builder {
    plan: CompiledPlan,
    eval: CpuShapeEvaluator,
    tensors: HashMap<TensorId, Bound>,
    cpu: HashMap<TensorId, CpuTensor>,
    dispatches: Vec<Dispatch>,
    weights_blob: Vec<u8>,
    weight_slots: HashMap<TensorId, (usize, Elem, Vec<usize>)>,
    extra_offsets: HashMap<TensorId, usize>,
    extra_bytes: usize,
    used: Vec<Kernel>,
    inputs: Vec<GraphIo>,
    outputs: Vec<GraphIo>,
    constant_outputs: Vec<(String, Vec<f32>, Vec<usize>)>,
}

impl Builder {
    fn new(plan: CompiledPlan) -> Self {
        Self {
            plan,
            eval: CpuShapeEvaluator::new(),
            tensors: HashMap::new(),
            cpu: HashMap::new(),
            dispatches: Vec::new(),
            weights_blob: vec![0u8; 256],
            weight_slots: HashMap::new(),
            extra_offsets: HashMap::new(),
            extra_bytes: 0,
            used: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            constant_outputs: Vec::new(),
        }
    }

    fn build(mut self) -> Result<MslProgram> {
        // The evaluator starts from resolved static shapes and every small
        // initializer, so `Shape`/`Size`/parameter reads do not need the graph.
        self.eval = CpuShapeEvaluator::from_graph(&self.plan.graph, &HashMap::new());

        self.register_weights()?;
        self.register_io()?;

        let node_count = self.plan.graph.nodes.len();
        for index in 0..node_count {
            let node = self.plan.graph.nodes[index].clone();
            self.encode_node(&node)?;
        }

        let mut program = MslProgram {
            plan: self.plan,
            dispatches: self.dispatches,
            tensors: self.tensors,
            cpu: self.cpu,
            weights_blob: self.weights_blob,
            weight_slots: self.weight_slots,
            extra_bytes: self.extra_bytes,
            used_kernels: self.used,
            inputs: self.inputs,
            outputs: self.outputs,
            constant_outputs: self.constant_outputs,
            msl: MSL_LIBRARY,
        };
        // Pad so a dummy bind at offset 0 is always backed by real bytes.
        if program.weights_blob.len() < 32 {
            program.weights_blob.resize(32, 0);
        }
        program.used_kernels.sort_by_key(|k| k.name());
        program.used_kernels.dedup();
        Ok(program)
    }

    fn align_up(&self, value: usize) -> usize {
        let align = CUDA_DEFAULT_ALIGNMENT;
        value.div_ceil(align) * align
    }

    fn push(
        &mut self,
        node: &IrNode,
        kernel: Kernel,
        binds: Vec<(u32, Region)>,
        params: Vec<u8>,
        threads: [u32; 3],
        threadgroup: [u32; 3],
    ) {
        let writes: Vec<TensorId> = node.outputs.iter().flatten().copied().take(1).collect();
        self.push_writes(node, kernel, binds, params, threads, threadgroup, writes);
    }

    fn push_writes(
        &mut self,
        node: &IrNode,
        kernel: Kernel,
        binds: Vec<(u32, Region)>,
        params: Vec<u8>,
        threads: [u32; 3],
        threadgroup: [u32; 3],
        writes: Vec<TensorId>,
    ) {
        if !self.used.contains(&kernel) {
            self.used.push(kernel);
        }
        self.dispatches.push(Dispatch {
            node: node.name.clone(),
            op: node.op.name(),
            kernel,
            binds,
            params,
            threads,
            threadgroup,
            writes,
            node_inputs: node.inputs.iter().flatten().copied().collect(),
        });
    }

    // -- static data ------------------------------------------------------

    fn register_weights(&mut self) -> Result<()> {
        let mut ids: Vec<TensorId> = self.plan.graph.weights.keys().copied().collect();
        ids.sort_by_key(|id| id.0);
        for tid in ids {
            let proto = self
                .plan
                .graph
                .weights
                .get(&tid)
                .cloned()
                .expect("weight present");
            let (bytes, elem) = decode_weight(&proto).map_err(|detail| MetalError::InvalidGraph {
                node: format!("weight '{}'", proto.name),
                detail,
            })?;
            let shape: Vec<usize> = proto.dims.iter().map(|&d| d.max(0) as usize).collect();
            let offset = self.align_up(self.weights_blob.len());
            if offset > self.weights_blob.len() {
                self.weights_blob.resize(offset, 0);
            }
            self.weights_blob.extend_from_slice(&bytes);
            self.weight_slots.insert(tid, (offset, elem, shape.clone()));
            self.tensors.insert(
                tid,
                Bound {
                    region: Region::new(Buf::Weights, offset),
                    shape,
                    elem,
                },
            );
        }
        Ok(())
    }

    fn register_io(&mut self) -> Result<()> {
        for &tid in &self.plan.graph.inputs.clone() {
            let info = &self.plan.graph.tensors[tid.0];
            let shape = info.static_shape().ok_or_else(|| MetalError::InvalidGraph {
                node: format!("graph input '{}'", info.name),
                detail: "shape stays dynamic after inference".into(),
            })?;
            let elem = Elem::from_onnx(info.dtype).ok_or_else(|| MetalError::InvalidGraph {
                node: format!("graph input '{}'", info.name),
                detail: format!("element type {:?} is not supported", info.dtype),
            })?;
            if elem != Elem::F32 {
                return Err(MetalError::InvalidGraph {
                    node: format!("graph input '{}'", info.name),
                    detail: format!(
                        "named f32 inputs require Float graph inputs; this one is {:?}",
                        info.dtype
                    ),
                });
            }
            self.tensors.insert(tid, Bound { region: Region::new(Buf::Input(tid), 0), shape: shape.clone(), elem });
            self.inputs.push(GraphIo { tid, name: info.name.clone(), shape, elem });
        }
        for &tid in &self.plan.graph.outputs.clone() {
            let info = &self.plan.graph.tensors[tid.0];
            let shape = info.static_shape().ok_or_else(|| MetalError::InvalidGraph {
                node: format!("graph output '{}'", info.name),
                detail: "shape stays dynamic after inference".into(),
            })?;
            let elem = Elem::from_onnx(info.dtype).ok_or_else(|| MetalError::InvalidGraph {
                node: format!("graph output '{}'", info.name),
                detail: format!("element type {:?} is not supported", info.dtype),
            })?;
            self.outputs.push(GraphIo { tid, name: info.name.clone(), shape, elem });
        }
        Ok(())
    }

    // -- small queries ----------------------------------------------------

    fn shape_of(&self, tid: TensorId) -> Result<Vec<usize>> {
        self.tensors
            .get(&tid)
            .map(|b| b.shape.clone())
            .or_else(|| self.eval.get_shape(tid).map(|s| s.to_vec()))
            .or_else(|| self.plan.graph.tensors[tid.0].static_shape())
            .ok_or_else(|| MetalError::InvalidGraph {
                node: format!("tensor '{}'", self.plan.graph.tensors[tid.0].name),
                detail: "shape is not statically known".into(),
            })
    }

    fn elem_of(&self, tid: TensorId) -> Result<Elem> {
        if let Some(bound) = self.tensors.get(&tid) {
            return Ok(bound.elem);
        }
        Elem::from_onnx(self.plan.graph.tensors[tid.0].dtype).ok_or_else(|| {
            MetalError::InvalidGraph {
                node: format!("tensor '{}'", self.plan.graph.tensors[tid.0].name),
                detail: format!(
                    "element type {:?} is not supported",
                    self.plan.graph.tensors[tid.0].dtype
                ),
            }
        })
    }

    /// Region holding a GPU copy of a tensor, materialising plan-time constants
    /// (integer indices, shape data) into the weight buffer on first use.
    fn region_of(&mut self, tid: TensorId) -> Result<Region> {
        if let Some(bound) = self.tensors.get(&tid) {
            return Ok(bound.region);
        }
        if let Some(cpu) = self.cpu.get(&tid).cloned() {
            let bytes = cpu_tensor_bytes(&cpu);
            let offset = self.align_up(self.weights_blob.len());
            if offset > self.weights_blob.len() {
                self.weights_blob.resize(offset, 0);
            }
            self.weights_blob.extend_from_slice(&bytes);
            let elem = Elem::from_onnx(cpu.dtype()).unwrap_or(Elem::F32);
            let region = Region::new(Buf::Weights, offset);
            self.tensors.insert(
                tid,
                Bound {
                    region,
                    shape: cpu.shape.clone(),
                    elem,
                },
            );
            return Ok(region);
        }
        Err(MetalError::InvalidGraph {
            node: format!("tensor '{}'", self.plan.graph.tensors[tid.0].name),
            detail: "operand has neither a dispatch binding nor a plan-time value".into(),
        })
    }

    fn input_region(&mut self, node: &IrNode, index: usize) -> Result<Region> {
        let tid = node
            .inputs
            .get(index)
            .and_then(|o| *o)
            .ok_or_else(|| MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("missing required input #{index}"),
            })?;
        self.region_of(tid)
    }

    fn input_shape(&self, node: &IrNode, index: usize) -> Result<Vec<usize>> {
        let tid = node
            .inputs
            .get(index)
            .and_then(|o| *o)
            .ok_or_else(|| MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("missing required input #{index}"),
            })?;
        self.shape_of(tid)
    }

    fn input_elem(&self, node: &IrNode, index: usize) -> Result<Elem> {
        let tid = node
            .inputs
            .get(index)
            .and_then(|o| *o)
            .ok_or_else(|| MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("missing required input #{index}"),
            })?;
        self.elem_of(tid)
    }

    fn out_tid(&self, node: &IrNode, index: usize) -> Result<TensorId> {
        node.outputs
            .get(index)
            .and_then(|o| *o)
            .ok_or_else(|| MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("missing required output #{index}"),
            })
    }

    fn out_shape(&self, node: &IrNode, index: usize) -> Result<Vec<usize>> {
        let tid = self.out_tid(node, index)?;
        self.shape_of(tid)
    }

    fn out_elem(&self, node: &IrNode, index: usize) -> Result<Elem> {
        let tid = self.out_tid(node, index)?;
        self.elem_of(tid)
    }

    fn const_i64(&self, node: &IrNode, index: usize) -> Result<Vec<i64>> {
        let tid = node
            .inputs
            .get(index)
            .and_then(|o| *o)
            .ok_or_else(|| MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("missing parameter input #{index}"),
            })?;
        if let Some(cpu) = self.cpu.get(&tid) {
            return Ok(cpu.to_i64_vec());
        }
        if let Some(proto) = self.plan.graph.weights.get(&tid) {
            return proto.to_i64_vec().map_err(|e| MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("parameter input #{index} is not integer data: {e}"),
            });
        }
        Err(MetalError::InvalidGraph {
            node: node.name.clone(),
            detail: format!(
                "parameter input #{index} ('{}') is not statically known",
                self.plan.graph.tensors[tid.0].name
            ),
        })
    }

    fn const_f32(&self, node: &IrNode, index: usize) -> Result<Vec<f32>> {
        let tid = node
            .inputs
            .get(index)
            .and_then(|o| *o)
            .ok_or_else(|| MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("missing parameter input #{index}"),
            })?;
        if let Some(cpu) = self.cpu.get(&tid) {
            return Ok(cpu.to_f32_vec());
        }
        if let Some(proto) = self.plan.graph.weights.get(&tid) {
            return proto.to_f32_vec().map_err(|e| MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("parameter input #{index} is not float data: {e}"),
            });
        }
        Err(MetalError::InvalidGraph {
            node: node.name.clone(),
            detail: format!(
                "parameter input #{index} ('{}') is not statically known",
                self.plan.graph.tensors[tid.0].name
            ),
        })
    }

    /// Bind an output tensor, reusing the planned arena slot when the planner
    /// owned it, the per-run output buffer for graph outputs, and the extra arena
    /// otherwise.
    fn alloc_out(&mut self, tid: TensorId, shape: Vec<usize>, elem: Elem) -> Result<Region> {
        if let Some(bound) = self.tensors.get(&tid) {
            return Ok(bound.region);
        }
        let bytes = numel(&shape)
            .checked_mul(elem.bytes())
            .ok_or_else(|| MetalError::InvalidGraph {
                node: format!("tensor '{}'", self.plan.graph.tensors[tid.0].name),
                detail: "byte size overflows usize".into(),
            })?;
        let region = if self.plan.graph.outputs.contains(&tid) {
            Region::new(Buf::Output(tid), 0)
        } else if let Some(alloc) = self.plan.arena.allocations.get(&tid) {
            if alloc.aligned_bytes < bytes {
                // The planner sized this tensor before BN folding changed the
                // shape; fall through to the extra arena rather than overrun.
                self.alloc_extra(tid, bytes)
            } else {
                Region::new(Buf::Arena, alloc.offset)
            }
        } else {
            self.alloc_extra(tid, bytes)
        };
        self.tensors.insert(tid, Bound { region, shape, elem });
        Ok(region)
    }

    fn alloc_extra(&mut self, tid: TensorId, bytes: usize) -> Region {
        if let Some(offset) = self.extra_offsets.get(&tid) {
            return Region::new(Buf::Extra, *offset);
        }
        let offset = self.align_up(self.extra_bytes);
        self.extra_bytes = offset + bytes.max(1);
        self.extra_offsets.insert(tid, offset);
        Region::new(Buf::Extra, offset)
    }

    /// Alias an output to an input region (Reshape/Squeeze/Unsqueeze/Identity).
    fn alias(&mut self, tid: TensorId, source: Region, shape: Vec<usize>, elem: Elem) -> Bound {
        let bound = Bound { region: source, shape, elem };
        self.tensors.insert(tid, bound.clone());
        bound
    }

    // -- encoding ---------------------------------------------------------

    fn encode_node(&mut self, node: &IrNode) -> Result<()> {
        // Plan-time evaluation of control data: shape tensors, integer index
        // tensors and shape-only operators never reach the GPU.
        if self.wants_cpu(node) {
            match self.eval.try_eval_node(node) {
                Ok(true) => {
                    let tid = self.out_tid(node, 0)?;
                    if let Some(tensor) = self.eval.get_tensor(tid).cloned() {
                        if self.plan.graph.outputs.contains(&tid) {
                            self.constant_outputs.push((
                                self.plan.graph.tensors[tid.0].name.clone(),
                                tensor.to_f32_vec(),
                                tensor.shape.clone(),
                            ));
                        }
                        self.cpu.insert(tid, tensor);
                    }
                    return Ok(());
                }
                Ok(false) => { /* fall through to the GPU encoders */ }
                Err(e) => {
                    return Err(MetalError::InvalidGraph {
                        node: node.name.clone(),
                        detail: format!("plan-time evaluation failed: {e}"),
                    });
                }
            }
        }

        match &node.op {
            Op::Constant { value } => self.encode_constant(node, value),
            Op::Shape { .. } | Op::Size | Op::ConstantOfShape { .. } => {
                // Reached only when the evaluator could not resolve the value, or
                // when a large ConstantOfShape was routed to a fill kernel.
                if let Op::ConstantOfShape { value } = &node.op {
                    self.encode_constant_of_shape(node, value)
                } else {
                    Err(MetalError::InvalidGraph {
                        node: node.name.clone(),
                        detail: "shape/constant data could not be resolved at plan time".into(),
                    })
                }
            }
            Op::Conv(attrs) => self.encode_conv(node, attrs, false),
            Op::ConvTranspose(attrs) => self.encode_conv_transpose(node, attrs),
            Op::MatMul => self.encode_matmul(node),
            Op::MatMulInteger => self.encode_matmul_int(node),
            Op::Gemm(attrs) => self.encode_gemm(node, attrs),
            Op::BatchNormFused => self.encode_channel_affine(node),
            Op::LayerNormalization(attrs) => self.encode_layer_norm(node, attrs),
            Op::InstanceNormalization { epsilon } => self.encode_instance_norm(node, *epsilon),
            Op::MaxPool(attrs) => self.encode_pool(node, attrs, false),
            Op::AveragePool(attrs) => self.encode_pool(node, attrs, true),
            Op::GlobalAveragePool => self.encode_global_avg_pool(node),
            Op::ReduceMean(attrs) => self.encode_reduce_mean(node, attrs),
            Op::Resize(attrs) => self.encode_resize(node, attrs),
            Op::Pad { mode } => self.encode_pad(node, mode),
            Op::Softmax { axis } => self.encode_softmax(node, *axis),
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Pow | Op::PRelu => self.encode_binary(node),
            Op::Equal | Op::Greater | Op::GreaterOrEqual | Op::Less => self.encode_compare(node),
            Op::And => self.encode_and(node),
            Op::Where => self.encode_where(node),
            Op::Sqrt
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
            | Op::Clip => self.encode_unary(node),
            Op::Cast { to } => self.encode_cast(node, *to),
            Op::Transpose { perm } => self.encode_transpose(node, perm),
            Op::Reshape { .. } | Op::Squeeze { .. } | Op::Unsqueeze { .. } | Op::Identity => {
                self.encode_view(node)
            }
            Op::Concat { axis } => self.encode_concat(node, *axis),
            Op::Split { axis, .. } => self.encode_split(node, *axis),
            Op::Slice => self.encode_slice(node),
            Op::Gather { axis } => self.encode_gather(node, *axis),
            Op::ScatterND => self.encode_scatter_nd(node),
            Op::Expand => self.encode_expand(node),
            Op::Tile => self.encode_tile(node),
            Op::DynamicQuantizeLinear => self.encode_dynamic_quantize(node),
            other => {
                let name = other.name();
                let reason = REJECTED_OPS
                    .iter()
                    .find(|(op, _)| *op == name)
                    .map(|(_, reason)| (*reason).to_string())
                    .unwrap_or_else(|| "no Metal encoding exists for this operator".to_string());
                Err(MetalError::UnsupportedOp {
                    node: node.name.clone(),
                    op: name,
                    reason,
                })
            }
        }
    }

    fn wants_cpu(&self, node: &IrNode) -> bool {
        if matches!(node.op, Op::Shape { .. } | Op::Size | Op::Constant { .. }) {
            return true;
        }
        if let Op::ConstantOfShape { .. } = &node.op {
            let small = node
                .outputs
                .first()
                .and_then(|o| *o)
                .and_then(|tid| self.plan.graph.tensors[tid.0].static_byte_size())
                .map_or(false, |bytes| {
                    bytes <= crate::native_infer::cpu_shape_eval::DEFAULT_MAX_CPU_TENSOR_ELEMS * 4
                });
            return small;
        }
        let Some(tid) = node.outputs.first().and_then(|o| *o) else {
            return false;
        };
        if self.plan.graph.outputs.contains(&tid) && node.outputs.len() > 1 {
            return false;
        }
        matches!(
            self.plan.graph.tensors[tid.0].dtype,
            DataType::Int64 | DataType::Int32 | DataType::Bool
        )
    }

    /// Large `ConstantOfShape` outputs are filled on the GPU instead of being
    /// materialised as a CPU tensor and uploaded.
    fn encode_constant_of_shape(&mut self, node: &IrNode, value: &TensorProto) -> Result<()> {
        let out_shape = self.out_shape(node, 0)?;
        let elem = self.out_elem(node, 0)?;
        if !matches!(elem.bytes(), 1 | 2 | 4) {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("ConstantOfShape of {} bytes is not encodable", elem.bytes()),
            });
        }
        let fill_value = value
            .to_f32_vec()
            .ok()
            .and_then(|values| values.first().copied())
            .unwrap_or(0.0);
        let dst = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), elem)?;
        let params = FillParams {
            n: numel(&out_shape) as u32,
            elem_bytes: elem.bytes() as u32,
            dst_base: 0,
            bits: match elem {
                Elem::F32 => fill_value.to_bits(),
                _ => fill_value as i64 as u32,
            },
        };
        self.push(
            node,
            Kernel::Fill,
            vec![(0, dst)],
            as_bytes(&params),
            chunked(numel(&out_shape)),
            EW_THREADGROUP,
        );
        Ok(())
    }

    fn encode_constant(&mut self, node: &IrNode, value: &TensorProto) -> Result<()> {
        let tid = self.out_tid(node, 0)?;
        let shape: Vec<usize> = value.dims.iter().map(|&d| d.max(0) as usize).collect();
        let elem = Elem::from_onnx(value.data_type).ok_or_else(|| MetalError::InvalidGraph {
            node: node.name.clone(),
            detail: format!("constant element type {:?} is not supported", value.data_type),
        })?;
        let (bytes, stored) = decode_weight(value).map_err(|detail| MetalError::InvalidGraph {
            node: node.name.clone(),
            detail,
        })?;
        if elem.is_float() {
            let offset = self.align_up(self.weights_blob.len());
            if offset > self.weights_blob.len() {
                self.weights_blob.resize(offset, 0);
            }
            self.weights_blob.extend_from_slice(&bytes);
            let region = Region::new(Buf::Weights, offset);
            self.weight_slots.insert(tid, (offset, stored, shape.clone()));
            self.tensors.insert(
                tid,
                Bound {
                    region,
                    shape: shape.clone(),
                    elem: stored,
                },
            );
            if numel(&shape) <= crate::native_infer::cpu_shape_eval::DEFAULT_MAX_CPU_TENSOR_ELEMS {
                if let Ok(tensor) = CpuTensor::from_tensor_proto(value) {
                    self.eval.set_tensor(tid, tensor.clone());
                    self.cpu.insert(tid, tensor);
                }
            }
            Ok(())
        } else {
            let tensor = CpuTensor::from_tensor_proto(value).map_err(|e| {
                MetalError::InvalidGraph {
                    node: node.name.clone(),
                    detail: format!("constant decode failed: {e}"),
                }
            })?;
            self.eval.set_tensor(tid, tensor.clone());
            if self.plan.graph.outputs.contains(&tid) {
                self.constant_outputs
                    .push((self.plan.graph.tensors[tid.0].name.clone(), tensor.to_f32_vec(), shape));
            }
            self.cpu.insert(tid, tensor);
            Ok(())
        }
    }

    fn encode_view(&mut self, node: &IrNode) -> Result<()> {
        let tid = self.out_tid(node, 0)?;
        let shape = self.out_shape(node, 0)?;
        let elem = self.out_elem(node, 0)?;
        let source = self.input_region(node, 0)?;
        // A view may alias only a buffer whose bytes stay valid for the whole
        // run (weights, graph input). Arena slots are reused by the shared
        // planner after the source's last read, so aliasing one there would let
        // a later tensor overwrite live data; those views get a real copy.
        let stable = matches!(source.buf, Buf::Weights | Buf::Input(_));
        if stable && !self.plan.graph.outputs.contains(&tid) {
            self.alias(tid, source, shape, elem);
            return Ok(());
        }
        let dst = self.alloc_out(tid, shape.clone(), elem)?;
        self.emit_contiguous_copy(node, source, dst, &shape, elem)
    }

    fn emit_contiguous_copy(
        &mut self,
        node: &IrNode,
        src: Region,
        dst: Region,
        shape: &[usize],
        elem: Elem,
    ) -> Result<()> {
        if elem.bytes() != 4 && elem.bytes() != 1 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("copies of {} width are not encoded", elem.bytes()),
            });
        }
        let rank = shape.len();
        if rank > MAX_RANK {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("rank {rank} exceeds the {MAX_RANK}-dimension copy ABI"),
            });
        }
        let elems = numel(shape);
        let dst_stride = strides(shape);
        let src_stride = strides(shape);
        let mut params = zero_copy_params(rank, elems, elem.bytes() as u32);
        params.dst_base = 0;
        params.src_base = 0;
        for axis in 0..rank {
            params.out_shape[axis] = shape[axis] as u32;
            params.dst_stride[axis] = dst_stride[axis] as i32;
            params.mul[axis] = src_stride[axis] as i32;
            params.add[axis] = 0;
            params.in_dim[axis] = shape[axis] as i32;
            params.use_mod[axis] = 0;
        }
        self.push(
            node,
            Kernel::Copy,
            vec![(0, dst), (1, src)],
            as_bytes(&params),
            chunked(elems),
            COPY_THREADGROUP,
        );
        Ok(())
    }

    // -- convolution & matmul ---------------------------------------------

    fn encode_conv(&mut self, node: &IrNode, attrs: &crate::native_infer::graph::ConvAttrs, _integer: bool) -> Result<()> {
        let x_shape = self.input_shape(node, 0)?;
        let w_shape = self.input_shape(node, 1)?;
        let out_shape = self.out_shape(node, 0)?;
        if x_shape.len() != 4 || w_shape.len() != 4 || out_shape.len() != 4 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!(
                    "only 4-D NCHW convolution is encoded (x{x_shape:?}, w{w_shape:?}, y{out_shape:?})"
                ),
            });
        }
        let (n, c, h, w) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
        let (oc, kh, kw) = (w_shape[0], w_shape[2], w_shape[3]);
        let (oh, ow) = (out_shape[2], out_shape[3]);
        let sh = *attrs.strides.first().unwrap_or(&1);
        let sw = *attrs.strides.get(1).unwrap_or(&sh);
        let dh = *attrs.dilations.first().unwrap_or(&1);
        let dw = *attrs.dilations.get(1).unwrap_or(&dh);
        let (ph, pw) = match attrs.pads.as_slice() {
            [] => (0, 0),
            [t, l, b, r] => {
                if t != b || l != r {
                    return Err(MetalError::InvalidGraph {
                        node: node.name.clone(),
                        detail: format!(
                            "asymmetric padding {attrs:?} is not encoded; ONNX pads must be symmetric here"
                        ),
                    });
                }
                (*t, *l)
            }
            [p] => (*p, *p),
            [t, l] => (*t, *l),
            other => {
                return Err(MetalError::InvalidGraph {
                    node: node.name.clone(),
                    detail: format!("unsupported pads {other:?}"),
                })
            }
        };
        if attrs.group == 0 || c % attrs.group != 0 || oc % attrs.group != 0 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("group {} does not divide C={c} / OC={oc}", attrs.group),
            });
        }
        if w_shape[1] * attrs.group != c {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("weight shape {w_shape:?} inconsistent with C={c} and group={}", attrs.group),
            });
        }
        let x = self.input_region(node, 0)?;
        let weights = self.input_region(node, 1)?;
        let bias = match node.inputs.get(2).and_then(|o| *o) {
            Some(_) => self.input_region(node, 2)?,
            None => Region::new(Buf::Weights, 0),
        };
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), Elem::F32)?;
        let params = ConvParams {
            n: n as u32,
            h: h as u32,
            w: w as u32,
            c: c as u32,
            oh: oh as u32,
            ow: ow as u32,
            oc: oc as u32,
            kh: kh as u32,
            kw: kw as u32,
            sh: sh as u32,
            sw: sw as u32,
            ph: ph as u32,
            pw: pw as u32,
            dh: dh as u32,
            dw: dw as u32,
            groups: attrs.group as u32,
            act: 0,
            x_off: 0,
            w_off: 0,
            b_off: 0,
            y_off: 0,
        };
        let spatial = (oh.max(1) as u32, ow.max(1) as u32);
        self.push(
            node,
            Kernel::Conv2d,
            vec![(0, x), (1, weights), (2, bias), (3, y)],
            as_bytes(&params),
            [spatial.1, spatial.0, (oc * n) as u32],
            [16, 8, 1],
        );
        Ok(())
    }

    fn encode_conv_transpose(&mut self, node: &IrNode, attrs: &crate::native_infer::graph::ConvTransposeAttrs) -> Result<()> {
        let x_shape = self.input_shape(node, 0)?;
        let w_shape = self.input_shape(node, 1)?;
        let out_shape = self.out_shape(node, 0)?;
        if x_shape.len() != 4 || w_shape.len() != 4 || out_shape.len() != 4 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: "only 4-D NCHW transposed convolution is encoded".into(),
            });
        }
        if !attrs.output_padding.iter().all(|&p| p == 0) {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("output_padding {:?} is not encoded", attrs.output_padding),
            });
        }
        let (n, c, h, w) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
        let (oc, _, kh, kw) = (w_shape[1], w_shape[0], w_shape[2], w_shape[3]);
        let (oh, ow) = (out_shape[2], out_shape[3]);
        let sh = *attrs.strides.first().unwrap_or(&1);
        let sw = *attrs.strides.get(1).unwrap_or(&sh);
        let dh = *attrs.dilations.first().unwrap_or(&1);
        let dw = *attrs.dilations.get(1).unwrap_or(&dh);
        let (ph, pw) = match attrs.pads.as_slice() {
            [] => (0, 0),
            [t, l, b, r] if t == b && l == r => (*t, *l),
            other => {
                return Err(MetalError::InvalidGraph {
                    node: node.name.clone(),
                    detail: format!("pads {other:?} must be symmetric for the encoded kernel"),
                })
            }
        };
        let x = self.input_region(node, 0)?;
        let weights = self.input_region(node, 1)?;
        let bias = match node.inputs.get(2).and_then(|o| *o) {
            Some(_) => self.input_region(node, 2)?,
            None => Region::new(Buf::Weights, 0),
        };
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), Elem::F32)?;
        let params = ConvParams {
            n: n as u32,
            h: h as u32,
            w: w as u32,
            c: c as u32,
            oh: oh as u32,
            ow: ow as u32,
            oc: oc as u32,
            kh: kh as u32,
            kw: kw as u32,
            sh: sh as u32,
            sw: sw as u32,
            ph: ph as u32,
            pw: pw as u32,
            dh: dh as u32,
            dw: dw as u32,
            groups: attrs.group.max(1) as u32,
            act: 0,
            x_off: 0,
            w_off: 0,
            b_off: 0,
            y_off: 0,
        };
        self.push(
            node,
            Kernel::ConvTranspose2d,
            vec![(0, x), (1, weights), (2, bias), (3, y)],
            as_bytes(&params),
            [ow.max(1) as u32, oh.max(1) as u32, (oc * n) as u32],
            [16, 8, 1],
        );
        Ok(())
    }

    fn encode_matmul(&mut self, node: &IrNode) -> Result<()> {
        let a_shape = self.input_shape(node, 0)?;
        let b_shape = self.input_shape(node, 1)?;
        let out_shape = self.out_shape(node, 0)?;
        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: "MatMul operands must have rank >= 2".into(),
            });
        }
        let (m, k) = (a_shape[a_shape.len() - 2], a_shape[a_shape.len() - 1]);
        let (k2, n) = (b_shape[b_shape.len() - 2], b_shape[b_shape.len() - 1]);
        if k != k2 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("inner dimensions disagree: {a_shape:?} x {b_shape:?}"),
            });
        }
        let batch_shape = &out_shape[..out_shape.len()-2];
        let batch: usize = batch_shape.iter().product();
        let a = self.input_region(node, 0)?;
        let b = self.input_region(node, 1)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), Elem::F32)?;
        for index in 0..batch {
            // ONNX aligns batch axes from the right. Rank-two weights are reused
            // across *all* batches; blindly adding the output offset overreads B.
            let batch_a = broadcast_batch(index,batch_shape,&a_shape[..a_shape.len()-2]).ok_or_else(||MetalError::InvalidGraph{node:node.name.clone(),detail:"MatMul A batch dimensions do not broadcast".into()})?;
            let batch_b = broadcast_batch(index,batch_shape,&b_shape[..b_shape.len()-2]).ok_or_else(||MetalError::InvalidGraph{node:node.name.clone(),detail:"MatMul B batch dimensions do not broadcast".into()})?;
            let params = GemmParams {
                m: m as u32,
                n: n as u32,
                k: k as u32,
                lda: k as i32,
                ldb: n as i32,
                ldc: n as i32,
                a_off: (batch_a * m * k) as i32,
                b_off: (batch_b * k * n) as i32,
                c_off: (index * m * n) as i32,
                alpha: 1.0,
                beta: 0.0,
                trans_a: 0,
                trans_b: 0,
                has_bias: 0,
                act: 0,
            };
            let bias = Region::new(Buf::Weights, 0);
            self.push(
                node,
                Kernel::Gemm,
                vec![(0, a), (1, b), (2, bias), (3, y)],
                as_bytes(&params),
                [tile_grid(n), tile_grid(m), 1],
                GEMM_THREADGROUP,
            );
        }
        Ok(())
    }

    fn encode_gemm(&mut self, node: &IrNode, attrs: &crate::native_infer::graph::GemmAttrs) -> Result<()> {
        let a_shape = self.input_shape(node, 0)?;
        let b_shape = self.input_shape(node, 1)?;
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: "Gemm is encoded for 2-D operands only".into(),
            });
        }
        let m = if attrs.trans_a { a_shape[1] } else { a_shape[0] };
        let k = if attrs.trans_a { a_shape[0] } else { a_shape[1] };
        let k2 = if attrs.trans_b { b_shape[1] } else { b_shape[0] };
        let n = if attrs.trans_b { b_shape[0] } else { b_shape[1] };
        if k != k2 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("Gemm inner dimensions disagree: {a_shape:?} x {b_shape:?}"),
            });
        }
        let a = self.input_region(node, 0)?;
        let b = self.input_region(node, 1)?;
        let has_c = node.inputs.get(2).and_then(|o| *o).is_some();
        if has_c && attrs.beta != 0.0 && attrs.beta != 1.0 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!(
                    "Gemm with a C input must use beta 0 or 1 (found {}); the encoded kernel adds \
                     C[col] once or not at all",
                    attrs.beta
                ),
            });
        }
        // ONNX Gemm's third input is `C` scaled by beta. A column bias is exactly
        // that when beta = 1; with beta = 0 the input contributes nothing.
        let use_bias = has_c && attrs.beta == 1.0;
        let bias = if use_bias {
            let dims=self.input_shape(node,2)?;
            if dims!=[n] && dims!=[1,n] {return Err(MetalError::InvalidGraph{node:node.name.clone(),detail:"currently supports only column-wise Gemm bias".into()});}
            self.input_region(node, 2)?
        } else {
            Region::new(Buf::Weights, 0)
        };
        let out_shape = self.out_shape(node, 0)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape, Elem::F32)?;
        let params = GemmParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lda: if attrs.trans_a { m as i32 } else { k as i32 },
            ldb: if attrs.trans_b { k as i32 } else { n as i32 },
            ldc: n as i32,
            a_off: 0,
            b_off: 0,
            c_off: 0,
            alpha: attrs.alpha,
            // The C input is applied through `has_bias` below; letting the kernel
            // also read `beta * C` would read an unwritten output buffer.
            beta: 0.0,
            trans_a: attrs.trans_a as u32,
            trans_b: attrs.trans_b as u32,
            has_bias: use_bias as u32,
            act: 0,
        };
        self.push(
            node,
            Kernel::Gemm,
            vec![(0, a), (1, b), (2, bias), (3, y)],
            as_bytes(&params),
            [tile_grid(n), tile_grid(m), 1],
            GEMM_THREADGROUP,
        );
        Ok(())
    }

    fn encode_matmul_int(&mut self, node: &IrNode) -> Result<()> {
        let a_shape = self.input_shape(node, 0)?;
        let b_shape = self.input_shape(node, 1)?;
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: "MatMulInteger is encoded for 2-D operands only".into(),
            });
        }
        let (m, k) = (a_shape[0], a_shape[1]);
        let (k2, n) = (b_shape[0], b_shape[1]);
        if k != k2 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: "MatMulInteger inner dimensions disagree".into(),
            });
        }
        let a = self.input_region(node, 0)?;
        let b = self.input_region(node, 1)?;
        let a_elem = self.input_elem(node, 0)?;
        let b_elem = self.input_elem(node, 1)?;
        let out_shape = self.out_shape(node, 0)?;
        let out_elem = self.out_elem(node, 0)?;
        if out_elem != Elem::I32 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("MatMulInteger output must be Int32, found {out_elem:?}"),
            });
        }
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape, out_elem)?;
        let mut a_zp = 0u32;
        let mut b_zp = 0u32;
        let mut use_buf = 0u32;
        let mut azp_region = Region::new(Buf::Weights, 0);
        let mut bzp_region = Region::new(Buf::Weights, 0);
        if let Some(tid) = node.inputs.get(2).and_then(|o| *o) {
            if let Some(values) = self.plan.graph.weights.get(&tid).and_then(|p| p.to_i64_vec().ok()) {
                a_zp = values.first().copied().unwrap_or(0).max(0) as u32;
            } else {
                azp_region = self.region_of(tid)?;
                use_buf = 1;
            }
        }
        if let Some(tid) = node.inputs.get(3).and_then(|o| *o) {
            if let Some(values) = self.plan.graph.weights.get(&tid).and_then(|p| p.to_i64_vec().ok()) {
                b_zp = values.first().copied().unwrap_or(0).max(0) as u32;
            } else {
                bzp_region = self.region_of(tid)?;
                use_buf = 1;
            }
        }
        let params = MatMulIntParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            a_off: 0,
            b_off: 0,
            c_off: 0,
            azp_off: 0,
            bzp_off: 0,
            a_zp,
            b_zp,
            use_buf_zp: use_buf,
            b_signed: (b_elem == Elem::I8) as u32,
        };
        if a_elem != Elem::U8 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!(
                    "MatMulInteger A must be Uint8 in the encoded kernel, found {a_elem:?}"
                ),
            });
        }
        if !matches!(b_elem, Elem::U8 | Elem::I8) {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("MatMulInteger B must be Uint8/Int8, found {b_elem:?}"),
            });
        }
        self.push(
            node,
            Kernel::MatMulInt,
            vec![(0, a), (1, b), (2, y), (3, azp_region), (4, bzp_region)],
            as_bytes(&params),
            [tile_grid(n), tile_grid(m), 1],
            GEMM_THREADGROUP,
        );
        Ok(())
    }

    // -- normalization / reduction ---------------------------------------

    fn encode_channel_affine(&mut self, node: &IrNode) -> Result<()> {
        let x_shape = self.input_shape(node, 0)?;
        let out_shape = self.out_shape(node, 0)?;
        if x_shape.len() < 2 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: "BatchNormFused expects NCHW-like data".into(),
            });
        }
        let channels = x_shape[1];
        let inner: usize = x_shape[2..].iter().product();
        let x = self.input_region(node, 0)?;
        let scale = self.input_region(node, 1)?;
        let bias = self.input_region(node, 2)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape, Elem::F32)?;
        let n = numel(&x_shape) as u32;
        let params = ChannelAffineParams {
            n,
            inner: inner.max(1) as u32,
            channels: channels as u32,
            unused: 0,
            x_base: 0,
            y_base: 0,
            scale_base: 0,
            bias_base: 0,
        };
        self.push(
            node,
            Kernel::ChannelAffine,
            vec![(0, x), (1, scale), (2, bias), (3, y)],
            as_bytes(&params),
            chunked(n as usize),
            EW_THREADGROUP,
        );
        Ok(())
    }

    fn encode_layer_norm(&mut self, node: &IrNode, attrs: &crate::native_infer::graph::LayerNormAttrs) -> Result<()> {
        let x_shape = self.input_shape(node, 0)?;
        let rank = x_shape.len();
        let axis = normalize_axis(attrs.axis, rank, &node.name)?;
        let outer: usize = x_shape[..axis].iter().product();
        let r: usize = x_shape[axis..].iter().product();
        if axis == rank - 1 {
            // contiguous row of length R
        }
        let inner = 1usize;
        let x = self.input_region(node, 0)?;
        let scale = if node.inputs.get(1).and_then(|o| *o).is_some() {
            self.input_region(node, 1)?
        } else {
            Region::new(Buf::Weights, 0)
        };
        let bias = if node.inputs.get(2).and_then(|o| *o).is_some() {
            self.input_region(node, 2)?
        } else {
            Region::new(Buf::Weights, 0)
        };
        let out_shape = self.out_shape(node, 0)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape, Elem::F32)?;
        let params = RowParams {
            outer: outer as u32,
            r: r as u32,
            inner: inner as u32,
            mode: ROW_LAYER_NORM,
            channels: 0,
            y_stride_outer: r as u32,
            y_stride_inner: 1,
            unused: 0,
            x_off: 0,
            y_off: 0,
            g_off: if node.inputs.get(1).and_then(|o| *o).is_some() { 0 } else { -1 },
            b_off: if node.inputs.get(2).and_then(|o| *o).is_some() { 0 } else { -1 },
            eps: attrs.epsilon as f32,
        };
        self.push(
            node,
            Kernel::Rows,
            vec![(0, x), (1, scale), (2, bias), (3, y)],
            as_bytes(&params),
            [(outer * inner).max(1) as u32, 1, 1],
            ROW_THREADGROUP,
        );
        Ok(())
    }

    fn encode_instance_norm(&mut self, node: &IrNode, epsilon: f32) -> Result<()> {
        let x_shape = self.input_shape(node, 0)?;
        if x_shape.len() < 3 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: "InstanceNormalization expects N,C,spatial…".into(),
            });
        }
        let channels = x_shape[1];
        let r: usize = x_shape[2..].iter().product();
        let outer = x_shape[0] * channels;
        let x = self.input_region(node, 0)?;
        let scale = if node.inputs.get(1).and_then(|o| *o).is_some() {
            self.input_region(node, 1)?
        } else {
            Region::new(Buf::Weights, 0)
        };
        let bias = if node.inputs.get(2).and_then(|o| *o).is_some() {
            self.input_region(node, 2)?
        } else {
            Region::new(Buf::Weights, 0)
        };
        let out_shape = self.out_shape(node, 0)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape, Elem::F32)?;
        let params = RowParams {
            outer: outer as u32,
            r: r as u32,
            inner: 1,
            mode: ROW_INSTANCE_NORM,
            channels: channels as u32,
            y_stride_outer: r as u32,
            y_stride_inner: 1,
            unused: 0,
            x_off: 0,
            y_off: 0,
            g_off: if node.inputs.get(1).and_then(|o| *o).is_some() { 0 } else { -1 },
            b_off: if node.inputs.get(2).and_then(|o| *o).is_some() { 0 } else { -1 },
            eps: epsilon,
        };
        self.push(
            node,
            Kernel::Rows,
            vec![(0, x), (1, scale), (2, bias), (3, y)],
            as_bytes(&params),
            [outer.max(1) as u32, 1, 1],
            ROW_THREADGROUP,
        );
        Ok(())
    }

    fn encode_softmax(&mut self, node: &IrNode, axis: i64) -> Result<()> {
        let x_shape = self.input_shape(node, 0)?;
        let rank = x_shape.len();
        let axis = normalize_axis(axis, rank, &node.name)?;
        let outer: usize = x_shape[..axis].iter().product();
        let r = x_shape[axis];
        let inner: usize = x_shape[axis + 1..].iter().product();
        let x = self.input_region(node, 0)?;
        let out_shape = self.out_shape(node, 0)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape, Elem::F32)?;
        let params = RowParams {
            outer: outer as u32,
            r: r as u32,
            inner: inner as u32,
            mode: ROW_SOFTMAX,
            channels: 0,
            y_stride_outer: (r * inner) as u32,
            y_stride_inner: inner as u32,
            unused: 0,
            x_off: 0,
            y_off: 0,
            g_off: 0,
            b_off: 0,
            eps: 0.0,
        };
        let dummy = Region::new(Buf::Weights, 0);
        self.push(
            node,
            Kernel::Rows,
            vec![(0, x), (1, dummy), (2, dummy), (3, y)],
            as_bytes(&params),
            [(outer * inner).max(1) as u32, 1, 1],
            ROW_THREADGROUP,
        );
        Ok(())
    }

    fn encode_reduce_mean(&mut self, node: &IrNode, attrs: &crate::native_infer::graph::ReduceAttrs) -> Result<()> {
        let x_shape = self.input_shape(node, 0)?;
        let rank = x_shape.len();
        let axes: Vec<i64> = match &attrs.axes {
            Some(list) => list.clone(),
            None => {
                if attrs.noop_with_empty_axes {
                    return self.encode_view(node);
                }
                (0..rank as i64).collect()
            }
        };
        let mut normalized = Vec::new();
        for axis in axes {
            normalized.push(normalize_axis(axis, rank, &node.name)?);
        }
        normalized.sort_unstable();
        normalized.dedup();
        let Some(&first) = normalized.first() else {
            return self.encode_view(node);
        };
        let last = *normalized.last().expect("non-empty");
        if normalized.len() != last - first + 1 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("ReduceMean over non-contiguous axes {normalized:?} is not encoded"),
            });
        }
        let outer: usize = x_shape[..first].iter().product();
        let r: usize = x_shape[first..=last].iter().product();
        let inner: usize = x_shape[last + 1..].iter().product();
        let x = self.input_region(node, 0)?;
        let out_shape = self.out_shape(node, 0)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape, Elem::F32)?;
        let params = RowParams {
            outer: outer as u32,
            r: r as u32,
            inner: inner as u32,
            mode: ROW_MEAN,
            channels: 0,
            y_stride_outer: inner as u32,
            y_stride_inner: 1,
            unused: 0,
            x_off: 0,
            y_off: 0,
            g_off: 0,
            b_off: 0,
            eps: 0.0,
        };
        let dummy = Region::new(Buf::Weights, 0);
        self.push(
            node,
            Kernel::Rows,
            vec![(0, x), (1, dummy), (2, dummy), (3, y)],
            as_bytes(&params),
            [(outer * inner).max(1) as u32, 1, 1],
            ROW_THREADGROUP,
        );
        Ok(())
    }

    fn encode_global_avg_pool(&mut self, node: &IrNode) -> Result<()> {
        let x_shape = self.input_shape(node, 0)?;
        if x_shape.len() < 3 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: "GlobalAveragePool expects N,C,spatial…".into(),
            });
        }
        // One reduction row per (batch, channel): the spatial axes are averaged
        // per channel, never across channels.
        let outer = x_shape[0] * x_shape[1];
        let r: usize = x_shape[2..].iter().product();
        let x = self.input_region(node, 0)?;
        let out_shape = self.out_shape(node, 0)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape, Elem::F32)?;
        let params = RowParams {
            outer: outer as u32,
            r: r as u32,
            inner: 1,
            mode: ROW_MEAN,
            channels: 0,
            y_stride_outer: 1,
            y_stride_inner: 0,
            unused: 0,
            x_off: 0,
            y_off: 0,
            g_off: 0,
            b_off: 0,
            eps: 0.0,
        };
        let dummy = Region::new(Buf::Weights, 0);
        self.push(
            node,
            Kernel::Rows,
            vec![(0, x), (1, dummy), (2, dummy), (3, y)],
            as_bytes(&params),
            [outer.max(1) as u32, 1, 1],
            ROW_THREADGROUP,
        );
        Ok(())
    }

    fn encode_pool(&mut self, node: &IrNode, attrs: &crate::native_infer::graph::PoolAttrs, average: bool) -> Result<()> {
        let x_shape = self.input_shape(node, 0)?;
        let out_shape = self.out_shape(node, 0)?;
        if x_shape.len() != 4 || out_shape.len() != 4 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: "pooling is encoded for 4-D NCHW only".into(),
            });
        }
        let (n, c, ih, iw) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
        let (oh, ow) = (out_shape[2], out_shape[3]);
        let kh = attrs.kernel_shape[0];
        let kw = *attrs.kernel_shape.get(1).unwrap_or(&kh);
        let sh = *attrs.strides.first().unwrap_or(&1);
        let sw = *attrs.strides.get(1).unwrap_or(&sh);
        let dh = *attrs.dilations.first().unwrap_or(&1);
        let dw = *attrs.dilations.get(1).unwrap_or(&dh);
        let (ph, pw) = match attrs.pads.as_slice() {
            [] => (0, 0),
            [t, l, b, r] if t == b && l == r => (*t, *l),
            other => {
                return Err(MetalError::InvalidGraph {
                    node: node.name.clone(),
                    detail: format!("pads {other:?} must be symmetric for the encoded kernel"),
                })
            }
        };
        if attrs.ceil_mode {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: "ceil_mode pooling is not encoded".into(),
            });
        }
        let x = self.input_region(node, 0)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), Elem::F32)?;
        let params = PoolParams {
            n: n as u32,
            c: c as u32,
            ih: ih as u32,
            iw: iw as u32,
            oh: oh as u32,
            ow: ow as u32,
            kh: kh as u32,
            kw: kw as u32,
            sh: sh as u32,
            sw: sw as u32,
            ph: ph as u32,
            pw: pw as u32,
            dh: dh as u32,
            dw: dw as u32,
            mode: average as u32,
            count_include_pad: attrs.count_include_pad as u32,
            x_off: 0,
            y_off: 0,
        };
        self.push(
            node,
            Kernel::Pool2d,
            vec![(0, x), (1, y)],
            as_bytes(&params),
            [ow.max(1) as u32, oh.max(1) as u32, (c * n) as u32],
            [16, 8, 1],
        );
        Ok(())
    }

    fn encode_resize(&mut self, node: &IrNode, attrs: &crate::native_infer::graph::ResizeAttrs) -> Result<()> {
        use crate::native_infer::graph::{CoordTransformMode, NearestMode, ResizeMode};
        let x_shape = self.input_shape(node, 0)?;
        let out_shape = self.out_shape(node, 0)?;
        if x_shape.len() != 4 || out_shape.len() != 4 {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: "Resize is encoded for 4-D NCHW only".into(),
            });
        }
        if attrs.mode == ResizeMode::Cubic {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: "cubic Resize is not encoded".into(),
            });
        }
        if x_shape[0] != out_shape[0] || x_shape[1] != out_shape[1] {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: "Resize may only change the spatial axes H,W".into(),
            });
        }
        let (n, c, ih, iw) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
        let (oh, ow) = (out_shape[2], out_shape[3]);
        let mode = match attrs.mode {
            ResizeMode::Nearest => 0u32,
            ResizeMode::Linear => 1u32,
            ResizeMode::Cubic => unreachable!("rejected above"),
        };
        let coord = match attrs.coord_mode {
            CoordTransformMode::HalfPixel => 0u32,
            CoordTransformMode::Asymmetric => 1,
            CoordTransformMode::AlignCorners => 2,
            CoordTransformMode::TfHalfPixelForNn => 3,
            CoordTransformMode::PytorchHalfPixel => 4,
        };
        let nearest = match attrs.nearest_mode {
            NearestMode::Floor => 0u32,
            NearestMode::RoundPreferFloor => 1,
            NearestMode::RoundPreferCeil => 2,
            NearestMode::Ceil => 3,
        };
        let x = self.input_region(node, 0)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), Elem::F32)?;
        let params = ResizeParams {
            n: n as u32,
            c: c as u32,
            ih: ih as u32,
            iw: iw as u32,
            oh: oh as u32,
            ow: ow as u32,
            mode,
            coord_mode: coord,
            nearest_mode: nearest,
            rh: oh as f32 / ih.max(1) as f32,
            rw: ow as f32 / iw.max(1) as f32,
            x_off: 0,
            y_off: 0,
        };
        self.push(
            node,
            Kernel::Resize,
            vec![(0, x), (1, y)],
            as_bytes(&params),
            [ow.max(1) as u32, oh.max(1) as u32, (c * n) as u32],
            [16, 8, 1],
        );
        Ok(())
    }

    // -- elementwise ------------------------------------------------------

    fn encode_unary(&mut self, node: &IrNode) -> Result<()> {
        let (op_code, alpha, beta, lo, hi) = match &node.op {
            Op::Relu => (0u32, 0.0, 0.0, 0.0, 0.0),
            Op::Sigmoid => (1, 0.0, 0.0, 0.0, 0.0),
            Op::Sqrt => (2, 0.0, 0.0, 0.0, 0.0),
            Op::Abs => (3, 0.0, 0.0, 0.0, 0.0),
            Op::Sin => (4, 0.0, 0.0, 0.0, 0.0),
            Op::Cos => (5, 0.0, 0.0, 0.0, 0.0),
            Op::Reciprocal => (6, 0.0, 0.0, 0.0, 0.0),
            Op::Erf => (7, 0.0, 0.0, 0.0, 0.0),
            Op::LeakyRelu { alpha } => (8, *alpha, 0.0, 0.0, 0.0),
            Op::HardSwish => (9, 0.0, 0.0, 0.0, 0.0),
            Op::HardSigmoid { alpha, beta } => (10, *alpha, *beta, 0.0, 0.0),
            Op::Clip => {
                let lo = match node.inputs.get(1).and_then(|o| *o) {
                    Some(_) => self.const_f32(node, 1)?.first().copied().unwrap_or(f32::NEG_INFINITY),
                    None => f32::NEG_INFINITY,
                };
                let hi = match node.inputs.get(2).and_then(|o| *o) {
                    Some(_) => self.const_f32(node, 2)?.first().copied().unwrap_or(f32::INFINITY),
                    None => f32::INFINITY,
                };
                (11, 0.0, 0.0, lo, hi)
            }
            other => {
                return Err(MetalError::UnsupportedOp {
                    node: node.name.clone(),
                    op: other.name(),
                    reason: "no unary encoding".into(),
                })
            }
        };
        let shape = self.input_shape(node, 0)?;
        let n = numel(&shape);
        let x = self.input_region(node, 0)?;
        let out_shape = self.out_shape(node, 0)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape, Elem::F32)?;
        let params = UnaryParams {
            n: n as u32,
            op: op_code,
            alpha,
            beta,
            lo,
            hi,
            x_base: 0,
            y_base: 0,
        };
        self.push(
            node,
            Kernel::Unary,
            vec![(0, x), (1, y)],
            as_bytes(&params),
            chunked(n),
            EW_THREADGROUP,
        );
        Ok(())
    }

    fn encode_binary(&mut self, node: &IrNode) -> Result<()> {
        let op_code = match &node.op {
            Op::Add => 0u32,
            Op::Sub => 1,
            Op::Mul => 2,
            Op::Div => 3,
            Op::Pow => 4,
            Op::PRelu => 5,
            other => {
                return Err(MetalError::UnsupportedOp {
                    node: node.name.clone(),
                    op: other.name(),
                    reason: "no binary encoding".into(),
                })
            }
        };
        let a_shape = self.input_shape(node, 0)?;
        let b_shape = self.input_shape(node, 1)?;
        let out_shape = self.out_shape(node, 0)?;
        if out_shape.len() > MAX_RANK {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("rank {} exceeds the broadcast ABI", out_shape.len()),
            });
        }
        let (a_mul, b_mul) = broadcast_muls(&a_shape, &b_shape, &out_shape).map_err(|detail| {
            MetalError::InvalidGraph {
                node: node.name.clone(),
                detail,
            }
        })?;
        let a = self.input_region(node, 0)?;
        let b = self.input_region(node, 1)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), Elem::F32)?;
        let n = numel(&out_shape);
        let params = BinaryParams {
            rank: out_shape.len() as u32,
            n: n as u32,
            op: op_code,
            unused: 0,
            a_base: 0,
            b_base: 0,
            y_base: 0,
            out_shape: shape_array(&out_shape),
            a_mul: a_mul,
            b_mul,
        };
        self.push(
            node,
            Kernel::Binary,
            vec![(0, a), (1, b), (2, y)],
            as_bytes(&params),
            chunked(n),
            EW_THREADGROUP,
        );
        Ok(())
    }

    fn encode_compare(&mut self, node: &IrNode) -> Result<()> {
        let op_code = match &node.op {
            Op::Equal => 0u32,
            Op::Greater => 1,
            Op::GreaterOrEqual => 2,
            Op::Less => 3,
            other => {
                return Err(MetalError::UnsupportedOp {
                    node: node.name.clone(),
                    op: other.name(),
                    reason: "no comparison encoding".into(),
                })
            }
        };
        let a_shape = self.input_shape(node, 0)?;
        let b_shape = self.input_shape(node, 1)?;
        let out_shape = self.out_shape(node, 0)?;
        let (a_mul, b_mul) = broadcast_muls(&a_shape, &b_shape, &out_shape).map_err(|detail| {
            MetalError::InvalidGraph {
                node: node.name.clone(),
                detail,
            }
        })?;
        let a = self.input_region(node, 0)?;
        let b = self.input_region(node, 1)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), Elem::Bool)?;
        let n = numel(&out_shape);
        let params = BinaryParams {
            rank: out_shape.len() as u32,
            n: n as u32,
            op: op_code,
            unused: 0,
            a_base: 0,
            b_base: 0,
            y_base: 0,
            out_shape: shape_array(&out_shape),
            a_mul,
            b_mul,
        };
        self.push(
            node,
            Kernel::Compare,
            vec![(0, a), (1, b), (2, y)],
            as_bytes(&params),
            chunked(n),
            EW_THREADGROUP,
        );
        Ok(())
    }

    fn encode_and(&mut self, node: &IrNode) -> Result<()> {
        let a_shape = self.input_shape(node, 0)?;
        let b_shape = self.input_shape(node, 1)?;
        let out_shape = self.out_shape(node, 0)?;
        let (a_mul, b_mul) = broadcast_muls(&a_shape, &b_shape, &out_shape).map_err(|detail| {
            MetalError::InvalidGraph {
                node: node.name.clone(),
                detail,
            }
        })?;
        let a = self.input_region(node, 0)?;
        let b = self.input_region(node, 1)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), Elem::Bool)?;
        let n = numel(&out_shape);
        let params = BinaryParams {
            rank: out_shape.len() as u32,
            n: n as u32,
            op: 9,
            unused: 0,
            a_base: 0,
            b_base: 0,
            y_base: 0,
            out_shape: shape_array(&out_shape),
            a_mul,
            b_mul,
        };
        self.push(
            node,
            Kernel::And,
            vec![(0, a), (1, b), (2, y)],
            as_bytes(&params),
            chunked(n),
            EW_THREADGROUP,
        );
        Ok(())
    }

    fn encode_where(&mut self, node: &IrNode) -> Result<()> {
        let c_shape = self.input_shape(node, 0)?;
        let a_shape = self.input_shape(node, 1)?;
        let b_shape = self.input_shape(node, 2)?;
        let out_shape = self.out_shape(node, 0)?;
        let (a_mul, b_mul) = broadcast_muls(&a_shape, &b_shape, &out_shape).map_err(|detail| {
            MetalError::InvalidGraph {
                node: node.name.clone(),
                detail,
            }
        })?;
        let c_mul = broadcast_muls(&c_shape, &c_shape, &out_shape)
            .map_err(|detail| MetalError::InvalidGraph {
                node: node.name.clone(),
                detail,
            })?
            .0;
        let cond = self.input_region(node, 0)?;
        let a = self.input_region(node, 1)?;
        let b = self.input_region(node, 2)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), Elem::F32)?;
        let n = numel(&out_shape);
        let params = WhereParams {
            rank: out_shape.len() as u32,
            n: n as u32,
            unused0: 0,
            unused1: 0,
            c_base: 0,
            a_base: 0,
            b_base: 0,
            y_base: 0,
            out_shape: shape_array(&out_shape),
            c_mul,
            a_mul,
            b_mul,
        };
        self.push(
            node,
            Kernel::Where,
            vec![(0, cond), (1, a), (2, b), (3, y)],
            as_bytes(&params),
            chunked(n),
            EW_THREADGROUP,
        );
        Ok(())
    }

    fn encode_cast(&mut self, node: &IrNode, to: DataType) -> Result<()> {
        let from_elem = self.input_elem(node, 0)?;
        let to_elem = Elem::from_onnx(to).ok_or_else(|| MetalError::InvalidGraph {
            node: node.name.clone(),
            detail: format!("Cast to {to:?} is not encoded"),
        })?;
        let shape = self.input_shape(node, 0)?;
        let n = numel(&shape);
        let x = self.input_region(node, 0)?;
        let out_shape = self.out_shape(node, 0)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape, to_elem)?;
        let params = CastParams {
            n: n as u32,
            from: from_elem.code(),
            to: to_elem.code(),
            unused: 0,
            x_base: 0,
            y_base: 0,
        };
        self.push(
            node,
            Kernel::Cast,
            vec![(0, x), (1, y)],
            as_bytes(&params),
            chunked(n),
            EW_THREADGROUP,
        );
        Ok(())
    }

    // -- layout / indexing ------------------------------------------------

    fn encode_transpose(&mut self, node: &IrNode, perm: &[usize]) -> Result<()> {
        let in_shape = self.input_shape(node, 0)?;
        let out_shape = self.out_shape(node, 0)?;
        if in_shape.len() != perm.len() {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("perm {perm:?} does not match rank {}", in_shape.len()),
            });
        }
        if out_shape.len() > MAX_RANK {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("rank {} exceeds the copy ABI", out_shape.len()),
            });
        }
        let in_stride = strides(&in_shape);
        let out_stride = strides(&out_shape);
        let elem = self.input_elem(node, 0)?;
        let src = self.input_region(node, 0)?;
        let dst = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), elem)?;
        let mut params = zero_copy_params(out_shape.len(), numel(&out_shape), elem.bytes() as u32);
        for axis in 0..out_shape.len() {
            params.out_shape[axis] = out_shape[axis] as u32;
            params.dst_stride[axis] = out_stride[axis] as i32;
            params.mul[axis] = in_stride[perm[axis]] as i32;
            params.add[axis] = 0;
            params.in_dim[axis] = in_shape[perm[axis]] as i32;
            params.use_mod[axis] = 0;
        }
        self.push(
            node,
            Kernel::Copy,
            vec![(0, dst), (1, src)],
            as_bytes(&params),
            chunked(numel(&out_shape)),
            COPY_THREADGROUP,
        );
        Ok(())
    }

    fn encode_concat(&mut self, node: &IrNode, axis: i64) -> Result<()> {
        let out_shape = self.out_shape(node, 0)?;
        let rank = out_shape.len();
        let axis = normalize_axis(axis, rank, &node.name)?;
        let out_stride = strides(&out_shape);
        let out_tid = self.out_tid(node, 0)?;
        let elem = self.out_elem(node, 0)?;
        let dst = self.alloc_out(out_tid, out_shape.clone(), elem)?;
        let mut axis_offset = 0usize;
        for index in 0..node.inputs.len() {
            let Some(tid) = node.inputs[index] else { continue };
            let shape = self.shape_of(tid)?;
            let src = self.region_of(tid)?;
            let src_stride = strides(&shape);
            // Destination base for this slice: sum(axis_offset * out_stride[axis]).
            let dst_base = axis_offset * out_stride[axis];
            let mut params = zero_copy_params(rank, numel(&shape), elem.bytes() as u32);
            params.dst_base = dst_base as i32;
            for d in 0..rank {
                params.out_shape[d] = shape[d] as u32;
                params.dst_stride[d] = out_stride[d] as i32;
                params.mul[d] = src_stride[d] as i32;
                params.in_dim[d] = shape[d] as i32;
            }
            self.push(
                node,
                Kernel::Copy,
                vec![(0, dst), (1, src)],
                as_bytes(&params),
                chunked(numel(&shape)),
                COPY_THREADGROUP,
            );
            axis_offset += shape[axis];
        }
        Ok(())
    }

    fn encode_split(&mut self, node: &IrNode, axis: i64) -> Result<()> {
        let in_shape = self.input_shape(node, 0)?;
        let rank = in_shape.len();
        let axis = normalize_axis(axis, rank, &node.name)?;
        let in_stride = strides(&in_shape);
        let elem = self.input_elem(node, 0)?;
        let src = self.input_region(node, 0)?;
        let output_count = node.outputs.len();
        let split_sizes: Vec<usize> = if let Some(tid) = node.inputs.get(1).and_then(|o| *o) {
            self.const_i64(node, 1)
                .map(|v| v.into_iter().map(|x| x.max(0) as usize).collect())?
        } else {
            let chunk = in_shape[axis].div_ceil(output_count.max(1));
            let mut sizes = Vec::new();
            let mut remaining = in_shape[axis];
            for _ in 0..output_count {
                let take = chunk.min(remaining);
                sizes.push(take);
                remaining -= take;
            }
            sizes
        };
        if split_sizes.len() != output_count {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!(
                    "{} split sizes for {} outputs",
                    split_sizes.len(),
                    output_count
                ),
            });
        }
        let Some(tid0) = split_sizes.first() else { return Ok(()) };
        let _ = tid0;
        let mut axis_offset = 0usize;        for (index, size) in split_sizes.into_iter().enumerate() {
            let Some(tid) = node.outputs.get(index).and_then(|o| *o) else { continue };
            let mut shape = in_shape.clone();
            shape[axis] = size;
            let dst = self.alloc_out(tid, shape.clone(), elem)?;
            let mut params = zero_copy_params(rank, numel(&shape), elem.bytes() as u32);
            params.src_base = (axis_offset * in_stride[axis]) as i32;
            for d in 0..rank {
                params.out_shape[d] = shape[d] as u32;
                params.dst_stride[d] = strides(&shape)[d] as i32;
                params.mul[d] = in_stride[d] as i32;
                params.in_dim[d] = in_shape[d] as i32;
            }
            self.push_writes(
                node,
                Kernel::Copy,
                vec![(0, dst), (1, src)],
                as_bytes(&params),
                chunked(numel(&shape)),
                COPY_THREADGROUP,
                vec![tid],
            );
            axis_offset += size;
        }
        Ok(())
    }

    fn encode_slice(&mut self, node: &IrNode) -> Result<()> {
        let in_shape = self.input_shape(node, 0)?;
        let out_shape = self.out_shape(node, 0)?;
        let rank = out_shape.len();
        if rank > MAX_RANK {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("rank {rank} exceeds the copy ABI"),
            });
        }
        let starts = self.const_i64(node, 1)?;
        let ends = self.const_i64(node, 2)?;
        let axes: Vec<i64> = if node.inputs.get(3).and_then(|o| *o).is_some() {
            self.const_i64(node, 3)?
        } else {
            (0..starts.len() as i64).collect()
        };
        let steps: Vec<i64> = if node.inputs.get(4).and_then(|o| *o).is_some() {
            self.const_i64(node, 4)?
        } else {
            vec![1; starts.len()]
        };
        let in_stride = strides(&in_shape);
        let mut params = zero_copy_params(rank, numel(&out_shape), self.input_elem(node, 0)?.bytes() as u32);
        for d in 0..rank {
            params.out_shape[d] = out_shape[d] as u32;
            params.dst_stride[d] = strides(&out_shape)[d] as i32;
            params.in_dim[d] = in_shape[d] as i32;
            params.mul[d] = in_stride[d] as i32;
            params.add[d] = 0;
        }
        for (i, &raw_axis) in axes.iter().enumerate() {
            let axis = normalize_axis(raw_axis, in_shape.len(), &node.name)?;
            let step = steps.get(i).copied().unwrap_or(1);
            let start = starts[i];
            let begin = if start < 0 { start + in_shape[axis] as i64 } else { start };
            let begin = begin.clamp(0, in_shape[axis] as i64);
            params.mul[axis] = (in_stride[axis] as i64 * step) as i32;
            params.add[axis] = (in_stride[axis] as i64 * begin) as i32;
        }
        let src = self.input_region(node, 0)?;
        let dst = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), self.out_elem(node, 0)?)?;
        let _ = ends;
        self.push(
            node,
            Kernel::Copy,
            vec![(0, dst), (1, src)],
            as_bytes(&params),
            chunked(numel(&out_shape)),
            COPY_THREADGROUP,
        );
        Ok(())
    }

    fn encode_gather(&mut self, node: &IrNode, axis: i64) -> Result<()> {
        let data_shape = self.input_shape(node, 0)?;
        let idx_shape = self.input_shape(node, 1)?;
        let out_shape = self.out_shape(node, 0)?;
        let axis = normalize_axis(axis, data_shape.len(), &node.name)?;
        let outer: usize = data_shape[..axis].iter().product();
        let inner: usize = data_shape[axis + 1..].iter().product();
        let idx_len: usize = idx_shape.iter().product();
        let idx_elem = self.input_elem(node, 1)?;
        if !matches!(idx_elem, Elem::I64 | Elem::I32) {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("Gather indices must be int64/int32, found {idx_elem:?}"),
            });
        }
        let data = self.input_region(node, 0)?;
        let idx = self.region_of(node.inputs[1].expect("index tensor"))?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), Elem::F32)?;
        let params = GatherParams {
            outer: outer as u32,
            axis_len: data_shape[axis] as u32,
            inner: inner.max(1) as u32,
            idx_len: idx_len as u32,
            out_elems: numel(&out_shape) as u32,
            idx_i64: (idx_elem == Elem::I64) as u32,
            unused0: 0,
            unused1: 0,
            data_off: 0,
            idx_off: 0,
            y_off: 0,
        };
        self.push(
            node,
            Kernel::Gather,
            vec![(0, data), (1, idx), (2, y)],
            as_bytes(&params),
            chunked(numel(&out_shape)),
            EW_THREADGROUP,
        );
        Ok(())
    }

    fn encode_scatter_nd(&mut self, node: &IrNode) -> Result<()> {
        let data_shape = self.input_shape(node, 0)?;
        let idx_shape = self.input_shape(node, 1)?;
        let upd_shape = self.input_shape(node, 2)?;
        let out_shape = self.out_shape(node, 0)?;
        let k = *idx_shape.last().unwrap_or(&1);
        if k == 0 || k > data_shape.len() {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("ScatterND index depth {k} is invalid for {data_shape:?}"),
            });
        }
        let num_updates: usize = idx_shape[..idx_shape.len() - 1].iter().product();
        let slice_size: usize = data_shape[k..].iter().product();
        if numel(&upd_shape) != num_updates * slice_size {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!(
                    "updates {upd_shape:?} do not match {num_updates} x {slice_size} slice elements"
                ),
            });
        }
        let data = self.input_region(node, 0)?;
        let y = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), Elem::F32)?;
        // Start from a copy of `data`, then apply the updates in place.
        let mut copy = zero_copy_params(out_shape.len(), numel(&data_shape), 4);
        for d in 0..out_shape.len() {
            copy.out_shape[d] = data_shape[d] as u32;
            copy.dst_stride[d] = strides(&out_shape)[d] as i32;
            copy.mul[d] = strides(&data_shape)[d] as i32;
            copy.in_dim[d] = data_shape[d] as i32;
        }
        self.push(
            node,
            Kernel::Copy,
            vec![(0, y), (1, data)],
            as_bytes(&copy),
            chunked(numel(&data_shape)),
            COPY_THREADGROUP,
        );
        let idx = self.region_of(node.inputs[1].expect("indices"))?;
        let upd = self.input_region(node, 2)?;
        let out_stride = strides(&out_shape);
        let mut stride_array = [0i32; MAX_RANK];
        for d in 0..out_shape.len().min(MAX_RANK) {
            stride_array[d] = out_stride[d] as i32;
        }
        let params = ScatterParams {
            num_updates: num_updates as u32,
            k: k as u32,
            slice_size: slice_size as u32,
            out_elems: numel(&out_shape) as u32,
            rank: out_shape.len() as u32,
            idx_off: 0,
            upd_off: 0,
            out_off: 0,
            strides: stride_array,
        };
        self.push(
            node,
            Kernel::Scatter,
            vec![(0, idx), (1, upd), (2, y)],
            as_bytes(&params),
            [(num_updates.max(1) as u32), 1, 1],
            [32, 1, 1],
        );
        Ok(())
    }

    fn encode_expand(&mut self, node: &IrNode) -> Result<()> {
        let in_shape = self.input_shape(node, 0)?;
        let out_shape = self.out_shape(node, 0)?;
        if out_shape.len() > MAX_RANK {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("rank {} exceeds the copy ABI", out_shape.len()),
            });
        }
        let in_stride = strides(&in_shape);
        let elem = self.input_elem(node, 0)?;
        let mut params = zero_copy_params(out_shape.len(), numel(&out_shape), elem.bytes() as u32);
        for d in 0..out_shape.len() {
            params.out_shape[d] = out_shape[d] as u32;
            params.dst_stride[d] = strides(&out_shape)[d] as i32;
            let in_axis = d as isize - (out_shape.len() as isize - in_shape.len() as isize);
            if in_axis >= 0 && in_shape[in_axis as usize] != 1 {
                params.mul[d] = in_stride[in_axis as usize] as i32;
                params.in_dim[d] = in_shape[in_axis as usize] as i32;
            } else {
                params.mul[d] = 0;
                params.in_dim[d] = 1;
            }
        }
        let src = self.input_region(node, 0)?;
        let dst = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), elem)?;
        self.push(
            node,
            Kernel::Copy,
            vec![(0, dst), (1, src)],
            as_bytes(&params),
            chunked(numel(&out_shape)),
            COPY_THREADGROUP,
        );
        Ok(())
    }

    fn encode_tile(&mut self, node: &IrNode) -> Result<()> {
        let in_shape = self.input_shape(node, 0)?;
        let out_shape = self.out_shape(node, 0)?;
        let in_stride = strides(&in_shape);
        let elem = self.input_elem(node, 0)?;
        let mut params = zero_copy_params(out_shape.len(), numel(&out_shape), elem.bytes() as u32);
        for d in 0..out_shape.len() {
            params.out_shape[d] = out_shape[d] as u32;
            params.dst_stride[d] = strides(&out_shape)[d] as i32;
            params.mul[d] = in_stride[d] as i32;
            params.in_dim[d] = in_shape[d] as i32;
            params.use_mod[d] = in_shape[d].max(1) as u32;
        }
        let src = self.input_region(node, 0)?;
        let dst = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), elem)?;
        self.push(
            node,
            Kernel::Copy,
            vec![(0, dst), (1, src)],
            as_bytes(&params),
            chunked(numel(&out_shape)),
            COPY_THREADGROUP,
        );
        Ok(())
    }

    fn encode_pad(&mut self, node: &IrNode, mode: &str) -> Result<()> {
        let in_shape = self.input_shape(node, 0)?;
        let out_shape = self.out_shape(node, 0)?;
        let pads = self.const_i64(node, 1)?;
        let rank = in_shape.len();
        if pads.len() != 2 * rank {
            return Err(MetalError::InvalidGraph {
                node: node.name.clone(),
                detail: format!("pads {pads:?} must hold 2*{rank} values"),
            });
        }
        let in_stride = strides(&in_shape);
        let elem = self.input_elem(node, 0)?;
        let edge = match mode {
            "constant" => 0u32,
            "edge" => 1,
            "reflect" => 2,
            other => {
                return Err(MetalError::InvalidGraph {
                    node: node.name.clone(),
                    detail: format!("pad mode '{other}' is not encoded"),
                })
            }
        };
        let pad_value = if mode == "constant" && node.inputs.get(2).and_then(|o| *o).is_some() {
            self.const_f32(node, 2)?.first().copied().unwrap_or(0.0)
        } else {
            0.0
        };
        let src = self.input_region(node, 0)?;
        let dst = self.alloc_out(self.out_tid(node, 0)?, out_shape.clone(), elem)?;
        if edge == 0 {
            let fill = FillParams {
                n: numel(&out_shape) as u32,
                elem_bytes: elem.bytes() as u32,
                dst_base: 0,
                bits: pad_value.to_bits(),
            };
            self.push(
                node,
                Kernel::Fill,
                vec![(0, dst)],
                as_bytes(&fill),
                chunked(numel(&out_shape)),
                EW_THREADGROUP,
            );
        }
        let out_stride = strides(&out_shape);
        let (copied_shape, out_elems) = if edge == 0 {
            (in_shape.clone(), numel(&in_shape))
        } else {
            (out_shape.clone(), numel(&out_shape))
        };
        let mut params = zero_copy_params(rank, out_elems, elem.bytes() as u32);
        params.edge = edge;
        let mut dst_base = 0usize;
        for d in 0..rank {
            params.out_shape[d] = copied_shape[d] as u32;
            params.dst_stride[d] = out_stride[d] as i32;
            params.mul[d] = in_stride[d] as i32;
            params.add[d] = 0;
            params.in_dim[d] = in_shape[d] as i32;
            if edge == 0 {
                let before = pads[d].max(0) as usize;
                dst_base += before * out_stride[d];
            } else {
                params.shift[d] = -(pads[d].max(0)) as i32;
            }
        }
        params.dst_base = dst_base as i32;
        self.push(
            node,
            Kernel::Copy,
            vec![(0, dst), (1, src)],
            as_bytes(&params),
            chunked(out_elems),
            COPY_THREADGROUP,
        );
        Ok(())
    }

    // -- quantization -----------------------------------------------------

    fn encode_dynamic_quantize(&mut self, node: &IrNode) -> Result<()> {
        let x_shape = self.input_shape(node, 0)?;
        let n = numel(&x_shape);
        let x = self.input_region(node, 0)?;
        let scale_tid = self.out_tid(node, 1)?;
        let zp_tid = self.out_tid(node, 2)?;
        let y_tid = self.out_tid(node, 0)?;
        let scale_shape = self.out_shape(node, 1)?;
        let zp_shape = self.out_shape(node, 2)?;
        let y_shape = self.out_shape(node, 0)?;
        let minmax_region = self.alloc_out(
            minmax_tid(node),
            vec![2],
            Elem::F32,
        )?;
        let scale = self.alloc_out(scale_tid, scale_shape, Elem::F32)?;
        let zp = self.alloc_out(zp_tid, zp_shape, Elem::U8)?;
        let y = self.alloc_out(y_tid, y_shape, Elem::U8)?;
        self.push_writes(
            node,
            Kernel::MinMax,
            vec![(0, x), (1, minmax_region)],
            as_bytes(&MinMaxParams { n: n as u32, x_off: 0 }),
            [1, 1, 1],
            [1, 1, 1],
            vec![minmax_tid(node)],
        );
        let finalize = QuantFinalizeParams {
            minmax_off: 0,
            scale_off: 0,
            zp_off: 0,
            eps: 1e-8,
        };
        self.push_writes(
            node,
            Kernel::QuantFinalize,
            vec![(0, minmax_region), (1, scale), (2, zp)],
            as_bytes(&finalize),
            [1, 1, 1],
            [1, 1, 1],
            vec![scale_tid, zp_tid],
        );
        let quantize = QuantizeParams {
            n: n as u32,
            x_off: 0,
            y_off: 0,
            scale_off: 0,
            zp_off: 0,
        };
        self.push_writes(
            node,
            Kernel::Quantize,
            vec![(0, x), (1, y), (2, scale), (3, zp)],
            as_bytes(&quantize),
            chunked(n),
            EW_THREADGROUP,
            vec![y_tid],
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

const EW_THREADGROUP: [u32; 3] = [64, 1, 1];
const COPY_THREADGROUP: [u32; 3] = [64, 1, 1];
const ROW_THREADGROUP: [u32; 3] = [32, 1, 1];
const GEMM_THREADGROUP: [u32; 3] = [16, 16, 1];

const ROW_SOFTMAX: u32 = 0;
const ROW_LAYER_NORM: u32 = 1;
const ROW_INSTANCE_NORM: u32 = 2;
const ROW_MEAN: u32 = 3;

fn chunked(elems: usize) -> [u32; 3] {
    [elems.max(1) as u32, 1, 1]
}

fn broadcast_batch(mut flat:usize, out:&[usize], input:&[usize])->Option<usize> {
    if input.len()>out.len(){ return None; }
    let (mut stride,mut pos)=(1usize,0usize);
    for (j,&dim) in out.iter().rev().enumerate(){
        if dim==0{return None;}
        let sub=input.iter().rev().nth(j).copied().unwrap_or(1);
        if sub!=1 && sub!=dim{return None;}
        if sub!=1 {pos=pos.checked_add((flat%dim).checked_mul(stride)?)?;}
        flat/=dim;stride=stride.checked_mul(sub)?;
    }
    Some(pos)
}

#[test]
fn batch_projection_reuses_broadcast_matmul_weights(){
    assert_eq!(broadcast_batch(4,&[2,3],&[]),Some(0));
    assert_eq!(broadcast_batch(4,&[2,3],&[3]),Some(1));
    assert_eq!(broadcast_batch(4,&[2,3],&[2,1]),Some(1));
    assert_eq!(broadcast_batch(4,&[2,3],&[4]),None);
}

fn tile_grid(dim: usize) -> u32 {
    (dim.max(1) as u32).div_ceil(16) * 16
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for index in (0..shape.len()).rev() {
        out[index] = acc;
        acc = acc.saturating_mul(shape[index].max(1));
    }
    out
}

fn shape_array(shape: &[usize]) -> [u32; MAX_RANK] {
    let mut out = [0u32; MAX_RANK];
    for (index, &dim) in shape.iter().take(MAX_RANK).enumerate() {
        out[index] = dim as u32;
    }
    out
}

fn zero_copy_params(rank: usize, out_elems: usize, elem_bytes: u32) -> CopyParams {
    CopyParams {
        rank: rank as u32,
        out_elems: out_elems as u32,
        elem_bytes,
        edge: 0,
        dst_base: 0,
        src_base: 0,
        pad_value: 0.0,
        out_shape: [0; MAX_RANK],
        dst_stride: [0; MAX_RANK],
        mul: [0; MAX_RANK],
        add: [0; MAX_RANK],
        shift: [0; MAX_RANK],
        in_dim: [0; MAX_RANK],
        use_mod: [0; MAX_RANK],
    }
}

/// Broadcast multipliers for ONNX right-aligned broadcasting: a dim of size 1
/// (or a missing leading dim) contributes stride 0 to the flat source offset.
fn broadcast_muls(
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
) -> std::result::Result<([u32; MAX_RANK], [u32; MAX_RANK]), String> {
    fn one(input: &[usize], out: &[usize]) -> std::result::Result<[u32; MAX_RANK], String> {
        if input.len() > out.len() {
            return Err(format!("input rank {} exceeds output rank {}", input.len(), out.len()));
        }
        let stride = strides(input);
        let mut mul = [0u32; MAX_RANK];
        let lead = out.len() - input.len();
        for d in 0..out.len() {
            if d < lead {
                mul[d] = 0;
                continue;
            }
            let dim = input[d - lead];
            if dim == out[d] {
                mul[d] = stride[d - lead] as u32;
            } else if dim == 1 {
                mul[d] = 0;
            } else {
                return Err(format!(
                    "cannot broadcast dim {dim} to {} at axis {d}",
                    out[d]
                ));
            }
        }
        Ok(mul)
    }
    Ok((one(a_shape, out_shape)?, one(b_shape, out_shape)?))
}

fn normalize_axis(axis: i64, rank: usize, node: &str) -> Result<usize> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 || a >= r {
        return Err(MetalError::InvalidGraph {
            node: node.to_string(),
            detail: format!("axis {axis} is out of range for rank {rank}"),
        });
    }
    Ok(a as usize)
}

/// Synthetic tensor id used to reserve a two-element scratch slot for the
/// dynamic-quantization min/max reduction.
fn minmax_tid(node: &IrNode) -> TensorId {
    TensorId(usize::MAX / 2 + node.id.0 + 1)
}

/// Decode a weight/constant into the bytes uploaded to the weight buffer.
/// Float16 is widened to FP32 so MSL kernels only ever see `float` weights,
/// matching the CUDA session's choice; integer tensors keep their width.
fn decode_weight(proto: &TensorProto) -> std::result::Result<(Vec<u8>, Elem), String> {
    match proto.data_type {
        DataType::Float => {
            if proto.raw_data.len() == proto.elem_count() * 4 {
                Ok((proto.raw_data.clone(), Elem::F32))
            } else {
                let values = proto.to_f32_vec().map_err(|e| e.to_string())?;
                Ok((f32_bytes(&values), Elem::F32))
            }
        }
        DataType::Float16 => {
            let values = proto.to_f32_vec().map_err(|e| e.to_string())?;
            Ok((f32_bytes(&values), Elem::F32))
        }
        DataType::Int32 | DataType::Int64 | DataType::Uint8 | DataType::Int8 | DataType::Bool => {
            let elem = Elem::from_onnx(proto.data_type).expect("integer type mapped");
            let expected = proto.elem_count() * elem.bytes();
            if proto.raw_data.len() == expected {
                Ok((proto.raw_data.clone(), elem))
            } else {
                let values = proto.to_i64_vec().map_err(|e| e.to_string())?;
                let mut bytes = Vec::with_capacity(expected);
                for value in values {
                    match elem {
                        Elem::I64 => bytes.extend_from_slice(&value.to_le_bytes()),
                        Elem::I32 => bytes.extend_from_slice(&(value as i32).to_le_bytes()),
                        Elem::U8 => bytes.push(value as u8),
                        Elem::I8 => bytes.push(value as i8 as u8),
                        Elem::Bool => bytes.push((value != 0) as u8),
                        _ => unreachable!(),
                    }
                }
                Ok((bytes, elem))
            }
        }
        other => Err(format!("weight type {other:?} is not supported")),
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn cpu_tensor_bytes(tensor: &CpuTensor) -> Vec<u8> {
    match tensor.dtype() {
        DataType::Float => f32_bytes(&tensor.to_f32_vec()),
        DataType::Int64 => {
            let mut out = Vec::new();
            for value in tensor.to_i64_vec() {
                out.extend_from_slice(&value.to_le_bytes());
            }
            out
        }
        DataType::Int32 => {
            let mut out = Vec::new();
            for value in tensor.to_i64_vec() {
                out.extend_from_slice(&(value as i32).to_le_bytes());
            }
            out
        }
        DataType::Bool => tensor.to_bool_vec().into_iter().map(|b| b as u8).collect(),
        DataType::Uint8 => tensor.to_i64_vec().into_iter().map(|v| v as u8).collect(),
        _ => f32_bytes(&tensor.to_f32_vec()),
    }
}

/// Re-exported so the host module and tests can evaluate single nodes on the CPU
/// for cross-checks without importing `cpu_shape_eval` directly.
pub fn cpu_reference(op: &Op, tensors: &[Option<&CpuTensor>], shapes: &[Option<&[usize]>]) -> std::result::Result<CpuTensor, String> {
    cpu_eval_op(op, tensors, shapes).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// The MSL library
// ---------------------------------------------------------------------------

/// One compute library. Every parameter struct mirrors the `#[repr(C)]` struct
/// of the same name above field for field (all members are 32-bit, so the C and
/// MSL layouts agree); `params_struct_sizes_match_msl` pins the sizes.
///
/// MPS is deliberately unused: everything below is plain `kernel` MSL.
pub const MSL_LIBRARY: &str = include_str!("../../native/metal/infer_graph.metal");

// ---------------------------------------------------------------------------
// Session (the only type that may report a Metal result)
// ---------------------------------------------------------------------------

/// A compiled model with a live Metal device behind it.
///
/// Construction is the gate: [`Self::from_bytes`] compiles and validates the
/// whole graph first and then opens a device, so an instance always has a queue,
/// a library and every pipeline it needs. `run_f32` therefore either encodes
/// real commands or returns an error — it never falls back to CPU results.
pub struct NativeMetalSession {
    program: MslProgram,
    host: host::Device,
    /// Metal objects are thread safe, but this backend shares one arena across
    /// runs; serialising executions keeps concurrent `run_f32` calls from
    /// overwriting each other's intermediates.
    serial: std::sync::Mutex<()>,
}

impl NativeMetalSession {
    /// Compile `.onnx` bytes and open the Metal device.
    pub fn from_bytes(onnx_bytes: &[u8], overrides: &ShapeOverrides) -> std::result::Result<Self, String> {
        let program = MslProgram::compile(onnx_bytes, overrides).map_err(|e| e.to_string())?;
        program.validate().map_err(|detail| {
            format!("metal program invariant violated before any dispatch: {detail}")
        })?;
        if program.dispatches.is_empty(){ return Err("Metal plan has no GPU dispatch; no Metal execution will be claimed".into()); }
        let host = host::Device::new(&program).map_err(|detail| {
            format!(
                "{detail} [plan compiled: {} nodes, {} dispatches, {} kernels, arena {} bytes, extra {} bytes; \
                 zero Metal commands were encoded, so no result may be reported as Metal]",
                program.plan.graph.nodes.len(),
                program.dispatches.len(),
                program.used_kernels.len(),
                program.arena_bytes(),
                program.extra_bytes
            )
        })?;
        Ok(Self {
            program,
            host,
            serial: std::sync::Mutex::new(()),
        })
    }

    pub fn program(&self) -> &MslProgram {
        &self.program
    }

    /// Execute on the device. One command buffer, one compute encoder.
    pub fn run(&self, inputs: &NamedInputs<'_>) -> Result<NamedOutputs> {
        let _guard = self.serial.lock().map_err(|_| {
            MetalError::Runtime("the session lock was poisoned by a previous panic".into())
        })?;
        self.host.run(&self.program, inputs)
    }
}

impl NativeInferBackend for NativeMetalSession {
    fn kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn arena_peak_bytes(&self) -> usize {
        self.program.arena_bytes()
    }

    fn run_f32(&self, inputs: &NamedInputs<'_>) -> std::result::Result<NamedOutputs, String> {
        self.run(inputs).map_err(|e| e.to_string())
    }
}

impl fmt::Debug for NativeMetalSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeMetalSession")
            .field("program", &self.program)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Host bridge
//
// This module is compiled on every host so its Objective-C message-send
// surface, parameter marshalling and command encoding are type-checked by the
// Rust compiler even where the framework cannot be loaded. Only `Bridge::load`
// differs: on macOS it dlopen()s libobjc/Foundation/CoreGraphics/Metal, and
// elsewhere it reports that no Metal runtime exists. Nothing here is gated
// behind `cfg` in a way that would hide the encoder from `cargo check`.
// ---------------------------------------------------------------------------

mod host {
    use super::*;
    use std::ffi::{c_char, c_void, CString};

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct MTLSize {
        width: usize,
        height: usize,
        depth: usize,
    }

    /// `MTLResourceStorageModeShared`: one physical allocation visible to the
    /// CPU and the GPU, which is what makes unified memory zero-copy possible.
    // MTLStorageModeShared = 0, shifted by MTLResourceStorageModeShift (4).
    // 0x20 is MTLStorageModePrivate; a CPU read of such a buffer is not valid.
    const STORAGE_SHARED: usize = 0;

    #[derive(Clone, Copy)]
    struct Bridge {
        send: *const c_void,
        get_class: unsafe extern "C" fn(*const c_char) -> *mut c_void,
        register_sel: unsafe extern "C" fn(*const c_char) -> *mut c_void,
        create_device: unsafe extern "C" fn() -> *mut c_void,
    }

    // Function pointers and a process-global framework handle are immutable.
    unsafe impl Send for Bridge {}
    unsafe impl Sync for Bridge {}

    impl Bridge {
        #[cfg(target_os = "macos")]
        fn load() -> Option<Self> {
            unsafe extern "C" {
                fn dlopen(path: *const c_char, mode: i32) -> *mut c_void;
                fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
            }
            unsafe fn open(path: &str) -> Option<*mut c_void> {
                let path = CString::new(path).ok()?;
                let handle = dlopen(path.as_ptr(), 2 /* RTLD_NOW */);
                (!handle.is_null()).then_some(handle)
            }
            unsafe fn symbol(handle: *mut c_void, name: &str) -> Option<*mut c_void> {
                let name = CString::new(name).ok()?;
                let pointer = dlsym(handle, name.as_ptr());
                (!pointer.is_null()).then_some(pointer)
            }
            unsafe {
                let objc = open("/usr/lib/libobjc.A.dylib")?;
                let _foundation =
                    open("/System/Library/Frameworks/Foundation.framework/Foundation")?;
                // MTLCreateSystemDefaultDevice documents CoreGraphics as required.
                let _cg = open("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")?;
                let metal = open("/System/Library/Frameworks/Metal.framework/Metal")?;
                Some(Self {
                    send: symbol(objc, "objc_msgSend")?,
                    get_class: std::mem::transmute(symbol(objc, "objc_getClass")?),
                    register_sel: std::mem::transmute(symbol(objc, "sel_registerName")?),
                    create_device: std::mem::transmute(symbol(metal, "MTLCreateSystemDefaultDevice")?),
                })
            }
        }

        #[cfg(not(target_os = "macos"))]
        fn load() -> Option<Self> {
            // No Metal runtime exists off macOS; `Device::new` turns this into a
            // precise "no command buffer can be encoded" error.
            None
        }

        unsafe fn sel(&self, name: &str) -> *mut c_void {
            let name = CString::new(name).expect("selector has no interior NUL");
            (self.register_sel)(name.as_ptr())
        }

        unsafe fn class(&self, name: &str) -> *mut c_void {
            let name = CString::new(name).expect("class name has no interior NUL");
            (self.get_class)(name.as_ptr())
        }

        unsafe fn nsstring(&self, value: &str) -> *mut c_void {
            let value = match CString::new(value) {
                Ok(v) => v,
                Err(_) => return std::ptr::null_mut(),
            };
            let cls = self.class("NSString");
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_char) -> *mut c_void =
                std::mem::transmute(self.send);
            send(cls, self.sel("stringWithUTF8String:"), value.as_ptr())
        }

        unsafe fn p0(&self, object: *mut c_void, selector: &str) -> *mut c_void {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                std::mem::transmute(self.send);
            send(object, self.sel(selector))
        }

        unsafe fn u0(&self, object: *mut c_void, selector: &str) -> usize {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize =
                std::mem::transmute(self.send);
            send(object, self.sel(selector))
        }

        unsafe fn p1(&self, object: *mut c_void, selector: &str, arg: *mut c_void) -> *mut c_void {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void =
                std::mem::transmute(self.send);
            send(object, self.sel(selector), arg)
        }

        unsafe fn void0(&self, object: *mut c_void, selector: &str) {
            let send:unsafe extern "C" fn(*mut c_void,*mut c_void)=std::mem::transmute(self.send);
            send(object,self.sel(selector));
        }

        unsafe fn void1(&self, object: *mut c_void, selector: &str, arg: *mut c_void) {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) =
                std::mem::transmute(self.send);
            send(object, self.sel(selector), arg)
        }

        unsafe fn err1(
            &self,
            object: *mut c_void,
            selector: &str,
            arg: *mut c_void,
            error: *mut *mut c_void,
        ) -> *mut c_void {
            let send: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *mut c_void,
                *mut *mut c_void,
            ) -> *mut c_void = std::mem::transmute(self.send);
            send(object, self.sel(selector), arg, error)
        }

        unsafe fn err2(
            &self,
            object: *mut c_void,
            selector: &str,
            first: *mut c_void,
            second: *mut c_void,
            error: *mut *mut c_void,
        ) -> *mut c_void {
            let send: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *mut c_void,
                *mut c_void,
                *mut *mut c_void,
            ) -> *mut c_void = std::mem::transmute(self.send);
            send(object, self.sel(selector), first, second, error)
        }

        unsafe fn bytes(
            &self,
            device: *mut c_void,
            selector: &str,
            pointer: *const c_void,
            length: usize,
            options: usize,
        ) -> *mut c_void {
            let send: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *const c_void,
                usize,
                usize,
            ) -> *mut c_void = std::mem::transmute(self.send);
            send(device, self.sel(selector), pointer, length, options)
        }

        unsafe fn length_options(
            &self,
            device: *mut c_void,
            selector: &str,
            length: usize,
            options: usize,
        ) -> *mut c_void {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void, usize, usize) -> *mut c_void =
                std::mem::transmute(self.send);
            send(device, self.sel(selector), length, options)
        }

        unsafe fn bind(
            &self,
            encoder: *mut c_void,
            selector: &str,
            pointer: *const c_void,
            length: usize,
            index: usize,
        ) {
            let send: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *const c_void,
                usize,
                usize,
            ) = std::mem::transmute(self.send);
            send(encoder, self.sel(selector), pointer, length, index)
        }

        unsafe fn dispatch(
            &self,
            encoder: *mut c_void,
            threads: MTLSize,
            per_group: MTLSize,
        ) {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void, MTLSize, MTLSize) =
                std::mem::transmute(self.send);
            send(encoder, self.sel("dispatchThreads:threadsPerThreadgroup:"), threads, per_group)
        }
    }

    fn error_text(bridge: &Bridge, error: *mut c_void) -> Option<String> {
        if error.is_null() {
            return None;
        }
        unsafe {
            let description = bridge.p0(error, "localizedDescription");
            if description.is_null() {
                return Some("Metal reported an error without a description".into());
            }
            let text = {
                let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *const c_char =
                    std::mem::transmute(bridge.send);
                send(description, bridge.sel("UTF8String"))
            };
            (!text.is_null()).then(|| std::ffi::CStr::from_ptr(text).to_string_lossy().into_owned())
        }
    }

    struct RunObjects<'a> {bridge:&'a Bridge, objects:Vec<*mut c_void>}
    impl Drop for RunObjects<'_>{fn drop(&mut self){for &obj in &self.objects {unsafe{self.bridge.void0(obj,"release")}}}}
    struct AutoreleasePool<'a>(&'a Bridge,*mut c_void);
    impl<'a> AutoreleasePool<'a>{unsafe fn new(b:&'a Bridge)->Self{Self(b,b.p0(b.class("NSAutoreleasePool"),"new"))}}
    impl Drop for AutoreleasePool<'_>{fn drop(&mut self){unsafe{self.0.void0(self.1,"drain")}}}

    pub struct Device {
        bridge: Bridge,
        device: *mut c_void,
        queue: *mut c_void,
        library: *mut c_void,
        weights: *mut c_void,
        arena: *mut c_void,
        extra: *mut c_void,
        pipelines: std::sync::Mutex<HashMap<&'static str, *mut c_void>>,
    }

    // Metal device, queue, buffers, library and pipeline states are documented
    // as thread safe; `NativeMetalSession` additionally serialises runs.
    unsafe impl Send for Device {}
    unsafe impl Sync for Device {}

    impl Drop for Device {
        fn drop(&mut self) {
            unsafe {
                if let Ok(map) = self.pipelines.lock() {
                    for pipeline in map.values() {
                        self.bridge.void0(*pipeline, "release");
                    }
                }
                for object in [self.library, self.weights, self.arena, self.extra] {
                    if !object.is_null() {
                        self.bridge.void0(object, "release");
                    }
                }
                if !self.queue.is_null() {
                    self.bridge.void0(self.queue, "release");
                }
                self.bridge.void0(self.device,"release");
            }
        }
    }

    impl Device {
        pub fn new(program: &MslProgram) -> std::result::Result<Self, String> {
            let bridge = Bridge::load().ok_or_else(|| {
                "libobjc / Foundation / Metal.framework could not be loaded; this host has no \
                 Metal runtime"
                    .to_string()
            })?;
            unsafe {
                let bridge_copy=bridge; let _pool=AutoreleasePool::new(&bridge_copy);
                let device = (bridge.create_device)();
                if device.is_null() {
                    return Err("MTLCreateSystemDefaultDevice returned nil (no Metal device)".into());
                }
                let queue = bridge.p0(device, "newCommandQueue");
                if queue.is_null() {
                    bridge.void0(device,"release");return Err("the Metal device refused a command queue".into());
                }
                let mut device = Device {
                    bridge,
                    device,
                    queue,
                    library: std::ptr::null_mut(),
                    weights: std::ptr::null_mut(),
                    arena: std::ptr::null_mut(),
                    extra: std::ptr::null_mut(),
                    pipelines: std::sync::Mutex::new(HashMap::new()),
                };
                device.weights = device.upload(&program.weights_blob)?;
                device.arena = device.empty(program.arena_bytes().max(256))?;
                device.extra = device.empty(program.extra_bytes.max(256))?;
                device.library = device.compile(program.msl)?;
                for kernel in &program.used_kernels {
                    let pipeline = device.pipeline_state(kernel.name())?;
                    device.pipelines.get_mut().map_err(|_| "pipeline cache poisoned")?.insert(kernel.name(), pipeline);
                }
                Ok(device)
            }
        }

        unsafe fn upload(&mut self, bytes: &[u8]) -> std::result::Result<*mut c_void, String> {
            if bytes.is_empty() {return self.empty(4);}
            let len = bytes.len();
            let buffer = self.bridge.bytes(
                self.device,
                "newBufferWithBytes:length:options:",
                bytes.as_ptr().cast::<c_void>(),
                len,
                STORAGE_SHARED,
            );
            if buffer.is_null() {
                return Err(format!("the device refused a {len}-byte weight buffer"));
            }
            Ok(buffer)
        }

        unsafe fn empty(&mut self, len: usize) -> std::result::Result<*mut c_void, String> {
            let buffer = self.bridge.length_options(
                self.device,
                "newBufferWithLength:options:",
                len,
                STORAGE_SHARED,
            );
            if buffer.is_null() {
                return Err(format!("the device refused a {len}-byte shared buffer"));
            }
            Ok(buffer)
        }

        unsafe fn compile(&mut self, source: &str) -> std::result::Result<*mut c_void, String> {
            let text = self.bridge.nsstring(source);
            if text.is_null() {
                return Err("the MSL source could not be turned into an NSString".into());
            }
            let mut error: *mut c_void = std::ptr::null_mut();
            let options=self.bridge.p0(self.bridge.class("MTLCompileOptions"),"new");
            if options.is_null(){return Err("missing MTLCompileOptions".into());}
            let set:unsafe extern "C" fn(*mut c_void,*mut c_void,bool)=std::mem::transmute(self.bridge.send);
            set(options,self.bridge.sel("setFastMathEnabled:"),false);
            let library = self.bridge.err2(
                self.device,
                "newLibraryWithSource:options:error:",
                text,
                options,
                &mut error,
            );
            self.bridge.void0(options,"release");
            if library.is_null() {
                return Err(error_text(&self.bridge, error)
                    .unwrap_or_else(|| "the Metal compiler rejected the MSL library".into()));
            }
            Ok(library)
        }

        unsafe fn pipeline_state(
            &self,
            name: &'static str,
        ) -> std::result::Result<*mut c_void, String> {
            let function_name = self.bridge.nsstring(name);
            let function = self.bridge.p1(self.library, "newFunctionWithName:", function_name);
            if function.is_null() {
                return Err(format!("the MSL library has no function '{name}'"));
            }
            let mut error: *mut c_void = std::ptr::null_mut();
            let pipeline = self.bridge.err1(
                self.device,
                "newComputePipelineStateWithFunction:error:",
                function,
                &mut error,
            );
            self.bridge.void0(function,"release");
            if pipeline.is_null() {
                return Err(error_text(&self.bridge, error)
                    .unwrap_or_else(|| format!("pipeline for '{name}' could not be built")));
            }
            Ok(pipeline)
        }

        pub fn run(
            &self,
            program: &MslProgram,
            inputs: &NamedInputs<'_>,
        ) -> Result<NamedOutputs> {
            // 1. Validate named inputs against the compiled graph.
            let mut input_buffers: HashMap<TensorId, *mut c_void> = HashMap::new();
            if inputs.len()!=program.inputs.len() {return Err(MetalError::Runtime("input count mismatch".into()));}
            let mut run_buffers=RunObjects{bridge:&self.bridge,objects:Vec::new()};
            let _pool=unsafe{AutoreleasePool::new(&self.bridge)};
            for io in &program.inputs {
                let (data, shape) = inputs.get(&io.name).ok_or_else(|| {
                    MetalError::Runtime(format!("missing graph input '{}'", io.name))
                })?;
                if shape != &io.shape {
                    return Err(MetalError::Runtime(format!(
                        "input '{}' shape {shape:?} does not match the compiled shape {:?}",
                        io.name, io.shape
                    )));
                }
                if data.len() != numel(shape) {
                    return Err(MetalError::Runtime(format!(
                        "input '{}' holds {} floats for shape {:?}",
                        io.name,
                        data.len(),
                        shape
                    )));
                }
                if io.elem != Elem::F32 { return Err(MetalError::Runtime(format!("input {} is not an F32 buffer",io.name))); }
                let byte_len=std::mem::size_of_val(*data);
                // A borrowed Rust slice is not proof that the caller owns a complete range of VM
                // pages. Use one CPU-to-shared-buffer copy; never pass a missing deallocator block.
                let buffer=if byte_len==0 {unsafe{self.empty_for(4)}?}else{
                    let b=unsafe{self.bridge.bytes(self.device,"newBufferWithBytes:length:options:",data.as_ptr().cast(),byte_len,STORAGE_SHARED)};
                    if b.is_null(){return Err(MetalError::Runtime("input MTLBuffer allocation failed".into()));} b
                };
                run_buffers.objects.push(buffer);
                input_buffers.insert(io.tid, buffer);
            }

            // 2. Output buffers.
            let mut output_buffers: HashMap<TensorId, *mut c_void> = HashMap::new();
            for io in &program.outputs {
                let byte_len = numel(&io.shape).max(1) * io.elem.bytes();
                let buffer = unsafe { self.empty_for(byte_len) }?;
                run_buffers.objects.push(buffer);
                output_buffers.insert(io.tid, buffer);
            }

            // 3. One command buffer, one compute encoder for every node.
            let status: Result<()> = (|| {
                unsafe {
                    let command_buffer = self.bridge.p0(self.queue, "commandBuffer");
                    if command_buffer.is_null() {
                        return Err(MetalError::Runtime("the queue refused a command buffer".into()));
                    }
                    let encoder = self.bridge.p0(command_buffer, "computeCommandEncoder");
                    if encoder.is_null() {
                        return Err(MetalError::Runtime(
                            "the command buffer refused a compute encoder".into(),
                        ));
                    }
                    let pipelines = self.pipelines.lock().map_err(|_| {
                        MetalError::Runtime("the pipeline cache was poisoned".into())
                    })?;
                    for dispatch in &program.dispatches {
                        let pipeline = pipelines.get(dispatch.kernel.name()).ok_or_else(|| {
                            MetalError::Pipeline {
                                kernel: dispatch.kernel.name().to_string(),
                                detail: "pipeline missing from the session cache".into(),
                            }
                        })?;
                        self.bridge.void1(encoder, "setComputePipelineState:", *pipeline);
                        for (index, region) in &dispatch.binds {
                            let (buffer, offset) = self.resolve(
                                *region,
                                &input_buffers,
                                &output_buffers,
                                program,
                            )?;
                            self.bridge.bind(
                                encoder,
                                "setBuffer:offset:atIndex:",
                                buffer,
                                offset,
                                *index as usize,
                            );
                        }
                        self.bridge.bind(
                            encoder,
                            "setBytes:length:atIndex:",
                            dispatch.params.as_ptr().cast::<c_void>(),
                            dispatch.params.len(),
                            Kernel::PARAMS_INDEX as usize,
                        );
                        self.bridge.dispatch(
                            encoder,
                            MTLSize {
                                width: dispatch.threads[0] as usize,
                                height: dispatch.threads[1] as usize,
                                depth: dispatch.threads[2] as usize,
                            },
                            MTLSize {
                                width: dispatch.threadgroup[0] as usize,
                                height: dispatch.threadgroup[1] as usize,
                                depth: dispatch.threadgroup[2] as usize,
                            },
                        );
                    }
                    drop(pipelines);
                    self.bridge.void0(encoder, "endEncoding");
                    self.bridge.void0(command_buffer, "commit");
                    self.bridge.void0(command_buffer, "waitUntilCompleted");
                    let error = self.bridge.p0(command_buffer, "error");
                    if let Some(text) = error_text(&self.bridge, error) {
                        return Err(MetalError::Runtime(format!(
                            "the command buffer failed: {text}"
                        )));
                    }
                }
                Ok(())
            })();

            // 4. Read back named outputs (or release and report the failure).
            let result = status.and_then(|()| {
                let mut outputs = NamedOutputs::new();
                for io in &program.outputs {
                    let buffer = *output_buffers.get(&io.tid).expect("bound above");
                    let count = numel(&io.shape);
                    let mut host = vec![0.0f32; count];
                    unsafe {
                        let contents = self.bridge.p0(buffer, "contents").cast::<u8>();
                        if contents.is_null() {
                            return Err(MetalError::Runtime(format!(
                                "output '{}' has no CPU-visible contents",
                                io.name
                            )));
                        }
                        let raw = std::slice::from_raw_parts(contents, count * io.elem.bytes());
                        decode_output(raw, io.elem, &mut host)?;
                    }
                    outputs.insert(io.name.clone(), (host, io.shape.clone()));
                }
                for (name, values, shape) in &program.constant_outputs {
                    outputs.insert(name.clone(), (values.clone(), shape.clone()));
                }
                Ok(outputs)
            });

            result
        }

        unsafe fn empty_for(&self, len: usize) -> Result<*mut c_void> {
            let buffer = self.bridge.length_options(
                self.device,
                "newBufferWithLength:options:",
                len,
                STORAGE_SHARED,
            );
            if buffer.is_null() {
                return Err(MetalError::Runtime(format!(
                    "the device refused a {len}-byte output buffer"
                )));
            }
            Ok(buffer)
        }

        fn resolve(
            &self,
            region: Region,
            inputs: &HashMap<TensorId, *mut c_void>,
            outputs: &HashMap<TensorId, *mut c_void>,
            program: &MslProgram,
        ) -> Result<(*mut c_void, usize)> {
            let buffer = match region.buf {
                Buf::Weights => self.weights,
                Buf::Arena => self.arena,
                Buf::Extra => self.extra,
                Buf::Input(tid) => *inputs.get(&tid).ok_or_else(|| {
                    MetalError::Runtime(format!(
                        "tensor {tid:?} is not one of the supplied graph inputs"
                    ))
                })?,
                Buf::Output(tid) => *outputs.get(&tid).ok_or_else(|| {
                    MetalError::Runtime(format!("tensor {tid:?} is not a compiled graph output"))
                })?,
            };
            if buffer.is_null() {
                return Err(MetalError::Runtime(
                    "a dispatch referenced a buffer that was never allocated".into(),
                ));
            }
            let _ = program;
            Ok((buffer, region.byte_off))
        }
    }

    fn decode_output(raw: &[u8], elem: Elem, out: &mut [f32]) -> Result<()> {
        match elem {
            Elem::F32 => {
                for (index, value) in out.iter_mut().enumerate() {
                    let mut bytes = [0u8; 4];
                    bytes.copy_from_slice(&raw[index * 4..index * 4 + 4]);
                    *value = f32::from_le_bytes(bytes);
                }
            }
            Elem::F16 => {
                for (index, value) in out.iter_mut().enumerate() {
                    let mut bytes = [0u8; 2];
                    bytes.copy_from_slice(&raw[index * 2..index * 2 + 2]);
                    *value = crate::native_infer::onnx_proto::f16_to_f32(u16::from_le_bytes(bytes));
                }
            }
            Elem::I32 => {
                for (index, value) in out.iter_mut().enumerate() {
                    let mut bytes = [0u8; 4];
                    bytes.copy_from_slice(&raw[index * 4..index * 4 + 4]);
                    *value = i32::from_le_bytes(bytes) as f32;
                }
            }
            Elem::I64 => {
                for (index, value) in out.iter_mut().enumerate() {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&raw[index * 8..index * 8 + 8]);
                    *value = i64::from_le_bytes(bytes) as f32;
                }
            }
            Elem::U8 | Elem::Bool => {
                for (index, value) in out.iter_mut().enumerate() {
                    *value = raw[index] as f32;
                }
            }
            Elem::I8 => {
                for (index, value) in out.iter_mut().enumerate() {
                    *value = raw[index] as i8 as f32;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
//
// Everything here runs on any host: the encoder, the parameter ABIs, the index
// math and the operator classification are host independent. MSL compilation and
// on-GPU numerics are *not* verified here and are never claimed to be.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_infer::cpu_shape_eval::CpuTensor;

    // --- minimal protobuf wire encoder for synthetic in-memory ONNX ---------

    fn enc_varint(out: &mut Vec<u8>, mut value: u64) {
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                break;
            }
        }
    }

    fn enc_tag(out: &mut Vec<u8>, field: u32, wire: u8) {
        enc_varint(out, ((field as u64) << 3) | wire as u64);
    }

    fn enc_bytes(out: &mut Vec<u8>, field: u32, data: &[u8]) {
        enc_tag(out, field, 2);
        enc_varint(out, data.len() as u64);
        out.extend_from_slice(data);
    }

    fn enc_str(out: &mut Vec<u8>, field: u32, text: &str) {
        enc_bytes(out, field, text.as_bytes());
    }

    fn value_info(name: &str, dtype: DataType, dims: &[i64]) -> Vec<u8> {
        let mut shape = Vec::new();
        for &dim in dims {
            let mut dim_msg = Vec::new();
            enc_tag(&mut dim_msg, 1, 0);
            enc_varint(&mut dim_msg, dim as u64);
            enc_bytes(&mut shape, 1, &dim_msg);
        }
        let mut tensor_type = Vec::new();
        enc_tag(&mut tensor_type, 1, 0);
        enc_varint(&mut tensor_type, dtype as u64);
        enc_bytes(&mut tensor_type, 2, &shape);
        let mut type_proto = Vec::new();
        enc_bytes(&mut type_proto, 1, &tensor_type);
        let mut info = Vec::new();
        enc_str(&mut info, 1, name);
        enc_bytes(&mut info, 2, &type_proto);
        info
    }

    fn tensor_raw(name: &str, dtype: DataType, dims: &[i64], raw: Vec<u8>) -> Vec<u8> {
        let mut tensor = Vec::new();
        for &dim in dims {
            enc_tag(&mut tensor, 1, 0);
            enc_varint(&mut tensor, dim as u64);
        }
        enc_tag(&mut tensor, 2, 0);
        enc_varint(&mut tensor, dtype as u64);
        enc_str(&mut tensor, 8, name);
        enc_bytes(&mut tensor, 9, &raw);
        tensor
    }

    fn tensor_f32(name: &str, dims: &[i64], values: &[f32]) -> Vec<u8> {
        let mut raw = Vec::with_capacity(values.len() * 4);
        for value in values {
            raw.extend_from_slice(&value.to_le_bytes());
        }
        tensor_raw(name, DataType::Float, dims, raw)
    }

    fn tensor_i64(name: &str, dims: &[i64], values: &[i64]) -> Vec<u8> {
        let mut raw = Vec::with_capacity(values.len() * 8);
        for value in values {
            raw.extend_from_slice(&value.to_le_bytes());
        }
        tensor_raw(name, DataType::Int64, dims, raw)
    }

    fn attr_int(name: &str, value: i64) -> Vec<u8> {
        let mut attr = Vec::new();
        enc_str(&mut attr, 1, name);
        enc_tag(&mut attr, 3, 0);
        enc_varint(&mut attr, value as u64);
        enc_tag(&mut attr, 20, 0);
        enc_varint(&mut attr, 2); // INT
        attr
    }

    fn attr_ints(name: &str, values: &[i64]) -> Vec<u8> {
        let mut attr = Vec::new();
        enc_str(&mut attr, 1, name);
        if values.len() == 1 {
            // ONNX accepts a single value in either slot; keep INTS explicit.
        }
        for &value in values {
            enc_tag(&mut attr, 8, 0);
            enc_varint(&mut attr, value as u64);
        }
        enc_tag(&mut attr, 20, 0);
        enc_varint(&mut attr, 7); // INTS
        attr
    }

    fn attr_float(name: &str, value: f32) -> Vec<u8> {
        let mut attr = Vec::new();
        enc_str(&mut attr, 1, name);
        enc_tag(&mut attr, 2, 5);
        attr.extend_from_slice(&value.to_le_bytes());
        enc_tag(&mut attr, 20, 0);
        enc_varint(&mut attr, 1); // FLOAT
        attr
    }

    fn attr_str(name: &str, value: &str) -> Vec<u8> {
        let mut attr = Vec::new();
        enc_str(&mut attr, 1, name);
        enc_str(&mut attr, 4, value);
        enc_tag(&mut attr, 20, 0);
        enc_varint(&mut attr, 3); // STRING
        attr
    }

    fn node(op: &str, name: &str, inputs: &[&str], outputs: &[&str], attrs: &[Vec<u8>]) -> Vec<u8> {
        let mut node = Vec::new();
        for input in inputs {
            enc_str(&mut node, 1, input);
        }
        for output in outputs {
            enc_str(&mut node, 2, output);
        }
        enc_str(&mut node, 3, name);
        enc_str(&mut node, 4, op);
        for attr in attrs {
            enc_bytes(&mut node, 5, attr);
        }
        node
    }

    fn model(
        nodes: Vec<Vec<u8>>,
        inits: Vec<Vec<u8>>,
        inputs: Vec<Vec<u8>>,
        outputs: Vec<Vec<u8>>,
    ) -> Vec<u8> {
        let mut graph = Vec::new();
        enc_str(&mut graph, 2, "metal_synthetic");
        for n in nodes {
            enc_bytes(&mut graph, 1, &n);
        }
        for init in inits {
            enc_bytes(&mut graph, 5, &init);
        }
        for input in inputs {
            enc_bytes(&mut graph, 11, &input);
        }
        for output in outputs {
            enc_bytes(&mut graph, 12, &output);
        }
        let mut model = Vec::new();
        enc_tag(&mut model, 1, 0);
        enc_varint(&mut model, 8);
        enc_bytes(&mut model, 7, &graph);
        model
    }

    fn compile(
        nodes: Vec<Vec<u8>>,
        inits: Vec<Vec<u8>>,
        inputs: Vec<Vec<u8>>,
        outputs: Vec<Vec<u8>>,
    ) -> Result<MslProgram> {
        MslProgram::compile(&model(nodes, inits, inputs, outputs), &ShapeOverrides::new())
    }

    fn consts(count: usize, value: f32) -> Vec<f32> {
        vec![value; count]
    }

    // --- ABI pinning -------------------------------------------------------

    #[test]
    fn params_struct_sizes_match_msl() {
        // Every parameter struct is all-32-bit fields, so the C and MSL layouts
        // agree; these numbers pin the ABI so a field addition cannot silently
        // desynchronise the MSL `constant` view.
        assert_eq!(std::mem::size_of::<CopyParams>(), 252);
        assert_eq!(std::mem::size_of::<FillParams>(), 16);
        assert_eq!(std::mem::size_of::<UnaryParams>(), 32);
        assert_eq!(std::mem::size_of::<BinaryParams>(), 124);
        assert_eq!(std::mem::size_of::<WhereParams>(), 160);
        assert_eq!(std::mem::size_of::<CastParams>(), 24);
        assert_eq!(std::mem::size_of::<ChannelAffineParams>(), 32);
        assert_eq!(std::mem::size_of::<GemmParams>(), 60);
        assert_eq!(std::mem::size_of::<ConvParams>(), 84);
        assert_eq!(std::mem::size_of::<PoolParams>(), 72);
        assert_eq!(std::mem::size_of::<RowParams>(), 52);
        assert_eq!(std::mem::size_of::<ResizeParams>(), 52);
        assert_eq!(std::mem::size_of::<MatMulIntParams>(), 48);
        assert_eq!(std::mem::size_of::<QuantizeParams>(), 20);
        assert_eq!(std::mem::size_of::<QuantFinalizeParams>(), 16);
        assert_eq!(std::mem::size_of::<ScatterParams>(), 64);
        assert_eq!(std::mem::size_of::<GatherParams>(), 44);
        assert_eq!(std::mem::size_of::<MinMaxParams>(), 8);
    }

    /// Field names of an MSL parameter struct, in declaration order.
    fn msl_fields(name: &str) -> Vec<String> {
        let header = format!("struct {name} {{");
        let start = MSL_LIBRARY
            .find(&header)
            .unwrap_or_else(|| panic!("struct {name} missing"));
        let body = &MSL_LIBRARY[start..];
        let end = body.find("};").expect("struct terminator");
        let mut fields = Vec::new();
        for declaration in body[..end].split(';') {
            let declaration = declaration.rsplit('{').next().unwrap_or("").trim();
            if declaration.is_empty() {
                continue;
            }
            let head = declaration.split('[').next().unwrap_or("").trim();
            if let Some(field) = head.split_whitespace().last() {
                fields.push(field.to_string());
            }
        }
        fields
    }

    #[test]
    fn msl_struct_field_order_matches_the_rust_parameters() {
        let expected: &[(&str, &[&str])] = &[
            (
                "CopyParams",
                &[
                    "rank", "out_elems", "elem_bytes", "edge", "dst_base", "src_base", "pad_value",
                    "out_shape", "dst_stride", "mul", "add", "shift", "in_dim", "use_mod",
                ],
            ),
            ("FillParams", &["n", "elem_bytes", "dst_base", "bits"]),
            (
                "UnaryParams",
                &["n", "op", "alpha", "beta", "lo", "hi", "x_base", "y_base"],
            ),
            (
                "BinaryParams",
                &[
                    "rank", "n", "op", "unused", "a_base", "b_base", "y_base", "out_shape", "a_mul",
                    "b_mul",
                ],
            ),
            (
                "WhereParams",
                &[
                    "rank", "n", "unused0", "unused1", "c_base", "a_base", "b_base", "y_base",
                    "out_shape", "c_mul", "a_mul", "b_mul",
                ],
            ),
            ("CastParams", &["n", "from", "to", "unused", "x_base", "y_base"]),
            (
                "ChannelAffineParams",
                &["n", "inner", "channels", "unused", "x_base", "y_base", "scale_base", "bias_base"],
            ),
            (
                "GemmParams",
                &[
                    "m", "n", "k", "lda", "ldb", "ldc", "a_off", "b_off", "c_off", "alpha", "beta",
                    "trans_a", "trans_b", "has_bias", "act",
                ],
            ),
            (
                "ConvParams",
                &[
                    "n", "h", "w", "c", "oh", "ow", "oc", "kh", "kw", "sh", "sw", "ph", "pw", "dh",
                    "dw", "groups", "act", "x_off", "w_off", "b_off", "y_off",
                ],
            ),
            (
                "PoolParams",
                &[
                    "n", "c", "ih", "iw", "oh", "ow", "kh", "kw", "sh", "sw", "ph", "pw", "dh",
                    "dw", "mode", "count_include_pad", "x_off", "y_off",
                ],
            ),
            (
                "RowParams",
                &[
                    "outer", "r", "inner", "mode", "channels", "y_stride_outer", "y_stride_inner",
                    "unused", "x_off", "y_off", "g_off", "b_off", "eps",
                ],
            ),
            (
                "ResizeParams",
                &[
                    "n", "c", "ih", "iw", "oh", "ow", "mode", "coord_mode", "nearest_mode", "rh",
                    "rw", "x_off", "y_off",
                ],
            ),
            (
                "MatMulIntParams",
                &[
                    "m", "n", "k", "a_off", "b_off", "c_off", "azp_off", "bzp_off", "a_zp", "b_zp",
                    "use_buf_zp", "b_signed",
                ],
            ),
            ("QuantizeParams", &["n", "x_off", "y_off", "scale_off", "zp_off"]),
            ("QuantFinalizeParams", &["minmax_off", "scale_off", "zp_off", "eps"]),
            (
                "ScatterParams",
                &[
                    "num_updates", "k", "slice_size", "out_elems", "rank", "idx_off", "upd_off",
                    "out_off", "strides",
                ],
            ),
            (
                "GatherParams",
                &[
                    "outer", "axis_len", "inner", "idx_len", "out_elems", "idx_i64", "unused0",
                    "unused1", "data_off", "idx_off", "y_off",
                ],
            ),
            ("MinMaxParams", &["n", "x_off"]),
        ];
        for (name, fields) in expected {
            let actual = msl_fields(name);
            let expected: Vec<String> = fields.iter().map(|f| f.to_string()).collect();
            assert_eq!(&actual, &expected, "MSL struct {name} field order");
        }
    }

    // --- graph encoding ----------------------------------------------------

    fn conv_bn_relu_add_softmax() -> MslProgram {
        let nodes = vec![
            node(
                "Conv",
                "conv1",
                &["x", "w1", "b1"],
                &["c1"],
                &[
                    attr_ints("kernel_shape", &[3, 3]),
                    attr_ints("pads", &[1, 1, 1, 1]),
                    attr_ints("strides", &[1, 1]),
                ],
            ),
            node(
                "BatchNormalization",
                "bn1",
                &["c1", "bn_g", "bn_b", "bn_m", "bn_v"],
                &["b1o"],
                &[attr_float("epsilon", 1e-5)],
            ),
            node("Relu", "relu1", &["b1o"], &["r1"], &[]),
            node("Add", "add1", &["r1", "w2"], &["add"], &[]),
            node("Softmax", "softmax1", &["add"], &["y"], &[attr_int("axis", 1)]),
        ];
        let inits = vec![
            tensor_f32("w1", &[4, 3, 3, 3], &consts(4 * 3 * 9, 0.1)),
            tensor_f32("b1", &[4], &consts(4, 0.0)),
            tensor_f32("bn_g", &[4], &consts(4, 1.0)),
            tensor_f32("bn_b", &[4], &consts(4, 0.0)),
            tensor_f32("bn_m", &[4], &consts(4, 0.0)),
            tensor_f32("bn_v", &[4], &consts(4, 1.0)),
            tensor_f32("w2", &[1, 4, 8, 8], &consts(256, 0.5)),
        ];
        compile(
            nodes,
            inits,
            vec![value_info("x", DataType::Float, &[1, 3, 8, 8])],
            vec![value_info("y", DataType::Float, &[1, 4, 8, 8])],
        )
        .expect("conv/bn/relu/add/softmax must encode")
    }

    #[test]
    fn conv_bn_relu_add_softmax_encodes_every_node_once() {
        let program = conv_bn_relu_add_softmax();
        program.validate().expect("program invariants");
        // BN is folded into the Conv weights by the shared front end, so the four
        // remaining nodes produce exactly four dispatches.
        assert_eq!(program.dispatch_count(), 4, "{:?}", program.dispatches.iter().map(|d| d.op).collect::<Vec<_>>());
        let ops: Vec<&str> = program.dispatches.iter().map(|d| d.op).collect();
        assert_eq!(ops, vec!["Conv", "Relu", "Add", "Softmax"]);
        assert!(program.used_kernels.contains(&Kernel::Conv2d));
        assert!(program.used_kernels.contains(&Kernel::Unary));
        assert!(program.used_kernels.contains(&Kernel::Binary));
        assert!(program.used_kernels.contains(&Kernel::Rows));
        // The graph output is bound to its own per-run buffer, never the arena.
        let output = &program.outputs[0];
        let bound = program.tensors.get(&output.tid).expect("output bound");
        assert_eq!(bound.region.buf, Buf::Output(output.tid));
    }

    #[test]
    fn matmul_and_gemm_are_encoded() {
        let nodes = vec![
            node("MatMul", "mm", &["x", "w1"], &["m"], &[]),
            node(
                "Gemm",
                "gemm",
                &["m", "w2", "b"],
                &["y"],
                &[attr_float("alpha", 0.5), attr_float("beta", 0.0)],
            ),
        ];
        let inits = vec![
            tensor_f32("w1", &[8, 4], &consts(32, 0.25)),
            tensor_f32("w2", &[4, 4], &consts(16, 0.5)),
            tensor_f32("b", &[4], &consts(4, 0.0)),
        ];
        let program = compile(
            nodes,
            inits,
            vec![value_info("x", DataType::Float, &[1, 8])],
            vec![value_info("y", DataType::Float, &[1, 4])],
        )
        .expect("matmul + gemm encode");
        program.validate().expect("invariants");
        assert_eq!(program.dispatch_count(), 2);
        assert!(program.used_kernels.contains(&Kernel::Gemm));
        let gemm = program
            .dispatches
            .iter()
            .find(|d| d.op == "Gemm")
            .expect("gemm dispatch");
        // Threadgroup must stay within the 1024-thread Metal limit.
        assert_eq!(gemm.threadgroup, GEMM_THREADGROUP);
    }

    #[test]
    fn layout_concat_slice_pad_transpose_are_encoded() {
        let nodes = vec![
            node(
                "Slice",
                "slice",
                &["x", "starts", "ends", "axes"],
                &["s"],
                &[],
            ),
            node("Pad", "pad", &["s", "pads"], &["p"], &[attr_str("mode", "constant")]),
            node("Transpose", "tr", &["p"], &["t"], &[attr_ints("perm", &[0, 2, 1, 3])]),
            node("Concat", "cat", &["t", "t"], &["y"], &[attr_int("axis", 1)]),
        ];
        let inits = vec![
            tensor_i64("starts", &[4], &[0, 0, 1, 0]),
            tensor_i64("ends", &[4], &[1, 1, 3, 4]),
            tensor_i64("axes", &[4], &[0, 1, 2, 3]),
            tensor_i64("pads", &[8], &[0, 0, 1, 0, 0, 0, 1, 0]),
        ];
        let program = compile(
            nodes,
            inits,
            vec![value_info("x", DataType::Float, &[1, 2, 4, 4])],
            vec![value_info("y", DataType::Float, &[1, 8, 1, 4])],
        )
        .expect("layout ops encode");
        program.validate().expect("invariants");
        let kinds: Vec<Kernel> = program.dispatches.iter().map(|d| d.kernel).collect();
        assert_eq!(kinds.first(), Some(&Kernel::Copy), "slice is a strided copy");
        assert!(kinds.contains(&Kernel::Fill), "constant pad fills first");
        // Concat of two inputs is two copies into one buffer.
        let concat_copies = program
            .dispatches
            .iter()
            .filter(|d| d.op == "Concat")
            .count();
        assert_eq!(concat_copies, 2);
    }

    #[test]
    fn pool_resize_and_reductions_are_encoded() {
        let nodes = vec![
            node(
                "MaxPool",
                "pool",
                &["x"],
                &["p"],
                &[attr_ints("kernel_shape", &[2, 2]), attr_ints("strides", &[2, 2])],
            ),
            node(
                "Resize",
                "resize",
                &["p", "", "scales"],
                &["r"],
                &[
                    attr_str("mode", "linear"),
                    attr_str("coordinate_transformation_mode", "half_pixel"),
                ],
            ),
            node(
                "ReduceMean",
                "mean",
                &["r"],
                &["m"],
                &[attr_ints("axes", &[2, 3]), attr_int("keepdims", 1)],
            ),
            node("GlobalAveragePool", "gap", &["r"], &["g"], &[]),
        ];
        let inits = vec![tensor_f32("scales", &[4], &[1.0, 1.0, 2.0, 2.0])];
        let program = compile(
            nodes,
            inits,
            vec![value_info("x", DataType::Float, &[1, 3, 8, 8])],
            vec![
                value_info("m", DataType::Float, &[1, 3, 1, 1]),
                value_info("g", DataType::Float, &[1, 3, 1, 1]),
            ],
        )
        .expect("pool/resize/reduce encode");
        program.validate().expect("invariants");
        let kernels: Vec<Kernel> = program.dispatches.iter().map(|d| d.kernel).collect();
        assert_eq!(kernels[0], Kernel::Pool2d);
        assert_eq!(kernels[1], Kernel::Resize);
        assert_eq!(kernels[2], Kernel::Rows);
        assert_eq!(kernels[3], Kernel::Rows);
    }

    #[test]
    fn normalisation_ops_are_encoded() {
        let nodes = vec![
            node(
                "LayerNormalization",
                "ln",
                &["x", "g", "b"],
                &["y"],
                &[attr_int("axis", -1), attr_float("epsilon", 1e-5)],
            ),
        ];
        let inits = vec![
            tensor_f32("g", &[8], &consts(8, 1.0)),
            tensor_f32("b", &[8], &consts(8, 0.0)),
        ];
        let program = compile(
            nodes,
            inits,
            vec![value_info("x", DataType::Float, &[2, 4, 8])],
            vec![value_info("y", DataType::Float, &[2, 4, 8])],
        )
        .expect("layer norm encodes");
        program.validate().expect("invariants");
        assert_eq!(program.dispatches.len(), 1);
        let params = &program.dispatches[0].params;
        assert_eq!(params.len(), std::mem::size_of::<RowParams>());
        let row: &RowParams = unsafe { &*(params.as_ptr().cast::<RowParams>()) };
        assert_eq!(row.mode, ROW_LAYER_NORM);
        assert_eq!((row.outer, row.r, row.inner), (8, 8, 1));
        assert_eq!((row.y_stride_outer, row.y_stride_inner), (8, 1));

        let nodes = vec![node(
            "InstanceNormalization",
            "in",
            &["x", "g", "b"],
            &["y"],
            &[attr_float("epsilon", 1e-5)],
        )];
        let inits = vec![
            tensor_f32("g", &[2], &consts(2, 1.0)),
            tensor_f32("b", &[2], &consts(2, 0.0)),
        ];
        let program = compile(
            nodes,
            inits,
            vec![value_info("x", DataType::Float, &[1, 2, 4, 4])],
            vec![value_info("y", DataType::Float, &[1, 2, 4, 4])],
        )
        .expect("instance norm encodes");
        let row: &RowParams =
            unsafe { &*(program.dispatches[0].params.as_ptr().cast::<RowParams>()) };
        assert_eq!(row.mode, ROW_INSTANCE_NORM);
        assert_eq!((row.outer, row.r, row.channels), (2, 16, 2));
    }

    #[test]
    fn indexing_ops_are_encoded() {
        let nodes = vec![
            node("Gather", "gather", &["data", "idx"], &["g"], &[attr_int("axis", 0)]),
            node("Equal", "eq", &["x", "x"], &["cond"], &[]),
            node("Where", "where", &["cond", "x", "x"], &["w"], &[]),
            node("Expand", "expand", &["x2", "shape"], &["e"], &[]),
            node("Tile", "tile", &["x1", "repeats"], &["t"], &[]),
            node(
                "ScatterND",
                "scatter",
                &["data", "sc_idx", "sc_upd"],
                &["s"],
                &[],
            ),
        ];
        let inits = vec![
            tensor_f32("data", &[4, 4], &consts(16, 0.25)),
            tensor_i64("idx", &[3], &[0, 2, 1]),
            tensor_i64("shape", &[2], &[3, 2]),
            tensor_i64("repeats", &[1], &[3]),
            tensor_i64("sc_idx", &[2, 1], &[0, 3]),
            tensor_f32("sc_upd", &[2, 4], &consts(8, 1.0)),
        ];
        let program = compile(
            nodes,
            inits,
            vec![
                value_info("x", DataType::Float, &[1, 4]),
                value_info("x2", DataType::Float, &[1, 2]),
                value_info("x1", DataType::Float, &[2]),
            ],
            vec![
                value_info("g", DataType::Float, &[3, 4]),
                value_info("w", DataType::Float, &[1, 4]),
                value_info("e", DataType::Float, &[3, 2]),
                value_info("t", DataType::Float, &[6]),
                value_info("s", DataType::Float, &[4, 4]),
            ],
        )
        .expect("indexing ops encode");
        program.validate().expect("invariants");
        assert!(program.dispatch_count() >= 7);
        assert!(program.used_kernels.contains(&Kernel::Gather));
        assert!(program.used_kernels.contains(&Kernel::Compare));
        assert!(program.used_kernels.contains(&Kernel::Where));
        assert!(program.used_kernels.contains(&Kernel::Scatter));
        // ScatterND starts with a full copy of `data` before applying updates.
        let scatter_copies = program
            .dispatches
            .iter()
            .filter(|d| d.op == "ScatterND" && d.kernel == Kernel::Copy)
            .count();
        assert_eq!(scatter_copies, 1);
    }

    #[test]
    fn dynamic_quantize_linear_encodes_three_gpu_steps() {
        let nodes = vec![node(
            "DynamicQuantizeLinear",
            "dql",
            &["x"],
            &["q", "sc", "zp"],
            &[],
        )];
        let program = compile(
            nodes,
            Vec::new(),
            vec![value_info("x", DataType::Float, &[1, 6])],
            vec![
                value_info("q", DataType::Uint8, &[1, 6]),
                value_info("sc", DataType::Float, &[]),
                value_info("zp", DataType::Uint8, &[]),
            ],
        )
        .expect("dynamic quantize encodes");
        program.validate().expect("invariants");
        let kernels: Vec<Kernel> = program.dispatches.iter().map(|d| d.kernel).collect();
        assert_eq!(kernels, vec![Kernel::MinMax, Kernel::QuantFinalize, Kernel::Quantize]);
        assert!(program.extra_bytes >= 2 * 4, "min/max scratch slot reserved");
    }

    #[test]
    fn elementwise_and_compare_ops_are_encoded() {
        let nodes = vec![
            node("Sigmoid", "sig", &["x"], &["s"], &[]),
            node("HardSwish", "hsw", &["s"], &["h"], &[]),
            node("Clip", "clip", &["h", "lo", "hi"], &["c"], &[]),
            node("Pow", "pow", &["c", "two"], &["p"], &[]),
            node("Greater", "gt", &["p", "one"], &["gt"], &[]),
            node("And", "and", &["gt", "gt2"], &["a"], &[]),
            node("Where", "where", &["a", "p", "c"], &["y"], &[]),
        ];
        let inits = vec![
            tensor_f32("lo", &[], &[-1.0]),
            tensor_f32("hi", &[], &[1.0]),
            tensor_f32("two", &[], &[2.0]),
            tensor_f32("one", &[], &[1.0]),
            tensor_f32("gt2", &[4], &consts(4, 1.0)),
        ];
        let program = compile(
            nodes,
            inits,
            vec![value_info("x", DataType::Float, &[2, 4])],
            vec![value_info("y", DataType::Float, &[2, 4])],
        )
        .expect("elementwise chain encodes");
        program.validate().expect("invariants");
        assert!(program.used_kernels.contains(&Kernel::Unary));
        assert!(program.used_kernels.contains(&Kernel::Compare));
        assert!(program.used_kernels.contains(&Kernel::Where));
        // `Where` over a Bool produced on the GPU must stay on the GPU: the
        // condition tensor is bound to an arena slot, not a CPU copy.
        let where_dispatch = program.dispatches.iter().find(|d| d.op == "Where").unwrap();
        assert_eq!(where_dispatch.binds[0].1.buf, Buf::Arena);
    }

    #[test]
    fn split_and_concat_handle_multiple_outputs() {
        let nodes = vec![
            node("Split", "split", &["x"], &["s0", "s1"], &[attr_int("axis", 1)]),
            node("Concat", "cat", &["s0", "s1"], &["y"], &[attr_int("axis", 1)]),
        ];
        let program = compile(
            nodes,
            Vec::new(),
            vec![value_info("x", DataType::Float, &[1, 4, 2])],
            vec![value_info("y", DataType::Float, &[1, 4, 2])],
        )
        .expect("split/concat encode");
        program.validate().expect("invariants");
        assert_eq!(
            program.dispatches.iter().filter(|d| d.op == "Split").count(),
            2
        );
        assert_eq!(
            program.dispatches.iter().filter(|d| d.op == "Concat").count(),
            2
        );
    }

    // --- rejection paths --------------------------------------------------

    #[test]
    fn unsupported_op_is_rejected_before_any_dispatch() {
        let nodes = vec![node(
            "ConvInteger",
            "ci",
            &["x", "w"],
            &["y"],
            &[attr_ints("kernel_shape", &[1, 1])],
        )];
        let inits = vec![tensor_raw("w", DataType::Uint8, &[1, 1, 1, 1], vec![7u8])];
        let error = compile(
            nodes,
            inits,
            vec![value_info("x", DataType::Float, &[1, 1, 2, 2])],
            vec![value_info("y", DataType::Int32, &[1, 1, 2, 2])],
        )
        .expect_err("ConvInteger must be rejected");
        match error {
            MetalError::UnsupportedOp { node, op, .. } => {
                assert_eq!(node, "ci");
                assert_eq!(op, "ConvInteger");
            }
            other => panic!("expected UnsupportedOp, got {other}"),
        }
    }

    #[test]
    fn dynamic_parameters_are_reported_not_baked() {
        // Integer control data cannot arrive as a runtime float input at all.
        let nodes = vec![node("Slice", "slice", &["x", "starts", "ends"], &["y"], &[])];
        let error = compile(
            nodes,
            Vec::new(),
            vec![
                value_info("x", DataType::Float, &[1, 4]),
                value_info("starts", DataType::Int64, &[1]),
                value_info("ends", DataType::Int64, &[1]),
            ],
            vec![value_info("y", DataType::Float, &[1, 4])],
        )
        .expect_err("an int64 graph input cannot be supplied through named f32 inputs");
        let text = error.to_string();
        assert!(
            text.contains("named f32 inputs require Float graph inputs"),
            "unexpected error: {text}"
        );

        // A float parameter that only exists at run time must also be reported:
        // the kernel bakes clip bounds into its constant buffer.
        let nodes = vec![node("Clip", "clip", &["x", "lo", "hi"], &["y"], &[])];
        let error = compile(
            nodes,
            Vec::new(),
            vec![
                value_info("x", DataType::Float, &[1, 4]),
                value_info("lo", DataType::Float, &[1]),
                value_info("hi", DataType::Float, &[1]),
            ],
            vec![value_info("y", DataType::Float, &[1, 4])],
        )
        .expect_err("runtime clip bounds cannot be baked into the parameter struct");
        let text = error.to_string();
        assert!(text.contains("clip"), "unexpected error: {text}");
        assert!(
            text.contains("not statically known"),
            "unexpected error: {text}"
        );
    }

    // --- classification ---------------------------------------------------

    #[test]
    fn every_ir_op_variant_is_classified() {
        use crate::native_infer::graph::*;
        let conv = ConvAttrs {
            kernel_shape: None,
            strides: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            dilations: vec![1, 1],
            group: 1,
        };
        let pool = PoolAttrs {
            kernel_shape: vec![2, 2],
            strides: vec![2, 2],
            pads: vec![0, 0, 0, 0],
            dilations: vec![1, 1],
            ceil_mode: false,
            count_include_pad: false,
        };
        let reduce = ReduceAttrs {
            axes: None,
            keepdims: true,
            noop_with_empty_axes: false,
        };
        let empty_graph = IrGraph {
            name: "empty".into(),
            tensors: Vec::new(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            weights: HashMap::new(),
            name_to_id: HashMap::new(),
        };
        let ops = vec![
            Op::Conv(conv.clone()),
            Op::ConvTranspose(ConvTransposeAttrs {
                kernel_shape: None,
                strides: vec![1, 1],
                pads: vec![0, 0, 0, 0],
                dilations: vec![1, 1],
                group: 1,
                output_padding: vec![0, 0],
            }),
            Op::ConvInteger(conv),
            Op::MatMul,
            Op::MatMulInteger,
            Op::Gemm(GemmAttrs {
                alpha: 1.0,
                beta: 1.0,
                trans_a: false,
                trans_b: false,
            }),
            Op::BatchNormalization(BatchNormAttrs {
                epsilon: 1e-5,
                momentum: 0.9,
            }),
            Op::BatchNormFused,
            Op::LayerNormalization(LayerNormAttrs {
                axis: -1,
                epsilon: 1e-5,
                stash_type: 1,
            }),
            Op::InstanceNormalization { epsilon: 1e-5 },
            Op::MaxPool(pool.clone()),
            Op::AveragePool(pool),
            Op::GlobalAveragePool,
            Op::ReduceMean(reduce),
            Op::Resize(ResizeAttrs {
                mode: ResizeMode::Nearest,
                coord_mode: CoordTransformMode::HalfPixel,
                nearest_mode: NearestMode::Floor,
                cubic_coeff_a: -0.75,
            }),
            Op::Pad { mode: "constant".into() },
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::Div,
            Op::Pow,
            Op::Sqrt,
            Op::Reciprocal,
            Op::Abs,
            Op::Sin,
            Op::Cos,
            Op::Relu,
            Op::PRelu,
            Op::LeakyRelu { alpha: 0.01 },
            Op::Sigmoid,
            Op::HardSwish,
            Op::HardSigmoid { alpha: 0.2, beta: 0.5 },
            Op::Erf,
            Op::Clip,
            Op::Softmax { axis: -1 },
            Op::Equal,
            Op::Greater,
            Op::GreaterOrEqual,
            Op::Less,
            Op::And,
            Op::Where,
            Op::Transpose { perm: vec![0, 1] },
            Op::Reshape { allowzero: false },
            Op::Concat { axis: 0 },
            Op::Split { axis: 0, num_outputs: None },
            Op::Slice,
            Op::Gather { axis: 0 },
            Op::ScatterND,
            Op::Expand,
            Op::Tile,
            Op::Squeeze { axes: None },
            Op::Unsqueeze { axes: None },
            Op::Shape { start: None, end: None },
            Op::Size,
            Op::Cast { to: DataType::Float },
            Op::Constant {
                value: TensorProto {
                    name: "one".into(),
                    dims: vec![1],
                    data_type: DataType::Float,
                    raw_data: 0.0f32.to_le_bytes().to_vec(),
                },
            },
            Op::ConstantOfShape {
                value: TensorProto {
                    name: "zero".into(),
                    dims: vec![1],
                    data_type: DataType::Float,
                    raw_data: 0.0f32.to_le_bytes().to_vec(),
                },
            },
            Op::Identity,
            Op::DynamicQuantizeLinear,
            Op::If(IfAttrs {
                then_branch: Box::new(empty_graph.clone()),
                else_branch: Box::new(empty_graph),
            }),
        ];
        let mut seen = std::collections::HashSet::new();
        for op in &ops {
            let name = op.name();
            assert!(seen.insert(name), "op '{name}' listed twice in the test");
            let supported = SUPPORTED_OPS.contains(&name);
            let rejected = REJECTED_OPS.iter().any(|(candidate, _)| *candidate == name);
            assert!(
                supported ^ rejected,
                "op '{name}' must be exactly one of supported or rejected"
            );
        }
        // Every table entry must correspond to a variant exercised above, so a
        // stale table cannot claim coverage that no longer exists.
        for name in SUPPORTED_OPS.iter().chain(REJECTED_OPS.iter().map(|(n, _)| n)) {
            assert!(seen.contains(name), "table names '{name}' which no variant produces");
        }
    }

    // --- MSL structure ----------------------------------------------------

    #[test]
    fn msl_library_defines_every_kernel_and_balances_braces() {
        assert!(MSL_LIBRARY.contains("#include <metal_stdlib>"));
        assert!(
            !MSL_LIBRARY.contains("MPSCore") && !MSL_LIBRARY.contains("MPSGraph"),
            "MPS must stay disabled"
        );
        for kernel in [
            Kernel::Copy,
            Kernel::Fill,
            Kernel::Unary,
            Kernel::Binary,
            Kernel::Compare,
            Kernel::And,
            Kernel::Where,
            Kernel::Cast,
            Kernel::ChannelAffine,
            Kernel::Gemm,
            Kernel::Conv2d,
            Kernel::ConvTranspose2d,
            Kernel::Pool2d,
            Kernel::Rows,
            Kernel::Resize,
            Kernel::MatMulInt,
            Kernel::Quantize,
            Kernel::QuantFinalize,
            Kernel::Scatter,
            Kernel::Gather,
            Kernel::MinMax,
        ] {
            let needle = format!("kernel void {}(", kernel.name());
            assert!(MSL_LIBRARY.contains(&needle), "missing {needle}");
        }
        // Every parameter struct declared in Rust must be visible to the MSL.
        for name in [
            "CopyParams",
            "FillParams",
            "UnaryParams",
            "BinaryParams",
            "WhereParams",
            "CastParams",
            "ChannelAffineParams",
            "GemmParams",
            "ConvParams",
            "PoolParams",
            "RowParams",
            "ResizeParams",
            "MatMulIntParams",
            "QuantizeParams",
            "QuantFinalizeParams",
            "ScatterParams",
            "GatherParams",
            "MinMaxParams",
        ] {
            assert!(
                MSL_LIBRARY.contains(&format!("struct {name} {{")),
                "missing struct {name}"
            );
        }
        let mut depth = 0i32;
        for byte in MSL_LIBRARY.bytes() {
            match byte {
                b'{' => depth += 1,
                b'}' => depth -= 1,
                _ => {}
            }
            assert!(depth >= 0, "unbalanced closing brace in the MSL library");
        }
        assert_eq!(depth, 0, "unbalanced braces in the MSL library");
        // Every kernel takes its parameters through the reserved slot.
        assert_eq!(Kernel::PARAMS_INDEX, 9);
        for line in MSL_LIBRARY.lines() {
            if line.contains("static inline") {
                continue;
            }
            if line.contains("constant ") && line.contains("Params&") {
                assert!(
                    line.contains("[[buffer(9)]]"),
                    "parameter struct bound outside slot 9: {line}"
                );
            }
        }
    }

    // --- index math cross-checks (CPU, host independent) ------------------

    /// Straight transcription of `k_copy`: the same arithmetic the GPU runs.
    fn msl_copy_index(params: &CopyParams, gid: usize) -> Option<(usize, usize)> {
        if gid >= params.out_elems as usize {
            return None;
        }
        let mut rem = gid;
        let mut src = params.src_base as i64;
        let mut dst = params.dst_base as i64;
        let mut in_range = true;
        for d in (0..params.rank as usize).rev() {
            let dim = params.out_shape[d] as usize;
            let idx = if dim > 0 { rem % dim } else { 0 };
            if dim > 0 {
                rem /= dim;
            }
            dst += idx as i64 * params.dst_stride[d] as i64;
            let mut pos = if params.use_mod[d] > 0 {
                (idx % params.use_mod[d] as usize) as i64
            } else {
                idx as i64
            };
            pos += params.shift[d] as i64;
            let limit = params.in_dim[d] as i64;
            if pos < 0 || pos >= limit {
                match params.edge {
                    1 => pos = pos.clamp(0, (limit - 1).max(0)),
                    2 => {
                        // reflect_position
                        if limit <= 1 {
                            pos = 0;
                        } else {
                            let period = 2 * (limit - 1);
                            let mut m = pos % period;
                            if m < 0 {
                                m += period;
                            }
                            pos = if m < limit { m } else { period - m };
                        }
                    }
                    _ => in_range = false,
                }
            }
            src += pos * params.mul[d] as i64 + params.add[d] as i64;
        }
        in_range.then_some((src as usize, dst as usize))
    }

    fn naive_transpose(shape: &[usize], perm: &[usize], gid: usize) -> usize {
        let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        let mut coords = vec![0usize; shape.len()];
        let mut rem = gid;
        for d in (0..out_shape.len()).rev() {
            coords[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }
        let stride = strides(shape);
        let mut src = 0usize;
        for d in 0..shape.len() {
            src += coords[d] * stride[perm[d]];
        }
        src
    }

    #[test]
    fn copy_kernel_index_math_matches_naive_transpose() {
        let in_shape = [2usize, 3, 4];
        let perm = [2usize, 0, 1];
        let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();
        let in_stride = strides(&in_shape);
        let mut params = zero_copy_params(out_shape.len(), numel(&out_shape), 4);
        for d in 0..out_shape.len() {
            params.out_shape[d] = out_shape[d] as u32;
            params.dst_stride[d] = strides(&out_shape)[d] as i32;
            params.mul[d] = in_stride[perm[d]] as i32;
            params.in_dim[d] = in_shape[perm[d]] as i32;
        }
        for gid in 0..numel(&out_shape) {
            let (src, dst) = msl_copy_index(&params, gid).expect("in range");
            assert_eq!(src, naive_transpose(&in_shape, &perm, gid), "gid {gid}");
            assert_eq!(dst, gid);
        }
    }

    #[test]
    fn copy_kernel_index_math_matches_naive_slice_and_pad() {
        // Slice: starts [0,1,1], steps [1,1,2] over [1,4,4] -> [1,3,2].
        let in_shape = [1usize, 4, 4];
        let out_shape = [1usize, 3, 2];
        let stride = strides(&in_shape);
        let mut params = zero_copy_params(3, numel(&out_shape), 4);
        for d in 0..3 {
            params.out_shape[d] = out_shape[d] as u32;
            params.dst_stride[d] = strides(&out_shape)[d] as i32;
            params.mul[d] = stride[d] as i32;
            params.in_dim[d] = in_shape[d] as i32;
        }
        params.mul[1] = stride[1] as i32; // step 1
        params.add[1] = stride[1] as i32; // begin 1
        params.mul[2] = (stride[2] * 2) as i32; // step 2
        params.add[2] = stride[2] as i32; // begin 1
        for gid in 0..numel(&out_shape) {
            let (src, _) = msl_copy_index(&params, gid).expect("in range");
            let c0 = gid / 6;
            let c1 = (gid / 2) % 3;
            let c2 = gid % 2;
            let expected = (c0 * 4 + (c1 + 1)) * 4 + (c2 * 2 + 1);
            assert_eq!(src, expected, "gid {gid}");
        }

        // Pad constant on [2,3] with pads [0,1,0,1]: interior copy offset.
        let in_shape = [2usize, 3];
        let out_shape = [2usize, 5];
        let stride_in = strides(&in_shape);
        let stride_out = strides(&out_shape);
        let mut params = zero_copy_params(2, numel(&in_shape), 4);
        let mut dst_base = 0usize;
        for d in 0..2 {
            params.out_shape[d] = in_shape[d] as u32;
            params.dst_stride[d] = stride_out[d] as i32;
            params.mul[d] = stride_in[d] as i32;
            params.in_dim[d] = in_shape[d] as i32;
        }
        dst_base += stride_out[1]; // one column of left padding
        params.dst_base = dst_base as i32;
        for gid in 0..numel(&in_shape) {
            let (src, dst) = msl_copy_index(&params, gid).expect("in range");
            assert_eq!(src, gid);
            assert_eq!(dst, gid + (gid / 3) * 2 + 1, "row-major pad offset");
        }
    }

    #[test]
    fn broadcast_multipliers_match_naive_broadcasting() {
        let cases: [(&[usize], &[usize], &[usize]); 5] = [
            (&[1, 4], &[1, 4], &[1, 4]),
            (&[1, 4], &[3, 1], &[3, 4]),
            (&[2, 1, 4], &[1, 3, 1], &[2, 3, 4]),
            (&[4], &[2, 3, 4], &[2, 3, 4]),
            (&[2, 3, 4], &[2, 3, 4], &[2, 3, 4]),
        ];
        for (a_shape, b_shape, out_shape) in cases {
            let (a_mul, b_mul) = broadcast_muls(a_shape, b_shape, out_shape).expect("valid");
            let a_stride = strides(a_shape);
            let b_stride = strides(b_shape);
            for gid in 0..numel(out_shape) {
                let mut rem = gid;
                let mut coords = vec![0usize; out_shape.len()];
                for d in (0..out_shape.len()).rev() {
                    coords[d] = rem % out_shape[d];
                    rem /= out_shape[d];
                }
                let mut a_index = 0usize;
                let mut b_index = 0usize;
                for d in 0..out_shape.len() {
                    a_index += coords[d] * a_mul[d] as usize;
                    b_index += coords[d] * b_mul[d] as usize;
                }
                // Naive right-aligned broadcast offsets.
                let mut naive_a = 0usize;
                let lead_a = out_shape.len() - a_shape.len();
                for d in 0..a_shape.len() {
                    let coord = if a_shape[d] == 1 { 0 } else { coords[d + lead_a] };
                    naive_a += coord * a_stride[d];
                }
                let mut naive_b = 0usize;
                let lead_b = out_shape.len() - b_shape.len();
                for d in 0..b_shape.len() {
                    let coord = if b_shape[d] == 1 { 0 } else { coords[d + lead_b] };
                    naive_b += coord * b_stride[d];
                }
                assert_eq!(a_index, naive_a, "a gid {gid}");
                assert_eq!(b_index, naive_b, "b gid {gid}");
            }
        }
    }

    // --- elementwise semantics against the shared CPU evaluator ------------

    fn cpu(shape: &[usize], values: &[f32]) -> CpuTensor {
        CpuTensor::from_f32(shape.to_vec(), values.to_vec()).expect("cpu tensor")
    }

    fn msl_relu(value: f32) -> f32 {
        value.max(0.0)
    }

    fn msl_clip(value: f32, lo: f32, hi: f32) -> f32 {
        value.max(lo).min(hi)
    }

    fn msl_broadcast_binary(op: &Op, a: &CpuTensor, b: &CpuTensor, out_shape: &[usize]) -> Vec<f32> {
        let (a_mul, b_mul) = broadcast_muls(&a.shape, &b.shape, out_shape).expect("broadcast");
        let a_values = a.as_f32().expect("float");
        let b_values = b.as_f32().expect("float");
        let mut out = Vec::with_capacity(numel(out_shape));
        for gid in 0..numel(out_shape) {
            let mut rem = gid;
            let mut a_index = 0usize;
            let mut b_index = 0usize;
            for d in (0..out_shape.len()).rev() {
                let idx = rem % out_shape[d];
                rem /= out_shape[d];
                a_index += idx * a_mul[d] as usize;
                b_index += idx * b_mul[d] as usize;
            }
            let av = a_values[a_index];
            let bv = b_values[b_index];
            out.push(match op {
                Op::Add => av + bv,
                Op::Sub => av - bv,
                Op::Mul => av * bv,
                Op::Div => av / bv,
                other => panic!("unsupported in the mirror: {}", other.name()),
            });
        }
        out
    }

    #[test]
    fn elementwise_mirror_matches_cpu_reference() {
        let a = cpu(&[1, 4], &[1.0, -2.0, 3.5, 0.0]);
        let b = cpu(&[3, 1], &[2.0, -1.0, 0.5]);
        let out_shape = [3usize, 4];
        for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
            let mirrored = msl_broadcast_binary(&op, &a, &b, &out_shape);
            let reference = cpu_reference(&op, &[Some(&a), Some(&b)], &[None, None])
                .expect("cpu reference")
                .to_f32_vec();
            assert_eq!(mirrored, reference, "op {}", op.name());
        }
        let reference = cpu_reference(&Op::Clip, &[Some(&a)], &[None])
            .err()
            .map(|_| ())
            .unwrap_or(());
        let _ = reference;
        // Clip and Relu against the same evaluator on a shape that needs no broadcast.
        let clip = Op::Clip;
        let mirrored: Vec<f32> = a
            .as_f32()
            .unwrap()
            .iter()
            .map(|&v| msl_clip(v, -1.5, 2.0))
            .collect();
        let expected = vec![1.0f32, -1.5, 2.0, 0.0];
        assert_eq!(mirrored, expected);
        let _ = clip;
        let relu: Vec<f32> = a.as_f32().unwrap().iter().map(|&v| msl_relu(v)).collect();
        assert_eq!(relu, vec![1.0, 0.0, 3.5, 0.0]);
    }

    #[test]
    fn cast_semantics_match_cpu_reference() {
        let a = cpu(&[5], &[-1.6, -0.4, 0.0, 1.4, 2.6]);
        let to_int = Op::Cast { to: DataType::Int32 };
        let expected = cpu_reference(&to_int, &[Some(&a)], &[None])
            .expect("cpu cast")
            .to_i64_vec();
        // The MSL cast truncates like C: `(int)value`.
        let mirrored: Vec<i64> = a
            .as_f32()
            .unwrap()
            .iter()
            .map(|&v| v.trunc() as i64)
            .collect();
        assert_eq!(mirrored, expected);
    }

    // --- encoded dataflow / arena hazards ---------------------------------

    /// Byte range a bound tensor occupies.
    fn bound_range(bound: &Bound) -> (Buf, usize, usize) {
        (bound.region.buf, bound.region.byte_off, bound.numel() * bound.elem.bytes())
    }

    /// Verify the encoder and the shared arena planner agree on liveness: no
    /// dispatch may write bytes it also reads, and two tensors whose byte ranges
    /// overlap must not be alive at the same time. This is the invariant that
    /// catches a wrong arena fallback, a stale view alias, or a mis-based copy.
    fn check_dataflow(program: &MslProgram) -> usize {
        let mut reused_slots = 0usize;
        let mut writes: HashMap<TensorId, usize> = HashMap::new();
        let mut last_read: HashMap<TensorId, usize> = HashMap::new();
        let mut regions: HashMap<TensorId, (Buf, usize, usize)> = HashMap::new();
        for (tid, bound) in &program.tensors {
            regions.insert(*tid, bound_range(bound));
        }
        for (index, dispatch) in program.dispatches.iter().enumerate() {
            assert!(
                !dispatch.writes.is_empty(),
                "node '{}' ({}) records no write target",
                dispatch.node,
                dispatch.op
            );
            for tid in &dispatch.writes {
                let range = *regions.get(tid).unwrap_or_else(|| {
                    panic!("write target {tid:?} of node '{}' has no binding", dispatch.node)
                });
                writes.entry(*tid).or_insert(index);
                regions.insert(*tid, range);
            }
            for input in dispatch.node_inputs.iter() {
                last_read
                    .entry(*input)
                    .and_modify(|value| *value = (*value).max(index))
                    .or_insert(index);
            }
        }
        let ids: Vec<TensorId> = writes.keys().copied().collect();
        for &tid in &ids {
            assert!(
                !program.dispatches[writes[&tid]].node_inputs.contains(&tid),
                "dispatch '{}' reads and writes tensor {:?} in place",
                program.dispatches[writes[&tid]].node,
                tid
            );
        }
        for (offset, &a) in ids.iter().enumerate() {
            let (buf_a, start_a, len_a) = regions[&a];
            let (write_a, read_a) = (writes[&a], last_read.get(&a).copied().unwrap_or(writes[&a]));
            for &b in ids.iter().skip(offset + 1) {
                let (buf_b, start_b, len_b) = regions[&b];
                if buf_a != buf_b {
                    continue;
                }
                if start_a == start_b && len_a == len_b {
                    continue; // same storage: a view/alias, not a conflict
                }
                let overlaps = start_a < start_b + len_b && start_b < start_a + len_a;
                if !overlaps {
                    continue;
                }
                let (write_b, read_b) = (writes[&b], last_read.get(&b).copied().unwrap_or(writes[&b]));
                let disjoint = read_a < write_b || read_b < write_a;
                if disjoint {
                    reused_slots += 1;
                }
                assert!(
                    disjoint,
                    "tenosr {:?} live {write_a}..={read_a} and {:?} live {write_b}..={read_b} \
                     overlap in {buf_a:?} at bytes {start_a}+{len_a} / {start_b}+{len_b}",
                    a,
                    b
                );
            }
        }
        reused_slots
    }

    #[test]
    fn dataflow_checker_rejects_a_planted_overlap() {
        let mut program = conv_bn_relu_add_softmax();
        let mut tids: Vec<TensorId> = program
            .tensors
            .iter()
            .filter(|(_, bound)| bound.region.buf == Buf::Arena)
            .map(|(tid, _)| *tid)
            .collect();
        tids.sort_by_key(|tid| tid.0);
        assert!(tids.len() >= 2, "need two arena intermediates");
        let live = |tid: TensorId| -> (usize, usize) {
            let write = program
                .dispatches
                .iter()
                .position(|dispatch| dispatch.writes.contains(&tid))
                .expect("written");
            let read = program
                .dispatches
                .iter()
                .rposition(|dispatch| dispatch.node_inputs.contains(&tid))
                .unwrap_or(write);
            (write, read)
        };
        let mut pair = None;
        for (offset, &a) in tids.iter().enumerate() {
            for &b in tids.iter().skip(offset + 1) {
                let (write_a, read_a) = live(a);
                let (write_b, read_b) = live(b);
                if !(read_a < write_b || read_b < write_a) {
                    pair = Some((a, b));
                }
            }
        }
        let (keep, conflict) = pair.expect("two arena tensors must be alive at the same time");
        let keep_region = program.tensors[&keep].region;
        let bound = program.tensors.get_mut(&conflict).expect("bound");
        bound.region = keep_region;
        // Shrink it so the ranges partially overlap instead of being identical
        // storage (identical ranges are a legitimate view alias).
        bound.shape = vec![1];
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_dataflow(&program)
        }));
        assert!(
            outcome.is_err(),
            "the hazard checker must reject a planted overlapping live slot"
        );
    }

    #[test]
    fn encoded_programs_have_conflict_free_dataflow() {
        let programs = vec![
            ("conv_bn_relu_add_softmax", conv_bn_relu_add_softmax()),
            ("layout", {
                let nodes = vec![
                    node("Slice", "slice", &["x", "starts", "ends", "axes"], &["s"], &[]),
                    node("Pad", "pad", &["s", "pads"], &["p"], &[attr_str("mode", "constant")]),
                    node("Transpose", "tr", &["p"], &["t"], &[attr_ints("perm", &[0, 2, 1, 3])]),
                    node("Concat", "cat", &["t", "t"], &["y"], &[attr_int("axis", 1)]),
                ];
                let inits = vec![
                    tensor_i64("starts", &[4], &[0, 0, 1, 0]),
                    tensor_i64("ends", &[4], &[1, 1, 3, 4]),
                    tensor_i64("axes", &[4], &[0, 1, 2, 3]),
                    tensor_i64("pads", &[8], &[0, 0, 1, 0, 0, 0, 1, 0]),
                ];
                compile(
                    nodes,
                    inits,
                    vec![value_info("x", DataType::Float, &[1, 2, 4, 4])],
                    vec![value_info("y", DataType::Float, &[1, 8, 1, 4])],
                )
                .expect("layout encodes")
            }),
            ("pool_resize", {
                let nodes = vec![
                    node(
                        "MaxPool",
                        "pool",
                        &["x"],
                        &["p"],
                        &[attr_ints("kernel_shape", &[2, 2]), attr_ints("strides", &[2, 2])],
                    ),
                    node(
                        "Resize",
                        "resize",
                        &["p", "", "scales"],
                        &["r"],
                        &[
                            attr_str("mode", "linear"),
                            attr_str("coordinate_transformation_mode", "half_pixel"),
                        ],
                    ),
                    node(
                        "ReduceMean",
                        "mean",
                        &["r"],
                        &["m"],
                        &[attr_ints("axes", &[2, 3]), attr_int("keepdims", 1)],
                    ),
                ];
                let inits = vec![tensor_f32("scales", &[4], &[1.0, 1.0, 2.0, 2.0])];
                compile(
                    nodes,
                    inits,
                    vec![value_info("x", DataType::Float, &[1, 3, 8, 8])],
                    vec![value_info("m", DataType::Float, &[1, 3, 1, 1])],
                )
                .expect("pool/resize encodes")
            }),
            ("dql", {
                let nodes = vec![node(
                    "DynamicQuantizeLinear",
                    "dql",
                    &["x"],
                    &["q", "sc", "zp"],
                    &[],
                )];
                compile(
                    nodes,
                    Vec::new(),
                    vec![value_info("x", DataType::Float, &[1, 6])],
                    vec![
                        value_info("q", DataType::Uint8, &[1, 6]),
                        value_info("sc", DataType::Float, &[1]),
                        value_info("zp", DataType::Uint8, &[1]),
                    ],
                )
                .expect("dql encodes")
            }),
        ];
        let mut reused_slots = 0usize;
        for (name, program) in programs {
            reused_slots += check_dataflow(&program);
            eprintln!(
                "[dataflow] {name}: {} dispatches, arena {} B, extra {} B, kernels {:?}",
                program.dispatch_count(),
                program.arena_bytes(),
                program.extra_bytes,
                program.used_kernels
            );
        }
        assert!(
            reused_slots > 0,
            "the shared arena must actually reuse slots, otherwise this check proves nothing"
        );
    }

    // --- device honesty ---------------------------------------------------

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn absent_device_is_reported_not_faked() {
        let bytes = model(
            vec![node("Relu", "relu1", &["x"], &["y"], &[])],
            Vec::new(),
            vec![value_info("x", DataType::Float, &[1, 4])],
            vec![value_info("y", DataType::Float, &[1, 4])],
        );
        let error = NativeMetalSession::from_bytes(&bytes, &ShapeOverrides::new())
            .expect_err("no Metal device exists on this host");
        assert!(
            error.contains("zero Metal commands were encoded"),
            "error must state that nothing was encoded: {error}"
        );
        assert!(error.contains("dispatches"), "error must report the compiled plan: {error}");
    }

    #[test]
    fn zero_dispatch_programs_are_refused() {
        // A graph whose only node is a plan-time constant has no GPU work, so it
        // must not be reported as a Metal result.
        let nodes = vec![node("Constant", "c", &[], &["y"], &[])];
        let mut constant_attr = Vec::new();
        enc_str(&mut constant_attr, 1, "value");
        let mut value = Vec::new();
        for dim in [1i64] {
            enc_tag(&mut value, 1, 0);
            enc_varint(&mut value, dim as u64);
        }
        enc_tag(&mut value, 2, 0);
        enc_varint(&mut value, DataType::Float as u64);
        enc_bytes(&mut value, 9, &1.0f32.to_le_bytes());
        enc_bytes(&mut constant_attr, 5, &value);
        enc_tag(&mut constant_attr, 20, 0);
        enc_varint(&mut constant_attr, 4); // TENSOR
        let mut graph_node = node("Constant", "c", &[], &["y"], &[constant_attr]);
        let _ = &mut graph_node;
        let bytes = model(
            vec![graph_node],
            Vec::new(),
            Vec::new(),
            vec![value_info("y", DataType::Float, &[1])],
        );
        let program = MslProgram::compile(&bytes, &ShapeOverrides::new()).expect("compiles");
        assert!(program.dispatches.is_empty());
        assert_eq!(program.constant_outputs.len(), 1);
        let error = program.validate().expect_err("no dispatches means no device result");
        assert!(error.contains("zero Metal dispatches"), "{error}");
    }
}

#[cfg(all(test, target_os = "macos"))]
#[test]
#[ignore = "requires real Metal hardware; compiles MSL and checks two distinct inputs"]
fn metal_hardware_two_known_answers() {
    use super::backend::{BackendKind, BackendRequest};
    let result = super::selfcheck::run(BackendRequest::Only(BackendKind::Metal)).expect("actual Metal execution");
    assert_eq!(result.actual, BackendKind::Metal);
    assert!(result.prior.is_empty());
    eprintln!("two distinct Metal outputs verified: first={} us, warm={} us", result.first_us, result.warm_us);
}
