//! End-to-end GPU inference session (`NativeCudaSession`) for Aurora Native Inference.
//!
//! Combines the Phase 1 pure-Rust ONNX loader / BN folder / Arena memory planner
//! with the Phase 2 native CUDA PTX operator suites (`ops_conv_gemm`,
//! `ops_elem_norm`, `ops_spatial_tensor`, `ops_quant`) and `cpu_shape_eval` to
//! execute real `.onnx` models directly on NVIDIA GPUs with zero external DLLs.

use std::collections::HashMap;
use std::fmt;

use crate::cuda::CudaError;
use crate::native_infer::graph::{IrGraph, Op, TensorId};
use crate::native_infer::onnx_proto::DataType;
use crate::native_infer::{load_and_plan, CompiledPlan, NativeInferError};

#[derive(Debug)]
pub enum SessionError {
    Infer(NativeInferError),
    Cuda(CudaError),
    Runtime(String),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Infer(e) => write!(f, "native infer plan error: {e}"),
            Self::Cuda(e) => write!(f, "native infer CUDA error: {e:?}"),
            Self::Runtime(msg) => write!(f, "native infer runtime error: {msg}"),
        }
    }
}

impl std::error::Error for SessionError {}

impl From<NativeInferError> for SessionError {
    fn from(e: NativeInferError) -> Self {
        Self::Infer(e)
    }
}

impl From<CudaError> for SessionError {
    fn from(e: CudaError) -> Self {
        Self::Cuda(e)
    }
}

#[cfg(windows)]
mod imp {
    use super::*;
    use crate::cuda::DeviceBuffer;
    use crate::native_infer::cpu_shape_eval::{
        eval_op as eval_cpu_node, CpuData, CpuTensor, DEFAULT_MAX_CPU_TENSOR_ELEMS,
    };
    use crate::native_infer::cuda_graph;
    use crate::native_infer::ops_conv_gemm::{
        Conv2dParams, ConvGemmKernels, ConvTranspose2dParams, FusedActivation,
    };
    use crate::native_infer::ops_elem_norm::{BinaryOp, CompareOp, ElemNormKernels, UnaryOp};
    use crate::native_infer::ops_quant::{
        ConvIntegerParams, MatMulIntegerParams, QuantInt8Kernels, ZeroPointArg,
    };
    use crate::native_infer::ops_spatial_tensor::{
        pool2d_out_dim, ResizeKernelMode, SpatialTensorKernels,
    };

    #[inline]
    fn f32_as_bytes(s: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s)) }
    }

    #[inline]
    fn f32_as_bytes_mut(s: &mut [f32]) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(s))
        }
    }

    #[inline]
    fn numel(shape: &[usize]) -> usize {
        shape.iter().copied().product::<usize>()
    }

    #[inline]
    fn align_up_usize(value: usize, alignment: usize) -> usize {
        let rem = value % alignment;
        if rem == 0 {
            value
        } else {
            value + (alignment - rem)
        }
    }

    #[inline]
    fn norm_axis(axis: i64, rank: usize) -> Result<usize, SessionError> {
        let r = rank as i64;
        let a = if axis < 0 { axis + r } else { axis };
        if a < 0 || a >= r {
            return Err(SessionError::Runtime(format!(
                "axis {axis} out of range for rank {rank}"
            )));
        }
        Ok(a as usize)
    }

    /// Compiled, GPU-resident native CUDA inference session.
    ///
    /// Note: holds owned `DeviceBuffer<'static>` backed by the process-global
    /// primary CUDA context inside `ConvGemmKernels` (which owns `Nvidia`).
    pub struct NativeCudaSession {
        plan: CompiledPlan,
        conv: ConvGemmKernels,
        elem: ElemNormKernels,
        spatial: SpatialTensorKernels,
        quant: QuantInt8Kernels,
        /// Persistent GPU buffers for static weights (`TensorId -> (ptr, shape, dtype)`).
        weight_ptrs: HashMap<TensorId, (u64, Vec<usize>, DataType)>,
        /// Backing device allocations kept alive for the session lifetime.
        /// Stored as raw `(ptr, len)` freed via `cuMemFree` or held via leaked/owned wrappers;
        /// here we allocate via `conv.nvidia()` using a boxed-leaked/transmuted lifetime bound
        /// to `self.conv` and drop buffers BEFORE `self.conv` in `Drop`.
        owned_buffers: Vec<DeviceBuffer<'static>>,
        /// Single contiguous Arena VRAM buffer for intermediate activations.
        arena_buf: Option<DeviceBuffer<'static>>,
        arena_base_ptr: u64,
        /// Shared `im2col` scratch workspace.
        im2col_buf: Option<DeviceBuffer<'static>>,
        im2col_ptr: u64,
        im2col_bytes: usize,
        /// Small static weights mirrored on CPU for shape/control-flow evaluation.
        static_cpu_tensors: HashMap<TensorId, CpuTensor>,
        /// Process-wide driver handle, kept `'static` so persistent buffers and a
        /// captured graph can outlive a single call. The underlying context is
        /// process-global, exactly as `from_bytes` documents for `device buffers`.
        gpu: &'static crate::cuda::Nvidia,
        /// Static-graph slot: one installed CUDA Graph plus its fallback state.
        graph: cuda_graph::GraphSlot,
        // All paths reuse the same arena and im2col buffer (not just graph replay).
        run_gate: std::sync::Mutex<()>,
        /// Reusable `1.0` / `0.0` operands for the `Bool -> Float` cast.
        const_one: Option<DeviceBuffer<'static>>,
        const_zero: Option<DeviceBuffer<'static>>,
        /// Reusable FP16 staging for `Float -> Float16` casts, sized at load.
        ///
        /// A per-run allocation here would make the model uncapturable: the
        /// recorded node would point at a buffer that dies with the call.
        f16_scratch: Option<DeviceBuffer<'static>>,
        f16_scratch_elems: usize,
        /// Device staging for CPU-evaluated shape/parameter tensors.
        ///
        /// A ViT builds its `Reshape`/`Expand` targets from `Shape -> Gather ->
        /// Concat -> Where`, and the executor evaluates those chains on the host
        /// and mirrors the small results to the device so downstream GPU kernels
        /// can read them. Doing that with `alloc_buffer` + `upload_buffer` per
        /// node meant one `cuMemAlloc` per call *and* a pointer that dies with
        /// the call, so no model with a shape subgraph could ever be captured.
        /// The staged blocks live in one buffer whose layout is the same on every
        /// pass over the same graph.
        cpu_stage: std::sync::Mutex<CpuStage>,
    }

    /// Deterministic bump-allocated device staging for CPU-evaluated tensors.
    ///
    /// `slots` describes the block layout the last non-recording pass produced;
    /// a record pass must reproduce it byte for byte, which is what proves the
    /// staged values are a function of the shape and the weights rather than of
    /// the pixels.
    struct CpuStage {
        buffer: Option<DeviceBuffer<'static>>,
        base: u64,
        capacity: usize,
        cursor: usize,
        /// Buffers replaced by a growth; kept alive because a kernel enqueued
        /// earlier in the same pass may still reference them.
        retired: Vec<DeviceBuffer<'static>>,
        /// Blocks staged by the pass that is running now.
        current: Vec<StageSlot>,
        /// Blocks staged by the last completed non-recording pass.
        committed: Vec<StageSlot>,
    }

    /// One staged block: `(offset, len, fnv1a-64 of the bytes)`.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct StageSlot {
        offset: usize,
        len: usize,
        hash: u64,
        address: u64, // allocation that actually received this block
    }

    impl CpuStage {
        fn new() -> Self {
            Self {
                buffer: None,
                base: 0,
                capacity: 0,
                cursor: 0,
                retired: Vec::new(),
                current: Vec::new(),
                committed: Vec::new(),
            }
        }
    }

    /// 256-byte alignment for every staged block, matching the arena.
    const CPU_STAGE_ALIGNMENT: usize = 256;
    /// Initial staging capacity; a shape subgraph of a 640x640 FPN needs ~64 KiB.
    const CPU_STAGE_INITIAL_BYTES: usize = 256 * 1024;

    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for &b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    impl Drop for NativeCudaSession {
        fn drop(&mut self) {
            // The graph executable replays against the arena, the im2col
            // workspace and `owned_buffers`, so it has to go first: its `Drop`
            // drains the context and destroys the executable before any buffer
            // it names is freed.
            self.graph.clear();
            // Explicitly drop all device buffers while `self.conv` (holding `Nvidia`) is still alive.
            self.f16_scratch.take();
            self.const_one.take();
            self.const_zero.take();
            if let Ok(mut stage) = self.cpu_stage.lock() {
                stage.buffer.take();
                stage.retired.clear();
            }
            self.im2col_buf.take();
            self.arena_buf.take();
            self.owned_buffers.clear();
        }
    }

    impl NativeCudaSession {
        /// Load an `.onnx` model from bytes, fold BatchNorms, infer shapes, plan
        /// the VRAM arena, upload all static weights, and prepare CUDA kernels.
        pub fn from_bytes(
            onnx_bytes: &[u8],
            input_overrides: &HashMap<String, Vec<usize>>,
        ) -> Result<Self, SessionError> {
            let plan = load_and_plan(onnx_bytes, input_overrides)?;
            let conv = ConvGemmKernels::open()?;
            let elem = ElemNormKernels::open()?;
            let spatial = SpatialTensorKernels::open()?;
            let quant = QuantInt8Kernels::open()?;

            // A process-wide CUDA context does not give a reference to a local
            // Rust handle a static lifetime. Store the handle itself at a stable
            // address: sessions move, but their allocations and graph replays do not.
            static DRIVER: std::sync::OnceLock<crate::cuda::Nvidia> = std::sync::OnceLock::new();
            if DRIVER.get().is_none() {
                let handle = crate::cuda::Nvidia::open_ptx(crate::native_infer::ops_conv_gemm::PTX)?;
                let _ = DRIVER.set(handle);
            }
            let gpu_static = DRIVER.get().expect("CUDA handle initialized above");

            let mut weight_ptrs = HashMap::with_capacity(plan.graph.weights.len());
            let mut owned_buffers = Vec::with_capacity(plan.graph.weights.len());
            let mut static_cpu_tensors = HashMap::new();

            for (&tid, proto) in &plan.graph.weights {
                let shape: Vec<usize> = proto.dims.iter().map(|&d| d.max(0) as usize).collect();
                if proto.elem_count() <= DEFAULT_MAX_CPU_TENSOR_ELEMS {
                    if let Ok(ct) = CpuTensor::from_tensor_proto(proto) {
                        static_cpu_tensors.insert(tid, ct);
                    }
                }
                // Upload weights to VRAM. Float16 weights are unpacked to FP32 so all
                // FP32 compute kernels operate on valid 4-byte floats without alignment hazards;
                // explicit `Cast(to=Float16)` nodes still round through IEEE FP16 on GPU.
                let (bytes, stored_dt): (Vec<u8>, DataType) = match proto.data_type {
                    DataType::Float => {
                        if proto.raw_data.len() == proto.elem_count() * 4 {
                            (proto.raw_data.clone(), DataType::Float)
                        } else {
                            let v = proto.to_f32_vec().map_err(|e| {
                                SessionError::Runtime(format!("weight decode error: {e}"))
                            })?;
                            (f32_as_bytes(&v).to_vec(), DataType::Float)
                        }
                    }
                    DataType::Float16 => {
                        let v = proto.to_f32_vec().map_err(|e| {
                            SessionError::Runtime(format!("fp16 weight decode error: {e}"))
                        })?;
                        (f32_as_bytes(&v).to_vec(), DataType::Float)
                    }
                    other => (proto.raw_data.clone(), other),
                };
                if bytes.is_empty() {
                    weight_ptrs.insert(tid, (0u64, shape, stored_dt));
                    continue;
                }
                let buf = gpu_static.alloc_buffer(bytes.len())?;
                gpu_static.upload_buffer(&buf, &bytes)?;
                let ptr = buf.handle();
                owned_buffers.push(buf);
                weight_ptrs.insert(tid, (ptr, shape, stored_dt));
            }

            // Compute maximum im2col workspace needed across all Conv nodes.
            let mut max_im2col_bytes = 0usize;
            for node in &plan.graph.nodes {
                if let Op::Conv(attrs) = &node.op {
                    if let (Some(Some(x_id)), Some(Some(w_id))) =
                        (node.inputs.first(), node.inputs.get(1))
                    {
                        if let (Some(x_sh), Some(w_sh)) = (
                            plan.graph.tensors[x_id.0].static_shape(),
                            plan.graph.tensors[w_id.0].static_shape(),
                        ) {
                            if let Ok(p) =
                                Conv2dParams::from_attrs(&x_sh, &w_sh, attrs, FusedActivation::None)
                            {
                                if !p.is_depthwise() && !p.is_direct_1x1() {
                                    max_im2col_bytes =
                                        max_im2col_bytes.max(p.im2col_workspace_bytes());
                                }
                            }
                        }
                    }
                }
            }

            let arena_buf = if plan.arena.peak_bytes > 0 {
                Some(gpu_static.alloc_buffer(plan.arena.peak_bytes)?)
            } else {
                None
            };
            let arena_base_ptr = arena_buf.as_ref().map_or(0, DeviceBuffer::handle);

            let im2col_buf = if max_im2col_bytes > 0 {
                Some(gpu_static.alloc_buffer(max_im2col_bytes)?)
            } else {
                None
            };
            let im2col_ptr = im2col_buf.as_ref().map_or(0, DeviceBuffer::handle);

            // Persistent cast operands. The `Bool -> Float` cast needs scalar
            // 1.0/0.0 buffers and the `Float -> Float16` round trip needs a
            // staging buffer whose size is fixed by the static input shape;
            // allocating either per run would put a pointer that dies with the
            // call inside a captured graph.
            let mut f16_scratch_elems = 0usize;
            let mut needs_bool_cast = false;
            for node in &plan.graph.nodes {
                if let Op::Cast { to } = &node.op
                    && let Some(Some(x_id)) = node.inputs.first()
                {
                    if *to == DataType::Float16
                        && let Some(shape) = plan.graph.tensors[x_id.0].static_shape()
                    {
                        f16_scratch_elems = f16_scratch_elems.max(numel(&shape));
                    }
                    if *to == DataType::Float
                        && plan.graph.tensors[x_id.0].dtype == DataType::Bool
                    {
                        needs_bool_cast = true;
                    }
                }
            }
            let f16_scratch = if f16_scratch_elems > 0 {
                Some(gpu_static.alloc_buffer(f16_scratch_elems.max(1) * 2)?)
            } else {
                None
            };
            let (const_one, const_zero) = if needs_bool_cast {
                let one = gpu_static.alloc_buffer(4)?;
                gpu_static.upload_buffer(&one, f32_as_bytes(&[1.0f32]))?;
                let zero = gpu_static.alloc_buffer(4)?;
                gpu_static.upload_buffer(&zero, f32_as_bytes(&[0.0f32]))?;
                (Some(one), Some(zero))
            } else {
                (None, None)
            };

            Ok(Self {
                plan,
                conv,
                elem,
                spatial,
                quant,
                weight_ptrs,
                owned_buffers,
                arena_buf,
                arena_base_ptr,
                im2col_buf,
                im2col_ptr,
                im2col_bytes: max_im2col_bytes,
                static_cpu_tensors,
                gpu: gpu_static,
                graph: cuda_graph::GraphSlot::new(),
                run_gate: std::sync::Mutex::new(()),
                const_one,
                const_zero,
                f16_scratch,
                f16_scratch_elems,
                cpu_stage: std::sync::Mutex::new(CpuStage::new()),
            })
        }

        pub fn graph(&self) -> &IrGraph {
            &self.plan.graph
        }

        pub(crate) fn actual_device_label(&self)->String {self.gpu.device_label()}

        pub fn arena_peak_bytes(&self) -> usize {
            self.plan.arena.peak_bytes
        }

        /// Clears the per-pass staging cursor before a pass over the graph.
        fn reset_cpu_stage(&self) {
            let mut stage = self.cpu_stage.lock().unwrap_or_else(|p| p.into_inner());
            stage.cursor = 0;
            stage.current.clear();
        }

        /// Copies one CPU-evaluated tensor into the staging buffer and returns its
        /// device pointer.
        ///
        /// During a record pass the bytes are only *compared* with the block the
        /// last plain pass staged at the same offset: a `cuMemcpyHtoD` inside the
        /// captured region has no replay equivalent, and re-uploading a value the
        /// graph would otherwise freeze is exactly what must not happen.
        fn stage_cpu_tensor(
            &self,
            bytes: &[u8],
            recording: bool,
        ) -> Result<u64, SessionError> {
            if bytes.is_empty() {
                return Err(SessionError::Runtime("empty staged tensor".into()));
            }
            let mut stage = self.cpu_stage.lock().unwrap_or_else(|p| p.into_inner());
            let offset = align_up_usize(stage.cursor, CPU_STAGE_ALIGNMENT);
            let end=offset.checked_add(bytes.len()).ok_or_else(||SessionError::Runtime("CPU stage overflow".into()))?;
            if end > stage.capacity {
                if recording {
                    return Err(SessionError::Runtime(
                        "CPU staging capacity exceeded during a record pass".into(),
                    ));
                }
                let capacity = end.checked_mul(2).ok_or_else(||SessionError::Runtime("CPU staging size overflow".into()))?.max(CPU_STAGE_INITIAL_BYTES);
                let buffer = self.gpu.alloc_buffer(capacity)?;
                if let Some(previous) = stage.buffer.replace(buffer) {
                    // An earlier kernel in this pass may still read the old block.
                    stage.retired.push(previous);
                }
                stage.base = stage
                    .buffer
                    .as_ref()
                    .map_or(0, DeviceBuffer::handle);
                stage.capacity = capacity;
            }
            stage.cursor = end;
            let slot = StageSlot {
                offset,
                len: bytes.len(),
                hash: fnv1a64(bytes),
                address: stage.base + offset as u64,
            };
            if recording {
                // The plain pass that ran immediately before this one staged the
                // same tensor from the same shape and weights. A different hash
                // means the value depends on the pixels, and a captured graph
                // would freeze the first image's value into every later call.
                let expected = stage.committed.get(stage.current.len()).copied();
                stage.current.push(slot);
                if !expected.is_some_and(|old| (old.offset,old.len,old.hash)==(slot.offset,slot.len,slot.hash)) {
                    return Err(SessionError::Runtime(
                        "cpu-evaluated tensor is not shape-stable across passes".into(),
                    ));
                }
                // An earlier block can still live in the previous page after growth.
                return Ok(expected.expect("verified staged block").address);
            }
            stage.current.push(slot);
            let buffer = stage
                .buffer
                .as_ref()
                .ok_or_else(|| SessionError::Runtime("CPU staging buffer missing".into()))?;
            self.gpu.upload_buffer_at(buffer, offset, bytes)?;
            Ok(stage.base + offset as u64)
        }

        /// Publishes the block layout of a completed non-recording pass so the
        /// next record pass can prove it reproduces the same bytes.
        fn commit_cpu_stage(&self) {
            let mut stage = self.cpu_stage.lock().unwrap_or_else(|p| p.into_inner());
            stage.committed = stage.current.clone();
        }

        /// Execute the compiled graph with named `f32` inputs and return all
        /// graph outputs as `(name -> Vec<f32>)`.
        ///
        /// Once a static graph has been captured and verified, this call costs
        /// `H2D(input) -> cuGraphLaunch -> stream sync -> D2H(output)` instead of
        /// one kernel launch per node. Every path that cannot be captured keeps
        /// the ordinary execution below, and the reason is recorded in
        /// [`Self::graph_status`] rather than returned as an error.
        pub fn run_f32(
            &self,
            inputs: &HashMap<String, (&[f32], Vec<usize>)>,
        ) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>, SessionError> {
            let _guard = self.run_gate.lock().unwrap_or_else(|p| p.into_inner());
            for &tid in &self.plan.graph.inputs {
                let info = &self.plan.graph.tensors[tid.0];
                let (values, shape) = inputs.get(&info.name).ok_or_else(|| SessionError::Runtime(format!("missing input {}", info.name)))?;
                if shape.iter().try_fold(1usize, |a, &b| a.checked_mul(b)) != Some(values.len())
                    || info.static_shape().is_some_and(|expected| expected != *shape) {
                    return Err(SessionError::Runtime(format!("input {} does not match the compiled shape or element count", info.name)));
                }
            }
            if let Some(replayed) = self.try_replay(inputs) {
                return Ok(replayed);
            }
            let gpu = self.gpu;
            let graph = &self.plan.graph;

            // Runtime tensor bindings: TensorId -> (device_ptr, shape, dtype).
            let mut gpu_tensors: HashMap<TensorId, (u64, Vec<usize>, DataType)> =
                self.weight_ptrs.clone();
            let mut cpu_tensors: HashMap<TensorId, CpuTensor> = self.static_cpu_tensors.clone();
            // Per-run allocations for graph inputs, graph outputs, and dynamic/non-arena tensors.
            let mut run_bufs: Vec<DeviceBuffer<'static>> = Vec::new();

            // 1. Upload graph inputs.
            for &in_id in &graph.inputs {
                let info = &graph.tensors[in_id.0];
                let (data, shape) = inputs.get(&info.name).ok_or_else(|| {
                    SessionError::Runtime(format!("missing graph input '{}'", info.name))
                })?;
                if data.len() != numel(shape) {
                    return Err(SessionError::Runtime(format!(
                        "input '{}' length {} != shape {:?} numel {}",
                        info.name,
                        data.len(),
                        shape,
                        numel(shape)
                    )));
                }
                let buf = gpu.alloc_buffer(data.len().max(1) * 4)?;
                if !data.is_empty() {
                    gpu.upload_buffer(&buf, f32_as_bytes(data))?;
                }
                let ptr = buf.handle();
                run_bufs.push(buf);
                gpu_tensors.insert(in_id, (ptr, shape.clone(), DataType::Float));
                if data.len() <= DEFAULT_MAX_CPU_TENSOR_ELEMS {
                    if let Ok(ct) = CpuTensor::from_f32(shape.clone(), data.to_vec()) {
                        cpu_tensors.insert(in_id, ct);
                    }
                }
            }

            // Pre-allocate graph outputs that are not in the scratch arena.
            let mut output_ids: HashMap<TensorId, (u64, Vec<usize>)> = HashMap::new();
            for &out_id in &graph.outputs {
                if let Some(sh) = graph.tensors[out_id.0].static_shape() {
                    let elem_bytes = graph.tensors[out_id.0].dtype.byte_size().unwrap_or(4);
                    let elem_bytes = elem_bytes.max(4); // promoted floats may have a 16-bit ONNX declaration
                    let bytes = numel(&sh).max(1) * elem_bytes;
                    let buf = gpu.alloc_buffer(bytes)?;
                    let ptr = buf.handle();
                    run_bufs.push(buf);
                    output_ids.insert(out_id, (ptr, sh));
                }
            }

            // 2. Execute nodes in topological order.
            let use_arena = std::env::var("AURORA_NATIVE_NO_ARENA").ok().as_deref() != Some("1");
            self.reset_cpu_stage();
            self.exec_graph(
                graph,
                use_arena,
                false,
                &output_ids,
                &mut gpu_tensors,
                &mut cpu_tensors,
                &mut run_bufs,
            )?;
            // Publish the staged-CPU layout: a later record pass must reproduce it.
            self.commit_cpu_stage();

            // 3. Download all graph outputs.
            let mut results = HashMap::with_capacity(graph.outputs.len());
            for &out_id in &graph.outputs {
                let info = &graph.tensors[out_id.0];
                let (ptr, shape, actual_type) = gpu_tensors.get(&out_id).ok_or_else(|| {
                    SessionError::Runtime(format!("graph output '{}' was not produced", info.name))
                })?;
                if *actual_type != DataType::Float {
                    return Err(SessionError::Runtime(format!("run_f32 cannot return {:?} output {}", actual_type, info.name)));
                }
                let count = numel(shape);
                let mut host = vec![0.0f32; count];
                if count > 0 {
                    // Find matching DeviceBuffer or copy into a staging DeviceBuffer for safe D2H.
                    if let Some(buf) = run_bufs.iter().find(|b| b.handle() == *ptr) {
                        gpu.download_buffer(buf, f32_as_bytes_mut(&mut host))?;
                    } else {
                        let tmp = gpu.alloc_buffer(count * 4)?;
                        self.spatial.copy_f32(*ptr, tmp.handle(), count)?;
                        gpu.download_buffer(&tmp, f32_as_bytes_mut(&mut host))?;
                    }
                }
                results.insert(info.name.clone(), (host, shape.clone()));
            }
            // The plain run just produced the authoritative result, so a capture
            // attempt can only add speed: it is verified against `results` and
            // dropped on any doubt.
            self.maybe_capture(inputs, &results);
            Ok(results)
        }

        /// State of the static-graph path for logs, benchmarks and tests.
        pub fn graph_status(&self) -> cuda_graph::GraphStatus {
            self.graph.status()
        }

        /// Answers the call from the installed graph when it matches this shape.
        fn try_replay(
            &self,
            inputs: &HashMap<String, (&[f32], Vec<usize>)>,
        ) -> Option<HashMap<String, (Vec<f32>, Vec<usize>)>> {
            if !self.graph.is_enabled() {
                return None;
            }
            let signature =
                cuda_graph::signature(&self.plan.graph, |name| {
                    inputs.get(name).map(|(_, shape)| shape.clone())
                })?;
            match self.graph.run(self.gpu, inputs, &signature) {
                cuda_graph::SlotRun::Replay(results) => Some(results),
                cuda_graph::SlotRun::Plain => None,
            }
        }

        /// Captures a static graph for this call shape, once per shape.
        ///
        /// All failures are swallowed into [`cuda_graph::GraphStatus`]: the
        /// caller already has a correct result and must not pay for a capture
        /// that turns out not to be provable.
        fn maybe_capture(
            &self,
            inputs: &HashMap<String, (&[f32], Vec<usize>)>,
            plain: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        ) {
            if !self.graph.is_enabled() {
                return;
            }
            let graph = &self.plan.graph;
            if let Err(fallback) = cuda_graph::precheck(graph) {
                self.graph.disable(fallback);
                return;
            }
            // Without the static arena every intermediate tensor is a per-run
            // allocation, so there are no pointers to replay against.
            if std::env::var("AURORA_NATIVE_NO_ARENA").ok().as_deref() == Some("1") {
                self.graph.disable(cuda_graph::Fallback::NoArena);
                return;
            }
            let Some(signature) = cuda_graph::signature(graph, |name| {
                inputs.get(name).map(|(_, shape)| shape.clone())
            }) else {
                self.graph.disable(cuda_graph::Fallback::Unsupported(
                    "an input of this call is missing from the signature".into(),
                ));
                return;
            };
            if !self.graph.begin_capture(&signature) {
                return;
            }
            match self.capture_static_graph(&signature, inputs, plain) {
                Ok((static_graph, verify_max_abs)) => {
                    self.graph.install(static_graph, verify_max_abs)
                }
                Err(fallback) => self.graph.disable(fallback),
            }
        }

        /// One record pass over the compiled graph, then a replay checked against
        /// the result the plain path just produced.
        fn capture_static_graph(
            &self,
            signature: &cuda_graph::InputSignature,
            inputs: &HashMap<String, (&[f32], Vec<usize>)>,
            plain: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        ) -> Result<(cuda_graph::StaticGraph, f32), cuda_graph::Fallback> {
            let graph = &self.plan.graph;
            let gpu = self.gpu;

            // The plain path mirrors small inputs on the host. A recorded kernel
            // could bake a scalar derived from those values into its arguments.
            if graph.inputs.iter().any(|id| inputs.get(&graph.tensors[id.0].name)
                .is_some_and(|(v, _)| v.len() <= DEFAULT_MAX_CPU_TENSOR_ELEMS)) {
                return Err(cuda_graph::Fallback::Unsupported("runtime input may be folded on the host".into()));
            }
            if graph.outputs.iter().any(|id| graph.tensors[id.0].dtype != DataType::Float) {
                return Err(cuda_graph::Fallback::Unsupported("replay downloads float32 outputs only".into()));
            }

            // 1. Every pointer the graph will name is allocated here and owned by
            //    the `StaticGraph` until it is replaced or the session drops.
            let mut graph_inputs = Vec::with_capacity(graph.inputs.len());
            for &in_id in &graph.inputs {
                let info = &graph.tensors[in_id.0];
                let (values, shape) = inputs.get(&info.name).ok_or_else(|| {
                    cuda_graph::Fallback::Unsupported(format!(
                        "input '{}' is not supplied by the call",
                        info.name
                    ))
                })?;
                let buffer = gpu
                    .alloc_buffer((values.len() * 4).max(4))
                    .map_err(cuda_graph::Fallback::from)?;
                graph_inputs.push(cuda_graph::GraphInput {
                    name: info.name.clone(),
                    shape: shape.clone(),
                    bytes: values.len() * 4,
                    upload: gpu.alloc_pinned((values.len() * 4).max(4)).map_err(cuda_graph::Fallback::from)?,
                    buffer,
                });
            }
            let mut graph_outputs = Vec::with_capacity(graph.outputs.len());
            for &out_id in &graph.outputs {
                let info = &graph.tensors[out_id.0];
                let shape = info.static_shape().ok_or_else(|| {
                    cuda_graph::Fallback::DynamicShape(info.name.clone())
                })?;
                let element_bytes = info.dtype.byte_size().unwrap_or(4);
                let buffer = gpu
                    .alloc_buffer((numel(&shape) * element_bytes).max(4))
                    .map_err(cuda_graph::Fallback::from)?;
                let values = numel(&shape);
                graph_outputs.push(cuda_graph::GraphOutput {
                    name: info.name.clone(),
                    shape,
                    values,
                    buffer,
                });
            }

            // 2. Bind the graph exactly as this call sees it: weights, the fixed
            //    input/output buffers and the static arena.
            let mut gpu_tensors: HashMap<TensorId, (u64, Vec<usize>, DataType)> =
                self.weight_ptrs.clone();
            let mut cpu_tensors: HashMap<TensorId, CpuTensor> = self.static_cpu_tensors.clone();
            for (&in_id, input) in graph.inputs.iter().zip(&graph_inputs) {
                gpu_tensors.insert(
                    in_id,
                    (input.buffer.handle(), input.shape.clone(), DataType::Float),
                );
            }
            // Mirror the plain path's small-input CPU mirrors, so both passes take
            // the same branch for every node: a node that the plain run evaluates
            // on the CPU must not become a GPU node only inside the recording.
            for (&in_id, input) in graph.inputs.iter().zip(&graph_inputs) {
                let Some((values, _)) = inputs.get(&input.name) else {
                    continue;
                };
                if values.len() <= DEFAULT_MAX_CPU_TENSOR_ELEMS
                    && let Ok(tensor) = CpuTensor::from_f32(input.shape.clone(), values.to_vec())
                {
                    cpu_tensors.insert(in_id, tensor);
                }
            }
            let mut output_ids: HashMap<TensorId, (u64, Vec<usize>)> =
                HashMap::with_capacity(graph_outputs.len());
            for (&out_id, output) in graph.outputs.iter().zip(&graph_outputs) {
                output_ids.insert(out_id, (output.buffer.handle(), output.shape.clone()));
            }

            // 3. Record. Nothing executes; a readback, an upload or an allocation
            //    between two nodes flags the pass and drops the recording.
            let recording = cuda_graph::Recording::begin()?;
            let mut run_bufs = Vec::new();
            self.reset_cpu_stage();
            let executed = self.exec_graph(
                graph,
                true,
                true,
                &output_ids,
                &mut gpu_tensors,
                &mut cpu_tensors,
                &mut run_bufs,
            );
            if let Some(what) = recording.violation() {
                return Err(cuda_graph::Fallback::for_violation(what));
            }
            executed.map_err(|error| cuda_graph::Fallback::RecordFailed(error.to_string()))?;
            if !run_bufs.is_empty() {
                return Err(cuda_graph::Fallback::Allocation(
                    "per-run buffer allocated during the record pass",
                ));
            }
            // A view/identity may bind an output name to its source instead of
            // writing the fixed IO buffer. Never replay an uninitialised output.
            for (&id, out) in graph.outputs.iter().zip(&graph_outputs) {
                if gpu_tensors.get(&id).map(|tensor| tensor.0) != Some(out.buffer.handle()) {
                    return Err(cuda_graph::Fallback::Unsupported("graph output aliases another tensor".into()));
                }
            }
            let exec = recording.finish()?;

            // 4. Replay once on the dedicated stream and prove it against the
            //    plain result before the graph is allowed to serve a call.
            let stream = gpu
                .create_stream()
                .map_err(cuda_graph::Fallback::from)?;
            let mut static_graph = cuda_graph::StaticGraph::new(
                exec,
                stream,
                signature.clone(),
                graph_inputs,
                graph_outputs,
            );
            let replayed = static_graph
                .launch(gpu, inputs)
                .map_err(|error| cuda_graph::Fallback::LaunchFailed(format!("{error:?}")))?;
            let verify_max_abs =
                cuda_graph::compare_outputs(plain, &replayed, cuda_graph::VERIFY_TOLERANCE)?;
            Ok((static_graph, verify_max_abs))
        }

        fn exec_graph(
            &self,
            graph: &IrGraph,
            use_arena: bool,
            recording: bool,
            output_ids: &HashMap<TensorId, (u64, Vec<usize>)>,
            gpu_tensors: &mut HashMap<TensorId, (u64, Vec<usize>, DataType)>,
            cpu_tensors: &mut HashMap<TensorId, CpuTensor>,
            run_bufs: &mut Vec<DeviceBuffer<'static>>,
        ) -> Result<(), SessionError> {
            let gpu = self.gpu;
            // Staged CPU-evaluated tensors are bump-allocated from a fixed base in
            // graph order, so every pass over the same graph reproduces the layout.
            // The top-level call owns the bump cursor. If branches retain parent slots.

            // Helper: resolve output device pointer for `tid` (arena if planned, else output/dynamic buf).
            let alloc_out = |tid: TensorId,
                             shape: &[usize],
                             dtype: DataType,
                             run_bufs: &mut Vec<DeviceBuffer<'static>>|
             -> Result<u64, SessionError> {
                if let Some(&(ptr, _)) = output_ids.get(&tid) {
                    return Ok(ptr);
                }
                let bytes = numel(shape).max(1) * dtype.byte_size().unwrap_or(4);
                if use_arena {
                    if let Some(alloc) = self.plan.arena.allocations.get(&tid) {
                        if bytes <= alloc.aligned_bytes && self.arena_base_ptr != 0 {
                            return Ok(self.arena_base_ptr + alloc.offset as u64);
                        }
                    }
                }
                let buf = gpu.alloc_buffer(bytes)?;
                let ptr = buf.handle();
                run_bufs.push(buf);
                Ok(ptr)
            };

            for node in &graph.nodes {
                // Fast-path: pure CPU shape/constant evaluation when applicable.
                let is_pure_shape_op = matches!(
                    node.op,
                    Op::Constant { .. } | Op::Shape { .. } | Op::Size | Op::ConstantOfShape { .. }
                );
                let is_cpu_evaluable_op = matches!(
                    node.op,
                    Op::Gather { .. }
                        | Op::Slice
                        | Op::Concat { .. }
                        | Op::Unsqueeze { .. }
                        | Op::Squeeze { .. }
                        | Op::Reshape { .. }
                        | Op::Expand
                        | Op::Tile
                        | Op::Add
                        | Op::Sub
                        | Op::Mul
                        | Op::Div
                        | Op::Mod { .. }
                        | Op::Equal
                        | Op::Less
                        | Op::Greater
                        | Op::GreaterOrEqual
                        | Op::And
                        | Op::Where
                        | Op::Cast { .. }
                        | Op::Identity
                        | Op::Clip
                );

                // If any input of an integer/shape op is a GPU-resident Int64/Bool tensor
                // (e.g. produced by `Cast(Float->Int64)` in landmark indexing subgraphs),
                // pull those small tensors to CPU so exact integer arithmetic runs on CPU
                // instead of falling into f32 CUDA kernels.
                let has_gpu_i64_or_bool_input = is_cpu_evaluable_op
                    && !matches!(node.op, Op::Gather { .. } | Op::Slice | Op::Reshape { .. } | Op::Squeeze { .. } | Op::Unsqueeze { .. } | Op::Expand | Op::Tile)
                    && node.inputs.iter().any(|opt| {
                        opt.and_then(|id| gpu_tensors.get(&id))
                            .map_or(false, |(_, sh, dt)| {
                                matches!(dt, DataType::Int64 | DataType::Bool)
                                    && numel(sh) <= 65_536
                            })
                    });
                if has_gpu_i64_or_bool_input {
                    let all_small = node.inputs.iter().all(|opt| {
                        opt.map_or(true, |id| {
                            cpu_tensors.contains_key(&id)
                                || gpu_tensors
                                    .get(&id)
                                    .map_or(false, |(_, sh, _)| numel(sh) <= 65_536)
                        })
                    });
                    if all_small {
                        for opt in &node.inputs {
                            if let Some(id) = *opt {
                                if !cpu_tensors.contains_key(&id) {
                                    if let Some(&(ptr, ref sh, dt)) = gpu_tensors.get(&id) {
                                        let count = numel(sh);
                                        if count > 0 {
                                            let eb = dt.byte_size().unwrap_or(4);
                                            let blen = count * eb;
                                            let tmp = gpu.alloc_buffer(blen.max(4))?;
                                            self.spatial.copy_bytes_u8(ptr, tmp.handle(), blen)?;
                                            let mut raw = vec![0u8; blen];
                                            gpu.download_buffer(&tmp, &mut raw)?;
                                            let ct_opt = match dt {
                                                DataType::Int64 => CpuTensor::from_i64(
                                                    sh.clone(),
                                                    raw.chunks_exact(8)
                                                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                                                        .collect(),
                                                )
                                                .ok(),
                                                DataType::Bool => CpuTensor::from_bool(
                                                    sh.clone(),
                                                    raw.iter().map(|&b| b != 0).collect(),
                                                )
                                                .ok(),
                                                DataType::Float => CpuTensor::from_f32(
                                                    sh.clone(),
                                                    raw.chunks_exact(4)
                                                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                                                        .collect(),
                                                )
                                                .ok(),
                                                _ => None,
                                            };
                                            if let Some(ct) = ct_opt {
                                                cpu_tensors.insert(id, ct);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                let all_inputs_on_cpu = !node.inputs.is_empty()
                    && node
                        .inputs
                        .iter()
                        .all(|opt| opt.map_or(true, |id| cpu_tensors.contains_key(&id)));

                if is_pure_shape_op
                    || (all_inputs_on_cpu
                        && is_cpu_evaluable_op
                        && node.outputs.len() == 1
                        && node.outputs[0]
                            .map_or(false, |oid| !graph.outputs.contains(&oid)))
                {
                    let inp_vals: Vec<Option<&CpuTensor>> = node
                        .inputs
                        .iter()
                        .map(|opt| opt.and_then(|id| cpu_tensors.get(&id)))
                        .collect();
                    let inp_shapes: Vec<Option<&[usize]>> = node
                        .inputs
                        .iter()
                        .map(|opt| {
                            opt.and_then(|id| {
                                gpu_tensors
                                    .get(&id)
                                    .map(|(_, s, _)| s.as_slice())
                                    .or_else(|| cpu_tensors.get(&id).map(|c| c.shape.as_slice()))
                            })
                        })
                        .collect();
                    let cpu_res = if matches!(node.op, Op::Clip) {
                        match inp_vals.first().and_then(|o| *o) {
                            Some(x_ct) => match &x_ct.data {
                                CpuData::I64(v) => {
                                    let lo = inp_vals
                                        .get(1)
                                        .and_then(|o| *o)
                                        .and_then(|c| c.to_i64_vec().into_iter().next())
                                        .unwrap_or(i64::MIN);
                                    let hi = inp_vals
                                        .get(2)
                                        .and_then(|o| *o)
                                        .and_then(|c| c.to_i64_vec().into_iter().next())
                                        .unwrap_or(i64::MAX);
                                    CpuTensor::from_i64(
                                        x_ct.shape.clone(),
                                        v.iter().map(|&x| x.clamp(lo, hi)).collect(),
                                    )
                                }
                                CpuData::F32(v) => {
                                    let lo = inp_vals
                                        .get(1)
                                        .and_then(|o| *o)
                                        .and_then(|c| c.to_f32_vec().into_iter().next())
                                        .unwrap_or(f32::NEG_INFINITY);
                                    let hi = inp_vals
                                        .get(2)
                                        .and_then(|o| *o)
                                        .and_then(|c| c.to_f32_vec().into_iter().next())
                                        .unwrap_or(f32::INFINITY);
                                    CpuTensor::from_f32(
                                        x_ct.shape.clone(),
                                        v.iter().map(|&x| x.clamp(lo, hi)).collect(),
                                    )
                                }
                                _ => eval_cpu_node(&node.op, &inp_vals, &inp_shapes),
                            },
                            None => eval_cpu_node(&node.op, &inp_vals, &inp_shapes),
                        }
                    } else {
                        eval_cpu_node(&node.op, &inp_vals, &inp_shapes)
                    };
                    if let Ok(ct) = cpu_res {
                        if let Some(Some(out_id)) = node.outputs.first() {
                            // A `Constant` payload was already uploaded with the
                            // weights. Re-uploading it would allocate a fresh
                            // device buffer on every call, which costs a
                            // `cuMemAlloc` + H2D per run and gives a captured
                            // graph a pointer that dies with the call.
                            if matches!(node.op, Op::Constant { .. })
                                && gpu_tensors.get(out_id).is_some_and(|(_, shape, dtype)| {
                                    shape.as_slice() == ct.shape.as_slice()
                                        && *dtype == ct.data.dtype()
                                })
                            {
                                cpu_tensors.insert(*out_id, ct);
                                continue;
                            }
                            // Also mirror small F32/I64/Bool constants to GPU if needed by downstream GPU ops.
                            if let CpuData::F32(ref v) = ct.data {
                                if !v.is_empty() {
                                    let ptr = self.stage_cpu_tensor(f32_as_bytes(v), recording)?;
                                    gpu_tensors.insert(*out_id, (ptr, ct.shape.clone(), DataType::Float));
                                }
                            } else if let CpuData::I64(ref v) = ct.data {
                                if !v.is_empty() {
                                    let bytes: &[u8] = unsafe {
                                        std::slice::from_raw_parts(
                                            v.as_ptr().cast::<u8>(),
                                            std::mem::size_of_val(v.as_slice()),
                                        )
                                    };
                                    let ptr = self.stage_cpu_tensor(bytes, recording)?;
                                    gpu_tensors.insert(*out_id, (ptr, ct.shape.clone(), DataType::Int64));
                                }
                            } else if let CpuData::Bool(ref v) = ct.data {
                                if !v.is_empty() {
                                    let bytes: Vec<u8> = v.iter().map(|&b| u8::from(b)).collect();
                                    let ptr = self.stage_cpu_tensor(&bytes, recording)?;
                                    gpu_tensors.insert(*out_id, (ptr, ct.shape.clone(), DataType::Bool));
                                }
                            }
                            cpu_tensors.insert(*out_id, ct);
                            continue;
                        }
                    }
                }

                // Helper to fetch required GPU tensor.
                let req_gpu = |idx: usize| -> Result<(u64, Vec<usize>, DataType), SessionError> {
                    let tid = node
                        .inputs
                        .get(idx)
                        .and_then(|o| *o)
                        .ok_or_else(|| {
                            SessionError::Runtime(format!(
                                "node '{}' ({}) missing input #{idx}",
                                node.name,
                                node.op.name()
                            ))
                        })?;
                    gpu_tensors.get(&tid).cloned().ok_or_else(|| {
                        SessionError::Runtime(format!(
                            "node '{}' ({}) input '{}' not resident on GPU",
                            node.name,
                            node.op.name(),
                            graph.tensors[tid.0].name
                        ))
                    })
                };
                let opt_gpu = |idx: usize| -> Option<(u64, Vec<usize>, DataType)> {
                    node.inputs
                        .get(idx)
                        .and_then(|o| *o)
                        .and_then(|tid| gpu_tensors.get(&tid).cloned())
                };
                let req_out_id = |idx: usize| -> Result<TensorId, SessionError> {
                    node.outputs.get(idx).and_then(|o| *o).ok_or_else(|| {
                        SessionError::Runtime(format!(
                            "node '{}' ({}) missing output #{idx}",
                            node.name,
                            node.op.name()
                        ))
                    })
                };
                let read_i64_vec = |tid: TensorId| -> Result<Vec<i64>, SessionError> {
                    if let Some(ct) = cpu_tensors.get(&tid) {
                        return Ok(ct.to_i64_vec());
                    }
                    if let Some(&(ptr, ref sh, dt)) = gpu_tensors.get(&tid) {
                        let count = numel(sh);
                        if count == 0 {
                            return Ok(Vec::new());
                        }
                        let eb = dt.byte_size().unwrap_or(4);
                        let bytes_len = count * eb;
                        let tmp = gpu.alloc_buffer(bytes_len.max(4))?;
                        self.spatial.copy_bytes_u8(ptr, tmp.handle(), bytes_len)?;
                        let mut raw = vec![0u8; bytes_len];
                        gpu.download_buffer(&tmp, &mut raw)?;
                        return match dt {
                            DataType::Int64 => Ok(raw
                                .chunks_exact(8)
                                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                                .collect()),
                            DataType::Int32 => Ok(raw
                                .chunks_exact(4)
                                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                                .collect()),
                            DataType::Float => Ok(raw
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as i64)
                                .collect()),
                            _ => Err(SessionError::Runtime(format!(
                                "cannot read {:?} GPU tensor as i64",
                                dt
                            ))),
                        };
                    }
                    Err(SessionError::Runtime(format!(
                        "tensor '{}' not found on CPU or GPU",
                        graph.tensors[tid.0].name
                    )))
                };

                match &node.op {
                    // --- View ops (copy into arena slot to respect arena liveness) ---
                    Op::Identity => {
                        let (p, sh, dt) = req_gpu(0)?;
                        let out_id = req_out_id(0)?;
                        let optr = alloc_out(out_id, &sh, dt, run_bufs)?;
                        let total = numel(&sh);
                        if optr != p && total > 0 {
                            if dt == DataType::Float {
                                self.spatial.copy_f32(p, optr, total)?;
                            } else {
                                let eb = dt.byte_size().unwrap_or(4);
                                self.spatial.copy_bytes_u8(p, optr, total * eb)?;
                            }
                            gpu_tensors.insert(out_id, (optr, sh, dt));
                        } else {
                            gpu_tensors.insert(out_id, (p, sh, dt));
                        }
                    }
                    Op::Reshape { allowzero } => {
                        let (p, in_sh, dt) = req_gpu(0)?;
                        let out_id = req_out_id(0)?;
                        let target_spec = node
                            .inputs
                            .get(1)
                            .and_then(|o| *o)
                            .and_then(|id| read_i64_vec(id).ok())
                            .or_else(|| {
                                graph.tensors[out_id.0]
                                    .static_shape()
                                    .map(|s| s.into_iter().map(|d| d as i64).collect())
                            })
                            .ok_or_else(|| {
                                SessionError::Runtime(format!(
                                    "Reshape '{}' missing target shape",
                                    node.name
                                ))
                            })?;
                        let total = numel(&in_sh);
                        let mut out_sh = Vec::with_capacity(target_spec.len());
                        let mut neg_idx = None;
                        let mut known_prod = 1usize;
                        for (i, &d) in target_spec.iter().enumerate() {
                            if d == -1 {
                                neg_idx = Some(i);
                                out_sh.push(1);
                            } else if d == 0 && !allowzero {
                                let v = in_sh.get(i).copied().unwrap_or(1);
                                known_prod *= v;
                                out_sh.push(v);
                            } else {
                                let v = d as usize;
                                known_prod *= v;
                                out_sh.push(v);
                            }
                        }
                        if let Some(ni) = neg_idx {
                            out_sh[ni] = if known_prod > 0 { total / known_prod } else { 0 };
                        }
                        let optr = alloc_out(out_id, &out_sh, dt, run_bufs)?;
                        let out_elems = numel(&out_sh);
                        if optr != p && out_elems > 0 {
                            if dt == DataType::Float {
                                self.spatial.copy_f32(p, optr, out_elems)?;
                            } else {
                                let eb = dt.byte_size().unwrap_or(4);
                                self.spatial.copy_bytes_u8(p, optr, out_elems * eb)?;
                            }
                            gpu_tensors.insert(out_id, (optr, out_sh, dt));
                        } else {
                            gpu_tensors.insert(out_id, (p, out_sh, dt));
                        }
                    }
                    Op::Squeeze { axes } => {
                        let (p, in_sh, dt) = req_gpu(0)?;
                        let out_id = req_out_id(0)?;
                        let axes_v: Option<Vec<usize>> = if let Some(a) = axes {
                            Some(
                                a.iter()
                                    .map(|&ax| norm_axis(ax, in_sh.len()))
                                    .collect::<Result<Vec<_>, _>>()?,
                            )
                        } else if let Some(Some(aid)) = node.inputs.get(1) {
                            cpu_tensors.get(aid).map(|c| {
                                c.to_i64_vec()
                                    .into_iter()
                                    .filter_map(|ax| norm_axis(ax, in_sh.len()).ok())
                                    .collect()
                            })
                        } else {
                            None
                        };
                        let out_sh: Vec<usize> = in_sh
                            .iter()
                            .enumerate()
                            .filter(|&(i, &d)| match &axes_v {
                                Some(a) => !a.contains(&i),
                                None => d != 1,
                            })
                            .map(|(_, &d)| d)
                            .collect();
                        let optr = alloc_out(out_id, &out_sh, dt, run_bufs)?;
                        let out_elems = numel(&out_sh);
                        if optr != p && out_elems > 0 {
                            if dt == DataType::Float {
                                self.spatial.copy_f32(p, optr, out_elems)?;
                            } else {
                                let eb = dt.byte_size().unwrap_or(4);
                                self.spatial.copy_bytes_u8(p, optr, out_elems * eb)?;
                            }
                            gpu_tensors.insert(out_id, (optr, out_sh, dt));
                        } else {
                            gpu_tensors.insert(out_id, (p, out_sh, dt));
                        }
                    }
                    Op::Unsqueeze { axes } => {
                        let (p, in_sh, dt) = req_gpu(0)?;
                        let out_id = req_out_id(0)?;
                        let raw_axes: Vec<i64> = if let Some(a) = axes {
                            a.clone()
                        } else if let Some(Some(aid)) = node.inputs.get(1) {
                            cpu_tensors
                                .get(aid)
                                .map(|c| c.to_i64_vec())
                                .unwrap_or_default()
                        } else {
                            Vec::new()
                        };
                        let out_rank = in_sh.len() + raw_axes.len();
                        let mut norm: Vec<usize> = raw_axes
                            .into_iter()
                            .map(|ax| norm_axis(ax, out_rank))
                            .collect::<Result<Vec<_>, _>>()?;
                        norm.sort_unstable();
                        let mut out_sh = Vec::with_capacity(out_rank);
                        let mut src_i = 0usize;
                        for r in 0..out_rank {
                            if norm.contains(&r) {
                                out_sh.push(1);
                            } else {
                                out_sh.push(in_sh[src_i]);
                                src_i += 1;
                            }
                        }
                        let optr = alloc_out(out_id, &out_sh, dt, run_bufs)?;
                        let out_elems = numel(&out_sh);
                        if optr != p && out_elems > 0 {
                            if dt == DataType::Float {
                                self.spatial.copy_f32(p, optr, out_elems)?;
                            } else {
                                let eb = dt.byte_size().unwrap_or(4);
                                self.spatial.copy_bytes_u8(p, optr, out_elems * eb)?;
                            }
                            gpu_tensors.insert(out_id, (optr, out_sh, dt));
                        } else {
                            gpu_tensors.insert(out_id, (p, out_sh, dt));
                        }
                    }

                    // --- Convolution & MatMul ---
                    Op::Conv(attrs) => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let (w_ptr, w_sh, _) = req_gpu(1)?;
                        let bias_ptr = opt_gpu(2).map(|(p, _, _)| p);
                        let params = Conv2dParams::from_attrs(
                            &x_sh,
                            &w_sh,
                            attrs,
                            self.plan.conv_activations.get(&node.id).copied().unwrap_or(FusedActivation::None),
                        )?;
                        let out_sh = vec![params.batch, params.c_out, params.out_h, params.out_w];
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        let ws_ptr = if !params.is_depthwise()
                            && !params.is_direct_1x1()
                            && params.im2col_workspace_bytes() > self.im2col_bytes
                        {
                            let tmp = gpu.alloc_buffer(params.im2col_workspace_bytes())?;
                            let p = tmp.handle();
                            run_bufs.push(tmp);
                            p
                        } else {
                            self.im2col_ptr
                        };
                        self.conv
                            .conv2d_nchw(x_ptr, w_ptr, bias_ptr, y_ptr, ws_ptr, &params, None)?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }
                    Op::ConvTranspose(attrs) => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let (w_ptr, w_sh, _) = req_gpu(1)?;
                        let bias_ptr = opt_gpu(2).map(|(p, _, _)| p);
                        let params = ConvTranspose2dParams::from_attrs(
                            &x_sh,
                            &w_sh,
                            attrs,
                            FusedActivation::None,
                        )?;
                        let out_sh = vec![params.batch, params.c_out, params.out_h, params.out_w];
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        self.conv.conv_transpose2d_nchw(
                            x_ptr, w_ptr, bias_ptr, y_ptr, &params, None,
                        )?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }
                    Op::MatMul => {
                        let (a_ptr, a_sh, _) = req_gpu(0)?;
                        let (b_ptr, b_sh, _) = req_gpu(1)?;
                        let out_id = req_out_id(0)?;
                        let m = a_sh[a_sh.len() - 2];
                        let n = b_sh[b_sh.len() - 1];
                        let a_batch = &a_sh[..a_sh.len() - 2];
                        let b_batch = &b_sh[..b_sh.len() - 2];
                        let br = a_batch.len().max(b_batch.len());
                        let mut out_sh = Vec::with_capacity(br + 2);
                        for i in 0..br {
                            let da = if i + a_batch.len() >= br {
                                a_batch[i + a_batch.len() - br]
                            } else {
                                1
                            };
                            let db = if i + b_batch.len() >= br {
                                b_batch[i + b_batch.len() - br]
                            } else {
                                1
                            };
                            out_sh.push(da.max(db));
                        }
                        out_sh.push(m);
                        out_sh.push(n);
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        self.conv
                            .batched_matmul(a_ptr, b_ptr, y_ptr, &a_sh, &b_sh, None)?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }
                    Op::Gemm(attrs) => {
                        let (a_ptr, a_sh, _) = req_gpu(0)?;
                        let (b_ptr, b_sh, _) = req_gpu(1)?;
                        let c_opt = opt_gpu(2);
                        let c_ptr = c_opt.as_ref().map(|(p, _, _)| *p);
                        let c_sh = c_opt.as_ref().map(|(_, s, _)| s.as_slice());
                        let m = if attrs.trans_a { a_sh[1] } else { a_sh[0] };
                        let n = if attrs.trans_b { b_sh[0] } else { b_sh[1] };
                        let out_sh = vec![m, n];
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        self.conv
                            .gemm(a_ptr, b_ptr, c_ptr, c_sh, y_ptr, &a_sh, &b_sh, attrs, None)?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }

                    // --- Normalization & Softmax ---
                    Op::BatchNormFused => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let (s_ptr, _, _) = req_gpu(1)?;
                        let (b_ptr, _, _) = req_gpu(2)?;
                        let (n, c, h, w) = match x_sh.as_slice() {
                            &[n, c] => (n, c, 1, 1),
                            &[n, c, l] => (n, c, l, 1),
                            &[n, c, h, w] => (n, c, h, w),
                            _ => {
                                return Err(SessionError::Runtime(format!(
                                    "BatchNormFused '{}' unsupported shape {:?}",
                                    node.name, x_sh
                                )));
                            }
                        };
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &x_sh, DataType::Float, run_bufs)?;
                        self.elem
                            .batch_norm_fused_nchw_f32(x_ptr, s_ptr, b_ptr, y_ptr, n, c, h, w)?;
                        gpu_tensors.insert(out_id, (y_ptr, x_sh, DataType::Float));
                    }
                    Op::LayerNormalization(attrs) => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let s_ptr = opt_gpu(1).map(|(p, _, _)| p);
                        let b_ptr = opt_gpu(2).map(|(p, _, _)| p);
                        let norm_dim = *x_sh.last().unwrap_or(&1);
                        let rows = numel(&x_sh) / norm_dim.max(1);
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &x_sh, DataType::Float, run_bufs)?;
                        self.elem.layer_norm_last_f32(
                            x_ptr,
                            s_ptr,
                            b_ptr,
                            y_ptr,
                            rows,
                            norm_dim,
                            attrs.epsilon,
                        )?;
                        gpu_tensors.insert(out_id, (y_ptr, x_sh, DataType::Float));
                    }
                    Op::InstanceNormalization { epsilon } => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let s_ptr = opt_gpu(1).map(|(p, _, _)| p);
                        let b_ptr = opt_gpu(2).map(|(p, _, _)| p);
                        let (n, c, h, w) = match x_sh.as_slice() {
                            &[n, c, l] => (n, c, l, 1),
                            &[n, c, h, w] => (n, c, h, w),
                            _ => {
                                return Err(SessionError::Runtime(format!(
                                    "InstanceNormalization '{}' unsupported shape {:?}",
                                    node.name, x_sh
                                )));
                            }
                        };
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &x_sh, DataType::Float, run_bufs)?;
                        self.elem
                            .instance_norm_nchw_f32(x_ptr, s_ptr, b_ptr, y_ptr, n, c, h, w, *epsilon)?;
                        gpu_tensors.insert(out_id, (y_ptr, x_sh, DataType::Float));
                    }
                    Op::Softmax { axis } => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let ax = norm_axis(*axis, x_sh.len())?;
                        let outer: usize = x_sh[..ax].iter().product();
                        let axis_dim = x_sh[ax];
                        let inner: usize = x_sh[ax + 1..].iter().product();
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &x_sh, DataType::Float, run_bufs)?;
                        self.elem
                            .softmax_axis_f32(x_ptr, y_ptr, outer, axis_dim, inner)?;
                        gpu_tensors.insert(out_id, (y_ptr, x_sh, DataType::Float));
                    }

                    // --- Elementwise Binary / Unary / Cast ---
                    Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Pow => {
                        let (a_ptr, a_sh, _) = req_gpu(0)?;
                        let (b_ptr, b_sh, _) = req_gpu(1)?;
                        let rank = a_sh.len().max(b_sh.len());
                        let mut out_sh = Vec::with_capacity(rank);
                        for i in 0..rank {
                            let da = if i + a_sh.len() >= rank {
                                a_sh[i + a_sh.len() - rank]
                            } else {
                                1
                            };
                            let db = if i + b_sh.len() >= rank {
                                b_sh[i + b_sh.len() - rank]
                            } else {
                                1
                            };
                            out_sh.push(da.max(db));
                        }
                        let bop = match node.op {
                            Op::Add => BinaryOp::Add,
                            Op::Sub => BinaryOp::Sub,
                            Op::Mul => BinaryOp::Mul,
                            Op::Div => BinaryOp::Div,
                            Op::Pow => BinaryOp::Pow,
                            _ => unreachable!(),
                        };
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        self.elem
                            .binary_broadcast_f32(bop, a_ptr, &a_sh, b_ptr, &b_sh, y_ptr)
                            .map_err(|e| {
                                SessionError::Runtime(format!(
                                    "{} '{}' a={a_sh:?} b={b_sh:?}: {e:?}",
                                    node.op.name(),
                                    node.name
                                ))
                            })?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }
                    Op::Equal | Op::Greater | Op::GreaterOrEqual | Op::Less => {
                        let (a_ptr, a_sh, _) = req_gpu(0)?;
                        let (b_ptr, b_sh, _) = req_gpu(1)?;
                        let rank = a_sh.len().max(b_sh.len());
                        let mut out_sh = Vec::with_capacity(rank);
                        for i in 0..rank {
                            let da = if i + a_sh.len() >= rank {
                                a_sh[i + a_sh.len() - rank]
                            } else {
                                1
                            };
                            let db = if i + b_sh.len() >= rank {
                                b_sh[i + b_sh.len() - rank]
                            } else {
                                1
                            };
                            out_sh.push(da.max(db));
                        }
                        let cop = match node.op {
                            Op::Equal => CompareOp::Equal,
                            Op::Greater => CompareOp::Greater,
                            Op::GreaterOrEqual => CompareOp::GreaterOrEqual,
                            Op::Less => CompareOp::Less,
                            _ => unreachable!(),
                        };
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Bool, run_bufs)?;
                        self.elem
                            .compare_broadcast_f32(cop, a_ptr, &a_sh, b_ptr, &b_sh, y_ptr)?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Bool));
                    }
                    Op::And => {
                        let (a_ptr, a_sh, _) = req_gpu(0)?;
                        let (b_ptr, b_sh, _) = req_gpu(1)?;
                        let rank = a_sh.len().max(b_sh.len());
                        let mut out_sh = Vec::with_capacity(rank);
                        for i in 0..rank {
                            let da = if i + a_sh.len() >= rank {
                                a_sh[i + a_sh.len() - rank]
                            } else {
                                1
                            };
                            let db = if i + b_sh.len() >= rank {
                                b_sh[i + b_sh.len() - rank]
                            } else {
                                1
                            };
                            out_sh.push(da.max(db));
                        }
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Bool, run_bufs)?;
                        self.elem
                            .and_broadcast_u8(a_ptr, &a_sh, b_ptr, &b_sh, y_ptr)?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Bool));
                    }
                    Op::Where => {
                        let (c_ptr, c_sh, _) = req_gpu(0)?;
                        let (x_ptr, x_sh, _) = req_gpu(1)?;
                        let (y_ptr_in, y_sh, _) = req_gpu(2)?;
                        let rank = c_sh.len().max(x_sh.len()).max(y_sh.len());
                        let mut out_sh = Vec::with_capacity(rank);
                        for i in 0..rank {
                            let dc = if i + c_sh.len() >= rank { c_sh[i + c_sh.len() - rank] } else { 1 };
                            let dx = if i + x_sh.len() >= rank { x_sh[i + x_sh.len() - rank] } else { 1 };
                            let dy = if i + y_sh.len() >= rank { y_sh[i + y_sh.len() - rank] } else { 1 };
                            out_sh.push(dc.max(dx).max(dy));
                        }
                        let out_id = req_out_id(0)?;
                        let out_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        self.elem.where_broadcast_f32(
                            c_ptr, &c_sh, x_ptr, &x_sh, y_ptr_in, &y_sh, out_ptr,
                        )?;
                        gpu_tensors.insert(out_id, (out_ptr, out_sh, DataType::Float));
                    }
                    Op::PRelu => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let (s_ptr, mut s_sh, _) = req_gpu(1)?;
                        // ONNX unidirectional broadcast: 1D slope [C] on NCHW [N,C,H,W] -> [1,C,1,1].
                        if s_sh.len() == 1 && x_sh.len() == 4 && s_sh[0] == x_sh[1] {
                            s_sh = vec![1, s_sh[0], 1, 1];
                        }
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &x_sh, DataType::Float, run_bufs)?;
                        self.elem
                            .prelu_broadcast_f32(x_ptr, &x_sh, s_ptr, &s_sh, y_ptr)
                            .map_err(|e| {
                                SessionError::Runtime(format!(
                                    "PRelu '{}' x={x_sh:?} s={s_sh:?}: {e:?}",
                                    node.name
                                ))
                            })?;
                        gpu_tensors.insert(out_id, (y_ptr, x_sh, DataType::Float));
                    }
                    Op::Relu
                    | Op::Gelu
                    | Op::SiLU
                    | Op::Sigmoid
                    | Op::Erf
                    | Op::Sqrt
                    | Op::Reciprocal
                    | Op::Abs
                    | Op::Sin
                    | Op::Cos
                    | Op::HardSwish
                    | Op::HardSigmoid { .. }
                    | Op::LeakyRelu { .. }
                    | Op::Clip => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let (uop, p0, p1) = match node.op {
                            Op::Relu => (UnaryOp::Relu, 0.0, 0.0),
                            Op::Gelu => (UnaryOp::Gelu, 0.0, 0.0),
                            Op::SiLU => (UnaryOp::SiLU, 0.0, 0.0),
                            Op::Sigmoid => (UnaryOp::Sigmoid, 0.0, 0.0),
                            Op::Erf => (UnaryOp::Erf, 0.0, 0.0),
                            Op::Sqrt => (UnaryOp::Sqrt, 0.0, 0.0),
                            Op::Reciprocal => (UnaryOp::Reciprocal, 0.0, 0.0),
                            Op::Abs => (UnaryOp::Abs, 0.0, 0.0),
                            Op::Sin => (UnaryOp::Sin, 0.0, 0.0),
                            Op::Cos => (UnaryOp::Cos, 0.0, 0.0),
                            Op::HardSwish => (UnaryOp::HardSwish, 0.0, 0.0),
                            Op::HardSigmoid { alpha, beta } => (UnaryOp::HardSigmoid, alpha, beta),
                            Op::LeakyRelu { alpha } => (UnaryOp::LeakyRelu, alpha, 0.0),
                            Op::Clip => {
                                let lo = node
                                    .inputs
                                    .get(1)
                                    .and_then(|o| *o)
                                    .and_then(|id| cpu_tensors.get(&id))
                                    .and_then(|c| c.to_f32_vec().into_iter().next())
                                    .unwrap_or(f32::NEG_INFINITY);
                                let hi = node
                                    .inputs
                                    .get(2)
                                    .and_then(|o| *o)
                                    .and_then(|id| cpu_tensors.get(&id))
                                    .and_then(|c| c.to_f32_vec().into_iter().next())
                                    .unwrap_or(f32::INFINITY);
                                (UnaryOp::Clip, lo, hi)
                            }
                            _ => unreachable!(),
                        };
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &x_sh, DataType::Float, run_bufs)?;
                        self.elem
                            .unary_f32(uop, x_ptr, y_ptr, numel(&x_sh), p0, p1)?;
                        gpu_tensors.insert(out_id, (y_ptr, x_sh, DataType::Float));
                    }
                    Op::Cast { to } => {
                        let (x_ptr, x_sh, src_dt) = req_gpu(0)?;
                        let out_id = req_out_id(0)?;
                        let total = numel(&x_sh);
                        if src_dt == *to {
                            let y_ptr = alloc_out(out_id, &x_sh, *to, run_bufs)?;
                            if y_ptr != x_ptr && total > 0 {
                                if *to == DataType::Float {
                                    self.spatial.copy_f32(x_ptr, y_ptr, total)?;
                                } else {
                                    let eb = to.byte_size().unwrap_or(4);
                                    self.spatial.copy_bytes_u8(x_ptr, y_ptr, total * eb)?;
                                }
                                gpu_tensors.insert(out_id, (y_ptr, x_sh, *to));
                            } else {
                                gpu_tensors.insert(out_id, (x_ptr, x_sh, *to));
                            }
                        } else if src_dt == DataType::Float && *to == DataType::Float16 {
                            // Keep activations in FP32 storage for downstream FP32 kernels,
                            // while rounding through IEEE FP16 on GPU to preserve exact FP16 semantics.
                            let y_ptr = alloc_out(out_id, &x_sh, DataType::Float, run_bufs)?;
                            if total > 0 {
                                // The load-time staging buffer keeps the pointer
                                // stable so the two casts can live in a graph; a
                                // dynamic shape that outgrew it still gets its own.
                                let tmp_ptr = match self.f16_scratch.as_ref() {
                                    Some(buffer) if self.f16_scratch_elems >= total => {
                                        buffer.handle()
                                    }
                                    _ => {
                                        let scratch = gpu.alloc_buffer(total.max(1) * 2)?;
                                        let ptr = scratch.handle();
                                        run_bufs.push(scratch);
                                        ptr
                                    }
                                };
                                self.elem.cast_f32_to_f16(x_ptr, tmp_ptr, total)?;
                                self.elem.cast_f16_to_f32(tmp_ptr, y_ptr, total)?;
                            }
                            gpu_tensors.insert(out_id, (y_ptr, x_sh, DataType::Float));
                        } else {
                            let y_ptr = alloc_out(out_id, &x_sh, *to, run_bufs)?;
                            match (src_dt, *to) {
                                (DataType::Float16, DataType::Float) => {
                                    self.elem.cast_f16_to_f32(x_ptr, y_ptr, total)?;
                                }
                                (DataType::Int64, DataType::Float) => {
                                    self.elem.cast_i64_to_f32(x_ptr, y_ptr, total)?;
                                }
                                (DataType::Int32, DataType::Float) => {
                                    self.elem.cast_i32_to_f32(x_ptr, y_ptr, total)?;
                                }
                                (DataType::Uint8, DataType::Float) => {
                                    self.elem.cast_u8_to_f32(x_ptr, y_ptr, total)?;
                                }
                                (DataType::Float, DataType::Int64) => {
                                    self.elem.cast_f32_to_i64(x_ptr, y_ptr, total)?;
                                }
                                (DataType::Bool, DataType::Float) => {
                                    // The scalar operands are load-time constants;
                                    // a per-run allocation would break capture.
                                    let (one_ptr, zero_ptr) =
                                        match (self.const_one.as_ref(), self.const_zero.as_ref())
                                        {
                                            (Some(one), Some(zero)) => {
                                                (one.handle(), zero.handle())
                                            }
                                            _ => {
                                                let one = gpu.alloc_buffer(4)?;
                                                let zero = gpu.alloc_buffer(4)?;
                                                gpu.upload_buffer(&one, f32_as_bytes(&[1.0f32]))?;
                                                gpu.upload_buffer(&zero, f32_as_bytes(&[0.0f32]))?;
                                                let ptrs = (one.handle(), zero.handle());
                                                run_bufs.push(one);
                                                run_bufs.push(zero);
                                                ptrs
                                            }
                                        };
                                    self.elem.where_broadcast_f32(
                                        x_ptr,
                                        &x_sh,
                                        one_ptr,
                                        &[1],
                                        zero_ptr,
                                        &[1],
                                        y_ptr,
                                    )?;
                                }
                                _ => {
                                    return Err(SessionError::Runtime(format!(
                                        "unsupported GPU Cast {:?} -> {:?} at '{}'",
                                        src_dt, to, node.name
                                    )));
                                }
                            }
                            gpu_tensors.insert(out_id, (y_ptr, x_sh, *to));
                        }
                    }

                    // --- Pooling, Resize, Pad ---
                    Op::MaxPool(attrs) => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let (n, c, h, w) = (x_sh[0], x_sh[1], x_sh[2], x_sh[3]);
                        let kh = attrs.kernel_shape[0];
                        let kw = attrs.kernel_shape[1];
                        let sh = attrs.strides.first().copied().unwrap_or(1);
                        let sw = attrs.strides.get(1).copied().unwrap_or(1);
                        let pad_t = attrs.pads.first().copied().unwrap_or(0);
                        let pad_l = attrs.pads.get(1).copied().unwrap_or(0);
                        let pad_b = attrs.pads.get(2).copied().unwrap_or(pad_t);
                        let pad_r = attrs.pads.get(3).copied().unwrap_or(pad_l);
                        let dh = attrs.dilations.first().copied().unwrap_or(1);
                        let dw = attrs.dilations.get(1).copied().unwrap_or(1);
                        let out_h = pool2d_out_dim(h, kh, sh, pad_t, pad_b, dh, attrs.ceil_mode)?;
                        let out_w = pool2d_out_dim(w, kw, sw, pad_l, pad_r, dw, attrs.ceil_mode)?;
                        let out_sh = vec![n, c, out_h, out_w];
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        self.spatial.maxpool2d_nchw_f32(
                            x_ptr, y_ptr, n, c, h, w, out_h, out_w, kh, kw, sh, sw, pad_t, pad_l,
                            dh, dw,
                        )?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }
                    Op::AveragePool(attrs) => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let (n, c, h, w) = (x_sh[0], x_sh[1], x_sh[2], x_sh[3]);
                        let kh = attrs.kernel_shape[0];
                        let kw = attrs.kernel_shape[1];
                        let sh = attrs.strides.first().copied().unwrap_or(1);
                        let sw = attrs.strides.get(1).copied().unwrap_or(1);
                        let pad_t = attrs.pads.first().copied().unwrap_or(0);
                        let pad_l = attrs.pads.get(1).copied().unwrap_or(0);
                        let pad_b = attrs.pads.get(2).copied().unwrap_or(pad_t);
                        let pad_r = attrs.pads.get(3).copied().unwrap_or(pad_l);
                        let out_h = pool2d_out_dim(h, kh, sh, pad_t, pad_b, 1, attrs.ceil_mode)?;
                        let out_w = pool2d_out_dim(w, kw, sw, pad_l, pad_r, 1, attrs.ceil_mode)?;
                        let out_sh = vec![n, c, out_h, out_w];
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        self.spatial.avgpool2d_nchw_f32(
                            x_ptr,
                            y_ptr,
                            n,
                            c,
                            h,
                            w,
                            out_h,
                            out_w,
                            kh,
                            kw,
                            sh,
                            sw,
                            pad_t,
                            pad_l,
                            attrs.count_include_pad,
                        )?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }
                    Op::GlobalAveragePool => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let (n, c, h, w, out_sh) = if x_sh.len() == 4 {
                            (x_sh[0], x_sh[1], x_sh[2], x_sh[3], vec![x_sh[0], x_sh[1], 1, 1])
                        } else if x_sh.len() == 3 {
                            (x_sh[0], x_sh[1], x_sh[2], 1, vec![x_sh[0], x_sh[1], 1])
                        } else {
                            return Err(SessionError::Runtime(format!(
                                "GlobalAveragePool '{}' unsupported rank {:?}",
                                node.name, x_sh
                            )));
                        };
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        self.spatial.global_avgpool_nchw_f32(x_ptr, y_ptr, n, c, h, w)?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }
                    Op::ReduceMean(attrs) | Op::ReduceSum(attrs) | Op::ReduceMax(attrs) => {
                        let (x_ptr, x_sh, dtype) = req_gpu(0)?;
                        let axes = if let Some(axes) = &attrs.axes {
                            axes.clone()
                        } else if let Some(Some(axes_id)) = node.inputs.get(1) {
                            read_i64_vec(*axes_id)?
                        } else {
                            Vec::new()
                        };
                        let out_id = req_out_id(0)?;
                        if axes.is_empty() && attrs.noop_with_empty_axes {
                            let y_ptr = alloc_out(out_id, &x_sh, dtype, run_bufs)?;
                            let bytes = numel(&x_sh) * dtype.byte_size().unwrap_or(4);
                            if bytes > 0 && x_ptr != y_ptr {
                                self.spatial.copy_bytes_u8(x_ptr, y_ptr, bytes)?;
                            }
                            gpu_tensors.insert(out_id, (y_ptr, x_sh, dtype));
                            continue;
                        }
                        if dtype != DataType::Float {
                            return Err(SessionError::Runtime(format!(
                                "{} '{}' cannot reduce {dtype:?}", node.op.name(), node.name
                            )));
                        }
                        let norm_axes: Vec<usize> = axes.iter()
                            .map(|&axis| norm_axis(axis, x_sh.len()))
                            .collect::<Result<_, _>>()?;
                        let mut out_sh = Vec::with_capacity(x_sh.len());
                        for (axis, &dim) in x_sh.iter().enumerate() {
                            if axes.is_empty() || norm_axes.contains(&axis) {
                                if attrs.keepdims {
                                    out_sh.push(1);
                                }
                            } else {
                                out_sh.push(dim);
                            }
                        }
                        let y_ptr = alloc_out(out_id, &out_sh, dtype, run_bufs)?;
                        match node.op {
                            Op::ReduceMean(_) => self.spatial.reduce_mean_f32(x_ptr, y_ptr, &x_sh, &axes)?,
                            Op::ReduceSum(_) => self.spatial.reduce_sum_f32(x_ptr, y_ptr, &x_sh, &axes)?,
                            Op::ReduceMax(_) => self.spatial.reduce_max_f32(x_ptr, y_ptr, &x_sh, &axes)?,
                            _ => unreachable!(),
                        }
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, dtype));
                    }
                    Op::Resize(attrs) => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let (n, c, in_h, in_w) = (x_sh[0], x_sh[1], x_sh[2], x_sh[3]);
                        let out_id = req_out_id(0)?;
                        let (out_h, out_w) = if let Some(v) = node
                            .inputs
                            .get(3)
                            .and_then(|o| *o)
                            .and_then(|sid| read_i64_vec(sid).ok())
                            .filter(|v| v.len() >= 4)
                        {
                            (v[2] as usize, v[3] as usize)
                        } else if let Some(v) = node
                            .inputs
                            .get(2)
                            .and_then(|o| *o)
                            .and_then(|sc_id| cpu_tensors.get(&sc_id))
                            .map(|ct| ct.to_f32_vec())
                            .filter(|v| v.len() >= 4)
                        {
                            (
                                (in_h as f32 * v[2]).floor() as usize,
                                (in_w as f32 * v[3]).floor() as usize,
                            )
                        } else if let Some(sh) = graph.tensors[out_id.0].static_shape() {
                            (sh[2], sh[3])
                        } else {
                            return Err(SessionError::Runtime(format!(
                                "Resize '{}' could not determine output H/W",
                                node.name
                            )));
                        };
                        let mode = ResizeKernelMode::from_attrs(
                            attrs.mode,
                            attrs.coord_mode,
                            attrs.nearest_mode,
                        ).map_err(|err| SessionError::Runtime(format!(
                            "Resize '{}': unsupported attributes {attrs:?}: {err:?}", node.name
                        )))?;
                        let out_sh = vec![n, c, out_h, out_w];
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        self.spatial.resize_nchw_f32(
                            x_ptr, y_ptr, n, c, in_h, in_w, out_h, out_w, mode,
                        )?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }
                    Op::Pad { .. } => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let pads = node
                            .inputs
                            .get(1)
                            .and_then(|o| *o)
                            .and_then(|id| cpu_tensors.get(&id))
                            .map(|c| c.to_i64_vec())
                            .ok_or_else(|| {
                                SessionError::Runtime(format!("Pad '{}' missing CPU pads", node.name))
                            })?;
                        let const_val = node
                            .inputs
                            .get(2)
                            .and_then(|o| *o)
                            .and_then(|id| cpu_tensors.get(&id))
                            .and_then(|c| c.to_f32_vec().into_iter().next())
                            .unwrap_or(0.0);
                        let in_shape = [x_sh[0], x_sh[1], x_sh[2], x_sh[3]];
                        let pad_begin = [
                            pads[0] as usize,
                            pads[1] as usize,
                            pads[2] as usize,
                            pads[3] as usize,
                        ];
                        let pad_end = [
                            pads[4] as usize,
                            pads[5] as usize,
                            pads[6] as usize,
                            pads[7] as usize,
                        ];
                        let out_sh = vec![
                            in_shape[0] + pad_begin[0] + pad_end[0],
                            in_shape[1] + pad_begin[1] + pad_end[1],
                            in_shape[2] + pad_begin[2] + pad_end[2],
                            in_shape[3] + pad_begin[3] + pad_end[3],
                        ];
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        self.spatial.pad_constant_4d_f32(
                            x_ptr, y_ptr, in_shape, pad_begin, pad_end, const_val,
                        )?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }

                    // --- Tensor Rearrangement & Indexing ---
                    Op::Transpose { perm } => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let out_sh: Vec<usize> = perm.iter().map(|&p| x_sh[p]).collect();
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        self.spatial.transpose_nd_f32(x_ptr, y_ptr, &x_sh, perm)?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }
                    Op::Concat { axis } => {
                        let mut segs = Vec::with_capacity(node.inputs.len());
                        let mut base_sh: Option<Vec<usize>> = None;
                        let mut ax = 0usize;
                        let mut total_axis = 0usize;
                        for i in 0..node.inputs.len() {
                            let (p, sh, _) = req_gpu(i)?;
                            if base_sh.is_none() {
                                ax = norm_axis(*axis, sh.len())?;
                                base_sh = Some(sh.clone());
                            }
                            total_axis += sh[ax];
                            segs.push((p, sh[ax]));
                        }
                        let mut out_sh = base_sh.unwrap();
                        let outer: usize = out_sh[..ax].iter().product();
                        let inner: usize = out_sh[ax + 1..].iter().product();
                        out_sh[ax] = total_axis;
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        self.spatial.concat_axis_f32(&segs, y_ptr, outer, inner)?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }
                    Op::Split { axis, num_outputs } => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let ax = norm_axis(*axis, x_sh.len())?;
                        let n_out = num_outputs.unwrap_or(node.outputs.len());
                        let splits: Vec<usize> = if let Some(Some(sid)) = node.inputs.get(1) {
                            cpu_tensors
                                .get(sid)
                                .map(|c| c.to_i64_vec().into_iter().map(|v| v as usize).collect())
                                .unwrap_or_else(|| vec![x_sh[ax] / n_out; n_out])
                        } else {
                            let base = x_sh[ax] / n_out;
                            let rem = x_sh[ax] % n_out;
                            (0..n_out).map(|i| base + usize::from(i < rem)).collect()
                        };
                        let mut offset = 0i64;
                        for (i, &len) in splits.iter().enumerate() {
                            if let Some(Some(out_id)) = node.outputs.get(i) {
                                let mut out_sh = x_sh.clone();
                                out_sh[ax] = len;
                                let mut starts = vec![0i64; x_sh.len()];
                                starts[ax] = offset;
                                let steps = vec![1i64; x_sh.len()];
                                let y_ptr =
                                    alloc_out(*out_id, &out_sh, DataType::Float, run_bufs)?;
                                self.spatial.slice_nd_f32(
                                    x_ptr, y_ptr, &x_sh, &out_sh, &starts, &steps,
                                )?;
                                gpu_tensors.insert(*out_id, (y_ptr, out_sh, DataType::Float));
                            }
                            offset += len as i64;
                        }
                    }
                    Op::Slice => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let rank = x_sh.len();
                        let starts_in = cpu_tensors
                            .get(&node.inputs[1].unwrap())
                            .map(|c| c.to_i64_vec())
                            .ok_or_else(|| {
                                SessionError::Runtime(format!("Slice '{}' missing starts", node.name))
                            })?;
                        let ends_in = cpu_tensors
                            .get(&node.inputs[2].unwrap())
                            .map(|c| c.to_i64_vec())
                            .ok_or_else(|| {
                                SessionError::Runtime(format!("Slice '{}' missing ends", node.name))
                            })?;
                        let axes_in: Vec<i64> = node
                            .inputs
                            .get(3)
                            .and_then(|o| *o)
                            .and_then(|id| cpu_tensors.get(&id))
                            .map(|c| c.to_i64_vec())
                            .unwrap_or_else(|| (0..starts_in.len() as i64).collect());
                        let steps_in: Vec<i64> = node
                            .inputs
                            .get(4)
                            .and_then(|o| *o)
                            .and_then(|id| cpu_tensors.get(&id))
                            .map(|c| c.to_i64_vec())
                            .unwrap_or_else(|| vec![1; starts_in.len()]);

                        let mut full_starts = vec![0i64; rank];
                        let mut full_steps = vec![1i64; rank];
                        let mut out_sh = x_sh.clone();
                        for idx in 0..axes_in.len() {
                            let ax = norm_axis(axes_in[idx], rank)?;
                            let dim = x_sh[ax] as i64;
                            let step = steps_in[idx];
                            let mut s = starts_in[idx];
                            let mut e = ends_in[idx];
                            if s < 0 {
                                s += dim;
                            }
                            if e < 0 {
                                e += dim;
                            }
                            if step > 0 {
                                s = s.clamp(0, dim);
                                e = e.clamp(0, dim);
                                out_sh[ax] = if e > s {
                                    ((e - s + step - 1) / step) as usize
                                } else {
                                    0
                                };
                            } else {
                                s = s.clamp(0, dim - 1);
                                e = e.clamp(-1, dim - 1);
                                out_sh[ax] = if s > e {
                                    ((s - e - step - 1) / (-step)) as usize
                                } else {
                                    0
                                };
                            }
                            full_starts[ax] = s;
                            full_steps[ax] = step;
                        }
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        self.spatial.slice_nd_f32(
                            x_ptr, y_ptr, &x_sh, &out_sh, &full_starts, &full_steps,
                        )?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }
                    Op::Gather { axis } => {
                        let (data_ptr, data_sh, _) = req_gpu(0)?;
                        let (idx_ptr, idx_sh, _) = req_gpu(1)?;
                        let ax = norm_axis(*axis, data_sh.len())?;
                        let outer: usize = data_sh[..ax].iter().product();
                        let axis_dim = data_sh[ax];
                        let inner: usize = data_sh[ax + 1..].iter().product();
                        let num_indices = numel(&idx_sh);
                        let mut out_sh = Vec::new();
                        out_sh.extend_from_slice(&data_sh[..ax]);
                        out_sh.extend_from_slice(&idx_sh);
                        out_sh.extend_from_slice(&data_sh[ax + 1..]);
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        self.spatial.gather_axis_f32(
                            data_ptr, idx_ptr, y_ptr, outer, axis_dim, num_indices, inner,
                        )?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }
                    Op::ScatterND => {
                        let (data_ptr, data_sh, _) = req_gpu(0)?;
                        let (idx_ptr, idx_sh, _) = req_gpu(1)?;
                        let (upd_ptr, _upd_sh, _) = req_gpu(2)?;
                        let index_depth_k = *idx_sh.last().unwrap_or(&1);
                        let num_updates = if idx_sh.len() > 1 {
                            numel(&idx_sh[..idx_sh.len() - 1])
                        } else {
                            1
                        };
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &data_sh, DataType::Float, run_bufs)?;
                        self.spatial.scatter_nd_f32(
                            data_ptr,
                            idx_ptr,
                            upd_ptr,
                            y_ptr,
                            &data_sh,
                            num_updates,
                            index_depth_k,
                        )?;
                        gpu_tensors.insert(out_id, (y_ptr, data_sh, DataType::Float));
                    }
                    Op::Expand => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let tgt_tid = node.inputs.get(1).and_then(|o| *o).ok_or_else(|| {
                            SessionError::Runtime(format!("Expand '{}' missing target input", node.name))
                        })?;
                        let target: Vec<usize> = read_i64_vec(tgt_tid)?
                            .into_iter()
                            .map(|v| v as usize)
                            .collect();
                        let rank = x_sh.len().max(target.len());
                        let mut out_sh = Vec::with_capacity(rank);
                        for i in 0..rank {
                            let dx = if i + x_sh.len() >= rank {
                                x_sh[i + x_sh.len() - rank]
                            } else {
                                1
                            };
                            let dt = if i + target.len() >= rank {
                                target[i + target.len() - rank]
                            } else {
                                1
                            };
                            out_sh.push(dx.max(dt));
                        }
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        self.spatial.expand_nd_f32(x_ptr, y_ptr, &x_sh, &out_sh)?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }
                    Op::Tile => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let rep_tid = node.inputs.get(1).and_then(|o| *o).ok_or_else(|| {
                            SessionError::Runtime(format!("Tile '{}' missing repeats input", node.name))
                        })?;
                        let repeats: Vec<usize> = read_i64_vec(rep_tid)?
                            .into_iter()
                            .map(|v| v as usize)
                            .collect();
                        let out_sh: Vec<usize> =
                            x_sh.iter().zip(&repeats).map(|(&d, &r)| d * r).collect();
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, run_bufs)?;
                        self.spatial.tile_nd_f32(x_ptr, y_ptr, &x_sh, &repeats)?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }

                    // --- Quantized UInt8 ops (eSamEncoderPreUInt8) ---
                    Op::DynamicQuantizeLinear => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let y_id = req_out_id(0)?;
                        let sc_id = req_out_id(1)?;
                        let zp_id = req_out_id(2)?;
                        let y_ptr = alloc_out(y_id, &x_sh, DataType::Uint8, run_bufs)?;
                        let sc_ptr = alloc_out(sc_id, &[], DataType::Float, run_bufs)?;
                        let zp_ptr = alloc_out(zp_id, &[], DataType::Uint8, run_bufs)?;
                        self.quant.dynamic_quantize_linear(
                            x_ptr,
                            numel(&x_sh),
                            y_ptr,
                            sc_ptr,
                            zp_ptr,
                        )?;
                        gpu_tensors.insert(y_id, (y_ptr, x_sh, DataType::Uint8));
                        gpu_tensors.insert(sc_id, (sc_ptr, Vec::new(), DataType::Float));
                        gpu_tensors.insert(zp_id, (zp_ptr, Vec::new(), DataType::Uint8));
                    }
                    Op::MatMulInteger => {
                        let (a_ptr, a_sh, _) = req_gpu(0)?;
                        let (b_ptr, b_sh, _) = req_gpu(1)?;
                        let a_zp = match opt_gpu(2) {
                            None => ZeroPointArg::None,
                            Some((p, sh, _)) => {
                                if numel(&sh) <= 1 {
                                    ZeroPointArg::Scalar(p)
                                } else if sh.len() == 1 {
                                    ZeroPointArg::PerAxis(p)
                                } else {
                                    ZeroPointArg::BatchedPerAxis(p)
                                }
                            }
                        };
                        let b_zp = match opt_gpu(3) {
                            None => ZeroPointArg::None,
                            Some((p, sh, _)) => {
                                if numel(&sh) <= 1 {
                                    ZeroPointArg::Scalar(p)
                                } else if sh.len() == 1 {
                                    ZeroPointArg::PerAxis(p)
                                } else {
                                    ZeroPointArg::BatchedPerAxis(p)
                                }
                            }
                        };
                        let m = a_sh[a_sh.len() - 2];
                        let k = a_sh[a_sh.len() - 1];
                        let n = b_sh[b_sh.len() - 1];
                        let a_batch = &a_sh[..a_sh.len() - 2];
                        let b_batch = &b_sh[..b_sh.len() - 2];
                        let br = a_batch.len().max(b_batch.len());
                        let mut out_sh = Vec::with_capacity(br + 2);
                        for i in 0..br {
                            let da = if i + a_batch.len() >= br {
                                a_batch[i + a_batch.len() - br]
                            } else {
                                1
                            };
                            let db = if i + b_batch.len() >= br {
                                b_batch[i + b_batch.len() - br]
                            } else {
                                1
                            };
                            out_sh.push(da.max(db));
                        }
                        out_sh.push(m);
                        out_sh.push(n);
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Int32, run_bufs)?;
                        let params = MatMulIntegerParams::new_batched(
                            a_ptr, b_ptr, y_ptr, a_batch, b_batch, m, n, k, a_zp, b_zp,
                        )?;
                        self.quant.matmul_integer(&params)?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Int32));
                    }
                    Op::ConvInteger(attrs) => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let (w_ptr, w_sh, _) = req_gpu(1)?;
                        let x_zp = match opt_gpu(2) {
                            None => ZeroPointArg::None,
                            Some((p, sh, _)) => {
                                if numel(&sh) <= 1 {
                                    ZeroPointArg::Scalar(p)
                                } else {
                                    ZeroPointArg::PerAxis(p)
                                }
                            }
                        };
                        let w_zp = match opt_gpu(3) {
                            None => ZeroPointArg::None,
                            Some((p, sh, _)) => {
                                if numel(&sh) <= 1 {
                                    ZeroPointArg::Scalar(p)
                                } else {
                                    ZeroPointArg::PerAxis(p)
                                }
                            }
                        };
                        let p = Conv2dParams::from_attrs(
                            &x_sh,
                            &w_sh,
                            attrs,
                            FusedActivation::None,
                        )?;
                        let out_sh = vec![p.batch, p.c_out, p.out_h, p.out_w];
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Int32, run_bufs)?;
                        let cparams = ConvIntegerParams {
                            x_ptr,
                            w_ptr,
                            y_ptr,
                            x_zero_point: x_zp,
                            w_zero_point: w_zp,
                            n: p.batch,
                            c_in: p.c_in,
                            h_in: p.h_in,
                            w_in: p.w_in,
                            c_out: p.c_out,
                            k_h: p.kh,
                            k_w: p.kw,
                            h_out: p.out_h,
                            w_out: p.out_w,
                            stride_h: p.sh,
                            stride_w: p.sw,
                            pad_top: p.pad_t,
                            pad_left: p.pad_l,
                            dilation_h: p.dh,
                            dilation_w: p.dw,
                            group: p.group,
                        };
                        self.quant.conv_integer(&cparams)?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Int32));
                    }

                    // --- Control Flow (`If` subgraphs in eSamDecoder) ---
                    Op::If(if_attrs) => {
                        let cond_tid = node.inputs.first().and_then(|o| *o).ok_or_else(|| {
                            SessionError::Runtime(format!("If '{}' missing condition input", node.name))
                        })?;
                        let cond_val: bool = if let Some(ct) = cpu_tensors.get(&cond_tid) {
                            match &ct.data {
                                CpuData::Bool(v) => v.first().copied().unwrap_or(false),
                                CpuData::I64(v) => v.first().copied().unwrap_or(0) != 0,
                                CpuData::F32(v) => v.first().copied().unwrap_or(0.0) != 0.0,
                            }
                        } else if let Some(&(ptr, _, _)) = gpu_tensors.get(&cond_tid) {
                            let tmp = gpu.alloc_buffer(4)?;
                            self.spatial.copy_bytes_u8(ptr, tmp.handle(), 1)?;
                            let mut byte = [0u8; 4];
                            gpu.download_buffer(&tmp, &mut byte)?;
                            byte[0] != 0
                        } else {
                            return Err(SessionError::Runtime(format!(
                                "If '{}' condition tensor not found",
                                node.name
                            )));
                        };

                        let branch = if cond_val {
                            &if_attrs.then_branch
                        } else {
                            &if_attrs.else_branch
                        };

                        let mut sub_gpu: HashMap<TensorId, (u64, Vec<usize>, DataType)> =
                            HashMap::new();
                        let mut sub_cpu: HashMap<TensorId, CpuTensor> = HashMap::new();

                        // Bind subgraph weights.
                        for (&wtid, proto) in &branch.weights {
                            let shape: Vec<usize> =
                                proto.dims.iter().map(|&d| d.max(0) as usize).collect();
                            if proto.elem_count() <= DEFAULT_MAX_CPU_TENSOR_ELEMS {
                                if let Ok(ct) = CpuTensor::from_tensor_proto(proto) {
                                    sub_cpu.insert(wtid, ct);
                                }
                            }
                            let (bytes, stored_dt): (Vec<u8>, DataType) = match proto.data_type {
                                DataType::Float => {
                                    if proto.raw_data.len() == proto.elem_count() * 4 {
                                        (proto.raw_data.clone(), DataType::Float)
                                    } else {
                                        let v = proto.to_f32_vec().map_err(|e| {
                                            SessionError::Runtime(format!(
                                                "If subgraph weight decode error: {e}"
                                            ))
                                        })?;
                                        (f32_as_bytes(&v).to_vec(), DataType::Float)
                                    }
                                }
                                DataType::Float16 => {
                                    let v = proto.to_f32_vec().map_err(|e| {
                                        SessionError::Runtime(format!(
                                            "If subgraph fp16 weight decode error: {e}"
                                        ))
                                    })?;
                                    (f32_as_bytes(&v).to_vec(), DataType::Float)
                                }
                                other => (proto.raw_data.clone(), other),
                            };
                            if !bytes.is_empty() {
                                let buf = gpu.alloc_buffer(bytes.len())?;
                                gpu.upload_buffer(&buf, &bytes)?;
                                let ptr = buf.handle();
                                run_bufs.push(buf);
                                sub_gpu.insert(wtid, (ptr, shape, stored_dt));
                            } else {
                                sub_gpu.insert(wtid, (0u64, shape, stored_dt));
                            }
                        }

                        // Bind outer-scope captured tensors by name.
                        for info in &branch.tensors {
                            if let Some(&outer_tid) = graph.name_to_id.get(&info.name) {
                                if let Some(g) = gpu_tensors.get(&outer_tid) {
                                    sub_gpu.insert(info.id, g.clone());
                                }
                                if let Some(c) = cpu_tensors.get(&outer_tid) {
                                    sub_cpu.insert(info.id, c.clone());
                                }
                            }
                        }

                        let empty_out_ids = HashMap::new();
                        self.exec_graph(
                            branch,
                            false,
                            recording,
                            &empty_out_ids,
                            &mut sub_gpu,
                            &mut sub_cpu,
                            run_bufs,
                        )?;

                        // Propagate branch outputs back to outer node outputs.
                        for (idx, &branch_out_tid) in branch.outputs.iter().enumerate() {
                            if let Some(Some(outer_out_tid)) = node.outputs.get(idx) {
                                if let Some(g) = sub_gpu.get(&branch_out_tid) {
                                    let optr = alloc_out(*outer_out_tid, &g.1, g.2, run_bufs)?;
                                    let total = numel(&g.1);
                                    if optr != g.0 && total > 0 {
                                        if g.2 == DataType::Float {
                                            self.spatial.copy_f32(g.0, optr, total)?;
                                        } else {
                                            let eb = g.2.byte_size().unwrap_or(4);
                                            self.spatial.copy_bytes_u8(g.0, optr, total * eb)?;
                                        }
                                        gpu_tensors.insert(*outer_out_tid, (optr, g.1.clone(), g.2));
                                    } else {
                                        gpu_tensors.insert(*outer_out_tid, g.clone());
                                    }
                                }
                                if let Some(c) = sub_cpu.get(&branch_out_tid) {
                                    cpu_tensors.insert(*outer_out_tid, c.clone());
                                }
                            }
                        }
                    }

                    other => {
                        return Err(SessionError::Runtime(format!(
                            "op '{}' at node '{}' not yet wired in NativeCudaSession::run_f32",
                            other.name(),
                            node.name
                        )));
                    }
                }
            }
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::path::PathBuf;
        use std::time::Instant;

        /// Smoke-run real shipped ONNX models end-to-end on RTX 4070 Ti using
        /// `NativeCudaSession` (zero ORT DLLs!), reporting load time, VRAM arena
        /// peak, warm inference latency, and finite output verification.
        #[test]
        #[ignore = "requires shipped ONNX models on disk and NVIDIA CUDA GPU"]
        fn run_real_shipped_models_end_to_end_native_cuda() {
            let root = PathBuf::from(std::env::var_os("AURORA_MODELS_DIR").expect("set AURORA_MODELS_DIR for the full 19-model smoke suite"));
            assert!(root.is_dir(), "model directory missing: {}", root.display());
            let cases: &[(&str, &str, &[usize])] = &[
                (r"landmark\assets\1063025060.onnx", "3070435246", &[1, 3, 64, 64]),
                (r"landmark\assets\323773030.onnx", "t1511", &[1, 3, 160, 160]),
                (r"AIModels\Common\BlemishRemoval\blemishRemoval.onnx", "input.1", &[1, 3, 278, 278]),
                (r"AIModels\Common\FaceMesh\faceMesh.onnx", "input_12", &[1, 3, 256, 256]),
                (r"AIModels\Common\ClosedEyesDetection\EyeDetector.onnx", "input", &[1, 3, 64, 64]),
                (r"AIModels\Common\ClosedEyesDetection\DenseNet121_FP32.onnx", "input", &[1, 3, 64, 64]),
                (r"AIModels\Common\BeautyMarks\beautyMarks.onnx", "images", &[1, 3, 416, 416]),
                (r"AIModels\Common\DustRemoval\dust.onnx", "images", &[1, 3, 416, 416]),
                (r"AIModels\Common\Rotation\autoRotation.onnx", "conv2d_input", &[1, 384, 384, 1]),
                (r"AIModels\Common\Similarity\similarity.onnx", "input_1", &[1, 224, 224, 3]),
                (r"AIModels\Common\PeopleMasking\skinMasking.onnx", "input_29", &[1, 512, 512, 3]),
                (r"AIModels\Common\SubjectMasking\subjectMasking.onnx", "x", &[1, 3, 320, 320]),
                (r"AIModels\Common\MatchLook\MatchLookBrightness.onnx", "input.1", &[1, 3, 384, 384]),
                (r"AIModels\Common\MatchLook\MatchLookBw.onnx", "input.1", &[1, 3, 224, 224]),
                (r"AIModels\Common\MatchLook\MatchLookColor.onnx", "input.1", &[1, 3, 384, 384]),
                (r"AIModels\Common\MatchLook\MatchLookExp.onnx", "input.1", &[1, 3, 384, 384]),
                (r"AIModels\Common\FaceDetection\faceDetection.onnx", "input", &[1, 3, 320, 320]),
                (r"AIModels\Common\SamEncoder\eSamEncoderPreUInt8.onnx", "input_image", &[1024, 1024, 3]),
            ];

            let mut passed = 0usize;
            for &(rel, input_name, shape) in cases {
                let path = root.join(rel);
                if !path.is_file() {
                    eprintln!("[MISSING] {rel}");
                    continue;
                }
                let bytes = std::fs::read(&path).expect("read onnx");
                let mut overrides = HashMap::new();
                overrides.insert(input_name.to_string(), shape.to_vec());

                let t_load = Instant::now();
                let session = match NativeCudaSession::from_bytes(&bytes, &overrides) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("[SKIP/LOAD-ERR] {rel}: {e}");
                        continue;
                    }
                };
                let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

                let n = numel(shape);
                let input_data: Vec<f32> =
                    (0..n).map(|i| ((i % 251) as f32) / 251.0 - 0.5).collect();
                let mut inputs = HashMap::new();
                inputs.insert(input_name.to_string(), (input_data.as_slice(), shape.to_vec()));

                // Warmup + 3 timed runs.
                match session.run_f32(&inputs) {
                    Ok(first) => {
                        let mut best_ms = f64::INFINITY;
                        for _ in 0..3 {
                            let t = Instant::now();
                            let out = session.run_f32(&inputs).expect("timed run");
                            best_ms = best_ms.min(t.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(out.len(), first.len());
                        }
                        let mut summary = String::new();
                        for (k, (v, sh)) in &first {
                            assert!(
                                v.iter().all(|x| x.is_finite()),
                                "{rel} output '{k}' contains non-finite values"
                            );
                            let mean = v.iter().sum::<f32>() / (v.len().max(1) as f32);
                            summary.push_str(&format!("{k}{sh:?}(mean={mean:.4}) "));
                        }
                        eprintln!(
                            "[PASS] {:<32} load={:6.1}ms arena={:6.2}MiB infer_best={:6.2}ms | {}",
                            rel,
                            load_ms,
                            session.arena_peak_bytes() as f64 / (1024.0 * 1024.0),
                            best_ms,
                            summary
                        );
                        passed += 1;
                    }
                    Err(e) => {
                        eprintln!("[ERR ] {:<32} {e}", rel);
                    }
                }
            }
            // Also test multi-input eSamDecoder.onnx (contains Op::If subgraphs).
            let dec_path = root.join(r"AIModels\Common\SamDecoder\eSamDecoder.onnx");
            if dec_path.is_file() {
                let bytes = std::fs::read(&dec_path).expect("read eSamDecoder");
                let mut overrides = HashMap::new();
                overrides.insert("image_embeddings".to_string(), vec![1, 256, 64, 64]);
                overrides.insert("point_coords".to_string(), vec![1, 1, 80, 2]);
                overrides.insert("point_labels".to_string(), vec![1, 1, 80]);
                overrides.insert("orig_im_size".to_string(), vec![2]);
                let t_load = Instant::now();
                match NativeCudaSession::from_bytes(&bytes, &overrides) {
                    Ok(session) => {
                        let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
                        let emb = vec![0.1f32; 256 * 64 * 64];
                        let mut pts = vec![0.0f32; 80 * 2];
                        pts[0] = 512.0;
                        pts[1] = 512.0;
                        let mut lbl = vec![-1.0f32; 80];
                        lbl[0] = 1.0;
                        let orig = vec![1024.0f32, 1024.0f32];
                        let mut inputs = HashMap::new();
                        inputs.insert("image_embeddings".to_string(), (emb.as_slice(), vec![1, 256, 64, 64]));
                        inputs.insert("point_coords".to_string(), (pts.as_slice(), vec![1, 1, 80, 2]));
                        inputs.insert("point_labels".to_string(), (lbl.as_slice(), vec![1, 1, 80]));
                        inputs.insert("orig_im_size".to_string(), (orig.as_slice(), vec![2]));
                        match session.run_f32(&inputs) {
                            Ok(first) => {
                                let t = Instant::now();
                                let _ = session.run_f32(&inputs).expect("eSamDecoder timed run");
                                let ms = t.elapsed().as_secs_f64() * 1000.0;
                                let mut summary = String::new();
                                for (k, (v, sh)) in &first {
                                    assert!(v.iter().all(|x| x.is_finite()), "eSamDecoder '{k}' non-finite");
                                    let mean = v.iter().sum::<f32>() / (v.len().max(1) as f32);
                                    summary.push_str(&format!("{k}{sh:?}(mean={mean:.4}) "));
                                }
                                eprintln!(
                                    "[PASS] {:<32} load={:6.1}ms arena={:6.2}MiB infer={:6.2}ms | {}",
                                    r"AIModels\Common\SamDecoder\eSamDecoder.onnx",
                                    load_ms,
                                    session.arena_peak_bytes() as f64 / (1024.0 * 1024.0),
                                    ms,
                                    summary
                                );
                                passed += 1;
                            }
                            Err(e) => eprintln!("[ERR ] eSamDecoder.onnx: {e}"),
                        }
                    }
                    Err(e) => eprintln!("[LOAD-ERR] eSamDecoder.onnx: {e}"),
                }
            }
            if !dec_path.is_file() { eprintln!("[MISSING] eSamDecoder.onnx"); }
            let expected = cases.len() + 1;
            eprintln!("=== SMOKE SUMMARY: {passed}/{expected} ===");
            assert_eq!(expected, 19, "shipped manifest drift");
            assert_eq!(passed, expected, "incomplete native end-to-end smoke suite");
        }

        /// A/B proof that graph replay reproduces the plain path, with the
        /// per-model capture verdict.
        ///
        /// The comparison is two-sided on purpose: the first call runs the plain
        /// path and captures, the second call is answered by the replay, and both
        /// are checked against a session that ran with the feature switched off.
        ///
        /// Mutates `AURORA_NATIVE_CUDA_GRAPH` for the whole process, so run it
        /// alone:
        /// `cargo test -p aurora-portrait --lib native_infer::session::imp::tests::graph_replay_matches_the_plain_path_on_shipped_models -- --ignored --test-threads=1 --nocapture`
        #[test]
        #[ignore = "requires shipped ONNX models on disk and NVIDIA CUDA GPU"]
        fn graph_replay_matches_the_plain_path_on_shipped_models() {
            let root = match std::env::var_os("AURORA_MODELS_DIR") {
                Some(v) => PathBuf::from(v),
                None => {
                    eprintln!("set AURORA_MODELS_DIR to the models directory");
                    return;
                }
            };
            let cases: &[(&str, &str, &[usize])] = &[
                (r"AIModels\Common\FaceMesh\faceMesh.onnx", "input_12", &[1, 3, 256, 256]),
                (r"AIModels\Common\Rotation\autoRotation.onnx", "conv2d_input", &[1, 384, 384, 1]),
                (r"landmark\assets\323773030.onnx", "t1511", &[1, 3, 160, 160]),
                (r"AIModels\Common\BlemishRemoval\blemishRemoval.onnx", "input.1", &[1, 3, 278, 278]),
                (r"AIModels\Common\PeopleMasking\skinMasking.onnx", "input_29", &[1, 512, 512, 3]),
                (r"AIModels\Common\SubjectMasking\subjectMasking.onnx", "x", &[1, 3, 320, 320]),
                (r"AIModels\Common\MatchLook\MatchLookColor.onnx", "input.1", &[1, 3, 384, 384]),
                (r"AIModels\Common\FaceDetection\faceDetection.onnx", "input", &[1, 3, 320, 320]),
                (r"AIModels\Common\SamEncoder\eSamEncoderPreUInt8.onnx", "input_image", &[1024, 1024, 3]),
            ];

            let mut present = 0usize;
            let mut captured = 0usize;
            let mut replayed = 0usize;
            for &(rel, input_name, shape) in cases {
                let path = root.join(rel);
                if !path.is_file() {
                    eprintln!("[MISSING] {rel}");
                    continue;
                }
                present += 1;
                let bytes = std::fs::read(&path).expect("read onnx");
                let mut overrides = HashMap::new();
                overrides.insert(input_name.to_string(), shape.to_vec());
                let count = numel(shape);
                let values: Vec<f32> = (0..count)
                    .map(|i| ((i % 251) as f32) / 251.0 - 0.5)
                    .collect();
                let mut inputs = HashMap::new();
                inputs.insert(input_name.to_string(), (values.as_slice(), shape.to_vec()));

                // Reference: the same model with the feature switched off.
                // SAFETY: the repository runs its suite with `--test-threads=1`,
                // which this test documents in its own invocation line.
                unsafe { std::env::set_var(cuda_graph::SWITCH_ENV, "0") };
                let plain = NativeCudaSession::from_bytes(&bytes, &overrides).expect("plain session");
                assert_eq!(
                    plain.graph_status(),
                    cuda_graph::GraphStatus::default(),
                    "{rel}: the switch did not turn capture off"
                );
                let expected = plain.run_f32(&inputs).expect("plain run");
                let changed = values.iter().map(|&v| 0.23 - v * 0.67).collect::<Vec<_>>();
                let changed_inputs = HashMap::from([(input_name.to_string(), (changed.as_slice(), shape.to_vec()))]);
                let expected_changed = plain.run_f32(&changed_inputs).expect("second plain input");
                drop(plain);

                unsafe { std::env::set_var(cuda_graph::SWITCH_ENV, "1") };
                let session =
                    NativeCudaSession::from_bytes(&bytes, &overrides).expect("graph session");
                let capture_started = Instant::now();
                let first = session.run_f32(&inputs).expect("first run");
                let first_ms = capture_started.elapsed().as_secs_f64() * 1000.0;
                cuda_graph::compare_outputs(&expected, &first, cuda_graph::VERIFY_TOLERANCE)
                    .expect("the plain path changed");
                let after_capture = session.graph_status();

                let replay_started = Instant::now();
                let replay = session.run_f32(&inputs).expect("replay run");
                let replay_ms = replay_started.elapsed().as_secs_f64() * 1000.0;
                let max_abs = cuda_graph::compare_outputs(
                    &expected,
                    &replay,
                    cuda_graph::VERIFY_TOLERANCE,
                )
                .unwrap_or_else(|fallback| {
                    panic!("{rel}: replay diverged from the plain path: {fallback}")
                });
                let after_replay = session.graph_status();
                let changed_replay = session.run_f32(&changed_inputs).expect("replay with different pixels");
                cuda_graph::compare_outputs(&expected_changed, &changed_replay, cuda_graph::VERIFY_TOLERANCE)
                    .unwrap_or_else(|error| panic!("{rel}: stale pixels on second input: {error}"));
                if after_capture.active {
                    captured += 1;
                }
                if after_replay.replays > 0 {
                    replayed += 1;
                }
                assert!(
                    !after_capture.active || after_replay.replays == 1,
                    "{rel}: an active graph did not serve the second call"
                );
                eprintln!(
                    "[GRAPH] {:<42} nodes={:4} first={:7.2}ms replay={:7.2}ms max_abs={max_abs:.3e} | {}",
                    rel, after_capture.nodes, first_ms, replay_ms, after_replay.summary()
                );
            }
            unsafe { std::env::remove_var(cuda_graph::SWITCH_ENV) };
            assert!(present > 0 && replayed > 0, "no real model was replayed");
            eprintln!(
                "=== GRAPH A/B: {captured} captured, {replayed} replayed of {} models ===",
                cases.len()
            );
        }

        #[test]
        #[ignore = "requires CUDA GPU and 2102828093.onnx"]
        fn test_2102828093_native_cuda() {
            let path = std::env::var("AURORA_FACE_PARSING_MODEL")
                .expect("set AURORA_FACE_PARSING_MODEL");
            let bytes = std::fs::read(&path).expect("read 2102828093.onnx");
            let model = crate::native_infer::onnx_proto::decode_model(&bytes).expect("decode");
            let mut ops: Vec<String> = model.graph.nodes.iter().map(|n| n.op_type.clone()).collect();
            ops.sort();
            ops.dedup();
            eprintln!("2102828093.onnx ops ({}): {:?}", ops.len(), ops);
            let mut overrides = HashMap::new();
            let in_name = model.graph.inputs[0].name.clone();
            overrides.insert(in_name.clone(), vec![1, 3, 384, 384]);
            let session = NativeCudaSession::from_bytes(&bytes, &overrides)
                .expect("NativeCudaSession::from_bytes(2102828093.onnx)");
            let input = vec![0.25f32; 3 * 384 * 384];
            let mut inputs = HashMap::new();
            inputs.insert(in_name, (input.as_slice(), vec![1, 3, 384, 384]));
            let t0 = std::time::Instant::now();
            let out = session.run_f32(&inputs).expect("run_f32(2102828093.onnx)");
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            for (k, (v, sh)) in &out {
                assert!(v.iter().all(|x| x.is_finite()), "non-finite in {k}");
                let mean = v.iter().sum::<f32>() / (v.len().max(1) as f32);
                eprintln!("[PASS] 2102828093.onnx {k}{sh:?} mean={mean:.4} infer={ms:.2}ms");
            }
        }

        #[test]
        #[ignore = "requires CUDA GPU and faceplump ONNX models"]
        fn test_faceplump_3_models_native_cuda() {
            let dir = std::env::var("AURORA_FACEPLUMP_DIR")
                .expect("set AURORA_FACEPLUMP_DIR");
            use crate::native_infer::onnx_proto::{DataType, DimSpec, decode_model};

            for (name, input_shape, output_shape) in [
                ("597383622.onnx", [1, 4, 768, 768], [1, 4, 768, 768]),
                ("3861222348.onnx", [1, 3, 768, 768], [1, 8, 512, 512]),
                ("1139333499.onnx", [1, 3, 256, 512], [1, 4, 256, 512]),
            ] {
                let path = std::path::Path::new(&dir).join(name);
                let bytes = std::fs::read(&path).expect("read FacePlump ONNX");
                let model = decode_model(&bytes).expect("decode FacePlump ONNX");
                let [input_info] = model.graph.inputs.as_slice() else {
                    panic!("{name}: expected one input");
                };
                let [output_info] = model.graph.outputs.as_slice() else {
                    panic!("{name}: expected one output");
                };
                assert_eq!(input_info.data_type, DataType::Float, "{name}");
                assert_eq!(output_info.data_type, DataType::Float, "{name}");
                assert_eq!(input_info.dims, input_shape.map(|dim| DimSpec::Value(dim as i64)), "{name}");
                let session = NativeCudaSession::from_bytes(&bytes, &HashMap::new())
                    .unwrap_or_else(|err| panic!("{name}: open CUDA session: {err}"));
                let input: Vec<f32> = (0..input_shape.iter().product())
                    .map(|index: usize| (index % 251) as f32 / 250.0)
                    .collect();
                let inputs = HashMap::from([(input_info.name.clone(), (input.as_slice(), input_shape.to_vec()))]);
                let start = std::time::Instant::now();
                let output = session.run_f32(&inputs)
                    .unwrap_or_else(|err| panic!("{name}: CUDA inference: {err}"));
                assert_eq!(output.len(), 1, "{name}");
                let (values, shape) = output.get(&output_info.name).expect("declared output tensor");
                assert_eq!(shape.as_slice(), output_shape, "{name}");
                assert_eq!(values.len(), output_shape.iter().product::<usize>(), "{name}");
                assert!(values.iter().all(|value| value.is_finite()), "{name}");
                eprintln!("{name}: {shape:?} finite FP32 output in {:.2?}", start.elapsed());
                if let Some(root) = std::env::var_os("AURORA_FACEPLUMP_REFERENCE_DIR") {
                    let reference = std::fs::read(std::path::Path::new(&root).join(path.file_stem().unwrap()).with_extension("f32"))
                        .expect("read CPU reference in little-endian FP32");
                    assert_eq!(reference.len(), values.len() * 4, "{name}");
                    let mut exact = 0usize;
                    let mut max_abs = 0.0f32;
                    let mut sum_abs = 0.0f64;
                    for (&value, bytes) in values.iter().zip(reference.chunks_exact(4)) {
                        let expected = f32::from_le_bytes(bytes.try_into().unwrap());
                        assert!(expected.is_finite(), "{name}: nonfinite CPU reference");
                        exact += usize::from(value == expected);
                        let diff = (value - expected).abs();
                        max_abs = max_abs.max(diff);
                        sum_abs += f64::from(diff);
                    }
                    eprintln!("{name}: CPU reference exact={exact}/{} max_abs={max_abs:.8e} mae={:.8e}", values.len(), sum_abs / values.len() as f64);
                }
            }
        }
    }
}

#[cfg(windows)]
pub use imp::NativeCudaSession;
