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
    use crate::native_infer::ops_conv_gemm::{
        Conv2dParams, ConvGemmKernels, ConvTranspose2dParams, FusedActivation,
    };
    use crate::native_infer::ops_elem_norm::{BinaryOp, CompareOp, ElemNormKernels, UnaryOp};
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
    }

    impl Drop for NativeCudaSession {
        fn drop(&mut self) {
            // Explicitly drop all device buffers while `self.conv` (holding `Nvidia`) is still alive.
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

            // SAFETY: `owned_buffers`, `arena_buf`, and `im2col_buf` are explicitly dropped
            // in `NativeCudaSession::drop` before `self.conv` is dropped, and the underlying
            // CUDA driver primary context is process-global.
            let gpu_static: &'static crate::cuda::Nvidia =
                unsafe { std::mem::transmute(conv.nvidia()) };

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
                // Upload weights to VRAM. For Float16 weights we keep raw FP16 bytes OR
                // upload as-is (Cast nodes handle FP16->FP32, or Conv/MatMul unpack if needed).
                let bytes: Vec<u8> = match proto.data_type {
                    DataType::Float => {
                        if proto.raw_data.len() == proto.elem_count() * 4 {
                            proto.raw_data.clone()
                        } else {
                            let v = proto.to_f32_vec().map_err(|e| {
                                SessionError::Runtime(format!("weight decode error: {e}"))
                            })?;
                            f32_as_bytes(&v).to_vec()
                        }
                    }
                    _ => proto.raw_data.clone(),
                };
                if bytes.is_empty() {
                    weight_ptrs.insert(tid, (0u64, shape, proto.data_type));
                    continue;
                }
                let buf = gpu_static.alloc_buffer(bytes.len())?;
                gpu_static.upload_buffer(&buf, &bytes)?;
                let ptr = buf.handle();
                owned_buffers.push(buf);
                weight_ptrs.insert(tid, (ptr, shape, proto.data_type));
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

            Ok(Self {
                plan,
                conv,
                elem,
                spatial,
                weight_ptrs,
                owned_buffers,
                arena_buf,
                arena_base_ptr,
                im2col_buf,
                im2col_ptr,
                im2col_bytes: max_im2col_bytes,
                static_cpu_tensors,
            })
        }

        pub fn graph(&self) -> &IrGraph {
            &self.plan.graph
        }

        pub fn arena_peak_bytes(&self) -> usize {
            self.plan.arena.peak_bytes
        }

        /// Execute the compiled graph with named `f32` inputs and return all
        /// graph outputs as `(name -> Vec<f32>)`.
        pub fn run_f32<'a>(
            &'a self,
            inputs: &HashMap<String, (&[f32], Vec<usize>)>,
        ) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>, SessionError> {
            let gpu = self.conv.nvidia();
            let graph = &self.plan.graph;

            // Runtime tensor bindings: TensorId -> (device_ptr, shape, dtype).
            let mut gpu_tensors: HashMap<TensorId, (u64, Vec<usize>, DataType)> =
                self.weight_ptrs.clone();
            let mut cpu_tensors: HashMap<TensorId, CpuTensor> = self.static_cpu_tensors.clone();
            // Per-run allocations for graph inputs, graph outputs, and dynamic/non-arena tensors.
            let mut run_bufs: Vec<DeviceBuffer<'a>> = Vec::new();

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
            }

            // Pre-allocate graph outputs that are not in the scratch arena.
            let mut output_ids: HashMap<TensorId, (u64, Vec<usize>)> = HashMap::new();
            for &out_id in &graph.outputs {
                if let Some(sh) = graph.tensors[out_id.0].static_shape() {
                    let elem_bytes = graph.tensors[out_id.0].dtype.byte_size().unwrap_or(4);
                    let bytes = numel(&sh).max(1) * elem_bytes;
                    let buf = gpu.alloc_buffer(bytes)?;
                    let ptr = buf.handle();
                    run_bufs.push(buf);
                    output_ids.insert(out_id, (ptr, sh));
                }
            }

            // Helper: resolve output device pointer for `tid` (arena if planned, else output/dynamic buf).
            let alloc_out = |tid: TensorId,
                             shape: &[usize],
                             dtype: DataType,
                             run_bufs: &mut Vec<DeviceBuffer<'a>>|
             -> Result<u64, SessionError> {
                if let Some(&(ptr, _)) = output_ids.get(&tid) {
                    return Ok(ptr);
                }
                let bytes = numel(shape).max(1) * dtype.byte_size().unwrap_or(4);
                if let Some(alloc) = self.plan.arena.allocations.get(&tid) {
                    if bytes <= alloc.aligned_bytes && self.arena_base_ptr != 0 {
                        return Ok(self.arena_base_ptr + alloc.offset as u64);
                    }
                }
                let buf = gpu.alloc_buffer(bytes)?;
                let ptr = buf.handle();
                run_bufs.push(buf);
                Ok(ptr)
            };

            // 2. Execute nodes in topological order.
            for node in &graph.nodes {
                // Fast-path: pure CPU shape/constant evaluation when applicable.
                let is_pure_shape_op = matches!(
                    node.op,
                    Op::Constant { .. } | Op::Shape { .. } | Op::Size | Op::ConstantOfShape { .. }
                );
                let all_inputs_on_cpu = !node.inputs.is_empty()
                    && node
                        .inputs
                        .iter()
                        .all(|opt| opt.map_or(true, |id| cpu_tensors.contains_key(&id)));
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
                        | Op::Equal
                        | Op::Less
                        | Op::Greater
                        | Op::GreaterOrEqual
                        | Op::And
                        | Op::Where
                        | Op::Cast { .. }
                        | Op::Identity
                );

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
                    if let Ok(ct) = eval_cpu_node(&node.op, &inp_vals, &inp_shapes) {
                        if let Some(Some(out_id)) = node.outputs.first() {
                            // Also mirror small F32/I64 constants to GPU if needed by downstream GPU ops.
                            if let CpuData::F32(ref v) = ct.data {
                                if !v.is_empty() {
                                    let buf = gpu.alloc_buffer(v.len() * 4)?;
                                    gpu.upload_buffer(&buf, f32_as_bytes(v))?;
                                    let ptr = buf.handle();
                                    run_bufs.push(buf);
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
                                    let buf = gpu.alloc_buffer(bytes.len())?;
                                    gpu.upload_buffer(&buf, bytes)?;
                                    let ptr = buf.handle();
                                    run_bufs.push(buf);
                                    gpu_tensors.insert(*out_id, (ptr, ct.shape.clone(), DataType::Int64));
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

                match &node.op {
                    // --- Zero-copy view ops ---
                    Op::Identity => {
                        let (p, sh, dt) = req_gpu(0)?;
                        let out_id = req_out_id(0)?;
                        if let Some(&(optr, _)) = output_ids.get(&out_id) {
                            self.spatial.copy_f32(p, optr, numel(&sh))?;
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
                            .and_then(|id| cpu_tensors.get(&id))
                            .map(|c| c.to_i64_vec())
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
                        if let Some(&(optr, _)) = output_ids.get(&out_id) {
                            self.spatial.copy_f32(p, optr, numel(&out_sh))?;
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
                        if let Some(&(optr, _)) = output_ids.get(&out_id) {
                            self.spatial.copy_f32(p, optr, numel(&out_sh))?;
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
                        if let Some(&(optr, _)) = output_ids.get(&out_id) {
                            self.spatial.copy_f32(p, optr, numel(&out_sh))?;
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
                            FusedActivation::None,
                        )?;
                        let out_sh = vec![params.batch, params.c_out, params.out_h, params.out_w];
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
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
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
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
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
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
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
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
                        let y_ptr = alloc_out(out_id, &x_sh, DataType::Float, &mut run_bufs)?;
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
                        let y_ptr = alloc_out(out_id, &x_sh, DataType::Float, &mut run_bufs)?;
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
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &x_sh, DataType::Float, &mut run_bufs)?;
                        self.elem.instance_norm_nchw_f32(
                            x_ptr, s_ptr, b_ptr, y_ptr, x_sh[0], x_sh[1], x_sh[2], x_sh[3],
                            *epsilon,
                        )?;
                        gpu_tensors.insert(out_id, (y_ptr, x_sh, DataType::Float));
                    }
                    Op::Softmax { axis } => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let ax = norm_axis(*axis, x_sh.len())?;
                        let outer: usize = x_sh[..ax].iter().product();
                        let axis_dim = x_sh[ax];
                        let inner: usize = x_sh[ax + 1..].iter().product();
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &x_sh, DataType::Float, &mut run_bufs)?;
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
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
                        self.elem
                            .binary_broadcast_f32(bop, a_ptr, &a_sh, b_ptr, &b_sh, y_ptr)?;
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
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Bool, &mut run_bufs)?;
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
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Bool, &mut run_bufs)?;
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
                        let out_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
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
                        let y_ptr = alloc_out(out_id, &x_sh, DataType::Float, &mut run_bufs)?;
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
                        let y_ptr = alloc_out(out_id, &x_sh, DataType::Float, &mut run_bufs)?;
                        self.elem
                            .unary_f32(uop, x_ptr, y_ptr, numel(&x_sh), p0, p1)?;
                        gpu_tensors.insert(out_id, (y_ptr, x_sh, DataType::Float));
                    }
                    Op::Cast { to } => {
                        let (x_ptr, x_sh, src_dt) = req_gpu(0)?;
                        let out_id = req_out_id(0)?;
                        if src_dt == *to {
                            gpu_tensors.insert(out_id, (x_ptr, x_sh, *to));
                        } else {
                            let y_ptr = alloc_out(out_id, &x_sh, *to, &mut run_bufs)?;
                            let total = numel(&x_sh);
                            match (src_dt, *to) {
                                (DataType::Float16, DataType::Float) => {
                                    self.elem.cast_f16_to_f32(x_ptr, y_ptr, total)?;
                                }
                                (DataType::Float, DataType::Float16) => {
                                    self.elem.cast_f32_to_f16(x_ptr, y_ptr, total)?;
                                }
                                (DataType::Int64, DataType::Float) => {
                                    self.elem.cast_i64_to_f32(x_ptr, y_ptr, total)?;
                                }
                                (DataType::Float, DataType::Int64) => {
                                    self.elem.cast_f32_to_i64(x_ptr, y_ptr, total)?;
                                }
                                (DataType::Bool, DataType::Float) => {
                                    let d_one = gpu.alloc_buffer(4)?;
                                    let d_zero = gpu.alloc_buffer(4)?;
                                    gpu.upload_buffer(&d_one, f32_as_bytes(&[1.0f32]))?;
                                    gpu.upload_buffer(&d_zero, f32_as_bytes(&[0.0f32]))?;
                                    self.elem.where_broadcast_f32(
                                        x_ptr,
                                        &x_sh,
                                        d_one.handle(),
                                        &[1],
                                        d_zero.handle(),
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
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
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
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
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
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
                        self.spatial.global_avgpool_nchw_f32(x_ptr, y_ptr, n, c, h, w)?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }
                    Op::ReduceMean(attrs) => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let axes: Vec<i64> = if let Some(a) = &attrs.axes {
                            a.clone()
                        } else if let Some(Some(aid)) = node.inputs.get(1) {
                            cpu_tensors
                                .get(aid)
                                .map(|c| c.to_i64_vec())
                                .unwrap_or_default()
                        } else {
                            (0..x_sh.len() as i64).collect()
                        };
                        let norm_axes: Vec<usize> = axes
                            .iter()
                            .map(|&a| norm_axis(a, x_sh.len()))
                            .collect::<Result<Vec<_>, _>>()?;
                        let mut out_sh = Vec::new();
                        for (i, &d) in x_sh.iter().enumerate() {
                            if norm_axes.contains(&i) {
                                if attrs.keepdims {
                                    out_sh.push(1);
                                }
                            } else {
                                out_sh.push(d);
                            }
                        }
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
                        self.spatial.reduce_mean_f32(x_ptr, y_ptr, &x_sh, &axes)?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }
                    Op::Resize(attrs) => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let (n, c, in_h, in_w) = (x_sh[0], x_sh[1], x_sh[2], x_sh[3]);
                        let out_id = req_out_id(0)?;
                        let (out_h, out_w) = if let Some(v) = node
                            .inputs
                            .get(3)
                            .and_then(|o| *o)
                            .and_then(|sid| cpu_tensors.get(&sid))
                            .map(|ct| ct.to_i64_vec())
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
                        )?;
                        let out_sh = vec![n, c, out_h, out_w];
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
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
                        let (n, c, in_h, in_w) = (x_sh[0], x_sh[1], x_sh[2], x_sh[3]);
                        let (pt, pl, pb, pr) = (
                            pads[2] as usize,
                            pads[3] as usize,
                            pads[6] as usize,
                            pads[7] as usize,
                        );
                        let out_sh = vec![n, c, in_h + pt + pb, in_w + pl + pr];
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
                        self.spatial.pad_constant_nchw_f32(
                            x_ptr, y_ptr, n, c, in_h, in_w, pt, pb, pl, pr, const_val,
                        )?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }

                    // --- Tensor Rearrangement & Indexing ---
                    Op::Transpose { perm } => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let out_sh: Vec<usize> = perm.iter().map(|&p| x_sh[p]).collect();
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
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
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
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
                                    alloc_out(*out_id, &out_sh, DataType::Float, &mut run_bufs)?;
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
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
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
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
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
                        let y_ptr = alloc_out(out_id, &data_sh, DataType::Float, &mut run_bufs)?;
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
                        let target: Vec<usize> = cpu_tensors
                            .get(&node.inputs[1].unwrap())
                            .map(|c| c.to_i64_vec().into_iter().map(|v| v as usize).collect())
                            .ok_or_else(|| {
                                SessionError::Runtime(format!(
                                    "Expand '{}' missing CPU target shape",
                                    node.name
                                ))
                            })?;
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
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
                        self.spatial.expand_nd_f32(x_ptr, y_ptr, &x_sh, &out_sh)?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
                    }
                    Op::Tile => {
                        let (x_ptr, x_sh, _) = req_gpu(0)?;
                        let repeats: Vec<usize> = cpu_tensors
                            .get(&node.inputs[1].unwrap())
                            .map(|c| c.to_i64_vec().into_iter().map(|v| v as usize).collect())
                            .ok_or_else(|| {
                                SessionError::Runtime(format!("Tile '{}' missing repeats", node.name))
                            })?;
                        let out_sh: Vec<usize> =
                            x_sh.iter().zip(&repeats).map(|(&d, &r)| d * r).collect();
                        let out_id = req_out_id(0)?;
                        let y_ptr = alloc_out(out_id, &out_sh, DataType::Float, &mut run_bufs)?;
                        self.spatial.tile_nd_f32(x_ptr, y_ptr, &x_sh, &repeats)?;
                        gpu_tensors.insert(out_id, (y_ptr, out_sh, DataType::Float));
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

            // 3. Download all graph outputs.
            let mut results = HashMap::with_capacity(graph.outputs.len());
            for &out_id in &graph.outputs {
                let info = &graph.tensors[out_id.0];
                let (ptr, shape, _dt) = gpu_tensors.get(&out_id).ok_or_else(|| {
                    SessionError::Runtime(format!("graph output '{}' was not produced", info.name))
                })?;
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
            Ok(results)
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
            let root = match std::env::var_os("AURORA_MODELS_DIR") {
                Some(v) => PathBuf::from(v),
                None => {
                    eprintln!("set AURORA_MODELS_DIR to the models directory");
                    return;
                }
            };
            if !root.is_dir() {
                eprintln!("models dir missing: {}", root.display());
                return;
            }
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
            ];

            let mut passed = 0usize;
            for &(rel, input_name, shape) in cases {
                let path = root.join(rel);
                if !path.is_file() {
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
            assert!(passed >= 5, "expected at least 5 real models to run end-to-end, got {passed}");
        }
    }
}

#[cfg(windows)]
pub use imp::NativeCudaSession;
