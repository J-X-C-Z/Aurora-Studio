//! Pure-Rust CPU shape & small-tensor evaluator for Aurora Native Inference.
//!
//! Evaluates ONNX shape/control-flow subgraphs (`Shape`, `Size`, `Constant`,
//! `ConstantOfShape`, `Gather`, `Slice`, `Concat`, `Unsqueeze`, `Squeeze`,
//! `Reshape`, `Add`, `Sub`, `Mul`, `Div`, `Equal`, `Less`, `Greater`,
//! `GreaterOrEqual`, `And`, `Where`, `Cast`, `Identity`, `Expand`, `Tile`)
//! directly on the CPU without launching GPU kernels or GPU→CPU sync round-trips.
//!
//! Designed for `NativeCudaSession` to resolve dynamic `Reshape`, `Slice`,
//! `Resize`, `Expand`, and `If` parameters at graph compile time or runtime.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::native_infer::graph::{IrGraph, IrNode, NodeId, Op, TensorId};
use crate::native_infer::onnx_proto::{f16_to_f32, DataType, TensorProto};

/// Maximum element count of a graph weight/initializer automatically mirrored
/// into [`CpuShapeEvaluator`] for CPU shape-subgraph evaluation.
pub const DEFAULT_MAX_CPU_TENSOR_ELEMS: usize = 4096;

/// Error returned when a CPU shape/small-tensor evaluation fails.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShapeEvalError {
    UnsupportedOp(&'static str),
    UnsupportedDType(DataType),
    MissingInput { op: &'static str, index: usize },
    ShapeMismatch { op: &'static str, detail: String },
    IndexOutOfBounds { op: &'static str, index: i64, limit: usize },
    DivisionByZero,
    InvalidAxis { op: &'static str, axis: i64, rank: usize },
}

impl fmt::Display for ShapeEvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedOp(op) => write!(f, "CPU shape evaluator does not support op '{op}'"),
            Self::UnsupportedDType(dt) => write!(f, "unsupported CpuTensor dtype: {dt:?}"),
            Self::MissingInput { op, index } => {
                write!(f, "op '{op}' missing required input #{index}")
            }
            Self::ShapeMismatch { op, detail } => write!(f, "op '{op}' shape mismatch: {detail}"),
            Self::IndexOutOfBounds { op, index, limit } => {
                write!(f, "op '{op}' index {index} out of bounds for dim {limit}")
            }
            Self::DivisionByZero => write!(f, "integer division by zero in shape subgraph"),
            Self::InvalidAxis { op, axis, rank } => {
                write!(f, "op '{op}' invalid axis {axis} for rank {rank}")
            }
        }
    }
}

impl std::error::Error for ShapeEvalError {}

/// Compact typed payload for [`CpuTensor`].
#[derive(Clone, Debug, PartialEq)]
pub enum CpuData {
    I64(Vec<i64>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
}

impl CpuData {
    pub fn len(&self) -> usize {
        match self {
            Self::I64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Bool(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> DataType {
        match self {
            Self::I64(_) => DataType::Int64,
            Self::F32(_) => DataType::Float,
            Self::Bool(_) => DataType::Bool,
        }
    }
}

/// Lightweight CPU tensor (`I64` / `F32` / `Bool`) for shape & control-flow evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuTensor {
    pub shape: Vec<usize>,
    pub data: CpuData,
}

impl CpuTensor {
    pub fn from_i64(shape: Vec<usize>, data: Vec<i64>) -> Result<Self, ShapeEvalError> {
        let expected = numel(&shape);
        if data.len() != expected {
            return Err(ShapeEvalError::ShapeMismatch {
                op: "CpuTensor::from_i64",
                detail: format!("shape {:?} expects {expected} elems, got {}", shape, data.len()),
            });
        }
        Ok(Self {
            shape,
            data: CpuData::I64(data),
        })
    }

    pub fn from_f32(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, ShapeEvalError> {
        let expected = numel(&shape);
        if data.len() != expected {
            return Err(ShapeEvalError::ShapeMismatch {
                op: "CpuTensor::from_f32",
                detail: format!("shape {:?} expects {expected} elems, got {}", shape, data.len()),
            });
        }
        Ok(Self {
            shape,
            data: CpuData::F32(data),
        })
    }

    pub fn from_bool(shape: Vec<usize>, data: Vec<bool>) -> Result<Self, ShapeEvalError> {
        let expected = numel(&shape);
        if data.len() != expected {
            return Err(ShapeEvalError::ShapeMismatch {
                op: "CpuTensor::from_bool",
                detail: format!("shape {:?} expects {expected} elems, got {}", shape, data.len()),
            });
        }
        Ok(Self {
            shape,
            data: CpuData::Bool(data),
        })
    }

    pub fn scalar_i64(v: i64) -> Self {
        Self {
            shape: Vec::new(),
            data: CpuData::I64(vec![v]),
        }
    }

    pub fn scalar_f32(v: f32) -> Self {
        Self {
            shape: Vec::new(),
            data: CpuData::F32(vec![v]),
        }
    }

    pub fn scalar_bool(v: bool) -> Self {
        Self {
            shape: Vec::new(),
            data: CpuData::Bool(vec![v]),
        }
    }

    /// Decode an ONNX [`TensorProto`] into a [`CpuTensor`].
    pub fn from_tensor_proto(tp: &TensorProto) -> Result<Self, ShapeEvalError> {
        let shape: Vec<usize> = tp.dims.iter().map(|&d| d.max(0) as usize).collect();
        let expected = numel(&shape);
        match tp.data_type {
            DataType::Int64 | DataType::Int32 => {
                let vals = tp.to_i64_vec().map_err(|e| ShapeEvalError::ShapeMismatch {
                    op: "from_tensor_proto",
                    detail: e.to_string(),
                })?;
                if vals.len() != expected {
                    return Err(ShapeEvalError::ShapeMismatch {
                        op: "from_tensor_proto",
                        detail: format!("expected {expected} i64 elems, got {}", vals.len()),
                    });
                }
                Ok(Self {
                    shape,
                    data: CpuData::I64(vals),
                })
            }
            DataType::Float | DataType::Float16 | DataType::Double => {
                let vals = tp.to_f32_vec().map_err(|e| ShapeEvalError::ShapeMismatch {
                    op: "from_tensor_proto",
                    detail: e.to_string(),
                })?;
                if vals.len() != expected {
                    return Err(ShapeEvalError::ShapeMismatch {
                        op: "from_tensor_proto",
                        detail: format!("expected {expected} f32 elems, got {}", vals.len()),
                    });
                }
                Ok(Self {
                    shape,
                    data: CpuData::F32(vals),
                })
            }
            DataType::Bool => {
                let vals: Vec<bool> = tp.raw_data.iter().map(|&b| b != 0).collect();
                if vals.len() != expected {
                    return Err(ShapeEvalError::ShapeMismatch {
                        op: "from_tensor_proto",
                        detail: format!("expected {expected} bool elems, got {}", vals.len()),
                    });
                }
                Ok(Self {
                    shape,
                    data: CpuData::Bool(vals),
                })
            }
            DataType::Uint8 => {
                let vals: Vec<i64> = tp.raw_data.iter().map(|&b| b as i64).collect();
                if vals.len() != expected {
                    return Err(ShapeEvalError::ShapeMismatch {
                        op: "from_tensor_proto",
                        detail: format!("expected {expected} u8 elems, got {}", vals.len()),
                    });
                }
                Ok(Self {
                    shape,
                    data: CpuData::I64(vals),
                })
            }
            other => Err(ShapeEvalError::UnsupportedDType(other)),
        }
    }

    pub fn dtype(&self) -> DataType {
        self.data.dtype()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_i64(&self) -> Option<&[i64]> {
        match &self.data {
            CpuData::I64(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.data {
            CpuData::F32(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<&[bool]> {
        match &self.data {
            CpuData::Bool(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Convert payload to `Vec<i64>` (coercing `F32` / `Bool` if needed).
    pub fn to_i64_vec(&self) -> Vec<i64> {
        match &self.data {
            CpuData::I64(v) => v.clone(),
            CpuData::F32(v) => v.iter().map(|&x| x as i64).collect(),
            CpuData::Bool(v) => v.iter().map(|&b| i64::from(b)).collect(),
        }
    }

    /// Convert payload to `Vec<f32>` (coercing `I64` / `Bool` if needed).
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match &self.data {
            CpuData::F32(v) => v.clone(),
            CpuData::I64(v) => v.iter().map(|&x| x as f32).collect(),
            CpuData::Bool(v) => v.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect(),
        }
    }

    /// Convert payload to `Vec<bool>`.
    pub fn to_bool_vec(&self) -> Vec<bool> {
        match &self.data {
            CpuData::Bool(v) => v.clone(),
            CpuData::I64(v) => v.iter().map(|&x| x != 0).collect(),
            CpuData::F32(v) => v.iter().map(|&x| x != 0.0).collect(),
        }
    }

    pub fn as_i64_scalar(&self) -> Option<i64> {
        if self.len() == 1 {
            self.as_i64().map(|s| s[0])
        } else {
            None
        }
    }

    pub fn as_bool_scalar(&self) -> Option<bool> {
        if self.len() == 1 {
            Some(self.to_bool_vec()[0])
        } else {
            None
        }
    }
}

// ============================================================================
// Pure-Rust Operator Evaluation
// ============================================================================

#[inline]
fn numel(shape: &[usize]) -> usize {
    shape.iter().copied().product::<usize>()
}

fn normalize_axis(op: &'static str, axis: i64, rank: usize) -> Result<usize, ShapeEvalError> {
    let r = rank as i64;
    let norm = if axis < 0 { axis + r } else { axis };
    if norm < 0 || norm >= r {
        return Err(ShapeEvalError::InvalidAxis { op, axis, rank });
    }
    Ok(norm as usize)
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn broadcast_shapes(op: &'static str, a: &[usize], b: &[usize]) -> Result<Vec<usize>, ShapeEvalError> {
    let rank = a.len().max(b.len());
    let mut out = vec![1usize; rank];
    for i in 0..rank {
        let da = if i + a.len() >= rank {
            a[i + a.len() - rank]
        } else {
            1
        };
        let db = if i + b.len() >= rank {
            b[i + b.len() - rank]
        } else {
            1
        };
        if da == db {
            out[i] = da;
        } else if da == 1 {
            out[i] = db;
        } else if db == 1 {
            out[i] = da;
        } else {
            return Err(ShapeEvalError::ShapeMismatch {
                op,
                detail: format!("cannot broadcast {a:?} and {b:?}"),
            });
        }
    }
    Ok(out)
}

fn broadcast_strides(src_shape: &[usize], out_rank: usize) -> Vec<usize> {
    let src_strides = compute_strides(src_shape);
    let mut out = vec![0usize; out_rank];
    let pad = out_rank - src_shape.len();
    for i in 0..src_shape.len() {
        out[pad + i] = if src_shape[i] == 1 { 0 } else { src_strides[i] };
    }
    out
}

#[inline]
fn unravel_offset(mut linear: usize, out_shape: &[usize], b_strides: &[usize]) -> usize {
    let mut off = 0usize;
    for i in (0..out_shape.len()).rev() {
        let dim = out_shape[i];
        let coord = if dim > 0 { linear % dim } else { 0 };
        linear = if dim > 0 { linear / dim } else { 0 };
        off += coord * b_strides[i];
    }
    off
}

/// Evaluate `Shape` on an input tensor's known runtime shape.
pub fn eval_shape(
    input_shape: &[usize],
    start: Option<i64>,
    end: Option<i64>,
) -> CpuTensor {
    let rank = input_shape.len() as i64;
    let s = start.unwrap_or(0);
    let e = end.unwrap_or(rank);
    let s = (if s < 0 { s + rank } else { s }).clamp(0, rank) as usize;
    let e = (if e < 0 { e + rank } else { e }).clamp(0, rank) as usize;
    let dims: Vec<i64> = if s <= e {
        input_shape[s..e].iter().map(|&d| d as i64).collect()
    } else {
        Vec::new()
    };
    CpuTensor {
        shape: vec![dims.len()],
        data: CpuData::I64(dims),
    }
}

/// Evaluate `Size` on an input tensor's known runtime shape.
pub fn eval_size(input_shape: &[usize]) -> CpuTensor {
    CpuTensor::scalar_i64(numel(input_shape) as i64)
}

/// Evaluate `ConstantOfShape`.
pub fn eval_constant_of_shape(
    shape_tensor: &CpuTensor,
    value: &TensorProto,
) -> Result<CpuTensor, ShapeEvalError> {
    let dims_i64 = shape_tensor.to_i64_vec();
    let mut out_shape = Vec::with_capacity(dims_i64.len());
    for &d in &dims_i64 {
        if d < 0 {
            return Err(ShapeEvalError::ShapeMismatch {
                op: "ConstantOfShape",
                detail: format!("negative dim {d}"),
            });
        }
        out_shape.push(d as usize);
    }
    let n = numel(&out_shape);
    if value.data_type == DataType::Undefined || value.raw_data.is_empty() {
        return Ok(CpuTensor {
            shape: out_shape,
            data: CpuData::F32(vec![0.0; n]),
        });
    }
    let fill = CpuTensor::from_tensor_proto(value)?;
    let data = match fill.data {
        CpuData::I64(v) => CpuData::I64(vec![v.first().copied().unwrap_or(0); n]),
        CpuData::F32(v) => CpuData::F32(vec![v.first().copied().unwrap_or(0.0); n]),
        CpuData::Bool(v) => CpuData::Bool(vec![v.first().copied().unwrap_or(false); n]),
    };
    Ok(CpuTensor {
        shape: out_shape,
        data,
    })
}

/// Evaluate ONNX `Gather` along `axis`.
pub fn eval_gather(
    data: &CpuTensor,
    indices: &CpuTensor,
    axis: i64,
) -> Result<CpuTensor, ShapeEvalError> {
    let axis = normalize_axis("Gather", axis, data.rank())?;
    let axis_dim = data.shape[axis];
    let outer: usize = data.shape[..axis].iter().product();
    let inner: usize = data.shape[axis + 1..].iter().product();

    let idx_i64 = indices.to_i64_vec();
    let mut norm_indices = Vec::with_capacity(idx_i64.len());
    for &raw_idx in &idx_i64 {
        let idx = if raw_idx < 0 {
            raw_idx + axis_dim as i64
        } else {
            raw_idx
        };
        if idx < 0 || idx >= axis_dim as i64 {
            return Err(ShapeEvalError::IndexOutOfBounds {
                op: "Gather",
                index: raw_idx,
                limit: axis_dim,
            });
        }
        norm_indices.push(idx as usize);
    }

    let mut out_shape = Vec::with_capacity(data.rank() - 1 + indices.rank());
    out_shape.extend_from_slice(&data.shape[..axis]);
    out_shape.extend_from_slice(&indices.shape);
    out_shape.extend_from_slice(&data.shape[axis + 1..]);

    let out_len = numel(&out_shape);
    let gather_copy = |out_data: &mut dyn FnMut(usize)| {
        for o in 0..outer {
            let base_o = o * axis_dim * inner;
            for &idx in &norm_indices {
                let src = base_o + idx * inner;
                for k in 0..inner {
                    out_data(src + k);
                }
            }
        }
    };

    let out_data = match &data.data {
        CpuData::I64(src) => {
            let mut dst = Vec::with_capacity(out_len);
            gather_copy(&mut |i| dst.push(src[i]));
            CpuData::I64(dst)
        }
        CpuData::F32(src) => {
            let mut dst = Vec::with_capacity(out_len);
            gather_copy(&mut |i| dst.push(src[i]));
            CpuData::F32(dst)
        }
        CpuData::Bool(src) => {
            let mut dst = Vec::with_capacity(out_len);
            gather_copy(&mut |i| dst.push(src[i]));
            CpuData::Bool(dst)
        }
    };

    Ok(CpuTensor {
        shape: out_shape,
        data: out_data,
    })
}

/// Evaluate ONNX opset>=10 `Slice(data, starts, ends, [axes], [steps])`.
pub fn eval_slice(
    data: &CpuTensor,
    starts: &CpuTensor,
    ends: &CpuTensor,
    axes: Option<&CpuTensor>,
    steps: Option<&CpuTensor>,
) -> Result<CpuTensor, ShapeEvalError> {
    let rank = data.rank();
    let starts_v = starts.to_i64_vec();
    let ends_v = ends.to_i64_vec();
    if starts_v.len() != ends_v.len() {
        return Err(ShapeEvalError::ShapeMismatch {
            op: "Slice",
            detail: "starts and ends length mismatch".into(),
        });
    }
    let axes_v: Vec<usize> = match axes {
        Some(ax) => ax
            .to_i64_vec()
            .into_iter()
            .map(|a| normalize_axis("Slice", a, rank))
            .collect::<Result<Vec<_>, _>>()?,
        None => (0..starts_v.len()).collect(),
    };
    let steps_v: Vec<i64> = match steps {
        Some(st) => st.to_i64_vec(),
        None => vec![1i64; starts_v.len()],
    };
    if axes_v.len() != starts_v.len() || steps_v.len() != starts_v.len() {
        return Err(ShapeEvalError::ShapeMismatch {
            op: "Slice",
            detail: "axes/steps length mismatch with starts".into(),
        });
    }

    // Per-axis (start, step, out_dim)
    let mut axis_start: Vec<i64> = vec![0; rank];
    let mut axis_step: Vec<i64> = vec![1; rank];
    let mut out_shape: Vec<usize> = data.shape.clone();

    for i in 0..axes_v.len() {
        let ax = axes_v[i];
        let dim = data.shape[ax] as i64;
        let step = steps_v[i];
        if step == 0 {
            return Err(ShapeEvalError::ShapeMismatch {
                op: "Slice",
                detail: "step cannot be 0".into(),
            });
        }
        let mut s = starts_v[i];
        let mut e = ends_v[i];
        if s < 0 {
            s += dim;
        }
        if e < 0 {
            e += dim;
        }
        let (clamped_s, clamped_e, len) = if step > 0 {
            let cs = s.clamp(0, dim);
            let ce = e.clamp(0, dim);
            let len = if ce > cs {
                ((ce - cs + step - 1) / step) as usize
            } else {
                0
            };
            (cs, ce, len)
        } else {
            let cs = s.clamp(0, (dim - 1).max(0));
            let ce = e.clamp(-1, dim - 1);
            let len = if dim > 0 && cs > ce {
                ((cs - ce + (-step) - 1) / (-step)) as usize
            } else {
                0
            };
            (cs, ce, len)
        };
        let _ = clamped_e;
        axis_start[ax] = clamped_s;
        axis_step[ax] = step;
        out_shape[ax] = len;
    }

    let in_strides = compute_strides(&data.shape);
    let out_len = numel(&out_shape);

    let map_idx = |mut out_linear: usize| -> usize {
        let mut in_off = 0usize;
        for ax in (0..rank).rev() {
            let d = out_shape[ax];
            let coord = out_linear % d;
            out_linear /= d;
            let src_coord = (axis_start[ax] + (coord as i64) * axis_step[ax]) as usize;
            in_off += src_coord * in_strides[ax];
        }
        in_off
    };

    let out_data = match &data.data {
        CpuData::I64(src) => CpuData::I64((0..out_len).map(|i| src[map_idx(i)]).collect()),
        CpuData::F32(src) => CpuData::F32((0..out_len).map(|i| src[map_idx(i)]).collect()),
        CpuData::Bool(src) => CpuData::Bool((0..out_len).map(|i| src[map_idx(i)]).collect()),
    };

    Ok(CpuTensor {
        shape: out_shape,
        data: out_data,
    })
}

/// Evaluate ONNX `Concat` along `axis`.
pub fn eval_concat(inputs: &[&CpuTensor], axis: i64) -> Result<CpuTensor, ShapeEvalError> {
    if inputs.is_empty() {
        return Err(ShapeEvalError::MissingInput {
            op: "Concat",
            index: 0,
        });
    }
    let rank = inputs[0].rank();
    let axis = normalize_axis("Concat", axis, rank)?;
    let dtype = inputs[0].dtype();

    let mut out_shape = inputs[0].shape.clone();
    out_shape[axis] = 0;
    for t in inputs {
        if t.rank() != rank || t.dtype() != dtype {
            return Err(ShapeEvalError::ShapeMismatch {
                op: "Concat",
                detail: "rank or dtype mismatch across Concat inputs".into(),
            });
        }
        for d in 0..rank {
            if d != axis && t.shape[d] != out_shape[d] {
                return Err(ShapeEvalError::ShapeMismatch {
                    op: "Concat",
                    detail: format!("non-concat dim {d} mismatch"),
                });
            }
        }
        out_shape[axis] += t.shape[axis];
    }

    let outer: usize = out_shape[..axis].iter().product();
    let inner: usize = out_shape[axis + 1..].iter().product();
    let out_len = numel(&out_shape);

    let out_data = match dtype {
        DataType::Int64 => {
            let mut dst = Vec::with_capacity(out_len);
            for o in 0..outer {
                for t in inputs {
                    let chunk = t.shape[axis] * inner;
                    let src = t.as_i64().unwrap();
                    dst.extend_from_slice(&src[o * chunk..(o + 1) * chunk]);
                }
            }
            CpuData::I64(dst)
        }
        DataType::Float => {
            let mut dst = Vec::with_capacity(out_len);
            for o in 0..outer {
                for t in inputs {
                    let chunk = t.shape[axis] * inner;
                    let src = t.as_f32().unwrap();
                    dst.extend_from_slice(&src[o * chunk..(o + 1) * chunk]);
                }
            }
            CpuData::F32(dst)
        }
        DataType::Bool => {
            let mut dst = Vec::with_capacity(out_len);
            for o in 0..outer {
                for t in inputs {
                    let chunk = t.shape[axis] * inner;
                    let src = t.as_bool().unwrap();
                    dst.extend_from_slice(&src[o * chunk..(o + 1) * chunk]);
                }
            }
            CpuData::Bool(dst)
        }
        other => return Err(ShapeEvalError::UnsupportedDType(other)),
    };

    Ok(CpuTensor {
        shape: out_shape,
        data: out_data,
    })
}

/// Evaluate ONNX `Unsqueeze`.
pub fn eval_unsqueeze(data: &CpuTensor, axes: &[i64]) -> Result<CpuTensor, ShapeEvalError> {
    let out_rank = data.rank() + axes.len();
    let mut norm_axes: Vec<usize> = axes
        .iter()
        .map(|&a| normalize_axis("Unsqueeze", a, out_rank))
        .collect::<Result<Vec<_>, _>>()?;
    norm_axes.sort_unstable();
    norm_axes.dedup();
    if norm_axes.len() != axes.len() {
        return Err(ShapeEvalError::ShapeMismatch {
            op: "Unsqueeze",
            detail: "duplicate axes".into(),
        });
    }
    let mut out_shape = Vec::with_capacity(out_rank);
    let mut src_i = 0;
    for d in 0..out_rank {
        if norm_axes.binary_search(&d).is_ok() {
            out_shape.push(1);
        } else {
            out_shape.push(data.shape[src_i]);
            src_i += 1;
        }
    }
    Ok(CpuTensor {
        shape: out_shape,
        data: data.data.clone(),
    })
}

/// Evaluate ONNX `Squeeze` (`axes = None` removes all unit dimensions).
pub fn eval_squeeze(data: &CpuTensor, axes: Option<&[i64]>) -> Result<CpuTensor, ShapeEvalError> {
    let rank = data.rank();
    let out_shape = match axes {
        None => data.shape.iter().copied().filter(|&d| d != 1).collect(),
        Some(ax_list) => {
            let mut remove = HashSet::with_capacity(ax_list.len());
            for &a in ax_list {
                let norm = normalize_axis("Squeeze", a, rank)?;
                if data.shape[norm] != 1 {
                    return Err(ShapeEvalError::ShapeMismatch {
                        op: "Squeeze",
                        detail: format!("cannot squeeze axis {norm} with dim {}", data.shape[norm]),
                    });
                }
                remove.insert(norm);
            }
            data.shape
                .iter()
                .enumerate()
                .filter_map(|(i, &d)| if remove.contains(&i) { None } else { Some(d) })
                .collect()
        }
    };
    Ok(CpuTensor {
        shape: out_shape,
        data: data.data.clone(),
    })
}

/// Evaluate ONNX `Reshape` with `allowzero`.
pub fn eval_reshape(
    data: &CpuTensor,
    target_shape: &[i64],
    allowzero: bool,
) -> Result<CpuTensor, ShapeEvalError> {
    let total = data.len();
    let mut out_shape = Vec::with_capacity(target_shape.len());
    let mut neg_one_idx = None;
    let mut known_prod = 1usize;

    for (i, &d) in target_shape.iter().enumerate() {
        if d == -1 {
            if neg_one_idx.is_some() {
                return Err(ShapeEvalError::ShapeMismatch {
                    op: "Reshape",
                    detail: "multiple -1 dimensions".into(),
                });
            }
            neg_one_idx = Some(i);
            out_shape.push(1);
        } else if d == 0 && !allowzero {
            let copied = *data.shape.get(i).ok_or_else(|| ShapeEvalError::ShapeMismatch {
                op: "Reshape",
                detail: format!("0 dim at axis {i} exceeds input rank {}", data.rank()),
            })?;
            known_prod = known_prod.saturating_mul(copied);
            out_shape.push(copied);
        } else if d < 0 {
            return Err(ShapeEvalError::ShapeMismatch {
                op: "Reshape",
                detail: format!("invalid target dim {d}"),
            });
        } else {
            let u = d as usize;
            known_prod = known_prod.saturating_mul(u);
            out_shape.push(u);
        }
    }

    if let Some(idx) = neg_one_idx {
        if known_prod == 0 || total % known_prod != 0 {
            return Err(ShapeEvalError::ShapeMismatch {
                op: "Reshape",
                detail: format!("cannot infer -1 for total={total}, known_prod={known_prod}"),
            });
        }
        out_shape[idx] = total / known_prod;
    }

    if numel(&out_shape) != total {
        return Err(ShapeEvalError::ShapeMismatch {
            op: "Reshape",
            detail: format!("target {:?} numel != input numel {total}", out_shape),
        });
    }

    Ok(CpuTensor {
        shape: out_shape,
        data: data.data.clone(),
    })
}

/// Evaluate ONNX `Expand` (NumPy unidirectional/bidirectional broadcast to target shape).
pub fn eval_expand(data: &CpuTensor, target_shape: &[i64]) -> Result<CpuTensor, ShapeEvalError> {
    let target: Vec<usize> = target_shape
        .iter()
        .map(|&d| {
            if d < 0 {
                Err(ShapeEvalError::ShapeMismatch {
                    op: "Expand",
                    detail: format!("negative dim {d}"),
                })
            } else {
                Ok(d as usize)
            }
        })
        .collect::<Result<_, _>>()?;
    let out_shape = broadcast_shapes("Expand", &data.shape, &target)?;
    let b_strides = broadcast_strides(&data.shape, out_shape.len());
    let out_len = numel(&out_shape);
    let out_data = match &data.data {
        CpuData::I64(src) => CpuData::I64(
            (0..out_len)
                .map(|i| src[unravel_offset(i, &out_shape, &b_strides)])
                .collect(),
        ),
        CpuData::F32(src) => CpuData::F32(
            (0..out_len)
                .map(|i| src[unravel_offset(i, &out_shape, &b_strides)])
                .collect(),
        ),
        CpuData::Bool(src) => CpuData::Bool(
            (0..out_len)
                .map(|i| src[unravel_offset(i, &out_shape, &b_strides)])
                .collect(),
        ),
    };
    Ok(CpuTensor {
        shape: out_shape,
        data: out_data,
    })
}

/// Evaluate ONNX `Tile`.
pub fn eval_tile(data: &CpuTensor, repeats: &[i64]) -> Result<CpuTensor, ShapeEvalError> {
    if repeats.len() != data.rank() {
        return Err(ShapeEvalError::ShapeMismatch {
            op: "Tile",
            detail: format!("repeats len {} != rank {}", repeats.len(), data.rank()),
        });
    }
    let out_shape: Vec<usize> = data
        .shape
        .iter()
        .zip(repeats.iter())
        .map(|(&d, &r)| d * r.max(0) as usize)
        .collect();
    let in_strides = compute_strides(&data.shape);
    let out_len = numel(&out_shape);
    let map_idx = |mut lin: usize| -> usize {
        let mut off = 0usize;
        for ax in (0..data.rank()).rev() {
            let od = out_shape[ax];
            let coord = lin % od;
            lin /= od;
            off += (coord % data.shape[ax]) * in_strides[ax];
        }
        off
    };
    let out_data = match &data.data {
        CpuData::I64(src) => CpuData::I64((0..out_len).map(|i| src[map_idx(i)]).collect()),
        CpuData::F32(src) => CpuData::F32((0..out_len).map(|i| src[map_idx(i)]).collect()),
        CpuData::Bool(src) => CpuData::Bool((0..out_len).map(|i| src[map_idx(i)]).collect()),
    };
    Ok(CpuTensor {
        shape: out_shape,
        data: out_data,
    })
}

#[derive(Clone, Copy)]
enum BinArith {
    Add,
    Sub,
    Mul,
    Div,
}

fn eval_bin_arith(
    op_name: &'static str,
    kind: BinArith,
    a: &CpuTensor,
    b: &CpuTensor,
) -> Result<CpuTensor, ShapeEvalError> {
    let out_shape = broadcast_shapes(op_name, &a.shape, &b.shape)?;
    let sa = broadcast_strides(&a.shape, out_shape.len());
    let sb = broadcast_strides(&b.shape, out_shape.len());
    let out_len = numel(&out_shape);

    match (&a.data, &b.data) {
        (CpuData::I64(va), CpuData::I64(vb)) => {
            let mut out = Vec::with_capacity(out_len);
            for i in 0..out_len {
                let x = va[unravel_offset(i, &out_shape, &sa)];
                let y = vb[unravel_offset(i, &out_shape, &sb)];
                let v = match kind {
                    BinArith::Add => x.wrapping_add(y),
                    BinArith::Sub => x.wrapping_sub(y),
                    BinArith::Mul => x.wrapping_mul(y),
                    BinArith::Div => {
                        if y == 0 {
                            return Err(ShapeEvalError::DivisionByZero);
                        }
                        x / y
                    }
                };
                out.push(v);
            }
            Ok(CpuTensor {
                shape: out_shape,
                data: CpuData::I64(out),
            })
        }
        (CpuData::F32(va), CpuData::F32(vb)) => {
            let mut out = Vec::with_capacity(out_len);
            for i in 0..out_len {
                let x = va[unravel_offset(i, &out_shape, &sa)];
                let y = vb[unravel_offset(i, &out_shape, &sb)];
                let v = match kind {
                    BinArith::Add => x + y,
                    BinArith::Sub => x - y,
                    BinArith::Mul => x * y,
                    BinArith::Div => x / y,
                };
                out.push(v);
            }
            Ok(CpuTensor {
                shape: out_shape,
                data: CpuData::F32(out),
            })
        }
        _ => Err(ShapeEvalError::ShapeMismatch {
            op: op_name,
            detail: format!("dtype mismatch: {:?} vs {:?}", a.dtype(), b.dtype()),
        }),
    }
}

#[derive(Clone, Copy)]
enum BinCmp {
    Equal,
    Less,
    Greater,
    GreaterOrEqual,
}

fn eval_bin_cmp(
    op_name: &'static str,
    kind: BinCmp,
    a: &CpuTensor,
    b: &CpuTensor,
) -> Result<CpuTensor, ShapeEvalError> {
    let out_shape = broadcast_shapes(op_name, &a.shape, &b.shape)?;
    let sa = broadcast_strides(&a.shape, out_shape.len());
    let sb = broadcast_strides(&b.shape, out_shape.len());
    let out_len = numel(&out_shape);
    let mut out = Vec::with_capacity(out_len);

    match (&a.data, &b.data) {
        (CpuData::I64(va), CpuData::I64(vb)) => {
            for i in 0..out_len {
                let x = va[unravel_offset(i, &out_shape, &sa)];
                let y = vb[unravel_offset(i, &out_shape, &sb)];
                out.push(match kind {
                    BinCmp::Equal => x == y,
                    BinCmp::Less => x < y,
                    BinCmp::Greater => x > y,
                    BinCmp::GreaterOrEqual => x >= y,
                });
            }
        }
        (CpuData::F32(va), CpuData::F32(vb)) => {
            for i in 0..out_len {
                let x = va[unravel_offset(i, &out_shape, &sa)];
                let y = vb[unravel_offset(i, &out_shape, &sb)];
                out.push(match kind {
                    BinCmp::Equal => x == y,
                    BinCmp::Less => x < y,
                    BinCmp::Greater => x > y,
                    BinCmp::GreaterOrEqual => x >= y,
                });
            }
        }
        (CpuData::Bool(va), CpuData::Bool(vb)) => {
            for i in 0..out_len {
                let x = va[unravel_offset(i, &out_shape, &sa)];
                let y = vb[unravel_offset(i, &out_shape, &sb)];
                out.push(match kind {
                    BinCmp::Equal => x == y,
                    BinCmp::Less => !x && y,
                    BinCmp::Greater => x && !y,
                    BinCmp::GreaterOrEqual => x >= y,
                });
            }
        }
        _ => {
            return Err(ShapeEvalError::ShapeMismatch {
                op: op_name,
                detail: format!("dtype mismatch: {:?} vs {:?}", a.dtype(), b.dtype()),
            })
        }
    }

    Ok(CpuTensor {
        shape: out_shape,
        data: CpuData::Bool(out),
    })
}

fn eval_and(a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor, ShapeEvalError> {
    let out_shape = broadcast_shapes("And", &a.shape, &b.shape)?;
    let sa = broadcast_strides(&a.shape, out_shape.len());
    let sb = broadcast_strides(&b.shape, out_shape.len());
    let out_len = numel(&out_shape);
    let va = a.to_bool_vec();
    let vb = b.to_bool_vec();
    let out = (0..out_len)
        .map(|i| {
            va[unravel_offset(i, &out_shape, &sa)] && vb[unravel_offset(i, &out_shape, &sb)]
        })
        .collect();
    Ok(CpuTensor {
        shape: out_shape,
        data: CpuData::Bool(out),
    })
}

/// Evaluate ONNX `Where(cond, x, y)` with 3-way NumPy broadcasting.
pub fn eval_where(
    cond: &CpuTensor,
    x: &CpuTensor,
    y: &CpuTensor,
) -> Result<CpuTensor, ShapeEvalError> {
    let xy_shape = broadcast_shapes("Where", &x.shape, &y.shape)?;
    let out_shape = broadcast_shapes("Where", &cond.shape, &xy_shape)?;
    let sc = broadcast_strides(&cond.shape, out_shape.len());
    let sx = broadcast_strides(&x.shape, out_shape.len());
    let sy = broadcast_strides(&y.shape, out_shape.len());
    let out_len = numel(&out_shape);
    let vc = cond.to_bool_vec();

    let out_data = match (&x.data, &y.data) {
        (CpuData::I64(vx), CpuData::I64(vy)) => CpuData::I64(
            (0..out_len)
                .map(|i| {
                    if vc[unravel_offset(i, &out_shape, &sc)] {
                        vx[unravel_offset(i, &out_shape, &sx)]
                    } else {
                        vy[unravel_offset(i, &out_shape, &sy)]
                    }
                })
                .collect(),
        ),
        (CpuData::F32(vx), CpuData::F32(vy)) => CpuData::F32(
            (0..out_len)
                .map(|i| {
                    if vc[unravel_offset(i, &out_shape, &sc)] {
                        vx[unravel_offset(i, &out_shape, &sx)]
                    } else {
                        vy[unravel_offset(i, &out_shape, &sy)]
                    }
                })
                .collect(),
        ),
        (CpuData::Bool(vx), CpuData::Bool(vy)) => CpuData::Bool(
            (0..out_len)
                .map(|i| {
                    if vc[unravel_offset(i, &out_shape, &sc)] {
                        vx[unravel_offset(i, &out_shape, &sx)]
                    } else {
                        vy[unravel_offset(i, &out_shape, &sy)]
                    }
                })
                .collect(),
        ),
        _ => {
            return Err(ShapeEvalError::ShapeMismatch {
                op: "Where",
                detail: format!("x/y dtype mismatch: {:?} vs {:?}", x.dtype(), y.dtype()),
            })
        }
    };

    Ok(CpuTensor {
        shape: out_shape,
        data: out_data,
    })
}

/// Evaluate ONNX `Cast`.
pub fn eval_cast(x: &CpuTensor, to: DataType) -> Result<CpuTensor, ShapeEvalError> {
    let data = match to {
        DataType::Int64 | DataType::Int32 | DataType::Uint8 => CpuData::I64(x.to_i64_vec()),
        DataType::Float | DataType::Double => CpuData::F32(x.to_f32_vec()),
        DataType::Float16 => {
            let vals = x
                .to_f32_vec()
                .into_iter()
                .map(|v| f16_to_f32(crate::native_infer::onnx_proto::f32_to_f16(v)))
                .collect();
            CpuData::F32(vals)
        }
        DataType::Bool => CpuData::Bool(x.to_bool_vec()),
        other => return Err(ShapeEvalError::UnsupportedDType(other)),
    };
    Ok(CpuTensor {
        shape: x.shape.clone(),
        data,
    })
}

/// Dispatch a single IR [`Op`] on CPU tensors and/or known input shapes.
pub fn eval_op(
    op: &Op,
    inputs: &[Option<&CpuTensor>],
    input_shapes: &[Option<&[usize]>],
) -> Result<CpuTensor, ShapeEvalError> {
    let req_tensor = |idx: usize| -> Result<&CpuTensor, ShapeEvalError> {
        inputs
            .get(idx)
            .and_then(|o| *o)
            .ok_or(ShapeEvalError::MissingInput {
                op: op.name(),
                index: idx,
            })
    };

    match op {
        Op::Shape { start, end } => {
            let shape = input_shapes
                .first()
                .and_then(|s| *s)
                .or_else(|| inputs.first().and_then(|t| t.map(|c| c.shape.as_slice())))
                .ok_or(ShapeEvalError::MissingInput {
                    op: "Shape",
                    index: 0,
                })?;
            Ok(eval_shape(shape, *start, *end))
        }
        Op::Size => {
            let shape = input_shapes
                .first()
                .and_then(|s| *s)
                .or_else(|| inputs.first().and_then(|t| t.map(|c| c.shape.as_slice())))
                .ok_or(ShapeEvalError::MissingInput {
                    op: "Size",
                    index: 0,
                })?;
            Ok(eval_size(shape))
        }
        Op::Constant { value } => CpuTensor::from_tensor_proto(value),
        Op::ConstantOfShape { value } => eval_constant_of_shape(req_tensor(0)?, value),
        Op::Gather { axis } => eval_gather(req_tensor(0)?, req_tensor(1)?, *axis),
        Op::Slice => eval_slice(
            req_tensor(0)?,
            req_tensor(1)?,
            req_tensor(2)?,
            inputs.get(3).and_then(|o| *o),
            inputs.get(4).and_then(|o| *o),
        ),
        Op::Concat { axis } => {
            let mut list = Vec::with_capacity(inputs.len());
            for (i, inp) in inputs.iter().enumerate() {
                list.push(inp.ok_or(ShapeEvalError::MissingInput {
                    op: "Concat",
                    index: i,
                })?);
            }
            eval_concat(&list, *axis)
        }
        Op::Unsqueeze { axes } => {
            let data = req_tensor(0)?;
            let axes_vec: Vec<i64> = match axes {
                Some(a) => a.clone(),
                None => req_tensor(1)?.to_i64_vec(),
            };
            eval_unsqueeze(data, &axes_vec)
        }
        Op::Squeeze { axes } => {
            let data = req_tensor(0)?;
            let axes_opt: Option<Vec<i64>> = match axes {
                Some(a) => Some(a.clone()),
                None => inputs.get(1).and_then(|o| *o).map(|t| t.to_i64_vec()),
            };
            eval_squeeze(data, axes_opt.as_deref())
        }
        Op::Reshape { allowzero } => {
            let data = req_tensor(0)?;
            let shape_v = req_tensor(1)?.to_i64_vec();
            eval_reshape(data, &shape_v, *allowzero)
        }
        Op::Expand => {
            let data = req_tensor(0)?;
            let shape_v = req_tensor(1)?.to_i64_vec();
            eval_expand(data, &shape_v)
        }
        Op::Tile => {
            let data = req_tensor(0)?;
            let repeats = req_tensor(1)?.to_i64_vec();
            eval_tile(data, &repeats)
        }
        Op::Add => eval_bin_arith("Add", BinArith::Add, req_tensor(0)?, req_tensor(1)?),
        Op::Sub => eval_bin_arith("Sub", BinArith::Sub, req_tensor(0)?, req_tensor(1)?),
        Op::Mul => eval_bin_arith("Mul", BinArith::Mul, req_tensor(0)?, req_tensor(1)?),
        Op::Div => eval_bin_arith("Div", BinArith::Div, req_tensor(0)?, req_tensor(1)?),
        Op::Equal => eval_bin_cmp("Equal", BinCmp::Equal, req_tensor(0)?, req_tensor(1)?),
        Op::Less => eval_bin_cmp("Less", BinCmp::Less, req_tensor(0)?, req_tensor(1)?),
        Op::Greater => eval_bin_cmp("Greater", BinCmp::Greater, req_tensor(0)?, req_tensor(1)?),
        Op::GreaterOrEqual => {
            eval_bin_cmp("GreaterOrEqual", BinCmp::GreaterOrEqual, req_tensor(0)?, req_tensor(1)?)
        }
        Op::And => eval_and(req_tensor(0)?, req_tensor(1)?),
        Op::Where => eval_where(req_tensor(0)?, req_tensor(1)?, req_tensor(2)?),
        Op::Cast { to } => eval_cast(req_tensor(0)?, *to),
        Op::Identity => Ok(req_tensor(0)?.clone()),
        other => Err(ShapeEvalError::UnsupportedOp(other.name())),
    }
}

/// Returns `true` if the operator is supported by the CPU shape evaluator.
pub fn is_cpu_shape_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Shape { .. }
            | Op::Size
            | Op::Constant { .. }
            | Op::ConstantOfShape { .. }
            | Op::Gather { .. }
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
    )
}

// ============================================================================
// Graph-Level CPU Shape Subgraph Evaluator
// ============================================================================

/// Summary of nodes resolved on CPU during [`CpuShapeEvaluator::eval_graph`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShapeEvalStats {
    pub evaluated_nodes: HashSet<NodeId>,
    pub total_nodes: usize,
}

impl ShapeEvalStats {
    pub fn evaluated_count(&self) -> usize {
        self.evaluated_nodes.len()
    }
}

/// Stateful CPU shape/parameter evaluator for an [`IrGraph`].
///
/// Tracks known runtime tensor shapes and small CPU-resident tensors so that
/// `NativeCudaSession` can skip launching GPU kernels for shape subgraphs and
/// query resolved `Reshape` / `Slice` / `Resize` / `If` parameters directly.
#[derive(Clone, Debug, Default)]
pub struct CpuShapeEvaluator {
    tensors: HashMap<TensorId, CpuTensor>,
    shapes: HashMap<TensorId, Vec<usize>>,
    max_elems: usize,
}

impl CpuShapeEvaluator {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            shapes: HashMap::new(),
            max_elems: DEFAULT_MAX_CPU_TENSOR_ELEMS,
        }
    }

    /// Initialize from an [`IrGraph`] and runtime input shapes.
    ///
    /// Automatically seeds:
    /// - All statically known tensor shapes from `graph.tensors` (overridden by `runtime_shapes`);
    /// - Small initializers (`<= DEFAULT_MAX_CPU_TENSOR_ELEMS` elements, `I64`/`I32`/`F32`/`F16`/`Bool`/`U8`).
    pub fn from_graph(
        graph: &IrGraph,
        runtime_shapes: &HashMap<TensorId, Vec<usize>>,
    ) -> Self {
        let mut eval = Self::new();
        for info in &graph.tensors {
            if let Some(s) = runtime_shapes.get(&info.id) {
                eval.shapes.insert(info.id, s.clone());
            } else if let Some(static_s) = info.static_shape() {
                eval.shapes.insert(info.id, static_s);
            }
        }
        for (&tid, weight) in &graph.weights {
            if weight.elem_count() <= eval.max_elems {
                if let Ok(cpu_t) = CpuTensor::from_tensor_proto(weight) {
                    eval.shapes.insert(tid, cpu_t.shape.clone());
                    eval.tensors.insert(tid, cpu_t);
                }
            }
        }
        eval
    }

    pub fn with_max_elems(mut self, max_elems: usize) -> Self {
        self.max_elems = max_elems;
        self
    }

    pub fn set_tensor(&mut self, id: TensorId, tensor: CpuTensor) {
        self.shapes.insert(id, tensor.shape.clone());
        self.tensors.insert(id, tensor);
    }

    pub fn set_shape(&mut self, id: TensorId, shape: Vec<usize>) {
        self.shapes.insert(id, shape);
    }

    pub fn get_tensor(&self, id: TensorId) -> Option<&CpuTensor> {
        self.tensors.get(&id)
    }

    pub fn get_shape(&self, id: TensorId) -> Option<&[usize]> {
        self.shapes.get(&id).map(Vec::as_slice)
    }

    pub fn get_i64_vec(&self, id: TensorId) -> Option<Vec<i64>> {
        self.tensors.get(&id).map(CpuTensor::to_i64_vec)
    }

    pub fn get_f32_vec(&self, id: TensorId) -> Option<Vec<f32>> {
        self.tensors.get(&id).map(CpuTensor::to_f32_vec)
    }

    pub fn get_bool_scalar(&self, id: TensorId) -> Option<bool> {
        self.tensors.get(&id).and_then(CpuTensor::as_bool_scalar)
    }

    /// Attempt to evaluate a single [`IrNode`] on the CPU.
    ///
    /// Returns `Ok(true)` if the node was evaluated and its output stored,
    /// `Ok(false)` if the node is not a CPU-evaluable shape op or its inputs are
    /// GPU-only tensors, or `Err` if evaluation failed on available inputs.
    pub fn try_eval_node(&mut self, node: &IrNode) -> Result<bool, ShapeEvalError> {
        if !is_cpu_shape_op(&node.op) {
            return Ok(false);
        }
        let Some(Some(out_id)) = node.outputs.first() else {
            return Ok(false);
        };

        // Check whether required inputs are available in CPU cache / shapes map.
        match &node.op {
            Op::Shape { .. } | Op::Size => {
                let Some(Some(in0)) = node.inputs.first() else {
                    return Ok(false);
                };
                if !self.shapes.contains_key(in0) && !self.tensors.contains_key(in0) {
                    return Ok(false);
                }
            }
            Op::Constant { value } => {
                if value.elem_count() > self.max_elems {
                    return Ok(false);
                }
            }
            _ => {
                for inp in &node.inputs {
                    if let Some(tid) = inp {
                        if !self.tensors.contains_key(tid) {
                            return Ok(false);
                        }
                    }
                }
            }
        }

        let input_tensors: Vec<Option<&CpuTensor>> = node
            .inputs
            .iter()
            .map(|opt| opt.and_then(|tid| self.tensors.get(&tid)))
            .collect();
        let input_shapes: Vec<Option<&[usize]>> = node
            .inputs
            .iter()
            .map(|opt| opt.and_then(|tid| self.shapes.get(&tid).map(Vec::as_slice)))
            .collect();

        let result = eval_op(&node.op, &input_tensors, &input_shapes)?;
        if result.len() > self.max_elems {
            // Keep shape metadata even if payload exceeds CPU mirror cap.
            self.shapes.insert(*out_id, result.shape);
            return Ok(false);
        }
        self.set_tensor(*out_id, result);
        Ok(true)
    }

    /// Walk the graph in topological order and evaluate all resolvable shape/parameter nodes on CPU.
    pub fn eval_graph(&mut self, graph: &IrGraph) -> ShapeEvalStats {
        let mut stats = ShapeEvalStats {
            evaluated_nodes: HashSet::new(),
            total_nodes: graph.nodes.len(),
        };
        for node in &graph.nodes {
            if let Ok(true) = self.try_eval_node(node) {
                stats.evaluated_nodes.insert(node.id);
            }
        }
        stats
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_infer::graph::{IrGraph, IrNode, NodeId, Op, TensorInfo};
    use crate::native_infer::onnx_proto::TensorProto;

    #[test]
    fn shape_size_constant_and_constant_of_shape() {
        let s = eval_shape(&[2, 3, 64, 64], None, None);
        assert_eq!(s.shape, vec![4]);
        assert_eq!(s.as_i64(), Some(&[2, 3, 64, 64][..]));

        let s_sub = eval_shape(&[2, 3, 64, 64], Some(1), Some(-1));
        assert_eq!(s_sub.as_i64(), Some(&[3, 64][..]));

        let sz = eval_size(&[2, 3, 4]);
        assert_eq!(sz.as_i64_scalar(), Some(24));

        let c = CpuTensor::from_tensor_proto(&TensorProto::from_i64("c", vec![2], &[3, 4])).unwrap();
        let cos = eval_constant_of_shape(
            &c,
            &TensorProto::from_i64("v", vec![1], &[7]),
        )
        .unwrap();
        assert_eq!(cos.shape, vec![3, 4]);
        assert_eq!(cos.as_i64(), Some(&[7; 12][..]));
    }

    #[test]
    fn gather_unsqueeze_squeeze_concat_reshape() {
        let shape_t = CpuTensor::from_i64(vec![4], vec![1, 256, 64, 64]).unwrap();
        let idx0 = CpuTensor::scalar_i64(0);
        let idx_last = CpuTensor::scalar_i64(-1);

        let g0 = eval_gather(&shape_t, &idx0, 0).unwrap();
        let g3 = eval_gather(&shape_t, &idx_last, 0).unwrap();
        assert_eq!(g0.shape, Vec::<usize>::new());
        assert_eq!(g0.as_i64_scalar(), Some(1));
        assert_eq!(g3.as_i64_scalar(), Some(64));

        let u0 = eval_unsqueeze(&g0, &[0]).unwrap();
        let u3 = eval_unsqueeze(&g3, &[0]).unwrap();
        let neg1 = CpuTensor::from_i64(vec![1], vec![-1]).unwrap();
        let cat = eval_concat(&[&u0, &neg1, &u3], 0).unwrap();
        assert_eq!(cat.shape, vec![3]);
        assert_eq!(cat.as_i64(), Some(&[1, -1, 64][..]));

        let unsq2 = eval_unsqueeze(&cat, &[0, -1]).unwrap();
        assert_eq!(unsq2.shape, vec![1, 3, 1]);
        let sq = eval_squeeze(&unsq2, None).unwrap();
        assert_eq!(sq.shape, vec![3]);
        assert_eq!(sq.as_i64(), Some(&[1, -1, 64][..]));

        let data = CpuTensor::from_i64(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
        let reshaped = eval_reshape(&data, &[0, -1], false).unwrap();
        assert_eq!(reshaped.shape, vec![2, 3]);
        let reshaped2 = eval_reshape(&data, &[3, 2], false).unwrap();
        assert_eq!(reshaped2.shape, vec![3, 2]);
        assert_eq!(reshaped2.as_i64(), Some(&[1, 2, 3, 4, 5, 6][..]));
    }

    #[test]
    fn slice_with_negative_indices_and_steps() {
        let t = CpuTensor::from_i64(vec![2, 5], vec![0, 1, 2, 3, 4, 10, 11, 12, 13, 14]).unwrap();
        let starts = CpuTensor::from_i64(vec![1], vec![1]).unwrap();
        let ends = CpuTensor::from_i64(vec![1], vec![-1]).unwrap();
        let axes = CpuTensor::from_i64(vec![1], vec![1]).unwrap();
        let steps = CpuTensor::from_i64(vec![1], vec![2]).unwrap();
        let out = eval_slice(&t, &starts, &ends, Some(&axes), Some(&steps)).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.as_i64(), Some(&[1, 3, 11, 13][..]));

        // Reverse slice
        let r_starts = CpuTensor::from_i64(vec![1], vec![4]).unwrap();
        let r_ends = CpuTensor::from_i64(vec![1], vec![0]).unwrap();
        let r_steps = CpuTensor::from_i64(vec![1], vec![-2]).unwrap();
        let rev = eval_slice(&t, &r_starts, &r_ends, Some(&axes), Some(&r_steps)).unwrap();
        assert_eq!(rev.shape, vec![2, 2]);
        assert_eq!(rev.as_i64(), Some(&[4, 2, 14, 12][..]));
    }

    #[test]
    fn broadcast_arith_compare_where_and_cast() {
        let a = CpuTensor::from_i64(vec![2, 1], vec![10, -7]).unwrap();
        let b = CpuTensor::from_i64(vec![1, 3], vec![2, 3, 5]).unwrap();

        let add = eval_op(&Op::Add, &[Some(&a), Some(&b)], &[]).unwrap();
        assert_eq!(add.shape, vec![2, 3]);
        assert_eq!(add.as_i64(), Some(&[12, 13, 15, -5, -4, -2][..]));

        let sub = eval_op(&Op::Sub, &[Some(&a), Some(&b)], &[]).unwrap();
        assert_eq!(sub.as_i64(), Some(&[8, 7, 5, -9, -10, -12][..]));

        let mul = eval_op(&Op::Mul, &[Some(&a), Some(&b)], &[]).unwrap();
        assert_eq!(mul.as_i64(), Some(&[20, 30, 50, -14, -21, -35][..]));

        let div = eval_op(&Op::Div, &[Some(&a), Some(&b)], &[]).unwrap();
        assert_eq!(div.as_i64(), Some(&[5, 3, 2, -3, -2, -1][..]));

        let eq = eval_op(&Op::Equal, &[Some(&div), Some(&CpuTensor::scalar_i64(2))], &[]).unwrap();
        assert_eq!(eq.as_bool(), Some(&[false, false, true, false, false, false][..]));

        let lt = eval_op(&Op::Less, &[Some(&a), Some(&CpuTensor::scalar_i64(0))], &[]).unwrap();
        assert_eq!(lt.as_bool(), Some(&[false, true][..]));

        let gt = eval_op(&Op::Greater, &[Some(&b), Some(&CpuTensor::scalar_i64(2))], &[]).unwrap();
        assert_eq!(gt.as_bool(), Some(&[false, true, true][..]));

        let wh = eval_where(&lt, &CpuTensor::scalar_i64(99), &b).unwrap();
        assert_eq!(wh.shape, vec![2, 3]);
        assert_eq!(wh.as_i64(), Some(&[2, 3, 5, 99, 99, 99][..]));

        let cf = eval_cast(&wh, DataType::Float).unwrap();
        assert_eq!(cf.as_f32(), Some(&[2.0, 3.0, 5.0, 99.0, 99.0, 99.0][..]));
    }

    #[test]
    fn graph_level_shape_subgraph_evaluator() {
        // Synthetic dynamic Reshape target subgraph:
        // x:[4, 3, 64, 64] -> Shape -> Gather(0) -> Unsqueeze(0) -> Concat([batch, 192, 64])
        let mut graph = IrGraph {
            name: "shape_subgraph_test".into(),
            tensors: Vec::new(),
            nodes: Vec::new(),
            inputs: vec![TensorId(0)],
            outputs: vec![TensorId(6)],
            weights: HashMap::new(),
            name_to_id: HashMap::new(),
        };
        let mk_info = |id: usize, name: &str, dtype: DataType, is_in: bool| TensorInfo {
            id: TensorId(id),
            name: name.into(),
            dtype,
            shape: None,
            is_weight: false,
            is_graph_input: is_in,
            is_graph_output: id == 6,
        };
        graph.tensors = vec![
            mk_info(0, "x", DataType::Float, true),
            mk_info(1, "x_shape", DataType::Int64, false),
            mk_info(2, "c_zero", DataType::Int64, false),
            mk_info(3, "batch_scalar", DataType::Int64, false),
            mk_info(4, "batch_1d", DataType::Int64, false),
            mk_info(5, "c_tail", DataType::Int64, false),
            mk_info(6, "target_shape", DataType::Int64, false),
        ];
        graph.nodes = vec![
            IrNode {
                id: NodeId(0),
                name: "n_shape".into(),
                op: Op::Shape { start: None, end: None },
                inputs: vec![Some(TensorId(0))],
                outputs: vec![Some(TensorId(1))],
            },
            IrNode {
                id: NodeId(1),
                name: "n_c0".into(),
                op: Op::Constant {
                    value: TensorProto::from_i64("c0", vec![], &[0]),
                },
                inputs: vec![],
                outputs: vec![Some(TensorId(2))],
            },
            IrNode {
                id: NodeId(2),
                name: "n_gather".into(),
                op: Op::Gather { axis: 0 },
                inputs: vec![Some(TensorId(1)), Some(TensorId(2))],
                outputs: vec![Some(TensorId(3))],
            },
            IrNode {
                id: NodeId(3),
                name: "n_unsq".into(),
                op: Op::Unsqueeze { axes: Some(vec![0]) },
                inputs: vec![Some(TensorId(3))],
                outputs: vec![Some(TensorId(4))],
            },
            IrNode {
                id: NodeId(4),
                name: "n_ctail".into(),
                op: Op::Constant {
                    value: TensorProto::from_i64("ctail", vec![2], &[192, 64]),
                },
                inputs: vec![],
                outputs: vec![Some(TensorId(5))],
            },
            IrNode {
                id: NodeId(5),
                name: "n_concat".into(),
                op: Op::Concat { axis: 0 },
                inputs: vec![Some(TensorId(4)), Some(TensorId(5))],
                outputs: vec![Some(TensorId(6))],
            },
        ];

        let mut rt_shapes = HashMap::new();
        rt_shapes.insert(TensorId(0), vec![4, 3, 64, 64]);

        let mut eval = CpuShapeEvaluator::from_graph(&graph, &rt_shapes);
        let stats = eval.eval_graph(&graph);
        assert_eq!(stats.evaluated_count(), 6);
        assert_eq!(eval.get_i64_vec(TensorId(6)), Some(vec![4, 192, 64]));
    }
}
