//! Portable, bounded F32 executor for small static graphs. Unsupported whole graphs are
//! explicitly rejected at open so a caller can use its optional full CPU reference runtime.
use super::{
    backend::*,
    cpu_shape_eval::{CpuData, CpuTensor, eval_op},
    graph::{IrGraph, Op},
    load_and_plan_portable as load_and_plan,
    onnx_proto::DataType,
};
use std::sync::Arc;
const MAX_LIVE_ESTIMATE: usize = 128 << 20;

pub struct NativeCpuSession {
    graph: IrGraph,
    weights: Vec<(usize, Arc<CpuTensor>)>,
    last: Vec<usize>,
    peak_estimate: usize,
}
fn product(s: &[usize]) -> Option<usize> {
    s.iter().try_fold(1usize, |a, &b| a.checked_mul(b))
}
fn unary(op: &Op) -> bool {
    matches!(
        op,
        Op::Identity
            | Op::Relu
            | Op::LeakyRelu { .. }
            | Op::Abs
            | Op::Sqrt
            | Op::Reciprocal
            | Op::Sigmoid
            | Op::SiLU
            | Op::HardSigmoid { .. }
            | Op::HardSwish
            | Op::Sin
            | Op::Cos
    )
}
fn value(op: &Op, x: f32) -> f32 {
    match op {
        Op::Identity => x,
        Op::Relu => {
            if x < 0.0 {
                0.0
            } else {
                x
            }
        }
        Op::LeakyRelu { alpha } => {
            if x < 0.0 {
                x * alpha
            } else {
                x
            }
        }
        Op::Abs => x.abs(),
        Op::Sqrt => x.sqrt(),
        Op::Reciprocal => 1.0 / x,
        Op::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Op::SiLU => x / (1.0 + (-x).exp()),
        Op::HardSigmoid { alpha, beta } => (alpha * x + beta).clamp(0., 1.),
        Op::HardSwish => x * (x / 6. + 0.5).clamp(0., 1.),
        Op::Sin => x.sin(),
        Op::Cos => x.cos(),
        _ => unreachable!(),
    }
}
impl NativeCpuSession {
    pub fn from_bytes(bytes: &[u8], o: &ShapeOverrides) -> Result<Self, String> {
        let graph = load_and_plan(bytes, o).map_err(|e| e.to_string())?.graph;
        let mut estimate = 0usize;
        let mut last = vec![0; graph.tensors.len()];
        for t in &graph.tensors {
            if t.dtype != DataType::Float {
                return Err(format!(
                    "CPU subset requires F32: {} is {:?}",
                    t.name, t.dtype
                ));
            }
            let n = t
                .static_byte_size()
                .ok_or_else(|| format!("CPU needs static bound for {}", t.name))?;
            estimate = estimate
                .checked_add(n)
                .filter(|&v| v <= MAX_LIVE_ESTIMATE)
                .ok_or("CPU static graph exceeds allocation budget")?;
        }
        for (i, node) in graph.nodes.iter().enumerate() {
            if node.outputs.len() != 1 || node.outputs[0].is_none() {
                return Err(format!("CPU: {} needs exactly one output", node.name));
            }
            let count = if unary(&node.op) {
                1
            } else if matches!(node.op, Op::Add | Op::Sub | Op::Mul | Op::Div | Op::MatMul) {
                2
            } else {
                return Err(format!(
                    "CPU subset does not execute {} ({})",
                    node.op.name(),
                    node.name
                ));
            };
            if node.inputs.len() != count || node.inputs.iter().any(Option::is_none) {
                return Err(format!("invalid {} input arity", node.name));
            }
            for id in node.inputs.iter().flatten() {
                last[id.0] = i;
            }
            if matches!(node.op, Op::MatMul) {
                let dims = |j: usize| {
                    graph.tensors[node.inputs[j].unwrap().0]
                        .static_shape()
                        .unwrap()
                };
                let (a, b) = (dims(0), dims(1));
                if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
                    return Err("CPU MatMul requires compatible rank-two tensors".into());
                }
            }
        }
        for t in &graph.outputs {
            last[t.0] = usize::MAX;
        }
        let mut weights = Vec::new();
        for (id, w) in &graph.weights {
            weights.push((
                id.0,
                Arc::new(CpuTensor::from_tensor_proto(w).map_err(|e| e.to_string())?),
            ));
        }
        Ok(Self {
            graph,
            weights,
            last,
            peak_estimate: estimate,
        })
    }
}
impl NativeInferBackend for NativeCpuSession {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }
    // Conservative graph-wide storage bound, not a measured RSS or device allocation.
    fn arena_peak_bytes(&self) -> usize {
        self.peak_estimate
    }
    fn run_f32(&self, inputs: &NamedInputs<'_>) -> Result<NamedOutputs, String> {
        let g = &self.graph;
        if inputs.len() != g.inputs.len() {
            return Err("CPU input count mismatch".into());
        }
        let mut slots: Vec<Option<Arc<CpuTensor>>> = vec![None; g.tensors.len()];
        for (id, w) in &self.weights {
            slots[*id] = Some(Arc::clone(w));
        }
        for t in &g.inputs {
            let spec = &g.tensors[t.0];
            let (v, s) = inputs
                .get(&spec.name)
                .ok_or_else(|| format!("missing input {}", spec.name))?;
            if Some(v.len()) != product(s) || spec.static_shape().as_ref() != Some(s) {
                return Err(format!("input {} shape mismatch", spec.name));
            }
            slots[t.0] = Some(Arc::new(CpuTensor {
                shape: s.clone(),
                data: CpuData::F32(v.to_vec()),
            }));
        }
        for (ix, node) in g.nodes.iter().enumerate() {
            let args = node
                .inputs
                .iter()
                .map(|id| {
                    slots[id.unwrap().0]
                        .as_deref()
                        .ok_or("CPU: unavailable input slot")
                })
                .collect::<Result<Vec<_>, _>>()?;
            let x = args[0];
            let out = if matches!(node.op, Op::Identity) {
                Arc::clone(slots[node.inputs[0].unwrap().0].as_ref().unwrap())
            } else if unary(&node.op) {
                Arc::new(CpuTensor {
                    shape: x.shape.clone(),
                    data: CpuData::F32(
                        x.as_f32()
                            .ok_or("expected Float")?
                            .iter()
                            .map(|&v| value(&node.op, v))
                            .collect(),
                    ),
                })
            } else if matches!(node.op, Op::MatMul) {
                let y = args[1];
                let (m, k, n) = (x.shape[0], x.shape[1], y.shape[1]);
                let (a, b) = (
                    x.as_f32().ok_or("expected matrix")?,
                    y.as_f32().ok_or("expected matrix")?,
                );
                let mut out = vec![0f32; m.checked_mul(n).ok_or("matrix overflow")?];
                if n > 0 {
                    let threads = if m.saturating_mul(n).saturating_mul(k) >= 131072 {
                        std::thread::available_parallelism()
                            .map_or(1, usize::from)
                            .min(8)
                    } else {
                        1
                    };
                    let rows = m.div_ceil(threads).max(1);
                    let calc = |start: usize, d: &mut [f32]| {
                        for (i, row) in d.chunks_mut(n).enumerate() {
                            for l in 0..k {
                                let coeff = a[(start + i) * k + l];
                                for (u, &z) in row.iter_mut().zip(&b[l * n..(l + 1) * n]) {
                                    *u += coeff * z;
                                }
                            }
                        }
                    };
                    if threads == 1 {
                        calc(0, &mut out)
                    } else {
                        std::thread::scope(|s| {
                            for (i, part) in out.chunks_mut(rows * n).enumerate() {
                                let f = &calc;
                                s.spawn(move || f(i * rows, part));
                            }
                        });
                    }
                }
                Arc::new(CpuTensor {
                    shape: vec![m, n],
                    data: CpuData::F32(out),
                })
            } else {
                let a: Vec<_> = args.iter().map(|&v| Some(v)).collect();
                let sh: Vec<_> = args.iter().map(|v| Some(v.shape.as_slice())).collect();
                Arc::new(eval_op(&node.op, &a, &sh).map_err(|e| e.to_string())?)
            };
            let id = node.outputs[0].unwrap();
            if g.tensors[id.0].static_shape().as_ref() != Some(&out.shape) {
                return Err(format!("{} returned the wrong shape", node.name));
            }
            slots[id.0] = Some(out);
            for inp in node.inputs.iter().flatten() {
                if self.last[inp.0] == ix {
                    slots[inp.0] = None;
                }
            }
        }
        g.outputs
            .iter()
            .map(|id| {
                let x = slots[id.0].as_ref().ok_or("missing model output")?;
                Ok((
                    g.tensors[id.0].name.clone(),
                    (
                        x.as_f32().ok_or("expected F32 output")?.to_vec(),
                        x.shape.clone(),
                    ),
                ))
            })
            .collect()
    }
}
