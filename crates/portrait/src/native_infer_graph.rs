//! Platform-independent computation graph IR, compile passes and FP32 CPU
//! reference executor for a zero-ORT `native_infer` path.
//!
//! The graph is backend-neutral: tensors are plain byte buffers plus a shape
//! and dtype descriptor, operators carry their attributes inline, and
//! constants live in a side table. A frontend lowers a model into this IR once;
//! CPU / CUDA / Metal backends can schedule the same graph.
//!
//! This module implements:
//!   * topological scheduling with explicit cycle/unknown-node errors,
//!   * shape inference for every operator,
//!   * a constant-fold pass for `Conv2d -> BatchNorm`,
//!   * a lifetime-aware memory slot planner with reuse accounting, and
//!   * a complete FP32 CPU reference executor for all declared operators.
//!
//! It is pure Rust with no external dependencies. The executor is a
//! correctness oracle, not a performance claim: tensors are NCHW and
//! convolution uses straightforward direct loops. Fp16 buffers are modelled in
//! the IR, but reference kernels require Fp32 (quantized kernels use Int8).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced while validating, compiling or executing a graph.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphError {
    /// An operator references an input that is neither a node output, a
    /// constant nor a declared runtime input.
    UnknownInput { node: usize, input: usize },
    /// The graph contains a cycle, so no topological order exists.
    Cycle { nodes: Vec<usize> },
    /// A tensor shape disagrees with an operator's expectations.
    ShapeMismatch(String),
    /// A tensor's dtype or buffer size disagrees with its descriptor.
    TensorMismatch(String),
    /// An unsupported dtype or attribute combination.
    Unsupported(String),
    /// The requested output is not produced by the graph.
    UnknownOutput(usize),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownInput { node, input } => {
                write!(f, "node {node} references unknown input tensor {input}")
            }
            Self::Cycle { nodes } => write!(f, "graph contains a cycle: {nodes:?}"),
            Self::ShapeMismatch(msg) => write!(f, "shape mismatch: {msg}"),
            Self::TensorMismatch(msg) => write!(f, "tensor mismatch: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported operation: {msg}"),
            Self::UnknownOutput(id) => write!(f, "unknown graph output tensor {id}"),
        }
    }
}

impl std::error::Error for GraphError {}

/// Convenience alias used throughout this module.
pub type GraphResult<T> = Result<T, GraphError>;

// ---------------------------------------------------------------------------
// Tensor IR
// ---------------------------------------------------------------------------

/// Element type of a tensor. Quantization is explicit via quantize/dequantize
/// operators rather than an implicit per-tensor flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    Fp32,
    Fp16,
    Int8,
}

impl DType {
    /// Bytes per element in a backend buffer.
    pub fn byte_width(self) -> usize {
        match self {
            Self::Fp32 => 4,
            Self::Fp16 => 2,
            Self::Int8 => 1,
        }
    }
}

/// NCHW tensor shape. Flattened tensors use `TensorShape::vector`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorShape {
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl TensorShape {
    pub fn new(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self { n, c, h, w }
    }

    /// Flattened vector stored in the `c` dimension (n=h=w=1).
    pub fn vector(len: usize) -> Self {
        Self::new(1, len, 1, 1)
    }

    pub fn numel(&self) -> usize {
        self.n * self.c * self.h * self.w
    }

    pub fn byte_len(&self, dtype: DType) -> usize {
        self.numel() * dtype.byte_width()
    }

    /// Map `(n, c, h, w)` to a linear index, with bounds checking.
    pub fn index(&self, n: usize, c: usize, h: usize, w: usize) -> GraphResult<usize> {
        if n >= self.n || c >= self.c || h >= self.h || w >= self.w {
            return Err(GraphError::ShapeMismatch(format!(
                "index ({n},{c},{h},{w}) out of bounds for {self:?}"
            )));
        }
        Ok(((n * self.c + c) * self.h + h) * self.w + w)
    }
}

/// Owned tensor payload. Multi-byte values are stored little-endian so a
/// backend can copy buffers verbatim; the CPU executor decodes via
/// `f32_values` / `set_f32_values` (no unaligned pointer casts).
#[derive(Clone, Debug, PartialEq)]
pub struct TensorBuffer {
    pub dtype: DType,
    pub shape: TensorShape,
    pub bytes: Vec<u8>,
}

impl TensorBuffer {
    /// Allocate a zero-filled buffer for a shape/dtype pair.
    pub fn zeros(dtype: DType, shape: TensorShape) -> Self {
        Self {
            dtype,
            shape,
            bytes: vec![0u8; shape.byte_len(dtype)],
        }
    }

    /// Wrap existing bytes, validating their length.
    pub fn from_bytes(dtype: DType, shape: TensorShape, bytes: Vec<u8>) -> GraphResult<Self> {
        if bytes.len() != shape.byte_len(dtype) {
            return Err(GraphError::TensorMismatch(format!(
                "buffer has {} bytes, {shape:?}/{dtype:?} needs {}",
                bytes.len(),
                shape.byte_len(dtype)
            )));
        }
        Ok(Self {
            dtype,
            shape,
            bytes,
        })
    }

    /// Build an FP32 tensor from a row-major value slice.
    pub fn from_f32(shape: TensorShape, values: &[f32]) -> GraphResult<Self> {
        if values.len() != shape.numel() {
            return Err(GraphError::TensorMismatch(format!(
                "{} values for {shape:?} (need {})",
                values.len(),
                shape.numel()
            )));
        }
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self::from_bytes(DType::Fp32, shape, bytes)
    }

    /// Decode the payload as FP32 values. Errors for other dtypes.
    pub fn f32_values(&self) -> GraphResult<Vec<f32>> {
        if self.dtype != DType::Fp32 {
            return Err(GraphError::TensorMismatch(format!(
                "expected Fp32 tensor, got {:?}",
                self.dtype
            )));
        }
        if self.bytes.len() % 4 != 0 {
            return Err(GraphError::TensorMismatch(
                "FP32 buffer length is not a multiple of 4".to_string(),
            ));
        }
        Ok(self
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Overwrite the payload with FP32 values.
    pub fn set_f32_values(&mut self, values: &[f32]) -> GraphResult<()> {
        if self.dtype != DType::Fp32 {
            return Err(GraphError::TensorMismatch(format!(
                "expected Fp32 tensor, got {:?}",
                self.dtype
            )));
        }
        if values.len() * 4 != self.bytes.len() {
            return Err(GraphError::TensorMismatch(format!(
                "{} FP32 values do not fit a {} byte buffer",
                values.len(),
                self.bytes.len()
            )));
        }
        for (chunk, value) in self.bytes.chunks_exact_mut(4).zip(values) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Operator IR
// ---------------------------------------------------------------------------

/// Padding `(top, bottom, left, right)` in elements.
pub type Pad4 = (usize, usize, usize, usize);

/// Attributes for one operator. Constants are referenced by tensor id and
/// kept in `Graph::constants`, so a fold pass can rewrite them without
/// invalidating node identities.
#[derive(Clone, Debug, PartialEq)]
pub enum OpKind {
    /// `out[n,oc,oh,ow] = bias[oc] + sum w[oc,ic,kh,kw] * x[..]`
    Conv2d {
        stride: (usize, usize),
        pad: Pad4,
        dilation: (usize, usize),
        weight: usize,
        bias: Option<usize>,
    },
    /// Per-channel spatial convolution, weight shape `(c, 1, kh, kw)`.
    DepthwiseConv2d {
        stride: (usize, usize),
        pad: Pad4,
        weight: usize,
        bias: Option<usize>,
    },
    /// Per-channel affine normalization with running statistics.
    BatchNorm {
        scale: usize,
        offset: usize,
        mean: usize,
        var: usize,
        epsilon: f32,
    },
    /// Normalize over (C, H, W) per sample with learnable scale/offset.
    GroupNorm {
        groups: usize,
        scale: usize,
        offset: usize,
        epsilon: f32,
    },
    Relu,
    /// `x * sigmoid(x)`.
    SiLU,
    Sigmoid,
    /// Broadcasted elementwise addition (`y = x + b`).
    Add,
    /// Broadcasted elementwise multiplication (`y = x * m`).
    Mul,
    /// Bilinear resampling with explicit output scale factors.
    ResizeBilinear {
        scale_h: f32,
        scale_w: f32,
        align_corners: bool,
    },
    /// Differentiable bilinear grid sampling, normalized coordinates in
    /// `[-1, 1]` (ONNX `GridSample` convention).
    GridSample,
    /// `out[n,i,j] = sum_k x[n,i,k] * w[n,j,k]`.
    MatMul,
    /// Scale + zero-point quantization to Int8.
    QuantizeLinear { scale: f32, zero_point: i8 },
    /// Inverse of `QuantizeLinear`, producing Fp32.
    DequantizeLinear { scale: f32, zero_point: i8 },
}

impl OpKind {
    /// Weight tensor id for convolution nodes (used by fold tests).
    pub fn weight_id(&self) -> usize {
        match self {
            Self::Conv2d { weight, .. } | Self::DepthwiseConv2d { weight, .. } => *weight,
            _ => panic!("{} has no weight tensor", self.name()),
        }
    }

    /// Bias tensor id for convolution nodes (used by fold tests).
    pub fn bias_id(&self) -> Option<usize> {
        match self {
            Self::Conv2d { bias, .. } | Self::DepthwiseConv2d { bias, .. } => *bias,
            _ => panic!("{} has no bias tensor", self.name()),
        }
    }

    /// Name used in error messages and graph dumps.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Conv2d { .. } => "Conv2d",
            Self::DepthwiseConv2d { .. } => "DepthwiseConv2d",
            Self::BatchNorm { .. } => "BatchNorm",
            Self::GroupNorm { .. } => "GroupNorm",
            Self::Relu => "Relu",
            Self::SiLU => "SiLU",
            Self::Sigmoid => "Sigmoid",
            Self::Add => "Add",
            Self::Mul => "Mul",
            Self::ResizeBilinear { .. } => "ResizeBilinear",
            Self::GridSample => "GridSample",
            Self::MatMul => "MatMul",
            Self::QuantizeLinear { .. } => "QuantizeLinear",
            Self::DequantizeLinear { .. } => "DequantizeLinear",
        }
    }
}

/// One graph instruction. At most two data inputs are used by any operator.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub op: OpKind,
    /// Input tensor ids: node outputs, constants or runtime inputs.
    pub inputs: Vec<usize>,
    /// Output tensor id, unique within the graph.
    pub output: usize,
    /// Human-readable label used in error messages and dumps.
    pub name: String,
}

impl Node {
    pub fn new(op: OpKind, inputs: Vec<usize>, output: usize, name: &str) -> Self {
        Self {
            op,
            inputs,
            output,
            name: name.to_string(),
        }
    }
}

/// Platform-independent computation graph.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub constants: HashMap<usize, TensorBuffer>,
    /// Tensor ids supplied by the caller at execution time.
    pub runtime_inputs: HashSet<usize>,
    pub outputs: Vec<usize>,
    /// Monotonic counter used by builder helpers.
    next_tensor: usize,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve and return a fresh tensor id.
    pub fn fresh_id(&mut self) -> usize {
        let id = self.next_tensor;
        self.next_tensor += 1;
        id
    }

    /// Register a constant tensor, returning its id.
    pub fn add_constant(&mut self, tensor: TensorBuffer) -> usize {
        let id = self.fresh_id();
        self.constants.insert(id, tensor);
        id
    }

    /// Declare a tensor id as a caller-supplied runtime input.
    pub fn declare_runtime_input(&mut self, id: usize) {
        self.runtime_inputs.insert(id);
    }

    /// Append a node and return its index in `nodes`.
    pub fn add_node(&mut self, node: Node) -> usize {
        let index = self.nodes.len();
        self.nodes.push(node);
        index
    }

    /// Declare graph outputs, marking them as live for memory planning.
    pub fn set_outputs(&mut self, outputs: Vec<usize>) {
        self.outputs = outputs;
    }

    /// Validate inputs and return a topological order of node indices.
    pub fn topo_order(&self) -> GraphResult<Vec<usize>> {
        self.topo_order_with_known(&HashSet::new())
    }

    /// Internal variant that also accepts runtime input ids that were not
    /// explicitly declared (used by `execute_graph`).
    fn topo_order_with_known(&self, extra: &HashSet<usize>) -> GraphResult<Vec<usize>> {
        let mut producers: HashMap<usize, usize> = HashMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if producers.insert(node.output, i).is_some() {
                return Err(GraphError::ShapeMismatch(format!(
                    "duplicate tensor id {} produced by more than one node",
                    node.output
                )));
            }
        }
        for (i, node) in self.nodes.iter().enumerate() {
            for &input in &node.inputs {
                let known = self.constants.contains_key(&input)
                    || self.runtime_inputs.contains(&input)
                    || extra.contains(&input)
                    || producers.contains_key(&input);
                if !known {
                    return Err(GraphError::UnknownInput { node: i, input });
                }
            }
        }

        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        let mut indegree = vec![0usize; self.nodes.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            for &input in &node.inputs {
                if self.constants.contains_key(&input)
                    || self.runtime_inputs.contains(&input)
                    || extra.contains(&input)
                {
                    continue;
                }
                let producer = producers[&input];
                successors[producer].push(i);
                indegree[i] += 1;
            }
        }

        let mut queue: VecDeque<usize> = (0..self.nodes.len())
            .filter(|&i| indegree[i] == 0)
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(i) = queue.pop_front() {
            order.push(i);
            for &succ in &successors[i] {
                indegree[succ] -= 1;
                if indegree[succ] == 0 {
                    queue.push_back(succ);
                }
            }
        }

        if order.len() != self.nodes.len() {
            let cyclic: Vec<usize> = (0..self.nodes.len())
                .filter(|&i| indegree[i] > 0)
                .map(|i| self.nodes[i].output)
                .collect();
            return Err(GraphError::Cycle { nodes: cyclic });
        }
        Ok(order)
    }

    /// Infer the shape of every tensor produced by the graph. Constants must
    /// already carry shapes.
    pub fn infer_shapes(&self) -> GraphResult<HashMap<usize, TensorShape>> {
        self.infer_shapes_with(&HashMap::new())
    }

    /// Shape inference with caller-supplied shapes for runtime inputs.
    pub fn infer_shapes_with(
        &self,
        extra: &HashMap<usize, TensorShape>,
    ) -> GraphResult<HashMap<usize, TensorShape>> {
        let known: HashSet<usize> = extra.keys().copied().collect();
        let order = self.topo_order_with_known(&known)?;
        let mut shapes: HashMap<usize, TensorShape> = self
            .constants
            .iter()
            .map(|(&id, t)| (id, t.shape))
            .collect();
        shapes.extend(extra.iter().map(|(&id, &shape)| (id, shape)));

        for &i in &order {
            let node = &self.nodes[i];
            let input_shape = |k: usize| -> GraphResult<TensorShape> {
                let id = *node.inputs.get(k).ok_or_else(|| {
                    GraphError::ShapeMismatch(format!(
                        "{} node {} needs at least {} inputs",
                        node.op.name(),
                        node.output,
                        k + 1
                    ))
                })?;
                shapes
                    .get(&id)
                    .copied()
                    .ok_or_else(|| GraphError::ShapeMismatch(format!("unresolved input {id}")))
            };
            let param_shape = |id: usize, what: &str| -> GraphResult<TensorShape> {
                shapes.get(&id).copied().ok_or_else(|| {
                    GraphError::ShapeMismatch(format!("unknown {what} tensor {id}"))
                })
            };
            let shape = match &node.op {
                OpKind::Conv2d {
                    stride, pad, weight, ..
                }
                | OpKind::DepthwiseConv2d {
                    stride, pad, weight, ..
                } => {
                    let x = input_shape(0)?;
                    let w = param_shape(*weight, "conv weight")?;
                    let depthwise = matches!(node.op, OpKind::DepthwiseConv2d { .. });
                    if depthwise {
                        if w.n != x.c || w.c != 1 {
                            return Err(GraphError::ShapeMismatch(format!(
                                "depthwise weight {w:?} does not match input {x:?}"
                            )));
                        }
                    } else if w.c != x.c {
                        return Err(GraphError::ShapeMismatch(format!(
                            "conv weight {w:?} does not match input channels {}",
                            x.c
                        )));
                    }
                    let dil = match &node.op {
                        OpKind::Conv2d { dilation, .. } => *dilation,
                        _ => (1, 1),
                    };
                    let kh = (w.h - 1) * dil.0 + 1;
                    let kw = (w.w - 1) * dil.1 + 1;
                    if stride.0 == 0 || stride.1 == 0 {
                        return Err(GraphError::ShapeMismatch("conv stride must be positive".into()));
                    }
                    let oh = (x.h + pad.0 + pad.1).checked_sub(kh).ok_or_else(|| {
                        GraphError::ShapeMismatch("conv kernel exceeds padded height".into())
                    })? / stride.0 + 1;
                    let ow = (x.w + pad.2 + pad.3).checked_sub(kw).ok_or_else(|| {
                        GraphError::ShapeMismatch("conv kernel exceeds padded width".into())
                    })? / stride.1 + 1;
                    TensorShape::new(x.n, w.n, oh, ow)
                }
                OpKind::BatchNorm { scale, .. } => {
                    let x = input_shape(0)?;
                    let s = param_shape(*scale, "BatchNorm scale")?;
                    if s.numel() != x.c {
                        return Err(GraphError::ShapeMismatch(format!(
                            "BN scale has {} elements, input has {} channels",
                            s.numel(),
                            x.c
                        )));
                    }
                    x
                }
                OpKind::GroupNorm { groups, scale, .. } => {
                    let x = input_shape(0)?;
                    if *groups == 0 || x.c % *groups != 0 {
                        return Err(GraphError::ShapeMismatch(format!(
                            "channel count {} not divisible by {groups} groups",
                            x.c
                        )));
                    }
                    let s = param_shape(*scale, "GroupNorm scale")?;
                    if s.numel() != x.c {
                        return Err(GraphError::ShapeMismatch(format!(
                            "GN scale has {} elements, input has {} channels",
                            s.numel(),
                            x.c
                        )));
                    }
                    x
                }
                OpKind::Relu | OpKind::SiLU | OpKind::Sigmoid => input_shape(0)?,
                OpKind::Add | OpKind::Mul => {
                    let a = input_shape(0)?;
                    let b = input_shape(1)?;
                    // Secondary operand may broadcast into primary shape.
                    broadcast_shape(a, b)?;
                    a
                }
                OpKind::ResizeBilinear { scale_h, scale_w, .. } => {
                    let x = input_shape(0)?;
                    if *scale_h <= 0.0 || *scale_w <= 0.0 {
                        return Err(GraphError::ShapeMismatch("resize scale must be positive".into()));
                    }
                    TensorShape::new(
                        x.n,
                        x.c,
                        ((x.h as f32) * scale_h).round().max(1.0) as usize,
                        ((x.w as f32) * scale_w).round().max(1.0) as usize,
                    )
                }
                OpKind::GridSample => {
                    let x = input_shape(0)?;
                    let grid = input_shape(1)?;
                    if grid.c != 2 {
                        return Err(GraphError::ShapeMismatch(
                            "GridSample coordinate tensor must have 2 channels".into(),
                        ));
                    }
                    // Output geometry follows the grid; channels follow the
                    // sampled feature map.
                    TensorShape::new(grid.n, x.c, grid.h, grid.w)
                }
                OpKind::MatMul => {
                    let a = input_shape(0)?;
                    let b = input_shape(1)?;
                    if a.n != b.n || a.w != b.h {
                        return Err(GraphError::ShapeMismatch(format!(
                            "MatMul shapes {a:?} and {b:?} are incompatible"
                        )));
                    }
                    TensorShape::new(a.n, a.c, b.c, 1)
                }
                OpKind::QuantizeLinear { .. } | OpKind::DequantizeLinear { .. } => input_shape(0)?,
            };
            shapes.insert(node.output, shape);
        }
        for &out in &self.outputs {
            if !shapes.contains_key(&out) {
                return Err(GraphError::UnknownOutput(out));
            }
        }
        Ok(shapes)
    }
}

/// NCHW broadcasting: a singleton axis may be stretched, other axes must be
/// equal. Returns an explicit error on incompatible shapes.
fn broadcast_shape(a: TensorShape, b: TensorShape) -> GraphResult<TensorShape> {
    let axis = |sa: usize, sb: usize| -> GraphResult<usize> {
        if sa == sb {
            Ok(sa)
        } else if sa == 1 {
            Ok(sb)
        } else if sb == 1 {
            Ok(sa)
        } else {
            Err(GraphError::ShapeMismatch(format!(
                "cannot broadcast {a:?} and {b:?}"
            )))
        }
    };
    Ok(TensorShape::new(
        axis(a.n, b.n)?,
        axis(a.c, b.c)?,
        axis(a.h, b.h)?,
        axis(a.w, b.w)?,
    ))
}

// ---------------------------------------------------------------------------
// Compile passes
// ---------------------------------------------------------------------------

/// One convolution that was fused with a following BatchNorm.
#[derive(Clone, Debug, PartialEq)]
pub struct FusedConv {
    pub conv_node: usize,
    pub batch_norm_node: usize,
    pub folded_weight: usize,
    pub folded_bias: usize,
}

/// Result of the Conv+BatchNorm folding pass.
#[derive(Clone, Debug, PartialEq)]
pub struct FoldReport {
    pub fused: Vec<FusedConv>,
}

impl FoldReport {
    /// Number of fused `(Conv2d, BatchNorm)` pairs.
    pub fn fused_pairs(&self) -> usize {
        self.fused.len()
    }
}

/// Fold every `Conv2d -> BatchNorm` pair where the BatchNorm is the conv's
/// only consumer and all parameters are graph constants.
///
/// For each output channel:
///   `w' = w * gamma / sqrt(var + eps)`
///   `b' = (b - mean) * gamma / sqrt(var + eps) + beta`
/// The folded parameters replace the conv's original weight and bias, and
/// every consumer of the BN output is rewired to the conv output before the
/// BN node is removed.
pub fn fold_conv_batchnorm(graph: &mut Graph) -> GraphResult<FoldReport> {
    let order = graph.topo_order()?;
    let mut consumers: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        for &input in &node.inputs {
            consumers.entry(input).or_default().push(i);
        }
    }
    let mut removed: HashSet<usize> = HashSet::new();
    let mut fused: Vec<FusedConv> = Vec::new();

    for &conv_idx in &order {
        if removed.contains(&conv_idx) {
            continue;
        }
        let (weight_id, bias_id, conv_output) = match &graph.nodes[conv_idx].op {
            OpKind::Conv2d {
                weight, bias, ..
            } => (*weight, *bias, graph.nodes[conv_idx].output),
            _ => continue,
        };
        let bn_indices = consumers.get(&conv_output).cloned().unwrap_or_default();
        if bn_indices.len() != 1 {
            continue;
        }
        let bn_idx = bn_indices[0];
        let (scale_id, offset_id, mean_id, var_id, epsilon) = match &graph.nodes[bn_idx].op {
            OpKind::BatchNorm {
                scale,
                offset,
                mean,
                var,
                epsilon,
            } => (*scale, *offset, *mean, *var, *epsilon),
            _ => continue,
        };
        let bn_output = graph.nodes[bn_idx].output;

        let (w_shape, new_w, new_b) = {
            let w_tensor = graph
                .constants
                .get(&weight_id)
                .ok_or_else(|| GraphError::Unsupported("conv weight is not a constant".into()))?;
            let (gamma, beta, mu, variance) = {
                let read = |id: usize| -> GraphResult<Vec<f32>> {
                    graph
                        .constants
                        .get(&id)
                        .ok_or_else(|| GraphError::Unsupported("BN parameter is not a constant".into()))?
                        .f32_values()
                };
                (read(scale_id)?, read(offset_id)?, read(mean_id)?, read(var_id)?)
            };
            let w = w_tensor.f32_values()?;
            let oc = w_tensor.shape.n;
            let per_channel = w_tensor.shape.c * w_tensor.shape.h * w_tensor.shape.w;
            if gamma.len() != oc || beta.len() != oc || mu.len() != oc || variance.len() != oc {
                return Err(GraphError::ShapeMismatch(
                    "BatchNorm parameter length does not match conv output channels".into(),
                ));
            }
            let mut folded_w = vec![0.0f32; w.len()];
            let mut folded_b = vec![0.0f32; oc];
            for ch in 0..oc {
                let inv_std = 1.0 / (variance[ch] + epsilon).sqrt();
                let a = gamma[ch] * inv_std;
                for k in 0..per_channel {
                    folded_w[ch * per_channel + k] = w[ch * per_channel + k] * a;
                }
                let old_bias = match bias_id {
                    Some(id) => graph
                        .constants
                        .get(&id)
                        .and_then(|t| t.f32_values().ok())
                        .and_then(|v| v.get(ch).copied())
                        .unwrap_or(0.0),
                    None => 0.0,
                };
                folded_b[ch] = (old_bias - mu[ch]) * a + beta[ch];
            }
            (w_tensor.shape, folded_w, folded_b)
        };

        let folded_weight =
            graph.add_constant(TensorBuffer::from_f32(w_shape, &new_w)?);
        let folded_bias = graph.add_constant(TensorBuffer::from_f32(
            TensorShape::vector(new_b.len()),
            &new_b,
        )?);
        if let OpKind::Conv2d { weight, bias, .. } = &mut graph.nodes[conv_idx].op {
            *weight = folded_weight;
            *bias = Some(folded_bias);
        }
        for node in &mut graph.nodes {
            for input in &mut node.inputs {
                if *input == bn_output {
                    *input = conv_output;
                }
            }
        }
        for out in &mut graph.outputs {
            if *out == bn_output {
                *out = conv_output;
            }
        }
        removed.insert(bn_idx);
        fused.push(FusedConv {
            conv_node: conv_idx,
            batch_norm_node: bn_idx,
            folded_weight,
            folded_bias,
        });
    }

    // Remove BN nodes by original index (indices are still valid here).
    let mut indexed: Vec<(usize, Node)> = std::mem::take(&mut graph.nodes)
        .into_iter()
        .enumerate()
        .collect();
    indexed.retain(|(i, _)| !removed.contains(i));
    graph.nodes = indexed.into_iter().map(|(_, node)| node).collect();

    Ok(FoldReport { fused })
}

// ---------------------------------------------------------------------------
// Memory planning
// ---------------------------------------------------------------------------

/// A tensor's live range in the schedule (`[start, end)` in node steps).
#[derive(Clone, Debug, PartialEq)]
pub struct Lifetime {
    pub tensor: usize,
    pub shape: TensorShape,
    pub dtype: DType,
    pub start: usize,
    pub end: usize,
    /// Reusable storage slot index.
    pub slot: usize,
}

/// Result of lifetime-aware memory slot allocation.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryPlan {
    pub lifetimes: Vec<Lifetime>,
    /// Number of reusable storage slots.
    pub slot_count: usize,
    /// Bytes required if every tensor kept a permanent allocation (no
    /// lifetime-based reuse at all).
    pub naive_peak_bytes: usize,
    /// Peak bytes simultaneously occupied with slot reuse.
    pub reused_peak_bytes: usize,
    pub total_tensors: usize,
}

impl MemoryPlan {
    /// `reused_peak_bytes / naive_peak_bytes`, rounded to four decimals.
    pub fn compression_ratio(&self) -> f32 {
        if self.naive_peak_bytes == 0 {
            return 1.0;
        }
        let ratio = self.reused_peak_bytes as f32 / self.naive_peak_bytes as f32;
        (ratio * 10_000.0).round() / 10_000.0
    }
}

/// Plan memory slots using first-fit reuse: a tensor may occupy a slot only
/// after every previous occupant's live range has ended.
///
/// Two tensors that are live at the same step always receive different slots,
/// so a backend executing the schedule in topological order cannot overwrite
/// a value before its last reader.
pub fn plan_memory(graph: &Graph) -> GraphResult<MemoryPlan> {
    let order = graph.topo_order()?;
    let shapes = graph.infer_shapes()?;
    let steps = order.len();
    let step_of: HashMap<usize, usize> = order
        .iter()
        .enumerate()
        .map(|(step, &idx)| (idx, step))
        .collect();

    // Collect all tensors: constants plus node outputs.
    let mut tensor_info: Vec<(usize, TensorShape, DType)> = graph
        .constants
        .iter()
        .map(|(&id, t)| (id, t.shape, t.dtype))
        .collect();
    for &idx in &order {
        let node = &graph.nodes[idx];
        tensor_info.push((
            node.output,
            shapes[&node.output],
            DType::Fp32,
        ));
    }

    // First and last read step for each tensor.
    let mut first_read: HashMap<usize, usize> = HashMap::new();
    let mut last_read_end: HashMap<usize, usize> = HashMap::new();
    for (step, &idx) in order.iter().enumerate() {
        for &input in &graph.nodes[idx].inputs {
            first_read.entry(input).or_insert(step);
            last_read_end.insert(input, step + 1);
        }
    }

    let output_set: HashSet<usize> = graph.outputs.iter().copied().collect();
    let mut lifetimes: Vec<Lifetime> = Vec::new();
    for (tensor, shape, dtype) in &tensor_info {
        let is_constant = graph.constants.contains_key(tensor);
        let start = if is_constant {
            *first_read.get(tensor).unwrap_or(&0)
        } else {
            step_of[&graph.nodes.iter().position(|n| n.output == *tensor).unwrap()]
        };
        let mut end = *last_read_end.get(tensor).unwrap_or(&start);
        if !is_constant && output_set.contains(tensor) {
            end = end.max(steps);
        }
        end = end.max(start + 1);
        lifetimes.push(Lifetime {
            tensor: *tensor,
            shape: *shape,
            dtype: *dtype,
            start,
            end,
            slot: usize::MAX,
        });
    }

    // First-fit slot assignment, processed by increasing start step.
    let mut order_indices: Vec<usize> = (0..lifetimes.len()).collect();
    order_indices.sort_by_key(|&i| (lifetimes[i].start, lifetimes[i].tensor));
    let mut slot_ends: Vec<usize> = Vec::new();
    for i in order_indices {
        let start = lifetimes[i].start;
        let end = lifetimes[i].end;
        let mut chosen = slot_ends.iter().position(|&slot_end| slot_end <= start);
        if chosen.is_none() {
            slot_ends.push(0);
            chosen = Some(slot_ends.len() - 1);
        }
        let slot = chosen.unwrap();
        lifetimes[i].slot = slot;
        slot_ends[slot] = slot_ends[slot].max(end);
    }
    let slot_count = slot_ends.len();

    // Naive = every tensor allocated permanently; reused = peak sum of bytes
    // of distinct occupied slots at any step.
    let naive_peak_bytes: usize = lifetimes
        .iter()
        .map(|lt| lt.shape.byte_len(lt.dtype))
        .sum();
    let mut reused_peak_bytes = 0usize;
    for step in 0..steps {
        let mut occupied: HashMap<usize, usize> = HashMap::new();
        for lt in &lifetimes {
            if lt.start <= step && step < lt.end {
                let bytes = lt.shape.byte_len(lt.dtype);
                let entry = occupied.entry(lt.slot).or_insert(0);
                *entry = (*entry).max(bytes);
            }
        }
        reused_peak_bytes = reused_peak_bytes.max(occupied.values().sum());
    }

    Ok(MemoryPlan {
        total_tensors: lifetimes.len(),
        lifetimes,
        slot_count,
        naive_peak_bytes,
        reused_peak_bytes,
    })
}

// ---------------------------------------------------------------------------
// CPU reference operator kernels
// ---------------------------------------------------------------------------

fn read_f32(tensors: &HashMap<usize, TensorBuffer>, id: usize) -> GraphResult<Vec<f32>> {
    tensors
        .get(&id)
        .ok_or_else(|| GraphError::TensorMismatch(format!("missing tensor {id}")))?
        .f32_values()
}

fn read_vec_f32(
    tensors: &HashMap<usize, TensorBuffer>,
    id: usize,
    expected: usize,
) -> GraphResult<Vec<f32>> {
    let v = read_f32(tensors, id)?;
    if v.len() != expected {
        return Err(GraphError::ShapeMismatch(format!(
            "parameter {id} has {} elements, expected {expected}",
            v.len()
        )));
    }
    Ok(v)
}

#[allow(clippy::too_many_arguments)]
fn conv_core(
    x: &[f32],
    xs: TensorShape,
    w: &[f32],
    ws: TensorShape,
    bias: Option<&[f32]>,
    out: &mut [f32],
    os: TensorShape,
    stride: (usize, usize),
    pad: Pad4,
    dilation: (usize, usize),
    depthwise: bool,
) -> GraphResult<()> {
    if w.len() != ws.numel() || x.len() != xs.numel() || out.len() != os.numel() {
        return Err(GraphError::TensorMismatch("conv buffer length mismatch".into()));
    }
    for n in 0..os.n {
        for oc in 0..os.c {
            let ic_start = if depthwise { oc } else { 0 };
            let ic_end = if depthwise { oc + 1 } else { ws.c };
            for oh in 0..os.h {
                for ow in 0..os.w {
                    let mut acc = bias.map(|b| b[oc]).unwrap_or(0.0);
                    for ic in ic_start..ic_end {
                        for kh in 0..ws.h {
                            for kw in 0..ws.w {
                                let ih = oh * stride.0 + kh * dilation.0;
                                let iw = ow * stride.1 + kw * dilation.1;
                                if ih < pad.0 || iw < pad.2 {
                                    continue;
                                }
                                let ih = ih - pad.0;
                                let iw = iw - pad.2;
                                if ih >= xs.h || iw >= xs.w {
                                    continue;
                                }
                                acc += x[xs.index(n, ic, ih, iw)?]
                                    * w[ws.index(oc, if depthwise { 0 } else { ic }, kh, kw)?];
                            }
                        }
                    }
                    out[os.index(n, oc, oh, ow)?] = acc;
                }
            }
        }
    }
    Ok(())
}

fn batchnorm_core(
    x: &mut [f32],
    shape: TensorShape,
    scale: &[f32],
    offset: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
) -> GraphResult<()> {
    for n in 0..shape.n {
        for c in 0..shape.c {
            let inv_std = 1.0 / (var[c] + eps).sqrt();
            for h in 0..shape.h {
                for w in 0..shape.w {
                    let i = shape.index(n, c, h, w)?;
                    x[i] = (x[i] - mean[c]) * inv_std * scale[c] + offset[c];
                }
            }
        }
    }
    Ok(())
}

fn groupnorm_core(
    x: &[f32],
    out: &mut [f32],
    shape: TensorShape,
    groups: usize,
    scale: &[f32],
    offset: &[f32],
    eps: f32,
) -> GraphResult<()> {
    if groups == 0 || shape.c % groups != 0 || scale.len() != shape.c || offset.len() != shape.c {
        return Err(GraphError::ShapeMismatch("invalid GroupNorm parameters".into()));
    }
    let per_group = shape.c / groups;
    let spatial = shape.h * shape.w;
    for n in 0..shape.n {
        for g in 0..groups {
            // Welford's online algorithm for numerical stability.
            let mut mean = 0.0f32;
            let mut m2 = 0.0f32;
            let mut count = 0usize;
            for c in g * per_group..(g + 1) * per_group {
                for s in 0..spatial {
                    let v = x[shape.index(n, c, s / shape.w, s % shape.w)?];
                    count += 1;
                    let delta = v - mean;
                    mean += delta / count as f32;
                    m2 += delta * (v - mean);
                }
            }
            let variance = if count == 0 { 0.0 } else { m2 / count as f32 };
            let inv_std = 1.0 / (variance + eps).sqrt();
            for c in g * per_group..(g + 1) * per_group {
                for s in 0..spatial {
                    let i = shape.index(n, c, s / shape.w, s % shape.w)?;
                    out[i] = (x[i] - mean) * inv_std * scale[c] + offset[c];
                }
            }
        }
    }
    Ok(())
}

fn resize_bilinear_core(
    x: &[f32],
    xs: TensorShape,
    out: &mut [f32],
    os: TensorShape,
    scale_h: f32,
    scale_w: f32,
    align_corners: bool,
) -> GraphResult<()> {
    let in_h = xs.h as f32;
    let in_w = xs.w as f32;
    for n in 0..os.n {
        for c in 0..os.c {
            for y in 0..os.h {
                let src_y = if align_corners && os.h > 1 {
                    y as f32 * (in_h - 1.0) / (os.h - 1) as f32
                } else {
                    (y as f32 + 0.5) / scale_h - 0.5
                };
                let y0 = src_y.floor().max(0.0) as usize;
                let y1 = (y0 + 1).min(xs.h - 1);
                let ay = (src_y - y0 as f32).clamp(0.0, 1.0);
                for x_pos in 0..os.w {
                    let src_x = if align_corners && os.w > 1 {
                        x_pos as f32 * (in_w - 1.0) / (os.w - 1) as f32
                    } else {
                        (x_pos as f32 + 0.5) / scale_w - 0.5
                    };
                    let x0 = src_x.floor().max(0.0) as usize;
                    let x1 = (x0 + 1).min(xs.w - 1);
                    let ax = (src_x - x0 as f32).clamp(0.0, 1.0);
                    let v00 = x[xs.index(n, c, y0, x0)?];
                    let v01 = x[xs.index(n, c, y0, x1)?];
                    let v10 = x[xs.index(n, c, y1, x0)?];
                    let v11 = x[xs.index(n, c, y1, x1)?];
                    out[os.index(n, c, y, x_pos)?] = (1.0 - ay) * ((1.0 - ax) * v00 + ax * v01)
                        + ay * ((1.0 - ax) * v10 + ax * v11);
                }
            }
        }
    }
    Ok(())
}

fn grid_sample_core(
    x: &[f32],
    xs: TensorShape,
    grid: &[f32],
    gs: TensorShape,
    out: &mut [f32],
    os: TensorShape,
) -> GraphResult<()> {
    if gs.c != 2 || gs.n != os.n || gs.h != os.h || gs.w != os.w {
        return Err(GraphError::ShapeMismatch(
            "GridSample grid shape does not match output geometry".into(),
        ));
    }
    for n in 0..os.n {
        for c in 0..os.c {
            for h in 0..os.h {
                for w in 0..os.w {
                    let i = os.index(n, c, h, w)?;
                    let gx = grid[gs.index(n, 0, h, w)?];
                    let gy = grid[gs.index(n, 1, h, w)?];
                    let ix = (gx + 1.0) * 0.5 * xs.w as f32 - 0.5;
                    let iy = (gy + 1.0) * 0.5 * xs.h as f32 - 0.5;
                    let x0 = ix.floor() as isize;
                    let y0 = iy.floor() as isize;
                    let ax = ix - x0 as f32;
                    let ay = iy - y0 as f32;
                    let mut acc = 0.0f32;
                    for (dy, wy) in [(0isize, 1.0 - ay), (1isize, ay)] {
                        for (dx, wx) in [(0isize, 1.0 - ax), (1isize, ax)] {
                            let sx = x0 + dx;
                            let sy = y0 + dy;
                            // Out-of-bounds samples contribute zero.
                            if sx >= 0 && sy >= 0 && sx < xs.w as isize && sy < xs.h as isize {
                                acc += wx * wy * x[xs.index(n, c, sy as usize, sx as usize)?];
                            }
                        }
                    }
                    out[i] = acc;
                }
            }
        }
    }
    Ok(())
}

fn matmul_core(
    a: &[f32],
    as_: TensorShape,
    b: &[f32],
    bs: TensorShape,
    out: &mut [f32],
    os: TensorShape,
) -> GraphResult<()> {
    /// Read A[i,k]: rows live in the `c` axis, columns in `w`.
    let a_index = |n: usize, i: usize, k: usize| -> GraphResult<usize> {
        as_.index(n, i, 0, k)
    };
    /// Read B[k,j]: rows live in `h`, columns in `c` (transposed layout).
    let b_index = |n: usize, k: usize, j: usize| -> GraphResult<usize> {
        bs.index(n, j, k, 0)
    };
    for n in 0..os.n {
        for i in 0..os.c {
            for j in 0..os.h {
                let mut acc = 0.0f32;
                for k in 0..as_.w {
                    acc += a[a_index(n, i, k)?] * b[b_index(n, k, j)?];
                }
                out[os.index(n, i, j, 0)?] = acc;
            }
        }
    }
    Ok(())
}

fn quantize_f32_to_i8(x: &[f32], scale: f32, zero_point: i8) -> GraphResult<Vec<u8>> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(GraphError::Unsupported("quantization scale must be positive".into()));
    }
    let mut bytes = Vec::with_capacity(x.len());
    for &v in x {
        let q = (v / scale).round() as i32 + zero_point as i32;
        bytes.push(q.clamp(i8::MIN as i32, i8::MAX as i32) as i8 as u8);
    }
    Ok(bytes)
}

fn dequantize_i8_to_f32(bytes: &[u8], scale: f32, zero_point: i8) -> Vec<f32> {
    bytes
        .iter()
        .map(|&b| (b as i8 - zero_point) as f32 * scale)
        .collect()
}

/// Execute one node, reading constants and previously computed values from
/// `tensors` and inserting its output there.
fn execute_node(
    graph: &Graph,
    node: &Node,
    shapes: &HashMap<usize, TensorShape>,
    tensors: &mut HashMap<usize, TensorBuffer>,
) -> GraphResult<()> {
    let out_shape = shapes[&node.output];
    let make = |values: Vec<f32>| TensorBuffer::from_f32(out_shape, &values);

    match &node.op {
        OpKind::Conv2d {
            stride, pad, weight, bias, ..
        }
        | OpKind::DepthwiseConv2d {
            stride, pad, weight, bias, ..
        } => {
            let depthwise = matches!(node.op, OpKind::DepthwiseConv2d { .. });
            let dil = match &node.op {
                OpKind::Conv2d { dilation, .. } => *dilation,
                _ => (1, 1),
            };
            let x_tensor = tensors
                .get(&node.inputs[0])
                .ok_or_else(|| GraphError::TensorMismatch("conv: missing input tensor".into()))?;
            let (xs, xv) = (x_tensor.shape, x_tensor.f32_values()?);
            let wv = read_f32(tensors, *weight)?;
            let ws = tensors[weight].shape;
            let b = match bias {
                Some(id) => Some(read_vec_f32(tensors, *id, ws.n)?),
                None => None,
            };
            let mut out = vec![0.0f32; out_shape.numel()];
            conv_core(
                &xv, xs, &wv, ws, b.as_deref(), &mut out, out_shape,
                *stride, *pad, dil, depthwise,
            )?;
            tensors.insert(node.output, make(out)?);
        }
        OpKind::BatchNorm {
            scale,
            offset,
            mean,
            var,
            epsilon,
        } => {
            let x_tensor = tensors
                .get(&node.inputs[0])
                .ok_or_else(|| GraphError::TensorMismatch("BN: missing input tensor".into()))?;
            let shape = x_tensor.shape;
            let mut values = x_tensor.f32_values()?;
            let ch = shape.c;
            let s = read_vec_f32(tensors, *scale, ch)?;
            let o = read_vec_f32(tensors, *offset, ch)?;
            let m = read_vec_f32(tensors, *mean, ch)?;
            let v = read_vec_f32(tensors, *var, ch)?;
            batchnorm_core(&mut values, shape, &s, &o, &m, &v, *epsilon)?;
            tensors.insert(node.output, make(values)?);
        }
        OpKind::GroupNorm {
            groups,
            scale,
            offset,
            epsilon,
        } => {
            let x_tensor = tensors
                .get(&node.inputs[0])
                .ok_or_else(|| GraphError::TensorMismatch("GN: missing input tensor".into()))?;
            let (shape, values) = (x_tensor.shape, x_tensor.f32_values()?);
            let s = read_vec_f32(tensors, *scale, shape.c)?;
            let o = read_vec_f32(tensors, *offset, shape.c)?;
            let mut out = vec![0.0f32; shape.numel()];
            groupnorm_core(&values, &mut out, shape, *groups, &s, &o, *epsilon)?;
            tensors.insert(node.output, make(out)?);
        }
        OpKind::Relu | OpKind::SiLU | OpKind::Sigmoid => {
            let values = read_f32(tensors, node.inputs[0])?;
            let mapped: Vec<f32> = values
                .iter()
                .map(|&v| match node.op {
                    OpKind::Relu => v.max(0.0),
                    OpKind::SiLU => v / (1.0 + (-v).exp()),
                    OpKind::Sigmoid => 1.0 / (1.0 + (-v).exp()),
                    _ => unreachable!(),
                })
                .collect();
            tensors.insert(node.output, make(mapped)?);
        }
        OpKind::Add | OpKind::Mul => {
            let av = read_f32(tensors, node.inputs[0])?;
            let bv = read_f32(tensors, node.inputs[1])?;
            let ashape = tensors[&node.inputs[0]].shape;
            let bshape = tensors[&node.inputs[1]].shape;
            // Operators are defined to keep the primary input's shape; a
            // secondary operand may broadcast into it (skip connections).
            let out_shape = ashape;
            let mut out = vec![0.0f32; out_shape.numel()];
            for n in 0..out_shape.n {
                for c in 0..out_shape.c {
                    for h in 0..out_shape.h {
                        for w in 0..out_shape.w {
                            let ai = ashape.index(
                                if ashape.n == 1 { 0 } else { n },
                                if ashape.c == 1 { 0 } else { c },
                                if ashape.h == 1 { 0 } else { h },
                                if ashape.w == 1 { 0 } else { w },
                            )?;
                            let bi = bshape.index(
                                if bshape.n == 1 { 0 } else { n },
                                if bshape.c == 1 { 0 } else { c },
                                if bshape.h == 1 { 0 } else { h },
                                if bshape.w == 1 { 0 } else { w },
                            )?;
                            let i = out_shape.index(n, c, h, w)?;
                            out[i] = match node.op {
                                OpKind::Add => av[ai] + bv[bi],
                                OpKind::Mul => av[ai] * bv[bi],
                                _ => unreachable!(),
                            };
                        }
                    }
                }
            }
            tensors.insert(node.output, make(out)?);
        }
        OpKind::ResizeBilinear {
            scale_h,
            scale_w,
            align_corners,
        } => {
            let xv = read_f32(tensors, node.inputs[0])?;
            let xs = tensors[&node.inputs[0]].shape;
            let mut out = vec![0.0f32; out_shape.numel()];
            resize_bilinear_core(&xv, xs, &mut out, out_shape, *scale_h, *scale_w, *align_corners)?;
            tensors.insert(node.output, make(out)?);
        }
        OpKind::GridSample => {
            let xv = read_f32(tensors, node.inputs[0])?;
            let gv = read_f32(tensors, node.inputs[1])?;
            let xs = tensors[&node.inputs[0]].shape;
            let gs = tensors[&node.inputs[1]].shape;
            let mut out = vec![0.0f32; out_shape.numel()];
            grid_sample_core(&xv, xs, &gv, gs, &mut out, out_shape)?;
            tensors.insert(node.output, make(out)?);
        }
        OpKind::MatMul => {
            let av = read_f32(tensors, node.inputs[0])?;
            let bv = read_f32(tensors, node.inputs[1])?;
            let as_ = tensors[&node.inputs[0]].shape;
            let bs = tensors[&node.inputs[1]].shape;
            if tensors[&node.inputs[0]].dtype != DType::Fp32
                || tensors[&node.inputs[1]].dtype != DType::Fp32
            {
                return Err(GraphError::Unsupported("MatMul needs FP32 inputs".into()));
            }
            let mut out = vec![0.0f32; out_shape.numel()];
            matmul_core(&av, as_, &bv, bs, &mut out, out_shape)?;
            tensors.insert(node.output, make(out)?);
        }
        OpKind::QuantizeLinear { scale, zero_point } => {
            let values = read_f32(tensors, node.inputs[0])?;
            let bytes = quantize_f32_to_i8(&values, *scale, *zero_point)?;
            let tensor = TensorBuffer::from_bytes(DType::Int8, out_shape, bytes)?;
            tensors.insert(node.output, tensor);
        }
        OpKind::DequantizeLinear { scale, zero_point } => {
            let input = tensors
                .get(&node.inputs[0])
                .ok_or_else(|| GraphError::TensorMismatch("dequant: missing input".into()))?;
            if input.dtype != DType::Int8 {
                return Err(GraphError::Unsupported(
                    "DequantizeLinear needs Int8 input".into(),
                ));
            }
            let values = dequantize_i8_to_f32(&input.bytes, *scale, *zero_point);
            tensors.insert(node.output, make(values)?);
        }
    }
    Ok(())
}

/// Result of running a graph on the CPU reference executor.
#[derive(Clone, Debug)]
pub struct ExecutionResult {
    /// Declared graph outputs, in declaration order.
    pub outputs: Vec<TensorBuffer>,
    /// All intermediate tensors, useful for debugging memory plans.
    pub tensors: HashMap<usize, TensorBuffer>,
}

/// Execute a graph with caller-supplied runtime inputs. Constants are taken
/// from the graph; `inputs` may add or override tensors by id.
pub fn execute_graph(
    graph: &Graph,
    inputs: &HashMap<usize, TensorBuffer>,
) -> GraphResult<ExecutionResult> {
    let extra_shapes: HashMap<usize, TensorShape> =
        inputs.iter().map(|(&id, t)| (id, t.shape)).collect();
    let known: HashSet<usize> = extra_shapes.keys().copied().collect();
    let order = graph.topo_order_with_known(&known)?;
    let shapes = graph.infer_shapes_with(&extra_shapes)?;
    let mut tensors = graph.constants.clone();
    for (&id, tensor) in inputs {
        tensors.insert(id, tensor.clone());
    }
    for &idx in &order {
        execute_node(graph, &graph.nodes[idx], &shapes, &mut tensors)?;
    }
    let mut outputs = Vec::with_capacity(graph.outputs.len());
    for &id in &graph.outputs {
        outputs.push(
            tensors
                .get(&id)
                .cloned()
                .ok_or(GraphError::UnknownOutput(id))?,
        );
    }
    Ok(ExecutionResult { outputs, tensors })
}

/// Largest absolute difference between two FP32 tensors.
pub fn max_abs_diff(a: &TensorBuffer, b: &TensorBuffer) -> GraphResult<f32> {
    let av = a.f32_values()?;
    let bv = b.f32_values()?;
    if av.len() != bv.len() {
        return Err(GraphError::TensorMismatch(
            "cannot compare tensors of different lengths".into(),
        ));
    }
    Ok(av.iter().zip(&bv).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ones(shape: TensorShape) -> TensorBuffer {
        TensorBuffer::from_f32(shape, &vec![1.0; shape.numel()]).unwrap()
    }

    fn param(len: usize, values: &[f32]) -> TensorBuffer {
        assert_eq!(values.len(), len);
        TensorBuffer::from_f32(TensorShape::vector(len), values).unwrap()
    }

    #[test]
    fn dtype_and_tensor_buffer_validation() {
        assert_eq!(DType::Fp32.byte_width(), 4);
        assert_eq!(DType::Fp16.byte_width(), 2);
        assert_eq!(DType::Int8.byte_width(), 1);
        let shape = TensorShape::new(2, 3, 2, 2);
        assert_eq!(shape.numel(), 24);
        assert_eq!(shape.index(1, 2, 1, 1).unwrap(), 23);
        assert!(shape.index(0, 3, 0, 0).is_err());
        let mut t = TensorBuffer::zeros(DType::Fp32, shape);
        assert_eq!(t.bytes.len(), 96);
        assert!(TensorBuffer::from_bytes(DType::Fp32, shape, vec![0; 95]).is_err());
        assert!(TensorBuffer::from_f32(shape, &vec![0.0; 23]).is_err());
        t.set_f32_values(&vec![0.5; 24]).unwrap();
        assert!(t.f32_values().unwrap().iter().all(|&v| v == 0.5));
        assert!(t.set_f32_values(&[0.0; 23]).is_err());
        let mut i8_tensor = TensorBuffer::zeros(DType::Int8, TensorShape::vector(2));
        i8_tensor.bytes[0] = 3;
        assert_eq!(i8_tensor.bytes, vec![3, 0]);
        assert!(i8_tensor.f32_values().is_err());
    }

    #[test]
    fn topo_order_rejects_cycle_unknown_input_and_duplicate_output() {
        let mut g = Graph::new();
        let a = g.add_constant(ones(TensorShape::vector(1)));
        let o1 = g.fresh_id();
        let o2 = g.fresh_id();
        g.add_node(Node::new(OpKind::Relu, vec![o2], o1, "first"));
        g.add_node(Node::new(OpKind::Relu, vec![o1], o2, "second"));
        match g.topo_order() {
            Err(GraphError::Cycle { nodes }) => assert_eq!(nodes.len(), 2),
            other => panic!("expected cycle error, got {other:?}"),
        }

        let mut g2 = Graph::new();
        let missing = g2.fresh_id();
        let out = g2.fresh_id();
        g2.add_node(Node::new(OpKind::Relu, vec![missing], out, "bad"));
        match g2.topo_order() {
            Err(GraphError::UnknownInput { node: 0, input }) => assert_eq!(input, missing),
            other => panic!("expected unknown input error, got {other:?}"),
        }

        let mut g3 = Graph::new();
        let x = g3.add_constant(ones(TensorShape::vector(1)));
        let dup = g3.fresh_id();
        g3.add_node(Node::new(OpKind::Relu, vec![x], dup, "a"));
        g3.add_node(Node::new(OpKind::Relu, vec![x], dup, "b"));
        assert!(matches!(g3.topo_order(), Err(GraphError::ShapeMismatch(_))));
    }

    #[test]
    fn invalid_shapes_return_explicit_errors() {
        let mut g = Graph::new();
        let x = g.add_constant(ones(TensorShape::new(1, 2, 4, 4)));
        let w = g.add_constant(ones(TensorShape::new(2, 3, 3, 3)));
        let out = g.fresh_id();
        g.add_node(Node::new(
            OpKind::Conv2d {
                stride: (1, 1),
                pad: (1, 1, 1, 1),
                dilation: (1, 1),
                weight: w,
                bias: None,
            },
            vec![x],
            out,
            "bad_conv",
        ));
        assert!(matches!(g.infer_shapes(), Err(GraphError::ShapeMismatch(_))));

        let mut g2 = Graph::new();
        let a = g2.add_constant(ones(TensorShape::new(1, 1, 2, 3)));
        let b = g2.add_constant(ones(TensorShape::new(1, 1, 4, 3)));
        let out = g2.fresh_id();
        g2.add_node(Node::new(OpKind::Add, vec![a, b], out, "bad_add"));
        assert!(matches!(
            execute_graph(&g2, &HashMap::new()),
            Err(GraphError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn conv_batchnorm_fold_matches_unfolded_output() {
        let shape = TensorShape::new(1, 2, 4, 4);
        let values: Vec<f32> = (0..shape.numel()).map(|i| (i as f32) * 0.07 - 0.3).collect();
        let input = TensorBuffer::from_f32(shape, &values).unwrap();
        let wshape = TensorShape::new(2, 2, 3, 3);
        let wvals: Vec<f32> = (0..wshape.numel())
            .map(|i| ((i % 7) as f32 - 3.0) * 0.11)
            .collect();

        let mut g = Graph::new();
        let x = g.add_constant(input);
        let w = g.add_constant(TensorBuffer::from_f32(wshape, &wvals).unwrap());
        let b = g.add_constant(param(2, &[0.17, -0.23]));
        let s = g.add_constant(param(2, &[1.3, 0.7]));
        let o = g.add_constant(param(2, &[-0.2, 0.4]));
        let m = g.add_constant(param(2, &[0.1, -0.15]));
        let v = g.add_constant(param(2, &[0.9, 1.21]));
        let conv_out = g.fresh_id();
        g.add_node(Node::new(
            OpKind::Conv2d {
                stride: (1, 1),
                pad: (1, 1, 1, 1),
                dilation: (1, 1),
                weight: w,
                bias: Some(b),
            },
            vec![x],
            conv_out,
            "conv",
        ));
        let bn_out = g.fresh_id();
        g.add_node(Node::new(
            OpKind::BatchNorm {
                scale: s,
                offset: o,
                mean: m,
                var: v,
                epsilon: 1e-5,
            },
            vec![conv_out],
            bn_out,
            "bn",
        ));
        let activated = g.fresh_id();
        g.add_node(Node::new(OpKind::SiLU, vec![bn_out], activated, "silu"));
        g.set_outputs(vec![activated]);

        let before = execute_graph(&g, &HashMap::new()).unwrap();
        let mut folded = g.clone();
        let report = fold_conv_batchnorm(&mut folded).unwrap();
        assert_eq!(report.fused_pairs(), 1);
        assert_eq!(folded.nodes.len(), g.nodes.len() - 1);
        assert_eq!(report.fused[0].folded_weight, folded.nodes[0].op.weight_id());
        assert_eq!(report.fused[0].folded_bias, folded.nodes[0].op.bias_id().unwrap());
        let after = execute_graph(&folded, &HashMap::new()).unwrap();
        let diff = max_abs_diff(&before.outputs[0], &after.outputs[0]).unwrap();
        assert!(diff < 1e-5, "folded output differs by {diff}");
    }

    #[test]
    fn groupnorm_resize_gridsample_matmul_quantization_numerics() {
        // GroupNorm: two groups of two channels, unit variance per group.
        let shape = TensorShape::new(1, 4, 2, 2);
        let vals: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        ];
        let mut g = Graph::new();
        let xid = g.add_constant(TensorBuffer::from_f32(shape, &vals).unwrap());
        let s = g.add_constant(param(4, &[1.0; 4]));
        let o = g.add_constant(param(4, &[0.0; 4]));
        let gn_out = g.fresh_id();
        g.add_node(Node::new(
            OpKind::GroupNorm { groups: 2, scale: s, offset: o, epsilon: 1e-5 },
            vec![xid], gn_out, "gn",
        ));
        g.set_outputs(vec![gn_out]);
        let gn = execute_graph(&g, &HashMap::new()).unwrap().outputs[0].f32_values().unwrap();
        let mean: f32 = gn.iter().sum::<f32>() / gn.len() as f32;
        assert!(mean.abs() < 1e-5, "GroupNorm mean {mean}");
        let var: f32 = gn.iter().map(|v| v * v).sum::<f32>() / gn.len() as f32;
        assert!((var - 1.0).abs() < 1e-3, "GroupNorm variance {var}");

        // ResizeBilinear: a constant field stays constant after 2x upsampling.
        let mut g2 = Graph::new();
        let c = g2.add_constant(
            TensorBuffer::from_f32(TensorShape::new(1, 1, 2, 2), &[2.5; 4]).unwrap(),
        );
        let out = g2.fresh_id();
        g2.add_node(Node::new(
            OpKind::ResizeBilinear { scale_h: 2.0, scale_w: 2.0, align_corners: false },
            vec![c], out, "resize",
        ));
        g2.set_outputs(vec![out]);
        let r2 = execute_graph(&g2, &HashMap::new()).unwrap();
        assert_eq!(r2.outputs[0].shape, TensorShape::new(1, 1, 4, 4));
        assert!(r2.outputs[0].f32_values().unwrap().iter().all(|&v| (v - 2.5).abs() < 1e-6));

        // GridSample: pixel-center coordinates reproduce the source pixels.
        let mut g3 = Graph::new();
        let src = g3.add_constant(
            TensorBuffer::from_f32(TensorShape::new(1, 1, 2, 2), &[1.0, 2.0, 3.0, 4.0]).unwrap(),
        );
        // NCHW grid: x = [-0.5, 0.5, -0.5, 0.5], y = [-0.5, -0.5, 0.5, 0.5]
        let grid_vals = [-0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5];
        let grid = g3.add_constant(
            TensorBuffer::from_f32(TensorShape::new(1, 2, 2, 2), &grid_vals).unwrap(),
        );
        let out = g3.fresh_id();
        g3.add_node(Node::new(OpKind::GridSample, vec![src, grid], out, "grid"));
        g3.set_outputs(vec![out]);
        let sampled = execute_graph(&g3, &HashMap::new()).unwrap().outputs[0]
            .f32_values().unwrap();
        assert_eq!(sampled.len(), 4);
        for (got, want) in sampled.iter().zip([1.0, 2.0, 3.0, 4.0]) {
            assert!((got - want).abs() < 1e-5, "grid sample {got} vs {want}");
        }

        // MatMul: [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]].
        let mut g4 = Graph::new();
        let a = g4.add_constant(
            TensorBuffer::from_f32(TensorShape::new(1, 2, 1, 2), &[1.0, 2.0, 3.0, 4.0]).unwrap(),
        );
        let b = g4.add_constant(
            TensorBuffer::from_f32(TensorShape::new(1, 2, 2, 1), &[5.0, 7.0, 6.0, 8.0]).unwrap(),
        );
        let out = g4.fresh_id();
        g4.add_node(Node::new(OpKind::MatMul, vec![a, b], out, "mm"));
        g4.set_outputs(vec![out]);
        let r4 = execute_graph(&g4, &HashMap::new()).unwrap();
        assert_eq!(r4.outputs[0].f32_values().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);

        // Quantize / dequantize round-trip within one quantization step.
        let mut g5 = Graph::new();
        let qin = g5.add_constant(
            TensorBuffer::from_f32(TensorShape::vector(4), &[0.0, 0.25, -0.25, 1.0]).unwrap(),
        );
        let qout = g5.fresh_id();
        g5.add_node(Node::new(
            OpKind::QuantizeLinear { scale: 0.125, zero_point: 0 }, vec![qin], qout, "q",
        ));
        let dqout = g5.fresh_id();
        g5.add_node(Node::new(
            OpKind::DequantizeLinear { scale: 0.125, zero_point: 0 }, vec![qout], dqout, "dq",
        ));
        g5.set_outputs(vec![qout, dqout]);
        let r5 = execute_graph(&g5, &HashMap::new()).unwrap();
        assert_eq!(r5.outputs[0].dtype, DType::Int8);
        assert_eq!(r5.outputs[0].bytes, vec![0, 2, 254, 8]);
        assert_eq!(r5.outputs[1].f32_values().unwrap(), vec![0.0, 0.25, -0.25, 1.0]);
    }

    #[test]
    fn memory_plan_reuses_slots_and_preserves_values() {
        // Residual graph with a long-lived skip branch and short-lived
        // normalization constants.
        let mut g = Graph::new();
        let x = g.add_constant(
            TensorBuffer::from_f32(TensorShape::new(1, 2, 4, 4), &vec![0.5; 32]).unwrap(),
        );
        let s = g.add_constant(param(2, &[1.0, 1.0]));
        let o = g.add_constant(param(2, &[0.0, 0.0]));
        let m = g.add_constant(param(2, &[0.0, 0.0]));
        let v = g.add_constant(param(2, &[1.0, 1.0]));
        let skip = g.fresh_id();
        g.add_node(Node::new(OpKind::Relu, vec![x], skip, "skip"));
        let normed = g.fresh_id();
        g.add_node(Node::new(
            OpKind::BatchNorm { scale: s, offset: o, mean: m, var: v, epsilon: 1e-5 },
            vec![x], normed, "bn",
        ));
        let activated = g.fresh_id();
        g.add_node(Node::new(OpKind::SiLU, vec![normed], activated, "silu"));
        let skip_up = g.fresh_id();
        g.add_node(Node::new(
            OpKind::ResizeBilinear { scale_h: 1.0, scale_w: 1.0, align_corners: false },
            vec![skip], skip_up, "skip_identity",
        ));
        let sum = g.fresh_id();
        g.add_node(Node::new(OpKind::Add, vec![skip_up, activated], sum, "residual"));
        g.set_outputs(vec![sum]);

        let plan = plan_memory(&g).unwrap();
        assert_eq!(plan.total_tensors, 10);
        assert!(plan.slot_count < plan.total_tensors);
        assert!(plan.compression_ratio() < 0.8, "ratio {}", plan.compression_ratio());
        // Strictly overlapping live ranges never share a slot: no write can
        // clobber a value that still has readers.
        for a in &plan.lifetimes {
            for b in &plan.lifetimes {
                if a.tensor != b.tensor && a.start < b.end && b.start < a.end {
                    assert_ne!(a.slot, b.slot, "overlapping lifetimes share slot {}", a.slot);
                }
            }
        }
        let res = execute_graph(&g, &HashMap::new()).unwrap();
        let values = res.outputs[0].f32_values().unwrap();
        let got = values.first().copied().unwrap_or(0.0);
        // Skip branch is ReLU(0.5) = 0.5. The BN branch applies epsilon:
        // BN(0.5) = 0.5 / sqrt(1 + 1e-5), then SiLU of that value.
        let bn_value = 0.5 / (1.0f32 + 1e-5).sqrt();
        let silu = bn_value / (1.0 + (-bn_value).exp());
        let expected = 0.5 + silu;
        assert!(
            (got - expected).abs() < 1e-6,
            "residual got {got}, expected {expected} (silu={silu})"
        );
        assert!(values.iter().all(|&v| (v - got).abs() < 1e-7));
    }

    #[test]
    fn mini_portrait_residual_network_runs_end_to_end() {
        // Conv+BN+SiLU -> DepthwiseConv -> GroupNorm -> ResizeBilinear
        // -> Residual Add -> Sigmoid, all on FP32 CPU kernels.
        let mut g = Graph::new();
        let x = g.add_constant(
            TensorBuffer::from_f32(TensorShape::new(1, 2, 4, 4), &vec![0.2f32; 32]).unwrap(),
        );
        let w1 = g.add_constant(
            TensorBuffer::from_f32(TensorShape::new(2, 2, 3, 3), &vec![0.1f32; 36]).unwrap(),
        );
        let b1 = g.add_constant(param(2, &[0.0, 0.0]));
        let s1 = g.add_constant(param(2, &[1.0, 1.0]));
        let o1 = g.add_constant(param(2, &[0.0, 0.0]));
        let m1 = g.add_constant(param(2, &[0.0, 0.0]));
        let v1 = g.add_constant(param(2, &[1.0, 1.0]));
        let c1 = g.fresh_id();
        g.add_node(Node::new(
            OpKind::Conv2d {
                stride: (1, 1), pad: (1, 1, 1, 1), dilation: (1, 1),
                weight: w1, bias: Some(b1),
            },
            vec![x], c1, "conv1",
        ));
        let n1 = g.fresh_id();
        g.add_node(Node::new(
            OpKind::BatchNorm { scale: s1, offset: o1, mean: m1, var: v1, epsilon: 1e-5 },
            vec![c1], n1, "bn1",
        ));
        let a1 = g.fresh_id();
        g.add_node(Node::new(OpKind::SiLU, vec![n1], a1, "silu1"));
        let dw_w = g.add_constant(
            TensorBuffer::from_f32(TensorShape::new(2, 1, 3, 3), &vec![0.05f32; 18]).unwrap(),
        );
        let dw = g.fresh_id();
        g.add_node(Node::new(
            OpKind::DepthwiseConv2d {
                stride: (1, 1), pad: (1, 1, 1, 1), weight: dw_w, bias: None,
            },
            vec![a1], dw, "dw",
        ));
        let gs = g.add_constant(param(2, &[1.0, 1.0]));
        let go = g.add_constant(param(2, &[0.0, 0.0]));
        let gn = g.fresh_id();
        g.add_node(Node::new(
            OpKind::GroupNorm { groups: 2, scale: gs, offset: go, epsilon: 1e-5 },
            vec![dw], gn, "gn",
        ));
        let skip_up = g.fresh_id();
        g.add_node(Node::new(
            OpKind::ResizeBilinear { scale_h: 2.0, scale_w: 2.0, align_corners: false },
            vec![gn], skip_up, "skip_up",
        ));
        let up = g.fresh_id();
        g.add_node(Node::new(
            OpKind::ResizeBilinear { scale_h: 2.0, scale_w: 2.0, align_corners: false },
            vec![a1], up, "up",
        ));
        let sum = g.fresh_id();
        // The upsampled activation is added to the upsampled skip.
        g.add_node(Node::new(OpKind::Add, vec![up, skip_up], sum, "residual"));
        let out = g.fresh_id();
        g.add_node(Node::new(OpKind::Sigmoid, vec![sum], out, "sigmoid"));
        g.set_outputs(vec![out]);

        let res = execute_graph(&g, &HashMap::new()).unwrap();
        let values = res.outputs[0].f32_values().unwrap();
        assert_eq!(res.outputs[0].shape, TensorShape::new(1, 2, 8, 8));
        assert!(values.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        assert!(values.iter().any(|&v| v > 0.5));
    }
}
