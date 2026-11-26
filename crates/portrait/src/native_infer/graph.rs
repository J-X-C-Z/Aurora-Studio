//! Strongly-typed IR graph, static topology validation, and shape inference.
//!
//! Covers all 59 distinct ONNX operators discovered in the Aurora Studio model
//! inventory (58 main-graph ops + `Abs` inside `If` subgraphs).

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::native_infer::onnx_proto::{
    DataType, DimSpec, GraphProto, ModelProto, NodeProto, TensorProto,
};

/// Dense index identifying a tensor in [`IrGraph::tensors`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub usize);

/// Dense index identifying a node in [`IrGraph::nodes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphError {
    DuplicateTensorName(String),
    UnknownTensor(String),
    TopologicalViolation { node: String, input: String },
    UnsupportedOp { op_type: String, node: String },
    MissingAttribute { op_type: String, attr: &'static str },
    InvalidAttribute { op_type: String, detail: String },
    ShapeInferenceFailed { node: String, detail: String },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateTensorName(n) => write!(f, "duplicate SSA tensor name: {n}"),
            Self::UnknownTensor(n) => write!(f, "unknown tensor referenced in graph: {n}"),
            Self::TopologicalViolation { node, input } => {
                write!(f, "topological violation at node '{node}': input '{input}' not yet defined")
            }
            Self::UnsupportedOp { op_type, node } => {
                write!(f, "unsupported ONNX op '{op_type}' at node '{node}'")
            }
            Self::MissingAttribute { op_type, attr } => {
                write!(f, "missing required attribute '{attr}' for op '{op_type}'")
            }
            Self::InvalidAttribute { op_type, detail } => {
                write!(f, "invalid attribute for op '{op_type}': {detail}")
            }
            Self::ShapeInferenceFailed { node, detail } => {
                write!(f, "shape inference failed at node '{node}': {detail}")
            }
        }
    }
}

impl std::error::Error for GraphError {}

// ---------------------------------------------------------------------------
// Strongly-typed operator attributes
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvAttrs {
    pub kernel_shape: Option<Vec<usize>>,
    pub strides: Vec<usize>,
    pub pads: Vec<usize>,
    pub dilations: Vec<usize>,
    pub group: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvTransposeAttrs {
    pub kernel_shape: Option<Vec<usize>>,
    pub strides: Vec<usize>,
    pub pads: Vec<usize>,
    pub dilations: Vec<usize>,
    pub group: usize,
    pub output_padding: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolAttrs {
    pub kernel_shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub pads: Vec<usize>,
    pub dilations: Vec<usize>,
    pub ceil_mode: bool,
    pub count_include_pad: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeMode {
    Nearest,
    Linear,
    Cubic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordTransformMode {
    HalfPixel,
    Asymmetric,
    AlignCorners,
    TfHalfPixelForNn,
    PytorchHalfPixel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NearestMode {
    Floor,
    RoundPreferFloor,
    RoundPreferCeil,
    Ceil,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResizeAttrs {
    pub mode: ResizeMode,
    pub coord_mode: CoordTransformMode,
    pub nearest_mode: NearestMode,
    pub cubic_coeff_a: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BatchNormAttrs {
    pub epsilon: f32,
    pub momentum: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerNormAttrs {
    pub axis: i64,
    pub epsilon: f32,
    pub stash_type: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GemmAttrs {
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: bool,
    pub trans_b: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReduceAttrs {
    pub axes: Option<Vec<i64>>,
    pub keepdims: bool,
    pub noop_with_empty_axes: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IfAttrs {
    pub then_branch: Box<IrGraph>,
    pub else_branch: Box<IrGraph>,
}

/// Complete strongly-typed operator set covering all 59 audited ONNX ops
/// plus internal fused ops produced by optimization passes.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    // Convolution & Matrix Multiplication
    Conv(ConvAttrs),
    ConvTranspose(ConvTransposeAttrs),
    ConvInteger(ConvAttrs),
    MatMul,
    MatMulInteger,
    Gemm(GemmAttrs),

    // Normalization
    BatchNormalization(BatchNormAttrs),
    /// Standalone BN after static pre-fusion into `(scale_eff, bias_eff)`.
    BatchNormFused,
    LayerNormalization(LayerNormAttrs),
    InstanceNormalization { epsilon: f32 },

    // Pooling & Reduction
    MaxPool(PoolAttrs),
    AveragePool(PoolAttrs),
    GlobalAveragePool,
    ReduceMean(ReduceAttrs),
    ReduceSum(ReduceAttrs),
    ReduceMax(ReduceAttrs),

    // Spatial / Resize / Pad
    Resize(ResizeAttrs),
    Pad { mode: String },

    // Elementwise Arithmetic & Math
    Add,
    Sub,
    Mul,
    Div,
    Mod { fmod: bool },
    Pow,
    Sqrt,
    Reciprocal,
    Abs,
    Sin,
    Cos,

    // Activations
    Relu,
    PRelu,
    LeakyRelu { alpha: f32 },
    Sigmoid,
    HardSwish,
    HardSigmoid { alpha: f32, beta: f32 },
    Erf,
    Clip,
    Softmax { axis: i64 },

    // Comparisons & Logic
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    And,
    Where,

    // Shape, Layout & Indexing
    Transpose { perm: Vec<usize> },
    Reshape { allowzero: bool },
    Concat { axis: i64 },
    Split { axis: i64, num_outputs: Option<usize> },
    Slice,
    Gather { axis: i64 },
    ScatterND,
    Expand,
    Tile,
    Squeeze { axes: Option<Vec<i64>> },
    Unsqueeze { axes: Option<Vec<i64>> },
    Shape { start: Option<i64>, end: Option<i64> },
    Size,
    Cast { to: DataType },

    // Constants, Quantization & Control Flow
    Constant { value: TensorProto },
    ConstantOfShape { value: TensorProto },
    Identity,
    DynamicQuantizeLinear,
    If(IfAttrs),
}

impl Op {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Conv(_) => "Conv",
            Self::ConvTranspose(_) => "ConvTranspose",
            Self::ConvInteger(_) => "ConvInteger",
            Self::MatMul => "MatMul",
            Self::MatMulInteger => "MatMulInteger",
            Self::Gemm(_) => "Gemm",
            Self::BatchNormalization(_) => "BatchNormalization",
            Self::BatchNormFused => "BatchNormFused",
            Self::LayerNormalization(_) => "LayerNormalization",
            Self::InstanceNormalization { .. } => "InstanceNormalization",
            Self::MaxPool(_) => "MaxPool",
            Self::AveragePool(_) => "AveragePool",
            Self::GlobalAveragePool => "GlobalAveragePool",
            Self::ReduceMean(_) => "ReduceMean",
            Self::ReduceSum(_) => "ReduceSum",
            Self::ReduceMax(_) => "ReduceMax",
            Self::Resize(_) => "Resize",
            Self::Pad { .. } => "Pad",
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mul => "Mul",
            Self::Div => "Div",
            Self::Mod { .. } => "Mod",
            Self::Pow => "Pow",
            Self::Sqrt => "Sqrt",
            Self::Reciprocal => "Reciprocal",
            Self::Abs => "Abs",
            Self::Sin => "Sin",
            Self::Cos => "Cos",
            Self::Relu => "Relu",
            Self::PRelu => "PRelu",
            Self::LeakyRelu { .. } => "LeakyRelu",
            Self::Sigmoid => "Sigmoid",
            Self::HardSwish => "HardSwish",
            Self::HardSigmoid { .. } => "HardSigmoid",
            Self::Erf => "Erf",
            Self::Clip => "Clip",
            Self::Softmax { .. } => "Softmax",
            Self::Equal => "Equal",
            Self::Greater => "Greater",
            Self::GreaterOrEqual => "GreaterOrEqual",
            Self::Less => "Less",
            Self::And => "And",
            Self::Where => "Where",
            Self::Transpose { .. } => "Transpose",
            Self::Reshape { .. } => "Reshape",
            Self::Concat { .. } => "Concat",
            Self::Split { .. } => "Split",
            Self::Slice => "Slice",
            Self::Gather { .. } => "Gather",
            Self::ScatterND => "ScatterND",
            Self::Expand => "Expand",
            Self::Tile => "Tile",
            Self::Squeeze { .. } => "Squeeze",
            Self::Unsqueeze { .. } => "Unsqueeze",
            Self::Shape { .. } => "Shape",
            Self::Size => "Size",
            Self::Cast { .. } => "Cast",
            Self::Constant { .. } => "Constant",
            Self::ConstantOfShape { .. } => "ConstantOfShape",
            Self::Identity => "Identity",
            Self::DynamicQuantizeLinear => "DynamicQuantizeLinear",
            Self::If(_) => "If",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub id: TensorId,
    pub name: String,
    pub dtype: DataType,
    /// `None` means unknown/dynamic dimension; `Some(v)` means statically known.
    pub shape: Option<Vec<Option<usize>>>,
    pub is_weight: bool,
    pub is_graph_input: bool,
    pub is_graph_output: bool,
}

impl TensorInfo {
    /// Returns concrete shape if all dimensions are statically known.
    pub fn static_shape(&self) -> Option<Vec<usize>> {
        let dims = self.shape.as_ref()?;
        dims.iter().copied().collect()
    }

    /// Total byte size if both `dtype` and all dimensions are statically known.
    pub fn static_byte_size(&self) -> Option<usize> {
        let elem_bytes = self.dtype.byte_size()?;
        let shape = self.static_shape()?;
        let elems = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))?;
        elems.checked_mul(elem_bytes)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IrNode {
    pub id: NodeId,
    pub name: String,
    pub op: Op,
    /// `None` represents an omitted optional ONNX input (`""`).
    pub inputs: Vec<Option<TensorId>>,
    pub outputs: Vec<Option<TensorId>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IrGraph {
    pub name: String,
    pub tensors: Vec<TensorInfo>,
    pub nodes: Vec<IrNode>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub weights: HashMap<TensorId, TensorProto>,
    pub name_to_id: HashMap<String, TensorId>,
}

impl IrGraph {
    pub fn from_model_proto(model: &ModelProto) -> Result<Self, GraphError> {
        Self::from_graph_proto(&model.graph, None)
    }

    fn from_graph_proto(
        gp: &GraphProto,
        outer_scope: Option<&HashMap<String, TensorInfo>>,
    ) -> Result<Self, GraphError> {
        let mut graph = Self {
            name: gp.name.clone(),
            tensors: Vec::new(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            weights: HashMap::new(),
            name_to_id: HashMap::new(),
        };

        // 1. Register initializers (weights).
        let mut init_names = HashSet::new();
        for init in &gp.initializers {
            init_names.insert(init.name.clone());
            let shape = Some(
                init.dims
                    .iter()
                    .map(|&d| if d >= 0 { Some(d as usize) } else { None })
                    .collect(),
            );
            let id = graph.register_tensor(
                init.name.clone(),
                init.data_type,
                shape,
                true,
                false,
                false,
            );
            graph.weights.insert(id, init.clone());
        }

        // 2. Register graph inputs (excluding initializers listed in graph.inputs).
        for inp in &gp.inputs {
            if init_names.contains(&inp.name) {
                continue;
            }
            let shape = dims_from_spec(&inp.dims);
            let id = graph.register_tensor(
                inp.name.clone(),
                inp.data_type,
                shape,
                false,
                true,
                false,
            );
            graph.inputs.push(id);
        }

        // 3. Build nodes in proto order.
        for (idx, np) in gp.nodes.iter().enumerate() {
            let mut input_ids = Vec::with_capacity(np.inputs.len());
            for inp_name in &np.inputs {
                if inp_name.is_empty() {
                    input_ids.push(None);
                } else if let Some(&id) = graph.name_to_id.get(inp_name) {
                    input_ids.push(Some(id));
                } else if let Some(outer) = outer_scope.and_then(|s| s.get(inp_name)) {
                    // Subgraph capturing a tensor from outer graph scope.
                    let id = graph.register_tensor(
                        outer.name.clone(),
                        outer.dtype,
                        outer.shape.clone(),
                        outer.is_weight,
                        true,
                        false,
                    );
                    input_ids.push(Some(id));
                } else {
                    // Forward reference or implicit tensor; topology validator catches cycles.
                    let id = graph.register_tensor(
                        inp_name.clone(),
                        DataType::Undefined,
                        None,
                        false,
                        false,
                        false,
                    );
                    input_ids.push(Some(id));
                }
            }

            let op = parse_op(np, &graph)?;

            let mut output_ids = Vec::with_capacity(np.outputs.len());
            for out_name in &np.outputs {
                if out_name.is_empty() {
                    output_ids.push(None);
                } else {
                    let id = if let Some(&existing) = graph.name_to_id.get(out_name) {
                        existing
                    } else {
                        graph.register_tensor(
                            out_name.clone(),
                            DataType::Undefined,
                            None,
                            false,
                            false,
                            false,
                        )
                    };
                    output_ids.push(Some(id));
                }
            }

            // If this is a Constant node, also cache its payload in `weights` so
            // shape inference & folding can read it uniformly.
            if let Op::Constant { value } = &op {
                if let Some(Some(out_id)) = output_ids.first() {
                    let info = &mut graph.tensors[out_id.0];
                    info.dtype = value.data_type;
                    info.shape = Some(
                        value
                            .dims
                            .iter()
                            .map(|&d| if d >= 0 { Some(d as usize) } else { None })
                            .collect(),
                    );
                    info.is_weight = true;
                    graph.weights.insert(*out_id, value.clone());
                }
            }

            graph.nodes.push(IrNode {
                id: NodeId(idx),
                name: if np.name.is_empty() {
                    format!("{}_{}", np.op_type, idx)
                } else {
                    np.name.clone()
                },
                op,
                inputs: input_ids,
                outputs: output_ids,
            });
        }

        // 4. Mark graph outputs.
        for out in &gp.outputs {
            let id = if let Some(&existing) = graph.name_to_id.get(&out.name) {
                let info = &mut graph.tensors[existing.0];
                info.is_graph_output = true;
                if info.dtype == DataType::Undefined {
                    info.dtype = out.data_type;
                }
                if info.shape.is_none() {
                    info.shape = dims_from_spec(&out.dims);
                }
                existing
            } else {
                graph.register_tensor(
                    out.name.clone(),
                    out.data_type,
                    dims_from_spec(&out.dims),
                    false,
                    false,
                    true,
                )
            };
            graph.outputs.push(id);
        }

        Ok(graph)
    }

    pub fn register_tensor(
        &mut self,
        name: String,
        dtype: DataType,
        shape: Option<Vec<Option<usize>>>,
        is_weight: bool,
        is_graph_input: bool,
        is_graph_output: bool,
    ) -> TensorId {
        if let Some(&id) = self.name_to_id.get(&name) {
            return id;
        }
        let id = TensorId(self.tensors.len());
        self.name_to_id.insert(name.clone(), id);
        self.tensors.push(TensorInfo {
            id,
            name,
            dtype,
            shape,
            is_weight,
            is_graph_input,
            is_graph_output,
        });
        id
    }

    /// Add a new synthetic weight tensor (e.g. folded bias) with a unique name.
    pub fn add_weight(&mut self, mut tensor: TensorProto) -> TensorId {
        while self.name_to_id.contains_key(&tensor.name) {
            tensor.name.push_str("_folded");
        }
        let shape = Some(
            tensor
                .dims
                .iter()
                .map(|&d| if d >= 0 { Some(d as usize) } else { None })
                .collect(),
        );
        let id = self.register_tensor(
            tensor.name.clone(),
            tensor.data_type,
            shape,
            true,
            false,
            false,
        );
        self.weights.insert(id, tensor);
        id
    }

    /// Validate SSA topological order: every node input must be defined either
    /// as a graph input, initializer, or an output of an earlier node.
    pub fn validate_topology(&self) -> Result<(), GraphError> {
        let mut defined: HashSet<TensorId> = HashSet::new();
        for &id in &self.inputs {
            defined.insert(id);
        }
        for &id in self.weights.keys() {
            defined.insert(id);
        }

        for node in &self.nodes {
            for inp in node.inputs.iter().flatten() {
                if !defined.contains(inp) {
                    return Err(GraphError::TopologicalViolation {
                        node: node.name.clone(),
                        input: self.tensors[inp.0].name.clone(),
                    });
                }
            }
            for out in node.outputs.iter().flatten() {
                defined.insert(*out);
            }
        }

        for &out_id in &self.outputs {
            if !defined.contains(&out_id) {
                return Err(GraphError::UnknownTensor(
                    self.tensors[out_id.0].name.clone(),
                ));
            }
        }
        Ok(())
    }

    /// Static shape and dtype propagation across the graph.
    ///
    /// `input_overrides` lets callers supply concrete dimensions for dynamic
    /// inputs (e.g. `batch_size = 1` or `H = 640, W = 640`).
    /// Returns the number of tensors with fully resolved static shapes.
    pub fn infer_shapes(
        &mut self,
        input_overrides: &HashMap<String, Vec<usize>>,
    ) -> Result<usize, GraphError> {
        // Apply user-specified input overrides.
        for (name, dims) in input_overrides {
            if let Some(&id) = self.name_to_id.get(name) {
                self.tensors[id.0].shape = Some(dims.iter().map(|&d| Some(d)).collect());
            }
        }

        // Small integer constant cache for shape-calculation subgraphs
        // (`Shape`, `Gather`, `Unsqueeze`, `Squeeze`, `Concat`, `Constant`).
        let mut const_i64: HashMap<TensorId, Vec<i64>> = HashMap::new();
        for (&id, tp) in &self.weights {
            if matches!(tp.data_type, DataType::Int64 | DataType::Int32) && tp.elem_count() <= 64 {
                if let Ok(vals) = tp.to_i64_vec() {
                    const_i64.insert(id, vals);
                }
            }
        }

        for node_idx in 0..self.nodes.len() {
            let node = self.nodes[node_idx].clone();
            let in_shapes: Vec<Option<Vec<usize>>> = node
                .inputs
                .iter()
                .map(|opt| opt.and_then(|id| self.tensors[id.0].static_shape()))
                .collect();
            let in_dtypes: Vec<DataType> = node
                .inputs
                .iter()
                .map(|opt| {
                    opt.map(|id| self.tensors[id.0].dtype)
                        .unwrap_or(DataType::Undefined)
                })
                .collect();

            let primary_dtype = in_dtypes.first().copied().unwrap_or(DataType::Float);
            let out0_id = node.outputs.first().copied().flatten();

            match &node.op {
                Op::Constant { value } => {
                    if let Some(oid) = out0_id {
                        let dims: Vec<usize> = value.dims.iter().map(|&d| d.max(0) as usize).collect();
                        self.set_static_tensor(oid, value.data_type, dims);
                        if matches!(value.data_type, DataType::Int64 | DataType::Int32)
                            && value.elem_count() <= 64
                        {
                            if let Ok(v) = value.to_i64_vec() {
                                const_i64.insert(oid, v);
                            }
                        }
                    }
                }

                // Unary shape-preserving ops
                Op::Relu
                | Op::PRelu
                | Op::LeakyRelu { .. }
                | Op::Sigmoid
                | Op::HardSwish
                | Op::HardSigmoid { .. }
                | Op::Erf
                | Op::Clip
                | Op::Sqrt
                | Op::Reciprocal
                | Op::Abs
                | Op::Sin
                | Op::Cos
                | Op::Softmax { .. }
                | Op::Identity
                | Op::BatchNormalization(_)
                | Op::BatchNormFused
                | Op::InstanceNormalization { .. } => {
                    if let (Some(oid), Some(s)) = (out0_id, in_shapes.first().and_then(|x| x.as_ref())) {
                        self.set_static_tensor(oid, primary_dtype, s.clone());
                        if matches!(node.op, Op::Identity) {
                            if let Some(Some(in0)) = node.inputs.first() {
                                if let Some(v) = const_i64.get(in0).cloned() {
                                    const_i64.insert(oid, v);
                                }
                            }
                        }
                    }
                }

                Op::LayerNormalization(_) => {
                    if let (Some(oid), Some(s)) = (out0_id, in_shapes.first().and_then(|x| x.as_ref())) {
                        self.set_static_tensor(oid, primary_dtype, s.clone());
                    }
                }

                Op::Cast { to } => {
                    if let (Some(oid), Some(s)) = (out0_id, in_shapes.first().and_then(|x| x.as_ref())) {
                        self.set_static_tensor(oid, *to, s.clone());
                    }
                }

                // Binary broadcast ops
                Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Mod { .. } | Op::Pow => {
                    if let (Some(oid), Some(a), Some(b)) = (
                        out0_id,
                        in_shapes.first().and_then(|x| x.as_ref()),
                        in_shapes.get(1).and_then(|x| x.as_ref()),
                    ) {
                        if let Some(out_s) = broadcast_shape(a, b) {
                            self.set_static_tensor(oid, primary_dtype, out_s);
                        }
                    }
                }

                // Comparison / logic ops -> Bool
                Op::Equal | Op::Greater | Op::GreaterOrEqual | Op::Less | Op::And => {
                    if let (Some(oid), Some(a), Some(b)) = (
                        out0_id,
                        in_shapes.first().and_then(|x| x.as_ref()),
                        in_shapes.get(1).and_then(|x| x.as_ref()),
                    ) {
                        if let Some(out_s) = broadcast_shape(a, b) {
                            self.set_static_tensor(oid, DataType::Bool, out_s);
                        }
                    }
                }

                Op::Where => {
                    if let (Some(oid), Some(c), Some(x), Some(y)) = (
                        out0_id,
                        in_shapes.first().and_then(|s| s.as_ref()),
                        in_shapes.get(1).and_then(|s| s.as_ref()),
                        in_shapes.get(2).and_then(|s| s.as_ref()),
                    ) {
                        if let Some(xy) = broadcast_shape(x, y) {
                            if let Some(out_s) = broadcast_shape(c, &xy) {
                                let dt = in_dtypes.get(1).copied().unwrap_or(DataType::Float);
                                self.set_static_tensor(oid, dt, out_s);
                            }
                        }
                    }
                }

                Op::Conv(attrs) | Op::ConvInteger(attrs) => {
                    if let (Some(oid), Some(x), Some(w)) = (
                        out0_id,
                        in_shapes.first().and_then(|s| s.as_ref()),
                        in_shapes.get(1).and_then(|s| s.as_ref()),
                    ) {
                        if x.len() == 4 && w.len() == 4 {
                            let n = x[0];
                            let c_out = w[0];
                            let kh = attrs.kernel_shape.as_ref().map(|k| k[0]).unwrap_or(w[2]);
                            let kw = attrs.kernel_shape.as_ref().map(|k| k[1]).unwrap_or(w[3]);
                            let sh = *attrs.strides.first().unwrap_or(&1);
                            let sw = *attrs.strides.get(1).unwrap_or(&1);
                            let dh = *attrs.dilations.first().unwrap_or(&1);
                            let dw = *attrs.dilations.get(1).unwrap_or(&1);
                            let pt = *attrs.pads.first().unwrap_or(&0);
                            let pl = *attrs.pads.get(1).unwrap_or(&0);
                            let pb = *attrs.pads.get(2).unwrap_or(&pt);
                            let pr = *attrs.pads.get(3).unwrap_or(&pl);
                            let eff_kh = dh * (kh - 1) + 1;
                            let eff_kw = dw * (kw - 1) + 1;
                            if x[2] + pt + pb >= eff_kh && x[3] + pl + pr >= eff_kw {
                                let oh = (x[2] + pt + pb - eff_kh) / sh + 1;
                                let ow = (x[3] + pl + pr - eff_kw) / sw + 1;
                                let dt = if matches!(node.op, Op::ConvInteger(_)) {
                                    DataType::Int32
                                } else {
                                    primary_dtype
                                };
                                self.set_static_tensor(oid, dt, vec![n, c_out, oh, ow]);
                            }
                        }
                    }
                }

                Op::ConvTranspose(attrs) => {
                    if let (Some(oid), Some(x), Some(w)) = (
                        out0_id,
                        in_shapes.first().and_then(|s| s.as_ref()),
                        in_shapes.get(1).and_then(|s| s.as_ref()),
                    ) {
                        if x.len() == 4 && w.len() == 4 {
                            let n = x[0];
                            let c_out = w[1] * attrs.group;
                            let kh = attrs.kernel_shape.as_ref().map(|k| k[0]).unwrap_or(w[2]);
                            let kw = attrs.kernel_shape.as_ref().map(|k| k[1]).unwrap_or(w[3]);
                            let sh = *attrs.strides.first().unwrap_or(&1);
                            let sw = *attrs.strides.get(1).unwrap_or(&1);
                            let dh = *attrs.dilations.first().unwrap_or(&1);
                            let dw = *attrs.dilations.get(1).unwrap_or(&1);
                            let pt = *attrs.pads.first().unwrap_or(&0);
                            let pl = *attrs.pads.get(1).unwrap_or(&0);
                            let pb = *attrs.pads.get(2).unwrap_or(&pt);
                            let pr = *attrs.pads.get(3).unwrap_or(&pl);
                            let oph = *attrs.output_padding.first().unwrap_or(&0);
                            let opw = *attrs.output_padding.get(1).unwrap_or(&0);
                            let oh = sh * (x[2] - 1) + oph + dh * (kh - 1) + 1 - pt - pb;
                            let ow = sw * (x[3] - 1) + opw + dw * (kw - 1) + 1 - pl - pr;
                            self.set_static_tensor(oid, primary_dtype, vec![n, c_out, oh, ow]);
                        }
                    }
                }

                Op::MaxPool(attrs) | Op::AveragePool(attrs) => {
                    if let (Some(oid), Some(x)) = (out0_id, in_shapes.first().and_then(|s| s.as_ref())) {
                        if x.len() == 4 && attrs.kernel_shape.len() == 2 {
                            let kh = attrs.kernel_shape[0];
                            let kw = attrs.kernel_shape[1];
                            let sh = *attrs.strides.first().unwrap_or(&1);
                            let sw = *attrs.strides.get(1).unwrap_or(&1);
                            let dh = *attrs.dilations.first().unwrap_or(&1);
                            let dw = *attrs.dilations.get(1).unwrap_or(&1);
                            let pt = *attrs.pads.first().unwrap_or(&0);
                            let pl = *attrs.pads.get(1).unwrap_or(&0);
                            let pb = *attrs.pads.get(2).unwrap_or(&pt);
                            let pr = *attrs.pads.get(3).unwrap_or(&pl);
                            let eff_kh = dh * (kh - 1) + 1;
                            let eff_kw = dw * (kw - 1) + 1;
                            let num_h = (x[2] + pt + pb).saturating_sub(eff_kh);
                            let num_w = (x[3] + pl + pr).saturating_sub(eff_kw);
                            let oh = if attrs.ceil_mode {
                                num_h.div_ceil(sh) + 1
                            } else {
                                num_h / sh + 1
                            };
                            let ow = if attrs.ceil_mode {
                                num_w.div_ceil(sw) + 1
                            } else {
                                num_w / sw + 1
                            };
                            self.set_static_tensor(oid, primary_dtype, vec![x[0], x[1], oh, ow]);
                        }
                    }
                }

                Op::GlobalAveragePool => {
                    if let (Some(oid), Some(x)) = (out0_id, in_shapes.first().and_then(|s| s.as_ref())) {
                        if x.len() >= 2 {
                            let mut out = vec![1usize; x.len()];
                            out[0] = x[0];
                            out[1] = x[1];
                            self.set_static_tensor(oid, primary_dtype, out);
                        }
                    }
                }

                Op::ReduceMean(attrs) | Op::ReduceSum(attrs) | Op::ReduceMax(attrs) => {
                    if let (Some(oid), Some(x)) = (out0_id, in_shapes.first().and_then(|s| s.as_ref())) {
                        let axes_opt = attrs.axes.clone().or_else(|| {
                            match node.inputs.get(1).and_then(|input| *input) {
                                Some(id) => const_i64.get(&id).cloned(),
                                None => Some(Vec::new()),
                            }
                        });
                        if let Some(axes) = axes_opt {
                            let rank = x.len() as i64;
                            let reduce_all = axes.is_empty() && !attrs.noop_with_empty_axes;
                            let axis_set: HashSet<usize> = axes
                                .iter()
                                .map(|&axis| if axis < 0 { axis + rank } else { axis })
                                .map(|axis| axis as usize)
                                .collect();
                            if axis_set.iter().any(|&axis| axis >= x.len()) {
                                continue;
                            }
                            let mut out = Vec::new();
                            for (i, &d) in x.iter().enumerate() {
                                if reduce_all || axis_set.contains(&i) {
                                    if attrs.keepdims {
                                        out.push(1);
                                    }
                                } else {
                                    out.push(d);
                                }
                            }
                            self.set_static_tensor(oid, primary_dtype, out);
                        }
                    }
                }

                Op::MatMul | Op::MatMulInteger => {
                    if let (Some(oid), Some(a), Some(b)) = (
                        out0_id,
                        in_shapes.first().and_then(|s| s.as_ref()),
                        in_shapes.get(1).and_then(|s| s.as_ref()),
                    ) {
                        if a.len() >= 2 && b.len() >= 2 {
                            let a_batch = &a[..a.len() - 2];
                            let b_batch = &b[..b.len() - 2];
                            if let Some(mut out) = broadcast_shape(a_batch, b_batch) {
                                out.push(a[a.len() - 2]);
                                out.push(b[b.len() - 1]);
                                let dt = if matches!(node.op, Op::MatMulInteger) {
                                    DataType::Int32
                                } else {
                                    primary_dtype
                                };
                                self.set_static_tensor(oid, dt, out);
                            }
                        }
                    }
                }

                Op::Gemm(attrs) => {
                    if let (Some(oid), Some(a), Some(b)) = (
                        out0_id,
                        in_shapes.first().and_then(|s| s.as_ref()),
                        in_shapes.get(1).and_then(|s| s.as_ref()),
                    ) {
                        if a.len() == 2 && b.len() == 2 {
                            let m = if attrs.trans_a { a[1] } else { a[0] };
                            let n = if attrs.trans_b { b[0] } else { b[1] };
                            self.set_static_tensor(oid, primary_dtype, vec![m, n]);
                        }
                    }
                }

                Op::Transpose { perm } => {
                    if let (Some(oid), Some(x)) = (out0_id, in_shapes.first().and_then(|s| s.as_ref())) {
                        if perm.len() == x.len() && perm.iter().all(|&p| p < x.len()) {
                            let out: Vec<usize> = perm.iter().map(|&p| x[p]).collect();
                            self.set_static_tensor(oid, primary_dtype, out);
                        }
                    }
                }

                Op::Reshape { allowzero } => {
                    if let (Some(oid), Some(x), Some(shape_id)) = (
                        out0_id,
                        in_shapes.first().and_then(|s| s.as_ref()),
                        node.inputs.get(1).copied().flatten(),
                    ) {
                        if let Some(target) = const_i64.get(&shape_id) {
                            if let Some(out) = resolve_reshape(x, target, *allowzero) {
                                self.set_static_tensor(oid, primary_dtype, out);
                            }
                        }
                    }
                }

                Op::Concat { axis } => {
                    let valid_shapes: Option<Vec<&Vec<usize>>> =
                        in_shapes.iter().map(|s| s.as_ref()).collect();
                    if let (Some(oid), Some(shapes)) = (out0_id, valid_shapes) {
                        if let Some(first) = shapes.first() {
                            let rank = first.len() as i64;
                            let ax = if *axis < 0 { *axis + rank } else { *axis } as usize;
                            if ax < first.len() && shapes.iter().all(|s| s.len() == first.len()) {
                                let mut out = (*first).clone();
                                out[ax] = shapes.iter().map(|s| s[ax]).sum();
                                self.set_static_tensor(oid, primary_dtype, out);

                                // Propagate small 1D i64 constant concat (shape builders).
                                if first.len() == 1 && ax == 0 {
                                    let mut merged = Vec::new();
                                    let mut all_const = true;
                                    for inp in &node.inputs {
                                        if let Some(v) = inp.and_then(|id| const_i64.get(&id)) {
                                            merged.extend_from_slice(v);
                                        } else {
                                            all_const = false;
                                            break;
                                        }
                                    }
                                    if all_const {
                                        const_i64.insert(oid, merged);
                                    }
                                }
                            }
                        }
                    }
                }

                Op::Split { axis, .. } => {
                    if let Some(x) = in_shapes.first().and_then(|s| s.as_ref()) {
                        let rank = x.len() as i64;
                        let ax = if *axis < 0 { *axis + rank } else { *axis } as usize;
                        if ax < x.len() && !node.outputs.is_empty() {
                            let splits: Option<Vec<usize>> = node
                                .inputs
                                .get(1)
                                .copied()
                                .flatten()
                                .and_then(|id| const_i64.get(&id))
                                .map(|v| v.iter().map(|&n| n.max(0) as usize).collect());
                            let n_out = node.outputs.len();
                            for (i, opt_oid) in node.outputs.iter().enumerate() {
                                if let Some(oid) = *opt_oid {
                                    let mut out = x.clone();
                                    if let Some(ref sp) = splits {
                                        if i < sp.len() {
                                            out[ax] = sp[i];
                                            self.set_static_tensor(oid, primary_dtype, out);
                                        }
                                    } else if x[ax] % n_out == 0 {
                                        out[ax] = x[ax] / n_out;
                                        self.set_static_tensor(oid, primary_dtype, out);
                                    }
                                }
                            }
                        }
                    }
                }

                Op::Shape { start, end } => {
                    if let (Some(oid), Some(x)) = (out0_id, in_shapes.first().and_then(|s| s.as_ref())) {
                        let rank = x.len() as i64;
                        let s = start.unwrap_or(0);
                        let e = end.unwrap_or(rank);
                        let s = (if s < 0 { s + rank } else { s }).clamp(0, rank) as usize;
                        let e = (if e < 0 { e + rank } else { e }).clamp(0, rank) as usize;
                        let slice: Vec<i64> = if s <= e {
                            x[s..e].iter().map(|&d| d as i64).collect()
                        } else {
                            Vec::new()
                        };
                        self.set_static_tensor(oid, DataType::Int64, vec![slice.len()]);
                        const_i64.insert(oid, slice);
                    }
                }

                Op::Size => {
                    if let (Some(oid), Some(x)) = (out0_id, in_shapes.first().and_then(|s| s.as_ref())) {
                        let total: i64 = x.iter().fold(1i64, |a, &d| a.saturating_mul(d as i64));
                        self.set_static_tensor(oid, DataType::Int64, vec![]);
                        const_i64.insert(oid, vec![total]);
                    }
                }

                Op::Unsqueeze { axes } => {
                    if let (Some(oid), Some(x)) = (out0_id, in_shapes.first().and_then(|s| s.as_ref())) {
                        let ax_list = axes.clone().or_else(|| {
                            node.inputs.get(1).copied().flatten().and_then(|id| const_i64.get(&id).cloned())
                        });
                        if let Some(ax_list) = ax_list {
                            let out_rank = x.len() + ax_list.len();
                            let mut norm: Vec<usize> = ax_list
                                .iter()
                                .map(|&a| {
                                    if a < 0 {
                                        (a + out_rank as i64) as usize
                                    } else {
                                        a as usize
                                    }
                                })
                                .collect();
                            norm.sort_unstable();
                            let mut out = x.clone();
                            for a in norm {
                                if a <= out.len() {
                                    out.insert(a, 1);
                                }
                            }
                            self.set_static_tensor(oid, primary_dtype, out);
                            if let Some(Some(in0)) = node.inputs.first() {
                                if let Some(v) = const_i64.get(in0).cloned() {
                                    const_i64.insert(oid, v);
                                }
                            }
                        }
                    }
                }

                Op::Squeeze { axes } => {
                    if let (Some(oid), Some(x)) = (out0_id, in_shapes.first().and_then(|s| s.as_ref())) {
                        let ax_list = axes.clone().or_else(|| {
                            node.inputs.get(1).copied().flatten().and_then(|id| const_i64.get(&id).cloned())
                        });
                        let out: Vec<usize> = if let Some(ax_list) = ax_list {
                            let rank = x.len() as i64;
                            let rm: HashSet<usize> = ax_list
                                .iter()
                                .map(|&a| if a < 0 { (a + rank) as usize } else { a as usize })
                                .collect();
                            x.iter()
                                .enumerate()
                                .filter_map(|(i, &d)| (!rm.contains(&i)).then_some(d))
                                .collect()
                        } else {
                            x.iter().copied().filter(|&d| d != 1).collect()
                        };
                        self.set_static_tensor(oid, primary_dtype, out);
                        if let Some(Some(in0)) = node.inputs.first() {
                            if let Some(v) = const_i64.get(in0).cloned() {
                                const_i64.insert(oid, v);
                            }
                        }
                    }
                }

                Op::Gather { axis } => {
                    if let (Some(oid), Some(data_s), Some(idx_s)) = (
                        out0_id,
                        in_shapes.first().and_then(|s| s.as_ref()),
                        in_shapes.get(1).and_then(|s| s.as_ref()),
                    ) {
                        let rank = data_s.len() as i64;
                        let ax = if *axis < 0 { *axis + rank } else { *axis } as usize;
                        if ax < data_s.len() {
                            let mut out = Vec::with_capacity(data_s.len() - 1 + idx_s.len());
                            out.extend_from_slice(&data_s[..ax]);
                            out.extend_from_slice(idx_s);
                            out.extend_from_slice(&data_s[ax + 1..]);
                            self.set_static_tensor(oid, primary_dtype, out);

                            // Constant fold 1D i64 Gather (shape indexing).
                            if data_s.len() == 1 && ax == 0 {
                                if let (Some(Some(d_id)), Some(Some(i_id))) =
                                    (node.inputs.first(), node.inputs.get(1))
                                {
                                    if let (Some(d_vals), Some(i_vals)) =
                                        (const_i64.get(d_id), const_i64.get(i_id))
                                    {
                                        let len = d_vals.len() as i64;
                                        let gathered: Option<Vec<i64>> = i_vals
                                            .iter()
                                            .map(|&idx| {
                                                let j = if idx < 0 { idx + len } else { idx };
                                                d_vals.get(j as usize).copied()
                                            })
                                            .collect();
                                        if let Some(g) = gathered {
                                            const_i64.insert(oid, g);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Op::Resize(_) => {
                    if let (Some(oid), Some(x)) = (out0_id, in_shapes.first().and_then(|s| s.as_ref())) {
                        // Input 3 = sizes, input 2 = scales
                        if let Some(sizes) = node
                            .inputs
                            .get(3)
                            .copied()
                            .flatten()
                            .and_then(|id| const_i64.get(&id))
                        {
                            if sizes.len() == x.len() {
                                let out: Vec<usize> = sizes.iter().map(|&v| v.max(0) as usize).collect();
                                self.set_static_tensor(oid, primary_dtype, out);
                            }
                        } else if let Some(scales_tp) = node
                            .inputs
                            .get(2)
                            .copied()
                            .flatten()
                            .and_then(|id| self.weights.get(&id))
                        {
                            if let Ok(scales) = scales_tp.to_f32_vec() {
                                if scales.len() == x.len() {
                                    let out: Vec<usize> = x
                                        .iter()
                                        .zip(scales.iter())
                                        .map(|(&d, &sc)| ((d as f32) * sc).floor().max(1.0) as usize)
                                        .collect();
                                    self.set_static_tensor(oid, primary_dtype, out);
                                }
                            }
                        }
                    }
                }

                Op::Pad { .. } => {
                    if let (Some(oid), Some(x), Some(pads_id)) = (
                        out0_id,
                        in_shapes.first().and_then(|s| s.as_ref()),
                        node.inputs.get(1).copied().flatten(),
                    ) {
                        if let Some(pads) = const_i64.get(&pads_id) {
                            if pads.len() == x.len() * 2 {
                                let rank = x.len();
                                let out: Vec<usize> = (0..rank)
                                    .map(|i| (x[i] as i64 + pads[i] + pads[i + rank]).max(0) as usize)
                                    .collect();
                                self.set_static_tensor(oid, primary_dtype, out);
                            }
                        }
                    }
                }

                Op::Expand => {
                    if let (Some(oid), Some(x), Some(sh_id)) = (
                        out0_id,
                        in_shapes.first().and_then(|s| s.as_ref()),
                        node.inputs.get(1).copied().flatten(),
                    ) {
                        if let Some(target) = const_i64.get(&sh_id) {
                            let t_usize: Vec<usize> =
                                target.iter().map(|&v| v.max(1) as usize).collect();
                            if let Some(out) = broadcast_shape(x, &t_usize) {
                                self.set_static_tensor(oid, primary_dtype, out);
                            }
                        }
                    }
                }

                Op::Tile => {
                    if let (Some(oid), Some(x), Some(rep_id)) = (
                        out0_id,
                        in_shapes.first().and_then(|s| s.as_ref()),
                        node.inputs.get(1).copied().flatten(),
                    ) {
                        if let Some(reps) = const_i64.get(&rep_id) {
                            if reps.len() == x.len() {
                                let out: Vec<usize> = x
                                    .iter()
                                    .zip(reps.iter())
                                    .map(|(&d, &r)| d.saturating_mul(r.max(0) as usize))
                                    .collect();
                                self.set_static_tensor(oid, primary_dtype, out);
                            }
                        }
                    }
                }

                Op::ConstantOfShape { value } => {
                    if let (Some(oid), Some(sh_id)) = (out0_id, node.inputs.first().copied().flatten()) {
                        if let Some(dims) = const_i64.get(&sh_id) {
                            let out: Vec<usize> = dims.iter().map(|&d| d.max(0) as usize).collect();
                            self.set_static_tensor(oid, value.data_type, out);
                        }
                    }
                }

                Op::ScatterND => {
                    if let (Some(oid), Some(x)) = (out0_id, in_shapes.first().and_then(|s| s.as_ref())) {
                        self.set_static_tensor(oid, primary_dtype, x.clone());
                    }
                }

                Op::DynamicQuantizeLinear => {
                    if let Some(x) = in_shapes.first().and_then(|s| s.as_ref()) {
                        if let Some(Some(y_id)) = node.outputs.first() {
                            self.set_static_tensor(*y_id, DataType::Uint8, x.clone());
                        }
                        if let Some(Some(sc_id)) = node.outputs.get(1) {
                            self.set_static_tensor(*sc_id, DataType::Float, vec![]);
                        }
                        if let Some(Some(zp_id)) = node.outputs.get(2) {
                            self.set_static_tensor(*zp_id, DataType::Uint8, vec![]);
                        }
                    }
                }

                Op::Slice => {
                    if let (Some(oid), Some(x)) = (out0_id, in_shapes.first().and_then(|s| s.as_ref())) {
                        let starts = node.inputs.get(1).copied().flatten().and_then(|id| const_i64.get(&id));
                        let ends = node.inputs.get(2).copied().flatten().and_then(|id| const_i64.get(&id));
                        if let (Some(starts), Some(ends)) = (starts, ends) {
                            let axes: Vec<i64> = node
                                .inputs
                                .get(3)
                                .copied()
                                .flatten()
                                .and_then(|id| const_i64.get(&id).cloned())
                                .unwrap_or_else(|| (0..starts.len() as i64).collect());
                            let steps: Vec<i64> = node
                                .inputs
                                .get(4)
                                .copied()
                                .flatten()
                                .and_then(|id| const_i64.get(&id).cloned())
                                .unwrap_or_else(|| vec![1; starts.len()]);
                            let mut out = x.clone();
                            let rank = x.len() as i64;
                            for i in 0..starts.len().min(ends.len()).min(axes.len()).min(steps.len()) {
                                let ax = if axes[i] < 0 { axes[i] + rank } else { axes[i] } as usize;
                                let step = steps[i];
                                if ax < out.len() && step != 0 {
                                    let dim = x[ax] as i64;
                                    let st = clamp_slice_idx(starts[i], dim, step > 0);
                                    let en = clamp_slice_idx(ends[i], dim, step > 0);
                                    let len = if step > 0 {
                                        if en <= st { 0 } else { (en - st + step - 1) / step }
                                    } else if st <= en {
                                        0
                                    } else {
                                        (st - en + (-step) - 1) / (-step)
                                    };
                                    out[ax] = len.max(0) as usize;
                                }
                            }
                            self.set_static_tensor(oid, primary_dtype, out);
                        }
                    }
                }

                Op::If(_) => {
                    // Output shape of `If` depends on branch; keep declared output info if present.
                }
            }
        }

        let resolved = self
            .tensors
            .iter()
            .filter(|t| t.static_shape().is_some())
            .count();
        Ok(resolved)
    }

    fn set_static_tensor(&mut self, id: TensorId, dtype: DataType, dims: Vec<usize>) {
        let info = &mut self.tensors[id.0];
        if dtype != DataType::Undefined {
            info.dtype = dtype;
        }
        info.shape = Some(dims.into_iter().map(Some).collect());
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dims_from_spec(specs: &[DimSpec]) -> Option<Vec<Option<usize>>> {
    if specs.is_empty() {
        return None;
    }
    Some(
        specs
            .iter()
            .map(|d| match d {
                DimSpec::Value(v) if *v >= 0 => Some(*v as usize),
                _ => None,
            })
            .collect(),
    )
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for i in 0..rank {
        let da = if i < rank - a.len() { 1 } else { a[i - (rank - a.len())] };
        let db = if i < rank - b.len() { 1 } else { b[i - (rank - b.len())] };
        if da == db {
            out[i] = da;
        } else if da == 1 {
            out[i] = db;
        } else if db == 1 {
            out[i] = da;
        } else {
            return None;
        }
    }
    Some(out)
}

fn resolve_reshape(input: &[usize], target: &[i64], allowzero: bool) -> Option<Vec<usize>> {
    let total: usize = input.iter().product();
    let mut out = vec![0usize; target.len()];
    let mut infer_idx: Option<usize> = None;
    let mut known_prod: usize = 1;

    for (i, &d) in target.iter().enumerate() {
        if d == -1 {
            if infer_idx.is_some() {
                return None;
            }
            infer_idx = Some(i);
        } else if d == 0 && !allowzero {
            let copied = *input.get(i)?;
            out[i] = copied;
            known_prod = known_prod.checked_mul(copied)?;
        } else if d >= 0 {
            let v = d as usize;
            out[i] = v;
            known_prod = known_prod.checked_mul(v)?;
        } else {
            return None;
        }
    }

    if let Some(idx) = infer_idx {
        if known_prod == 0 || total % known_prod != 0 {
            return None;
        }
        out[idx] = total / known_prod;
    }
    Some(out)
}

fn clamp_slice_idx(idx: i64, dim: i64, positive_step: bool) -> i64 {
    let v = if idx < 0 { idx + dim } else { idx };
    if positive_step {
        v.clamp(0, dim)
    } else {
        v.clamp(-1, dim - 1)
    }
}

fn ints_to_usize(v: &[i64]) -> Vec<usize> {
    v.iter().map(|&x| x.max(0) as usize).collect()
}

fn parse_conv_attrs(np: &NodeProto) -> ConvAttrs {
    ConvAttrs {
        kernel_shape: np.attr_ints("kernel_shape").map(ints_to_usize),
        strides: np
            .attr_ints("strides")
            .map(ints_to_usize)
            .unwrap_or_else(|| vec![1, 1]),
        pads: np
            .attr_ints("pads")
            .map(ints_to_usize)
            .unwrap_or_else(|| vec![0, 0, 0, 0]),
        dilations: np
            .attr_ints("dilations")
            .map(ints_to_usize)
            .unwrap_or_else(|| vec![1, 1]),
        group: np.attr_int("group").unwrap_or(1).max(1) as usize,
    }
}

fn parse_op(np: &NodeProto, parent: &IrGraph) -> Result<Op, GraphError> {
    match np.op_type.as_str() {
        "Conv" => Ok(Op::Conv(parse_conv_attrs(np))),
        "ConvInteger" => Ok(Op::ConvInteger(parse_conv_attrs(np))),
        "ConvTranspose" => Ok(Op::ConvTranspose(ConvTransposeAttrs {
            kernel_shape: np.attr_ints("kernel_shape").map(ints_to_usize),
            strides: np
                .attr_ints("strides")
                .map(ints_to_usize)
                .unwrap_or_else(|| vec![1, 1]),
            pads: np
                .attr_ints("pads")
                .map(ints_to_usize)
                .unwrap_or_else(|| vec![0, 0, 0, 0]),
            dilations: np
                .attr_ints("dilations")
                .map(ints_to_usize)
                .unwrap_or_else(|| vec![1, 1]),
            group: np.attr_int("group").unwrap_or(1).max(1) as usize,
            output_padding: np
                .attr_ints("output_padding")
                .map(ints_to_usize)
                .unwrap_or_else(|| vec![0, 0]),
        })),
        "MatMul" => Ok(Op::MatMul),
        "MatMulInteger" => Ok(Op::MatMulInteger),
        "Gemm" => Ok(Op::Gemm(GemmAttrs {
            alpha: np.attr_float("alpha").unwrap_or(1.0),
            beta: np.attr_float("beta").unwrap_or(1.0),
            trans_a: np.attr_int("transA").unwrap_or(0) != 0,
            trans_b: np.attr_int("transB").unwrap_or(0) != 0,
        })),
        "BatchNormalization" => Ok(Op::BatchNormalization(BatchNormAttrs {
            epsilon: np.attr_float("epsilon").unwrap_or(1e-5),
            momentum: np.attr_float("momentum").unwrap_or(0.9),
        })),
        "LayerNormalization" => Ok(Op::LayerNormalization(LayerNormAttrs {
            axis: np.attr_int("axis").unwrap_or(-1),
            epsilon: np.attr_float("epsilon").unwrap_or(1e-5),
            stash_type: np.attr_int("stash_type").unwrap_or(1),
        })),
        "InstanceNormalization" => Ok(Op::InstanceNormalization {
            epsilon: np.attr_float("epsilon").unwrap_or(1e-5),
        }),
        "MaxPool" => Ok(Op::MaxPool(PoolAttrs {
            kernel_shape: np
                .attr_ints("kernel_shape")
                .map(ints_to_usize)
                .ok_or(GraphError::MissingAttribute {
                    op_type: "MaxPool".into(),
                    attr: "kernel_shape",
                })?,
            strides: np
                .attr_ints("strides")
                .map(ints_to_usize)
                .unwrap_or_else(|| vec![1, 1]),
            pads: np
                .attr_ints("pads")
                .map(ints_to_usize)
                .unwrap_or_else(|| vec![0, 0, 0, 0]),
            dilations: np
                .attr_ints("dilations")
                .map(ints_to_usize)
                .unwrap_or_else(|| vec![1, 1]),
            ceil_mode: np.attr_int("ceil_mode").unwrap_or(0) != 0,
            count_include_pad: false,
        })),
        "AveragePool" => Ok(Op::AveragePool(PoolAttrs {
            kernel_shape: np
                .attr_ints("kernel_shape")
                .map(ints_to_usize)
                .ok_or(GraphError::MissingAttribute {
                    op_type: "AveragePool".into(),
                    attr: "kernel_shape",
                })?,
            strides: np
                .attr_ints("strides")
                .map(ints_to_usize)
                .unwrap_or_else(|| vec![1, 1]),
            pads: np
                .attr_ints("pads")
                .map(ints_to_usize)
                .unwrap_or_else(|| vec![0, 0, 0, 0]),
            dilations: vec![1, 1],
            ceil_mode: np.attr_int("ceil_mode").unwrap_or(0) != 0,
            count_include_pad: np.attr_int("count_include_pad").unwrap_or(0) != 0,
        })),
        "GlobalAveragePool" => Ok(Op::GlobalAveragePool),
        "ReduceMean" => Ok(Op::ReduceMean(ReduceAttrs {
            axes: np.attr_ints("axes").map(|s| s.to_vec()),
            keepdims: np.attr_int("keepdims").unwrap_or(1) != 0,
            noop_with_empty_axes: np.attr_int("noop_with_empty_axes").unwrap_or(0) != 0,
        })),
        "ReduceSum" => Ok(Op::ReduceSum(ReduceAttrs {
            axes: np.attr_ints("axes").map(|s| s.to_vec()),
            keepdims: np.attr_int("keepdims").unwrap_or(1) != 0,
            noop_with_empty_axes: np.attr_int("noop_with_empty_axes").unwrap_or(0) != 0,
        })),
        "ReduceMax" => Ok(Op::ReduceMax(ReduceAttrs {
            axes: np.attr_ints("axes").map(|s| s.to_vec()),
            keepdims: np.attr_int("keepdims").unwrap_or(1) != 0,
            noop_with_empty_axes: np.attr_int("noop_with_empty_axes").unwrap_or(0) != 0,
        })),
        "Resize" => {
            let mode = match np.attr_string("mode").unwrap_or("nearest") {
                "linear" => ResizeMode::Linear,
                "cubic" => ResizeMode::Cubic,
                _ => ResizeMode::Nearest,
            };
            let coord_mode = match np
                .attr_string("coordinate_transformation_mode")
                .unwrap_or("half_pixel")
            {
                "asymmetric" => CoordTransformMode::Asymmetric,
                "align_corners" => CoordTransformMode::AlignCorners,
                "tf_half_pixel_for_nn" => CoordTransformMode::TfHalfPixelForNn,
                "pytorch_half_pixel" => CoordTransformMode::PytorchHalfPixel,
                _ => CoordTransformMode::HalfPixel,
            };
            let nearest_mode = match np
                .attr_string("nearest_mode")
                .unwrap_or("round_prefer_floor")
            {
                "floor" => NearestMode::Floor,
                "round_prefer_ceil" => NearestMode::RoundPreferCeil,
                "ceil" => NearestMode::Ceil,
                _ => NearestMode::RoundPreferFloor,
            };
            Ok(Op::Resize(ResizeAttrs {
                mode,
                coord_mode,
                nearest_mode,
                cubic_coeff_a: np.attr_float("cubic_coeff_a").unwrap_or(-0.75),
            }))
        }
        "Pad" => Ok(Op::Pad {
            mode: np.attr_string("mode").unwrap_or("constant").to_owned(),
        }),
        "Add" => Ok(Op::Add),
        "Sub" => Ok(Op::Sub),
        "Mul" => Ok(Op::Mul),
        "Div" => Ok(Op::Div),
        "Mod" => Ok(Op::Mod {
            fmod: np.attr_int("fmod").unwrap_or(0) != 0,
        }),
        "Pow" => Ok(Op::Pow),
        "Sqrt" => Ok(Op::Sqrt),
        "Reciprocal" => Ok(Op::Reciprocal),
        "Abs" => Ok(Op::Abs),
        "Sin" => Ok(Op::Sin),
        "Cos" => Ok(Op::Cos),
        "Relu" => Ok(Op::Relu),
        "PRelu" => Ok(Op::PRelu),
        "LeakyRelu" => Ok(Op::LeakyRelu {
            alpha: np.attr_float("alpha").unwrap_or(0.01),
        }),
        "Sigmoid" => Ok(Op::Sigmoid),
        "HardSwish" => Ok(Op::HardSwish),
        "HardSigmoid" => Ok(Op::HardSigmoid {
            alpha: np.attr_float("alpha").unwrap_or(0.2),
            beta: np.attr_float("beta").unwrap_or(0.5),
        }),
        "Erf" => Ok(Op::Erf),
        "Clip" => Ok(Op::Clip),
        "Softmax" => Ok(Op::Softmax {
            axis: np.attr_int("axis").unwrap_or(-1),
        }),
        "Equal" => Ok(Op::Equal),
        "Greater" => Ok(Op::Greater),
        "GreaterOrEqual" => Ok(Op::GreaterOrEqual),
        "Less" => Ok(Op::Less),
        "And" => Ok(Op::And),
        "Where" => Ok(Op::Where),
        "Transpose" => Ok(Op::Transpose {
            perm: np
                .attr_ints("perm")
                .map(ints_to_usize)
                .unwrap_or_default(),
        }),
        "Reshape" => Ok(Op::Reshape {
            allowzero: np.attr_int("allowzero").unwrap_or(0) != 0,
        }),
        "Concat" => Ok(Op::Concat {
            axis: np.attr_int("axis").ok_or(GraphError::MissingAttribute {
                op_type: "Concat".into(),
                attr: "axis",
            })?,
        }),
        "Split" => Ok(Op::Split {
            axis: np.attr_int("axis").unwrap_or(0),
            num_outputs: np.attr_int("num_outputs").map(|v| v.max(0) as usize),
        }),
        "Slice" => Ok(Op::Slice),
        "Gather" => Ok(Op::Gather {
            axis: np.attr_int("axis").unwrap_or(0),
        }),
        "ScatterND" => Ok(Op::ScatterND),
        "Expand" => Ok(Op::Expand),
        "Tile" => Ok(Op::Tile),
        "Squeeze" => Ok(Op::Squeeze {
            axes: np.attr_ints("axes").map(|s| s.to_vec()),
        }),
        "Unsqueeze" => Ok(Op::Unsqueeze {
            axes: np.attr_ints("axes").map(|s| s.to_vec()),
        }),
        "Shape" => Ok(Op::Shape {
            start: np.attr_int("start"),
            end: np.attr_int("end"),
        }),
        "Size" => Ok(Op::Size),
        "Cast" => {
            let to = np.attr_int("to").ok_or(GraphError::MissingAttribute {
                op_type: "Cast".into(),
                attr: "to",
            })?;
            let dt = DataType::from_u32(to as u32).map_err(|e| GraphError::InvalidAttribute {
                op_type: "Cast".into(),
                detail: e.to_string(),
            })?;
            Ok(Op::Cast { to: dt })
        }
        "Constant" => {
            let value = np
                .attr_tensor("value")
                .cloned()
                .ok_or(GraphError::MissingAttribute {
                    op_type: "Constant".into(),
                    attr: "value",
                })?;
            Ok(Op::Constant { value })
        }
        "ConstantOfShape" => {
            let value = np
                .attr_tensor("value")
                .cloned()
                .unwrap_or_else(|| TensorProto::from_f32("const_of_shape_zero", vec![1], &[0.0]));
            Ok(Op::ConstantOfShape { value })
        }
        "Identity" => Ok(Op::Identity),
        "DynamicQuantizeLinear" => Ok(Op::DynamicQuantizeLinear),
        "If" => {
            let outer_map: HashMap<String, TensorInfo> = parent
                .tensors
                .iter()
                .map(|t| (t.name.clone(), t.clone()))
                .collect();
            let then_gp = np
                .attr_graph("then_branch")
                .ok_or(GraphError::MissingAttribute {
                    op_type: "If".into(),
                    attr: "then_branch",
                })?;
            let else_gp = np
                .attr_graph("else_branch")
                .ok_or(GraphError::MissingAttribute {
                    op_type: "If".into(),
                    attr: "else_branch",
                })?;
            Ok(Op::If(IfAttrs {
                then_branch: Box::new(IrGraph::from_graph_proto(then_gp, Some(&outer_map))?),
                else_branch: Box::new(IrGraph::from_graph_proto(else_gp, Some(&outer_map))?),
            }))
        }
        other => Err(GraphError::UnsupportedOp {
            op_type: other.to_owned(),
            node: np.name.clone(),
        }),
    }
}
