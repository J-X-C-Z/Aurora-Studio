//! Static graph-optimization passes for the native inference engine.
//!
//! Runs after [`crate::native_infer::bn_fold::fold_batch_norms`] and before
//! `IrGraph::infer_shapes` / `arena::plan_memory_arena`, so every rewritten node
//! is shape-inferred and arena-planned exactly like a node that came straight
//! out of the ONNX file.
//!
//! Enabled passes, cheapest first:
//!
//! | pass | pattern | result |
//! |---|---|---|
//! | [`fuse_gelu`] | `Div(x, sqrt 2) -> Erf -> Add(1) -> Mul(x, .) -> Mul(., 0.5)` | one `Op::Gelu` |
//! | [`fuse_silu`] | `Sigmoid(x)` + `Mul(x, sigmoid)` | one `Op::SiLU` |
//! | [`fuse_conv_activation`] | `Conv -> Relu` / `Conv -> SiLU` | `Conv` epilogue |
//! | [`eliminate_aliases`] | `Identity`, `Cast(T -> T)` | tensor-id redirect |
//! | [`eliminate_dead_nodes`] | unused pure nodes (e.g. freed constants) | node removal |
//!
//! Every pass is *strict*: each intermediate tensor must have exactly one
//! consumer, must not be a graph output, and must live in the same fp32 domain
//! as the fused op. Anything else is left untouched — a missed fusion only costs
//! speed, a wrong fusion costs pixels.
//!
//! The fused elementwise op is executed by `unary_f32` kernel ids `2` (`SiLU`) and
//! `3` (`Gelu`, exact erf form) in `native/cuda/infer_elem_norm.cu`, which is why
//! `Op::Gelu` / `Op::SiLU` are exact and take no attributes.

use std::collections::{HashMap, HashSet};

use crate::native_infer::graph::{IrGraph, NodeId, Op, TensorId};
use crate::native_infer::onnx_proto::DataType;
use crate::native_infer::ops_conv_gemm::FusedActivation;

/// Environment variable selecting the fusion level.
///
/// `0` / `off` — disabled (A/B baseline, bit-identical to the pre-fusion plan);
/// `safe` / `1` — executor-independent passes only (the default);
/// `all` / `full` / `2` — also `Gelu`, `SiLU` and `Conv + activation` fusion.
/// The `Conv` epilogue additionally needs [`ENV_CONV_EPILOGUE`]; see
/// [`FuseConfig::from_env`].
pub const ENV_FUSE: &str = "AURORA_NATIVE_FUSE";

/// Second switch for the `Conv + activation` epilogue in the env-driven config.
///
/// That pass deletes the activation node and expects the executor to apply
/// `CompiledPlan::conv_activations` to the convolution output. An executor that
/// does not read the table would silently drop the activation, so the pass is only
/// enabled from the environment when this flag is explicitly `1` (set it together
/// with the `session.rs` epilogue hook).
pub const ENV_CONV_EPILOGUE: &str = "AURORA_NATIVE_CONV_EPILOGUE";

/// Fusion level used when [`ENV_FUSE`] is unset.
///
/// [`FuseLevel::Safe`] is the default because every pass it enables is executable
/// by the current `session.rs` dispatcher. The activation passes additionally need
/// two dispatcher hooks (both one line, listed under [`FuseLevel::Full`]); until
/// they land, `Full` would abort at the first fused node instead of reproducing
/// the unfused numerics, so it stays opt-in. Flip this constant to
/// [`FuseLevel::Full`] once those hooks exist.
pub const DEFAULT_FUSE_LEVEL: FuseLevel = FuseLevel::Safe;

const SQRT_2: f32 = std::f32::consts::SQRT_2;
const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
/// Relative tolerance when recognising the two GELU scale constants.
const CONST_TOL: f32 = 1e-6;

/// Graph rewrite level, see [`DEFAULT_FUSE_LEVEL`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FuseLevel {
    /// No rewriting at all (`bn_fold` still runs).
    Off,
    /// `Identity` / redundant `Cast` alias elimination + dead-node elimination.
    /// Needs nothing beyond the existing dispatcher.
    Safe,
    /// Additionally folds GELU / SiLU chains and `Conv + activation` epilogues.
    ///
    /// Requires the dispatcher hooks
    /// `session.rs`: `Op::Gelu => (UnaryOp::Gelu, ..)` and
    /// `Op::SiLU => (UnaryOp::SiLU, ..)` in the unary arm, plus reading
    /// `CompiledPlan::conv_activations` when building `Conv2dParams`.
    Full,
}

impl FuseLevel {
    /// Parse [`ENV_FUSE`]; unknown values fall back to [`DEFAULT_FUSE_LEVEL`].
    pub fn from_env() -> Self {
        match std::env::var(ENV_FUSE) {
            Ok(raw) => Self::parse(&raw),
            Err(_) => DEFAULT_FUSE_LEVEL,
        }
    }

    /// Parse an [`ENV_FUSE`] value, case-insensitively.
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "off" | "false" | "no" => Self::Off,
            "safe" | "1" | "true" | "yes" => Self::Safe,
            "all" | "full" | "2" => Self::Full,
            _ => DEFAULT_FUSE_LEVEL,
        }
    }

    /// Pass switches for this level.
    pub const fn config(self) -> FuseConfig {
        match self {
            Self::Off => FuseConfig::OFF,
            Self::Safe => FuseConfig::SAFE,
            Self::Full => FuseConfig::FULL,
        }
    }
}

/// Individual pass switches (lets tests and A/B runs enable one pass at a time).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FuseConfig {
    pub fuse_gelu: bool,
    pub fuse_silu: bool,
    pub fuse_conv_activation: bool,
    pub eliminate_identity: bool,
    pub eliminate_cast: bool,
    pub eliminate_dead_nodes: bool,
}

impl FuseConfig {
    /// A/B baseline: nothing is rewritten.
    pub const OFF: Self = Self {
        fuse_gelu: false,
        fuse_silu: false,
        fuse_conv_activation: false,
        eliminate_identity: false,
        eliminate_cast: false,
        eliminate_dead_nodes: false,
    };

    /// Executor-independent passes (see [`FuseLevel::Safe`]).
    pub const SAFE: Self = Self {
        fuse_gelu: false,
        fuse_silu: false,
        fuse_conv_activation: false,
        eliminate_identity: true,
        eliminate_cast: true,
        eliminate_dead_nodes: true,
    };

    /// Everything, including the passes that need the session hooks.
    pub const FULL: Self = Self {
        fuse_gelu: true,
        fuse_silu: true,
        fuse_conv_activation: true,
        eliminate_identity: true,
        eliminate_cast: true,
        eliminate_dead_nodes: true,
    };

    /// `true` when every switch is off.
    pub const fn is_noop(&self) -> bool {
        !(self.fuse_gelu
            || self.fuse_silu
            || self.fuse_conv_activation
            || self.eliminate_identity
            || self.eliminate_cast
            || self.eliminate_dead_nodes)
    }

    /// Pass switches from [`ENV_FUSE`], falling back to [`DEFAULT_FUSE_LEVEL`].
    ///
    /// The `Conv` epilogue is gated a second time by [`ENV_CONV_EPILOGUE`] because
    /// it is only correct once the executor applies `conv_activations`.
    pub fn from_env() -> Self {
        let mut config = FuseLevel::from_env().config();
        if config.fuse_conv_activation {
            config.fuse_conv_activation =
                env_flag_is_on(std::env::var(ENV_CONV_EPILOGUE).ok().as_deref());
        }
        config
    }
}

/// `true` for an explicit `1` switch value.
fn env_flag_is_on(raw: Option<&str>) -> bool {
    raw.map(|v| v.trim() == "1").unwrap_or(false)
}

impl Default for FuseConfig {
    fn default() -> Self {
        DEFAULT_FUSE_LEVEL.config()
    }
}

/// What [`fuse_activations`] + [`finalize_graph`] did, for logging and A/B comparison.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FuseStats {
    /// `Div/Erf/Add/Mul/Mul` chains collapsed into a single `Op::Gelu`.
    pub gelu_fused: usize,
    /// `Sigmoid` + `Mul` pairs collapsed into a single `Op::SiLU`.
    pub silu_fused: usize,
    /// `Conv` nodes that absorbed their `Relu` / `SiLU` consumer.
    pub conv_activation_fused: usize,
    /// `Identity` nodes replaced by a tensor-id redirect.
    pub identity_removed: usize,
    /// Redundant `Cast(T -> T)` nodes replaced by a tensor-id redirect.
    pub cast_removed: usize,
    /// Pure nodes whose outputs nobody reads.
    pub dead_nodes_removed: usize,
    /// `true` when rewriting was skipped because the graph owns `If` branches
    /// (subgraph captures make outer consumer counts unreliable).
    pub skipped_control_flow: bool,
    /// `true` when the activation passes were skipped because the graph carries
    /// fp16 tensors (the fused activations are fp32 only).
    pub skipped_fp16: bool,
}

impl FuseStats {
    /// Total number of IR nodes removed from the graph.
    pub fn nodes_removed(&self) -> usize {
        // A fused GELU also drops one of its five chain nodes into the result node,
        // the SiLU pair drops one, and a Conv epilogue drops one activation node.
        self.gelu_fused * 4
            + self.silu_fused
            + self.conv_activation_fused
            + self.identity_removed
            + self.cast_removed
            + self.dead_nodes_removed
    }

    /// `true` when the graph is byte-for-byte the input graph's node list.
    pub fn is_noop(&self) -> bool {
        self.nodes_removed() == 0
    }
}

impl std::ops::AddAssign for FuseStats {
    fn add_assign(&mut self, rhs: Self) {
        self.gelu_fused += rhs.gelu_fused;
        self.silu_fused += rhs.silu_fused;
        self.conv_activation_fused += rhs.conv_activation_fused;
        self.identity_removed += rhs.identity_removed;
        self.cast_removed += rhs.cast_removed;
        self.dead_nodes_removed += rhs.dead_nodes_removed;
        self.skipped_control_flow |= rhs.skipped_control_flow;
        self.skipped_fp16 |= rhs.skipped_fp16;
    }
}

/// Activation fusion passes (`Gelu`, `SiLU`).
///
/// Run **before** `IrGraph::infer_shapes`, so the rewritten graph is the one that
/// gets shape-inferred and arena-planned (a fused chain no longer allocates a
/// single intermediate). Both matchers read graph structure only — never inferred
/// dtypes — which is why they work on the raw IR.
pub fn fuse_activations(graph: &mut IrGraph, config: FuseConfig) -> FuseStats {
    let mut stats = FuseStats::default();
    if !(config.fuse_gelu || config.fuse_silu) || graph.nodes.is_empty() {
        return stats;
    }

    // Global conservative fallbacks.
    //
    // 1. Subgraphs capture outer tensors by re-registering them under fresh ids,
    //    so the outer graph's consumer map cannot prove that a tensor is unused.
    //    Any graph that owns `If` branches therefore keeps its node list intact.
    // 2. The fused ops are fp32 only. A model that carries fp16 tensors
    //    (MatchLook ViT) keeps the unfused chain.
    let has_control_flow = graph.nodes.iter().any(|n| matches!(n.op, Op::If(_)));
    let has_f16 = graph
        .tensors
        .iter()
        .any(|t| matches!(t.dtype, DataType::Float16 | DataType::BFloat16));
    stats.skipped_control_flow = has_control_flow;
    stats.skipped_fp16 = has_f16;
    if has_control_flow || has_f16 {
        return stats;
    }

    if config.fuse_gelu {
        fuse_gelu(graph, &mut stats);
    }
    if config.fuse_silu {
        fuse_silu(graph, &mut stats);
    }
    stats
}

/// Aliasing, dead-node and `Conv + activation` passes.
///
/// Run **after** `IrGraph::infer_shapes`: a `Cast(T -> T)` can only be proven
/// redundant once `T` is known, and the `Conv` epilogue decision wants real fp32
/// confirmation. The returned table is keyed by node id in the rewritten graph and
/// nothing re-indexes the node list afterwards, so the executor can use it
/// directly (`CompiledPlan::conv_activations`).
pub fn finalize_graph(
    graph: &mut IrGraph,
    config: FuseConfig,
) -> (FuseStats, HashMap<NodeId, FusedActivation>) {
    let mut stats = FuseStats::default();
    let mut conv_activations: HashMap<NodeId, FusedActivation> = HashMap::new();
    let wants_alias = config.eliminate_identity || config.eliminate_cast;
    if !(wants_alias
        || config.eliminate_dead_nodes
        || config.fuse_conv_activation)
        || graph.nodes.is_empty()
    {
        return (stats, conv_activations);
    }

    let has_control_flow = graph.nodes.iter().any(|n| matches!(n.op, Op::If(_)));
    let has_f16 = graph
        .tensors
        .iter()
        .any(|t| matches!(t.dtype, DataType::Float16 | DataType::BFloat16));
    stats.skipped_control_flow = has_control_flow;
    stats.skipped_fp16 = has_f16 && config.fuse_conv_activation;
    if has_control_flow {
        return (stats, conv_activations);
    }

    // Aliases first: `Conv -> Identity -> Relu` must become `Conv -> Relu` before
    // the epilogue pass looks at the consumer.
    if wants_alias {
        eliminate_aliases(graph, config, &mut stats);
    }
    if config.eliminate_dead_nodes {
        eliminate_dead_nodes(graph, &mut stats);
    }
    // The epilogue pass runs last: it keys `conv_activations` by node id and
    // removes the activation nodes.
    if config.fuse_conv_activation && !has_f16 {
        fuse_conv_activation(graph, &mut stats, &mut conv_activations);
    }
    (stats, conv_activations)
}

// ---------------------------------------------------------------------------
// Graph indices
// ---------------------------------------------------------------------------

/// Consumer / producer maps over one snapshot of the node list.
struct Index {
    consumers: HashMap<TensorId, usize>,
    /// The single consuming node of a tensor, valid when `consumers == 1`.
    consumer: HashMap<TensorId, usize>,
    producer: HashMap<TensorId, usize>,
    graph_outputs: HashSet<TensorId>,
}

impl Index {
    fn new(graph: &IrGraph) -> Self {
        let mut consumers: HashMap<TensorId, usize> = HashMap::new();
        let mut consumer: HashMap<TensorId, usize> = HashMap::new();
        let mut producer: HashMap<TensorId, usize> = HashMap::new();
        for (idx, node) in graph.nodes.iter().enumerate() {
            for inp in node.inputs.iter().flatten() {
                *consumers.entry(*inp).or_insert(0) += 1;
                consumer.insert(*inp, idx);
            }
            for out in node.outputs.iter().flatten() {
                producer.insert(*out, idx);
            }
        }
        Self {
            consumers,
            consumer,
            producer,
            graph_outputs: graph.outputs.iter().copied().collect(),
        }
    }

    fn single_consumer(&self, tensor: TensorId) -> bool {
        self.consumers.get(&tensor).copied().unwrap_or(0) == 1
    }

    /// The node that consumes `tensor`, when it is the only one.
    fn consumer_of(&self, tensor: TensorId) -> Option<usize> {
        if self.single_consumer(tensor) {
            self.consumer.get(&tensor).copied()
        } else {
            None
        }
    }

    fn producer_of(&self, tensor: TensorId) -> Option<usize> {
        self.producer.get(&tensor).copied()
    }

    /// `true` for tensors that may be redirected / eliminated safely: everything
    /// that is neither a graph output nor an observable boundary value.
    fn is_plain_intermediate(&self, graph: &IrGraph, tensor: TensorId) -> bool {
        let info = &graph.tensors[tensor.0];
        !self.graph_outputs.contains(&tensor) && !info.is_graph_input && !info.is_graph_output
    }
}

fn node_input(graph: &IrGraph, idx: usize, slot: usize) -> Option<TensorId> {
    graph.nodes[idx].inputs.get(slot).copied().flatten()
}

fn node_output(graph: &IrGraph, idx: usize) -> Option<TensorId> {
    graph.nodes[idx].outputs.first().copied().flatten()
}

/// `true` when the tensor may carry an fp32 activation result.
///
/// `Undefined` is accepted because activation fusion runs before shape/dtype
/// inference, where every intermediate is still untyped; the graph-level fp16 guard
/// in [`fuse_activations`] is what keeps real fp16 models out.
fn fp32_like(graph: &IrGraph, tensor: TensorId) -> bool {
    matches!(
        graph.tensors[tensor.0].dtype,
        DataType::Float | DataType::Undefined
    )
}

/// Value of a single-element fp32 initializer / folded `Constant`.
fn scalar_const_f32(graph: &IrGraph, tensor: TensorId) -> Option<f32> {
    if !graph.tensors[tensor.0].is_weight {
        return None;
    }
    let proto = graph.weights.get(&tensor)?;
    if proto.data_type != DataType::Float || proto.elem_count() != 1 {
        return None;
    }
    proto.to_f32_vec().ok()?.first().copied()
}

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= CONST_TOL * b.abs().max(1.0)
}

/// `tensor <op> scalar` with `scalar` a one-element constant.
struct ScalarBinary {
    other: TensorId,
    scalar: f32,
    /// `true` for `other / scalar`, `false` for `other * scalar`.
    is_div: bool,
}

/// Match `Add` / `Sub` / `Mul` / `Div` against a one-element constant operand.
fn split_scalar_binary(graph: &IrGraph, idx: usize) -> Option<ScalarBinary> {
    let node = &graph.nodes[idx];
    if node.outputs.len() != 1 || node.inputs.len() != 2 {
        return None;
    }
    if !matches!(node.op, Op::Add | Op::Sub | Op::Mul | Op::Div) {
        return None;
    }
    let a = node_input(graph, idx, 0)?;
    let b = node_input(graph, idx, 1)?;
    if let Some(scalar) = scalar_const_f32(graph, b) {
        return Some(ScalarBinary {
            other: a,
            scalar,
            is_div: matches!(node.op, Op::Div),
        });
    }
    // `scalar - x` / `scalar / x` are not the shapes GELU/SiLU emit, so only the
    // commutative ops accept a leading constant.
    if matches!(node.op, Op::Add | Op::Mul) {
        if let Some(scalar) = scalar_const_f32(graph, a) {
            return Some(ScalarBinary {
                other: b,
                scalar,
                is_div: false,
            });
        }
    }
    None
}

/// Match `producer_of(tensor)` as `x + 1` and return `(node index, x)`.
fn as_unit_add(graph: &IrGraph, index: &Index, tensor: TensorId) -> Option<(usize, TensorId)> {
    let idx = index.producer_of(tensor)?;
    let node = &graph.nodes[idx];
    if !matches!(node.op, Op::Add) {
        return None;
    }
    let split = split_scalar_binary(graph, idx)?;
    if !approx_eq(split.scalar, 1.0) {
        return None;
    }
    Some((idx, split.other))
}

// ---------------------------------------------------------------------------
// Node removal
// ---------------------------------------------------------------------------

fn resolve_redirect(redirects: &HashMap<TensorId, TensorId>, mut tensor: TensorId) -> TensorId {
    // Chains of aliases (`Identity -> Identity`) collapse transitively; the depth
    // cap only exists so a malformed redirect map cannot spin forever.
    for _ in 0..64 {
        match redirects.get(&tensor) {
            Some(&next) if next != tensor => tensor = next,
            _ => break,
        }
    }
    tensor
}

/// Drop `remove` (indices into the current node list) and apply `redirects` to
/// every surviving node input, then re-index `NodeId`s densely.
fn apply_removals(
    graph: &mut IrGraph,
    remove: &HashSet<usize>,
    redirects: &HashMap<TensorId, TensorId>,
) {
    if remove.is_empty() && redirects.is_empty() {
        return;
    }
    let mut kept = Vec::with_capacity(graph.nodes.len().saturating_sub(remove.len()));
    for (idx, mut node) in graph.nodes.drain(..).enumerate() {
        if remove.contains(&idx) {
            continue;
        }
        if !redirects.is_empty() {
            for inp in node.inputs.iter_mut().flatten() {
                *inp = resolve_redirect(redirects, *inp);
            }
        }
        node.id = NodeId(kept.len());
        kept.push(node);
    }
    graph.nodes = kept;
}

/// Index `old` would have in the node list after removing `remove`.
fn remap_index(old: usize, remove: &HashSet<usize>) -> usize {
    old - remove.iter().filter(|&&r| r < old).count()
}

// ---------------------------------------------------------------------------
// GELU: Div(x, sqrt2) -> Erf -> Add(1) -> Mul(x, .) -> Mul(., 0.5)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct GeluMatch {
    /// Tensor consumed by the first scaling node (the GELU input).
    x: TensorId,
    scale: usize,
    erf: usize,
    add: usize,
    mul_x: usize,
}

/// Match the exact-erf GELU decomposition whose last node is `final_idx`.
///
/// Accepted shapes of the same computation:
/// - `t1 = x / sqrt(2)` or `t1 = x * (1 / sqrt(2))`,
/// - `t2 = Erf(t1)`, `t3 = t2 + 1` (either operand order),
/// - `t4 = x * t3` (either operand order),
/// - `t5 = t4 * 0.5` (either operand order) or `t5 = t4 / 2`.
fn match_gelu(graph: &IrGraph, index: &Index, final_idx: usize) -> Option<GeluMatch> {
    let final_op = &graph.nodes[final_idx].op;
    let outer = split_scalar_binary(graph, final_idx)?;
    let outer_ok = match final_op {
        Op::Mul => approx_eq(outer.scalar, 0.5),
        Op::Div => approx_eq(outer.scalar, 2.0),
        _ => return None,
    };
    if !outer_ok {
        return None;
    }

    // t4 = x * (1 + erf(x / sqrt 2))
    let mul_x = index.producer_of(outer.other)?;
    if !matches!(graph.nodes[mul_x].op, Op::Mul)
        || graph.nodes[mul_x].inputs.len() != 2
        || graph.nodes[mul_x].outputs.len() != 1
    {
        return None;
    }
    let a = node_input(graph, mul_x, 0)?;
    let b = node_input(graph, mul_x, 1)?;
    let add_a = as_unit_add(graph, index, a);
    let add_b = as_unit_add(graph, index, b);
    let (add, t2, t3, x) = match (add_a, add_b) {
        (Some((add, unit_src)), None) => (add, unit_src, a, b),
        (None, Some((add, unit_src))) => (add, unit_src, b, a),
        // Neither operand is a unit-add (not GELU) or both are (`(1+e) * (1+e)`).
        _ => return None,
    };

    // t2 = Erf(t1)
    let erf = index.producer_of(t2)?;
    if !matches!(graph.nodes[erf].op, Op::Erf)
        || graph.nodes[erf].inputs.len() != 1
        || graph.nodes[erf].outputs.len() != 1
    {
        return None;
    }
    let t1 = node_input(graph, erf, 0)?;

    // t1 = x / sqrt(2) or t1 = x * (1 / sqrt(2))
    let scale = index.producer_of(t1)?;
    let scaled = split_scalar_binary(graph, scale)?;
    if scaled.other != x {
        return None;
    }
    let scale_ok = if scaled.is_div {
        approx_eq(scaled.scalar, SQRT_2)
    } else {
        approx_eq(scaled.scalar, FRAC_1_SQRT_2)
    };
    if !scale_ok {
        return None;
    }

    // Strictness: every intermediate is a single-consumer plain intermediate of
    // the same fp32 domain, so dropping it cannot change any other value.
    for tensor in [t1, t2, t3, outer.other] {
        if !index.single_consumer(tensor) || !index.is_plain_intermediate(graph, tensor) {
            return None;
        }
        if !fp32_like(graph, tensor) {
            return None;
        }
    }
    if !fp32_like(graph, x) {
        return None;
    }
    Some(GeluMatch {
        x,
        scale,
        erf,
        add,
        mul_x,
    })
}

fn fuse_gelu(graph: &mut IrGraph, stats: &mut FuseStats) {
    let index = Index::new(graph);
    let mut remove: HashSet<usize> = HashSet::new();
    let mut fused: Vec<(usize, GeluMatch)> = Vec::new();
    for idx in 0..graph.nodes.len() {
        if !matches!(graph.nodes[idx].op, Op::Mul | Op::Div) {
            continue;
        }
        let Some(m) = match_gelu(graph, &index, idx) else {
            continue;
        };
        if [m.scale, m.erf, m.add, m.mul_x]
            .iter()
            .any(|i| remove.contains(i))
        {
            continue;
        }
        remove.insert(m.scale);
        remove.insert(m.erf);
        remove.insert(m.add);
        remove.insert(m.mul_x);
        fused.push((idx, m));
    }
    if fused.is_empty() {
        return;
    }
    for (idx, m) in &fused {
        let node = &mut graph.nodes[*idx];
        node.op = Op::Gelu;
        node.inputs = vec![Some(m.x)];
        node.outputs.truncate(1);
        node.name = format!("{}_gelu_fused", node.name);
    }
    apply_removals(graph, &remove, &HashMap::new());
    stats.gelu_fused += fused.len();
}

// ---------------------------------------------------------------------------
// SiLU: Mul(x, Sigmoid(x))
// ---------------------------------------------------------------------------

fn fuse_silu(graph: &mut IrGraph, stats: &mut FuseStats) {
    let index = Index::new(graph);
    let mut remove: HashSet<usize> = HashSet::new();
    let mut fused: Vec<(usize, TensorId)> = Vec::new();
    for idx in 0..graph.nodes.len() {
        if !matches!(graph.nodes[idx].op, Op::Mul)
            || graph.nodes[idx].inputs.len() != 2
            || graph.nodes[idx].outputs.len() != 1
        {
            continue;
        }
        let (Some(a), Some(b)) = (node_input(graph, idx, 0), node_input(graph, idx, 1)) else {
            continue;
        };
        let sigmoid_input = |tensor: TensorId| -> Option<usize> {
            let producer = index.producer_of(tensor)?;
            matches!(graph.nodes[producer].op, Op::Sigmoid).then_some(producer)
        };
        let (sigmoid, x) = match (sigmoid_input(a), sigmoid_input(b)) {
            (Some(p), None) => (p, b),
            (None, Some(p)) => (p, a),
            // Both operands are sigmoid outputs (`sigmoid * sigmoid`) or neither
            // is: not SiLU either way.
            _ => continue,
        };
        if remove.contains(&sigmoid) {
            continue;
        }
        let Some(sigmoid_out) = node_output(graph, sigmoid) else {
            continue;
        };
        if graph.nodes[sigmoid].inputs.len() != 1 || graph.nodes[sigmoid].outputs.len() != 1 {
            continue;
        }
        if !index.single_consumer(sigmoid_out)
            || !index.is_plain_intermediate(graph, sigmoid_out)
            || !fp32_like(graph, sigmoid_out)
            || !fp32_like(graph, x)
        {
            continue;
        }
        remove.insert(sigmoid);
        fused.push((idx, x));
    }
    if fused.is_empty() {
        return;
    }
    for (idx, x) in &fused {
        let node = &mut graph.nodes[*idx];
        node.op = Op::SiLU;
        node.inputs = vec![Some(*x)];
        node.outputs.truncate(1);
        node.name = format!("{}_silu_fused", node.name);
    }
    apply_removals(graph, &remove, &HashMap::new());
    stats.silu_fused += fused.len();
}

// ---------------------------------------------------------------------------
// Conv + activation epilogue
// ---------------------------------------------------------------------------

fn fuse_conv_activation(
    graph: &mut IrGraph,
    stats: &mut FuseStats,
    conv_activations: &mut HashMap<NodeId, FusedActivation>,
) {
    let index = Index::new(graph);
    let mut remove: HashSet<usize> = HashSet::new();
    let mut fused: Vec<(usize, FusedActivation, usize, TensorId)> = Vec::new();
    for idx in 0..graph.nodes.len() {
        if !matches!(graph.nodes[idx].op, Op::Conv(_)) {
            continue;
        }
        let Some(conv_out) = node_output(graph, idx) else {
            continue;
        };
        // A graph output must keep the name it was declared with; redirecting it
        // would leave the declared output tensor unproduced.
        if index.graph_outputs.contains(&conv_out) || !index.single_consumer(conv_out) {
            continue;
        }
        if !fp32_like(graph, conv_out) {
            continue;
        }
        let Some(act_idx) = index.consumer_of(conv_out) else {
            continue;
        };
        let activation = match graph.nodes[act_idx].op {
            Op::Relu => FusedActivation::Relu,
            Op::SiLU => FusedActivation::Silu,
            _ => continue,
        };
        if graph.nodes[act_idx].inputs.len() != 1 || graph.nodes[act_idx].outputs.len() != 1 {
            continue;
        }
        let Some(act_out) = node_output(graph, act_idx) else {
            continue;
        };
        if remove.contains(&act_idx) {
            continue;
        }
        remove.insert(act_idx);
        fused.push((idx, activation, act_idx, act_out));
    }
    if fused.is_empty() {
        return;
    }
    for (conv_idx, activation, _, act_out) in &fused {
        graph.nodes[*conv_idx].outputs[0] = Some(*act_out);
        conv_activations.insert(NodeId(remap_index(*conv_idx, &remove)), *activation);
    }
    apply_removals(graph, &remove, &HashMap::new());
    stats.conv_activation_fused += fused.len();
}

// ---------------------------------------------------------------------------
// Identity / redundant Cast
// ---------------------------------------------------------------------------

fn eliminate_aliases(graph: &mut IrGraph, config: FuseConfig, stats: &mut FuseStats) {
    let index = Index::new(graph);
    let mut remove: HashSet<usize> = HashSet::new();
    let mut redirects: HashMap<TensorId, TensorId> = HashMap::new();
    for idx in 0..graph.nodes.len() {
        if graph.nodes[idx].inputs.len() != 1 || graph.nodes[idx].outputs.len() != 1 {
            continue;
        }
        let (Some(inp), Some(out)) = (node_input(graph, idx, 0), node_output(graph, idx)) else {
            continue;
        };
        if inp == out || index.graph_outputs.contains(&out) {
            continue;
        }
        let in_dtype = graph.tensors[inp.0].dtype;
        let out_dtype = graph.tensors[out.0].dtype;
        let dtype_compatible = in_dtype == out_dtype
            || in_dtype == DataType::Undefined
            || out_dtype == DataType::Undefined;
        match &graph.nodes[idx].op {
            Op::Identity if config.eliminate_identity && dtype_compatible => {
                stats.identity_removed += 1;
            }
            // Only a provably redundant `Cast(T -> T)` is dropped; a cast that
            // might actually change the element width stays.
            Op::Cast { to }
                if config.eliminate_cast
                    && in_dtype == *to
                    && !matches!(to, DataType::Undefined) =>
            {
                stats.cast_removed += 1;
            }
            _ => continue,
        }
        redirects.insert(out, inp);
        remove.insert(idx);
    }
    apply_removals(graph, &remove, &redirects);
}

// ---------------------------------------------------------------------------
// Dead nodes
// ---------------------------------------------------------------------------

fn eliminate_dead_nodes(graph: &mut IrGraph, stats: &mut FuseStats) {
    // Removing one dead node can orphan its producers (`Constant` scalars left
    // behind by a fused chain), so iterate until the node list is stable.
    for _ in 0..=graph.nodes.len() {
        let index = Index::new(graph);
        let mut remove: HashSet<usize> = HashSet::new();
        for (idx, node) in graph.nodes.iter().enumerate() {
            let live = node.outputs.iter().flatten().any(|out| {
                index.graph_outputs.contains(out)
                    || index.consumers.get(out).copied().unwrap_or(0) > 0
            });
            if !live {
                remove.insert(idx);
            }
        }
        if remove.is_empty() {
            return;
        }
        stats.dead_nodes_removed += remove.len();
        apply_removals(graph, &remove, &HashMap::new());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_infer::graph::{ConvAttrs, IfAttrs, IrNode};
    use crate::native_infer::onnx_proto::TensorProto;

    // ---- tiny IR builder -------------------------------------------------

    struct Builder {
        graph: IrGraph,
    }

    impl Builder {
        fn new() -> Self {
            Self {
                graph: IrGraph {
                    name: "fuse_test".into(),
                    tensors: Vec::new(),
                    nodes: Vec::new(),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    weights: HashMap::new(),
                    name_to_id: HashMap::new(),
                },
            }
        }

        fn input(&mut self, name: &str) -> TensorId {
            let id = self.graph.register_tensor(
                name.into(),
                DataType::Float,
                Some(vec![Some(1), Some(4), Some(2), Some(2)]),
                false,
                true,
                false,
            );
            self.graph.inputs.push(id);
            id
        }

        fn mid(&mut self, name: &str) -> TensorId {
            self.graph
                .register_tensor(name.into(), DataType::Undefined, None, false, false, false)
        }

        fn output(&mut self, name: &str) -> TensorId {
            let id = self
                .graph
                .register_tensor(name.into(), DataType::Undefined, None, false, false, true);
            self.graph.outputs.push(id);
            id
        }

        fn scalar(&mut self, name: &str, value: f32) -> TensorId {
            self.graph
                .add_weight(TensorProto::from_f32(name, vec![], &[value]))
        }

        fn node(&mut self, name: &str, op: Op, inputs: &[TensorId], outputs: &[TensorId]) {
            self.graph.nodes.push(IrNode {
                id: NodeId(self.graph.nodes.len()),
                name: name.into(),
                op,
                inputs: inputs.iter().copied().map(Some).collect(),
                outputs: outputs.iter().copied().map(Some).collect(),
            });
        }

        /// `x -> Div(x, sqrt2) -> Erf -> Add(1) -> Mul(x, .) -> Mul(., 0.5)`
        fn gelu_chain(&mut self, x: TensorId, prefix: &str, scalar_first: bool) -> TensorId {
            let sqrt2 = self.scalar(&format!("{prefix}_sqrt2"), SQRT_2);
            let one = self.scalar(&format!("{prefix}_one"), 1.0);
            let half = self.scalar(&format!("{prefix}_half"), 0.5);
            let t1 = self.mid(&format!("{prefix}_t1"));
            let t2 = self.mid(&format!("{prefix}_t2"));
            let t3 = self.mid(&format!("{prefix}_t3"));
            let t4 = self.mid(&format!("{prefix}_t4"));
            let y = self.mid(&format!("{prefix}_y"));
            self.node(&format!("{prefix}_div"), Op::Div, &[x, sqrt2], &[t1]);
            self.node(&format!("{prefix}_erf"), Op::Erf, &[t1], &[t2]);
            if scalar_first {
                self.node(&format!("{prefix}_add"), Op::Add, &[one, t2], &[t3]);
                self.node(&format!("{prefix}_mul_x"), Op::Mul, &[t3, x], &[t4]);
                self.node(&format!("{prefix}_mul_half"), Op::Mul, &[half, t4], &[y]);
            } else {
                self.node(&format!("{prefix}_add"), Op::Add, &[t2, one], &[t3]);
                self.node(&format!("{prefix}_mul_x"), Op::Mul, &[x, t3], &[t4]);
                self.node(&format!("{prefix}_mul_half"), Op::Mul, &[t4, half], &[y]);
            }
            y
        }
    }

    // ---- GELU -----------------------------------------------------------

    /// Both production phases back to back, in the order `load_and_plan` uses.
    fn run_all(
        graph: &mut IrGraph,
        config: FuseConfig,
    ) -> (FuseStats, HashMap<NodeId, FusedActivation>) {
        let mut stats = fuse_activations(graph, config);
        let (tail, conv) = finalize_graph(graph, config);
        stats += tail;
        (stats, conv)
    }

    #[test]
    fn gelu_chain_collapses_to_one_node_in_both_operand_orders() {
        for scalar_first in [false, true] {
            let mut b = Builder::new();
            let x = b.input("x");
            let y = b.gelu_chain(x, "g", scalar_first);
            b.graph.outputs.push(y);
            assert_eq!(b.graph.nodes.len(), 5);

            let (stats, conv) = run_all(&mut b.graph, FuseConfig::FULL);

            assert_eq!(stats.gelu_fused, 1, "scalar_first={scalar_first}");
            assert_eq!(stats.nodes_removed(), 4);
            assert!(conv.is_empty());
            assert_eq!(b.graph.nodes.len(), 1);
            assert!(matches!(b.graph.nodes[0].op, Op::Gelu));
            assert_eq!(b.graph.nodes[0].inputs, vec![Some(x)]);
            assert_eq!(b.graph.nodes[0].outputs, vec![Some(y)]);
            b.graph
                .validate_topology()
                .expect("fused GELU graph must stay topologically valid");
        }
    }

    #[test]
    fn gelu_accepts_reciprocal_scale_and_div_by_two_epilogue() {
        // t1 = x * (1/sqrt2), y = t4 / 2
        let mut b = Builder::new();
        let x = b.input("x");
        let inv_sqrt2 = b.scalar("inv_sqrt2", FRAC_1_SQRT_2);
        let one = b.scalar("one", 1.0);
        let two = b.scalar("two", 2.0);
        let t1 = b.mid("t1");
        let t2 = b.mid("t2");
        let t3 = b.mid("t3");
        let t4 = b.mid("t4");
        let y = b.output("y");
        b.node("mul_scale", Op::Mul, &[x, inv_sqrt2], &[t1]);
        b.node("erf", Op::Erf, &[t1], &[t2]);
        b.node("add", Op::Add, &[t2, one], &[t3]);
        b.node("mul_x", Op::Mul, &[x, t3], &[t4]);
        b.node("div_two", Op::Div, &[t4, two], &[y]);

        let (stats, _) = run_all(&mut b.graph, FuseConfig::FULL);
        assert_eq!(stats.gelu_fused, 1);
        assert_eq!(b.graph.nodes.len(), 1);
        assert!(matches!(b.graph.nodes[0].op, Op::Gelu));
    }

    #[test]
    fn gelu_is_rejected_when_any_intermediate_has_a_second_consumer() {
        // The `erf` output is also read by an extra node: the chain must stay.
        let mut b = Builder::new();
        let x = b.input("x");
        let y = b.gelu_chain(x, "g", false);
        b.graph.outputs.push(y);
        let t2 = b.graph.name_to_id["g_t2"];
        let side = b.mid("side");
        b.node("side_relu", Op::Relu, &[t2], &[side]);
        b.graph.outputs.push(side);

        let before = b.graph.nodes.len();
        let (stats, _) = run_all(&mut b.graph, FuseConfig::FULL);
        assert_eq!(stats.gelu_fused, 0);
        assert_eq!(b.graph.nodes.len(), before);
    }

    #[test]
    fn gelu_is_rejected_when_an_intermediate_is_a_graph_output() {
        let mut b = Builder::new();
        let x = b.input("x");
        let y = b.gelu_chain(x, "g", false);
        let t3 = b.graph.name_to_id["g_t3"];
        b.graph.outputs.push(y);
        // `t3` is exposed as an output as well.
        b.graph.outputs.push(t3);
        b.graph.tensors[t3.0].is_graph_output = true;

        let before = b.graph.nodes.len();
        let (stats, _) = run_all(&mut b.graph, FuseConfig::FULL);
        assert_eq!(stats.gelu_fused, 0);
        assert_eq!(b.graph.nodes.len(), before);
    }

    #[test]
    fn gelu_is_rejected_for_fp16_graphs() {
        let mut b = Builder::new();
        let x = b.input("x");
        b.graph.tensors[x.0].dtype = DataType::Float16;
        let y = b.gelu_chain(x, "g", false);
        b.graph.outputs.push(y);

        let before = b.graph.nodes.len();
        let (stats, _) = run_all(&mut b.graph, FuseConfig::FULL);
        assert_eq!(stats.gelu_fused, 0);
        assert!(stats.skipped_fp16, "fp16 graphs must report the guard");
        assert_eq!(b.graph.nodes.len(), before);
    }

    #[test]
    fn gelu_is_rejected_when_the_scale_constant_is_wrong() {
        let mut b = Builder::new();
        let x = b.input("x");
        let bad = b.scalar("bad", 1.5);
        let one = b.scalar("one", 1.0);
        let half = b.scalar("half", 0.5);
        let t1 = b.mid("t1");
        let t2 = b.mid("t2");
        let t3 = b.mid("t3");
        let t4 = b.mid("t4");
        let y = b.mid("y");
        b.node("scale", Op::Div, &[x, bad], &[t1]);
        b.node("erf", Op::Erf, &[t1], &[t2]);
        b.node("add", Op::Add, &[t2, one], &[t3]);
        b.node("mul_x", Op::Mul, &[x, t3], &[t4]);
        b.node("mul_half", Op::Mul, &[t4, half], &[y]);
        b.graph.outputs.push(y);

        let (stats, _) = run_all(&mut b.graph, FuseConfig::FULL);
        assert_eq!(stats.gelu_fused, 0);
    }

    // ---- SiLU -----------------------------------------------------------

    #[test]
    fn silu_pair_collapses_and_respects_multi_consumer() {
        let mut b = Builder::new();
        let x = b.input("x");
        let sig = b.mid("sig");
        let y = b.mid("y");
        b.node("sigmoid", Op::Sigmoid, &[x], &[sig]);
        b.node("mul", Op::Mul, &[sig, x], &[y]);
        b.graph.outputs.push(y);

        let (stats, _) = run_all(&mut b.graph, FuseConfig::FULL);
        assert_eq!(stats.silu_fused, 1);
        assert_eq!(b.graph.nodes.len(), 1);
        assert!(matches!(b.graph.nodes[0].op, Op::SiLU));
        assert_eq!(b.graph.nodes[0].inputs, vec![Some(x)]);
        b.graph.validate_topology().unwrap();

        // Same pattern, but `sig` feeds two consumers -> must not fuse.
        let mut b = Builder::new();
        let x = b.input("x");
        let sig = b.mid("sig");
        let y = b.mid("y");
        let side = b.mid("side");
        b.node("sigmoid", Op::Sigmoid, &[x], &[sig]);
        b.node("mul", Op::Mul, &[sig, x], &[y]);
        b.node("side_relu", Op::Relu, &[sig], &[side]);
        b.graph.outputs.push(y);
        let (stats, _) = run_all(&mut b.graph, FuseConfig::FULL);
        assert_eq!(stats.silu_fused, 0);
    }

    // ---- Conv + activation ---------------------------------------------

    fn conv_graph(multi_consumer: bool) -> Builder {
        let mut b = Builder::new();
        let x = b.input("x");
        let w = b.graph.add_weight(TensorProto::from_f32(
            "w",
            vec![4, 4, 1, 1],
            &[0.25; 16],
        ));
        let conv_out = b.mid("conv_out");
        let act_out = b.mid("act_out");
        b.node(
            "conv",
            Op::Conv(ConvAttrs {
                kernel_shape: Some(vec![1, 1]),
                strides: vec![1, 1],
                pads: vec![0, 0, 0, 0],
                dilations: vec![1, 1],
                group: 1,
            }),
            &[x, w],
            &[conv_out],
        );
        b.node("relu", Op::Relu, &[conv_out], &[act_out]);
        b.graph.outputs.push(act_out);
        if multi_consumer {
            let side = b.mid("side");
            b.node("side_relu", Op::Relu, &[conv_out], &[side]);
            b.graph.outputs.push(side);
        }
        b
    }

    #[test]
    fn conv_absorbs_its_single_consumer_activation() {
        let mut b = conv_graph(false);
        let (stats, conv) = run_all(&mut b.graph, FuseConfig::FULL);
        assert_eq!(stats.conv_activation_fused, 1);
        assert_eq!(b.graph.nodes.len(), 1);
        assert_eq!(conv.get(&NodeId(0)), Some(&FusedActivation::Relu));
        let act_out = b.graph.name_to_id["act_out"];
        assert_eq!(b.graph.nodes[0].outputs, vec![Some(act_out)]);
        b.graph.validate_topology().unwrap();
    }

    #[test]
    fn conv_keeps_activation_when_its_output_is_shared() {
        let mut b = conv_graph(true);
        let before = b.graph.nodes.len();
        let (stats, conv) = run_all(&mut b.graph, FuseConfig::FULL);
        assert_eq!(stats.conv_activation_fused, 0);
        assert!(conv.is_empty());
        assert_eq!(b.graph.nodes.len(), before);
    }

    #[test]
    fn conv_activation_table_follows_node_reindexing() {
        // Two dead `Constant`-like nodes first, so the Conv node id shifts.
        let mut b = conv_graph(false);
        let stats_before = run_all(&mut b.graph, FuseConfig::SAFE).0;
        assert_eq!(stats_before.conv_activation_fused, 0);
        // Prepend an orphan chain: an unused Relu over the input.
        let orphan = b.mid("orphan");
        b.graph.nodes.insert(
            0,
            IrNode {
                id: NodeId(0),
                name: "orphan_relu".into(),
                op: Op::Relu,
                inputs: vec![Some(b.graph.inputs[0])],
                outputs: vec![Some(orphan)],
            },
        );
        for (i, node) in b.graph.nodes.iter_mut().enumerate() {
            node.id = NodeId(i);
        }
        let (stats, conv) = run_all(&mut b.graph, FuseConfig::FULL);
        assert_eq!(stats.dead_nodes_removed, 1);
        assert_eq!(stats.conv_activation_fused, 1);
        // conv ended up at index 0 after the orphan was dropped.
        assert_eq!(conv.get(&NodeId(0)), Some(&FusedActivation::Relu));
        assert!(matches!(b.graph.nodes[0].op, Op::Conv(_)));
    }

    // ---- Identity / Cast ------------------------------------------------

    #[test]
    fn identity_and_redundant_cast_are_redirected() {
        let mut b = Builder::new();
        let x = b.input("x");
        let through = b.graph.add_weight(TensorProto::from_f32(
            "w",
            vec![4, 4, 1, 1],
            &[0.5; 16],
        ));
        let conv_out = b.mid("conv_out");
        let id1 = b.mid("id1");
        let id2 = b.mid("id2");
        let cast_out = b.mid("cast_out");
        let y = b.mid("y");
        b.node(
            "conv",
            Op::Conv(ConvAttrs {
                kernel_shape: Some(vec![1, 1]),
                strides: vec![1, 1],
                pads: vec![0, 0, 0, 0],
                dilations: vec![1, 1],
                group: 1,
            }),
            &[x, through],
            &[conv_out],
        );
        b.node("id1", Op::Identity, &[conv_out], &[id1]);
        b.node("id2", Op::Identity, &[id1], &[id2]);
        b.node("cast", Op::Cast { to: DataType::Float }, &[id2], &[cast_out]);
        b.node("relu", Op::Relu, &[cast_out], &[y]);
        b.graph.outputs.push(y);

        // `finalize_graph` runs after shape inference in production: only then is the
        // Cast's input dtype known, and only a *proven* `Cast(Float -> Float)` goes.
        b.graph.infer_shapes(&HashMap::new()).expect("shape inference");

        let (stats, _) = run_all(&mut b.graph, FuseConfig::SAFE);
        assert_eq!(stats.identity_removed, 2);
        assert_eq!(stats.cast_removed, 1);
        assert_eq!(b.graph.nodes.len(), 2);
        assert!(matches!(b.graph.nodes[0].op, Op::Conv(_)));
        assert_eq!(b.graph.nodes[1].inputs, vec![Some(conv_out)]);
        b.graph.validate_topology().unwrap();
    }

    #[test]
    fn identity_that_produces_a_graph_output_is_kept() {
        let mut b = Builder::new();
        let x = b.input("x");
        let y = b.output("y");
        b.node("id", Op::Identity, &[x], &[y]);
        let (stats, _) = run_all(&mut b.graph, FuseConfig::SAFE);
        assert_eq!(stats.identity_removed, 0);
        assert_eq!(b.graph.nodes.len(), 1);
    }

    #[test]
    fn casting_cast_is_kept() {
        let mut b = Builder::new();
        let x = b.input("x");
        let cast = b.mid("cast");
        let y = b.mid("y");
        b.node("cast16", Op::Cast { to: DataType::Float16 }, &[x], &[cast]);
        b.node("relu", Op::Relu, &[cast], &[y]);
        b.graph.outputs.push(y);
        let (stats, _) = run_all(&mut b.graph, FuseConfig::SAFE);
        assert_eq!(stats.cast_removed, 0);
        assert_eq!(b.graph.nodes.len(), 2);
    }

    // ---- dead nodes / control flow --------------------------------------

    #[test]
    fn dead_nodes_are_removed_to_a_fixpoint() {
        let mut b = Builder::new();
        let x = b.input("x");
        let a = b.mid("a");
        let dead_a = b.mid("dead_a");
        let dead_b = b.mid("dead_b");
        let y = b.mid("y");
        b.node("keep", Op::Relu, &[x], &[a]);
        b.node("dead1", Op::Relu, &[a], &[dead_a]);
        b.node("dead2", Op::Relu, &[dead_a], &[dead_b]);
        b.node("keep2", Op::Relu, &[a], &[y]);
        b.graph.outputs.push(y);

        let (stats, _) = run_all(&mut b.graph, FuseConfig::SAFE);
        assert_eq!(stats.dead_nodes_removed, 2);
        assert_eq!(b.graph.nodes.len(), 2);
        b.graph.validate_topology().unwrap();
    }

    #[test]
    fn graphs_with_control_flow_are_left_untouched() {
        let mut b = Builder::new();
        let x = b.input("x");
        let y = b.gelu_chain(x, "g", false);
        b.graph.outputs.push(y);
        let empty = || IrGraph {
            name: "branch".into(),
            tensors: Vec::new(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            weights: HashMap::new(),
            name_to_id: HashMap::new(),
        };
        let flag = b.mid("flag");
        let if_out = b.mid("if_out");
        b.graph.outputs.push(if_out);
        b.node(
            "cond",
            Op::If(IfAttrs {
                then_branch: Box::new(empty()),
                else_branch: Box::new(empty()),
            }),
            &[flag],
            &[if_out],
        );

        let before = b.graph.nodes.len();
        let (stats, _) = run_all(&mut b.graph, FuseConfig::FULL);
        assert!(stats.is_noop());
        assert!(stats.skipped_control_flow);
        assert_eq!(b.graph.nodes.len(), before);
    }

    // ---- configuration / idempotency ------------------------------------

    #[test]
    fn fuse_level_parsing_and_defaults() {
        assert_eq!(FuseLevel::parse("0"), FuseLevel::Off);
        assert_eq!(FuseLevel::parse(" OFF "), FuseLevel::Off);
        assert_eq!(FuseLevel::parse("safe"), FuseLevel::Safe);
        assert_eq!(FuseLevel::parse("1"), FuseLevel::Safe);
        assert_eq!(FuseLevel::parse("all"), FuseLevel::Full);
        assert_eq!(FuseLevel::parse("2"), FuseLevel::Full);
        assert_eq!(FuseLevel::parse("nonsense"), DEFAULT_FUSE_LEVEL);
        assert_eq!(DEFAULT_FUSE_LEVEL, FuseLevel::Safe);
        assert!(FuseConfig::OFF.is_noop());
        assert!(!FuseConfig::SAFE.is_noop());
        assert!(!FuseConfig::FULL.is_noop());
        assert_eq!(FuseConfig::default(), FuseLevel::Safe.config());
        // `AURORA_NATIVE_CONV_EPILOGUE` semantics (pure helper, no env mutation).
        assert!(!env_flag_is_on(None));
        assert!(!env_flag_is_on(Some("0")));
        assert!(!env_flag_is_on(Some("yes")));
        assert!(env_flag_is_on(Some("1")));
        assert!(env_flag_is_on(Some(" 1 ")));
    }

    #[test]
    fn rewrites_are_idempotent() {
        let mut b = Builder::new();
        let x = b.input("x");
        let y1 = b.gelu_chain(x, "g", false);
        let sig = b.mid("sig");
        let y2 = b.mid("y2");
        b.node("sigmoid", Op::Sigmoid, &[y1], &[sig]);
        b.node("mul", Op::Mul, &[sig, y1], &[y2]);
        b.graph.outputs.push(y2);

        let (first, _) = run_all(&mut b.graph, FuseConfig::FULL);
        assert_eq!(first.gelu_fused, 1);
        assert_eq!(first.silu_fused, 1);
        let nodes_after = b.graph.nodes.len();

        let (second, conv) = run_all(&mut b.graph, FuseConfig::FULL);
        assert!(second.is_noop(), "second pass must be a no-op: {second:?}");
        assert!(conv.is_empty());
        assert_eq!(b.graph.nodes.len(), nodes_after);
        b.graph.validate_topology().unwrap();
    }

    // ---- numerical equivalence ------------------------------------------

    /// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7 (plenty for a 1e-6 budget).
    fn erf_ref(x: f64) -> f64 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let poly = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592;
        sign * (1.0 - poly * t * (-x * x).exp())
    }

    fn gelu_chain_f32(x: f32) -> f32 {
        let t1 = x / SQRT_2;
        let t2 = erf_ref(t1 as f64) as f32;
        let t3 = t2 + 1.0;
        let t4 = x * t3;
        t4 * 0.5
    }

    fn gelu_fused_f32(x: f32) -> f32 {
        // Same expression as the `Gelu` case of `unary_f32` in
        // `native/cuda/infer_elem_norm.cu`.
        0.5 * x * (1.0 + erf_ref((x as f64) * std::f64::consts::FRAC_1_SQRT_2) as f32)
    }

    #[test]
    fn fused_gelu_stays_within_one_ulp_of_the_stepwise_chain() {
        let mut max_diff = 0.0f32;
        let mut worst = 0.0f32;
        for step in -8000..=8000 {
            let x = step as f32 * 0.004; // -32 .. 32, the range where erf matters
            let chain = gelu_chain_f32(x);
            let fused = gelu_fused_f32(x);
            let diff = (chain - fused).abs();
            if diff > max_diff {
                max_diff = diff;
                worst = x;
            }
        }
        assert!(
            max_diff < 1e-6,
            "GELU fusion changed the value: max_abs_diff={max_diff} at x={worst}"
        );
        // Peak magnitude of |gelu| in this range keeps the absolute budget honest.
        assert!(gelu_fused_f32(3.0) > 2.98, "gelu(3) sanity");
    }

    #[test]
    fn fused_gelu_and_silu_keep_infinity_and_nan_behaviour() {
        for x in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let chain = gelu_chain_f32(x);
            let fused = gelu_fused_f32(x);
            assert_eq!(chain.is_nan(), fused.is_nan(), "gelu boundary x={x}");
            if !chain.is_nan() {
                assert_eq!(chain.is_infinite(), fused.is_infinite(), "gelu boundary x={x}");
                assert_eq!(chain.signum(), fused.signum(), "gelu boundary x={x}");
            }

            // SiLU: `x * sigmoid(x)` (kernel: `x / (1 + exp(-x))`).
            let chain_silu = x * (1.0 / (1.0 + (-x).exp()));
            let fused_silu = x / (1.0 + (-x).exp());
            assert_eq!(chain_silu.is_nan(), fused_silu.is_nan(), "silu boundary x={x}");
        }

        // Finite sweep: |x * sigmoid(x) - x / (1 + exp(-x))| stays tiny.
        let mut max_diff = 0.0f32;
        for step in -4000..=4000 {
            let x = step as f32 * 0.01;
            let a = x * (1.0 / (1.0 + (-x).exp()));
            let b = x / (1.0 + (-x).exp());
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(max_diff < 1e-6, "SiLU fusion max_abs_diff={max_diff}");
    }

    // Helper removed: the SiLU matcher spells the rejection out with `continue`.
}
