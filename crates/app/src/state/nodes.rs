//! Layers as nodes, after DaVinci Resolve: each layer is a corrector node, and
//! links decide what each one grades. Serial nodes chain; parallel and layer
//! mixers join branches that start from the same image. The graph is stored
//! with the adjustments (`AdjustmentSet::nodes`) and the layer list is kept in
//! its render order, so engines that only read the list still draw it.

use gpui_kit::{Context, SharedString};
use aurora_engine::{AdjustmentSet, GraphError, GraphNode, Layer, LayerKind, Mask, MaskOp, MaskShape, NodeGraph, NodeKind, NodeLink, RadialGradient};

use super::AppState;
use crate::i18n::t;

/// Editor units between columns and rows of nodes.
pub const COL: f32 = 104.0;
pub const ROW: f32 = 64.0;

pub const INPUT: u64 = NodeGraph::INPUT;
pub const OUTPUT: u64 = NodeGraph::OUTPUT;

/// What a new node does relative to the selected one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NewNode {
    /// Grades the selected node's output.
    Serial,
    /// Grades the selected node's input beside it; a parallel mixer blends both.
    Parallel,
    /// Like parallel, joined by a layer mixer, the new branch on top.
    Layer,
    /// Grades what the selected node's mask leaves out.
    Outside,
}

impl NewNode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Serial => t("Add Serial Node", "添加串行节点"),
            Self::Parallel => t("Add Parallel Node", "添加并行节点"),
            Self::Layer => t("Add Layer Node", "添加图层节点"),
            Self::Outside => t("Add Outside Node", "添加外部节点"),
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Self::Serial => t("Serial", "串行"),
            Self::Parallel => t("Parallel", "并行"),
            Self::Layer => t("Layer", "图层"),
            Self::Outside => t("Outside", "外部"),
        }
    }

    pub fn keys(self) -> &'static str {
        match self {
            Self::Serial => "Alt+S",
            Self::Parallel => "Alt+Shift+P",
            Self::Layer => "Alt+L",
            Self::Outside => "Alt+Shift+O",
        }
    }
}

/// A mask over the whole frame: a corrector node grades everything until a
/// window or a key narrows it, as in Resolve.
pub fn full_mask() -> Mask {
    Mask {
        shapes: vec![MaskShape::Radial(RadialGradient { op: MaskOp::Add, center: [0.5, 0.5], radius: [2.0, 2.0], rotation: 0.0, feather: 0.0, inverted: false })],
        ..Default::default()
    }
}

/// Whether the mask is only the whole-frame fill.
pub fn is_full(mask: &Mask) -> bool {
    matches!(&mask.shapes[..], [MaskShape::Radial(r)] if r.op == MaskOp::Add && r.radius[0] >= 2.0 && r.radius[1] >= 2.0)
}

/// Adds a shape to a mask; the first shape that adds area replaces the
/// whole-frame fill, so drawing a window narrows the node.
pub fn add_shape(mask: &mut Mask, shape: MaskShape) {
    let adds = match &shape {
        MaskShape::Brush(b) => b.op == MaskOp::Add,
        MaskShape::Linear(_) | MaskShape::Radial(_) | MaskShape::Ai(_) => true,
        MaskShape::Raster(r) => r.op == MaskOp::Add,
        MaskShape::Copy { op, .. } => *op == MaskOp::Add,
    };
    if adds && is_full(mask) {
        mask.shapes.clear();
    }
    mask.shapes.push(shape);
}

/// Graph helpers the interface needs beyond the engine's data.
pub trait GraphExt {
    fn has(&self, id: u64) -> bool;
    fn pos(&self, id: u64) -> [f32; 2];
    fn set_pos(&mut self, id: u64, pos: [f32; 2]);
    fn outputs(&self, id: u64) -> Vec<NodeLink>;
    /// Inputs a node shows: mixers one more than they use, so a branch can join.
    fn input_count(&self, id: u64) -> u32;
    fn link(&mut self, from: u64, to: u64, input: u32);
    fn unlink(&mut self, to: u64, input: u32);
    /// Removes `id`, joining what fed its first input to what it fed.
    fn bypass(&mut self, id: u64);
    /// Moves everything downstream of `id` right by `dx`.
    fn shift_after(&mut self, id: u64, dx: f32);
}

impl GraphExt for NodeGraph {
    fn has(&self, id: u64) -> bool {
        self.nodes.iter().any(|n| n.id == id)
    }

    fn pos(&self, id: u64) -> [f32; 2] {
        self.node(id).map(|n| n.pos).unwrap_or_default()
    }

    fn set_pos(&mut self, id: u64, pos: [f32; 2]) {
        if let Some(n) = self.nodes.iter_mut().find(|n| n.id == id) {
            n.pos = pos;
        }
    }

    fn outputs(&self, id: u64) -> Vec<NodeLink> {
        self.links.iter().filter(|l| l.from == id).copied().collect()
    }

    fn input_count(&self, id: u64) -> u32 {
        match self.node(id).map(|n| n.kind) {
            Some(NodeKind::Input) | None => 0,
            Some(NodeKind::Parallel | NodeKind::LayerMixer) => {
                self.links.iter().filter(|l| l.to == id).map(|l| l.input + 1).max().unwrap_or(0).max(2) + 1
            }
            _ => 1,
        }
    }

    fn link(&mut self, from: u64, to: u64, input: u32) {
        self.unlink(to, input);
        self.links.push(NodeLink { from, to, input });
    }

    fn unlink(&mut self, to: u64, input: u32) {
        self.links.retain(|l| !(l.to == to && l.input == input));
    }

    fn bypass(&mut self, id: u64) {
        let src = self.source(id, 0);
        let outs = self.outputs(id);
        self.links.retain(|l| l.from != id && l.to != id);
        if let Some(src) = src {
            for o in outs {
                self.link(src, o.to, o.input);
            }
        }
        self.nodes.retain(|n| n.id != id);
    }

    fn shift_after(&mut self, id: u64, dx: f32) {
        let ids: Vec<u64> = self.nodes.iter().map(|n| n.id).filter(|n| *n != id && self.reaches(id, *n)).collect();
        for n in self.nodes.iter_mut().filter(|n| ids.contains(&n.id)) {
            n.pos[0] += dx;
        }
    }
}

/// The graph for a set of adjustments, repaired to match its layers: removed
/// layers are bypassed, layers it does not know yet join in series before the
/// output. Without a stored graph, the layers form one serial chain.
///
/// The repair itself lives in the engine (`NodeGraph::repaired`), so the
/// interface and the render/export path cannot disagree about which layers a
/// stored document has.
pub fn normalized(adj: &AdjustmentSet) -> NodeGraph {
    adj.nodes.clone().unwrap_or_default().repaired(&adj.layers)
}

/// The adjustments as they should render: only the layers on the graph's
/// path, in its order. Disconnected nodes stay in the saved list.
///
/// A graph the engine refuses (a cycle) renders the background only: inventing
/// an order would show a frame the graph cannot explain. Callers that can
/// report the reason use [`flattened_checked`].
pub fn flattened(adj: &AdjustmentSet) -> AdjustmentSet {
    match flattened_checked(adj) {
        Ok(out) => out,
        Err(error) => {
            static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            if LOGGED.set(()).is_ok() {
                log::error!("node graph refused: {error}; showing the background only");
            }
            AdjustmentSet { layers: Vec::new(), ..adj.clone() }
        }
    }
}

/// Result of backward reachability pruning from `OUTPUT` with cycle detection.
#[derive(Clone, Debug, PartialEq)]
pub struct PrunedGraph {
    /// Compact graph containing only nodes and links on live paths to `OUTPUT`.
    pub graph: NodeGraph,
    /// Live node ids in topological evaluation order.
    pub active_nodes: Vec<u64>,
    /// Live `NodeKind::Layer` ids in render order.
    pub active_layers: Vec<u64>,
    /// Non-I/O node ids removed because they cannot reach `OUTPUT` (or `INPUT`).
    pub pruned_nodes: Vec<u64>,
    /// Layer node ids removed during pruning.
    pub pruned_layers: Vec<u64>,
}

/// Semantics of a multi-input mixer node in the node graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixerSemantics {
    /// Parallel mixer: branches grade the same upstream image independently and
    /// are blended equally without earlier branches contaminating later ones.
    ParallelBlend,
    /// Layer mixer: branches are stacked in input-port order (higher input port
    /// composites over lower input port, top wins).
    LayerStackTopWins,
}

impl MixerSemantics {
    pub fn from_kind(kind: NodeKind) -> Option<Self> {
        match kind {
            NodeKind::Parallel => Some(Self::ParallelBlend),
            NodeKind::LayerMixer => Some(Self::LayerStackTopWins),
            _ => None,
        }
    }

    /// Blending weights for `branch_count` connected inputs.
    pub fn branch_weights(self, branch_count: usize) -> Vec<f32> {
        if branch_count == 0 {
            return Vec::new();
        }
        match self {
            Self::ParallelBlend => vec![1.0 / branch_count as f32; branch_count],
            Self::LayerStackTopWins => vec![1.0; branch_count],
        }
    }

    /// Whether higher-numbered input ports override lower-numbered ports.
    pub fn overrides_previous(self) -> bool {
        matches!(self, Self::LayerStackTopWins)
    }
}

/// One input branch feeding a mixer node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixerBranch {
    pub input: u32,
    pub steps: Vec<EvalStep>,
}

/// Structured evaluation step preserving Parallel vs. Layer mixer semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalStep {
    Layer(u64),
    Mixer {
        id: u64,
        semantics: MixerSemantics,
        branches: Vec<MixerBranch>,
    },
}

/// Structured evaluation plan produced from a pruned, acyclic node graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuredPlan {
    pub steps: Vec<EvalStep>,
    /// Linearized layer render order compatible with flat-layer renderers.
    pub flat_layers: Vec<u64>,
    /// Detached/unreachable layer ids pruned before evaluation.
    pub pruned_layers: Vec<u64>,
}

/// Walks backwards from [`OUTPUT`] to prune disconnected/orphan nodes and
/// detect cycles on the output path.
pub fn prune_reachable_dag(graph: &NodeGraph) -> Result<PrunedGraph, GraphError> {
    use std::collections::HashSet;

    let valid_ids: HashSet<u64> = graph.nodes.iter().map(|n| n.id).collect();
    let clean_links: Vec<NodeLink> = graph
        .links
        .iter()
        .copied()
        .filter(|l| valid_ids.contains(&l.from) && valid_ids.contains(&l.to))
        .collect();

    // Backward DFS from OUTPUT with explicit visiting stack for cycle detection.
    fn visit_backward(
        id: u64,
        links: &[NodeLink],
        visiting: &mut Vec<u64>,
        visited: &mut HashSet<u64>,
        post_order: &mut Vec<u64>,
    ) -> Result<(), GraphError> {
        if visited.contains(&id) {
            return Ok(());
        }
        if let Some(pos) = visiting.iter().position(|&n| n == id) {
            return Err(GraphError::Cycle { nodes: visiting[pos..].to_vec() });
        }
        visiting.push(id);
        let mut incoming: Vec<NodeLink> = links.iter().copied().filter(|l| l.to == id).collect();
        incoming.sort_by_key(|l| l.input);
        for link in incoming {
            visit_backward(link.from, links, visiting, visited, post_order)?;
        }
        visiting.pop();
        visited.insert(id);
        post_order.push(id);
        Ok(())
    }

    let mut post_order = Vec::new();
    let mut visited = HashSet::new();
    let mut visiting = Vec::new();
    if valid_ids.contains(&OUTPUT) {
        visit_backward(OUTPUT, &clean_links, &mut visiting, &mut visited, &mut post_order)?;
    }

    // Intersect with forward reachability from INPUT when INPUT exists.
    let active_nodes: Vec<u64> = if valid_ids.contains(&INPUT) {
        let mut forward = HashSet::new();
        let mut stack = vec![INPUT];
        while let Some(id) = stack.pop() {
            if !forward.insert(id) {
                continue;
            }
            for link in clean_links.iter().filter(|l| l.from == id) {
                stack.push(link.to);
            }
        }
        post_order.into_iter().filter(|id| forward.contains(id)).collect()
    } else {
        post_order
    };

    let active_set: HashSet<u64> = active_nodes.iter().copied().collect();
    let mut keep_set = active_set.clone();
    if valid_ids.contains(&INPUT) {
        keep_set.insert(INPUT);
    }
    if valid_ids.contains(&OUTPUT) {
        keep_set.insert(OUTPUT);
    }

    let active_layers: Vec<u64> = active_nodes
        .iter()
        .copied()
        .filter(|id| graph.node(*id).is_some_and(|n| n.kind == NodeKind::Layer))
        .collect();

    let mut pruned_nodes: Vec<u64> = graph
        .nodes
        .iter()
        .filter(|n| n.id != INPUT && n.id != OUTPUT && !active_set.contains(&n.id))
        .map(|n| n.id)
        .collect();
    pruned_nodes.sort_unstable();
    pruned_nodes.dedup();

    let mut pruned_layers: Vec<u64> = graph
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::Layer && !active_set.contains(&n.id))
        .map(|n| n.id)
        .collect();
    pruned_layers.sort_unstable();
    pruned_layers.dedup();

    let pruned_graph = NodeGraph {
        nodes: graph.nodes.iter().filter(|n| keep_set.contains(&n.id)).cloned().collect(),
        links: clean_links
            .into_iter()
            .filter(|l| active_set.contains(&l.from) && active_set.contains(&l.to))
            .collect(),
    };

    Ok(PrunedGraph {
        graph: pruned_graph,
        active_nodes,
        active_layers,
        pruned_nodes,
        pruned_layers,
    })
}

/// Builds a structured evaluation plan that preserves Parallel / Layer mixer
/// branch topology while pruning unreachable nodes and rejecting cycles.
pub fn build_structured_plan(adj: &AdjustmentSet) -> Result<StructuredPlan, GraphError> {
    use std::collections::HashSet;

    let Some(_) = adj.nodes.as_ref() else {
        let flat_layers: Vec<u64> = adj.layers.iter().map(|l| l.id).collect();
        let steps = flat_layers.iter().copied().map(EvalStep::Layer).collect();
        return Ok(StructuredPlan {
            steps,
            flat_layers,
            pruned_layers: Vec::new(),
        });
    };

    let repaired = normalized(adj);
    let pruned = prune_reachable_dag(&repaired)?;
    let g = &pruned.graph;

    fn ancestors_of(g: &NodeGraph, start: u64) -> HashSet<u64> {
        let mut set = HashSet::new();
        let mut stack = vec![start];
        while let Some(id) = stack.pop() {
            if !set.insert(id) {
                continue;
            }
            for link in g.links.iter().filter(|l| l.to == id) {
                stack.push(link.from);
            }
        }
        set
    }

    fn collect_steps(g: &NodeGraph, id: u64, stop_at: &HashSet<u64>, out: &mut Vec<EvalStep>) {
        if id == INPUT || stop_at.contains(&id) {
            return;
        }
        let Some(node) = g.node(id) else { return };
        match node.kind {
            NodeKind::Output => {
                if let Some(src) = g.source(id, 0) {
                    collect_steps(g, src, stop_at, out);
                }
            }
            NodeKind::Layer => {
                if let Some(src) = g.source(id, 0) {
                    collect_steps(g, src, stop_at, out);
                }
                out.push(EvalStep::Layer(id));
            }
            NodeKind::Parallel | NodeKind::LayerMixer => {
                let semantics = MixerSemantics::from_kind(node.kind).expect("mixer kind");
                let mut incoming: Vec<NodeLink> = g.links.iter().copied().filter(|l| l.to == id).collect();
                incoming.sort_by_key(|l| l.input);

                // Shared ancestors common to all connected branches run once before the mixer.
                let mut common: Option<HashSet<u64>> = None;
                for link in &incoming {
                    let anc = ancestors_of(g, link.from);
                    common = Some(match common {
                        None => anc,
                        Some(prev) => prev.intersection(&anc).copied().collect(),
                    });
                }
                let common = common.unwrap_or_default();

                // Emit the deepest shared trunk node(s) before the mixer.
                let mut shared_roots: Vec<u64> = common
                    .iter()
                    .copied()
                    .filter(|&c| {
                        c != INPUT
                            && !stop_at.contains(&c)
                            && !g
                                .links
                                .iter()
                                .any(|l| l.from == c && common.contains(&l.to) && l.to != c)
                    })
                    .collect();
                shared_roots.sort_unstable();
                for root in &shared_roots {
                    collect_steps(g, *root, stop_at, out);
                }

                let mut branch_stop = stop_at.clone();
                branch_stop.extend(common);
                let mut branches = Vec::with_capacity(incoming.len());
                for link in incoming {
                    let mut branch_steps = Vec::new();
                    collect_steps(g, link.from, &branch_stop, &mut branch_steps);
                    branches.push(MixerBranch {
                        input: link.input,
                        steps: branch_steps,
                    });
                }
                out.push(EvalStep::Mixer {
                    id,
                    semantics,
                    branches,
                });
            }
            NodeKind::Input => {}
        }
    }

    let mut steps = Vec::new();
    collect_steps(g, OUTPUT, &HashSet::new(), &mut steps);

    Ok(StructuredPlan {
        steps,
        flat_layers: pruned.active_layers,
        pruned_layers: pruned.pruned_layers,
    })
}

/// [`flattened`], with the graph's refusal instead of a fallback frame.
pub fn flattened_checked(adj: &AdjustmentSet) -> Result<AdjustmentSet, GraphError> {
    let mut out = adj.clone();
    if adj.nodes.is_some() {
        let repaired = normalized(adj);
        let pruned = prune_reachable_dag(&repaired)?;
        out.nodes = Some(pruned.graph);
        out.layers = pruned
            .active_layers
            .iter()
            .filter_map(|id| out.layers.iter().find(|l| l.id == *id).cloned())
            .collect();
    }
    Ok(out)
}

/// Stores `g` and puts the layer list in its render order, disconnected
/// layers last.
fn store(a: &mut AdjustmentSet, g: NodeGraph) {
    let order = g.render_order();
    a.layers.sort_by_key(|l| order.iter().position(|id| *id == l.id).unwrap_or(usize::MAX));
    a.nodes = Some(g);
}

impl AppState {
    pub fn node_graph(&self) -> NodeGraph {
        normalized(self.adj())
    }

    /// The selected node: the active layer, a selected mixer, or the input.
    pub fn selected_node(&self) -> u64 {
        self.editor.active_layer.or(self.editor.active_node).unwrap_or(INPUT)
    }

    pub fn select_node(&mut self, id: u64, cx: &mut Context<Self>) {
        let is_layer = self.adj().layers.iter().any(|l| l.id == id);
        self.editor.active_layer = is_layer.then_some(id);
        self.editor.active_node = (!is_layer && id != INPUT).then_some(id);
        cx.notify();
    }

    /// An id no layer or node of this image uses.
    pub(crate) fn fresh_node_id(&mut self) -> u64 {
        let adj = self.adj();
        let used = adj.layers.iter().map(|l| l.id).chain(adj.nodes.iter().flat_map(|g| g.nodes.iter().map(|n| n.id)).filter(|id| *id < INPUT)).max().unwrap_or(0);
        self.editor.next_layer_id(used)
    }

    /// Edits the graph as one undo step and keeps the layer list in step.
    pub fn edit_graph(&mut self, label: impl Into<SharedString>, live: bool, f: impl FnOnce(&mut NodeGraph, &mut Vec<Layer>), cx: &mut Context<Self>) {
        let apply = move |a: &mut AdjustmentSet| {
            let mut g = normalized(a);
            f(&mut g, &mut a.layers);
            store(a, g);
        };
        if live { self.edit_live(label, apply, cx) } else { self.edit(label, apply, cx) }
    }

    /// Adds a corrector node next to the selected one, as Resolve's Alt+S / P / L / O do.
    pub fn add_node(&mut self, how: NewNode, cx: &mut Context<Self>) {
        self.add_node_of(how, LayerKind::Adjustment, cx);
    }

    /// Adds a node of `kind`. Correctors start over the whole frame; heal,
    /// clone and fill nodes start empty, to be painted.
    pub fn add_node_of(&mut self, how: NewNode, kind: LayerKind, cx: &mut Context<Self>) {
        if self.library.current.is_none() {
            return;
        }
        let sel = self.selected_node();
        let sel = if sel == OUTPUT { self.node_graph().source(OUTPUT, 0).unwrap_or(INPUT) } else { sel };
        let id = self.fresh_node_id();
        let mixer_id = self.fresh_node_id();
        let count = self.adj().layers.len() + 1;
        let sel_layer = self.adj().layers.iter().find(|l| l.id == sel).cloned();
        // An outside node needs a node with a mask; elsewhere it is a serial node.
        let how = if how == NewNode::Outside && sel_layer.as_ref().is_none_or(|l| l.mask.shapes.is_empty() || is_full(&l.mask)) {
            self.notify(t("Outside nodes need a node with a mask; added a serial node", "外部节点需要选中带蒙版的节点，已添加串行节点"), cx);
            NewNode::Serial
        } else {
            how
        };
        let (name, mask) = match (how, &sel_layer) {
            // A snapshot of the mask, inverted: it keeps working while copied
            // shapes still render with no weight.
            (NewNode::Outside, Some(l)) => (format!("{} {}", t("Outside", "外部"), l.name), Mask { invert: !l.mask.invert, ..l.mask.clone() }),
            _ => {
                let name = match kind {
                    LayerKind::Adjustment => t("Node", "节点"),
                    LayerKind::Heal => t("Heal", "修复"),
                    LayerKind::Clone => t("Clone", "仿制"),
                    LayerKind::Fill => t("Fill", "填色"),
                };
                (format!("{name} {count}"), if kind == LayerKind::Adjustment { full_mask() } else { Mask::default() })
            }
        };
        self.edit_graph(how.label(), false, move |g, layers| {
            layers.push(Layer { id, name, kind, mask, ..Default::default() });
            let at = g.pos(sel);
            let parallel = matches!(how, NewNode::Parallel | NewNode::Layer) && sel != INPUT;
            if !parallel {
                g.shift_after(sel, COL);
                for o in g.outputs(sel) {
                    g.link(id, o.to, o.input);
                }
                g.links.retain(|l| l.from != sel);
                g.nodes.push(GraphNode { id, kind: NodeKind::Layer, pos: [at[0] + COL, at[1]] });
                g.link(sel, id, 0);
                return;
            }
            let kind = if how == NewNode::Parallel { NodeKind::Parallel } else { NodeKind::LayerMixer };
            let src = g.source(sel, 0).unwrap_or(INPUT);
            let outs = g.outputs(sel);
            // Joining a mixer of the same kind adds a branch to it.
            if let [o] = outs[..]
                && g.node(o.to).is_some_and(|n| n.kind == kind)
            {
                let input = g.input_count(o.to) - 1;
                let below = (0..input).filter_map(|i| g.source(o.to, i)).map(|n| g.pos(n)[1]).fold(at[1], f32::max);
                g.nodes.push(GraphNode { id, kind: NodeKind::Layer, pos: [at[0], below + ROW] });
                g.link(src, id, 0);
                g.link(id, o.to, input);
                return;
            }
            g.shift_after(sel, COL);
            g.nodes.push(GraphNode { id: mixer_id, kind, pos: [at[0] + COL, at[1] + ROW / 2.0] });
            for o in outs {
                g.link(mixer_id, o.to, o.input);
            }
            g.links.retain(|l| l.from != sel);
            g.nodes.push(GraphNode { id, kind: NodeKind::Layer, pos: [at[0], at[1] + ROW] });
            g.link(sel, mixer_id, 0);
            g.link(src, id, 0);
            g.link(id, mixer_id, 1);
        }, cx);
        self.select_node(id, cx);
    }

    /// Deletes a node, joining its input to its outputs.
    pub fn delete_node(&mut self, id: u64, cx: &mut Context<Self>) {
        if id == INPUT || id == OUTPUT {
            return;
        }
        let back = self.node_graph().source(id, 0).unwrap_or(INPUT);
        self.edit_graph(t("Delete node", "删除节点"), false, |g, layers| {
            g.bypass(id);
            layers.retain(|l| l.id != id);
        }, cx);
        self.select_node(back, cx);
    }

    /// Connects `from` to input `input` of `to`, unless that makes a loop.
    pub fn connect_nodes(&mut self, from: u64, to: u64, input: u32, cx: &mut Context<Self>) {
        let g = self.node_graph();
        if from == to || g.reaches(to, from) || g.source(to, input) == Some(from) {
            return;
        }
        self.edit_graph(t("Connect nodes", "连接节点"), false, |g, _| g.link(from, to, input), cx);
    }

    pub fn disconnect_input(&mut self, to: u64, input: u32, cx: &mut Context<Self>) {
        self.edit_graph(t("Disconnect node", "断开节点"), false, |g, _| g.unlink(to, input), cx);
    }

    /// Moves a node while it is dragged; `end_gesture` closes the step.
    pub fn move_node(&mut self, id: u64, pos: [f32; 2], cx: &mut Context<Self>) {
        self.edit_graph(t("Move node", "移动节点"), true, |g, _| g.set_pos(id, pos), cx);
    }

    pub fn toggle_node(&mut self, id: u64, cx: &mut Context<Self>) {
        self.edit_layer(t("Enable / disable node", "启用 / 停用节点"), id, false, |l| l.enabled = !l.enabled, cx);
    }

    /// Clears a node's grade, keeping its mask, as Resolve's Reset Node Grade.
    pub fn reset_node_grade(&mut self, id: u64, cx: &mut Context<Self>) {
        self.edit_layer(t("Reset node grade", "重置节点调色"), id, false, |l| l.adjustments = Default::default(), cx);
    }

    /// Lays the graph out in columns by depth, branches stacked in rows.
    pub fn tidy_nodes(&mut self, cx: &mut Context<Self>) {
        self.edit_graph(t("Tidy nodes", "整理节点"), false, |g, _| {
            let ids: Vec<u64> = g.nodes.iter().map(|n| n.id).collect();
            let mut depth: std::collections::HashMap<u64, usize> = ids.iter().map(|id| (*id, 0)).collect();
            // Longest path from the input; bounded, since links never loop.
            for _ in 0..ids.len() {
                for l in &g.links {
                    let d = depth[&l.from] + 1;
                    if depth[&l.to] < d {
                        depth.insert(l.to, d);
                    }
                }
            }
            let mut rows: std::collections::HashMap<usize, f32> = Default::default();
            let mut order = g.nodes.clone();
            order.sort_by(|a, b| a.pos[1].total_cmp(&b.pos[1]));
            for n in order {
                let d = depth[&n.id];
                let row = rows.entry(d).or_insert(0.0);
                g.set_pos(n.id, [d as f32 * COL, *row]);
                *row += ROW;
            }
        }, cx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layers(ids: &[u64]) -> AdjustmentSet {
        AdjustmentSet { layers: ids.iter().map(|id| Layer { id: *id, ..Default::default() }).collect(), ..Default::default() }
    }

    #[test]
    fn layers_without_a_graph_form_a_chain() {
        let g = normalized(&layers(&[1, 2, 3]));
        assert_eq!(g.render_order(), vec![1, 2, 3]);
        assert_eq!(g.source(OUTPUT, 0), Some(3));
    }

    #[test]
    fn parallel_branches_share_an_input_and_render_after_it() {
        let mut a = layers(&[1, 2]);
        let mut g = normalized(&a);
        // 3 beside 2: both fed by 1, joined by mixer 10.
        g.nodes.push(GraphNode { id: 10, kind: NodeKind::Parallel, pos: [300.0, 0.0] });
        g.nodes.push(GraphNode { id: 3, kind: NodeKind::Layer, pos: [200.0, 64.0] });
        g.links.retain(|l| l.from != 2);
        g.link(2, 10, 0);
        g.link(1, 3, 0);
        g.link(3, 10, 1);
        g.link(10, OUTPUT, 0);
        a.layers.push(Layer { id: 3, ..Default::default() });
        store(&mut a, g);
        assert_eq!(a.layers.iter().map(|l| l.id).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(flattened(&a).layers.len(), 3);
    }

    #[test]
    fn disconnected_nodes_do_not_render_and_deleted_ones_are_bypassed() {
        let mut a = layers(&[1, 2, 3]);
        let mut g = normalized(&a);
        g.unlink(2, 0);
        store(&mut a, g);
        assert_eq!(flattened(&a).layers.iter().map(|l| l.id).collect::<Vec<_>>(), Vec::<u64>::new());
        let mut a = layers(&[1, 2, 3]);
        a.nodes = Some(normalized(&a));
        a.layers.retain(|l| l.id != 2);
        let g = normalized(&a);
        assert_eq!(g.render_order(), vec![1, 3]);
    }

    fn graph_with(links: &[(u64, u64, u32)], ids: &[u64]) -> NodeGraph {
        let mut nodes = vec![
            GraphNode { id: INPUT, kind: NodeKind::Input, pos: [0.0; 2] },
            GraphNode { id: OUTPUT, kind: NodeKind::Output, pos: [0.0; 2] },
        ];
        nodes.extend(ids.iter().map(|id| GraphNode { id: *id, kind: NodeKind::Layer, pos: [0.0; 2] }));
        NodeGraph {
            nodes,
            links: links.iter().map(|&(from, to, input)| NodeLink { from, to, input }).collect(),
        }
    }

    #[test]
    fn flattened_follows_the_graph_order_and_prunes_detached_rows() {
        // Stored list says 3, 1, 2; the graph says 1 -> 2 -> output and keeps 3
        // as a detached node.
        let mut a = layers(&[3, 1, 2]);
        a.nodes = Some(graph_with(&[(INPUT, 1, 0), (1, 2, 0), (2, OUTPUT, 0)], &[1, 2, 3]));
        let ids: Vec<u64> = flattened(&a).layers.iter().map(|l| l.id).collect();
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(flattened_checked(&a).unwrap().layers.len(), 2);
    }

    #[test]
    fn a_cyclic_graph_flattens_to_the_background_only() {
        let mut a = layers(&[1, 2]);
        a.nodes = Some(graph_with(
            &[(INPUT, 1, 0), (1, 2, 0), (2, 1, 1), (2, OUTPUT, 0)],
            &[1, 2],
        ));
        assert!(flattened_checked(&a).is_err());
        assert!(flattened(&a).layers.is_empty());
    }

    #[test]
    fn drawing_a_window_replaces_the_whole_frame() {
        let mut m = full_mask();
        assert!(is_full(&m));
        add_shape(&mut m, MaskShape::Radial(RadialGradient { op: MaskOp::Add, center: [0.3, 0.3], radius: [0.1, 0.1], rotation: 0.0, feather: 0.0, inverted: false }));
        assert_eq!(m.shapes.len(), 1);
        assert!(!is_full(&m));
    }

    #[test]
    fn prune_reachable_dag_removes_orphan_nodes_and_dead_branches() {
        // Live path: INPUT -> 1 -> 2 -> OUTPUT
        // Dead branch from INPUT: INPUT -> 3 (never reaches OUTPUT)
        // Orphan node: 4 (not linked at all)
        // Detached mixer: 20 fed by 3 (never reaches OUTPUT)
        let mut g = graph_with(&[(INPUT, 1, 0), (1, 2, 0), (2, OUTPUT, 0), (INPUT, 3, 0)], &[1, 2, 3, 4]);
        g.nodes.push(GraphNode { id: 20, kind: NodeKind::Parallel, pos: [200.0, 128.0] });
        g.link(3, 20, 0);

        let pruned = prune_reachable_dag(&g).unwrap();
        assert_eq!(pruned.active_layers, vec![1, 2]);
        assert_eq!(pruned.pruned_layers, vec![3, 4]);
        assert_eq!(pruned.pruned_nodes, vec![3, 4, 20]);
        assert!(pruned.graph.node(3).is_none());
        assert!(pruned.graph.node(4).is_none());
        assert!(pruned.graph.node(20).is_none());
        assert_eq!(pruned.graph.links.len(), 3);

        // flattened_checked also prunes both layers and the stored graph.
        let mut adj = layers(&[1, 2, 3, 4]);
        adj.nodes = Some(g);
        let flat = flattened_checked(&adj).unwrap();
        assert_eq!(flat.layers.iter().map(|l| l.id).collect::<Vec<_>>(), vec![1, 2]);
        let flat_graph = flat.nodes.unwrap();
        assert!(flat_graph.node(3).is_none() && flat_graph.node(4).is_none() && flat_graph.node(20).is_none());
    }

    #[test]
    fn prune_reachable_dag_detects_cycles_without_infinite_loop() {
        let g = graph_with(
            &[(INPUT, 1, 0), (1, 2, 0), (2, 3, 0), (3, 1, 1), (2, OUTPUT, 0)],
            &[1, 2, 3],
        );
        let err = prune_reachable_dag(&g).unwrap_err();
        match err {
            GraphError::Cycle { nodes } => {
                assert!(!nodes.is_empty(), "cycle error must report involved nodes");
            }
        }
    }

    #[test]
    fn structured_plan_distinguishes_parallel_and_layer_mixers() {
        // Shared trunk: INPUT -> 1; branches: 1 -> 2 and 1 -> 3; joined by mixer 10 -> 4 -> OUTPUT; plus orphan 99.
        let build = |mixer_kind: NodeKind| {
            let mut adj = layers(&[1, 2, 3, 4, 99]);
            let mut g = graph_with(&[], &[1, 2, 3, 4, 99]);
            g.nodes.push(GraphNode { id: 10, kind: mixer_kind, pos: [300.0, 32.0] });
            g.link(INPUT, 1, 0);
            g.link(1, 2, 0);
            g.link(1, 3, 0);
            g.link(2, 10, 0);
            g.link(3, 10, 1);
            g.link(10, 4, 0);
            g.link(4, OUTPUT, 0);
            adj.nodes = Some(g);
            adj
        };

        let parallel_plan = build_structured_plan(&build(NodeKind::Parallel)).unwrap();
        assert_eq!(parallel_plan.flat_layers, vec![1, 2, 3, 4]);
        assert_eq!(parallel_plan.pruned_layers, vec![99]);
        assert_eq!(
            parallel_plan.steps,
            vec![
                EvalStep::Layer(1),
                EvalStep::Mixer {
                    id: 10,
                    semantics: MixerSemantics::ParallelBlend,
                    branches: vec![
                        MixerBranch { input: 0, steps: vec![EvalStep::Layer(2)] },
                        MixerBranch { input: 1, steps: vec![EvalStep::Layer(3)] },
                    ],
                },
                EvalStep::Layer(4),
            ]
        );
        assert_eq!(MixerSemantics::ParallelBlend.branch_weights(2), vec![0.5, 0.5]);
        assert!(!MixerSemantics::ParallelBlend.overrides_previous());

        let layer_plan = build_structured_plan(&build(NodeKind::LayerMixer)).unwrap();
        assert_eq!(layer_plan.flat_layers, vec![1, 2, 3, 4]);
        assert_eq!(layer_plan.pruned_layers, vec![99]);
        assert_eq!(
            layer_plan.steps,
            vec![
                EvalStep::Layer(1),
                EvalStep::Mixer {
                    id: 10,
                    semantics: MixerSemantics::LayerStackTopWins,
                    branches: vec![
                        MixerBranch { input: 0, steps: vec![EvalStep::Layer(2)] },
                        MixerBranch { input: 1, steps: vec![EvalStep::Layer(3)] },
                    ],
                },
                EvalStep::Layer(4),
            ]
        );
        assert_eq!(MixerSemantics::LayerStackTopWins.branch_weights(2), vec![1.0, 1.0]);
        assert!(MixerSemantics::LayerStackTopWins.overrides_previous());
    }
}
