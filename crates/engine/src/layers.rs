//! Local adjustment layers and the masks that place them.

use serde::{Deserialize, Serialize};

use crate::adjust::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerKind {
    #[default]
    Adjustment,
    Heal,
    Clone,
    /// A layer that fills the mask with a flat color.
    Fill,
}

/// Which adjustments a local layer may carry. Capture One limits layers to the
/// tools that make sense locally; the same list is used here.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LocalAdjustments {
    pub white_balance: Option<WhiteBalance>,
    pub exposure: Exposure,
    pub hdr: HighDynamicRange,
    pub clarity: Clarity,
    pub dehaze: Dehaze,
    /// Present only when the layer edits tone; neutral or absent is `None`.
    pub curves: Option<Curves>,
    pub levels: Option<Levels>,
    /// Present only when the layer converts to black and white.
    pub black_white: Option<BlackWhite>,
    pub color_balance: ColorBalance,
    pub color_editor: ColorEditor,
    pub sharpening: Option<Sharpening>,
    pub noise_reduction: Option<NoiseReduction>,
    pub moire: Moire,
    pub fill_color: Option<[f32; 3]>,
    /// `<LA><E K="AIColorGrade">`: this layer is a Match Look layer. The style
    /// document carries the reference thumbnail in `<AICGR>`, not here.
    pub ai_color_grade: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Layer {
    pub id: u64,
    pub name: String,
    pub kind: LayerKind,
    pub enabled: bool,
    /// 0 ..= 100.
    pub opacity: f32,
    pub mask: Mask,
    pub adjustments: LocalAdjustments,
    /// `ZVARIANTLAYER.ZMASKUUID`, the `.cos` layer's `MaskUUID`: the key that
    /// links this layer to its `.comask` mask record.
    pub mask_uuid: Option<String>,
    /// `ZSTYLEUUID`/`ZSTYLESOURCE`/`ZSTYLEMODIFIED` when the layer came from a style.
    pub style: Option<LayerStyle>,
    /// `LAYERSELECTPOINT`, raw drawing pixels (`x;y`).
    pub select_point: Option<[f32; 2]>,
    /// `.comask` Cover arrows of a heal/clone layer, in oriented coordinates.
    pub cover: Vec<CoverPatch>,
}

impl Default for Layer {
    fn default() -> Self {
        Self {
            id: 0,
            name: "Layer".into(),
            kind: LayerKind::Adjustment,
            enabled: true,
            opacity: 100.0,
            mask: Mask::default(),
            adjustments: LocalAdjustments::default(),
            mask_uuid: None,
            style: None,
            select_point: None,
            cover: Vec::new(),
        }
    }
}

/// One `.comask` `Cover` arrow, in the oriented, uncropped image.
///
/// C1 keeps only the two points in the arrow; the stroke shape lives in the
/// layer's mask plane, so `half` is the mask plane's bounding box.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CoverPatch {
    pub kind: CoverKind,
    /// Source point, normalized.
    pub src: [f32; 2],
    /// Destination point, normalized.
    pub dst: [f32; 2],
    /// Half extents of the repaired rectangle, fractions of width/height.
    pub half: [f32; 2],
}

impl Default for CoverPatch {
    fn default() -> Self {
        Self { kind: CoverKind::Clone, src: [0.0; 2], dst: [0.0; 2], half: [0.0; 2] }
    }
}

/// `IC_MaskCoverArrowType`: 1 clone, 2 heal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoverKind {
    #[default]
    Clone,
    Heal,
}

/// The style a layer was applied from. C1 keeps the same UUID on every layer of
/// a layered style, and `source` is the style folder/tool (`Styles`, `Exposure`, ...).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerStyle {
    pub uuid: String,
    pub source: Option<String>,
    pub modified: bool,
}

/// A mask is a stack of shapes combined in order; each shape adds or subtracts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Mask {
    pub shapes: Vec<MaskShape>,
    pub invert: bool,
    /// Feather applied to the combined mask, 0 ..= 1000 px.
    pub feather: f32,
    /// Refine-edge radius, 0 ..= 100.
    pub refine: f32,
    /// Restricts the mask to a luminosity window when present.
    pub luma_range: Option<LumaRange>,
    /// Restricts the mask to a color window when present.
    pub color_range: Option<ColorRange>,
    /// 0 ..= 100, the mask's overall density.
    pub density: f32,
}

impl Default for Mask {
    fn default() -> Self {
        Self {
            shapes: Vec::new(),
            invert: false,
            feather: 0.0,
            refine: 0.0,
            luma_range: None,
            color_range: None,
            density: 100.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MaskOp {
    #[default]
    Add,
    Subtract,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MaskShape {
    Brush(BrushStroke),
    Linear(LinearGradient),
    Radial(RadialGradient),
    /// A coverage bitmap decoded from a Capture One `.comask` mask.
    Raster(RasterMask),
    /// A mask the engine computes, such as the subject or the sky.
    Ai(AiMask),
    /// A mask copied from another layer by id.
    Copy { from_layer: u64, op: MaskOp },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrushStroke {
    pub op: MaskOp,
    /// Normalized image coordinates with pressure in `0.0 ..= 1.0`.
    pub points: Vec<[f32; 3]>,
    /// Size as a fraction of the image's long side.
    pub size: f32,
    /// 0 ..= 100.
    pub hardness: f32,
    pub opacity: f32,
    pub flow: f32,
    pub auto_mask: bool,
    /// C1's `ZBRUSHAIRBRUSH`. Parsed and carried; the renderer ignores it.
    #[serde(default)]
    pub hair_brush: bool,
    /// C1's `ZBRUSHUSEPENPRESSURE` (eraser strokes use `ZERASERUSEPENPRESSURE`).
    #[serde(default)]
    pub use_pen_pressure: bool,
}

/// A pixel mask over a normalized rectangle: `width * height` 8-bit coverage
/// bytes, row-major, `rect = [x0, y0, x1, y1]` in oriented image coordinates.
///
/// This is the `.comask` `Map` bitmap after decoding, cropped to its non-zero
/// bounding box, so an imported local layer carries its own geometry instead
/// of an engine-side cache handle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RasterMask {
    pub op: MaskOp,
    pub rect: [f32; 4],
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl RasterMask {
    /// A one-pixel zero mask: keeps a layer attached without changing pixels.
    pub fn empty(op: MaskOp) -> Self {
        Self { op, rect: [0.0, 0.0, 1.0, 1.0], width: 1, height: 1, data: vec![0] }
    }

    /// Bilinear coverage at a normalized image point; `0.0` outside `rect`.
    pub fn sample(&self, p: [f32; 2]) -> f32 {
        let [x0, y0, x1, y1] = self.rect;
        let (dw, dh) = (x1 - x0, y1 - y0);
        if dw <= 0.0 || dh <= 0.0 || self.width == 0 || self.height == 0 {
            return 0.0;
        }
        let (u, v) = ((p[0] - x0) / dw, (p[1] - y0) / dh);
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return 0.0;
        }
        let x = (u * self.width as f32 - 0.5).clamp(0.0, (self.width as f32 - 1.001).max(0.0));
        let y = (v * self.height as f32 - 0.5).clamp(0.0, (self.height as f32 - 1.001).max(0.0));
        let (xi, yi) = (x as usize, y as usize);
        let (xn, yn) = ((xi + 1).min(self.width - 1), (yi + 1).min(self.height - 1));
        let at = |xx: usize, yy: usize| self.data[yy * self.width + xx] as f32 / 255.0;
        let top = at(xi, yi) + (at(xn, yi) - at(xi, yi)) * (x - xi as f32);
        let bottom = at(xi, yn) + (at(xn, yn) - at(xi, yn)) * (x - xi as f32);
        (top + (bottom - top) * (y - yi as f32)).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearGradient {
    pub op: MaskOp,
    /// Full-effect and zero-effect lines through these normalized points.
    pub start: [f32; 2],
    pub end: [f32; 2],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RadialGradient {
    pub op: MaskOp,
    pub center: [f32; 2],
    pub radius: [f32; 2],
    pub rotation: f32,
    /// 0 ..= 100: the share of the radius that fades.
    pub feather: f32,
    pub inverted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AiMaskKind {
    Subject,
    Background,
    Sky,
    Person,
    Skin,
    Hair,
    Clothing,
    Eyes,
    Lips,
}

impl AiMaskKind {
    pub const ALL: [Self; 9] = [
        Self::Subject,
        Self::Background,
        Self::Sky,
        Self::Person,
        Self::Skin,
        Self::Hair,
        Self::Clothing,
        Self::Eyes,
        Self::Lips,
    ];
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AiMask {
    pub op: MaskOp,
    pub kind: AiMaskKind,
    /// Engine-owned handle to its cached segmentation, if computed.
    pub cache_key: Option<String>,
}

/// A luminosity window: full effect between the inner points, falling off to the outer.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LumaRange {
    pub low_outer: f32,
    pub low_inner: f32,
    pub high_inner: f32,
    pub high_outer: f32,
    pub radius: f32,
    pub sensitivity: f32,
    pub invert: bool,
}

impl Default for LumaRange {
    fn default() -> Self {
        Self {
            low_outer: 0.0,
            low_inner: 0.1,
            high_inner: 0.9,
            high_outer: 1.0,
            radius: 4.0,
            sensitivity: 30.0,
            invert: false,
        }
    }
}

// ───────────────────────────── Node graph ─────────────────────────────
// Data only: see `docs/frontend-contract/node-graph.md`. Engines render the
// layers in `render_order()`; mixers are not blended yet.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeKind {
    /// The image after the background (global) adjustments.
    Input,
    Output,
    /// A corrector node: the `Layer` with the same id.
    #[default]
    Layer,
    /// Blends branches that grade the same image.
    Parallel,
    /// Stacks its inputs; a higher input number is on top.
    LayerMixer,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: u64,
    pub kind: NodeKind,
    /// Position in the node editor, in editor units.
    pub pos: [f32; 2],
}

/// `from`'s output feeds input number `input` of `to`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeLink {
    pub from: u64,
    pub to: u64,
    pub input: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeGraph {
    pub nodes: Vec<GraphNode>,
    pub links: Vec<NodeLink>,
}

/// Why a node graph cannot be evaluated at all.
///
/// Reachability itself is never an error: nodes the output does not depend on
/// are pruned and reported. A cycle is only an error when the output depends on
/// it; a detached loop cannot change the rendered frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphError {
    /// The output depends on a loop, so there is no evaluation order. `nodes`
    /// names the members of the loop, not everything that waits on it.
    Cycle { nodes: Vec<u64> },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cycle { nodes } => write!(f, "node graph has a cycle through {nodes:?}"),
        }
    }
}

impl std::error::Error for GraphError {}

/// A structural detail that still renders. Warnings are reported instead of
/// guessed away, so the interface can ask for the missing piece.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphWarning {
    /// A mixer input is absent even though a higher-numbered input is wired.
    MissingInput { node: u64, input: u32 },
    /// A mixer on the render path has no connected inputs.
    EmptyMixer { node: u64 },
    /// A layer node on the render path has no `Layer` row to grade with.
    LayerWithoutRow { id: u64 },
    /// A `Layer` row is not a node in the graph, so it never renders.
    RowWithoutNode { id: u64 },
    /// The graph has no input node, so there is no base image marker to test
    /// reachability against; every layer the output reaches is treated as
    /// connected.
    NoInputNode,
    /// A link starts or ends at a node the graph does not contain. It is
    /// dropped, not fatal: a stale edge in a branch the output never uses must
    /// not blank the frame.
    DanglingLink { link: NodeLink },
}

/// The evaluation plan for one graph: what renders, in what order, and what it
/// deliberately leaves out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphPlan {
    /// Nodes on a path from [`NodeGraph::INPUT`] to [`NodeGraph::OUTPUT`], in
    /// evaluation order (every node after what feeds it).
    pub order: Vec<u64>,
    /// Layer nodes of `order`, in render order.
    pub layers: Vec<u64>,
    /// Layer nodes the output does not depend on. They must not render.
    pub pruned: Vec<u64>,
    /// Structural details that do not refuse the graph.
    pub warnings: Vec<GraphWarning>,
}

impl NodeGraph {
    pub const INPUT: u64 = u64::MAX - 1;
    pub const OUTPUT: u64 = u64::MAX;

    pub fn node(&self, id: u64) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// What feeds input `input` of `to`.
    pub fn source(&self, to: u64, input: u32) -> Option<u64> {
        self.links.iter().find(|l| l.to == to && l.input == input).map(|l| l.from)
    }

    /// Whether a path of links leads from `from` to `to`.
    pub fn reaches(&self, from: u64, to: u64) -> bool {
        let mut seen = vec![from];
        let mut stack = vec![from];
        while let Some(n) = stack.pop() {
            for l in self.links.iter().filter(|l| l.from == n) {
                if l.to == to {
                    return true;
                }
                if !seen.contains(&l.to) {
                    seen.push(l.to);
                    stack.push(l.to);
                }
            }
        }
        false
    }

    /// The evaluation plan for this graph: the nodes and layers that actually
    /// feed the output, in evaluation order.
    ///
    /// Reachability is not an error. Nodes the output does not depend on are
    /// reported in [`GraphPlan::pruned`] and must not render, but the rest of
    /// the graph still evaluates. Only a graph with no evaluation order
    /// ([`GraphError::Cycle`]) is refused; a dangling link is dropped with a
    /// warning so a stale edge in an unused branch cannot blank the frame.
    ///
    /// A cycle is only reported when it is on the output's side: a detached
    /// loop cannot influence the rendered frame, so it stays a pruning detail.
    pub fn plan(&self) -> Result<GraphPlan, GraphError> {
        let ids: Vec<u64> = self.nodes.iter().map(|node| node.id).collect();
        let mut warnings: Vec<GraphWarning> = self
            .links
            .iter()
            .filter(|link| !ids.contains(&link.from) || !ids.contains(&link.to))
            .map(|link| GraphWarning::DanglingLink { link: *link })
            .collect();
        let mut graph = self.clone();
        graph.links.retain(|link| ids.contains(&link.from) && ids.contains(&link.to));

        // Depth-first post-order from the output, inputs in input order. This
        // is the same order the old `render_order` used, so a graph that
        // rendered before keeps its stacking order (Kahn sorted by id would
        // have reordered mixer branches).
        let mut order = Vec::new();
        let mut seen = Vec::new();
        let mut stack = Vec::new();
        graph.visit(Self::OUTPUT, &mut seen, &mut stack, &mut order)?;

        // Only nodes that also see the base image can grade it. A document
        // without an input node has no marker to test against, so every
        // reached layer counts as connected (the renderer treats it as a
        // chain, as before).
        let live = match graph.node(Self::INPUT) {
            Some(_) => {
                let mut forward = Vec::new();
                let mut pending = vec![Self::INPUT];
                while let Some(id) = pending.pop() {
                    if forward.contains(&id) {
                        continue;
                    }
                    forward.push(id);
                    pending.extend(graph.links.iter().filter(|l| l.from == id).map(|l| l.to));
                }
                order.iter().copied().filter(|id| forward.contains(id)).collect::<Vec<_>>()
            }
            None => order.clone(),
        };

        let layers: Vec<u64> = live.iter().copied().filter(|id| graph.is_layer(*id)).collect();
        let mut pruned: Vec<u64> = graph
            .nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Layer)
            .map(|node| node.id)
            .filter(|id| !layers.contains(id))
            .collect();
        pruned.sort_unstable();
        pruned.dedup();

        warnings.extend(graph.mixer_warnings(&order));
        if graph.node(Self::INPUT).is_none() {
            warnings.push(GraphWarning::NoInputNode);
        }
        Ok(GraphPlan { order, layers, pruned, warnings })
    }

    /// The graph repaired to match `rows`: a missing input or output is added,
    /// nodes whose layer was deleted are bypassed, and rows the graph does not
    /// know yet join in series before the output. Both the interface and the
    /// render path call this, so a stored document cannot preview and export
    /// with different layer sets.
    pub fn repaired(&self, rows: &[Layer]) -> NodeGraph {
        let mut graph = self.clone();
        let fresh = graph.node(Self::INPUT).is_none() && graph.node(Self::OUTPUT).is_none();
        if graph.node(Self::INPUT).is_none() {
            graph
                .nodes
                .insert(0, GraphNode { id: Self::INPUT, kind: NodeKind::Input, pos: [0.0; 2] });
        }
        if graph.node(Self::OUTPUT).is_none() {
            let x = graph.nodes.iter().map(|node| node.pos[0]).fold(0.0, f32::max) + Self::COL;
            graph
                .nodes
                .push(GraphNode { id: Self::OUTPUT, kind: NodeKind::Output, pos: [x, 0.0] });
        }
        if fresh {
            graph.link(Self::INPUT, Self::OUTPUT, 0);
        }
        let gone: Vec<u64> = graph
            .nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Layer && !rows.iter().any(|row| row.id == node.id))
            .map(|node| node.id)
            .collect();
        for id in gone {
            graph.bypass(id);
        }
        let ids: Vec<u64> = graph.nodes.iter().map(|node| node.id).collect();
        graph.links.retain(|link| ids.contains(&link.from) && ids.contains(&link.to));
        for row in rows {
            if graph.node(row.id).is_some() {
                continue;
            }
            let tail = graph.source(Self::OUTPUT, 0).unwrap_or(Self::INPUT);
            let at = graph.node(tail).map(|node| node.pos).unwrap_or_default();
            graph.nodes.push(GraphNode {
                id: row.id,
                kind: NodeKind::Layer,
                pos: [at[0] + Self::COL, at[1]],
            });
            graph.link(tail, row.id, 0);
            graph.link(row.id, Self::OUTPUT, 0);
            if let Some(output) = graph.nodes.iter_mut().find(|node| node.id == Self::OUTPUT) {
                output.pos = [output.pos[0].max(at[0] + 2.0 * Self::COL), output.pos[1]];
            }
        }
        graph
    }

    /// Editor column width, in editor units. Layout only: it never reaches a
    /// pixel. Shared with the interface through [`repaired`](Self::repaired).
    const COL: f32 = 104.0;

    /// A link replaces whatever occupied the same input.
    fn link(&mut self, from: u64, to: u64, input: u32) {
        self.links.retain(|link| !(link.to == to && link.input == input));
        self.links.push(NodeLink { from, to, input });
    }

    /// Removes `id`, joining what fed its first input to everything it fed.
    fn bypass(&mut self, id: u64) {
        let source = self.source(id, 0);
        let outputs: Vec<NodeLink> =
            self.links.iter().filter(|link| link.from == id).copied().collect();
        self.links.retain(|link| link.from != id && link.to != id);
        if let Some(source) = source {
            for output in outputs {
                self.link(source, output.to, output.input);
            }
        }
        self.nodes.retain(|node| node.id != id);
    }

    /// Post-order visit used by [`plan`](Self::plan). `stack` holds the path
    /// being walked, so a repeat names exactly the loop's members.
    fn visit(
        &self,
        id: u64,
        seen: &mut Vec<u64>,
        stack: &mut Vec<u64>,
        out: &mut Vec<u64>,
    ) -> Result<(), GraphError> {
        if seen.contains(&id) {
            return Ok(());
        }
        if let Some(at) = stack.iter().position(|node| *node == id) {
            return Err(GraphError::Cycle { nodes: stack[at..].to_vec() });
        }
        stack.push(id);
        let mut inputs: Vec<&NodeLink> = self.links.iter().filter(|link| link.to == id).collect();
        inputs.sort_by_key(|link| link.input);
        for link in inputs {
            self.visit(link.from, seen, stack, out)?;
        }
        stack.pop();
        seen.push(id);
        out.push(id);
        Ok(())
    }

    /// [`plan`](Self::plan), plus the checks that need the layer rows: a layer
    /// node without a `Layer` cannot grade, and a `Layer` that is not a node
    /// never renders. Both are warnings, not errors, because a stored document
    /// may carry either while the user repairs it.
    pub fn inspect(&self, rows: &[Layer]) -> Result<GraphPlan, GraphError> {
        let mut plan = self.plan()?;
        for id in &plan.layers {
            if !rows.iter().any(|row| row.id == *id) {
                plan.warnings.push(GraphWarning::LayerWithoutRow { id: *id });
            }
        }
        for row in rows {
            if self.node(row.id).is_none() {
                plan.warnings.push(GraphWarning::RowWithoutNode { id: row.id });
            }
        }
        Ok(plan)
    }

    /// Mixers are not clocked yet, so a gap is reported and the renderer keeps
    /// the serial order; a connected input is never guessed.
    fn mixer_warnings(&self, live: &[u64]) -> Vec<GraphWarning> {
        let mut out = Vec::new();
        for id in live {
            let Some(node) = self.node(*id) else { continue };
            if !matches!(node.kind, NodeKind::Parallel | NodeKind::LayerMixer) {
                continue;
            }
            let mut inputs: Vec<u32> =
                self.links.iter().filter(|l| l.to == *id).map(|l| l.input).collect();
            inputs.sort_unstable();
            let Some(last) = inputs.last().copied() else {
                out.push(GraphWarning::EmptyMixer { node: *id });
                continue;
            };
            for input in 0..=last {
                if !inputs.contains(&input) {
                    out.push(GraphWarning::MissingInput { node: *id, input });
                }
            }
        }
        out
    }

    fn is_layer(&self, id: u64) -> bool {
        self.node(id).is_some_and(|node| node.kind == NodeKind::Layer)
    }

    /// The layer nodes on a path from the input to the output, each after
    /// everything that feeds it. This is the UI's ordering helper: it keeps
    /// working on a graph [`plan`](Self::plan) refuses, so numbering and
    /// tidy-up never go blank mid-edit. Render and export paths must use
    /// [`plan`](Self::plan) or [`layered`].
    pub fn render_order(&self) -> Vec<u64> {
        match self.plan() {
            Ok(plan) => plan.layers,
            Err(_) => self.dfs_render_order(),
        }
    }

    /// The pre-`plan` walk: depth-first from the output, a cycle cut where it
    /// closes. Only the fallback for [`render_order`](Self::render_order).
    fn dfs_render_order(&self) -> Vec<u64> {
        fn visit(g: &NodeGraph, id: u64, seen: &mut Vec<u64>, out: &mut Vec<u64>) {
            if seen.contains(&id) {
                return;
            }
            seen.push(id);
            let mut inputs: Vec<&NodeLink> = g.links.iter().filter(|l| l.to == id).collect();
            inputs.sort_by_key(|l| l.input);
            for l in inputs {
                visit(g, l.from, seen, out);
            }
            out.push(id);
        }
        let mut order = Vec::new();
        visit(self, Self::OUTPUT, &mut Vec::new(), &mut order);
        order
            .into_iter()
            .filter(|id| self.is_layer(*id) && self.reaches(Self::INPUT, *id))
            .collect()
    }
}

/// The layers a render uses, in render order, or the reason the graph is
/// refused.
///
/// Without a stored graph the layer list is the graph: every row renders in
/// list order, as it always has. With a graph, the graph is first repaired
/// against the rows — a layer created without a node joins the chain instead
/// of disappearing — and then only the rows on the input-to-output path
/// render, in the plan's order. A stored adjustment that reaches the engine
/// directly (export, batch, thumbnails, the picker) therefore renders exactly
/// what the live preview renders.
pub fn layered(adj: &AdjustmentSet) -> Result<Vec<&Layer>, GraphError> {
    let Some(graph) = adj.nodes.as_ref() else {
        return Ok(adj.layers.iter().collect());
    };
    let plan = graph.repaired(&adj.layers).plan()?;
    Ok(plan
        .layers
        .iter()
        .filter_map(|id| adj.layers.iter().find(|layer| layer.id == *id))
        .collect())
}

/// [`layered`] for the render path: a refused graph is logged once and renders
/// the background only. Refusing to invent an order is deliberate — the UI
/// still gets the error through [`layered`] / [`NodeGraph::plan`].
pub fn render_layers(adj: &AdjustmentSet) -> Vec<&Layer> {
    match layered(adj) {
        Ok(layers) => layers,
        Err(error) => {
            static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            if LOGGED.set(()).is_ok() {
                log::error!("node graph refused: {error}; rendering the background only");
            }
            Vec::new()
        }
    }
}

#[cfg(test)]
mod node_tests {
    use super::*;

    fn graph(links: &[(u64, u64, u32)], layers: &[u64]) -> NodeGraph {
        let mut nodes = vec![
            GraphNode { id: NodeGraph::INPUT, kind: NodeKind::Input, pos: [0.0; 2] },
            GraphNode { id: NodeGraph::OUTPUT, kind: NodeKind::Output, pos: [0.0; 2] },
            GraphNode { id: 10, kind: NodeKind::Parallel, pos: [0.0; 2] },
        ];
        nodes.extend(layers.iter().map(|id| GraphNode { id: *id, kind: NodeKind::Layer, pos: [0.0; 2] }));
        NodeGraph { nodes, links: links.iter().map(|&(from, to, input)| NodeLink { from, to, input }).collect() }
    }

    const I: u64 = NodeGraph::INPUT;
    const O: u64 = NodeGraph::OUTPUT;

    #[test]
    fn serial_and_parallel_order() {
        assert_eq!(graph(&[(I, 1, 0), (1, 2, 0), (2, O, 0)], &[1, 2]).render_order(), vec![1, 2]);
        let g = graph(&[(I, 1, 0), (1, 2, 0), (1, 3, 0), (2, 10, 0), (3, 10, 1), (10, O, 0)], &[1, 2, 3]);
        assert_eq!(g.render_order(), vec![1, 2, 3]);
    }

    #[test]
    fn disconnected_and_cyclic_nodes() {
        // 2 has no input: nothing reaches the output from the input.
        assert!(graph(&[(I, 1, 0), (2, 3, 0), (3, O, 0)], &[1, 2, 3]).render_order().is_empty());
        let g = graph(&[(I, 1, 0), (1, 2, 0), (2, 1, 1), (2, O, 0)], &[1, 2]);
        assert_eq!(g.render_order(), vec![1, 2]);
    }

    fn row(id: u64) -> Layer {
        Layer { id, ..Default::default() }
    }

    #[test]
    fn plan_keeps_the_input_to_output_path_and_reports_the_rest() {
        // 1 -> 2 -> output; 3 hangs off 1 but the output does not use it.
        let g = graph(&[(I, 1, 0), (1, 2, 0), (1, 3, 0), (2, O, 0)], &[1, 2, 3]);
        let plan = g.plan().unwrap();
        assert_eq!(plan.layers, vec![1, 2]);
        assert_eq!(plan.pruned, vec![3]);
        assert_eq!(plan.order, vec![I, 1, 2, O]);
        assert!(plan.warnings.is_empty());
    }

    #[test]
    fn plan_orders_parallel_branches_before_their_mixer() {
        let g = graph(
            &[(I, 1, 0), (1, 2, 0), (1, 3, 0), (2, 10, 0), (3, 10, 1), (10, O, 0)],
            &[1, 2, 3],
        );
        let plan = g.plan().unwrap();
        assert_eq!(plan.layers, vec![1, 2, 3]);
        let at = |id: u64| plan.order.iter().position(|node| *node == id).unwrap();
        assert!(at(1) < at(2) && at(2) < at(10));
        assert!(at(3) < at(10) && at(10) < at(O));
        assert!(plan.pruned.is_empty());
    }

    #[test]
    fn plan_refuses_a_cycle_the_output_depends_on() {
        let g = graph(&[(I, 1, 0), (1, 2, 0), (2, 1, 1), (2, O, 0)], &[1, 2]);
        match g.plan() {
            Err(GraphError::Cycle { mut nodes }) => {
                nodes.sort_unstable();
                assert_eq!(nodes, vec![1, 2]);
            }
            other => panic!("expected a cycle, got {other:?}"),
        }
        // The interface helper still answers while the graph is being repaired.
        assert_eq!(g.render_order(), vec![1, 2]);
    }

    #[test]
    fn plan_ignores_a_cycle_the_output_does_not_use() {
        let g = graph(&[(I, 1, 0), (1, O, 0), (2, 3, 0), (3, 2, 0)], &[1, 2, 3]);
        let plan = g.plan().unwrap();
        assert_eq!(plan.layers, vec![1]);
        assert_eq!(plan.pruned, vec![2, 3]);
    }

    #[test]
    fn a_dangling_link_is_a_warning_and_does_not_blank_the_frame() {
        // The stale edge is dropped; the branch the output uses still renders.
        let g = graph(&[(I, 1, 0), (1, O, 0), (1, 77, 0)], &[1]);
        let plan = g.plan().unwrap();
        assert_eq!(plan.layers, vec![1]);
        assert!(plan.warnings.contains(&GraphWarning::DanglingLink {
            link: NodeLink { from: 1, to: 77, input: 0 },
        }));
    }

    #[test]
    fn plan_orders_mixer_inputs_by_input_index_not_id() {
        // Layer 5 feeds input 0, layer 2 feeds input 1 (the top of the stack).
        // Sorting ready nodes by id would draw 2 under 5 and change the frame.
        let g = graph(&[(I, 5, 0), (5, 10, 0), (I, 2, 0), (2, 10, 1), (10, O, 0)], &[2, 5]);
        let plan = g.plan().unwrap();
        assert_eq!(plan.layers, vec![5, 2]);
    }

    #[test]
    fn repaired_appends_rows_that_have_no_node() {
        let g = graph(&[(I, 1, 0), (1, O, 0)], &[1]);
        let repaired = g.repaired(&[row(1), row(2)]);
        assert!(repaired.node(2).is_some());
        let plan = repaired.plan().unwrap();
        assert_eq!(plan.layers, vec![1, 2]);
    }

    #[test]
    fn a_mixer_gap_is_a_warning_not_a_refusal() {
        // Input 1 is wired while input 0 is not: the renderer keeps the serial
        // order, and the warning names the missing piece.
        let g = graph(&[(I, 1, 0), (1, 2, 0), (2, 10, 1), (10, O, 0)], &[1, 2]);
        let plan = g.plan().unwrap();
        assert_eq!(plan.layers, vec![1, 2]);
        assert!(plan.warnings.contains(&GraphWarning::MissingInput { node: 10, input: 0 }));
    }

    #[test]
    fn an_empty_mixer_on_the_output_path_is_a_warning() {
        let g = graph(&[(I, 1, 0), (1, O, 0), (10, O, 1)], &[1]);
        let plan = g.plan().unwrap();
        assert!(plan.warnings.contains(&GraphWarning::EmptyMixer { node: 10 }));
    }

    #[test]
    fn a_graph_without_an_input_node_warns_and_keeps_the_reached_layers() {
        let g = NodeGraph {
            nodes: vec![
                GraphNode { id: 1, kind: NodeKind::Layer, pos: [0.0; 2] },
                GraphNode { id: O, kind: NodeKind::Output, pos: [0.0; 2] },
            ],
            links: vec![NodeLink { from: 1, to: O, input: 0 }],
        };
        let plan = g.plan().unwrap();
        assert_eq!(plan.layers, vec![1]);
        assert!(plan.warnings.contains(&GraphWarning::NoInputNode));
    }

    #[test]
    fn inspect_reports_rows_and_nodes_that_do_not_match() {
        let g = graph(&[(I, 1, 0), (1, 2, 0), (2, O, 0)], &[1, 2]);
        let rows = vec![row(1), row(3)];
        let plan = g.inspect(&rows).unwrap();
        assert!(plan.warnings.contains(&GraphWarning::LayerWithoutRow { id: 2 }));
        assert!(plan.warnings.contains(&GraphWarning::RowWithoutNode { id: 3 }));
    }

    #[test]
    fn layered_takes_the_graph_order_and_prunes_detached_rows() {
        let mut adj = AdjustmentSet { layers: vec![row(2), row(1), row(3)], ..Default::default() };
        adj.nodes = Some(graph(&[(I, 1, 0), (1, 2, 0), (2, O, 0), (1, 3, 0)], &[1, 2, 3]));
        let ids: Vec<u64> = layered(&adj).unwrap().iter().map(|layer| layer.id).collect();
        assert_eq!(ids, vec![1, 2]);
        // Without a graph the stored list is the graph, in its stored order.
        adj.nodes = None;
        let ids: Vec<u64> = layered(&adj).unwrap().iter().map(|layer| layer.id).collect();
        assert_eq!(ids, vec![2, 1, 3]);
    }

    #[test]
    fn render_layers_refuses_a_cyclic_graph_instead_of_guessing() {
        let mut adj = AdjustmentSet { layers: vec![row(1), row(2)], ..Default::default() };
        adj.nodes = Some(graph(&[(I, 1, 0), (1, 2, 0), (2, 1, 1), (2, O, 0)], &[1, 2]));
        assert!(layered(&adj).is_err());
        assert!(render_layers(&adj).is_empty());
    }

    #[test]
    fn missing_graph_field_still_parses() {
        let a: crate::AdjustmentSet = serde_json::from_str("{}").unwrap();
        assert!(a.nodes.is_none());
    }

    /// A saved `.portrait` document is a serialized `AdjustmentSet`. A local layer
    /// written before it carried curves or levels must load with both absent.
    #[test]
    fn old_portrait_document_loads_local_layer_without_tone_edits() {
        let json = r#"{"layers":[{"id":1,"name":"Skin","kind":"Adjustment","enabled":true,"opacity":100.0,"mask":{"shapes":[],"invert":false,"feather":0.0,"density":100.0},"adjustments":{"exposure":{"exposure":1.0,"contrast":0.0,"brightness":0.0,"saturation":0.0}}}]}"#;
        let set: crate::AdjustmentSet = serde_json::from_str(json).unwrap();
        let a = &set.layers[0].adjustments;
        assert!(a.curves.is_none());
        assert!(a.levels.is_none());
        assert!(a.black_white.is_none());
        assert_eq!(a.exposure.exposure, 1.0);
    }

    #[test]
    fn raster_mask_survives_a_json_round_trip() {
        let shape = MaskShape::Raster(RasterMask {
            op: MaskOp::Subtract,
            rect: [0.1, 0.2, 0.9, 0.8],
            width: 3,
            height: 2,
            data: vec![0, 64, 255, 128, 32, 7],
        });
        let json = serde_json::to_string(&shape).unwrap();
        let back: MaskShape = serde_json::from_str(&json).unwrap();
        assert_eq!(shape, back);
    }
}
