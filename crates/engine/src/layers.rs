//! Local adjustment layers and the masks that place them.

use std::collections::BTreeMap;

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
// layers in `render_order()`; mixers render through `mix_plan()`.

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
    /// A `Layer` node has no row, so [`repaired`](NodeGraph::repaired) bypasses
    /// it and nothing grades with it.
    LayerWithoutRow { id: u64 },
    /// A `Layer` row has no node in the stored graph; repair appends it in
    /// series, so it renders (and exports) at the tail. This is a hint to add
    /// the node, not a "never renders" report.
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

    /// [`plan`](Self::plan) for the repaired graph, plus the row/node checks
    /// that repair papers over. Both are warnings, not errors, and neither
    /// changes what renders: a row the graph does not know is appended in
    /// series, and a node with no row is bypassed.
    pub fn inspect(&self, rows: &[Layer]) -> Result<GraphPlan, GraphError> {
        let mut plan = self.repaired(rows).plan()?;
        for node in &self.nodes {
            if node.kind == NodeKind::Layer && !rows.iter().any(|row| row.id == node.id) {
                plan.warnings.push(GraphWarning::LayerWithoutRow { id: node.id });
            }
        }
        for row in rows {
            if self.node(row.id).is_none() {
                plan.warnings.push(GraphWarning::RowWithoutNode { id: row.id });
            }
        }
        Ok(plan)
    }

    /// A gap is reported and the mixer blends the inputs that are connected; a
    /// missing input is never guessed.
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

/// One step of a graph that contains mixers, in evaluation order. A value is
/// an image keyed by the node that produces it; [`NodeGraph::INPUT`] is the
/// frame before any layer, and a key nothing has produced yet means that frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MixStep {
    /// Layers applied in series to the image of `from`; the result is the value
    /// of `to` (the last node of the run). A layer that is disabled or has no
    /// opacity passes its input through.
    Chain { layers: Vec<u64>, from: u64, to: u64 },
    /// Every branch grades the same image `reference` (the nearest node all
    /// inputs descend from). `out = I + sum_i (B_i - I)`: each branch's own mask
    /// and opacity are already inside `B_i`.
    Parallel { node: u64, reference: u64, inputs: Vec<u64> },
    /// Inputs stack in ascending input number; the highest is on top.
    Stack { node: u64, inputs: Vec<StackInput> },
}

/// One input of a [`MixStep::Stack`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackInput {
    /// The node whose image this input carries.
    pub source: u64,
    /// The first non-layer node upstream of `source`: the image the layers of
    /// this input were graded from.
    pub origin: u64,
    /// The layers between `origin` and `source`, upstream first. Their combined
    /// mask coverage is how much of this input replaces the stack below it. An
    /// empty list is an opaque image.
    pub layers: Vec<u64>,
}

/// How a graph with mixers renders. Absent for a purely serial graph, which
/// keeps rendering as the flat [`render_layers`] list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixPlan {
    pub steps: Vec<MixStep>,
    /// The node that feeds the output.
    pub output: u64,
}

impl NodeGraph {
    /// The evaluation steps for a graph whose output depends on a mixer, or
    /// `None` when it is purely serial (or has no mixer on the render path).
    pub fn mix_plan(&self, rows: &[Layer]) -> Result<Option<MixPlan>, GraphError> {
        let graph = self.repaired(rows);
        let plan = graph.plan()?;
        let live = |id: u64| plan.order.contains(&id);
        let is_mixer = |id: u64| {
            graph.node(id).is_some_and(|n| matches!(n.kind, NodeKind::Parallel | NodeKind::LayerMixer))
        };
        if !plan.order.iter().any(|id| is_mixer(*id)) {
            return Ok(None);
        }
        // A link counts only between live nodes; a dangling or pruned feeder
        // is the unprocessed frame.
        let feeder = |to: u64, input: u32| graph.source(to, input).filter(|from| live(*from));
        let fanout = |id: u64| {
            graph.links.iter().filter(|l| l.from == id && live(l.to)).count()
        };
        let ancestors = |id: u64| {
            let mut seen = vec![id];
            let mut stack = vec![id];
            while let Some(n) = stack.pop() {
                for l in graph.links.iter().filter(|l| l.to == n && live(l.from)) {
                    if !seen.contains(&l.from) {
                        seen.push(l.from);
                        stack.push(l.from);
                    }
                }
            }
            seen
        };
        let mut steps: Vec<MixStep> = Vec::new();
        for &id in &plan.order {
            let Some(node) = graph.node(id) else { continue };
            match node.kind {
                NodeKind::Layer => {
                    let from = feeder(id, 0);
                    let extend = from.filter(|f| graph.is_layer(*f) && fanout(*f) == 1);
                    let open = extend.and_then(|f| {
                        steps.iter().position(
                            |s| matches!(s, MixStep::Chain { to, .. } if *to == f),
                        )
                    });
                    match open {
                        Some(at) => {
                            if let MixStep::Chain { layers, to, .. } = &mut steps[at] {
                                layers.push(id);
                                *to = id;
                            }
                        }
                        None => steps.push(MixStep::Chain {
                            layers: vec![id],
                            from: from.unwrap_or(Self::INPUT),
                            to: id,
                        }),
                    }
                }
                NodeKind::Parallel | NodeKind::LayerMixer => {
                    let mut inputs: Vec<(u32, u64)> = graph
                        .links
                        .iter()
                        .filter(|l| l.to == id && live(l.from))
                        .map(|l| (l.input, l.from))
                        .collect();
                    inputs.sort_by_key(|(input, _)| *input);
                    if node.kind == NodeKind::Parallel {
                        let sources: Vec<u64> = inputs.iter().map(|(_, from)| *from).collect();
                        // Latest node in evaluation order that every input
                        // descends from (or is).
                        let reference = sources
                            .first()
                            .map(|first| {
                                let rest: Vec<Vec<u64>> =
                                    sources.iter().skip(1).map(|s| ancestors(*s)).collect();
                                plan.order
                                    .iter()
                                    .rev()
                                    .copied()
                                    .find(|c| {
                                        ancestors(*first).contains(c)
                                            && rest.iter().all(|a| a.contains(c))
                                    })
                                    .unwrap_or(Self::INPUT)
                            })
                            .unwrap_or(Self::INPUT);
                        steps.push(MixStep::Parallel { node: id, reference, inputs: sources });
                    } else {
                        let stack = inputs
                            .iter()
                            .map(|(_, source)| {
                                let mut layers = Vec::new();
                                let mut at = *source;
                                while graph.is_layer(at) {
                                    layers.push(at);
                                    at = feeder(at, 0).unwrap_or(Self::INPUT);
                                }
                                layers.reverse();
                                StackInput { source: *source, origin: at, layers }
                            })
                            .collect();
                        steps.push(MixStep::Stack { node: id, inputs: stack });
                    }
                }
                NodeKind::Input | NodeKind::Output => {}
            }
        }
        let output = feeder(Self::OUTPUT, 0).unwrap_or(Self::INPUT);
        Ok(Some(MixPlan { steps, output }))
    }
}

/// [`NodeGraph::mix_plan`] for the render path. `None` means "render the flat
/// [`render_layers`] list", which is also what a refused graph does.
pub fn mix_plan(adj: &AdjustmentSet) -> Option<MixPlan> {
    adj.nodes.as_ref()?.mix_plan(&adj.layers).ok().flatten()
}

/// What a graph edit makes stale.
///
/// Design BO-10 §3 B5. This is the *decision* layer: it says whether a render
/// is needed at all and which nodes feed the changed ones. It does not cache
/// node outputs (a 24MP frame is ~96 MiB, so per-node caching is opt-in and
/// lives elsewhere); it only narrows what has to be recomputed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphDirty {
    /// Nodes whose own fingerprint changed, edited and otherwise.
    pub changed: Vec<u64>,
    /// `changed` plus everything downstream of it reachable from the output.
    pub dirty: Vec<u64>,
    /// Layer rows that must re-render, in the plan's render order.
    pub layers: Vec<u64>,
    /// The output's recursive identity moved, so no cached frame can stand in
    /// for a render. A pruned-only edit leaves this `false` even though
    /// `changed` and `dirty` are non-empty: those nodes cannot reach a pixel.
    pub output_changed: bool,
}

impl GraphDirty {
    /// The rendered frame is unchanged: the output's recursive identity is the
    /// same and no live layer has to re-render. A pruned-only edit is empty.
    ///
    /// This is the predicate a preview cache must use to serve a previous
    /// frame. It deliberately does not test `dirty`: a change confined to a
    /// pruned branch still names that branch, and a deleted live layer leaves
    /// `layers` empty because it is no longer live — `output_changed` is what
    /// distinguishes those from a no-op edit.
    pub fn is_empty(&self) -> bool {
        !self.output_changed && self.layers.is_empty()
    }
}

/// FNV-1a over a serialized value. Stable across runs and Rust versions, and
/// sensitive to every field that reaches a pixel.
fn fnv_of_serialized<T: Serialize>(value: &T) -> u64 {
    struct Fnv(u64);

    impl std::io::Write for Fnv {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            for byte in bytes {
                self.0 ^= u64::from(*byte);
                self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
            }
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let mut hash = Fnv(0xcbf2_9ce4_8422_2325u64);
    serde_json::to_writer(&mut hash, value).unwrap_or_default();
    hash.0
}

fn mix(hash: u64, value: u64) -> u64 {
    let mut hash = hash ^ value;
    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    hash
}

fn node_kind_tag(kind: NodeKind) -> u64 {
    match kind {
        NodeKind::Input => 1,
        NodeKind::Output => 2,
        NodeKind::Layer => 3,
        NodeKind::Parallel => 4,
        NodeKind::LayerMixer => 5,
    }
}

impl NodeGraph {
    /// Fingerprint of every node's output: its kind, its layer row (when it has
    /// one) and the fingerprints of the inputs that feed it, in input order.
    ///
    /// Two graphs that render the same pixels at a node have the same value; a
    /// change anywhere upstream changes every value downstream of it.
    pub fn fingerprints(&self, rows: &[Layer]) -> BTreeMap<u64, u64> {
        let mut memo = BTreeMap::new();
        let mut stack = Vec::new();
        for node in &self.nodes {
            self.node_fingerprint(node.id, rows, &mut memo, &mut stack);
        }
        // The output can be implicit in a stored graph; include it anyway so
        // `fingerprint()` never depends on node order.
        self.node_fingerprint(Self::OUTPUT, rows, &mut memo, &mut stack);
        memo
    }

    fn node_fingerprint(
        &self,
        id: u64,
        rows: &[Layer],
        memo: &mut BTreeMap<u64, u64>,
        stack: &mut Vec<u64>,
    ) -> u64 {
        if let Some(value) = memo.get(&id) {
            return *value;
        }
        if stack.contains(&id) {
            // A loop has no stable fingerprint; report a fixed value rather
            // than recursing forever. `plan()` refuses such a graph anyway.
            return 0;
        }
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        match self.node(id) {
            Some(node) => {
                hash = mix(hash, node_kind_tag(node.kind));
                let row = match node.kind {
                    NodeKind::Layer => rows
                        .iter()
                        .find(|row| row.id == id)
                        .map(fnv_of_serialized)
                        .unwrap_or(0),
                    _ => 0,
                };
                hash = mix(hash, row);
            }
            // The output node may be absent from the stored list; it still has
            // an identity so a link change shows up downstream.
            None if id == Self::OUTPUT => hash = mix(hash, 2),
            None => return 0,
        }
        stack.push(id);
        let mut inputs: Vec<&NodeLink> = self.links.iter().filter(|link| link.to == id).collect();
        inputs.sort_by_key(|link| link.input);
        for link in inputs {
            let from = self.node_fingerprint(link.from, rows, memo, stack);
            hash = mix(hash, u64::from(link.input));
            hash = mix(hash, from);
        }
        stack.pop();
        memo.insert(id, hash);
        hash
    }

    /// The identity of the whole graph's output.
    pub fn fingerprint(&self, rows: &[Layer]) -> u64 {
        self.fingerprints(rows).get(&Self::OUTPUT).copied().unwrap_or(0)
    }

    /// Each node's *own* inputs: its kind, its row and the `(input, from)`
    /// pairs wired into it. Not recursive, so a diff of this map names exactly
    /// where an edit happened instead of flagging everything downstream.
    pub fn local_fingerprints(&self, rows: &[Layer]) -> BTreeMap<u64, u64> {
        let mut out = BTreeMap::new();
        for node in &self.nodes {
            let mut hash = mix(0xcbf2_9ce4_8422_2325u64, node_kind_tag(node.kind));
            let row = match node.kind {
                NodeKind::Layer => {
                    rows.iter().find(|row| row.id == node.id).map(fnv_of_serialized).unwrap_or(0)
                }
                _ => 0,
            };
            hash = mix(hash, row);
            let mut inputs: Vec<&NodeLink> =
                self.links.iter().filter(|link| link.to == node.id).collect();
            inputs.sort_by_key(|link| link.input);
            for link in inputs {
                hash = mix(hash, u64::from(link.input));
                hash = mix(hash, link.from);
            }
            out.insert(node.id, hash);
        }
        if !out.contains_key(&Self::OUTPUT) {
            let mut hash = mix(0xcbf2_9ce4_8422_2325u64, 2);
            let mut inputs: Vec<&NodeLink> =
                self.links.iter().filter(|link| link.to == Self::OUTPUT).collect();
            inputs.sort_by_key(|link| link.input);
            for link in inputs {
                hash = mix(hash, u64::from(link.input));
                hash = mix(hash, link.from);
            }
            out.insert(Self::OUTPUT, hash);
        }
        out
    }

    /// What changed between `previous` and `self`, and what has to recompute.
    ///
    /// `rows`/`previous_rows` are the layer rows each graph is paired with, so
    /// a parameter edit on a layer row is seen even when the topology is
    /// unchanged. Nodes that the output does not depend on are reported in
    /// `changed` but never in `dirty.layers`: a pruned edit cannot reach a
    /// pixel.
    pub fn dirty_between(
        &self,
        previous: &NodeGraph,
        rows: &[Layer],
        previous_rows: &[Layer],
    ) -> GraphDirty {
        // The recursive output identity is the ground truth for "do these two
        // graphs render the same pixels"; local fingerprints only name the
        // edit site. Computing both once is cheap next to a frame render.
        let output_changed = self.fingerprint(rows) != previous.fingerprint(previous_rows);
        // Local fingerprints name the edit site; the recursive ones would flag
        // every downstream node as "changed" too.
        let now = self.local_fingerprints(rows);
        let before = previous.local_fingerprints(previous_rows);
        let mut changed: Vec<u64> = Vec::new();
        for (id, value) in &now {
            if *id == Self::INPUT || *id == Self::OUTPUT {
                continue;
            }
            if before.get(id).copied().unwrap_or(u64::MAX) != *value {
                changed.push(*id);
            }
        }
        // A node that only existed before also changed something.
        for (id, value) in &before {
            if *id == Self::INPUT || *id == Self::OUTPUT {
                continue;
            }
            if now.get(id).copied().unwrap_or(u64::MAX) != *value && !changed.contains(id) {
                changed.push(*id);
            }
        }
        // Rewiring the output changes the composition without changing either
        // branch's own inputs. Name both branch roots so the change is never
        // mistaken for a pruned edit (the output marker itself is not a render
        // node, so it is deliberately excluded from `changed` above).
        if now.get(&Self::OUTPUT) != before.get(&Self::OUTPUT) {
            for link in self.links.iter().filter(|link| link.to == Self::OUTPUT) {
                if !changed.contains(&link.from) {
                    changed.push(link.from);
                }
            }
            for link in previous.links.iter().filter(|link| link.to == Self::OUTPUT) {
                if !changed.contains(&link.from) {
                    changed.push(link.from);
                }
            }
        }
        changed.sort_unstable();
        changed.dedup();

        // Downstream closure over this graph's links.
        let mut dirty: Vec<u64> = changed.clone();
        let mut pending = changed.clone();
        while let Some(id) = pending.pop() {
            for link in self.links.iter().filter(|link| link.from == id) {
                // The output marker consumes the frame; it is not a render node.
                if link.to == Self::OUTPUT || dirty.contains(&link.to) {
                    continue;
                }
                dirty.push(link.to);
                pending.push(link.to);
            }
        }
        dirty.sort_unstable();
        dirty.dedup();

        // Only live layer rows matter; a pruned change renders nothing.
        let layers = match self.plan() {
            Ok(plan) => {
                // A pruned edit must not invalidate a live frame.
                let live: Vec<u64> = plan.layers;
                if dirty.iter().any(|id| plan.pruned.contains(id))
                    && !dirty.iter().any(|id| live.contains(id))
                {
                    Vec::new()
                } else {
                    live.into_iter().filter(|id| dirty.contains(id)).collect()
                }
            }
            Err(_) => Vec::new(),
        };
        GraphDirty { changed, dirty, layers, output_changed }
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
    fn mix_plan_branches_share_their_common_ancestor() {
        // 1 feeds branches 2 and 3; the parallel mixer 10 grades 1's image.
        let g = graph(&[(I, 1, 0), (1, 2, 0), (1, 3, 0), (2, 10, 0), (3, 10, 1), (10, O, 0)], &[1, 2, 3]);
        let rows = [row(1), row(2), row(3)];
        let mix = g.mix_plan(&rows).unwrap().unwrap();
        assert_eq!(mix.output, 10);
        assert!(mix.steps.contains(&MixStep::Parallel { node: 10, reference: 1, inputs: vec![2, 3] }));
        assert!(mix.steps.contains(&MixStep::Chain { layers: vec![1], from: I, to: 1 }));
        // A serial graph has no mix plan.
        let serial = graph(&[(I, 1, 0), (1, 2, 0), (2, O, 0)], &[1, 2]);
        assert_eq!(serial.mix_plan(&rows).unwrap(), None);
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

    fn layer_with_exposure(id: u64, exposure: f32) -> Layer {
        let mut layer = row(id);
        layer.adjustments.exposure.exposure = exposure;
        layer
    }

    #[test]
    fn an_unchanged_graph_is_not_dirty() {
        let g = graph(&[(I, 1, 0), (1, 2, 0), (2, O, 0)], &[1, 2]);
        let rows = vec![layer_with_exposure(1, 0.25), layer_with_exposure(2, -0.5)];
        let dirty = g.dirty_between(&g, &rows, &rows);
        assert!(dirty.changed.is_empty());
        assert!(dirty.is_empty());
    }

    #[test]
    fn editing_a_layer_dirties_it_and_its_downstream_only() {
        let g = graph(&[(I, 1, 0), (1, 2, 0), (2, 3, 0), (3, O, 0)], &[1, 2, 3]);
        let before =
            vec![layer_with_exposure(1, 0.0), layer_with_exposure(2, 0.5), layer_with_exposure(3, 0.0)];
        let mut after = before.clone();
        after[1].adjustments.exposure.exposure = 1.0;
        let dirty = g.dirty_between(&g, &after, &before);
        assert_eq!(dirty.changed, vec![2], "only the edited row changed");
        assert_eq!(dirty.dirty, vec![2, 3], "downstream only: 1 is upstream");
        assert_eq!(dirty.layers, vec![2, 3], "render order, not id order");
        assert!(dirty.output_changed, "a live edit changes the frame");
    }

    #[test]
    fn editing_a_pruned_layer_does_not_invalidate_a_live_frame() {
        // 3 hangs off 1 but never reaches the output.
        let g = graph(&[(I, 1, 0), (1, 2, 0), (1, 3, 0), (2, O, 0)], &[1, 2, 3]);
        let before = vec![layer_with_exposure(1, 0.0), layer_with_exposure(2, 0.0), layer_with_exposure(3, 0.0)];
        let mut after = before.clone();
        after[2].adjustments.exposure.exposure = 2.0;
        let dirty = g.dirty_between(&g, &after, &before);
        assert_eq!(dirty.changed, vec![3]);
        assert!(dirty.layers.is_empty(), "a pruned edit cannot reach a pixel");
        assert!(!dirty.output_changed, "a pruned edit cannot change the frame");
        assert!(dirty.is_empty(), "a cached frame still holds after a pruned edit");
    }

    #[test]
    fn a_topology_change_dirties_the_nodes_it_now_feeds() {
        let before = graph(&[(I, 1, 0), (1, 2, 0), (2, O, 0)], &[1, 2, 3]);
        // 3 joins in series between 2 and the output.
        let after = graph(&[(I, 1, 0), (1, 2, 0), (2, 3, 0), (3, O, 0)], &[1, 2, 3]);
        let rows = vec![row(1), row(2), row(3)];
        let dirty = after.dirty_between(&before, &rows, &rows);
        // The output's own input moved from 2 to 3, so both branch roots are
        // named: 3 must render, and 2 is the previous tail. Conservative by
        // design — the frame is stale either way.
        assert_eq!(dirty.layers, vec![2, 3], "both sides of the rewired output re-render");
        assert!(dirty.output_changed);
        assert!(!dirty.is_empty());
    }

    #[test]
    fn rewiring_the_output_is_never_a_pruned_edit() {
        // Both layers are live; only which one the output consumes changes.
        let before = graph(&[(I, 1, 0), (I, 2, 0), (1, O, 0), (2, O, 1)], &[1, 2]);
        let after = graph(&[(I, 1, 0), (I, 2, 0), (2, O, 0), (1, O, 1)], &[1, 2]);
        let rows = vec![row(1), row(2)];
        let dirty = after.dirty_between(&before, &rows, &rows);
        assert!(dirty.output_changed, "the output composition changed");
        assert!(!dirty.layers.is_empty(), "a swapped output must re-render a live layer");
        assert!(!dirty.is_empty(), "the cached frame must not be reused");
    }

    #[test]
    fn a_parallel_branch_edit_does_not_dirty_its_sibling() {
        // 1 -> 2 and 1 -> 3 both feed mixer 10, which feeds the output.
        let g = graph(&[(I, 1, 0), (1, 2, 0), (1, 3, 0), (2, 10, 0), (3, 10, 1), (10, O, 0)], &[1, 2, 3]);
        let before = vec![layer_with_exposure(1, 0.0), layer_with_exposure(2, 0.0), layer_with_exposure(3, 0.0)];
        let mut after = before.clone();
        after[1].adjustments.exposure.exposure = 0.5;
        let dirty = g.dirty_between(&g, &after, &before);
        assert_eq!(dirty.layers, vec![2], "3 is a sibling, not downstream");
    }

    #[test]
    fn the_graph_fingerprint_covers_rows_and_links() {
        let g = graph(&[(I, 1, 0), (1, O, 0)], &[1]);
        let rows = vec![layer_with_exposure(1, 0.0)];
        let base = g.fingerprint(&rows);
        let mut edited = rows.clone();
        edited[0].adjustments.exposure.exposure = 0.5;
        assert_ne!(base, g.fingerprint(&edited));

        let relinked = graph(&[(I, 77, 0), (77, O, 0)], &[1]);
        assert_ne!(base, relinked.fingerprint(&rows), "a different source is a different frame");
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
