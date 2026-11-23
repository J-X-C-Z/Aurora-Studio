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

    /// The layer nodes on a path from the input to the output, each after
    /// everything that feeds it (inputs in input order). Disconnected nodes are
    /// left out; a cycle is cut where it closes.
    pub fn render_order(&self) -> Vec<u64> {
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
            .filter(|id| self.node(*id).is_some_and(|n| n.kind == NodeKind::Layer) && self.reaches(Self::INPUT, *id))
            .collect()
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
