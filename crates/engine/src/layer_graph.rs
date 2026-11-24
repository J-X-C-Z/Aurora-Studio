//! Local layer node graph: a tree of blended RGB layers over a base image.
//!
//! This is the backend skeleton for a node-graph UI. It deliberately replaces
//! nothing: the flat [`crate::Layer`] list stays the UI model, and this module
//! only mixes finished per-layer pixels.
//!
//! Blend functions follow the W3C *Compositing and Blending Level 1* separable
//! formulas and are evaluated on the linear f32 values the engine works in, not
//! on sRGB bytes.

use serde::{Deserialize, Serialize};

use crate::layers::GraphError;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Screen,
    Overlay,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerNode {
    pub id: u64,
    /// Parent node id. Missing parents are treated as roots.
    pub parent: Option<u64>,
    pub blend: BlendMode,
    /// 0 ..= 1.
    pub opacity: f32,
    /// Per-pixel weight over `width * height`; `None` means fully opaque.
    pub mask: Option<Vec<f32>>,
    pub enabled: bool,
}

/// Composites `layers` over `base`, bottom to top, parent before child, siblings
/// by id. All buffers are RGB f32, interleaved, `width * height * 3` long.
///
/// Layers whose pixel buffer or mask does not match the canvas are skipped, so
/// a partially built graph never panics. The result is clamped to `0.0 ..= 1.0`.
/// A parent chain that loops is [`GraphError::Cycle`]: the old depth cap mixed
/// a partial stack and said nothing.
pub fn composite(
    width: usize,
    height: usize,
    base: &[f32],
    layers: &[(LayerNode, Vec<f32>)],
) -> Result<Vec<f32>, GraphError> {
    let len = width * height * 3;
    let mut out = vec![0.0; len];
    let copied = base.len().min(len);
    out[..copied].copy_from_slice(&base[..copied]);
    if width == 0 || height == 0 {
        return Ok(out);
    }
    let pixels = width * height;

    for index in tree_order(layers)? {
        let (node, source) = &layers[index];
        if !node.enabled || source.len() != pixels * 3 {
            continue;
        }
        if let Some(mask) = &node.mask {
            if mask.len() != pixels {
                continue;
            }
        }
        if !(node.opacity > 0.0) {
            continue;
        }
        let opacity = node.opacity.min(1.0);

        for pixel in 0..pixels {
            let alpha = match &node.mask {
                Some(mask) => opacity * mask[pixel].clamp(0.0, 1.0),
                None => opacity,
            };
            if !(alpha > 0.0) {
                continue;
            }
            for channel in 0..3 {
                let at = pixel * 3 + channel;
                let under = out[at];
                let blended = blend(node.blend, under, source[at]);
                out[at] = (under + (blended - under) * alpha).clamp(0.0, 1.0);
            }
        }
    }

    Ok(out)
}

fn blend(mode: BlendMode, backdrop: f32, source: f32) -> f32 {
    match mode {
        BlendMode::Normal => source,
        BlendMode::Multiply => backdrop * source,
        BlendMode::Screen => backdrop + source - backdrop * source,
        BlendMode::Overlay => {
            if backdrop <= 0.5 {
                2.0 * backdrop * source
            } else {
                1.0 - 2.0 * (1.0 - backdrop) * (1.0 - source)
            }
        }
    }
}

/// Indices of `layers` sorted by tree depth, then id. Depth follows the parent
/// chain; a chain that returns to a node it already passed is
/// [`GraphError::Cycle`], naming the members of the loop.
fn tree_order(layers: &[(LayerNode, Vec<f32>)]) -> Result<Vec<usize>, GraphError> {
    let mut by_id = std::collections::HashMap::new();
    for (index, (node, _)) in layers.iter().enumerate() {
        by_id.entry(node.id).or_insert(index);
    }

    let mut cyclic = Vec::new();
    let mut keyed: Vec<(usize, u64, usize)> = Vec::with_capacity(layers.len());
    for (index, (node, _)) in layers.iter().enumerate() {
        let mut depth = 0usize;
        let mut chain = vec![node.id];
        let mut parent = node.parent;
        while let Some(id) = parent {
            let Some(&up) = by_id.get(&id) else { break };
            let up_id = layers[up].0.id;
            if let Some(at) = chain.iter().position(|id| *id == up_id) {
                // Report the loop only: nodes that merely lead into it (an
                // earlier `5` in `5 -> 1 -> 2 -> 1`) are not members.
                cyclic.extend(chain[at..].iter().copied());
                break;
            }
            chain.push(up_id);
            depth += 1;
            parent = layers[up].0.parent;
        }
        keyed.push((depth, node.id, index));
    }
    if !cyclic.is_empty() {
        cyclic.sort_unstable();
        cyclic.dedup();
        return Err(GraphError::Cycle { nodes: cyclic });
    }
    keyed.sort_unstable();
    Ok(keyed.into_iter().map(|(_, _, index)| index).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u64, parent: Option<u64>, blend: BlendMode) -> LayerNode {
        LayerNode {
            id,
            parent,
            blend,
            opacity: 1.0,
            mask: None,
            enabled: true,
        }
    }

    fn fill(rgb: [f32; 3]) -> Vec<f32> {
        rgb.iter().copied().cycle().take(3).collect()
    }

    fn assert_px(actual: &[f32], expected: [f32; 3]) {
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-6, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn no_layers_is_identity() {
        let base = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        assert_eq!(composite(2, 1, &base, &[]).unwrap(), base);
    }

    #[test]
    fn opacity_and_mask_scale_the_layer() {
        let mut layer = node(1, None, BlendMode::Normal);
        layer.opacity = 0.5;
        layer.mask = Some(vec![0.5]);
        let out = composite(1, 1, &[0.0, 0.0, 0.0], &[(layer, vec![1.0, 1.0, 1.0])]).unwrap();
        assert_px(&out, [0.25, 0.25, 0.25]);
    }

    #[test]
    fn wrong_mask_size_skips_the_layer() {
        let mut layer = node(1, None, BlendMode::Normal);
        layer.mask = Some(vec![1.0, 1.0]);
        let out = composite(1, 1, &[0.2, 0.2, 0.2], &[(layer, vec![1.0, 1.0, 1.0])]).unwrap();
        assert_px(&out, [0.2, 0.2, 0.2]);
    }

    #[test]
    fn disabled_layer_is_ignored() {
        let mut layer = node(1, None, BlendMode::Normal);
        layer.enabled = false;
        let out = composite(1, 1, &[0.2, 0.2, 0.2], &[(layer, vec![1.0, 1.0, 1.0])]).unwrap();
        assert_px(&out, [0.2, 0.2, 0.2]);
    }

    #[test]
    fn parent_composites_before_child() {
        let mut parent = node(10, None, BlendMode::Screen);
        parent.opacity = 1.0;
        let child = node(11, Some(10), BlendMode::Multiply);
        // Passed child-first; the tree order must apply Screen then Multiply.
        let out = composite(
            1,
            1,
            &[0.25, 0.25, 0.25],
            &[(child, vec![0.5, 0.5, 0.5]), (parent, vec![0.5, 0.5, 0.5])],
        )
        .unwrap();
        assert_px(&out, [0.3125, 0.3125, 0.3125]);
    }

    #[test]
    fn four_blend_modes_at_known_pixels() {
        let base = [0.5, 1.0, 0.25];
        let top = [0.5, 0.5, 0.5];
        let cases = [
            (BlendMode::Normal, [0.5, 0.5, 0.5]),
            (BlendMode::Multiply, [0.25, 0.5, 0.125]),
            (BlendMode::Screen, [0.75, 1.0, 0.625]),
            (BlendMode::Overlay, [0.5, 1.0, 0.25]),
        ];
        for (mode, expected) in cases {
            let out = composite(1, 1, &base, &[(node(1, None, mode), top.to_vec())]).unwrap();
            assert_px(&out, expected);
        }
    }

    #[test]
    fn a_parent_cycle_is_an_error() {
        let a = node(1, Some(2), BlendMode::Normal);
        let b = node(2, Some(1), BlendMode::Normal);
        match composite(1, 1, &[0.0; 3], &[(a, vec![1.0; 3]), (b, vec![1.0; 3])]) {
            Err(crate::layers::GraphError::Cycle { nodes }) => assert_eq!(nodes, vec![1, 2]),
            other => panic!("expected a cycle, got {other:?}"),
        }
    }

    #[test]
    fn a_chain_into_a_cycle_reports_only_the_loop() {
        // 5 -> 1 -> 2 -> 1: 5 leads into the loop but is not part of it.
        let a = node(5, Some(1), BlendMode::Normal);
        let b = node(1, Some(2), BlendMode::Normal);
        let c = node(2, Some(1), BlendMode::Normal);
        match composite(
            1,
            1,
            &[0.0; 3],
            &[(a, vec![1.0; 3]), (b, vec![1.0; 3]), (c, vec![1.0; 3])],
        ) {
            Err(crate::layers::GraphError::Cycle { nodes }) => assert_eq!(nodes, vec![1, 2]),
            other => panic!("expected a cycle, got {other:?}"),
        }
    }

    #[test]
    fn empty_graph_and_short_base_are_safe() {
        assert!(composite(0, 0, &[], &[]).unwrap().is_empty());
        let out = composite(
            1,
            1,
            &[],
            &[(node(1, None, BlendMode::Normal), vec![1.0; 3])],
        )
        .unwrap();
        assert_px(&out, [1.0, 1.0, 1.0]);
        let orphan = composite(
            1,
            1,
            &[0.5, 0.5, 0.5],
            &[(node(7, Some(99), BlendMode::Normal), vec![0.0; 3])],
        )
        .unwrap();
        assert_px(&orphan, [0.0, 0.0, 0.0]);
    }
}
