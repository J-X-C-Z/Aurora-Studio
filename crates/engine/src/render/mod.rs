//! Production render pipeline facade and `RenderPlan` execution integration.
//!
//! This module provides the stable production entry point for image rendering
//! (`render`, `render16`, `render_at`, `render_preview_bgra`, `PlanExecutor`,
//! `RenderSession`, `FrameCache`). To avoid maintaining a forked second copy of
//! pixel algorithms, the low-level entry points delegate zero-cost to the
//! verified core in `crate::mock::process`, preserving exact pixel semantics
//! and the GPU-resident preview fast path while enabling incremental migration.

pub mod cache;
pub mod execute;

pub use cache::{
    FrameCache, FrameCacheKey, FrameCacheStats, DEFAULT_FRAME_CACHE_BUDGET_BYTES,
};
pub use execute::{
    execute_with_plan, PlanExecutionReceipt, PlanExecutor, RenderError, RenderOutput,
    RenderRequestOptions, RenderSession, RenderTarget, RENDER_ALGORITHM_VERSION,
};

pub use crate::mock::process::{Rendered, Rendered16, Source};

use crate::{AdjustmentSet, NormRect};

/// Production 8-bit RGBA render entry point.
///
/// Zero-cost delegation to the verified `crate::mock::process::render`
/// implementation so mock and production paths share a single pixel pipeline.
#[inline]
pub fn render(
    src: &Source,
    adj: &AdjustmentSet,
    region: NormRect,
    max_size: u32,
    bypass: bool,
    mask_overlay: Option<u64>,
) -> Rendered {
    crate::mock::process::render(src, adj, region, max_size, bypass, mask_overlay)
}

/// Production 16-bit RGBA export render entry point.
///
/// Preserves the 16-bit quantization of the developed float pipeline.
#[inline]
pub fn render16(
    src: &Source,
    adj: &AdjustmentSet,
    region: NormRect,
    max_size: u32,
    bypass: bool,
) -> Rendered16 {
    crate::mock::process::render16(src, adj, region, max_size, bypass)
}

/// Production preview render that may return pixels directly in BGRA display
/// order when the GPU-resident fast path handles the whole frame.
#[inline]
pub(crate) fn render_preview_bgra(
    src: &Source,
    adj: &AdjustmentSet,
    region: NormRect,
    max_size: u32,
    bypass: bool,
    mask_overlay: Option<u64>,
    want_bgra: bool,
    scopes: crate::scopes::ScopeWants,
) -> Rendered {
    crate::mock::process::render_preview_bgra(
        src,
        adj,
        region,
        max_size,
        bypass,
        mask_overlay,
        want_bgra,
        scopes,
    )
}

/// Production preview; the legacy priority hint does not reduce sampling quality.
#[inline]
pub(crate) fn render_at(
    src: &Source,
    adj: &AdjustmentSet,
    region: NormRect,
    max_size: u32,
    bypass: bool,
    mask_overlay: Option<u64>,
    fast: bool,
    scopes: crate::scopes::ScopeWants,
) -> Rendered {
    crate::mock::process::render_at(src, adj, region, max_size, bypass, mask_overlay, fast, scopes)
}

/// Clamp a normalized region to `[0, 1] x [0, 1]`.
#[inline]
pub(crate) fn clean_region(r: NormRect) -> NormRect {
    crate::mock::process::clean_region(r)
}

/// Compute the output dimensions for a render request at `max_size`.
#[inline]
pub(crate) fn preview_size(
    src: &Source,
    adj: &AdjustmentSet,
    region: NormRect,
    max_size: u32,
) -> (u32, u32) {
    crate::mock::process::preview_size(src, adj, region, max_size)
}

/// Sample a single developed pixel at normalized frame coordinates `f`.
#[inline]
pub(crate) fn sample_point(src: &Source, adj: &AdjustmentSet, f: [f32; 2]) -> [f32; 3] {
    crate::mock::process::sample_point(src, adj, f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::compute::BackendPolicy;
    use crate::render_plan::BackendTag;
    use crate::{GraphError, GraphNode, Layer, NodeGraph, NodeKind, NodeLink};

    fn synthetic_source(w: u32, h: u32) -> Source {
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 4) as usize;
                rgba[idx] = ((x * 255) / w.max(1)) as u8;
                rgba[idx + 1] = ((y * 255) / h.max(1)) as u8;
                rgba[idx + 2] = (((x + y) * 127) / (w + h).max(1)) as u8;
                rgba[idx + 3] = 255;
            }
        }
        Source::from_rgba(w, h, &rgba)
    }

    fn doc_with_active_and_pruned_layer() -> AdjustmentSet {
        let active_id = 10;
        let pruned_id = 20;
        let mut active_layer = Layer {
            id: active_id,
            name: "Active".into(),
            enabled: true,
            opacity: 1.0,
            ..Default::default()
        };
        active_layer.adjustments.exposure.exposure = 0.25;

        let mut pruned_layer = Layer {
            id: pruned_id,
            name: "Pruned".into(),
            enabled: true,
            opacity: 1.0,
            ..Default::default()
        };
        pruned_layer.adjustments.exposure.exposure = 0.5;

        let graph = NodeGraph {
            nodes: vec![
                GraphNode { id: NodeGraph::INPUT, kind: NodeKind::Input, pos: [0.0, 0.0] },
                GraphNode { id: active_id, kind: NodeKind::Layer, pos: [100.0, 0.0] },
                GraphNode { id: pruned_id, kind: NodeKind::Layer, pos: [100.0, 200.0] },
                GraphNode { id: NodeGraph::OUTPUT, kind: NodeKind::Output, pos: [200.0, 0.0] },
            ],
            links: vec![
                NodeLink { from: NodeGraph::INPUT, to: active_id, input: 0 },
                NodeLink { from: active_id, to: NodeGraph::OUTPUT, input: 0 },
                // `pruned_id` is fed by INPUT but never reaches OUTPUT.
                NodeLink { from: NodeGraph::INPUT, to: pruned_id, input: 0 },
            ],
        };

        AdjustmentSet {
            layers: vec![active_layer, pruned_layer],
            nodes: Some(graph),
            ..Default::default()
        }
    }

    #[test]
    fn layout_or_pruned_branch_edit_reuses_cache_without_rerender() {
        let src = synthetic_source(32, 24);
        let doc1 = doc_with_active_and_pruned_layer();
        let opts = RenderRequestOptions::preview(NormRect::FULL, 32);
        let mut session = RenderSession::default();

        let (out1, r1) = execute_with_plan(&mut session, &src, &doc1, opts).expect("initial render");
        assert!(!r1.cache_hit);
        assert_eq!(r1.render_invocations, 1);
        assert_eq!(r1.cache_reuses, 0);
        assert_eq!(r1.pruned_nodes, vec![20]);
        assert_eq!(r1.compiled_order_len, 3);

        // Move node layout coordinates only -> dirty_count == 0, cache hit, no rerender.
        let mut doc2 = doc1.clone();
        if let Some(graph) = doc2.nodes.as_mut() {
            for node in &mut graph.nodes {
                node.pos[0] += 42.0;
                node.pos[1] -= 15.0;
            }
        }
        let (out2, r2) = execute_with_plan(&mut session, &src, &doc2, opts).expect("layout edit");
        assert!(r2.cache_hit);
        assert!(!r2.dirty_global);
        assert_eq!(r2.dirty_count, 0);
        assert_eq!(r2.render_invocations, 1);
        assert_eq!(r2.cache_reuses, 1);
        assert_eq!(
            out1.as_rgba8().unwrap().rgba,
            out2.as_rgba8().unwrap().rgba
        );

        // Edit pruned branch parameter only -> dirty_count == 0, cache hit, no rerender.
        let mut doc3 = doc2.clone();
        doc3.layers
            .iter_mut()
            .find(|l| l.id == 20)
            .unwrap()
            .adjustments
            .exposure
            .exposure = 2.0;
        let (out3, r3) =
            execute_with_plan(&mut session, &src, &doc3, opts).expect("pruned branch edit");
        assert!(r3.cache_hit);
        assert!(!r3.dirty_global);
        assert_eq!(r3.dirty_count, 0);
        assert_eq!(r3.render_invocations, 1);
        assert_eq!(r3.cache_reuses, 2);
        assert_eq!(
            out1.as_rgba8().unwrap().rgba,
            out3.as_rgba8().unwrap().rgba
        );
    }

    #[test]
    fn global_exposure_and_active_layer_edits_rerender_byte_identical_to_mock_process() {
        let src = synthetic_source(32, 24);
        let mut doc = doc_with_active_and_pruned_layer();
        let opts = RenderRequestOptions::preview(NormRect::FULL, 32);
        let mut session = RenderSession::default();

        let (out1, r1) = session.execute(&src, &doc, opts).expect("initial");
        assert_eq!(r1.render_invocations, 1);
        let baseline1 = crate::mock::process::render(&src, &doc, NormRect::FULL, 32, false, None);
        assert_eq!(out1.as_rgba8().unwrap().rgba, baseline1.rgba);

        // Global exposure edit -> triggers rerender and matches mock::process byte-for-byte.
        doc.exposure.exposure = 1.2;
        let (out2, r2) = session.execute(&src, &doc, opts).expect("global exposure edit");
        assert!(!r2.cache_hit);
        assert!(r2.dirty_count > 0);
        assert_eq!(r2.render_invocations, 2);
        let baseline2 = crate::mock::process::render(&src, &doc, NormRect::FULL, 32, false, None);
        assert_eq!(out2.as_rgba8().unwrap().rgba, baseline2.rgba);
        assert_ne!(out1.as_rgba8().unwrap().rgba, out2.as_rgba8().unwrap().rgba);

        // Active layer parameter edit -> triggers rerender and matches mock::process byte-for-byte.
        doc.layers
            .iter_mut()
            .find(|l| l.id == 10)
            .unwrap()
            .adjustments
            .exposure
            .exposure = -0.8;
        let (out3, r3) = session.execute(&src, &doc, opts).expect("active layer edit");
        assert!(!r3.cache_hit);
        assert!(r3.dirty_count > 0);
        assert_eq!(r3.render_invocations, 3);
        let baseline3 = crate::mock::process::render(&src, &doc, NormRect::FULL, 32, false, None);
        assert_eq!(out3.as_rgba8().unwrap().rgba, baseline3.rgba);
    }

    #[test]
    fn preview_8bit_and_export_16bit_share_same_render_plan_semantics() {
        let src = synthetic_source(32, 24);
        let mut doc = doc_with_active_and_pruned_layer();
        doc.exposure.exposure = 0.6;

        let mut session = RenderSession::default();
        let preview_opts = RenderRequestOptions::preview(NormRect::FULL, 32);
        let export_opts = RenderRequestOptions::export16(NormRect::FULL, 32);

        let (out8, r8) = session.execute(&src, &doc, preview_opts).expect("8-bit preview");
        let (out16, r16) = session.execute(&src, &doc, export_opts).expect("16-bit export");

        // Same compiled plan topology & pruned set, separate precision cache entries.
        assert_eq!(r8.compiled_order_len, r16.compiled_order_len);
        assert_eq!(r8.pruned_nodes, r16.pruned_nodes);
        assert_ne!(r8.cache_key, r16.cache_key);
        assert!(!r16.cache_hit);
        assert_eq!(r16.render_invocations, 2);

        let rgba8 = out8.as_rgba8().expect("8-bit output");
        let rgba16 = out16.as_rgba16().expect("16-bit output");
        assert_eq!((rgba8.width, rgba8.height), (rgba16.width, rgba16.height));

        // Both 8-bit preview and 16-bit export match the reference pipeline byte-for-byte.
        let baseline8 = crate::mock::process::render(&src, &doc, NormRect::FULL, 32, false, None);
        assert_eq!(rgba8.rgba, baseline8.rgba);
        let baseline16 = crate::mock::process::render16(&src, &doc, NormRect::FULL, 32, false);
        assert_eq!(rgba16.rgba, baseline16.rgba);

        // Quantized 16-bit channels match the 8-bit preview semantically
        // (matching dimensions, channel order, tone direction, small mean diff).
        let mut sum_diff = 0u64;
        let mut max_diff = 0i32;
        for (&u8_val, &u16_val) in rgba8.rgba.iter().zip(rgba16.rgba.iter()) {
            let from_16 = ((u32::from(u16_val) * 255 + 32767) / 65535) as i32;
            let diff = (i32::from(u8_val) - from_16).abs();
            sum_diff += diff as u64;
            max_diff = max_diff.max(diff);
        }
        let mean_diff = sum_diff as f64 / rgba8.rgba.len() as f64;
        assert!(
            mean_diff <= 1.0 && max_diff <= 4,
            "8-bit preview and 16-bit export diverged unexpectedly: mean={mean_diff:.3}, max={max_diff}"
        );

        // Editing the pruned branch reuses cache for both 8-bit and 16-bit targets.
        let mut doc_pruned_edit = doc.clone();
        doc_pruned_edit
            .layers
            .iter_mut()
            .find(|l| l.id == 20)
            .unwrap()
            .adjustments
            .exposure
            .exposure = -1.5;
        let (_, r8_hit) = session.execute(&src, &doc_pruned_edit, preview_opts).unwrap();
        let (_, r16_hit) = session.execute(&src, &doc_pruned_edit, export_opts).unwrap();
        assert!(r8_hit.cache_hit);
        assert!(r16_hit.cache_hit);
        assert_eq!(session.render_invocations(), 2);
    }

    #[test]
    fn cycle_in_node_graph_returns_explicit_error() {
        let src = synthetic_source(16, 16);
        let l1 = Layer { id: 10, ..Default::default() };
        let l2 = Layer { id: 11, ..Default::default() };
        let graph = NodeGraph {
            nodes: vec![
                GraphNode { id: NodeGraph::INPUT, kind: NodeKind::Input, pos: [0.0; 2] },
                GraphNode { id: 10, kind: NodeKind::Layer, pos: [10.0, 0.0] },
                GraphNode { id: 11, kind: NodeKind::Layer, pos: [20.0, 0.0] },
                GraphNode { id: NodeGraph::OUTPUT, kind: NodeKind::Output, pos: [30.0, 0.0] },
            ],
            links: vec![
                NodeLink { from: NodeGraph::INPUT, to: 10, input: 0 },
                NodeLink { from: 10, to: 11, input: 0 },
                NodeLink { from: 11, to: 10, input: 1 },
                NodeLink { from: 11, to: NodeGraph::OUTPUT, input: 0 },
            ],
        };
        let doc = AdjustmentSet {
            layers: vec![l1, l2],
            nodes: Some(graph),
            ..Default::default()
        };

        let mut session = RenderSession::default();
        let opts = RenderRequestOptions::preview(NormRect::FULL, 16)
            .with_backend_policy(BackendPolicy::CpuOnly);
        let err = session
            .execute(&src, &doc, opts)
            .err()
            .expect("cycle must fail compilation");
        assert!(
            matches!(err, RenderError::Graph(GraphError::Cycle { .. })),
            "expected GraphError::Cycle, got {err:?}"
        );
        assert_eq!(session.render_invocations(), 0);
    }

    #[test]
    fn different_roi_and_sizes_are_isolated_in_cache() {
        let src = synthetic_source(64, 48);
        let doc = AdjustmentSet::default();
        let mut session = RenderSession::default();

        let full_32 = RenderRequestOptions::preview(NormRect::FULL, 32);
        let full_64 = RenderRequestOptions::preview(NormRect::FULL, 64);
        let left_half = RenderRequestOptions::preview(
            NormRect { x: 0.0, y: 0.0, width: 0.5, height: 1.0 },
            32,
        );
        let right_half = RenderRequestOptions::preview(
            NormRect { x: 0.5, y: 0.0, width: 0.5, height: 1.0 },
            32,
        );

        let (o_full32, r1) = session.execute(&src, &doc, full_32).unwrap();
        let (o_full64, r2) = session.execute(&src, &doc, full_64).unwrap();
        let (o_left, r3) = session.execute(&src, &doc, left_half).unwrap();
        let (o_right, r4) = session.execute(&src, &doc, right_half).unwrap();

        assert!(!r1.cache_hit && !r2.cache_hit && !r3.cache_hit && !r4.cache_hit);
        assert_eq!(session.render_invocations(), 4);
        assert_eq!(o_full32.width(), 32);
        assert_eq!(o_full64.width(), 64);
        assert_ne!(
            o_left.as_rgba8().unwrap().rgba,
            o_right.as_rgba8().unwrap().rgba
        );

        // Re-requesting left_half must hit cache and return exact left pixels.
        let (o_left_again, r5) = session.execute(&src, &doc, left_half).unwrap();
        assert!(r5.cache_hit);
        assert_eq!(session.render_invocations(), 4);
        assert_eq!(
            o_left.as_rgba8().unwrap().rgba,
            o_left_again.as_rgba8().unwrap().rgba
        );
    }

    #[test]
    fn frame_cache_evicts_lru_when_byte_budget_exceeded() {
        let src = synthetic_source(16, 16);
        let mut doc_a = AdjustmentSet::default();
        doc_a.exposure.exposure = 0.1;
        let mut doc_b = AdjustmentSet::default();
        doc_b.exposure.exposure = 0.2;
        let mut doc_c = AdjustmentSet::default();
        doc_c.exposure.exposure = 0.3;

        // Each 16x16 RGBA8 frame is 1024 bytes. Budget of 2048 bytes holds 2 frames.
        let mut session = RenderSession::new(2048);
        let opts = RenderRequestOptions::preview(NormRect::FULL, 16);

        let (_, ra) = session.execute(&src, &doc_a, opts).unwrap();
        let (_, rb) = session.execute(&src, &doc_b, opts).unwrap();
        assert!(!ra.cache_hit && !rb.cache_hit);
        assert_eq!(session.cache_stats().entries, 2);
        assert_eq!(session.cache_stats().used_bytes, 2048);
        assert_eq!(session.cache_stats().evictions, 0);

        // Touch `doc_a` so `doc_b` becomes least-recently-used.
        let (_, ra_hit) = session.execute(&src, &doc_a, opts).unwrap();
        assert!(ra_hit.cache_hit);
        assert_eq!(session.render_invocations(), 2);

        // Insert `doc_c`: must evict `doc_b` (LRU) while keeping `doc_a`.
        let (_, rc) = session.execute(&src, &doc_c, opts).unwrap();
        assert!(!rc.cache_hit);
        assert_eq!(session.cache_stats().entries, 2);
        assert_eq!(session.cache_stats().used_bytes, 2048);
        assert_eq!(session.cache_stats().evictions, 1);

        // `doc_a` is still cached; `doc_b` was evicted and must re-render.
        let (_, ra_still) = session.execute(&src, &doc_a, opts).unwrap();
        assert!(ra_still.cache_hit);
        let (_, rb_miss) = session.execute(&src, &doc_b, opts).unwrap();
        assert!(!rb_miss.cache_hit);
    }

    #[test]
    fn backend_equivalence_tags_isolate_unproven_backends_and_share_equivalent_ones() {
        let src = synthetic_source(16, 16);
        let doc = AdjustmentSet::default();
        let mut session = RenderSession::default();

        let cpu_opts = RenderRequestOptions::preview(NormRect::FULL, 16)
            .with_backend_tag(BackendTag::CPU);
        let cuda_opts = RenderRequestOptions::preview(NormRect::FULL, 16)
            .with_backend_tag(BackendTag::CUDA);
        let cpu_alias_opts = RenderRequestOptions::preview(NormRect::FULL, 16)
            .with_backend_tag(BackendTag::new("cpu-ref-alias", BackendTag::CPU.equivalence));

        let (_, r1) = session.execute(&src, &doc, cpu_opts).unwrap();
        assert!(!r1.cache_hit);
        assert_eq!(session.render_invocations(), 1);

        // Different equivalence tag -> isolated, does not reuse CPU entry.
        let (_, r2) = session.execute(&src, &doc, cuda_opts).unwrap();
        assert!(!r2.cache_hit);
        assert_eq!(session.render_invocations(), 2);

        // Same equivalence tag -> shares CPU entry.
        let (_, r3) = session.execute(&src, &doc, cpu_alias_opts).unwrap();
        assert!(r3.cache_hit);
        assert_eq!(session.render_invocations(), 2);
    }
}
