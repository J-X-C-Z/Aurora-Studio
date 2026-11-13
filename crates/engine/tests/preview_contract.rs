//! Contracts the interface relies on while a slider is dragged.

use std::sync::Arc;

use lumen_engine::{AdjustmentSet, ImageEngine, MockEngine, NormRect, PreviewRequest, RenderQuality};

fn request(image: lumen_engine::ImageId) -> PreviewRequest {
    PreviewRequest {
        image,
        adjustments: AdjustmentSet::default(),
        region: NormRect::FULL,
        max_size: 1400,
        quality: RenderQuality::Draft,
        bypass: false,
        mask_overlay: None,
        generation: 0,
     display_profile: None, proof_profile: None, proof_paper_white: 1.0, proof_black_point: 0.0, }
}

#[test]
fn fast_and_full_previews_have_the_same_output_size() {
    let engine = MockEngine::new();
    let mut req = request(engine.images()[0].id);
    req.adjustments.exposure.exposure = 0.5;
    let full = engine.render_preview(&req).unwrap();
    let fast = engine.render_preview_fast(&req, true).unwrap();
    assert_eq!((full.image.width, full.image.height), (fast.image.width, fast.image.height));
    assert_eq!(fast.image.pixels.len(), (fast.image.width * fast.image.height * 4) as usize);
}

#[test]
fn a_cache_hit_reuses_the_rendered_buffer() {
    let engine = MockEngine::new();
    let mut req = request(engine.images()[0].id);
    req.adjustments.exposure.exposure = 0.25;
    let first = engine.render_preview_fast(&req, true).unwrap();
    // A repaint keeps the same content but arrives as a newer generation.
    req.generation = 9;
    let second = engine.render_preview_fast(&req, true).unwrap();
    assert!(Arc::ptr_eq(&first.image.pixels, &second.image.pixels));
    assert_eq!(first.histogram, second.histogram);
    assert_eq!(second.generation, 9);
}
