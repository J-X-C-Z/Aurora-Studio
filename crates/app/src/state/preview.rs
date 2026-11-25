//! Viewer, thumbnail and preview-pipeline state split out of [`super::AppState`].
//!
//! This is a plain data holder, not a second model: the preview scheduler, the
//! thumbnail queue and the viewer's zoom/pan live here, but the UI still
//! observes `AppState` and all decisions stay in `state::render`. Field
//! semantics are unchanged from the former `AppState` fields: a fresh viewer
//! fits the image (`zoom.scale == None`), the viewport starts at its
//! pre-layout size, and `thumb_rev` starts at `0`.

use std::collections::HashMap;
use std::sync::Arc;

use aurora_engine::{AdjustmentSet, ImageId};
use gpui_kit::{Pixels, Point, RenderImage, SharedString, Task};

use super::render::ThumbQueue;
use super::{Preview, ViewZoom};

/// The viewer's image pipeline: the preview frames, thumbnail cache and queue,
/// hover look, zoom/pan, and the latest color-picker sample.
pub struct PreviewState {
    /// The after/before frames and the request scheduler for the viewer.
    pub preview: Preview,
    /// The engine-rendered thumbnail for each image.
    pub thumbs: HashMap<ImageId, Arc<RenderImage>>,
    /// Images still waiting for a thumbnail, current image first.
    pub(super) thumb_queue: ThumbQueue,
    /// Bumped when cached thumbnails are invalidated; older results are dropped.
    pub(super) thumb_rev: u64,
    pub(super) thumb_task: Option<Task<()>>,
    /// The current image with a hovered style applied, shown instead of its own
    /// adjustments until the pointer leaves the style.
    pub hover_look: Option<AdjustmentSet>,
    /// Zoom and pan of the viewer, in developed-frame space.
    pub zoom: ViewZoom,
    /// Viewer size in device pixels, written by the viewer each layout.
    pub viewport: (f32, f32),
    /// Window position of the viewer's top-left corner.
    pub viewport_origin: Point<Pixels>,
    pub sampled_color: Option<[f32; 3]>,
    pub picker_point: Option<[f32; 2]>,
    /// Why the engine shows the current image from something other than its full
    /// data (`planned::source_status`), refreshed with each preview.
    pub source_note: Option<(ImageId, SharedString)>,
}

impl Default for PreviewState {
    fn default() -> Self {
        Self {
            preview: Preview::default(),
            thumbs: HashMap::new(),
            thumb_queue: ThumbQueue::default(),
            thumb_rev: 0,
            thumb_task: None,
            hover_look: None,
            zoom: ViewZoom::default(),
            viewport: (1200.0, 800.0),
            viewport_origin: Point::default(),
            sampled_color: None,
            picker_point: None,
            source_note: None,
        }
    }
}
