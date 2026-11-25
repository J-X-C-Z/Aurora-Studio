//! Per-image edits with undo history, plus the editor's transient state:
//! the active layer or node, the spot tool's anchor and the adjustments
//! clipboard. Named operations (gestures, undo/redo, copy/apply, layers) live
//! on `AppState` because they also render, save and touch the catalog.

use std::collections::{BTreeSet, HashMap};

use gpui_kit::SharedString;
use aurora_engine::{AdjustmentSet, ImageId};

use super::clipboard::ClipGroup;

/// One image's adjustments and history.
#[derive(Clone, Default)]
pub struct Edit {
    pub adj: AdjustmentSet,
    pub undo: Vec<(SharedString, AdjustmentSet)>,
    pub redo: Vec<(SharedString, AdjustmentSet)>,
    /// Set while a drag is in progress: the state before it began, and its label.
    pub(crate) gesture: Option<(SharedString, AdjustmentSet)>,
}

/// Edits and undo history for every image, and the editor's own selection.
pub struct EditorState {
    pub edits: HashMap<ImageId, Edit>,
    /// `None` edits the background layer.
    pub active_layer: Option<u64>,
    /// A selected mixer or the output in the node editor; layers use `active_layer`.
    pub active_node: Option<u64>,
    /// The next id handed to a new layer or node.
    next_layer: u64,
    /// Clone tool's first click: the image and crop-normalised source point
    /// waiting for its target.
    pub spot_anchor: Option<(ImageId, [f32; 2])>,
    /// Adjustments copied from the current image, applied per `clip_groups`.
    pub clipboard: Option<AdjustmentSet>,
    /// Tools the adjustments clipboard copies and applies.
    pub clip_groups: BTreeSet<ClipGroup>,
}

impl Default for EditorState {
    fn default() -> Self {
        Self {
            edits: HashMap::new(),
            active_layer: None,
            active_node: None,
            next_layer: 1,
            spot_anchor: None,
            clipboard: None,
            clip_groups: ClipGroup::ALL.iter().copied().filter(|g| !g.is_composition()).collect(),
        }
    }
}

impl EditorState {
    /// The editor for a catalog whose edits are already loaded.
    pub fn with_edits(edits: HashMap<ImageId, Edit>) -> Self {
        Self { edits, ..Default::default() }
    }

    /// This image's edit, when it has one.
    pub fn edit_for(&self, id: ImageId) -> Option<&Edit> {
        self.edits.get(&id)
    }

    /// An id no layer or node using an id below `used` has taken.
    pub fn next_layer_id(&mut self, used: u64) -> u64 {
        let id = self.next_layer.max(used + 1);
        self.next_layer = id + 1;
        id
    }
}
