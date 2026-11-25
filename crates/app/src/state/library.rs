//! Catalog, selection and browser-arrangement state split out of
//! [`super::AppState`].
//!
//! This is a plain data holder, not a second model: the image list, which one
//! is current, the selection, the recent/trash sets, annotations, the custom
//! order and the stacks live here, but the UI still observes `AppState` and all
//! transitions stay on the named `AppState` methods. Field semantics are
//! unchanged from the former `AppState` fields: the built-in "All Images"
//! collection (`1`) starts active and a fresh viewer has no image
//! (`current == None`).

use std::collections::{BTreeSet, HashMap, HashSet};

use aurora_engine::{ImageId, ImageInfo};
use gpui_kit::SharedString;

use super::Stack;
use crate::i18n::t;

/// The catalog's images plus what the browser currently selects, orders and
/// folds into stacks.
pub struct LibraryState {
    /// Every image the engine has for this window.
    pub images: Vec<ImageInfo>,
    /// The images the tools and batch actions apply to.
    pub selection: BTreeSet<ImageId>,
    /// The image the tools edit.
    pub current: Option<ImageId>,
    /// Images imported in this session.
    pub recent_imports: BTreeSet<ImageId>,
    /// Images captured in this session.
    pub recent_captures: BTreeSet<ImageId>,
    /// Images moved to the trash collection.
    pub trash: BTreeSet<ImageId>,
    /// The collection or folder being browsed.
    pub active_collection: u64,
    /// Its display name.
    pub active_collection_name: SharedString,
    /// Freehand notes drawn with the annotate tool, in frame space, per image.
    pub annotations: HashMap<ImageId, Vec<Vec<(f32, f32)>>>,
    /// The order the user dragged thumbnails into.
    pub custom_order: Vec<ImageId>,
    /// Runs of images folded under one pick.
    pub stacks: Vec<Stack>,
    /// Picks of the stacks shown expanded.
    pub open_stacks: HashSet<ImageId>,
}

impl Default for LibraryState {
    fn default() -> Self {
        Self {
            images: Vec::new(),
            selection: BTreeSet::new(),
            current: None,
            recent_imports: BTreeSet::new(),
            recent_captures: BTreeSet::new(),
            trash: BTreeSet::new(),
            // The built-in "All Images" collection.
            active_collection: 1,
            active_collection_name: t("All Images", "所有图像").into(),
            annotations: HashMap::new(),
            custom_order: Vec::new(),
            stacks: Vec::new(),
            open_stacks: HashSet::new(),
        }
    }
}

impl LibraryState {
    /// The image record for `id`, if it is in the catalog.
    pub fn info(&self, id: ImageId) -> Option<&ImageInfo> {
        self.images.iter().find(|i| i.id == id)
    }

    /// The record of the current image, if there is one.
    pub fn current_info(&self) -> Option<&ImageInfo> {
        self.current.and_then(|id| self.info(id))
    }

    /// Whether `id` is in the selection.
    pub fn is_selected(&self, id: ImageId) -> bool {
        self.selection.contains(&id)
    }
}
