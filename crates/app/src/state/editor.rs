//! Per-image edits with undo history, plus the editor's transient state:
//! the active layer or node, the spot tool's anchor and the adjustments
//! clipboard. The pure undo/redo/history transitions live here; the named
//! operations on `AppState` coordinate them with rendering, saving and the
//! catalog.

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

    /// Whether `id` has a step to undo.
    pub fn can_undo(&self, id: ImageId) -> bool {
        self.edit_for(id).is_some_and(|e| !e.undo.is_empty())
    }

    /// Whether `id` has a step to redo.
    pub fn can_redo(&self, id: ImageId) -> bool {
        self.edit_for(id).is_some_and(|e| !e.redo.is_empty())
    }

    /// `id`'s undo and redo step labels, oldest first.
    pub fn history_entries(&self, id: ImageId) -> (Vec<SharedString>, Vec<SharedString>) {
        self.edit_for(id)
            .map(|e| (e.undo.iter().map(|u| u.0.clone()).collect(), e.redo.iter().map(|u| u.0.clone()).collect()))
            .unwrap_or_default()
    }

    /// Moves `id`'s latest undo step onto its redo stack, returning its label
    /// when there was one. This is the whole state change; the caller renders,
    /// saves and notifies.
    pub fn undo_step(&mut self, id: ImageId) -> Option<SharedString> {
        let edit = self.edits.get_mut(&id)?;
        let (label, prev) = edit.undo.pop()?;
        let now = std::mem::replace(&mut edit.adj, prev);
        edit.redo.push((label.clone(), now));
        Some(label)
    }

    /// The reverse of [`Self::undo_step`].
    pub fn redo_step(&mut self, id: ImageId) -> Option<SharedString> {
        let edit = self.edits.get_mut(&id)?;
        let (label, next) = edit.redo.pop()?;
        let now = std::mem::replace(&mut edit.adj, next);
        edit.undo.push((label.clone(), now));
        Some(label)
    }

    /// Undoes or redoes `id` until its undo stack has `undo_len` steps. Returns
    /// `false` when the image has no edit to move.
    pub fn jump_history_step(&mut self, id: ImageId, undo_len: usize) -> bool {
        let Some(edit) = self.edits.get_mut(&id) else { return false };
        while edit.undo.len() > undo_len {
            let Some((label, prev)) = edit.undo.pop() else { break };
            let now = std::mem::replace(&mut edit.adj, prev);
            edit.redo.push((label, now));
        }
        while edit.undo.len() < undo_len {
            let Some((label, next)) = edit.redo.pop() else { break };
            let now = std::mem::replace(&mut edit.adj, next);
            edit.undo.push((label, now));
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adj(exposure: f32) -> AdjustmentSet {
        let mut a = AdjustmentSet::default();
        a.exposure.exposure = exposure;
        a
    }

    fn edit_with_undo(states: &[f32], current: f32) -> Edit {
        let mut edit = Edit { adj: adj(current), ..Default::default() };
        edit.undo = states.iter().map(|&v| (SharedString::from("Exposure"), adj(v))).collect();
        edit
    }

    #[test]
    fn undo_step_moves_one_step_and_redo_restores_it() {
        let id = ImageId(1);
        let mut editor = EditorState::default();
        editor.edits.insert(id, edit_with_undo(&[0.0], 2.0));

        assert!(editor.can_undo(id));
        assert!(!editor.can_redo(id));
        assert_eq!(editor.history_entries(id), (vec![SharedString::from("Exposure")], Vec::new()));

        let label = editor.undo_step(id).expect("one step to undo");
        assert_eq!(label, SharedString::from("Exposure"));
        assert_eq!(editor.edit_for(id).unwrap().adj.exposure.exposure, 0.0);
        assert!(!editor.can_undo(id));
        assert!(editor.can_redo(id));

        editor.redo_step(id).expect("one step to redo");
        assert_eq!(editor.edit_for(id).unwrap().adj.exposure.exposure, 2.0);
        assert!(editor.can_undo(id));
        assert!(!editor.can_redo(id));
    }

    #[test]
    fn undo_and_redo_are_none_without_an_edit_or_step() {
        let id = ImageId(2);
        let mut editor = EditorState::default();
        assert!(!editor.can_undo(id));
        assert!(!editor.can_redo(id));
        assert_eq!(editor.history_entries(id), (Vec::new(), Vec::new()));
        assert!(editor.undo_step(id).is_none());
        assert!(editor.redo_step(id).is_none());
        assert!(!editor.jump_history_step(id, 0));

        editor.edits.insert(id, Edit::default());
        assert!(editor.undo_step(id).is_none());
        assert!(editor.redo_step(id).is_none());
    }

    #[test]
    fn jump_history_step_walks_the_two_stacks() {
        let id = ImageId(3);
        let mut editor = EditorState::default();
        editor.edits.insert(id, edit_with_undo(&[0.0, 1.0, 2.0], 3.0));

        assert!(editor.jump_history_step(id, 1));
        assert_eq!(editor.edit_for(id).unwrap().adj.exposure.exposure, 1.0);
        assert_eq!(editor.history_entries(id).0.len(), 1);
        assert_eq!(editor.history_entries(id).1.len(), 2);

        assert!(editor.jump_history_step(id, 3));
        assert_eq!(editor.edit_for(id).unwrap().adj.exposure.exposure, 3.0);
        assert_eq!(editor.history_entries(id).0.len(), 3);
        assert!(editor.history_entries(id).1.is_empty());
    }
}
