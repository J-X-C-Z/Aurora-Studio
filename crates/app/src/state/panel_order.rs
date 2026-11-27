//! Where tool panels sit inside a tab: the order the user dragged them into
//! and the panels pinned to the top. Both live in the workspace layout, so a
//! saved workspace brings its arrangement along.

use gpui_kit::Context;

use super::{AppState, ToolTab, tab_id};
use crate::i18n::t;

/// Where a dragged tool panel would land if it were let go now.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum DropAt {
    /// Over nothing that takes it: no line shows.
    #[default]
    Nowhere,
    /// Against a panel; `true` puts it under that panel rather than over it.
    Panel(&'static str, bool),
    /// The space under a stack, keyed by the stack's first panel.
    End(&'static str),
    /// A tab in the strip that takes panels (Quick or the user's own).
    Tab(ToolTab),
}

/// The panel being dragged and where it would land, shown as a line that
/// does not move the panels around it. `just_moved` lets the dropped panel
/// fade into its new place once.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PanelDrag {
    pub dragged: Option<&'static str>,
    pub at: DropAt,
    pub just_moved: Option<(&'static str, u64)>,
}

/// The panel a drop `after` `target` goes in front of: the next one shown that
/// is not the dragged panel itself, or `None` for the end.
pub fn before_for(shown: &[&'static str], moved: &str, target: &'static str, after: bool) -> Option<&'static str> {
    if !after {
        return Some(target);
    }
    let at = shown.iter().position(|s| *s == target)?;
    shown[at + 1..].iter().copied().find(|s| *s != moved)
}

/// `natural` (the tab's built-in order) rearranged by `saved`: panels the saved
/// list names come in its order, panels it does not name keep their built-in
/// index, and pinned panels then move to the front in pin order.
pub fn arrange(natural: &[&'static str], saved: &[String], pinned: &[String]) -> Vec<&'static str> {
    let mut order: Vec<&'static str> = Vec::with_capacity(natural.len());
    for id in saved {
        if let Some(n) = natural.iter().find(|n| **n == id.as_str())
            && !order.contains(n)
        {
            order.push(n);
        }
    }
    for (i, id) in natural.iter().enumerate() {
        if !order.contains(id) {
            order.insert(i.min(order.len()), id);
        }
    }
    let mut top: Vec<&'static str> = pinned.iter().filter_map(|p| order.iter().find(|o| **o == p.as_str()).copied()).collect();
    top.dedup();
    order.retain(|o| !top.contains(o));
    top.extend(order);
    top
}

/// `shown` with `moved` placed in front of `before` (at the end with `None`).
/// `None` when nothing changes or `moved` is not in the list.
pub fn reorder(shown: &[&'static str], moved: &str, before: Option<&str>) -> Option<Vec<&'static str>> {
    if before == Some(moved) {
        return None;
    }
    let item = *shown.iter().find(|s| **s == moved)?;
    let mut order: Vec<&'static str> = shown.iter().copied().filter(|s| *s != moved).collect();
    let at = before.and_then(|b| order.iter().position(|s| *s == b)).unwrap_or(order.len());
    order.insert(at, item);
    (order != shown).then_some(order)
}

impl AppState {
    pub fn panel_pinned(&self, id: &str) -> bool {
        self.prefs.settings.workspace.pinned_panels.iter().any(|p| p == id)
    }

    /// Pins a panel to the top of its tab, or unpins it.
    pub fn toggle_pin(&mut self, id: &str, cx: &mut Context<Self>) {
        let pins = &mut self.prefs.settings.workspace.pinned_panels;
        let pinned = if let Some(i) = pins.iter().position(|p| p == id) {
            pins.remove(i);
            false
        } else {
            pins.push(id.to_string());
            true
        };
        // A pinned panel is there to be used: open it.
        if pinned {
            self.prefs.panels.collapsed.retain(|c| *c != id);
        }
        self.save_settings();
        cx.notify();
    }

    /// The order the current tab shows `natural` in.
    pub fn arranged_panels(&self, natural: &[&'static str]) -> Vec<&'static str> {
        let w = &self.prefs.settings.workspace;
        let saved = w.panel_order.get(&tab_id(self.tool_tab)).map(Vec::as_slice).unwrap_or(&[]);
        arrange(natural, saved, &w.pinned_panels)
    }

    /// Records where the dragged panel would land; repaints only when that changes.
    pub fn hover_panel_drop(&mut self, dragged: &'static str, at: DropAt, cx: &mut Context<Self>) {
        let d = &mut self.panel_drag;
        if d.dragged != Some(dragged) || d.at != at {
            d.dragged = Some(dragged);
            d.at = at;
            cx.notify();
        }
    }

    /// A drag began: a tool panel (`Some`) or anything else (`None`). A drag
    /// that ended away from every drop place leaves its panel and place
    /// behind; this keeps them from dimming a panel or lighting a tab during
    /// the next drag.
    pub fn start_drag(&mut self, panel: Option<&'static str>, cx: &mut Context<Self>) {
        let d = &mut self.panel_drag;
        if d.dragged != panel || d.at != DropAt::Nowhere {
            d.dragged = panel;
            d.at = DropAt::Nowhere;
            cx.notify();
        }
    }

    /// The pointer left a drop place: clears it if it is still the recorded one,
    /// so a line never lingers where the panel would not go.
    pub fn leave_panel_drop(&mut self, left: impl Fn(DropAt) -> bool, cx: &mut Context<Self>) {
        if left(self.panel_drag.at) {
            self.panel_drag.at = DropAt::Nowhere;
            cx.notify();
        }
    }

    /// Drops the dragged panel in front of `before` (`None`: the end) and
    /// lets it fade into its new place.
    pub fn drop_panel(&mut self, shown: &[&'static str], moved: &'static str, before: Option<&'static str>, cx: &mut Context<Self>) {
        let generation = self.panel_drag.just_moved.map_or(1, |(_, g)| g + 1);
        self.panel_drag = PanelDrag { just_moved: Some((moved, generation)), ..Default::default() };
        if before == Some(moved) {
            cx.notify();
            return;
        }
        self.move_panel(shown, moved, before, cx);
    }

    /// Drops `moved` in front of `before` in the current tab, `shown` being
    /// what the tab showed when the drag ended. A panel dragged among pinned
    /// ones is pinned; one dragged below them is unpinned.
    pub fn move_panel(&mut self, shown: &[&'static str], moved: &str, before: Option<&str>, cx: &mut Context<Self>) {
        let Some(order) = reorder(shown, moved, before) else { return };
        let pinned_before = before.is_some_and(|b| self.panel_pinned(b));
        let was_pinned = self.panel_pinned(moved);
        let w = &mut self.prefs.settings.workspace;
        if pinned_before && !was_pinned {
            w.pinned_panels.push(moved.to_string());
        } else if !pinned_before && was_pinned {
            // Dropped onto an unpinned panel or the end: it leaves the pinned group.
            w.pinned_panels.retain(|p| p != moved);
        }
        // Pins follow the dragged order among themselves.
        let pins: Vec<String> = order.iter().filter(|o| w.pinned_panels.iter().any(|p| p == *o)).map(|o| o.to_string()).collect();
        let others: Vec<String> = w.pinned_panels.iter().filter(|p| !order.contains(&p.as_str())).cloned().collect();
        w.pinned_panels = others.into_iter().chain(pins).collect();
        let ids: Vec<String> = order.iter().map(|o| o.to_string()).collect();
        let tab = self.tool_tab;
        if let Some(list) = self.tab_panels_mut(tab) {
            // A Quick or user tab's list is its order; keep panels it shows elsewhere.
            let rest: Vec<String> = list.iter().filter(|q| !ids.contains(q)).cloned().collect();
            *list = ids.into_iter().chain(rest).collect();
        } else {
            let w = &mut self.prefs.settings.workspace;
            // Panels this stack does not show (another section, another scope)
            // keep their saved places behind it.
            let key = tab_id(self.tool_tab);
            let rest: Vec<String> = w.panel_order.get(&key).into_iter().flatten().filter(|q| !ids.contains(q)).cloned().collect();
            w.panel_order.insert(key, ids.into_iter().chain(rest).collect());
        }
        self.save_settings();
        cx.notify();
    }

    /// Puts the current tab's panels back in their built-in order and unpins them.
    pub fn reset_panel_order(&mut self, shown: &[&'static str], cx: &mut Context<Self>) {
        let w = &mut self.prefs.settings.workspace;
        let saved = w.panel_order.remove(&tab_id(self.tool_tab)).unwrap_or_default();
        w.pinned_panels.retain(|p| !shown.contains(&p.as_str()) && !saved.contains(p));
        self.save_settings();
        self.notify(t("Panel order reset", "面板顺序已复位"), cx);
        cx.notify();
    }

    pub fn panel_order_custom(&self, shown: &[&'static str]) -> bool {
        let w = &self.prefs.settings.workspace;
        w.panel_order.contains_key(&tab_id(self.tool_tab)) || w.pinned_panels.iter().any(|p| shown.contains(&p.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: [&str; 4] = ["a", "b", "c", "d"];

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn nothing_saved_keeps_the_built_in_order() {
        assert_eq!(arrange(&N, &[], &[]), N);
    }

    #[test]
    fn saved_order_wins_and_unknown_ids_are_ignored() {
        assert_eq!(arrange(&N, &s(&["d", "zz", "a", "d"]), &[]), ["d", "b", "c", "a"]);
    }

    #[test]
    fn panels_new_to_a_tab_keep_their_place() {
        // "b" arrived after the order was saved.
        assert_eq!(arrange(&N, &s(&["c", "a", "d"]), &[]), ["c", "b", "a", "d"]);
    }

    #[test]
    fn pinned_panels_lead_in_pin_order() {
        assert_eq!(arrange(&N, &s(&["d", "c", "b", "a"]), &s(&["b", "x", "c"])), ["b", "c", "d", "a"]);
    }

    #[test]
    fn drop_after_a_panel_goes_in_front_of_the_next_one() {
        assert_eq!(before_for(&N, "a", "b", false), Some("b"));
        assert_eq!(before_for(&N, "a", "b", true), Some("c"));
        assert_eq!(before_for(&N, "c", "b", true), Some("d"), "skips the dragged panel");
        assert_eq!(before_for(&N, "a", "d", true), None, "after the last: the end");
        assert_eq!(before_for(&N, "d", "c", true), None);
    }

    #[test]
    fn reorder_moves_in_front_of_the_target_or_to_the_end() {
        assert_eq!(reorder(&N, "d", Some("b")).unwrap(), ["a", "d", "b", "c"]);
        assert_eq!(reorder(&N, "a", None).unwrap(), ["b", "c", "d", "a"]);
        assert_eq!(reorder(&N, "a", Some("b")), None, "already in front of b");
        assert_eq!(reorder(&N, "b", Some("b")), None);
        assert_eq!(reorder(&N, "zz", Some("b")), None);
    }
}
