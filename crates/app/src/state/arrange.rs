//! The browser's own arrangement: a custom order the user drags into shape, and
//! stacks that fold a burst or a bracket under one pick. Both are kept per
//! folder or catalog in the settings file, keyed by file path and variant, so
//! they survive restarts without touching the catalog.

use std::collections::{BTreeMap, HashSet};

use gpui_kit::Context;
use lumen_engine::{ImageId, ImageInfo};
use serde::{Deserialize, Serialize};

use super::{AppState, SortKey};
use crate::i18n::t;

/// What the settings file keeps for one folder or catalog.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Arrangement {
    /// Image keys in custom order.
    pub order: Vec<String>,
    /// Image keys per stack, the pick first.
    pub stacks: Vec<Vec<String>>,
}

/// A run of images shown as one: the pick stands for the rest until opened.
#[derive(Clone, Debug, PartialEq)]
pub struct Stack {
    pub ids: Vec<ImageId>,
}

impl Stack {
    pub fn pick(&self) -> ImageId {
        self.ids[0]
    }
}

pub fn image_key(info: &ImageInfo) -> String {
    format!("{}#{}", info.path.display(), info.variant)
}

/// Seconds since an arbitrary epoch from an EXIF-style timestamp
/// (`2026-10-01 06:00:00` or `2026:10:01 06:00:00`).
fn seconds(stamp: &str) -> Option<i64> {
    let digits: Vec<i64> = stamp.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).filter_map(|s| s.parse().ok()).collect();
    let [y, mo, d, h, mi, s, ..] = digits[..] else { return None };
    // Days are approximate across months; only gaps of seconds matter here.
    Some((((y * 12 + mo) * 31 + d) * 24 + h) * 3600 + mi * 60 + s)
}

impl AppState {
    fn arrangement_key(&self) -> String {
        self.source.path().map(|p| p.display().to_string()).unwrap_or_else(|| "demo".into())
    }

    /// Restores the custom order and stacks saved for what is open.
    pub(crate) fn load_arrangement(&mut self) {
        let Some(saved) = self.settings.arrangements.get(&self.arrangement_key()).cloned() else { return };
        let by_key: BTreeMap<String, ImageId> = self.images.iter().map(|i| (image_key(i), i.id)).collect();
        self.custom_order = saved.order.iter().filter_map(|k| by_key.get(k).copied()).collect();
        self.stacks = saved
            .stacks
            .iter()
            .map(|keys| Stack { ids: keys.iter().filter_map(|k| by_key.get(k).copied()).collect() })
            .filter(|s| s.ids.len() > 1)
            .collect();
    }

    fn save_arrangement(&mut self) {
        let key_of = |id: &ImageId| self.info(*id).map(image_key);
        let saved = Arrangement {
            order: self.custom_order.iter().filter_map(key_of).collect(),
            stacks: self.stacks.iter().map(|s| s.ids.iter().filter_map(key_of).collect()).collect(),
        };
        let key = self.arrangement_key();
        if saved == Arrangement::default() {
            self.settings.arrangements.remove(&key);
        } else {
            self.settings.arrangements.insert(key, saved);
        }
        self.settings.save();
    }

    pub fn stack_of(&self, id: ImageId) -> Option<&Stack> {
        self.stacks.iter().find(|s| s.ids.contains(&id))
    }

    pub fn stack_open(&self, pick: ImageId) -> bool {
        self.open_stacks.contains(&pick)
    }

    /// Folds stack members under their pick in an already sorted list; members
    /// of an open stack follow the pick directly.
    pub(crate) fn fold_stacks<'a>(&'a self, sorted: Vec<&'a ImageInfo>) -> Vec<&'a ImageInfo> {
        if self.stacks.is_empty() {
            return sorted;
        }
        let present: HashSet<ImageId> = sorted.iter().map(|i| i.id).collect();
        let mut out = Vec::with_capacity(sorted.len());
        for info in &sorted {
            match self.stack_of(info.id) {
                Some(stack) if stack.pick() == info.id => {
                    out.push(*info);
                    if self.stack_open(info.id) {
                        out.extend(stack.ids[1..].iter().filter(|id| present.contains(id)).filter_map(|id| sorted.iter().find(|i| i.id == *id).copied()));
                    }
                }
                // A member whose pick is filtered out stands in for the stack.
                Some(stack) if !present.contains(&stack.pick()) && stack.ids.iter().find(|id| present.contains(id)) == Some(&info.id) => out.push(*info),
                Some(_) => {}
                None => out.push(*info),
            }
        }
        out
    }

    /// Moves `ids` in front of `before` (or to the end) and switches the
    /// browser to the custom order, as dragging thumbnails does in Capture One.
    pub fn move_images(&mut self, ids: Vec<ImageId>, before: Option<ImageId>, cx: &mut Context<Self>) {
        if before.is_some_and(|b| ids.contains(&b)) {
            return;
        }
        if self.sort != SortKey::Custom {
            // Start from what the user is looking at.
            let shown: Vec<ImageId> = self.visible_images().iter().map(|i| i.id).collect();
            let mut order = shown.clone();
            order.extend(self.images.iter().map(|i| i.id).filter(|id| !shown.contains(id)));
            self.custom_order = order;
            self.sort = SortKey::Custom;
            self.sort_desc = false;
        }
        for id in self.images.iter().map(|i| i.id) {
            if !self.custom_order.contains(&id) {
                self.custom_order.push(id);
            }
        }
        self.custom_order.retain(|i| !ids.contains(i));
        let at = before.and_then(|b| self.custom_order.iter().position(|i| *i == b)).unwrap_or(self.custom_order.len());
        for (k, id) in ids.iter().enumerate() {
            self.custom_order.insert(at + k, *id);
        }
        self.save_arrangement();
        cx.notify();
    }

    /// Stacks the selection under the current image (or the first selected).
    pub fn stack_selection(&mut self, cx: &mut Context<Self>) {
        let order: Vec<ImageId> = self.visible_images().iter().map(|i| i.id).collect();
        let mut ids: Vec<ImageId> = order.iter().copied().filter(|id| self.selection.contains(id)).collect();
        // Members of folded stacks are selected through their pick.
        let folded: Vec<ImageId> = self.stacks.iter().filter(|s| ids.contains(&s.pick())).flat_map(|s| s.ids.clone()).collect();
        for id in folded {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        if ids.len() < 2 {
            self.notify(t("Select at least two images to stack", "请至少选择两张图像来堆叠"), cx);
            return;
        }
        if let Some(cur) = self.current.filter(|c| ids.contains(c)) {
            ids.retain(|i| *i != cur);
            ids.insert(0, cur);
        }
        self.stacks.retain(|s| !s.ids.iter().any(|i| ids.contains(i)));
        let n = ids.len();
        self.stacks.push(Stack { ids });
        self.save_arrangement();
        self.notify(format!("{} {n} {}", t("Stacked", "已堆叠"), t("images", "张图像")), cx);
    }

    pub fn unstack_selection(&mut self, cx: &mut Context<Self>) {
        let before = self.stacks.len();
        let sel = self.selection.clone();
        self.stacks.retain(|s| !s.ids.iter().any(|i| sel.contains(i)));
        if self.stacks.len() != before {
            self.save_arrangement();
            self.notify(t("Unstacked", "已取消堆叠"), cx);
        }
    }

    /// Stacks runs of images taken within `gap` seconds of each other.
    pub fn auto_stack(&mut self, gap: i64, cx: &mut Context<Self>) {
        let mut timed: Vec<(i64, ImageId)> = self
            .images
            .iter()
            .filter(|i| self.stack_of(i.id).is_none())
            .filter_map(|i| i.exif.captured_at.as_deref().and_then(seconds).map(|s| (s, i.id)))
            .collect();
        timed.sort();
        let mut made = 0;
        let mut run: Vec<ImageId> = Vec::new();
        let mut last = i64::MIN;
        let mut flush = |run: &mut Vec<ImageId>, stacks: &mut Vec<Stack>| {
            if run.len() > 1 {
                stacks.push(Stack { ids: std::mem::take(run) });
                made += 1;
            }
            run.clear();
        };
        for (s, id) in timed {
            if s - last > gap {
                flush(&mut run, &mut self.stacks);
            }
            run.push(id);
            last = s;
        }
        flush(&mut run, &mut self.stacks);
        self.save_arrangement();
        let msg = if made == 0 {
            t("No images were taken close enough together to stack", "没有拍摄时间足够接近的图像可以堆叠").to_string()
        } else {
            format!("{} {made} {}", t("Made", "已生成"), t("stacks", "个堆叠"))
        };
        self.notify(msg, cx);
    }

    /// Opens or closes the current image's stack, or all stacks.
    pub fn toggle_stacks_open(&mut self, cx: &mut Context<Self>) {
        match self.current.and_then(|c| self.stack_of(c)).map(|s| s.pick()) {
            Some(pick) => self.toggle_stack(pick, cx),
            None => {
                if self.open_stacks.is_empty() {
                    self.open_stacks = self.stacks.iter().map(|s| s.pick()).collect();
                } else {
                    self.open_stacks.clear();
                }
                cx.notify();
            }
        }
    }

    pub fn toggle_stack(&mut self, pick: ImageId, cx: &mut Context<Self>) {
        if !self.open_stacks.remove(&pick) {
            self.open_stacks.insert(pick);
        }
        cx.notify();
    }

    /// Makes `id` the image that stands for its stack.
    pub fn set_stack_pick(&mut self, id: ImageId, cx: &mut Context<Self>) {
        let Some(stack) = self.stacks.iter_mut().find(|s| s.ids.contains(&id)) else { return };
        let old = stack.pick();
        stack.ids.retain(|i| *i != id);
        stack.ids.insert(0, id);
        if self.open_stacks.remove(&old) {
            self.open_stacks.insert(id);
        }
        self.save_arrangement();
        cx.notify();
    }
}
