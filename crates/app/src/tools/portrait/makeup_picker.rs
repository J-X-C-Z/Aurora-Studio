//! Makeup style/colour bar and the whole-look (suit) picker for the Portrait tab.
//!
//! `style_bar` sits under one live makeup slider and edits only the keys the
//! engine reads today (`portrait_makeup_wiring`, `portrait_makeup_recipes`):
//! `pc:<pf>:style` (1-based ordinal), `pc:<pf>:color` (1-based palette ordinal)
//! and `pc:<pf>:rgb` (packed RGB24, stored even when it is black). Nothing here
//! writes `:r`/`:g`/`:b`/`:left`, because no consumer reads them. PF100 and
//! PF117 are the real left and right brows: the right one shows the engine's
//! "follows the left brow" fallback instead of borrowing a side that does not
//! exist, and the upper (PF105) and lower (PF118) lashes keep separate bars.
//!
//! Style choices are the material folders `makeup_presets_for_pf` discovers. Each
//! cell draws that folder's real PNG through GPUI's asynchronous `img` path (the
//! same `img(...).object_fit(ObjectFit::Contain)` the viewer uses for overlays);
//! the name and ordinal are the cell's title, never a substitute for the image.
//! Folder scans are cached per PF with a short TTL — an empty scan is retried
//! sooner than a full one so a hot-mounted asset folder appears without a restart
//! — and opening the grid forces one rescan instead of scanning every frame.
//!
//! Every click is a single `AppState::edit`, i.e. one undo step. A row the engine
//! cannot render on this photo stays visible but inert: a disabled chip never
//! writes, and the click re-checks the image it was drawn for (`ImageId`) and
//! `row_available` instead of trusting the frame it was drawn in.
//!
//! A whole look is validated **once** on a copy of the current adjustments and
//! then committed with one assignment of the validated `extra`; a refusal writes
//! nothing and never opens an undo step. Recipes may carry auxiliary
//! `pc:111:catchlight_*` keys (the eye catchlight sub-material of PF111). Those
//! stay attached to PF111 — they are not a new PF — and are replaced together
//! with the rest of the look.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use gpui_kit::assets::IconName;
use gpui_kit::component::{Icon, Sizable as _, h_flex, input::Input, tooltip::Tooltip, v_flex};
use gpui_kit::{
    AnyElement, App, Entity, InteractiveElement, IntoElement, ObjectFit, ParentElement, RenderOnce,
    SharedString, StatefulInteractiveElement, Styled, StyledImage as _, Window, div, img,
    prelude::FluentBuilder as _, px, rgb,
};

use aurora_engine::{AdjustmentSet, ImageId, MakeupPresetInfo, MakeupSuit};

use crate::i18n::t;
use crate::state::AppState;
use crate::theme::{Palette, pal};
use crate::widgets::field_state;

/// Makeup rows a whole-look recipe owns (the engine's `suit_key` set): switching
/// looks replaces exactly these controls.
const SUIT_CONTROL_PFS: [u32; 13] = [
    100, 101, 103, 105, 106, 107, 111, 115, 116, 117, 118, 30007, 90032,
];

const STYLE_CHIP_W: f32 = 78.0;
const SUIT_CHIP_W: f32 = 134.0;
const THUMB_H: f32 = 46.0;
const SWATCH: f32 = 14.0;

/// An empty scan is retried quickly so assets mounted while the app is running
/// show up; a full scan is trusted for longer so hovering never rescans.
const EMPTY_TTL: Duration = Duration::from_secs(2);
const FULL_TTL: Duration = Duration::from_secs(30);

type Act = Rc<dyn Fn(&mut Window, &mut App)>;

/// Wraps a click handler once so call sites keep an inferable closure signature.
fn act(f: impl Fn(&mut Window, &mut App) + 'static) -> Act {
    Rc::new(f)
}

// ───────────── Keys and pure state helpers ─────────────

fn pf_of(key: &str) -> Option<u32> {
    key.strip_prefix("pc:")?.parse().ok()
}

fn style_key(pf: u32) -> String {
    format!("pc:{pf}:style")
}

fn color_key(pf: u32) -> String {
    format!("pc:{pf}:color")
}

fn rgb_key(pf: u32) -> String {
    format!("pc:{pf}:rgb")
}

/// 1-based ordinals are stored as whole floats; below 1 means "not set".
fn ordinal(raw: Option<f32>) -> Option<usize> {
    let value = raw.filter(|v| v.is_finite())?;
    let index = value.round() as isize;
    (index >= 1).then_some(index as usize)
}

/// Packed RGB24, the only form `resolve_mix_color` accepts as a custom colour.
fn packed(raw: Option<f32>) -> Option<u32> {
    let value = raw.filter(|v| v.is_finite())?;
    ((0.0..=16_777_215.0).contains(&value) && value.fract() == 0.0).then_some(value as u32)
}

/// The style stored on `pf` itself.
fn own_style(extra: &BTreeMap<String, f32>, pf: u32) -> Option<usize> {
    ordinal(extra.get(&style_key(pf)).copied())
}

/// What the renderer will use. The right brow (PF117) falls back to the left one
/// exactly as `resolve_style_selection` does, so the bar shows that style rather
/// than a neutral "default" that is not what renders.
fn effective_style(extra: &BTreeMap<String, f32>, pf: u32) -> Option<usize> {
    own_style(extra, pf).or_else(|| (pf == 117).then(|| own_style(extra, 100)).flatten())
}

fn own_color(extra: &BTreeMap<String, f32>, pf: u32) -> Option<usize> {
    ordinal(extra.get(&color_key(pf)).copied())
}

fn own_rgb(extra: &BTreeMap<String, f32>, pf: u32) -> Option<u32> {
    packed(extra.get(&rgb_key(pf)).copied())
}

/// A colour choice with its two channels named, so a tuple can never be read the
/// other way round: `swatch` is the 1-based palette ordinal, `rgb` is packed RGB24.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ColorPick {
    swatch: Option<usize>,
    rgb: Option<u32>,
}

impl ColorPick {
    fn is_set(self) -> bool {
        self.swatch.is_some() || self.rgb.is_some()
    }

    fn of(extra: &BTreeMap<String, f32>, pf: u32) -> Self {
        Self {
            swatch: own_color(extra, pf),
            rgb: own_rgb(extra, pf),
        }
    }
}

/// The colour the renderer will use. The right brow inherits both channels from
/// the left brow, as `resolve_mix_color` does.
fn effective_color(extra: &BTreeMap<String, f32>, pf: u32) -> ColorPick {
    let own = ColorPick::of(extra, pf);
    if own.is_set() || pf != 117 {
        return own;
    }
    let left = ColorPick::of(extra, 100);
    if left.is_set() { left } else { own }
}

fn set_style(adjustments: &mut AdjustmentSet, pf: u32, style: Option<usize>) {
    match style {
        Some(index) => {
            adjustments.extra.insert(style_key(pf), index as f32);
        }
        None => {
            adjustments.extra.remove(&style_key(pf));
        }
    }
}

/// A recommended swatch is a 1-based palette ordinal and replaces a custom RGB,
/// so the renderer cannot keep showing a colour the user no longer picked.
fn set_palette_color(adjustments: &mut AdjustmentSet, pf: u32, color: Option<usize>) {
    adjustments.extra.remove(&rgb_key(pf));
    match color {
        Some(index) => {
            adjustments.extra.insert(color_key(pf), index as f32);
        }
        None => {
            adjustments.extra.remove(&color_key(pf));
        }
    }
}

/// A typed colour is packed RGB24. Black (0) is stored as a key: absence, not
/// zero, is what means "no custom colour".
fn set_custom_rgb(adjustments: &mut AdjustmentSet, pf: u32, rgb24: u32) {
    adjustments.extra.remove(&color_key(pf));
    adjustments.extra.insert(rgb_key(pf), rgb24 as f32);
}

fn clear_color(adjustments: &mut AdjustmentSet, pf: u32) {
    adjustments.extra.remove(&color_key(pf));
    adjustments.extra.remove(&rgb_key(pf));
}

/// Drops the right brow's own choice so it follows the left brow again.
fn follow_left_brow(adjustments: &mut AdjustmentSet) {
    adjustments.extra.remove(&style_key(117));
    adjustments.extra.remove(&color_key(117));
    adjustments.extra.remove(&rgb_key(117));
}

fn pack_rgb([r, g, b, _]: [u8; 4]) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | b as u32
}

fn hex_label(rgb24: u32) -> String {
    format!("#{rgb24:06X}")
}

/// `#RRGGBB`, `RRGGBB`, `#RGB` or `RGB`; anything else is not a colour.
fn parse_hex_rgb(text: &str) -> Option<u32> {
    let digits = text.trim().trim_start_matches('#');
    let full: String = match digits.chars().count() {
        6 => digits.to_string(),
        3 => digits.chars().flat_map(|c| [c, c]).collect(),
        _ => return None,
    };
    if !full.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    u32::from_str_radix(&full, 16).ok()
}

/// The makeup row a recipe key belongs to: `pc:<pf>` plus a known channel.
///
/// `catchlight_*` is the eye-catchlight sub-material of PF111, so it belongs to
/// PF111 instead of inventing a PF of its own; it is replaced with the look.
fn suit_control_pf(key: &str) -> Option<u32> {
    let rest = key.strip_prefix("pc:")?;
    // `pc:100:` is not the base control; the engine rejects it too.
    if rest.ends_with(':') {
        return None;
    }
    let (id, kind) = rest.split_once(':').unwrap_or((rest, ""));
    let pf: u32 = id.parse().ok()?;
    if id != pf.to_string() || !SUIT_CONTROL_PFS.contains(&pf) {
        return None;
    }
    match kind {
        "" | "style" | "color" | "rgb" => Some(pf),
        // Only the three catchlight channels the engine actually consumes count as
        // this row's own material. `catchlight_*` used to accept any suffix, which
        // made unimplemented names look like consumed settings.
        "catchlight_style" | "catchlight_amount" | "catchlight_rgb" if pf == 111 => Some(pf),
        _ => None,
    }
}

/// A look counts as applied only when every control it sets has its value and no
/// other makeup control is left over from a different look.
fn controls_match(extra: &BTreeMap<String, f32>, effects: &BTreeMap<String, f32>) -> bool {
    effects.iter().all(|(key, value)| {
        extra
            .get(key)
            .is_some_and(|current| (current - value).abs() <= 1.0e-6)
    }) && extra
        .iter()
        .all(|(key, _)| suit_control_pf(key).is_none() || effects.contains_key(key))
}

// ───────────── Caches ─────────────

/// Whether a cached scan must be replaced. Empty results go stale first, so a
/// folder mounted after the app started appears without a restart; a full result
/// is kept long enough that painting the grid never scans.
fn cache_stale(is_empty: bool, age: Duration) -> bool {
    age >= if is_empty { EMPTY_TTL } else { FULL_TTL }
}

struct PresetCache {
    list: Arc<Vec<MakeupPresetInfo>>,
    at: Instant,
}

fn preset_cell() -> &'static Mutex<HashMap<u32, PresetCache>> {
    static CELL: OnceLock<Mutex<HashMap<u32, PresetCache>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Real material presets for `pf`. The scan happens outside the lock and only
/// when the cached entry is stale.
fn presets_for(pf: u32) -> Arc<Vec<MakeupPresetInfo>> {
    if let Ok(cell) = preset_cell().lock() {
        if let Some(hit) = cell.get(&pf) {
            if !cache_stale(hit.list.is_empty(), hit.at.elapsed()) {
                return Arc::clone(&hit.list);
            }
        }
    }
    let fresh = Arc::new(aurora_engine::makeup_presets_for_pf(pf));
    if let Ok(mut cell) = preset_cell().lock() {
        cell.insert(
            pf,
            PresetCache {
                list: Arc::clone(&fresh),
                at: Instant::now(),
            },
        );
    }
    fresh
}

/// Drops one PF's scan so the next read rescans — used when the user opens the
/// grid, which is an explicit "show me what exists now".
fn refresh_presets(pf: u32) {
    if let Ok(mut cell) = preset_cell().lock() {
        cell.remove(&pf);
    }
}

struct SuitCache {
    list: Arc<Vec<MakeupSuit>>,
    at: Instant,
}

fn suit_cell() -> &'static Mutex<Option<SuitCache>> {
    static CELL: OnceLock<Mutex<Option<SuitCache>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(None))
}

/// The product's own typed recipes: reading and parsing their JSON is cached, and
/// an empty discovery is retried after the short TTL so a late-mounted look set
/// appears without a restart.
fn suits_cached() -> Arc<Vec<MakeupSuit>> {
    if let Ok(cell) = suit_cell().lock() {
        if let Some(hit) = cell.as_ref() {
            if !cache_stale(hit.list.is_empty(), hit.at.elapsed()) {
                return Arc::clone(&hit.list);
            }
        }
    }
    let fresh = Arc::new(aurora_engine::makeup_suits());
    if let Ok(mut cell) = suit_cell().lock() {
        *cell = Some(SuitCache {
            list: Arc::clone(&fresh),
            at: Instant::now(),
        });
    }
    fresh
}

/// Whether a whole look can be applied here and now. The reason stays visible on
/// the chip when it cannot, instead of the look quietly disappearing.
fn suit_blocker(state: &AppState, suit: &MakeupSuit) -> Option<String> {
    if let Some(reason) = &suit.unavailable_reason {
        return Some(reason.clone());
    }
    suit.effects
        .keys()
        .filter_map(|key| suit_control_pf(key))
        .find_map(|pf| {
            let status = state.capability_status(pf);
            (!status.can_probe_selection()).then(|| format!("PF{pf}: {}", status.line()))
        })
}

// ───────────── One click, one edit ─────────────

/// The click shared by every style/colour control: re-check the image the cell
/// was drawn for, re-check the row, then write one undo step. `drawn` is the
/// `ImageId` captured while painting, so a click that lands after the user
/// switched photos is dropped instead of writing to the new photo.
fn commit(
    state: &Entity<AppState>,
    pf: u32,
    drawn: Option<ImageId>,
    label: &'static str,
    change: impl FnOnce(&mut AdjustmentSet) + 'static,
    cx: &mut App,
) {
    state.update(cx, |s, cx| {
        if s.library.current.is_none()
            || s.library.current != drawn
            || !s.capability_status(pf).can_probe_selection()
        {
            return;
        }
        let mut candidate = s.adj().clone();
        change(&mut candidate);
        s.precheck_then_edit_extra(label, candidate.extra, vec![pf], cx);
    });
}

// ───────────── Style / colour bar ─────────────

/// The style and colour controls for one makeup row, or `None` when the engine
/// has neither presets nor recommended colours for it, or when no image is open.
pub(super) fn style_bar(
    state: &Entity<AppState>,
    key: &'static str,
    cx: &App,
) -> Option<AnyElement> {
    let pf = pf_of(key)?;
    if state.read(cx).library.current.is_none() {
        return None;
    }
    let palette = aurora_engine::makeup_palette_for_pf(pf);
    let presets = presets_for(pf);
    // 没有推荐色、也没有第二个款式，但这一行真的有自定义 RGB 消费者时，
    // 仍然要给 hex 控件，否则材质行永远改不了颜色。
    if presets.len() <= 1 && palette.is_empty() && !aurora_engine::makeup_custom_color_supported(pf)
    {
        return None;
    }
    Some(
        MakeupStyleBar {
            state: state.clone(),
            pf,
            presets,
            palette,
        }
        .into_any_element(),
    )
}

#[derive(Default)]
struct BarUi {
    open: bool,
}

#[derive(IntoElement)]
struct MakeupStyleBar {
    state: Entity<AppState>,
    pf: u32,
    presets: Arc<Vec<MakeupPresetInfo>>,
    palette: &'static [(&'static str, [u8; 4])],
}

impl RenderOnce for MakeupStyleBar {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let p = pal(cx).clone();
        let pf = self.pf;
        let total = self.presets.len();
        let right_brow = pf == 117;

        let (available, drawn, style, own, own_color, color) = {
            let s = self.state.read(cx);
            let extra = &s.adj().extra;
            (
                s.capability_status(pf).can_probe_selection(),
                s.library.current,
                effective_style(extra, pf),
                own_style(extra, pf),
                ColorPick::of(extra, pf),
                effective_color(extra, pf),
            )
        };
        let follows = right_brow && own.is_none() && !own_color.is_set();

        let ui = window.use_keyed_state(SharedString::from(format!("mk-bar-{pf}")), cx, |_, _| {
            BarUi::default()
        });
        let open = ui.read(cx).open;

        let mut title: Vec<String> = Vec::new();
        if right_brow && own.is_none() {
            title.push(t("Follows the left brow", "跟随左眉").to_string());
        }
        match style {
            Some(index) => title.push(format!(
                "{} {:02}/{:02}",
                t("Style", "款式"),
                index,
                total.max(1)
            )),
            None => title.push(t("Default style", "默认款式").to_string()),
        }
        if let Some(name) = style
            .and_then(|index| self.presets.get(index - 1))
            .map(|preset| preset.id.as_str())
        {
            if !name.is_empty() {
                title.push(name.to_string());
            }
        }
        let title = title.join(" · ");

        let mut card = v_flex()
            .w_full()
            .mx_3()
            .mb_1p5()
            .rounded(px(4.0))
            .bg(p.well)
            .border_1()
            .border_color(if follows {
                p.accent.opacity(0.45)
            } else {
                p.edge
            })
            .when(!available, |d| d.opacity(0.55));

        let mut head = h_flex()
            .w_full()
            .items_center()
            .gap_1()
            .px(px(6.0))
            .pt(px(4.0))
            .pb(px(2.0))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .overflow_hidden()
                    .whitespace_nowrap()
                    .text_ellipsis()
                    .text_size(px(10.0))
                    .text_color(if available {
                        p.text_muted
                    } else {
                        p.text_disabled
                    })
                    .child(SharedString::from(title)),
            );

        if right_brow {
            let st_link = self.state.clone();
            let tip = if follows {
                t(
                    "The right brow uses the left brow's style and colour",
                    "右眉沿用左眉的款式与颜色",
                )
            } else {
                t(
                    "Let the right brow follow the left brow again",
                    "让右眉重新沿用左眉",
                )
            };
            head = head.child(
                icon_button(
                    SharedString::from(format!("mk-link-{pf}")),
                    if follows {
                        IconName::Link
                    } else {
                        IconName::Unlink
                    },
                    follows,
                    !follows && available,
                    tip,
                    act(move |_, cx| {
                        commit(
                            &st_link,
                            pf,
                            drawn,
                            t("Right brow follows the left", "右眉跟随左眉"),
                            follow_left_brow,
                            cx,
                        )
                    }),
                    &p,
                )
                .into_any_element(),
            );
        }

        if total > 1 {
            let (st_prev, st_next) = (self.state.clone(), self.state.clone());
            let (prev, next) = match style {
                Some(index) => (
                    if index <= 1 { total } else { index - 1 },
                    if index >= total { 1 } else { index + 1 },
                ),
                None => (total, 1),
            };
            head = head
                .child(
                    icon_button(
                        SharedString::from(format!("mk-prev-{pf}")),
                        IconName::ChevronLeft,
                        false,
                        available,
                        t("Previous style", "上一款式"),
                        act(move |_, cx| {
                            commit(
                                &st_prev,
                                pf,
                                drawn,
                                t("Switch makeup style", "切换妆容款式"),
                                move |a| set_style(a, pf, Some(prev)),
                                cx,
                            )
                        }),
                        &p,
                    )
                    .into_any_element(),
                )
                .child(
                    icon_button(
                        SharedString::from(format!("mk-next-{pf}")),
                        IconName::ChevronRight,
                        false,
                        available,
                        t("Next style", "下一款式"),
                        act(move |_, cx| {
                            commit(
                                &st_next,
                                pf,
                                drawn,
                                t("Switch makeup style", "切换妆容款式"),
                                move |a| set_style(a, pf, Some(next)),
                                cx,
                            )
                        }),
                        &p,
                    )
                    .into_any_element(),
                );
        }

        {
            let st_toggle = self.state.clone();
            let tip = if open {
                t("Hide all styles", "收起全部款式")
            } else {
                t("Show all styles", "展开全部款式")
            };
            head = head.child(
                icon_button(
                    SharedString::from(format!("mk-more-{pf}")),
                    if open {
                        IconName::ChevronUp
                    } else {
                        IconName::ChevronDown
                    },
                    open,
                    total > 0,
                    tip,
                    act(move |_, cx| {
                        // Opening is the explicit refresh: drop the cached scan once so
                        // a folder mounted since the last paint shows up immediately.
                        let opened = ui.update(cx, |bar, _| {
                            bar.open = !bar.open;
                            bar.open
                        });
                        if opened {
                            refresh_presets(pf);
                        }
                        st_toggle.update(cx, |_, cx| cx.notify());
                    }),
                    &p,
                )
                .into_any_element(),
            );
        }

        card = card.child(head);

        let mut body = v_flex().w_full().gap(px(5.0)).px(px(6.0)).pb(px(5.0));

        // 材质行（PF101 卧蚕、PF111 等）在一条预设都没装时也要给出明确状态：
        // 说明「没有可选款式」而不是整块静默消失或留一片空白。
        if self.presets.is_empty() {
            body = body.child(
                div()
                    .w_full()
                    .text_size(px(10.0))
                    .text_color(p.text_disabled)
                    .child(SharedString::from(t(
                        "No material preset installed for this row — the makeup asset folder has nothing to pick; colour and strength still apply where the renderer consumes them.",
                        "这一行还没有安装任何材质预设（妆容素材目录里没有可选款式）；引擎真正消费颜色与强度的地方仍然生效。",
                    ))),
            );
        }

        // 两条独立的事实：
        // * 推荐色板只画引擎真的给了的那几款（`makeup_palette_for_pf`），不编造。
        // * 自定义 RGB 只要这一行在引擎里真有颜色消费者就画（
        //   `makeup_custom_color_supported`）：材质行 PF107/111/115/116/90032 与
        //   卧蚕 PF101 把显式 RGB 接进同一张着色后材料，缺推荐色板不代表没有消费者。
        let custom_rgb = aurora_engine::makeup_custom_color_supported(pf);
        if !self.palette.is_empty() || custom_rgb {
            let mut swatches = h_flex().w_full().flex_wrap().gap(px(4.0)).items_center();
            for (index, (label, rgba)) in self.palette.iter().enumerate() {
                let ordinal = index + 1;
                let selected = color.rgb.is_none() && color.swatch == Some(ordinal);
                let st = self.state.clone();
                swatches = swatches.child(
                    div()
                        .id(SharedString::from(format!("mk-swatch-{pf}-{ordinal}")))
                        .size(px(SWATCH))
                        .rounded_full()
                        .bg(rgb(pack_rgb(*rgba)))
                        .border_1()
                        .border_color(if selected { p.text } else { p.edge })
                        .when(available, |d| {
                            d.cursor_pointer()
                                .hover(|s| s.border_color(p.text_muted))
                                .on_click(move |_, _, cx| {
                                    commit(
                                        &st,
                                        pf,
                                        drawn,
                                        t("Switch makeup colour", "切换妆容颜色"),
                                        move |a| set_palette_color(a, pf, Some(ordinal)),
                                        cx,
                                    )
                                })
                        })
                        .tooltip({
                            let tip = SharedString::from(*label);
                            move |window, cx| Tooltip::new(tip.clone()).build(window, cx)
                        }),
                );
            }
            if let Some(rgb24) = color.rgb {
                let st = self.state.clone();
                let inherited = own_color.rgb.is_none() && right_brow;
                let tip = if inherited {
                    format!(
                        "{}\n{}",
                        hex_label(rgb24),
                        t("Inherited from the left brow", "继承自左眉")
                    )
                } else {
                    hex_label(rgb24)
                };
                swatches = swatches.child(
                    div()
                        .id(SharedString::from(format!("mk-custom-{pf}")))
                        .size(px(SWATCH))
                        .rounded_full()
                        .bg(rgb(rgb24))
                        .border_1()
                        .border_color(p.accent)
                        // Only a colour stored on this row itself can be cleared here.
                        .when(available && own_color.rgb.is_some(), |d| {
                            d.cursor_pointer().on_click(move |_, _, cx| {
                                commit(
                                    &st,
                                    pf,
                                    drawn,
                                    t("Clear custom colour", "清除自定义颜色"),
                                    move |a| clear_color(a, pf),
                                    cx,
                                )
                            })
                        })
                        .tooltip(move |window, cx| {
                            Tooltip::new(SharedString::from(tip.clone())).build(window, cx)
                        }),
                );
            }
            body = body.child(swatches);

            let st_hex = self.state.clone();
            let initial = color.rgb.map(hex_label).unwrap_or_default();
            let input = field_state(
                SharedString::from(format!("mk-hex-{pf}")),
                initial,
                "#RRGGBB",
                move |text, _enter, cx| {
                    let Some(rgb24) = parse_hex_rgb(&text) else {
                        return;
                    };
                    commit(
                        &st_hex,
                        pf,
                        drawn,
                        t("Custom makeup colour", "自定义妆容颜色"),
                        move |a| set_custom_rgb(a, pf, rgb24),
                        cx,
                    );
                },
                window,
                cx,
            );
            body = body.child(
                h_flex()
                    .w_full()
                    .items_center()
                    .gap_1()
                    .child(
                        div()
                            .w(px(66.0))
                            .flex_shrink_0()
                            .child(Input::new(&input).xsmall()),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .overflow_hidden()
                            .whitespace_nowrap()
                            .text_ellipsis()
                            .text_size(px(9.5))
                            .text_color(p.text_subtle)
                            .child(SharedString::from(match color.rgb {
                                Some(rgb24) => hex_label(rgb24),
                                None => {
                                    t("Type a colour, or pick a swatch", "输入颜色，或点选推荐色")
                                        .to_string()
                                }
                            })),
                    ),
            );
        }

        if open && total > 0 {
            let mut cells: Vec<AnyElement> = Vec::with_capacity(total + 1);
            {
                let st = self.state.clone();
                cells.push(chip(
                    ChipSpec {
                        id: SharedString::from(format!("mk-chip-{pf}-default")),
                        caption: SharedString::from(if right_brow {
                            t("Follow the left brow", "跟随左眉")
                        } else {
                            t("Default", "默认")
                        }),
                        sub: None,
                        thumb: None,
                        placeholder: Some(if right_brow {
                            t("Uses the left brow", "沿用左眉")
                        } else {
                            t("Engine default", "引擎默认")
                        }),
                        selected: own.is_none(),
                        enabled: available,
                        tip: SharedString::from(if right_brow {
                            t(
                                "No style of its own: the left brow's style is used",
                                "不设独立款式：沿用左眉款式",
                            )
                        } else {
                            t(
                                "The engine's default material for this row",
                                "该行使用引擎默认材质",
                            )
                        }),
                        width: STYLE_CHIP_W,
                    },
                    act(move |_, cx| {
                        commit(
                            &st,
                            pf,
                            drawn,
                            t("Switch makeup style", "切换妆容款式"),
                            move |a| set_style(a, pf, None),
                            cx,
                        )
                    }),
                    &p,
                ));
            }
            for (index, preset) in self.presets.iter().enumerate() {
                let ordinal = index + 1;
                let st = self.state.clone();
                let mut tip = preset.path.display().to_string();
                if preset.mirror_x {
                    tip = format!(
                        "{tip}\n{}",
                        t("Mirrored horizontally for the right brow", "按右眉水平镜像")
                    );
                }
                cells.push(chip(
                    ChipSpec {
                        id: SharedString::from(format!("mk-chip-{pf}-{ordinal}")),
                        caption: SharedString::from(format!("{ordinal:02} {}", preset.id)),
                        sub: None,
                        thumb: Some(preset.path.clone()),
                        placeholder: None,
                        selected: own == Some(ordinal),
                        enabled: available,
                        tip: SharedString::from(tip),
                        width: STYLE_CHIP_W,
                    },
                    act(move |_, cx| {
                        commit(
                            &st,
                            pf,
                            drawn,
                            t("Switch makeup style", "切换妆容款式"),
                            move |a| set_style(a, pf, Some(ordinal)),
                            cx,
                        )
                    }),
                    &p,
                ));
            }
            body = body.child(
                div()
                    .id(SharedString::from(format!("mk-grid-{pf}")))
                    .w_full()
                    .max_h(px(178.0))
                    .overflow_y_scroll()
                    .child(h_flex().w_full().flex_wrap().gap_1().children(cells)),
            );
        }

        card.child(body)
    }
}

// ───────────── Shared chip glue ─────────────

/// Everything one material / look cell needs. Both the style grid and the look
/// board render through [`chip`], so border, selection, hover, tooltip and the
/// "inert but visible" rule exist once.
struct ChipSpec {
    id: SharedString,
    caption: SharedString,
    sub: Option<SharedString>,
    thumb: Option<PathBuf>,
    /// Shown instead of a thumbnail when the cell has no material of its own.
    placeholder: Option<&'static str>,
    selected: bool,
    enabled: bool,
    tip: SharedString,
    width: f32,
}

fn chip(spec: ChipSpec, act: Act, p: &Palette) -> AnyElement {
    let click = act.clone();
    let caption_color = if spec.selected {
        p.text
    } else if spec.enabled {
        p.text_muted
    } else {
        p.text_disabled
    };
    v_flex()
        .id(spec.id)
        .w(px(spec.width))
        .p(px(3.0))
        .gap(px(2.0))
        .rounded(px(4.0))
        .border_1()
        .border_color(if spec.selected { p.accent } else { p.edge })
        .bg(if spec.selected {
            p.accent_soft
        } else {
            p.raised
        })
        .when(spec.enabled, |d| {
            d.cursor_pointer()
                .hover(|s| s.bg(p.hover))
                .on_click(move |_, window, cx| (click)(window, cx))
        })
        .child(thumb(spec.thumb, spec.placeholder, p))
        .child(
            div()
                .w_full()
                .overflow_hidden()
                .whitespace_nowrap()
                .text_ellipsis()
                .text_size(px(9.5))
                .text_color(caption_color)
                .child(spec.caption),
        )
        .when_some(spec.sub, |d, sub| {
            d.child(
                div()
                    .w_full()
                    .overflow_hidden()
                    .whitespace_nowrap()
                    .text_ellipsis()
                    .text_size(px(9.0))
                    .text_color(p.text_subtle)
                    .child(sub),
            )
        })
        .tooltip(move |window, cx| Tooltip::new(spec.tip.clone()).build(window, cx))
        .into_any_element()
}

/// The real material: GPUI decodes the PNG asynchronously and keeps it in its own
/// image cache, so painting a grid decodes nothing on this thread.
fn thumb(path: Option<PathBuf>, placeholder: Option<&'static str>, p: &Palette) -> AnyElement {
    let frame = v_flex()
        .w_full()
        .h(px(THUMB_H))
        .rounded(px(3.0))
        .bg(p.well)
        .border_1()
        .border_color(p.divider)
        .overflow_hidden()
        .items_center()
        .justify_center();
    match path {
        Some(path) => frame
            .child(
                img(path)
                    .w_full()
                    .h(px(THUMB_H))
                    .object_fit(ObjectFit::Contain),
            )
            .into_any_element(),
        None => frame
            .child(
                div()
                    .px(px(2.0))
                    .text_size(px(9.0))
                    .text_color(p.text_subtle)
                    .child(SharedString::from(placeholder.unwrap_or_default())),
            )
            .into_any_element(),
    }
}

fn icon_button(
    id: SharedString,
    icon: IconName,
    active: bool,
    enabled: bool,
    tip: &'static str,
    act: Act,
    p: &Palette,
) -> impl IntoElement {
    let click = act.clone();
    div()
        .id(id)
        .size(px(18.0))
        .flex()
        .items_center()
        .justify_center()
        .rounded(px(3.0))
        .text_color(if active {
            p.accent
        } else if enabled {
            p.text_subtle
        } else {
            p.text_disabled
        })
        .when(enabled, |d| {
            d.cursor_pointer()
                .hover(|s| s.bg(p.pressed).text_color(p.text))
                .on_click(move |_, window, cx| (click)(window, cx))
        })
        .tooltip(move |window, cx| Tooltip::new(SharedString::from(tip)).build(window, cx))
        .child(Icon::new(icon).xsmall())
}

// ───────────── Whole-look picker ─────────────

/// One block of the product's own whole-look recipes, or `None` when there are
/// none. Disabled looks keep their reason on the chip; applying one is validated
/// on a copy of the current adjustments and committed with a single edit.
pub(super) fn suit_picker(state: &Entity<AppState>, cx: &App) -> Option<AnyElement> {
    let _ = state.read(cx);
    let suits = suits_cached();
    if suits.is_empty() {
        return None;
    }
    Some(
        SuitBoard {
            state: state.clone(),
            suits,
        }
        .into_any_element(),
    )
}

#[derive(IntoElement)]
struct SuitBoard {
    state: Entity<AppState>,
    suits: Arc<Vec<MakeupSuit>>,
}

/// The first material a look would actually put on the face, resolved through the
/// live catalogue (the same paths the renderer will use).
fn look_thumb(suit: &MakeupSuit) -> Option<PathBuf> {
    let mut rows: Vec<(u32, usize)> = suit
        .effects
        .iter()
        .filter_map(|(key, &value)| {
            if !key.ends_with(":style") {
                return None;
            }
            Some((suit_control_pf(key)?, ordinal(Some(value))?))
        })
        .collect();
    rows.sort_unstable();
    rows.dedup();
    rows.into_iter().find_map(|(pf, index)| {
        presets_for(pf)
            .get(index - 1)
            .map(|preset| preset.path.clone())
    })
}

impl RenderOnce for SuitBoard {
    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let p = pal(cx).clone();
        let (has_image, drawn, active, blocked) = {
            let s = self.state.read(cx);
            let extra = &s.adj().extra;
            let blocked: Vec<Option<String>> = self
                .suits
                .iter()
                .map(|suit| suit_blocker(s, suit))
                .collect();
            let active: Vec<bool> = self
                .suits
                .iter()
                .map(|suit| controls_match(extra, &suit.effects))
                .collect();
            (
                s.library.current.is_some(),
                s.library.current,
                active,
                blocked,
            )
        };

        let mut cells: Vec<AnyElement> = Vec::with_capacity(self.suits.len());
        for (index, suit) in self.suits.iter().enumerate() {
            let reason = blocked.get(index).cloned().flatten();
            let enabled = has_image && reason.is_none();
            let st = self.state.clone();
            let look = suit.clone();
            let count = suit.effects.len();
            let tip = match &reason {
                Some(reason) => SharedString::from(reason.clone()),
                None => {
                    SharedString::from(format!("{} · {count} {}", suit.id, t("controls", "项")))
                }
            };
            let sub = match &reason {
                Some(reason) => SharedString::from(reason.clone()),
                None => SharedString::from(format!("{count} {}", t("controls", "项"))),
            };
            cells.push(chip(
                ChipSpec {
                    id: SharedString::from(format!("suit-{index}")),
                    caption: SharedString::from(suit.label.clone()),
                    sub: Some(sub),
                    thumb: look_thumb(suit),
                    placeholder: Some(t("Whole look", "整妆")),
                    selected: active.get(index).copied().unwrap_or(false),
                    enabled,
                    tip,
                    width: SUIT_CHIP_W,
                },
                act(move |_, cx| {
                    let look = look.clone();
                    st.update(cx, |s, cx| {
                        if s.library.current.is_none() || s.library.current != drawn {
                            return;
                        }
                        if suit_blocker(s, &look).is_some() {
                            return;
                        }
                        // Validate once on a copy of what is on screen now, then run a
                        // single detail precheck of that candidate. Only an accepted
                        // candidate becomes one undo step; a refusal writes nothing and
                        // a result that arrives after the document moved is dropped.
                        let mut validated = s.adj().clone();
                        match aurora_engine::apply_makeup_suit(&look, &mut validated) {
                            Ok(()) => {
                                let required: Vec<u32> = look
                                    .effects
                                    .keys()
                                    .filter_map(|key| suit_control_pf(key))
                                    .collect();
                                s.precheck_then_edit_extra(
                                    t("Apply whole look", "应用整妆"),
                                    validated.extra,
                                    required,
                                    cx,
                                );
                            }
                            Err(reason) => s.notify(format!("{}: {reason}", look.label), cx),
                        }
                    });
                }),
                &p,
            ));
        }

        let help = SharedString::from(t(
            "The selected materials are checked on this photo before one whole-look edit.",
            "先检查所选材质与当前照片；全部可用时才写入整妆，可一次撤销。",
        ));
        let total = self.suits.len();

        v_flex()
            .w_full()
            .gap(px(6.0))
            .px_3()
            .pb_1p5()
            .child(
                h_flex()
                    .id("suit-picker-head")
                    .w_full()
                    .gap_1p5()
                    .items_center()
                    .text_size(px(10.0))
                    .text_color(p.text_subtle)
                    .child(Icon::new(IconName::Palette).xsmall())
                    .child(div().flex_1().min_w_0().child(SharedString::from(format!(
                        "{} · {total}",
                        t("Whole look", "整妆")
                    ))))
                    .child(
                        div()
                            .flex_shrink_0()
                            .child(SharedString::from(t("one undo step", "一次撤销"))),
                    )
                    .tooltip(move |window, cx| Tooltip::new(help.clone()).build(window, cx)),
            )
            .child(h_flex().w_full().flex_wrap().gap_1().children(cells))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extra(pairs: &[(&str, f32)]) -> BTreeMap<String, f32> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), *value))
            .collect()
    }

    #[test]
    fn hex_colours_parse_only_real_colours() {
        assert_eq!(parse_hex_rgb("#8A5A44"), Some(0x8A5A44));
        assert_eq!(parse_hex_rgb("8a5a44"), Some(0x8A5A44));
        assert_eq!(parse_hex_rgb("#f00"), Some(0xFF0000));
        assert_eq!(parse_hex_rgb("#000000"), Some(0));
        assert_eq!(parse_hex_rgb("#12345"), None);
        assert_eq!(parse_hex_rgb("#GGGGGG"), None);
        assert_eq!(parse_hex_rgb(""), None);
    }

    #[test]
    fn black_custom_colour_is_stored_because_absence_means_unset() {
        let mut adj = AdjustmentSet::default();
        set_custom_rgb(&mut adj, 100, 0);
        assert_eq!(adj.extra.get("pc:100:rgb"), Some(&0.0));
        assert_eq!(
            ColorPick::of(&adj.extra, 100),
            ColorPick {
                swatch: None,
                rgb: Some(0)
            }
        );
        clear_color(&mut adj, 100);
        assert_eq!(ColorPick::of(&adj.extra, 100), ColorPick::default());
    }

    #[test]
    fn picking_a_swatch_clears_the_custom_colour_and_the_other_way_round() {
        let mut adj = AdjustmentSet::default();
        set_custom_rgb(&mut adj, 106, 0xEA_7081);
        set_palette_color(&mut adj, 106, Some(2));
        assert_eq!(
            ColorPick::of(&adj.extra, 106),
            ColorPick {
                swatch: Some(2),
                rgb: None
            }
        );
        set_custom_rgb(&mut adj, 106, 0x00_0000);
        assert_eq!(
            ColorPick::of(&adj.extra, 106),
            ColorPick {
                swatch: None,
                rgb: Some(0)
            }
        );
    }

    #[test]
    fn right_brow_shows_the_left_choice_until_it_has_its_own() {
        let mut adj = AdjustmentSet::default();
        set_style(&mut adj, 100, Some(3));
        set_palette_color(&mut adj, 100, Some(4));
        assert_eq!(own_style(&adj.extra, 117), None);
        assert_eq!(effective_style(&adj.extra, 117), Some(3));
        // The two channels keep their own identity: swatch is an ordinal, rgb is packed.
        assert_eq!(
            effective_color(&adj.extra, 117),
            ColorPick {
                swatch: Some(4),
                rgb: None
            }
        );

        set_style(&mut adj, 117, Some(2));
        assert_eq!(effective_style(&adj.extra, 117), Some(2));
        follow_left_brow(&mut adj);
        assert_eq!(own_style(&adj.extra, 117), None);
        assert_eq!(effective_style(&adj.extra, 117), Some(3));
        assert_eq!(own_style(&adj.extra, 100), Some(3));
        assert_eq!(own_color(&adj.extra, 100), Some(4));
    }

    #[test]
    fn a_custom_rgb_on_the_left_brow_is_inherited_by_the_right_brow() {
        let mut adj = AdjustmentSet::default();
        set_custom_rgb(&mut adj, 100, 0x12_3456);
        assert_eq!(
            effective_color(&adj.extra, 117),
            ColorPick {
                swatch: None,
                rgb: Some(0x12_3456)
            }
        );
        set_palette_color(&mut adj, 117, Some(1));
        assert_eq!(
            effective_color(&adj.extra, 117),
            ColorPick {
                swatch: Some(1),
                rgb: None
            }
        );
    }

    #[test]
    fn upper_and_lower_lashes_never_share_a_key() {
        let mut adj = AdjustmentSet::default();
        set_style(&mut adj, 105, Some(2));
        assert_eq!(adj.extra.get("pc:105:style"), Some(&2.0));
        assert_eq!(adj.extra.get("pc:118:style"), None);
        assert_eq!(effective_style(&adj.extra, 118), None);
    }

    #[test]
    fn style_and_colour_ordinals_survive_a_round_trip() {
        let mut adj = AdjustmentSet::default();
        set_style(&mut adj, 103, Some(7));
        set_palette_color(&mut adj, 103, Some(1));
        assert_eq!(own_style(&adj.extra, 103), Some(7));
        assert_eq!(own_color(&adj.extra, 103), Some(1));
        set_style(&mut adj, 103, None);
        assert_eq!(own_style(&adj.extra, 103), None);
        assert_eq!(ordinal(Some(0.0)), None);
        assert_eq!(ordinal(Some(-2.0)), None);
        assert_eq!(ordinal(Some(3.0)), Some(3));
    }

    #[test]
    fn recipe_keys_are_limited_to_owned_makeup_controls() {
        assert_eq!(suit_control_pf("pc:100"), Some(100));
        assert_eq!(suit_control_pf("pc:90032:rgb"), Some(90032));
        assert_eq!(suit_control_pf("pc:117:style"), Some(117));
        assert_eq!(
            suit_control_pf("pc:108"),
            None,
            "teeth are not a makeup control"
        );
        assert_eq!(suit_control_pf("pc:100:left"), None, "no side channel");
        assert_eq!(suit_control_pf("pc:0100"), None);
        assert_eq!(suit_control_pf("pc:100:"), None);
        assert_eq!(suit_control_pf("100"), None);
    }

    #[test]
    fn the_eye_catchlight_names_only_the_three_channels_the_engine_consumes() {
        assert_eq!(suit_control_pf("pc:111:catchlight_style"), Some(111));
        assert_eq!(suit_control_pf("pc:111:catchlight_amount"), Some(111));
        assert_eq!(suit_control_pf("pc:111:catchlight_rgb"), Some(111));
        // Names nobody implements are not "already applied" evidence.
        assert_eq!(suit_control_pf("pc:111:catchlight_strength"), None);
        assert_eq!(suit_control_pf("pc:111:catchlight_mix"), None);
        assert_eq!(
            suit_control_pf("pc:111:catchlight_"),
            None,
            "a bare prefix names nothing"
        );
        assert_eq!(
            suit_control_pf("pc:100:catchlight_style"),
            None,
            "only PF111 owns it"
        );
    }

    #[test]
    fn a_look_counts_as_applied_only_when_its_controls_are_exactly_what_is_set() {
        let effects = BTreeMap::from([
            ("pc:100".to_owned(), 0.3_f32),
            ("pc:100:rgb".to_owned(), 0.0),
        ]);
        assert!(controls_match(
            &extra(&[("pc:100", 0.3), ("pc:100:rgb", 0.0)]),
            &effects
        ));
        assert!(!controls_match(&extra(&[("pc:100", 0.3)]), &effects));
        // A leftover makeup control from another look breaks the match.
        assert!(!controls_match(
            &extra(&[("pc:100", 0.3), ("pc:100:rgb", 0.0), ("pc:115", 0.4)]),
            &effects
        ));
        // Teeth, skin or face shape do not belong to the look.
        assert!(controls_match(
            &extra(&[("pc:100", 0.3), ("pc:100:rgb", 0.0), ("pc:108", 0.5)]),
            &effects
        ));
    }

    #[test]
    fn a_look_that_carries_the_catchlight_counts_it_as_its_own_control() {
        let effects = BTreeMap::from([
            ("pc:111".to_owned(), 0.2_f32),
            ("pc:111:catchlight_amount".to_owned(), 0.5),
        ]);
        assert!(controls_match(
            &extra(&[("pc:111", 0.2), ("pc:111:catchlight_amount", 0.5)]),
            &effects
        ));
        // A catchlight left over from another look is a leftover, not a neutral key.
        assert!(!controls_match(&extra(&[("pc:111", 0.2)]), &effects));
    }

    #[test]
    fn empty_scans_go_stale_sooner_than_full_ones() {
        assert!(!cache_stale(true, Duration::from_millis(500)));
        assert!(cache_stale(true, EMPTY_TTL));
        assert!(!cache_stale(false, Duration::from_secs(3)));
        assert!(cache_stale(false, FULL_TTL));
    }
}
