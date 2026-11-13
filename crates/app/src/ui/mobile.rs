//! The phone and tablet shell: a library grid and an editing screen with the
//! image on top and everything that is touched in the lower half, under the
//! thumb. One parameter at a time sits on a thick slider, chosen from a row of
//! cards; a bar at the bottom switches between adjusting, styles, cropping,
//! rating and the full desktop tool panels in a sheet. State, viewer and tool
//! panels are the desktop workspace's own.

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, Sizable as _,
    button::{Button, ButtonVariants as _},
    h_flex, h_virtual_list,
    menu::{DropdownMenu as _, PopupMenuItem},
    v_flex, v_virtual_list,
};
use gpui_kit::{
    AnyElement, App, AppContext as _, BackGesturePhase, Context, Entity, FocusHandle, InteractiveElement, IntoElement,
    ObjectFit, ParentElement, PathPromptOptions, Render, SharedString, StatefulInteractiveElement, Styled,
    StyledImage as _, Subscription, Window, div, img, prelude::FluentBuilder as _, px, size,
};
use lumen_engine::{AspectRatio, AutoTarget, ColorTag, ImageId, NoiseReduction, RenderQuality, Sharpening};

use super::inspector;
use super::viewer::Viewer;
use crate::actions::Import;
use crate::i18n::{self, t};
use crate::motion;
use crate::state::{AppState, Binding, CompareMode, CursorTool, Scope, StateEvent, ToolTab};
use crate::theme::{Palette, hex, pal};
use crate::widgets::IconBtn;
use crate::widgets::param_c1::{ParamSpec, param_spec};
use crate::{global, scoped};
use crate::widgets::surface::Surface;

const TOP_BAR: f32 = 60.0;
const TAB_BAR: f32 = 58.0;
const TAB_W: f32 = 68.0;
const FILM: f32 = 72.0;
const HANDLE: f32 = 22.0;
/// Smallest finger target.
const TAP: f32 = 44.0;
/// Height of the mode bar at the bottom.
const NAV: f32 = 66.0;
/// Height of the big slider's track, which is also its knob.
const SLIDER: f32 = 52.0;
const CHIP_W: f32 = 76.0;
const CHIP_H: f32 = 74.0;
/// Space kept clear at the left and right edges.
const GUTTER: f32 = 14.0;
/// From this width on, a landscape window puts the tools beside the viewer.
const WIDE: f32 = 640.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Screen {
    Library,
    Edit,
}

/// Where the tool sheet rests.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Detent {
    Closed,
    Half,
    Full,
}

pub struct MobileShell {
    state: Entity<AppState>,
    viewer: Entity<Viewer>,
    screen: Screen,
    detent: Detent,
    /// The sheet's height while its handle is dragged.
    drag_h: Option<f32>,
    /// Heights of the half and full detents and the window y of the sheet's
    /// bottom edge, from the last layout.
    sheet: (f32, f32, f32),
    mode: Mode,
    /// The parameter on the big slider.
    knob: Knob,
    /// Where on the track a slider drag began, and the value it began from.
    slide: Option<(f32, f32)>,
    /// The original is shown while its button is held.
    original: bool,
    focus: FocusHandle,
    notice_serial: u64,
    _subs: Vec<Subscription>,
}

impl MobileShell {
    pub fn new(state: Entity<AppState>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let viewer = cx.new(|cx| Viewer::new(&state, cx));
        let focus = cx.focus_handle();
        window.focus(&focus, cx);
        cx.set_global(super::WorkspaceFocus(focus.clone()));
        state.update(cx, |s, cx| {
            // A phone has no room for the info overlay or several images at once.
            s.welcome = false;
            s.view.show_info_overlay = false;
            s.view.multi = false;
            if !s.visible_tabs().contains(&s.tool_tab) || matches!(s.tool_tab, ToolTab::Library) {
                s.tool_tab = ToolTab::Exposure;
            }
            cx.notify();
        });
        let subs = vec![
            cx.observe(&state, |_, _, cx| cx.notify()),
            cx.subscribe(&state, |this: &mut Self, _, event, cx| {
                if matches!(event, StateEvent::Notice(_)) {
                    this.notice_serial += 1;
                    cx.notify();
                }
            }),
        ];
        Self {
            state,
            viewer,
            screen: Screen::Library,
            detent: Detent::Half,
            drag_h: None,
            sheet: (0.0, 0.0, 0.0),
            mode: Mode::Adjust,
            knob: Knob::Exposure,
            slide: None,
            original: false,
            focus,
            notice_serial: 0,
            _subs: subs,
        }
    }

    /// The system back action: fold the professional tools' sheet, then leave the editor.
    fn back(&mut self, cx: &mut Context<Self>) {
        match self.screen {
            Screen::Edit if self.mode == Mode::Pro && self.detent != Detent::Closed => self.detent = Detent::Closed,
            Screen::Edit => self.leave_editor(cx),
            Screen::Library => {}
        }
        cx.notify();
    }

    fn open_image(&mut self, id: ImageId, cx: &mut Context<Self>) {
        self.state.update(cx, |s, cx| s.select(id, false, false, cx));
        self.screen = Screen::Edit;
        cx.notify();
    }

    fn pick_tab(&mut self, tab: ToolTab, cx: &mut Context<Self>) {
        let same = self.state.read(cx).tool_tab == tab;
        self.detent = match (same, self.detent) {
            (true, Detent::Closed) | (false, Detent::Closed) => Detent::Half,
            (true, _) => Detent::Closed,
            (false, d) => d,
        };
        self.state.update(cx, |s, cx| {
            s.tool_tab = tab;
            cx.notify();
        });
        cx.notify();
    }

    fn import(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: Some(t("Import", "导入").into()),
        });
        let state = self.state.clone();
        cx.spawn(async move |_, cx| {
            let Ok(Ok(Some(paths))) = rx.await else { return };
            cx.update(|cx| state.update(cx, |s, cx| s.import(paths, cx)));
        })
        .detach();
    }

    fn ids(&self, cx: &App) -> Rc<Vec<ImageId>> {
        Rc::new(self.state.read(cx).visible_images().iter().map(|i| i.id).collect())
    }

    fn top_bar(&self, lead: AnyElement, title: SharedString, note: SharedString, trail: AnyElement, p: &Palette) -> impl IntoElement + use<> {
        h_flex()
            .h(px(TOP_BAR))
            .flex_shrink_0()
            .px_1()
            .gap_1()
            .bg(p.chrome)
            .border_b_1()
            .border_color(p.divider)
            .child(lead)
            .child(
                v_flex()
                    .flex_1()
                    .min_w_0()
                    .child(div().text_sm().font_weight(gpui_kit::FontWeight::SEMIBOLD).text_color(p.text).truncate().child(title))
                    .child(div().text_xs().text_color(p.text_subtle).truncate().child(note)),
            )
            .child(trail)
    }

    fn library(&mut self, width: f32, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let ids = self.ids(cx);
        let count = ids.len();
        let title: SharedString = i18n::builtin(self.state.read(cx).active_collection_name.as_ref()).into();
        let note: SharedString = format!("{count} {}", t("images", "张图像")).into();
        let this = cx.entity();
        let import = IconBtn::new("m-import", IconName::ImagePlus, t("Import", "导入")).size(TAP).on_click({
            let this = this.clone();
            move |_, cx| this.update(cx, |s, cx| s.import(cx))
        });

        let gap = 3.0;
        let per_row = ((width / 124.0).floor() as usize).max(3);
        let cell = (width - gap * (per_row as f32 + 1.0)) / per_row as f32;
        let rows = count.div_ceil(per_row);
        let sizes = Rc::new(vec![size(px(width), px(cell + gap)); rows]);
        let grid = v_virtual_list(this.clone(), "m-grid", sizes, move |shell, range, _, cx| {
            range
                .map(|r| {
                    let start = r * per_row;
                    let end = (start + per_row).min(ids.len());
                    h_flex().pl(px(gap)).gap(px(gap)).h(px(cell + gap)).items_start().children((start..end).map(|i| shell.thumb(ids[i], cell, false, cx)))
                })
                .collect()
        })
        .pt(px(gap))
        .size_full();

        v_flex()
            .size_full()
            .child(self.top_bar(div().w(px(10.0)).into_any_element(), title, note, import.into_any_element(), &p))
            .child(div().flex_1().min_h_0().bg(p.well).map(|d| {
                if count == 0 {
                    d.flex().flex_col().items_center().justify_center().gap_3().text_color(p.text_subtle).child(Icon::new(IconName::Images).size(px(40.0))).child(t("No images here yet", "这里还没有图像")).child(
                        Button::new("m-import-empty").primary().label(t("Import Images", "导入图像")).on_click({
                            let this = this.clone();
                            move |_, _, cx| this.update(cx, |s, cx| s.import(cx))
                        }),
                    )
                } else {
                    d.child(grid)
                }
            }))
            .into_any_element()
    }

    /// One square thumbnail; `strip` marks the current image for the filmstrip.
    fn thumb(&self, id: ImageId, side: f32, strip: bool, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        let image = s.thumbs.get(&id).cloned();
        let failed = image.is_none() && s.thumb_failed.contains(&id);
        let info = s.info(id);
        let (rating, tag, edited) = info.map_or((0, lumen_engine::ColorTag::None, false), |i| (i.rating, i.color_tag, i.has_adjustments));
        let current = strip && s.current == Some(id);
        let this = cx.entity();
        div()
            .id(SharedString::from(format!("m-thumb-{}-{}", if strip { "s" } else { "g" }, id.0)))
            .relative()
            .size(px(side))
            .flex_shrink_0()
            .rounded(px(3.0))
            .overflow_hidden()
            .bg(p.raised)
            .cursor_pointer()
            .on_click(move |_, _, cx| this.update(cx, |shell, cx| shell.open_image(id, cx)))
            .when_some(image, |d, image| d.child(img(image).size_full().object_fit(ObjectFit::Cover)))
            .when(failed, |d| d.flex().items_center().justify_center().text_color(p.text_disabled).child(Icon::new(IconName::TriangleAlert).small()))
            .when(!strip && (rating != 0 || edited || tag != lumen_engine::ColorTag::None), |d| {
                d.child(
                    h_flex()
                        .absolute()
                        .left_0()
                        .right_0()
                        .bottom_0()
                        .h(px(18.0))
                        .px_1()
                        .gap_1()
                        .bg(gpui_kit::black().opacity(0.45))
                        .text_color(p.warning)
                        .text_size(px(10.0))
                        .child(if rating < 0 { "✕".to_string() } else { "★".repeat(rating as usize) })
                        .child(div().flex_1())
                        .when(edited, |d| d.child(Icon::new(IconName::SlidersHorizontal).size(px(10.0)).text_color(p.text)))
                        .when(tag != lumen_engine::ColorTag::None, |d| d.child(div().size(px(8.0)).rounded_full().bg(hex(tag.rgb())))),
                )
            })
            .when(current, |d| d.child(div().absolute().top_0().left_0().size_full().rounded(px(3.0)).border_2().border_color(p.accent)))
            .into_any_element()
    }

    /// A round button of the top bar.
    fn round(&self, id: &'static str, icon: IconName, enabled: bool, p: &Palette, f: impl Fn(&mut Self, &mut Context<Self>) + 'static, cx: &mut Context<Self>) -> AnyElement {
        let this = cx.entity();
        div()
            .id(id)
            .size(px(TAP))
            .flex_shrink_0()
            .flex()
            .items_center()
            .justify_center()
            .rounded_full()
            .bg(p.raised)
            .text_color(if enabled { p.text } else { p.text_disabled })
            .when(enabled, |d| d.cursor_pointer().active(|s| s.bg(p.pressed)).on_click(move |_, _, cx| this.update(cx, |s, cx| f(s, cx))))
            .child(Icon::new(icon).size(px(20.0)))
            .into_any_element()
    }

    /// Close, undo and redo on the left; export and the menu on the right.
    fn edit_bar(&self, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        let has_image = s.current.is_some();
        let title: SharedString = s.current_info().map(|i| i.file_name.clone()).unwrap_or_default().into();
        let (can_undo, can_redo) = (s.can_undo(), s.can_redo());
        let (warning, has_clip, split) = (s.view.exposure_warning, s.clipboard.is_some(), s.view.compare == CompareMode::Split);
        let st = self.state.clone();
        let more = Button::new("m-more").ghost().icon(Icon::new(IconName::Ellipsis)).w(px(TAP)).h(px(TAP)).rounded(px(TAP / 2.0)).bg(p.raised).dropdown_menu(move |menu, _, _| {
            let item = |label: &'static str, f: fn(&mut AppState, &mut Context<AppState>)| {
                let st = st.clone();
                PopupMenuItem::new(label).on_click(move |_, _, cx| st.update(cx, |s, cx| f(s, cx)))
            };
            menu.item(item(t("Auto Adjust", "自动调整"), |s, cx| s.auto_adjust(AutoTarget::All, cx)).disabled(!has_image))
                .item(item(t("Copy Adjustments", "拷贝调整"), |s, cx| s.copy_adjustments(cx)).disabled(!has_image))
                .item(item(t("Apply Adjustments", "应用调整"), |s, cx| s.apply_adjustments(cx)).disabled(!has_clip))
                .item(item(t("New Variant", "新建变体"), |s, cx| s.new_variant(false, cx)).disabled(!has_image))
                .item(item(t("Reset All", "全部复位"), |s, cx| s.reset_all(cx)).disabled(!has_image))
                .separator()
                .item(item(t("Fit to Screen", "适合屏幕"), |s, cx| s.set_zoom(None, None, cx)))
                .item(item("100%", |s, cx| {
                    let (sf, c) = (s.scale_factor, s.zoom.center);
                    s.set_zoom(Some(sf), Some(c), cx)
                }))
                .item(
                    item(t("Before / After Split", "前后对比分屏"), |s, cx| {
                        s.view.compare = if s.view.compare == CompareMode::Split { CompareMode::Off } else { CompareMode::Split };
                        s.preview.before_dirty = true;
                        s.request_render(RenderQuality::Final, cx);
                        cx.notify();
                    })
                    .checked(split),
                )
                .item(
                    item(t("Exposure Warning", "曝光警告"), |s, cx| {
                        s.view.exposure_warning = !s.view.exposure_warning;
                        s.request_render(RenderQuality::Final, cx);
                        cx.notify();
                    })
                    .checked(warning),
                )
        });
        let st = self.state.clone();
        h_flex()
            .h(px(TOP_BAR))
            .flex_shrink_0()
            .px(px(GUTTER))
            .gap_2()
            .child(self.round("m-close", IconName::X, true, &p, |s, cx| s.leave_editor(cx), cx))
            .child(self.round("m-undo", IconName::Undo2, can_undo, &p, |s, cx| s.state.update(cx, |s, cx| s.undo(cx)), cx))
            .child(self.round("m-redo", IconName::Redo2, can_redo, &p, |s, cx| s.state.update(cx, |s, cx| s.redo(cx)), cx))
            .child(div().flex_1().min_w_0().text_xs().text_color(p.text_subtle).text_center().truncate().child(title))
            .child(
                div()
                    .id("m-export")
                    .h(px(TAP))
                    .px(px(18.0))
                    .flex_shrink_0()
                    .flex()
                    .items_center()
                    .rounded_full()
                    .bg(p.raised)
                    .text_size(px(15.0))
                    .font_weight(gpui_kit::FontWeight::SEMIBOLD)
                    .text_color(if has_image { p.text } else { p.text_disabled })
                    .when(has_image, |d| d.cursor_pointer().active(|s| s.bg(p.pressed)).on_click(move |_, _, cx| st.update(cx, |s, cx| s.process_selection(cx))))
                    .child(t("Export", "导出")),
            )
            .child(more)
            .into_any_element()
    }

    /// A card in a scrolling row: a parameter, a crop action, a ratio or a style.
    fn chip(&self, id: SharedString, icon: Option<IconName>, label: SharedString, note: Option<SharedString>, selected: bool, dot: bool, p: &Palette, f: impl Fn(&mut Self, &mut Context<Self>) + 'static, cx: &mut Context<Self>) -> AnyElement {
        let this = cx.entity();
        v_flex()
            .id(id)
            .relative()
            .min_w(px(CHIP_W))
            .h(px(CHIP_H))
            .px_2()
            .flex_shrink_0()
            .items_center()
            .justify_center()
            .gap(px(6.0))
            .rounded(px(16.0))
            .bg(p.raised)
            .border_2()
            .border_color(if selected { p.accent } else { gpui_kit::transparent_black() })
            .text_color(if selected { p.text } else { p.text_muted })
            .cursor_pointer()
            .active(|s| s.bg(p.pressed))
            .on_click(move |_, _, cx| this.update(cx, |s, cx| f(s, cx)))
            .when_some(icon, |d, icon| d.child(Icon::new(icon).size(px(24.0))))
            .child(div().text_size(px(13.0)).child(label))
            .when_some(note, |d, note| d.child(div().text_size(px(10.0)).text_color(p.text_subtle).child(note)))
            // The parameter differs from its default.
            .when(dot, |d| d.child(div().absolute().top(px(7.0)).right(px(7.0)).size(px(6.0)).rounded_full().bg(p.accent)))
            .into_any_element()
    }

    fn chip_row(&self, id: &'static str, chips: Vec<AnyElement>) -> AnyElement {
        h_flex().id(id).h(px(CHIP_H + 16.0)).flex_shrink_0().overflow_x_scroll().child(h_flex().flex_shrink_0().px(px(GUTTER)).gap_2().children(chips)).into_any_element()
    }

    /// One parameter on a thick track: the label and value above, and the
    /// hold-to-see-the-original button beside it. Dragging moves the value by
    /// the distance travelled, so a touch never makes it jump; a double tap
    /// puts the default back.
    fn big_slider(&self, knob: Knob, width: f32, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let scope = self.state.read(cx).scope();
        let (spec, binding) = knob.spec(&self.state, scope, cx);
        let has_image = self.state.read(cx).current.is_some();
        let value = (binding.get)(self.state.read(cx).adj());
        let (min, max, default) = (spec.min, spec.max, spec.default.clamp(spec.min, spec.max));
        let span = (max - min).max(1e-6);
        let track_w = (width - GUTTER * 2.0 - SLIDER - 10.0).max(SLIDER * 2.0);
        let (r, travel) = (SLIDER / 2.0, track_w - SLIDER);
        let x = |v: f32| r + ((v - min) / span).clamp(0.0, 1.0) * travel;
        let (xk, x0) = (x(value), x(default));
        let modified = (value - default).abs() > span * 1e-4;
        let text = {
            let sign = if min < 0.0 && value > 0.0 { "+" } else { "" };
            format!("{sign}{value:.*}{}", spec.decimals, spec.unit)
        };
        let label = knob.label();
        let step = spec.step;
        let (down, drag, up, reset) = (cx.entity(), cx.entity(), cx.entity(), self.state.clone());
        let (set, set_reset) = (binding.set.clone(), binding.set.clone());
        let state = self.state.clone();
        let track = Surface::new("m-slider")
            .relative()
            .h(px(SLIDER))
            .w(px(track_w))
            .flex_shrink_0()
            .rounded_full()
            .bg(p.well)
            .when(has_image, |d| {
                d.on_down(move |ptr, _, cx| {
                    if ptr.clicks >= 2 {
                        let set = set_reset.clone();
                        reset.update(cx, |s, cx| s.edit(label, move |a| set(a, default), cx));
                    } else {
                        down.update(cx, |s, _| s.slide = Some((ptr.pos.0, value)));
                    }
                })
                .on_drag(move |ptr, _, cx| {
                    let Some((start, from)) = drag.read(cx).slide else { return };
                    let mut v = (from + (ptr.pos.0 - start) * track_w / travel * span).clamp(min, max);
                    if step > 0.0 {
                        v = ((v / step).round() * step).clamp(min, max);
                    }
                    // The default holds the knob for a moment, so it is easy to return to.
                    if (v - default).abs() < span * 0.012 {
                        v = default;
                    }
                    let set = set.clone();
                    state.update(cx, |s, cx| s.edit_live(label, move |a| set(a, v), cx));
                })
                .on_up(move |_, cx| {
                    up.update(cx, |s, cx| {
                        s.slide = None;
                        s.state.update(cx, |s, cx| s.end_gesture(cx));
                    })
                })
            })
            .when(modified, |d| {
                d.child(div().absolute().top_0().left(px(xk.min(x0) - r)).w(px((xk - x0).abs() + SLIDER)).h_full().rounded_full().bg(p.accent))
                    .child(div().absolute().top(px(r - 4.0)).left(px(x0 - 4.0)).size(px(8.0)).rounded_full().bg(p.on_accent.opacity(0.55)))
            })
            .child(
                div()
                    .absolute()
                    .top(px(4.0))
                    .left(px(xk - r + 4.0))
                    .size(px(SLIDER - 8.0))
                    .rounded_full()
                    .bg(if has_image { hex(0xffffff) } else { p.text_disabled })
                    .border_4()
                    .border_color(if modified { hex(0xffffff) } else { p.accent })
                    .shadow_md(),
            );
        let st = self.state.clone();
        let set = binding.set.clone();
        let (hold, release) = (cx.entity(), cx.entity());
        v_flex()
            .flex_shrink_0()
            .child(
                h_flex()
                    .h(px(40.0))
                    .px(px(GUTTER + 4.0))
                    .gap_3()
                    .child(div().text_size(px(15.0)).font_weight(gpui_kit::FontWeight::SEMIBOLD).text_color(p.text).child(label))
                    .when(modified, |d| {
                        d.child(
                            div()
                                .id("m-slider-reset")
                                .h(px(28.0))
                                .px_3()
                                .flex()
                                .items_center()
                                .rounded_full()
                                .bg(p.raised)
                                .text_size(px(12.0))
                                .text_color(p.text_muted)
                                .cursor_pointer()
                                .on_click(move |_, _, cx| {
                                    let set = set.clone();
                                    st.update(cx, |s, cx| s.edit(label, move |a| set(a, default), cx))
                                })
                                .child(t("Reset", "复位")),
                        )
                    })
                    .child(div().flex_1())
                    .child(div().text_size(px(17.0)).font_weight(gpui_kit::FontWeight::MEDIUM).text_color(if modified { p.accent } else { p.text_muted }).child(text)),
            )
            .child(
                h_flex().px(px(GUTTER)).gap(px(10.0)).child(track).child(
                    div()
                        .id("m-original")
                        .size(px(SLIDER))
                        .flex_shrink_0()
                        .flex()
                        .items_center()
                        .justify_center()
                        .rounded_full()
                        .bg(if self.original { p.accent_soft } else { p.raised })
                        .text_color(if self.original { p.accent } else if has_image { p.text_muted } else { p.text_disabled })
                        .when(has_image, |d| {
                            d.on_mouse_down(gpui_kit::MouseButton::Left, move |_, _, cx| hold.update(cx, |s, cx| s.show_original(true, cx)))
                                .on_mouse_up(gpui_kit::MouseButton::Left, move |_, _, cx| release.update(cx, |s, cx| s.show_original(false, cx)))
                        })
                        .child(Icon::new(IconName::Columns2).size(px(22.0))),
                ),
            )
            .into_any_element()
    }

    /// While the button is held the viewer shows the image without its
    /// adjustments, keeping the crop so the two line up.
    fn show_original(&mut self, on: bool, cx: &mut Context<Self>) {
        if self.original == on {
            return;
        }
        self.original = on;
        self.state.update(cx, |s, cx| {
            s.hover_look = on.then(|| lumen_engine::AdjustmentSet { geometry: s.adj().geometry.clone(), ..Default::default() });
            s.request_render(RenderQuality::Final, cx);
            cx.notify();
        });
        cx.notify();
    }

    fn adjust(&self, width: f32, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let scope = self.state.read(cx).scope();
        let group = self.knob.group();
        let chips = Knob::ALL
            .iter()
            .filter(|k| k.group() == Some(group.unwrap_or(Group::Tone)))
            .map(|&k| {
                let (spec, binding) = k.spec(&self.state, scope, cx);
                let value = (binding.get)(self.state.read(cx).adj());
                let dot = (value - spec.default).abs() > (spec.max - spec.min).abs() * 1e-4;
                self.chip(format!("m-knob-{k:?}").into(), Some(k.icon()), k.label().into(), None, k == self.knob, dot, &p, move |s, cx| {
                    s.knob = k;
                    cx.notify();
                }, cx)
            })
            .collect();
        let this = cx.entity();
        v_flex()
            .flex_shrink_0()
            .child(self.big_slider(self.knob, width, cx))
            .child(h_flex().h(px(TAP + 8.0)).justify_center().gap_2().children(Group::ALL.map(|g| {
                let this = this.clone();
                let selected = Some(g) == group;
                div()
                    .id(SharedString::from(format!("m-group-{g:?}")))
                    .h(px(TAP))
                    .px_4()
                    .flex()
                    .items_center()
                    .text_size(px(15.0))
                    .when(selected, |d| d.font_weight(gpui_kit::FontWeight::SEMIBOLD))
                    .text_color(if selected { p.text } else { p.text_subtle })
                    .cursor_pointer()
                    .on_click(move |_, _, cx| {
                        this.update(cx, |s, cx| {
                            // Each group opens on its first parameter.
                            if s.knob.group() != Some(g) {
                                s.knob = Knob::ALL.into_iter().find(|k| k.group() == Some(g)).unwrap_or(Knob::Exposure);
                            }
                            cx.notify();
                        })
                    })
                    .child(g.label())
            })))
            .child(self.chip_row("m-knobs", chips))
            .into_any_element()
    }

    fn styles(&self, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let styles = self.state.read(cx).styles.clone();
        if styles.is_empty() {
            return div().h(px(CHIP_H + 16.0)).flex().items_center().justify_center().text_size(px(14.0)).text_color(p.text_subtle).child(t("No styles available", "没有可用的风格")).into_any_element();
        }
        let chips = styles
            .into_iter()
            .enumerate()
            .map(|(i, style)| {
                let (name, group): (SharedString, SharedString) = (style.name.clone().into(), i18n::builtin(&style.group).into());
                self.chip(format!("m-style-{i}").into(), None, name, Some(group), false, false, &p, move |s, cx| s.state.update(cx, |s, cx| s.apply_style(&style, false, cx)), cx)
            })
            .collect();
        v_flex()
            .flex_shrink_0()
            .child(div().h(px(40.0)).px(px(GUTTER + 4.0)).flex().items_center().text_size(px(13.0)).text_color(p.text_subtle).child(t("Tap a style to apply it · undo takes it back", "点按即可套用风格 · 可撤销")))
            .child(self.chip_row("m-styles", chips))
            .into_any_element()
    }

    /// Straighten on the slider; turns, flips and ratios on the cards. The
    /// crop frame itself is dragged on the image.
    fn crop(&self, width: f32, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        let g = s.adj().geometry.clone();
        let original = s.current_info().map_or(1.5, |i| {
            let (w, h) = if g.orientation % 2 == 1 { (i.height, i.width) } else { (i.width, i.height) };
            w as f32 / (h as f32).max(1.0)
        });
        let turn = |id: &'static str, icon: IconName, label: &'static str, f: fn(&mut lumen_engine::Geometry), cx: &mut Context<Self>| {
            self.chip(id.into(), Some(icon), label.into(), None, false, false, &p, move |s, cx| s.state.update(cx, |s, cx| s.edit(label, move |a| f(&mut a.geometry), cx)), cx)
        };
        let mut chips = vec![
            turn("m-rot-l", IconName::RotateCcwSquare, t("Left", "向左"), |g| g.orientation = (g.orientation + 3) % 4, cx),
            turn("m-rot-r", IconName::RotateCwSquare, t("Right", "向右"), |g| g.orientation = (g.orientation + 1) % 4, cx),
            turn("m-flip-h", IconName::FlipHorizontal2, t("Mirror", "镜像"), |g| g.flip_horizontal = !g.flip_horizontal, cx),
            turn("m-flip-v", IconName::FlipVertical2, t("Flip", "翻转"), |g| g.flip_vertical = !g.flip_vertical, cx),
            div().w(px(1.0)).h(px(36.0)).mx_1().flex_shrink_0().bg(p.edge).into_any_element(),
        ];
        chips.extend(AspectRatio::ALL.into_iter().map(|a| {
            let name = match a {
                AspectRatio::Unconstrained => t("Free", "自由"),
                AspectRatio::Original => t("Original", "原始"),
                a => crate::tools::lens::aspect_name(a),
            };
            self.chip(format!("m-ratio-{a:?}").into(), None, name.into(), None, g.aspect == a, false, &p, move |s, cx| {
                s.state.update(cx, |s, cx| s.edit(t("Crop ratio", "裁剪比例"), move |adj| crate::tools::lens::set_aspect(adj, a, original), cx))
            }, cx)
        }));
        chips.push(self.chip("m-crop-reset".into(), Some(IconName::RotateCcw), t("Reset", "复位").into(), None, false, false, &p, |s, cx| {
            s.state.update(cx, |s, cx| {
                s.edit(t("Reset crop", "复位裁剪"), |a| a.geometry = lumen_engine::Geometry { keystone: a.geometry.keystone.clone(), ..Default::default() }, cx)
            })
        }, cx));
        v_flex().flex_shrink_0().child(self.big_slider(Knob::Rotation, width, cx)).child(div().h(px(8.0))).child(self.chip_row("m-crop", chips)).into_any_element()
    }

    /// Stars, reject, color tags and the filmstrip, for sorting through a shoot.
    fn rate(&self, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        let (rating, tag) = s.current_info().map_or((0, ColorTag::None), |i| (i.rating, i.color_tag));
        let has_image = s.current.is_some();
        let st = self.state.clone();
        let step = move |id: &'static str, icon: IconName, tip: &'static str, delta: isize| {
            let st = st.clone();
            IconBtn::new(id, icon, tip).size(TAP + 4.0).disabled(!has_image).on_click(move |_, cx| st.update(cx, |s, cx| s.step(delta, cx)))
        };
        let (st, st_tag) = (self.state.clone(), self.state.clone());
        v_flex()
            .flex_shrink_0()
            .child(
                h_flex()
                    .h(px(56.0))
                    .px_2()
                    .child(step("m-prev", IconName::ChevronLeft, t("Previous", "上一张"), -1))
                    .child(div().flex_1())
                    .children((1..=5i8).map(|n| {
                        let st = st.clone();
                        div()
                            .id(SharedString::from(format!("m-star-{n}")))
                            .size(px(TAP + 4.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_size(px(28.0))
                            .text_color(if n <= rating { p.warning } else { p.text_disabled })
                            // Tapping the current rating clears it.
                            .when(has_image, |d| d.cursor_pointer().on_click(move |_, _, cx| st.update(cx, |s, cx| s.set_rating(if rating == n { 0 } else { n }, cx))))
                            .child("★")
                    }))
                    .child({
                        let st = self.state.clone();
                        IconBtn::new("m-reject", IconName::X, t("Reject", "排除")).size(TAP + 4.0).active(rating < 0).disabled(!has_image).on_click(move |_, cx| {
                            st.update(cx, |s, cx| s.set_rating(if rating < 0 { 0 } else { -1 }, cx))
                        })
                    })
                    .child(div().flex_1())
                    .child(step("m-next", IconName::ChevronRight, t("Next", "下一张"), 1)),
            )
            .child(h_flex().h(px(TAP + 4.0)).justify_center().children(ColorTag::ALL.into_iter().map(|c| {
                let st = st_tag.clone();
                let none = c == ColorTag::None;
                div()
                    .id(SharedString::from(format!("m-tag-{c:?}")))
                    .size(px(TAP))
                    .flex()
                    .items_center()
                    .justify_center()
                    .when(has_image, |d| d.cursor_pointer().on_click(move |_, _, cx| st.update(cx, |s, cx| s.set_color_tag(c, cx))))
                    .child(
                        div()
                            .size(px(26.0))
                            .rounded_full()
                            .when(!none, |d| d.bg(hex(c.rgb())))
                            .border_2()
                            .border_color(if c == tag { p.text } else if none { p.text_disabled } else { gpui_kit::transparent_black() }),
                    )
            })))
            .child(self.filmstrip(cx))
            .into_any_element()
    }

    fn filmstrip(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let ids = self.ids(cx);
        let side = FILM - 8.0;
        let sizes = Rc::new(vec![size(px(side + 4.0), px(side)); ids.len()]);
        div().h(px(FILM)).flex_shrink_0().child(
            h_virtual_list(cx.entity(), "m-film", sizes, move |shell, range, _, cx| {
                range.map(|i| div().pr(px(4.0)).child(shell.thumb(ids[i], side, true, cx))).collect()
            })
            .px(px(GUTTER))
            .py(px(4.0))
            .size_full(),
        )
    }

    /// The five ways to work on an image, under the thumb.
    fn nav(&self, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let this = cx.entity();
        h_flex()
            .h(px(NAV))
            .flex_shrink_0()
            .px_1()
            .children(Mode::ALL.map(|m| {
                let this = this.clone();
                let selected = m == self.mode;
                div().id(SharedString::from(format!("m-mode-{m:?}"))).flex_1().h_full().py(px(4.0)).px(px(3.0)).cursor_pointer().on_click(move |_, _, cx| this.update(cx, |s, cx| s.set_mode(m, cx))).child(
                    v_flex()
                        .size_full()
                        .items_center()
                        .justify_center()
                        .gap(px(3.0))
                        .rounded(px(18.0))
                        .when(selected, |d| d.bg(p.raised))
                        .text_color(if selected { p.text } else { p.text_subtle })
                        .child(Icon::new(m.icon()).size(px(22.0)))
                        .child(div().text_size(px(11.0)).child(m.label())),
                )
            }))
            .into_any_element()
    }

    fn set_mode(&mut self, mode: Mode, cx: &mut Context<Self>) {
        // Tapping the professional tools again folds their sheet away.
        if mode == Mode::Pro && self.mode == Mode::Pro {
            self.detent = if self.detent == Detent::Closed { Detent::Half } else { Detent::Closed };
        } else if mode == Mode::Pro {
            self.detent = Detent::Half;
        }
        self.mode = mode;
        self.state.update(cx, |s, cx| {
            // The crop frame shows only while cropping.
            let tool = if mode == Mode::Crop { CursorTool::Crop } else { CursorTool::Select };
            if (s.cursor == CursorTool::Crop) != (mode == Mode::Crop) && s.current.is_some() {
                s.set_cursor(tool, cx);
            }
        });
        cx.notify();
    }

    fn leave_editor(&mut self, cx: &mut Context<Self>) {
        self.set_mode(Mode::Adjust, cx);
        self.screen = Screen::Library;
        cx.notify();
    }

    fn tab_bar(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        let current = s.tool_tab;
        let tabs = s.visible_tabs();
        // Beside the image the panels are always open; `sheet` is zero there.
        let open = self.detent != Detent::Closed || self.sheet.2 == 0.0;
        let index = tabs.iter().position(|t| *t == current).unwrap_or(0) as f32;
        let x = motion::value(motion::key("m-tabs", "indicator"), index * TAB_W, motion::GLIDE, window, cx);
        let lit = motion::toggle(motion::key("m-tabs", "lit"), open, motion::QUICK, window, cx);
        let this = cx.entity();
        h_flex().id("m-tabs").h(px(TAB_BAR)).flex_shrink_0().bg(p.chrome).overflow_x_scroll().child(
            h_flex()
                .relative()
                .h_full()
                .flex_shrink_0()
                .child(div().absolute().left(px(x + 6.0)).top(px(5.0)).w(px(TAB_W - 12.0)).h(px(TAB_BAR - 10.0)).rounded(px(8.0)).bg(p.accent_soft.opacity(p.accent_soft.a * lit)))
                .children(tabs.into_iter().map(|tab| {
                    let selected = tab == current;
                    let this = this.clone();
                    v_flex()
                        .id(SharedString::from(format!("m-tab-{tab:?}")))
                        .relative()
                        .w(px(TAB_W))
                        .h_full()
                        .flex_shrink_0()
                        .items_center()
                        .justify_center()
                        .gap(px(3.0))
                        .cursor_pointer()
                        .text_color(if selected && open { p.accent } else if selected { p.text } else { p.text_subtle })
                        .on_click(move |_, _, cx| this.update(cx, |s, cx| s.pick_tab(tab, cx)))
                        .child(Icon::new(tab.icon()).size(px(21.0)))
                        .child(div().text_size(px(10.0)).child(tab.label()))
                })),
        )
    }

    /// The tool panels with a handle to drag between closed, half and full.
    fn sheet_el(&mut self, half: f32, full: f32, window: &mut Window, cx: &mut Context<Self>) -> Option<AnyElement> {
        let p = pal(cx).clone();
        let target = self.drag_h.unwrap_or(match self.detent {
            Detent::Closed => 0.0,
            Detent::Half => half,
            Detent::Full => full,
        });
        let h = motion::follow(motion::key("m-sheet", "h"), target, motion::PANEL, self.drag_h.is_none(), window, cx);
        if h < 1.0 {
            return None;
        }
        let (this, up) = (cx.entity(), cx.entity());
        let handle = Surface::new("m-sheet-handle")
            .h(px(HANDLE))
            .w_full()
            .flex_shrink_0()
            .flex()
            .items_center()
            .justify_center()
            .cursor_ns_resize()
            .bg(p.chrome)
            .child(div().w(px(40.0)).h(px(4.0)).rounded_full().bg(p.text_disabled))
            .on_drag(move |ptr, _, cx| {
                this.update(cx, |s, cx| {
                    let (_, full, bottom) = s.sheet;
                    s.drag_h = Some((bottom - f32::from(ptr.point.y) - HANDLE / 2.0).clamp(0.0, full));
                    cx.notify();
                })
            })
            .on_up(move |_, cx| {
                up.update(cx, |s, cx| {
                    let (half, full, _) = s.sheet;
                    // Let go: rest at the nearest detent.
                    if let Some(h) = s.drag_h.take() {
                        s.detent = if h < half * 0.5 { Detent::Closed } else if h < (half + full) / 2.0 { Detent::Half } else { Detent::Full };
                    }
                    cx.notify();
                })
            });
        Some(
            v_flex()
                .h(px(h))
                .flex_shrink_0()
                .overflow_hidden()
                .border_t_1()
                .border_color(p.divider)
                .child(handle)
                // The content keeps its resting height while the sheet moves, so it slides instead of reflowing.
                .child(div().h(px((h.max(half) - HANDLE).max(0.0))).flex_shrink_0().child(inspector::inspector(&self.state, window, cx)))
                .into_any_element(),
        )
    }

    fn viewer_el(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let p = pal(cx).clone();
        let notice = self.state.read(cx).notice.clone().filter(|_| self.notice_serial > 0);
        let focus = self.focus.clone();
        div()
            .relative()
            .flex_1()
            .min_h_0()
            .min_w_0()
            // Touching the image leaves any text field, which puts the keyboard away.
            .child(div().absolute().size_full().capture_any_mouse_down(move |_, window, cx| window.focus(&focus, cx)).child(self.viewer.clone()))
            .when_some(notice, |d, n| d.child(super::workspace::notice_hud(n, self.notice_serial, &p)))
    }

    /// The controls of every mode but the professional tools, `width` wide.
    fn controls(&self, width: f32, cx: &mut Context<Self>) -> AnyElement {
        match self.mode {
            Mode::Adjust | Mode::Pro => self.adjust(width, cx),
            Mode::Style => self.styles(cx),
            Mode::Crop => self.crop(width, cx),
            Mode::Rate => self.rate(cx),
        }
    }

    fn editor(&mut self, width: f32, height: f32, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let wide = width > height && width >= WIDE;
        let pro = self.mode == Mode::Pro;
        let bar = self.edit_bar(cx);
        if wide {
            // Landscape: the controls sit beside the image, under the right thumb.
            self.sheet = (0.0, 0.0, 0.0);
            let side = (width * 0.42).clamp(320.0, 440.0);
            return v_flex()
                .size_full()
                .child(bar)
                .child(
                    h_flex().flex_1().min_h_0().child(v_flex().flex_1().h_full().min_w_0().child(self.viewer_el(cx))).child(
                        v_flex()
                            .w(px(side))
                            .h_full()
                            .flex_shrink_0()
                            .map(|d| {
                                if pro {
                                    d.child(div().flex_1().min_h_0().border_l_1().border_color(p.divider).child(inspector::inspector(&self.state, window, cx))).child(self.tab_bar(window, cx))
                                } else {
                                    d.child(div().flex_1().min_h_0()).child(self.controls(side, cx))
                                }
                            })
                            .child(self.nav(cx)),
                    ),
                )
                .into_any_element();
        }
        if !pro {
            self.sheet = (0.0, 0.0, 0.0);
            return v_flex().size_full().child(bar).child(self.viewer_el(cx)).child(self.controls(width, cx)).child(self.nav(cx)).into_any_element();
        }
        let room = (height - TOP_BAR - TAB_BAR - NAV).max(0.0);
        let half = (room * 0.5).clamp(200.0, 440.0).min(room);
        let full = (room * 0.82).max(half);
        self.sheet = (half, full, height.max(1.0) - TAB_BAR - NAV);
        let sheet = self.sheet_el(half, full, window, cx);
        v_flex().size_full().child(bar).child(self.viewer_el(cx)).children(sheet).child(self.tab_bar(window, cx)).child(self.nav(cx)).into_any_element()
    }
}

/// The ways to work on an image, in the bar at the bottom.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Adjust,
    Style,
    Crop,
    Rate,
    /// Every desktop tool panel, in a sheet.
    Pro,
}

impl Mode {
    const ALL: [Mode; 5] = [Mode::Adjust, Mode::Style, Mode::Crop, Mode::Rate, Mode::Pro];

    fn label(self) -> &'static str {
        match self {
            Mode::Adjust => t("Adjust", "调节"),
            Mode::Style => t("Styles", "风格"),
            Mode::Crop => t("Crop", "裁切"),
            Mode::Rate => t("Rate", "评级"),
            Mode::Pro => t("Pro", "专业"),
        }
    }

    fn icon(self) -> IconName {
        match self {
            Mode::Adjust => IconName::SlidersHorizontal,
            Mode::Style => IconName::WandSparkles,
            Mode::Crop => IconName::Crop,
            Mode::Rate => IconName::Star,
            Mode::Pro => IconName::Layers,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Group {
    Tone,
    Color,
    Detail,
}

impl Group {
    const ALL: [Group; 3] = [Group::Tone, Group::Color, Group::Detail];

    fn label(self) -> &'static str {
        match self {
            Group::Tone => t("Tone", "影调"),
            Group::Color => t("Color", "颜色"),
            Group::Detail => t("Detail", "细节"),
        }
    }
}

/// A parameter the big slider can hold.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Knob {
    Exposure,
    Contrast,
    Brightness,
    Highlights,
    Shadows,
    Whites,
    Blacks,
    Clarity,
    Dehaze,
    Vignette,
    Kelvin,
    Tint,
    Saturation,
    Structure,
    Sharpen,
    Noise,
    ColorNoise,
    Grain,
    /// Straightening, on the crop screen.
    Rotation,
}

impl Knob {
    const ALL: [Knob; 19] = [
        Knob::Exposure, Knob::Contrast, Knob::Brightness, Knob::Highlights, Knob::Shadows, Knob::Whites, Knob::Blacks, Knob::Clarity, Knob::Dehaze, Knob::Vignette,
        Knob::Kelvin, Knob::Tint, Knob::Saturation,
        Knob::Structure, Knob::Sharpen, Knob::Noise, Knob::ColorNoise, Knob::Grain,
        Knob::Rotation,
    ];

    fn group(self) -> Option<Group> {
        use Knob::*;
        match self {
            Exposure | Contrast | Brightness | Highlights | Shadows | Whites | Blacks | Clarity | Dehaze | Vignette => Some(Group::Tone),
            Kelvin | Tint | Saturation => Some(Group::Color),
            Structure | Sharpen | Noise | ColorNoise | Grain => Some(Group::Detail),
            Rotation => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Knob::Exposure => t("Exposure", "曝光"),
            Knob::Contrast => t("Contrast", "对比度"),
            Knob::Brightness => t("Brightness", "亮度"),
            Knob::Highlights => t("Highlights", "高光"),
            Knob::Shadows => t("Shadows", "阴影"),
            Knob::Whites => t("Whites", "白色"),
            Knob::Blacks => t("Blacks", "黑色"),
            Knob::Clarity => t("Clarity", "清晰度"),
            Knob::Dehaze => t("Dehaze", "去雾"),
            Knob::Vignette => t("Vignette", "暗角"),
            Knob::Kelvin => t("Temperature", "色温"),
            Knob::Tint => t("Tint", "色调"),
            Knob::Saturation => t("Saturation", "饱和度"),
            Knob::Structure => t("Structure", "结构"),
            Knob::Sharpen => t("Sharpen", "锐化"),
            Knob::Noise => t("Denoise", "降噪"),
            Knob::ColorNoise => t("Color Noise", "彩噪"),
            Knob::Grain => t("Grain", "颗粒"),
            Knob::Rotation => t("Straighten", "拉直"),
        }
    }

    fn icon(self) -> IconName {
        match self {
            Knob::Exposure => IconName::Sun,
            Knob::Contrast => IconName::Contrast,
            Knob::Brightness => IconName::SunMedium,
            Knob::Highlights => IconName::Sunrise,
            Knob::Shadows => IconName::Moon,
            Knob::Whites => IconName::Circle,
            Knob::Blacks => IconName::CircleDot,
            Knob::Clarity => IconName::Diamond,
            Knob::Dehaze => IconName::CloudFog,
            Knob::Vignette => IconName::Aperture,
            Knob::Kelvin => IconName::Thermometer,
            Knob::Tint => IconName::Pipette,
            Knob::Saturation => IconName::Palette,
            Knob::Structure => IconName::Grid3x3,
            Knob::Sharpen => IconName::Triangle,
            Knob::Noise => IconName::AudioLines,
            Knob::ColorNoise => IconName::Droplets,
            Knob::Grain => IconName::Sparkles,
            Knob::Rotation => IconName::Ruler,
        }
    }

    /// The same contract key and binding the desktop tool panels use.
    fn spec(self, state: &Entity<AppState>, scope: Scope, cx: &App) -> (ParamSpec, Binding) {
        use crate::tools::{color::wb_binding, details::opt_binding};
        let sharpening = |field: fn(&mut Sharpening) -> &mut f32, read: fn(&Sharpening) -> f32| opt_binding(scope, |a| &mut a.sharpening, |l| &mut l.sharpening, |a| &a.sharpening, |l| &l.sharpening, field, read);
        let noise = |field: fn(&mut NoiseReduction) -> &mut f32, read: fn(&NoiseReduction) -> f32| opt_binding(scope, |a| &mut a.noise_reduction, |l| &mut l.noise_reduction, |a| &a.noise_reduction, |l| &l.noise_reduction, field, read);
        let (key, binding) = match self {
            Knob::Exposure => ("Exposure", scoped!(scope, exposure.exposure)),
            Knob::Contrast => ("Contrast", scoped!(scope, exposure.contrast)),
            Knob::Brightness => ("Brightness", scoped!(scope, exposure.brightness)),
            Knob::Saturation => ("Saturation", scoped!(scope, exposure.saturation)),
            Knob::Highlights => ("HighlightRecovery", scoped!(scope, hdr.highlights)),
            Knob::Shadows => ("ShadowRecovery", scoped!(scope, hdr.shadows)),
            Knob::Whites => ("WhiteRecovery", scoped!(scope, hdr.whites)),
            Knob::Blacks => ("BlackRecovery", scoped!(scope, hdr.blacks)),
            Knob::Clarity => ("Clarity", scoped!(scope, clarity.clarity)),
            Knob::Structure => ("ClarityStructure", scoped!(scope, clarity.structure)),
            Knob::Dehaze => ("DehazeAmount", scoped!(scope, dehaze.amount)),
            Knob::Vignette => ("LensVignettingAmount", global!(vignetting.amount)),
            Knob::Kelvin => ("WhiteBalanceTemperature", wb_binding(scope, true)),
            Knob::Tint => ("WhiteBalanceTint", wb_binding(scope, false)),
            Knob::Sharpen => ("USMAmount", sharpening(|s| &mut s.amount, |s| s.amount)),
            Knob::Noise => ("NoiseReductionAmount", noise(|n| &mut n.luminance, |n| n.luminance)),
            Knob::ColorNoise => ("ColorNoiseReductionAmount", noise(|n| &mut n.color, |n| n.color)),
            Knob::Grain => ("FilmGrainAmount", global!(film_grain.impact)),
            Knob::Rotation => ("RotationAdjustment", global!(geometry.rotation)),
        };
        let mut spec = param_spec(state, key, &binding, cx).unwrap_or(ParamSpec { label: self.label().into(), min: 0.0, max: 100.0, default: 0.0, decimals: 0, unit: "", step: 0.0 });
        // `AdjustmentSet` keeps these two in -100..100, as the desktop panel narrows them.
        if matches!(self, Knob::Clarity | Knob::Structure) {
            (spec.min, spec.max, spec.default) = (-100.0, 100.0, 0.0);
        }
        (spec, binding)
    }
}

impl Render for MobileShell {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let p = pal(cx).clone();
        let viewport = window.viewport_size();
        // Status bar, cutouts, gesture bar and the keyboard stay clear of the interface.
        let inset = window.insets().effective();
        let (top, bottom, left, right) = (f32::from(inset.top), f32::from(inset.bottom), f32::from(inset.left), f32::from(inset.right));
        let width = (f32::from(viewport.width) - left - right).max(1.0);
        let height = (f32::from(viewport.height) - top - bottom).max(1.0);
        let body = match self.screen {
            Screen::Library => self.library(width, cx),
            Screen::Edit => {
                let el = self.editor(width, height, window, cx);
                // The handle reports window coordinates.
                self.sheet.2 += if self.sheet.2 > 0.0 { top } else { 0.0 };
                el
            }
        };
        let state = self.state.clone();
        div()
            .id("mobile-shell")
            .size_full()
            .pt(px(top))
            .pb(px(bottom))
            .pl(px(left))
            .pr(px(right))
            .bg(p.chrome)
            .text_color(p.text)
            .track_focus(&self.focus)
            .capture_any_mouse_up(cx.listener(move |this, _, _, cx| {
                // Wherever the finger lifts, the original goes away again.
                this.show_original(false, cx);
                state.update(cx, |s, cx| s.end_gesture(cx));
            }))
            .on_back_gesture(cx.listener(|this, e: &gpui_kit::BackGestureEvent, _, cx| {
                if e.phase == BackGesturePhase::Committed {
                    this.back(cx)
                }
            }))
            .on_action(cx.listener(|this, _: &Import, _, cx| this.import(cx)))
            .child(motion::enter(if self.screen == Screen::Library { "m-library" } else { "m-editor" }, div().size_full().child(body), 6.0))
    }
}

/// Opens the touch shell. On a phone the platform makes the window the screen;
/// on a desktop it is a phone-sized window for trying the layout.
pub fn open(state: Entity<AppState>, cx: &mut App) {
    let options = gpui_kit::WindowOptions {
        window_bounds: (!gpui_kit::is_mobile()).then(|| gpui_kit::WindowBounds::centered(size(px(412.0), px(892.0)), cx)),
        window_min_size: Some(size(px(320.0), px(480.0))),
        ..Default::default()
    };
    gpui_kit::open_window(options, cx, move |window, cx| {
        window.set_window_title("Aurora Studio");
        cx.new(|cx| MobileShell::new(state, window, cx))
    })
    .expect("failed to open the main window");
}
