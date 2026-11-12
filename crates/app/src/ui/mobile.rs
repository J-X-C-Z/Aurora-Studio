//! The phone and tablet shell: a library grid and an editing screen with the
//! viewer on top and the tool panels in a sheet that slides up from the tab
//! bar. It shows the same state, viewer and tool panels as the desktop
//! workspace, laid out for one hand and sized for fingers.

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
use lumen_engine::{AutoTarget, ImageId};

use super::inspector;
use super::viewer::Viewer;
use crate::actions::Import;
use crate::i18n::{self, t};
use crate::motion;
use crate::state::{AppState, CompareMode, StateEvent, ToolTab};
use crate::theme::{Palette, hex, pal};
use crate::widgets::IconBtn;
use crate::widgets::surface::Surface;

const TOP_BAR: f32 = 48.0;
const TAB_BAR: f32 = 58.0;
const TAB_W: f32 = 68.0;
const RATE_ROW: f32 = 44.0;
const FILM: f32 = 68.0;
const HANDLE: f32 = 22.0;
/// Smallest finger target.
const TAP: f32 = 44.0;
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
    film: bool,
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
            film: true,
            focus,
            notice_serial: 0,
            _subs: subs,
        }
    }

    /// The system back action: close the sheet, then leave the editor.
    fn back(&mut self, cx: &mut Context<Self>) {
        match (self.screen, self.detent) {
            (Screen::Edit, Detent::Half | Detent::Full) if self.drag_h.is_none() => self.detent = Detent::Closed,
            (Screen::Edit, _) => self.screen = Screen::Library,
            (Screen::Library, _) => {}
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
            .when(current, |d| d.child(div().absolute().size_full().rounded(px(3.0)).border_2().border_color(p.accent)))
            .into_any_element()
    }

    fn edit_bar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        let has_image = s.current.is_some();
        let title: SharedString = s.current_info().map(|i| i.file_name.clone()).unwrap_or_else(|| t("No image", "没有图像").to_string()).into();
        let ids = s.visible_images();
        let note: SharedString = match s.current.and_then(|id| ids.iter().position(|i| i.id == id)) {
            Some(i) => format!("{} / {}", i + 1, ids.len()).into(),
            None => SharedString::default(),
        };
        let (can_undo, can_redo) = (s.can_undo(), s.can_redo());
        let comparing = s.view.compare == CompareMode::Split;
        let (warning, has_clip, film) = (s.view.exposure_warning, s.clipboard.is_some(), self.film);
        let this = cx.entity();
        let st = self.state.clone();
        let act = move |id: &'static str, icon: IconName, tip: &'static str, f: fn(&mut AppState, &mut Context<AppState>)| {
            let st = st.clone();
            IconBtn::new(id, icon, tip).size(TAP).on_click(move |_, cx| st.update(cx, |s, cx| f(s, cx)))
        };
        let back = IconBtn::new("m-back", IconName::ChevronLeft, t("Library", "图库")).size(TAP).on_click({
            let this = this.clone();
            move |_, cx| {
                this.update(cx, |s, cx| {
                    s.screen = Screen::Library;
                    cx.notify();
                })
            }
        });
        let st = self.state.clone();
        let more = Button::new("m-more").ghost().icon(Icon::new(IconName::Ellipsis)).w(px(TAP)).h(px(TAP)).dropdown_menu(move |menu, _, _| {
            let item = |label: &'static str, f: fn(&mut AppState, &mut Context<AppState>)| {
                let st = st.clone();
                PopupMenuItem::new(label).on_click(move |_, _, cx| st.update(cx, |s, cx| f(s, cx)))
            };
            let this = this.clone();
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
                    item(t("Exposure Warning", "曝光警告"), |s, cx| {
                        s.view.exposure_warning = !s.view.exposure_warning;
                        s.request_render(lumen_engine::RenderQuality::Final, cx);
                        cx.notify();
                    })
                    .checked(warning),
                )
                .item(PopupMenuItem::new(t("Filmstrip", "胶片条")).checked(film).on_click(move |_, _, cx| {
                    this.update(cx, |s, cx| {
                        s.film = !s.film;
                        cx.notify();
                    })
                }))
                .separator()
                .item(item(t("Process", "处理输出"), |s, cx| s.process_selection(cx)).disabled(!has_image))
        });
        let trail = h_flex()
            .flex_shrink_0()
            .child(act("m-undo", IconName::Undo2, t("Undo", "撤销"), |s, cx| s.undo(cx)).disabled(!can_undo))
            .child(act("m-redo", IconName::Redo2, t("Redo", "重做"), |s, cx| s.redo(cx)).disabled(!can_redo))
            .child(
                act("m-compare", IconName::Columns2, t("Before / After", "调整前后"), |s, cx| {
                    s.view.compare = if s.view.compare == CompareMode::Split { CompareMode::Off } else { CompareMode::Split };
                    s.preview.before_dirty = true;
                    s.request_render(lumen_engine::RenderQuality::Final, cx);
                    cx.notify();
                })
                .active(comparing)
                .disabled(!has_image),
            )
            .child(more);
        self.top_bar(back.into_any_element(), title, note, trail.into_any_element(), &p)
    }

    /// Stars, reject and stepping through the images.
    fn rate_row(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        let rating = s.current_info().map_or(0, |i| i.rating);
        let has_image = s.current.is_some();
        let pct = (s.effective_scale() / s.scale_factor.max(0.5) * 100.0).round();
        let zoom: SharedString = if s.zoom.scale.is_none() { t("Fit", "适合").into() } else { format!("{pct:.0}%").into() };
        let st = self.state.clone();
        let step = move |id: &'static str, icon: IconName, tip: &'static str, delta: isize| {
            let st = st.clone();
            IconBtn::new(id, icon, tip).size(TAP).disabled(!has_image).on_click(move |_, cx| st.update(cx, |s, cx| s.step(delta, cx)))
        };
        let st = self.state.clone();
        h_flex()
            .h(px(RATE_ROW))
            .flex_shrink_0()
            .px_1()
            .bg(p.chrome)
            .border_t_1()
            .border_color(p.divider)
            .child(step("m-prev", IconName::ChevronLeft, t("Previous", "上一张"), -1))
            .child(div().flex_1())
            .children((1..=5i8).map(|n| {
                let st = st.clone();
                div()
                    .id(SharedString::from(format!("m-star-{n}")))
                    .w(px(34.0))
                    .h(px(TAP))
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_size(px(20.0))
                    .text_color(if n <= rating { p.warning } else { p.text_disabled })
                    .when(has_image, |d| {
                        // Tapping the current rating clears it.
                        d.cursor_pointer().on_click(move |_, _, cx| st.update(cx, |s, cx| s.set_rating(if rating == n { 0 } else { n }, cx)))
                    })
                    .child("★")
            }))
            .child({
                let st = self.state.clone();
                IconBtn::new("m-reject", IconName::X, t("Reject", "排除")).size(TAP).active(rating < 0).disabled(!has_image).on_click(move |_, cx| {
                    st.update(cx, |s, cx| s.set_rating(if rating < 0 { 0 } else { -1 }, cx))
                })
            })
            .child(div().flex_1().flex().justify_end().pr_1().text_xs().text_color(p.text_subtle).child(zoom))
            .child(step("m-next", IconName::ChevronRight, t("Next", "下一张"), 1))
    }

    fn filmstrip(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let p = pal(cx).clone();
        let ids = self.ids(cx);
        let side = FILM - 8.0;
        let sizes = Rc::new(vec![size(px(side + 4.0), px(side)); ids.len()]);
        div().h(px(FILM)).flex_shrink_0().bg(p.well).child(
            h_virtual_list(cx.entity(), "m-film", sizes, move |shell, range, _, cx| {
                range.map(|i| div().pr(px(4.0)).child(shell.thumb(ids[i], side, true, cx))).collect()
            })
            .p(px(4.0))
            .size_full(),
        )
    }

    fn tab_bar(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        let current = s.tool_tab;
        let tabs = s.visible_tabs();
        let open = self.detent != Detent::Closed || self.sheet.2 == 0.0;
        let index = tabs.iter().position(|t| *t == current).unwrap_or(0) as f32;
        let x = motion::value(motion::key("m-tabs", "indicator"), index * TAB_W, motion::GLIDE, window, cx);
        let lit = motion::toggle(motion::key("m-tabs", "lit"), open, motion::QUICK, window, cx);
        let this = cx.entity();
        h_flex().id("m-tabs").h(px(TAB_BAR)).flex_shrink_0().bg(p.chrome).border_t_1().border_color(p.divider).overflow_x_scroll().child(
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

    fn editor(&mut self, width: f32, height: f32, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let wide = width > height && width >= WIDE;
        let bar = self.edit_bar(cx);
        if wide {
            // Landscape: tools beside the image, always open.
            self.sheet = (0.0, 0.0, 0.0);
            let side = (width * 0.4).clamp(300.0, 420.0);
            return v_flex()
                .size_full()
                .child(bar)
                .child(
                    h_flex()
                        .flex_1()
                        .min_h_0()
                        .child(v_flex().flex_1().h_full().min_w_0().child(self.viewer_el(cx)).child(self.rate_row(cx)).when(self.film, |d| d.child(self.filmstrip(cx))))
                        .child(
                            v_flex()
                                .w(px(side))
                                .h_full()
                                .flex_shrink_0()
                                .border_l_1()
                                .border_color(p.divider)
                                .child(div().flex_1().min_h_0().child(inspector::inspector(&self.state, window, cx)))
                                .child(self.tab_bar(window, cx)),
                        ),
                )
                .into_any_element();
        }
        let room = (height - TOP_BAR - TAB_BAR).max(0.0);
        let half = (room * 0.46).clamp(200.0, 440.0).min(room);
        let full = (room * 0.8).max(half);
        // Never reads as zero in portrait, which is how the tab bar tells the layouts apart.
        self.sheet = (half, full, height.max(1.0) - TAB_BAR);
        // At full height the sheet leaves room for the image alone.
        let tall = self.drag_h.map_or(self.detent == Detent::Full, |h| h > (half + full) / 2.0);
        let sheet = self.sheet_el(half, full, window, cx);
        v_flex()
            .size_full()
            .child(bar)
            .child(self.viewer_el(cx))
            .when(!tall, |d| d.child(self.rate_row(cx)).when(self.film, |d| d.child(self.filmstrip(cx))))
            .children(sheet)
            .child(self.tab_bar(window, cx))
            .into_any_element()
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
            .capture_any_mouse_up(move |_, _, cx| state.update(cx, |s, cx| s.end_gesture(cx)))
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
