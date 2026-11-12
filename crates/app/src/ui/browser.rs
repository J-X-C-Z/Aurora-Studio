//! The image browser: filmstrip, grid and list, with sorting, filtering and search.
//! Every mode virtualizes, so catalogs of any size scroll at full frame rate.

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, Sizable as _, VirtualListScrollHandle, h_flex, h_virtual_list,
    menu::{ContextMenuExt as _, PopupMenuItem},
    input::{Input, InputEvent, InputState},
    v_flex, v_virtual_list,
};
use gpui_kit::{
    AppContext as _, ClickEvent, Context, Entity, InteractiveElement, IntoElement, ObjectFit, ParentElement, Render,
    SharedString, StatefulInteractiveElement, Styled, StyledImage as _, Subscription, Window, div, img,
    prelude::FluentBuilder as _, px, size,
};
use lumen_engine::{ColorTag, ImageId, ImageInfo};

use crate::i18n::t;
use crate::state::{AppState, BrowserMode, SortKey};
use crate::theme::{hex, hexa, pal};
use crate::widgets::{IconBtn, select};

pub struct Browser {
    state: Entity<AppState>,
    search: Entity<InputState>,
    scroll: VirtualListScrollHandle,
    width: f32,
    _subs: Vec<Subscription>,
}

impl Browser {
    pub fn new(state: &Entity<AppState>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let search = cx.new(|cx| InputState::new(window, cx).placeholder(t("Search name or keyword", "搜索文件名或关键词")));
        let subs = vec![
            cx.observe(state, |_, _, cx| cx.notify()),
            cx.subscribe_in(&search, window, |this, input, e: &InputEvent, _, cx| {
                if let InputEvent::Change = e {
                    let text = input.read(cx).value().to_string();
                    this.state.update(cx, |s, cx| {
                        s.filter.search = text;
                        cx.notify();
                    });
                }
            }),
        ];
        Self { state: state.clone(), search, scroll: VirtualListScrollHandle::new(), width: 800.0, _subs: subs }
    }

    fn header(&self, mode: BrowserMode, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        let total = s.images.len();
        let shown = s.visible_images().len();
        let selected = s.selection.len();
        let filter = s.filter.clone();
        let sort = s.sort;
        let desc = s.sort_desc;
        let thumb = s.view.thumb_size;
        let (st1, st2, st3, st4, st5) = (self.state.clone(), self.state.clone(), self.state.clone(), self.state.clone(), self.state.clone());
        let sorts = vec![
            (SortKey::Name, t("Name", "名称").into()),
            (SortKey::CaptureDate, t("Capture date", "拍摄日期").into()),
            (SortKey::Rating, t("Rating", "评级").into()),
            (SortKey::ColorTag, t("Color tag", "色标").into()),
            (SortKey::FileSize, t("File size", "文件大小").into()),
            (SortKey::Custom, t("Custom", "自定义").into()),
        ];
        let ratings: Vec<(i8, SharedString)> = vec![
            (-1, t("All", "全部").into()),
            (0, t("Unrated+", "未评级及以上").into()),
            (1, "★+".into()),
            (2, "★★+".into()),
            (3, "★★★+".into()),
            (4, "★★★★+".into()),
            (5, "★★★★★".into()),
        ];
        h_flex()
            .h(px(30.0))
            .flex_shrink_0()
            .px_2()
            .gap_2()
            .bg(p.chrome)
            .border_b_1()
            .border_color(p.divider)
            .child(div().text_xs().text_color(p.text_muted).child(if selected > 1 {
                format!("{shown}/{total} · {} {selected}", t("selected", "已选"))
            } else {
                format!("{shown}/{total}")
            }))
            .child(div().w(px(130.0)).child(select("sort", sorts, sort, move |k, _, cx| {
                st1.update(cx, |s, cx| {
                    s.sort = k;
                    cx.notify();
                })
            })))
            .child(
                IconBtn::new("sort-dir", if desc { IconName::ArrowDownWideNarrow } else { IconName::ArrowUpNarrowWide }, t("Sort direction", "排序方向"))
                    .size(22.0)
                    .on_click(move |_, cx| {
                        st2.update(cx, |s, cx| {
                            s.sort_desc = !s.sort_desc;
                            cx.notify();
                        })
                    }),
            )
            .child(div().w(px(110.0)).child(select("rating-filter", ratings, filter.min_rating, move |r, _, cx| {
                st3.update(cx, |s, cx| {
                    s.filter.min_rating = r;
                    cx.notify();
                })
            })))
            .child(h_flex().gap_0p5().children(ColorTag::ALL.iter().map(|tag| {
                let tag = *tag;
                let on = filter.color_tag == Some(tag);
                let st = st4.clone();
                div()
                    .id(SharedString::from(format!("tagf-{tag:?}")))
                    .size(px(14.0))
                    .rounded_full()
                    .cursor_pointer()
                    .border_2()
                    .border_color(if on { p.text } else { gpui_kit::transparent_black() })
                    .when(tag == ColorTag::None, |d| d.bg(p.well).child(div().m(px(3.0)).size(px(4.0)).rounded_full().bg(p.text_subtle)))
                    .when(tag != ColorTag::None, |d| d.bg(hex(tag.rgb())))
                    .on_click(move |_, _, cx| {
                        st.update(cx, |s, cx| {
                            s.filter.color_tag = if s.filter.color_tag == Some(tag) { None } else { Some(tag) };
                            cx.notify();
                        })
                    })
            })))
            .child(div().flex_1())
            .when(mode == BrowserMode::Grid, |this| {
                this.child(
                    div().w(px(140.0)).child(
                        crate::widgets::ParamSlider::new("thumb-size", t("Size", "大小"), thumb, 80.0, 320.0)
                            .compact_label()
                            .on_change(move |v, _, cx| {
                                st5.update(cx, |s, cx| {
                                    s.view.thumb_size = v;
                                    cx.notify();
                                })
                            }),
                    ),
                )
            })
            .child(div().w(px(200.0)).child(Input::new(&self.search).xsmall().cleanable(true).prefix(Icon::new(IconName::Search).xsmall())))
    }
}

/// One thumbnail cell.
fn cell(
    state: &Entity<AppState>,
    info: &ImageInfo,
    box_size: f32,
    with_name: bool,
    window: &mut gpui_kit::Window,
    cx: &mut gpui_kit::App,
) -> impl IntoElement + use<> {
    let p = pal(cx).clone();
    let (current, selected) = {
        let s = state.read(cx);
        (s.current == Some(info.id), s.selection.contains(&info.id))
    };
    // Selection answers a click, so it eases in rather than switching.
    let key = format!("thumb-{}", info.id.0);
    let sel_t = crate::motion::toggle(crate::motion::key(&key, "sel"), selected && !current, crate::motion::QUICK, window, cx);
    let cur_t = crate::motion::toggle(crate::motion::key(&key, "cur"), current, crate::motion::QUICK, window, cx);
    let s = state.read(cx);
    let id = info.id;
    let thumb = s.thumbs.get(&id).cloned();
    let thumb_failed = thumb.is_none() && s.thumb_failed.contains(&id);
    let eyes_closed = s.eyes.get(&id).map(|r| r.closed).filter(|n| *n > 0);
    let edited = info.has_adjustments;
    let st = state.clone();
    let rating = info.rating;
    let tag = info.color_tag;
    let name: SharedString = info.file_name.clone().into();
    let variant = info.variant;
    let dragged: Vec<ImageId> = if selected { s.selection.iter().copied().collect() } else { vec![id] };
    // Stack: the pick carries a badge with the count; open members sit on a well.
    let stack = s.stack_of(id).map(|st| (st.pick() == id, st.ids.len(), s.stack_open(st.pick()), st.pick()));
    let member = stack.is_some_and(|(pick, ..)| !pick);
    let (st_drop, st_badge, st_menu) = (state.clone(), state.clone(), state.clone());
    let accent = p.accent;

    v_flex()
        .id(SharedString::from(format!("thumb-{}", id.0)))
        // Dropping images here puts them in front of this one (custom order).
        .drag_over::<crate::ui::collections::DraggedImages>(move |d, _, _, _| d.border_l_2().border_color(accent))
        .on_drop(move |d: &crate::ui::collections::DraggedImages, _, cx| {
            let ids = d.0.clone();
            st_drop.update(cx, |s, cx| s.move_images(ids, Some(id), cx))
        })
        .when(member, |d| d.border_1().border_color(p.divider))
        .w(px(box_size))
        .p(px(4.0))
        .gap_1()
        .rounded(px(6.0))
        .cursor_pointer()
        .bg(p.accent_soft.opacity(p.accent_soft.a * sel_t).blend(p.pressed.opacity(p.pressed.a * cur_t)))
        .hover(|d| d.bg(p.hover))
        .on_drag(crate::ui::collections::DraggedImages(dragged), |d, _, _, cx| {
            let count = d.0.len();
            cx.new(|_| crate::ui::collections::DragCard { count })
        })
        .on_click(move |e: &ClickEvent, _, cx| {
            let m = e.modifiers();
            let double = e.click_count() >= 2;
            st.update(cx, |s, cx| {
                s.select(id, m.control || m.platform, m.shift, cx);
                if double && s.view.browser == BrowserMode::Grid {
                    s.view.browser = BrowserMode::Filmstrip;
                    cx.notify();
                }
            })
        })
        .child(
            div()
                .relative()
                .w_full()
                .h(px(box_size - 8.0))
                .flex()
                .items_center()
                .justify_center()
                .child(match thumb {
                    Some(img_data) => crate::motion::enter(
                        SharedString::from(format!("thumb-img-{}", id.0)),
                        div()
                            .size_full()
                            .flex()
                            .items_center()
                            .justify_center()
                            .child(
                                img(img_data)
                                    .max_w_full()
                                    .max_h_full()
                                    .object_fit(ObjectFit::Contain)
                                    .rounded(px(2.0))
                                    .border_1()
                                    .border_color(hexa(0x000000, 0.6).blend(gpui_kit::white().opacity(cur_t))),
                            ),
                        0.0,
                    )
                    .into_any_element(),
                    None if thumb_failed => v_flex()
                        .size(px(box_size * 0.7))
                        .rounded(px(3.0))
                        .bg(p.well)
                        .items_center()
                        .justify_center()
                        .gap_1()
                        .text_color(p.warning)
                        .child(Icon::new(IconName::TriangleAlert).small())
                        .child(div().text_xs().text_color(p.text_subtle).child(t("Cannot read", "无法读取")))
                        .into_any_element(),
                    None => div().size(px(box_size * 0.7)).rounded(px(3.0)).bg(p.well).into_any_element(),
                })
                .when_some(eyes_closed, |d, n| {
                    d.child(
                        h_flex()
                            .absolute()
                            .top(px(16.0))
                            .right(px(3.0))
                            .gap_0p5()
                            .px_1()
                            .rounded(px(3.0))
                            .bg(p.warning)
                            .text_xs()
                            .text_color(hexa(0x000000, 0.85))
                            .child(Icon::new(IconName::ScanEye).xsmall())
                            .when(n > 1, |d| d.child(format!("{n}"))),
                    )
                })
                .when(tag != ColorTag::None, |d| {
                    d.child(div().absolute().top(px(4.0)).right(px(4.0)).size(px(9.0)).rounded_full().border_1().border_color(hexa(0x000000, 0.4)).bg(hex(tag.rgb())))
                })
                .when(edited, |d| {
                    d.child(
                        div()
                            .absolute()
                            .top(px(3.0))
                            .left(px(4.0))
                            .px_1()
                            .rounded(px(3.0))
                            .bg(hexa(0x000000, 0.55))
                            .text_color(hexa(0xffffff, 0.85))
                            .child(Icon::new(IconName::SlidersHorizontal).xsmall()),
                    )
                })
                .when_some(stack.filter(|(pick, ..)| *pick), |d, (_, count, open, pick)| {
                    d.child(
                        h_flex()
                            .id(SharedString::from(format!("stack-badge-{}", id.0)))
                            .absolute()
                            .bottom(px(3.0))
                            .right(px(4.0))
                            .gap_0p5()
                            .px_1()
                            .rounded(px(3.0))
                            .bg(if open { p.accent } else { hexa(0x000000, 0.65) })
                            .text_xs()
                            .text_color(hexa(0xffffff, 0.92))
                            .cursor_pointer()
                            .child(Icon::new(IconName::Layers).xsmall())
                            .child(format!("{count}"))
                            .on_click(move |_, _, cx| {
                                cx.stop_propagation();
                                st_badge.update(cx, |s, cx| s.toggle_stack(pick, cx))
                            }),
                    )
                })
                .when(variant > 1, |d| {
                    d.child(
                        div()
                            .absolute()
                            .bottom(px(3.0))
                            .left(px(4.0))
                            .px_1()
                            .rounded(px(3.0))
                            .bg(hexa(0x000000, 0.55))
                            .text_xs()
                            .text_color(hexa(0xffffff, 0.85))
                            .child(format!("{variant}")),
                    )
                }),
        )
        .child(
            h_flex()
                .h(px(12.0))
                .justify_center()
                .gap(px(1.0))
                .text_xs()
                .when(rating < 0, |d| d.text_color(p.danger).child(Icon::new(IconName::X).xsmall()))
                .when(rating >= 0, |d| {
                    d.children((1..=5).map(|i| {
                        div().text_color(if i <= rating { p.warning } else { p.text_disabled.opacity(0.6) }).child(Icon::new(IconName::Star).size(px(9.0)))
                    }))
                }),
        )
        .when(with_name, |d| {
            d.child(div().w_full().text_xs().text_color(if current { p.text } else { p.text_muted }).text_center().overflow_hidden().whitespace_nowrap().text_ellipsis().child(name))
        })
        .context_menu(move |menu, _, cx| {
            let s = st_menu.read(cx);
            let in_stack = s.stack_of(id).is_some();
            let is_pick = s.stack_of(id).is_some_and(|st| st.pick() == id);
            let st = st_menu.clone();
            menu.item(PopupMenuItem::new(t("Stack Selected", "堆叠所选")).on_click(|_, window, cx| window.dispatch_action(Box::new(crate::actions::StackSelected), cx)))
                .when(in_stack, |m| {
                    m.item(PopupMenuItem::new(t("Unstack", "取消堆叠")).on_click(|_, window, cx| window.dispatch_action(Box::new(crate::actions::UnstackSelected), cx)))
                })
                .when(in_stack && !is_pick, move |m| {
                    m.item(PopupMenuItem::new(t("Make Stack Pick", "设为堆叠首选")).on_click(move |_, _, cx| st.update(cx, |s, cx| s.set_stack_pick(id, cx))))
                })
                .item(PopupMenuItem::new(t("Auto Stack by Capture Time", "按拍摄时间自动堆叠")).on_click(|_, window, cx| window.dispatch_action(Box::new(crate::actions::AutoStack), cx)))
        })
}

fn list_row(state: &Entity<AppState>, info: &ImageInfo, cx: &gpui_kit::App) -> impl IntoElement + use<> {
    let p = pal(cx).clone();
    let s = state.read(cx);
    let id: ImageId = info.id;
    let current = s.current == Some(id);
    let selected = s.selection.contains(&id);
    let thumb = s.thumbs.get(&id).cloned();
    let st = state.clone();
    let e = &info.exif;
    let col = |w: f32, text: String| div().w(px(w)).flex_shrink_0().overflow_hidden().whitespace_nowrap().text_ellipsis().child(text);
    h_flex()
        .id(SharedString::from(format!("row-{}", id.0)))
        .h(px(36.0))
        .px_2()
        .gap_3()
        .text_xs()
        .text_color(if current { p.text } else { p.text_muted })
        .cursor_pointer()
        .border_b_1()
        .border_color(p.divider)
        .when(selected, |d| d.bg(p.accent_soft))
        .hover(|d| d.bg(p.hover))
        .on_click(move |ev: &ClickEvent, _, cx| {
            let m = ev.modifiers();
            st.update(cx, |s, cx| s.select(id, m.control || m.platform, m.shift, cx))
        })
        .child(div().w(px(44.0)).h(px(30.0)).flex().items_center().justify_center().when_some(thumb, |d, t| d.child(img(t).max_w_full().max_h_full().object_fit(ObjectFit::Contain))))
        .child(col(240.0, info.file_name.clone()).text_color(p.text))
        .child(col(150.0, e.captured_at.clone().unwrap_or_default().replace('T', " ").chars().take(16).collect()))
        .child(col(160.0, format!("{} {}", e.camera_make.clone().unwrap_or_default(), e.camera_model.clone().unwrap_or_default())))
        .child(col(70.0, e.iso.map(|i| format!("ISO {i}")).unwrap_or_default()))
        .child(col(70.0, e.shutter.map(|s| if s >= 1.0 { format!("{s:.0}s") } else { format!("1/{:.0}", 1.0 / s) }).unwrap_or_default()))
        .child(col(60.0, e.aperture.map(|a| format!("f/{a:.1}")).unwrap_or_default()))
        .child(col(80.0, if info.rating < 0 { "✕".into() } else { "★".repeat(info.rating as usize) }).text_color(p.warning))
        .child(div().w(px(12.0)).h(px(12.0)).rounded_full().when(info.color_tag != ColorTag::None, |d| d.bg(hex(info.color_tag.rgb()))))
        .child(col(80.0, format!("{:.1} MB", info.file_size as f64 / 1_000_000.0)))
        .child(col(120.0, format!("{} × {}", info.width, info.height)))
}

impl Render for Browser {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        let mode = s.view.browser;
        let height = s.view.filmstrip_height;
        let thumb = s.view.thumb_size;
        let ids: Vec<ImageId> = s.visible_images().iter().map(|i| i.id).collect();
        let count = ids.len();
        let filtered = s.filter_active();
        let ids = Rc::new(ids);
        let header = self.header(mode, cx);
        let this = cx.entity();
        let width_probe = {
            let this = this.clone();
            move |bounds: gpui_kit::Bounds<gpui_kit::Pixels>, _: &mut Window, cx: &mut gpui_kit::App| {
                let w = f32::from(bounds.size.width);
                if (this.read(cx).width - w).abs() > 1.0 {
                    this.update(cx, |b, cx| {
                        b.width = w;
                        cx.notify();
                    });
                }
            }
        };
        let _ = window;

        let body = match mode {
            BrowserMode::Filmstrip | BrowserMode::Hidden => {
                let box_size = (height - 58.0).max(60.0);
                let sizes = Rc::new(vec![size(px(box_size + 4.0), px(box_size)); count]);
                let ids = ids.clone();
                h_virtual_list(this.clone(), "filmstrip", sizes, move |b, range, window, cx| {
                    range
                        .filter_map(|i| {
                            let id = ids[i];
                            let info = b.state.read(cx).info(id)?.clone();
                            Some(div().px(px(2.0)).child(cell(&b.state, &info, box_size, false, window, cx)))
                        })
                        .collect()
                })
                .track_scroll(&self.scroll)
                .px_1()
                .py_1()
                .size_full()
                .into_any_element()
            }
            BrowserMode::Grid => {
                let cell_w = thumb + 8.0;
                let per_row = ((self.width - 16.0) / cell_w).floor().max(1.0) as usize;
                let rows = count.div_ceil(per_row);
                let row_h = thumb + 30.0;
                let sizes = Rc::new(vec![size(px(self.width), px(row_h)); rows]);
                let ids = ids.clone();
                v_virtual_list(this.clone(), "grid", sizes, move |b, range, window, cx| {
                    range
                        .map(|r| {
                            let start = r * per_row;
                            let end = (start + per_row).min(ids.len());
                            h_flex().px_2().gap_2().children((start..end).filter_map(|i| {
                                let info = b.state.read(cx).info(ids[i])?.clone();
                                Some(cell(&b.state, &info, thumb, true, window, cx))
                            }))
                        })
                        .collect()
                })
                .track_scroll(&self.scroll)
                .py_2()
                .size_full()
                .into_any_element()
            }
            BrowserMode::List => {
                let sizes = Rc::new(vec![size(px(self.width), px(36.0)); count]);
                let ids = ids.clone();
                v_virtual_list(this.clone(), "list", sizes, move |b, range, _, cx| {
                    range.filter_map(|i| Some(list_row(&b.state, &b.state.read(cx).info(ids[i])?.clone(), cx))).collect()
                })
                .track_scroll(&self.scroll)
                .size_full()
                .into_any_element()
            }
        };

        use gpui_kit::base::ElementExt as _;
        v_flex()
            .size_full()
            .bg(p.panel)
            .child(header)
            .child(
                div()
                    .id("browser-body")
                    .relative()
                    .flex_1()
                    .min_h_0()
                    .child(body)
                    .on_prepaint(width_probe)
                    .when(count == 0, |d| {
                        let st = self.state.clone();
                        d.child(div().absolute().top_0().left_0().size_full().child(crate::motion::enter(
                            if filtered { "browser-empty-filter" } else { "browser-empty" },
                            v_flex()
                                .size_full()
                                .items_center()
                                .justify_center()
                                .gap_1p5()
                                .text_xs()
                                .text_color(p.text_subtle)
                                .child(if filtered {
                                    t("No images match the filter", "没有符合筛选条件的图像")
                                } else {
                                    t("Nothing here yet", "这里还没有图像")
                                })
                                .child(if filtered {
                                    gpui_kit::component::button::Button::new("browser-clear-filter")
                                        .xsmall()
                                        .outline()
                                        .label(t("Clear Filter", "清除筛选"))
                                        .on_click(move |_, _, cx| st.update(cx, |s, cx| s.clear_filter(cx)))
                                } else {
                                    gpui_kit::component::button::Button::new("browser-import")
                                        .xsmall()
                                        .outline()
                                        .label(t("Import…", "导入…"))
                                        .on_click(|_, window, cx| window.dispatch_action(Box::new(crate::actions::Import), cx))
                                }),
                            4.0,
                        )))
                    }),
            )
    }
}
