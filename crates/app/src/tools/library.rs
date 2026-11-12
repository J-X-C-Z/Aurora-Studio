//! Library tab: collections, filters and catalog information.

use gpui_kit::assets::IconName;
use gpui_kit::base::MotionReveal;
use gpui_kit::component::{
    Icon, Sizable as _,
    button::{Button, ButtonVariants as _},
    menu::{ContextMenuExt as _, DropdownMenu as _, PopupMenuItem},
    h_flex, v_flex,
};
use gpui_kit::{
    AnyElement, App, Entity, InteractiveElement, IntoElement, ParentElement, SharedString, StatefulInteractiveElement,
    Styled, Window, div, prelude::FluentBuilder as _, px, radians,
};
use lumen_engine::*;

use crate::i18n::t;
use crate::motion;
use crate::state::{AppState, CollNode};
use crate::ui::collections::{DraggedImages, rename, smart_rules};
use crate::theme::{hex, pal};
use crate::widgets::*;

pub fn render(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    v_flex()
        .w_full()
        .child(collections_panel(state, window, cx))
        .child(filters_panel(state, window, cx))
        .child(catalog_panel(state, cx))
        .into_any_element()
}

fn kind_icon(k: CollectionKind) -> IconName {
    match k {
        CollectionKind::AllImages => IconName::Images,
        CollectionKind::RecentImports => IconName::ImageDown,
        CollectionKind::RecentCaptures => IconName::Camera,
        CollectionKind::Trash => IconName::Trash,
        CollectionKind::Folder => IconName::Folder,
        CollectionKind::Album => IconName::BookImage,
        CollectionKind::SmartAlbum => IconName::Sparkles,
        CollectionKind::Project => IconName::Briefcase,
        CollectionKind::Group => IconName::FolderTree,
    }
}

fn node(state: &Entity<AppState>, c: &CollNode, depth: usize, window: &mut Window, cx: &mut App) -> AnyElement {
    let p = pal(cx).clone();
    let s = state.read(cx);
    let active = s.active_collection == c.id;
    let count = s.collection_count(c.id);
    let children: Vec<CollNode> = s.child_nodes(Some(c.id)).into_iter().cloned().collect();
    let has_children = !children.is_empty();
    let selection: Vec<ImageId> = s.selection.iter().copied().collect();
    let open_state = window.use_keyed_state(SharedString::from(format!("coll-open-{}", c.id)), cx, |_, _| true);
    let open = *open_state.read(cx);
    let progress = motion::toggle(motion::key(format!("coll-{}", c.id), "open"), open, motion::PANEL, window, cx);
    let st = state.clone();
    let id = c.id;
    let kind = c.kind;
    let user = c.user;
    let name: SharedString = if c.user { c.name.clone() } else { crate::i18n::builtin(&c.name) }.into();
    let toggle = open_state.clone();
    let st_menu = state.clone();
    let menu_name = c.name.clone();
    let row = h_flex()
        .id(SharedString::from(format!("coll-{}", c.id)))
        .h(px(26.0))
        .mx_2()
        .pl(px(8.0 + depth as f32 * 14.0))
        .pr_2()
        .gap_1p5()
        .rounded(px(5.0))
        .cursor_pointer()
        .when(active, |d| d.bg(p.accent_soft))
        .hover(|d| d.bg(p.hover))
        .when(kind == CollectionKind::Album, |d| {
            let st = state.clone();
            d.drag_over::<DraggedImages>(move |style, _, _, cx| style.bg(pal(cx).accent.opacity(0.35)))
                .on_drop(move |drag: &DraggedImages, _, cx| st.update(cx, |s, cx| s.add_to_album(id, &drag.0, cx)))
        })
        .on_click({
            let name = name.clone();
            move |e: &gpui_kit::ClickEvent, window, cx| {
                if e.click_count() >= 2 && user {
                    rename(&st, id, name.to_string(), window, cx);
                    return;
                }
                st.update(cx, |s, cx| s.set_active_collection(id, cx))
            }
        })
        .child(
            div()
                .id(SharedString::from(format!("coll-chev-{}", c.id)))
                .size(px(14.0))
                .flex()
                .items_center()
                .justify_center()
                .when(has_children, |d| {
                    d.child(Icon::new(IconName::ChevronRight).xsmall().text_color(p.text_subtle).rotate(radians(std::f32::consts::FRAC_PI_2 * progress)))
                        .on_click(move |_, _, cx| {
                            cx.stop_propagation();
                            toggle.update(cx, |o, cx| {
                                *o = !*o;
                                cx.notify();
                            })
                        })
                }),
        )
        .child(Icon::new(kind_icon(kind)).xsmall().text_color(if active { p.accent } else { p.text_muted }))
        .child(div().flex_1().text_xs().text_color(if active { p.text } else { p.text_muted }).overflow_hidden().whitespace_nowrap().text_ellipsis().child(name))
        .child(div().text_xs().text_color(p.text_subtle).child(format!("{count}")))
        .context_menu(move |mut menu, _, _| {
            let st = st_menu.clone();
            let container = matches!(kind, CollectionKind::Project | CollectionKind::Group);
            if container {
                for (k, label) in [
                    (CollectionKind::Album, t("New Album inside", "在其中新建相册")),
                    (CollectionKind::SmartAlbum, t("New Smart Album inside", "在其中新建智能相册")),
                    (CollectionKind::Project, t("New Project inside", "在其中新建项目")),
                    (CollectionKind::Group, t("New Group inside", "在其中新建组")),
                ] {
                    if k == CollectionKind::Project && kind == CollectionKind::Project {
                        continue;
                    }
                    let st = st.clone();
                    menu = menu.item(PopupMenuItem::new(label).on_click(move |_, _, cx| {
                        st.update(cx, |s, cx| {
                            s.create_collection(Some(id), k, None, cx);
                        })
                    }));
                }
                menu = menu.separator();
            }
            if kind == CollectionKind::Album {
                let (a, b) = (st.clone(), st.clone());
                let sel = selection.clone();
                let sel2 = selection.clone();
                menu = menu
                    .item(PopupMenuItem::new(format!("{} ({})", t("Add selected images", "加入所选图像"), sel.len())).on_click(move |_, _, cx| a.update(cx, |s, cx| s.add_to_album(id, &sel, cx))))
                    .item(PopupMenuItem::new(t("Remove selected images", "移出所选图像")).on_click(move |_, _, cx| b.update(cx, |s, cx| s.remove_from_album(id, &sel2, cx))))
                    .separator();
            }
            if kind == CollectionKind::SmartAlbum {
                let st = st.clone();
                menu = menu.item(PopupMenuItem::new(t("Edit rules…", "编辑规则…")).on_click(move |_, window, cx| smart_rules(&st, id, window, cx))).separator();
            }
            if kind == CollectionKind::Trash {
                let st = st.clone();
                menu = menu.item(PopupMenuItem::new(t("Empty Trash", "清空废纸篓")).on_click(move |_, _, cx| st.update(cx, |s, cx| s.empty_trash(cx))));
            }
            if user {
                let (a, b) = (st.clone(), st.clone());
                let n = menu_name.clone();
                menu = menu
                    .item(PopupMenuItem::new(t("Rename…", "重命名…")).on_click(move |_, window, cx| rename(&a, id, n.clone(), window, cx)))
                    .item(PopupMenuItem::new(t("Delete", "删除")).on_click(move |_, _, cx| b.update(cx, |s, cx| s.delete_collection(id, cx))));
            }
            if kind == CollectionKind::Folder {
                menu = menu.item(PopupMenuItem::new(t("Show in Explorer", "在资源管理器中显示")).disabled(true));
            }
            menu
        });

    let mut col = v_flex().w_full().child(row);
    if has_children {
        let kids: Vec<AnyElement> = children.iter().map(|ch| node(state, ch, depth + 1, window, cx)).collect();
        col = col.child(MotionReveal::new(SharedString::from(format!("coll-reveal-{}", c.id)), progress, v_flex().w_full().children(kids).into_any_element()));
    }
    col.into_any_element()
}

fn collections_panel(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> impl IntoElement + use<> {
    let roots: Vec<CollNode> = state.read(cx).child_nodes(None).into_iter().cloned().collect();
    let p = pal(cx).clone();
    let is_catalog = |c: &CollNode| {
        matches!(c.kind, CollectionKind::AllImages | CollectionKind::RecentImports | CollectionKind::RecentCaptures | CollectionKind::Trash)
    };
    let section = |label: &'static str| div().px_3().pt_2().pb_1().text_xs().text_color(p.text_subtle).child(label);
    let catalog: Vec<AnyElement> = roots.iter().filter(|c| is_catalog(c)).map(|c| node(state, c, 0, window, cx)).collect();
    let user: Vec<AnyElement> = roots.iter().filter(|c| !is_catalog(c) && c.kind != CollectionKind::Folder).map(|c| node(state, c, 0, window, cx)).collect();
    let folders: Vec<AnyElement> = roots.iter().filter(|c| c.kind == CollectionKind::Folder).map(|c| node(state, c, 0, window, cx)).collect();
    let st_new = state.clone();
    let new_menu = Button::new("coll-new")
        .xsmall()
        .ghost()
        .icon(Icon::new(IconName::Plus))
        .dropdown_menu(move |mut menu, _, _| {
            for (k, label) in [
                (CollectionKind::Album, t("New Album", "新建相册")),
                (CollectionKind::SmartAlbum, t("New Smart Album", "新建智能相册")),
                (CollectionKind::Project, t("New Project", "新建项目")),
                (CollectionKind::Group, t("New Group", "新建组")),
            ] {
                let st = st_new.clone();
                menu = menu.item(PopupMenuItem::new(label).on_click(move |_, window, cx| {
                    let id = st.update(cx, |s, cx| s.create_collection(None, k, None, cx));
                    let name = st.read(cx).node(id).map(|n| n.name.clone()).unwrap_or_default();
                    if k == CollectionKind::SmartAlbum {
                        smart_rules(&st, id, window, cx);
                    } else {
                        rename(&st, id, name, window, cx);
                    }
                }));
            }
            menu
        });
    ToolPanel::new(state, "collections", t("Library", "图库"))
        .action(HeaderAction::new("import", IconName::ImageDown, t("Import images", "导入图像"), |window, cx| {
            window.dispatch_action(Box::new(crate::actions::Import), cx)
        }))
        .child(section(t("Catalog Collections", "目录集合")))
        .children(catalog)
        .child(h_flex().pr_2().child(div().flex_1().child(section(t("User Collections", "用户集合")))).child(new_menu))
        .children(user)
        .child(div().px_3().py_1().text_xs().text_color(p.text_disabled).child(t("Drag thumbnails onto an album to add them · right-click for more", "将缩略图拖到相册即可加入 · 右键查看更多")))
        .child(section(t("Folders", "文件夹")))
        .children(folders)
}

fn filters_panel(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let f = s.filter.clone();
    let counts: Vec<usize> = (-1..=5).map(|r| s.images.iter().filter(|i| i.rating == r).count()).collect();
    let tag_counts: Vec<usize> = ColorTag::ALL.iter().map(|t| s.images.iter().filter(|i| i.color_tag == *t).count()).collect();
    let (st, st2, st3) = (state.clone(), state.clone(), state.clone());
    let search = text_field(
        "lib-search",
        f.search.clone(),
        t("Search", "搜索"),
        move |text, _, cx| {
            st3.update(cx, |s, cx| {
                s.filter.search = text;
                cx.notify();
            })
        },
        window,
        cx,
    );
    let rating_rows = (-1..=5i8).rev().map(|r| {
        let st = state.clone();
        let active = f.min_rating == r && r != -1;
        h_flex()
            .id(SharedString::from(format!("rf-{r}")))
            .h(px(22.0))
            .mx_2()
            .px_2()
            .gap_2()
            .rounded(px(4.0))
            .cursor_pointer()
            .when(active, |d| d.bg(p.accent_soft))
            .hover(|d| d.bg(p.hover))
            .on_click(move |_, _, cx| {
                st.update(cx, |s, cx| {
                    s.filter.min_rating = if s.filter.min_rating == r { -1 } else { r };
                    cx.notify();
                })
            })
            .child(match r {
                -1 => h_flex().text_color(p.text_muted).child(Icon::new(IconName::X).xsmall().text_color(p.danger)).child(div().ml_1().text_xs().child(t("Rejected", "已拒绝"))).into_any_element(),
                0 => div().text_xs().text_color(p.text_muted).child(t("Unrated and up", "未评级及以上")).into_any_element(),
                n => h_flex()
                    .gap(px(1.0))
                    .children((1..=5).map(move |i| div().text_color(if i <= n { p.warning } else { p.text_disabled }).child(Icon::new(IconName::Star).size(px(10.0)))))
                    .into_any_element(),
            })
            .child(div().flex_1())
            .child(div().text_xs().text_color(p.text_subtle).child(format!("{}", counts[(r + 1) as usize])))
    });
    ToolPanel::new(state, "filters", t("Filters", "筛选器"))
        .modified(f != crate::state::Filter::default())
        .on_reset(move |_, cx| {
            st.update(cx, |s, cx| {
                s.filter = Default::default();
                cx.notify();
            })
        })
        .child(div().px_3().pb_2().child(search))
        .child(caption(t("Rating", "评级"), cx))
        .children(rating_rows)
        .child(caption(t("Color Tag", "色标"), cx))
        .child(h_flex().px_3().gap_1().flex_wrap().children(ColorTag::ALL.iter().enumerate().map(|(i, tag)| {
            let tag = *tag;
            let st = state.clone();
            let on = f.color_tag == Some(tag);
            h_flex()
                .id(SharedString::from(format!("tf-{i}")))
                .gap_1()
                .px_1p5()
                .h(px(20.0))
                .rounded(px(10.0))
                .cursor_pointer()
                .bg(if on { p.pressed } else { p.well })
                .child(div().size(px(9.0)).rounded_full().when(tag == ColorTag::None, |d| d.border_1().border_color(p.text_subtle)).when(tag != ColorTag::None, |d| d.bg(hex(tag.rgb()))))
                .child(div().text_xs().text_color(p.text_muted).child(format!("{}", tag_counts[i])))
                .on_click(move |_, _, cx| {
                    st.update(cx, |s, cx| {
                        s.filter.color_tag = if s.filter.color_tag == Some(tag) { None } else { Some(tag) };
                        cx.notify();
                    })
                })
        })))
        .child(div().h(px(4.0)))
        .child(switch_row("edited-only", t("Edited images only", "仅已编辑图像"), f.edited_only, move |v, _, cx| {
            st2.update(cx, |s, cx| {
                s.filter.edited_only = v;
                cx.notify();
            })
        }, cx))
}

fn catalog_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let total = s.images.len();
    let edited = s.images.iter().filter(|i| i.has_adjustments).count();
    let bytes: u64 = s.images.iter().map(|i| i.file_size).sum();
    let picked = s.images.iter().filter(|i| i.rating >= 4).count();
    let line = |k: &'static str, v: String| h_flex().px_3().h(px(20.0)).text_xs().child(div().flex_1().text_color(p.text_subtle).child(k)).child(div().text_color(p.text_muted).child(v));
    ToolPanel::new(state, "catalog", t("Catalog", "目录"))
        .child(line(t("Collection", "当前集合"), s.active_collection_name.to_string()))
        .child(line(t("Images", "图像"), format!("{total}")))
        .child(line(t("Edited", "已编辑"), format!("{edited}")))
        .child(line(t("Rated 4★+", "4 星及以上"), format!("{picked}")))
        .child(line(t("Size on disk", "磁盘占用"), format!("{:.1} GB", bytes as f64 / 1e9)))
        .child(div().px_3().pt_1().child(Button::new("lib-import").xsmall().outline().icon(Icon::new(IconName::ImageDown)).label(t("Import images…", "导入图像…")).on_click(|_, window, cx| {
            window.dispatch_action(Box::new(crate::actions::Import), cx)
        })))
}
