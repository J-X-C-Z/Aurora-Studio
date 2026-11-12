//! Metadata tab: rating and tag, file details, EXIF, IPTC and keywords.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Sizable as _,
    button::ButtonVariants as _,
    description_list::{DescriptionItem, DescriptionList},
    h_flex,
    rating::Rating,
    v_flex,
};
use gpui_kit::{
    AnyElement, App, Entity, InteractiveElement, IntoElement, ParentElement, SharedString, StatefulInteractiveElement,
    Styled, Window, div, prelude::FluentBuilder as _, px,
};
use lumen_engine::{ColorTag, ImageInfo, Iptc};

use crate::i18n::t;
use crate::state::AppState;
use crate::theme::{hex, pal};
use crate::widgets::*;

pub fn render(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    let Some(info) = state.read(cx).current_info().cloned() else {
        return div()
            .p_4()
            .child(ely_core::feedback::EmptyState::new("meta-empty", ely_core::primitives::IconName::Image, t("No image selected", "未选择图像")))
            .into_any_element();
    };
    v_flex()
        .w_full()
        .child(presets_panel(state, cx))
        .child(rating_panel(state, &info, cx))
        .child(file_panel(state, &info, cx))
        .child(exif_panel(state, &info, cx))
        .child(iptc_panel(state, &info, window, cx))
        .child(keywords_panel(state, &info, window, cx))
        .into_any_element()
}

/// Metadata presets: IPTC fields and keywords saved from one image and stamped
/// onto the selection.
fn presets_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let presets = s.settings.metadata_presets.clone();
    let n = s.selection.len().max(1);
    let st_save = state.clone();
    let mut panel = ToolPanel::new(state, "metadata-presets", t("Metadata Presets", "元数据预设"))
        .badge(presets.len().to_string())
        .child(
            h_flex().px_3().pb_1().child(
                gpui_kit::component::button::Button::new("meta-preset-save")
                    .xsmall()
                    .outline()
                    .icon(gpui_kit::component::Icon::new(IconName::Plus))
                    .label(t("Save Current as Preset…", "将当前元数据存储为预设…"))
                    .on_click(move |_, window, cx| {
                        let st = st_save.clone();
                        let count = st.read(cx).settings.metadata_presets.len();
                        crate::ui::naming::prompt(
                            t("Save Metadata Preset", "存储元数据预设"),
                            t("Saves this image's IPTC fields and keywords. Applying fills only the fields the preset has and adds its keywords.", "存储当前图像的 IPTC 字段与关键词。应用时只填写预设中有内容的字段，并添加关键词。"),
                            format!("{} {}", t("Preset", "预设"), count + 1),
                            t("Preset name", "预设名称"),
                            window,
                            cx,
                            move |name, cx| st.update(cx, |s, cx| s.save_metadata_preset(name, cx)),
                        );
                    }),
            ),
        );
    if presets.is_empty() {
        panel = panel.child(div().px_3().pb_2().text_xs().text_color(p.text_subtle).child(t(
            "Fill in copyright and creator once, save them as a preset, then apply it to any selection.",
            "填好版权、创作者等信息后存储为预设，之后可一次应用到所选图像。",
        )));
    }
    for (i, preset) in presets.into_iter().enumerate() {
        let (st_apply, st_ren, st_del) = (state.clone(), state.clone(), state.clone());
        let filled = [&preset.iptc.creator, &preset.iptc.copyright, &preset.iptc.city, &preset.iptc.country, &preset.iptc.title, &preset.iptc.headline]
            .into_iter()
            .filter(|v| !v.is_empty())
            .cloned()
            .chain(preset.keywords.iter().map(|k| format!("#{k}")))
            .collect::<Vec<_>>()
            .join("  ");
        let name = preset.name.clone();
        panel = panel.child(
            h_flex()
                .id(SharedString::from(format!("mpreset-{i}")))
                .group("mpreset")
                .mx_2()
                .px_2()
                .py_1()
                .gap_2()
                .rounded(px(5.0))
                .hover(|d| d.bg(p.hover))
                .child(gpui_kit::component::Icon::new(IconName::Tags).xsmall().text_color(p.text_subtle))
                .child(
                    v_flex()
                        .flex_1()
                        .min_w_0()
                        .child(div().text_xs().text_color(p.text).child(preset.name.clone()))
                        .child(div().text_size(px(10.0)).text_color(p.text_subtle).overflow_hidden().whitespace_nowrap().text_ellipsis().child(filled)),
                )
                .child(
                    h_flex()
                        .gap_0p5()
                        .invisible()
                        .group_hover("mpreset", |d| d.visible())
                        .child(IconBtn::new(SharedString::from(format!("mpreset-ren-{i}")), IconName::Pencil, t("Rename", "重命名")).size(20.0).on_click(move |window, cx| {
                            let st = st_ren.clone();
                            crate::ui::naming::prompt(t("Rename Preset", "重命名预设"), "", name.clone(), t("Preset name", "预设名称"), window, cx, move |n, cx| {
                                st.update(cx, |s, cx| s.rename_metadata_preset(i, n, cx))
                            });
                        }))
                        .child(IconBtn::new(SharedString::from(format!("mpreset-del-{i}")), IconName::Trash, t("Delete", "删除")).size(20.0).on_click(move |_, cx| {
                            st_del.update(cx, |s, cx| s.delete_metadata_preset(i, cx))
                        })),
                )
                .child(
                    gpui_kit::component::button::Button::new(SharedString::from(format!("mpreset-apply-{i}")))
                        .xsmall()
                        .primary()
                        .label(format!("{} ({n})", t("Apply", "应用")))
                        .on_click(move |_, _, cx| st_apply.update(cx, |s, cx| s.apply_metadata_preset(i, cx))),
                ),
        );
    }
    panel
}

fn rating_panel(state: &Entity<AppState>, info: &ImageInfo, cx: &App) -> impl IntoElement + use<> {
    let p = pal(cx).clone();
    let rating = info.rating;
    let tag = info.color_tag;
    let (st, st2) = (state.clone(), state.clone());
    let n = state.read(cx).selection.len();
    ToolPanel::new(state, "rating", t("Rating & Tag", "评级与色标"))
        .when(n > 1, |p| p.badge(format!("{} {n}", t("Editing", "正在编辑"))))
        .child(row(
            t("Rating", "评级"),
            h_flex()
                .gap_2()
                .child(Rating::new("meta-rating").small().color(p.warning).max(5).value(rating.max(0) as usize).on_click(move |v, _, cx| {
                    let v = *v as i8;
                    st.update(cx, |s, cx| {
                        let cur = s.current_info().map(|i| i.rating).unwrap_or(0);
                        s.set_rating(if v == cur { 0 } else { v }, cx)
                    })
                }))
                .child(
                    IconBtn::new("meta-reject", IconName::X, t("Reject", "拒绝")).size(22.0).active(rating < 0).on_click(move |_, cx| {
                        st2.update(cx, |s, cx| {
                            let cur = s.current_info().map(|i| i.rating).unwrap_or(0);
                            s.set_rating(if cur < 0 { 0 } else { -1 }, cx)
                        })
                    }),
                ),
            cx,
        ))
        .child(row(
            t("Color Tag", "色标"),
            h_flex().gap_1().children(ColorTag::ALL.iter().map(|c| {
                let c = *c;
                let st = state.clone();
                div()
                    .id(SharedString::from(format!("meta-tag-{c:?}")))
                    .size(px(18.0))
                    .rounded_full()
                    .cursor_pointer()
                    .border_2()
                    .border_color(if c == tag { p.text } else { p.panel })
                    .when(c == ColorTag::None, |d| d.bg(p.well).child(div().m(px(5.0)).size(px(4.0)).rounded_full().bg(p.text_subtle)))
                    .when(c != ColorTag::None, |d| d.bg(hex(c.rgb())))
                    .on_click(move |_, _, cx| st.update(cx, |s, cx| s.set_color_tag(c, cx)))
            })),
            cx,
        ))
}

fn file_panel(state: &Entity<AppState>, info: &ImageInfo, _cx: &App) -> impl IntoElement + use<> {
    let kind = format!("{:?}", info.kind).to_uppercase();
    ToolPanel::new(state, "file", t("File", "文件")).child(
        div().px_3().text_xs().child(
            DescriptionList::new()
                .columns(1)
                .bordered(false)
                .label_width(px(84.0))
                .child(DescriptionItem::new(t("Name", "名称")).value(info.file_name.clone()))
                .child(DescriptionItem::new(t("Type", "类型")).value(kind))
                .child(DescriptionItem::new(t("Size", "大小")).value(format!("{:.1} MB", info.file_size as f64 / 1e6)))
                .child(DescriptionItem::new(t("Dimensions", "尺寸")).value(format!("{} × {}  ({:.1} MP)", info.width, info.height, info.width as f64 * info.height as f64 / 1e6)))
                .child(DescriptionItem::new(t("Variant", "变体")).value(format!("{}", info.variant)))
                .child(DescriptionItem::new(t("Path", "路径")).value(info.path.display().to_string())),
        ),
    )
}

fn exif_panel(state: &Entity<AppState>, info: &ImageInfo, _cx: &App) -> impl IntoElement + use<> {
    let e = info.exif.clone();
    let opt = |v: Option<String>| v.unwrap_or_else(|| "—".into());
    ToolPanel::new(state, "exif", t("EXIF", "EXIF")).child(
        div().px_3().text_xs().child(
            DescriptionList::new()
                .columns(1)
                .bordered(false)
                .label_width(px(84.0))
                .child(DescriptionItem::new(t("Camera", "相机")).value(opt(e.camera_model.clone().map(|m| format!("{} {m}", e.camera_make.clone().unwrap_or_default())))))
                .child(DescriptionItem::new(t("Lens", "镜头")).value(opt(e.lens.clone())))
                .child(DescriptionItem::new(t("Focal Length", "焦距")).value(opt(e.focal_length_mm.map(|f| format!("{f:.0} mm")))))
                .child(DescriptionItem::new(t("Aperture", "光圈")).value(opt(e.aperture.map(|a| format!("f/{a:.1}")))))
                .child(DescriptionItem::new(t("Shutter", "快门")).value(opt(e.shutter.map(|s| if s >= 1.0 { format!("{s:.1} s") } else { format!("1/{:.0} s", 1.0 / s) }))))
                .child(DescriptionItem::new("ISO").value(opt(e.iso.map(|i| i.to_string()))))
                .child(DescriptionItem::new(t("Exposure Bias", "曝光补偿")).value(opt(e.exposure_bias.map(|b| format!("{b:+.1} EV")))))
                .child(DescriptionItem::new(t("Flash", "闪光灯")).value(opt(e.flash.map(|f| if f { t("Fired", "已闪光").into() } else { t("Off", "关闭").into() }))))
                .child(DescriptionItem::new(t("Captured", "拍摄时间")).value(opt(e.captured_at.map(|d| d.replace('T', " ").chars().take(19).collect()))))
                .child(DescriptionItem::new("GPS").value(opt(e.gps.map(|(a, b)| format!("{a:.5}, {b:.5}"))))),
        ),
    )
}

fn iptc_panel(state: &Entity<AppState>, info: &ImageInfo, window: &mut Window, cx: &mut App) -> impl IntoElement + use<> {
    let id = info.id.0;
    let rev = state.read(cx).metadata_rev;
    let iptc = info.iptc.clone();
    let field = |key: &'static str, label: &'static str, value: String, set: fn(&mut Iptc, String), window: &mut Window, cx: &mut App| {
        let st = state.clone();
        let input = text_field(
            SharedString::from(format!("iptc-{key}-{id}-{rev}")),
            value,
            "",
            move |text, _, cx| st.update(cx, |s, cx| s.update_current_info(|i| set(&mut i.iptc, text), cx)),
            window,
            cx,
        );
        row(label, input, cx)
    };
    ToolPanel::new(state, "iptc", t("IPTC", "IPTC"))
        .child(field("title", t("Title", "标题"), iptc.title.clone(), |i, v| i.title = v, window, cx))
        .child(field("headline", t("Headline", "大标题"), iptc.headline.clone(), |i, v| i.headline = v, window, cx))
        .child(field("description", t("Description", "描述"), iptc.description.clone(), |i, v| i.description = v, window, cx))
        .child(field("creator", t("Creator", "创作者"), iptc.creator.clone(), |i, v| i.creator = v, window, cx))
        .child(field("copyright", t("Copyright", "版权"), iptc.copyright.clone(), |i, v| i.copyright = v, window, cx))
        .child(field("city", t("City", "城市"), iptc.city.clone(), |i, v| i.city = v, window, cx))
        .child(field("country", t("Country", "国家"), iptc.country.clone(), |i, v| i.country = v, window, cx))
        .child(field("job", t("Job ID", "任务编号"), iptc.job_id.clone(), |i, v| i.job_id = v, window, cx))
}

fn keywords_panel(state: &Entity<AppState>, info: &ImageInfo, window: &mut Window, cx: &mut App) -> impl IntoElement + use<> {
    let p = pal(cx).clone();
    let keywords = info.keywords.clone();
    let st = state.clone();
    // The key includes the keyword count, so the field starts empty after each addition.
    let add = text_field(
        SharedString::from(format!("kw-add-{}-{}", info.id.0, keywords.len())),
        "",
        t("Add keyword and press Enter", "输入关键词后按回车"),
        move |text, enter, cx| {
            let text = text.trim().to_string();
            if enter && !text.is_empty() {
                st.update(cx, |s, cx| {
                    s.update_current_info(|i| {
                        if !i.keywords.contains(&text) {
                            i.keywords.push(text.clone())
                        }
                    }, cx)
                });
            }
        },
        window,
        cx,
    );
    ToolPanel::new(state, "keywords", t("Keywords", "关键词"))
        .badge(format!("{}", keywords.len()))
        .child(div().px_3().pb_1p5().child(add))
        .child(
            h_flex().px_3().gap_1().flex_wrap().children(keywords.iter().map(|k| {
                let st = state.clone();
                let k2 = k.clone();
                h_flex()
                    .gap_0p5()
                    .child(ely_core::data_display::Tag::new(SharedString::from(format!("kw-{k}")), k.clone()))
                    .child(IconBtn::new(SharedString::from(format!("kw-del-{k}")), IconName::X, t("Remove", "移除")).size(16.0).on_click(move |_, cx| {
                        let k = k2.clone();
                        st.update(cx, |s, cx| s.update_current_info(|i| i.keywords.retain(|x| *x != k), cx))
                    }))
            })),
        )
        .when(keywords.is_empty(), |d| d.child(div().px_3().text_xs().text_color(p.text_subtle).child(t("No keywords", "暂无关键词"))))
}
