//! Output tab (process recipes and summary) and Batch tab (the processing queue).

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Disableable as _, Icon, Sizable as _,
    button::{Button, ButtonVariants as _},
    checkbox::Checkbox,
    h_flex, v_flex,
};
use gpui_kit::{
    AnyElement, App, Entity, InteractiveElement, IntoElement, ParentElement, PathPromptOptions,
    SharedString, StatefulInteractiveElement, Styled, Window, div, prelude::FluentBuilder as _, px,
};
use lumen_engine::*;

use crate::i18n::t;
use crate::state::AppState;
use crate::theme::pal;
use crate::widgets::*;

pub fn render(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    v_flex()
        .w_full()
        .child(recipes_panel(state, cx))
        .child(recipe_panel(state, window, cx))
        .child(summary_panel(state, cx))
        .into_any_element()
}

pub fn render_batch(state: &Entity<AppState>, _window: &mut Window, cx: &mut App) -> AnyElement {
    v_flex().w_full().child(queue_panel(state, cx)).into_any_element()
}

fn edit_recipe(state: &Entity<AppState>, cx: &mut App, f: impl FnOnce(&mut ProcessRecipe)) {
    state.update(cx, |s, cx| {
        let i = s.selected_recipe;
        if let Some(r) = s.recipes.get_mut(i) {
            f(r);
        }
        cx.notify();
    });
}

fn recipes_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let recipes = s.recipes.clone();
    let selected = s.selected_recipe;
    let (st_add, st_dup, st_del) = (state.clone(), state.clone(), state.clone());
    ToolPanel::new(state, "recipes", t("Process Recipes", "处理配方"))
        .badge(format!("{}/{}", recipes.iter().filter(|r| r.enabled).count(), recipes.len()))
        .children(recipes.iter().enumerate().map(|(i, r)| {
            let st = state.clone();
            let st2 = state.clone();
            h_flex()
                .id(SharedString::from(format!("recipe-{}", r.id)))
                .h(px(28.0))
                .mx_2()
                .px_2()
                .gap_2()
                .rounded(px(5.0))
                .cursor_pointer()
                .when(i == selected, |d| d.bg(p.accent_soft))
                .hover(|d| d.bg(p.hover))
                .on_click(move |_, _, cx| {
                    st.update(cx, |s, cx| {
                        s.selected_recipe = i;
                        cx.notify();
                    })
                })
                .child(Checkbox::new(SharedString::from(format!("recipe-on-{}", r.id))).xsmall().checked(r.enabled).on_change(move |v: &bool, _, cx| {
                    let v = *v;
                    st2.update(cx, |s, cx| {
                        if let Some(r) = s.recipes.get_mut(i) {
                            r.enabled = v;
                        }
                        cx.notify();
                    })
                }))
                .child(div().flex_1().text_xs().text_color(p.text).child(r.name.clone()))
                .child(div().px_1().rounded(px(3.0)).bg(p.well).text_xs().text_color(p.text_muted).child(r.format.extension().to_uppercase()))
        }))
        .child(
            h_flex()
                .px_3()
                .pt_1()
                .gap_1()
                .child(Button::new("recipe-add").xsmall().outline().icon(Icon::new(IconName::Plus)).label(t("New", "新建")).on_click(move |_, _, cx| {
                    st_add.update(cx, |s, cx| {
                        let id = s.recipes.iter().map(|r| r.id).max().unwrap_or(0) + 1;
                        s.recipes.push(ProcessRecipe { id, name: format!("{} {id}", t("Recipe", "配方")), ..Default::default() });
                        s.selected_recipe = s.recipes.len() - 1;
                        cx.notify();
                    })
                }))
                .child(Button::new("recipe-dup").xsmall().ghost().icon(Icon::new(IconName::Copy)).label(t("Duplicate", "复制")).on_click(move |_, _, cx| {
                    st_dup.update(cx, |s, cx| {
                        if let Some(r) = s.recipes.get(s.selected_recipe).cloned() {
                            let id = s.recipes.iter().map(|r| r.id).max().unwrap_or(0) + 1;
                            s.recipes.push(ProcessRecipe { id, name: format!("{} {}", r.name, t("copy", "副本")), ..r });
                            s.selected_recipe = s.recipes.len() - 1;
                        }
                        cx.notify();
                    })
                }))
                .child(div().flex_1())
                .child(IconBtn::new("recipe-del", IconName::Trash, t("Delete recipe", "删除配方")).size(22.0).disabled(recipes.len() <= 1).on_click(move |_, cx| {
                    st_del.update(cx, |s, cx| {
                        if s.recipes.len() > 1 {
                            let i = s.selected_recipe.min(s.recipes.len() - 1);
                            s.recipes.remove(i);
                            s.selected_recipe = i.saturating_sub(1);
                        }
                        cx.notify();
                    })
                })),
        )
}

fn scale_name(m: ScaleMode) -> &'static str {
    match m {
        ScaleMode::Fixed100 => t("Fixed 100%", "固定 100%"),
        ScaleMode::Width => t("Width", "宽度"),
        ScaleMode::Height => t("Height", "高度"),
        ScaleMode::LongEdge => t("Long Edge", "长边"),
        ScaleMode::ShortEdge => t("Short Edge", "短边"),
        ScaleMode::Dimensions => t("Dimensions", "尺寸"),
        ScaleMode::Percent => t("Percent", "百分比"),
    }
}

fn recipe_panel(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let Some(r) = s.recipes.get(s.selected_recipe).cloned() else {
        return div().into_any_element();
    };
    let profiles = s.icc_profiles.clone();
    let rid = r.id;
    let st = state.clone();
    let slider = |id: &'static str, label: &'static str, value: f32, min: f32, max: f32, f: fn(&mut ProcessRecipe, f32)| {
        let st = state.clone();
        ParamSlider::new(format!("{id}-{rid}"), label, value, min, max).step(1.0).on_change(move |v, _, cx| edit_recipe(&st, cx, |r| f(r, v)))
    };
    let name_field = text_field(SharedString::from(format!("recipe-name-{rid}")), r.name.clone(), "", {
        let st = state.clone();
        move |v, _, cx| edit_recipe(&st, cx, |r| r.name = v)
    }, window, cx);
    let naming_field = text_field(SharedString::from(format!("recipe-naming-{rid}")), r.file_name_format.clone(), "[Image Name]", {
        let st = state.clone();
        move |v, _, cx| edit_recipe(&st, cx, |r| r.file_name_format = v)
    }, window, cx);
    let dest = r.destination.as_ref().map(|d| d.display().to_string()).unwrap_or_else(|| t("Output folder next to originals", "原图旁的 Output 文件夹").into());
    let st_dest = state.clone();
    // Policies the exporter does not implement read as EXIF.
    let md = if r.include_metadata.is_implemented() { r.include_metadata } else { MetadataPolicy::Exif };
    let icc_idx = profiles.iter().position(|p| *p == r.icc_profile);

    ToolPanel::new(state, "recipe", t("Basic", "基本"))
        .badge(r.name.clone())
        .child(row(t("Name", "名称"), name_field, cx))
        .child(row(
            t("Format", "格式"),
            select(format!("fmt-{rid}"), ExportFormat::ALL.iter().map(|f| (*f, crate::contract::format_label(*f).into())).collect(), r.format, move |f, _, cx| edit_recipe(&st, cx, |r| r.format = f)),
            cx,
        ))
        .when(r.format.has_quality(), |d| d.child(slider("quality", t("Quality", "质量"), r.quality as f32, 1.0, 100.0, |r, v| r.quality = v.round() as u8)))
        .when(r.format.has_bit_depth(), |d| {
            let st = state.clone();
            d.child(row(
                t("Bit Depth", "位深"),
                Segmented::new(format!("bits-{rid}"), if r.bit_depth == 16 { 1 } else { 0 }).text("8-bit").text("16-bit").on_select(move |i, _, cx| {
                    edit_recipe(&st, cx, |r| r.bit_depth = if i == 1 { 16 } else { 8 })
                }),
                cx,
            ))
        })
        .child(row(
            t("ICC Profile", "ICC 配置"),
            select(format!("ricc-{rid}"), std::iter::once((None, t("Default", "默认").into())).chain(profiles.iter().enumerate().map(|(i, p)| (Some(i), p.clone().into()))).collect(), icc_idx, {
                let st = state.clone();
                move |i, _, cx| {
                    let name = i.and_then(|i| st.read(cx).icc_profiles.get(i).cloned()).unwrap_or_default();
                    edit_recipe(&st, cx, |r| r.icc_profile = name)
                }
            }),
            cx,
        ))
        .child(slider("ppi", t("Resolution", "分辨率"), r.resolution_ppi as f32, 72.0, 600.0, |r, v| r.resolution_ppi = v.round() as u32).unit(" ppi"))
        .child(row(
            t("Scale", "缩放"),
            select(format!("scale-{rid}"), ScaleMode::ALL.iter().map(|m| (*m, scale_name(*m).into())).collect(), r.scale, {
                let st = state.clone();
                move |m, _, cx| {
                    edit_recipe(&st, cx, |r| {
                        r.scale = m;
                        if r.scale_value.0 == 0 {
                            r.scale_value = if m == ScaleMode::Percent { (50, 0) } else { (2048, 2048) };
                        }
                    })
                }
            }),
            cx,
        ))
        .when(r.scale != ScaleMode::Fixed100, |d| {
            let (min, max, unit) = if r.scale == ScaleMode::Percent { (1.0, 200.0, "%") } else { (256.0, 12000.0, " px") };
            d.child(slider("scale-v", t("Size", "尺寸"), r.scale_value.0 as f32, min, max, |r, v| r.scale_value.0 = v.round() as u32).unit(unit))
        })
        .child(row(
            t("Sharpening", "输出锐化"),
            select(
                format!("osharp-{rid}"),
                vec![
                    (OutputSharpening::Disabled, t("Disabled", "关闭").into()),
                    (OutputSharpening::Screen, t("For Screen", "用于屏幕").into()),
                    (OutputSharpening::Print, t("For Print", "用于打印").into()),
                ],
                r.output_sharpening,
                {
                    let st = state.clone();
                    move |v, _, cx| edit_recipe(&st, cx, |r| r.output_sharpening = v)
                },
            ),
            cx,
        ))
        .child(caption(t("File Name", "文件名"), cx))
        .child(row(t("Format", "格式"), naming_field, cx))
        .child(caption(t("Location", "位置"), cx))
        .child(
            h_flex()
                .px_3()
                .gap_2()
                .child(div().flex_1().text_xs().text_color(pal(cx).text_muted).overflow_hidden().whitespace_nowrap().text_ellipsis().child(dest))
                .child(Button::new(SharedString::from(format!("dest-{rid}"))).xsmall().outline().icon(Icon::new(IconName::FolderOpen)).label(t("Choose…", "选择…")).on_click(move |_, _, cx| {
                    let rx = cx.prompt_for_paths(PathPromptOptions { files: false, directories: true, multiple: false, prompt: None });
                    let st = st_dest.clone();
                    cx.spawn(async move |cx| {
                        if let Ok(Ok(Some(paths))) = rx.await
                            && let Some(dir) = paths.into_iter().next()
                        {
                            cx.update(|cx| edit_recipe(&st, cx, |r| r.destination = Some(dir)));
                        }
                    })
                    .detach();
                })),
        )
        .child(caption(t("Metadata", "元数据"), cx))
        .child(row(
            t("Metadata", "元数据"),
            {
                let st = state.clone();
                select(
                    format!("md-{rid}"),
                    METADATA_POLICIES.iter().filter(|(m, ..)| m.is_implemented()).map(|(m, en, zh)| (*m, SharedString::from(t(en, zh)))).collect(),
                    md,
                    move |v, _, cx| edit_recipe(&st, cx, |r| r.include_metadata = v),
                )
            },
            cx,
        ))
        .child(div().px_3().pb_1().text_xs().text_color(pal(cx).text_subtle).child(metadata_note(md)))

        .child(caption(t("Watermark", "水印"), cx))
        .child({
            let st = state.clone();
            switch_row(format!("wm-{rid}"), t("Add text watermark", "添加文字水印"), r.watermark.is_some(), move |v, _, cx| {
                edit_recipe(&st, cx, |r| r.watermark = v.then(|| Watermark { text: "© Aurora Studio".into(), opacity: 60.0, scale: 1.0, position: [0.95, 0.95] }))
            }, cx)
        })
        .into_any_element()
}

fn summary_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let images = s.selection.len();
    let recipes: Vec<ProcessRecipe> = s.recipes.iter().filter(|r| r.enabled).cloned().collect();
    let pixels: f64 = s.selection.iter().filter_map(|id| s.info(*id)).map(|i| i.width as f64 * i.height as f64).sum();
    let est_mb: f64 = recipes
        .iter()
        .map(|r| {
            let bpp = match r.format {
                ExportFormat::Tiff | ExportFormat::Psd => 3.0 * r.bit_depth as f64 / 8.0,
                ExportFormat::Png => 1.6,
                ExportFormat::Dng => 2.0,
                _ => 0.25 * r.quality as f64 / 90.0,
            };
            pixels * bpp / 1e6
        })
        .sum();
    let st = state.clone();
    ToolPanel::new(state, "summary", t("Process Summary", "处理汇总"))
        .child(
            v_flex()
                .px_3()
                .gap_1()
                .text_xs()
                .text_color(p.text_muted)
                .child(format!("{images} {} × {} {}", t("images", "张图像"), recipes.len(), t("recipes", "个配方")))
                .child(format!("{} ≈ {est_mb:.0} MB", t("Estimated size", "预计大小")))
                .children(recipes.iter().map(|r| div().text_color(p.text_subtle).child(format!("• {} — {}", r.name, r.format.extension().to_uppercase())))),
        )
        .child(
            div().px_3().pt_2().child(
                Button::new("process")
                    .primary()
                    .w_full()
                    .icon(Icon::new(IconName::Upload))
                    .label(format!("{} {images} {}", t("Process", "处理"), t("images", "张图像")))
                    .disabled(images == 0 || recipes.is_empty())
                    .on_click(move |_, _, cx| st.update(cx, |s, cx| s.process_selection(cx))),
            ),
        )
}

fn queue_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let jobs = s.jobs.clone();
    let active = jobs.iter().filter(|j| matches!(j.status, JobStatus::Queued | JobStatus::Running { .. })).count();
    let st = state.clone();
    ToolPanel::new(state, "queue", t("Queue", "队列"))
        .badge(format!("{active}/{}", jobs.len()))
        .on_reset(move |_, cx| {
            st.update(cx, |s, cx| {
                s.jobs.retain(|j| matches!(j.status, JobStatus::Queued | JobStatus::Running { .. }));
                cx.notify();
            })
        })
        .when(jobs.is_empty(), |d| {
            d.child(div().p_3().child(ely_core::feedback::EmptyState::new("queue-empty", ely_core::primitives::IconName::Inbox, t("Nothing to process", "没有待处理的任务"))))
        })
        .children(jobs.iter().rev().map(|j| {
            let (progress, label, color, icon) = match &j.status {
                JobStatus::Queued => (0.0, t("Queued", "排队中").to_string(), p.text_subtle, IconName::Clock),
                JobStatus::Running { progress } => (*progress, format!("{:.0}%", progress * 100.0), p.accent, IconName::LoaderCircle),
                JobStatus::Done { output } => (1.0, output.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), p.success, IconName::CircleCheck),
                JobStatus::Failed { message } => (1.0, message.clone(), p.danger, IconName::CircleX),
            };
            v_flex()
                .px_3()
                .py_1p5()
                .gap_1()
                .border_b_1()
                .border_color(p.divider)
                .child(
                    h_flex()
                        .gap_2()
                        .child(Icon::new(icon).xsmall().text_color(color))
                        .child(div().flex_1().text_xs().text_color(p.text).overflow_hidden().whitespace_nowrap().text_ellipsis().child(j.name.clone()))
                        .child(div().text_xs().text_color(p.text_subtle).child(j.recipe.clone()))
                        .when(matches!(j.status, JobStatus::Queued | JobStatus::Running { .. }), |d| {
                            let (st, id) = (state.clone(), j.id);
                            let ready = crate::planned::CANCEL_READY;
                            d.child(
                                IconBtn::new(SharedString::from(format!("job-cancel-{id}")), IconName::X, if ready { t("Cancel", "取消") } else { t("Cancel · the engine cannot stop an export yet", "取消 · 引擎尚不能中止导出") })
                                    .size(18.0)
                                    .disabled(!ready)
                                    .on_click(move |_, cx| st.update(cx, |s, cx| s.cancel_job(id, cx))),
                            )
                        }),
                )
                .child(ely_core::motion::ProgressBar::new(SharedString::from(format!("job-{}", j.id)), progress))
                .child(div().text_xs().text_color(color).overflow_hidden().whitespace_nowrap().text_ellipsis().child(label))
        }))
}

/// Every metadata policy the recipe can carry, in menu order.
const METADATA_POLICIES: [(MetadataPolicy, &str, &str); 7] = [
    (MetadataPolicy::Exif, "Copy source EXIF", "保留原 EXIF"),
    (MetadataPolicy::StripAll, "Strip all", "全部移除"),
    (MetadataPolicy::Copyright, "Copyright only", "仅版权"),
    (MetadataPolicy::Camera, "Camera and exposure only", "仅相机与曝光"),
    (MetadataPolicy::Gps, "GPS only", "仅 GPS"),
    (MetadataPolicy::Keywords, "Keywords only", "仅关键词"),
    (MetadataPolicy::RatingAndTag, "Rating and color tag only", "仅星级与色标"),
];

/// What the chosen policy writes, from the exporter's own description.
fn metadata_note(md: MetadataPolicy) -> &'static str {
    match md {
        MetadataPolicy::Exif => t("EXIF is copied from JPEG sources; RAW, TIFF and PNG sources export without it.", "EXIF 从 JPEG 源文件复制；RAW、TIFF、PNG 源文件导出时不带 EXIF。"),
        MetadataPolicy::StripAll => t("No metadata is written; the output color profile is still embedded.", "不写入任何元数据；输出色彩配置文件仍会嵌入。"),
        MetadataPolicy::Copyright => t("Only the copyright field is written.", "只写入版权字段。"),
        MetadataPolicy::Camera => t("Only camera make and model, lens and exposure settings are written.", "只写入相机品牌型号、镜头与曝光参数。"),
        MetadataPolicy::Gps => t("Only the GPS location is written.", "只写入 GPS 位置。"),
        MetadataPolicy::Keywords => t("Only keywords are written, as XMP.", "只以 XMP 写入关键词。"),
        MetadataPolicy::RatingAndTag => t("Only the rating and color tag are written, as XMP.", "只以 XMP 写入星级与色标。"),
    }
}
