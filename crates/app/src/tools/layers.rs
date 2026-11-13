//! Nodes tab: the node editor, the selected node's masks (brush, gradients,
//! AI, luma range) and brush settings.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Disableable as _, Icon, Sizable as _, WindowExt as _,
    button::{Button, ButtonVariants as _},
    h_flex,
    input::{Input, InputState},
    menu::{DropdownMenu as _, PopupMenuItem},
    spinner::Spinner,
    v_flex,
};
use gpui_kit::{
    AnyElement, App, AppContext as _, Entity, Focusable as _, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, prelude::FluentBuilder as _, px,
};
use lumen_engine::*;

use crate::i18n::t;
use crate::layer_field;
use crate::state::{AppState, CursorTool};
use crate::theme::pal;
use crate::widgets::*;

pub fn render(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    let layer = state.read(cx).active_layer_ref().cloned();
    v_flex()
        .w_full()
        .child(node_panel(state, window, cx))
        .when_some(layer, |d, l| d.child(mask_panel(state, &l, cx)).child(luma_panel(state, &l, cx)).child(color_range_panel(state, &l, cx)).child(brush_panel(state, cx)))
        .into_any_element()
}

pub(crate) fn rename(state: &Entity<AppState>, id: u64, current: String, window: &mut Window, cx: &mut App) {
    let input = cx.new(|cx| InputState::new(window, cx).default_value(current));
    let st = state.clone();
    window.open_dialog(cx, move |dialog, window, cx| {
        let input2 = input.clone();
        let st = st.clone();
        input.read(cx).focus_handle(cx).focus(window, cx);
        dialog
            .title(t("Rename node", "重命名节点"))
            .w(px(360.0))
            .child(Input::new(&input))
            .footer(
                h_flex().gap_2().justify_end().child(Button::new("rename-ok").primary().label(t("Rename", "重命名")).on_click(move |_, window, cx| {
                    let name = input2.read(cx).value().to_string();
                    st.update(cx, |s, cx| s.edit_layer(t("Rename node", "重命名节点"), id, false, |l| l.name = name, cx));
                    window.close_dialog(cx);
                })),
            )
    });
}

/// The node editor (or a note when it is docked under the viewer) and the
/// selected node's switch, name and key opacity.
fn node_panel(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let docked = s.view.node_dock;
    let count = s.adj().layers.len();
    let sel = s.selected_node();
    let graph = s.node_graph();
    let kind = graph.node(sel).map(|n| n.kind);
    let layer = s.active_layer_ref().cloned();
    let number = graph.render_order().iter().position(|i| Some(*i) == layer.as_ref().map(|l| l.id)).map(|i| i + 1);
    let editor = if docked {
        h_flex()
            .mx_2()
            .h(px(36.0))
            .px_3()
            .gap_2()
            .rounded(px(5.0))
            .bg(p.well)
            .text_xs()
            .text_color(p.text_muted)
            .child(Icon::new(IconName::PanelBottom).xsmall())
            .child(t("The node editor is docked under the viewer", "节点编辑器已停靠在查看器下方"))
            .into_any_element()
    } else {
        div().px_2().child(super::nodes::editor(state, "node-editor", 220.0, window, cx)).into_any_element()
    };
    let note = |text: &'static str| div().px_3().pt_2().text_xs().text_color(p.text_subtle).child(text).into_any_element();
    let selected = match (kind, layer) {
        (Some(NodeKind::Layer), Some(l)) => {
            let id = l.id;
            let st_eye = state.clone();
            let st_name = state.clone();
            let name = l.name.clone();
            v_flex()
                .pt_2()
                .child(
                    h_flex()
                        .px_2()
                        .gap_2()
                        .child(
                            IconBtn::new(SharedString::from(format!("node-on-{id}")), if l.enabled { IconName::Power } else { IconName::PowerOff }, t("Enable / disable node", "启用 / 停用节点"))
                                .shortcut(crate::platform::keys("ctrl-shift-e"))
                                .active(l.enabled)
                                .size(22.0)
                                .on_click(move |_, cx| st_eye.update(cx, |s, cx| s.toggle_node(id, cx))),
                        )
                        .child(div().text_xs().text_color(p.accent).child(number.map(|i| format!("{i:02}")).unwrap_or_else(|| "—".into())))
                        .child(
                            div()
                                .id("node-name")
                                .flex_1()
                                .text_xs()
                                .text_color(p.text)
                                .truncate()
                                .cursor_pointer()
                                .child(l.name.clone())
                                .on_click(move |_, window, cx| rename(&st_name, id, name.clone(), window, cx)),
                        )
                        .child(div().text_xs().text_color(p.text_subtle).child(match l.kind {
                            LayerKind::Adjustment => t("Corrector", "校正器"),
                            LayerKind::Heal => t("Heal", "修复"),
                            LayerKind::Clone => t("Clone", "仿制"),
                            LayerKind::Fill => t("Fill", "填色"),
                        })),
                )
                .when(number.is_none(), |d| d.child(note(t("Not connected: this node does not change the image.", "未连接：此节点不影响图像。"))))
                .child(div().h(px(4.0)))
                .child(param_c1(state, format!("layer-opacity-{id}"), "LayerOpacity", layer_field!(id, opacity), cx))
                .into_any_element()
        }
        (Some(NodeKind::Parallel), _) => note(t("Parallel mixer: blends branches that grade the same image.", "并行混合器：混合对同一图像调色的各分支。")),
        (Some(NodeKind::LayerMixer), _) => note(t("Layer mixer: stacks its inputs, the lowest on top.", "图层混合器：叠加各输入，越靠下的越在上层。")),
        (Some(NodeKind::Output), _) => note(t("Output: what reaches it is the image.", "输出：到达这里的就是最终图像。")),
        _ => note(t("Base: the global adjustments from the other tabs. Select a node to grade it.", "基础：其他工具页的全局调整。选中一个节点即可单独调色。")),
    };
    ToolPanel::new(state, "nodes", t("Nodes", "节点"))
        .status(crate::contract::NODE_GRAPH)
        .badge(format!("{count}"))
        .child(super::nodes::toolbar(state, docked, cx))
        .child(editor)
        .child(selected)
}

fn ai_name(k: AiMaskKind) -> &'static str {
    match k {
        AiMaskKind::Subject => t("Subject", "主体"),
        AiMaskKind::Background => t("Background", "背景"),
        AiMaskKind::Sky => t("Sky", "天空"),
        AiMaskKind::Person => t("Person", "人物"),
        AiMaskKind::Skin => t("Skin", "皮肤"),
        AiMaskKind::Hair => t("Hair", "头发"),
        AiMaskKind::Clothing => t("Clothing", "服装"),
        AiMaskKind::Eyes => t("Eyes", "眼睛"),
        AiMaskKind::Lips => t("Lips", "嘴唇"),
    }
}

fn mask_panel(state: &Entity<AppState>, layer: &Layer, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let id = layer.id;
    let cursor = s.cursor;
    let overlay = s.panels.mask_overlay;
    let busy = s.ai_busy;
    let mask = layer.mask.clone();

    let tool = |tool: CursorTool| {
        let st = state.clone();
        IconBtn::new(SharedString::from(format!("mask-tool-{tool:?}")), tool.icon(), tool.label())
            .shortcut(tool.key())
            .active(cursor == tool)
            .on_click(move |_, cx| {
                st.update(cx, |s, cx| {
                    let next = if s.cursor == tool { CursorTool::Select } else { tool };
                    s.set_cursor(next, cx)
                })
            })
    };
    let cmd = |key: &'static str, icon: IconName, tip: &'static str, f: fn(&mut Mask)| {
        let st = state.clone();
        IconBtn::new(SharedString::from(format!("mask-{key}")), icon, tip).on_click(move |_, cx| st.update(cx, |s, cx| s.edit_layer(tip, id, false, |l| f(&mut l.mask), cx)))
    };
    let st_overlay = state.clone();
    let st_ai = state.clone();

    let (ai_busy, ai_failed) = {
        let s = state.read(cx);
        (s.ai_busy, s.ai_unavailable.clone())
    };
    let ai_menu = Button::new("ai-mask")
        .xsmall()
        .outline()
        .icon(Icon::new(IconName::Sparkles))
        .label(t("AI Mask", "AI 蒙版"))
        .loading(ai_busy)
        .disabled(ai_busy)
        .tooltip(crate::contract::AI_MASK.note().unwrap_or_default())
        .dropdown_menu(move |mut menu, _, _| {
            for k in AiMaskKind::ALL {
                let st = st_ai.clone();
                // A kind the engine refused this session stays listed, marked, so the
                // reason can be read again by choosing it.
                let label = match ai_failed.contains(&k) {
                    true => format!("{} · {}", ai_name(k), t("unavailable", "不可用")),
                    false => ai_name(k).to_string(),
                };
                menu = menu.item(PopupMenuItem::new(label).on_click(move |_, _, cx| st.update(cx, |s, cx| s.compute_ai_mask(k, cx))));
            }
            menu
        });

    let shapes = mask.shapes.iter().enumerate().map(|(i, shape)| {
        let (icon, label) = match shape {
            MaskShape::Brush(b) => (if b.op == MaskOp::Add { IconName::Brush } else { IconName::Eraser }, format!("{} · {} {}", t("Brush", "画笔"), b.points.len(), t("pts", "点"))),
            MaskShape::Linear(_) => (IconName::Blend, t("Linear gradient", "线性渐变").to_string()),
            MaskShape::Radial(_) if crate::state::is_full(&mask) => (IconName::Square, t("Whole frame", "整幅画面").to_string()),
            MaskShape::Radial(_) => (IconName::CircleDot, t("Radial gradient", "径向渐变").to_string()),
            MaskShape::Ai(a) => (IconName::Sparkles, format!("AI · {}", ai_name(a.kind))),
            MaskShape::Copy { .. } => (IconName::Copy, t("Copied mask", "复制的蒙版").to_string()),
        };
        let st = state.clone();
        h_flex()
            .h(px(22.0))
            .px_3()
            .gap_2()
            .text_xs()
            .text_color(p.text_muted)
            .child(Icon::new(icon).xsmall())
            .child(div().flex_1().child(label))
            .child(IconBtn::new(SharedString::from(format!("shape-del-{id}-{i}")), IconName::X, t("Remove", "移除")).size(18.0).on_click(move |_, cx| {
                st.update(cx, |s, cx| {
                    s.edit_layer(t("Remove mask shape", "移除蒙版形状"), id, false, |l| {
                        if i < l.mask.shapes.len() {
                            l.mask.shapes.remove(i);
                        }
                    }, cx)
                })
            }))
    });

    ToolPanel::new(state, "mask", t("Mask", "蒙版"))
        .status(crate::contract::LOCAL_ADJUSTMENTS)
        .modified(!mask.shapes.is_empty() || mask.invert)
        .child(
            h_flex()
                .px_2()
                .gap_0p5()
                .child(tool(CursorTool::Brush))
                .child(tool(CursorTool::Eraser))
                .child(tool(CursorTool::LinearGradient))
                .child(tool(CursorTool::RadialGradient))
                .child(div().w(px(1.0)).h(px(18.0)).mx_1().bg(p.divider))
                .child(cmd("fill", IconName::PaintBucket, t("Fill mask", "填充蒙版"), |m| {
                    m.shapes = vec![MaskShape::Radial(RadialGradient { op: MaskOp::Add, center: [0.5, 0.5], radius: [2.0, 2.0], rotation: 0.0, feather: 0.0, inverted: false })]
                }))
                .child(cmd("clear", IconName::Eraser, t("Clear mask", "清除蒙版"), |m| m.shapes.clear()))
                .child(cmd("invert", IconName::Contrast, t("Invert mask", "反相蒙版"), |m| m.invert = !m.invert))
                .child(div().flex_1())
                .child(
                    IconBtn::new("mask-overlay", IconName::ScanEye, t("Display mask", "显示蒙版"))
                        .shortcut("M")
                        .active(overlay)
                        .on_click(move |_, cx| {
                            st_overlay.update(cx, |s, cx| {
                                s.panels.mask_overlay = !s.panels.mask_overlay;
                                s.request_render(RenderQuality::Final, cx);
                                cx.notify();
                            })
                        }),
                ),
        )
        .child(h_flex().px_3().py_1().gap_2().child(ai_menu).when(busy, |d| {
            d.child(h_flex().gap_1().text_xs().text_color(p.text_muted).child(Spinner::new().xsmall()).child(t("Analyzing…", "正在分析…")))
        }))
        .children(shapes)
        .child(div().h(px(4.0)))
        .child(param(state, format!("mask-feather-{id}"), t("Feather", "羽化"), layer_field!(id, mask.feather), 0.0, 1000.0, cx).unit(" px"))
        .child(param(state, format!("mask-refine-{id}"), t("Refine Edge", "精修边缘"), layer_field!(id, mask.refine), 0.0, 100.0, cx))
        .child(param(state, format!("mask-density-{id}"), t("Density", "密度"), layer_field!(id, mask.density), 0.0, 100.0, cx).default_value(100.0).unit("%"))
}

fn luma_panel(state: &Entity<AppState>, layer: &Layer, cx: &App) -> impl IntoElement + use<> {
    let id = layer.id;
    let range = layer.mask.luma_range;
    let st = state.clone();
    let field = |label: &'static str, key: &'static str, get: fn(&LumaRange) -> f32, set: fn(&mut LumaRange, f32)| {
        let binding = crate::state::Binding::new(
            move |a: &AdjustmentSet| a.layers.iter().find(|l| l.id == id).and_then(|l| l.mask.luma_range).map(|r| get(&r) * 255.0).unwrap_or_default(),
            move |a: &mut AdjustmentSet, v: f32| {
                if let Some(l) = a.layers.iter_mut().find(|l| l.id == id) {
                    set(l.mask.luma_range.get_or_insert_default(), v / 255.0);
                }
            },
        );
        param(state, format!("luma-{key}-{id}"), label, binding, 0.0, 255.0, cx).disabled(range.is_none())
    };
    ToolPanel::new(state, "luma", t("Luma Range", "亮度范围"))
        .modified(range.is_some())
        .child(switch_row(format!("luma-on-{id}"), t("Restrict mask to luma range", "按亮度范围限制蒙版"), range.is_some(), move |v, _, cx| {
            st.update(cx, |s, cx| s.edit_layer(t("Luma range", "亮度范围"), id, false, |l| l.mask.luma_range = v.then(LumaRange::default), cx))
        }, cx))
        .child(field(t("Shadow Falloff", "阴影衰减"), "lo", |r| r.low_outer, |r, v| r.low_outer = v.min(r.low_inner)))
        .child(field(t("Shadows", "阴影"), "li", |r| r.low_inner, |r, v| r.low_inner = v.clamp(r.low_outer, r.high_inner)))
        .child(field(t("Highlights", "高光"), "hi", |r| r.high_inner, |r, v| r.high_inner = v.clamp(r.low_inner, r.high_outer)))
        .child(field(t("Highlight Falloff", "高光衰减"), "ho", |r| r.high_outer, |r, v| r.high_outer = v.max(r.high_inner)))
}

/// Restricts the mask to a hue band, as Capture One's color range masks do.
fn color_range_panel(state: &Entity<AppState>, layer: &Layer, cx: &App) -> impl IntoElement + use<> {
    let id = layer.id;
    let range = layer.mask.color_range.clone();
    let off = range.is_none();
    let st = state.clone();
    let field = |label: &'static str, key: &'static str, min: f32, max: f32, get: fn(&ColorRange) -> f32, set: fn(&mut ColorRange, f32)| {
        let binding = crate::state::Binding::new(
            move |a: &AdjustmentSet| a.layers.iter().find(|l| l.id == id).and_then(|l| l.mask.color_range.as_ref()).map(get).unwrap_or_default(),
            move |a: &mut AdjustmentSet, v: f32| {
                if let Some(l) = a.layers.iter_mut().find(|l| l.id == id) {
                    set(l.mask.color_range.get_or_insert_default(), v);
                }
            },
        );
        param(state, format!("mcr-{key}-{id}"), label, binding, min, max, cx).disabled(off)
    };
    ToolPanel::new(state, "mask-color", t("Color Range", "颜色范围"))
        .modified(range.is_some())
        .child(switch_row(format!("mcr-on-{id}"), t("Restrict mask to a color range", "按颜色范围限制蒙版"), range.is_some(), move |v, _, cx| {
            st.update(cx, |s, cx| s.edit_layer(t("Color range", "颜色范围"), id, false, |l| l.mask.color_range = v.then(ColorRange::default), cx))
        }, cx))
        .when_some(range, |d, r| d.child(div().px_3().pb_1().child(crate::tools::color::hue_range_bar(&r, cx))))
        .child(field(t("Center", "中心"), "hue", 0.0, 360.0, |r| r.hue, |r, v| r.hue = v).unit("°").track(Track::Gradient(crate::tools::color::hue_stops())))
        .child(field(t("Range", "范围"), "w", 5.0, 180.0, |r| r.width, |r, v| r.width = v).unit("°").default_value(40.0))
        .child(field(t("Smoothness", "平滑度"), "sm", 0.0, 60.0, |r| r.smoothness, |r, v| r.smoothness = v).default_value(15.0))
        .child(field(t("Min. Saturation", "最低饱和度"), "ms", 0.0, 1.0, |r| r.min_saturation, |r, v| r.min_saturation = v).decimals(2))
}

fn brush_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let pr = state.read(cx).panels.clone();
    let slider = |id: &'static str, label: &'static str, value: f32, min: f32, max: f32, default: f32, unit: &'static str, f: fn(&mut crate::state::PanelPrefs, f32)| {
        let st = state.clone();
        ParamSlider::new(id, label, value, min, max).default_value(default).unit(unit).on_change(move |v, _, cx| {
            st.update(cx, |s, cx| {
                f(&mut s.panels, v);
                cx.notify();
            })
        })
    };
    let st = state.clone();
    ToolPanel::new(state, "brush", t("Brush Settings", "画笔设置"))
        .child(slider("brush-size", t("Size", "大小"), pr.brush_size * 100.0, 0.5, 40.0, 5.0, "%", |p, v| p.brush_size = v / 100.0).decimals(1))
        .child(slider("brush-hard", t("Hardness", "硬度"), pr.brush_hardness, 0.0, 100.0, 50.0, "", |p, v| p.brush_hardness = v))
        .child(slider("brush-opacity", t("Opacity", "不透明度"), pr.brush_opacity, 1.0, 100.0, 100.0, "%", |p, v| p.brush_opacity = v))
        .child(slider("brush-flow", t("Flow", "流量"), pr.brush_flow, 1.0, 100.0, 100.0, "%", |p, v| p.brush_flow = v))
        .child(switch_row("brush-auto", t("Auto Mask", "自动蒙版"), pr.brush_auto_mask, move |v, _, cx| {
            st.update(cx, |s, cx| {
                s.panels.brush_auto_mask = v;
                cx.notify();
            })
        }, cx))
}
