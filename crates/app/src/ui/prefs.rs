//! Preferences: a paged window (General, Appearance, Browser, Exposure, Crop,
//! Workspaces, Keyboard). Every change applies immediately and is saved to the
//! settings file; the shortcut editor records keys in place and flags conflicts.

use std::collections::BTreeMap;

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, Sizable as _, WindowExt as _,
    button::{Button, ButtonVariants as _},
    h_flex,
    scroll::ScrollableElement as _,
    tooltip::Tooltip,
    v_flex,
};
use gpui_kit::{
    AnyElement, App, AppContext as _, Context, Entity, FocusHandle, Focusable, InteractiveElement, IntoElement, KeyDownEvent, ParentElement,
    Render, SharedString, StatefulInteractiveElement, Styled, Window, div, prelude::FluentBuilder as _, px,
};
use lumen_engine::RenderQuality;

use crate::actions::{self, display_keys, keys_for, shortcuts};
use crate::i18n::{self, Locale, t};
use crate::motion;
use crate::state::{AppState, CropGuide, Settings, ToolTab};
use crate::theme::{self, ACCENTS, hex, pal};
use crate::widgets::{ParamSlider, Segmented, caption, row, select, switch_row, text_field};

const PAGES: [(IconName, &str, &str); 7] = [
    (IconName::Settings, "General", "常规"),
    (IconName::Palette, "Appearance", "外观"),
    (IconName::LayoutGrid, "Browser & Viewer", "浏览器与查看器"),
    (IconName::TriangleAlert, "Exposure", "曝光警告"),
    (IconName::Crop, "Crop", "裁剪"),
    (IconName::LayoutPanelLeft, "Workspaces", "工作区"),
    (IconName::Keyboard, "Keyboard", "键盘快捷键"),
];

const NAV_ROW: f32 = 34.0;

/// Animation speeds offered in Appearance: (factor, label).
const SPEEDS: [(f32, &str, &str); 4] = [(0.0, "Off", "关闭"), (0.6, "Brisk", "轻快"), (1.0, "Natural", "自然"), (1.6, "Relaxed", "舒缓")];

pub struct PrefsView {
    state: Entity<AppState>,
    page: usize,
    /// Shortcut id waiting for a key press.
    recording: Option<String>,
    query: String,
    focus: FocusHandle,
}

impl Focusable for PrefsView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl PrefsView {
    fn edit(&self, cx: &mut Context<Self>, f: impl FnOnce(&mut AppState)) {
        self.state.update(cx, |s, cx| {
            f(s);
            s.save_settings();
            cx.notify();
        });
        cx.notify();
    }

    fn edit_render(&self, cx: &mut Context<Self>, f: impl FnOnce(&mut Settings)) {
        self.state.update(cx, |s, cx| {
            f(&mut s.settings);
            s.save_settings();
            s.request_render(RenderQuality::Final, cx);
            cx.notify();
        });
        cx.notify();
    }

    fn set_keys(&mut self, id: &str, keys: Option<Vec<String>>, cx: &mut Context<Self>) {
        let id = id.to_string();
        let overrides = self.state.update(cx, |s, _| {
            match keys {
                Some(k) => {
                    let default: Vec<String> = shortcuts().into_iter().find(|x| x.id == id).map(|x| x.defaults.iter().map(|k| k.to_string()).collect()).unwrap_or_default();
                    if k == default {
                        s.settings.shortcuts.remove(&id);
                    } else {
                        s.settings.shortcuts.insert(id, k);
                    }
                }
                None => {
                    s.settings.shortcuts.remove(&id);
                }
            }
            s.save_settings();
            s.settings.shortcuts.clone()
        });
        actions::bind_keys(cx, &overrides);
        cx.notify();
    }

    fn on_key(&mut self, e: &KeyDownEvent, _window: &mut Window, cx: &mut Context<Self>) {
        let Some(id) = self.recording.clone() else { return };
        cx.stop_propagation();
        let ks = &e.keystroke;
        let key = ks.key.to_lowercase();
        if matches!(key.as_str(), "shift" | "control" | "alt" | "platform" | "function" | "capslock") {
            return;
        }
        if key == "escape" && !ks.modifiers.modified() {
            self.recording = None;
            cx.notify();
            return;
        }
        let mut parts = Vec::new();
        if ks.modifiers.control {
            parts.push("ctrl".to_string());
        }
        if ks.modifiers.alt {
            parts.push("alt".to_string());
        }
        if ks.modifiers.shift {
            parts.push("shift".to_string());
        }
        if ks.modifiers.platform {
            parts.push("cmd".to_string());
        }
        parts.push(key);
        self.recording = None;
        self.set_keys(&id, Some(vec![parts.join("-")]), cx);
    }
}

impl Render for PrefsView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let p = pal(cx).clone();
        let y = motion::value(motion::key("prefs", "nav"), self.page as f32 * NAV_ROW, motion::GLIDE, window, cx);
        let nav = v_flex()
            .w(px(188.0))
            .flex_shrink_0()
            .pr_2()
            .border_r_1()
            .border_color(p.divider)
            .child(
                div()
                    .relative()
                    .child(div().absolute().left_0().right_0().top(px(y)).h(px(NAV_ROW)).rounded(px(6.0)).bg(p.accent_soft))
                    .children(PAGES.iter().enumerate().map(|(i, (icon, en, zh))| {
                        let selected = i == self.page;
                        h_flex()
                            .id(SharedString::from(format!("prefs-nav-{i}")))
                            .relative()
                            .h(px(NAV_ROW))
                            .px_2()
                            .gap_2()
                            .rounded(px(6.0))
                            .cursor_pointer()
                            .text_sm()
                            .text_color(if selected { p.text } else { p.text_muted })
                            .hover(|d| d.text_color(p.text))
                            .on_click(cx.listener(move |v, _, _, cx| {
                                v.page = i;
                                v.recording = None;
                                cx.notify();
                            }))
                            .child(Icon::new(*icon).small().text_color(if selected { p.accent } else { p.text_subtle }))
                            .child(t(en, zh))
                    })),
            );

        let page = match self.page {
            0 => self.general(window, cx),
            1 => self.appearance(cx),
            2 => self.browser(cx),
            3 => self.exposure(cx),
            4 => self.crop(cx),
            5 => self.workspaces(cx),
            _ => self.keyboard(window, cx),
        };
        let page_id = self.page;

        h_flex()
            .id("prefs")
            .track_focus(&self.focus)
            .capture_key_down(cx.listener(Self::on_key))
            .h(px(540.0))
            .items_start()
            .gap_3()
            .child(nav)
            .child(
                div()
                    .id("prefs-page")
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .overflow_y_scrollbar()
                    .child(motion::slide_in(SharedString::from(format!("prefs-page-{page_id}")), v_flex().w_full().pb_4().child(page), 16.0)),
            )
    }
}

fn note(text: &'static str, cx: &App) -> AnyElement {
    div().px_3().pb_1().text_xs().text_color(pal(cx).text_subtle).child(text).into_any_element()
}

impl PrefsView {
    fn general(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let s = self.state.read(cx).settings.clone();
        let this = cx.entity();
        let (t1, t2, t3, t4, t5) = (this.clone(), this.clone(), this.clone(), this.clone(), this.clone());
        v_flex()
            .child(caption(t("Language", "语言"), cx))
            .child(
                div().px_3().w(px(280.0)).child(
                    Segmented::new("pref-lang", if i18n::locale() == Locale::En { 0 } else { 1 }).text("English").text("简体中文").on_select(move |i, window, cx| {
                        i18n::set_locale(if i == 0 { Locale::En } else { Locale::ZhCn });
                        t1.update(cx, |v, cx| v.edit(cx, |_| {}));
                        window.refresh();
                    }),
                ),
            )
            .child(caption(t("Startup", "启动"), cx))
            .child(switch_row("pref-reopen", t("Reopen the last folder or catalog", "启动时打开上次的文件夹或目录"), s.reopen_last, move |v, _, cx| {
                t5.update(cx, |p, cx| p.edit(cx, |s| s.settings.reopen_last = v))
            }, cx))
            .child(note(t("Otherwise Aurora starts on the welcome screen.", "否则启动时显示欢迎页。"), cx))
            .child(caption(t("Editing", "编辑"), cx))
            .child(ParamSlider::new("pref-undo", t("Undo steps", "撤销步数"), s.undo_levels as f32, 20.0, 1000.0).step(10.0).default_value(200.0).on_change(move |v, _, cx| {
                t2.update(cx, |p, cx| p.edit(cx, |s| s.settings.undo_levels = v.round() as usize))
            }))
            .child(note(t("Per image; older steps are dropped first.", "按图像计；超出后最早的步骤先被丢弃。"), cx))
            .child(caption(t("Capture & Output", "拍摄与输出"), cx))
            .child(switch_row("pref-newcap", t("Select each new capture", "自动选中新拍摄的图像"), s.select_new_capture, move |v, _, cx| {
                t3.update(cx, |p, cx| p.edit(cx, |s| s.settings.select_new_capture = v))
            }, cx))
            .child(switch_row("pref-notify", t("Notify when processing finishes", "处理完成时通知"), s.notify_on_process, move |v, _, cx| {
                t4.update(cx, |p, cx| p.edit(cx, |s| s.settings.notify_on_process = v))
            }, cx))
            .child(caption(t("Reset", "复位"), cx))
            .child(
                h_flex().px_3().gap_2().child(Button::new("pref-reset-all").xsmall().outline().icon(Icon::new(IconName::RotateCcw)).label(t("Restore all defaults", "恢复全部默认设置")).on_click(cx.listener(
                    |v, _, window, cx| {
                        v.state.update(cx, |s, cx| {
                            // The recent list is history, not a preference.
                            let recent = std::mem::take(&mut s.settings.recent);
                            s.settings = Settings { recent, ..Settings::default() };
                            s.view.viewer_gray = s.settings.viewer_gray;
                            s.view.thumb_size = s.settings.thumb_size;
                            s.view.show_info_overlay = s.settings.info_overlay;
                            s.apply_workspace(0, cx);
                        });
                        i18n::set_locale(Locale::ZhCn);
                        motion::set_speed(1.0);
                        theme::set_accent(ACCENTS[0].0, cx);
                        actions::bind_keys(cx, &BTreeMap::new());
                        v.edit(cx, |_| {});
                        window.refresh();
                    },
                ))),
            )
            .into_any_element()
    }

    fn appearance(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        let gray = s.view.viewer_gray;
        let info = s.view.show_info_overlay;
        let current_accent = cx.global::<theme::AuroraTheme>().accent;
        let speed = motion::speed();
        let speed_index = SPEEDS.iter().position(|(f, _, _)| (f - speed).abs() < 0.05).unwrap_or(2);
        let this = cx.entity();
        let (t1, t2, t3, t4) = (this.clone(), this.clone(), this.clone(), this.clone());
        v_flex()
            .child(caption(t("Accent color", "强调色"), cx))
            .child(h_flex().px_3().gap_2().children(ACCENTS.iter().map(|(c, name)| {
                let c = *c;
                let selected = c == current_accent;
                let t1 = t1.clone();
                div()
                    .id(SharedString::from(format!("accent-{name}")))
                    .size(px(26.0))
                    .rounded_full()
                    .bg(hex(c))
                    .border_2()
                    .border_color(if selected { p.text } else { p.panel })
                    .cursor_pointer()
                    .hover(|d| d.border_color(p.text_muted))
                    .on_click(move |_, _, cx| {
                        theme::set_accent(c, cx);
                        t1.update(cx, |v, cx| v.edit(cx, |s| s.settings.accent = Some(c)));
                    })
            })))
            .child(caption(t("Motion", "动效"), cx))
            .child(
                div().px_3().w(px(360.0)).child(
                    SPEEDS
                        .iter()
                        .fold(Segmented::new("pref-speed", speed_index), |seg, (_, en, zh)| seg.text(t(en, zh)))
                        .on_select(move |i, _, cx| {
                            motion::set_speed(SPEEDS[i].0);
                            t2.update(cx, |v, cx| v.edit(cx, |_| {}));
                        }),
                ),
            )
            .child(note(
                t(
                    "Panels, tiles and the loupe settle like physical objects; values and zoom never overshoot. Off removes all animation.",
                    "面板、平铺块与放大镜像实物一样自然落定；数值与缩放从不过冲。关闭则去除全部动画。",
                ),
                cx,
            ))
            .child(caption(t("Viewer", "查看器"), cx))
            .child(ParamSlider::new("pref-gray", t("Background", "背景"), gray * 100.0, 0.0, 60.0).default_value(11.0).unit("%").on_change(move |v, _, cx| {
                t3.update(cx, |p, cx| p.edit(cx, |s| s.view.viewer_gray = v / 100.0))
            }))
            .child(switch_row("pref-info", t("Show info overlay", "显示信息叠加"), info, move |v, _, cx| {
                t4.update(cx, |p, cx| p.edit(cx, |s| s.view.show_info_overlay = v))
            }, cx))
            .child(self.display_profile(cx))
            .into_any_element()
    }

    /// Monitor profile: listed from the system's ICC folders and saved; the
    /// engine still has to convert previews to it (see `contract::DISPLAY_PROFILE`).
    fn display_profile(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let chosen = self.state.read(cx).settings.display_profile.clone();
        let mut profiles = crate::state::display_profiles();
        if let Some(c) = chosen.as_ref().filter(|c| !profiles.contains(c)) {
            profiles.insert(0, c.clone());
        }
        let current = chosen.as_ref().and_then(|c| profiles.iter().position(|p| p == c)).map_or(0, |i| i + 1);
        let mut options: Vec<(usize, SharedString)> = vec![(0, t("sRGB (no conversion)", "sRGB（不转换）").into())];
        options.extend(profiles.iter().enumerate().map(|(i, path)| (i + 1, path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default().into())));
        let this = cx.entity();
        let status = crate::contract::DISPLAY_PROFILE;
        v_flex()
            .child(caption(t("Color management", "色彩管理"), cx))
            .child(row(
                t("Monitor profile", "显示器配置文件"),
                select("pref-display-profile", options, current, move |i, _, cx| {
                    let path = i.checked_sub(1).and_then(|i| profiles.get(i).cloned());
                    this.update(cx, |v, cx| v.edit(cx, |s| s.settings.display_profile = path));
                }),
                cx,
            ))
            .child(
                h_flex()
                    .px_3()
                    .pb_1()
                    .gap_1p5()
                    .text_xs()
                    .text_color(p.warning)
                    .child(Icon::new(IconName::TriangleAlert).xsmall())
                    .child(div().flex_1().child(status.note().unwrap_or_default())),
            )
            .into_any_element()
    }

    fn browser(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let s = self.state.read(cx);
        let thumb = s.view.thumb_size;
        let multi = s.view.multi;
        let divisions = s.settings.grid_divisions;
        let this = cx.entity();
        let (t1, t2, t3) = (this.clone(), this.clone(), this.clone());
        v_flex()
            .child(caption(t("Browser", "浏览器"), cx))
            .child(ParamSlider::new("pref-thumb", t("Thumbnail size", "缩略图大小"), thumb, 80.0, 320.0).default_value(120.0).unit(" px").on_change(move |v, _, cx| {
                t1.update(cx, |p, cx| p.edit(cx, |s| s.view.thumb_size = v))
            }))
            .child(caption(t("Viewer", "查看器"), cx))
            .child(switch_row("pref-multi", t("Show multiple selected images side by side", "选中多张时并排显示"), multi, move |v, _, cx| {
                t2.update(cx, |p, cx| p.edit(cx, |s| s.view.multi = v))
            }, cx))
            .child(note(t("Shortcut N toggles it; Shift+Enter pins the current image as the compare variant.", "快捷键 N 切换；Shift+Enter 将当前图像固定为对比变体。"), cx))
            .child(ParamSlider::new("pref-grid", t("Grid divisions", "网格分格"), divisions as f32, 2.0, 12.0).step(1.0).default_value(3.0).on_change(move |v, _, cx| {
                t3.update(cx, |p, cx| p.edit(cx, |s| s.settings.grid_divisions = v.round() as u8))
            }))
            .into_any_element()
    }

    fn exposure(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let s = self.state.read(cx).settings.clone();
        let this = cx.entity();
        let (t1, t2, t3, t4) = (this.clone(), this.clone(), this.clone(), this.clone());
        // A ramp showing which levels the warning marks.
        let (lo, hi) = (s.warn_low as f32 / 255.0, s.warn_high as f32 / 255.0);
        let ramp = h_flex()
            .mx_3()
            .my_2()
            .h(px(18.0))
            .rounded(px(3.0))
            .overflow_hidden()
            .relative()
            .bg(gpui_kit::linear_gradient(90.0, gpui_kit::linear_color_stop(gpui_kit::black(), 0.0), gpui_kit::linear_color_stop(gpui_kit::white(), 1.0)))
            .when(s.warn_shadows, |d| d.child(div().absolute().left_0().top_0().bottom_0().w(gpui_kit::relative(lo.max(0.004))).bg(hex(0x2878ff))))
            .child(div().absolute().right_0().top_0().bottom_0().w(gpui_kit::relative((1.0 - hi).max(0.004))).bg(hex(0xff2828)));
        v_flex()
            .child(caption(t("Exposure warning", "曝光警告"), cx))
            .child(ramp)
            .child(ParamSlider::new("pref-warn-high", t("Highlights at", "高光阈值"), s.warn_high as f32, 200.0, 255.0).step(1.0).default_value(250.0).on_change(move |v, _, cx| {
                t1.update(cx, |p, cx| p.edit_render(cx, |s| s.warn_high = v.round() as u8))
            }))
            .child(switch_row("pref-warn-shadows", t("Mark shadows", "标记阴影"), s.warn_shadows, move |v, _, cx| {
                t2.update(cx, |p, cx| p.edit_render(cx, |s| s.warn_shadows = v))
            }, cx))
            .child(
                ParamSlider::new("pref-warn-low", t("Shadows at", "阴影阈值"), s.warn_low as f32, 0.0, 40.0)
                    .step(1.0)
                    .default_value(4.0)
                    .disabled(!s.warn_shadows)
                    .on_change(move |v, _, cx| t3.update(cx, |p, cx| p.edit_render(cx, |s| s.warn_low = v.round() as u8))),
            )
            .child(div().px_3().text_xs().text_color(p.text_subtle).child(format!("{}（{}）", t("Red marks clipped highlights, blue blocked shadows", "红色标记高光溢出，蓝色标记暗部死黑"), crate::platform::keys("alt-e"))))
            .child(caption(t("Focus mask", "焦点蒙版"), cx))
            .child(ParamSlider::new("pref-focus", t("Edge threshold", "边缘阈值"), s.focus_threshold, 5.0, 80.0).step(1.0).default_value(22.0).on_change(move |v, _, cx| {
                t4.update(cx, |p, cx| p.edit_render(cx, |s| s.focus_threshold = v))
            }))
            .child(note(t("Lower values mark softer edges as in focus.", "数值越低，越柔和的边缘也会被标记为合焦。"), cx))
            .into_any_element()
    }

    fn crop(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let s = self.state.read(cx).settings.clone();
        let this = cx.entity();
        let (t1, t2) = (this.clone(), this.clone());
        v_flex()
            .child(caption(t("Crop", "裁剪"), cx))
            .child(row(
                t("Guides", "参考线"),
                select("pref-crop-guide", CropGuide::ALL.iter().map(|g| (*g, g.label().into())).collect(), s.crop_guide, move |g, _, cx| {
                    t1.update(cx, |p, cx| p.edit(cx, |s| s.settings.crop_guide = g))
                }),
                cx,
            ))
            .child(ParamSlider::new("pref-crop-mask", t("Mask outside crop", "裁剪外遮罩"), s.crop_mask * 100.0, 0.0, 100.0).unit("%").default_value(55.0).on_change(move |v, _, cx| {
                t2.update(cx, |p, cx| p.edit(cx, |s| s.settings.crop_mask = v / 100.0))
            }))
            .into_any_element()
    }

    fn workspaces(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        let list = s.workspace_list();
        let builtin = crate::state::builtin_workspaces().len();
        let active = s.settings.workspace.name.clone();
        let right = s.view.inspector_right;
        let hidden: Vec<ToolTab> = ToolTab::ALL.into_iter().filter(|t| s.tab_hidden(*t)).collect();
        let this = cx.entity();
        let t_side = this.clone();
        v_flex()
            .child(caption(t("Workspaces", "工作区"), cx))
            .children(list.into_iter().enumerate().map(|(i, w)| {
                let is_active = w.name == active;
                let (ta, td) = (this.clone(), this.clone());
                let name = w.name.clone();
                h_flex()
                    .id(SharedString::from(format!("pref-ws-{i}")))
                    .mx_3()
                    .px_2()
                    .h(px(30.0))
                    .gap_2()
                    .rounded(px(5.0))
                    .when(is_active, |d| d.bg(p.accent_soft))
                    .hover(|d| d.bg(p.hover))
                    .child(Icon::new(if i < builtin { IconName::LayoutPanelLeft } else { IconName::Bookmark }).small().text_color(if is_active { p.accent } else { p.text_subtle }))
                    .child(div().flex_1().text_sm().child(w.name.clone()))
                    .child(div().text_xs().text_color(p.text_subtle).child(format!("{} {}", ToolTab::ALL.len() - w.hidden_tabs.len(), t("tabs", "个页签"))))
                    .child(Button::new(SharedString::from(format!("pref-ws-apply-{i}"))).xsmall().ghost().label(t("Use", "使用")).on_click(move |_, _, cx| {
                        ta.update(cx, |v, cx| {
                            v.state.update(cx, |s, cx| s.apply_workspace(i, cx));
                            cx.notify();
                        })
                    }))
                    .when(i >= builtin, |d| {
                        d.child(Button::new(SharedString::from(format!("pref-ws-del-{i}"))).xsmall().ghost().icon(Icon::new(IconName::Trash)).on_click(move |_, _, cx| {
                            let name = name.clone();
                            td.update(cx, |v, cx| {
                                v.state.update(cx, |s, cx| s.delete_workspace(&name, cx));
                                cx.notify();
                            })
                        }))
                    })
            }))
            .child(
                h_flex().px_3().pt_2().child(Button::new("pref-ws-save").xsmall().outline().icon(Icon::new(IconName::Plus)).label(t("Save current as…", "将当前存储为…")).on_click(|_, window, cx| {
                    window.close_dialog(cx);
                    window.dispatch_action(Box::new(actions::SaveWorkspace), cx);
                })),
            )
            .child(caption(t("Layout", "布局"), cx))
            .child(row(
                t("Tools", "工具栏"),
                Segmented::new("pref-side", if right { 1 } else { 0 }).text(t("Left", "左侧")).text(t("Right", "右侧")).on_select(move |i, _, cx| {
                    t_side.update(cx, |p, cx| p.edit(cx, |s| s.view.inspector_right = i == 1))
                }),
                cx,
            ))
            .child(caption(t("Tool tabs in this workspace", "此工作区显示的页签"), cx))
            .child(h_flex().px_3().flex_wrap().gap_1p5().children(ToolTab::ALL.into_iter().map(|tab| {
                let shown = !hidden.contains(&tab);
                let th = this.clone();
                h_flex()
                    .id(SharedString::from(format!("pref-tab-{tab:?}")))
                    .gap_1()
                    .px_2()
                    .h(px(26.0))
                    .rounded(px(13.0))
                    .border_1()
                    .border_color(if shown { p.accent } else { p.edge })
                    .bg(if shown { p.accent_soft } else { p.well })
                    .text_xs()
                    .text_color(if shown { p.text } else { p.text_subtle })
                    .cursor_pointer()
                    .on_click(move |_, _, cx| {
                        th.update(cx, |v, cx| {
                            v.state.update(cx, |s, cx| s.set_tab_hidden(tab, shown, cx));
                            cx.notify();
                        })
                    })
                    .child(Icon::new(tab.icon()).xsmall())
                    .child(tab.label())
            })))
            .child(note(t("Right-click the tab strip to change this from the workspace.", "也可以在页签栏上右键修改。"), cx))
            .into_any_element()
    }

    fn keyboard(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let overrides = self.state.read(cx).settings.shortcuts.clone();
        let all = shortcuts();
        // Keys used by more than one command.
        let mut uses: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for s in &all {
            for k in keys_for(s, &overrides) {
                uses.entry(k).or_default().push(s.label.clone());
            }
        }
        let query = self.query.to_lowercase();
        let this = cx.entity();
        let tq = this.clone();
        let search = text_field("pref-keys-search", self.query.clone(), t("Search commands or keys…", "搜索命令或按键…"), move |q, _, cx| {
            tq.update(cx, |v, cx| {
                v.query = q;
                cx.notify();
            })
        }, window, cx);

        let mut list = v_flex().px_3();
        if query.is_empty() {
            // Speed Edit keys are held, not pressed, so they are listed apart.
            let keys = crate::state::SpeedParam::ALL.iter().map(|p| format!("{} {}", p.key().to_uppercase(), p.label())).collect::<Vec<_>>().join("　");
            list = list.child(div().pt_3().pb_1().text_xs().text_color(p.text_subtle).child(t("Speed Edit", "快速编辑"))).child(
                v_flex()
                    .px_2()
                    .gap_1()
                    .text_xs()
                    .child(div().text_color(p.text_muted).child(t(
                        "Hold a key and scroll or drag over the image. Shift for fine steps. A quick tap still selects the key's cursor tool.",
                        "按住按键并在图像上滚动或拖动；按住 Shift 微调。轻点按键仍会切换对应的光标工具。",
                    )))
                    .child(div().text_color(p.text).child(keys)),
            );
        }
        let mut last_group = "";
        for (i, s) in all.iter().enumerate() {
            let keys = keys_for(s, &overrides);
            let shown = display_keys(&keys.join(" / "));
            if !query.is_empty() && !s.label.to_lowercase().contains(&query) && !shown.to_lowercase().contains(&query) && !keys.iter().any(|k| k.contains(&query)) {
                continue;
            }
            if s.group != last_group {
                last_group = s.group;
                list = list.child(div().pt_3().pb_1().text_xs().text_color(p.text_subtle).child(s.group));
            }
            let recording = self.recording.as_deref() == Some(s.id.as_str());
            let custom = overrides.contains_key(&s.id);
            let conflicts: Vec<String> = keys.iter().flat_map(|k| uses.get(k).into_iter().flatten().filter(|l| **l != s.label).cloned()).collect();
            let (tr, tc, tx) = (this.clone(), this.clone(), this.clone());
            let (id_r, id_c, id_x) = (s.id.clone(), s.id.clone(), s.id.clone());
            let conflict_tip = format!("{} {}", t("Also used by", "同时用于"), conflicts.join(", "));
            list = list.child(
                h_flex()
                    .id(SharedString::from(format!("key-row-{i}")))
                    .h(px(30.0))
                    .px_2()
                    .gap_2()
                    .rounded(px(5.0))
                    .when(recording, |d| d.bg(p.accent_soft))
                    .hover(|d| d.bg(p.hover))
                    .child(div().flex_1().min_w_0().text_sm().text_color(p.text).overflow_hidden().whitespace_nowrap().text_ellipsis().child(s.label.clone()))
                    .when(!conflicts.is_empty(), |d| {
                        d.child(
                            div()
                                .id(SharedString::from(format!("key-conflict-{i}")))
                                .child(Icon::new(IconName::TriangleAlert).xsmall().text_color(p.warning))
                                .tooltip(move |window, cx| Tooltip::new(conflict_tip.clone()).build(window, cx)),
                        )
                    })
                    .child(
                        div()
                            .id(SharedString::from(format!("key-chip-{i}")))
                            .min_w(px(96.0))
                            .px_2()
                            .h(px(22.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .rounded(px(4.0))
                            .border_1()
                            .border_color(if recording { p.accent } else if custom { p.accent.opacity(0.5) } else { p.edge })
                            .bg(p.well)
                            .text_xs()
                            .text_color(if recording { p.accent } else if keys.is_empty() { p.text_subtle } else { p.text_muted })
                            .cursor_pointer()
                            .child(if recording {
                                t("Press keys… (Esc cancels)", "请按下按键…（Esc 取消）").to_string()
                            } else if keys.is_empty() {
                                t("None", "无").to_string()
                            } else {
                                shown
                            })
                            .on_click(move |_, window, cx| {
                                let id = id_r.clone();
                                tr.update(cx, |v, cx| {
                                    v.recording = Some(id);
                                    window.focus(&v.focus, cx);
                                    cx.notify();
                                })
                            }),
                    )
                    .child(
                        Button::new(SharedString::from(format!("key-clear-{i}")))
                            .xsmall()
                            .ghost()
                            .icon(Icon::new(IconName::X))
                            .tooltip(t("Remove shortcut", "移除快捷键"))
                            .on_click(move |_, _, cx| {
                                let id = id_c.clone();
                                tc.update(cx, |v, cx| v.set_keys(&id, Some(Vec::new()), cx))
                            }),
                    )
                    .child(
                        Button::new(SharedString::from(format!("key-reset-{i}")))
                            .xsmall()
                            .ghost()
                            .icon(Icon::new(IconName::RotateCcw))
                            .tooltip(t("Restore default", "恢复默认"))
                            .when(!custom, |b| b.invisible())
                            .on_click(move |_, _, cx| {
                                let id = id_x.clone();
                                tx.update(cx, |v, cx| v.set_keys(&id, None, cx))
                            }),
                    ),
            );
        }
        let custom = overrides.len();
        v_flex()
            .child(
                h_flex()
                    .px_3()
                    .pt_1()
                    .gap_2()
                    .child(div().flex_1().child(search))
                    .child(Button::new("keys-reset-all").xsmall().outline().label(format!("{} ({custom})", t("Restore all", "全部恢复默认"))).on_click(cx.listener(|v, _, _, cx| {
                        v.state.update(cx, |s, _| {
                            s.settings.shortcuts.clear();
                            s.save_settings();
                        });
                        actions::bind_keys(cx, &BTreeMap::new());
                        cx.notify();
                    }))),
            )
            .child(list)
            .into_any_element()
    }
}

pub fn open(state: &Entity<AppState>, window: &mut Window, cx: &mut App) {
    let view = cx.new(|cx| PrefsView { state: state.clone(), page: 0, recording: None, query: String::new(), focus: cx.focus_handle() });
    window.open_dialog(cx, move |dialog, _, _| dialog.title(t("Preferences", "首选项")).w(px(820.0)).child(view.clone()));
}

/// Opens Preferences on the keyboard page.
pub fn open_shortcuts(state: &Entity<AppState>, window: &mut Window, cx: &mut App) {
    let view = cx.new(|cx| PrefsView { state: state.clone(), page: PAGES.len() - 1, recording: None, query: String::new(), focus: cx.focus_handle() });
    window.open_dialog(cx, move |dialog, _, _| dialog.title(t("Preferences", "首选项")).w(px(820.0)).child(view.clone()));
}
