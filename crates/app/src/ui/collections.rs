//! Collection dialogs (rename, smart-album rules) and the drag payload that
//! carries images from the browser onto albums.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, Sizable as _, WindowExt as _,
    button::{Button, ButtonVariants as _},
    h_flex,
    input::{Input, InputState},
    v_flex,
};
use gpui_kit::{
    App, AppContext as _, Context, Entity, Focusable as _, IntoElement, ParentElement, Render, SharedString, Styled,
    Window, div, px,
};
use lumen_engine::{ColorTag, FileKind, ImageId, SmartRule};

use crate::i18n::t;
use crate::state::AppState;
use crate::theme::{hexa, pal};
use crate::widgets::{Segmented, select, text_field};

/// Images being dragged out of the browser.
#[derive(Clone)]
pub struct DraggedImages(pub Vec<ImageId>);

/// The small card that follows the pointer while dragging images.
pub struct DragCard {
    pub count: usize,
}

impl Render for DragCard {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let p = pal(cx).clone();
        h_flex()
            .gap_1p5()
            .px_2()
            .py_1()
            .rounded(px(6.0))
            .bg(p.raised)
            .border_1()
            .border_color(p.accent)
            .shadow_lg()
            .text_xs()
            .text_color(p.text)
            .child(Icon::new(IconName::Images).xsmall().text_color(p.accent))
            .child(format!("{} {}", self.count, t("images", "张图像")))
    }
}

pub fn rename(state: &Entity<AppState>, id: u64, current: String, window: &mut Window, cx: &mut App) {
    let input = cx.new(|cx| InputState::new(window, cx).default_value(current));
    let st = state.clone();
    window.open_dialog(cx, move |dialog, window, cx| {
        let input2 = input.clone();
        let st = st.clone();
        input.read(cx).focus_handle(cx).focus(window, cx);
        dialog
            .title(t("Rename collection", "重命名集合"))
            .w(px(380.0))
            .child(Input::new(&input))
            .footer(
                h_flex().gap_2().justify_end().child(Button::new("coll-rename-ok").primary().label(t("Rename", "重命名")).on_click(move |_, window, cx| {
                    let name = input2.read(cx).value().to_string();
                    if !name.trim().is_empty() {
                        st.update(cx, |s, cx| s.rename_collection(id, name.trim().to_string(), cx));
                    }
                    window.close_dialog(cx);
                })),
            )
    });
}

#[derive(Clone, Copy, PartialEq)]
enum RuleKind {
    Rating,
    Tag,
    Keyword,
    Name,
    Camera,
    Lens,
    Iso,
    Edited,
    Kind,
}

impl RuleKind {
    const ALL: [RuleKind; 9] = [
        RuleKind::Rating,
        RuleKind::Tag,
        RuleKind::Keyword,
        RuleKind::Name,
        RuleKind::Camera,
        RuleKind::Lens,
        RuleKind::Iso,
        RuleKind::Edited,
        RuleKind::Kind,
    ];

    fn of(r: &SmartRule) -> Self {
        match r {
            SmartRule::RatingAtLeast(_) => RuleKind::Rating,
            SmartRule::ColorTag(_) => RuleKind::Tag,
            SmartRule::KeywordContains(_) => RuleKind::Keyword,
            SmartRule::NameContains(_) => RuleKind::Name,
            SmartRule::CameraContains(_) => RuleKind::Camera,
            SmartRule::LensContains(_) => RuleKind::Lens,
            SmartRule::IsoAtLeast(_) => RuleKind::Iso,
            SmartRule::Edited(_) => RuleKind::Edited,
            SmartRule::Kind(_) => RuleKind::Kind,
        }
    }

    fn label(self) -> &'static str {
        match self {
            RuleKind::Rating => t("Rating at least", "评级不低于"),
            RuleKind::Tag => t("Color tag is", "色标为"),
            RuleKind::Keyword => t("Keyword contains", "关键词包含"),
            RuleKind::Name => t("File name contains", "文件名包含"),
            RuleKind::Camera => t("Camera contains", "相机包含"),
            RuleKind::Lens => t("Lens contains", "镜头包含"),
            RuleKind::Iso => t("ISO at least", "ISO 不低于"),
            RuleKind::Edited => t("Edited", "已编辑"),
            RuleKind::Kind => t("File type is", "文件类型为"),
        }
    }

    fn default_rule(self) -> SmartRule {
        match self {
            RuleKind::Rating => SmartRule::RatingAtLeast(3),
            RuleKind::Tag => SmartRule::ColorTag(ColorTag::Red),
            RuleKind::Keyword => SmartRule::KeywordContains(String::new()),
            RuleKind::Name => SmartRule::NameContains(String::new()),
            RuleKind::Camera => SmartRule::CameraContains(String::new()),
            RuleKind::Lens => SmartRule::LensContains(String::new()),
            RuleKind::Iso => SmartRule::IsoAtLeast(800),
            RuleKind::Edited => SmartRule::Edited(true),
            RuleKind::Kind => SmartRule::Kind(FileKind::Raw),
        }
    }
}

pub(crate) fn tag_name(c: ColorTag) -> &'static str {
    match c {
        ColorTag::None => t("None", "无"),
        ColorTag::Red => t("Red", "红"),
        ColorTag::Orange => t("Orange", "橙"),
        ColorTag::Yellow => t("Yellow", "黄"),
        ColorTag::Green => t("Green", "绿"),
        ColorTag::Blue => t("Blue", "蓝"),
        ColorTag::Pink => t("Pink", "粉"),
        ColorTag::Purple => t("Purple", "紫"),
    }
}

fn set_rule(state: &Entity<AppState>, id: u64, index: usize, rule: SmartRule, cx: &mut App) {
    state.update(cx, |s, cx| {
        let Some(n) = s.node(id) else { return };
        let mut rules = n.rules.clone();
        let all = n.match_all;
        if index < rules.len() {
            rules[index] = rule;
        }
        s.set_smart_rules(id, rules, all, cx);
    });
}

/// Edits a smart album's rules live: the browser updates as rules change.
pub fn smart_rules(state: &Entity<AppState>, id: u64, window: &mut Window, cx: &mut App) {
    let st = state.clone();
    window.open_dialog(cx, move |dialog, window, cx| {
        let p = pal(cx).clone();
        let Some(node) = st.read(cx).node(id).cloned() else { return dialog };
        let count = st.read(cx).collection_count(id);
        let rows = node.rules.iter().enumerate().map(|(i, rule)| {
            let kind = RuleKind::of(rule);
            let st_kind = st.clone();
            let st_del = st.clone();
            let value: gpui_kit::AnyElement = match rule.clone() {
                SmartRule::RatingAtLeast(r) => {
                    let st = st.clone();
                    select(format!("rule-r-{id}-{i}"), (-1..=5).map(|v| (v, if v < 0 { t("Rejected", "拒绝").into() } else { SharedString::from("★".repeat(v as usize) + if v == 0 { "0" } else { "" }) })).collect(), r, move |v, _, cx| {
                        set_rule(&st, id, i, SmartRule::RatingAtLeast(v), cx)
                    })
                }
                SmartRule::ColorTag(c) => {
                    let st = st.clone();
                    select(format!("rule-t-{id}-{i}"), ColorTag::ALL.iter().map(|c| (*c, tag_name(*c).into())).collect(), c, move |v, _, cx| set_rule(&st, id, i, SmartRule::ColorTag(v), cx))
                }
                SmartRule::Edited(e) => {
                    let st = st.clone();
                    select(format!("rule-e-{id}-{i}"), vec![(true, t("Yes", "是").into()), (false, t("No", "否").into())], e, move |v, _, cx| set_rule(&st, id, i, SmartRule::Edited(v), cx))
                }
                SmartRule::Kind(k) => {
                    let st = st.clone();
                    let kinds = [FileKind::Raw, FileKind::Dng, FileKind::Jpeg, FileKind::Tiff, FileKind::Png, FileKind::Heif];
                    select(format!("rule-k-{id}-{i}"), kinds.iter().map(|k| (*k, format!("{k:?}").to_uppercase().into())).collect(), k, move |v, _, cx| set_rule(&st, id, i, SmartRule::Kind(v), cx))
                }
                SmartRule::IsoAtLeast(iso) => {
                    let st = st.clone();
                    text_field(SharedString::from(format!("rule-iso-{id}-{i}")), iso.to_string(), "800", move |v, _, cx| {
                        if let Ok(n) = v.trim().parse() {
                            set_rule(&st, id, i, SmartRule::IsoAtLeast(n), cx)
                        }
                    }, window, cx)
                }
                SmartRule::KeywordContains(v) | SmartRule::NameContains(v) | SmartRule::CameraContains(v) | SmartRule::LensContains(v) => {
                    let st = st.clone();
                    text_field(SharedString::from(format!("rule-txt-{id}-{i}-{}", kind as u8)), v, "", move |text, _, cx| {
                        let rule = match kind {
                            RuleKind::Keyword => SmartRule::KeywordContains(text),
                            RuleKind::Name => SmartRule::NameContains(text),
                            RuleKind::Camera => SmartRule::CameraContains(text),
                            _ => SmartRule::LensContains(text),
                        };
                        set_rule(&st, id, i, rule, cx)
                    }, window, cx)
                }
            };
            h_flex()
                .gap_2()
                .child(div().w(px(150.0)).child(select(format!("rule-kind-{id}-{i}"), RuleKind::ALL.iter().map(|k| (*k as u8, k.label().into())).collect(), kind as u8, move |k, _, cx| {
                    set_rule(&st_kind, id, i, RuleKind::ALL[k as usize].default_rule(), cx)
                })))
                .child(div().flex_1().child(value))
                .child(Button::new(SharedString::from(format!("rule-del-{id}-{i}"))).xsmall().ghost().icon(Icon::new(IconName::X)).on_click(move |_, _, cx| {
                    st_del.update(cx, |s, cx| {
                        let Some(n) = s.node(id) else { return };
                        let mut rules = n.rules.clone();
                        let all = n.match_all;
                        if i < rules.len() {
                            rules.remove(i);
                        }
                        s.set_smart_rules(id, rules, all, cx);
                    })
                }))
        });
        let (st_all, st_add) = (st.clone(), st.clone());
        dialog
            .title(format!("{} · {}", t("Smart Album", "智能相册"), node.name))
            .w(px(560.0))
            .child(
                v_flex()
                    .gap_2()
                    .child(
                        h_flex()
                            .gap_2()
                            .text_sm()
                            .text_color(p.text_muted)
                            .child(t("Include images that match", "包含匹配以下条件的图像："))
                            .child(div().w(px(160.0)).child(Segmented::new(format!("rule-all-{id}"), if node.match_all { 0 } else { 1 }).text(t("all rules", "全部条件")).text(t("any rule", "任一条件")).on_select(move |i, _, cx| {
                                st_all.update(cx, |s, cx| {
                                    let rules = s.node(id).map(|n| n.rules.clone()).unwrap_or_default();
                                    s.set_smart_rules(id, rules, i == 0, cx)
                                })
                            }))),
                    )
                    .children(rows)
                    .child(
                        h_flex()
                            .justify_between()
                            .child(Button::new("rule-add").xsmall().outline().icon(Icon::new(IconName::Plus)).label(t("Add rule", "添加条件")).on_click(move |_, _, cx| {
                                st_add.update(cx, |s, cx| {
                                    let Some(n) = s.node(id) else { return };
                                    let mut rules = n.rules.clone();
                                    let all = n.match_all;
                                    rules.push(SmartRule::RatingAtLeast(3));
                                    s.set_smart_rules(id, rules, all, cx);
                                })
                            }))
                            .child(div().px_2().py_0p5().rounded(px(4.0)).bg(hexa(0x4c8dff, 0.15)).text_xs().text_color(p.accent).child(format!("{count} {}", t("matching images", "张匹配图像")))),
                    ),
            )
    });
}
