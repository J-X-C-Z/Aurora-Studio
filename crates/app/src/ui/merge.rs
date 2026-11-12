//! The HDR and panorama merge dialogs. Built ahead of the engine: the options
//! are the ones the request asks the engine to take (`planned::HdrOptions`,
//! `planned::PanoramaOptions`), and Merge stays off until it can.

use std::cell::RefCell;
use std::rc::Rc;

use gpui_kit::component::{
    Disableable as _, WindowExt as _,
    button::{Button, ButtonVariants as _},
    h_flex, v_flex,
};
use gpui_kit::{App, Entity, ParentElement, Styled, Window, div, prelude::FluentBuilder as _, px};

use crate::i18n::t;
use crate::planned::{HdrOptions, PanoramaOptions, Projection};
use crate::state::{AppState, MergeKind};
use crate::theme::pal;
use crate::widgets::{ParamSlider, Segmented, row, select, switch_row};

const DEGHOST: [(&str, &str); 4] = [("Off", "关"), ("Low", "低"), ("Medium", "中"), ("High", "高")];

pub fn open(state: &Entity<AppState>, kind: MergeKind, window: &mut Window, cx: &mut App) {
    let count = state.read(cx).selection.len();
    if count < 2 {
        state.update(cx, |s, cx| s.notify(t("Select at least two images to merge", "请至少选择两张图像进行合并"), cx));
        return;
    }
    let hdr = Rc::new(RefCell::new(HdrOptions::default()));
    let pano = Rc::new(RefCell::new(PanoramaOptions::default()));
    let state = state.clone();
    window.open_dialog(cx, move |dialog, _window, cx| {
        let p = pal(cx).clone();
        let (h, o) = (*hdr.borrow(), *pano.borrow());
        let options = match kind {
            MergeKind::Hdr => {
                let (a, b, c) = (hdr.clone(), hdr.clone(), hdr.clone());
                v_flex()
                    .child(switch_row("hdr-align", t("Align frames", "对齐帧"), h.align, move |v, window, _| {
                        a.borrow_mut().align = v;
                        window.refresh();
                    }, cx))
                    .child(row(
                        t("Deghosting", "去鬼影"),
                        DEGHOST.iter().fold(Segmented::new("hdr-deghost", h.deghost as usize), |s, (en, zh)| s.text(t(en, zh))).on_select(move |i, window, _| {
                            b.borrow_mut().deghost = i as u8;
                            window.refresh();
                        }),
                        cx,
                    ))
                    .child(switch_row("hdr-tone", t("Auto tone", "自动色调"), h.auto_tone, move |v, window, _| {
                        c.borrow_mut().auto_tone = v;
                        window.refresh();
                    }, cx))
            }
            MergeKind::Panorama => {
                let (a, b, c) = (pano.clone(), pano.clone(), pano.clone());
                v_flex()
                    .child(row(
                        t("Projection", "投影"),
                        select("pano-projection", Projection::ALL.iter().map(|p| (*p, p.label().into())).collect(), o.projection, move |v, window, _| {
                            a.borrow_mut().projection = v;
                            window.refresh();
                        }),
                        cx,
                    ))
                    .child(ParamSlider::new("pano-warp", t("Boundary warp", "边界变形"), o.boundary_warp, 0.0, 100.0).on_change(move |v, window, _| {
                        b.borrow_mut().boundary_warp = v;
                        window.refresh();
                    }))
                    .child(switch_row("pano-crop", t("Auto crop", "自动裁剪"), o.auto_crop, move |v, window, _| {
                        c.borrow_mut().auto_crop = v;
                        window.refresh();
                    }, cx))
            }
        };
        let st = state.clone();
        let ready = crate::planned::MERGE_READY;
        dialog.title(kind.label()).w(px(460.0)).child(
            v_flex()
                .gap_3()
                .child(div().text_xs().text_color(p.text_muted).child(format!(
                    "{count} {}",
                    match kind {
                        MergeKind::Hdr => t("bracketed frames become one HDR image beside the first frame.", "张包围曝光帧将合并为一张 HDR 图像，存放在第一张旁边。"),
                        MergeKind::Panorama => t("overlapping frames are stitched into one image beside the first frame.", "张重叠的图像将拼接为一张全景，存放在第一张旁边。"),
                    }
                )))
                .child(options)
                .when(!ready, |d| d.child(div().text_xs().text_color(p.warning).child(crate::contract::MERGE.note().unwrap_or_default())))
                .child(
                    h_flex()
                        .gap_2()
                        .justify_end()
                        .child(Button::new("merge-cancel").outline().label(t("Cancel", "取消")).on_click(|_, window, cx| window.close_dialog(cx)))
                        .child(Button::new("merge-go").primary().label(t("Merge", "合并")).disabled(!ready).on_click({
                            let (hdr, pano) = (hdr.clone(), pano.clone());
                            move |_, window, cx| {
                                let (h, o) = (*hdr.borrow(), *pano.borrow());
                                st.update(cx, |s, cx| s.merge_selection(kind, h, o, cx));
                                window.close_dialog(cx);
                            }
                        })),
                ),
        )
    });
}
