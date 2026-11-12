//! The Print window: paper, orientation, a rows × columns layout with margins
//! and spacing, and a live page preview whose cells glide when the layout
//! changes. Sending pages to a printer is not connected to a backend yet.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, Sizable as _, WindowExt as _,
    button::{Button, ButtonVariants as _},
    h_flex, v_flex,
};
use gpui_kit::{
    App, AppContext as _, Context, Entity, IntoElement, ObjectFit, ParentElement, Render, SharedString, Styled, StyledImage as _, Window, div,
    img, prelude::FluentBuilder as _, px,
};
use lumen_engine::ImageId;

use crate::i18n::t;
use crate::motion;
use crate::state::AppState;
use crate::theme::{hexa, pal};
use crate::widgets::{ParamSlider, Segmented, caption, row, select, switch_row};

const PAPERS: [(&str, f32, f32); 6] = [
    ("A4", 210.0, 297.0),
    ("A3", 297.0, 420.0),
    ("Letter", 215.9, 279.4),
    ("4 × 6 in", 101.6, 152.4),
    ("5 × 7 in", 127.0, 177.8),
    ("8 × 10 in", 203.2, 254.0),
];

pub struct PrintDialog {
    state: Entity<AppState>,
    images: Vec<ImageId>,
    printer: usize,
    paper: usize,
    landscape: bool,
    rows: f32,
    cols: f32,
    margin: f32,
    spacing: f32,
    fill: bool,
    captions: bool,
    copies: f32,
    page: usize,
}

impl PrintDialog {
    fn per_page(&self) -> usize {
        (self.rows.round() as usize).max(1) * (self.cols.round() as usize).max(1)
    }

    fn pages(&self) -> usize {
        self.images.len().div_ceil(self.per_page()).max(1)
    }

    fn set(&mut self, cx: &mut Context<Self>, f: impl FnOnce(&mut Self)) {
        f(self);
        self.page = self.page.min(self.pages() - 1);
        cx.notify();
    }
}

impl Render for PrintDialog {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let p = pal(cx).clone();
        let this = cx.entity();
        let (_, pw, ph) = PAPERS[self.paper];
        let (pw, ph) = if self.landscape { (ph, pw) } else { (pw, ph) };

        // Page preview, fitted into a 300 × 330 well.
        let scale = (300.0 / pw).min(330.0 / ph);
        let page_w = motion::value(motion::key("print", "pw"), pw * scale, motion::FLOW, window, cx);
        let page_h = motion::value(motion::key("print", "ph"), ph * scale, motion::FLOW, window, cx);
        let (rows, cols) = ((self.rows.round() as usize).max(1), (self.cols.round() as usize).max(1));
        let m = self.margin * scale;
        let g = self.spacing * scale;
        let cap = if self.captions { 12.0 } else { 0.0 };
        let cell_w = ((page_w - 2.0 * m - g * (cols as f32 - 1.0)) / cols as f32).max(4.0);
        let cell_h = ((page_h - 2.0 * m - g * (rows as f32 - 1.0)) / rows as f32).max(4.0);
        let start = self.page * self.per_page();
        let mut sheet = div().relative().w(px(page_w)).h(px(page_h)).bg(gpui_kit::white()).shadow_lg().rounded(px(2.0));
        for (i, id) in self.images.iter().skip(start).take(self.per_page()).enumerate() {
            let (r, c) = (i / cols, i % cols);
            let key = format!("print-cell-{i}");
            let (thumb, name) = {
                let s = self.state.read(cx);
                (s.thumbs.get(id).cloned(), s.info(*id).map(|i| i.file_name.clone()).unwrap_or_default())
            };
            let x = motion::value(motion::key(&key, "x"), m + c as f32 * (cell_w + g), motion::FLOW, window, cx);
            let y = motion::value(motion::key(&key, "y"), m + r as f32 * (cell_h + g), motion::FLOW, window, cx);
            let w = motion::value(motion::key(&key, "w"), cell_w, motion::FLOW, window, cx);
            let h = motion::value(motion::key(&key, "h"), cell_h, motion::FLOW, window, cx);
            sheet = sheet.child(
                v_flex()
                    .absolute()
                    .left(px(x))
                    .top(px(y))
                    .w(px(w))
                    .h(px(h))
                    .child(
                        div()
                            .relative()
                            .flex_1()
                            .min_h_0()
                            .overflow_hidden()
                            .bg(hexa(0x000000, 0.04))
                            .when_some(thumb, |d, t| d.child(img(t).absolute().size_full().object_fit(if self.fill { ObjectFit::Cover } else { ObjectFit::Contain }))),
                    )
                    .when(self.captions, |d| {
                        d.child(div().h(px(cap)).text_size(px(7.0)).text_color(hexa(0x333333, 1.0)).overflow_hidden().whitespace_nowrap().child(name))
                    }),
            );
        }
        // Margin guide.
        sheet = sheet.child(div().absolute().left(px(m)).top(px(m)).w(px((page_w - 2.0 * m).max(0.0))).h(px((page_h - 2.0 * m).max(0.0))).border_1().border_color(hexa(0x4a90d9, 0.35)));

        let pages = self.pages();
        let page = self.page;
        let preview = v_flex()
            .w(px(340.0))
            .flex_shrink_0()
            .gap_2()
            .child(div().h(px(350.0)).rounded(px(6.0)).bg(p.well).flex().items_center().justify_center().child(motion::enter(SharedString::from(format!("print-page-{page}")), sheet, 6.0)))
            .child(
                h_flex()
                    .justify_center()
                    .gap_2()
                    .text_xs()
                    .text_color(p.text_muted)
                    .child(Button::new("print-prev").xsmall().ghost().icon(Icon::new(IconName::ChevronLeft)).on_click(cx.listener(|d, _, _, cx| d.set(cx, |d| d.page = d.page.saturating_sub(1)))))
                    .child(format!("{} {} / {pages}", t("Page", "页"), page + 1))
                    .child(Button::new("print-next").xsmall().ghost().icon(Icon::new(IconName::ChevronRight)).on_click(cx.listener(|d, _, _, cx| d.set(cx, |d| d.page += 1)))),
            );

        let slider = |id: &'static str, label: &'static str, value: f32, min: f32, max: f32, unit: &'static str, f: fn(&mut PrintDialog, f32)| {
            let this = this.clone();
            ParamSlider::new(id, label, value, min, max).step(1.0).unit(unit).default_value(value).on_change(move |v, _, cx| this.update(cx, |d, cx| d.set(cx, |d| f(d, v))))
        };

        let printers: Vec<(usize, SharedString)> = vec![(0, t("System default printer", "系统默认打印机").into()), (1, "Microsoft Print to PDF".into())];
        let papers: Vec<(usize, SharedString)> = PAPERS.iter().enumerate().map(|(i, (n, _, _))| (i, SharedString::from(*n))).collect();
        let (t1, t2, t3, t4, t5) = (this.clone(), this.clone(), this.clone(), this.clone(), this.clone());
        let settings = v_flex()
            .flex_1()
            .min_w_0()
            .child(caption(t("Printer", "打印机"), cx))
            .child(row(t("Printer", "打印机"), select("print-printer", printers, self.printer, move |i, _, cx| t1.update(cx, |d, cx| d.set(cx, |d| d.printer = i))), cx))
            .child(row(t("Paper", "纸张"), select("print-paper", papers, self.paper, move |i, _, cx| t2.update(cx, |d, cx| d.set(cx, |d| d.paper = i))), cx))
            .child(row(
                t("Orientation", "方向"),
                Segmented::new("print-orient", if self.landscape { 1 } else { 0 }).text(t("Portrait", "纵向")).text(t("Landscape", "横向")).on_select(move |i, _, cx| t3.update(cx, |d, cx| d.set(cx, |d| d.landscape = i == 1))),
                cx,
            ))
            .child(caption(t("Layout", "版式"), cx))
            .child(slider("print-rows", t("Rows", "行"), self.rows, 1.0, 5.0, "", |d, v| d.rows = v.round()))
            .child(slider("print-cols", t("Columns", "列"), self.cols, 1.0, 5.0, "", |d, v| d.cols = v.round()))
            .child(slider("print-margin", t("Margins", "页边距"), self.margin, 0.0, 40.0, " mm", |d, v| d.margin = v.round()))
            .child(slider("print-spacing", t("Spacing", "间距"), self.spacing, 0.0, 30.0, " mm", |d, v| d.spacing = v.round()))
            .child(row(
                t("Image", "图像"),
                Segmented::new("print-fit", if self.fill { 1 } else { 0 }).text(t("Fit", "适合")).text(t("Fill", "填充")).on_select(move |i, _, cx| t4.update(cx, |d, cx| d.set(cx, |d| d.fill = i == 1))),
                cx,
            ))
            .child(switch_row("print-captions", t("File name captions", "文件名说明"), self.captions, move |v, _, cx| t5.update(cx, |d, cx| d.set(cx, |d| d.captions = v)), cx))
            .child(slider("print-copies", t("Copies", "份数"), self.copies, 1.0, 20.0, "", |d, v| d.copies = v.round()));

        let n = self.images.len();
        let status = crate::contract::PRINT;
        let state = self.state.clone();
        v_flex()
            .gap_3()
            .child(h_flex().items_start().gap_4().child(preview).child(settings))
            .child(
                h_flex()
                    .gap_2()
                    .child(div().flex_1().text_xs().text_color(p.text_subtle).child(format!("{n} {} · {pages} {}", t("images", "张图像"), t("pages", "页"))))
                    .child(Button::new("print-cancel").outline().label(t("Cancel", "取消")).on_click(|_, window, cx| window.close_dialog(cx)))
                    .child(
                        Button::new("print-go")
                            .primary()
                            .icon(Icon::new(IconName::Printer))
                            .label(t("Print", "打印"))
                            .on_click(move |_, _, cx| state.update(cx, |s, cx| s.notify(status.note().unwrap_or_default(), cx))),
                    ),
            )
    }
}

pub fn open(state: &Entity<AppState>, window: &mut Window, cx: &mut App) {
    let images: Vec<ImageId> = {
        let s = state.read(cx);
        if s.selection.is_empty() { s.current.into_iter().collect() } else { s.selection.iter().copied().collect() }
    };
    let view = cx.new(|_| PrintDialog {
        state: state.clone(),
        images,
        printer: 0,
        paper: 0,
        landscape: false,
        rows: 2.0,
        cols: 2.0,
        margin: 12.0,
        spacing: 6.0,
        fill: false,
        captions: true,
        copies: 1.0,
        page: 0,
    });
    window.open_dialog(cx, move |dialog, _, _| dialog.title(t("Print", "打印")).w(px(760.0)).child(view.clone()));
}
