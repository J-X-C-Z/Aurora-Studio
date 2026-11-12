//! The live histogram: red, green and blue overlaid with the luminance outline,
//! plus shadow and highlight clipping indicators.

use gpui_kit::{
    App, ElementId, IntoElement, ParentElement, RenderOnce, SharedString, Styled, Window, canvas, div, prelude::FluentBuilder as _, px,
};
use lumen_engine::Histogram;

use super::paint;
use crate::motion;
use crate::theme::{hexa, pal};

const POINTS: usize = 128;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum HistogramMode {
    Rgb,
    Luma,
    Red,
    Green,
    Blue,
}

#[derive(IntoElement)]
pub struct HistogramView {
    id: SharedString,
    data: Histogram,
    mode: HistogramMode,
    height: f32,
    show_clipping: bool,
}

impl HistogramView {
    pub fn new(id: impl Into<SharedString>, data: Histogram) -> Self {
        Self { id: id.into(), data, mode: HistogramMode::Rgb, height: 96.0, show_clipping: true }
    }

    pub fn mode(mut self, mode: HistogramMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn height(mut self, h: f32) -> Self {
        self.height = h;
        self
    }

}

/// A clipping indicator: an outlined dot that fills and grows as clipping
/// appears, with the clipped share of pixels beside it.
fn clip_dot(id: &SharedString, side: &str, share: f32, color: gpui_kit::Hsla, window: &mut Window, cx: &mut App) -> impl IntoElement + use<> {
    let p = pal(cx).clone();
    let on = motion::toggle(motion::key(id, side), share > 0.002, motion::QUICK, window, cx);
    let size = 7.0 + 2.0 * on;
    let left = side == "lo";
    let label = if share >= 0.001 { format!("{:.1}%", share * 100.0) } else { "<0.1%".into() };
    gpui_kit::component::h_flex()
        .absolute()
        .top(px(4.0 - on))
        .when(left, |d| d.left(px(4.0 - on)))
        .when(!left, |d| d.right(px(4.0 - on)).flex_row_reverse())
        .gap_1()
        .child(div().size(px(size)).rounded_full().border_1().border_color(p.text_subtle.blend(color.opacity(on))).bg(color.opacity(on)))
        .when(on > 0.01, |d| d.child(div().text_size(px(9.0)).text_color(color.opacity(0.9 * on)).child(label)))
}

impl RenderOnce for HistogramView {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let p = pal(cx).clone();
        let h = self.data;
        let mode = self.mode;
        // All four channels tween together, so switching mode or a new preview
        // frame reshapes the curves instead of replacing them.
        let mut flat = paint::normalize_bins(&h.red, POINTS);
        flat.extend(paint::normalize_bins(&h.green, POINTS));
        flat.extend(paint::normalize_bins(&h.blue, POINTS));
        flat.extend(paint::normalize_bins(&h.luma, POINTS));
        let shown = motion::morph(ElementId::Name(format!("{}::bins", self.id).into()), flat, window, cx);
        let lo = clip_dot(&self.id, "lo", h.clipped_shadows, hexa(0x4a8aff, 1.0), window, cx);
        let hi = clip_dot(&self.id, "hi", h.clipped_highlights, p.danger, window, cx);
        let grid = p.edge;
        div()
            .relative()
            .w_full()
            .h(px(self.height))
            .rounded(px(4.0))
            .bg(p.well)
            .overflow_hidden()
            .child(
                canvas(
                    |_, _, _| (),
                    move |bounds, _, window, _| {
                        let pts = POINTS;
                        for i in 1..4 {
                            let x = bounds.origin.x + bounds.size.width * (i as f32 / 4.0);
                            paint::segment(
                                window,
                                gpui_kit::point(x, bounds.origin.y),
                                gpui_kit::point(x, bounds.origin.y + bounds.size.height),
                                1.0,
                                grid,
                            );
                        }
                        let (r, rest) = shown.split_at(pts);
                        let (g, rest) = rest.split_at(pts);
                        let (b, l) = rest.split_at(pts);
                        match mode {
                            HistogramMode::Rgb => {
                                paint::fill_area(window, bounds, r, hexa(0xff4a4a, 0.42));
                                paint::fill_area(window, bounds, g, hexa(0x4ade6a, 0.38));
                                paint::fill_area(window, bounds, b, hexa(0x4a8aff, 0.45));
                                let line: Vec<(f32, f32)> =
                                    l.iter().enumerate().map(|(i, v)| (i as f32 / (pts - 1) as f32, *v)).collect();
                                paint::stroke_line(window, bounds, &line, 1.0, hexa(0xffffff, 0.55));
                            }
                            HistogramMode::Luma => paint::fill_area(window, bounds, l, hexa(0xd8d8dc, 0.55)),
                            HistogramMode::Red => paint::fill_area(window, bounds, r, hexa(0xff4a4a, 0.6)),
                            HistogramMode::Green => paint::fill_area(window, bounds, g, hexa(0x4ade6a, 0.6)),
                            HistogramMode::Blue => paint::fill_area(window, bounds, b, hexa(0x4a8aff, 0.65)),
                        }
                    },
                )
                .size_full(),
            )
            .when(self.show_clipping, |this| this.child(lo).child(hi))
    }
}
