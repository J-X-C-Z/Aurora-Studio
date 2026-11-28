//! Video scopes for the preview: luma waveform, RGB parade and vectorscope.
//! Like the exposure warning, this is display analysis of the pixels the engine
//! returned, drawn into a small BGRA image the scopes panel shows.
//!
//! Counting stays here — it only needs the column and the value, and one pass
//! per frame already runs in the panel's own background task. The density maps
//! are laid out by `aurora_engine::scopes`, whose drawing then turns them into
//! the panel image; the vectorscope graticule and per-cell trace colour are
//! precomputed there instead of running `atan2` 25 600 times per frame.

use std::sync::Arc;

use gpui_kit::RenderImage;
use aurora_engine::{
    viewer_vector_image, viewer_waveform_image, RgbaImage, ScopeMaps, ScopeWants, VIEWER_COLUMNS,
    VIEWER_LEVELS, VIEWER_LANES, VIEWER_MAX_SAMPLES, VIEWER_VECTOR, VIEWER_WORDS,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScopeKind {
    #[default]
    Off,
    Waveform,
    Parade,
    Vectorscope,
}

impl ScopeKind {
    pub const ALL: [Self; 4] = [Self::Off, Self::Waveform, Self::Parade, Self::Vectorscope];

    /// The engine-side map this kind draws from.
    pub(crate) fn wants(self) -> ScopeWants {
        match self {
            Self::Off => ScopeWants::default(),
            Self::Waveform => ScopeWants::WAVEFORM,
            Self::Parade => ScopeWants::PARADE,
            Self::Vectorscope => ScopeWants::VECTOR,
        }
    }
}

/// Output size: waveforms are W×H, the vectorscope is H×H.
const W: usize = VIEWER_COLUMNS;
const H: usize = VIEWER_LEVELS;
/// At most this many source pixels are sampled; plenty for a scope.
const SAMPLES: usize = VIEWER_MAX_SAMPLES;

fn luma(r: u8, g: u8, b: u8) -> f32 {
    0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32
}

/// The level a 0–255 value lands on, rounded half away from zero. The engine's
/// counting loop derives the same bins, and
/// `scopes::viewer_tests::rounding_matches_the_panel_for_every_byte_triple`
/// pins the two forms together.
fn level(value: f32) -> usize {
    ((value / 255.0) * (H - 1) as f32).round() as usize
}

fn finish(w: usize, h: usize, bgra: Vec<u8>) -> Arc<RenderImage> {
    let buffer = image::RgbaImage::from_raw(w as u32, h as u32, bgra).expect("scope buffer size");
    Arc::new(RenderImage::new(vec![image::Frame::new(buffer)]))
}

/// Draws `kind` for `img`, or `None` when scopes are off or the frame is empty.
pub(crate) fn render(img: &RgbaImage, kind: ScopeKind) -> Option<Arc<RenderImage>> {
    render_pixels(img.pixels.as_ref(), img.width as usize, img.height as usize, false, kind)
}

/// Draws `kind` from maps the engine already counted on the device.
///
/// This is the fast path: no counting, just the density map becoming the panel
/// image. `None` when the engine did not fill this kind, so the caller falls
/// back to [`render_pixels`] instead of drawing an empty map.
pub(crate) fn render_maps(maps: &ScopeMaps, kind: ScopeKind) -> Option<Arc<RenderImage>> {
    if kind == ScopeKind::Off || maps.is_empty() || maps.wants() != kind.wants() {
        return None;
    }
    let lanes = if kind == ScopeKind::Parade { VIEWER_LANES } else { 1 };
    let (out_w, out_h, bgra_out) = match kind {
        ScopeKind::Waveform | ScopeKind::Parade => {
            (W, H, viewer_waveform_image(maps, lanes))
        }
        ScopeKind::Vectorscope => {
            let n = VIEWER_VECTOR;
            (n, n, viewer_vector_image(maps))
        }
        ScopeKind::Off => unreachable!(),
    };
    Some(finish(out_w, out_h, bgra_out))
}

/// Draws `kind` directly from RGBA or BGRA bytes (e.g. a GPUI `RenderImage`).
pub(crate) fn render_pixels(
    px: &[u8],
    w: usize,
    h: usize,
    bgra: bool,
    kind: ScopeKind,
) -> Option<Arc<RenderImage>> {
    if w == 0 || h == 0 || kind == ScopeKind::Off || px.len() < w * h * 4 {
        return None;
    }
    let (r_idx, b_idx) = if bgra { (2, 0) } else { (0, 2) };
    let step = ((w * h) as f32 / SAMPLES as f32).sqrt().ceil().max(1.0) as usize;
    let lanes = if kind == ScopeKind::Parade { VIEWER_LANES } else { 1 };
    let lane_w = W / lanes;
    let parade_base = VIEWER_LEVELS * VIEWER_COLUMNS;
    let vector_base = 2 * VIEWER_LEVELS * VIEWER_COLUMNS;
    let mut words = vec![0u32; VIEWER_WORDS];
    let mut sampled = 0u32;
    match kind {
        ScopeKind::Waveform | ScopeKind::Parade => {
            let base = if lanes == 1 { 0 } else { parade_base };
            for y in (0..h).step_by(step) {
                for x in (0..w).step_by(step) {
                    let o = (y * w + x) * 4;
                    let (r, g, b) = (px[o + r_idx], px[o + 1], px[o + b_idx]);
                    sampled += 1;
                    for lane in 0..lanes {
                        let v = if lanes == 1 { luma(r, g, b) } else { [r, g, b][lane] as f32 };
                        let col = lane * lane_w + (x * lane_w / w).min(lane_w - 1);
                        let row = H - 1 - level(v);
                        words[base + row * W + col] += 1;
                    }
                }
            }
        }
        ScopeKind::Vectorscope => {
            let n = VIEWER_VECTOR;
            for y in (0..h).step_by(step) {
                for x in (0..w).step_by(step) {
                    let o = (y * w + x) * 4;
                    let (r, g, b) = (px[o + r_idx], px[o + 1], px[o + b_idx]);
                    sampled += 1;
                    let y_luma = luma(r, g, b);
                    // BT.709 chroma, ±0.5 of full scale mapped onto the circle.
                    let cb = (b as f32 - y_luma) / 1.8556 / 255.0;
                    let cr = (r as f32 - y_luma) / 1.5748 / 255.0;
                    let vx = (((0.5 + cb) * (n - 1) as f32).round() as usize).min(n - 1);
                    let vy = (((0.5 - cr) * (n - 1) as f32).round() as usize).min(n - 1);
                    words[vector_base + vy * n + vx] += 1;
                }
            }
        }
        ScopeKind::Off => unreachable!(),
    }
    let maps = ScopeMaps::from_owned_words(words, sampled);
    let (out_w, out_h, bgra_out) = match kind {
        ScopeKind::Waveform | ScopeKind::Parade => {
            (W, H, viewer_waveform_image(&maps, lanes))
        }
        ScopeKind::Vectorscope => {
            let n = VIEWER_VECTOR;
            (n, n, viewer_vector_image(&maps))
        }
        ScopeKind::Off => unreachable!(),
    };
    Some(finish(out_w, out_h, bgra_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(r: u8, g: u8, b: u8) -> RgbaImage {
        RgbaImage::new(8, 8, [r, g, b, 255].repeat(64))
    }

    #[test]
    fn off_draws_nothing_and_each_kind_draws_its_size() {
        let img = flat(128, 128, 128);
        assert!(render(&img, ScopeKind::Off).is_none());
        for kind in [ScopeKind::Waveform, ScopeKind::Parade, ScopeKind::Vectorscope] {
            let s = render(&img, kind).unwrap();
            let size = s.size(0);
            let expect = if kind == ScopeKind::Vectorscope { (H, H) } else { (W, H) };
            assert_eq!((size.width.0 as usize, size.height.0 as usize), expect);
        }
    }

    #[test]
    fn a_flat_frame_lights_its_trace_row() {
        // Cover every output column: an eight-pixel image correctly lights only eight bins.
        let img = RgbaImage::new(W as u32, 8, [0, 0, 0, 255].repeat(W * 8));
        let s = render(&img, ScopeKind::Waveform).unwrap();
        let bytes = s.as_bytes(0).unwrap();
        // Black sits on the bottom row, which is also graticule row 0, so the
        // trace has to out-shine the 60/60/60 line there and nowhere else.
        let bottom = (H - 1) * W;
        for x in 0..W {
            assert!(bytes[(bottom + x) * 4] > 100, "bottom trace at column {x}");
        }
        let middle = (H / 2) * W;
        assert!(
            (0..W).all(|x| bytes[(middle + x) * 4] <= 100),
            "no trace in the middle rows"
        );
    }

    /// The picture drawn from engine-counted maps has to be the picture this
    /// module's own counting draws. That is the whole point of taking the maps
    /// from the renderer: the panel must not change what it shows.
    #[test]
    fn drawing_engine_maps_matches_drawing_local_counts() {
        let (w, h) = (37usize, 23usize);
        let mut px = vec![0u8; w * h * 4];
        for (i, p) in px.chunks_exact_mut(4).enumerate() {
            let x = (i % w) as u8;
            let y = (i / w) as u8;
            p[0] = x.wrapping_mul(7).wrapping_add(y.wrapping_mul(3));
            p[1] = 90u8.wrapping_add(y.wrapping_mul(5));
            p[2] = 200u8.wrapping_sub(x.wrapping_mul(4));
            p[3] = 255;
        }
        for kind in [ScopeKind::Waveform, ScopeKind::Parade, ScopeKind::Vectorscope] {
            let maps = aurora_engine::viewer_maps(&px, w, h, true, kind.wants());
            let from_maps = render_maps(&maps, kind).expect("engine maps for this kind");
            let from_pixels = render_pixels(&px, w, h, true, kind).expect("local counts");
            assert_eq!(
                from_maps.as_bytes(0).unwrap(),
                from_pixels.as_bytes(0).unwrap(),
                "{kind:?}: engine-counted map draws a different picture"
            );
        }
    }

    /// Maps that answer for another kind are never drawn as if they did.
    #[test]
    fn maps_for_another_kind_are_refused() {
        let (w, h) = (8usize, 8usize);
        let px = [90u8, 120, 200, 255].repeat(w * h);
        let maps = aurora_engine::viewer_maps(&px, w, h, true, ScopeKind::Waveform.wants());
        assert!(render_maps(&maps, ScopeKind::Parade).is_none());
        assert!(render_maps(&maps, ScopeKind::Waveform).is_some());
        assert!(render_maps(&maps, ScopeKind::Off).is_none());
    }
}
