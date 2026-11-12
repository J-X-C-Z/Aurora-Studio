//! Video scopes for the preview: luma waveform, RGB parade and vectorscope.
//! Like the exposure warning, this is display analysis of the pixels the engine
//! returned, drawn into a small BGRA image the scopes panel shows.

use std::sync::Arc;

use gpui_kit::RenderImage;
use lumen_engine::RgbaImage;

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
}

/// Output size: waveforms are W×H, the vectorscope is H×H.
const W: usize = 256;
const H: usize = 160;
/// At most this many source pixels are sampled; plenty for a scope.
const SAMPLES: usize = 160_000;

fn luma(r: u8, g: u8, b: u8) -> f32 {
    0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32
}

/// Counts to brightness: a square root keeps sparse traces visible next to dense ones.
fn tone(count: u32, max: u32) -> f32 {
    if count == 0 { 0.0 } else { (count as f32 / max.max(1) as f32).sqrt().max(0.12) }
}

fn finish(w: usize, h: usize, bgra: Vec<u8>) -> Arc<RenderImage> {
    let buffer = image::RgbaImage::from_raw(w as u32, h as u32, bgra).expect("scope buffer size");
    Arc::new(RenderImage::new(vec![image::Frame::new(buffer)]))
}

/// Draws `kind` for `img`, or `None` when scopes are off or the frame is empty.
pub(crate) fn render(img: &RgbaImage, kind: ScopeKind) -> Option<Arc<RenderImage>> {
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0 || h == 0 || kind == ScopeKind::Off {
        return None;
    }
    let px = img.pixels.as_ref();
    let step = ((w * h) as f32 / SAMPLES as f32).sqrt().ceil().max(1.0) as usize;
    let samples = || (0..h).step_by(step).flat_map(move |y| (0..w).step_by(step).map(move |x| (x, &px[(y * w + x) * 4..(y * w + x) * 4 + 3])));
    // Graticule: faint lines at 0, 25, 50, 75 and 100 %.
    let grid = |out: &mut [u8], ow: usize| {
        for q in 0..=4 {
            let y = (H - 1) * q / 4;
            for x in 0..ow {
                let o = &mut out[(y * ow + x) * 4..(y * ow + x) * 4 + 4];
                o.copy_from_slice(&[60, 60, 60, 255]);
            }
        }
    };
    Some(match kind {
        ScopeKind::Waveform | ScopeKind::Parade => {
            // Parade: three side-by-side waveforms, one per channel.
            let lanes = if kind == ScopeKind::Parade { 3 } else { 1 };
            let lane_w = W / lanes;
            let mut counts = vec![0u32; W * H];
            for (x, p) in samples() {
                for lane in 0..lanes {
                    let v = if lanes == 1 { luma(p[0], p[1], p[2]) } else { p[lane] as f32 };
                    let col = lane * lane_w + (x * lane_w / w).min(lane_w - 1);
                    let row = H - 1 - ((v / 255.0) * (H - 1) as f32).round() as usize;
                    counts[row * W + col] += 1;
                }
            }
            let max = *counts.iter().max().unwrap_or(&1);
            let mut out = vec![0u8; W * H * 4];
            for i in 0..W * H {
                out[i * 4 + 3] = 255;
            }
            grid(&mut out, W);
            for (i, c) in counts.iter().enumerate() {
                let t = tone(*c, max);
                if t == 0.0 {
                    continue;
                }
                let lane = (i % W) / lane_w;
                // BGRA: luma traces are white-green, parade lanes take their channel's color.
                let rgb = match (lanes, lane) {
                    (1, _) => [0.75, 1.0, 0.8],
                    (_, 0) => [1.0, 0.35, 0.35],
                    (_, 1) => [0.35, 1.0, 0.45],
                    _ => [0.4, 0.55, 1.0],
                };
                let o = &mut out[i * 4..i * 4 + 4];
                o[0] = (rgb[2] * t * 255.0) as u8;
                o[1] = (rgb[1] * t * 255.0) as u8;
                o[2] = (rgb[0] * t * 255.0) as u8;
            }
            finish(W, H, out)
        }
        ScopeKind::Vectorscope => {
            let n = H;
            let mut counts = vec![0u32; n * n];
            for (_, p) in samples() {
                let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
                let y = luma(p[0], p[1], p[2]);
                // BT.709 chroma, ±0.5 of full scale mapped to the circle.
                let cb = (b - y) / 1.8556 / 255.0;
                let cr = (r - y) / 1.5748 / 255.0;
                let _ = g;
                let x = ((0.5 + cb) * (n - 1) as f32).round().clamp(0.0, (n - 1) as f32) as usize;
                let yy = ((0.5 - cr) * (n - 1) as f32).round().clamp(0.0, (n - 1) as f32) as usize;
                counts[yy * n + x] += 1;
            }
            let max = *counts.iter().max().unwrap_or(&1);
            let mut out = vec![0u8; n * n * 4];
            let c = (n - 1) as f32 / 2.0;
            for yy in 0..n {
                for x in 0..n {
                    let i = yy * n + x;
                    let o = &mut out[i * 4..i * 4 + 4];
                    o[3] = 255;
                    let d = ((x as f32 - c).powi(2) + (yy as f32 - c).powi(2)).sqrt();
                    // Outer circle, center cross and the skin-tone line (about 123°).
                    let skin = {
                        let (dx, dy) = (x as f32 - c, c - yy as f32);
                        let a = dy.atan2(dx).to_degrees();
                        (a - 123.0).abs() < 0.8 && d < c
                    };
                    if (d - c).abs() < 0.8 || ((x as f32 - c).abs() < 0.5 || (yy as f32 - c).abs() < 0.5) && d < c || skin {
                        o.copy_from_slice(&[60, 60, 60, 255]);
                    }
                    let t = tone(counts[i], max);
                    if t > 0.0 {
                        // The trace takes the hue it sits on, lifted toward white.
                        let (cb, cr) = (x as f32 / (n - 1) as f32 - 0.5, 0.5 - yy as f32 / (n - 1) as f32);
                        let r = (0.6 + 1.5748 * cr).clamp(0.0, 1.0);
                        let b = (0.6 + 1.8556 * cb).clamp(0.0, 1.0);
                        let g = (0.6 - 0.1873 * cb - 0.4681 * cr).clamp(0.0, 1.0);
                        o[0] = (b * t * 255.0) as u8;
                        o[1] = (g * t * 255.0) as u8;
                        o[2] = (r * t * 255.0) as u8;
                    }
                }
            }
            finish(n, n, out)
        }
        ScopeKind::Off => unreachable!(),
    })
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
}
