//! Preview and thumbnail rendering. One preview is in flight at a time; requests
//! made meanwhile collapse into a single follow-up with the newest state.

use std::{
    collections::{HashSet, VecDeque},
    sync::Arc,
};

use gpui_kit::{AppContext, Context, RenderImage, SharedString};
use aurora_engine::*;
use rayon::prelude::*;

use super::{AppState, CursorTool, Frame};
use crate::i18n::t;

#[derive(Default)]
pub(super) struct ThumbQueue {
    order: VecDeque<ImageId>,
    queued: HashSet<ImageId>,
}

impl ThumbQueue {
    pub(super) fn push(&mut self, ids: impl IntoIterator<Item = ImageId>, current: Option<ImageId>) {
        for id in ids {
            if self.queued.insert(id) {
                self.order.push_back(id);
            }
        }
        if let Some(id) = current {
            self.prioritize(id);
        }
    }

    pub(super) fn prioritize(&mut self, id: ImageId) {
        if self.order.front().copied() == Some(id) || !self.queued.contains(&id) {
            return;
        }
        self.order.retain(|queued| *queued != id);
        self.order.push_front(id);
    }

    pub(super) fn pop_front(&mut self) -> Option<ImageId> {
        let id = self.order.pop_front()?;
        self.queued.remove(&id);
        Some(id)
    }

    pub(super) fn clear(&mut self) {
        self.order.clear();
        self.queued.clear();
    }

    pub(super) fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

fn into_bgra_pixels(img: RgbaImage) -> (u32, u32, Vec<u8>) {
    let width = img.width;
    let height = img.height;
    let mut pixels = Arc::try_unwrap(img.pixels).unwrap_or_else(|shared| (*shared).clone());
    pixels.par_chunks_exact_mut(4).for_each(|pixel| pixel.swap(0, 2));
    (width, height, pixels)
}

pub(crate) fn to_render_image(img: RgbaImage) -> Arc<RenderImage> {
    let (width, height, pixels) = into_bgra_pixels(img);
    let buffer = image::RgbaImage::from_raw(width, height, pixels).expect("engine returned a buffer of the wrong size");
    Arc::new(RenderImage::new(vec![image::Frame::new(buffer)]))
}

fn prepare_display_image(
    image: RgbaImage,
    previous: Option<(usize, &Arc<RenderImage>)>,
) -> (Arc<RenderImage>, usize, bool) {
    let source_pixels = Arc::as_ptr(&image.pixels) as usize;
    if let Some((key, prepared)) = previous {
        if key == source_pixels {
            return (Arc::clone(prepared), source_pixels, true);
        }
    }
    (to_render_image(image), source_pixels, false)
}

#[cfg(test)]
mod display_tests {
    use super::*;

    fn serial_overlays(
        img: &RgbaImage,
        warn: (u8, u8, bool),
        focus: f32,
    ) -> (Vec<u8>, Vec<u8>) {
        let (w, h) = (img.width as usize, img.height as usize);
        let px = img.pixels.as_ref();
        let (high, low, shadows) = warn;
        let mut warning = vec![0u8; w * h * 4];
        for (i, p) in px.chunks_exact(4).enumerate() {
            let o = &mut warning[i * 4..i * 4 + 4];
            if p[0] >= high || p[1] >= high || p[2] >= high {
                o.copy_from_slice(&[40, 40, 255, 200]);
            } else if shadows && p[0] <= low && p[1] <= low && p[2] <= low {
                o.copy_from_slice(&[255, 120, 40, 200]);
            }
        }
        let luma = |x: usize, y: usize| {
            let p = &px[(y * w + x) * 4..];
            0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
        };
        let mut focus_out = vec![0u8; w * h * 4];
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let gx = luma(x + 1, y) - luma(x - 1, y);
                let gy = luma(x, y + 1) - luma(x, y - 1);
                let g = (gx * gx + gy * gy).sqrt();
                if g > focus {
                    let a = ((g - focus) * 5.0).min(230.0) as u8;
                    focus_out[(y * w + x) * 4..(y * w + x) * 4 + 4]
                        .copy_from_slice(&[80, 255, 80, a]);
                }
            }
        }
        (warning, focus_out)
    }

    #[test]
    fn bgra_conversion_swaps_only_red_and_blue() {
        let image = RgbaImage::new(2, 1, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let (w, h, pixels) = into_bgra_pixels(image);
        assert_eq!((w, h), (2, 1));
        assert_eq!(pixels, vec![3, 2, 1, 4, 7, 6, 5, 8]);

        let shared = Arc::new(vec![9, 8, 7, 6, 5, 4, 3, 2]);
        let source = RgbaImage::from_shared(2, 1, shared.clone());
        let (_, _, converted) = into_bgra_pixels(source);
        assert_eq!(converted, vec![7, 8, 9, 6, 3, 4, 5, 2]);
        assert_eq!(shared.as_slice(), &[9, 8, 7, 6, 5, 4, 3, 2]);
    }

    #[test]
    fn identical_engine_pixels_reuse_the_prepared_display_image() {
        let pixels = Arc::new(vec![1u8, 2, 3, 255, 4, 5, 6, 255]);
        let source = RgbaImage::from_shared(2, 1, Arc::clone(&pixels));
        let source_key = Arc::as_ptr(&pixels) as usize;
        let prepared = to_render_image(source);

        let same = RgbaImage::from_shared(2, 1, Arc::clone(&pixels));
        let (reused, key, hit) =
            prepare_display_image(same, Some((source_key, &prepared)));
        assert!(hit);
        assert_eq!(key, source_key);
        assert!(Arc::ptr_eq(&prepared, &reused));

        let distinct = RgbaImage::new(2, 1, pixels.as_ref().clone());
        let (_, key, hit) = prepare_display_image(distinct, Some((source_key, &prepared)));
        assert!(!hit);
        assert_ne!(key, source_key);
    }

    #[test]
    fn parallel_analysis_matches_serial_bytes() {
        let (w, h) = (37u32, 23u32);
        let pixels: Vec<u8> = (0..(w * h) as usize)
            .flat_map(|i| {
                [
                    (i * 37 % 251) as u8,
                    (i * 73 % 241) as u8,
                    (i * 109 % 239) as u8,
                    255,
                ]
            })
            .collect();
        let image = RgbaImage::new(w, h, pixels);
        let expected = serial_overlays(&image, (210, 30, true), 7.5);
        let (warning, focus) = analysis(&image, Some((210, 30, true)), Some(7.5));
        assert_eq!(warning.unwrap().as_bytes(0).unwrap(), expected.0);
        assert_eq!(focus.unwrap().as_bytes(0).unwrap(), expected.1);
    }

    #[test]
    fn thumb_queue_dedupes_and_prioritizes_the_current_image() {
        let id = |n| ImageId(n);
        let mut queue = ThumbQueue::default();
        queue.push([id(1), id(2), id(3)], Some(id(3)));
        assert_eq!(queue.pop_front(), Some(id(3)));
        queue.push([id(2), id(4), id(1)], Some(id(4)));
        assert_eq!(queue.pop_front(), Some(id(4)));
        assert_eq!(queue.pop_front(), Some(id(1)));
        assert_eq!(queue.pop_front(), Some(id(2)));
        assert_eq!(queue.pop_front(), None);

        queue.push([id(5), id(5)], None);
        assert_eq!(queue.pop_front(), Some(id(5)));
        assert_eq!(queue.pop_front(), None);
    }

    #[test]
    #[ignore = "profiles 10k-item thumbnail queue bookkeeping"]
    fn bench_thumb_queue_bookkeeping() {
        const IDS: usize = 10_000;
        let ids: Vec<ImageId> = (0..IDS as u64).map(ImageId).collect();
        let mut old_times = Vec::new();
        let mut new_times = Vec::new();
        for _ in 0..5 {
            let start = std::time::Instant::now();
            let mut old = Vec::new();
            for id in &ids {
                if !old.contains(id) {
                    old.push(*id);
                }
            }
            while !old.is_empty() {
                std::hint::black_box(old.remove(0));
            }
            old_times.push(start.elapsed().as_secs_f64() * 1000.0);

            let start = std::time::Instant::now();
            let mut new = ThumbQueue::default();
            new.push(ids.iter().copied(), None);
            while let Some(id) = new.pop_front() {
                std::hint::black_box(id);
            }
            new_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        old_times.sort_by(f64::total_cmp);
        new_times.sort_by(f64::total_cmp);
        println!(
            "10k thumbnail queue bookkeeping: old={:.3} ms new={:.3} ms ({:.2}x)",
            old_times[2],
            new_times[2],
            old_times[2] / new_times[2].max(f64::EPSILON),
        );
    }

    #[test]
    #[ignore = "profiles 4K exposure warning and focus overlays"]
    fn bench_analysis_overlays() {
        let (w, h) = (3840u32, 2160u32);
        let pixels: Vec<u8> = (0..(w * h) as usize)
            .flat_map(|i| [
                (i * 37 % 251) as u8,
                (i * 73 % 241) as u8,
                (i * 109 % 239) as u8,
                255,
            ])
            .collect();
        let image = RgbaImage::new(w, h, pixels);
        let mut serial = Vec::new();
        let mut parallel = Vec::new();
        for _ in 0..3 {
            let start = std::time::Instant::now();
            std::hint::black_box(serial_overlays(&image, (240, 20, true), 8.0));
            serial.push(start.elapsed().as_secs_f64() * 1000.0);
            let start = std::time::Instant::now();
            std::hint::black_box(analysis(&image, Some((240, 20, true)), Some(8.0)));
            parallel.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        serial.sort_by(f64::total_cmp);
        parallel.sort_by(f64::total_cmp);
        println!(
            "4K analysis overlays: serial={:.3} ms parallel={:.3} ms ({:.2}x)",
            serial[1],
            parallel[1],
            serial[1] / parallel[1].max(f64::EPSILON),
        );
    }

    #[test]
    #[ignore = "profiles 4K RGBA to BGRA display conversion"]
    fn bench_display_bgra_conversion() {
        let (w, h) = (3840u32, 2160u32);
        let template: Vec<u8> = (0..(w * h * 4) as usize)
            .map(|i| (i * 37 % 251) as u8)
            .collect();
        let run = |shared: bool| {
            let image = if shared {
                let pixels = Arc::new(template.clone());
                let keep_alive = pixels.clone();
                let image = RgbaImage::from_shared(w, h, pixels);
                (image, Some(keep_alive))
            } else {
                (RgbaImage::new(w, h, template.clone()), None)
            };
            let start = std::time::Instant::now();
            std::hint::black_box(into_bgra_pixels(image.0));
            std::hint::black_box(image.1);
            start.elapsed().as_secs_f64() * 1000.0
        };
        let run_old = || {
            let image = RgbaImage::new(w, h, template.clone());
            let start = std::time::Instant::now();
            let mut pixels = image.pixels.as_ref().clone();
            for pixel in pixels.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
            std::hint::black_box(pixels);
            start.elapsed().as_secs_f64() * 1000.0
        };
        let (mut old, mut unique, mut shared) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..3 {
            old.push(run_old());
            unique.push(run(false));
            shared.push(run(true));
        }
        old.sort_by(f64::total_cmp);
        unique.sort_by(f64::total_cmp);
        shared.sort_by(f64::total_cmp);
        println!(
            "4K display BGRA: old={:.3} ms unique={:.3} ms ({:.2}x) shared={:.3} ms ({:.2}x)",
            old[1],
            unique[1],
            old[1] / unique[1].max(f64::EPSILON),
            shared[1],
            old[1] / shared[1].max(f64::EPSILON),
        );
    }

    #[test]
    #[ignore = "profiles repeated preparation of an unchanged 4K display frame"]
    fn bench_display_image_reuse() {
        let (w, h) = (3840u32, 2160u32);
        let pixels: Arc<Vec<u8>> = Arc::new(
            (0..(w * h * 4) as usize)
                .map(|i| (i * 37 % 251) as u8)
                .collect(),
        );
        let source_key = Arc::as_ptr(&pixels) as usize;
        let prepared = to_render_image(RgbaImage::from_shared(w, h, Arc::clone(&pixels)));
        let mut convert = Vec::new();
        let mut reuse = Vec::new();
        for _ in 0..5 {
            let start = std::time::Instant::now();
            std::hint::black_box(to_render_image(RgbaImage::from_shared(
                w,
                h,
                Arc::clone(&pixels),
            )));
            convert.push(start.elapsed().as_secs_f64() * 1000.0);

            let start = std::time::Instant::now();
            std::hint::black_box(prepare_display_image(
                RgbaImage::from_shared(w, h, Arc::clone(&pixels)),
                Some((source_key, &prepared)),
            ));
            reuse.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        convert.sort_by(f64::total_cmp);
        reuse.sort_by(f64::total_cmp);
        println!(
            "4K unchanged display prep: convert={:.3} ms reuse={:.3} ms ({:.2}x)",
            convert[2],
            reuse[2],
            convert[2] / reuse[2].max(f64::EPSILON)
        );
    }
}

/// Builds the exposure-warning and focus-mask overlays for a processed frame.
/// This is display analysis of the pixels the engine returned, so it lives here.
/// Exposure-warning and focus-mask overlays for a rendered frame. `warn` is
/// (highlight level, shadow level, mark shadows); `focus` the edge threshold.
pub(crate) fn analysis(img: &RgbaImage, warn: Option<(u8, u8, bool)>, focus: Option<f32>) -> (Option<Arc<RenderImage>>, Option<Arc<RenderImage>>) {
    let (w, h) = (img.width as usize, img.height as usize);
    let px = img.pixels.as_ref();
    let make = |data: Vec<u8>| {
        let buffer = image::RgbaImage::from_raw(w as u32, h as u32, data).expect("overlay buffer size");
        Arc::new(RenderImage::new(vec![image::Frame::new(buffer)]))
    };
    let warning = || warn.map(|(high, low, shadows)| {
        let mut out = vec![0u8; w * h * 4];
        out.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let i = y * w + x;
                let p = &px[i * 4..i * 4 + 4];
                let o = &mut row[x * 4..x * 4 + 4];
                // BGRA.
                if p[0] >= high || p[1] >= high || p[2] >= high {
                    o.copy_from_slice(&[40, 40, 255, 200]);
                } else if shadows && p[0] <= low && p[1] <= low && p[2] <= low {
                    o.copy_from_slice(&[255, 120, 40, 200]);
                }
            }
        });
        make(out)
    });
    let focus = || focus.map(|threshold| {
        let luma = |x: usize, y: usize| {
            let p = &px[(y * w + x) * 4..];
            0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
        };
        let mut out = vec![0u8; w * h * 4];
        out.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
            if y == 0 || y + 1 >= h {
                return;
            }
            for x in 1..w.saturating_sub(1) {
                let gx = luma(x + 1, y) - luma(x - 1, y);
                let gy = luma(x, y + 1) - luma(x, y - 1);
                let g = (gx * gx + gy * gy).sqrt();
                if g > threshold {
                    let a = ((g - threshold) * 5.0).min(230.0) as u8;
                    row[x * 4..x * 4 + 4].copy_from_slice(&[80, 255, 80, a]);
                }
            }
        });
        make(out)
    });
    rayon::join(warning, focus)
}

impl AppState {
    /// Adjustments as the viewer should draw them: the crop tool shows the whole frame.
    pub fn display_adjustments(&self) -> AdjustmentSet {
        let mut adj = super::flattened(self.hover_look.as_ref().unwrap_or(self.adj()));
        if self.cursor == CursorTool::Crop {
            adj.geometry.crop = NormRect::FULL;
        }
        adj
    }

    /// Developed-frame size in source pixels for the current image as displayed.
    pub fn frame_size(&self) -> (f32, f32) {
        let Some(info) = self.current_info() else { return (3.0, 2.0) };
        let adj = self.adj();
        let (w, h) = if adj.geometry.orientation % 2 == 1 {
            (info.height as f32, info.width as f32)
        } else {
            (info.width as f32, info.height as f32)
        };
        let crop = if self.cursor == CursorTool::Crop { NormRect::FULL } else { adj.geometry.crop };
        (w * crop.width.max(1e-3), h * crop.height.max(1e-3))
    }

    /// Scale (device pixels per frame pixel) at which the frame fits the viewport.
    pub fn fit_scale(&self) -> f32 {
        let (fw, fh) = self.frame_size();
        let (vw, vh) = self.viewport_inner();
        (vw * self.scale_factor / fw).min(vh * self.scale_factor / fh)
    }

    /// Viewport minus the breathing margin around a fitted image, in logical pixels.
    pub fn viewport_inner(&self) -> (f32, f32) {
        let margin = 24.0;
        ((self.viewport.0 - margin * 2.0).max(32.0), (self.viewport.1 - margin * 2.0).max(32.0))
    }

    pub fn effective_scale(&self) -> f32 {
        self.zoom.scale.unwrap_or_else(|| self.fit_scale())
    }

    /// The part of the frame visible at the current zoom, normalized.
    pub fn visible_region(&self) -> NormRect {
        let Some(scale) = self.zoom.scale else { return NormRect::FULL };
        let (fw, fh) = self.frame_size();
        let w = (self.viewport.0 * self.scale_factor / (scale * fw)).min(1.0);
        let h = (self.viewport.1 * self.scale_factor / (scale * fh)).min(1.0);
        let (cx, cy) = self.zoom.center;
        NormRect { x: (cx - w / 2.0).clamp(0.0, 1.0 - w), y: (cy - h / 2.0).clamp(0.0, 1.0 - h), width: w, height: h }
    }

    pub(crate) fn clamp_center(&mut self) {
        let r = self.visible_region();
        self.zoom.center = (r.x + r.width / 2.0, r.y + r.height / 2.0);
    }

    fn preview_request(&mut self, quality: RenderQuality, bypass: bool) -> Option<PreviewRequest> {
        let image = self.current?;
        let region = self.visible_region();
        let (fw, fh) = self.frame_size();
        let scale = self.effective_scale();
        let max_size = ((fw * region.width).max(fh * region.height) * scale).clamp(64.0, 8192.0) as u32;
        self.preview.generation += 1;
        let mask_overlay = if self.panels.mask_overlay || self.cursor.is_mask_tool() { self.active_layer } else { None };
        Some(PreviewRequest {
            image,
            adjustments: self.display_adjustments(),
            region,
            max_size,
            quality,
            bypass,
            mask_overlay: mask_overlay.filter(|_| !bypass),
            generation: self.preview.generation,
            display_profile: self.settings.display_profile.clone().or_else(|| crate::state::detected_display_profile(self.window_hwnd)),
            proof_profile: self.view.soft_proof.then(|| self.settings.proof_profile.clone()).flatten(),
            proof_paper_white: if self.view.soft_proof { self.settings.proof_paper_white } else { 1.0 },
            proof_black_point: if self.view.soft_proof { self.settings.proof_black_point } else { 0.0 },
            rendering_intent: None,
            black_point_compensation: false,
        })
    }

    pub fn request_render(&mut self, quality: RenderQuality, cx: &mut Context<Self>) {
        self.preview.draft = quality == RenderQuality::Draft;
        self.enqueue_render(quality, std::time::Instant::now(), cx);
    }

    fn enqueue_render(&mut self, quality: RenderQuality, requested_at: std::time::Instant, cx: &mut Context<Self>) {
        if self.preview.in_flight {
            // A new drag supersedes an older release's pending final frame.
            self.preview.pending = Some((quality, requested_at));
            return;
        }
        let Some(request) = self.preview_request(quality, false) else { return };
        let stamp = crate::perf::Stamp::new(request.generation, quality, requested_at);
        let requested_image = request.image;
        let before = (self.view.compare != super::CompareMode::Off && (self.preview.before_dirty || self.preview.before.is_none()))
            .then(|| self.preview_request(RenderQuality::Final, true))
            .flatten();
        self.preview.in_flight = true;
        self.preview.rendering = true;
        let engine = self.engine.clone();
        let st = &self.settings;
        let want_warning = self.view.exposure_warning.then_some((st.warn_high, st.warn_low, st.warn_shadows));
        let want_focus = self.view.focus_mask.then_some(st.focus_threshold);
        let want_scope = self.panels.scope_kind;
        let previous = self.preview.frame.as_ref().map(|frame| {
            (frame.source_pixels, Arc::clone(&frame.image), frame.image_id)
        });
        let previous_before = self.preview.before.as_ref().map(|frame| {
            (frame.source_pixels, Arc::clone(&frame.image), frame.image_id)
        });
        cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move {
                    let started_at = std::time::Instant::now();
                    let frame = engine.render_preview(&request).map(|f| {
                        let PreviewFrame { image, histogram, elapsed_ms, degraded, .. } = f;
                        let (width, height) = (image.width, image.height);
                        let prepared_at = std::time::Instant::now();
                        let (overlays, scope) = rayon::join(
                            || analysis(&image, want_warning, want_focus),
                            || super::scopes::render(&image, want_scope),
                        );
                        if crate::perf::enabled() { crate::perf::record(serde_json::json!({
                            "event": "prepared", "id": stamp.id, "quality": format!("{:?}", quality),
                            "queue_ms": started_at.duration_since(requested_at).as_secs_f64() * 1000.0,
                            "engine_ms": elapsed_ms, "prepare_ms": prepared_at.elapsed().as_secs_f64() * 1000.0,
                            "width": image.width, "height": image.height,
                        })); }
                        let (img, source_pixels, reused) = prepare_display_image(
                            image,
                            previous.as_ref().map(|(key, image, _)| (*key, image)),
                        );
                        (histogram, elapsed_ms, degraded, img, source_pixels, reused, width, height, overlays, scope, request.region, request.image)
                    });
                    let before = before.map(|b| {
                        engine.render_preview(&b).map(|f| {
                            let (img, source_pixels, reused) = prepare_display_image(
                                f.image,
                                previous_before.as_ref().map(|(key, image, _)| (*key, image)),
                            );
                            (img, source_pixels, reused, b.region, b.image)
                        })
                    });
                    (frame, before)
                })
                .await;
            this.update(cx, |this, cx| {
                this.preview.in_flight = false;
                let (frame, before) = result;
                match frame {
                    Ok((histogram, elapsed_ms, degraded, img, source_pixels, reused, width, height, (warning, focus), scope, region, image_id)) => {
                        let stale = this.preview.frame.as_ref().is_some_and(|old| old.image_id == image_id && old.stamp.id > stamp.id);
                        if this.current == Some(image_id) && !stale {
                            if let Some(old) = std::mem::replace(&mut this.preview.scope, scope) {
                                cx.drop_image(old, None);
                            }
                            this.preview.error = None;
                            this.source_note = crate::planned::source_status(this.engine.as_ref(), image_id).map(|n| {
                                log::warn!("source status: {n}");
                                let summary = match SourceStatusKind::of(&n) {
                                    SourceStatusKind::DecodeFailed => t("RAW decode failed; showing the embedded preview", "RAW 解码失败，显示嵌入预览"),
                                    SourceStatusKind::UnsupportedCompression => t("unsupported RAW compression; showing the embedded preview", "不支持该 RAW 压缩，显示嵌入预览"),
                                    SourceStatusKind::Other => t("showing an embedded preview", "显示嵌入预览"),
                                };
                                (image_id, summary.into())
                            });
                            let old = this.preview.frame.replace(Frame {
                                image: img,
                                source_pixels,
                                region,
                                image_id,
                                warning,
                                focus,
                                degraded: degraded.map(SharedString::from),
                                stamp,
                            });
                            if let Some(old) = old {
                                if reused {
                                    old.release_reusing_image(cx);
                                } else {
                                    old.release(cx);
                                }
                            }
                            log::debug!("preview: {}×{} in {:.1} ms", width, height, elapsed_ms);
                            this.preview.histogram = histogram;
                            this.preview.elapsed_ms = elapsed_ms;
                        } else {
                            if !reused {
                                cx.drop_image(img, None);
                            }
                            if let Some(scope) = scope {
                                cx.drop_image(scope, None);
                            }
                        }
                    }
                    Err(err) => {
                        log::error!("preview: render failed: {err}");
                        this.preview.error = Some((requested_image, err.to_string().into()));
                    }
                }
                if let Some(Ok((img, source_pixels, reused, region, image_id))) = before {
                    if this.current != Some(image_id) {
                        if !reused {
                            cx.drop_image(img, None);
                        }
                    } else {
                        this.preview.before_dirty = false;
                        let old = this.preview.before.replace(Frame {
                            image: img,
                            source_pixels,
                            region,
                            image_id,
                            warning: None,
                            focus: None,
                            degraded: None,
                            stamp,
                        });
                        if let Some(old) = old {
                            if reused {
                                old.release_reusing_image(cx);
                            } else {
                                old.release(cx);
                            }
                        }
                    }
                }
                crate::note_first_preview_ready();
                if let Some((next, requested_at)) = this.preview.pending.take() {
                    this.enqueue_render(next, requested_at, cx);
                } else {
                    this.preview.rendering = false;
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    pub fn queue_thumbs(&mut self, ids: Vec<ImageId>, cx: &mut Context<Self>) {
        self.thumb_queue.push(ids, self.current);
        if self.thumb_task.is_some() {
            return;
        }
        let rev = self.thumb_rev;
        self.thumb_task = Some(cx.spawn(async move |this, cx| {
            loop {
                let next = this
                    .update(cx, |this, _| {
                        if this.thumb_rev != rev {
                            return None;
                        }
                        if this.thumb_queue.is_empty() {
                            return None;
                        }
                        if let Some(id) = this.current {
                            this.thumb_queue.prioritize(id);
                        }
                        let id = this.thumb_queue.pop_front()?;
                        let adj = this.edits.get(&id).map(|e| super::flattened(&e.adj)).unwrap_or_default();
                        let display = this.settings.display_profile.clone().or_else(|| crate::state::detected_display_profile(this.window_hwnd));
                        let proof = this.view.soft_proof.then(|| this.settings.proof_profile.clone()).flatten();
                        let levels = if this.view.soft_proof { (this.settings.proof_paper_white, this.settings.proof_black_point) } else { (1.0, 0.0) };
                        Some((id, adj, this.engine.clone(), display, proof, levels))
                    })
                    .ok()
                    .flatten();
                let Some((id, adj, engine, display, proof, levels)) = next else { break };
                let result = cx
                    .background_spawn(async move { engine.thumbnail_profiled(id, &adj, 360, display.as_deref(), proof.as_deref(), levels).map(to_render_image) })
                    .await;
                this.update(cx, |this, cx| {
                    if this.thumb_rev != rev {
                        return;
                    }
                    match result {
                        Ok(img) => {
                            this.thumb_failed.remove(&id);
                            if let Some(old) = this.thumbs.insert(id, img) {
                                cx.drop_image(old, None);
                            }
                        }
                        Err(err) => {
                            log::error!("thumbnail {id:?}: {err}");
                            this.thumb_failed.insert(id);
                        }
                    }
                    cx.notify();
                })
                .ok();
            }
            this.update(cx, |this, _| {
                if this.thumb_rev == rev {
                    this.thumb_task = None;
                }
            })
            .ok();
        }));
    }

    /// Drops cached thumbnails and rebuilds them after a display/profile change.
    pub fn refresh_thumbs(&mut self, cx: &mut Context<Self>) {
        for (_, img) in self.thumbs.drain() {
            cx.drop_image(img, None);
        }
        self.thumb_failed.clear();
        self.thumb_queue.clear();
        self.thumb_rev += 1;
        self.thumb_task = None;
        let ids = self.visible_images().iter().map(|i| i.id).collect();
        self.queue_thumbs(ids, cx);
    }
}

