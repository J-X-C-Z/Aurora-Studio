//! Preview and thumbnail rendering. One preview is in flight at a time; requests
//! made meanwhile collapse into a single follow-up with the newest state.

use std::sync::Arc;

use gpui_kit::{AppContext, Context, RenderImage};
use lumen_engine::*;

use super::{AppState, CursorTool, Frame};

pub(crate) fn to_render_image(img: &RgbaImage) -> Arc<RenderImage> {
    let (pixels, remainder) = img.pixels.as_chunks::<4>();
    let mut bgra = pixels.iter().map(|&[r, g, b, a]| [b, g, r, a]).collect::<Vec<_>>().into_flattened();
    bgra.extend_from_slice(remainder);
    let buffer = image::RgbaImage::from_raw(img.width, img.height, bgra).expect("engine returned a buffer of the wrong size");
    Arc::new(RenderImage::new(vec![image::Frame::new(buffer)]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clone_swap(img: &RgbaImage) -> Arc<RenderImage> {
        let mut bgra = img.pixels.as_ref().clone();
        for px in bgra.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
        let buffer = image::RgbaImage::from_raw(img.width, img.height, bgra).expect("engine returned a buffer of the wrong size");
        Arc::new(RenderImage::new(vec![image::Frame::new(buffer)]))
    }

    #[test]
    fn swizzle_preserves_dimensions_alpha_and_shared_source() {
        let rgba = vec![255, 0, 0, 0, 0, 255, 0, 1, 0, 0, 255, 127, 12, 34, 56, 128, 255, 255, 255, 254, 9, 8, 7, 255];
        let img = RgbaImage::new(3, 2, rgba.clone());
        let shared = Arc::clone(&img.pixels);
        let rendered = to_render_image(&img);
        assert_eq!(rendered.frame_count(), 1);
        assert_eq!((u32::from(rendered.size(0).width), u32::from(rendered.size(0).height)), (3, 2));
        assert_eq!(rendered.as_bytes(0).unwrap(), &[0, 0, 255, 0, 0, 255, 0, 1, 255, 0, 0, 127, 56, 34, 12, 128, 255, 255, 255, 254, 7, 8, 9, 255]);
        assert!(Arc::ptr_eq(&shared, &img.pixels));
        assert_eq!(Arc::strong_count(&shared), 2);
        assert_eq!(shared.as_slice(), rgba.as_slice());
    }

    #[test]
    fn swizzle_matches_clone_swap_for_empty_odd_and_trailing_bytes() {
        for (width, height) in [(0, 0), (1, 1), (3, 1), (17, 3), (257, 1)] {
            // from_raw accepts excess bytes; keep the old conversion's remainder too.
            for extra in 0..4 {
                let rgba = (0..width * height * 4 + extra).map(|i| ((i * 73 + 19) % 256) as u8).collect::<Vec<_>>();
                let img = RgbaImage { width, height, pixels: Arc::new(rgba.clone()) };
                let rendered = to_render_image(&img);
                assert_eq!(rendered.as_bytes(0), clone_swap(&img).as_bytes(0));
                assert_eq!((u32::from(rendered.size(0).width), u32::from(rendered.size(0).height)), (width, height));
                assert_eq!(img.pixels.as_slice(), rgba.as_slice());
            }
        }
    }

    #[test]
    #[should_panic(expected = "engine returned a buffer of the wrong size")]
    fn swizzle_rejects_short_buffer() {
        to_render_image(&RgbaImage { width: 2, height: 1, pixels: Arc::new(vec![1, 2, 3, 4]) });
    }

    #[test]
    #[ignore = "manual release benchmark; run with --release --ignored --nocapture --test-threads=1"]
    fn benchmark_to_render_image() {
        use std::{hint::black_box, time::Instant};

        for (width, height, batch) in [(360, 240, 200), (1280, 720, 30), (1920, 1080, 12), (3840, 2160, 3)] {
            let img = RgbaImage::new(width, height, (0..width * height * 4).map(|i| ((i * 73 + 19) % 256) as u8).collect());
            assert_eq!(to_render_image(&img).as_bytes(0), clone_swap(&img).as_bytes(0));
            let methods: [fn(&RgbaImage) -> Arc<RenderImage>; 2] = [clone_swap, to_render_image];
            for method in methods { for _ in 0..4 { drop(black_box(method(black_box(&img)))); } }
            let mut samples = [Vec::new(), Vec::new()];
            for round in 0..31 {
                for offset in 0..2 {
                    let index = (round + offset) % 2;
                    let method = black_box(methods[index]);
                    let start = Instant::now();
                    for _ in 0..batch { drop(black_box(method(black_box(&img)))); }
                    samples[index].push(start.elapsed().as_secs_f64() * 1000.0 / batch as f64);
                }
            }
            for (name, times) in ["clone_swap", "copy_swizzle"].into_iter().zip(&mut samples) {
                times.sort_by(f64::total_cmp);
                eprintln!("{width}x{height} {name}: p50={:.6} ms p95={:.6} ms", times[times.len() / 2], times[(times.len() - 1) * 95 / 100]);
            }
        }
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
    let warning = warn.map(|(high, low, shadows)| {
        let mut out = vec![0u8; w * h * 4];
        for (i, p) in px.chunks_exact(4).enumerate() {
            let o = &mut out[i * 4..i * 4 + 4];
            // BGRA.
            if p[0] >= high || p[1] >= high || p[2] >= high {
                o.copy_from_slice(&[40, 40, 255, 200]);
            } else if shadows && p[0] <= low && p[1] <= low && p[2] <= low {
                o.copy_from_slice(&[255, 120, 40, 200]);
            }
        }
        make(out)
    });
    let focus = focus.map(|threshold| {
        let luma = |x: usize, y: usize| {
            let p = &px[(y * w + x) * 4..];
            0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
        };
        let mut out = vec![0u8; w * h * 4];
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let gx = luma(x + 1, y) - luma(x - 1, y);
                let gy = luma(x, y + 1) - luma(x, y - 1);
                let g = (gx * gx + gy * gy).sqrt();
                if g > threshold {
                    let a = ((g - threshold) * 5.0).min(230.0) as u8;
                    out[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&[80, 255, 80, a]);
                }
            }
        }
        make(out)
    });
    (warning, focus)
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
         display_profile: None, proof_profile: None, proof_paper_white: 1.0, proof_black_point: 0.0, })
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
        cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move {
                    let started_at = std::time::Instant::now();
                    let frame = engine.render_preview(&request).map(|f| {
                        let prepared_at = std::time::Instant::now();
                        let img = to_render_image(&f.image);
                        let overlays = analysis(&f.image, want_warning, want_focus);
                        let scope = super::scopes::render(&f.image, want_scope);
                        if crate::perf::enabled() { crate::perf::record(serde_json::json!({
                            "event": "prepared", "id": stamp.id, "quality": format!("{:?}", quality),
                            "queue_ms": started_at.duration_since(requested_at).as_secs_f64() * 1000.0,
                            "engine_ms": f.elapsed_ms, "prepare_ms": prepared_at.elapsed().as_secs_f64() * 1000.0,
                            "width": f.image.width, "height": f.image.height,
                        })); }
                        (f, img, overlays, scope, request.region, request.image)
                    });
                    let before = before.map(|b| {
                        engine.render_preview(&b).map(|f| {
                            let img = to_render_image(&f.image);
                            (f, img, b.region, b.image)
                        })
                    });
                    (frame, before)
                })
                .await;
            this.update(cx, |this, cx| {
                this.preview.in_flight = false;
                let (frame, before) = result;
                match frame {
                    Ok((f, img, (warning, focus), scope, region, image_id)) => {
                        if this.current == Some(image_id) {
                            if let Some(old) = std::mem::replace(&mut this.preview.scope, scope) {
                                cx.drop_image(old, None);
                            }
                            this.preview.error = None;
                            this.source_note = crate::planned::source_status(this.engine.as_ref(), image_id).map(|n| (image_id, n.into()));
                            let old = this.preview.frame.replace(Frame {
                                image: img,
                                region,
                                image_id,
                                warning,
                                focus,
                                stamp,
                            });
                            if let Some(old) = old {
                                old.release(cx);
                            }
                            log::debug!("preview: {}×{} in {:.1} ms", f.image.width, f.image.height, f.elapsed_ms);
                            this.preview.histogram = f.histogram;
                            this.preview.elapsed_ms = f.elapsed_ms;
                        } else {
                            cx.drop_image(img, None);
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
                if let Some(Ok((_, img, region, image_id))) = before {
                    if this.current != Some(image_id) {
                        cx.drop_image(img, None);
                    } else {
                        this.preview.before_dirty = false;
                        let old = this.preview.before.replace(Frame {
                            image: img,
                            region,
                            image_id,
                            warning: None,
                            focus: None,
                            stamp,
                        });
                        if let Some(old) = old {
                            old.release(cx);
                        }
                    }
                }
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
        for id in ids {
            if !self.thumb_queue.contains(&id) {
                self.thumb_queue.push(id);
            }
        }
        if self.thumb_task.is_some() {
            return;
        }
        self.thumb_task = Some(cx.spawn(async move |this, cx| {
            loop {
                let next = this
                    .update(cx, |this, _| {
                        if this.thumb_queue.is_empty() {
                            return None;
                        }
                        let id = this.thumb_queue.remove(0);
                        let adj = this.edits.get(&id).map(|e| super::flattened(&e.adj)).unwrap_or_default();
                        Some((id, adj, this.engine.clone()))
                    })
                    .ok()
                    .flatten();
                let Some((id, adj, engine)) = next else { break };
                let result = cx
                    .background_spawn(async move { engine.thumbnail(id, &adj, 360).map(|t| to_render_image(&t)) })
                    .await;
                this.update(cx, |this, cx| {
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
            this.update(cx, |this, _| this.thumb_task = None).ok();
        }));
    }
}

