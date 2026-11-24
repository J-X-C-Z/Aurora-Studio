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

/// The display-side conversion of one engine frame, with its cost split.
pub(crate) struct PreparedDisplay {
    pub image: Arc<RenderImage>,
    /// Identity of the engine RGBA buffer used to build `image`.
    pub source_pixels: usize,
    /// The previous display image already matched these pixels.
    pub reused: bool,
    /// Time materialising a private pixel buffer (the cache keeps its Arc).
    pub copy_ms: f64,
    /// Time swapping R/B for GPUI's BGRA contract.
    pub convert_ms: f64,
    /// Time building the `RenderImage` around the pixels.
    pub create_ms: f64,
}

fn into_bgra_pixels(img: RgbaImage) -> (u32, u32, Vec<u8>, f64, f64) {
    let width = img.width;
    let height = img.height;
    let convert_at = std::time::Instant::now();
    let pixels = match Arc::try_unwrap(img.pixels) {
        Ok(mut pixels) => {
            // Sole owner (thumbnails): swap in place, no second buffer.
            pixels.par_chunks_exact_mut(4).for_each(|pixel| pixel.swap(0, 2));
            pixels
        }
        Err(shared) => {
            // The preview cache keeps its own Arc of the engine's RGBA frame,
            // so the display needs a private BGRA buffer. Read and write once:
            // the old path cloned the whole frame and swapped in a second pass.
            let row = (width as usize * 4).max(4);
            let mut pixels = vec![0u8; shared.len()];
            pixels
                .par_chunks_mut(row)
                .zip(shared.par_chunks(row))
                .for_each(|(dst, src)| {
                    for (d, s) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                        d[0] = s[2];
                        d[1] = s[1];
                        d[2] = s[0];
                        d[3] = s[3];
                    }
                });
            pixels
        }
    };
    let convert_ms = convert_at.elapsed().as_secs_f64() * 1000.0;
    (width, height, pixels, 0.0, convert_ms)
}

pub(crate) fn to_render_image(img: RgbaImage) -> Arc<RenderImage> {
    to_render_image_timed(img).0
}

fn to_render_image_timed(img: RgbaImage) -> (Arc<RenderImage>, f64, f64, f64) {
    let (width, height, pixels, copy_ms, convert_ms) = into_bgra_pixels(img);
    let create_at = std::time::Instant::now();
    let buffer = image::RgbaImage::from_raw(width, height, pixels).expect("engine returned a buffer of the wrong size");
    let image = Arc::new(RenderImage::new(vec![image::Frame::new(buffer)]));
    let create_ms = create_at.elapsed().as_secs_f64() * 1000.0;
    (image, copy_ms, convert_ms, create_ms)
}

fn prepare_display_image(
    image: RgbaImage,
    previous: Option<(usize, &Arc<RenderImage>)>,
) -> PreparedDisplay {
    let source_pixels = Arc::as_ptr(&image.pixels) as usize;
    if let Some((key, prepared)) = previous {
        if key == source_pixels {
            return PreparedDisplay {
                image: Arc::clone(prepared),
                source_pixels,
                reused: true,
                copy_ms: 0.0,
                convert_ms: 0.0,
                create_ms: 0.0,
            };
        }
    }
    let (image, copy_ms, convert_ms, create_ms) = to_render_image_timed(image);
    PreparedDisplay { image, source_pixels, reused: false, copy_ms, convert_ms, create_ms }
}

fn full_frame_action(previous: Option<(ImageId, NormRect)>, image: ImageId, region: NormRect) -> (bool, bool) {
    let retain = region != NormRect::FULL && previous == Some((image, NormRect::FULL));
    (region == NormRect::FULL || retain, retain)
}

/// What a placement hands back to the caller.
pub(super) enum FrameDisposal {
    /// The frame is no longer shown. `true` when its display image is shared
    /// with a frame that stays, so only overlays are released.
    Frame(Frame, bool),
    /// An overlay of the frame retained as the full backing.
    Overlay(Arc<RenderImage>),
}

/// Places a finished frame into the shown slot, keeping the previous full
/// backing as `full` when a tile replaces it. Disposals run through a callback
/// so the ordering can be exercised without a GPUI app.
pub(super) fn place_frame(
    slot: &mut Option<Frame>,
    full: &mut Option<Frame>,
    next: Frame,
    dispose: &mut impl FnMut(FrameDisposal),
) {
    let (clear_full, keep_previous) = full_frame_action(
        slot.as_ref().map(|frame| (frame.image_id, frame.region)), next.image_id, next.region,
    );
    let next_image = Arc::clone(&next.image);
    if clear_full {
        if let Some(old) = full.take() {
            let reused = Arc::ptr_eq(&old.image, &next_image)
                || slot.as_ref().is_some_and(|frame| Arc::ptr_eq(&old.image, &frame.image));
            dispose(FrameDisposal::Frame(old, reused));
        }
    }
    if let Some(mut old) = slot.replace(next) {
        if keep_previous {
            for overlay in [old.warning.take(), old.focus.take()].into_iter().flatten() {
                dispose(FrameDisposal::Overlay(overlay));
            }
            *full = Some(old);
        } else {
            let reused = Arc::ptr_eq(&old.image, &next_image)
                || full.as_ref().is_some_and(|frame| Arc::ptr_eq(&old.image, &frame.image));
            dispose(FrameDisposal::Frame(old, reused));
        }
    }
}

fn display_frame(slot: &mut Option<Frame>, full: &mut Option<Frame>, next: Frame, cx: &mut gpui_kit::App) {
    place_frame(slot, full, next, &mut |disposal| match disposal {
        FrameDisposal::Frame(frame, true) => frame.release_reusing_image(cx),
        FrameDisposal::Frame(frame, false) => frame.release(cx),
        FrameDisposal::Overlay(overlay) => {
            cx.drop_image(overlay, None);
        }
    });
}

/// Whether a before/after request still has to render the bypass frame.
fn needs_before(
    compare_on: bool,
    before_dirty: bool,
    before: Option<&Frame>,
    image: ImageId,
    geometry: &Geometry,
    region: NormRect,
) -> bool {
    compare_on
        && (before_dirty
            || before.is_none_or(|frame| {
                frame.image_id != image || &frame.geometry != geometry || frame.region != region
            }))
}

/// Whether two requests would produce the same engine frame. The generation
/// differs by definition and the quality does not change interactive pixels
/// (both go through the exact `render_preview` path).
fn same_render_inputs(a: &PreviewRequest, b: &PreviewRequest) -> bool {
    a.image == b.image
        && a.region == b.region
        && a.max_size == b.max_size
        && a.bypass == b.bypass
        && a.mask_overlay == b.mask_overlay
        && a.display_profile == b.display_profile
        && a.proof_profile == b.proof_profile
        && a.proof_paper_white == b.proof_paper_white
        && a.proof_black_point == b.proof_black_point
        && a.rendering_intent == b.rendering_intent
        && a.black_point_compensation == b.black_point_compensation
        && a.adjustments == b.adjustments
}

/// What the scheduler did with a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreviewRequested {
    /// The engine will run it now.
    Started,
    /// One preview is already in flight; this request waits as the newest value.
    Coalesced,
}

impl super::Preview {
    /// Accepts a request. While one is in flight the newest request replaces
    /// the pending one instead of starting a second engine call.
    pub(super) fn accept_request(
        &mut self,
        quality: RenderQuality,
        requested_at: std::time::Instant,
    ) -> PreviewRequested {
        if self.in_flight {
            self.pending = Some((quality, requested_at));
            PreviewRequested::Coalesced
        } else {
            PreviewRequested::Started
        }
    }

    /// Marks the accepted request as handed to the engine.
    pub(super) fn start_request(&mut self) {
        self.in_flight = true;
        self.rendering = true;
    }

    /// Marks the running request finished and returns the newest coalesced one.
    pub(super) fn finish_request(&mut self) -> Option<(RenderQuality, std::time::Instant)> {
        self.in_flight = false;
        let next = self.pending.take();
        if next.is_none() {
            self.rendering = false;
        }
        next
    }

    /// Whether a finished after frame may replace the one on screen: only for
    /// the current image, and never for a generation the shown frame outranks.
    pub(super) fn may_install(
        &self,
        current: Option<ImageId>,
        image_id: ImageId,
        generation: u64,
    ) -> bool {
        let superseded = self
            .frame
            .as_ref()
            .is_some_and(|old| old.image_id == image_id && old.stamp.id > generation);
        current == Some(image_id) && !superseded
    }

    /// Whether the bypass frame requested alongside this one may be installed.
    pub(super) fn may_install_before(&self, current: Option<ImageId>, image_id: ImageId) -> bool {
        current == Some(image_id)
    }

    /// Whether the bypass frame should still be rendered now that the after
    /// frame is done. Turning compare off or changing images drops it.
    pub(super) fn wants_before(
        &self,
        compare_on: bool,
        current: Option<ImageId>,
        image_id: ImageId,
        scheduled: bool,
    ) -> bool {
        scheduled && compare_on && current == Some(image_id)
    }

    /// Thumbnails run only while no preview is in flight and the queue has work.
    pub(super) fn thumbs_allowed(&self, thumb_running: bool, queue_empty: bool) -> bool {
        !thumb_running && !self.in_flight && !queue_empty
    }
}

#[cfg(test)]
mod display_tests {
    use super::*;

    #[test]
    fn retains_only_full_frames_of_the_same_image_across_pan_updates() {
        let full = NormRect::FULL;
        let tile = NormRect { x: 0.5, y: 0.25, width: 0.5, height: 0.5 };
        assert_eq!(full_frame_action(Some((ImageId(1), full)), ImageId(1), tile), (true, true));
        assert_eq!(full_frame_action(Some((ImageId(1), tile)), ImageId(1), tile), (false, false));
        assert_eq!(full_frame_action(Some((ImageId(1), full)), ImageId(2), tile), (false, false));
        assert_eq!(full_frame_action(Some((ImageId(1), tile)), ImageId(1), full), (true, false));
    }

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
        let (w, h, pixels, _, _) = into_bgra_pixels(image);
        assert_eq!((w, h), (2, 1));
        assert_eq!(pixels, vec![3, 2, 1, 4, 7, 6, 5, 8]);

        let shared = Arc::new(vec![9, 8, 7, 6, 5, 4, 3, 2]);
        let source = RgbaImage::from_shared(2, 1, shared.clone());
        let (_, _, converted, _, _) = into_bgra_pixels(source);
        assert_eq!(converted, vec![7, 8, 9, 6, 3, 4, 5, 2]);
        assert_eq!(shared.as_slice(), &[9, 8, 7, 6, 5, 4, 3, 2]);
    }

    #[test]
    fn shared_pixels_convert_in_one_pass_without_touching_the_cache_copy() {
        let (w, h) = (37u32, 23u32);
        let rgba: Vec<u8> = (0..(w * h * 4) as usize).map(|i| (i * 37 % 251) as u8).collect();
        let mut reference = rgba.clone();
        for pixel in reference.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }

        let shared = Arc::new(rgba.clone());
        let source = RgbaImage::from_shared(w, h, Arc::clone(&shared));
        let (cw, ch, converted, copy_ms, _) = into_bgra_pixels(source);
        assert_eq!((cw, ch), (w, h));
        assert_eq!(converted, reference);
        assert_eq!(copy_ms, 0.0, "no separate clone pass remains");
        assert_eq!(shared.as_slice(), rgba.as_slice(), "the cache keeps RGBA");
    }

    #[test]
    fn identical_engine_pixels_reuse_the_prepared_display_image() {
        let pixels = Arc::new(vec![1u8, 2, 3, 255, 4, 5, 6, 255]);
        let source = RgbaImage::from_shared(2, 1, Arc::clone(&pixels));
        let source_key = Arc::as_ptr(&pixels) as usize;
        let prepared = to_render_image(source);

        let same = RgbaImage::from_shared(2, 1, Arc::clone(&pixels));
        let display = prepare_display_image(same, Some((source_key, &prepared)));
        assert!(display.reused);
        assert_eq!(display.source_pixels, source_key);
        assert!(Arc::ptr_eq(&prepared, &display.image));

        let distinct = RgbaImage::new(2, 1, pixels.as_ref().clone());
        let display = prepare_display_image(distinct, Some((source_key, &prepared)));
        assert!(!display.reused);
        assert_ne!(display.source_pixels, source_key);
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
    #[ignore = "profiles 24MP clone-then-swap vs fused BGRA conversion"]
    fn bench_fused_bgra_conversion_24mp() {
        let (w, h) = (6000u32, 4000u32);
        let template: Arc<Vec<u8>> = Arc::new(
            (0..(w * h * 4) as usize).map(|i| (i * 37 % 251) as u8).collect(),
        );
        let old = || {
            let start = std::time::Instant::now();
            let mut pixels = (*template).clone();
            pixels.par_chunks_exact_mut(4).for_each(|pixel| pixel.swap(0, 2));
            std::hint::black_box(pixels);
            start.elapsed().as_secs_f64() * 1000.0
        };
        let new = || {
            let source = RgbaImage::from_shared(w, h, Arc::clone(&template));
            let start = std::time::Instant::now();
            std::hint::black_box(into_bgra_pixels(source));
            start.elapsed().as_secs_f64() * 1000.0
        };
        let (mut old_ms, mut new_ms) = (Vec::new(), Vec::new());
        for _ in 0..3 {
            old_ms.push(old());
            new_ms.push(new());
        }
        old_ms.sort_by(f64::total_cmp);
        new_ms.sort_by(f64::total_cmp);
        println!(
            "24MP BGRA: clone+swap={:.1} ms fused={:.1} ms ({:.2}x)",
            old_ms[1],
            new_ms[1],
            old_ms[1] / new_ms[1].max(f64::EPSILON),
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
/// A single-channel overlay mask. One byte per pixel: warning stores 1 for a
/// clipped highlight and 2 for a clipped shadow, focus stores its alpha.
/// GPUI's `RenderImage` only takes BGRA and has no tint, so the mask is
/// colorized once into the image the viewer draws.
pub(crate) struct OverlayMask {
    width: usize,
    height: usize,
    values: Vec<u8>,
    kind: OverlayKind,
}

#[derive(Clone, Copy)]
enum OverlayKind {
    Warning,
    Focus,
}

impl OverlayMask {
    fn colorized(&self) -> Arc<RenderImage> {
        let mut out = vec![0u8; self.width * self.height * 4];
        out.par_chunks_mut(4)
            .zip(self.values.par_iter())
            .for_each(|(o, &value)| {
                if value == 0 {
                    return;
                }
                match self.kind {
                    // BGRA, as GPUI draws it.
                    OverlayKind::Warning if value == 1 => o.copy_from_slice(&[40, 40, 255, 200]),
                    OverlayKind::Warning => o.copy_from_slice(&[255, 120, 40, 200]),
                    OverlayKind::Focus => o.copy_from_slice(&[80, 255, 80, value]),
                }
            });
        let buffer = image::RgbaImage::from_raw(self.width as u32, self.height as u32, out)
            .expect("overlay buffer size");
        Arc::new(RenderImage::new(vec![image::Frame::new(buffer)]))
    }
}

fn warning_mask(
    px: &[u8],
    w: usize,
    h: usize,
    (high, low, shadows): (u8, u8, bool),
    (r, b): (usize, usize),
) -> OverlayMask {
    let mut values = vec![0u8; w * h];
    values.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = y * w + x;
            let p = &px[i * 4..i * 4 + 4];
            if p[r] >= high || p[1] >= high || p[b] >= high {
                row[x] = 1;
            } else if shadows && p[r] <= low && p[1] <= low && p[b] <= low {
                row[x] = 2;
            }
        }
    });
    OverlayMask { width: w, height: h, values, kind: OverlayKind::Warning }
}

fn focus_mask(px: &[u8], w: usize, h: usize, threshold: f32, (r, b): (usize, usize)) -> OverlayMask {
    let luma = |x: usize, y: usize| {
        let p = &px[(y * w + x) * 4..];
        0.2126 * p[r] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[b] as f32
    };
    let mut values = vec![0u8; w * h];
    values.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y == 0 || y + 1 >= h {
            return;
        }
        for x in 1..w.saturating_sub(1) {
            let gx = luma(x + 1, y) - luma(x - 1, y);
            let gy = luma(x, y + 1) - luma(x, y - 1);
            let g = (gx * gx + gy * gy).sqrt();
            if g > threshold {
                row[x] = ((g - threshold) * 5.0).min(230.0) as u8;
            }
        }
    });
    OverlayMask { width: w, height: h, values, kind: OverlayKind::Focus }
}

/// Builds both overlays from `px`, which is RGBA or BGRA (GPUI display image).
fn analysis_pixels(
    px: &[u8],
    w: usize,
    h: usize,
    bgra: bool,
    warn: Option<(u8, u8, bool)>,
    focus: Option<f32>,
) -> (Option<Arc<RenderImage>>, Option<Arc<RenderImage>>) {
    if w == 0 || h == 0 || px.len() < w * h * 4 {
        return (None, None);
    }
    let channels = if bgra { (2, 0) } else { (0, 2) };
    rayon::join(
        || warn.map(|settings| warning_mask(px, w, h, settings, channels).colorized()),
        || focus.map(|threshold| focus_mask(px, w, h, threshold, channels).colorized()),
    )
}

pub(crate) fn analysis(img: &RgbaImage, warn: Option<(u8, u8, bool)>, focus: Option<f32>) -> (Option<Arc<RenderImage>>, Option<Arc<RenderImage>>) {
    analysis_pixels(img.pixels.as_ref(), img.width as usize, img.height as usize, false, warn, focus)
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

    /// The request inputs without consuming a generation.
    fn preview_inputs(&self, quality: RenderQuality, bypass: bool) -> Option<PreviewRequest> {
        let image = self.current?;
        let region = self.visible_region();
        let (fw, fh) = self.frame_size();
        let scale = self.effective_scale();
        let max_size = ((fw * region.width).max(fh * region.height) * scale).clamp(64.0, 8192.0) as u32;
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

    fn preview_request(&mut self, quality: RenderQuality, bypass: bool) -> Option<PreviewRequest> {
        let request = self.preview_inputs(quality, bypass)?;
        self.preview.generation += 1;
        Some(PreviewRequest { generation: self.preview.generation, ..request })
    }

    fn warning_settings(&self) -> Option<(u8, u8, bool)> {
        self.view
            .exposure_warning
            .then_some((self.settings.warn_high, self.settings.warn_low, self.settings.warn_shadows))
    }

    fn focus_threshold(&self) -> Option<f32> {
        self.view.focus_mask.then_some(self.settings.focus_threshold)
    }

    /// Handles a request whose engine inputs match the shown frame: toggling
    /// the exposure warning or focus mask, or changing their thresholds, only
    /// rebuilds those overlays. Returns true when no render is needed.
    fn refresh_overlays_if_unchanged(&mut self, request: &PreviewRequest, cx: &mut Context<Self>) -> bool {
        if self.preview.error.is_some() || self.view.compare != super::CompareMode::Off {
            return false;
        }
        // Scopes are drawn with the frame; a scope toggle needs a normal render.
        if self.panels.scope_kind != super::ScopeKind::Off || self.preview.last_scope != super::ScopeKind::Off {
            return false;
        }
        let Some(shown) = self.preview.last_request.as_ref() else { return false };
        if !same_render_inputs(shown, request) {
            return false;
        }
        let warning = self.warning_settings();
        let focus = self.focus_threshold();
        if (warning, focus) != self.preview.last_overlay {
            self.rebuild_overlays(warning, focus, cx);
        }
        true
    }

    /// Rebuilds warning/focus from the frame already on screen, in display
    /// (BGRA) order, without asking the engine for the same pixels again.
    fn rebuild_overlays(&mut self, warning: Option<(u8, u8, bool)>, focus: Option<f32>, cx: &mut Context<Self>) {
        let rebuilt = {
            let Some(frame) = self.preview.frame.as_ref() else { return };
            let size = frame.image.size(0);
            let (w, h) = (size.width.0.max(0) as usize, size.height.0.max(0) as usize);
            let Some(bytes) = frame.image.as_bytes(0) else { return };
            analysis_pixels(bytes, w, h, true, warning, focus)
        };
        let Some(frame) = self.preview.frame.as_mut() else { return };
        if let Some(old) = std::mem::replace(&mut frame.warning, rebuilt.0) {
            cx.drop_image(old, None);
        }
        if let Some(old) = std::mem::replace(&mut frame.focus, rebuilt.1) {
            cx.drop_image(old, None);
        }
        self.preview.last_overlay = (warning, focus);
        cx.notify();
    }

    pub fn request_render(&mut self, quality: RenderQuality, cx: &mut Context<Self>) {
        self.preview.draft = quality == RenderQuality::Draft;
        self.enqueue_render(quality, std::time::Instant::now(), cx);
    }

    fn enqueue_render(&mut self, quality: RenderQuality, requested_at: std::time::Instant, cx: &mut Context<Self>) {
        // A new drag supersedes an older release's pending final frame.
        if self.preview.accept_request(quality, requested_at) == PreviewRequested::Coalesced {
            return;
        }
        let Some(mut request) = self.preview_inputs(quality, false) else { return };
        if self.refresh_overlays_if_unchanged(&request, cx) {
            return;
        }
        self.preview.generation += 1;
        request.generation = self.preview.generation;
        let stamp = crate::perf::Stamp::new(request.generation, quality, requested_at);
        let requested_image = request.image;
        let before = needs_before(
            self.view.compare != super::CompareMode::Off,
            self.preview.before_dirty,
            self.preview.before.as_ref(),
            request.image,
            &request.adjustments.geometry,
            request.region,
        )
        .then(|| self.preview_request(RenderQuality::Final, true))
        .flatten();
        if before.is_some() {
            self.preview.before_dirty = false;
        }
        let geometry = request.adjustments.geometry.clone();
        self.preview.start_request();
        let engine = self.engine.clone();
        let st = &self.settings;
        let want_warning = self.view.exposure_warning.then_some((st.warn_high, st.warn_low, st.warn_shadows));
        let want_focus = self.view.focus_mask.then_some(st.focus_threshold);
        let want_scope = self.panels.scope_kind;
        // Recorded with the installed frame so overlay toggles can reuse it.
        let placed_overlay = (want_warning, want_focus);
        let placed_scope = want_scope;
        let previous = self.preview.frame.as_ref().map(|frame| {
            (frame.source_pixels, Arc::clone(&frame.image), frame.image_id)
        });
        let previous_before = self.preview.before.as_ref().map(|frame| {
            (frame.source_pixels, Arc::clone(&frame.image), frame.image_id)
        });
        cx.spawn(async move |this, cx| {
            let after_engine = engine.clone();
            let frame = cx
                .background_spawn(async move {
                    let started_at = std::time::Instant::now();
                    let ram_before = crate::perf::enabled()
                        .then(crate::perf::working_set_bytes)
                        .flatten();
                    let (region, image_id) = (request.region, request.image);
                    after_engine.render_preview(&request).map(move |f| {
                        let engine_wall_ms = started_at.elapsed().as_secs_f64() * 1000.0;
                        let PreviewFrame { image, histogram, elapsed_ms, degraded, .. } = f;
                        let (width, height) = (image.width, image.height);
                        let analysis_at = std::time::Instant::now();
                        let ((overlays, overlay_ms), (scope, scope_ms)) = rayon::join(
                            || {
                                let at = std::time::Instant::now();
                                (analysis(&image, want_warning, want_focus), at.elapsed().as_secs_f64() * 1000.0)
                            },
                            || {
                                let at = std::time::Instant::now();
                                (super::scopes::render(&image, want_scope), at.elapsed().as_secs_f64() * 1000.0)
                            },
                        );
                        let analysis_ms = analysis_at.elapsed().as_secs_f64() * 1000.0;
                        let display = prepare_display_image(
                            image,
                            previous.as_ref().map(|(key, image, _)| (*key, image)),
                        );
                        if crate::perf::enabled() {
                            crate::perf::record_prepared(crate::perf::PreparedTiming {
                                id: stamp.id,
                                image_id: request.image.0,
                                quality: format!("{quality:?}"),
                                width,
                                height,
                                display_reused: display.reused,
                                queue_ms: started_at.duration_since(requested_at).as_secs_f64() * 1000.0,
                                engine_ms: elapsed_ms as f64,
                                engine_wall_ms,
                                analysis_ms,
                                overlay_ms,
                                scope_ms,
                                display_copy_ms: display.copy_ms,
                                display_channel_convert_ms: display.convert_ms,
                                render_image_create_ms: display.create_ms,
                                prepare_ms: started_at.elapsed().as_secs_f64() * 1000.0,
                                total_frame_ms: requested_at.elapsed().as_secs_f64() * 1000.0,
                                ram_before_bytes: ram_before,
                                ram_after_bytes: crate::perf::working_set_bytes(),
                                peak_ram_bytes: crate::perf::peak_working_set_bytes(),
                            });
                        }
                        let PreparedDisplay { image: img, source_pixels, reused, .. } = display;
                        (histogram, elapsed_ms, degraded, img, source_pixels, reused, width, height, overlays, scope, region, image_id, request)
                    })
                })
                .await;
            let Ok(render_before) = this.update(cx, |this, cx| {
                match frame {
                    Ok((histogram, elapsed_ms, degraded, img, source_pixels, reused, width, height, (warning, focus), scope, region, image_id, request)) => {
                        if this.preview.may_install(this.current, image_id, stamp.id) {
                            this.preview.last_request = Some(request);
                            this.preview.last_overlay = placed_overlay;
                            this.preview.last_scope = placed_scope;
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
                            display_frame(&mut this.preview.frame, &mut this.preview.full_after, Frame {
                                image: img,
                                source_pixels,
                                region,
                                geometry: geometry.clone(),
                                image_id,
                                warning,
                                focus,
                                degraded: degraded.map(SharedString::from),
                                stamp,
                            }, cx);
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
                        if this.current == Some(requested_image) {
                            this.preview.error = Some((requested_image, err.to_string().into()));
                        }
                    }
                }
                let render_before = this.preview.wants_before(
                    this.view.compare != super::CompareMode::Off,
                    this.current,
                    requested_image,
                    before.is_some(),
                );
                if render_before {
                    cx.notify();
                }
                render_before
            }) else { return };
            let before = if let Some(request) = before.filter(|_| render_before) {
                Some(cx.background_spawn(async move {
                    engine.render_preview(&request).map(|frame| {
                        let display = prepare_display_image(
                            frame.image, previous_before.as_ref().map(|(key, image, _)| (*key, image)),
                        );
                        (display.image, display.source_pixels, display.reused, request.region, request.image)
                    })
                }).await)
            } else {
                None
            };
            this.update(cx, |this, cx| {
                match before {
                    Some(Ok((img, source_pixels, _, region, image_id))) if this.preview.may_install_before(this.current, image_id) => {
                        display_frame(&mut this.preview.before, &mut this.preview.full_before, Frame {
                            image: img,
                            source_pixels,
                            region,
                            geometry,
                            image_id,
                            warning: None,
                            focus: None,
                            degraded: None,
                            stamp,
                        }, cx);
                    }
                    Some(Ok((img, _, false, _, _))) => cx.drop_image(img, None),
                    Some(Err(_)) if this.current == Some(requested_image) => this.preview.before_dirty = true,
                    _ => {}
                }
                crate::note_first_preview_ready();
                if let Some((next, requested_at)) = this.preview.finish_request() {
                    this.enqueue_render(next, requested_at, cx);
                } else {
                    this.start_queued_thumbs(cx);
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    pub fn queue_thumbs(&mut self, ids: Vec<ImageId>, cx: &mut Context<Self>) {
        self.thumb_queue.push(ids, self.current);
        self.start_queued_thumbs(cx);
    }

    fn start_queued_thumbs(&mut self, cx: &mut Context<Self>) {
        if !self.preview.thumbs_allowed(self.thumb_task.is_some(), self.thumb_queue.is_empty()) {
            return;
        }
        let rev = self.thumb_rev;
        self.thumb_task = Some(cx.spawn(async move |this, cx| {
            loop {
                let next = this
                    .update(cx, |this, _| {
                        if this.thumb_rev != rev || this.preview.in_flight {
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
            this.update(cx, |this, cx| {
                if this.thumb_rev == rev {
                    this.thumb_task = None;
                    this.start_queued_thumbs(cx);
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


/// Preview ordering tests: fake engine + controllable gate.
///
/// These drive the same decision helpers `enqueue_render` uses (`accept_request`,
/// `may_install`, `needs_before`, `wants_before`, `finish_request`,
/// `thumbs_allowed`, `place_frame`) while real engine calls run on threads whose
/// finish order the test chooses. GPUI's `WeakEntity::update` and the image
/// upload path are not exercised here; the session report records that boundary.
#[cfg(test)]
mod ordering_tests {
    use super::*;
    use crate::state::Preview;
    use std::{
        collections::HashMap,
        sync::{
            mpsc::{self, Receiver, Sender},
            Arc, Mutex,
        },
        thread,
        time::{Duration, Instant},
    };

    const FULL: NormRect = NormRect::FULL;
    const TILE: NormRect = NormRect { x: 0.5, y: 0.5, width: 0.5, height: 0.5 };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum FakeReply {
        Frame,
        Fail,
    }

    /// An engine whose preview calls park until the test releases their gate.
    struct FakeEngine {
        pixels: Arc<Vec<u8>>,
        started: Sender<u64>,
        started_rx: Mutex<Receiver<u64>>,
        gates: Mutex<HashMap<u64, Receiver<FakeReply>>>,
        releases: Mutex<HashMap<u64, Sender<FakeReply>>>,
        calls: Mutex<Vec<PreviewRequest>>,
    }

    impl FakeEngine {
        fn new() -> Arc<Self> {
            let (started, started_rx) = mpsc::channel();
            Arc::new(Self {
                pixels: Arc::new(vec![10, 20, 30, 255, 40, 50, 60, 255]),
                started,
                started_rx: Mutex::new(started_rx),
                gates: Mutex::new(HashMap::new()),
                releases: Mutex::new(HashMap::new()),
                calls: Mutex::new(Vec::new()),
            })
        }

        /// Registers where call `generation` waits. Call before spawning it.
        fn gate(&self, generation: u64) {
            let (tx, rx) = mpsc::channel();
            assert!(self.gates.lock().unwrap().insert(generation, rx).is_none(), "one gate per generation");
            self.releases.lock().unwrap().insert(generation, tx);
        }

        fn release(&self, generation: u64, reply: FakeReply) {
            let tx = self.releases.lock().unwrap().remove(&generation).expect("gate registered");
            tx.send(reply).expect("the engine is waiting");
        }

        fn wait_started(&self, generation: u64) {
            let seen = self
                .started_rx
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(5))
                .expect("engine was called");
            assert_eq!(seen, generation);
        }

        fn calls(&self) -> Vec<PreviewRequest> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl ImageEngine for FakeEngine {
        fn name(&self) -> String {
            "fake".into()
        }
        fn images(&self) -> Vec<ImageInfo> {
            Vec::new()
        }
        fn import(&self, _paths: &[std::path::PathBuf]) -> anyhow::Result<Vec<ImageInfo>> {
            anyhow::bail!("fake engine does not import")
        }
        fn collections(&self) -> Vec<Collection> {
            Vec::new()
        }
        fn update_info(&self, _info: &ImageInfo) -> anyhow::Result<()> {
            Ok(())
        }
        fn load_adjustments(&self, _image: ImageId) -> AdjustmentSet {
            AdjustmentSet::default()
        }
        fn save_adjustments(&self, _image: ImageId, _adjustments: &AdjustmentSet) -> anyhow::Result<()> {
            Ok(())
        }
        fn new_variant(&self, _image: ImageId, _adjustments: &AdjustmentSet) -> anyhow::Result<ImageInfo> {
            anyhow::bail!("fake engine has no variants")
        }
        fn thumbnail(&self, _image: ImageId, _adjustments: &AdjustmentSet, _max_size: u32) -> anyhow::Result<RgbaImage> {
            Ok(RgbaImage::from_shared(2, 1, Arc::clone(&self.pixels)))
        }
        fn render_preview(&self, request: &PreviewRequest) -> anyhow::Result<PreviewFrame> {
            self.calls.lock().unwrap().push(request.clone());
            self.started.send(request.generation).ok();
            let gate = self
                .gates
                .lock()
                .unwrap()
                .remove(&request.generation)
                .expect("test registered this gate");
            let reply = gate.recv().expect("test released this gate");
            match reply {
                FakeReply::Frame => Ok(PreviewFrame {
                    image: RgbaImage::from_shared(2, 1, Arc::clone(&self.pixels)),
                    histogram: Histogram::default(),
                    generation: request.generation,
                    elapsed_ms: 0.0,
                    degraded: None,
                    gamut_warning: None,
                }),
                FakeReply::Fail => anyhow::bail!("fake engine failure"),
            }
        }
        fn sample_color(&self, _image: ImageId, _adjustments: &AdjustmentSet, _point: [f32; 2]) -> anyhow::Result<[f32; 3]> {
            Ok([0.0; 3])
        }
        fn auto_adjust(&self, _image: ImageId, _current: &AdjustmentSet, _target: AutoTarget) -> anyhow::Result<AutoResult> {
            anyhow::bail!("fake engine does not auto adjust")
        }
        fn pick_white_balance(&self, _image: ImageId, _point: [f32; 2]) -> anyhow::Result<(f32, f32)> {
            Ok((5500.0, 0.0))
        }
        fn compute_ai_mask(&self, _image: ImageId, _kind: AiMaskKind) -> anyhow::Result<String> {
            anyhow::bail!("fake engine has no AI masks")
        }
        fn icc_profiles(&self) -> Vec<String> {
            Vec::new()
        }
        fn lens_profiles(&self) -> Vec<String> {
            Vec::new()
        }
        fn styles(&self) -> Vec<StyleInfo> {
            Vec::new()
        }
        fn start_export(&self, _job: ExportJob) -> anyhow::Result<()> {
            Ok(())
        }
        fn job_status(&self, _job_id: u64) -> JobStatus {
            JobStatus::Queued
        }
    }

    fn request(image: u64, generation: u64, region: NormRect) -> PreviewRequest {
        PreviewRequest {
            image: ImageId(image),
            adjustments: AdjustmentSet::default(),
            region,
            max_size: 64,
            quality: RenderQuality::Final,
            bypass: false,
            mask_overlay: None,
            generation,
            display_profile: None,
            proof_profile: None,
            proof_paper_white: 1.0,
            proof_black_point: 0.0,
            rendering_intent: None,
            black_point_compensation: false,
        }
    }

    fn frame(image_id: u64, generation: u64, region: NormRect, tag: u8) -> Frame {
        Frame {
            image: to_render_image(RgbaImage::new(2, 1, vec![tag; 8])),
            source_pixels: 0,
            region,
            geometry: Geometry::default(),
            image_id: ImageId(image_id),
            warning: None,
            focus: None,
            degraded: None,
            stamp: crate::perf::Stamp::new(generation, RenderQuality::Final, Instant::now()),
        }
    }

    fn frame_for(image: RgbaImage, request: &PreviewRequest) -> Frame {
        Frame {
            image: to_render_image(image),
            source_pixels: 0,
            region: request.region,
            geometry: request.adjustments.geometry.clone(),
            image_id: request.image,
            warning: None,
            focus: None,
            degraded: None,
            stamp: crate::perf::Stamp::new(request.generation, request.quality, Instant::now()),
        }
    }

    #[derive(Debug, PartialEq)]
    enum Released {
        /// Image id, region, and whether the display image was shared.
        Frame(u64, NormRect, bool),
        Overlay,
    }

    fn record(log: &mut Vec<Released>, disposal: FrameDisposal) {
        match disposal {
            FrameDisposal::Frame(frame, reused) => {
                log.push(Released::Frame(frame.image_id.0, frame.region, reused))
            }
            FrameDisposal::Overlay(_) => log.push(Released::Overlay),
        }
    }

    /// Same install decision and placement as the completion closure.
    fn deliver(
        preview: &mut Preview,
        current: Option<ImageId>,
        request: &PreviewRequest,
        image: RgbaImage,
    ) -> (bool, Vec<Released>) {
        let mut log = Vec::new();
        if !preview.may_install(current, request.image, request.generation) {
            return (false, log);
        }
        preview.error = None;
        let next = frame_for(image, request);
        place_frame(&mut preview.frame, &mut preview.full_after, next, &mut |disposal| {
            record(&mut log, disposal)
        });
        (true, log)
    }

    fn spawn_render(engine: &Arc<FakeEngine>, request: PreviewRequest) -> thread::JoinHandle<anyhow::Result<PreviewFrame>> {
        let engine = Arc::clone(engine);
        thread::spawn(move || engine.render_preview(&request))
    }

    /// Scenario A: switching to B while A renders coalesces B, and A's frame
    /// is discarded instead of covering B.
    #[test]
    fn switching_images_while_a_renders_drops_a_and_runs_b() {
        let engine = FakeEngine::new();
        let mut preview = Preview::default();
        let mut current = Some(ImageId(1));

        assert_eq!(preview.accept_request(RenderQuality::Final, Instant::now()), PreviewRequested::Started);
        preview.start_request();
        let first = request(1, 1, FULL);
        engine.gate(1);
        let job = spawn_render(&engine, first.clone());
        engine.wait_started(1);

        // The user picks B while A is still rendering.
        current = Some(ImageId(2));
        assert_eq!(preview.accept_request(RenderQuality::Final, Instant::now()), PreviewRequested::Coalesced);
        assert_eq!(preview.pending.map(|(quality, _)| quality), Some(RenderQuality::Final));

        engine.release(1, FakeReply::Frame);
        let a_frame = job.join().unwrap().unwrap();
        let (installed, _) = deliver(&mut preview, current, &first, a_frame.image);
        assert!(!installed, "A must not replace B");
        assert!(preview.frame.is_none());

        let queued = preview.finish_request().expect("B waited as the newest value");
        assert_eq!(queued.0, RenderQuality::Final);
        assert_eq!(preview.accept_request(queued.0, queued.1), PreviewRequested::Started);
        preview.start_request();
        let second = request(2, 2, FULL);
        engine.gate(2);
        let job = spawn_render(&engine, second.clone());
        engine.wait_started(2);
        engine.release(2, FakeReply::Frame);
        let b_frame = job.join().unwrap().unwrap();
        let (installed, _) = deliver(&mut preview, current, &second, b_frame.image);
        assert!(installed);
        assert_eq!(preview.frame.as_ref().map(|f| f.image_id), Some(ImageId(2)));
        assert_eq!(engine.calls().len(), 2);
        assert_eq!(preview.finish_request(), None);
        assert!(!preview.rendering);
    }

    /// Scenario B: a frame from an older generation arriving after a newer one
    /// must be discarded, for the same image or after the user moved on.
    #[test]
    fn older_generation_never_overwrites_a_newer_frame() {
        let mut preview = Preview::default();
        let newer = request(1, 7, FULL);
        let (installed, _) = deliver(&mut preview, Some(ImageId(1)), &newer, RgbaImage::new(2, 1, vec![1; 8]));
        assert!(installed);

        let late = request(1, 6, FULL);
        assert!(!preview.may_install(Some(ImageId(1)), late.image, late.generation));
        let (installed, _) = deliver(&mut preview, Some(ImageId(1)), &late, RgbaImage::new(2, 1, vec![2; 8]));
        assert!(!installed);
        assert_eq!(preview.frame.as_ref().map(|f| f.stamp.id), Some(7));

        // A stale frame for A arriving while B is current is dropped too.
        let other = request(1, 8, FULL);
        assert!(!preview.may_install(Some(ImageId(2)), other.image, other.generation));
    }

    /// Scenario C: closing compare while the bypass request is running drops
    /// it; it is never installed behind the after frame.
    #[test]
    fn closing_compare_drops_the_bypass_frame() {
        let preview = Preview::default();
        let after = request(1, 3, FULL);
        assert!(preview.wants_before(true, Some(ImageId(1)), after.image, true));
        assert!(!preview.wants_before(false, Some(ImageId(1)), after.image, true), "compare closed");
        assert!(!preview.wants_before(true, Some(ImageId(2)), after.image, true), "image changed");
        assert!(!preview.wants_before(true, Some(ImageId(1)), after.image, false), "not scheduled");

        assert!(preview.may_install_before(Some(ImageId(1)), after.image));
        assert!(!preview.may_install_before(Some(ImageId(2)), after.image));
        assert!(!preview.may_install_before(None, after.image));
    }

    /// Scenario D: a full backing is kept behind a tile and replaced when the
    /// full frame returns; shared display images are not released twice.
    #[test]
    fn full_backing_survives_a_tile_and_is_released_on_return() {
        let mut slot = Some(frame(1, 1, FULL, 1));
        let mut full = None;

        // Tile replaces the full frame: it is retained as the backing.
        let mut log = Vec::new();
        place_frame(&mut slot, &mut full, frame(1, 2, TILE, 2), &mut |disposal| {
            record(&mut log, disposal)
        });
        assert!(log.is_empty(), "the full backing must not be released");
        assert_eq!(slot.as_ref().map(|f| f.region), Some(TILE));
        assert_eq!(full.as_ref().map(|f| f.region), Some(FULL));

        // Full returns: the backing and the tile are both released.
        let mut log = Vec::new();
        place_frame(&mut slot, &mut full, frame(1, 3, FULL, 3), &mut |disposal| {
            record(&mut log, disposal)
        });
        assert_eq!(
            log,
            vec![
                Released::Frame(1, FULL, false),
                Released::Frame(1, TILE, false),
            ]
        );
        assert_eq!(slot.as_ref().map(|f| f.region), Some(FULL));
        assert!(full.is_none());

        // Shared display image (the engine reused the pixel buffer): once.
        let mut slot = Some(frame(1, 4, FULL, 4));
        let mut full = None;
        let shared_image = Arc::clone(&slot.as_ref().unwrap().image);
        let mut tile = frame(1, 5, TILE, 5);
        tile.image = shared_image;
        place_frame(&mut slot, &mut full, tile, &mut |_| {});
        let mut log = Vec::new();
        place_frame(&mut slot, &mut full, frame(1, 6, FULL, 6), &mut |disposal| {
            record(&mut log, disposal)
        });
        assert_eq!(log[0], Released::Frame(1, FULL, true));
        assert_eq!(log[1], Released::Frame(1, TILE, false));
    }

    /// Scenario E: crop/geometry changes reuse the source image but request a
    /// new after frame and a new bypass frame; an unchanged view does not.
    #[test]
    fn geometry_only_changes_request_new_frames() {
        let mut before = frame(1, 1, FULL, 1);
        before.geometry = Geometry { crop: NormRect { x: 0.0, y: 0.0, width: 0.5, height: 0.5 }, ..Geometry::default() };
        let new_geometry = Geometry { crop: NormRect { x: 0.25, y: 0.25, width: 0.5, height: 0.5 }, ..Geometry::default() };

        assert!(needs_before(true, false, Some(&before), ImageId(1), &new_geometry, FULL), "geometry changed");
        assert!(!needs_before(true, false, Some(&before), ImageId(1), &before.geometry, FULL), "unchanged");
        assert!(needs_before(true, false, Some(&before), ImageId(2), &before.geometry, FULL), "other image");
        assert!(!needs_before(false, true, Some(&before), ImageId(1), &new_geometry, FULL), "compare off");
        assert!(needs_before(true, true, Some(&before), ImageId(1), &before.geometry, FULL), "dirty");

        let mut preview = Preview::default();
        let (installed, _) = deliver(&mut preview, Some(ImageId(1)), &request(1, 2, FULL), RgbaImage::new(2, 1, vec![9; 8]));
        assert!(installed);
        // Same image, newer generation, different geometry: still installs.
        assert!(preview.may_install(Some(ImageId(1)), ImageId(1), 3));
    }

    /// Scenario F: a failed preview clears its in-flight slot and the newest
    /// pending request still runs.
    #[test]
    fn a_failed_preview_keeps_the_pending_request_running() {
        let engine = FakeEngine::new();
        let mut preview = Preview::default();
        let current = Some(ImageId(1));

        assert_eq!(preview.accept_request(RenderQuality::Final, Instant::now()), PreviewRequested::Started);
        preview.start_request();
        let first = request(1, 1, FULL);
        engine.gate(1);
        let job = spawn_render(&engine, first.clone());
        engine.wait_started(1);
        assert_eq!(preview.accept_request(RenderQuality::Final, Instant::now()), PreviewRequested::Coalesced);

        engine.release(1, FakeReply::Fail);
        let error = job.join().unwrap().unwrap_err();
        // Production records the error only for the current image.
        assert_eq!(current, Some(first.image));
        preview.error = Some((first.image, error.to_string().into()));

        let queued = preview.finish_request().expect("pending survives the failure");
        assert_eq!(queued.0, RenderQuality::Final);
        assert_eq!(preview.accept_request(queued.0, queued.1), PreviewRequested::Started);
        preview.start_request();
        let second = request(1, 2, FULL);
        engine.gate(2);
        let job = spawn_render(&engine, second.clone());
        engine.wait_started(2);
        engine.release(2, FakeReply::Frame);
        let frame = job.join().unwrap().unwrap();
        let (installed, _) = deliver(&mut preview, current, &second, frame.image);
        assert!(installed);
        assert!(preview.error.is_none());
        assert_eq!(preview.frame.as_ref().map(|f| f.stamp.id), Some(2));
    }

    /// Scenario G: a superseded generation's display image is released and
    /// nothing is installed; the reuse path does not release it twice.
    #[test]
    fn superseded_generation_releases_the_prepared_image() {
        let mut preview = Preview::default();
        let shown = request(1, 5, FULL);
        let (installed, _) = deliver(&mut preview, Some(ImageId(1)), &shown, RgbaImage::new(2, 1, vec![1; 8]));
        assert!(installed);

        let late = request(1, 4, FULL);
        let image = RgbaImage::new(2, 1, vec![2; 8]);
        let pixels = Arc::clone(&image.pixels);
        assert!(!preview.may_install(Some(ImageId(1)), late.image, late.generation));
        // The production discard branch drops the display image unless the
        // engine reused the previous one.
        drop(to_render_image(image));
        assert_eq!(Arc::strong_count(&pixels), 1, "the discarded frame owns no pixels");

        let image = RgbaImage::new(2, 1, vec![3; 8]);
        let pixels = Arc::clone(&image.pixels);
        assert!(!preview.may_install(Some(ImageId(1)), late.image, late.generation));
        assert_eq!(Arc::strong_count(&pixels), 2, "a reused result is not released here");
    }

    /// Scenario H: thumbnails pause while a preview is in flight and resume
    /// once the queue has no pending preview.
    #[test]
    fn thumbnails_pause_for_a_preview_and_resume_after_it() {
        let mut preview = Preview::default();
        assert!(preview.thumbs_allowed(false, false));
        assert!(!preview.thumbs_allowed(true, false), "a thumbnail task is running");
        assert!(!preview.thumbs_allowed(false, true), "nothing queued");

        preview.start_request();
        assert!(!preview.thumbs_allowed(false, false), "preview in flight");
        assert_eq!(preview.accept_request(RenderQuality::Draft, Instant::now()), PreviewRequested::Coalesced);
        assert!(preview.finish_request().is_some());
        // The coalesced request is starting, so thumbnails stay paused.
        assert!(!preview.in_flight);
        assert_eq!(preview.accept_request(RenderQuality::Draft, Instant::now()), PreviewRequested::Started);
        preview.start_request();
        assert!(!preview.thumbs_allowed(false, false));
        assert_eq!(preview.finish_request(), None);
        assert!(preview.thumbs_allowed(false, false), "resumed when idle");
    }

    /// Overlay-only handling needs identical engine inputs; generation and
    /// quality differ per request but do not change interactive pixels.
    #[test]
    fn overlay_only_requests_share_render_inputs() {
        let a = request(1, 5, FULL);
        let mut newer = a.clone();
        newer.generation = 99;
        newer.quality = RenderQuality::Draft;
        assert!(same_render_inputs(&a, &newer));

        let mut region = a.clone();
        region.region = TILE;
        assert!(!same_render_inputs(&a, &region));

        let mut edited = a.clone();
        edited.adjustments.exposure.exposure = 0.5;
        assert!(!same_render_inputs(&a, &edited));

        let mut profile = a.clone();
        profile.display_profile = Some(std::path::PathBuf::from("display.icc"));
        assert!(!same_render_inputs(&a, &profile));
    }

    /// The overlay-only path reads GPUI's BGRA display frame; it must produce
    /// the same layers as the RGBA engine path.
    #[test]
    fn bgra_analysis_matches_rgba_analysis() {
        let (w, h) = (37u32, 23u32);
        let rgba: Vec<u8> = (0..(w * h) as usize)
            .flat_map(|i| {
                [
                    (i * 37 % 251) as u8,
                    (i * 73 % 241) as u8,
                    (i * 109 % 239) as u8,
                    255,
                ]
            })
            .collect();
        let settings = (210, 30, true);
        let expected = analysis(&RgbaImage::new(w, h, rgba.clone()), Some(settings), Some(7.5));

        let (_, _, bgra, _, _) = into_bgra_pixels(RgbaImage::new(w, h, rgba));
        let (warning, focus) = analysis_pixels(&bgra, w as usize, h as usize, true, Some(settings), Some(7.5));
        assert_eq!(
            expected.0.expect("rgba warning").as_bytes(0).unwrap(),
            warning.expect("bgra warning").as_bytes(0).unwrap()
        );
        assert_eq!(
            expected.1.expect("rgba focus").as_bytes(0).unwrap(),
            focus.expect("bgra focus").as_bytes(0).unwrap()
        );
    }

    /// Scenario I: a viewer destroyed while a render is running retains no
    /// frame, and the late result is released without touching state.
    #[test]
    fn a_destroyed_viewer_does_not_retain_the_background_result() {
        let engine = FakeEngine::new();
        let request = request(1, 1, FULL);
        engine.gate(1);
        let job = spawn_render(&engine, request.clone());
        engine.wait_started(1);

        // The viewer goes away; the task's WeakEntity update would fail here.
        let viewer = Preview::default();
        drop(viewer);

        engine.release(1, FakeReply::Frame);
        let result = job.join().unwrap().unwrap();
        assert_eq!(Arc::strong_count(&engine.pixels), 2, "engine + the late frame");
        drop(result);
        assert_eq!(Arc::strong_count(&engine.pixels), 1, "the late frame released its pixels");
    }
}
