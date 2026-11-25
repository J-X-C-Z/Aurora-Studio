//! Compositional services over [`FileEngine`].
//!
//! `FileEngine` started as one large `impl` holding catalog, source, render and
//! persistence responsibilities. These services borrow the engine and own one
//! domain's operations; they deliberately keep no state and no second lock.
//! The trait methods stay on `FileEngine` as thin delegations so the public API
//! (and every existing caller) is unchanged.
//!
//! Migration status: catalog, source, render, export, merge, persistence,
//! inference, capture, import/scan and the catalog-mutating library operations
//! (move/copy/rename/locate/synchronize/trash) are moved here. The library
//! path helpers and their tests still live in `library_ops.rs`, and the
//! `FileEngine` entry points remain one-line delegations.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::Context as _;
use rayon::prelude::*;

use crate::catalog::{Collection, ImageId, ImageInfo};
use crate::save_queue::{PathState, SubmitOutcome};
use crate::save_report::DocumentSaveResult;
use crate::{LibraryAction, UserCollection};

use super::library_ops::{
    canonical_root, copy_no_replace, ensure_absent, entry_record, existing_directory, expand_sidecars,
    folder_is_offline, is_private, move_no_replace, path_key, recycle_paths, rel_key, resolve_inside,
    rollback, selected_paths, sidecars, validate_name,
};
use super::*;

/// Read-only catalog and metadata queries.
pub(crate) struct CatalogService<'a> {
    engine: &'a FileEngine,
}

impl<'a> CatalogService<'a> {
    pub(crate) fn new(engine: &'a FileEngine) -> Self {
        Self { engine }
    }

    pub(crate) fn images(&self) -> Vec<ImageInfo> {
        self.engine.state.lock().entries.iter().map(|entry| entry.info.clone()).collect()
    }

    pub(crate) fn collections(&self) -> Vec<Collection> {
        let c1 = self.engine.state.lock().c1_collections.clone();
        if c1.is_empty() {
            self.engine.collection_tree()
        } else {
            c1
        }
    }

    pub(crate) fn collection_images(&self, collection: u64) -> Option<Vec<ImageId>> {
        if let Some(ids) = self.engine.state.lock().c1_members.get(&collection).cloned() {
            return Some(ids);
        }
        let index = collection.checked_sub(super::FOLDER_BASE)? as usize;
        let dir = self.engine.folder_dirs().get(index)?.0.clone();
        let st = self.engine.state.lock();
        Some(
            st.entries
                .iter()
                .filter(|entry| match entry.rel_path.rfind('/') {
                    Some(at) => entry.rel_path[..at] == dir,
                    None => dir.is_empty(),
                })
                .map(|entry| entry.info.id)
                .collect(),
        )
    }

    pub(crate) fn library_root(&self) -> Option<PathBuf> {
        Some(self.engine.root.clone())
    }

    pub(crate) fn collection_path(&self, id: u64) -> Option<PathBuf> {
        let index = id.checked_sub(super::FOLDER_BASE)? as usize;
        let (dir, _) = self.engine.folder_dirs().get(index)?.clone();
        Some(if dir.is_empty() {
            self.engine.root.clone()
        } else {
            self.engine.root.join(dir)
        })
    }

    pub(crate) fn user_collections(&self) -> Vec<crate::UserCollection> {
        self.engine
            .store
            .as_ref()
            .map(|store| store.load_user_collections())
            .unwrap_or_default()
    }

    pub(crate) fn source_status(&self, image: ImageId) -> Option<String> {
        self.engine.state.lock().source_reasons.get(&image).cloned()
    }

    pub(crate) fn layer_summaries(&self, image: ImageId) -> Vec<crate::engine::LayerSummary> {
        self.engine.state.lock().layers.get(&image).cloned().unwrap_or_default()
    }

    pub(crate) fn retouch_masks(
        &self,
        image: ImageId,
    ) -> Vec<crate::c1_retouch::RetouchMaskSummary> {
        self.engine.state.lock().retouch.get(&image).cloned().unwrap_or_default()
    }
}

/// Source acquisition: single-flight proxy decode, decode admission and cache
/// bookkeeping. The service borrows the engine; it owns no state.
pub(crate) struct SourceService<'a> {
    engine: &'a FileEngine,
}

impl<'a> SourceService<'a> {
    pub(crate) fn new(engine: &'a FileEngine) -> Self {
        Self { engine }
    }

    pub(crate) fn cached_source(&self, image: ImageId) -> Option<Arc<Source>> {
        self.engine.cached_source(image)
    }

    pub(crate) fn source(&self, image: ImageId) -> anyhow::Result<Arc<Source>> {
        self.source_with_priority(image, DecodePriority::Interactive)
    }

    /// One decode per image: callers that arrive mid-decode wait on the same
    /// flight. Different images run in parallel, bounded by their gate (heavy
    /// for RAW/EIP/COP/HEIF, light for other rasters) and the RAM-derived units
    /// the source-resolution estimate needs; thumbnails queue behind previews.
    pub(crate) fn source_with_priority(&self, image: ImageId, priority: DecodePriority) -> anyhow::Result<Arc<Source>> {
        let engine = self.engine;
        if let Some(src) = self.cached_source(image) {
            return Ok(src);
        }
        let (path, lens, width, height, heavy, format) = {
            let st = engine.state.lock();
            let entry = st
                .entries
                .iter()
                .find(|e| e.info.id == image)
                .ok_or_else(|| anyhow::anyhow!("unknown image {image:?}"))?;
            let heavy = crate::raw::is_raw(&entry.path)
                || is_eip(&entry.path)
                || is_cop(&entry.path)
                || file_kind(&entry.path) == FileKind::Heif;
            (
                entry.path.clone(),
                crate::mock::lens_profile::LensShot::from_exif(&entry.info.exif),
                entry.info.width,
                entry.info.height,
                heavy,
                proxy_decode_format(&entry.path),
            )
        };
        let key = SourceKey::Proxy(image);
        let leader = match engine.join_flight(key) {
            FlightRole::Waiter(flight) => {
                engine.note_flight_wait();
                return flight.wait().map_err(|message| anyhow::anyhow!(message));
            }
            FlightRole::Leader(leader) => leader,
        };
        let bytes = estimated_decode_bytes(width, height, format);
        let _permit = if heavy {
            let units = decode_units(bytes);
            engine.decode_gates.heavy.acquire(priority, units)
        } else {
            let units = decode_units(bytes);
            engine.decode_gates.light.acquire(priority, units)
        };
        // A decode that was already in flight may have finished while this
        // request waited for admission; take its result instead of decoding.
        if let Some(src) = self.cached_source(image) {
            return leader.finish(Ok(src)).map_err(|message| anyhow::anyhow!(message));
        }
        engine.state.lock().source_cache.misses += 1;
        // Lifecycle record for the leader's decode. Waiters returned above and
        // re-use the leader's result, so they never submit a second job. A
        // fresh token is enough: single-flight has no cancel path. Terminal
        // records are disposable; live leader flights are never pruned.
        const DECODE_JOB_CAP: usize = 64;
        if engine.decode_jobs.len() >= DECODE_JOB_CAP {
            engine.decode_jobs.prune_terminal(DECODE_JOB_CAP / 2);
        }
        let decode_jid = engine.decode_jobs.submit(JobScope::Decode, CancelToken::new());
        let _ = engine.decode_jobs.transition(decode_jid, JobState::running(0.0));
        let decoded = (|| -> Result<Arc<Source>, String> {
        // Decoding happens outside the state lock.
        // A RAW keeps libraw's 16-bit linear samples; every other file and any
        // RAW libraw cannot demosaic arrives as 8-bit sRGB.
        // An EIP hides its RAW inside a ZIP; hand every decoder the member.
        let decoded_path = if is_eip(&path) {
            materialize_eip(&engine.root, &path).map_err(|err| format!("{err:#}"))?.raw
        } else {
            path.clone()
        };
        let mut decoded_reason = None;
        let (source16, failed16) = raw16_source_with_bytes(&decoded_path);
        let (source, provenance) = match source16 {
            Some((source, provenance)) => (source, provenance),
            None => {
                let img = if is_cop(&decoded_path) {
                    match crate::jxl::decode_cop(&path) {
                        Ok(img) => img,
                        Err(err) => {
                            let reason = format!(".cop decode failed: {err:#}");
                            engine.state.lock().source_reasons.insert(image, reason.clone());
                            return Err(reason);
                        }
                    }
                } else if crate::raw::is_raw(&decoded_path) {
                    match crate::raw::decode8_fallback(&decoded_path, failed16.as_ref()) {
                        Ok(img) => img,
                        Err(err) => {
                            let reason = format!("RAW decode failed: {err:#}");
                            engine.state.lock().source_reasons.insert(image, reason.clone());
                            return Err(reason);
                        }
                    }
                } else if file_kind(&decoded_path) == FileKind::Heif {
                    match crate::heif::decode(&decoded_path) {
                        Ok(img) => img,
                        Err(err) => match image::open(&decoded_path) {
                            Ok(img) => {
                                decoded_reason = Some(format!(
                                    "HEIF via libheif failed ({err:#}); used the image crate fallback"
                                ));
                                img.to_rgba8()
                            }
                            Err(fallback) => {
                                let reason =
                                    format!("HEIF decode failed: libheif: {err:#}; image crate: {fallback}");
                                engine.state.lock().source_reasons.insert(image, reason.clone());
                                return Err(reason);
                            }
                        },
                    }
                } else {
                    image::open(&decoded_path).map_err(|err| format!("{err:#}"))?.to_rgba8()
                };
                let (w, h) = fit_long_side(img.dimensions());
                let img = if img.dimensions() != (w, h) {
                    downscale_rgba(&img, w, h)
                } else {
                    img
                };
                (
                    Source::from_rgba(w, h, img.as_raw()),
                    crate::raw::decode_fallback_provenance(&decoded_path),
                )
            }
        };
        // The decoded provenance is the only authority for the preview flag and
        // for the C1 Z6III pre-matrix: a LibRaw or embedded-preview fallback
        // must never take the camera-linear path.
        let camera_preview = provenance.pixel_space == crate::raw::PixelSpace::EmbeddedPreview;
        if camera_preview {
            decoded_reason = Some("unsupported RAW compression; showing the camera's embedded preview".into());
        }
        // BandingSuppression is keyed on ISO and body; the stage stays off when
        // either is missing rather than falling back to a guessed amount.
        let banding_identity = crate::raw::is_raw(&decoded_path)
            .then(|| crate::exif::read_exif(&decoded_path))
            .map(|exif| raw_banding_identity(&exif));
        let source = Arc::new(
            source
                .with_lens_shot(lens)
                .with_camera_preview(camera_preview)
                .with_c1_z6iii_linear(provenance.unlocks_c1_z6iii_linear())
                .with_banding_identity(
                    banding_identity.as_ref().and_then(|(iso, _)| *iso),
                    banding_identity.as_ref().and_then(|(_, model)| model.as_deref()),
                ),
        );

        {
            let mut st = engine.state.lock();
            match decoded_reason {
                Some(reason) => {
                    st.source_reasons.insert(image, reason);
                }
                None => {
                    st.source_reasons.remove(&image);
                }
            }
        }
        engine.store_cached_source(image, &path, Arc::clone(&source));
        Ok(source)
        })();
        let _ = match &decoded {
            Ok(_) => engine.decode_jobs.transition(decode_jid, JobState::Succeeded),
            Err(error) => engine.decode_jobs.transition(decode_jid, JobState::failed(error.clone())),
        };
        let result = leader.finish(decoded);
        result.map_err(|message| anyhow::anyhow!(message))
    }

}

/// Preview/thumbnail rendering service.
pub(crate) struct RenderService<'a> {
    engine: &'a FileEngine,
}

impl<'a> RenderService<'a> {
    pub(crate) fn new(engine: &'a FileEngine) -> Self {
        Self { engine }
    }

    /// One lifecycle record per thumbnail call. It records that a thumbnail
    /// started and how it ended; the decode priority, source/render path and
    /// pixels are unchanged.
    pub(crate) fn thumbnail(
        &self,
        image: ImageId,
        adjustments: &AdjustmentSet,
        max_size: u32,
    ) -> anyhow::Result<RgbaImage> {
        let engine = self.engine;
        const THUMBNAIL_JOB_CAP: usize = 64;
        if engine.thumbnail_jobs.len() >= THUMBNAIL_JOB_CAP {
            engine.thumbnail_jobs.prune_terminal(THUMBNAIL_JOB_CAP / 2);
        }
        let jid = engine.thumbnail_jobs.submit(JobScope::Thumbnail, CancelToken::new());
        let _ = engine.thumbnail_jobs.transition(jid, JobState::running(0.0));
        let result = self.thumbnail_inner(image, adjustments, max_size);
        let _ = match &result {
            Ok(_) => engine.thumbnail_jobs.transition(jid, JobState::Succeeded),
            Err(error) => engine.thumbnail_jobs.transition(jid, JobState::failed(format!("{error:#}"))),
        };
        result
    }

    fn thumbnail_inner(
        &self,
        image: ImageId,
        adjustments: &AdjustmentSet,
        max_size: u32,
    ) -> anyhow::Result<RgbaImage> {
        // Thumbnails queue behind the visible preview on the decode gate.
        let src = self.engine.source_with_priority(image, DecodePriority::Thumbnail)?;
        let out = process::render(
            &src,
            &crate::styles::resolve(adjustments),
            NormRect::FULL,
            max_size,
            false,
            None,
        );
        Ok(RgbaImage::new(out.width, out.height, out.rgba))
    }

    pub(crate) fn thumbnail_job_snapshot(&self) -> crate::job_lifecycle::RegistrySnapshot {
        self.engine.thumbnail_jobs.snapshot()
    }

    fn cached_gamut_warning(
        &self,
        key: preview_cache::PreviewKey,
        proof: &str,
        image: &RgbaImage,
    ) -> Option<crate::GamutWarning> {
        let engine = self.engine;
        let mut entries = engine.gamut.lock();
        if let Some(index) = entries
            .iter()
            .position(|(cached_key, cached_proof, _)| *cached_key == key && cached_proof == proof)
        {
            let entry = entries.remove(index);
            let warning = entry.2;
            entries.push(entry);
            return Some(warning);
        }
        let warning = crate::engine::gamut_warning(image, Some(proof))?;
        entries.push((key, proof.to_owned(), warning));
        while entries.len() > 8 {
            entries.remove(0);
        }
        Some(warning)
    }

    /// Cached preview frame; `fast` renders at half size and upsamples.
    /// One lifecycle record per preview call, shared by `render_preview` and
    /// `render_preview_fast`. This records that a render started and how it
    /// ended; it is not a scheduler. Generation handling, latest-value policy
    /// and the preview cache remain in [`Self::preview_frame_inner`] untouched.
    pub(crate) fn preview_frame(&self, request: &PreviewRequest, fast: bool) -> anyhow::Result<PreviewFrame> {
        let engine = self.engine;
        // Keep the record map bounded across a drag: terminal records are
        // disposable, live ones are never pruned.
        const PREVIEW_JOB_CAP: usize = 64;
        if engine.preview_jobs.len() >= PREVIEW_JOB_CAP {
            engine.preview_jobs.prune_terminal(PREVIEW_JOB_CAP / 2);
        }
        let jid = engine.preview_jobs.submit(JobScope::Preview, CancelToken::new());
        let _ = engine.preview_jobs.transition(jid, JobState::running(0.0));
        let result = self.preview_frame_inner(request, fast);
        let _ = match &result {
            Ok(_) => engine.preview_jobs.transition(jid, JobState::Succeeded),
            Err(error) => engine.preview_jobs.transition(jid, JobState::failed(format!("{error:#}"))),
        };
        result
    }

    fn preview_frame_inner(&self, request: &PreviewRequest, fast: bool) -> anyhow::Result<PreviewFrame> {
        let engine = self.engine;
        let start = Instant::now();
        let perf = std::env::var_os("AURORA_PERF_REPLAY").is_some();
        // Protect the currently previewed image from being evicted by concurrent
        // background thumbnail decodes so slider drags never trigger a full RAW
        // re-decode.
        {
            engine.state.lock().source_cache.active_preview = Some(request.image);
        }
        // Draft and Final must keep the same output size: a lower draft cap made
        // the preview jump when the pointer was released. Draft is cheaper
        // because `fast` renders half size and upsamples, not because it is smaller.
        let max = request.max_size;
        let adj = crate::styles::resolve(&request.adjustments);
        let (src, source_kind) = match engine.native_raw_source(request.image, Some((&adj.geometry, request.region, max))) {
            Some(native) => (native, "native_raw"),
            None => (engine.source(request.image)?, "source"),
        };
        let (rendering, intent_note) = crate::engine::request_rendering(request);
        let display_is_noop = match request.display_profile.as_deref() {
            None => true,
            Some(path) => {
                static DISPLAY_SRGB_CACHE: std::sync::OnceLock<
                    parking_lot::Mutex<Option<(std::path::PathBuf, bool)>>,
                > = std::sync::OnceLock::new();
                let mut guard = DISPLAY_SRGB_CACHE
                    .get_or_init(|| parking_lot::Mutex::new(None))
                    .lock();
                if let Some((cached_path, is_srgb)) = guard.as_ref() {
                    if cached_path.as_path() == path {
                        *is_srgb
                    } else {
                        let is_srgb = crate::icc::load_profile(path)
                            .map(|p| crate::icc::is_srgb(&p))
                            .unwrap_or(false);
                        *guard = Some((path.to_path_buf(), is_srgb));
                        is_srgb
                    }
                } else {
                    let is_srgb = crate::icc::load_profile(path)
                        .map(|p| crate::icc::is_srgb(&p))
                        .unwrap_or(false);
                    *guard = Some((path.to_path_buf(), is_srgb));
                    is_srgb
                }
            }
        };
        // Short-circuit interactive Draft drags before computing preview_cache::key
        // (8x serde_json), cloning/repairing adj.nodes, or touching preview/graph_gate
        // locks. Draft frames never populate st.preview anyway.
        let direct_bgra = !fast
            && request.quality == RenderQuality::Draft
            && display_is_noop
            && request.proof_profile.is_none();
        if direct_bgra {
            let out = process::render_preview_bgra(
                &src,
                &adj,
                request.region,
                max,
                request.bypass,
                request.mask_overlay,
                true,
            );
            let elapsed_ms = start.elapsed().as_secs_f32() * 1000.0;
            if perf {
                eprintln!(
                    "AURORA_PERF {{\"event\":\"engine_preview\",\"id\":{},\"image_id\":{},\"hit\":false,\"fast\":false,\"bgra\":{},\"width\":{},\"height\":{},\"engine_ms\":{:.3},\"source\":\"{}\",\"bypass\":{}}}",
                    request.generation,
                    request.image.0,
                    out.bgra,
                    out.width,
                    out.height,
                    elapsed_ms,
                    source_kind,
                    request.bypass
                );
            }
            return Ok(PreviewFrame {
                image: RgbaImage::new(out.width, out.height, out.rgba),
                histogram: out.histogram,
                generation: request.generation,
                elapsed_ms,
                degraded: intent_note,
                gamut_warning: None,
                bgra: out.bgra,
            });
        }
        let key = preview_cache::key(request.image, &adj, request.region, max, request.bypass, request.mask_overlay, fast)
            .with_source(src.snapshot_id());
        let profile_key = crate::engine::ProfileKey::new(key, request.proof_profile.as_deref(), request.display_profile.as_deref(), (request.proof_paper_white, request.proof_black_point), rendering);
        let graph = adj.nodes.clone().unwrap_or_default().repaired(&adj.layers);
        let mut frame_key = key;
        let mut graph_reused = false;
        let hit = {
            let mut st = engine.state.lock();
            st.preview.get(key)
        };
        let hit = match hit {
            Some(hit) => Some(hit),
            None => {
                let reuse_key = { engine.graph_gate.lock().reuse(&key, &graph, &adj.layers) };
                match reuse_key {
                    Some(reuse_key) => {
                        let hit = {
                            let mut st = engine.state.lock();
                            st.preview.get(reuse_key)
                        };
                        if hit.is_some() {
                            frame_key = reuse_key;
                            graph_reused = true;
                        }
                        hit
                    }
                    None => None,
                }
            }
        };
        {
            let mut gate = engine.graph_gate.lock();
            gate.record(key, frame_key, graph, adj.layers.clone());
        }
        if let Some(hit) = hit {
            let histogram = hit.histogram.clone();
            let source = hit.into_image();
            let (width, height) = (source.width, source.height);
            let gamut_warning = request
                .proof_profile
                .as_deref()
                .and_then(|proof| self.cached_gamut_warning(key, proof, &source));
            let (image, degraded) = engine.profiled.lock().converted(profile_key, source, request.proof_profile.as_deref(), request.display_profile.as_deref(), (request.proof_paper_white, request.proof_black_point), rendering);
            let degraded = degraded.or(intent_note);
            let elapsed_ms = start.elapsed().as_secs_f32() * 1000.0;
            if perf {
                eprintln!("AURORA_PERF {{\"event\":\"engine_preview\",\"id\":{},\"image_id\":{},\"hit\":true,\"graph_reuse\":{},\"fast\":{},\"width\":{},\"height\":{},\"engine_ms\":{:.3},\"source\":\"{}\",\"bypass\":{}}}",
                    request.generation, request.image.0, graph_reused, fast, width, height, elapsed_ms, source_kind, request.bypass);
            }
            return Ok(PreviewFrame {
                histogram,
                image,
                generation: request.generation,
                elapsed_ms,
                degraded,
                gamut_warning,
                bgra: false,
            });
        }
        let out = process::render_at(&src, &adj, request.region, max, request.bypass, request.mask_overlay, fast);
        let frame = CachedFrame {
            width: out.width,
            height: out.height,
            pixels: Arc::new(out.rgba),
            histogram: out.histogram.clone(),
        };
        engine.state.lock().preview.insert(key, frame.clone());
        let source = frame.into_image();
        let gamut_warning = request
            .proof_profile
            .as_deref()
            .and_then(|proof| self.cached_gamut_warning(key, proof, &source));
        let (image, degraded) = engine.profiled.lock().converted(profile_key, source, request.proof_profile.as_deref(), request.display_profile.as_deref(), (request.proof_paper_white, request.proof_black_point), rendering);
        let degraded = degraded.or(intent_note);
        let elapsed_ms = start.elapsed().as_secs_f32() * 1000.0;
        if perf {
            eprintln!("AURORA_PERF {{\"event\":\"engine_preview\",\"id\":{},\"image_id\":{},\"hit\":false,\"fast\":{},\"width\":{},\"height\":{},\"engine_ms\":{:.3},\"source\":\"{}\",\"bypass\":{}}}",
                request.generation, request.image.0, fast, out.width, out.height, elapsed_ms, source_kind, request.bypass);
        }
        Ok(PreviewFrame {
            image,
            histogram: out.histogram,
            generation: request.generation,
            elapsed_ms,
            degraded,
            gamut_warning,
            bgra: false,
        })
    }

}

/// AI inference: the cached model-sized input, subject/people masks and the
/// closed-eye report.
///
/// The service borrows the engine and owns no state. The process-wide mask
/// cache (`crate::aimask`), the on-disk mask store, the engine's `ai_inputs`
/// cache and the per-image closed-eye cache stay exactly where they were; the
/// only addition is one `JobScope::Inference` lifecycle record per call.
pub(crate) struct InferenceService<'a> {
    engine: &'a FileEngine,
}

impl<'a> InferenceService<'a> {
    pub(crate) fn new(engine: &'a FileEngine) -> Self {
        Self { engine }
    }

    /// Puts masks referenced by saved layers back into the in-memory cache, so
    /// a reopened folder renders the segmentation instead of the ellipse.
    pub(crate) fn preload_ai_masks(&self) {
        let keys: Vec<String> = {
            let st = self.engine.state.lock();
            st.entries
                .iter()
                .flat_map(|e| e.adjustments.layers.iter())
                .flat_map(|layer| layer.mask.shapes.iter())
                .filter_map(|shape| match shape {
                    crate::MaskShape::Ai(ai) => ai.cache_key.clone(),
                    _ => None,
                })
                .collect()
        };
        for key in keys {
            if crate::aimask::lookup(&key).is_none() {
                if let Some(plane) = crate::mask_store::load(&self.engine.root, &key) {
                    crate::aimask::store(&key, plane);
                }
            }
        }
    }

    pub(crate) fn ai_input(&self, image: ImageId) -> anyhow::Result<Arc<AiInput>> {
        let engine = self.engine;
        let source = engine.source(image)?;
        let geometry = engine.load_adjustments(image).geometry;
        let fingerprint = geometry_fingerprint(&geometry);
        if let Some(input) = engine
            .state
            .lock()
            .ai_inputs
            .get(&image)
            .filter(|input| input.source_id == source.snapshot_id() && input.geometry == fingerprint)
            .cloned()
        {
            return Ok(input);
        }
        let (rgba, width, height) = process::ai_input_rgba(&source, &geometry, 1024);
        let input = Arc::new(AiInput {
            source_id: source.snapshot_id(),
            geometry: fingerprint,
            rgba,
            width,
            height,
        });
        let mut st = engine.state.lock();
        let bytes = ai_input_bytes(&input);
        if let Some(old) = st.ai_inputs.insert(image, input.clone()) {
            st.source_cache.current_bytes = st.source_cache.current_bytes.saturating_sub(ai_input_bytes(&old));
        }
        st.source_cache.current_bytes = st.source_cache.current_bytes.saturating_add(bytes);
        st.source_cache.peak_bytes = st.source_cache.peak_bytes.max(st.source_cache.current_bytes);
        evict_sources_to_budget(&mut st, Some(image));
        Ok(input)
    }

    /// `compute_ai_mask` wrapped in one `JobScope::Inference` record. A cache
    /// hit still settles as `Succeeded`; the mask/cache/store code below is
    /// unchanged.
    pub(crate) fn compute_ai_mask(&self, image: ImageId, kind: AiMaskKind) -> anyhow::Result<String> {
        let engine = self.engine;
        const INFERENCE_JOB_CAP: usize = 64;
        if engine.inference_jobs.len() >= INFERENCE_JOB_CAP {
            engine.inference_jobs.prune_terminal(INFERENCE_JOB_CAP / 2);
        }
        let jid = engine.inference_jobs.submit(JobScope::Inference, CancelToken::new());
        let _ = engine.inference_jobs.transition(jid, JobState::running(0.0));
        let result = self.compute_ai_mask_inner(image, kind);
        let _ = match &result {
            Ok(_) => engine.inference_jobs.transition(jid, JobState::Succeeded),
            Err(error) => engine.inference_jobs.transition(jid, JobState::failed(format!("{error:#}"))),
        };
        result
    }

    fn compute_ai_mask_inner(&self, image: ImageId, kind: AiMaskKind) -> anyhow::Result<String> {
        let engine = self.engine;
        let (rel_path, variant) = {
            let st = engine.state.lock();
            let entry = st
                .entries
                .iter()
                .find(|e| e.info.id == image)
                .ok_or_else(|| anyhow::anyhow!("unknown image {image:?}"))?;
            (entry.rel_path.clone(), entry.info.variant)
        };
        let cache_key = |mask_kind: AiMaskKind| crate::mask_store::key(&rel_path, variant, mask_kind);
        let key = cache_key(kind);
        if crate::aimask::lookup(&key).is_some() {
            return Ok(key);
        }
        if let Some(plane) = crate::mask_store::load(&engine.root, &key) {
            crate::aimask::store(&key, plane);
            return Ok(key);
        }
        let start = Instant::now();
        let base_key = |group: &str| {
            format!(
                "mask-base-v1:{}#v{variant}:{group}",
                rel_path.replace('\\', "/")
            )
        };
        let plane = match kind {
            AiMaskKind::Subject
            | AiMaskKind::Person
            | AiMaskKind::Clothing
            | AiMaskKind::Background
            | AiMaskKind::Sky => {
                let subject_cache = base_key("subject");
                let subject = match crate::aimask::lookup(&subject_cache) {
                    Some(subject) => subject,
                    None => {
                        let input = self.ai_input(image)?;
                        match crate::aimask::subject_mask(&input.rgba, input.width, input.height) {
                            Ok(subject) => {
                                let subject = std::sync::Arc::new(crate::aimask::MaskPlane::new(
                                    input.width as usize,
                                    input.height as usize,
                                    subject,
                                ));
                                crate::aimask::store(&subject_cache, subject.clone());
                                subject
                            }
                            Err(err) => {
                                log::warn!("AI mask {kind:?} unavailable: {err:#}");
                                return Err(err);
                            }
                        }
                    }
                };
                for alias in [
                    AiMaskKind::Subject,
                    AiMaskKind::Person,
                    AiMaskKind::Clothing,
                ] {
                    crate::aimask::store(&cache_key(alias), subject.clone());
                }
                match kind {
                    AiMaskKind::Subject | AiMaskKind::Person | AiMaskKind::Clothing => subject,
                    AiMaskKind::Background => {
                        let mut background = subject.data.clone();
                        for value in &mut background {
                            *value = 1.0 - *value;
                        }
                        std::sync::Arc::new(crate::aimask::MaskPlane::new(
                            subject.w,
                            subject.h,
                            background,
                        ))
                    }
                    AiMaskKind::Sky => {
                        let input = self.ai_input(image)?;
                        let sky = crate::aimask::sky_mask_from_subject(
                            &input.rgba,
                            input.width as usize,
                            input.height as usize,
                            &subject.data,
                        );
                        std::sync::Arc::new(crate::aimask::MaskPlane::new(
                            input.width as usize,
                            input.height as usize,
                            sky,
                        ))
                    }
                    _ => unreachable!(),
                }
            }
            AiMaskKind::Skin | AiMaskKind::Hair => {
                let skin_cache = base_key("people-face");
                let hair_cache = base_key("people-hair");
                let skin = crate::aimask::lookup(&skin_cache);
                let hair = crate::aimask::lookup(&hair_cache);
                let (skin, hair) = match (skin, hair) {
                    (Some(skin), Some(hair)) => (skin, hair),
                    (skin, hair) => {
                        let input = self.ai_input(image)?;
                        match crate::aimask::people_masks(&input.rgba, input.width, input.height) {
                        Ok(masks) => {
                            let skin = skin.unwrap_or_else(|| {
                                let skin = std::sync::Arc::new(crate::aimask::MaskPlane::new(
                                    input.width as usize,
                                    input.height as usize,
                                    masks.face,
                                ));
                                crate::aimask::store(&skin_cache, skin.clone());
                                skin
                            });
                            let hair = hair.unwrap_or_else(|| {
                                let hair = std::sync::Arc::new(crate::aimask::MaskPlane::new(
                                    input.width as usize,
                                    input.height as usize,
                                    masks.hair,
                                ));
                                crate::aimask::store(&hair_cache, hair.clone());
                                hair
                            });
                            (skin, hair)
                        }
                        Err(err) => {
                            log::warn!("AI mask {kind:?} unavailable: {err:#}");
                            return Err(err);
                        }
                        }
                    }
                };
                crate::aimask::store(&cache_key(AiMaskKind::Skin), skin.clone());
                crate::aimask::store(&cache_key(AiMaskKind::Hair), hair.clone());
                if kind == AiMaskKind::Skin {
                    skin
                } else {
                    hair
                }
            }
            other => anyhow::bail!(
                "Capture One's six-class people parser has no {other:?} class (background, hair, \
                 body skin, face skin, clothing, held object); no model segments it"
            ),
        };
        crate::aimask::store(&key, plane.clone());
        log::info!(
            "AI mask {kind:?} for image {}: {}x{}, {:.0} ms, foreground {:.1}%",
            image.0,
            plane.w,
            plane.h,
            start.elapsed().as_secs_f32() * 1000.0,
            plane.foreground_fraction() * 100.0
        );
        if let Err(err) = crate::mask_store::save(&engine.root, &key, &plane) {
            log::warn!("AI mask {kind:?} for image {} not saved: {err:#}", image.0);
        }
        Ok(key)
    }

    /// `eye_report` wrapped in one `JobScope::Inference` record. A cached
    /// report still settles as `Succeeded`.
    pub(crate) fn eye_report(&self, image: ImageId) -> anyhow::Result<crate::EyeReport> {
        let engine = self.engine;
        const INFERENCE_JOB_CAP: usize = 64;
        if engine.inference_jobs.len() >= INFERENCE_JOB_CAP {
            engine.inference_jobs.prune_terminal(INFERENCE_JOB_CAP / 2);
        }
        let jid = engine.inference_jobs.submit(JobScope::Inference, CancelToken::new());
        let _ = engine.inference_jobs.transition(jid, JobState::running(0.0));
        let result = self.eye_report_inner(image);
        let _ = match &result {
            Ok(_) => engine.inference_jobs.transition(jid, JobState::Succeeded),
            Err(error) => engine.inference_jobs.transition(jid, JobState::failed(format!("{error:#}"))),
        };
        result
    }

    fn eye_report_inner(&self, image: ImageId) -> anyhow::Result<crate::EyeReport> {
        let engine = self.engine;
        if let Some(report) = engine.eyes.lock().get(&image) {
            return Ok(*report);
        }
        let input = self.ai_input(image)?;
        let start = Instant::now();
        let report = match crate::any_eye_closed(&input.rgba, input.width, input.height) {
            Ok(report) => report,
            Err(err) => {
                log::warn!("closed-eye check for image {} unavailable: {err:#}", image.0);
                return Err(err);
            }
        };
        log::info!(
            "closed-eye check for image {}: {} eyes, {} closed, worst p={:.2}, {:.0} ms",
            image.0,
            report.evaluated,
            report.closed,
            report.worst_confidence,
            start.elapsed().as_secs_f32() * 1000.0
        );
        engine.eyes.lock().insert(image, report);
        Ok(report)
    }

    pub(crate) fn inference_job_snapshot(&self) -> crate::job_lifecycle::RegistrySnapshot {
        self.engine.inference_jobs.snapshot()
    }
}

/// Tethered capture and live view. The service borrows the engine (for the
/// capture output folder and catalog) and owns no state; every camera call is
/// the same `tether::*` entry point as before.
pub(crate) struct CaptureService<'a> {
    engine: &'a FileEngine,
}

impl<'a> CaptureService<'a> {
    pub(crate) fn new(engine: &'a FileEngine) -> Self {
        Self { engine }
    }

    pub(crate) fn tether_status(&self) -> TetherStatus {
        tether::probe()
    }

    pub(crate) fn capture_devices(&self) -> Vec<crate::CaptureDevice> {
        tether::devices()
    }

    pub(crate) fn camera_status(&self) -> CameraStatus {
        tether::camera_status()
    }

    pub(crate) fn camera_settings(&self) -> Option<CameraSettings> {
        tether::camera_settings()
    }

    pub(crate) fn capture(&self) -> anyhow::Result<ImageInfo> {
        let frame = tether::capture_frame()?;
        let dir = self.engine.root.join("Capture");
        std::fs::create_dir_all(&dir)?;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let path = dir.join(format!("CAP_{stamp}.jpg"));
        std::fs::write(&path, &frame.jpeg)?;
        self.engine.add_file(&path)
    }

    pub(crate) fn live_view_frame(&self, width: u32) -> Option<RgbaImage> {
        match tether::preview_frame(width) {
            Ok(frame) => Some(frame),
            Err(err) => {
                log::debug!("live view: {err:#}");
                None
            }
        }
    }

    /// libgphoto2 exposes focus only through the camera's config tree, which this
    /// binding does not read yet, so a focus step is a no-op.
    pub(crate) fn live_view_focus(&self, _steps: i32) {}
}

/// Cap on retained terminal export history.
///
/// The engine keeps one `export_job_ids` entry and, on success, one
/// `export_outputs` entry per UI job id for its whole life. Live jobs are never
/// removed; once the terminal count exceeds this cap the oldest terminal ids
/// (and their outputs) are dropped. A dropped id then behaves exactly like an
/// id that was never submitted: [`ExportService::job_status`] returns
/// `JobStatus::Failed { message: "unknown job" }`.
const EXPORT_JOB_CAP: usize = 128;

/// Keep the terminal export maps and registry bounded to `cap` entries.
///
/// Lock order matches the rest of the export path: job id map -> registry
/// (cloned out one call at a time) -> output map. Live jobs and the newest
/// `cap` terminal outputs survive. Registry records are pruned even when the
/// maps are within the cap, because tests and tooling may submit directly to
/// the registry. Returns how many UI job ids were removed.
fn prune_export_history(engine: &FileEngine, cap: usize) -> usize {
    let mut terminal: Vec<(u64, u64)> = {
        let ids = engine.export_job_ids.lock();
        ids.iter()
            .filter_map(|(&job_id, &jid)| {
                // A jid missing from the registry is already pruned and can
                // only be reported as unknown, so treat it as terminal.
                let is_terminal = engine
                    .export_jobs
                    .job(jid)
                    .map(|record| record.state().is_terminal())
                    .unwrap_or(true);
                is_terminal.then_some((jid.serial, job_id))
            })
            .collect()
    };
    engine.export_jobs.prune_terminal(cap);
    if terminal.len() <= cap {
        return 0;
    }
    terminal.sort_unstable_by_key(|(serial, _)| *serial);
    let remove = terminal.len() - cap;
    let victims: Vec<u64> = terminal.into_iter().take(remove).map(|(_, job_id)| job_id).collect();
    {
        let mut ids = engine.export_job_ids.lock();
        for job_id in &victims {
            ids.remove(job_id);
        }
    }
    {
        let mut outputs = engine.export_outputs.lock();
        for job_id in &victims {
            outputs.remove(job_id);
        }
    }
    remove
}

/// Export scheduling, job status and cancellation.
///
/// The engine keeps the job registry, publish gate, id map and output map; the
/// worker gets its own clones. The service owns no state and takes no lock the
/// worker needs beyond the documented `gate -> maps -> registry` order.
pub(crate) struct ExportService<'a> {
    engine: &'a FileEngine,
}

impl<'a> ExportService<'a> {
    pub(crate) fn new(engine: &'a FileEngine) -> Self {
        Self { engine }
    }

    pub(crate) fn start_export(&self, job: ExportJob) -> anyhow::Result<()> {
        let engine = self.engine;
        job.recipe.format.ensure_supported()?;
        job.recipe.format.supports_bit_depth(job.recipe.bit_depth)?;
        let state = engine.state.clone();
        let publish_gate = engine.publish_gate.clone();
        let job_id = job.job_id;
        let render = {
            let st = state.lock();
            let entry = st.entries.iter().find(|entry| entry.info.id == job.image)
                .ok_or_else(|| anyhow::anyhow!("export: unknown image {}", job.image.0))?;
            #[cfg(all(windows, feature = "z6iii-raw-export"))]
            let exact = (entry.path.clone(), entry.adjustments.clone());
            (entry.info.clone(), #[cfg(all(windows, feature = "z6iii-raw-export"))] exact)
        };
        let destination = job.recipe.destination.clone().unwrap_or_else(|| std::env::temp_dir().join("portrait-output"));
        // Canonicalize and prove the destination is a folder before a single
        // pixel is rendered, so a bad recipe fails at `start_export` rather
        // than after minutes of work.
        let dir = crate::output::resolve_output_dir(&destination)?;
        // A crashed export leaves a staging folder behind; reclaim it here.
        crate::output::sweep_stale_staging(&dir);
        let info = render.0;
        #[cfg(all(windows, feature = "z6iii-raw-export"))]
        let exact = match crate::z6iii::desktop::Export::prepare(&render.1.0, &render.1.1, &job.recipe) {
            Ok(exact) => exact,
            Err(error) => {
                log::warn!("Z6III export unavailable; using standard engine: {error:#}");
                None
            }
        };
        #[cfg(all(windows, feature = "z6iii-raw-export"))]
        let use_standard = exact.is_none();
        #[cfg(not(all(windows, feature = "z6iii-raw-export")))]
        let use_standard = true;
        let (standard, export_dims) = if use_standard {
            let adjustments = crate::styles::resolve(&engine.load_adjustments(job.image)).into_owned();
            // Native RAW is always preferred: it is the real full-resolution
            // decode. Without it the preview source is capped at SOURCE_LONG_SIDE.
            let native = engine.native_raw_source(job.image, None);
            let frame_source_size = native.as_ref().map_or((info.width, info.height), |native| native.size());
            let (width, height) =
                crate::mock::geometry::FrameMap::new(frame_source_size, &adjustments.geometry).frame_size();
            let edited_size = (width.round().max(1.0) as u32, height.round().max(1.0) as u32);
            let (width, height) = crate::recipe::output_size(&job.recipe, edited_size);
            let source = match native {
                Some(native) => StandardExportSource::Ready(native),
                // Above the proxy's long side the worker decodes the original at
                // full resolution after scheduler admission, before the render;
                // that one-shot decode is not part of the admission budget yet
                // (see the helper's note). The calling thread only reads state.
                None => engine.export_source_plan(job.image, width.max(height))?,
            };
            let mut recipe = job.recipe.clone();
            recipe.scale = crate::ScaleMode::Fixed100;
            (Some((source, adjustments, width.max(height), recipe)), (width, height))
        } else {
            (None, (info.width, info.height))
        };
        let estimated_bytes = estimated_export_bytes_for_recipe(export_dims.0, export_dims.1, &job.recipe);
        // Register the job before spawning so `cancel_job` can reach it without
        // racing the worker. This is where the old gate inserted
        // `JobStatus::Queued`, so a bad recipe or destination still returns
        // `Err` before any job exists.
        let token = CancelToken::new();
        let jid = engine.export_jobs.submit(JobScope::Export, token.clone());
        engine.export_job_ids.lock().insert(job_id, jid);
        // Bound the process-lifetime history. The job just registered is live,
        // so pruning cannot drop it; only old terminal ids and outputs go.
        prune_export_history(engine, EXPORT_JOB_CAP);
        let registry = engine.export_jobs.clone();
        let job_ids = Arc::clone(&engine.export_job_ids);
        let outputs = Arc::clone(&engine.export_outputs);
        std::thread::spawn(move || {
            let is_cancelled = || token.is_cancelled();
            let Some(_export_permit) =
                export_scheduler().acquire_cancellable(estimated_bytes, || token.is_cancelled())
            else {
                // Cancelled before admission. No output was staged, but the
                // registry must still settle in a terminal state.
                let _ = registry.transition(jid, JobState::Cancelled);
                return;
            };
            if is_cancelled() {
                let _ = registry.transition(jid, JobState::Cancelled);
                return;
            }
            // Queued -> Running. A rejection here means a cancel moved the job
            // terminal first; the render still polls the token at its own safe
            // boundaries.
            let _ = registry.transition(jid, JobState::running(0.05));
            let result = (|| {
                // Every format renders into a staging folder on the target's
                // own volume. The finished file is synced and only then renamed
                // to its final name, so cancel or failure never truncates (or
                // touches) an existing output.
                let staged = crate::output::StagedExport::new(&dir, job_id)?;
                let written = (|| {
                    #[cfg(all(windows, feature = "z6iii-raw-export"))]
                    if let Some(exact) = exact {
                        return exact.write(staged.dir(), &info, &job.recipe, || is_cancelled());
                    }
                    let (source, adjustments, max_size, recipe) = standard.expect("standard export source");
                    // The full-resolution fallback is decoded here rather than in
                    // `start_export`, after the scheduler permit and before any
                    // render buffer is allocated. A decode failure reaches the
                    // normal job-failure path below, so nothing is published.
                    let src = match source {
                        StandardExportSource::Ready(source) => source,
                        StandardExportSource::FullRes(request) => Arc::new(request.decode()?),
                    };
                    // A rejected progress transition (the job went terminal
                    // first) is intentionally ignored; progress is cosmetic and
                    // must never overwrite a terminal state.
                    if recipe.bit_depth == 16 {
                        let out = process::render16(
                            &src,
                            &adjustments,
                            NormRect::FULL,
                            max_size,
                            false,
                        );
                        crate::recipe::write_export16(
                            staged.dir(),
                            &info,
                            &recipe,
                            info.variant as usize,
                            out.rgba,
                            out.width,
                            out.height,
                            |p| {
                                if !is_cancelled() {
                                    let _ = registry.transition(jid, JobState::running(0.1 + 0.9 * p));
                                }
                            },
                        )
                    } else {
                        let out = process::render(
                            &src,
                            &adjustments,
                            NormRect::FULL,
                            max_size,
                            false,
                            None,
                        );
                        crate::recipe::write_export(
                            staged.dir(),
                            &info,
                            &recipe,
                            info.variant as usize,
                            out.rgba,
                            out.width,
                            out.height,
                            |p| {
                                if !is_cancelled() {
                                    let _ = registry.transition(jid, JobState::running(0.1 + 0.9 * p));
                                }
                            },
                        )
                    }
                })();
                let written = written?;
                // The gate makes the final cancel check and the publish one
                // step: `cancel_job` takes the same gate, so a cancel that
                // returns `true` happened before this publish and the job is
                // dropped below; a publish that got here first records
                // `Succeeded` under the gate and `cancel_job` then reports
                // `false`. Lock order: gate -> maps -> registry. The registry
                // holds no lock of its own across the publish's file I/O
                // because each call below takes and releases it.
                let _gate = publish_gate.lock();
                if is_cancelled() {
                    anyhow::bail!("cancelled");
                }
                // A reused UI job id may already name a newer export; a stale
                // worker must not publish into, or record output for, it.
                if job_ids.lock().get(&job_id).copied() != Some(jid) {
                    anyhow::bail!("export superseded by a newer job");
                }
                let output = staged.publish(&written)?;
                outputs.lock().insert(job_id, output.clone());
                if registry.transition(jid, JobState::Succeeded).is_err() {
                    // The registry went terminal first; remove the path so a
                    // late success is never exposed as `Done`.
                    outputs.lock().remove(&job_id);
                    anyhow::bail!("export finished after the job was cancelled");
                }
                Ok(output)
            })();
            match result {
                Ok(output) => {
                    // File name only: an export log must not reveal the folder
                    // a photograph lives in.
                    log::info!("export job {job_id} wrote {}", crate::output::log_name(&output));
                }
                Err(err) => {
                    let terminal = if is_cancelled() {
                        JobState::Cancelled
                    } else {
                        JobState::failed(err.to_string())
                    };
                    // Ignore a rejected transition: a cancel may already have
                    // made the job terminal, and that state must win.
                    let _ = registry.transition(jid, terminal);
                    log::warn!(
                        "export job {job_id} for {} failed: {}",
                        info.file_name,
                        crate::output::sanitize_log_text(&err.to_string())
                    );
                }
            }
        });
        Ok(())
    }

    /// Status for one UI export job id.
    ///
    /// Only a bounded history is retained: once more than [`EXPORT_JOB_CAP`]
    /// terminal jobs exist, the oldest ids are pruned and report
    /// `Failed { message: "unknown job" }`, the same answer this method has
    /// always given for an id that was never submitted.
    pub(crate) fn job_status(&self, job_id: u64) -> JobStatus {
        let engine = self.engine;
        // Lock order: job id map -> registry (cloned out, then released) ->
        // output map. The registry lock is never held across a map lock.
        let Some(jid) = engine.export_job_ids.lock().get(&job_id).copied() else {
            return JobStatus::Failed { message: "unknown job".into() };
        };
        let Some(record) = engine.export_jobs.job(jid) else {
            return JobStatus::Failed { message: "unknown job".into() };
        };
        match record.state() {
            JobState::Queued => JobStatus::Queued,
            JobState::Running { progress } => JobStatus::Running { progress: *progress },
            // The old gate reported a cancellation immediately, before the
            // worker observed it. Keep that contract.
            JobState::Cancelling | JobState::Cancelled => {
                JobStatus::Failed { message: "cancelled".into() }
            }
            JobState::Failed { error } => JobStatus::Failed { message: error.clone() },
            JobState::Succeeded => match engine.export_outputs.lock().get(&job_id) {
                Some(output) => JobStatus::Done { output: output.clone() },
                None => JobStatus::Failed {
                    message: "export finished without an output path".into(),
                },
            },
        }
    }

    pub(crate) fn cancel_job(&self, job_id: u64) -> bool {
        let engine = self.engine;
        // Same gate as the export's publish point: either the cancel lands
        // before the publish (and the job settles terminal) or the publish
        // lands first and the job is already terminal here. Lock order:
        // gate -> job id map -> registry.
        let _gate = engine.publish_gate.lock();
        let Some(jid) = engine.export_job_ids.lock().get(&job_id).copied() else {
            return false;
        };
        let requested = engine.export_jobs.cancel(jid).unwrap_or(false);
        if requested {
            export_scheduler().notify_all();
        }
        requested
    }

    pub(crate) fn export_job_snapshot(&self) -> crate::job_lifecycle::RegistrySnapshot {
        self.engine.export_jobs.snapshot()
    }

    /// Test-only entry point to the cap logic, so a test can drive history
    /// without running hundreds of real exports.
    #[cfg(test)]
    pub(crate) fn prune_export_history_for_test(&self, cap: usize) -> usize {
        prune_export_history(self.engine, cap)
    }
}

/// HDR and panorama merge, both the synchronous one-shot path and the async
/// registry hand-off. The service borrows the engine; `merge_registry` and the
/// background worker stay on `FileEngine`/`merge_async`.
pub(crate) struct MergeService<'a> {
    engine: &'a FileEngine,
}

impl<'a> MergeService<'a> {
    pub(crate) fn new(engine: &'a FileEngine) -> Self {
        Self { engine }
    }

    pub(crate) fn merge_hdr_frames(&self, ids: &[ImageId]) -> anyhow::Result<ImageInfo> {
        self.merge_hdr_impl(ids, None)
    }

    pub(crate) fn merge_hdr_with_options(
        &self,
        ids: &[ImageId],
        options: &HdrOptions,
    ) -> anyhow::Result<ImageInfo> {
        self.merge_hdr_impl(ids, Some(options))
    }

    pub(crate) fn start_merge(
        &self,
        kind: crate::merge_async::MergeKind,
        images: &[ImageId],
        options: &crate::MergeRequest,
    ) -> anyhow::Result<crate::MergeJobId> {
        let engine = self.engine;
        anyhow::ensure!(!images.is_empty(), "merge needs at least one image");
        let mut sources = Vec::with_capacity(images.len());
        let mut stops = Vec::with_capacity(images.len());
        let first_path;
        {
            let st = engine.state.lock();
            let mut exif = Vec::with_capacity(images.len());
            let mut paths = Vec::with_capacity(images.len());
            for id in images {
                let entry = st
                    .entries
                    .iter()
                    .find(|entry| entry.info.id == *id)
                    .ok_or_else(|| anyhow::anyhow!("merge: unknown image {id:?}"))?;
                paths.push(entry.path.clone());
                exif.push(entry.info.exif.clone());
            }
            first_path = paths.first().cloned().ok_or_else(|| anyhow::anyhow!("merge: no first frame"))?;
            if let Some(first) = exif.first() {
                stops = exif.iter().map(|frame| crate::hdr::relative_stops(first, frame)).collect();
            }
        }
        for id in images {
            sources.push(engine.source(*id)?);
        }
        let mut request = options.clone();
        request.images = images.to_vec();
        Ok(engine.merge_registry.start(kind, sources, stops, first_path, request))
    }

    fn merge_hdr_impl(&self, ids: &[ImageId], options: Option<&HdrOptions>) -> anyhow::Result<ImageInfo> {
        let engine = self.engine;
        if let Some(options) = options {
            Self::warn_hdr_options(options);
        }
        let mut paths = Vec::with_capacity(ids.len());
        let mut exposures = Vec::with_capacity(ids.len());
        {
            let st = engine.state.lock();
            for id in ids {
                let entry = st
                    .entries
                    .iter()
                    .find(|e| e.info.id == *id)
                    .ok_or_else(|| anyhow::anyhow!("HDR merge: unknown image {id:?}"))?;
                paths.push(entry.path.clone());
                exposures.push(entry.info.exif.clone());
            }
        }
        let Some(first_path) = paths.first() else {
            anyhow::bail!("HDR merge needs at least one frame");
        };
        let stops: Vec<f32> = exposures
            .iter()
            .map(|exif| crate::hdr::relative_stops(&exposures[0], exif))
            .collect();
        let sources: Vec<Arc<Source>> = ids.iter().map(|id| engine.source(*id)).collect::<anyhow::Result<_>>()?;
        let frames: Vec<&Source> = sources.iter().map(|s| s.as_ref()).collect();
        let defaults = HdrOptions::default();
        let fused = crate::hdr::fuse_with_options(&frames, &stops, options.unwrap_or(&defaults))?;
        let wide = crate::hdr::to_srgb16(&fused);

        let dir = first_path.parent().unwrap_or_else(|| Path::new("."));
        let stem = first_path.file_stem().and_then(|s| s.to_str()).unwrap_or("merge");
        let mut path = dir.join(format!("{stem}_HDR.tif"));
        let mut counter = 2;
        while path.exists() {
            if counter > 999 {
                anyhow::bail!("HDR merge: no free output name beside {}", first_path.display());
            }
            path = dir.join(format!("{stem}_HDR-{counter}.tif"));
            counter += 1;
        }
        {
            use image::ImageEncoder as _;
            let bytes: Vec<u8> = wide.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let mut out = std::io::Cursor::new(Vec::new());
            let encoder = image::codecs::tiff::TiffEncoder::new(&mut out);
            encoder.write_image(&bytes, fused.width, fused.height, image::ExtendedColorType::Rgb16)?;
            std::fs::write(&path, out.into_inner())?;
        }

        let info = engine.add_file(&path)?;
        let rel = {
            let st = engine.state.lock();
            st.entries
                .iter()
                .find(|e| e.info.id == info.id)
                .map(|e| e.rel_path.clone())
                .ok_or_else(|| anyhow::anyhow!("HDR merge: output missing from the catalog"))?
        };
        engine.persist(&rel, &AdjustmentSet::default(), &info);
        Ok(info)
    }

    pub(crate) fn merge_panorama_frames(&self, ids: &[ImageId]) -> anyhow::Result<ImageInfo> {
        self.merge_panorama_impl(ids, None)
    }

    pub(crate) fn merge_panorama_with_options(
        &self,
        ids: &[ImageId],
        options: &PanoramaOptions,
    ) -> anyhow::Result<ImageInfo> {
        self.merge_panorama_impl(ids, Some(options))
    }

    fn merge_panorama_impl(
        &self,
        ids: &[ImageId],
        options: Option<&PanoramaOptions>,
    ) -> anyhow::Result<ImageInfo> {
        let engine = self.engine;
        if let Some(options) = options {
            Self::warn_panorama_options(options);
        }
        let first_path = {
            let st = engine.state.lock();
            let id = ids.first().ok_or_else(|| anyhow::anyhow!("Panorama merge needs at least one frame"))?;
            st.entries
                .iter()
                .find(|e| e.info.id == *id)
                .map(|e| e.path.clone())
                .ok_or_else(|| anyhow::anyhow!("Panorama merge: unknown image {id:?}"))?
        };
        let sources: Vec<Arc<Source>> = ids.iter().map(|id| engine.source(*id)).collect::<anyhow::Result<_>>()?;
        let frames: Vec<&Source> = sources.iter().map(|s| s.as_ref()).collect();
        let defaults = PanoramaOptions::default();
        let fused = crate::pano::stitch_with_options(&frames, options.unwrap_or(&defaults))?;
        let bytes = crate::pano::to_srgb8(&fused);
        self.write_merge(&first_path, "Pano", fused.width, fused.height, &bytes, image::ExtendedColorType::Rgb8)
    }

    fn warn_hdr_options(options: &HdrOptions) {
        if options.align {
            log::warn!("HDR merge: align=true is not applied; frames are never aligned");
        }
        if options.auto_tone {
            log::warn!("HDR merge: auto_tone=true is not applied");
        }
    }

    fn warn_panorama_options(options: &PanoramaOptions) {
        if options.boundary_warp != 0.0 {
            log::warn!(
                "Panorama merge: boundary_warp={} is not applied",
                options.boundary_warp
            );
        }
        if options.auto_crop {
            log::warn!(
                "Panorama merge: auto_crop=true is not applied; the stitched canvas keeps its edges"
            );
        } else {
            log::debug!("Panorama merge: auto_crop=false, keeping the stitched canvas edges");
        }
    }

    /// Writes a merged TIFF beside the first frame, names it `{stem}_{tag}.tif`
    /// without clobbering an existing file, then catalogs and persists it.
    fn write_merge(
        &self,
        first_path: &Path,
        tag: &str,
        width: u32,
        height: u32,
        bytes: &[u8],
        color: image::ExtendedColorType,
    ) -> anyhow::Result<ImageInfo> {
        let engine = self.engine;
        let dir = first_path.parent().unwrap_or_else(|| Path::new("."));
        let stem = first_path.file_stem().and_then(|s| s.to_str()).unwrap_or("merge");
        let mut path = dir.join(format!("{stem}_{tag}.tif"));
        let mut counter = 2;
        while path.exists() {
            if counter > 999 {
                anyhow::bail!("{tag} merge: no free output name beside {}", first_path.display());
            }
            path = dir.join(format!("{stem}_{tag}-{counter}.tif"));
            counter += 1;
        }
        {
            use image::ImageEncoder as _;
            let mut out = std::io::Cursor::new(Vec::new());
            let encoder = image::codecs::tiff::TiffEncoder::new(&mut out);
            encoder.write_image(bytes, width, height, color)?;
            std::fs::write(&path, out.into_inner())?;
        }

        let info = engine.add_file(&path)?;
        let rel = {
            let st = engine.state.lock();
            st.entries
                .iter()
                .find(|e| e.info.id == info.id)
                .map(|e| e.rel_path.clone())
                .ok_or_else(|| anyhow::anyhow!("{tag} merge: output missing from the catalog"))?
        };
        engine.persist(&rel, &AdjustmentSet::default(), &info);
        Ok(info)
    }
}

/// Outstanding documents the queued save path may hold (queued plus the one
/// being written; see [`SaveQueue::new`]). Each slot retains one serialized
/// [`AdjustmentSet`], which for a heavily layered edit is tens of kilobytes, so
/// 64 slots bound the queue near a few MiB while still absorbing a large
/// multi-select edit. `submit` never blocks: when this bound is reached a new
/// path is reported `Full` and the caller can flush and retry. A newer revision
/// for an already-outstanding path is always accepted (soft bound by one).
const SAVE_QUEUE_CAPACITY: usize = 64;

/// Orders batch saves per relative path: a write queued before a newer write
/// must not land after it. The mutex is held only for the claim check, never
/// across disk I/O.
#[derive(Default)]
pub(crate) struct SaveOrder {
    next_seq: u64,
    /// Newest sequence already accepted (queued or written) per path.
    newest: HashMap<String, u64>,
}

impl SaveOrder {
    /// Allocates the sequence number for one batch call. Starts at 1 so the
    /// zero value is a usable "stale" marker in tests.
    fn begin(&mut self) -> u64 {
        self.next_seq += 1;
        self.next_seq
    }

    /// Claims `path` for `seq`. Returns false when a newer sequence already
    /// owns the path; the caller must then skip the write.
    fn claim(&mut self, path: &str, seq: u64) -> bool {
        match self.newest.get(path) {
            Some(&newest) if seq < newest => false,
            _ => {
                self.newest.insert(path.to_string(), seq);
                true
            }
        }
    }
}

/// Catalog entry for one image at save time. Shared by the synchronous batch
/// path and the queued writer so both record identical metadata.
fn entry_for_info(rel_path: &str, info: &ImageInfo) -> Entry {
    Entry {
        rel_path: rel_path.to_string(),
        rating: info.rating,
        color_tag: ColorTag::ALL.iter().position(|c| *c == info.color_tag).unwrap_or(0) as i8,
        keywords: info.keywords.clone(),
        variant: info.variant,
        updated_at: now_secs(),
        trashed: info.trashed,
    }
}

/// Bound on the terminal [`ImageInfo`] entries retained for the save writer.
/// Paths still queued or in flight are always kept; older terminal entries are
/// dropped once the count exceeds this.
pub(crate) const SAVE_INFO_CAP: usize = 256;

/// The [`ImageInfo`] captured when a document was queued, shared with the
/// single writer.
///
/// The writer needs the metadata only while writing one path, so entries are
/// bounded: [`Self::prune`] keeps every path the queue still has queued or in
/// flight plus the newest [`SAVE_INFO_CAP`] terminal paths. Evicting a terminal
/// entry is safe because a retry submits again and re-inserts the freshest
/// [`ImageInfo`] before its new revision is written.
#[derive(Default)]
pub(crate) struct SaveInfos {
    inner: Mutex<SaveInfosInner>,
}

#[derive(Default)]
struct SaveInfosInner {
    /// `rel_path -> (insertion sequence, info)`. The sequence orders eviction;
    /// a submit always re-inserts, so the map holds the newest info per path.
    entries: HashMap<String, (u64, ImageInfo)>,
    next_seq: u64,
}

impl SaveInfos {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn insert(&self, rel_path: String, info: ImageInfo) {
        let mut inner = self.inner.lock();
        inner.next_seq = inner.next_seq.saturating_add(1);
        let seq = inner.next_seq;
        inner.entries.insert(rel_path, (seq, info));
    }

    pub(crate) fn get(&self, rel_path: &str) -> Option<ImageInfo> {
        self.inner.lock().entries.get(rel_path).map(|(_, info)| info.clone())
    }

    /// Drop the oldest terminal entries beyond `cap`, never touching a path the
    /// queue still has queued or in flight. Returns how many entries were
    /// removed.
    ///
    /// The queue snapshot is taken before the map lock so this never holds
    /// [`SaveInfos`] while the writer needs it. A terminal entry that a caller
    /// later retries is re-inserted by that retry's submit, so a write that
    /// fails and is resubmitted still finds its metadata.
    pub(crate) fn prune(&self, cap: usize, queue: Option<&SaveQueue>) -> usize {
        // Cheap common case: the total is within the cap, so no queue snapshot
        // is needed. Live entries are a subset of the total.
        if self.inner.lock().entries.len() <= cap {
            return 0;
        }
        let live: HashSet<String> = queue
            .map(|queue| {
                queue
                    .documents()
                    .into_iter()
                    .filter(|status| matches!(status.state, PathState::Queued | PathState::InFlight))
                    .map(|status| status.rel_path)
                    .collect()
            })
            .unwrap_or_default();
        let mut inner = self.inner.lock();
        let mut terminal: Vec<(u64, String)> = inner
            .entries
            .iter()
            .filter(|(path, _)| !live.contains(*path))
            .map(|(path, (seq, _))| (*seq, path.clone()))
            .collect();
        if terminal.len() <= cap {
            return 0;
        }
        terminal.sort_unstable_by_key(|(seq, _)| *seq);
        let remove = terminal.len() - cap;
        for (_, path) in terminal.into_iter().take(remove) {
            inner.entries.remove(&path);
        }
        remove
    }

    /// Number of retained entries; used by the bound tests.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.inner.lock().entries.len()
    }
}

/// The [`SaveWriter`] behind `FileEngine::save_queue`.
///
/// It owns a [`Store`] clone and the engine state `Arc`, but never a
/// `FileEngine` reference, so it runs on the queue's writer thread with no
/// engine or catalog lock held across file I/O:
///
/// * [`SaveWriter::write`] writes one adjustment file first. Only after that
///   succeeds does it take the state lock briefly to upsert the catalog entry
///   from the [`ImageInfo`] recorded at queue time. A failed write leaves the
///   catalog untouched and returns the error to the queue.
/// * [`SaveWriter::commit_index`] clones the catalog under the state lock,
///   drops the lock, then writes the index.
struct StoreSaveWriter {
    store: Store,
    state: Arc<Mutex<State>>,
    infos: Arc<SaveInfos>,
}

impl SaveWriter for StoreSaveWriter {
    fn write(&self, rel_path: &str, adjustments: &AdjustmentSet) -> anyhow::Result<()> {
        self.store.save_adjustments(rel_path, adjustments)?;
        // The map is keyed by path, so an in-flight older revision may upsert
        // the metadata of a newer queued submit for the same path. That is
        // harmless: the newer revision owns those fields anyway, and the index
        // is committed after the item is written.
        if let Some(info) = self.infos.get(rel_path) {
            self.state.lock().catalog.upsert(entry_for_info(rel_path, &info));
        }
        Ok(())
    }

    fn commit_index(&self) -> anyhow::Result<()> {
        let catalog = self.state.lock().catalog.clone();
        self.store.save_index(&catalog)
    }
}

/// Builds the queued persistence path for a library with a writable store.
///
/// Returns `None` when there is no store. If the OS refuses to start the writer
/// thread the queue is returned already closed; submissions then report
/// [`SubmitOutcome::Closed`] instead of the engine panicking.
pub(crate) fn start_save_queue(
    store: Option<&Store>,
    state: &Arc<Mutex<State>>,
    infos: &Arc<SaveInfos>,
) -> Option<Arc<SaveQueue>> {
    let store = store?;
    let writer = Arc::new(StoreSaveWriter {
        store: store.clone(),
        state: Arc::clone(state),
        infos: Arc::clone(infos),
    });
    let sink: Arc<dyn SaveWriter> = writer;
    Some(Arc::new(SaveQueue::new(sink, SAVE_QUEUE_CAPACITY)))
}

/// Adjustment persistence: synchronous batch writes and the bounded
/// single-writer queue, with per-path ordering, revisions and flush/status.
///
/// The service borrows the engine and owns no state; the store, queue, ordering
/// mutex and revision counters stay on `FileEngine`. In-memory edits are always
/// applied, and the catalog is only upserted after a successful file write.
pub(crate) struct PersistenceService<'a> {
    engine: &'a FileEngine,
}

impl<'a> PersistenceService<'a> {
    pub(crate) fn new(engine: &'a FileEngine) -> Self {
        Self { engine }
    }

    pub(crate) fn persist(&self, rel_path: &str, adjustments: &AdjustmentSet, info: &ImageInfo) -> SaveReport {
        self.persist_batch(&[(rel_path.to_string(), adjustments.clone(), info.clone())])
    }

    pub(crate) fn persist_batch(&self, items: &[(String, AdjustmentSet, ImageInfo)]) -> SaveReport {
        let engine = self.engine;
        let seq = engine.save_order.lock().begin();
        self.persist_batch_at(seq, items)
    }

    pub(crate) fn persist_batch_at(&self, seq: u64, items: &[(String, AdjustmentSet, ImageInfo)]) -> SaveReport {
        let engine = self.engine;
        let Some(store) = &engine.store else {
            let documents = items
                .iter()
                .map(|(rel_path, _, info)| {
                    DocumentSaveResult::failed(info.id, rel_path, "no writable store for this library")
                })
                .collect();
            return SaveReport::new(documents, None);
        };
        let mut documents = Vec::with_capacity(items.len());
        let mut saved = Vec::new();
        for (rel_path, adjustments, info) in items {
            if !engine.save_order.lock().claim(rel_path, seq) {
                documents.push(DocumentSaveResult::superseded(info.id, rel_path));
                continue;
            }
            match store.save_adjustments(rel_path, adjustments) {
                Ok(()) => {
                    documents.push(DocumentSaveResult::saved(info.id, rel_path));
                    saved.push((rel_path, info));
                }
                Err(err) => {
                    let error = err.to_string();
                    log::warn!("save adjustments {rel_path}: {error}");
                    documents.push(DocumentSaveResult::failed(info.id, rel_path, error));
                }
            }
        }
        let mut index_error = None;
        if !saved.is_empty() {
            let mut st = engine.state.lock();
            for (rel_path, info) in saved {
                st.catalog.upsert(entry_for_info(rel_path, info));
            }
            let catalog = st.catalog.clone();
            drop(st);
            if let Err(err) = store.save_index(&catalog) {
                log::warn!("save catalog: {err}");
                index_error = Some(err.to_string());
            }
        }
        SaveReport::new(documents, index_error)
    }

    pub(crate) fn save_adjustments_batch_report(&self, items: &[(ImageId, AdjustmentSet)]) -> SaveReport {
        if items.is_empty() {
            return SaveReport::default();
        }
        let engine = self.engine;
        let seq = engine.save_order.lock().begin();
        self.save_adjustments_batch_report_at(seq, items)
    }

    pub(crate) fn save_adjustments_batch_report_at(
        &self,
        seq: u64,
        items: &[(ImageId, AdjustmentSet)],
    ) -> SaveReport {
        let engine = self.engine;
        // One sequence per public save call, used for *both* the in-memory
        // edit and its disk write. Assigning the disk sequence later would let
        // an older edit that reaches `persist` late overwrite newer bytes.
        let (records, unknown, superseded) = {
            let mut st = engine.state.lock();
            let mut records = Vec::with_capacity(items.len());
            let mut unknown = Vec::new();
            let mut superseded = Vec::new();
            for (image, adjustments) in items {
                let Some(entry) =
                    st.entries.iter_mut().find(|entry| entry.info.id == *image)
                else {
                    unknown.push(*image);
                    continue;
                };
                let rel_path = entry.rel_path.clone();
                // Claim before touching memory: an older call must not restore
                // stale adjustments after a newer call already applied its own.
                if !engine.save_order.lock().claim(&rel_path, seq) {
                    superseded.push(DocumentSaveResult::superseded(*image, rel_path));
                    continue;
                }
                entry.adjustments = adjustments.clone();
                entry.info.has_adjustments = !adjustments.is_default();
                records.push((rel_path, adjustments.clone(), entry.info.clone()));
            }
            (records, unknown, superseded)
        };
        // `persist_batch_at` re-claims each path; the equal sequence passes, so
        // the memory and disk halves cannot diverge.
        let mut report = if records.is_empty() {
            SaveReport::default()
        } else {
            self.persist_batch_at(seq, &records)
        };
        report.documents.extend(superseded);
        for image in unknown {
            report
                .documents
                .push(DocumentSaveResult::failed(image, "", "unknown image"));
        }
        report
    }

    pub(crate) fn save_adjustments_queued(&self, items: &[(ImageId, AdjustmentSet)]) -> QueuedSaveSummary {
        let engine = self.engine;
        let mut summary = QueuedSaveSummary::default();
        if items.is_empty() {
            return summary;
        }

        // One short state lock: apply every edit and capture the path, info and
        // document the writer will need. Nothing here touches the queue yet.
        let mut queued = Vec::with_capacity(items.len());
        {
            let mut st = engine.state.lock();
            for (image, adjustments) in items {
                let Some(entry) = st.entries.iter_mut().find(|entry| entry.info.id == *image) else {
                    summary.documents.push(QueuedDocumentResult {
                        image: *image,
                        path: String::new(),
                        outcome: QueuedSaveOutcome::Failed { error: "unknown image".to_string() },
                    });
                    continue;
                };
                entry.adjustments = adjustments.clone();
                entry.info.has_adjustments = !adjustments.is_default();
                queued.push((*image, entry.rel_path.clone(), entry.info.clone(), adjustments.clone()));
            }
        }

        let Some(queue) = &engine.save_queue else {
            for (image, rel_path, _, _) in queued {
                summary.documents.push(QueuedDocumentResult {
                    image,
                    path: rel_path,
                    outcome: QueuedSaveOutcome::Failed {
                        error: "no writable store for this library".to_string(),
                    },
                });
            }
            return summary;
        };

        for (image, rel_path, info, adjustments) in queued {
            // Record before submitting: the writer may start before `submit`
            // returns.
            engine.save_infos.insert(rel_path.clone(), info);
            let revision = self.next_save_revision(&rel_path);
            let outcome = match queue.submit(&rel_path, adjustments, revision) {
                SubmitOutcome::Accepted { revision } => QueuedSaveOutcome::Accepted { revision },
                SubmitOutcome::Full { capacity } => QueuedSaveOutcome::Full { capacity },
                SubmitOutcome::Closed => QueuedSaveOutcome::Closed,
                SubmitOutcome::Superseded { newest } => QueuedSaveOutcome::Superseded { newest },
            };
            summary.documents.push(QueuedDocumentResult { image, path: rel_path, outcome });
        }
        // Keep the metadata map bounded: queued/in-flight paths survive, older
        // terminal paths are dropped. A dropped path's next submit re-inserts
        // its info before the writer can need it, so retries stay correct.
        engine.save_infos.prune(SAVE_INFO_CAP, engine.save_queue.as_deref());
        summary
    }

    pub(crate) fn next_save_revision(&self, rel_path: &str) -> DocumentRevision {
        let engine = self.engine;
        let mut revisions = engine.save_revisions.lock();
        let next = revisions.get(rel_path).copied().unwrap_or_default().next();
        revisions.insert(rel_path.to_string(), next);
        next
    }

    pub(crate) fn flush_saves(&self, timeout: Duration) -> Option<FlushReport> {
        self.engine.save_queue.as_ref().map(|queue| queue.flush(timeout))
    }

    pub(crate) fn save_queue_status(&self) -> Option<QueueStatus> {
        self.engine.save_queue.as_ref().map(|queue| queue.status())
    }

    /// Replaces the production queue with a caller-supplied writer and capacity.
    /// Test-only: the production capacity is fixed and its writer is the real
    /// [`StoreSaveWriter`]. The queue field is plain state on the engine, so this
    /// is the one persistence operation that needs it mutably.
    #[cfg(test)]
    pub(crate) fn install_test_save_queue(
        engine: &mut FileEngine,
        writer: Arc<dyn SaveWriter>,
        capacity: usize,
    ) {
        engine.save_queue = Some(Arc::new(SaveQueue::new(writer, capacity)));
    }
}

/// Library import and folder rescan.
///
/// The engine keeps the catalog, the store and the job registry; the service
/// owns no state. Each public `import`/`rescan` is wrapped in one
/// `JobScope::Import` lifecycle record. The constructor's opening scan calls
/// [`Self::rescan_inner`] so opening a library does not add a snapshot record.
pub(crate) struct ImportService<'a> {
    engine: &'a FileEngine,
}

impl<'a> ImportService<'a> {
    pub(crate) fn new(engine: &'a FileEngine) -> Self {
        Self { engine }
    }

    /// Imports explicit files and folders outside the library root.
    pub(crate) fn import(&self, paths: &[PathBuf]) -> anyhow::Result<Vec<ImageInfo>> {
        self.with_import_job(|| self.import_inner(paths))
    }

    /// Rescans the library folder, keeping edits already made.
    pub(crate) fn rescan(&self) -> anyhow::Result<()> {
        self.with_import_job(|| self.rescan_inner())
    }

    /// The rescan body without a lifecycle record; used by the opening scan.
    pub(crate) fn rescan_inner(&self) -> anyhow::Result<()> {
        let engine = self.engine;
        let mut found = Vec::new();
        collect(&engine.root, &engine.root, 0, &mut found)?;
        found.sort_by(|a, b| a.0.cmp(&b.0));
        let excluded: std::collections::HashSet<PathBuf> =
            engine.store.as_ref().map(|store| store.load_excluded().into_iter().collect()).unwrap_or_default();

        let mut st = engine.state.lock();
        let previous: HashMap<String, (AdjustmentSet, i8, ColorTag, Vec<String>, u32)> = st
            .entries
            .iter()
            .map(|e| {
                (
                    e.rel_path.clone(),
                    (e.adjustments.clone(), e.info.rating, e.info.color_tag, e.info.keywords.clone(), e.info.variant),
                )
            })
            .collect();
        // Keep an unchanged file's ImageId so its cached source, AI input,
        // pins and preview frames stay valid across a rescan.
        let previous_ids: HashMap<String, (ImageId, u64, u32, u32, FileKind)> = st
            .entries
            .iter()
            .map(|e| {
                (
                    e.rel_path.clone(),
                    (e.info.id, e.info.file_size, e.info.width, e.info.height, e.info.kind),
                )
            })
            .collect();

        let scanned: Vec<ScannedFile> = found
            .into_par_iter()
            .map(|(rel, path)| scan_file(&engine.root, rel, path))
            .collect();

        let mut from_cos = Vec::new();
        let mut reassigned = false;
        let mut removed_any = false;
        st.entries.clear();
        for scanned in scanned
            .into_iter()
            .filter(|scanned| !excluded.contains(&scanned.path))
            .flat_map(ScannedFile::library_variants)
        {
            let rel_path = &scanned.rel_path;
            let id = previous_ids
                .get(rel_path)
                .filter(|(_, file_size, width, height, kind)| {
                    *file_size == scanned.file_size
                        && *width == scanned.width
                        && *height == scanned.height
                        && *kind == scanned.kind
                })
                .map(|(id, ..)| *id)
                .unwrap_or_else(|| {
                    reassigned = true;
                    let id = ImageId(st.next_id);
                    st.next_id += 1;
                    id
                });
            let stored = previous.get(rel_path);
            let mut adjustments = stored.map(|s| s.0.clone()).unwrap_or_default();
            // Restore edits saved in a previous process; keep unsaved in-memory
            // edits made before this rescan.
            if adjustments.is_default() {
                if let Some(store) = &engine.store {
                    if let Ok(saved) = store.load_adjustments(rel_path) {
                        adjustments = saved;
                    }
                }
            }
            // C1 keeps per-image settings in a `.cos` sidecar; apply it when
            // neither our store nor this process has edits for the path.
            if adjustments.is_default() && st.catalog.entry(rel_path).is_none() {
                if let Some(cos) = scanned.c1_adjustments() {
                    from_cos.push((scanned.rel_path.clone(), cos.clone()));
                    adjustments = cos;
                }
            }
            let mut rating = stored.map(|s| s.1).unwrap_or(0);
            let mut color_tag = stored.map(|s| s.2).unwrap_or(ColorTag::None);
            let mut keywords = stored.map(|s| s.3.clone()).unwrap_or_default();
            let mut variant = stored.map(|s| s.4).unwrap_or(scanned.packed_variant);
            let mut trashed = false;

            // The catalog wins over the in-memory copy: it survives restarts.
            if let Some(entry) = st.catalog.entry(rel_path) {
                rating = entry.rating;
                color_tag = ColorTag::ALL[(entry.color_tag.clamp(0, 7)) as usize];
                keywords = entry.keywords.clone();
                variant = variant.max(entry.variant);
                trashed = entry.trashed;
            }

            if keywords.is_empty() {
                keywords = scanned.iptc.keywords.clone();
            }
            st.entries.push(FileEntry {
                info: ImageInfo {
                    id,
                    path: scanned.path.clone(),
                    file_name: scanned.file_name,
                    kind: scanned.kind,
                    heif: None,
                    width: scanned.width,
                    height: scanned.height,
                    file_size: scanned.file_size,
                    exif: scanned.exif,
                    iptc: scanned.iptc,
                    keywords,
                    rating,
                    color_tag,
                    variant,
                    has_adjustments: !adjustments.is_default(),
                    trashed,
                },
                path: scanned.path,
                rel_path: scanned.rel_path,
                adjustments,
            });
        }
        // Drop every cached resource whose id is no longer in the library.
        // IDs are stable for unchanged files, so hot sources and pins survive
        // a rescan; removed or rewritten files release their bytes at once.
        let living: std::collections::HashSet<ImageId> = st.entries.iter().map(|entry| entry.info.id).collect();
        let mut removed_bytes = 0usize;
        let stale_sources: Vec<ImageId> = st.sources.keys().copied().filter(|id| !living.contains(id)).collect();
        for id in stale_sources {
            removed_bytes = removed_bytes.saturating_add(evict_cached_source(&mut st, id));
        }
        let stale_ai: Vec<ImageId> = st.ai_inputs.keys().copied().filter(|id| !living.contains(id)).collect();
        for id in stale_ai {
            if let Some(input) = st.ai_inputs.remove(&id) {
                removed_bytes = removed_bytes.saturating_add(ai_input_bytes(&input));
            }
        }
        if st.native_raw.as_ref().is_some_and(|slot| !living.contains(&slot.image)) {
            if let Some(slot) = st.native_raw.take() {
                removed_bytes = removed_bytes.saturating_add(slot.bytes);
            }
        }
        removed_any |= removed_bytes > 0;
        st.source_cache.current_bytes = st.source_cache.current_bytes.saturating_sub(removed_bytes);
        let pins_before = st.source_cache.pins.len();
        st.source_cache.pins.retain(|id, _| living.contains(id));
        st.source_cache.recent.retain(|id| living.contains(id));
        st.source_reasons.retain(|id, _| living.contains(id));
        removed_any |= reassigned || pins_before != st.source_cache.pins.len();
        if removed_any {
            // Rendered frames of a rewritten or removed file must never be hit.
            st.preview.clear();
        }
        // Save the C1 snapshot once, so a later metadata edit and restart does
        // not fall back to defaults when the catalog entry blocks a re-read.
        drop(st);
        if let Some(store) = &engine.store {
            for (rel_path, adjustments) in from_cos {
                if let Err(err) = store.save_adjustments(&rel_path, &adjustments) {
                    log::warn!("save `.cos` adjustments {rel_path}: {err}");
                }
            }
        }
        Ok(())
    }

    fn import_inner(&self, paths: &[PathBuf]) -> anyhow::Result<Vec<ImageInfo>> {
        let engine = self.engine;
        let mut files = Vec::new();
        for path in paths {
            if path.is_dir() && !path.starts_with(&engine.root) {
                let mut found = Vec::new();
                collect(path, path, 0, &mut found)?;
                files.extend(found.into_iter().map(|(_, path)| path));
            } else if path.is_file() && is_supported(path) {
                files.push(path.clone());
            }
        }
        let scanned: Vec<ScannedFile> = files
            .into_par_iter()
            .map(|path| {
                let rel = path
                    .strip_prefix(&engine.root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                scan_file(&engine.root, rel, path)
            })
            .collect();
        let mut st = engine.state.lock();
        let mut added = Vec::with_capacity(scanned.len());
        for scanned in scanned.into_iter().flat_map(ScannedFile::library_variants) {
            // Re-importing a path that is already in the library must not add a
            // second entry. Packed EIP variants carry distinct `#eip-vN` rel
            // paths, so this never drops a variant; it only extends the
            // existing packed de-duplication to plain files.
            if st.entries.iter().any(|entry| entry.rel_path == scanned.rel_path) {
                continue;
            }
            let adjustments = if scanned.packed.is_some() {
                let saved = engine.store.as_ref().map(|store| store.load_adjustments(&scanned.rel_path));
                match saved {
                    Some(Ok(saved))
                        if !saved.is_default() || st.catalog.entry(&scanned.rel_path).is_some() => saved,
                    Some(Ok(_)) => {
                        let adjustments = scanned.c1_adjustments().unwrap_or_default();
                        if !adjustments.is_default() {
                            if let Some(store) = &engine.store {
                                if let Err(err) = store.save_adjustments(&scanned.rel_path, &adjustments) {
                                    log::warn!("save EIP adjustments {}: {err}", scanned.rel_path);
                                }
                            }
                        }
                        adjustments
                    }
                    _ => scanned.c1_adjustments().unwrap_or_default(),
                }
            } else {
                AdjustmentSet::default()
            };
            let id = ImageId(st.next_id);
            st.next_id += 1;
            let keywords = scanned.iptc.keywords.clone();
            let info = ImageInfo {
                id,
                path: scanned.path.clone(),
                file_name: scanned.file_name,
                kind: scanned.kind,
                heif: None,
                width: scanned.width,
                height: scanned.height,
                file_size: scanned.file_size,
                exif: scanned.exif,
                iptc: scanned.iptc,
                keywords,
                rating: 0,
                color_tag: ColorTag::None,
                variant: scanned.packed_variant,
                has_adjustments: !adjustments.is_default(),
                trashed: false,
            };
            added.push(info.clone());
            st.entries.push(FileEntry {
                info,
                path: scanned.path,
                rel_path: scanned.rel_path,
                adjustments,
            });
        }
        Ok(added)
    }

    pub(crate) fn import_job_snapshot(&self) -> crate::job_lifecycle::RegistrySnapshot {
        self.engine.import_jobs.snapshot()
    }

    /// One lifecycle record per explicit import/rescan call. Terminal records
    /// are pruned so repeated imports cannot grow the map without bound; live
    /// records are never pruned.
    fn with_import_job<T>(&self, body: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
        let engine = self.engine;
        const IMPORT_JOB_CAP: usize = 64;
        if engine.import_jobs.len() >= IMPORT_JOB_CAP {
            engine.import_jobs.prune_terminal(IMPORT_JOB_CAP / 2);
        }
        let jid = engine.import_jobs.submit(JobScope::Import, CancelToken::new());
        let _ = engine.import_jobs.transition(jid, JobState::running(0.0));
        // An empty path list still reaches `Succeeded`; only an error fails.
        let result = body();
        let _ = match &result {
            Ok(_) => engine.import_jobs.transition(jid, JobState::Succeeded),
            Err(error) => engine.import_jobs.transition(jid, JobState::failed(format!("{error:#}"))),
        };
        result
    }
}

/// Catalog-mutating library filesystem commands: move, copy, rename,
/// locate/synchronize, folder creation and delete/forget.
///
/// The service borrows the engine and owns no state. Path normalization,
/// no-clobber primitives and their tests stay in `library_ops.rs`; the
/// `FileEngine` entry points delegate here. Every batch is preflighted before
/// any file is touched, C1 catalogs stay read-only, and the error text and
/// ordering match the original `impl FileEngine` bodies.
pub(crate) struct LibraryService<'a> {
    engine: &'a FileEngine,
}

impl<'a> LibraryService<'a> {
    pub(crate) fn new(engine: &'a FileEngine) -> Self {
        Self { engine }
    }

    pub(crate) fn manage_library(&self, action: LibraryAction) -> anyhow::Result<Vec<ImageInfo>> {
        let engine = self.engine;
        anyhow::ensure!(!engine.c1_catalog, "library filesystem operations are read-only for a Capture One catalog");
        if let LibraryAction::Synchronize { folder, recursive } = &action {
            let folder = existing_directory(&engine.root, folder)?;
            engine.rescan()?;
            let _ = recursive;
            return Ok(engine.images().into_iter().filter(|image| image.path.starts_with(&folder)).collect());
        }
        match action {
            LibraryAction::Move { images, destination } => self.move_images(&images, &destination),
            LibraryAction::MoveFolder { from, to } => self.move_folder(&from, &to),
            LibraryAction::Rename { image, name } => {
                validate_name(&name)?;
                let source = selected_paths(&engine.state.lock().entries, &[image])?.remove(0);
                let target = source.with_file_name(name);
                self.move_paths(vec![(source, target)])
            }
            LibraryAction::RenameMany { names } => self.rename_many(&names),
            LibraryAction::Locate { from, to } => self.relink_folder(&from, &to),
            LibraryAction::LocateImage { image, to } => self.relink_image(image, &to),
            LibraryAction::Forget { images } => self.forget(&images),
            LibraryAction::CreateFolder { path } => self.create_folder(&path),
            LibraryAction::Copy { images, destination } => self.copy_images(&images, &destination),
            LibraryAction::Delete { images, permanent } => self.delete_images(&images, permanent),
            LibraryAction::Synchronize { .. } => unreachable!(),
        }
    }

    fn move_images(&self, images: &[ImageId], destination: &Path) -> anyhow::Result<Vec<ImageInfo>> {
        let engine = self.engine;
        let destination = existing_directory(&engine.root, destination)?;
        let entries = engine.state.lock().entries.clone();
        let mut pairs = Vec::new();
        for source in selected_paths(&entries, images)? {
            let target = destination.join(source.file_name().context("source has no filename")?);
            if source != target {
                pairs.push((source, target));
            }
        }
        self.move_paths(pairs)
    }

    fn rename_many(&self, names: &[(ImageId, String)]) -> anyhow::Result<Vec<ImageInfo>> {
        let engine = self.engine;
        let entries = engine.state.lock().entries.clone();
        let mut pairs: Vec<(PathBuf, PathBuf)> = Vec::new();
        for (image, name) in names {
            validate_name(name)?;
            let source = selected_paths(&entries, &[*image])?.remove(0);
            let target = source.with_file_name(name);
            if let Some((_, previous)) = pairs.iter().find(|(from, _)| path_key(from) == path_key(&source)) {
                anyhow::ensure!(path_key(previous) == path_key(&target), "variants of one file have conflicting rename targets");
            } else if source != target {
                pairs.push((source, target));
            }
        }
        self.move_paths(pairs)
    }

    fn move_paths(&self, pairs: Vec<(PathBuf, PathBuf)>) -> anyhow::Result<Vec<ImageInfo>> {
        let engine = self.engine;
        if pairs.is_empty() {
            return Ok(engine.images());
        }
        let pairs = expand_sidecars(pairs)?;
        self.apply_pairs(&pairs, false)?;
        let map: HashMap<PathBuf, PathBuf> = pairs
            .iter()
            .filter(|(from, _)| !from.extension().is_some_and(|extension| extension.to_string_lossy().eq_ignore_ascii_case("xmp")))
            .cloned()
            .collect();
        self.relink_entries(&map)
    }

    fn move_folder(&self, from: &Path, to: &Path) -> anyhow::Result<Vec<ImageInfo>> {
        let engine = self.engine;
        let from = existing_directory(&engine.root, from)?;
        anyhow::ensure!(from != engine.root && !engine.root.starts_with(&from), "cannot move the library root or its ancestor");
        let to = resolve_inside(&engine.root, to)?;
        validate_name(to.file_name().and_then(|name| name.to_str()).context("invalid folder name")?)?;
        let parent = existing_directory(&engine.root, to.parent().context("missing destination parent")?)?;
        let to = parent.join(to.file_name().context("missing destination name")?);
        anyhow::ensure!(!to.starts_with(&from), "cannot move a folder into itself");
        ensure_absent(&to)?;
        let entries = engine.state.lock().entries.clone();
        let map: HashMap<PathBuf, PathBuf> = entries
            .iter()
            .filter(|entry| entry.path.starts_with(&from))
            .map(|entry| (entry.path.clone(), to.join(entry.path.strip_prefix(&from).unwrap_or(Path::new("")))))
            .collect();
        std::fs::rename(&from, &to).with_context(|| format!("moving folder {} to {}", from.display(), to.display()))?;
        match self.relink_entries(&map) {
            Ok(images) => Ok(images),
            Err(error) => {
                let _ = std::fs::rename(&to, &from);
                Err(error)
            }
        }
    }

    fn relink_folder(&self, from: &Path, to: &Path) -> anyhow::Result<Vec<ImageInfo>> {
        let engine = self.engine;
        let from = resolve_inside(&engine.root, from)?;
        anyhow::ensure!(from != engine.root && !is_private(&from), "cannot relink the library root or private data");
        let to = existing_directory(&engine.root, to)?;
        let entries = engine.state.lock().entries.clone();
        anyhow::ensure!(folder_is_offline(&entries, &from), "Locate requires an offline folder");
        let map: HashMap<PathBuf, PathBuf> = entries
            .iter()
            .filter(|entry| entry.path.starts_with(&from))
            .map(|entry| (entry.path.clone(), to.join(entry.path.strip_prefix(&from).unwrap_or(Path::new("")))))
            .collect();
        for target in map.values() {
            anyhow::ensure!(target.is_file(), "relink target unavailable: {}", target.display());
        }
        self.relink_entries(&map)
    }

    fn relink_image(&self, image: ImageId, to: &Path) -> anyhow::Result<Vec<ImageInfo>> {
        let engine = self.engine;
        let entries = engine.state.lock().entries.clone();
        let from = selected_paths(&entries, &[image])?.remove(0);
        anyhow::ensure!(!from.exists(), "LocateImage requires an offline source");
        let to = canonical_root(&resolve_inside(&engine.root, to)?)?;
        anyhow::ensure!(to.is_file() && !is_private(&to), "relink target must be an existing image");
        anyhow::ensure!(
            !entries.iter().any(|entry| entry.path != from && path_key(&entry.path) == path_key(&to)),
            "relink target already belongs to another source"
        );
        self.relink_entries(&HashMap::from([(from, to)]))
    }

    fn create_folder(&self, path: &Path) -> anyhow::Result<Vec<ImageInfo>> {
        let engine = self.engine;
        let path = resolve_inside(&engine.root, path)?;
        validate_name(path.file_name().and_then(|name| name.to_str()).context("invalid folder name")?)?;
        let parent = existing_directory(&engine.root, path.parent().context("folder has no parent")?)?;
        let path = parent.join(path.file_name().context("missing folder name")?);
        anyhow::ensure!(!is_private(&path), "folder name is reserved for private data");
        ensure_absent(&path)?;
        std::fs::create_dir(&path)?;
        Ok(engine.images())
    }

    fn copy_images(&self, images: &[ImageId], destination: &Path) -> anyhow::Result<Vec<ImageInfo>> {
        let engine = self.engine;
        let destination = existing_directory(&engine.root, destination)?;
        let entries = engine.state.lock().entries.clone();
        let sources = selected_paths(&entries, images)?;
        let mut pairs = Vec::new();
        for source in &sources {
            let target = destination.join(source.file_name().context("source has no filename")?);
            anyhow::ensure!(path_key(&target) != path_key(source), "cannot copy a file onto itself");
            pairs.push((source.clone(), target));
        }
        let pairs = expand_sidecars(pairs)?;
        self.apply_pairs(&pairs, true)?;
        let mut copied = Vec::new();
        for source in sources {
            let source_entry = entries.iter().find(|entry| entry.path == source).cloned().context("source disappeared")?;
            let target = destination.join(source.file_name().context("source has no filename")?);
            let info = engine.add_file(&target)?;
            if let Some(entry) = engine.state.lock().entries.iter_mut().find(|entry| entry.info.id == info.id) {
                entry.adjustments = source_entry.adjustments.clone();
            }
            if let Some(store) = &engine.store {
                store.save_adjustments(&rel_key(&engine.root, &target)?, &source_entry.adjustments)?;
            }
            let record = {
                let state = engine.state.lock();
                let entry = state.entries.iter().find(|entry| entry.info.id == info.id).unwrap();
                entry_record(&entry.info, entry.rel_path.clone())
            };
            let catalog = {
                let mut state = engine.state.lock();
                state.catalog.upsert(record);
                state.catalog.clone()
            };
            if let Some(store) = &engine.store {
                store.save_index(&catalog)?;
            }
            copied.push(engine.images().into_iter().find(|image| image.id == info.id).unwrap());
        }
        Ok(copied)
    }

    fn delete_images(&self, images: &[ImageId], permanent: bool) -> anyhow::Result<Vec<ImageInfo>> {
        let engine = self.engine;
        let entries = engine.state.lock().entries.clone();
        let paths = selected_paths(&entries, images)?;
        let mut delete = Vec::new();
        for path in &paths {
            anyhow::ensure!(path.is_file() && !is_private(path), "source unavailable or protected: {}", path.display());
            delete.push(path.clone());
            for sidecar in sidecars(path)? {
                delete.push(sidecar);
            }
        }
        if permanent {
            for path in &delete {
                std::fs::remove_file(path)?;
            }
        } else {
            recycle_paths(&delete)?;
        }
        let mut state = engine.state.lock();
        state.entries.retain(|entry| !images.contains(&entry.info.id));
        for path in &paths {
            if !state.entries.iter().any(|entry| entry.path == *path) {
                if let Some(store) = &engine.store {
                    let mut excluded = store.load_excluded();
                    excluded.push(path.clone());
                    store.save_excluded(&excluded)?;
                }
            }
        }
        for image in images {
            if let Some(entry) = entries.iter().find(|entry| entry.info.id == *image) {
                state.catalog.remove(&entry.rel_path);
                if let Some(store) = &engine.store {
                    store.remove_adjustments(&entry.rel_path)?;
                }
            }
        }
        if let Some(store) = &engine.store {
            store.save_index(&state.catalog)?;
        }
        Ok(state.entries.iter().map(|entry| entry.info.clone()).collect())
    }

    fn forget(&self, images: &[ImageId]) -> anyhow::Result<Vec<ImageInfo>> {
        let engine = self.engine;
        let mut state = engine.state.lock();
        let removed: Vec<FileEntry> = state.entries.iter().filter(|entry| images.contains(&entry.info.id)).cloned().collect();
        anyhow::ensure!(!removed.is_empty(), "no matching images");
        state.entries.retain(|entry| !images.contains(&entry.info.id));
        for entry in &removed {
            state.catalog.remove(&entry.rel_path);
            if let Some(store) = &engine.store {
                store.remove_adjustments(&entry.rel_path)?;
            }
        }
        if let Some(store) = &engine.store {
            let mut excluded = store.load_excluded();
            for entry in &removed {
                if !state.entries.iter().any(|remaining| remaining.path == entry.path) {
                    excluded.push(entry.path.clone());
                }
            }
            store.save_excluded(&excluded)?;
            let mut albums: Vec<UserCollection> = store.load_user_collections();
            for album in &mut albums {
                album.images.retain(|id| !images.contains(id));
            }
            store.save_user_collections(&albums)?;
            store.save_index(&state.catalog)?;
        }
        Ok(state.entries.iter().map(|entry| entry.info.clone()).collect())
    }

    fn apply_pairs(&self, pairs: &[(PathBuf, PathBuf)], copy: bool) -> anyhow::Result<()> {
        let engine = self.engine;
        let mut sources = HashSet::new();
        let mut targets = HashSet::new();
        for (from, to) in pairs {
            anyhow::ensure!(from.is_file() && !is_private(from), "source unavailable or protected: {}", from.display());
            anyhow::ensure!(!is_private(to), "destination is private data");
            anyhow::ensure!(to.parent().is_some_and(Path::is_dir), "destination folder unavailable: {}", to.display());
            validate_name(to.file_name().and_then(|name| name.to_str()).context("invalid destination filename")?)?;
            anyhow::ensure!(sources.insert(path_key(from)), "one source cannot move to multiple destinations");
            anyhow::ensure!(targets.insert(path_key(to)), "duplicate destination: {}", to.display());
            ensure_absent(to)?;
            anyhow::ensure!(
                !engine.state.lock().entries.iter().any(|entry| {
                    path_key(&entry.path) == path_key(to)
                        && !pairs.iter().any(|(source, _)| path_key(source) == path_key(&entry.path))
                }),
                "destination already cataloged: {}",
                to.display()
            );
        }
        let mut completed = Vec::new();
        for (from, to) in pairs {
            let result = if copy { copy_no_replace(from, to) } else { move_no_replace(from, to) };
            if let Err(error) = result {
                return Err(rollback(&completed, copy, error));
            }
            completed.push((from.clone(), to.clone()));
        }
        Ok(())
    }

    fn relink_entries(&self, map: &HashMap<PathBuf, PathBuf>) -> anyhow::Result<Vec<ImageInfo>> {
        let engine = self.engine;
        if map.is_empty() {
            return Ok(engine.images());
        }
        let mut state = engine.state.lock();
        let mut entries = std::mem::take(&mut state.entries);
        for entry in &mut entries {
            let Some(target) = map.get(&entry.path) else { continue };
            let old_path = entry.path.clone();
            let old_rel = entry.rel_path.clone();
            let old_base = rel_key(&engine.root, &old_path)?;
            let suffix = old_rel.strip_prefix(&old_base).unwrap_or("");
            let new_rel = format!("{}{suffix}", rel_key(&engine.root, target)?);
            entry.path = target.clone();
            entry.info.path = target.clone();
            entry.info.file_name = target.file_name().unwrap_or_default().to_string_lossy().into_owned();
            entry.rel_path = new_rel.clone();
            state.catalog.remove(&old_rel);
            state.catalog.upsert(entry_record(&entry.info, new_rel.clone()));
            if let Some(store) = &engine.store {
                store.save_adjustments(&new_rel, &entry.adjustments)?;
                store.remove_adjustments(&old_rel)?;
            }
        }
        state.entries = entries;
        state.sources.clear();
        state.ai_inputs.clear();
        state.native_raw = None;
        // Ids were reassigned, so every pin would point at the wrong image.
        state.source_cache.pins.clear();
        state.source_cache.recent.clear();
        state.source_cache.current_bytes = 0;
        state.preview = PreviewCache::default();
        if let Some(store) = &engine.store {
            store.save_index(&state.catalog)?;
        }
        Ok(state.entries.iter().map(|entry| entry.info.clone()).collect())
    }
}
