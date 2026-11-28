//! Frame cache for interactive previews.
//!
//! The pipeline is deterministic, so the same image, resolved parameters,
//! region and output size always produce the same bytes. A slider drag that
//! lands on a value already rendered (or a repaint of an unchanged frame) can
//! reuse the pixels instead of running the pipeline again.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;

use serde::Serialize;

use crate::{
    AdjustmentSet, CoverPatch, Histogram, ImageId, Layer, LayerKind, LocalAdjustments, Mask,
    NodeGraph, NodeKind, NodeLink, NormRect, RgbaImage, TechnicalProcessingMethod, ToneCurvePreset,
    WhiteBalanceMode,
};

/// Bumped by the RAW cache when the bytes behind a source path change. Every
/// cache drops its frames on the next access, so a rendered frame from the
/// replaced file can never be served again.
static SOURCE_EPOCH: AtomicU64 = AtomicU64::new(0);

pub(crate) fn invalidate_source_identity() {
    SOURCE_EPOCH.fetch_add(1, Ordering::Relaxed);
}

/// Serialises tests that inspect a cache with tests (in other modules) that
/// replace a source file and therefore bump [`SOURCE_EPOCH`].
#[cfg(test)]
pub(crate) fn test_source_epoch_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
pub(crate) fn source_epoch() -> u64 {
    SOURCE_EPOCH.load(Ordering::Relaxed)
}

/// Keeps every frame the viewer is likely to bounce between (drag back and
/// forth, before/after, resize).
pub(crate) const CAPACITY: usize = 8;
/// Eight 8192 px finals would be ~1.4 GB, so the cache also has a byte budget.
const MAX_BYTES: usize = 256 << 20;

/// Bump this whenever a change to decoding or rendering alters output pixels.
/// Cached frames outlive rebuilds, so without a bump an upgraded engine would
/// keep serving frames produced by the old algorithm.
const RENDER_VERSION: u32 = 1;

/// Everything that changes the rendered bytes. `generation` is deliberately
/// absent: a redraw of the same content must hit.
///
/// This key is **document-only**. Producer/runtime inputs are not fields of
/// `AdjustmentSet` (they live in `crate::document::RuntimeInputs`), so they are
/// not covered here. When a render path starts threading runtime inputs into
/// previews, it MUST add them to this key (for example a
/// `RuntimeInputs::fingerprint` field) in the same change; otherwise a frame
/// rendered with different masks could be served from cache. No production path
/// attaches producer inputs today, so the current omission is not a regression.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PreviewKey {
    build: u64,
    source_snapshot: Option<u64>,
    image: ImageId,
    params: ParamsFingerprint,
    long_side: u32,
    region: [u32; 4],
    bypass: bool,
    mask_overlay: Option<u64>,
    fast: bool,
}

impl PreviewKey {
    /// Separate frames made from different in-memory source resolutions.
    pub(crate) fn with_source(mut self, snapshot: u64) -> Self {
        self.source_snapshot = Some(snapshot);
        self
    }

    /// True when two requests differ at most in their layer rows and node
    /// topology. The caller must still run [`NodeGraph::dirty_between`]: this
    /// only rules out every other input, including spots and all develop /
    /// detail / portrait parameters.
    pub(crate) fn graph_reusable_with(&self, other: &Self) -> bool {
        self.build == other.build
            && self.source_snapshot == other.source_snapshot
            && self.image == other.image
            && self.long_side == other.long_side
            && self.region == other.region
            && self.bypass == other.bypass
            && self.mask_overlay == other.mask_overlay
            && self.fast == other.fast
            && self.params.same_except_graph(&other.params)
    }
}

/// The last preview of one image, for reusing its pixels after an edit that
/// only touched pruned branches.
///
/// `request` is the logical request the frame answers; `frame` is the cache key
/// that actually holds its pixels. They differ only after a graph-dirty reuse:
/// the pixels keep the earlier key, so no alias entry (and no double-counted
/// byte budget) is needed.
#[derive(Clone, Default)]
pub(crate) struct GraphGate {
    request: Option<PreviewKey>,
    frame: Option<PreviewKey>,
    graph: NodeGraph,
    rows: Vec<Layer>,
}

impl GraphGate {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The cache key of a frame that is byte-identical to what `key` would
    /// render, or `None` when the render must run.
    ///
    /// Both graphs must already be [`NodeGraph::repaired`] against their rows;
    /// repair is what turns a row without a node into a real render node, so a
    /// raw row edit cannot hide from `dirty_between`.
    pub(crate) fn reuse(
        &self,
        key: &PreviewKey,
        graph: &NodeGraph,
        rows: &[Layer],
    ) -> Option<PreviewKey> {
        let request = self.request.as_ref()?;
        let frame = self.frame.as_ref()?;
        if !key.graph_reusable_with(request) {
            return None;
        }
        // A refused graph (cycle through the output) renders background only;
        // never carry a frame across one. The dirty API must not guess an order.
        if graph.plan().is_err() || self.graph.plan().is_err() {
            return None;
        }
        if !graph.dirty_between(&self.graph, rows, &self.rows).is_empty() {
            return None;
        }
        Some(*frame)
    }

    /// Remember what produced the frame now on screen. `request` is the
    /// logical request, `frame` the cache key holding its pixels.
    pub(crate) fn record(
        &mut self,
        request: PreviewKey,
        frame: PreviewKey,
        graph: NodeGraph,
        rows: Vec<Layer>,
    ) {
        self.request = Some(request);
        self.frame = Some(frame);
        self.graph = graph;
        self.rows = rows;
    }
}

/// One cached frame. The pixel buffer is shared, so a hit costs an `Arc` clone.
#[derive(Clone)]
pub(crate) struct CachedFrame {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixels: Arc<Vec<u8>>,
    pub(crate) histogram: Histogram,
}

impl CachedFrame {
    pub(crate) fn into_image(self) -> RgbaImage {
        RgbaImage::from_shared(self.width, self.height, self.pixels)
    }
}

/// LRU of rendered frames, most recently used last.
pub(crate) struct PreviewCache {
    entries: Vec<(PreviewKey, CachedFrame)>,
    capacity: usize,
    max_bytes: usize,
    bytes: usize,
    /// [`SOURCE_EPOCH`] at the last access; a newer epoch clears the cache.
    source_epoch: u64,
}

impl Default for PreviewCache {
    fn default() -> Self {
        Self::new(CAPACITY)
    }
}

impl PreviewCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self::with_budget(capacity, MAX_BYTES)
    }

    fn with_budget(capacity: usize, max_bytes: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            max_bytes,
            bytes: 0,
            source_epoch: SOURCE_EPOCH.load(Ordering::Relaxed),
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Drops every frame when a source file's bytes changed since the last
    /// access. Cheap to call on every hit.
    fn sync_source_epoch(&mut self) {
        let epoch = SOURCE_EPOCH.load(Ordering::Relaxed);
        if epoch != self.source_epoch {
            self.source_epoch = epoch;
            self.entries.clear();
            self.bytes = 0;
        }
    }

    pub(crate) fn get(&mut self, key: PreviewKey) -> Option<CachedFrame> {
        self.sync_source_epoch();
        let pos = self.entries.iter().position(|(k, _)| *k == key)?;
        let (key, frame) = self.entries.remove(pos);
        self.entries.push((key, frame.clone()));
        Some(frame)
    }

    pub(crate) fn insert(&mut self, key: PreviewKey, frame: CachedFrame) {
        self.sync_source_epoch();
        if let Some(pos) = self.entries.iter().position(|(k, _)| *k == key) {
            self.bytes -= self.entries.remove(pos).1.pixels.len();
        }
        self.bytes += frame.pixels.len();
        self.entries.push((key, frame));
        while self.entries.len() > self.capacity || (self.bytes > self.max_bytes && self.entries.len() > 1) {
            self.bytes -= self.entries.remove(0).1.pixels.len();
        }
    }

    /// Drops every frame, for when the engine build behind a long-lived
    /// instance changes.
    #[allow(dead_code)] // callers live outside this crate-internal module
    pub fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }
}

/// Key for one preview request. `max_size` must already be the effective
/// output size (Draft clamps it), so a resize that keeps the output size hits.
pub(crate) fn key(
    image: ImageId,
    adj: &AdjustmentSet,
    region: NormRect,
    max_size: u32,
    bypass: bool,
    mask_overlay: Option<u64>,
    fast: bool,
) -> PreviewKey {
    PreviewKey {
        build: build_fingerprint(),
        source_snapshot: None,
        image,
        params: ParamsFingerprint::of(adj),
        long_side: max_size,
        region: [region.x.to_bits(), region.y.to_bits(), region.width.to_bits(), region.height.to_bits()],
        bypass,
        mask_overlay,
        fast,
    }
}

/// Semantic groups of the parameter fingerprint.
///
/// [`PreviewKey`] compares all six groups, so a change in any of them still
/// misses the whole frame. The split exists so a future stage cache can depend
/// on one group plus its upstream groups instead of re-serialising the whole
/// [`AdjustmentSet`]. A single group is not a frame key: stages run in the
/// order characterization → geometry → develop → detail → local → portrait, so
/// a stage cache must include every group at or before its own stage.
///
/// This covers the persistent document only. Per-job producer inputs are no
/// longer part of `AdjustmentSet`, so a caller that lets runtime inputs affect
/// output must mix `crate::document::RuntimeInputs::fingerprint` into the cache
/// key itself.
///
/// The groups are disjoint and every field of [`AdjustmentSet`] belongs to
/// exactly one of them. `preview_cache::tests` enforces that against the
/// serialized schema and against a fully populated set, and also pins the
/// fields that are deliberately excluded because they never reach a pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ParamsFingerprint {
    /// Input profile, film curve, tone-curve preset and style selector.
    characterization: u64,
    /// Crop, orientation, keystone and lens geometry/resampling.
    geometry: u64,
    /// White balance, tone, colour and vignetting point/near-point chain.
    develop: u64,
    /// Sharpening, noise reduction and moire.
    detail: u64,
    /// Spots, layers, masks and node topology.
    local: u64,
    /// The non-graph half of [`Self::local`] (spots). Two requests that agree
    /// here and on every group above differ at most in their layer rows and
    /// node topology, which [`NodeGraph::dirty_between`] then adjudicates.
    local_rest: u64,
    /// The layer-row and node-topology half of [`Self::local`]. Kept separate
    /// so the preview can ask "did only the graph move?" without treating a
    /// pruned edit as a frame miss.
    local_graph: u64,
    /// Retouch faces, portrait contract parameters and producer inputs.
    portrait: u64,
}

impl ParamsFingerprint {
    fn of(adj: &AdjustmentSet) -> Self {
        let local_rest = local_rest_fingerprint(adj);
        let local_graph = local_graph_fingerprint(adj);
        Self {
            characterization: characterization_fingerprint(adj),
            geometry: geometry_fingerprint(adj),
            develop: develop_fingerprint(adj),
            detail: detail_fingerprint(adj),
            local: local_fingerprint_of(local_rest, local_graph),
            local_rest,
            local_graph,
            portrait: portrait_fingerprint(adj),
        }
    }
    /// Every group except the layer rows and node topology agrees. A true
    /// result means `NodeGraph::dirty_between` alone decides whether the two
    /// requests render differently; it must never be used to skip that check.
    pub(crate) fn same_except_graph(&self, other: &Self) -> bool {
        self.characterization == other.characterization
            && self.geometry == other.geometry
            && self.develop == other.develop
            && self.detail == other.detail
            && self.portrait == other.portrait
            && self.local_rest == other.local_rest
    }
}

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a writer over whatever `serde_json` emits. Every field is tagged so
/// adjacent values cannot alias each other, and optional/empty fields are
/// always written (as `null`/`[]`), so a defaulted field still contributes.
struct Fnv(u64);

impl Fnv {
    fn new(tag: &str) -> Self {
        let mut hash = Self(FNV_OFFSET_BASIS);
        hash.tag(tag);
        hash
    }

    fn tag(&mut self, tag: &str) {
        let _ = self.write_all(tag.as_bytes());
        let _ = self.write_all(&[0xff]);
    }

    fn raw(&mut self, bytes: &[u8]) {
        let _ = self.write_all(bytes);
    }

    fn u64(&mut self, tag: &str, value: u64) {
        self.tag(tag);
        self.raw(&value.to_le_bytes());
    }

    fn bytes(&mut self, tag: &str, value: &[u8]) {
        self.tag(tag);
        self.u64("len", value.len() as u64);
        self.raw(value);
    }

    fn f32s(&mut self, tag: &str, values: &[f32]) {
        self.tag(tag);
        self.u64("len", values.len() as u64);
        for value in values {
            self.raw(&value.to_bits().to_le_bytes());
        }
    }

    fn f64s(&mut self, tag: &str, values: &[f64]) {
        self.tag(tag);
        self.u64("len", values.len() as u64);
        for value in values {
            self.raw(&value.to_bits().to_le_bytes());
        }
    }

    fn json<T: Serialize + ?Sized>(&mut self, tag: &str, value: &T) {
        self.tag(tag);
        if serde_json::to_writer(&mut *self, value).is_err() {
            self.tag("<serialize-error>");
        }
        self.tag(";");
    }
}

impl Write for Fnv {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        for byte in bytes {
            self.0 ^= *byte as u64;
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// `base` without `engine_version`: a display string no render path reads.
#[derive(Serialize)]
struct BaseView<'a> {
    icc_profile: &'a Option<String>,
    curve: ToneCurvePreset,
    processing_method: TechnicalProcessingMethod,
    film_curve: &'a Option<String>,
}

/// `white_balance` without `skin_tone`: a UI preset name the renderer never
/// reads (the tone it would describe comes from `mode`/`kelvin`/`tint`/`gains`).
#[derive(Serialize)]
struct WhiteBalanceView<'a> {
    mode: WhiteBalanceMode,
    kelvin: f32,
    tint: f32,
    gains: &'a Option<[f32; 3]>,
}

/// A layer minus the three fields that only label or catalogue it.
#[derive(Serialize)]
struct LayerView<'a> {
    id: u64,
    kind: LayerKind,
    enabled: bool,
    opacity: f32,
    mask: &'a Mask,
    adjustments: &'a LocalAdjustments,
    select_point: &'a Option<[f32; 2]>,
    cover: &'a [CoverPatch],
}

/// A node without its editor position.
#[derive(Serialize)]
struct NodeView {
    id: u64,
    kind: NodeKind,
}

#[derive(Serialize)]
struct NodeGraphView<'a> {
    nodes: Vec<NodeView>,
    links: &'a [NodeLink],
}

fn characterization_fingerprint(adj: &AdjustmentSet) -> u64 {
    let mut hash = Fnv::new("characterization-v1");
    let base = BaseView {
        icc_profile: &adj.base.icc_profile,
        curve: adj.base.curve,
        processing_method: adj.base.processing_method,
        film_curve: &adj.base.film_curve,
    };
    hash.json("base", &base);
    hash.json("style", &adj.style);
    hash.0
}

fn geometry_fingerprint(adj: &AdjustmentSet) -> u64 {
    let mut hash = Fnv::new("geometry-v1");
    hash.json("geometry", &adj.geometry);
    hash.json("lens", &adj.lens);
    // `serde(skip)` hides these from the document, but they still decide
    // pixels, so hash the values themselves.
    let geometry = &adj.geometry;
    hash.f64s("geometry.distortion.v", &geometry.distortion.v);
    hash.f64s("geometry.distortion.x", &geometry.distortion.x);
    hash.f64s("geometry.distortion.m", &geometry.distortion.m);
    hash.f32s("geometry.optical_center", &geometry.optical_center);
    match adj.lens.uniform_light_grid.as_deref() {
        Some(grid) => {
            hash.tag("lens.uniform_light_grid.some");
            hash.u64("columns", u64::from(grid.columns));
            hash.u64("rows", u64::from(grid.rows));
            hash.u64("nodes", grid.nodes.len() as u64);
            for node in &grid.nodes {
                hash.u64("luminance", u64::from(node.luminance));
                hash.u64("dx", node.dx as i64 as u64);
                hash.u64("dy", node.dy as i64 as u64);
            }
            hash.bytes("header", &grid.header);
        }
        None => hash.tag("lens.uniform_light_grid.none"),
    }
    hash.0
}

fn develop_fingerprint(adj: &AdjustmentSet) -> u64 {
    let mut hash = Fnv::new("develop-v1");
    let white_balance = WhiteBalanceView {
        mode: adj.white_balance.mode,
        kelvin: adj.white_balance.kelvin,
        tint: adj.white_balance.tint,
        gains: &adj.white_balance.gains,
    };
    hash.json("white_balance", &white_balance);
    hash.json("exposure", &adj.exposure);
    hash.json("hdr", &adj.hdr);
    hash.json("levels", &adj.levels);
    hash.json("curves", &adj.curves);
    hash.json("clarity", &adj.clarity);
    hash.json("dehaze", &adj.dehaze);
    hash.json("vignetting", &adj.vignetting);
    hash.json("color_balance", &adj.color_balance);
    hash.json("color_balance_gains", &adj.color_balance_gains);
    hash.json("color_editor", &adj.color_editor);
    hash.json("black_white", &adj.black_white);
    hash.json("film_grain", &adj.film_grain);
    hash.0
}

fn detail_fingerprint(adj: &AdjustmentSet) -> u64 {
    let mut hash = Fnv::new("detail-v1");
    hash.json("sharpening", &adj.sharpening);
    hash.json("noise_reduction", &adj.noise_reduction);
    hash.json("moire", &adj.moire);
    hash.0
}

/// Local inputs that are not a layer row or the node graph. The graph half is
/// re-derived by [`NodeGraph::dirty_between`]; spots are not, so they must
/// still match before a frame can be reused.
fn local_rest_fingerprint(adj: &AdjustmentSet) -> u64 {
    let mut hash = Fnv::new("local-rest-v1");
    hash.json("spots", &adj.spots);
    hash.0
}

/// Layer rows and node topology, exactly the part [`NodeGraph::dirty_between`]
/// compares.
fn local_graph_fingerprint(adj: &AdjustmentSet) -> u64 {
    let mut hash = Fnv::new("local-graph-v1");
    let layers: Vec<LayerView<'_>> = adj
        .layers
        .iter()
        .map(|layer| LayerView {
            id: layer.id,
            kind: layer.kind,
            enabled: layer.enabled,
            opacity: layer.opacity,
            mask: &layer.mask,
            adjustments: &layer.adjustments,
            select_point: &layer.select_point,
            cover: &layer.cover,
        })
        .collect();
    hash.json("layers", &layers);
    match adj.nodes.as_ref() {
        Some(graph) => {
            let view = NodeGraphView {
                nodes: graph
                    .nodes
                    .iter()
                    .map(|node| NodeView { id: node.id, kind: node.kind })
                    .collect(),
                links: &graph.links,
            };
            hash.json("nodes", &view);
        }
        None => hash.tag("nodes.none"),
    }
    hash.0
}

/// The whole local group, from its two halves, so the frame key still moves
/// whenever any layer, node or spot moves.
fn local_fingerprint_of(rest: u64, graph: u64) -> u64 {
    let mut hash = Fnv::new("local-v1");
    hash.u64("rest", rest);
    hash.u64("graph", graph);
    hash.0
}

fn portrait_fingerprint(adj: &AdjustmentSet) -> u64 {
    let mut hash = Fnv::new("portrait-v1");
    hash.json("retouch", &adj.retouch);
    hash.json("extra", &adj.extra);
    hash.0
}

/// Fingerprints of the parameter groups that run before automatic dust
/// detection reads the developed frame.
///
/// `Local` (spots/layers/nodes) and `Portrait` (retouch/extras) are applied
/// after dust consumes the frame, so they must not invalidate its cache.
/// Everything else — `Characterization` (base/style), `Geometry` (geometry and
/// the whole lens group, including LCC and diffraction), `Develop` (white
/// balance through film grain) and `Detail` (sharpening/NR/moire) — changes the
/// pixels the model sees. The schema test above assigns every top-level
/// `AdjustmentSet` field to exactly one group, so a new field cannot silently
/// escape this key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DustInputFingerprint {
    pub characterization: u64,
    pub geometry: u64,
    pub develop: u64,
    pub detail: u64,
}

pub(crate) fn dust_input_fingerprint(adj: &AdjustmentSet) -> DustInputFingerprint {
    DustInputFingerprint {
        characterization: characterization_fingerprint(adj),
        geometry: geometry_fingerprint(adj),
        develop: develop_fingerprint(adj),
        detail: detail_fingerprint(adj),
    }
}

/// Package version plus an optional build id, so a rebuilt engine never reuses
/// frames rendered by the previous build.
fn build_fingerprint() -> u64 {
    static FINGERPRINT: OnceLock<u64> = OnceLock::new();
    *FINGERPRINT.get_or_init(|| {
        let id = option_env!("AURORA_BUILD_ID").unwrap_or(env!("CARGO_PKG_VERSION"));
        build_fingerprint_of(id, RENDER_VERSION)
    })
}

fn build_fingerprint_of(id: &str, render_version: u32) -> u64 {
    let bytes = format!("{id}\0{render_version}");
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests that touch the process-wide source epoch run one at a time: a bump
    /// in one test must not clear a cache another test is inspecting.
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        test_source_epoch_lock()
    }

    fn key_for(image: u64, exposure: f32) -> PreviewKey {
        let mut adj = AdjustmentSet::default();
        adj.exposure.exposure = exposure;
        key(ImageId(image), &adj, NormRect::FULL, 1400, false, None, false)
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum ParamGroup {
        Characterization,
        Geometry,
        Develop,
        Detail,
        Local,
        Portrait,
    }

    impl ParamGroup {
        const ALL: [Self; 6] = [
            Self::Characterization,
            Self::Geometry,
            Self::Develop,
            Self::Detail,
            Self::Local,
            Self::Portrait,
        ];
    }

    /// Every serialized top-level field of [`AdjustmentSet`], exactly once.
    /// The schema test below keeps this list honest when a field is added.
    const GROUPED_FIELDS: &[(&str, ParamGroup)] = &[
        ("base", ParamGroup::Characterization),
        ("white_balance", ParamGroup::Develop),
        ("exposure", ParamGroup::Develop),
        ("hdr", ParamGroup::Develop),
        ("levels", ParamGroup::Develop),
        ("curves", ParamGroup::Develop),
        ("clarity", ParamGroup::Develop),
        ("dehaze", ParamGroup::Develop),
        ("vignetting", ParamGroup::Develop),
        ("color_balance", ParamGroup::Develop),
        ("color_balance_gains", ParamGroup::Develop),
        ("color_editor", ParamGroup::Develop),
        ("black_white", ParamGroup::Develop),
        ("sharpening", ParamGroup::Detail),
        ("noise_reduction", ParamGroup::Detail),
        ("film_grain", ParamGroup::Develop),
        ("moire", ParamGroup::Detail),
        ("lens", ParamGroup::Geometry),
        ("geometry", ParamGroup::Geometry),
        ("spots", ParamGroup::Local),
        ("retouch", ParamGroup::Portrait),
        ("layers", ParamGroup::Local),
        ("nodes", ParamGroup::Local),
        ("style", ParamGroup::Characterization),
        ("extra", ParamGroup::Portrait),
    ];

    /// Top-level fields hidden from the document by `serde(skip)`; they are
    /// still assigned to a group and hashed by hand where the value is known.
    const NON_SERIALIZED_FIELDS: &[(&str, ParamGroup)] =
        &[("portrait_producer_inputs", ParamGroup::Portrait)];

    fn group_of(fingerprint: ParamsFingerprint, group: ParamGroup) -> u64 {
        match group {
            ParamGroup::Characterization => fingerprint.characterization,
            ParamGroup::Geometry => fingerprint.geometry,
            ParamGroup::Develop => fingerprint.develop,
            ParamGroup::Detail => fingerprint.detail,
            ParamGroup::Local => fingerprint.local,
            ParamGroup::Portrait => fingerprint.portrait,
        }
    }

    fn changed_groups(before: ParamsFingerprint, after: ParamsFingerprint) -> Vec<ParamGroup> {
        ParamGroup::ALL
            .iter()
            .copied()
            .filter(|group| group_of(before, *group) != group_of(after, *group))
            .collect()
    }

    fn key_of(adj: &AdjustmentSet) -> PreviewKey {
        key(ImageId(1), adj, NormRect::FULL, 1400, false, None, false)
    }

    /// A set where every grouped field differs from `Default`, so clearing one
    /// field through the serialized schema must change exactly its group.
    fn populated() -> AdjustmentSet {
        let mut adj = AdjustmentSet::default();
        adj.base.icc_profile = Some("Probe ICC".into());
        adj.base.curve = crate::ToneCurvePreset::Linear;
        adj.base.processing_method = crate::TechnicalProcessingMethod::Scientific;
        adj.base.film_curve = Some("Probe.fcrv".into());
        adj.base.engine_version = "Probe 9".into();
        adj.white_balance.mode = crate::WhiteBalanceMode::Custom;
        adj.white_balance.kelvin = 5000.0;
        adj.white_balance.tint = 7.0;
        adj.white_balance.skin_tone = Some("Skin".into());
        adj.white_balance.gains = Some([1.1, 1.0, 0.9]);
        adj.exposure.exposure = 0.5;
        adj.exposure.contrast = 3.0;
        adj.exposure.brightness = 4.0;
        adj.exposure.saturation = 5.0;
        adj.hdr.highlights = 6.0;
        adj.hdr.shadows = 7.0;
        adj.hdr.whites = 8.0;
        adj.hdr.blacks = 9.0;
        adj.levels.rgb.input_black = 0.05;
        adj.levels.rgb.gamma = 1.2;
        adj.levels.red.input_white = 0.9;
        adj.curves.rgb.points =
            vec![crate::CurvePoint::new(0.0, 0.0), crate::CurvePoint::new(0.5, 0.6), crate::CurvePoint::new(1.0, 1.0)];
        adj.curves.red.points = vec![crate::CurvePoint::new(0.0, 0.1), crate::CurvePoint::new(1.0, 0.9)];
        adj.clarity.method = crate::ClarityMethod::Punch;
        adj.clarity.clarity = 12.0;
        adj.dehaze.amount = 12.0;
        adj.dehaze.shadow_tone = [0.1, 0.2, 0.3];
        adj.dehaze.level_low = [0.01, 0.02, 0.03];
        adj.dehaze.level_high = [0.9, 0.91, 0.92];
        adj.vignetting.amount = 1.5;
        adj.vignetting.method = crate::VignetteMethod::Circular;
        adj.vignetting.center = crate::VignetteCenter::Custom;
        adj.vignetting.x = 10.0;
        adj.vignetting.y = 20.0;
        adj.color_balance.master.hue = 20.0;
        adj.color_balance_gains = Some(crate::ColorBalanceGains {
            master: [0.9, 1.0, 1.1],
            ..Default::default()
        });
        adj.color_editor.basic[0].hue = 4.0;
        adj.color_editor.bands[0].center_hue = 2.0;
        adj.color_editor.global.smoothness = 3.0;
        adj.black_white.enabled = true;
        adj.black_white.sensitivity[0] = 5.0;
        adj.sharpening.amount = 200.0;
        adj.noise_reduction.luminance = 61.0;
        adj.film_grain.kind = crate::GrainKind::SoftGrain;
        adj.film_grain.impact = 30.0;
        adj.film_grain.granularity = 40.0;
        adj.film_grain.density = 20.0;
        adj.moire.amount = 5.0;
        adj.lens.profile = Some("Probe Lens".into());
        adj.lens.use_profile = true;
        adj.lens.distortion = 3.0;
        adj.lens.light_falloff = 4.0;
        adj.lens.uniform_light = 5.0;
        adj.lens.uniform_light_enabled = true;
        adj.lens.chromatic_aberration = true;
        adj.lens.purple_fringing = 6.0;
        adj.lens.diffraction_enabled = true;
        adj.lens.aperture = 5.6;
        adj.lens.shift_x = 1.0;
        adj.lens.tilt = 2.0;
        adj.lens.mount = Some("Probe Mount".into());
        adj.geometry.rotation = 1.5;
        adj.geometry.orientation = 1;
        adj.geometry.flip_horizontal = true;
        adj.geometry.crop_outside_image = true;
        adj.geometry.crop = crate::NormRect { x: 0.1, y: 0.1, width: 0.8, height: 0.8 };
        adj.geometry.canvas = [100.0, 50.0];
        adj.geometry.aspect = crate::AspectRatio::R4x5;
        adj.geometry.portrait = true;
        adj.geometry.keystone.vertical = 5.0;
        adj.geometry.optical_center = [3.0, 4.0];
        adj.geometry.distortion.v = vec![1.0, 2.0];
        adj.geometry.distortion.x = vec![1.1, 2.2];
        adj.geometry.distortion.m = vec![0.5, 0.6];
        adj.spots.push(crate::Spot {
            id: 1,
            kind: crate::SpotKind::Spot,
            x: 0.5,
            y: 0.5,
            radius: 0.1,
        });
        adj.retouch.push(crate::RetouchFace { skin_amount: 10.0, ..Default::default() });
        adj.layers.push(crate::Layer {
            id: 7,
            name: "Probe Layer".into(),
            enabled: true,
            opacity: 80.0,
            mask: crate::Mask {
                shapes: vec![crate::MaskShape::Linear(crate::LinearGradient {
                    op: crate::MaskOp::Add,
                    start: [0.2, 0.3],
                    end: [0.8, 0.7],
                })],
                ..Default::default()
            },
            ..Default::default()
        });
        adj.nodes = Some(crate::NodeGraph {
            nodes: vec![
                crate::GraphNode { id: crate::NodeGraph::INPUT, kind: crate::NodeKind::Input, pos: [0.0, 0.0] },
                crate::GraphNode { id: 7, kind: crate::NodeKind::Layer, pos: [10.0, 20.0] },
                crate::GraphNode { id: crate::NodeGraph::OUTPUT, kind: crate::NodeKind::Output, pos: [30.0, 0.0] },
            ],
            links: vec![
                crate::NodeLink { from: crate::NodeGraph::INPUT, to: 7, input: 0 },
                crate::NodeLink { from: 7, to: crate::NodeGraph::OUTPUT, input: 0 },
            ],
        });
        adj.style = Some("Probe Style".into());
        adj.extra.insert("pc:21093".into(), 1.0);
        adj
    }

    /// The declared field list must match the serialized schema. A field added
    /// to `AdjustmentSet` without a group fails here.
    #[test]
    fn declared_fields_match_the_serialized_schema() {
        let _serial = serial();
        let value = serde_json::to_value(populated()).unwrap();
        let mut keys: Vec<String> = value.as_object().unwrap().keys().cloned().collect();
        keys.sort_unstable();
        let mut expected: Vec<String> =
            GROUPED_FIELDS.iter().map(|(field, _)| (*field).to_owned()).collect();
        expected.sort_unstable();
        assert_eq!(keys, expected, "GROUPED_FIELDS must list every serialized top-level field");

        let mut all: Vec<&str> = GROUPED_FIELDS
            .iter()
            .chain(NON_SERIALIZED_FIELDS.iter())
            .map(|(field, _)| *field)
            .collect();
        all.sort_unstable();
        let count = all.len();
        all.dedup();
        assert_eq!(all.len(), count, "every field must belong to exactly one group");
    }

    /// Clearing one top-level field to its default through the serialized
    /// schema must change exactly the group the table assigns it to.
    #[test]
    fn every_serialized_field_changes_exactly_its_group() {
        let _serial = serial();
        let full = populated();
        let full_value = serde_json::to_value(&full).unwrap();
        let default_value = serde_json::to_value(&AdjustmentSet::default()).unwrap();
        let roundtripped: AdjustmentSet = serde_json::from_value(full_value.clone()).unwrap();
        let before = ParamsFingerprint::of(&roundtripped);
        for (field, group) in GROUPED_FIELDS {
            let mut value = full_value.clone();
            let object = value.as_object_mut().unwrap();
            match default_value.get(*field) {
                Some(default_field) => {
                    object.insert((*field).to_owned(), default_field.clone());
                }
                None => {
                    object.remove(*field);
                }
            }
            let cleared: AdjustmentSet = serde_json::from_value(value).unwrap();
            let after = ParamsFingerprint::of(&cleared);
            assert_eq!(changed_groups(before, after), vec![*group], "field {field}");
        }
    }

    fn assert_ui_only(
        before: &ParamsFingerprint,
        serialized: &serde_json::Value,
        changed: &AdjustmentSet,
    ) {
        assert_eq!(
            before,
            &ParamsFingerprint::of(changed),
            "a non-pixel field changed the pixel fingerprint"
        );
        assert_ne!(
            serialized,
            &serde_json::to_value(changed).unwrap(),
            "the field must still be serialized, or this test proves nothing"
        );
    }

    #[test]
    fn ui_only_fields_are_not_hashed() {
        let _serial = serial();
        let base = populated();
        let before = ParamsFingerprint::of(&base);
        let serialized = serde_json::to_value(&base).unwrap();

        let mut changed = base.clone();
        changed.base.engine_version = "Probe 10".into();
        assert_ui_only(&before, &serialized, &changed);

        let mut changed = base.clone();
        changed.white_balance.skin_tone = Some("Deep".into());
        assert_ui_only(&before, &serialized, &changed);

        let mut changed = base.clone();
        changed.layers[0].name = "Renamed".into();
        assert_ui_only(&before, &serialized, &changed);

        let mut changed = base.clone();
        changed.layers[0].mask_uuid = Some("mask-uuid".into());
        assert_ui_only(&before, &serialized, &changed);

        let mut changed = base.clone();
        changed.layers[0].style = Some(crate::LayerStyle {
            uuid: "style-uuid".into(),
            source: Some("Styles".into()),
            modified: true,
        });
        assert_ui_only(&before, &serialized, &changed);

        let mut changed = base.clone();
        changed.nodes.as_mut().unwrap().nodes[1].pos = [999.0, -999.0];
        assert_ui_only(&before, &serialized, &changed);
    }

    /// `serde(skip)` fields do reach pixels; the manual hash must cover them.
    #[test]
    fn nested_serde_skipped_pixel_inputs_are_hashed() {
        let _serial = serial();
        let base = populated();
        let before = ParamsFingerprint::of(&base);

        let mut changed = base.clone();
        changed.geometry.distortion.v.push(3.0);
        changed.geometry.distortion.x.push(3.3);
        changed.geometry.distortion.m.push(0.7);
        assert_eq!(
            changed_groups(before, ParamsFingerprint::of(&changed)),
            vec![ParamGroup::Geometry]
        );

        let mut changed = base.clone();
        changed.geometry.optical_center = [11.0, 12.0];
        assert_eq!(
            changed_groups(before, ParamsFingerprint::of(&changed)),
            vec![ParamGroup::Geometry]
        );

        let mut changed = base.clone();
        changed.lens.uniform_light_grid = Some(Arc::new(crate::dust_lcc::LightFalloff {
            header: vec![1, 2],
            columns: 2,
            rows: 1,
            nodes: vec![crate::dust_lcc::LightFalloffNode { luminance: 7, dx: 1, dy: -1 }],
        }));
        assert_eq!(
            changed_groups(before, ParamsFingerprint::of(&changed)),
            vec![ParamGroup::Geometry]
        );
    }

    /// The DAG lane report listed `select_point` as non-pixel. At this HEAD a
    /// heal/clone layer with no cover patch uses it as the repair source
    /// (`mock/process.rs:4612`), so it must stay in the fingerprint, as must an
    /// AI mask's cache key, which selects the stored segmentation.
    #[test]
    fn pixel_bearing_layer_metadata_stays_in_the_fingerprint() {
        let _serial = serial();
        let mut base = populated();
        base.layers.push(crate::Layer {
            id: 9,
            kind: crate::LayerKind::Heal,
            select_point: Some([100.0, 200.0]),
            ..Default::default()
        });
        let before = ParamsFingerprint::of(&base);
        let mut changed = base.clone();
        changed.layers.last_mut().unwrap().select_point = Some([101.0, 200.0]);
        assert_eq!(
            changed_groups(before, ParamsFingerprint::of(&changed)),
            vec![ParamGroup::Local]
        );

        let mut base = populated();
        base.layers.push(crate::Layer {
            id: 11,
            mask: crate::Mask {
                shapes: vec![crate::MaskShape::Ai(crate::AiMask {
                    op: crate::MaskOp::Add,
                    kind: crate::AiMaskKind::Subject,
                    cache_key: Some("ai-a".into()),
                })],
                ..Default::default()
            },
            ..Default::default()
        });
        let before = ParamsFingerprint::of(&base);
        let mut changed = base.clone();
        changed.layers.last_mut().unwrap().mask.shapes =
            vec![crate::MaskShape::Ai(crate::AiMask {
                op: crate::MaskOp::Add,
                kind: crate::AiMaskKind::Subject,
                cache_key: Some("ai-b".into()),
            })];
        assert_eq!(
            changed_groups(before, ParamsFingerprint::of(&changed)),
            vec![ParamGroup::Local]
        );
    }

    #[test]
    fn moving_a_node_keeps_the_preview_key() {
        let _serial = serial();
        let mut adj = populated();
        let before = key_of(&adj);
        adj.nodes.as_mut().unwrap().nodes[1].pos = [1234.0, 5678.0];
        assert_eq!(before, key_of(&adj), "node-editor layout must not invalidate a frame");
    }

    #[test]
    fn every_semantic_group_changes_the_preview_key() {
        let _serial = serial();
        let base = populated();
        let before = key_of(&base);

        let mut changed = base.clone();
        changed.base.icc_profile = Some("Other ICC".into());
        assert_ne!(before, key_of(&changed), "characterization group");

        let mut changed = base.clone();
        changed.geometry.rotation += 0.5;
        assert_ne!(before, key_of(&changed), "geometry group");

        let mut changed = base.clone();
        changed.exposure.exposure += 0.25;
        assert_ne!(before, key_of(&changed), "develop group");

        let mut changed = base.clone();
        changed.sharpening.amount += 10.0;
        assert_ne!(before, key_of(&changed), "detail group");

        let mut changed = base.clone();
        changed.layers[0].opacity = 42.0;
        assert_ne!(before, key_of(&changed), "local group");

        let mut changed = base.clone();
        changed.retouch[0].skin_amount = 20.0;
        assert_ne!(before, key_of(&changed), "portrait group");
    }

    fn frame(tag: u8) -> CachedFrame {
        CachedFrame {
            width: 2,
            height: 2,
            pixels: Arc::new(vec![tag; 16]),
            histogram: Histogram::default(),
        }
    }

    #[test]
    fn one_parameter_change_misses() {
        let _serial = serial();
        assert_eq!(key_for(1, 0.5), key_for(1, 0.5));
        assert_ne!(key_for(1, 0.5), key_for(1, 0.500_001));
        assert_ne!(key_for(1, 0.5), key_for(2, 0.5));
    }

    #[test]
    fn lru_evicts_the_least_recently_used() {
        let _serial = serial();
        let mut cache = PreviewCache::new(2);
        cache.insert(key_for(1, 0.0), frame(1));
        cache.insert(key_for(2, 0.0), frame(2));
        // Touch the first one so the second becomes the eviction candidate.
        assert!(cache.get(key_for(1, 0.0)).is_some());
        cache.insert(key_for(3, 0.0), frame(3));
        assert_eq!(cache.len(), 2);
        assert!(cache.get(key_for(2, 0.0)).is_none());
        assert!(cache.get(key_for(1, 0.0)).is_some());
        assert!(cache.get(key_for(3, 0.0)).is_some());
    }

    #[test]
    fn the_byte_budget_evicts_before_the_entry_limit() {
        let _serial = serial();
        let mut cache = PreviewCache::with_budget(8, 20);
        cache.insert(key_for(1, 0.0), frame(1));
        cache.insert(key_for(2, 0.0), frame(2));
        assert_eq!(cache.len(), 1);
        assert!(cache.get(key_for(1, 0.0)).is_none());
        assert!(cache.get(key_for(2, 0.0)).is_some());
    }

    #[test]
    fn the_same_build_fingerprint_hits() {
        let _serial = serial();
        let mut cache = PreviewCache::new(4);
        cache.insert(key_for(1, 0.0), frame(1));
        assert_eq!(key_for(1, 0.0).build, build_fingerprint());
        assert!(cache.get(key_for(1, 0.0)).is_some());
    }

    #[test]
    fn a_different_build_fingerprint_misses() {
        let _serial = serial();
        let mut cache = PreviewCache::new(4);
        let cached = key_for(1, 0.0);
        cache.insert(cached, frame(1));
        let other_build = PreviewKey { build: build_fingerprint_of("9.9.9", RENDER_VERSION), ..cached };
        assert_ne!(other_build.build, cached.build);
        assert!(cache.get(other_build).is_none());
    }

    #[test]
    fn a_source_identity_change_drops_rendered_frames() {
        let _serial = serial();
        let mut cache = PreviewCache::new(4);
        let key = key_for(1, 0.0);
        cache.insert(key, frame(1));
        assert!(cache.get(key).is_some());
        invalidate_source_identity();
        assert!(
            cache.get(key).is_none(),
            "a frame rendered from replaced source bytes must not be served"
        );
        cache.insert(key, frame(2));
        assert_eq!(cache.get(key).unwrap().pixels[0], 2);
    }

    #[test]
    fn the_build_fingerprint_tracks_the_version_text() {
        let _serial = serial();
        assert_eq!(build_fingerprint_of("1.2.3", 1), build_fingerprint_of("1.2.3", 1));
        assert_ne!(build_fingerprint_of("1.2.3", 1), build_fingerprint_of("1.2.4", 1));
        assert_ne!(build_fingerprint_of("1.2.3", 1), build_fingerprint_of("1.2.3", 2));
    }

    #[test]
    fn clear_drops_every_frame() {
        let _serial = serial();
        let mut cache = PreviewCache::new(4);
        cache.insert(key_for(1, 0.0), frame(1));
        cache.insert(key_for(2, 0.0), frame(2));
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.get(key_for(1, 0.0)).is_none());
        assert!(cache.get(key_for(2, 0.0)).is_none());
    }

    // ── Graph-dirty reuse gate ──────────────────────────────────────────

    fn gate_layer(id: u64, exposure: f32) -> Layer {
        let mut layer = Layer { id, ..Default::default() };
        layer.adjustments.exposure.exposure = exposure;
        layer
    }

    fn gate_graph(links: &[(u64, u64, u32)], ids: &[u64]) -> NodeGraph {
        let mut nodes = vec![
            crate::GraphNode { id: NodeGraph::INPUT, kind: NodeKind::Input, pos: [0.0; 2] },
            crate::GraphNode { id: NodeGraph::OUTPUT, kind: NodeKind::Output, pos: [0.0; 2] },
        ];
        nodes.extend(
            ids.iter().map(|id| crate::GraphNode { id: *id, kind: NodeKind::Layer, pos: [0.0; 2] }),
        );
        NodeGraph {
            nodes,
            links: links.iter().map(|&(from, to, input)| NodeLink { from, to, input }).collect(),
        }
    }

    /// The graph the render path uses, so a row without a node still renders.
    fn repaired(adj: &AdjustmentSet) -> NodeGraph {
        adj.nodes.clone().unwrap_or_default().repaired(&adj.layers)
    }

    fn gate_set() -> AdjustmentSet {
        let mut adj = AdjustmentSet::default();
        adj.layers = vec![gate_layer(1, 0.0), gate_layer(2, 0.0), gate_layer(3, 0.0)];
        // 3 hangs off 1 but never reaches the output.
        adj.nodes = Some(gate_graph(
            &[
                (NodeGraph::INPUT, 1, 0),
                (1, 2, 0),
                (1, 3, 0),
                (2, NodeGraph::OUTPUT, 0),
            ],
            &[1, 2, 3],
        ));
        adj
    }

    #[test]
    fn graph_gate_reuses_a_frame_after_a_pruned_layer_edit() {
        let _serial = serial();
        let adj = gate_set();
        let key = key_of(&adj);
        let mut gate = GraphGate::new();
        gate.record(key, key, repaired(&adj), adj.layers.clone());

        let mut edited = adj.clone();
        edited.layers[2].adjustments.exposure.exposure = 2.0;
        let new_key = key_of(&edited);
        assert_ne!(key, new_key, "the frame key still sees the pruned edit");
        assert_eq!(
            gate.reuse(&new_key, &repaired(&edited), &edited.layers),
            Some(key),
            "the pruned edit cannot change a pixel, so the earlier frame is reused"
        );
    }

    #[test]
    fn graph_gate_rejects_a_live_layer_edit() {
        let _serial = serial();
        let adj = gate_set();
        let key = key_of(&adj);
        let mut gate = GraphGate::new();
        gate.record(key, key, repaired(&adj), adj.layers.clone());

        let mut edited = adj.clone();
        edited.layers[1].adjustments.exposure.exposure = 1.0;
        assert_eq!(gate.reuse(&key_of(&edited), &repaired(&edited), &edited.layers), None);
    }

    #[test]
    fn graph_gate_rejects_edits_outside_the_graph() {
        let _serial = serial();
        let adj = gate_set();
        let key = key_of(&adj);
        let mut gate = GraphGate::new();
        gate.record(key, key, repaired(&adj), adj.layers.clone());

        // A develop change keeps the graph identical but still changes pixels.
        let mut exposed = adj.clone();
        exposed.exposure.exposure = 0.6;
        assert_eq!(gate.reuse(&key_of(&exposed), &repaired(&exposed), &exposed.layers), None);

        // A spot is local but not part of the graph, so it is not reusable.
        let mut spotted = adj.clone();
        spotted.spots.push(crate::Spot {
            id: 1,
            kind: crate::SpotKind::Spot,
            x: 0.5,
            y: 0.5,
            radius: 0.1,
        });
        assert_eq!(gate.reuse(&key_of(&spotted), &repaired(&spotted), &spotted.layers), None);

        // A different image must never reuse.
        let other_image = super::key(ImageId(2), &adj, NormRect::FULL, 1400, false, None, false);
        assert_eq!(gate.reuse(&other_image, &repaired(&adj), &adj.layers), None);
    }

    #[test]
    fn graph_gate_rejects_a_deleted_live_layer() {
        let _serial = serial();
        let adj = gate_set();
        let key = key_of(&adj);
        let mut gate = GraphGate::new();
        gate.record(key, key, repaired(&adj), adj.layers.clone());

        let mut edited = adj.clone();
        edited.layers.retain(|layer| layer.id != 2);
        // Repair reconnects 1 straight to the output; the removed layer's
        // contribution is gone even though no current layer replaced it.
        assert_eq!(gate.reuse(&key_of(&edited), &repaired(&edited), &edited.layers), None);
    }
}
