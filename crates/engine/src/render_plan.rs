//! Compiled render graph: the execution view of a saved node graph.
//!
//! [`layers::NodeGraph`] is the editable topology the user builds and saves in
//! `AdjustmentSet::nodes`. This module compiles it into a [`RenderPlan`]: which
//! nodes the output depends on, in what deterministic order they run, which
//! nodes a document change dirties, and which cache identity a node result
//! carries. It never mutates the saved document and never touches pixels.
//!
//! # Saved topology versus execution
//!
//! The document is the source of truth. A plan is derived and may be thrown
//! away at any time; compiling the same graph twice is required to produce the
//! same plan. The plan deliberately does **not** repair the document: callers
//! that want row/node repair use [`layers::NodeGraph::repaired`] or
//! [`RenderPlan::compile_document`], which applies the same repair the render
//! and export paths use.
//!
//! [`RenderPlan::compile`] delegates evaluation order, layer pruning and the
//! dangling-link/mixer-gap warnings to [`layers::NodeGraph::plan`] rather than
//! keeping a second graph walk. It adds only the typed dependency map/kinds
//! and two reports `plan()` does not make: a serial node with no input, and
//! detached non-layer nodes in `pruned`.
//!
//! # Semantics
//!
//! * `order` contains every node the output reaches (including the
//!   [`NodeGraph::INPUT`] and [`NodeGraph::OUTPUT`] markers, which are part of
//!   the evaluation contract). Every node follows everything that feeds it.
//!   Input indices decide the tie-break for mixer branches, matching the
//!   existing renderer: a higher input number is stacked on top, so it is
//!   evaluated later.
//! * `pruned` contains the nodes the renderer would not execute — detached
//!   branches, orphan mixers, detached cycles. Pruning is reported, never
//!   silent. Layer pruning matches [`layers::NodeGraph::plan`] exactly; this
//!   module only adds detached non-layer nodes to the report. A `Layer` the
//!   output reaches but the input marker does not is listed in `pruned` even
//!   though it appears in `order`, because the renderer skips it. The `INPUT`
//!   and `OUTPUT` markers are structural and are never listed as pruned.
//! * A cycle the output depends on is [`GraphError::Cycle`], naming the loop
//!   members. A detached cycle is pruned: it cannot change a rendered frame.
//! * A missing input is a [`GraphWarning`], not a refusal. A serial node with
//!   no input and a mixer with a gap both still compile; the renderer decides
//!   what that means. Dangling links are dropped and reported.
//!
//! # Dirty propagation
//!
//! [`dirty_nodes`] compares only the parameter subset each node consumes:
//!
//! * `Input` consumes the background/global adjustments (everything in
//!   [`AdjustmentSet`] except `layers`, `nodes` and the ephemeral producer
//!   inputs). A global edit therefore dirties the input and every consumer of
//!   it.
//! * A `Layer` node consumes its matching [`Layer`] row: kind, enabled state,
//!   opacity, mask, local adjustments and cover arrows. Editing another row —
//!   or renaming this one — does not dirty it.
//! * Mixers consume no parameters of their own. They are dirtied only through
//!   their inputs.
//!
//! Crop/rotation/keystone (`geometry`) and lens remapping (`lens`) are
//! *global-geometry*: they change the pixel grid every node samples, so a
//! change sets [`DirtySet::global`] and invalidates **every** node in the plan,
//! not only the locally edited subgraph. Likewise a topology change (node ids /
//! kinds / links; node positions are layout only) invalidates the whole plan.
//! The caller must compile the `next` plan before asking for a dirty set.
//!
//! The parameter fingerprint used for dirty comparison is the same byte
//! encoding used by [`RenderPlan::cache_key`], so "consumed subset" cannot mean
//! two different things in two places. Fields marked `#[serde(skip)]` on
//! adjustments (for example a hydrated LCC grid or lens distortion curve) are
//! runtime inputs, not document edits; they belong in the caller's revision or
//! source identity, not in this comparison.
//!
//! # Cache identity
//!
//! [`NodeCacheKey`] is a stable FNV-1a hash of: a version tag, the source
//! identity, the node id, the caller's node revision/fingerprint, the node's
//! parameter subset (serialized with `serde_json`, whose struct field order is
//! fixed), the ROI, the color space, the precision, the algorithm version, and
//! a backend-equivalence tag. Backends with different equivalence tags never
//! share keys; two backends explicitly declared equivalent (same
//! [`BackendTag::equivalence`]) do.
//!
//! # Limits
//!
//! * Node positions are layout, not pixels, and never dirty a node or change a
//!   key.
//! * Mask `Copy { from_layer }` is a *data* dependency that is not expressed as
//!   a [`NodeLink`]. It is out of scope here: editing the source layer dirties
//!   the source node, not the copying node. Callers that start using copy masks
//!   as render inputs need an explicit edge or an additional seed.
//! * The plan is a pure compile. Execution scheduling, GPU/CPU dispatch and
//!   pixel work stay in the renderer.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::{
    AdjustmentSet, BaseCharacteristics, BlackWhite, Clarity, ColorBalance,
    ColorBalanceGains, ColorEditor, CoverPatch, Curves, Dehaze, Exposure, FilmGrain,
    Geometry, GraphError, GraphWarning, HighDynamicRange, Layer, LayerKind, LensCorrection,
    Levels, LocalAdjustments, Mask, Moire, NodeGraph, NodeKind, NodeLink, NoiseReduction,
    RetouchFace, Sharpening, Spot, Vignetting, WhiteBalance,
};

/// A node id in the saved graph.
pub type NodeId = u64;

// ───────────────────────────── plan ─────────────────────────────

/// The compiled execution view of one [`NodeGraph`].
///
/// See the module docs for the exact ordering, pruning and warning rules.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderPlan {
    /// Every node the output depends on, in evaluation order: a node always
    /// follows everything that feeds it. Includes [`NodeGraph::INPUT`] and
    /// [`NodeGraph::OUTPUT`] when the graph declares them.
    pub order: Vec<NodeId>,
    /// Real nodes the output does not reach, sorted by id. Reported, not
    /// silently dropped.
    pub pruned: Vec<NodeId>,
    /// Structural details that still compile. Duplicates are not emitted.
    pub warnings: Vec<GraphWarning>,
    /// Direct dependencies of each node in `order`, in input order.
    inputs: BTreeMap<NodeId, Vec<NodeId>>,
    /// Node kind, for every node the graph declares.
    kinds: BTreeMap<NodeId, NodeKind>,
}

impl RenderPlan {
    /// Compiles `graph` exactly as saved. Does not repair missing rows/nodes
    /// and does not read any layer data.
    ///
    /// Ordering, layer pruning and the dangling/mixer warnings are delegated
    /// to [`NodeGraph::plan`], the one implementation of graph reachability
    /// and evaluation order. This module only derives the typed dependency map
    /// and kinds that dirty tracking and cache keys need, and adds two reports
    /// `plan()` does not make:
    ///
    /// * a serial [`NodeKind::Layer`] on the output path with no input is a
    ///   [`GraphWarning::MissingInput`] (the renderer's `plan()` only checks
    ///   mixer gaps);
    /// * a detached node that is not a `Layer` (a mixer, for example) is added
    ///   to [`Self::pruned`], because `plan().pruned` lists only layer rows.
    ///
    /// Consequently `Self::order` is byte-for-byte `plan().order`, and every
    /// id in `plan().pruned` is in `Self::pruned`. As in `plan()`, a `Layer`
    /// the output reaches but the input marker does not is still listed as
    /// pruned: the renderer skips it, so the execution view reports that.
    pub fn compile(graph: &NodeGraph) -> Result<Self, GraphError> {
        let base = graph.plan()?;

        let mut kinds: BTreeMap<NodeId, NodeKind> = BTreeMap::new();
        for node in &graph.nodes {
            // A duplicate id is malformed; the first declaration wins so the
            // plan stays deterministic instead of depending on scan order.
            kinds.entry(node.id).or_insert(node.kind);
        }

        // Same validity rule as `plan()`: an edge with an endpoint the graph
        // does not contain is dropped. Sort by (to, input, from) so the
        // dependency map never depends on stored edge order; when two edges
        // claim the same input, the smaller source id wins (stable, though
        // arbitrary).
        let ids: BTreeSet<NodeId> = graph.nodes.iter().map(|node| node.id).collect();
        let mut links: Vec<NodeLink> = graph
            .links
            .iter()
            .copied()
            .filter(|link| ids.contains(&link.from) && ids.contains(&link.to))
            .collect();
        links.sort_by_key(|link| (link.to, link.input, link.from));
        links.dedup_by_key(|link| (link.to, link.input));

        let mut direct: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
        for link in &links {
            direct.entry(link.to).or_default().push(link.from);
        }

        let mut warnings = base.warnings.clone();
        for id in &base.order {
            if kinds.get(id) == Some(&NodeKind::Layer)
                && direct.get(id).is_none_or(Vec::is_empty)
            {
                warnings.push(GraphWarning::MissingInput { node: *id, input: 0 });
            }
        }

        let mut pruned = base.pruned.clone();
        for node in &graph.nodes {
            if node.id == NodeGraph::INPUT || node.id == NodeGraph::OUTPUT {
                continue;
            }
            if !base.order.contains(&node.id) && !pruned.contains(&node.id) {
                pruned.push(node.id);
            }
        }
        pruned.sort_unstable();
        pruned.dedup();

        let inputs = base
            .order
            .iter()
            .map(|id| (*id, direct.get(id).cloned().unwrap_or_default()))
            .collect();

        Ok(Self { order: base.order, pruned, warnings, inputs, kinds })
    }

    /// Compiles a saved document: repairs the stored graph against its layer
    /// rows exactly as the render/export path does, then compiles it. Reports
    /// layer rows without nodes and nodes without rows as warnings.
    ///
    /// A document with no stored graph is the legacy serial chain over
    /// `document.layers`, the same shape [`crate::layered`] renders.
    pub fn compile_document(document: &AdjustmentSet) -> Result<Self, GraphError> {
        let graph = document.nodes.clone().unwrap_or_default().repaired(&document.layers);
        let mut plan = Self::compile(&graph)?;
        if let Some(stored) = &document.nodes {
            for node in &stored.nodes {
                if node.kind == NodeKind::Layer
                    && !document.layers.iter().any(|row| row.id == node.id)
                {
                    plan.warnings.push(GraphWarning::LayerWithoutRow { id: node.id });
                }
            }
            for row in &document.layers {
                if stored.node(row.id).is_none() {
                    plan.warnings.push(GraphWarning::RowWithoutNode { id: row.id });
                }
            }
        }
        Ok(plan)
    }

    /// Direct dependencies of `node`, in input order. Empty for the input
    /// marker, pruned nodes and unknown ids.
    pub fn dependencies(&self, node: NodeId) -> &[NodeId] {
        self.inputs.get(&node).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Every node that directly consumes `node`, in evaluation order.
    pub fn consumers(&self, node: NodeId) -> Vec<NodeId> {
        self.order
            .iter()
            .copied()
            .filter(|id| self.inputs.get(id).is_some_and(|inputs| inputs.contains(&node)))
            .collect()
    }

    /// Whether the output depends on `node` in this plan.
    pub fn contains(&self, node: NodeId) -> bool {
        self.order.contains(&node)
    }

    /// The kind recorded for `node`, if the graph declares it.
    pub fn kind(&self, node: NodeId) -> Option<NodeKind> {
        self.kinds.get(&node).copied()
    }

    /// The cache key for one node result under `ctx`.
    ///
    /// `revision` is the caller's node revision/fingerprint: a document
    /// revision, a dependency-chain hash, or a fingerprint of runtime inputs
    /// that are not stored on the document. The node's persisted parameter
    /// subset is added here, so two documents that differ only in this node's
    /// parameters produce different keys.
    ///
    /// Returns `None` when the node is not in the plan, a `Layer` node has no
    /// row, or the parameter encoding fails (which should not happen for these
    /// types). Never panics on a missing row.
    pub fn cache_key(
        &self,
        document: &AdjustmentSet,
        node: NodeId,
        revision: u64,
        ctx: &CacheContext<'_>,
    ) -> Option<NodeCacheKey> {
        if !self.kinds.contains_key(&node) {
            return None;
        }
        let params = node_params(document, node, self.kind(node)?)?;
        Some(NodeCacheKey { node, revision, hash: ctx.hash(node, revision, &params) })
    }

    /// Records every node in the plan as executed. The sentinels
    /// [`NodeGraph::INPUT`] and [`NodeGraph::OUTPUT`] are recorded too, because
    /// they are part of `order`; tests that count pixel work should filter to
    /// `Layer` ids.
    pub fn record_all(&self, recorder: &mut ExecutionRecorder) {
        for id in &self.order {
            recorder.record(*id);
        }
    }

    /// Records the nodes a render would re-execute for `dirty`: the dirty
    /// closure in evaluation order. This is the scheduling hook that lets a
    /// test prove reuse without rendering.
    ///
    /// [`DirtySet::global`] records the whole plan. Otherwise only
    /// [`DirtySet::nodes`] are recorded; a dirty set is already the
    /// downstream-closed set produced by [`dirty_nodes`].
    pub fn record_runs(&self, dirty: &DirtySet, recorder: &mut ExecutionRecorder) {
        if dirty.global {
            self.record_all(recorder);
            return;
        }
        for id in &self.order {
            if dirty.contains(*id) {
                recorder.record(*id);
            }
        }
    }
}

// ───────────────────────────── dirty sets ─────────────────────────────

/// Nodes that must re-run for a document change, in evaluation order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DirtySet {
    /// Dirty nodes, downstream-closed, in evaluation order.
    pub nodes: Vec<NodeId>,
    /// A global-geometry or topology change invalidated the whole plan, not a
    /// local subgraph. `nodes` already contains every node in that case; this
    /// flag is for diagnostics and for [`RenderPlan::record_runs`].
    pub global: bool,
}

impl DirtySet {
    /// Every node in `plan` is dirty (first render, structural change).
    pub fn all(plan: &RenderPlan) -> Self {
        Self { nodes: plan.order.clone(), global: true }
    }

    pub fn contains(&self, node: NodeId) -> bool {
        self.nodes.contains(&node)
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// The nodes a change from `previous` to `next` dirties under `plan`.
///
/// `plan` must be the plan compiled for `next`. Only each node's consumed
/// parameter subset is compared; a sibling's edit leaves it untouched. The
/// result is downstream-closed: every consumer of a dirty node is dirty. A
/// global-geometry (crop/rotation/keystone/lens) or topology change dirties
/// every node in the plan and sets [`DirtySet::global`].
pub fn dirty_nodes(
    previous: &AdjustmentSet,
    next: &AdjustmentSet,
    plan: &RenderPlan,
) -> DirtySet {
    // A changed sampling grid or wiring invalidates every cached result, even
    // nodes whose own parameters did not move. Node positions are layout and
    // deliberately excluded.
    if global_geometry_changed(previous, next) || topology_changed(previous, next) {
        return DirtySet::all(plan);
    }

    let mut seeds: BTreeSet<NodeId> = BTreeSet::new();

    // The input marker carries every background/global adjustment.
    if global_params_changed(previous, next) {
        seeds.insert(NodeGraph::INPUT);
    }

    for id in &plan.order {
        if *id == NodeGraph::INPUT || *id == NodeGraph::OUTPUT {
            continue;
        }
        if plan.kind(*id) != Some(NodeKind::Layer) {
            // Mixers have no parameters of their own; they dirty through
            // their inputs.
            continue;
        }
        let before = previous.layers.iter().find(|row| row.id == *id);
        let after = next.layers.iter().find(|row| row.id == *id);
        if layer_params(before) != layer_params(after) {
            seeds.insert(*id);
        }
    }

    // Close over consumers. `plan.consumers` is in evaluation order, and the
    // final filter preserves plan order, so scheduling stays deterministic.
    let mut dirty: BTreeSet<NodeId> = BTreeSet::new();
    let mut queue: Vec<NodeId> = seeds.iter().copied().collect();
    while let Some(id) = queue.pop() {
        if !dirty.insert(id) {
            continue;
        }
        queue.extend(plan.consumers(id));
    }

    DirtySet {
        nodes: plan.order.iter().copied().filter(|id| dirty.contains(id)).collect(),
        global: false,
    }
}

/// The consumed subset of a document, serialized. `None` means "no document",
/// used for comparisons against missing rows.
fn global_params(document: &AdjustmentSet) -> Option<Vec<u8>> {
    #[derive(Serialize)]
    struct GlobalParams<'a> {
        base: &'a BaseCharacteristics,
        white_balance: &'a WhiteBalance,
        exposure: &'a Exposure,
        hdr: &'a HighDynamicRange,
        levels: &'a Levels,
        curves: &'a Curves,
        clarity: &'a Clarity,
        dehaze: &'a Dehaze,
        vignetting: &'a Vignetting,
        color_balance: &'a ColorBalance,
        color_balance_gains: &'a Option<ColorBalanceGains>,
        color_editor: &'a ColorEditor,
        black_white: &'a BlackWhite,
        sharpening: &'a Sharpening,
        noise_reduction: &'a NoiseReduction,
        film_grain: &'a FilmGrain,
        moire: &'a Moire,
        lens: &'a LensCorrection,
        geometry: &'a Geometry,
        spots: &'a [Spot],
        retouch: &'a [RetouchFace],
        style: &'a Option<String>,
        extra: &'a BTreeMap<String, f32>,
    }
    let params = GlobalParams {
        base: &document.base,
        white_balance: &document.white_balance,
        exposure: &document.exposure,
        hdr: &document.hdr,
        levels: &document.levels,
        curves: &document.curves,
        clarity: &document.clarity,
        dehaze: &document.dehaze,
        vignetting: &document.vignetting,
        color_balance: &document.color_balance,
        color_balance_gains: &document.color_balance_gains,
        color_editor: &document.color_editor,
        black_white: &document.black_white,
        sharpening: &document.sharpening,
        noise_reduction: &document.noise_reduction,
        film_grain: &document.film_grain,
        moire: &document.moire,
        lens: &document.lens,
        geometry: &document.geometry,
        spots: &document.spots,
        retouch: &document.retouch,
        style: &document.style,
        extra: &document.extra,
    };
    serde_json::to_vec(&params).ok()
}

/// The consumed subset of one layer row, serialized. `None` means no row.
///
/// Name, `mask_uuid`, `style` metadata and `select_point` are deliberately
/// absent: they are identity/interface data, not pixels.
fn layer_params(layer: Option<&Layer>) -> Option<Vec<u8>> {
    #[derive(Serialize)]
    struct LayerParams<'a> {
        kind: LayerKind,
        enabled: bool,
        opacity: f32,
        mask: &'a Mask,
        adjustments: &'a LocalAdjustments,
        cover: &'a [CoverPatch],
    }
    let layer = layer?;
    let params = LayerParams {
        kind: layer.kind,
        enabled: layer.enabled,
        opacity: layer.opacity,
        mask: &layer.mask,
        adjustments: &layer.adjustments,
        cover: &layer.cover,
    };
    serde_json::to_vec(&params).ok()
}

/// The parameter subset a node consumes. The same encoding feeds dirty
/// comparison and cache identity.
fn node_params(document: &AdjustmentSet, node: NodeId, kind: NodeKind) -> Option<Vec<u8>> {
    match kind {
        NodeKind::Input => global_params(document),
        NodeKind::Layer => layer_params(document.layers.iter().find(|row| row.id == node)),
        // Mixers and the output carry no parameters of their own. Their result
        // depends on their inputs, which the caller folds into `revision`.
        NodeKind::Parallel | NodeKind::LayerMixer | NodeKind::Output => {
            Some(Vec::new())
        }
    }
}

fn global_params_changed(previous: &AdjustmentSet, next: &AdjustmentSet) -> bool {
    let before = global_params(previous);
    let after = global_params(next);
    // A serialization failure is treated as a change: over-invalidating is
    // safe, under-invalidating would show stale pixels.
    match (before, after) {
        (Some(before), Some(after)) => before != after,
        _ => true,
    }
}

/// Crop, rotation, orientation, keystone and lens remapping move the sampling
/// grid for every node, so they are not local edits.
fn global_geometry_changed(previous: &AdjustmentSet, next: &AdjustmentSet) -> bool {
    let before = serde_json::to_vec(&(&previous.geometry, &previous.lens)).ok();
    let after = serde_json::to_vec(&(&next.geometry, &next.lens)).ok();
    match (before, after) {
        (Some(before), Some(after)) => before != after,
        _ => true,
    }
}

/// Node ids/kinds and links, ignoring `pos` (layout only) and the stored order
/// of links (semantically irrelevant). `None` is distinct from an empty graph:
/// adding or removing the stored topology is structural.
fn topology_changed(previous: &AdjustmentSet, next: &AdjustmentSet) -> bool {
    // `NodeKind` derives `Eq`/`Hash` but not `Ord`, and it is outside this
    // module's edit scope; a stable tag keeps the signature ordered.
    type Signature = (BTreeSet<(NodeId, u8)>, BTreeSet<(NodeId, NodeId, u32)>);
    fn kind_tag(kind: NodeKind) -> u8 {
        match kind {
            NodeKind::Input => 0,
            NodeKind::Output => 1,
            NodeKind::Layer => 2,
            NodeKind::Parallel => 3,
            NodeKind::LayerMixer => 4,
        }
    }
    fn signature(graph: Option<&NodeGraph>) -> Option<Signature> {
        graph.map(|graph| {
            (
                graph
                    .nodes
                    .iter()
                    .map(|node| (node.id, kind_tag(node.kind)))
                    .collect(),
                graph.links.iter().map(|link| (link.from, link.to, link.input)).collect(),
            )
        })
    }
    signature(previous.nodes.as_ref()) != signature(next.nodes.as_ref())
}

// ───────────────────────────── cache identity ─────────────────────────────

/// A backend's key identity.
///
/// Two backends share cache entries only when their [`equivalence`] tags match.
/// The `name` is diagnostic and is not hashed: a CPU emulator and native CPU
/// can be declared equivalent by giving them the same tag.
///
/// [`equivalence`]: BackendTag::equivalence
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BackendTag {
    pub name: &'static str,
    pub equivalence: u16,
}

impl BackendTag {
    pub const CPU: Self = Self::new("cpu", 0);
    pub const CUDA: Self = Self::new("cuda", 1);
    pub const METAL: Self = Self::new("metal", 2);
    pub const DIRECTML: Self = Self::new("directml", 3);

    pub const fn new(name: &'static str, equivalence: u16) -> Self {
        Self { name, equivalence }
    }

    /// Whether results from `self` and `other` may share cache entries.
    pub const fn equivalent(self, other: Self) -> bool {
        self.equivalence == other.equivalence
    }
}

/// The color space a node result is cached in.
///
/// This is the render/cache output encoding, not the frame color space in
/// [`crate::contracts::ColorSpace`] (which also models camera-native and ICC
/// spaces). They are deliberately separate until the renderer migrates onto
/// the public contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CacheColorSpace {
    LinearRgb,
    Srgb,
    DisplayP3,
    AdobeRgb,
}

impl CacheColorSpace {
    const fn tag(self) -> u8 {
        match self {
            Self::LinearRgb => 0,
            Self::Srgb => 1,
            Self::DisplayP3 => 2,
            Self::AdobeRgb => 3,
        }
    }
}

/// The numeric precision a node result is cached in.
///
/// This is storage precision (u8/u16/f32), not the compute precision in
/// [`crate::contracts::Precision`] (f32/f16/mixed).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CachePrecision {
    U8,
    U16,
    F32,
}

impl CachePrecision {
    const fn tag(self) -> u8 {
        match self {
            Self::U8 => 0,
            Self::U16 => 1,
            Self::F32 => 2,
        }
    }
}

/// Per-image context shared by every node key.
///
/// `source_identity` is the source's identity bytes (path + content/time
/// fingerprint, whatever the caller uses to prove it is the same pixels).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheContext<'a> {
    pub source_identity: &'a [u8],
    /// Region of interest in pixels `[x, y, width, height]`.
    pub roi: [u32; 4],
    pub color_space: CacheColorSpace,
    pub precision: CachePrecision,
    /// Bump when a node's algorithm changes without a document edit.
    pub algorithm_version: u32,
    pub backend: BackendTag,
}

impl CacheContext<'_> {
    fn hash(&self, node: NodeId, revision: u64, params: &[u8]) -> u64 {
        let mut hash = Fnv::new();
        hash.bytes(b"aurora-render-plan-node-cache-v1");
        hash.bytes(self.source_identity);
        hash.u64(node);
        hash.u64(revision);
        hash.bytes(params);
        for value in self.roi {
            hash.u32(value);
        }
        hash.u8(self.color_space.tag());
        hash.u8(self.precision.tag());
        hash.u32(self.algorithm_version);
        hash.u16(self.backend.equivalence);
        hash.finish()
    }
}

/// Stable cache identity for one node result.
///
/// Equality is structural (`node`, `revision`, `hash`); the hash is FNV-1a
/// over a fixed byte encoding, so the same inputs give the same key across
/// runs and process restarts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeCacheKey {
    pub node: NodeId,
    pub revision: u64,
    pub hash: u64,
}

/// Local FNV-1a 64. Length-prefixing byte strings keeps fields unambiguous.
struct Fnv(u64);

impl Fnv {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.u64(bytes.len() as u64);
        self.write(bytes);
    }

    fn u8(&mut self, value: u8) {
        self.write(&[value]);
    }

    fn u16(&mut self, value: u16) {
        self.write(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    fn finish(self) -> u64 {
        self.0
    }
}

// ───────────────────────────── execution counter ─────────────────────────────

/// A tiny record of which nodes ran, for proving reuse without pixels.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionRecorder {
    runs: Vec<NodeId>,
}

impl ExecutionRecorder {
    /// Records one execution, in order. Repeated runs are kept.
    pub fn record(&mut self, node: NodeId) {
        self.runs.push(node);
    }

    /// Every recorded run, in execution order.
    pub fn recorded(&self) -> &[NodeId] {
        &self.runs
    }

    /// How many times `node` ran.
    pub fn count(&self, node: NodeId) -> usize {
        self.runs.iter().filter(|id| **id == node).count()
    }

    pub fn ran(&self, node: NodeId) -> bool {
        self.runs.contains(&node)
    }

    pub fn clear(&mut self) {
        self.runs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const I: NodeId = NodeGraph::INPUT;
    const O: NodeId = NodeGraph::OUTPUT;

    fn graph(links: &[(NodeId, NodeId, u32)], layers: &[NodeId], mixers: &[NodeId]) -> NodeGraph {
        let mut nodes = vec![
            crate::GraphNode { id: I, kind: NodeKind::Input, pos: [0.0; 2] },
            crate::GraphNode { id: O, kind: NodeKind::Output, pos: [0.0; 2] },
        ];
        nodes.extend(
            layers
                .iter()
                .map(|id| crate::GraphNode { id: *id, kind: NodeKind::Layer, pos: [0.0; 2] }),
        );
        nodes.extend(mixers.iter().map(|id| crate::GraphNode {
            id: *id,
            kind: NodeKind::Parallel,
            pos: [0.0; 2],
        }));
        NodeGraph {
            nodes,
            links: links
                .iter()
                .map(|&(from, to, input)| NodeLink { from, to, input })
                .collect(),
        }
    }

    fn document(layers: &[NodeId], graph: NodeGraph) -> AdjustmentSet {
        AdjustmentSet {
            layers: layers
                .iter()
                .map(|id| Layer { id: *id, ..Default::default() })
                .collect(),
            nodes: Some(graph),
            ..Default::default()
        }
    }

    fn index(plan: &RenderPlan, id: NodeId) -> usize {
        plan.order.iter().position(|node| *node == id).unwrap()
    }

    #[test]
    fn linear_graph_compiles_in_evaluation_order() {
        let plan =
            RenderPlan::compile(&graph(&[(I, 1, 0), (1, 2, 0), (2, O, 0)], &[1, 2], &[]))
                .unwrap();
        assert_eq!(plan.order, vec![I, 1, 2, O]);
        assert_eq!(plan.pruned, Vec::<NodeId>::new());
        assert!(plan.warnings.is_empty());
        // The same graph, declared in a different edge order, compiles the same.
        let reordered =
            RenderPlan::compile(&graph(&[(2, O, 0), (I, 1, 0), (1, 2, 0)], &[1, 2], &[]))
                .unwrap();
        assert_eq!(plan, reordered);
    }

    #[test]
    fn compile_delegates_ordering_and_layer_pruning_to_node_graph_plan() {
        // One graph with a branch, a mixer, a detached layer and a dangling
        // link. The execution view must not disagree with the single plan().
        let g = graph(
            &[
                (I, 1, 0),
                (1, 2, 0),
                (1, 3, 0),
                (2, 10, 0),
                (3, 10, 1),
                (10, O, 0),
                (1, 4, 0),
                (1, 77, 0),
            ],
            &[1, 2, 3, 4],
            &[10],
        );
        let base = g.plan().unwrap();
        let plan = RenderPlan::compile(&g).unwrap();
        assert_eq!(plan.order, base.order);
        // Every layer plan() prunes is reported here too; this graph has no
        // detached non-layer node, so the sets coincide.
        assert_eq!(plan.pruned, base.pruned);
        for warning in &base.warnings {
            assert!(plan.warnings.contains(warning), "missing {warning:?}");
        }
    }

    #[test]
    fn branch_and_mixer_put_both_branches_before_the_mixer_and_report_the_detached_branch() {
        let plan = RenderPlan::compile(&graph(
            &[
                (I, 1, 0),
                (1, 2, 0),
                (1, 3, 0),
                (2, 10, 0),
                (3, 10, 1),
                (10, O, 0),
                // Detached: hangs off 1 but never reaches the output.
                (1, 4, 0),
            ],
            &[1, 2, 3, 4],
            &[10],
        ))
        .unwrap();

        assert!(index(&plan, 1) < index(&plan, 2));
        assert!(index(&plan, 2) < index(&plan, 10));
        assert!(index(&plan, 3) < index(&plan, 10));
        assert!(index(&plan, 10) < index(&plan, O));
        // Mixer input order is the tie-break: input 0 (layer 2) runs before
        // input 1 (layer 3), matching the renderer's stacking.
        assert!(index(&plan, 2) < index(&plan, 3));
        assert_eq!(plan.pruned, vec![4]);
        assert!(!plan.contains(4));
    }

    #[test]
    fn a_cycle_on_the_output_path_is_an_error_naming_the_loop_members() {
        let error = RenderPlan::compile(&graph(
            &[(I, 1, 0), (1, 2, 0), (2, 1, 1), (2, O, 0)],
            &[1, 2],
            &[],
        ))
        .unwrap_err();
        match error {
            GraphError::Cycle { mut nodes } => {
                nodes.sort_unstable();
                assert_eq!(nodes, vec![1, 2]);
            }
        }
    }

    #[test]
    fn a_detached_cycle_only_prunes() {
        let plan = RenderPlan::compile(&graph(
            &[(I, 1, 0), (1, O, 0), (2, 3, 0), (3, 2, 0)],
            &[1, 2, 3],
            &[],
        ))
        .unwrap();
        assert_eq!(plan.order, vec![I, 1, O]);
        assert_eq!(plan.pruned, vec![2, 3]);
    }

    #[test]
    fn a_missing_mixer_input_is_a_warning_not_a_refusal() {
        let plan = RenderPlan::compile(&graph(
            &[(I, 1, 0), (1, 2, 0), (2, 10, 1), (10, O, 0)],
            &[1, 2],
            &[10],
        ))
        .unwrap();
        assert!(plan.warnings.contains(&GraphWarning::MissingInput { node: 10, input: 0 }));
        assert_eq!(plan.order, vec![I, 1, 2, 10, O]);
    }

    #[test]
    fn a_serial_node_without_an_input_is_a_warning_and_still_compiles() {
        let plan =
            RenderPlan::compile(&graph(&[(I, 1, 0), (1, O, 0), (2, O, 1)], &[1, 2], &[]))
                .unwrap();
        assert!(plan.contains(2));
        assert!(plan.warnings.contains(&GraphWarning::MissingInput { node: 2, input: 0 }));
    }

    #[test]
    fn a_dangling_link_is_reported_and_dropped() {
        let plan = RenderPlan::compile(&graph(
            &[(I, 1, 0), (1, O, 0), (1, 77, 0)],
            &[1],
            &[],
        ))
        .unwrap();
        assert!(plan.warnings.contains(&GraphWarning::DanglingLink {
            link: NodeLink { from: 1, to: 77, input: 0 },
        }));
        assert_eq!(plan.order, vec![I, 1, O]);
    }

    fn branch_document() -> AdjustmentSet {
        document(
            &[1, 2, 3],
            graph(
                &[
                    (I, 1, 0),
                    (1, 2, 0),
                    (1, 3, 0),
                    (2, 10, 0),
                    (3, 10, 1),
                    (10, O, 0),
                ],
                &[1, 2, 3],
                &[10],
            ),
        )
    }

    #[test]
    fn changing_one_node_dirties_it_and_its_consumers_but_not_its_siblings() {
        let before = branch_document();
        let plan = RenderPlan::compile_document(&before).unwrap();
        let mut after = before.clone();
        after.layers[1].adjustments.exposure.exposure = 1.5;

        let dirty = dirty_nodes(&before, &after, &plan);
        assert!(!dirty.global);
        assert!(!dirty.contains(I));
        assert!(!dirty.contains(1));
        assert!(!dirty.contains(3), "sibling branch must stay reusable");
        assert!(dirty.contains(2));
        assert!(dirty.contains(10), "the mixer consumes the edited branch");
        assert!(dirty.contains(O));
        // The dirty set is in evaluation order.
        assert_eq!(dirty.nodes, vec![2, 10, O]);
    }

    #[test]
    fn a_name_or_layout_change_does_not_dirty_anything() {
        let before = branch_document();
        let plan = RenderPlan::compile_document(&before).unwrap();
        let mut after = before.clone();
        after.layers[1].name = "Renamed".into();
        after.nodes.as_mut().unwrap().nodes[2].pos = [999.0, 999.0];

        assert!(dirty_nodes(&before, &after, &plan).is_empty());
        let mut global = after.clone();
        global.white_balance.kelvin = 4000.0;
        let dirty = dirty_nodes(&before, &global, &plan);
        assert!(dirty.contains(I));
        assert!(dirty.contains(2) && dirty.contains(1) && dirty.contains(3));
    }

    #[test]
    fn a_global_geometry_change_invalidates_every_node() {
        let before = branch_document();
        let plan = RenderPlan::compile_document(&before).unwrap();
        let mut after = before.clone();
        after.geometry.rotation = 2.0;

        let dirty = dirty_nodes(&before, &after, &plan);
        assert!(dirty.global);
        assert_eq!(dirty.nodes, plan.order);
    }

    fn context() -> CacheContext<'static> {
        CacheContext {
            source_identity: b"image-42-content-hash",
            roi: [0, 0, 1920, 1080],
            color_space: CacheColorSpace::LinearRgb,
            precision: CachePrecision::F32,
            algorithm_version: 3,
            backend: BackendTag::CPU,
        }
    }

    #[test]
    fn cache_keys_are_deterministic_and_sensitive_to_backend_and_roi() {
        let document = branch_document();
        let plan = RenderPlan::compile_document(&document).unwrap();
        let ctx = context();

        let first = plan.cache_key(&document, 2, 7, &ctx).unwrap();
        let again = plan.cache_key(&document, 2, 7, &ctx).unwrap();
        assert_eq!(first, again);

        let mut other_backend = ctx;
        other_backend.backend = BackendTag::CUDA;
        let cuda = plan.cache_key(&document, 2, 7, &other_backend).unwrap();
        assert_ne!(first, cuda);

        let mut other_roi = ctx;
        other_roi.roi = [0, 0, 960, 540];
        assert_ne!(first, plan.cache_key(&document, 2, 7, &other_roi).unwrap());

        // Explicitly equivalent backends share a key.
        let same_class = BackendTag::new("cpu-emulated", BackendTag::CPU.equivalence);
        let mut equivalent = ctx;
        equivalent.backend = same_class;
        assert_eq!(first, plan.cache_key(&document, 2, 7, &equivalent).unwrap());

        // Same document but a changed parameter subset is a different key.
        let mut edited = document.clone();
        edited.layers[1].adjustments.exposure.exposure = 1.5;
        assert_ne!(first, plan.cache_key(&edited, 2, 7, &ctx).unwrap());
        // A sibling edit does not change this node's key.
        let mut sibling = document.clone();
        sibling.layers[2].adjustments.exposure.exposure = -1.0;
        assert_eq!(first, plan.cache_key(&sibling, 2, 7, &ctx).unwrap());
    }

    #[test]
    fn execution_counter_proves_only_the_dirty_subgraph_reruns() {
        let before = document(
            &[1, 2, 3],
            graph(&[(I, 1, 0), (1, 2, 0), (2, 3, 0), (3, O, 0)], &[1, 2, 3], &[]),
        );
        let plan = RenderPlan::compile_document(&before).unwrap();

        // First render: every reachable node runs once.
        let mut recorder = ExecutionRecorder::default();
        plan.record_all(&mut recorder);
        assert_eq!(recorder.recorded(), plan.order.as_slice());
        for id in &plan.order {
            assert_eq!(recorder.count(*id), 1);
        }

        // Edit layer 2: it, its consumer 3, and the output re-run; 1 and the
        // input stay cached.
        let mut after = before.clone();
        after.layers[1].adjustments.exposure.exposure = 0.75;
        let dirty = dirty_nodes(&before, &after, &plan);
        recorder.clear();
        plan.record_runs(&dirty, &mut recorder);
        assert_eq!(recorder.recorded(), &[2, 3, O]);
        assert!(!recorder.ran(1));
        assert!(!recorder.ran(I));
    }

    #[test]
    fn document_without_a_graph_compiles_the_legacy_serial_chain() {
        let document = AdjustmentSet {
            layers: [1, 2, 3]
                .iter()
                .map(|id| Layer { id: *id, ..Default::default() })
                .collect(),
            ..Default::default()
        };
        let plan = RenderPlan::compile_document(&document).unwrap();
        assert_eq!(plan.order, vec![I, 1, 2, 3, O]);
        assert!(plan.warnings.is_empty());
    }
}
