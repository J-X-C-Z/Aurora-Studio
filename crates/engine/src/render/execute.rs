//! `RenderPlan`-driven execution, dirty-subgraph cache reuse and structured
//! execution receipts.
//!
//! [`PlanExecutor`] compiles every document through
//! [`crate::render_plan::RenderPlan::compile_document`], rejects cycles
//! explicitly (`RenderError::Graph(GraphError::Cycle { .. })`), computes the
//! affected node set with [`crate::render_plan::dirty_nodes`], and delegates
//! actual pixel work to the single verified renderer in [`super`] (which
//! currently delegates zero-cost to `crate::mock::process`).

use std::fmt;
use std::sync::Arc;

use crate::contracts::compute::{BackendPolicy, SelectionError};
use crate::render_plan::{
    dirty_nodes, BackendTag, CacheColorSpace, CacheContext, CachePrecision, DirtySet,
    ExecutionRecorder, NodeCacheKey, NodeId, RenderPlan,
};
use crate::{AdjustmentSet, GraphError, GraphWarning, NodeGraph, NodeKind, NormRect};

use super::cache::{FrameCache, FrameCacheKey, FrameCacheStats, DEFAULT_FRAME_CACHE_BUDGET_BYTES};
use super::{clean_region, preview_size, render, render16, render_at, render_preview_bgra, Rendered, Rendered16, Source};

/// Default algorithm version hashed into [`CacheContext`].
pub const RENDER_ALGORITHM_VERSION: u32 = 1;

/// Output buffer produced by [`PlanExecutor`], covering both 8-bit preview and
/// 16-bit export paths under a unified `RenderPlan` contract.
pub enum RenderOutput {
    Rgba8(Rendered),
    Rgba16(Rendered16),
}

impl RenderOutput {
    /// Width of the rendered frame in pixels.
    pub fn width(&self) -> u32 {
        match self {
            Self::Rgba8(r) => r.width,
            Self::Rgba16(r) => r.width,
        }
    }

    /// Height of the rendered frame in pixels.
    pub fn height(&self) -> u32 {
        match self {
            Self::Rgba8(r) => r.height,
            Self::Rgba16(r) => r.height,
        }
    }

    /// Pixel payload size in bytes, used for [`FrameCache`] byte-budget accounting.
    pub fn byte_len(&self) -> usize {
        match self {
            Self::Rgba8(r) => r.rgba.len(),
            Self::Rgba16(r) => r.rgba.len().saturating_mul(std::mem::size_of::<u16>()),
        }
    }

    /// Storage precision of this frame.
    pub fn precision(&self) -> CachePrecision {
        match self {
            Self::Rgba8(_) => CachePrecision::U8,
            Self::Rgba16(_) => CachePrecision::U16,
        }
    }

    /// Borrow the underlying 8-bit [`Rendered`] frame if this is `Rgba8`.
    pub fn as_rgba8(&self) -> Option<&Rendered> {
        match self {
            Self::Rgba8(r) => Some(r),
            Self::Rgba16(_) => None,
        }
    }

    /// Borrow the underlying 16-bit [`Rendered16`] frame if this is `Rgba16`.
    pub fn as_rgba16(&self) -> Option<&Rendered16> {
        match self {
            Self::Rgba8(_) => None,
            Self::Rgba16(r) => Some(r),
        }
    }

    /// True when the 8-bit buffer is already in display BGRA channel order.
    pub fn is_bgra(&self) -> bool {
        match self {
            Self::Rgba8(r) => r.bgra,
            Self::Rgba16(_) => false,
        }
    }
}

/// Target format and preview mode for a [`PlanExecutor`] request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderTarget {
    /// 8-bit preview render (`render`, `render_preview_bgra`, or `render_at`).
    PreviewRgba8 {
        /// Request BGRA display channel order when the GPU resident fast path
        /// handles the whole frame.
        want_bgra: bool,
        /// Legacy interactive priority; it never changes the output pixel grid.
        fast: bool,
    },
    /// 16-bit export render (`render16`).
    ExportRgba16,
}

impl RenderTarget {
    /// Storage precision corresponding to this target.
    pub const fn precision(self) -> CachePrecision {
        match self {
            Self::PreviewRgba8 { .. } => CachePrecision::U8,
            Self::ExportRgba16 => CachePrecision::U16,
        }
    }

    pub const fn want_bgra(self) -> bool {
        match self {
            Self::PreviewRgba8 { want_bgra, .. } => want_bgra,
            Self::ExportRgba16 => false,
        }
    }

    pub const fn fast(self) -> bool {
        match self {
            Self::PreviewRgba8 { fast, .. } => fast,
            Self::ExportRgba16 => false,
        }
    }
}

/// Request options for [`PlanExecutor::execute`] / [`execute_with_plan`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderRequestOptions {
    pub region: NormRect,
    pub max_size: u32,
    pub bypass: bool,
    pub mask_overlay: Option<u64>,
    pub target: RenderTarget,
    pub color_space: CacheColorSpace,
    pub backend_policy: BackendPolicy,
    pub backend_tag: BackendTag,
    pub algorithm_version: u32,
}

impl RenderRequestOptions {
    /// Standard 8-bit sRGB RGBA preview request.
    pub fn preview(region: NormRect, max_size: u32) -> Self {
        Self {
            region,
            max_size,
            bypass: false,
            mask_overlay: None,
            target: RenderTarget::PreviewRgba8 { want_bgra: false, fast: false },
            color_space: CacheColorSpace::Srgb,
            backend_policy: BackendPolicy::Auto,
            backend_tag: BackendTag::CPU,
            algorithm_version: RENDER_ALGORITHM_VERSION,
        }
    }

    /// 8-bit preview request with optional BGRA display fast path.
    pub fn preview_bgra(region: NormRect, max_size: u32, want_bgra: bool) -> Self {
        Self {
            target: RenderTarget::PreviewRgba8 { want_bgra, fast: false },
            ..Self::preview(region, max_size)
        }
    }

    /// 16-bit export request sharing the same compiled [`RenderPlan`].
    pub fn export16(region: NormRect, max_size: u32) -> Self {
        Self {
            target: RenderTarget::ExportRgba16,
            ..Self::preview(region, max_size)
        }
    }

    pub fn with_bypass(mut self, bypass: bool) -> Self {
        self.bypass = bypass;
        self
    }

    pub fn with_mask_overlay(mut self, mask_overlay: Option<u64>) -> Self {
        self.mask_overlay = mask_overlay;
        self
    }

    pub fn with_fast(mut self, fast: bool) -> Self {
        if let RenderTarget::PreviewRgba8 { want_bgra, .. } = self.target {
            self.target = RenderTarget::PreviewRgba8 { want_bgra, fast };
        }
        self
    }

    pub fn with_color_space(mut self, color_space: CacheColorSpace) -> Self {
        self.color_space = color_space;
        self
    }

    pub fn with_backend_policy(mut self, backend_policy: BackendPolicy) -> Self {
        self.backend_policy = backend_policy;
        self
    }

    pub fn with_backend_tag(mut self, backend_tag: BackendTag) -> Self {
        self.backend_tag = backend_tag;
        self
    }

    pub fn with_algorithm_version(mut self, algorithm_version: u32) -> Self {
        self.algorithm_version = algorithm_version;
        self
    }
}

/// Errors returned by [`PlanExecutor::execute`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderError {
    /// The node graph failed to compile (e.g. [`GraphError::Cycle`]).
    Graph(GraphError),
    /// Backend selection refused execution under the requested policy.
    Backend(SelectionError),
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(err) => write!(f, "render plan compilation failed: {err:?}"),
            Self::Backend(err) => write!(f, "render backend selection failed: {err}"),
        }
    }
}

impl std::error::Error for RenderError {}

impl From<GraphError> for RenderError {
    fn from(err: GraphError) -> Self {
        Self::Graph(err)
    }
}

impl From<SelectionError> for RenderError {
    fn from(err: SelectionError) -> Self {
        Self::Backend(err)
    }
}

/// Structured proof of how one [`PlanExecutor::execute`] call ran.
///
/// Tracks real invocation and cache-reuse counters so callers and tests can
/// verify reuse without relying solely on pixel equality.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanExecutionReceipt {
    /// Number of nodes in the compiled evaluation order.
    pub compiled_order_len: usize,
    /// Nodes pruned as unreachable from the output or disconnected from input.
    pub pruned_nodes: Vec<NodeId>,
    /// Number of active (non-pruned) nodes dirtied since the previous snapshot.
    pub dirty_count: usize,
    /// True when global geometry, topology, or source identity invalidated the
    /// entire compiled plan.
    pub dirty_global: bool,
    /// True when this call served a cached frame and skipped pixel rendering.
    pub cache_hit: bool,
    /// Cumulative number of cache reuses served by this executor.
    pub cache_reuses: u64,
    /// Cumulative number of underlying pixel renders executed by this executor.
    pub render_invocations: u64,
    /// Backend policy attached to this request.
    pub backend_policy: BackendPolicy,
    /// Non-fatal structural warnings reported by [`RenderPlan::compile_document`].
    pub warnings: Vec<GraphWarning>,
    /// Compiled plan output cache key for this request.
    pub cache_key: NodeCacheKey,
}

/// Per-session `RenderPlan` executor with bounded LRU frame caching and
/// execution counters.
pub struct PlanExecutor {
    cache: FrameCache,
    last_document: Option<AdjustmentSet>,
    last_source_id: Option<u64>,
    render_invocations: u64,
    cache_reuses: u64,
    recorder: ExecutionRecorder,
}

/// Session-scoped production render facade alias.
pub type RenderSession = PlanExecutor;

impl Default for PlanExecutor {
    fn default() -> Self {
        Self::new(DEFAULT_FRAME_CACHE_BUDGET_BYTES)
    }
}

impl PlanExecutor {
    /// Create a new executor with a bounded frame cache of `cache_budget_bytes`.
    pub fn new(cache_budget_bytes: usize) -> Self {
        Self {
            cache: FrameCache::new(cache_budget_bytes),
            last_document: None,
            last_source_id: None,
            render_invocations: 0,
            cache_reuses: 0,
            recorder: ExecutionRecorder::default(),
        }
    }

    /// Compile the document's `RenderPlan`, compute dirty nodes against the
    /// previous snapshot, reuse the frame cache whenever possible, and return
    /// both the rendered output and its [`PlanExecutionReceipt`].
    pub fn execute(
        &mut self,
        src: &Source,
        adj: &AdjustmentSet,
        opts: RenderRequestOptions,
    ) -> Result<(Arc<RenderOutput>, PlanExecutionReceipt), RenderError> {
        // 1. Compile the execution graph. Cycles are reported explicitly and
        //    never silently swallowed.
        let plan = RenderPlan::compile_document(adj)?;

        // 2. Compute dirty nodes against the previous execution snapshot for
        //    the same source.
        let source_id = src.snapshot_id();
        let dirty = match (&self.last_document, self.last_source_id) {
            (Some(prev), Some(prev_src)) if prev_src == source_id => {
                dirty_nodes(prev, adj, &plan)
            }
            _ => DirtySet::all(&plan),
        };
        let dirty_count = active_dirty_count(&plan, &dirty);

        // 3. Build deterministic plan + frame cache key incorporating source
        //    identity, active subgraph parameters/topology, ROI, output size,
        //    precision, color space and backend equivalence.
        let cleaned_region = clean_region(opts.region);
        let roi = compute_roi(src, adj, cleaned_region, opts.max_size);
        let source_identity = source_identity_bytes(src);
        let ctx = CacheContext {
            source_identity: &source_identity,
            roi,
            color_space: opts.color_space,
            precision: opts.target.precision(),
            algorithm_version: opts.algorithm_version,
            backend: opts.backend_tag,
        };
        let plan_key = compute_plan_output_key(&plan, adj, &ctx);
        let frame_key = FrameCacheKey::new(
            plan_key,
            cleaned_region,
            roi,
            opts.max_size,
            opts.target.precision(),
            opts.color_space,
            opts.backend_tag,
            opts.bypass,
            opts.mask_overlay,
            opts.target.want_bgra(),
            opts.target.fast(),
        );

        // 4. Check frame cache. When `!dirty.global && dirty_count == 0` (e.g.
        //    only node layout coordinates or pruned branch parameters changed),
        //    `frame_key` is unchanged and hits the cache directly. Even when
        //    `dirty_count > 0` (e.g. undo back to a previously cached state or
        //    switching back to a cached ROI), a matching `frame_key` safely
        //    reuses the exact cached frame.
        let (output, cache_hit) = if let Some(cached) = self.cache.get(&frame_key) {
            self.cache_reuses = self.cache_reuses.saturating_add(1);
            (cached, true)
        } else {
            let rendered = invoke_backend_render(src, adj, opts);
            let output = Arc::new(rendered);
            self.render_invocations = self.render_invocations.saturating_add(1);
            plan.record_runs(&dirty, &mut self.recorder);
            self.cache.insert(frame_key, Arc::clone(&output));
            (output, false)
        };

        // 5. Record snapshot for subsequent incremental dirty checks.
        self.last_document = Some(adj.clone());
        self.last_source_id = Some(source_id);

        let receipt = PlanExecutionReceipt {
            compiled_order_len: plan.order.len(),
            pruned_nodes: plan.pruned,
            dirty_count,
            dirty_global: dirty.global,
            cache_hit,
            cache_reuses: self.cache_reuses,
            render_invocations: self.render_invocations,
            backend_policy: opts.backend_policy,
            warnings: plan.warnings,
            cache_key: plan_key,
        };

        Ok((output, receipt))
    }

    /// Total number of underlying pixel renders executed in this session.
    pub fn render_invocations(&self) -> u64 {
        self.render_invocations
    }

    /// Total number of cache hits reused in this session.
    pub fn cache_reuses(&self) -> u64 {
        self.cache_reuses
    }

    /// Current bounded frame cache statistics.
    pub fn cache_stats(&self) -> FrameCacheStats {
        self.cache.stats()
    }

    /// Node execution recorder accumulated on cache misses.
    pub fn execution_recorder(&self) -> &ExecutionRecorder {
        &self.recorder
    }

    /// Clear cached frames while keeping session counters intact.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Functional entry point delegating to [`PlanExecutor::execute`].
pub fn execute_with_plan(
    executor: &mut PlanExecutor,
    src: &Source,
    adj: &AdjustmentSet,
    opts: RenderRequestOptions,
) -> Result<(Arc<RenderOutput>, PlanExecutionReceipt), RenderError> {
    executor.execute(src, adj, opts)
}

fn invoke_backend_render(
    src: &Source,
    adj: &AdjustmentSet,
    opts: RenderRequestOptions,
) -> RenderOutput {
    match opts.target {
        RenderTarget::PreviewRgba8 { want_bgra, fast } => {
            let rendered = if fast {
                render_at(
                    src,
                    adj,
                    opts.region,
                    opts.max_size,
                    opts.bypass,
                    opts.mask_overlay,
                    true,
                    crate::scopes::ScopeWants::default(),
                )
            } else if want_bgra {
                render_preview_bgra(
                    src,
                    adj,
                    opts.region,
                    opts.max_size,
                    opts.bypass,
                    opts.mask_overlay,
                    true,
                    crate::scopes::ScopeWants::default(),
                )
            } else {
                render(
                    src,
                    adj,
                    opts.region,
                    opts.max_size,
                    opts.bypass,
                    opts.mask_overlay,
                )
            };
            RenderOutput::Rgba8(rendered)
        }
        RenderTarget::ExportRgba16 => {
            let rendered16 = render16(src, adj, opts.region, opts.max_size, opts.bypass);
            RenderOutput::Rgba16(rendered16)
        }
    }
}

fn active_dirty_count(plan: &RenderPlan, dirty: &DirtySet) -> usize {
    if dirty.global {
        return dirty.nodes.len();
    }
    dirty
        .nodes
        .iter()
        .filter(|id| !plan.pruned.contains(id))
        .count()
}

fn source_identity_bytes(src: &Source) -> [u8; 16] {
    let (w, h) = src.size();
    let mut out = [0u8; 16];
    out[0..8].copy_from_slice(&src.snapshot_id().to_le_bytes());
    out[8..12].copy_from_slice(&w.to_le_bytes());
    out[12..16].copy_from_slice(&h.to_le_bytes());
    out
}

fn compute_roi(src: &Source, adj: &AdjustmentSet, region: NormRect, max_size: u32) -> [u32; 4] {
    let (src_w, src_h) = src.size();
    let (out_w, out_h) = preview_size(src, adj, region, max_size);
    let rx = (region.x * src_w as f32).round().max(0.0) as u32;
    let ry = (region.y * src_h as f32).round().max(0.0) as u32;
    [rx, ry, out_w, out_h]
}

/// Fold every active (non-pruned) node in `plan.order`—its id, kind, input
/// wiring, and parameter-subset hash—into a single deterministic revision for
/// `NodeGraph::OUTPUT`.
///
/// Pruned nodes and node layout coordinates (`pos`) are excluded, so edits to
/// pruned branches or node positions never alter the output cache key.
fn compute_plan_output_key(
    plan: &RenderPlan,
    adj: &AdjustmentSet,
    ctx: &CacheContext<'_>,
) -> NodeCacheKey {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = FNV_OFFSET;
    let mut mix_u64 = |v: u64| {
        for b in v.to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };

    for &node_id in &plan.order {
        if plan.pruned.contains(&node_id) {
            continue;
        }
        mix_u64(node_id);
        let kind_tag = match plan.kind(node_id) {
            Some(NodeKind::Input) => 1,
            Some(NodeKind::Output) => 2,
            Some(NodeKind::Layer) => 3,
            Some(NodeKind::Parallel) => 4,
            Some(NodeKind::LayerMixer) => 5,
            None => 0,
        };
        mix_u64(kind_tag);
        let deps = plan.dependencies(node_id);
        mix_u64(deps.len() as u64);
        for &dep in deps {
            mix_u64(dep);
        }
        if let Some(node_key) = plan.cache_key(adj, node_id, 0, ctx) {
            mix_u64(node_key.hash);
        }
    }

    plan.cache_key(adj, NodeGraph::OUTPUT, hash, ctx)
        .unwrap_or(NodeCacheKey {
            node: NodeGraph::OUTPUT,
            revision: hash,
            hash,
        })
}
