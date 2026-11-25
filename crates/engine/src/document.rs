//! The persistent edit document and the runtime render context.
//!
//! `AdjustmentSet` is the on-disk shape. Everything that belongs to one render
//! job but must never reach a saved file or an undo step (producer bytes, model
//! outputs, device handles, cancellation) belongs to the runtime context this
//! module describes. The split is introduced as an adapter first: existing
//! `.portrait` files keep their schema, while new code can depend on the
//! document identity types instead of the whole adjustment set.
//!
//! # Status
//!
//! Producer bytes and lens/geometry calibration are now owned by
//! [`RuntimeInputs`] and threaded explicitly through the real render path:
//! `FileEngine` stores one per image, `RenderService`/`ExportService` pass it
//! into `mock::process`, and the preview cache mixes its fingerprint into the
//! key. [`RenderRequest`] remains the vocabulary carrier for job-shaped code.
//! [`AdjustmentSet`] therefore contains persistent state only.
//!
//! Rehydrated calibration (`Geometry::distortion`, `Geometry::optical_center`
//! and `LensCorrection::uniform_light_grid` before this move) never enters a
//! saved file, an undo step, equality or `document_fingerprint`; see
//! [`LensRuntime`].
//!
//! Producer inputs were never attached in production (the only writer,
//! `FaceParsingProducer::produce_bundle`, had no callers), so removing the
//! document field changes no preview or export pixels.
//!
//! # What each identity means
//!
//! These are deliberately separate axes; two documents can agree on one and
//! differ on another.
//!
//! - [`DocumentRevision`] — a monotonic counter for edits *inside one catalog
//!   session*. It answers "which saved edit state is this?", not "does this
//!   render match that pixel?".
//! - [`RenderGeneration`] — the engine's frame ordering token. A newer
//!   generation may be discarded in favour of a newer one; it never identifies
//!   a document.
//! - [`SourceId`] — the source file the document was applied to. The same edit
//!   document over a different source is a different render.
//! - [`RenderIntent`] — draft preview, final preview, export or thumbnail. Same
//!   document, different required quality and caching policy.
//!
//! Two further axes matter to a cache key but have no type here yet because no
//! caller needs them: the **model version** (portrait ONNX assets / kernels) and
//! the **device policy** plus **algorithm version** (CPU reference, CUDA, or a
//! future Metal adapter, each of which may change pixels). Until those are
//! modelled, a cache key that omits them is only valid within one build and one
//! device. Do not read the fingerprints below as covering them.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::adjust::{AdjustmentSet, NormRect};
use crate::portrait_stage::ProducerInputs;

/// Version of the on-disk adjustment document this build writes.
///
/// Mirrors `crate::store::SCHEMA_VERSION`, which lives in the private `store`
/// module and is therefore not reachable from here. Keep the two equal: bump
/// both when the persisted shape changes. This constant describes the *file*
/// schema; it is not the in-memory `AdjustmentSet` layout and is not a render
/// cache version.
pub const DOCUMENT_SCHEMA_VERSION: u32 = 1;

/// A stable, non-cryptographic fingerprint of the *persistent* document.
///
/// Hashes the serde JSON of `adjustments` with FNV-1a. The document struct no
/// longer carries any producer field: runtime inputs live in [`RuntimeInputs`],
/// so this fingerprint is producer-free by construction. Two documents with the
/// same stored edits therefore share a fingerprint regardless of which render
/// job they belonged to.
///
/// This is a cache and identity *hint*, not a cryptographic hash: FNV-1a is
/// cheap and collision-prone by design, so never use it for integrity or
/// authenticity. It hashes JSON bytes, not values, so float edge cases
/// (`-0.0` vs `0.0`, or two `NaN`s that compare unequal) may disagree with
/// `AdjustmentSet`'s `PartialEq`; do not use it as an equality oracle. It also
/// does not prove that two documents refer to the same source pixels across
/// catalogs; pair it with [`SourceId`]. When runtime inputs affect output, a
/// render cache key must include [`RuntimeInputs::fingerprint`] as well.
pub fn document_fingerprint(adjustments: &AdjustmentSet) -> u64 {
    let bytes = serde_json::to_vec(adjustments).unwrap_or_default();
    fnv1a64(FNV1A64_OFFSET_BASIS, &bytes)
}

// ───────────────────────────── Identity newtypes ─────────────────────────────

// The public identity vocabulary lives in [`crate::contracts::identity`]; it is
// re-exported here so document code has one import path and there is no second
// `SourceId`/`DocumentRevision`/`RenderGeneration` definition in the crate.
/// Identifies the source asset a document is applied to.
///
/// The value is an opaque library-level key; a bare filesystem path is not a
/// cross-library identity. See [`crate::contracts::identity::SourceId`].
pub use crate::contracts::identity::SourceId;
/// Monotonic edit revision of one document within one catalog session.
///
/// It orders saved edit states; it is not a render generation. See
/// [`crate::contracts::identity::DocumentRevision`].
pub use crate::contracts::identity::DocumentRevision;
/// The engine's frame ordering token for one render request.
///
/// A newer generation supersedes an older in-flight one; it never identifies a
/// document. See [`crate::contracts::identity::RenderGeneration`].
pub use crate::contracts::identity::RenderGeneration;

/// Why a frame is being rendered. The same document produces different work for
/// each intent; see [`RenderRequest`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderIntent {
    /// Fast, possibly reduced-quality interactive preview.
    Draft,
    /// Full-quality still preview.
    Final,
    /// Full-resolution export to a file.
    Export,
    /// Small cached thumbnail.
    Thumbnail,
}

impl Default for RenderIntent {
    fn default() -> Self {
        Self::Draft
    }
}

// ───────────────────────────── Snapshots and requests ─────────────────────────────

/// A shared, revision-tagged view of one adjustment document.
///
/// Holds the document behind an [`Arc`] so cloning a snapshot (undo stacks,
/// request queues, in-flight jobs) never deep-copies it. The snapshot owns
/// document state only: per-job producer bytes live in [`RenderRequest::runtime`],
/// not here. The revision travels with the shared value so callers can compare
/// snapshots without re-running [`document_fingerprint`].
#[derive(Clone, Debug)]
pub struct DocumentSnapshot {
    document: Arc<AdjustmentSet>,
    revision: DocumentRevision,
}

impl DocumentSnapshot {
    /// Wraps an owned document or an existing `Arc<AdjustmentSet>`.
    pub fn from_adjustments(
        adjustments: impl Into<Arc<AdjustmentSet>>,
        revision: DocumentRevision,
    ) -> Self {
        Self { document: adjustments.into(), revision }
    }

    /// The adjustment document. It is persistent state only; per-job runtime
    /// inputs travel in [`RenderRequest::runtime`].
    pub fn adjustments(&self) -> &AdjustmentSet {
        &self.document
    }

    /// The shared document, for callers that need to move it into another
    /// snapshot or job without copying.
    pub fn shared(&self) -> &Arc<AdjustmentSet> {
        &self.document
    }

    pub fn revision(&self) -> DocumentRevision {
        self.revision
    }

    /// Fingerprint of the persistent document, ignoring producer inputs.
    pub fn document_fingerprint(&self) -> u64 {
        document_fingerprint(&self.document)
    }
}

/// A render request skeleton: which source, which document revision, which
/// region, which generation, which intent and which per-job runtime inputs.
///
/// This is the job-shaped carrier: document-only state stays in
/// [`DocumentSnapshot`], and the ephemeral calibration/producer bundle for
/// *this* job travels in `runtime`. `FileEngine` currently constructs the
/// equivalent runtime inputs from its per-image entry and passes them to the
/// renderer directly; this type remains the public vocabulary for a future
/// queue that owns requests end to end. It is `Clone` and lock-free by design.
#[derive(Clone, Debug)]
pub struct RenderRequest {
    pub source: SourceId,
    pub document: DocumentSnapshot,
    pub region: NormRect,
    pub generation: RenderGeneration,
    pub intent: RenderIntent,
    /// Per-job inputs that must never reach the document, an undo step or a
    /// saved file.
    pub runtime: RuntimeInputs,
}

/// Lens and geometry calibration that is rehydrated at load time but must never
/// be written back to a document, an undo step or a saved file.
///
/// These are exactly the values that used to sit on [`AdjustmentSet`] as
/// `#[serde(skip)]` fields: the C1 radial distortion curve, its optical centre
/// and the `.lcc` uniform-light grid. They are derived from the source/lens
/// database, the `.lcc` sidecar or a `.cos` file, so they describe *this render*
/// of *this source*, not the user's saved edit. They are owned here so the
/// document is persistent state only.
///
/// The type is deliberately `pub(crate)`: the curve type lives in the private
/// `c1_model` module, and no public API needs to name calibration directly. A
/// public field of this type would be an E0446 private-in-public-interface
/// error. The outer [`RuntimeInputs`] keeps a `pub(crate)` field and the
/// crate-private accessors below are the only way to reach it.
#[derive(Clone, Debug, Default)]
pub(crate) struct LensRuntime {
    /// `ZLENSDISTORTION` as a source-pixel radial curve. Empty is the identity.
    pub(crate) distortion: crate::c1_model::crop_limit::RadialCurve,
    /// `ZLENSOPTICCENTER` in source pixels; only read when `distortion` is set.
    pub(crate) optical_center: [f32; 2],
    /// The `.lcc` uniform-light grid, shared so cloning stays cheap.
    pub(crate) uniform_light_grid: Option<Arc<crate::dust_lcc::LightFalloff>>,
}

impl LensRuntime {
    /// True when the calibration cannot change a pixel.
    pub(crate) fn is_empty(&self) -> bool {
        self.distortion.is_empty()
            && self.optical_center == [0.0, 0.0]
            && self.uniform_light_grid.is_none()
    }

    /// Stable FNV-1a fingerprint of the calibration, or `None` when empty.
    ///
    /// Uses the raw bit patterns of the curve/centre and the grid nodes, so it
    /// is independent of `serde` shape and of `Debug` formatting. This is a
    /// cache hint, not an integrity hash.
    pub(crate) fn fingerprint(&self) -> Option<u64> {
        if self.is_empty() {
            return None;
        }
        let mut hash = FNV1A64_OFFSET_BASIS;
        hash = fnv1a64(hash, &self.distortion.v.len().to_le_bytes());
        for value in &self.distortion.v {
            hash = fnv1a64(hash, &value.to_bits().to_le_bytes());
        }
        hash = fnv1a64(hash, &self.distortion.x.len().to_le_bytes());
        for value in &self.distortion.x {
            hash = fnv1a64(hash, &value.to_bits().to_le_bytes());
        }
        hash = fnv1a64(hash, &self.distortion.m.len().to_le_bytes());
        for value in &self.distortion.m {
            hash = fnv1a64(hash, &value.to_bits().to_le_bytes());
        }
        for value in self.optical_center {
            hash = fnv1a64(hash, &value.to_bits().to_le_bytes());
        }
        match self.uniform_light_grid.as_deref() {
            None => hash = fnv1a64(hash, &[0]),
            Some(grid) => {
                hash = fnv1a64(hash, &[1]);
                hash = fnv1a64(hash, &grid.columns.to_le_bytes());
                hash = fnv1a64(hash, &grid.rows.to_le_bytes());
                hash = fnv1a64(hash, &(grid.header.len() as u64).to_le_bytes());
                for value in &grid.header {
                    hash = fnv1a64(hash, &value.to_le_bytes());
                }
                for node in &grid.nodes {
                    hash = fnv1a64(hash, &node.luminance.to_le_bytes());
                    hash = fnv1a64(hash, &node.dx.to_le_bytes());
                    hash = fnv1a64(hash, &node.dy.to_le_bytes());
                }
            }
        }
        Some(hash)
    }
}

/// The ephemeral per-job inputs that are not document state.
///
/// This is the sole owner of producer bytes and of the rehydrated lens
/// calibration. Both belong to one render job and must never live on
/// [`AdjustmentSet`], an undo identity, equality or `document_fingerprint`.
/// The real render path (`FileEngine` -> `RenderService`/`ExportService`) reads
/// the calibration from its per-image entry and passes it explicitly into
/// `mock::process`; there is no global or thread-local fallback.
///
/// Cancellation is intentionally not modelled here: no cancellation type exists
/// yet, and inventing one now would be a guess. Model outputs, device handles,
/// model version, device policy and algorithm version are also not represented;
/// see the module status note. Keep this cheap to clone: the producer bundle is
/// behind [`Arc`] and so is the light-falloff grid.
#[derive(Clone, Debug, Default)]
pub struct RuntimeInputs {
    pub producer_inputs: Option<Arc<ProducerInputs>>,
    /// Rehydrated lens/geometry calibration. Crate-private because its curve
    /// type is not publicly reachable; document identity never names it.
    pub(crate) lens: LensRuntime,
}

impl RuntimeInputs {
    /// Wraps an optional producer bundle.
    pub fn new(producer_inputs: Option<Arc<ProducerInputs>>) -> Self {
        Self { producer_inputs, lens: LensRuntime::default() }
    }

    /// No runtime inputs; the same as `RuntimeInputs::default()`.
    pub fn none() -> Self {
        Self { producer_inputs: None, lens: LensRuntime::default() }
    }

    /// Builds a runtime context from calibration alone.
    pub(crate) fn from_lens(lens: LensRuntime) -> Self {
        Self { producer_inputs: None, lens }
    }

    /// Attaches calibration to this context, replacing any existing one.
    pub(crate) fn with_lens(mut self, lens: LensRuntime) -> Self {
        self.lens = lens;
        self
    }

    /// The rehydrated calibration. Empty for sources without a lens profile.
    pub(crate) fn lens(&self) -> &LensRuntime {
        &self.lens
    }

    /// Fingerprint of all runtime inputs, or `None` when nothing is present.
    ///
    /// Combines [`ProducerInputs::fingerprint`] with the calibration
    /// fingerprint. A render cache key must include this alongside
    /// [`document_fingerprint`] whenever runtime inputs affect output;
    /// including only one can serve a frame rendered with different masks,
    /// model outputs, material planes or lens calibration.
    pub fn fingerprint(&self) -> Option<u64> {
        let producer = self.producer_inputs.as_deref().map(ProducerInputs::fingerprint);
        let lens = self.lens.fingerprint();
        match (producer, lens) {
            (None, None) => None,
            (Some(producer), None) => Some(producer),
            (None, Some(lens)) => Some(lens),
            (Some(producer), Some(lens)) => {
                let mut hash = FNV1A64_OFFSET_BASIS;
                hash = fnv1a64(hash, &producer.to_le_bytes());
                hash = fnv1a64(hash, &lens.to_le_bytes());
                Some(hash)
            }
        }
    }

    /// True when no runtime input is present.
    pub fn is_empty(&self) -> bool {
        self.producer_inputs.is_none() && self.lens.is_empty()
    }
}

// ───────────────────────────── FNV-1a ─────────────────────────────

const FNV1A64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a64(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portrait_stage::SmileProducerInputs;

    /// A tiny, cheap producer bundle. Only `smile.ticket` and `material` vary,
    /// both of which `ProducerInputs::fingerprint` reads.
    fn producer(ticket: u64, material: &str) -> ProducerInputs {
        ProducerInputs {
            smile: Some(SmileProducerInputs {
                ticket,
                material: Some(material.to_string()),
                ..SmileProducerInputs::default()
            }),
            ..ProducerInputs::default()
        }
    }

    #[test]
    fn serialized_document_never_carries_producer_bytes() {
        const SENTINEL: &str = "AURORA_PRODUCER_BYTES_SENTINEL";
        let mut adj = AdjustmentSet::default();
        adj.exposure.exposure = 0.5;
        // Producer bytes live only in the per-job runtime context now.
        let runtime = RuntimeInputs::new(Some(Arc::new(producer(7, SENTINEL))));

        let json = serde_json::to_string(&adj).expect("document serializes");
        assert!(!json.contains("portrait_producer_inputs"), "{json}");
        assert!(!json.contains(SENTINEL), "{json}");
        // The persistent edit is still present, so this is not an empty object.
        assert!(json.contains("\"exposure\""), "{json}");
        // The runtime context is the sole owner and does fingerprint the bytes.
        assert!(!runtime.is_empty());
        assert!(runtime.fingerprint().is_some());
    }

    #[test]
    fn producer_inputs_do_not_change_equality_or_document_fingerprint() {
        let mut a = AdjustmentSet::default();
        a.exposure.exposure = 0.5;
        let b = a.clone();

        let runtime_a = RuntimeInputs::new(Some(Arc::new(producer(1, "first job"))));
        let runtime_b = RuntimeInputs::new(Some(Arc::new(producer(2, "second job"))));

        // Two documents with identical stored edits stay equal and share a
        // fingerprint regardless of which job renders them. The runtime context
        // is deliberately not an input to either.
        assert_eq!(a, b, "runtime inputs must not define document equality");
        assert_eq!(document_fingerprint(&a), document_fingerprint(&b));

        let runtime_a_fp = runtime_a.fingerprint().expect("a has producer inputs");
        let runtime_b_fp = runtime_b.fingerprint().expect("b has producer inputs");
        assert_ne!(runtime_a_fp, runtime_b_fp, "distinct producer bytes must fingerprint apart");
        assert_ne!(runtime_a.fingerprint(), RuntimeInputs::none().fingerprint());

        let mut c = a.clone();
        c.exposure.exposure = 0.6;
        assert_ne!(a, c, "persistent changes still compare unequal");
        assert_ne!(document_fingerprint(&a), document_fingerprint(&c));
    }

    #[test]
    fn runtime_inputs_none_is_empty_and_fingerprints_to_none() {
        let none = RuntimeInputs::none();
        assert!(none.is_empty());
        assert_eq!(none.fingerprint(), None);
        assert_eq!(RuntimeInputs::default().fingerprint(), None);
        assert_eq!(RuntimeInputs::new(None).fingerprint(), None);

        let present = RuntimeInputs::new(Some(Arc::new(producer(9, "job"))));
        assert!(!present.is_empty());
        assert!(present.fingerprint().is_some());
    }

    #[test]
    fn is_default_ignores_producer_inputs() {
        let adj = AdjustmentSet::default();
        assert!(adj.is_default());
        // A runtime context can carry producer bytes for the exact job without
        // making the persistent document look edited.
        let _runtime = RuntimeInputs::new(Some(Arc::new(producer(9, "job"))));
        assert!(adj.is_default(), "runtime bytes must not make a document edited");
    }

    #[test]
    fn render_request_carries_runtime_inputs_for_the_job() {
        let document = DocumentSnapshot::from_adjustments(
            AdjustmentSet::default(),
            DocumentRevision::new(1),
        );
        let runtime = RuntimeInputs::new(Some(Arc::new(producer(3, "job"))));
        let request = RenderRequest {
            source: SourceId::new("catalog/IMG_0001.cr3"),
            document,
            region: NormRect::FULL,
            generation: RenderGeneration::new(4),
            intent: RenderIntent::Draft,
            runtime,
        };
        assert!(!request.runtime.is_empty());
        assert!(request.runtime.fingerprint().is_some());
        // The document inside the request remains producer-free.
        assert!(request.document.adjustments().is_default());
    }

    #[test]
    fn legacy_json_with_vendor_keys_round_trips_without_loss() {
        // No `nodes` field (older document) and an unknown vendor key in
        // `extra`; both must survive a round trip.
        let legacy = r#"{"exposure":{"exposure":0.25},"extra":{"VendorThing":12.5}}"#;
        let parsed: AdjustmentSet = serde_json::from_str(legacy).expect("legacy document parses");
        assert_eq!(parsed.exposure.exposure, 0.25);
        assert_eq!(parsed.extra.get("VendorThing"), Some(&12.5));
        assert!(parsed.nodes.is_none(), "absent nodes reads as None");

        let round = serde_json::to_string(&parsed).expect("serializes");
        let again: AdjustmentSet = serde_json::from_str(&round).expect("round trip parses");
        assert_eq!(again, parsed);
        assert_eq!(again.extra.get("VendorThing"), Some(&12.5));
        assert!(again.nodes.is_none());
    }

    #[test]
    fn document_snapshot_clones_share_the_document_arc() {
        let mut adj = AdjustmentSet::default();
        adj.exposure.exposure = 1.0;
        let snapshot =
            DocumentSnapshot::from_adjustments(adj, DocumentRevision::new(3));
        let clone = snapshot.clone();

        assert!(Arc::ptr_eq(snapshot.shared(), clone.shared()));
        assert_eq!(snapshot.revision(), DocumentRevision::new(3));
        assert_eq!(snapshot.adjustments().exposure.exposure, 1.0);
        assert_eq!(snapshot.document_fingerprint(), clone.document_fingerprint());
    }

    #[test]
    fn document_snapshot_accepts_an_existing_arc() {
        let arc = Arc::new(AdjustmentSet::default());
        let snapshot =
            DocumentSnapshot::from_adjustments(Arc::clone(&arc), DocumentRevision::new(1));
        assert!(Arc::ptr_eq(snapshot.shared(), &arc));
    }

    #[test]
    fn identity_newtypes_hold_distinct_values() {
        assert_eq!(SourceId::new("catalog/IMG_0001.cr3"), SourceId::new("catalog/IMG_0001.cr3"));
        assert_ne!(SourceId::new("a"), SourceId::new("b"));
        assert_eq!(SourceId::new(String::from("x")).as_str(), "x");

        assert_eq!(DocumentRevision::new(7).get(), 7);
        assert_ne!(DocumentRevision::new(7), DocumentRevision::new(8));
        assert_eq!(DocumentRevision::new(7).next(), DocumentRevision::new(8));
        assert_eq!(DocumentRevision::new(u64::MAX).next().get(), u64::MAX);

        assert_eq!(RenderGeneration::new(7).get(), 7);
        assert_ne!(RenderGeneration::new(7), RenderGeneration::new(8));

        assert_eq!(RenderIntent::default(), RenderIntent::Draft);
        assert_ne!(RenderIntent::Draft, RenderIntent::Export);
    }

    /// `SourceId("7")` and `DocumentRevision(7)` carry the same number but are
    /// different axes; their wire forms and accessors show they are not
    /// interchangeable.
    #[test]
    fn source_id_and_document_revision_are_not_interchangeable() {
        let source = SourceId::new("7");
        let revision = DocumentRevision::new(7);
        // Same underlying value...
        assert_eq!(source.as_str().parse::<u64>().unwrap(), 7);
        assert_eq!(revision.get(), 7);
        // ...but a string source id and a numeric revision never share a wire
        // form, so one cannot be read back as the other.
        assert_eq!(serde_json::to_string(&source).unwrap(), r#""7""#);
        assert_eq!(serde_json::to_string(&revision).unwrap(), "7");
    }

    #[test]
    fn document_schema_version_matches_store() {
        // `crate::store` is private; this is the same literal as
        // `store::SCHEMA_VERSION`. If store.rs bumps, update both.
        assert_eq!(DOCUMENT_SCHEMA_VERSION, 1);
    }
}
