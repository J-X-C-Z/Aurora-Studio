//! Identity vocabulary: source, revision, generation, algorithm and version.
//!
//! Each identifier is a distinct newtype on purpose. A source is not a path, a
//! document revision is not a render generation, and an algorithm version is
//! not a model version. They advance for different reasons and on different
//! clocks, so they must not be collapsed into one global counter: doing so
//! would make a stale render indistinguishable from a newer document.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Stable identity of one source image.
///
/// The value is an opaque library-level key chosen when the source is first
/// catalogued. It survives the file being renamed or moved and is scoped to the
/// library. A bare filesystem path is *not* a cross-library identity: the same
/// path can name different files on another machine, and the same file can be
/// reached by several paths (symlinks, case differences, remounts).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SourceId(String);

impl SourceId {
    /// Construct a source identity from a library key.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The key as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Revision of a document's editable state.
///
/// Advances whenever the document changes in a way that can alter output
/// (adjustments, masks, layers). It is per-document and has no relation to the
/// render generation counter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DocumentRevision(u64);

impl DocumentRevision {
    /// Construct a document revision.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// The underlying counter value.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The next revision. Saturates rather than wrapping: a document that hit
    /// `u64::MAX` must not silently look like revision 0.
    pub const fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

/// Monotonic generation of a rendered frame.
///
/// Advances each time a new render is requested or displayed. It is used to
/// discard stale results; it is not a document revision and does not imply the
/// document changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RenderGeneration(u64);

impl RenderGeneration {
    /// Construct a render generation.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// The underlying counter value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Version of the CPU reference algorithm for an operation.
///
/// Changing the arithmetic, rounding or tap layout of the reference advances
/// this version. GPU adapters may match a specific algorithm version; a receipt
/// or cache key must say which one produced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AlgorithmVersion(u32);

impl AlgorithmVersion {
    /// Construct an algorithm version.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// The underlying version value.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Version of a model used by an operation.
///
/// Orthogonal to [`AlgorithmVersion`]: the same algorithm can consume a newer
/// model, and the same model can be run by a newer algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelVersion(u32);

impl ModelVersion {
    /// Construct a model version.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// The underlying version value.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Version of the device/backend policy in force.
///
/// Advancing it invalidates cached decisions about which backend was chosen,
/// without touching document or render generations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DevicePolicyVersion(u32);

impl DevicePolicyVersion {
    /// Construct a device policy version.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// The underlying version value.
    pub const fn get(self) -> u32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_types_do_not_share_a_domain() {
        // Equal numeric payloads in different domains stay equal only to
        // themselves. Assigning one type to the other is a compile error, so
        // the guarantee holds at type level as well.
        let revision = DocumentRevision::new(7);
        let generation = RenderGeneration::new(7);
        assert_eq!(revision.get(), generation.get());
        assert_eq!(revision, DocumentRevision::new(7));
        assert_eq!(generation, RenderGeneration::new(7));
        assert_ne!(revision, DocumentRevision::new(8));
        assert_ne!(generation, RenderGeneration::new(8));

        assert_ne!(
            std::any::type_name_of_val(&revision),
            std::any::type_name_of_val(&generation)
        );
        assert_ne!(
            std::any::type_name::<AlgorithmVersion>(),
            std::any::type_name::<ModelVersion>()
        );
        assert_ne!(
            std::any::type_name::<SourceId>(),
            std::any::type_name::<DocumentRevision>()
        );
    }

    #[test]
    fn source_ids_compare_by_exact_key() {
        // A bare path is not a cross-library identity; equality is exact.
        assert_eq!(SourceId::new("library://2026/abc"), SourceId::new("library://2026/abc"));
        assert_ne!(SourceId::new("/photos/a.nef"), SourceId::new("a.nef"));
        assert_eq!(SourceId::new("library://2026/abc").as_str(), "library://2026/abc");
        assert_eq!(SourceId::new("library://2026/abc").to_string(), "library://2026/abc");
    }

    #[test]
    fn versions_order_within_their_own_type() {
        assert!(AlgorithmVersion::new(2) < AlgorithmVersion::new(10));
        assert!(ModelVersion::new(1) < ModelVersion::new(2));
        assert!(DevicePolicyVersion::new(3) > DevicePolicyVersion::new(1));
        assert!(DocumentRevision::new(0) < DocumentRevision::new(1));
        assert!(RenderGeneration::new(41) > RenderGeneration::new(40));
    }

    #[test]
    fn identity_values_serialize_transparently() {
        assert_eq!(
            serde_json::to_string(&DocumentRevision::new(42)).expect("serialize revision"),
            "42"
        );
        assert_eq!(
            serde_json::to_string(&SourceId::new("library://2026/abc")).expect("serialize source"),
            "\"library://2026/abc\""
        );

        let source: SourceId =
            serde_json::from_str("\"library://2026/abc\"").expect("deserialize source");
        assert_eq!(source, SourceId::new("library://2026/abc"));

        let generation: RenderGeneration =
            serde_json::from_str("9").expect("deserialize generation");
        assert_eq!(generation, RenderGeneration::new(9));
    }
}
