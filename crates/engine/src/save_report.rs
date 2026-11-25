//! Per-document results for a batch save.
//!
//! A batch save writes one adjustment file per image and then one catalog
//! index. Those are separate atomic writes, not one transaction: each file is
//! individually all-or-nothing (see `store::write_atomic`), but a batch can
//! still end with document A on disk, document B failed, and the index stale.
//! Callers need to know which documents reached disk, which failed and why,
//! and whether the index still reflects the catalog, instead of a single "ok"
//! that hides a partially failed batch.
//!
//! [`SaveOutcome::Superseded`] is deliberately not a failure: a newer save for
//! the same document had already been queued/written, so this older write was
//! skipped to avoid overwriting newer bytes. The newer write owns the result.
//!
//! An index failure is reported separately in [`SaveReport::index_error`]:
//! every adjustment file may be on disk while the catalog still carries the
//! previous metadata for those images.

use crate::catalog::ImageId;

/// What happened to one document in a batch save.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SaveOutcome {
    /// The document's adjustment file was written to disk and atomically
    /// renamed into place.
    Saved,
    /// The write failed; `error` is the underlying message.
    Failed { error: String },
    /// A newer save for the same relative path won the race, so this older
    /// write was deliberately not performed. Not an error.
    Superseded,
}

impl SaveOutcome {
    /// The error text for a failed write, `None` otherwise.
    pub fn error(&self) -> Option<&str> {
        match self {
            SaveOutcome::Failed { error } => Some(error),
            _ => None,
        }
    }
}

/// One document's result in a batch save.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentSaveResult {
    /// The image variant the edit belongs to.
    pub image: ImageId,
    /// Path relative to the library root, as stored in the adjustment file
    /// name. Empty when the image id was already unknown to the engine.
    pub path: String,
    pub outcome: SaveOutcome,
}

impl DocumentSaveResult {
    pub fn saved(image: ImageId, path: impl Into<String>) -> Self {
        DocumentSaveResult { image, path: path.into(), outcome: SaveOutcome::Saved }
    }

    pub fn failed(image: ImageId, path: impl Into<String>, error: impl Into<String>) -> Self {
        DocumentSaveResult {
            image,
            path: path.into(),
            outcome: SaveOutcome::Failed { error: error.into() },
        }
    }

    pub fn superseded(image: ImageId, path: impl Into<String>) -> Self {
        DocumentSaveResult { image, path: path.into(), outcome: SaveOutcome::Superseded }
    }

    pub fn is_saved(&self) -> bool {
        matches!(self.outcome, SaveOutcome::Saved)
    }
}

/// The per-document results of one batch save, plus the index result.
///
/// This is not a multi-file transaction: a non-empty `documents` list with
/// `index_error: None` means each listed file is on disk and the catalog was
/// rewritten afterwards. A failed document does not roll back the others.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SaveReport {
    pub documents: Vec<DocumentSaveResult>,
    /// Why `catalog.json` could not be rewritten, if it was attempted and
    /// failed. Document writes are independent of this.
    pub index_error: Option<String>,
}

impl SaveReport {
    pub fn new(documents: Vec<DocumentSaveResult>, index_error: Option<String>) -> Self {
        SaveReport { documents, index_error }
    }

    /// Number of documents written this batch.
    pub fn saved(&self) -> usize {
        self.documents.iter().filter(|d| d.is_saved()).count()
    }

    /// Number of documents whose write failed.
    pub fn failed(&self) -> usize {
        self.documents.iter().filter(|d| d.outcome.error().is_some()).count()
    }

    /// Number of documents skipped because a newer save had already won.
    pub fn superseded(&self) -> usize {
        self.documents
            .iter()
            .filter(|d| matches!(d.outcome, SaveOutcome::Superseded))
            .count()
    }

    /// True when no document failed and the index was written. Superseded
    /// documents count as complete: a newer write owns their bytes.
    pub fn is_complete(&self) -> bool {
        self.failed() == 0 && self.index_error.is_none()
    }

    /// The first failed document, for a concise diagnostic.
    pub fn first_failure(&self) -> Option<&DocumentSaveResult> {
        self.documents.iter().find(|d| d.outcome.error().is_some())
    }

    /// Turns a partial batch into an error with counts and the first failure.
    ///
    /// Returns `None` when the batch is complete. A document failure and an
    /// index failure are both reported, because the caller must know about
    /// either one; they stay distinguishable in the message.
    pub fn into_error(self) -> Option<anyhow::Error> {
        if self.is_complete() {
            return None;
        }
        let mut parts = Vec::new();
        let failed = self.failed();
        if failed > 0 {
            parts.push(format!(
                "{failed} of {} adjustment file(s) failed to save",
                self.documents.len()
            ));
            if let Some(first) = self.first_failure() {
                let error = first.outcome.error().unwrap_or_default();
                if first.path.is_empty() {
                    parts.push(format!("first failure: {error}"));
                } else {
                    parts.push(format!("first failure {}: {error}", first.path));
                }
            }
        }
        if let Some(index) = &self.index_error {
            parts.push(format!("catalog index not saved: {index}"));
        }
        Some(anyhow::anyhow!(parts.join("; ")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregates_saved_failed_and_superseded() {
        let report = SaveReport::new(
            vec![
                DocumentSaveResult::saved(ImageId(1), "a.cr3"),
                DocumentSaveResult::failed(ImageId(2), "b.cr3", "disk full"),
                DocumentSaveResult::superseded(ImageId(3), "c.cr3"),
            ],
            None,
        );
        assert_eq!(report.saved(), 1);
        assert_eq!(report.failed(), 1);
        assert_eq!(report.superseded(), 1);
        assert!(!report.is_complete());
        assert_eq!(report.first_failure().map(|d| d.path.as_str()), Some("b.cr3"));
    }

    #[test]
    fn superseded_documents_are_complete_and_produce_no_error() {
        let report = SaveReport::new(
            vec![DocumentSaveResult::superseded(ImageId(1), "a.cr3")],
            None,
        );
        assert!(report.is_complete());
        assert!(report.into_error().is_none());
    }

    #[test]
    fn index_failure_alone_is_incomplete_and_names_the_index() {
        let report = SaveReport::new(
            vec![DocumentSaveResult::saved(ImageId(1), "a.cr3")],
            Some("permission denied".into()),
        );
        assert!(!report.is_complete());
        let message = report.into_error().expect("index failure is an error").to_string();
        assert!(message.contains("catalog index not saved"), "{message}");
        assert!(message.contains("permission denied"), "{message}");
        assert!(!message.contains("adjustment file"), "{message}");
    }

    #[test]
    fn error_message_has_counts_and_first_failure() {
        let report = SaveReport::new(
            vec![
                DocumentSaveResult::failed(ImageId(1), "first.cr3", "read-only"),
                DocumentSaveResult::failed(ImageId(2), "second.cr3", "gone"),
                DocumentSaveResult::saved(ImageId(3), "third.cr3"),
            ],
            Some("index locked".into()),
        );
        let message = report.into_error().expect("partial batch is an error").to_string();
        assert!(message.contains("2 of 3 adjustment file(s) failed"), "{message}");
        assert!(message.contains("first failure first.cr3: read-only"), "{message}");
        assert!(message.contains("catalog index not saved: index locked"), "{message}");
    }
}
