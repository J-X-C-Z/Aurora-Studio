//! Cancellation-safe atomic publisher for staged exports.
//!
//! [`AtomicPublisher`] writes into a hidden staging directory inside the
//! destination volume (reusing [`crate::output::StagedExport`]), validates
//! written byte counts, polls [`CancelToken`] at every safe boundary, and
//! atomically publishes only complete, synced files. If cancelled or failed at
//! any point, the staging folder is dropped and any existing destination files
//! remain byte-for-byte intact.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::job_lifecycle::CancelToken;
use crate::output::{contained_output, resolve_output_dir, sweep_stale_staging, StagedExport};

/// Typed failure returned by [`AtomicPublisher`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PublishError {
    /// Cancellation was observed at a safe boundary before publish.
    Cancelled,
    /// Destination path is empty, not a directory, or cannot be prepared.
    InvalidDestination(String),
    /// Output file name is empty or attempts directory traversal.
    InvalidFileName(String),
    /// The staged file ended up empty (zero bytes).
    EmptyPayload,
    /// Written byte count did not match the expected payload size.
    IncompleteWrite { expected: u64, actual: u64 },
    /// Underlying filesystem error.
    Io(String),
}

impl fmt::Display for PublishError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => f.write_str("export publish cancelled"),
            Self::InvalidDestination(reason) => {
                write!(f, "export publish: invalid destination: {reason}")
            }
            Self::InvalidFileName(reason) => {
                write!(f, "export publish: invalid output file name: {reason}")
            }
            Self::EmptyPayload => f.write_str("export publish: staged file is empty"),
            Self::IncompleteWrite { expected, actual } => {
                write!(
                    f,
                    "export publish: incomplete write (expected {expected} bytes, got {actual})"
                )
            }
            Self::Io(reason) => write!(f, "export publish I/O error: {reason}"),
        }
    }
}

impl std::error::Error for PublishError {}

/// Successful publish receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublishOutcome {
    /// Final published path inside the destination directory.
    pub final_path: PathBuf,
    /// Verified byte count of the published file.
    pub bytes_written: u64,
}

/// Atomic staging and publication helper with cooperative cancellation checks.
#[derive(Clone, Debug)]
pub struct AtomicPublisher {
    destination: PathBuf,
    tag: String,
    cancel: CancelToken,
}

impl AtomicPublisher {
    /// Creates a publisher bound to `destination`, validating and canonicalizing
    /// the directory up front.
    pub fn new(
        destination: &Path,
        tag: impl Into<String>,
        cancel: CancelToken,
    ) -> Result<Self, PublishError> {
        if cancel.is_cancelled() {
            return Err(PublishError::Cancelled);
        }
        let canonical = resolve_output_dir(destination)
            .map_err(|err| PublishError::InvalidDestination(err.to_string()))?;
        let tag = tag.into();
        let tag = if tag.trim().is_empty() { "job".to_string() } else { tag };
        Ok(Self { destination: canonical, tag, cancel })
    }

    /// Canonical destination directory.
    pub fn destination(&self) -> &Path {
        &self.destination
    }

    /// Cooperative cancellation token associated with this publisher.
    pub fn cancel_token(&self) -> &CancelToken {
        &self.cancel
    }

    /// Removes stale staging directories left behind by crashed runs.
    pub fn sweep_stale(&self) -> usize {
        sweep_stale_staging(&self.destination)
    }

    /// Stages and atomically publishes a complete in-memory byte slice.
    pub fn publish_bytes(
        &self,
        file_name: &str,
        bytes: &[u8],
    ) -> Result<PublishOutcome, PublishError> {
        if bytes.is_empty() {
            return Err(PublishError::EmptyPayload);
        }
        let expected = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        self.publish_with(file_name, Some(expected), |file, cancel| {
            if cancel.is_cancelled() {
                return Err(PublishError::Cancelled);
            }
            file.write_all(bytes)
                .map_err(|err| PublishError::Io(err.to_string()))
        })
    }

    /// Stages a file via `write_fn`, verifies completion and cancellation at
    /// safe boundaries, syncs the file, and atomically publishes it into
    /// `self.destination`.
    ///
    /// Guarantees:
    /// - If `self.cancel` is triggered before, during, or right after writing,
    ///   returns [`PublishError::Cancelled`] and removes the staging folder.
    /// - Never publishes a partial/empty file.
    /// - Never overwrites or damages an existing file at `destination`:
    ///   [`StagedExport::publish`] claims a free name with no-clobber semantics.
    pub fn publish_with<F>(
        &self,
        file_name: &str,
        expected_bytes: Option<u64>,
        write_fn: F,
    ) -> Result<PublishOutcome, PublishError>
    where
        F: FnOnce(&mut File, &CancelToken) -> Result<(), PublishError>,
    {
        if self.cancel.is_cancelled() {
            return Err(PublishError::Cancelled);
        }

        // Reject traversal or multi-component names before creating staging.
        let validated_target = contained_output(&self.destination, Path::new(file_name))
            .map_err(|err| PublishError::InvalidFileName(err.to_string()))?;
        let safe_name = validated_target
            .file_name()
            .ok_or_else(|| PublishError::InvalidFileName("missing file name".into()))?;

        let staged = StagedExport::new(&self.destination, &self.tag)
            .map_err(|err| PublishError::Io(err.to_string()))?;
        let staged_file = staged.dir().join(safe_name);

        {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&staged_file)
                .map_err(|err| PublishError::Io(err.to_string()))?;

            write_fn(&mut file, &self.cancel)?;

            if self.cancel.is_cancelled() {
                return Err(PublishError::Cancelled);
            }

            file.flush().map_err(|err| PublishError::Io(err.to_string()))?;
            file.sync_all().map_err(|err| PublishError::Io(err.to_string()))?;
        }

        if self.cancel.is_cancelled() {
            return Err(PublishError::Cancelled);
        }

        let actual = std::fs::metadata(&staged_file)
            .map_err(|err| PublishError::Io(err.to_string()))?
            .len();
        if actual == 0 {
            return Err(PublishError::EmptyPayload);
        }
        if let Some(expected) = expected_bytes {
            if actual != expected {
                return Err(PublishError::IncompleteWrite { expected, actual });
            }
        }

        if self.cancel.is_cancelled() {
            return Err(PublishError::Cancelled);
        }

        let final_path = staged
            .publish(&staged_file)
            .map_err(|err| PublishError::Io(err.to_string()))?;

        Ok(PublishOutcome { final_path, bytes_written: actual })
    }
}
