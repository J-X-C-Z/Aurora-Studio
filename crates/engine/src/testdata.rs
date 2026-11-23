//! Runtime access to `crates/engine/testdata`.
//!
//! That directory is git-ignored, so `include_bytes!` anywhere in it makes a
//! clean checkout fail to compile. Tests read the fixture at run time and skip
//! when it is absent; `lens_correct` simply has no built-in lens DB then.

pub(crate) fn path(relative: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

/// Reads one fixture, printing a single line and returning `None` when missing.
pub(crate) fn bytes(relative: &str) -> Option<Vec<u8>> {
    let path = path(relative);
    match std::fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(err) => {
            eprintln!("testdata: skipping, {} is unavailable ({err})", path.display());
            None
        }
    }
}
