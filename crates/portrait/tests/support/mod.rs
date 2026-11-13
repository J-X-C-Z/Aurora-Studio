//! Opt-in local fixtures; no fixture data ships with this crate.
#![allow(dead_code)]

pub fn root() -> std::path::PathBuf {
    std::env::var_os("PORTRAIT_EXTERNAL_FIXTURES")
        .map(std::path::PathBuf::from)
        .expect("external-fixtures requires PORTRAIT_EXTERNAL_FIXTURES")
}

pub fn read(relative: &str) -> Vec<u8> {
    let path = root().join(relative);
    std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}
