//! Assets that may not be published with the source.
//!
//! Decoded third-party LUTs, meshes and presets, and fixtures captured from
//! third-party software, live in the gitignored `private-assets/` directory
//! (see `private-assets.md` at the workspace root). The build embeds them only
//! when that directory exists; otherwise [`private_asset!`] yields `None` and
//! the stage that needs the asset must refuse rather than run without it.
//!
//! Tests that need one are marked
//! `#[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]`,
//! so a run without the directory reports them as ignored, never as passed.

use std::path::PathBuf;

/// The private asset directory this crate was built against, if any.
pub const ROOT: Option<&str> = option_env!("AURORA_PRIVATE_ASSETS_DIR");

/// Whether this build embedded the private assets.
pub const AVAILABLE: bool = ROOT.is_some();

/// `rel` inside the private asset directory, for assets read at run time.
/// `AURORA_PRIVATE_ASSETS` at run time overrides the build-time directory.
pub fn path(rel: &str) -> Option<PathBuf> {
    let root = std::env::var_os("AURORA_PRIVATE_ASSETS").map(PathBuf::from).or_else(|| ROOT.map(PathBuf::from))?;
    Some(root.join(rel))
}

/// `Option<&'static [u8]>`: the bytes of `$rel` inside `private-assets/`,
/// embedded at build time, or `None` in a build without the directory.
///
/// The calling crate's build script must run `build-support/private_assets.rs`.
#[macro_export]
macro_rules! private_asset {
    ($rel:literal) => {{
        #[cfg(aurora_private_assets)]
        let asset: ::core::option::Option<&'static [u8]> =
            ::core::option::Option::Some(include_bytes!(concat!(env!("AURORA_PRIVATE_ASSETS_DIR"), "/", $rel)));
        #[cfg(not(aurora_private_assets))]
        let asset: ::core::option::Option<&'static [u8]> = ::core::option::Option::None;
        asset
    }};
}

/// Like [`private_asset!`] for UTF-8 text.
#[macro_export]
macro_rules! private_asset_str {
    ($rel:literal) => {{
        #[cfg(aurora_private_assets)]
        let asset: ::core::option::Option<&'static str> =
            ::core::option::Option::Some(include_str!(concat!(env!("AURORA_PRIVATE_ASSETS_DIR"), "/", $rel)));
        #[cfg(not(aurora_private_assets))]
        let asset: ::core::option::Option<&'static str> = ::core::option::Option::None;
        asset
    }};
}

/// `asset` or an empty slice, for `const` items: a test reading an empty
/// fixture fails its own length checks, and is marked as needing the assets.
pub const fn or_empty(asset: Option<&'static [u8]>) -> &'static [u8] {
    match asset {
        Some(bytes) => bytes,
        None => &[],
    }
}
