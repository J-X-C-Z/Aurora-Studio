// Shared by the build scripts of aurora-engine and aurora-portrait.
//
// Data that may not be published (decoded third-party assets, Capture One
// extracts, captured fixtures) lives outside Git in `private-assets/` at the
// workspace root, or wherever `AURORA_PRIVATE_ASSETS` points. When that
// directory exists the build embeds from it and sets `cfg(aurora_private_assets)`;
// without it every private asset reads as `None` and the features that need one
// refuse instead of running. See `private-assets.md`.

fn private_assets() {
    use std::path::PathBuf;
    println!("cargo::rustc-check-cfg=cfg(aurora_private_assets)");
    println!("cargo::rerun-if-env-changed=AURORA_PRIVATE_ASSETS");
    let dir = match std::env::var_os("AURORA_PRIVATE_ASSETS") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"))
            .join("../../private-assets"),
    };
    println!("cargo::rerun-if-changed={}", dir.display());
    if dir.is_dir() {
        // `absolute` rather than `canonicalize`: no `\\?\` prefix on Windows,
        // which `include_bytes!` would not accept mixed with `/`.
        let dir = std::path::absolute(&dir).expect("absolute private asset path");
        println!("cargo::rustc-cfg=aurora_private_assets");
        println!("cargo::rustc-env=AURORA_PRIVATE_ASSETS_DIR={}", dir.display());
    }
}
