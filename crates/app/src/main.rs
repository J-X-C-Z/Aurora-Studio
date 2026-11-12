//! Aurora Studio's desktop entry point. `--touch` (or `AURORA_TOUCH=1`) shows
//! the phone and tablet interface in a phone-sized window.

#[cfg(not(any(target_os = "android", target_os = "ios")))]
fn main() {
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    let mut touch = std::env::var_os("AURORA_TOUCH").is_some();
    for arg in std::env::args_os().skip(1) {
        if arg == "--touch" { touch = true } else { paths.push(arg.into()) }
    }
    lumen_app::run(gpui_kit::application(), lumen_app::Launch { paths, touch });
}

// Mobile hosts link the library and call `lumen_app::run` with their platform.
#[cfg(any(target_os = "android", target_os = "ios"))]
fn main() {}
