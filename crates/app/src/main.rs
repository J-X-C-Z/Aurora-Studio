#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! Aurora Studio's desktop entry point. `--touch` (or `AURORA_TOUCH=1`) shows
//! the phone and tablet interface in a phone-sized window.

#[cfg(not(any(target_os = "android", target_os = "ios")))]
fn main() {
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    let mut touch = std::env::var_os("AURORA_TOUCH").is_some();
    let mut selfcheck = std::env::var_os("AURORA_GPU_SELFCHECK").is_some();
    for arg in std::env::args_os().skip(1) {
        if arg == "--touch" {
            touch = true;
        } else if arg == "--gpu-selfcheck" {
            selfcheck = true;
        } else {
            paths.push(arg.into())
        }
    }
    if selfcheck {
        match aurora_engine::aimask::gpu_selfcheck() {
            Ok(()) => return,
            Err(error) => {
                eprintln!("GPU selfcheck failed: {error:#}");
                std::process::exit(1);
            }
        }
    }
    portrait_app::run(gpui_kit::application(), portrait_app::Launch { paths, touch });
}

// Mobile hosts link the library and call `portrait_app::run` with their platform.
#[cfg(any(target_os = "android", target_os = "ios"))]
fn main() {}
