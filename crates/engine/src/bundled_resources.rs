//! Self-contained resource discovery for Aurora Studio.
//!
//! All runtime models, camera curves, color profiles, and portrait assets live
//! beside `aurora-studio.exe` (`models/`, `assets/camera/`, `assets/portrait/`).
//! No external installation directory (`Program Files\Capture One`, `04_reference`,
//! `90_work/build-cache`, or `CARGO_MANIFEST_DIR`) is queried at runtime.

use std::path::{Path, PathBuf};

/// Directory containing the running executable.
pub(crate) fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
}

/// Self-contained roots searched for `AIModels/...` (ONNX models).
pub(crate) fn ai_model_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(dir) = std::env::var_os("AURORA_C1_DIR") {
        roots.push(PathBuf::from(dir));
    }
    if let Some(dir) = exe_dir() {
        roots.push(dir.join("models"));
        roots.push(dir);
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd.join("models"));
    }
    roots
}

/// Bundled `coreConfig.cdx` path (`assets/camera/coreConfig.cdx` beside the exe).
pub(crate) fn core_config_path() -> PathBuf {
    for var in ["AURORA_CORECONFIG", "AURORA_CORE_CONFIG"] {
        if let Some(path) = std::env::var_os(var) {
            return PathBuf::from(path);
        }
    }
    if let Some(dir) = exe_dir() {
        for rel in ["assets/camera/coreConfig.cdx", "coreConfig.cdx"] {
            let candidate = dir.join(rel);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let candidate = cwd.join("assets/camera/coreConfig.cdx");
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from("testdata/coreConfig.cdx")
}

/// Bundled `Film Curves/` directory (`assets/camera/Film Curves` beside the exe).
pub(crate) fn film_curves_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("AURORA_FILM_CURVES") {
        return PathBuf::from(dir);
    }
    if let Some(dir) = exe_dir() {
        for rel in ["assets/camera/Film Curves", "Film Curves"] {
            let candidate = dir.join(rel);
            if candidate.is_dir() {
                return candidate;
            }
        }
        return dir.join("assets/camera/Film Curves");
    }
    PathBuf::from("assets/camera/Film Curves")
}

/// Bundled camera asset root (`assets/camera` beside the exe, holding
/// `coreConfig.cdx`, `Film Curves/`, `Color Profiles/`, `banding_tables.bin`).
pub(crate) fn camera_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("AURORA_Z6III_C1_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(dir) = exe_dir() {
        let bundled = dir.join("assets/camera");
        if bundled.is_dir() {
            return bundled;
        }
        return dir;
    }
    PathBuf::from("assets/camera")
}

/// Bundled 1,536-byte banding LUT table (`assets/camera/banding_tables.bin`).
pub(crate) fn banding_tables_path() -> Option<PathBuf> {
    if let Some(dir) = exe_dir() {
        for rel in ["assets/camera/banding_tables.bin", "banding_tables.bin"] {
            let candidate = dir.join(rel);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let candidate = cwd.join("assets/camera/banding_tables.bin");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Bundled blemish ONNX model path (`models/AIModels/Common/BlemishRemoval/blemishRemoval.onnx`).
pub(crate) fn blemish_model_path() -> PathBuf {
    if let Some(path) = std::env::var_os("AURORA_BLEMISH_MODEL") {
        return PathBuf::from(path);
    }
    for root in ai_model_roots() {
        let candidate = root.join("AIModels/Common/BlemishRemoval/blemishRemoval.onnx");
        if candidate.is_file() {
            return candidate;
        }
    }
    exe_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("models/AIModels/Common/BlemishRemoval/blemishRemoval.onnx")
}

/// Bundled landmark model root (`models/landmark` beside the exe).
pub(crate) fn face_landmark_root() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(dir) = exe_dir() {
        candidates.push(dir.join("models").join("landmark"));
        candidates.push(dir.join("landmark"));
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("models").join("landmark"));
    }
    candidates.into_iter().find(|root| {
        root.join("assets").join("1063025060.onnx").is_file()
            && root.join("assets").join("323773030.onnx").is_file()
            && root.join("constants.json").is_file()
    })
}

/// Bundled portrait makeup asset directory (`assets/portrait/makeup` or `models/makeup`).
pub(crate) fn makeup_asset_root() -> Option<PathBuf> {
    if let Some(dir) = exe_dir() {
        for rel in ["assets/portrait/makeup", "models/makeup"] {
            let candidate = dir.join(rel);
            if candidate.is_dir() {
                return Some(candidate);
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        for rel in ["assets/portrait/makeup", "models/makeup"] {
            let candidate = cwd.join(rel);
            if candidate.is_dir() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Bundled PF8033 smile CPU-forward receipt (`assets/portrait/smile/pf8033-cpu-forward.json`).
pub(crate) fn pf8033_receipt_path() -> Option<PathBuf> {
    if let Some(dir) = exe_dir() {
        let candidate = dir.join("assets/portrait/smile/pf8033-cpu-forward.json");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let candidate = cwd.join("assets/portrait/smile/pf8033-cpu-forward.json");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}
