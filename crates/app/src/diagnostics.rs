//! A GUI-subsystem process has no console when launched from Explorer/Finder.
//! Leave startup and renderer failures in a local, bounded diagnostic log.
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, Once};

struct Logger(Mutex<Option<File>>);
static LOGGER: Logger = Logger(Mutex::new(None));
static INIT: Once = Once::new();

fn log_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("AURORA_LOG_FILE").filter(|p| !p.is_empty()) { return Some(path.into()); }
    #[cfg(windows)]
    { std::env::var_os("LOCALAPPDATA").map(|p| PathBuf::from(p).join("Aurora Studio/Logs/aurora-studio.log")) }
    #[cfg(target_os = "macos")]
    { std::env::var_os("HOME").map(|p| PathBuf::from(p).join("Library/Logs/Aurora Studio/aurora-studio.log")) }
    #[cfg(not(any(windows, target_os = "macos")))]
    { None }
}

fn open_log() -> Option<File> {
    let path = log_path()?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) { std::fs::create_dir_all(parent).ok()?; }
    // Rotate on startup, not from render/preview worker threads.
    if path.metadata().is_ok_and(|m| m.len() > 2 * 1024 * 1024) {
        let _ = std::fs::rename(&path, path.with_extension("previous.log"));
    }
    OpenOptions::new().create(true).append(true).open(path).ok()
}

impl log::Log for Logger {
    fn enabled(&self, m: &log::Metadata) -> bool { m.level() <= log::max_level() }
    fn log(&self, r: &log::Record) {
        if !self.enabled(r.metadata()) { return; }
        let _ = writeln!(std::io::stderr().lock(), "[{}] {}: {}", r.level(), r.target(), r.args());
        if let Ok(mut guard) = self.0.lock() && let Some(f) = guard.as_mut() {
            let _ = writeln!(f, "[{}] {}: {}", r.level(), r.target(), r.args());
        }
    }
    fn flush(&self) {
        if let Ok(mut guard) = self.0.lock() && let Some(f) = guard.as_mut() { let _ = f.flush(); }
    }
}

/// Install before GPUI constructs its Windows renderer, and at most once.
pub fn init() {
    INIT.call_once(|| {
        let level = match std::env::var("RUST_LOG").as_deref() {
            Ok("debug") => log::LevelFilter::Debug, Ok("info") => log::LevelFilter::Info,
            Ok("error") => log::LevelFilter::Error, _ => log::LevelFilter::Warn,
        };
        if log::set_logger(&LOGGER).is_ok() {
            *LOGGER.0.lock().unwrap_or_else(|e| e.into_inner()) = open_log();
            log::set_max_level(level);
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |err| {
                // Do not reacquire a file mutex held by the panicking thread.
                if let Ok(mut guard) = LOGGER.0.try_lock() && let Some(f) = guard.as_mut() { let _ = writeln!(f, "[PANIC] {err}"); }
                previous(err);
            }));
        }
    });
}
