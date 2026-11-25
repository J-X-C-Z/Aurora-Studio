//! Aurora Studio: a professional raw editor interface on GPUI.

mod actions;
mod capability;
mod contract;
mod evidence;
mod i18n;
mod motion;
mod perf;
mod planned;
mod state;
mod theme;
mod tools;
mod ui;
mod widgets;

use std::sync::atomic::{AtomicBool, Ordering};

static FIRST_PREVIEW_READY: AtomicBool = AtomicBool::new(false);

/// What a host hands the interface when it starts.
#[derive(Default)]
pub struct Launch {
    /// A folder or a Capture One catalog/session to open, and loose files to import.
    pub paths: Vec<std::path::PathBuf>,
    /// Show the touch interface for phones and tablets instead of the desktop
    /// workspace. Always on for Android and iOS builds.
    pub touch: bool,
}

/// File URLs handed over by AppKit before the app state exists, plus the handles
/// used to import the ones that arrive later (`open -a`, Open With, Finder).
#[cfg(target_os = "macos")]
#[derive(Default)]
struct OpenUrlSink {
    app: Option<gpui_kit::AsyncApp>,
    state: Option<gpui_kit::Entity<state::AppState>>,
    pending: Vec<std::path::PathBuf>,
}

/// `file:///Users/me/Photos/a%20b.jpg` → `/Users/me/Photos/a b.jpg`.
#[cfg(target_os = "macos")]
fn file_url_to_path(url: &str) -> Option<std::path::PathBuf> {
    let rest = url.strip_prefix("file://")?;
    // Drop an authority component (`localhost` or a host name) when present;
    // the path itself always starts with `/`.
    let path = if rest.starts_with('/') {
        rest
    } else {
        &rest[rest.find('/')?..]
    };
    let bytes = percent_decode(path.as_bytes());
    let decoded = String::from_utf8(bytes).ok()?;
    (!decoded.is_empty()).then(|| std::path::PathBuf::from(decoded))
}

#[cfg(target_os = "macos")]
fn percent_decode(input: &[u8]) -> Vec<u8> {
    fn hex(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        if input[i] == b'%'
            && i + 2 < input.len()
            && let (Some(high), Some(low)) = (hex(input[i + 1]), hex(input[i + 2]))
        {
            out.push(high * 16 + low);
            i += 3;
        } else {
            out.push(input[i]);
            i += 1;
        }
    }
    out
}

#[cfg(target_os = "macos")]
fn import_opened(
    state: &gpui_kit::Entity<state::AppState>,
    paths: Vec<std::path::PathBuf>,
    cx: &mut gpui_kit::App,
) {
    state.update(cx, |state, cx| {
        state.welcome = false;
        state.import(paths, cx);
    });
}

/// Runs the interface on `app`. Desktop builds pass `gpui_kit::application()`;
/// a mobile host passes `Application::with_platform` over its own platform.
pub fn run(app: gpui_kit::Application, launch: Launch) {
    env_logger_init();
    let touch = launch.touch || gpui_kit::is_mobile();
    theme::set_touch(touch);
    let args = launch.paths;
    prewarm_in_background();

    // A folder or a Capture One catalog/session given on the command line opens
    // directly; otherwise the welcome screen offers recent ones (or reopens the
    // last, if the preferences say so), with the demo images behind it.
    let target = args.iter().find(|p| p.is_dir() || state::open::is_catalog(p)).cloned().or_else(|| {
        let settings = state::Settings::load();
        settings.reopen_last.then(|| settings.recent.first().cloned()).flatten().filter(|p| p.exists())
    });
    let mut error = None;
    let (engine, source) = match &target {
        Some(path) => state::open::open(path).unwrap_or_else(|err| {
            log::error!("cannot open {}: {err}", path.display());
            error = Some(format!("{}: {err}", path.display()));
            state::open::demo()
        }),
        None => state::open::demo(),
    };
    let files: Vec<_> = args.into_iter().filter(|p| p.is_file() && !state::open::is_catalog(p)).collect();

    // macOS delivers Finder / Open With / `open -a` paths through
    // `application:openURLs:`, not argv. Register before `run`; URLs that arrive
    // while launching are queued and imported once the app state exists.
    #[cfg(target_os = "macos")]
    let open_urls = {
        let sink = std::rc::Rc::new(std::cell::RefCell::new(OpenUrlSink::default()));
        app.on_open_urls({
            let sink = sink.clone();
            move |urls| {
                let paths: Vec<_> = urls.iter().filter_map(|url| file_url_to_path(url)).collect();
                if paths.is_empty() {
                    return;
                }
                let mut sink = sink.borrow_mut();
                match (sink.app.clone(), sink.state.clone()) {
                    (Some(app), Some(state)) => {
                        drop(sink);
                        app.update(|cx| import_opened(&state, paths, cx));
                    }
                    _ => sink.pending.extend(paths),
                }
            }
        });
        sink
    };

    app.with_assets(ely_core::Assets::before(gpui_kit::assets::AllAssets))
        .run(move |cx| {
            gpui_kit::init(cx);
            if let Err(err) = ely_core::init(cx) {
                log::error!("ely: failed to initialize: {err}");
            }
            theme::init(cx);
            actions::bind_keys(cx, &Default::default());
            let quit_engine = engine.clone();
            cx.on_action(move |_: &actions::Quit, cx| {
                // Bounded close flush: queued saves must not be dropped just
                // because the window closed. A failure is logged; the
                // workspace's own Quit handler also shows it in the UI.
                if let Some(report) =
                    quit_engine.flush_pending_saves(std::time::Duration::from_secs(2))
                {
                    if !report.is_complete() {
                        log::warn!(
                            "save queue incomplete at quit: {} saved, {} failed, {} not written, index_error={:?}",
                            report.saved(),
                            report.failed(),
                            report.not_written(),
                            report.index_error,
                        );
                    }
                }
                cx.quit();
            });
            // macOS application-menu actions. They are no-ops elsewhere.
            cx.on_action(|_: &actions::Hide, cx| cx.hide());
            cx.on_action(|_: &actions::HideOthers, cx| cx.hide_other_apps());
            cx.on_action(|_: &actions::ShowAll, cx| cx.unhide_other_apps());
            cx.activate(true);

            let state = state::init_state(engine.clone(), source.clone(), cx);
            #[cfg(target_os = "macos")]
            {
                let queued = {
                    let mut sink = open_urls.borrow_mut();
                    sink.app = Some(cx.to_async());
                    sink.state = Some(state.clone());
                    std::mem::take(&mut sink.pending)
                };
                if !queued.is_empty() {
                    import_opened(&state, queued, cx);
                }
            }
            state.update(cx, |s, cx| {
                s.open_error = error.clone();
                // Loose files on the command line are imported into what opened.
                if !files.is_empty() {
                    s.welcome = false;
                    s.import(files.clone(), cx);
                }
            });
            if touch {
                ui::mobile::open(state, cx);
            } else {
                ui::workspace::open(state, cx);
            }
        });
}

fn prewarm_in_background() {
    if gpui_kit::is_mobile()
        || std::env::var("AURORA_AI_PREWARM").is_ok_and(|value| value == "0")
    {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("aurora-prewarm".into())
        .spawn(|| {
            lower_current_thread_priority();
            if std::env::var_os("AURORA_GPU_SELFCHECK").is_some() {
                match aurora_engine::aimask::gpu_selfcheck() {
                    Ok(()) => log::info!("GPU selfcheck finished"),
                    Err(error) => log::warn!("GPU selfcheck failed: {error:#}"),
                }
            }
            let gpu_start = std::time::Instant::now();
            aurora_engine::prewarm_gpu();
            log::info!(
                "detail GPU prewarm finished in {:.0} ms",
                gpu_start.elapsed().as_secs_f64() * 1000.0
            );
            // Session creation competes with the first real preview and initial
            // slider interaction. Wait briefly after the first frame before
            // loading heavy ONNX/CUDA sessions in the background.
            while !FIRST_PREVIEW_READY.load(Ordering::Acquire) {
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            std::thread::sleep(std::time::Duration::from_secs(3));
            lower_current_thread_priority();
            let start = std::time::Instant::now();
            match aurora_engine::aimask::prewarm_models() {
                Ok(()) => log::info!(
                    "AI sessions ready in {:.0} ms",
                    start.elapsed().as_secs_f64() * 1000.0
                ),
                Err(err) => log::warn!("AI session prewarm skipped: {err:#}"),
            }
        });
}

pub(crate) fn note_first_preview_ready() {
    FIRST_PREVIEW_READY.store(true, Ordering::Release);
}

#[cfg(windows)]
fn lower_current_thread_priority() {
    use std::ffi::c_void;
    unsafe extern "system" {
        fn GetCurrentThread() -> *mut c_void;
        fn SetThreadPriority(thread: *mut c_void, priority: i32) -> i32;
    }
    const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
    unsafe {
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
}

#[cfg(not(windows))]
fn lower_current_thread_priority() {}

/// Logs to stderr everywhere. On macOS it also appends to
/// `~/Library/Logs/Aurora Studio/aurora-studio.log`, because a bundle launched
/// from Finder has no visible stderr.
struct Logger {
    #[cfg(target_os = "macos")]
    file: std::sync::Mutex<Option<std::fs::File>>,
}

impl log::Log for Logger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::max_level()
    }
    fn log(&self, r: &log::Record) {
        if !self.enabled(r.metadata()) {
            return;
        }
        eprintln!("[{}] {}: {}", r.level(), r.target(), r.args());
        #[cfg(target_os = "macos")]
        {
            use std::io::Write as _;
            if let Ok(mut file) = self.file.lock()
                && let Some(file) = file.as_mut()
            {
                let _ = writeln!(file, "[{}] {}: {}", r.level(), r.target(), r.args());
                let _ = file.flush();
            }
        }
    }
    fn flush(&self) {
        #[cfg(target_os = "macos")]
        {
            use std::io::Write as _;
            if let Ok(mut file) = self.file.lock()
                && let Some(file) = file.as_mut()
            {
                let _ = file.flush();
            }
        }
    }
}

static LOGGER: Logger = Logger {
    #[cfg(target_os = "macos")]
    file: std::sync::Mutex::new(None),
};

#[cfg(target_os = "macos")]
fn open_log_file() -> Option<std::fs::File> {
    let home = std::env::var_os("HOME")?;
    let dir = std::path::PathBuf::from(home).join("Library/Logs/Aurora Studio");
    std::fs::create_dir_all(&dir).ok()?;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("aurora-studio.log"))
        .ok()
}

fn env_logger_init() {
    #[cfg(target_os = "macos")]
    if let Some(file) = open_log_file() {
        *LOGGER.file.lock().unwrap_or_else(|e| e.into_inner()) = Some(file);
    }
    let level = match std::env::var("RUST_LOG").as_deref() {
        Ok("debug") => log::LevelFilter::Debug,
        Ok("info") => log::LevelFilter::Info,
        Ok("error") => log::LevelFilter::Error,
        _ => log::LevelFilter::Warn,
    };
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(level);
    }
}
