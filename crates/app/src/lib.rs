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
            cx.activate(true);

            let state = state::init_state(engine.clone(), source.clone(), cx);
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
            // Session creation competes with the first real preview. Let that
            // frame finish before consuming the same CPU/GPU resources.
            while !FIRST_PREVIEW_READY.load(Ordering::Acquire) {
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
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

fn env_logger_init() {
    struct Stderr;
    impl log::Log for Stderr {
        fn enabled(&self, m: &log::Metadata) -> bool {
            m.level() <= log::max_level()
        }
        fn log(&self, r: &log::Record) {
            if self.enabled(r.metadata()) {
                eprintln!("[{}] {}: {}", r.level(), r.target(), r.args());
            }
        }
        fn flush(&self) {}
    }
    static LOGGER: Stderr = Stderr;
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
