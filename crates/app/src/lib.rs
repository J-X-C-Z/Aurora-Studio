//! Aurora Studio: a professional raw editor interface on GPUI.

mod actions;
mod capability;
mod contract;
mod diagnostics;
pub use diagnostics::init as init_diagnostics;
mod evidence;
mod gesture;
mod haptics;
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
    init_diagnostics();
    let touch = launch.touch || gpui_kit::is_mobile();
    theme::set_touch(touch);
    let args = launch.paths;
    prewarm_in_background();

    let mut error = None;
    let mut requests = state::open::requests(&args).unwrap_or_else(|err| {
        error = Some(format!("{err:#}"));
        Vec::new()
    }).into_iter();
    let first = requests.next();
    let (target, files) = match first {
        Some(request) => (Some(request.root), request.photos),
        None if args.is_empty() => {
            let settings = state::Settings::load();
            (settings.reopen_last.then(|| settings.recent.first().cloned()).flatten().filter(|p| p.exists()), vec![])
        }
        None => (None, vec![]),
    };
    // As in File → Open Folder, distinct libraries have distinct processes and
    // edit stores. Never import an absolute external path into another catalog.
    for request in requests {
        let result = std::env::current_exe().and_then(|exe| {
            let mut cmd = std::process::Command::new(exe);
            if touch { cmd.arg("--touch"); }
            cmd.arg(&request.root).args(request.photos).spawn()
        });
        if let Err(err) = result { error = Some(format!("{}: {err}", request.root.display())); }
    }
    let (engine, source) = match &target {
        Some(path) => state::open::open(path).unwrap_or_else(|err| {
            log::error!("cannot open {}: {err}", path.display());
            error = Some(format!("{}: {err}", path.display()));
            state::open::demo()
        }),
        None => state::open::demo(),
    };

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
            // macOS application-menu actions. They are no-ops elsewhere.
            cx.on_action(|_: &actions::Hide, cx| cx.hide());
            cx.on_action(|_: &actions::HideOthers, cx| cx.hide_other_apps());
            cx.on_action(|_: &actions::ShowAll, cx| cx.unhide_other_apps());
            cx.activate(true);

            let state = state::init_state(engine.clone(), source.clone(), cx);
            // The workspace can change its backing engine after launch. Global
            // Quit (including the application menu) must save the live state.
            cx.on_action(|_: &actions::Quit, cx| {
                if let Some(state) = cx.try_global::<state::ActiveLibrary>().map(|g| g.0.clone()) {
                    let _ = state.update(cx, |s, cx| s.flush_and_quit(cx));
                }
            });
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
                // The real folder scanner has already indexed the originals. Select
                // them instead of importing a second copy (or editing a demo image).
                if !files.is_empty() && s.source != state::Source::Demo {
                    let mut ids = Vec::new();
                    let mut absent = Vec::new();
                    for path in &files {
                        let direct = s.library.images.iter().find(|i| i.path == *path && !i.trashed);
                        let found = direct.or_else(|| {
                            let key = std::fs::canonicalize(path).ok()?;
                            s.library.images.iter().filter(|i| !i.trashed && i.path.file_name().zip(path.file_name()).is_some_and(|(a,b)| a.eq_ignore_ascii_case(b)))
                                .find(|i| std::fs::canonicalize(&i.path).ok().as_ref() == Some(&key))
                        });
                        if let Some(info) = found { ids.push(info.id); } else { absent.push(path.display().to_string()); }
                    }
                    if let Some(&first) = ids.first() {
                        s.select(first, false, false, cx);
                        s.library.selection.extend(ids);
                    }
                    if !absent.is_empty() {
                        s.notify(format!("{}: {}", crate::i18n::t("Not in this library", "图库中未找到"), absent.join(", ")), cx);
                    }
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
    if gpui_kit::is_mobile() { return; }
    // RAW and AI warmup have distinct switches. Loading a driver module in
    // advance can remove first-click work, but an early click may still race it.
    aurora_engine::aurora_raw::prewarm::spawn_background();
    if std::env::var("AURORA_AI_PREWARM").is_ok_and(|value| value == "0") { return; }
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
