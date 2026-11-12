//! Aurora Studio: a professional raw editor interface on GPUI.

mod actions;
mod contract;
mod i18n;
mod motion;
mod perf;
mod planned;
mod state;
mod theme;
mod tools;
mod ui;
mod widgets;

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
            cx.on_action(|_: &actions::Quit, cx| cx.quit());
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
