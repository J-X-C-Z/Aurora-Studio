//! Preferences that outlive a session: appearance, viewer behavior, keyboard
//! shortcuts and workspaces. Stored as JSON in the user's config folder
//! (`%APPDATA%\Aurora Studio\settings.json` on Windows; settings saved under
//! the earlier Aurora Studio name are picked up on first launch).

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{BrowserMode, ToolTab};
use crate::i18n::t;

/// Where the display-profile lookup should look.
///
/// Windows resolves the monitor under the window's handle with the same ICM
/// calls as before; other platforms treat every target as a no-op. `None`
/// carries no handle (before the first prepaint, or off Windows), so the
/// Windows path keeps its primary-display fallback exactly as a `0` handle did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DisplayTarget {
    /// The monitor under this native window handle.
    Window(isize),
    /// No window handle yet.
    #[default]
    None,
}

impl DisplayTarget {
    /// A target for `hwnd`; a null handle means no window yet.
    pub fn from_handle(hwnd: isize) -> Self {
        if hwnd == 0 { DisplayTarget::None } else { DisplayTarget::Window(hwnd) }
    }

    /// The native handle, when there is one.
    pub fn handle(self) -> Option<isize> {
        match self {
            DisplayTarget::Window(hwnd) => Some(hwnd),
            DisplayTarget::None => None,
        }
    }

    /// The handle to pass to the Windows ICM calls; `0` (primary display) when
    /// there is no window, matching the former plain `isize` field.
    pub fn hwnd(self) -> isize {
        self.handle().unwrap_or(0)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CropGuide {
    #[default]
    Thirds,
    Golden,
    Grid,
    Diagonal,
    None,
}

impl CropGuide {
    pub const ALL: [Self; 5] = [Self::Thirds, Self::Golden, Self::Grid, Self::Diagonal, Self::None];

    pub fn label(self) -> &'static str {
        match self {
            Self::Thirds => t("Rule of thirds", "三分法"),
            Self::Golden => t("Golden ratio", "黄金分割"),
            Self::Grid => t("Grid", "网格"),
            Self::Diagonal => t("Diagonals", "对角线"),
            Self::None => t("None", "无"),
        }
    }
}

/// One arrangement of the window: where the tools are, which tabs show, which
/// browser, and how large things are.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkspaceLayout {
    pub name: String,
    pub inspector_right: bool,
    pub show_inspector: bool,
    /// `filmstrip`, `grid`, `list` or `hidden`.
    pub browser: String,
    pub hidden_tabs: Vec<String>,
    /// Tab ids in the order the strip shows them; empty is the default order.
    pub tab_order: Vec<String>,
    /// Tool panel ids on the Quick tab, top to bottom.
    pub quick_tools: Vec<String>,
    /// Panel ids per tab id in the order the user dragged them into; panels a
    /// list does not name keep their built-in place.
    pub panel_order: BTreeMap<String, Vec<String>>,
    /// Panels pinned to the top of every tab that shows them, in pin order.
    pub pinned_panels: Vec<String>,
    pub tab: String,
    pub inspector_width: f32,
    pub filmstrip_height: f32,
    pub multi: bool,
}

impl Default for WorkspaceLayout {
    fn default() -> Self {
        Self {
            name: t("Default", "默认").into(),
            inspector_right: false,
            show_inspector: true,
            browser: "filmstrip".into(),
            // The Portrait tab is partly rendered; the Portrait Retouch workspace shows it.
            hidden_tabs: vec![tab_id(ToolTab::Portrait)],
            tab_order: Vec::new(),
            quick_tools: ["wb", "exposure", "hdr", "clarity"].map(String::from).to_vec(),
            panel_order: Default::default(),
            pinned_panels: Vec::new(),
            tab: "Exposure".into(),
            inspector_width: 320.0,
            filmstrip_height: 132.0,
            multi: true,
        }
    }
}

pub fn browser_name(b: BrowserMode) -> &'static str {
    match b {
        BrowserMode::Filmstrip => "filmstrip",
        BrowserMode::Grid => "grid",
        BrowserMode::List => "list",
        BrowserMode::Hidden => "hidden",
    }
}

pub fn browser_from(name: &str) -> BrowserMode {
    match name {
        "grid" => BrowserMode::Grid,
        "list" => BrowserMode::List,
        "hidden" => BrowserMode::Hidden,
        _ => BrowserMode::Filmstrip,
    }
}

pub fn tab_id(tab: ToolTab) -> String {
    format!("{tab:?}")
}

/// ICC profiles installed for displays on this system, by file name.
pub fn display_profiles() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if cfg!(windows) {
        let root = std::env::var_os("WINDIR").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
        dirs.push(root.join(r"System32\spool\drivers\color"));
    } else if cfg!(target_os = "macos") {
        dirs.push(PathBuf::from("/Library/ColorSync/Profiles"));
        dirs.extend(std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/ColorSync/Profiles")));
    } else {
        // colord keeps calibrated and EDID profiles under the user's data folder.
        dirs.extend(std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share/icc")));
        dirs.push(PathBuf::from("/usr/share/color/icc"));
        dirs.push(PathBuf::from("/usr/share/color/icc/colord"));
    }
    let mut found: Vec<PathBuf> = dirs
        .iter()
        .filter_map(|d| std::fs::read_dir(d).ok())
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("icc") || e.eq_ignore_ascii_case("icm")))
        .collect();
    found.sort_by_key(|p| p.file_name().map(|n| n.to_ascii_lowercase()));
    found.dedup();
    found
}

/// The ICC profile Windows reports for the monitor under `target` (the app's
/// window); falls back to the primary display. `None` off Windows or on failure.
pub fn detected_display_profile(target: DisplayTarget) -> Option<PathBuf> {
    let hwnd = target.hwnd();
    #[cfg(windows)]
    {
        win_icm::current(hwnd)
    }
    #[cfg(not(windows))]
    {
        let _ = hwnd;
        None
    }
}

/// What the current monitor lookup resolved, and whether it fell back to the
/// primary display.
pub struct DisplayProfileStatus {
    pub path: Option<PathBuf>,
    pub fallback: bool,
}

pub fn detected_display_profile_status(target: DisplayTarget) -> DisplayProfileStatus {
    let hwnd = target.hwnd();
    #[cfg(windows)]
    {
        let (path, fallback) = win_icm::status(hwnd);
        DisplayProfileStatus { path, fallback }
    }
    #[cfg(not(windows))]
    {
        let _ = hwnd;
        DisplayProfileStatus { path: None, fallback: false }
    }
}

/// Drops cached monitor profiles, e.g. on a display change.
pub fn invalidate_display_profile_cache() {
    #[cfg(windows)]
    win_icm::invalidate();
}

#[cfg(windows)]
mod win_icm {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStringExt as _;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    use std::time::{Duration, Instant};

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Rect {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }

    #[repr(C)]
    struct MonitorInfo {
        cb_size: u32,
        rc_monitor: Rect,
        rc_work: Rect,
        flags: u32,
        /// `MONITORINFOEXW.szDevice`, NUL-terminated.
        device: [u16; 32],
    }

    #[link(name = "user32")]
    unsafe extern "system" {
        fn GetDC(hwnd: *mut c_void) -> *mut c_void;
        fn ReleaseDC(hwnd: *mut c_void, hdc: *mut c_void) -> i32;
        fn GetWindowRect(hwnd: *mut c_void, rect: *mut Rect) -> i32;
        fn IsWindow(hwnd: *mut c_void) -> i32;
        fn EnumDisplayMonitors(dc: *mut c_void, clip: *mut Rect, cb: Option<unsafe extern "system" fn(*mut c_void, *mut c_void, *mut Rect, isize) -> i32>, data: isize) -> i32;
        fn GetMonitorInfoW(monitor: *mut c_void, info: *mut MonitorInfo) -> i32;
    }

    #[link(name = "gdi32")]
    unsafe extern "system" {
        fn GetICMProfileW(hdc: *mut c_void, size: *mut u32, file: *mut u16) -> i32;
        fn CreateDCW(driver: *const u16, device: *const u16, port: *const u16, init: *mut c_void) -> *mut c_void;
        fn DeleteDC(hdc: *mut c_void) -> i32;
    }

    /// Display profiles change rarely; a short TTL also covers hot-plug and
    /// Windows colour-management changes that we get no callback for.
    const CACHE_TTL: Duration = Duration::from_secs(5);

    type Entry = (String, Option<PathBuf>, Instant);

    fn cache() -> &'static Mutex<Vec<Entry>> {
        static CACHE: OnceLock<Mutex<Vec<Entry>>> = OnceLock::new();
        CACHE.get_or_init(Default::default)
    }

    fn cached(key: &str) -> Option<Option<PathBuf>> {
        let now = Instant::now();
        let mut cache = cache().lock().unwrap_or_else(|e| e.into_inner());
        let at = cache.iter().position(|(k, _, _)| k == key)?;
        if !fresh(cache[at].2, now) {
            cache.remove(at);
            return None;
        }
        Some(cache[at].1.clone())
    }

    fn remember(key: String, value: Option<PathBuf>) {
        let mut cache = cache().lock().unwrap_or_else(|e| e.into_inner());
        cache.retain(|(k, _, _)| *k != key);
        if cache.len() < 16 {
            cache.push((key, value, Instant::now()));
        }
    }

    fn fresh(since: Instant, now: Instant) -> bool {
        now.saturating_duration_since(since) <= CACHE_TTL
    }

    pub(super) fn invalidate() {
        cache().lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// ICC of the monitor under the app window; the primary screen when that
    /// window, its monitor, or its device context cannot be resolved.
    pub(super) fn current(hwnd: isize) -> Option<PathBuf> {
        status(hwnd).0
    }

    pub(super) fn status(hwnd: isize) -> (Option<PathBuf>, bool) {
        if let Some(device) = window_point(hwnd).and_then(device_at) {
            let key = device_key(&device);
            if let Some(hit) = cached(&key) {
                return (hit, false);
            }
            let out = unsafe {
                let hdc = CreateDCW(std::ptr::null(), device.as_ptr(), std::ptr::null(), std::ptr::null_mut());
                if hdc.is_null() {
                    None
                } else {
                    let out = profile(hdc);
                    DeleteDC(hdc);
                    out
                }
            };
            if out.is_some() {
                remember(key, out.clone());
                return (out, false);
            }
            remember(key, None);
        }
        let key = "primary".to_string();
        if let Some(hit) = cached(&key) {
            return (hit, true);
        }
        let out = unsafe {
            let hdc = GetDC(std::ptr::null_mut());
            if hdc.is_null() {
                None
            } else {
                let out = profile(hdc);
                ReleaseDC(std::ptr::null_mut(), hdc);
                out
            }
        };
        remember(key, out.clone());
        (out, true)
    }

    fn device_key(device: &[u16; 32]) -> String {
        let len = device.iter().position(|c| *c == 0).unwrap_or(device.len());
        String::from_utf16_lossy(&device[..len]).to_ascii_lowercase()
    }

    fn window_point(hwnd: isize) -> Option<(i32, i32)> {
        if hwnd == 0 {
            return None;
        }
        unsafe {
            let hwnd = hwnd as *mut c_void;
            if IsWindow(hwnd) == 0 {
                return None;
            }
            let mut rect = Rect::default();
            if GetWindowRect(hwnd, &mut rect) == 0 {
                return None;
            }
            Some(((rect.left + rect.right) / 2, (rect.top + rect.bottom) / 2))
        }
    }

    unsafe fn profile(hdc: *mut c_void) -> Option<PathBuf> {
        unsafe {
            let mut size: u32 = 0;
            GetICMProfileW(hdc, &mut size, std::ptr::null_mut());
            let mut buf = vec![0u16; size.max(1) as usize];
            size = buf.len() as u32;
            let ok = GetICMProfileW(hdc, &mut size, buf.as_mut_ptr());
            decode(&buf, ok)
        }
    }

    struct Search {
        point: (i32, i32),
        device: Option<[u16; 32]>,
    }

    unsafe extern "system" fn find_monitor(monitor: *mut c_void, _dc: *mut c_void, _clip: *mut Rect, data: isize) -> i32 {
        let search = unsafe { &mut *(data as *mut Search) };
        let mut info = MonitorInfo {
            cb_size: std::mem::size_of::<MonitorInfo>() as u32,
            rc_monitor: Rect::default(),
            rc_work: Rect::default(),
            flags: 0,
            device: [0; 32],
        };
        unsafe {
            if GetMonitorInfoW(monitor, &mut info) == 0 {
                return 0;
            }
        }
        if contains(&info.rc_monitor, search.point.0, search.point.1) {
            search.device = Some(info.device);
            1
        } else {
            0
        }
    }

    fn device_at(point: (i32, i32)) -> Option<[u16; 32]> {
        let mut search = Search { point, device: None };
        unsafe {
            EnumDisplayMonitors(std::ptr::null_mut(), std::ptr::null_mut(), Some(find_monitor), &mut search as *mut Search as isize);
        }
        search.device
    }

    fn contains(rect: &Rect, x: i32, y: i32) -> bool {
        x >= rect.left && x < rect.right && y >= rect.top && y < rect.bottom
    }

    /// NUL-terminated wide path, only when the call reported success.
    fn decode(buf: &[u16], ok: i32) -> Option<PathBuf> {
        if ok == 0 {
            return None;
        }
        let len = buf.iter().position(|c| *c == 0).filter(|len| *len > 0)?;
        Some(PathBuf::from(std::ffi::OsString::from_wide(&buf[..len])))
    }

    #[cfg(test)]
    mod tests {
        use super::{Rect, cached, contains, current, decode, device_key, fresh, invalidate, remember, status, window_point};
        use std::time::{Duration, Instant};

        #[test]
        fn monitor_rect_contains_edges_exclusively() {
            let rect = Rect { left: -1920, top: 0, right: 0, bottom: 1080 };
            assert!(contains(&rect, -1920, 0));
            assert!(contains(&rect, -1, 1079));
            assert!(!contains(&rect, 0, 0));
            assert!(!contains(&rect, -1920, 1080));
        }

        #[test]
        fn null_dc_has_no_profile() {
            assert!(unsafe { super::profile(std::ptr::null_mut()) }.is_none());
        }

        #[test]
        fn current_profile_is_readable_when_present() {
            if let Some(path) = current(0) {
                assert!(path.exists(), "{}", path.display());
            }
        }

        #[test]
        fn invalid_window_falls_back_to_primary() {
            assert_eq!(window_point(0), None);
            assert_eq!(window_point(1), None);
            let wide: Vec<u16> = "\\\\.\\DISPLAY1".encode_utf16().chain([0]).collect();
            let mut device = [0u16; 32];
            device[..wide.len()].copy_from_slice(&wide);
            assert_eq!(device_key(&device), "\\\\.\\display1");
        }

        #[test]
        fn expired_entries_are_dropped_and_can_be_cleared() {
            let now = Instant::now();
            assert!(fresh(now, now));
            assert!(!fresh(now - Duration::from_secs(10), now));
            remember("test-device".into(), None);
            assert_eq!(cached("test-device"), Some(None));
            invalidate();
            assert_eq!(cached("test-device"), None);
        }

        #[test]
        fn missing_window_reports_primary_fallback() {
            let (path, fallback) = status(0);
            assert!(fallback);
            if let Some(path) = path {
                assert!(path.exists(), "{}", path.display());
            }
        }

        #[test]
        fn failed_or_empty_profile_is_none() {
            let path = "C:\\Windows\\System32\\spool\\drivers\\color\\sRGB Color Space Profile.icm";
            let wide: Vec<u16> = path.encode_utf16().chain([0]).collect();
            assert_eq!(decode(&wide, 0), None);
            assert_eq!(decode(&[], 1), None);
            assert_eq!(decode(&[0], 1), None);
            assert_eq!(decode(&wide, 1).unwrap().to_string_lossy(), path);
        }
    }
}

/// Where the tabs sit before the user moves them: Quick after Capture, as in Capture One.
pub fn default_tab_order() -> Vec<ToolTab> {
    let mut order: Vec<ToolTab> = ToolTab::ALL.into_iter().filter(|t| *t != ToolTab::Quick).collect();
    order.insert(2, ToolTab::Quick);
    order
}

pub fn tab_from(id: &str) -> Option<ToolTab> {
    match id {
        "Color" | "Exposure" | "Layers" => Some(ToolTab::Grading),
        _ => ToolTab::ALL.into_iter().find(|t| tab_id(*t) == id),
    }
}

/// Workspaces that ship with Aurora, in the order of View → Workspace.
pub fn builtin_workspaces() -> Vec<WorkspaceLayout> {
    let hide = |keep: &[ToolTab]| ToolTab::ALL.iter().filter(|t| !keep.contains(t)).map(|t| tab_id(*t)).collect::<Vec<_>>();
    vec![
        WorkspaceLayout::default(),
        WorkspaceLayout {
            name: t("Culling", "筛选").into(),
            browser: "grid".into(),
            hidden_tabs: hide(&[ToolTab::Library, ToolTab::Metadata, ToolTab::Output, ToolTab::Batch]),
            tab: "Library".into(),
            ..Default::default()
        },
        WorkspaceLayout {
            name: t("Tethering", "联机拍摄").into(),
            hidden_tabs: hide(&[ToolTab::Capture, ToolTab::Library, ToolTab::Grading, ToolTab::Adjustments, ToolTab::Output]),
            tab: "Capture".into(),
            filmstrip_height: 150.0,
            ..Default::default()
        },
        WorkspaceLayout {
            name: t("Portrait Retouch", "人像精修").into(),
            hidden_tabs: hide(&[ToolTab::Grading, ToolTab::Details, ToolTab::Portrait, ToolTab::Adjustments]),
            tab: "Portrait".into(),
            inspector_width: 360.0,
            multi: false,
            ..Default::default()
        },
        WorkspaceLayout { name: t("Minimal", "简洁").into(), show_inspector: false, browser: "hidden".into(), ..Default::default() },
    ]
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub chinese: bool,
    pub accent: Option<u32>,
    /// Multiplier on every animation's response time; 0 turns motion off.
    pub motion_speed: f32,
    pub viewer_gray: f32,
    pub thumb_size: f32,
    pub info_overlay: bool,
    /// Channel value at or above which the exposure warning marks highlights.
    pub warn_high: u8,
    /// Channel value at or below which shadows are marked.
    pub warn_low: u8,
    pub warn_shadows: bool,
    /// Gradient strength above which the focus mask paints an edge.
    pub focus_threshold: f32,
    pub crop_guide: CropGuide,
    /// Darkening outside the crop, 0 ..= 1.
    pub crop_mask: f32,
    /// Divisions of the viewer grid overlay.
    pub grid_divisions: u8,
    pub undo_levels: usize,
    /// Jump to each new capture as it arrives.
    pub select_new_capture: bool,
    /// Show a notice when a process job finishes.
    pub notify_on_process: bool,
    /// Keys per shortcut id that differ from the defaults.
    pub shortcuts: BTreeMap<String, Vec<String>>,
    pub workspace: WorkspaceLayout,
    pub workspaces: Vec<WorkspaceLayout>,
    /// Folders and catalogs opened most recently, newest first.
    pub recent: Vec<PathBuf>,
    /// Open the most recent folder or catalog when started without one.
    pub reopen_last: bool,
    /// Styles saved from the user's own adjustments.
    pub user_styles: Vec<UserStyle>,
    pub metadata_presets: Vec<MetadataPreset>,
    /// The composition overlay: a transparent PNG laid over the viewer.
    pub overlay_image: Option<PathBuf>,
    pub overlay_opacity: f32,
    pub slideshow_interval: f32,
    /// Culling moves on after each rating.
    pub cull_auto_advance: bool,
    /// The monitor's ICC profile; `None` shows previews as sRGB.
    pub display_profile: Option<PathBuf>,
    /// Soft-proof profile name (engine profile stem); `None` skips proofing.
    pub proof_profile: Option<String>,
    /// Soft-proof white level; 1.0 is neutral.
    #[serde(default = "one")]
    pub proof_paper_white: f32,
    /// Soft-proof black level; 0.0 is neutral.
    pub proof_black_point: f32,
    /// Custom order and stacks per folder or catalog path.
    pub arrangements: BTreeMap<String, super::Arrangement>,
    /// ICC picker group: 0 all, 1 camera, 2 looks, 3 output. Unknown values
    /// read as 0, so an old or hand-edited file never blocks startup.
    #[serde(deserialize_with = "icc_group_value")]
    pub icc_group: u8,
}

fn icc_group_value<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<u8, D::Error> {
    let raw = serde_json::Value::deserialize(deserializer)?;
    Ok(match raw {
        serde_json::Value::Number(n) => n.as_u64().filter(|v| *v <= 3).unwrap_or(0) as u8,
        serde_json::Value::String(s) => s.parse::<u8>().ok().filter(|v| *v <= 3).unwrap_or(0),
        _ => 0,
    })
}

/// IPTC fields and keywords to stamp onto images. Empty fields are left alone
/// when applied; keywords are added, never removed.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MetadataPreset {
    pub name: String,
    pub iptc: aurora_engine::Iptc,
    pub keywords: Vec<String>,
}

/// A style saved from an image's adjustments: only the tools in `groups` are
/// stored and applied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UserStyle {
    pub name: String,
    /// [`super::ClipGroup::id`] of each included tool.
    pub groups: Vec<String>,
    pub adjustments: aurora_engine::AdjustmentSet,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            chinese: true,
            accent: None,
            motion_speed: 1.0,
            viewer_gray: 0.11,
            thumb_size: 120.0,
            info_overlay: true,
            warn_high: 250,
            warn_low: 4,
            warn_shadows: true,
            focus_threshold: 22.0,
            crop_guide: CropGuide::Thirds,
            crop_mask: 0.55,
            grid_divisions: 3,
            undo_levels: 200,
            select_new_capture: true,
            notify_on_process: true,
            shortcuts: BTreeMap::new(),
            workspace: WorkspaceLayout::default(),
            workspaces: Vec::new(),
            recent: Vec::new(),
            reopen_last: false,
            user_styles: Vec::new(),
            metadata_presets: Vec::new(),
            arrangements: BTreeMap::new(),
            overlay_image: None,
            overlay_opacity: 0.6,
            slideshow_interval: 4.0,
            cull_auto_advance: true,
            display_profile: None,
            proof_profile: None,             proof_paper_white: 1.0,             proof_black_point: 0.0,             icc_group: 0,
        }
    }
}

fn one() -> f32 {
    1.0
}

/// `AURORA_SETTINGS` (or the older `AURORA_SETTINGS`), else the config folder.
fn path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("AURORA_SETTINGS").or_else(|| std::env::var_os("AURORA_SETTINGS")) {
        return Some(p.into());
    }
    config_dir("Aurora Studio", "aurora-studio").map(|d| d.join("settings.json"))
}

fn config_dir(windows: &str, unix: &str) -> Option<PathBuf> {
    if cfg!(windows) {
        std::env::var_os("APPDATA").map(|d| PathBuf::from(d).join(windows))
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .map(|d| d.join(unix))
    }
}

/// Settings written before the rename to Aurora Studio.
fn legacy_path() -> Option<PathBuf> {
    config_dir("Aurora Studio", "aurora-studio").map(|d| d.join("settings.json"))
}

const RECENT: usize = 10;

impl Settings {
    /// Moves `path` to the front of the recent list.
    pub fn remember(&mut self, path: &std::path::Path) {
        // `absolute`, not `canonicalize`: no `\\?\` prefix on Windows.
        let path = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
        self.recent.retain(|p| *p != path);
        self.recent.insert(0, path);
        self.recent.truncate(RECENT);
    }

    pub fn forget(&mut self, path: &std::path::Path) {
        self.recent.retain(|p| p != path);
    }

    pub fn load() -> Self {
        path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .or_else(|| legacy_path().and_then(|p| std::fs::read_to_string(p).ok()))
            .and_then(|text| match serde_json::from_str(&text) {
                Ok(s) => Some(s),
                Err(e) => {
                    log::warn!("settings: {e}");
                    None
                }
            })
            .unwrap_or_default()
    }

    pub fn save(&self) {
        let Some(path) = path() else { return };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match serde_json::to_string_pretty(self) {
            Ok(text) => {
                if let Err(e) = std::fs::write(&path, text) {
                    log::warn!("settings: cannot write {}: {e}", path.display());
                }
            }
            Err(e) => log::warn!("settings: {e}"),
        }
    }
}

#[cfg(test)]
mod tab_tests {
    use super::*;

    #[test]
    fn display_target_round_trips_none_and_window() {
        assert_eq!(DisplayTarget::None.handle(), None);
        assert_eq!(DisplayTarget::None.hwnd(), 0);
        assert_eq!(DisplayTarget::Window(42).handle(), Some(42));
        assert_eq!(DisplayTarget::Window(42).hwnd(), 42);

        assert_eq!(DisplayTarget::from_handle(0), DisplayTarget::None);
        assert_eq!(DisplayTarget::from_handle(42), DisplayTarget::Window(42));
        assert_eq!(DisplayTarget::from_handle(42).handle(), Some(42));
        assert_eq!(DisplayTarget::default(), DisplayTarget::None);
    }

    #[test]
    fn default_order_has_every_tab_once_with_quick_third() {
        let order = default_tab_order();
        assert_eq!(order.len(), ToolTab::ALL.len());
        assert_eq!(order[2], ToolTab::Quick);
        assert!(ToolTab::ALL.iter().all(|t| order.contains(t)));
        assert_eq!(*ToolTab::ALL.last().unwrap(), ToolTab::Quick, "Ctrl+1…9 keep their tabs");
    }

    #[test]
    fn quick_panels_are_unique() {
        let mut ids = crate::tools::QUICK_PANELS.to_vec();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), crate::tools::QUICK_PANELS.len());
        assert!(WorkspaceLayout::default().quick_tools.iter().all(|id| crate::tools::QUICK_PANELS.contains(&id.as_str())));
    }

    #[test]
    fn old_workspaces_still_load() {
        let w: WorkspaceLayout = serde_json::from_str(r#"{"name":"Mine","hidden_tabs":["Batch"]}"#).unwrap();
        assert!(w.tab_order.is_empty());
        assert!(w.panel_order.is_empty() && w.pinned_panels.is_empty());
        assert_eq!(w.hidden_tabs, vec!["Batch".to_string()]);
    }

    #[test]
    fn settings_without_proof_profile_defaults_to_none() {
        let s: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(s.proof_profile, None);
        assert_eq!(s.proof_paper_white, 1.0);
        assert_eq!(s.proof_black_point, 0.0);
    }

    #[test]
    fn the_icc_group_defaults_to_all_and_rejects_unknown_values() {
        assert_eq!(Settings::default().icc_group, 0);
        assert_eq!(serde_json::from_str::<Settings>("{}").unwrap().icc_group, 0);
        assert_eq!(serde_json::from_str::<Settings>(r#"{"icc_group":3}"#).unwrap().icc_group, 3);
        assert_eq!(serde_json::from_str::<Settings>(r#"{"icc_group":7}"#).unwrap().icc_group, 0);
        assert_eq!(serde_json::from_str::<Settings>(r#"{"icc_group":"2"}"#).unwrap().icc_group, 2);
        assert_eq!(serde_json::from_str::<Settings>(r#"{"icc_group":"nope"}"#).unwrap().icc_group, 0);
        let round = serde_json::to_string(&Settings { icc_group: 1, ..Settings::default() }).unwrap();
        assert_eq!(serde_json::from_str::<Settings>(&round).unwrap().icc_group, 1);
    }
}
