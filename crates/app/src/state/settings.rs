//! Preferences that outlive a session: appearance, viewer behavior, keyboard
//! shortcuts and workspaces. Stored as JSON in the user's config folder
//! (`%APPDATA%\Aurora Studio\settings.json` on Windows; settings saved under
//! the earlier Lumen Studio name are picked up on first launch).

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{BrowserMode, ToolTab};
use crate::i18n::t;

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
            // Nothing on the Portrait tab renders yet; the Portrait Retouch workspace shows it.
            hidden_tabs: vec![tab_id(ToolTab::Portrait)],
            tab_order: Vec::new(),
            quick_tools: ["wb", "exposure", "hdr", "clarity"].map(String::from).to_vec(),
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

/// Where the tabs sit before the user moves them: Quick after Capture, as in Capture One.
pub fn default_tab_order() -> Vec<ToolTab> {
    let mut order: Vec<ToolTab> = ToolTab::ALL.into_iter().filter(|t| *t != ToolTab::Quick).collect();
    order.insert(2, ToolTab::Quick);
    order
}

pub fn tab_from(id: &str) -> Option<ToolTab> {
    ToolTab::ALL.into_iter().find(|t| tab_id(*t) == id)
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
            hidden_tabs: hide(&[ToolTab::Capture, ToolTab::Library, ToolTab::Exposure, ToolTab::Color, ToolTab::Adjustments, ToolTab::Output]),
            tab: "Capture".into(),
            filmstrip_height: 150.0,
            ..Default::default()
        },
        WorkspaceLayout {
            name: t("Portrait Retouch", "人像精修").into(),
            hidden_tabs: hide(&[ToolTab::Color, ToolTab::Exposure, ToolTab::Details, ToolTab::Layers, ToolTab::Portrait, ToolTab::Adjustments]),
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
    /// Custom order and stacks per folder or catalog path.
    pub arrangements: BTreeMap<String, super::Arrangement>,
}

/// IPTC fields and keywords to stamp onto images. Empty fields are left alone
/// when applied; keywords are added, never removed.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MetadataPreset {
    pub name: String,
    pub iptc: lumen_engine::Iptc,
    pub keywords: Vec<String>,
}

/// A style saved from an image's adjustments: only the tools in `groups` are
/// stored and applied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UserStyle {
    pub name: String,
    /// [`super::ClipGroup::id`] of each included tool.
    pub groups: Vec<String>,
    pub adjustments: lumen_engine::AdjustmentSet,
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
        }
    }
}

/// `AURORA_SETTINGS` (or the older `LUMEN_SETTINGS`), else the config folder.
fn path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("AURORA_SETTINGS").or_else(|| std::env::var_os("LUMEN_SETTINGS")) {
        return Some(p.into());
    }
    config_dir("Aurora Studio", "aurora-studio").map(|d| d.join("settings.json"))
}

fn config_dir(windows: &str, unix: &str) -> Option<PathBuf> {
    if cfg!(windows) {
        std::env::var_os("APPDATA").map(|d| PathBuf::from(d).join(windows))
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support").join(windows))
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .map(|d| d.join(unix))
    }
}

/// Settings written before the rename to Aurora Studio.
fn legacy_path() -> Option<PathBuf> {
    config_dir("Lumen Studio", "lumen-studio").map(|d| d.join("settings.json"))
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
        assert_eq!(w.hidden_tabs, vec!["Batch".to_string()]);
    }
}
