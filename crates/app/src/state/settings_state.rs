//! Session preferences and the engine identity read once at start.
//!
//! `settings` persists across sessions (see [`Settings`]); `panels` holds the
//! per-panel UI choices; `engine_name` and `capabilities` are what the engine
//! reported when the window opened. They are grouped here so the rest of the
//! model can be read without scrolling past them.

use aurora_engine::EngineCapabilities;

use super::{PanelPrefs, Settings, ViewPrefs};

pub struct SettingsState {
    /// Preferences that survive a restart.
    pub settings: Settings,
    /// Per-panel UI choices that are not adjustments.
    pub panels: PanelPrefs,
    /// Viewer/workspace display preferences.
    pub view: ViewPrefs,
    /// The engine's display name.
    pub engine_name: String,
    /// What the engine reported it can do on this machine, read once at start.
    pub capabilities: EngineCapabilities,
}

impl Default for SettingsState {
    fn default() -> Self {
        Self {
            settings: Settings::default(),
            panels: PanelPrefs::default(),
            view: ViewPrefs::default(),
            engine_name: String::new(),
            capabilities: EngineCapabilities::default(),
        }
    }
}

impl SettingsState {
    /// Builds the group with the same initialization the model always used:
    /// the loaded settings, default panel choices, and the engine's name and
    /// capabilities as read at startup.
    pub fn from_parts(engine_name: String, capabilities: EngineCapabilities, settings: Settings) -> Self {
        Self { settings, panels: PanelPrefs::default(), view: ViewPrefs::default(), engine_name, capabilities }
    }
}
