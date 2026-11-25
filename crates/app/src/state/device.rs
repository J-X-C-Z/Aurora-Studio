//! Device and display state split out of [`super::AppState`].
//!
//! This is a plain data holder for tethering status and the window's display,
//! plus the small cross-platform [`DisplayTarget`] the ICC lookup uses. It has
//! no event bus of its own; the UI still observes `AppState`.

use aurora_engine::{CameraSettings, CameraStatus, TetherStatus};

use super::settings::DisplayTarget;

/// Camera/tether status and the window's display identity.
///
/// Field semantics are unchanged from the former `AppState` fields: the
/// window handle starts at `0`, the scale factor at `1.0`, and `tether` stays
/// `None` until the first probe returns.
#[derive(Clone, Debug)]
pub struct DeviceState {
    pub camera: CameraStatus,
    pub camera_settings: Option<CameraSettings>,
    /// What the engine's tethering backend answered; `None` until the first probe returns.
    pub tether: Option<TetherStatus>,
    pub tether_probing: bool,
    /// Native handle of the main window, for display-profile lookup.
    pub window_hwnd: isize,
    pub scale_factor: f32,
}

impl Default for DeviceState {
    fn default() -> Self {
        Self {
            camera: CameraStatus::Disconnected,
            camera_settings: None,
            tether: None,
            tether_probing: false,
            window_hwnd: 0,
            scale_factor: 1.0,
        }
    }
}

impl DeviceState {
    /// Records the camera the tether probe found and its settings.
    pub fn set_camera(&mut self, camera: CameraStatus, settings: Option<CameraSettings>) {
        self.camera = camera;
        self.camera_settings = settings;
    }

    /// Records what the tether backend answered.
    pub fn set_tether(&mut self, tether: Option<TetherStatus>) {
        self.tether = tether;
    }

    pub fn set_tether_probing(&mut self, probing: bool) {
        self.tether_probing = probing;
    }

    pub fn set_window_hwnd(&mut self, hwnd: isize) {
        self.window_hwnd = hwnd;
    }

    /// Sets the window scale factor, returning whether it changed.
    pub fn set_scale_factor(&mut self, scale_factor: f32) -> bool {
        if self.scale_factor == scale_factor {
            return false;
        }
        self.scale_factor = scale_factor;
        true
    }

    /// The display the window is on, for ICC profile lookup.
    pub fn display_target(&self) -> DisplayTarget {
        DisplayTarget::from_handle(self.window_hwnd)
    }

    /// Whether tethered capture can be offered: the engine's tethering backend has
    /// a camera open (`tether_status`, not `camera_status`, per the engine contract).
    pub fn camera_ready(&self) -> bool {
        matches!(self.tether, Some(TetherStatus::Available { .. }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_defaults_and_scale_factor_track_changes() {
        let mut device = DeviceState::default();
        assert_eq!(device.scale_factor, 1.0);
        assert_eq!(device.window_hwnd, 0);
        assert_eq!(device.display_target().handle(), None);
        assert!(!device.camera_ready());

        assert!(device.set_scale_factor(2.0));
        assert_eq!(device.scale_factor, 2.0);
        assert!(!device.set_scale_factor(2.0), "same value is not a change");

        device.set_window_hwnd(7);
        assert_eq!(device.display_target().handle(), Some(7));
    }

    #[test]
    fn tether_available_is_what_makes_capture_ready() {
        let mut device = DeviceState::default();
        device.set_tether(Some(TetherStatus::NoCamera));
        assert!(!device.camera_ready());
        device.set_tether(Some(TetherStatus::Available { lib_version: "2.5".into() }));
        assert!(device.camera_ready());
    }
}
