//! Speed Edit, as in Capture One: hold a key and scroll or drag over the image
//! to change one adjustment without going to its tool. A quick tap of the same
//! key still does what the key normally does (most are cursor tools); once the
//! key has been used for Speed Edit, releasing it puts the cursor back.

use std::time::{Duration, Instant};

use gpui_kit::Context;

use super::{AppState, Binding, CursorTool};
use crate::i18n::t;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeedParam {
    Exposure,
    Contrast,
    Brightness,
    Saturation,
    Highlights,
    Shadows,
    Whites,
    Blacks,
}

impl SpeedParam {
    pub const ALL: [Self; 8] =
        [Self::Exposure, Self::Contrast, Self::Brightness, Self::Saturation, Self::Highlights, Self::Shadows, Self::Whites, Self::Blacks];

    pub fn key(self) -> &'static str {
        match self {
            Self::Exposure => "q",
            Self::Contrast => "w",
            Self::Brightness => "e",
            Self::Saturation => "r",
            Self::Highlights => "a",
            Self::Shadows => "s",
            Self::Whites => "d",
            Self::Blacks => "f",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.key() == key)
    }

    /// The parameter's key in the frontend contract (`params.json`).
    pub fn c1_key(self) -> &'static str {
        match self {
            Self::Exposure => "Exposure",
            Self::Contrast => "Contrast",
            Self::Brightness => "Brightness",
            Self::Saturation => "Saturation",
            Self::Highlights => "HighlightRecovery",
            Self::Shadows => "ShadowRecovery",
            Self::Whites => "WhiteRecovery",
            Self::Blacks => "BlackRecovery",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Exposure => t("Exposure", "曝光"),
            Self::Contrast => t("Contrast", "对比度"),
            Self::Brightness => t("Brightness", "亮度"),
            Self::Saturation => t("Saturation", "饱和度"),
            Self::Highlights => t("Highlights", "高光"),
            Self::Shadows => t("Shadows", "阴影"),
            Self::Whites => t("Whites", "白色"),
            Self::Blacks => t("Blacks", "黑色"),
        }
    }

    pub fn binding(self, scope: super::Scope) -> Binding {
        match self {
            Self::Exposure => crate::scoped!(scope, exposure.exposure),
            Self::Contrast => crate::scoped!(scope, exposure.contrast),
            Self::Brightness => crate::scoped!(scope, exposure.brightness),
            Self::Saturation => crate::scoped!(scope, exposure.saturation),
            Self::Highlights => crate::scoped!(scope, hdr.highlights),
            Self::Shadows => crate::scoped!(scope, hdr.shadows),
            Self::Whites => crate::scoped!(scope, hdr.whites),
            Self::Blacks => crate::scoped!(scope, hdr.blacks),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpeedEdit {
    pub param: SpeedParam,
    pub since: Instant,
    /// Scrolled or dragged at least once.
    pub used: bool,
    /// The cursor to return to once a used Speed Edit key is released.
    restore: CursorTool,
}

impl SpeedEdit {
    /// Whether the overlay should show: as soon as it is used, or once the key
    /// has been held long enough that it is not a tap.
    pub fn visible(&self) -> bool {
        self.used || self.since.elapsed() >= HOLD
    }
}

const HOLD: Duration = Duration::from_millis(280);

impl AppState {
    /// Range, default and decimals of a Speed Edit parameter, from the contract.
    pub fn speed_meta(&self, param: SpeedParam) -> (f32, f32, usize) {
        match self.engine.param_metadata().iter().find(|m| m.key == param.c1_key()) {
            Some(m) => (m.min, m.max, m.decimals as usize),
            None => (-100.0, 100.0, 0),
        }
    }

    pub fn speed_value(&self, param: SpeedParam) -> f32 {
        (param.binding(self.scope()).get)(self.adj())
    }

    pub fn speed_begin(&mut self, param: SpeedParam, cx: &mut Context<Self>) {
        if self.speed.as_ref().is_some_and(|s| s.param == param) || self.current.is_none() {
            return;
        }
        // The key may already have switched the cursor in this same keystroke.
        let restore = if self.cursor_changed.elapsed() < Duration::from_millis(80) { self.prev_cursor } else { self.cursor };
        self.speed = Some(SpeedEdit { param, since: Instant::now(), used: false, restore });
        // Show the overlay once the key has clearly been held.
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(HOLD).await;
            let _ = this.update(cx, |_, cx| cx.notify());
        })
        .detach();
    }

    /// Sets the Speed Edit parameter to `value`, clamped, as part of one gesture.
    pub fn speed_set(&mut self, value: f32, cx: &mut Context<Self>) {
        let Some(speed) = self.speed.as_mut() else { return };
        speed.used = true;
        let param = speed.param;
        let (min, max, _) = self.speed_meta(param);
        let set = param.binding(self.scope()).set;
        let v = value.clamp(min, max);
        self.edit_live(format!("{} {}", t("Speed Edit", "快速编辑"), param.label()), move |a| set(a, v), cx);
    }

    /// Nudges the parameter by `steps` (one wheel notch is one step, 1/100 of its range).
    pub fn speed_nudge(&mut self, steps: f32, cx: &mut Context<Self>) {
        let Some(param) = self.speed.as_ref().map(|s| s.param) else { return };
        let (min, max, _) = self.speed_meta(param);
        let value = self.speed_value(param) + steps * (max - min) / 100.0;
        self.speed_set(value, cx);
    }

    pub fn speed_end(&mut self, cx: &mut Context<Self>) {
        let Some(speed) = self.speed.take() else { return };
        if speed.used {
            self.end_gesture(cx);
            self.set_cursor(speed.restore, cx);
        }
        cx.notify();
    }
}
