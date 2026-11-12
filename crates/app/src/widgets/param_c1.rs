//! Sliders for Capture One parameters from `docs/frontend-contract/params.json`.
//! Every tool asks for a slider by its C1 key; label, range, default, step,
//! decimals and unit come from the engine's [`lumen_engine::ParamMeta`], with
//! the current image's RAW-derived overrides applied.

use gpui_kit::{App, Entity, SharedString};

use super::slider::{ParamSlider, param};
use crate::i18n::t;
use crate::state::{AppState, Binding};

thread_local! {
    /// The current image's RAW-derived range overrides, fetched once per image.
    static OVERRIDES: std::cell::RefCell<Option<(lumen_engine::ImageId, lumen_engine::ParamOverrides)>> = const { std::cell::RefCell::new(None) };
}

/// A slider for a contract parameter (`docs/frontend-contract/params.json`):
/// label, range, default, step, decimals and unit come from the engine's
/// `ParamMeta` for `key`, with the current image's RAW overrides applied.
///
/// Where `AdjustmentSet` documents a different unit than the contract's C1
/// storage unit, the caller narrows it back with [`ParamSlider::range`].
pub fn param_c1(state: &Entity<AppState>, id: impl Into<SharedString>, key: &'static str, binding: Binding, cx: &App) -> ParamSlider {
    let s = state.read(cx);
    let Some(meta) = s.engine.param_metadata().iter().find(|m| m.key == key) else {
        return param(state, id, key, binding, 0.0, 100.0, cx);
    };
    let (min, max, default) = match s.current {
        Some(image) => OVERRIDES.with_borrow_mut(|cache| {
            if cache.as_ref().is_none_or(|(id, _)| *id != image) {
                *cache = Some((image, s.engine.param_overrides(image)));
            }
            cache.as_ref().map(|(_, o)| o.apply(meta)).unwrap_or((meta.min, meta.max, meta.default))
        }),
        None => (meta.min, meta.max, meta.default),
    };
    // A RAW-sourced row's static default is only a neutral fallback; unless the
    // image supplied its own, reset to what the engine gives a fresh image, so
    // an untouched image never reads as edited.
    let fresh = (binding.get)(&lumen_engine::AdjustmentSet::default());
    let default = if meta.source == lumen_engine::MetaSource::RawImage && default == meta.default { fresh } else { default };
    let slider = param(state, id, t(meta.label, meta.label_zh), binding, min, max, cx)
        .default_value(default)
        .decimals(meta.decimals as usize)
        .unit(crate::contract::unit_suffix(meta.unit));
    if meta.step > 0.0 { slider.step(meta.step) } else { slider }
}

/// Forgets the cached overrides, for example when a different catalog opens.
pub fn reset_overrides() {
    OVERRIDES.with_borrow_mut(|cache| *cache = None);
}
