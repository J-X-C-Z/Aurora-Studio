//! Z6III ColorManagement setup. Factors are resolved settings, not defaults.

/// `base` is in preference order (+0x11028, +0x1102C), `extra` likewise
/// (+0x11034, +0x11030). `iso` is the reported integer, if available.
pub fn z6iii_exposure_scale(
    ev: f32,
    special_mode: bool,
    base: [f32; 2],
    extra: [f32; 2],
    iso: Option<i32>,
) -> Option<f32> {
    if !ev.is_finite() || !base.into_iter().chain(extra).all(f32::is_finite) {
        return None;
    }
    let exposure = ev.exp2();
    let scale = if special_mode {
        exposure
    } else {
        let base = base.into_iter().find(|value| *value > 0.0).unwrap_or(1.0);
        let extra = if extra[0] > 0.0 { extra[0] } else { extra[1] };
        let adjusted = base * exposure;
        let adjusted = if extra > 0.001 { adjusted * extra } else { adjusted };
        sensitivity_scale(iso) * adjusted
    };
    (scale.is_finite() && scale > 0.0).then_some(scale)
}

fn sensitivity_scale(iso: Option<i32>) -> f32 {
    match iso {
        Some(value @ 1..=99) => value as f32 / 100.0,
        _ => 1.0,
    }
}

/// Legacy tone LUT `pre_scale` for the ordinary Z6III path (no special exposure
/// flags, film mode != 3). `base` is in +0x10CB8/+0x10CBC preference order;
/// the latter must already include the selected film object's native gain.
pub fn z6iii_legacy_pre_scale(
    ev: f32,
    base: [f32; 2],
    extra: [f32; 2],
    iso: Option<i32>,
    fold_exposure: bool,
    split_headroom: bool,
) -> Option<f32> {
    if !ev.is_finite() || !base.into_iter().chain(extra).all(f32::is_finite) {
        return None;
    }
    let exposure = ev.exp2();
    let base = base.into_iter().find(|value| *value > 0.0).unwrap_or(1.0);
    let extra = if extra[0] > 0.0 { extra[0] } else { extra[1] };
    let gain = base * exposure;
    let gain = if extra > 0.001 { gain * extra } else { gain };
    let sensitivity = sensitivity_scale(iso);
    let gain = gain * sensitivity;
    let scale = if !fold_exposure && split_headroom && gain > 1.0 {
        let headroom = (gain / exposure.max(1.0)).max(1.0);
        (gain / headroom) * headroom
    } else {
        gain
    };
    (scale.is_finite() && scale > 0.0).then_some(scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z6iii_exposure_precedence() {
        assert_eq!(z6iii_exposure_scale(1.0, false, [0.0, 1.5], [0.0, 2.0], Some(50)), Some(3.0));
        assert_eq!(z6iii_exposure_scale(1.0, false, [1.25, 1.5], [0.001, 2.0], Some(100)), Some(2.5));
        assert_eq!(z6iii_exposure_scale(1.0, true, [0.0, 1.5], [0.0, 2.0], Some(50)), Some(2.0));
        assert_eq!(z6iii_exposure_scale(f32::NAN, false, [0.0; 2], [0.0; 2], None), None);
    }

    #[test]
    fn z6iii_legacy_pre_scale_includes_iso_factor() {
        assert_eq!(z6iii_legacy_pre_scale(0.0, [0.0, 1.0], [0.0; 2], Some(50), false, false), Some(0.5));
        assert_eq!(z6iii_legacy_pre_scale(0.0, [0.0, 1.0], [0.0; 2], Some(200), false, true), Some(1.0));
        assert_eq!(z6iii_legacy_pre_scale(f32::INFINITY, [0.0, 1.0], [0.0; 2], None, false, false), None);
    }

}
