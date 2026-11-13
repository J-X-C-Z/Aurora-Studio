//! Built-in styles, shared by both engines, and the rule for applying one.
//!
//! Capture One applies a style per tool group: a Film style replaces the White
//! Balance, Exposure and Color Balance settings it defines and leaves the other
//! tools alone. We do the same, with the tool groups being the fields of
//! [`AdjustmentSet`]: when a style defines anything in a group (any field of it
//! differs from the default), it replaces that whole group; a group the style
//! does not touch keeps the current values.

use std::borrow::Cow;

use crate::{AdjustmentSet, ColorWheel, CurvePoint, SplitTone, StyleInfo};

pub fn all() -> Vec<StyleInfo> {
    let mut out = Vec::new();
    let mut add = |group: &str, name: &str, f: &dyn Fn(&mut AdjustmentSet)| {
        let mut a = AdjustmentSet::default();
        f(&mut a);
        out.push(StyleInfo { name: name.into(), group: group.into(), adjustments: a });
    };
    add("Film", "Portra Warm", &|a| {
        a.white_balance.kelvin = 6200.0;
        a.exposure.saturation = -8.0;
        a.color_balance.highlight = ColorWheel { hue: 40.0, saturation: 0.25, lightness: 0.0 };
    });
    add("Film", "Ektar Vivid", &|a| {
        a.exposure.saturation = 25.0;
        a.exposure.contrast = 12.0;
    });
    add("Film", "Tri-X Push", &|a| {
        a.black_white.enabled = true;
        a.exposure.contrast = 30.0;
        a.film_grain.impact = 40.0;
    });
    add("Black & White", "Soft Silver", &|a| {
        a.black_white.enabled = true;
        a.black_white.highlight_tone = SplitTone { hue: 45.0, saturation: 12.0 };
    });
    add("Black & White", "Selenium", &|a| {
        a.black_white.enabled = true;
        a.black_white.shadow_tone = SplitTone { hue: 260.0, saturation: 20.0 };
    });
    add("Cinematic", "Teal & Orange", &|a| {
        a.color_balance.shadow = ColorWheel { hue: 190.0, saturation: 0.45, lightness: 0.0 };
        a.color_balance.highlight = ColorWheel { hue: 30.0, saturation: 0.4, lightness: 0.0 };
        a.exposure.contrast = 10.0;
    });
    add("Cinematic", "Faded Matte", &|a| {
        a.curves.rgb.points =
            vec![CurvePoint::new(0.0, 0.12), CurvePoint::new(0.5, 0.5), CurvePoint::new(1.0, 0.94)];
        a.exposure.saturation = -15.0;
    });
    add("Landscape", "Crisp Air", &|a| {
        a.dehaze.amount = 25.0;
        a.clarity.clarity = 20.0;
        a.clarity.structure = 15.0;
    });
    out
}

pub fn lookup(name: &str) -> Option<StyleInfo> {
    all()
        .into_iter()
        .find(|s| s.name == name)
        // Capture One's own presets and styles, when the install is present.
        .or_else(|| crate::presets::style_by_name(name))
}

pub fn apply_style(base: &AdjustmentSet, style: &AdjustmentSet) -> AdjustmentSet {
    let default = AdjustmentSet::default();
    let mut out = base.clone();
    macro_rules! group {
        ($field:ident) => {
            if style.$field != default.$field {
                out.$field = style.$field.clone();
            }
        };
    }
    group!(white_balance);
    group!(exposure);
    group!(hdr);
    group!(levels);
    group!(curves);
    group!(clarity);
    group!(dehaze);
    group!(vignetting);
    group!(color_balance);
    group!(color_editor);
    group!(black_white);
    group!(sharpening);
    group!(noise_reduction);
    group!(film_grain);
    group!(moire);
    group!(lens);
    group!(geometry);
    // Base characteristics: a C1 preset can carry the film curve and the tone
    // curve, but not the camera's input profile, so those stay.
    if style.base.film_curve != default.base.film_curve {
        out.base.film_curve = style.base.film_curve.clone();
    }
    if style.base.curve != default.base.curve {
        out.base.curve = style.base.curve;
    }
    if !style.spots.is_empty() {
        out.spots = style.spots.clone();
    }
    if !style.layers.is_empty() {
        out.layers = style.layers.clone();
    }
    out
}

/// Adjustments as rendered: the style's fields laid over the current ones.
pub fn resolve(adj: &AdjustmentSet) -> Cow<'_, AdjustmentSet> {
    match adj.style.as_deref().and_then(lookup) {
        Some(style) => Cow::Owned(apply_style(adj, &style.adjustments)),
        None => Cow::Borrowed(adj),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_grouped() {
        let all = all();
        let mut names: Vec<&str> = all.iter().map(|s| s.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), all.len());
        assert!(all.iter().any(|s| s.group == "Film"));
    }

    #[test]
    fn apply_replaces_the_whole_group_the_style_defines() {
        let mut base = AdjustmentSet::default();
        base.exposure.exposure = 0.75;
        base.noise_reduction.luminance = 30.0;
        let style = lookup("Ektar Vivid").expect("style");
        let out = apply_style(&base, &style.adjustments);
        // Exposure is one tool group: the style defines it, so it replaces it
        // whole, and the group the style does not define is left alone.
        assert_eq!(out.exposure.exposure, 0.0);
        assert_eq!(out.noise_reduction.luminance, 30.0);
        assert_eq!(out.exposure.contrast, 12.0);
        assert_eq!(out.exposure.saturation, 25.0);
    }

    #[test]
    fn a_style_that_only_changes_exposure_leaves_the_white_balance_alone() {
        let mut base = AdjustmentSet::default();
        base.white_balance.kelvin = 3200.0;
        base.white_balance.tint = -8.0;
        let mut style = AdjustmentSet::default();
        style.exposure.exposure = 0.5;
        let out = apply_style(&base, &style);
        assert_eq!(out.exposure.exposure, 0.5);
        assert_eq!(out.white_balance.kelvin, 3200.0);
        assert_eq!(out.white_balance.tint, -8.0);
    }

    #[test]
    fn the_curve_group_is_replaced_whole_leaving_no_old_points() {
        let mut base = AdjustmentSet::default();
        base.curves.rgb.points =
            vec![CurvePoint::new(0.0, 0.0), CurvePoint::new(0.25, 0.4), CurvePoint::new(1.0, 1.0)];
        let mut style = AdjustmentSet::default();
        style.curves.rgb.points = vec![CurvePoint::new(0.0, 0.1), CurvePoint::new(1.0, 0.95)];
        let out = apply_style(&base, &style);
        assert_eq!(out.curves.rgb.points, style.curves.rgb.points);
        assert_eq!(out.curves.rgb.points.len(), 2);
    }

    #[test]
    fn apply_overwrites_a_field_the_style_sets() {
        let mut base = AdjustmentSet::default();
        base.white_balance.kelvin = 3200.0;
        let style = lookup("Portra Warm").expect("style");
        assert_eq!(apply_style(&base, &style.adjustments).white_balance.kelvin, 6200.0);
    }

    #[test]
    fn apply_against_a_default_style_is_the_base() {
        let mut base = AdjustmentSet::default();
        base.exposure.exposure = 1.0;
        base.layers.push(crate::Layer::default());
        assert_eq!(apply_style(&base, &AdjustmentSet::default()), base);
    }

    #[test]
    fn resolve_ignores_an_unknown_style() {
        let mut adj = AdjustmentSet::default();
        adj.style = Some("No Such Style".into());
        assert!(matches!(resolve(&adj), Cow::Borrowed(_)));
    }

    #[test]
    fn resolve_applies_a_known_style_once() {
        let mut adj = AdjustmentSet::default();
        adj.style = Some("Tri-X Push".into());
        let once = resolve(&adj).into_owned();
        assert!(once.black_white.enabled);
        assert_eq!(once.exposure.contrast, 30.0);
        let twice = apply_style(&once, &lookup("Tri-X Push").unwrap().adjustments);
        assert_eq!(once, twice);
    }

    fn preview(engine: &crate::MockEngine, adj: AdjustmentSet) -> crate::RgbaImage {
        use crate::ImageEngine as _;
        let image = engine.images()[0].id;
        engine
            .render_preview(&crate::PreviewRequest {
                image,
                adjustments: adj,
                region: crate::NormRect::FULL,
                max_size: 64,
                quality: crate::RenderQuality::Draft,
                bypass: false,
                mask_overlay: None,
                generation: 0,
                display_profile: None,
                proof_profile: None,
                proof_paper_white: 1.0,
                proof_black_point: 0.0,
            })
            .unwrap()
            .image
    }

    #[test]
    fn unknown_style_renders_exactly_like_no_style() {
        let engine = crate::MockEngine::new();
        let mut adj = AdjustmentSet::default();
        adj.exposure.exposure = 0.4;
        let plain = preview(&engine, adj.clone());
        adj.style = Some("No Such Style".into());
        assert_eq!(preview(&engine, adj).pixels, plain.pixels);
    }

    #[test]
    fn known_style_reaches_the_render() {
        let engine = crate::MockEngine::new();
        let mut adj = AdjustmentSet::default();
        let plain = preview(&engine, adj.clone());
        adj.style = Some("Tri-X Push".into());
        assert_ne!(preview(&engine, adj).pixels, plain.pixels);
    }
}
