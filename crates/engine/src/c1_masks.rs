//! Capture One local mask layers (`ZVARIANTLAYER.Z_ENT = 23`).
//!
//! Column semantics, confirmed against `~/agents/evidence/c1-real.cocatalogdb`
//! (opened read-only): the catalog holds 6 local-mask rows in 4 variants.
//!
//! ```text
//! Z_ENT       → row kind; 23 = local / brush mask layer
//! Z_PK        → row id
//! ZVARIANT    → the variant the layer belongs to
//! ZINDEX      → layer order inside its variant; local rows of variant 1 are
//!               0,1,2,3,4 and variant 4 has one row at 0
//! ZNAME       → user-visible layer name (`Skin Mask 1`, `背景遮罩 1`, …)
//! ZMASKUUID   → one UUID per layer, unique across the 6 rows
//! ZENABLED    → 1 on every row
//! ZOPACITY    → 100.0 on every row
//! ZSURFACESMOOTHNESSAMOUNT → 0.0 on every row
//! ZBRUSHSIZE  → 12.96581745147705 on 5 rows, 100.0 on 1
//! ZBRUSHHARDNESS / ZBRUSHOPACITY / ZBRUSHFLOW → 80 / 100 / 100
//! ZBRUSHUSEPENPRESSURE → 1, ZBRUSHAIRBRUSH / ZBRUSHAUTOMASK → 0
//! ZERASER*    → C1 defaults (100 / 80 / 100 / 100) on every row
//! ```
//!
//! Not proven: stroke geometry. `ZVARIANTLAYER` has no BLOB column and the
//! evidence catalog ships no mask sidecar files, so brush points cannot be
//! recovered here; callers pass them in. `ZBRUSH*` / `ZERASER*` are the *tool
//! state at save time* (each local layer stores its current brush and eraser
//! settings), not necessarily the settings each stroke was painted with, so
//! they are used as the default for the strokes supplied by the caller. The
//! direction of `ZINDEX` (whether a higher value draws on top) is also
//! unproven — only the ordering itself is observed. This module keeps
//! ascending order and the renderer paints layers in list order.

//! Local colour adjustments ride on the same row: `ZWHITEBALANCE`,
//! `ZHIGHLIGHTRECOVERY` / `ZSHADOWRECOVERY`, `ZCLARITY` /
//! `ZCLARITYSTRUCTURE`, `ZCONTRAST`, `ZBRIGHTNESS`, `ZSATURATION`,
//! `ZCOLORCORRECTIONS` (only groups with a non-zero shift are an edit; the
//! eight basic bands are always stored), `ZGRADATIONCURVE*` / `ZLEVELS*`, and
//! `ZBW*`. They map onto [`LocalAdjustments`], which `local()` applies.

use std::collections::HashMap;

use crate::adjust::{
    Clarity, ClarityMethod, Curves, Dehaze, Exposure, HighDynamicRange, Levels, WhiteBalance,
};
use crate::comask::{ComaskDocument, ComaskMask};
use crate::c1_model::{
    black_white_from_c1, color_editor_from_c1, curves_from_c1, from_layer_row,
    gains_to_kelvin_tint, levels_from_c1, C1Brush, C1Eraser, C1EntityKind, ColorCorrection,
};
use crate::layers::{
    BrushStroke, Layer, LayerKind, LocalAdjustments, Mask, MaskOp, MaskShape,
};

/// The brush-tool columns of one local mask layer.
#[derive(Clone, Debug, PartialEq)]
pub struct C1MaskLayer {
    /// `ZINDEX`: order inside the variant.
    pub index: i64,
    pub name: Option<String>,
    pub mask_uuid: Option<String>,
    pub enabled: bool,
    /// `ZOPACITY`, 0 ..= 100.
    pub opacity: f32,
    /// `ZSURFACESMOOTHNESSAMOUNT`.
    pub surface_smoothness: f32,
    pub brush: C1Brush,
    pub eraser: C1Eraser,
    /// The layer's own colour adjustments, already in engine form.
    pub adjustments: LocalAdjustments,
}

/// Display-only summary of a local layer. Mask pixels are deliberately absent.
#[derive(Clone, Debug, PartialEq)]
pub struct C1LayerSummary {
    pub index: i64,
    pub name: String,
    pub enabled: bool,
    pub mask_uuid: Option<String>,
    /// Engine layer kind. C1's finer mask type is not a `ZVARIANTLAYER` column.
    pub kind: LayerKind,
}

impl C1MaskLayer {
    pub fn summary(&self) -> C1LayerSummary {
        C1LayerSummary {
            index: self.index,
            name: self.name.clone().unwrap_or_else(|| format!("Layer {}", self.index)),
            enabled: self.enabled,
            mask_uuid: self.mask_uuid.clone(),
            kind: LayerKind::Adjustment,
        }
    }
}

/// Summaries in the same order as [`mask_layers_from_rows`].
pub fn layer_summaries(layers: &[C1MaskLayer]) -> Vec<C1LayerSummary> {
    layers.iter().map(C1MaskLayer::summary).collect()
}

/// Reads one `ZVARIANTLAYER` row; `None` when it is not a local mask layer.
pub fn mask_layer_from_row(row: &HashMap<&str, Option<String>>) -> Option<C1MaskLayer> {
    let s = from_layer_row(row);
    (s.entity == C1EntityKind::LocalMask).then(|| {
        let (kelvin, tint) = gains_to_kelvin_tint(s.white_balance_gains);
        let curves = curves_from_c1(&s);
        let levels = levels_from_c1(&s);
        let black_white = black_white_from_c1(&s);
        let adjustments = LocalAdjustments {
            white_balance: (s.white_balance_gains != [1.0; 3])
                .then_some(WhiteBalance { kelvin, tint, gains: None, ..Default::default() }),
            exposure: Exposure {
                exposure: s.exposure,
                contrast: s.contrast,
                brightness: s.brightness,
                saturation: s.saturation,
            },
            hdr: HighDynamicRange {
                highlights: s.highlights_recovery,
                shadows: s.shadows_recovery,
                whites: s.whites_recovery,
                blacks: s.blacks_recovery,
            },
            clarity: Clarity {
                // Numeric mapping of ZCLARITYMETHOD is unproven; keep C1's order.
                method: ClarityMethod::ALL[(s.clarity_method.clamp(0, 3)) as usize],
                clarity: s.clarity,
                structure: s.clarity_structure,
            },
            dehaze: Dehaze {
                amount: s.dehaze_amount.clamp(0.0, 100.0),
                shadow_tone: s.dehaze_color,
            },
            curves: (curves != Curves::default()).then_some(curves),
            levels: (levels != Levels::default()).then_some(levels),
            black_white: black_white.enabled.then_some(black_white),
            color_editor: if s
                .color_corrections
                .iter()
                .any(|row| ColorCorrection::from_row(row).affects_color())
            {
                color_editor_from_c1(&s.color_corrections)
            } else {
                Default::default()
            },
            ..Default::default()
        };
        C1MaskLayer {
            index: s.index,
            name: s.name,
            mask_uuid: s.mask_uuid,
            enabled: s.enabled,
            opacity: s.opacity,
            surface_smoothness: s.surface_smoothness,
            brush: s.brush,
            eraser: s.eraser,
            adjustments,
        }
    })
}

/// Reads every local mask row and orders it by `ZINDEX`.
pub fn mask_layers_from_rows(rows: &[HashMap<&str, Option<String>>]) -> Vec<C1MaskLayer> {
    let mut out: Vec<C1MaskLayer> = rows.iter().filter_map(mask_layer_from_row).collect();
    out.sort_by_key(|l| l.index);
    out
}

impl C1MaskLayer {
    /// Links this catalog row to its `.comask` metadata by UUID. Geometry is
    /// intentionally not implied by this lookup.
    pub fn comask_metadata<'a>(&self, document: &'a ComaskDocument) -> Option<&'a ComaskMask> {
        let uuid = self.mask_uuid.as_deref()?;
        document.masks.iter().find(|mask| mask.uuid.as_deref() == Some(uuid))
    }

    /// Builds the render layer. The catalog stores no stroke geometry (see the
    /// module docs), so `add` and `subtract` carry the normalized points with
    /// pressure; `long_side_px` converts C1's pixel brush size to the
    /// normalized size [`BrushStroke`] expects. Eraser strokes become
    /// [`MaskOp::Subtract`] shapes.
    pub fn to_layer(
        &self,
        id: u64,
        long_side_px: f32,
        add: Vec<[f32; 3]>,
        subtract: Vec<[f32; 3]>,
    ) -> Layer {
        let norm = |px: f32| px / long_side_px.max(1e-6);
        let mut shapes = Vec::new();
        if !add.is_empty() {
            shapes.push(MaskShape::Brush(BrushStroke {
                op: MaskOp::Add,
                points: add,
                size: norm(self.brush.size),
                hardness: self.brush.hardness,
                opacity: self.brush.opacity,
                flow: self.brush.flow,
                auto_mask: self.brush.auto_mask,
            }));
        }
        if !subtract.is_empty() {
            shapes.push(MaskShape::Brush(BrushStroke {
                op: MaskOp::Subtract,
                points: subtract,
                size: norm(self.eraser.size),
                hardness: self.eraser.hardness,
                opacity: self.eraser.opacity,
                flow: self.eraser.flow,
                auto_mask: self.eraser.auto_mask,
            }));
        }
        Layer {
            id,
            name: self.name.clone().unwrap_or_else(|| format!("Layer {}", self.index)),
            kind: LayerKind::Adjustment,
            enabled: self.enabled,
            opacity: self.opacity,
            mask: Mask {
                shapes,
                // C1 exposes no mask-edge column in ZVARIANTLAYER (the .comask
                // sidecar is absent), so its surface-smoothness value drives the
                // engine's edge refine; both use the same 0..=100 scale.
                refine: self.surface_smoothness.clamp(0.0, 100.0),
                ..Default::default()
            },
            adjustments: self.adjustments.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pairs: &[(&'static str, Option<&str>)], map: &mut HashMap<&'static str, Option<String>>) {
        for (k, v) in pairs {
            map.insert(k, v.map(str::to_string));
        }
    }

    fn layer_row(index: &str, name: &str, size: &str) -> HashMap<&'static str, Option<String>> {
        let mut m = HashMap::new();
        row(
            &[
                ("Z_ENT", Some("23")),
                ("ZINDEX", Some(index)),
                ("ZNAME", Some(name)),
                ("ZMASKUUID", Some("24391591-4129-4c9f-a9f5-bb4dd4c41289")),
                ("ZENABLED", Some("1")),
                ("ZOPACITY", Some("100")),
                ("ZSURFACESMOOTHNESSAMOUNT", Some("0.0")),
                ("ZBRUSHSIZE", Some(size)),
                ("ZBRUSHHARDNESS", Some("80")),
                ("ZBRUSHOPACITY", Some("90")),
                ("ZBRUSHFLOW", Some("70")),
                ("ZERASERSIZE", Some("50")),
                ("ZERASERHARDNESS", Some("60")),
            ],
            &mut m,
        );
        m
    }

    #[test]
    fn brush_columns_map_into_the_layer() {
        let rows = [layer_row("2", "Skin Mask 1", "500")];
        let layers = mask_layers_from_rows(&rows);
        assert_eq!(layers.len(), 1);
        let layer = layers[0].to_layer(7, 1000.0, vec![[0.5, 0.5, 1.0]], vec![]);
        assert_eq!((layer.id, layer.name.as_str()), (7, "Skin Mask 1"));
        assert_eq!(layer.opacity, 100.0);
        match &layer.mask.shapes[..] {
            [MaskShape::Brush(b)] => {
                assert_eq!(b.op, MaskOp::Add);
                assert_eq!((b.hardness, b.opacity, b.flow), (80.0, 90.0, 70.0));
                assert!((b.size - 0.5).abs() < 1e-6, "500 px of 1000 must be 0.5");
            }
            other => panic!("expected one brush shape, got {other:?}"),
        }
    }

    #[test]
    fn eraser_strokes_subtract_with_eraser_settings() {
        let rows = [layer_row("0", "Mask", "200")];
        let layer = mask_layers_from_rows(&rows)[0].to_layer(
            1,
            1000.0,
            vec![[0.4, 0.5, 1.0]],
            vec![[0.6, 0.5, 1.0]],
        );
        assert_eq!(layer.mask.shapes.len(), 2);
        match &layer.mask.shapes[1] {
            MaskShape::Brush(b) => {
                assert_eq!(b.op, MaskOp::Subtract);
                assert_eq!((b.hardness, b.opacity), (60.0, 100.0));
                assert!((b.size - 0.05).abs() < 1e-6);
            }
            other => panic!("expected the eraser brush, got {other:?}"),
        }
    }

    #[test]
    fn zindex_orders_the_layers() {
        let rows = [
            layer_row("4", "top", "100"),
            layer_row("0", "base", "100"),
            layer_row("2", "middle", "100"),
        ];
        let names: Vec<String> =
            mask_layers_from_rows(&rows).into_iter().map(|l| l.name.unwrap()).collect();
        assert_eq!(names, ["base", "middle", "top"]);
    }

    /// PK 7 (`背景遮罩 1`) stores local tone and colour edits on the mask row;
    /// they must reach the layer, not be dropped.
    #[test]
    fn local_colour_adjustments_reach_the_layer() {
        let mut m = layer_row("1", "背景遮罩 1", "12.9658");
        row(
            &[
                ("ZCLARITY", Some("-33.62")),
                ("ZCLARITYSTRUCTURE", Some("-34.78")),
                ("ZHIGHLIGHTRECOVERY", Some("57.39")),
                ("ZSHADOWRECOVERY", Some("6.667")),
                ("ZCONTRAST", Some("3.768")),
                ("ZDEHAZEAMOUNT", Some("22.5")),
                ("ZDEHAZECOLOR", Some("0.1;0.2;0.3")),
                ("ZWHITEBALANCE", Some("1.385986;1.000000;1.524762")),
                ("ZSATURATION", Some("13.33")),
                ("ZBRIGHTNESS", Some("7.536")),
            ],
            &mut m,
        );
        let layer = mask_layer_from_row(&m).unwrap().to_layer(3, 1000.0, vec![], vec![]);
        let a = &layer.adjustments;
        assert_eq!((a.clarity.clarity, a.clarity.structure), (-33.62, -34.78));
        assert_eq!((a.hdr.highlights, a.hdr.shadows), (57.39, 6.667));
        assert_eq!(a.exposure.contrast, 3.768);
        assert_eq!(a.dehaze.amount, 22.5);
        assert_eq!(a.dehaze.shadow_tone, [0.1, 0.2, 0.3]);
        assert!(a.white_balance.is_some(), "a non-neutral ZWHITEBALANCE must survive");
        assert_eq!((a.exposure.saturation, a.exposure.brightness), (13.33, 7.536));
    }

    /// Both rows are real C1 editor state: the eight basic bands are always
    /// stored, but only a non-zero shift is an edit.
    const NEUTRAL_CORRECTIONS: &str =
        "1,1,1,0,0,0,255,0,0,-10,10,-100,100,15,0,0,0,0";
    const RED_SHIFT_CORRECTIONS: &str =
        "1,1,1,20,30,0,255,0,0,-10,10,-100,100,15,0,0,0,0";

    fn corrections_row(corrections: &'static str) -> HashMap<&'static str, Option<String>> {
        let mut m = layer_row("0", "背景遮罩 1", "100");
        row(&[("ZCOLORCORRECTIONS", Some(corrections))], &mut m);
        m
    }

    #[test]
    fn neutral_color_corrections_do_not_populate_the_editor() {
        let parsed = mask_layer_from_row(&corrections_row(NEUTRAL_CORRECTIONS)).unwrap();
        assert_eq!(parsed.adjustments.color_editor, Default::default());
    }

    #[test]
    fn colour_editor_shift_reaches_the_layer() {
        let parsed = mask_layer_from_row(&corrections_row(RED_SHIFT_CORRECTIONS)).unwrap();
        let shift =
            parsed.adjustments.color_editor.basic[crate::adjust::HueSector::Red as usize];
        assert_eq!((shift.hue, shift.saturation, shift.lightness), (20.0, 30.0, 0.0));
    }

    /// The mapped editor must be neutral on a plain row and change pixels when
    /// a real shift is present.
    #[test]
    fn colour_editor_mapping_changes_the_rendered_output_only_when_edited() {
        use crate::{render, AdjustmentSet, NormRect, Source};

        let rgba: Vec<u8> = (0..16 * 16).flat_map(|_| [220, 40, 30, 255]).collect();
        let src = Source::from_rgba(16, 16, &rgba);
        let render_layer = |corrections| {
            let layer = mask_layer_from_row(&corrections_row(corrections))
                .unwrap()
                .to_layer(1, 1000.0, vec![], vec![]);
            let set = AdjustmentSet { layers: vec![layer], ..Default::default() };
            render(&src, &set, NormRect::FULL, 16, false, None).rgba
        };
        let plain = render(&src, &AdjustmentSet::default(), NormRect::FULL, 16, false, None).rgba;
        assert_eq!(render_layer(NEUTRAL_CORRECTIONS), plain);
        assert_ne!(render_layer(RED_SHIFT_CORRECTIONS), plain);
    }

    /// Real rows store identity gradation curves and neutral levels; those are
    /// not an edit and must stay absent.
    #[test]
    fn neutral_curves_and_levels_stay_absent() {
        let mut m = layer_row("0", "Mask", "100");
        row(
            &[
                ("ZGRADATIONCURVE", Some("0.000,0.000;1.000,1.000")),
                ("ZGRADATIONCURVERED", Some("0.000,0.000;1.000,1.000")),
                ("ZLEVELSSHADOW", Some("0.000;0.000;0.000;0.000")),
                ("ZLEVELSHIGHLIGHT", Some("1.000;1.000;1.000;1.000")),
                ("ZMIDTONE", Some("0.000;0.000;0.000;0.000")),
                ("ZLEVELSTARGETHIGHLIGHT", Some("1.000;1.000;1.000;1.000")),
            ],
            &mut m,
        );
        let a = mask_layer_from_row(&m).unwrap().adjustments;
        assert!(a.curves.is_none(), "identity curves are not an edit");
        assert!(a.levels.is_none(), "neutral levels are not an edit");
    }

    /// The local rows carry the same `ZGRADATIONCURVE*` / `ZLEVELS*` columns as
    /// a variant, so a real edit must reach the layer unclamped.
    #[test]
    fn real_curve_and_levels_columns_reach_the_layer() {
        let mut m = layer_row("0", "Mask", "100");
        row(
            &[
                ("ZGRADATIONCURVE", Some("0.000,0.000;0.500,0.900;1.000,1.000")),
                ("ZGRADATIONCURVEBLUE", Some("0.000,0.000;0.500,0.400;1.000,1.000")),
                ("ZLEVELSSHADOW", Some("0.100;0.000;0.000;0.000")),
                ("ZLEVELSHIGHLIGHT", Some("0.900;1.000;1.000;1.000")),
                ("ZMIDTONE", Some("0.200;0.000;0.000;0.000")),
                ("ZLEVELSTARGETHIGHLIGHT", Some("1.000;1.000;1.000;0.900")),
            ],
            &mut m,
        );
        let a = mask_layer_from_row(&m).unwrap().adjustments;
        let curves = a.curves.expect("a real curve must reach the layer");
        assert_eq!(curves.rgb.points.len(), 3);
        assert_eq!((curves.rgb.points[1].x, curves.rgb.points[1].y), (0.5, 0.9));
        assert!(curves.blue.points[1].y < 0.5, "the blue curve must stay separate");
        let levels = a.levels.expect("real levels must reach the layer");
        assert_eq!((levels.rgb.input_black, levels.rgb.input_white), (0.1, 0.9));
        assert!(levels.rgb.gamma > 1.0, "a positive midtone lifts midtones");
        assert_eq!((levels.rgb.output_white, levels.blue.output_white), (1.0, 0.9));
    }

    /// A neutral local row must leave the frame byte-identical; a real curve or
    /// levels edit must change it.
    #[test]
    fn local_curve_and_levels_only_change_the_frame_when_edited() {
        use crate::{render, AdjustmentSet, NormRect, Source};

        let rgba: Vec<u8> = (0..16 * 16).flat_map(|_| [90, 110, 160, 255]).collect();
        let src = Source::from_rgba(16, 16, &rgba);
        let render_columns = |columns: &[(&'static str, Option<&'static str>)]| {
            let mut m = layer_row("0", "Mask", "100");
            row(columns, &mut m);
            let layer = mask_layer_from_row(&m)
                .unwrap()
                .to_layer(1, 1000.0, vec![], vec![]);
            let set = AdjustmentSet { layers: vec![layer], ..Default::default() };
            render(&src, &set, NormRect::FULL, 16, false, None).rgba
        };
        let plain = render(&src, &AdjustmentSet::default(), NormRect::FULL, 16, false, None).rgba;
        assert_eq!(
            render_columns(&[("ZGRADATIONCURVE", Some("0.000,0.000;1.000,1.000"))]),
            plain
        );
        assert_ne!(
            render_columns(&[("ZGRADATIONCURVE", Some("0.000,0.000;0.500,0.900;1.000,1.000"))]),
            plain
        );
        assert_eq!(
            render_columns(&[
                ("ZLEVELSHIGHLIGHT", Some("1.000;1.000;1.000;1.000")),
                ("ZLEVELSTARGETHIGHLIGHT", Some("1.000;1.000;1.000;1.000")),
            ]),
            plain
        );
        assert_ne!(
            render_columns(&[("ZLEVELSSHADOW", Some("0.250;0.000;0.000;0.000"))]),
            plain
        );
    }

    /// C1 writes `ZBWENABLED=0` on a neutral row; even with channel values
    /// stored, the conversion is off and must stay absent.
    #[test]
    fn disabled_black_white_stays_absent() {
        let mut m = layer_row("0", "Mask", "100");
        row(
            &[("ZBWENABLED", Some("0")), ("ZBWRED", Some("80.0"))],
            &mut m,
        );
        let a = mask_layer_from_row(&m).unwrap().adjustments;
        assert!(a.black_white.is_none());
    }

    /// A real conversion must reach the layer with its channel sensitivities
    /// and tones, clamped to the same ranges the global path uses.
    #[test]
    fn black_white_columns_reach_the_layer() {
        let mut m = layer_row("0", "Mask", "100");
        row(
            &[
                ("ZBWENABLED", Some("1")),
                ("ZBWRED", Some("12.5")),
                ("ZBWYELLOW", Some("-40")),
                ("ZBWBLUE", Some("25")),
                ("ZBWDARKTONEHUE", Some("210")),
                ("ZBWDARKTONESATURATION", Some("8")),
            ],
            &mut m,
        );
        let bw = mask_layer_from_row(&m)
            .unwrap()
            .adjustments
            .black_white
            .expect("an enabled conversion must reach the layer");
        assert!(bw.enabled);
        assert_eq!(bw.sensitivity, [12.5, -40.0, 0.0, 0.0, 25.0, 0.0]);
        assert_eq!((bw.shadow_tone.hue, bw.shadow_tone.saturation), (210.0, 8.0));
    }

    /// Neutral stays byte-identical; an enabled conversion desaturates and its
    /// channel sensitivity changes the result.
    #[test]
    fn black_white_only_changes_the_frame_when_enabled() {
        use crate::{render, AdjustmentSet, NormRect, Source};

        let rgba: Vec<u8> = (0..16 * 16).flat_map(|_| [200, 60, 40, 255]).collect();
        let src = Source::from_rgba(16, 16, &rgba);
        let render_columns = |columns: &[(&'static str, Option<&'static str>)]| {
            let mut m = layer_row("0", "Mask", "100");
            row(columns, &mut m);
            let layer = mask_layer_from_row(&m)
                .unwrap()
                .to_layer(1, 1000.0, vec![], vec![]);
            let set = AdjustmentSet { layers: vec![layer], ..Default::default() };
            render(&src, &set, NormRect::FULL, 16, false, None).rgba
        };
        let plain = render(&src, &AdjustmentSet::default(), NormRect::FULL, 16, false, None).rgba;
        assert_eq!(
            render_columns(&[("ZBWENABLED", Some("0")), ("ZBWRED", Some("80"))]),
            plain
        );
        let gray = render_columns(&[("ZBWENABLED", Some("1"))]);
        assert_ne!(gray, plain, "an enabled conversion must change the frame");
        assert!(
            gray.chunks_exact(4).all(|p| p[0] == p[1] && p[1] == p[2]),
            "black and white output must be neutral"
        );
        assert_ne!(
            render_columns(&[("ZBWENABLED", Some("1")), ("ZBWRED", Some("80"))]),
            gray,
            "red sensitivity must change the converted output"
        );
    }

    #[test]
    fn comask_metadata_matches_the_mask_uuid() {
        let rows = [layer_row("1", "背景遮罩 1", "12.9658")];
        let layer = mask_layer_from_row(&rows[0]).unwrap();
        let document = ComaskDocument {
            sections: Vec::new(),
            masks: vec![ComaskMask {
                index: 7,
                uuid: Some("24391591-4129-4c9f-a9f5-bb4dd4c41289".into()),
                mask_type: Some(3),
                generator: None,
                generator_options: None,
                comp_params: vec![1.0, 0.0, 0.0],
                linear_gradient: Some([1.0, 2.0, 3.0, 4.0, 0.5]),
                rand_seed: None,
                rasters: Vec::new(),
            }],
        };
        let metadata = layer.comask_metadata(&document).expect("UUID match");
        assert_eq!(metadata.index, 7);
        assert_eq!(metadata.mask_type, Some(3));
        assert_eq!(metadata.linear_gradient, Some([1.0, 2.0, 3.0, 4.0, 0.5]));
    }

    #[test]
    fn default_and_delta_rows_are_not_mask_layers() {
        let mut m = HashMap::new();
        row(&[("Z_ENT", Some("18")), ("ZINDEX", Some("0"))], &mut m);
        assert!(mask_layer_from_row(&m).is_none());
    }

    #[test]
    fn surface_smoothness_drives_the_mask_refine() {
        let mut m = layer_row("0", "Mask", "100");
        row(&[("ZSURFACESMOOTHNESSAMOUNT", Some("40"))], &mut m);
        let layer = mask_layer_from_row(&m).unwrap().to_layer(1, 1000.0, vec![], vec![]);
        assert_eq!(layer.mask.refine, 40.0);
    }

    #[test]
    fn absent_columns_keep_c1_defaults() {
        let mut m = HashMap::new();
        row(&[("Z_ENT", Some("23")), ("ZINDEX", Some("0"))], &mut m);
        let parsed = mask_layer_from_row(&m).unwrap();
        let layer = parsed.to_layer(1, 1000.0, vec![], vec![]);
        assert_eq!((layer.enabled, layer.opacity), (true, 100.0));
        assert_eq!(layer.mask.refine, 0.0);
        assert_eq!(
            (parsed.brush.size, parsed.brush.hardness, parsed.brush.opacity),
            (100.0, 80.0, 100.0)
        );
        assert_eq!(
            (parsed.eraser.size, parsed.eraser.hardness, parsed.eraser.opacity),
            (100.0, 80.0, 100.0)
        );
    }

    #[test]
    fn summaries_expose_name_enabled_and_mask_uuid() {
        let rows = [
            layer_row("2", "Skin Mask 1", "500"),
            layer_row("0", "背景遮罩 1", "100"),
        ];
        let layers = mask_layers_from_rows(&rows);
        let summaries = layer_summaries(&layers);
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].index, 0);
        assert_eq!(summaries[0].name, "背景遮罩 1");
        assert!(summaries[0].enabled);
        assert!(summaries[0].mask_uuid.is_some());
        assert_eq!(summaries[0].kind, LayerKind::Adjustment);
        assert_eq!(summaries[1].name, "Skin Mask 1");
    }
}
