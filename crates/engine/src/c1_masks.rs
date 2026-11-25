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
//! they are used as the default for the strokes supplied by the caller.
//!
//! `ZINDEX` is the layer's position in C1's `LocalAdjustmentsLayerCollection`:
//! `Add` appends to the end and `TrySelectTopmostEnabledLayer` takes
//! `LastOrDefault`, so the end of the list is the topmost layer. Ascending
//! `ZINDEX` is therefore bottom-to-top; this module keeps that order and the
//! renderer paints in list order, so the highest index lands last on top.

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
    BrushStroke, Layer, LayerKind, LayerStyle, LocalAdjustments, Mask, MaskOp, MaskShape,
    RasterMask,
};

/// The brush-tool columns of one local mask layer.
#[derive(Clone, Debug, PartialEq)]
pub struct C1MaskLayer {
    /// `ZINDEX`: order inside the variant.
    pub index: i64,
    pub name: Option<String>,
    pub mask_uuid: Option<String>,
    /// `ZLAYERSELECTPOINT`: the viewer's layer handle in raw image pixels.
    /// UI-only; see [`crate::Layer::select_point`].
    pub select_point: Option<[f32; 2]>,
    pub enabled: bool,
    /// `ZOPACITY`, 0 ..= 100.
    pub opacity: f32,
    /// `ZSURFACESMOOTHNESSAMOUNT`.
    pub surface_smoothness: f32,
    pub brush: C1Brush,
    pub eraser: C1Eraser,
    /// `ZBRUSHSTYLEUUID` / `ZBRUSHSTYLENAME`: the selected brush preset.
    /// Parsed for round-trip bookkeeping; the renderer does not read them.
    pub brush_style_uuid: Option<String>,
    pub brush_style_name: Option<String>,
    /// `ZSTYLEUUID` / `ZSTYLESOURCE` / `ZSTYLEMODIFIED`: the style this layer
    /// was applied from. Round-trip bookkeeping only; no pixel path reads it.
    pub style_uuid: Option<String>,
    pub style_source: Option<String>,
    pub style_modified: bool,
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
                method: ClarityMethod::from_c1(s.clarity_method),
                clarity: s.clarity,
                structure: s.clarity_structure,
            },
            dehaze: Dehaze {
                amount: s.dehaze_amount.clamp(0.0, 100.0),
                shadow_tone: s.dehaze_color,
                level_low: s.dehaze_level_low,
                level_high: s.dehaze_level_high,
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
            ai_color_grade: s.ai_color_grade,
            ..Default::default()
        };
        C1MaskLayer {
            index: s.index,
            name: s.name,
            mask_uuid: s.mask_uuid,
            select_point: s.layer_select_point,
            enabled: s.enabled,
            opacity: s.opacity,
            surface_smoothness: s.surface_smoothness,
            brush: s.brush,
            eraser: s.eraser,
            brush_style_uuid: row.get("ZBRUSHSTYLEUUID").cloned().flatten(),
            brush_style_name: row.get("ZBRUSHSTYLENAME").cloned().flatten(),
            style_uuid: s.style_uuid,
            style_source: s.style_source,
            style_modified: s.style_modified,
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
                hair_brush: self.brush.hair_brush,
                use_pen_pressure: self.brush.use_pen_pressure,
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
                // C1's eraser has an air-brush flag, not a hair brush; the
                // closest stroke field stays false.
                hair_brush: false,
                use_pen_pressure: self.eraser.use_pen_pressure,
            }));
        }
        Layer {
            id,
            name: self.name.clone().unwrap_or_else(|| format!("Layer {}", self.index)),
            kind: LayerKind::Adjustment,
            enabled: self.enabled,
            opacity: self.opacity,
            select_point: self.select_point,
            mask: Mask {
                shapes,
                // C1 exposes no mask-edge column in ZVARIANTLAYER (the .comask
                // sidecar is absent), so its surface-smoothness value drives the
                // engine's edge refine; both use the same 0..=100 scale.
                refine: self.surface_smoothness.clamp(0.0, 100.0),
                ..Default::default()
            },
            adjustments: self.adjustments.clone(),
            mask_uuid: self.mask_uuid.clone(),
            style: self.style_uuid.as_ref().map(|uuid| LayerStyle {
                uuid: uuid.clone(),
                source: self.style_source.clone(),
                modified: self.style_modified,
            }),
            ..Default::default()
        }
    }
}

/// Longest side of the raster kept in [`RasterMask`]. The renderer builds its
/// mask grid 256 cells across, so 512 px is two samples per cell: enough to
/// keep the decoded edge, cheap enough to clone with the layer every frame.
const RASTER_LONG_SIDE: usize = 512;

/// Crops a decoded `.comask` plane to its non-zero bounding box and turns it
/// into an 8-bit [`RasterMask`]. `None` when the plane is empty.
pub fn raster_shape_from_plane(
    width: usize,
    height: usize,
    data: &[f32],
    op: MaskOp,
) -> Option<RasterMask> {
    if width == 0 || height == 0 || data.len() < width * height {
        return None;
    }
    let mut bounds: Option<(usize, usize, usize, usize)> = None;
    for y in 0..height {
        for x in 0..width {
            if data[y * width + x] <= 0.0 {
                continue;
            }
            bounds = Some(match bounds {
                None => (x, y, x, y),
                Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
            });
        }
    }
    let (x0, y0, x1, y1) = bounds?;
    let (bw, bh) = (x1 - x0 + 1, y1 - y0 + 1);
    let scale = (RASTER_LONG_SIDE as f32 / bw.max(bh) as f32).min(1.0);
    let out_w = ((bw as f32 * scale).round() as usize).max(1);
    let out_h = ((bh as f32 * scale).round() as usize).max(1);
    let mut cells = Vec::with_capacity(out_w * out_h);
    for y in 0..out_h {
        let fy = y0 as f32 + (y as f32 + 0.5) * bh as f32 / out_h as f32;
        for x in 0..out_w {
            let fx = x0 as f32 + (x as f32 + 0.5) * bw as f32 / out_w as f32;
            let (xi, yi) = (fx.floor() as usize, fy.floor() as usize);
            let (xn, yn) = ((xi + 1).min(width - 1), (yi + 1).min(height - 1));
            let (tx, ty) = (fx - xi as f32, fy - yi as f32);
            let at = |xx: usize, yy: usize| data[yy * width + xx];
            let top = at(xi, yi) + (at(xn, yi) - at(xi, yi)) * tx;
            let bottom = at(xi, yn) + (at(xn, yn) - at(xi, yn)) * tx;
            let v = (top + (bottom - top) * ty).clamp(0.0, 1.0);
            cells.push((v * 255.0 + 0.5) as u8);
        }
    }
    Some(RasterMask {
        op,
        rect: [
            x0 as f32 / width as f32,
            y0 as f32 / height as f32,
            (x1 + 1) as f32 / width as f32,
            (y1 + 1) as f32 / height as f32,
        ],
        width: out_w,
        height: out_h,
        data: cells,
    })
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

    /// The select point is the viewer's raw-pixel handle; it must reach the
    /// engine layer unchanged and must not be treated as normalized.
    #[test]
    fn layer_select_point_reaches_the_layer_in_raw_pixels() {
        let mut m = layer_row("3", "1 局部图层", "12.9658");
        row(&[("ZLAYERSELECTPOINT", Some("778.75641;2700.73657"))], &mut m);
        let layers = mask_layers_from_rows(&[m]);
        assert_eq!(layers[0].select_point, Some([778.75641, 2700.73657]));
        let layer = layers[0].to_layer(9, 6048.0, vec![], vec![]);
        assert_eq!(layer.select_point, Some([778.75641, 2700.73657]));
    }

    #[test]
    fn layer_select_point_is_none_when_null() {
        let layers = mask_layers_from_rows(&[layer_row("0", "Skin Mask 1", "100")]);
        assert_eq!(layers[0].select_point, None);
        assert_eq!(layers[0].to_layer(1, 1000.0, vec![], vec![]).select_point, None);
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

    /// A decoded plane becomes a raster cropped to its non-zero bbox; the
    /// rect is normalized against the whole canvas.
    #[test]
    fn plane_maps_to_a_cropped_normalized_raster() {
        let (w, h) = (8usize, 4usize);
        let mut data = vec![0.0f32; w * h];
        data[w + 2] = 1.0;
        data[w + 3] = 1.0;
        data[2 * w + 2] = 1.0;
        data[2 * w + 3] = 1.0;
        data[2 * w + 5] = 0.5;
        let raster = raster_shape_from_plane(w, h, &data, MaskOp::Add).expect("non-empty plane");
        assert_eq!((raster.width, raster.height), (4, 2));
        assert_eq!(raster.rect, [2.0 / 8.0, 1.0 / 4.0, 6.0 / 8.0, 3.0 / 4.0]);
        assert!(raster.sample([2.5 / 8.0, 1.5 / 4.0]) > 0.9, "the first cell must stay full");
        assert!(raster.sample([5.5 / 8.0, 2.5 / 4.0]) > 0.1, "the half cell must survive");

        let empty = vec![0.0f32; w * h];
        assert!(raster_shape_from_plane(w, h, &empty, MaskOp::Add).is_none());
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
                ("ZLEVELSTARGETHIGHLIGHT", Some("0.900;1.000;1.000;1.000")),
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
        // `ApplyTargetLevelsCommon` reads only the `.x` master component, so the
        // per-channel slots stay identity even when they carry a value.
        assert_eq!((levels.rgb.output_white, levels.blue.output_white), (0.9, 1.0));
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
                cover: None,
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

    /// `ZSTYLEUUID/SOURCE/MODIFIED` and `ZAICOLORGRADE` are pass-through: they
    /// reach the layer model, and nothing in the render path reads them.
    #[test]
    fn style_uuid_and_ai_grade_reach_the_layer() {
        let plain = mask_layer_from_row(&layer_row("0", "Plain", "100")).unwrap();
        assert_eq!(plain.style_uuid, None);
        assert!(!plain.adjustments.ai_color_grade);

        let mut m = layer_row("1", "Look", "100");
        row(
            &[
                ("ZSTYLEUUID", Some("4FD0E17F-B1FC-4E78-BEAC-92315B3B15E3")),
                ("ZSTYLESOURCE", Some("Exposure")),
                ("ZSTYLEMODIFIED", Some("1")),
                ("ZAICOLORGRADE", Some("1")),
            ],
            &mut m,
        );
        let parsed = mask_layer_from_row(&m).unwrap();
        assert_eq!(parsed.style_uuid.as_deref(), Some("4FD0E17F-B1FC-4E78-BEAC-92315B3B15E3"));
        assert_eq!(parsed.style_source.as_deref(), Some("Exposure"));
        assert!(parsed.style_modified);
        let layer = parsed.to_layer(1, 1000.0, vec![], vec![]);
        let style = layer.style.expect("style carries through");
        assert_eq!(style.uuid, "4FD0E17F-B1FC-4E78-BEAC-92315B3B15E3");
        assert_eq!(style.source.as_deref(), Some("Exposure"));
        assert!(style.modified);
        assert!(layer.adjustments.ai_color_grade);
    }
}

/// r87: all 16 brush columns are readable and land on the layer.
#[cfg(test)]
mod brush_wire_tests {
    use super::*;

    const COLUMNS: [&str; 16] = [
        "ZBRUSHSTYLEUUID", "ZBRUSHSTYLENAME", "ZBRUSHSIZE", "ZBRUSHHARDNESS",
        "ZBRUSHOPACITY", "ZBRUSHFLOW", "ZBRUSHAIRBRUSH", "ZBRUSHUSEPENPRESSURE",
        "ZBRUSHAUTOMASK", "ZERASERSIZE", "ZERASERHARDNESS", "ZERASEROPACITY",
        "ZERASERFLOW", "ZERASERAIRBRUSH", "ZERASERUSEPENPRESSURE", "ZERASERAUTOMASK",
    ];

    fn catalog() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/c1-sample.cocatalogdb")
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn all_sixteen_columns_exist_and_parse() {
        let path = catalog();
        let mut db = crate::c1catalog::SqliteReader::open(&path).unwrap();
        let cols = db.columns("ZVARIANTLAYER").unwrap();
        for name in COLUMNS {
            assert!(cols.iter().any(|c| c == name), "{name} missing from ZVARIANTLAYER");
        }
        let cat = crate::C1Catalog::open_ro(&path).unwrap();
        let layers: Vec<&C1MaskLayer> = cat.variants.iter().flat_map(|v| &v.local_layers).collect();
        assert!(!layers.is_empty(), "fixture must carry local layers");
        // Real data: hair brush never used, pen pressure used at least once.
        assert!(layers.iter().all(|l| !l.brush.hair_brush && !l.eraser.air_brush));
        assert!(layers.iter().any(|l| l.brush.use_pen_pressure || l.eraser.use_pen_pressure));
        // The fixture has no brush preset.
        assert!(layers.iter().all(|l| l.brush_style_uuid.is_none() && l.brush_style_name.is_none()));
    }

    #[test]
    fn new_stroke_fields_reach_the_shape_and_default_false() {
        let mut layer = C1MaskLayer {
            index: 0,
            name: None,
            mask_uuid: None,
            select_point: None,
            enabled: true,
            opacity: 100.0,
            surface_smoothness: 0.0,
            brush: C1Brush { hair_brush: true, use_pen_pressure: true, ..Default::default() },
            eraser: C1Eraser { use_pen_pressure: true, ..Default::default() },
            brush_style_uuid: None,
            brush_style_name: None,
            style_uuid: None,
            style_source: None,
            style_modified: false,
            adjustments: LocalAdjustments::default(),
        };
        let out = layer.to_layer(1, 1000.0, vec![[0.5, 0.5, 1.0]], vec![[0.2, 0.2, 1.0]]);
        match (&out.mask.shapes[0], &out.mask.shapes[1]) {
            (MaskShape::Brush(add), MaskShape::Brush(sub)) => {
                assert!(add.hair_brush && add.use_pen_pressure);
                assert!(!sub.hair_brush && sub.use_pen_pressure);
            }
            other => panic!("unexpected shapes {other:?}"),
        }
        layer.brush.hair_brush = false;
        let out = layer.to_layer(2, 1000.0, vec![[0.5, 0.5, 1.0]], vec![]);
        match &out.mask.shapes[0] {
            MaskShape::Brush(add) => assert!(!add.hair_brush && add.use_pen_pressure),
            other => panic!("unexpected shape {other:?}"),
        }
    }

    #[test]
    fn strokes_without_the_new_fields_still_deserialize() {
        let json = r#"{"op":"Add","points":[[0.5,0.5,1.0]],"size":0.1,"hardness":50.0,
                       "opacity":100.0,"flow":100.0,"auto_mask":false}"#;
        let stroke: BrushStroke = serde_json::from_str(json).unwrap();
        assert!(!stroke.hair_brush && !stroke.use_pen_pressure);
    }
}
