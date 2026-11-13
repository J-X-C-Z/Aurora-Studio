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
//! `ZCLARITYSTRUCTURE`, `ZCONTRAST`, `ZBRIGHTNESS`, `ZSATURATION`
//! (variant 1, PK 7 stores non-neutral values for all of them). They map onto
//! [`LocalAdjustments`], which `local()` already applies.

use std::collections::HashMap;

use crate::adjust::{Clarity, ClarityMethod, Dehaze, Exposure, HighDynamicRange, WhiteBalance};
use crate::comask::{ComaskDocument, ComaskMask};
use crate::c1_model::{from_layer_row, gains_to_kelvin_tint, C1Brush, C1Eraser, C1EntityKind};
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

/// Reads one `ZVARIANTLAYER` row; `None` when it is not a local mask layer.
pub fn mask_layer_from_row(row: &HashMap<&str, Option<String>>) -> Option<C1MaskLayer> {
    let s = from_layer_row(row);
    (s.entity == C1EntityKind::LocalMask).then(|| {
        let (kelvin, tint) = gains_to_kelvin_tint(s.white_balance_gains);
        let adjustments = LocalAdjustments {
            white_balance: (s.white_balance_gains != [1.0; 3])
                .then_some(WhiteBalance { kelvin, tint, ..Default::default() }),
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
}
