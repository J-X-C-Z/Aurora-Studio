//! Read-only decode of Capture One's `ZRETOUCHINGLAYER` row (41 columns) and
//! its `ZFACE` face record.
//!
//! Evidence: `C1-RETOUCHINGLAYER.md` and `C1-FACE-LANDMARKS.md` in
//! `win_auto_rotate_r3-20261112-232128`. This module only shapes the stored
//! values; it is not attached to [`crate::AdjustmentSet`] and has no renderer.
//!
//! * `ZFACE` is one VARCHAR holding 19 floats in `McFaceInfo` field order:
//!   bbox `x,y,w,h`, Eye1, Eye2, Nose, MouthLeft, MouthRight, Head, then
//!   Confidence, Importance, FaceSizeRatio.
//! * Mask crop columns stay opaque strings: their coordinate space (source,
//!   rotated canvas or cropped canvas) is not proven.
//! * Columns outside [`RETOUCHING_LAYER_COLUMNS`] are preserved verbatim in
//!   [`RetouchingLayer::other`]; no semantics are guessed for them.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::c1_model::{clean, flag, parse_scalar, text};

/// The 41 columns observed in the real `ZRETOUCHINGLAYER` table.
pub static RETOUCHING_LAYER_COLUMNS: &[&str] = &[
    "Z_ENT",
    "Z_PK",
    "ZVARIANT",
    "ZFACE",
    "ZBLEMISHREMOVALMASKUUID",
    "ZBLEMISHREMOVALMASKCROP",
    "ZBLEMISHREMOVALAMOUNT",
    "ZDARKCIRCLESREDUCTIONMASKUUID",
    "ZDARKCIRCLESREDUCTIONMASKCROP",
    "ZDARKCIRCLESREDUCTIONAMOUNT",
    "ZFACESCULPTINGMASKUUID",
    "ZFACESCULPTINGMASKCROP",
    "ZFACESCULPTINGCONTOURING",
    "ZFACESCULPTINGHIGHLIGHTING",
    "ZSKINEVENINGMASKUUID",
    "ZSKINEVENINGMASKCROP",
    "ZSKINEVENINGAMOUNT",
    "ZSKINEVENINGTEXTURE",
    "ZTEETHWHITENINGMASKUUID",
    "ZTEETHWHITENINGMASKCROP",
    "ZTEETHWHITENINGBRIGHTNESS",
    "ZTEETHWHITENINGDESATURATION",
    "ZOPACITY",
    "ZINCLUDENECKINBLEMISHREMOVAL",
    "ZINCLUDENECKINSKINEVENING",
    "ZEYEBRIGHTLEFTIRISMASKUUID",
    "ZEYEBRIGHTLEFTIRISMASKCROP",
    "ZEYEBRIGHTLEFTIRISAMOUNT",
    "ZEYEBRIGHTLEFTSCLERAMASKUUID",
    "ZEYEBRIGHTLEFTSCLERAMASKCROP",
    "ZEYEBRIGHTLEFTSCLERAAMOUNT",
    "ZEYEBRIGHTRIGHTIRISMASKUUID",
    "ZEYEBRIGHTRIGHTIRISMASKCROP",
    "ZEYEBRIGHTRIGHTIRISAMOUNT",
    "ZEYEBRIGHTRIGHTSCLERAMASKUUID",
    "ZEYEBRIGHTRIGHTSCLERAMASKCROP",
    "ZEYEBRIGHTRIGHTSCLERAAMOUNT",
    "ZEYEBRIGHTBOTHIRISAMOUNT",
    "ZEYEBRIGHTBOTHSCLERAAMOUNT",
    "ZSTRAYHAIRREMOVALINNERAMOUNT",
    "ZSTRAYHAIRREMOVALOUTERAMOUNT",
];

/// The 19 floats of one `ZFACE` row, in `McFaceInfo` field order.
///
/// Eye left/right identity, the Head anchor's exact meaning and the coordinate
/// basis are not proven, so the fields keep C1's own names.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FaceInfo {
    /// `BoundingBox`: `x, y, width, height`.
    pub bbox: [f32; 4],
    /// `Eye1`: `x, y`; left/right identity not proven.
    pub eye1: [f32; 2],
    /// `Eye2`: `x, y`; left/right identity not proven.
    pub eye2: [f32; 2],
    pub nose: [f32; 2],
    pub mouth_left: [f32; 2],
    pub mouth_right: [f32; 2],
    /// `Head`: top of head or centre not proven.
    pub head: [f32; 2],
    pub confidence: f32,
    pub importance: f32,
    pub face_size_ratio: f32,
}

impl FaceInfo {
    pub const VALUE_COUNT: usize = 19;

    /// Parses a `ZFACE` string.
    ///
    /// All 19 required values must parse; a missing or malformed value returns
    /// `None` instead of shifting later fields into the wrong slot. Values past
    /// 19 are ignored because their semantics are unknown.
    pub fn parse(raw: &str) -> Option<Self> {
        let values: Vec<f32> = clean(raw).split(';').map(parse_scalar).collect::<Option<_>>()?;
        if values.len() < Self::VALUE_COUNT {
            return None;
        }
        Some(Self {
            bbox: [values[0], values[1], values[2], values[3]],
            eye1: [values[4], values[5]],
            eye2: [values[6], values[7]],
            nose: [values[8], values[9]],
            mouth_left: [values[10], values[11]],
            mouth_right: [values[12], values[13]],
            head: [values[14], values[15]],
            confidence: values[16],
            importance: values[17],
            face_size_ratio: values[18],
        })
    }
}

/// One `...MASKUUID` / `...MASKCROP` pair.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MaskSlot {
    pub mask_uuid: Option<String>,
    /// Opaque text; the crop's coordinate space is not proven.
    pub mask_crop: Option<String>,
}

/// One decoded `ZRETOUCHINGLAYER` row.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RetouchingLayer {
    pub entity: Option<i64>,
    pub pk: Option<i64>,
    pub variant: Option<i64>,
    /// `None` when `ZFACE` is NULL or malformed.
    pub face: Option<FaceInfo>,
    /// Raw `ZFACE` text, kept even when it does not parse.
    pub face_raw: Option<String>,

    pub blemish_removal_mask: MaskSlot,
    pub blemish_removal_amount: Option<f32>,
    pub dark_circles_mask: MaskSlot,
    pub dark_circles_amount: Option<f32>,
    pub face_sculpting_mask: MaskSlot,
    pub face_sculpting_contouring: Option<f32>,
    pub face_sculpting_highlighting: Option<f32>,
    pub skin_evening_mask: MaskSlot,
    pub skin_evening_amount: Option<f32>,
    pub skin_evening_texture: Option<f32>,
    pub teeth_whitening_mask: MaskSlot,
    pub teeth_whitening_brightness: Option<f32>,
    pub teeth_whitening_desaturation: Option<f32>,
    pub opacity: Option<f32>,
    pub include_neck_in_blemish_removal: Option<bool>,
    pub include_neck_in_skin_evening: Option<bool>,
    pub eye_bright_left_iris_mask: MaskSlot,
    pub eye_bright_left_iris_amount: Option<f32>,
    pub eye_bright_left_sclera_mask: MaskSlot,
    pub eye_bright_left_sclera_amount: Option<f32>,
    pub eye_bright_right_iris_mask: MaskSlot,
    pub eye_bright_right_iris_amount: Option<f32>,
    pub eye_bright_right_sclera_mask: MaskSlot,
    pub eye_bright_right_sclera_amount: Option<f32>,
    pub eye_bright_both_iris_amount: Option<f32>,
    pub eye_bright_both_sclera_amount: Option<f32>,
    pub stray_hair_removal_inner_amount: Option<f32>,
    pub stray_hair_removal_outer_amount: Option<f32>,

    /// Columns present in the row but not in [`RETOUCHING_LAYER_COLUMNS`],
    /// kept exactly as read.
    pub other: BTreeMap<String, Option<String>>,
}

impl RetouchingLayer {
    /// Builds the layer from one `ZRETOUCHINGLAYER` row, using the same
    /// borrowed-row shape as [`crate::c1_model::from_layer_row`].
    pub fn from_row(row: &HashMap<&str, Option<String>>) -> Self {
        let get = |key: &str| row.get(key).and_then(|v| v.as_deref()).map(clean);
        let number = |key: &str| get(key).and_then(parse_scalar);
        let switch = |key: &str| get(key).map(|v| flag(Some(v), false));
        let mask = |prefix: &str| MaskSlot {
            mask_uuid: text(get(&format!("{prefix}MASKUUID"))),
            mask_crop: text(get(&format!("{prefix}MASKCROP"))),
        };

        let face_raw = text(get("ZFACE"));
        let mut other = BTreeMap::new();
        for (key, value) in row {
            if !RETOUCHING_LAYER_COLUMNS.contains(key) {
                other.insert((*key).to_string(), value.clone());
            }
        }

        Self {
            entity: number("Z_ENT").map(|v| v as i64),
            pk: number("Z_PK").map(|v| v as i64),
            variant: number("ZVARIANT").map(|v| v as i64),
            face: face_raw.as_deref().and_then(FaceInfo::parse),
            face_raw,

            blemish_removal_mask: mask("ZBLEMISHREMOVAL"),
            blemish_removal_amount: number("ZBLEMISHREMOVALAMOUNT"),
            dark_circles_mask: mask("ZDARKCIRCLESREDUCTION"),
            dark_circles_amount: number("ZDARKCIRCLESREDUCTIONAMOUNT"),
            face_sculpting_mask: mask("ZFACESCULPTING"),
            face_sculpting_contouring: number("ZFACESCULPTINGCONTOURING"),
            face_sculpting_highlighting: number("ZFACESCULPTINGHIGHLIGHTING"),
            skin_evening_mask: mask("ZSKINEVENING"),
            skin_evening_amount: number("ZSKINEVENINGAMOUNT"),
            skin_evening_texture: number("ZSKINEVENINGTEXTURE"),
            teeth_whitening_mask: mask("ZTEETHWHITENING"),
            teeth_whitening_brightness: number("ZTEETHWHITENINGBRIGHTNESS"),
            teeth_whitening_desaturation: number("ZTEETHWHITENINGDESATURATION"),
            opacity: number("ZOPACITY"),
            include_neck_in_blemish_removal: switch("ZINCLUDENECKINBLEMISHREMOVAL"),
            include_neck_in_skin_evening: switch("ZINCLUDENECKINSKINEVENING"),
            eye_bright_left_iris_mask: mask("ZEYEBRIGHTLEFTIRIS"),
            eye_bright_left_iris_amount: number("ZEYEBRIGHTLEFTIRISAMOUNT"),
            eye_bright_left_sclera_mask: mask("ZEYEBRIGHTLEFTSCLERA"),
            eye_bright_left_sclera_amount: number("ZEYEBRIGHTLEFTSCLERAAMOUNT"),
            eye_bright_right_iris_mask: mask("ZEYEBRIGHTRIGHTIRIS"),
            eye_bright_right_iris_amount: number("ZEYEBRIGHTRIGHTIRISAMOUNT"),
            eye_bright_right_sclera_mask: mask("ZEYEBRIGHTRIGHTSCLERA"),
            eye_bright_right_sclera_amount: number("ZEYEBRIGHTRIGHTSCLERAAMOUNT"),
            eye_bright_both_iris_amount: number("ZEYEBRIGHTBOTHIRISAMOUNT"),
            eye_bright_both_sclera_amount: number("ZEYEBRIGHTBOTHSCLERAAMOUNT"),
            stray_hair_removal_inner_amount: number("ZSTRAYHAIRREMOVALINNERAMOUNT"),
            stray_hair_removal_outer_amount: number("ZSTRAYHAIRREMOVALOUTERAMOUNT"),

            other,
        }
    }
}

fn cell<'a>(row: &'a HashMap<String, Option<String>>, key: &str) -> Option<&'a str> {
    row.get(key).and_then(|v| v.as_deref()).map(clean)
}

/// Result of grouping raw `ZRETOUCHINGLAYER` rows.
///
/// `orphans` counts rows that name no variant or a variant absent from
/// `ZVARIANT`; `non_retouch_rows` counts rows whose `Z_ENT` is not 51 or absent.
/// Both kinds are dropped instead of being attached to some other image.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RetouchingGroup {
    pub by_variant: HashMap<i64, Vec<RetouchingLayer>>,
    pub orphans: usize,
    pub non_retouch_rows: usize,
}

/// Groups `ZRETOUCHINGLAYER` rows (column -> stored text) by `ZVARIANT`.
///
/// Only `Z_ENT = 51` rows are retouching layers. Rows are ordered by `Z_PK`
/// before grouping so a variant's list is stable across reads. Unknown columns
/// still reach [`RetouchingLayer::other`].
pub fn group_retouching_rows(
    rows: &[HashMap<String, Option<String>>],
    variants: &HashSet<i64>,
) -> RetouchingGroup {
    let mut group = RetouchingGroup::default();
    let mut parsed: Vec<(i64, i64, RetouchingLayer)> = Vec::new();
    for raw in rows {
        if cell(raw, "Z_ENT").and_then(parse_scalar) != Some(51.0) {
            group.non_retouch_rows += 1;
            continue;
        }
        let Some(variant) = cell(raw, "ZVARIANT").and_then(parse_scalar).map(|v| v as i64) else {
            group.orphans += 1;
            continue;
        };
        if !variants.contains(&variant) {
            group.orphans += 1;
            continue;
        }
        let pk = cell(raw, "Z_PK").and_then(parse_scalar).map(|v| v as i64).unwrap_or(0);
        let borrowed: HashMap<&str, Option<String>> =
            raw.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        parsed.push((pk, variant, RetouchingLayer::from_row(&borrowed)));
    }

    parsed.sort_by_key(|(pk, _, _)| *pk);
    for (_, variant, layer) in parsed {
        group.by_variant.entry(variant).or_default().push(layer);
    }
    group
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pairs: &[(&'static str, Option<&'static str>)]) -> HashMap<&'static str, Option<String>> {
        pairs.iter().map(|(k, v)| (*k, v.map(str::to_string))).collect()
    }

    fn owned_row(pairs: &[(&str, Option<&str>)]) -> HashMap<String, Option<String>> {
        pairs.iter().map(|(k, v)| ((*k).to_string(), v.map(str::to_string))).collect()
    }

    const SYNTHETIC_FACE: &str = "0.1;0.2;0.3;0.4;\
0.15;0.25;0.35;0.25;\
0.25;0.35;0.2;0.4;\
0.3;0.4;0.25;0.1;\
0.9;1;0.125";

    #[test]
    fn face_field_order_matches_mcfaceinfo() {
        let face = FaceInfo::parse(SYNTHETIC_FACE).unwrap();
        assert_eq!(face.bbox, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(face.eye1, [0.15, 0.25]);
        assert_eq!(face.eye2, [0.35, 0.25]);
        assert_eq!(face.nose, [0.25, 0.35]);
        assert_eq!(face.mouth_left, [0.2, 0.4]);
        assert_eq!(face.mouth_right, [0.3, 0.4]);
        assert_eq!(face.head, [0.25, 0.1]);
        assert_eq!(face.confidence, 0.9);
        assert_eq!(face.importance, 1.0);
        assert_eq!(face.face_size_ratio, 0.125);
    }

    #[test]
    fn face_missing_value_is_not_shifted() {
        let short = "0.1;0.2;0.3;0.4;\
0.15;0.25;0.35;0.25;\
0.25;0.35;0.2;0.4;\
0.3;0.4;0.25;0.1;\
0.9;1";
        assert_eq!(FaceInfo::parse(short), None);
        assert_eq!(FaceInfo::parse(""), None);
        assert_eq!(FaceInfo::parse("1;2;3;not-a-number;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19"), None);
    }

    #[test]
    fn column_table_is_the_41_observed_columns() {
        assert_eq!(RETOUCHING_LAYER_COLUMNS.len(), 41);
        let unique: BTreeMap<_, _> =
            RETOUCHING_LAYER_COLUMNS.iter().map(|name| (*name, ())).collect();
        assert_eq!(unique.len(), 41);
        assert!(RETOUCHING_LAYER_COLUMNS.contains(&"ZFACE"));
        assert!(RETOUCHING_LAYER_COLUMNS.contains(&"ZEYEBRIGHTRIGHTSCLERAAMOUNT"));
        assert!(RETOUCHING_LAYER_COLUMNS.contains(&"ZSTRAYHAIRREMOVALOUTERAMOUNT"));
    }

    #[test]
    fn every_known_field_maps_to_its_column() {
        let layer = RetouchingLayer::from_row(&row(&[
            ("Z_ENT", Some("51")),
            ("Z_PK", Some("7")),
            ("ZVARIANT", Some("9")),
            ("ZFACE", Some(SYNTHETIC_FACE)),
            ("ZBLEMISHREMOVALMASKUUID", Some("blemish-uuid")),
            ("ZBLEMISHREMOVALMASKCROP", Some("0;0;1;1")),
            ("ZBLEMISHREMOVALAMOUNT", Some("51.0989")),
            ("ZDARKCIRCLESREDUCTIONMASKUUID", Some("dark-uuid")),
            ("ZDARKCIRCLESREDUCTIONMASKCROP", Some("0;0;2;2")),
            ("ZDARKCIRCLESREDUCTIONAMOUNT", Some("12.5")),
            ("ZFACESCULPTINGMASKUUID", Some("sculpt-uuid")),
            ("ZFACESCULPTINGMASKCROP", Some("crop-a")),
            ("ZFACESCULPTINGCONTOURING", Some("30")),
            ("ZFACESCULPTINGHIGHLIGHTING", Some("40")),
            ("ZSKINEVENINGMASKUUID", Some("skin-uuid")),
            ("ZSKINEVENINGMASKCROP", Some("crop-b")),
            ("ZSKINEVENINGAMOUNT", Some("65.9341")),
            ("ZSKINEVENINGTEXTURE", Some("55")),
            ("ZTEETHWHITENINGMASKUUID", Some("teeth-uuid")),
            ("ZTEETHWHITENINGMASKCROP", Some("crop-c")),
            ("ZTEETHWHITENINGBRIGHTNESS", Some("20")),
            ("ZTEETHWHITENINGDESATURATION", Some("10")),
            ("ZOPACITY", Some("100")),
            ("ZINCLUDENECKINBLEMISHREMOVAL", Some("1")),
            ("ZINCLUDENECKINSKINEVENING", Some("0")),
            ("ZEYEBRIGHTLEFTIRISMASKUUID", Some("left-iris-uuid")),
            ("ZEYEBRIGHTLEFTIRISMASKCROP", Some("crop-d")),
            ("ZEYEBRIGHTLEFTIRISAMOUNT", Some("11")),
            ("ZEYEBRIGHTLEFTSCLERAMASKUUID", Some("left-sclera-uuid")),
            ("ZEYEBRIGHTLEFTSCLERAMASKCROP", Some("crop-e")),
            ("ZEYEBRIGHTLEFTSCLERAAMOUNT", Some("12")),
            ("ZEYEBRIGHTRIGHTIRISMASKUUID", Some("right-iris-uuid")),
            ("ZEYEBRIGHTRIGHTIRISMASKCROP", Some("crop-f")),
            ("ZEYEBRIGHTRIGHTIRISAMOUNT", Some("13")),
            ("ZEYEBRIGHTRIGHTSCLERAMASKUUID", Some("right-sclera-uuid")),
            ("ZEYEBRIGHTRIGHTSCLERAMASKCROP", Some("crop-g")),
            ("ZEYEBRIGHTRIGHTSCLERAAMOUNT", Some("14")),
            ("ZEYEBRIGHTBOTHIRISAMOUNT", Some("15")),
            ("ZEYEBRIGHTBOTHSCLERAAMOUNT", Some("16")),
            ("ZSTRAYHAIRREMOVALINNERAMOUNT", Some("17")),
            ("ZSTRAYHAIRREMOVALOUTERAMOUNT", Some("18")),
        ]));

        assert_eq!(layer.entity, Some(51));
        assert_eq!(layer.pk, Some(7));
        assert_eq!(layer.variant, Some(9));
        assert_eq!(layer.face.unwrap().mouth_left, [0.2, 0.4]);
        assert_eq!(layer.blemish_removal_mask.mask_uuid.as_deref(), Some("blemish-uuid"));
        assert_eq!(layer.blemish_removal_mask.mask_crop.as_deref(), Some("0;0;1;1"));
        assert_eq!(layer.blemish_removal_amount, Some(51.0989));
        assert_eq!(layer.dark_circles_mask.mask_uuid.as_deref(), Some("dark-uuid"));
        assert_eq!(layer.dark_circles_amount, Some(12.5));
        assert_eq!(layer.face_sculpting_mask.mask_crop.as_deref(), Some("crop-a"));
        assert_eq!(layer.face_sculpting_contouring, Some(30.0));
        assert_eq!(layer.face_sculpting_highlighting, Some(40.0));
        assert_eq!(layer.skin_evening_mask.mask_uuid.as_deref(), Some("skin-uuid"));
        assert_eq!(layer.skin_evening_amount, Some(65.9341));
        assert_eq!(layer.skin_evening_texture, Some(55.0));
        assert_eq!(layer.teeth_whitening_mask.mask_crop.as_deref(), Some("crop-c"));
        assert_eq!(layer.teeth_whitening_brightness, Some(20.0));
        assert_eq!(layer.teeth_whitening_desaturation, Some(10.0));
        assert_eq!(layer.opacity, Some(100.0));
        assert_eq!(layer.include_neck_in_blemish_removal, Some(true));
        assert_eq!(layer.include_neck_in_skin_evening, Some(false));
        assert_eq!(layer.eye_bright_left_iris_mask.mask_uuid.as_deref(), Some("left-iris-uuid"));
        assert_eq!(layer.eye_bright_left_iris_amount, Some(11.0));
        assert_eq!(layer.eye_bright_left_sclera_amount, Some(12.0));
        assert_eq!(layer.eye_bright_right_iris_amount, Some(13.0));
        assert_eq!(layer.eye_bright_right_sclera_amount, Some(14.0));
        assert_eq!(layer.eye_bright_both_iris_amount, Some(15.0));
        assert_eq!(layer.eye_bright_both_sclera_amount, Some(16.0));
        assert_eq!(layer.stray_hair_removal_inner_amount, Some(17.0));
        assert_eq!(layer.stray_hair_removal_outer_amount, Some(18.0));
        assert!(layer.other.is_empty());
    }

    #[test]
    fn missing_columns_stay_none_without_shifting() {
        let layer = RetouchingLayer::from_row(&row(&[
            ("ZSKINEVENINGTEXTURE", Some("55")),
            ("ZEYEBRIGHTRIGHTSCLERAAMOUNT", Some("14")),
            ("ZSTRAYHAIRREMOVALOUTERAMOUNT", None),
        ]));

        assert_eq!(layer.skin_evening_texture, Some(55.0));
        assert_eq!(layer.skin_evening_amount, None);
        assert_eq!(layer.eye_bright_right_sclera_amount, Some(14.0));
        assert_eq!(layer.eye_bright_right_iris_amount, None);
        assert_eq!(layer.stray_hair_removal_outer_amount, None);
        assert_eq!(layer.face, None);
        assert_eq!(layer.face_raw, None);
    }

    #[test]
    fn malformed_face_keeps_the_raw_text() {
        let layer = RetouchingLayer::from_row(&row(&[("ZFACE", Some("1;2;3"))]));
        assert_eq!(layer.face, None);
        assert_eq!(layer.face_raw.as_deref(), Some("1;2;3"));
    }

    #[test]
    fn unknown_columns_are_preserved_verbatim() {
        let layer = RetouchingLayer::from_row(&row(&[
            ("ZFUTURECOLUMN", Some("'quoted'")),
            ("ZOTHEREMPTY", None),
            ("ZFACE", None),
        ]));

        assert_eq!(layer.other.get("ZFUTURECOLUMN"), Some(&Some("'quoted'".to_string())));
        assert_eq!(layer.other.get("ZOTHEREMPTY"), Some(&None));
        assert!(!layer.other.contains_key("ZFACE"));
    }

    #[test]
    fn groups_multiple_variants_in_pk_order() {
        let rows = vec![
            owned_row(&[
                ("Z_ENT", Some("51")),
                ("Z_PK", Some("2")),
                ("ZVARIANT", Some("10")),
                ("ZSKINEVENINGTEXTURE", Some("55")),
            ]),
            owned_row(&[
                ("Z_ENT", Some("51")),
                ("Z_PK", Some("1")),
                ("ZVARIANT", Some("10")),
                ("ZBLEMISHREMOVALAMOUNT", Some("51.0989")),
            ]),
            owned_row(&[
                ("Z_ENT", Some("51")),
                ("Z_PK", Some("3")),
                ("ZVARIANT", Some("10")),
                ("ZFACE", Some(SYNTHETIC_FACE)),
            ]),
            owned_row(&[
                ("Z_ENT", Some("51")),
                ("Z_PK", Some("5")),
                ("ZVARIANT", Some("20")),
                ("ZEYEBRIGHTLEFTIRISAMOUNT", Some("11")),
            ]),
        ];
        let variants = HashSet::from([10, 20]);

        let group = group_retouching_rows(&rows, &variants);

        assert_eq!(group.orphans, 0);
        assert_eq!(group.non_retouch_rows, 0);
        let ten = &group.by_variant[&10];
        assert_eq!(ten.len(), 3);
        assert_eq!(ten.iter().map(|l| l.pk.unwrap()).collect::<Vec<_>>(), [1, 2, 3]);
        assert_eq!(ten[0].blemish_removal_amount, Some(51.0989));
        assert_eq!(ten[1].skin_evening_texture, Some(55.0));
        assert_eq!(ten[1].skin_evening_amount, None);
        assert_eq!(ten[2].face.unwrap().face_size_ratio, 0.125);
        assert_eq!(group.by_variant[&20].len(), 1);
    }

    #[test]
    fn orphans_and_non_retouch_rows_are_dropped_and_counted() {
        let rows = vec![
            owned_row(&[
                ("Z_ENT", Some("51")),
                ("Z_PK", Some("1")),
                ("ZVARIANT", Some("99")),
            ]),
            owned_row(&[("Z_ENT", Some("51")), ("Z_PK", Some("2"))]),
            owned_row(&[
                ("Z_ENT", Some("17")),
                ("Z_PK", Some("3")),
                ("ZVARIANT", Some("10")),
            ]),
            owned_row(&[
                ("Z_ENT", Some("51")),
                ("Z_PK", Some("4")),
                ("ZVARIANT", Some("10")),
            ]),
        ];
        let variants = HashSet::from([10]);

        let group = group_retouching_rows(&rows, &variants);

        assert_eq!(group.orphans, 2);
        assert_eq!(group.non_retouch_rows, 1);
        assert_eq!(group.by_variant.len(), 1);
        assert_eq!(group.by_variant[&10].len(), 1);
        assert_eq!(group.by_variant[&10][0].pk, Some(4));
    }
}
