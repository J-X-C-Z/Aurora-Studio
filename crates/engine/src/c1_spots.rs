//! Capture One's spot-removal list: the `ZSPOTS` / `ZRELATIVESPOTS` strings and
//! their mapping onto [`crate::Spot`], which `.lumen.json` stores as `spots`.
//!
//! Grammar (`P1.C1.Core.ImagePersistency.StrSpots`):
//!
//! ```text
//! selectedIndex;type,x,y,radius;type,x,y,radius;...
//! ```
//!
//! * `selectedIndex` is the UI selection; C1 persists `-1`.
//! * `type` is `IC_SpotRemovalType`: 0 None, 1 Black, 2 Spot, 3 Dust, 4 Clone.
//! * `x,y,radius` are *raw* image pixels (the `ZIMAGE` frame), six decimals,
//!   invariant culture. The 5..100 radius range is in canvas units and is
//!   enforced by the editor, never by the stored string.
//! * `ZRELATIVESPOTS` uses the same grammar with the three measured values
//!   `x/W`, `y/H` and `pi*r^2/(W*H)` (area fraction) so a paste can rescale to
//!   another image. [`parse_relative`] also accepts entries that carry only
//!   those three components, giving them the default spot type.
//! * At most [`MAX_SPOTS`] spots; the editor silently refuses the 201st.
//! * A spot with fewer than three fields, a `ZSPOTS` entry with only three, or a
//!   non-integer type becomes [`C1Spot::EMPTY`]; a bad float reads as 0,
//!   matching `SafeParseFloat`.
//!
//! `.lumen.json` spot coordinates are normalized in the oriented, uncropped
//! image and the radius is a fraction of the long side. C1 stores raw pixels, so
//! [`to_lumen`] divides by the `ZIMAGE` size and applies the quarter turn.
//! `ZSPOTS` has no source point: a type-4 Clone keeps it in the `.comask` cover
//! mask, which `ZSPOTS` does not carry, so a Clone cannot be imported and is
//! dropped rather than turned into a heal spot at the wrong place.
//! When `ZRELATIVESPOTS` and `ZSPOTS` are both present, `ZSPOTS` wins (C1's
//! behaviour); the relative copy is only the fallback for a variant without an
//! absolute list.

use crate::comask::ComaskCover;
use crate::{CoverKind, CoverPatch, Geometry, Spot, SpotKind};

/// `IC_MAX_SPOT_POINTS`: the editor stops adding spots at this count.
pub const MAX_SPOTS: usize = 200;
/// `SpotRemovalDefinitions.MIN_SPOT_RADIUS`, canvas pixels.
pub const MIN_RADIUS_CANVAS: f32 = 5.0;
/// `SpotRemovalDefinitions.MAX_SPOT_RADIUS`, canvas pixels.
pub const MAX_RADIUS_CANVAS: f32 = 100.0;
/// `ApplicationSettings.SpotRemovalDefaultRadius` (`[DefaultSettingValue("10")]`).
pub const DEFAULT_RADIUS_CANVAS: f32 = 10.0;
/// Type a click creates with the Spot cursor; the Dust cursor uses 3.
pub const DEFAULT_KIND: i32 = KIND_SPOT;

pub const KIND_NONE: i32 = 0;
pub const KIND_BLACK: i32 = 1;
pub const KIND_SPOT: i32 = 2;
pub const KIND_DUST: i32 = 3;
pub const KIND_CLONE: i32 = 4;

/// One C1 spot in raw image pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C1Spot {
    pub kind: i32,
    pub x: f32,
    pub y: f32,
    pub radius: f32,
}

impl C1Spot {
    /// `Spot.Empty`: what C1 appends when one entry cannot be parsed.
    pub const EMPTY: Self = Self { kind: KIND_NONE, x: 0.0, y: 0.0, radius: 0.0 };
}

/// The decoded `selectedIndex` and spot list of one `ZSPOTS` value.
#[derive(Clone, Debug, PartialEq)]
pub struct SpotList {
    pub selected: i32,
    pub spots: Vec<C1Spot>,
}

impl Default for SpotList {
    fn default() -> Self {
        Self { selected: -1, spots: Vec::new() }
    }
}

impl SpotList {
    pub fn is_empty(&self) -> bool {
        self.spots.is_empty()
    }
}

/// Decodes `ZSPOTS`. `None` means C1's `null` (empty string); an empty list is a
/// string that names only the selection.
pub fn parse(raw: &str) -> Option<SpotList> {
    parse_with(raw, false)
}

/// Decodes `ZRELATIVESPOTS`, whose spot entries may omit the leading `type`.
pub fn parse_relative(raw: &str) -> Option<SpotList> {
    parse_with(raw, true)
}

fn parse_with(raw: &str, three_components: bool) -> Option<SpotList> {
    let raw = crate::c1_model::clean(raw);
    if raw.is_empty() {
        return None;
    }
    let mut parts = raw.split(';');
    let head = parts.next().unwrap_or_default();
    let mut list = SpotList::default();
    list.spots = parts.map(|entry| parse_spot(entry, three_components)).collect();
    // C1 parses the selection only when at least one spot entry follows it.
    if !list.spots.is_empty() {
        list.selected = clean_int(head).unwrap_or(-1);
    }
    Some(list)
}

/// Encodes the C1 grammar. `selected` is written as-is; C1 itself writes `-1`.
pub fn serialize(list: &SpotList) -> String {
    let mut out = list.selected.to_string();
    for spot in &list.spots {
        out.push(';');
        out.push_str(&spot.kind.to_string());
        out.push_str(&format!(",{:.6},{:.6},{:.6}", spot.x, spot.y, spot.radius));
    }
    out
}

/// `SaveSpotsInRelativeRawCoords`: `x/W`, `y/H`, `pi*r^2/(W*H)`.
pub fn to_relative(list: &SpotList, image: (u32, u32)) -> SpotList {
    let (w, h) = (image.0 as f32, image.1 as f32);
    if w <= 0.0 || h <= 0.0 {
        return SpotList::default();
    }
    SpotList {
        selected: list.selected,
        spots: list
            .spots
            .iter()
            .map(|s| C1Spot {
                kind: s.kind,
                x: s.x / w,
                y: s.y / h,
                radius: std::f32::consts::PI * s.radius * s.radius / (w * h),
            })
            .collect(),
    }
}

/// The inverse of [`to_relative`], as `TransferSpots` applies it on paste.
pub fn from_relative(list: &SpotList, image: (u32, u32)) -> SpotList {
    let (w, h) = (image.0 as f32, image.1 as f32);
    if w <= 0.0 || h <= 0.0 {
        return SpotList::default();
    }
    SpotList {
        selected: list.selected,
        spots: list
            .spots
            .iter()
            .map(|s| C1Spot {
                kind: s.kind,
                x: s.x * w,
                y: s.y * h,
                radius: (s.radius.max(0.0) * w * h / std::f32::consts::PI).sqrt(),
            })
            .collect(),
    }
}

/// `CoordinatesAreRelative`: every `x`, `y` and `radius` is below 1.01.
pub fn coordinates_are_relative(list: &SpotList) -> bool {
    !list.spots.is_empty()
        && list.spots.iter().all(|s| s.x < 1.01 && s.y < 1.01 && s.radius < 1.01)
}

/// The list a variant renders: `ZSPOTS` when it has entries, otherwise the
/// `ZRELATIVESPOTS` copy. A variant that carries both resolves the same way in
/// C1, so the relative table never overrides the absolute one.
pub fn effective(absolute: &SpotList, relative: &SpotList, image: (u32, u32)) -> SpotList {
    if absolute.is_empty() {
        from_relative(relative, image)
    } else {
        absolute.clone()
    }
}

/// What [`to_lumen`] kept and what it had to drop.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LumenMapping {
    pub spots: Vec<Spot>,
    /// Type 4: the source point lives in the `.comask` cover, not `ZSPOTS`.
    pub clone_without_source: usize,
    /// Type 0 placeholders.
    pub none: usize,
}

/// Maps C1 raw-pixel spots onto the `.lumen` spot array.
///
/// `geometry` supplies the quarter turn the catalog reader settled on; the
/// source size is the `ZIMAGE` size the raw coordinates refer to. Types 1 and 2
/// are both heals, 3 is dust, 4 is dropped, unknown codes are treated as a heal.
pub fn to_lumen(list: &SpotList, image: (u32, u32), geometry: &Geometry) -> LumenMapping {
    let (w, h) = (image.0 as f32, image.1 as f32);
    let mut out = LumenMapping::default();
    if w <= 0.0 || h <= 0.0 {
        return out;
    }
    let long_side = w.max(h);
    for (index, spot) in list.spots.iter().enumerate() {
        match spot.kind {
            KIND_NONE => {
                out.none += 1;
                continue;
            }
            KIND_CLONE => {
                // The clone source is persisted in the `.comask` cover mask, not
                // in ZSPOTS, so there is no point to rebuild this spot from.
                out.clone_without_source += 1;
                continue;
            }
            _ => {}
        }
        let p = orient([spot.x / w, spot.y / h], geometry);
        out.spots.push(Spot {
            id: index as u64 + 1,
            kind: if spot.kind == KIND_DUST { SpotKind::Dust } else { SpotKind::Spot },
            x: p[0],
            y: p[1],
            radius: spot.radius / long_side,
        });
    }
    out
}

/// Fills [`crate::AdjustmentSet::spots`] from `ZSPOTS` / `ZRELATIVESPOTS`.
///
/// `image` is the `ZIMAGE` size the absolute list is expressed in; without it
/// the coordinates cannot be normalised and the spots are left untouched.
/// `ZSPOTS` wins whenever it names at least one spot, matching C1's paste
/// rule (see [`effective`]).
pub fn apply_to_adjustments(
    adjustments: &mut crate::AdjustmentSet,
    settings: &crate::c1_model::C1VariantSettings,
    image: Option<(u32, u32)>,
) {
    let Some(image) = image else {
        return;
    };
    if image.0 == 0 || image.1 == 0 {
        return;
    }
    let list = effective(&settings.spots, &settings.relative_spots, image);
    adjustments.spots = to_lumen(&list, image, &adjustments.geometry).spots;
}

/// The inverse of [`to_lumen`], for writing `.lumen` edits back to `ZSPOTS`.
///
/// A `.lumen` Clone keeps its source offset, which `ZSPOTS` cannot store, so it
/// becomes type 4 and loses that offset.
pub fn from_lumen(spots: &[Spot], image: (u32, u32), geometry: &Geometry) -> SpotList {
    let (w, h) = (image.0 as f32, image.1 as f32);
    let long_side = w.max(h);
    if w <= 0.0 || h <= 0.0 {
        return SpotList::default();
    }
    let spots = spots
        .iter()
        .filter(|s| s.x.is_finite() && s.y.is_finite() && s.radius.is_finite())
        .map(|s| {
            let p = unorient([s.x, s.y], geometry);
            C1Spot {
                kind: match s.kind {
                    SpotKind::Dust => KIND_DUST,
                    SpotKind::Spot => KIND_SPOT,
                    SpotKind::Clone { .. } => KIND_CLONE,
                },
                x: p[0] * w,
                y: p[1] * h,
                radius: s.radius * long_side,
            }
        })
        .collect();
    SpotList { selected: -1, spots }
}

/// Maps a repair layer's `.comask` `Cover` arrows onto oriented patches.
///
/// The arrow stores only the two points; C1 keeps the stroke shape in the
/// layer's mask plane, so `mask_bbox` (raw pixels) gives the patch its size.
/// Without a painted mask there is nothing to repair.
///
/// shortcut: every arrow on one layer uses the whole mask bbox, so a layer with
/// several arrows patches the same rectangle several times. Upgrading needs a
/// sample that splits strokes within one layer.
pub fn cover_patches(
    cover: &ComaskCover,
    image: (u32, u32),
    mask_bbox: Option<(usize, usize, usize, usize)>,
    geometry: &Geometry,
) -> Vec<CoverPatch> {
    let (w, h) = (image.0 as f32, image.1 as f32);
    let Some((x0, y0, x1, y1)) = mask_bbox else {
        return Vec::new();
    };
    if w <= 0.0 || h <= 0.0 {
        return Vec::new();
    }
    let mut half = [((x1 - x0 + 1) as f32) / (2.0 * w), ((y1 - y0 + 1) as f32) / (2.0 * h)];
    if geometry.orientation % 2 == 1 {
        half.swap(0, 1);
    }
    cover
        .arrows
        .iter()
        .filter_map(|arrow| {
            let kind = match arrow.arrow_type {
                1 => CoverKind::Clone,
                2 => CoverKind::Heal,
                _ => return None,
            };
            if !arrow.src.iter().all(|v| v.is_finite()) || !arrow.dst.iter().all(|v| v.is_finite()) {
                return None;
            }
            Some(CoverPatch {
                kind,
                src: orient([arrow.src[0] / w, arrow.src[1] / h], geometry),
                dst: orient([arrow.dst[0] / w, arrow.dst[1] / h], geometry),
                half,
            })
        })
        .collect()
}

/// Source frame to oriented frame; mirrors `mock::geometry::map_frame`.
fn orient(p: [f32; 2], geometry: &Geometry) -> [f32; 2] {
    let mut q = p;
    if geometry.flip_horizontal {
        q[0] = 1.0 - q[0];
    }
    if geometry.flip_vertical {
        q[1] = 1.0 - q[1];
    }
    turn(q, geometry.orientation)
}

fn unorient(p: [f32; 2], geometry: &Geometry) -> [f32; 2] {
    let mut q = turn(p, 4 - geometry.orientation % 4);
    if geometry.flip_horizontal {
        q[0] = 1.0 - q[0];
    }
    if geometry.flip_vertical {
        q[1] = 1.0 - q[1];
    }
    q
}

fn turn(p: [f32; 2], orientation: u8) -> [f32; 2] {
    match orientation % 4 {
        1 => [p[1], 1.0 - p[0]],
        2 => [1.0 - p[0], 1.0 - p[1]],
        3 => [1.0 - p[1], p[0]],
        _ => p,
    }
}

fn parse_spot(raw: &str, three_components: bool) -> C1Spot {
    let fields: Vec<&str> = raw.split(',').collect();
    let (kind, values) = match fields.len() {
        3 if three_components => (DEFAULT_KIND, &fields[..]),
        n if n >= 4 => match clean_int(fields[0]) {
            Some(kind) => (kind, &fields[1..]),
            None => return C1Spot::EMPTY,
        },
        _ => return C1Spot::EMPTY,
    };
    C1Spot {
        kind,
        x: parse_or_zero(values[0]),
        y: parse_or_zero(values[1]),
        radius: parse_or_zero(values[2]),
    }
}

fn clean_int(raw: &str) -> Option<i32> {
    crate::c1_model::clean(raw).parse().ok()
}

fn parse_or_zero(raw: &str) -> f32 {
    crate::c1_model::clean(raw).parse::<f32>().ok().filter(|v| !v.is_nan()).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "-1;2,1000.000000,750.000000,60.000000;3,2000.000000,1500.000000,5.000000";

    #[test]
    fn catalog_mapping_fills_the_adjustment_spots() {
        let mut settings = crate::c1_model::C1VariantSettings::defaults();
        settings.spots = parse("-1;2,1000.000000,500.000000,60.000000").unwrap();
        let mut adjustments = crate::AdjustmentSet::default();
        apply_to_adjustments(&mut adjustments, &settings, Some((6000, 4000)));

        assert_eq!(adjustments.spots.len(), 1);
        assert!((adjustments.spots[0].x - 1000.0 / 6000.0).abs() < 1e-6);
        assert!((adjustments.spots[0].y - 500.0 / 4000.0).abs() < 1e-6);
        assert!((adjustments.spots[0].radius - 60.0 / 6000.0).abs() < 1e-6);
        assert_eq!(adjustments.spots[0].kind, crate::SpotKind::Spot);
    }

    #[test]
    fn catalog_mapping_falls_back_to_relative_spots() {
        let mut settings = crate::c1_model::C1VariantSettings::defaults();
        settings.relative_spots =
            parse_relative("-1;0.500000,0.250000,0.010000").unwrap();
        let mut adjustments = crate::AdjustmentSet::default();
        apply_to_adjustments(&mut adjustments, &settings, Some((6000, 4000)));

        assert_eq!(adjustments.spots.len(), 1);
        assert!((adjustments.spots[0].x - 0.5).abs() < 1e-6);
        assert!((adjustments.spots[0].y - 0.25).abs() < 1e-6);
        let expected = (0.01 * 6000.0 * 4000.0 / std::f32::consts::PI).sqrt() / 6000.0;
        assert!((adjustments.spots[0].radius - expected).abs() < 1e-6);
    }

    #[test]
    fn catalog_mapping_needs_the_source_size() {
        let mut settings = crate::c1_model::C1VariantSettings::defaults();
        settings.spots = parse(SAMPLE).unwrap();
        let mut adjustments = crate::AdjustmentSet::default();
        apply_to_adjustments(&mut adjustments, &settings, None);
        assert!(adjustments.spots.is_empty());
    }

    #[test]
    fn parses_the_documented_grammar() {
        let list = parse(SAMPLE).unwrap();
        assert_eq!(list.selected, -1);
        assert_eq!(list.spots.len(), 2);
        assert_eq!(
            list.spots[0],
            C1Spot { kind: KIND_SPOT, x: 1000.0, y: 750.0, radius: 60.0 }
        );
        assert_eq!(list.spots[1].kind, KIND_DUST);
    }

    #[test]
    fn round_trips_six_decimals() {
        assert_eq!(serialize(&parse(SAMPLE).unwrap()), SAMPLE);
        let list = SpotList {
            selected: -1,
            spots: vec![C1Spot { kind: 2, x: 1.0 / 3.0, y: 0.0, radius: 0.1 }],
        };
        assert_eq!(serialize(&list), "-1;2,0.333333,0.000000,0.100000");
    }

    #[test]
    fn empty_string_is_null_and_a_lone_selection_is_an_empty_list() {
        assert_eq!(parse(""), None);
        assert_eq!(parse("   "), None);
        let list = parse("-1").unwrap();
        assert!(list.is_empty());
        assert_eq!(list.selected, -1);
    }

    #[test]
    fn a_bad_entry_becomes_empty_without_shifting_the_next_spot() {
        let list = parse("-1;2,1,2;3,4,5,6").unwrap();
        assert_eq!(list.spots[0], C1Spot::EMPTY);
        assert_eq!(list.spots[1], C1Spot { kind: 3, x: 4.0, y: 5.0, radius: 6.0 });
        // SafeParseFloat reads a bad number as 0 and keeps the spot.
        assert_eq!(
            parse("-1;2,nope,5,6").unwrap().spots[0],
            C1Spot { kind: 2, x: 0.0, y: 5.0, radius: 6.0 }
        );
    }

    #[test]
    fn relative_three_component_entries_default_to_the_spot_type() {
        // `ZSPOTS` requires the type; C1's DeserializeSpot returns Empty without it.
        assert_eq!(parse("-1;0.5,0.25,0.01").unwrap().spots[0], C1Spot::EMPTY);
        let list = parse_relative("-1;0.500000,0.250000,0.010000").unwrap();
        assert_eq!(
            list.spots[0],
            C1Spot { kind: DEFAULT_KIND, x: 0.5, y: 0.25, radius: 0.01 }
        );
        // Serialization is always C1's four-field form.
        assert_eq!(serialize(&list), "-1;2,0.500000,0.250000,0.010000");
    }

    #[test]
    fn the_editor_limit_is_not_a_parser_limit() {
        let raw: String = std::iter::once("-1".to_string())
            .chain((0..MAX_SPOTS + 1).map(|i| format!(";2,{i},1,5")))
            .collect();
        assert_eq!(parse(&raw).unwrap().spots.len(), MAX_SPOTS + 1);
        assert_eq!(MAX_SPOTS, 200);
        assert_eq!((MIN_RADIUS_CANVAS, MAX_RADIUS_CANVAS, DEFAULT_RADIUS_CANVAS), (5.0, 100.0, 10.0));
    }

    #[test]
    fn relative_coordinates_use_the_area_fraction_for_the_radius() {
        let absolute = parse("-1;2,3000,2000,100").unwrap();
        let relative = to_relative(&absolute, (6000, 4000));
        assert_eq!(relative.spots[0].x, 0.5);
        assert_eq!(relative.spots[0].y, 0.5);
        assert!((relative.spots[0].radius - std::f32::consts::PI * 100.0 * 100.0 / 24_000_000.0).abs() < 1e-9);
        assert!(coordinates_are_relative(&relative));
        let back = from_relative(&relative, (6000, 4000));
        assert!((back.spots[0].radius - 100.0).abs() < 1e-3);
        assert!(!coordinates_are_relative(&absolute));
    }

    #[test]
    fn lumen_mapping_normalizes_radius_and_applies_the_quarter_turn() {
        let list = parse("-1;2,1000,750,100").unwrap();
        let mapped = to_lumen(&list, (4000, 3000), &Geometry::default());
        assert_eq!(mapped.spots.len(), 1);
        assert_eq!((mapped.spots[0].x, mapped.spots[0].y), (0.25, 0.25));
        assert_eq!(mapped.spots[0].radius, 100.0 / 4000.0);
        assert_eq!(mapped.spots[0].kind, SpotKind::Spot);

        let turned = to_lumen(&list, (4000, 3000), &Geometry { orientation: 1, ..Default::default() });
        assert_eq!((turned.spots[0].x, turned.spots[0].y), (0.25, 0.75));
    }

    #[test]
    fn lumen_round_trip_keeps_a_spot_where_it_was() {
        let list = parse("-1;3,1000,750,100;2,3000,2000,60").unwrap();
        let geometry = Geometry { orientation: 3, ..Default::default() };
        let lumen = to_lumen(&list, (4000, 3000), &geometry);
        let back = from_lumen(&lumen.spots, (4000, 3000), &geometry);
        assert_eq!(back.spots.len(), 2);
        assert_eq!(back.spots[0].kind, KIND_DUST);
        for (a, b) in list.spots.iter().zip(&back.spots) {
            assert!((a.x - b.x).abs() < 1e-3 && (a.y - b.y).abs() < 1e-3);
            assert!((a.radius - b.radius).abs() < 1e-3);
        }
    }

    #[test]
    fn clone_and_none_are_counted_not_guessed() {
        let list = parse("-1;0,0,0,0;4,100,100,50;2,200,200,50").unwrap();
        let mapped = to_lumen(&list, (1000, 1000), &Geometry::default());
        assert_eq!((mapped.none, mapped.clone_without_source), (1, 1));
        assert_eq!(mapped.spots.len(), 1);
    }

    #[test]
    fn zspots_wins_when_the_relative_copy_is_also_present() {
        let absolute = parse("-1;2,1000,750,60").unwrap();
        let relative = parse_relative("-1;2,0.100000,0.100000,0.010000").unwrap();
        assert_eq!(effective(&absolute, &relative, (4000, 3000)), absolute);
        // The relative copy is only the fallback when the absolute list is absent.
        let fallback = effective(&SpotList::default(), &relative, (4000, 3000));
        assert_eq!(fallback.spots.len(), 1);
        assert_eq!((fallback.spots[0].x, fallback.spots[0].y), (400.0, 300.0));
    }

    #[test]
    fn zero_sized_images_map_to_nothing_instead_of_dividing_by_zero() {
        let list = parse("-1;2,10,10,5").unwrap();
        assert!(to_lumen(&list, (0, 0), &Geometry::default()).spots.is_empty());
        assert!(from_relative(&list, (0, 0)).is_empty());
    }

    // ── ZSPOTS -> renderer ───────────────────────────────────────────────────

    use crate::mock::spots::apply_spots;

    fn flat(w: u32, h: u32, v: u8) -> Vec<u8> {
        let mut buf = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..w * h {
            buf.extend_from_slice(&[v, v, v, 255]);
        }
        buf
    }

    fn pixel(buf: &[u8], w: u32, x: u32, y: u32) -> [u8; 3] {
        let p = ((y * w + x) * 4) as usize;
        [buf[p], buf[p + 1], buf[p + 2]]
    }

    fn dark_disc(buf: &mut [u8], w: u32, h: u32, cx: f32, cy: f32, radius: f32) {
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                if d <= radius {
                    let p = ((y * w + x) * 4) as usize;
                    buf[p..p + 3].fill(0);
                }
            }
        }
    }

    fn render(list: &SpotList, w: u32, h: u32, buf: &mut [u8]) {
        let mut adj = crate::AdjustmentSet::default();
        adj.spots = to_lumen(list, (w, h), &Geometry::default()).spots;
        apply_spots(buf, w, h, &adj, (w, h));
    }

    #[test]
    fn a_selected_index_is_not_mistaken_for_a_kind() {
        // `selectedIndex` is the head, not the first spot's type.
        let list = parse("1;3,100,200,30;2,300,400,5").unwrap();
        assert_eq!(list.selected, 1);
        assert_eq!((list.spots[0].kind, list.spots[1].kind), (KIND_DUST, KIND_SPOT));
        assert_eq!((list.spots[0].x, list.spots[1].x), (100.0, 300.0));
    }

    #[test]
    fn no_spots_reach_the_renderer_as_an_identity() {
        let mut buf = flat(64, 64, 128);
        let before = buf.clone();
        render(&parse("-1").unwrap(), 64, 64, &mut buf);
        assert_eq!(buf, before);
    }

    #[test]
    fn a_radius_60_spot_heals_its_centre_and_leaves_the_outside_alone() {
        let (w, h) = (1000u32, 1000u32);
        let mut buf = flat(w, h, 200);
        dark_disc(&mut buf, w, h, 500.0, 500.0, 40.0);
        let far_before = pixel(&buf, w, 500, 570);

        render(&parse("-1;2,500,500,60").unwrap(), w, h, &mut buf);

        let centre = pixel(&buf, w, 500, 500);
        assert!(centre[0] > 170, "radius-60 centre is {centre:?}, the spot was not healed");
        assert_eq!(pixel(&buf, w, 500, 570), far_before, "pixels outside the patch moved");
    }

    #[test]
    fn the_radius_5_minimum_spot_still_heals() {
        let (w, h) = (200u32, 200u32);
        let mut buf = flat(w, h, 180);
        let p = ((100 * w + 100) * 4) as usize;
        buf[p..p + 3].fill(0);

        render(&parse("-1;2,100,100,5").unwrap(), w, h, &mut buf);

        let centre = pixel(&buf, w, 100, 100);
        assert!(centre[0] > 150, "radius-5 centre is {centre:?}, the spot was not healed");
    }

    #[test]
    fn a_type_3_dust_spot_is_inpainted_not_median_filtered() {
        let (w, h) = (1000u32, 1000u32);
        let mut buf = flat(w, h, 200);
        dark_disc(&mut buf, w, h, 500.0, 500.0, 40.0);

        render(&parse("-1;3,500,500,60").unwrap(), w, h, &mut buf);

        let centre = pixel(&buf, w, 500, 500);
        assert!(centre[0] > 170, "type-3 centre is {centre:?}, a 3x3 median cannot fill it");
    }

    // ── .comask Cover arrows -> clone/heal ───────────────────────────────────

    use crate::comask::{ComaskArrow, ComaskCover};
    use crate::mock::spots::apply_cover_patches;

    fn cover(arrow_type: u32, src: [f32; 2], dst: [f32; 2]) -> ComaskCover {
        ComaskCover {
            version: 2,
            mask_index: 1,
            cover_type: arrow_type,
            count: 1,
            arrows: vec![ComaskArrow { arrow_type, src, dst, flags: 0, data_len: 3 }],
        }
    }

    fn block(buf: &mut [u8], w: u32, x0: u32, y0: u32, side: u32, v: u8) {
        for y in y0..y0 + side {
            for x in x0..x0 + side {
                let p = ((y * w + x) * 4) as usize;
                buf[p..p + 3].fill(v);
            }
        }
    }

    #[test]
    fn a_cover_clone_replaces_the_sixty_by_sixty_destination_with_the_source() {
        let (w, h) = (400u32, 400u32);
        let mut buf = flat(w, h, 90);
        block(&mut buf, w, 40, 40, 60, 200);
        block(&mut buf, w, 300, 300, 60, 7);
        // The mask plane paints the destination block; 60x60 sets the patch size.
        let patches =
            cover_patches(&cover(1, [70.0, 70.0], [330.0, 330.0]), (w, h), Some((300, 300, 359, 359)), &Geometry::default());
        assert_eq!(patches.len(), 1);
        assert_eq!(patches[0].kind, CoverKind::Clone);

        apply_cover_patches(&mut buf, w, h, &Geometry::default(), &patches, (w, h));

        for y in 300..360 {
            for x in 300..360 {
                assert_eq!(pixel(&buf, w, x, y), [200, 200, 200], "({x},{y}) did not take the source");
            }
        }
        assert_eq!(pixel(&buf, w, 380, 380), [90, 90, 90], "pixels outside the patch moved");
    }

    #[test]
    fn the_cover_rim_feathers_over_two_pixels() {
        let (w, h) = (400u32, 400u32);
        let mut buf = flat(w, h, 10);
        // Wider than the patch so the rim copies a source pixel, not background.
        block(&mut buf, w, 40, 40, 80, 200);

        let patches =
            cover_patches(&cover(1, [70.0, 70.0], [330.0, 330.0]), (w, h), Some((300, 300, 359, 359)), &Geometry::default());
        apply_cover_patches(&mut buf, w, h, &Geometry::default(), &patches, (w, h));

        let rim = pixel(&buf, w, 360, 330);
        assert!(rim[0] > 20 && rim[0] < 200, "rim is {rim:?}, expected a partial blend");
        assert_eq!(pixel(&buf, w, 363, 330), [10, 10, 10], "past the 2 px feather");
    }

    #[test]
    fn a_cover_heal_adds_the_source_minus_ring_colour_offset() {
        let (w, h) = (400u32, 400u32);
        let mut buf = flat(w, h, 120);
        block(&mut buf, w, 40, 40, 60, 180);
        block(&mut buf, w, 300, 300, 60, 20);

        let patches =
            cover_patches(&cover(2, [70.0, 70.0], [330.0, 330.0]), (w, h), Some((300, 300, 359, 359)), &Geometry::default());
        assert_eq!(patches[0].kind, CoverKind::Heal);
        apply_cover_patches(&mut buf, w, h, &Geometry::default(), &patches, (w, h));

        // 180 (source) + (180 - 120) = 240.
        assert_eq!(pixel(&buf, w, 330, 330), [240, 240, 240]);
    }

    #[test]
    fn no_cover_arrows_leave_the_frame_untouched() {
        let mut buf = flat(64, 64, 128);
        let before = buf.clone();
        apply_cover_patches(&mut buf, 64, 64, &Geometry::default(), &[], (64, 64));
        assert_eq!(buf, before);
    }

    #[test]
    fn only_clone_and_heal_arrow_types_become_patches() {
        let mut list = cover(1, [70.0, 70.0], [330.0, 330.0]);
        list.arrows[0].arrow_type = 1;
        list.arrows.push(ComaskArrow { arrow_type: 2, src: [0.0, 0.0], dst: [1.0, 1.0], flags: 0, data_len: 0 });
        list.arrows.push(ComaskArrow { arrow_type: 9, src: [0.0, 0.0], dst: [1.0, 1.0], flags: 0, data_len: 0 });
        let patches = cover_patches(&list, (400, 400), Some((300, 300, 359, 359)), &Geometry::default());
        assert_eq!(patches.len(), 2);
        assert_eq!(patches[1].kind, CoverKind::Heal);
    }
}
