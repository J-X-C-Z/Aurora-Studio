//! Live body-geometry rows on a **generated** standing figure, through the
//! production chain.
//!
//! Every test drives the same entry point the renderer uses:
//!
//! ```text
//! portrait_body_producer::apply_body_subjects(
//!     rgba, wide, size, job_id, extra, subjects) -> Vec<PortraitEffectRun>
//! ```
//!
//! `subjects` are [`BodySubject`] values built here from a painted figure, a
//! Halpe-26 anchor set and a local probability mask — not from a hand-written
//! `BodyGeometryInput`. That is what separates this file from
//! `portrait_body_stage_tests`, which proves the kernels accept a payload but
//! never walks the producer → stage chain with a real subject.
//!
//! ## Wiring (module lead adds both sides)
//!
//! ```text
//! #[cfg(test)]
//! #[path = "portrait_body_live_tests.rs"]
//! mod body_live_tests;
//! ```
//!
//! inside `portrait_stage.rs`, next to `portrait_body_stage_tests`.
//!
//! One textured Halpe-26 fixture for all 23 controls. The tests require
//! movement in the anatomical band, intact unowned pixels, neutral identity,
//! and a wide-only marker moved out of its original position. This measures
//! engineering behavior, not trained inference or vendor equality.

use super::*;
use super::portrait_body_inputs::BodySubject;
use super::portrait_body_producer::apply_body_subjects;
use aurora_portrait::kernels::body_pose_adapter::{
    adopt, AnchorSet, ModelLayout, ObservedKeypoint, MIN_SCORE_DEFAULT,
};

// ───────────── Generated frame, subject rect, anchors, mask ─────────────

/// Frame is larger than the subject on purpose: the ring outside the rect is the
/// ROI-out probe.
const W: u32 = 128;
const H: u32 = 176;
/// The standing figure's bounding box inside the frame.
const RECT: (u32, u32, u32, u32) = (24, 8, 80, 160);
const JOB_ID: u64 = 7;

const BG: [u8; 3] = [18, 20, 24];
const HEAD: [u8; 3] = [232, 186, 150];
const TORSO: [u8; 3] = [58, 92, 200];
const ARM: [u8; 3] = [86, 182, 96];
const LEG: [u8; 3] = [152, 92, 182];

/// Local (subject-rect) coordinate → frame pixel.
fn at(u: f32, v: f32) -> (i32, i32) {
    (
        RECT.0 as i32 + (u * RECT.2 as f32) as i32,
        RECT.1 as i32 + (v * RECT.3 as f32) as i32,
    )
}

fn put(pixels: &mut [u8], x: i32, y: i32, rgb: [u8; 3]) {
    if x < 0 || y < 0 || x >= W as i32 || y >= H as i32 {
        return;
    }
    let base = ((y as u32 * W + x as u32) * 4) as usize;
    pixels[base] = rgb[0];
    pixels[base + 1] = rgb[1];
    pixels[base + 2] = rgb[2];
    pixels[base + 3] = 255;
}

fn fill_local(pixels: &mut [u8], u0: f32, v0: f32, u1: f32, v1: f32, rgb: [u8; 3]) {
    let (x0, y0) = at(u0, v0);
    let (x1, y1) = at(u1, v1);
    for y in y0..y1 {
        for x in x0..x1 {
            put(pixels, x, y, rgb);
        }
    }
}

fn fill_ellipse_local(pixels: &mut [u8], cu: f32, cv: f32, ru: f32, rv: f32, rgb: [u8; 3]) {
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            let u = (x as f32 - RECT.0 as f32) / RECT.2 as f32;
            let v = (y as f32 - RECT.1 as f32) / RECT.3 as f32;
            let du = (u - cu) / ru;
            let dv = (v - cv) / rv;
            if du * du + dv * dv <= 1.0 {
                put(pixels, x, y, rgb);
            }
        }
    }
}

/// Standing figure: head, neck, torso, two arms, two legs, flat background.
pub(super) fn body_frame() -> Image {
    let mut pixels = vec![0u8; (W * H * 4) as usize];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            put(&mut pixels, x, y, BG);
        }
    }
    fill_ellipse_local(&mut pixels, 0.500, 0.085, 0.105, 0.070, HEAD);
    fill_local(&mut pixels, 0.435, 0.140, 0.565, 0.215, HEAD);
    fill_local(&mut pixels, 0.340, 0.215, 0.660, 0.680, TORSO);
    fill_local(&mut pixels, 0.180, 0.200, 0.280, 0.900, ARM);
    fill_local(&mut pixels, 0.720, 0.200, 0.820, 0.900, ARM);
    fill_local(&mut pixels, 0.360, 0.620, 0.470, 0.960, LEG);
    fill_local(&mut pixels, 0.530, 0.620, 0.640, 0.960, LEG);
    fill_local(&mut pixels, 0.265, 0.208, 0.735, 0.265, TORSO);
    // Internal texture, not only silhouette edges, must move. Keep each
    // palette band distinguishable and the background exactly uniform.
    for (i, p) in pixels.chunks_exact_mut(4).enumerate() {
        if p[..3] == BG { continue; }
        let (x, y) = (i % W as usize, i / W as usize);
        for (c, channel) in p[..3].iter_mut().enumerate() {
            let offset = ((x * (9 + c * 4) + y * (7 + c * 8)) % 31) as i16 - 15;
            *channel = (*channel as i16 + offset).clamp(0, 255) as u8;
        }
    }
    Image::new(W, H, pixels).expect("synthetic body frame")
}

/// Halpe-26 in model order, in **subject-local** normalized coordinates.
pub(super) fn body_halpe26() -> Vec<ObservedKeypoint> {
    let mut points = vec![ObservedKeypoint::new(0.5, 0.5, 1.0); 26];
    let mut set = |index: usize, u: f32, v: f32| {
        points[index] = ObservedKeypoint::new(u, v, 0.95);
    };
    set(0, 0.500, 0.070);
    set(1, 0.478, 0.065);
    set(2, 0.522, 0.065);
    set(3, 0.462, 0.080);
    set(4, 0.538, 0.080);
    set(5, 0.300, 0.220);
    set(6, 0.700, 0.220);
    set(7, 0.220, 0.420);
    set(8, 0.780, 0.420);
    set(9, 0.200, 0.600);
    set(10, 0.800, 0.600);
    set(11, 0.420, 0.620);
    set(12, 0.580, 0.620);
    set(13, 0.440, 0.800);
    set(14, 0.560, 0.800);
    set(15, 0.450, 0.960);
    set(16, 0.550, 0.960);
    set(17, 0.500, 0.020);
    set(18, 0.500, 0.160);
    set(19, 0.500, 0.620);
    set(20, 0.440, 0.980);
    set(21, 0.560, 0.980);
    set(22, 0.470, 0.980);
    set(23, 0.530, 0.980);
    set(24, 0.445, 0.975);
    set(25, 0.555, 0.975);
    points
}

pub(super) fn body_anchors() -> AnchorSet {
    adopt(ModelLayout::Halpe26, &body_halpe26(), MIN_SCORE_DEFAULT).expect("halpe26 anchors")
}

/// Local probability plane: 1.0 inside the silhouette, 0.35 on the rim, 0.0 outside.
pub(super) fn body_mask() -> Vec<f32> {
    let (width, height) = (RECT.2 as usize, RECT.3 as usize);
    let mut mask = vec![0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;
            let head = {
                let du = (u - 0.500) / 0.115;
                let dv = (v - 0.085) / 0.080;
                du * du + dv * dv
            };
            let torso = ((0.340..0.660).contains(&u) && (0.215..0.680).contains(&v))
                || ((0.265..0.735).contains(&u) && (0.208..0.265).contains(&v))
                || ((0.435..0.565).contains(&u) && (0.140..0.215).contains(&v));
            let arms = ((0.180..0.280).contains(&u) || (0.720..0.820).contains(&u))
                && (0.200..0.900).contains(&v);
            let legs = ((0.360..0.470).contains(&u) || (0.530..0.640).contains(&u))
                && (0.620..0.960).contains(&v);
            let inside = head <= 1.0 || torso || arms || legs;
            let rim = head <= 1.45 || ((u - 0.5).abs() <= 0.42 && (v - 0.5).abs() <= 0.52);
            mask[y * width + x] = if inside {
                1.0
            } else if rim {
                0.35
            } else {
                0.0
            };
        }
    }
    mask
}

pub(super) fn body_subject() -> BodySubject {
    BodySubject {
        x: RECT.0,
        y: RECT.1,
        width: RECT.2,
        height: RECT.3,
        anchors: body_anchors(),
        mask: body_mask(),
    }
}

// ───────────── Measurement helpers ─────────────

fn near(pixel: &[u8], rgb: [u8; 3]) -> bool {
    (0..3).all(|channel| (pixel[channel] as i32 - rgb[channel] as i32).abs() <= 24)
}

fn region(pixels: &[u8], rgb: [u8; 3]) -> Vec<usize> {
    pixels
        .chunks_exact(4)
        .enumerate()
        .filter(|(_, pixel)| near(pixel, rgb))
        .map(|(index, _)| index)
        .collect()
}

fn inside_rect(index: usize) -> bool {
    let x = (index as u32) % W;
    let y = (index as u32) / W;
    x >= RECT.0 && x < RECT.0 + RECT.2 && y >= RECT.1 && y < RECT.1 + RECT.3
}

fn changed(before: &[u8], after: &[u8]) -> Vec<usize> {
    before
        .chunks_exact(4)
        .zip(after.chunks_exact(4))
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(index, _)| index)
        .collect()
}

fn hits(changed: &[usize], band: &[usize]) -> usize {
    changed.iter().filter(|index| band.contains(index)).count()
}

// ───────────── The production call ─────────────

/// A wide-only probe in the stretched leg. It must move outside this box;
/// neither preserving the input nor applying an 8-bit delta is sufficient.
const WIDE_PATCH: (u32, u32, u32, u32) = (RECT.0 + 32, RECT.1 + 114, 5, 8);

fn wide_seed(pixels: &[u8]) -> Vec<u16> {
    let mut wide = pixels.iter().map(|byte| *byte as u16 * 257).collect::<Vec<u16>>();
    let (px, py, pw, ph) = WIDE_PATCH;
    for y in py..py + ph {
        for x in px..px + pw {
            let base = ((y * W + x) * 4) as usize;
            wide[base] = 60_000;
            wide[base + 1] = 60_000;
            wide[base + 2] = 60_000;
            wide[base + 3] = u16::MAX;
        }
    }
    wide
}

/// Runs one row through `apply_body_subjects` with `0..1` strength.
///
/// Returns the frame before, the frame after, the optional wide frame after, and
/// the rows the producer produced.
fn run_row(pf: u32, strength: f32, wide: bool) -> (Image, Vec<u8>, Option<Vec<u16>>, Vec<PortraitEffectRun>) {
    let source = body_frame();
    let subjects = [body_subject()];
    let extra = BTreeMap::from([(format!("pc:{pf}"), strength)]);
    let mut rgba = source.pixels.clone();
    let mut wide_buffer = wide.then(|| wide_seed(&source.pixels));
    let rows = apply_body_subjects(
        &mut rgba,
        wide_buffer.as_mut().map(|buffer| buffer.as_mut_slice()),
        (W, H),
        JOB_ID,
        &extra,
        &subjects,
    );
    (source, rgba, wide_buffer, rows)
}

fn executed(rows: &[PortraitEffectRun], pf: u32) -> &PortraitEffectRun {
    assert_eq!(
        rows.len(),
        1,
        "PF{pf}: one body row, got {:?}",
        rows.iter().map(|row| row.pf).collect::<Vec<_>>()
    );
    let row = &rows[0];
    assert_eq!(row.pf, pf);
    assert!(row.executed && !row.skipped, "PF{pf}: {:?}", row.refusal);
    assert!(row.refusal.is_none(), "PF{pf}: {:?}", row.refusal);
    row
}

/// The three expectations every body row must satisfy: it runs, it changes pixels
/// inside the subject, and it leaves the ring outside the subject alone.
fn assert_row_changes(pf: u32, strength: f32) -> (Image, Vec<u8>) {
    let (source, rendered, _, rows) = run_row(pf, strength, false);
    executed(&rows, pf);
    let changed = changed(&source.pixels, &rendered);
    assert!(
        !changed.is_empty(),
        "PF{pf} at {strength} changed nothing (row executed but produced identical pixels)"
    );
    assert!(
        changed.iter().all(|index| inside_rect(*index)),
        "PF{pf} changed {} pixels outside the subject rect",
        changed.iter().filter(|index| !inside_rect(**index)).count()
    );
    (source, rendered)
}

/// Require a change inside the row's tissue band; neighboring silhouette pixels
/// may move but the unrelated band still must be identical.
fn assert_band(pf: u32, strength: f32, band: [u8; 3], untouched: [u8; 3]) {
    let (source, rendered) = assert_row_changes(pf, strength);
    let changed = changed(&source.pixels, &rendered);
    let band_pixels = region(&source.pixels, band);
    assert!(
        hits(&changed, &band_pixels) > 0,
        "PF{pf}: {}/{} changed pixels fall in its own band",
        hits(&changed, &band_pixels),
        changed.len()
    );
    assert_eq!(
        hits(&changed, &region(&source.pixels, untouched)),
        0,
        "PF{pf} must not touch that band"
    );
}

// ───────────── The 13 §B body rows ─────────────

#[test]
fn pf400_slim_arm_changes_the_arm_band_only() {
    assert_band(400, 0.8, ARM, LEG);
}

#[test]
fn pf401_slim_waist_changes_the_torso_band_only() {
    assert_band(401, 0.8, TORSO, HEAD);
}

#[test]
fn pf403_breast_enhance_changes_the_torso_band_only() {
    assert_band(403, 0.8, TORSO, LEG);
}

#[test]
fn pf404_hip_enhance_moves_the_gluteal_band_not_the_head() {
    let (source, rendered) = assert_row_changes(404, 0.8);
    let changes = changed(&source.pixels, &rendered);
    // The hip anchors sit at the torso/leg paint boundary. The anatomical
    // band spans both materials; do not confuse a shirt seam with the glutes.
    let is_hip = |i: usize| {
        let v = (i / W as usize - RECT.1 as usize) as f32 / RECT.3 as f32;
        (0.58..0.73).contains(&v)
    };
    assert!(changes.iter().all(|&i| is_hip(i)), "hip warp escaped its anatomical band");
    assert!(changes.iter().any(|&i| near(&source.pixels[4*i..4*i+4], LEG)
        || near(&source.pixels[4*i..4*i+4], TORSO)), "PF404 changed only the background rim");
}

#[test]
fn pf405_body_slim_narrows_the_silhouette_inside_the_subject() {
    let (source, rendered) = assert_row_changes(405, 0.8);
    // Whole-body slimming owns every band, so it is checked by silhouette width
    // instead of a band share: the body-coloured area must shrink.
    let before = region(&source.pixels, TORSO).len()
        + region(&source.pixels, ARM).len()
        + region(&source.pixels, LEG).len();
    let after = region(&rendered, TORSO).len()
        + region(&rendered, ARM).len()
        + region(&rendered, LEG).len();
    assert!(
        after < before,
        "PF405: body silhouette {before} -> {after} pixels did not slim"
    );
}

#[test]
fn pf406_head_zoom_changes_the_head_band_only() {
    assert_band(406, 0.8, HEAD, LEG);
}

#[test]
fn pf407_leg_lengthen_changes_the_leg_band_only() {
    // Direction of the leg warp is the kernel's business; the band is this row's
    // contract. The legacy horizontal `Leg` asymmetry is deliberately not asserted.
    assert_band(407, 0.8, LEG, HEAD);
}

#[test]
fn pf408_height_writes_a_genuine_16_bit_displacement() {
    let (source, rendered, wide, rows) = run_row(408, 0.8, true);
    executed(&rows, 408);
    let changed = changed(&source.pixels, &rendered);
    assert!(!changed.is_empty(), "PF408 changed nothing");
    assert!(
        changed.iter().all(|index| inside_rect(*index)),
        "PF408 changed pixels outside the subject rect"
    );
    let wide = wide.expect("PF408 must be given a wide frame");
    let before = wide_seed(&source.pixels);
    let carried = (0..(W * H) as usize).any(|pixel| {
        let g = pixel * 4 + 1;
        before[g] < 28_000 && wide[g] > 44_000 && rendered[g] < 113
    });
    assert!(carried, "PF408 did not move the wide-only leg marker out of its source pixels");
}

#[test]
fn pf410_neck_length_changes_the_neck_band_only() {
    // The neck is painted with the head colour in this fixture, so the head band is
    // the neck's own band here; the legs must stay put.
    assert_band(410, 0.8, HEAD, LEG);
}

#[test]
fn pf415_upper_body_ratio_changes_the_torso_band_only() {
    assert_band(415, 0.8, TORSO, HEAD);
}

#[test]
fn pf416_waist_pull_changes_the_torso_without_moving_the_head() {
    assert_band(416, 0.8, TORSO, HEAD);
}

#[test]
fn pf38556_breast_lift_changes_the_torso_band_only() {
    assert_band(38556, 0.8, TORSO, LEG);
}

#[test]
fn pf90041_ai_body_shift_changes_the_body_at_three_quarter_strength() {
    // 0.75 is the UI's stored fraction for this row, not a percent.
    let (source, rendered) = assert_row_changes(90041, 0.75);
    let changed = changed(&source.pixels, &rendered);
    let body = region(&source.pixels, TORSO);
    assert!(
        hits(&changed, &body) > 0,
        "PF90041 must move body pixels, not only the rim"
    );
    // Alpha is not this row's output.
    assert!(
        source
            .pixels
            .chunks_exact(4)
            .zip(rendered.chunks_exact(4))
            .all(|(a, b)| a[3] == b[3]),
        "PF90041 must preserve alpha"
    );
}

// ───────────── Strength and neutrality contracts ─────────────

/// The A-side neutral of each row: `PF90041` is neutral at `0.5` (a shift map has
/// no "zero strength" direction), every other row at `0.0`.
fn neutral_strength(pf: u32) -> f32 {
    if pf == 90041 {
        0.5
    } else {
        0.0
    }
}

#[test]
fn neutral_values_are_a_no_op_for_every_row() {
    // One pass over the family instead of 23 copies. `PF90041` is included at its
    // own neutral (0.5), not at 0.0, so the saved A value cannot look "off".
    for pf in BODY_ROWS {
        let (source, rendered, _, rows) = run_row(pf, neutral_strength(pf), false);
        assert_eq!(
            rendered, source.pixels,
            "PF{pf} at its neutral ({}) changed pixels",
            neutral_strength(pf)
        );
        assert!(
            rows.iter().all(|row| !row.executed),
            "PF{pf} at its neutral reported an executed row"
        );
    }
}

#[test]
fn an_empty_subject_list_is_never_a_silent_success() {
    let source = body_frame();
    let extra = BTreeMap::from([("pc:400".to_string(), 0.8f32)]);
    let mut rgba = source.pixels.clone();
    let rows = apply_body_subjects(&mut rgba, None, (W, H), JOB_ID, &extra, &[]);
    assert_eq!(rgba, source.pixels, "no subject must mean no pixels");
    assert!(
        rows.iter().all(|row| !row.executed),
        "no subject must never produce an executed row: {rows:?}"
    );
    assert_eq!(rows.len(), 1);
    assert!(matches!(rows[0].refusal, Some(Refusal::MissingInput("body subject"))));
}

#[test]
fn two_owners_and_two_controls_compose_without_leaking() {
    let source = body_frame();
    let extra = BTreeMap::from([("pc:401".to_owned(), 0.8), ("pc:408".to_owned(), 0.8)]);
    let first = body_subject();
    let mut reference = source.pixels.clone();
    let single = apply_body_subjects(&mut reference, None, (W, H), JOB_ID, &extra, std::slice::from_ref(&first));
    assert_eq!(single.len(), 2);
    assert!(single.iter().all(|r| r.executed));
    assert_ne!(reference, source.pixels);
    let stride = W as usize * 4;
    let mut pair = Vec::with_capacity(source.pixels.len() * 2);
    for row in source.pixels.chunks_exact(stride) {
        pair.extend_from_slice(row);
        pair.extend_from_slice(row);
    }
    let mut second = first.clone();
    second.x += W;
    let rows = apply_body_subjects(&mut pair, None, (W * 2, H), JOB_ID, &extra, &[first, second]);
    assert_eq!(rows, single);
    for (row, expected) in pair.chunks_exact(stride * 2).zip(reference.chunks_exact(stride)) {
        assert_eq!(&row[..stride], expected);
        assert_eq!(&row[stride..], expected);
    }
}

// ───────────── The 10 §B sub-controls ─────────────

/// Every §B sub-control is a real row in this family: it must run through the
/// same producer and change pixels inside the subject. A silent neutral row is the
/// failure this catches.
fn assert_sub_control_row(pf: u32) {
    let (source, rendered) = assert_row_changes(pf, 0.8);
    let changed = changed(&source.pixels, &rendered);
    let band = match pf {
        413 | 414 => HEAD,
        30074 | 30075 | 30094 | 30095 => LEG,
        _ => TORSO,
    };
    assert!(hits(&changed, &region(&source.pixels, band)) > 0,
        "PF{pf} must move the target band, not just its background rim");
}

#[test]
fn pf411_shoulder_corner_left_runs_and_changes_the_body() {
    assert_sub_control_row(411);
}

#[test]
fn pf412_shoulder_corner_right_runs_and_changes_the_body() {
    assert_sub_control_row(412);
}

#[test]
fn pf413_swan_neck_left_runs_and_changes_the_body() {
    assert_sub_control_row(413);
}

#[test]
fn pf414_swan_neck_right_runs_and_changes_the_body() {
    assert_sub_control_row(414);
}

#[test]
fn pf417_shoulder_width_left_runs_and_changes_the_body() {
    assert_sub_control_row(417);
}

#[test]
fn pf418_shoulder_width_right_runs_and_changes_the_body() {
    assert_sub_control_row(418);
}

#[test]
fn pf30074_thigh_slim_runs_and_changes_the_body() {
    assert_sub_control_row(30074);
}

#[test]
fn pf30075_calf_slim_runs_and_changes_the_body() {
    assert_sub_control_row(30075);
}

#[test]
fn pf30094_leg_lengthen_upper_runs_and_changes_the_body() {
    assert_sub_control_row(30094);
}

#[test]
fn pf30095_leg_lengthen_lower_runs_and_changes_the_body() {
    assert_sub_control_row(30095);
}

// ───────────── The family list ─────────────

/// The 13 body PFs of §B followed by its 10 sub-controls. `PF409` is not a member
/// and is not invented here.
const BODY_ROWS: [u32; 23] = [
    400, 401, 403, 404, 405, 406, 407, 408, 410, 415, 416, 38556, 90041, //
    411, 412, 413, 414, 417, 418, 30074, 30075, 30094, 30095,
];
