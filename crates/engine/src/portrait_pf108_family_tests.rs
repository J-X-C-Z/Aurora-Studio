//! PF108 family gate: `pc:108` (common), `pc:109` (bright) and `pc:110`
//! (de-yellow) are one same-job kernel invocation, not three effects.
//!
//! The fixtures are synthetic on purpose: a 17x13 frame, a half-frame teeth
//! mask and two uniform (constant-texel) 512x512 LUTs. A uniform LUT makes
//! `teeth_lut_sample` return the same colour for every texel, so the expected
//! pixels are exactly `lerp(source, lut, mask * common * child)` and do not
//! depend on decoding a vendor asset. Nothing here is production data.

use std::collections::BTreeMap;

use super::*;

const WIDTH: u32 = 17;
const HEIGHT: u32 = 13;

/// Uniform white whiten LUT: sampling any texel yields `[255, 255, 255]`.
const WHITEN_RGB: [u8; 3] = [255, 255, 255];
/// Uniform bright LUT, deliberately far from the frame's own colour.
const BRIGHT_RGB: [u8; 3] = [0, 128, 255];

struct TeethFixture {
    frame: Vec<u8>,
    mask: Vec<u8>,
    whiten: Vec<u8>,
    bright: Vec<u8>,
}

fn uniform_rgb(rgb: [u8; 3]) -> Vec<u8> {
    rgb.iter().copied().cycle().take(LUT_BYTES).collect()
}

/// Warm frame (`r 200..223`, `g 160..183`, `b 120..143`), opaque.
fn warm_frame() -> Vec<u8> {
    let mut rgba = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let index = ((y * WIDTH + x) * 4) as usize;
            let step = ((x * 7 + y * 11) % 24) as u8;
            rgba[index] = 200 + step;
            rgba[index + 1] = 160 + step;
            rgba[index + 2] = 120 + step;
            rgba[index + 3] = 255;
        }
    }
    rgba
}

/// `id2.G` support stand-in: zero on the left half, full on the right half, so
/// "changed only inside the mask" is a real assertion and not a tautology.
fn half_mask() -> Vec<u8> {
    let mut mask = vec![0u8; (WIDTH * HEIGHT) as usize];
    for y in 0..HEIGHT {
        for x in (WIDTH / 2)..WIDTH {
            mask[(y * WIDTH + x) as usize] = 255;
        }
    }
    mask
}

fn fixture() -> TeethFixture {
    TeethFixture {
        frame: warm_frame(),
        mask: half_mask(),
        whiten: uniform_rgb(WHITEN_RGB),
        bright: uniform_rgb(BRIGHT_RGB),
    }
}

fn document(extra: &[(&str, f32)]) -> BTreeMap<String, f32> {
    extra
        .iter()
        .map(|(key, value)| ((*key).to_string(), *value))
        .collect()
}

fn family_of(extra: &[(&str, f32)]) -> Pf108Family {
    pf108_family(&document(extra)).expect("PF108 family control")
}

/// Runs the production entry point with one synthetic PF108 same-job bundle.
///
/// The producer percentages come from [`pf108_ui_percents`] exactly as the
/// renderer derives them; a refused control attaches nothing, mirroring
/// `mock::process`. `force_cpu` decides whether the CUDA attempt is refused
/// before it starts ([`with_forced_cpu`], the same switch the other stage tests
/// use) or whether the renderer is allowed to reach the device.
fn run_family(extra: &[(&str, f32)], force_cpu: bool) -> (Vec<u8>, Vec<u8>, PortraitRunReport) {
    let fixture = fixture();
    let params = document(extra);
    let (bright_ui, deyellow_ui) = pf108_ui_percents(&params).unwrap_or((0.0, 0.0));

    let masks = PortraitMasks::solid(1.0);
    let mut inputs = PortraitJobInputs::from_renderer_job(77, &masks);
    inputs.teeth = Some(TeethInputs {
        job_id: 77,
        mask_width: WIDTH,
        mask_height: HEIGHT,
        mask_source: TeethMaskSource::FaceParsingV2Id2G,
        mask_format: TeethMaskFormat::U8,
        mask_sha256: sha256_32(&fixture.mask),
        whiten_lut_identity: PF108_WHITEN_LUT_IDENTITY,
        bright_lut_identity: PF108_BRIGHT_LUT_IDENTITY,
        mask_g: &fixture.mask,
        bright_ui,
        deyellow_ui,
    });
    inputs.assets.teeth_whiten_lut = Some(&fixture.whiten);
    inputs.assets.teeth_bright_lut = Some(&fixture.bright);

    let source = fixture.frame.clone();
    let mut rendered = fixture.frame.clone();
    #[cfg(windows)]
    let report = if force_cpu {
        with_forced_cpu(|| {
            apply_portrait_effects_with_job(
                &mut rendered,
                WIDTH,
                HEIGHT,
                NormRect::FULL,
                &params,
                &inputs,
            )
        })
    } else {
        apply_portrait_effects_with_job(
            &mut rendered,
            WIDTH,
            HEIGHT,
            NormRect::FULL,
            &params,
            &inputs,
        )
    };
    #[cfg(not(windows))]
    let report = {
        // The device path is Windows-only; the flag is still part of the API so
        // every call site reads the same on both platforms.
        let _ = force_cpu;
        apply_portrait_effects_with_job(
            &mut rendered,
            WIDTH,
            HEIGHT,
            NormRect::FULL,
            &params,
            &inputs,
        )
    };
    (rendered, source, report)
}

/// The kernel oracle for the same synthetic fixture, called directly.
///
/// `common_ui` is derived from the same `0..=1` control the dispatcher feeds
/// the kernel, so the comparison is bit-exact rather than "close enough".
fn oracle(common01: f32, bright_ui: f32, deyellow_ui: f32) -> Vec<u8> {
    let fixture = fixture();
    let source = Image::new(WIDTH, HEIGHT, fixture.frame.clone()).expect("source frame");
    aurora_portrait::kernels::face_detail::teeth_whitening(
        &source,
        &fixture.mask,
        &fixture.whiten,
        &fixture.bright,
        common01 * 100.0,
        bright_ui,
        deyellow_ui,
    )
    .expect("PF108 kernel oracle")
    .output
    .pixels
}

fn changed_pixels(source: &[u8], rendered: &[u8], mask: &[u8]) -> (usize, usize) {
    let mut inside = 0usize;
    let mut outside = 0usize;
    for (index, (before, after)) in source
        .chunks_exact(4)
        .zip(rendered.chunks_exact(4))
        .enumerate()
    {
        if before != after {
            if mask[index] != 0 {
                inside += 1;
            } else {
                outside += 1;
            }
        }
    }
    (inside, outside)
}

fn changed_alpha(source: &[u8], rendered: &[u8]) -> usize {
    let mut changed = 0usize;
    for (before, after) in source.chunks_exact(4).zip(rendered.chunks_exact(4)) {
        if before[3] != after[3] {
            changed += 1;
        }
    }
    changed
}

fn only_teeth_row(report: &PortraitRunReport) -> &PortraitEffectRun {
    assert_eq!(
        report.effects.len(),
        1,
        "one PF108 job, got {:?}",
        report.effects.iter().map(|row| row.pf).collect::<Vec<_>>()
    );
    let row = &report.effects[0];
    assert_eq!(row.pf, PF108_COMMON_PC, "the family runs as PF108");
    row
}

#[test]
fn pf108_bright_child_alone_changes_masked_pixels() {
    // Only `pc:109`: the parent is absent, so the common weight must be 1.0
    // (not 0.0, which would leave the kernel at `NoEffect`), and the de-yellow
    // LUT must not be pulled up to the legacy 70 %.
    let extra = [("pc:109", 0.6f32)];
    let family = family_of(&extra);
    assert_eq!(family.common01, 1.0, "a driven child carries full common");
    let (bright_ui, deyellow_ui) = pf108_ui_percents(&document(&extra)).expect("percents");
    assert_eq!(deyellow_ui, 0.0, "no legacy sibling default for a child");

    let (rendered, source, report) = run_family(&extra, true);
    let row = only_teeth_row(&report);
    assert!(row.executed, "PF108 must execute: {:?}", row.refusal);
    assert!(row.refusal.is_none(), "{:?}", row.refusal);

    let (inside, outside) = changed_pixels(&source, &rendered, &half_mask());
    assert!(inside > 0, "the bright LUT must change masked pixels");
    assert_eq!(
        outside, 0,
        "pixels outside the id2.G support stay untouched"
    );
    assert_eq!(changed_alpha(&source, &rendered), 0, "alpha is preserved");
    assert_eq!(
        rendered,
        oracle(family.common01, bright_ui, deyellow_ui),
        "common=100 %, bright=60 %, de-yellow=0 %"
    );
}

#[test]
fn pf108_deyellow_child_alone_changes_masked_pixels() {
    let extra = [("pc:110", 0.7f32)];
    let family = family_of(&extra);
    assert_eq!(family.common01, 1.0, "a driven child carries full common");
    let (bright_ui, deyellow_ui) = pf108_ui_percents(&document(&extra)).expect("percents");
    assert_eq!(bright_ui, 0.0, "no legacy sibling default for a child");

    let (rendered, source, report) = run_family(&extra, true);
    let row = only_teeth_row(&report);
    assert!(row.executed, "PF108 must execute: {:?}", row.refusal);
    assert!(row.refusal.is_none(), "{:?}", row.refusal);

    let (inside, outside) = changed_pixels(&source, &rendered, &half_mask());
    assert!(inside > 0, "the de-yellow LUT must change masked pixels");
    assert_eq!(
        outside, 0,
        "pixels outside the id2.G support stay untouched"
    );
    assert_eq!(changed_alpha(&source, &rendered), 0, "alpha is preserved");
    assert_eq!(
        rendered,
        oracle(family.common01, bright_ui, deyellow_ui),
        "common=100 %, bright=0 %, de-yellow=70 %"
    );
}

#[test]
fn pf108_parent_and_children_dispatch_exactly_one_row() {
    // Parent + both children: one kernel invocation, one report row. The
    // children are the same job's inputs, never extra primary items.
    let extra = [("pc:108", 0.8f32), ("pc:109", 0.6f32), ("pc:110", 0.7f32)];
    let family = family_of(&extra);
    assert_eq!(family.common01, 0.8);

    let (rendered, _source, report) = run_family(&extra, true);
    let row = only_teeth_row(&report);
    assert!(row.executed, "PF108 must execute: {:?}", row.refusal);
    assert!(
        report.effects.iter().all(|row| row.pf == PF108_COMMON_PC),
        "no child may appear as its own effect row"
    );
    assert_eq!(
        rendered,
        oracle(family.common01, family.bright_ui(), family.deyellow_ui()),
        "common=80 %, bright=60 %, de-yellow=70 %"
    );
    assert_ne!(
        rendered,
        oracle(1.0, family.bright_ui(), family.deyellow_ui()),
        "a parent-driven document must not run at the full common weight"
    );
}

#[test]
fn pf108_all_zero_is_not_scheduled() {
    let (rendered, source, report) =
        run_family(&[("pc:108", 0.0), ("pc:109", 0.0), ("pc:110", 0.0)], true);
    assert!(
        report.effects.is_empty(),
        "all-neutral controls must not schedule PF108: {:?}",
        report.effects.iter().map(|row| row.pf).collect::<Vec<_>>()
    );
    assert!(report.effects.iter().all(|row| !row.executed));
    assert_eq!(rendered, source, "no dispatch, no pixel change");

    // The same holds when only the parent is stored at zero.
    let (rendered, source, report) = run_family(&[("pc:108", 0.0)], true);
    assert!(report.effects.is_empty());
    assert_eq!(rendered, source);
}

#[test]
fn pf108_family_controls_refuse_non_finite_and_out_of_range() {
    let cases: Vec<(Vec<(&str, f32)>, &str)> = vec![
        (vec![("pc:108", 1.5f32)], "PF108 common UI"),
        (vec![("pc:109", f32::NAN)], "PF108 bright UI"),
        (vec![("pc:110", f32::INFINITY)], "PF108 de-yellow UI"),
        (vec![("pc:108", 0.5), ("pc:109", -0.25)], "PF108 bright UI"),
    ];
    for (extra, name) in &cases {
        let (rendered, source, report) = run_family(extra, true);
        let row = only_teeth_row(&report);
        assert!(!row.executed, "{extra:?} must not execute");
        assert_eq!(
            row.refusal,
            Some(Refusal::ValueOutOfRange(*name)),
            "{extra:?}"
        );
        assert_eq!(
            row.skip_reason,
            Some(PortraitSkipReason::InputRefused(Refusal::ValueOutOfRange(
                *name
            ))),
            "{extra:?}"
        );
        assert_eq!(rendered, source, "{extra:?} must not change pixels");
    }
}

#[test]
fn pf108_family_keeps_legacy_parent_defaults() {
    // Legacy document: only `pc:108`, so both sibling LUTs keep their 70 %.
    assert_eq!(
        family_of(&[("pc:108", 0.5)]),
        Pf108Family {
            common01: 0.5,
            bright01: PF108_LEGACY_SIBLING_DEFAULT01,
            deyellow01: PF108_LEGACY_SIBLING_DEFAULT01,
        }
    );

    // Child alone: full common weight, the sibling stays neutral.
    assert_eq!(
        family_of(&[("pc:109", 0.6)]),
        Pf108Family {
            common01: 1.0,
            bright01: 0.6,
            deyellow01: 0.0,
        }
    );
    assert_eq!(
        family_of(&[("pc:110", 0.7)]),
        Pf108Family {
            common01: 1.0,
            bright01: 0.0,
            deyellow01: 0.7,
        }
    );

    // An explicit zero parent behaves like an absent one for a driven child.
    assert_eq!(
        family_of(&[("pc:108", 0.0), ("pc:109", 0.6)]),
        Pf108Family {
            common01: 1.0,
            bright01: 0.6,
            deyellow01: 0.0,
        }
    );

    // Parent + one child: the absent sibling must not inherit 70 %.
    assert_eq!(
        family_of(&[("pc:108", 0.5), ("pc:109", 0.3)]),
        Pf108Family {
            common01: 0.5,
            bright01: 0.3,
            deyellow01: 0.0,
        }
    );

    // Nothing set, or every control neutral: not scheduled.
    assert!(!pf108_family(&BTreeMap::new()).unwrap().is_active());
    assert!(
        !pf108_family(&document(&[
            ("pc:108", 0.0),
            ("pc:109", 0.0),
            ("pc:110", 0.0)
        ]))
        .unwrap()
        .is_active()
    );
    // A driven parent with both children explicitly zeroed is a zero mask
    // product, so it is not scheduled either.
    assert!(
        !pf108_family(&document(&[
            ("pc:108", 0.5),
            ("pc:109", 0.0),
            ("pc:110", 0.0)
        ]))
        .unwrap()
        .is_active()
    );

    // The producer percentages are the same resolver, in percent.
    let (bright_ui, deyellow_ui) =
        pf108_ui_percents(&document(&[("pc:109", 0.6)])).expect("percents");
    assert!((bright_ui - 60.0).abs() < 1.0e-3, "bright_ui={bright_ui}");
    assert_eq!(deyellow_ui, 0.0);
}

/// Largest per-channel difference between two RGBA renders, as
/// `(pixels that differ, largest code-value delta)`.
fn max_channel_diff(left: &[u8], right: &[u8]) -> (usize, usize) {
    let mut pixels = 0usize;
    let mut worst = 0i32;
    for (a, b) in left.chunks_exact(4).zip(right.chunks_exact(4)) {
        let mut differs = false;
        for channel in 0..4 {
            let delta = (a[channel] as i32 - b[channel] as i32).abs();
            if delta != 0 {
                differs = true;
                worst = worst.max(delta);
            }
        }
        if differs {
            pixels += 1;
        }
    }
    (pixels, worst as usize)
}

/// PF109 alone and PF110 alone on Windows: each is the same single PF108
/// same-job invocation, and when a CUDA device is present that invocation must
/// be the device kernel — non-empty dispatch, no CPU fallback — reproducing the
/// CPU kernel's pixels exactly, or within one u8 rounding step that is reported
/// on its own line rather than hidden in a tolerance.
///
/// The two children are also compared with each other: they drive different
/// LUTs, so "one PF108 row" must not be read as "one result".
///
/// Gating matches the other device tests: a host without a usable CUDA device
/// only reports what it saw, and `PORTRAIT_CUDA_REQUIRED=1` turns a CPU row that
/// the stage chose on its own into a failure that names the missing branch.
#[test]
fn pf108_child_rows_share_one_cuda_dispatch_on_windows() {
    #[cfg(not(windows))]
    {
        eprintln!("skip: the PF108 device path is Windows-only");
        return;
    }

    #[cfg(windows)]
    {
        let mask = half_mask();
        let mut rendered_children: Vec<Vec<u8>> = Vec::with_capacity(2);
        for (label, extra) in [
            ("PF109 alone", [("pc:109", 0.6f32)]),
            ("PF110 alone", [("pc:110", 0.7f32)]),
        ] {
            let (cuda, source, report) = run_family(&extra, false);
            let row = only_teeth_row(&report);
            assert!(
                row.executed,
                "{label}: PF108 must execute: {:?}",
                row.refusal
            );
            assert!(row.refusal.is_none(), "{label}: {:?}", row.refusal);

            let (inside, outside) = changed_pixels(&source, &cuda, &mask);
            assert!(
                inside > 0,
                "{label}: the child LUT must change masked pixels"
            );
            assert_eq!(
                outside, 0,
                "{label}: pixels outside the id2.G support stay untouched"
            );
            assert_eq!(
                changed_alpha(&source, &cuda),
                0,
                "{label}: alpha is preserved"
            );

            if row.backend == PortraitBackend::Cuda {
                assert!(
                    !row.dispatches.is_empty(),
                    "{label}: a CUDA row must carry a dispatch"
                );
                assert_eq!(
                    row.fallback_reason, None,
                    "{label}: CUDA ran, so nothing fell back"
                );
                eprintln!(
                    "{label}: CUDA {} on {}",
                    row.dispatches[0].kernel, row.dispatches[0].device
                );

                let (cpu, cpu_source, cpu_report) = run_family(&extra, true);
                let cpu_row = only_teeth_row(&cpu_report);
                assert_eq!(
                    cpu_row.backend,
                    PortraitBackend::Cpu,
                    "{label}: forced CPU row"
                );
                assert_eq!(cpu_source, source, "{label}: same source frame");
                assert_eq!(cuda.len(), cpu.len(), "{label}: same frame shape");
                let (pixels, worst) = max_channel_diff(&cuda, &cpu);
                if pixels == 0 {
                    eprintln!("{label}: CUDA and CPU are bit-identical");
                } else {
                    // u8 has exactly one rounding step; a different FMA order on
                    // the device may land on the neighbouring code value.
                    assert_eq!(
                        worst, 1,
                        "{label}: {pixels} pixels differ by up to {worst} code values"
                    );
                    eprintln!(
                        "{label}: {pixels} pixels differ by exactly 1 code value (u8 rounding)"
                    );
                }
            } else {
                assert!(
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                    "{label}: PORTRAIT_CUDA_REQUIRED=1 but PF108 ran on {:?} with fallback {:?}. \
                     The same-job device route (try_cuda_same_job_effect -> cuda_router::same_job(108) -> \
                     run::<router::TeethWhiten>) either was not reachable on this host or declined the job; \
                     the fallback above is the stage's own reason.",
                    row.backend,
                    row.fallback_reason
                );
                eprintln!(
                    "{label}: no CUDA row on this host (backend {:?}, fallback {:?}); device assertions skipped",
                    row.backend, row.fallback_reason
                );
            }
            rendered_children.push(cuda);
        }

        assert_eq!(rendered_children.len(), 2);
        assert_ne!(
            rendered_children[0], rendered_children[1],
            "PF109 alone and PF110 alone drive different LUTs and must not render the same frame"
        );
    }
}
