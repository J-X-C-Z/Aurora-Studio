//! Multi-body Banding-reduction slider ranges and automatic ISO ladders.
//!
//! This is the query layer between the RAW frame and the frontend
//! "Banding Reduction" slider (`BandingReductionAmount`):
//!
//! * [`resolve_banding_ui_range`] returns the inclusive `min..=max` slider
//!   range and its default. An empty range (`min == max`) must disable the
//!   slider, never widen to a guessed `0..=100`.
//! * [`auto_banding_amount`] returns the amount the vendor pipeline uses when
//!   the user has not overridden the field, for the bodies whose automatic
//!   rule is known.
//!
//! The module is `std` only and has no crate dependencies, so it can be
//! compiled and tested on its own. Wiring in `lib.rs`:
//!
//! ```ignore
//! pub mod banding_camera_rules;
//! ```
//!
//! A frame override is produced from the vendor triple by
//! [`BandingRange::from_vendor`], for example from
//! `c1_banding::range_for_frame(info, model)`:
//!
//! ```ignore
//! let vendor = aurora_engine::c1_banding::range_for_frame(info, model);
//! let ui = banding_camera_rules::resolve_banding_ui_range(
//!     model,
//!     vendor.map(|r| BandingRange::from_vendor(r.start, r.end, r.default)),
//! );
//! ```
//!
//! ## Provenance (read before trusting a row)
//!
//! Only the Nikon Z6III row is verified: Capture One reports
//! `IB_F_BANDINGREDUCTIONAMOUNT_START/END/DEFAULT = 0/0/0` on `_DSC5058.NEF`
//! and `_DSC5673.NEF`, and the automatic ladder is the vendor `+0x668` rule
//! anchored by an archived preview queue at ISO 1000 (`201142/782/83`,
//! A = 314.2857). Every other row is an engineering placeholder with
//! `verified = false`, a `0.0..=100.0` slider range and the automatic rule
//! disabled. Those rows must not be described as calibrated or vendor-matched
//! until a real body and a real photograph are measured; see the project rule
//! that isolated tests are static verification only.
//!
//! ## Why the ladder is piecewise-linear
//!
//! The verified Z6III rule is `0.5` up to ISO 100, `ISO/200` up to 200,
//! `1 + (ISO - 200)/1400` up to 1600, then `2`. The three knots used here
//! (100 -> 0.5, 200 -> 1.0, 1600 -> 2.0) plus linear interpolation and
//! clamping outside the ends reproduce that curve exactly, including the
//! verified ISO 1000 value `200 * (1 + 800/1400) = 314.2857`.

/// Inclusive slider range for `BandingReductionAmount`.
///
/// This is the UI-facing shape (`min`/`max`/`default`). The vendor info block
/// reports the same three numbers as `START`/`END`/`DEFAULT`; use
/// [`BandingRange::from_vendor`] to convert without depending on another
/// module's type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandingRange {
    pub min: f32,
    pub max: f32,
    pub default: f32,
}

impl BandingRange {
    /// Engineering fallback for a body that has no table entry: an adjustable
    /// `0..=100` slider starting at zero. This is a placeholder range, not a
    /// vendor measurement.
    pub const UNVERIFIED_FALLBACK: Self = Self { min: 0.0, max: 100.0, default: 0.0 };

    /// Converts a frame's vendor `start`/`end`/`default` triple into the
    /// UI-facing range. No validation happens here; a non-finite or inverted
    /// triple is rejected later by [`resolve_banding_ui_range`].
    pub const fn from_vendor(start: f32, end: f32, default: f32) -> Self {
        Self { min: start, max: end, default }
    }

    /// All three fields finite and ordered. An empty range (`min == max`) is
    /// valid on purpose: it means "this frame has no slider".
    pub fn is_valid(&self) -> bool {
        self.min.is_finite() && self.max.is_finite() && self.default.is_finite() && self.min <= self.max
    }

    /// `false` for the vendor's empty range; the slider must then be disabled
    /// rather than widened to a guessed range.
    pub fn is_adjustable(&self) -> bool {
        self.max > self.min
    }

    /// Clamps the default into `min..=max` after validation. For an empty
    /// range this pins the default to the single allowed value.
    fn with_clamped_default(&self) -> Self {
        Self { default: self.default.clamp(self.min, self.max), ..*self }
    }
}

/// How the automatic (user-not-overriding) amount is derived.
///
/// [`AutoBandingMode::FixedDefault`] and [`AutoBandingMode::IsoLadder`] are
/// vendor rules; [`AutoBandingMode::Disabled`] means "no proven automatic
/// rule", so [`auto_banding_amount`] returns `None` instead of inventing one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AutoBandingMode {
    Disabled,
    FixedDefault(f32),
    IsoLadder { base_amount: f32, iso_knots: &'static [(u32, f32)] },
}

/// One body's Banding slider range and automatic rule.
///
/// `verified` is the only field that distinguishes a measured rule from a
/// placeholder; the UI/report layer must surface it and must not present
/// `verified = false` rows as calibrated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraBandingRule {
    pub model_aliases: &'static [&'static str],
    pub range: BandingRange,
    pub auto_mode: AutoBandingMode,
    pub verified: bool,
}

/// Verified Z6III ladder: factors multiply `base_amount = 200`.
///
/// The knots are the vendor ladder's breakpoints. Interpolation is linear in
/// ISO between knots, and ISO values outside the ends clamp to the end factor
/// (`0.5` below 100, `2.0` above 1600), which is exactly the measured rule.
const Z6III_ISO_KNOTS: &[(u32, f32)] = &[(100, 0.5), (200, 1.0), (1600, 2.0)];

/// The empty range Capture One reports for the Z6III: no slider unit.
const Z6III_EMPTY_RANGE: BandingRange = BandingRange { min: 0.0, max: 0.0, default: 0.0 };

/// Built-in body table, matched in order.
///
/// The verified row comes first. Every other row is an uncalibrated
/// placeholder: range `0.0..=100.0`, default `0.0`, automatic rule disabled,
/// `verified = false`. 工程占位：待真机标定；没有真机证据前不得写成已验证。
pub static CAMERA_BANDING_RULES: &[CameraBandingRule] = &[
    // Nikon Z6III — verified. The EXIF model is "NIKON Z6_3"; "Z6III" and
    // "Nikon Z 6III" are the marketing spellings the UI may pass.
    CameraBandingRule {
        model_aliases: &["Nikon Z 6_3", "NIKON Z6_3", "Nikon Z6III", "Nikon Z 6III", "Z6III"],
        range: Z6III_EMPTY_RANGE,
        auto_mode: AutoBandingMode::IsoLadder { base_amount: 200.0, iso_knots: Z6III_ISO_KNOTS },
        verified: true,
    },
    // ---- Nikon placeholders (engineering defaults, pending calibration) ----
    CameraBandingRule {
        model_aliases: &["Nikon Z8", "NIKON Z 8"],
        range: BandingRange::UNVERIFIED_FALLBACK,
        auto_mode: AutoBandingMode::Disabled,
        verified: false,
    },
    CameraBandingRule {
        model_aliases: &["Nikon Z9", "NIKON Z 9"],
        range: BandingRange::UNVERIFIED_FALLBACK,
        auto_mode: AutoBandingMode::Disabled,
        verified: false,
    },
    CameraBandingRule {
        model_aliases: &["Nikon Z6II", "NIKON Z 6_2"],
        range: BandingRange::UNVERIFIED_FALLBACK,
        auto_mode: AutoBandingMode::Disabled,
        verified: false,
    },
    CameraBandingRule {
        model_aliases: &["Nikon Z7II", "NIKON Z 7_2"],
        range: BandingRange::UNVERIFIED_FALLBACK,
        auto_mode: AutoBandingMode::Disabled,
        verified: false,
    },
    // ---- Sony placeholders ----
    CameraBandingRule {
        model_aliases: &["Sony A7M4", "SONY ILCE-7M4", "ILCE-7M4"],
        range: BandingRange::UNVERIFIED_FALLBACK,
        auto_mode: AutoBandingMode::Disabled,
        verified: false,
    },
    CameraBandingRule {
        model_aliases: &["Sony A7R5", "SONY ILCE-7RM5", "ILCE-7RM5"],
        range: BandingRange::UNVERIFIED_FALLBACK,
        auto_mode: AutoBandingMode::Disabled,
        verified: false,
    },
    CameraBandingRule {
        model_aliases: &["Sony A1", "SONY ILCE-1", "ILCE-1"],
        range: BandingRange::UNVERIFIED_FALLBACK,
        auto_mode: AutoBandingMode::Disabled,
        verified: false,
    },
    // ---- Canon placeholders ----
    CameraBandingRule {
        model_aliases: &["Canon R5", "Canon EOS R5"],
        range: BandingRange::UNVERIFIED_FALLBACK,
        auto_mode: AutoBandingMode::Disabled,
        verified: false,
    },
    CameraBandingRule {
        model_aliases: &["Canon R6m2", "Canon EOS R6m2", "Canon EOS R6 Mark II"],
        range: BandingRange::UNVERIFIED_FALLBACK,
        auto_mode: AutoBandingMode::Disabled,
        verified: false,
    },
    // ---- Fujifilm placeholders ----
    CameraBandingRule {
        model_aliases: &["Fujifilm X-T5", "FUJIFILM X-T5"],
        range: BandingRange::UNVERIFIED_FALLBACK,
        auto_mode: AutoBandingMode::Disabled,
        verified: false,
    },
    CameraBandingRule {
        model_aliases: &["Fujifilm X-H2", "FUJIFILM X-H2"],
        range: BandingRange::UNVERIFIED_FALLBACK,
        auto_mode: AutoBandingMode::Disabled,
        verified: false,
    },
];

/// Normalises a model string for matching: ASCII alphanumerics only, upper
/// case. Spaces, hyphens and underscores are ignored, so `"NIKON Z 6_3"`,
/// `"Nikon Z6_3"` and `"nIkon z6_3"` share one key. One allocation per query;
/// aliases are compared without allocating (see [`alias_matches_key`]).
fn model_key(model: &str) -> String {
    model
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .collect()
}

/// Case-insensitive, separator-insensitive comparison against a normalised
/// key. The key must already be in [`model_key`] form.
fn alias_matches_key(alias: &str, key: &str) -> bool {
    let mut key_chars = key.chars();
    for alias_char in alias.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_uppercase()) {
        if key_chars.next() != Some(alias_char) {
            return false;
        }
    }
    key_chars.next().is_none()
}

/// Table lookup for one body. `None` for an unknown or empty model; callers
/// must not borrow another body's rule.
#[must_use]
pub fn camera_banding_rule(camera_model: Option<&str>) -> Option<&'static CameraBandingRule> {
    let key = model_key(camera_model?);
    if key.is_empty() {
        return None;
    }
    CAMERA_BANDING_RULES
        .iter()
        .find(|rule| rule.model_aliases.iter().any(|alias| alias_matches_key(alias, &key)))
}

/// Resolves the slider range for one frame.
///
/// Precedence:
/// 1. A valid `frame_override` wins over the table. All three fields must be
///    finite and `min <= max`; non-finite (`NaN`, `±inf`) or inverted values
///    are rejected and fall through to the table. The default is clamped into
///    `min..=max`, and an empty range (`min == max`) is preserved so the
///    slider stays disabled.
/// 2. A known body's table range (the Z6III row is the empty verified range).
/// 3. Unknown or missing body: [`BandingRange::UNVERIFIED_FALLBACK`]
///    (`0.0..=100.0`, default `0.0`).
#[must_use]
pub fn resolve_banding_ui_range(
    camera_model: Option<&str>,
    frame_override: Option<BandingRange>,
) -> BandingRange {
    if let Some(range) = frame_override.filter(BandingRange::is_valid) {
        return range.with_clamped_default();
    }
    camera_banding_rule(camera_model).map_or(BandingRange::UNVERIFIED_FALLBACK, |rule| rule.range)
}

/// Automatic amount for one frame, or `None` when the body has no verified
/// automatic rule (unknown body, placeholder row, or [`AutoBandingMode::Disabled`]).
#[must_use]
pub fn auto_banding_amount(camera_model: Option<&str>, iso: u32) -> Option<f32> {
    auto_banding_amount_for_mode(camera_banding_rule(camera_model)?.auto_mode, iso)
}

/// Evaluates an [`AutoBandingMode`] for one ISO.
///
/// `None` is fail-closed: disabled rule, non-finite or negative amounts, an
/// empty ladder, a ladder whose ISO knots are not strictly increasing, or any
/// non-finite/negative factor. Never fabricates a fallback amount.
#[must_use]
pub fn auto_banding_amount_for_mode(mode: AutoBandingMode, iso: u32) -> Option<f32> {
    match mode {
        AutoBandingMode::Disabled => None,
        AutoBandingMode::FixedDefault(amount) => valid_amount(amount),
        AutoBandingMode::IsoLadder { base_amount, iso_knots } => {
            let base = valid_amount(base_amount)?;
            let factor = iso_ladder_factor(iso, iso_knots)?;
            let amount = base * factor;
            amount.is_finite().then_some(amount)
        }
    }
}

fn valid_amount(amount: f32) -> Option<f32> {
    (amount.is_finite() && amount >= 0.0).then_some(amount)
}

/// Piecewise-linear ladder factor for `iso`.
///
/// ISO values at or below the first knot clamp to its factor, and values at or
/// above the last knot clamp to the last factor. Between knots the factor is
/// interpolated linearly in ISO. `None` for an empty or malformed ladder.
fn iso_ladder_factor(iso: u32, knots: &[(u32, f32)]) -> Option<f32> {
    let first = *knots.first()?;
    for pair in knots.windows(2) {
        let (low_iso, low) = pair[0];
        let (high_iso, high) = pair[1];
        if high_iso <= low_iso || !valid_factor(low) || !valid_factor(high) {
            return None;
        }
    }
    if !valid_factor(first.1) {
        return None;
    }
    if iso <= first.0 {
        return Some(first.1);
    }
    let last = knots[knots.len() - 1];
    if iso >= last.0 {
        return Some(last.1);
    }
    for pair in knots.windows(2) {
        let (low_iso, low) = pair[0];
        let (high_iso, high) = pair[1];
        if (low_iso..=high_iso).contains(&iso) {
            let span = (high_iso - low_iso) as f32;
            let t = (iso - low_iso) as f32 / span;
            return Some(low + (high - low) * t);
        }
    }
    None
}

fn valid_factor(factor: f32) -> bool {
    factor.is_finite() && factor >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const Z6III_EMPTY: BandingRange = BandingRange { min: 0.0, max: 0.0, default: 0.0 };
    const PLACEHOLDER: BandingRange = BandingRange { min: 0.0, max: 100.0, default: 0.0 };

    #[test]
    fn empty_frame_override_disables_the_slider() {
        let empty = BandingRange { min: 0.0, max: 0.0, default: 0.0 };
        assert_eq!(resolve_banding_ui_range(Some("NIKON Z6_3"), Some(empty)), Z6III_EMPTY);

        // Any `min == max` pair is an empty range, including a non-zero one.
        let pinned = BandingRange { min: 8.0, max: 8.0, default: 8.0 };
        assert_eq!(resolve_banding_ui_range(Some("Sony A7M4"), Some(pinned)), pinned);
        assert!(!resolve_banding_ui_range(None, Some(pinned)).is_adjustable());

        // The default is pinned to the empty range's single value.
        let stray = BandingRange { min: 0.0, max: 0.0, default: 25.0 };
        assert_eq!(resolve_banding_ui_range(None, Some(stray)), Z6III_EMPTY);
    }

    #[test]
    fn z6iii_matches_its_verified_rule() {
        for model in ["Nikon Z 6_3", "NIKON Z6_3", "Nikon Z6III", "Nikon Z 6III", "Z6III", "nikon z6_3"] {
            assert_eq!(resolve_banding_ui_range(Some(model), None), Z6III_EMPTY, "{model}");
            let rule = camera_banding_rule(Some(model)).unwrap_or_else(|| panic!("{model}"));
            assert!(rule.verified, "{model}");
            assert_eq!(
                rule.auto_mode,
                AutoBandingMode::IsoLadder { base_amount: 200.0, iso_knots: Z6III_ISO_KNOTS },
                "{model}"
            );
        }
        // A sibling body must not inherit the verified Z6III row.
        let z6ii = camera_banding_rule(Some("Nikon Z6II")).expect("Z6II placeholder");
        assert!(!z6ii.verified);
        assert_eq!(z6ii.range, PLACEHOLDER);
    }

    #[test]
    fn finite_frame_override_wins_over_the_table() {
        let frame = BandingRange { min: 0.0, max: 60.0, default: 20.0 };
        assert_eq!(resolve_banding_ui_range(Some("Nikon Z8"), Some(frame)), frame);
        // The default is pulled into the range instead of starting the slider
        // outside it.
        let outside = BandingRange { min: 0.0, max: 60.0, default: 90.0 };
        assert_eq!(
            resolve_banding_ui_range(Some("Nikon Z 6_3"), Some(outside)),
            BandingRange { min: 0.0, max: 60.0, default: 60.0 }
        );
        // The vendor empty range stays empty even with a stray default.
        let vendor_empty = BandingRange::from_vendor(0.0, 0.0, 0.0);
        assert_eq!(resolve_banding_ui_range(Some("Canon EOS R5"), Some(vendor_empty)), Z6III_EMPTY);
    }

    #[test]
    fn non_finite_or_inverted_overrides_are_rejected() {
        let bad = [
            BandingRange { min: f32::NAN, max: 100.0, default: 0.0 },
            BandingRange { min: 0.0, max: f32::NAN, default: 0.0 },
            BandingRange { min: 0.0, max: 100.0, default: f32::NAN },
            BandingRange { min: f32::NEG_INFINITY, max: 100.0, default: 0.0 },
            BandingRange { min: 0.0, max: f32::INFINITY, default: 0.0 },
            BandingRange { min: 0.0, max: 100.0, default: f32::NEG_INFINITY },
            BandingRange { min: 60.0, max: 0.0, default: 0.0 },
        ];
        for range in bad {
            assert_eq!(
                resolve_banding_ui_range(Some("NIKON Z6_3"), Some(range)),
                Z6III_EMPTY,
                "{range:?} must fall back to the verified body range"
            );
            assert_eq!(
                resolve_banding_ui_range(Some("Sony A7M4"), Some(range)),
                PLACEHOLDER,
                "{range:?} must fall back to the placeholder range"
            );
            assert_eq!(
                resolve_banding_ui_range(None, Some(range)),
                PLACEHOLDER,
                "{range:?} must fall back to the unknown-body range"
            );
        }
    }

    #[test]
    fn unverified_bodies_and_unknown_models_get_the_placeholder_range() {
        let bodies = [
            "Nikon Z8",
            "NIKON Z 9",
            "Nikon Z6II",
            "NIKON Z 7_2",
            "Sony A7M4",
            "SONY ILCE-7RM5",
            "Sony A1",
            "Canon R5",
            "Canon EOS R6 Mark II",
            "Fujifilm X-T5",
            "FUJIFILM X-H2",
        ];
        for model in bodies {
            let rule = camera_banding_rule(Some(model)).unwrap_or_else(|| panic!("{model}"));
            assert!(!rule.verified, "{model}");
            assert_eq!(rule.range, PLACEHOLDER, "{model}");
            assert_eq!(rule.auto_mode, AutoBandingMode::Disabled, "{model}");
            assert_eq!(resolve_banding_ui_range(Some(model), None), PLACEHOLDER, "{model}");
        }

        assert_eq!(camera_banding_rule(Some("Unknown Body")), None);
        assert_eq!(camera_banding_rule(Some("")), None);
        assert_eq!(camera_banding_rule(None), None);
        assert_eq!(resolve_banding_ui_range(Some("Unknown Body"), None), PLACEHOLDER);
        assert_eq!(resolve_banding_ui_range(None, None), PLACEHOLDER);
    }

    #[test]
    fn z6iii_iso_ladder_hits_the_verified_knots_and_ends() {
        let amount = |iso| auto_banding_amount(Some("NIKON Z 6_3"), iso).expect("verified Z6III ladder");
        // Below the first knot: clamped to 0.5 x 200.
        assert_eq!(amount(0), 100.0);
        assert_eq!(amount(50), 100.0);
        // Verified knots.
        assert_eq!(amount(100), 100.0);
        assert_eq!(amount(200), 200.0);
        assert_eq!(amount(1600), 400.0);
        // Verified interior anchor: 201142/782/83 at ISO 1000.
        let iso_1000 = amount(1000);
        assert!((iso_1000 - 314.285_74).abs() < 1e-3, "{iso_1000}");
        // Above the last knot: clamped to 2 x 200.
        assert_eq!(amount(12800), 400.0);
    }

    #[test]
    fn automatic_amounts_are_refused_without_a_verified_rule() {
        for model in ["Nikon Z8", "Sony A7M4", "Canon EOS R5", "Fujifilm X-T5", "Unknown Body"] {
            assert_eq!(auto_banding_amount(Some(model), 1000), None, "{model}");
        }
        assert_eq!(auto_banding_amount(None, 1000), None);

        assert_eq!(auto_banding_amount_for_mode(AutoBandingMode::Disabled, 100), None);
        assert_eq!(auto_banding_amount_for_mode(AutoBandingMode::FixedDefault(200.0), 100), Some(200.0));
        assert_eq!(auto_banding_amount_for_mode(AutoBandingMode::FixedDefault(f32::NAN), 100), None);
        assert_eq!(auto_banding_amount_for_mode(AutoBandingMode::FixedDefault(-1.0), 100), None);

        // A malformed ladder is refused, never partially evaluated.
        static DESCENDING: &[(u32, f32)] = &[(200, 1.0), (100, 0.5)];
        static NEGATIVE: &[(u32, f32)] = &[(100, -0.5), (1600, 2.0)];
        static EMPTY_LADDER: &[(u32, f32)] = &[];
        for knots in [DESCENDING, NEGATIVE, EMPTY_LADDER] {
            assert_eq!(
                auto_banding_amount_for_mode(
                    AutoBandingMode::IsoLadder { base_amount: 200.0, iso_knots: knots },
                    1000
                ),
                None
            );
        }
        assert_eq!(
            auto_banding_amount_for_mode(
                AutoBandingMode::IsoLadder { base_amount: f32::INFINITY, iso_knots: Z6III_ISO_KNOTS },
                1000
            ),
            None
        );
    }

    #[test]
    fn every_table_entry_is_well_formed() {
        assert!(!CAMERA_BANDING_RULES.is_empty());
        for rule in CAMERA_BANDING_RULES {
            assert!(rule.range.is_valid(), "{:?}", rule.model_aliases);
            assert!(!rule.model_aliases.is_empty(), "{rule:?}");
            assert!(
                rule.model_aliases.iter().any(|alias| !model_key(alias).is_empty()),
                "{rule:?}"
            );
            match rule.auto_mode {
                AutoBandingMode::Disabled => assert!(!rule.verified, "an unverified row must not carry an auto rule"),
                AutoBandingMode::FixedDefault(amount) => {
                    assert!(amount.is_finite() && amount >= 0.0, "{rule:?}");
                }
                AutoBandingMode::IsoLadder { base_amount, iso_knots } => {
                    assert!(base_amount.is_finite() && base_amount >= 0.0, "{rule:?}");
                    assert!(!iso_knots.is_empty(), "{rule:?}");
                    assert!(iso_knots.windows(2).all(|w| w[0].0 < w[1].0), "{rule:?}");
                    assert!(iso_knots.iter().all(|&(_, factor)| valid_factor(factor)), "{rule:?}");
                }
            }
        }
    }

    #[test]
    fn model_aliases_do_not_collide() {
        let mut seen = std::collections::BTreeMap::new();
        for (index, rule) in CAMERA_BANDING_RULES.iter().enumerate() {
            for alias in rule.model_aliases {
                let key = model_key(alias);
                if let Some(previous) = seen.insert(key.clone(), index) {
                    assert_eq!(previous, index, "alias {key:?} appears in two rules");
                }
            }
        }
    }
}
