//! Vendor-exact long-tail portrait gates and pixel laws (SG13–SG15, SG18–SG24).
//!
//! Standalone `std`-only module: no external crates, no `unsafe`, no file I/O, no
//! environment access. Every constant and every arithmetic step below is a
//! transcription of an evidence row that was read back from the fixed binaries
//! (`pix-worker.exe` SHA256 `509d2293…6b84`, `pixcake.exe` SHA256 `a84519da…64c7`)
//! or from the read-only shader sources under `04_reference/`.
//!
//! Evidence index: `03_algorithms/reports/PORTRAIT-STATIC-GAPS-20261125.tsv`
//!   * SG13–SG15 — PF0 *Spotless* effect factory, program ids, sampler binding,
//!     numeric uniforms and default values.
//!   * SG18–SG19 — PF90034 / PF30188 / PF30189 double-chin weight products and
//!     the six-channel → four-mask mapping.
//!   * SG20 — PF30209 jawline boolean activation and the fixed 33-point window.
//!   * SG23 — PF90088 / PF90113 / PF90114 stray-hair product gate.
//!   * SG24 — PF90085 face skin-tone piecewise remap.
//!
//! Provenance reports:
//!   `90_work/lanes/portrait-parse/static-gaps-20261124/task03-pf0-shader-binding/FINDINGS.md`
//!   `90_work/lanes/portrait-parse/static-gaps-20261124/task05-longtail-chin-jaw/FINDINGS.md`
//!   `90_work/lanes/portrait-parse/static-gaps-20261124/task06-longtail-hair-skin/FINDINGS.md`
//!   `04_reference/legacy-root/Aurora/逆向工程/像素蛋糕/glsl/Spotless/SpotlessProcessNormal.frag`
//!   `04_reference/legacy-root/Aurora/逆向工程/像素蛋糕/glsl/Spotless/SpotlessProcessWithSkinMask.frag`
//!
//! Scope and honesty:
//!   * What is implemented here is PROVEN at instruction/machine level: the
//!     uniforms, thresholds, products and the per-pixel GPU fragment law.
//!   * What is *not* proven and therefore not implemented: the runtime value of
//!     `u_degree` (static default 0.0, injection point LIVE_ONLY), which FBO
//!     produces the blur / face-mask / skin-mask planes, the runtime physical
//!     meaning of PF90085's `v`, and any vendor pixel parity. No smoothing,
//!     clamping or normalization has been added beyond what the evidence shows.
//!
//! Build and test (zero warnings required):
//!   `rustc --edition 2021 --test longtail_exact_vendor.rs`
//!   `./longtail_exact_vendor`  (or `longtail_exact_vendor.exe` on Windows)

#![forbid(unsafe_code)]

// ===========================================================================
// 1. PF0 — Spotless (祛斑祛痘), program 0xD1 / 0xD2
// ===========================================================================

/// `lT` — `u_dist_factor`, filter field `+0x2AC`, default written at creation.
///
/// SG15: binary float `28.87`, constant cluster `0x8EAF370` (worker) /
/// `0x91B6E30` (pixcake).
pub const SPOTLESS_DIST_FACTOR: f32 = 28.87;

/// `lU` — `u_dist_threshold`, filter field `+0x2B0`, default `0.5` (SG15).
pub const SPOTLESS_DIST_THRESHOLD: f32 = 0.5;

/// `lR` — `u_dark_diff_min_threshold`, filter field `+0x2B4`, default `0.0`.
pub const SPOTLESS_DARK_MIN: f32 = 0.0;

/// `lW` — `u_dark_diff_max_threshold`, filter field `+0x2B8`, default `0.3`.
pub const SPOTLESS_DARK_MAX: f32 = 0.3;

/// `lQ` — `u_white_diff_factor`, filter field `+0x2BC`, default `0.7`.
pub const SPOTLESS_WHITE_FACTOR: f32 = 0.7;

/// `ag` — `u_degree`, filter field `+0x2A8`.
///
/// SG15: the *static* default is `0.0`; the runtime per-face injection point is
/// LIVE_ONLY (see SG16). Callers must not assume a non-zero strength.
pub const SPOTLESS_DEGREE_DEFAULT: f32 = 0.0;

/// Program id of `GLSpotlessProcessNormalFilter` (worker `0x57F52F5`).
pub const SPOTLESS_PROGRAM_NORMAL: u32 = 0xD1;

/// Program id of `GLSpotlessProcessWithSkinMaskFilter` (worker `0x57F54B5`).
pub const SPOTLESS_PROGRAM_WITH_SKIN_MASK: u32 = 0xD2;

/// The six numeric uniforms of the PF0 fragment stage.
///
/// Field names map one-to-one onto the GLSL uniforms. The recovered binder
/// `0x57F4850` writes them in the order `lT → ag → lU → lR → lW → lQ`, from the
/// filter fields at `+0x2AC / +0x2A8 / +0x2B0 / +0x2B4 / +0x2B8 / +0x2BC`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpotlessUniforms {
    /// `lT` / `u_dist_factor`.
    pub dist_factor: f32,
    /// `lU` / `u_dist_threshold`.
    pub dist_threshold: f32,
    /// `lR` / `u_dark_diff_min_threshold`.
    pub dark_min: f32,
    /// `lW` / `u_dark_diff_max_threshold`.
    pub dark_max: f32,
    /// `lQ` / `u_white_diff_factor`.
    pub white_factor: f32,
    /// `ag` / `u_degree` (static default 0.0; LIVE_ONLY at runtime).
    pub degree: f32,
}

impl SpotlessUniforms {
    /// The values written by the vendor at filter creation (SG13–SG15).
    pub const fn vendor_default() -> Self {
        Self {
            dist_factor: SPOTLESS_DIST_FACTOR,
            dist_threshold: SPOTLESS_DIST_THRESHOLD,
            dark_min: SPOTLESS_DARK_MIN,
            dark_max: SPOTLESS_DARK_MAX,
            white_factor: SPOTLESS_WHITE_FACTOR,
            degree: SPOTLESS_DEGREE_DEFAULT,
        }
    }
}

/// The two exact PF0 fragment variants.
///
/// `Normal` binds `s_texture`, `s_blur_texture`, `s_facemask_texture` and derives
/// the skin weight from an inline HSV heuristic. `WithSkinMask` binds the same
/// three planes plus `s_skinmask_texture` and reads the weight from that plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpotlessMode {
    /// `GLSpotlessProcessNormalFilter`, program id `0xD1`.
    Normal,
    /// `GLSpotlessProcessWithSkinMaskFilter`, program id `0xD2`.
    WithSkinMask,
}

impl SpotlessMode {
    /// Program id recovered from the filter `Init` slot (SG13).
    pub const fn program_id(self) -> u32 {
        match self {
            SpotlessMode::Normal => SPOTLESS_PROGRAM_NORMAL,
            SpotlessMode::WithSkinMask => SPOTLESS_PROGRAM_WITH_SKIN_MASK,
        }
    }

    /// True for the variant that samples `s_skinmask_texture` (texture unit 3).
    pub const fn uses_skin_mask(self) -> bool {
        matches!(self, SpotlessMode::WithSkinMask)
    }

    /// Sampler names bound by `BindTextures`, in texture-unit order.
    ///
    /// `Normal`: `0x57F53F0`; `WithSkinMask`: `0x57F55B0` (SG13).
    pub const fn samplers(self) -> &'static [&'static str] {
        match self {
            SpotlessMode::Normal => &["s_texture", "s_blur_texture", "s_facemask_texture"],
            SpotlessMode::WithSkinMask => &[
                "s_texture",
                "s_blur_texture",
                "s_facemask_texture",
                "s_skinmask_texture",
            ],
        }
    }
}

/// GLSL `step(edge, x)`.
#[inline]
fn glsl_step(edge: f32, x: f32) -> f32 {
    if x < edge {
        0.0
    } else {
        1.0
    }
}

/// GLSL `mix(x, y, a)` = `x * (1 - a) + y * a`.
#[inline]
fn glsl_mix(x: f32, y: f32, a: f32) -> f32 {
    x * (1.0 - a) + y * a
}

/// `length(vec3)` = `sqrt(x*x + y*y + z*z)`.
#[inline]
fn vec3_length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// `max(a, 0.0)` per component, as written by the vendor fragment.
#[inline]
fn max_zero(v: [f32; 3]) -> [f32; 3] {
    [v[0].max(0.0), v[1].max(0.0), v[2].max(0.0)]
}

/// `rgb2hsv` copied verbatim from `SpotlessProcessNormal.frag` lines 38–48.
///
/// Saturation is hard-coded to `0.0` by the vendor ("for optimization"), so only
/// hue and value carry information. `K = (0, -1/3, 2/3, -1)` and `e = 1e-10`.
pub fn spotless_rgb2hsv(c: [f32; 3]) -> [f32; 3] {
    let (r, g, b) = (c[0], c[1], c[2]);
    let s = glsl_step(b, g);
    let p_x = glsl_mix(b, g, s);
    let p_y = glsl_mix(g, b, s);
    let p_z = glsl_mix(-1.0, 0.0, s);
    let p_w = glsl_mix(2.0 / 3.0, -1.0 / 3.0, s);
    let t = glsl_step(p_x, r);
    let q_x = glsl_mix(p_x, r, t);
    let q_y = p_y;
    let q_z = glsl_mix(p_w, p_z, t);
    let q_w = glsl_mix(r, p_x, t);
    let d = q_x - q_w.min(q_y);
    let e = 1.0e-10f32;
    let hue = (q_z + (q_w - q_y) / (6.0 * d + e)).abs();
    [hue, 0.0, q_x]
}

/// `GetSkinValue` from an already computed hue/value pair
/// (`SpotlessProcessNormal.frag` lines 50–77).
///
/// The hard band `0.18 <= hue <= 0.89` or `value <= 0.2` returns 0; the three
/// soft ramps are evaluated in vendor order and combined with `min`.
pub fn spotless_skin_value_from_hue(hue: f32, value: f32) -> f32 {
    if (0.18 <= hue && hue <= 0.89) || value <= 0.2 {
        return 0.0;
    }
    let mut skin_value = 1.0f32;
    if 0.16 < hue && hue < 0.18 {
        skin_value = skin_value.min((0.18 - hue) * 50.0);
    } else if 0.89 < hue && hue < 0.91 {
        skin_value = skin_value.min(1.0 - (0.91 - hue) * 50.0);
    } else if 0.2 < value && hue < 0.3 {
        skin_value = skin_value.min(1.0 - (0.3 - value) * 10.0);
    }
    skin_value
}

/// Inline HSV skin heuristic used only by [`SpotlessMode::Normal`].
pub fn spotless_skin_value(rgb: [f32; 3]) -> f32 {
    let hsv = spotless_rgb2hsv(rgb);
    spotless_skin_value_from_hue(hsv[0], hsv[2])
}

/// One pixel of the exact vendor Spotless fragment.
///
/// Inputs mirror the samplers the variant actually reads:
///   * `src` — `s_texture` (`.rgba`; alpha is copied unchanged to the output),
///   * `blur` — `s_blur_texture` (`.rgb`),
///   * `face_mask` — `s_facemask_texture` (`.r`),
///   * `skin_mask` — `s_skinmask_texture` (`.r`), read **only** by
///     [`SpotlessMode::WithSkinMask`]; ignored by `Normal`, which uses the
///     inline heuristic on `src.rgb` instead.
///
/// The law, line for line from the recovery:
/// ```text
/// res = src.rgb
/// if face_mask > 0:
///     skin   = (mode == WithSkinMask) ? skin_mask : GetSkinValue(src.rgb)
///     factor = min(1, max(0, face_mask * skin * degree))
///     if factor > 0:
///         dist = length(max(blur - src, 0)) * dist_factor
///         if dist > dist_threshold:
///             diff = min(clamp(blur - src, dark_min, dark_max) + src, 1)
///             res  = mix(src, diff, factor)
///         else:
///             inv = length(max(src - blur, 0)) * dist_factor
///             if inv > dist_threshold:
///                 ref = blur + (src - blur) * white_factor
///                 res = mix(src, ref, factor)
/// out = vec4(res, src.a)
/// ```
///
/// The vendor declares `mediump` / `lowp` locals, so a real GPU may quantize
/// intermediate values; this function is the f32 reference of the same law and
/// makes no claim of byte-identity with a specific GPU.
pub fn spotless_fragment(
    mode: SpotlessMode,
    src: [f32; 4],
    blur: [f32; 3],
    face_mask: f32,
    skin_mask: f32,
    uniforms: &SpotlessUniforms,
) -> [f32; 4] {
    let rgb = [src[0], src[1], src[2]];
    let mut res = rgb;

    if face_mask > 0.0 {
        let skin = if mode.uses_skin_mask() {
            skin_mask
        } else {
            spotless_skin_value(rgb)
        };
        let factor = (face_mask * skin * uniforms.degree)
            .max(0.0)
            .min(1.0);

        if factor > 0.0 {
            let dark = max_zero([blur[0] - rgb[0], blur[1] - rgb[1], blur[2] - rgb[2]]);
            let dist = vec3_length(dark) * uniforms.dist_factor;

            if dist > uniforms.dist_threshold {
                let mut diff = [0.0f32; 3];
                for channel in 0..3 {
                    let clamped = (blur[channel] - rgb[channel])
                        .max(uniforms.dark_min)
                        .min(uniforms.dark_max);
                    diff[channel] = (rgb[channel] + clamped).min(1.0);
                }
                for channel in 0..3 {
                    res[channel] = glsl_mix(rgb[channel], diff[channel], factor);
                }
            } else {
                let light = max_zero([rgb[0] - blur[0], rgb[1] - blur[1], rgb[2] - blur[2]]);
                let inv_dist = vec3_length(light) * uniforms.dist_factor;

                if inv_dist > uniforms.dist_threshold {
                    for channel in 0..3 {
                        let difference = rgb[channel] - blur[channel];
                        let reference = blur[channel] + difference * uniforms.white_factor;
                        res[channel] = glsl_mix(rgb[channel], reference, factor);
                    }
                }
            }
        }
    }

    [res[0], res[1], res[2], src[3]]
}

// ===========================================================================
// 2. PF90034 / PF30188 / PF30189 — double-chin weights and masks
// ===========================================================================

/// Execution gate for the double-chin stage.
///
/// SG18: binary64 `1e-4` at `0x8BFC350` (`2D431CEBE2361A3F`), loaded by
/// `movsd` at `0x7E4731C` and compared with `ucomisd` at `0x7E47324` /
/// `0x7E47331`. The stage is skipped only when **both** weights are below it.
pub const DOUBLE_CHIN_GATE: f64 = 1.0e-4;

/// The six FaceParsing channels that feed the double-chin masks, in the exact
/// order of the vendor selection points (SG19, `0x7AEE08A`…`0x7AEE58F`).
pub const DOUBLE_CHIN_CHANNELS: [u8; 6] = [10, 11, 14, 6, 8, 9];

/// Fallback used for an out-of-bounds weight lookup (SG18, `0x8B96F30` = `1.0f`).
pub const DOUBLE_CHIN_WEIGHT_DEFAULT: f32 = 1.0;

/// The two per-face binary32 weights produced by the double-chin effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoubleChinWeights {
    /// `W_geometry = PF30188 × PF90034` (`mulss` at `0x7AEEA97`).
    pub w_geometry: f32,
    /// `W_shadow = PF90034 × PF30189` (`mulss` at `0x7AEEA9C`).
    pub w_shadow: f32,
}

impl DoubleChinWeights {
    /// `true` when the vendor stage would run.
    ///
    /// The vendor comparison is binary64 (`ucomisd`), and the reported machine
    /// semantics skip only when both values are `< 1e-4`; a NaN operand makes
    /// both `<` comparisons false, so the stage is *not* skipped. That machine
    /// behaviour is reproduced exactly here.
    pub fn passes_gate(&self) -> bool {
        double_chin_passes_gate(self.w_shadow, self.w_geometry)
    }
}

/// Compute the two weights. No clamp, no `/100`, no `0..1` normalization:
/// the vendor multiplies the raw per-face binary32 values (`mulss` only).
///
/// The specification writes `f32::from(pf30188 * pf90034)`; that is the C++
/// `float(...)` narrowing of an already-`float` product, i.e. exactly the same
/// binary32 `mulss` result. There is no `f64` intermediate and no rounding step
/// to reproduce.
pub fn double_chin_weights(pf90034: f32, pf30188: f32, pf30189: f32) -> DoubleChinWeights {
    DoubleChinWeights {
        w_geometry: pf30188 * pf90034,
        w_shadow: pf90034 * pf30189,
    }
}

/// Out-of-bounds weight lookup fallback (SG18, `0x8B96F30` = `1.0f`).
pub fn double_chin_weight_or_default(found: Option<f32>) -> f32 {
    found.unwrap_or(DOUBLE_CHIN_WEIGHT_DEFAULT)
}

/// The recovered gate: skip the stage only when both weights are `< 1e-4`
/// under the vendor's binary64 comparison.
pub fn double_chin_passes_gate(w_shadow: f32, w_geometry: f32) -> bool {
    let shadow = w_shadow as f64;
    let geometry = w_geometry as f64;
    !(shadow < DOUBLE_CHIN_GATE && geometry < DOUBLE_CHIN_GATE)
}

/// Four `u8` single-channel masks consumed by the double-chin fold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DoubleChinMasks {
    /// `A` — saturating `ch10 + ch11` (`rbp+0x2180`, materialized `0x7E4B0E8`).
    pub a: u8,
    /// `B` — `ch6` (`rbp+0x2188`, materialized `0x7E4B0CE`).
    pub b: u8,
    /// `C` — `ch14` (`rbp+0x2190`, materialized `0x7E4B102`).
    pub c: u8,
    /// `D` — saturating `ch8 + ch9` (`rbp+0x2198`).
    pub d: u8,
}

/// Compose the six selected channels into the four `u8` masks.
///
/// `channels` is in [`DOUBLE_CHIN_CHANNELS`] order:
/// `[ch10, ch11, ch14, ch6, ch8, ch9]`.
///
/// SG19 / task05 §4.4: `A = sat_u8(ch10 + ch11)`, `B = ch6`, `C = ch14`,
/// `D = sat_u8(ch8 + ch9)`. The slot/fold order and channel selection are
/// PROVEN; the saturation location itself is PROVEN (s400) / INFERRED (position).
pub fn double_chin_masks(channels: [u8; 6]) -> DoubleChinMasks {
    DoubleChinMasks {
        a: channels[0].saturating_add(channels[1]),
        b: channels[3],
        c: channels[2],
        d: channels[4].saturating_add(channels[5]),
    }
}

// ===========================================================================
// 3. PF30209 — jawline enhancement
// ===========================================================================

/// First index of the fixed 33-point jaw contour window (`movups [rcx+0x368]`).
pub const JAWLINE_WINDOW_START: usize = 109;

/// Last index of the fixed window, inclusive (`movsd [rcx+0x468]`).
pub const JAWLINE_WINDOW_END: usize = 141;

/// Number of contour points in the window (`141 - 109 + 1`).
pub const JAWLINE_WINDOW_LEN: usize = 33;

/// Centre landmark (`movss` at `0x80A0A2C` / `0x80A0A44`, `+0x1B0`/`+0x1B4`).
pub const JAWLINE_CENTER_INDEX: usize = 54;

/// Length of the vendor landmark array (`float2 L[194]`).
pub const JAWLINE_LANDMARK_COUNT: usize = 194;

/// True when any per-face PF30209 value is strictly positive.
///
/// SG20 / task05 §5.2: `active = any(node float(+0x20) > 0)`;
/// the value is used only as a boolean and is never multiplied.
pub fn jawline_active(values: &[f32]) -> bool {
    values.iter().any(|value| *value > 0.0)
}

/// The fixed jawline window extracted from a landmark array.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JawlineWindow {
    /// `L[109..=141]` in array order (33 points).
    pub contour: [[f32; 2]; JAWLINE_WINDOW_LEN],
    /// `L[54]`.
    pub center: [f32; 2],
}

/// Extract `contour = L[109..=141]` and `center = L[54]`.
///
/// Returns `None` when the array is shorter than `142` entries. The vendor
/// indexes these offsets unconditionally; the check here only prevents a bogus
/// caller from reading outside the supplied slice.
///
/// Note: the ordered jawline index table is NOT a PE constant (SG20 NOT-FOUND);
/// this window is the fixed index range only, not the ordered contour order.
pub fn jawline_window(landmarks: &[[f32; 2]]) -> Option<JawlineWindow> {
    if landmarks.len() < JAWLINE_WINDOW_END + 1 {
        return None;
    }
    let mut contour = [[0.0f32; 2]; JAWLINE_WINDOW_LEN];
    contour.copy_from_slice(&landmarks[JAWLINE_WINDOW_START..=JAWLINE_WINDOW_END]);
    Some(JawlineWindow {
        contour,
        center: landmarks[JAWLINE_CENTER_INDEX],
    })
}

// ===========================================================================
// 4. PF90088 / PF90113 / PF90114 — stray-hair product gate
// ===========================================================================

/// Stray-hair activation gate on PF90088.
///
/// SG23 / task06 §3.3: binary64 `0.001` at `0x8BCBA40`, loaded by `movsd` at
/// `0x7B4FB2F` and compared at `0x7B4FE16` / `0x7B4FF77`.
pub const STRAY_HAIR_GATE: f64 = 0.001;

/// The two per-region degrees produced by the stray-hair effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrayHairDegrees {
    /// `personDeg = d × p` (`mulss` at `0x7B502DD` / `0x7B506F0` / `0x7B50A96`).
    pub person_deg: f32,
    /// `backgroundDeg = b × d` (`mulss` at `0x7B506D7` / `0x7B50A92`).
    pub background_deg: f32,
}

impl StrayHairDegrees {
    /// Re-check the vendor gate from the raw PF90088 value.
    ///
    /// The gate is `d > 0.001` on PF90088 alone; it cannot be recovered from the
    /// two products, so the caller has to supply `d`.
    pub fn active_from_source(&self, d: f32) -> bool {
        stray_hair_active(d)
    }
}

/// True when PF90088's own value passes `d > 0.001` (binary64 comparison).
///
/// Note the vendor loads the threshold as a binary64 `0.001` and compares the
/// converted `f32` value against it. The `f32` nearest to `0.001` converts to
/// `0.0010000000474974513`, which is *above* the binary64 constant, so
/// `stray_hair_active(0.001f32)` is `true`. That is the machine behaviour, not a
/// rounded-up convenience.
pub fn stray_hair_active(d: f32) -> bool {
    (d as f64) > STRAY_HAIR_GATE
}

/// Compute the two region degrees.
///
/// SG23: `person_deg = d * p`, `background_deg = b * d`, where
/// `d = PF90088`, `p = PF90113`, `b = PF90114`. Plain binary32 products.
pub fn stray_hair_degrees(d: f32, p: f32, b: f32) -> StrayHairDegrees {
    StrayHairDegrees {
        person_deg: d * p,
        background_deg: b * d,
    }
}

/// Convenience wrapper returning the degrees and the gate together.
pub fn stray_hair_gate(d: f32, p: f32, b: f32) -> (StrayHairDegrees, bool) {
    (stray_hair_degrees(d, p, b), stray_hair_active(d))
}

// ===========================================================================
// 5. PF90085 — face skin-tone unify piecewise remap
// ===========================================================================

/// Gate on the final degree, single precision (SG24: `0x8BEE250`, `0x7B13475`).
pub const TONE_UNIFY_GATE: f32 = 0.001;

/// Input knots `X` at `0x930C7B0` (SG24).
pub const TONE_UNIFY_KNOTS_X: [f32; 5] = [0.0, 4.5, 5.0, 8.0, 10.0];

/// Output knots `Y` at `0x930C79C` (SG24).
pub const TONE_UNIFY_KNOTS_Y: [f32; 5] = [0.0, 0.0, 0.1, 1.0, 1.0];

/// Piecewise linear remap exactly as the vendor function `0x38DE290` computes it:
/// locate the segment `[X[i], X[i+1]]` containing `x`, then
/// `t = (x - X[i]) / (X[i+1] - X[i])`, `y = Y[i] + t * (Y[i+1] - Y[i])`.
///
/// Segment selection clamps to the first/last segment outside the knot range.
/// With the vendor knot table `X = [0, 4.5, 5, 8, 10]` and `Y = [0, 0, 0.1, 1, 1]`
/// this is indistinguishable from endpoint clamping: both extreme segments are
/// flat, so no out-of-range convention is invented.
///
/// Panics when the tables are empty, shorter than two entries or of unequal
/// length; the module only ever calls it with the recovered five-knot table.
pub fn piecewise_remap(x: f32, knots_x: &[f32], knots_y: &[f32]) -> f32 {
    assert_eq!(
        knots_x.len(),
        knots_y.len(),
        "piecewise_remap: knot tables must have equal length"
    );
    assert!(
        knots_x.len() >= 2,
        "piecewise_remap: at least two knots are required"
    );

    let last_segment = knots_x.len() - 2;
    let mut segment = last_segment;
    for index in 0..=last_segment {
        if x < knots_x[index + 1] {
            segment = index;
            break;
        }
    }

    let x0 = knots_x[segment];
    let x1 = knots_x[segment + 1];
    let y0 = knots_y[segment];
    let y1 = knots_y[segment + 1];
    let t = (x - x0) / (x1 - x0);
    y0 + t * (y1 - y0)
}

/// `piecewise_remap(v, X, Y) * PF90085` (SG24: `0x7B12E10`, `mulss 0x7B13470`).
pub fn tone_unify_degree(v: f32, pf90085: f32) -> f32 {
    piecewise_remap(v, &TONE_UNIFY_KNOTS_X, &TONE_UNIFY_KNOTS_Y) * pf90085
}

/// `true` when the final degree passes the strict `> 0.001f` gate (SG24).
pub fn tone_unify_active(degree: f32) -> bool {
    degree > TONE_UNIFY_GATE
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(actual: f32, expected: f32) -> bool {
        (actual - expected).abs() <= 1.0e-5
    }

    fn assert_approx(actual: f32, expected: f32) {
        assert!(
            approx(actual, expected),
            "expected {expected}, got {actual}"
        );
    }

    fn degree_uniforms(degree: f32) -> SpotlessUniforms {
        SpotlessUniforms {
            degree,
            ..SpotlessUniforms::vendor_default()
        }
    }

    // --- PF0 Spotless ------------------------------------------------------

    #[test]
    fn spotless_vendor_defaults_match_the_recovered_constants() {
        let uniforms = SpotlessUniforms::vendor_default();
        assert_eq!(uniforms.dist_factor, 28.87);
        assert_eq!(uniforms.dist_threshold, 0.5);
        assert_eq!(uniforms.dark_min, 0.0);
        assert_eq!(uniforms.dark_max, 0.3);
        assert_eq!(uniforms.white_factor, 0.7);
        assert_eq!(uniforms.degree, 0.0);

        // The constants themselves are the documented values.
        assert_eq!(SPOTLESS_DIST_FACTOR, 28.87);
        assert_eq!(SPOTLESS_DIST_THRESHOLD, 0.5);
        assert_eq!(SPOTLESS_DARK_MIN, 0.0);
        assert_eq!(SPOTLESS_DARK_MAX, 0.3);
        assert_eq!(SPOTLESS_WHITE_FACTOR, 0.7);
        assert_eq!(SPOTLESS_DEGREE_DEFAULT, 0.0);
    }

    #[test]
    fn spotless_modes_report_program_ids_and_samplers() {
        assert_eq!(SpotlessMode::Normal.program_id(), 0xD1);
        assert_eq!(SpotlessMode::WithSkinMask.program_id(), 0xD2);
        assert_eq!(SPOTLESS_PROGRAM_NORMAL, 0xD1);
        assert_eq!(SPOTLESS_PROGRAM_WITH_SKIN_MASK, 0xD2);

        assert!(!SpotlessMode::Normal.uses_skin_mask());
        assert!(SpotlessMode::WithSkinMask.uses_skin_mask());

        assert_eq!(
            SpotlessMode::Normal.samplers(),
            &["s_texture", "s_blur_texture", "s_facemask_texture"][..]
        );
        assert_eq!(
            SpotlessMode::WithSkinMask.samplers(),
            &[
                "s_texture",
                "s_blur_texture",
                "s_facemask_texture",
                "s_skinmask_texture"
            ][..]
        );
    }

    #[test]
    fn spotless_static_default_degree_is_an_exact_identity() {
        let src = [0.25f32, 0.25, 0.25, 0.5];
        let blur = [0.9f32, 0.9, 0.9];
        let uniforms = SpotlessUniforms::vendor_default();
        for mode in [SpotlessMode::Normal, SpotlessMode::WithSkinMask] {
            let out = spotless_fragment(mode, src, blur, 1.0, 1.0, &uniforms);
            assert_eq!(out, src, "degree 0.0 must leave the pixel untouched");
        }
    }

    #[test]
    fn spotless_face_mask_gate_is_strict() {
        let uniforms = degree_uniforms(1.0);
        let src = [0.25f32, 0.25, 0.25, 0.5];
        let blur = [0.5f32, 0.5, 0.5];

        assert_eq!(
            spotless_fragment(SpotlessMode::WithSkinMask, src, blur, 0.0, 1.0, &uniforms),
            src,
            "face_mask == 0.0 must not enter the fragment body"
        );
        assert_eq!(
            spotless_fragment(SpotlessMode::WithSkinMask, src, blur, -1.0, 1.0, &uniforms),
            src,
            "negative face_mask must not enter the fragment body"
        );

        let out = spotless_fragment(SpotlessMode::WithSkinMask, src, blur, 1e-6, 1.0, &uniforms);
        assert_ne!(out, src, "any positive face_mask enters the body");
    }

    #[test]
    fn spotless_dark_branch_adds_the_clamped_blur_difference() {
        let uniforms = degree_uniforms(1.0);
        let src = [0.25f32, 0.25, 0.25, 0.5];
        let blur = [0.5f32, 0.5, 0.5];
        let out = spotless_fragment(SpotlessMode::WithSkinMask, src, blur, 1.0, 1.0, &uniforms);
        assert_eq!([out[0], out[1], out[2]], [0.5, 0.5, 0.5]);
        assert_eq!(out[3], 0.5, "alpha is copied from src");

        // dark_max caps the per-channel difference at 0.3.
        let bright_blur = [0.9f32, 0.9, 0.9];
        let capped =
            spotless_fragment(SpotlessMode::WithSkinMask, src, bright_blur, 1.0, 1.0, &uniforms);
        assert_approx(capped[0], 0.55); // 0.25 + 0.3

        // Negative blur-src is clamped to dark_min by `max(blur - src, 0)` for
        // the distance and by `clamp` for the difference itself.
        let mixed_blur = [0.5f32, 0.1, 0.5];
        let mixed =
            spotless_fragment(SpotlessMode::WithSkinMask, src, mixed_blur, 1.0, 1.0, &uniforms);
        assert_eq!([mixed[0], mixed[1], mixed[2]], [0.5, 0.25, 0.5]);
    }

    #[test]
    fn spotless_light_branch_pulls_towards_the_blur() {
        let uniforms = SpotlessUniforms {
            dist_factor: 2.0,
            dist_threshold: 0.1,
            dark_min: 0.0,
            dark_max: 1.0,
            white_factor: 0.5,
            degree: 1.0,
        };
        let src = [0.5f32, 0.5, 0.5, 0.25];
        let blur = [0.25f32, 0.25, 0.25];
        let out = spotless_fragment(SpotlessMode::WithSkinMask, src, blur, 1.0, 1.0, &uniforms);
        // ref = 0.25 + (0.5 - 0.25) * 0.5 = 0.375
        assert_eq!([out[0], out[1], out[2]], [0.375, 0.375, 0.375]);
        assert_eq!(out[3], 0.25, "alpha is copied from src");
    }

    #[test]
    fn spotless_distance_threshold_is_strict() {
        let uniforms = SpotlessUniforms {
            dist_factor: 2.0,
            dist_threshold: 0.25,
            dark_min: 0.0,
            dark_max: 1.0,
            white_factor: 0.5,
            degree: 1.0,
        };
        let src = [0.5f32, 0.5, 0.5, 1.0];
        let blur = [0.625f32, 0.5, 0.5];
        // dark distance == 0.125 * 2.0 == threshold, so `>` does not fire; the
        // light distance is zero. The pixel is unchanged.
        let out = spotless_fragment(SpotlessMode::WithSkinMask, src, blur, 1.0, 1.0, &uniforms);
        assert_eq!(out, src);
    }

    #[test]
    fn spotless_factor_is_clamped_to_unit_interval() {
        let src = [0.25f32, 0.25, 0.25, 0.5];
        let blur = [0.5f32, 0.5, 0.5];

        let full = degree_uniforms(1.0);
        let expected = spotless_fragment(
            SpotlessMode::WithSkinMask,
            src,
            blur,
            1.0,
            1.0,
            &full,
        );

        // face * skin * degree = 2 * 2 * 4 = 16, clamped to 1.
        let over = SpotlessUniforms {
            degree: 4.0,
            ..SpotlessUniforms::vendor_default()
        };
        let clamped = spotless_fragment(
            SpotlessMode::WithSkinMask,
            src,
            blur,
            2.0,
            2.0,
            &over,
        );
        assert_eq!(clamped, expected);

        let negative = degree_uniforms(-1.0);
        assert_eq!(
            spotless_fragment(SpotlessMode::WithSkinMask, src, blur, 1.0, 1.0, &negative),
            src
        );
    }

    #[test]
    fn spotless_normal_mode_uses_the_inline_skin_heuristic() {
        let uniforms = degree_uniforms(1.0);
        let blur = [0.25f32, 0.25, 0.25];

        // Green: hue 1/3 sits inside the 0.18..0.89 hard band → skin 0.
        let green = [0.0f32, 0.5, 0.0, 1.0];
        assert_eq!(spotless_skin_value([green[0], green[1], green[2]]), 0.0);
        let out = spotless_fragment(SpotlessMode::Normal, green, blur, 1.0, 1.0, &uniforms);
        assert_eq!(out, green, "Normal ignores the supplied skin plane");

        // Red: hue 0, value 1 → skin 1, so the light branch pulls towards blur.
        let red = [1.0f32, 0.0, 0.0, 1.0];
        let red_blur = [0.5f32, 0.0, 0.0];
        let red_out = spotless_fragment(SpotlessMode::Normal, red, red_blur, 1.0, 0.0, &uniforms);
        assert_approx(red_out[0], 0.85); // 0.5 + 0.5 * 0.7
        assert_eq!(red_out[1], 0.0);
    }

    #[test]
    fn spotless_with_skin_mask_mode_reads_the_plane_not_the_heuristic() {
        let uniforms = degree_uniforms(1.0);
        let blur = [0.25f32, 0.5, 0.25];

        // Green would be rejected by the heuristic, but the plane says 1.0.
        let green = [0.0f32, 0.5, 0.0, 1.0];
        let out = spotless_fragment(
            SpotlessMode::WithSkinMask,
            green,
            blur,
            1.0,
            1.0,
            &uniforms,
        );
        assert_ne!(out, green);

        // Red would be accepted by the heuristic, but the plane says 0.0.
        let red = [1.0f32, 0.0, 0.0, 1.0];
        let red_blur = [0.5f32, 0.0, 0.0];
        let held = spotless_fragment(
            SpotlessMode::WithSkinMask,
            red,
            red_blur,
            1.0,
            0.0,
            &uniforms,
        );
        assert_eq!(held, red);
    }

    #[test]
    fn spotless_alpha_is_copied_and_no_branch_is_identity() {
        let uniforms = degree_uniforms(1.0);
        let src = [0.5f32, 0.5, 0.5, 0.123];
        let blur = [0.5f32, 0.5, 0.5];
        let out = spotless_fragment(SpotlessMode::WithSkinMask, src, blur, 1.0, 1.0, &uniforms);
        assert_eq!(out, src);
        assert_eq!(out[3], 0.123);

        let colored = [0.5f32, 0.5, 0.5, 0.75];
        let changed_blur = [0.25f32, 0.25, 0.25];
        let changed =
            spotless_fragment(SpotlessMode::WithSkinMask, colored, changed_blur, 1.0, 1.0, &uniforms);
        assert_eq!(changed[3], 0.75, "alpha never comes from the blur plane");
    }

    #[test]
    fn spotless_skin_heuristic_bands_and_ramps() {
        // Hard band, inclusive on both ends; dark pixels are never skin.
        assert_eq!(spotless_skin_value_from_hue(0.18, 0.9), 0.0);
        assert_eq!(spotless_skin_value_from_hue(0.89, 0.9), 0.0);
        assert_eq!(spotless_skin_value_from_hue(0.5, 0.9), 0.0);
        assert_eq!(spotless_skin_value_from_hue(0.5, 0.2), 0.0);
        assert_eq!(spotless_skin_value_from_hue(0.5, 0.0), 0.0);

        // Low-hue ramp: 0.16 and 0.18 are strict boundaries.
        assert_eq!(spotless_skin_value_from_hue(0.16, 0.9), 1.0);
        assert_approx(spotless_skin_value_from_hue(0.17, 0.9), 0.5);
        assert_approx(spotless_skin_value_from_hue(0.179, 0.9), 0.05);

        // High-hue ramp: 0.89 and 0.91 are strict boundaries.
        assert_eq!(spotless_skin_value_from_hue(0.91, 0.9), 1.0);
        assert_approx(spotless_skin_value_from_hue(0.90, 0.9), 0.5);
        assert_approx(spotless_skin_value_from_hue(0.901, 0.9), 0.55);

        // Value ramp only applies outside the hard hue band and for hue < 0.3.
        assert_approx(spotless_skin_value_from_hue(0.1, 0.25), 0.5);
        assert_approx(spotless_skin_value_from_hue(0.1, 0.21), 0.1);
        assert_eq!(spotless_skin_value_from_hue(0.95, 0.25), 1.0);
        assert_eq!(spotless_skin_value_from_hue(0.2, 0.25), 0.0);
        assert_eq!(spotless_skin_value_from_hue(0.35, 0.25), 0.0);
    }

    #[test]
    fn spotless_hsv_matches_the_vendor_formula() {
        let cases = [
            ([1.0f32, 0.0, 0.0], 0.0f32, 1.0f32),
            ([0.0, 1.0, 0.0], 1.0 / 3.0, 1.0),
            ([0.0, 0.0, 1.0], 2.0 / 3.0, 1.0),
            ([0.5, 0.5, 0.5], 0.0, 0.5),
            ([0.0, 0.0, 0.0], 0.0, 0.0),
        ];
        for (rgb, expected_hue, expected_value) in cases {
            let hsv = spotless_rgb2hsv(rgb);
            assert_approx(hsv[0], expected_hue);
            assert_eq!(hsv[1], 0.0, "vendor forces saturation to 0");
            assert_approx(hsv[2], expected_value);
        }
    }

    // --- PF90034 / PF30188 / PF30189 --------------------------------------

    #[test]
    fn double_chin_weights_are_plain_binary32_products() {
        let weights = double_chin_weights(0.5, 0.4, 0.3);
        assert_eq!(weights.w_geometry, 0.5f32 * 0.4f32);
        assert_eq!(weights.w_shadow, 0.5f32 * 0.3f32);

        // No `/100` and no `0..1` normalization: large inputs stay large.
        let large = double_chin_weights(50.0, 2.0, 3.0);
        assert_eq!(large.w_geometry, 100.0);
        assert_eq!(large.w_shadow, 150.0);

        assert_eq!(double_chin_weight_or_default(None), 1.0);
        assert_eq!(double_chin_weight_or_default(Some(0.25)), 0.25);
        assert_eq!(DOUBLE_CHIN_WEIGHT_DEFAULT, 1.0);
    }

    #[test]
    fn double_chin_gate_is_a_binary64_disjunction() {
        assert_eq!(DOUBLE_CHIN_GATE, 1.0e-4);

        // Both below the gate → skipped.
        assert!(!double_chin_passes_gate(0.0, 0.0));
        assert!(!double_chin_passes_gate(-1.0, -1.0));
        assert!(!double_chin_passes_gate(f32::NEG_INFINITY, f32::NEG_INFINITY));

        // Either one at or above the gate → runs.
        assert!(double_chin_passes_gate(1.0e-3, 0.0));
        assert!(double_chin_passes_gate(0.0, 1.0e-3));

        // `1e-4f32` widens to `9.9999997473787516e-5` in binary64, which is
        // BELOW the binary64 gate, so the exact float literal does not pass; the
        // next representable f32 does. This is the vendor's binary64 comparison,
        // not an f32 one.
        assert!(!double_chin_passes_gate(1.0e-4, 0.0));
        let just_above = f32::from_bits(1.0e-4f32.to_bits() + 1);
        assert!(double_chin_passes_gate(just_above, 0.0));

        // Machine semantics: NaN is unordered, both `<` tests are false, so the
        // stage is not skipped.
        assert!(double_chin_passes_gate(f32::NAN, 0.0));
        assert!(double_chin_passes_gate(0.0, f32::NAN));

        // The struct method agrees with the free function.
        let weights = double_chin_weights(0.5, 0.4, 0.3);
        assert_eq!(
            weights.passes_gate(),
            double_chin_passes_gate(weights.w_shadow, weights.w_geometry)
        );
    }

    #[test]
    fn double_chin_masks_follow_the_six_channel_mapping() {
        assert_eq!(DOUBLE_CHIN_CHANNELS, [10, 11, 14, 6, 8, 9]);

        // Order is [ch10, ch11, ch14, ch6, ch8, ch9].
        let masks = double_chin_masks([10, 20, 30, 40, 50, 60]);
        assert_eq!(masks.a, 30, "A = ch10 + ch11 (10 + 20)");
        assert_eq!(masks.b, 40, "B = ch6");
        assert_eq!(masks.c, 30, "C = ch14");
        assert_eq!(masks.d, 110, "D = ch8 + ch9 (50 + 60)");

        // Saturating adds for A and D only.
        let saturated = double_chin_masks([200, 100, 7, 9, 250, 20]);
        assert_eq!(saturated.a, 255, "A saturates at 255");
        assert_eq!(saturated.b, 9, "B is copied, never saturated");
        assert_eq!(saturated.c, 7, "C is copied, never saturated");
        assert_eq!(saturated.d, 255, "D saturates at 255");
    }

    // --- PF30209 -----------------------------------------------------------

    #[test]
    fn jawline_activation_is_any_strictly_positive_value() {
        assert!(!jawline_active(&[]));
        assert!(!jawline_active(&[0.0]));
        assert!(!jawline_active(&[0.0, -0.0, -1.0]));
        assert!(!jawline_active(&[f32::NAN]));
        assert!(jawline_active(&[0.0, 1.0e-30]));
        assert!(jawline_active(&[-5.0, 0.001]));
        assert!(jawline_active(&[f32::INFINITY]));
    }

    #[test]
    fn jawline_window_extracts_the_fixed_contour_and_centre() {
        assert_eq!(JAWLINE_WINDOW_START, 109);
        assert_eq!(JAWLINE_WINDOW_END, 141);
        assert_eq!(JAWLINE_WINDOW_LEN, 33);
        assert_eq!(JAWLINE_CENTER_INDEX, 54);
        assert_eq!(JAWLINE_LANDMARK_COUNT, 194);

        let landmarks: Vec<[f32; 2]> = (0..JAWLINE_LANDMARK_COUNT)
            .map(|i| [i as f32, i as f32 + 0.5])
            .collect();
        let window = jawline_window(&landmarks).expect("194 landmarks is the vendor length");

        assert_eq!(window.center, [54.0, 54.5]);
        assert_eq!(window.contour.len(), JAWLINE_WINDOW_LEN);
        assert_eq!(window.contour[0], [109.0, 109.5]);
        assert_eq!(window.contour[32], [141.0, 141.5]);
        assert_eq!(window.contour[19], [128.0, 128.5]);

        // One landmark short of index 141 → no window.
        assert!(jawline_window(&landmarks[..JAWLINE_WINDOW_END]).is_none());
        // Exactly 142 entries (0..=141) is enough.
        assert!(jawline_window(&landmarks[..=JAWLINE_WINDOW_END]).is_some());
    }

    // --- PF90088 / PF90113 / PF90114 --------------------------------------

    #[test]
    fn stray_hair_products_and_gate() {
        assert_eq!(STRAY_HAIR_GATE, 0.001);

        let degrees = stray_hair_degrees(0.5, 0.4, 0.2);
        assert_eq!(degrees.person_deg, 0.2); // 0.5 * 0.4
        assert_eq!(degrees.background_deg, 0.1); // 0.2 * 0.5
        assert!(degrees.active_from_source(0.5));

        // The gate is on `d` (PF90088) alone, strict `> 0.001` in binary64.
        assert!(!stray_hair_active(0.0));
        assert!(!stray_hair_active(0.0005));
        assert!(!stray_hair_active(-1.0));
        assert!(stray_hair_active(0.002));

        // `0.001f32` widens to `1.0000000474974513e-3`, slightly ABOVE the
        // binary64 gate, so it activates; the next f32 down does not.
        assert!(stray_hair_active(0.001f32));
        let just_below = f32::from_bits(0.001f32.to_bits() - 1);
        assert!(!stray_hair_active(just_below));

        // Products alone do not gate: d below 0.001 stays inactive even when
        // the region sliders are large, because the gate is on `d` alone.
        let (gated, active) = stray_hair_gate(0.0005, 100.0, 100.0);
        assert_approx(gated.person_deg, 0.05);
        assert_approx(gated.background_deg, 0.05);
        assert!(!active);
    }

    // --- PF90085 -----------------------------------------------------------

    #[test]
    fn tone_unify_piecewise_remap_boundaries() {
        assert_eq!(TONE_UNIFY_KNOTS_X, [0.0, 4.5, 5.0, 8.0, 10.0]);
        assert_eq!(TONE_UNIFY_KNOTS_Y, [0.0, 0.0, 0.1, 1.0, 1.0]);

        let cases = [
            (0.0f32, 0.0f32),
            (4.5, 0.0),
            (4.75, 0.05), // midpoint of [4.5, 5.0]
            (5.0, 0.1),
            (6.5, 0.55), // midpoint of [5.0, 8.0]
            (8.0, 1.0),
            (10.0, 1.0),
        ];
        for (v, expected) in cases {
            let actual = piecewise_remap(v, &TONE_UNIFY_KNOTS_X, &TONE_UNIFY_KNOTS_Y);
            assert_approx(actual, expected);
            assert_approx(tone_unify_degree(v, 1.0), expected);
        }

        // Outside the knot range both extreme segments are flat.
        assert_eq!(
            piecewise_remap(-5.0, &TONE_UNIFY_KNOTS_X, &TONE_UNIFY_KNOTS_Y),
            0.0
        );
        assert_eq!(
            piecewise_remap(100.0, &TONE_UNIFY_KNOTS_X, &TONE_UNIFY_KNOTS_Y),
            1.0
        );
    }

    #[test]
    fn tone_unify_gate_is_strict_single_precision() {
        assert_eq!(TONE_UNIFY_GATE, 0.001);

        // v = 10.0 → remap 1.0, so the degree equals the slider value.
        assert!(!tone_unify_active(tone_unify_degree(10.0, 0.0)));
        assert!(!tone_unify_active(tone_unify_degree(10.0, 0.0005)));
        assert!(!tone_unify_active(tone_unify_degree(10.0, 0.001)));
        assert!(tone_unify_active(tone_unify_degree(10.0, 0.002)));
        assert!(tone_unify_active(1.0e-3 + 1.0e-6));

        // v = 4.5 maps to 0 regardless of the slider.
        assert_eq!(tone_unify_degree(4.5, 100.0), 0.0);
        assert!(!tone_unify_active(0.0));

        // v = 6.0 is one third into [5, 8] → remap = 0.4; times 0.4 = 0.16.
        assert_approx(tone_unify_degree(6.0, 0.4), 0.16);
    }

    #[test]
    #[should_panic(expected = "at least two knots")]
    fn piecewise_remap_rejects_too_few_knots() {
        let _ = piecewise_remap(1.0, &[0.0], &[0.0]);
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn piecewise_remap_rejects_unequal_knot_tables() {
        let _ = piecewise_remap(1.0, &[0.0, 1.0], &[0.0]);
    }
}
