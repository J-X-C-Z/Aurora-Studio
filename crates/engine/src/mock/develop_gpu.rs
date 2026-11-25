//! Optional CUDA path for the per-pixel develop point operations.
//!
//! Phase 1 owns only the point maths of `mock/process.rs`:
//!
//! `linear_edits` -> C1 camera colour stage (or sRGB encode) ->
//! `develop_display` (tone, levels, five curves, saturation, colour balance,
//! colour editor, B&W) -> `output_banding` -> gamma.
//!
//! Sampling, geometry/keystone/lens warp, clarity (a neighbourhood operator),
//! layer masks/compositing and every other spatial stage stay on the CPU. The
//! caller hands this module the already-sampled linear RGB plus a `valid` flag,
//! exactly like `SourceSample`.
//!
//! `AURORA_DEVELOP_GPU=0` keeps the CPU path; on Windows the module attempts to
//! open the CUDA module by default. A missing driver/PTX, a short frame or any
//! launch failure returns `false` so the caller runs the existing CPU code.
//!
//! Phase 2 must keep these on the CPU and `log::info!` the reason:
//!
//! - local layers/masks: their per-layer stacks and mask weights are mixed per
//!   pixel in `local()`, between the global develop plane and the output;
//! - clarity: its pre-curve is inserted between `develop_pre_curve` and
//!   `develop_display` and is a neighbourhood operator;
//! - the node graph, when present, may reorder or repeat point ops;
//! - LCC Uniform Light (and the manual vignette when a plane is not supplied):
//!   the gain needs a source-coordinate map, so the caller must precompute the
//!   output-pixel gain plane and pass it here;
//! - every spatial/post stage (sharpening, NR, moire, grain, spots, retouch,
//!   portrait producers) is outside this module and unaffected.
//!
//! `fallback_reason` encodes the first four checks; `process.rs` calls it once
//! per render and logs the first refusal.

#![allow(dead_code)] // phase 2 wires this into `process.rs`

use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use aurora_portrait::cuda::{CudaError, DeviceBuffer, Nvidia, PinnedBuffer, Stream};

use super::gpu_runtime::{self, VramLease};

use crate::*;

const PTX: &str = include_str!("../../native/cuda/develop_gpu.ptx");
const BLOCK: (u32, u32) = (16, 16);
const PREWARM: (usize, usize) = (1400, 933);
/// Reclaim priority: develop is the interactive main path, so its retained
/// frame is evicted last.
const RECLAIM_PRIORITY: u8 = 30;

/// Device bytes this module keeps resident no matter the frame size: the
/// parameter block, the film/gamma LUTs, the C1 colour tables, the colour-editor
/// skin table and the output-banding table. Mirrors `Pool::open` exactly.
fn fixed_bytes() -> usize {
    P_SIZE * 4 + (1280 + GAMMA_SIZE) * 4 + (FILM_MAX * 3 * 4 + CLUT_NODES * 3 * 2) + E_SKIN_BASE * 4
        + (256 * 4 + 512)
}

fn budget_error(failure: gpu_runtime::Failure) -> CudaError {
    log::warn!("develop_gpu: {failure:?}; over the shared VRAM budget");
    CudaError::Unavailable("vram budget")
}

const FILM_MAX: usize = 16;
const CLUT_N: usize = 49;
const CLUT_NODES: usize = CLUT_N * CLUT_N * CLUT_N;
const GAMMA_SIZE: usize = 8192;
const GAMMA_RANGE: f32 = 4.0;

// Parameter layout, byte-for-byte in sync with develop_gpu.cu.
const P_W: usize = 0;
const P_H: usize = 1;
const P_RX: usize = 2;
const P_RY: usize = 3;
const P_RW: usize = 4;
const P_RH: usize = 5;
const P_FINE: usize = 6;
const P_BYPASS: usize = 7;
const P_HAS_GAIN: usize = 8;
const P_WB: usize = 9;
const P_EXPOSURE: usize = 12;
const P_C1: usize = 13;
const P_PREMAT: usize = 14;
const P_FILM_N: usize = 23;
const P_BANDING: usize = 24;
const P_BG_GAIN: usize = 25;
const P_BG_LIMIT: usize = 26;
const P_BG_STRENGTH: usize = 27;
const P_BG_FW: usize = 28;
const P_BG_FH: usize = 29;
const P_DEHAZE_OMEGA: usize = 30;
const P_DEHAZE_C: usize = 31;
const P_DEHAZE_LO: usize = 34;
const P_DEHAZE_HI: usize = 37;
const P_HDR: usize = 40;
const P_BRIGHT_GAMMA: usize = 44;
const P_CONTRAST: usize = 45;
const P_LEVELS: usize = 46;
const P_LV_LO: usize = 47;
const P_LV_SPAN: usize = 50;
const P_LV_INV: usize = 53;
const P_LV_MLO: usize = 56;
const P_LV_MSPAN: usize = 59;
const P_LV_MINV: usize = 62;
const P_LV_OUT_GAIN: usize = 65;
const P_LV_OUT_SHADOW: usize = 66;
const P_CURVE: usize = 67;
const P_SAT: usize = 72;
const P_CB: usize = 73;
const P_CB_OFF: usize = 74;
const P_CBG: usize = 86;
const P_CBG_GAIN: usize = 87;
const P_BW: usize = 99;
const P_BW_WMIN: usize = 100;
const P_BW_WMAX: usize = 103;
const P_BW_SHIFT: usize = 106;
const P_BW_USE_LUMA: usize = 107;
const P_BW_TINT_LIGHT: usize = 108;
const P_BW_TINT_DARK: usize = 111;
const P_BW_DARK_HI: usize = 114;
const P_BW_LIGHT_LO: usize = 115;
const P_CE: usize = 116;
const P_CE_EXTRA: usize = 117;
const P_N_SKIN: usize = 118;
const P_N_SHIFT: usize = 119;
const P_BW_SPLIT: usize = 120;
const P_BANDING_OUTER: usize = 121;
const P_SIZE: usize = 122;

const E_GLOBAL_ENABLED: usize = 0;
const E_GLOBAL_BAND: usize = 1;
const E_GLOBAL_SHIFT: usize = 13;
const E_BASIC_EXTRA: usize = 16;
const E_BASIC: usize = 17;
const E_BASIC_STRIDE: usize = 16;
const E_SKIN_BASE: usize = 145;
const E_SKIN_STRIDE: usize = 24;
const E_SHIFT_STRIDE: usize = 22;

/// One sampled source pixel, mirroring `process::SourceSample`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sample {
    pub rgb: [f32; 3],
    pub valid: bool,
}

/// Per-channel levels, resolved on the host in the CPU evaluation order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LevelsParams {
    pub lo: [[f32; 3]; 2],
    pub span: [[f32; 3]; 2],
    pub inv_gamma: [[f32; 3]; 2],
    pub out_gain: f32,
    pub out_shadow: f32,
}

impl LevelsParams {
    fn from_levels(levels: &Levels) -> Self {
        let channel = |c: &crate::LevelsChannel| {
            (
                c.input_black,
                (c.input_white - c.input_black).max(1e-3),
                1.0 / c.gamma.clamp(0.1, 10.0),
            )
        };
        let per = [&levels.red, &levels.green, &levels.blue];
        let master = &levels.rgb;
        let mut out = Self {
            lo: [[0.0; 3]; 2],
            span: [[0.0; 3]; 2],
            inv_gamma: [[0.0; 3]; 2],
            out_gain: master.output_white - master.output_black,
            out_shadow: master.output_black,
        };
        for c in 0..3 {
            let (lo, span, inv) = channel(per[c]);
            out.lo[0][c] = lo;
            out.span[0][c] = span;
            out.inv_gamma[0][c] = inv;
            let (lo, span, inv) = channel(master);
            out.lo[1][c] = lo;
            out.span[1][c] = span;
            out.inv_gamma[1][c] = inv;
        }
        out
    }
}

/// Flattened colour-editor descriptors.
#[derive(Clone, Debug, Default)]
pub(crate) struct EditorParams {
    pub floats: Vec<f32>,
    pub n_skin: i32,
    pub n_shift: i32,
}

impl EditorParams {
    fn from_editor(ce: &ColorEditor) -> Self {
        let mut out = vec![0.0f32; E_SKIN_BASE];
        out[E_GLOBAL_ENABLED] = if ce.global.enabled { 1.0 } else { 0.0 };
        write_band(&mut out[E_GLOBAL_BAND..E_GLOBAL_BAND + 12], &ce.global);
        out[E_GLOBAL_SHIFT] = ce.global_shift.hue;
        out[E_GLOBAL_SHIFT + 1] = ce.global_shift.saturation;
        out[E_GLOBAL_SHIFT + 2] = ce.global_shift.lightness;
        let extra = ce.global.smoothness.max(0.0) * 0.5;
        out[E_BASIC_EXTRA] = extra;

        for i in 0..8 {
            let at = E_BASIC + i * E_BASIC_STRIDE;
            let band = &ce.bands[i];
            let shift = ce.basic[i];
            write_band(&mut out[at..at + 12], band);
            out[at + 12] = shift.hue;
            out[at + 13] = shift.saturation;
            out[at + 14] = shift.lightness;
            let skip = shift == HslShift::default() || !band.enabled;
            out[at + 15] = if skip { 1.0 } else { 0.0 };
        }

        let find_band = |id: u64| ce.advanced_bands.iter().find(|(key, _)| *key == id).map(|(_, b)| b);

        let mut skin = 0i32;
        for range in &ce.skin {
            let mut entry = vec![0.0f32; E_SKIN_STRIDE];
            if let Some(band) = find_band(range.id) {
                if band.enabled {
                    entry[0] = 1.0;
                    write_band(&mut entry[1..13], band);
                    entry[13..16].copy_from_slice(&band.homogeneity);
                }
            } else if let Some(u) = &range.uniformity {
                entry[0] = 2.0;
                entry[16] = range.hue;
                entry[17] = range.width;
                entry[18] = range.smoothness;
                entry[19] = range.min_saturation;
                entry[20] = u.hue;
                entry[21] = u.saturation;
                entry[22] = u.lightness;
            }
            out.extend_from_slice(&entry);
            skin += 1;
        }

        let mut shift_count = 0i32;
        let zero = 0.0f32;
        for (range, feather) in ce
            .advanced
            .iter()
            .map(|r| (r, zero))
            .chain(ce.skin.iter().map(|r| (r, zero)))
            .chain(ce.max_smooth.iter().map(|r| (r, extra)))
        {
            let mut entry = vec![0.0f32; E_SHIFT_STRIDE];
            entry[1] = feather;
            match find_band(range.id) {
                Some(band) if band.enabled => {
                    entry[0] = 0.0;
                    write_band(&mut entry[2..14], band);
                }
                Some(_) => entry[0] = 1.0,
                None => {
                    entry[0] = 2.0;
                    entry[17] = range.hue;
                    entry[18] = range.width;
                    entry[19] = range.smoothness;
                    entry[20] = range.min_saturation;
                }
            }
            entry[14] = range.shift.hue;
            entry[15] = range.shift.saturation;
            entry[16] = range.shift.lightness;
            out.extend_from_slice(&entry);
            shift_count += 1;
        }

        Self { floats: out, n_skin: skin, n_shift: shift_count }
    }
}

fn write_band(dst: &mut [f32], band: &ColorBand) {
    dst[0] = band.center_hue;
    dst[1] = band.center_saturation;
    dst[2] = band.center_value;
    dst[3] = band.hue_low;
    dst[4] = band.hue_high;
    dst[5] = band.sat_low;
    dst[6] = band.sat_high;
    dst[7] = band.value_low;
    dst[8] = band.value_high;
    dst[9] = band.smoothness;
    dst[10] = if band.inverted { 1.0 } else { 0.0 };
    dst[11] = if band.enabled { 1.0 } else { 0.0 };
}

/// B&W constants resolved on the host so the Q10 integer path is byte-exact.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BwParams {
    pub w_min: [i32; 3],
    pub w_max: [i32; 3],
    pub shift: u32,
    pub use_luma: bool,
    pub split: bool,
    pub tint_light: [f32; 3],
    pub tint_dark: [f32; 3],
    pub dark_hi: f32,
    pub light_lo: f32,
}

impl BwParams {
    fn from_black_white(bw: &BlackWhite) -> Self {
        let (w_min, w_max) = bw_weights(&bw.sensitivity);
        let shift = bw_shift(&w_min, &w_max);
        let use_luma = bw.sensitivity.iter().all(|&v| v == 0.0);
        let split = bw.highlight_tone.saturation > 0.0 || bw.shadow_tone.saturation > 0.0;
        let light = bw_tint(bw.highlight_tone.hue, bw.highlight_tone.saturation);
        let dark = bw_tint(bw.shadow_tone.hue, bw.shadow_tone.saturation);
        let dark_hi = light[0] * BW_C1_LUMA[0] + light[1] * BW_C1_LUMA[1] + light[2] * BW_C1_LUMA[2];
        let light_lo = dark[0].max(dark[1]).max(dark[2]);
        Self { w_min, w_max, shift, use_luma, split, tint_light: light, tint_dark: dark, dark_hi, light_lo }
    }
}

const BW_C1_LUMA: [f32; 3] = [0.3328, 0.6121, 0.0551];

fn bw_weights(sensitivity: &[f32; 6]) -> ([i32; 3], [i32; 3]) {
    const BASE: [f64; 3] = [340.7872, 626.7904, 56.4224];
    let luma = [
        BASE[0] + 10.24 * sensitivity[0] as f64,
        BASE[1] + 10.24 * sensitivity[2] as f64,
        BASE[2] + 10.24 * sensitivity[4] as f64,
    ];
    let comp = [
        (1024.0 - BASE[0]) + 10.24 * sensitivity[3] as f64,
        (1024.0 - BASE[1]) + 10.24 * sensitivity[5] as f64,
        (1024.0 - BASE[2]) + 10.24 * sensitivity[1] as f64,
    ];
    let q = |w: [f64; 3]| w.map(|v| v as i32);
    (q(comp), q(luma))
}

fn bw_shift(w_min: &[i32; 3], w_max: &[i32; 3]) -> u32 {
    let mut m = w_min.iter().chain(w_max).copied().max().unwrap_or(0).max(0);
    let mut shift = 0;
    while m > 1024 {
        m >>= 1;
        shift += 1;
    }
    shift
}

fn bw_tint(hue: f32, saturation: f32) -> [f32; 3] {
    let s = (saturation / 100.0).clamp(0.0, 1.0);
    [1.0f32, 2.0 / 3.0, 1.0 / 3.0].map(|off| {
        let x = (hue / 360.0 + off).fract();
        let ramp = ((x * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0);
        0.1 * ((1.0 - s) + s * ramp).powf(2.2)
    })
}

/// C1 post-demosaic colour stage, flattened for upload.
#[derive(Clone, Debug)]
pub(crate) struct C1ColorParams {
    pub pre_matrix: [[f32; 3]; 3],
    pub film: Vec<[f32; 3]>,
    pub clut: Vec<[u16; 3]>,
}

impl C1ColorParams {
    pub(crate) fn new(
        pre_matrix: [[f32; 3]; 3],
        film: Vec<[f32; 3]>,
        clut: Vec<[u16; 3]>,
    ) -> Option<Self> {
        ((2..=FILM_MAX).contains(&film.len()) && clut.len() == CLUT_NODES)
            .then_some(Self { pre_matrix, film, clut })
    }

    /// Flattened copy of the CPU stage's constants, for the develop plane.
    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    pub(crate) fn from_stage(stage: &crate::z6iii::Z6iiiColorStage) -> Option<Self> {
        Self::new(
            stage.pre_matrix(),
            stage.film_nodes().to_vec(),
            stage.clut_nodes().to_vec(),
        )
    }
}

/// Camera-matched parameter constants copied from `aurora_raw::z6iii::color`.
pub(crate) const Z6III_PRE_MATRIX: [[f32; 3]; 3] =
    [[1.08, -0.07, -0.01], [-0.03, 1.04, -0.01], [-0.03, -0.07, 1.10]];
pub(crate) const IDENTITY_PRE_MATRIX: [[f32; 3]; 3] =
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
pub(crate) const LINEAR_FILM: [[f32; 3]; 2] = [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]];

/// Output-domain BandingSuppression (C1 stage 77) with the vendor tables
/// pre-resolved, mirroring `c1_banding::Tables`.
#[derive(Clone, Debug)]
pub(crate) struct BandingParams {
    pub gain: i32,
    pub limit: i32,
    pub strength: i32,
    pub words: [u32; 256],
    pub left: [u8; 256],
    pub right: [u8; 256],
    pub frame: [f32; 2],
}

impl BandingParams {
    pub(crate) fn from_raw(
        params: crate::c1_banding::Params,
        left: &[u8],
        right: &[u8],
        packed: &[u8],
        frame: [f32; 2],
    ) -> Option<Self> {
        if left.len() != 256 || right.len() != 256 || packed.len() != 1024 {
            return None;
        }
        let mut words = [0u32; 256];
        for (slot, word) in words.iter_mut().zip(packed.chunks_exact(4)) {
            *slot = u32::from_le_bytes(word.try_into().ok()?);
        }
        Some(Self {
            gain: params.gain,
            limit: params.limit,
            strength: params.strength,
            words,
            left: left.try_into().ok()?,
            right: right.try_into().ok()?,
            frame,
        })
    }

    pub(crate) fn from_vendor(
        params: crate::c1_banding::Params,
        frame: [f32; 2],
    ) -> Option<Self> {
        let (left, right, packed) = crate::c1_banding::vendor_buffers()?;
        Self::from_raw(params, &left[..], &right[..], &packed[..], frame)
    }
}

/// Host-side description of one develop plane.
///
/// Built once per `render_sized`; the GPU uploads it per call (the tables are
/// small compared with the frame).
#[derive(Clone, Debug)]
pub(crate) struct DevelopParams {
    pub wb: [f32; 3],
    pub exposure: f32,
    pub fine: bool,
    pub bypass: bool,
    /// `develop()` applies banding after `develop_display`; the streamed
    /// `develop_line_with` path does not. Phase 2 chooses via this flag.
    pub banding_outer: bool,
    pub tone_gamma: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub dehaze: Dehaze,
    pub hdr: HighDynamicRange,
    pub levels: Option<LevelsParams>,
    pub curves: [Option<[f32; 256]>; 5],
    pub color_balance: Option<[f32; 12]>,
    pub color_balance_gains: Option<ColorBalanceGains>,
    pub editor: Option<EditorParams>,
    pub bw: Option<BwParams>,
    pub c1: Option<C1ColorParams>,
    pub banding: Option<BandingParams>,
}

impl DevelopParams {
    /// Mirrors the constants `build_plan` derives from one `AdjustmentSet`.
    /// `wb` is `render_wb_gains`, `exposure` is `2^EV`, and `fine` is the
    /// 16-bit export path that bypasses the preview gamma LUT.
    pub(crate) fn from_adjustments(
        adj: &AdjustmentSet,
        wb: [f32; 3],
        exposure: f32,
        fine: bool,
    ) -> Self {
        let curve = |c: &Curve| (!c.is_identity()).then(|| c.lut());
        let brightness = adj.exposure.brightness;
        let tone_gamma = if brightness.abs() > 1e-3 {
            2f32.powf(-brightness / 50.0 * 0.6)
        } else {
            1.0
        };
        let contrast = adj.exposure.contrast / 50.0 * 0.8;
        let saturation = 1.0 + adj.exposure.saturation / 100.0;
        let levels = (adj.levels != Levels::default()).then(|| LevelsParams::from_levels(&adj.levels));
        let color_balance = (adj.color_balance != ColorBalance::default()).then(|| {
            let wheels = [&adj.color_balance.master, &adj.color_balance.shadow,
                          &adj.color_balance.midtone, &adj.color_balance.highlight];
            let mut out = [0.0f32; 12];
            for (i, wheel) in wheels.into_iter().enumerate() {
                if wheel.saturation == 0.0 && wheel.lightness == 0.0 {
                    continue;
                }
                let offset = wheel_offset(wheel, 0.35);
                out[i * 3..i * 3 + 3].copy_from_slice(&offset);
            }
            out
        });
        let color_balance_gains = adj.color_balance_gains.filter(|g| !g.is_neutral());
        let editor = (adj.color_editor != ColorEditor::default())
            .then(|| EditorParams::from_editor(&adj.color_editor));
        let bw = adj.black_white.enabled.then(|| BwParams::from_black_white(&adj.black_white));
        Self {
            wb,
            exposure,
            fine,
            bypass: false,
            banding_outer: true,
            tone_gamma,
            contrast,
            saturation,
            dehaze: adj.dehaze.clone(),
            hdr: adj.hdr.clone(),
            levels,
            curves: [
                curve(&adj.curves.rgb),
                curve(&adj.curves.red),
                curve(&adj.curves.green),
                curve(&adj.curves.blue),
                curve(&adj.curves.luma),
            ],
            color_balance,
            color_balance_gains,
            editor,
            bw,
            c1: None,
            banding: None,
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference (same order as `process.rs`, used by the parity tests and as
// the fallback's comparison point; production fallback stays in process.rs).
// ---------------------------------------------------------------------------

fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn luma601(c: [f32; 3]) -> f32 {
    0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
}

fn sstep(a: f32, b: f32, x: f32) -> f32 {
    if b <= a {
        return if x < a { 0.0 } else { 1.0 };
    }
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

fn gamma_table() -> &'static [f32; GAMMA_SIZE] {
    static TABLE: OnceLock<[f32; GAMMA_SIZE]> = OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|i| {
            linear_to_srgb(i as f32 / (GAMMA_SIZE - 1) as f32 * GAMMA_RANGE)
        })
    })
}

fn gamma_get(v: f32) -> f32 {
    let table = gamma_table();
    let i = (v.max(0.0) / GAMMA_RANGE * (GAMMA_SIZE - 1) as f32) as usize;
    table[i.min(GAMMA_SIZE - 1)]
}

fn encode_srgb(p: &DevelopParams, c: f32) -> f32 {
    if p.fine {
        linear_to_srgb(c.max(0.0))
    } else {
        gamma_get(c)
    }
}

fn lut_get(lut: &[f32; 256], v: f32) -> f32 {
    let x = v.clamp(0.0, 1.0) * 255.0;
    let i = (x as usize).min(254);
    let t = x - i as f32;
    lut[i] + (lut[i + 1] - lut[i]) * t
}

fn dehaze_soft(x: f32, a: f32) -> f32 {
    if a <= 0.0 {
        return x.max(0.0);
    }
    let t = (a - x.abs()).max(0.0);
    x.max(0.0) + t * t * t / (6.0 * a * a)
}

fn apply_dehaze(g: [f32; 3], d: &Dehaze) -> [f32; 3] {
    let omega = d.amount / 100.0;
    if omega <= 0.0 || !omega.is_finite() {
        return g;
    }
    let c = d.shadow_tone.map(|v| v.max(0.0));
    let a = c.map(|v| v * 65535.0);
    if a.iter().all(|v| *v <= f32::EPSILON) {
        return g;
    }
    let level: [(bool, f32, f32); 3] = std::array::from_fn(|c| {
        let (lo, hi) = (d.level_low[c], d.level_high[c]);
        (hi - lo > 1e-3, lo, hi - lo)
    });
    let i: [f32; 3] = std::array::from_fn(|c| {
        if level[c].0 {
            (g[c] - level[c].1) / level[c].2 * 65535.0
        } else {
            g[c] * 65535.0
        }
    });
    let eps = 1e-6;
    let delta = (2.0 * omega).min(1.0);
    let ah = c.map(|v| v * delta);
    let mut i1 = [0.0f32; 3];
    let mut a1 = [0.0f32; 3];
    for c in 0..3 {
        i1[c] = dehaze_soft(i[c] - ah[c], ah[c]);
        a1[c] = dehaze_soft(a[c] - ah[c], ah[c]);
    }
    let mut recovered = [0.0f32; 3];
    let li = luma(i1);
    let la = luma(a1);
    let keep = la / (li + la).max(eps);
    for c in 0..3 {
        recovered[c] = (1.0 - keep) * i1[c];
        i1[c] *= keep;
    }
    let m_raw = (i1[0] / a1[0].max(eps))
        .min(i1[1] / a1[1].max(eps))
        .min(i1[2] / a1[2].max(eps));
    let m0 = m_raw.clamp(0.0, 1.0);
    let mag = (i1[0] * i1[0] + i1[1] * i1[1] + i1[2] * i1[2]).sqrt();
    let dvec = a1.map(|v| m0 * v);
    let mag_d =
        ((i1[0] - dvec[0]).powi(2) + (i1[1] - dvec[1]).powi(2) + (i1[2] - dvec[2]).powi(2))
            .sqrt();
    let ratio = (mag + eps).max(eps);
    let val = (ratio * 0.375 - (mag_d + eps) * (4.0 / 7.0)) / ratio;
    let t = 1.0 / (1.0 + ((0.5 - val) * 4.0).exp());
    let m = m0 * t;
    for c in 0..3 {
        recovered[c] += (1.0 - t) * i1[c];
        i1[c] *= t;
    }
    let t0 = (1.0 - omega * m).max(0.05);
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        out[c] = (i1[c] - a1[c]) / t0 + a1[c] + recovered[c];
    }
    let max_i = i1[0].max(i1[1]).max(i1[2]);
    if max_i > eps {
        let sat_i = (max_i - i1[0].min(i1[1]).min(i1[2])) / max_i;
        let limit = sat_i * (1.0 + (0.5 + 2.0 * m_raw).powi(2)).sqrt();
        let max_o = out[0].max(out[1]).max(out[2]);
        if max_o > eps {
            let sat_o = (max_o - out[0].min(out[1]).min(out[2])) / max_o;
            if sat_o > limit {
                let k = limit / sat_o;
                let l = luma(out);
                out = [k * out[0] + (1.0 - k) * l, k * out[1] + (1.0 - k) * l, k * out[2] + (1.0 - k) * l];
            }
        }
    }
    std::array::from_fn(|c| {
        if level[c].0 {
            (level[c].1 + out[c] * level[c].2 / 65535.0).max(0.0)
        } else {
            (out[c] / 65535.0).max(0.0)
        }
    })
}

fn hdr_toe(x: f32, k: f32) -> f32 {
    let u = x * (2.0 * 0.9);
    let v = u.abs() + 1.0;
    let r = (u * v) / (v + u * u) * 0.5 + 0.5;
    let r = r.max(u - r + 1.0) * (1.0 / (2.0 * k));
    let s = r * r;
    let num = (1.0 + r + s) * (r + s * s);
    num / (1.0 + num) * k
}

fn hdr_inv_toe(x: f32, k: f32) -> f32 {
    hdr_toe(-x, k)
}

fn apply_hdr(mut g: [f32; 3], hdr: &HighDynamicRange) -> [f32; 3] {
    let x = luma(g);
    let mut shift = 0.0;
    if hdr.whites != 0.0 {
        shift += hdr.whites / 100.0 * 0.2 * sstep(0.65, 1.0, x);
    }
    if hdr.blacks != 0.0 {
        shift += hdr.blacks / 100.0 * 0.12 * (1.0 - sstep(0.0, 0.25, x));
    }
    if shift.abs() > 1e-5 {
        let k = (x + shift).max(0.0) / x.max(1e-4);
        g = [g[0] * k, g[1] * k, g[2] * k];
    }
    if hdr.highlights == 0.0 && hdr.shadows == 0.0 {
        return g;
    }
    let x = luma(g);
    let m = (hdr.highlights / 100.0).clamp(-1.0, 1.0);
    let b = if m != 0.0 {
        let t = hdr_toe(x, 1.0);
        2f32.powf(if m < 0.0 { -t } else { t }) * m.abs() + (1.0 - m.abs())
    } else {
        1.0
    };
    let m = (hdr.shadows / 100.0).clamp(-1.0, 1.0);
    let a = if m != 0.0 {
        let t = hdr_inv_toe(x, 1.0);
        2f32.powf(if m < 0.0 { -t } else { t }) * m.abs() + (1.0 - m.abs())
    } else {
        1.0
    };
    let gain = a / b;
    [g[0] * gain, g[1] * gain, g[2] * gain]
}

fn linear_edits(p: &DevelopParams, lin: [f32; 3], gain: Option<&[f32]>, index: usize) -> [f32; 3] {
    let mut c = [lin[0] * p.wb[0], lin[1] * p.wb[1], lin[2] * p.wb[2]];
    if let Some(plane) = gain {
        let g = plane[index];
        c = [c[0] * g, c[1] * g, c[2] * g];
    }
    c = apply_dehaze(c, &p.dehaze);
    c = apply_hdr(c, &p.hdr);
    [c[0] * p.exposure, c[1] * p.exposure, c[2] * p.exposure]
}

fn film_at(film: &[[f32; 3]], code: f32) -> f32 {
    if film.is_empty() {
        return code;
    }
    let right = film.partition_point(|node| node[0] <= code).clamp(1, film.len() - 1);
    let [x0, y0, m0] = film[right - 1];
    let [x1, y1, m1] = film[right];
    let dx = x1 - x0;
    if dx.abs() < 0.0001 {
        return ((code - x0) * (m0 + m1) + (y0 + y1)) * 0.5;
    }
    let t = (code - x0) / dx;
    let s = t - 1.0;
    let base = t * t * (y1 - y0) * (-2.0 * t + 3.0) + y0;
    ((m0 * s + m1 * t) * t * s) * dx + base
}

fn couple_float(before: [f32; 3], curved: [f32; 3]) -> [f32; 3] {
    const K: f32 = 4.0 / 256.0;
    const SLOPE: f32 = 1.0 / (1.0 - 0.5 * K);
    const TOE: f32 = SLOPE / (2.0 * K);
    const INTERCEPT: f32 = 1.0 - SLOPE;
    let encode = |code: f32| if code > K * K { SLOPE * code.sqrt() + INTERCEPT } else { TOE * code };
    let luma = |[r, g, b]: [f32; 3]| (b * 0.114 + (r * 0.299 + g * 0.587)).abs() + 1e-10;
    let range = |[r, g, b]: [f32; 3]| r.max(g.max(b)) - r.min(g.min(b));
    let source = before.map(encode);
    let mut target = curved.map(encode);
    let mut y_target = luma(target);
    if target.into_iter().any(|value| value > 1.0) {
        let limit = luma(target.map(|value| value.min(1.0)));
        target = target.map(|value| value * (limit / y_target));
        y_target = limit;
    }
    let source = source.map(|value| value * (y_target / luma(source)));
    let gain = ((range(target) + 1e-6) / (range(source) + 1e-6)) * 0.5 + 0.5;
    let mut result = source.map(|value| (value - y_target) * gain + y_target);
    let distance = result[0].max(result[1].max(result[2])) - y_target;
    if distance > 0.0 && 1.0 - y_target < 2.0 * distance {
        let ratio = (1.0 - y_target) / distance;
        let edge = (1.0 - (ratio - 1.0).abs()).max(0.0);
        let smooth = edge * edge * edge * (-1.0 / 6.0) + ratio.min(1.0);
        result = result.map(|value| (value - y_target) * smooth + y_target);
    }
    std::array::from_fn(|channel| {
        let value = result[channel];
        let decoded = if value > K * K * TOE {
            let z = (value - INTERCEPT) * (1.0 / SLOPE);
            z * z
        } else {
            (1.0 / TOE) * value
        };
        curved[channel] * 0.35 + decoded * (1.0 - 0.35)
    })
}

fn encode_camera(sample: f32) -> f32 {
    if sample > 0.01947 { 1.075 * sample.sqrt() + 1.0 - 1.075 } else { 3.8521 * sample.max(0.0) }
}

fn clut_sample(nodes: &[[u16; 3]], rgb: [f32; 3]) -> [f32; 3] {
    let sizes = [CLUT_N; 3];
    let peak = rgb.into_iter().fold(1.0_f32, f32::max);
    let scale = 1.0 / peak;
    let [(red, red_fraction), (green, green_fraction), (blue, blue_fraction)] =
        std::array::from_fn(|channel| {
            let position = (rgb[channel] * scale).clamp(0.0, 1.0) * (sizes[channel] - 1) as f32;
            let base = position as usize;
            (base, position - base as f32)
        });
    let fractions = [red_fraction, green_fraction, blue_fraction];
    let mut order = [1, 0, 2];
    if red_fraction > green_fraction { order.swap(0, 1); }
    if fractions[2] > fractions[order[0]] { order.rotate_right(1); }
    else if fractions[2] >= fractions[order[1]] { order.swap(1, 2); }
    let [largest, middle, smallest] = order.map(|axis| fractions[axis]);
    let weights = [1.0 - largest, largest - middle, middle - smallest, smallest];
    let mut corner = [red, green, blue];
    let node = |[red, green, blue]: [usize; 3]| nodes[(red * sizes[1] + green) * sizes[2] + blue];
    let mut vertices = [node(corner); 4];
    for (slot, axis) in vertices[1..].iter_mut().zip(order) {
        corner[axis] = (corner[axis] + 1).min(sizes[axis] - 1);
        *slot = node(corner);
    }
    std::array::from_fn(|channel| {
        let [base, first, second, last] = vertices.map(|vertex| f32::from(vertex[channel]));
        let code = ((base * weights[0] + first * weights[1]) + second * weights[2]) + last * weights[3];
        (code * peak) * f32::from_bits(0x3780_0080)
    })
}

fn c1_develop(p: &C1ColorParams, rgb: [f32; 3]) -> [f32; 3] {
    let linear = rgb.map(|value| value.max(0.0));
    let matrix = p.pre_matrix.map(|[red, green, blue]| {
        (red * linear[0] + green * linear[1] + blue * linear[2]).max(0.0)
    });
    let film = matrix.map(|value| film_at(&p.film, value));
    let coupled = couple_float(matrix, film);
    let camera = coupled.map(encode_camera);
    clut_sample(&p.clut, camera)
}

fn tone_contrast(g: f32, amount: f32) -> f32 {
    if amount.abs() < 1e-4 {
        return g;
    }
    let k = 1.0 + amount;
    if k >= 1.0 {
        let x = (g - 0.5) * k;
        0.5 + x / (1.0 + (x.abs() * (k - 1.0) * 0.8))
    } else {
        0.5 + (g - 0.5) * k
    }
}

fn levels_input(v: f32, lo: f32, span: f32, inv_gamma: f32) -> f32 {
    ((v - lo) / span).clamp(0.0, 1.0).powf(inv_gamma)
}

fn color_balance(g: [f32; 3], offsets: &[f32; 12]) -> [f32; 3] {
    let l = luma(g).clamp(0.0, 1.0);
    let ws = (1.0 - l).powi(2);
    let wh = l.powi(2);
    let wm = (1.0 - ws - wh).max(0.0);
    let weights = [1.0, ws, wm, wh];
    let mut out = g;
    for wheel in 0..4 {
        for c in 0..3 {
            out[c] += offsets[wheel * 3 + c] * weights[wheel];
        }
    }
    out
}

fn color_balance_zone_weights(l: f32) -> [f32; 3] {
    let protect = if l <= 1.0 { 1.0 } else { (2.0 - l).clamp(0.0, 1.0) };
    let bell = 4.0 * l * (1.0 - l);
    [l * l * protect, bell * bell * bell * 0.5, (1.0 - l) * (1.0 - l) * 0.5]
}

fn color_balance_gains(g: [f32; 3], gains: &ColorBalanceGains) -> [f32; 3] {
    if gains.is_neutral() {
        return g;
    }
    let l_in = luma(g);
    let mut out = [g[0] * gains.master[0], g[1] * gains.master[1], g[2] * gains.master[2]];
    let maxc = out[0].max(out[1]).max(out[2]);
    if maxc > 1e-6 && maxc.is_finite() {
        let [wh, wm, ws] = color_balance_zone_weights(maxc);
        let norm = [out[0] / maxc, out[1] / maxc, out[2] / maxc];
        for (zone, w) in [(gains.highlight, wh), (gains.midtone, wm), (gains.shadow, ws)] {
            let gmin = zone[0].min(zone[1]).min(zone[2]);
            let gmax = zone[0].max(zone[1]).max(zone[2]);
            if !w.is_finite() || w.abs() <= 1e-6 || !(gmin > 1e-6) || gmax - gmin <= 1e-3 {
                continue;
            }
            for c in 0..3 {
                let step = (zone[c] - gmin) / gmin;
                if step.is_finite() {
                    out[c] += w * step * norm[c];
                }
            }
        }
    }
    let l_out = luma(out);
    if l_out.abs() > 1e-6 && l_in > 0.0 && l_out.is_finite() {
        let k = (l_in / l_out).clamp(0.0, 8.0);
        out = [out[0] * k, out[1] * k, out[2] * k];
    }
    out.map(|v| if v.is_finite() { v.max(0.0) } else { 0.0 })
}

const BW_C1_LUMA_Q10: [i32; 3] = [340, 626, 56];

fn bw_sector_y(g: [f32; 3], bw: &BwParams) -> f32 {
    let (w_min, w_max) = (bw.w_min, bw.w_max);
    let q = g.map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as i32);
    let (i_max, max) = if q[0] > q[2] { (0, q[0]) } else { (2, q[2]) };
    let (i_max, max) = if q[1] > max { (1, q[1]) } else { (i_max, max) };
    let (i_min, min) = if q[1] < q[0] { (1, q[1]) } else { (0, q[0]) };
    let (i_min, min) = if q[2] < min { (2, q[2]) } else { (i_min, min) };
    let mid = q[3 - i_max - i_min];
    let mut t = min + ((mid - min) * w_min[i_min] + (max - mid) * w_max[i_max]) / 1024;
    let y = (q[0] * BW_C1_LUMA_Q10[0] + q[1] * BW_C1_LUMA_Q10[1] + q[2] * BW_C1_LUMA_Q10[2]) / 1024;
    if t * 8 < y {
        t = if t <= 0 {
            0
        } else {
            let yu = y as u32;
            let a = ((yu * yu) >> 5) * 25;
            let b = 1 + (yu.wrapping_mul(288).wrapping_sub((t as u32).wrapping_mul(1024)) >> 5);
            ((a / b) as i32 - (yu / 32) as i32).max(0)
        };
    }
    let shift = bw.shift;
    (((t >> shift) << shift).clamp(0, 65535) as f32) / 65535.0
}

fn bw_split_tone(y: f32, bw: &BwParams) -> [f32; 3] {
    if bw.light_lo <= bw.dark_hi {
        return [y, y, y];
    }
    let dark_ramp = bw.tint_dark.map(|c| c * y / bw.light_lo);
    let light_ramp = bw.tint_light.map(|c| c * y / (bw.light_lo - bw.dark_hi));
    let out = if y < bw.dark_hi {
        dark_ramp
    } else if y > bw.light_lo {
        light_ramp
    } else {
        let t = (y - bw.dark_hi) / (bw.light_lo - bw.dark_hi);
        [0, 1, 2].map(|i| dark_ramp[i] + (light_ramp[i] - dark_ramp[i]) * t)
    };
    out.map(|v| v.clamp(0.0, 1.0))
}

fn black_white(g: [f32; 3], bw: &BwParams) -> [f32; 3] {
    let y = if bw.use_luma { luma(g) } else { bw_sector_y(g, bw) }.max(0.0);
    if !bw.split {
        return [y, y, y];
    }
    bw_split_tone(y, bw)
}

fn rgb_to_hsv(c: [f32; 3]) -> (f32, f32, f32) {
    let max = c[0].max(c[1]).max(c[2]);
    let min = c[0].min(c[1]).min(c[2]);
    let d = max - min;
    let h = if d < 1e-6 {
        0.0
    } else if max == c[0] {
        60.0 * ((c[1] - c[2]) / d).rem_euclid(6.0)
    } else if max == c[1] {
        60.0 * ((c[2] - c[0]) / d + 2.0)
    } else {
        60.0 * ((c[0] - c[1]) / d + 4.0)
    };
    let s = if max < 1e-6 { 0.0 } else { d / max };
    (h, s, max)
}

fn hue_rgb(hue: f32) -> [f32; 3] {
    let h = (hue.rem_euclid(360.0)) / 60.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    match h as u32 {
        0 => [1.0, x, 0.0],
        1 => [x, 1.0, 0.0],
        2 => [0.0, 1.0, x],
        3 => [0.0, x, 1.0],
        4 => [x, 0.0, 1.0],
        _ => [1.0, 0.0, x],
    }
}

fn wheel_offset(w: &ColorWheel, strength: f32) -> [f32; 3] {
    let c = hue_rgb(w.hue);
    let mean = (c[0] + c[1] + c[2]) / 3.0;
    let s = w.saturation * strength;
    let l = w.lightness * 0.25;
    [(c[0] - mean) * s + l, (c[1] - mean) * s + l, (c[2] - mean) * s + l]
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let base = hue_rgb(h);
    [v * (1.0 - s + s * base[0]), v * (1.0 - s + s * base[1]), v * (1.0 - s + s * base[2])]
}

fn signed_hue_delta(h: f32, center: f32) -> f32 {
    let d = (h - center).rem_euclid(360.0);
    if d > 180.0 { d - 360.0 } else { d }
}

fn hue_distance(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(360.0);
    d.min(360.0 - d)
}

fn band_weight(b: &[f32], extra_feather: f32, h: f32, s: f32, v: f32) -> f32 {
    let d = signed_hue_delta(h, b[0]);
    let feather = b[9].max(0.1) + extra_feather;
    let hue_w = if d < b[3] {
        sstep(b[3] - feather, b[3], d)
    } else if d > b[4] {
        1.0 - sstep(b[4], b[4] + feather, d)
    } else {
        1.0
    };
    let ds = (s - b[1]) * 100.0;
    let sat_feather = 10.0f32.max(b[9]);
    let sat_w = if ds < b[5] {
        sstep(b[5] - sat_feather, b[5], ds)
    } else if ds > b[6] {
        1.0 - sstep(b[6], b[6] + sat_feather, ds)
    } else {
        1.0
    };
    let vv = v.clamp(0.0, 1.0);
    let value_w = if vv < b[7] {
        sstep(b[7] - 0.05, b[7], vv)
    } else if vv > b[8] {
        1.0 - sstep(b[8], b[8] + 0.05, vv)
    } else {
        1.0
    };
    (hue_w * sat_w * value_w).clamp(0.0, 1.0)
}

fn color_range_weight(e: &[f32], c: [f32; 3]) -> f32 {
    let (h, s, _) = rgb_to_hsv(c);
    let half = e[1] / 2.0;
    let d = hue_distance(h, e[0]);
    let hue_w = 1.0 - sstep(half, half + e[2].max(0.1), d);
    hue_w * sstep(e[3] * 0.5, e[3] + 0.05, s)
}

fn apply_homogeneity(band: &[f32], homogeneity: &[f32], w: f32, h: &mut f32, s: &mut f32, v: &mut f32) {
    let hue = homogeneity[0] / 100.0 * w;
    if hue != 0.0 {
        *h += signed_hue_delta(band[0], *h) * hue;
    }
    let sat = homogeneity[1] / 100.0 * w;
    if sat != 0.0 {
        *s += (band[1] - *s) * sat;
    }
    let light = homogeneity[2] / 100.0 * w;
    if light != 0.0 {
        *v += (band[2] - *v) * light;
    }
}

fn color_editor(g: [f32; 3], e: &[f32], n_skin: i32, n_shift: i32) -> [f32; 3] {
    let (h0, s0, v0) = rgb_to_hsv([g[0].max(0.0), g[1].max(0.0), g[2].max(0.0)]);
    if s0 < 0.02 {
        return g;
    }
    let extra = e[E_BASIC_EXTRA].max(0.0);
    let mut h = h0;
    let mut s = s0;
    let mut v = v0;
    let mut dh = 0.0;
    let mut ds = 0.0;
    let mut dl = 0.0;

    if e[E_GLOBAL_ENABLED] != 0.0 {
        let band = &e[E_GLOBAL_BAND..E_GLOBAL_BAND + 12];
        let mut w = band_weight(band, 0.0, h0, s0, v0);
        if band[10] != 0.0 { w = 1.0 - w; }
        dh += e[E_GLOBAL_SHIFT] * w;
        ds += e[E_GLOBAL_SHIFT + 1] * w;
        dl += e[E_GLOBAL_SHIFT + 2] * w;
    }

    for i in 0..n_skin as usize {
        let base = E_SKIN_BASE + i * E_SKIN_STRIDE;
        let entry = &e[base..base + E_SKIN_STRIDE];
        let mode = entry[0] as i32;
        if mode == 1 {
            let band = &entry[1..13];
            let hom = &entry[13..16];
            let mut w = band_weight(band, 0.0, h0, s0, v0);
            if band[10] != 0.0 { w = 1.0 - w; }
            apply_homogeneity(band, hom, w, &mut h, &mut s, &mut v);
        } else if mode == 2 {
            let w = color_range_weight(&entry[16..20], g);
            h += signed_hue_delta(entry[16], h) * (entry[20] / 100.0 * w);
            if entry[21] > 0.0 {
                let target_s = (entry[19] + 0.35).clamp(0.15, 0.55);
                s += (target_s - s) * (entry[21] / 100.0 * w);
            }
            if entry[22] > 0.0 {
                let target_v = 0.65;
                v += (target_v - v) * (entry[22] / 100.0 * w * 0.5);
            }
        }
    }

    for i in 0..8 {
        let at = E_BASIC + i * E_BASIC_STRIDE;
        let entry = &e[at..at + E_BASIC_STRIDE];
        if entry[15] != 0.0 {
            continue;
        }
        let band = &entry[0..12];
        let mut w = band_weight(band, extra, h, s, v);
        if band[10] != 0.0 { w = 1.0 - w; }
        dh += entry[12] * w;
        ds += entry[13] * w;
        dl += entry[14] * w;
    }

    let shift_base = E_SKIN_BASE + n_skin as usize * E_SKIN_STRIDE;
    for i in 0..n_shift as usize {
        let at = shift_base + i * E_SHIFT_STRIDE;
        let entry = &e[at..at + E_SHIFT_STRIDE];
        let mode = entry[0] as i32;
        if mode == 1 {
            continue;
        }
        let w = if mode == 0 {
            let band = &entry[2..14];
            let mut w = band_weight(band, entry[1], h0, s0, v0);
            if band[10] != 0.0 { w = 1.0 - w; }
            w
        } else {
            color_range_weight(&entry[17..21], g)
        };
        dh += entry[14] * w;
        ds += entry[15] * w;
        dl += entry[16] * w;
    }

    let dh = dh.clamp(-180.0, 180.0);
    let ds = ds.clamp(-100.0, 100.0);
    let dl = dl.clamp(-100.0, 100.0);
    let s2 = (s * (1.0 + ds / 100.0)).clamp(0.0, 1.0);
    let v2 = (v * (1.0 + dl / 100.0 * 0.5 * s)).max(0.0);
    hsv_to_rgb(h + dh, s2, v2)
}

fn output_banding(p: &DevelopParams, rgb: [f32; 3], f: [f32; 2]) -> [f32; 3] {
    let Some(banding) = &p.banding else { return rgb };
    if banding.limit <= 0 || banding.strength <= 0 {
        return rgb;
    }
    let x = (f[0] * banding.frame[0]) as i32;
    let y = (f[1] * banding.frame[1]) as i32;
    let codes = rgb.map(|value| (value.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16);
    let cell = ((((x & !1) & 0x0f) << 4) | (y & 0x0f)) as usize;
    let block = (((x & 0xf0) >> 4) | (y & 0xf0)) as usize;
    let pair = banding.words[(block + usize::from(banding.left[cell])) & 255]
        .wrapping_sub(banding.words[(block + usize::from(banding.right[cell])) & 255]);
    let half = if x & 1 != 0 { pair >> 16 } else { pair & 0xffff };
    let high = (((half >> 8) as i32) * banding.gain) >> 16;
    let low = (((half & 255) as i32) * banding.gain) >> 16;
    magnitude_band(codes, high, low, banding.limit, banding.strength)
}

fn magnitude_band(mut codes: [u16; 3], high: i32, low: i32, limit: i32, strength: i32) -> [f32; 3] {
    for code in &mut codes {
        let value = i32::from(*code);
        let headroom = limit.min(65_535 - value);
        *code = (value + (((high - low) * headroom * strength) >> 16)).clamp(0, 65_535) as u16;
    }
    codes.map(|code| f32::from(code) / 65535.0)
}

fn develop_display(p: &DevelopParams, mut g: [f32; 3]) -> [f32; 3] {
    if p.tone_gamma != 1.0 {
        g = [g[0].max(0.0).powf(p.tone_gamma), g[1].max(0.0).powf(p.tone_gamma), g[2].max(0.0).powf(p.tone_gamma)];
    }
    g = [
        tone_contrast(g[0], p.contrast),
        tone_contrast(g[1], p.contrast),
        tone_contrast(g[2], p.contrast),
    ];
    if let Some(levels) = &p.levels {
        for c in 0..3 {
            g[c] = levels_input(g[c], levels.lo[0][c], levels.span[0][c], levels.inv_gamma[0][c]);
            g[c] = levels_input(g[c], levels.lo[1][c], levels.span[1][c], levels.inv_gamma[1][c]);
        }
    }
    if let Some(lut) = &p.curves[0] {
        g = [lut_get(lut, g[0]), lut_get(lut, g[1]), lut_get(lut, g[2])];
    }
    if let Some(lut) = &p.curves[1] { g[0] = lut_get(lut, g[0]); }
    if let Some(lut) = &p.curves[2] { g[1] = lut_get(lut, g[1]); }
    if let Some(lut) = &p.curves[3] { g[2] = lut_get(lut, g[2]); }
    if let Some(lut) = &p.curves[4] {
        let l = luma601(g).max(1e-4);
        let k = lut_get(lut, l) / l;
        g = [g[0] * k, g[1] * k, g[2] * k];
    }
    if let Some(levels) = &p.levels {
        g = g.map(|v| v * levels.out_gain + levels.out_shadow);
    }
    let sat = p.saturation;
    if (sat - 1.0).abs() > 1e-4 {
        let l = luma(g);
        g = [l + (g[0] - l) * sat, l + (g[1] - l) * sat, l + (g[2] - l) * sat];
    }
    if let Some(offsets) = &p.color_balance {
        g = color_balance(g, offsets);
    }
    if let Some(gains) = &p.color_balance_gains {
        g = color_balance_gains(g, gains);
    }
    if let Some(editor) = &p.editor {
        g = color_editor(g, &editor.floats, editor.n_skin, editor.n_shift);
    }
    if let Some(bw) = &p.bw {
        g = black_white(g, bw);
    }
    g
}

fn develop_pixel(
    p: &DevelopParams,
    sample: Sample,
    f: [f32; 2],
    index: usize,
    gain: Option<&[f32]>,
) -> [f32; 3] {
    if !sample.valid {
        return [0.11; 3];
    }
    let lin = sample.rgb;
    let mut display = if p.bypass {
        if let Some(c1) = &p.c1 {
            output_banding(p, c1_develop(c1, lin), f)
        } else {
            [encode_srgb(p, lin[0]), encode_srgb(p, lin[1]), encode_srgb(p, lin[2])]
        }
    } else {
        let c = linear_edits(p, lin, gain, index);
        match &p.c1 {
            Some(c1) => c1_develop(c1, c),
            None => [encode_srgb(p, c[0]), encode_srgb(p, c[1]), encode_srgb(p, c[2])],
        }
    };
    display = develop_display(p, display);
    if p.banding_outer {
        output_banding(p, display, f)
    } else {
        display
    }
}

/// CPU reference for one plane. Tests compare this with the CUDA result; the
/// production fallback remains the existing `process.rs` code.
pub(crate) fn develop_plane_cpu(
    samples: &[Sample],
    width: usize,
    height: usize,
    region: [f32; 4],
    params: &DevelopParams,
    gain_plane: Option<&[f32]>,
) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0; 3]; width * height];
    for y in 0..height {
        let fy = region[1] + (y as f32 + 0.5) / height as f32 * region[3];
        for x in 0..width {
            let fx = region[0] + (x as f32 + 0.5) / width as f32 * region[2];
            let index = y * width + x;
            out[index] = develop_pixel(params, samples[index], [fx, fy], index, gain_plane);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CUDA pool
// ---------------------------------------------------------------------------

struct SendBuffer(DeviceBuffer<'static>);

// The boxed driver table outlives every buffer it allocated and `State`
// serialises all use, so a raw handle never moves concurrently.
unsafe impl Send for SendBuffer {}

fn alloc_static(gpu: &Nvidia, bytes: usize) -> Result<DeviceBuffer<'static>, CudaError> {
    let buffer = gpu.alloc_buffer(bytes)?;
    Ok(unsafe { std::mem::transmute::<DeviceBuffer<'_>, DeviceBuffer<'static>>(buffer) })
}

struct Pool {
    gpu: Box<Nvidia>,
    io: SendBuffer,
    gain: SendBuffer,
    params: SendBuffer,
    luts: SendBuffer,
    c1: SendBuffer,
    editor: SendBuffer,
    banding: SendBuffer,
    io_bytes: usize,
    gain_bytes: usize,
    editor_bytes: usize,
    tables: TableCache,
    /// Reservation in the shared ledger; dropping the pool gives it back.
    lease: VramLease,
}

/// The source fingerprints of the static tables that are resident on the
/// device right now.
///
/// The curve LUTs, the colour-editor table, the banding words and the C1
/// tables do not depend on the band, so they stay on the device until the
/// data behind them changes. A fingerprint is the exact source bytes of one
/// table, so a resident copy is only reused when the source is byte for byte
/// the one that produced it.
#[derive(Default)]
struct TableCache {
    luts: Option<Vec<u8>>,
    editor: Option<Vec<u8>>,
    banding: Option<Vec<u8>>,
    c1: Option<Vec<u8>>,
}

impl Pool {
    fn open(width: usize, height: usize) -> Result<Self, CudaError> {
        let io_bytes = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(16))
            .ok_or(CudaError::Unavailable("develop frame size"))?;
        let gain_bytes = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(4))
            .ok_or(CudaError::Unavailable("develop gain size"))?;
        let editor_bytes = E_SKIN_BASE * 4;
        let lease = VramLease::acquire(
            "develop",
            fixed_bytes() + io_bytes + gain_bytes + editor_bytes,
            true,
        )
        .map_err(budget_error)?;
        let gpu = Box::new(Nvidia::open_ptx(PTX)?);
        Ok(Self {
            io: SendBuffer(alloc_static(&gpu, io_bytes.max(16))?),
            gain: SendBuffer(alloc_static(&gpu, gain_bytes.max(4))?),
            params: SendBuffer(alloc_static(&gpu, P_SIZE * 4)?),
            luts: SendBuffer(alloc_static(&gpu, (1280 + GAMMA_SIZE) * 4)?),
            c1: SendBuffer(alloc_static(
                &gpu,
                FILM_MAX * 3 * 4 + CLUT_NODES * 3 * 2,
            )?),
            editor: SendBuffer(alloc_static(&gpu, editor_bytes)?),
            banding: SendBuffer(alloc_static(&gpu, 256 * 4 + 512)?),
            gpu,
            io_bytes,
            gain_bytes,
            editor_bytes,
            tables: TableCache::default(),
            lease,
        })
    }

    /// Grows the frame buffers inside the shared budget. The lease is resized
    /// before allocating, and rolled back if the driver allocation fails, so
    /// the ledger never claims bytes that are not really resident.
    fn ensure(&mut self, io_bytes: usize, gain_bytes: usize, editor_bytes: usize) -> Result<(), CudaError> {
        let next_io = io_bytes.max(self.io_bytes);
        let next_gain = gain_bytes.max(self.gain_bytes);
        let next_editor = editor_bytes.max(self.editor_bytes);
        if next_io == self.io_bytes && next_gain == self.gain_bytes && next_editor == self.editor_bytes {
            return Ok(());
        }
        let old_total = self.lease.bytes();
        self.lease
            .resize(fixed_bytes() + next_io + next_gain + next_editor, true)
            .map_err(budget_error)?;
        let result = (|| {
            if next_io > self.io_bytes {
                self.io = SendBuffer(alloc_static(&self.gpu, next_io)?);
                self.io_bytes = next_io;
            }
            if next_gain > self.gain_bytes {
                self.gain = SendBuffer(alloc_static(&self.gpu, next_gain)?);
                self.gain_bytes = next_gain;
            }
            if next_editor > self.editor_bytes {
                self.editor = SendBuffer(alloc_static(&self.gpu, next_editor)?);
                self.editor_bytes = next_editor;
                // A fresh buffer holds no table, so the fingerprint that claimed
                // a resident copy is no longer true.
                self.tables.editor = None;
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = self.lease.resize(old_total, true);
        }
        result
    }

    /// Uploads the static tables whose source changed since the last band.
    ///
    /// The curve LUTs, the colour-editor table, the banding words and the C1
    /// tables are the same for every band of a frame and for every frame
    /// that keeps the same adjustments. They are compared byte for byte
    /// against the source that produced the resident copy, so an unchanged
    /// table is not rebuilt and not uploaded again.
    ///
    /// The per-band parameters (frame size, region, gain flag) are not part
    /// of any fingerprint: they live in the parameter table, which is
    /// uploaded for every band.
    fn upload_static_tables(&mut self, params: &DevelopParams) -> Result<(), CudaError> {
        let luts = luts_parts(params);
        if !source_matches(self.tables.luts.as_deref(), &luts) {
            let table = luts_floats(params);
            let outcome = self.gpu.upload_buffer(&self.luts.0, as_bytes(&table));
            commit_upload(&mut self.tables.luts, &luts, outcome)?;
        }

        let editor = editor_parts(params);
        if !source_matches(self.tables.editor.as_deref(), &editor) {
            let table = params
                .editor
                .as_ref()
                .map(|e| e.floats.clone())
                .unwrap_or_else(|| vec![0.0; E_SKIN_BASE]);
            let outcome = self.gpu.upload_buffer(&self.editor.0, as_bytes(&table));
            commit_upload(&mut self.tables.editor, &editor, outcome)?;
        }

        let banding = banding_parts(params);
        if !source_matches(self.tables.banding.as_deref(), &banding) {
            let table = params
                .banding
                .as_ref()
                .map(banding_bytes)
                .unwrap_or_else(|| vec![0u8; 256 * 4 + 512]);
            let outcome = self.gpu.upload_buffer(&self.banding.0, &table);
            commit_upload(&mut self.tables.banding, &banding, outcome)?;
        }

        // An absent C1 stage passes a null handle and P_C1 stays 0, so the
        // buffer is never read. The fingerprint is left untouched: it keeps
        // describing whatever the device still holds, which stays valid for
        // the next frame that asks for the same C1 tables again.
        if let Some(stage) = params.c1.as_ref() {
            let c1 = c1_parts(params);
            if !source_matches(self.tables.c1.as_deref(), &c1) {
                let table = c1_bytes(stage);
                let outcome = self.gpu.upload_buffer(&self.c1.0, &table);
                commit_upload(&mut self.tables.c1, &c1, outcome)?;
            }
        }
        Ok(())
    }

    fn retained_bytes(&self) -> usize {
        self.lease.bytes()
    }
}

enum State {
    NotTried,
    Ready(Pool),
    Unavailable,
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::NotTried))
}

/// Successful GPU plane count; the end-to-end parity test uses it to prove the
/// device path really ran (no fake pass).
#[cfg(test)]
pub(crate) static GPU_PLANES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `0` / `false` disables the path; a missing variable means "try on Windows".
pub(crate) fn enabled() -> bool {
    match std::env::var_os("AURORA_DEVELOP_GPU") {
        Some(value) => {
            let value = value.to_string_lossy();
            !(value == "0" || value.eq_ignore_ascii_case("false"))
        }
        None => cfg!(windows),
    }
}

/// Drops the retained develop frame so another stage can use the budget.
/// `try_lock` only: a render in progress owns the pool and must not be touched.
fn release_retained() -> usize {
    let mut state = match state().try_lock() {
        Ok(state) => state,
        Err(_) => return 0,
    };
    let freed = match &*state {
        State::Ready(pool) => pool.retained_bytes(),
        _ => 0,
    };
    if freed > 0 {
        *state = State::NotTried;
    }
    freed
}

fn register_reclaimer() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        gpu_runtime::vram_register_reclaimer("develop", RECLAIM_PRIORITY, release_retained)
    });
}

pub(crate) fn prewarm() -> bool {
    if !enabled() {
        return false;
    }
    register_reclaimer();
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    match &mut *state {
        State::Ready(_) => true,
        State::Unavailable => false,
        State::NotTried => match Pool::open(PREWARM.0, PREWARM.1) {
            Ok(pool) => {
                *state = State::Ready(pool);
                log::info!("CUDA develop path ready");
                true
            }
            Err(err) => {
                log::warn!("develop_gpu: {err:?}; staying on CPU");
                // A budget refusal is transient; retry on the next render
                // instead of disabling develop for the process.
                if !budget_refused(&err) {
                    *state = State::Unavailable;
                }
                false
            }
        },
    }
}

fn budget_refused(err: &CudaError) -> bool {
    matches!(err, CudaError::Unavailable("vram budget"))
}

fn as_bytes<T>(value: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value.as_ptr() as *const u8, std::mem::size_of_val(value)) }
}

fn as_bytes_mut<T>(value: &mut [T]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(value.as_mut_ptr() as *mut u8, std::mem::size_of_val(value))
    }
}

/// One part of a static table's source data.
///
/// The static tables are identified by their exact source bytes: the witness
/// stores the same tags and bytes, so an unchanged source is recognised by a
/// plain byte comparison and the table is neither rebuilt nor uploaded again.
struct SourcePart<'a> {
    /// `true` when the source has this part at all.
    present: bool,
    /// The exact bytes of the part. Empty when `present` is `false`.
    bytes: &'a [u8],
}

/// Exact comparison of a witnessed source with the current source parts.
///
/// `witness` is `None` when the device buffer was never written, so the
/// source always counts as changed. The comparison is byte for byte: two
/// sources that produce different bytes never share a witness.
fn source_matches(witness: Option<&[u8]>, parts: &[SourcePart<'_>]) -> bool {
    let Some(witness) = witness else { return false };
    let mut at = 0usize;
    for part in parts {
        if witness.get(at) != Some(&(part.present as u8)) {
            return false;
        }
        at += 1;
        if part.present {
            let end = at.saturating_add(part.bytes.len());
            if witness.get(at..end) != Some(part.bytes) {
                return false;
            }
            at = end;
        }
    }
    at == witness.len()
}

/// The witness bytes for a source.
fn build_witness(parts: &[SourcePart<'_>]) -> Vec<u8> {
    let mut out = Vec::new();
    for part in parts {
        out.push(part.present as u8);
        if part.present {
            out.extend_from_slice(part.bytes);
        }
    }
    out
}

/// Records one static-table upload.
///
/// The witness is cleared *before* the copy runs: a failed `upload_buffer`
/// can leave the device buffer partially written, and an error on the
/// `State::Ready` path keeps the pool alive, so the previous witness must
/// never keep claiming the buffer still holds the old table. It is set again
/// only after the copy reported success.
fn commit_upload(
    witness: &mut Option<Vec<u8>>,
    source: &[SourcePart<'_>],
    outcome: Result<(), CudaError>,
) -> Result<(), CudaError> {
    *witness = None;
    outcome?;
    *witness = Some(build_witness(source));
    Ok(())
}

/// The five curve LUTs of one parameter set, as source parts.
fn luts_parts(p: &DevelopParams) -> [SourcePart<'_>; 5] {
    std::array::from_fn(|index| match &p.curves[index] {
        Some(lut) => SourcePart { present: true, bytes: as_bytes(&lut[..]) },
        None => SourcePart { present: false, bytes: &[] },
    })
}

/// The colour-editor table of one parameter set, as source parts.
fn editor_parts(p: &DevelopParams) -> [SourcePart<'_>; 1] {
    match &p.editor {
        Some(editor) => [SourcePart { present: true, bytes: as_bytes(&editor.floats) }],
        None => [SourcePart { present: false, bytes: &[] }],
    }
}

/// The banding words of one parameter set, as source parts.
fn banding_parts(p: &DevelopParams) -> [SourcePart<'_>; 3] {
    match &p.banding {
        Some(banding) => [
            SourcePart { present: true, bytes: as_bytes(&banding.words[..]) },
            SourcePart { present: true, bytes: &banding.left[..] },
            SourcePart { present: true, bytes: &banding.right[..] },
        ],
        None => [
            SourcePart { present: false, bytes: &[] },
            SourcePart { present: false, bytes: &[] },
            SourcePart { present: false, bytes: &[] },
        ],
    }
}

/// The C1 colour tables of one parameter set, as source parts.
fn c1_parts(p: &DevelopParams) -> [SourcePart<'_>; 2] {
    match &p.c1 {
        Some(c1) => [
            SourcePart { present: true, bytes: as_bytes(&c1.film) },
            SourcePart { present: true, bytes: as_bytes(&c1.clut) },
        ],
        None => [
            SourcePart { present: false, bytes: &[] },
            SourcePart { present: false, bytes: &[] },
        ],
    }
}

fn params_floats(p: &DevelopParams) -> Vec<f32> {
    let mut out = vec![0.0f32; P_SIZE];
    out[P_FINE] = if p.fine { 1.0 } else { 0.0 };
    out[P_BYPASS] = if p.bypass { 1.0 } else { 0.0 };
    out[P_WB] = p.wb[0];
    out[P_WB + 1] = p.wb[1];
    out[P_WB + 2] = p.wb[2];
    out[P_EXPOSURE] = p.exposure;
    out[P_BRIGHT_GAMMA] = p.tone_gamma;
    out[P_CONTRAST] = p.contrast;
    out[P_SAT] = p.saturation;
    out[P_HDR] = p.hdr.highlights;
    out[P_HDR + 1] = p.hdr.shadows;
    out[P_HDR + 2] = p.hdr.whites;
    out[P_HDR + 3] = p.hdr.blacks;
    out[P_DEHAZE_OMEGA] = p.dehaze.amount / 100.0;
    for c in 0..3 {
        out[P_DEHAZE_C + c] = p.dehaze.shadow_tone[c];
        out[P_DEHAZE_LO + c] = p.dehaze.level_low[c];
        out[P_DEHAZE_HI + c] = p.dehaze.level_high[c];
    }
    if let Some(levels) = &p.levels {
        out[P_LEVELS] = 1.0;
        for c in 0..3 {
            out[P_LV_LO + c] = levels.lo[0][c];
            out[P_LV_SPAN + c] = levels.span[0][c];
            out[P_LV_INV + c] = levels.inv_gamma[0][c];
            out[P_LV_MLO + c] = levels.lo[1][c];
            out[P_LV_MSPAN + c] = levels.span[1][c];
            out[P_LV_MINV + c] = levels.inv_gamma[1][c];
        }
        out[P_LV_OUT_GAIN] = levels.out_gain;
        out[P_LV_OUT_SHADOW] = levels.out_shadow;
    }
    for i in 0..5 {
        out[P_CURVE + i] = if p.curves[i].is_some() { 1.0 } else { 0.0 };
    }
    if let Some(offsets) = &p.color_balance {
        out[P_CB] = 1.0;
        out[P_CB_OFF..P_CB_OFF + 12].copy_from_slice(offsets);
    }
    if let Some(gains) = &p.color_balance_gains {
        out[P_CBG] = 1.0;
        out[P_CBG_GAIN..P_CBG_GAIN + 3].copy_from_slice(&gains.master);
        out[P_CBG_GAIN + 3..P_CBG_GAIN + 6].copy_from_slice(&gains.shadow);
        out[P_CBG_GAIN + 6..P_CBG_GAIN + 9].copy_from_slice(&gains.midtone);
        out[P_CBG_GAIN + 9..P_CBG_GAIN + 12].copy_from_slice(&gains.highlight);
    }
    if let Some(bw) = &p.bw {
        out[P_BW] = 1.0;
        for c in 0..3 {
            out[P_BW_WMIN + c] = bw.w_min[c] as f32;
            out[P_BW_WMAX + c] = bw.w_max[c] as f32;
            out[P_BW_TINT_LIGHT + c] = bw.tint_light[c];
            out[P_BW_TINT_DARK + c] = bw.tint_dark[c];
        }
        out[P_BW_SHIFT] = bw.shift as f32;
        out[P_BW_USE_LUMA] = if bw.use_luma { 1.0 } else { 0.0 };
        out[P_BW_SPLIT] = if bw.split { 1.0 } else { 0.0 };
        out[P_BW_DARK_HI] = bw.dark_hi;
        out[P_BW_LIGHT_LO] = bw.light_lo;
    }
    if let Some(editor) = &p.editor {
        out[P_CE] = 1.0;
        out[P_CE_EXTRA] = editor.floats[E_BASIC_EXTRA];
        out[P_N_SKIN] = editor.n_skin as f32;
        out[P_N_SHIFT] = editor.n_shift as f32;
    }
    if let Some(c1) = &p.c1 {
        out[P_C1] = 1.0;
        out[P_FILM_N] = c1.film.len() as f32;
        for row in 0..3 {
            for col in 0..3 {
                out[P_PREMAT + row * 3 + col] = c1.pre_matrix[row][col];
            }
        }
    }
    if let Some(banding) = &p.banding {
        out[P_BANDING] = 1.0;
        out[P_BG_GAIN] = banding.gain as f32;
        out[P_BG_LIMIT] = banding.limit as f32;
        out[P_BG_STRENGTH] = banding.strength as f32;
        out[P_BG_FW] = banding.frame[0];
        out[P_BG_FH] = banding.frame[1];
    }
    out[P_BANDING_OUTER] = if p.banding_outer { 1.0 } else { 0.0 };
    out
}

fn luts_floats(p: &DevelopParams) -> Vec<f32> {
    let mut out = vec![0.0f32; 1280 + GAMMA_SIZE];
    for (i, lut) in p.curves.iter().enumerate() {
        let offset = i * 256;
        match lut {
            Some(lut) => out[offset..offset + 256].copy_from_slice(lut),
            None => {
                for (j, value) in out[offset..offset + 256].iter_mut().enumerate() {
                    *value = j as f32 / 255.0;
                }
            }
        }
    }
    out[1280..].copy_from_slice(gamma_table());
    out
}

fn c1_bytes(c1: &C1ColorParams) -> Vec<u8> {
    let mut out = vec![0u8; FILM_MAX * 3 * 4 + CLUT_NODES * 3 * 2];
    for (i, node) in c1.film.iter().take(FILM_MAX).enumerate() {
        for c in 0..3 {
            out[i * 12 + c * 4..i * 12 + c * 4 + 4].copy_from_slice(&node[c].to_le_bytes());
        }
    }
    let base = FILM_MAX * 3 * 4;
    for (i, node) in c1.clut.iter().enumerate() {
        for c in 0..3 {
            let at = base + i * 6 + c * 2;
            out[at..at + 2].copy_from_slice(&node[c].to_le_bytes());
        }
    }
    out
}

fn banding_bytes(b: &BandingParams) -> Vec<u8> {
    let mut out = vec![0u8; 256 * 4 + 512];
    for (i, word) in b.words.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    out[1024..1280].copy_from_slice(&b.left);
    out[1280..1536].copy_from_slice(&b.right);
    out
}

fn ptr<T>(value: &mut T) -> *mut c_void {
    value as *mut T as *mut c_void
}

fn launch<const N: usize>(
    gpu: &Nvidia,
    kernel: &'static str,
    w: usize,
    h: usize,
    mut args: [*mut c_void; N],
) -> Result<(), CudaError> {
    let grid = (((w as u32) + 15) / 16, ((h as u32) + 15) / 16);
    gpu.launch_raw(kernel, grid, BLOCK, &mut args)?;
    Ok(())
}

/// Why this frame must stay on the CPU. `None` means the point chain can run
/// here, provided the caller supplies the sampled RGB and any gain plane the
/// plan needs.
pub(crate) fn fallback_reason(
    adj: &AdjustmentSet,
    clarity_pre_curve: bool,
    uniform_light_without_plane: bool,
) -> Option<&'static str> {
    if !adj.layers.is_empty() {
        return Some("local layers/masks are composited per pixel on the CPU");
    }
    if clarity_pre_curve {
        return Some("clarity pre-curve is a neighbourhood stage between develop stages");
    }
    if adj.nodes.is_some() && !adj.layers.is_empty() {
        return Some("node graph may reorder the point chain");
    }
    if uniform_light_without_plane {
        return Some("LCC/vignette gain needs a CPU-computed output plane");
    }
    None
}

/// Retained device bytes for one develop frame: the 4-float IO frame, the
/// 4-byte gain plane and the colour-editor table. `None` on overflow.
pub(crate) fn retained_bytes(width: usize, height: usize, params: &DevelopParams) -> Option<usize> {
    let pixels = width.checked_mul(height)?;
    if pixels == 0 {
        return None;
    }
    let editor_bytes = params
        .editor
        .as_ref()
        .map(|e| e.floats.len().saturating_mul(4))
        .unwrap_or(0)
        .max(4);
    pixels
        .checked_mul(16)?
        .checked_add(pixels.checked_mul(4)?)?
        .checked_add(editor_bytes)
}

/// The retained-frame budget the device path enforces: now the shared ledger
/// budget instead of a private 768 MiB constant.
pub(crate) fn develop_budget() -> usize {
    gpu_runtime::vram_budget()
}

/// Non-allocating admission check.
///
/// Nothing frame-sized may be prepared before this returns `true`: no host
/// sample array, no gain plane, no CUDA module, no host IO frame. The caller
/// runs it once per render, before `prewarm()` and before the gain plane, so a
/// frame that would exceed the budget goes straight to the CPU with its old
/// buffers untouched.
pub(crate) fn can_run_develop_gpu(
    width: usize,
    height: usize,
    params: &DevelopParams,
    budget: usize,
) -> bool {
    if !enabled() {
        return false;
    }
    let Some(bytes) = retained_bytes(width, height, params) else { return false };
    if bytes > budget {
        return false;
    }
    // Unified admission: the shared ledger decides whether the whole engine can
    // hold this frame, counting rebuildable buffers other stages could release.
    gpu_runtime::vram_can_allocate("develop", fixed_bytes() + bytes)
}

/// Device bytes the develop pool retains for one band of `width × rows`.
///
/// The pool is band-shaped: `develop_band_sampled` opens it with the band
/// dimensions, so the 4-float IO band and the 4-byte gain band cover
/// `width × rows`, not the whole output frame. The parameter table, curve
/// LUTs, colour-editor table, banding words and C1 tables are allocated once
/// and stay resident for the life of the pool. `None` on overflow.
fn band_device_bytes_with(width: usize, rows: usize, editor_bytes: usize) -> Option<usize> {
    let pixels = width.checked_mul(rows)?;
    if pixels == 0 {
        return None;
    }
    pixels
        .checked_mul(16)?
        .checked_add(pixels.checked_mul(4)?)?
        .checked_add(editor_bytes.max(4))?
        .checked_add(fixed_bytes())
}

/// Host bytes one band needs: the full-frame gain plane the caller builds and
/// the pooled 4-float band.
///
/// The gain plane is counted unconditionally. It is `4 × width × full_height`
/// whenever the plan carries vignette/LCC, and counting it always keeps the
/// admission independent of slider state. Host memory is not on the device
/// ledger, so it is checked against the budget as its own conservative bound.
fn band_host_bytes_with(width: usize, rows: usize, full_height: usize) -> Option<usize> {
    let frame = width.checked_mul(full_height)?;
    let band = width.checked_mul(rows)?;
    if frame == 0 || band == 0 {
        return None;
    }
    frame.checked_mul(4)?.checked_add(band.checked_mul(16)?)
}

/// Device-side admission for one band of `width × rows`, against the shared
/// VRAM ledger: the same authority [`can_run_develop_gpu`] uses, but describing
/// the band the pool actually opens instead of the whole output frame.
pub(crate) fn band_device_admits(
    width: usize,
    rows: usize,
    params: &DevelopParams,
    budget: usize,
) -> bool {
    if !enabled() {
        return false;
    }
    let editor_bytes = params
        .editor
        .as_ref()
        .map(|e| e.floats.len().saturating_mul(4))
        .unwrap_or(0);
    let Some(bytes) = band_device_bytes_with(width, rows, editor_bytes) else { return false };
    if bytes > budget {
        return false;
    }
    gpu_runtime::vram_can_allocate("develop", bytes)
}

/// Admission for the band path: the device pool this band grows plus the host
/// plane the caller builds for it. Both must fit the budget.
///
/// `develop_plane` keeps [`can_run_develop_gpu`]: that API materialises a whole
/// frame, so its whole-frame ledger still applies.
pub(crate) fn can_run_band_develop_gpu(
    width: usize,
    rows: usize,
    full_height: usize,
    params: &DevelopParams,
    budget: usize,
) -> bool {
    if !band_device_admits(width, rows, params, budget) {
        return false;
    }
    band_host_bytes_with(width, rows, full_height).is_some_and(|bytes| bytes <= budget)
}

/// End-to-end CPU/GPU crossover policy of the point chain, in output pixels.
///
/// S6 measured the LCC/vignette gain planes end to end, not this kernel, so
/// these rows are a **policy table**, not measured develop timings: 256² and
/// 720p stay on the CPU, 1080p and above are admitted. `AURORA_DEVELOP_MIN_PIXELS`
/// overrides the floor; a real develop crossing has to be measured before any
/// row is quoted as a result.
const CROSSOVER_TABLE: [(u64, bool); 6] = [
    (65_536, false),       // 256×256
    (921_600, false),      // 1280×720
    (2_073_600, true),     // 1920×1080
    (8_294_400, true),     // 3840×2160
    (24_385_536, true),    // 6048×4032 (24.4 MP)
    (44_761_088, true),    // 8192×5464 (44.8 MP)
];

fn crossover_floor() -> u64 {
    static FLOOR: OnceLock<u64> = OnceLock::new();
    *FLOOR.get_or_init(|| {
        let measured = CROSSOVER_TABLE
            .iter()
            .find(|(_, wins)| *wins)
            .map(|(pixels, _)| *pixels)
            .unwrap_or(u64::MAX);
        std::env::var("AURORA_DEVELOP_MIN_PIXELS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|pixels| *pixels > 0)
            .unwrap_or(measured)
    })
}

/// `false` when the output is below the measured CPU/GPU crossover.
pub(crate) fn crossover_admits(width: usize, height: usize) -> bool {
    (width as u64).saturating_mul(height as u64) >= crossover_floor()
}

/// Host IO frames are reused between bands and frames. Bands are a few MiB, so
/// keeping one buffer resident removes the per-band allocator round trip without
/// pinning a whole 45 MP frame.
const HOST_IO_POOL_MAX_BYTES: usize = 64 * 1024 * 1024;

fn host_io_pool() -> &'static Mutex<Option<Vec<f32>>> {
    static POOL: OnceLock<Mutex<Option<Vec<f32>>>> = OnceLock::new();
    POOL.get_or_init(|| Mutex::new(None))
}

fn take_io(len: usize) -> Vec<f32> {
    let mut slot = host_io_pool().lock().unwrap_or_else(|error| error.into_inner());
    match slot.take() {
        Some(mut buffer) => {
            buffer.resize(len, 0.0);
            buffer
        }
        None => vec![0.0f32; len],
    }
}

fn give_io(buffer: Vec<f32>) {
    if buffer.capacity().saturating_mul(4) > HOST_IO_POOL_MAX_BYTES {
        return;
    }
    let mut slot = host_io_pool().lock().unwrap_or_else(|error| error.into_inner());
    *slot = Some(buffer);
}

fn fill_io<F>(io: &mut [f32], width: usize, sampler: &F)
where
    F: Fn(usize, usize) -> Option<[f32; 3]> + Sync,
{
    use rayon::prelude::*;
    io.par_chunks_exact_mut(4).enumerate().for_each(|(index, dst)| {
        match sampler(index % width, index / width) {
            Some(rgb) => {
                dst[0] = rgb[0];
                dst[1] = rgb[1];
                dst[2] = rgb[2];
                dst[3] = 1.0;
            }
            None => {
                dst[0] = 0.0;
                dst[1] = 0.0;
                dst[2] = 0.0;
                dst[3] = 0.0;
            }
        }
    });
}

/// Samples straight into the pooled host IO frame and runs the kernel on it.
///
/// This is the allocation-light entry point: no `Sample` / `SourceSample`
/// array is built, and `consume` receives the developed frame in place
/// (4 floats per pixel, row-major) before the buffer returns to the pool.
/// `None` means the caller must run the CPU path; nothing was left allocated.
pub(crate) fn develop_band_sampled<F, R>(
    width: usize,
    height: usize,
    region: [f32; 4],
    params: &DevelopParams,
    gain_plane: Option<&[f32]>,
    sampler: F,
    consume: impl FnOnce(&[f32]) -> R,
) -> Option<R>
where
    F: Fn(usize, usize) -> Option<[f32; 3]> + Sync,
{
    let pixels = width.checked_mul(height)?;
    if pixels == 0 || !can_run_develop_gpu(width, height, params, develop_budget()) {
        return None;
    }
    if gain_plane.is_some_and(|plane| plane.len() != pixels) {
        return None;
    }
    let mut io = take_io(pixels * 4);
    fill_io(&mut io, width, &sampler);
    match run_io(&mut io, width, height, region, params, gain_plane) {
        Ok(()) => {
            #[cfg(test)]
            GPU_PLANES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let out = consume(&io);
            give_io(io);
            Some(out)
        }
        Err(err) => {
            log::warn!("develop_gpu: {err:?}; staying on CPU");
            give_io(io);
            None
        }
    }
}

/// Runs the point chain on the GPU. `false` means the caller must run the CPU
/// path: the switch is off, the driver/PTX is missing, or a launch failed.
pub(crate) fn develop_plane(
    samples: &[Sample],
    width: usize,
    height: usize,
    region: [f32; 4],
    params: &DevelopParams,
    gain_plane: Option<&[f32]>,
    out: &mut [[f32; 3]],
) -> bool {
    let Some(pixels) = width.checked_mul(height) else { return false };
    if pixels == 0 || samples.len() != pixels || out.len() != pixels {
        return false;
    }
    if gain_plane.is_some_and(|plane| plane.len() != pixels) {
        return false;
    }
    if !can_run_develop_gpu(width, height, params, develop_budget()) {
        return false;
    }

    let mut io = take_io(pixels * 4);
    for (dst, sample) in io.chunks_exact_mut(4).zip(samples) {
        dst.copy_from_slice(&[sample.rgb[0], sample.rgb[1], sample.rgb[2], if sample.valid { 1.0 } else { 0.0 }]);
    }
    match run_io(&mut io, width, height, region, params, gain_plane) {
        Ok(()) => {
            #[cfg(test)]
            GPU_PLANES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for (dst, src) in out.iter_mut().zip(io.chunks_exact(4)) {
                *dst = [src[0], src[1], src[2]];
            }
            give_io(io);
            true
        }
        Err(err) => {
            log::warn!("develop_gpu: {err:?}; staying on CPU");
            give_io(io);
            false
        }
    }
}

/// Uploads the prepared IO frame, launches the point chain and downloads it.
fn run_io(
    io: &mut [f32],
    width: usize,
    height: usize,
    region: [f32; 4],
    params: &DevelopParams,
    gain_plane: Option<&[f32]>,
) -> Result<(), CudaError> {
    let pixels = width * height;
    let io_bytes = pixels.saturating_mul(16);
    let gain_bytes = pixels.saturating_mul(4);
    let editor_bytes = params
        .editor
        .as_ref()
        .map(|e| e.floats.len().saturating_mul(4))
        .unwrap_or(0)
        .max(4);
    let mut floats = params_floats(params);
    floats[P_W] = width as f32;
    floats[P_H] = height as f32;
    floats[P_RX] = region[0];
    floats[P_RY] = region[1];
    floats[P_RW] = region[2];
    floats[P_RH] = region[3];
    floats[P_HAS_GAIN] = if gain_plane.is_some() { 1.0 } else { 0.0 };

    let mut lock = state().lock().unwrap_or_else(|error| error.into_inner());
    let result = match &mut *lock {
        State::Unavailable => Err(CudaError::Unavailable("CUDA develop path is unavailable")),
        State::NotTried => {
            register_reclaimer();
            let mut pool = match Pool::open(width, height) {
                Ok(pool) => pool,
                Err(err) => {
                    if !budget_refused(&err) {
                        *lock = State::Unavailable;
                    }
                    return Err(err);
                }
            };
            let result = run_with_pool(
                &mut pool, io, &floats, params, gain_plane, width, height, io_bytes, gain_bytes,
                editor_bytes,
            );
            match result {
                Ok(()) => {
                    *lock = State::Ready(pool);
                    Ok(())
                }
                Err(err) => {
                    if !budget_refused(&err) {
                        *lock = State::Unavailable;
                    }
                    Err(err)
                }
            }
        }
        State::Ready(pool) => run_with_pool(
            pool, io, &floats, params, gain_plane, width, height, io_bytes, gain_bytes,
            editor_bytes,
        ),
    };
    // One throttled `AURORA_PERF {"event":"gpu_vram",...}` line when replaying.
    gpu_runtime::vram_maybe_emit_perf("develop");
    result
}

/// Grows the pool, refreshes the static tables and runs one band.
#[allow(clippy::too_many_arguments)]
fn run_with_pool(
    pool: &mut Pool,
    io: &mut [f32],
    floats: &[f32],
    params: &DevelopParams,
    gain_plane: Option<&[f32]>,
    width: usize,
    height: usize,
    io_bytes: usize,
    gain_bytes: usize,
    editor_bytes: usize,
) -> Result<(), CudaError> {
    pool.ensure(io_bytes, gain_bytes, editor_bytes)?;
    pool.upload_static_tables(params)?;
    run_pool(pool, io, floats, gain_plane, width, height, params.c1.is_some())
}

#[allow(clippy::too_many_arguments)]
fn run_pool(
    pool: &Pool,
    io: &mut [f32],
    floats: &[f32],
    gain_plane: Option<&[f32]>,
    width: usize,
    height: usize,
    c1_present: bool,
) -> Result<(), CudaError> {
    let gpu = pool.gpu.as_ref();
    gpu.upload_buffer(&pool.io.0, as_bytes(io))?;
    gpu.upload_buffer(&pool.params.0, as_bytes(floats))?;
    if let Some(plane) = gain_plane {
        gpu.upload_buffer(&pool.gain.0, as_bytes(plane))?;
    }

    let mut io_handle = pool.io.0.handle();
    let mut p_handle = pool.params.0.handle();
    let mut lut_handle = pool.luts.0.handle();
    let mut gain_handle = if gain_plane.is_some() { pool.gain.0.handle() } else { 0 };
    let mut c1_handle = if c1_present { pool.c1.0.handle() } else { 0 };
    let mut editor_handle = pool.editor.0.handle();
    let mut words = pool.banding.0.handle();
    let mut left = words + 1024;
    let mut right = words + 1280;
    launch(
        gpu,
        "develop_kernel",
        width,
        height,
        [
            ptr(&mut io_handle),
            ptr(&mut p_handle),
            ptr(&mut lut_handle),
            ptr(&mut gain_handle),
            ptr(&mut c1_handle),
            ptr(&mut editor_handle),
            ptr(&mut words),
            ptr(&mut left),
            ptr(&mut right),
        ],
    )?;
    gpu.download_buffer(&pool.io.0, as_bytes_mut(io))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Device-resident single-stream preview pipeline
// ---------------------------------------------------------------------------

struct SendStream(Stream<'static>);
unsafe impl Send for SendStream {}

struct SendPinned(PinnedBuffer<'static>);
unsafe impl Send for SendPinned {}

fn create_stream_static(gpu: &Nvidia) -> Result<Stream<'static>, CudaError> {
    let stream = gpu.create_stream()?;
    Ok(unsafe { std::mem::transmute::<Stream<'_>, Stream<'static>>(stream) })
}

fn alloc_pinned_static(gpu: &Nvidia, bytes: usize) -> Result<PinnedBuffer<'static>, CudaError> {
    let buf = gpu.alloc_pinned(bytes)?;
    Ok(unsafe { std::mem::transmute::<PinnedBuffer<'_>, PinnedBuffer<'static>>(buf) })
}

/// Exact identity of the device-resident sampled linear source frame.
/// Colour, tone, exposure, curves, NR and sharpening are absent so dragging
/// any of them hits the resident source with 0 CPU sampling and 0 H2D upload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResidentSourceKey {
    pub source: u64,
    pub geom_hash: u64,
    pub region_bits: [u32; 4],
    pub out_w: u32,
    pub out_h: u32,
    pub level: usize,
    pub cubic: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct UsmParams {
    pub sigma: f32,
    pub gain: f32,
    pub threshold: f32,
    pub halo: f32,
}

pub(crate) struct ResidentPreviewOut {
    pub pixels: Vec<u8>,
    pub histogram: Option<Histogram>,
    pub source_hit: bool,
}

const RESIDENT_RECLAIM_PRIORITY: u8 = 25;
const HIST_WORDS: usize = 1026;
const USM_MAX_WEIGHTS: usize = 25;

/// Resident preview crossover floor: because the linear source stays in VRAM
/// across slider drags (zero per-frame H2D), GPU wins even at modest viewport
/// sizes (e.g. 1080p / 2K / 4K), while 360px thumbnails (<= 86k px) stay off
/// the resident pool so they never evict the active preview source.
pub(crate) fn resident_crossover_admits(width: usize, height: usize) -> bool {
    static FLOOR: OnceLock<u64> = OnceLock::new();
    let floor = *FLOOR.get_or_init(|| {
        std::env::var("AURORA_DEVELOP_RESIDENT_MIN_PIXELS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|p| *p > 0)
            .unwrap_or(130_000)
    });
    (width as u64).saturating_mul(height as u64) >= floor
}

struct ResidentPool {
    // Device buffers and stream are declared BEFORE `lease` so Drop releases
    // VRAM before returning budget credit to the shared VramLedger.
    src_linear: Option<SendBuffer>,
    rgb3_work: Option<SendBuffer>,
    usm_cam: Option<SendBuffer>,
    usm_planes: Option<SendBuffer>,
    out_u8: Option<SendBuffer>,
    gain: Option<SendBuffer>,
    params: SendBuffer,
    luts: SendBuffer,
    c1: SendBuffer,
    editor: SendBuffer,
    banding: SendBuffer,
    usm_weights: SendBuffer,
    hist_dev: SendBuffer,
    pinned_in: Option<SendPinned>,
    pinned_out: Option<SendPinned>,
    pinned_params: SendPinned,
    pinned_weights: SendPinned,
    pinned_hist: SendPinned,
    stream: SendStream,
    gpu: Box<Nvidia>,
    src_bytes: usize,
    rgb3_bytes: usize,
    usm_cam_bytes: usize,
    usm_planes_bytes: usize,
    out_u8_bytes: usize,
    gain_bytes: usize,
    editor_bytes: usize,
    pinned_in_bytes: usize,
    pinned_out_bytes: usize,
    resident_key: Option<ResidentSourceKey>,
    tables: TableCache,
    last_usm_sigma_bits: Option<u32>,
    lease: VramLease,
}

impl ResidentPool {
    fn fixed_resident_bytes() -> usize {
        fixed_bytes() + USM_MAX_WEIGHTS * 4 + HIST_WORDS * 4
    }

    fn open() -> Result<Self, CudaError> {
        let editor_bytes = E_SKIN_BASE * 4;
        let base_bytes = Self::fixed_resident_bytes() + editor_bytes;
        let lease = VramLease::acquire("develop_resident", base_bytes, true).map_err(budget_error)?;
        let gpu = Box::new(Nvidia::open_ptx(PTX)?);
        let stream = SendStream(create_stream_static(&gpu)?);
        Ok(Self {
            src_linear: None,
            rgb3_work: None,
            usm_cam: None,
            usm_planes: None,
            out_u8: None,
            gain: None,
            params: SendBuffer(alloc_static(&gpu, P_SIZE * 4)?),
            luts: SendBuffer(alloc_static(&gpu, (1280 + GAMMA_SIZE) * 4)?),
            c1: SendBuffer(alloc_static(&gpu, FILM_MAX * 3 * 4 + CLUT_NODES * 3 * 2)?),
            editor: SendBuffer(alloc_static(&gpu, editor_bytes)?),
            banding: SendBuffer(alloc_static(&gpu, 256 * 4 + 512)?),
            usm_weights: SendBuffer(alloc_static(&gpu, USM_MAX_WEIGHTS * 4)?),
            hist_dev: SendBuffer(alloc_static(&gpu, HIST_WORDS * 4)?),
            pinned_in: None,
            pinned_out: None,
            pinned_params: SendPinned(alloc_pinned_static(&gpu, P_SIZE * 4)?),
            pinned_weights: SendPinned(alloc_pinned_static(&gpu, USM_MAX_WEIGHTS * 4)?),
            pinned_hist: SendPinned(alloc_pinned_static(&gpu, HIST_WORDS * 4)?),
            stream,
            gpu,
            src_bytes: 0,
            rgb3_bytes: 0,
            usm_cam_bytes: 0,
            usm_planes_bytes: 0,
            out_u8_bytes: 0,
            gain_bytes: 0,
            editor_bytes,
            pinned_in_bytes: 0,
            pinned_out_bytes: 0,
            resident_key: None,
            tables: TableCache::default(),
            last_usm_sigma_bits: None,
            lease,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure(
        &mut self,
        need_src: usize,
        need_rgb3: usize,
        need_usm_cam: usize,
        need_usm_planes: usize,
        need_out_u8: usize,
        need_gain: usize,
        need_editor: usize,
    ) -> Result<(), CudaError> {
        let next_src = need_src.max(self.src_bytes);
        let next_rgb3 = need_rgb3.max(self.rgb3_bytes);
        let next_usm_cam = need_usm_cam.max(self.usm_cam_bytes);
        let next_usm_planes = need_usm_planes.max(self.usm_planes_bytes);
        let next_out_u8 = need_out_u8.max(self.out_u8_bytes);
        let next_gain = need_gain.max(self.gain_bytes);
        let next_editor = need_editor.max(self.editor_bytes);

        let total = Self::fixed_resident_bytes()
            .saturating_add(next_src)
            .saturating_add(next_rgb3)
            .saturating_add(next_usm_cam)
            .saturating_add(next_usm_planes)
            .saturating_add(next_out_u8)
            .saturating_add(next_gain)
            .saturating_add(next_editor);

        let old_total = self.lease.bytes();
        if total > old_total {
            self.lease.resize(total, true).map_err(budget_error)?;
        }
        let alloc_res: Result<(), CudaError> = (|| {
            if next_src > self.src_bytes {
                self.src_linear = Some(SendBuffer(alloc_static(&self.gpu, next_src)?));
                self.src_bytes = next_src;
                self.resident_key = None;
            }
            if next_rgb3 > self.rgb3_bytes {
                self.rgb3_work = Some(SendBuffer(alloc_static(&self.gpu, next_rgb3)?));
                self.rgb3_bytes = next_rgb3;
            }
            if next_usm_cam > self.usm_cam_bytes {
                self.usm_cam = Some(SendBuffer(alloc_static(&self.gpu, next_usm_cam)?));
                self.usm_cam_bytes = next_usm_cam;
            }
            if next_usm_planes > self.usm_planes_bytes {
                self.usm_planes = Some(SendBuffer(alloc_static(&self.gpu, next_usm_planes)?));
                self.usm_planes_bytes = next_usm_planes;
            }
            if next_out_u8 > self.out_u8_bytes {
                self.out_u8 = Some(SendBuffer(alloc_static(&self.gpu, next_out_u8)?));
                self.out_u8_bytes = next_out_u8;
            }
            if next_gain > self.gain_bytes {
                self.gain = Some(SendBuffer(alloc_static(&self.gpu, next_gain)?));
                self.gain_bytes = next_gain;
            }
            if next_editor > self.editor_bytes {
                self.editor = SendBuffer(alloc_static(&self.gpu, next_editor)?);
                self.editor_bytes = next_editor;
                self.tables.editor = None;
            }
            if need_out_u8 > self.pinned_out_bytes {
                self.pinned_out = Some(SendPinned(alloc_pinned_static(&self.gpu, need_out_u8)?));
                self.pinned_out_bytes = need_out_u8;
            }
            Ok(())
        })();
        if alloc_res.is_err() && total > old_total {
            let _ = self.lease.resize(old_total, true);
        }
        alloc_res
    }

    fn ensure_pinned_in(&mut self, bytes: usize) -> Result<&mut [u8], CudaError> {
        if bytes > self.pinned_in_bytes {
            self.pinned_in = Some(SendPinned(alloc_pinned_static(&self.gpu, bytes)?));
            self.pinned_in_bytes = bytes;
        }
        Ok(&mut self.pinned_in.as_mut().unwrap().0.as_mut_slice()[..bytes])
    }

    fn upload_static_tables(&mut self, params: &DevelopParams) -> Result<(), CudaError> {
        let luts = luts_parts(params);
        if !source_matches(self.tables.luts.as_deref(), &luts) {
            let table = luts_floats(params);
            let outcome = self.gpu.upload_buffer(&self.luts.0, as_bytes(&table));
            commit_upload(&mut self.tables.luts, &luts, outcome)?;
        }
        let editor = editor_parts(params);
        if !source_matches(self.tables.editor.as_deref(), &editor) {
            let table = params
                .editor
                .as_ref()
                .map(|e| e.floats.clone())
                .unwrap_or_else(|| vec![0.0; E_SKIN_BASE]);
            let outcome = self.gpu.upload_buffer(&self.editor.0, as_bytes(&table));
            commit_upload(&mut self.tables.editor, &editor, outcome)?;
        }
        let banding = banding_parts(params);
        if !source_matches(self.tables.banding.as_deref(), &banding) {
            let table = params
                .banding
                .as_ref()
                .map(banding_bytes)
                .unwrap_or_else(|| vec![0u8; 256 * 4 + 512]);
            let outcome = self.gpu.upload_buffer(&self.banding.0, &table);
            commit_upload(&mut self.tables.banding, &banding, outcome)?;
        }
        if let Some(stage) = params.c1.as_ref() {
            let c1 = c1_parts(params);
            if !source_matches(self.tables.c1.as_deref(), &c1) {
                let table = c1_bytes(stage);
                let outcome = self.gpu.upload_buffer(&self.c1.0, &table);
                commit_upload(&mut self.tables.c1, &c1, outcome)?;
            }
        }
        Ok(())
    }

    fn upload_usm_weights(&mut self, sigma: f32) -> Result<i32, CudaError> {
        let half = ((3.0 * sigma).ceil() as usize).clamp(1, 24);
        let bits = sigma.to_bits();
        if self.last_usm_sigma_bits != Some(bits) {
            let mut weights = [0.0f32; USM_MAX_WEIGHTS];
            let inv2s2 = 1.0 / (2.0 * sigma * sigma);
            let mut sum = 1.0f32;
            weights[0] = 1.0;
            for k in 1..=half {
                let wk = (-((k * k) as f32) * inv2s2).exp();
                weights[k] = wk;
                sum += 2.0 * wk;
            }
            let inv_sum = 1.0 / sum;
            for w in &mut weights[..=half] {
                *w *= inv_sum;
            }
            let bytes = as_bytes(&weights);
            self.pinned_weights.0.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
            self.gpu.upload_async(
                &self.usm_weights.0,
                &self.pinned_weights.0.as_slice()[..bytes.len()],
                &self.stream.0,
            )?;
            self.last_usm_sigma_bits = Some(bits);
        }
        Ok(half as i32)
    }

    fn retained_bytes(&self) -> usize {
        self.lease.bytes()
    }
}

enum ResidentState {
    NotTried,
    Ready(ResidentPool),
    Unavailable,
}

fn resident_state() -> &'static Mutex<ResidentState> {
    static STATE: OnceLock<Mutex<ResidentState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(ResidentState::NotTried))
}

fn release_resident() -> usize {
    let mut state = match resident_state().try_lock() {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let freed = match &*state {
        ResidentState::Ready(pool) => pool.retained_bytes(),
        _ => 0,
    };
    if freed > 0 {
        *state = ResidentState::NotTried;
    }
    freed
}

fn register_resident_reclaimer() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        gpu_runtime::vram_register_reclaimer(
            "develop_resident",
            RESIDENT_RECLAIM_PRIORITY,
            release_resident,
        )
    });
}

fn launch_stream<const N: usize>(
    gpu: &Nvidia,
    stream: &Stream<'_>,
    kernel: &'static str,
    width: usize,
    height: usize,
    mut args: [*mut c_void; N],
) -> Result<(), CudaError> {
    let grid = (
        (width as u32).div_ceil(BLOCK.0),
        (height as u32).div_ceil(BLOCK.1),
    );
    gpu.launch_raw_stream(kernel, grid, BLOCK, &mut args, stream)?;
    Ok(())
}

fn launch_stream_1d<const N: usize>(
    gpu: &Nvidia,
    stream: &Stream<'_>,
    kernel: &'static str,
    n: usize,
    mut args: [*mut c_void; N],
) -> Result<(), CudaError> {
    let grid = ((n as u32).div_ceil(256), 1);
    gpu.launch_raw_stream(kernel, grid, (256, 1), &mut args, stream)?;
    Ok(())
}

fn histogram_from_words(words: &[u32], pixel_count: usize) -> Histogram {
    let total = pixel_count.max(1) as f32;
    Histogram {
        red: words[0..256].to_vec(),
        green: words[256..512].to_vec(),
        blue: words[512..768].to_vec(),
        luma: words[768..1024].to_vec(),
        clipped_shadows: words[1024] as f32 / total,
        clipped_highlights: words[1025] as f32 / total,
    }
}

/// Executes the full preview chain (develop + optional Camera USM / C1 ICC +
/// optional GPU NR + optional Display USM + quantize + GPU histogram) on a
/// single CUDA stream using a VRAM-resident linear source frame.
///
/// When `key` matches the resident source, `fill_source` is NOT called: zero
/// CPU sampling and zero H2D pixel transfer occur.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_resident_preview<F>(
    key: ResidentSourceKey,
    region: [f32; 4],
    params: &DevelopParams,
    gain_plane: Option<&[f32]>,
    camera_usm: Option<UsmParams>,
    display_usm: Option<UsmParams>,
    nr: Option<(&NoiseReduction, f32)>,
    bgra: bool,
    with_histogram: bool,
    fill_source: F,
) -> Option<ResidentPreviewOut>
where
    F: FnOnce(&mut [f32]),
{
    if !enabled() {
        return None;
    }
    let width = key.out_w as usize;
    let height = key.out_h as usize;
    let pixels = width.checked_mul(height)?;
    if pixels == 0 || !resident_crossover_admits(width, height) {
        return None;
    }
    if gain_plane.is_some_and(|g| g.len() != pixels) {
        return None;
    }
    let has_single_pixel = nr.is_some_and(|(n, _)| n.single_pixel > 0.0);
    let has_nr = nr.is_some_and(|(n, _)| n.luminance > 0.0 || n.color > 0.0);
    if has_nr && !super::nr_gpu::enabled() {
        return None;
    }
    let needs_rgb3 = camera_usm.is_some() || display_usm.is_some() || has_nr || has_single_pixel;
    let need_src = pixels.checked_mul(16)?;
    let need_rgb3 = if needs_rgb3 { pixels.checked_mul(12)? } else { 0 };
    let need_usm_cam = if camera_usm.is_some() { pixels.checked_mul(16)? } else { 0 };
    let need_usm_planes = if camera_usm.is_some() || display_usm.is_some() {
        pixels.checked_mul(12)?
    } else if has_single_pixel {
        pixels.checked_mul(4)?
    } else {
        0
    };
    let need_out_u8 = pixels.checked_mul(4)?;
    let need_gain = if gain_plane.is_some() { pixels.checked_mul(4)? } else { 0 };
    let need_editor = params
        .editor
        .as_ref()
        .map(|e| e.floats.len().saturating_mul(4))
        .unwrap_or(0)
        .max(E_SKIN_BASE * 4);

    register_resident_reclaimer();
    let mut lock = resident_state().lock().unwrap_or_else(|e| e.into_inner());
    let pool = match &mut *lock {
        ResidentState::Unavailable => return None,
        ResidentState::NotTried => match ResidentPool::open() {
            Ok(p) => {
                *lock = ResidentState::Ready(p);
                match &mut *lock {
                    ResidentState::Ready(p) => p,
                    _ => unreachable!(),
                }
            }
            Err(err) => {
                log::warn!("develop_gpu resident: {err:?}; falling back");
                if !budget_refused(&err) {
                    *lock = ResidentState::Unavailable;
                }
                return None;
            }
        },
        ResidentState::Ready(p) => p,
    };

    if let Err(err) = pool.ensure(
        need_src,
        need_rgb3,
        need_usm_cam,
        need_usm_planes,
        need_out_u8,
        need_gain,
        need_editor,
    ) {
        log::warn!("develop_gpu resident ensure: {err:?}; falling back");
        return None;
    }

    let result: Result<ResidentPreviewOut, CudaError> = (|| {
        pool.upload_static_tables(params)?;

        // 1. Source linear frame: reuse resident VRAM on slider drags!
        let source_hit = pool.resident_key == Some(key);
        if !source_hit {
            let pinned_bytes = pool.ensure_pinned_in(need_src)?;
            let pinned_f32 = unsafe {
                std::slice::from_raw_parts_mut(pinned_bytes.as_mut_ptr() as *mut f32, pixels * 4)
            };
            fill_source(pinned_f32);
            let src_dev = &pool.src_linear.as_ref().unwrap().0;
            let pinned_slice = &pool.pinned_in.as_ref().unwrap().0.as_slice()[..need_src];
            pool.gpu.upload_async(src_dev, pinned_slice, &pool.stream.0)?;
            pool.resident_key = Some(key);
        }

        // 2. Upload 488-byte parameter block asynchronously on the stream.
        let mut floats = params_floats(params);
        floats[P_W] = width as f32;
        floats[P_H] = height as f32;
        floats[P_RX] = region[0];
        floats[P_RY] = region[1];
        floats[P_RW] = region[2];
        floats[P_RH] = region[3];
        floats[P_HAS_GAIN] = if gain_plane.is_some() { 1.0 } else { 0.0 };
        let p_bytes = as_bytes(&floats);
        pool.pinned_params.0.as_mut_slice()[..p_bytes.len()].copy_from_slice(p_bytes);
        pool.gpu.upload_async(
            &pool.params.0,
            &pool.pinned_params.0.as_slice()[..p_bytes.len()],
            &pool.stream.0,
        )?;

        if let Some(plane) = gain_plane {
            let gain_dev = &pool.gain.as_ref().unwrap().0;
            pool.gpu.upload_buffer(gain_dev, as_bytes(plane))?;
        }

        let mut src_handle = pool.src_linear.as_ref().unwrap().0.handle();
        let mut out_u8_handle = pool.out_u8.as_ref().unwrap().0.handle();
        let mut hist_handle = if with_histogram { pool.hist_dev.0.handle() } else { 0u64 };
        let mut p_handle = pool.params.0.handle();
        let mut lut_handle = pool.luts.0.handle();
        let mut gain_handle = if gain_plane.is_some() {
            pool.gain.as_ref().unwrap().0.handle()
        } else {
            0u64
        };
        let mut c1_handle = if params.c1.is_some() { pool.c1.0.handle() } else { 0u64 };
        let mut editor_handle = pool.editor.0.handle();
        let mut words = pool.banding.0.handle();
        let mut left = words + 1024;
        let mut right = words + 1280;
        let mut bgra_i = if bgra { 1i32 } else { 0i32 };
        let mut w_i = width as i32;
        let mut h_i = height as i32;

        if !needs_rgb3 {
            // Fast 1-kernel path: develop -> quantize (BGRA/RGBA) + GPU histogram.
            if with_histogram {
                let mut n_hist = HIST_WORDS as i32;
                launch_stream_1d(
                    &pool.gpu,
                    &pool.stream.0,
                    "zero_u32_kernel",
                    HIST_WORDS,
                    [ptr(&mut hist_handle), ptr(&mut n_hist)],
                )?;
            }
            launch_stream(
                &pool.gpu,
                &pool.stream.0,
                "develop_resident_u8_hist_kernel",
                width,
                height,
                [
                    ptr(&mut src_handle),
                    ptr(&mut out_u8_handle),
                    ptr(&mut hist_handle),
                    ptr(&mut p_handle),
                    ptr(&mut lut_handle),
                    ptr(&mut gain_handle),
                    ptr(&mut c1_handle),
                    ptr(&mut editor_handle),
                    ptr(&mut words),
                    ptr(&mut left),
                    ptr(&mut right),
                    ptr(&mut bgra_i),
                ],
            )?;
        } else {
            let mut rgb3_handle = pool.rgb3_work.as_ref().unwrap().0.handle();
            if let Some(usm) = camera_usm {
                let mut half_i = pool.upload_usm_weights(usm.sigma)?;
                let mut cam_handle = pool.usm_cam.as_ref().unwrap().0.handle();
                let planes_base = pool.usm_planes.as_ref().unwrap().0.handle();
                let plane_bytes = (pixels * 4) as u64;
                let mut norm_handle = planes_base;
                let mut tmp_handle = planes_base + plane_bytes;
                let mut blur_handle = planes_base + plane_bytes * 2;
                let mut weights_handle = pool.usm_weights.0.handle();

                launch_stream(
                    &pool.gpu,
                    &pool.stream.0,
                    "develop_camera_encode_kernel",
                    width,
                    height,
                    [
                        ptr(&mut src_handle),
                        ptr(&mut cam_handle),
                        ptr(&mut norm_handle),
                        ptr(&mut p_handle),
                        ptr(&mut gain_handle),
                        ptr(&mut c1_handle),
                    ],
                )?;
                launch_stream(
                    &pool.gpu,
                    &pool.stream.0,
                    "gaussian_blur_h_kernel",
                    width,
                    height,
                    [
                        ptr(&mut norm_handle),
                        ptr(&mut tmp_handle),
                        ptr(&mut weights_handle),
                        ptr(&mut w_i),
                        ptr(&mut h_i),
                        ptr(&mut half_i),
                    ],
                )?;
                launch_stream(
                    &pool.gpu,
                    &pool.stream.0,
                    "gaussian_blur_v_kernel",
                    width,
                    height,
                    [
                        ptr(&mut tmp_handle),
                        ptr(&mut blur_handle),
                        ptr(&mut weights_handle),
                        ptr(&mut w_i),
                        ptr(&mut h_i),
                        ptr(&mut half_i),
                    ],
                )?;
                let mut g_usm = usm.gain;
                let mut t_usm = usm.threshold;
                let mut h_usm = usm.halo;
                launch_stream(
                    &pool.gpu,
                    &pool.stream.0,
                    "camera_usm_and_display_kernel",
                    width,
                    height,
                    [
                        ptr(&mut cam_handle),
                        ptr(&mut norm_handle),
                        ptr(&mut blur_handle),
                        ptr(&mut rgb3_handle),
                        ptr(&mut p_handle),
                        ptr(&mut lut_handle),
                        ptr(&mut c1_handle),
                        ptr(&mut editor_handle),
                        ptr(&mut words),
                        ptr(&mut left),
                        ptr(&mut right),
                        ptr(&mut g_usm),
                        ptr(&mut t_usm),
                        ptr(&mut h_usm),
                    ],
                )?;
            } else {
                launch_stream(
                    &pool.gpu,
                    &pool.stream.0,
                    "develop_resident_rgb3_kernel",
                    width,
                    height,
                    [
                        ptr(&mut src_handle),
                        ptr(&mut rgb3_handle),
                        ptr(&mut p_handle),
                        ptr(&mut lut_handle),
                        ptr(&mut gain_handle),
                        ptr(&mut c1_handle),
                        ptr(&mut editor_handle),
                        ptr(&mut words),
                        ptr(&mut left),
                        ptr(&mut right),
                    ],
                )?;
            }

            // Optional single-pixel noise reduction in place on `rgb3_work`, same stream.
            if let Some((nr_adj, _)) = nr {
                let amount = (nr_adj.single_pixel / 100.0).clamp(0.0, 1.0);
                if amount > 0.0 {
                    let mut luma_handle = pool.usm_planes.as_ref().unwrap().0.handle();
                    let mut threshold = 24.0f32 - 20.0f32 * amount;
                    launch_stream(
                        &pool.gpu,
                        &pool.stream.0,
                        "single_pixel_luma_kernel",
                        width,
                        height,
                        [
                            ptr(&mut rgb3_handle),
                            ptr(&mut luma_handle),
                            ptr(&mut w_i),
                            ptr(&mut h_i),
                        ],
                    )?;
                    launch_stream(
                        &pool.gpu,
                        &pool.stream.0,
                        "single_pixel_nr_apply_kernel",
                        width,
                        height,
                        [
                            ptr(&mut rgb3_handle),
                            ptr(&mut luma_handle),
                            ptr(&mut w_i),
                            ptr(&mut h_i),
                            ptr(&mut threshold),
                        ],
                    )?;
                }
            }

            // Cross-stage zero-copy NR in place on `rgb3_work`, same stream.
            if let Some((nr_adj, scale)) = nr {
                if nr_adj.luminance > 0.0 || nr_adj.color > 0.0 {
                    if !super::nr_gpu::run_resident_on_stream(
                        rgb3_handle,
                        width,
                        height,
                        nr_adj,
                        scale,
                        &pool.stream.0,
                    ) {
                        return Err(CudaError::Unavailable("nr_gpu resident failed"));
                    }
                }
            }

            // Optional display-space USM in place on `rgb3_work`, same stream.
            if let Some(usm) = display_usm {
                let mut half_i = pool.upload_usm_weights(usm.sigma)?;
                let planes_base = pool.usm_planes.as_ref().unwrap().0.handle();
                let plane_bytes = (pixels * 4) as u64;
                let mut luma_handle = planes_base;
                let mut tmp_handle = planes_base + plane_bytes;
                let mut blur_handle = planes_base + plane_bytes * 2;
                let mut weights_handle = pool.usm_weights.0.handle();

                launch_stream(
                    &pool.gpu,
                    &pool.stream.0,
                    "display_usm_luma_kernel",
                    width,
                    height,
                    [
                        ptr(&mut rgb3_handle),
                        ptr(&mut luma_handle),
                        ptr(&mut w_i),
                        ptr(&mut h_i),
                    ],
                )?;
                launch_stream(
                    &pool.gpu,
                    &pool.stream.0,
                    "gaussian_blur_h_kernel",
                    width,
                    height,
                    [
                        ptr(&mut luma_handle),
                        ptr(&mut tmp_handle),
                        ptr(&mut weights_handle),
                        ptr(&mut w_i),
                        ptr(&mut h_i),
                        ptr(&mut half_i),
                    ],
                )?;
                launch_stream(
                    &pool.gpu,
                    &pool.stream.0,
                    "gaussian_blur_v_kernel",
                    width,
                    height,
                    [
                        ptr(&mut tmp_handle),
                        ptr(&mut blur_handle),
                        ptr(&mut weights_handle),
                        ptr(&mut w_i),
                        ptr(&mut h_i),
                        ptr(&mut half_i),
                    ],
                )?;
                let mut g_usm = usm.gain;
                let mut t_usm = usm.threshold;
                let mut h_usm = usm.halo;
                launch_stream(
                    &pool.gpu,
                    &pool.stream.0,
                    "display_usm_apply_kernel",
                    width,
                    height,
                    [
                        ptr(&mut rgb3_handle),
                        ptr(&mut luma_handle),
                        ptr(&mut blur_handle),
                        ptr(&mut w_i),
                        ptr(&mut h_i),
                        ptr(&mut g_usm),
                        ptr(&mut t_usm),
                        ptr(&mut h_usm),
                    ],
                )?;
            }

            // Final quantize (BGRA/RGBA) + GPU histogram.
            if with_histogram {
                let mut n_hist = HIST_WORDS as i32;
                launch_stream_1d(
                    &pool.gpu,
                    &pool.stream.0,
                    "zero_u32_kernel",
                    HIST_WORDS,
                    [ptr(&mut hist_handle), ptr(&mut n_hist)],
                )?;
            }
            launch_stream(
                &pool.gpu,
                &pool.stream.0,
                "quantize_rgb3_u8_hist_kernel",
                width,
                height,
                [
                    ptr(&mut rgb3_handle),
                    ptr(&mut out_u8_handle),
                    ptr(&mut hist_handle),
                    ptr(&mut w_i),
                    ptr(&mut h_i),
                    ptr(&mut bgra_i),
                ],
            )?;
        }

        // Single async D2H of the compact 8-bit frame + 4 KB histogram, then one sync.
        let out_dev = &pool.out_u8.as_ref().unwrap().0;
        let pinned_out = &mut pool.pinned_out.as_mut().unwrap().0.as_mut_slice()[..need_out_u8];
        pool.gpu.download_async(out_dev, pinned_out, &pool.stream.0)?;
        if with_histogram {
            let hist_slice = &mut pool.pinned_hist.0.as_mut_slice()[..HIST_WORDS * 4];
            pool.gpu.download_async(&pool.hist_dev.0, hist_slice, &pool.stream.0)?;
        }
        pool.gpu.stream_sync(&pool.stream.0)?;

        #[cfg(test)]
        GPU_PLANES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut out_vec = vec![0u8; need_out_u8];
        out_vec.copy_from_slice(&pool.pinned_out.as_ref().unwrap().0.as_slice()[..need_out_u8]);
        let histogram = if with_histogram {
            let hist_words = unsafe {
                std::slice::from_raw_parts(
                    pool.pinned_hist.0.as_slice().as_ptr() as *const u32,
                    HIST_WORDS,
                )
            };
            Some(histogram_from_words(hist_words, pixels))
        } else {
            None
        };
        Ok(ResidentPreviewOut {
            pixels: out_vec,
            histogram,
            source_hit,
        })
    })();

    gpu_runtime::vram_maybe_emit_perf("develop_resident");
    match result {
        Ok(out) => Some(out),
        Err(err) => {
            log::warn!("develop_gpu resident run: {err:?}; falling back to CPU");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code8(v: f32) -> i32 {
        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as i32
    }

    fn compare(cpu: &[[f32; 3]], gpu: &[[f32; 3]]) -> (i32, f32, usize) {
        let (mut max_code, mut max_abs, mut over_one) = (0i32, 0.0f32, 0usize);
        for (a, b) in cpu.iter().zip(gpu) {
            for k in 0..3 {
                let d = (code8(a[k]) - code8(b[k])).abs();
                max_code = max_code.max(d);
                max_abs = max_abs.max((a[k] - b[k]).abs());
                if d > 1 {
                    over_one += 1;
                }
            }
        }
        (max_code, max_abs, over_one)
    }

    // Band-path retained ledger: the develop pool is band-shaped, so the
    // whole-frame numbers must not decide whether a band may run.

    #[test]
    fn band_ledger_describes_the_band_not_the_whole_frame() {
        let (w, h) = (8192usize, 5464usize);
        let band = band_device_bytes_with(w, 256, 0).expect("band ledger");
        // 45 MP at 256 rows: ~40.7 MiB, dominated by the 20 B/px band.
        assert!(band <= 64 * 1024 * 1024, "45 MP band {band} must stay ~40 MiB");
        let whole = band_device_bytes_with(w, h, 0).expect("frame ledger");
        // The whole-frame shape stays an order of magnitude larger (~854 MiB):
        // `develop_plane`'s whole-frame admission keeps refusing it.
        assert!(whole > 800 * 1024 * 1024, "whole-frame ledger {whole} must stay large");
    }

    #[test]
    fn band_host_ledger_counts_the_full_gain_plane() {
        let (w, h, rows) = (8192usize, 5464usize, 256usize);
        let host = band_host_bytes_with(w, rows, h).expect("host ledger");
        assert!(host >= w * h * 4, "the full-frame gain plane is counted");
        // 45 MP: gain plane 179 MiB + band IO 32 MiB ≈ 202.8 MiB.
        assert!(host <= 256 * 1024 * 1024, "45 MP host ledger {host} must fit");
    }

    #[test]
    fn the_odd_tail_band_is_cheaper_than_a_full_band() {
        let (w, h) = (6048usize, 4032usize);
        let tail = h % 256;
        assert!(tail > 0 && tail < 256, "4032 leaves a 192-row tail");
        let full = band_device_bytes_with(w, 256, 0).expect("full band");
        let odd = band_device_bytes_with(w, tail, 0).expect("tail band");
        assert!(odd < full);
    }

    #[test]
    fn a_very_wide_band_still_fits_both_ledgers() {
        let (w, h) = (40_000usize, 2_000usize);
        let device = band_device_bytes_with(w, 256, 0).expect("device ledger");
        let host = band_host_bytes_with(w, 256, h).expect("host ledger");
        assert!(device < 256 * 1024 * 1024, "40k-wide band device ledger {device}");
        assert!(host < 512 * 1024 * 1024, "40k-wide band host ledger {host}");
    }

    #[test]
    fn the_band_ledger_refuses_overflow() {
        assert!(band_device_bytes_with(usize::MAX / 8, 256, 0).is_none());
        assert!(band_host_bytes_with(usize::MAX / 4, 1, 2).is_none());
        assert!(band_device_bytes_with(0, 256, 0).is_none());
        assert!(band_host_bytes_with(1024, 0, 2).is_none());
    }

    fn gradient(w: usize, h: usize) -> Vec<Sample> {
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 37 + y * 11) % 173) as f32 / 128.0;
                let g = ((x * 19 + y * 43) % 151) as f32 / 128.0;
                let b = ((x * 7 + y * 29) % 139) as f32 / 128.0;
                let tint = ((x + y) % 97) as f32 / 96.0;
                out.push(Sample {
                    rgb: [
                        (r * 0.8 + tint * 0.5).min(1.6),
                        (g * 0.7 + (1.0 - tint) * 0.35).min(1.6),
                        (b * 0.9 + tint * 0.2).min(1.6),
                    ],
                    valid: (x + y) % 61 != 0,
                });
            }
        }
        out
    }

    fn run(
        samples: &[Sample],
        w: usize,
        h: usize,
        region: [f32; 4],
        params: &DevelopParams,
        gain: Option<&[f32]>,
        label: &str,
    ) {
        if !prewarm() {
            eprintln!("skipping {label}: CUDA develop device/PTX unavailable (AURORA_DEVELOP_GPU)");
            return;
        }
        let cpu = develop_plane_cpu(samples, w, h, region, params, gain);
        let mut gpu = vec![[0.0f32; 3]; w * h];
        assert!(
            develop_plane(samples, w, h, region, params, gain, &mut gpu),
            "{label}: develop_plane refused the frame"
        );
        let (max_code, max_abs, over_one) = compare(&cpu, &gpu);
        println!(
            "{label}: max_code8={max_code} max_abs={max_abs:.6} channels_over_1={over_one}"
        );
        assert!(max_code <= 1, "{label}: max code8 {max_code} exceeds 1");
    }

    fn tone_color_adjustments() -> AdjustmentSet {
        let mut adj = AdjustmentSet::default();
        adj.exposure.exposure = 0.35;
        adj.exposure.brightness = 14.0;
        adj.exposure.contrast = 22.0;
        adj.exposure.saturation = 18.0;
        adj.hdr = HighDynamicRange { highlights: -35.0, shadows: 28.0, whites: 12.0, blacks: -9.0 };
        adj.dehaze = Dehaze {
            amount: 55.0,
            shadow_tone: [0.62, 0.66, 0.71],
            level_low: [0.02, 0.03, 0.04],
            level_high: [0.96, 0.95, 0.94],
        };
        adj.levels.rgb.input_black = 0.02;
        adj.levels.rgb.input_white = 0.94;
        adj.levels.rgb.gamma = 1.15;
        adj.levels.rgb.output_black = 0.01;
        adj.levels.rgb.output_white = 0.98;
        adj.levels.red.input_black = 0.01;
        adj.levels.red.input_white = 0.97;
        adj.levels.red.gamma = 0.92;
        adj.levels.blue.gamma = 1.21;
        adj.curves.rgb = Curve { points: vec![CurvePoint::new(0.0, 0.0), CurvePoint::new(0.4, 0.48), CurvePoint::new(1.0, 1.0)] };
        adj.curves.red = Curve { points: vec![CurvePoint::new(0.0, 0.03), CurvePoint::new(0.6, 0.66), CurvePoint::new(1.0, 0.97)] };
        adj.curves.luma = Curve { points: vec![CurvePoint::new(0.0, 0.0), CurvePoint::new(0.5, 0.56), CurvePoint::new(1.0, 1.0)] };
        adj.color_balance.master = ColorWheel { hue: 210.0, saturation: 0.06, lightness: 0.02 };
        adj.color_balance.shadow = ColorWheel { hue: 30.0, saturation: 0.08, lightness: -0.01 };
        adj.color_balance.highlight = ColorWheel { hue: 55.0, saturation: 0.05, lightness: 0.01 };
        adj.color_balance_gains = Some(ColorBalanceGains {
            master: [1.02, 1.0, 0.98],
            shadow: [1.01, 0.99, 1.02],
            midtone: [1.0, 1.01, 0.99],
            highlight: [0.99, 1.0, 1.02],
        });
        adj
    }

    fn editor_adjustments() -> AdjustmentSet {
        let mut adj = AdjustmentSet::default();
        adj.exposure.exposure = 0.2;
        adj.exposure.saturation = 10.0;
        let mut ce = ColorEditor::default();
        ce.global.enabled = true;
        ce.global.smoothness = 12.0;
        ce.global_shift = HslShift { hue: 4.0, saturation: 8.0, lightness: 2.0 };
        ce.basic[3] = HslShift { hue: 12.0, saturation: 30.0, lightness: -8.0 };
        ce.bands[3].enabled = true;
        ce.advanced.push(ColorRange {
            id: 7,
            hue: 40.0,
            width: 55.0,
            smoothness: 12.0,
            min_saturation: 0.1,
            shift: HslShift { hue: -8.0, saturation: 18.0, lightness: 6.0 },
            uniformity: None,
        });
        ce.advanced_bands.push((
            7,
            ColorBand {
                enabled: true,
                center_hue: 40.0,
                center_saturation: 0.55,
                center_value: 0.6,
                hue_low: -28.0,
                hue_high: 30.0,
                sat_low: -55.0,
                sat_high: 40.0,
                smoothness: 14.0,
                homogeneity: [25.0, 15.0, 0.0],
                ..ColorBand::default()
            },
        ));
        ce.skin.push(ColorRange {
            id: 8,
            hue: 25.0,
            width: 45.0,
            smoothness: 15.0,
            min_saturation: 0.1,
            shift: HslShift { hue: 5.0, saturation: 12.0, lightness: 4.0 },
            uniformity: Some(SkinUniformity { hue: 40.0, saturation: 25.0, lightness: 15.0 }),
        });
        ce.skin.push(ColorRange {
            id: 9,
            hue: 25.0,
            width: 45.0,
            smoothness: 15.0,
            min_saturation: 0.1,
            shift: HslShift { hue: 2.0, saturation: 6.0, lightness: 3.0 },
            uniformity: Some(SkinUniformity { hue: 30.0, saturation: 20.0, lightness: 12.0 }),
        });
        ce.max_smooth.push(ColorRange {
            id: 10,
            hue: 125.0,
            width: 80.0,
            smoothness: 20.0,
            min_saturation: 0.05,
            shift: HslShift { hue: -6.0, saturation: -10.0, lightness: 5.0 },
            uniformity: None,
        });
        adj.color_editor = ce;
        adj.black_white = BlackWhite {
            enabled: true,
            sensitivity: [18.0, -22.0, 30.0, -15.0, 12.0, -8.0],
            highlight_tone: SplitTone { hue: 45.0, saturation: 30.0 },
            shadow_tone: SplitTone { hue: 220.0, saturation: 25.0 },
        };
        adj
    }

    fn banding_params(w: usize, h: usize) -> BandingParams {
        let mut left = [0u8; 256];
        let mut right = [0u8; 256];
        let mut packed = [0u8; 1024];
        for i in 0..256usize {
            left[i] = i as u8;
            right[i] = ((i * 7 + 13) & 0xFF) as u8;
            let half = ((i as u32) << 8) | (255 - i as u32);
            packed[i * 4..i * 4 + 4].copy_from_slice(&(half | (half << 16)).to_le_bytes());
        }
        BandingParams::from_raw(
            crate::c1_banding::params_for_amount(100.0),
            &left,
            &right,
            &packed,
            [w as f32, h as f32],
        )
        .expect("banding tables")
    }

    #[test]
    fn develop_gpu_matches_the_cpu_reference() {
        let (w, h) = (37usize, 23usize);
        let samples = gradient(w, h);
        let region = [0.125f32, 0.0625, 0.75, 0.875];
        let gain: Vec<f32> = (0..w * h)
            .map(|i| 0.85 + ((i * 13) % 31) as f32 / 100.0)
            .collect();

        let default = DevelopParams::from_adjustments(&AdjustmentSet::default(), [1.0; 3], 1.0, false);
        run(&samples, w, h, region, &default, None, "default");

        let tone = tone_color_adjustments();
        let params = DevelopParams::from_adjustments(&tone, [1.85, 1.0, 0.72], 2f32.powf(0.35), false);
        run(&samples, w, h, region, &params, Some(&gain), "tone+color+gamma-lut");

        let fine = DevelopParams::from_adjustments(&tone, [1.85, 1.0, 0.72], 2f32.powf(0.35), true);
        run(&samples, w, h, region, &fine, Some(&gain), "tone+color+fine-srgb");

        let editor = editor_adjustments();
        let params = DevelopParams::from_adjustments(&editor, [1.05, 1.0, 0.95], 2f32.powf(0.2), false);
        run(&samples, w, h, region, &params, None, "color-editor+bw");

        let mut banded = DevelopParams::from_adjustments(&tone, [1.2, 1.0, 0.9], 1.1, false);
        banded.banding = Some(banding_params(w, h));
        run(&samples, w, h, region, &banded, Some(&gain), "output-banding");

        let mut bypass = DevelopParams::from_adjustments(&tone, [1.2, 1.0, 0.9], 1.1, false);
        bypass.bypass = true;
        run(&samples, w, h, region, &bypass, None, "bypass");
    }

    /// A deterministic synthetic 49^3 Camera -> sRGB node table. The production
    /// table is built from the installed ICC profiles; phase 2 wires that path
    /// in, this test checks the kernel's matrix/film/couple/encode/type-1 CLUT
    /// chain against the CPU reference.
    fn synthetic_clut() -> Vec<[u16; 3]> {
        let mut nodes = Vec::with_capacity(CLUT_NODES);
        for r in 0..CLUT_N {
            for g in 0..CLUT_N {
                for b in 0..CLUT_N {
                    let (rf, gf, bf) = (
                        r as f32 / (CLUT_N - 1) as f32,
                        g as f32 / (CLUT_N - 1) as f32,
                        b as f32 / (CLUT_N - 1) as f32,
                    );
                    let base = 0.5 * rf + 0.35 * gf + 0.15 * bf;
                    let value = |v: f32| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                    nodes.push([
                        value(base.powf(0.9) + 0.08 * (rf - gf)),
                        value((base * 1.05).powf(0.85) + 0.05 * (gf - bf)),
                        value(base.powf(1.1) + 0.10 * (bf - rf)),
                    ]);
                }
            }
        }
        nodes
    }

    #[test]
    fn c1_colour_stage_matches_the_cpu_reference() {
        let samples = gradient(31, 19);
        let default = DevelopParams::from_adjustments(&AdjustmentSet::default(), [1.0; 3], 1.0, false);
        let c1 = C1ColorParams::new(Z6III_PRE_MATRIX, LINEAR_FILM.to_vec(), synthetic_clut())
            .expect("C1 params");
        let mut params = default.clone();
        params.c1 = Some(c1.clone());
        run(&samples, 31, 19, [0.0, 0.0, 1.0, 1.0], &params, None, "c1-colour-stage");

        // Nonlinear film exercises the cubic-Hermite path.
        let film = vec![
            [0.0, 0.0, 0.29],
            [0.18, 0.31, 0.74],
            [0.47, 0.69, 0.48],
            [1.0, 1.0, 0.22],
        ];
        let mut nonlinear = c1;
        nonlinear.film = film;
        let mut params = default;
        params.c1 = Some(nonlinear);
        run(&samples, 31, 19, [0.0, 0.0, 1.0, 1.0], &params, None, "c1-nonlinear-film");
    }

    // Static-table fingerprints: the device copy is reused only while the
    // source is byte for byte the one that produced it.

    fn table_params() -> DevelopParams {
        let mut params = DevelopParams::from_adjustments(
            &tone_color_adjustments(),
            [1.85, 1.0, 0.72],
            2f32.powf(0.35),
            false,
        );
        params.editor = Some(EditorParams::from_editor(&editor_adjustments().color_editor));
        params.banding = Some(banding_params(64, 48));
        params.c1 = C1ColorParams::new(IDENTITY_PRE_MATRIX, LINEAR_FILM.to_vec(), synthetic_clut());
        params
    }

    #[test]
    fn static_table_fingerprint_is_byte_exact() {
        let params = table_params();
        let luts = luts_parts(&params);
        let witness = build_witness(&luts);
        assert!(source_matches(Some(&witness[..]), &luts), "the same source must match");

        // One changed curve bit must invalidate the LUT table.
        let mut changed = params.clone();
        let curve = changed.curves[0].as_mut().expect("rgb curve");
        curve[97] = f32::from_bits(curve[97].to_bits() ^ 1);
        let changed_luts = luts_parts(&changed);
        assert!(!source_matches(Some(&witness[..]), &changed_luts), "one bit changed");

        // A whole missing curve is a different table too.
        let mut removed = params.clone();
        removed.curves[1] = None;
        assert!(!source_matches(Some(&witness[..]), &luts_parts(&removed)));
    }

    #[test]
    fn static_table_fingerprint_covers_editor_banding_and_c1() {
        let params = table_params();

        let editor = build_witness(&editor_parts(&params));
        let mut changed = params.clone();
        changed.editor.as_mut().expect("editor").floats[7] += 0.5;
        assert!(!source_matches(Some(&editor[..]), &editor_parts(&changed)));

        let banding = build_witness(&banding_parts(&params));
        let mut changed = params.clone();
        changed.banding.as_mut().expect("banding").words[3] ^= 1;
        assert!(!source_matches(Some(&banding[..]), &banding_parts(&changed)));

        let c1 = build_witness(&c1_parts(&params));
        let mut changed = params.clone();
        let node = changed.c1.as_mut().expect("c1");
        node.clut[42][1] = node.clut[42][1].wrapping_add(1);
        assert!(!source_matches(Some(&c1[..]), &c1_parts(&changed)));
    }

    #[test]
    fn static_table_fingerprint_never_matches_absent_copies() {
        let params = table_params();
        let luts = luts_parts(&params);
        assert!(!source_matches(None, &luts), "no resident copy yet");

        // A truncated witness must not be accepted as the full source.
        let mut witness = build_witness(&luts);
        witness.pop();
        assert!(!source_matches(Some(&witness[..]), &luts));
    }

    #[test]
    fn failed_upload_drops_the_previous_witness() {
        // A -> B fails -> A: the pool survives an error on the `State::Ready`
        // path, so the failed copy must not leave A's witness in place.
        let params = table_params();
        let luts_a = luts_parts(&params);
        let mut witness = None;
        commit_upload(&mut witness, &luts_a, Ok(())).expect("A uploads");
        assert!(source_matches(witness.as_deref(), &luts_a));

        let mut params_b = params.clone();
        params_b.curves[0] = None;
        let luts_b = luts_parts(&params_b);
        assert!(!source_matches(witness.as_deref(), &luts_b), "B is a new table");

        let failed = commit_upload(
            &mut witness,
            &luts_b,
            Err(CudaError::Unavailable("injected copy failure")),
        );
        assert!(failed.is_err());
        assert!(witness.is_none(), "a failed copy must not claim any table");
        assert!(
            !source_matches(witness.as_deref(), &luts_a),
            "asking for A again must upload instead of hitting"
        );

        // A successful retry re-establishes exactly A's witness.
        commit_upload(&mut witness, &luts_a, Ok(())).expect("A re-uploads");
        assert!(source_matches(witness.as_deref(), &luts_a));
    }

    #[test]
    fn static_tables_ignore_per_band_parameters() {
        // Exposure, region and frame size live in the parameter table, which is
        // uploaded for every band. They must never invalidate a static table.
        let params = table_params();
        let mut changed = params.clone();
        changed.exposure *= 1.5;
        changed.wb = [1.2, 0.9, 1.05];
        changed.contrast += 0.1;

        assert_eq!(build_witness(&luts_parts(&params)), build_witness(&luts_parts(&changed)));
        assert_eq!(
            build_witness(&editor_parts(&params)),
            build_witness(&editor_parts(&changed))
        );
        assert_eq!(
            build_witness(&banding_parts(&params)),
            build_witness(&banding_parts(&changed))
        );
        assert_eq!(build_witness(&c1_parts(&params)), build_witness(&c1_parts(&changed)));

        // The per-band parameter table itself does change, so the dynamic
        // values are still uploaded with every band.
        assert_ne!(params_floats(&params), params_floats(&changed));
    }
}
