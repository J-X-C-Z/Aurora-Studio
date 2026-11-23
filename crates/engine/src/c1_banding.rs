//! Capture One BandingSuppression variant 1 for packed 16-bit RGB/RGBA.
//!
//! The reference is the OpenCL kernel `p_041123744.cl` (entry `zba817fd993`)
//! shipped inside the installed ImgCoreDll.dll. It perturbs R, G and B by one
//! signed delta read from a 256x256 periodic table built from three immutable
//! buffers of 256, 256 and 1024 bytes.
//!
//! The public Capture One API stores the amount as `int` in
//! `IC_ImageProcessSettings.bandingReductionAmount`, but the render layer keeps
//! it as float32 (`settings[layer*0x148 + 0x7c]`, copied to image `+0x11054`)
//! and uses it directly as the amplitude; there is no tick scaling. The Z6III
//! info block reports `IB_F_BANDINGREDUCTIONAMOUNT_START/END = 0/0`, so on this
//! camera the amount is not user-adjustable and the automatic ISO rule below
//! is the only source.

use crate::rawinfo::RawInfoBlock;

/// Vendor range for the banding amount, as Capture One reports it through
/// `ReadInfoBlock(infbRanges)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandingRange {
    pub start: f32,
    pub end: f32,
    pub default: f32,
}

impl BandingRange {
    /// C1's public API takes the amount as an `int`, so one tick is one unit.
    pub const STEP: f32 = 1.0;

    pub fn new(start: f32, end: f32, default: f32) -> Option<Self> {
        (start.is_finite() && end.is_finite() && default.is_finite() && start <= end)
            .then_some(Self { start, end, default })
    }

    /// `false` for the vendor's empty range (`start == end`); the slider must be
    /// disabled rather than widened to a guessed 0..100.
    pub fn is_adjustable(&self) -> bool {
        self.end > self.start
    }

    pub fn step(&self) -> f32 {
        Self::STEP
    }

    /// Reads `IB_F_BANDINGREDUCTIONAMOUNT_START/END/DEFAULT`. All three keys
    /// must be present; a missing field yields `None`, never a fabricated value.
    pub fn from_info(info: &RawInfoBlock) -> Option<Self> {
        let start = info.get("IB_F_BANDINGREDUCTIONAMOUNT_START")? as f32;
        let end = info.get("IB_F_BANDINGREDUCTIONAMOUNT_END")? as f32;
        let default = info.get("IB_F_BANDINGREDUCTIONAMOUNT_DEFAULT")? as f32;
        Self::new(start, end, default)
    }

}

/// The Z6III E recipe is the only body whose amplitude rule is proven: the
/// vendor config uses factory base 200 and the ISO ladder, and the archived
/// preview queue pins ISO 1000 to 201142/782/83 (`A = 314.2857`).
fn is_z6iii(model: &str) -> bool {
    let normalised: String = model
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .collect();
    matches!(normalised.as_str(), "NIKONZ63" | "NIKONZ6III" | "Z63" | "Z6III")
}

/// Range for one frame.
///
/// A RAW that carries the three info-block keys wins. Otherwise only bodies
/// whose vendor descriptor was measured are answered: the Z6III reports
/// `START=END=DEFAULT=0` (`ReadInfoBlock(infbRanges)` on `_DSC5058.NEF` and
/// `_DSC5673.NEF`); the NEF itself stores no `IB_` text, so there is nothing
/// to parse from the file. Unknown bodies return `None`.
pub fn range_for_frame(info: Option<&RawInfoBlock>, model: Option<&str>) -> Option<BandingRange> {
    if let Some(range) = info.and_then(BandingRange::from_info) {
        return Some(range);
    }
    model.filter(|model| is_z6iii(model)).map(|_| BandingRange { start: 0.0, end: 0.0, default: 0.0 })
}

/// ISO scale from the reference config (virtual `+0x668`): 0.5 up to ISO 100,
/// `S/200` to 200, `1+(S-200)/1400` to 1600, then 2.
fn iso_scale(iso: u32) -> f32 {
    match iso {
        0..=100 => 0.5,
        101..=200 => iso as f32 / 200.0,
        201..=1600 => 1.0 + (iso - 200) as f32 / 1400.0,
        _ => 2.0,
    }
}

/// Factory-base amplitude before the `+0x7c` override. Only the Z6III rule is
/// proven; other bodies return `None` instead of borrowing it.
pub fn default_for_iso(model: &str, iso: u32) -> Option<f32> {
    is_z6iii(model).then_some(200.0 * iso_scale(iso))
}

fn amount_for_iso(iso: u32) -> f32 {
    200.0 * iso_scale(iso)
}

/// Resolved kernel integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    pub gain: i32,
    pub limit: i32,
    pub strength: i32,
}

impl Params {
    pub const OFF: Self = Self { gain: 0, limit: 0, strength: 0 };
}

/// Converts the resolved operation amplitude to the three kernel integers with
/// single-precision operations in the reference order.
pub fn params_for_amount(amount: f32) -> Params {
    if amount <= 0.0 {
        return Params::OFF;
    }
    let gain = (amount * (1.0_f32 / 128.0) * 1.25 * 65_536.0) as i32;
    let limit = (gain * 255) >> 16;
    let strength = if limit > 0 { (65_536.0_f32 / limit as f32) as i32 } else { 0 };
    Params { gain, limit, strength }
}

/// Resolves the amplitude from ISO through the verified factory base (200) and
/// the camera/ISO rules used by the Z6III E recipe.
pub fn params_for_iso(iso: u32) -> Params {
    params_for_amount(amount_for_iso(iso))
}

/// The three immutable vendor banding buffers, read from the installed
/// `ImgCoreDll.dll` at the offsets the native lane validated.
///
/// `None` when the DLL is absent or the bytes do not match the validated
/// fingerprints, so a different install can never silently change the pattern.
pub fn vendor_buffers() -> Option<&'static ([u8; 256], [u8; 256], [u8; 1024])> {
    use std::io::{Read, Seek, SeekFrom};

    fn fingerprint(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf29ce484222325_u64, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
    }

    static BUFFERS: std::sync::OnceLock<Option<([u8; 256], [u8; 256], [u8; 1024])>> =
        std::sync::OnceLock::new();
    BUFFERS
        .get_or_init(|| {
            let root = std::env::var_os("AURORA_Z6III_C1_DIR")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| std::path::PathBuf::from(r"C:\Program Files\Capture One\Capture One"));
            let mut core = std::fs::File::open(root.join("ImgCoreDll.dll")).ok()?;
            let mut left = [0_u8; 256];
            let mut right = [0_u8; 256];
            let mut packed = [0_u8; 1024];
            for (offset, table) in [
                (40_613_632, &mut left[..]),
                (40_614_912, &mut right[..]),
                (40_613_888, &mut packed[..]),
            ] {
                core.seek(SeekFrom::Start(offset)).ok()?;
                core.read_exact(table).ok()?;
            }
            ([fingerprint(&left), fingerprint(&right), fingerprint(&packed)]
                == [0x0528b3d3556eb0f5, 0x24060a098de1ec13, 0x065f8d932bb7fdef3])
                .then_some((left, right, packed))
        })
        .as_ref()
}

/// Applies the kernel to packed RGB16.
///
/// `origin` is the kernel's int2 phase argument. It is added to the pixel
/// coordinate before the 256x256 lookup, so any geometry or crop offset must be
/// carried here. Identity geometry is `(0, 0)`.
pub fn dispatch_rgb16(
    rgb: &mut [u16],
    width: usize,
    origin: (i32, i32),
    params: Params,
    tables: [&[u8]; 3],
) -> Result<(), &'static str> {
    dispatch(rgb, 3, width, origin, params, tables)
}

/// Applies the kernel to packed RGBA16; alpha is left untouched.
pub fn dispatch_rgba16(
    rgba: &mut [u16],
    width: usize,
    origin: (i32, i32),
    params: Params,
    tables: [&[u8]; 3],
) -> Result<(), &'static str> {
    dispatch(rgba, 4, width, origin, params, tables)
}

/// Pre-resolved banding tables: the two 256-byte index buffers and the 256
/// packed 32-bit words they index. Preparing them once lets a renderer that
/// evaluates pixels out of raster order (the interactive source sampler) apply
/// the same kernel per pixel without rebuilding the word table.
#[derive(Clone, Debug)]
pub struct Tables {
    left: [u8; 256],
    right: [u8; 256],
    words: [u32; 256],
}

impl Tables {
    pub fn new([left, right, packed]: [&[u8]; 3]) -> Option<Self> {
        if left.len() != 256 || right.len() != 256 || packed.len() != 1024 {
            return None;
        }
        let mut words = [0_u32; 256];
        for (slot, word) in words.iter_mut().zip(packed.chunks_exact(4)) {
            *slot = u32::from_le_bytes(word.try_into().ok()?);
        }
        Some(Self {
            left: left.try_into().ok()?,
            right: right.try_into().ok()?,
            words,
        })
    }

    /// Perturbs one packed RGB16 pixel at device coordinate `(x, y)`.
    /// `Params::OFF` (or a zero strength) leaves the pixel untouched.
    pub fn apply_pixel(&self, pixel: &mut [u16; 3], x: i32, y: i32, params: Params) {
        let Params { gain, limit, strength } = params;
        if limit <= 0 || strength <= 0 {
            return;
        }
        // The reference clears the column's low bit before taking the high
        // nibble, so columns 14 and 15 share a cell.
        let cell = ((((x & !1) & 0x0f) << 4) | (y & 0x0f)) as usize;
        let block = (((x & 0xf0) >> 4) | (y & 0xf0)) as usize;
        let pair = self.words[(block + usize::from(self.left[cell])) & 255]
            .wrapping_sub(self.words[(block + usize::from(self.right[cell])) & 255]);
        let half = if x & 1 != 0 { pair >> 16 } else { pair & 0xffff };
        let high = (((half >> 8) as i32) * gain) >> 16;
        let low = (((half & 255) as i32) * gain) >> 16;
        for code in pixel {
            let value = i32::from(*code);
            let headroom = limit.min(65_535 - value);
            *code = (value + (((high - low) * headroom * strength) >> 16)).clamp(0, 65_535) as u16;
        }
    }
}

fn dispatch(
    pixels: &mut [u16],
    channels: usize,
    width: usize,
    origin: (i32, i32),
    params: Params,
    tables: [&[u8]; 3],
) -> Result<(), &'static str> {
    if width == 0 || channels < 3 || pixels.len() % (channels * width) != 0 {
        return Err("invalid banding frame or tables");
    }
    let tables = Tables::new(tables).ok_or("invalid banding frame or tables")?;
    if params.limit <= 0 || params.strength <= 0 {
        return Ok(());
    }
    for (row, line) in pixels.chunks_exact_mut(channels * width).enumerate() {
        let y = row as i32 + origin.1;
        for (column, pixel) in line.chunks_exact_mut(channels).enumerate() {
            let x = column as i32 + origin.0;
            let pixel: &mut [u16; 3] = (&mut pixel[..3]).try_into().unwrap();
            tables.apply_pixel(pixel, x, y, params);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZERO: [u8; 256] = [0; 256];

    /// 6x4 RGB16 probe frame shared with the native golden in `out/golden.json`.
    const INPUT: [u16; 72] = [
        0, 0, 65535, 1000, 500, 64535, 20000, 10000, 45535, 40000, 20000, 25535, 65535, 32767, 0, 32768, 16384, 32767,
        5000, 2500, 60535, 15000, 7500, 50535, 25000, 12500, 40535, 35000, 17500, 30535, 45000, 22500, 20535, 55000, 27500, 10535,
        65535, 32767, 0, 0, 0, 65535, 12345, 6172, 53190, 54321, 27160, 11214, 22222, 11111, 43313, 44444, 22222, 21091,
        30000, 15000, 35535, 31000, 15500, 34535, 32000, 16000, 33535, 33000, 16500, 32535, 34000, 17000, 31535, 35000, 17500, 30535,
    ];

    /// Native kernel, amount 100, phase (0,0): 64000/249/263.
    const GOLDEN_100_PHASE0: [u16; 72] = [
        224, 224, 65535, 1225, 725, 64760, 19849, 9849, 45384, 39850, 19850, 25385, 65535, 32742, 0, 32744, 16360, 32743,
        5212, 2712, 60747, 15213, 7713, 50748, 24837, 12337, 40372, 34838, 17338, 30373, 44963, 22463, 20498, 54964, 27464, 10499,
        65535, 32967, 200, 201, 201, 65535, 12172, 5999, 53017, 54149, 26988, 11042, 22173, 11062, 43264, 44396, 22174, 21043,
        30188, 15188, 35723, 31189, 15689, 34724, 31814, 15814, 33349, 32814, 16314, 32349, 33939, 16939, 31474, 34940, 17440, 30475,
    ];

    /// Native kernel, amount 100, phase (7,5).
    const GOLDEN_100_PHASE75: [u16; 72] = [
        41, 41, 65535, 1166, 666, 64701, 20166, 10166, 45701, 39790, 19790, 25325, 65535, 32557, 0, 32684, 16300, 32683,
        5029, 2529, 60564, 15154, 7654, 50689, 25155, 12655, 40690, 34778, 17278, 30313, 44778, 22278, 20313, 54904, 27404, 10439,
        65535, 32784, 17, 142, 142, 65535, 12488, 6315, 53333, 54087, 26926, 10980, 21988, 10877, 43079, 44336, 22114, 20983,
        30005, 15005, 35540, 31130, 15630, 34665, 32131, 16131, 33666, 32754, 16254, 32289, 33754, 16754, 31289, 34880, 17380, 30415,
    ];

    /// Native kernel, amount 314.285736, phase (7,5): the Z6III ISO 1000 default.
    const GOLDEN_314_PHASE75: [u16; 72] = [
        130, 130, 65535, 1516, 1016, 65051, 20519, 10519, 46054, 39346, 19346, 24881, 65535, 32113, 0, 32503, 16119, 32502,
        5094, 2594, 60629, 15480, 7980, 51015, 25483, 12983, 41018, 34309, 16809, 29844, 44309, 21809, 19844, 54698, 27198, 10233,
        65535, 32824, 57, 443, 443, 65535, 12791, 6618, 53636, 53594, 26433, 10487, 21495, 10384, 42586, 44106, 21884, 20753,
        30021, 15021, 35556, 31407, 15907, 34942, 32410, 16410, 33945, 32237, 15737, 31772, 33237, 16237, 30772, 34625, 17125, 30160,
    ];

    fn synthetic() -> ([u8; 256], [u8; 256], [u8; 1024]) {
        let mut left = [0_u8; 256];
        let mut right = [0_u8; 256];
        let mut packed = [0_u8; 1024];
        for i in 0..256_u32 {
            left[i as usize] = i as u8;
            right[i as usize] = ((i * 7 + 13) & 0xFF) as u8;
            let half = (i << 8) | (255 - i);
            let value = half | (half << 16);
            packed[i as usize * 4..i as usize * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        (left, right, packed)
    }

    fn rgb_pixels() -> Vec<u16> {
        INPUT.to_vec()
    }

    fn rgba_pixels() -> Vec<u16> {
        INPUT.chunks_exact(3)
            .flat_map(|px| [px[0], px[1], px[2], 65_535])
            .collect()
    }

    #[test]
    fn matches_native_kernel_at_identity_and_offset_phase() {
        let (left, right, packed) = synthetic();
        let tables = [&left[..], &right[..], &packed[..]];
        let params = params_for_amount(100.0);
        assert_eq!(params, Params { gain: 64_000, limit: 249, strength: 263 });

        let mut rgb = rgb_pixels();
        dispatch_rgb16(&mut rgb, 6, (0, 0), params, tables).unwrap();
        assert_eq!(rgb, GOLDEN_100_PHASE0);

        let mut rgb = rgb_pixels();
        dispatch_rgb16(&mut rgb, 6, (7, 5), params, tables).unwrap();
        assert_eq!(rgb, GOLDEN_100_PHASE75);

        let mut rgba = rgba_pixels();
        dispatch_rgba16(&mut rgba, 6, (7, 5), params, tables).unwrap();
        let rgb_view: Vec<u16> = rgba.chunks_exact(4).flat_map(|px| [px[0], px[1], px[2]]).collect();
        assert_eq!(rgb_view, GOLDEN_100_PHASE75);
        assert!(rgba.chunks_exact(4).all(|px| px[3] == 65_535));

        let mut rgba = rgba_pixels();
        dispatch_rgba16(&mut rgba, 6, (7, 5), params_for_amount(314.285_74), tables).unwrap();
        let rgb_view: Vec<u16> = rgba.chunks_exact(4).flat_map(|px| [px[0], px[1], px[2]]).collect();
        assert_eq!(rgb_view, GOLDEN_314_PHASE75);
    }

    #[test]
    fn zero_amount_is_identity_and_phase_wraps() {
        let (left, right, packed) = synthetic();
        let tables = [&left[..], &right[..], &packed[..]];
        let mut off = rgb_pixels();
        dispatch_rgb16(&mut off, 6, (7, 5), Params::OFF, tables).unwrap();
        assert_eq!(off, INPUT);

        let mut a = rgb_pixels();
        dispatch_rgb16(&mut a, 6, (7, 5), params_for_amount(100.0), tables).unwrap();
        let mut b = rgb_pixels();
        dispatch_rgb16(&mut b, 6, (7 + 256, 5 + 256), params_for_amount(100.0), tables).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn per_pixel_entry_matches_the_raster_kernel() {
        let (left, right, packed) = synthetic();
        let tables = Tables::new([&left[..], &right[..], &packed[..]]).unwrap();
        let params = params_for_amount(100.0);
        let mut raster = rgb_pixels();
        dispatch_rgb16(&mut raster, 6, (0, 0), params, [&left[..], &right[..], &packed[..]]).unwrap();
        let mut input = INPUT;
        for (index, pixel) in raster.chunks_exact(3).enumerate() {
            let (x, y) = ((index % 6) as i32, (index / 6) as i32);
            let from_input: &mut [u16; 3] = (&mut input[index * 3..index * 3 + 3]).try_into().unwrap();
            tables.apply_pixel(from_input, x, y, params);
            assert_eq!(&pixel[..], &from_input[..], "pixel {index}");
        }
    }

    #[test]
    fn zero_amount_leaves_pixels_untouched() {
        let (left, right, packed) = synthetic();
        let tables = Tables::new([&left[..], &right[..], &packed[..]]).unwrap();
        assert_eq!(params_for_amount(0.0), Params::OFF);
        let mut pixel = [1234, 5678, 9000];
        tables.apply_pixel(&mut pixel, 3, 4, params_for_amount(0.0));
        assert_eq!(pixel, [1234, 5678, 9000]);
    }

    #[test]
    fn suppression_reduces_a_phase_locked_pattern() {
        let (left, right, packed) = synthetic();
        let tables = [&left[..], &right[..], &packed[..]];
        let params = params_for_amount(100.0);
        let (w, h) = (32, 32);
        let base = vec![30_000_u16; w * h * 3];
        let mut moved = base.clone();
        dispatch_rgb16(&mut moved, w, (0, 0), params, tables).unwrap();
        // The kernel adds a fixed pattern; content carrying its inverse must be
        // pulled back toward the base level.
        let content: Vec<u16> = base
            .iter()
            .zip(&moved)
            .map(|(b, m)| (i32::from(*b) - (i32::from(*m) - i32::from(*b))).clamp(0, 65_535) as u16)
            .collect();
        let mut corrected = content.clone();
        dispatch_rgb16(&mut corrected, w, (0, 0), params, tables).unwrap();
        let residual = |frame: &[u16]| -> i64 {
            frame.iter().zip(&base).map(|(a, b)| (i64::from(*a) - i64::from(*b)).abs()).sum()
        };
        assert!(residual(&content) > 0, "the injected pattern must be visible");
        assert!(
            residual(&corrected) < residual(&content),
            "suppression must reduce the phase-locked pattern: {} -> {}",
            residual(&content),
            residual(&corrected)
        );
    }

    #[test]
    fn odd_column_shares_even_table_cell() {
        let mut left = [0_u8; 256];
        left[224] = 7;
        left[240] = 9;
        let (_, _, packed) = synthetic();
        let tables = [&left[..], &ZERO[..], &packed[..]];
        let params = params_for_iso(200);
        let frame = [20_000_u16; 3];
        let mut odd = frame;
        dispatch_rgb16(&mut odd, 1, (15, 0), params, tables).unwrap();
        let mut even = frame;
        dispatch_rgb16(&mut even, 1, (14, 0), params, tables).unwrap();
        assert_eq!(odd, even);
    }

    #[test]
    fn amount_matches_reference_integers() {
        assert_eq!(params_for_amount(0.0), Params::OFF);
        assert_eq!(params_for_amount(50.0), Params { gain: 32_000, limit: 124, strength: 528 });
        assert_eq!(params_for_amount(100.0), Params { gain: 64_000, limit: 249, strength: 263 });
        assert_eq!(params_for_amount(125.0), Params { gain: 80_000, limit: 311, strength: 210 });
        assert_eq!(params_for_amount(200.0), Params { gain: 128_000, limit: 498, strength: 131 });
        assert_eq!(params_for_iso(1000), Params { gain: 201_142, limit: 782, strength: 83 });
    }

    #[test]
    fn info_range_requires_all_three_keys() {
        let mut info = RawInfoBlock::default();
        info.values = vec![
            ("IB_F_BANDINGREDUCTIONAMOUNT_START".into(), 0.0),
            ("IB_F_BANDINGREDUCTIONAMOUNT_END".into(), 60.0),
            ("IB_F_BANDINGREDUCTIONAMOUNT_DEFAULT".into(), 20.0),
        ];
        assert_eq!(
            BandingRange::from_info(&info),
            Some(BandingRange { start: 0.0, end: 60.0, default: 20.0 })
        );
        info.values.pop();
        assert_eq!(BandingRange::from_info(&info), None, "missing DEFAULT is not invented");
        assert_eq!(
            range_for_frame(Some(&info), Some("NIKON Z 6III")),
            Some(BandingRange { start: 0.0, end: 0.0, default: 0.0 }),
            "the measured body descriptor still answers"
        );
    }

    #[test]
    fn unknown_body_without_keys_has_no_range() {
        assert_eq!(range_for_frame(None, Some("SONY ILCE-7M4")), None);
        assert_eq!(range_for_frame(None, Some("Canon EOS R5")), None);
        assert_eq!(range_for_frame(None, None), None);
    }

    #[test]
    fn z6iii_reports_an_empty_range() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/nef-exif-head.bin");
        let info = crate::read_nikon_info_block(&path).expect("fixture header");
        assert_eq!(info.model.as_deref(), Some("NIKON Z6_3"));
        let range = range_for_frame(Some(&info), info.model.as_deref()).expect("Z6III range");
        assert_eq!(range, BandingRange { start: 0.0, end: 0.0, default: 0.0 });
        assert!(!range.is_adjustable(), "0..0 must disable the slider");
        assert_eq!(range.step(), 1.0);

        // The two real frames used for the vendor probe, when present.
        let dir = std::env::var_os("AURORA_BANDING_SAMPLE_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(r"C:\Users\Jxcz2\Desktop"));
        for name in ["_DSC5058.NEF", "_DSC5673.NEF"] {
            let Ok(info) = crate::read_nikon_info_block(&dir.join(name)) else { continue };
            let range = range_for_frame(Some(&info), info.model.as_deref()).expect("Z6III range");
            assert_eq!(range, BandingRange { start: 0.0, end: 0.0, default: 0.0 }, "{name}");
        }
    }

    #[test]
    fn iso_rule_is_the_verified_z6iii_table() {
        let amount = |iso| default_for_iso("NIKON Z 6III", iso).expect("Z6III rule");
        assert_eq!(amount(100), 100.0);
        assert_eq!(amount(200), 200.0);
        assert!((amount(1000) - 314.285_74).abs() < 1e-3);
        assert_eq!(amount(1600), 400.0);
        assert_eq!(amount(3200), 400.0, "above 1600 the rule caps at 2x the base");
        assert_eq!(default_for_iso("SONY ILCE-7M4", 1000), None);
        assert_eq!(default_for_iso("Canon EOS R5", 1600), None);
    }
}
