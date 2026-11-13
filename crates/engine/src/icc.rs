//! ICC v2/v4 profile parsing and sRGB<->profile conversion.
//! Support level: matrix/TRC profiles (PCS XYZ D50) and `mft2` A2B0 3D-LUT profiles (PCS Lab).
//! Unsupported colour spaces: CMYK and any device space wider than three channels.
//! Unsupported tag paths: `mAB`/`mBA`, multi-CLUT `A2B1`/`B2A1`, `gamt`/`bkpt`, `ncl`.
//! Precision: 16-bit CLUT on the Lab PCS path, single-precision `f32` on the matrix/TRC path.

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// ICC colour space of the profile data side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pcs {
    Xyz,
    Lab,
}

/// One tone curve. `Table` holds normalised 0..1 values.
#[derive(Debug, Clone, PartialEq)]
pub enum Curve {
    Identity,
    Gamma(f32),
    Table(Vec<f32>),
}

impl Curve {
    pub fn eval(&self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        match self {
            Curve::Identity => x,
            Curve::Gamma(g) => x.powf(*g),
            Curve::Table(t) => {
                if t.len() < 2 {
                    return x;
                }
                let f = x * (t.len() - 1) as f32;
                let i = (f as usize).min(t.len() - 2);
                let frac = f - i as f32;
                t[i] + (t[i + 1] - t[i]) * frac
            }
        }
    }

    /// Invert a monotone curve.
    pub fn invert(&self, y: f32) -> f32 {
        invert_curve(self, y)
    }
}

/// 3x3 matrix plus per-channel curves, D50 PCS.
#[derive(Debug, Clone)]
pub struct MatrixTrc {
    /// Columns are the r/g/b XYZ tags: `m[ch][0..3]` is one colourant.
    pub matrix: [[f32; 3]; 3],
    pub trc: [Curve; 3],
}

/// An `mft2` look-up: shaper curves -> 3D grid -> output curves.
#[derive(Debug, Clone)]
pub struct Lut2 {
    pub input: [Curve; 3],
    pub grid: usize,
    /// Interleaved RGB triples in 0..1, `grid^3` of them.
    pub table: Vec<[f32; 3]>,
    pub output: [Curve; 3],
}

impl Lut2 {
    /// Device RGB -> PCS (in the PCS' own 16-bit encoding).
    pub fn eval(&self, rgb: [f32; 3]) -> [f32; 3] {
        let v = [
            self.input[0].eval(rgb[0]),
            self.input[1].eval(rgb[1]),
            self.input[2].eval(rgb[2]),
        ];
        let g = self.grid;
        if g < 2 || self.table.len() != g * g * g {
            return v;
        }
        let pos = [
            v[0] * (g - 1) as f32,
            v[1] * (g - 1) as f32,
            v[2] * (g - 1) as f32,
        ];
        let idx = [
            (pos[0] as usize).min(g - 2),
            (pos[1] as usize).min(g - 2),
            (pos[2] as usize).min(g - 2),
        ];
        let frac = [
            pos[0] - idx[0] as f32,
            pos[1] - idx[1] as f32,
            pos[2] - idx[2] as f32,
        ];
        let at = |i: usize, j: usize, k: usize| self.table[(i * g + j) * g + k];
        let (i0, j0, k0) = (idx[0], idx[1], idx[2]);
        // Tetrahedral interpolation, as lcms does: trilinear drifts up to
        // 12/255 on dark saturated colours.
        let p = [
            at(i0, j0, k0),
            at(i0 + 1, j0, k0),
            at(i0, j0 + 1, k0),
            at(i0, j0, k0 + 1),
            at(i0 + 1, j0 + 1, k0),
            at(i0 + 1, j0, k0 + 1),
            at(i0, j0 + 1, k0 + 1),
            at(i0 + 1, j0 + 1, k0 + 1),
        ];
        let (rx, ry, rz) = (frac[0], frac[1], frac[2]);
        let mut pcs = [0.0f32; 3];
        for c in 0..3 {
            pcs[c] = if rx > ry {
                if ry > rz {
                    p[0][c] + (p[1][c] - p[0][c]) * rx + (p[4][c] - p[1][c]) * ry + (p[7][c] - p[4][c]) * rz
                } else if rx > rz {
                    p[0][c] + (p[1][c] - p[0][c]) * rx + (p[5][c] - p[1][c]) * rz + (p[7][c] - p[5][c]) * ry
                } else {
                    p[0][c] + (p[3][c] - p[0][c]) * rz + (p[5][c] - p[3][c]) * rx + (p[7][c] - p[5][c]) * ry
                }
            } else if rz > ry {
                p[0][c] + (p[3][c] - p[0][c]) * rz + (p[6][c] - p[3][c]) * ry + (p[7][c] - p[6][c]) * rx
            } else if rz > rx {
                p[0][c] + (p[2][c] - p[0][c]) * ry + (p[6][c] - p[2][c]) * rz + (p[7][c] - p[6][c]) * rx
            } else {
                p[0][c] + (p[2][c] - p[0][c]) * ry + (p[4][c] - p[2][c]) * rx + (p[7][c] - p[4][c]) * rz
            };
        }
        [
            self.output[0].eval(pcs[0]),
            self.output[1].eval(pcs[1]),
            self.output[2].eval(pcs[2]),
        ]
    }
}

/// A parsed ICC profile.
#[derive(Debug, Clone)]
pub struct Profile {
    pub description: String,
    pub data_space: [u8; 4],
    pub pcs: Pcs,
    pub matrix: Option<MatrixTrc>,
    pub a2b: Option<Lut2>,
    pub b2a: Option<Lut2>,
    /// Tone curve of a greyscale profile (`GRAY` data space).
    pub gray: Option<Curve>,
    pub path: Option<PathBuf>,
    inverse: OnceLock<Option<InverseLut>>,
}

impl Profile {
    /// Device RGB (0..1, as stored in the file) to linear sRGB.
    pub fn to_linear_srgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        xyz_d50_to_linear_srgb(self.decode(rgb))
    }

    /// Linear sRGB to device RGB. LUT profiles invert by nearest node.
    pub fn from_linear_srgb(&self, rgb: [f32; 3]) -> Result<[f32; 3]> {
        Ok(self.encode(linear_srgb_to_xyz_d50(rgb)))
    }
}

/// Invert a monotone curve: find the x with `curve(x) == y`.
fn invert_curve(curve: &Curve, y: f32) -> f32 {
    let y = y.clamp(0.0, 1.0);
    match curve {
        Curve::Identity => y,
        Curve::Gamma(g) => y.powf(1.0 / g.max(1e-6)),
        Curve::Table(t) => {
            if t.len() < 2 {
                return y;
            }
            let mut lo = 0usize;
            let mut hi = t.len() - 1;
            if y <= t[lo] {
                return 0.0;
            }
            if y >= t[hi] {
                return 1.0;
            }
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if t[mid] < y {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let span = t[hi] - t[lo];
            let f = if span > 1e-9 { (y - t[lo]) / span } else { 0.0 };
            (lo as f32 + f) / (t.len() - 1) as f32
        }
    }
}

// ---------------------------- PCS conversions ----------------------------

/// The D50 reference white the ICC PCS is defined against.
const D50: [f32; 3] = [0.9642, 1.0, 0.8249];

/// ICC 16-bit Lab decoding, from the normalised (0..1) `u16` encoding.
///
/// `L = v * 100 / 65280`, `a = b = (v - 32768) / 256`, which is what the
/// camera profiles in Capture One use.
fn lab_to_xyz_d50(raw: [f32; 3]) -> [f32; 3] {
    // L* is encoded over 0..65280 (0xFF00) in ICC v2 Lab, i.e. 0..100.
    let l = (raw[0] * 65535.0 * 100.0 / 65280.0).min(100.0);
    let a = (raw[1] * 65535.0 - 32768.0) / 256.0;
    let b = (raw[2] * 65535.0 - 32768.0) / 256.0;
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let finv = |t: f32| {
        let t3 = t * t * t;
        if t3 > 0.008856 {
            t3
        } else {
            (t - 16.0 / 116.0) / 7.787
        }
    };
    [D50[0] * finv(fx), D50[1] * finv(fy), D50[2] * finv(fz)]
}

/// XYZ D50 -> Lab in the usual L*/a*/b* units, for distance comparisons.
fn xyz_d50_to_lab_star(xyz: [f32; 3]) -> [f32; 3] {
    let f = |t: f32| {
        if t > 0.008856 {
            t.cbrt()
        } else {
            (903.3 * t + 16.0) / 116.0
        }
    };
    let fx = f(xyz[0] / D50[0]);
    let fy = f(xyz[1] / D50[1]);
    let fz = f(xyz[2] / D50[2]);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

fn xyz_d50_to_linear_srgb(xyz: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = bradford_d50_to_d65(xyz);
    [
        3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z,
        -0.969_266_0 * x + 1.876_010_8 * y + 0.041_556_0 * z,
        0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z,
    ]
}

fn linear_srgb_to_xyz_d50(rgb: [f32; 3]) -> [f32; 3] {
    let x = 0.412_456_4 * rgb[0] + 0.357_576_1 * rgb[1] + 0.180_437_5 * rgb[2];
    let y = 0.212_672_9 * rgb[0] + 0.715_152_2 * rgb[1] + 0.072_175_0 * rgb[2];
    let z = 0.019_333_9 * rgb[0] + 0.119_192_0 * rgb[1] + 0.950_304_1 * rgb[2];
    bradford_d65_to_d50([x, y, z])
}

fn bradford_d50_to_d65(v: [f32; 3]) -> [f32; 3] {
    [
        0.955_576_6 * v[0] - 0.023_039_3 * v[1] + 0.063_163_6 * v[2],
        -0.028_289_5 * v[0] + 1.009_941_6 * v[1] + 0.021_007_7 * v[2],
        0.012_298_2 * v[0] - 0.020_483_0 * v[1] + 1.329_909_8 * v[2],
    ]
}

fn bradford_d65_to_d50(v: [f32; 3]) -> [f32; 3] {
    [
        1.047_811_2 * v[0] + 0.022_886_6 * v[1] - 0.050_127_0 * v[2],
        0.029_542_4 * v[0] + 0.990_484_4 * v[1] - 0.017_049_1 * v[2],
        -0.009_234_5 * v[0] + 0.015_043_6 * v[1] + 0.752_131_6 * v[2],
    ]
}

/// sRGB (0..1, gamma encoded) -> XYZ D50.
fn srgb_to_xyz_d50(srgb: [f32; 3]) -> [f32; 3] {
    linear_srgb_to_xyz_d50([
        linearize_srgb(srgb[0]),
        linearize_srgb(srgb[1]),
        linearize_srgb(srgb[2]),
    ])
}

/// XYZ D50 -> sRGB (0..1, gamma encoded).
fn xyz_d50_to_srgb(xyz: [f32; 3]) -> [f32; 3] {
    let linear = xyz_d50_to_linear_srgb(xyz);
    [
        delinearize_srgb(linear[0]),
        delinearize_srgb(linear[1]),
        delinearize_srgb(linear[2]),
    ]
}

fn linearize_srgb(c: f32) -> f32 {
    let v = c.clamp(0.0, 1.0);
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn delinearize_srgb(c: f32) -> f32 {
    let v = c.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

// ------------------------------- parsing --------------------------------

struct Tag {
    sig: [u8; 4],
    offset: usize,
    size: usize,
}

fn u16b(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*b.get(at)?, *b.get(at + 1)?]))
}

fn u32b(b: &[u8], at: usize) -> Option<u32> {
    let s = b.get(at..at + 4)?;
    Some(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}

/// Parse an ICC profile.
pub fn parse_icc(bytes: &[u8]) -> Result<Profile> {
    if bytes.len() < 132 {
        bail!("ICC too short ({} bytes)", bytes.len());
    }
    if &bytes[36..40] != b"acsp" {
        bail!("not an ICC profile: 'acsp' signature missing");
    }
    let data_space = [bytes[16], bytes[17], bytes[18], bytes[19]];
    let pcs = match &bytes[20..24] {
        b"XYZ " => Pcs::Xyz,
        b"Lab " => Pcs::Lab,
        other => bail!("unsupported PCS {:?}", String::from_utf8_lossy(other)),
    };
    let count = u32b(bytes, 128).context("tag count")? as usize;
    if 132 + count * 12 > bytes.len() {
        bail!("tag table overruns the file");
    }
    let mut tags = Vec::with_capacity(count);
    for i in 0..count {
        let at = 132 + i * 12;
        let mut sig = [0u8; 4];
        sig.copy_from_slice(&bytes[at..at + 4]);
        let offset = u32b(bytes, at + 4).context("tag offset")? as usize;
        let size = u32b(bytes, at + 8).context("tag size")? as usize;
        if offset.checked_add(size).map_or(true, |end| end > bytes.len()) {
            continue;
        }
        tags.push(Tag { sig, offset, size });
    }
    let find = |name: &[u8; 4]| tags.iter().find(|t| &t.sig == name);

    let description = find(b"desc")
        .map(|t| desc_string(&bytes[t.offset..t.offset + t.size]))
        .or_else(|| find(b"mluc").map(|t| mluc_string(&bytes[t.offset..t.offset + t.size])))
        .unwrap_or_default();

    let matrix = match (find(b"rXYZ"), find(b"gXYZ"), find(b"bXYZ"), find(b"rTRC")) {
        (Some(r), Some(g), Some(b), Some(trc)) => {
            let col = |t: &Tag| {
                let d = &bytes[t.offset..t.offset + t.size];
                [
                    s15f16(d, 8).unwrap_or(1.0),
                    s15f16(d, 12).unwrap_or(0.0),
                    s15f16(d, 16).unwrap_or(0.0),
                ]
            };
            Some(MatrixTrc {
                matrix: [col(r), col(g), col(b)],
                trc: [
                    parse_curve(&bytes[trc.offset..trc.offset + trc.size])?,
                    find(b"gTRC")
                        .map(|t| parse_curve(&bytes[t.offset..t.offset + t.size]))
                        .unwrap_or_else(|| Ok(Curve::Identity))?,
                    find(b"bTRC")
                        .map(|t| parse_curve(&bytes[t.offset..t.offset + t.size]))
                        .unwrap_or_else(|| Ok(Curve::Identity))?,
                ],
            })
        }
        _ => None,
    };

    let gray = if &data_space == b"GRAY" {
        find(b"kTRC")
            .map(|t| parse_curve(&bytes[t.offset..t.offset + t.size]))
            .transpose()?
    } else {
        None
    };

    let a2b = find(b"A2B0")
        .map(|t| parse_mft2(&bytes[t.offset..t.offset + t.size]))
        .transpose()?;
    let b2a = find(b"B2A0")
        .map(|t| parse_mft2(&bytes[t.offset..t.offset + t.size]))
        .transpose()?;

    if matrix.is_none() && a2b.is_none() && gray.is_none() {
        bail!("profile has neither a matrix/TRC nor an A2B0 table");
    }
    Ok(Profile {
        description,
        data_space,
        pcs,
        matrix,
        a2b,
        b2a,
        gray,
        path: None,
        inverse: OnceLock::new(),
    })
}

fn s15f16(b: &[u8], at: usize) -> Option<f32> {
    Some(i32::from_be_bytes([*b.get(at)?, *b.get(at + 1)?, *b.get(at + 2)?, *b.get(at + 3)?]) as f32
        / 65536.0)
}

fn desc_string(d: &[u8]) -> String {
    let len = u32b(d, 8).unwrap_or(0) as usize;
    let raw = d.get(12..12 + len.min(d.len().saturating_sub(12))).unwrap_or(&[]);
    String::from_utf8_lossy(raw).trim_end_matches('\0').to_string()
}

/// `mluc` (multi-localised unicode) description.
fn mluc_string(d: &[u8]) -> String {
    let Some(count) = u32b(d, 8) else { return String::new() };
    let Some(len) = u32b(d, 20) else { return String::new() };
    let Some(offset) = u32b(d, 24) else { return String::new() };
    let start = offset as usize;
    let end = start.saturating_add(len as usize).min(d.len());
    let raw = d.get(start..end).unwrap_or(&[]);
    let text = String::from_utf16_lossy(
        &raw.chunks_exact(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
            .collect::<Vec<u16>>(),
    );
    let _ = count;
    text.trim_end_matches('\0').to_string()
}

fn parse_curve(d: &[u8]) -> Result<Curve> {
    if u32b(d, 0) != Some(u32::from_be_bytes(*b"curv")) {
        return Ok(Curve::Identity);
    }
    let n = u32b(d, 8).unwrap_or(0) as usize;
    if n == 0 {
        return Ok(Curve::Identity);
    }
    if n == 1 {
        let g = u16b(d, 12).unwrap_or(0x100) as f32 / 256.0;
        return Ok(Curve::Gamma(if g > 0.0 { g } else { 1.0 }));
    }
    let mut t = Vec::with_capacity(n);
    for i in 0..n {
        t.push(u16b(d, 12 + i * 2).ok_or_else(|| anyhow::anyhow!("curve truncated"))? as f32 / 65535.0);
    }
    Ok(Curve::Table(t))
}

/// Parse an `mft2` look-up table.
///
/// Layout (ICC.1 §6.5.5): signature, shape, a 3x3 matrix, then one `u16`
/// entry count for the input tables and one for the output tables, then the
/// input tables, the CLUT, and finally the output tables. This is the layout
/// every Capture One camera profile uses; the size check below proves it.
fn parse_mft2(d: &[u8]) -> Result<Lut2> {
    if u32b(d, 0) != Some(u32::from_be_bytes(*b"mft2")) {
        bail!("not an mft2 table");
    }
    let cin = *d.get(8).context("input channels")? as usize;
    let cout = *d.get(9).context("output channels")? as usize;
    let grid = *d.get(10).context("grid points")? as usize;
    if cin == 0 || cout < 3 || grid < 2 || cin > 4 {
        bail!("unsupported mft2 shape {cin}->{cout} grid {grid}");
    }
    let in_entries = u16b(d, 48).context("input table entries")? as usize;
    let out_entries = u16b(d, 50).context("output table entries")? as usize;
    let clut_entries = grid.pow(cin as u32) * cout;
    let need = 52 + in_entries * cin * 2 + clut_entries * 2 + out_entries * cout * 2;
    if need > d.len() {
        bail!("mft2 cube overruns the table (need {need}, have {})", d.len());
    }
    let mut pos = 52;
    let read_curve = |pos: &mut usize, entries: usize| -> Result<Curve> {
        if entries == 0 {
            return Ok(Curve::Identity);
        }
        if entries == 1 {
            *pos += 2;
            return Ok(Curve::Identity);
        }
        let mut t = Vec::with_capacity(entries);
        for i in 0..entries {
            t.push(
                u16b(d, *pos + i * 2)
                    .ok_or_else(|| anyhow::anyhow!("curve truncated"))? as f32
                    / 65535.0,
            );
        }
        *pos += entries * 2;
        Ok(Curve::Table(t))
    };
    let mut input = Vec::with_capacity(cin);
    for _ in 0..cin {
        input.push(read_curve(&mut pos, in_entries)?);
    }
    let mut table = Vec::with_capacity(grid.pow(3));
    for _ in 0..grid.pow(3) {
        let at = pos;
        let px = [
            u16b(d, at).unwrap_or(0) as f32 / 65535.0,
            u16b(d, at + 2).unwrap_or(0) as f32 / 65535.0,
            u16b(d, at + 4).unwrap_or(0) as f32 / 65535.0,
        ];
        table.push(px);
        pos += cout * 2;
    }
    let mut output = Vec::with_capacity(cout);
    for _ in 0..cout {
        output.push(read_curve(&mut pos, out_entries)?);
    }
    input.truncate(3);
    while input.len() < 3 {
        input.push(Curve::Identity);
    }
    output.truncate(3);
    while output.len() < 3 {
        output.push(Curve::Identity);
    }
    Ok(Lut2 {
        input: [input[0].clone(), input[1].clone(), input[2].clone()],
        grid,
        table,
        output: [output[0].clone(), output[1].clone(), output[2].clone()],
    })
}

/// Read an ICC file from disk.
pub fn load_profile(path: &Path) -> Result<Profile> {
    Profile::from_file(path)
}

/// Where Capture One keeps its profiles on this machine. Overridable for tests
/// and for installs that live elsewhere (`LUMEN_ICC_DIR`).
pub fn profile_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("LUMEN_ICC_DIR") {
        return PathBuf::from(dir);
    }
    let install = PathBuf::from(r"C:\Program Files\Capture One\Capture One\Color Profiles");
    if install.is_dir() {
        return install;
    }
    // `cargo test` runs with the crate as the working directory.
    PathBuf::from("testdata")
}

fn index() -> &'static Mutex<std::collections::HashMap<String, Option<Arc<Profile>>>> {
    static INDEX: OnceLock<Mutex<std::collections::HashMap<String, Option<Arc<Profile>>>>> =
        OnceLock::new();
    INDEX.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn profile_paths() -> &'static Vec<(String, PathBuf)> {
    static PATHS: OnceLock<Vec<(String, PathBuf)>> = OnceLock::new();
    PATHS.get_or_init(|| scan_profiles(&[profile_dir()]))
}

/// Profile names shipped with the install, sorted.
pub fn profile_names() -> Vec<String> {
    profile_paths().iter().map(|(name, _)| name.clone()).collect()
}

/// A profile by file stem, parsed once and shared.
pub fn profile(name: &str) -> Option<Arc<Profile>> {
    let mut index = index().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(hit) = index.get(name) {
        return hit.clone();
    }
    let loaded = profile_paths()
        .iter()
        .find(|(stem, _)| stem == name)
        .and_then(|(_, path)| Profile::from_file(path).ok())
        .map(Arc::new);
    index.insert(name.to_string(), loaded.clone());
    loaded
}

// --------------------------- high level profile ---------------------------

/// Number of bins per axis in the inverse-LUT index.
const INV_BINS: usize = 16;

/// Nearest-node map from the PCS back to device values.
///
/// The A2B0 table only goes device -> PCS, so the reverse direction is
/// approximated by finding the CLUT node whose PCS colour is closest to the
/// requested one.
#[derive(Debug, Clone)]
struct InverseLut {
    grid: usize,
    /// L*a*b* of every node.
    lab: Vec<[f32; 3]>,
    /// Device RGB that lands on every node.
    device: Vec<[f32; 3]>,
    /// `INV_BINS^3` buckets of node indices.
    bins: Vec<Vec<u32>>,
}

impl InverseLut {
    fn build(lut: &Lut2) -> Option<InverseLut> {
        let g = lut.grid;
        if g < 2 || lut.table.len() != g * g * g {
            return None;
        }
        let mut lab = Vec::with_capacity(g * g * g);
        let mut device = Vec::with_capacity(g * g * g);
        for i in 0..g {
            for j in 0..g {
                for k in 0..g {
                    let n = (i * g + j) * g + k;
                    let raw = lut.table[n];
                    let pcs = [
                        lut.output[0].eval(raw[0]),
                        lut.output[1].eval(raw[1]),
                        lut.output[2].eval(raw[2]),
                    ];
                    lab.push(xyz_d50_to_lab_star(lab_to_xyz_d50(pcs)));
                    let last = (g - 1) as f32;
                    device.push([
                        invert_curve(&lut.input[0], i as f32 / last),
                        invert_curve(&lut.input[1], j as f32 / last),
                        invert_curve(&lut.input[2], k as f32 / last),
                    ]);
                }
            }
        }
        let mut bins: Vec<Vec<u32>> = vec![Vec::new(); INV_BINS * INV_BINS * INV_BINS];
        for (n, l) in lab.iter().enumerate() {
            let b = bin_of(*l);
            bins[b].push(n as u32);
        }
        Some(InverseLut { grid: g, lab, device, bins })
    }

    /// The device colour whose PCS is closest to `pcs` (XYZ D50).
    fn nearest(&self, pcs: [f32; 3]) -> [f32; 3] {
        let target = xyz_d50_to_lab_star(pcs);
        let b = bin_of(target);
        let mut best: Option<u32> = None;
        let mut best_d = f32::MAX;
        let consider = |n: u32, best: &mut Option<u32>, best_d: &mut f32| {
            let d = lab_delta(self.lab[n as usize], target);
            if d < *best_d {
                *best_d = d;
                *best = Some(n);
            }
        };
        for dz in -1i32..=1 {
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let b = offset_bin(b, dx, dy, dz);
                    for &n in &self.bins[b] {
                        consider(n, &mut best, &mut best_d);
                    }
                }
            }
        }
        if best.is_none() {
            for n in 0..self.lab.len() as u32 {
                consider(n, &mut best, &mut best_d);
            }
        }
        let Some(mut n) = best else { return [0.0; 3] };
        // Refine inside the node's own 3x3x3 neighbourhood.
        let g = self.grid;
        let mut k = n as usize % g;
        let mut j = (n as usize / g) % g;
        let mut i = n as usize / (g * g);
        for step in 1..=1 {
            let (ci, cj, ck) = (i as i32, j as i32, k as i32);
            for di in -step..=step {
                for dj in -step..=step {
                    for dk in -step..=step {
                        let (ni, nj, nk) = (ci + di, cj + dj, ck + dk);
                        if ni < 0 || nj < 0 || nk < 0 {
                            continue;
                        }
                        let (ni, nj, nk) = (ni as usize, nj as usize, nk as usize);
                        if ni >= g || nj >= g || nk >= g {
                            continue;
                        }
                        let m = (ni * g + nj) * g + nk;
                        let d = lab_delta(self.lab[m], target);
                        if d < best_d {
                            best_d = d;
                            n = m as u32;
                            i = ni;
                            j = nj;
                            k = nk;
                        }
                    }
                }
            }
        }
        self.device[n as usize]
    }
}

fn lab_delta(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dl = a[0] - b[0];
    let da = a[1] - b[1];
    let db = a[2] - b[2];
    dl * dl + da * da + db * db
}

fn bin_of(lab: [f32; 3]) -> usize {
    let l = ((lab[0] / 100.0).clamp(0.0, 0.999) * INV_BINS as f32) as usize;
    let a = (((lab[1] + 128.0) / 255.0).clamp(0.0, 0.999) * INV_BINS as f32) as usize;
    let b = (((lab[2] + 128.0) / 255.0).clamp(0.0, 0.999) * INV_BINS as f32) as usize;
    (l * INV_BINS + a) * INV_BINS + b
}

fn offset_bin(base: usize, dx: i32, dy: i32, dz: i32) -> usize {
    let l = base / (INV_BINS * INV_BINS);
    let a = (base / INV_BINS) % INV_BINS;
    let b = base % INV_BINS;
    let l = (l as i32 + dx).clamp(0, INV_BINS as i32 - 1) as usize;
    let a = (a as i32 + dy).clamp(0, INV_BINS as i32 - 1) as usize;
    let b = (b as i32 + dz).clamp(0, INV_BINS as i32 - 1) as usize;
    (l * INV_BINS + a) * INV_BINS + b
}

impl Profile {
    /// Parse a profile from memory.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        Ok(parse_icc(data)?)
    }

    /// Parse a profile from disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let mut profile = parse_icc(&bytes).with_context(|| format!("parsing {}", path.display()))?;
        profile.path = Some(path.to_path_buf());
        Ok(profile)
    }

    /// The profile's description, or its file name.
    pub fn name(&self) -> &str {
        if !self.description.is_empty() {
            return &self.description;
        }
        self.path
            .as_ref()
            .and_then(|p| p.file_stem())
            .and_then(|s| s.to_str())
            .unwrap_or("")
    }

    /// True for matrix/TRC profiles, false for LUT profiles.
    pub fn is_matrix_shaper(&self) -> bool {
        self.matrix.is_some()
    }

    /// A PCS colour (XYZ, D50) -> device RGB (0..1).
    pub fn encode(&self, pcs: [f32; 3]) -> [f32; 3] {
        if let Some(curve) = &self.gray {
            let g = invert_curve(curve, pcs[1]);
            return [g, g, g];
        }
        match (&self.a2b, &self.matrix) {
            (Some(lut), _) => {
                let inverse = self.inverse.get_or_init(|| InverseLut::build(lut));
                match inverse {
                    Some(inv) => inv.nearest(pcs),
                    None => pcs,
                }
            }
            (None, Some(m)) => {
                let rows = [
                    [m.matrix[0][0], m.matrix[1][0], m.matrix[2][0]],
                    [m.matrix[0][1], m.matrix[1][1], m.matrix[2][1]],
                    [m.matrix[0][2], m.matrix[1][2], m.matrix[2][2]],
                ];
                match solve3(rows, pcs) {
                    Some(linear) => [
                        invert_curve(&m.trc[0], linear[0]),
                        invert_curve(&m.trc[1], linear[1]),
                        invert_curve(&m.trc[2], linear[2]),
                    ],
                    None => pcs,
                }
            }
            _ => pcs,
        }
    }

    /// Device RGB (0..1) -> PCS colour (XYZ, D50).
    pub fn decode(&self, rgb: [f32; 3]) -> [f32; 3] {
        if let Some(curve) = &self.gray {
            let y = curve.eval(rgb[0]);
            return [y * D50[0], y * D50[1], y * D50[2]];
        }
        match (&self.a2b, &self.matrix) {
            (Some(lut), _) => {
                let pcs = lut.eval(rgb);
                lab_to_xyz_d50(pcs)
            }
            (None, Some(m)) => {
                let linear = [
                    m.trc[0].eval(rgb[0]),
                    m.trc[1].eval(rgb[1]),
                    m.trc[2].eval(rgb[2]),
                ];
                [
                    m.matrix[0][0] * linear[0] + m.matrix[1][0] * linear[1] + m.matrix[2][0] * linear[2],
                    m.matrix[0][1] * linear[0] + m.matrix[1][1] * linear[1] + m.matrix[2][1] * linear[2],
                    m.matrix[0][2] * linear[0] + m.matrix[1][2] * linear[1] + m.matrix[2][2] * linear[2],
                ]
            }
            _ => rgb,
        }
    }
}

/// Solve `m * x = v` for a 3x3 system. Returns `None` when singular.
fn solve3(m: [[f32; 3]; 3], v: [f32; 3]) -> Option<[f32; 3]> {
    let mut a = m;
    let mut b = v;
    for col in 0..3 {
        let mut pivot = col;
        for row in (col + 1)..3 {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let d = a[col][col];
        for k in 0..3 {
            a[col][k] /= d;
        }
        b[col] /= d;
        for row in 0..3 {
            if row == col {
                continue;
            }
            let f = a[row][col];
            if f == 0.0 {
                continue;
            }
            for k in 0..3 {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    Some(b)
}

/// Convert an sRGB RGBA buffer in place into the profile's device space.
///
/// Uses rayon for the per-pixel work. A buffer whose length is not a multiple
/// of four is handled safely: the trailing bytes are left untouched.
pub fn convert_srgb_to(rgba: &mut [u8], profile: &Profile) {
    rgba.par_chunks_mut(4).for_each(|px| {
        if px.len() < 4 {
            return;
        }
        let srgb = [
            px[0] as f32 / 255.0,
            px[1] as f32 / 255.0,
            px[2] as f32 / 255.0,
        ];
        let out = profile.encode(srgb_to_xyz_d50(srgb));
        px[0] = to_u8(out[0]);
        px[1] = to_u8(out[1]);
        px[2] = to_u8(out[2]);
    });
}

/// Convert an RGBA buffer in the profile's device space into sRGB.
pub fn convert_to_srgb(rgba: &mut [u8], profile: &Profile) {
    rgba.par_chunks_mut(4).for_each(|px| {
        if px.len() < 4 {
            return;
        }
        let device = [
            px[0] as f32 / 255.0,
            px[1] as f32 / 255.0,
            px[2] as f32 / 255.0,
        ];
        let out = xyz_d50_to_srgb(profile.decode(device));
        px[0] = to_u8(out[0]);
        px[1] = to_u8(out[1]);
        px[2] = to_u8(out[2]);
    });
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Every ICC profile below the given directories, as (name, path).
pub fn scan_profiles(dirs: &[PathBuf]) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    let mut stack: Vec<PathBuf> = dirs.to_vec();
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let is_profile = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("icm") || e.eq_ignore_ascii_case("icc"))
                .unwrap_or(false);
            if is_profile {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    out.push((stem.to_string(), path));
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn srgb_profile_path() -> Option<PathBuf> {
        let direct = profile_dir().join("Common").join("sRGB Color Space Profile.icm");
        if direct.exists() {
            return Some(direct);
        }
        let sample = profile_dir().join("sRGB Color Space Profile.icm");
        if sample.exists() {
            return Some(sample);
        }
        let system = PathBuf::from(
            r"C:\Windows\System32\spool\drivers\color\sRGB Color Space Profile.icm",
        );
        if system.exists() {
            return Some(system);
        }
        None
    }

    fn srgb_profile() -> Option<Profile> {
        Profile::from_file(&srgb_profile_path()?).ok()
    }

    fn find_profile(name: &str) -> Option<PathBuf> {
        let mut stack = vec![profile_dir()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else { continue };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.file_stem().and_then(|s| s.to_str()) == Some(name) {
                    return Some(path);
                }
            }
        }
        None
    }

    #[test]
    fn gamma_and_table_curves_interpolate() {
        assert!((Curve::Gamma(2.0).eval(0.5) - 0.25).abs() < 1e-6);
        let t = Curve::Table(vec![0.0, 0.25, 1.0]);
        assert!((t.eval(0.5) - 0.25).abs() < 1e-6);
        assert_eq!(Curve::Identity.eval(1.4), 1.0);
    }

    #[test]
    fn table_curve_round_trips() {
        let t = Curve::Table(vec![0.0, 0.1, 0.5, 1.0]);
        for x in [0.0, 0.2, 0.5, 0.9, 1.0] {
            let back = invert_curve(&t, t.eval(x));
            assert!((back - x).abs() < 0.02, "{x} -> {back}");
        }
    }

    #[test]
    fn lab_black_and_white_are_the_pcs_neutral() {
        let black = lab_to_xyz_d50([0.0, 0.5, 0.5]);
        assert!(black.iter().all(|v| *v < 1e-4), "{black:?}");
        // The 16-bit Lab white point is 0xff00, not 0xffff.
        let white = lab_to_xyz_d50([65280.0 / 65535.0, 0.5, 0.5]);
        assert!((white[1] - 1.0).abs() < 0.005, "{white:?}");
    }

    #[test]
    fn bradford_round_trips() {
        let v = [0.4, 0.5, 0.6];
        let back = bradford_d65_to_d50(bradford_d50_to_d65(v));
        for (a, b) in v.iter().zip(back) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn rejects_non_icc_bytes() {
        assert!(parse_icc(b"not a profile at all").is_err());
        assert!(parse_icc(&[0u8; 200]).is_err());
        assert!(Profile::from_bytes(b"not a profile at all").is_err());
        assert!(Profile::from_bytes(&[]).is_err());
    }

    #[test]
    fn parses_srgb_matrix_profile() {
        let Some(path) = srgb_profile_path() else { return };
        let profile = Profile::from_file(&path).unwrap();
        assert_eq!(profile.pcs, Pcs::Xyz);
        assert!(profile.is_matrix_shaper());
        assert!(!profile.name().is_empty());
        // The system sRGB profile is D50 adapted: colourants sum to the D50 white.
        let m = profile.matrix.as_ref().unwrap();
        let white = [
            m.matrix[0][0] + m.matrix[1][0] + m.matrix[2][0],
            m.matrix[0][1] + m.matrix[1][1] + m.matrix[2][1],
            m.matrix[0][2] + m.matrix[1][2] + m.matrix[2][2],
        ];
        assert!((white[0] - 0.9642).abs() < 0.01, "{white:?}");
        assert!((white[1] - 1.0).abs() < 0.01, "{white:?}");
    }

    #[test]
    fn srgb_profile_roundtrips_through_its_own_pcs() {
        let Some(profile) = srgb_profile() else { return };
        for c in [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.2, 0.7, 0.4]] {
            let pcs = srgb_to_xyz_d50(c);
            let device = profile.encode(pcs);
            let back = profile.decode(device);
            let err = back
                .iter()
                .zip(pcs)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(err < 0.01, "roundtrip {c:?} -> {back:?} err {err}");
        }
    }

    #[test]
    fn matrix_inverse_is_exact() {
        let m = [
            [0.4361, 0.3851, 0.1431],
            [0.2225, 0.7169, 0.0606],
            [0.0139, 0.0971, 0.7141],
        ];
        let v = [0.21, 0.19, 0.17];
        let x = solve3(m, v).expect("solvable");
        for row in 0..3 {
            let got = m[row][0] * x[0] + m[row][1] * x[1] + m[row][2] * x[2];
            assert!((got - v[row]).abs() < 1e-5, "row {row}: {got} vs {}", v[row]);
        }
    }

    #[test]
    fn conversion_handles_odd_lengths_and_empty_buffers() {
        let Some(profile) = srgb_profile() else { return };
        let mut buf = vec![10u8; 7];
        convert_srgb_to(&mut buf, &profile);
        assert_eq!(buf.len(), 7);
        let mut empty: Vec<u8> = Vec::new();
        convert_to_srgb(&mut empty, &profile);
        assert!(empty.is_empty());
        let mut one = vec![200u8];
        convert_srgb_to(&mut one, &profile);
        assert_eq!(one[0], 200);
    }

    #[test]
    fn convert_to_srgb_undoes_convert_srgb_to() {
        let Some(profile) = srgb_profile() else { return };
        let mut buf: Vec<u8> = vec![20, 140, 240, 255, 250, 5, 90, 255];
        let before = buf.clone();
        convert_srgb_to(&mut buf, &profile);
        convert_to_srgb(&mut buf, &profile);
        for (a, b) in buf.iter().zip(before.iter()) {
            assert!((*a as i32 - *b as i32).abs() <= 3, "{a} vs {b}");
        }
    }

    #[test]
    fn camera_lut_matches_lcms_reference() {
        let Some(path) = find_profile("NikonZ6III-Generic") else { return };
        let profile = Profile::from_file(&path).unwrap();
        assert_eq!(profile.pcs, Pcs::Lab);
        let lut = profile.a2b.as_ref().expect("A2B0");
        assert_eq!(lut.grid, 33);
        // Ground truth from lcms2 (Pillow ImageCms), 24 colours: neutral ramp,
        // primaries, secondaries and skin/foliage tones.
        let cases: [[i32; 6]; 24] = [
            [0, 0, 0, 0, 0, 0],
            [255, 255, 255, 255, 255, 255],
            [128, 128, 128, 146, 146, 146],
            [64, 64, 64, 81, 81, 81],
            [192, 192, 192, 203, 203, 203],
            [255, 0, 0, 255, 0, 34],
            [0, 255, 0, 0, 255, 0],
            [0, 0, 255, 92, 0, 255],
            [255, 255, 0, 238, 255, 0],
            [0, 255, 255, 0, 255, 232],
            [255, 0, 255, 255, 0, 255],
            [200, 150, 100, 227, 165, 96],
            [64, 32, 16, 94, 40, 8],
            [250, 240, 230, 254, 242, 229],
            [30, 90, 200, 12, 95, 255],
            [120, 60, 30, 157, 73, 27],
            [45, 180, 120, 0, 198, 104],
            [220, 40, 90, 255, 0, 128],
            [10, 10, 10, 10, 10, 10],
            [245, 245, 240, 247, 247, 239],
            [90, 140, 60, 56, 163, 17],
            [160, 170, 190, 171, 183, 217],
            [70, 20, 110, 109, 0, 169],
            [255, 128, 0, 255, 146, 0],
        ];
        let mut worst = 0i32;
        for case in cases {
            let rgb = [case[0], case[1], case[2]];
            let want = [case[3], case[4], case[5]];
            let lin = profile.to_linear_srgb([
                rgb[0] as f32 / 255.0,
                rgb[1] as f32 / 255.0,
                rgb[2] as f32 / 255.0,
            ]);
            let got = lin.map(|v| {
                let v = v.clamp(0.0, 1.0);
                let e = if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
                (e * 255.0).round() as i32
            });
            for k in 0..3 {
                worst = worst.max((got[k] - want[k]).abs());
            }
        }
        assert!(worst <= 2, "worst channel error {worst}/255 vs lcms");
    }

    #[cfg(feature = "external-fixtures")]
    #[test]
    fn other_c1_profile_families_match_lcms() {
        // Ground truth from lcms2 for one profile from each C1 family.
        let cases: [(&str, [[i32; 6]; 5]); 3] = [
            (
                "LeafLF2-LF2 Portrait 5",
                [
                    [128, 128, 128, 141, 139, 139],
                    [255, 0, 0, 255, 0, 0],
                    [0, 0, 255, 0, 0, 194],
                    [200, 150, 100, 254, 142, 109],
                    [30, 90, 200, 0, 108, 209],
                ],
            ),
            (
                "NikonZ6-EcommStandard",
                [
                    [128, 128, 128, 146, 146, 146],
                    [255, 0, 0, 255, 0, 12],
                    [0, 0, 255, 76, 0, 247],
                    [200, 150, 100, 226, 165, 108],
                    [30, 90, 200, 0, 108, 217],
                ],
            ),
            (
                "Phase One Effects - Sepia",
                [
                    [128, 128, 128, 167, 126, 107],
                    [255, 0, 0, 176, 137, 121],
                    [0, 0, 255, 78, 47, 30],
                    [200, 150, 100, 194, 163, 152],
                    [30, 90, 200, 128, 84, 62],
                ],
            ),
        ];
        let mut checked = 0;
        for (name, rows) in cases {
            let Some(path) = find_profile(name) else { continue };
            let profile = Profile::from_file(&path).unwrap();
            checked += 1;
            for row in rows {
                let lin = profile.to_linear_srgb([
                    row[0] as f32 / 255.0,
                    row[1] as f32 / 255.0,
                    row[2] as f32 / 255.0,
                ]);
                let got = lin.map(|v| (delinearize_srgb(v) * 255.0).round() as i32);
                for k in 0..3 {
                    assert!(
                        (got[k] - row[3 + k]).abs() <= 2,
                        "{name} {:?} ch{k}: got {got:?} want {:?}",
                        [row[0], row[1], row[2]],
                        [row[3], row[4], row[5]]
                    );
                }
            }
        }
        assert!(checked > 0, "no family samples found");
    }

    #[test]
    fn most_c1_profiles_parse_successfully() {
        let dir = profile_dir();
        if !dir.is_dir() {
            return;
        }
        let files = scan_profiles(&[dir]);
        // The full Capture One install has 1553; the repo ships one per family
        // (DSLR / DB / EComm / Effects / Gray / matrix) so every shape is
        // exercised even when the install is absent.
        if files.len() < 4 {
            println!("only {} profiles available, skipping", files.len());
            return;
        }
        let mut parsed = 0usize;
        let mut with_lut = 0usize;
        let mut with_matrix = 0usize;
        for (_, path) in &files {
            if let Ok(p) = Profile::from_file(path) {
                parsed += 1;
                if p.a2b.is_some() {
                    with_lut += 1;
                }
                if p.matrix.is_some() {
                    with_matrix += 1;
                }
            }
        }
        println!("C1 profiles: {parsed}/{} parsed, {with_lut} with A2B0 LUT, {with_matrix} matrix/TRC", files.len());
        assert_eq!(parsed, files.len(), "every C1 profile must parse");
        assert!(with_lut > 0 && with_matrix > 0, "samples must cover both families");
    }

    #[cfg(feature = "external-fixtures")]
    #[test]
    fn scan_profiles_lists_c1_profiles() {
        let dir = profile_dir();
        if !dir.is_dir() {
            return;
        }
        let found = scan_profiles(&[dir.clone()]);
        assert!(!found.is_empty(), "no profiles below {}", dir.display());
        if found.len() >= 100 {
            assert!(found.len() > 1000, "found {}", found.len());
        }
        assert!(found.iter().any(|(n, _)| n.contains("sRGB")));
    }

    #[test]
    fn truncated_tables_do_not_panic() {
        let junk = b"mft2\0\0\0\0\x03\x03\x21\0";
        assert!(parse_mft2(junk).is_err());
        let mut short = vec![0u8; 60];
        short[0..4].copy_from_slice(b"mft2");
        short[8] = 3;
        short[9] = 3;
        short[10] = 33;
        assert!(parse_mft2(&short).is_err());
    }
}
