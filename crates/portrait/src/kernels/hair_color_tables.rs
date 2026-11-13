//! PF30009 256x256 tables: address math and GL-style sampling.
//!
//! Ported from `HSLTuner.frag`
//! (`C48D82AF9F45AB0EB8914BDF9D12E5A39B85F5BD7D343196547F8F0F3B02B080`):
//! `index = floor(value * 65535)`, `row = index >> 8`, `col = index & 255`,
//! `uv = (col/256, row/256)`.
//!
//! Filter and wrap are **caller inputs**. The shader only proves the address;
//! its sampler state is not recovered, so nothing here defaults to NEAREST,
//! LINEAR, or an identity table. The caller supplies real texels.

use crate::Refusal;

pub const SIDE: usize = 256;
pub const LEN: usize = SIDE * SIDE;

/// Sampler filter. No `Default`: the vendor state is unresolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    Nearest,
    Linear,
}

/// Sampler wrap. The shader clamps its own coordinates, so these differ only
/// at the `1.0` boundary; the caller still has to choose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wrap {
    ClampToEdge,
    Repeat,
    MirroredRepeat,
}

/// `floor(value * 65535)`, as the shader computes it.
pub fn lut_index(value: f32) -> usize {
    let clamped = if value.is_nan() { 0.0 } else { value.clamp(0.0, 1.0) };
    (clamped * 65535.0).floor().clamp(0.0, 65535.0) as usize
}

fn wrap_index(index: i64, side: i64, wrap: Wrap) -> i64 {
    match wrap {
        Wrap::ClampToEdge => index.clamp(0, side - 1),
        Wrap::Repeat => index.rem_euclid(side),
        Wrap::MirroredRepeat => {
            let period = 2 * side;
            let folded = index.rem_euclid(period);
            if folded < side {
                folded
            } else {
                period - 1 - folded
            }
        }
    }
}

/// GL-style sample of one channel on a `width x height` grid, explicit filter
/// and wrap. No allocation, no validation: callers pass an already checked
/// slice (see `Table256::new`), so this stays usable inside a pixel loop.
pub fn sample_grid(
    filter: Filter,
    wrap: Wrap,
    width: usize,
    height: usize,
    texels: &[f32],
    u: f32,
    v: f32,
) -> f32 {
    let (w, h) = (width as i64, height as i64);
    let fetch = |x: i64, y: i64| {
        texels[(wrap_index(y, h, wrap) * w + wrap_index(x, w, wrap)) as usize]
    };
    match filter {
        Filter::Nearest => fetch(
            (u * width as f32).floor() as i64,
            (v * height as f32).floor() as i64,
        ),
        Filter::Linear => {
            let px = u * width as f32 - 0.5;
            let py = v * height as f32 - 0.5;
            let (x0, y0) = (px.floor(), py.floor());
            let (tx, ty) = (px - x0, py - y0);
            let (x0, y0) = (x0 as i64, y0 as i64);
            let top = fetch(x0, y0) * (1.0 - tx) + fetch(x0 + 1, y0) * tx;
            let bottom = fetch(x0, y0 + 1) * (1.0 - tx) + fetch(x0 + 1, y0 + 1) * tx;
            top * (1.0 - ty) + bottom * ty
        }
    }
}

/// One 256x256 single-channel table (row-major, GL_R32F texels).
pub struct Table256<'a> {
    texels: &'a [f32],
    filter: Filter,
    wrap: Wrap,
}

impl<'a> Table256<'a> {
    pub fn new(texels: &'a [f32], filter: Filter, wrap: Wrap) -> Result<Self, Refusal> {
        if texels.len() != LEN {
            return Err(Refusal::ShapeMismatch("PF30009_TABLE"));
        }
        if texels.iter().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("PF30009_TABLE"));
        }
        Ok(Self {
            texels,
            filter,
            wrap,
        })
    }

    pub fn filter(&self) -> Filter {
        self.filter
    }

    pub fn wrap(&self) -> Wrap {
        self.wrap
    }

    /// Addresses the table like the shader, then samples with the caller's
    /// filter and wrap.
    pub fn sample(&self, value: f32) -> Result<f32, Refusal> {
        if value.is_nan() {
            return Err(Refusal::ValueOutOfRange("PF30009_TABLE_INDEX"));
        }
        let index = lut_index(value);
        let row = index / SIDE;
        let col = index % SIDE;
        Ok(sample_grid(
            self.filter,
            self.wrap,
            SIDE,
            SIDE,
            self.texels,
            col as f32 / SIDE as f32,
            row as f32 / SIDE as f32,
        ))
    }
}

/// Colour matrix applied before the tonal decode (shader `raw_*`).
pub const GAMMA_WEIGHTS: [[f32; 3]; 3] = [
    [0.529_258_31, 0.330_156_21, 0.140_585_48],
    [0.098_386_407, 0.873_452_07, 0.028_161_526],
    [0.016_864_598, 0.117_641_75, 0.865_493_66],
];

/// Inverse matrix applied before the final `invgamma_map_texture` lookups.
pub const INV_GAMMA_WEIGHTS: [[f32; 3]; 3] = [
    [2.034_487_2, -0.727_695_41, -0.306_791_79],
    [-0.228_891_37, 1.231_791_83, -0.002_900_47],
    [-0.008_531_12, -0.153_251_1, 1.161_782_2],
];

pub fn apply_matrix(matrix: [[f32; 3]; 3], rgb: [f32; 3]) -> [f32; 3] {
    [
        matrix[0][0] * rgb[0] + matrix[0][1] * rgb[1] + matrix[0][2] * rgb[2],
        matrix[1][0] * rgb[0] + matrix[1][1] * rgb[1] + matrix[1][2] * rgb[2],
        matrix[2][0] * rgb[0] + matrix[2][1] * rgb[1] + matrix[2][2] * rgb[2],
    ]
}

/// Shader `CalMiddleMapRes`.
fn middle_map(max_val: f32, mid_val: f32, min_val: f32, map_max: f32, map_min: f32) -> f32 {
    let diff = max_val - min_val;
    if map_max > map_min {
        map_min + (mid_val - min_val) * (map_max - map_min) / diff
    } else {
        map_max + (mid_val - max_val) * (map_min - map_max) / diff
    }
}

/// `invconstrast_texture` / `constrast_texture` stage: max and min lookups plus
/// the middle-channel interpolation.
pub fn decode_tonal(table: &Table256<'_>, raw: [f32; 3]) -> Result<[f32; 3], Refusal> {
    let max_val = raw[0].max(raw[1]).max(raw[2]);
    let min_val = raw[0].min(raw[1]).min(raw[2]);
    if max_val == min_val {
        let value = table.sample(min_val)?;
        return Ok([value, value, value]);
    }
    let map_max = table.sample(max_val)?;
    let map_min = table.sample(min_val)?;
    let mut out = [0.0f32; 3];
    for channel in 0..3 {
        out[channel] = if raw[channel] == max_val {
            map_max
        } else if raw[channel] == min_val {
            map_min
        } else {
            middle_map(max_val, raw[channel], min_val, map_max, map_min)
        };
    }
    Ok(out)
}

/// Final per-channel `invgamma_map_texture` lookups.
pub fn encode_inv_gamma(table: &Table256<'_>, rgb: [f32; 3]) -> Result<[f32; 3], Refusal> {
    Ok([
        table.sample(rgb[0])?,
        table.sample(rgb[1])?,
        table.sample(rgb[2])?,
    ])
}

/// The four explicit 256x256 tables. No table has a default or identity fill.
pub struct Pf30009Tables<'a> {
    pub gamma: Table256<'a>,
    pub inv_contrast: Table256<'a>,
    pub contrast: Table256<'a>,
    pub inv_gamma: Table256<'a>,
}

impl<'a> Pf30009Tables<'a> {
    /// Every table is mandatory; a missing one is a typed refusal naming it.
    pub fn from_slices(
        gamma: Option<&'a [f32]>,
        inv_contrast: Option<&'a [f32]>,
        contrast: Option<&'a [f32]>,
        inv_gamma: Option<&'a [f32]>,
        filter: Filter,
        wrap: Wrap,
    ) -> Result<Self, Refusal> {
        let gamma = gamma.ok_or(Refusal::MissingInput("PF30009_GAMMA_TABLE"))?;
        let inv_contrast =
            inv_contrast.ok_or(Refusal::MissingInput("PF30009_INV_CONTRAST_TABLE"))?;
        let contrast = contrast.ok_or(Refusal::MissingInput("PF30009_CONTRAST_TABLE"))?;
        let inv_gamma = inv_gamma.ok_or(Refusal::MissingInput("PF30009_INV_GAMMA_TABLE"))?;
        Ok(Self {
            gamma: Table256::new(gamma, filter, wrap)?,
            inv_contrast: Table256::new(inv_contrast, filter, wrap)?,
            contrast: Table256::new(contrast, filter, wrap)?,
            inv_gamma: Table256::new(inv_gamma, filter, wrap)?,
        })
    }
}
