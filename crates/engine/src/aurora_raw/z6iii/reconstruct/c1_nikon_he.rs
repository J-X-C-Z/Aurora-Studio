//! Nikon HE VCGH Bayer reconstruction from signed JPEG XS coefficient bands.
//! This stage accepts the coefficients, not NEF container bytes or metadata.

use std::io::{Error, ErrorKind, Result};
use std::sync::OnceLock;

use rayon::prelude::*;

use super::nikon_he_bitstream::{self, Decoder, PACKET_BANDS, TOTAL_COEFFS, Weights};

const RECIPROCALS: [u32; 15] = [
    0x15555, 0x12492, 0x11111, 0x10842, 0x10410, 0x10204, 0x10101, 0x10080, 0x10040, 0x10020,
    0x10010, 0x10008, 0x10004, 0x10002, 0x10001,
];

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

pub fn dequantize(magnitude: u16, negative: bool, bit_depth: u8, gtli: u8) -> Result<i32> {
    if bit_depth > 15 || gtli > 15 || u32::from(magnitude) >= 1_u32 << bit_depth {
        return Err(invalid("invalid HE coefficient magnitude/depth/GTLI"));
    }
    if bit_depth == 0 {
        return Ok(0);
    }
    let product = u32::from(magnitude).wrapping_mul(RECIPROCALS[usize::from(bit_depth - 1)]);
    let level = ((product >> (16 - gtli)) << 4) as i32;
    Ok(if negative { -level } else { level })
}

#[inline]
fn lift53_into(low: &[i64], high: &[i32], out: &mut [i64]) {
    let n_low = low.len();
    let n_high = high.len();
    debug_assert_eq!(out.len(), n_low + n_high);
    if n_low > 0 {
        let h0 = i64::from(high[0]);
        out[0] = low[0] - ((h0 + h0 + 2) >> 2);
        let mid_end = n_low.min(n_high);
        for index in 1..mid_end {
            let left = i64::from(high[index - 1]);
            let right = i64::from(high[index]);
            out[2 * index] = low[index] - ((left + right + 2) >> 2);
        }
        if n_low > n_high {
            let h_last = i64::from(high[n_high - 1]);
            out[2 * n_high] = low[n_high] - ((h_last + h_last + 2) >> 2);
        }
    }
    let last_even = 2 * (n_low - 1);
    for index in 0..n_high {
        let left = 2 * index;
        let right = if left + 2 <= last_even {
            left + 2
        } else {
            last_even
        };
        out[left + 1] = i64::from(high[index]) + ((out[left] + out[right]) >> 1);
    }
}

fn lift53(low: &[i64], high: &[i32]) -> Vec<i64> {
    let mut line = vec![0; low.len() + high.len()];
    lift53_into(low, high, &mut line);
    line
}

/// Sizes of coefficient bands in wire order, including code-group padding.
pub fn band_lengths(cell_width: usize, packet: usize) -> Result<Vec<usize>> {
    if !(32..=32767).contains(&cell_width) || packet >= 8 {
        return Err(invalid("unsupported HE width/packet index"));
    }
    let levels = if packet % 4 == 2 {
        0
    } else if packet < 4 {
        5
    } else {
        1
    };
    let mut width = cell_width;
    let mut lengths = Vec::with_capacity(levels + 1);
    for _ in 0..levels {
        lengths.push((width / 2).next_multiple_of(4));
        width = width.div_ceil(2);
    }
    lengths.push(width.next_multiple_of(4));
    lengths.reverse();
    Ok(lengths)
}

/// Horizontally reconstructed low/high lines; component 2 instead holds two direct Yh rows.
pub struct Precinct {
    pub low: [Vec<i64>; 4],
    pub high: [Vec<i64>; 4],
}

fn horizontal(width: usize, bands: &[Vec<i32>]) -> Vec<i64> {
    let levels = bands.len() - 1;
    let low_count = width.div_ceil(1 << levels);
    let mut line: Vec<i64> = bands[0][..low_count]
        .iter()
        .copied()
        .map(i64::from)
        .collect();
    for (index, band) in bands.iter().enumerate().skip(1) {
        let output_width = width.div_ceil(1 << (levels - index));
        line = lift53(&line, &band[..output_width / 2]);
    }
    line
}

#[inline]
fn horizontal_from_flat_i32(
    width: usize,
    slots: &[usize],
    coeff_buf: &[i32; TOTAL_COEFFS],
    scratch_a: &mut [i64],
    scratch_b: &mut [i64],
    dst: &mut [i32],
) {
    let levels = slots.len() - 1;
    if levels == 0 {
        let band = nikon_he_bitstream::slot_slice(coeff_buf, slots[0]);
        dst[..width].copy_from_slice(&band[..width]);
        return;
    }
    let mut cur_len = width.div_ceil(1 << levels);
    let band0 = nikon_he_bitstream::slot_slice(coeff_buf, slots[0]);
    for (d, &s) in scratch_a[..cur_len].iter_mut().zip(&band0[..cur_len]) {
        *d = i64::from(s);
    }
    let mut in_a = true;
    for (index, &slot) in slots.iter().enumerate().skip(1) {
        let output_width = width.div_ceil(1 << (levels - index));
        let high = &nikon_he_bitstream::slot_slice(coeff_buf, slot)[..output_width / 2];
        if in_a {
            lift53_into(&scratch_a[..cur_len], high, &mut scratch_b[..output_width]);
        } else {
            lift53_into(&scratch_b[..cur_len], high, &mut scratch_a[..output_width]);
        }
        cur_len = output_width;
        in_a = !in_a;
    }
    let final_src = if in_a {
        &scratch_a[..width]
    } else {
        &scratch_b[..width]
    };
    for (d, &s) in dst[..width].iter_mut().zip(final_src) {
        *d = s as i32;
    }
}

pub fn reconstruct_precinct(cell_width: usize, packets: &[Vec<Vec<i32>>; 8]) -> Result<Precinct> {
    for (index, packet) in packets.iter().enumerate() {
        let expected = band_lengths(cell_width, index)?;
        if !expected.iter().copied().eq(packet.iter().map(Vec::len)) {
            return Err(invalid("wrong HE coefficient band geometry"));
        }
    }
    Ok(Precinct {
        low: std::array::from_fn(|component| horizontal(cell_width, &packets[component])),
        high: std::array::from_fn(|component| horizontal(cell_width, &packets[component + 4])),
    })
}

#[inline]
fn inverse_curve_compute(value: i64) -> u16 {
    const OFFSET_SQRT: i64 = 16255;
    const OFFSET_16: i64 = 4032;
    const RFLIP: i64 = 84429;
    let input = value.clamp(0, 65535);
    let curve = if input <= OFFSET_SQRT {
        OFFSET_16 - (((OFFSET_SQRT - input).pow(2) + 0x8000) >> 16)
    } else {
        let linear = (RFLIP * (input - OFFSET_SQRT) + 0x8000) >> 16;
        (((linear.pow(2) + 0x8000) >> 16) + OFFSET_16).min(65535)
    };
    ((2 + curve) >> 2).clamp(0, 16383) as u16
}

fn inverse_curve_lut() -> &'static [u16; 65536] {
    static LUT: OnceLock<Box<[u16; 65536]>> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = vec![0u16; 65536].into_boxed_slice();
        for (i, entry) in table.iter_mut().enumerate() {
            *entry = inverse_curve_compute(i as i64);
        }
        table.try_into().unwrap()
    })
}

#[inline(always)]
fn inverse_curve(value: i64) -> u16 {
    inverse_curve_lut()[value.clamp(0, 65535) as usize]
}

trait PrecinctSource: Sync {
    fn precinct_count(&self) -> usize;
    fn fill_row_into(
        &self,
        width: usize,
        cell_rows: usize,
        has_bottom: bool,
        cell_row: usize,
        dst: &mut [Vec<i64>; 4],
    );
}

impl PrecinctSource for [Precinct] {
    #[inline(always)]
    fn precinct_count(&self) -> usize {
        self.len()
    }
    #[inline]
    fn fill_row_into(
        &self,
        width: usize,
        cell_rows: usize,
        has_bottom: bool,
        cell_row: usize,
        dst: &mut [Vec<i64>; 4],
    ) {
        let cell_row = if has_bottom {
            cell_row.min(cell_rows - 1)
        } else {
            cell_row
        };
        let index = cell_row / 2;
        let prev_idx = index.saturating_sub(1);
        let is_even = cell_row % 2 == 0;
        let at_end = index + 1 == self.len();

        let yh_src = if is_even {
            &self[index].low[2]
        } else {
            &self[index].high[2]
        };
        dst[2].copy_from_slice(yh_src);

        for component in [0usize, 1, 3] {
            let cur_low = &self[index].low[component];
            let cur_high = &self[index].high[component];
            let prev_high = &self[prev_idx].high[component];
            let out = &mut dst[component];
            if is_even {
                for col in 0..width {
                    out[col] = (4 * cur_low[col] - prev_high[col] - cur_high[col] + 1) >> 2;
                }
            } else if at_end {
                for col in 0..width {
                    let detail = cur_high[col];
                    let even = (4 * cur_low[col] - prev_high[col] - detail + 1) >> 2;
                    out[col] = detail + even;
                }
            } else {
                let next_low = &self[index + 1].low[component];
                let next_high = &self[index + 1].high[component];
                for col in 0..width {
                    let detail = cur_high[col];
                    let even = (4 * cur_low[col] - prev_high[col] - detail + 1) >> 2;
                    let next_even = (4 * next_low[col] - detail - next_high[col] + 1) >> 2;
                    out[col] = detail + ((even + next_even) >> 1);
                }
            }
        }
    }
}

struct FlatPrecincts {
    data: Vec<i32>,
    width: usize,
    count: usize,
}

impl FlatPrecincts {
    #[inline(always)]
    fn line(&self, precinct_idx: usize, line_idx: usize) -> &[i32] {
        let off = (precinct_idx * 8 + line_idx) * self.width;
        &self.data[off..off + self.width]
    }
}

impl PrecinctSource for FlatPrecincts {
    #[inline(always)]
    fn precinct_count(&self) -> usize {
        self.count
    }
    #[inline]
    fn fill_row_into(
        &self,
        width: usize,
        cell_rows: usize,
        has_bottom: bool,
        cell_row: usize,
        dst: &mut [Vec<i64>; 4],
    ) {
        let cell_row = if has_bottom {
            cell_row.min(cell_rows - 1)
        } else {
            cell_row
        };
        let index = cell_row / 2;
        let prev_idx = index.saturating_sub(1);
        let is_even = cell_row % 2 == 0;
        let at_end = index + 1 == self.count;

        let yh_src = if is_even {
            self.line(index, 2)
        } else {
            self.line(index, 6)
        };
        for (d, &s) in dst[2][..width].iter_mut().zip(&yh_src[..width]) {
            *d = i64::from(s);
        }

        for component in [0usize, 1, 3] {
            let cur_low = self.line(index, component);
            let cur_high = self.line(index, component + 4);
            let prev_high = self.line(prev_idx, component + 4);
            let out = &mut dst[component];
            if is_even {
                for col in 0..width {
                    out[col] = (4 * i64::from(cur_low[col])
                        - i64::from(prev_high[col])
                        - i64::from(cur_high[col])
                        + 1)
                        >> 2;
                }
            } else if at_end {
                for col in 0..width {
                    let detail = i64::from(cur_high[col]);
                    let even =
                        (4 * i64::from(cur_low[col]) - i64::from(prev_high[col]) - detail + 1) >> 2;
                    out[col] = detail + even;
                }
            } else {
                let next_low = self.line(index + 1, component);
                let next_high = self.line(index + 1, component + 4);
                for col in 0..width {
                    let detail = i64::from(cur_high[col]);
                    let even =
                        (4 * i64::from(cur_low[col]) - i64::from(prev_high[col]) - detail + 1) >> 2;
                    let next_even =
                        (4 * i64::from(next_low[col]) - detail - i64::from(next_high[col]) + 1)
                            >> 2;
                    out[col] = detail + ((even + next_even) >> 1);
                }
            }
        }
    }
}

/// Reconstruct Bayer rows from image origin; trailing precincts supply vertical context.
/// The 14-bit curve is the concrete variant verified against the Z6III inputs.
pub fn bayer_prefix(precincts: &[Precinct], output_rows: usize) -> Result<Vec<u16>> {
    assemble_bayer(precincts, output_rows, false)
}

pub fn bayer_frame(precincts: &[Precinct], output_rows: usize) -> Result<Vec<u16>> {
    if output_rows % 4 != 0 || output_rows / 4 != precincts.len() {
        return Err(invalid("HE frame must contain complete precincts"));
    }
    assemble_bayer(precincts, output_rows, true)
}

fn assemble_bayer(
    precincts: &[Precinct],
    output_rows: usize,
    has_bottom: bool,
) -> Result<Vec<u16>> {
    let Some(first) = precincts.first() else {
        return Err(invalid("missing HE precincts"));
    };
    let precinct_count = output_rows.div_ceil(4) + usize::from(!has_bottom);
    if output_rows == 0 || output_rows > 65535 || precinct_count > precincts.len() {
        return Err(invalid("Bayer prefix needs one trailing context precinct"));
    }
    let width = first.low[0].len();
    let precincts = &precincts[..precinct_count];
    if !(32..=32767).contains(&width)
        || precincts.iter().any(|precinct| {
            precinct
                .low
                .iter()
                .chain(&precinct.high)
                .any(|line| line.len() != width)
        })
    {
        return Err(invalid("inconsistent HE precinct width"));
    }
    assemble_bayer_from_source(precincts, width, output_rows, has_bottom)
}

struct AssembleScratch {
    row_a: [Vec<i64>; 4],
    row_b: [Vec<i64>; 4],
    row_c: [Vec<i64>; 4],
    row_d: [Vec<i64>; 4],
    green0: Vec<i64>,
    green1: Vec<i64>,
    next_green0: Vec<i64>,
    next_green1: Vec<i64>,
    above_green1: Vec<i64>,
}

impl AssembleScratch {
    fn new(width: usize) -> Self {
        Self {
            row_a: std::array::from_fn(|_| vec![0i64; width]),
            row_b: std::array::from_fn(|_| vec![0i64; width]),
            row_c: std::array::from_fn(|_| vec![0i64; width]),
            row_d: std::array::from_fn(|_| vec![0i64; width]),
            green0: vec![0i64; width],
            green1: vec![0i64; width],
            next_green0: vec![0i64; width],
            next_green1: vec![0i64; width],
            above_green1: vec![0i64; width],
        }
    }
}

#[inline]
fn fill_row<S: PrecinctSource + ?Sized>(
    source: &S,
    width: usize,
    cell_rows: usize,
    has_bottom: bool,
    cell_row: usize,
    dst: &mut [Vec<i64>; 4],
) {
    source.fill_row_into(width, cell_rows, has_bottom, cell_row, dst);
}

#[inline]
fn fill_green(
    width: usize,
    above: &[Vec<i64>; 4],
    current: &[Vec<i64>; 4],
    below: &[Vec<i64>; 4],
    bottom: bool,
    green0: &mut [i64],
    green1: &mut [i64],
) {
    let yl = &current[0];
    let cr = &current[1];
    let yh = &current[2];
    let cb = &current[3];
    let above_yh = &above[2];
    let above_cb = &above[3];
    let below_yl = &below[0];
    let below_cr = &below[1];
    let below_yh = &below[2];

    let mut y0_left = 0i64;
    let mut y0_below_left = 0i64;
    for col in 0..width {
        let left = col.saturating_sub(1);
        let right = (col + 1).min(width - 1);
        let y0 = yl[col] - ((above_yh[col] + above_yh[right] + yh[col] + yh[right]) >> 3);
        let y0_below = if bottom {
            y0
        } else {
            below_yl[col] - ((yh[col] + yh[right] + below_yh[col] + below_yh[right]) >> 3)
        };
        if col == 0 {
            y0_left = y0;
            y0_below_left = y0_below;
        }
        green0[col] = y0 - ((cb[col] + above_cb[col] + cr[col] + cr[right]) >> 3);
        green1[col] = ((y0_below + y0 + y0_below_left + y0_left) >> 2) + yh[col]
            - ((cb[left] + cb[col] + cr[col] + below_cr[col]) >> 3);
        y0_left = y0;
        y0_below_left = y0_below;
    }
}

fn assemble_bayer_from_source<S: PrecinctSource + ?Sized>(
    source: &S,
    width: usize,
    output_rows: usize,
    has_bottom: bool,
) -> Result<Vec<u16>> {
    let cell_rows = output_rows.div_ceil(2);
    let row_stride = 2 * width;
    let sample_count = row_stride
        .checked_mul(output_rows)
        .ok_or_else(|| invalid("HE dimensions overflow"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(sample_count)
        .map_err(Error::other)?;
    output.resize(sample_count, 0);

    let lut = inverse_curve_lut();
    const BAND_CELL_ROWS: usize = 32;
    let band_sample_stride = BAND_CELL_ROWS * 2 * row_stride;

    output
        .par_chunks_mut(band_sample_stride)
        .enumerate()
        .for_each_init(
            || AssembleScratch::new(width),
            |scratch, (band_idx, out_chunk)| {
                let start_cell = band_idx * BAND_CELL_ROWS;
                let end_cell = (start_cell + BAND_CELL_ROWS).min(cell_rows);
                if start_cell >= end_cell {
                    return;
                }

                // Initialize sliding state at `start_cell` to match the sequential loop identically:
                // - row_a holds `current = row(start_cell)`
                // - row_b holds `below = row(start_cell + 1)`
                // - green0, green1 hold green(row(start_cell - 1 or 0), row(start_cell), row(start_cell + 1), bottom_cur)
                // - above_green1 holds green1 from `start_cell - 1` (or green1 itself when start_cell == 0)
                fill_row(source, width, cell_rows, has_bottom, start_cell, &mut scratch.row_a);
                fill_row(source, width, cell_rows, has_bottom, start_cell + 1, &mut scratch.row_b);

                if start_cell == 0 {
                    fill_green(
                        width,
                        &scratch.row_a,
                        &scratch.row_a,
                        &scratch.row_b,
                        false,
                        &mut scratch.green0,
                        &mut scratch.green1,
                    );
                    scratch.above_green1.copy_from_slice(&scratch.green1);
                } else {
                    fill_row(
                        source,
                        width,
                        cell_rows,
                        has_bottom,
                        start_cell - 1,
                        &mut scratch.row_c,
                    );
                    if start_cell == 1 {
                        // Previous step was index = 0: green(row(0), row(0), row(1), false)
                        fill_green(
                            width,
                            &scratch.row_c,
                            &scratch.row_c,
                            &scratch.row_a,
                            false,
                            &mut scratch.next_green0,
                            &mut scratch.above_green1,
                        );
                    } else {
                        fill_row(
                            source,
                            width,
                            cell_rows,
                            has_bottom,
                            start_cell - 2,
                            &mut scratch.row_d,
                        );
                        fill_green(
                            width,
                            &scratch.row_d,
                            &scratch.row_c,
                            &scratch.row_a,
                            false, // start_cell < cell_rows => has_bottom && start_cell >= cell_rows is false
                            &mut scratch.next_green0,
                            &mut scratch.above_green1,
                        );
                    }
                    fill_green(
                        width,
                        &scratch.row_c,
                        &scratch.row_a,
                        &scratch.row_b,
                        has_bottom && start_cell + 1 >= cell_rows,
                        &mut scratch.green0,
                        &mut scratch.green1,
                    );
                }

                for index in start_cell..end_cell {
                    let has_next_context = index + 1 < cell_rows;
                    if has_next_context {
                        fill_row(
                            source,
                            width,
                            cell_rows,
                            has_bottom,
                            index + 2,
                            &mut scratch.row_c,
                        );
                    }
                    let next_below = if has_next_context {
                        &scratch.row_c
                    } else {
                        &scratch.row_b
                    };
                    fill_green(
                        width,
                        &scratch.row_a,
                        &scratch.row_b,
                        next_below,
                        has_bottom && index + 2 >= cell_rows,
                        &mut scratch.next_green0,
                        &mut scratch.next_green1,
                    );

                    let local_cell = index - start_cell;
                    let even_row_out = &mut out_chunk[local_cell * 2 * row_stride
                        ..(local_cell * 2 + 1) * row_stride];
                    let cr_row = &scratch.row_a[1];
                    let g0 = &scratch.green0;
                    let g1 = &scratch.green1;
                    let ag1 = &scratch.above_green1;

                    // col = 0
                    {
                        let red = ((g1[0] + g0[0] + g0[0] + ag1[0]) >> 2) + cr_row[0];
                        even_row_out[0] = lut[(red + 32768).clamp(0, 65535) as usize];
                        even_row_out[1] = lut[(g0[0] + 32768).clamp(0, 65535) as usize];
                    }
                    for col in 1..width {
                        let red = ((g1[col] + g0[col] + g0[col - 1] + ag1[col]) >> 2) + cr_row[col];
                        let dst = col * 2;
                        even_row_out[dst] = lut[(red + 32768).clamp(0, 65535) as usize];
                        even_row_out[dst + 1] = lut[(g0[col] + 32768).clamp(0, 65535) as usize];
                    }

                    if 2 * index + 1 < output_rows {
                        let odd_row_out = &mut out_chunk[(local_cell * 2 + 1) * row_stride
                            ..(local_cell * 2 + 2) * row_stride];
                        let cb_row = &scratch.row_a[3];
                        let bg0: &[i64] = if has_bottom && index + 1 == cell_rows {
                            &scratch.green0
                        } else {
                            &scratch.next_green0
                        };
                        let last = width - 1;
                        for col in 0..last {
                            let blue =
                                ((g1[col] + g0[col] + g1[col + 1] + bg0[col]) >> 2) + cb_row[col];
                            let dst = col * 2;
                            odd_row_out[dst] = lut[(g1[col] + 32768).clamp(0, 65535) as usize];
                            odd_row_out[dst + 1] = lut[(blue + 32768).clamp(0, 65535) as usize];
                        }
                        {
                            let blue =
                                ((g1[last] + g0[last] + g1[last] + bg0[last]) >> 2) + cb_row[last];
                            let dst = last * 2;
                            odd_row_out[dst] = lut[(g1[last] + 32768).clamp(0, 65535) as usize];
                            odd_row_out[dst + 1] = lut[(blue + 32768).clamp(0, 65535) as usize];
                        }
                    }

                    if has_next_context && index + 1 < end_cell {
                        std::mem::swap(&mut scratch.row_a, &mut scratch.row_b);
                        std::mem::swap(&mut scratch.row_b, &mut scratch.row_c);
                        std::mem::swap(&mut scratch.green0, &mut scratch.next_green0);
                        std::mem::swap(&mut scratch.above_green1, &mut scratch.green1);
                        std::mem::swap(&mut scratch.green1, &mut scratch.next_green1);
                    }
                }
            },
        );

    Ok(output)
}

/// Scans `rest` in parallel for exact 6-byte slice headers `[0xff, 0x20, 0x00, 0x04, hi, lo]`.
/// Returns `Some(offsets)` only if slice 0 starts at 0 and every required slice `0..num_slices`
/// has a unique, strictly increasing offset. Any ambiguity triggers a safe sequential fallback.
fn find_slice_offsets(rest: &[u8], num_slices: usize) -> Option<Vec<usize>> {
    if num_slices == 0 || rest.len() < 6 {
        return None;
    }
    if rest[..6] != [0xff, 0x20, 0x00, 0x04, 0x00, 0x00] {
        return None;
    }
    if num_slices == 1 {
        return Some(vec![0]);
    }

    let len = rest.len();
    let chunk_size = (len / 16).max(65536);
    let num_chunks = len.div_ceil(chunk_size);
    let candidates: Vec<(usize, usize)> = (0..num_chunks)
        .into_par_iter()
        .flat_map_iter(|chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(len);
            let mut local = Vec::new();
            let mut pos = start;
            let limit = end.min(len.saturating_sub(5));
            while pos < limit {
                if let Some(rel) = rest[pos..limit].iter().position(|&b| b == 0xff) {
                    pos += rel;
                    if rest[pos + 1] == 0x20 && rest[pos + 2] == 0x00 && rest[pos + 3] == 0x04 {
                        let idx = usize::from(u16::from_be_bytes([rest[pos + 4], rest[pos + 5]]));
                        if idx < num_slices {
                            local.push((idx, pos));
                        }
                        pos += 6;
                    } else {
                        pos += 1;
                    }
                } else {
                    break;
                }
            }
            local
        })
        .collect();

    let mut offsets = vec![usize::MAX; num_slices];
    for (idx, pos) in candidates {
        if offsets[idx] != usize::MAX {
            // Duplicate candidate for the same slice index: fall back to sequential scan.
            return None;
        }
        offsets[idx] = pos;
    }
    if offsets[0] != 0 {
        return None;
    }
    for s in 1..num_slices {
        if offsets[s] == usize::MAX || offsets[s] <= offsets[s - 1] + 6 {
            return None;
        }
    }
    Some(offsets)
}

struct SliceScratch {
    decoder: Decoder,
    coeff_buf: Box<[i32; TOTAL_COEFFS]>,
    scratch_a: Vec<i64>,
    scratch_b: Vec<i64>,
}

impl SliceScratch {
    fn new(weights: Weights, half_width: usize) -> Self {
        Self {
            decoder: Decoder::new(weights),
            coeff_buf: Box::new([0i32; TOTAL_COEFFS]),
            scratch_a: vec![0i64; half_width],
            scratch_b: vec![0i64; half_width],
        }
    }
}

fn decode_slice_into_flat(
    scratch: &mut SliceScratch,
    slice_bytes: &[u8],
    half_width: usize,
    precincts_in_slice: usize,
    require_exact_end: bool,
    flat_slice: &mut [i32],
) -> Result<()> {
    scratch.decoder.reset_slice();
    let mut cursor = slice_bytes;
    let precinct_stride = 8 * half_width;

    for p_idx in 0..precincts_in_slice {
        let consumed = scratch.decoder.decode_precinct_dequantized(
            cursor,
            &mut scratch.coeff_buf,
            &RECIPROCALS,
        )?;
        cursor = &cursor[consumed..];
        let precinct_out = &mut flat_slice[p_idx * precinct_stride..(p_idx + 1) * precinct_stride];
        for (pkt_idx, &slots) in PACKET_BANDS.iter().enumerate() {
            let dst = &mut precinct_out[pkt_idx * half_width..(pkt_idx + 1) * half_width];
            horizontal_from_flat_i32(
                half_width,
                slots,
                &scratch.coeff_buf,
                &mut scratch.scratch_a,
                &mut scratch.scratch_b,
                dst,
            );
        }
    }
    if require_exact_end && !cursor.is_empty() {
        return Err(invalid("slice byte boundary mismatch"));
    }
    Ok(())
}

fn decode_sequential_into_flat(
    mut rest: &[u8],
    weights: Weights,
    half_width: usize,
    precinct_count: usize,
    first_slice_only: bool,
) -> Result<FlatPrecincts> {
    let mut scratch = SliceScratch::new(weights, half_width);
    let precinct_stride = 8 * half_width;
    let mut data = vec![0i32; precinct_count * precinct_stride];

    for index in 0..precinct_count {
        if index % 16 == 0 {
            let header = rest
                .get(..6)
                .ok_or_else(|| Error::new(ErrorKind::UnexpectedEof, "truncated HE marker"))?;
            if header[..2] != [0xff, 0x20] {
                return Err(Error::new(ErrorKind::Unsupported, "unexpected HE marker"));
            }
            let length = usize::from(u16::from_be_bytes([header[2], header[3]]));
            if length < 2 || rest.len() < length + 2 {
                return Err(invalid("invalid HE marker length"));
            }
            let (slice, remaining) = rest.split_at(length + 2);
            rest = remaining;
            if slice[4..] != ((index / 16) as u16).to_be_bytes() {
                return Err(invalid("unexpected HE slice index"));
            }
            scratch.decoder.reset_slice();
        }
        let consumed = scratch.decoder.decode_precinct_dequantized(
            rest,
            &mut scratch.coeff_buf,
            &RECIPROCALS,
        )?;
        rest = &rest[consumed..];
        let precinct_out = &mut data[index * precinct_stride..(index + 1) * precinct_stride];
        for (pkt_idx, &slots) in PACKET_BANDS.iter().enumerate() {
            let dst = &mut precinct_out[pkt_idx * half_width..(pkt_idx + 1) * half_width];
            horizontal_from_flat_i32(
                half_width,
                slots,
                &scratch.coeff_buf,
                &mut scratch.scratch_a,
                &mut scratch.scratch_b,
                dst,
            );
        }
    }
    if !first_slice_only && !rest.is_empty() {
        return Err(invalid("extra HE payload after final precinct"));
    }
    Ok(FlatPrecincts {
        data,
        width: half_width,
        count: precinct_count,
    })
}

/// Parallel 64-slice HE bitstream decode, dequantize, horizontal 5/3 lift, and Bayer assembly.
/// Automatically falls back to sequential slice decoding if slice header scanning is ambiguous
/// or fails exact byte-boundary verification, guaranteeing 100% bit-exact output.
pub(super) fn decode_he_bayer_parallel(
    rest: &[u8],
    weights_marker: &[u8],
    width: usize,
    height: usize,
    first_slice_only: bool,
) -> Result<Vec<u16>> {
    if width % 2 != 0 || height % 4 != 0 || width / 2 != 3032 {
        return Err(Error::new(ErrorKind::Unsupported, "untested HE geometry"));
    }
    let weights = Weights::from_wgt(weights_marker)?;
    let half_width = width / 2;
    let precinct_count = if first_slice_only { 17 } else { height / 4 };
    let num_slices = precinct_count.div_ceil(16);

    let flat = if let Some(offsets) = find_slice_offsets(rest, num_slices) {
        let precinct_stride = 8 * half_width;
        let slice_stride = 16 * precinct_stride;
        let mut data = vec![0i32; precinct_count * precinct_stride];

        let parallel_result: Result<()> = data
            .par_chunks_mut(slice_stride)
            .enumerate()
            .try_for_each_init(
                || SliceScratch::new(weights, half_width),
                |scratch, (slice_idx, flat_slice)| {
                    let start = offsets[slice_idx] + 6;
                    let end = if slice_idx + 1 < num_slices {
                        offsets[slice_idx + 1]
                    } else {
                        rest.len()
                    };
                    let precincts_in_slice = flat_slice.len() / precinct_stride;
                    let require_exact_end = !first_slice_only || slice_idx + 1 < num_slices;
                    decode_slice_into_flat(
                        scratch,
                        &rest[start..end],
                        half_width,
                        precincts_in_slice,
                        require_exact_end,
                        flat_slice,
                    )
                },
            );

        match parallel_result {
            Ok(()) => FlatPrecincts {
                data,
                width: half_width,
                count: precinct_count,
            },
            Err(_) => {
                // Safe fallback: sequential decoding gives authoritative slice boundaries and error reporting.
                decode_sequential_into_flat(rest, weights, half_width, precinct_count, first_slice_only)?
            }
        }
    } else {
        decode_sequential_into_flat(rest, weights, half_width, precinct_count, first_slice_only)?
    };

    if first_slice_only {
        assemble_bayer_from_source(&flat, half_width, 64, false)
    } else {
        assemble_bayer_from_source(&flat, half_width, height, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_curve_lut_matches_exact_formula() {
        let lut = inverse_curve_lut();
        for v in 0..=65535i64 {
            assert_eq!(
                lut[v as usize],
                inverse_curve_compute(v),
                "LUT mismatch at {v}"
            );
        }
        assert_eq!(inverse_curve(-1000), inverse_curve_compute(-1000));
        assert_eq!(inverse_curve(70000), inverse_curve_compute(70000));
    }

    #[test]
    fn retains_low_bits_before_gtli_shift() {
        assert_eq!(dequantize(1, false, 1, 2).unwrap(), 80);
        assert_eq!(dequantize(1, true, 1, 2).unwrap(), -80);
        assert_eq!(dequantize(0, false, 0, 2).unwrap(), 0);
        assert!(dequantize(2, false, 1, 2).is_err());
    }

    #[test]
    fn validates_bands_and_trailing_context() {
        assert_eq!(
            band_lengths(3032, 0).unwrap(),
            [96, 96, 192, 380, 760, 1516]
        );
        assert_eq!(band_lengths(3032, 6).unwrap(), [3032]);
        let zero = std::array::from_fn(|index| {
            band_lengths(32, index)
                .unwrap()
                .into_iter()
                .map(|len| vec![0; len])
                .collect()
        });
        let mut precincts = vec![reconstruct_precinct(32, &zero).unwrap()];
        assert!(bayer_prefix(&precincts, 1).is_err());
        precincts.push(reconstruct_precinct(32, &zero).unwrap());
        assert_eq!(
            bayer_prefix(&precincts, 4).unwrap(),
            vec![inverse_curve(32768); 256]
        );
        let mut malformed = zero;
        malformed[0][0].pop();
        assert!(reconstruct_precinct(32, &malformed).is_err());
    }

    #[test]
    #[ignore = "requires real-NEF coefficient and LibRaw Bayer fixtures"]
    fn matches_real_nef_first_slices() {
        let fixture_dir = std::path::PathBuf::from(std::env::var_os("AURORA_HE_FIXTURES").unwrap());
        for stem in ["_DSC4133", "_DSC4136", "_DSC5673"] {
            let bytes = std::fs::read(fixture_dir.join(format!("{stem}.coeff.i32le"))).unwrap();
            let mut coefficients = bytes.chunks_exact(4);
            assert!(coefficients.remainder().is_empty());
            let precincts: Vec<_> = (0..17)
                .map(|_| {
                    let packets = std::array::from_fn(|index| {
                        band_lengths(3032, index)
                            .unwrap()
                            .into_iter()
                            .map(|len| {
                                (0..len)
                                    .map(|_| {
                                        i32::from_le_bytes(
                                            coefficients.next().unwrap().try_into().unwrap(),
                                        )
                                    })
                                    .collect()
                            })
                            .collect()
                    });
                    reconstruct_precinct(3032, &packets).unwrap()
                })
                .collect();
            assert!(coefficients.next().is_none());
            let oracle = std::fs::read(fixture_dir.join(format!("{stem}.bayer.u16le"))).unwrap();
            let bayer = bayer_prefix(&precincts, 64).unwrap();
            assert_eq!(bayer.len() * 2, oracle.len());
            for (index, (actual, expected)) in bayer.iter().zip(oracle.chunks_exact(2)).enumerate()
            {
                assert_eq!(actual.to_le_bytes(), expected, "{stem} sample {index}");
            }
            println!("{stem}: {} identical Bayer samples", bayer.len());
        }
    }

    #[test]
    #[ignore = "requires real Z6III NEF on Desktop"]
    fn bench_and_verify_dsc5673_bit_exact() {
        use std::time::Instant;

        let path = std::path::PathBuf::from(std::env::var_os("USERPROFILE").unwrap())
            .join("Desktop")
            .join("_DSC5673.NEF");
        let nef = std::fs::read(&path).expect("read _DSC5673.NEF");
        let strip = super::super::find_he_strip(&nef)
            .expect("find_he_strip")
            .expect("strip present");
        let (rest, weights) = super::super::verified_profile(&strip).expect("verified_profile");

        // Warmup LUT and rayon pool once
        let (orig_width, orig_bayer) = super::super::decode_frame(&nef)
            .expect("orig decode_frame")
            .expect("orig frame");
        let fast_bayer = decode_he_bayer_parallel(rest, weights, strip.width, strip.height, false)
            .expect("parallel decode");

        assert_eq!(orig_width, strip.width);
        assert_eq!(orig_bayer.len(), 24_498_560);
        assert_eq!(fast_bayer.len(), orig_bayer.len());
        let mut diffs = 0usize;
        for (i, (&a, &b)) in orig_bayer.iter().zip(fast_bayer.iter()).enumerate() {
            if a != b {
                if diffs < 5 {
                    eprintln!("diff at {i}: orig={a} fast={b}");
                }
                diffs += 1;
            }
        }
        assert_eq!(diffs, 0, "expected 0 bit differences across 24,498,560 samples");

        let mut orig_ms = Vec::new();
        let mut fast_ms = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let r = super::super::decode_frame(&nef).unwrap().unwrap();
            orig_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(r.1[0]);

            let t1 = Instant::now();
            let f = decode_he_bayer_parallel(rest, weights, strip.width, strip.height, false).unwrap();
            fast_ms.push(t1.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(f[0]);
        }
        let orig_best = orig_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let orig_avg = orig_ms.iter().sum::<f64>() / orig_ms.len() as f64;
        let fast_best = fast_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let fast_avg = fast_ms.iter().sum::<f64>() / fast_ms.len() as f64;
        println!(
            "BIT_EXACT_OK samples={} diffs=0 | orig: runs={:?} best={:.2}ms avg={:.2}ms | fast: runs={:?} best={:.2}ms avg={:.2}ms | speedup={:.2}x",
            fast_bayer.len(),
            orig_ms,
            orig_best,
            orig_avg,
            fast_ms,
            fast_best,
            fast_avg,
            orig_best / fast_best
        );
    }
}
