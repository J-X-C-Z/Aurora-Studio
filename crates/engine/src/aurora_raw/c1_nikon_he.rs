//! Nikon HE VCGH Bayer reconstruction from signed JPEG XS coefficient bands.
//! This stage accepts the coefficients, not NEF container bytes or metadata.

use std::io::{Error, ErrorKind, Result};

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

fn lift53(low: &[i64], high: &[i32]) -> Vec<i64> {
    let mut line = vec![0; low.len() + high.len()];
    for (index, &sample) in low.iter().enumerate() {
        let left = i64::from(high[index.saturating_sub(1)]);
        let right = i64::from(high[index.min(high.len() - 1)]);
        line[2 * index] = sample - ((left + right + 2) >> 2);
    }
    for (index, &detail) in high.iter().enumerate() {
        let left = 2 * index;
        line[left + 1] =
            i64::from(detail) + ((line[left] + line[(left + 2).min(2 * (low.len() - 1))]) >> 1);
    }
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

fn inverse_curve(value: i64) -> u16 {
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

fn green_pair(
    above: &[Vec<i64>; 4],
    current: &[Vec<i64>; 4],
    below: &[Vec<i64>; 4],
    bottom: bool,
    width: usize,
) -> (Vec<i64>, Vec<i64>) {
    let [yl, cr, yh, cb] = current;
    let [_, _, above_yh, above_cb] = above;
    let [below_yl, below_cr, below_yh, _] = below;
    let mut green0 = Vec::with_capacity(width);
    let mut green1 = Vec::with_capacity(width);
    let mut y0_left = 0;
    let mut y0_below_left = 0;
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
        green0.push(y0 - ((cb[col] + above_cb[col] + cr[col] + cr[right]) >> 3));
        green1.push(
            ((y0_below + y0 + y0_below_left + y0_left) >> 2) + yh[col]
                - ((cb[left] + cb[col] + cr[col] + below_cr[col]) >> 3),
        );
        y0_left = y0;
        y0_below_left = y0_below;
    }
    (green0, green1)
}

/// Row/column context for one cell-row block. `row` is the original closure,
/// unchanged; only the block start state is recomputed from the same formulas.
struct BayerContext<'a> {
    precincts: &'a [Precinct],
    width: usize,
    cell_rows: usize,
    has_bottom: bool,
    output_rows: usize,
}

impl BayerContext<'_> {
    fn row(&self, cell_row: usize) -> [Vec<i64>; 4] {
        let width = self.width;
        let cell_row = if self.has_bottom {
            cell_row.min(self.cell_rows - 1)
        } else {
            cell_row
        };
        let index = cell_row / 2;
        let current = &self.precincts[index];
        let previous = &self.precincts[index.saturating_sub(1)];
        std::array::from_fn(|component| {
            if component == 2 {
                return if cell_row % 2 == 0 {
                    &current.low[2]
                } else {
                    &current.high[2]
                }
                .clone();
            }
            (0..width)
                .map(|col| {
                    let detail = current.high[component][col];
                    let even =
                        (4 * current.low[component][col] - previous.high[component][col] - detail
                            + 1)
                            >> 2;
                    if cell_row % 2 == 0 {
                        even
                    } else if index + 1 == self.precincts.len() {
                        detail + even
                    } else {
                        let next = &self.precincts[index + 1];
                        let next_even =
                            (4 * next.low[component][col] - detail - next.high[component][col] + 1)
                                >> 2;
                        detail + ((even + next_even) >> 1)
                    }
                })
                .collect()
        })
    }

    /// `green` for cell row `index`. Index 0 is the top-edge call the sequential
    /// loop uses; every later index uses the row above, matching its iteration
    /// `index - 1`.
    fn greens(&self, index: usize) -> (Vec<i64>, Vec<i64>) {
        let current = self.row(index);
        let below = self.row(index + 1);
        if index == 0 {
            return green_pair(&current, &current, &below, false, self.width);
        }
        let above = self.row(index - 1);
        green_pair(
            &above,
            &current,
            &below,
            self.has_bottom && index + 1 >= self.cell_rows,
            self.width,
        )
    }

    fn emit_block(&self, first: usize, last: usize, output: &mut [u16]) -> usize {
        let width = self.width;
        let cell_rows = self.cell_rows;
        let output_rows = self.output_rows;
        let mut at = 0usize;
        let mut current = self.row(first);
        let mut below = self.row(first + 1);
        let (mut green0, mut green1, mut above_green1);
        if first == 0 {
            let (first0, first1) = green_pair(&current, &current, &below, false, width);
            green0 = first0;
            green1 = first1;
            above_green1 = green1.clone();
        } else {
            let (block0, block1) = self.greens(first);
            green0 = block0;
            green1 = block1;
            above_green1 = self.greens(first - 1).1;
        }
        for index in first..last {
            let next_context = (index + 1 < cell_rows).then(|| self.row(index + 2));
            let (next_green0, next_green1) = green_pair(
                &current,
                &below,
                next_context.as_ref().unwrap_or(&below),
                self.has_bottom && index + 2 >= cell_rows,
                width,
            );
            let blue_green0 = if self.has_bottom && index + 1 == cell_rows {
                &green0
            } else {
                &next_green0
            };
            for col in 0..width {
                let left = col.saturating_sub(1);
                let red = ((green1[col] + green0[col] + green0[left] + above_green1[col]) >> 2)
                    + current[1][col];
                output[at] = inverse_curve(red + 32768);
                output[at + 1] = inverse_curve(green0[col] + 32768);
                at += 2;
            }
            if 2 * index + 1 < output_rows {
                for col in 0..width {
                    let right = (col + 1).min(width - 1);
                    let blue = ((green1[col] + green0[col] + green1[right] + blue_green0[col])
                        >> 2)
                        + current[3][col];
                    output[at] = inverse_curve(green1[col] + 32768);
                    output[at + 1] = inverse_curve(blue + 32768);
                    at += 2;
                }
            }
            if let Some(next_context) = next_context {
                current = below;
                below = next_context;
                green0 = next_green0;
                above_green1 = std::mem::replace(&mut green1, next_green1);
            }
        }
        at
    }
}

fn bayer_workers(cell_rows: usize) -> usize {
    let requested = std::env::var("AURORA_HE_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value >= 1)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .unwrap_or(1)
                .min(8)
        });
    requested.clamp(1, cell_rows.max(1))
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
    let cell_rows = output_rows.div_ceil(2);
    let sample_count = (2 * width)
        .checked_mul(output_rows)
        .ok_or_else(|| invalid("HE dimensions overflow"))?;
    let workers = bayer_workers(cell_rows);
    if workers <= 1 || cell_rows < 32 {
        return assemble_bayer_sequential(precincts, output_rows, has_bottom);
    }
    let context = BayerContext {
        precincts,
        width,
        cell_rows,
        has_bottom,
        output_rows,
    };
    let mut output = Vec::new();
    output
        .try_reserve_exact(sample_count)
        .map_err(Error::other)?;
    output.resize(sample_count, 0);
    let block = cell_rows.div_ceil(workers);
    std::thread::scope(|scope| {
        for (block_index, chunk) in output.chunks_mut(block * 4 * width).enumerate() {
            let first = block_index * block;
            let last = (first + block).min(cell_rows);
            let context = &context;
            scope.spawn(move || {
                let written = context.emit_block(first, last, chunk);
                debug_assert!(written <= chunk.len());
            });
        }
    });
    Ok(output)
}

fn assemble_bayer_sequential(
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
    let cell_rows = output_rows.div_ceil(2);
    let sample_count = (2 * width)
        .checked_mul(output_rows)
        .ok_or_else(|| invalid("HE dimensions overflow"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(sample_count)
        .map_err(Error::other)?;

    let row = |cell_row: usize| -> [Vec<i64>; 4] {
        let cell_row = if has_bottom {
            cell_row.min(cell_rows - 1)
        } else {
            cell_row
        };
        let index = cell_row / 2;
        let current = &precincts[index];
        let previous = &precincts[index.saturating_sub(1)];
        std::array::from_fn(|component| {
            if component == 2 {
                return if cell_row % 2 == 0 {
                    &current.low[2]
                } else {
                    &current.high[2]
                }
                .clone();
            }
            (0..width)
                .map(|col| {
                    let detail = current.high[component][col];
                    let even =
                        (4 * current.low[component][col] - previous.high[component][col] - detail
                            + 1)
                            >> 2;
                    if cell_row % 2 == 0 {
                        even
                    } else if index + 1 == precincts.len() {
                        detail + even
                    } else {
                        let next = &precincts[index + 1];
                        let next_even =
                            (4 * next.low[component][col] - detail - next.high[component][col] + 1)
                                >> 2;
                        detail + ((even + next_even) >> 1)
                    }
                })
                .collect()
        })
    };
    let green =
        |above: &[Vec<i64>; 4], current: &[Vec<i64>; 4], below: &[Vec<i64>; 4], bottom: bool| {
            let [yl, cr, yh, cb] = current;
            let [_, _, above_yh, above_cb] = above;
            let [below_yl, below_cr, below_yh, _] = below;
            let mut green0 = Vec::with_capacity(width);
            let mut green1 = Vec::with_capacity(width);
            let mut y0_left = 0;
            let mut y0_below_left = 0;
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
                green0.push(y0 - ((cb[col] + above_cb[col] + cr[col] + cr[right]) >> 3));
                green1.push(
                    ((y0_below + y0 + y0_below_left + y0_left) >> 2) + yh[col]
                        - ((cb[left] + cb[col] + cr[col] + below_cr[col]) >> 3),
                );
                y0_left = y0;
                y0_below_left = y0_below;
            }
            (green0, green1)
        };

    let mut current = row(0);
    let mut below = row(1);
    let (mut green0, mut green1) = green(&current, &current, &below, false);
    let mut above_green1 = green1.clone();
    for index in 0..cell_rows {
        let next_context = (index + 1 < cell_rows).then(|| row(index + 2));
        let (next_green0, next_green1) = green(
            &current,
            &below,
            next_context.as_ref().unwrap_or(&below),
            has_bottom && index + 2 >= cell_rows,
        );
        let blue_green0 = if has_bottom && index + 1 == cell_rows {
            &green0
        } else {
            &next_green0
        };
        for col in 0..width {
            let left = col.saturating_sub(1);
            let red = ((green1[col] + green0[col] + green0[left] + above_green1[col]) >> 2)
                + current[1][col];
            output.extend([
                inverse_curve(red + 32768),
                inverse_curve(green0[col] + 32768),
            ]);
        }
        if 2 * index + 1 < output_rows {
            for col in 0..width {
                let right = (col + 1).min(width - 1);
                let blue = ((green1[col] + green0[col] + green1[right] + blue_green0[col]) >> 2)
                    + current[3][col];
                output.extend([
                    inverse_curve(green1[col] + 32768),
                    inverse_curve(blue + 32768),
                ]);
            }
        }
        if let Some(next_context) = next_context {
            current = below;
            below = next_context;
            green0 = next_green0;
            above_green1 = std::mem::replace(&mut green1, next_green1);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
