#![forbid(unsafe_code)]

#[path = "rb.rs"]
mod rb;
mod finish;
mod ggblur;

#[derive(Clone, Copy)]
pub enum Finish {
    Diagnostic { noise: f32, control: i32 },
    GgBlurComp { noise: f32 },
}

#[cfg(test)]
#[path = "finish_tests.rs"]
mod finish_tests;

/// Width and height refer to the output core. Input starts six pixels above
/// and left of the core, with stride measured in interleaved RGB pixels.
/// Core-local even/even is S0-native, odd/odd is S2-native.
pub fn planar_core(
    input: &[[f32; 3]],
    stride: usize,
    width: usize,
    height: usize,
    scale: [f32; 3],
    offset: f32,
) -> Option<Vec<[f32; 3]>> {
    planar(input, stride, width, height, scale, offset, None)
}

pub fn planar_with_finish(
    input: &[[f32; 3]],
    stride: usize,
    width: usize,
    height: usize,
    scale: [f32; 3],
    offset: f32,
    noise: f32,
    control: i32,
) -> Option<Vec<[f32; 3]>> {
    planar(input, stride, width, height, scale, offset, Some(Finish::Diagnostic { noise, control }))
}

pub fn planar(
    input: &[[f32; 3]],
    stride: usize,
    width: usize,
    height: usize,
    scale: [f32; 3],
    offset: f32,
    finish: Option<Finish>,
) -> Option<Vec<[f32; 3]>> {
    if width == 0 || height == 0 {
        return None;
    }
    let padded_width = width.checked_add(12)?;
    let padded_height = height.checked_add(12)?;
    let required = (padded_height - 1)
        .checked_mul(stride)?
        .checked_add(padded_width)?;
    if stride < padded_width || input.len() < required {
        return None;
    }

    let [mut first, mut green, mut second] =
        std::array::from_fn(|_| Vec::with_capacity(padded_width * padded_height));
    for row in 0..padded_height {
        for pixel in &input[row * stride..row * stride + padded_width] {
            first.push(pixel[0] * scale[0]);
            green.push(pixel[1] * scale[1]);
            second.push(pixel[2] * scale[2]);
        }
    }

    // shortcut: scalar centers inside each view, use native vector tails when
    // reproducing scratch bytes or hardware SSE rounding.
    for row in 2..padded_height - 2 {
        let parity = row & 1;
        let (color, guide) = if parity == 0 {
            (&mut second, &first)
        } else {
            (&mut first, &second)
        };
        for col in (2 + parity..padded_width - 2).step_by(2) {
            let value =
                rb::diagonal_position(color, &green, guide, padded_width, (row, col), offset, 1.0)
                    .0;
            color[row * padded_width + col] = value;
        }
    }

    for row in 4..padded_height - 4 {
        for col in (4 + ((row + 1) & 1)..padded_width - 4).step_by(2) {
            let ([s0, s2], _) =
                rb::green_position(&first, &green, &second, padded_width, row, col, offset);
            let index = row * padded_width + col;
            first[index] = s0;
            second[index] = s2;
        }
    }

    Some(finish_planar([first, green, second], padded_width, width, height, scale, offset, finish))
}

pub(crate) fn finish_planar(
    [mut first, mut green, mut second]: [Vec<f32>; 3],
    padded_width: usize,
    width: usize,
    height: usize,
    scale: [f32; 3],
    offset: f32,
    finish: Option<Finish>,
) -> Vec<[f32; 3]> {
    if let Some(Finish::Diagnostic { noise, control }) = finish {
        finish::apply([&mut first, &mut green, &mut second], padded_width, width, height, scale, offset, noise, control);
    }
    if let Some(Finish::GgBlurComp { noise }) = finish {
        return ggblur::core([&first, &green, &second], padded_width, width, height, scale, offset, noise);
    }
    let mut output = Vec::with_capacity(width * height);
    for row in 6..6 + height {
        for col in 6..6 + width {
            let index = row * padded_width + col;
            output.push([first[index], green[index], second[index]]);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_matches_even_origin_subtiles_and_uses_diagonal_results() {
        let stride = 36;
        let input: Vec<_> = (0..stride * 24)
            .map(|i| {
                [
                    25.0 + (i % 53) as f32,
                    90.0 + (i % 37) as f32,
                    55.0 + (i % 79) as f32,
                ]
            })
            .collect();
        let scale = [1.25, 0.75, 1.5];
        let offset = 4.0;
        let output = planar_core(&input, stride, 19, 11, scale, offset).unwrap();
        for (x, y, w, h) in [(0, 0, 8, 6), (8, 0, 11, 6), (0, 6, 8, 5), (8, 6, 11, 5)] {
            let tile = planar_core(&input[y * stride + x..], stride, w, h, scale, offset).unwrap();
            for row in 0..h {
                let start = (y + row) * 19 + x;
                assert_eq!(&tile[row * w..(row + 1) * w], &output[start..start + w]);
            }
        }

        let [first, green, second] = std::array::from_fn(|channel| {
            input
                .iter()
                .map(|pixel| pixel[channel] * scale[channel])
                .collect::<Vec<_>>()
        });
        let index = 6 * stride + 6;
        assert_eq!(output[0][0], first[index]);
        assert_eq!(output[0][1], green[index]);
        assert_eq!(
            output[0][2],
            rb::diagonal_position(&second, &green, &first, stride, (6, 6), offset, 1.0,).0
        );

        let mut corrected = second.clone();
        for (row, col) in [(6, 6), (6, 8)] {
            corrected[row * stride + col] =
                rb::diagonal_position(&second, &green, &first, stride, (row, col), offset, 1.0).0;
        }
        let expected = rb::green_position(&first, &green, &corrected, stride, 6, 7, offset).0[1];
        let stale = rb::green_position(&first, &green, &second, stride, 6, 7, offset).0[1];
        assert_ne!(expected, stale);
        assert_eq!(output[1][2], expected);
        assert!(planar_core(&input[..10], stride, 19, 11, scale, offset).is_none());
        assert!(planar_core(&input, 30, 19, 11, scale, offset).is_none());
    }
}
