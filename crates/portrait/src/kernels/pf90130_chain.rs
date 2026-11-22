//! Candidate PF90130 postprocess chain.
//!
//! Static order: 8-bit masked blend, three affine warps, then the float
//! knee/difference/add/strength stage. The three warp inputs used here are an
//! explicit candidate mapping (blend RGB, base RGB, score), not a same-job
//! observation of the vendor producer. No input is synthesized when missing.

use crate::Refusal;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl PixelRect {
    pub fn is_full(self, width: u32, height: u32) -> bool {
        self.x == 0 && self.y == 0 && self.width == width && self.height == height
    }

    fn inside(self, width: u32, height: u32) -> bool {
        self.width > 0
            && self.height > 0
            && self.x <= width
            && self.y <= height
            && self.width <= width - self.x
            && self.height <= height - self.y
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Inputs<'a> {
    pub ticket: Option<u64>,
    pub width: u32,
    pub height: u32,
    pub source_width: Option<u32>,
    pub source_height: Option<u32>,
    pub source_roi: Option<PixelRect>,
    pub roi: Option<PixelRect>,
    pub uv: Option<&'a [[f32; 2]]>,
    /// `job+0x490`: source-to-destination geometry. `warpAffine(flags=1)`
    /// inverts this once, so the sampler uses its inverse.
    pub src_to_dst: Option<[[f32; 3]; 2]>,
    pub source_rgb8: Option<&'a [u8]>,
    pub base_rgb8: Option<&'a [u8]>,
    pub add_rgb_f32: Option<&'a [f32]>,
    pub material_nhwc4: Option<&'a [u8]>,
    pub mask_nhwc4: Option<&'a [u8]>,
    pub strength: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub ticket: u64,
    pub blend_rgb8: Vec<u8>,
    pub score_unit_f32: Vec<f32>,
    pub warped_blend_rgb8: Vec<u8>,
    pub warped_base_rgb8: Vec<u8>,
    pub warped_score_f32: Vec<f32>,
    pub post_first_f32: Vec<f32>,
    pub post_second_f32: Vec<f32>,
    pub warp_calls: u8,
    pub post_gate_open: bool,
}

fn round_ties_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction > 0.5 {
        floor + 1.0
    } else if fraction < 0.5 {
        floor
    } else if (floor as i64) & 1 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

fn dequant_signed(value: f32) -> u8 {
    round_ties_even(((value as f64) + 1.0).clamp(0.0, 2.0) * 127.5)
        .clamp(0.0, 255.0) as u8
}

fn knee(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    if value < 0.2 {
        0.0
    } else if value > 0.3 {
        1.0
    } else {
        value
    }
}

fn mask_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) as f64 * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8
}

fn blend_byte(q: u8, effect: u8, source: u8) -> u8 {
    let n = q as i32 * effect as i32 + (255 - q as i32) * source as i32 + 1;
    ((n + (n >> 8)) >> 8) as u8
}

fn finite_matrix(matrix: [[f32; 3]; 2]) -> bool {
    matrix.into_iter().flatten().all(f32::is_finite)
}

fn invert_affine(matrix: [[f32; 3]; 2]) -> Result<[[f32; 3]; 2], Refusal> {
    let m00 = matrix[0][0] as f64;
    let m01 = matrix[0][1] as f64;
    let m02 = matrix[0][2] as f64;
    let m10 = matrix[1][0] as f64;
    let m11 = matrix[1][1] as f64;
    let m12 = matrix[1][2] as f64;
    let det = m00 * m11 - m01 * m10;
    if !det.is_finite() || det.abs() <= 1.0e-12 {
        return Err(Refusal::ValueOutOfRange("PF90130_AFFINE"));
    }
    let inverse = [
        [
            (m11 / det) as f32,
            (-m01 / det) as f32,
            ((m01 * m12 - m11 * m02) / det) as f32,
        ],
        [
            (-m10 / det) as f32,
            (m00 / det) as f32,
            ((m10 * m02 - m00 * m12) / det) as f32,
        ],
    ];
    if !finite_matrix(inverse) {
        return Err(Refusal::ValueOutOfRange("PF90130_AFFINE"));
    }
    Ok(inverse)
}

fn bilinear(plane: &[f32], width: u32, height: u32, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let at = |px: i64, py: i64| -> f32 {
        if px < 0 || py < 0 || px >= width as i64 || py >= height as i64 {
            0.0
        } else {
            plane[py as usize * width as usize + px as usize]
        }
    };
    let top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * tx;
    let bottom = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * tx;
    top + (bottom - top) * ty
}

fn warp_rgb8(
    input: &[u8],
    width: u32,
    height: u32,
    sample: [[f32; 3]; 2],
) -> Vec<u8> {
    let pixels = width as usize * height as usize;
    let mut planes = [
        Vec::with_capacity(pixels),
        Vec::with_capacity(pixels),
        Vec::with_capacity(pixels),
    ];
    for pixel in input.chunks_exact(3) {
        for channel in 0..3 {
            planes[channel].push(pixel[channel] as f32);
        }
    }
    let mut output = vec![0u8; pixels * 3];
    for y in 0..height {
        for x in 0..width {
            let sx = sample[0][0] * x as f32 + sample[0][1] * y as f32 + sample[0][2];
            let sy = sample[1][0] * x as f32 + sample[1][1] * y as f32 + sample[1][2];
            let pixel = y as usize * width as usize + x as usize;
            for channel in 0..3 {
                output[pixel * 3 + channel] =
                    round_ties_even(bilinear(&planes[channel], width, height, sx, sy) as f64)
                        .clamp(0.0, 255.0) as u8;
            }
        }
    }
    output
}

fn warp_f32(
    input: &[f32],
    width: u32,
    height: u32,
    sample: [[f32; 3]; 2],
) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for y in 0..height {
        for x in 0..width {
            let sx = sample[0][0] * x as f32 + sample[0][1] * y as f32 + sample[0][2];
            let sy = sample[1][0] * x as f32 + sample[1][1] * y as f32 + sample[1][2];
            output[y as usize * width as usize + x as usize] =
                bilinear(input, width, height, sx, sy);
        }
    }
    output
}

fn decode_ab(
    a: &[u8],
    b: &[u8],
    pixels: usize,
) -> Result<(Vec<[f32; 3]>, Vec<f32>), Refusal> {
    if a.len() != pixels * 16 || b.len() != pixels * 16 {
        return Err(Refusal::ShapeMismatch("PF90130_NHWC4"));
    }
    let mut material = Vec::with_capacity(pixels);
    let mut mask = Vec::with_capacity(pixels);
    for (a_pixel, b_pixel) in a.chunks_exact(16).zip(b.chunks_exact(16)) {
        let mut values = [0.0f32; 4];
        for lane in 0..3 {
            values[lane] = f32::from_le_bytes(
                a_pixel[lane * 4..lane * 4 + 4]
                    .try_into()
                    .expect("four-byte lane"),
            );
        }
        let mask_value = f32::from_le_bytes(
            b_pixel[0..4]
                .try_into()
                .expect("four-byte mask lane"),
        );
        if values[..3].iter().chain(std::iter::once(&mask_value)).any(|v| !v.is_finite()) {
            return Err(Refusal::ValueOutOfRange("PF90130_NHWC4"));
        }
        material.push([values[0], values[1], values[2]]);
        mask.push(mask_value);
    }
    Ok((material, mask))
}

/// Inputs after the fail-closed checks. The raw A/B bytes stay available so
/// the CUDA adapter can upload the same NHWC4 payload used by the CPU oracle.
pub struct Prepared<'a> {
    pub ticket: u64,
    pub width: u32,
    pub height: u32,
    pub pixels: usize,
    pub sample: [[f32; 3]; 2],
    pub source: &'a [u8],
    pub base: &'a [u8],
    pub add: &'a [f32],
    pub material_nhwc4: &'a [u8],
    pub mask_nhwc4: &'a [u8],
    pub material: Vec<[f32; 3]>,
    pub mask: Vec<f32>,
    pub strength: f32,
}

pub fn prepare<'a>(input: &Inputs<'a>) -> Result<Prepared<'a>, Refusal> {
    let ticket = input
        .ticket
        .filter(|ticket| *ticket != 0)
        .ok_or(Refusal::MissingInput("PF90130_SAME_JOB_RECEIPT"))?;
    if input.width == 0 || input.height == 0 {
        return Err(Refusal::ValueOutOfRange("PF90130_FRAME"));
    }
    let pixels = (input.width as usize)
        .checked_mul(input.height as usize)
        .ok_or(Refusal::ShapeMismatch("PF90130_FRAME"))?;
    let roi = input.roi.ok_or(Refusal::MissingInput("PF90130_ROI"))?;
    if !roi.is_full(input.width, input.height) {
        return Err(Refusal::ValueOutOfRange("PF90130_ROI"));
    }
    let source_width = input
        .source_width
        .filter(|value| *value > 0)
        .ok_or(Refusal::MissingInput("PF90130_SOURCE_GEOMETRY"))?;
    let source_height = input
        .source_height
        .filter(|value| *value > 0)
        .ok_or(Refusal::MissingInput("PF90130_SOURCE_GEOMETRY"))?;
    let source_roi = input
        .source_roi
        .ok_or(Refusal::MissingInput("PF90130_SOURCE_ROI"))?;
    if !source_roi.inside(source_width, source_height) {
        return Err(Refusal::ValueOutOfRange("PF90130_SOURCE_ROI"));
    }
    let uv = input.uv.ok_or(Refusal::MissingInput("PF90130_UV"))?;
    if uv.is_empty() {
        return Err(Refusal::MissingInput("PF90130_UV"));
    }
    if uv.len() != pixels {
        return Err(Refusal::ShapeMismatch("PF90130_UV"));
    }
    if uv
        .iter()
        .flatten()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(Refusal::ValueOutOfRange("PF90130_UV"));
    }
    let matrix = input
        .src_to_dst
        .filter(|matrix| finite_matrix(*matrix))
        .ok_or(Refusal::MissingInput("PF90130_AFFINE"))?;
    let sample = invert_affine(matrix)?;
    let source = input
        .source_rgb8
        .ok_or(Refusal::MissingInput("PF90130_SOURCE_RGB8"))?;
    let base = input
        .base_rgb8
        .ok_or(Refusal::MissingInput("PF90130_BASE_RGB8"))?;
    let add = input
        .add_rgb_f32
        .ok_or(Refusal::MissingInput("PF90130_ADD_RGB_F32"))?;
    if source.len() != pixels * 3 || base.len() != pixels * 3 || add.len() != pixels * 3 {
        return Err(Refusal::ShapeMismatch("PF90130_ALIGNED_PLANES"));
    }
    if add.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("PF90130_ADD_RGB_F32"));
    }
    let material = input
        .material_nhwc4
        .ok_or(Refusal::MissingInput("PF90130_MATERIAL"))?;
    let mask = input
        .mask_nhwc4
        .ok_or(Refusal::MissingInput("PF90130_LIVE_MASK"))?;
    let (material, mask) = decode_ab(material, mask, pixels)?;
    let strength = input
        .strength
        .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
        .ok_or(Refusal::MissingInput("PF90130_STRENGTH"))?;

    Ok(Prepared {
        ticket,
        width: input.width,
        height: input.height,
        pixels,
        sample,
        source,
        base,
        add,
        material_nhwc4: input.material_nhwc4.expect("validated material"),
        mask_nhwc4: input.mask_nhwc4.expect("validated mask"),
        material,
        mask,
        strength,
    })
}

pub fn run_prepared(prepared: &Prepared<'_>) -> Output {
    let pixels = prepared.pixels;
    let mut blend = vec![0u8; pixels * 3];
    let mut score = Vec::with_capacity(pixels);
    for pixel in 0..pixels {
        let mask8 = dequant_signed(prepared.mask[pixel]);
        let unit = mask8 as f32 / 255.0;
        let q = mask_byte(knee(unit));
        score.push(unit);
        for channel in 0..3 {
            let effect = dequant_signed(prepared.material[pixel][channel]);
            blend[pixel * 3 + channel] =
                blend_byte(q, effect, prepared.source[pixel * 3 + channel]);
        }
    }

    let warped_blend = warp_rgb8(&blend, prepared.width, prepared.height, prepared.sample);
    let warped_base = warp_rgb8(prepared.base, prepared.width, prepared.height, prepared.sample);
    let warped_score = warp_f32(&score, prepared.width, prepared.height, prepared.sample);

    let post_gate_open = prepared.strength > 0.001;
    let mut post_first = Vec::new();
    let mut post_second = Vec::new();
    if post_gate_open {
        post_first.reserve(pixels * 3);
        post_second.reserve(pixels * 3);
        for pixel in 0..pixels {
            let weight = knee(warped_score[pixel]);
            for channel in 0..3 {
                let index = pixel * 3 + channel;
                let delta = warped_blend[index] as f32 - warped_base[index] as f32;
                let first = delta * weight + prepared.add[index];
                let second = first * prepared.strength;
                post_first.push(first);
                post_second.push(second);
            }
        }
    }

    Output {
        ticket: prepared.ticket,
        blend_rgb8: blend,
        score_unit_f32: score,
        warped_blend_rgb8: warped_blend,
        warped_base_rgb8: warped_base,
        warped_score_f32: warped_score,
        post_first_f32: post_first,
        post_second_f32: post_second,
        warp_calls: 3,
        post_gate_open,
    }
}

pub fn run(input: &Inputs<'_>) -> Result<Output, Refusal> {
    prepare(input).map(|prepared| run_prepared(&prepared))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nhwc4(values: &[[f32; 4]]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|pixel| pixel.iter().flat_map(|value| value.to_le_bytes()))
            .collect()
    }

    fn base_inputs<'a>(
        width: u32,
        height: u32,
        uv: &'a [[f32; 2]],
        source: &'a [u8],
        base: &'a [u8],
        add: &'a [f32],
        material: &'a [u8],
        mask: &'a [u8],
    ) -> Inputs<'a> {
        Inputs {
            ticket: Some(966),
            width,
            height,
            source_width: Some(768),
            source_height: Some(384),
            source_roi: Some(PixelRect {
                x: 0,
                y: 64,
                width: 767,
                height: 320,
            }),
            roi: Some(PixelRect {
                x: 0,
                y: 0,
                width,
                height,
            }),
            uv: Some(uv),
            src_to_dst: Some([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            source_rgb8: Some(source),
            base_rgb8: Some(base),
            add_rgb_f32: Some(add),
            material_nhwc4: Some(material),
            mask_nhwc4: Some(mask),
            strength: Some(0.5),
        }
    }

    #[test]
    fn lanes_three_warps_and_float_post_are_ordered() {
        let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let source = [10u8, 20, 30, 11, 21, 31, 12, 22, 32, 13, 23, 33];
        let base = [20u8, 40, 60, 20, 40, 60, 20, 40, 60, 20, 40, 60];
        let add = [1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let material = nhwc4(&[
            [0.0, 0.0, 0.0, 99.0],
            [0.0, 0.0, 0.0, 99.0],
            [0.0, 0.0, 0.0, 99.0],
            [0.0, 0.0, 0.0, 99.0],
        ]);
        let mask = nhwc4(&[
            [-1.0, 9.0, 9.0, 9.0],
            [1.0, 0.0, 0.0, 0.0],
            [-0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]);
        let output = run(&base_inputs(
            2, 2, &uv, &source, &base, &add, &material, &mask,
        ))
        .unwrap();

        assert_eq!(output.warp_calls, 3);
        assert_eq!(&output.blend_rgb8[0..3], &[10, 20, 30]);
        assert_eq!(&output.blend_rgb8[3..6], &[128, 128, 128]);
        assert_eq!(&output.blend_rgb8[6..9], &[41, 48, 56]);
        assert_eq!(&output.blend_rgb8[9..12], &[128, 128, 128]);
        assert_eq!(output.warped_blend_rgb8, output.blend_rgb8);
        assert_eq!(output.warped_base_rgb8, base);
        assert_eq!(output.warped_score_f32, output.score_unit_f32);
        assert_eq!(&output.post_first_f32[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&output.post_second_f32[0..3], &[0.5, 1.0, 1.5]);
        assert_eq!(&output.post_first_f32[3..6], &[109.0, 90.0, 71.0]);
        assert_eq!(&output.post_second_f32[3..6], &[54.5, 45.0, 35.5]);
        let w = 64.0f32 / 255.0;
        assert_eq!(output.post_first_f32[6], 21.0 * w + 1.0);
        assert_eq!(output.post_first_f32[7], 8.0 * w + 2.0);
        assert_eq!(output.post_first_f32[8], -4.0 * w + 3.0);
    }

    #[test]
    fn affine_is_inverted_once_and_three_sources_follow_it() {
        let matrix = [[1.0f32, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let sample = invert_affine(matrix).unwrap();
        assert_eq!(sample, [[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]);
        let blend = warp_rgb8(
            &[10, 20, 30, 40, 50, 60],
            2,
            1,
            sample,
        );
        assert_eq!(blend, vec![0, 0, 0, 10, 20, 30]);
        let score = warp_f32(&[0.25, 0.75], 2, 1, sample);
        assert_eq!(score, vec![0.0, 0.25]);
    }

    #[test]
    fn missing_same_job_fields_refuse_without_defaults() {
        let uv = [[0.0f32, 0.0]];
        let source = [10u8, 20, 30];
        let base = [20u8, 40, 60];
        let add = [1.0f32, 2.0, 3.0];
        let material = nhwc4(&[[0.0, 0.0, 0.0, 0.0]]);
        let mask = nhwc4(&[[-1.0, 0.0, 0.0, 0.0]]);
        let complete = base_inputs(1, 1, &uv, &source, &base, &add, &material, &mask);
        assert!(run(&complete).is_ok());

        let cases = [
            (Inputs { ticket: None, ..complete }, "PF90130_SAME_JOB_RECEIPT"),
            (Inputs { ticket: Some(0), ..complete }, "PF90130_SAME_JOB_RECEIPT"),
            (Inputs { roi: None, ..complete }, "PF90130_ROI"),
            (Inputs { uv: None, ..complete }, "PF90130_UV"),
            (Inputs { source_roi: None, ..complete }, "PF90130_SOURCE_ROI"),
            (Inputs { src_to_dst: None, ..complete }, "PF90130_AFFINE"),
            (Inputs { source_rgb8: None, ..complete }, "PF90130_SOURCE_RGB8"),
            (Inputs { base_rgb8: None, ..complete }, "PF90130_BASE_RGB8"),
            (Inputs { add_rgb_f32: None, ..complete }, "PF90130_ADD_RGB_F32"),
            (Inputs { material_nhwc4: None, ..complete }, "PF90130_MATERIAL"),
            (Inputs { mask_nhwc4: None, ..complete }, "PF90130_LIVE_MASK"),
            (Inputs { strength: None, ..complete }, "PF90130_STRENGTH"),
        ];
        for (input, name) in cases {
            assert_eq!(run(&input), Err(Refusal::MissingInput(name)));
        }
    }

    #[test]
    fn post_gate_is_explicit_and_uses_original_f32_order() {
        let uv = [[0.0f32, 0.0]];
        let source = [128u8, 128, 128];
        let base = [20u8, 40, 60];
        let add = [1.0f32, 2.0, 3.0];
        let material = nhwc4(&[[0.0, 0.0, 0.0, 0.0]]);
        let mask = nhwc4(&[[0.0, 0.0, 0.0, 0.0]]);
        let mut input = base_inputs(1, 1, &uv, &source, &base, &add, &material, &mask);
        input.strength = Some(0.001);
        let closed = run(&input).unwrap();
        assert!(!closed.post_gate_open);
        assert!(closed.post_first_f32.is_empty() && closed.post_second_f32.is_empty());

        input.strength = Some(0.0010000002);
        let open = run(&input).unwrap();
        assert!(open.post_gate_open);
        assert_eq!(open.post_first_f32.len(), 3);
        assert_eq!(open.post_second_f32.len(), 3);
    }

    #[test]
    fn write_fixture_when_requested() {
        let Ok(dir) = std::env::var("PF90130_FIXTURE_DIR") else {
            return;
        };
        let uv = [
            [0.0f32, 0.0],
            [0.25, 0.0],
            [0.5, 0.0],
            [0.75, 0.0],
            [0.0, 0.25],
            [0.25, 0.25],
            [0.5, 0.25],
            [0.75, 0.25],
            [0.0, 0.5],
            [0.25, 0.5],
            [0.5, 0.5],
            [0.75, 0.5],
        ];
        let mut source = Vec::new();
        let mut base = Vec::new();
        let mut add = Vec::new();
        for pixel in 0..12u32 {
            for channel in 0..3u32 {
                source.push((pixel * 17 + channel * 31 + 3) as u8);
                base.push((pixel * 11 + channel * 19 + 7) as u8);
                add.push(((pixel * 7 + channel * 5) % 17) as f32 - 8.0);
            }
        }
        let material = nhwc4(
            &(0..12)
                .map(|i| {
                    [
                        ((i * 13) % 41) as f32 / 20.0 - 1.0,
                        ((i * 17) % 43) as f32 / 20.0 - 1.0,
                        ((i * 19) % 47) as f32 / 20.0 - 1.0,
                        99.0,
                    ]
                })
                .collect::<Vec<_>>(),
        );
        let mask = nhwc4(
            &(0..12)
                .map(|i| {
                    [
                        ((i * 23) % 61) as f32 / 30.0 - 1.0,
                        9.0,
                        9.0,
                        9.0,
                    ]
                })
                .collect::<Vec<_>>(),
        );
        let input = Inputs {
            ticket: Some(966),
            width: 4,
            height: 3,
            source_width: Some(512),
            source_height: Some(512),
            source_roi: Some(PixelRect {
                x: 0,
                y: 64,
                width: 511,
                height: 384,
            }),
            roi: Some(PixelRect {
                x: 0,
                y: 0,
                width: 4,
                height: 3,
            }),
            uv: Some(&uv),
            src_to_dst: Some([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]]),
            source_rgb8: Some(&source),
            base_rgb8: Some(&base),
            add_rgb_f32: Some(&add),
            material_nhwc4: Some(&material),
            mask_nhwc4: Some(&mask),
            strength: Some(0.7),
        };
        let output = run(&input).unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        let dir = std::path::PathBuf::from(dir);
        std::fs::write(dir.join("source.rgb8"), &source).unwrap();
        std::fs::write(dir.join("base.rgb8"), &base).unwrap();
        std::fs::write(
            dir.join("add.f32"),
            add.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        )
        .unwrap();
        std::fs::write(dir.join("material.nhwc4"), &material).unwrap();
        std::fs::write(dir.join("mask.nhwc4"), &mask).unwrap();
        std::fs::write(dir.join("blend.rgb8"), &output.blend_rgb8).unwrap();
        std::fs::write(
            dir.join("warped_blend.rgb8"),
            &output.warped_blend_rgb8,
        )
        .unwrap();
        std::fs::write(dir.join("warped_base.rgb8"), &output.warped_base_rgb8).unwrap();
        std::fs::write(
            dir.join("warped_score.f32"),
            output
                .warped_score_f32
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        std::fs::write(
            dir.join("post_first.f32"),
            output
                .post_first_f32
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        std::fs::write(
            dir.join("post_second.f32"),
            output
                .post_second_f32
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        std::fs::write(
            dir.join("fixture.json"),
            format!(
                "{{\"width\":4,\"height\":3,\"strength\":0.7,\"src_to_dst\":[1.0,0.0,0.5,0.0,1.0,0.0],\"warp_calls\":{}}}",
                output.warp_calls
            ),
        )
        .unwrap();
    }
}
