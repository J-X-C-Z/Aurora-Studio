use crate::model_sha256::sha256;
use crate::plump_session::{LoadedSession, SessionBinding, SessionStamp};

pub const PF_30210: u32 = 30_210;
pub const SLOT_KEYS: [u32; 17] = [
    0x75EF, 0x75F0, 0x75F1, 0x75F2, 0x75F3, 0x75F4, 0x75F5, 0x75F6, 0x75F7, 0x75F8,
    0x75F9, 0x75FA, 0x75FB, 0x75FC, 0x75FD, 0x75FE, 0x75FF,
];
pub const SLOT_NAMES: [&str; 17] = [
    "forehead_degree",
    "foreheadSmooth",
    "foreheadBrighten",
    "midface_degree",
    "midfaceSmooth",
    "midfaceBrighten",
    "cheek",
    "noseBase_degree",
    "noseBaseSmooth",
    "noseBaseBrighten",
    "eyeHoleLeft",
    "eyeHoleRight",
    "eyeBrow",
    "chin_degree",
    "chin",
    "chinBrighten",
    "mouthAngulus",
];

const PF_TO_SLOT: [(u32, usize); 7] = [
    (30_191, 0),
    (30_194, 3),
    (30_197, 6),
    (30_198, 7),
    (30_203, 12),
    (30_204, 13),
    (30_207, 16),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    ShapeMismatch(&'static str),
    ValueOutOfRange(&'static str),
    MissingInput(&'static str),
    SessionMismatch(&'static str),
    UnsupportedPf(u32),
    NormalCompositorRejected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Roi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Roi {
    fn validate(self, image: &Image) -> Result<(), Error> {
        if self.width == 0
            || self.height == 0
            || self.x > image.width
            || self.y > image.height
            || self.width > image.width - self.x
            || self.height > image.height - self.y
        {
            return Err(Error::ValueOutOfRange("roi"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

impl Image {
    pub fn new(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, Error> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(Error::ShapeMismatch("source"))?;
        if width == 0 || height == 0 || rgba8.len() != expected {
            return Err(Error::ShapeMismatch("source"));
        }
        Ok(Self {
            width,
            height,
            rgba8,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotValues {
    pub raw: [f32; 17],
}

impl SlotValues {
    pub fn new(raw: [f32; 17]) -> Result<Self, Error> {
        if raw.iter().any(|value| !value.is_finite()) {
            return Err(Error::ValueOutOfRange("17-slot map"));
        }
        Ok(Self { raw })
    }

    pub fn effective(self) -> [f32; 17] {
        let mut out = self.raw;
        for (slot, degree) in [(1, 0), (2, 0), (4, 3), (5, 3), (8, 7), (9, 7), (14, 13), (15, 13)] {
            out[slot] *= out[degree];
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlumpRequest {
    pub face_id: u32,
    pub slots: [f32; 17],
    pub requested_pfs: Vec<u32>,
    pub full_face_plump: Option<f32>,
}

impl PlumpRequest {
    pub fn is_active(&self) -> bool {
        self.full_face_plump.is_some_and(|value| value != 0.0)
            || self.requested_pfs.iter().any(|pf| {
                PF_TO_SLOT
                    .iter()
                    .find(|(candidate, _)| candidate == pf)
                    .is_some_and(|(_, slot)| self.slots[*slot] != 0.0)
            })
    }

    pub fn slot_for_pf(&self, pf: u32) -> Option<f32> {
        PF_TO_SLOT
            .iter()
            .find(|(candidate, _)| *candidate == pf)
            .map(|(_, slot)| self.slots[*slot])
    }
}

pub struct FacePlumpFactory;

impl FacePlumpFactory {
    pub fn build_request(
        face_id: u32,
        values: SlotValues,
        requested_pfs: &[u32],
        full_face_plump: Option<f32>,
    ) -> Result<PlumpRequest, Error> {
        if face_id == 0 {
            return Err(Error::ValueOutOfRange("face id"));
        }
        let mut seen = Vec::with_capacity(requested_pfs.len());
        for &pf in requested_pfs {
            if seen.contains(&pf) {
                continue;
            }
            if !PF_TO_SLOT.iter().any(|(candidate, _)| *candidate == pf) && pf != PF_30210 {
                return Err(Error::UnsupportedPf(pf));
            }
            seen.push(pf);
        }
        if seen.contains(&PF_30210) {
            let level = full_face_plump.ok_or(Error::MissingInput("30210 descriptor"))?;
            if !level.is_finite() || !(0.0..=4.0).contains(&level) {
                return Err(Error::ValueOutOfRange("30210 descriptor"));
            }
        }
        Ok(PlumpRequest {
            face_id,
            slots: values.effective(),
            requested_pfs: seen,
            full_face_plump,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputChannelOrder {
    Rgb,
    Bgr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelInput {
    pub shape: [u32; 4],
    pub nchw: Vec<f32>,
    pub digest: [u8; 32],
}

pub fn pack_model_input(
    session: &LoadedSession,
    source: &Image,
    roi: Roi,
    order: InputChannelOrder,
    plane4: Option<&[f32]>,
) -> Result<(ModelInput, SessionStamp), Error> {
    roi.validate(source)?;
    let spec = session.spec();
    let width = spec.input_width;
    let height = spec.input_height;
    let channels = spec.input_channels;
    let mut nchw = vec![0.0f32; channels * height as usize * width as usize];
    let plane = height as usize * width as usize;

    for y in 0..height as usize {
        let sy = sample_coord(y, height, roi.height);
        for x in 0..width as usize {
            let sx = sample_coord(x, width, roi.width);
            let source_x = roi.x as f32 + sx;
            let source_y = roi.y as f32 + sy;
            let rgba = bilinear_rgba8(source, source_x, source_y);
            let rgb = match order {
                InputChannelOrder::Rgb => [rgba[0], rgba[1], rgba[2]],
                InputChannelOrder::Bgr => [rgba[2], rgba[1], rgba[0]],
            };
            let at = y * width as usize + x;
            for channel in 0..3 {
                nchw[channel * plane + at] = rgb[channel] as f32 / 255.0;
            }
        }
    }

    if channels == 4 {
        let plane4 = plane4.ok_or(Error::MissingInput("597383622 plane 4"))?;
        if plane4.len() != plane || plane4.iter().any(|value| !value.is_finite()) {
            return Err(Error::ShapeMismatch("597383622 plane 4"));
        }
        nchw[3 * plane..4 * plane].copy_from_slice(plane4);
    } else if plane4.is_some() {
        return Err(Error::ShapeMismatch("unexpected plane 4"));
    }

    let mut raw = Vec::with_capacity(nchw.len() * 4);
    for value in &nchw {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    let digest = sha256(&raw);
    let shape = [1, channels as u32, height, width];
    Ok((
        ModelInput {
            shape,
            nchw,
            digest,
        },
        session.stamp(digest),
    ))
}

fn sample_coord(dst: usize, dst_size: u32, src_size: u32) -> f32 {
    if dst_size <= 1 || src_size <= 1 {
        return 0.0;
    }
    dst as f32 * (src_size - 1) as f32 / (dst_size - 1) as f32
}

fn bilinear_rgba8(image: &Image, x: f32, y: f32) -> [f32; 4] {
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width - 1);
    let y1 = (y0 + 1).min(image.height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let sample = |px: u32, py: u32, channel: usize| {
        image.rgba8[((py as usize * image.width as usize + px as usize) * 4) + channel]
            as f32
    };
    let mut out = [0.0f32; 4];
    for channel in 0..4 {
        let top = sample(x0, y0, channel) * (1.0 - fx) + sample(x1, y0, channel) * fx;
        let bottom = sample(x0, y1, channel) * (1.0 - fx) + sample(x1, y1, channel) * fx;
        out[channel] = top * (1.0 - fy) + bottom * fy;
    }
    out
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: [u32; 4],
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn validate(&self) -> Result<(), Error> {
        let [n, c, h, w] = self.shape;
        let expected = (n as usize)
            .checked_mul(c as usize)
            .and_then(|value| value.checked_mul(h as usize))
            .and_then(|value| value.checked_mul(w as usize))
            .ok_or(Error::ShapeMismatch("model output"))?;
        if n != 1 || c < 1 || h == 0 || w == 0 || self.data.len() != expected {
            return Err(Error::ShapeMismatch("model output"));
        }
        if self.data.iter().any(|value| !value.is_finite()) {
            return Err(Error::ValueOutOfRange("model output"));
        }
        Ok(())
    }

    pub fn channel(&self, index: usize) -> Result<&[f32], Error> {
        self.validate()?;
        let [_, c, h, w] = self.shape;
        if index >= c as usize {
            return Err(Error::ShapeMismatch("model output channel"));
        }
        let plane = h as usize * w as usize;
        Ok(&self.data[index * plane..(index + 1) * plane])
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SessionOutput {
    pub stamp: SessionStamp,
    pub tensor: Tensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelMap {
    pub color: [usize; 3],
    pub split_mask: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorEncoding {
    Unit,
    Byte,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureFormat {
    Rgba8,
    R8,
    F32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub bytes: Vec<u8>,
}

impl Texture {
    fn checked_len(width: u32, height: u32, format: TextureFormat) -> Result<usize, Error> {
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or(Error::ShapeMismatch("texture"))?;
        let stride = match format {
            TextureFormat::Rgba8 => 4,
            TextureFormat::R8 => 1,
            TextureFormat::F32 => 4,
        };
        pixels
            .checked_mul(stride)
            .ok_or(Error::ShapeMismatch("texture"))
    }

    pub fn new(width: u32, height: u32, format: TextureFormat) -> Result<Self, Error> {
        let len = Self::checked_len(width, height, format)?;
        Ok(Self {
            width,
            height,
            format,
            bytes: vec![0; len],
        })
    }

    pub fn write_region(&mut self, roi: Roi, other: &Texture) -> Result<(), Error> {
        if self.format != other.format
            || other.width != roi.width
            || other.height != roi.height
            || roi.x > self.width
            || roi.y > self.height
            || roi.width > self.width - roi.x
            || roi.height > self.height - roi.y
        {
            return Err(Error::ShapeMismatch("texture write"));
        }
        let stride = match self.format {
            TextureFormat::Rgba8 => 4,
            TextureFormat::R8 => 1,
            TextureFormat::F32 => 4,
        };
        let row_bytes = roi.width as usize * stride;
        for row in 0..roi.height as usize {
            let src = row * row_bytes;
            let dst = ((roi.y as usize + row) * self.width as usize + roi.x as usize) * stride;
            self.bytes[dst..dst + row_bytes]
                .copy_from_slice(&other.bytes[src..src + row_bytes]);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlumpLayer {
    pub roi: Roi,
    pub color: Texture,
    pub alpha_raw: Texture,
    pub alpha_binary: Texture,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputSink {
    FacialPlumpingLayer,
    NormalCompositor,
}

pub struct PlumpProducer;

impl PlumpProducer {
    pub fn produce(
        binding: &SessionBinding,
        input: &ModelInput,
        output: &SessionOutput,
        channels: ChannelMap,
        encoding: ColorEncoding,
        roi: Roi,
        sink: OutputSink,
    ) -> Result<PlumpLayer, Error> {
        if sink == OutputSink::NormalCompositor {
            return Err(Error::NormalCompositorRejected);
        }
        output.tensor.validate()?;
        if output.stamp.model_id != binding.model_id
            || output.stamp.net_sha256 != binding.net_sha256
            || output.stamp.input_digest != input.digest
            || output.stamp.input_shape != input.shape
            || output.stamp.output_tensor != binding.output_tensor
        {
            return Err(Error::SessionMismatch("model output session"));
        }
        if binding.input_shape != input.shape {
            return Err(Error::SessionMismatch("model input shape"));
        }
        let [_, channel_count, _, _] = output.tensor.shape;
        if channels.color.into_iter().chain([channels.split_mask]).any(|channel| {
            channel >= channel_count as usize
        }) {
            return Err(Error::ShapeMismatch("model output channel map"));
        }

        let color_channels = [
            output.tensor.channel(channels.color[0])?,
            output.tensor.channel(channels.color[1])?,
            output.tensor.channel(channels.color[2])?,
        ];
        let split = output.tensor.channel(channels.split_mask)?;
        let output_h = output.tensor.shape[2] as usize;
        let output_w = output.tensor.shape[3] as usize;

        let mut color = Texture::new(roi.width, roi.height, TextureFormat::Rgba8)?;
        let mut alpha_raw = Texture::new(roi.width, roi.height, TextureFormat::F32)?;
        let mut alpha_binary = Texture::new(roi.width, roi.height, TextureFormat::R8)?;
        for y in 0..roi.height as usize {
            let sy = sample_coord(y, roi.height, output_h as u32);
            for x in 0..roi.width as usize {
                let sx = sample_coord(x, roi.width, output_w as u32);
                let at = y * roi.width as usize + x;
                let rgba_at = at * 4;
                for channel in 0..3 {
                    let value = bilinear_plane(color_channels[channel], output_w, output_h, sx, sy);
                    let encoded = match encoding {
                        ColorEncoding::Unit => value.clamp(0.0, 1.0) * 255.0,
                        ColorEncoding::Byte => value.clamp(0.0, 255.0),
                    };
                    color.bytes[rgba_at + channel] = encoded.round() as u8;
                }
                color.bytes[rgba_at + 3] = 255;

                let split = bilinear_plane(split, output_w, output_h, sx, sy);
                let raw = if split > 1.0 { split } else { 0.0 };
                alpha_raw.bytes[at * 4..at * 4 + 4].copy_from_slice(&raw.to_le_bytes());
                alpha_binary.bytes[at] = if raw > 0.0 { 255 } else { 0 };
            }
        }
        Ok(PlumpLayer {
            roi,
            color,
            alpha_raw,
            alpha_binary,
        })
    }
}

fn bilinear_plane(plane: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let sample = |px: usize, py: usize| plane[py * width + px];
    let top = sample(x0, y0) * (1.0 - fx) + sample(x1, y0) * fx;
    let bottom = sample(x0, y1) * (1.0 - fx) + sample(x1, y1) * fx;
    top * (1.0 - fy) + bottom * fy
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roi(width: u32, height: u32) -> Roi {
        Roi {
            x: 0,
            y: 0,
            width,
            height,
        }
    }

    fn binding() -> SessionBinding {
        SessionBinding {
            model_id: 597_383_622,
            route: crate::plump_session::Route::HighRes4,
            net_sha256: [7; 32],
            input_shape: [1, 4, 2, 2],
            input_tensor: "3071265413",
            output_tensor: "2294125226",
        }
    }

    fn input(digest: [u8; 32]) -> ModelInput {
        ModelInput {
            shape: [1, 4, 2, 2],
            nchw: vec![0.0; 16],
            digest,
        }
    }

    fn output(digest: [u8; 32]) -> SessionOutput {
        let mut data = Vec::new();
        for channel in 0..4 {
            for pixel in 0..4 {
                data.push(if channel == 3 {
                    if pixel < 2 { 0.0 } else { 2.0 }
                } else {
                    (channel as f32 + 1.0) / 4.0
                });
            }
        }
        SessionOutput {
            stamp: SessionStamp {
                model_id: binding().model_id,
                net_sha256: binding().net_sha256,
                input_digest: digest,
                input_shape: binding().input_shape,
                output_tensor: binding().output_tensor,
            },
            tensor: Tensor {
                shape: [1, 4, 2, 2],
                data,
            },
        }
    }

    #[test]
    fn seventeen_slots_apply_the_recovered_degree_products_once() {
        let mut raw = [1.0; 17];
        raw[0] = 0.5;
        raw[3] = 0.25;
        raw[7] = 2.0;
        raw[13] = 4.0;
        let values = SlotValues::new(raw).unwrap().effective();
        assert_eq!(values[0], 0.5);
        assert_eq!(values[1], 0.5);
        assert_eq!(values[2], 0.5);
        assert_eq!(values[4], 0.25);
        assert_eq!(values[5], 0.25);
        assert_eq!(values[8], 2.0);
        assert_eq!(values[9], 2.0);
        assert_eq!(values[14], 4.0);
        assert_eq!(values[15], 4.0);
    }

    #[test]
    fn full_face_descriptor_is_never_slot_eighteen() {
        let request = FacePlumpFactory::build_request(
            9,
            SlotValues::new([0.5; 17]).unwrap(),
            &[30_203, 30_204, 30_207, PF_30210],
            Some(3.0),
        )
        .unwrap();
        assert_eq!(request.slots.len(), 17);
        assert_eq!(request.slot_for_pf(30_203), Some(0.5));
        assert_eq!(request.slot_for_pf(30_204), Some(0.5));
        assert_eq!(request.slot_for_pf(30_207), Some(0.5));
        assert_eq!(request.full_face_plump, Some(3.0));
    }

    #[test]
    fn model_tensor_is_written_to_layer_alpha_and_binary_mask() {
        let digest = [3; 32];
        let layer = PlumpProducer::produce(
            &binding(),
            &input(digest),
            &output(digest),
            ChannelMap {
                color: [0, 1, 2],
                split_mask: 3,
            },
            ColorEncoding::Unit,
            roi(4, 4),
            OutputSink::FacialPlumpingLayer,
        )
        .unwrap();
        assert_eq!(layer.color.format, TextureFormat::Rgba8);
        assert_eq!(layer.color.bytes[0..4], [64, 128, 191, 255]);
        assert_eq!(
            layer.alpha_binary.bytes,
            [0; 8].into_iter().chain([255; 8]).collect::<Vec<_>>()
        );
        assert_eq!(f32::from_le_bytes(layer.alpha_raw.bytes[0..4].try_into().unwrap()), 0.0);
        assert_eq!(
            f32::from_le_bytes(layer.alpha_raw.bytes[60..64].try_into().unwrap()),
            2.0
        );
    }

    #[test]
    fn normal_compositor_is_refused_by_name() {
        let digest = [3; 32];
        assert_eq!(
            PlumpProducer::produce(
                &binding(),
                &input(digest),
                &output(digest),
                ChannelMap {
                    color: [0, 1, 2],
                    split_mask: 3,
                },
                ColorEncoding::Unit,
                roi(4, 4),
                OutputSink::NormalCompositor,
            ),
            Err(Error::NormalCompositorRejected)
        );
    }

    #[test]
    fn model_output_from_another_session_is_refused() {
        let mut other = output([3; 32]);
        other.stamp.input_digest = [9; 32];
        assert_eq!(
            PlumpProducer::produce(
                &binding(),
                &input([3; 32]),
                &other,
                ChannelMap {
                    color: [0, 1, 2],
                    split_mask: 3,
                },
                ColorEncoding::Unit,
                roi(4, 4),
                OutputSink::FacialPlumpingLayer,
            ),
            Err(Error::SessionMismatch("model output session"))
        );
    }

    #[test]
    fn four_plane_model_requires_an_explicit_plane4() {
        let source = Image::new(2, 2, vec![255; 16]).unwrap();
        let session = crate::plump_session::LoadedSession::open(crate::plump_session::Route::HighRes4);
        if let Ok(session) = session {
            assert_eq!(
                pack_model_input(&session, &source, roi(2, 2), InputChannelOrder::Rgb, None),
                Err(Error::MissingInput("597383622 plane 4"))
            );
        }
    }
}

