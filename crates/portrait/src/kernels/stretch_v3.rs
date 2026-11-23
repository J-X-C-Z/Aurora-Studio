//! PF90008 `PixCook::EffectStretchMark` (`StretchV3Pipeline`) CPU boundary.
//!
//! The recovered processor, model slot and render slot are retained as ABI
//! evidence. The original model weights, tensor graph, non-identity affine and
//! PF-specific final blend are not present in this tree, so those paths refuse
//! instead of inventing a transform or model forward.

use std::hash::{Hash, Hasher};

use crate::kernels::mask_compose::MaskComposeMode;
use crate::kernels::skin_adjust::Roi;
use crate::{model_sha256, Image, Refusal};

pub const PF: u32 = 90008;
pub const PROCESSOR_ORDER: u64 = 0x1e9d_d78;
pub const MODEL_SLOT: u64 = 0x7ce_bcf0;
pub const RENDER_SLOT: u64 = 0x7ce_bf00;
pub const POSITIVE_GATE: f32 = 0.0001;

pub const TENSOR_BLOCKER: &str =
    "PF90008 original tensor/weights absent; StretchV3 model forward is refused";
pub const AFFINE_BLOCKER: &str =
    "PF90008 non-identity affine mapping is not recovered; transform execution is refused";
pub const DEGREE_BLOCKER: &str =
    "PF90008 degree-to-model-output law is not recovered; unpainted model texture is refused";
pub const PRODUCT_BLEND_BLOCKER: &str =
    "PF90008 PF-specific final blend is not recovered; shared 0x5B8CEA0 mask compose is oracle-only";

/// A row-strided RGBA8 Mat. Padding bytes are retained but never sampled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgba8Mat {
    pub width: u32,
    pub height: u32,
    pub row_stride: u32,
    pub data: Vec<u8>,
}

impl Rgba8Mat {
    pub fn packed(image: Image) -> Self {
        Self {
            width: image.width,
            height: image.height,
            row_stride: image.width * 4,
            data: image.pixels,
        }
    }

    fn required_end(&self, label: &'static str) -> Result<usize, Refusal> {
        if self.width == 0 || self.height == 0 || self.row_stride < self.width.saturating_mul(4) {
            return Err(Refusal::ShapeMismatch(label));
        }
        let last_row = (self.height as usize - 1)
            .checked_mul(self.row_stride as usize)
            .ok_or(Refusal::ShapeMismatch(label))?;
        last_row
            .checked_add(self.width as usize * 4)
            .ok_or(Refusal::ShapeMismatch(label))
    }

    fn validate(&self, label: &'static str) -> Result<(), Refusal> {
        if self.data.len() < self.required_end(label)? {
            return Err(Refusal::ShapeMismatch(label));
        }
        Ok(())
    }

    fn row(&self, y: u32) -> &[u8] {
        let start = y as usize * self.row_stride as usize;
        &self.data[start..start + self.width as usize * 4]
    }

    fn logical_pixels(&self) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(self.width as usize * self.height as usize * 4);
        for y in 0..self.height {
            pixels.extend_from_slice(self.row(y));
        }
        pixels
    }

    fn image(&self, label: &'static str) -> Result<Image, Refusal> {
        self.validate(label)?;
        Image::new(self.width, self.height, self.logical_pixels())
            .ok_or(Refusal::ShapeMismatch(label))
    }

    fn copy_from(&mut self, source: &Image, label: &'static str) -> Result<(), Refusal> {
        self.validate(label)?;
        if (self.width, self.height) != (source.width, source.height) {
            return Err(Refusal::ShapeMismatch(label));
        }
        for y in 0..self.height {
            let start = y as usize * self.row_stride as usize;
            let end = start + self.width as usize * 4;
            let src = y as usize * self.width as usize * 4;
            self.data[start..end]
                .copy_from_slice(&source.pixels[src..src + self.width as usize * 4]);
        }
        Ok(())
    }
}

/// A row-strided one-byte mask Mat with a ROI-tight payload selector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskMat {
    pub width: u32,
    pub height: u32,
    pub row_stride: u32,
    pub data: Vec<u8>,
}

impl MaskMat {
    pub fn packed(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            row_stride: width,
            data,
        }
    }

    fn validate(&self) -> Result<(), Refusal> {
        if self.width == 0 || self.height == 0 || self.row_stride < self.width {
            return Err(Refusal::ShapeMismatch("PF90008 mask mat"));
        }
        let last_row = (self.height as usize - 1)
            .checked_mul(self.row_stride as usize)
            .ok_or(Refusal::ShapeMismatch("PF90008 mask mat"))?;
        let end = last_row
            .checked_add(self.width as usize)
            .ok_or(Refusal::ShapeMismatch("PF90008 mask mat"))?;
        if self.data.len() < end {
            return Err(Refusal::ShapeMismatch("PF90008 mask mat"));
        }
        Ok(())
    }

    fn roi_bytes(&self, roi: Roi) -> Result<Vec<u8>, Refusal> {
        self.validate()?;
        roi.validate(self.width, self.height)?;
        let mut bytes = Vec::with_capacity(roi.width as usize * roi.height as usize);
        for y in roi.y..roi.y + roi.height {
            let start = y as usize * self.row_stride as usize + roi.x as usize;
            bytes.extend_from_slice(&self.data[start..start + roi.width as usize]);
        }
        Ok(bytes)
    }
}

/// Output texture stride. Execution allocates the texture and returns it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputTexture {
    pub row_stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Affine2x3 {
    pub matrix: [u32; 6],
}

impl Affine2x3 {
    pub const IDENTITY_F32: [f32; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DegreeState {
    Applied,
    NotApplied,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorDType {
    F32,
    F16,
    U8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorShape {
    pub dims: [u32; 4],
    pub dtype: TensorDType,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModelInput {
    /// Frame-space texture produced by the external model. Identity affine only.
    PrewarpedFrame {
        texture: Rgba8Mat,
        texture_sha256: [u8; 32],
        degree: DegreeState,
    },
    /// Typed intake so real vendor weights can be attached when recovered.
    Tensor {
        weights: Vec<u8>,
        input: TensorShape,
        output: TensorShape,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendLaw {
    /// Existing shared composer `0x5B8CEA0`; not a PF90008 parity claim.
    SharedMaskCompose(MaskComposeMode),
}

/// Same-job input ticket for the executable frame-space oracle subset.
#[derive(Clone, Debug, PartialEq)]
pub struct StretchV3StageTicket {
    pub model: ModelInput,
    pub mask: MaskMat,
    pub source_roi: Roi,
    pub mask_roi: Roi,
    pub affine: Affine2x3,
    pub output: OutputTexture,
    pub blend: BlendLaw,
}

impl StretchV3StageTicket {
    pub fn shared_oracle(
        model: Image,
        mask: Vec<u8>,
        roi: Roi,
        alternate: bool,
        output_stride: u32,
    ) -> Self {
        let texture_sha256 = model_sha256::sha256(&model.pixels);
        Self {
            model: ModelInput::PrewarpedFrame {
                texture: Rgba8Mat::packed(model),
                texture_sha256,
                degree: DegreeState::Applied,
            },
            mask: MaskMat::packed(roi.width, roi.height, mask),
            source_roi: roi,
            mask_roi: Roi {
                x: 0,
                y: 0,
                width: roi.width,
                height: roi.height,
            },
            affine: Affine2x3 {
                matrix: Affine2x3::IDENTITY_F32.map(f32::to_bits),
            },
            output: OutputTexture {
                row_stride: output_stride,
            },
            blend: BlendLaw::SharedMaskCompose(if alternate {
                MaskComposeMode::Alternate
            } else {
                MaskComposeMode::MaskOnly
            }),
        }
    }

    pub fn validate_for_size(&self, size: [u32; 2]) -> Result<(), Refusal> {
        let model = match &self.model {
            ModelInput::PrewarpedFrame { texture, .. } => texture,
            ModelInput::Tensor { weights, .. } => {
                if weights.is_empty() {
                    return Err(Refusal::MissingInput("PF90008 model weights"));
                }
                return Err(Refusal::MissingInput("PF90008 model forward"));
            }
        };
        model.validate("PF90008 model texture")?;
        if [model.width, model.height] != size {
            return Err(Refusal::ShapeMismatch("PF90008 model texture size"));
        }
        self.mask.validate()?;
        self.source_roi.validate(size[0], size[1])?;
        self.mask_roi.validate(self.mask.width, self.mask.height)?;
        if (self.mask_roi.width, self.mask_roi.height)
            != (self.source_roi.width, self.source_roi.height)
        {
            return Err(Refusal::ShapeMismatch("PF90008 mask roi"));
        }
        if self.output.row_stride < size[0].saturating_mul(4) {
            return Err(Refusal::ShapeMismatch("PF90008 output texture"));
        }
        Ok(())
    }

    /// Executes only the explicit frame-space texture + mask + stride subset.
    pub fn execute(
        &self,
        source: &Rgba8Mat,
        ticket: Option<u64>,
        control: f32,
    ) -> Result<Rgba8Mat, Refusal> {
        if ticket == Some(0) || ticket.is_none() {
            return Err(Refusal::ValueOutOfRange("PF90008 ticket"));
        }
        if !control.is_finite() || !(0.0..=100.0).contains(&control) {
            return Err(Refusal::ValueOutOfRange("PF90008 control"));
        }
        if control / 100.0 <= POSITIVE_GATE {
            return Err(Refusal::ValueOutOfRange("PF90008 positive gate"));
        }
        source.validate("PF90008 source mat")?;
        self.validate_for_size([source.width, source.height])?;

        if self.affine.matrix != Affine2x3::IDENTITY_F32.map(f32::to_bits) {
            return Err(Refusal::MissingInput(AFFINE_BLOCKER));
        }

        let (texture, texture_sha256, degree) = match &self.model {
            ModelInput::PrewarpedFrame {
                texture,
                texture_sha256,
                degree,
            } => (texture, texture_sha256, degree),
            ModelInput::Tensor { weights, .. } => {
                if weights.is_empty() {
                    return Err(Refusal::MissingInput("PF90008 model weights"));
                }
                return Err(Refusal::MissingInput("PF90008 model forward"));
            }
        };
        if *degree != DegreeState::Applied {
            return Err(Refusal::MissingInput(DEGREE_BLOCKER));
        }
        if model_sha256::sha256(&texture.logical_pixels()) != *texture_sha256 {
            return Err(Refusal::MissingInput("PF90008 model texture digest"));
        }

        let source_image = source.image("PF90008 source mat")?;
        let model_image = texture.image("PF90008 model texture")?;
        let mask = self.mask.roi_bytes(self.mask_roi)?;
        let mode = match self.blend {
            BlendLaw::SharedMaskCompose(mode) => mode,
        };
        let composed = crate::kernels::skin_adjust::masked_model_output(
            &source_image,
            &model_image,
            self.source_roi,
            &mask,
            mode == MaskComposeMode::Alternate,
        )?;

        let mut output = Rgba8Mat {
            width: source.width,
            height: source.height,
            row_stride: self.output.row_stride,
            data: vec![
                0;
                self.output.row_stride as usize * source.height as usize
            ],
        };
        output.copy_from(&source_image, "PF90008 output texture")?;
        for row in 0..self.source_roi.height {
            let y = self.source_roi.y + row;
            let src = ((y * source.width + self.source_roi.x) * 4) as usize;
            let dst = y as usize * output.row_stride as usize
                + self.source_roi.x as usize * 4;
            let len = self.source_roi.width as usize * 4;
            output.data[dst..dst + len].copy_from_slice(&composed.pixels[src..src + len]);
        }
        Ok(output)
    }

    pub fn execute_image(
        &self,
        source: &Image,
        ticket: Option<u64>,
        control: f32,
    ) -> Result<Image, Refusal> {
        let mat = Rgba8Mat {
            width: source.width,
            height: source.height,
            row_stride: source.width * 4,
            data: source.pixels.clone(),
        };
        let output = self.execute(&mat, ticket, control)?;
        output.image("PF90008 output texture")
    }

    pub fn hash_into(&self, state: &mut impl Hasher) {
        match &self.model {
            ModelInput::PrewarpedFrame {
                texture,
                texture_sha256,
                degree,
            } => {
                state.write_u8(0);
                texture.width.hash(state);
                texture.height.hash(state);
                texture.row_stride.hash(state);
                texture.data.hash(state);
                texture_sha256.hash(state);
                (*degree as u8).hash(state);
            }
            ModelInput::Tensor {
                weights,
                input,
                output,
            } => {
                state.write_u8(1);
                weights.hash(state);
                input.dims.hash(state);
                (input.dtype as u8).hash(state);
                output.dims.hash(state);
                (output.dtype as u8).hash(state);
            }
        }
        self.mask.width.hash(state);
        self.mask.height.hash(state);
        self.mask.row_stride.hash(state);
        self.mask.data.hash(state);
        self.source_roi.hash_fields(state);
        self.mask_roi.hash_fields(state);
        self.affine.matrix.hash(state);
        self.output.row_stride.hash(state);
        match self.blend {
            BlendLaw::SharedMaskCompose(mode) => (mode as u8).hash(state),
        }
    }
}

trait RoiHash {
    fn hash_fields(&self, state: &mut impl Hasher);
}

impl RoiHash for Roi {
    fn hash_fields(&self, state: &mut impl Hasher) {
        self.x.hash(state);
        self.y.hash(state);
        self.width.hash(state);
        self.height.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_stride() -> Rgba8Mat {
        Rgba8Mat {
            width: 2,
            height: 2,
            row_stride: 12,
            data: vec![
                1, 2, 3, 4, 10, 20, 30, 40, 99, 99, 99, 99, 5, 6, 7, 8, 50, 60, 70, 80, 99, 99,
                99, 99,
            ],
        }
    }

    fn ticket() -> StretchV3StageTicket {
        let model = Image::new(
            2,
            2,
            vec![
                0, 0, 0, 0, 100, 150, 200, 5, 0, 0, 0, 0, 80, 120, 160, 9,
            ],
        )
        .unwrap();
        let mut ticket = StretchV3StageTicket::shared_oracle(
            model,
            vec![0, 9, 0, 128],
            Roi {
                x: 1,
                y: 1,
                width: 1,
                height: 1,
            },
            false,
            12,
        );
        ticket.mask = MaskMat {
            width: 2,
            height: 2,
            row_stride: 3,
            data: vec![0, 0, 77, 0, 128, 88, 0, 0, 66],
        };
        ticket.mask_roi = Roi {
            x: 1,
            y: 1,
            width: 1,
            height: 1,
        };
        ticket
    }

    #[test]
    fn strided_mats_and_mask_write_only_the_roi() {
        let output = ticket().execute(&source_stride(), Some(7), 100.0).unwrap();
        assert_eq!(&output.data[0..4], &[1, 2, 3, 4]);
        assert_eq!(&output.data[4..8], &[10, 20, 30, 40]);
        assert_eq!(&output.data[8..12], &[0, 0, 0, 0]);
        assert_eq!(&output.data[12..16], &[5, 6, 7, 8]);
        assert_eq!(&output.data[16..20], &[40, 60, 80, 9]);
        assert_eq!(&output.data[20..24], &[0, 0, 0, 0]);
    }

    #[test]
    fn tensor_intake_is_typed_but_refused_without_forward() {
        let mut ticket = ticket();
        ticket.model = ModelInput::Tensor {
            weights: vec![1, 2, 3],
            input: TensorShape {
                dims: [1, 3, 2, 2],
                dtype: TensorDType::F32,
            },
            output: TensorShape {
                dims: [1, 4, 2, 2],
                dtype: TensorDType::F32,
            },
        };
        assert_eq!(
            ticket.execute(&source_stride(), Some(7), 100.0),
            Err(Refusal::MissingInput("PF90008 model forward"))
        );
    }

    #[test]
    fn non_identity_affine_and_unpainted_texture_refuse() {
        let mut ticket = ticket();
        ticket.affine.matrix[2] = 1;
        assert_eq!(
            ticket.execute(&source_stride(), Some(7), 100.0),
            Err(Refusal::MissingInput(AFFINE_BLOCKER))
        );

        let mut ticket = ticket();
        if let ModelInput::PrewarpedFrame { degree, .. } = &mut ticket.model {
            *degree = DegreeState::NotApplied;
        }
        assert_eq!(
            ticket.execute(&source_stride(), Some(7), 100.0),
            Err(Refusal::MissingInput(DEGREE_BLOCKER))
        );
    }

    #[test]
    fn positive_gate_and_digest_are_strict() {
        assert_eq!(
            ticket().execute(&source_stride(), Some(7), 0.01),
            Err(Refusal::ValueOutOfRange("PF90008 positive gate"))
        );
        let mut ticket = ticket();
        if let ModelInput::PrewarpedFrame {
            texture_sha256, ..
        } = &mut ticket.model
        {
            *texture_sha256 = [0; 32];
        }
        assert_eq!(
            ticket.execute(&source_stride(), Some(7), 100.0),
            Err(Refusal::MissingInput("PF90008 model texture digest"))
        );
    }
}
