//! PF101 Silkworm material stage.
//!
//! The recovered final consumer is `OverLay/BlendImageLayerWithMask.frag`.
//! PF101 fixes `EyeMaskType=0` to independent face-mask input id `0x15` and
//! its R component. `wcz*.png` / `wcy*.png` are material RGBA inputs, never
//! substitutes for that face mask. The nine effect records contain 34 material
//! rows: 14 Multiply (41), 14 SoftLight (42), 2 Lighten (43), 4 Screen (44).

use crate::kernels::makeup_material::{self, BlendMode, MakeupLayer, MaskChannel};
use crate::{Access, EffectMeta, Image, Refusal};

pub const PF: u32 = 101;
pub const EYE_MASK_TYPE: u32 = 0;
pub const FACE_MASK_INPUT_ID: u16 = 0x15;
pub const FILTER_TYPE: u32 = 42;
pub const FILTER_ROWS: [(u32, usize); 4] = [(41, 14), (42, 14), (43, 2), (44, 4)];
pub const EFFECT_CONFIG_COUNT: usize = 9;
pub const EFFECT_ROW_COUNT: usize = 34;
pub const ADJUST_FACTOR_BITS: u32 = 0x3F33_3333;
pub const ADJUST_FACTOR: f32 = f32::from_bits(ADJUST_FACTOR_BITS);
pub const SHADER: &str = "glsl/OverLay/BlendImageLayerWithMask.frag";
pub const FILTER_TYPE_SELECTOR_RVA: u32 = 0x60B8A70;
pub const MASK_CHANNEL_SELECTOR_RVA: u32 = 0x60B86E0;
pub const MASK_WEIGHT_ENTRY_RVA: u32 = 0x4C5AC30;
pub const DRAW_ENTRY_RVA: u32 = 0x4C5B370;
pub const DRAW_FACTOR_MERGE_RVA: u32 = 0x4AD6C50;
pub const F8_RECORD_SELECTOR_RVA: u32 = 0x60C3100;
pub const F8_DRAW_STATE_OFFSET: usize = 0x2D0;
pub const F8_CALL_RVA: u32 = 0x4C5B78B;
pub const VTABLE_F8_OFFSET: usize = 0xF8;
pub const COMMON_DRAW_LOOP_RVA: u32 = 0x4AECAC8;
pub const DRAW_CALL_RVA: u32 = 0x4AECACF;
pub const DRAW_MODE_GL_TRIANGLES: u32 = 4;
pub const DRAW_INDEX_COUNT: u32 = 0x1403;
pub const SHADER_SOURCE_RVA: u32 = 0x967BFD2;
pub const SHADER_SOURCE_BYTES: usize = 7579;
pub const SHADER_SOURCE_SHA256: &str =
    "b5ec8244f8e8eff13105c7c7f47273745c6a137051746128a10f037a894764b1";
pub const SHADER_FIXED_DEFINES: &[&str] = &["MASK_R", "REFINE_MATERIAL_COLOR", "FILPMASK"];
pub const SHADER_SOURCE_DEFINES: &[&str] =
    &["SOFT_LIGHT", "MASK_R", "REFINE_MATERIAL_COLOR", "FILPMASK"];
pub const FILTER_DEFINES: [(u32, &str); 4] = [
    (41, "MULTIPLY"),
    (42, "SOFT_LIGHT"),
    (43, "LIGHTEN"),
    (44, "SCREEN"),
];

pub const META: &[EffectMeta] = &[EffectMeta {
    pf: PF,
    key: "silkwormDegree",
    label: "Silkworm degree",
    access: Access::Cpu,
    inputs: &[
        "face mask input id 0x15 (EyeMaskType=0/MASK_R)",
        "wcz/wcy material RGBA",
        "material UV",
        "face mask UV",
        "W(maskIndex=0) scalar",
        "W(maskIndex=1) scalar",
        "AdjustFactor gate",
        "FilterType 41/42/43/44 effect record",
        "compiled FilterType + MASK_R + REFINE_MATERIAL_COLOR + FILPMASK",
        "draw entry 0x4C5B370; glDrawElements(4, 0x1403)",
        "per-face factor merge 0x4AD6C50 -> vtable+0xF8 at 0x4C5B78B (dynamic callee)",
    ],
}];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SilkwormFilter {
    Multiply,
    SoftLight,
    Lighten,
    Screen,
}

impl SilkwormFilter {
    pub fn from_filter_type(filter_type: u32) -> Result<Self, Refusal> {
        match filter_type {
            41 => Ok(Self::Multiply),
            42 => Ok(Self::SoftLight),
            43 => Ok(Self::Lighten),
            44 => Ok(Self::Screen),
            _ => Err(Refusal::MissingInput("pf101_filter_type")),
        }
    }

    pub fn blend_mode(self) -> BlendMode {
        match self {
            Self::Multiply => BlendMode::Multiply,
            Self::SoftLight => BlendMode::SoftLight,
            Self::Lighten => BlendMode::Lighten,
            Self::Screen => BlendMode::Screen,
        }
    }

    pub fn define(self) -> &'static str {
        match self {
            Self::Multiply => "MULTIPLY",
            Self::SoftLight => "SOFT_LIGHT",
            Self::Lighten => "LIGHTEN",
            Self::Screen => "SCREEN",
        }
    }

    pub fn shader_defines(self) -> [&'static str; 4] {
        [
            self.define(),
            "MASK_R",
            "REFINE_MATERIAL_COLOR",
            "FILPMASK",
        ]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SilkwormMask<'a> {
    image: &'a Image,
    input_id: u16,
}

impl<'a> SilkwormMask<'a> {
    /// The only accepted PF101 mask is the face-mask record selected by id 0x15.
    pub fn face_0x15(image: &'a Image) -> Self {
        Self {
            image,
            input_id: FACE_MASK_INPUT_ID,
        }
    }

    pub fn input_id(self) -> u16 {
        self.input_id
    }

    fn validate(self) -> Result<&'a Image, Refusal> {
        if self.input_id != FACE_MASK_INPUT_ID {
            return Err(Refusal::MissingInput("pf101_face_mask_0x15"));
        }
        self.image.validate()?;
        Ok(self.image)
    }
}

/// `EyeMaskType=0` is the only accepted PF101 selector value.
pub fn eye_mask_channel(eye_mask_type: u32) -> Result<MaskChannel, Refusal> {
    if eye_mask_type == EYE_MASK_TYPE {
        Ok(MaskChannel::R)
    } else {
        Err(Refusal::MissingInput("pf101_eye_mask_type"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SilkwormParams {
    pub filter_type: u32,
    /// `W(cell, face, maskIndex=0)` used by the draw entry.
    pub main_mask_scalar: f32,
    /// `W(cell, face, maskIndex=1)`, the `xmm7` value.
    pub axis_weight_w1: f32,
    /// Result of the live geometry predicate at `0x60C17C0(face, 75.0f)`.
    pub adjust_gate: bool,
}

impl SilkwormParams {
    pub fn degree01(self) -> Result<f32, Refusal> {
        SilkwormFilter::from_filter_type(self.filter_type)?;
        for weight in [self.main_mask_scalar, self.axis_weight_w1] {
            if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                return Err(Refusal::ValueOutOfRange("silkworm mask weights"));
            }
        }
        let adjust = if self.adjust_gate { ADJUST_FACTOR } else { 1.0 };
        let degree = self.main_mask_scalar * self.axis_weight_w1 * adjust;
        if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
            return Err(Refusal::ValueOutOfRange("silkworm degree"));
        }
        Ok(degree)
    }

    pub fn layer(self) -> Result<MakeupLayer, Refusal> {
        SilkwormFilter::from_filter_type(self.filter_type)?;
        Ok(MakeupLayer {
            mix_color: [0, 0, 0, 255],
            opacity_percent: 100.0,
            degree01: self.degree01()?,
            filter_type: self.filter_type,
            mask_channel: eye_mask_channel(EYE_MASK_TYPE)?,
        })
    }
}

/// CPU path for the recovered `BlendImageLayerWithMask.frag` law.
#[allow(clippy::too_many_arguments)]
pub fn apply_image_layer(
    source: &Image,
    material: &Image,
    mask: SilkwormMask<'_>,
    mask_uv: &[[f32; 2]],
    material_uv: &[[f32; 2]],
    params: SilkwormParams,
) -> Result<Image, Refusal> {
    let mask = mask.validate()?;
    makeup_material::image_makeup(
        source,
        material,
        mask,
        mask_uv,
        material_uv,
        params.layer()?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::makeup_material::{BlendMode, ColorMaterialParams};

    fn byte_even(value: f32) -> u8 {
        let value = value.clamp(0.0, 1.0) * 255.0;
        let floor = value.floor();
        let fraction = value - floor;
        let rounded = if fraction > 0.5 {
            floor + 1.0
        } else if fraction < 0.5 {
            floor
        } else if (floor as i32) & 1 == 0 {
            floor
        } else {
            floor + 1.0
        };
        rounded.clamp(0.0, 255.0) as u8
    }

    fn soft_light(base: f32, blend: f32) -> f32 {
        if blend < 0.5 {
            2.0 * base * blend + base * base * (1.0 - 2.0 * blend)
        } else {
            base.sqrt() * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend)
        }
    }

    #[test]
    fn fixed_contract() {
        assert_eq!(ADJUST_FACTOR_BITS, 0x3F33_3333);
        assert_eq!(ADJUST_FACTOR, 0.699_999_988_079_071);
        assert_eq!(DRAW_FACTOR_MERGE_RVA, 0x4AD6_C50);
        assert_eq!(F8_RECORD_SELECTOR_RVA, 0x60C3_100);
        assert_eq!(F8_DRAW_STATE_OFFSET, 0x2D0);
        assert_eq!(F8_CALL_RVA, 0x4C5B_78B);
        assert_eq!(VTABLE_F8_OFFSET, 0xF8);
        assert_eq!(SHADER_SOURCE_RVA, 0x967B_FD2);
        assert_eq!(SHADER_SOURCE_BYTES, 7579);
        assert_eq!(
            SHADER_SOURCE_SHA256,
            "b5ec8244f8e8eff13105c7c7f47273745c6a137051746128a10f037a894764b1"
        );
        assert_eq!(
            SHADER_SOURCE_DEFINES,
            [
                "SOFT_LIGHT",
                "MASK_R",
                "REFINE_MATERIAL_COLOR",
                "FILPMASK"
            ]
        );
        assert_eq!(EFFECT_CONFIG_COUNT, 9);
        assert_eq!(EFFECT_ROW_COUNT, 34);
        assert_eq!(FILTER_ROWS.iter().map(|row| row.1).sum::<usize>(), 34);
        assert_eq!(eye_mask_channel(0).unwrap(), MaskChannel::R);
        assert_eq!(
            eye_mask_channel(4).unwrap_err(),
            Refusal::MissingInput("pf101_eye_mask_type")
        );
        assert_eq!(
            SilkwormMask::face_0x15(&Image::solid(1, 1, [0; 4])).input_id(),
            FACE_MASK_INPUT_ID
        );
    }

    #[test]
    fn all_recovered_filter_types_map_to_their_macros() {
        let expected = [
            (41, SilkwormFilter::Multiply, BlendMode::Multiply),
            (42, SilkwormFilter::SoftLight, BlendMode::SoftLight),
            (43, SilkwormFilter::Lighten, BlendMode::Lighten),
            (44, SilkwormFilter::Screen, BlendMode::Screen),
        ];
        for (filter_type, selector, blend_mode) in expected {
            assert_eq!(SilkwormFilter::from_filter_type(filter_type), Ok(selector));
            assert_eq!(selector.blend_mode(), blend_mode);
            assert_eq!(
                FILTER_DEFINES
                    .iter()
                    .find(|row| row.0 == filter_type)
                    .map(|row| row.1),
                Some(selector.define())
            );
            assert_eq!(
                selector.shader_defines(),
                [
                    selector.define(),
                    "MASK_R",
                    "REFINE_MATERIAL_COLOR",
                    "FILPMASK"
                ]
            );
            let layer = SilkwormParams {
                filter_type,
                main_mask_scalar: 1.0,
                axis_weight_w1: 1.0,
                adjust_gate: false,
            }
            .layer()
            .unwrap();
            assert_eq!(layer.params_of().unwrap().blend_mode, blend_mode);
        }
        assert_eq!(
            SilkwormFilter::from_filter_type(40),
            Err(Refusal::MissingInput("pf101_filter_type"))
        );
    }

    #[test]
    fn adjust_factor_only_applies_when_gate_is_true() {
        let params = SilkwormParams {
            filter_type: FILTER_TYPE,
            main_mask_scalar: 0.5,
            axis_weight_w1: 0.5,
            adjust_gate: true,
        };
        assert_eq!(params.degree01().unwrap(), 0.5 * 0.5 * ADJUST_FACTOR);
        assert_eq!(
            SilkwormParams {
                adjust_gate: false,
                ..params
            }
            .degree01()
            .unwrap(),
            0.25
        );
    }

    #[test]
    fn macro_mapping_is_soft_light_and_mask_r() {
        let layer = SilkwormParams {
            filter_type: FILTER_TYPE,
            main_mask_scalar: 0.5,
            axis_weight_w1: 0.5,
            adjust_gate: true,
        }
        .layer()
        .unwrap();
        let params: ColorMaterialParams = layer.params_of().unwrap();
        assert_eq!(params.blend_mode, BlendMode::SoftLight);
        assert_eq!(params.mask_channel, MaskChannel::R);
        assert_eq!(params.mix_color, [0, 0, 0, 255]);
        assert_eq!(params.opacity_percent, 100.0);
    }

    #[test]
    fn t17_soft_light_fixture_is_byte_exact() {
        let source = Image::solid(1, 1, [100, 150, 200, 77]);
        let material = Image::solid(1, 1, [255, 255, 255, 128]);
        let mask = Image::solid(1, 1, [128, 64, 32, 16]);
        let params = SilkwormParams {
            filter_type: FILTER_TYPE,
            main_mask_scalar: 1.0,
            axis_weight_w1: 1.0,
            adjust_gate: true,
        };
        let out = apply_image_layer(
            &source,
            &material,
            SilkwormMask::face_0x15(&mask),
            &[[0.5, 0.5]],
            &[[0.5, 0.5]],
            params,
        )
        .unwrap();
        let factor = (128.0f32 / 255.0) * (128.0f32 / 255.0) * ADJUST_FACTOR;
        let src = [100.0f32 / 255.0, 150.0 / 255.0, 200.0 / 255.0];
        let expected = [
            byte_even(src[0] * (1.0 - factor) + soft_light(src[0], 1.0) * factor),
            byte_even(src[1] * (1.0 - factor) + soft_light(src[1], 1.0) * factor),
            byte_even(src[2] * (1.0 - factor) + soft_light(src[2], 1.0) * factor),
            77,
        ];
        assert_eq!(expected, [111, 158, 205, 77]);
        assert_eq!(out.pixel(0, 0), expected);
    }
}
