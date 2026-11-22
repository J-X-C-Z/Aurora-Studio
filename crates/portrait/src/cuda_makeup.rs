//! Shared makeup layer (`BlendImageLayerWithMask`) on NVIDIA.
//!
//! Same law as `kernels::makeup_material::apply_image_layer_dual`:
//! `factor = clamp01(material.a * mix_alpha * mask_weight)`, a `<= 0.01` gate,
//! Normal / Multiply / SoftLight / PinLight, optional `BLEND_ALPHA`, and two
//! independent UV streams. Source, material and owner mask may differ in size.
//!
//! The uniform block is passed **by value** (48 bytes, `struct` parameter), never
//! as a pointer; the PTX signature is the contract.

use crate::cuda::{CudaError, Dispatch};
use crate::kernels::makeup_material::{
    BlendMode, ColorMaterialParams, MakeupLayer as CpuMakeupLayer, MaskChannel,
};
use crate::kernels::makeup_mask_uv::color_mask_profile;
use crate::{Image, Refusal};

/// Recovered kernel entry point.
pub const KERNEL: &str = "makeup_layer_kernel";
const BLOCK: u32 = 16;

const PTX: &str = include_str!("../native/cuda/makeup_layer.ptx");

/// 48-byte kernel argument block, matching `native/cuda/makeup_layer.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub material_width: u32,
    pub material_height: u32,
    pub mask_width: u32,
    pub mask_height: u32,
    pub mix_alpha: f32,
    pub mode: u32,
    pub mask_channel: u32,
    pub blend_alpha: u32,
    pub reserved0: u32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 48);

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn uv_bytes(uv: &[[f32; 2]]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(uv.as_ptr() as *const u8, std::mem::size_of_val(uv)) }
}

/// Only the four recovered modes are executable; the rest refuse.
fn mode_index(mode: BlendMode) -> Option<u32> {
    match mode {
        BlendMode::Normal => Some(0),
        BlendMode::Multiply => Some(1),
        BlendMode::SoftLight => Some(2),
        BlendMode::PinLight => Some(3),
        _ => None,
    }
}

fn channel_index(channel: MaskChannel) -> u32 {
    match channel {
        MaskChannel::R => 0,
        MaskChannel::G => 1,
        MaskChannel::B => 2,
        MaskChannel::A => 3,
        MaskChannel::RG => 4,
        MaskChannel::RB => 5,
        MaskChannel::RA => 6,
        MaskChannel::GB => 7,
        MaskChannel::GA => 8,
        MaskChannel::BA => 9,
        MaskChannel::Rgb => 10,
        MaskChannel::Rga => 11,
        MaskChannel::Rba => 12,
        MaskChannel::Gba => 13,
        MaskChannel::Rgba => 14,
    }
}

pub const ROUTED_PFS: [u32; 14] = [
    8, 9, 100, 103, 105, 106, 107, 111, 115, 116, 117, 118, 30007, 90032,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MakeupRoute {
    Color {
        pf: u32,
        filter_type: u32,
        mask_channel: MaskChannel,
    },
    ImageMasked {
        pf: u32,
        filter_type: u32,
        mask_channel: MaskChannel,
    },
    ImageUnmasked {
        pf: u32,
        filter_type: u32,
    },
}

impl MakeupRoute {
    pub fn resolve(pf: u32) -> Result<Self, Refusal> {
        Ok(match pf {
            100 | 103 | 105 | 117 | 118 | 30007 => {
                let profile = color_mask_profile(pf)?;
                Self::Color {
                    pf,
                    filter_type: profile.filter_type,
                    mask_channel: profile.mask_channel,
                }
            }
            111 => Self::ImageUnmasked {
                pf,
                filter_type: 40,
            },
            _ => return Err(Refusal::UnknownEffect(pf)),
        })
    }

    /// Resolves a B3 row whose FilterType comes from the selected effect
    /// preset. PF107 can be either a Color or Image row; material presence is
    /// the recovered discriminator.
    pub fn resolve_same_job(
        pf: u32,
        filter_type: u32,
        has_material: bool,
    ) -> Result<Self, Refusal> {
        match pf {
            100 | 103 | 105 | 111 | 117 | 118 | 30007 => Self::resolve(pf),
            _ if filter_type == 0 => Err(Refusal::MissingInput("makeup filter type")),
            106 => Ok(Self::Color {
                pf,
                filter_type,
                mask_channel: MaskChannel::Rgba,
            }),
            107 if has_material => Ok(Self::ImageMasked {
                pf,
                filter_type,
                mask_channel: MaskChannel::R,
            }),
            107 => Ok(Self::Color {
                pf,
                filter_type,
                mask_channel: MaskChannel::R,
            }),
            8 | 9 | 115 | 116 => Ok(Self::ImageMasked {
                pf,
                filter_type,
                mask_channel: MaskChannel::Rgba,
            }),
            90032 => Ok(Self::ImageMasked {
                pf,
                filter_type,
                mask_channel: MaskChannel::Rgb,
            }),
            _ => Err(Refusal::UnknownEffect(pf)),
        }
    }

    pub fn pf(self) -> u32 {
        match self {
            Self::Color { pf, .. }
            | Self::ImageMasked { pf, .. }
            | Self::ImageUnmasked { pf, .. } => pf,
        }
    }

    pub fn filter_type(self) -> u32 {
        match self {
            Self::Color { filter_type, .. }
            | Self::ImageMasked { filter_type, .. }
            | Self::ImageUnmasked { filter_type, .. } => {
                filter_type
            }
        }
    }

    pub fn mask_channel(self) -> Option<MaskChannel> {
        match self {
            Self::Color { mask_channel, .. } | Self::ImageMasked { mask_channel, .. } => {
                Some(mask_channel)
            }
            Self::ImageUnmasked { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RouteInputs<'a> {
    pub mix_color: [u8; 4],
    pub opacity_percent: f32,
    pub degree01: f32,
    pub filter_type: u32,
    pub material: Option<&'a Image>,
    pub material_uv: Option<&'a [[f32; 2]]>,
    pub mask: Option<&'a Image>,
    pub mask_uv: Option<&'a [[f32; 2]]>,
}

fn route_layer(route: MakeupRoute, inputs: RouteInputs<'_>) -> Result<CpuMakeupLayer, Refusal> {
    if inputs.filter_type != route.filter_type() {
        return Err(Refusal::ValueOutOfRange("makeup FilterType"));
    }
    if !inputs.opacity_percent.is_finite() || !(0.0..=100.0).contains(&inputs.opacity_percent) {
        return Err(Refusal::ValueOutOfRange("opacity_percent"));
    }
    if !inputs.degree01.is_finite() || !(0.0..=1.0).contains(&inputs.degree01) {
        return Err(Refusal::ValueOutOfRange("degree01"));
    }
    Ok(CpuMakeupLayer {
        mix_color: inputs.mix_color,
        opacity_percent: inputs.opacity_percent,
        degree01: inputs.degree01,
        filter_type: route.filter_type(),
        mask_channel: route.mask_channel().unwrap_or(MaskChannel::A),
    })
}

/// Mirrors `apply_image_layer_dual`'s entry checks.
fn check(
    source: &Image,
    material: &Image,
    owner_mask: &Image,
    mask_uv: &[[f32; 2]],
    material_uv: &[[f32; 2]],
    params: ColorMaterialParams,
) -> Result<(usize, f32, u32), Refusal> {
    source.validate()?;
    material.validate()?;
    owner_mask.validate()?;
    let pixels = source.pixel_count();
    if mask_uv.len() != pixels || material_uv.len() != pixels {
        return Err(Refusal::ShapeMismatch("uv"));
    }
    if !params.opacity_percent.is_finite() || !(0.0..=100.0).contains(&params.opacity_percent) {
        return Err(Refusal::ValueOutOfRange("opacity_percent"));
    }
    if !params.degree01.is_finite() || !(0.0..=1.0).contains(&params.degree01) {
        return Err(Refusal::ValueOutOfRange("degree01"));
    }
    if !mask_uv
        .iter()
        .chain(material_uv)
        .all(|point| point[0].is_finite() && point[1].is_finite())
    {
        return Err(Refusal::ValueOutOfRange("uv"));
    }
    let mode = mode_index(params.blend_mode).ok_or(Refusal::NotImplemented(0x1_0000 + 1))?;
    let mix_alpha = params.mix_color[3] as f32 / 255.0 * params.opacity_percent / 100.0 * params.degree01;
    Ok((pixels, mix_alpha, mode))
}

#[cfg(windows)]
pub struct MakeupLayer {
    gpu: crate::cuda::Nvidia,
}

#[cfg(windows)]
impl MakeupLayer {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self { gpu: crate::cuda::Nvidia::open_ptx(PTX)? })
    }

    /// GPU twin of `makeup_material::apply_image_layer_dual`.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_image_layer_dual(
        &self,
        source: &Image,
        material: &Image,
        owner_mask: &Image,
        mask_uv: &[[f32; 2]],
        material_uv: &[[f32; 2]],
        params: ColorMaterialParams,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        let (_, mix_alpha, mode) = check(source, material, owner_mask, mask_uv, material_uv, params)
            .map_err(CudaError::Refused)?;
        if mix_alpha <= 0.0 {
            return Ok((source.clone(), Vec::new()));
        }
        let uniforms = Uniforms {
            width: source.width,
            height: source.height,
            stride: source.width * 4,
            material_width: material.width,
            material_height: material.height,
            mask_width: owner_mask.width,
            mask_height: owner_mask.height,
            mix_alpha,
            mode,
            mask_channel: channel_index(params.mask_channel),
            blend_alpha: u32::from(params.blend_alpha),
            reserved0: 0,
        };
        let grid = (
            source.width.div_ceil(BLOCK).max(1),
            source.height.div_ceil(BLOCK).max(1),
        );
        let (pixels, dispatch) = self.gpu.run_u8(
            KERNEL,
            grid,
            &[
                &source.pixels,
                &material.pixels,
                &owner_mask.pixels,
                uv_bytes(mask_uv),
                uv_bytes(material_uv),
            ],
            as_bytes(&uniforms),
        )?;
        let image = Image::new(source.width, source.height, pixels)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, vec![dispatch]))
    }

    /// Applies one of the seven recovered makeup rows through the shared
    /// image-layer kernel. Color rows use an opaque 1x1 material; PF111 uses a
    /// constant opaque alpha mask to express its fixed no-mask program.
    pub fn apply_route(
        &self,
        pf: u32,
        source: &Image,
        inputs: RouteInputs<'_>,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        let route =
            MakeupRoute::resolve_same_job(pf, inputs.filter_type, inputs.material.is_some())
                .map_err(CudaError::Refused)?;
        let layer = route_layer(route, inputs).map_err(CudaError::Refused)?;
        let params = layer.params_of().map_err(CudaError::Refused)?;
        match route {
            MakeupRoute::Color { .. } => {
                let mask = inputs
                    .mask
                    .ok_or(CudaError::Refused(Refusal::MissingInput("makeup mask")))?;
                let mask_uv = inputs.mask_uv.ok_or(CudaError::Refused(Refusal::MissingInput(
                    "makeup mask UV",
                )))?;
                let [r, g, b, _] = inputs.mix_color;
                let material = Image::solid(1, 1, [r, g, b, 255]);
                self.apply_image_layer_dual(
                    source, &material, mask, mask_uv, mask_uv, params,
                )
            }
            MakeupRoute::ImageMasked { .. } => {
                let material = inputs
                    .material
                    .ok_or(CudaError::Refused(Refusal::MissingInput("makeup material")))?;
                let mask = inputs
                    .mask
                    .ok_or(CudaError::Refused(Refusal::MissingInput("makeup mask")))?;
                let mask_uv = inputs.mask_uv.ok_or(CudaError::Refused(Refusal::MissingInput(
                    "makeup mask UV",
                )))?;
                let material_uv = inputs.material_uv.ok_or(CudaError::Refused(
                    Refusal::MissingInput("makeup material UV"),
                ))?;
                self.apply_image_layer_dual(source, material, mask, mask_uv, material_uv, params)
            }
            MakeupRoute::ImageUnmasked { .. } => {
                let material = inputs
                    .material
                    .ok_or(CudaError::Refused(Refusal::MissingInput("makeup material")))?;
                let material_uv = inputs.material_uv.ok_or(CudaError::Refused(
                    Refusal::MissingInput("makeup material UV"),
                ))?;
                if inputs.mask.is_some() || inputs.mask_uv.is_some() {
                    return Err(CudaError::Refused(Refusal::ValueOutOfRange(
                        "PF111 mask must be absent",
                    )));
                }
                let identity_mask = Image::solid(1, 1, [0, 0, 0, 255]);
                self.apply_image_layer_dual(
                    source,
                    material,
                    &identity_mask,
                    material_uv,
                    material_uv,
                    params,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_block_matches_the_c_header() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 48);
        assert_eq!(std::mem::align_of::<Uniforms>(), 4);
        assert_eq!(as_bytes(&Uniforms {
            width: 1,
            height: 1,
            stride: 4,
            material_width: 1,
            material_height: 1,
            mask_width: 1,
            mask_height: 1,
            mix_alpha: 0.0,
            mode: 0,
            mask_channel: 14,
            blend_alpha: 0,
            reserved0: 0,
        }).len(), 48);
    }

    #[test]
    fn only_the_four_recovered_modes_map() {
        assert_eq!(mode_index(BlendMode::Normal), Some(0));
        assert_eq!(mode_index(BlendMode::Multiply), Some(1));
        assert_eq!(mode_index(BlendMode::SoftLight), Some(2));
        assert_eq!(mode_index(BlendMode::PinLight), Some(3));
        assert_eq!(mode_index(BlendMode::Screen), None);
    }

    #[test]
    fn route_family_is_exact() {
        let expected = [
            (8, 42, Some(MaskChannel::Rgba)),
            (9, 42, Some(MaskChannel::Rgba)),
            (100, 11, Some(MaskChannel::RB)),
            (103, 11, Some(MaskChannel::RG)),
            (105, 11, Some(MaskChannel::RG)),
            (106, 11, Some(MaskChannel::Rgba)),
            (107, 42, Some(MaskChannel::R)),
            (111, 40, None),
            (115, 41, Some(MaskChannel::Rgba)),
            (116, 42, Some(MaskChannel::Rgba)),
            (117, 11, Some(MaskChannel::RB)),
            (118, 11, Some(MaskChannel::RG)),
            (30007, 11, Some(MaskChannel::RG)),
            (90032, 41, Some(MaskChannel::Rgb)),
        ];
        let actual = ROUTED_PFS.map(|pf| {
            let route = if matches!(pf, 100 | 103 | 105 | 111 | 117 | 118 | 30007) {
                MakeupRoute::resolve(pf).unwrap()
            } else {
                MakeupRoute::resolve_same_job(pf, expected.iter().find(|row| row.0 == pf).unwrap().1, pf != 106)
                    .unwrap()
            };
            (route.pf(), route.filter_type(), route.mask_channel())
        });
        assert_eq!(actual, expected);
        assert_eq!(MakeupRoute::resolve(106), Err(Refusal::UnknownEffect(106)));
        assert_eq!(MakeupRoute::resolve(90032), Err(Refusal::UnknownEffect(90032)));
    }
}
