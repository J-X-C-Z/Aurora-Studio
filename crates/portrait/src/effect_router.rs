//! Routing surface for the sixteen effect-level CUDA adapters.
//!
//! Product dispatch lives in `aurora_engine::portrait_stage`, which calls part
//! of the adapters in `cuda_*.rs` directly. This module gives all sixteen one
//! explicit selection surface. It is not called from product code yet, so it is
//! a library entry point, not an integration:
//!
//! * [`Backend::Cpu`] always runs the recovered CPU kernel;
//! * [`Backend::Cuda`] never falls back and never reports CPU as GPU;
//! * [`Backend::Auto`] tries CUDA first and records the exact CUDA error when
//!   it hands the frame to the CPU kernel.
//!
//! A CUDA result without a [`Dispatch`] is not accepted as evidence. The one
//! adapter that used to discard its dispatch (`PF30009` texture blend) now
//! exposes [`crate::cuda_texture_blend::Blend::apply_with_dispatch`].

use crate::cuda::{CudaError, Dispatch};
use crate::kernels::eye_bag;
use crate::kernels::facial_smooth;
use crate::kernels::makeup;
use crate::kernels::makeup_material::{self, ColorMaterialParams};
use crate::kernels::mesh_attribute::{self, DenseAttribute};
use crate::kernels::moisture::{self, FaceParams, Lut1D, MaskChannel, SkinMap};
use crate::kernels::skin_models;
use crate::kernels::smile_warp::{self, Rect};
use crate::kernels::teeth_blend;
use crate::kernels::texture_mask_blend::{self, NormalBlend};
use crate::kernels::uvmap_color::{self, Channel};
use crate::kernels::{beard, face_detail};
use crate::{Image, Refusal};

/// Caller-selected execution policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
    Auto,
}

/// Which lane actually produced [`Routed::output`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendUsed {
    Cpu,
    Cuda,
}

/// Why [`Backend::Auto`] used the CPU lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteNote {
    /// The CUDA adapter returned this error before producing pixels.
    Cuda(CudaError),
    /// The adapter returned success but no dispatch, so the call is not evidence.
    MissingDispatchEvidence(&'static str),
}

/// A routed frame plus the provenance the caller needs to report.
#[derive(Clone, Debug, PartialEq)]
pub struct Routed<T> {
    pub output: T,
    pub backend: BackendUsed,
    pub dispatches: Vec<Dispatch>,
    /// `Some` only when `backend == Cpu` and `Backend::Auto` had tried CUDA.
    pub fallback: Option<RouteNote>,
}

/// Typed failure for the router itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteError {
    Cpu(Refusal),
    Cuda(CudaError),
    MissingDispatchEvidence(&'static str),
}

impl From<Refusal> for RouteError {
    fn from(refusal: Refusal) -> Self {
        Self::Cpu(refusal)
    }
}

/// One effect adapter: its audited identity and both execution lanes.
pub trait CudaEffect {
    const NAME: &'static str;
    const PFS: &'static [u32];
    const KERNELS: &'static [&'static str];
    const CPU_ENTRY: &'static str;
    /// `false` only for an adapter that genuinely cannot report a dispatch.
    const DISPATCH_EVIDENCE: bool = true;

    type Input<'a>;
    type Output;

    fn cpu(input: &Self::Input<'_>) -> Result<Self::Output, Refusal>;
    fn cuda(input: &Self::Input<'_>) -> Result<(Self::Output, Vec<Dispatch>), CudaError>;
}

/// Registry row; the typed trait carries the callable input/output types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdapterMeta {
    pub name: &'static str,
    pub pfs: &'static [u32],
    pub kernels: &'static [&'static str],
    pub cpu_entry: &'static str,
    pub dispatch_evidence: bool,
}

macro_rules! meta {
    ($effect:ty) => {
        AdapterMeta {
            name: <$effect>::NAME,
            pfs: <$effect>::PFS,
            kernels: <$effect>::KERNELS,
            cpu_entry: <$effect>::CPU_ENTRY,
            dispatch_evidence: <$effect>::DISPATCH_EVIDENCE,
        }
    };
}

/// The sixteen audited effect adapters, in review order.
pub const ADAPTERS: &[AdapterMeta] = &[
    meta!(TeethWhiten),
    meta!(UvMap),
    meta!(Gloss),
    meta!(SkinTint),
    meta!(P15FacialSmooth),
    meta!(SkinAdjustLut),
    meta!(Beard),
    meta!(MoistureFace),
    meta!(MoistureBody),
    meta!(MakeupLayer),
    meta!(TeethBlend),
    meta!(TextureBlend),
    meta!(SmilePf100007),
    meta!(SmilePf8033),
    meta!(MeshAttribute),
    meta!(EyeBag),
];

pub fn adapter(name: &str) -> Option<&'static AdapterMeta> {
    ADAPTERS.iter().find(|meta| meta.name == name)
}

fn smile_refusal(refusal: smile_warp::Refusal) -> Refusal {
    match refusal {
        smile_warp::Refusal::MissingInput(name) => Refusal::MissingInput(name),
        smile_warp::Refusal::ShapeMismatch(name) => Refusal::ShapeMismatch(name),
        smile_warp::Refusal::ValueOutOfRange(name) => Refusal::ValueOutOfRange(name),
        smile_warp::Refusal::VendorConsumerOpen(pf) => Refusal::NotImplemented(pf),
    }
}

/// Runs `effect` under `policy` and reports which lane executed.
pub fn route<E: CudaEffect>(
    input: &E::Input<'_>,
    policy: Backend,
) -> Result<Routed<E::Output>, RouteError> {
    match policy {
        Backend::Cpu => cpu_only::<E>(input),
        Backend::Cuda => cuda_only::<E>(input),
        Backend::Auto => match cuda_only::<E>(input) {
            Ok(routed) => Ok(routed),
            Err(error) => {
                let mut routed = cpu_only::<E>(input)?;
                routed.fallback = Some(match error {
                    RouteError::Cuda(error) => RouteNote::Cuda(error),
                    RouteError::MissingDispatchEvidence(name) => {
                        RouteNote::MissingDispatchEvidence(name)
                    }
                    RouteError::Cpu(refusal) => return Err(RouteError::Cpu(refusal)),
                });
                Ok(routed)
            }
        },
    }
}

fn cpu_only<E: CudaEffect>(input: &E::Input<'_>) -> Result<Routed<E::Output>, RouteError> {
    Ok(Routed {
        output: E::cpu(input)?,
        backend: BackendUsed::Cpu,
        dispatches: Vec::new(),
        fallback: None,
    })
}

fn cuda_only<E: CudaEffect>(input: &E::Input<'_>) -> Result<Routed<E::Output>, RouteError> {
    let (output, dispatches) = E::cuda(input).map_err(RouteError::Cuda)?;
    if E::DISPATCH_EVIDENCE && dispatches.is_empty() {
        return Err(RouteError::MissingDispatchEvidence(E::NAME));
    }
    Ok(Routed {
        output,
        backend: BackendUsed::Cuda,
        dispatches,
        fallback: None,
    })
}

// ---------------------------------------------------------------------------
// PF108 / PF407 / PF8068
// ---------------------------------------------------------------------------

pub struct TeethWhiten;

pub struct TeethInput<'a> {
    pub source: &'a Image,
    pub mask_g: &'a [u8],
    pub whiten_lut: &'a [u8],
    pub bright_lut: &'a [u8],
    pub common_ui: f32,
    pub bright_ui: f32,
    pub deyellow_ui: f32,
}

impl CudaEffect for TeethWhiten {
    const NAME: &'static str = "teeth_whiten";
    const PFS: &'static [u32] = &[108];
    const KERNELS: &'static [&'static str] = &[crate::cuda_teeth::KERNEL];
    const CPU_ENTRY: &'static str = "kernels::face_detail::teeth_whitening";

    type Input<'a> = TeethInput<'a>;
    type Output = Image;

    fn cpu(input: &Self::Input<'_>) -> Result<Image, Refusal> {
        face_detail::teeth_whitening(
            input.source,
            input.mask_g,
            input.whiten_lut,
            input.bright_lut,
            input.common_ui,
            input.bright_ui,
            input.deyellow_ui,
        )
        .map(|result| result.output)
    }

    fn cuda(input: &Self::Input<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
        #[cfg(windows)]
        {
            let gpu = crate::cuda_teeth::Teeth::open()?;
            gpu.apply(
                input.source,
                input.mask_g,
                input.whiten_lut,
                input.bright_lut,
                input.common_ui,
                input.bright_ui,
                input.deyellow_ui,
            )
        }
        #[cfg(not(windows))]
        {
            let _ = input;
            Err(CudaError::Unavailable("NVIDIA CUDA host"))
        }
    }
}

pub struct UvMap;

pub struct UvMapInput<'a> {
    pub source: &'a Image,
    /// Normalized field taps. The CPU lane can only represent `0.0..=1.0`;
    /// values outside that range are refused instead of silently clamped.
    pub field: &'a [[f32; 4]],
    pub channel: Channel,
    pub catmull: Option<(f32, f32)>,
}

fn uv_field_image(width: u32, height: u32, field: &[[f32; 4]]) -> Result<Image, Refusal> {
    let pixels = width as usize * height as usize;
    if field.len() != pixels {
        return Err(Refusal::ShapeMismatch("uv field"));
    }
    let mut bytes = Vec::with_capacity(pixels * 4);
    for tap in field {
        for value in tap {
            if !value.is_finite() || !(0.0..=1.0).contains(value) {
                return Err(Refusal::ValueOutOfRange("uv field"));
            }
            bytes.push((value * 255.0).round_ties_even().clamp(0.0, 255.0) as u8);
        }
    }
    Image::new(width, height, bytes).ok_or(Refusal::ShapeMismatch("uv field"))
}

impl CudaEffect for UvMap {
    const NAME: &'static str = "uvmap_type143";
    const PFS: &'static [u32] = &[407];
    const KERNELS: &'static [&'static str] = &[crate::cuda_uvmap::KERNEL];
    const CPU_ENTRY: &'static str = "kernels::uvmap_color::apply";

    type Input<'a> = UvMapInput<'a>;
    type Output = Image;

    fn cpu(input: &Self::Input<'_>) -> Result<Image, Refusal> {
        let field = uv_field_image(input.source.width, input.source.height, input.field)?;
        uvmap_color::apply(input.source, &field, input.channel, input.catmull)
    }

    fn cuda(input: &Self::Input<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
        #[cfg(windows)]
        {
            let gpu = crate::cuda_uvmap::Uvmap::open()?;
            gpu.apply(input.source, input.field, input.channel, input.catmull)
        }
        #[cfg(not(windows))]
        {
            let _ = input;
            Err(CudaError::Unavailable("NVIDIA CUDA host"))
        }
    }
}

pub struct Gloss;

pub struct GlossInput<'a> {
    pub source: &'a Image,
    pub masks_r: [&'a [u8]; 3],
    pub extra_masks_r: [&'a [u8]; 3],
    pub native_percent: f32,
}

impl CudaEffect for Gloss {
    const NAME: &'static str = "gloss_pass";
    const PFS: &'static [u32] = &[8068];
    const KERNELS: &'static [&'static str] = &[crate::cuda_gloss::KERNEL];
    const CPU_ENTRY: &'static str = "kernels::skin_models::gloss_apply";

    type Input<'a> = GlossInput<'a>;
    type Output = Image;

    fn cpu(input: &Self::Input<'_>) -> Result<Image, Refusal> {
        skin_models::gloss_apply(
            input.source,
            input.masks_r,
            input.extra_masks_r,
            input.native_percent,
        )
    }

    fn cuda(input: &Self::Input<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
        #[cfg(windows)]
        {
            let gpu = crate::cuda_gloss::Gloss::open()?;
            gpu.apply(
                input.source,
                input.masks_r,
                input.extra_masks_r,
                input.native_percent,
            )
        }
        #[cfg(not(windows))]
        {
            let _ = input;
            Err(CudaError::Unavailable("NVIDIA CUDA host"))
        }
    }
}

// ---------------------------------------------------------------------------
// PF21015 / P15 / PF90005 / PF90007
// ---------------------------------------------------------------------------

pub struct SkinTint;

pub struct SkinTintInput<'a> {
    pub source: &'a Image,
    pub mask_r: &'a [u8],
    pub tint_green: &'a [u8],
    pub tint_magenta: &'a [u8],
    pub ui: f32,
}

impl CudaEffect for SkinTint {
    const NAME: &'static str = "skin_tint_pf21015";
    const PFS: &'static [u32] = &[21015];
    const KERNELS: &'static [&'static str] = &[crate::cuda_lut::KERNEL];
    const CPU_ENTRY: &'static str = "kernels::face_detail::skin_tint";

    type Input<'a> = SkinTintInput<'a>;
    type Output = Image;

    fn cpu(input: &Self::Input<'_>) -> Result<Image, Refusal> {
        face_detail::skin_tint(
            input.source,
            input.mask_r,
            input.tint_green,
            input.tint_magenta,
            input.ui,
        )
    }

    fn cuda(input: &Self::Input<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
        #[cfg(windows)]
        {
            let gpu = crate::cuda_lut::SkinLut::open()?;
            gpu.tint(
                input.source,
                input.mask_r,
                input.tint_green,
                input.tint_magenta,
                input.ui,
            )
            .map(|(image, dispatch)| (image, vec![dispatch]))
        }
        #[cfg(not(windows))]
        {
            let _ = input;
            Err(CudaError::Unavailable("NVIDIA CUDA host"))
        }
    }
}

pub struct P15FacialSmooth;

pub struct P15Input<'a> {
    pub source: &'a Image,
    pub base: &'a Image,
    pub residual: &'a Image,
    pub mask_r: &'a [u8],
    pub mask_b: &'a [u8],
    pub skin_mask: &'a [u8],
    pub pf: u32,
    pub pf_value: f32,
    pub uniforms: facial_smooth::Uniforms,
}

impl CudaEffect for P15FacialSmooth {
    const NAME: &'static str = "p15_facial_smooth";
    const PFS: &'static [u32] = &[90023, 90024, 21024];
    // The convenience entry dispatches the composite kernel. Surface, Gauss
    // and box blur remain separate `run_pass` calls in the same family.
    const KERNELS: &'static [&'static str] = &[crate::cuda::COMPOSITE];
    const CPU_ENTRY: &'static str = "kernels::facial_smooth::apply";

    type Input<'a> = P15Input<'a>;
    type Output = Image;

    fn cpu(input: &Self::Input<'_>) -> Result<Image, Refusal> {
        facial_smooth::apply(
            input.source,
            input.base,
            input.residual,
            input.mask_r,
            input.mask_b,
            input.skin_mask,
            input.uniforms,
        )
    }

    fn cuda(input: &Self::Input<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
        #[cfg(windows)]
        {
            let gpu = crate::cuda::Nvidia::open()?;
            let (image, dispatch) = gpu.facial_smooth(
                input.source,
                input.base,
                input.residual,
                input.mask_r,
                input.mask_b,
                input.skin_mask,
                input.pf,
                input.pf_value,
                &input.uniforms,
            )?;
            Ok((image, vec![dispatch]))
        }
        #[cfg(not(windows))]
        {
            let _ = input;
            Err(CudaError::Unavailable("NVIDIA CUDA host"))
        }
    }
}

pub struct SkinAdjustLut;

pub struct SkinLutInput<'a> {
    pub source: &'a Image,
    pub mask_r: &'a [u8],
    pub lut_rgb512: &'a [u8],
    pub mix: f32,
    pub pf: u32,
}

impl CudaEffect for SkinAdjustLut {
    const NAME: &'static str = "skin_adjust_lut";
    const PFS: &'static [u32] = &[90005, 90007];
    const KERNELS: &'static [&'static str] = &[crate::cuda_lut::KERNEL];
    const CPU_ENTRY: &'static str = "kernels::makeup::skin_adjust_lut";

    type Input<'a> = SkinLutInput<'a>;
    type Output = Image;

    fn cpu(input: &Self::Input<'_>) -> Result<Image, Refusal> {
        makeup::skin_adjust_lut(input.source, input.mask_r, input.lut_rgb512, input.mix)
    }

    fn cuda(input: &Self::Input<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
        #[cfg(windows)]
        {
            let gpu = crate::cuda_lut::SkinLut::open()?;
            gpu.skin_adjust(
                input.source,
                input.mask_r,
                input.lut_rgb512,
                input.mix,
                input.pf,
            )
            .map(|(image, dispatch)| (image, vec![dispatch]))
        }
        #[cfg(not(windows))]
        {
            let _ = input;
            Err(CudaError::Unavailable("NVIDIA CUDA host"))
        }
    }
}

// ---------------------------------------------------------------------------
// PF21026 / PF90002 / PF90010 / PF90033 / PF30026 / PF30009
// ---------------------------------------------------------------------------

pub struct Beard;

pub struct BeardInput<'a> {
    pub source: &'a Image,
    pub repair: &'a Image,
    pub mask_a: &'a [u8],
    pub mask_b: &'a [u8],
    pub strength: f32,
}

impl CudaEffect for Beard {
    const NAME: &'static str = "beard_blend";
    const PFS: &'static [u32] = &[21026];
    const KERNELS: &'static [&'static str] = &[crate::cuda_beard::KERNEL];
    const CPU_ENTRY: &'static str = "kernels::beard::apply";

    type Input<'a> = BeardInput<'a>;
    type Output = Image;

    fn cpu(input: &Self::Input<'_>) -> Result<Image, Refusal> {
        beard::apply(
            input.source,
            input.repair,
            input.mask_a,
            input.mask_b,
            input.strength,
        )
    }

    fn cuda(input: &Self::Input<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
        #[cfg(windows)]
        {
            let gpu = crate::cuda_beard::Beard::open()?;
            gpu.apply(
                input.source,
                input.repair,
                input.mask_a,
                input.mask_b,
                input.strength,
            )
        }
        #[cfg(not(windows))]
        {
            let _ = input;
            Err(CudaError::Unavailable("NVIDIA CUDA host"))
        }
    }
}

pub struct MoistureFace;

pub struct MoistureFaceInput<'a> {
    pub source: &'a Image,
    pub mask_rgba: &'a [u8],
    pub blur_y: &'a [u8],
    pub ui: f32,
    pub pivot: f32,
    pub gain: f32,
    pub channel: MaskChannel,
    pub skin_map_width: usize,
    pub skin_map_height: usize,
    pub skin_map: &'a [f32],
    pub hardlight: &'a [f32],
    pub beauty: &'a [f32],
}

impl CudaEffect for MoistureFace {
    const NAME: &'static str = "moisture_face";
    const PFS: &'static [u32] = &[90002];
    const KERNELS: &'static [&'static str] = &[crate::cuda_moisture::FACE_KERNEL];
    const CPU_ENTRY: &'static str = "kernels::moisture::face_apply";

    type Input<'a> = MoistureFaceInput<'a>;
    type Output = Image;

    fn cpu(input: &Self::Input<'_>) -> Result<Image, Refusal> {
        let params = FaceParams {
            ui: input.ui,
            pivot: input.pivot,
            gain: input.gain,
        };
        let skin_map = SkinMap {
            width: input.skin_map_width,
            height: input.skin_map_height,
            samples: input.skin_map,
        };
        moisture::face_apply(
            input.source,
            input.mask_rgba,
            input.blur_y,
            &params,
            input.channel,
            &skin_map,
            &Lut1D {
                samples: input.hardlight,
            },
            &Lut1D {
                samples: input.beauty,
            },
        )
    }

    fn cuda(input: &Self::Input<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
        #[cfg(windows)]
        {
            let params = FaceParams {
                ui: input.ui,
                pivot: input.pivot,
                gain: input.gain,
            };
            let skin_map = SkinMap {
                width: input.skin_map_width,
                height: input.skin_map_height,
                samples: input.skin_map,
            };
            let gpu = crate::cuda_moisture::Moisture::open()?;
            gpu.face_apply(
                input.source,
                input.mask_rgba,
                input.blur_y,
                &params,
                input.channel,
                &skin_map,
                &Lut1D {
                    samples: input.hardlight,
                },
                &Lut1D {
                    samples: input.beauty,
                },
            )
        }
        #[cfg(not(windows))]
        {
            let _ = input;
            Err(CudaError::Unavailable("NVIDIA CUDA host"))
        }
    }
}

pub struct MoistureBody;

pub struct MoistureBodyInput<'a> {
    pub source: &'a Image,
    pub weight_skin: &'a [f32],
    pub weight_dog: &'a [f32],
    pub beauty: &'a [f32],
    pub degree: f32,
    pub roi: (u32, u32, u32, u32),
}

impl CudaEffect for MoistureBody {
    const NAME: &'static str = "moisture_body";
    const PFS: &'static [u32] = &[90010];
    const KERNELS: &'static [&'static str] = &[crate::cuda_moisture::BODY_KERNEL];
    const CPU_ENTRY: &'static str = "kernels::moisture::body_apply";

    type Input<'a> = MoistureBodyInput<'a>;
    type Output = Image;

    fn cpu(input: &Self::Input<'_>) -> Result<Image, Refusal> {
        moisture::body_apply(
            input.source,
            input.weight_skin,
            input.weight_dog,
            &Lut1D {
                samples: input.beauty,
            },
            input.degree,
            input.roi,
        )
    }

    fn cuda(input: &Self::Input<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
        #[cfg(windows)]
        {
            let gpu = crate::cuda_moisture::Moisture::open()?;
            gpu.body_apply(
                input.source,
                input.weight_skin,
                input.weight_dog,
                &Lut1D {
                    samples: input.beauty,
                },
                input.degree,
                input.roi,
            )
        }
        #[cfg(not(windows))]
        {
            let _ = input;
            Err(CudaError::Unavailable("NVIDIA CUDA host"))
        }
    }
}

pub struct MakeupLayer;

pub struct MakeupInput<'a> {
    pub source: &'a Image,
    pub material: &'a Image,
    pub owner_mask: &'a Image,
    pub mask_uv: &'a [[f32; 2]],
    pub material_uv: &'a [[f32; 2]],
    pub params: ColorMaterialParams,
}

impl CudaEffect for MakeupLayer {
    const NAME: &'static str = "makeup_layer";
    const PFS: &'static [u32] = &[90033];
    const KERNELS: &'static [&'static str] = &[crate::cuda_makeup::KERNEL];
    const CPU_ENTRY: &'static str = "kernels::makeup_material::apply_image_layer_dual";

    type Input<'a> = MakeupInput<'a>;
    type Output = Image;

    fn cpu(input: &Self::Input<'_>) -> Result<Image, Refusal> {
        makeup_material::apply_image_layer_dual(
            input.source,
            input.material,
            input.owner_mask,
            input.mask_uv,
            input.material_uv,
            input.params,
        )
    }

    fn cuda(input: &Self::Input<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
        #[cfg(windows)]
        {
            let gpu = crate::cuda_makeup::MakeupLayer::open()?;
            gpu.apply_image_layer_dual(
                input.source,
                input.material,
                input.owner_mask,
                input.mask_uv,
                input.material_uv,
                input.params,
            )
        }
        #[cfg(not(windows))]
        {
            let _ = input;
            Err(CudaError::Unavailable("NVIDIA CUDA host"))
        }
    }
}

pub struct TeethBlend;

pub struct TeethBlendInput<'a> {
    pub original: &'a Image,
    pub effect: &'a Image,
    pub mask: &'a [u8],
}

impl CudaEffect for TeethBlend {
    const NAME: &'static str = "teeth_blend";
    const PFS: &'static [u32] = &[30026];
    const KERNELS: &'static [&'static str] = &[crate::cuda_teeth_blend::KERNEL];
    const CPU_ENTRY: &'static str = "kernels::teeth_blend::apply";

    type Input<'a> = TeethBlendInput<'a>;
    type Output = Image;

    fn cpu(input: &Self::Input<'_>) -> Result<Image, Refusal> {
        teeth_blend::apply(input.original, input.effect, input.mask)
    }

    fn cuda(input: &Self::Input<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
        #[cfg(windows)]
        {
            let gpu = crate::cuda_teeth_blend::TeethBlend::open()?;
            gpu.apply(input.original, input.effect, input.mask)
        }
        #[cfg(not(windows))]
        {
            let _ = input;
            Err(CudaError::Unavailable("NVIDIA CUDA host"))
        }
    }
}

pub struct TextureBlend;

pub struct TextureBlendInput<'a> {
    pub original: &'a Image,
    pub effect: &'a Image,
    pub mask: &'a [u8],
    pub effect_uv: Option<&'a [[f32; 2]]>,
    pub params: NormalBlend,
}

impl CudaEffect for TextureBlend {
    const NAME: &'static str = "texture_blend";
    const PFS: &'static [u32] = &[30009];
    const KERNELS: &'static [&'static str] = &[crate::cuda_texture_blend::KERNEL];
    const CPU_ENTRY: &'static str = "kernels::texture_mask_blend::apply";

    type Input<'a> = TextureBlendInput<'a>;
    type Output = Image;

    fn cpu(input: &Self::Input<'_>) -> Result<Image, Refusal> {
        texture_mask_blend::apply(
            input.original,
            input.effect,
            input.mask,
            input.effect_uv,
            input.params,
        )
    }

    fn cuda(input: &Self::Input<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
        #[cfg(windows)]
        {
            let gpu = crate::cuda_texture_blend::Blend::open()?;
            gpu.apply_with_dispatch(
                input.original,
                input.effect,
                input.mask,
                input.effect_uv,
                input.params,
            )
            .map(|(image, dispatch)| (image, vec![dispatch]))
        }
        #[cfg(not(windows))]
        {
            let _ = input;
            Err(CudaError::Unavailable("NVIDIA CUDA host"))
        }
    }
}

// ---------------------------------------------------------------------------
// PF100007 / PF8033 / PF90116
// ---------------------------------------------------------------------------

pub struct SmilePf100007;

pub struct SmileInput<'a> {
    pub source_bgra: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub degree: f32,
    pub lip_mask: &'a [u8],
    pub mouth_mask: &'a [u8],
    pub model_dx: &'a [f32],
    pub model_dy: &'a [f32],
    pub roi: Rect,
}

impl CudaEffect for SmilePf100007 {
    const NAME: &'static str = "smile_pf100007";
    const PFS: &'static [u32] = &[100007];
    const KERNELS: &'static [&'static str] = &[crate::cuda_smile::KERNEL];
    const CPU_ENTRY: &'static str = "kernels::smile_warp::pf100007_apply";

    type Input<'a> = SmileInput<'a>;
    type Output = Vec<u8>;

    fn cpu(input: &Self::Input<'_>) -> Result<Vec<u8>, Refusal> {
        smile_warp::pf100007_apply(
            input.source_bgra,
            input.width,
            input.height,
            input.degree,
            input.lip_mask,
            input.mouth_mask,
            input.model_dx,
            input.model_dy,
            input.roi,
        )
        .map_err(smile_refusal)
    }

    fn cuda(input: &Self::Input<'_>) -> Result<(Vec<u8>, Vec<Dispatch>), CudaError> {
        #[cfg(windows)]
        {
            let gpu = crate::cuda_smile::Smile::open()?;
            gpu.apply(
                input.source_bgra,
                input.width,
                input.height,
                input.degree,
                input.lip_mask,
                input.mouth_mask,
                input.model_dx,
                input.model_dy,
                input.roi,
            )
        }
        #[cfg(not(windows))]
        {
            let _ = input;
            Err(CudaError::Unavailable("NVIDIA CUDA host"))
        }
    }
}

pub struct SmilePf8033;

pub struct SmileMaskInput<'a> {
    pub source_bgra: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub dx: &'a [f32],
    pub dy: &'a [f32],
    pub mask: &'a [u8],
    pub degree01: f32,
}

impl CudaEffect for SmilePf8033 {
    const NAME: &'static str = "smile_pf8033";
    const PFS: &'static [u32] = &[8033];
    const KERNELS: &'static [&'static str] = &[crate::cuda_smile::MASK_KERNEL];
    const CPU_ENTRY: &'static str = "kernels::smile_warp::pf8033_dense_warp";

    type Input<'a> = SmileMaskInput<'a>;
    type Output = Vec<u8>;

    fn cpu(input: &Self::Input<'_>) -> Result<Vec<u8>, Refusal> {
        smile_warp::pf8033_dense_warp(
            input.source_bgra,
            input.width,
            input.height,
            input.dx,
            input.dy,
            input.mask,
            input.degree01,
        )
        .map_err(smile_refusal)
    }

    fn cuda(input: &Self::Input<'_>) -> Result<(Vec<u8>, Vec<Dispatch>), CudaError> {
        #[cfg(windows)]
        {
            let gpu = crate::cuda_smile::Smile::open()?;
            gpu.apply_pf8033(
                input.source_bgra,
                input.width,
                input.height,
                input.dx,
                input.dy,
                input.mask,
                input.degree01,
            )
        }
        #[cfg(not(windows))]
        {
            let _ = input;
            Err(CudaError::Unavailable("NVIDIA CUDA host"))
        }
    }
}

pub struct EyeBag;

pub struct EyeBagInput<'a> {
    pub source: &'a Image,
    pub color3: &'a [f32],
    pub response: &'a [f32],
    pub degree: f32,
}

impl CudaEffect for EyeBag {
    const NAME: &'static str = "eye_bag_compose";
    const PFS: &'static [u32] = &[90116];
    const KERNELS: &'static [&'static str] = &[crate::cuda_eye_bag::KERNEL];
    const CPU_ENTRY: &'static str = "kernels::eye_bag::compose_field_image_pf90116";

    type Input<'a> = EyeBagInput<'a>;
    type Output = Image;

    fn cpu(input: &Self::Input<'_>) -> Result<Image, Refusal> {
        eye_bag::compose_field_image_pf90116(
            input.source,
            input.color3,
            input.response,
            input.degree,
        )
    }

    fn cuda(input: &Self::Input<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
        #[cfg(windows)]
        {
            let gpu = crate::cuda_eye_bag::EyeBag::open()?;
            gpu.compose_field(input.source, input.color3, input.response, input.degree)
        }
        #[cfg(not(windows))]
        {
            let _ = input;
            Err(CudaError::Unavailable("NVIDIA CUDA host"))
        }
    }
}

pub struct MeshAttribute;

pub struct MeshAttributeInput<'a> {
    pub width: u32,
    pub height: u32,
    pub dest: &'a [[f32; 2]],
    pub attr: &'a [[f32; 2]],
    pub faces: &'a [u32],
}

impl CudaEffect for MeshAttribute {
    const NAME: &'static str = "mesh_attribute";
    const PFS: &'static [u32] = &[8033];
    const KERNELS: &'static [&'static str] = &[
        crate::cuda_mesh_attribute::SELECT_KERNEL,
        crate::cuda_mesh_attribute::RESOLVE_KERNEL,
    ];
    const CPU_ENTRY: &'static str = "kernels::mesh_attribute::rasterize";

    type Input<'a> = MeshAttributeInput<'a>;
    type Output = DenseAttribute;

    fn cpu(input: &Self::Input<'_>) -> Result<DenseAttribute, Refusal> {
        mesh_attribute::rasterize(
            input.width,
            input.height,
            input.dest,
            input.attr,
            input.faces,
        )
    }

    fn cuda(input: &Self::Input<'_>) -> Result<(DenseAttribute, Vec<Dispatch>), CudaError> {
        #[cfg(windows)]
        {
            let gpu = crate::cuda_mesh_attribute::MeshAttribute::open()?;
            gpu.rasterize(
                input.width,
                input.height,
                input.dest,
                input.attr,
                input.faces,
            )
        }
        #[cfg(not(windows))]
        {
            let _ = input;
            Err(CudaError::Unavailable("NVIDIA CUDA host"))
        }
    }
}
