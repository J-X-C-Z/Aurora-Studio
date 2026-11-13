//! Apple Metal host for the recovered P15 facial-smooth and O3B masked-whiten
//! kernels.
//!
//! The MSL is reused byte-for-byte from
//! `o3d-p15-metal-facial-smooth` and `o3b-apple-masked-whiten-integration`.
//! This module owns only the Rust host side: 88/48/80-byte uniform layouts,
//! float32 RGBA / one-byte mask planes, MTLBuffer binding indexes, dispatch
//! grid, synchronous completion and readback.
//!
//! It does not build a `CAMetalLayer`, touch C1, or probe the OS to guess a GPU.
//! On a non-macOS host every execution entry point returns
//! `MetalRefusal::HostUnavailable`; no CPU fallback is silently substituted.
//!
//! MSL samples `device float*` planes directly. There are no `texture2d`
//! bindings in either reused kernel, so the host texture layout is empty by
//! design; the buffer list below is the complete binding contract.
//!
//! `facial_smooth` returns unquantized float32 RGBA, so a macOS run can be
//! compared pixel-for-pixel against
//! `kernels::facial_smooth::composite_pixel` without an RGBA8 round trip.

use std::cell::OnceCell;
use std::ffi::{c_char, c_void, CString};

use crate::{metal_mesh, metal_smile};
use crate::{Image, Refusal as CrateRefusal};

/// P15 facial-smooth MSL, copied unchanged.
const P15_MSL: &str = include_str!("../native/metal/P15FacialSmooth.metal");
/// O3B masked whiten + coverage-max projection MSL, copied unchanged.
const WHITEN_MSL: &str = include_str!("../native/metal/o3b_masked_lut_whiten.metal");

/// P15 `pfValue` gate: strictly greater, finite, to be active.
pub const ACTIVE_GATE: f32 = 0.001;
/// ParamFlags pinned to `FacialSmoothProcessV2` rows 36-38.
pub const SUPPORTED_PF: [u32; 3] = [90023, 90024, 21024];
/// Recovered P15 material constants.
pub const SURFACE_RADIUS: f32 = 4.93;
pub const SURFACE_THRESHOLD: f32 = 10.45;
pub const GAUSS_OFFSET: f32 = 1.45;
pub const ROI_SIZE: u32 = 1000;
/// 512x512 RGBA float atlas: 8x8 tiles of 64x64.
pub const LUT_FLOATS: usize = 512 * 512 * 4;
pub const LUT_BYTES: usize = LUT_FLOATS * 4;

/// The ten product uniforms. No test-only "neutral" values are exposed: a real
/// caller must supply live product values.
pub const PRODUCT_UNIFORM_NAMES: [&str; 10] = [
    "legal_alpha",
    "dark_circle_alpha",
    "diff_factor",
    "diff_factor2",
    "color_threshold_factor",
    "mix_alpha",
    "diff_min_threshold",
    "diff_max_threshold",
    "high_freq_min_threshold",
    "high_freq_max_threshold",
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProductUniforms {
    pub legal_alpha: f32,
    pub dark_circle_alpha: f32,
    pub diff_factor: f32,
    pub diff_factor2: f32,
    pub color_threshold_factor: f32,
    pub mix_alpha: f32,
    pub diff_min_threshold: f32,
    pub diff_max_threshold: f32,
    pub high_freq_min_threshold: f32,
    pub high_freq_max_threshold: f32,
}

impl ProductUniforms {
    pub fn values(self) -> [f32; 10] {
        [
            self.legal_alpha,
            self.dark_circle_alpha,
            self.diff_factor,
            self.diff_factor2,
            self.color_threshold_factor,
            self.mix_alpha,
            self.diff_min_threshold,
            self.diff_max_threshold,
            self.high_freq_min_threshold,
            self.high_freq_max_threshold,
        ]
    }

    pub fn validate(self) -> MetalResult<()> {
        if !self.values().iter().all(|value| value.is_finite())
            || self.diff_min_threshold > self.diff_max_threshold
            || self.high_freq_min_threshold > self.high_freq_max_threshold
        {
            return Err(MetalRefusal::ValueOutOfRange("uniforms"));
        }
        Ok(())
    }
}

/// Exact 88-byte mirror of `P15FacialSmoothUniforms`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct P15Uniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub pf: u32,
    pub pf_value: f32,
    pub gate: f32,
    pub legal_alpha: f32,
    pub dark_circle_alpha: f32,
    pub diff_factor: f32,
    pub diff_factor2: f32,
    pub color_threshold_factor: f32,
    pub mix_alpha: f32,
    pub diff_min_threshold: f32,
    pub diff_max_threshold: f32,
    pub high_frequency_min_threshold: f32,
    pub high_frequency_max_threshold: f32,
    pub surface_radius: f32,
    pub surface_threshold: f32,
    pub gauss_offset: f32,
    pub roi_size: u32,
    pub algo_type: u32,
    pub box_blur_stage: u32,
}

const _: () = assert!(std::mem::size_of::<P15Uniforms>() == 88);

/// Exact 48-byte mirror of `O3MaskedWhitenParams`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct WhitenParams {
    pub surface_width: u32,
    pub surface_height: u32,
    pub mask_x: u32,
    pub mask_y: u32,
    pub mask_width: u32,
    pub mask_height: u32,
    pub pixel_count: u32,
    pub flags: u32,
    pub alpha: f32,
    pub reserved0: f32,
    pub reserved1: f32,
    pub reserved2: f32,
}

const _: () = assert!(std::mem::size_of::<WhitenParams>() == 48);

/// Exact 80-byte mirror of `O3MaskProjectionParams`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ProjectionParams {
    pub source_mask_width: i32,
    pub source_mask_height: i32,
    pub surface_width: i32,
    pub surface_height: i32,
    pub destination_x: i32,
    pub destination_y: i32,
    pub destination_width: i32,
    pub destination_height: i32,
    pub source_rect_x: i32,
    pub source_rect_y: i32,
    pub source_rect_width: i32,
    pub source_rect_height: i32,
    pub domain_x0: i32,
    pub domain_y0: i32,
    pub roi_width: i32,
    pub roi_height: i32,
    pub flags: u32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub reserved2: u32,
}

const _: () = assert!(std::mem::size_of::<ProjectionParams>() == 80);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub index: u32,
    pub name: &'static str,
    pub element: &'static str,
    pub element_bytes: u32,
}

pub const P15_COMPOSITE_BINDINGS: [Binding; 7] = [
    Binding {
        index: 0,
        name: "source",
        element: "float32 RGBA",
        element_bytes: 4,
    },
    Binding {
        index: 1,
        name: "mask_rgba",
        element: "float32 RGBA (r,0,b,0)",
        element_bytes: 4,
    },
    Binding {
        index: 2,
        name: "skin_mask",
        element: "float32 R",
        element_bytes: 4,
    },
    Binding {
        index: 3,
        name: "base",
        element: "float32 RGBA",
        element_bytes: 4,
    },
    Binding {
        index: 4,
        name: "residual",
        element: "float32 RGBA",
        element_bytes: 4,
    },
    Binding {
        index: 5,
        name: "output",
        element: "float32 RGBA",
        element_bytes: 4,
    },
    Binding {
        index: 6,
        name: "uniforms",
        element: "P15Uniforms",
        element_bytes: 88,
    },
];

pub const P15_BLUR_BINDINGS: [Binding; 3] = [
    Binding {
        index: 0,
        name: "source",
        element: "float32 RGBA",
        element_bytes: 4,
    },
    Binding {
        index: 1,
        name: "output",
        element: "float32 RGBA",
        element_bytes: 4,
    },
    Binding {
        index: 2,
        name: "uniforms",
        element: "P15Uniforms",
        element_bytes: 88,
    },
];

pub const WHITEN_BINDINGS: [Binding; 5] = [
    Binding {
        index: 0,
        name: "source_bgra",
        element: "uint8 BGRA",
        element_bytes: 1,
    },
    Binding {
        index: 1,
        name: "output_bgra",
        element: "uint8 BGRA",
        element_bytes: 1,
    },
    Binding {
        index: 2,
        name: "mask_plane",
        element: "uint8 R",
        element_bytes: 1,
    },
    Binding {
        index: 3,
        name: "lut_atlas",
        element: "float32 RGBA",
        element_bytes: 4,
    },
    Binding {
        index: 4,
        name: "params",
        element: "WhitenParams",
        element_bytes: 48,
    },
];

pub const PROJECTION_BINDINGS: [Binding; 3] = [
    Binding {
        index: 0,
        name: "source_mask",
        element: "uint8 R",
        element_bytes: 1,
    },
    Binding {
        index: 1,
        name: "destination_mask",
        element: "uint8 R",
        element_bytes: 1,
    },
    Binding {
        index: 2,
        name: "params",
        element: "ProjectionParams",
        element_bytes: 80,
    },
];

/// The reused kernels bind no textures and no samplers.
pub const TEXTURE_BINDINGS: &[&str] = &[];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalRefusal {
    MissingInput(&'static str),
    ShapeMismatch(&'static str),
    ValueOutOfRange(&'static str),
    UnsupportedPf(u32),
    /// Vendor mesh contract rejected with the missing capability details.
    UnsupportedMeshContract(metal_mesh::MeshContractRefusal),
    Inactive,
    MissingSameJobIdentity,
    HostUnavailable,
    CompileFailed(&'static str),
    AllocationFailed(&'static str),
    CommandFailed,
}

pub type MetalResult<T> = Result<T, MetalRefusal>;

#[derive(Clone, Copy)]
pub struct SmoothInput<'a> {
    pub source: &'a [f32],
    pub mask_rgba: &'a [f32],
    pub skin_mask: &'a [f32],
    pub base: &'a [f32],
    pub residual: &'a [f32],
    pub width: u32,
    pub height: u32,
    /// Row pitch in pixels; must be >= `width`.
    pub stride: u32,
    pub pf: u32,
    pub pf_value: f32,
    /// Same-job ticket; zero means the dispatch has no identity.
    pub ticket: u64,
    pub uniforms: ProductUniforms,
}

impl SmoothInput<'_> {
    pub fn validate(&self) -> MetalResult<()> {
        let rgba = plane_len(self.width, self.height, self.stride, 4)?;
        let scalar = plane_len(self.width, self.height, self.stride, 1)?;
        if self.source.len() < rgba {
            return Err(MetalRefusal::ShapeMismatch("source"));
        }
        if self.base.len() < rgba {
            return Err(MetalRefusal::ShapeMismatch("base"));
        }
        if self.residual.len() < rgba {
            return Err(MetalRefusal::ShapeMismatch("residual"));
        }
        if self.mask_rgba.len() < rgba {
            return Err(MetalRefusal::ShapeMismatch("mask_rgba"));
        }
        if self.skin_mask.len() < scalar {
            return Err(MetalRefusal::ShapeMismatch("skin_mask"));
        }
        if !self.source[..rgba].iter().all(|v| v.is_finite())
            || !self.base[..rgba].iter().all(|v| v.is_finite())
            || !self.residual[..rgba].iter().all(|v| v.is_finite())
        {
            return Err(MetalRefusal::ValueOutOfRange("pixel"));
        }
        if !self.mask_rgba[..rgba]
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            || !self.skin_mask[..scalar]
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        {
            return Err(MetalRefusal::ValueOutOfRange("mask"));
        }
        self.uniforms.validate()?;
        check_gate(self.pf, self.pf_value)?;
        if self.ticket == 0 {
            return Err(MetalRefusal::MissingSameJobIdentity);
        }
        Ok(())
    }

    fn msl_uniforms(&self) -> P15Uniforms {
        P15Uniforms {
            width: self.width,
            height: self.height,
            stride: self.stride,
            pf: self.pf,
            pf_value: self.pf_value,
            gate: ACTIVE_GATE,
            legal_alpha: self.uniforms.legal_alpha,
            dark_circle_alpha: self.uniforms.dark_circle_alpha,
            diff_factor: self.uniforms.diff_factor,
            diff_factor2: self.uniforms.diff_factor2,
            color_threshold_factor: self.uniforms.color_threshold_factor,
            mix_alpha: self.uniforms.mix_alpha,
            diff_min_threshold: self.uniforms.diff_min_threshold,
            diff_max_threshold: self.uniforms.diff_max_threshold,
            high_frequency_min_threshold: self.uniforms.high_freq_min_threshold,
            high_frequency_max_threshold: self.uniforms.high_freq_max_threshold,
            surface_radius: SURFACE_RADIUS,
            surface_threshold: SURFACE_THRESHOLD,
            gauss_offset: GAUSS_OFFSET,
            roi_size: ROI_SIZE,
            algo_type: 0,
            box_blur_stage: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlurKernel {
    Surface,
    Gauss,
    Box,
}

impl BlurKernel {
    fn entry_point(self) -> &'static str {
        match self {
            BlurKernel::Surface => "p15_surface_pass",
            BlurKernel::Gauss => "p15_gauss_pass",
            BlurKernel::Box => "p15_surface_blur_pass",
        }
    }
}

#[derive(Clone, Copy)]
pub struct WhitenInput<'a> {
    /// BGRA8 source. Rows may be padded; only `stride_bytes` is read.
    pub bgra: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub stride_bytes: u32,
    pub mask: &'a [u8],
    pub mask_x: u32,
    pub mask_y: u32,
    pub mask_width: u32,
    pub mask_height: u32,
    /// 512x512 RGBA float LUT, `LUT_FLOATS` values.
    pub lut: &'a [f32],
    pub alpha: f32,
    pub flags: u32,
    pub ticket: u64,
}

impl WhitenInput<'_> {
    pub fn validate(&self) -> MetalResult<()> {
        if self.width == 0 || self.height == 0 {
            return Err(MetalRefusal::ShapeMismatch("surface"));
        }
        let row_bytes = (self.width as usize)
            .checked_mul(4)
            .ok_or(MetalRefusal::ShapeMismatch("surface"))?;
        if (self.stride_bytes as usize) < row_bytes
            || self.bgra.len()
                < (self.stride_bytes as usize)
                    .checked_mul(self.height as usize)
                    .ok_or(MetalRefusal::ShapeMismatch("surface"))?
        {
            return Err(MetalRefusal::ShapeMismatch("source_bgra"));
        }
        if self.mask_width == 0 || self.mask_height == 0 {
            return Err(MetalRefusal::ShapeMismatch("mask"));
        }
        if self.mask.len()
            != (self.mask_width as usize)
                .checked_mul(self.mask_height as usize)
                .ok_or(MetalRefusal::ShapeMismatch("mask"))?
        {
            return Err(MetalRefusal::ShapeMismatch("mask"));
        }
        if self.mask_x as u64 + self.mask_width as u64 > self.width as u64
            || self.mask_y as u64 + self.mask_height as u64 > self.height as u64
        {
            return Err(MetalRefusal::ShapeMismatch("mask_rect"));
        }
        if (self.width as u64) * (self.height as u64) > u32::MAX as u64 {
            return Err(MetalRefusal::ShapeMismatch("surface"));
        }
        if self.lut.len() != LUT_FLOATS {
            return Err(MetalRefusal::ShapeMismatch("lut"));
        }
        if !self.lut.iter().all(|value| value.is_finite()) {
            return Err(MetalRefusal::ValueOutOfRange("lut"));
        }
        if !self.alpha.is_finite() || !(0.0..=1.0).contains(&self.alpha) {
            return Err(MetalRefusal::ValueOutOfRange("alpha"));
        }
        if self.ticket == 0 {
            return Err(MetalRefusal::MissingSameJobIdentity);
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct ProjectionInput<'a> {
    pub source_mask: &'a [u8],
    pub source_mask_width: u32,
    pub source_mask_height: u32,
    pub surface_width: u32,
    pub surface_height: u32,
    pub destination_x: u32,
    pub destination_y: u32,
    pub destination_width: u32,
    pub destination_height: u32,
    pub source_rect_x: u32,
    pub source_rect_y: u32,
    pub source_rect_width: u32,
    pub source_rect_height: u32,
    pub domain_x0: i32,
    pub domain_y0: i32,
    pub roi_width: u32,
    pub roi_height: u32,
    pub ticket: u64,
}

impl ProjectionInput<'_> {
    pub fn validate(&self) -> MetalResult<()> {
        if self.source_mask_width == 0
            || self.source_mask_height == 0
            || self.destination_width == 0
            || self.destination_height == 0
            || self.roi_width == 0
            || self.roi_height == 0
        {
            return Err(MetalRefusal::ShapeMismatch("projection"));
        }
        if [
            self.source_mask_width,
            self.source_mask_height,
            self.surface_width,
            self.surface_height,
            self.destination_width,
            self.destination_height,
            self.roi_width,
            self.roi_height,
        ]
        .iter()
        .any(|value| *value > i32::MAX as u32)
        {
            return Err(MetalRefusal::ShapeMismatch("projection"));
        }
        if self.source_mask.len()
            != (self.source_mask_width as usize)
                .checked_mul(self.source_mask_height as usize)
                .ok_or(MetalRefusal::ShapeMismatch("source_mask"))?
        {
            return Err(MetalRefusal::ShapeMismatch("source_mask"));
        }
        if self.source_rect_x as u64 + self.source_rect_width as u64 > self.source_mask_width as u64
            || self.source_rect_y as u64 + self.source_rect_height as u64
                > self.source_mask_height as u64
            || self.destination_x as u64 + self.destination_width as u64 > self.surface_width as u64
            || self.destination_y as u64 + self.destination_height as u64
                > self.surface_height as u64
        {
            return Err(MetalRefusal::ShapeMismatch("projection_rect"));
        }
        if self.ticket == 0 {
            return Err(MetalRefusal::MissingSameJobIdentity);
        }
        Ok(())
    }

    fn params(&self) -> ProjectionParams {
        ProjectionParams {
            source_mask_width: self.source_mask_width as i32,
            source_mask_height: self.source_mask_height as i32,
            surface_width: self.surface_width as i32,
            surface_height: self.surface_height as i32,
            destination_x: self.destination_x as i32,
            destination_y: self.destination_y as i32,
            destination_width: self.destination_width as i32,
            destination_height: self.destination_height as i32,
            source_rect_x: self.source_rect_x as i32,
            source_rect_y: self.source_rect_y as i32,
            source_rect_width: self.source_rect_width as i32,
            source_rect_height: self.source_rect_height as i32,
            domain_x0: self.domain_x0,
            domain_y0: self.domain_y0,
            roi_width: self.roi_width as i32,
            roi_height: self.roi_height as i32,
            flags: 0,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
        }
    }
}

/// Compile the reused libraries once and run synchronous compute work.
///
/// The context is not `Send`: it owns MTLDevice/MTLCommandQueue objects and
/// must be created, used and dropped on the same thread.
pub struct MetalContext {
    lib: Lib,
    device: Objc,
    queue: Objc,
    p15: Vec<(&'static str, Objc)>,
    whiten: Vec<(&'static str, Objc)>,
    // Optional mesh pipelines compile on first mesh call; a failure here must
    // not block P15/whiten initialization.
    mesh: OnceCell<Result<MeshPrograms, MetalRefusal>>,
    // Additive perspective mesh pipeline; independent of the affine `mesh`.
    mesh_perspective: OnceCell<Result<MeshPrograms, MetalRefusal>>,
    // Optional smile pipeline; lazy for the same reason.
    smile: OnceCell<Result<Objc, MetalRefusal>>,
    // Optional fixed-point skin LUT pipeline.
    skin_lut: OnceCell<Result<Objc, MetalRefusal>>,
}

enum MeshKind {
    Atomic,
    Serial,
}

struct MeshPrograms {
    kind: MeshKind,
    pipelines: Vec<(&'static str, Objc)>,
}

const P15_ENTRY_POINTS: [&str; 6] = [
    "p15_facial_smooth_composite",
    "p15_surface_blur_pass",
    "p15_gauss_pass",
    "p15_surface_pass",
    "p15_neck_repair_tails",
    "p15_neck_max_pool",
];

const WHITEN_ENTRY_POINTS: [&str; 2] = ["o3_coverage_max_mask_projection", "o3_masked_lut_whiten"];
const MESH_ENTRY_POINTS: [&str; 2] = ["mesh_raster_select", "mesh_raster_emit"];
const MESH_PERSPECTIVE_ENTRY_POINTS: [&str; 2] =
    ["mesh_raster_select", "mesh_raster_emit_perspective"];
const MESH_SERIAL_MSL: &str = include_str!("../native/metal/mesh_raster_serial.metal");
const MESH_SERIAL_ENTRY_POINTS: [&str; 1] = ["mesh_raster_serial"];
const MESH_PERSPECTIVE_SERIAL_ENTRY_POINTS: [&str; 1] = ["mesh_raster_serial_perspective"];
const SMILE_ENTRY_POINTS: [&str; 1] = ["pf100007_smile"];
const SKIN_LUT_MSL: &str = include_str!("../native/metal/skin_lut_fixed.metal");
const SKIN_LUT_ENTRY_POINTS: [&str; 1] = ["skin_lut_fixed"];

fn selected_fill(overlap: i32, pixel_count: usize) -> Vec<u8> {
    let fill = if overlap == metal_mesh::OverlapCode::Last as i32 { 0 } else { 0xFF };
    vec![fill; pixel_count * 4]
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkinLutParams {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub mask_row_bytes: u32,
    pub mix: f32,
    pub gate: f32,
    pub lut_width: u32,
    pub lut_height: u32,
}

const _: () = assert!(std::mem::size_of::<SkinLutParams>() == 32);

/// Framework loading only. This does not prove a default device exists.
pub fn framework_available() -> bool {
    cfg!(target_os = "macos") && lib().is_some()
}

/// Runtime Metal device availability: frameworks load and
/// `MTLCreateSystemDefaultDevice` returns a non-nil device. The device is
/// retained for the probe scope and released immediately, so a probe allocates
/// no long-lived object. False on Windows.
pub fn available() -> bool {
    if !cfg!(target_os = "macos") {
        return false;
    }
    let Some(lib) = lib() else {
        return false;
    };
    let _pool = Pool::new(lib);
    unsafe {
        let raw = (lib.create_device)();
        match Objc::retain(lib, raw) {
            Some(device) => {
                drop(device);
                true
            }
            None => false,
        }
    }
}

impl MetalContext {
    pub fn new() -> MetalResult<Self> {
        let lib = lib().ok_or(MetalRefusal::HostUnavailable)?;
        let _pool = Pool::new(lib);
        unsafe {
            let device_ptr = (lib.create_device)();
            let device = Objc::retain(lib, device_ptr).ok_or(MetalRefusal::HostUnavailable)?;
            let queue_ptr = send_ptr0(&lib, device.ptr, "newCommandQueue");
            let queue = Objc::adopt(lib, queue_ptr)
                .ok_or(MetalRefusal::AllocationFailed("command queue"))?;
            let p15 = compile_pipelines(&lib, &device, P15_MSL, &P15_ENTRY_POINTS)?;
            let whiten = compile_pipelines(&lib, &device, WHITEN_MSL, &WHITEN_ENTRY_POINTS)?;
            Ok(Self {
                lib,
                device,
                queue,
                p15,
                whiten,
                mesh: OnceCell::new(),
                mesh_perspective: OnceCell::new(),
                smile: OnceCell::new(),
                skin_lut: OnceCell::new(),
            })
        }
    }

    /// Runs `p15_facial_smooth_composite` and returns unquantized float32 RGBA.
    /// The alpha channel is restored bit-for-bit from `input.source`, so the
    /// golden path sees exactly the caller's alpha rather than a driver round.
    pub fn facial_smooth(&self, input: &SmoothInput<'_>) -> MetalResult<Vec<f32>> {
        input.validate()?;
        let _pool = Pool::new(self.lib);
        unsafe {
            let floats = plane_len(input.width, input.height, input.stride, 4)?;
            let uniforms = input.msl_uniforms();
            let src = self.buffer_f32(input.source)?;
            let mask = self.buffer_f32(input.mask_rgba)?;
            let skin = self.buffer_f32(input.skin_mask)?;
            let base = self.buffer_f32(input.base)?;
            let residual = self.buffer_f32(input.residual)?;
            let out = self.buffer_zero(floats * std::mem::size_of::<f32>())?;
            let uniform = self.buffer_bytes(as_bytes(&uniforms))?;
            let bindings = [&src, &mask, &skin, &base, &residual, &out, &uniform];
            self.dispatch(
                "p15_facial_smooth_composite",
                &bindings,
                grid_size(input.width, input.height),
            )?;
            let mut pixels = read_f32(&out, floats)?;
            restore_alpha_f32(&mut pixels, input.source);
            Ok(pixels)
        }
    }

    /// P15 host packing to RGBA8. Uses the same half-away rounding as the
    /// current CPU/CUDA stage while leaving `facial_smooth` as the float oracle.
    pub fn facial_smooth_rgba8(&self, input: &SmoothInput<'_>) -> MetalResult<Vec<u8>> {
        self.facial_smooth(input).map(|pixels| quantize_rgba8(&pixels))
    }

    /// Runs one P15 blur pass. `stage == 1` is horizontal, otherwise vertical.
    /// The box kernel ignores the stage because it shifts in both directions.
    pub fn blur(
        &self,
        kernel: BlurKernel,
        source: &[f32],
        width: u32,
        height: u32,
        stride: u32,
        stage: u32,
        ticket: u64,
        uniforms: ProductUniforms,
    ) -> MetalResult<Vec<f32>> {
        if ticket == 0 {
            return Err(MetalRefusal::MissingSameJobIdentity);
        }
        uniforms.validate()?;
        let floats = plane_len(width, height, stride, 4)?;
        if source.len() < floats {
            return Err(MetalRefusal::ShapeMismatch("source"));
        }
        if !source[..floats].iter().all(|v| v.is_finite()) {
            return Err(MetalRefusal::ValueOutOfRange("pixel"));
        }
        let _pool = Pool::new(self.lib);
        unsafe {
            let u = P15Uniforms {
                width,
                height,
                stride,
                pf: 0,
                pf_value: 0.0,
                gate: ACTIVE_GATE,
                legal_alpha: uniforms.legal_alpha,
                dark_circle_alpha: uniforms.dark_circle_alpha,
                diff_factor: uniforms.diff_factor,
                diff_factor2: uniforms.diff_factor2,
                color_threshold_factor: uniforms.color_threshold_factor,
                mix_alpha: uniforms.mix_alpha,
                diff_min_threshold: uniforms.diff_min_threshold,
                diff_max_threshold: uniforms.diff_max_threshold,
                high_frequency_min_threshold: uniforms.high_freq_min_threshold,
                high_frequency_max_threshold: uniforms.high_freq_max_threshold,
                surface_radius: SURFACE_RADIUS,
                surface_threshold: SURFACE_THRESHOLD,
                gauss_offset: GAUSS_OFFSET,
                roi_size: ROI_SIZE,
                algo_type: 0,
                box_blur_stage: stage,
            };
            let src = self.buffer_f32(source)?;
            let out = self.buffer_zero(floats * std::mem::size_of::<f32>())?;
            let uniform = self.buffer_bytes(as_bytes(&u))?;
            self.dispatch(
                kernel.entry_point(),
                &[&src, &out, &uniform],
                grid_size(width, height),
            )?;
            read_f32(&out, floats)
        }
    }

    /// Runs `o3_masked_lut_whiten`. Returns compact BGRA8 rows; alpha bytes are
    /// copied from the source, never from the GPU result.
    pub fn masked_whiten(&self, input: &WhitenInput<'_>) -> MetalResult<Vec<u8>> {
        input.validate()?;
        let row_bytes = (input.width as usize)
            .checked_mul(4)
            .ok_or(MetalRefusal::ShapeMismatch("source_bgra"))?;
        let compact_len = row_bytes
            .checked_mul(input.height as usize)
            .ok_or(MetalRefusal::ShapeMismatch("source_bgra"))?;
        let compact = if input.stride_bytes as usize == row_bytes {
            input.bgra[..compact_len].to_vec()
        } else {
            let mut rows = vec![0u8; compact_len];
            for y in 0..input.height as usize {
                let src = y * input.stride_bytes as usize;
                rows[y * row_bytes..(y + 1) * row_bytes]
                    .copy_from_slice(&input.bgra[src..src + row_bytes]);
            }
            rows
        };
        let params = WhitenParams {
            surface_width: input.width,
            surface_height: input.height,
            mask_x: input.mask_x,
            mask_y: input.mask_y,
            mask_width: input.mask_width,
            mask_height: input.mask_height,
            pixel_count: input.width * input.height,
            flags: input.flags,
            alpha: input.alpha,
            reserved0: 0.0,
            reserved1: 0.0,
            reserved2: 0.0,
        };
        let _pool = Pool::new(self.lib);
        unsafe {
            let src = self.buffer_bytes(&compact)?;
            let out = self.buffer_zero(compact.len())?;
            let mask = self.buffer_bytes(input.mask)?;
            let lut = self.buffer_f32(input.lut)?;
            let uniform = self.buffer_bytes(as_bytes(&params))?;
            self.dispatch(
                "o3_masked_lut_whiten",
                &[&src, &out, &mask, &lut, &uniform],
                grid_size(input.width, input.height),
            )?;
            let mut result = read_bytes(&out, compact.len())?;
            restore_alpha_bgra(&mut result, &compact);
            Ok(result)
        }
    }

    /// Runs `o3_coverage_max_mask_projection` and returns the ROI-local plane.
    pub fn project_mask(&self, input: &ProjectionInput<'_>) -> MetalResult<Vec<u8>> {
        input.validate()?;
        let params = input.params();
        let _pool = Pool::new(self.lib);
        unsafe {
            let src = self.buffer_bytes(input.source_mask)?;
            let out_len = input.destination_width as usize * input.destination_height as usize;
            let out = self.buffer_zero(out_len)?;
            let uniform = self.buffer_bytes(as_bytes(&params))?;
            self.dispatch(
                "o3_coverage_max_mask_projection",
                &[&src, &out, &uniform],
                grid_size(input.destination_width, input.destination_height),
            )?;
            read_bytes(&out, out_len)
        }
    }

    /// Runs `mesh_raster_select` then `mesh_raster_emit`.
    ///
    /// Each pass gets its own command buffer and `waitUntilCompleted`, so the
    /// select writes are visible to the emit pass by construction. The mesh
    /// pipelines compile on the first call only; a mesh compile failure is
    /// cached and does not affect a context that only uses P15/whiten.
    pub fn mesh_raster(&self, input: &metal_mesh::MeshRasterInput<'_>) -> MetalResult<Image> {
        input
            .validate_contract()
            .map_err(map_mesh_contract_refusal)?;
        input.validate().map_err(map_mesh_refusal)?;
        let params = input.params().map_err(map_mesh_refusal)?;
        let programs = self.mesh_programs()?;

        let width = input.source.width;
        let height = input.source.height;
        let pixel_count = input.source.pixel_count();
        let dest: Vec<f32> = input.dest.iter().flat_map(|p| p.iter().copied()).collect();
        let uv: Vec<f32> = input
            .source_uv
            .iter()
            .flat_map(|p| p.iter().copied())
            .collect();

        let _pool = Pool::new(self.lib);
        unsafe {
            let dest_buffer = self.buffer_f32(&dest)?;
            let uv_buffer = self.buffer_f32(&uv)?;
            let ebo_buffer = self.buffer_bytes(std::slice::from_raw_parts(
                input.ebo.as_ptr() as *const u8,
                input.ebo.len() * std::mem::size_of::<u16>(),
            ))?;
            let source_buffer = self.buffer_bytes(&input.source.pixels)?;
            let out_buffer = self.buffer_bytes(&input.source.pixels)?;
            let params_buffer = self.buffer_bytes(as_bytes(&params))?;
            match programs.kind {
                MeshKind::Atomic => {
                    let selected_bytes =
                        selected_fill(params.rules.overlap, pixel_count);
                    let coverage_bytes = vec![0u8; pixel_count * 4];
                    let selected_buffer = self.buffer_bytes(&selected_bytes)?;
                    let coverage_buffer = self.buffer_bytes(&coverage_bytes)?;
                    let select = mesh_pipeline(&programs.pipelines, "mesh_raster_select")?;
                    let emit = mesh_pipeline(&programs.pipelines, "mesh_raster_emit")?;
                    self.dispatch_pipeline(
                        select,
                        &[
                            &dest_buffer,
                            &ebo_buffer,
                            &params_buffer,
                            &selected_buffer,
                            &coverage_buffer,
                        ],
                        grid_1d(params.triangle_count as usize, 64),
                    )?;
                    self.dispatch_pipeline(
                        emit,
                        &[
                            &dest_buffer,
                            &uv_buffer,
                            &ebo_buffer,
                            &source_buffer,
                            &out_buffer,
                            &params_buffer,
                            &selected_buffer,
                            &coverage_buffer,
                        ],
                        grid_1d(pixel_count, 64),
                    )?;
                }
                MeshKind::Serial => {
                    let serial = mesh_pipeline(&programs.pipelines, "mesh_raster_serial")?;
                    self.dispatch_pipeline(
                        serial,
                        &[
                            &dest_buffer,
                            &uv_buffer,
                            &ebo_buffer,
                            &source_buffer,
                            &out_buffer,
                            &params_buffer,
                        ],
                        grid_1d(pixel_count, 64),
                    )?;
                }
            }

            let bytes = read_bytes(&out_buffer, input.source.pixels.len())?;
            Image::new(width, height, bytes).ok_or(MetalRefusal::ShapeMismatch("mesh output"))
        }
    }

    /// Runs the additive perspective-correct UV candidate.
    ///
    /// This is a separate pipeline/cache from affine `mesh_raster`; a
    /// perspective compilation failure cannot change affine behavior. The
    /// path is still a buffer-only RGBA8 candidate: it has no
    /// `MTLTexture`/`MTLSamplerState`/FBO and does not claim vendor
    /// PF8025/PF8033 parity. Vendor contracts continue to return
    /// `MetalRefusal::UnsupportedMeshContract` before dispatch.
    pub fn mesh_raster_perspective(
        &self,
        input: &metal_mesh::PerspectiveMeshInput<'_>,
    ) -> MetalResult<Image> {
        input
            .validate_contract()
            .map_err(map_mesh_contract_refusal)?;
        input.validate().map_err(map_mesh_refusal)?;
        let params = input.params().map_err(map_mesh_refusal)?;
        let programs = self.mesh_perspective_programs()?;

        let base = &input.base;
        let width = base.source.width;
        let height = base.source.height;
        let pixel_count = base.source.pixel_count();
        let dest: Vec<f32> = base.dest.iter().flat_map(|p| p.iter().copied()).collect();
        let uv: Vec<f32> = base
            .source_uv
            .iter()
            .flat_map(|p| p.iter().copied())
            .collect();
        let perspective = input.perspective_flat();

        let _pool = Pool::new(self.lib);
        unsafe {
            let dest_buffer = self.buffer_f32(&dest)?;
            let uv_buffer = self.buffer_f32(&uv)?;
            let perspective_buffer = self.buffer_f32(&perspective)?;
            let ebo_buffer = self.buffer_bytes(std::slice::from_raw_parts(
                base.ebo.as_ptr() as *const u8,
                base.ebo.len() * std::mem::size_of::<u16>(),
            ))?;
            let source_buffer = self.buffer_bytes(&base.source.pixels)?;
            let out_buffer = self.buffer_bytes(&base.source.pixels)?;
            let params_buffer = self.buffer_bytes(as_bytes(&params))?;
            match programs.kind {
                MeshKind::Atomic => {
                    let selected_bytes =
                        selected_fill(params.rules.overlap, pixel_count);
                    let coverage_bytes = vec![0u8; pixel_count * 4];
                    let selected_buffer = self.buffer_bytes(&selected_bytes)?;
                    let coverage_buffer = self.buffer_bytes(&coverage_bytes)?;
                    let select = mesh_pipeline(&programs.pipelines, "mesh_raster_select")?;
                    let emit = mesh_pipeline(&programs.pipelines, "mesh_raster_emit_perspective")?;
                    self.dispatch_pipeline(
                        select,
                        &[
                            &dest_buffer,
                            &ebo_buffer,
                            &params_buffer,
                            &selected_buffer,
                            &coverage_buffer,
                        ],
                        grid_1d(params.triangle_count as usize, 64),
                    )?;
                    self.dispatch_pipeline(
                        emit,
                        &[
                            &dest_buffer,
                            &uv_buffer,
                            &ebo_buffer,
                            &source_buffer,
                            &out_buffer,
                            &params_buffer,
                            &selected_buffer,
                            &coverage_buffer,
                            &perspective_buffer,
                        ],
                        grid_1d(pixel_count, 64),
                    )?;
                }
                MeshKind::Serial => {
                    let serial =
                        mesh_pipeline(&programs.pipelines, "mesh_raster_serial_perspective")?;
                    self.dispatch_pipeline(
                        serial,
                        &[
                            &dest_buffer,
                            &uv_buffer,
                            &ebo_buffer,
                            &source_buffer,
                            &out_buffer,
                            &params_buffer,
                            &perspective_buffer,
                        ],
                        grid_1d(pixel_count, 64),
                    )?;
                }
            }

            let bytes = read_bytes(&out_buffer, base.source.pixels.len())?;
            Image::new(width, height, bytes).ok_or(MetalRefusal::ShapeMismatch("mesh output"))
        }
    }

    /// Compile probe for the additive perspective path. It does not share the
    /// affine `mesh` cache or its fallback state.
    fn mesh_perspective_programs(&self) -> MetalResult<&MeshPrograms> {
        let compiled = self.mesh_perspective.get_or_init(|| unsafe {
            let atomic = compile_pipelines(
                &self.lib,
                &self.device,
                metal_mesh::MESH_MSL,
                &MESH_PERSPECTIVE_ENTRY_POINTS,
            );
            match atomic {
                Ok(pipelines) => Ok(MeshPrograms {
                    kind: MeshKind::Atomic,
                    pipelines,
                }),
                Err(refusal) if mesh_fallback_allowed(&refusal) => compile_pipelines(
                    &self.lib,
                    &self.device,
                    metal_mesh::MESH_MSL,
                    &MESH_PERSPECTIVE_SERIAL_ENTRY_POINTS,
                )
                .map(|pipelines| MeshPrograms {
                    kind: MeshKind::Serial,
                    pipelines,
                }),
                Err(refusal) => Err(refusal),
            }
        });
        match compiled {
            Ok(programs) => Ok(programs),
            Err(refusal) => Err(refusal.clone()),
        }
    }

    /// Compile probe: atomic first, serial only when the atomic MSL fails to
    /// compile. Other failures are returned unchanged. No GPU_LOCK or global
    /// ordering is involved.
    fn mesh_programs(&self) -> MetalResult<&MeshPrograms> {
        let compiled = self.mesh.get_or_init(|| unsafe {
            let atomic = compile_pipelines(
                &self.lib,
                &self.device,
                metal_mesh::MESH_MSL,
                &MESH_ENTRY_POINTS,
            );
            match atomic {
                Ok(pipelines) => Ok(MeshPrograms {
                    kind: MeshKind::Atomic,
                    pipelines,
                }),
                Err(refusal) if mesh_fallback_allowed(&refusal) => compile_pipelines(
                    &self.lib,
                    &self.device,
                    MESH_SERIAL_MSL,
                    &MESH_SERIAL_ENTRY_POINTS,
                )
                .map(|pipelines| MeshPrograms {
                    kind: MeshKind::Serial,
                    pipelines,
                }),
                Err(refusal) => Err(refusal),
            }
        });
        match compiled {
            Ok(programs) => Ok(programs),
            Err(refusal) => Err(refusal.clone()),
        }
    }

    /// PF100007 direct pixel displacement. Lazy-compiles `pf100007_smile` on
    /// the first call; input math is the CPU `smile_warp` law.
    pub fn smile(&self, input: &metal_smile::SmileInput<'_>) -> MetalResult<Vec<u8>> {
        let pixels = input.validate().map_err(map_mesh_refusal)?;
        let params = input.params().map_err(map_mesh_refusal)?;
        let pipeline = self.smile_pipeline()?;
        let _pool = Pool::new(self.lib);
        unsafe {
            let source = self.buffer_bytes(input.source_bgra)?;
            let lip = self.buffer_bytes(input.lip_mask)?;
            let mouth = self.buffer_bytes(input.mouth_mask)?;
            let dx = self.buffer_f32(input.model_dx)?;
            let dy = self.buffer_f32(input.model_dy)?;
            let out = self.buffer_bytes(input.source_bgra)?;
            let uniform = self.buffer_bytes(as_bytes(&params))?;
            self.dispatch_pipeline(
                pipeline,
                &[&source, &lip, &mouth, &dx, &dy, &out, &uniform],
                (
                    MTLSize {
                        width: (input.width as usize).div_ceil(16),
                        height: (input.height as usize).div_ceil(16),
                        depth: 1,
                    },
                    MTLSize {
                        width: 16,
                        height: 16,
                        depth: 1,
                    },
                ),
            )?;
            read_bytes(&out, pixels * 4)
        }
    }

    fn smile_pipeline(&self) -> MetalResult<&Objc> {
        let compiled = self.smile.get_or_init(|| unsafe {
            compile_pipelines(
                &self.lib,
                &self.device,
                metal_smile::SMILE_MSL,
                &SMILE_ENTRY_POINTS,
            )
            .map(|mut pipelines| pipelines.remove(0).1)
        });
        match compiled {
            Ok(pipeline) => Ok(pipeline),
            Err(refusal) => Err(refusal.clone()),
        }
    }

    /// Fixed-point SkinAdjust LUT path matching the Rust/NVIDIA oracle:
    /// RGBA8 source, R8 mask, packed RGB8 512x512 atlas, 12-bit floor store.
    /// The public `masked_whiten` float contract remains separate.
    pub fn skin_lut_fixed(
        &self,
        source: &Image,
        mask_r: &[u8],
        lut_rgb: &[u8],
        mix: f32,
    ) -> MetalResult<Image> {
        source
            .validate()
            .map_err(|_| MetalRefusal::ShapeMismatch("source"))?;
        if !mix.is_finite() || !(0.0..=1.0).contains(&mix) {
            return Err(MetalRefusal::ValueOutOfRange("mix"));
        }
        let pixels = source.pixel_count();
        if mask_r.len() != pixels {
            return Err(MetalRefusal::ShapeMismatch("mask"));
        }
        if lut_rgb.len() != 512 * 512 * 3 {
            return Err(MetalRefusal::ShapeMismatch("lut_rgb"));
        }
        let stride = source
            .width
            .checked_mul(4)
            .ok_or(MetalRefusal::ValueOutOfRange("stride"))?;
        let params = SkinLutParams {
            width: source.width,
            height: source.height,
            stride,
            mask_row_bytes: source.width,
            mix,
            gate: 0.001,
            lut_width: 512,
            lut_height: 512,
        };
        let pipeline = self.skin_lut_pipeline()?;
        let _pool = Pool::new(self.lib);
        unsafe {
            let source_buffer = self.buffer_bytes(&source.pixels)?;
            let mask_buffer = self.buffer_bytes(mask_r)?;
            let lut_buffer = self.buffer_bytes(lut_rgb)?;
            let out_buffer = self.buffer_bytes(&source.pixels)?;
            let params_buffer = self.buffer_bytes(as_bytes(&params))?;
            self.dispatch_pipeline(
                pipeline,
                &[
                    &source_buffer,
                    &mask_buffer,
                    &lut_buffer,
                    &out_buffer,
                    &params_buffer,
                ],
                (
                    MTLSize {
                        width: (source.width as usize).div_ceil(16),
                        height: (source.height as usize).div_ceil(16),
                        depth: 1,
                    },
                    MTLSize {
                        width: 16,
                        height: 16,
                        depth: 1,
                    },
                ),
            )?;
            let bytes = read_bytes(&out_buffer, pixels * 4)?;
            Image::new(source.width, source.height, bytes)
                .ok_or(MetalRefusal::ShapeMismatch("output"))
        }
    }

    fn skin_lut_pipeline(&self) -> MetalResult<&Objc> {
        let compiled = self.skin_lut.get_or_init(|| unsafe {
            compile_pipelines(
                &self.lib,
                &self.device,
                SKIN_LUT_MSL,
                &SKIN_LUT_ENTRY_POINTS,
            )
            .map(|mut pipelines| pipelines.remove(0).1)
        });
        match compiled {
            Ok(pipeline) => Ok(pipeline),
            Err(refusal) => Err(refusal.clone()),
        }
    }

    fn pipeline(&self, name: &'static str) -> MetalResult<&Objc> {
        self.p15
            .iter()
            .chain(self.whiten.iter())
            .find(|(entry, _)| *entry == name)
            .map(|(_, pipeline)| pipeline)
            .ok_or(MetalRefusal::CompileFailed("pipeline name"))
    }

    unsafe fn buffer_bytes(&self, bytes: &[u8]) -> MetalResult<Objc> {
        if bytes.is_empty() {
            return Err(MetalRefusal::MissingInput("empty buffer"));
        }
        let ptr = send_ptr_puu(
            &self.lib,
            self.device.ptr,
            "newBufferWithBytes:length:options:",
            bytes.as_ptr() as *mut c_void,
            bytes.len(),
            0,
        );
        Objc::adopt(self.lib, ptr).ok_or(MetalRefusal::AllocationFailed("buffer"))
    }

    unsafe fn buffer_f32(&self, values: &[f32]) -> MetalResult<Objc> {
        self.buffer_bytes(std::slice::from_raw_parts(
            values.as_ptr() as *const u8,
            values.len() * std::mem::size_of::<f32>(),
        ))
    }

    unsafe fn buffer_zero(&self, length: usize) -> MetalResult<Objc> {
        if length == 0 {
            return Err(MetalRefusal::MissingInput("empty buffer"));
        }
        let ptr = send_ptr_uu(
            &self.lib,
            self.device.ptr,
            "newBufferWithLength:options:",
            length,
            0,
        );
        Objc::adopt(self.lib, ptr).ok_or(MetalRefusal::AllocationFailed("buffer"))
    }

    unsafe fn dispatch(
        &self,
        name: &'static str,
        buffers: &[&Objc],
        (groups, threads): (MTLSize, MTLSize),
    ) -> MetalResult<()> {
        let pipeline = self.pipeline(name)?;
        self.dispatch_pipeline(pipeline, buffers, (groups, threads))
    }

    unsafe fn dispatch_pipeline(
        &self,
        pipeline: &Objc,
        buffers: &[&Objc],
        (groups, threads): (MTLSize, MTLSize),
    ) -> MetalResult<()> {
        let command_buffer = Objc::retain(
            self.lib,
            send_ptr0(&self.lib, self.queue.ptr, "commandBuffer"),
        )
        .ok_or(MetalRefusal::AllocationFailed("command buffer"))?;
        let encoder = Objc::retain(
            self.lib,
            send_ptr0(&self.lib, command_buffer.ptr, "computeCommandEncoder"),
        )
        .ok_or(MetalRefusal::AllocationFailed("compute encoder"))?;
        send_void1(
            &self.lib,
            encoder.ptr,
            "setComputePipelineState:",
            pipeline.ptr,
        );
        for (index, buffer) in buffers.iter().enumerate() {
            send_void_puu(
                &self.lib,
                encoder.ptr,
                "setBuffer:offset:atIndex:",
                buffer.ptr,
                0,
                index,
            );
        }
        send_sizes_void(
            &self.lib,
            encoder.ptr,
            "dispatchThreadgroups:threadsPerThreadgroup:",
            groups,
            threads,
        );
        send_void0(&self.lib, encoder.ptr, "endEncoding");
        send_void0(&self.lib, command_buffer.ptr, "commit");
        send_void0(&self.lib, command_buffer.ptr, "waitUntilCompleted");
        let status = send_usize0(&self.lib, command_buffer.ptr, "status");
        let error = send_ptr0(&self.lib, command_buffer.ptr, "error");
        if status != 4 || !error.is_null() {
            report_error(&self.lib, error);
            return Err(MetalRefusal::CommandFailed);
        }
        Ok(())
    }

    pub fn device_pointer(&self) -> *mut c_void {
        self.device.ptr
    }
}

pub fn execute_facial_smooth(input: &SmoothInput<'_>) -> MetalResult<Vec<f32>> {
    MetalContext::new()?.facial_smooth(input)
}

pub fn execute_facial_smooth_rgba8(input: &SmoothInput<'_>) -> MetalResult<Vec<u8>> {
    MetalContext::new()?.facial_smooth_rgba8(input)
}

fn quantize_rgba8(pixels: &[f32]) -> Vec<u8> {
    pixels
        .iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

pub fn execute_masked_whiten(input: &WhitenInput<'_>) -> MetalResult<Vec<u8>> {
    MetalContext::new()?.masked_whiten(input)
}

pub fn execute_mask_projection(input: &ProjectionInput<'_>) -> MetalResult<Vec<u8>> {
    MetalContext::new()?.project_mask(input)
}

pub fn execute_skin_lut_fixed(
    source: &Image,
    mask_r: &[u8],
    lut_rgb: &[u8],
    mix: f32,
) -> MetalResult<Image> {
    MetalContext::new()?.skin_lut_fixed(source, mask_r, lut_rgb, mix)
}

pub fn check_gate(pf: u32, pf_value: f32) -> MetalResult<()> {
    if !SUPPORTED_PF.contains(&pf) {
        return Err(MetalRefusal::UnsupportedPf(pf));
    }
    if !pf_value.is_finite() || pf_value <= ACTIVE_GATE {
        return Err(MetalRefusal::Inactive);
    }
    Ok(())
}

pub fn restore_alpha_f32(pixels: &mut [f32], source: &[f32]) {
    for (dst, src) in pixels.chunks_exact_mut(4).zip(source.chunks_exact(4)) {
        dst[3] = src[3];
    }
}

pub fn restore_alpha_bgra(pixels: &mut [u8], source: &[u8]) {
    for (dst, src) in pixels.chunks_exact_mut(4).zip(source.chunks_exact(4)) {
        dst[3] = src[3];
    }
}

fn map_mesh_refusal(refusal: CrateRefusal) -> MetalRefusal {
    match refusal {
        CrateRefusal::MissingInput(field) => MetalRefusal::MissingInput(field),
        CrateRefusal::ShapeMismatch(field) => MetalRefusal::ShapeMismatch(field),
        CrateRefusal::ValueOutOfRange(field) => MetalRefusal::ValueOutOfRange(field),
        CrateRefusal::UnknownEffect(_) | CrateRefusal::NotImplemented(_) => {
            MetalRefusal::ValueOutOfRange("mesh")
        }
    }
}

fn map_mesh_contract_refusal(refusal: metal_mesh::MeshContractRefusal) -> MetalRefusal {
    MetalRefusal::UnsupportedMeshContract(refusal)
}

fn mesh_fallback_allowed(refusal: &MetalRefusal) -> bool {
    matches!(refusal, MetalRefusal::CompileFailed(_))
}

fn mesh_pipeline<'a>(
    pipelines: &'a [(&'static str, Objc)],
    name: &'static str,
) -> MetalResult<&'a Objc> {
    pipelines
        .iter()
        .find(|(entry, _)| *entry == name)
        .map(|(_, pipeline)| pipeline)
        .ok_or(MetalRefusal::CompileFailed("mesh pipeline name"))
}

fn plane_len(width: u32, height: u32, stride: u32, channels: usize) -> MetalResult<usize> {
    if width == 0 || height == 0 || stride < width {
        return Err(MetalRefusal::ShapeMismatch("geometry"));
    }
    (stride as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or(MetalRefusal::ShapeMismatch("geometry"))
}

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
struct MTLSize {
    width: usize,
    height: usize,
    depth: usize,
}

fn grid_size(width: u32, height: u32) -> (MTLSize, MTLSize) {
    let threads = MTLSize {
        width: 16,
        height: 16,
        depth: 1,
    };
    let groups = MTLSize {
        width: (width as usize).div_ceil(16),
        height: (height as usize).div_ceil(16),
        depth: 1,
    };
    (groups, threads)
}

fn grid_1d(count: usize, threads: usize) -> (MTLSize, MTLSize) {
    let threads = threads.max(1);
    let groups = MTLSize {
        width: count.div_ceil(threads),
        height: 1,
        depth: 1,
    };
    (
        groups,
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    )
}

unsafe fn read_f32(buffer: &Objc, len: usize) -> MetalResult<Vec<f32>> {
    let contents = send_ptr0(&buffer.lib, buffer.ptr, "contents");
    if contents.is_null() {
        return Err(MetalRefusal::AllocationFailed("contents"));
    }
    let available = send_usize0(&buffer.lib, buffer.ptr, "length") / 4;
    if available < len {
        return Err(MetalRefusal::CommandFailed);
    }
    Ok(std::slice::from_raw_parts(contents as *const f32, len).to_vec())
}

unsafe fn read_bytes(buffer: &Objc, len: usize) -> MetalResult<Vec<u8>> {
    let contents = send_ptr0(&buffer.lib, buffer.ptr, "contents");
    if contents.is_null() {
        return Err(MetalRefusal::AllocationFailed("contents"));
    }
    let available = send_usize0(&buffer.lib, buffer.ptr, "length");
    if available < len {
        return Err(MetalRefusal::CommandFailed);
    }
    Ok(std::slice::from_raw_parts(contents as *const u8, len).to_vec())
}

unsafe fn compile_pipelines(
    lib: &Lib,
    device: &Objc,
    source: &str,
    names: &[&'static str],
) -> MetalResult<Vec<(&'static str, Objc)>> {
    let ns_source = lib
        .ns_string(source)
        .ok_or(MetalRefusal::CompileFailed("NSString"))?;
    let options = Objc::adopt(*lib, send_ptr0(lib, lib.class("MTLCompileOptions"), "new"))
        .ok_or(MetalRefusal::CompileFailed("MTLCompileOptions"))?;
    // MTLCompileOptions.fastMathEnabled defaults to true on Apple platforms.
    // The recovered f32 ordering must not become fast-math; keep it off.
    send_void_bool(lib, options.ptr, "setFastMathEnabled:", 0);
    let mut error = std::ptr::null_mut();
    let library_ptr = send_ptr3_err(
        lib,
        device.ptr,
        "newLibraryWithSource:options:error:",
        ns_source.ptr,
        options.ptr,
        &mut error,
    );
    let library = match Objc::adopt(*lib, library_ptr) {
        Some(library) => library,
        None => {
            report_error(lib, error);
            return Err(MetalRefusal::CompileFailed("newLibraryWithSource"));
        }
    };
    let mut pipelines = Vec::with_capacity(names.len());
    for name in names {
        let ns_name = lib
            .ns_string(name)
            .ok_or(MetalRefusal::CompileFailed("NSString"))?;
        let function_ptr = send_ptr1(lib, library.ptr, "newFunctionWithName:", ns_name.ptr);
        let function = Objc::adopt(*lib, function_ptr)
            .ok_or(MetalRefusal::CompileFailed("newFunctionWithName"))?;
        error = std::ptr::null_mut();
        let pipeline_ptr = send_ptr2_err(
            lib,
            device.ptr,
            "newComputePipelineStateWithFunction:error:",
            function.ptr,
            &mut error,
        );
        let pipeline = match Objc::adopt(*lib, pipeline_ptr) {
            Some(pipeline) => pipeline,
            None => {
                report_error(lib, error);
                return Err(MetalRefusal::CompileFailed("newComputePipelineState"));
            }
        };
        pipelines.push((*name, pipeline));
    }
    Ok(pipelines)
}

fn report_error(lib: &Lib, error: *mut c_void) {
    if error.is_null() {
        return;
    }
    unsafe {
        let description = send_ptr0(lib, error, "localizedDescription");
        if description.is_null() {
            return;
        }
        let text = send_cstr0(lib, description, "UTF8String");
        if !text.is_null() {
            eprintln!(
                "pixcake metal: {}",
                std::ffi::CStr::from_ptr(text).to_string_lossy()
            );
        }
    }
}

#[derive(Clone, Copy)]
struct Lib {
    send: *const c_void,
    get_class: unsafe extern "C" fn(*const c_char) -> *mut c_void,
    register_sel: unsafe extern "C" fn(*const c_char) -> *mut c_void,
    pool_push: unsafe extern "C" fn() -> *mut c_void,
    pool_pop: unsafe extern "C" fn(*mut c_void),
    create_device: unsafe extern "C" fn() -> *mut c_void,
}

// Function pointers and an immutable framework handle are safe to share.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

impl Lib {
    unsafe fn class(&self, name: &str) -> *mut c_void {
        let name = CString::new(name).expect("class name has no NUL");
        (self.get_class)(name.as_ptr())
    }

    unsafe fn selector(&self, name: &str) -> *mut c_void {
        let name = CString::new(name).expect("selector has no NUL");
        (self.register_sel)(name.as_ptr())
    }

    unsafe fn ns_string(&self, value: &str) -> Option<Objc> {
        let value = CString::new(value).ok()?;
        let raw = send_ptr1(
            self,
            self.class("NSString"),
            "stringWithUTF8String:",
            value.as_ptr() as *mut c_void,
        );
        Objc::retain(*self, raw)
    }
}

struct Objc {
    ptr: *mut c_void,
    lib: Lib,
}

impl Objc {
    fn adopt(lib: Lib, ptr: *mut c_void) -> Option<Self> {
        (!ptr.is_null()).then_some(Self { ptr, lib })
    }

    /// For `+0` autoreleased objects: retain so this owner can release in Drop.
    fn retain(lib: Lib, ptr: *mut c_void) -> Option<Self> {
        let object = Self::adopt(lib, ptr)?;
        unsafe {
            send_ptr0(&lib, object.ptr, "retain");
        }
        Some(object)
    }
}

impl Drop for Objc {
    fn drop(&mut self) {
        unsafe {
            send_void0(&self.lib, self.ptr, "release");
        }
    }
}

struct Pool {
    lib: Lib,
    token: *mut c_void,
}

impl Pool {
    fn new(lib: Lib) -> Self {
        let token = unsafe { (lib.pool_push)() };
        Self { lib, token }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        unsafe {
            (self.lib.pool_pop)(self.token);
        }
    }
}

fn lib() -> Option<Lib> {
    static CELL: std::sync::OnceLock<Option<Lib>> = std::sync::OnceLock::new();
    *CELL.get_or_init(load)
}

#[cfg(target_os = "macos")]
const RTLD_NOW: i32 = 2;

#[cfg(target_os = "macos")]
fn load() -> Option<Lib> {
    extern "C" {
        fn dlopen(path: *const c_char, mode: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    unsafe fn open(path: &str) -> Option<*mut c_void> {
        let path = CString::new(path).ok()?;
        let handle = dlopen(path.as_ptr(), RTLD_NOW);
        (!handle.is_null()).then_some(handle)
    }

    unsafe fn symbol(handle: *mut c_void, name: &str) -> Option<*mut c_void> {
        let name = CString::new(name).ok()?;
        let pointer = dlsym(handle, name.as_ptr());
        (!pointer.is_null()).then_some(pointer)
    }

    unsafe {
        let objc = open("/usr/lib/libobjc.A.dylib")?;
        let _foundation = open("/System/Library/Frameworks/Foundation.framework/Foundation")?;
        // macOS MTLCreateSystemDefaultDevice documents Core Graphics as a
        // required framework; a headless crate cannot assume GPUI linked it.
        let _core_graphics =
            open("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")?;
        let metal = open("/System/Library/Frameworks/Metal.framework/Metal")?;
        Some(Lib {
            send: symbol(objc, "objc_msgSend")?,
            get_class: std::mem::transmute(symbol(objc, "objc_getClass")?),
            register_sel: std::mem::transmute(symbol(objc, "sel_registerName")?),
            pool_push: std::mem::transmute(symbol(objc, "objc_autoreleasePoolPush")?),
            pool_pop: std::mem::transmute(symbol(objc, "objc_autoreleasePoolPop")?),
            create_device: std::mem::transmute(symbol(metal, "MTLCreateSystemDefaultDevice")?),
        })
    }
}

#[cfg(not(target_os = "macos"))]
fn load() -> Option<Lib> {
    None
}

unsafe fn send_ptr0(lib: &Lib, object: *mut c_void, selector: &str) -> *mut c_void {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
        std::mem::transmute(lib.send);
    send(object, lib.selector(selector))
}

unsafe fn send_usize0(lib: &Lib, object: *mut c_void, selector: &str) -> usize {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize =
        std::mem::transmute(lib.send);
    send(object, lib.selector(selector))
}

unsafe fn send_ptr1(
    lib: &Lib,
    object: *mut c_void,
    selector: &str,
    argument: *mut c_void,
) -> *mut c_void {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void =
        std::mem::transmute(lib.send);
    send(object, lib.selector(selector), argument)
}

unsafe fn send_cstr0(lib: &Lib, object: *mut c_void, selector: &str) -> *const c_char {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *const c_char =
        std::mem::transmute(lib.send);
    send(object, lib.selector(selector))
}

unsafe fn send_void0(lib: &Lib, object: *mut c_void, selector: &str) {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void) = std::mem::transmute(lib.send);
    send(object, lib.selector(selector))
}

unsafe fn send_void1(lib: &Lib, object: *mut c_void, selector: &str, argument: *mut c_void) {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) =
        std::mem::transmute(lib.send);
    send(object, lib.selector(selector), argument)
}

unsafe fn send_void_bool(lib: &Lib, object: *mut c_void, selector: &str, value: i8) {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, i8) = std::mem::transmute(lib.send);
    send(object, lib.selector(selector), value)
}

unsafe fn send_ptr2_err(
    lib: &Lib,
    object: *mut c_void,
    selector: &str,
    first: *mut c_void,
    error: *mut *mut c_void,
) -> *mut c_void {
    let send: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut *mut c_void,
    ) -> *mut c_void = std::mem::transmute(lib.send);
    send(object, lib.selector(selector), first, error)
}

unsafe fn send_ptr3_err(
    lib: &Lib,
    object: *mut c_void,
    selector: &str,
    first: *mut c_void,
    second: *mut c_void,
    error: *mut *mut c_void,
) -> *mut c_void {
    let send: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut *mut c_void,
    ) -> *mut c_void = std::mem::transmute(lib.send);
    send(object, lib.selector(selector), first, second, error)
}

unsafe fn send_ptr_puu(
    lib: &Lib,
    object: *mut c_void,
    selector: &str,
    pointer: *mut c_void,
    length: usize,
    options: usize,
) -> *mut c_void {
    let send: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut c_void,
        usize,
        usize,
    ) -> *mut c_void = std::mem::transmute(lib.send);
    send(object, lib.selector(selector), pointer, length, options)
}

unsafe fn send_void_puu(
    lib: &Lib,
    object: *mut c_void,
    selector: &str,
    pointer: *mut c_void,
    length: usize,
    index: usize,
) {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, usize, usize) =
        std::mem::transmute(lib.send);
    send(object, lib.selector(selector), pointer, length, index)
}

unsafe fn send_ptr_uu(
    lib: &Lib,
    object: *mut c_void,
    selector: &str,
    length: usize,
    options: usize,
) -> *mut c_void {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, usize, usize) -> *mut c_void =
        std::mem::transmute(lib.send);
    send(object, lib.selector(selector), length, options)
}

unsafe fn send_sizes_void(
    lib: &Lib,
    object: *mut c_void,
    selector: &str,
    groups: MTLSize,
    threads: MTLSize,
) {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, MTLSize, MTLSize) =
        std::mem::transmute(lib.send);
    send(object, lib.selector(selector), groups, threads)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_uniforms() -> ProductUniforms {
        ProductUniforms {
            legal_alpha: 0.8,
            dark_circle_alpha: 0.6,
            diff_factor: 1.2,
            diff_factor2: 0.1,
            color_threshold_factor: 0.3,
            mix_alpha: 0.4,
            diff_min_threshold: -1.0,
            diff_max_threshold: 1.0,
            high_freq_min_threshold: -0.5,
            high_freq_max_threshold: 0.5,
        }
    }

    fn one_pixel() -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4], [f32; 1]) {
        (
            [100.0 / 255.0, 150.0 / 255.0, 200.0 / 255.0, 77.0 / 255.0],
            [10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0, 0.0],
            [100.0 / 255.0, 150.0 / 255.0, 200.0 / 255.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0],
        )
    }

    #[test]
    fn abi_sizes_and_field_offsets() {
        assert_eq!(std::mem::size_of::<P15Uniforms>(), 88);
        assert_eq!(std::mem::size_of::<WhitenParams>(), 48);
        assert_eq!(std::mem::size_of::<ProjectionParams>(), 80);
        let uniforms = P15Uniforms {
            width: 7,
            height: 9,
            stride: 11,
            pf: 90023,
            pf_value: 0.5,
            gate: ACTIVE_GATE,
            legal_alpha: 0.0,
            dark_circle_alpha: 0.0,
            diff_factor: 0.0,
            diff_factor2: 0.0,
            color_threshold_factor: 0.0,
            mix_alpha: 0.0,
            diff_min_threshold: 0.0,
            diff_max_threshold: 0.0,
            high_frequency_min_threshold: 0.0,
            high_frequency_max_threshold: 0.0,
            surface_radius: SURFACE_RADIUS,
            surface_threshold: SURFACE_THRESHOLD,
            gauss_offset: GAUSS_OFFSET,
            roi_size: ROI_SIZE,
            algo_type: 0,
            box_blur_stage: 1,
        };
        let bytes = as_bytes(&uniforms);
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 7);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 90023);
        assert_eq!(f32::from_le_bytes(bytes[16..20].try_into().unwrap()), 0.5);
        assert_eq!(
            f32::from_le_bytes(bytes[20..24].try_into().unwrap()),
            ACTIVE_GATE
        );
        assert_eq!(u32::from_le_bytes(bytes[84..88].try_into().unwrap()), 1);
    }

    #[test]
    fn binding_tables_are_complete_and_contiguous() {
        for table in [
            P15_COMPOSITE_BINDINGS.as_slice(),
            P15_BLUR_BINDINGS.as_slice(),
            WHITEN_BINDINGS.as_slice(),
            PROJECTION_BINDINGS.as_slice(),
        ] {
            for (index, binding) in table.iter().enumerate() {
                assert_eq!(binding.index as usize, index);
            }
        }
        assert_eq!(P15_COMPOSITE_BINDINGS[6].element_bytes, 88);
        assert_eq!(WHITEN_BINDINGS[4].element_bytes, 48);
        assert_eq!(PROJECTION_BINDINGS[2].element_bytes, 80);
        assert!(TEXTURE_BINDINGS.is_empty());
    }

    #[test]
    fn plane_length_is_scalars_not_bytes() {
        assert_eq!(plane_len(3, 2, 4, 4).unwrap(), 32);
        assert_eq!(plane_len(3, 2, 4, 1).unwrap(), 8);
        assert_eq!(plane_len(3, 2, 4, 4).unwrap() * 4, 128);
        assert_eq!(
            plane_len(3, 2, 2, 4),
            Err(MetalRefusal::ShapeMismatch("geometry"))
        );
    }

    #[test]
    fn ten_product_uniforms_are_exact_and_finite_checked() {
        assert_eq!(PRODUCT_UNIFORM_NAMES.len(), 10);
        let mut uniforms = test_uniforms();
        assert_eq!(uniforms.values().len(), 10);
        assert!(uniforms.validate().is_ok());
        uniforms.mix_alpha = f32::NAN;
        assert_eq!(
            uniforms.validate(),
            Err(MetalRefusal::ValueOutOfRange("uniforms"))
        );
        let mut inverted = test_uniforms();
        inverted.diff_min_threshold = 2.0;
        assert_eq!(
            inverted.validate(),
            Err(MetalRefusal::ValueOutOfRange("uniforms"))
        );
    }

    #[test]
    fn mesh_refusal_mapping_and_1d_grid_are_stable() {
        assert_eq!(
            map_mesh_refusal(CrateRefusal::ShapeMismatch("ebo")),
            MetalRefusal::ShapeMismatch("ebo")
        );
        assert_eq!(
            map_mesh_refusal(CrateRefusal::MissingInput("ebo")),
            MetalRefusal::MissingInput("ebo")
        );
        let (groups, threads) = grid_1d(65, 64);
        assert_eq!(groups.width, 2);
        assert_eq!(groups.height, 1);
        assert_eq!(threads.width, 64);
        assert_eq!(threads.depth, 1);
    }

    #[test]
    fn mesh_contract_refusal_preserves_capabilities() {
        for effect in [
            metal_mesh::VendorMeshEffect::Pf8025,
            metal_mesh::VendorMeshEffect::Pf8033,
        ] {
            let refusal = metal_mesh::MeshContractRefusal::VendorForwardMeshUnsupported {
                effect,
                missing_perspective_uv: true,
                missing_texture_sampler: true,
                missing_fbo: true,
            };
            match map_mesh_contract_refusal(refusal) {
                MetalRefusal::UnsupportedMeshContract(actual) => {
                    assert_eq!(actual, refusal);
                }
                other => panic!("contract refusal was folded: {other:?}"),
            }
        }
    }

    #[test]
    fn mesh_fallback_is_compile_failure_only() {
        assert!(mesh_fallback_allowed(&MetalRefusal::CompileFailed(
            "atomic"
        )));
        assert!(!mesh_fallback_allowed(&MetalRefusal::AllocationFailed(
            "buffer"
        )));
        assert!(!mesh_fallback_allowed(&MetalRefusal::CommandFailed));
        assert!(MESH_SERIAL_MSL.contains("kernel void mesh_raster_serial"));
        assert!(metal_mesh::MESH_MSL.contains("kernel void mesh_raster_emit_perspective"));
        assert_eq!(MESH_SERIAL_MSL.matches("kernel void ").count(), 1);
        assert_eq!(MESH_SERIAL_ENTRY_POINTS, ["mesh_raster_serial"]);
        assert_eq!(
            metal_mesh::MESH_PERSPECTIVE_EMIT_ENTRY,
            "mesh_raster_emit_perspective"
        );
    }

    #[test]
    fn perspective_entry_points_are_separate_from_affine() {
        assert_eq!(
            MESH_ENTRY_POINTS,
            ["mesh_raster_select", "mesh_raster_emit"]
        );
        assert_eq!(
            MESH_PERSPECTIVE_ENTRY_POINTS,
            ["mesh_raster_select", "mesh_raster_emit_perspective"]
        );
        assert_eq!(MESH_SERIAL_ENTRY_POINTS, ["mesh_raster_serial"]);
        assert_eq!(
            MESH_PERSPECTIVE_SERIAL_ENTRY_POINTS,
            ["mesh_raster_serial_perspective"]
        );
        assert!(metal_mesh::MESH_MSL.contains("kernel void mesh_raster_emit_perspective"));
        assert!(metal_mesh::MESH_MSL.contains("kernel void mesh_raster_serial_perspective"));
        assert_eq!(
            metal_mesh::MESH_SERIAL_MSL.matches("kernel void ").count(),
            1
        );
    }

    #[test]
    fn a11_host_capability_probe_reports_blocked_device_tests() {
        const RUNTIME_TEST_SOURCE: &str = include_str!("../tests/metal_mesh_runtime.rs");
        let perspective_device_test =
            RUNTIME_TEST_SOURCE.contains("perspective_mesh_bytes_match_raster");
        let serial_device_test =
            RUNTIME_TEST_SOURCE.contains("serial_force_mesh_bytes_match_raster");

        assert_eq!(
            MESH_PERSPECTIVE_ENTRY_POINTS,
            ["mesh_raster_select", "mesh_raster_emit_perspective"]
        );
        assert_eq!(
            MESH_PERSPECTIVE_SERIAL_ENTRY_POINTS,
            ["mesh_raster_serial_perspective"]
        );
        assert_eq!(MESH_SERIAL_ENTRY_POINTS, ["mesh_raster_serial"]);

        println!("A11_HOST_CAPABILITY perspective_host=wired");
        println!(
            "A11_HOST_CAPABILITY perspective_device_test={}",
            if perspective_device_test {
                "present"
            } else {
                "absent"
            }
        );
        println!("A11_HOST_CAPABILITY serial_fallback=wired");
        println!(
            "A11_HOST_CAPABILITY serial_device_test={}",
            if serial_device_test {
                "present"
            } else {
                "absent"
            }
        );
    }

    #[test]
    fn gate_keeps_pf_and_strict_threshold() {
        assert!(check_gate(90023, 0.002).is_ok());
        assert!(check_gate(90024, 0.002).is_ok());
        assert!(check_gate(21024, 0.002).is_ok());
        assert_eq!(check_gate(1, 0.002), Err(MetalRefusal::UnsupportedPf(1)));
        assert_eq!(check_gate(90023, ACTIVE_GATE), Err(MetalRefusal::Inactive));
        assert_eq!(check_gate(90023, f32::NAN), Err(MetalRefusal::Inactive));
    }

    #[test]
    fn smooth_shape_finite_and_identity_are_refused() {
        let (source, base, residual, mask, skin) = one_pixel();
        let short: [f32; 3] = [0.0; 3];
        let wide_mask: [f32; 4] = [1.5, 0.0, 1.0, 0.0];
        let mut bad = source;
        let mut input = SmoothInput {
            source: &source,
            mask_rgba: &mask,
            skin_mask: &skin,
            base: &base,
            residual: &residual,
            width: 1,
            height: 1,
            stride: 1,
            pf: 90023,
            pf_value: 0.5,
            ticket: 1,
            uniforms: test_uniforms(),
        };
        assert!(input.validate().is_ok());

        input.stride = 0;
        assert_eq!(
            input.validate(),
            Err(MetalRefusal::ShapeMismatch("geometry"))
        );
        input.stride = 1;

        input.base = &short;
        assert_eq!(input.validate(), Err(MetalRefusal::ShapeMismatch("base")));
        input.base = &base;

        input.skin_mask = &[];
        assert_eq!(
            input.validate(),
            Err(MetalRefusal::ShapeMismatch("skin_mask"))
        );
        input.skin_mask = &skin;

        input.mask_rgba = &wide_mask;
        assert_eq!(input.validate(), Err(MetalRefusal::ValueOutOfRange("mask")));
        input.mask_rgba = &mask;

        bad[2] = f32::INFINITY;
        input.source = &bad;
        assert_eq!(
            input.validate(),
            Err(MetalRefusal::ValueOutOfRange("pixel"))
        );
        input.source = &source;

        input.ticket = 0;
        assert_eq!(input.validate(), Err(MetalRefusal::MissingSameJobIdentity));
        input.ticket = 1;

        input.pf = 7;
        assert_eq!(input.validate(), Err(MetalRefusal::UnsupportedPf(7)));
        input.pf = 90023;

        input.pf_value = 0.0;
        assert_eq!(input.validate(), Err(MetalRefusal::Inactive));
        input.pf_value = 7.5;
        assert!(input.validate().is_ok());
    }

    #[test]
    fn wide_stride_is_accepted_but_short_plane_is_not() {
        let mut source = vec![0.0f32; 2 * 4 * 4];
        let base = vec![0.0f32; 8 * 4];
        let residual = vec![0.0f32; 8 * 4];
        let mask = vec![0.0f32; 8 * 4];
        let skin = vec![1.0f32; 8];
        source[..8].copy_from_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let input = SmoothInput {
            source: &source,
            mask_rgba: &mask,
            skin_mask: &skin,
            base: &base,
            residual: &residual,
            width: 1,
            height: 2,
            stride: 4,
            pf: 90023,
            pf_value: 1.0,
            ticket: 1,
            uniforms: test_uniforms(),
        };
        assert!(input.validate().is_ok());
        let short = SmoothInput {
            source: &source[..8],
            ..input
        };
        assert_eq!(short.validate(), Err(MetalRefusal::ShapeMismatch("source")));
    }

    #[test]
    fn alpha_restore_is_bit_exact_for_both_formats() {
        let mut floats = vec![0.1, 0.2, 0.3, 0.0, 0.4, 0.5, 0.6, 0.0];
        let source = [0.1f32, 0.2, 0.3, 0.123_456_79, 0.4, 0.5, 0.6, 0.987_654_3];
        restore_alpha_f32(&mut floats, &source);
        assert_eq!(floats[3].to_bits(), source[3].to_bits());
        assert_eq!(floats[7].to_bits(), source[7].to_bits());

        let mut bytes = vec![1u8, 2, 3, 0, 4, 5, 6, 0];
        let source = [1u8, 2, 3, 77, 4, 5, 6, 255];
        restore_alpha_bgra(&mut bytes, &source);
        assert_eq!(bytes, vec![1, 2, 3, 77, 4, 5, 6, 255]);
    }

    #[test]
    fn whiten_layout_and_lengths_are_enforced() {
        let bgra = vec![0u8; 4];
        let mask = vec![255u8; 1];
        let lut = vec![0.0f32; LUT_FLOATS];
        let mut input = WhitenInput {
            bgra: &bgra,
            width: 1,
            height: 1,
            stride_bytes: 4,
            mask: &mask,
            mask_x: 0,
            mask_y: 0,
            mask_width: 1,
            mask_height: 1,
            lut: &lut,
            alpha: 1.0,
            flags: 0,
            ticket: 1,
        };
        assert!(input.validate().is_ok());
        assert_eq!(LUT_BYTES, 4_194_304);

        input.lut = &lut[..LUT_FLOATS - 1];
        assert_eq!(input.validate(), Err(MetalRefusal::ShapeMismatch("lut")));
        input.lut = &lut;

        input.stride_bytes = 3;
        assert_eq!(
            input.validate(),
            Err(MetalRefusal::ShapeMismatch("source_bgra"))
        );
        input.stride_bytes = 4;

        input.mask_x = 1;
        assert_eq!(
            input.validate(),
            Err(MetalRefusal::ShapeMismatch("mask_rect"))
        );
        input.mask_x = 0;

        input.alpha = f32::NAN;
        assert_eq!(
            input.validate(),
            Err(MetalRefusal::ValueOutOfRange("alpha"))
        );
        input.alpha = 1.0;

        input.ticket = 0;
        assert_eq!(input.validate(), Err(MetalRefusal::MissingSameJobIdentity));
    }

    #[test]
    fn whiten_refuses_out_of_range_alpha_and_non_finite_lut() {
        let bgra = vec![0u8; 4];
        let mask = vec![255u8; 1];
        let lut = vec![0.0f32; LUT_FLOATS];
        let mut input = WhitenInput {
            bgra: &bgra,
            width: 1,
            height: 1,
            stride_bytes: 4,
            mask: &mask,
            mask_x: 0,
            mask_y: 0,
            mask_width: 1,
            mask_height: 1,
            lut: &lut,
            alpha: 0.0,
            flags: 0,
            ticket: 1,
        };

        for alpha in [0.0, 1.0] {
            input.alpha = alpha;
            assert!(input.validate().is_ok());
        }
        for alpha in [f32::NAN, f32::INFINITY, -0.001, 1.001, 2.0] {
            input.alpha = alpha;
            assert_eq!(
                input.validate(),
                Err(MetalRefusal::ValueOutOfRange("alpha"))
            );
        }
        input.alpha = 1.0;

        {
            let mut non_finite = vec![0.0f32; LUT_FLOATS];
            non_finite[0] = f32::NAN;
            let bad = WhitenInput {
                lut: &non_finite,
                ..input
            };
            assert_eq!(bad.validate(), Err(MetalRefusal::ValueOutOfRange("lut")));
        }
        {
            let mut non_finite = vec![0.0f32; LUT_FLOATS];
            non_finite[LUT_FLOATS - 1] = f32::INFINITY;
            let bad = WhitenInput {
                lut: &non_finite,
                ..input
            };
            assert_eq!(bad.validate(), Err(MetalRefusal::ValueOutOfRange("lut")));
        }
    }

    #[test]
    fn whiten_ticket_is_nonzero_identity() {
        let bgra = vec![0u8; 4];
        let mask = vec![255u8; 1];
        let lut = vec![0.0f32; LUT_FLOATS];
        let mut input = WhitenInput {
            bgra: &bgra,
            width: 1,
            height: 1,
            stride_bytes: 4,
            mask: &mask,
            mask_x: 0,
            mask_y: 0,
            mask_width: 1,
            mask_height: 1,
            lut: &lut,
            alpha: 1.0,
            flags: 0,
            ticket: 1,
        };

        assert!(input.validate().is_ok());
        input.ticket = 0;
        assert_eq!(input.validate(), Err(MetalRefusal::MissingSameJobIdentity));
        input.ticket = u64::MAX;
        assert!(input.validate().is_ok());
    }

    #[test]
    fn projection_rejects_bad_rectangles() {
        let mask = vec![255u8; 4];
        let mut input = ProjectionInput {
            source_mask: &mask,
            source_mask_width: 2,
            source_mask_height: 2,
            surface_width: 8,
            surface_height: 8,
            destination_x: 0,
            destination_y: 0,
            destination_width: 2,
            destination_height: 2,
            source_rect_x: 0,
            source_rect_y: 0,
            source_rect_width: 2,
            source_rect_height: 2,
            domain_x0: 0,
            domain_y0: 0,
            roi_width: 2,
            roi_height: 2,
            ticket: 1,
        };
        assert!(input.validate().is_ok());
        assert_eq!(std::mem::size_of_val(&input.params()), 80);
        input.destination_width = 9;
        assert_eq!(
            input.validate(),
            Err(MetalRefusal::ShapeMismatch("projection_rect"))
        );
        input.destination_width = 2;
        input.source_mask = &mask[..3];
        assert_eq!(
            input.validate(),
            Err(MetalRefusal::ShapeMismatch("source_mask"))
        );
    }

    #[test]
    fn embedded_msl_is_the_reused_source() {
        assert_eq!(P15_MSL.len(), 13_511);
        assert_eq!(WHITEN_MSL.len(), 8_940);
        assert_eq!(P15_MSL.matches("kernel void ").count(), 6);
        assert_eq!(WHITEN_MSL.matches("kernel void ").count(), 2);
        assert!(P15_MSL.contains("#pragma clang fp contract(off)"));
        assert!(WHITEN_MSL.contains("#pragma clang fp contract(off)"));
        assert!(P15_MSL.contains("static_assert(sizeof(P15FacialSmoothUniforms) == 88"));
        assert!(P15_MSL.contains("p15_facial_smooth_composite"));
        assert!(P15_MSL.contains("[[buffer(6)]]"));
        assert!(WHITEN_MSL.contains("static_assert(sizeof(O3MaskedWhitenParams) == 48"));
        assert!(WHITEN_MSL.contains("static_assert(sizeof(O3MaskProjectionParams) == 80"));
        assert!(WHITEN_MSL.contains("o3_masked_lut_whiten"));
        assert!(WHITEN_MSL.contains("[[buffer(4)]]"));
    }

    #[test]
    fn mesh_last_overlap_uses_zero_fill() {
        assert_eq!(
            selected_fill(metal_mesh::OverlapCode::First as i32, 3),
            vec![0xFF; 12]
        );
        assert_eq!(
            selected_fill(metal_mesh::OverlapCode::Last as i32, 3),
            vec![0; 12]
        );
    }

    #[test]
    fn p15_rgba8_packing_uses_half_away_rounding() {
        let pixels = [0.0, 0.5 / 255.0, 1.0 / 255.0, 1.0];
        assert_eq!(quantize_rgba8(&pixels), vec![0, 1, 1, 255]);

        let over = [f32::NAN, -1.0, 0.5, 2.0];
        assert_eq!(quantize_rgba8(&over), vec![0, 0, 128, 255]);
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn non_macos_host_reports_unavailable_without_dispatch() {
        assert!(!available());
        assert!(!framework_available());
        assert_eq!(
            MetalContext::new().err(),
            Some(MetalRefusal::HostUnavailable)
        );
        let (source, base, residual, mask, skin) = one_pixel();
        let input = SmoothInput {
            source: &source,
            mask_rgba: &mask,
            skin_mask: &skin,
            base: &base,
            residual: &residual,
            width: 1,
            height: 1,
            stride: 1,
            pf: 90023,
            pf_value: 0.5,
            ticket: 1,
            uniforms: test_uniforms(),
        };
        assert_eq!(
            execute_facial_smooth(&input),
            Err(MetalRefusal::HostUnavailable)
        );
    }
}
