//! Apple Metal geometry-stage candidate for the generic mesh rasterizer.
//!
//! This module owns only host-side contract work: the strict 36-byte
//! `MeshRasterParams` ABI, buffer binding indexes, scratch/out initialization,
//! input validation and CPU golden access. The MSL is a two-pass port of
//! `native/cuda/mesh_raster.cu`, which in turn matches `src/raster.rs`:
//!
//! * `dest` is framebuffer pixels, top-left origin, pixel centers at
//!   `(x + 0.5, y + 0.5)`.
//! * `source_uv` is normalized, row 0 at `v = 0`.
//! * `ebo` is a `u16` triangle list; its order resolves overlap.
//! * `area2 = cross(p1 - p0, p2 - p0)`, `|area2| <= 1e-12` is degenerate.
//! * Uncovered pixels keep the prefilled source bytes; alpha is straight and
//!   sampled/quantized exactly like RGB.
//!
//! `MetalContext::mesh_raster` executes the affine RGBA8 buffer candidate. An
//! additive `PerspectiveMeshInput` contract carries `[depth, reciprocal_w]`
//! per vertex and has matching `mesh_raster_*_perspective` MSL entry points,
//! but `metal.rs` does not dispatch them yet. Neither path has
//! `MTLTexture`/`MTLSamplerState` or an FBO render target. Requesting
//! `MeshContract::VendorForwardMesh` is refused before dispatch; never treat
//! either candidate as PF8025/PF8033 vendor parity.

use crate::raster::{self, Cull, Edge, Overlap, Rules, Sampler, Winding, Wrap};
use crate::{Image, Refusal};

pub const MESH_RASTER_ABI_VERSION: u32 = 1;
pub const MESH_MSL: &str = include_str!("../native/metal/mesh_raster.metal");
pub const MESH_SERIAL_MSL: &str = include_str!("../native/metal/mesh_raster_serial.metal");
pub const MESH_PERSPECTIVE_EMIT_ENTRY: &str = "mesh_raster_emit_perspective";
pub const MESH_PERSPECTIVE_SERIAL_ENTRY: &str = "mesh_raster_serial_perspective";
pub const DEGENERATE_AREA2: f32 = 1e-12;
pub const UNCOVERED: u32 = u32::MAX;

/// Same numeric values as `native/cuda/mesh_raster.h`.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindingCode {
    PositiveAreaIsFront = 1,
    NegativeAreaIsFront = 2,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CullCode {
    None = 1,
    Front = 2,
    Back = 3,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeCode {
    Exclusive = 1,
    Inclusive = 2,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlapCode {
    Reject = 1,
    First = 2,
    Last = 3,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WrapCode {
    ClampToEdge = 0,
    Repeat = 1,
}

/// 16 bytes, field-for-field `MeshRasterRules`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshRasterRules {
    pub winding: i32,
    pub cull: i32,
    pub edge: i32,
    pub overlap: i32,
}

/// 36 bytes. Buffer 2 for select, buffer 5 for emit.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshRasterParams {
    pub width: i32,
    pub height: i32,
    pub vertex_count: i32,
    pub triangle_count: i32,
    pub wrap: i32,
    pub rules: MeshRasterRules,
}

const _: () = assert!(std::mem::size_of::<MeshRasterRules>() == 16);
const _: () = assert!(std::mem::size_of::<MeshRasterParams>() == 36);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub index: u32,
    pub name: &'static str,
    pub element: &'static str,
    pub bytes_per_element: u32,
}

pub const SELECT_BINDINGS: [Binding; 5] = [
    Binding {
        index: 0,
        name: "dest",
        element: "float2",
        bytes_per_element: 8,
    },
    Binding {
        index: 1,
        name: "ebo",
        element: "uint16",
        bytes_per_element: 2,
    },
    Binding {
        index: 2,
        name: "params",
        element: "MeshRasterParams",
        bytes_per_element: 36,
    },
    Binding {
        index: 3,
        name: "selected",
        element: "atomic_uint",
        bytes_per_element: 4,
    },
    Binding {
        index: 4,
        name: "coverage",
        element: "atomic_uint",
        bytes_per_element: 4,
    },
];

pub const EMIT_BINDINGS: [Binding; 8] = [
    Binding {
        index: 0,
        name: "dest",
        element: "float2",
        bytes_per_element: 8,
    },
    Binding {
        index: 1,
        name: "source_uv",
        element: "float2",
        bytes_per_element: 8,
    },
    Binding {
        index: 2,
        name: "ebo",
        element: "uint16",
        bytes_per_element: 2,
    },
    Binding {
        index: 3,
        name: "source",
        element: "uint8 RGBA",
        bytes_per_element: 1,
    },
    Binding {
        index: 4,
        name: "out",
        element: "uint8 RGBA",
        bytes_per_element: 1,
    },
    Binding {
        index: 5,
        name: "params",
        element: "MeshRasterParams",
        bytes_per_element: 36,
    },
    Binding {
        index: 6,
        name: "selected",
        element: "uint",
        bytes_per_element: 4,
    },
    Binding {
        index: 7,
        name: "coverage",
        element: "uint",
        bytes_per_element: 4,
    },
];

pub const PERSPECTIVE_EMIT_BINDINGS: [Binding; 9] = [
    EMIT_BINDINGS[0],
    EMIT_BINDINGS[1],
    EMIT_BINDINGS[2],
    EMIT_BINDINGS[3],
    EMIT_BINDINGS[4],
    EMIT_BINDINGS[5],
    EMIT_BINDINGS[6],
    EMIT_BINDINGS[7],
    Binding {
        index: 8,
        name: "vertex_perspective",
        element: "float2 [depth, reciprocal_w]",
        bytes_per_element: 8,
    },
];

pub const PERSPECTIVE_SERIAL_BINDINGS: [Binding; 7] = [
    Binding {
        index: 0,
        name: "dest",
        element: "float2",
        bytes_per_element: 8,
    },
    Binding {
        index: 1,
        name: "source_uv",
        element: "float2",
        bytes_per_element: 8,
    },
    Binding {
        index: 2,
        name: "ebo",
        element: "uint16",
        bytes_per_element: 2,
    },
    Binding {
        index: 3,
        name: "source",
        element: "uint8 RGBA",
        bytes_per_element: 1,
    },
    Binding {
        index: 4,
        name: "out",
        element: "uint8 RGBA",
        bytes_per_element: 1,
    },
    Binding {
        index: 5,
        name: "params",
        element: "MeshRasterParams",
        bytes_per_element: 36,
    },
    Binding {
        index: 6,
        name: "vertex_perspective",
        element: "float2 [depth, reciprocal_w]",
        bytes_per_element: 8,
    },
];

/// The MSL binds no textures; bilinear taps read `device const uchar*`.
pub const TEXTURE_BINDINGS: &[&str] = &[];

/// Vendor mesh families whose forward-mesh contract this candidate cannot meet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VendorMeshEffect {
    Pf8025,
    Pf8033,
}

impl VendorMeshEffect {
    pub const fn pf(self) -> u32 {
        match self {
            Self::Pf8025 => 8025,
            Self::Pf8033 => 8033,
        }
    }
}

/// Explicit caller intent. Only the affine buffer candidate is executable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshContract {
    /// Host/MSL buffer path with affine source UV. Not vendor parity.
    AffineBufferCandidate,
    /// Request the vendor forward-mesh contract; currently refused.
    VendorForwardMesh(VendorMeshEffect),
}

/// Typed reason the requested contract cannot run on this candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshContractRefusal {
    VendorForwardMeshUnsupported {
        effect: VendorMeshEffect,
        missing_perspective_uv: bool,
        missing_texture_sampler: bool,
        missing_fbo: bool,
    },
}

/// Static verification state for a backend capability. This module never
/// probes a Metal device, so newly declared vendor-parity requirements stay
/// `Unverified` until a macOS gate proves them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerificationState {
    Unverified,
    Verified,
}

/// What the mesh backend can and cannot provide. The affine buffer path is
/// executable, but it is not an `MTLTexture`/`MTLSamplerState`/FBO render path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshBackendCapabilities {
    /// `true` only when a real `MTLTexture` source/output exists.
    pub has_mtl_texture: bool,
    /// `true` only when a real `MTLSamplerState` controls filter/wrap/sRGB.
    pub has_mtl_sampler_state: bool,
    /// `true` only when the output is an FBO/render-target texture.
    pub has_fbo_render_target: bool,
    /// Requested vendor-equivalent format is RGBA8 UNORM; device conversion is
    /// not verified by this module.
    pub rgba8_unorm: VerificationState,
    /// Filter, wrap, mip and sRGB mapping are not verified by this module.
    pub sampler_filter_wrap_srgb: VerificationState,
    /// The FBO internal precision is not verified by this module.
    pub fbo_precision: VerificationState,
}

impl MeshBackendCapabilities {
    pub const AFFINE_BUFFER: Self = Self {
        has_mtl_texture: false,
        has_mtl_sampler_state: false,
        has_fbo_render_target: false,
        rgba8_unorm: VerificationState::Unverified,
        sampler_filter_wrap_srgb: VerificationState::Unverified,
        fbo_precision: VerificationState::Unverified,
    };

    pub const fn refusal(self) -> MeshBackendRefusal {
        MeshBackendRefusal::AffineBufferOnly(self)
    }
}

/// Typed backend gap payload. It deliberately carries the whole capability
/// set so a caller can inspect every missing/unverified item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshBackendRefusal {
    AffineBufferOnly(MeshBackendCapabilities),
}

impl MeshContractRefusal {
    /// Vendor-parity backend gaps; the affine candidate itself may still run.
    pub const fn backend_refusal(self) -> MeshBackendRefusal {
        match self {
            Self::VendorForwardMeshUnsupported { .. } => {
                MeshBackendCapabilities::AFFINE_BUFFER.refusal()
            }
        }
    }
}

impl MeshContract {
    pub fn validate(self) -> Result<(), MeshContractRefusal> {
        match self {
            Self::AffineBufferCandidate => Ok(()),
            Self::VendorForwardMesh(effect) => {
                Err(MeshContractRefusal::VendorForwardMeshUnsupported {
                    effect,
                    missing_perspective_uv: true,
                    missing_texture_sampler: true,
                    missing_fbo: true,
                })
            }
        }
    }
}

impl From<MeshContractRefusal> for Refusal {
    fn from(refusal: MeshContractRefusal) -> Self {
        match refusal {
            MeshContractRefusal::VendorForwardMeshUnsupported { effect, .. } => {
                Refusal::NotImplemented(effect.pf())
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MeshRasterInput<'a> {
    pub dest: &'a [[f32; 2]],
    pub source_uv: &'a [[f32; 2]],
    pub ebo: &'a [u16],
    pub source: &'a Image,
    pub rules: Rules,
    pub sampler: Sampler,
    /// `AffineBufferCandidate` is the only executable mode. Vendor forward
    /// mesh requests are refused before any CPU/GPU pixel work.
    pub contract: MeshContract,
}

impl MeshRasterInput<'_> {
    pub fn validate_contract(&self) -> Result<(), MeshContractRefusal> {
        self.contract.validate()
    }

    /// Vendor-parity gaps; this does not reject the affine candidate.
    pub const fn backend_refusal(&self) -> MeshBackendRefusal {
        MeshBackendCapabilities::AFFINE_BUFFER.refusal()
    }

    /// Mirrors `raster::validate` so a Metal launch refuses exactly what the
    /// CPU reference refuses. No landmark, EBO or GL state is inferred.
    pub fn validate(&self) -> Result<(), Refusal> {
        self.validate_contract()?;
        self.source.validate()?;
        if self.dest.len() != self.source_uv.len() {
            return Err(Refusal::ShapeMismatch("source_uv"));
        }
        if self.dest.len() < 3 {
            return Err(Refusal::ShapeMismatch("dest"));
        }
        if self.ebo.is_empty() {
            return Err(Refusal::MissingInput("ebo"));
        }
        if self.ebo.len() % 3 != 0 {
            return Err(Refusal::ShapeMismatch("ebo"));
        }
        if self.dest.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Refusal::ValueOutOfRange("dest"));
        }
        if self.source_uv.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Refusal::ValueOutOfRange("source_uv"));
        }
        if self.ebo.iter().any(|&i| i as usize >= self.dest.len()) {
            return Err(Refusal::ValueOutOfRange("ebo index"));
        }
        if [
            self.dest.len(),
            self.ebo.len() / 3,
            self.source.width as usize,
            self.source.height as usize,
        ]
        .iter()
        .any(|value| *value > i32::MAX as usize)
        {
            return Err(Refusal::ValueOutOfRange("dimensions"));
        }
        Ok(())
    }

    pub fn params(&self) -> Result<MeshRasterParams, Refusal> {
        self.validate()?;
        Ok(MeshRasterParams {
            width: self.source.width as i32,
            height: self.source.height as i32,
            vertex_count: self.dest.len() as i32,
            triangle_count: (self.ebo.len() / 3) as i32,
            wrap: wrap_code(self.sampler.wrap) as i32,
            rules: rules_code(self.rules),
        })
    }

    /// CPU golden through the mounted `raster.rs` reference.
    pub fn cpu_golden(&self) -> Result<Image, Refusal> {
        self.validate()?;
        raster::rasterize(
            self.source,
            raster::Mesh {
                dest: self.dest,
                source_uv: self.source_uv,
                ebo: self.ebo,
            },
            self.rules,
            self.sampler,
        )
    }
}

/// Additive perspective candidate. `vertex_perspective[i]` is
/// `[depth, reciprocal_w]` for vertex `i`; `base` remains the affine input
/// contract and its 36-byte params ABI is unchanged.
#[derive(Clone, Copy, Debug)]
pub struct PerspectiveMeshInput<'a> {
    pub base: MeshRasterInput<'a>,
    pub vertex_perspective: &'a [[f32; 2]],
}

/// Barycentric perspective sample used by the host contract tests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerspectiveSample {
    pub u: f32,
    pub v: f32,
    pub depth: f32,
}

impl PerspectiveMeshInput<'_> {
    pub fn validate_contract(&self) -> Result<(), MeshContractRefusal> {
        self.base.validate_contract()
    }

    pub fn validate(&self) -> Result<(), Refusal> {
        self.base.validate()?;
        if self.vertex_perspective.len() != self.base.dest.len() {
            return Err(Refusal::ShapeMismatch("vertex_perspective"));
        }
        if self
            .vertex_perspective
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(Refusal::ValueOutOfRange("vertex_perspective"));
        }
        if self.vertex_perspective.iter().any(|value| value[1] <= 0.0) {
            return Err(Refusal::ValueOutOfRange("vertex_inv_w"));
        }
        Ok(())
    }

    pub fn params(&self) -> Result<MeshRasterParams, Refusal> {
        self.validate()?;
        self.base.params()
    }

    pub fn perspective_flat(&self) -> Vec<f32> {
        self.vertex_perspective
            .iter()
            .flat_map(|value| value.iter().copied())
            .collect()
    }
}

/// Perspective-correct barycentric sample:
/// `sum(b_i * uv_i / w_i) / sum(b_i / w_i)`.
pub fn perspective_uv(
    uv: [[f32; 2]; 3],
    vertex_perspective: [[f32; 2]; 3],
    weights: [f32; 3],
) -> Option<PerspectiveSample> {
    if uv.iter().flatten().any(|value| !value.is_finite())
        || vertex_perspective
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        || weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || vertex_perspective.iter().any(|value| value[1] <= 0.0)
    {
        return None;
    }

    let q0 = weights[0] * vertex_perspective[0][1];
    let q1 = weights[1] * vertex_perspective[1][1];
    let q2 = weights[2] * vertex_perspective[2][1];
    let inv_w = q0 + q1 + q2;
    if !inv_w.is_finite() || inv_w <= 0.0 {
        return None;
    }

    let u = (q0 * uv[0][0] + q1 * uv[1][0] + q2 * uv[2][0]) / inv_w;
    let v = (q0 * uv[0][1] + q1 * uv[1][1] + q2 * uv[2][1]) / inv_w;
    let depth = (q0 * vertex_perspective[0][0]
        + q1 * vertex_perspective[1][0]
        + q2 * vertex_perspective[2][0])
        / inv_w;
    if !u.is_finite() || !v.is_finite() || !depth.is_finite() {
        return None;
    }
    Some(PerspectiveSample { u, v, depth })
}

pub fn rules_code(rules: Rules) -> MeshRasterRules {
    MeshRasterRules {
        winding: winding_code(rules.winding) as i32,
        cull: cull_code(rules.cull) as i32,
        edge: edge_code(rules.edge) as i32,
        overlap: overlap_code(rules.overlap) as i32,
    }
}

pub fn winding_code(winding: Winding) -> WindingCode {
    match winding {
        Winding::PositiveAreaIsFront => WindingCode::PositiveAreaIsFront,
        Winding::NegativeAreaIsFront => WindingCode::NegativeAreaIsFront,
    }
}

pub fn cull_code(cull: Cull) -> CullCode {
    match cull {
        Cull::None => CullCode::None,
        Cull::Front => CullCode::Front,
        Cull::Back => CullCode::Back,
    }
}

pub fn edge_code(edge: Edge) -> EdgeCode {
    match edge {
        Edge::Exclusive => EdgeCode::Exclusive,
        Edge::Inclusive => EdgeCode::Inclusive,
    }
}

pub fn overlap_code(overlap: Overlap) -> OverlapCode {
    match overlap {
        Overlap::Reject => OverlapCode::Reject,
        Overlap::First => OverlapCode::First,
        Overlap::Last => OverlapCode::Last,
    }
}

pub fn wrap_code(wrap: Wrap) -> WrapCode {
    match wrap {
        Wrap::ClampToEdge => WrapCode::ClampToEdge,
        Wrap::Repeat => WrapCode::Repeat,
    }
}

/// Deterministic caller-owned scratch: `selected = UNCOVERED`, `coverage = 0`.
pub fn new_scratch(pixel_count: usize) -> (Vec<u32>, Vec<u32>) {
    (vec![UNCOVERED; pixel_count], vec![0u32; pixel_count])
}

/// The emit pass writes only covered pixels, so `out` starts as the source.
pub fn prefill_out(source: &Image) -> Vec<u8> {
    source.pixels.clone()
}

/// Future real-device comparison record. `max_abs` alone is not acceptance;
/// both values must be reported per run instead of assuming a 1e-5 line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiffStats {
    pub max_abs: f32,
    pub max_ulp: u32,
}

pub fn diff_stats(actual: &[f32], expected: &[f32]) -> DiffStats {
    let mut max_abs = 0.0f32;
    let mut max_ulp = 0u32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        let diff = (a - e).abs();
        if !diff.is_finite() || !a.is_finite() || !e.is_finite() {
            return DiffStats {
                max_abs: f32::INFINITY,
                max_ulp: u32::MAX,
            };
        }
        max_abs = max_abs.max(diff);
        let ulp = (ordered_bits(*a) - ordered_bits(*e)).unsigned_abs();
        max_ulp = max_ulp.max(ulp.min(u32::MAX as u64) as u32);
    }
    DiffStats { max_abs, max_ulp }
}

fn ordered_bits(value: f32) -> i64 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        -((bits & 0x7fff_ffff) as i64)
    } else {
        bits as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> Image {
        let mut pixels = Vec::new();
        for y in 0..4u32 {
            for x in 0..4u32 {
                let i = y * 4 + x;
                pixels.extend_from_slice(&[
                    (x * 16 + 4) as u8,
                    (y * 16 + 8) as u8,
                    (i * 4 + 2) as u8,
                    64 + (i * 3) as u8,
                ]);
            }
        }
        Image::new(4, 4, pixels).unwrap()
    }

    fn rules() -> Rules {
        Rules {
            winding: Winding::PositiveAreaIsFront,
            cull: Cull::None,
            edge: Edge::Inclusive,
            overlap: Overlap::First,
        }
    }

    fn input<'a>(
        source: &'a Image,
        dest: &'a [[f32; 2]],
        uv: &'a [[f32; 2]],
        ebo: &'a [u16],
    ) -> MeshRasterInput<'a> {
        MeshRasterInput {
            dest,
            source_uv: uv,
            ebo,
            source,
            rules: rules(),
            sampler: Sampler {
                wrap: Wrap::ClampToEdge,
            },
            contract: MeshContract::AffineBufferCandidate,
        }
    }

    #[test]
    fn abi_sizes_are_pinned() {
        assert_eq!(std::mem::size_of::<MeshRasterRules>(), 16);
        assert_eq!(std::mem::size_of::<MeshRasterParams>(), 36);
        let params = MeshRasterParams {
            width: 4,
            height: 4,
            vertex_count: 3,
            triangle_count: 1,
            wrap: 0,
            rules: rules_code(rules()),
        };
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const MeshRasterParams as *const u8,
                std::mem::size_of::<MeshRasterParams>(),
            )
        };
        assert_eq!(i32::from_le_bytes(bytes[0..4].try_into().unwrap()), 4);
        assert_eq!(i32::from_le_bytes(bytes[8..12].try_into().unwrap()), 3);
        assert_eq!(i32::from_le_bytes(bytes[16..20].try_into().unwrap()), 0);
        assert_eq!(i32::from_le_bytes(bytes[20..24].try_into().unwrap()), 1);
    }

    #[test]
    fn binding_tables_are_contiguous_and_texture_free() {
        for table in [
            SELECT_BINDINGS.as_slice(),
            EMIT_BINDINGS.as_slice(),
            PERSPECTIVE_EMIT_BINDINGS.as_slice(),
            PERSPECTIVE_SERIAL_BINDINGS.as_slice(),
        ] {
            for (index, binding) in table.iter().enumerate() {
                assert_eq!(binding.index as usize, index);
            }
        }
        assert_eq!(SELECT_BINDINGS[2].bytes_per_element, 36);
        assert_eq!(EMIT_BINDINGS[5].bytes_per_element, 36);
        assert_eq!(PERSPECTIVE_EMIT_BINDINGS[8].name, "vertex_perspective");
        assert_eq!(PERSPECTIVE_SERIAL_BINDINGS[6].name, "vertex_perspective");
        assert!(TEXTURE_BINDINGS.is_empty());
    }

    #[test]
    fn validation_matches_raster_contract() {
        let image = source();
        let dest = [[0.0f32, 0.0], [8.0, 0.0], [0.0, 8.0]];
        let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let ebo = [0u16, 1, 2];
        let valid = input(&image, &dest, &uv, &ebo);
        assert_eq!(valid.validate_contract(), Ok(()));
        assert!(valid.validate().is_ok());
        assert_eq!(valid.params().unwrap().triangle_count, 1);

        let short_uv = input(&image, &dest, &uv[..2], &ebo);
        assert_eq!(
            short_uv.validate(),
            Err(Refusal::ShapeMismatch("source_uv"))
        );
        let bad_ebo = input(&image, &dest, &uv, &[0u16, 1]);
        assert_eq!(bad_ebo.validate(), Err(Refusal::ShapeMismatch("ebo")));
        let empty_ebo = input(&image, &dest, &uv, &[]);
        assert_eq!(empty_ebo.validate(), Err(Refusal::MissingInput("ebo")));
        let oob = input(&image, &dest, &uv, &[0u16, 1, 3]);
        assert_eq!(oob.validate(), Err(Refusal::ValueOutOfRange("ebo index")));
        let mut bad_dest = dest;
        bad_dest[1][0] = f32::NAN;
        assert_eq!(
            input(&image, &bad_dest, &uv, &ebo).validate(),
            Err(Refusal::ValueOutOfRange("dest"))
        );
        let malformed = Image {
            width: 4,
            height: 4,
            pixels: vec![0; 8],
        };
        assert_eq!(
            input(&malformed, &dest, &uv, &ebo).validate(),
            Err(Refusal::ShapeMismatch("image"))
        );
    }

    #[test]
    fn perspective_uv_is_affine_when_reciprocal_w_matches() {
        let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let vertex_perspective = [[0.0f32, 1.0], [0.0, 1.0], [0.0, 1.0]];
        let sample = perspective_uv(uv, vertex_perspective, [0.25, 0.25, 0.5]).unwrap();
        assert!((sample.u - 0.25).abs() < 1e-6);
        assert!((sample.v - 0.5).abs() < 1e-6);
        assert_eq!(sample.depth, 0.0);
    }

    #[test]
    fn perspective_uv_divides_by_inverse_w_and_rejects_bad_identity() {
        let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let vertex_perspective = [[0.0f32, 1.0], [0.0, 2.0], [0.0, 1.0]];
        let sample = perspective_uv(uv, vertex_perspective, [0.25, 0.25, 0.5]).unwrap();
        assert!((sample.u - 0.4).abs() < 1e-6);
        assert!((sample.v - 0.4).abs() < 1e-6);
        assert_eq!(sample.depth, 0.0);

        let zero_w = [[0.0f32, 0.0], [0.0, 1.0], [0.0, 1.0]];
        assert_eq!(perspective_uv(uv, zero_w, [0.25, 0.25, 0.5]), None);
        let nan_w = [[0.0f32, f32::NAN], [0.0, 1.0], [0.0, 1.0]];
        assert_eq!(perspective_uv(uv, nan_w, [0.25, 0.25, 0.5]), None);
    }

    #[test]
    fn perspective_input_validates_depth_and_reciprocal_w() {
        let image = source();
        let dest = [[0.0f32, 0.0], [8.0, 0.0], [0.0, 8.0]];
        let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let ebo = [0u16, 1, 2];
        let vertex_perspective = [[0.0f32, 1.0], [0.25, 2.0], [0.5, 1.0]];
        let base = input(&image, &dest, &uv, &ebo);

        let valid = PerspectiveMeshInput {
            base,
            vertex_perspective: &vertex_perspective,
        };
        assert!(valid.validate().is_ok());
        assert_eq!(valid.params().unwrap().vertex_count, 3);
        assert_eq!(
            valid.perspective_flat(),
            vec![0.0, 1.0, 0.25, 2.0, 0.5, 1.0]
        );

        let short = PerspectiveMeshInput {
            base,
            vertex_perspective: &vertex_perspective[..2],
        };
        assert_eq!(
            short.validate(),
            Err(Refusal::ShapeMismatch("vertex_perspective"))
        );

        let nan_depth = [[0.0f32, 1.0], [f32::NAN, 2.0], [0.5, 1.0]];
        let bad_depth = PerspectiveMeshInput {
            base,
            vertex_perspective: &nan_depth,
        };
        assert_eq!(
            bad_depth.validate(),
            Err(Refusal::ValueOutOfRange("vertex_perspective"))
        );

        let zero_w = [[0.0f32, 0.0], [0.25, 2.0], [0.5, 1.0]];
        let bad_w = PerspectiveMeshInput {
            base,
            vertex_perspective: &zero_w,
        };
        assert_eq!(
            bad_w.validate(),
            Err(Refusal::ValueOutOfRange("vertex_inv_w"))
        );
    }

    #[test]
    fn vendor_effect_ids_are_pinned() {
        assert_eq!(VendorMeshEffect::Pf8025.pf(), 8025);
        assert_eq!(VendorMeshEffect::Pf8033.pf(), 8033);
    }

    #[test]
    fn backend_capabilities_expose_texture_sampler_fbo_gaps() {
        let caps = MeshBackendCapabilities::AFFINE_BUFFER;
        assert!(!caps.has_mtl_texture);
        assert!(!caps.has_mtl_sampler_state);
        assert!(!caps.has_fbo_render_target);
        assert_eq!(caps.rgba8_unorm, VerificationState::Unverified);
        assert_eq!(caps.sampler_filter_wrap_srgb, VerificationState::Unverified);
        assert_eq!(caps.fbo_precision, VerificationState::Unverified);
        assert_eq!(caps.refusal(), MeshBackendRefusal::AffineBufferOnly(caps));

        let image = source();
        let dest = [[0.0f32, 0.0], [8.0, 0.0], [0.0, 8.0]];
        let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let vendor = MeshRasterInput {
            contract: MeshContract::VendorForwardMesh(VendorMeshEffect::Pf8033),
            ..input(&image, &dest, &uv, &[0u16, 1, 2])
        };
        let refusal = vendor.validate_contract().unwrap_err();
        assert_eq!(refusal.backend_refusal(), caps.refusal());
        assert_eq!(vendor.backend_refusal(), caps.refusal());
    }

    #[test]
    fn vendor_forward_mesh_is_refused_before_pixel_work() {
        let image = source();
        let dest = [[0.0f32, 0.0], [8.0, 0.0], [0.0, 8.0]];
        let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let ebo = [0u16, 1, 2];
        let affine = input(&image, &dest, &uv, &ebo);
        assert_eq!(affine.validate_contract(), Ok(()));

        for effect in [VendorMeshEffect::Pf8025, VendorMeshEffect::Pf8033] {
            let vendor = MeshRasterInput {
                contract: MeshContract::VendorForwardMesh(effect),
                ..input(&image, &dest, &uv, &ebo)
            };
            let expected = MeshContractRefusal::VendorForwardMeshUnsupported {
                effect,
                missing_perspective_uv: true,
                missing_texture_sampler: true,
                missing_fbo: true,
            };
            assert_eq!(vendor.validate_contract(), Err(expected));
            assert_eq!(vendor.validate(), Err(Refusal::NotImplemented(effect.pf())));
            assert_eq!(vendor.params(), Err(Refusal::NotImplemented(effect.pf())));
            assert_eq!(
                vendor.cpu_golden(),
                Err(Refusal::NotImplemented(effect.pf()))
            );
        }
    }

    #[test]
    fn perspective_vendor_contract_still_refuses_capabilities() {
        let image = source();
        let dest = [[0.0f32, 0.0], [8.0, 0.0], [0.0, 8.0]];
        let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let ebo = [0u16, 1, 2];
        let vertex_perspective = [[0.0f32, 1.0]; 3];
        let vendor = MeshRasterInput {
            contract: MeshContract::VendorForwardMesh(VendorMeshEffect::Pf8033),
            ..input(&image, &dest, &uv, &ebo)
        };
        let perspective = PerspectiveMeshInput {
            base: vendor,
            vertex_perspective: &vertex_perspective,
        };
        assert_eq!(
            perspective.validate_contract(),
            Err(MeshContractRefusal::VendorForwardMeshUnsupported {
                effect: VendorMeshEffect::Pf8033,
                missing_perspective_uv: true,
                missing_texture_sampler: true,
                missing_fbo: true,
            })
        );
        assert_eq!(perspective.validate(), Err(Refusal::NotImplemented(8033)));
    }

    #[test]
    fn rule_codes_match_cuda_header_values() {
        assert_eq!(winding_code(Winding::PositiveAreaIsFront) as i32, 1);
        assert_eq!(winding_code(Winding::NegativeAreaIsFront) as i32, 2);
        assert_eq!(cull_code(Cull::None) as i32, 1);
        assert_eq!(cull_code(Cull::Front) as i32, 2);
        assert_eq!(cull_code(Cull::Back) as i32, 3);
        assert_eq!(edge_code(Edge::Exclusive) as i32, 1);
        assert_eq!(edge_code(Edge::Inclusive) as i32, 2);
        assert_eq!(overlap_code(Overlap::Reject) as i32, 1);
        assert_eq!(overlap_code(Overlap::First) as i32, 2);
        assert_eq!(overlap_code(Overlap::Last) as i32, 3);
        assert_eq!(wrap_code(Wrap::ClampToEdge) as i32, 0);
        assert_eq!(wrap_code(Wrap::Repeat) as i32, 1);
    }

    #[test]
    fn scratch_and_out_start_from_the_documented_state() {
        let image = source();
        let (selected, coverage) = new_scratch(image.pixel_count());
        assert_eq!(selected, vec![UNCOVERED; 16]);
        assert_eq!(coverage, vec![0; 16]);
        assert_eq!(prefill_out(&image), image.pixels);
    }

    #[test]
    fn cpu_golden_uses_the_mounted_raster_reference() {
        let image = source();
        let dest = [[0.0f32, 0.0], [8.0, 0.0], [0.0, 8.0]];
        let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let ebo = [0u16, 1, 2];
        let golden = input(&image, &dest, &uv, &ebo).cpu_golden().unwrap();
        assert_eq!(golden.width, 4);
        assert_eq!(golden.height, 4);
        assert_ne!(golden, image);
        let direct = raster::rasterize(
            &image,
            raster::Mesh {
                dest: &dest,
                source_uv: &uv,
                ebo: &ebo,
            },
            rules(),
            Sampler {
                wrap: Wrap::ClampToEdge,
            },
        )
        .unwrap();
        assert_eq!(golden, direct);
    }

    #[test]
    fn embedded_msl_has_affine_and_perspective_paths() {
        assert!(MESH_MSL.contains("kernel void mesh_raster_select"));
        assert!(MESH_MSL.contains("kernel void mesh_raster_emit"));
        assert!(MESH_MSL.contains(MESH_PERSPECTIVE_EMIT_ENTRY));
        assert!(MESH_MSL.contains("device const float* vertex_perspective"));
        assert!(MESH_MSL.contains("static_assert(sizeof(MeshRasterParams) == 36"));
        assert!(MESH_MSL.contains("atomic_fetch_add_explicit"));
        assert_eq!(MESH_MSL.matches("if (coverage[pixel] == 0u)").count(), 2);
        assert!(MESH_MSL.contains("rint("));
        assert!(MESH_MSL.contains(MESH_PERSPECTIVE_SERIAL_ENTRY));
        assert_eq!(MESH_MSL.matches("kernel void ").count(), 4);
        assert_eq!(MESH_SERIAL_MSL.matches("kernel void ").count(), 1);
    }

    #[test]
    fn diff_stats_record_max_abs_and_ulp() {
        let one = 1.0f32;
        let next = f32::from_bits(one.to_bits() + 1);
        let stats = diff_stats(&[one, 2.0, -0.0, 4.0], &[next, 2.0, 0.0, 4.5]);
        assert_eq!(stats.max_ulp, 1 << 20);
        assert!((stats.max_abs - 0.5).abs() < 1e-6);
        assert_eq!(diff_stats(&[-0.0, 0.0], &[0.0, -0.0]).max_ulp, 0);
        let bad = diff_stats(&[f32::NAN], &[0.0]);
        assert_eq!(bad.max_ulp, u32::MAX);
        assert!(bad.max_abs.is_infinite());
    }
}
