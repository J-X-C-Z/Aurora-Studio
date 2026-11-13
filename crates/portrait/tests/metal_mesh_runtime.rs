//! macOS-only real-device checks for `MetalContext::mesh_raster`.
//!
//! Default `cargo test` only compiles these; every test is `#[ignore]`.
//! Run on macOS with:
//!   cargo test -p aurora-pixcake --test metal_mesh_runtime -- --ignored
//!
//! On non-macOS an explicit `--ignored` run panics instead of passing.
//!
//! A12 perspective gate markers:
//! `perspective_mesh_bytes_match_raster`,
//! `perspective_constant_w_matches_affine`,
//! `perspective_zero_and_negative_reciprocal_are_refused`,
//! `perspective_vendor_contract_is_typed_refused`.

use aurora_pixcake::metal::{MetalContext, MetalRefusal};
use aurora_pixcake::metal_mesh::{
    perspective_uv, MeshContract, MeshContractRefusal, MeshRasterInput, PerspectiveMeshInput,
    VendorMeshEffect, MESH_RASTER_ABI_VERSION,
};
use aurora_pixcake::raster::{Cull, Edge, Overlap, Rules, Sampler, Winding, Wrap};
use aurora_pixcake::Image;

fn require_macos() {
    if !cfg!(target_os = "macos") {
        panic!("requires macOS and a Metal device; non-macOS is not a pass");
    }
}

const READBACK_COLOR_FORMAT: &str = "RGBA8Unorm";
const READBACK_PIXEL_TYPE: &str = "Uint8";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReadbackContract {
    width: u32,
    height: u32,
    row_bytes: usize,
    stride_bytes: usize,
    color_format: &'static str,
    pixel_type: &'static str,
    origin_top_left: bool,
    completion_fence: bool,
}

impl ReadbackContract {
    fn tight(width: u32, height: u32) -> Self {
        let row_bytes = width as usize * 4;
        Self {
            width,
            height,
            row_bytes,
            stride_bytes: row_bytes,
            color_format: READBACK_COLOR_FORMAT,
            pixel_type: READBACK_PIXEL_TYPE,
            origin_top_left: true,
            completion_fence: true,
        }
    }

    fn expected_bytes(self) -> Result<usize, MetalRefusal> {
        self.stride_bytes
            .checked_mul(self.height as usize)
            .ok_or(MetalRefusal::ShapeMismatch("readback_bytes"))
    }

    fn validate(self) -> Result<(), MetalRefusal> {
        if self.row_bytes != self.width as usize * 4 || self.stride_bytes != self.row_bytes {
            return Err(MetalRefusal::ValueOutOfRange("readback_stride"));
        }
        if self.color_format != READBACK_COLOR_FORMAT {
            return Err(MetalRefusal::ValueOutOfRange("readback_format"));
        }
        if self.pixel_type != READBACK_PIXEL_TYPE {
            return Err(MetalRefusal::ValueOutOfRange("readback_type"));
        }
        if !self.origin_top_left {
            return Err(MetalRefusal::ValueOutOfRange("readback_origin"));
        }
        if !self.completion_fence {
            return Err(MetalRefusal::ValueOutOfRange("readback_fence"));
        }
        Ok(())
    }

    fn validate_bytes(self, bytes: &[u8]) -> Result<(), MetalRefusal> {
        self.validate()?;
        if bytes.len() != self.expected_bytes()? {
            return Err(MetalRefusal::ShapeMismatch("readback_bytes"));
        }
        Ok(())
    }
}

fn serial_readback_contract(_width: u32, _height: u32) -> Result<ReadbackContract, MetalRefusal> {
    Err(MetalRefusal::ValueOutOfRange("serial_force"))
}

fn source_image() -> Image {
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

fn rules(edge: Edge, overlap: Overlap, cull: Cull) -> Rules {
    Rules {
        winding: Winding::PositiveAreaIsFront,
        cull,
        edge,
        overlap,
    }
}

fn run(
    image: &Image,
    dest: &[[f32; 2]],
    uv: &[[f32; 2]],
    ebo: &[u16],
    rules: Rules,
) -> (Vec<u8>, Vec<u8>) {
    let input = MeshRasterInput {
        dest,
        source_uv: uv,
        ebo,
        source: image,
        rules,
        sampler: Sampler {
            wrap: Wrap::ClampToEdge,
        },
        contract: MeshContract::AffineBufferCandidate,
    };
    let cpu = input.cpu_golden().expect("CPU golden");
    let gpu = MetalContext::new()
        .expect("MetalContext")
        .mesh_raster(&input)
        .expect("mesh dispatch");
    (gpu.pixels, cpu.pixels)
}

fn perspective_source() -> Image {
    let mut pixels = Vec::new();
    for y in 0..8u32 {
        for x in 0..8u32 {
            pixels.extend_from_slice(&[
                (x * 29 + 3) as u8,
                (y * 31 + 5) as u8,
                (x * 17 + y * 13 + 11) as u8,
                (40 + x + y) as u8,
            ]);
        }
    }
    Image::new(8, 8, pixels).unwrap()
}

fn perspective_axis(pos: f32, size: u32) -> (u32, u32, f32) {
    let limit = (size - 1) as f32;
    let p = pos.clamp(0.0, limit);
    let i0 = p.floor();
    let i0 = i0 as u32;
    (i0, (i0 + 1).min(size - 1), p - i0 as f32)
}

fn perspective_texel(image: &Image, x: u32, y: u32, channel: usize) -> f32 {
    image.pixels[((y * image.width + x) * 4) as usize + channel] as f32 / 255.0
}

fn perspective_sample(image: &Image, u: f32, v: f32) -> [u8; 4] {
    let (x0, x1, fx) = perspective_axis(u * image.width as f32 - 0.5, image.width);
    let (y0, y1, fy) = perspective_axis(v * image.height as f32 - 0.5, image.height);
    let mut out = [0u8; 4];
    for channel in 0..4 {
        let top = perspective_texel(image, x0, y0, channel)
            + (perspective_texel(image, x1, y0, channel)
                - perspective_texel(image, x0, y0, channel))
                * fx;
        let bottom = perspective_texel(image, x0, y1, channel)
            + (perspective_texel(image, x1, y1, channel)
                - perspective_texel(image, x0, y1, channel))
                * fx;
        let value = top + (bottom - top) * fy;
        out[channel] = (value * 255.0).round_ties_even().clamp(0.0, 255.0) as u8;
    }
    out
}

fn perspective_expected(
    image: &Image,
    dest: [[f32; 2]; 3],
    uv: [[f32; 2]; 3],
    vertex_perspective: [[f32; 2]; 3],
) -> Vec<u8> {
    let (p0, p1, p2) = (dest[0], dest[1], dest[2]);
    let area2 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
    let mut pixels = image.pixels.clone();
    for y in 0..image.height {
        for x in 0..image.width {
            let sx = x as f32 + 0.5;
            let sy = y as f32 + 0.5;
            let w0 = ((p1[0] - sx) * (p2[1] - sy) - (p1[1] - sy) * (p2[0] - sx)) / area2;
            let w1 = ((p2[0] - sx) * (p0[1] - sy) - (p2[1] - sy) * (p0[0] - sx)) / area2;
            let w2 = ((p0[0] - sx) * (p1[1] - sy) - (p0[1] - sy) * (p1[0] - sx)) / area2;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }
            let Some(sample) = perspective_uv(uv, vertex_perspective, [w0, w1, w2]) else {
                continue;
            };
            let color = perspective_sample(image, sample.u, sample.v);
            let offset = ((y * image.width + x) * 4) as usize;
            pixels[offset..offset + 4].copy_from_slice(&color);
        }
    }
    pixels
}

#[test]
#[ignore = "requires macOS Metal device"]
fn mesh_bytes_match_raster_and_uncovered_alpha_is_preserved() {
    require_macos();
    assert_eq!(MESH_RASTER_ABI_VERSION, 1);

    let image = source_image();
    let dest = [[0.0f32, 0.0], [4.0, 0.0], [0.0, 4.0]];
    let uv = [[0.25f32, 0.25], [0.75, 0.25], [0.25, 0.75]];
    let ebo = [0u16, 1, 2];
    let (gpu, cpu) = run(
        &image,
        &dest,
        &uv,
        &ebo,
        rules(Edge::Inclusive, Overlap::First, Cull::None),
    );
    assert_eq!(gpu, cpu);

    // Bottom-right pixel is outside the triangle and must keep all source bytes.
    let offset = (3 * 4 + 3) * 4;
    assert_eq!(
        &gpu[offset..offset + 4],
        &image.pixels[offset..offset + 4],
        "uncovered alpha/RGB changed"
    );
}

#[test]
#[ignore = "requires macOS Metal device"]
fn overlap_first_last_reject_match_raster() {
    require_macos();
    let image = source_image();
    let dest = [
        [0.0f32, 0.0],
        [8.0, 0.0],
        [0.0, 8.0],
        [0.5, 0.5],
        [4.5, 0.5],
        [0.5, 4.5],
    ];
    let uv = [
        [0.0f32, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.2, 0.2],
        [0.8, 0.2],
        [0.2, 0.8],
    ];
    let ebo = [0u16, 1, 2, 3, 4, 5];
    for overlap in [Overlap::First, Overlap::Last, Overlap::Reject] {
        let (gpu, cpu) = run(
            &image,
            &dest,
            &uv,
            &ebo,
            rules(Edge::Inclusive, overlap, Cull::None),
        );
        assert_eq!(gpu, cpu, "overlap {overlap:?} mismatch");
    }
}

#[test]
#[ignore = "requires macOS Metal device"]
fn cull_front_and_back_match_raster() {
    require_macos();
    let image = source_image();
    let dest = [[0.0f32, 0.0], [8.0, 0.0], [0.0, 8.0]];
    let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let ebo = [0u16, 1, 2];

    let (front, front_cpu) = run(
        &image,
        &dest,
        &uv,
        &ebo,
        rules(Edge::Inclusive, Overlap::First, Cull::Front),
    );
    assert_eq!(front, front_cpu);
    assert_eq!(front, image.pixels, "front-cull must keep source bytes");

    let (back, back_cpu) = run(
        &image,
        &dest,
        &uv,
        &ebo,
        rules(Edge::Inclusive, Overlap::First, Cull::Back),
    );
    assert_eq!(back, back_cpu);
    assert_ne!(back, image.pixels, "back-cull must draw the front triangle");
}

#[test]
#[ignore = "requires macOS Metal device"]
fn perspective_mesh_bytes_match_raster() {
    require_macos();
    let image = perspective_source();
    let dest = [[0.0f32, 0.0], [8.0, 0.0], [0.0, 8.0]];
    let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let vertex_perspective = [[0.25f32, 1.0], [0.5, 2.0], [0.75, 1.5]];
    let ebo = [0u16, 1, 2];
    let base = MeshRasterInput {
        dest: &dest,
        source_uv: &uv,
        ebo: &ebo,
        source: &image,
        rules: rules(Edge::Inclusive, Overlap::First, Cull::None),
        sampler: Sampler {
            wrap: Wrap::ClampToEdge,
        },
        contract: MeshContract::AffineBufferCandidate,
    };
    let affine = base.cpu_golden().expect("affine CPU golden");
    let expected = perspective_expected(&image, dest, uv, vertex_perspective);
    assert_ne!(
        expected, affine.pixels,
        "fixture must produce a perspective/affine difference"
    );

    let input = PerspectiveMeshInput {
        base,
        vertex_perspective: &vertex_perspective,
    };
    let gpu = MetalContext::new()
        .expect("MetalContext")
        .mesh_raster_perspective(&input)
        .expect("perspective mesh dispatch");
    assert_eq!(gpu.pixels, expected);
}

#[test]
#[ignore = "requires macOS Metal device"]
fn perspective_constant_w_matches_affine() {
    require_macos();
    let image = perspective_source();
    let dest = [[0.0f32, 0.0], [8.0, 0.0], [0.0, 8.0]];
    let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let ebo = [0u16, 1, 2];
    let vertex_perspective = [[0.1f32, 1.0], [0.2, 1.0], [0.3, 1.0]];
    let base = MeshRasterInput {
        dest: &dest,
        source_uv: &uv,
        ebo: &ebo,
        source: &image,
        rules: rules(Edge::Inclusive, Overlap::First, Cull::None),
        sampler: Sampler {
            wrap: Wrap::ClampToEdge,
        },
        contract: MeshContract::AffineBufferCandidate,
    };
    let input = PerspectiveMeshInput {
        base,
        vertex_perspective: &vertex_perspective,
    };
    let context = MetalContext::new().expect("MetalContext");
    let affine = context.mesh_raster(&base).expect("affine dispatch");
    let perspective = context
        .mesh_raster_perspective(&input)
        .expect("perspective dispatch");
    assert_eq!(perspective.pixels, affine.pixels);
    assert_eq!(
        perspective.pixels,
        base.cpu_golden().expect("CPU golden").pixels
    );
}

#[test]
#[ignore = "requires macOS Metal device"]
fn perspective_zero_and_negative_reciprocal_are_refused() {
    require_macos();
    let image = perspective_source();
    let dest = [[0.0f32, 0.0], [8.0, 0.0], [0.0, 8.0]];
    let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let ebo = [0u16, 1, 2];
    let context = MetalContext::new().expect("MetalContext");
    for invalid_inv_w in [0.0f32, -1.0] {
        let vertex_perspective = [[0.0f32, invalid_inv_w], [0.0, 1.0], [0.0, 1.0]];
        let input = PerspectiveMeshInput {
            base: MeshRasterInput {
                dest: &dest,
                source_uv: &uv,
                ebo: &ebo,
                source: &image,
                rules: rules(Edge::Inclusive, Overlap::First, Cull::None),
                sampler: Sampler {
                    wrap: Wrap::ClampToEdge,
                },
                contract: MeshContract::AffineBufferCandidate,
            },
            vertex_perspective: &vertex_perspective,
        };
        assert_eq!(
            context.mesh_raster_perspective(&input),
            Err(MetalRefusal::ValueOutOfRange("vertex_inv_w"))
        );
    }
}

#[test]
#[ignore = "requires macOS Metal device"]
fn perspective_vendor_contract_is_typed_refused() {
    require_macos();
    let image = perspective_source();
    let dest = [[0.0f32, 0.0], [8.0, 0.0], [0.0, 8.0]];
    let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let ebo = [0u16, 1, 2];
    let vertex_perspective = [[0.0f32, 1.0]; 3];
    let input = PerspectiveMeshInput {
        base: MeshRasterInput {
            dest: &dest,
            source_uv: &uv,
            ebo: &ebo,
            source: &image,
            rules: rules(Edge::Inclusive, Overlap::First, Cull::None),
            sampler: Sampler {
                wrap: Wrap::ClampToEdge,
            },
            contract: MeshContract::VendorForwardMesh(VendorMeshEffect::Pf8033),
        },
        vertex_perspective: &vertex_perspective,
    };
    let context = MetalContext::new().expect("MetalContext");
    assert_eq!(
        context.mesh_raster_perspective(&input),
        Err(MetalRefusal::UnsupportedMeshContract(
            MeshContractRefusal::VendorForwardMeshUnsupported {
                effect: VendorMeshEffect::Pf8033,
                missing_perspective_uv: true,
                missing_texture_sampler: true,
                missing_fbo: true,
            }
        ))
    );
}

#[test]
fn readback_contract_rejects_bad_format_stride_length_and_fence() {
    let valid = ReadbackContract::tight(2, 2);
    assert_eq!(valid.validate(), Ok(()));
    assert_eq!(valid.expected_bytes(), Ok(16));
    assert_eq!(valid.validate_bytes(&[0u8; 16]), Ok(()));

    let bad_stride = ReadbackContract {
        stride_bytes: 9,
        ..valid
    };
    assert_eq!(
        bad_stride.validate(),
        Err(MetalRefusal::ValueOutOfRange("readback_stride"))
    );
    let bad_format = ReadbackContract {
        color_format: "BGRA8Unorm",
        ..valid
    };
    assert_eq!(
        bad_format.validate(),
        Err(MetalRefusal::ValueOutOfRange("readback_format"))
    );
    let bad_type = ReadbackContract {
        pixel_type: "Float32",
        ..valid
    };
    assert_eq!(
        bad_type.validate(),
        Err(MetalRefusal::ValueOutOfRange("readback_type"))
    );
    let bad_origin = ReadbackContract {
        origin_top_left: false,
        ..valid
    };
    assert_eq!(
        bad_origin.validate(),
        Err(MetalRefusal::ValueOutOfRange("readback_origin"))
    );
    let bad_fence = ReadbackContract {
        completion_fence: false,
        ..valid
    };
    assert_eq!(
        bad_fence.validate(),
        Err(MetalRefusal::ValueOutOfRange("readback_fence"))
    );
    assert_eq!(
        valid.validate_bytes(&[0u8; 15]),
        Err(MetalRefusal::ShapeMismatch("readback_bytes"))
    );
}

#[test]
fn serial_readback_contract_is_typed_blocked() {
    assert_eq!(
        serial_readback_contract(4, 4),
        Err(MetalRefusal::ValueOutOfRange("serial_force"))
    );
}

#[test]
#[ignore = "requires macOS Metal device"]
fn affine_readback_contract_matches_cpu_golden() {
    require_macos();
    let image = source_image();
    let dest = [[0.0f32, 0.0], [8.0, 0.0], [0.0, 8.0]];
    let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let ebo = [0u16, 1, 2];
    let (gpu, cpu) = run(
        &image,
        &dest,
        &uv,
        &ebo,
        rules(Edge::Inclusive, Overlap::First, Cull::None),
    );
    let contract = ReadbackContract::tight(image.width, image.height);
    contract
        .validate_bytes(&gpu)
        .expect("affine readback contract");
    assert_eq!(gpu, cpu);
}

#[test]
#[ignore = "requires macOS Metal device"]
fn perspective_readback_contract_matches_expected() {
    require_macos();
    let image = perspective_source();
    let dest = [[0.0f32, 0.0], [8.0, 0.0], [0.0, 8.0]];
    let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let vertex_perspective = [[0.25f32, 1.0], [0.5, 2.0], [0.75, 1.5]];
    let ebo = [0u16, 1, 2];
    let base = MeshRasterInput {
        dest: &dest,
        source_uv: &uv,
        ebo: &ebo,
        source: &image,
        rules: rules(Edge::Inclusive, Overlap::First, Cull::None),
        sampler: Sampler {
            wrap: Wrap::ClampToEdge,
        },
        contract: MeshContract::AffineBufferCandidate,
    };
    let expected = perspective_expected(&image, dest, uv, vertex_perspective);
    let input = PerspectiveMeshInput {
        base,
        vertex_perspective: &vertex_perspective,
    };
    let gpu = MetalContext::new()
        .expect("MetalContext")
        .mesh_raster_perspective(&input)
        .expect("perspective mesh dispatch");
    let contract = ReadbackContract::tight(image.width, image.height);
    contract
        .validate_bytes(&gpu.pixels)
        .expect("perspective readback contract");
    assert_eq!(gpu.pixels, expected);
}

#[test]
#[ignore = "BLOCKED: no force-serial host selector"]
fn serial_force_mesh_bytes_match_raster() {
    require_macos();
    let image = source_image();
    let dest = [[0.0f32, 0.0], [8.0, 0.0], [0.0, 8.0]];
    let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let ebo = [0u16, 1, 2];
    let input = MeshRasterInput {
        dest: &dest,
        source_uv: &uv,
        ebo: &ebo,
        source: &image,
        rules: rules(Edge::Inclusive, Overlap::First, Cull::None),
        sampler: Sampler {
            wrap: Wrap::ClampToEdge,
        },
        contract: MeshContract::AffineBufferCandidate,
    };
    let expected = input.cpu_golden().expect("serial CPU golden");
    // A13_SERIAL_FORCE_BLOCKED: no public force-serial host selector exists.
    panic!(
        "BLOCKED: serial_force_mesh_bytes_match_raster has fixture input/expected ({} bytes) \
         but no force-serial host selector; do not report PASS",
        expected.pixels.len()
    );
}

#[test]
#[ignore = "requires macOS Metal device"]
fn invalid_input_is_refused_before_any_dispatch() {
    require_macos();
    let image = source_image();
    let dest = [[0.0f32, 0.0], [8.0, 0.0], [0.0, 8.0]];
    let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let input = MeshRasterInput {
        dest: &dest,
        source_uv: &uv,
        ebo: &[0u16, 1],
        source: &image,
        rules: rules(Edge::Inclusive, Overlap::First, Cull::None),
        sampler: Sampler {
            wrap: Wrap::ClampToEdge,
        },
        contract: MeshContract::AffineBufferCandidate,
    };
    let context = MetalContext::new().expect("MetalContext");
    assert_eq!(
        context.mesh_raster(&input),
        Err(aurora_pixcake::metal::MetalRefusal::ShapeMismatch("ebo"))
    );
}

#[test]
#[ignore = "requires macOS Metal device"]
fn vendor_forward_mesh_refusal_preserves_capabilities() {
    require_macos();
    let image = source_image();
    let dest = [[0.0f32, 0.0], [8.0, 0.0], [0.0, 8.0]];
    let uv = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let input = MeshRasterInput {
        dest: &dest,
        source_uv: &uv,
        ebo: &[0u16, 1, 2],
        source: &image,
        rules: rules(Edge::Inclusive, Overlap::First, Cull::None),
        sampler: Sampler {
            wrap: Wrap::ClampToEdge,
        },
        contract: MeshContract::VendorForwardMesh(VendorMeshEffect::Pf8033),
    };
    let context = MetalContext::new().expect("MetalContext");
    assert_eq!(
        context.mesh_raster(&input),
        Err(MetalRefusal::UnsupportedMeshContract(
            MeshContractRefusal::VendorForwardMeshUnsupported {
                effect: VendorMeshEffect::Pf8033,
                missing_perspective_uv: true,
                missing_texture_sampler: true,
                missing_fbo: true,
            }
        ))
    );
}
