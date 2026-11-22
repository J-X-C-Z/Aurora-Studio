//! PF247/PF248 V-face adapter for the forward V2Warp candidate.
//!
//! The four rows are distinct atlas bindings, not a single left/right atlas.
//! `V2FuseOffset` remains the producer of the 280x280 encoded offset field;
//! this adapter validates that handoff and consumes it through the forward
//! 526/2982 V2Warp path. It never substitutes an inverse dense field.

use crate::cuda_v2warp::{Mesh, OffsetTexture};
use crate::kernels::face_geometry::{self, Texel};
use crate::raster::Rules;
use crate::{Image, Refusal};

pub const OFFSET_EDGE: u32 = 280;
pub const VERTEX_COUNT: usize = 526;
pub const INDEX_COUNT: usize = 2982;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Row {
    LeftPositive,
    RightPositive,
    LeftNegative,
    RightNegative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowSpec {
    pub index: u8,
    pub pf: u32,
    pub atlas: &'static str,
    /// Placement in the 280x280 Fuse target: x, y, width, height.
    pub rectangle: [u32; 4],
}

pub const ROWS: [Row; 4] = [
    Row::LeftPositive,
    Row::RightPositive,
    Row::LeftNegative,
    Row::RightNegative,
];

impl Row {
    pub const fn spec(self) -> RowSpec {
        match self {
            Row::LeftPositive => RowSpec {
                index: 0,
                pf: 247,
                atlas: "Grid_2010_Positive.png",
                rectangle: [83, 165, 53, 53],
            },
            Row::RightPositive => RowSpec {
                index: 1,
                pf: 248,
                atlas: "Grid_2011_Positive.png",
                rectangle: [147, 164, 51, 55],
            },
            Row::LeftNegative => RowSpec {
                index: 2,
                pf: 247,
                atlas: "Grid_2010_Negative.png",
                rectangle: [87, 171, 47, 46],
            },
            Row::RightNegative => RowSpec {
                index: 3,
                pf: 248,
                atlas: "Grid_2011_Negative.png",
                rectangle: [150, 169, 44, 48],
            },
        }
    }
}

/// Vendor PNG decoded to top-down RGBA8 bytes. Row origin remains live-only.
#[derive(Clone, Copy, Debug)]
pub struct Atlas<'a> {
    pub row: Row,
    pub width: u32,
    pub height: u32,
    pub pixels: &'a [u8],
}

impl Atlas<'_> {
    pub fn validate(self) -> Result<(), Refusal> {
        let spec = self.row.spec();
        let [x, y, width, height] = spec.rectangle;
        let ends_inside = x.checked_add(width).is_some_and(|right| right <= OFFSET_EDGE)
            && y.checked_add(height).is_some_and(|bottom| bottom <= OFFSET_EDGE);
        if !ends_inside {
            return Err(Refusal::ValueOutOfRange("PF247/248 atlas rectangle"));
        }
        if self.width != width || self.height != height {
            return Err(Refusal::ShapeMismatch("PF247/248 atlas dimensions"));
        }
        let bytes = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|value| value.checked_mul(4))
            .ok_or(Refusal::ShapeMismatch("PF247/248 atlas RGBA8"))?;
        if self.pixels.len() != bytes {
            return Err(Refusal::ShapeMismatch("PF247/248 atlas RGBA8"));
        }
        Ok(())
    }
}

/// V2FuseOffset input bundle. The offset texture must be the post-`0x48`
/// encoded field, not a field reconstructed by an inverse UV-map consumer.
#[derive(Clone, Copy, Debug)]
pub struct Inputs<'a> {
    pub row: Row,
    pub atlas: Atlas<'a>,
    pub offset: OffsetTexture<'a>,
    pub mesh: Mesh<'a>,
}

/// Reuses the existing signed-UI mapping so the adapter cannot drift from the
/// PF247/PF248 `[degree, degree]` contract.
pub fn degree_for(row: Row, ui_left: f32, ui_right: f32) -> Result<f32, Refusal> {
    let geometry = face_geometry::v_face(
        ui_left,
        ui_right,
        Texel {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        },
    )?;
    let part = geometry
        .parts
        .get(row.spec().index as usize)
        .ok_or(Refusal::ShapeMismatch("PF247/248 degree rows"))?;
    if part.pf != row.spec().pf || part.degree != part.second_degree {
        return Err(Refusal::ValueOutOfRange("PF247/248 degree row"));
    }
    if !part.degree.is_finite() || !(0.0..=1.0).contains(&part.degree) {
        return Err(Refusal::ValueOutOfRange("PF247/248 degree"));
    }
    Ok(part.degree)
}

/// CPU fixture/reference for `0x48 V2FuseOffset` at the pinned `F/O=280`
/// boundary. Production still needs the GPU producer and live precision state.
pub fn fuse_offset_cpu(
    base: &Image,
    atlas: Atlas<'_>,
    degree: f32,
) -> Result<Vec<[f32; 4]>, Refusal> {
    atlas.validate()?;
    base.validate()?;
    if (base.width, base.height) != (OFFSET_EDGE, OFFSET_EDGE) {
        return Err(Refusal::ShapeMismatch("V2FuseOffset base 280x280"));
    }
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("PF247/248 degree"));
    }
    let [x0, y0, width, height] = atlas.row.spec().rectangle;
    let mut out = vec![[0.0; 4]; (OFFSET_EDGE * OFFSET_EDGE) as usize];
    for y in y0..y0 + height {
        for x in x0..x0 + width {
            let source_uv = [
                (x as f32 + 0.5) / OFFSET_EDGE as f32,
                (y as f32 + 0.5) / OFFSET_EDGE as f32,
            ];
            let atlas_uv = [
                (x - x0) as f32 / width as f32 + 0.5 / width as f32,
                (y - y0) as f32 / height as f32 + 0.5 / height as f32,
            ];
            let source = sample_rgba8(base.width, base.height, &base.pixels, source_uv);
            let roi = sample_rgba8(atlas.width, atlas.height, atlas.pixels, atlas_uv);
            let d = [
                (source[0] - source[1]) * 0.25 + (roi[0] - roi[1]) * 0.25 * degree,
                (source[2] - source[3]) * 0.25 + (roi[2] - roi[3]) * 0.25 * degree,
            ];
            out[(y * OFFSET_EDGE + x) as usize] = [
                4.0 * d[0].max(0.0),
                4.0 * (-d[0]).max(0.0),
                4.0 * d[1].max(0.0),
                4.0 * (-d[1]).max(0.0),
            ];
        }
    }
    Ok(out)
}

pub struct Output {
    pub row: Row,
    pub degree: f32,
    pub warp: crate::cuda_v2warp::Output,
}

fn validate_inputs(inputs: Inputs<'_>) -> Result<(), Refusal> {
    if inputs.atlas.row != inputs.row {
        return Err(Refusal::ValueOutOfRange("PF247/248 atlas row"));
    }
    inputs.atlas.validate()?;
    if (inputs.offset.width, inputs.offset.height) != (OFFSET_EDGE, OFFSET_EDGE)
        || inputs.offset.pixels.len() != (OFFSET_EDGE * OFFSET_EDGE) as usize
    {
        return Err(Refusal::ShapeMismatch("PF247/248 encoded offset 280x280"));
    }
    crate::cuda_v2warp::validate_mesh(inputs.mesh)
}

#[cfg(windows)]
pub fn render(
    gpu: &crate::cuda_v2warp::V2Warp,
    source: &Image,
    inputs: Inputs<'_>,
    ui_left: f32,
    ui_right: f32,
    rules: Rules,
    mat2: [f32; 4],
    filter_size: [f32; 2],
) -> Result<Option<Output>, crate::cuda::CudaError> {
    validate_inputs(inputs).map_err(crate::cuda::CudaError::Refused)?;
    let degree = degree_for(inputs.row, ui_left, ui_right)
        .map_err(crate::cuda::CudaError::Refused)?;
    if degree == 0.0 {
        return Ok(None);
    }
    let warp = gpu.render(
        source,
        inputs.offset,
        inputs.mesh,
        rules,
        mat2,
        filter_size,
    )?;
    Ok(Some(Output { row: inputs.row, degree, warp }))
}

fn sample_rgba8(width: u32, height: u32, pixels: &[u8], uv: [f32; 2]) -> [f32; 4] {
    let x = (uv[0] * width as f32 - 0.5).clamp(0.0, width as f32 - 1.0);
    let y = (uv[1] * height as f32 - 0.5).clamp(0.0, height as f32 - 1.0);
    let x0 = x.floor();
    let y0 = y.floor();
    let x1 = (x0 + 1.0).min(width as f32 - 1.0);
    let y1 = (y0 + 1.0).min(height as f32 - 1.0);
    let fx = x - x0;
    let fy = y - y0;
    let at = |x: f32, y: f32, channel: usize| {
        let offset = ((y as usize * width as usize + x as usize) * 4) + channel;
        pixels[offset] as f32 / 255.0
    };
    let mut out = [0.0; 4];
    for channel in 0..4 {
        let top = at(x0, y0, channel)
            + (at(x1, y0, channel) - at(x0, y0, channel)) * fx;
        let bottom = at(x0, y1, channel)
            + (at(x1, y1, channel) - at(x0, y1, channel)) * fx;
        out[channel] = top + (bottom - top) * fy;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::{Cull, Edge, Overlap, Winding};

    const RULES: Rules = Rules {
        winding: Winding::NegativeAreaIsFront,
        cull: Cull::None,
        edge: Edge::Inclusive,
        overlap: Overlap::Last,
    };

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    fn source(width: u32, height: u32) -> Image {
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.extend_from_slice(&[
                    (x * 11 + y * 3) as u8,
                    (y * 13 + x * 2) as u8,
                    (x * 5 + y * 7) as u8,
                    255,
                ]);
            }
        }
        Image::new(width, height, pixels).unwrap()
    }

    fn mesh_17x13() -> (
        Vec<[f32; 2]>,
        Vec<[f32; 2]>,
        Vec<[f32; 2]>,
        Vec<[f32; 2]>,
        Vec<u16>,
    ) {
        let mut positions = vec![[0.0; 2]; VERTEX_COUNT];
        let mut texcoord = vec![[0.0; 2]; VERTEX_COUNT];
        let offset_texcoord = vec![[0.5; 2]; VERTEX_COUNT];
        let face_scales = vec![[1.0; 2]; VERTEX_COUNT];
        positions[0] = [0.0, 0.0];
        positions[1] = [17.0, 0.0];
        positions[2] = [0.0, 13.0];
        texcoord[0] = [0.0, 0.0];
        texcoord[1] = [1.0, 0.0];
        texcoord[2] = [0.0, 1.0];
        let mut ebo = vec![0u16; INDEX_COUNT];
        ebo[..3].copy_from_slice(&[0, 1, 2]);
        (positions, texcoord, offset_texcoord, face_scales, ebo)
    }

    fn atlas_zero(row: Row) -> (Vec<u8>, [u32; 2]) {
        let [_, _, width, height] = row.spec().rectangle;
        (vec![0; (width * height * 4) as usize], [width, height])
    }

    #[test]
    fn four_rows_pin_pf_atlas_and_rectangle() {
        let expected = [
            (247, "Grid_2010_Positive.png", [83, 165, 53, 53]),
            (248, "Grid_2011_Positive.png", [147, 164, 51, 55]),
            (247, "Grid_2010_Negative.png", [87, 171, 47, 46]),
            (248, "Grid_2011_Negative.png", [150, 169, 44, 48]),
        ];
        for (row, (pf, atlas, rectangle)) in ROWS.into_iter().zip(expected) {
            let spec = row.spec();
            assert_eq!(spec.pf, pf);
            assert_eq!(spec.atlas, atlas);
            assert_eq!(spec.rectangle, rectangle);
        }
    }

    #[test]
    fn degree_uses_vendor_curve_and_degree2_equality() {
        assert!(close(degree_for(Row::LeftPositive, 100.0, 0.0).unwrap(), 1.0));
        assert!(close(degree_for(Row::LeftNegative, -100.0, 0.0).unwrap(), 1.0));
        assert!(close(degree_for(Row::RightPositive, 0.0, 100.0).unwrap(), 1.0));
        assert!(close(degree_for(Row::RightNegative, 0.0, -100.0).unwrap(), 1.0));
        for row in ROWS {
            assert_eq!(degree_for(row, 0.0, 0.0).unwrap(), 0.0);
        }
    }

    #[test]
    fn fuse_offset_is_normalized_rgba8_and_encoded() {
        let row = Row::LeftPositive;
        let [_, _, width, height] = row.spec().rectangle;
        let mut pixels = vec![0; (width * height * 4) as usize];
        pixels[..4].copy_from_slice(&[0, 4, 8, 0]);
        let atlas = Atlas {
            row,
            width,
            height,
            pixels: &pixels,
        };
        let offset = fuse_offset_cpu(
            &Image::solid(OFFSET_EDGE, OFFSET_EDGE, [0, 0, 0, 0]),
            atlas,
            1.0,
        )
        .unwrap();
        let texel = offset[(165 * OFFSET_EDGE + 83) as usize];
        assert!(close(texel[0], 0.0));
        assert!(close(texel[1], 16.0 / 1020.0));
        assert!(close(texel[2], 32.0 / 1020.0));
        assert!(close(texel[3], 0.0));
    }

    #[test]
    fn cpu_route_is_inactive_at_ui_zero_and_forward_when_active() {
        let (positions, texcoord, offset_texcoord, face_scales, ebo) = mesh_17x13();
        let mesh = Mesh {
            framebuffer_positions_w1: &positions,
            texcoord: &texcoord,
            offset_texcoord: &offset_texcoord,
            face_scales: &face_scales,
            ebo: &ebo,
        };
        let (atlas_pixels, [width, height]) = atlas_zero(Row::LeftPositive);
        let atlas = Atlas {
            row: Row::LeftPositive,
            width,
            height,
            pixels: &atlas_pixels,
        };
        let offset_pixels = vec![[4.0, 0.0, 0.0, 0.0]; (OFFSET_EDGE * OFFSET_EDGE) as usize];
        let inputs = Inputs {
            row: Row::LeftPositive,
            atlas,
            offset: OffsetTexture {
                width: OFFSET_EDGE,
                height: OFFSET_EDGE,
                pixels: &offset_pixels,
            },
            mesh,
        };
        let source = source(17, 13);
        let cpu = crate::cuda_v2warp::apply_cpu(
            &source,
            inputs.offset,
            inputs.mesh,
            RULES,
            [1.0, 0.0, 0.0, 1.0],
            [17.0, 13.0],
        )
        .unwrap();
        assert_ne!(cpu.pixels, source.pixels);
        assert_eq!(
            degree_for(Row::LeftPositive, 0.0, 0.0).unwrap(),
            0.0
        );
    }

    #[cfg(windows)]
    #[test]
    fn gpu_route_matches_cpu_and_reports_forward_path() {
        let gpu = match crate::cuda_v2warp::V2Warp::open() {
            Ok(gpu) => gpu,
            Err(crate::cuda::CudaError::Unavailable(reason)) => {
                eprintln!("SKIP_NO_NVIDIA_DRIVER: {reason}");
                assert!(std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none());
                return;
            }
            Err(err) => panic!("NVIDIA path present but broken: {err:?}"),
        };
        let (positions, texcoord, offset_texcoord, face_scales, ebo) = mesh_17x13();
        let mesh = Mesh {
            framebuffer_positions_w1: &positions,
            texcoord: &texcoord,
            offset_texcoord: &offset_texcoord,
            face_scales: &face_scales,
            ebo: &ebo,
        };
        let (atlas_pixels, [width, height]) = atlas_zero(Row::LeftPositive);
        let atlas = Atlas {
            row: Row::LeftPositive,
            width,
            height,
            pixels: &atlas_pixels,
        };
        let offset_pixels = vec![[4.0, 0.0, 0.0, 0.0]; (OFFSET_EDGE * OFFSET_EDGE) as usize];
        let inputs = Inputs {
            row: Row::LeftPositive,
            atlas,
            offset: OffsetTexture {
                width: OFFSET_EDGE,
                height: OFFSET_EDGE,
                pixels: &offset_pixels,
            },
            mesh,
        };
        let source = source(17, 13);
        let expected = crate::cuda_v2warp::apply_cpu(
            &source,
            inputs.offset,
            inputs.mesh,
            RULES,
            [1.0, 0.0, 0.0, 1.0],
            [17.0, 13.0],
        )
        .unwrap();
        let actual = render(
            &gpu,
            &source,
            inputs,
            100.0,
            0.0,
            RULES,
            [1.0, 0.0, 0.0, 1.0],
            [17.0, 13.0],
        )
        .unwrap()
        .expect("active row");
        let max_diff = actual
            .warp
            .image
            .pixels
            .iter()
            .zip(&expected.pixels)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        eprintln!(
            "VFACE_GPU path={} row={:?} pf={} degree={} device={} covered={} max_diff={}",
            actual.warp.path,
            actual.row,
            actual.row.spec().pf,
            actual.degree,
            actual.warp.select_dispatch.device,
            actual.warp.coverage.iter().filter(|value| **value > 0).count(),
            max_diff
        );
        assert_eq!(actual.warp.path, crate::cuda_v2warp::PATH);
        assert_eq!(actual.warp.select_dispatch.kernel, "v2warp_select_kernel");
        assert_eq!(actual.warp.eval_dispatch.kernel, "v2warp_eval_kernel");
        assert!(max_diff <= 1);
    }

    #[test]
    fn archived_pf247_fixture_runs_cpu_and_real_gpu() {
        let Some(root) = std::env::var_os("S521_VFACE_FIXTURE") else {
            eprintln!("SKIP_NO_S521_VFACE_FIXTURE");
            return;
        };
        let root = std::path::PathBuf::from(root);
        let positions = read_pairs(&root.join("input/vface.positions526.f32le"));
        let texcoord = read_pairs(&root.join("input/vface.texcoord526.f32le"));
        let offset_texcoord = read_pairs(&root.join("input/vface.offset_texcoord526.f32le"));
        let face_scales = read_pairs(&root.join("input/vface.face_scales526.f32le"));
        let ebo = read_u16(&root.join("input/vface.topology2982.u16le"));
        assert_eq!(positions.len(), VERTEX_COUNT);
        assert_eq!(texcoord.len(), VERTEX_COUNT);
        assert_eq!(offset_texcoord.len(), VERTEX_COUNT);
        assert_eq!(face_scales.len(), VERTEX_COUNT);
        assert_eq!(ebo.len(), INDEX_COUNT);

        let (width, height) = (64u32, 48u32);
        let scaled: Vec<[f32; 2]> = positions
            .iter()
            .map(|p| [p[0] * width as f32 / 6048.0, p[1] * height as f32 / 4032.0])
            .collect();
        let mesh = Mesh {
            framebuffer_positions_w1: &scaled,
            texcoord: &texcoord,
            offset_texcoord: &offset_texcoord,
            face_scales: &face_scales,
            ebo: &ebo,
        };
        let source = source(width, height);
        let base = Image::solid(OFFSET_EDGE, OFFSET_EDGE, [0, 0, 0, 255]);
        #[cfg(windows)]
        let gpu = match crate::cuda_v2warp::V2Warp::open() {
            Ok(gpu) => gpu,
            Err(crate::cuda::CudaError::Unavailable(reason)) => {
                eprintln!("SKIP_NO_NVIDIA_DRIVER: {reason}");
                assert!(std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none());
                return;
            }
            Err(err) => panic!("NVIDIA path present but broken: {err:?}"),
        };

        for row in ROWS {
            let spec = row.spec();
            let [_, _, atlas_width, atlas_height] = spec.rectangle;
            let atlas_buf = std::fs::read(
                root.join("atlas")
                    .join(spec.atlas.replace(".png", ".rgba8")),
            )
            .unwrap();
            let atlas = Atlas {
                row,
                width: atlas_width,
                height: atlas_height,
                pixels: &atlas_buf,
            };
            let offset_pixels = fuse_offset_cpu(&base, atlas, 1.0).unwrap();
            assert!(
                offset_pixels.iter().any(|texel| texel != &[0.0; 4]),
                "{} must emit encoded offsets",
                spec.atlas
            );
            let inputs = Inputs {
                row,
                atlas,
                offset: OffsetTexture {
                    width: OFFSET_EDGE,
                    height: OFFSET_EDGE,
                    pixels: &offset_pixels,
                },
                mesh,
            };
            let cpu = crate::cuda_v2warp::apply_cpu(
                &source,
                inputs.offset,
                inputs.mesh,
                RULES,
                [1.0, 0.0, 0.0, 1.0],
                [width as f32, height as f32],
            )
            .unwrap();
            eprintln!(
                "VFACE_FIXTURE_CPU row={:?} pf={} atlas={} nonzero_offsets={}",
                row,
                spec.pf,
                spec.atlas,
                offset_pixels.iter().filter(|texel| **texel != [0.0; 4]).count()
            );
            #[cfg(windows)]
            {
                let (ui_left, ui_right) = match row {
                    Row::LeftPositive => (100.0, 0.0),
                    Row::RightPositive => (0.0, 100.0),
                    Row::LeftNegative => (-100.0, 0.0),
                    Row::RightNegative => (0.0, -100.0),
                };
                let actual = render(
                    &gpu,
                    &source,
                    inputs,
                    ui_left,
                    ui_right,
                    RULES,
                    [1.0, 0.0, 0.0, 1.0],
                    [width as f32, height as f32],
                )
                .unwrap()
                .expect("active row");
                let max_diff = actual
                    .warp
                    .image
                    .pixels
                    .iter()
                    .zip(&cpu.pixels)
                    .map(|(a, b)| a.abs_diff(*b))
                    .max()
                    .unwrap();
                eprintln!(
                    "VFACE_FIXTURE_GPU row={:?} pf={} device={} covered={} max_diff={} path={}",
                    row,
                    spec.pf,
                    actual.warp.select_dispatch.device,
                    actual.warp.coverage.iter().filter(|value| **value > 0).count(),
                    max_diff,
                    actual.warp.path
                );
                assert_eq!(actual.warp.path, crate::cuda_v2warp::PATH);
                assert_eq!(actual.warp.select_dispatch.kernel, "v2warp_select_kernel");
                assert_eq!(actual.warp.eval_dispatch.kernel, "v2warp_eval_kernel");
                assert!(max_diff <= 1, "{} GPU/CPU max diff {max_diff}", spec.atlas);
            }
            #[cfg(not(windows))]
            {
                let _ = &cpu;
            }
        }
    }

    fn read_pairs(path: &std::path::Path) -> Vec<[f32; 2]> {
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len(), VERTEX_COUNT * 8);
        bytes
            .chunks_exact(8)
            .map(|chunk| {
                [
                    f32::from_le_bytes(chunk[0..4].try_into().unwrap()),
                    f32::from_le_bytes(chunk[4..8].try_into().unwrap()),
                ]
            })
            .collect()
    }

    fn read_u16(path: &std::path::Path) -> Vec<u16> {
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len(), INDEX_COUNT * 2);
        bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect()
    }
}
