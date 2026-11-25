//! PF405 / `SubPoseRefine` `BodyRefine` — host 12x2 mesh and the downstream
//! degree consumer.
//!
//! All addresses below are RVAs in `portrait.exe` (the vendor binary shipped as
//! `pixcake.exe`, SHA-256 `A84519DA489BC52E4E914AF7880EDE43E3CE58992657723D02197161832464C7`).
//! The same code is present in `pix-worker.exe` at `0x3CA9FC0` (identical 0x402
//! bytes, different layout). In `pix-worker.exe` the address `0x4B7E0F0` is
//! *not* a function entry (it is the middle of a `call rel32` immediate); the
//! earlier write-up that cited it for pix-worker was a mis-bound disassembly.
//!
//! Recovered chain:
//!   `0x4B7C0D0` `BodyRefineMesh::Build` writes `cell+0x2B4 = 12`,
//!   `cell+0x2B8 = 2` (`movabs rax, 0x20000000C`), allocates the 3-float
//!   auxiliary stream, the 24 x float2 vertex stream at `cell+0x2C8`, and the
//!   uint16 index stream at `cell+0x2D8` with
//!   `index_count = (6*cols - 3)*rows - 6` (`cell+0x2D0 = 102`). The index
//!   builder at `0x4B7C2BC..0x4B7C388` is a 10-triangle fan to vertex 0 followed
//!   by a 12-quad wrapped ribbon; the emitted triangles are unit-tested below.
//!   `0x4B7E0F0` (`0x402` bytes, function-entry in `.pdata`) generates the
//!   vertices from three record points plus a query point. It has **no degree
//!   input**; only `rows = [cell+0x2B4]`, `cols = [cell+0x2B8]`,
//!   `width = [param]`, `height = [param+4]` and the four points are read.
//!   `0x4B7E46A` issues one indexed GL draw through `[vtable+0x120]` with
//!   `vertex_count = rows*cols = 24` and `index_count = [cell+0x2D0] = 102`.
//!
//! Vertex law (all constants are f32 immediates in the function):
//!   center  = (record0 + record1) * 0.5
//!   origin  = (query + (record1b + center) * 0.5) * 0.5
//!   radial0 = 0.7 * |center - record1b|            (f32 constant 0.7)
//!   radial1 = 5.5 * radial0
//!   angle_r = (r * (1/12)) * PI * 2                (r = 0..11)
//!   col 0:     v = (0.8 * (radial0 * (0.8*cos)), radial0 * sin) + origin
//!   col t>0:   radius = (1-t)*radial0 + t*radial1, t = c/(cols-1)
//!              v = (0.3 * (radius * (0.8*cos)), radius * sin) + origin
//!   vertex = v / (width, height)                   (multiply by 1/w, 1/h)
//!
//! Trig-slot caveat: the function obtains sin/cos through two indirect pointers
//! loaded from `.rdata` (`0x9A58478`, `0x9A58330`) that are resolved by the
//! vendor's protected loader; the pointed-to slots do not contain executable
//! RVAs in the file image, so which slot is `sin` and which is `cos` is
//! `LIVE_ONLY` from static bytes. The two coefficients are used on opposite
//! axes, so this is a 90-degree rotation of the ring. [`TrigOrder`] makes the
//! choice explicit; [`body_refine_vertices`] uses `X = cos, Y = sin`, matching
//! the first port, and this is recorded as `INFERRED`.
//!
//! Degree: the control curve `[0, 0.5, 1] -> [-2, 0, 2]`
//! (`crates/portrait/src/kernels/body_geometry_params.rs`, PF405) never reaches
//! this producer. `BodyRefine.frag` computes `degree = 1 - mask.r` per pixel and
//! lerps `v_texcoordOut -> v_texcoordOut1` when `degree > 0.01`; the slider
//! therefore has to be baked into the mask texture (or into the record points)
//! upstream of this function, and that binding is not proven here. The degree
//! entry points below are therefore fail-closed refusals.

use crate::kernels::body_geometry::Vec2;
use crate::Refusal;

/// `[cell+0x2B4]` written by `0x4B7C0E2` (`movabs rax, 0x20000000C`).
pub const BODY_REFINE_ROWS: u32 = 12;
/// `[cell+0x2B8]` written by the same instruction.
pub const BODY_REFINE_COLS: u32 = 2;
/// `rows * cols`.
pub const BODY_REFINE_VERTEX_COUNT: usize = 24;
/// `[cell+0x2D0] = (6*cols - 3)*rows - 6`.
pub const BODY_REFINE_INDEX_COUNT: usize = 102;

/// Which trig coefficient feeds which axis. `XCosYSin` is the first port's
/// choice (`INFERRED`: the two indirect slots are loader-resolved).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrigOrder {
    /// `x` uses `cos(angle) * 0.8`, `y` uses `sin(angle)`.
    XCosYSin,
    /// `x` uses `sin(angle) * 0.8`, `y` uses `cos(angle)`.
    XSinYCos,
}

/// Recovered 12x2 mesh in normalized image coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyRefineMesh {
    pub rows: u32,
    pub cols: u32,
    /// Row-major `[row * cols + col]`, each component divided by `(width, height)`.
    pub vertices: Vec<[f32; 2]>,
    /// 10-triangle fan to vertex 0, then 12 wrapped quads per column step.
    pub indices: Vec<u16>,
}

fn finite_points(points: &[[f32; 2]]) -> bool {
    points.iter().flatten().all(|value| value.is_finite())
}

/// Exact index stream of `0x4B7C2BC..0x4B7C388` for the recovered `rows/cols`.
///
/// The vendor builds `(6*cols - 3)*rows - 6` indices: first a fan
/// `(0, (r-1)*cols, r*cols)` for `r = 2..rows`, then, for every column step
/// `c = 0..cols-2`, one wrapped quad `(a, b, a+1, b, a+1, b+1)` per row with
/// `b = ((row+1) % rows) * cols + c`.
pub fn body_refine_indices(rows: u32, cols: u32) -> Result<Vec<u16>, Refusal> {
    if rows < 3 || cols < 2 {
        return Err(Refusal::ValueOutOfRange("PF405_MESH_SHAPE"));
    }
    let vertex_count = rows as usize * cols as usize;
    if vertex_count > u16::MAX as usize {
        return Err(Refusal::ValueOutOfRange("PF405_MESH_SHAPE"));
    }
    let expected = (6 * cols as usize - 3) * rows as usize - 6;
    let mut indices = Vec::with_capacity(expected);
    for row in 2..rows {
        indices.push(0);
        indices.push(((row - 1) * cols) as u16);
        indices.push((row * cols) as u16);
    }
    for column_step in 0..cols - 1 {
        for row in 0..rows {
            let next = if row + 1 == rows { 0 } else { row + 1 };
            let a = row * cols + column_step;
            let b = next * cols + column_step;
            indices.push(a as u16);
            indices.push(b as u16);
            indices.push((a + 1) as u16);
            indices.push(b as u16);
            indices.push((a + 1) as u16);
            indices.push((b + 1) as u16);
        }
    }
    debug_assert_eq!(indices.len(), expected);
    Ok(indices)
}

fn build_vertices(
    width: u32,
    height: u32,
    query: [f32; 2],
    record0: [f32; 2],
    record1: [f32; 2],
    record1b: [f32; 2],
    order: TrigOrder,
) -> Result<BodyRefineMesh, Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("image"));
    }
    if !finite_points(&[query, record0, record1, record1b]) {
        return Err(Refusal::ValueOutOfRange("PF405_LANDMARKS"));
    }

    // f32 throughout, matching addps/mulps/sqrtss in `0x4B7E0F0`.
    let center = [
        (record0[0] + record1[0]) * 0.5,
        (record0[1] + record1[1]) * 0.5,
    ];
    let origin = [
        (query[0] + (record1b[0] + center[0]) * 0.5) * 0.5,
        (query[1] + (record1b[1] + center[1]) * 0.5) * 0.5,
    ];
    let dx = center[0] - record1b[0];
    let dy = center[1] - record1b[1];
    let radial0 = 0.7f32 * (dx * dx + dy * dy).sqrt();
    let radial1 = 5.5f32 * radial0;

    let inv_width = 1.0f32 / width as f32;
    let inv_height = 1.0f32 / height as f32;
    let row_step = 1.0f32 / BODY_REFINE_ROWS as f32;
    let col_step = 1.0f32 / (BODY_REFINE_COLS as f32 - 1.0);

    let mut vertices = Vec::with_capacity(BODY_REFINE_VERTEX_COUNT);
    for row in 0..BODY_REFINE_ROWS {
        let angle = (row as f32 * row_step) * std::f32::consts::PI * 2.0;
        let sine = angle.sin();
        let cosine = angle.cos();
        let (x_trig, y_trig) = match order {
            TrigOrder::XCosYSin => (cosine, sine),
            TrigOrder::XSinYCos => (sine, cosine),
        };

        // Column 0: radius0 with the 0.8 * (0.8 * cos) x compression.
        let x = 0.8f32 * (radial0 * (0.8f32 * x_trig)) + origin[0];
        let y = radial0 * y_trig + origin[1];
        vertices.push([x * inv_width, y * inv_height]);

        for col in 1..BODY_REFINE_COLS {
            let t = col as f32 * col_step;
            let radius = (1.0f32 - t) * radial0 + t * radial1;
            let x = 0.3f32 * (radius * (0.8f32 * x_trig)) + origin[0];
            let y = radius * y_trig + origin[1];
            vertices.push([x * inv_width, y * inv_height]);
        }
    }

    let indices = body_refine_indices(BODY_REFINE_ROWS, BODY_REFINE_COLS)?;
    Ok(BodyRefineMesh {
        rows: BODY_REFINE_ROWS,
        cols: BODY_REFINE_COLS,
        vertices,
        indices,
    })
}

/// `0x4B7E0F0` vertex producer for the recovered PF405 instance (`12 x 2`).
///
/// `query`, `record0`, `record1`, `record1b` are the four points read by the
/// binary: `record0/record1/record1b` are vector slots `0/21/22` of the record
/// array returned by `0x4B44AC0(param, 1)`, and `query` is slot `0` of the
/// array read by `0x4B446A0(param, out, 0)`. They are caller-supplied; nothing
/// is detected here. Uses [`TrigOrder::XCosYSin`].
pub fn body_refine_vertices(
    width: u32,
    height: u32,
    query: [f32; 2],
    record0: [f32; 2],
    record1: [f32; 2],
    record1b: [f32; 2],
) -> Result<BodyRefineMesh, Refusal> {
    build_vertices(
        width,
        height,
        query,
        record0,
        record1,
        record1b,
        TrigOrder::XCosYSin,
    )
}

/// Same as [`body_refine_vertices`] with an explicit trig-slot choice; see the
/// module note about `LIVE_ONLY` sin/cos binding.
pub fn body_refine_vertices_with_trig_order(
    width: u32,
    height: u32,
    query: [f32; 2],
    record0: [f32; 2],
    record1: [f32; 2],
    record1b: [f32; 2],
    order: TrigOrder,
) -> Result<BodyRefineMesh, Refusal> {
    build_vertices(width, height, query, record0, record1, record1b, order)
}

/// Fail-closed degree entry: the recovered vertex producer has no degree input.
///
/// `0x4B7E0F0` reads only the four points, `rows`, `cols`, `width` and `height`;
/// the PF405 control curve never reaches it. Returning a scaled mesh here would
/// invent a law. The per-pixel degree lives in `BodyRefine.frag` (see
/// [`body_refine_fragment_uv`]).
pub fn body_refine_vertices_with_degree(
    width: u32,
    height: u32,
    query: [f32; 2],
    record0: [f32; 2],
    record1: [f32; 2],
    record1b: [f32; 2],
    degree: f32,
) -> Result<BodyRefineMesh, Refusal> {
    let _ = (width, height, query, record0, record1, record1b);
    if !degree.is_finite() {
        return Err(Refusal::ValueOutOfRange("PF405_DEGREE"));
    }
    Err(Refusal::MissingInput("PF405_DEGREE_TO_VERTEX"))
}

/// Per-pixel UV field of the 12x2 mesh.
///
/// The host does not rasterize this field on the CPU: `0x4B7E46A` uploads the
/// 24 vertices and the 102 uint16 indices and issues one indexed draw through
/// the render-object vtable slot `+0x120`; the field is produced by GPU
/// rasterization of `glsl/PoseRefine/BodyRefine.vert`. The vertex shader
/// exposes `v_texcoordOut = a_texcoord` (source UV) and
/// `v_texcoordOut1 = gl_Position.xy * 0.5 + 0.5` (target UV), but which host
/// stream binds to `a_texcoord` is not statically resolved (the auxiliary
/// `cell+0x2C0` stream is a column-fraction vec3 and the draw slot target is
/// loader-resolved). Producing a source/target field here would fabricate the
/// missing binding.
pub fn body_refine_uv_field(
    width: u32,
    height: u32,
    query: [f32; 2],
    record0: [f32; 2],
    record1: [f32; 2],
    record1b: [f32; 2],
) -> Result<(Vec<Vec2>, Vec<Vec2>), Refusal> {
    let _ = (query, record0, record1, record1b);
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("image"));
    }
    Err(Refusal::MissingInput("PF405_SOURCE_UV_ATTRIBUTE"))
}

/// `BodyRefine.frag` degree: `1.0 - mask.r / 255.0`.
pub fn body_refine_degree(mask_r: u8) -> f32 {
    1.0f32 - mask_r as f32 / 255.0f32
}

/// `BodyRefine.frag` non-`UVMAP` consumer: when `degree > 0.01`,
/// `uv + (target_uv - uv) * degree`; otherwise the source UV.
///
/// This is the only place the recovered pipeline consumes a degree; the host
/// mesh does not.
pub fn body_refine_fragment_uv(
    source_uv: (f32, f32),
    target_uv: (f32, f32),
    mask_r: u8,
) -> Result<(f32, f32), Refusal> {
    if ![source_uv.0, source_uv.1, target_uv.0, target_uv.1]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("PF405_UV"));
    }
    let degree = body_refine_degree(mask_r);
    if degree > 0.01 {
        Ok((
            source_uv.0 + (target_uv.0 - source_uv.0) * degree,
            source_uv.1 + (target_uv.1 - source_uv.1) * degree,
        ))
    } else {
        Ok(source_uv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 100;
    const H: u32 = 400;
    // center = (20, 30); record1b = (25, 35); radial0 = 0.7*sqrt(50).
    const QUERY: [f32; 2] = [50.0, 200.0];
    const REC0: [f32; 2] = [10.0, 20.0];
    const REC1: [f32; 2] = [30.0, 40.0];
    const REC1B: [f32; 2] = [25.0, 35.0];

    fn mesh() -> BodyRefineMesh {
        body_refine_vertices(W, H, QUERY, REC0, REC1, REC1B).unwrap()
    }

    #[test]
    fn shape_and_index_count_match_binary() {
        let m = mesh();
        assert_eq!((m.rows, m.cols), (12, 2));
        assert_eq!(m.vertices.len(), BODY_REFINE_VERTEX_COUNT);
        assert_eq!(m.indices.len(), BODY_REFINE_INDEX_COUNT);
    }

    #[test]
    fn row_zero_matches_hand_computed_constants() {
        // origin = (query + (record1b + center)*0.5)*0.5 = (36.25, 116.25)
        // radial0 = 0.7*sqrt(50) = 4.9497475 ; radial1 = 27.223612
        // angle = 0 -> cos = 1, sin = 0
        let m = mesh();
        let radial0 = 0.7f32 * 50.0f32.sqrt();
        let radial1 = 5.5f32 * radial0;
        let ox = 36.25f32;
        let oy = 116.25f32;
        let col0_x = 0.8 * (radial0 * 0.8) + ox;
        let col1_x = 0.3 * (radial1 * 0.8) + ox;
        let want0 = [col0_x / W as f32, oy / H as f32];
        let want1 = [col1_x / W as f32, oy / H as f32];
        for (got, want) in [(&m.vertices[0], want0), (&m.vertices[1], want1)].into_iter() {
            assert!((got[0] - want[0]).abs() < 1e-6, "{got:?} vs {want:?}");
            assert!((got[1] - want[1]).abs() < 1e-6, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn indices_match_binary_fan_then_wrapped_ribbon() {
        let m = mesh();
        // Fan over rows 2..11.
        for (k, row) in (2u16..12).enumerate() {
            let base = k * 3;
            assert_eq!(&m.indices[base..base + 3], &[0, (row - 1) * 2, row * 2]);
        }
        // Ribbon starts at index 30: row 0 -> (0, 2, 1, 2, 1, 3).
        assert_eq!(&m.indices[30..36], &[0, 2, 1, 2, 1, 3]);
        // Last row wraps: row 11 -> (22, 0, 23, 0, 23, 1).
        assert_eq!(&m.indices[96..102], &[22, 0, 23, 0, 23, 1]);
    }

    #[test]
    fn coincident_records_collapse_to_origin_without_nan() {
        let same = [25.0, 35.0];
        let m = body_refine_vertices(W, H, QUERY, same, same, same).unwrap();
        assert!(m.vertices.iter().flatten().all(|v| v.is_finite()));
        let ox = (QUERY[0] + (same[0] + same[0]) * 0.5) * 0.5;
        let oy = (QUERY[1] + (same[1] + same[1]) * 0.5) * 0.5;
        for v in &m.vertices {
            assert!((v[0] - ox / W as f32).abs() < 1e-6);
            assert!((v[1] - oy / H as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn refuses_non_finite_landmark() {
        let bad = [f32::NAN, 0.0];
        assert_eq!(
            body_refine_vertices(W, H, bad, REC0, REC1, REC1B),
            Err(Refusal::ValueOutOfRange("PF405_LANDMARKS"))
        );
        let inf = [f32::INFINITY, 0.0];
        assert_eq!(
            body_refine_vertices(W, H, QUERY, REC0, REC1, inf),
            Err(Refusal::ValueOutOfRange("PF405_LANDMARKS"))
        );
    }

    #[test]
    fn refuses_zero_image_size() {
        assert_eq!(
            body_refine_vertices(0, H, QUERY, REC0, REC1, REC1B),
            Err(Refusal::ShapeMismatch("image"))
        );
        assert_eq!(
            body_refine_vertices(W, 0, QUERY, REC0, REC1, REC1B),
            Err(Refusal::ShapeMismatch("image"))
        );
    }

    #[test]
    fn degree_does_not_enter_the_vertex_producer() {
        for degree in [0.0f32, 2.0, -2.0] {
            assert_eq!(
                body_refine_vertices_with_degree(W, H, QUERY, REC0, REC1, REC1B, degree),
                Err(Refusal::MissingInput("PF405_DEGREE_TO_VERTEX"))
            );
        }
        assert_eq!(
            body_refine_vertices_with_degree(W, H, QUERY, REC0, REC1, REC1B, f32::NAN),
            Err(Refusal::ValueOutOfRange("PF405_DEGREE"))
        );
    }

    #[test]
    fn cpu_uv_field_is_not_proven() {
        assert_eq!(
            body_refine_uv_field(W, H, QUERY, REC0, REC1, REC1B),
            Err(Refusal::MissingInput("PF405_SOURCE_UV_ATTRIBUTE"))
        );
        assert_eq!(
            body_refine_uv_field(0, H, QUERY, REC0, REC1, REC1B),
            Err(Refusal::ShapeMismatch("image"))
        );
    }

    #[test]
    fn trig_order_rotates_the_ring() {
        let a = body_refine_vertices_with_trig_order(
            W,
            H,
            QUERY,
            REC0,
            REC1,
            REC1B,
            TrigOrder::XCosYSin,
        )
        .unwrap();
        let b = body_refine_vertices_with_trig_order(
            W,
            H,
            QUERY,
            REC0,
            REC1,
            REC1B,
            TrigOrder::XSinYCos,
        )
        .unwrap();
        assert_ne!(a.vertices, b.vertices);
        // At row 3 (90 deg) the two orders swap the axes.
        assert!(a.vertices.iter().flatten().all(|v| v.is_finite()));
        assert!(b.vertices.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn fragment_degree_is_one_minus_mask() {
        assert!((body_refine_degree(255) - 0.0).abs() < 1e-7);
        assert!((body_refine_degree(0) - 1.0).abs() < 1e-7);
        assert!((body_refine_degree(128) - (1.0 - 128.0 / 255.0)).abs() < 1e-7);
    }

    #[test]
    fn fragment_uv_lerps_between_source_and_target() {
        let src = (0.2f32, 0.3f32);
        let dst = (0.6f32, 0.7f32);
        // mask 255 -> degree 0 -> source unchanged.
        assert_eq!(body_refine_fragment_uv(src, dst, 255).unwrap(), src);
        // mask 0 -> degree 1 -> target.
        let full = body_refine_fragment_uv(src, dst, 0).unwrap();
        assert!((full.0 - dst.0).abs() < 1e-6 && (full.1 - dst.1).abs() < 1e-6);
        // half way.
        let half = body_refine_fragment_uv(src, dst, 128).unwrap();
        let d = body_refine_degree(128);
        assert!((half.0 - (src.0 + (dst.0 - src.0) * d)).abs() < 1e-6);
        assert_eq!(
            body_refine_fragment_uv((f32::NAN, 0.0), dst, 0),
            Err(Refusal::ValueOutOfRange("PF405_UV"))
        );
    }
}
