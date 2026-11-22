//! PF8033 mouth-corner smile, PF21076 closed-mouth smile and PF100007
//! open-mouth smile.
//!
//! PE: `a84519da489bc52e4e914af7880ede43e3ce58992657723d02197161832464c7`
//!
//! ```text
//! PF8033   DL3D_Warp / Effect3DSmileModel
//!          0x4B666C0  config parser
//!          0xC013BB0  RTTI "?AVEffect3DSmileModel@PixCook@@"
//!          0x91CF128  model vtable
//! PF21076  0x42C2618  node anchor
//!          0x4F35B98  "CloseMou" compare
//!          vendor SmileV1 output/consumer: OPEN (see pf21076_*)
//! PF100007 0x42C2550  node anchor / 0x4F35150 ctor
//!          0x4F352D0  active gate ("Grinning" + value > 0.001)
//!          0x5E367E0  render consumer / 0x5E36875 PF read
//! ```
//!
//! PF8033 and PF100007 have explicit-input pixel warps. PF21076 does **not**:
//! the previous lane's landmark path is an alternative, not the vendor tensor,
//! so this module refuses instead of inventing it.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// `x' = m00*x + m01*y + m02*z + m03` (row 1 analogous).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine3x4 {
    pub m: [f32; 8],
}

impl Affine3x4 {
    pub fn apply(&self, p: Vec3) -> Vec2 {
        Vec2 {
            x: self.m[0] * p.x + self.m[1] * p.y + self.m[2] * p.z + self.m[3],
            y: self.m[4] * p.x + self.m[5] * p.y + self.m[6] * p.z + self.m[7],
        }
    }

    pub fn identity() -> Self {
        Self {
            m: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    MissingInput(&'static str),
    ShapeMismatch(&'static str),
    ValueOutOfRange(&'static str),
    VendorConsumerOpen(u32),
}

// ---------------------------------------------------------------- PF8033

/// `shape = mean + degree01 * (reshape - mean)`, 2206 vertices / 4269 faces.
pub fn pf8033_morph(mean: &[Vec3], reshape: &[Vec3], degree01: f32) -> Result<Vec<Vec3>, Refusal> {
    if mean.is_empty() || mean.len() != reshape.len() {
        return Err(Refusal::ShapeMismatch("pf8033 mean/reshape"));
    }
    if !degree01.is_finite() || !(0.0..=1.0).contains(&degree01) {
        return Err(Refusal::ValueOutOfRange("pf8033 degree01"));
    }
    let mut out = Vec::with_capacity(mean.len());
    for (m, r) in mean.iter().zip(reshape) {
        for v in [m.x, m.y, m.z, r.x, r.y, r.z] {
            if !v.is_finite() {
                return Err(Refusal::ValueOutOfRange("pf8033 vertex"));
            }
        }
        out.push(Vec3 {
            x: m.x + degree01 * (r.x - m.x),
            y: m.y + degree01 * (r.y - m.y),
            z: m.z + degree01 * (r.z - m.z),
        });
    }
    Ok(out)
}

/// Per-vertex 2D displacement `project(morphed) - project(mean)`.
pub fn pf8033_vertex_field(
    mean: &[Vec3],
    morphed: &[Vec3],
    projection: Affine3x4,
) -> Result<Vec<Vec2>, Refusal> {
    if mean.is_empty() || mean.len() != morphed.len() {
        return Err(Refusal::ShapeMismatch("pf8033 mean/morphed"));
    }
    if projection.m.iter().any(|v| !v.is_finite()) {
        return Err(Refusal::ValueOutOfRange("pf8033 projection"));
    }
    Ok(mean
        .iter()
        .zip(morphed)
        .map(|(a, b)| {
            let pa = projection.apply(*a);
            let pb = projection.apply(*b);
            Vec2 {
                x: pb.x - pa.x,
                y: pb.y - pa.y,
            }
        })
        .collect())
}

/// `sx = x + degree01*mask/255*dx`, `sy` alike; bilinear BGRA, alpha copied.
pub fn pf8033_dense_warp(
    source_bgra: &[u8],
    width: u32,
    height: u32,
    dx: &[f32],
    dy: &[f32],
    mask: &[u8],
    degree01: f32,
) -> Result<Vec<u8>, Refusal> {
    if crate::Image::byte_len(width, height) != Some(source_bgra.len()) {
        return Err(Refusal::ShapeMismatch("pf8033 source_bgra"));
    }
    let pixels = source_bgra.len() / 4;
    if dx.len() != pixels || dy.len() != pixels || mask.len() != pixels {
        return Err(Refusal::MissingInput("pf8033 dx/dy/mask"));
    }
    if !degree01.is_finite() || !(0.0..=1.0).contains(&degree01) {
        return Err(Refusal::ValueOutOfRange("pf8033 degree01"));
    }
    if dx.iter().chain(dy).any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("pf8033 displacement"));
    }
    Ok(warp_bgra(
        source_bgra,
        width,
        height,
        dx,
        dy,
        mask,
        degree01,
        0,
        0,
        width,
        height,
    ))
}

/// Pixel threshold above which `*_parallel` variants use row bands. Measured
/// on the recovered warp: 256² is the first size where every thread count
/// tested is faster than serial; 128² and below stay serial.
pub const SMILE_PARALLEL_MIN_PIXELS: usize = 65_536;

/// Row-band form of [`pf8033_dense_warp`]. Output is byte-identical to the
/// serial function; bands are disjoint and never share a row.
pub fn pf8033_dense_warp_parallel(
    source_bgra: &[u8],
    width: u32,
    height: u32,
    dx: &[f32],
    dy: &[f32],
    mask: &[u8],
    degree01: f32,
    threads: usize,
) -> Result<Vec<u8>, Refusal> {
    if crate::Image::byte_len(width, height) != Some(source_bgra.len()) {
        return Err(Refusal::ShapeMismatch("pf8033 source_bgra"));
    }
    let pixels = source_bgra.len() / 4;
    if dx.len() != pixels || dy.len() != pixels || mask.len() != pixels {
        return Err(Refusal::MissingInput("pf8033 dx/dy/mask"));
    }
    if !degree01.is_finite() || !(0.0..=1.0).contains(&degree01) {
        return Err(Refusal::ValueOutOfRange("pf8033 degree01"));
    }
    if dx.iter().chain(dy).any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("pf8033 displacement"));
    }
    if threads <= 1 || pixels < SMILE_PARALLEL_MIN_PIXELS {
        return Ok(warp_bgra(
            source_bgra, width, height, dx, dy, mask, degree01, 0, 0, width, height,
        ));
    }
    let band_rows = (height as usize).div_ceil(threads).max(1);
    let band_bytes = band_rows * width as usize * 4;
    let mut out = vec![0u8; source_bgra.len()];
    std::thread::scope(|scope| {
        for (band, chunk) in out.chunks_mut(band_bytes).enumerate() {
            let y0 = (band * band_rows) as u32;
            let rows = (chunk.len() / (width as usize * 4)) as u32;
            scope.spawn(move || {
                warp_bgra_rows(
                    chunk, source_bgra, width, height, dx, dy, mask, degree01,
                    0, 0, width, height, y0, rows,
                );
            });
        }
    });
    Ok(out)
}

// --------------------------------------------------------------- PF100007

pub const PF100007_ACTIVE_GATE: f32 = 0.001;

/// `0x4F352D0`: degree = ui/100, active only for material "Grinning".
pub fn pf100007_gate(ui: f32, material: Option<&str>) -> Result<(bool, f32), Refusal> {
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("pf100007 ui"));
    }
    let degree = ui / 100.0;
    if material != Some("Grinning") {
        return Ok((false, degree));
    }
    Ok((degree > PF100007_ACTIVE_GATE, degree))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// SmileV1 pixel consumer: `weight = degree * lipMask*mouthMask/255^2`,
/// `source = p + weight * (dx, dy)`, bilinear BGRA, alpha copied.
#[allow(clippy::too_many_arguments)]
pub fn pf100007_apply(
    source_bgra: &[u8],
    width: u32,
    height: u32,
    degree: f32,
    lip_mask: &[u8],
    mouth_mask: &[u8],
    model_dx: &[f32],
    model_dy: &[f32],
    roi: Rect,
) -> Result<Vec<u8>, Refusal> {
    pf100007_check(
        source_bgra, width, height, degree, lip_mask, mouth_mask, model_dx, model_dy, roi,
    )?;
    let mut out = source_bgra.to_vec();
    pf100007_rows(
        &mut out, source_bgra, width, height, degree, lip_mask, mouth_mask, model_dx,
        model_dy, roi, roi.y, roi.height, 0,
    );
    Ok(out)
}

/// Row-band form of [`pf100007_apply`]. Output is byte-identical to serial;
/// bands are disjoint and only ROI rows are touched.
#[allow(clippy::too_many_arguments)]
pub fn pf100007_apply_parallel(
    source_bgra: &[u8],
    width: u32,
    height: u32,
    degree: f32,
    lip_mask: &[u8],
    mouth_mask: &[u8],
    model_dx: &[f32],
    model_dy: &[f32],
    roi: Rect,
    threads: usize,
) -> Result<Vec<u8>, Refusal> {
    let pixels = pf100007_check(
        source_bgra, width, height, degree, lip_mask, mouth_mask, model_dx, model_dy, roi,
    )?;
    if threads <= 1 || pixels < SMILE_PARALLEL_MIN_PIXELS || roi.height == 0 {
        let mut out = source_bgra.to_vec();
        pf100007_rows(
            &mut out, source_bgra, width, height, degree, lip_mask, mouth_mask, model_dx,
            model_dy, roi, roi.y, roi.height, 0,
        );
        return Ok(out);
    }
    let band_rows = (height as usize).div_ceil(threads).max(1);
    let band_bytes = band_rows * width as usize * 4;
    let mut out = source_bgra.to_vec();
    std::thread::scope(|scope| {
        for (band, chunk) in out.chunks_mut(band_bytes).enumerate() {
            let y0 = band * band_rows;
            let rows = chunk.len() / (width as usize * 4);
            let start = y0.max(roi.y as usize);
            let end = (y0 + rows).min((roi.y + roi.height) as usize);
            if start >= end {
                continue;
            }
            let band_y0 = y0 as u32;
            scope.spawn(move || {
                pf100007_rows(
                    chunk, source_bgra, width, height, degree, lip_mask, mouth_mask,
                    model_dx, model_dy, roi, start as u32, (end - start) as u32,
                    band_y0,
                );
            });
        }
    });
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn pf100007_check(
    source_bgra: &[u8],
    width: u32,
    height: u32,
    degree: f32,
    lip_mask: &[u8],
    mouth_mask: &[u8],
    model_dx: &[f32],
    model_dy: &[f32],
    roi: Rect,
) -> Result<usize, Refusal> {
    if crate::Image::byte_len(width, height) != Some(source_bgra.len()) {
        return Err(Refusal::ShapeMismatch("pf100007 bgra"));
    }
    let pixels = source_bgra.len() / 4;
    for (plane, name) in [
        (lip_mask.len(), "pf100007 lip_mask"),
        (mouth_mask.len(), "pf100007 mouth_mask"),
        (model_dx.len(), "pf100007 model_dx"),
        (model_dy.len(), "pf100007 model_dy"),
    ] {
        if plane != pixels {
            return Err(Refusal::MissingInput(name));
        }
    }
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("pf100007 degree"));
    }
    if roi.x > width || roi.y > height
        || roi.width > width - roi.x || roi.height > height - roi.y
    {
        return Err(Refusal::ValueOutOfRange("pf100007 roi"));
    }
    if model_dx.iter().chain(model_dy).any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("pf100007 displacement"));
    }
    Ok(pixels)
}

#[allow(clippy::too_many_arguments)]
fn pf100007_rows(
    out: &mut [u8],
    source_bgra: &[u8],
    width: u32,
    height: u32,
    degree: f32,
    lip_mask: &[u8],
    mouth_mask: &[u8],
    model_dx: &[f32],
    model_dy: &[f32],
    roi: Rect,
    start_y: u32,
    rows: u32,
    band_y0: u32,
) {
    for y in start_y..start_y + rows {
        for x in roi.x..roi.x + roi.width {
            let i = y as usize * width as usize + x as usize;
            let local = (y - band_y0) as usize * width as usize + x as usize;
            let support = (lip_mask[i] as f32 / 255.0) * (mouth_mask[i] as f32 / 255.0);
            let weight = degree * support;
            if support <= PF100007_ACTIVE_GATE || weight <= 0.0 {
                continue;
            }
            let sx = x as f32 + weight * model_dx[i];
            let sy = y as f32 + weight * model_dy[i];
            let (b, g, r) = sample_bilinear(source_bgra, width, height, sx, sy);
            out[local * 4] = b;
            out[local * 4 + 1] = g;
            out[local * 4 + 2] = r;
        }
    }
}

// ---------------------------------------------------------------- PF21076

pub const PF21076_STATUS: &str = "OPEN_VENDOR_CONSUMER";

/// Minimal live capture for the closed-mouth smile; the vendor SmileV1 tensor
/// is not statically present, so no warp is fabricated here.
pub const PF21076_CAPTURE: &[&str] = &[
    "node at 0x42C2618 / CloseMouth compare 0x4F35B98",
    "PF21076 value and material/model handle",
    "SmileV1 landmark/mesh or displacement field",
    "SmileV1 lip/mouth mask planes",
    "ROI expansion factor and face rect",
];

pub fn pf21076_warp() -> Result<(), Refusal> {
    Err(Refusal::VendorConsumerOpen(21076))
}

// --------------------------------------------------------------- helpers

fn sample_bilinear(source: &[u8], width: u32, height: u32, x: f32, y: f32) -> (u8, u8, u8) {
    let cx = x.clamp(0.0, (width - 1) as f32);
    let cy = y.clamp(0.0, (height - 1) as f32);
    let x0 = cx.floor() as u32;
    let y0 = cy.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = cx - x0 as f32;
    let fy = cy - y0 as f32;
    let row0 = y0 as usize * width as usize;
    let row1 = y1 as usize * width as usize;
    let mut rgb = [0u8; 3];
    for c in 0..3 {
        let p00 = source[(row0 + x0 as usize) * 4 + c] as f32;
        let p10 = source[(row0 + x1 as usize) * 4 + c] as f32;
        let p01 = source[(row1 + x0 as usize) * 4 + c] as f32;
        let p11 = source[(row1 + x1 as usize) * 4 + c] as f32;
        let top = p00 + (p10 - p00) * fx;
        let bottom = p01 + (p11 - p01) * fx;
        rgb[c] = (top + (bottom - top) * fy).clamp(0.0, 255.0) as u8;
    }
    (rgb[0], rgb[1], rgb[2])
}

#[allow(clippy::too_many_arguments)]
fn warp_bgra(
    source: &[u8],
    width: u32,
    height: u32,
    dx: &[f32],
    dy: &[f32],
    mask: &[u8],
    degree01: f32,
    ox: u32,
    oy: u32,
    ow: u32,
    oh: u32,
) -> Vec<u8> {
    let mut out = vec![0u8; source.len()];
    warp_bgra_rows(
        &mut out, source, width, height, dx, dy, mask, degree01, ox, oy, ow, oh, 0, height,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn warp_bgra_rows(
    out: &mut [u8],
    source: &[u8],
    width: u32,
    height: u32,
    dx: &[f32],
    dy: &[f32],
    mask: &[u8],
    degree01: f32,
    ox: u32,
    oy: u32,
    ow: u32,
    oh: u32,
    y0: u32,
    rows: u32,
) {
    for y in y0..y0 + rows {
        for x in 0..width {
            let global = y as usize * width as usize + x as usize;
            let local = (y - y0) as usize * width as usize + x as usize;
            out[local * 4 + 3] = source[global * 4 + 3];
            if x < ox || y < oy || x >= ox + ow || y >= oy + oh {
                out[local * 4..local * 4 + 3]
                    .copy_from_slice(&source[global * 4..global * 4 + 3]);
                continue;
            }
            let weight = mask[global] as f32 / 255.0 * degree01;
            let sx = x as f32 + weight * dx[global];
            let sy = y as f32 + weight * dy[global];
            let (b, g, r) = sample_bilinear(source, width, height, sx, sy);
            out[local * 4] = b;
            out[local * 4 + 1] = g;
            out[local * 4 + 2] = r;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_warps_reject_invalid_dimensions_and_displacements() {
        for (width, height) in [(0, 0), (u32::MAX, u32::MAX)] {
            assert_eq!(
                pf8033_dense_warp(&[], width, height, &[], &[], &[], 1.0),
                Err(Refusal::ShapeMismatch("pf8033 source_bgra"))
            );
            assert_eq!(
                pf100007_apply(
                    &[], width, height, 1.0, &[], &[], &[], &[],
                    Rect { x: 0, y: 0, width: 0, height: 0 },
                ),
                Err(Refusal::ShapeMismatch("pf100007 bgra"))
            );
        }
        let source = [10, 20, 30, 77];
        let roi = Rect { x: 0, y: 0, width: 1, height: 1 };
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                pf8033_dense_warp(&source, 1, 1, &[bad], &[0.0], &[255], 1.0),
                Err(Refusal::ValueOutOfRange("pf8033 displacement"))
            );
            assert_eq!(
                pf100007_apply(&source, 1, 1, 1.0, &[255], &[255], &[0.0], &[bad], roi),
                Err(Refusal::ValueOutOfRange("pf100007 displacement"))
            );
        }
    }

    #[test]
    fn open_mouth_roi_addition_cannot_overflow() {
        let source = [10, 20, 30, 77];
        for roi in [
            Rect { x: 1, y: 0, width: u32::MAX, height: 1 },
            Rect { x: 0, y: 1, width: 1, height: u32::MAX },
        ] {
            assert_eq!(
                pf100007_apply(&source, 1, 1, 1.0, &[255], &[255], &[0.0], &[0.0], roi),
                Err(Refusal::ValueOutOfRange("pf100007 roi"))
            );
        }
    }

    fn gradient(width: u32, height: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                out.extend_from_slice(&[(x * 40) as u8, (y * 40) as u8, 7, 200]);
            }
        }
        out
    }

    #[test]
    fn morph_halfway() {
        let mean = [Vec3 { x: 0.0, y: 0.0, z: 0.0 }];
        let reshape = [Vec3 { x: 2.0, y: 4.0, z: -2.0 }];
        let out = pf8033_morph(&mean, &reshape, 0.5).unwrap();
        assert_eq!(out[0], Vec3 { x: 1.0, y: 2.0, z: -1.0 });
        assert_eq!(
            pf8033_morph(&mean, &reshape, 1.5).unwrap_err(),
            Refusal::ValueOutOfRange("pf8033 degree01")
        );
    }

    #[test]
    fn vertex_field_uses_projection_difference() {
        let mean = [Vec3 { x: 0.0, y: 0.0, z: 0.0 }];
        let morphed = [Vec3 { x: 2.0, y: 3.0, z: 0.0 }];
        let field = pf8033_vertex_field(&mean, &morphed, Affine3x4::identity()).unwrap();
        assert_eq!(field[0], Vec2 { x: 2.0, y: 3.0 });
    }

    #[test]
    fn pf8033_zero_degree_is_identity() {
        let src = gradient(3, 3);
        let out = pf8033_dense_warp(&src, 3, 3, &[9.0; 9], &[9.0; 9], &[255; 9], 0.0).unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn pf8033_displaces_texel() {
        let src = gradient(3, 1);
        let mut dx = vec![0.0f32; 3];
        dx[1] = 1.0;
        let out = pf8033_dense_warp(&src, 3, 1, &dx, &[0.0; 3], &[255; 3], 1.0).unwrap();
        assert_eq!(&out[4..7], &[80, 0, 7]);
        assert_eq!(out[7], 200);
    }

    #[test]
    fn pf100007_gate_requires_grinning() {
        assert_eq!(pf100007_gate(50.0, None).unwrap(), (false, 0.5));
        assert_eq!(pf100007_gate(0.0, Some("Grinning")).unwrap(), (false, 0.0));
        assert_eq!(pf100007_gate(50.0, Some("Grinning")).unwrap(), (true, 0.5));
        assert_eq!(
            pf100007_gate(101.0, Some("Grinning")).unwrap_err(),
            Refusal::ValueOutOfRange("pf100007 ui")
        );
    }

    #[test]
    fn pf100007_zero_support_is_identity() {
        let src = gradient(2, 2);
        let out = pf100007_apply(
            &src,
            2,
            2,
            1.0,
            &[0; 4],
            &[255; 4],
            &[5.0; 4],
            &[5.0; 4],
            Rect { x: 0, y: 0, width: 2, height: 2 },
        )
        .unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn pf21076_stays_open() {
        assert_eq!(pf21076_warp().unwrap_err(), Refusal::VendorConsumerOpen(21076));
        assert_eq!(PF21076_STATUS, "OPEN_VENDOR_CONSUMER");
        assert_eq!(PF21076_CAPTURE.len(), 5);
    }

    #[test]
    fn parallel_bands_match_serial() {
        let (width, height) = (320u32, 240u32);
        let pixels = (width * height) as usize;
        let src = gradient(width, height);
        let dx: Vec<f32> = (0..pixels).map(|i| ((i % 37) as f32 - 18.0) * 0.25).collect();
        let dy: Vec<f32> = (0..pixels)
            .map(|i| (((i / 37) % 29) as f32 - 14.0) * 0.25)
            .collect();
        let mask: Vec<u8> = (0..pixels).map(|i| ((i * 7) % 256) as u8).collect();
        let serial =
            pf8033_dense_warp(&src, width, height, &dx, &dy, &mask, 0.7).unwrap();
        let parallel =
            pf8033_dense_warp_parallel(&src, width, height, &dx, &dy, &mask, 0.7, 5).unwrap();
        assert_eq!(serial, parallel);

        let lip: Vec<u8> = (0..pixels)
            .map(|i| if i % 3 == 0 { 0 } else { ((i * 11) % 256) as u8 })
            .collect();
        let mouth: Vec<u8> = (0..pixels).map(|i| ((i * 5) % 256) as u8).collect();
        let roi = Rect { x: 17, y: 13, width: 200, height: 150 };
        let a = pf100007_apply(&src, width, height, 0.7, &lip, &mouth, &dx, &dy, roi)
            .unwrap();
        let b = pf100007_apply_parallel(
            &src, width, height, 0.7, &lip, &mouth, &dx, &dy, roi, 5,
        )
        .unwrap();
        assert_eq!(a, b);
        let before_roi = 13 * width as usize * 4;
        assert_eq!(&a[..before_roi], &src[..before_roi]);
    }
}
/// Two smile rows have executable math; the closed-mouth row stays refused
/// until its vendor consumer is captured.
pub const META: &[crate::EffectMeta] = &[
    crate::EffectMeta {
        pf: 8033,
        key: "smileMouthCorner",
        label: "Smile mouth corner",
        access: crate::Access::Cpu,
        inputs: &["mean/reshape mesh", "projection matrix", "degree", "dense dx/dy/mask"],
    },
    crate::EffectMeta {
        pf: 100007,
        key: "smileDegree_openMouth",
        label: "Open-mouth smile",
        access: crate::Access::Cpu,
        inputs: &["Grinning material", "lip/mouth masks", "model dx/dy", "roi"],
    },
];
