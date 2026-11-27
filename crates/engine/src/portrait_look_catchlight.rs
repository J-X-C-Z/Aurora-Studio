//! Product-original per-eye catchlight overlay (opt-in, non-vendor).
//!
//! A look set may carry an active `Catchlights` sub-material that has no vendor
//! `ParamFlag` of its own. This module composites that material onto the source
//! frame as a **separate, product-original step**: it is not the vendor's PF111
//! (pupil/contact-lens) effect, it does not create or impersonate an official PF,
//! and it does not touch the historical PF111 path
//! ([`super::portrait_makeup_bundle`], `EyePupilMaskKind`). Nothing here is an
//! OEM-parity claim: the materials are the self-owned `AUCG…` catchlight set, drawn
//! procedurally by the eye-studio generator (no vendor asset is read, traced or
//! redistributed).
//!
//! This is the **CPU reference** for that capability. It is exact and testable
//! without a device, and it is the thing a future device pass has to match. It
//! does not claim to be a GPU path, and no stage may report this capability as
//! device-executed until a real dispatch exists.
//!
//! Five properties are enforced by construction rather than by convention:
//!
//! 1. **The two intensities never multiply.** The function takes the material's
//!    own `highlight_amount` and no iris amount at all; the composite is
//!    `rgb = tint·tex·α + rgb·(1−α)` with `α = tex.alpha · feather · highlight_amount`.
//!    Whoever wires it decides whether the iris material ran, but this step cannot
//!    scale itself by it.
//! 2. **Each eye gets its own local UV frame.** Both eyes' iris rings used to be
//!    fed through one region box ([`super::portrait_makeup_mesh2d::build_region_uv`]),
//!    which stretches one texture across the whole face. Here every eye derives an
//!    oriented frame from *its own* eight ring points, so the left and right
//!    materials are placed independently.
//! 3. **The iris is the only region that changes.** Pixels outside the ring (plus
//!    the feather band) keep their exact bytes, and the frame's alpha channel is
//!    never written — only the RGB of covered iris texels.
//! 4. **Real PNG alpha, filtered premultiplied.** [`CatchlightMaterial::Png`]
//!    decodes the authored PNG and keeps its straight alpha; sampling is done on a
//!    premultiplied copy with the shipped shader-rule sampler
//!    ([`aurora_portrait::raster_fragment::sample_bilinear`]) and divided back by the
//!    filtered alpha, so a soft alpha edge cannot bleed dark or foreign RGB inward.
//!    No mask is synthesized and no procedural texture is invented.
//! 5. **All present eyes are validated before the first write.** A malformed right
//!    eye refuses the whole call and leaves the frame byte-identical; there is no
//!    half-composited result.
//!
//! Only human catchlight materials are expressible: [`CatchlightMaterial`] has no
//! animal variant, so `AnimalCatchlights/AUCP*` cannot be passed to a human frame.
//! A missing ring, a missing material, a degenerate ring or an out-of-range
//! parameter is a typed [`Refusal`] — never a silent skip and never a guess.

use aurora_portrait::raster_fragment::sample_bilinear;
use aurora_portrait::{Image, Refusal};

/// Ring points used by [`super::portrait_makeup_mesh2d`] for the iris of one eye:
/// `IRIS_LEFT = [27, 28, …, 34]` and `IRIS_RIGHT = [45, 46, …, 52]` of the same
/// frame's 194-point template.
pub(crate) const CATCHLIGHT_RING_POINTS: usize = 8;

/// A neutral tint reproduces the authored material colour exactly.
pub(crate) const CATCHLIGHT_TINT_WHITE: [u8; 3] = [255, 255, 255];

/// Which eye a sample belongs to. Only used for receipts and docs; both eyes take
/// the same code path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CatchlightEyeSide {
    Left,
    Right,
}

impl CatchlightEyeSide {
    fn slot(self) -> usize {
        match self {
            Self::Left => 0,
            Self::Right => 1,
        }
    }
}

/// A catchlight material as the caller has it.
///
/// `Png` is the production form (the authored 300×300 `AUCG*.png`); the raw
/// variant exists so a caller that already decoded the texture once, and the unit
/// tests, do not pay for a second decode. Both are **straight** RGBA8: alpha is
/// opacity, RGB is the un-premultiplied colour.
pub(crate) enum CatchlightMaterial<'a> {
    Png(&'a [u8]),
    StraightRgba8 {
        rgba8: &'a [u8],
        width: u32,
        height: u32,
    },
}

/// One eye's input: the same-frame iris ring, the material, that material's own
/// intensity and an optional multiplicative tint.
pub(crate) struct CatchlightEye<'a> {
    /// Eight frame-space points, in ring order (the mesh2d iris ring verbatim).
    pub iris_ring: &'a [[f32; 2]; CATCHLIGHT_RING_POINTS],
    pub material: CatchlightMaterial<'a>,
    /// Independent of any iris/pupil amount, `0.0..=1.0`.
    pub highlight_amount: f32,
    /// Mirror the texture horizontally for this eye (asymmetric authored art).
    pub mirror_u: bool,
    /// Colour multiplier applied to the material's own RGB; white keeps it as
    /// authored. `[0,0,255]` turns the highlight cold, `[255,180,120]` warm.
    pub tint_rgb: [u8; 3],
}

impl<'a> CatchlightEye<'a> {
    /// Convenience constructor that keeps the call sites short: no mirroring and the
    /// authored material colour.
    pub(crate) fn new(
        iris_ring: &'a [[f32; 2]; CATCHLIGHT_RING_POINTS],
        material: CatchlightMaterial<'a>,
        highlight_amount: f32,
    ) -> Self {
        Self {
            iris_ring,
            material,
            highlight_amount,
            mirror_u: false,
            tint_rgb: CATCHLIGHT_TINT_WHITE,
        }
    }

    /// Same eye with an explicit tint, for the look sets that carry their own palette.
    pub(crate) fn with_tint(mut self, tint_rgb: [u8; 3]) -> Self {
        self.tint_rgb = tint_rgb;
        self
    }
}

/// Geometry knobs. Defaults are the shipped ones.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CatchlightParams {
    /// Shrink the ring toward its centroid, `0.0..=0.5` (0 keeps the ring hull).
    pub inset: f32,
    /// Soft edge, as a fraction of the ring's smaller half-extent, `0.0..=0.5`.
    pub feather: f32,
}

impl Default for CatchlightParams {
    fn default() -> Self {
        Self {
            inset: 0.0,
            feather: 0.18,
        }
    }
}

impl CatchlightParams {
    fn validate(self) -> Result<Self, Refusal> {
        let in_range = |value: f32| value.is_finite() && (0.0..=0.5).contains(&value);
        if !in_range(self.inset) || !in_range(self.feather) {
            return Err(Refusal::ValueOutOfRange("catchlight params"));
        }
        Ok(self)
    }
}

/// What the composite actually did, for the stage receipt.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct CatchlightReceipt {
    pub touched_pixels: usize,
    pub per_eye: [usize; 2],
    pub highlight_amounts: [f32; 2],
    pub material_sizes: [[u32; 2]; 2],
    pub tints: [[u8; 3]; 2],
}

impl CatchlightReceipt {
    pub(crate) fn touched(&self, side: CatchlightEyeSide) -> usize {
        self.per_eye[side.slot()]
    }
}

/// One eye's oriented, locally normalised sampling frame.
///
/// `u`/`v` are unit vectors in frame space, ordered so that `u` follows the ring's
/// longest axis. `[0,1]²` covers the ring's oriented bounding box (after `inset`),
/// which is why a round material stays round on a tilted eye instead of being
/// stretched across the face.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CatchlightUvFrame {
    pub center: [f32; 2],
    pub u: [f32; 2],
    pub v: [f32; 2],
    /// Half extents along `u` and `v`, in pixels.
    pub half_extent: [f32; 2],
    /// The clipped ring, already inset toward the centroid.
    pub ring: [[f32; 2]; CATCHLIGHT_RING_POINTS],
}

impl CatchlightUvFrame {
    /// Frame-space point -> local UV in `[0,1]²` (not clamped; the sampler clamps).
    pub(crate) fn uv_of(&self, point: [f32; 2]) -> [f32; 2] {
        let d = [point[0] - self.center[0], point[1] - self.center[1]];
        let su = d[0] * self.u[0] + d[1] * self.u[1];
        let sv = d[0] * self.v[0] + d[1] * self.v[1];
        [
            0.5 + su / (2.0 * self.half_extent[0]),
            0.5 + sv / (2.0 * self.half_extent[1]),
        ]
    }
}

/// Builds one eye's local frame from its own eight ring points.
///
/// Deterministic and allocation-free: centroid, 2×2 covariance principal axis, then
/// the extremes of the ring projected on that axis pair. A ring with no area (all
/// points equal, or collinear) is refused instead of producing a zero-scale frame.
pub(crate) fn catchlight_uv_frame(
    ring: &[[f32; 2]; CATCHLIGHT_RING_POINTS],
    inset: f32,
) -> Result<CatchlightUvFrame, Refusal> {
    if !(0.0..=0.5).contains(&inset) || !inset.is_finite() {
        return Err(Refusal::ValueOutOfRange("catchlight params"));
    }
    if ring.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("catchlight iris ring"));
    }

    let mut center = [0.0f32; 2];
    for point in ring {
        center[0] += point[0];
        center[1] += point[1];
    }
    center[0] /= CATCHLIGHT_RING_POINTS as f32;
    center[1] /= CATCHLIGHT_RING_POINTS as f32;

    let shrink = 1.0 - inset;
    let mut clipped = [[0.0f32; 2]; CATCHLIGHT_RING_POINTS];
    for (slot, point) in clipped.iter_mut().zip(ring.iter()) {
        slot[0] = center[0] + (point[0] - center[0]) * shrink;
        slot[1] = center[1] + (point[1] - center[1]) * shrink;
    }

    let (mut sxx, mut syy, mut sxy) = (0.0f32, 0.0f32, 0.0f32);
    for point in &clipped {
        let dx = point[0] - center[0];
        let dy = point[1] - center[1];
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    // Principal axis of a symmetric 2x2 covariance: 2θ = atan2(2·sxy, sxx − syy).
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let u = [theta.cos(), theta.sin()];
    let v = [-u[1], u[0]];

    let (mut half_u, mut half_v) = (0.0f32, 0.0f32);
    for point in &clipped {
        let dx = point[0] - center[0];
        let dy = point[1] - center[1];
        half_u = half_u.max((dx * u[0] + dy * u[1]).abs());
        half_v = half_v.max((dx * v[0] + dy * v[1]).abs());
    }
    // A usable frame needs both axes to span at least a pixel: below that the
    // material would collapse into a line and every UV in it would be noise.
    if !(half_u >= 0.5) || !(half_v >= 0.5) {
        return Err(Refusal::ShapeMismatch("catchlight iris ring"));
    }

    Ok(CatchlightUvFrame {
        center,
        u,
        v,
        half_extent: [half_u, half_v],
        ring: clipped,
    })
}

/// A validated, premultiplied material ready for sampling.
struct PreparedMaterial {
    size: [u32; 2],
    /// `rgb * alpha`, straight alpha kept: premultiplied filtering without fringes.
    premul: Image,
}

/// One eye after validation: everything the write loop needs, nothing it can fail on.
struct PreparedEye {
    frame: CatchlightUvFrame,
    material: PreparedMaterial,
    amount: f32,
    mirror_u: bool,
    tint: [f32; 3],
}

/// Composites the catchlight materials of both eyes onto `image` in place.
///
/// * `eyes[0]` is the left eye, `eyes[1]` the right; `None` means "this eye has no
///   ring in this frame" and leaves that side untouched — one-sided operation is a
///   normal input, both sides absent is a typed [`Refusal::MissingInput`].
/// * `highlight_amount` is this material's own intensity. `0.0` returns a zero
///   receipt and writes nothing, so zero strength is byte-identical to "off".
/// * `tint_rgb` multiplies the material's own colour; white is the authored colour.
/// * Only RGB bytes inside the (inset, feathered) iris ring change; the alpha
///   channel and everything outside the ring keep their exact values.
pub(crate) fn composite_catchlight_materials(
    image: &mut Image,
    eyes: [Option<CatchlightEye<'_>>; 2],
    params: CatchlightParams,
) -> Result<CatchlightReceipt, Refusal> {
    image.validate()?;
    let params = params.validate()?;
    if eyes.iter().all(Option::is_none) {
        return Err(Refusal::MissingInput("catchlight iris ring"));
    }

    let mut receipt = CatchlightReceipt::default();
    let mut prepared: [Option<PreparedEye>; 2] = [None, None];
    // Prepare both eyes first: a malformed right eye must not leave a
    // half-composited frame behind.
    for (slot, eye) in eyes.iter().enumerate() {
        let Some(eye) = eye else { continue };
        if !eye.highlight_amount.is_finite() || !(0.0..=1.0).contains(&eye.highlight_amount) {
            return Err(Refusal::ValueOutOfRange("catchlight highlight amount"));
        }
        receipt.highlight_amounts[slot] = eye.highlight_amount;
        receipt.tints[slot] = eye.tint_rgb;
        if eye.highlight_amount == 0.0 {
            continue;
        }
        let frame = catchlight_uv_frame(eye.iris_ring, params.inset)?;
        let material = prepare_material(&eye.material)?;
        receipt.material_sizes[slot] = material.size;
        prepared[slot] = Some(PreparedEye {
            frame,
            material,
            amount: eye.highlight_amount,
            mirror_u: eye.mirror_u,
            tint: [
                eye.tint_rgb[0] as f32 / 255.0,
                eye.tint_rgb[1] as f32 / 255.0,
                eye.tint_rgb[2] as f32 / 255.0,
            ],
        });
    }

    for (slot, eye) in prepared.iter().enumerate() {
        let Some(eye) = eye else { continue };
        let touched = composite_one_eye(image, eye, params.feather);
        receipt.per_eye[slot] = touched;
        receipt.touched_pixels += touched;
    }
    Ok(receipt)
}

fn prepare_material(material: &CatchlightMaterial<'_>) -> Result<PreparedMaterial, Refusal> {
    let (width, height, straight): (u32, u32, Vec<u8>) = match material {
        CatchlightMaterial::Png(bytes) => {
            let decoded = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)
                .map_err(|_| Refusal::ShapeMismatch("catchlight material png"))?
                .to_rgba8();
            let (width, height) = decoded.dimensions();
            (width, height, decoded.into_raw())
        }
        CatchlightMaterial::StraightRgba8 {
            rgba8,
            width,
            height,
        } => {
            let expected = (*width as usize)
                .checked_mul(*height as usize)
                .and_then(|count| count.checked_mul(4))
                .ok_or(Refusal::ShapeMismatch("catchlight material"))?;
            if rgba8.len() != expected {
                return Err(Refusal::ShapeMismatch("catchlight material"));
            }
            (*width, *height, rgba8.to_vec())
        }
    };
    if width < 2 || height < 2 {
        return Err(Refusal::ShapeMismatch("catchlight material"));
    }

    let mut premul = Vec::with_capacity((width as usize) * (height as usize) * 4);
    for texel in straight.chunks_exact(4) {
        let alpha = texel[3];
        for channel in 0..3 {
            let scaled = (texel[channel] as u32 * alpha as u32 + 127) / 255;
            premul.push(scaled as u8);
        }
        premul.push(alpha);
    }
    Ok(PreparedMaterial {
        size: [width, height],
        premul: Image {
            width,
            height,
            pixels: premul,
        },
    })
}

fn composite_one_eye(image: &mut Image, eye: &PreparedEye, feather: f32) -> usize {
    let frame = &eye.frame;
    let feather_px = feather * frame.half_extent[0].min(frame.half_extent[1]);
    let margin = feather_px + 1.0;
    let (ux, uy) = (frame.u[0].abs(), frame.u[1].abs());
    let (vx, vy) = (frame.v[0].abs(), frame.v[1].abs());
    let span_x = frame.half_extent[0] * ux + frame.half_extent[1] * vx + margin;
    let span_y = frame.half_extent[0] * uy + frame.half_extent[1] * vy + margin;

    let min_x = ((frame.center[0] - span_x).floor().max(0.0)) as u32;
    let max_x = ((frame.center[0] + span_x)
        .ceil()
        .min(image.width as f32 - 1.0))
    .max(0.0) as u32;
    let min_y = ((frame.center[1] - span_y).floor().max(0.0)) as u32;
    let max_y = ((frame.center[1] + span_y)
        .ceil()
        .min(image.height as f32 - 1.0))
    .max(0.0) as u32;

    let mut touched = 0usize;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            let signed = signed_distance_to_ring(point, &frame.ring);
            if signed <= 0.0 {
                continue;
            }
            let coverage = if feather_px > 0.0 {
                smoothstep(0.0, feather_px, signed)
            } else {
                1.0
            };
            let mut uv = frame.uv_of(point);
            if eye.mirror_u {
                uv[0] = 1.0 - uv[0];
            }
            // Premultiplied filtering: sample rgb*alpha, divide by the filtered
            // alpha, so a soft edge cannot pull dark RGB into the highlight.
            let sampled = sample_bilinear(&eye.material.premul, uv[0], uv[1]);
            let alpha = sampled[3];
            if alpha <= 0.0 {
                continue;
            }
            let alpha = alpha.clamp(0.0, 1.0) * coverage * eye.amount;
            if alpha <= 0.0 {
                continue;
            }
            let offset = (y as usize * image.width as usize + x as usize) * 4;
            for channel in 0..3 {
                let straight = (sampled[channel] / sampled[3]).clamp(0.0, 1.0);
                let tinted = straight * eye.tint[channel];
                let dst = image.pixels[offset + channel] as f32 / 255.0;
                let blended = tinted * alpha + dst * (1.0 - alpha);
                image.pixels[offset + channel] = (blended.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
            // The frame's own alpha is deliberately left alone.
            touched += 1;
        }
    }
    touched
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Distance from `point` to the ring boundary, positive inside the polygon.
fn signed_distance_to_ring(point: [f32; 2], ring: &[[f32; 2]; CATCHLIGHT_RING_POINTS]) -> f32 {
    let mut inside = false;
    let mut nearest = f32::INFINITY;
    for index in 0..CATCHLIGHT_RING_POINTS {
        let a = ring[index];
        let b = ring[(index + 1) % CATCHLIGHT_RING_POINTS];
        if (a[1] > point[1]) != (b[1] > point[1]) {
            let t = (point[1] - a[1]) / (b[1] - a[1]);
            if point[0] < a[0] + t * (b[0] - a[0]) {
                inside = !inside;
            }
        }
        nearest = nearest.min(distance_to_segment(point, a, b));
    }
    if inside { nearest } else { -nearest }
}

fn distance_to_segment(point: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [point[0] - a[0], point[1] - a[1]];
    let len_sq = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if len_sq > 0.0 {
        ((ap[0] * ab[0] + ap[1] * ab[1]) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let dx = ap[0] - ab[0] * t;
    let dy = ap[1] - ab[1] * t;
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: (u32, u32) = (64, 48);

    fn frame_image() -> Image {
        let mut pixels = Vec::with_capacity((FRAME.0 * FRAME.1 * 4) as usize);
        for y in 0..FRAME.1 {
            for x in 0..FRAME.0 {
                pixels.push((x * 3 % 251) as u8);
                pixels.push((y * 5 % 241) as u8);
                pixels.push(((x + y) * 7 % 233) as u8);
                pixels.push(255);
            }
        }
        Image::new(FRAME.0, FRAME.1, pixels).expect("frame")
    }

    /// Eight points on an ellipse, in ring order.
    fn ring(cx: f32, cy: f32, rx: f32, ry: f32) -> [[f32; 2]; CATCHLIGHT_RING_POINTS] {
        let mut points = [[0.0f32; 2]; CATCHLIGHT_RING_POINTS];
        for (slot, point) in points.iter_mut().enumerate() {
            let angle = slot as f32 * std::f32::consts::TAU / CATCHLIGHT_RING_POINTS as f32;
            point[0] = cx + rx * angle.cos();
            point[1] = cy + ry * angle.sin();
        }
        points
    }

    /// A 4×4 material whose four quadrants are distinct, so a misplaced UV is visible.
    fn quadrant_material() -> Vec<u8> {
        let mut rgba = Vec::with_capacity(4 * 4 * 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let left = x < 2;
                let top = y < 2;
                let (r, g, b) = match (left, top) {
                    (true, true) => (255u8, 0u8, 0u8),
                    (false, true) => (0, 255, 0),
                    (true, false) => (0, 0, 255),
                    (false, false) => (255, 255, 0),
                };
                rgba.extend_from_slice(&[r, g, b, 255]);
            }
        }
        rgba
    }

    fn straight(rgba: &[u8], width: u32, height: u32) -> CatchlightMaterial<'_> {
        CatchlightMaterial::StraightRgba8 {
            rgba8: rgba,
            width,
            height,
        }
    }

    fn alpha_of(image: &Image) -> Vec<u8> {
        image.pixels.chunks_exact(4).map(|p| p[3]).collect()
    }

    fn rgb_at(image: &Image, x: u32, y: u32) -> [u8; 3] {
        let offset = ((y * image.width + x) * 4) as usize;
        [
            image.pixels[offset],
            image.pixels[offset + 1],
            image.pixels[offset + 2],
        ]
    }

    #[test]
    fn zero_strength_is_byte_identical_and_writes_nothing() {
        let mut image = frame_image();
        let before = image.pixels.clone();
        let left = ring(16.0, 24.0, 6.0, 6.0);
        let right = ring(44.0, 24.0, 5.0, 6.0);
        let material = quadrant_material();
        let receipt = composite_catchlight_materials(
            &mut image,
            [
                Some(CatchlightEye::new(&left, straight(&material, 4, 4), 0.0)),
                Some(CatchlightEye::new(&right, straight(&material, 4, 4), 0.0)),
            ],
            CatchlightParams::default(),
        )
        .expect("zero strength is not an error");
        assert_eq!(receipt.touched_pixels, 0);
        assert_eq!(receipt.per_eye, [0, 0]);
        assert_eq!(
            image.pixels, before,
            "zero strength must not touch one byte"
        );
    }

    #[test]
    fn both_eyes_absent_is_a_typed_refusal() {
        let mut image = frame_image();
        assert_eq!(
            composite_catchlight_materials(&mut image, [None, None], CatchlightParams::default()),
            Err(Refusal::MissingInput("catchlight iris ring"))
        );
    }

    #[test]
    fn a_single_eye_changes_only_its_own_iris() {
        let mut image = frame_image();
        let before = image.pixels.clone();
        let left = ring(16.0, 24.0, 6.0, 6.0);
        let right = ring(44.0, 24.0, 5.0, 6.0);
        let material = [200u8, 180, 160, 255].repeat(16);
        let receipt = composite_catchlight_materials(
            &mut image,
            [
                Some(CatchlightEye::new(&left, straight(&material, 4, 4), 1.0)),
                None,
            ],
            CatchlightParams::default(),
        )
        .expect("one side is a normal input");

        assert!(receipt.per_eye[0] > 0, "the left iris must be touched");
        assert_eq!(receipt.per_eye[1], 0, "the absent side must stay at zero");
        assert_eq!(receipt.touched(CatchlightEyeSide::Left), receipt.per_eye[0]);

        for (x, y) in [(44u32, 24u32), (63, 47), (0, 0), (30, 10)] {
            let offset = ((y * image.width + x) * 4) as usize;
            assert_eq!(
                image.pixels[offset..offset + 4],
                before[offset..offset + 4],
                "pixel ({x},{y}) is outside the left ring"
            );
        }
        let offset = ((24 * image.width + 16) * 4) as usize;
        assert_ne!(image.pixels[offset..offset + 3], before[offset..offset + 3]);
        let reference = Image::new(FRAME.0, FRAME.1, before).expect("frame");
        assert_eq!(alpha_of(&image), alpha_of(&reference));
    }

    #[test]
    fn fully_transparent_material_is_a_no_op() {
        let mut image = frame_image();
        let before = image.pixels.clone();
        let ring_left = ring(16.0, 24.0, 6.0, 6.0);
        let clear = [10u8, 20, 30, 0].repeat(16);
        let receipt = composite_catchlight_materials(
            &mut image,
            [
                Some(CatchlightEye::new(&ring_left, straight(&clear, 4, 4), 1.0)),
                None,
            ],
            CatchlightParams::default(),
        )
        .expect("a fully transparent material is not an error");
        assert_eq!(receipt.touched_pixels, 0);
        assert_eq!(image.pixels, before);
    }

    #[test]
    fn each_eye_is_placed_by_its_own_frame_not_one_shared_uv() {
        let material = quadrant_material();
        let left = ring(16.0, 24.0, 6.0, 6.0);
        let right = ring(44.0, 26.0, 9.0, 4.0);
        let left_frame = catchlight_uv_frame(&left, 0.0).unwrap();
        let right_frame = catchlight_uv_frame(&right, 0.0).unwrap();
        assert_ne!(left_frame.center, right_frame.center);
        assert_ne!(left_frame.half_extent, right_frame.half_extent);

        let mut image = frame_image();
        let receipt = composite_catchlight_materials(
            &mut image,
            [
                Some(CatchlightEye::new(&left, straight(&material, 4, 4), 1.0)),
                Some(CatchlightEye::new(&right, straight(&material, 4, 4), 1.0)),
            ],
            CatchlightParams {
                inset: 0.0,
                feather: 0.0,
            },
        )
        .expect("two eyes");
        assert!(receipt.per_eye[0] > 0 && receipt.per_eye[1] > 0);

        for frame in [&left_frame, &right_frame] {
            let centre_uv = frame.uv_of(frame.center);
            assert!((centre_uv[0] - 0.5).abs() < 0.02 && (centre_uv[1] - 0.5).abs() < 0.02);
        }
        // A ring stretched horizontally keeps its own span; a shared face-wide box
        // could not produce both numbers at once.
        assert!(
            (right_frame.half_extent[0] - 9.0).abs() < 0.3,
            "right span {}",
            right_frame.half_extent[0]
        );
        assert!(
            (left_frame.half_extent[0] - 6.0).abs() < 0.3,
            "left span {}",
            left_frame.half_extent[0]
        );
    }

    #[test]
    fn mirroring_moves_the_texture_to_the_other_side_of_the_iris() {
        let material = quadrant_material();
        let left = ring(16.0, 24.0, 7.0, 7.0);
        let mut plain = frame_image();
        let mut mirrored = frame_image();
        let params = CatchlightParams {
            inset: 0.0,
            feather: 0.0,
        };
        composite_catchlight_materials(
            &mut plain,
            [
                Some(CatchlightEye::new(&left, straight(&material, 4, 4), 1.0)),
                None,
            ],
            params,
        )
        .unwrap();
        let mut eye = CatchlightEye::new(&left, straight(&material, 4, 4), 1.0);
        eye.mirror_u = true;
        composite_catchlight_materials(&mut mirrored, [Some(eye), None], params).unwrap();

        assert!(rgb_at(&plain, 14, 22)[0] > rgb_at(&plain, 14, 22)[2]);
        assert!(rgb_at(&mirrored, 18, 22)[0] > rgb_at(&mirrored, 18, 22)[2]);
        assert_ne!(plain.pixels, mirrored.pixels);
    }

    #[test]
    fn tint_multiplies_the_material_without_changing_coverage() {
        // A warm authored colour, so "white keeps it" and "cold makes blue win" are
        // both observable. A neutral grey would make the white case undecidable.
        let material = [220u8, 90, 60, 255].repeat(16);
        let left = ring(16.0, 24.0, 7.0, 7.0);
        let params = CatchlightParams {
            inset: 0.0,
            feather: 0.0,
        };
        let mut white = frame_image();
        let mut cold = frame_image();
        let white_receipt = composite_catchlight_materials(
            &mut white,
            [
                Some(CatchlightEye::new(&left, straight(&material, 4, 4), 1.0)),
                None,
            ],
            params,
        )
        .unwrap();
        let cold_receipt = composite_catchlight_materials(
            &mut cold,
            [
                Some(
                    CatchlightEye::new(&left, straight(&material, 4, 4), 1.0)
                        .with_tint([0, 0, 255]),
                ),
                None,
            ],
            params,
        )
        .unwrap();
        // Same geometry, same touched set: only the colour changes.
        assert_eq!(white_receipt.touched_pixels, cold_receipt.touched_pixels);
        assert_eq!(white_receipt.tints[0], CATCHLIGHT_TINT_WHITE);
        assert_eq!(cold_receipt.tints[0], [0, 0, 255]);
        let centre = rgb_at(&cold, 16, 24);
        assert!(
            centre[2] > centre[0],
            "cold tint must leave blue dominant: {centre:?}"
        );
        // The authored colour survives a white tint: red channel unchanged.
        let authored = rgb_at(&white, 16, 24);
        assert!(
            authored[0] > authored[2],
            "white tint keeps the authored colour"
        );
    }

    #[test]
    fn a_non_square_material_never_paints_outside_the_ring() {
        let mut image = frame_image();
        let before = image.pixels.clone();
        let left = ring(20.0, 20.0, 6.0, 6.0);
        let wide: Vec<u8> = [250u8, 40, 200, 255].repeat(16);
        let receipt = composite_catchlight_materials(
            &mut image,
            [
                Some(CatchlightEye::new(&left, straight(&wide, 8, 2), 1.0)),
                None,
            ],
            CatchlightParams::default(),
        )
        .unwrap();
        assert_eq!(receipt.material_sizes[0], [8, 2]);
        assert!(receipt.touched_pixels > 0);

        for y in 0..FRAME.1 {
            for x in 0..FRAME.0 {
                let inside_box =
                    (14.0..=26.0).contains(&(x as f32)) && (14.0..=26.0).contains(&(y as f32));
                if inside_box {
                    continue;
                }
                let offset = ((y * image.width + x) * 4) as usize;
                assert_eq!(
                    image.pixels[offset..offset + 4],
                    before[offset..offset + 4],
                    "pixel ({x},{y}) is outside the iris box"
                );
            }
        }
    }

    #[test]
    fn the_alpha_channel_is_never_written() {
        let mut image = frame_image();
        let alpha_before = alpha_of(&image);
        let left = ring(16.0, 24.0, 6.0, 6.0);
        let right = ring(44.0, 24.0, 6.0, 6.0);
        let material = [250u8, 250, 250, 255].repeat(16);
        composite_catchlight_materials(
            &mut image,
            [
                Some(CatchlightEye::new(&left, straight(&material, 4, 4), 1.0)),
                Some(CatchlightEye::new(&right, straight(&material, 4, 4), 1.0)),
            ],
            CatchlightParams::default(),
        )
        .unwrap();
        assert_eq!(alpha_of(&image), alpha_before);
    }

    #[test]
    fn a_degenerate_ring_is_refused() {
        let mut image = frame_image();
        let point = [[10.0f32, 10.0]; CATCHLIGHT_RING_POINTS];
        let material = quadrant_material();
        assert_eq!(
            catchlight_uv_frame(&point, 0.0),
            Err(Refusal::ShapeMismatch("catchlight iris ring"))
        );
        assert_eq!(
            composite_catchlight_materials(
                &mut image,
                [
                    Some(CatchlightEye::new(&point, straight(&material, 4, 4), 1.0)),
                    None
                ],
                CatchlightParams::default(),
            ),
            Err(Refusal::ShapeMismatch("catchlight iris ring"))
        );
    }

    #[test]
    fn out_of_range_parameters_and_amounts_are_refused() {
        let mut image = frame_image();
        let left = ring(16.0, 24.0, 6.0, 6.0);
        let material = quadrant_material();
        let eye = |amount: f32| Some(CatchlightEye::new(&left, straight(&material, 4, 4), amount));
        assert_eq!(
            composite_catchlight_materials(
                &mut image,
                [eye(1.0), None],
                CatchlightParams {
                    inset: 0.0,
                    feather: 0.9
                }
            ),
            Err(Refusal::ValueOutOfRange("catchlight params"))
        );
        assert_eq!(
            composite_catchlight_materials(
                &mut image,
                [eye(1.5), None],
                CatchlightParams::default()
            ),
            Err(Refusal::ValueOutOfRange("catchlight highlight amount"))
        );
        assert_eq!(
            composite_catchlight_materials(
                &mut image,
                [eye(f32::NAN), None],
                CatchlightParams::default()
            ),
            Err(Refusal::ValueOutOfRange("catchlight highlight amount"))
        );
    }

    #[test]
    fn a_malformed_material_is_refused() {
        let mut image = frame_image();
        let left = ring(16.0, 24.0, 6.0, 6.0);
        let short = [1u8, 2, 3];
        assert_eq!(
            composite_catchlight_materials(
                &mut image,
                [
                    Some(CatchlightEye::new(&left, straight(&short, 2, 2), 1.0)),
                    None
                ],
                CatchlightParams::default(),
            ),
            Err(Refusal::ShapeMismatch("catchlight material"))
        );
        assert_eq!(
            composite_catchlight_materials(
                &mut image,
                [
                    Some(CatchlightEye::new(
                        &left,
                        CatchlightMaterial::Png(b"not a png"),
                        1.0
                    )),
                    None
                ],
                CatchlightParams::default(),
            ),
            Err(Refusal::ShapeMismatch("catchlight material png"))
        );
    }

    /// The Lead's requirement, tested directly: a bad right eye must leave the whole
    /// frame byte-identical, including the left iris the good eye would have painted.
    #[test]
    fn a_bad_second_eye_leaves_the_whole_frame_untouched() {
        let mut image = frame_image();
        let before = image.pixels.clone();
        let left = ring(16.0, 24.0, 6.0, 6.0);
        let right = ring(44.0, 24.0, 6.0, 6.0);
        let material = quadrant_material();
        let short = [9u8, 9, 9];
        let result = composite_catchlight_materials(
            &mut image,
            [
                Some(CatchlightEye::new(&left, straight(&material, 4, 4), 1.0)),
                Some(CatchlightEye::new(&right, straight(&short, 2, 2), 1.0)),
            ],
            CatchlightParams::default(),
        );
        assert_eq!(result, Err(Refusal::ShapeMismatch("catchlight material")));
        assert_eq!(
            image.pixels, before,
            "no side may be written when one side fails"
        );

        // Same for a bad ring on the second eye.
        let point = [[0.0f32, 0.0]; CATCHLIGHT_RING_POINTS];
        let result = composite_catchlight_materials(
            &mut image,
            [
                Some(CatchlightEye::new(&left, straight(&material, 4, 4), 1.0)),
                Some(CatchlightEye::new(&point, straight(&material, 4, 4), 1.0)),
            ],
            CatchlightParams::default(),
        );
        assert_eq!(result, Err(Refusal::ShapeMismatch("catchlight iris ring")));
        assert_eq!(image.pixels, before);
    }

    #[test]
    fn the_inset_keeps_the_composite_off_the_ring_edge() {
        let mut image = frame_image();
        let before = image.pixels.clone();
        let left = ring(32.0, 24.0, 10.0, 10.0);
        let material = [10u8, 250, 10, 255].repeat(16);
        composite_catchlight_materials(
            &mut image,
            [
                Some(CatchlightEye::new(&left, straight(&material, 4, 4), 1.0)),
                None,
            ],
            CatchlightParams {
                inset: 0.5,
                feather: 0.0,
            },
        )
        .unwrap();
        let edge = (41u32, 24u32);
        let offset = ((edge.1 * image.width + edge.0) * 4) as usize;
        assert_eq!(image.pixels[offset..offset + 3], before[offset..offset + 3]);
    }
}
