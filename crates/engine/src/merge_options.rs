//! Options the interface sends with an HDR or panorama merge request.
//!
//! They live in the engine crate so `ImageEngine` can take them by reference.
//! The current engine consumes only the switches it can honour exactly; the
//! rest are logged and left for a later cut.

/// How the frames of an HDR merge are combined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrOptions {
    /// Align the frames before merging (hand-held brackets).
    pub align: bool,
    /// 0 off, 1 low, 2 medium, 3 high: how hard moving things are suppressed.
    pub deghost: u8,
    /// Set a starting tone on the merged result.
    pub auto_tone: bool,
}

impl Default for HdrOptions {
    fn default() -> Self {
        Self { align: true, deghost: 0, auto_tone: true }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Projection {
    #[default]
    Spherical,
    Cylindrical,
    Perspective,
}

impl Projection {
    pub const ALL: [Self; 3] = [Self::Spherical, Self::Cylindrical, Self::Perspective];
}

/// How the frames of a panorama are stitched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanoramaOptions {
    pub projection: Projection,
    /// 0 ..= 100: warp the edges to fill the rectangle.
    pub boundary_warp: f32,
    /// Crop to the largest rectangle with no empty corners.
    pub auto_crop: bool,
}

impl Default for PanoramaOptions {
    fn default() -> Self {
        Self { projection: Projection::Spherical, boundary_warp: 0.0, auto_crop: true }
    }
}
