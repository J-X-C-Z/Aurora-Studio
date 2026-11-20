//! Options the interface sends with an HDR or panorama merge request.
//!
//! They live in the engine crate so `ImageEngine` can take them by reference.
//! The current engine consumes only the switches it can honour exactly; the
//! rest are logged and left for a later cut.

/// How moving objects are handled in an HDR merge.
///
/// Capture One has no user-facing deghost switch: its HDR path builds
/// per-image masks and blends them multiband, so this enum is a local product
/// choice. `Off` skips the pass, `Auto` uses the median exposure as the
/// replacement frame, and `Manual` takes [`HdrOptions::deghost_reference`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeghostMode {
    Off,
    #[default]
    Auto,
    Manual,
}

impl DeghostMode {
    pub const ALL: [Self; 3] = [Self::Off, Self::Auto, Self::Manual];

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Auto => "Auto",
            Self::Manual => "Manual",
        }
    }
}

/// How the frames of an HDR merge are combined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrOptions {
    /// Align the frames before merging (hand-held brackets).
    pub align: bool,
    /// Off / Auto / Manual deghosting.
    pub deghost: DeghostMode,
    /// Frame index kept for `DeghostMode::Manual`; out of range falls back to
    /// the median-exposure frame.
    pub deghost_reference: Option<usize>,
    /// Set a starting tone on the merged result.
    pub auto_tone: bool,
}

impl Default for HdrOptions {
    fn default() -> Self {
        Self { align: true, deghost: DeghostMode::Auto, deghost_reference: None, auto_tone: true }
    }
}

/// Panorama projection. Capture One's `IC_WarpType` declares
/// `Default, Perspective, Affine, Cylindrical, Spherical, Panini`; the UI
/// exposes the four values in [`Projection::ALL`]. `Default` and `Affine`
/// fall back to Perspective in this engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Projection {
    #[default]
    Perspective,
    Cylindrical,
    Spherical,
    Panini,
}

impl Projection {
    pub const ALL: [Self; 4] = [
        Self::Perspective,
        Self::Cylindrical,
        Self::Spherical,
        Self::Panini,
    ];

    /// Map C1's `IC_WarpType` integer to the local enum.
    /// `0 Default`, `1 Perspective`, `2 Affine`, `3 Cylindrical`,
    /// `4 Spherical`, `5 Panini`; anything else keeps the no-op Perspective.
    pub fn from_warp_type(code: i32) -> Self {
        match code {
            3 => Self::Cylindrical,
            4 => Self::Spherical,
            5 => Self::Panini,
            _ => Self::Perspective,
        }
    }

    /// The C1 `IC_WarpType` value this projection writes back.
    pub fn warp_type(self) -> i32 {
        match self {
            Self::Perspective => 1,
            Self::Cylindrical => 3,
            Self::Spherical => 4,
            Self::Panini => 5,
        }
    }
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
        Self { projection: Projection::Perspective, boundary_warp: 0.0, auto_crop: true }
    }
}
