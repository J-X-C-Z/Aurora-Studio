//! Frame vocabulary: pixel formats, plane layout, coordinate spaces and regions.
//!
//! These types describe what a buffer *is*. They never allocate, never convert
//! and never inspect pixels. The CPU reference produces a [`FrameDescriptor`],
//! the CUDA adapter returns one with its device pitch, and the future Metal
//! adapter does the same; every consumer can validate the same descriptor with
//! the same arithmetic.
//!
//! All distances are bytes and all sizes use checked arithmetic. A descriptor
//! that overflows, has a zero dimension, a short stride or a region outside the
//! frame is rejected rather than silently clamped.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A single interleaved sample layout.
///
/// `Cfa*` formats are Bayer/CFA mosaics: one sample per photosite, not an RGB
/// image. A CFA plane must never be silently expanded to RGB by relabeling it;
/// demosaicing is an explicit operation with its own algorithm version.
/// `*F16`/`*F32` are floating-point samples; `*16` are unsigned 16-bit samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelFormat {
    /// 8-bit CFA mosaic, one sample per photosite.
    Cfa8,
    /// 16-bit CFA mosaic, one sample per photosite.
    Cfa16,
    /// 8-bit unsigned RGB, three interleaved samples per pixel.
    Rgb8,
    /// 8-bit unsigned RGBA, four interleaved samples per pixel.
    Rgba8,
    /// 16-bit unsigned RGB, three interleaved samples per pixel.
    Rgb16,
    /// 16-bit unsigned RGBA, four interleaved samples per pixel.
    Rgba16,
    /// IEEE-754 binary16 RGB, three interleaved samples per pixel.
    RgbF16,
    /// IEEE-754 binary16 RGBA, four interleaved samples per pixel.
    RgbaF16,
    /// IEEE-754 binary32 RGB, three interleaved samples per pixel.
    RgbF32,
    /// IEEE-754 binary32 RGBA, four interleaved samples per pixel.
    RgbaF32,
}

impl PixelFormat {
    /// Number of samples stored per pixel.
    ///
    /// A CFA mosaic reports one channel because a photosite carries one sample;
    /// reconstructing three color channels is demosaicing, not a channel count.
    pub const fn channels(self) -> u32 {
        match self {
            Self::Cfa8 | Self::Cfa16 => 1,
            Self::Rgb8 | Self::Rgb16 | Self::RgbF16 | Self::RgbF32 => 3,
            Self::Rgba8 | Self::Rgba16 | Self::RgbaF16 | Self::RgbaF32 => 4,
        }
    }

    /// Bits occupied by one sample.
    pub const fn bits_per_channel(self) -> u32 {
        match self {
            Self::Cfa8 | Self::Rgb8 | Self::Rgba8 => 8,
            Self::Cfa16 | Self::Rgb16 | Self::Rgba16 | Self::RgbF16 | Self::RgbaF16 => 16,
            Self::RgbF32 | Self::RgbaF32 => 32,
        }
    }

    /// Bytes occupied by one sample.
    pub const fn bytes_per_sample(self) -> u32 {
        self.bits_per_channel() / 8
    }

    /// Bytes occupied by one fully packed pixel.
    pub const fn bytes_per_pixel(self) -> u32 {
        self.channels() * self.bytes_per_sample()
    }

    /// Whether this is a Bayer/CFA mosaic (one sample per photosite).
    pub const fn is_cfa(self) -> bool {
        matches!(self, Self::Cfa8 | Self::Cfa16)
    }

    /// Whether the format carries an alpha channel.
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::Rgba8 | Self::Rgba16 | Self::RgbaF16 | Self::RgbaF32)
    }

    /// Bytes a fully packed row of `width` pixels occupies.
    pub fn row_bytes_for_width(self, width: u32) -> Result<u64, ContractError> {
        u64::from(width)
            .checked_mul(u64::from(self.bytes_per_pixel()))
            .ok_or(ContractError::Overflow { what: "row_bytes" })
    }

    /// Reject a CFA mosaic for a call that requires color channels.
    ///
    /// This is the explicit guard that keeps a Bayer plane from being treated
    /// as RGB: the caller must demosaic first and label the result as RGB.
    pub fn require_color(self) -> Result<Self, ContractError> {
        if self.is_cfa() {
            Err(ContractError::UnsupportedFormat { format: self })
        } else {
            Ok(self)
        }
    }
}

/// How an alpha channel, if present, is interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlphaMode {
    /// The format has no alpha channel.
    None,
    /// An alpha channel exists but is ignored; samples are treated as 1.0.
    Opaque,
    /// Alpha is straight (not premultiplied) coverage.
    Straight,
    /// Color samples are already multiplied by alpha.
    Premultiplied,
}

/// Byte offset, row pitch and base alignment of one interleaved plane.
///
/// All fields are in bytes. `row_stride` is the distance between the start of
/// consecutive rows, including any padding; it must be at least the packed row
/// size. `alignment` is the required byte alignment of `offset` and
/// `row_stride`; use `1` when no alignment is required.
///
/// Multi-plane (planar) buffers use [`PlanarLayout`] instead; the two are
/// mutually exclusive on a [`FrameDescriptor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PlaneLayout {
    /// Byte offset of sample (0, 0) from the start of the buffer.
    pub offset: u64,
    /// Byte distance between the starts of consecutive rows.
    pub row_stride: u64,
    /// Required byte alignment of `offset` and `row_stride`.
    pub alignment: u64,
}

impl PlaneLayout {
    /// Construct a layout; all arguments are bytes.
    pub const fn new(offset: u64, row_stride: u64, alignment: u64) -> Self {
        Self { offset, row_stride, alignment }
    }

    /// Check that this layout can describe `width × height` pixels of `format`.
    ///
    /// Rejects zero dimensions, a stride shorter than one packed row, alignment
    /// that is not a non-zero power of two, and an offset or stride that is not
    /// a multiple of that alignment. Arithmetic is checked.
    pub fn validate(
        &self,
        format: PixelFormat,
        width: u32,
        height: u32,
    ) -> Result<(), ContractError> {
        if width == 0 {
            return Err(ContractError::ZeroDimension { what: "width" });
        }
        if height == 0 {
            return Err(ContractError::ZeroDimension { what: "height" });
        }

        let minimum = format.row_bytes_for_width(width)?;
        if self.row_stride < minimum {
            return Err(ContractError::InvalidStride {
                row_stride: self.row_stride,
                minimum,
            });
        }

        if self.alignment == 0 || !self.alignment.is_power_of_two() {
            return Err(ContractError::InvalidAlignment {
                alignment: self.alignment,
            });
        }
        if self.offset % self.alignment != 0 || self.row_stride % self.alignment != 0 {
            return Err(ContractError::InvalidAlignment {
                alignment: self.alignment,
            });
        }

        Ok(())
    }
}

/// One plane of a planar (multi-plane) buffer.
///
/// A plane stores exactly one sample per pixel of the frame's [`PixelFormat`],
/// so a plane of `RgbF32` holds one `f32` per pixel and a plane of `Cfa16`
/// holds one `u16` per photosite. The plane is full-width: its packed row is
/// `width × bytes_per_sample`. `height` is the number of rows in this plane;
/// it may be smaller than the frame height for a vertically subsampled plane,
/// but per-plane horizontal subsampling is not represented.
///
/// All fields are in bytes and are validated with checked arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PlaneSpec {
    /// Byte offset of sample (0, 0) from the start of the buffer.
    pub offset: u64,
    /// Byte distance between the starts of consecutive rows, including padding.
    pub row_stride: u64,
    /// Number of rows in this plane.
    pub height: u32,
}

impl PlaneSpec {
    /// Construct a plane; all distances are bytes.
    pub const fn new(offset: u64, row_stride: u64, height: u32) -> Self {
        Self { offset, row_stride, height }
    }

    /// End byte of this plane, exclusive, with checked arithmetic.
    ///
    /// The end includes the trailing pad of the last row (`row_stride × height`),
    /// matching [`FrameDescriptor::byte_len`] for an interleaved plane. Two
    /// planes may therefore touch (`end == next.offset`) but must not straddle.
    pub fn end(&self) -> Result<u64, ContractError> {
        self.row_stride
            .checked_mul(u64::from(self.height))
            .and_then(|bytes| self.offset.checked_add(bytes))
            .ok_or(ContractError::Overflow { what: "plane_end" })
    }
}

/// Checked description of a planar (multi-plane) buffer.
///
/// A planar frame stores one full-width sample plane per color channel: the
/// number of planes must equal [`PixelFormat::channels`], so a CFA mosaic has
/// one plane and an RGB frame has three. Planes may appear in any order in the
/// buffer and may be padded; validation rejects overlapping byte intervals and
/// a stride that cannot hold one packed row.
///
/// This is the vocabulary the GPU runtime's `F32Planes` residency and a future
/// Metal plane adapter would use. It is documented and validated here, but no
/// adapter emits it yet.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PlanarLayout {
    /// Planes in buffer order; must be non-empty.
    pub planes: Vec<PlaneSpec>,
    /// Required byte alignment of every plane's offset and row stride; a
    /// non-zero power of two. Use `1` when no alignment is required.
    pub alignment: u64,
}

impl PlanarLayout {
    /// Construct a planar layout; `alignment` applies to every plane.
    pub fn new(planes: Vec<PlaneSpec>, alignment: u64) -> Self {
        Self { planes, alignment }
    }

    /// Validate the planes against `format`, `width` and frame `height`.
    ///
    /// Rejects: zero frame dimensions, no planes, a plane count that does not
    /// match the format's channel count, a zero-height plane, a stride shorter
    /// than one packed row of one sample per pixel, alignment that is not a
    /// non-zero power of two dividing every offset and stride, and overlapping
    /// plane byte intervals. All arithmetic is checked.
    pub fn validate(
        &self,
        format: PixelFormat,
        width: u32,
        height: u32,
    ) -> Result<(), ContractError> {
        if width == 0 {
            return Err(ContractError::ZeroDimension { what: "width" });
        }
        if height == 0 {
            return Err(ContractError::ZeroDimension { what: "height" });
        }
        let channels = format.channels();
        if self.planes.is_empty() {
            return Err(ContractError::NoPlanes);
        }
        if self.planes.len() != channels as usize {
            return Err(ContractError::PlaneCountMismatch {
                planes: self.planes.len(),
                channels,
            });
        }

        if self.alignment == 0 || !self.alignment.is_power_of_two() {
            return Err(ContractError::InvalidAlignment {
                alignment: self.alignment,
            });
        }

        // One sample per pixel per plane: a packed row is width × sample size.
        let minimum = u64::from(width)
            .checked_mul(u64::from(format.bytes_per_sample()))
            .ok_or(ContractError::Overflow { what: "row_bytes" })?;

        let mut intervals = Vec::with_capacity(self.planes.len());
        for (index, plane) in self.planes.iter().enumerate() {
            if plane.height == 0 {
                return Err(ContractError::ZeroDimension { what: "plane.height" });
            }
            if plane.row_stride < minimum {
                return Err(ContractError::InvalidStride {
                    row_stride: plane.row_stride,
                    minimum,
                });
            }
            if plane.offset % self.alignment != 0 || plane.row_stride % self.alignment != 0 {
                return Err(ContractError::InvalidAlignment {
                    alignment: self.alignment,
                });
            }
            let end = plane.end()?;
            intervals.push((plane.offset, end, index));
        }

        // Overlap is checked on the sorted byte intervals, not on plane order:
        // a plane may be placed before an earlier index in the buffer.
        intervals.sort_unstable_by_key(|(offset, _, _)| *offset);
        for pair in intervals.windows(2) {
            let (_, first_end, first_index) = pair[0];
            let (second_offset, _, second_index) = pair[1];
            if first_end > second_offset {
                return Err(ContractError::PlaneOverlap {
                    first: first_index,
                    second: second_index,
                });
            }
        }

        Ok(())
    }

    /// Allocated bytes spanned by all planes, with checked arithmetic.
    ///
    /// This is the largest plane end, so a buffer whose planes are not ordered
    /// by offset is still measured correctly. Fails with
    /// [`ContractError::NoPlanes`] when empty and [`ContractError::Overflow`]
    /// when a plane does not fit in `u64`.
    pub fn byte_len(&self) -> Result<u64, ContractError> {
        if self.planes.is_empty() {
            return Err(ContractError::NoPlanes);
        }
        let mut total = 0u64;
        for plane in &self.planes {
            total = total.max(plane.end()?);
        }
        Ok(total)
    }
}

/// Color interpretation of the samples in a frame.
///
/// `CameraNative` and `Icc` identify a response by name only; they do not embed
/// profile bytes. A consumer that needs the profile must load it through its
/// own loader.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorSpace {
    /// Linear-light sRGB primaries.
    LinearSrgb,
    /// sRGB primaries with the sRGB transfer function.
    Srgb,
    /// Display P3 primaries with the sRGB transfer function.
    DisplayP3,
    /// The camera's native response, named by `profile` (not an ICC profile).
    CameraNative { profile: String },
    /// An ICC-described response, identified by `name` (not embedded bytes).
    Icc { name: String },
}

/// The coordinate frame a descriptor's dimensions and region refer to.
///
/// These variants name a space; converting between them belongs to a later
/// pipeline node with its own tests, not to this contract module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateSpace {
    /// Raw source pixel grid, before orientation metadata is applied.
    Source,
    /// Source grid with EXIF orientation already applied.
    Oriented,
    /// A crop of the oriented image; `(0, 0)` is the crop's own origin.
    Crop,
    /// The preview or export viewport currently being displayed.
    Viewport,
    /// A model's region of interest inside the oriented image.
    ModelRoi {
        /// Horizontal offset in oriented pixels.
        x: i32,
        /// Vertical offset in oriented pixels.
        y: i32,
        /// Region width in pixels.
        width: u32,
        /// Region height in pixels.
        height: u32,
    },
}

/// A rectangular region of a frame, in that frame's coordinate space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ValidRegion {
    /// Left edge, in pixels.
    pub x: u32,
    /// Top edge, in pixels.
    pub y: u32,
    /// Region width, in pixels.
    pub width: u32,
    /// Region height, in pixels.
    pub height: u32,
}

impl ValidRegion {
    /// Whether `(x, y)` lies inside the region. Arithmetic cannot overflow.
    pub fn contains(&self, x: u32, y: u32) -> bool {
        let x = u64::from(x);
        let y = u64::from(y);
        let x0 = u64::from(self.x);
        let y0 = u64::from(self.y);
        x >= x0
            && y >= y0
            && x - x0 < u64::from(self.width)
            && y - y0 < u64::from(self.height)
    }

    /// Whether the region covers no pixels.
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// A validated description of one image buffer.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FrameDescriptor {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Sample layout.
    pub format: PixelFormat,
    /// Byte offset, pitch and alignment of the single interleaved plane.
    ///
    /// This field is required and describes the interleaved buffer. When
    /// `planes` is `Some`, the frame is planar and this field is retained for
    /// compatibility only: it is not validated against the format and is
    /// ignored by the byte-size accessors. Use [`PlanarLayout`] then.
    pub layout: PlaneLayout,
    /// Multi-plane layout, when the frame is planar.
    ///
    /// `None` is the interleaved case described by `layout`. `Some` means the
    /// buffer holds one sample plane per channel, and `layout` is ignored. The
    /// field is `Option` (rather than an enum) so old serialized descriptors
    /// without it still deserialize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planes: Option<PlanarLayout>,
    /// Color interpretation of the samples.
    pub color_space: ColorSpace,
    /// Coordinate space named by `width`, `height` and `valid_region`.
    pub coordinates: CoordinateSpace,
    /// Alpha interpretation.
    pub alpha_mode: AlphaMode,
    /// Pixels inside the frame that are actually meaningful.
    pub valid_region: ValidRegion,
}

impl FrameDescriptor {
    /// Validate dimensions, layout and region.
    ///
    /// Rejects a zero width or height, a layout that cannot hold one packed
    /// row, invalid alignment, and a `valid_region` that extends beyond the
    /// frame. An empty region is allowed; it covers no pixels but is in bounds.
    /// A planar frame is validated through [`PlanarLayout`] instead of
    /// `layout`.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.width == 0 {
            return Err(ContractError::ZeroDimension { what: "width" });
        }
        if self.height == 0 {
            return Err(ContractError::ZeroDimension { what: "height" });
        }

        match &self.planes {
            Some(planes) => planes.validate(self.format, self.width, self.height)?,
            None => self.layout.validate(self.format, self.width, self.height)?,
        }

        if u64::from(self.valid_region.x) + u64::from(self.valid_region.width)
            > u64::from(self.width)
            || u64::from(self.valid_region.y) + u64::from(self.valid_region.height)
                > u64::from(self.height)
        {
            return Err(ContractError::RegionOutsideFrame {
                region: self.valid_region,
                width: self.width,
                height: self.height,
            });
        }

        Ok(())
    }

    /// Whether the frame uses a planar (multi-plane) buffer.
    pub const fn is_planar(&self) -> bool {
        self.planes.is_some()
    }

    /// Allocated bytes between the starts of consecutive rows (`row_stride`).
    ///
    /// This is the padded pitch, not the packed `width × bytes_per_pixel`, and
    /// it is only defined for an interleaved frame. A planar frame has one
    /// pitch per plane; use [`PlanarLayout::byte_len`] and inspect the plane
    /// specs instead.
    pub fn row_bytes(&self) -> Result<usize, ContractError> {
        if self.planes.is_some() {
            return Err(ContractError::LayoutIsPlanar { what: "row_bytes" });
        }
        self.validate()?;
        usize::try_from(self.layout.row_stride)
            .map_err(|_| ContractError::Overflow { what: "row_bytes" })
    }

    /// Bytes from the plane offset to the end of the last row.
    ///
    /// Interleaved frames only; a planar frame returns
    /// [`ContractError::LayoutIsPlanar`].
    pub fn plane_bytes(&self) -> Result<usize, ContractError> {
        if self.planes.is_some() {
            return Err(ContractError::LayoutIsPlanar { what: "plane_bytes" });
        }
        self.validate()?;
        let bytes = self.checked_plane_bytes()?;
        usize::try_from(bytes).map_err(|_| ContractError::Overflow { what: "plane_bytes" })
    }

    /// Bytes spanned by the frame, including its plane offset.
    ///
    /// For an interleaved frame this is `offset + row_stride × height`; for a
    /// planar frame it is the largest plane end. Both use checked arithmetic.
    pub fn byte_len(&self) -> Result<usize, ContractError> {
        self.validate()?;
        let total = match &self.planes {
            Some(planes) => planes.byte_len()?,
            None => self
                .layout
                .offset
                .checked_add(self.checked_plane_bytes()?)
                .ok_or(ContractError::Overflow { what: "byte_len" })?,
        };
        usize::try_from(total).map_err(|_| ContractError::Overflow { what: "byte_len" })
    }

    fn checked_plane_bytes(&self) -> Result<u64, ContractError> {
        self.layout
            .row_stride
            .checked_mul(u64::from(self.height))
            .ok_or(ContractError::Overflow { what: "plane_bytes" })
    }
}

/// Reasons a frame descriptor cannot describe its buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContractError {
    /// A required dimension is zero; `what` names the offending field.
    ZeroDimension { what: &'static str },
    /// Checked arithmetic overflowed while computing `what`.
    Overflow { what: &'static str },
    /// The row stride cannot hold one packed row.
    InvalidStride { row_stride: u64, minimum: u64 },
    /// The alignment is not a non-zero power of two, or does not divide the
    /// offset and stride.
    InvalidAlignment { alignment: u64 },
    /// The valid region extends beyond the frame.
    RegionOutsideFrame {
        region: ValidRegion,
        width: u32,
        height: u32,
    },
    /// The format cannot be used in this context (for example, a CFA mosaic
    /// where a color plane is required).
    UnsupportedFormat { format: PixelFormat },
    /// A planar layout has no planes.
    NoPlanes,
    /// A planar layout's plane count does not match the format's channel count.
    PlaneCountMismatch {
        /// Planes described by the layout.
        planes: usize,
        /// Channels the pixel format declares.
        channels: u32,
    },
    /// Two plane byte intervals overlap.
    PlaneOverlap {
        /// Index of the plane that ends later.
        first: usize,
        /// Index of the plane that starts before `first` ends.
        second: usize,
    },
    /// An accessor that only applies to an interleaved frame was called on a
    /// planar frame; `what` names the accessor.
    LayoutIsPlanar { what: &'static str },
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension { what } => write!(f, "frame {what} must be non-zero"),
            Self::Overflow { what } => write!(f, "byte-size overflow while computing {what}"),
            Self::InvalidStride { row_stride, minimum } => write!(
                f,
                "row stride {row_stride} bytes is shorter than the packed row {minimum} bytes"
            ),
            Self::InvalidAlignment { alignment } => write!(
                f,
                "alignment {alignment} is not a power of two dividing the plane offset and stride"
            ),
            Self::RegionOutsideFrame {
                region,
                width,
                height,
            } => write!(
                f,
                "valid region {}x{} at ({}, {}) lies outside the {width}x{height} frame",
                region.width, region.height, region.x, region.y
            ),
            Self::UnsupportedFormat { format } => {
                write!(f, "pixel format {format:?} is not supported in this context")
            }
            Self::NoPlanes => f.write_str("a planar layout must describe at least one plane"),
            Self::PlaneCountMismatch { planes, channels } => write!(
                f,
                "planar layout has {planes} planes but the pixel format declares {channels} channels"
            ),
            Self::PlaneOverlap { first, second } => write!(
                f,
                "plane {second} starts before plane {first}'s byte interval ends"
            ),
            Self::LayoutIsPlanar { what } => {
                write!(f, "{what} is only defined for an interleaved frame, not a planar one")
            }
        }
    }
}

impl std::error::Error for ContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(format: PixelFormat, width: u32, height: u32) -> FrameDescriptor {
        FrameDescriptor {
            width,
            height,
            format,
            layout: PlaneLayout {
                offset: 0,
                row_stride: format
                    .row_bytes_for_width(width)
                    .expect("test width fits a row"),
                alignment: 1,
            },
            planes: None,
            color_space: ColorSpace::LinearSrgb,
            coordinates: CoordinateSpace::Source,
            alpha_mode: AlphaMode::None,
            valid_region: ValidRegion { x: 0, y: 0, width, height },
        }
    }

    #[test]
    fn zero_dimension_is_rejected() {
        let mut frame = descriptor(PixelFormat::Rgb8, 4, 4);
        frame.width = 0;
        assert_eq!(
            frame.byte_len(),
            Err(ContractError::ZeroDimension { what: "width" })
        );

        let mut frame = descriptor(PixelFormat::Rgb8, 4, 4);
        frame.height = 0;
        assert_eq!(
            frame.row_bytes(),
            Err(ContractError::ZeroDimension { what: "height" })
        );

        assert_eq!(
            PlaneLayout::new(0, 30, 1).validate(PixelFormat::Rgb8, 0, 1),
            Err(ContractError::ZeroDimension { what: "width" })
        );
    }

    #[test]
    fn plane_byte_overflow_is_rejected() {
        let mut frame = descriptor(PixelFormat::Rgb8, 1, u32::MAX);
        frame.layout.row_stride = u64::MAX;
        assert!(matches!(
            frame.plane_bytes(),
            Err(ContractError::Overflow { .. })
        ));
        assert!(matches!(
            frame.byte_len(),
            Err(ContractError::Overflow { .. })
        ));

        let mut frame = descriptor(PixelFormat::Rgb8, 1, 1);
        frame.layout.row_stride = 1 << 63;
        frame.layout.offset = 1 << 63;
        assert!(matches!(
            frame.byte_len(),
            Err(ContractError::Overflow { .. })
        ));
    }

    #[test]
    fn stride_smaller_than_packed_row_is_rejected() {
        let layout = PlaneLayout::new(0, 29, 1);
        assert_eq!(
            layout.validate(PixelFormat::Rgb8, 10, 2),
            Err(ContractError::InvalidStride {
                row_stride: 29,
                minimum: 30
            })
        );

        let layout = PlaneLayout::new(0, 30, 1);
        assert_eq!(layout.validate(PixelFormat::Rgb8, 10, 2), Ok(()));
    }

    #[test]
    fn alignment_must_be_a_dividing_power_of_two() {
        let layout = PlaneLayout::new(0, 32, 3);
        assert_eq!(
            layout.validate(PixelFormat::Rgb8, 10, 1),
            Err(ContractError::InvalidAlignment { alignment: 3 })
        );

        let layout = PlaneLayout::new(0, 32, 0);
        assert_eq!(
            layout.validate(PixelFormat::Rgb8, 10, 1),
            Err(ContractError::InvalidAlignment { alignment: 0 })
        );

        let layout = PlaneLayout::new(2, 32, 8);
        assert_eq!(
            layout.validate(PixelFormat::Rgb8, 10, 1),
            Err(ContractError::InvalidAlignment { alignment: 8 })
        );

        let layout = PlaneLayout::new(16, 36, 8);
        assert_eq!(
            layout.validate(PixelFormat::Rgb8, 10, 1),
            Err(ContractError::InvalidAlignment { alignment: 8 })
        );

        let layout = PlaneLayout::new(16, 32, 8);
        assert_eq!(layout.validate(PixelFormat::Rgb8, 10, 1), Ok(()));
    }

    #[test]
    fn cfa_is_one_channel_and_never_color() {
        assert!(PixelFormat::Cfa8.is_cfa());
        assert!(PixelFormat::Cfa16.is_cfa());
        assert_eq!(PixelFormat::Cfa8.channels(), 1);
        assert_eq!(PixelFormat::Cfa16.channels(), 1);
        assert_eq!(PixelFormat::Cfa16.bytes_per_sample(), 2);
        assert_eq!(PixelFormat::Cfa16.bytes_per_pixel(), 2);

        assert_eq!(PixelFormat::Rgb8.channels(), 3);
        assert_eq!(PixelFormat::Rgba8.channels(), 4);
        assert_eq!(PixelFormat::RgbF32.bytes_per_pixel(), 12);
        assert_eq!(PixelFormat::RgbaF16.bytes_per_pixel(), 8);
        assert!(!PixelFormat::Rgb8.is_cfa());
        assert!(PixelFormat::Rgba8.has_alpha());
        assert!(!PixelFormat::Rgb8.has_alpha());

        assert_eq!(
            PixelFormat::Cfa8.require_color(),
            Err(ContractError::UnsupportedFormat {
                format: PixelFormat::Cfa8
            })
        );
        assert_eq!(
            PixelFormat::Rgb8.require_color(),
            Ok(PixelFormat::Rgb8)
        );

        // A CFA row uses one sample per pixel, so 10 photosites need 10 bytes,
        // not the 30 bytes an RGB frame of the same width would need.
        let layout = PlaneLayout::new(0, 10, 1);
        assert_eq!(layout.validate(PixelFormat::Cfa8, 10, 1), Ok(()));
    }

    #[test]
    fn valid_region_must_stay_inside_the_frame() {
        let mut frame = descriptor(PixelFormat::Rgb8, 8, 8);
        frame.valid_region = ValidRegion { x: 8, y: 0, width: 1, height: 1 };
        assert_eq!(
            frame.validate(),
            Err(ContractError::RegionOutsideFrame {
                region: ValidRegion { x: 8, y: 0, width: 1, height: 1 },
                width: 8,
                height: 8
            })
        );

        frame.valid_region = ValidRegion { x: 7, y: 7, width: 1, height: 1 };
        assert_eq!(frame.validate(), Ok(()));

        frame.valid_region = ValidRegion { x: 0, y: 0, width: 0, height: 4 };
        assert_eq!(frame.validate(), Ok(()));

        let region = ValidRegion { x: 2, y: 3, width: 4, height: 5 };
        assert!(region.contains(2, 3));
        assert!(region.contains(5, 7));
        assert!(!region.contains(6, 7));
        assert!(!region.contains(1, 3));
        assert!(!region.contains(2, 8));
        assert!(!region.is_empty());
        assert!(ValidRegion { x: 0, y: 0, width: 0, height: 5 }.is_empty());
    }

    #[test]
    fn camera_native_descriptor_round_trips_through_json() {
        let frame = FrameDescriptor {
            width: 6048,
            height: 4024,
            format: PixelFormat::Cfa16,
            layout: PlaneLayout::new(0, 12288, 256),
            planes: None,
            color_space: ColorSpace::CameraNative {
                profile: "Nikon NEF linear".to_owned(),
            },
            coordinates: CoordinateSpace::Oriented,
            alpha_mode: AlphaMode::None,
            valid_region: ValidRegion {
                x: 0,
                y: 0,
                width: 6048,
                height: 4024,
            },
        };

        assert_eq!(frame.validate(), Ok(()));
        assert_eq!(frame.row_bytes(), Ok(12288));
        assert_eq!(frame.plane_bytes(), Ok(12288 * 4024));

        let json = serde_json::to_string(&frame).expect("serialize descriptor");
        assert!(json.contains("camera_native"));
        // An interleaved frame omits `planes` entirely.
        assert!(!json.contains("planes"));
        let back: FrameDescriptor = serde_json::from_str(&json).expect("deserialize descriptor");
        assert_eq!(back, frame);
        assert!(!back.is_planar());
    }

    fn planar_descriptor(
        format: PixelFormat,
        width: u32,
        height: u32,
        planes: Vec<PlaneSpec>,
        alignment: u64,
    ) -> FrameDescriptor {
        FrameDescriptor {
            width,
            height,
            format,
            // Ignored when `planes` is `Some`; kept structurally valid-looking.
            layout: PlaneLayout::new(0, 0, alignment),
            planes: Some(PlanarLayout::new(planes, alignment)),
            color_space: ColorSpace::LinearSrgb,
            coordinates: CoordinateSpace::Source,
            alpha_mode: AlphaMode::None,
            valid_region: ValidRegion { x: 0, y: 0, width, height },
        }
    }

    #[test]
    fn planar_single_plane_cfa_is_valid() {
        // One photosite sample per pixel: a 10×4 Cfa16 plane at 2-byte samples
        // packs to a 20-byte row.
        let layout = PlanarLayout::new(vec![PlaneSpec::new(0, 20, 4)], 2);
        assert_eq!(layout.validate(PixelFormat::Cfa16, 10, 4), Ok(()));
        assert_eq!(layout.byte_len(), Ok(80));

        let frame = planar_descriptor(
            PixelFormat::Cfa16,
            10,
            4,
            vec![PlaneSpec::new(0, 20, 4)],
            2,
        );
        assert!(frame.is_planar());
        assert_eq!(frame.validate(), Ok(()));
        assert_eq!(frame.byte_len(), Ok(80));
        // Interleaved-only accessors refuse a planar frame instead of guessing.
        assert_eq!(
            frame.row_bytes(),
            Err(ContractError::LayoutIsPlanar { what: "row_bytes" })
        );
        assert_eq!(
            frame.plane_bytes(),
            Err(ContractError::LayoutIsPlanar { what: "plane_bytes" })
        );
    }

    #[test]
    fn planar_multi_plane_rgb_is_valid() {
        // Three F32 planes, one f32 per pixel: 8×4 => 32-byte row, 128-byte
        // plane, packed back to back and 4-byte aligned.
        let planes = vec![
            PlaneSpec::new(0, 32, 4),
            PlaneSpec::new(128, 32, 4),
            PlaneSpec::new(256, 32, 4),
        ];
        let layout = PlanarLayout::new(planes.clone(), 4);
        assert_eq!(layout.validate(PixelFormat::RgbF32, 8, 4), Ok(()));
        assert_eq!(layout.byte_len(), Ok(384));

        // Planes may appear in buffer order different from channel order, and
        // touching intervals (`end == next offset`) do not overlap.
        let reordered = vec![
            PlaneSpec::new(256, 32, 4),
            PlaneSpec::new(0, 32, 4),
            PlaneSpec::new(128, 32, 4),
        ];
        assert_eq!(
            PlanarLayout::new(reordered, 4).validate(PixelFormat::RgbF32, 8, 4),
            Ok(())
        );

        let frame = planar_descriptor(PixelFormat::RgbF32, 8, 4, planes, 4);
        assert_eq!(frame.validate(), Ok(()));
        assert_eq!(frame.byte_len(), Ok(384));
    }

    #[test]
    fn planar_overlapping_intervals_are_rejected() {
        let planes = vec![
            PlaneSpec::new(0, 32, 4),
            PlaneSpec::new(64, 32, 4),
            PlaneSpec::new(256, 32, 4),
        ];
        assert_eq!(
            PlanarLayout::new(planes, 4).validate(PixelFormat::RgbF32, 8, 4),
            Err(ContractError::PlaneOverlap {
                first: 0,
                second: 1
            })
        );

        // A single-byte straddle is still an overlap.
        let planes = vec![
            PlaneSpec::new(0, 33, 4), // end = 132
            PlaneSpec::new(131, 32, 4),
            PlaneSpec::new(256, 32, 4),
        ];
        assert_eq!(
            PlanarLayout::new(planes, 1).validate(PixelFormat::RgbF32, 8, 4),
            Err(ContractError::PlaneOverlap {
                first: 0,
                second: 1
            })
        );
    }

    #[test]
    fn planar_overflow_is_rejected() {
        let plane = PlaneSpec::new(u64::MAX, 8, 2);
        assert_eq!(
            plane.end(),
            Err(ContractError::Overflow { what: "plane_end" })
        );
        assert_eq!(
            PlanarLayout::new(vec![plane], 1).byte_len(),
            Err(ContractError::Overflow { what: "plane_end" })
        );
        assert_eq!(
            PlanarLayout::new(vec![plane], 1).validate(PixelFormat::Cfa8, 1, 2),
            Err(ContractError::Overflow { what: "plane_end" })
        );

        // A stride × height product that overflows before the offset is added.
        let plane = PlaneSpec::new(0, 1 << 63, 3);
        assert_eq!(
            PlanarLayout::new(vec![plane], 1).byte_len(),
            Err(ContractError::Overflow { what: "plane_end" })
        );
    }

    #[test]
    fn planar_alignment_must_be_a_dividing_power_of_two() {
        let planes = vec![
            PlaneSpec::new(0, 32, 4),
            PlaneSpec::new(128, 32, 4),
            PlaneSpec::new(256, 32, 4),
        ];
        assert_eq!(
            PlanarLayout::new(planes.clone(), 3).validate(PixelFormat::RgbF32, 8, 4),
            Err(ContractError::InvalidAlignment { alignment: 3 })
        );
        assert_eq!(
            PlanarLayout::new(planes.clone(), 0).validate(PixelFormat::RgbF32, 8, 4),
            Err(ContractError::InvalidAlignment { alignment: 0 })
        );
        // Offset 2 is not a multiple of 8.
        let misaligned = vec![
            PlaneSpec::new(2, 32, 4),
            PlaneSpec::new(128, 32, 4),
            PlaneSpec::new(256, 32, 4),
        ];
        assert_eq!(
            PlanarLayout::new(misaligned, 8).validate(PixelFormat::RgbF32, 8, 4),
            Err(ContractError::InvalidAlignment { alignment: 8 })
        );
        // Stride 36 is not a multiple of 8.
        let bad_stride = vec![
            PlaneSpec::new(0, 36, 4),
            PlaneSpec::new(128, 32, 4),
            PlaneSpec::new(256, 32, 4),
        ];
        assert_eq!(
            PlanarLayout::new(bad_stride, 8).validate(PixelFormat::RgbF32, 8, 4),
            Err(ContractError::InvalidAlignment { alignment: 8 })
        );
        // 8-byte alignment with offset 16 and stride 32 is fine.
        let aligned = vec![
            PlaneSpec::new(16, 32, 4),
            PlaneSpec::new(144, 32, 4),
            PlaneSpec::new(272, 32, 4),
        ];
        assert_eq!(
            PlanarLayout::new(aligned, 8).validate(PixelFormat::RgbF32, 8, 4),
            Ok(())
        );
    }

    #[test]
    fn planar_plane_count_matches_cfa_and_rgb() {
        let cfa_plane = PlaneSpec::new(0, 20, 4);
        assert_eq!(
            PlanarLayout::new(vec![cfa_plane], 2).validate(PixelFormat::Cfa16, 10, 4),
            Ok(())
        );
        assert_eq!(
            PlanarLayout::new(
                vec![cfa_plane, PlaneSpec::new(80, 20, 4), PlaneSpec::new(160, 20, 4)],
                2
            )
            .validate(PixelFormat::Cfa16, 10, 4),
            Err(ContractError::PlaneCountMismatch {
                planes: 3,
                channels: 1
            })
        );

        let rgb_plane = PlaneSpec::new(0, 32, 4);
        assert_eq!(
            PlanarLayout::new(vec![rgb_plane], 4).validate(PixelFormat::RgbF32, 8, 4),
            Err(ContractError::PlaneCountMismatch {
                planes: 1,
                channels: 3
            })
        );

        let rgba_planes = (0..4)
            .map(|index| PlaneSpec::new(index * 128, 32, 4))
            .collect();
        assert_eq!(
            PlanarLayout::new(rgba_planes, 4).validate(PixelFormat::RgbaF32, 8, 4),
            Ok(())
        );

        assert_eq!(
            PlanarLayout::new(Vec::new(), 1).validate(PixelFormat::RgbF32, 8, 4),
            Err(ContractError::NoPlanes)
        );
    }

    #[test]
    fn planar_descriptor_round_trips_through_json() {
        let frame = planar_descriptor(
            PixelFormat::RgbF32,
            8,
            4,
            vec![
                PlaneSpec::new(0, 32, 4),
                PlaneSpec::new(128, 32, 4),
                PlaneSpec::new(256, 32, 4),
            ],
            4,
        );
        assert_eq!(frame.validate(), Ok(()));

        let json = serde_json::to_string(&frame).expect("serialize planar descriptor");
        assert!(json.contains("planes"));
        let back: FrameDescriptor =
            serde_json::from_str(&json).expect("deserialize planar descriptor");
        assert_eq!(back, frame);
        assert!(back.is_planar());
    }

    #[test]
    fn planar_zero_height_plane_is_rejected() {
        let plane = PlaneSpec::new(0, 32, 0);
        assert_eq!(
            PlanarLayout::new(vec![plane, PlaneSpec::new(128, 32, 4), PlaneSpec::new(256, 32, 4)], 4)
                .validate(PixelFormat::RgbF32, 8, 4),
            Err(ContractError::ZeroDimension { what: "plane.height" })
        );
    }
}
