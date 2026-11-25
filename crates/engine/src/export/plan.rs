//! Export planning and pre-flight validation.
//!
//! [`ExportPlan::from_recipe`] validates format support, bit depth, scaling,
//! destination directory and geometry before any source decode or pixel buffer
//! is allocated. Invalid parameters always return a typed [`ExportPlanError`]
//! instead of silently downgrading bit depth or clamping zero dimensions.

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::adjust::{Geometry, NormRect};
use crate::output::{ExportFormat, MetadataPolicy, ProcessRecipe, ScaleMode};

/// Ordered stages of the layered export pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportStage {
    Plan,
    SourceAcquire,
    Render,
    Encode,
    Metadata,
    Publish,
}

impl ExportStage {
    pub const ALL: [Self; 6] = [
        Self::Plan,
        Self::SourceAcquire,
        Self::Render,
        Self::Encode,
        Self::Metadata,
        Self::Publish,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Plan => "plan",
            Self::SourceAcquire => "source_acquire",
            Self::Render => "render",
            Self::Encode => "encode",
            Self::Metadata => "metadata",
            Self::Publish => "publish",
        }
    }
}

impl fmt::Display for ExportStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Explicit memory budget estimated during the planning stage.
///
/// Covers source acquisition, working render/resample buffers and encoded
/// output staging so admission control can account for peak memory before
/// work starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportMemoryBudget {
    /// Working set for acquiring the full-resolution source frame.
    pub source_bytes: u64,
    /// Working set for rendering and resampling to the target output size.
    pub render_buffer_bytes: u64,
    /// Working set for encoding and metadata assembly.
    pub encode_buffer_bytes: u64,
    /// Conservative peak estimate across all active buffers.
    pub peak_bytes: u64,
}

/// Typed validation error returned by [`ExportPlan::from_recipe`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExportPlanError {
    ZeroSourceDimensions {
        width: u32,
        height: u32,
    },
    InvalidCrop {
        width: String,
        height: String,
    },
    UnsupportedFormat {
        format: ExportFormat,
        reason: String,
    },
    UnsupportedBitDepth {
        format: ExportFormat,
        bit_depth: u8,
        reason: String,
    },
    InvalidScaleValue {
        mode: ScaleMode,
        value: (u32, u32),
    },
    InvalidQuality {
        quality: u8,
    },
    InvalidDestination {
        path: PathBuf,
        reason: String,
    },
    DimensionsOverflow {
        width: u32,
        height: u32,
    },
}

impl fmt::Display for ExportPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSourceDimensions { width, height } => {
                write!(f, "export plan: invalid source dimensions {width}x{height} (must be non-zero)")
            }
            Self::InvalidCrop { width, height } => {
                write!(f, "export plan: invalid crop dimensions ({width} x {height})")
            }
            Self::UnsupportedFormat { format, reason } => {
                write!(f, "export plan: unsupported format {format:?}: {reason}")
            }
            Self::UnsupportedBitDepth { format, bit_depth, reason } => {
                write!(f, "export plan: unsupported {bit_depth}-bit depth for {format:?}: {reason}")
            }
            Self::InvalidScaleValue { mode, value } => {
                write!(f, "export plan: invalid scale value {value:?} for mode {mode:?}")
            }
            Self::InvalidQuality { quality } => {
                write!(f, "export plan: invalid quality {quality} (must be 0..=100)")
            }
            Self::InvalidDestination { path, reason } => {
                write!(
                    f,
                    "export plan: invalid destination {}: {reason}",
                    crate::output::log_name(path)
                )
            }
            Self::DimensionsOverflow { width, height } => {
                write!(f, "export plan: dimensions {width}x{height} overflow pixel/byte arithmetic")
            }
        }
    }
}

impl std::error::Error for ExportPlanError {}

/// Validated, immutable export plan produced before source decode or rendering.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportPlan {
    pub format: ExportFormat,
    pub bit_depth: u8,
    pub quality: u8,
    pub source_size: (u32, u32),
    pub developed_size: (u32, u32),
    pub output_size: (u32, u32),
    pub destination: Option<PathBuf>,
    pub icc_profile: String,
    pub metadata_policy: MetadataPolicy,
    pub estimated_output_bytes: u64,
    pub memory_budget: ExportMemoryBudget,
}

impl ExportPlan {
    /// Builds and validates an [`ExportPlan`] from a recipe, source dimensions
    /// and document geometry.
    ///
    /// Rejects zero source sizes, invalid crops, unsupported formats/bit depths,
    /// invalid scale parameters, out-of-range quality and non-directory
    /// destinations without silent fallback.
    pub fn from_recipe(
        recipe: &ProcessRecipe,
        source_size: (u32, u32),
        geometry: &Geometry,
    ) -> Result<Self, ExportPlanError> {
        let (src_w, src_h) = validate_source_size(source_size)?;

        recipe.format.ensure_supported().map_err(|err| ExportPlanError::UnsupportedFormat {
            format: recipe.format,
            reason: err.to_string(),
        })?;

        recipe.format.supports_bit_depth(recipe.bit_depth).map_err(|err| {
            ExportPlanError::UnsupportedBitDepth {
                format: recipe.format,
                bit_depth: recipe.bit_depth,
                reason: err.to_string(),
            }
        })?;

        if recipe.format.has_quality() && recipe.quality > 100 {
            return Err(ExportPlanError::InvalidQuality { quality: recipe.quality });
        }

        validate_scale(recipe.scale, recipe.scale_value)?;

        if let Some(dest) = recipe.destination.as_deref() {
            validate_destination_hint(dest)?;
        }

        let developed_size = developed_dimensions((src_w, src_h), geometry)?;
        let output_size = crate::recipe::output_size(recipe, developed_size);
        if output_size.0 == 0 || output_size.1 == 0 {
            return Err(ExportPlanError::ZeroSourceDimensions {
                width: output_size.0,
                height: output_size.1,
            });
        }

        let memory_budget = estimate_memory_budget(
            (src_w, src_h),
            developed_size,
            output_size,
            recipe.format,
            recipe.bit_depth,
        )?;
        let estimated_output_bytes =
            crate::recipe::estimate_bytes(output_size.0, output_size.1, recipe.format);

        Ok(Self {
            format: recipe.format,
            bit_depth: recipe.bit_depth,
            quality: recipe.quality,
            source_size: (src_w, src_h),
            developed_size,
            output_size,
            destination: recipe.destination.clone(),
            icc_profile: recipe.icc_profile.clone(),
            metadata_policy: recipe.include_metadata,
            estimated_output_bytes,
            memory_budget,
        })
    }

    /// Resolves and validates the concrete output directory (creating it if
    /// needed), falling back to `fallback_dir` when the recipe leaves
    /// `destination` as `None`.
    pub fn resolve_destination(&self, fallback_dir: &Path) -> Result<PathBuf, ExportPlanError> {
        let candidate = self.destination.as_deref().unwrap_or(fallback_dir);
        validate_destination_hint(candidate)?;
        crate::output::resolve_output_dir(candidate).map_err(|err| {
            ExportPlanError::InvalidDestination {
                path: candidate.to_path_buf(),
                reason: err.to_string(),
            }
        })
    }
}

fn validate_source_size((w, h): (u32, u32)) -> Result<(u32, u32), ExportPlanError> {
    if w == 0 || h == 0 {
        return Err(ExportPlanError::ZeroSourceDimensions { width: w, height: h });
    }
    if u64::from(w).checked_mul(u64::from(h)).is_none() {
        return Err(ExportPlanError::DimensionsOverflow { width: w, height: h });
    }
    Ok((w, h))
}

fn validate_scale(mode: ScaleMode, value: (u32, u32)) -> Result<(), ExportPlanError> {
    let (a, b) = value;
    let valid = match mode {
        ScaleMode::Fixed100 => true,
        ScaleMode::Width => a > 0,
        ScaleMode::Height | ScaleMode::LongEdge | ScaleMode::Percent => a > 0 || b > 0,
        ScaleMode::ShortEdge => a > 0 || b > 0,
        ScaleMode::Dimensions => a > 0 && b > 0,
    };
    if valid {
        Ok(())
    } else {
        Err(ExportPlanError::InvalidScaleValue { mode, value })
    }
}

fn validate_destination_hint(path: &Path) -> Result<(), ExportPlanError> {
    if path.as_os_str().is_empty() {
        return Err(ExportPlanError::InvalidDestination {
            path: path.to_path_buf(),
            reason: "destination path is empty".into(),
        });
    }
    if path.exists() && !path.is_dir() {
        return Err(ExportPlanError::InvalidDestination {
            path: path.to_path_buf(),
            reason: "destination exists and is not a directory".into(),
        });
    }
    Ok(())
}

fn developed_dimensions(
    source_size: (u32, u32),
    geometry: &Geometry,
) -> Result<(u32, u32), ExportPlanError> {
    let crop: NormRect = geometry.crop;
    if !crop.width.is_finite() || !crop.height.is_finite() || crop.width <= 0.0 || crop.height <= 0.0 {
        return Err(ExportPlanError::InvalidCrop {
            width: format!("{}", crop.width),
            height: format!("{}", crop.height),
        });
    }

    let (oriented_w, oriented_h) = if geometry.orientation % 2 == 1 {
        (source_size.1, source_size.0)
    } else {
        source_size
    };

    let dev_w = (f64::from(oriented_w) * f64::from(crop.width)).round();
    let dev_h = (f64::from(oriented_h) * f64::from(crop.height)).round();
    if !dev_w.is_finite() || !dev_h.is_finite() || dev_w < 1.0 || dev_h < 1.0 {
        return Err(ExportPlanError::InvalidCrop {
            width: format!("{}", crop.width),
            height: format!("{}", crop.height),
        });
    }
    if dev_w > f64::from(u32::MAX) || dev_h > f64::from(u32::MAX) {
        return Err(ExportPlanError::DimensionsOverflow {
            width: oriented_w,
            height: oriented_h,
        });
    }
    Ok((dev_w as u32, dev_h as u32))
}

fn estimate_memory_budget(
    source_size: (u32, u32),
    developed_size: (u32, u32),
    output_size: (u32, u32),
    format: ExportFormat,
    bit_depth: u8,
) -> Result<ExportMemoryBudget, ExportPlanError> {
    let checked_pixels = |(w, h): (u32, u32)| -> Result<u64, ExportPlanError> {
        u64::from(w)
            .checked_mul(u64::from(h))
            .ok_or(ExportPlanError::DimensionsOverflow { width: w, height: h })
    };

    let src_px = checked_pixels(source_size)?;
    let dev_px = checked_pixels(developed_size)?;
    let out_px = checked_pixels(output_size)?;

    // Source decode working set: 12 bytes/px (linear RGB f32 plane).
    let source_bytes = src_px
        .checked_mul(12)
        .ok_or(ExportPlanError::DimensionsOverflow {
            width: source_size.0,
            height: source_size.1,
        })?;

    // Output pixel format bytes per pixel: RGBA8 (4 B/px) or RGBA16 (8 B/px).
    let out_bpp: u64 = if bit_depth == 16 { 8 } else { 4 };

    // Render working buffer: developed RGBA f32 (16 B/px) + output raster.
    let dev_rgba_f32 = dev_px
        .checked_mul(16)
        .ok_or(ExportPlanError::DimensionsOverflow {
            width: developed_size.0,
            height: developed_size.1,
        })?;
    let out_raster = out_px
        .checked_mul(out_bpp)
        .ok_or(ExportPlanError::DimensionsOverflow {
            width: output_size.0,
            height: output_size.1,
        })?;
    let render_buffer_bytes = dev_rgba_f32
        .checked_add(out_raster)
        .ok_or(ExportPlanError::DimensionsOverflow {
            width: output_size.0,
            height: output_size.1,
        })?;

    // Encode staging: encoded byte estimate + output raster copy.
    let encoded_estimate = crate::recipe::estimate_bytes(output_size.0, output_size.1, format);
    let encode_buffer_bytes = out_raster
        .checked_add(encoded_estimate)
        .ok_or(ExportPlanError::DimensionsOverflow {
            width: output_size.0,
            height: output_size.1,
        })?;

    let peak_bytes = source_bytes
        .checked_add(render_buffer_bytes)
        .and_then(|v| v.checked_add(encode_buffer_bytes))
        .ok_or(ExportPlanError::DimensionsOverflow {
            width: output_size.0,
            height: output_size.1,
        })?;

    Ok(ExportMemoryBudget {
        source_bytes,
        render_buffer_bytes,
        encode_buffer_bytes,
        peak_bytes,
    })
}
