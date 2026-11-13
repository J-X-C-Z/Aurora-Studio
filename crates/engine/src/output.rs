//! Process (export) recipes and the jobs that run them.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::catalog::ImageId;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExportFormat {
    #[default]
    Jpeg,
    JpegXl,
    Tiff,
    Png,
    Heif,
    Dng,
    Psd,
    WebP,
}

impl ExportFormat {
    pub const ALL: [Self; 8] =
        [Self::Jpeg, Self::JpegXl, Self::Tiff, Self::Png, Self::Heif, Self::Dng, Self::Psd, Self::WebP];

    pub fn extension(self) -> &'static str {
        match self {
            Self::Jpeg => "jpg",
            Self::JpegXl => "jxl",
            Self::Tiff => "tif",
            Self::Png => "png",
            Self::Heif => "heic",
            Self::Dng => "dng",
            Self::Psd => "psd",
            Self::WebP => "webp",
        }
    }

    pub fn has_quality(self) -> bool {
        matches!(self, Self::Jpeg | Self::JpegXl | Self::Heif | Self::WebP)
    }

    pub fn has_bit_depth(self) -> bool {
        matches!(self, Self::Tiff | Self::Png | Self::Psd | Self::Heif)
    }

    /// The recipe-facing name, as Capture One writes it.
    pub fn label(self) -> &'static str {
        match self {
            Self::Jpeg => "JPEG",
            Self::JpegXl => "JPEG XL",
            Self::Tiff => "TIFF",
            Self::Png => "PNG",
            Self::Heif => "HEIF",
            Self::Dng => "DNG",
            Self::Psd => "PSD",
            Self::WebP => "WebP",
        }
    }

    /// Whether this build has an encoder for the format. The interface greys
    /// the others out; [`Self::ensure_supported`] is the exporter's own gate.
    pub fn is_supported(self) -> bool {
        match self {
            Self::Heif => crate::heif_write::available(),
            _ => matches!(self, Self::Jpeg | Self::Png | Self::Tiff | Self::WebP),
        }
    }

    /// Fails with [`UnsupportedFormat`] when there is no encoder, so a job is
    /// rejected before anything is rendered or written.
    pub fn ensure_supported(self) -> anyhow::Result<()> {
        if self == Self::Heif {
            return crate::heif_write::probe();
        }
        if self.is_supported() {
            Ok(())
        } else {
            Err(UnsupportedFormat { format: self }.into())
        }
    }

    /// Whether this format can be written at `depth` bits. Only PNG and TIFF
    /// have a 16-bit encoder; requesting 16 bits for anything else is an
    /// error, never a silent downgrade.
    pub fn supports_bit_depth(self, depth: u8) -> anyhow::Result<()> {
        match depth {
            8 => Ok(()),
            16 if matches!(self, Self::Png | Self::Tiff) => Ok(()),
            _ => anyhow::bail!(
                "{self:?} has no {depth}-bit encoder; only PNG and TIFF support 16-bit"
            ),
        }
    }
}

/// Stable category for a failed export job, for translated queue labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportErrorKind {
    Cancelled,
    Unsupported,
    Io,
    Other,
}

impl ExportErrorKind {
    pub fn of(message: &str) -> Self {
        let lower = message.to_ascii_lowercase();
        if lower.contains("cancelled") {
            Self::Cancelled
        } else if lower.contains("unsupported") || lower.contains("no encoder") || lower.contains("not supported") {
            Self::Unsupported
        } else if lower.contains("os error")
            || lower.contains("no such file")
            || lower.contains("permission denied")
            || lower.contains("disk full")
            || lower.contains("read-only")
        {
            Self::Io
        } else {
            Self::Other
        }
    }
}

/// A format the exporter has no encoder for. Downcast from the job error to
/// show the format in the interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnsupportedFormat {
    pub format: ExportFormat,
}

impl std::fmt::Display for UnsupportedFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "export: {} (.{}) is not implemented; supported formats are JPEG, PNG, TIFF and WebP",
            self.format.label(),
            self.format.extension()
        )
    }
}

impl std::error::Error for UnsupportedFormat {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScaleMode {
    #[default]
    Fixed100,
    Width,
    Height,
    LongEdge,
    ShortEdge,
    Dimensions,
    Percent,
}

impl ScaleMode {
    pub const ALL: [Self; 7] = [
        Self::Fixed100,
        Self::Width,
        Self::Height,
        Self::LongEdge,
        Self::ShortEdge,
        Self::Dimensions,
        Self::Percent,
    ];
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputSharpening {
    #[default]
    Disabled,
    Screen,
    Print,
}

/// Output-sharpening numbers read from Capture One's recipe XML.
///
/// `Control`, `Method` and viewing distance are preserved without acting on
/// them: their semantics have not been cross-checked against Capture One.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputSharpeningSettings {
    /// XML value; C1's scale is not proven, so it is passed through unscaled.
    pub amount: f32,
    /// Pixel radius after rounding in the current Aurora implementation.
    pub radius: f32,
    /// Conservative detail threshold in 8-bit levels.
    pub threshold: f32,
    pub control: u8,
    pub method: u8,
    pub viewing_distance: u32,
    pub viewing_distance_unit: u8,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProcessRecipe {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    pub format: ExportFormat,
    /// 0 ..= 100.
    pub quality: u8,
    /// 8 or 16.
    pub bit_depth: u8,
    pub icc_profile: String,
    pub resolution_ppi: u32,
    pub scale: ScaleMode,
    pub scale_value: (u32, u32),
    pub output_sharpening: OutputSharpening,
    pub output_sharpening_settings: Option<OutputSharpeningSettings>,
    /// Folder; `None` writes beside the source in an "Output" folder.
    pub destination: Option<PathBuf>,
    /// Tokens such as `[Image Name]`, `[Counter]`, `[Date]`.
    pub file_name_format: String,
    pub include_metadata: MetadataPolicy,
    pub watermark: Option<Watermark>,
    pub open_with: Option<String>,
}

impl Default for ProcessRecipe {
    fn default() -> Self {
        Self {
            id: 0,
            name: "JPEG".into(),
            enabled: true,
            format: ExportFormat::Jpeg,
            quality: 90,
            bit_depth: 8,
            icc_profile: "sRGB IEC61966-2.1".into(),
            resolution_ppi: 300,
            scale: ScaleMode::Fixed100,
            scale_value: (0, 0),
            output_sharpening: OutputSharpening::Disabled,
            output_sharpening_settings: None,
            destination: None,
            file_name_format: "[Image Name]".into(),
            include_metadata: MetadataPolicy::default(),
            watermark: None,
            open_with: None,
        }
    }
}

/// What the exporter writes into the file's metadata.
///
/// Each variant writes the one group it names: the EXIF-only groups replace the
/// whole EXIF block with just their tags, and the XMP groups write an XMP
/// packet (JPEG APP1, PNG `XML:com.adobe.xmp` iTXt) with nothing else.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetadataPolicy {
    /// Copy the source EXIF into the output.
    ///
    /// Implemented for sources that carry a JPEG APP1 EXIF segment: the
    /// segment payload is copied verbatim. Sources without one (RAW, TIFF,
    /// PNG) get no EXIF in the output.
    #[default]
    Exif,
    /// Write no metadata at all.
    ///
    /// The output profile is still embedded, because a colour-managed file
    /// without its profile is not readable.
    StripAll,
    /// IFD0 `Copyright` (0x8298) only.
    Copyright,
    /// Camera make and model, lens and exposure settings.
    Camera,
    /// The GPS IFD only.
    Gps,
    /// XMP `dc:subject` only.
    Keywords,
    /// XMP `xmp:Rating` and `xmp:Label` only.
    RatingAndTag,
}

impl MetadataPolicy {
    /// Whether the exporter knows how to write this policy.
    pub fn is_implemented(self) -> bool {
        true
    }

    /// The recipe-facing name of this policy.
    pub fn label(self) -> &'static str {
        match self {
            Self::Exif => "Exif",
            Self::StripAll => "Strip all",
            Self::Copyright => "Copyright",
            Self::Camera => "Camera",
            Self::Gps => "GPS",
            Self::Keywords => "Keywords",
            Self::RatingAndTag => "Rating and tag",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Watermark {
    pub text: String,
    /// 0 ..= 100.
    pub opacity: f32,
    pub scale: f32,
    pub position: [f32; 2],
}

/// One image through one recipe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportJob {
    pub job_id: u64,
    pub image: ImageId,
    pub recipe: ProcessRecipe,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum JobStatus {
    Queued,
    Running { progress: f32 },
    Done { output: PathBuf },
    Failed { message: String },
}

/// The output name template used when a recipe does not set one.
pub const DEFAULT_OUTPUT_TEMPLATE: &str = "{name}.{ext}";

/// Path for one exported image.
///
/// Placeholders: `{name}` source stem, `{counter}` four-digit index, `{ext}`
/// file extension, `{date}` today as `YYYYMMDD`. Unknown placeholders are kept
/// verbatim, and characters Windows forbids in a file name become `_`.
pub fn output_path(template: &str, stem: &str, counter: u32, ext: &str) -> PathBuf {
    let mut name = String::with_capacity(template.len() + stem.len());
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        name.push_str(&rest[..open]);
        rest = &rest[open + 1..];
        let Some(close) = rest.find('}') else {
            name.push('{');
            break;
        };
        match &rest[..close] {
            "name" => name.push_str(stem),
            "counter" => name.push_str(&format!("{counter:04}")),
            "ext" => name.push_str(ext),
            "date" => name.push_str(&today()),
            token => {
                name.push('{');
                name.push_str(token);
                name.push('}');
            }
        }
        rest = &rest[close + 1..];
    }
    name.push_str(rest);
    PathBuf::from(
        name.replace(|c: char| matches!(c, '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|'), "_"),
    )
}

fn today() -> String {
    let days = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| (d.as_secs() / 86_400) as i64);
    ymd(days)
}

/// Civil date for a count of days since 1970-01-01 (Howard Hinnant's algorithm).
fn ymd(days: i64) -> String {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{:04}{:02}{:02}", if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_template_keeps_the_stem_and_extension() {
        assert_eq!(
            output_path(DEFAULT_OUTPUT_TEMPLATE, "DSC_0042", 7, "jpg"),
            PathBuf::from("DSC_0042.jpg")
        );
    }

    #[test]
    fn counter_is_four_digit_zero_padded() {
        assert_eq!(
            output_path("{name}_{counter}.{ext}", "DSC", 7, "tif"),
            PathBuf::from("DSC_0007.tif")
        );
        assert_eq!(
            output_path("{name}_{counter}.{ext}", "DSC", 12_345, "tif"),
            PathBuf::from("DSC_12345.tif")
        );
    }

    #[test]
    fn date_is_eight_digit_yyyymmdd() {
        assert_eq!(ymd(0), "19700101");
        assert_eq!(ymd(19_782), "20240229");
        let path = output_path("{date}.{ext}", "ignored", 0, "jpg");
        let name = path.to_string_lossy();
        assert_eq!(name.len(), 12, "{name}");
        assert!(name.starts_with(|c: char| c.is_ascii_digit()), "{name}");
        assert!(name[..8].chars().all(|c| c.is_ascii_digit()), "{name}");
        assert!(name.ends_with(".jpg"), "{name}");
    }

    #[test]
    fn unknown_placeholders_are_kept_verbatim() {
        assert_eq!(
            output_path("{name}_{bogus}.{ext}", "DSC", 1, "png"),
            PathBuf::from("DSC_{bogus}.png")
        );
        assert_eq!(output_path("{name", "DSC", 1, "png"), PathBuf::from("{name"));
    }

    #[test]
    fn illegal_characters_become_underscores() {
        assert_eq!(
            output_path("{name}.{ext}", "a/b:c*?\"<>|", 0, "jpg"),
            PathBuf::from("a_b_c______.jpg")
        );
        assert_eq!(
            output_path("{name}.{ext}", "back\\slash", 0, "jpg"),
            PathBuf::from("back_slash.jpg")
        );
    }

    #[test]
    fn export_failures_map_to_stable_kinds() {
        assert_eq!(ExportErrorKind::of("cancelled"), ExportErrorKind::Cancelled);
        assert_eq!(ExportErrorKind::of("Unsupported: no encoder for PSD"), ExportErrorKind::Unsupported);
        assert_eq!(ExportErrorKind::of("No such file or directory (os error 2)"), ExportErrorKind::Io);
        assert_eq!(ExportErrorKind::of("render failed"), ExportErrorKind::Other);
    }
}
