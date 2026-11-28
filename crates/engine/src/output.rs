//! Process (export) recipes and the jobs that run them.
//!
//! The bottom of this file also holds the export write path shared by the
//! engine, the batch runner and the session writer: a **same-directory**
//! temporary file, `flush` + `sync_all`, then an atomic rename, with a
//! copy-based fallback when a rename would cross volumes. Nothing here writes
//! to the final path before the new bytes are durable, so a cancelled or
//! failed export leaves the previous target file byte-for-byte intact.

use std::fs::{File, OpenOptions};
use std::io::{self, Write as _};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

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
            // HEIF stores 10/12-bit HEVC; the recipe's 16-bit buffer is
            // quantised to 12 bits by `heif_write::encode_rgb16`.
            16 if matches!(self, Self::Png | Self::Tiff | Self::Heif) => Ok(()),
            _ => anyhow::bail!(
                "{self:?} has no {depth}-bit encoder; only PNG, TIFF and HEIF support 16-bit"
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

/// Raw `Watermark*` values read from a Capture One recipe XML.
///
/// Every field is optional so a missing key stays `None`. Text and color
/// values are preserved as C1 wrote them; this type does not expand tokens or
/// drive the watermark renderer.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WatermarkSettings {
    pub watermark_type: Option<u8>,
    pub text: Option<String>,
    pub font_name: Option<String>,
    pub font_type: Option<u8>,
    pub font_size: Option<f32>,
    pub pixel_font_size: Option<f32>,
    pub point_font_size: Option<f32>,
    pub text_scale: Option<f32>,
    pub text_relative_size: Option<f32>,
    pub text_opacity: Option<f32>,
    /// C1's `red:..;green:..;blue:..;` string, kept verbatim.
    pub color: Option<String>,
    pub underline: Option<u8>,
    /// C1's `red:..;green:..;blue:..;` string, kept verbatim.
    pub underline_color: Option<String>,
    pub strike_through: Option<u8>,
    /// C1's `red:..;green:..;blue:..;` string, kept verbatim.
    pub strike_through_color: Option<String>,
    pub h_text_placement: Option<f32>,
    pub v_text_placement: Option<f32>,
    pub image_file_path: Option<String>,
    pub image_opacity: Option<f32>,
    pub image_scale: Option<f32>,
    pub h_image_placement: Option<f32>,
    pub v_image_placement: Option<f32>,
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
    /// First value of `[Counter]`; Capture One starts at 1.
    pub file_name_counter_start: u32,
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
            file_name_counter_start: 1,
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

/// Why an output path was refused before anything was written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputPathRefusal {
    /// The destination is empty, or names something that is not a folder.
    NotADirectory,
    /// The file name would leave the destination (`..`, a separator, a drive).
    EscapesDestination,
}

impl std::fmt::Display for OutputPathRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotADirectory => write!(f, "export: the destination is not a folder"),
            Self::EscapesDestination => {
                write!(f, "export: the output name would leave the destination folder")
            }
        }
    }
}

impl std::error::Error for OutputPathRefusal {}

/// Marker prefix of a staging directory and of a same-directory temporary file.
/// Hidden on POSIX, ignored by the image scan on both platforms because of the
/// `.tmp` suffix.
pub(crate) const STAGING_PREFIX: &str = ".aurora-export-";
const TEMP_SUFFIX: &str = ".tmp";

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

/// Creates `dir` if needed and returns its canonical form.
///
/// The canonical path is what the containment check compares against, so a
/// junction or a `..` in the recipe's destination resolves before any name is
/// joined to it.
pub(crate) fn resolve_output_dir(dir: &Path) -> anyhow::Result<PathBuf> {
    if dir.as_os_str().is_empty() {
        return Err(OutputPathRefusal::NotADirectory.into());
    }
    if dir.exists() && !dir.is_dir() {
        return Err(OutputPathRefusal::NotADirectory.into());
    }
    std::fs::create_dir_all(dir)?;
    let canonical = dir
        .canonicalize()
        .map_err(|_| anyhow::Error::from(OutputPathRefusal::NotADirectory))?;
    if !canonical.is_dir() {
        return Err(OutputPathRefusal::NotADirectory.into());
    }
    Ok(canonical)
}

/// Joins exactly one file name to the destination and refuses everything that
/// could leave it: `..`, `.`, a separator, an absolute path or a drive prefix.
///
/// The returned path is not canonicalized (the file does not exist yet); the
/// caller passes it to [`publish`], which never writes through it before the
/// new bytes are durable.
pub(crate) fn contained_output(dir: &Path, file_name: &Path) -> anyhow::Result<PathBuf> {
    // A backslash or a colon is a separator/drive on Windows even when the
    // check runs on another host, so both are refused everywhere.
    if file_name
        .to_str()
        .is_some_and(|name| name.contains('\\') || name.contains(':'))
    {
        return Err(OutputPathRefusal::EscapesDestination.into());
    }
    let mut components = file_name.components();
    let single = match (components.next(), components.next()) {
        (Some(Component::Normal(name)), None) => name,
        _ => return Err(OutputPathRefusal::EscapesDestination.into()),
    };
    if single.is_empty() || single == "." || single == ".." {
        return Err(OutputPathRefusal::EscapesDestination.into());
    }
    Ok(dir.join(single))
}

/// `name.ext` for `n == 0`, `name-1.ext` … for later attempts.
fn numbered_name(name: &str, ext: &str, n: u32) -> String {
    let ext = ext.trim_start_matches('.');
    let suffix = if n == 0 { String::new() } else { format!("-{n}") };
    if ext.is_empty() {
        format!("{name}{suffix}")
    } else {
        format!("{name}{suffix}.{ext}")
    }
}

/// Turns an untrusted file name (a catalog's `ZIMAGE.ZFILENAME`, for example)
/// into one path component. Returns `None` for names that would name the folder
/// itself or a parent; separators and Windows-reserved characters become `_`.
pub(crate) fn sanitize_file_name(name: &str) -> Option<String> {
    let cleaned: String = name
        .chars()
        .map(|c| if matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control() {
            '_'
        } else {
            c
        })
        .collect();
    let cleaned = cleaned.trim().trim_end_matches(['.', ' ']).to_owned();
    if cleaned.is_empty() || cleaned == "." || cleaned == ".." {
        None
    } else {
        Some(cleaned)
    }
}

/// A hidden temporary file beside `target`, so the later rename stays on one
/// volume. `create_new` is the caller's job; a crashed run leaves one behind
/// and [`sweep_stale_staging`] removes it.
pub(crate) fn sibling_temp(target: &Path, tag: &str) -> io::Result<PathBuf> {
    let name = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("export");
    let dir = target.parent().unwrap_or(Path::new("."));
    Ok(dir.join(format!(
        "{STAGING_PREFIX}{name}.{tag}-{}-{}{TEMP_SUFFIX}",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    )))
}

fn not_same_device(error: &io::Error) -> bool {
    #[cfg(windows)]
    {
        // ERROR_NOT_SAME_DEVICE
        return error.raw_os_error() == Some(17);
    }
    #[cfg(unix)]
    {
        // EXDEV
        return error.raw_os_error() == Some(18);
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = error;
        false
    }
}

/// Publishes a same-directory temp for a caller that owns the target and may
/// replace its own previous bytes (the test-only atomic writer).
#[cfg(test)]
fn publish_replace(temp: &Path, target: &Path) -> io::Result<()> {
    publish_using(temp, target, &mut |from, to| std::fs::rename(from, to), true)
}

fn publish_using(
    temp: &Path,
    target: &Path,
    rename: &mut dyn FnMut(&Path, &Path) -> io::Result<()>,
    replace: bool,
) -> io::Result<()> {
    match rename(temp, target) {
        Ok(()) => {
            sync_parent(target);
            Ok(())
        }
        Err(error) if not_same_device(&error) => copy_publish(temp, target, replace),
        Err(error) => Err(error),
    }
}

/// Publishes without ever replacing an existing file.
///
/// A same-volume hard link is atomic and fails with `AlreadyExists` when the
/// name is taken, so two exports that pick the same name cannot clobber each
/// other. Filesystems without hard links fall back to [`publish`]; the caller
/// re-checks the name before that anyway.
fn publish_no_clobber(temp: &Path, target: &Path) -> io::Result<()> {
    match std::fs::hard_link(temp, target) {
        Ok(()) => {
            let _ = std::fs::remove_file(temp);
            sync_parent(target);
            Ok(())
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Err(error),
        Err(_) => {
            // Filesystems without hard links fall back to a rename, which
            // replaces on POSIX: refuse an existing target first.
            if target.exists() {
                return Err(io::Error::from(io::ErrorKind::AlreadyExists));
            }
            publish_using(temp, target, &mut |from, to| std::fs::rename(from, to), false)
        }
    }
}

/// Cross-volume degradation: copy beside the target, sync, then rename.
fn copy_publish(temp: &Path, target: &Path, replace: bool) -> io::Result<()> {
    let staged = sibling_temp(target, "vol")?;
    let result = (|| {
        let mut source = File::open(temp)?;
        let mut dest = OpenOptions::new().write(true).create_new(true).open(&staged)?;
        io::copy(&mut source, &mut dest)?;
        dest.flush()?;
        dest.sync_all()?;
        drop(dest);
        if !replace && target.exists() {
            return Err(io::Error::from(io::ErrorKind::AlreadyExists));
        }
        std::fs::rename(&staged, target)?;
        sync_parent(target);
        Ok(())
    })();
    match result {
        Ok(()) => {
            let _ = std::fs::remove_file(temp);
            Ok(())
        }
        Err(error) => {
            let _ = std::fs::remove_file(&staged);
            Err(error)
        }
    }
}

/// Writes `target` through a same-directory temp: `create_new`, `write`,
/// `flush`, `sync_all`, rename. On any error the temp is removed and the
/// existing target keeps its bytes.
///
/// The closure is the injection point for the fault matrix (a writer that
/// reports a full disk after N bytes), so this helper is test-only.
#[cfg(test)]
pub(crate) fn write_atomic_with(
    target: &Path,
    tag: &str,
    write: impl FnOnce(&mut File) -> io::Result<()>,
) -> io::Result<()> {
    let dir = target.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)?;
    let temp = sibling_temp(target, tag)?;
    let result = (|| {
        let mut file = OpenOptions::new().write(true).create_new(true).open(&temp)?;
        write(&mut file)?;
        file.flush()?;
        file.sync_all()?;
        drop(file);
        publish_replace(&temp, target)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

/// `sync_all` on one file; used by the session writer before its rename.
///
/// Opened for writing because `FlushFileBuffers` rejects a read-only handle
/// (Windows error 5); every caller passes a file this process just wrote.
pub(crate) fn sync_file(path: &Path) -> io::Result<()> {
    let file = OpenOptions::new().write(true).open(path)?;
    file.sync_all()
}

/// `sync_all` on every regular file below `dir`; used for the small
/// `Adjustments/` subtree of a session bundle.
pub(crate) fn sync_tree(dir: &Path) -> io::Result<()> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            sync_tree(&path)?;
        } else if file_type.is_file() {
            sync_file(&path)?;
        }
    }
    Ok(())
}

#[cfg(unix)]
pub(crate) fn sync_parent(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }
}

#[cfg(not(unix))]
pub(crate) fn sync_parent(_path: &Path) {}

/// A staging folder inside the destination's volume. Exporters write their
/// bytes here; [`StagedExport::publish`] renames the finished file to a free
/// name in the destination. Dropping without publishing removes the folder.
pub(crate) struct StagedExport {
    dir: PathBuf,
    target_dir: PathBuf,
}

impl StagedExport {
    pub(crate) fn new(target_dir: &Path, tag: impl std::fmt::Display) -> anyhow::Result<Self> {
        let dir = target_dir.join(format!(
            "{STAGING_PREFIX}{}-{tag}{TEMP_SUFFIX}",
            std::process::id()
        ));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir, target_dir: target_dir.to_path_buf() })
    }

    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    /// Publishes `written` (a file inside the staging folder) to the first free
    /// name in the destination. `written` is synced first.
    ///
    /// The name is claimed with a hard link, so a concurrent export that chose
    /// the same name gets `AlreadyExists` and moves to the next suffix instead
    /// of overwriting a file.
    pub(crate) fn publish(self, written: &Path) -> anyhow::Result<PathBuf> {
        let name = written
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| "export".into());
        let ext = written.extension().map(|ext| ext.to_string_lossy().into_owned()).unwrap_or_default();
        let first = contained_output(&self.target_dir, &Path::new(&name).with_extension(&ext))?;
        sync_file(written)?;
        for n in 0..10_000u32 {
            let candidate = if n == 0 {
                first.clone()
            } else {
                let file_name = numbered_name(&name, &ext, n);
                contained_output(&self.target_dir, Path::new(&file_name))?
            };
            if candidate.exists() {
                continue;
            }
            match publish_no_clobber(written, &candidate) {
                Ok(()) => return Ok(candidate),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        anyhow::bail!("export: no free output name for {name}.{ext}")
    }
}

impl Drop for StagedExport {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Whether `name` is one of our staging folders or temporary files.
pub(crate) fn is_staging_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with(STAGING_PREFIX))
}

/// Removes staging folders/files in `dir` last modified before `cutoff`.
///
/// A crashed export leaves one behind; running this on the next export of the
/// same destination reclaims the space without touching anything newer.
pub(crate) fn sweep_stale_staging_before(dir: &Path, cutoff: SystemTime) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else { return 0 };
    let mut removed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if !is_staging_path(&path) {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        let Ok(modified) = meta.modified() else { continue };
        if modified >= cutoff {
            continue;
        }
        let ok = if meta.is_dir() {
            std::fs::remove_dir_all(&path).is_ok()
        } else {
            std::fs::remove_file(&path).is_ok()
        };
        if ok {
            removed += 1;
        }
    }
    removed
}

/// [`sweep_stale_staging_before`] with a one-hour age, as used at export start.
pub(crate) fn sweep_stale_staging(dir: &Path) -> usize {
    sweep_stale_staging_before(dir, SystemTime::now() - Duration::from_secs(3600))
}

/// The file name of `path`, with no directory at all, for a log line.
pub(crate) fn log_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "<unnamed>".into())
}

/// Masks absolute paths (Windows drive, UNC, POSIX) in free-form log text.
///
/// Export diagnostics keep the file name and the error, never the folder a
/// photograph lives in, so a log file cannot be used to locate a user's shoot.
pub(crate) fn sanitize_log_text(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        let rest = &text[i..];
        let drive = bytes.get(i).is_some_and(|b| b.is_ascii_alphabetic())
            && bytes.get(i + 1) == Some(&b':')
            && matches!(bytes.get(i + 2), Some(b'\\' | b'/'));
        let unc = rest.starts_with("\\\\");
        let posix = bytes.get(i) == Some(&b'/');
        if drive || unc || posix {
            out.push_str("<path>");
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                // Keep sentence punctuation that ends the path in the message.
                if matches!(bytes[i], b')' | b',' | b';' | b'"' | b'\'') {
                    break;
                }
                i += 1;
            }
            continue;
        }
        let ch = rest.chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Effective USM parameters for output sharpening.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputSharpeningParams {
    /// Sharpening strength multiplier (e.g. 0.45 for screen, 0.75 for print).
    pub amount: f32,
    /// Gaussian blur radius in pixels.
    pub radius: f32,
    /// Detail threshold in 8-bit luminance levels (0..=255).
    pub threshold: f32,
}

impl OutputSharpening {
    /// Resolves effective USM parameters from the recipe's output sharpening mode,
    /// target dimensions, resolution (PPI), and optional explicit settings.
    ///
    /// Returns `None` when sharpening is disabled or resolves to zero strength.
    pub fn resolve_params(
        self,
        width: u32,
        height: u32,
        resolution_ppi: u32,
        explicit: Option<&OutputSharpeningSettings>,
    ) -> Option<OutputSharpeningParams> {
        if let Some(settings) = explicit {
            // Capture One control codes: 0 = Screen, 1 = Print, 2..=4 disable output USM.
            if settings.control > 1 {
                return None;
            }
            if settings.amount.is_finite() && settings.amount > 0.0 {
                let default_radius = match settings.control {
                    1 => (resolution_ppi.max(72) as f32 / 200.0).clamp(0.8, 3.0),
                    _ => {
                        let long_edge = width.max(height).max(1) as f32;
                        (0.6 + (long_edge / 2000.0) * 0.6).clamp(0.6, 1.8)
                    }
                };
                let radius = if settings.radius.is_finite() && settings.radius > 0.0 {
                    settings.radius.clamp(0.3, 12.0)
                } else {
                    default_radius
                };
                let threshold = if settings.threshold.is_finite() && settings.threshold > 0.0 {
                    settings.threshold.clamp(0.0, 255.0)
                } else {
                    0.0
                };
                return Some(OutputSharpeningParams {
                    amount: settings.amount,
                    radius,
                    threshold,
                });
            }
        }

        match self {
            Self::Disabled => None,
            Self::Screen => {
                let long_edge = width.max(height).max(1) as f32;
                let radius = (0.6 + (long_edge / 2000.0) * 0.6).clamp(0.6, 1.8);
                Some(OutputSharpeningParams {
                    amount: 0.45,
                    radius,
                    threshold: 2.0,
                })
            }
            Self::Print => {
                let ppi = resolution_ppi.max(72) as f32;
                let radius = (ppi / 200.0).clamp(0.8, 3.0);
                Some(OutputSharpeningParams {
                    amount: 0.75,
                    radius,
                    threshold: 1.5,
                })
            }
        }
    }
}

fn gaussian_kernel_1d(radius: f32) -> Vec<f32> {
    let sigma = radius.clamp(0.3, 12.0);
    let half = (sigma * 2.5).ceil().clamp(1.0, 16.0) as usize;
    let mut kernel = Vec::with_capacity(half * 2 + 1);
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);
    let mut sum = 0.0f32;
    for i in 0..=(half * 2) {
        let d = i as f32 - half as f32;
        let w = (-d * d * inv_two_sigma_sq).exp();
        kernel.push(w);
        sum += w;
    }
    if sum > 0.0 {
        for w in &mut kernel {
            *w /= sum;
        }
    }
    kernel
}

/// Applies separable Gaussian unsharp-mask (USM) sharpening to an 8-bit RGBA buffer in place.
///
/// Alpha is preserved untouched. Returns `true` if sharpening ran.
pub fn apply_output_sharpening_rgba8(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    params: OutputSharpeningParams,
) -> bool {
    let (w, h) = (width as usize, height as usize);
    let Some(pixel_count) = w.checked_mul(h) else { return false };
    let Some(byte_len) = pixel_count.checked_mul(4) else { return false };
    if w == 0 || h == 0 || rgba.len() < byte_len {
        return false;
    }
    if !params.amount.is_finite() || params.amount <= 0.0 {
        return false;
    }

    let kernel = gaussian_kernel_1d(params.radius);
    let half = kernel.len() / 2;
    let mut horiz = vec![0.0f32; pixel_count * 3];

    for y in 0..h {
        let row_base = y * w;
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for (ki, &weight) in kernel.iter().enumerate() {
                let sx = (x as isize + ki as isize - half as isize).clamp(0, w as isize - 1) as usize;
                let idx = (row_base + sx) * 4;
                acc[0] += rgba[idx] as f32 * weight;
                acc[1] += rgba[idx + 1] as f32 * weight;
                acc[2] += rgba[idx + 2] as f32 * weight;
            }
            let out = (row_base + x) * 3;
            horiz[out..out + 3].copy_from_slice(&acc);
        }
    }

    let threshold = if params.threshold.is_finite() && params.threshold > 0.0 {
        params.threshold
    } else {
        0.0
    };

    for y in 0..h {
        let row_base = y * w;
        for x in 0..w {
            let mut blurred = [0.0f32; 3];
            for (ki, &weight) in kernel.iter().enumerate() {
                let sy = (y as isize + ki as isize - half as isize).clamp(0, h as isize - 1) as usize;
                let hidx = (sy * w + x) * 3;
                blurred[0] += horiz[hidx] * weight;
                blurred[1] += horiz[hidx + 1] * weight;
                blurred[2] += horiz[hidx + 2] * weight;
            }
            let idx = (row_base + x) * 4;
            for c in 0..3 {
                let orig = rgba[idx + c] as f32;
                let detail = orig - blurred[c];
                if detail.abs() > threshold {
                    rgba[idx + c] = (orig + params.amount * detail).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    true
}

/// Applies separable Gaussian unsharp-mask (USM) sharpening to a 16-bit RGBA buffer in place.
///
/// `params.threshold` is interpreted in 8-bit levels and scaled to 16-bit (`* 257.0`).
/// Alpha is preserved untouched. Returns `true` if sharpening ran.
pub fn apply_output_sharpening_rgba16(
    rgba: &mut [u16],
    width: u32,
    height: u32,
    params: OutputSharpeningParams,
) -> bool {
    let (w, h) = (width as usize, height as usize);
    let Some(pixel_count) = w.checked_mul(h) else { return false };
    let Some(sample_len) = pixel_count.checked_mul(4) else { return false };
    if w == 0 || h == 0 || rgba.len() < sample_len {
        return false;
    }
    if !params.amount.is_finite() || params.amount <= 0.0 {
        return false;
    }

    let kernel = gaussian_kernel_1d(params.radius);
    let half = kernel.len() / 2;
    let mut horiz = vec![0.0f32; pixel_count * 3];

    for y in 0..h {
        let row_base = y * w;
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for (ki, &weight) in kernel.iter().enumerate() {
                let sx = (x as isize + ki as isize - half as isize).clamp(0, w as isize - 1) as usize;
                let idx = (row_base + sx) * 4;
                acc[0] += rgba[idx] as f32 * weight;
                acc[1] += rgba[idx + 1] as f32 * weight;
                acc[2] += rgba[idx + 2] as f32 * weight;
            }
            let out = (row_base + x) * 3;
            horiz[out..out + 3].copy_from_slice(&acc);
        }
    }

    let threshold_16 = if params.threshold.is_finite() && params.threshold > 0.0 {
        params.threshold * 257.0
    } else {
        0.0
    };

    for y in 0..h {
        let row_base = y * w;
        for x in 0..w {
            let mut blurred = [0.0f32; 3];
            for (ki, &weight) in kernel.iter().enumerate() {
                let sy = (y as isize + ki as isize - half as isize).clamp(0, h as isize - 1) as usize;
                let hidx = (sy * w + x) * 3;
                blurred[0] += horiz[hidx] * weight;
                blurred[1] += horiz[hidx + 1] * weight;
                blurred[2] += horiz[hidx + 2] * weight;
            }
            let idx = (row_base + x) * 4;
            for c in 0..3 {
                let orig = rgba[idx + c] as f32;
                let detail = orig - blurred[c];
                if detail.abs() > threshold_16 {
                    rgba[idx + c] = (orig + params.amount * detail).round().clamp(0.0, 65535.0) as u16;
                }
            }
        }
    }
    true
}

/// Applies the recipe's configured output sharpening to an 8-bit RGBA frame.
pub fn apply_recipe_output_sharpening_rgba8(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    recipe: &ProcessRecipe,
) -> bool {
    let Some(params) = recipe.output_sharpening.resolve_params(
        width,
        height,
        recipe.resolution_ppi,
        recipe.output_sharpening_settings.as_ref(),
    ) else {
        return false;
    };
    apply_output_sharpening_rgba8(rgba, width, height, params)
}

/// Applies the recipe's configured output sharpening to a 16-bit RGBA frame.
pub fn apply_recipe_output_sharpening_rgba16(
    rgba: &mut [u16],
    width: u32,
    height: u32,
    recipe: &ProcessRecipe,
) -> bool {
    let Some(params) = recipe.output_sharpening.resolve_params(
        width,
        height,
        recipe.resolution_ppi,
        recipe.output_sharpening_settings.as_ref(),
    ) else {
        return false;
    };
    apply_output_sharpening_rgba16(rgba, width, height, params)
}

/// Anchor position for image watermark placement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WatermarkAnchor {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    #[default]
    BottomRight,
    /// Places the watermark center at normalized `[x, y]` in `0.0..=1.0`.
    Normalized,
}

/// Parameters for compositing an RGBA image watermark onto an exported frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageWatermark {
    pub anchor: WatermarkAnchor,
    /// Normalized `[x, y]` (`0.0..=1.0`) used when `anchor == WatermarkAnchor::Normalized`.
    pub position: [f32; 2],
    /// Uniform scale factor applied to the watermark dimensions (`1.0` = 100%).
    pub scale: f32,
    /// Opacity percentage in `0.0..=100.0`. Values `<= 0.0` are a strict no-op.
    pub opacity: f32,
    /// Margin in pixels from anchored edges (may be negative to intentionally clip).
    pub margin_px: i32,
}

impl Default for ImageWatermark {
    fn default() -> Self {
        Self {
            anchor: WatermarkAnchor::BottomRight,
            position: [0.5, 0.5],
            scale: 1.0,
            opacity: 100.0,
            margin_px: 16,
        }
    }
}

impl ImageWatermark {
    /// Builds an [`ImageWatermark`] from parsed Capture One [`WatermarkSettings`].
    pub fn from_settings(settings: &WatermarkSettings) -> Self {
        let opacity = settings.image_opacity.unwrap_or(100.0).clamp(0.0, 100.0);
        let scale = settings.image_scale.unwrap_or(1.0).max(0.0);
        let pos_x = settings.h_image_placement.unwrap_or(0.5);
        let pos_y = settings.v_image_placement.unwrap_or(0.5);
        let anchor = if settings.h_image_placement.is_some() || settings.v_image_placement.is_some() {
            WatermarkAnchor::Normalized
        } else {
            WatermarkAnchor::BottomRight
        };
        Self {
            anchor,
            position: [pos_x, pos_y],
            scale,
            opacity,
            margin_px: 16,
        }
    }
}

fn watermark_origin(
    canvas_w: u32,
    canvas_h: u32,
    scaled_w: u32,
    scaled_h: u32,
    spec: &ImageWatermark,
) -> (i64, i64) {
    let cw = canvas_w as i64;
    let ch = canvas_h as i64;
    let ww = scaled_w as i64;
    let wh = scaled_h as i64;
    let m = spec.margin_px as i64;
    match spec.anchor {
        WatermarkAnchor::TopLeft => (m, m),
        WatermarkAnchor::TopCenter => ((cw - ww) / 2, m),
        WatermarkAnchor::TopRight => (cw - ww - m, m),
        WatermarkAnchor::CenterLeft => (m, (ch - wh) / 2),
        WatermarkAnchor::Center => ((cw - ww) / 2, (ch - wh) / 2),
        WatermarkAnchor::CenterRight => (cw - ww - m, (ch - wh) / 2),
        WatermarkAnchor::BottomLeft => (m, ch - wh - m),
        WatermarkAnchor::BottomCenter => ((cw - ww) / 2, ch - wh - m),
        WatermarkAnchor::BottomRight => (cw - ww - m, ch - wh - m),
        WatermarkAnchor::Normalized => {
            let cx = (spec.position[0] as f64 * cw as f64).round() as i64;
            let cy = (spec.position[1] as f64 * ch as f64).round() as i64;
            (cx - ww / 2, cy - wh / 2)
        }
    }
}

fn sample_rgba8_bilinear(src: &[u8], src_w: u32, src_h: u32, fx: f32, fy: f32) -> [f32; 4] {
    let max_x = (src_w - 1) as f32;
    let max_y = (src_h - 1) as f32;
    let x = fx.clamp(0.0, max_x);
    let y = fy.clamp(0.0, max_y);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(src_w - 1);
    let y1 = (y0 + 1).min(src_h - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;

    let fetch = |px: u32, py: u32| -> [f32; 4] {
        let i = ((py as usize) * (src_w as usize) + (px as usize)) * 4;
        [src[i] as f32, src[i + 1] as f32, src[i + 2] as f32, src[i + 3] as f32]
    };
    let c00 = fetch(x0, y0);
    let c10 = fetch(x1, y0);
    let c01 = fetch(x0, y1);
    let c11 = fetch(x1, y1);

    let mut out = [0.0f32; 4];
    for ch in 0..4 {
        let top = c00[ch] + (c10[ch] - c00[ch]) * tx;
        let bot = c01[ch] + (c11[ch] - c01[ch]) * tx;
        out[ch] = top + (bot - top) * ty;
    }
    out
}

/// Composites an 8-bit RGBA watermark onto an 8-bit RGBA canvas using Porter-Duff source-over
/// blending with anchor positioning, bilinear scaling, opacity control, and safe bounds clipping.
///
/// Returns `false` (leaving `canvas` byte-for-byte unchanged) when opacity is `<= 0`, scale is
/// `<= 0`, either buffer is empty/undersized, or the watermark falls completely outside the canvas.
pub fn composite_image_watermark_rgba8(
    canvas: &mut [u8],
    canvas_w: u32,
    canvas_h: u32,
    watermark: &[u8],
    wm_w: u32,
    wm_h: u32,
    spec: &ImageWatermark,
) -> bool {
    let Some(canvas_bytes) = (canvas_w as usize).checked_mul(canvas_h as usize).and_then(|n| n.checked_mul(4)) else {
        return false;
    };
    let Some(wm_bytes) = (wm_w as usize).checked_mul(wm_h as usize).and_then(|n| n.checked_mul(4)) else {
        return false;
    };
    if canvas_w == 0 || canvas_h == 0 || wm_w == 0 || wm_h == 0 {
        return false;
    }
    if canvas.len() < canvas_bytes || watermark.len() < wm_bytes {
        return false;
    }
    if !spec.opacity.is_finite() || spec.opacity <= 0.0 || !spec.scale.is_finite() || spec.scale <= 0.0 {
        return false;
    }

    let opacity = (spec.opacity / 100.0).clamp(0.0, 1.0);
    let scaled_w = ((wm_w as f32 * spec.scale).round() as u32).max(1);
    let scaled_h = ((wm_h as f32 * spec.scale).round() as u32).max(1);
    let (ox, oy) = watermark_origin(canvas_w, canvas_h, scaled_w, scaled_h, spec);

    let x_start = ox.max(0);
    let y_start = oy.max(0);
    let x_end = (ox + scaled_w as i64).min(canvas_w as i64);
    let y_end = (oy + scaled_h as i64).min(canvas_h as i64);
    if x_start >= x_end || y_start >= y_end {
        return false;
    }

    let exact_scale = scaled_w == wm_w && scaled_h == wm_h;
    let sx_ratio = wm_w as f32 / scaled_w as f32;
    let sy_ratio = wm_h as f32 / scaled_h as f32;

    for dy in y_start..y_end {
        let local_y = (dy - oy) as u32;
        for dx in x_start..x_end {
            let local_x = (dx - ox) as u32;
            let src = if exact_scale {
                let i = ((local_y as usize) * (wm_w as usize) + (local_x as usize)) * 4;
                [watermark[i] as f32, watermark[i + 1] as f32, watermark[i + 2] as f32, watermark[i + 3] as f32]
            } else {
                let fx = (local_x as f32 + 0.5) * sx_ratio - 0.5;
                let fy = (local_y as f32 + 0.5) * sy_ratio - 0.5;
                sample_rgba8_bilinear(watermark, wm_w, wm_h, fx, fy)
            };

            let src_a = (src[3] / 255.0) * opacity;
            if src_a <= 0.0 {
                continue;
            }
            let dst_idx = ((dy as usize) * (canvas_w as usize) + (dx as usize)) * 4;
            let dst_a = canvas[dst_idx + 3] as f32 / 255.0;
            let out_a = src_a + dst_a * (1.0 - src_a);
            if out_a <= 0.0 {
                continue;
            }
            for c in 0..3 {
                let dst_c = canvas[dst_idx + c] as f32;
                let blended = (src[c] * src_a + dst_c * dst_a * (1.0 - src_a)) / out_a;
                canvas[dst_idx + c] = blended.round().clamp(0.0, 255.0) as u8;
            }
            canvas[dst_idx + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    true
}

/// Validation failure for a recipe's `open_with` external application command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenWithError {
    /// The application string is empty or whitespace-only.
    EmptyApplication,
    /// The application string contains control characters (e.g. NUL, CR, LF).
    ControlCharacters,
    /// The application string contains shell metacharacters or unsafe syntax.
    UnsafeCharacters,
    /// The application path contains parent-directory traversal (`..`) or starts with `-`.
    UnsafePath,
    /// The exported file path is empty or has no file name.
    InvalidExportedFile,
}

impl std::fmt::Display for OpenWithError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyApplication => write!(f, "open_with: application path is empty"),
            Self::ControlCharacters => write!(f, "open_with: application contains control characters"),
            Self::UnsafeCharacters => write!(f, "open_with: application contains unsafe shell characters"),
            Self::UnsafePath => write!(f, "open_with: application path is unsafe"),
            Self::InvalidExportedFile => write!(f, "open_with: exported file path is invalid"),
        }
    }
}

impl std::error::Error for OpenWithError {}

/// A validated, shell-free external launch command for an exported image.
///
/// Construction validates the executable path and exported target without
/// spawning any external process. Callers must explicitly invoke
/// [`Self::spawn_explicit`] when the user opts into launching the app.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenWithCommand {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub exported_file: PathBuf,
}

impl OpenWithCommand {
    /// Builds a `std::process::Command` directly (without invoking a shell).
    pub fn to_std_command(&self) -> std::process::Command {
        let mut cmd = std::process::Command::new(&self.program);
        cmd.args(&self.args);
        cmd
    }

    /// Explicitly spawns the external application for an existing exported file.
    ///
    /// Export pipelines do not call this automatically; it is an opt-in entry
    /// point that verifies the exported file exists on disk before launching.
    pub fn spawn_explicit(&self) -> io::Result<std::process::Child> {
        if !self.exported_file.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "open_with: exported file does not exist",
            ));
        }
        self.to_std_command().spawn()
    }
}

/// Validates a recipe's `open_with` executable string and builds a safe
/// [`OpenWithCommand`] targeting `exported_file` without spawning a process.
pub fn validate_open_with(app: &str, exported_file: &Path) -> Result<OpenWithCommand, OpenWithError> {
    let trimmed = app.trim();
    if trimmed.is_empty() {
        return Err(OpenWithError::EmptyApplication);
    }
    if trimmed.chars().any(char::is_control) {
        return Err(OpenWithError::ControlCharacters);
    }
    if trimmed
        .chars()
        .any(|c| matches!(c, '|' | '&' | ';' | '`' | '$' | '<' | '>' | '*' | '?'))
    {
        return Err(OpenWithError::UnsafeCharacters);
    }

    let unquoted = trimmed
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(trimmed)
        .trim();
    if unquoted.is_empty() {
        return Err(OpenWithError::EmptyApplication);
    }
    if unquoted.contains('"') || unquoted.contains('\'') {
        return Err(OpenWithError::UnsafeCharacters);
    }
    if unquoted.starts_with('-') {
        return Err(OpenWithError::UnsafePath);
    }

    let program = PathBuf::from(unquoted);
    if program.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(OpenWithError::UnsafePath);
    }

    if exported_file.as_os_str().is_empty() || exported_file.file_name().is_none() {
        return Err(OpenWithError::InvalidExportedFile);
    }

    Ok(OpenWithCommand {
        program,
        args: vec![exported_file.to_string_lossy().into_owned()],
        exported_file: exported_file.to_path_buf(),
    })
}

/// Prepares the optional `open_with` command for a [`ProcessRecipe`] without executing it.
pub fn prepare_open_with(
    recipe: &ProcessRecipe,
    exported_file: &Path,
) -> Result<Option<OpenWithCommand>, OpenWithError> {
    let Some(app) = recipe.open_with.as_deref().map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    validate_open_with(app, exported_file).map(Some)
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

    // ---- export write path (BO-4) -------------------------------------------

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempFolder(PathBuf);

    impl TempFolder {
        fn new() -> Self {
            let id = COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir()
                .join(format!("aurora-export-atomic-{}-{id}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempFolder {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn resolve_output_dir_creates_and_canonicalizes_a_folder() {
        let temp = TempFolder::new();
        let nested = temp.path().join("a").join("b");
        let resolved = resolve_output_dir(&nested).unwrap();
        assert!(resolved.is_dir());
        assert_eq!(resolved, nested.canonicalize().unwrap());
    }

    #[test]
    fn resolve_output_dir_refuses_a_file() {
        let temp = TempFolder::new();
        let file = temp.path().join("not-a-folder");
        std::fs::write(&file, b"x").unwrap();
        let error = resolve_output_dir(&file).unwrap_err();
        assert_eq!(error.downcast_ref::<OutputPathRefusal>(), Some(&OutputPathRefusal::NotADirectory));
        assert_eq!(std::fs::read(&file).unwrap(), b"x", "the file is not touched");
    }

    #[test]
    fn contained_output_rejects_directory_traversal() {
        let dir = Path::new("C:/out");
        assert_eq!(contained_output(dir, Path::new("a.jpg")).unwrap(), dir.join("a.jpg"));
        for name in ["..", ".", "../evil.jpg", "a/b.jpg", "a\\b.jpg", "C:\\evil.jpg", "/etc/passwd"] {
            let error = contained_output(dir, Path::new(name)).unwrap_err();
            assert_eq!(
                error.downcast_ref::<OutputPathRefusal>(),
                Some(&OutputPathRefusal::EscapesDestination),
                "{name} must be refused"
            );
        }
    }

    #[test]
    fn sanitize_file_name_keeps_one_component() {
        assert_eq!(sanitize_file_name("DSC_0001.NEF"), Some("DSC_0001.NEF".into()));
        assert_eq!(sanitize_file_name("..\\..\\evil.exe"), Some(".._.._evil.exe".into()));
        assert_eq!(sanitize_file_name("a/b:c.jpg"), Some("a_b_c.jpg".into()));
        assert_eq!(sanitize_file_name("trailing. . "), Some("trailing".into()));
        assert_eq!(sanitize_file_name(".."), None);
        assert_eq!(sanitize_file_name("  "), None);
    }

    #[test]
    fn a_successful_write_syncs_then_renames() {
        let temp = TempFolder::new();
        let target = temp.path().join("out.jpg");
        std::fs::write(&target, b"old").unwrap();
        write_atomic_with(&target, "test", |file| file.write_all(b"new bytes")).unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"new bytes");
        let leftovers: Vec<_> = std::fs::read_dir(temp.path())
            .unwrap()
            .flatten()
            .filter(|entry| is_staging_path(&entry.path()))
            .collect();
        assert!(leftovers.is_empty(), "the same-directory temp must be gone");
    }

    #[test]
    fn a_full_disk_leaves_the_target_untouched_and_cleans_the_temp() {
        let temp = TempFolder::new();
        let target = temp.path().join("out.jpg");
        std::fs::write(&target, b"previous export").unwrap();

        // Windows ERROR_DISK_FULL; on other targets any write error is enough.
        let error = write_atomic_with(&target, "test", |file| {
            file.write_all(b"half a photo")?;
            Err(io::Error::from_raw_os_error(112))
        })
        .unwrap_err();
        assert_eq!(error.raw_os_error(), Some(112));
        assert_eq!(std::fs::read(&target).unwrap(), b"previous export");
        let leftovers: Vec<_> = std::fs::read_dir(temp.path())
            .unwrap()
            .flatten()
            .filter(|entry| is_staging_path(&entry.path()))
            .collect();
        assert!(leftovers.is_empty(), "the temp of a failed write must be removed");
    }

    #[test]
    fn cross_volume_publish_copies_into_the_target_volume() {
        let temp = TempFolder::new();
        let source = temp.path().join("elsewhere.tmp");
        let target = temp.path().join("out.jpg");
        std::fs::write(&source, b"bytes from another volume").unwrap();
        std::fs::write(&target, b"old").unwrap();

        // First rename reports ERROR_NOT_SAME_DEVICE / EXDEV, which is exactly
        // what a cross-volume temp looks like; the copy fallback must recover.
        let mut calls = 0;
        publish_using(&source, &target, &mut |from, to| {
            calls += 1;
            if calls == 1 {
                Err(io::Error::from_raw_os_error(if cfg!(windows) { 17 } else { 18 }))
            } else {
                std::fs::rename(from, to)
            }
        }, true)
        .unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"bytes from another volume");
        assert!(!source.exists(), "the cross-volume temp is consumed");
    }

    #[test]
    fn a_failed_cross_volume_publish_keeps_the_target() {
        let temp = TempFolder::new();
        let source = temp.path().join("elsewhere.tmp");
        let target = temp.path().join("out.jpg");
        std::fs::write(&target, b"old").unwrap();

        // The cross-volume source disappeared between write and publish: the
        // copy fallback fails and the existing target keeps its bytes.
        let result = publish_using(&source, &target, &mut |_, _| {
            Err(io::Error::from_raw_os_error(if cfg!(windows) { 17 } else { 18 }))
        }, false);
        assert!(result.is_err());
        assert_eq!(std::fs::read(&target).unwrap(), b"old");
    }

    #[test]
    fn a_cross_volume_fallback_never_replaces_an_existing_target() {
        let temp = TempFolder::new();
        let source = temp.path().join("elsewhere.tmp");
        let target = temp.path().join("out.jpg");
        std::fs::write(&source, b"new").unwrap();
        std::fs::write(&target, b"old").unwrap();

        let error = publish_using(
            &source,
            &target,
            &mut |_, _| Err(io::Error::from_raw_os_error(if cfg!(windows) { 17 } else { 18 })),
            false,
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&target).unwrap(), b"old");
        assert!(source.exists(), "the staged file is kept for a retry");
    }

    #[test]
    fn staged_export_publishes_to_a_free_name_and_cleans_up() {
        let temp = TempFolder::new();
        let out = temp.path().join("Output");
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(out.join("shot.jpg"), b"existing").unwrap();

        let staged = StagedExport::new(&out, 7).unwrap();
        assert!(staged.dir().starts_with(&out), "staging is on the target's volume");
        let written = staged.dir().join("shot.jpg");
        std::fs::write(&written, b"fresh").unwrap();
        let published = staged.publish(&written).unwrap();

        assert_eq!(published, out.join("shot-1.jpg"));
        assert_eq!(std::fs::read(&published).unwrap(), b"fresh");
        assert_eq!(std::fs::read(out.join("shot.jpg")).unwrap(), b"existing");
        let staging_left = std::fs::read_dir(&out)
            .unwrap()
            .flatten()
            .any(|entry| is_staging_path(&entry.path()));
        assert!(!staging_left, "the staging folder is removed after publish");
    }

    #[test]
    fn publish_never_clobbers_a_file_that_appeared_first() {
        let temp = TempFolder::new();
        let source = temp.path().join("staged.jpg");
        let target = temp.path().join("shot.jpg");
        std::fs::write(&source, b"new").unwrap();
        std::fs::write(&target, b"older").unwrap();

        let error = publish_no_clobber(&source, &target).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&target).unwrap(), b"older");
        assert_eq!(std::fs::read(&source).unwrap(), b"new", "the staged file is kept for a retry");
    }

    #[test]
    fn two_staged_exports_of_the_same_name_get_distinct_files() {
        let temp = TempFolder::new();
        let out = temp.path().join("Output");
        std::fs::create_dir_all(&out).unwrap();

        let first = StagedExport::new(&out, 1).unwrap();
        let second = StagedExport::new(&out, 2).unwrap();
        let first_file = first.dir().join("shot.jpg");
        let second_file = second.dir().join("shot.jpg");
        std::fs::write(&first_file, b"first").unwrap();
        std::fs::write(&second_file, b"second").unwrap();
        let a = first.publish(&first_file).unwrap();
        let b = second.publish(&second_file).unwrap();

        assert_ne!(a, b);
        let mut contents = vec![std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap()];
        contents.sort();
        assert_eq!(contents, vec![b"first".to_vec(), b"second".to_vec()]);
    }

    #[test]
    fn dropping_a_staged_export_removes_the_partial_output() {
        let temp = TempFolder::new();
        let out = temp.path().join("Output");
        std::fs::create_dir_all(&out).unwrap();
        let staging_path;
        {
            let staged = StagedExport::new(&out, 8).unwrap();
            staging_path = staged.dir().to_path_buf();
            std::fs::write(staging_path.join("shot.jpg"), b"partial").unwrap();
        }
        assert!(!staging_path.exists(), "a cancelled export leaves no staging folder");
        assert!(!out.join("shot.jpg").exists(), "nothing is published for a cancelled export");
    }

    #[test]
    fn stale_staging_is_identified_and_swept() {
        let temp = TempFolder::new();
        let stale = temp.path().join(format!("{STAGING_PREFIX}1-1{TEMP_SUFFIX}"));
        let fresh = temp.path().join(format!("{STAGING_PREFIX}1-2{TEMP_SUFFIX}"));
        std::fs::create_dir_all(&stale).unwrap();
        std::fs::create_dir_all(&fresh).unwrap();
        std::fs::write(temp.path().join("keep.jpg"), b"user file").unwrap();

        assert!(is_staging_path(&stale) && is_staging_path(&fresh));
        assert!(!is_staging_path(&temp.path().join("keep.jpg")));

        let cutoff = SystemTime::now() + Duration::from_secs(1);
        assert_eq!(sweep_stale_staging_before(temp.path(), cutoff), 2);
        assert!(!stale.exists() && !fresh.exists());
        assert!(temp.path().join("keep.jpg").exists(), "user files are never swept");
    }

    #[test]
    fn log_text_never_keeps_an_absolute_photo_path() {
        let cases = [
            ("render failed for C:\\Users\\alice\\Shoot\\DSC_0001.NEF", "render failed for <path>"),
            ("open \\\\nas\\photos\\DSC_0001.NEF failed", "open <path> failed"),
            ("open /home/alice/photos/DSC_0001.NEF failed", "open <path> failed"),
            ("no path here", "no path here"),
        ];
        for (input, want) in cases {
            assert_eq!(sanitize_log_text(input), want);
            assert!(!sanitize_log_text(input).contains("alice"));
            assert!(!sanitize_log_text(input).contains("DSC_0001"));
        }
        assert_eq!(log_name(Path::new("C:/Users/alice/Shoot/DSC_0001.NEF")), "DSC_0001.NEF");
        assert_eq!(log_name(Path::new("/tmp/out.jpg")), "out.jpg");
    }

    // ---- output sharpening, image watermark, open_with ----------------------

    #[test]
    fn output_sharpening_8bit_enhances_edge_and_preserves_alpha() {
        let (w, h) = (9u32, 9u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let v = if x < 4 { 60 } else { 200 };
                rgba[i] = v;
                rgba[i + 1] = v;
                rgba[i + 2] = v;
                rgba[i + 3] = 180;
            }
        }
        let before = rgba.clone();
        let mut recipe = ProcessRecipe::default();
        recipe.output_sharpening = OutputSharpening::Screen;
        assert!(apply_recipe_output_sharpening_rgba8(&mut rgba, w, h, &recipe));
        assert_ne!(rgba, before, "edge pixels must change after USM sharpening");
        // Pixel immediately to the left of the step edge should darken; right should brighten.
        let left_edge = ((4 * w + 3) * 4) as usize;
        let right_edge = ((4 * w + 4) * 4) as usize;
        assert!(rgba[left_edge] < before[left_edge], "dark side of edge should deepen");
        assert!(rgba[right_edge] > before[right_edge], "bright side of edge should lift");
        // Alpha channel must remain completely untouched.
        for px in rgba.chunks_exact(4) {
            assert_eq!(px[3], 180);
        }
    }

    #[test]
    fn output_sharpening_16bit_enhances_edge_and_preserves_alpha() {
        let (w, h) = (9u32, 9u32);
        let mut rgba = vec![0u16; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let v = if x < 4 { 15_000 } else { 50_000 };
                rgba[i] = v;
                rgba[i + 1] = v;
                rgba[i + 2] = v;
                rgba[i + 3] = 40_000;
            }
        }
        let before = rgba.clone();
        let mut recipe = ProcessRecipe::default();
        recipe.output_sharpening = OutputSharpening::Print;
        recipe.resolution_ppi = 300;
        assert!(apply_recipe_output_sharpening_rgba16(&mut rgba, w, h, &recipe));
        assert_ne!(rgba, before);
        for px in rgba.chunks_exact(4) {
            assert_eq!(px[3], 40_000);
        }
    }

    #[test]
    fn output_sharpening_disabled_or_zero_amount_is_idempotent() {
        let mut rgba = vec![10, 20, 30, 255, 200, 210, 220, 255];
        let before = rgba.clone();
        let recipe = ProcessRecipe::default(); // Disabled
        assert!(!apply_recipe_output_sharpening_rgba8(&mut rgba, 2, 1, &recipe));
        assert_eq!(rgba, before);
        assert!(!apply_output_sharpening_rgba8(
            &mut rgba,
            2,
            1,
            OutputSharpeningParams { amount: 0.0, radius: 1.0, threshold: 0.0 },
        ));
        assert_eq!(rgba, before);
    }

    #[test]
    fn output_sharpening_resolve_params_adapts_screen_and_print_dpi() {
        let screen = OutputSharpening::Screen.resolve_params(2000, 1333, 72, None).unwrap();
        let print_lo = OutputSharpening::Print.resolve_params(2000, 1333, 180, None).unwrap();
        let print_hi = OutputSharpening::Print.resolve_params(2000, 1333, 600, None).unwrap();
        assert!(print_hi.amount > screen.amount);
        assert!(print_hi.radius > print_lo.radius);
        // C1 control > 1 disables output sharpening even if amount > 0.
        let disabled_c1 = OutputSharpeningSettings { amount: 1.0, radius: 1.0, control: 4, ..Default::default() };
        assert!(OutputSharpening::Screen.resolve_params(2000, 1333, 300, Some(&disabled_c1)).is_none());
    }

    #[test]
    fn image_watermark_zero_opacity_is_byte_idempotent() {
        let mut canvas = vec![42u8; 8 * 8 * 4];
        let before = canvas.clone();
        let wm = vec![255u8; 4 * 4 * 4];
        let spec = ImageWatermark {
            anchor: WatermarkAnchor::Center,
            opacity: 0.0,
            scale: 1.0,
            margin_px: 0,
            ..Default::default()
        };
        assert!(!composite_image_watermark_rgba8(&mut canvas, 8, 8, &wm, 4, 4, &spec));
        assert_eq!(canvas, before);
    }

    #[test]
    fn image_watermark_alpha_blends_at_anchor_and_scaled() {
        let (cw, ch) = (10u32, 10u32);
        let mut canvas = vec![0u8; (cw * ch * 4) as usize];
        for px in canvas.chunks_exact_mut(4) {
            px.copy_from_slice(&[100, 100, 100, 255]);
        }
        // Solid red 2x2 watermark scaled 2x -> 4x4, placed at TopLeft with margin 1, opacity 50%.
        let mut wm = vec![0u8; 2 * 2 * 4];
        for px in wm.chunks_exact_mut(4) {
            px.copy_from_slice(&[200, 0, 0, 255]);
        }
        let spec = ImageWatermark {
            anchor: WatermarkAnchor::TopLeft,
            scale: 2.0,
            opacity: 50.0,
            margin_px: 1,
            ..Default::default()
        };
        assert!(composite_image_watermark_rgba8(&mut canvas, cw, ch, &wm, 2, 2, &spec));
        // Inside watermark rect [1..5, 1..5]: R = 200*0.5 + 100*0.5 = 150, G = 50, B = 50.
        let inside = ((2 * cw + 2) * 4) as usize;
        assert_eq!(&canvas[inside..inside + 4], &[150, 50, 50, 255]);
        // Outside watermark rect at (0, 0): untouched.
        assert_eq!(&canvas[0..4], &[100, 100, 100, 255]);
    }

    #[test]
    fn image_watermark_out_of_bounds_clips_safely() {
        let (cw, ch) = (6u32, 6u32);
        let mut canvas = vec![20u8; (cw * ch * 4) as usize];
        for px in canvas.chunks_exact_mut(4) {
            px[3] = 255;
        }
        let wm = vec![255u8; 4 * 4 * 4];

        // Partial clip on top-left corner via negative margin: only [0..2, 0..2] overlaps.
        let partial = ImageWatermark {
            anchor: WatermarkAnchor::TopLeft,
            scale: 1.0,
            opacity: 100.0,
            margin_px: -2,
            ..Default::default()
        };
        assert!(composite_image_watermark_rgba8(&mut canvas, cw, ch, &wm, 4, 4, &partial));
        assert_eq!(&canvas[0..4], &[255, 255, 255, 255]);
        let outside = ((3 * cw + 3) * 4) as usize;
        assert_eq!(&canvas[outside..outside + 4], &[20, 20, 20, 255]);

        // Completely outside canvas: returns false and does not modify canvas.
        let before = canvas.clone();
        let offscreen = ImageWatermark {
            anchor: WatermarkAnchor::Normalized,
            position: [5.0, 5.0],
            scale: 1.0,
            opacity: 100.0,
            margin_px: 0,
        };
        assert!(!composite_image_watermark_rgba8(&mut canvas, cw, ch, &wm, 4, 4, &offscreen));
        assert_eq!(canvas, before);
    }

    #[test]
    fn open_with_validates_and_builds_shell_free_command() {
        let exported = Path::new("C:/Exports/photo.jpg");
        let cmd = validate_open_with(r#""C:\Program Files\Viewer\viewer.exe""#, exported).unwrap();
        assert_eq!(cmd.program, PathBuf::from(r"C:\Program Files\Viewer\viewer.exe"));
        assert_eq!(cmd.args, vec!["C:/Exports/photo.jpg"]);
        let std_cmd = cmd.to_std_command();
        assert_eq!(std_cmd.get_program(), Path::new(r"C:\Program Files\Viewer\viewer.exe"));

        let mut recipe = ProcessRecipe::default();
        assert_eq!(prepare_open_with(&recipe, exported).unwrap(), None);
        recipe.open_with = Some("photoshop.exe".into());
        assert!(prepare_open_with(&recipe, exported).unwrap().is_some());
    }

    #[test]
    fn open_with_rejects_unsafe_or_empty_inputs() {
        let exported = Path::new("C:/Exports/photo.jpg");
        assert_eq!(validate_open_with("   ", exported), Err(OpenWithError::EmptyApplication));
        assert_eq!(validate_open_with("app.exe\nwhoami", exported), Err(OpenWithError::ControlCharacters));
        assert_eq!(validate_open_with("app.exe & calc.exe", exported), Err(OpenWithError::UnsafeCharacters));
        assert_eq!(validate_open_with("app.exe | tee", exported), Err(OpenWithError::UnsafeCharacters));
        assert_eq!(validate_open_with("../evil.exe", exported), Err(OpenWithError::UnsafePath));
        assert_eq!(validate_open_with("--flag", exported), Err(OpenWithError::UnsafePath));
        assert_eq!(validate_open_with("viewer.exe", Path::new("")), Err(OpenWithError::InvalidExportedFile));
    }

    #[test]
    fn open_with_spawn_explicit_refuses_missing_file() {
        let missing = std::env::temp_dir().join("aurora-does-not-exist-open-with.jpg");
        let cmd = validate_open_with("viewer.exe", &missing).unwrap();
        let err = cmd.spawn_explicit().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }
}
