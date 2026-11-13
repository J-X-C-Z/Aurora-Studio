//! Pure helpers for turning a [`ProcessRecipe`] into pixels, names and sizes.
//!
//! Nothing here touches the disk or the engine state: the exporter calls these,
//! then writes whatever path it is handed.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::catalog::ImageInfo;
use crate::icc::{self, Profile};
use crate::output::{
    ExportFormat, MetadataPolicy, OutputSharpening, OutputSharpeningSettings, ProcessRecipe, ScaleMode,
    UnsupportedFormat, Watermark,
};

/// 8x8 bitmap font for the watermark, covering ASCII 0x20..=0x7E.
///
/// Public domain (font8x8_basic, derived from the IBM VGA fonts via Marcel
/// Sondaar). One byte per row, bit `x` set means the pixel at column `x`.
const FONT8X8: [[u8; 8]; 95] = [
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00],
    [0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00],
    [0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00],
    [0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00],
    [0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00],
    [0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00],
    [0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00],
    [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00],
    [0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06],
    [0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00],
    [0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00],
    [0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00],
    [0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00],
    [0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00],
    [0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00],
    [0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00],
    [0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00],
    [0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00],
    [0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00],
    [0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00],
    [0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00],
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00],
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06],
    [0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00],
    [0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00],
    [0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00],
    [0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00],
    [0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00],
    [0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00],
    [0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00],
    [0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00],
    [0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00],
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00],
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00],
    [0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00],
    [0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00],
    [0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    [0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00],
    [0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00],
    [0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00],
    [0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00],
    [0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00],
    [0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00],
    [0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00],
    [0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00],
    [0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00],
    [0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00],
    [0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00],
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00],
    [0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00],
    [0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00],
    [0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00],
    [0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00],
    [0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00],
    [0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00],
    [0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00],
    [0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF],
    [0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00],
    [0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00],
    [0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00],
    [0x38, 0x30, 0x30, 0x3E, 0x33, 0x33, 0x6E, 0x00],
    [0x00, 0x00, 0x1E, 0x33, 0x3F, 0x03, 0x1E, 0x00],
    [0x1C, 0x36, 0x06, 0x0F, 0x06, 0x06, 0x0F, 0x00],
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F],
    [0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00],
    [0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    [0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E],
    [0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00],
    [0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    [0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00],
    [0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00],
    [0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00],
    [0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F],
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78],
    [0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00],
    [0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00],
    [0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00],
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00],
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00],
    [0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00],
    [0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00],
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F],
    [0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00],
    [0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00],
    [0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00],
    [0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00],
    [0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
];

/// Characters that must not survive into a file name.
const RESERVED: [char; 9] = ['/', '\\', ':', '*', '?', '"', '<', '>', '|'];

/// Pixel size of one export.
///
/// `Width`, `Height` and `Percent` are explicit requests and may enlarge the
/// image. `LongEdge`, `ShortEdge` and `Dimensions` are fit modes: they keep the
/// aspect ratio and never upscale.
pub fn output_size(recipe: &ProcessRecipe, src: (u32, u32)) -> (u32, u32) {
    let (sw, sh) = (src.0.max(1), src.1.max(1));
    let (a, b) = recipe.scale_value;
    let long = a.max(b);
    let short = if a > 0 && b > 0 { a.min(b) } else { long };

    match recipe.scale {
        ScaleMode::Fixed100 => (sw, sh),
        ScaleMode::Width if a > 0 => scaled(sw, sh, a as f32 / sw as f32),
        ScaleMode::Height if long > 0 => scaled(sw, sh, long as f32 / sh as f32),
        ScaleMode::LongEdge if long > 0 => {
            let edge = sw.max(sh);
            if long >= edge { (sw, sh) } else { scaled(sw, sh, long as f32 / edge as f32) }
        }
        ScaleMode::ShortEdge if short > 0 => {
            let edge = sw.min(sh);
            if short >= edge { (sw, sh) } else { scaled(sw, sh, short as f32 / edge as f32) }
        }
        ScaleMode::Dimensions if a > 0 && b > 0 => {
            let factor = (a as f32 / sw as f32).min(b as f32 / sh as f32);
            if factor >= 1.0 { (sw, sh) } else { scaled(sw, sh, factor) }
        }
        ScaleMode::Percent if long > 0 => scaled(sw, sh, long as f32 / 100.0),
        _ => (sw, sh),
    }
}

fn scaled(w: u32, h: u32, factor: f32) -> (u32, u32) {
    let one = |v: u32| ((v as f32 * factor).round() as i64).clamp(1, u32::MAX as i64) as u32;
    (one(w), one(h))
}

/// File name (no extension) for one exported image.
///
/// Placeholders: `{name}` source stem, `{seq}` zero-padded index, `{date}`
/// `YYYYMMDD` from the capture time, `{rating}`, `{variant}`, `{counter}`.
/// Capture One's bracketed spellings (`[Image Name]`, `[Counter]`, ...) work too.
/// An unknown placeholder or a path separator falls back to `{name}`.
pub fn output_name(recipe: &ProcessRecipe, info: &ImageInfo, index: usize) -> String {
    let fallback = source_stem(info);
    let template = c1_tokens(recipe.file_name_format.trim());
    if template.is_empty() {
        return fallback;
    }
    let mut out = String::with_capacity(template.len() + 16);
    let mut rest = template.as_str();
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        rest = &rest[open + 1..];
        let Some(close) = rest.find('}') else { return fallback };
        let token = rest[..close].trim();
        match token {
            "name" => out.push_str(&fallback),
            "seq" => out.push_str(&format!("{index:03}")),
            // short: caller supplies the counter, so a start value is not modelled here
            "counter" => out.push_str(&index.to_string()),
            "date" => out.push_str(&capture_date(info)),
            "rating" => out.push_str(&info.rating.to_string()),
            "variant" => out.push_str(&info.variant.to_string()),
            _ => return fallback,
        }
        rest = &rest[close + 1..];
    }
    if rest.contains('}') {
        return fallback;
    }
    out.push_str(rest);

    let cleaned = out.trim();
    let bad = cleaned.is_empty()
        || cleaned.contains("..")
        || cleaned.chars().any(|c| RESERVED.contains(&c) || c.is_control());
    if bad { fallback } else { cleaned.to_owned() }
}

fn source_stem(info: &ImageInfo) -> String {
    Path::new(&info.file_name)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("image-{}", info.id.0))
}

fn c1_tokens(template: &str) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(open) = rest.find('[') {
        out.push_str(&rest[..open]);
        rest = &rest[open + 1..];
        let Some(close) = rest.find(']') else {
            out.push('[');
            break;
        };
        let token = match rest[..close].trim().to_ascii_lowercase().as_str() {
            "image name" | "name" => "{name}",
            "counter" => "{counter}",
            "sequence" | "seq" => "{seq}",
            "date" => "{date}",
            "rating" => "{rating}",
            "variant" => "{variant}",
            _ => "[",
        };
        match token {
            "{" => {
                out.push('[');
                out.push_str(&rest[..close]);
                out.push(']');
            }
            _ => out.push_str(token),
        }
        rest = &rest[close + 1..];
    }
    out.push_str(rest);
    out
}

/// `YYYYMMDD` from the EXIF timestamp; `00000000` when there is none.
fn capture_date(info: &ImageInfo) -> String {
    let digits: String = info
        .exif
        .captured_at
        .as_deref()
        .unwrap_or_default()
        .chars()
        .take(10)
        .filter(char::is_ascii_digit)
        .collect();
    if digits.len() == 8 { digits } else { "00000000".into() }
}

/// `dir/name.ext`, or the first free `dir/name-1.ext`, `name-2`, ... .
///
/// Only computes the path; the caller does the writing.
pub fn collision_free(dir: &Path, name: &str, ext: &str) -> PathBuf {
    let ext = ext.trim_start_matches('.');
    let candidate = |suffix: &str| {
        if ext.is_empty() { dir.join(format!("{name}{suffix}")) } else { dir.join(format!("{name}{suffix}.{ext}")) }
    };
    let first = candidate("");
    if !first.exists() {
        return first;
    }
    for n in 1..=9999u32 {
        let path = candidate(&format!("-{n}"));
        if !path.exists() {
            return path;
        }
    }
    // Practically unreachable; a timestamp beats overwriting an existing file.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    candidate(&format!("-{stamp}"))
}

/// Unsharp mask for 8-bit RGBA output.
///
/// Blurs luminance-preserving per channel with a separable box pass, then adds
/// `amount * (original - blurred)`. Alpha is left alone. `amount <= 0` is a no-op.
pub fn output_sharpen(rgba: &mut [u8], w: u32, h: u32, amount: f32, radius: f32, threshold: f32) {
    let (wu, hu) = (w as usize, h as usize);
    if !(amount > 0.0) || !amount.is_finite() || wu == 0 || hu == 0 || rgba.len() < wu * hu * 4 {
        return;
    }
    let r = if radius.is_finite() && radius >= 1.0 { radius.round() as usize } else { 1 };
    let px = wu * hu;
    let mut horiz = vec![0u16; px * 3];
    let mut prefix = vec![0u32; wu + 1];

    for y in 0..hu {
        let row = &rgba[y * wu * 4..(y + 1) * wu * 4];
        for c in 0..3 {
            for x in 0..wu {
                prefix[x + 1] = prefix[x] + row[x * 4 + c] as u32;
            }
            for x in 0..wu {
                let lo = x.saturating_sub(r);
                let hi = (x + r + 1).min(wu);
                let count = (hi - lo) as u32;
                horiz[(y * wu + x) * 3 + c] = ((prefix[hi] - prefix[lo] + count / 2) / count) as u16;
            }
        }
    }

    let mut blur = vec![0u16; px * 3];
    let mut vprefix = vec![0u32; hu + 1];
    for x in 0..wu {
        for c in 0..3 {
            for y in 0..hu {
                vprefix[y + 1] = vprefix[y] + horiz[(y * wu + x) * 3 + c] as u32;
            }
            for y in 0..hu {
                let lo = y.saturating_sub(r);
                let hi = (y + r + 1).min(hu);
                let count = (hi - lo) as u32;
                blur[(y * wu + x) * 3 + c] = ((vprefix[hi] - vprefix[lo] + count / 2) / count) as u16;
            }
        }
    }

    let threshold = if threshold.is_finite() && threshold > 0.0 { threshold } else { 0.0 };

    use rayon::prelude::*;
    // Only `w * h` pixels belong to this frame; a caller may hand over a larger
    // buffer, and chunking that whole slice would make `y` exceed `h`.
    rgba[..wu * hu * 4].par_chunks_mut(wu * 4).enumerate().for_each(|(y, row)| {
        for x in 0..wu {
            for c in 0..3 {
                let i = x * 4 + c;
                let orig = row[i] as f32;
                let blurred = blur[(y * wu + x) * 3 + c] as f32;
                let detail = orig - blurred;
                if detail.abs() <= threshold {
                    continue;
                }
                row[i] = (orig + amount * detail).round().clamp(0.0, 255.0) as u8;
            }
        }
    });
}

/// Reads `SharpeningOutput*` from a Capture One recipe XML.
///
/// Unknown properties are ignored; missing fields are conservative zeroes.
pub fn parse_output_sharpening_xml(text: &str) -> OutputSharpeningSettings {
    let mut out = OutputSharpeningSettings::default();
    for (key, value) in crate::presets::parse_sl(text) {
        if !key.starts_with("SharpeningOutput") {
            continue;
        }
        let suffix = &key["SharpeningOutput".len()..];
        match suffix {
            "Amount" => out.amount = value.trim().parse().unwrap_or(0.0),
            "Radius" => out.radius = value.trim().parse().unwrap_or(0.0),
            "Threshold" => out.threshold = value.trim().parse().unwrap_or(0.0),
            "Control" => out.control = value.trim().parse().unwrap_or(0),
            "Method" => out.method = value.trim().parse().unwrap_or(0),
            "ViewingDistance" => out.viewing_distance = value.trim().parse().unwrap_or(0),
            "ViewingDistanceUnit" => out.viewing_distance_unit = value.trim().parse().unwrap_or(0),
            _ => {}
        }
    }
    out
}

/// Reads one Capture One recipe XML and keeps only the output-sharpening fields.
///
/// The returned recipe is synthetic and UI-only until imported into app state.
pub fn recipe_from_output_sharpening_xml(text: &str) -> ProcessRecipe {
    let settings = parse_output_sharpening_xml(text);
    let has_settings = text.contains("SharpeningOutput");
    ProcessRecipe {
        output_sharpening_settings: has_settings.then_some(settings),
        ..ProcessRecipe::default()
    }
}

/// Rough output size in bytes for queue progress and volume warnings.
///
/// Guesswork from format and pixel count; compression and content are ignored.
pub fn estimate_bytes(w: u32, h: u32, format: ExportFormat) -> u64 {
    let bytes_per_pixel = match format {
        ExportFormat::Jpeg => 0.6,
        ExportFormat::JpegXl => 0.35,
        ExportFormat::WebP => 0.5,
        ExportFormat::Heif => 0.5,
        ExportFormat::Png => 4.0,
        ExportFormat::Tiff => 6.0,
        ExportFormat::Psd => 8.0,
        ExportFormat::Dng => 3.0,
    };
    let pixels = w as f64 * h as f64;
    (pixels * bytes_per_pixel) as u64 + 16 * 1024
}

// ------------------------------------------------------------------- ICC

/// The output profile of a recipe, with the file it came from.
pub struct OutputProfile {
    pub profile: Arc<Profile>,
    pub path: PathBuf,
}

/// Finds the profile a recipe names.
///
/// Capture One names profiles by their file stem (`Adobe RGB (1998)`) or by
/// the description inside the file (`sRGB IEC61966-2.1`); both are accepted.
pub fn resolve_profile(name: &str) -> Option<OutputProfile> {
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    if let Some(profile) = icc::profile(name) {
        let path = profile.path.clone()?;
        return Some(OutputProfile { profile, path });
    }
    // Description match: only files whose stem shares a word with the
    // request are loaded, so this stays cheap with 1500 installed profiles.
    let tokens: Vec<String> = name
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 3)
        .map(|t| t.to_ascii_lowercase())
        .collect();
    if tokens.is_empty() {
        return None;
    }
    for (stem, path) in icc::scan_profiles(&[icc::profile_dir()]) {
        let stem_lower = stem.to_ascii_lowercase();
        if !tokens.iter().any(|t| stem_lower.contains(t.as_str())) {
            continue;
        }
        let Ok(profile) = icc::Profile::from_file(&path) else { continue };
        if profile.description.trim().eq_ignore_ascii_case(name) {
            return Some(OutputProfile { profile: Arc::new(profile), path });
        }
    }
    None
}

/// True when the profile is (close to) sRGB, so no conversion is needed.
fn is_srgb(profile: &Profile, path: &Path) -> bool {
    let desc = profile.description.to_ascii_lowercase();
    if desc.contains("srgb") {
        return true;
    }
    path.file_stem().map(|s| s.to_string_lossy().to_ascii_lowercase()).unwrap_or_default().contains("srgb")
}

/// Converts a rendered sRGB buffer into the recipe's profile.
///
/// The rendered buffer is gamma-encoded sRGB, so `convert_srgb_to` (which
/// linearizes, goes through XYZ D50 and encodes into the device space) is the
/// right direction. LUT profiles invert by nearest node, so the result is an
/// approximation, not an exact lcms match.
pub fn convert_to_profile(rgba: &mut [u8], profile: &Profile, path: &Path) {
    if is_srgb(profile, path) {
        return;
    }
    icc::convert_srgb_to(rgba, profile);
}

// --------------------------------------------------------------- metadata

/// The EXIF payload of a source file, as it would sit behind `Exif\0\0`.
///
/// JPEG sources are read verbatim from their APP1 segment. RAW, TIFF and PNG
/// sources carry their EXIF inside their own container, which is not an APP1
/// segment, so they yield nothing here.
pub fn source_exif(path: &Path) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    if !bytes.starts_with(&[0xFF, 0xD8]) {
        return None;
    }
    let mut i = 2usize;
    while i + 3 < bytes.len() {
        if bytes[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = bytes[i + 1];
        if marker == 0xD8 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            i += 2;
            continue;
        }
        if marker == 0xDA || marker == 0xD9 {
            break;
        }
        let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
        if len < 2 {
            break;
        }
        let start = i + 4;
        let end = start.checked_add(len - 2)?;
        if end > bytes.len() {
            break;
        }
        if marker == 0xE1 && &bytes[start..start + 6.min(end - start)] == b"Exif\0\0" {
            return Some(bytes[start + 6..end].to_vec());
        }
        i = end;
    }
    None
}

// ------------------------------------------------------- EXIF generation

/// One TIFF entry: a tag, its type and the raw bytes of its value.
struct TiffEntry {
    tag: u16,
    typ: u16,
    count: u32,
    value: Vec<u8>,
}

const TIFF_BYTE: u16 = 1;
const TIFF_ASCII: u16 = 2;
const TIFF_SHORT: u16 = 3;
const TIFF_LONG: u16 = 4;
const TIFF_RATIONAL: u16 = 5;

impl TiffEntry {
    fn ascii(tag: u16, text: &str) -> Self {
        let mut value = text.as_bytes().to_vec();
        value.push(0);
        Self { tag, typ: TIFF_ASCII, count: value.len() as u32, value }
    }

    fn short(tag: u16, value: u16) -> Self {
        Self { tag, typ: TIFF_SHORT, count: 1, value: value.to_le_bytes().to_vec() }
    }

    fn long(tag: u16, value: u32) -> Self {
        Self { tag, typ: TIFF_LONG, count: 1, value: value.to_le_bytes().to_vec() }
    }

    fn rational(tag: u16, numerator: u32, denominator: u32) -> Self {
        let mut value = numerator.to_le_bytes().to_vec();
        value.extend_from_slice(&denominator.to_le_bytes());
        Self { tag, typ: TIFF_RATIONAL, count: 1, value }
    }

    fn rationals(tag: u16, parts: &[(u32, u32)]) -> Self {
        let mut value = Vec::new();
        for (n, d) in parts {
            value.extend_from_slice(&n.to_le_bytes());
            value.extend_from_slice(&d.to_le_bytes());
        }
        Self { tag, typ: TIFF_RATIONAL, count: parts.len() as u32, value }
    }

    /// Bytes that live in the value field rather than in the data area.
    fn inline(&self) -> Option<[u8; 4]> {
        if self.value.len() > 4 {
            return None;
        }
        let mut out = [0u8; 4];
        out[..self.value.len()].copy_from_slice(&self.value);
        Some(out)
    }
}

/// Lays out a little-endian TIFF with an IFD0, an Exif sub-IFD and a GPS
/// sub-IFD, and returns the raw TIFF block (no `Exif\0\0` header).
fn build_tiff(ifd0: &[TiffEntry], exif_ifd: &[TiffEntry], gps_ifd: &[TiffEntry]) -> Vec<u8> {
    let ifd_size = |entries: &[TiffEntry]| 2 + entries.len() * 12 + 4;
    let ifd0_at = 8usize;
    let exif_at = ifd0_at + ifd_size(ifd0);
    let gps_at = exif_at + ifd_size(exif_ifd);
    let mut data_at = gps_at + ifd_size(gps_ifd);
    if data_at % 2 == 1 {
        data_at += 1;
    }

    // Values longer than four bytes live in a data area after the IFDs.
    let mut data = Vec::new();
    let mut offsets = Vec::new();
    for entry in ifd0.iter().chain(exif_ifd).chain(gps_ifd) {
        if entry.value.len() > 4 {
            offsets.push(data_at + data.len());
            data.extend_from_slice(&entry.value);
            if data.len() % 2 == 1 {
                data.push(0);
            }
        }
    }

    let write_ifd = |out: &mut Vec<u8>, entries: &[TiffEntry], next: u32, data_index: &mut usize| {
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        for entry in entries {
            out.extend_from_slice(&entry.tag.to_le_bytes());
            out.extend_from_slice(&entry.typ.to_le_bytes());
            out.extend_from_slice(&entry.count.to_le_bytes());
            match entry.inline() {
                Some(bytes) => out.extend_from_slice(&bytes),
                None => {
                    out.extend_from_slice(&(offsets[*data_index] as u32).to_le_bytes());
                    *data_index += 1;
                }
            }
        }
        out.extend_from_slice(&next.to_le_bytes());
    };

    let mut out = Vec::new();
    out.extend_from_slice(b"II");
    out.extend_from_slice(&42u16.to_le_bytes());
    out.extend_from_slice(&(ifd0_at as u32).to_le_bytes());
    let mut index = 0usize;
    write_ifd(&mut out, ifd0, 0, &mut index);
    write_ifd(&mut out, exif_ifd, 0, &mut index);
    write_ifd(&mut out, gps_ifd, 0, &mut index);
    while out.len() < data_at {
        out.push(0);
    }
    out.extend_from_slice(&data);
    out
}

/// Builds an EXIF block from the metadata the engine already parsed.
///
/// RAW and TIFF sources keep their EXIF inside their own container, so there
/// is no APP1 segment to copy; this writes the known fields instead. The
/// result is not a verbatim copy of the source.
pub fn build_exif(exif: &crate::Exif) -> Option<Vec<u8>> {
    let mut ifd0 = camera_tags(exif);
    let mut exif_ifd = capture_tags(exif);
    if let Some(stamp) = exif.captured_at.as_deref().and_then(exif_datetime) {
        ifd0.push(TiffEntry::ascii(0x0132, &stamp));
        exif_ifd.push(TiffEntry::ascii(0x9003, &stamp));
    }
    tiff_block(ifd0, exif_ifd, gps_tags(exif.gps))
}

/// Make and model: the two IFD0 tags that name the camera.
fn camera_tags(exif: &crate::Exif) -> Vec<TiffEntry> {
    let mut ifd0 = Vec::new();
    if let Some(make) = exif.camera_make.as_deref() {
        ifd0.push(TiffEntry::ascii(0x010F, make));
    }
    if let Some(model) = exif.camera_model.as_deref() {
        ifd0.push(TiffEntry::ascii(0x0110, model));
    }
    ifd0
}

/// Lens and exposure settings, all of which live in the Exif sub-IFD.
fn capture_tags(exif: &crate::Exif) -> Vec<TiffEntry> {
    let mut exif_ifd = Vec::new();
    if let Some(lens) = exif.lens.as_deref() {
        exif_ifd.push(TiffEntry::ascii(0xA434, lens));
    }
    if let Some(iso) = exif.iso {
        exif_ifd.push(TiffEntry::short(0x8827, iso.min(u16::MAX as u32) as u16));
    }
    if let Some(shutter) = exif.shutter {
        if shutter > 0.0 {
            exif_ifd.push(TiffEntry::rational(0x829A, (shutter * 1_000_000.0).round() as u32, 1_000_000));
        }
    }
    if let Some(aperture) = exif.aperture {
        if aperture > 0.0 {
            exif_ifd.push(TiffEntry::rational(0x829D, (aperture * 100.0).round() as u32, 100));
        }
    }
    if let Some(focal) = exif.focal_length_mm {
        if focal > 0.0 {
            exif_ifd.push(TiffEntry::rational(0x920A, (focal * 100.0).round() as u32, 100));
        }
    }
    exif_ifd
}

/// The four GPS IFD tags: latitude ref/value and longitude ref/value.
fn gps_tags(gps: Option<(f64, f64)>) -> Vec<TiffEntry> {
    let mut gps_ifd = Vec::new();
    if let Some((lat, lon)) = gps {
        let (lat_ref, lat_parts) = gps_parts(lat, "N", "S");
        let (lon_ref, lon_parts) = gps_parts(lon, "E", "W");
        gps_ifd.push(TiffEntry::ascii(0x0001, lat_ref));
        gps_ifd.push(TiffEntry::rationals(0x0002, &lat_parts));
        gps_ifd.push(TiffEntry::ascii(0x0003, lon_ref));
        gps_ifd.push(TiffEntry::rationals(0x0004, &lon_parts));
    }
    gps_ifd
}

/// Lays the IFDs out, adding the sub-IFD pointer IFD0 needs for a sub-IFD that
/// carries anything. `None` when nothing was recorded.
fn tiff_block(mut ifd0: Vec<TiffEntry>, exif_ifd: Vec<TiffEntry>, gps_ifd: Vec<TiffEntry>) -> Option<Vec<u8>> {
    if ifd0.is_empty() && exif_ifd.is_empty() && gps_ifd.is_empty() {
        return None;
    }
    // Exif wants every IFD in ascending tag order.
    ifd0.sort_by_key(|entry| entry.tag);
    // The pointers take part in IFD0's size, so they are reserved before the
    // offsets are computed and filled in afterwards.
    if !exif_ifd.is_empty() {
        ifd0.push(TiffEntry::long(0x8769, 0));
    }
    if !gps_ifd.is_empty() {
        ifd0.push(TiffEntry::long(0x8825, 0));
    }
    let ifd0_at = 8usize;
    let exif_at = ifd0_at + 2 + ifd0.len() * 12 + 4;
    let gps_at = exif_at + 2 + exif_ifd.len() * 12 + 4;
    for entry in ifd0.iter_mut() {
        match entry.tag {
            0x8769 => entry.value = (exif_at as u32).to_le_bytes().to_vec(),
            0x8825 => entry.value = (gps_at as u32).to_le_bytes().to_vec(),
            _ => {}
        }
    }
    let mut exif_ifd = exif_ifd;
    exif_ifd.sort_by_key(|entry| entry.tag);
    let mut gps_ifd = gps_ifd;
    gps_ifd.sort_by_key(|entry| entry.tag);
    Some(build_tiff(&ifd0, &exif_ifd, &gps_ifd))
}

/// The EXIF block a metadata group asks for.
///
/// `Exif` copies the source segment instead, so it never reaches here. Each
/// group replaces the whole block: what it does not name is not written.
/// Offset of the first IFD in a TIFF header.
fn tiff_ifd_at(tiff: &[u8]) -> Option<usize> {
    let b = tiff.get(4..8)?;
    Some(if tiff.starts_with(b"II") {
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    } else if tiff.starts_with(b"MM") {
        u32::from_be_bytes([b[0], b[1], b[2], b[3]])
    } else {
        return None;
    } as usize)
}

fn ifd_entries(exif: &[u8], ifd: usize) -> Vec<(u16, u16, u32, Vec<u8>)> {
    let le = exif.starts_with(b"II");
    let r16 = |at: usize| -> Option<u16> {
        let s = exif.get(at..at + 2)?;
        Some(if le { u16::from_le_bytes([s[0], s[1]]) } else { u16::from_be_bytes([s[0], s[1]]) })
    };
    let r32 = |at: usize| -> Option<u32> {
        let s = exif.get(at..at + 4)?;
        Some(if le {
            u32::from_le_bytes([s[0], s[1], s[2], s[3]])
        } else {
            u32::from_be_bytes([s[0], s[1], s[2], s[3]])
        })
    };
    let Some(count) = r16(ifd) else { return Vec::new() };
    let mut out = Vec::new();
    for i in 0..count as usize {
        let at = ifd + 2 + i * 12;
        let (Some(tag), Some(typ), Some(count)) = (r16(at), r16(at + 2), r32(at + 4)) else { continue };
        let unit = match typ {
            TIFF_BYTE => 1,
            TIFF_ASCII => 1,
            TIFF_SHORT => 2,
            TIFF_LONG => 4,
            TIFF_RATIONAL => 8,
            _ => continue,
        };
        let size = count as usize * unit;
        let value = if size <= 4 {
            exif.get(at + 8..at + 8 + size).map(<[u8]>::to_vec)
        } else {
            r32(at + 8).and_then(|off| exif.get(off as usize..off as usize + size)).map(<[u8]>::to_vec)
        };
        let Some(value) = value else { continue };
        out.push((tag, typ, count, value));
    }
    out
}

/// IFD0 entries of an EXIF block, values resolved to raw bytes. Sub-IFD
/// pointers are dropped: their targets would not survive the move into another
/// file.
fn ifd0_entries(exif: &[u8]) -> Vec<(u16, u16, u32, Vec<u8>)> {
    let Some(ifd) = tiff_ifd_at(exif) else { return Vec::new() };
    ifd_entries(exif, ifd)
        .into_iter()
        .filter(|(tag, ..)| !matches!(*tag, 0x8769 | 0x8825))
        .collect()
}

/// The Exif sub-IFD an EXIF block points at (tag 0x8769), resolved to bytes.
fn exif_ifd_entries(exif: &[u8]) -> Vec<(u16, u16, u32, Vec<u8>)> {
    let Some(ifd0) = tiff_ifd_at(exif) else { return Vec::new() };
    let Some((_, _, _, value)) = ifd_entries(exif, ifd0).into_iter().find(|e| e.0 == 0x8769) else {
        return Vec::new();
    };
    let Some(b) = value.get(..4) else { return Vec::new() };
    let at = if exif.starts_with(b"II") {
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    } else {
        u32::from_be_bytes([b[0], b[1], b[2], b[3]])
    } as usize;
    ifd_entries(exif, at)
}

/// The GPS sub-IFD an EXIF block points at (tag 0x8825), resolved to bytes.
fn gps_ifd_entries(exif: &[u8]) -> Vec<(u16, u16, u32, Vec<u8>)> {
    let Some(ifd0) = tiff_ifd_at(exif) else { return Vec::new() };
    let Some((_, _, _, value)) = ifd_entries(exif, ifd0).into_iter().find(|e| e.0 == 0x8825) else {
        return Vec::new();
    };
    let Some(b) = value.get(..4) else { return Vec::new() };
    let at = if exif.starts_with(b"II") {
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    } else {
        u32::from_be_bytes([b[0], b[1], b[2], b[3]])
    } as usize;
    ifd_entries(exif, at)
}

/// One 12-byte IFD entry, valid only when `value` matches the type and count.
fn make_entry(tag: u16, typ: u16, count: u32, value: &[u8], le: bool) -> Option<[u8; 12]> {
    let unit = match typ {
        TIFF_BYTE => 1,
        TIFF_ASCII => 1,
        TIFF_SHORT => 2,
        TIFF_LONG => 4,
        TIFF_RATIONAL => 8,
        _ => return None,
    };
    if value.len() != count as usize * unit {
        return None;
    }
    let mut entry = [0u8; 12];
    let u16b = |v: u16| if le { v.to_le_bytes() } else { v.to_be_bytes() };
    let u32b = |v: u32| if le { v.to_le_bytes() } else { v.to_be_bytes() };
    entry[0..2].copy_from_slice(&u16b(tag));
    entry[2..4].copy_from_slice(&u16b(typ));
    entry[4..8].copy_from_slice(&u32b(count));
    if value.len() <= 4 {
        entry[8..8 + value.len()].copy_from_slice(value);
    }
    Some(entry)
}

/// Serialises one IFD that starts at `ifd_at`: its header, its entries and its
/// out-of-line values, with every value offset made absolute.
fn pack_ifd(
    fields: &[(u16, u16, u32, Vec<u8>)],
    ifd_at: usize,
    le: bool,
) -> Option<(Vec<u8>, Vec<u8>)> {
    let u16b = |v: u16| if le { v.to_le_bytes() } else { v.to_be_bytes() };
    let u32b = |v: u32| if le { v.to_le_bytes() } else { v.to_be_bytes() };
    let mut entries: Vec<(u16, [u8; 12])> = Vec::with_capacity(fields.len());
    for (tag, typ, count, value) in fields {
        entries.push((*tag, make_entry(*tag, *typ, *count, value, le)?));
    }
    entries.sort_by_key(|(tag, _)| *tag);
    let data_at = ifd_at + 2 + entries.len() * 12 + 4;
    let mut data: Vec<u8> = Vec::new();
    for (tag, entry) in entries.iter_mut() {
        let value = fields.iter().find(|(t, ..)| t == tag)?.3.clone();
        if value.len() <= 4 {
            continue;
        }
        entry[8..12].copy_from_slice(&u32b((data_at + data.len()) as u32));
        data.extend_from_slice(&value);
        if data.len() % 2 == 1 {
            data.push(0);
        }
    }
    let mut blob = u16b(entries.len() as u16).to_vec();
    for (_, entry) in &entries {
        blob.extend_from_slice(entry);
    }
    blob.extend_from_slice(&u32b(0));
    Some((blob, data))
}

/// Rewrites the encoder's last IFD with `extra` tags appended and `exif_ifd`
/// / `gps_ifd` as its Exif and GPS sub-IFDs. The image crate's `TiffEncoder`
/// has no metadata API, but its output keeps the IFD at the end, so a new IFD
/// can point at the untouched pixel data. `None` when the layout is not the
/// one it writes.
fn tiff_add_tags(
    tiff: &[u8],
    extra: &[(u16, u16, u32, Vec<u8>)],
    exif_ifd: &[(u16, u16, u32, Vec<u8>)],
    gps_ifd: &[(u16, u16, u32, Vec<u8>)],
) -> Option<Vec<u8>> {
    let le = match tiff.get(..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16b = |v: u16| if le { v.to_le_bytes() } else { v.to_be_bytes() };
    let u32b = |v: u32| if le { v.to_le_bytes() } else { v.to_be_bytes() };
    let r16 = |b: &[u8], at: usize| -> Option<u16> {
        let s = b.get(at..at + 2)?;
        Some(if le { u16::from_le_bytes([s[0], s[1]]) } else { u16::from_be_bytes([s[0], s[1]]) })
    };
    let r32 = |b: &[u8], at: usize| -> Option<u32> {
        let s = b.get(at..at + 4)?;
        Some(if le {
            u32::from_le_bytes([s[0], s[1], s[2], s[3]])
        } else {
            u32::from_be_bytes([s[0], s[1], s[2], s[3]])
        })
    };
    let ifd = r32(tiff, 4)? as usize;
    let old_count = r16(tiff, ifd)? as usize;
    let mut entries: Vec<(u16, [u8; 12])> = Vec::with_capacity(old_count);
    for i in 0..old_count {
        let at = ifd + 2 + i * 12;
        let mut entry = [0u8; 12];
        entry.copy_from_slice(tiff.get(at..at + 12)?);
        entries.push((r16(tiff, at)?, entry));
    }
    let mut added: Vec<u16> = Vec::new();
    let mut added_values: Vec<(u16, Vec<u8>)> = Vec::new();
    for (tag, typ, count, value) in extra {
        if entries.iter().any(|(existing, _)| existing == tag) {
            continue;
        }
        entries.push((*tag, make_entry(*tag, *typ, *count, value, le)?));
        added.push(*tag);
        added_values.push((*tag, value.clone()));
    }
    if !exif_ifd.is_empty() && !entries.iter().any(|(tag, _)| *tag == 0x8769) {
        entries.push((0x8769, make_entry(0x8769, TIFF_LONG, 1, &[0; 4], le)?));
    }
    if !gps_ifd.is_empty() && !entries.iter().any(|(tag, _)| *tag == 0x8825) {
        entries.push((0x8825, make_entry(0x8825, TIFF_LONG, 1, &[0; 4], le)?));
    }
    entries.sort_by_key(|(tag, _)| *tag);
    let mut out = tiff.to_vec();
    if out.len() % 2 == 1 {
        out.push(0);
    }
    let new_ifd = out.len();
    let count = entries.len();
    let data_at = new_ifd + 2 + count * 12 + 4;
    let mut data: Vec<u8> = Vec::new();
    for (tag, entry) in entries.iter_mut() {
        if *tag == 0x8769 || *tag == 0x8825 {
            continue;
        }
        if !added.contains(tag) {
            continue;
        }
        let value = added_values.iter().find(|(t, _)| t == tag)?.1.clone();
        if value.len() <= 4 {
            continue;
        }
        entry[8..12].copy_from_slice(&u32b((data_at + data.len()) as u32));
        data.extend_from_slice(&value);
        if data.len() % 2 == 1 {
            data.push(0);
        }
    }
    let exif_at = data_at + data.len();
    let (exif_blob, exif_data) =
        if exif_ifd.is_empty() { (Vec::new(), Vec::new()) } else { pack_ifd(exif_ifd, exif_at, le)? };
    let gps_at = exif_at + exif_blob.len() + exif_data.len();
    let (gps_blob, gps_data) =
        if gps_ifd.is_empty() { (Vec::new(), Vec::new()) } else { pack_ifd(gps_ifd, gps_at, le)? };
    for (tag, entry) in entries.iter_mut() {
        if *tag == 0x8769 {
            entry[8..12].copy_from_slice(&u32b(exif_at as u32));
        }
        if *tag == 0x8825 {
            entry[8..12].copy_from_slice(&u32b(gps_at as u32));
        }
    }
    out[4..8].copy_from_slice(&u32b(new_ifd as u32));
    out.extend_from_slice(&u16b(count as u16));
    for (_, entry) in &entries {
        out.extend_from_slice(entry);
    }
    out.extend_from_slice(&u32b(0));
    out.extend_from_slice(&data);
    out.extend_from_slice(&exif_blob);
    out.extend_from_slice(&exif_data);
    out.extend_from_slice(&gps_blob);
    out.extend_from_slice(&gps_data);
    Some(out)
}

fn policy_exif(info: &ImageInfo, policy: MetadataPolicy) -> Option<Vec<u8>> {
    match policy {
        MetadataPolicy::Copyright => {
            let text = info.iptc.copyright.trim();
            // 0x8298 is a 7-bit ASCII tag; anything else goes to XMP dc:rights.
            if text.is_empty() || !text.is_ascii() {
                None
            } else {
                tiff_block(vec![TiffEntry::ascii(0x8298, text)], Vec::new(), Vec::new())
            }
        }
        // A "camera" group is the capture device and its exposure record:
        // make, model, lens, ISO, shutter, aperture, focal length. The capture
        // date is a separate group in Capture One, so it is left out.
        MetadataPolicy::Camera => tiff_block(camera_tags(&info.exif), capture_tags(&info.exif), Vec::new()),
        MetadataPolicy::Gps => tiff_block(Vec::new(), Vec::new(), gps_tags(info.exif.gps)),
        _ => None,
    }
}

/// The XMP packet a metadata group asks for; `None` when it has nothing to say.
fn policy_xmp(info: &ImageInfo, policy: MetadataPolicy) -> Option<String> {
    match policy {
        MetadataPolicy::Keywords if !info.keywords.is_empty() => Some(
            crate::XmpMeta { rating: 0, label: String::new(), keywords: info.keywords.clone() }.packet(),
        ),
        MetadataPolicy::RatingAndTag => Some(
            crate::XmpMeta { rating: info.rating, label: info.color_tag.label().into(), keywords: Vec::new() }
                .packet(),
        ),
        // A copyright EXIF cannot hold (non-ASCII) still has an XMP home.
        MetadataPolicy::Copyright => {
            let text = info.iptc.copyright.trim();
            (!text.is_empty() && !text.is_ascii()).then(|| rights_packet(text))
        }
        _ => None,
    }
}

/// `dc:rights` for the copyright strings EXIF's ASCII tag cannot carry.
fn rights_packet(text: &str) -> String {
    let text = text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
    format!(
        "<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n\
         <x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\n\
         \x20<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n\
         \x20 <rdf:Description rdf:about=\"\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\">\n\
         \x20  <dc:rights><rdf:Alt><rdf:li xml:lang=\"x-default\">{text}</rdf:li></rdf:Alt></dc:rights>\n\
         \x20 </rdf:Description>\n\
         \x20</rdf:RDF>\n\
         </x:xmpmeta>\n\
         <?xpacket end=\"w\"?>"
    )
}

/// `YYYY:MM:DD HH:MM:SS` from an RFC 3339 timestamp.
fn exif_datetime(raw: &str) -> Option<String> {
    let bytes = raw.as_bytes();
    if bytes.len() < 19 || bytes[4] != b'-' || bytes[7] != b'-' || bytes[10] != b'T' {
        return None;
    }
    Some(format!(
        "{}:{}:{} {}:{}:{}",
        &raw[0..4],
        &raw[5..7],
        &raw[8..10],
        &raw[11..13],
        &raw[14..16],
        &raw[17..19]
    ))
}

/// Degrees, minutes and seconds with a hundredth of a second resolution.
fn gps_parts(value: f64, positive: &'static str, negative: &'static str) -> (&'static str, [(u32, u32); 3]) {
    let (sign, abs) = if value < 0.0 { (negative, -value) } else { (positive, value) };
    let degrees = abs.floor();
    let minutes = (abs - degrees) * 60.0;
    let seconds = (minutes - minutes.floor()) * 60.0;
    (
        sign,
        [
            (degrees as u32, 1),
            (minutes.floor() as u32, 1),
            ((seconds * 100.0).round() as u32, 100),
        ],
    )
}

// --------------------------------------------------------------- watermark

/// Draws the watermark text at `Watermark::position`.
///
/// `position` is a normalized anchor for the centre of the text, clamped so
/// the block stays inside the frame. `opacity` is a percentage, `scale` a pixel
/// multiple of the 8x8 cell.
pub fn draw_watermark(rgba: &mut [u8], w: u32, h: u32, mark: &Watermark) {
    if mark.text.is_empty() || mark.opacity <= 0.0 {
        return;
    }
    let alpha = (mark.opacity.clamp(0.0, 100.0) / 100.0).clamp(0.0, 1.0);
    let scale = if mark.scale.is_finite() && mark.scale >= 1.0 {
        mark.scale.round() as u32
    } else {
        1
    };
    let chars: Vec<char> = mark.text.chars().collect();
    let cell = 8 * scale;
    let text_w = chars.len() as u32 * cell;
    let margin = 4u32 * scale;
    if text_w + margin * 2 > w || cell + margin * 2 > h {
        return;
    }
    let anchor_x = if mark.position[0].is_finite() { mark.position[0].clamp(0.0, 1.0) } else { 0.95 };
    let anchor_y = if mark.position[1].is_finite() { mark.position[1].clamp(0.0, 1.0) } else { 0.95 };
    let x0 = (anchor_x * w as f32 - text_w as f32 / 2.0).round() as u32;
    let y0 = (anchor_y * h as f32 - cell as f32 / 2.0).round() as u32;
    let x0 = x0.clamp(margin, w - margin - text_w);
    let y0 = y0.clamp(margin, h - margin - cell);
    for (index, ch) in chars.iter().enumerate() {
        let code = if ch.is_ascii() { *ch as usize } else { '?' as usize };
        let Some(glyph) = FONT8X8.get(code.saturating_sub(0x20)) else { continue };
        for row in 0..8u32 {
            for col in 0..8u32 {
                if glyph[row as usize] & (1 << col) == 0 {
                    continue;
                }
                for dy in 0..scale {
                    for dx in 0..scale {
                        let px = x0 + index as u32 * cell + col * scale + dx;
                        let py = y0 + row * scale + dy;
                        if px >= w || py >= h {
                            continue;
                        }
                        let at = ((py as usize) * (w as usize) + px as usize) * 4;
                        if at + 3 >= rgba.len() {
                            continue;
                        }
                        for c in 0..3 {
                            let base = rgba[at + c] as f32;
                            rgba[at + c] = (base + (255.0 - base) * alpha).round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
        }
    }
}

/// Writes one developed frame through `recipe`: scales it to the recipe's size,
/// applies the recipe's output sharpening, encodes the recipe's format, and
/// picks a name that does not overwrite an existing file.
///
/// `progress` receives 0..=1 while the bytes are written, against
/// [`estimate_bytes`]; the estimate is guesswork, so the last call is always 1.0.
pub fn write_export(
    dir: &Path,
    info: &ImageInfo,
    recipe: &ProcessRecipe,
    index: usize,
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    mut progress: impl FnMut(f32),
) -> anyhow::Result<PathBuf> {
    write_export_pixels(dir, info, recipe, index, Pixels::Rgba8(rgba), width, height, &mut progress)
}

/// [`write_export`] for a 16-bit frame. Only PNG and TIFF carry 16 bits;
/// any other format is an error rather than a silent 8-bit downgrade.
pub fn write_export16(
    dir: &Path,
    info: &ImageInfo,
    recipe: &ProcessRecipe,
    index: usize,
    rgba: Vec<u16>,
    width: u32,
    height: u32,
    mut progress: impl FnMut(f32),
) -> anyhow::Result<PathBuf> {
    write_export_pixels(dir, info, recipe, index, Pixels::Rgba16(rgba), width, height, &mut progress)
}

enum Pixels {
    Rgba8(Vec<u8>),
    Rgba16(Vec<u16>),
}

fn checked_len(width: u32, height: u32) -> anyhow::Result<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| anyhow::anyhow!("export: {width}x{height} overflows the pixel buffer"))
}

fn narrow_from_wide(rgba: &[u16]) -> Vec<u8> {
    rgba.iter().map(|&v| ((v as u32 * 255 + 32767) / 65535) as u8).collect()
}

#[allow(clippy::too_many_arguments)]
fn write_export_pixels(
    dir: &Path,
    info: &ImageInfo,
    recipe: &ProcessRecipe,
    index: usize,
    pixels: Pixels,
    width: u32,
    height: u32,
    progress: &mut dyn FnMut(f32),
) -> anyhow::Result<PathBuf> {
    use std::io::Write as _;

    // Before the directory or the file exists: an unsupported format must not
    // leave any file behind under the recipe's name.
    recipe.format.ensure_supported()?;
    std::fs::create_dir_all(dir)?;
    let (rgba, mut wide) = match pixels {
        Pixels::Rgba8(rgba) => {
            if rgba.len() != checked_len(width, height)? {
                anyhow::bail!("engine returned a buffer of the wrong size");
            }
            (rgba, None)
        }
        Pixels::Rgba16(rgba) => {
            if rgba.len() != checked_len(width, height)? {
                anyhow::bail!("engine returned a buffer of the wrong size");
            }
            let narrow = narrow_from_wide(&rgba);
            (narrow, Some(rgba))
        }
    };
    let (tw, th) = output_size(recipe, (width, height));
    let mut rgba = if let Some(wide) = wide.as_mut() {
        if (tw, th) != (width, height) {
            let img = image::ImageBuffer::<image::Rgba<u16>, _>::from_raw(width, height, std::mem::take(wide))
                .ok_or_else(|| anyhow::anyhow!("engine returned a buffer of the wrong size"))?;
            *wide = image::imageops::resize(&img, tw, th, image::imageops::FilterType::Triangle).into_raw();
        }
        narrow_from_wide(wide)
    } else if (tw, th) != (width, height) {
        let img = image::RgbaImage::from_raw(width, height, rgba)
            .ok_or_else(|| anyhow::anyhow!("engine returned a buffer of the wrong size"))?;
        image::imageops::resize(&img, tw, th, image::imageops::FilterType::Triangle).into_raw()
    } else {
        rgba
    };
    let develop8 = wide.is_some().then(|| rgba.clone());
    let amount = match recipe.output_sharpening {
        OutputSharpening::Disabled => 0.0,
        OutputSharpening::Screen => 0.6,
        OutputSharpening::Print => 1.0,
    };
    if let Some(settings) = recipe.output_sharpening_settings {
        output_sharpen(&mut rgba, tw, th, settings.amount, settings.radius, settings.threshold);
    } else {
        output_sharpen(&mut rgba, tw, th, amount, 1.0, 0.0);
    }
    // shortcut: Control/Method/viewing distance are saved but not applied; C1
    // semantics are unproven, revisit with real output pairs.

    // The rendered buffer is gamma-encoded sRGB. Converting it into the
    // recipe's profile is what makes the profile take effect.
    let profile = resolve_profile(&recipe.icc_profile);
    if profile.is_none() {
        log::warn!("export: ICC profile '{}' not found, writing untagged sRGB", recipe.icc_profile);
    }
    if let Some(target) = profile.as_ref() {
        convert_to_profile(&mut rgba, &target.profile, &target.path);
    }
    if let Some(mark) = recipe.watermark.as_ref() {
        draw_watermark(&mut rgba, tw, th, mark);
    }
    if let (Some(wide), Some(before)) = (wide.as_mut(), develop8.as_deref()) {
        for ((px, base), after) in
            wide.chunks_exact_mut(4).zip(before.chunks_exact(4)).zip(rgba.chunks_exact(4))
        {
            for k in 0..3 {
                px[k] = (px[k] as i32 + (after[k] as i32 - base[k] as i32) * 257)
                    .clamp(0, 65535) as u16;
            }
            px[3] = if after[3] == base[3] { px[3] } else { after[3] as u16 * 257 };
        }
    }

    let icc_bytes = profile.as_ref().and_then(|target| std::fs::read(&target.path).ok());
    let exif_bytes = match recipe.include_metadata {
        // JPEG sources keep their APP1 segment verbatim; RAW and TIFF sources
        // have no APP1 segment, so the fields the engine already parsed are
        // written into a fresh EXIF block instead.
        MetadataPolicy::Exif => {
            source_exif(Path::new(&info.path)).or_else(|| build_exif(&info.exif))
        }
        policy => policy_exif(info, policy),
    };
    let xmp_packet = policy_xmp(info, recipe.include_metadata);

    let (mut bytes, extension) = encode(
        &rgba,
        wide.as_deref(),
        tw,
        th,
        recipe,
        icc_bytes.as_deref(),
        exif_bytes.as_deref(),
        xmp_packet.as_deref().map(str::as_bytes),
    )?;
    // `image`'s TIFF encoder has no metadata API, so the EXIF tags go in by
    // rewriting its last IFD. Sub-IFDs (Exif, GPS) are not carried over.
    if extension == "tif" {
        let mut tags = exif_bytes.as_deref().map(ifd0_entries).unwrap_or_default();
        if let Some(packet) = xmp_packet.as_deref() {
            tags.push((700, TIFF_BYTE, packet.len() as u32, packet.as_bytes().to_vec()));
        }
        let exif_ifd = exif_bytes.as_deref().map(exif_ifd_entries).unwrap_or_default();
        let gps_ifd = exif_bytes.as_deref().map(gps_ifd_entries).unwrap_or_default();
        if !tags.is_empty() || !exif_ifd.is_empty() || !gps_ifd.is_empty() {
            match tiff_add_tags(&bytes, &tags, &exif_ifd, &gps_ifd) {
                Some(tagged) => bytes = tagged,
                None => log::warn!("export: could not add the EXIF tags to the TIFF"),
            }
        }
    }
    if let Some(packet) = xmp_packet {
        if extension != "tif" && extension != "heic" {
            embed_xmp(&mut bytes, packet.as_bytes());
        }
    }
    let path = collision_free(dir, &output_name(recipe, info, index), extension);
    let estimate = estimate_bytes(tw, th, recipe.format).max(1);
    let mut file = std::fs::File::create(&path)?;
    let mut written = 0u64;
    for chunk in bytes.chunks(256 * 1024) {
        file.write_all(chunk)?;
        written += chunk.len() as u64;
        progress((written as f32 / estimate as f32).min(1.0));
    }
    progress(1.0);
    Ok(path)
}

/// Encodes `rgba` in the recipe's format and returns the bytes and extension.
///
/// JpegXl, Dng and Psd have no encoder in the `image` crate and fail with
/// [`UnsupportedFormat`] rather than falling back to JPEG. Heif uses the
/// runtime libheif encoder when it is installed.
fn encode(
    rgba: &[u8],
    wide: Option<&[u16]>,
    w: u32,
    h: u32,
    recipe: &ProcessRecipe,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
) -> anyhow::Result<(Vec<u8>, &'static str)> {
    use image::ImageEncoder as _;

    // `TiffEncoder` seeks, so the sink is a cursor rather than a bare `Vec`.
    let mut out = std::io::Cursor::new(Vec::new());
    let color = image::ExtendedColorType::Rgba8;
    if let Some(wide) = wide {
        if wide.len() != checked_len(w, h)? {
            anyhow::bail!("export: 16-bit buffer of the wrong size");
        }
        let wide_color = image::ExtendedColorType::Rgba16;
        // The 16-bit encoders take bytes and expect native-endian samples.
        let mut bytes = Vec::with_capacity(wide.len() * 2);
        for v in wide {
            bytes.extend_from_slice(&v.to_ne_bytes());
        }
        match recipe.format {
            ExportFormat::Png => {
                let mut encoder = image::codecs::png::PngEncoder::new(&mut out);
                if let Some(icc) = icc {
                    encoder.set_icc_profile(icc.to_vec())?;
                }
                if let Some(exif) = exif {
                    encoder.set_exif_metadata(exif.to_vec())?;
                }
                encoder.write_image(&bytes, w, h, wide_color)?;
                return Ok((out.into_inner(), "png"));
            }
            ExportFormat::Tiff => {
                let mut encoder = image::codecs::tiff::TiffEncoder::new(&mut out);
                if let Some(icc) = icc {
                    encoder.set_icc_profile(icc.to_vec())?;
                }
                encoder.write_image(&bytes, w, h, wide_color)?;
                return Ok((out.into_inner(), "tif"));
            }
            other => anyhow::bail!(
                "export: {other:?} has no 16-bit encoder; use PNG or TIFF"
            ),
        }
    }
    let quality = recipe.quality.clamp(1, 100);
    match recipe.format {
        ExportFormat::Png => {
            {
                let mut encoder = image::codecs::png::PngEncoder::new(&mut out);
                if let Some(icc) = icc {
                    encoder.set_icc_profile(icc.to_vec())?;
                }
                if let Some(exif) = exif {
                    encoder.set_exif_metadata(exif.to_vec())?;
                }
                encoder.write_image(rgba, w, h, color)?;
            }
            Ok((out.into_inner(), "png"))
        }
        ExportFormat::Tiff => {
            let mut encoder = image::codecs::tiff::TiffEncoder::new(&mut out);
            if let Some(icc) = icc {
                encoder.set_icc_profile(icc.to_vec())?;
            }
            encoder.write_image(rgba, w, h, color)?;
            Ok((out.into_inner(), "tif"))
        }
        ExportFormat::WebP => {
            {
                let mut encoder = image::codecs::webp::WebPEncoder::new_lossless(&mut out);
                if let Some(icc) = icc {
                    encoder.set_icc_profile(icc.to_vec())?;
                }
                if let Some(exif) = exif {
                    encoder.set_exif_metadata(exif.to_vec())?;
                }
                encoder.write_image(rgba, w, h, color)?;
            }
            Ok((out.into_inner(), "webp"))
        }
        ExportFormat::Jpeg => {
            // JPEG has no alpha channel; drop it rather than fail the job.
            let rgb: Vec<u8> = rgba
                .chunks_exact(4)
                .flat_map(|px| [px[0], px[1], px[2]])
                .collect();
            {
                let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality);
                // The recipe's ppi becomes the JFIF density of the file.
                let ppi = recipe.resolution_ppi.clamp(1, u16::MAX as u32) as u16;
                encoder.set_pixel_density(image::codecs::jpeg::PixelDensity::dpi(ppi));
                if let Some(icc) = icc {
                    encoder.set_icc_profile(icc.to_vec())?;
                }
                if let Some(exif) = exif {
                    encoder.set_exif_metadata(exif.to_vec())?;
                }
                encoder.write_image(&rgb, w, h, image::ExtendedColorType::Rgb8)?;
            }
            Ok((out.into_inner(), "jpg"))
        }
        ExportFormat::Heif => Ok((
            crate::heif_write::encode_to_vec(rgba, w, h, quality, icc, exif, xmp)?,
            "heic",
        )),
        other => Err(UnsupportedFormat { format: other }.into()),
    }
}

/// The XMP signature that opens a JPEG APP1 XMP segment, NUL included.
const XMP_SIGNATURE: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";

/// Inserts `packet` where the still-image container keeps XMP: a second APP1
/// segment in JPEG, right after SOI, and an `XML:com.adobe.xmp` iTXt chunk in
/// PNG. Every existing segment and chunk is kept as it is. Formats with no
/// wired-up spot (TIFF, WebP) are left untouched.
fn embed_xmp(image: &mut Vec<u8>, packet: &[u8]) {
    if image.starts_with(&[0xFF, 0xD8]) {
        // The length field counts itself, so 0xFFFF is the ceiling.
        let len = packet.len() + XMP_SIGNATURE.len() + 2;
        if len > u16::MAX as usize {
            log::warn!("export: XMP packet too large for a JPEG APP1 segment, dropped");
            return;
        }
        let mut out = Vec::with_capacity(image.len() + len + 2);
        out.extend_from_slice(&image[..2]);
        out.extend_from_slice(&[0xFF, 0xE1]);
        out.extend_from_slice(&(len as u16).to_be_bytes());
        out.extend_from_slice(XMP_SIGNATURE);
        out.extend_from_slice(packet);
        out.extend_from_slice(&image[2..]);
        *image = out;
        return;
    }
    if !image.starts_with(b"\x89PNG\r\n\x1a\n") {
        return;
    }
    let mut data = b"XML:com.adobe.xmp\0".to_vec();
    data.extend_from_slice(&[0, 0]); // Not compressed, zlib would be method 0 too.
    data.extend_from_slice(b"\0\0"); // Empty language tag and translated keyword.
    data.extend_from_slice(packet);
    let mut chunk = (data.len() as u32).to_be_bytes().to_vec();
    chunk.extend_from_slice(b"iTXt");
    chunk.extend_from_slice(&data);
    let crc = crc32(&chunk[4..]);
    chunk.extend_from_slice(&crc.to_be_bytes());

    let mut at = 8usize;
    while at + 12 <= image.len() {
        let len = u32::from_be_bytes([image[at], image[at + 1], image[at + 2], image[at + 3]]) as usize;
        if &image[at + 4..at + 8] == b"IEND" {
            image.splice(at..at, chunk);
            return;
        }
        at += 12 + len;
    }
}

/// CRC-32 (IEEE 802.3), the checksum every PNG chunk ends with.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{ColorTag, Exif, FileKind, ImageId, Iptc};

    fn info() -> ImageInfo {
        ImageInfo {
            id: ImageId(7),
            path: PathBuf::from(r"C:\photos\DSC_0042.ARW"),
            file_name: "DSC_0042.ARW".into(),
            kind: FileKind::Raw,
            width: 6000,
            height: 4000,
            file_size: 0,
            exif: Exif { captured_at: Some("2026-11-10T08:30:00+08:00".into()), ..Exif::default() },
            iptc: Iptc::default(),
            keywords: Vec::new(),
            rating: 4,
            color_tag: ColorTag::None,
            variant: 2,
            has_adjustments: false,
            trashed: false,
            heif: None,
        }
    }

    fn recipe(scale: ScaleMode, value: (u32, u32)) -> ProcessRecipe {
        ProcessRecipe { scale, scale_value: value, ..ProcessRecipe::default() }
    }


    #[test]
    fn missing_or_bad_output_sharpening_fields_fall_back_conservatively() {
        let recipe = recipe_from_output_sharpening_xml("<OR><E K=\"Name\" V=\"x\" /></OR>");
        assert_eq!(recipe.output_sharpening_settings, None);
        assert_eq!(recipe.output_sharpening, OutputSharpening::Disabled);

        let recipe = recipe_from_output_sharpening_xml(
            r#"<OR><E K="SharpeningOutputAmount" V="not-a-number" /><E K="SharpeningOutputRadius" V="NaN" /></OR>"#,
        );
        let settings = recipe.output_sharpening_settings.unwrap();
        assert_eq!(settings.amount, 0.0);
        assert!(settings.radius.is_nan(), "invalid XML passes through for diagnostics");
        assert_eq!(settings.threshold, 0.0);
    }

    #[test]
    fn partial_settings_survive_json_round_trip() {
        let json = r#"{"output_sharpening":"Screen","output_sharpening_settings":{"amount":80.0}}"#;
        let recipe: ProcessRecipe = serde_json::from_str(json).unwrap();
        assert_eq!(recipe.output_sharpening, OutputSharpening::Screen);
        assert_eq!(recipe.output_sharpening_settings.as_ref().unwrap().amount, 80.0);
        assert_eq!(recipe.output_sharpening_settings.as_ref().unwrap().radius, 0.0);

        let old: ProcessRecipe = serde_json::from_str(r#"{"output_sharpening":"Screen"}"#).unwrap();
        assert_eq!(old.output_sharpening_settings, None);
    }

    #[test]
    fn long_edge_never_upscales() {
        let r = recipe(ScaleMode::LongEdge, (8000, 0));
        assert_eq!(output_size(&r, (6000, 4000)), (6000, 4000));
    }

    #[test]
    fn long_edge_keeps_aspect() {
        let r = recipe(ScaleMode::LongEdge, (3000, 0));
        assert_eq!(output_size(&r, (6000, 4000)), (3000, 2000));
        let portrait = recipe(ScaleMode::LongEdge, (0, 1000));
        assert_eq!(output_size(&portrait, (4000, 6000)), (667, 1000));
    }

    #[test]
    fn dimensions_fit_inside_box_without_upscaling() {
        let r = recipe(ScaleMode::Dimensions, (1000, 1000));
        assert_eq!(output_size(&r, (6000, 4000)), (1000, 667));
        let big = recipe(ScaleMode::Dimensions, (9000, 9000));
        assert_eq!(output_size(&big, (6000, 4000)), (6000, 4000));
    }

    #[test]
    fn width_height_percent_are_explicit() {
        assert_eq!(output_size(&recipe(ScaleMode::Width, (1200, 0)), (6000, 4000)), (1200, 800));
        assert_eq!(output_size(&recipe(ScaleMode::Height, (0, 900)), (6000, 4000)), (1350, 900));
        assert_eq!(output_size(&recipe(ScaleMode::Percent, (50, 0)), (6000, 4000)), (3000, 2000));
        assert_eq!(output_size(&recipe(ScaleMode::Fixed100, (10, 10)), (6000, 4000)), (6000, 4000));
    }

    #[test]
    fn name_expands_every_placeholder() {
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.file_name_format = "{name}_{date}_r{rating}_v{variant}_{seq}_{counter}".into();
        assert_eq!(output_name(&r, &info(), 5), "DSC_0042_20261110_r4_v2_005_5");
    }

    #[test]
    fn name_accepts_capture_one_brackets() {
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.file_name_format = "[Image Name]-[Counter]".into();
        assert_eq!(output_name(&r, &info(), 3), "DSC_0042-3");
    }

    #[test]
    fn invalid_template_falls_back_to_source_name() {
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        for bad in ["{bogus}", "{name", "a/b", "  ", ".."] {
            r.file_name_format = bad.into();
            assert_eq!(output_name(&r, &info(), 0), "DSC_0042", "{bad}");
        }
    }

    #[test]
    fn collision_appends_suffix() {
        let dir = std::env::temp_dir().join(format!(
            "lumen-recipe-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let first = collision_free(&dir, "shot", "jpg");
        assert_eq!(first, dir.join("shot.jpg"));
        std::fs::write(&first, b"x").unwrap();
        let second = collision_free(&dir, "shot", "jpg");
        assert_eq!(second, dir.join("shot-1.jpg"));
        std::fs::write(&second, b"x").unwrap();
        assert_eq!(collision_free(&dir, "shot", "jpg"), dir.join("shot-2.jpg"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sharpen_zero_is_byte_identical() {
        let mut px: Vec<u8> = (0..16 * 16 * 4).map(|i| (i * 7 % 256) as u8).collect();
        let before = px.clone();
        output_sharpen(&mut px, 16, 16, 0.0, 2.0, 0.0);
        assert_eq!(px, before);
    }

    #[test]
    fn sharpen_keeps_alpha_and_edges() {
        let (w, h) = (32u32, 32u32);
        let mut px = vec![128u8; (w * h * 4) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let v = if x < 16 { 40 } else { 210 };
                let i = (y * w as usize + x) * 4;
                px[i] = v;
                px[i + 3] = 255;
            }
        }
        let before = px.clone();
        output_sharpen(&mut px, w, h, 1.0, 2.0, 0.0);
        for i in (3..px.len()).step_by(4) {
            assert_eq!(px[i], 255);
        }
        let edge = (16 * 4) as usize;
        assert!(px[edge - 4] < before[edge - 4], "left of edge should darken");
        assert!(px[edge] > before[edge], "right of edge should brighten");
        assert_eq!(px[0], before[0], "flat area away from the edge is untouched");
    }

    #[test]
    fn sharpen_threshold_skips_small_detail() {
        let mut px = vec![0u8; 16 * 4];
        px[0] = 100;
        px[4] = 130;
        let before = px.clone();
        output_sharpen(&mut px, 4, 1, 1.0, 1.0, 100.0);
        assert_eq!(px, before);
    }

    #[test]
    fn estimate_grows_with_pixels_and_is_cheaper_for_jpeg() {
        let small = estimate_bytes(1000, 1000, ExportFormat::Jpeg);
        assert!(estimate_bytes(2000, 2000, ExportFormat::Jpeg) > small);
        assert!(estimate_bytes(4000, 3000, ExportFormat::Jpeg) < estimate_bytes(4000, 3000, ExportFormat::Tiff));
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lumen-export-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn write_export_uses_the_recipe_size_and_name() {
        let dir = temp_dir("size");
        let mut r = recipe(ScaleMode::LongEdge, (32, 0));
        r.file_name_format = "{name}_out_{seq}".into();
        let rgba = vec![128u8; 64 * 48 * 4];
        let path = write_export(&dir, &info(), &r, 2, rgba, 64, 48, |_| {}).unwrap();

        assert_eq!(path.file_name().unwrap().to_string_lossy(), "DSC_0042_out_002.jpg");
        assert_eq!(image::image_dimensions(&path).unwrap(), (32, 24));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn write_export_never_overwrites() {
        let dir = temp_dir("collision");
        let r = recipe(ScaleMode::Fixed100, (0, 0));
        let first = write_export(&dir, &info(), &r, 0, vec![10u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let second = write_export(&dir, &info(), &r, 0, vec![200u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();

        assert_eq!(first.file_name().unwrap().to_string_lossy(), "DSC_0042.jpg");
        assert_eq!(second.file_name().unwrap().to_string_lossy(), "DSC_0042-1.jpg");
        assert!(first.exists() && second.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn write_export_reports_progress_and_honours_the_format() {
        let dir = temp_dir("format");
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.format = ExportFormat::Png;
        let mut last = 0.0f32;
        let path = write_export(&dir, &info(), &r, 0, vec![64u8; 16 * 16 * 4], 16, 16, |p| last = p).unwrap();
        assert_eq!(path.extension().unwrap(), "png");
        assert_eq!(last, 1.0);
        assert_eq!(image::image_dimensions(&path).unwrap(), (16, 16));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn settings_amount_zero_is_byte_identical_and_amount_eighty_changes_pixels() {
        let mut zero = recipe(ScaleMode::Fixed100, (0, 0));
        zero.format = ExportFormat::Png;
        zero.output_sharpening_settings = Some(OutputSharpeningSettings { amount: 0.0, radius: 0.6, ..Default::default() });
        let mut eighty = zero.clone();
        eighty.output_sharpening_settings.as_mut().unwrap().amount = 80.0;
        let px: Vec<u8> = (0..16 * 16 * 4).map(|i| (i * 23 % 256) as u8).collect();

        let dir = temp_dir("sharp-zero");
        let path = write_export(&dir, &info(), &zero, 0, px.clone(), 16, 16, |_| {}).unwrap();
        let unchanged = image::open(&path).unwrap().to_rgba8().into_raw();
        std::fs::remove_dir_all(&dir).unwrap();

        let dir = temp_dir("sharp-eighty");
        let path = write_export(&dir, &info(), &eighty, 0, px.clone(), 16, 16, |_| {}).unwrap();
        let sharpened = image::open(&path).unwrap().to_rgba8().into_raw();
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(unchanged, px);
        assert_ne!(unchanged, sharpened);
    }

    #[test]
    fn radius_and_threshold_change_the_output() {
        let make = |radius: f32, threshold: f32| {
            let mut r = recipe(ScaleMode::Fixed100, (0, 0));
            r.format = ExportFormat::Png;
            r.output_sharpening_settings =
                Some(OutputSharpeningSettings { amount: 80.0, radius, threshold, ..Default::default() });
            r
        };
        let px: Vec<u8> = (0..32 * 32 * 4).map(|i| ((i / 4) * 13 % 256) as u8).collect();
        let mut outputs = Vec::new();
        for (radius, threshold) in [(0.6, 0.0), (3.0, 0.0), (0.6, 100.0)] {
            let dir = temp_dir("sharp-variant");
            let path = write_export(&dir, &info(), &make(radius, threshold), 0, px.clone(), 32, 32, |_| {}).unwrap();
            outputs.push(image::open(&path).unwrap().to_rgba8().into_raw());
            std::fs::remove_dir_all(&dir).unwrap();
        }
        assert_ne!(outputs[0], outputs[1]);
        assert_ne!(outputs[0], outputs[2]);
    }

    #[test]
    fn unsupported_formats_fail_and_write_nothing() {
        for format in [ExportFormat::JpegXl, ExportFormat::Dng, ExportFormat::Psd] {
            let base = temp_dir("unsupported");
            let dir = base.join("out");
            let mut r = recipe(ScaleMode::Fixed100, (0, 0));
            r.format = format;
            let err = write_export(&dir, &info(), &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap_err();

            let message = err.to_string();
            assert!(message.contains(format.label()), "{format:?}: {message}");
            assert!(message.contains("not implemented"), "{format:?}: {message}");
            assert!(!dir.exists(), "{format:?} left an output behind");
            std::fs::remove_dir_all(&base).unwrap();
        }
    }

    #[test]
    fn the_export_error_carries_the_format() {
        let err = ExportFormat::Dng.ensure_supported().unwrap_err();
        let unsupported = err.downcast_ref::<UnsupportedFormat>().expect("structured format error");
        assert_eq!(unsupported.format, ExportFormat::Dng);
    }

    #[test]
    fn supported_formats_keep_writing_their_own_extension() {
        let dir = temp_dir("supported");
        for (format, ext) in [
            (ExportFormat::Jpeg, "jpg"),
            (ExportFormat::Png, "png"),
            (ExportFormat::Tiff, "tif"),
            (ExportFormat::WebP, "webp"),
        ] {
            let mut r = recipe(ScaleMode::Fixed100, (0, 0));
            r.format = format;
            let path = write_export(&dir, &info(), &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
            assert_eq!(path.extension().unwrap(), ext, "{format:?}");
            assert_eq!(image::image_dimensions(&path).unwrap(), (8, 8), "{format:?}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn is_supported_matches_runtime_encoders() {
        for format in ExportFormat::ALL {
            assert_eq!(format.ensure_supported().is_ok(), format.is_supported(), "{format:?}");
        }
    }

    #[test]
    fn bit_depth_matrix_holds_for_every_format() {
        for format in ExportFormat::ALL {
            if !format.is_supported() {
                assert!(format.ensure_supported().is_err(), "{format:?}");
                continue;
            }
            assert!(format.supports_bit_depth(8).is_ok(), "{format:?}");
            assert_eq!(
                format.supports_bit_depth(16).is_ok(),
                matches!(format, ExportFormat::Png | ExportFormat::Tiff),
                "{format:?}"
            );
        }
    }

    // --------------------------------------------------------------- ICC

    #[cfg(feature = "external-fixtures")]
    #[test]
    fn icc_profile_changes_the_output() {
        let dir = temp_dir("icc");
        let mut srgb = recipe(ScaleMode::Fixed100, (0, 0));
        srgb.icc_profile = "sRGB IEC61966-2.1".into();
        let mut wide = recipe(ScaleMode::Fixed100, (0, 0));
        wide.icc_profile = "LeafLF2-LF2 Portrait 5".into();
        srgb.format = ExportFormat::Png;
        wide.format = ExportFormat::Png;
        let pixels: Vec<u8> = (0..16 * 16 * 4).map(|i| (i * 11 % 256) as u8).collect();

        let a = write_export(&dir, &info(), &srgb, 0, pixels.clone(), 16, 16, |_| {}).unwrap();
        let b = write_export(&dir, &info(), &wide, 0, pixels, 16, 16, |_| {}).unwrap();
        let a = image::open(a).unwrap().to_rgba8();
        let b = image::open(b).unwrap().to_rgba8();
        assert_ne!(a.as_raw(), b.as_raw(), "the recipe's ICC profile must change the pixels");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(feature = "external-fixtures")]
    #[test]
    fn the_icc_profile_is_embedded_in_the_file() {
        let dir = temp_dir("icc-embed");
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.icc_profile = "sRGB IEC61966-2.1".into();
        let path = write_export(&dir, &info(), &r, 0, vec![90u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();
        assert!(find(&bytes, b"ICC_PROFILE\0").is_some(), "the profile must be embedded");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_exported_icc_bytes_match_the_source_profile() {
        let Some(target) = resolve_profile("sRGB IEC61966-2.1") else { return };
        let source = std::fs::read(&target.path).unwrap();
        let dir = temp_dir("icc-bytes");
        for format in [ExportFormat::Jpeg, ExportFormat::Tiff] {
            let mut r = recipe(ScaleMode::Fixed100, (0, 0));
            r.icc_profile = "sRGB IEC61966-2.1".into();
            r.format = format;
            let path =
                write_export(&dir, &info(), &r, 0, vec![90u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
            let back = crate::file_engine::embedded_icc(&path).expect("ICC must read back");
            assert_eq!(back, source, "{format:?}: ICC bytes must survive verbatim");
            assert!(
                !crate::icc::parse_icc(&back).unwrap().description.is_empty(),
                "{format:?}: the description must still parse"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_16_bit_tiff_export_keeps_the_icc_bytes() {
        let Some(target) = resolve_profile("sRGB IEC61966-2.1") else { return };
        let source = std::fs::read(&target.path).unwrap();
        let dir = temp_dir("icc-bytes16");
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.icc_profile = "sRGB IEC61966-2.1".into();
        r.format = ExportFormat::Tiff;
        r.bit_depth = 16;
        let path =
            write_export16(&dir, &info(), &r, 0, vec![40000u16; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        if let Ok(copy) = std::env::var("LUMEN_ICC_ARTIFACT") {
            let _ = std::fs::copy(&path, copy);
        }
        let back = crate::file_engine::embedded_icc(&path).expect("ICC must read back");
        assert_eq!(back, source, "16-bit TIFF: ICC bytes must survive verbatim");
        assert!(!crate::icc::parse_icc(&back).unwrap().description.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn camera_make_and_model_are_written_into_a_tiff() {
        let dir = temp_dir("tiff-meta");
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.format = ExportFormat::Tiff;
        r.include_metadata = MetadataPolicy::Camera;
        let mut info = info();
        info.exif.camera_make = Some("Lumen".into());
        info.exif.camera_model = Some("Synthetic One".into());
        info.exif.lens = Some("Lumen 50mm".into());
        info.exif.iso = Some(400);
        let path = write_export(&dir, &info, &r, 0, vec![90u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        if let Ok(copy) = std::env::var("LUMEN_TIFF_ARTIFACT") {
            let _ = std::fs::copy(&path, copy);
        }
        let bytes = std::fs::read(&path).unwrap();
        assert!(find(&bytes, b"Lumen\0").is_some(), "make must be in the TIFF");
        assert!(find(&bytes, b"Synthetic One\0").is_some(), "model must be in the TIFF");
        assert!(find(&bytes, b"Lumen 50mm\0").is_some(), "lens must be in the Exif sub-IFD");

        assert!(image::open(&path).is_ok(), "the rewritten TIFF must still decode");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn gps_is_written_into_a_tiff() {
        let dir = temp_dir("tiff-gps");
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.format = ExportFormat::Tiff;
        r.include_metadata = MetadataPolicy::Gps;
        let mut info = info();
        info.exif.gps = Some((31.2304, 121.4737));
        let path = write_export(&dir, &info, &r, 0, vec![90u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        if let Ok(copy) = std::env::var("LUMEN_TIFF_GPS_ARTIFACT") {
            let _ = std::fs::copy(&path, copy);
        }
        let bytes = std::fs::read(&path).unwrap();
        let gps = gps_ifd_entries(&bytes);
        assert!(gps.iter().any(|(tag, ..)| *tag == 1), "GPSLatitudeRef must be in the GPS IFD");
        assert!(gps.iter().any(|(tag, ..)| *tag == 2), "GPSLatitude must be in the GPS IFD");
        assert!(image::open(&path).is_ok(), "the rewritten TIFF must still decode");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_16_bit_tiff_carries_the_metadata_and_icc() {
        let Some(target) = resolve_profile("sRGB IEC61966-2.1") else { return };
        let source = std::fs::read(&target.path).unwrap();
        let dir = temp_dir("tiff16-meta");

        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.format = ExportFormat::Tiff;
        r.bit_depth = 16;
        r.icc_profile = "sRGB IEC61966-2.1".into();
        r.include_metadata = MetadataPolicy::Camera;
        let mut sample = info();
        sample.exif.camera_make = Some("Lumen".into());
        sample.exif.camera_model = Some("Synthetic One".into());
        sample.exif.lens = Some("Lumen 50mm".into());
        sample.exif.iso = Some(400);
        let path =
            write_export16(&dir, &sample, &r, 0, vec![40000u16; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        if let Ok(copy) = std::env::var("LUMEN_TIFF16_ARTIFACT") {
            let _ = std::fs::copy(&path, copy);
        }
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            crate::file_engine::embedded_icc(&path).unwrap(),
            source,
            "16-bit TIFF must keep the ICC bytes"
        );
        let ifd0 = ifd_entries(&bytes, tiff_ifd_at(&bytes).unwrap());
        assert!(
            ifd0.iter().any(|(tag, .., value)| *tag == 0x010F && find(value, b"Lumen").is_some()),
            "Make must be in the 16-bit IFD0"
        );
        assert!(ifd0.iter().any(|(tag, ..)| *tag == 0x0110), "Model must be in the 16-bit IFD0");
        let sub = exif_ifd_entries(&bytes);
        assert!(sub.iter().any(|(tag, ..)| *tag == 0xA434), "LensModel must be in the Exif IFD");
        assert!(sub.iter().any(|(tag, ..)| *tag == 0x8827), "ISO must be in the Exif IFD");
        assert_eq!(
            image::open(&path).unwrap().to_rgba16().as_raw()[0],
            40000,
            "the tags must not disturb the 16-bit samples"
        );

        let mut xmp = r.clone();
        xmp.include_metadata = MetadataPolicy::Keywords;
        let mut tagged = info();
        tagged.keywords = vec!["synthetic".into()];
        let path =
            write_export16(&dir, &tagged, &xmp, 0, vec![40000u16; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        if let Ok(copy) = std::env::var("LUMEN_TIFF16_XMP_ARTIFACT") {
            let _ = std::fs::copy(&path, copy);
        }
        let bytes = std::fs::read(&path).unwrap();
        let packet = ifd_entries(&bytes, tiff_ifd_at(&bytes).unwrap())
            .into_iter()
            .find(|entry| entry.0 == 700)
            .expect("XMP must be in tag 700");
        assert!(find(&packet.3, b"dc:subject").is_some(), "16-bit TIFF must carry the keywords XMP");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn xmp_is_written_into_a_tiff() {
        let dir = temp_dir("tiff-xmp");
        let mut keywords = recipe(ScaleMode::Fixed100, (0, 0));
        keywords.format = ExportFormat::Tiff;
        keywords.include_metadata = MetadataPolicy::Keywords;
        let mut sample = info();
        sample.keywords = vec!["synthetic".into()];
        let path =
            write_export(&dir, &sample, &keywords, 0, vec![90u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        if let Ok(copy) = std::env::var("LUMEN_TIFF_XMP_ARTIFACT") {
            let _ = std::fs::copy(&path, copy);
        }
        let read_packet = |bytes: &[u8]| {
            let ifd = tiff_ifd_at(bytes).unwrap();
            ifd_entries(bytes, ifd).into_iter().find(|entry| entry.0 == 700).map(|entry| entry.3)
        };
        let bytes = std::fs::read(&path).unwrap();
        let packet = read_packet(&bytes).expect("XMP must be in tag 700");
        assert!(find(&packet, b"dc:subject").is_some(), "the keywords packet must name dc:subject");
        assert!(image::open(&path).is_ok(), "the rewritten TIFF must still decode");

        let mut rated = recipe(ScaleMode::Fixed100, (0, 0));
        rated.format = ExportFormat::Tiff;
        rated.include_metadata = MetadataPolicy::RatingAndTag;
        let plain = info();
        let path =
            write_export(&dir, &plain, &rated, 0, vec![90u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        if let Ok(copy) = std::env::var("LUMEN_TIFF_RATING_ARTIFACT") {
            let _ = std::fs::copy(&path, copy);
        }
        let packet = read_packet(&std::fs::read(&path).unwrap()).expect("rating XMP must be in tag 700");
        assert!(find(&packet, b"xmp:Rating").is_some(), "the rating packet must name xmp:Rating");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_unknown_icc_profile_is_reported_but_still_exports() {
        let dir = temp_dir("icc-missing");
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.icc_profile = "No Such Profile".into();
        let path = write_export(&dir, &info(), &r, 0, vec![90u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        assert!(path.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // ----------------------------------------------------------- metadata

    /// A JPEG file with one APP1 EXIF segment carrying `payload`.
    fn jpeg_with_exif(payload: &[u8]) -> Vec<u8> {
        let mut out = vec![0xFF, 0xD8];
        let len = (payload.len() + 8) as u16;
        out.extend_from_slice(&[0xFF, 0xE1]);
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(b"Exif\0\0");
        out.extend_from_slice(payload);
        out.extend_from_slice(&[0xFF, 0xD9]);
        out
    }

    #[test]
    fn exif_is_preserved_verbatim() {
        let dir = temp_dir("exif-keep");
        let src = dir.join("source.jpg");
        let payload = b"MM\0*\0\0\0\x08\0\0";
        std::fs::write(&src, jpeg_with_exif(payload)).unwrap();
        let mut info = info();
        info.path = src.clone();
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.include_metadata = MetadataPolicy::Exif;
        let path = write_export(&dir, &info, &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();
        let at = find(&bytes, b"Exif\0\0").expect("the EXIF segment must survive");
        assert_eq!(&bytes[at + 6..at + 6 + payload.len()], payload, "the EXIF payload must be copied verbatim");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn strip_all_removes_the_exif_segment() {
        let dir = temp_dir("exif-strip");
        let src = dir.join("source.jpg");
        std::fs::write(&src, jpeg_with_exif(b"MM\0*\0\0\0\x08\0\0")).unwrap();
        let mut info = info();
        info.path = src;
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.include_metadata = MetadataPolicy::StripAll;
        let path = write_export(&dir, &info, &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();
        assert!(find(&bytes, b"Exif\0\0").is_none(), "strip all must not write EXIF");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Every marker segment of a JPEG as (marker, payload), up to the scan.
    fn segments(jpeg: &[u8]) -> Vec<(u8, &[u8])> {
        let mut out = Vec::new();
        let mut i = 2usize;
        while i + 4 <= jpeg.len() && jpeg[i] == 0xFF {
            let marker = jpeg[i + 1];
            if marker == 0xDA || marker == 0xD9 {
                break;
            }
            let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
            out.push((marker, &jpeg[i + 4..i + 2 + len]));
            i += 2 + len;
        }
        out
    }

    /// The APP1 payload behind `signature`, when the file has such a segment.
    fn app1<'a>(jpeg: &'a [u8], signature: &[u8]) -> Option<&'a [u8]> {
        segments(jpeg).into_iter().filter(|(marker, _)| *marker == 0xE1).find_map(|(_, payload)| payload.strip_prefix(signature))
    }

    /// (tag, type, count) of a little-endian TIFF's IFD at `at`.
    fn entries(tiff: &[u8], at: usize) -> Vec<(u16, u16, u32)> {
        let count = u16::from_le_bytes(tiff[at..at + 2].try_into().unwrap()) as usize;
        (0..count)
            .map(|i| {
                let entry = &tiff[at + 2 + i * 12..];
                (
                    u16::from_le_bytes(entry[0..2].try_into().unwrap()),
                    u16::from_le_bytes(entry[2..4].try_into().unwrap()),
                    u32::from_le_bytes(entry[4..8].try_into().unwrap()),
                )
            })
            .collect()
    }

    /// The offset an IFD0 entry stores as its value, for the sub-IFD pointers.
    fn pointer(tiff: &[u8], tag: u16) -> usize {
        let count = u16::from_le_bytes(tiff[10..12].try_into().unwrap()) as usize;
        let entry = (0..count)
            .map(|i| &tiff[10 + i * 12..])
            .find(|entry| u16::from_le_bytes(entry[0..2].try_into().unwrap()) == tag)
            .expect("the pointer tag must be present");
        u32::from_le_bytes(entry[8..12].try_into().unwrap()) as usize
    }

    fn tags(tiff: &[u8], at: usize) -> Vec<u16> {
        entries(tiff, at).iter().map(|entry| entry.0).collect()
    }

    /// (type, count, value bytes) of `tag` in the little-endian IFD at `at`.
    fn field<'a>(tiff: &'a [u8], at: usize, tag: u16) -> (u16, u32, &'a [u8]) {
        let count = u16::from_le_bytes(tiff[at..at + 2].try_into().unwrap()) as usize;
        let entry = (0..count)
            .map(|i| &tiff[at + 2 + i * 12..at + 14 + i * 12])
            .find(|entry| u16::from_le_bytes(entry[0..2].try_into().unwrap()) == tag)
            .expect("the tag must be present");
        let typ = u16::from_le_bytes(entry[2..4].try_into().unwrap());
        let count = u32::from_le_bytes(entry[4..8].try_into().unwrap());
        let size = match typ {
            TIFF_ASCII => 1,
            TIFF_SHORT => 2,
            TIFF_LONG => 4,
            TIFF_RATIONAL => 8,
            other => panic!("unexpected TIFF type {other}"),
        };
        let len = count as usize * size;
        let data = if len <= 4 {
            &entry[8..8 + len]
        } else {
            let off = u32::from_le_bytes(entry[8..12].try_into().unwrap()) as usize;
            &tiff[off..off + len]
        };
        (typ, count, data)
    }

    fn ascii_field(tiff: &[u8], at: usize, tag: u16) -> String {
        let (typ, _, data) = field(tiff, at, tag);
        assert_eq!(typ, TIFF_ASCII);
        String::from_utf8_lossy(data.strip_suffix(&[0]).unwrap_or(data)).to_string()
    }

    fn short_field(tiff: &[u8], at: usize, tag: u16) -> u16 {
        let (typ, _, data) = field(tiff, at, tag);
        assert_eq!(typ, TIFF_SHORT);
        u16::from_le_bytes(data[0..2].try_into().unwrap())
    }

    fn rationals_field(tiff: &[u8], at: usize, tag: u16) -> Vec<f64> {
        let (typ, count, data) = field(tiff, at, tag);
        assert_eq!(typ, TIFF_RATIONAL);
        (0..count as usize)
            .map(|i| {
                let n = u32::from_le_bytes(data[i * 8..i * 8 + 4].try_into().unwrap()) as f64;
                let d = u32::from_le_bytes(data[i * 8 + 4..i * 8 + 8].try_into().unwrap()) as f64;
                n / d
            })
            .collect()
    }

    /// The TIFF block of a PNG's `eXIf` chunk.
    fn png_exif(png: &[u8]) -> &[u8] {
        let at = find(png, b"eXIf").expect("the PNG must carry an eXIf chunk");
        &png[at + 4..]
    }

    /// The XMP packet carried by a JPEG APP1 or PNG iTXt chunk.
    fn packet_text(bytes: &[u8]) -> String {
        let start = find(bytes, b"<?xpacket").expect("the file must carry an XMP packet");
        let tail = b"<?xpacket end=\"w\"?>";
        let end = start + find(&bytes[start..], tail).unwrap() + tail.len();
        std::str::from_utf8(&bytes[start..end]).unwrap().to_string()
    }

    fn dc_rights(bytes: &[u8]) -> String {
        let text = packet_text(bytes);
        let doc = roxmltree::Document::parse(&text).unwrap();
        let dc = "http://purl.org/dc/elements/1.1/";
        let rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
        doc.descendants()
            .find(|node| node.has_tag_name((dc, "rights")))
            .and_then(|rights| rights.descendants().find(|node| node.has_tag_name((rdf, "li"))))
            .and_then(|node| node.text())
            .expect("dc:rights must carry a value")
            .to_string()
    }

    /// The XMP packet of a JPEG, as text.
    fn xmp_text(jpeg: &[u8]) -> String {
        let packet = app1(jpeg, XMP_SIGNATURE).expect("the file must carry an XMP APP1 segment");
        String::from_utf8(packet.to_vec()).unwrap()
    }

    #[test]
    fn every_metadata_policy_exports() {
        let dir = temp_dir("meta-all");
        let mut info = info();
        info.iptc.copyright = "© 2026 Ada".into();
        info.keywords = vec!["wedding".into()];
        info.exif = Exif {
            camera_make: Some("Sony".into()),
            camera_model: Some("ILCE-7M4".into()),
            gps: Some((31.2397, 121.4998)),
            ..Exif::default()
        };
        for policy in [
            MetadataPolicy::Exif,
            MetadataPolicy::StripAll,
            MetadataPolicy::Copyright,
            MetadataPolicy::Camera,
            MetadataPolicy::Gps,
            MetadataPolicy::Keywords,
            MetadataPolicy::RatingAndTag,
        ] {
            let mut r = recipe(ScaleMode::Fixed100, (0, 0));
            r.include_metadata = policy;
            let path = write_export(&dir, &info, &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
            assert!(path.exists(), "{} must export", policy.label());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn copyright_writes_only_the_ifd0_copyright_tag() {
        let dir = temp_dir("meta-copyright");
        let mut info = info();
        info.iptc.copyright = "Ada 2026".into();
        info.exif.camera_make = Some("Sony".into());
        info.keywords = vec!["wedding".into()];
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.include_metadata = MetadataPolicy::Copyright;
        let path = write_export(&dir, &info, &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();

        let tiff = app1(&bytes, b"Exif\0\0").expect("copyright must write an EXIF segment");
        assert_eq!(tags(tiff, 8), vec![0x8298], "IFD0 must hold the copyright tag only");
        assert_eq!(ascii_field(tiff, 8, 0x8298), "Ada 2026");
        assert!(app1(&bytes, XMP_SIGNATURE).is_none(), "no XMP for a copyright export");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn camera_writes_make_model_and_the_exposure_record() {
        let dir = temp_dir("meta-camera");
        let mut info = info();
        info.iptc.copyright = "© 2026 Ada".into();
        info.exif = Exif {
            camera_make: Some("Sony".into()),
            camera_model: Some("ILCE-7M4".into()),
            lens: Some("FE 35mm F1.8".into()),
            focal_length_mm: Some(35.0),
            aperture: Some(1.8),
            shutter: Some(0.004),
            iso: Some(400),
            captured_at: Some("2026-11-10T08:30:00+08:00".into()),
            gps: Some((31.2397, 121.4998)),
            ..Exif::default()
        };
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.include_metadata = MetadataPolicy::Camera;
        let path = write_export(&dir, &info, &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();

        let tiff = app1(&bytes, b"Exif\0\0").expect("camera metadata must write EXIF");
        assert_eq!(tags(tiff, 8), vec![0x010F, 0x0110, 0x8769], "IFD0 must hold make, model and the Exif pointer");
        let exif_ifd = pointer(tiff, 0x8769);
        assert_eq!(tags(tiff, exif_ifd), vec![0x829A, 0x829D, 0x8827, 0x920A, 0xA434]);
        assert!(find(tiff, b"2026:11:10 08:30:00").is_none(), "the capture date is not camera metadata");
        assert!(app1(&bytes, XMP_SIGNATURE).is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn gps_writes_only_the_gps_ifd() {
        let dir = temp_dir("meta-gps");
        let mut info = info();
        info.exif = Exif { camera_make: Some("Sony".into()), gps: Some((31.2397, 121.4998)), ..Exif::default() };
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.include_metadata = MetadataPolicy::Gps;
        let path = write_export(&dir, &info, &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();

        let tiff = app1(&bytes, b"Exif\0\0").expect("GPS must write EXIF");
        assert_eq!(tags(tiff, 8), vec![0x8825], "IFD0 must hold the GPS pointer only");
        assert_eq!(tags(tiff, pointer(tiff, 0x8825)), vec![0x0001, 0x0002, 0x0003, 0x0004]);
        assert!(find(tiff, b"Sony").is_none(), "the camera group is not part of GPS");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn keywords_write_an_xmp_app1_segment() {
        let dir = temp_dir("meta-keywords");
        let mut info = info();
        info.keywords = vec!["wedding".into(), "hero & co".into()];
        info.rating = 4;
        info.color_tag = ColorTag::Red;
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.include_metadata = MetadataPolicy::Keywords;
        let path = write_export(&dir, &info, &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();

        let meta = crate::XmpMeta::parse(&xmp_text(&bytes)).unwrap();
        assert_eq!(meta, crate::XmpMeta { rating: 0, label: String::new(), keywords: info.keywords.clone() });
        assert!(app1(&bytes, b"Exif\0\0").is_none(), "keywords must not write EXIF");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn keywords_write_an_xmp_itxt_chunk_in_png() {
        let dir = temp_dir("meta-keywords-png");
        let mut info = info();
        info.keywords = vec!["wedding".into(), "hero & co".into()];
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.format = ExportFormat::Png;
        r.include_metadata = MetadataPolicy::Keywords;
        let path = write_export(&dir, &info, &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();

        assert!(find(&bytes, b"XML:com.adobe.xmp").is_some(), "PNG must carry the XMP iTXt chunk");
        let start = find(&bytes, b"<?xpacket").expect("the chunk must hold the packet");
        let tail = b"<?xpacket end=\"w\"?>";
        let end = start + find(&bytes[start..], tail).unwrap() + tail.len();
        let meta = crate::XmpMeta::parse(std::str::from_utf8(&bytes[start..end]).unwrap()).unwrap();
        assert_eq!(meta.keywords, info.keywords, "the chunk must round-trip");
        // A valid CRC is what keeps the image readable.
        assert_eq!(image::load_from_memory(&bytes).unwrap().width(), 8);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn ascii_copyright_round_trips_through_the_ifd() {
        let dir = temp_dir("meta-copyright-ascii");
        let mut info = info();
        info.iptc.copyright = "Ada Lovelace 2026".into();
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.include_metadata = MetadataPolicy::Copyright;
        let path = write_export(&dir, &info, &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();

        let tiff = app1(&bytes, b"Exif\0\0").expect("ASCII copyright belongs in EXIF");
        assert_eq!(ascii_field(tiff, 8, 0x8298), "Ada Lovelace 2026");
        assert!(app1(&bytes, XMP_SIGNATURE).is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unicode_copyright_round_trips_through_dc_rights() {
        let dir = temp_dir("meta-copyright-unicode");
        let mut info = info();
        info.iptc.copyright = "© 2026 Ada".into();
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.include_metadata = MetadataPolicy::Copyright;
        let path = write_export(&dir, &info, &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();

        assert!(app1(&bytes, b"Exif\0\0").is_none(), "the ASCII tag would mangle non-ASCII text");
        assert_eq!(dc_rights(&bytes), "© 2026 Ada");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn camera_fields_round_trip() {
        let dir = temp_dir("meta-camera-values");
        let mut info = info();
        info.exif = Exif {
            camera_make: Some("Sony".into()),
            camera_model: Some("ILCE-7M4".into()),
            lens: Some("FE 35mm F1.8".into()),
            focal_length_mm: Some(35.0),
            aperture: Some(1.8),
            shutter: Some(0.004),
            iso: Some(400),
            ..Exif::default()
        };
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.include_metadata = MetadataPolicy::Camera;
        let path = write_export(&dir, &info, &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();

        let tiff = app1(&bytes, b"Exif\0\0").expect("camera metadata must write EXIF");
        assert_eq!(ascii_field(tiff, 8, 0x010F), "Sony");
        assert_eq!(ascii_field(tiff, 8, 0x0110), "ILCE-7M4");
        let exif_ifd = pointer(tiff, 0x8769);
        assert_eq!(ascii_field(tiff, exif_ifd, 0xA434), "FE 35mm F1.8");
        assert_eq!(short_field(tiff, exif_ifd, 0x8827), 400);
        assert!((rationals_field(tiff, exif_ifd, 0x829A)[0] - 0.004).abs() < 1e-6);
        assert!((rationals_field(tiff, exif_ifd, 0x829D)[0] - 1.8).abs() < 1e-6);
        assert!((rationals_field(tiff, exif_ifd, 0x920A)[0] - 35.0).abs() < 1e-6);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn gps_fields_round_trip_with_their_hemisphere() {
        let dir = temp_dir("meta-gps-values");
        let mut info = info();
        info.exif.gps = Some((-33.8688, 151.2093));
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.include_metadata = MetadataPolicy::Gps;
        let path = write_export(&dir, &info, &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();

        let tiff = app1(&bytes, b"Exif\0\0").expect("GPS must write EXIF");
        let gps = pointer(tiff, 0x8825);
        assert_eq!(ascii_field(tiff, gps, 0x0001), "S");
        assert_eq!(ascii_field(tiff, gps, 0x0003), "E");
        let dms = |parts: Vec<f64>| parts[0] + parts[1] / 60.0 + parts[2] / 3600.0;
        assert!((dms(rationals_field(tiff, gps, 0x0002)) - 33.8688).abs() < 1e-4);
        assert!((dms(rationals_field(tiff, gps, 0x0004)) - 151.2093).abs() < 1e-4);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sixteen_bit_png_writes_exif_and_xmp() {
        let dir = temp_dir("meta-16bit");
        let mut info = info();
        info.iptc.copyright = "Ada 2026".into();
        info.keywords = vec!["wedding".into()];

        let mut copyright = recipe(ScaleMode::Fixed100, (0, 0));
        copyright.format = ExportFormat::Png;
        copyright.bit_depth = 16;
        copyright.include_metadata = MetadataPolicy::Copyright;
        let path = write_export16(&dir, &info, &copyright, 0, vec![40000u16; 8 * 8 * 4], 8, 8, |_| {})
            .unwrap();
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(ascii_field(png_exif(&bytes), 8, 0x8298), "Ada 2026");

        let mut keywords = recipe(ScaleMode::Fixed100, (0, 0));
        keywords.format = ExportFormat::Png;
        keywords.bit_depth = 16;
        keywords.include_metadata = MetadataPolicy::Keywords;
        let path = write_export16(&dir, &info, &keywords, 0, vec![40000u16; 8 * 8 * 4], 8, 8, |_| {})
            .unwrap();
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(crate::XmpMeta::parse(&packet_text(&bytes)).unwrap().keywords, info.keywords);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn strip_all_writes_no_metadata_on_either_path() {
        let dir = temp_dir("meta-strip-both");
        let mut info = info();
        info.iptc.copyright = "© 2026 Ada".into();
        info.exif = Exif {
            camera_make: Some("Sony".into()),
            camera_model: Some("ILCE-7M4".into()),
            gps: Some((31.2397, 121.4998)),
            ..Exif::default()
        };
        info.keywords = vec!["wedding".into()];

        let mut jpeg = recipe(ScaleMode::Fixed100, (0, 0));
        jpeg.include_metadata = MetadataPolicy::StripAll;
        let path = write_export(&dir, &info, &jpeg, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();
        for needle in [&b"Exif\0\0"[..], b"dc:subject", b"dc:rights"] {
            assert!(find(&bytes, needle).is_none(), "strip all must not write {needle:?}");
        }

        let mut png = recipe16(ExportFormat::Png);
        png.include_metadata = MetadataPolicy::StripAll;
        let path = write_export16(&dir, &info, &png, 0, vec![40000u16; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();
        for needle in [&b"eXIf"[..], b"dc:subject", b"dc:rights"] {
            assert!(find(&bytes, needle).is_none(), "strip all must not write {needle:?}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn damaged_metadata_never_panics() {
        let dir = temp_dir("meta-bad");
        let mut info = info();
        info.exif.gps = Some((f64::NAN, f64::INFINITY));
        info.exif.camera_make = Some(String::new());
        info.keywords = vec![String::new(), "x".repeat(70_000)];

        for policy in [MetadataPolicy::Gps, MetadataPolicy::Camera, MetadataPolicy::Keywords] {
            let mut r = recipe(ScaleMode::Fixed100, (0, 0));
            r.include_metadata = policy;
            let path = write_export(&dir, &info, &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
            let bytes = std::fs::read(path).unwrap();
            assert!(image::load_from_memory(&bytes).is_ok(), "{} must stay a valid JPEG", policy.label());
        }

        // A copyright too large for one APP1 segment may fail the export, but
        // it must not take the process down.
        info.iptc.copyright = "A".repeat(100_000);
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.include_metadata = MetadataPolicy::Copyright;
        if let Ok(path) = write_export(&dir, &info, &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}) {
            assert!(image::load_from_memory(&std::fs::read(path).unwrap()).is_ok());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn every_ifd_is_in_ascending_tag_order() {
        let tiff = build_exif(&Exif {
            camera_make: Some("Sony".into()),
            camera_model: Some("ILCE-7M4".into()),
            lens: Some("FE 35mm F1.8".into()),
            focal_length_mm: Some(35.0),
            aperture: Some(1.8),
            shutter: Some(0.004),
            iso: Some(400),
            captured_at: Some("2026-11-10T08:30:00+08:00".into()),
            gps: Some((31.2397, 121.4998)),
            ..Exif::default()
        })
        .unwrap();
        for at in [8, pointer(&tiff, 0x8769), pointer(&tiff, 0x8825)] {
            let tags = tags(&tiff, at);
            assert!(tags.windows(2).all(|pair| pair[0] < pair[1]), "{tags:?} must be sorted");
        }
    }

    #[test]
    fn rating_and_tag_write_rating_and_label_only() {
        let dir = temp_dir("meta-rating");
        let mut info = info();
        info.rating = 4;
        info.color_tag = ColorTag::Red;
        info.keywords = vec!["wedding".into()];
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.include_metadata = MetadataPolicy::RatingAndTag;
        let path = write_export(&dir, &info, &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();

        let text = xmp_text(&bytes);
        assert_eq!(
            crate::XmpMeta::parse(&text).unwrap(),
            crate::XmpMeta { rating: 4, label: "Red".into(), keywords: Vec::new() }
        );
        assert!(find(text.as_bytes(), b"dc:subject").is_none(), "rating does not write keywords");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn xmp_does_not_displace_the_exif_segment() {
        let payload = b"MM\0*\0\0\0\x08\0\0";
        let mut jpeg = jpeg_with_exif(&payload[..]);
        let packet = b"<?xpacket end=\"w\"?>";
        embed_xmp(&mut jpeg, packet);

        assert_eq!(app1(&jpeg, b"Exif\0\0"), Some(&payload[..]), "the EXIF payload must stay verbatim");
        assert_eq!(app1(&jpeg, XMP_SIGNATURE), Some(&packet[..]));
        assert_eq!(segments(&jpeg).iter().filter(|(marker, _)| *marker == 0xE1).count(), 2);
    }

    // --------------------------------------------------------------- ppi

    #[test]
    fn the_recipe_ppi_becomes_the_jfif_density() {
        let dir = temp_dir("ppi");
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.resolution_ppi = 300;
        let path = write_export(&dir, &info(), &r, 0, vec![128u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();
        let app0 = find(&bytes, b"JFIF").expect("the file must carry a JFIF header");
        // JFIF: magic(5) version(2) units(1) xdensity(2) ydensity(2)
        let units = bytes[app0 + 7];
        let x = u16::from_be_bytes([bytes[app0 + 8], bytes[app0 + 9]]);
        let y = u16::from_be_bytes([bytes[app0 + 10], bytes[app0 + 11]]);
        assert_eq!(units, 1, "the density unit must be inches");
        assert_eq!((x, y), (300, 300));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // ----------------------------------------------------------- watermark

    #[test]
    fn the_watermark_reaches_the_bottom_right_corner() {
        let dir = temp_dir("watermark");
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.watermark = Some(Watermark {
            text: "W".into(),
            opacity: 100.0,
            scale: 1.0,
            position: [1.0, 1.0],
        });
        r.format = ExportFormat::Png;
        let path = write_export(&dir, &info(), &r, 0, vec![0u8; 32 * 32 * 4], 32, 32, |_| {}).unwrap();
        let img = image::open(path).unwrap().to_rgba8();
        let drawn = (16..32).any(|y| (16..32).any(|x| img.get_pixel(x, y).0[0] > 200));
        assert!(drawn, "the watermark must be drawn in the bottom-right corner");
        assert_eq!(img.get_pixel(1, 1).0, [0, 0, 0, 0], "the rest must stay untouched");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `position` is a normalized centre anchor; it must actually move the text.
    #[test]
    fn the_watermark_follows_its_position() {
        let dir = temp_dir("watermark-pos");
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.format = ExportFormat::Png;
        r.watermark = Some(Watermark {
            text: "W".into(),
            opacity: 100.0,
            scale: 1.0,
            position: [0.1, 0.1],
        });
        let path = write_export(&dir, &info(), &r, 0, vec![0u8; 32 * 32 * 4], 32, 32, |_| {}).unwrap();
        let img = image::open(path).unwrap().to_rgba8();
        let top_left = (0..16).any(|y| (0..16).any(|x| img.get_pixel(x, y).0[0] > 200));
        let bottom_right = (16..32).any(|y| (16..32).any(|x| img.get_pixel(x, y).0[0] > 200));
        assert!(top_left, "position [0.1, 0.1] must draw near the top-left");
        assert!(!bottom_right, "nothing must be drawn in the opposite corner");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rating_and_tag_write_an_xmp_itxt_chunk_in_png() {
        let dir = temp_dir("meta-rating-png");
        let mut info = info();
        info.rating = 4;
        info.color_tag = ColorTag::Red;
        info.keywords = vec!["wedding".into()];
        let mut r = recipe(ScaleMode::Fixed100, (0, 0));
        r.format = ExportFormat::Png;
        r.include_metadata = MetadataPolicy::RatingAndTag;
        let path = write_export(&dir, &info, &r, 0, vec![70u8; 8 * 8 * 4], 8, 8, |_| {}).unwrap();
        let bytes = std::fs::read(path).unwrap();

        let start = find(&bytes, b"<?xpacket").expect("the PNG must carry an XMP packet");
        let tail = b"<?xpacket end=\"w\"?>";
        let end = start + find(&bytes[start..], tail).unwrap() + tail.len();
        let meta = crate::XmpMeta::parse(std::str::from_utf8(&bytes[start..end]).unwrap()).unwrap();
        assert_eq!(meta, crate::XmpMeta { rating: 4, label: "Red".into(), keywords: Vec::new() });
        assert!(find(&bytes, b"dc:subject").is_none(), "rating does not write keywords");
        assert_eq!(image::load_from_memory(&bytes).unwrap().width(), 8, "the chunk must keep the PNG valid");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // --------------------------------------------------------------- 16-bit

    /// A scratch directory under the crate, so tests never write to /tmp.
    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/export16-tests")
            .join(format!("{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn recipe16(format: ExportFormat) -> ProcessRecipe {
        ProcessRecipe {
            format,
            bit_depth: 16,
            icc_profile: String::new(),
            include_metadata: MetadataPolicy::StripAll,
            ..ProcessRecipe::default()
        }
    }

    fn wide_image(w: u32, h: u32) -> Vec<u16> {
        (0..(w * h) as usize)
            .flat_map(|i| {
                let x = (i as u32) % w;
                let y = (i as u32) / w;
                [
                    (x * 65535 / (w - 1)) as u16,
                    (y * 65535 / (h - 1)) as u16,
                    40000,
                    65535,
                ]
            })
            .collect()
    }

    #[test]
    fn png16_round_trips_exact_values() {
        let dir = scratch_dir("png16-values");
        let (w, h) = (8u32, 4u32);
        let wide = wide_image(w, h);
        let path = write_export16(&dir, &info(), &recipe16(ExportFormat::Png), 0, wide.clone(), w, h, |_| {}).unwrap();
        let img = image::open(&path).unwrap().to_rgba16();
        assert_eq!((img.width(), img.height()), (w, h));
        assert_eq!(img.as_raw(), &wide, "16-bit PNG must round-trip every code");
        assert_eq!(img.as_raw()[3], 65535, "alpha stays 16-bit");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn tiff16_round_trips_exact_values() {
        let dir = scratch_dir("tiff16-values");
        let (w, h) = (8u32, 4u32);
        let wide = wide_image(w, h);
        let path = write_export16(&dir, &info(), &recipe16(ExportFormat::Tiff), 0, wide.clone(), w, h, |_| {}).unwrap();
        assert_eq!(path.extension().unwrap(), "tif");
        let img = image::open(&path).unwrap().to_rgba16();
        assert_eq!(img.as_raw(), &wide, "16-bit TIFF must round-trip every code");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn eight_bit_export_is_unchanged() {
        let dir = scratch_dir("png8");
        let mut r = recipe16(ExportFormat::Png);
        r.bit_depth = 8;
        let px: Vec<u8> = (0..8 * 4 * 4).map(|i| (i * 37 % 256) as u8).collect();
        let path = write_export(&dir, &info(), &r, 0, px.clone(), 8, 4, |_| {}).unwrap();
        let img = image::open(&path).unwrap();
        assert!(matches!(img, image::DynamicImage::ImageRgba8(_)), "8-bit export must stay 8-bit");
        assert_eq!(img.to_rgba8().as_raw(), &px);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sixteen_bit_export_keeps_alpha() {
        let dir = scratch_dir("png16-alpha");
        let (w, h) = (4u32, 1u32);
        let wide: Vec<u16> = [0u16, 4096, 32768, 65535]
            .iter()
            .flat_map(|&v| [v, v, v, v])
            .collect();
        let path = write_export16(&dir, &info(), &recipe16(ExportFormat::Png), 0, wide, w, h, |_| {}).unwrap();
        let img = image::open(&path).unwrap().to_rgba16();
        let alpha: Vec<u16> = img.pixels().map(|p| p.0[3]).collect();
        assert_eq!(alpha, vec![0, 4096, 32768, 65535]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sixteen_bit_rejects_formats_without_an_encoder() {
        let dir = scratch_dir("png16-unsupported");
        let mut r = recipe16(ExportFormat::WebP);
        r.format = ExportFormat::WebP;
        let err = write_export16(&dir, &info(), &r, 0, vec![0u16; 4 * 4 * 4], 4, 4, |_| {});
        assert!(err.is_err(), "16-bit WebP must be an error, not a 8-bit downgrade");
        assert!(err.unwrap_err().to_string().contains("16-bit"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn bit_depth_support_is_an_error_not_a_downgrade() {
        assert!(ExportFormat::Png.supports_bit_depth(16).is_ok());
        assert!(ExportFormat::Tiff.supports_bit_depth(16).is_ok());
        for format in [ExportFormat::Jpeg, ExportFormat::WebP] {
            let err = format.supports_bit_depth(16).unwrap_err().to_string();
            assert!(err.contains("16-bit"), "{format:?}: {err}");
        }
        assert!(ExportFormat::Jpeg.supports_bit_depth(8).is_ok());
        assert!(ExportFormat::Png.supports_bit_depth(12).is_err());
    }

    #[test]
    fn sixteen_bit_rejects_an_oversized_frame_without_panicking() {
        let dir = scratch_dir("png16-huge");
        let err = write_export16(&dir, &info(), &recipe16(ExportFormat::Png), 0, vec![0u16; 4], u32::MAX, u32::MAX, |_| {});
        assert!(err.is_err(), "an overflowing frame size must be an error");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Export timings and shadow code counts for the 3000x2000 report.
    /// `cargo test -p lumen-engine --offline bench_export16 -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_export16_against_8bit() {
        use std::collections::BTreeSet;
        use std::time::Instant;
        let dir = scratch_dir("bench");
        let (w, h) = (3000u32, 2000u32);
        let wide: Vec<u16> = (0..(w * h) as usize)
            .flat_map(|i| {
                let x = (i as u32) % w;
                let y = (i as u32) / w;
                let v = 60 + (x * 3400 / w) as u16 + (y * 500 / h) as u16 + (x as u16 % 7);
                [v, v, v, 65535]
            })
            .collect();
        let src = crate::mock::process::Source::from_rgba16(w, h, &wide);
        let adj = crate::AdjustmentSet::default();
        let dark = |px: &[u16]| -> BTreeSet<u16> {
            px.chunks_exact(4).filter(|p| p[0] < 4096).map(|p| p[0]).collect()
        };
        let r16 = recipe16(ExportFormat::Png);
        let start = Instant::now();
        let out16 = crate::mock::process::render16(&src, &adj, crate::NormRect::FULL, w.max(h), false);
        let p16 =
            write_export16(&dir, &info(), &r16, 0, out16.rgba, out16.width, out16.height, |_| {}).unwrap();
        let t16 = start.elapsed();
        let mut r8 = r16.clone();
        r8.bit_depth = 8;
        let start = Instant::now();
        let out8 = crate::mock::process::render(&src, &adj, crate::NormRect::FULL, w.max(h), false, None);
        let p8 = write_export(&dir, &info(), &r8, 0, out8.rgba, out8.width, out8.height, |_| {}).unwrap();
        let t8 = start.elapsed();
        let img16 = image::open(&p16).unwrap().to_rgba16();
        let img8 = image::open(&p8).unwrap().to_rgba8();
        let d16 = dark(img16.as_raw()).len();
        let d8: BTreeSet<u8> = img8.pixels().map(|p| p.0[0]).filter(|&v| v < 16).collect();
        let all16: BTreeSet<u16> = img16.as_raw().chunks_exact(4).map(|p| p[0]).collect();
        let (lo, hi) = (
            img16.as_raw().chunks_exact(4).map(|p| p[0]).min().unwrap(),
            img16.as_raw().chunks_exact(4).map(|p| p[0]).max().unwrap(),
        );
        println!(
            "3000x2000 PNG render+export: 16-bit {} KiB in {:.0} ms, 8-bit {} KiB in {:.0} ms; \
             distinct dark codes 16-bit {} vs 8-bit {} (all {} codes, {lo}..={hi})",
            std::fs::metadata(&p16).unwrap().len() / 1024,
            t16.as_secs_f64() * 1000.0,
            std::fs::metadata(&p8).unwrap().len() / 1024,
            t8.as_secs_f64() * 1000.0,
            d16,
            d8.len(),
            all16.len()
        );
        assert!(d16 > d8.len(), "16-bit must carry more shadow codes");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The first occurrence of `needle` in `haystack`.
    fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    #[cfg(feature = "external-fixtures")]
    include!("../local-fixtures/recipe-tests.rs");
}
