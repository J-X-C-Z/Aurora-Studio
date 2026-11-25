//! RAW import. When the system `libraw` shared library is present the sensor data is
//! demosaiced and white-balanced by the capture's own camera multipliers, output as
//! linear light in the camera's own colour space; otherwise the camera-embedded JPEG
//! is all we have. The demosaic filter is selectable through [`set_demosaic`] or the
//! `AURORA_DEMOSAIC` environment variable; AHD stays the default.
//!
//! libraw is resolved at run time, so a build without it falls back to the
//! preview instead of failing. On Windows the DLL and its dependencies are
//! vendored under `vendor/libraw/bin`, which is added to the search path.

use std::io::{BufReader, ErrorKind, Read};
use std::path::Path;
use std::time::Instant;
use std::sync::atomic::{AtomicI32, Ordering};

use image::{ImageFormat, RgbaImage};

macro_rules! raw_trace {
    ($($arg:tt)*) => {
        if std::env::var_os("AURORA_RAW_TRACE").is_some() {
            eprintln!($($arg)*);
        }
    };
}

/// Bytes read per pass while scanning the container.
const CHUNK: usize = 1 << 20;
/// Upper bound for one candidate segment. No camera preview is anywhere near this;
/// it only stops a stray SOI in sensor data from growing the buffer without limit.
const MAX_SEGMENT: usize = 96 << 20;
/// Upper bound for decode attempts per candidate, so a segment full of stray `FF D9`
/// bytes cannot make the scan quadratic.
const MAX_ATTEMPTS: usize = 64;

fn evaluate_preview(segment: &[u8]) -> Option<(u64, RgbaImage)> {
    let decoded = image::load_from_memory_with_format(segment, ImageFormat::Jpeg).ok()?;
    let (w, h) = (decoded.width(), decoded.height());
    if w == 0 || h == 0 {
        return None;
    }
    Some((u64::from(w) * u64::from(h), decoded.to_rgba8()))
}

/// Returns the largest embedded JPEG preview, decoded.
pub fn embedded_preview(path: &Path) -> anyhow::Result<RgbaImage> {
    let found = scan_best(path, evaluate_preview)?;
    found.ok_or_else(|| anyhow::anyhow!("no embedded JPEG preview in {}", path.display()))
}

/// Fast progressive first-frame preview for a cold RAW: streams chunks and
/// stops as soon as an embedded JPEG with long side `>= target_long_side.clamp(640, 1600)`
/// is found, then applies EXIF orientation so portrait frames appear upright.
pub fn quick_embedded_preview(path: &Path, target_long_side: u32) -> anyhow::Result<RgbaImage> {
    let min_long_side = target_long_side.clamp(640, 1600);
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::with_capacity(CHUNK, file);
    let mut buf = vec![0u8; CHUNK];
    let mut state = ScanState::default();
    let mut best: Option<(u64, RgbaImage)> = None;

    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                feed(&mut state, &buf[..n], &mut evaluate_preview, &mut best);
                if let Some((_, ref img)) = best {
                    if img.width().max(img.height()) >= min_long_side {
                        break;
                    }
                }
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    let img = best
        .map(|(_, value)| value)
        .ok_or_else(|| anyhow::anyhow!("no embedded JPEG preview in {}", path.display()))?;
    let oriented = match exif_orientation_tag(path).unwrap_or(1) {
        3 => image::imageops::rotate180(&img),
        6 => image::imageops::rotate90(&img),
        8 => image::imageops::rotate270(&img),
        _ => img,
    };
    Ok(oriented)
}

/// [`embedded_preview`] over bytes we already hold: the sensor/preview sanity
/// check must not read the RAW a second time.
fn embedded_preview_bytes(bytes: &[u8]) -> Option<RgbaImage> {
    scan_best_bytes(bytes, evaluate_preview)
}

/// The sRGB transfer curve on linear light 0..=1.
fn srgb_f(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}
/// Linear light 0..=1 to an sRGB-encoded byte, rounded to the nearest code.
fn encode_srgb(v: f32) -> u8 {
    (srgb_f(v) * 255.0 + 0.5) as u8
}

/// libraw's `user_qual` for AHD, the long-standing default.
pub const DEMOSAIC_DEFAULT: i32 = 3;

/// Process-wide override, -1 meaning "not set".
///
/// AHD stays the default because it wins detail: on `_DSC5673.NEF` (6064x4040,
/// ISO 5000, a high-frequency colour chart) its mean green gradient energy was
/// 3.88e6 versus 3.86e6 for PPG/DCB, 3.36e6 for VNG and 2.68e6 for linear, at
/// 1.93 s per frame. Linear's lower false-colour spread (79 vs 122) is bought
/// with that 31% detail loss, and DCB/AFD are worse on both axes.
static DEMOSAIC: AtomicI32 = AtomicI32::new(-1);

/// The filters this build of libraw really implements. Anything else is mapped
/// back to AHD, because libraw itself validates nothing: `libraw_set_demosaic`
/// returns `void` and an out-of-range value just lands in `user_qual`.
fn normalise_demosaic(quality: i32) -> i32 {
    match quality {
        0 | 1 | 2 | 3 | 11 | 12 => quality,
        _ => DEMOSAIC_DEFAULT,
    }
}

/// Parses a numeric `AURORA_DEMOSAIC` value, rejecting filters libraw lacks here.
fn parse_demosaic(text: &str) -> Option<i32> {
    match text.trim().parse::<i32>() {
        Ok(quality @ (0 | 1 | 2 | 3 | 11 | 12)) => Some(quality),
        _ => None,
    }
}

/// Overrides the demosaic filter for every later decode.
///
/// 0 linear, 1 VNG, 2 PPG, 3 AHD, 11 DCB, 12 AFD. Invalid values fall back to AHD.
pub fn set_demosaic(quality: i32) {
    DEMOSAIC.store(normalise_demosaic(quality), Ordering::Relaxed);
}

/// Resolves the filter in force: `AURORA_DEMOSAIC` wins over [`set_demosaic`],
/// and AHD is the final default.
pub fn demosaic_quality() -> i32 {
    match std::env::var("AURORA_DEMOSAIC") {
        Ok(text) => parse_demosaic(&text).unwrap_or_else(|| {
            log::warn!("AURORA_DEMOSAIC={text:?} is not 0/1/2/3/11/12; using AHD");
            DEMOSAIC_DEFAULT
        }),
        Err(_) => match DEMOSAIC.load(Ordering::Relaxed) {
            -1 => DEMOSAIC_DEFAULT,
            quality => quality,
        },
    }
}

/// Decodes the sensor data with libraw when available, else the embedded preview.
///
/// Pixels come out sRGB-transfer-encoded. libraw hands over 16-bit linear, and
/// encoding at that width keeps about eight times the shadow levels that
/// truncating to 8-bit linear leaves.
pub fn decode(path: &Path) -> anyhow::Result<RgbaImage> {
    decode_from_path(path)
}

/// The historical 8-bit decode: open the path, demosaic with LibRaw, else scan
/// the embedded preview.
fn decode_from_path(path: &Path) -> anyhow::Result<RgbaImage> {
    match libraw::decode(path, demosaic_quality()) {
        Ok(decoded) if matches_preview(&decoded, path) => {
            remember_sensor(path, true);
            return Ok(decoded);
        }
        Ok(_) => log::warn!(
            "libraw {}: the sensor decode does not resemble the camera's own preview \
             (unsupported compression?); using the embedded preview",
            path.display()
        ),
        Err(err) => log::warn!("libraw {}: {err}; using the embedded preview", path.display()),
    }
    remember_sensor(path, false);
    embedded_preview(path)
}

/// The 8-bit fallback for a RAW whose 16-bit attempt already read the file.
///
/// `failed` carries the container bytes that attempt read, so this function
/// never has to read the file again. The pixels are the same [`decode`]
/// produces:
///
/// * `libraw_unpack_unsupported` means LibRaw itself returned
///   `LIBRAW_FILE_UNSUPPORTED` for these exact bytes during the 16-bit attempt.
///   The same build would reject the same bytes again, so the embedded preview
///   is used without a second LibRaw open. This is an observed result, not a
///   guess about the file format.
/// * Otherwise LibRaw runs on the in-memory buffer (`libraw_open_buffer`) when
///   the build exports it, falling back to the path exactly as before. The
///   similarity check scans the same bytes instead of re-reading the file.
pub(crate) fn decode8_fallback(
    path: &Path,
    failed: Option<&FailedRaw16>,
) -> anyhow::Result<RgbaImage> {
    let Some(failed) = failed else {
        return decode_from_path(path);
    };
    if failed.libraw_unpack_unsupported {
        log::debug!(
            "libraw {}: unpack already reported unsupported for these bytes; \
             using the embedded preview without reopening",
            path.display()
        );
        remember_sensor(path, false);
        return embedded_preview_bytes(&failed.bytes)
            .ok_or_else(|| anyhow::anyhow!("no embedded JPEG preview in {}", path.display()));
    }
    log::debug!(
        "libraw {}: reusing the {} bytes the 16-bit attempt read for the 8-bit fallback",
        path.display(),
        failed.bytes.len()
    );
    let decoded = libraw::decode_bytes(path, &failed.bytes, demosaic_quality());
    match decoded {
        Ok(decoded) if same_picture_bytes(&luma_grid(&decoded), &failed.bytes) => {
            remember_sensor(path, true);
            return Ok(decoded);
        }
        Ok(_) => log::warn!(
            "libraw {}: the sensor decode does not resemble the camera's own preview \
             (unsupported compression?); using the embedded preview",
            path.display()
        ),
        Err(err) => log::warn!("libraw {}: {err}; using the embedded preview", path.display()),
    }
    remember_sensor(path, false);
    embedded_preview_bytes(&failed.bytes)
        .ok_or_else(|| anyhow::anyhow!("no embedded JPEG preview in {}", path.display()))
}

/// Containers individually larger than this are not retained for the 8-bit
/// fallback; the fallback then reads the file again instead of holding a second
/// large copy while decoding. The limit is a memory guard only; well above any
/// current camera RAW, so normal files always take the reuse path.
const MAX_FAILED16_BYTES: usize = 512 << 20;

/// What a failed 16-bit attempt leaves behind for the 8-bit fallback.
pub(crate) struct FailedRaw16 {
    /// The bytes the 16-bit attempt already read.
    pub bytes: Vec<u8>,
    /// LibRaw returned `LIBRAW_FILE_UNSUPPORTED` while unpacking these exact
    /// bytes. A second LibRaw attempt cannot succeed, so the fallback goes
    /// straight to the embedded preview.
    pub libraw_unpack_unsupported: bool,
}

/// Whether [`decode`] returned real sensor data for path, or the camera's own
/// preview because this libraw cannot demosaic the file. Camera profiles and the
/// film curve only make sense on sensor data; the preview is already processed.
pub fn from_sensor(path: &Path) -> Option<bool> {
    sensor_map().lock().unwrap_or_else(|e| e.into_inner()).get(path).copied()
}

fn remember_sensor(path: &Path, sensor: bool) {
    let mut map = sensor_map().lock().unwrap_or_else(|e| e.into_inner());
    if map.len() > 64 {
        map.clear();
    }
    map.insert(path.to_path_buf(), sensor);
}

static SENSOR: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, bool>>> =
    std::sync::OnceLock::new();

fn sensor_map() -> &'static std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, bool>> {
    SENSOR.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}
/// Decodes the sensor data as 16-bit linear RGBA, without any transfer encoding,
/// for the engine's 16-bit source path. `None` when libraw is missing, the file
/// cannot be demosaiced (Nikon High Efficiency in this libraw build), or the
/// result does not resemble the camera preview.
pub fn decode16(path: &Path) -> Option<(u32, u32, Vec<u16>)> {
    decode16_traced(path).map(|(width, height, pixels, _)| (width, height, pixels))
}

/// One RAW decode's shared inputs: the file is read once, EXIF is parsed from
/// those bytes, and the LibRaw handle is opened on the same buffer so the
/// fallback never has to decode the file again.
struct RawDecodeContext {
    bytes: Vec<u8>,
    exif: crate::Exif,
    timings: RawStageTimings,
}

impl RawDecodeContext {
    fn read(path: &Path) -> Option<Self> {
        let read_started = Instant::now();
        let bytes = std::fs::read(path).ok()?;
        let read_ms = read_started.elapsed().as_millis() as u64;
        let exif_started = Instant::now();
        let exif = crate::exif::parse_exif(&bytes);
        let exif_ms = exif_started.elapsed().as_millis() as u64;
        let len = bytes.len() as u64;
        Some(Self {
            bytes,
            exif,
            timings: RawStageTimings {
                bytes_read: len,
                read_ms,
                exif_ms,
                peak_tracked_bytes: len,
                ..RawStageTimings::default()
            },
        })
    }
}

/// [`decode16`] plus the provenance of the branch that actually produced the
/// pixels. Cache headers persist it so a hit can still route the colour
/// pipeline (the Z6III pre-matrix must never run on a LibRaw/preview fallback).
pub fn decode16_traced(path: &Path) -> Option<(u32, u32, Vec<u16>, DecodeProvenance)> {
    decode16_traced_routed(path, decode_route(path))
}

/// [`decode16_traced`] with the route the caller already computed.
///
/// `raw_cache` builds a [`DecodeRoute`] for its key; passing it through keeps
/// the backend choice ahead of the decode and avoids a second Z6III file scan.
pub(crate) fn decode16_traced_routed(
    path: &Path,
    route: DecodeRoute,
) -> Option<(u32, u32, Vec<u16>, DecodeProvenance)> {
    decode16_attempt(path, route).0
}

/// Routes [`decode16_traced_routed`] and hands a failed attempt's bytes back.
///
/// The 8-bit fallback in the file engine reuses those bytes
/// ([`decode8_fallback`]) instead of reading the container and reopening LibRaw.
pub(crate) fn decode16_attempt(
    path: &Path,
    route: DecodeRoute,
) -> (
    Option<(u32, u32, Vec<u16>, DecodeProvenance)>,
    Option<FailedRaw16>,
) {
    let started = Instant::now();
    // The Aurora Raw chain decodes the same Bayer as LibRaw but demosaics with
    // Aurora Raw's own stages. It handles the verified HE variant; `AURORA_C1_RAW=0`
    // is an escape hatch back to the LibRaw path.
    if route.c1_raw_enabled {
        let Some(mut ctx) = RawDecodeContext::read(path) else {
            // The shared read failed; keep the historic path, which lets LibRaw
            // open the file itself. No bytes exist to hand to the fallback.
            return decode16_libraw_only(path, started);
        };
        let forced = libraw_forced();
        let forbidden = libraw_forbidden();
        // A verified Z6III HE NEF needs no LibRaw Bayer plane: the
        // reconstruction decodes the HE codestream itself. Deciding this from
        // the route plus the codestream header, before opening LibRaw, avoids a
        // fruitless open/unpack for compression this build rejects.
        let mut bayerless_aurora_done = false;
        if crate::aurora_raw::route_skips_libraw(&route, &ctx.bytes) {
            ctx.timings.note_bytes(ctx.bytes.len() as u64);
            if let Some(frame) = crate::aurora_raw::decode_he_rgba16_with(
                &ctx.bytes,
                &ctx.exif,
                None,
                &mut ctx.timings,
            ) {
                let mut kind = RawKind::classify(path, None, ctx.exif.camera_model.as_deref());
                return (
                    Some(finish_aurora(path, &mut kind, frame, &mut ctx.timings, started)),
                    None,
                );
            }
            bayerless_aurora_done = true;
            log::debug!(
                "Aurora Raw declined {} before LibRaw unpack; continuing with LibRaw",
                path.display()
            );
        }
        let open_started = Instant::now();
        // `AURORA_NO_LIBRAW=1` makes the native front-end the only allowed
        // source; an installed LibRaw must not silently win instead.
        let mut session = if forbidden && !forced {
            None
        } else {
            libraw::Session::open(path, &ctx.bytes, demosaic_quality()).ok()
        };
        ctx.timings.libraw_open_ms = open_started.elapsed().as_millis() as u64;
        let mut unpack_unsupported = false;
        let bayer = session.as_mut().and_then(|session| {
            let unpack_started = Instant::now();
            let unpacked = match session.unpack() {
                Ok(()) => true,
                Err(BayerError::Unsupported) => {
                    // The same bytes and the same LibRaw build: a second open
                    // cannot turn this into a successful unpack.
                    unpack_unsupported = true;
                    false
                }
                Err(_) => false,
            };
            ctx.timings.libraw_unpack_ms = unpack_started.elapsed().as_millis() as u64;
            if !unpacked {
                return None;
            }
            let bayer_started = Instant::now();
            let bayer = session.bayer().ok();
            ctx.timings.bayer_ms = bayer_started.elapsed().as_millis() as u64;
            bayer
        });
        // LibRaw first; when it produced no Bayer plane, the native
        // pure-Rust container decoder reuses these bytes instead of reading
        // the file again.
        let effective = effective_bayer(&ctx.bytes, bayer.as_ref());
        let front_end = effective.as_deref();
        if let Some(frame) = front_end {
            ctx.timings
                .note_bytes(ctx.bytes.len() as u64 + (frame.data.len() as u64) * 2);
        }
        let mut kind = RawKind::classify(path, front_end, ctx.exif.camera_model.as_deref());
        // The bayer-less attempt above already ran this exact call when LibRaw
        // produced no plane; repeating it would decode the same bytes a second
        // time for the same answer. A native front-end is a *different* input,
        // so the chain still runs for it.
        let aurora = if front_end.is_none() && bayerless_aurora_done {
            None
        } else {
            crate::aurora_raw::decode_he_rgba16_with(
                &ctx.bytes,
                &ctx.exif,
                front_end,
                &mut ctx.timings,
            )
        };
        if let Some(frame) = aurora {
            return (
                Some(finish_aurora(path, &mut kind, frame, &mut ctx.timings, started)),
                None,
            );
        }
        if let Some(session) = session.as_mut() {
            // The same unpacked handle serves libraw's own demosaic, so the
            // fallback no longer reads and decodes the file a second time.
            let fallback_started = Instant::now();
            let decoded = session.process16();
            ctx.timings.fallback_ms = fallback_started.elapsed().as_millis() as u64;
            match decoded {
                Ok((width, height, rgba)) => {
                    ctx.timings
                        .note_bytes(ctx.bytes.len() as u64 + (rgba.len() as u64) * 2);
                    if same_picture_bytes(&luma_grid16(width, height, &rgba), &ctx.bytes) {
                        remember_sensor(path, true);
                        ctx.timings.total_ms = started.elapsed().as_millis() as u64;
                        record_raw_decode(kind, ctx.timings);
                        return (Some((width, height, rgba, DecodeProvenance::libraw16())), None);
                    }
                    log::warn!(
                        "libraw {}: the 16-bit sensor decode does not resemble the camera's \
                         own preview (unsupported compression?); falling back to the 8-bit path",
                        path.display()
                    );
                }
                Err(err) => log::warn!(
                    "libraw {}: 16-bit {err}; falling back to the 8-bit path",
                    path.display()
                ),
            }
        } else {
            if forbidden && !forced {
                // The native front-end already ran and declined. `AURORA_NO_LIBRAW=1`
                // must not then fall through to LibRaw's own 16-bit decode.
                ctx.timings.total_ms = started.elapsed().as_millis() as u64;
                record_raw_decode(kind, ctx.timings);
                // Under AURORA_NO_LIBRAW the 8-bit fallback must not open
                // LibRaw either, so no bytes are handed over.
                return (None, None);
            }
            // No libraw handle at all (library missing, or the format needs a
            // file handle): keep the historic path and record its cost.
            let fallback_started = Instant::now();
            let decoded = libraw::decode16(path, demosaic_quality());
            ctx.timings.fallback_ms = fallback_started.elapsed().as_millis() as u64;
            if let Ok((width, height, rgba)) = decoded {
                if same_picture_bytes(&luma_grid16(width, height, &rgba), &ctx.bytes) {
                    remember_sensor(path, true);
                    ctx.timings.total_ms = started.elapsed().as_millis() as u64;
                    record_raw_decode(kind, ctx.timings);
                    return (Some((width, height, rgba, DecodeProvenance::libraw16())), None);
                }
            }
        }
        drop(session);
        ctx.timings.total_ms = started.elapsed().as_millis() as u64;
        record_raw_decode(kind, ctx.timings);
        let failed = (ctx.bytes.len() <= MAX_FAILED16_BYTES).then(|| FailedRaw16 {
            bytes: ctx.bytes,
            libraw_unpack_unsupported: unpack_unsupported,
        });
        return (None, failed);
    }
    decode16_libraw_only(path, started)
}

/// The historic LibRaw-only 16-bit attempt, used when C1 RAW is off or when
/// the shared whole-file read failed. It keeps its own `fs` access through
/// LibRaw and never hands bytes to the fallback.
fn decode16_libraw_only(
    path: &Path,
    started: Instant,
) -> (
    Option<(u32, u32, Vec<u16>, DecodeProvenance)>,
    Option<FailedRaw16>,
) {
    let mut timings = RawStageTimings::default();
    let fallback_started = Instant::now();
    let decoded = libraw::decode16(path, demosaic_quality());
    timings.fallback_ms = fallback_started.elapsed().as_millis() as u64;
    match decoded {
        Ok((width, height, rgba)) if same_picture(&luma_grid16(width, height, &rgba), path) => {
            remember_sensor(path, true);
            timings.total_ms = started.elapsed().as_millis() as u64;
            record_raw_decode(RawKind::classify(path, None, None), timings);
            return (Some((width, height, rgba, DecodeProvenance::libraw16())), None);
        }
        Ok(_) => log::warn!(
            "libraw {}: the 16-bit sensor decode does not resemble the camera's own preview \
             (unsupported compression?); falling back to the 8-bit path",
            path.display()
        ),
        Err(err) => log::warn!("libraw {}: 16-bit {err}; falling back to the 8-bit path", path.display()),
    }
    (None, None)
}

/// Records a successful Aurora frame exactly once: kind, sensor flag, timings.
fn finish_aurora(
    path: &Path,
    kind: &mut RawKind,
    frame: (u32, u32, Vec<u16>, crate::aurora_raw::AuroraRoute),
    timings: &mut RawStageTimings,
    started: Instant,
) -> (u32, u32, Vec<u16>, DecodeProvenance) {
    let (width, height, rgba, route) = frame;
    log::info!("Aurora Raw decoded {}", path.display());
    remember_sensor(path, true);
    if matches!(
        route,
        crate::aurora_raw::AuroraRoute::Z6iiiOpenCl
            | crate::aurora_raw::AuroraRoute::Z6iiiCpuOpenClUnavailable
            | crate::aurora_raw::AuroraRoute::Z6iiiCpuOpenClFailed
    ) {
        *kind = RawKind::Z6iiiHe;
    }
    timings.total_ms = started.elapsed().as_millis() as u64;
    record_raw_decode(*kind, *timings);
    (width, height, rgba, DecodeProvenance::aurora(route))
}

/// [`decode16_attempt`] for callers that only need the frame, but want the
/// fallback bytes parked in `failed` when the attempt produced nothing.
pub(crate) fn decode16_traced_into(
    path: &Path,
    route: DecodeRoute,
    failed: &mut Option<FailedRaw16>,
) -> Option<(u32, u32, Vec<u16>, DecodeProvenance)> {
    let (frame, leftover) = decode16_attempt(path, route);
    if frame.is_some() {
        *failed = None;
    } else {
        *failed = leftover;
    }
    frame
}

/// Revision of each independent stage of the RAW pipeline.
///
/// Bump only the stage whose algorithm changed. A cache key is built from the
/// stages a [`DecodeRoute`] actually runs, so an unrelated bump neither
/// invalidates every entry nor hides a changed stage behind a stale one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DecodeStageRevisions {
    pub demosaic: u16,
    pub highlight_recovery: u16,
    pub chromatic_aberration: u16,
    pub noise_model: u16,
    pub z6iii_reconstruction: u16,
    pub orientation_active_area: u16,
    pub output_packing: u16,
}

/// Current revision of every decode stage. The values start at 1 at the split
/// from the former single `DECODE_ALGORITHM_REVISION` constant.
pub const DECODE_STAGE_REVISIONS: DecodeStageRevisions = DecodeStageRevisions {
    demosaic: 1,
    highlight_recovery: 1,
    chromatic_aberration: 1,
    noise_model: 1,
    // Bumped when macOS became able to route a Z6III HE NEF through the Z6III
    // stage shape (CPU C1 instead of the generic Bayer shape). Windows pays one
    // conservative re-decode; the persisted layout and `RAW16_MAGIC` are intact.
    z6iii_reconstruction: 2,
    orientation_active_area: 1,
    output_packing: 1,
};

/// Which stages a decode ran. `LibRaw` runs the always-on stages inside the
/// library; a camera preview or a non-RAW source runs no decode stage at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum StageShape {
    None,
    LibRaw,
    AuroraGeneric,
    AuroraZ6iii,
}

const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const PIPELINE_DOMAIN: u64 = 0xD3C0_DE00_0000_0001;

impl DecodeStageRevisions {
    fn shape_for(backend: DecodeBackend, z6iii_linear: bool) -> StageShape {
        match backend {
            DecodeBackend::AuroraZ6iiiOpenCl => StageShape::AuroraZ6iii,
            DecodeBackend::AuroraCpuChain if z6iii_linear => StageShape::AuroraZ6iii,
            DecodeBackend::AuroraCpuChain => StageShape::AuroraGeneric,
            DecodeBackend::LibRaw => StageShape::LibRaw,
            DecodeBackend::CameraPreview | DecodeBackend::Other => StageShape::None,
        }
    }

    fn fingerprint_of(&self, shape: StageShape) -> u64 {
        fn stage(mut hash: u64, tag: u8, value: u16) -> u64 {
            hash ^= u64::from(tag);
            hash = hash.wrapping_mul(FNV_PRIME);
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            hash
        }

        let mut hash = stage(PIPELINE_DOMAIN, 0, shape as u16);
        if matches!(shape, StageShape::None) {
            return hash ^ (hash >> 32);
        }
        // Always-on stages: demosaic, orientation/active area, output packing.
        hash = stage(hash, 1, self.demosaic);
        hash = stage(hash, 2, self.orientation_active_area);
        hash = stage(hash, 3, self.output_packing);
        if matches!(shape, StageShape::AuroraGeneric | StageShape::AuroraZ6iii) {
            hash = stage(hash, 4, self.highlight_recovery);
            hash = stage(hash, 5, self.chromatic_aberration);
            hash = stage(hash, 6, self.noise_model);
        }
        if matches!(shape, StageShape::AuroraZ6iii) {
            hash = stage(hash, 7, self.z6iii_reconstruction);
        }
        hash ^ (hash >> 32)
    }

    fn route_shape(route: &DecodeRoute) -> StageShape {
        if !route.c1_raw_enabled {
            StageShape::LibRaw
        } else if route.z6iii_nef && route.z6iii_feature {
            StageShape::AuroraZ6iii
        } else {
            StageShape::AuroraGeneric
        }
    }

    /// Stable fingerprint of the stages `route` may run before a decode.
    ///
    /// It goes into the RAW16 cache key, so a changed but *used* stage lands on
    /// a new file while a changed unused stage leaves older entries alone.
    pub fn pipeline_fingerprint(&self, route: &DecodeRoute) -> u64 {
        self.fingerprint_of(Self::route_shape(route))
    }

    /// Stable fingerprint of the stages that produced `provenance`'s pixels,
    /// recomputed from the current constants. A persisted header that carries a
    /// different value was written by an older pipeline and must miss.
    pub fn provenance_fingerprint(&self, provenance: &DecodeProvenance) -> u64 {
        self.fingerprint_of(Self::shape_for(
            provenance.backend,
            provenance.z6iii_linear,
        ))
    }
}

/// Wall time of one RAW decode, split by the stage that spent it, plus the
/// largest tracked buffer the path held (not process RSS).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RawStageTimings {
    pub bytes_read: u64,
    pub read_ms: u64,
    pub exif_ms: u64,
    pub libraw_open_ms: u64,
    pub libraw_unpack_ms: u64,
    pub bayer_ms: u64,
    pub z6iii_ms: u64,
    pub demosaic_ms: u64,
    pub c1_stages_ms: u64,
    pub fallback_ms: u64,
    pub total_ms: u64,
    pub peak_tracked_bytes: u64,
}

impl RawStageTimings {
    fn add(&mut self, other: &Self) {
        self.bytes_read += other.bytes_read;
        self.read_ms += other.read_ms;
        self.exif_ms += other.exif_ms;
        self.libraw_open_ms += other.libraw_open_ms;
        self.libraw_unpack_ms += other.libraw_unpack_ms;
        self.bayer_ms += other.bayer_ms;
        self.z6iii_ms += other.z6iii_ms;
        self.demosaic_ms += other.demosaic_ms;
        self.c1_stages_ms += other.c1_stages_ms;
        self.fallback_ms += other.fallback_ms;
        self.total_ms += other.total_ms;
        self.peak_tracked_bytes = self.peak_tracked_bytes.max(other.peak_tracked_bytes);
    }

    pub(crate) fn note_bytes(&mut self, bytes: u64) {
        self.peak_tracked_bytes = self.peak_tracked_bytes.max(bytes);
    }
}

/// Camera/format bucket the per-stage numbers are aggregated under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RawKind {
    Z6iiiHe,
    Nef,
    Arw,
    XTrans,
    Dng,
    Other,
}

impl RawKind {
    pub fn label(self) -> &'static str {
        match self {
            RawKind::Z6iiiHe => "z6iii-he",
            RawKind::Nef => "nef",
            RawKind::Arw => "arw",
            RawKind::XTrans => "x-trans",
            RawKind::Dng => "dng",
            RawKind::Other => "other",
        }
    }

    fn classify(path: &Path, bayer: Option<&BayerFrame>, model: Option<&str>) -> Self {
        if bayer.is_some_and(|bayer| bayer.filters == 9) {
            return RawKind::XTrans;
        }
        // Nikon writes "NIKON Z 6_3" (and some bodies "Z 6III"); both are HE
        // candidates even when this build has no Z6III reconstruction feature.
        if let Some(model) = model {
            let model = model.to_ascii_lowercase();
            if model.contains("z 6_3") || model.contains("z6iii") || model.contains("z 6iii") {
                return RawKind::Z6iiiHe;
            }
        }
        match path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref()
        {
            Some("nef") => RawKind::Nef,
            Some("arw") => RawKind::Arw,
            Some("dng") => RawKind::Dng,
            _ => RawKind::Other,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RawDecodeAggregate {
    pub decodes: u64,
    pub timings: RawStageTimings,
}

static RAW_DECODE_STATS: std::sync::Mutex<Vec<(RawKind, RawDecodeAggregate)>> =
    std::sync::Mutex::new(Vec::new());

/// Per-format decode timings since the last reset, one entry per kind that ran.
pub fn raw_decode_stats() -> Vec<(RawKind, RawDecodeAggregate)> {
    RAW_DECODE_STATS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

pub fn reset_raw_decode_stats() {
    RAW_DECODE_STATS.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

fn record_raw_decode(kind: RawKind, timings: RawStageTimings) {
    {
        let mut stats = RAW_DECODE_STATS.lock().unwrap_or_else(|e| e.into_inner());
        let slot = match stats.iter_mut().find(|(stored, _)| *stored == kind) {
            Some(slot) => slot,
            None => {
                stats.push((kind, RawDecodeAggregate::default()));
                stats.last_mut().expect("just pushed")
            }
        };
        slot.1.decodes += 1;
        slot.1.timings.add(&timings);
    }
    log::debug!(
        "raw decode {} bytes={} read={}ms exif={}ms open={}ms unpack={}ms bayer={}ms \
         z6iii={}ms demosaic={}ms c1={}ms fallback={}ms total={}ms peak={}B",
        kind.label(),
        timings.bytes_read,
        timings.read_ms,
        timings.exif_ms,
        timings.libraw_open_ms,
        timings.libraw_unpack_ms,
        timings.bayer_ms,
        timings.z6iii_ms,
        timings.demosaic_ms,
        timings.c1_stages_ms,
        timings.fallback_ms,
        timings.total_ms,
        timings.peak_tracked_bytes
    );
}

/// Which decoder produced a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeBackend {
    /// Z6III HE reconstruction through the OpenCL pipeline.
    AuroraZ6iiiOpenCl,
    /// Aurora Raw's CPU C1 chain (Z6III fallback or generic Bayer / X-Trans).
    AuroraCpuChain,
    /// LibRaw demosaic.
    LibRaw,
    /// The camera's embedded JPEG, already developed.
    CameraPreview,
    /// A non-RAW path (HEIF, JPEG, `.cop`, ...).
    Other,
}

/// Colour space of the decoded samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelSpace {
    /// C1 camera-linear 16-bit samples from the Aurora Raw chain.
    CameraLinear16,
    /// LibRaw's camera-linear 16-bit samples.
    LibRawLinear16,
    /// The camera-embedded preview, already developed.
    EmbeddedPreview,
    /// Display-encoded 8-bit samples widened to [`RgbaImage`].
    Srgb8,
}

/// Outcome of the Z6III OpenCL attempt for one decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenClStatus {
    NotAttempted,
    Succeeded,
    Unavailable,
    Failed,
}

/// What actually produced a decoded frame.
///
/// `z6iii_linear` is kept separate from [`PixelSpace::CameraLinear16`] because
/// generic Bayer / X-Trans frames are also camera-linear but must not receive
/// the Z6III pre-matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DecodeProvenance {
    /// [`DecodeStageRevisions::provenance_fingerprint`] of the stages that
    /// produced these pixels. Compared on every cache hit.
    pub pipeline_fingerprint: u64,
    pub backend: DecodeBackend,
    pub opencl: OpenClStatus,
    pub pixel_space: PixelSpace,
    /// Samples need the C1 Z6III pre-matrix.
    pub z6iii_linear: bool,
}

impl DecodeProvenance {
    fn with_fingerprint(mut self) -> Self {
        self.pipeline_fingerprint = DECODE_STAGE_REVISIONS.provenance_fingerprint(&self);
        self
    }

    pub fn aurora(route: crate::aurora_raw::AuroraRoute) -> Self {
        use crate::aurora_raw::AuroraRoute as Route;
        let (backend, opencl, z6iii_linear) = match route {
            Route::Z6iiiOpenCl => {
                (DecodeBackend::AuroraZ6iiiOpenCl, OpenClStatus::Succeeded, true)
            }
            Route::Z6iiiCpuOpenClUnavailable => {
                (DecodeBackend::AuroraCpuChain, OpenClStatus::Unavailable, true)
            }
            Route::Z6iiiCpuOpenClFailed => {
                (DecodeBackend::AuroraCpuChain, OpenClStatus::Failed, true)
            }
            Route::Cpu => (DecodeBackend::AuroraCpuChain, OpenClStatus::NotAttempted, false),
        };
        Self {
            pipeline_fingerprint: 0,
            backend,
            opencl,
            pixel_space: PixelSpace::CameraLinear16,
            z6iii_linear,
        }
        .with_fingerprint()
    }

    pub fn libraw16() -> Self {
        Self {
            pipeline_fingerprint: 0,
            backend: DecodeBackend::LibRaw,
            opencl: OpenClStatus::NotAttempted,
            pixel_space: PixelSpace::LibRawLinear16,
            z6iii_linear: false,
        }
        .with_fingerprint()
    }

    pub fn libraw_srgb8() -> Self {
        Self {
            pipeline_fingerprint: 0,
            backend: DecodeBackend::LibRaw,
            opencl: OpenClStatus::NotAttempted,
            pixel_space: PixelSpace::Srgb8,
            z6iii_linear: false,
        }
        .with_fingerprint()
    }

    pub fn camera_preview() -> Self {
        Self {
            pipeline_fingerprint: 0,
            backend: DecodeBackend::CameraPreview,
            opencl: OpenClStatus::NotAttempted,
            pixel_space: PixelSpace::EmbeddedPreview,
            z6iii_linear: false,
        }
        .with_fingerprint()
    }

    pub fn other_srgb8() -> Self {
        Self {
            pipeline_fingerprint: 0,
            backend: DecodeBackend::Other,
            opencl: OpenClStatus::NotAttempted,
            pixel_space: PixelSpace::Srgb8,
            z6iii_linear: false,
        }
        .with_fingerprint()
    }

    /// The only condition that may enable the C1 Z6III pre-matrix.
    pub const fn unlocks_c1_z6iii_linear(&self) -> bool {
        self.z6iii_linear && matches!(self.pixel_space, PixelSpace::CameraLinear16)
    }

    /// Stable byte codes persisted in the RAW16 cache header.
    pub const fn backend_code(&self) -> u8 {
        match self.backend {
            DecodeBackend::AuroraZ6iiiOpenCl => 0,
            DecodeBackend::AuroraCpuChain => 1,
            DecodeBackend::LibRaw => 2,
            DecodeBackend::CameraPreview => 3,
            DecodeBackend::Other => 4,
        }
    }

    pub const fn opencl_code(&self) -> u8 {
        match self.opencl {
            OpenClStatus::NotAttempted => 0,
            OpenClStatus::Succeeded => 1,
            OpenClStatus::Unavailable => 2,
            OpenClStatus::Failed => 3,
        }
    }

    pub const fn pixel_space_code(&self) -> u8 {
        match self.pixel_space {
            PixelSpace::CameraLinear16 => 0,
            PixelSpace::LibRawLinear16 => 1,
            PixelSpace::EmbeddedPreview => 2,
            PixelSpace::Srgb8 => 3,
        }
    }

    pub const fn flags(&self) -> u8 {
        self.z6iii_linear as u8
    }

    /// Inverse of the cache header codes. Unknown codes reject the entry.
    pub fn from_codes(
        pipeline_fingerprint: u64,
        backend: u8,
        opencl: u8,
        pixel_space: u8,
        flags: u8,
    ) -> Option<Self> {
        if flags & !1 != 0 {
            return None;
        }
        Some(Self {
            pipeline_fingerprint,
            backend: match backend {
                0 => DecodeBackend::AuroraZ6iiiOpenCl,
                1 => DecodeBackend::AuroraCpuChain,
                2 => DecodeBackend::LibRaw,
                3 => DecodeBackend::CameraPreview,
                4 => DecodeBackend::Other,
                _ => return None,
            },
            opencl: match opencl {
                0 => OpenClStatus::NotAttempted,
                1 => OpenClStatus::Succeeded,
                2 => OpenClStatus::Unavailable,
                3 => OpenClStatus::Failed,
                _ => return None,
            },
            pixel_space: match pixel_space {
                0 => PixelSpace::CameraLinear16,
                1 => PixelSpace::LibRawLinear16,
                2 => PixelSpace::EmbeddedPreview,
                3 => PixelSpace::Srgb8,
                _ => return None,
            },
            z6iii_linear: flags & 1 != 0,
        })
    }

    /// Whether these pixels were produced by the current stage revisions.
    ///
    /// Recomputed from the stage constants instead of trusting the header, so a
    /// header written before a stage bump is rejected even if it was copied
    /// under a newer key.
    pub fn is_current(&self) -> bool {
        self.pipeline_fingerprint == DECODE_STAGE_REVISIONS.provenance_fingerprint(self)
    }

    /// Rejects a cached entry whose provenance cannot belong to `route`.
    pub fn consistent_with(&self, route: &DecodeRoute) -> bool {
        self.is_current()
            && !(self.z6iii_linear && !route.z6iii_nef)
            && !(matches!(self.pixel_space, PixelSpace::CameraLinear16)
                && !route.c1_raw_enabled)
    }
}

/// Inputs that decide which backend [`decode16_traced`] selects *before* a
/// decode runs. Any change here must produce a different cache entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecodeRoute {
    /// [`DecodeStageRevisions::pipeline_fingerprint`] for this route's stage
    /// mix. Combined with the other fields by the cache key.
    pub pipeline_fingerprint: u64,
    pub c1_raw_enabled: bool,
    /// The file is a recognisable Z6III HE NEF and C1 RAW is enabled.
    pub z6iii_nef: bool,
    /// This build contains a Z6III reconstruction: OpenCL on Windows with the
    /// feature, the CPU C1 chain on every other platform.
    pub z6iii_feature: bool,
    /// The libraw shared library resolves in this process.
    pub libraw_available: bool,
    pub opencl: OpenClRouteState,
    pub demosaic: i32,
}

/// Known OpenCL capability for the current build/machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpenClRouteState {
    /// This build cannot run the OpenCL reconstruction.
    NotCompiled,
    /// The build supports it and a compatible device was found.
    Available,
    /// The build supports it but no compatible device is present.
    Unavailable,
}

/// The route inputs for `path`, suitable for hashing into a cache key.
pub fn decode_route(path: &Path) -> DecodeRoute {
    let c1_raw_enabled = c1_raw_enabled();
    // The CPU C1 reconstruction is compiled on every platform; only the OpenCL
    // backend and the desktop export helpers are Windows-only. `opencl` records
    // that separately, so a macOS Z6III route can still take the Z6III stage
    // shape without pretending an OpenCL stage exists.
    let z6iii_feature = !cfg!(windows) || cfg!(feature = "z6iii-raw-export");
    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    let z6iii_nef = c1_raw_enabled
        && z6iii_feature
        && path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("nef"))
        && crate::z6iii::is_z6iii_nef_file(path);
    #[cfg(not(all(windows, feature = "z6iii-raw-export")))]
    let z6iii_nef = c1_raw_enabled
        && z6iii_feature
        && path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("nef"))
        && crate::aurora_raw::c1_nef_he::is_z6iii_nef_file(path);
    let mut route = DecodeRoute {
        pipeline_fingerprint: 0,
        c1_raw_enabled,
        z6iii_nef,
        z6iii_feature,
        libraw_available: libraw::available(),
        opencl: opencl_route_state(),
        demosaic: demosaic_quality(),
    };
    route.pipeline_fingerprint = DECODE_STAGE_REVISIONS.pipeline_fingerprint(&route);
    route
}

fn opencl_route_state() -> OpenClRouteState {
    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    {
        static STATE: std::sync::OnceLock<OpenClRouteState> = std::sync::OnceLock::new();
        *STATE.get_or_init(|| {
            if crate::z6iii::opencl_available() {
                OpenClRouteState::Available
            } else {
                OpenClRouteState::Unavailable
            }
        })
    }
    #[cfg(not(all(windows, feature = "z6iii-raw-export")))]
    {
        OpenClRouteState::NotCompiled
    }
}

/// Provenance for the 8-bit fallback in the file engine, after [`decode`] ran.
pub fn decode_fallback_provenance(path: &Path) -> DecodeProvenance {
    match from_sensor(path) {
        Some(false) => DecodeProvenance::camera_preview(),
        _ if is_raw(path) => DecodeProvenance::libraw_srgb8(),
        _ => DecodeProvenance::other_srgb8(),
    }
}

pub(crate) fn c1_raw_enabled() -> bool {
    !std::env::var("AURORA_C1_RAW").is_ok_and(|value| value == "0")
}

/// A LibRaw Bayer plane plus the metadata the stages after it need.
///
/// `data` holds LibRaw's stored sample values (12/14/16-bit).  Capture One's
/// stages run on 16-bit left-aligned samples, so callers that feed the C1 chain
/// use [`BayerFrame::left_aligned`] and [`BayerFrame::black16`].  On the three
/// validated Z6III HE files the C1 decoder and the LibRaw fixture match exactly
/// after that 14->16-bit shift.
#[derive(Clone, Debug)]
pub struct BayerFrame {
    pub raw_width: usize,
    pub raw_height: usize,
    pub width: usize,
    pub height: usize,
    pub top_margin: usize,
    pub left_margin: usize,
    /// Row stride in bytes, including any padding.
    pub stride: usize,
    pub bits: u8,
    pub data: Vec<u16>,
    pub filters: u32,
    pub colors: u32,
    pub black: u32,
    pub maximum: u32,
    pub cam_mul: [f32; 4],
    /// LibRaw stores `rgb_cam` as `[3][4]`; the fourth column is normally zero.
    pub rgb_cam: [[f32; 4]; 3],
}

impl BayerFrame {
    /// Bits needed to left-align this sensor's samples to 16 bits.
    pub fn shift(&self) -> u32 {
        16u32.saturating_sub(u32::from(self.bits))
    }

    /// A copy with every sample left-aligned to 16 bits.
    pub fn left_aligned(&self) -> Vec<u16> {
        let shift = self.shift();
        if shift == 0 {
            return self.data.clone();
        }
        self.data.iter().map(|&value| (u32::from(value) << shift) as u16).collect()
    }

    /// The black level in the same 16-bit scale as [`Self::left_aligned`].
    pub fn black16(&self) -> u32 {
        self.black << self.shift()
    }

    /// One row of the raw (not left-aligned) plane.
    pub fn row(&self, y: usize) -> Option<&[u16]> {
        let start = y.checked_mul(self.raw_width)?;
        self.data.get(start..start + self.raw_width)
    }

    /// CFA colour at a raw-origin sample (0=R, 1=G, 2=B, 3=G2).
    pub fn color_at(&self, x: usize, y: usize) -> u8 {
        ((self.filters >> ((((y << 1) & 14) + (x & 1)) << 1)) & 3) as u8
    }
}

#[derive(Debug)]
pub enum BayerError {
    /// LibRaw's `LIBRAW_FILE_UNSUPPORTED` (-2).  Nikon HE/HE* lands here; the
    /// caller falls back to the C1 HE decoder.
    Unsupported,
    Other(String),
}

impl std::fmt::Display for BayerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported => write!(f, "libraw does not support this RAW compression"),
            Self::Other(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for BayerError {}

/// Decodes the sensor plane through LibRaw without demosaicing.
///
/// This is the front-end for bodies whose container LibRaw understands
/// (classic NEF, ARW, CR3).  Nikon HE/HE* returns [`BayerError::Unsupported`];
/// use [`crate::aurora_raw::decode_he_rgba16`] for those.
pub fn decode_bayer16(path: &Path) -> Result<BayerFrame, BayerError> {
    decode_bayer16_with(path, libraw::unpack_bayer)
}

fn libraw_forced() -> bool {
    std::env::var("AURORA_USE_LIBRAW").is_ok_and(|value| value == "1")
}

fn libraw_forbidden() -> bool {
    std::env::var("AURORA_NO_LIBRAW").is_ok_and(|value| value == "1")
}

/// The sensor frame the product 16-bit route feeds to the C1 chain.
///
/// LibRaw's unpacked frame keeps priority. When no LibRaw frame exists (library
/// missing, container unsupported, or unpack failed) the pure-Rust native
/// front-end decodes the bytes the caller already read, so the product path
/// gains DNG/ARW/CR2/CR3/ORF/RW2/RAF/classic-NEF support without a second disk
/// read. `AURORA_USE_LIBRAW=1` still forbids the native front-end.
fn effective_bayer<'a>(
    bytes: &[u8],
    libraw_bayer: Option<&'a BayerFrame>,
) -> Option<std::borrow::Cow<'a, BayerFrame>> {
    use std::borrow::Cow;
    if let Some(frame) = libraw_bayer {
        return Some(Cow::Borrowed(frame));
    }
    if libraw_forced() {
        return None;
    }
    match crate::aurora_raw::rust_raw::decode_bytes(bytes) {
        Ok(frame) => {
            raw_trace!(
                "RAWTRACE native product16 {}x{}",
                frame.raw_width,
                frame.raw_height
            );
            Some(Cow::Owned(frame))
        }
        Err(error) => {
            raw_trace!("RAWTRACE native product16: {error}");
            None
        }
    }
}

/// Native front-end switch.
///
/// `AURORA_USE_LIBRAW=1` forces the DLL path; `AURORA_NO_LIBRAW=1` forbids the
/// fallback so an unsupported or corrupt container fails loudly instead of
/// silently depending on the DLL.  With neither set the native decoder runs
/// first and LibRaw covers what it does not reproduce yet.
pub(crate) fn decode_bayer16_with(
    path: &Path,
    libraw: impl FnOnce(&Path) -> Result<BayerFrame, BayerError>,
) -> Result<BayerFrame, BayerError> {
    let forced = libraw_forced();
    let forbidden = libraw_forbidden();
    if !forced {
        match crate::aurora_raw::rust_raw::decode(path) {
            Ok(frame) => {
                raw_trace!("RAWTRACE native {} {}x{}", path.display(), frame.raw_width, frame.raw_height);
                return Ok(frame);
            }
            Err(error) => {
                raw_trace!("RAWTRACE native {}: {error}", path.display());
                if forbidden {
                    return Err(match error {
                        crate::aurora_raw::rust_raw::NativeError::Unsupported(_) => {
                            BayerError::Unsupported
                        }
                        other => BayerError::Other(format!("native RAW: {other}")),
                    });
                }
            }
        }
    }
    libraw(path)
}

/// Whether a sensor decode looks like the same picture as the embedded preview.
///
/// Nikon's High Efficiency compression is not in every libraw build: it returns
/// stripes instead of an error, and stripes must not reach the user. Comparing
/// 32x32 luminance against the camera's own JPEG catches that and any future
/// decoder that silently mis-reads a file. A preview that cannot be decoded is
/// not evidence of anything, so the sensor decode is kept then.
fn matches_preview(decoded: &RgbaImage, path: &Path) -> bool {
    same_picture(&luma_grid(decoded), path)
}
fn same_picture(a: &[f32], path: &Path) -> bool {
    let Ok(preview) = embedded_preview(path) else {
        return true;
    };
    similar(a, &luma_grid(&preview))
}
fn same_picture_bytes(a: &[f32], bytes: &[u8]) -> bool {
    match embedded_preview_bytes(bytes) {
        Some(preview) => similar(a, &luma_grid(&preview)),
        None => true,
    }
}
fn similar(a: &[f32], preview: &[f32]) -> bool {
    // Rank correlation of 2x2-averaged cells: the sensor decode is linear light
    // and the preview is display-encoded and tone-mapped, so the same picture can
    // differ by a whole exposure; ranks ignore that, averaging ignores the noise
    // that made a night shot at ISO 5000 score 0.82. A stripe decode still scores
    // near zero.
    let b = ranks(&halve(preview));
    let mut g = ranks(&halve(a));
    // libraw hands over the sensor in its stored orientation while the camera's
    // preview is already turned; any quarter turn is still the same picture.
    for _ in 0..4 {
        if correlation(&g, &b) > 0.90 {
            return true;
        }
        g = transpose16(&g);
    }
    false
}
/// 2x2 box average of a 32x32 luma grid.
fn halve(grid: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; 16 * 16];
    for y in 0..16 {
        for x in 0..16 {
            out[y * 16 + x] = (grid[(2 * y) * 32 + 2 * x]
                + grid[(2 * y) * 32 + 2 * x + 1]
                + grid[(2 * y + 1) * 32 + 2 * x]
                + grid[(2 * y + 1) * 32 + 2 * x + 1])
                / 4.0;
        }
    }
    out
}
/// Values replaced by their rank, so a monotone tone curve cannot change them.
fn ranks(grid: &[f32]) -> Vec<f32> {
    let mut order: Vec<usize> = (0..grid.len()).collect();
    order.sort_by(|&i, &j| grid[i].total_cmp(&grid[j]));
    let mut out = vec![0.0; grid.len()];
    let mut start = 0;
    while start < order.len() {
        // Ties share their average rank, so a flat frame cannot masquerade as a
        // gradient just because the sort happened to be stable.
        let mut end = start;
        while end + 1 < order.len() && grid[order[end + 1]] == grid[order[start]] {
            end += 1;
        }
        let shared = (start + end) as f32 / 2.0;
        for &i in &order[start..=end] {
            out[i] = shared;
        }
        start = end + 1;
    }
    out
}
/// 90-degree turn of a 16x16 grid.
fn transpose16(grid: &[f32]) -> Vec<f32> {
    const N: usize = 16;
    let mut out = vec![0.0; grid.len()];
    for y in 0..N {
        for x in 0..N {
            out[y * N + x] = grid[x * N + (N - 1 - y)];
        }
    }
    out
}
/// Pearson correlation of two equally sized grids.
fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()).max(1) as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (dx, dy) = (x - ma, y - mb);
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    cov / (va.sqrt() * vb.sqrt()).max(1e-6)
}
/// 90-degree turn of a 32x32 grid.
fn transpose(grid: &[f32]) -> Vec<f32> {
    const N: usize = 32;
    let mut out = vec![0.0; grid.len()];
    for y in 0..N {
        for x in 0..N {
            out[y * N + x] = grid[x * N + (N - 1 - y)];
        }
    }
    out
}
/// 32x32 luminance of a frame, sampled on a grid, normalised to 0..=1.
fn luma_grid(img: &RgbaImage) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let mut out = Vec::with_capacity(32 * 32);
    for gy in 0..32u32 {
        for gx in 0..32u32 {
            let x = ((gx * 2 + 1) * w / 64).min(w.saturating_sub(1));
            let y = ((gy * 2 + 1) * h / 64).min(h.saturating_sub(1));
            let p = img.get_pixel(x, y);
            out.push(0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32);
        }
    }
    let max = out.iter().cloned().fold(0.0f32, f32::max).max(1.0);
    out.iter().map(|v| v / max).collect()
}
/// The same grid over 16-bit linear RGBA, sRGB-encoded per sample so it is
/// comparable with the camera preview's own grid.
fn luma_grid16(w: u32, h: u32, px: &[u16]) -> Vec<f32> {
    let mut out = Vec::with_capacity(32 * 32);
    for gy in 0..32u32 {
        for gx in 0..32u32 {
            let x = ((gx * 2 + 1) * w / 64).min(w.saturating_sub(1));
            let y = ((gy * 2 + 1) * h / 64).min(h.saturating_sub(1));
            let i = ((y as usize) * w as usize + x as usize) * 4;
            let p = [px[i], px[i + 1], px[i + 2]];
            let e = |v: u16| srgb_f(v as f32 / 65535.0);
            out.push(0.2126 * e(p[0]) + 0.7152 * e(p[1]) + 0.0722 * e(p[2]));
        }
    }
    let max = out.iter().cloned().fold(0.0f32, f32::max).max(1.0);
    out.iter().map(|v| v / max).collect()
}
/// True if the extension is a RAW type we can take a preview from.
pub fn is_raw(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "iiq"
            | "nef"
            | "cr3"
            | "cr2"
            | "arw"
            | "dng"
            | "raf"
            | "orf"
            | "rw2"
            | "pef"
            | "srw"
            | "3fr"
            | "fff"
            | "mos"
    )
}

/// Decoded dimensions of the embedded preview, for catalog metadata.
///
/// Only the JPEG headers are read, so this is cheap enough to run over a whole folder.
pub fn preview_dimensions(path: &Path) -> Option<(u32, u32)> {
    scan_best(path, |segment| {
        let reader =
            image::ImageReader::with_format(std::io::Cursor::new(segment), ImageFormat::Jpeg);
        let (w, h) = reader.into_dimensions().ok()?;
        if w == 0 || h == 0 {
            return None;
        }
        Some((u64::from(w) * u64::from(h), (w, h)))
    })
    .ok()
    .flatten()
}

/// Dimensions of the decoded sensor data, or the preview's when libraw has no answer.
///
/// libraw reports the sensor size unrotated (6064x4040) while its output follows the
/// camera's orientation (4040x6064), so the embedded preview settles which one applies.
pub fn sensor_dimensions(path: &Path) -> Option<(u32, u32)> {
    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    if c1_raw_enabled() && path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("nef")) {
        if let Some(size) = crate::z6iii::active_size(path) {
            return Some(size);
        }
    }
    if let Some((width, height)) = libraw::dimensions(path) {
        // libraw rotates the pixels but reports the sensor size unrotated.
        return Some(match orientation(path) {
            Some(5..=8) => (height, width),
            _ => (width, height),
        });
    }
    preview_dimensions(path)
}

/// EXIF orientation tag of a TIFF-based RAW (IFD0, tag 0x0112).
fn orientation(path: &Path) -> Option<u16> {
    exif_orientation_tag(path)
}

/// Clockwise rotation in degrees that the camera's EXIF orientation implies.
///
/// libraw applies this rotation to the decoded pixels, so anything that also
/// stores the display rotation (Capture One's `ZROTATION`) has to subtract it
/// to avoid turning the image twice.
pub fn exif_rotation_deg(path: &Path) -> u16 {
    match exif_orientation_tag(path) {
        Some(3) => 180,
        Some(6) => 90,
        Some(8) => 270,
        _ => 0,
    }
}

fn exif_orientation_tag(path: &Path) -> Option<u16> {
    let bytes = std::fs::read(path).ok()?;
    exif_orientation_bytes(&bytes)
}

pub(crate) fn exif_orientation_bytes(bytes: &[u8]) -> Option<u16> {
    if bytes.len() < 8 {
        return None;
    }
    let little = &bytes[..2] == b"II";
    let u16at = |o: usize| {
        let s = bytes.get(o..o + 2)?;
        Some(if little { u16::from_le_bytes([s[0], s[1]]) } else { u16::from_be_bytes([s[0], s[1]]) })
    };
    let u32at = |o: usize| {
        let s = bytes.get(o..o + 4)?;
        Some(if little { u32::from_le_bytes([s[0], s[1], s[2], s[3]]) } else { u32::from_be_bytes([s[0], s[1], s[2], s[3]]) })
    };
    if &bytes[..4] != b"II\x2a\x00" && &bytes[..4] != b"MM\x00\x2a" {
        return None;
    }
    let ifd = u32at(4)? as usize;
    let count = u16at(ifd)? as usize;
    for index in 0..count {
        let entry = ifd + 2 + index * 12;
        if u16at(entry)? == 0x0112 {
            return u16at(entry + 8);
        }
    }
    None
}

mod libraw {
    use super::*;
    use std::ffi::{c_char, c_int, c_void, CString};
    use std::path::PathBuf;
    use std::sync::OnceLock;

    const RTLD_NOW: c_int = 2;

    #[cfg(target_os = "windows")]
    const LIBRARY_NAMES: [&str; 1] = ["libraw.dll\0"];
    /// macOS ships the ABI in a versioned dylib (`libraw.25.dylib` for 0.22.x);
    /// newest first, then the unversioned symlink Homebrew also creates.
    #[cfg(target_os = "macos")]
    const POSIX_LIBRARY_NAMES: [&str; 4] =
        ["libraw.25.dylib", "libraw.24.dylib", "libraw.23.dylib", "libraw.dylib"];
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    const POSIX_LIBRARY_NAMES: [&str; 2] = ["libraw.so.23", "libraw.so"];
    /// Directories `dlopen` does not search on its own. Homebrew on Apple
    /// silicon installs into `/opt/homebrew/lib`; Intel and source builds use
    /// `/usr/local/lib`. Linux gets the common multi-arch directories.
    #[cfg(target_os = "macos")]
    const LIBRARY_PREFIXES: [&str; 2] = ["/opt/homebrew/lib", "/usr/local/lib"];
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    const LIBRARY_PREFIXES: [&str; 3] =
        ["/usr/local/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib"];

    unsafe extern "C" {
        #[cfg(not(target_os = "windows"))]
        fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
        #[cfg(not(target_os = "windows"))]
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        #[cfg(target_os = "windows")]
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        #[cfg(target_os = "windows")]
        fn GetProcAddress(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        #[cfg(target_os = "windows")]
        fn SetDllDirectoryW(path: *const u16) -> i32;
    }

    /// `libraw_processed_image_t`; `data` is a byte cursor, not a real array.
    #[repr(C)]
    struct Processed {
        kind: c_int,
        height: u16,
        width: u16,
        colors: u16,
        bits: u16,
        data_size: u32,
        data: [u8; 1],
    }

    struct Api {
        init: unsafe extern "C" fn(u32) -> *mut c_void,
        open: unsafe extern "C" fn(*mut c_void, *const c_char) -> c_int,
        #[cfg(target_os = "windows")]
        open_w: unsafe extern "C" fn(*mut c_void, *const u16) -> c_int,
        /// Reads from the buffer we already hold instead of opening the path a
        /// second time. Optional: older builds fall back to `open_file`.
        open_buffer: Option<unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> c_int>,
        unpack_name: unsafe extern "C" fn(*mut c_void) -> *const c_char,
        unpack: unsafe extern "C" fn(*mut c_void) -> c_int,
        process: unsafe extern "C" fn(*mut c_void) -> c_int,
        make: unsafe extern "C" fn(*mut c_void, *mut c_int) -> *mut Processed,
        clear: unsafe extern "C" fn(*mut Processed),
        close: unsafe extern "C" fn(*mut c_void) -> c_int,
        output_bps: unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        no_auto_bright: unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        output_color: unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        gamma: unsafe extern "C" fn(*mut c_void, c_int, c_int),
        /// Returns `void` on libraw 0.23 (verified against the exported symbol:
        /// no return value, no validation) even though older headers declare int.
        demosaic: unsafe extern "C" fn(*mut c_void, c_int),
        iwidth: unsafe extern "C" fn(*mut c_void) -> c_int,
        iheight: unsafe extern "C" fn(*mut c_void) -> c_int,
        raw_width: unsafe extern "C" fn(*mut c_void) -> c_int,
        raw_height: unsafe extern "C" fn(*mut c_void) -> c_int,
        strerror: unsafe extern "C" fn(c_int) -> *const c_char,
        version_number: unsafe extern "C" fn() -> c_int,
    }

    /// Releases the libraw handle and the decoded pixels even on an early return.
    struct Guard {
        api: &'static Api,
        data: *mut c_void,
        pixels: *mut Processed,
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            unsafe {
                if !self.pixels.is_null() {
                    (self.api.clear)(self.pixels);
                }
                (self.api.close)(self.data);
            }
        }
    }

    fn api() -> Option<&'static Api> {
        api_or_error().ok()
    }

    /// The load result, including why the library is unavailable. The reason is
    /// cached with the API so a failed `dlopen` is not retried per decode.
    fn api_or_error() -> Result<&'static Api, String> {
        static API: OnceLock<Result<Api, String>> = OnceLock::new();
        API.get_or_init(load).as_ref().map_err(|message| message.clone())
    }

    /// Whether the shared library resolved. Used by the decode-route fingerprint.
    pub(super) fn available() -> bool {
        api().is_some()
    }

    /// The loaded library's version number, or `None` when it is absent.
    /// Exposed so tests can exercise real discovery without a RAW fixture.
    pub(super) fn version() -> Option<i32> {
        let api = api()?;
        Some(unsafe { (api.version_number)() })
    }

    fn load() -> Result<Api, String> {
        // The handle stays loaded for the process lifetime: the function pointers
        // below would dangle after a `dlclose`.
        //
        // Windows keeps its DLL search (vendor folder then PATH). Every other
        // platform walks an explicit candidate list because `dlopen` does not
        // search Homebrew's prefix on its own.
        #[cfg(target_os = "windows")]
        let (handle, loaded_from) = LIBRARY_NAMES
            .into_iter()
            .find_map(|name| {
                let handle = load_library(name);
                (!handle.is_null()).then(|| (handle, name.trim_end_matches('\0').to_string()))
            })
            .ok_or_else(missing_library_error)?;
        #[cfg(not(target_os = "windows"))]
        let (handle, loaded_from) = {
            let candidates = library_candidates();
            let mut loaded = None;
            for candidate in &candidates {
                let handle = load_library(candidate);
                if !handle.is_null() {
                    loaded = Some((handle, candidate.display().to_string()));
                    break;
                }
            }
            loaded.ok_or_else(|| missing_library_error(&candidates))?
        };
        build_api(handle, &loaded_from)
    }

    fn build_api(handle: *mut c_void, loaded_from: &str) -> Result<Api, String> {
        macro_rules! required {
            ($name:literal) => {
                symbol(handle, $name).ok_or_else(|| {
                    format!("libraw loaded from {loaded_from} is missing export {}", $name)
                })?
            };
        }
        Ok(Api {
            init: required!("libraw_init"),
            open: required!("libraw_open_file"),
            #[cfg(target_os = "windows")]
            open_w: required!("libraw_open_wfile"),
            open_buffer: symbol(handle, "libraw_open_buffer"),
            unpack_name: required!("libraw_unpack_function_name"),
            unpack: required!("libraw_unpack"),
            process: required!("libraw_dcraw_process"),
            make: required!("libraw_dcraw_make_mem_image"),
            clear: required!("libraw_dcraw_clear_mem"),
            close: required!("libraw_close"),
            output_bps: required!("libraw_set_output_bps"),
            no_auto_bright: required!("libraw_set_no_auto_bright"),
            output_color: required!("libraw_set_output_color"),
            gamma: required!("libraw_set_gamma"),
            demosaic: required!("libraw_set_demosaic"),
            iwidth: required!("libraw_get_iwidth"),
            iheight: required!("libraw_get_iheight"),
            raw_width: required!("libraw_get_raw_width"),
            raw_height: required!("libraw_get_raw_height"),
            strerror: required!("libraw_strerror"),
            version_number: required!("libraw_versionNumber"),
        })
    }

    /// Empty result when no candidate is present, never a panic.
    #[cfg(target_os = "windows")]
    fn missing_library_error() -> String {
        format!(
            "libraw library not found; searched: {}",
            LIBRARY_NAMES
                .iter()
                .map(|name| name.trim_end_matches('\0'))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    #[cfg(not(target_os = "windows"))]
    pub(super) fn missing_library_error(candidates: &[PathBuf]) -> String {
        format!(
            "libraw library not found; searched: {}",
            candidates
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    /// The library names to try, and the directories `dlopen` will not search
    /// by itself. A caller-set `AURORA_LIBRAW_LIB` is tried before any of them.
    #[cfg(not(target_os = "windows"))]
    fn library_candidates() -> Vec<PathBuf> {
        let override_path = std::env::var_os("AURORA_LIBRAW_LIB").map(PathBuf::from);
        candidate_paths(override_path.as_deref())
    }

    /// Pure, deterministic and de-duplicated search order: the explicit
    /// override first, then each known directory crossed with the versioned
    /// names, then the bare sonames for the platform loader's own path.
    #[cfg(not(target_os = "windows"))]
    pub(super) fn candidate_paths(override_path: Option<&Path>) -> Vec<PathBuf> {
        let mut candidates = Vec::new();
        if let Some(path) = override_path {
            candidates.push(path.to_path_buf());
        }
        for prefix in LIBRARY_PREFIXES {
            for name in POSIX_LIBRARY_NAMES {
                candidates.push(Path::new(prefix).join(name));
            }
        }
        for name in POSIX_LIBRARY_NAMES {
            candidates.push(PathBuf::from(name));
        }
        let mut seen = std::collections::HashSet::new();
        candidates.into_iter().filter(|path| seen.insert(path.clone())).collect()
    }

    #[cfg(target_os = "windows")]
    fn load_library(name: &str) -> *mut c_void {
        // LoadLibraryW searches the application directory and PATH but not a
        // project's vendor folder, so that folder goes on the search path first.
        // The name itself must be UTF-16: passing str bytes as u16 never loads.
        use std::os::windows::ffi::OsStrExt;
        let wide_name: Vec<u16> =
            name.trim_end_matches('\0').encode_utf16().chain(std::iter::once(0)).collect();
        let mut dirs: Vec<std::path::PathBuf> = Vec::new();
        if let Some(dir) = std::env::var_os("AURORA_LIBRAW_DIR") {
            dirs.push(dir.into());
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                dirs.push(parent.join("libraw"));
                dirs.push(parent.join("vendor/libraw/bin"));
                dirs.push(parent.join("../vendor/libraw/bin"));
                dirs.push(parent.join("../../vendor/libraw/bin"));
            }
        }
        // A dev build keeps the DLLs in the repository's vendor folder.
        dirs.push(std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../vendor/libraw/bin")));
        dirs.push(std::path::PathBuf::from("vendor/libraw/bin"));
        dirs.push(std::path::PathBuf::from("../vendor/libraw/bin"));
        for dir in dirs {
            let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
            unsafe { SetDllDirectoryW(wide.as_ptr()) };
            let handle = unsafe { LoadLibraryW(wide_name.as_ptr()) };
            if !handle.is_null() {
                unsafe { SetDllDirectoryW(std::ptr::null()) };
                return handle;
            }
        }
        unsafe { SetDllDirectoryW(std::ptr::null()) };
        std::ptr::null_mut()
    }

    #[cfg(not(target_os = "windows"))]
    fn load_library(path: &Path) -> *mut c_void {
        // `dlopen` needs a NUL-terminated path; a path with an interior NUL
        // cannot name a library, so treat it as "not found" instead of
        // aborting.
        let Ok(name) = CString::new(path.to_string_lossy().as_bytes()) else {
            return std::ptr::null_mut();
        };
        unsafe { dlopen(name.as_ptr() as *const c_char, RTLD_NOW) }
    }

    /// # Safety
    /// `handle` must come from a successful library load, and `name` must match `T`.
    fn symbol<T>(handle: *mut c_void, name: &str) -> Option<T> {
        let name = CString::new(name).ok()?;
        #[cfg(target_os = "windows")]
        let ptr = unsafe { GetProcAddress(handle, name.as_ptr()) };
        #[cfg(not(target_os = "windows"))]
        let ptr = unsafe { dlsym(handle, name.as_ptr()) };
        (!ptr.is_null()).then(|| unsafe { std::mem::transmute_copy(&ptr) })
    }

    /// The argument type `libraw_open_wfile` / `libraw_open_file` expects.
    #[cfg(target_os = "windows")]
    type LoadPath = Vec<u16>;
    #[cfg(not(target_os = "windows"))]
    type LoadPath = CString;

    fn start(path: &Path) -> Result<(*mut c_void, &'static Api, LoadPath), String> {
        let api = api_or_error()?;
        #[cfg(target_os = "windows")]
        let path: LoadPath = {
            use std::os::windows::ffi::OsStrExt;
            path.as_os_str().encode_wide().chain(std::iter::once(0)).collect::<Vec<u16>>()
        };
        #[cfg(not(target_os = "windows"))]
        let path: LoadPath =
            CString::new(path.to_string_lossy().as_bytes()).map_err(|_| "path has a NUL")?;
        let data = unsafe { (api.init)(0) };
        if data.is_null() {
            return Err("libraw_init returned null".into());
        }
        Ok((data, api, path))
    }

    /// Output settings shared by every libraw decode. `processed` and the
    /// reused [`Session`] must apply exactly the same ones or their pixels differ.
    fn apply_output_settings(api: &Api, data: *mut c_void, quality: i32) {
        unsafe {
            (api.output_bps)(data, 16);
            (api.no_auto_bright)(data, 1);
            // 0 is the camera's own colour space; 1 would be sRGB, and the camera
            // ICC profile in our own pipeline expects the camera's space.
            (api.output_color)(data, 0);
            // Gamma 1 is linear. Anything else would bake in a second transfer
            // curve on top of the one the develop pipeline applies.
            (api.gamma)(data, 1, 1);
            // Must land before dcraw_process; the call is legal right after open.
            (api.demosaic)(data, quality);
        }
    }

    /// libraw opened, unpacked and processed; the memory image is in `pixels`.
    fn processed(path: &Path, quality: i32) -> Result<Guard, String> {
        let (data, api, path) = start(path)?;
        let mut guard = Guard { api, data, pixels: std::ptr::null_mut() };
        apply_output_settings(api, data, quality);
        #[cfg(target_os = "windows")]
        let code = unsafe { (api.open_w)(data, path.as_ptr()) };
        #[cfg(not(target_os = "windows"))]
        let code = unsafe { (api.open)(data, path.as_ptr()) };
        if code != 0 {
            raw_trace!("RAWTRACE open code {code}");
            return Err(format!("open_file code {code}"));
        }
        let code = unsafe { (api.unpack)(data) };
        if code != 0 {
            let name_ptr = unsafe { (api.unpack_name)(data) };
            let name = if name_ptr.is_null() { "" } else {
                let owned =
                    unsafe { std::ffi::CStr::from_ptr(name_ptr).to_string_lossy().into_owned() };
                Box::leak(owned.into_boxed_str())
            };
            raw_trace!("RAWTRACE unpack function {name:?}");
            raw_trace!("RAWTRACE unpack code {code}");
            return Err(format!("unpack code {code}"));
        }
        let code = unsafe { (api.process)(data) };
        if code != 0 {
            raw_trace!("RAWTRACE process code {code}");
            return Err(format!("dcraw_process code {code}"));
        }
        let mut code = 0;
        let pixels = unsafe { (api.make)(data, &mut code) };
        if pixels.is_null() {
            raw_trace!("RAWTRACE make code {code}");
            return Err(format!("dcraw_make_mem_image code {code}"));
        }
        guard.pixels = pixels;
        unsafe {
            let out = &*pixels;
            raw_trace!(
                "RAWTRACE processed {}x{} colors {} bits {} size {}",
                out.width, out.height, out.colors, out.bits, out.data_size
            );
        }
        Ok(guard)
    }

    /// The memory image, its sample bytes and its dimensions.
    fn view<'a>(guard: &'a Guard) -> Result<(&'a Processed, &'a [u8], (u32, u32)), String> {
        view_ptr(guard.pixels)
    }

    fn view_ptr<'a>(
        pixels: *mut Processed,
    ) -> Result<(&'a Processed, &'a [u8], (u32, u32)), String> {
        let out = unsafe { &*pixels };
        let (width, height) = (u32::from(out.width), u32::from(out.height));
        if width == 0 || height == 0 {
            return Err("empty image".into());
        }
        let samples = width as usize * height as usize * usize::from(out.colors);
        let bytes = match out.bits {
            8 => samples,
            16 => samples * 2,
            bits => return Err(format!("bits {bits} unsupported")),
        };
        if out.data_size as usize != bytes {
            raw_trace!(
                "RAWTRACE view mismatch expected {bytes} got {} dims {}x{} colors {} bits {}",
                out.data_size, out.width, out.height, out.colors, out.bits
            );
            return Err(format!("data_size {} != {bytes}", out.data_size));
        }
        let offset = std::mem::offset_of!(Processed, data);
        let raw =
            unsafe { std::slice::from_raw_parts((pixels as *const u8).add(offset), bytes) };
        Ok((out, raw, (width, height)))
    }

    /// One libraw handle for one decode. Opening, unpacking and the Bayer read
    /// happen once; if the Aurora chain declines, `process16` runs libraw's own
    /// demosaic on the *same* unpacked handle instead of decoding again.
    pub(super) struct Session<'a> {
        api: &'static Api,
        data: *mut c_void,
        /// libraw reads from this buffer for as long as the handle lives.
        _buffer: Option<&'a [u8]>,
        unpacked: bool,
    }

    impl Drop for Session<'_> {
        fn drop(&mut self) {
            unsafe { (self.api.close)(self.data) };
        }
    }

    impl<'a> Session<'a> {
        /// Opens `path`, preferring the bytes the caller already read.
        pub(super) fn open(path: &Path, bytes: &'a [u8], quality: i32) -> Result<Self, String> {
            let api = api_or_error()?;
            if let Some(open_buffer) = api.open_buffer {
                let data = unsafe { (api.init)(0) };
                if data.is_null() {
                    return Err("libraw_init returned null".into());
                }
                apply_output_settings(api, data, quality);
                let code =
                    unsafe { open_buffer(data, bytes.as_ptr() as *const c_void, bytes.len()) };
                if code == 0 {
                    return Ok(Self {
                        api,
                        data,
                        _buffer: Some(bytes),
                        unpacked: false,
                    });
                }
                unsafe { (api.close)(data) };
                raw_trace!("RAWTRACE open_buffer code {code}; retrying as a file");
            }
            let (data, api, path) = start(path)?;
            apply_output_settings(api, data, quality);
            #[cfg(target_os = "windows")]
            let code = unsafe { (api.open_w)(data, path.as_ptr()) };
            #[cfg(not(target_os = "windows"))]
            let code = unsafe { (api.open)(data, path.as_ptr()) };
            if code != 0 {
                unsafe { (api.close)(data) };
                return Err(format!("open_file code {code}"));
            }
            Ok(Self { api, data, _buffer: None, unpacked: false })
        }

        pub(super) fn unpack(&mut self) -> Result<(), super::BayerError> {
            let code = unsafe { (self.api.unpack)(self.data) };
            if code == -2 {
                return Err(super::BayerError::Unsupported);
            }
            if code != 0 {
                return Err(super::BayerError::Other(format!(
                    "libraw unpack {code}: {}",
                    error_text(self.api, code)
                )));
            }
            self.unpacked = true;
            Ok(())
        }

        pub(super) fn bayer(&self) -> Result<super::BayerFrame, super::BayerError> {
            let version = unsafe { (self.api.version_number)() };
            let layout = layout(version)?;
            unsafe { read_bayer(self.api, self.data, layout) }
        }

        /// libraw's own demosaic on this already-unpacked handle.
        pub(super) fn process16(&mut self) -> Result<(u32, u32, Vec<u16>), String> {
            if !self.unpacked {
                return Err("libraw handle is not unpacked".into());
            }
            let api = self.api;
            let code = unsafe { (api.process)(self.data) };
            if code != 0 {
                return Err(format!("dcraw_process code {code}"));
            }
            let mut code = 0;
            let pixels = unsafe { (api.make)(self.data, &mut code) };
            if pixels.is_null() {
                return Err(format!("dcraw_make_mem_image code {code}"));
            }
            let result = view_ptr(pixels).and_then(|(out, raw, (width, height))| {
                if out.bits != 16 {
                    return Err(format!("bits {} where 16 were requested", out.bits));
                }
                expand_raw16(raw, out.colors, width as usize * height as usize)
                    .map(|rgba| (width, height, rgba))
            });
            unsafe { (api.clear)(pixels) };
            result
        }

        /// libraw's own demosaic and sRGB encode on this handle, the same
        /// pipeline as [`processed`] + [`expand_raw8`].
        pub(super) fn process8(&mut self) -> Result<(u32, u32, Vec<u8>), String> {
            if !self.unpacked {
                return Err("libraw handle is not unpacked".into());
            }
            let api = self.api;
            let code = unsafe { (api.process)(self.data) };
            if code != 0 {
                return Err(format!("dcraw_process code {code}"));
            }
            let mut code = 0;
            let pixels = unsafe { (api.make)(self.data, &mut code) };
            if pixels.is_null() {
                return Err(format!("dcraw_make_mem_image code {code}"));
            }
            let result = view_ptr(pixels).and_then(|(out, raw, (width, height))| {
                expand_raw8(raw, out.bits, out.colors, width as usize * height as usize)
                    .map(|rgba| (width, height, rgba))
            });
            unsafe { (api.clear)(pixels) };
            result
        }
    }

    pub fn decode(path: &Path, quality: i32) -> Result<RgbaImage, String> {
        let guard = processed(path, quality)?;
        let (out, raw, (width, height)) = view(&guard)?;
        let rgba = expand_raw8(raw, out.bits, out.colors, width as usize * height as usize)?;
        RgbaImage::from_raw(width, height, rgba).ok_or_else(|| "pixel buffer mismatch".into())
    }

    /// [`decode`] over bytes the caller already read.
    ///
    /// The output settings, unpack and demosaic are the same calls as
    /// [`processed`]; only the container source changes, and `Session::open`
    /// still falls back to opening the path when the build lacks
    /// `libraw_open_buffer` or that call rejects the buffer.
    pub fn decode_bytes(path: &Path, bytes: &[u8], quality: i32) -> Result<RgbaImage, String> {
        let mut session = Session::open(path, bytes, quality)?;
        session.unpack().map_err(|err| err.to_string())?;
        let (width, height, rgba) = session.process8()?;
        RgbaImage::from_raw(width, height, rgba).ok_or_else(|| "pixel buffer mismatch".into())
    }

    /// 16-bit linear RGBA straight from libraw: no transfer curve, no 8-bit
    /// truncation. Alpha is filled opaque because libraw decodes three colours.
    pub fn decode16(path: &Path, quality: i32) -> Result<(u32, u32, Vec<u16>), String> {
        let guard = processed(path, quality)?;
        let (out, raw, (width, height)) = view(&guard)?;
        if out.bits != 16 {
            return Err(format!("bits {} where 16 were requested", out.bits));
        }
        let rgba = expand_raw16(raw, out.colors, width as usize * height as usize)?;
        Ok((width, height, rgba))
    }

    pub fn dimensions(path: &Path) -> Option<(u32, u32)> {
        let (data, api, path) = start(path).ok()?;
        let _guard = Guard { api, data, pixels: std::ptr::null_mut() };
        #[cfg(target_os = "windows")]
        if unsafe { (api.open_w)(data, path.as_ptr()) } != 0 {
            return None;
        }
        #[cfg(not(target_os = "windows"))]
        if unsafe { (api.open)(data, path.as_ptr()) } != 0 {
            return None;
        }
        let width = unsafe { (api.iwidth)(data) };
        let height = unsafe { (api.iheight)(data) };
        (width > 0 && height > 0).then_some((width as u32, height as u32))
    }

    // Offsets into `libraw_data_t` for the x64 layouts we ship.  The sizes
    // block is unchanged between 0.22.3 and the 0.23 snapshot; the colour and
    // rawdata blocks moved by 16 bytes.  `unpack_bayer` checks the two size
    // fields against the C getters so an unknown build fails loudly instead of
    // reading another field.
    const OFF_RAW_PITCH: usize = 24;
    const OFF_RAW_WIDTH: usize = 10;
    const OFF_RAW_HEIGHT: usize = 8;
    const OFF_WIDTH: usize = 14;
    const OFF_HEIGHT: usize = 12;
    const OFF_TOP_MARGIN: usize = 16;
    const OFF_LEFT_MARGIN: usize = 18;
    const OFF_FILTERS: usize = 536;
    const OFF_COLORS: usize = 532;

    #[derive(Clone, Copy)]
    struct Layout {
        black: usize,
        maximum: usize,
        cam_mul: usize,
        rgb_cam: usize,
        raw_bps: usize,
        raw_image: usize,
    }

    /// `work/libraw_bayer/offsets.txt`, generated with the 0.22.3 headers.
    const LAYOUT_22_3: Layout = Layout {
        black: 153_080,
        maximum: 153_088,
        cam_mul: 153_244,
        rgb_cam: 153_372,
        raw_bps: 192_668,
        raw_image: 193_776,
    };

    /// LibRaw 0.23.0 struct layout snapshot.
    const LAYOUT_23_0: Layout = Layout {
        black: 153_096,
        maximum: 153_104,
        cam_mul: 153_260,
        rgb_cam: 153_388,
        raw_bps: 192_684,
        raw_image: 193_808,
    };

    fn layout(version: c_int) -> Result<Layout, super::BayerError> {
        if (0x1700..0x1800).contains(&version) {
            Ok(LAYOUT_23_0)
        } else if (0x1603..0x1700).contains(&version) {
            Ok(LAYOUT_22_3)
        } else {
            Err(super::BayerError::Other(format!(
                "libraw {version:#x} layout is not supported"
            )))
        }
    }

    unsafe fn read_u16(base: *const u8, offset: usize) -> u16 {
        unsafe { *(base.add(offset) as *const u16) }
    }
    unsafe fn read_u32(base: *const u8, offset: usize) -> u32 {
        unsafe { *(base.add(offset) as *const u32) }
    }
    unsafe fn read_i32(base: *const u8, offset: usize) -> i32 {
        unsafe { *(base.add(offset) as *const i32) }
    }
    unsafe fn read_f32(base: *const u8, offset: usize) -> f32 {
        f32::from_bits(unsafe { read_u32(base, offset) })
    }

    fn error_text(api: &Api, code: c_int) -> String {
        let ptr = unsafe { (api.strerror)(code) };
        if ptr.is_null() {
            return format!("code {code}");
        }
        unsafe { std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }

    /// Opens and unpacks without `dcraw_process`, so the returned plane is the
    /// same Bayer data LibRaw hands to its own demosaic.
    pub fn unpack_bayer(path: &Path) -> Result<super::BayerFrame, super::BayerError> {
        let (data, api, path) = start(path).map_err(super::BayerError::Other)?;
        let guard = Guard { api, data, pixels: std::ptr::null_mut() };
        #[cfg(target_os = "windows")]
        let code = unsafe { (api.open_w)(guard.data, path.as_ptr()) };
        #[cfg(not(target_os = "windows"))]
        let code = unsafe { (api.open)(guard.data, path.as_ptr()) };
        if code != 0 {
            return Err(super::BayerError::Other(format!(
                "libraw open {code}: {}",
                error_text(guard.api, code)
            )));
        }
        let code = unsafe { (guard.api.unpack)(guard.data) };
        if code == -2 {
            return Err(super::BayerError::Unsupported);
        }
        if code != 0 {
            return Err(super::BayerError::Other(format!(
                "libraw unpack {code}: {}",
                error_text(guard.api, code)
            )));
        }
        let version = unsafe { (guard.api.version_number)() };
        let layout = layout(version)?;
        unsafe { read_bayer(guard.api, guard.data, layout) }
    }

    unsafe fn read_bayer(
        api: &'static Api,
        data: *mut c_void,
        layout: Layout,
    ) -> Result<super::BayerFrame, super::BayerError> {
        let base = data as *const u8;
        let raw_width = usize::from(unsafe { read_u16(base, OFF_RAW_WIDTH) });
        let raw_height = usize::from(unsafe { read_u16(base, OFF_RAW_HEIGHT) });
        let got_width = unsafe { (api.raw_width)(data) };
        let got_height = unsafe { (api.raw_height)(data) };
        if raw_width == 0
            || raw_height == 0
            || got_width as usize != raw_width
            || got_height as usize != raw_height
        {
            return Err(super::BayerError::Other(format!(
                "libraw layout mismatch: offsets {raw_width}x{raw_height}, getters {got_width}x{got_height}"
            )));
        }
        let width = usize::from(unsafe { read_u16(base, OFF_WIDTH) });
        let height = usize::from(unsafe { read_u16(base, OFF_HEIGHT) });
        let top_margin = usize::from(unsafe { read_u16(base, OFF_TOP_MARGIN) });
        let left_margin = usize::from(unsafe { read_u16(base, OFF_LEFT_MARGIN) });
        if width == 0
            || height == 0
            || left_margin + width > raw_width
            || top_margin + height > raw_height
        {
            return Err(super::BayerError::Other(format!(
                "libraw active area {width}x{height}+{left_margin}+{top_margin} is outside {raw_width}x{raw_height}"
            )));
        }
        let mut stride = unsafe { read_u32(base, OFF_RAW_PITCH) } as usize;
        if stride == 0 {
            stride = raw_width * 2;
        }
        if stride < raw_width * 2 || stride % 2 != 0 {
            return Err(super::BayerError::Other(format!(
                "libraw row stride {stride} cannot hold {raw_width} u16 samples"
            )));
        }
        let raw_image = unsafe { *(base.add(layout.raw_image) as *const *const u16) };
        if raw_image.is_null() {
            return Err(super::BayerError::Other(
                "libraw has no raw_image (4-colour or float sensor is unsupported)".into(),
            ));
        }
        let mut data_out = vec![0u16; raw_width * raw_height];
        let row_samples = stride / 2;
        for y in 0..raw_height {
            let source = unsafe { raw_image.add(y * row_samples) };
            let row = unsafe { std::slice::from_raw_parts(source, raw_width) };
            data_out[y * raw_width..(y + 1) * raw_width].copy_from_slice(row);
        }
        let black = unsafe { read_u32(base, layout.black) };
        let maximum = unsafe { read_u32(base, layout.maximum) };
        let mut bits = unsafe { read_u32(base, layout.raw_bps) as u8 };
        if !(8..=16).contains(&bits) {
            bits = match maximum {
                0..=4095 => 12,
                0..=16383 => 14,
                _ => 16,
            };
        }
        let mut cam_mul = [0.0f32; 4];
        for (index, value) in cam_mul.iter_mut().enumerate() {
            *value = unsafe { read_f32(base, layout.cam_mul + index * 4) };
        }
        let mut rgb_cam = [[0.0f32; 4]; 3];
        for (row, values) in rgb_cam.iter_mut().enumerate() {
            for (column, value) in values.iter_mut().enumerate() {
                *value = unsafe { read_f32(base, layout.rgb_cam + (row * 4 + column) * 4) };
            }
        }
        Ok(super::BayerFrame {
            raw_width,
            raw_height,
            width,
            height,
            top_margin,
            left_margin,
            stride,
            bits,
            data: data_out,
            filters: unsafe { read_u32(base, OFF_FILTERS) },
            colors: unsafe { read_i32(base, OFF_COLORS) }.max(0) as u32,
            black,
            maximum,
            cam_mul,
            rgb_cam,
        })
    }
}

fn expand_raw16(raw: &[u8], colors: u16, pixels: usize) -> Result<Vec<u16>, String> {
    use rayon::prelude::*;

    let mut rgba = vec![0u16; pixels.saturating_mul(4)];
    let read = |p: &[u8]| u16::from_le_bytes([p[0], p[1]]);
    match colors {
        3 => rgba
            .par_chunks_exact_mut(4)
            .zip(raw.par_chunks_exact(6))
            .for_each(|(dst, src)| {
                dst[0] = read(&src[0..2]);
                dst[1] = read(&src[2..4]);
                dst[2] = read(&src[4..6]);
                dst[3] = 65535;
            }),
        4 => rgba
            .par_chunks_exact_mut(4)
            .zip(raw.par_chunks_exact(8))
            .for_each(|(dst, src)| {
                dst[0] = read(&src[0..2]);
                dst[1] = read(&src[2..4]);
                dst[2] = read(&src[4..6]);
                dst[3] = 65535;
            }),
        1 => rgba
            .par_chunks_exact_mut(4)
            .zip(raw.par_chunks_exact(2))
            .for_each(|(dst, src)| {
                dst[0] = read(src);
                dst[1] = dst[0];
                dst[2] = dst[0];
                dst[3] = 65535;
            }),
        colors => return Err(format!("colors {colors} unsupported")),
    }
    Ok(rgba)
}

fn expand_raw8(raw: &[u8], bits: u16, colors: u16, pixels: usize) -> Result<Vec<u8>, String> {
    use rayon::prelude::*;

    let mut rgba = vec![0u8; pixels.saturating_mul(4)];
    if bits == 16 {
        let read = |p: &[u8]| encode_srgb(u16::from_le_bytes([p[0], p[1]]) as f32 / 65535.0);
        match colors {
            3 => rgba
                .par_chunks_exact_mut(4)
                .zip(raw.par_chunks_exact(6))
                .for_each(|(dst, src)| {
                    dst[0] = read(&src[0..2]);
                    dst[1] = read(&src[2..4]);
                    dst[2] = read(&src[4..6]);
                    dst[3] = 255;
                }),
            4 => rgba
                .par_chunks_exact_mut(4)
                .zip(raw.par_chunks_exact(8))
                .for_each(|(dst, src)| {
                    dst[0] = read(&src[0..2]);
                    dst[1] = read(&src[2..4]);
                    dst[2] = read(&src[4..6]);
                    dst[3] = 255;
                }),
            1 => rgba
                .par_chunks_exact_mut(4)
                .zip(raw.par_chunks_exact(2))
                .for_each(|(dst, src)| {
                    dst[0] = read(src);
                    dst[1] = dst[0];
                    dst[2] = dst[0];
                    dst[3] = 255;
                }),
            colors => return Err(format!("colors {colors} unsupported")),
        }
        return Ok(rgba);
    }
    if bits == 8 {
        match colors {
            3 => rgba
                .par_chunks_exact_mut(4)
                .zip(raw.par_chunks_exact(3))
                .for_each(|(dst, src)| {
                    dst[0] = encode_srgb(src[0] as f32 / 255.0);
                    dst[1] = encode_srgb(src[1] as f32 / 255.0);
                    dst[2] = encode_srgb(src[2] as f32 / 255.0);
                    dst[3] = 255;
                }),
            4 => rgba
                .par_chunks_exact_mut(4)
                .zip(raw.par_chunks_exact(4))
                .for_each(|(dst, src)| {
                    dst[0] = encode_srgb(src[0] as f32 / 255.0);
                    dst[1] = encode_srgb(src[1] as f32 / 255.0);
                    dst[2] = encode_srgb(src[2] as f32 / 255.0);
                    dst[3] = src[3];
                }),
            1 => rgba
                .par_chunks_exact_mut(4)
                .zip(raw.par_chunks_exact(1))
                .for_each(|(dst, src)| {
                    dst[0] = encode_srgb(src[0] as f32 / 255.0);
                    dst[1] = dst[0];
                    dst[2] = dst[0];
                    dst[3] = 255;
                }),
            colors => return Err(format!("colors {colors} unsupported")),
        }
        return Ok(rgba);
    }
    Err(format!("bits {bits} unsupported"))
}

/// Streams the file and keeps the highest-scoring decodable `FF D8 FF .. FF D9` segment.
fn scan_best<T>(
    path: &Path,
    mut evaluate: impl FnMut(&[u8]) -> Option<(u64, T)>,
) -> anyhow::Result<Option<T>> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::with_capacity(CHUNK, file);
    let mut buf = vec![0u8; CHUNK];
    let mut state = ScanState::default();
    let mut best: Option<(u64, T)> = None;

    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => feed(&mut state, &buf[..n], &mut evaluate, &mut best),
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(best.map(|(_, value)| value))
}

/// [`scan_best`] over a buffer the caller already read. Same chunking as the
/// streaming reader, so both find the same preview.
fn scan_best_bytes<T>(
    bytes: &[u8],
    mut evaluate: impl FnMut(&[u8]) -> Option<(u64, T)>,
) -> Option<T> {
    let mut state = ScanState::default();
    let mut best: Option<(u64, T)> = None;
    for chunk in bytes.chunks(CHUNK) {
        feed(&mut state, chunk, &mut evaluate, &mut best);
    }
    best.map(|(_, value)| value)
}

#[derive(Default)]
struct ScanState {
    /// Bytes from the last SOI to the end of what has been read so far.
    candidate: Vec<u8>,
    /// Where the next EOI search starts inside `candidate`.
    scan_from: usize,
    attempts: usize,
}

/// Feeds one chunk of the container into the scanner.
fn feed<T>(
    state: &mut ScanState,
    data: &[u8],
    evaluate: &mut impl FnMut(&[u8]) -> Option<(u64, T)>,
    best: &mut Option<(u64, T)>,
) {
    let mut pending = data.to_vec();
    loop {
        if state.candidate.is_empty() {
            match find_soi(&pending, 0) {
                Some(start) => {
                    state.candidate.extend_from_slice(&pending[start..]);
                    state.scan_from = 3;
                    state.attempts = 0;
                    pending.clear();
                    continue;
                }
                None => return,
            }
        }

        state.candidate.extend_from_slice(&pending);
        pending.clear();

        let mut advanced = false;
        while state.attempts < MAX_ATTEMPTS {
            let Some(eoi) = find_eoi(&state.candidate, state.scan_from) else {
                break;
            };
            let end = eoi + 2;
            state.attempts += 1;
            match evaluate(&state.candidate[..end]) {
                Some((score, value)) => {
                    if best.as_ref().is_none_or(|(top, _)| score > *top) {
                        *best = Some((score, value));
                    }
                    // Everything after the segment may hold the next preview.
                    state.candidate.drain(..end);
                    state.scan_from = 0;
                    state.attempts = 0;
                    advanced = true;
                    break;
                }
                // A truncated or bogus `FF D9`. Usually the SOI itself was stray sensor
                // data, so restart at the next SOI if the segment already holds one.
                None => match find_soi(&state.candidate, 1) {
                    Some(next) => {
                        state.candidate.drain(..next);
                        state.scan_from = 3;
                    }
                    None => state.scan_from = end,
                },
            }
        }

        if advanced {
            pending = std::mem::take(&mut state.candidate);
            continue;
        }

        // Rescan the last byte next time so an EOI split across chunks is not missed.
        state.scan_from = state.candidate.len().saturating_sub(1);
        if state.candidate.len() > MAX_SEGMENT || state.attempts >= MAX_ATTEMPTS {
            // shortcut: give up on this SOI instead of buffering more; a real preview
            // segment is far below both limits, so this only fires on stray markers.
            state.candidate = Vec::new();
            state.scan_from = 0;
            state.attempts = 0;
        }
        return;
    }
}

fn find_soi(data: &[u8], from: usize) -> Option<usize> {
    let last = data.len().checked_sub(3)?;
    let mut i = from;
    while i <= last {
        if data[i] == 0xFF && data[i + 1] == 0xD8 && data[i + 2] == 0xFF {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn find_eoi(data: &[u8], from: usize) -> Option<usize> {
    if data.len() < 2 {
        return None;
    }
    let mut i = from.min(data.len() - 1);
    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] == 0xD9 {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage, codecs::jpeg::JpegEncoder};
    use std::path::PathBuf;

    fn jpeg(width: u32, height: u32, seed: u8) -> Vec<u8> {
        let img = RgbImage::from_fn(width, height, |x, y| {
            Rgb([(x as u8).wrapping_mul(3).wrapping_add(seed), y as u8, seed])
        });
        let mut out = Vec::new();
        JpegEncoder::new_with_quality(&mut out, 80)
            .encode_image(&img)
            .unwrap();
        out
    }

    fn temp_file(name: &str, bytes: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("portrait-raw-{}-{}", std::process::id(), name));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn candidates_put_the_override_first_then_platform_names() {
        let override_path = PathBuf::from("/definitely/custom/libraw.dylib");
        let candidates = libraw::candidate_paths(Some(&override_path));
        assert_eq!(candidates.first(), Some(&override_path));

        #[cfg(target_os = "macos")]
        let expected = [
            "/opt/homebrew/lib/libraw.25.dylib",
            "/usr/local/lib/libraw.24.dylib",
            "libraw.23.dylib",
        ];
        #[cfg(not(target_os = "macos"))]
        let expected = [
            "/usr/local/lib/libraw.so.23",
            "/usr/lib/libraw.so.23",
            "libraw.so",
        ];
        for name in expected {
            assert!(
                candidates.iter().any(|path| path.as_path() == Path::new(name)),
                "missing {name} in {candidates:?}"
            );
        }
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn candidates_drop_a_duplicate_override() {
        #[cfg(target_os = "macos")]
        let duplicate = PathBuf::from("/opt/homebrew/lib/libraw.25.dylib");
        #[cfg(not(target_os = "macos"))]
        let duplicate = PathBuf::from("/usr/local/lib/libraw.so.23");

        let candidates = libraw::candidate_paths(Some(&duplicate));
        assert_eq!(candidates.first(), Some(&duplicate));
        assert_eq!(
            candidates.iter().filter(|path| **path == duplicate).count(),
            1,
            "override must not repeat the same default candidate: {candidates:?}"
        );
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn missing_library_message_lists_every_candidate() {
        let candidates = libraw::candidate_paths(Some(Path::new("/no/such/libraw.dylib")));
        let message = libraw::missing_library_error(&candidates);
        assert!(message.contains("libraw library not found"), "{message}");
        for candidate in &candidates {
            assert!(message.contains(&candidate.display().to_string()), "{message}");
        }
    }

    #[test]
    fn real_libraw_probe_is_conditional() {
        // No assertion when the host has no library: this only exercises the
        // real loader path (Homebrew here, absent on the default CI runner).
        match libraw::version() {
            Some(version) => assert!(version > 0, "unexpected libraw version {version:#x}"),
            None => eprintln!("libraw is not installed on this host; skipping real-library probe"),
        }
    }

    #[test]
    fn extracts_preview_from_wrapped_bytes() {
        let mut file = vec![0x49, 0x49, 0x2A, 0x00, 0x08, 0x00]; // TIFF-ish container start
        file.extend_from_slice(&jpeg(40, 24, 7));
        file.extend_from_slice(&[0u8; 4096]); // trailing sensor-ish bytes
        let path = temp_file("wrapped.iiq", &file);

        let img = embedded_preview(&path).unwrap();
        assert_eq!((img.width(), img.height()), (40, 24));
    }

    #[test]
    fn picks_the_largest_preview() {
        let mut file = Vec::new();
        file.extend_from_slice(&[0xAB; 1000]);
        file.extend_from_slice(&jpeg(16, 16, 1));
        file.extend_from_slice(&[0xCD; 1000]);
        file.extend_from_slice(&jpeg(96, 64, 2));
        file.extend_from_slice(&[0xEF; 1000]);
        file.extend_from_slice(&jpeg(32, 32, 3));
        let path = temp_file("multi.nef", &file);

        let img = embedded_preview(&path).unwrap();
        assert_eq!((img.width(), img.height()), (96, 64));
        assert_eq!(preview_dimensions(&path), Some((96, 64)));
    }

    #[test]
    fn stray_eoi_inside_garbage_falls_through() {
        // SOI, junk, an early FF D9 that cannot decode, then the real preview.
        let mut file = vec![0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x08, 1, 2, 3, 4];
        file.extend_from_slice(&[0x00; 64]);
        file.extend_from_slice(&[0xFF, 0xD9]);
        file.extend_from_slice(&[0x11; 2048]);
        file.extend_from_slice(&jpeg(48, 32, 9));
        let path = temp_file("stray-eoi.cr3", &file);

        let img = embedded_preview(&path).unwrap();
        assert_eq!((img.width(), img.height()), (48, 32));
    }

    #[test]
    fn no_jpeg_is_an_error_not_a_panic() {
        let path = temp_file("empty.arw", &[0u8; 8192]);
        assert!(embedded_preview(&path).is_err());
        assert_eq!(preview_dimensions(&path), None);
    }

    #[test]
    fn truncated_preview_does_not_panic() {
        let whole = jpeg(64, 48, 5);
        let path = temp_file("cut.dng", &whole[..whole.len() / 2]);
        assert!(embedded_preview(&path).is_err());

        // Missing file behaves the same way.
        assert!(embedded_preview(Path::new("no-such-file.iiq")).is_err());
    }

    #[test]
    fn recognises_raw_extensions() {
        for ext in [
            "iiq", "nef", "cr3", "cr2", "arw", "dng", "raf", "orf", "rw2", "pef", "srw", "3fr",
            "fff", "mos",
        ] {
            assert!(is_raw(Path::new(&format!("shot.{ext}"))), "{ext}");
            assert!(
                is_raw(Path::new(&format!("SHOT.{}", ext.to_uppercase()))),
                "{ext}"
            );
        }
        for ext in ["jpg", "jpeg", "png", "tif", "tiff", "webp", "txt"] {
            assert!(!is_raw(Path::new(&format!("shot.{ext}"))), "{ext}");
        }
        assert!(!is_raw(Path::new("no-extension")));
    }

    #[test]
    fn linear_light_is_encoded_before_it_is_reduced_to_eight_bits() {
        assert_eq!(encode_srgb(0.0), 0);
        assert_eq!(encode_srgb(1.0), 255);
        assert_eq!(encode_srgb(0.5), 188);
        // A value 8-bit linear would round to zero still gets a code.
        assert_eq!((1.0f32 / 4096.0 * 255.0 + 0.5) as u8, 0);
        assert!(encode_srgb(1.0 / 4096.0) > 0);
        assert!(encode_srgb(0.2) < encode_srgb(0.4));
    }

    fn expand_raw16_serial(raw: &[u8], colors: u16) -> Vec<u16> {
        let wide: Vec<u16> =
            raw.chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect();
        let mut rgba = Vec::with_capacity(wide.len() / colors as usize * 4);
        match colors {
            3 => wide
                .chunks_exact(3)
                .for_each(|p| rgba.extend_from_slice(&[p[0], p[1], p[2], 65535])),
            4 => wide
                .chunks_exact(4)
                .for_each(|p| rgba.extend_from_slice(&[p[0], p[1], p[2], 65535])),
            1 => wide
                .iter()
                .for_each(|g| rgba.extend_from_slice(&[*g, *g, *g, 65535])),
            _ => unreachable!(),
        }
        rgba
    }

    #[test]
    fn parallel_raw16_expansion_matches_the_serial_original_exactly() {
        let pixels = 73usize * 41;
        for colors in [1usize, 3, 4] {
            let raw: Vec<u8> = (0..pixels * colors * 2)
                .map(|i| i.wrapping_mul(37).wrapping_add(i / 11) as u8)
                .collect();
            let expected = expand_raw16_serial(&raw, colors as u16);
            let actual = expand_raw16(&raw, colors as u16, pixels).unwrap();
            assert_eq!(actual, expected, "colors={colors}");
        }
        assert!(expand_raw16(&[], 2, 0).is_err());
    }

    fn expand_raw8_serial(raw: &[u8], bits: u16, colors: u16) -> Vec<u8> {
        let enc16 = |v: u16| encode_srgb(v as f32 / 65535.0);
        let enc8 = |v: u8| encode_srgb(v as f32 / 255.0);
        let mut rgba = Vec::new();
        if bits == 16 {
            let wide: Vec<u16> =
                raw.chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect();
            match colors {
                3 => wide.chunks_exact(3).for_each(|p| {
                    rgba.extend_from_slice(&[enc16(p[0]), enc16(p[1]), enc16(p[2]), 255])
                }),
                4 => wide.chunks_exact(4).for_each(|p| {
                    rgba.extend_from_slice(&[enc16(p[0]), enc16(p[1]), enc16(p[2]), 255])
                }),
                1 => wide.iter().for_each(|g| {
                    let v = enc16(*g);
                    rgba.extend_from_slice(&[v, v, v, 255])
                }),
                _ => unreachable!(),
            }
        } else {
            match colors {
                3 => raw.chunks_exact(3).for_each(|p| {
                    rgba.extend_from_slice(&[enc8(p[0]), enc8(p[1]), enc8(p[2]), 255])
                }),
                4 => raw.chunks_exact(4).for_each(|p| {
                    rgba.extend_from_slice(&[enc8(p[0]), enc8(p[1]), enc8(p[2]), p[3]])
                }),
                1 => raw.iter().for_each(|g| {
                    let v = enc8(*g);
                    rgba.extend_from_slice(&[v, v, v, 255])
                }),
                _ => unreachable!(),
            }
        }
        rgba
    }

    #[test]
    fn parallel_raw8_expansion_matches_the_serial_original_exactly() {
        let pixels = 73usize * 41;
        for bits in [8usize, 16] {
            for colors in [1usize, 3, 4] {
                let bytes = pixels * colors * if bits == 16 { 2 } else { 1 };
                let raw: Vec<u8> = (0..bytes)
                    .map(|i| i.wrapping_mul(37).wrapping_add(i / 11) as u8)
                    .collect();
                let expected = expand_raw8_serial(&raw, bits as u16, colors as u16);
                let actual = expand_raw8(&raw, bits as u16, colors as u16, pixels).unwrap();
                assert_eq!(actual, expected, "bits={bits} colors={colors}");
            }
        }
        assert!(expand_raw8(&[], 12, 3, 0).is_err());
        assert!(expand_raw8(&[], 8, 2, 0).is_err());
    }

    #[test]
    #[ignore]
    fn bench_parallel_raw16_expansion_24mp() {
        use std::time::Instant;

        let pixels = 6000usize * 4000;
        let raw: Vec<u8> = (0..pixels * 6)
            .map(|i| i.wrapping_mul(40503).wrapping_add(i / 17) as u8)
            .collect();
        let measure = |parallel: bool| {
            let mut times = Vec::with_capacity(7);
            for _ in 0..7 {
                let started = Instant::now();
                let out = if parallel {
                    expand_raw16(&raw, 3, pixels).unwrap()
                } else {
                    expand_raw16_serial(&raw, 3)
                };
                times.push(started.elapsed().as_secs_f64() * 1000.0);
                std::hint::black_box(out);
            }
            times.sort_by(f64::total_cmp);
            (times[times.len() / 2], times[(times.len() * 95 / 100).max(times.len() - 1)])
        };
        let (serial50, serial95) = measure(false);
        let (parallel50, parallel95) = measure(true);
        println!(
            "6000x4000 RAW16 RGB expansion: serial {serial50:.3}/{serial95:.3} ms, parallel \
             {parallel50:.3}/{parallel95:.3} ms, {:.2}x P50",
            serial50 / parallel50
        );
    }

    #[test]
    #[ignore]
    fn bench_parallel_raw8_expansion_24mp() {
        use std::time::Instant;

        let pixels = 6000usize * 4000;
        let raw: Vec<u8> = (0..pixels * 6)
            .map(|i| i.wrapping_mul(40503).wrapping_add(i / 17) as u8)
            .collect();
        let measure = |parallel: bool| {
            let mut times = Vec::with_capacity(7);
            for _ in 0..7 {
                let started = Instant::now();
                let out = if parallel {
                    expand_raw8(&raw, 16, 3, pixels).unwrap()
                } else {
                    expand_raw8_serial(&raw, 16, 3)
                };
                times.push(started.elapsed().as_secs_f64() * 1000.0);
                std::hint::black_box(out);
            }
            times.sort_by(f64::total_cmp);
            (times[times.len() / 2], times[(times.len() * 95 / 100).max(times.len() - 1)])
        };
        let (serial50, serial95) = measure(false);
        let (parallel50, parallel95) = measure(true);
        println!(
            "6000x4000 RAW16 to RGBA8 expansion: serial {serial50:.3}/{serial95:.3} ms, parallel \
             {parallel50:.3}/{parallel95:.3} ms, {:.2}x P50",
            serial50 / parallel50
        );
    }


    #[test]
    fn demosaic_values_map_to_the_filters_this_libraw_has() {
        for quality in [0, 1, 2, 3, 11, 12] {
            assert_eq!(normalise_demosaic(quality), quality);
        }
        // 13 and anything else fall through to AHD inside libraw; do the same here.
        for quality in [4, 13, 14, 99, -1, i32::MAX] {
            assert_eq!(normalise_demosaic(quality), DEMOSAIC_DEFAULT, "{quality}");
        }
    }
    #[test]
    fn env_parser_accepts_numbers_and_rejects_the_rest() {
        assert_eq!(parse_demosaic("11"), Some(11));
        assert_eq!(parse_demosaic(" 0 "), Some(0));
        assert_eq!(parse_demosaic("3\n"), Some(3));
        for text in ["13", "abc", "", "-1", "11.0", "999999999999"] {
            assert_eq!(parse_demosaic(text), None, "{text:?}");
        }
    }
    #[test]
    fn invalid_override_degrades_to_ahd_without_panicking() {
        set_demosaic(12);
        assert_eq!(DEMOSAIC.load(Ordering::Relaxed), 12);
        set_demosaic(99);
        assert_eq!(DEMOSAIC.load(Ordering::Relaxed), DEMOSAIC_DEFAULT);
        set_demosaic(-1);
        assert_eq!(DEMOSAIC.load(Ordering::Relaxed), DEMOSAIC_DEFAULT);
    }
    #[test]
    fn decoding_with_an_unusual_filter_never_panics() {
        // No libraw answer for these bytes, so every path must reach the preview
        // fallback and report an error instead of unwinding.
        for quality in [0, 11, 12, i32::MAX] {
            set_demosaic(quality);
            let path = temp_file(&format!("f{quality}.dng"), &[0u8; 4096]);
            assert!(decode(&path).is_err());
            assert!(decode(Path::new("no-such-file.dng")).is_err());
        }
        set_demosaic(DEMOSAIC_DEFAULT);
    }
    #[test]
    fn decode_falls_back_to_the_embedded_preview() {
        let mut file = vec![0x49, 0x49, 0x2A, 0x00];
        file.extend_from_slice(&jpeg(40, 24, 3));
        let path = temp_file("fallback.nef", &file);

        let img = decode(&path).unwrap();
        assert_eq!((img.width(), img.height()), (40, 24));

        // A file with no preview at all is an error, not a panic.
        let broken = temp_file("broken.dng", &[0u8; 4096]);
        assert!(decode(&broken).is_err());
        assert!(decode(Path::new("no-such-file.nef")).is_err());
    }

    #[test]
    fn the_preview_gate_ignores_tone_and_turn_but_rejects_stripes() {
        // A linear-light decode of a ramp, the camera's tone-mapped preview of
        // the same ramp, and a quarter-turned copy are the same picture.
        let ramp: Vec<f32> = (0..32)
            .flat_map(|y| (0..32).map(move |x| (x * 8 + y * 3) as f32))
            .collect();
        let display: Vec<f32> = ramp.iter().map(|v| (v / 255.0).powf(0.45) * 255.0).collect();
        assert!(similar(&ramp, &display));
        assert!(similar(&transpose(&ramp), &display));
        assert!(similar(&transpose(&transpose(&transpose(&ramp))), &display));

        // A decoder that hands back stripes instead of the picture must not pass,
        // however plausible its levels look.
        let stripes: Vec<f32> = (0..32)
            .flat_map(|_| (0..32).map(|x| if x % 2 == 0 { 0.0 } else { 255.0 }))
            .collect();
        assert!(!similar(&stripes, &display));
    }

    #[cfg(target_os = "windows")]
    mod windows_libraw {
        use super::*;
        use std::path::PathBuf;

        fn sample() -> PathBuf {
            std::env::var_os("AURORA_TEST_RAW")
                .map(PathBuf::from)
                .expect("set AURORA_TEST_RAW to a real NEF file")
        }

        fn he_sample() -> Option<PathBuf> {
            std::env::var_os("AURORA_TEST_HE_RAW")
                .map(PathBuf::from)
                .filter(|path| path.is_file())
        }

        #[test]
        fn dll_missing_falls_back_to_preview() {
            // The same decode path must still return the preview when the
            // DLL is absent. We cannot unload it here, so we simulate the
            // failure with a file that libraw cannot open but the preview
            // scanner can, i.e. any TIFF container with an embedded JPEG.
            let mut file = vec![0x49, 0x49, 0x2A, 0x00];
            file.extend_from_slice(&jpeg(48, 32, 4));
            let path = temp_file("dll-missing.nef", &file);
            let img = decode(&path).unwrap();
            assert_eq!((img.width(), img.height()), (48, 32));
        }

        #[test]
        #[ignore = "needs a real NEF file; set AURORA_TEST_RAW"]
        fn real_nef_decodes_successfully() {
            let path = sample();
            let img = decode(&path).unwrap();
            assert!(img.width().max(img.height()) > 3000, "{:?}", img.dimensions());
            assert!(sensor_dimensions(&path).unwrap().0.max(sensor_dimensions(&path).unwrap().1) > 3000);
        }

        #[test]
        fn invalid_file_does_not_panic() {
            let path = temp_file("invalid.nef", &[0u8; 4096]);
            assert!(decode(&path).is_err());
            assert!(decode16(&path).is_none());
        }

        #[test]
        #[ignore = "needs a real NEF file; set AURORA_TEST_RAW"]
        fn sixteen_bit_path_returns_or_falls_back() {
            let path = sample();
            match decode16(&path) {
                Some((w, h, rgba)) => {
                    assert_eq!(rgba.len(), (w * h * 4) as usize);
                    assert!(w.max(h) > 3000);
                }
                None => {
                    // HE* compression is not demosaiced by this libraw build:
                    // the 8-bit path must still deliver the embedded preview.
                    let fallback = he_sample().unwrap_or(path);
                    let eight = decode(&fallback).unwrap();
                    assert!(eight.width() > 0);
                }
            }
        }
    }

    /// Needs a real camera file and a machine with libraw; run with
    /// `AURORA_TEST_RAW=/path/to/file.nef cargo test -- --include-ignored`.
    #[test]
    #[ignore = "needs a real RAW file (AURORA_TEST_RAW) and libraw"]
    fn decodes_sensor_data() {
        let Ok(path) = std::env::var("AURORA_TEST_RAW") else {
            panic!("set AURORA_TEST_RAW to a RAW file");
        };
        let path = PathBuf::from(path);
        let img = decode(&path).unwrap();
        assert!(img.width().max(img.height()) > 3000, "sensor decode: {:?}", img.dimensions());
        assert!(
            img.pixels().any(|p| p.0 != [0, 0, 0, 255]),
            "the frame is all black"
        );
        let preview = preview_dimensions(&path).unwrap();
        assert_ne!(
            (img.width(), img.height()),
            preview,
            "sensor decode must differ from the embedded preview"
        );
        let sensor = sensor_dimensions(&path).unwrap();
        assert!(sensor.0.max(sensor.1) > 3000, "sensor dimensions {sensor:?}");
    }

    /// The 16-bit path on a real file: either libraw hands over linear u16, or it
    /// cannot demosaic the file (Nikon High Efficiency) and the 8-bit path takes
    /// over. Both outcomes must leave a usable picture.
    /// `AURORA_TEST_RAW=/path/to/file.nef cargo test -- --include-ignored`
    #[test]
    #[ignore = "needs a real RAW file (AURORA_TEST_RAW) and libraw"]
    fn decodes_sensor_data_at_sixteen_bits() {
        let Ok(path) = std::env::var("AURORA_TEST_RAW") else {
            panic!("set AURORA_TEST_RAW to a RAW file");
        };
        let path = PathBuf::from(path);
        match decode16(&path) {
            Some((w, h, rgba)) => {
                assert_eq!(rgba.len(), (w * h * 4) as usize);
                assert!(w.max(h) > 3000, "16-bit decode {w}x{h}");
                assert!(rgba.chunks_exact(4).any(|p| p[0] | p[1] | p[2] != 0), "all black");
                let eight = decode(&path).unwrap();
                assert_eq!((w, h), eight.dimensions(), "16-bit and 8-bit frames must align");
                eprintln!("16-bit sensor decode: {w}x{h}");
            }
            None => {
                let eight = decode(&path).unwrap();
                assert!(eight.width() > 0, "the 8-bit fallback must still deliver a frame");
                eprintln!("16-bit decode unavailable, 8-bit path delivered {}x{}", eight.width(), eight.height());
            }
        }
    }

    #[test]
    fn bayer_left_alignment_matches_the_c1_scale() {
        let frame = BayerFrame {
            raw_width: 4,
            raw_height: 2,
            width: 4,
            height: 2,
            top_margin: 0,
            left_margin: 0,
            stride: 8,
            bits: 14,
            data: vec![0, 1, 2, 16383, 1008, 1009, 1010, 1011],
            filters: 0xb4b4_b4b4,
            colors: 3,
            black: 1008,
            maximum: 16383,
            cam_mul: [1.0; 4],
            rgb_cam: [[0.0; 4]; 3],
        };
        assert_eq!(frame.shift(), 2);
        assert_eq!(
            frame.left_aligned(),
            vec![0, 4, 8, 65532, 4032, 4036, 4040, 4044]
        );
        assert_eq!(frame.black16(), 4032);
        assert_eq!(frame.color_at(0, 0), 0);
        assert_eq!(frame.color_at(1, 0), 1);
        assert_eq!(frame.color_at(0, 1), 3);
        assert_eq!(frame.color_at(1, 1), 2);
    }

    /// `AURORA_TEST_BAYER_RAW=<classic NEF/ARW/CR3> cargo test ... -- --ignored`
    #[test]
    #[ignore = "needs a non-HE RAW file (AURORA_TEST_BAYER_RAW) and libraw"]
    fn decodes_a_real_bayer_plane() {
        let path = std::env::var("AURORA_TEST_BAYER_RAW")
            .expect("set AURORA_TEST_BAYER_RAW to a classic NEF, ARW or CR3");
        let frame = decode_bayer16(Path::new(&path)).expect("LibRaw unpack");
        assert!(frame.raw_width > 3000 && frame.raw_height > 2000);
        assert_eq!(frame.data.len(), frame.raw_width * frame.raw_height);
        assert!((12..=16).contains(&frame.bits), "bits {}", frame.bits);
        assert!(frame.maximum > 0);
        assert!(frame.data.iter().any(|&value| value != 0));
        eprintln!(
            "{}: {}x{} active {}x{}+{}+{}, {} bit, black {}, max {}, filters {:#x}",
            path,
            frame.raw_width,
            frame.raw_height,
            frame.width,
            frame.height,
            frame.left_margin,
            frame.top_margin,
            frame.bits,
            frame.black,
            frame.maximum,
            frame.filters
        );
    }

    /// Compares the LibRaw front-end with the C1 HE decoder on the same NEF.
    /// The C1 decoder emits 16-bit left-aligned samples, LibRaw 14-bit ones;
    /// after the documented shift every sample must be identical.
    /// `AURORA_TEST_RAW=<Z6III HE NEF> cargo test ... -- --ignored`
    #[test]
    #[ignore = "needs a Z6III HE NEF (AURORA_TEST_RAW)"]
    fn libraw_bayer_matches_the_c1_he_decoder() {
        let path = std::env::var("AURORA_TEST_RAW")
            .expect("set AURORA_TEST_RAW to a Z6III HE NEF");
        let lib = decode_bayer16(Path::new(&path)).expect("LibRaw Bayer");
        let bytes = std::fs::read(&path).expect("read NEF");
        let c1 = crate::aurora_raw::c1_raw_he::decode_frame_cropped(&bytes)
            .expect("C1 HE decode")
            .expect("verified HE variant");
        assert_eq!((lib.raw_width, lib.raw_height), (c1.full_width(), c1.full_height()));
        let c1_samples = c1.full_samples();
        assert_eq!(lib.data.len(), c1_samples.len());
        let mut mismatches = 0usize;
        let mut maximum = 0u16;
        let mut first = None;
        for (index, (&a, &b)) in lib.data.iter().zip(c1_samples).enumerate() {
            if a != b {
                mismatches += 1;
                maximum = maximum.max(a.abs_diff(b));
                first.get_or_insert((index / lib.raw_width, index % lib.raw_width, a, b));
            }
        }
        eprintln!(
            "{path}: {} 14-bit samples, {mismatches} mismatches, max diff {maximum}, first {first:?}",
            lib.data.len()
        );
        eprintln!(
            "first raw libraw={:?} c1={:?}",
            &lib.data[..8.min(lib.data.len())],
            &c1_samples[..8.min(c1_samples.len())]
        );
        assert_eq!(mismatches, 0, "LibRaw and C1 Bayer differ");
    }

    #[test]
    fn fallback_provenance_never_unlocks_the_c1_z6iii_pre_matrix() {
        let libraw = DecodeProvenance::libraw16();
        let libraw8 = DecodeProvenance::libraw_srgb8();
        let preview = DecodeProvenance::camera_preview();
        for provenance in [libraw, libraw8, preview] {
            assert!(!provenance.unlocks_c1_z6iii_linear(), "{provenance:?}");
        }
        assert_eq!(libraw.pixel_space, PixelSpace::LibRawLinear16);
        assert_eq!(preview.pixel_space, PixelSpace::EmbeddedPreview);

        // The Aurora chain unlocks it, whether OpenCL or the CPU fallback ran.
        let opencl = DecodeProvenance::aurora(crate::aurora_raw::AuroraRoute::Z6iiiOpenCl);
        let cpu = DecodeProvenance::aurora(
            crate::aurora_raw::AuroraRoute::Z6iiiCpuOpenClUnavailable,
        );
        assert!(opencl.unlocks_c1_z6iii_linear());
        assert!(cpu.unlocks_c1_z6iii_linear());
        // A generic Bayer frame is camera-linear but not Z6III.
        let generic = DecodeProvenance::aurora(crate::aurora_raw::AuroraRoute::Cpu);
        assert_eq!(generic.pixel_space, PixelSpace::CameraLinear16);
        assert!(!generic.unlocks_c1_z6iii_linear());
    }

    #[test]
    fn opencl_failure_states_stay_distinguishable() {
        use crate::aurora_raw::AuroraRoute as Route;
        let succeeded = DecodeProvenance::aurora(Route::Z6iiiOpenCl);
        let unavailable = DecodeProvenance::aurora(Route::Z6iiiCpuOpenClUnavailable);
        let failed = DecodeProvenance::aurora(Route::Z6iiiCpuOpenClFailed);
        assert_eq!(succeeded.opencl, OpenClStatus::Succeeded);
        assert_eq!(unavailable.opencl, OpenClStatus::Unavailable);
        assert_eq!(failed.opencl, OpenClStatus::Failed);
        assert_eq!(succeeded.backend, DecodeBackend::AuroraZ6iiiOpenCl);
        assert_eq!(unavailable.backend, DecodeBackend::AuroraCpuChain);
        assert_eq!(failed.backend, DecodeBackend::AuroraCpuChain);
        assert_ne!(unavailable.opencl_code(), failed.opencl_code());
    }

    #[test]
    fn provenance_header_codes_round_trip_and_reject_unknown_values() {
        for provenance in [
            DecodeProvenance::aurora(crate::aurora_raw::AuroraRoute::Z6iiiOpenCl),
            DecodeProvenance::aurora(crate::aurora_raw::AuroraRoute::Z6iiiCpuOpenClFailed),
            DecodeProvenance::libraw16(),
            DecodeProvenance::camera_preview(),
        ] {
            let decoded = DecodeProvenance::from_codes(
                provenance.pipeline_fingerprint,
                provenance.backend_code(),
                provenance.opencl_code(),
                provenance.pixel_space_code(),
                provenance.flags(),
            )
            .expect("known codes");
            assert_eq!(decoded, provenance);
        }
        assert!(DecodeProvenance::from_codes(1, 9, 0, 0, 0).is_none());
        assert!(DecodeProvenance::from_codes(1, 0, 9, 0, 0).is_none());
        assert!(DecodeProvenance::from_codes(1, 0, 0, 9, 0).is_none());
        assert!(DecodeProvenance::from_codes(1, 0, 0, 0, 0b10).is_none());
    }

    #[test]
    fn decode_route_changes_with_behavioural_inputs() {
        let base = DecodeRoute {
            pipeline_fingerprint: 0,
            c1_raw_enabled: true,
            z6iii_nef: true,
            z6iii_feature: true,
            libraw_available: true,
            opencl: OpenClRouteState::Available,
            demosaic: 3,
        };
        let mut disabled = base;
        disabled.c1_raw_enabled = false;
        let mut no_libraw = base;
        no_libraw.libraw_available = false;
        let mut no_opencl = base;
        no_opencl.opencl = OpenClRouteState::Unavailable;
        let mut not_z6iii = base;
        not_z6iii.z6iii_nef = false;
        let mut fingerprint = base;
        fingerprint.pipeline_fingerprint += 1;
        let mut demosaic = base;
        demosaic.demosaic = 0;
        for variant in [disabled, no_libraw, no_opencl, not_z6iii, fingerprint, demosaic] {
            assert_ne!(base, variant);
        }
    }

    #[test]
    fn stage_revisions_only_invalidate_the_routes_that_run_them() {
        let z6iii = DecodeRoute {
            pipeline_fingerprint: 0,
            c1_raw_enabled: true,
            z6iii_nef: true,
            z6iii_feature: true,
            libraw_available: true,
            opencl: OpenClRouteState::Available,
            demosaic: 3,
        };
        let libraw_only = DecodeRoute { c1_raw_enabled: false, ..z6iii };
        let base = DECODE_STAGE_REVISIONS;
        assert_eq!(base.pipeline_fingerprint(&z6iii), base.pipeline_fingerprint(&z6iii));
        assert_eq!(
            base.pipeline_fingerprint(&libraw_only),
            base.pipeline_fingerprint(&libraw_only)
        );

        // A stage only the Aurora chain runs must leave a LibRaw-only route
        // fingerprint alone, and a Z6III-only stage must not touch the generic
        // Aurora route.
        let chroma = DecodeStageRevisions { chromatic_aberration: 2, ..base };
        let generic = DecodeRoute { z6iii_nef: false, ..z6iii };
        assert_ne!(chroma.pipeline_fingerprint(&z6iii), base.pipeline_fingerprint(&z6iii));
        assert_ne!(
            chroma.pipeline_fingerprint(&generic),
            base.pipeline_fingerprint(&generic),
            "the generic Aurora route runs the same chromatic aberration stage"
        );
        assert_eq!(
            chroma.pipeline_fingerprint(&libraw_only),
            base.pipeline_fingerprint(&libraw_only),
            "a LibRaw-only route never runs our chromatic aberration stage"
        );
        let z6iii_bump = DecodeStageRevisions {
            z6iii_reconstruction: base.z6iii_reconstruction + 1,
            ..base
        };
        assert_ne!(
            z6iii_bump.pipeline_fingerprint(&z6iii),
            base.pipeline_fingerprint(&z6iii)
        );
        assert_eq!(
            z6iii_bump.pipeline_fingerprint(&generic),
            base.pipeline_fingerprint(&generic),
            "the generic Aurora route does not run the Z6III reconstruction"
        );
        let demosaic_bump = DecodeStageRevisions { demosaic: 2, ..base };
        assert_ne!(
            demosaic_bump.pipeline_fingerprint(&libraw_only),
            base.pipeline_fingerprint(&libraw_only),
            "demosaic runs on every route"
        );
    }

    #[test]
    fn persisted_provenance_must_carry_the_current_stage_fingerprint() {
        let libraw = DecodeProvenance::libraw16();
        assert!(libraw.is_current());
        assert_eq!(
            libraw.pipeline_fingerprint,
            DECODE_STAGE_REVISIONS.provenance_fingerprint(&libraw)
        );
        let z6iii = DecodeProvenance::aurora(crate::aurora_raw::AuroraRoute::Z6iiiOpenCl);
        let generic = DecodeProvenance::aurora(crate::aurora_raw::AuroraRoute::Cpu);
        let preview = DecodeProvenance::camera_preview();
        let mut fingerprints = vec![
            libraw.pipeline_fingerprint,
            z6iii.pipeline_fingerprint,
            generic.pipeline_fingerprint,
            preview.pipeline_fingerprint,
        ];
        fingerprints.sort_unstable();
        fingerprints.dedup();
        assert_eq!(fingerprints.len(), 4, "every provenance shape is distinct");
        assert!(!DecodeProvenance { pipeline_fingerprint: 0, ..z6iii }.is_current());
        assert!(!DecodeProvenance { pipeline_fingerprint: 0, ..preview }.is_current());
    }

    /// P-05 manual benchmark: same file, same build profile. Prints the exact
    /// pixel hash so the pre/post trees can be compared byte for byte.
    #[test]
    #[ignore = "manual P-05 bench: set AURORA_TEST_RAW"]
    fn bench_p05_decode_path() {
        use std::time::Instant;
        let path = std::env::var_os("AURORA_TEST_RAW")
            .map(std::path::PathBuf::from)
            .expect("set AURORA_TEST_RAW to a real NEF file");
        let mut best = f64::MAX;
        for round in 0..3 {
            let started = Instant::now();
            let (width, height, pixels, provenance) =
                decode16_traced(&path).expect("decode");
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            best = best.min(ms);
            let hash = pixels.iter().fold(0xcbf2_9ce4_8422_2325u64, |mut hash, &value| {
                hash ^= u64::from(value);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                hash
            });
            println!(
                "P05 round={round} {width}x{height} backend={:?} px={} ms={ms:.1} best={best:.1} hash={hash:016x}",
                provenance.backend,
                pixels.len()
            );
        }
        for (kind, aggregate) in raw_decode_stats() {
            println!(
                "P05 stats {} decodes={} timings={:?}",
                kind.label(),
                aggregate.decodes,
                aggregate.timings
            );
        }
        reset_raw_decode_stats();
    }

    #[test]
    fn fallback_reuses_bytes_after_a_rejected_unpack() {
        let mut container = vec![0x49, 0x49, 0x2A, 0x00, 0x08, 0x00];
        container.extend_from_slice(&jpeg(48, 32, 11));
        container.extend_from_slice(&[0u8; 4096]);
        let path = temp_file("fallback-reuse.nef", &container);
        // Any attempt to read the path from here on must fail: only the bytes
        // handed over by the 16-bit attempt can produce a frame.
        std::fs::remove_file(&path).unwrap();
        let failed = FailedRaw16 { bytes: container, libraw_unpack_unsupported: true };
        let decoded = decode8_fallback(&path, Some(&failed)).expect("preview from bytes");
        assert_eq!(decoded.dimensions(), (48, 32));
        assert_eq!(from_sensor(&path), Some(false));
    }

    #[test]
    fn fallback_reuses_bytes_when_libraw_cannot_use_the_buffer() {
        let mut container = vec![0x4D, 0x4D, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08];
        container.extend_from_slice(&jpeg(40, 24, 3));
        container.extend_from_slice(&[0x7F; 4096]);
        let path = temp_file("fallback-buffer.cr3", &container);
        std::fs::remove_file(&path).unwrap();
        let failed = FailedRaw16 { bytes: container, libraw_unpack_unsupported: false };
        // LibRaw may or may not exist here; either it rejects the fake
        // container (file is gone, buffer parse fails) or it is absent. Both
        // end at the embedded preview scanned from the same bytes.
        let decoded = decode8_fallback(&path, Some(&failed)).expect("preview from bytes");
        assert_eq!(decoded.dimensions(), (40, 24));
    }

    #[test]
    fn failed_16bit_attempt_returns_the_bytes_it_read() {
        let container = vec![0x5A; 512];
        let path = temp_file("attempt-bytes.nef", &container);
        let route = DecodeRoute {
            pipeline_fingerprint: 0,
            c1_raw_enabled: true,
            z6iii_nef: false,
            z6iii_feature: false,
            libraw_available: false,
            opencl: OpenClRouteState::NotCompiled,
            demosaic: 3,
        };
        let (frame, failed) = decode16_attempt(&path, route);
        assert!(frame.is_none());
        let failed = failed.expect("the attempt must hand its bytes to the fallback");
        assert_eq!(failed.bytes, container);
        assert!(!failed.libraw_unpack_unsupported);
    }

    #[test]
    #[ignore = "requires a real Z6III HE NEF in AURORA_TEST_Z6III_HE"]
    fn z6iii_he_route_skips_libraw_before_aurora() {
        let Some(path) = std::env::var_os("AURORA_TEST_Z6III_HE").map(PathBuf::from) else {
            return;
        };
        if !path.is_file() {
            return;
        }
        let route = decode_route(&path);
        assert!(
            route.z6iii_nef && route.z6iii_feature,
            "fixture must be a Z6III NEF in a z6iii-raw-export build"
        );
        let bytes = std::fs::read(&path).unwrap();
        assert!(
            crate::aurora_raw::route_skips_libraw(&route, &bytes),
            "fixture must be the verified HE profile this gate claims"
        );
        reset_raw_decode_stats();
        let (frame, failed) = decode16_attempt(&path, route);
        let (_, _, _, provenance) = frame.expect("HE decode");
        assert!(matches!(
            provenance.backend,
            DecodeBackend::AuroraZ6iiiOpenCl | DecodeBackend::AuroraCpuChain
        ));
        assert!(failed.is_none());
        let stats = raw_decode_stats();
        let (_, aggregate) = stats
            .iter()
            .find(|(kind, _)| *kind == RawKind::Z6iiiHe)
            .expect("z6iii stats");
        assert_eq!(
            aggregate.timings.libraw_open_ms, 0,
            "the verified HE route must not open LibRaw before Aurora"
        );
        assert_eq!(aggregate.timings.libraw_unpack_ms, 0);
    }

    #[test]
    fn preview_scan_over_bytes_matches_the_file_scan() {
        let dir = std::env::temp_dir().join(format!("p05-scan-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let jpeg_path = dir.join("preview.jpg");
        let pixels: Vec<u8> = (0..64 * 48 * 3).map(|i| (i % 251) as u8).collect();
        image::save_buffer_with_format(
            &jpeg_path,
            &pixels,
            64,
            48,
            image::ExtendedColorType::Rgb8,
            image::ImageFormat::Jpeg,
        )
        .unwrap();
        let jpeg = std::fs::read(&jpeg_path).unwrap();
        let mut container = vec![0u8; 4096];
        container.extend_from_slice(&jpeg);
        container.extend(vec![0u8; 4096]);
        let container_path = dir.join("container.nef");
        std::fs::write(&container_path, &container).unwrap();

        let from_file = embedded_preview(&container_path).unwrap();
        let from_bytes = embedded_preview_bytes(&container).unwrap();
        assert_eq!(
            (from_file.width(), from_file.height()),
            (from_bytes.width(), from_bytes.height())
        );
        assert_eq!(from_file.as_raw(), from_bytes.as_raw());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn raw_decode_stats_aggregate_by_kind() {
        reset_raw_decode_stats();
        record_raw_decode(
            RawKind::Nef,
            RawStageTimings { read_ms: 3, c1_stages_ms: 10, ..RawStageTimings::default() },
        );
        record_raw_decode(
            RawKind::Nef,
            RawStageTimings { read_ms: 5, ..RawStageTimings::default() },
        );
        let stats = raw_decode_stats();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].0, RawKind::Nef);
        assert_eq!(stats[0].1.decodes, 2);
        assert_eq!(stats[0].1.timings.read_ms, 8);
        assert_eq!(stats[0].1.timings.c1_stages_ms, 10);
        reset_raw_decode_stats();
        assert!(raw_decode_stats().is_empty());
    }

    #[test]
    fn raw_kind_classifies_extension_and_z6iii_models() {
        assert_eq!(RawKind::classify(Path::new("a.nef"), None, None), RawKind::Nef);
        assert_eq!(RawKind::classify(Path::new("a.ARW"), None, None), RawKind::Arw);
        assert_eq!(RawKind::classify(Path::new("a.dng"), None, None), RawKind::Dng);
        assert_eq!(RawKind::classify(Path::new("a.cr3"), None, None), RawKind::Other);
        assert_eq!(
            RawKind::classify(Path::new("a.nef"), None, Some("NIKON Z 6_3")),
            RawKind::Z6iiiHe
        );
        assert_eq!(
            RawKind::classify(Path::new("a.nef"), None, Some("NIKON Z 8")),
            RawKind::Nef
        );
    }
    /// Real-sample acceptance for the native front-end: every metadata field
    /// and every sensor sample compared with LibRaw on the same file.
    /// `AURORA_TEST_RAW=<NEF/ARW/CR2/DNG> cargo test -p aurora-engine --lib
    ///  native_bayer_matches_libraw -- --ignored --nocapture`
    #[test]
    #[ignore = "needs a real RAW file (AURORA_TEST_RAW) and libraw"]
    fn native_bayer_matches_libraw() {
        let path = std::env::var("AURORA_TEST_RAW").expect("set AURORA_TEST_RAW");
        let path = PathBuf::from(path);
        let native = crate::aurora_raw::rust_raw::decode(&path).expect("native decode");
        let oracle = libraw::unpack_bayer(&path).expect("LibRaw unpack");
        eprintln!(
            "native: {}x{} active {}x{}+{}+{} bits {} filters {:#x} colors {} black {} max {} cam_mul {:?} rgb_cam {:?}",
            native.raw_width, native.raw_height, native.width, native.height,
            native.left_margin, native.top_margin, native.bits, native.filters, native.colors,
            native.black, native.maximum, native.cam_mul, native.rgb_cam
        );
        eprintln!(
            "libraw: {}x{} active {}x{}+{}+{} bits {} filters {:#x} colors {} black {} max {} cam_mul {:?} rgb_cam {:?}",
            oracle.raw_width, oracle.raw_height, oracle.width, oracle.height,
            oracle.left_margin, oracle.top_margin, oracle.bits, oracle.filters, oracle.colors,
            oracle.black, oracle.maximum, oracle.cam_mul, oracle.rgb_cam
        );
        assert_eq!(
            (native.raw_width, native.raw_height, native.width, native.height, native.left_margin, native.top_margin),
            (oracle.raw_width, oracle.raw_height, oracle.width, oracle.height, oracle.left_margin, oracle.top_margin),
            "native geometry differs: native {}x{} active {}x{}+{}+{}, libraw {}x{} active {}x{}+{}+{}",
            native.raw_width,
            native.raw_height,
            native.width,
            native.height,
            native.left_margin,
            native.top_margin,
            oracle.raw_width,
            oracle.raw_height,
            oracle.width,
            oracle.height,
            oracle.left_margin,
            oracle.top_margin
        );
        assert_eq!(native.bits, oracle.bits, "bit depth");
        assert_eq!(native.filters, oracle.filters, "CFA filters");
        assert_eq!(native.colors, oracle.colors, "colour count");
        assert_eq!(native.black, oracle.black, "black level");
        assert_eq!(native.maximum, oracle.maximum, "white level");
        eprintln!(
            "native cam_mul {:?} rgb_cam {:?}\nlibraw cam_mul {:?} rgb_cam {:?}",
            native.cam_mul, native.rgb_cam, oracle.cam_mul, oracle.rgb_cam
        );
        assert_eq!(native.data.len(), oracle.data.len(), "plane length");
        let mut mismatches = 0usize;
        let mut maximum = 0u16;
        let mut first = None;
        for (index, (&a, &b)) in native.data.iter().zip(&oracle.data).enumerate() {
            if a != b {
                mismatches += 1;
                maximum = maximum.max(a.abs_diff(b));
                first.get_or_insert((index / native.raw_width, index % native.raw_width, a, b));
            }
        }
        eprintln!(
            "{path:?}: {} samples, {mismatches} mismatches, max diff {maximum}, first {first:?}",
            native.data.len()
        );
        assert_eq!(mismatches, 0, "native Bayer differs from LibRaw");
    }

    /// Batch acceptance over a folder of real camera files.
    ///
    /// `AURORA_TEST_RAW_DIR=<folder> cargo test -p aurora-engine --lib
    ///  native_bayer_matches_libraw_batch -- --ignored --nocapture`
    ///
    /// Walks the folder (8 levels), decodes every supported RAW with the native
    /// front-end and LibRaw, and prints a TSV table. Formats the native side
    /// intentionally rejects are reported as SKIP, not silently ignored.
    #[test]
    #[ignore = "needs a folder of real RAW files (AURORA_TEST_RAW_DIR) and libraw"]
    fn native_bayer_matches_libraw_batch() {
        fn collect(dir: &Path, depth: u8, out: &mut Vec<PathBuf>) {
            if depth > 8 {
                return;
            }
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    collect(&path, depth + 1, out);
                } else if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
                    if matches!(
                        extension.to_ascii_lowercase().as_str(),
                        "nef" | "arw" | "cr2" | "cr3" | "dng" | "raf" | "orf" | "rw2"
                    ) {
                        out.push(path);
                    }
                }
            }
        }

        let dir = std::env::var("AURORA_TEST_RAW_DIR").expect("set AURORA_TEST_RAW_DIR");
        let mut files = Vec::new();
        collect(Path::new(&dir), 0, &mut files);
        files.sort();
        assert!(!files.is_empty(), "no supported RAW files under {dir}");

        println!("file\tdims\tbits\tfilters\tblack\tmax\ttags\tpixel_mismatch\tmax_diff\tresult");
        let mut checked = 0usize;
        let mut failed = 0usize;
        for path in &files {
            let name = path.file_name().and_then(|value| value.to_str()).unwrap_or("?");
            let native = match crate::aurora_raw::rust_raw::decode(path) {
                Ok(frame) => frame,
                Err(error) => {
                    println!("{name}\t-\t-\t-\t-\t-\t-\t-\t-\tSKIP: {error}");
                    continue;
                }
            };
            let oracle = match libraw::unpack_bayer(path) {
                Ok(frame) => frame,
                Err(error) => {
                    println!("{name}\t-\t-\t-\t-\t-\t-\t-\t-\tSKIP libraw: {error}");
                    continue;
                }
            };
            let tags_equal = native.raw_width == oracle.raw_width
                && native.raw_height == oracle.raw_height
                && native.width == oracle.width
                && native.height == oracle.height
                && native.left_margin == oracle.left_margin
                && native.top_margin == oracle.top_margin
                && native.bits == oracle.bits
                && native.filters == oracle.filters
                && native.colors == oracle.colors
                && native.black == oracle.black
                && native.maximum == oracle.maximum
                && native.cam_mul == oracle.cam_mul
                && native.rgb_cam == oracle.rgb_cam;
            let (mut mismatches, mut maximum) = (0usize, 0u16);
            if native.data.len() == oracle.data.len() {
                for (&a, &b) in native.data.iter().zip(&oracle.data) {
                    if a != b {
                        mismatches += 1;
                        maximum = maximum.max(a.abs_diff(b));
                    }
                }
            } else {
                mismatches = native.data.len().max(oracle.data.len());
                maximum = u16::MAX;
            }
            let ok = tags_equal && mismatches == 0;
            if !ok {
                failed += 1;
                eprintln!(
                    "  first native {:?}",
                    &native.data[..16.min(native.data.len())]
                );
                eprintln!(
                    "  first libraw {:?}",
                    &oracle.data[..16.min(oracle.data.len())]
                );
                let mut shown = 0usize;
                for (index, (&a, &b)) in native.data.iter().zip(&oracle.data).enumerate() {
                    if a != b {
                        eprintln!(
                            "  at {index} ({}, {}): native {a} libraw {b}",
                            index / native.raw_width,
                            index % native.raw_width
                        );
                        shown += 1;
                        if shown >= 6 {
                            break;
                        }
                    }
                }
            }
            checked += 1;
            println!(
                "{name}\t{}x{}\t{}\t{:#x}\t{}\t{}\t{}\t{mismatches}\t{maximum}\t{}",
                native.raw_width,
                native.raw_height,
                native.bits,
                native.filters,
                native.black,
                native.maximum,
                if tags_equal { "equal" } else { "DIFF" },
                if ok { "PASS" } else { "FAIL" }
            );
        }
        eprintln!("{checked}/{} files compared, {failed} failed", files.len());
        assert_eq!(failed, 0, "native Bayer differs from LibRaw on {failed} file(s)");
    }

    /// Pixel-only acceptance, useful while metadata fields are still being
    /// ported. `AURORA_TEST_RAW=<CR3/ARW/...> cargo test ... -- --exact --ignored`
    #[test]
    #[ignore = "needs a real RAW file (AURORA_TEST_RAW) and libraw"]
    fn native_pixels_match_libraw() {
        let path = std::env::var("AURORA_TEST_RAW").expect("set AURORA_TEST_RAW");
        let path = PathBuf::from(path);
        let native = crate::aurora_raw::rust_raw::decode(&path).expect("native decode");
        let oracle = libraw::unpack_bayer(&path).expect("LibRaw unpack");
        let mut mismatches = 0usize;
        let mut maximum = 0u16;
        let mut first = None;
        for (index, (&a, &b)) in native.data.iter().zip(&oracle.data).enumerate() {
            if a != b {
                mismatches += 1;
                maximum = maximum.max(a.abs_diff(b));
                first.get_or_insert((index / native.raw_width, index % native.raw_width, a, b));
            }
        }
        // Samples past the common prefix have no counterpart to match, so each
        // one is a mismatch; `maximum` stays the largest measured sample diff.
        mismatches += native.data.len().abs_diff(oracle.data.len());
        eprintln!(
            "{path:?}: native {}x{} libraw {}x{}, {} native vs {} libraw samples, {mismatches} mismatches, max diff {maximum}, first {first:?}",
            native.raw_width,
            native.raw_height,
            oracle.raw_width,
            oracle.raw_height,
            native.data.len(),
            oracle.data.len()
        );
        assert_eq!(mismatches, 0, "native sensor pixels differ from LibRaw");
    }

    /// Minimal little-endian TIFF whose IFD0 carries only Make and Model.
    fn z6iii_body_tiff() -> Vec<u8> {
        let make = b"NIKON CORPORATION\0";
        let model = b"NIKON Z6_3\0";
        let mut out = Vec::from(*b"II\x2a\x00\x08\x00\x00\x00");
        out.extend_from_slice(&2u16.to_le_bytes());
        let ifd0 = out.len();
        out.resize(ifd0 + 2 * 12, 0);
        out.extend_from_slice(&0u32.to_le_bytes());
        let make_offset = out.len();
        out.extend_from_slice(make);
        let model_offset = out.len();
        out.extend_from_slice(model);
        let entry = |out: &mut Vec<u8>, index: usize, tag: u16, count: usize, offset: usize| {
            let at = ifd0 + index * 12;
            out[at..at + 2].copy_from_slice(&tag.to_le_bytes());
            out[at + 2..at + 4].copy_from_slice(&2u16.to_le_bytes());
            out[at + 4..at + 8].copy_from_slice(&(count as u32).to_le_bytes());
            out[at + 8..at + 12].copy_from_slice(&(offset as u32).to_le_bytes());
        };
        entry(&mut out, 0, 0x010f, make.len(), make_offset);
        entry(&mut out, 1, 0x0110, model.len(), model_offset);
        out
    }

    /// Minimal uncompressed NEF: one 64x64 16-bit RGGB strip. The native
    /// front-end can decode it without LibRaw. These are not camera pixels.
    fn synthetic_uncompressed_nef() -> Vec<u8> {
        const W: usize = 64;
        const H: usize = 64;
        let make = b"NIKON CORPORATION\0";
        let model = b"NIKON Z 6_2\0";
        let filters = 0x9494_9494u32;
        let mut samples = Vec::with_capacity(W * H * 2);
        for y in 0..H {
            for x in 0..W {
                let c = ((filters >> ((((y << 1) & 14) + (x & 1)) << 1)) & 3) as u8;
                let value = match c {
                    0 => 16000u16,
                    2 => 12000,
                    _ => 14000,
                };
                samples.extend_from_slice(&value.to_le_bytes());
            }
        }
        let mut out = Vec::from(*b"II\x2a\x00\x08\x00\x00\x00");
        let count = 11u16;
        out.extend_from_slice(&count.to_le_bytes());
        let ifd0 = out.len();
        out.resize(ifd0 + count as usize * 12, 0);
        out.extend_from_slice(&0u32.to_le_bytes());
        let make_offset = out.len();
        out.extend_from_slice(make);
        let model_offset = out.len();
        out.extend_from_slice(model);
        if out.len() % 2 != 0 {
            out.push(0);
        }
        let strip_offset = out.len();
        out.extend_from_slice(&samples);

        let entry = |out: &mut Vec<u8>, index: usize, tag: u16, typ: u16, count: u32, value: [u8; 4]| {
            let at = ifd0 + index * 12;
            out[at..at + 2].copy_from_slice(&tag.to_le_bytes());
            out[at + 2..at + 4].copy_from_slice(&typ.to_le_bytes());
            out[at + 4..at + 8].copy_from_slice(&count.to_le_bytes());
            out[at + 8..at + 12].copy_from_slice(&value);
        };
        let offset = |value: usize| (value as u32).to_le_bytes();
        let short = |value: u16| {
            let mut bytes = [0u8; 4];
            bytes[..2].copy_from_slice(&value.to_le_bytes());
            bytes
        };
        entry(&mut out, 0, 0x0100, 4, 1, offset(W));
        entry(&mut out, 1, 0x0101, 4, 1, offset(H));
        entry(&mut out, 2, 0x0102, 3, 1, short(16));
        entry(&mut out, 3, 0x0103, 3, 1, short(1));
        entry(&mut out, 4, 0x0106, 3, 1, short(32803));
        entry(&mut out, 5, 0x010f, 2, make.len() as u32, offset(make_offset));
        entry(&mut out, 6, 0x0110, 2, model.len() as u32, offset(model_offset));
        entry(&mut out, 7, 0x0111, 4, 1, offset(strip_offset));
        entry(&mut out, 8, 0x0115, 3, 1, short(1));
        entry(&mut out, 9, 0x0116, 4, 1, offset(H));
        entry(&mut out, 10, 0x0117, 4, 1, offset(samples.len()));
        out
    }

    struct EnvGuard {
        key: &'static str,
        old: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let old = std::env::var_os(key);
            unsafe { std::env::set_var(key, value) };
            Self { key, old }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.old {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    #[test]
    fn decode_route_recognises_a_z6iii_body_on_this_platform() {
        if !c1_raw_enabled() {
            return;
        }
        let path = temp_file("z6iii-route.nef", &z6iii_body_tiff());
        let route = decode_route(&path);
        let expected = cfg!(not(windows)) || cfg!(feature = "z6iii-raw-export");
        assert_eq!(route.z6iii_feature, expected);
        assert_eq!(route.z6iii_nef, expected, "a Z6III body must be recognised");
        #[cfg(not(windows))]
        {
            assert_eq!(route.opencl, OpenClRouteState::NotCompiled);
            let generic = DecodeRoute { z6iii_nef: false, ..route };
            assert_ne!(
                route.pipeline_fingerprint,
                DECODE_STAGE_REVISIONS.pipeline_fingerprint(&generic),
                "the Z6III route must not share the generic Bayer stage shape"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn native_front_end_feeds_the_product_16_bit_route_without_libraw() {
        if !c1_raw_enabled() {
            return;
        }
        let bytes = synthetic_uncompressed_nef();
        let native = effective_bayer(&bytes, None)
            .expect("the native front-end must decode the synthetic NEF from bytes alone");
        assert_eq!((native.raw_width, native.raw_height), (64, 64));
        assert_eq!(native.data.len(), 64 * 64);

        let path = temp_file("native-product16.nef", &bytes);
        // Force the no-LibRaw branch so the test proves the native path rather
        // than whatever LibRaw happens to be installed on the test machine.
        let guard = EnvGuard::set("AURORA_NO_LIBRAW", "1");
        let decoded = decode16_traced(&path);
        drop(guard);
        let (width, height, pixels, provenance) =
            decoded.expect("the native C1 chain should decode the synthetic NEF");
        assert!(width >= 46 && height >= 46, "{width}x{height}");
        assert_eq!(pixels.len(), (width as usize) * (height as usize) * 4);
        assert_eq!(provenance.backend, DecodeBackend::AuroraCpuChain);
        assert_eq!(provenance.pixel_space, PixelSpace::CameraLinear16);
        assert!(!provenance.z6iii_linear);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn quick_embedded_preview_returns_embedded_jpeg() {
        let mut container = vec![0x49, 0x49, 0x2A, 0x00, 0, 0, 0, 0];
        container.extend_from_slice(&jpeg(64, 48, 4));
        let path = temp_file("quick-preview.nef", &container);
        let img = quick_embedded_preview(&path, 1600).expect("embedded preview");
        assert_eq!(img.dimensions(), (64, 48));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[ignore = "profiles quick embedded JPEG preview on a real NEF; set AURORA_TEST_RAW"]
    fn bench_quick_embedded_preview() {
        let path = std::env::var_os("AURORA_TEST_RAW")
            .map(std::path::PathBuf::from)
            .expect("set AURORA_TEST_RAW to a real NEF file");
        for round in 0..3 {
            let started = Instant::now();
            let img = quick_embedded_preview(&path, 1600).expect("quick embedded preview");
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            println!("QUICK_PREVIEW round={round} {}x{} ms={ms:.1}", img.width(), img.height());
        }
    }
}

