//! RAW import. When the system `libraw` shared library is present the sensor data is
//! demosaiced and white-balanced by the capture's own camera multipliers, output as
//! linear light in the camera's own colour space; otherwise the camera-embedded JPEG
//! is all we have. The demosaic filter is selectable through [`set_demosaic`] or the
//! `LUMEN_DEMOSAIC` environment variable; AHD stays the default.
//!
//! libraw is resolved at run time, so a build without it falls back to the
//! preview instead of failing. On Windows the DLL and its dependencies are
//! vendored under `vendor/libraw/bin`, which is added to the search path.

use std::io::{BufReader, ErrorKind, Read};
use std::path::Path;
use std::sync::atomic::{AtomicI32, Ordering};

use image::{ImageFormat, RgbaImage};

macro_rules! raw_trace {
    ($($arg:tt)*) => {
        if std::env::var_os("LUMEN_RAW_TRACE").is_some() {
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

/// Returns the largest embedded JPEG preview, decoded.
pub fn embedded_preview(path: &Path) -> anyhow::Result<RgbaImage> {
    let found = scan_best(path, |segment| {
        let decoded = image::load_from_memory_with_format(segment, ImageFormat::Jpeg).ok()?;
        let (w, h) = (decoded.width(), decoded.height());
        if w == 0 || h == 0 {
            return None;
        }
        Some((u64::from(w) * u64::from(h), decoded.to_rgba8()))
    })?;
    found.ok_or_else(|| anyhow::anyhow!("no embedded JPEG preview in {}", path.display()))
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

/// Parses a numeric `LUMEN_DEMOSAIC` value, rejecting filters libraw lacks here.
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

/// Resolves the filter in force: `LUMEN_DEMOSAIC` wins over [`set_demosaic`],
/// and AHD is the final default.
pub fn demosaic_quality() -> i32 {
    match std::env::var("LUMEN_DEMOSAIC") {
        Ok(text) => parse_demosaic(&text).unwrap_or_else(|| {
            log::warn!("LUMEN_DEMOSAIC={text:?} is not 0/1/2/3/11/12; using AHD");
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
    match libraw::decode16(path, demosaic_quality()) {
        Ok((width, height, rgba)) if same_picture(&luma_grid16(width, height, &rgba), path) => {
            remember_sensor(path, true);
            return Some((width, height, rgba));
        }
        Ok(_) => log::warn!(
            "libraw {}: the 16-bit sensor decode does not resemble the camera's own preview \
             (unsupported compression?); falling back to the 8-bit path",
            path.display()
        ),
        Err(err) => log::warn!("libraw {}: 16-bit {err}; falling back to the 8-bit path", path.display()),
    }
    None
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
    let b = luma_grid(&preview);
    // libraw hands over the sensor in its stored orientation while the camera's
    // preview is already turned, so a quarter turn must not count as a mismatch.
    // Correlation is used rather than a difference because the sensor decode is
    // linear and the preview is display-encoded: only the picture shape may be
    // compared, not the level.
    let straight = correlation(a, &b);
    let quarter = correlation(&transpose(a), &b);
    straight.max(quarter) > 0.90
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
    use std::sync::OnceLock;

    const RTLD_NOW: c_int = 2;

    #[cfg(target_os = "windows")]
    const LIBRARY_NAMES: [&str; 1] = ["libraw.dll\0"];
    #[cfg(not(target_os = "windows"))]
    const LIBRARY_NAMES: [&str; 2] = ["libraw.so.23\0", "libraw.so\0"];

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
        static API: OnceLock<Option<Api>> = OnceLock::new();
        API.get_or_init(load).as_ref()
    }

    fn load() -> Option<Api> {
        // The handle stays loaded for the process lifetime: the function pointers
        // below would dangle after a `dlclose`.
        let handle = LIBRARY_NAMES
            .into_iter()
            .find_map(|name| {
                let handle = load_library(name);
                (!handle.is_null()).then_some(handle)
            })?;
        Some(Api {
            init: symbol(handle, "libraw_init")?,
            open: symbol(handle, "libraw_open_file")?,
            #[cfg(target_os = "windows")]
            open_w: symbol(handle, "libraw_open_wfile")?,
            unpack_name: symbol(handle, "libraw_unpack_function_name")?,
            unpack: symbol(handle, "libraw_unpack")?,
            process: symbol(handle, "libraw_dcraw_process")?,
            make: symbol(handle, "libraw_dcraw_make_mem_image")?,
            clear: symbol(handle, "libraw_dcraw_clear_mem")?,
            close: symbol(handle, "libraw_close")?,
            output_bps: symbol(handle, "libraw_set_output_bps")?,
            no_auto_bright: symbol(handle, "libraw_set_no_auto_bright")?,
            output_color: symbol(handle, "libraw_set_output_color")?,
            gamma: symbol(handle, "libraw_set_gamma")?,
            demosaic: symbol(handle, "libraw_set_demosaic")?,
            iwidth: symbol(handle, "libraw_get_iwidth")?,
            iheight: symbol(handle, "libraw_get_iheight")?,
        })
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
        if let Some(dir) = std::env::var_os("LUMEN_LIBRAW_DIR") {
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
    fn load_library(name: &str) -> *mut c_void {
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

    fn start(path: &Path) -> Result<(*mut c_void, &'static Api, Vec<u16>), String> {
        let api = api().ok_or("libraw library not found")?;
        #[cfg(target_os = "windows")]
        let path = {
            use std::os::windows::ffi::OsStrExt;
            path.as_os_str().encode_wide().chain(std::iter::once(0)).collect::<Vec<u16>>()
        };
        #[cfg(not(target_os = "windows"))]
        let path = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| "path has a NUL")?;
        let data = unsafe { (api.init)(0) };
        if data.is_null() {
            return Err("libraw_init returned null".into());
        }
        Ok((data, api, path))
    }

    /// libraw opened, unpacked and processed; the memory image is in `pixels`.
    fn processed(path: &Path, quality: i32) -> Result<Guard, String> {
        let (data, api, path) = start(path)?;
        let mut guard = Guard { api, data, pixels: std::ptr::null_mut() };
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
        let out = unsafe { &*guard.pixels };
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
            unsafe { std::slice::from_raw_parts((guard.pixels as *const u8).add(offset), bytes) };
        Ok((out, raw, (width, height)))
    }

    pub fn decode(path: &Path, quality: i32) -> Result<RgbaImage, String> {
        let guard = processed(path, quality)?;
        let (out, raw, (width, height)) = view(&guard)?;
        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
        // libraw gives linear light; the sRGB encode happens here, at 16-bit
        // precision, so the shadows keep their gradation.
        let enc16 = |v: u16| encode_srgb(v as f32 / 65535.0);
        let enc8 = |v: u8| encode_srgb(v as f32 / 255.0);
        if out.bits == 16 {
            // 16-bit linear down to 8-bit linear: the caller treats these bytes as
            // linear light, so this is a scale, not a transfer curve.
            let wide: Vec<u16> = raw
                .chunks_exact(2)
                .map(|p| u16::from_le_bytes([p[0], p[1]]))
                .collect();
            match out.colors {
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
                colors => return Err(format!("colors {colors} unsupported")),
            }
        } else {
            match out.colors {
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
                colors => return Err(format!("colors {colors} unsupported")),
            }
        }
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
        let wide: Vec<u16> =
            raw.chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect();
        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
        match out.colors {
            3 => wide.chunks_exact(3).for_each(|p| rgba.extend_from_slice(&[p[0], p[1], p[2], 65535])),
            4 => wide.chunks_exact(4).for_each(|p| rgba.extend_from_slice(&[p[0], p[1], p[2], 65535])),
            1 => wide.iter().for_each(|g| rgba.extend_from_slice(&[*g, *g, *g, 65535])),
            colors => return Err(format!("colors {colors} unsupported")),
        }
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
        let dir = std::env::temp_dir().join(format!("lumen-raw-{}-{}", std::process::id(), name));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
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

    #[cfg(all(target_os = "windows", feature = "external-fixtures"))]
    mod windows_libraw {
        include!("../local-fixtures/raw-windows-tests.rs");
    }

    /// Needs a real camera file and a machine with libraw; run with
    /// `LUMEN_TEST_RAW=/path/to/file.nef cargo test -- --include-ignored`.
    #[test]
    #[ignore = "needs a real RAW file (LUMEN_TEST_RAW) and libraw"]
    fn decodes_sensor_data() {
        let Ok(path) = std::env::var("LUMEN_TEST_RAW") else {
            panic!("set LUMEN_TEST_RAW to a RAW file");
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
    /// `LUMEN_TEST_RAW=/path/to/file.nef cargo test -- --include-ignored`
    #[test]
    #[ignore = "needs a real RAW file (LUMEN_TEST_RAW) and libraw"]
    fn decodes_sensor_data_at_sixteen_bits() {
        let Ok(path) = std::env::var("LUMEN_TEST_RAW") else {
            panic!("set LUMEN_TEST_RAW to a RAW file");
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
}

