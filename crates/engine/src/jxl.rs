use std::ffi::{c_char, c_int, c_void};
#[cfg(windows)]
use std::ffi::OsStr;
#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::slice;

use anyhow::{anyhow, bail, Context, Result};
use image::RgbaImage;

type Status = c_int;
type CreateFn = unsafe extern "C" fn(*const c_void) -> *mut JxlDecoder;
type DestroyFn = unsafe extern "C" fn(*mut JxlDecoder);
type SetInputFn = unsafe extern "C" fn(*mut JxlDecoder, *const u8, usize);
type CloseInputFn = unsafe extern "C" fn(*mut JxlDecoder);
type StatusFn = unsafe extern "C" fn(*mut JxlDecoder) -> Status;
type SetCoalescingFn = unsafe extern "C" fn(*mut JxlDecoder, c_int) -> Status;
type SubscribeFn = unsafe extern "C" fn(*mut JxlDecoder, c_int) -> Status;
type OutBufferSizeFn = unsafe extern "C" fn(*const JxlDecoder, *const PixelFormat, *mut usize) -> Status;
type SetOutBufferFn = unsafe extern "C" fn(*mut JxlDecoder, *const PixelFormat, *mut c_void, usize) -> Status;
type BasicInfoFn = unsafe extern "C" fn(*const JxlDecoder, *mut BasicInfo) -> Status;
type FrameHeaderFn = unsafe extern "C" fn(*const JxlDecoder, *mut FrameHeader) -> Status;

#[repr(C)]
struct PixelFormat {
    num_channels: u32,
    data_type: u32,
    endianness: u32,
    align: usize,
}

#[repr(C)]
struct BasicInfo {
    have_container: c_int,
    xsize: u32,
    ysize: u32,
    bits_per_sample: u32,
    exponent_bits_per_sample: u32,
    intensity_target: f32,
    min_nits: f32,
    relative_to_max_display: c_int,
    linear_below: f32,
    uses_original_profile: c_int,
    have_preview: c_int,
    have_animation: c_int,
    orientation: u32,
    num_color_channels: u32,
    num_extra_channels: u32,
    alpha_bits: u32,
    alpha_exponent_bits: u32,
    alpha_premultiplied: c_int,
    preview_xsize: u32,
    preview_ysize: u32,
    animation_tps_numerator: u32,
    animation_tps_denominator: u32,
    animation_num_loops: u32,
    animation_have_timecodes: c_int,
    intrinsic_xsize: u32,
    intrinsic_ysize: u32,
    padding: [u8; 100],
}

#[repr(C)]
struct BlendInfo {
    blendmode: u32,
    source: u32,
    alpha: u32,
    clamp: c_int,
}

#[repr(C)]
struct LayerInfo {
    have_crop: c_int,
    crop_x0: i32,
    crop_y0: i32,
    xsize: u32,
    ysize: u32,
    blend_info: BlendInfo,
    save_as_reference: u32,
}

#[repr(C)]
struct FrameHeader {
    duration: u32,
    timecode: u32,
    name_length: u32,
    is_last: c_int,
    layer_info: LayerInfo,
}

#[repr(C)]
struct JxlDecoder {
    _private: [u8; 0],
}

const SUCCESS: Status = 0;
const ERROR: Status = 1;
const NEED_MORE_INPUT: Status = 2;
const BASIC_INFO: Status = 0x40;
const FRAME: Status = 0x400;
const FULL_IMAGE: Status = 0x1000;
const NEED_IMAGE_OUT_BUFFER: Status = 5;
const FLOAT: u32 = 0;
const NATIVE_ENDIAN: u32 = 0;
const BASIC_INFO_EVENT: c_int = 0x40;
const FRAME_EVENT: c_int = 0x400;
const FULL_IMAGE_EVENT: c_int = 0x1000;

struct Input {
    data: Vec<u8>,
    len: usize,
}

struct Decoder<'library> {
    handle: *mut JxlDecoder,
    symbols: &'library Symbols,
}

struct Symbols {
    create: CreateFn,
    destroy: DestroyFn,
    set_input: SetInputFn,
    close_input: CloseInputFn,
    process_input: StatusFn,
    set_coalescing: SetCoalescingFn,
    subscribe: SubscribeFn,
    out_buffer_size: OutBufferSizeFn,
    set_out_buffer: SetOutBufferFn,
    basic_info: BasicInfoFn,
    frame_header: FrameHeaderFn,
}

struct Library {
    handle: *mut c_void,
}

/// Typed absence for the POSIX loader: lists every candidate path that was
/// considered, so a failure names the searched locations instead of a generic
/// "could not load" message. Windows keeps its historical `AURORA_JXL_DIR`
/// error strings.
#[cfg(unix)]
#[derive(Debug, Clone)]
pub struct LibraryNotFound {
    searched: Vec<PathBuf>,
    details: Vec<String>,
}

#[cfg(unix)]
impl LibraryNotFound {
    /// Candidate paths that were searched, in order, including ones rejected
    /// because the file was absent or the dynamic loader failed.
    pub fn searched(&self) -> &[PathBuf] {
        &self.searched
    }
}

#[cfg(unix)]
impl std::fmt::Display for LibraryNotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let searched = self
            .searched
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "libjxl library not found; searched: {searched}")?;
        if !self.details.is_empty() {
            write!(f, "; load errors: {}", self.details.join("; "))?;
        }
        Ok(())
    }
}

#[cfg(unix)]
impl std::error::Error for LibraryNotFound {}

#[cfg(windows)]
impl Library {
    fn load() -> Result<Self> {
        let dir = std::env::var_os("AURORA_JXL_DIR")
            .map(PathBuf::from)
            .ok_or_else(|| anyhow!("AURORA_JXL_DIR is not set"))?;
        let path = dir.join("libjxl.dll");
        if !path.is_file() {
            bail!("libjxl.dll not found: {}", path.display());
        }
        let path_wide: Vec<u16> = wide(path.as_os_str());
        let handle = unsafe { LoadLibraryW(path_wide.as_ptr()) };
        if handle.is_null() {
            bail!("failed to load {}", path.display());
        }
        Ok(Self { handle })
    }
}

#[cfg(unix)]
impl Library {
    fn load() -> Result<Self> {
        let mut searched = Vec::new();
        let mut details = Vec::new();
        for candidate in library_candidates() {
            // Absolute/path-like candidates are checked on disk first so a
            // missing prefix entry cannot hide a later valid one. Bare names
            // (no separator) are resolved by the dynamic loader's own search
            // path, so they are handed to dlopen directly.
            let path_like = candidate.is_absolute() || candidate.components().count() > 1;
            if path_like && !candidate.is_file() {
                searched.push(candidate);
                continue;
            }
            match unsafe { dlopen_path(&candidate) } {
                Ok(handle) => return Ok(Self { handle }),
                Err(detail) => {
                    details.push(format!("{}: {detail}", candidate.display()));
                    searched.push(candidate);
                }
            }
        }
        Err(anyhow::Error::new(LibraryNotFound { searched, details }))
    }
}

#[cfg(not(any(windows, unix)))]
impl Library {
    fn load() -> Result<Self> {
        bail!("JPEG XL decoding is not available on this platform")
    }
}

impl Library {
    fn symbols(&self) -> Result<Symbols> {
        unsafe {
            Ok(Symbols {
                create: self.symbol("JxlDecoderCreate")?,
                destroy: self.symbol("JxlDecoderDestroy")?,
                set_input: self.symbol("JxlDecoderSetInput")?,
                close_input: self.symbol("JxlDecoderCloseInput")?,
                process_input: self.symbol("JxlDecoderProcessInput")?,
                set_coalescing: self.symbol("JxlDecoderSetCoalescing")?,
                subscribe: self.symbol("JxlDecoderSubscribeEvents")?,
                out_buffer_size: self.symbol("JxlDecoderImageOutBufferSize")?,
                set_out_buffer: self.symbol("JxlDecoderSetImageOutBuffer")?,
                basic_info: self.symbol("JxlDecoderGetBasicInfo")?,
                frame_header: self.symbol("JxlDecoderGetFrameHeader")?,
            })
        }
    }

    unsafe fn symbol<T>(&self, name: &str) -> Result<T> {
        let name = std::ffi::CString::new(name)?;
        let pointer = unsafe { self.lookup(&name) }?;
        Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&pointer) })
    }

    #[cfg(windows)]
    unsafe fn lookup(&self, name: &std::ffi::CStr) -> Result<*mut c_void> {
        let pointer = unsafe { GetProcAddress(self.handle, name.as_ptr() as *const c_char) };
        if pointer.is_null() {
            bail!("missing libjxl export {}", name.to_string_lossy());
        }
        Ok(pointer)
    }

    #[cfg(unix)]
    unsafe fn lookup(&self, name: &std::ffi::CStr) -> Result<*mut c_void> {
        unsafe { dlerror() };
        let pointer = unsafe { dlsym(self.handle, name.as_ptr()) };
        if pointer.is_null() {
            match unsafe { dl_error() } {
                Some(detail) => bail!("missing libjxl export {}: {detail}", name.to_string_lossy()),
                None => bail!("missing libjxl export {}", name.to_string_lossy()),
            }
        }
        Ok(pointer)
    }

    #[cfg(not(any(windows, unix)))]
    unsafe fn lookup(&self, _name: &std::ffi::CStr) -> Result<*mut c_void> {
        bail!("JPEG XL decoding is not available on this platform")
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            FreeLibrary(self.handle)
        };
        #[cfg(unix)]
        unsafe {
            dlclose(self.handle)
        };
    }
}

/// Built-in POSIX search order. `AURORA_JXL_LIB` is authoritative when set: it
/// is the only candidate, so a bad override fails instead of silently falling
/// back to an installed copy.
#[cfg(unix)]
fn library_candidates() -> Vec<PathBuf> {
    if let Some(override_path) = std::env::var_os("AURORA_JXL_LIB") {
        return vec![PathBuf::from(override_path)];
    }

    let mut paths = Vec::new();
    if cfg!(target_os = "macos") {
        const DIRS: &[&str] = &["/opt/homebrew/lib", "/usr/local/lib"];
        const NAMES: &[&str] = &[
            "libjxl.0.13.0.dylib",
            "libjxl.0.13.dylib",
            "libjxl.dylib",
        ];
        for dir in DIRS {
            for name in NAMES {
                paths.push(Path::new(dir).join(name));
            }
        }
        for name in NAMES {
            paths.push(PathBuf::from(name));
        }
    } else {
        const DIRS: &[&str] = &[
            "/usr/local/lib",
            "/usr/lib",
            "/usr/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
        ];
        const NAMES: &[&str] = &["libjxl.so.0.13", "libjxl.so.0", "libjxl.so"];
        for dir in DIRS {
            for name in NAMES {
                paths.push(Path::new(dir).join(name));
            }
        }
        for name in NAMES {
            paths.push(PathBuf::from(name));
        }
    }
    deduplicate(paths)
}

#[cfg(unix)]
fn deduplicate(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut unique: Vec<PathBuf> = Vec::with_capacity(paths.len());
    for path in paths {
        if !unique.contains(&path) {
            unique.push(path);
        }
    }
    unique
}

#[cfg(unix)]
unsafe fn dlopen_path(path: &Path) -> std::result::Result<*mut c_void, String> {
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes())
        .map_err(|_| "path contains an interior NUL byte".to_string())?;
    unsafe { dlerror() };
    let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW) };
    if handle.is_null() {
        Err(unsafe { dl_error() }.unwrap_or_else(|| "dlopen returned NULL".to_string()))
    } else {
        Ok(handle)
    }
}

#[cfg(unix)]
unsafe fn dl_error() -> Option<String> {
    let error = unsafe { dlerror() };
    if error.is_null() {
        None
    } else {
        Some(unsafe { std::ffi::CStr::from_ptr(error) }.to_string_lossy().into_owned())
    }
}

impl<'library> Decoder<'library> {
    fn new(symbols: &'library Symbols) -> Result<Self> {
        unsafe {
            let handle = (symbols.create)(std::ptr::null());
            if handle.is_null() {
                bail!("JxlDecoderCreate failed");
            }
            Ok(Self { symbols, handle })
        }
    }
}

impl Drop for Decoder<'_> {
    fn drop(&mut self) {
        unsafe { (self.symbols.destroy)(self.handle) };
    }
}

fn feed(data: &[u8], decoder: &Decoder<'_>) {
        unsafe {
            (decoder.symbols.set_input)(decoder.handle, data.as_ptr(), data.len());
            (decoder.symbols.close_input)(decoder.handle);
        }
}

const ORIGINAL_EXTENSIONS: &[&str] = &[
    "nef", "cr2", "cr3", "arw", "dng", "orf", "rw2", "raf", "pef", "srw", "iiq", "3fr", "fff",
    "raw", "rwl", "jpg", "jpeg", "png", "tif", "tiff", "webp", "heic", "heif",
];

fn is_cop_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("cop"))
}

fn is_heif_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("heic") || e.eq_ignore_ascii_case("heif"))
}

fn is_usable_original(path: &Path) -> bool {
    path.is_file() && !is_cop_path(path)
}

fn probe_stem_extensions(dir: &Path, stem: &std::ffi::OsStr) -> Option<PathBuf> {
    let stem_str = stem.to_str()?;
    for ext in ORIGINAL_EXTENSIONS {
        for candidate_ext in [ext.to_ascii_lowercase(), ext.to_ascii_uppercase()] {
            let candidate = dir.join(format!("{stem_str}.{candidate_ext}"));
            if is_usable_original(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

/// Resolves the original RAW or raster image corresponding to a Capture One
/// `.cop` cache file when `libjxl` is unavailable.
pub fn resolve_original(path: &Path) -> Option<PathBuf> {
    if !is_cop_path(path) {
        return None;
    }

    // 1. Double extension in the same directory: e.g. `_DSC5673.NEF.cop` -> `_DSC5673.NEF`.
    let stripped = path.with_extension("");
    if is_usable_original(&stripped) {
        return Some(stripped);
    }

    let stripped_name = stripped.file_name();

    // 2. Capture One Session/Catalog cache layout: `.../CaptureOne/Cache/.../<name>.cop`.
    let mut search_dirs = Vec::new();
    if let Some(parent) = path.parent() {
        search_dirs.push(parent.to_path_buf());
    }
    for ancestor in path.ancestors().skip(1) {
        if ancestor
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.eq_ignore_ascii_case("CaptureOne"))
            && let Some(session_root) = ancestor.parent()
        {
            if let Some(name) = stripped_name {
                let direct = session_root.join(name);
                if is_usable_original(&direct) {
                    return Some(direct);
                }
                let originals = session_root.join("Originals").join(name);
                if is_usable_original(&originals) {
                    return Some(originals);
                }
            }
            search_dirs.push(session_root.to_path_buf());
            search_dirs.push(session_root.join("Originals"));
            break;
        }
    }

    // 3. Same stem with a known RAW/raster extension (e.g. `photo.cop` -> `photo.NEF`).
    if let Some(stem) = stripped.file_stem() {
        for dir in &search_dirs {
            if let Some(found) = probe_stem_extensions(dir, stem) {
                return Some(found);
            }
        }
    }

    None
}

fn decode_original(orig: &Path) -> Result<RgbaImage> {
    if crate::raw::is_raw(orig) {
        return crate::raw::decode(orig);
    }
    if is_heif_path(orig) {
        return match crate::heif::decode(orig) {
            Ok(img) => Ok(img),
            Err(err) => image::open(orig)
                .map(|img| img.to_rgba8())
                .map_err(|fallback| anyhow!("libheif: {err:#}; image crate: {fallback}")),
        };
    }
    Ok(image::open(orig)
        .with_context(|| format!("failed to decode original image: {}", orig.display()))?
        .to_rgba8())
}

fn original_dimensions(orig: &Path) -> Option<(u32, u32)> {
    if crate::raw::is_raw(orig) {
        return crate::raw::sensor_dimensions(orig);
    }
    if is_heif_path(orig) {
        return crate::heif::dimensions(orig).ok();
    }
    image::image_dimensions(orig).ok()
}

pub fn decode_cop(path: impl AsRef<Path>) -> Result<RgbaImage> {
    let path = path.as_ref();
    let library = match Library::load() {
        Ok(library) => library,
        Err(load_err) => {
            if let Some(orig) = resolve_original(path) {
                log::warn!(
                    "libjxl unavailable ({load_err:#}); falling back to original file {} for .cop {}",
                    orig.display(),
                    path.display()
                );
                return decode_original(&orig).with_context(|| {
                    format!(
                        "fallback decode from {} for .cop {}",
                        orig.display(),
                        path.display()
                    )
                });
            }
            log::warn!(
                "libjxl unavailable ({load_err:#}); no original file found for .cop {}",
                path.display()
            );
            return Err(load_err);
        }
    };
    let symbols = library.symbols()?;

    let data = std::fs::read(path).with_context(|| {
        format!("failed to read .cop file: {}", path.display())
    })?;
    let decoder = Decoder::new(&symbols)?;
    feed(&data, &decoder);
    decode_image(&decoder)
}

fn basic_info_zero() -> BasicInfo {
    BasicInfo {
        have_container: 0,
        xsize: 0,
        ysize: 0,
        bits_per_sample: 0,
        exponent_bits_per_sample: 0,
        intensity_target: 0.0,
        min_nits: 0.0,
        relative_to_max_display: 0,
        linear_below: 0.0,
        uses_original_profile: 0,
        have_preview: 0,
        have_animation: 0,
        orientation: 0,
        num_color_channels: 0,
        num_extra_channels: 0,
        alpha_bits: 0,
        alpha_exponent_bits: 0,
        alpha_premultiplied: 0,
        preview_xsize: 0,
        preview_ysize: 0,
        animation_tps_numerator: 0,
        animation_tps_denominator: 0,
        animation_num_loops: 0,
        animation_have_timecodes: 0,
        intrinsic_xsize: 0,
        intrinsic_ysize: 0,
        padding: [0; 100],
    }
}

pub fn dimensions(path: impl AsRef<Path>) -> Option<(u32, u32)> {
    let path = path.as_ref();
    let library = match Library::load() {
        Ok(library) => library,
        Err(_) => {
            let orig = resolve_original(path)?;
            log::debug!(
                "libjxl unavailable; reading dimensions from original file {} for .cop {}",
                orig.display(),
                path.display()
            );
            return original_dimensions(&orig);
        }
    };
    let symbols = library.symbols().ok()?;
    let data = std::fs::read(path).ok()?;
    let decoder = Decoder::new(&symbols).ok()?;
    feed(&data, &decoder);
    unsafe {
        if (symbols.set_coalescing)(decoder.handle, 0) != SUCCESS {
            return None;
        }
        if (symbols.subscribe)(decoder.handle, BASIC_INFO_EVENT) != SUCCESS {
            return None;
        }
        loop {
            match (symbols.process_input)(decoder.handle) {
                BASIC_INFO => {
                    let mut info = basic_info_zero();
                    if (symbols.basic_info)(decoder.handle, &mut info) != SUCCESS {
                        return None;
                    }
                    return (info.xsize > 0 && info.ysize > 0).then_some((info.xsize, info.ysize));
                }
                NEED_MORE_INPUT | ERROR => return None,
                SUCCESS => return None,
                _ => continue,
            }
        }
    }
}

fn decode_image(decoder: &Decoder<'_>) -> Result<RgbaImage> {
    let symbols = decoder.symbols;
    unsafe {
        if (symbols.set_coalescing)(decoder.handle, 0) != SUCCESS {
            bail!("failed to disable JXL frame coalescing");
        }
        let events = BASIC_INFO_EVENT | FRAME_EVENT | FULL_IMAGE_EVENT;
        if (symbols.subscribe)(decoder.handle, events) != SUCCESS {
            bail!("failed to subscribe to JXL decoder events");
        }
    }

    let format = PixelFormat {
        num_channels: 3,
        data_type: FLOAT,
        endianness: NATIVE_ENDIAN,
        align: 12,
    };
    let mut main_size = (0u32, 0u32);
    let mut selected = false;
    let mut frame_pixels = Vec::new();
    let mut frame_size = (0u32, 0u32);

    loop {
        let status = unsafe { (symbols.process_input)(decoder.handle) };
        match status {
            BASIC_INFO => {
                let mut info = basic_info_zero();
                check(unsafe { (symbols.basic_info)(decoder.handle, &mut info) }, "get JXL basic info")?;
                main_size = (info.xsize, info.ysize);
            }
            FRAME => {
                let mut header = FrameHeader {
                    duration: 0,
                    timecode: 0,
                    name_length: 0,
                    is_last: 0,
                    layer_info: LayerInfo {
                        have_crop: 0,
                        crop_x0: 0,
                        crop_y0: 0,
                        xsize: 0,
                        ysize: 0,
                        blend_info: BlendInfo {
                            blendmode: 0,
                            source: 0,
                            alpha: 0,
                            clamp: 0,
                        },
                        save_as_reference: 0,
                    },
                };
                check(unsafe { (symbols.frame_header)(decoder.handle, &mut header) }, "get JXL frame header")?;
                frame_size = (header.layer_info.xsize, header.layer_info.ysize);
                selected = frame_size == main_size;
            }
            NEED_IMAGE_OUT_BUFFER => {
                let mut size = 0usize;
                check(unsafe { (symbols.out_buffer_size)(decoder.handle, &format, &mut size) }, "query JXL output buffer size")?;
                frame_pixels = vec![0u8; size];
                check(
                    unsafe {
                        (symbols.set_out_buffer)(
                            decoder.handle,
                            &format,
                            frame_pixels.as_mut_ptr().cast::<c_void>(),
                            size,
                        )
                    },
                    "set JXL output buffer",
                )?;
            }
            FULL_IMAGE => {
                if selected {
                    return rgba_image(frame_size, frame_pixels);
                }
            }
            SUCCESS => break,
            NEED_MORE_INPUT => bail!("unexpected end of JXL input"),
            ERROR => bail!("libjxl decoder error"),
            status => bail!("unexpected libjxl status {status:#x}"),
        }
    }

    bail!("JXL file does not contain a full-size proxy frame")
}

fn rgba_image((width, height): (u32, u32), bytes: Vec<u8>) -> Result<RgbaImage> {
    if width == 0 || height == 0 {
        bail!("invalid JXL frame dimensions {width}x{height}");
    }
    let pixel_count = usize::try_from(width)
        .ok()
        .and_then(|w| w.checked_mul(usize::try_from(height).ok()?))
        .context("JXL frame size exceeds memory limits")?;
    if bytes.len() != pixel_count * 3 * 4 {
        bail!(
            "unexpected JXL output size {} for {}x{} RGB float",
            bytes.len(),
            width,
            height
        );
    }
    let floats: &[f32] = unsafe { slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), pixel_count * 3) };
    let pixels = floats.chunks_exact(3).flat_map(|rgb| {
        [
            srgb8(rgb[0]),
            srgb8(rgb[1]),
            srgb8(rgb[2]),
            255,
        ]
    }).collect::<Vec<u8>>();
    RgbaImage::from_raw(width, height, pixels).context("failed to build RGBA image")
}

fn srgb8(value: f32) -> u8 {
    let value = value.clamp(0.0, 1.0);
    let converted = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (converted * 255.0 + 0.5) as u8
}

fn check(status: Status, action: &'static str) -> Result<()> {
    if status == SUCCESS {
        Ok(())
    } else {
        Err(anyhow!("failed to {action}: status {status:#x}"))
    }
}

#[cfg(windows)]
fn wide(value: &OsStr) -> Vec<u16> {
    value.encode_wide().chain(std::iter::once(0)).collect()
}

#[cfg(windows)]
unsafe extern "system" {
    fn LoadLibraryW(path: *const u16) -> *mut c_void;
    fn FreeLibrary(library: *mut c_void) -> c_int;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
}

#[cfg(unix)]
#[cfg_attr(target_os = "linux", link(name = "dl"))]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
    fn dlclose(handle: *mut c_void) -> c_int;
}

#[cfg(unix)]
const RTLD_NOW: c_int = 2;
#[cfg(test)]
mod tests {
    use super::{decode_cop, dimensions, resolve_original};
    #[cfg(unix)]
    use super::LibraryNotFound;
    use anyhow::Result;
    use std::fs;
    use std::path::{Path, PathBuf};

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn temp_dir(tag: &str) -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("aurora-jxl-{tag}-{}-{nanos}", std::process::id()))
    }

    fn with_missing_jxl<T>(dir: &Path, f: impl FnOnce() -> T) -> T {
        struct Restore {
            key: &'static str,
            previous: Option<std::ffi::OsString>,
        }
        impl Drop for Restore {
            fn drop(&mut self) {
                unsafe {
                    match &self.previous {
                        Some(value) => std::env::set_var(self.key, value),
                        None => std::env::remove_var(self.key),
                    }
                }
            }
        }
        #[cfg(windows)]
        let (key, value) = ("AURORA_JXL_DIR", dir.join("missing-jxl-dir"));
        #[cfg(unix)]
        let (key, value) = ("AURORA_JXL_LIB", dir.join("missing-libjxl.so"));
        let previous = std::env::var_os(key);
        unsafe { std::env::set_var(key, &value) };
        let _restore = Restore { key, previous };
        f()
    }

    #[cfg(windows)]
    const SAMPLE: &str = r"C:\Users\Jxcz2\Pictures\Capture One Catalog\Cache\Previews\2026\9\13\0\_DSC5673.NEF.cop";

    #[cfg(windows)]
    fn dir() -> PathBuf {
        std::env::var_os("AURORA_JXL_DIR").map(PathBuf::from).unwrap_or_default()
    }

    #[cfg(windows)]
    fn source() -> &'static Path {
        Path::new(SAMPLE)
    }

    #[cfg(windows)]
    #[test]
    fn decodes_real_proxy() -> Result<()> {
        let _guard = env_guard();
        if !source().is_file() || !dir().join("libjxl.dll").is_file() {
            eprintln!("skipped: real sample or libjxl.dll unavailable");
            return Ok(());
        }
        let image = decode_cop(source())?;
        assert_eq!((image.width(), image.height()), (3840, 2560));
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn missing_dll_returns_err() -> Result<()> {
        let _guard = env_guard();
        let lane = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let empty_dir = lane.join("empty-jxl");
        fs::create_dir_all(&empty_dir)?;
        let result = with_missing_jxl(&empty_dir, || decode_cop(source()));
        let _ = fs::remove_dir_all(&empty_dir);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn missing_dll_falls_back_to_original_image() -> Result<()> {
        let _guard = env_guard();
        let dir = temp_dir("fallback-same-dir");
        fs::create_dir_all(&dir)?;

        let mut rgba = image::RgbaImage::new(12, 8);
        for pixel in rgba.pixels_mut() {
            *pixel = image::Rgba([11, 22, 33, 255]);
        }
        let orig = dir.join("frame.png");
        rgba.save(&orig)?;
        let cop = dir.join("frame.png.cop");
        fs::write(&cop, b"not-jxl")?;

        let (decoded, dims) = with_missing_jxl(&dir, || (decode_cop(&cop), dimensions(&cop)));
        let _ = fs::remove_dir_all(&dir);

        let image = decoded?;
        assert_eq!(dims, Some((12, 8)));
        assert_eq!(image.dimensions(), (12, 8));
        assert_eq!(image.get_pixel(0, 0).0, [11, 22, 33, 255]);
        Ok(())
    }

    #[test]
    fn missing_dll_falls_back_from_capture_one_cache_layout() -> Result<()> {
        let _guard = env_guard();
        let dir = temp_dir("fallback-c1-layout");
        let proxies = dir.join("CaptureOne").join("Cache").join("Proxies");
        fs::create_dir_all(&proxies)?;

        let mut rgba = image::RgbaImage::new(10, 10);
        for pixel in rgba.pixels_mut() {
            *pixel = image::Rgba([200, 100, 50, 255]);
        }
        let orig = dir.join("shot.png");
        rgba.save(&orig)?;
        let cop = proxies.join("shot.cop");
        fs::write(&cop, b"not-jxl")?;

        assert_eq!(resolve_original(&cop).as_deref(), Some(orig.as_path()));
        let (decoded, dims) = with_missing_jxl(&dir, || (decode_cop(&cop), dimensions(&cop)));
        let _ = fs::remove_dir_all(&dir);

        let image = decoded?;
        assert_eq!(dims, Some((10, 10)));
        assert_eq!(image.dimensions(), (10, 10));
        assert_eq!(image.get_pixel(0, 0).0, [200, 100, 50, 255]);
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn invalid_cop_returns_err() -> Result<()> {
        if !dir().join("libjxl.dll").is_file() {
            eprintln!("skipped: libjxl.dll unavailable");
            return Ok(());
        }
        let lane = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = lane.join("invalid.cop");
        fs::write(&path, b"not jxl")?;
        let result = decode_cop(&path);
        fs::remove_file(&path)?;
        assert!(result.is_err());
        Ok(())
    }

    // POSIX loader tests. They use the real Homebrew libjxl when present and
    // skip cleanly when only the encoder (`cjxl`) or the library is missing.

    #[cfg(unix)]
    fn unique_temp_dir(tag: &str) -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("aurora-jxl-{tag}-{}-{nanos}", std::process::id()))
    }

    #[cfg(unix)]
    fn with_override<T>(path: &Path, f: impl FnOnce() -> T) -> T {
        struct Restore {
            key: &'static str,
            previous: Option<std::ffi::OsString>,
        }
        impl Drop for Restore {
            fn drop(&mut self) {
                unsafe {
                    match &self.previous {
                        Some(value) => std::env::set_var(self.key, value),
                        None => std::env::remove_var(self.key),
                    }
                }
            }
        }
        let key = "AURORA_JXL_LIB";
        let previous = std::env::var_os(key);
        unsafe { std::env::set_var(key, path) };
        let _restore = Restore { key, previous };
        f()
    }

    #[cfg(unix)]
    #[test]
    fn missing_override_reports_searched_candidates() -> Result<()> {
        let _guard = env_guard();
        let dir = unique_temp_dir("missing-override");
        let missing = dir.join("libjxl-does-not-exist.dylib");
        let (decoded, header) = with_override(&missing, || {
            (
                decode_cop(Path::new("no-such.cop")),
                dimensions(Path::new("no-such.cop")),
            )
        });

        assert_eq!(header, None, "dimensions must not panic when absent");
        let error = decoded.expect_err("a missing override must not fall back");
        let typed = error
            .downcast_ref::<LibraryNotFound>()
            .expect("absence must be a typed LibraryNotFound");
        assert_eq!(typed.searched(), std::slice::from_ref(&missing));
        let message = format!("{error:#}");
        assert!(
            message.contains("libjxl library not found; searched:"),
            "{message}"
        );
        assert!(message.contains("libjxl-does-not-exist.dylib"), "{message}");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn invalid_jxl_returns_err() -> Result<()> {
        let _guard = env_guard();
        if super::Library::load().is_err() {
            eprintln!("skipped: libjxl library unavailable");
            return Ok(());
        }
        let dir = unique_temp_dir("invalid");
        fs::create_dir_all(&dir)?;
        let path = dir.join("invalid.jxl");
        fs::write(&path, b"not jxl")?;
        let result = decode_cop(&path);
        fs::remove_dir_all(&dir)?;
        assert!(result.is_err());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn cjxl_synthetic_roundtrip() -> Result<()> {
        let _guard = env_guard();
        let dir = unique_temp_dir("synthetic");
        fs::create_dir_all(&dir)?;
        let png = dir.join("synthetic.png");
        let jxl = dir.join("synthetic.jxl");

        let mut rgba = image::RgbaImage::new(8, 8);
        for (x, y, pixel) in rgba.enumerate_pixels_mut() {
            *pixel = image::Rgba([(x * 32) as u8, (y * 32) as u8, 128, 255]);
        }
        rgba.save(&png)?;

        let cjxl = std::env::var_os("AURORA_CJXL").unwrap_or_else(|| "cjxl".into());
        let encoded = std::process::Command::new(&cjxl)
            .arg(&png)
            .arg(&jxl)
            .args(["-d", "0", "-e", "1"])
            .output();
        let cleanup = || {
            let _ = fs::remove_dir_all(&dir);
        };
        let encoded = match encoded {
            Ok(output) => output,
            Err(error) => {
                eprintln!("skipped: cjxl unavailable: {error}");
                cleanup();
                return Ok(());
            }
        };
        if !encoded.status.success() || !jxl.is_file() {
            eprintln!(
                "skipped: cjxl could not encode: {}",
                String::from_utf8_lossy(&encoded.stderr)
            );
            cleanup();
            return Ok(());
        }

        let image = match decode_cop(&jxl) {
            Ok(image) => image,
            Err(error) if error.downcast_ref::<LibraryNotFound>().is_some() => {
                eprintln!("skipped: libjxl library unavailable: {error:#}");
                cleanup();
                return Ok(());
            }
            Err(error) => {
                cleanup();
                return Err(error);
            }
        };
        let header = dimensions(&jxl);
        cleanup();

        assert_eq!((image.width(), image.height()), (8, 8));
        assert_eq!(header, Some((8, 8)));
        assert!(image.pixels().all(|pixel| pixel.0[3] == 255));
        Ok(())
    }
}
