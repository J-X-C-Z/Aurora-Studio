use std::ffi::{c_char, c_int, c_void};
#[cfg(windows)]
use std::ffi::OsStr;
#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt as _;
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

impl Library {
    #[cfg(windows)]
    fn load(dir: &Path) -> Result<Self> {
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

    #[cfg(unix)]
    fn load(dir: &Path) -> Result<Self> {
        #[cfg(target_os = "macos")]
        const NAMES: [&str; 3] = ["libjxl.dylib", "libjxl.0.dylib", "libjxl.1.dylib"];
        #[cfg(not(target_os = "macos"))]
        const NAMES: [&str; 3] = ["libjxl.so", "libjxl.so.0", "libjxl.so.1"];
        let mut tried: Vec<String> = Vec::new();
        // A library shipped beside the executable (or pointed at by the
        // AURORA_JXL_DIR the caller resolved) wins over the system search path.
        for name in NAMES {
            let path = dir.join(name);
            if !path.is_file() {
                continue;
            }
            if let Ok(c) = std::ffi::CString::new(path.as_os_str().as_bytes()) {
                let handle = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
                if !handle.is_null() {
                    return Ok(Self { handle });
                }
            }
            tried.push(path.display().to_string());
        }
        // Fall back to the dynamic loader's own search path (DYLD_LIBRARY_PATH
        // and the installed system copy on macOS, ldconfig paths on Linux).
        for name in NAMES {
            if let Ok(c) = std::ffi::CString::new(name) {
                let handle = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
                if !handle.is_null() {
                    return Ok(Self { handle });
                }
            }
        }
        bail!(
            "libjxl not found in {} or the dynamic loader search path (tried {})",
            dir.display(),
            tried.join(", ")
        )
    }

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
        #[cfg(windows)]
        let pointer = unsafe { GetProcAddress(self.handle, name.as_ptr() as *const c_char) };
        #[cfg(unix)]
        let pointer = unsafe { dlsym(self.handle, name.as_ptr()) };
        if pointer.is_null() {
            bail!("missing libjxl export {}", name.to_string_lossy());
        }
        Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&pointer) })
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            FreeLibrary(self.handle);
        }
        #[cfg(unix)]
        unsafe {
            dlclose(self.handle);
        }
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

pub fn decode_cop(path: impl AsRef<Path>) -> Result<RgbaImage> {
    let dir = std::env::var_os("AURORA_JXL_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("AURORA_JXL_DIR is not set"))?;
    let library = Library::load(&dir)?;
    let symbols = library.symbols()?;

    let data = std::fs::read(path.as_ref()).with_context(|| {
        format!("failed to read .cop file: {}", path.as_ref().display())
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
    let dir = std::env::var_os("AURORA_JXL_DIR").map(PathBuf::from)?;
    let library = Library::load(&dir).ok()?;
    let symbols = library.symbols().ok()?;
    let data = std::fs::read(path.as_ref()).ok()?;
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
const RTLD_NOW: c_int = 2;
#[cfg(unix)]
#[cfg_attr(target_os = "linux", link(name = "dl"))]
unsafe extern "C" {
    fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::decode_cop;
    use anyhow::Result;
    use std::fs;
    use std::path::{Path, PathBuf};

    const SAMPLE: &str = r"C:\Users\Jxcz2\Pictures\Capture One Catalog\Cache\Previews\2026\9\13\0\_DSC5673.NEF.cop";

    fn dir() -> PathBuf {
        std::env::var_os("AURORA_JXL_DIR").map(PathBuf::from).unwrap_or_default()
    }

    fn source() -> &'static Path {
        Path::new(SAMPLE)
    }

    #[test]
    fn decodes_real_proxy() -> Result<()> {
        if !source().is_file() || !dir().join("libjxl.dll").is_file() {
            eprintln!("skipped: real sample or libjxl.dll unavailable");
            return Ok(());
        }
        let image = decode_cop(source())?;
        assert_eq!((image.width(), image.height()), (3840, 2560));
        Ok(())
    }

    #[test]
    fn missing_dll_returns_err() -> Result<()> {
        let lane = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let empty_dir = lane.join("empty-jxl");
        fs::create_dir_all(&empty_dir)?;
        let previous = std::env::var_os("AURORA_JXL_DIR");
        unsafe { std::env::set_var("AURORA_JXL_DIR", &empty_dir) };
        let result = decode_cop(source());
        match previous {
            Some(value) => unsafe { std::env::set_var("AURORA_JXL_DIR", value) },
            None => unsafe { std::env::remove_var("AURORA_JXL_DIR") },
        }
        assert!(result.is_err());
        Ok(())
    }

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
}
