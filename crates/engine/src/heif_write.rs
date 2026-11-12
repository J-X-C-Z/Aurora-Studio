//! HEIF/HEIC writing through the runtime libheif library.
//!
//! The encoder is optional: when the host has no `libheif` (or no HEVC
//! encoder) the job fails with a structured error instead of leaving a half
//! written file behind. `LUMEN_HEIF_LIB` points at a non-standard library.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr;

use anyhow::{Context, Result, anyhow, bail};

type Handle = *mut c_void;
type HeifError = *mut c_void;
type HeifContext = *mut c_void;
type HeifImage = *mut c_void;
type HeifEncoder = *mut c_void;

const RTLD_NOW: c_int = 2;
const HEIF_COLORSPACE_RGB: c_int = 99;
const HEIF_CHANNEL_INTERLEAVED_RGBA: c_int = 131;
const HEIF_CHROMA_444: c_int = 4;

struct Api {
    handle: Handle,
    path: PathBuf,
    error_code: unsafe extern "C" fn(HeifError) -> c_int,
    error_message: unsafe extern "C" fn(HeifError) -> *const c_char,
    error_free: unsafe extern "C" fn(HeifError),
    context_alloc: unsafe extern "C" fn() -> HeifContext,
    context_free: unsafe extern "C" fn(HeifContext),
    image_create: unsafe extern "C" fn(c_int, u32, u32, c_int) -> HeifImage,
    image_free: unsafe extern "C" fn(HeifImage),
    image_add_plane:
        unsafe extern "C" fn(HeifImage, c_int, c_int, c_int, c_int) -> c_int,
    image_get_plane:
        unsafe extern "C" fn(HeifImage, c_int, c_int) -> *mut u8,
    image_set_stride:
        unsafe extern "C" fn(HeifImage, c_int, c_int) -> c_int,
    context_add_image:
        unsafe extern "C" fn(HeifContext, HeifImage) -> c_int,
    get_encoder_for_format:
        unsafe extern "C" fn(HeifContext, c_int) -> HeifEncoder,
    encoder_free: unsafe extern "C" fn(HeifEncoder),
    encoder_set_quality:
        unsafe extern "C" fn(HeifEncoder, c_int) -> c_int,
    encoder_set_lossless:
        unsafe extern "C" fn(HeifEncoder, c_int) -> c_int,
    encoder_has_lossless:
        unsafe extern "C" fn(HeifEncoder) -> c_int,
    encoder_has_quality:
        unsafe extern "C" fn(HeifEncoder) -> c_int,
    context_encode_image:
        unsafe extern "C" fn(HeifContext, HeifImage, HeifEncoder) -> c_int,
    context_write_to_file:
        unsafe extern "C" fn(HeifContext, *const c_char) -> c_int,
}

struct Symbols {
    names: &'static [&'static str],
}

impl Symbols {
    const REQUIRED: Self = Self {
        names: &[
            "heif_context_alloc",
            "heif_context_free",
            "heif_image_create",
            "heif_image_release",
            "heif_image_add_plane",
            "heif_image_get_plane",
            "heif_image_set_stride",
            "heif_context_add_image",
            "heif_context_get_encoder_for_format",
            "heif_encoder_release",
            "heif_encoder_set_quality",
            "heif_encoder_set_lossless",
            "heif_encoder_has_lossless",
            "heif_encoder_has_quality",
            "heif_context_encode_image",
            "heif_context_write_to_file",
        ],
    };
}

#[cfg(windows)]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> Handle;
    fn GetProcAddress(module: Handle, name: *const u8) -> *mut c_void;
}

#[cfg(not(windows))]
unsafe extern "C" {
    fn dlopen(file: *const c_char, mode: c_int) -> Handle;
    fn dlsym(handle: Handle, symbol: *const c_char) -> *mut c_void;
}

#[cfg(windows)]
unsafe fn open(path: &Path) -> Result<Handle> {
    use std::os::windows::ffi::OsStrExt as _;
    let wide: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
    if handle.is_null() {
        bail!("failed to load {}", path.display());
    }
    Ok(handle)
}

#[cfg(not(windows))]
unsafe fn open(path: &Path) -> Result<Handle> {
    let bytes = path.as_os_str().to_string_lossy().as_bytes().to_vec();
    let name = CString::new(bytes)?;
    let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
    if handle.is_null() {
        bail!("failed to load {}", path.display());
    }
    Ok(handle)
}

#[cfg(windows)]
unsafe fn symbol(handle: Handle, name: &str) -> Result<*mut c_void> {
    let name = CString::new(name)?;
    let address = unsafe { GetProcAddress(handle, name.as_ptr() as *const u8) };
    if address.is_null() {
        bail!("missing export {name:?}");
    }
    Ok(address)
}

#[cfg(not(windows))]
unsafe fn symbol(handle: Handle, name: &str) -> Result<*mut c_void> {
    let name = CString::new(name)?;
    let address = unsafe { dlsym(handle, name.as_ptr()) };
    if address.is_null() {
        bail!("missing export {name:?}");
    }
    Ok(address)
}

fn symbol_names(handle: Handle, api: &Symbols) -> Result<()> {
    for name in api.names {
        if unsafe { symbol(handle, name) }.is_err() {
            bail!("missing export {name:?}");
        }
    }
    Ok(())
}

fn check(status: c_int, error: HeifError, api: &Api, what: &str) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let code = unsafe { (api.error_code)(error) };
    let message = unsafe { CStr::from_ptr((api.error_message)(error)) }
        .to_string_lossy()
        .into_owned();
    unsafe { (api.error_free)(error) };
    Err(anyhow!("{what} failed: libheif error {code}: {message}"))
}

fn free(handle: Handle) {
    if cfg!(windows) {
        unsafe extern "system" {
            fn FreeLibrary(module: Handle) -> c_int;
        }
        unsafe { FreeLibrary(handle) };
    } else {
        unsafe extern "C" {
            fn dlclose(handle: Handle) -> c_int;
        }
        unsafe { dlclose(handle) };
    }
}

fn library_candidates() -> Vec<PathBuf> {
    if let Some(path) = std::env::var_os("LUMEN_HEIF_LIB") {
        return vec![PathBuf::from(path)];
    }
    if cfg!(windows) {
        vec!["libheif.dll".into(), "heif.dll".into()]
    } else {
        vec!["libheif.so.1".into(), "libheif.so".into()]
    }
}

fn missing_library_error(reason: &str) -> anyhow::Error {
    anyhow!(
        "HEIF export needs libheif.dll with a HEVC encoder (libx265) and decoder (libde265); \
         no suitable library was found. Set LUMEN_HEIF_LIB to the full path of libheif.dll. \
         Load error: {reason}"
    )
}

/// Encodes RGBA pixels to `path` using the runtime libheif encoder.
///
/// `quality` is the recipe's 0..=100 scale; 100 requests lossless when the
/// encoder supports it, otherwise it becomes the highest lossy quality.
pub fn encode(path: &Path, rgba: &[u8], width: u32, height: u32, quality: u8) -> Result<()> {
    if width == 0 || height == 0 {
        bail!("HEIF export needs a non-empty image");
    }
    let expected = width as usize * height as usize * 4;
    if rgba.len() != expected {
        bail!("HEIF export: buffer is {} bytes, expected {expected}", rgba.len());
    }
    let quality = quality.clamp(1, 100);
    let candidates = library_candidates();
    let mut load_error = String::new();
    let mut api = None;
    for candidate in candidates {
        if !candidate.is_absolute() {
            load_error.push_str(&format!("{} not found; ", candidate.display()));
            continue;
        }
        let handle = match unsafe { open(&candidate) } {
            Ok(handle) => handle,
            Err(err) => {
                load_error.push_str(&format!("{}: {err}; ", candidate.display()));
                continue;
            }
        };
        if let Err(err) = symbol_names(handle, &Symbols::REQUIRED) {
            free(handle);
            load_error.push_str(&format!("{}: {err}; ", candidate.display()));
            continue;
        }
        api = Some((handle, candidate));
        break;
    }
    let Some((handle, library_path)) = api else {
        if load_error.is_empty() {
            let names = library_candidates()
                .into_iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            load_error = format!("no absolute library path to load ({names})");
        }
        return Err(missing_library_error(&load_error));
    };
    let result = unsafe { encode_with(handle, &library_path, path, rgba, width, height, quality) };
    free(handle);
    result
}

unsafe fn encode_with(
    handle: Handle,
    library_path: &Path,
    output: &Path,
    rgba: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<()> {
    unsafe fn function<T: Copy>(handle: Handle, name: &str) -> Result<T> {
        let address = unsafe { symbol(handle, name)? };
        Ok(unsafe { std::mem::transmute_copy(&address) })
    }
    let error_code = unsafe { function(handle, "heif_error_code")? };
    let error_message = unsafe { function(handle, "heif_error_message")? };
    let error_free = unsafe { function(handle, "heif_error_free")? };
    let context_alloc = unsafe { function(handle, "heif_context_alloc")? };
    let context_free = unsafe { function(handle, "heif_context_free")? };
    let image_create = unsafe { function(handle, "heif_image_create")? };
    let image_free = unsafe { function(handle, "heif_image_release")? };
    let image_add_plane = unsafe { function(handle, "heif_image_add_plane")? };
    let image_get_plane = unsafe { function(handle, "heif_image_get_plane")? };
    let image_set_stride = unsafe { function(handle, "heif_image_set_stride")? };
    let context_add_image = unsafe { function(handle, "heif_context_add_image")? };
    let get_encoder_for_format = unsafe {
        function::<unsafe extern "C" fn(HeifContext, c_int) -> HeifEncoder>(
            handle,
            "heif_context_get_encoder_for_format",
        )?
    };
    let encoder_free = unsafe { function(handle, "heif_encoder_release")? };
    let encoder_set_quality = unsafe { function(handle, "heif_encoder_set_quality")? };
    let encoder_set_lossless = unsafe { function(handle, "heif_encoder_set_lossless")? };
    let encoder_has_lossless = unsafe { function(handle, "heif_encoder_has_lossless")? };
    let encoder_has_quality = unsafe { function(handle, "heif_encoder_has_quality")? };
    let context_encode_image = unsafe { function(handle, "heif_context_encode_image")? };
    let context_write_to_file = unsafe { function(handle, "heif_context_write_to_file")? };
    let api = Api {
        handle,
        path: library_path.to_path_buf(),
        error_code,
        error_message,
        error_free,
        context_alloc,
        context_free,
        image_create,
        image_free,
        image_add_plane,
        image_get_plane,
        image_set_stride,
        context_add_image,
        get_encoder_for_format,
        encoder_free,
        encoder_set_quality,
        encoder_set_lossless,
        encoder_has_lossless,
        encoder_has_quality,
        context_encode_image,
        context_write_to_file,
    };
    let error = ptr::null_mut();
    let context = unsafe { (api.context_alloc)() };
    if context.is_null() {
        bail!("libheif could not allocate an encoder context");
    }
    let mut image = ptr::null_mut();
    let mut encoder = ptr::null_mut();
    let mut tmp = output.with_extension("tmp");
    let result = (|| {
        let name = output.file_stem().and_then(|s| s.to_str()).context("output name is not UTF-8")?;
        let extension = output.extension().and_then(|s| s.to_str()).unwrap_or("heic");
        tmp.set_file_name(format!("{name}.heifwrite.tmp.{extension}"));
        let tmp_name = CString::new(tmp.as_os_str().to_string_lossy().as_bytes())?;
        let status = unsafe { (api.image_create)(HEIF_COLORSPACE_RGB, width, height, HEIF_CHROMA_444) };
        if status.is_null() {
            bail!("libheif could not allocate the image");
        }
        image = status;
        let bits = 8;
        let stride = width as usize * 4;
        let status = unsafe {
            (api.image_add_plane)(
                image,
                HEIF_CHANNEL_INTERLEAVED_RGBA,
                width as c_int,
                height as c_int,
                bits,
            )
        };
        if status != 0 {
            bail!("libheif could not allocate the RGBA plane");
        }
        let plane = unsafe {
            (api.image_get_plane)(image, HEIF_CHANNEL_INTERLEAVED_RGBA, 0)
        };
        if plane.is_null() {
            bail!("libheif returned no RGBA plane");
        }
        let status = unsafe { (api.image_set_stride)(image, HEIF_CHANNEL_INTERLEAVED_RGBA, stride as c_int) };
        if status != 0 {
            bail!("libheif rejected the stride");
        }
        for (row, source) in rgba.chunks_exact(stride).enumerate() {
            let target = unsafe { plane.add(row * stride) };
            unsafe { std::ptr::copy_nonoverlapping(source.as_ptr(), target, stride) };
        }
        encoder = unsafe { (api.get_encoder_for_format)(context, 4) };
        if encoder.is_null() {
            bail!("libheif has no HEVC encoder; install libheif with libx265");
        }
        if quality >= 100 && unsafe { (api.encoder_has_lossless)(encoder) } != 0 {
            let status = unsafe { (api.encoder_set_lossless)(encoder, 1) };
            check(status, error, &api, "enabling lossless HEIF")?;
        } else if unsafe { (api.encoder_has_quality)(encoder) } != 0 {
            let status = unsafe { (api.encoder_set_quality)(encoder, quality as c_int) };
            check(status, error, &api, "setting HEIF quality")?;
        }
        let status = unsafe { (api.context_add_image)(context, image) };
        check(status, error, &api, "adding the image")?;
        let status = unsafe { (api.context_encode_image)(context, image, encoder) };
        check(status, error, &api, "encoding HEIF")?;
        let status = unsafe { (api.context_write_to_file)(context, tmp_name.as_ptr()) };
        check(status, error, &api, "writing HEIF")?;
        if tmp.is_file() {
            std::fs::rename(&tmp, output).with_context(|| {
                format!("replacing {} with the encoded HEIF", output.display())
            })?;
        } else {
            bail!("libheif reported success but did not write {}", tmp.display());
        }
        Ok(())
    })();
    if !encoder.is_null() {
        unsafe { (api.encoder_free)(encoder) };
    }
    if !image.is_null() {
        unsafe { (api.image_free)(image) };
    }
    unsafe { (api.context_free)(context) };
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_file(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "lumen-heifwrite-{}-{name}",
            std::process::id()
        ));
        path
    }

    fn pixels(width: u32, height: u32) -> Vec<u8> {
        (0..width as usize * height as usize * 4)
            .map(|i| (i * 7 % 255) as u8)
            .collect()
    }

    #[test]
    fn missing_library_reports_structured_error() {
        // Run with LUMEN_HEIF_LIB set to a missing file to exercise the path.
        let previous = std::env::var_os("LUMEN_HEIF_LIB");
        unsafe { std::env::set_var("LUMEN_HEIF_LIB", "Z:/definitely/not/libheif.dll") };
        let path = temp_file("missing.heic");
        let err = encode(&path, &pixels(2, 2), 2, 2, 80).unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("LUMEN_HEIF_LIB"), "{text}");
        assert!(text.contains("libheif.dll"), "{text}");
        assert!(text.contains("libx265"), "{text}");
        assert!(!path.exists());
        match previous {
            Some(value) => unsafe { std::env::set_var("LUMEN_HEIF_LIB", value) },
            None => unsafe { std::env::remove_var("LUMEN_HEIF_LIB") },
        }
    }

    #[test]
    fn encode_reads_back_when_encoder_available() {
        let path = temp_file("roundtrip.heic");
        let _ = std::fs::remove_file(&path);
        let rgba = pixels(16, 12);
        let result = encode(&path, &rgba, 16, 12, 80);
        match result {
            Ok(()) => {
                let dimensions = image::image_dimensions(&path).expect("read HEIF dimensions");
                assert_eq!(dimensions, (16, 12));
            }
            Err(err) => {
                let text = format!("{err:#}");
                assert!(
                    text.contains("no suitable library")
                        || text.contains("no HEVC encoder"),
                    "unexpected HEIF result: {text}"
                );
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn failed_encode_leaves_no_temp_file() {
        let path = temp_file("no-temp.heic");
        let _ = std::fs::remove_file(&path);
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        let tmp = path.with_file_name(format!("{stem}.heifwrite.tmp.heic"));
        let _ = std::fs::remove_file(&tmp);
        let err = encode(&path, &pixels(2, 2), 2, 2, 80).unwrap_err();
        assert!(format!("{err:#}").contains("libheif"), "{err:#}");
        assert!(!path.exists());
        assert!(!tmp.exists());
    }

    #[test]
    fn quality_boundaries_are_clamped() {
        let path = temp_file("quality.heic");
        let _ = std::fs::remove_file(&path);
        for quality in [0, 1, 99, 100, 255] {
            let result = encode(&path, &pixels(8, 8), 8, 8, quality);
            match result {
                Ok(()) => {}
                Err(err) => {
                    let text = format!("{err:#}");
                    assert!(
                        text.contains("no suitable library")
                            || text.contains("no HEVC encoder"),
                        "unexpected HEIF result at quality {quality}: {text}"
                    );
                }
            }
        }
        assert!(!path.exists());
    }
}
