//! HEIF/HEIC writing through the runtime libheif library.
//!
//! The encoder is optional: when the host has no `libheif` (or no HEVC
//! encoder) the job fails with a structured error instead of leaving a half
//! written file behind. `LUMEN_HEIF_LIB` points at a non-standard library.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Context, Result, anyhow, bail};

type Handle = *mut c_void;
type HeifContext = *mut c_void;
type HeifImage = *mut c_void;
type HeifEncoder = *mut c_void;

/// `struct heif_error`, returned by value from every libheif call.
#[repr(C)]
struct HeifError {
    code: c_int,
    subcode: c_int,
    message: *const c_char,
}

const HEIF_ERROR_OK: c_int = 0;
const HEIF_COMPRESSION_HEVC: c_int = 1;
const HEIF_COLORSPACE_RGB: c_int = 1;
const HEIF_CHROMA_INTERLEAVED_RGBA: c_int = 11;
const HEIF_CHANNEL_INTERLEAVED: c_int = 10;
const RTLD_NOW: c_int = 2;

const REQUIRED_EXPORTS: &[&str] = &[
    "heif_context_alloc",
    "heif_context_free",
    "heif_image_create",
    "heif_image_release",
    "heif_image_add_plane",
    "heif_image_get_plane",
    "heif_context_get_encoder_for_format",
    "heif_image_handle_release",
    "heif_encoder_release",
    "heif_encoder_set_lossy_quality",
    "heif_encoder_set_lossless",
    "heif_context_encode_image",
    "heif_context_write_to_file",
];

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
        "HEIF export needs libheif.dll with a HEVC encoder (libx265); \
         no suitable library was found. Set LUMEN_HEIF_LIB to the full path of libheif.dll. \
         Load error: {reason}"
    )
}

fn check(error: HeifError, what: &str) -> Result<()> {
    if error.code == HEIF_ERROR_OK {
        return Ok(());
    }
    let message = if error.message.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(error.message) }
            .to_string_lossy()
            .into_owned()
    };
    Err(anyhow!(
        "{what} failed: libheif error {}: {message}",
        error.code
    ))
}

/// Encodes RGBA pixels to `path` using the runtime libheif encoder.
///
/// `quality` is the recipe's 0..=100 scale; 100 requests lossless. `icc` is
/// embedded as the image's raw colour profile when present.
pub fn encode(
    path: &Path,
    rgba: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
) -> Result<()> {
    if width == 0 || height == 0 {
        bail!("HEIF export needs a non-empty image");
    }
    let expected = width as usize * height as usize * 4;
    if rgba.len() != expected {
        bail!(
            "HEIF export: buffer is {} bytes, expected {expected}",
            rgba.len()
        );
    }
    let quality = quality.clamp(1, 100);
    let (handle, library_path) = load()?;
    let result = unsafe {
        encode_with(
            handle, library_path, path, rgba, width, height, quality, icc, exif, xmp,
        )
    };
    free(handle);
    result
}

/// Encodes into memory so the normal export path can write the file.
pub fn encode_to_vec(
    rgba: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
) -> Result<Vec<u8>> {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "lumen-heif-{}-{}.heic",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let result = encode(&path, rgba, width, height, quality, icc, exif, xmp)
        .and_then(|()| std::fs::read(&path).map_err(Into::into));
    let _ = std::fs::remove_file(&path);
    result
}

fn load() -> Result<(Handle, PathBuf)> {
    let mut load_error = String::new();
    for candidate in library_candidates() {
        let handle = match unsafe { open(&candidate) } {
            Ok(handle) => handle,
            Err(err) => {
                load_error.push_str(&format!("{}: {err}; ", candidate.display()));
                continue;
            }
        };
        let mut missing = None;
        for name in REQUIRED_EXPORTS {
            if unsafe { symbol(handle, name) }.is_err() {
                missing = Some(*name);
                break;
            }
        }
        if let Some(name) = missing {
            free(handle);
            load_error.push_str(&format!(
                "{}: missing export {name:?}; ",
                candidate.display()
            ));
            continue;
        }
        return Ok((handle, candidate));
    }
    Err(missing_library_error(load_error.trim_end_matches("; ")))
}

/// Checks that the runtime library loads, exposes the required API and has a
/// HEVC encoder. This is the only signal that marks HEIF as supported.
pub fn probe() -> Result<()> {
    let (handle, library_path) = load()?;
    let result = unsafe { probe_with(handle) };
    free(handle);
    result.map_err(|err| err.context(format!("{} is not a usable libheif", library_path.display())))
}

pub fn available() -> bool {
    probe().is_ok()
}

unsafe fn probe_with(handle: Handle) -> Result<()> {
    unsafe fn function<T: Copy>(handle: Handle, name: &str) -> Result<T> {
        let address = unsafe { symbol(handle, name)? };
        Ok(unsafe { std::mem::transmute_copy(&address) })
    }
    let context_alloc: unsafe extern "C" fn() -> HeifContext =
        unsafe { function(handle, "heif_context_alloc")? };
    let context_free: unsafe extern "C" fn(HeifContext) =
        unsafe { function(handle, "heif_context_free")? };
    let get_encoder_for_format: unsafe extern "C" fn(HeifContext, c_int, *mut HeifEncoder) -> HeifError =
        unsafe { function(handle, "heif_context_get_encoder_for_format")? };
    let encoder_free: unsafe extern "C" fn(HeifEncoder) =
        unsafe { function(handle, "heif_encoder_release")? };
    let context = unsafe { (context_alloc)() };
    if context.is_null() {
        bail!("libheif could not allocate an encoder context");
    }
    let mut encoder: HeifEncoder = ptr::null_mut();
    let status = unsafe { (get_encoder_for_format)(context, HEIF_COMPRESSION_HEVC, &mut encoder) };
    if !encoder.is_null() {
        unsafe { (encoder_free)(encoder) };
    }
    unsafe { (context_free)(context) };
    check(status, "probing the HEVC encoder")?;
    if encoder.is_null() {
        bail!("libheif has no HEVC encoder; install libheif with libx265");
    }
    Ok(())
}

unsafe fn encode_with(
    handle: Handle,
    _library_path: PathBuf,
    output: &Path,
    rgba: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
) -> Result<()> {
    unsafe fn function<T: Copy>(handle: Handle, name: &str) -> Result<T> {
        let address = unsafe { symbol(handle, name)? };
        Ok(unsafe { std::mem::transmute_copy(&address) })
    }
    let context_alloc: unsafe extern "C" fn() -> HeifContext =
        unsafe { function(handle, "heif_context_alloc")? };
    let context_free: unsafe extern "C" fn(HeifContext) =
        unsafe { function(handle, "heif_context_free")? };
    let image_create: unsafe extern "C" fn(c_int, c_int, c_int, c_int, *mut HeifImage) -> HeifError =
        unsafe { function(handle, "heif_image_create")? };
    let image_free: unsafe extern "C" fn(HeifImage) =
        unsafe { function(handle, "heif_image_release")? };
    let image_add_plane: unsafe extern "C" fn(HeifImage, c_int, c_int, c_int, c_int) -> HeifError =
        unsafe { function(handle, "heif_image_add_plane")? };
    let image_get_plane: unsafe extern "C" fn(HeifImage, c_int, *mut c_int) -> *mut u8 =
        unsafe { function(handle, "heif_image_get_plane")? };
    let image_set_color_profile: Option<
        unsafe extern "C" fn(HeifImage, *const c_char, *const c_void, usize) -> HeifError,
    > = match icc {
        Some(_) => Some(unsafe { function(handle, "heif_image_set_raw_color_profile")? }),
        None => None,
    };
    let get_encoder_for_format: unsafe extern "C" fn(HeifContext, c_int, *mut HeifEncoder) -> HeifError =
        unsafe { function(handle, "heif_context_get_encoder_for_format")? };
    let encoder_free: unsafe extern "C" fn(HeifEncoder) =
        unsafe { function(handle, "heif_encoder_release")? };
    let encoder_set_quality: unsafe extern "C" fn(HeifEncoder, c_int) -> HeifError =
        unsafe { function(handle, "heif_encoder_set_lossy_quality")? };
    let encoder_set_lossless: unsafe extern "C" fn(HeifEncoder, c_int) -> HeifError =
        unsafe { function(handle, "heif_encoder_set_lossless")? };
    let context_encode_image: unsafe extern "C" fn(
        HeifContext,
        HeifImage,
        HeifEncoder,
        *const c_void,
        *mut *mut c_void,
    ) -> HeifError = unsafe { function(handle, "heif_context_encode_image")? };
    let image_handle_release: unsafe extern "C" fn(*mut c_void) =
        unsafe { function(handle, "heif_image_handle_release")? };
    let add_exif: Option<unsafe extern "C" fn(HeifContext, *mut c_void, *const c_void, c_int) -> HeifError> =
        match exif {
            Some(_) => Some(unsafe { function(handle, "heif_context_add_exif_metadata")? }),
            None => None,
        };
    let add_xmp: Option<unsafe extern "C" fn(HeifContext, *mut c_void, *const c_void, c_int) -> HeifError> =
        match xmp {
            Some(_) => Some(unsafe { function(handle, "heif_context_add_XMP_metadata")? }),
            None => None,
        };
    let context_write_to_file: unsafe extern "C" fn(HeifContext, *const c_char) -> HeifError =
        unsafe { function(handle, "heif_context_write_to_file")? };

    let context = unsafe { (context_alloc)() };
    if context.is_null() {
        bail!("libheif could not allocate an encoder context");
    }
    let mut image: HeifImage = ptr::null_mut();
    let mut encoder: HeifEncoder = ptr::null_mut();
    let mut image_handle: *mut c_void = ptr::null_mut();
    let mut tmp = output.to_path_buf();
    let result = (|| {
        let stem = output
            .file_stem()
            .and_then(|s| s.to_str())
            .context("output name is not UTF-8")?;
        let extension = output.extension().and_then(|s| s.to_str()).unwrap_or("heic");
        tmp.set_file_name(format!("{stem}.heifwrite.tmp.{extension}"));
        let tmp_name = CString::new(tmp.as_os_str().to_string_lossy().as_bytes())?;

        let status = unsafe {
            (image_create)(
                width as c_int,
                height as c_int,
                HEIF_COLORSPACE_RGB,
                HEIF_CHROMA_INTERLEAVED_RGBA,
                &mut image,
            )
        };
        check(status, "allocating the HEIF image")?;
        if image.is_null() {
            bail!("libheif returned no image");
        }
        let status = unsafe {
            (image_add_plane)(
                image,
                HEIF_CHANNEL_INTERLEAVED,
                width as c_int,
                height as c_int,
                8,
            )
        };
        check(status, "allocating the RGBA plane")?;
        let mut stride: c_int = 0;
        let plane = unsafe { (image_get_plane)(image, HEIF_CHANNEL_INTERLEAVED, &mut stride) };
        let row_bytes = width as usize * 4;
        if plane.is_null() || (stride as usize) < row_bytes {
            bail!("libheif returned no usable RGBA plane");
        }
        for row in 0..height as usize {
            let source = unsafe { rgba.as_ptr().add(row * row_bytes) };
            let target = unsafe { plane.add(row * stride as usize) };
            unsafe { std::ptr::copy_nonoverlapping(source, target, row_bytes) };
        }
        if let Some(icc) = icc {
            let set_profile =
                image_set_color_profile.expect("resolved when an ICC profile is present");
            let status = unsafe {
                set_profile(
                    image,
                    b"prof\0".as_ptr().cast(),
                    icc.as_ptr().cast(),
                    icc.len(),
                )
            };
            check(status, "embedding the ICC profile")?;
        }

        let status = unsafe {
            (get_encoder_for_format)(context, HEIF_COMPRESSION_HEVC, &mut encoder)
        };
        check(status, "finding the HEVC encoder")?;
        if encoder.is_null() {
            bail!("libheif has no HEVC encoder; install libheif with libx265");
        }
        let status = if quality >= 100 {
            unsafe { (encoder_set_lossless)(encoder, 1) }
        } else {
            unsafe { (encoder_set_quality)(encoder, quality as c_int) }
        };
        check(status, "setting HEIF quality")?;

        let status =
            unsafe { (context_encode_image)(context, image, encoder, ptr::null(), &mut image_handle) };
        check(status, "encoding HEIF")?;
        if let Some(exif) = exif {
            let add_exif = add_exif.expect("resolved when EXIF is present");
            let mut payload = Vec::with_capacity(exif.len() + 6);
            payload.extend_from_slice(b"Exif\0\0");
            payload.extend_from_slice(exif);
            let size = c_int::try_from(payload.len()).context("EXIF metadata is too large")?;
            let status =
                unsafe { add_exif(context, image_handle, payload.as_ptr().cast(), size) };
            check(status, "embedding the EXIF metadata")?;
        }
        if let Some(xmp) = xmp {
            let add_xmp = add_xmp.expect("resolved when XMP is present");
            let size = c_int::try_from(xmp.len()).context("XMP metadata is too large")?;
            let status = unsafe { add_xmp(context, image_handle, xmp.as_ptr().cast(), size) };
            check(status, "embedding the XMP metadata")?;
        }
        let status = unsafe { (context_write_to_file)(context, tmp_name.as_ptr()) };
        check(status, "writing HEIF")?;
        if !tmp.is_file() {
            bail!("libheif reported success but did not write {}", tmp.display());
        }
        std::fs::rename(&tmp, output)
            .with_context(|| format!("replacing {} with the encoded HEIF", output.display()))?;
        Ok(())
    })();

    if !encoder.is_null() {
        unsafe { (encoder_free)(encoder) };
    }
    if !image_handle.is_null() {
        unsafe { (image_handle_release)(image_handle) };
    }
    if !image.is_null() {
        unsafe { (image_free)(image) };
    }
    unsafe { (context_free)(context) };
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("lumen-heifwrite-{}-{name}", std::process::id()));
        path
    }

    fn pixels(width: u32, height: u32) -> Vec<u8> {
        (0..width as usize * height as usize * 4)
            .map(|i| (i * 7 % 255) as u8)
            .collect()
    }

    #[test]
    fn missing_library_reports_structured_error() {
        let previous = std::env::var_os("LUMEN_HEIF_LIB");
        unsafe { std::env::set_var("LUMEN_HEIF_LIB", "Z:/definitely/not/libheif.dll") };
        let path = temp_file("missing.heic");
        let err = encode(&path, &pixels(2, 2), 2, 2, 80, None, None, None).unwrap_err();
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
    fn encode_writes_a_file_when_the_encoder_is_available() {
        let path = temp_file("roundtrip.heic");
        let _ = std::fs::remove_file(&path);
        match encode(&path, &pixels(16, 12), 16, 12, 80, None, None, None) {
            Ok(()) => {
                let bytes = std::fs::metadata(&path).expect("encoded file").len();
                assert!(bytes > 0);
            }
            Err(err) => {
                let text = format!("{err:#}");
                assert!(
                    text.contains("no suitable library") || text.contains("no HEVC encoder"),
                    "unexpected HEIF result: {text}"
                );
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn failed_encode_leaves_no_temp_file() {
        let previous = std::env::var_os("LUMEN_HEIF_LIB");
        unsafe { std::env::set_var("LUMEN_HEIF_LIB", "Z:/definitely/not/libheif.dll") };
        let path = temp_file("no-temp.heic");
        let _ = std::fs::remove_file(&path);
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        let tmp = path.with_file_name(format!("{stem}.heifwrite.tmp.heic"));
        let _ = std::fs::remove_file(&tmp);
        let err = encode(&path, &pixels(2, 2), 2, 2, 80, None, None, None).unwrap_err();
        assert!(format!("{err:#}").contains("libheif"), "{err:#}");
        assert!(!path.exists());
        assert!(!tmp.exists());
        match previous {
            Some(value) => unsafe { std::env::set_var("LUMEN_HEIF_LIB", value) },
            None => unsafe { std::env::remove_var("LUMEN_HEIF_LIB") },
        }
    }

    #[test]
    fn rejects_a_wrong_buffer_size_before_writing() {
        let path = temp_file("short.heic");
        let err = encode(&path, &[0; 3], 2, 2, 80, None, None, None).unwrap_err();
        assert!(format!("{err:#}").contains("expected 16"), "{err:#}");
        assert!(!path.exists());
    }

    #[test]
    fn encode_embeds_an_icc_profile_when_available() {
        if !available() {
            return;
        }
        let mut profile = vec![0u8; 132];
        profile[0..4].copy_from_slice(&132u32.to_be_bytes());
        profile[36..40].copy_from_slice(b"acsp");
        let path = temp_file("icc.heic");
        let _ = std::fs::remove_file(&path);
        encode(&path, &pixels(16, 12), 16, 12, 80, Some(&profile), None, None).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(
            bytes.windows(4).any(|w| w == b"colr"),
            "no colr box in the HEIF output"
        );
        assert!(
            bytes.windows(4).any(|w| w == b"prof"),
            "no ICC profile box in the HEIF output"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn encode_embeds_exif_metadata_when_available() {
        if !available() {
            return;
        }
        let tiff = b"MM\0*\0\0\0\x08\0\0\0\0\0\0";
        let path = temp_file("exif.heic");
        let _ = std::fs::remove_file(&path);
        encode(&path, &pixels(16, 12), 16, 12, 80, None, Some(tiff), None).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let marker = b"Exif\0\0";
        assert!(
            bytes
                .windows(marker.len() + tiff.len())
                .any(|w| w.starts_with(marker) && &w[marker.len()..] == tiff),
            "no Exif payload in the HEIF output"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn encode_embeds_xmp_metadata_when_available() {
        if !available() {
            return;
        }
        let packet = br#"<?xpacket begin=""?><x:xmpmeta xmlns:x="adobe:ns:meta/"/><?xpacket end="w"?>"#;
        let path = temp_file("xmp.heic");
        let _ = std::fs::remove_file(&path);
        encode(&path, &pixels(16, 12), 16, 12, 80, None, None, Some(packet)).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(
            bytes.windows(packet.len()).any(|w| w == packet),
            "the XMP packet is missing from the HEIF output"
        );
        let _ = std::fs::remove_file(&path);
    }
}
