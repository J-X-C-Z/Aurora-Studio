//! HEIF/HEIC decoding through a runtime-loaded `libheif`.
//!
//! The loader is platform-neutral. Windows resolves `libheif.dll` with
//! `LoadLibraryW`/`GetProcAddress` and links `kernel32`; POSIX/macOS finds a
//! Homebrew or system dylib and loads it with `dlopen`/`dlsym`. The decode,
//! EXIF/XMP and auxiliary-image paths are identical on both.
//!
//! When no library can be loaded the reader returns an explicit error instead
//! of a silent failure, so callers can distinguish a missing `libheif` from a
//! malformed HEIC.
//!
//! The full auxiliary-image surface (components, thumbnails, depth) is kept
//! compiled on every platform; some of it is only exercised by tests or by
//! Windows-only call sites, so unused-item warnings are not useful here.
#![allow(dead_code)]

use std::ffi::{c_char, c_int, c_void, CString};
use std::path::Path;
#[cfg(not(windows))]
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use image::RgbaImage;

type ContextAllocFn = unsafe extern "C" fn() -> *mut c_void;
type ContextFreeFn = unsafe extern "C" fn(*mut c_void);
type HeifError = (c_int, c_int, *const c_char);
type ReadFileFn =
    unsafe extern "C" fn(*mut c_void, *const c_char, *const c_void) -> HeifError;
type PrimaryHandleFn =
    unsafe extern "C" fn(*const c_void, *mut *mut c_void) -> HeifError;
type ReleaseHandleFn = unsafe extern "C" fn(*mut c_void);
type GetSizeFn = unsafe extern "C" fn(*const c_void) -> c_int;
type HandleSizeFn = unsafe extern "C" fn(*const c_void) -> c_int;
type GetImageSizeFn = unsafe extern "C" fn(*const c_void, c_int) -> c_int;
type DecodeFn =
    unsafe extern "C" fn(
        *const c_void,
        *mut *mut c_void,
        c_int,
        c_int,
        *const c_void,
    ) -> HeifError;
type ReleaseImageFn = unsafe extern "C" fn(*mut c_void);
type GetPlaneFn = unsafe extern "C" fn(*const c_void, c_int, *mut c_int) -> *const u8;
type MetadataCountFn = unsafe extern "C" fn(*const c_void, *const c_char) -> c_int;
type MetadataIdsFn =
    unsafe extern "C" fn(*const c_void, *const c_char, *mut u32, c_int) -> c_int;
type MetadataSizeFn = unsafe extern "C" fn(*const c_void, u32) -> usize;
type MetadataFn = unsafe extern "C" fn(*const c_void, u32, *mut c_void) -> HeifError;
type MetadataContentTypeFn = unsafe extern "C" fn(*const c_void, u32) -> *const c_char;
type AuxCountFn = unsafe extern "C" fn(*const c_void) -> c_int;
type AuxIdsFn = unsafe extern "C" fn(*const c_void, *mut u32, c_int) -> c_int;
type ThumbnailHandleFn =
    unsafe extern "C" fn(*const c_void, u32, *mut *mut c_void) -> HeifError;
type DepthHandleFn =
    unsafe extern "C" fn(*const c_void, u32, *mut *mut c_void) -> HeifError;
type DepthInfoFn =
    unsafe extern "C" fn(*const c_void, u32, *mut *const DepthRepresentationInfo) -> c_int;
type DepthInfoFreeFn = unsafe extern "C" fn(*const DepthRepresentationInfo);
type BitsFn = unsafe extern "C" fn(*const c_void) -> c_int;

/// `heif_depth_representation_info` from libheif 1.23.7 `heif_aux_images.h`.
#[repr(C)]
struct DepthRepresentationInfo {
    version: u8,
    has_z_near: u8,
    has_z_far: u8,
    has_d_min: u8,
    has_d_max: u8,
    z_near: f64,
    z_far: f64,
    d_min: f64,
    d_max: f64,
    depth_representation_type: c_int,
    disparity_reference_view: u32,
    depth_nonlinear_representation_model_size: u32,
    depth_nonlinear_representation_model: *mut u8,
}
const HEIF_COLORSPACE_RGB: c_int = 1;
const HEIF_CHROMA_INTERLEAVED_RGBA: c_int = 11;
const PLANE_INTERLEAVED: c_int = 10;

#[repr(C)]
struct Hmodule(*mut c_void);

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(filename: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> c_int;
}

#[cfg(not(windows))]
unsafe extern "C" {
    fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(module: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
    fn dlclose(module: *mut c_void) -> c_int;
}

/// Resolve every symbol before returning, so a partially usable library is
/// rejected at load time instead of at the first decode call.
#[cfg(not(windows))]
const RTLD_NOW: c_int = 2;

/// Typed failure returned when no candidate `libheif` library can be opened.
///
/// It is a real error type rather than a bare string so callers can tell a
/// missing library apart from a malformed HEIC file.
#[cfg(not(windows))]
#[derive(Debug)]
struct LibraryNotFound {
    candidates: Vec<PathBuf>,
}

#[cfg(not(windows))]
impl std::fmt::Display for LibraryNotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "libheif library not found; searched: ")?;
        for (index, candidate) in self.candidates.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", candidate.display())?;
        }
        Ok(())
    }
}

#[cfg(not(windows))]
impl std::error::Error for LibraryNotFound {}

#[cfg(windows)]
unsafe fn open_library(path: &Path) -> Result<*mut c_void> {
    use std::os::windows::ffi::OsStrExt as _;

    let wide: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let module = unsafe { LoadLibraryW(wide.as_ptr()) };
    if module.is_null() {
        bail!("failed to load {}", path.display());
    }
    Ok(module)
}

#[cfg(not(windows))]
unsafe fn open_library(path: &Path) -> Result<*mut c_void> {
    let name = CString::new(path.as_os_str().to_string_lossy().as_bytes())?;
    let module = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
    if module.is_null() {
        bail!("failed to load {}: {}", path.display(), dl_error());
    }
    Ok(module)
}

/// Message for the most recent failed `dlopen`/`dlsym`, consumed with the call.
#[cfg(not(windows))]
fn dl_error() -> String {
    let error = unsafe { dlerror() };
    if error.is_null() {
        "unknown dynamic-loader error".to_string()
    } else {
        unsafe { std::ffi::CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(windows)]
unsafe fn library_symbol(module: *mut c_void, name: &str) -> Result<*mut c_void> {
    let symbol = CString::new(name)?;
    let address = unsafe { GetProcAddress(module, symbol.as_ptr().cast()) };
    if address.is_null() {
        bail!("libheif.dll 缺少导出 {name}");
    }
    Ok(address)
}

#[cfg(not(windows))]
unsafe fn library_symbol(module: *mut c_void, name: &str) -> Result<*mut c_void> {
    let symbol = CString::new(name)?;
    let address = unsafe { dlsym(module, symbol.as_ptr()) };
    if address.is_null() {
        bail!("missing export {name}: {}", dl_error());
    }
    Ok(address)
}

#[cfg(windows)]
unsafe fn close_library(module: *mut c_void) {
    unsafe { FreeLibrary(module) };
}

#[cfg(not(windows))]
unsafe fn close_library(module: *mut c_void) {
    unsafe { dlclose(module) };
}

struct Dll {
    module: Hmodule,
    context_alloc: ContextAllocFn,
    context_free: ContextFreeFn,
    read_from_file: ReadFileFn,
    primary_handle: PrimaryHandleFn,
    release_handle: ReleaseHandleFn,
    handle_width: HandleSizeFn,
    handle_height: HandleSizeFn,
    bits_per_pixel: BitsFn,
    get_image_width: GetImageSizeFn,
    get_image_height: GetImageSizeFn,
    decode: DecodeFn,
    release_image: ReleaseImageFn,
    get_plane: GetPlaneFn,
    metadata_count: MetadataCountFn,
    metadata_ids: MetadataIdsFn,
    metadata_size: MetadataSizeFn,
    metadata_get: MetadataFn,
    metadata_content_type: MetadataContentTypeFn,
    thumbnail_count: AuxCountFn,
    thumbnail_ids: AuxIdsFn,
    thumbnail_handle: ThumbnailHandleFn,
    depth_count: AuxCountFn,
    depth_ids: AuxIdsFn,
    depth_handle: DepthHandleFn,
    depth_info: DepthInfoFn,
    depth_info_free: DepthInfoFreeFn,
}

unsafe impl Send for Dll {}
unsafe impl Sync for Dll {}

/// Name used in diagnostics; matches the loader compiled into this build.
#[cfg(windows)]
const LIB_HEIF: &str = "libheif.dll";
#[cfg(not(windows))]
const LIB_HEIF: &str = "libheif";

/// Candidate libraries for the non-Windows loader.
///
/// Order: `AURORA_HEIF_LIB`, then executable-relative locations for a bundled
/// app (`Contents/Frameworks` and the binary's own directory, i.e. the
/// `@loader_path` equivalents), then bare names resolved by the system loader.
/// Nothing is anchored to Homebrew, a user directory or a drive letter.
#[cfg(not(windows))]
fn library_candidates() -> Vec<PathBuf> {
    if let Some(path) = std::env::var_os("AURORA_HEIF_LIB") {
        return vec![PathBuf::from(path)];
    }
    #[cfg(target_os = "macos")]
    let names: &[&str] = &["libheif.dylib", "libheif.1.dylib"];
    #[cfg(all(not(windows), not(target_os = "macos")))]
    let names: &[&str] = &["libheif.so.1", "libheif.so"];
    let mut candidates = Vec::new();
    #[cfg(target_os = "macos")]
    {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(bin_dir) = exe.parent() {
                let frameworks = bin_dir.join("..").join("Frameworks");
                for name in names {
                    candidates.push(frameworks.join(name));
                }
                for name in names {
                    candidates.push(bin_dir.join(name));
                }
            }
        }
    }
    for name in names {
        candidates.push(PathBuf::from(name));
    }
    candidates
}

#[cfg(windows)]
unsafe fn resolve(module: *mut c_void, name: &str) -> *mut c_void {
    let mut bytes = Vec::with_capacity(name.len() + 1);
    bytes.extend_from_slice(name.as_bytes());
    bytes.push(0);
    unsafe { GetProcAddress(module, bytes.as_ptr()) }
}

#[cfg(not(windows))]
unsafe fn resolve(module: *mut c_void, name: &str) -> *mut c_void {
    let Ok(name) = CString::new(name) else {
        return std::ptr::null_mut();
    };
    unsafe { dlsym(module, name.as_ptr()) }
}

#[cfg(not(windows))]
unsafe fn open(path: &Path) -> Option<*mut c_void> {
    let name = CString::new(path.as_os_str().to_string_lossy().as_bytes()).ok()?;
    let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
    (!handle.is_null()).then_some(handle)
}

#[cfg(not(windows))]
unsafe fn open_first_available() -> Result<*mut c_void> {
    let mut tried = String::new();
    for candidate in library_candidates() {
        if let Some(handle) = unsafe { open(&candidate) } {
            return Ok(handle);
        }
        tried.push_str(&format!("{}; ", candidate.display()));
    }
    bail!(
        "libheif 未找到或加载失败；已尝试 {}。请安装 libheif 或将 AURORA_HEIF_LIB 指向库文件",
        tried.trim_end_matches("; ")
    )
}

impl Drop for Dll {
    fn drop(&mut self) {
        unsafe { close_library(self.module.0) };
    }
}

impl Dll {
    /// Windows keeps its existing `LoadLibraryW("libheif.dll")` search: the
    /// DLL directory/PATH order is what callers already depend on.
    #[cfg(windows)]
    fn load() -> Result<Self> {
        let module = match unsafe { open_library(Path::new("libheif.dll")) } {
            Ok(module) => module,
            Err(_) => {
                bail!("libheif.dll 未找到或加载失败；请将 libheif.dll 及其依赖加入 PATH")
            }
        };
        match unsafe { Self::from_handle(module) } {
            Ok(dll) => Ok(dll),
            Err(error) => {
                unsafe { close_library(module) };
                Err(error)
            }
        }
    }

    /// `AURORA_HEIF_LIB` is authoritative when set, matching the writer: a
    /// stale override must fail loudly instead of silently selecting another
    /// library. Each candidate must exist on disk before `dlopen`, and the
    /// typed error lists everything that was searched.
    #[cfg(not(windows))]
    fn load() -> Result<Self> {
        let override_path = std::env::var_os("AURORA_HEIF_LIB").map(PathBuf::from);
        let candidates = crate::heif_write::candidate_paths(override_path.as_deref());
        let mut refused = Vec::new();
        for candidate in &candidates {
            if !candidate.is_file() {
                refused.push(format!("{}: not found", candidate.display()));
                continue;
            }
            let module = match unsafe { open_library(candidate) } {
                Ok(module) => module,
                Err(error) => {
                    refused.push(error.to_string());
                    continue;
                }
            };
            match unsafe { Self::from_handle(module) } {
                Ok(dll) => return Ok(dll),
                Err(error) => {
                    unsafe { close_library(module) };
                    refused.push(format!("{}: {error}", candidate.display()));
                }
            }
        }
        log::debug!("libheif candidates exhausted: {}", refused.join("; "));
        Err(anyhow::Error::new(LibraryNotFound { candidates }))
    }

    unsafe fn from_handle(module: *mut c_void) -> Result<Self> {
        macro_rules! symbol {
            ($name:literal, $ty:ty) => {{
                let address = unsafe { library_symbol(module, $name)? };
                unsafe { std::mem::transmute::<*mut c_void, $ty>(address) }
            }};
        }
        Ok(Self {
            context_alloc: symbol!("heif_context_alloc", ContextAllocFn),
            context_free: symbol!("heif_context_free", ContextFreeFn),
            read_from_file: symbol!("heif_context_read_from_file", ReadFileFn),
            primary_handle: symbol!(
                "heif_context_get_primary_image_handle",
                PrimaryHandleFn
            ),
            release_handle: symbol!("heif_image_handle_release", ReleaseHandleFn),
            handle_width: symbol!(
                "heif_image_handle_get_width",
                HandleSizeFn
            ),
            handle_height: symbol!(
                "heif_image_handle_get_height",
                HandleSizeFn
            ),
            bits_per_pixel: symbol!(
                "heif_image_handle_get_luma_bits_per_pixel",
                BitsFn
            ),
            get_image_width: symbol!(
                "heif_image_get_width",
                GetImageSizeFn
            ),
            get_image_height: symbol!(
                "heif_image_get_height",
                GetImageSizeFn
            ),
            decode: symbol!("heif_decode_image", DecodeFn),
            release_image: symbol!("heif_image_release", ReleaseImageFn),
            get_plane: symbol!("heif_image_get_plane_readonly", GetPlaneFn),
            metadata_count: symbol!(
                "heif_image_handle_get_number_of_metadata_blocks",
                MetadataCountFn
            ),
            metadata_ids: symbol!(
                "heif_image_handle_get_list_of_metadata_block_IDs",
                MetadataIdsFn
            ),
            metadata_size: symbol!(
                "heif_image_handle_get_metadata_size",
                MetadataSizeFn
            ),
            metadata_get: symbol!("heif_image_handle_get_metadata", MetadataFn),
            metadata_content_type: symbol!(
                "heif_image_handle_get_metadata_content_type",
                MetadataContentTypeFn
            ),
            thumbnail_count: symbol!(
                "heif_image_handle_get_number_of_thumbnails",
                AuxCountFn
            ),
            thumbnail_ids: symbol!(
                "heif_image_handle_get_list_of_thumbnail_IDs",
                AuxIdsFn
            ),
            thumbnail_handle: symbol!(
                "heif_image_handle_get_thumbnail",
                ThumbnailHandleFn
            ),
            depth_count: symbol!(
                "heif_image_handle_get_number_of_depth_images",
                AuxCountFn
            ),
            depth_ids: symbol!(
                "heif_image_handle_get_list_of_depth_image_IDs",
                AuxIdsFn
            ),
            depth_handle: symbol!(
                "heif_image_handle_get_depth_image_handle",
                DepthHandleFn
            ),
            depth_info: symbol!(
                "heif_image_handle_get_depth_image_representation_info",
                DepthInfoFn
            ),
            depth_info_free: symbol!(
                "heif_depth_representation_info_free",
                DepthInfoFreeFn
            ),
            module: Hmodule(module),
        })
    }
}

/// Pixel size from the HEIF header, without decoding the image.
pub fn dimensions(path: &Path) -> Result<(u32, u32)> {
    let dll = Dll::load()?;
    let file_name = CString::new(path.to_str().context("HEIF 路径必须能转成 UTF-8")?)?;
    unsafe {
        let context = (dll.context_alloc)();
        if context.is_null() {
            bail!("libheif 分配上下文失败");
        }
        let result = (|| {
            let error =
                (dll.read_from_file)(context, file_name.as_ptr(), std::ptr::null());
            if error.0 != 0 {
                bail!("libheif 读取失败：{}", error_string(error));
            }
            let mut handle = std::ptr::null_mut();
            let error = (dll.primary_handle)(context, &mut handle);
            if error.0 != 0 || handle.is_null() {
                bail!("libheif 获取 primary image 失败：{}", error_string(error));
            }
            let _guard = HandleGuard { dll: &dll, handle };
            let width = (dll.handle_width)(handle);
            let height = (dll.handle_height)(handle);
            if width <= 0 || height <= 0 {
                bail!("libheif 返回了空的图像尺寸");
            }
            Ok((width as u32, height as u32))
        })();
        (dll.context_free)(context);
        result
    }
}

/// Bits per sample of the primary image, or `None` when unreadable.
pub fn bit_depth(path: &Path) -> Option<u8> {
    let dll = Dll::load().ok()?;
    let file_name = CString::new(path.to_str()?).ok()?;
    unsafe {
        let context = (dll.context_alloc)();
        if context.is_null() {
            return None;
        }
        let result = (|| {
            if (dll.read_from_file)(context, file_name.as_ptr(), std::ptr::null()).0 != 0 {
                return None;
            }
            let mut handle = std::ptr::null_mut();
            if (dll.primary_handle)(context, &mut handle).0 != 0 || handle.is_null() {
                return None;
            }
            let _guard = HandleGuard { dll: &dll, handle };
            u8::try_from((dll.bits_per_pixel)(handle)).ok().filter(|&bits| bits > 0)
        })();
        (dll.context_free)(context);
        result
    }
}

/// Raw TIFF block of the first EXIF item, without the HEIF offset prefix.
///
/// Returns `None` for files without EXIF or when libheif cannot read it.
pub fn read_exif(path: &Path) -> Option<Vec<u8>> {
    let data = read_metadata_item(path, Some(b"Exif\0"), None).ok().flatten()?;
    // The item starts with a 4-byte offset; its convention varies, so find the
    // TIFF header instead of trusting it.
    let at = data
        .windows(4)
        .position(|w| w == b"II*\0" || w == b"MM\0*");
    at.map(|at| data[at..].to_vec())
}

/// Text of the first XMP metadata item, or `None` when absent/unreadable.
pub fn read_xmp(path: &Path) -> Option<String> {
    let data = read_metadata_item(path, None, Some(b"application/rdf+xml"))
        .ok()
        .flatten()?;
    String::from_utf8(data).ok()
}

fn read_metadata_item(
    path: &Path,
    filter: Option<&[u8]>,
    content_type: Option<&[u8]>,
) -> Result<Option<Vec<u8>>> {
    let dll = Dll::load()?;
    let file_name = CString::new(path.to_str().context("HEIF 路径必须能转成 UTF-8")?)?;
    unsafe {
        let context = (dll.context_alloc)();
        if context.is_null() {
            bail!("libheif 分配上下文失败");
        }
        let result = (|| {
            let error =
                (dll.read_from_file)(context, file_name.as_ptr(), std::ptr::null());
            if error.0 != 0 {
                bail!("libheif 读取失败：{}", error_string(error));
            }
            let mut handle = std::ptr::null_mut();
            let error = (dll.primary_handle)(context, &mut handle);
            if error.0 != 0 || handle.is_null() {
                bail!("libheif 获取 primary image 失败：{}", error_string(error));
            }
            let _guard = HandleGuard { dll: &dll, handle };
            let filter_ptr = filter.map_or(std::ptr::null(), |f| f.as_ptr().cast());
            let count = (dll.metadata_count)(handle, filter_ptr);
            if count <= 0 {
                return Ok(None);
            }
            let mut ids = vec![0u32; count as usize];
            let listed = (dll.metadata_ids)(handle, filter_ptr, ids.as_mut_ptr(), count);
            if listed <= 0 {
                return Ok(None);
            }
            let id = match content_type {
                Some(wanted) => ids.iter().copied().find(|&id| {
                    let found = (dll.metadata_content_type)(handle, id);
                    !found.is_null()
                        && unsafe { std::ffi::CStr::from_ptr(found) }.to_bytes() == wanted
                }),
                None => ids.first().copied(),
            };
            let Some(id) = id else { return Ok(None) };
            let size = (dll.metadata_size)(handle, id);
            // A metadata item larger than this is not plausible; refuse rather
            // than allocate it.
            if size == 0 || size > 64 * 1024 * 1024 {
                return Ok(None);
            }
            let mut data = vec![0u8; size];
            let error = (dll.metadata_get)(handle, id, data.as_mut_ptr().cast());
            if error.0 != 0 {
                return Ok(None);
            }
            Ok(Some(data))
        })();
        (dll.context_free)(context);
        result
    }
}

fn error_string(error: HeifError) -> String {
    unsafe {
        if error.2.is_null() {
            "unknown libheif error".to_string()
        } else {
            std::ffi::CStr::from_ptr(error.2)
                .to_string_lossy()
                .into_owned()
        }
    }
}

pub fn decode(path: &Path) -> Result<RgbaImage> {
    let started = Instant::now();
    let dll = Dll::load()?;
    let file_name = CString::new(path.to_str().context("HEIF 路径必须能转成 UTF-8")?)?;
    unsafe {
        let context = (dll.context_alloc)();
        if context.is_null() {
            bail!("libheif 分配上下文失败");
        }
        let result = decode_with_context(&dll, context, &file_name);
        (dll.context_free)(context);
        if result.is_ok() {
            log::debug!(
                "HEIF decoded {} in {:?}",
                path.display(),
                started.elapsed()
            );
        }
        result
    }
}

unsafe fn decode_with_context(
    dll: &Dll,
    context: *mut c_void,
    file_name: &CString,
) -> Result<RgbaImage> {
    let error = unsafe { (dll.read_from_file)(context, file_name.as_ptr(), std::ptr::null()) };
    if error.0 != 0 {
        bail!(
            "libheif 读取失败：{}",
            error_string(error)
        );
    }
    let mut handle = std::ptr::null_mut();
    let error = unsafe { (dll.primary_handle)(context, &mut handle) };
    if error.0 != 0 || handle.is_null() {
        bail!(
            "libheif 获取 primary image 失败：{}",
            error_string(error)
        );
    }
    let _handle_guard = HandleGuard { dll, handle };
    unsafe { decode_handle(dll, handle) }
}

unsafe fn decode_handle(dll: &Dll, handle: *mut c_void) -> Result<RgbaImage> {
    let mut image = std::ptr::null_mut();
    let error = unsafe {
        (dll.decode)(
            handle,
            &mut image,
            HEIF_COLORSPACE_RGB,
            HEIF_CHROMA_INTERLEAVED_RGBA,
            std::ptr::null(),
        )
    };
    if error.0 != 0 || image.is_null() {
        bail!(
            "libheif 解码失败：{}",
            error_string(error)
        );
    }
    let _image_guard = ImageGuard { dll, image };
    let width = unsafe { (dll.get_image_width)(image, PLANE_INTERLEAVED) };
    let height = unsafe { (dll.get_image_height)(image, PLANE_INTERLEAVED) };
    let mut stride = 0;
    let plane = unsafe { (dll.get_plane)(image, PLANE_INTERLEAVED, &mut stride) };
    if plane.is_null() || stride < width * 4 {
        bail!("libheif 没有可用的 interleaved RGB plane");
    }
    let mut output = RgbaImage::new(width as u32, height as u32);
    for y in 0..height as usize {
        let source = unsafe { plane.add(y * stride as usize) };
        let destination = unsafe {
            output
                .as_mut_ptr()
                .cast::<u8>()
                .add(y * width as usize * 4)
        };
        unsafe { std::ptr::copy_nonoverlapping(source, destination, width as usize * 4) };
    }
    Ok(output)
}

/// Auxiliary images attached to the primary image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Components {
    pub thumbnails: u32,
    pub depth_images: u32,
}

/// First depth image of the primary image, decoded as RGBA.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DepthImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// How the depth samples map to distance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthRepresentation {
    UniformInverseZ,
    UniformDisparity,
    UniformZ,
    NonuniformDisparity,
    Unknown(i32),
}

impl DepthRepresentation {
    /// True when samples are linear in Z (`uniform_Z`).
    pub fn is_linear(self) -> bool {
        matches!(self, Self::UniformZ)
    }
}

/// Depth range metadata; fields are present only when libheif reports them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthInfo {
    pub representation: DepthRepresentation,
    pub z_near: Option<f64>,
    pub z_far: Option<f64>,
    pub d_min: Option<f64>,
    pub d_max: Option<f64>,
}

/// Reads the thumbnail/depth counts without decoding anything.
pub fn components(path: &Path) -> Result<Components> {
    with_primary(path, |dll, handle| unsafe {
        Ok(Components {
            thumbnails: (dll.thumbnail_count)(handle).max(0) as u32,
            depth_images: (dll.depth_count)(handle).max(0) as u32,
        })
    })
}

/// Decodes the first thumbnail, or `None` when the file has none.
pub fn thumbnail(path: &Path) -> Result<Option<RgbaImage>> {
    with_primary(path, |dll, handle| unsafe {
        let count = (dll.thumbnail_count)(handle);
        if count <= 0 {
            return Ok(None);
        }
        let mut ids = vec![0u32; (count as usize).min(16)];
        let listed = (dll.thumbnail_ids)(handle, ids.as_mut_ptr(), ids.len() as c_int);
        if listed <= 0 {
            return Ok(None);
        }
        let mut thumb = std::ptr::null_mut();
        let error = (dll.thumbnail_handle)(handle, ids[0], &mut thumb);
        if error.0 != 0 || thumb.is_null() {
            bail!("libheif 获取缩略图失败：{}", error_string(error));
        }
        let _guard = HandleGuard { dll, handle: thumb };
        decode_handle(dll, thumb).map(Some)
    })
}

/// Decodes the first depth image, or `None` when the file has none.
///
/// A present-but-undecodable depth item is an error, never a fabricated image.
pub fn depth(path: &Path) -> Result<Option<DepthImage>> {
    with_primary(path, |dll, handle| unsafe {
        let count = (dll.depth_count)(handle);
        if count <= 0 {
            return Ok(None);
        }
        let mut ids = vec![0u32; (count as usize).min(16)];
        let listed = (dll.depth_ids)(handle, ids.as_mut_ptr(), ids.len() as c_int);
        if listed <= 0 {
            return Ok(None);
        }
        let mut depth = std::ptr::null_mut();
        let error = (dll.depth_handle)(handle, ids[0], &mut depth);
        if error.0 != 0 || depth.is_null() {
            bail!("libheif 获取 depth image 失败：{}", error_string(error));
        }
        let _guard = HandleGuard { dll, handle: depth };
        let image = decode_handle(dll, depth)?;
        Ok(Some(DepthImage {
            width: image.width(),
            height: image.height(),
            rgba: image.into_raw(),
        }))
    })
}

/// Reads the first depth image's representation info, or `None` when absent.
pub fn depth_info(path: &Path) -> Option<DepthInfo> {
    with_primary(path, |dll, handle| unsafe {
        let count = (dll.depth_count)(handle);
        if count <= 0 {
            return Ok(None);
        }
        let mut ids = vec![0u32; (count as usize).min(16)];
        let listed = (dll.depth_ids)(handle, ids.as_mut_ptr(), ids.len() as c_int);
        if listed <= 0 {
            return Ok(None);
        }
        let mut info: *const DepthRepresentationInfo = std::ptr::null();
        if (dll.depth_info)(handle, ids[0], &mut info) == 0 || info.is_null() {
            return Ok(None);
        }
        let raw = &*info;
        let result = (raw.version >= 1).then(|| DepthInfo {
            representation: match raw.depth_representation_type {
                0 => DepthRepresentation::UniformInverseZ,
                1 => DepthRepresentation::UniformDisparity,
                2 => DepthRepresentation::UniformZ,
                3 => DepthRepresentation::NonuniformDisparity,
                other => DepthRepresentation::Unknown(other),
            },
            z_near: (raw.has_z_near != 0).then_some(raw.z_near),
            z_far: (raw.has_z_far != 0).then_some(raw.z_far),
            d_min: (raw.has_d_min != 0).then_some(raw.d_min),
            d_max: (raw.has_d_max != 0).then_some(raw.d_max),
        });
        (dll.depth_info_free)(info);
        Ok(result)
    })
    .ok()
    .flatten()
}

fn with_primary<T>(
    path: &Path,
    f: impl FnOnce(&Dll, *mut c_void) -> Result<T>,
) -> Result<T> {
    let dll = Dll::load()?;
    let file_name = CString::new(path.to_str().context("HEIF 路径必须能转成 UTF-8")?)?;
    unsafe {
        let context = (dll.context_alloc)();
        if context.is_null() {
            bail!("libheif 分配上下文失败");
        }
        let result = (|| {
            let error =
                (dll.read_from_file)(context, file_name.as_ptr(), std::ptr::null());
            if error.0 != 0 {
                bail!("libheif 读取失败：{}", error_string(error));
            }
            let mut handle = std::ptr::null_mut();
            let error = (dll.primary_handle)(context, &mut handle);
            if error.0 != 0 || handle.is_null() {
                bail!("libheif 获取 primary image 失败：{}", error_string(error));
            }
            let _guard = HandleGuard { dll: &dll, handle };
            f(&dll, handle)
        })();
        (dll.context_free)(context);
        result
    }
}

struct HandleGuard<'a> {
    dll: &'a Dll,
    handle: *mut c_void,
}

impl Drop for HandleGuard<'_> {
    fn drop(&mut self) {
        unsafe { (self.dll.release_handle)(self.handle) };
    }
}

struct ImageGuard<'a> {
    dll: &'a Dll,
    image: *mut c_void,
}

impl Drop for ImageGuard<'_> {
    fn drop(&mut self) {
        unsafe { (self.dll.release_image)(self.image) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn missing_dll_returns_clear_error() {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn SetDllDirectoryW(path: *const u16) -> c_int;
        }
        let original_path = std::env::var_os("PATH");
        unsafe { std::env::set_var("PATH", "Z:\\no-such-path") };
        unsafe {
            let empty: Vec<u16> = "\0".encode_utf16().collect();
            SetDllDirectoryW(empty.as_ptr());
        }
        let result = decode(Path::new("missing.heic"));
        unsafe {
            let system32: Vec<u16> = "C:\\Windows\\System32\0".encode_utf16().collect();
            SetDllDirectoryW(system32.as_ptr());
        }
        if let Some(path) = original_path {
            unsafe { std::env::set_var("PATH", path) };
        }
        if let Ok(image) = result {
            assert_eq!((image.width(), image.height()), (33, 17));
            return;
        }
        let error = result.unwrap_err().to_string();
        assert!(error.contains("libheif.dll") || error.contains("libheif"));
    }

    #[cfg(not(windows))]
    #[test]
    fn missing_override_reports_typed_error_with_candidates() {
        let previous = std::env::var_os("AURORA_HEIF_LIB");
        let missing = std::env::temp_dir().join(format!(
            "aurora-missing-libheif-{}.dylib",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&missing);
        unsafe { std::env::set_var("AURORA_HEIF_LIB", &missing) };
        let result = dimensions(Path::new("missing.heic"));
        match previous {
            Some(value) => unsafe { std::env::set_var("AURORA_HEIF_LIB", value) },
            None => unsafe { std::env::remove_var("AURORA_HEIF_LIB") },
        }
        let error = result.expect_err("a missing override must be reported, not panic");
        let typed = error
            .downcast_ref::<LibraryNotFound>()
            .expect("the missing library error must keep its type");
        assert_eq!(typed.candidates, vec![missing.clone()]);
        let text = error.to_string();
        assert!(text.contains("libheif library not found"), "{text}");
        assert!(text.contains(missing.to_str().unwrap()), "{text}");
    }

    #[test]
    fn corrupt_file_does_not_panic() {
        let temp = std::env::temp_dir().join("portrait-heif-corrupt.heic");
        std::fs::write(&temp, b"not-heif").unwrap();
        assert!(decode(&temp).is_err());
    }

    #[test]
    fn missing_file_does_not_panic() {
        assert!(decode(Path::new(r"Z:\no-such-file.heic")).is_err());
    }

    #[test]
    #[ignore = "needs local HEIC sample; set AURORA_TEST_HEIC"]
    fn real_sample_decodes_at_expected_size() {
        let path = std::env::var_os("AURORA_TEST_HEIC")
            .map(std::path::PathBuf::from)
            .expect("set AURORA_TEST_HEIC to a local HEIC file");
        let image = decode(&path).unwrap();
        assert!(image.width() > 0 && image.height() > 0);
    }

    fn tiff_with_make(make: &str) -> Vec<u8> {
        let mut text = make.as_bytes().to_vec();
        text.push(0);
        let mut tiff = Vec::new();
        tiff.extend_from_slice(b"II*\0");
        tiff.extend_from_slice(&8u32.to_le_bytes());
        tiff.extend_from_slice(&1u16.to_le_bytes());
        tiff.extend_from_slice(&0x010Fu16.to_le_bytes());
        tiff.extend_from_slice(&2u16.to_le_bytes());
        tiff.extend_from_slice(&(text.len() as u32).to_le_bytes());
        tiff.extend_from_slice(&26u32.to_le_bytes());
        tiff.extend_from_slice(&0u32.to_le_bytes());
        tiff.extend_from_slice(&text);
        tiff
    }

    #[test]
    fn reads_exif_from_a_written_heif() {
        if !crate::heif_write::available() {
            return;
        }
        let path = std::env::temp_dir().join(format!("portrait-heif-read-{}.heic", std::process::id()));
        let tiff = tiff_with_make("PortraitCam");
        crate::heif_write::encode(
            &path,
            &vec![90u8; 16 * 12 * 4],
            16,
            12,
            80,
            None,
            Some(&tiff),
            None,
        )
        .unwrap();
        let raw = read_exif(&path).expect("EXIF item");
        assert_eq!(raw, tiff);
        let exif = crate::exif::parse_exif(&raw);
        assert_eq!(exif.camera_make.as_deref(), Some("PortraitCam"));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_heif_without_exif_reads_none() {
        if !crate::heif_write::available() {
            return;
        }
        let path = std::env::temp_dir().join(format!("portrait-heif-noexif-{}.heic", std::process::id()));
        crate::heif_write::encode(&path, &vec![90u8; 16 * 12 * 4], 16, 12, 80, None, None, None)
            .unwrap();
        assert!(read_exif(&path).is_none());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn reads_xmp_from_a_written_heif() {
        if !crate::heif_write::available() {
            return;
        }
        let path = std::env::temp_dir().join(format!("portrait-heif-xmp-{}.heic", std::process::id()));
        let meta = crate::XmpMeta {
            rating: 4,
            label: "Red".into(),
            keywords: vec!["wedding".into(), "hero & co".into()],
            ..Default::default()
        };
        let packet = meta.packet();
        crate::heif_write::encode(
            &path,
            &vec![90u8; 16 * 12 * 4],
            16,
            12,
            80,
            None,
            None,
            Some(packet.as_bytes()),
        )
        .unwrap();
        let text = read_xmp(&path).expect("XMP item");
        assert_eq!(text, packet);
        assert_eq!(crate::XmpMeta::parse(&text).unwrap(), meta);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_heif_without_xmp_reads_none() {
        if !crate::heif_write::available() {
            return;
        }
        let path = std::env::temp_dir().join(format!("portrait-heif-noxmp-{}.heic", std::process::id()));
        crate::heif_write::encode(&path, &vec![90u8; 16 * 12 * 4], 16, 12, 80, None, None, None)
            .unwrap();
        assert!(read_xmp(&path).is_none());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn auxiliary_images_are_probed_without_decoding() {
        if !crate::heif_write::available() {
            return;
        }
        let path = std::env::temp_dir().join(format!("portrait-heif-aux-{}.heic", std::process::id()));
        crate::heif_write::encode(&path, &vec![90u8; 16 * 12 * 4], 16, 12, 80, None, None, None)
            .unwrap();
        let components = components(&path).expect("handle probe");
        assert_eq!(components, Components::default());
        assert!(thumbnail(&path).unwrap().is_none());
        assert!(depth(&path).unwrap().is_none());
        assert!(depth_info(&path).is_none());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    #[ignore = "needs local HEIC sample and libheif.dll; set AURORA_TEST_HEIC"]
    fn real_sample_auxiliary_counts_are_readable() {
        let path = std::env::var_os("AURORA_TEST_HEIC")
            .map(std::path::PathBuf::from)
            .expect("set AURORA_TEST_HEIC to a local HEIC file");
        let components = components(&path).expect("handle probe");
        assert_eq!(components.depth_images, 0);
        let _ = thumbnail(&path).expect("thumbnail probe");
        assert!(depth(&path).expect("depth probe").is_none());
        assert!(depth_info(&path).is_none());
    }

    #[test]
    fn depth_representation_maps_to_linearity() {
        assert!(DepthRepresentation::UniformZ.is_linear());
        for other in [
            DepthRepresentation::UniformInverseZ,
            DepthRepresentation::UniformDisparity,
            DepthRepresentation::NonuniformDisparity,
            DepthRepresentation::Unknown(9),
        ] {
            assert!(!other.is_linear(), "{other:?}");
        }
    }
}


