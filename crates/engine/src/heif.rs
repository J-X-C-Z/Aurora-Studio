use std::ffi::{c_char, c_int, c_void, CString};
use std::path::Path;
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

struct Dll {
    module: Hmodule,
    context_alloc: ContextAllocFn,
    context_free: ContextFreeFn,
    read_from_file: ReadFileFn,
    primary_handle: PrimaryHandleFn,
    release_handle: ReleaseHandleFn,
    handle_width: HandleSizeFn,
    handle_height: HandleSizeFn,
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

impl Drop for Dll {
    fn drop(&mut self) {
        unsafe {
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn FreeLibrary(module: *mut c_void) -> c_int;
            }
            drop(FreeLibrary(self.module.0));
        }
    }
}

impl Dll {
    fn load() -> Result<Self> {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn LoadLibraryW(filename: *const u16) -> *mut c_void;
            fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        }
        unsafe {
            let name: Vec<u16> = "libheif.dll\0".encode_utf16().collect();
            let module = LoadLibraryW(name.as_ptr());
            if module.is_null() {
                bail!("libheif.dll 未找到或加载失败；请将 libheif.dll 及其依赖加入 PATH");
            }
            macro_rules! symbol {
                ($name:literal, $ty:ty) => {{
                    let address = GetProcAddress(
                        module,
                        concat!($name, "\0").as_ptr(),
                    );
                    if address.is_null() {
                        bail!("libheif.dll 缺少导出 {}", $name);
                    }
                    std::mem::transmute::<*mut c_void, $ty>(address)
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

    #[test]
    fn corrupt_file_does_not_panic() {
        let temp = std::env::temp_dir().join("lumen-heif-corrupt.heic");
        std::fs::write(&temp, b"not-heif").unwrap();
        assert!(decode(&temp).is_err());
    }

    #[test]
    fn missing_file_does_not_panic() {
        assert!(decode(Path::new(r"Z:\no-such-file.heic")).is_err());
    }

    #[cfg(feature = "external-fixtures")]
    include!("../local-fixtures/heif-tests.rs");
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
        let path = std::env::temp_dir().join(format!("lumen-heif-read-{}.heic", std::process::id()));
        let tiff = tiff_with_make("LumenCam");
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
        assert_eq!(exif.camera_make.as_deref(), Some("LumenCam"));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_heif_without_exif_reads_none() {
        if !crate::heif_write::available() {
            return;
        }
        let path = std::env::temp_dir().join(format!("lumen-heif-noexif-{}.heic", std::process::id()));
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
        let path = std::env::temp_dir().join(format!("lumen-heif-xmp-{}.heic", std::process::id()));
        let meta = crate::XmpMeta {
            rating: 4,
            label: "Red".into(),
            keywords: vec!["wedding".into(), "hero & co".into()],
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
        let path = std::env::temp_dir().join(format!("lumen-heif-noxmp-{}.heic", std::process::id()));
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
        let path = std::env::temp_dir().join(format!("lumen-heif-aux-{}.heic", std::process::id()));
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


