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
    get_image_width: GetImageSizeFn,
    get_image_height: GetImageSizeFn,
    decode: DecodeFn,
    release_image: ReleaseImageFn,
    get_plane: GetPlaneFn,
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
                module: Hmodule(module),
            })
        }
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
}


