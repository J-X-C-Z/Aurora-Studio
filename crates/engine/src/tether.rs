//! Tethered capture through the system camera stack.
//!
//! * Linux goes through `libgphoto2`, resolved at run time because distros ship
//!   the runtime without the headers. Only the exported functions used here are
//!   bound; `Camera`, `CameraFile` and `GPContext` stay opaque pointers.
//! * Windows goes through WIA (Windows Image Acquisition) COM. Enumeration is
//!   real; capture only uses `WIA_CMD_TAKE_PICTURE`, which most mirrorless
//!   bodies do not implement from their WIA driver, so it reports
//!   `Unsupported` instead of pretending a vendor SDK is present. Preview uses
//!   the WIA preview component when the driver provides one.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::engine::RgbaImage;

/// How libgphoto2 answered the last probe.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum TetherStatus {
    /// The runtime library could not be loaded, or this is not Linux.
    #[default]
    NoLibrary,
    /// libgphoto2 loaded but no camera answered `gp_camera_init`.
    NoCamera,
    /// A camera is open and ready to capture.
    Available { lib_version: String },
}

/// One captured frame: the camera JPEG bytes and its decoded dimensions.
#[derive(Clone, Debug)]
pub struct CapturedFrame {
    pub jpeg: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Default budget for a still capture; the camera may need seconds to expose.
pub const CAPTURE_TIMEOUT: Duration = Duration::from_secs(30);
/// Probe and live-view calls give up sooner so the interface never stalls.
const QUICK_TIMEOUT: Duration = Duration::from_secs(3);

/// Reports whether a camera can actually be opened. Opens and keeps a session,
/// so a following capture or preview does not pay for `gp_camera_init` again.
pub fn probe() -> TetherStatus {
    #[cfg(target_os = "linux")]
    {
        let Some(api) = gp::api() else {
            return TetherStatus::NoLibrary;
        };
        match run(QUICK_TIMEOUT, "camera init", || gp::ensure_session(api)) {
            Ok(()) => TetherStatus::Available { lib_version: gp::lib_version(api) },
            Err(err) => {
                log::debug!("tether probe: {err:#}");
                TetherStatus::NoCamera
            }
        }
    }
    #[cfg(windows)]
    {
        match run(QUICK_TIMEOUT, "WIA device enumeration", || Ok(wia::probe())) {
            Ok(status) => status,
            Err(err) => {
                log::debug!("tether probe: {err:#}");
                TetherStatus::NoCamera
            }
        }
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        TetherStatus::NoLibrary
    }
}

/// Captures a still and downloads it, returning the JPEG and its dimensions.
pub fn capture_frame() -> Result<CapturedFrame> {
    capture_frame_with_timeout(CAPTURE_TIMEOUT)
}

/// [`capture_frame`] with an explicit budget for slow exposures.
pub fn capture_frame_with_timeout(timeout: Duration) -> Result<CapturedFrame> {
    #[cfg(target_os = "linux")]
    {
        let api = gp::api().ok_or_else(gp::missing_library)?;
        run(timeout, "capture", move || gp::capture(api))
    }
    #[cfg(windows)]
    {
        run(timeout, "WIA capture", wia::capture)
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        let _ = timeout;
        Err(gp::missing_library())
    }
}

/// One live-view frame scaled to at most `max_width` pixels wide.
pub fn preview_frame(max_width: u32) -> Result<RgbaImage> {
    #[cfg(target_os = "linux")]
    {
        let api = gp::api().ok_or_else(gp::missing_library)?;
        run(QUICK_TIMEOUT, "live view", move || gp::preview(api, max_width.max(1)))
    }
    #[cfg(windows)]
    {
        run(QUICK_TIMEOUT, "WIA preview", move || wia::preview(max_width.max(1)))
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        let _ = max_width;
        Err(gp::missing_library())
    }
}

/// Runs `job` on its own thread and gives up after `timeout`.
///
/// A timed-out call keeps running inside libgphoto2 (there is no portable way to
/// cancel it), so further calls are refused until it returns rather than racing
/// the same camera handle.
fn run<T: Send + 'static>(timeout: Duration, label: &str, job: impl FnOnce() -> Result<T> + Send + 'static) -> Result<T> {
    static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
    if IN_FLIGHT.fetch_add(1, Ordering::SeqCst) != 0 {
        IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
        bail!("NoCamera: libgphoto2 is still busy with a previous call that timed out");
    }
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let out = job();
        IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
        let _ = tx.send(out);
    });
    match rx.recv_timeout(timeout) {
        Ok(out) => out,
        Err(_) => bail!("{label} timed out after {:?}; the camera stopped responding", timeout),
    }
}

fn jpeg_dimensions(jpeg: &[u8]) -> Result<(u32, u32)> {
    let (w, h) = image::ImageReader::with_format(std::io::Cursor::new(jpeg), image::ImageFormat::Jpeg)
        .into_dimensions()
        .map_err(|err| anyhow::anyhow!("camera sent a JPEG without readable dimensions: {err}"))?;
    if w == 0 || h == 0 {
        bail!("camera sent an empty JPEG");
    }
    Ok((w, h))
}

#[cfg(target_os = "linux")]
mod gp {
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_int, c_ulong, c_void};
    use std::ptr;
    use std::sync::{Mutex, OnceLock};

    use anyhow::{anyhow, bail, Result};

    use super::{jpeg_dimensions, CapturedFrame};

    const RTLD_NOW: c_int = 2;
    const GP_OK: c_int = 0;
    /// `GP_CAPTURE_IMAGE`; other capture types are 1 and 2.
    const CAPTURE_IMAGE: c_int = 0;
    /// `GP_FILE_TYPE_NORMAL`.
    const FILE_TYPE_NORMAL: c_int = 1;
    /// `GP_VERSION_VERBOSE`.
    const VERSION_VERBOSE: c_int = 1;

    type Ctx = *mut c_void;
    type Cam = *mut c_void;
    type File = *mut c_void;

    /// `CameraFilePath` from `gphoto2-camera.h` 2.5.33: name[128], folder[1024].
    #[repr(C)]
    struct CameraFilePath {
        name: [c_char; 128],
        folder: [c_char; 1024],
    }

    unsafe extern "C" {
        fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    pub(super) struct Api {
        context_new: unsafe extern "C" fn() -> Ctx,
        context_unref: unsafe extern "C" fn(Ctx),
        camera_new: unsafe extern "C" fn(*mut Cam) -> c_int,
        camera_init: unsafe extern "C" fn(Cam, Ctx) -> c_int,
        camera_unref: unsafe extern "C" fn(Cam) -> c_int,
        capture: unsafe extern "C" fn(Cam, c_int, *mut CameraFilePath, Ctx) -> c_int,
        capture_preview: unsafe extern "C" fn(Cam, File, Ctx) -> c_int,
        file_get: unsafe extern "C" fn(Cam, *const c_char, *const c_char, c_int, File, Ctx) -> c_int,
        file_new: unsafe extern "C" fn(*mut File) -> c_int,
        file_unref: unsafe extern "C" fn(File) -> c_int,
        file_data: unsafe extern "C" fn(File, *mut *const c_char, *mut c_ulong) -> c_int,
        result_as_string: unsafe extern "C" fn(c_int) -> *const c_char,
        library_version: unsafe extern "C" fn(c_int) -> *const *const c_char,
    }

    /// An open camera; unrefs both handles on drop.
    struct Session {
        ctx: Ctx,
        cam: Cam,
    }

    // The C handles are not thread-affine; every use happens under `SESSION`.
    unsafe impl Send for Session {}

    impl Drop for Session {
        fn drop(&mut self) {
            if let Some(api) = api() {
                unsafe {
                    (api.camera_unref)(self.cam);
                    (api.context_unref)(self.ctx);
                }
            }
        }
    }

    static SESSION: Mutex<Option<Session>> = Mutex::new(None);

    pub(super) fn api() -> Option<&'static Api> {
        static API: OnceLock<Option<Api>> = OnceLock::new();
        API.get_or_init(load).as_ref()
    }

    fn load() -> Option<Api> {
        // Keep the handle for the process lifetime: the function pointers would
        // dangle after a `dlclose`.
        let handle = ["libgphoto2.so.6\0", "libgphoto2.so\0"].into_iter().find_map(|name| {
            let handle = unsafe { dlopen(name.as_ptr() as *const c_char, RTLD_NOW) };
            (!handle.is_null()).then_some(handle)
        })?;
        Some(Api {
            context_new: unsafe { symbol(handle, "gp_context_new") }?,
            context_unref: unsafe { symbol(handle, "gp_context_unref") }?,
            camera_new: unsafe { symbol(handle, "gp_camera_new") }?,
            camera_init: unsafe { symbol(handle, "gp_camera_init") }?,
            camera_unref: unsafe { symbol(handle, "gp_camera_unref") }?,
            capture: unsafe { symbol(handle, "gp_camera_capture") }?,
            capture_preview: unsafe { symbol(handle, "gp_camera_capture_preview") }?,
            file_get: unsafe { symbol(handle, "gp_camera_file_get") }?,
            file_new: unsafe { symbol(handle, "gp_file_new") }?,
            file_unref: unsafe { symbol(handle, "gp_file_unref") }?,
            file_data: unsafe { symbol(handle, "gp_file_get_data_and_size") }?,
            result_as_string: unsafe { symbol(handle, "gp_result_as_string") }?,
            library_version: unsafe { symbol(handle, "gp_library_version") }?,
        })
    }

    /// # Safety
    /// `handle` must come from a successful `dlopen`, and `name` must match `T`.
    unsafe fn symbol<T>(handle: *mut c_void, name: &str) -> Option<T> {
        let name = CString::new(name).ok()?;
        let ptr = unsafe { dlsym(handle, name.as_ptr()) };
        (!ptr.is_null()).then(|| unsafe { std::mem::transmute_copy(&ptr) })
    }

    pub fn missing_library() -> anyhow::Error {
        anyhow!("NoCamera: libgphoto2.so.6 is not loadable on this machine")
    }

    fn check(api: &Api, code: c_int, what: &str) -> Result<()> {
        if code == GP_OK {
            return Ok(());
        }
        let text = unsafe { (api.result_as_string)(code) };
        let text = if text.is_null() {
            String::from("unknown error")
        } else {
            unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
        };
        Err(anyhow!("NoCamera: {what} failed ({code}): {text}"))
    }

    pub fn lib_version(api: &Api) -> String {
        let list = unsafe { (api.library_version)(VERSION_VERBOSE) };
        if list.is_null() {
            return "unknown".into();
        }
        let first = unsafe { *list };
        if first.is_null() {
            return "unknown".into();
        }
        unsafe { CStr::from_ptr(first) }.to_string_lossy().into_owned()
    }

    /// Opens a camera once and keeps it in the static session.
    pub fn ensure_session(api: &'static Api) -> Result<()> {
        let mut slot = SESSION.lock().unwrap_or_else(|err| err.into_inner());
        if slot.is_some() {
            return Ok(());
        }
        let ctx = unsafe { (api.context_new)() };
        if ctx.is_null() {
            bail!("NoCamera: gp_context_new returned null");
        }
        let mut cam: Cam = ptr::null_mut();
        let code = unsafe { (api.camera_new)(&mut cam) };
        if code != GP_OK || cam.is_null() {
            unsafe { (api.context_unref)(ctx) };
            check(api, code, "gp_camera_new")?;
            bail!("NoCamera: gp_camera_new returned null");
        }
        let code = unsafe { (api.camera_init)(cam, ctx) };
        if code != GP_OK {
            unsafe {
                (api.camera_unref)(cam);
                (api.context_unref)(ctx);
            }
            check(api, code, "gp_camera_init (no camera attached?)")?;
        }
        *slot = Some(Session { ctx, cam });
        Ok(())
    }

    fn with_session<T>(api: &'static Api, f: impl FnOnce(&Api, Cam, Ctx) -> Result<T>) -> Result<T> {
        ensure_session(api)?;
        let mut slot = SESSION.lock().unwrap_or_else(|err| err.into_inner());
        let session = slot.as_mut().ok_or_else(|| anyhow!("NoCamera: camera session disappeared"))?;
        f(api, session.cam, session.ctx)
    }

    /// Copies the bytes out of a `CameraFile` before it is unreffed.
    fn file_bytes(api: &Api, file: File) -> Result<Vec<u8>> {
        let mut data: *const c_char = ptr::null();
        let mut len: c_ulong = 0;
        let code = unsafe { (api.file_data)(file, &mut data, &mut len) };
        check(api, code, "gp_file_get_data_and_size")?;
        if data.is_null() || len == 0 {
            bail!("NoCamera: camera returned an empty file");
        }
        Ok(unsafe { std::slice::from_raw_parts(data as *const u8, len as usize) }.to_vec())
    }

    pub fn capture(api: &'static Api) -> Result<CapturedFrame> {
        with_session(api, |api, cam, ctx| unsafe {
            let mut path = CameraFilePath { name: [0; 128], folder: [0; 1024] };
            let code = (api.capture)(cam, CAPTURE_IMAGE, &mut path, ctx);
            check(api, code, "gp_camera_capture")?;

            let mut file: File = ptr::null_mut();
            let code = (api.file_new)(&mut file);
            check(api, code, "gp_file_new")?;
            if file.is_null() {
                bail!("NoCamera: gp_file_new returned null");
            }
            let downloaded = (|| {
                let code = (api.file_get)(
                    cam,
                    path.folder.as_ptr(),
                    path.name.as_ptr(),
                    FILE_TYPE_NORMAL,
                    file,
                    ctx,
                );
                check(api, code, "gp_camera_file_get")?;
                let jpeg = file_bytes(api, file)?;
                let (width, height) = jpeg_dimensions(&jpeg)?;
                Ok(CapturedFrame { jpeg, width, height })
            })();
            (api.file_unref)(file);
            downloaded
        })
    }

    pub fn preview(api: &'static Api, max_width: u32) -> Result<crate::engine::RgbaImage> {
        let jpeg = with_session(api, |api, cam, ctx| unsafe {
            let mut file: File = ptr::null_mut();
            let code = (api.file_new)(&mut file);
            check(api, code, "gp_file_new")?;
            if file.is_null() {
                bail!("NoCamera: gp_file_new returned null");
            }
            let grabbed = (|| {
                let code = (api.capture_preview)(cam, file, ctx);
                check(api, code, "gp_camera_capture_preview")?;
                file_bytes(api, file)
            })();
            (api.file_unref)(file);
            grabbed
        })?;

        let decoded = image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg)
            .map_err(|err| anyhow!("camera preview is not a decodable JPEG: {err}"))?;
        let rgba = decoded.into_rgba8();
        if rgba.width() <= max_width {
            return Ok(crate::engine::RgbaImage::new(rgba.width(), rgba.height(), rgba.into_raw()));
        }
        let height = (u64::from(rgba.height()) * u64::from(max_width) / u64::from(rgba.width())).max(1) as u32;
        let scaled = image::imageops::resize(&rgba, max_width, height, image::imageops::FilterType::Triangle);
        Ok(crate::engine::RgbaImage::new(max_width, height, scaled.into_raw()))
    }
}

#[cfg(windows)]
mod wia {
    //! Minimal WIA COM binding, built on the raw vtables from `wia_lh.h`.
    //!
    //! Only the slots we call are bound. Every interface pointer is owned by
    //! [`ComPtr`], which calls `Release`, and [`ComInit`] balances
    //! `CoInitializeEx`. WIA has no "take picture" without a device dialog, so
    //! capture uses the optional `WIA_CMD_TAKE_PICTURE` command and otherwise
    //! reports `Unsupported`; images are then pulled through `IWiaTransfer`.
    use std::ffi::c_void;
    use std::ptr;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use anyhow::{anyhow, bail, Result};

    use super::{jpeg_dimensions, CapturedFrame, TetherStatus};

    #[repr(C)]
    #[derive(Clone, Copy, PartialEq, Eq)]
    struct Guid {
        data1: u32,
        data2: u16,
        data3: u16,
        data4: [u8; 8],
    }

    const fn guid(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Guid {
        Guid { data1, data2, data3, data4 }
    }

    // wia_lh.h: IID_IWiaDevMgr2 / IID_IWiaItem2 / IID_IWiaTransfer / IID_IWiaPreview,
    // the WiaDevMgr2 class registration, and WiaDef.h: WIA_CMD_TAKE_PICTURE.
    const CLSID_WIA_DEV_MGR2: Guid = guid(0xB6C292BC, 0x7C88, 0x41EE, [0x8B, 0x54, 0x8E, 0xC9, 0x26, 0x17, 0xE5, 0x99]);
    const IID_IWIA_DEV_MGR2: Guid = guid(0x79C07CF1, 0xCBDD, 0x41EE, [0x8E, 0xC3, 0xF0, 0x00, 0x80, 0xCA, 0xDA, 0x7A]);
    const IID_IWIA_TRANSFER: Guid = guid(0xC39D6942, 0x2F4E, 0x4D04, [0x92, 0xFE, 0x4E, 0xF4, 0xD3, 0xA1, 0xDE, 0x5A]);
    const IID_IWIA_TRANSFER_CALLBACK: Guid = guid(0x27D4EAAF, 0x28A6, 0x4CA5, [0x9A, 0xAB, 0xE6, 0x78, 0x16, 0x8B, 0x95, 0x27]);
    const IID_IUNKNOWN: Guid = guid(0, 0, 0, [0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46]);
    const WIA_CMD_TAKE_PICTURE: Guid = guid(0xAF933CAC, 0xACAD, 0x11D2, [0xA0, 0x93, 0x00, 0xC0, 0x4F, 0x72, 0xDC, 0x3C]);

    /// `StiDeviceTypeDigitalCamera`; `WIA_DEVICE_TYPE_ALL` catches drivers that
    /// register as streaming video instead.
    const WIA_DEVICE_TYPE_IMAGE: i32 = 2;
    const WIA_DEVICE_TYPE_ALL: i32 = 0;
    const WIA_DIP_DEV_ID: u32 = 2;
    const WIA_DIP_DEV_NAME: u32 = 7;

    const COINIT_MULTITHREADED: u32 = 0;
    const CLSCTX_ALL: u32 = 0x17;
    const S_OK: i32 = 0;
    const S_FALSE: i32 = 1;
    const E_NOINTERFACE: i32 = 0x8000_4002u32 as i32;
    const E_INVALIDARG: i32 = 0x8007_0057u32 as i32;
    const RPC_E_CHANGED_MODE: i32 = 0x8001_0106u32 as i32;
    const VT_BSTR: u16 = 8;
    const VT_LPWSTR: u16 = 31;
    const PRSPEC_PROPID: u32 = 1;

    #[link(name = "ole32")]
    unsafe extern "system" {
        fn CoInitializeEx(reserved: *mut c_void, coinit: u32) -> i32;
        fn CoUninitialize();
        fn CoCreateInstance(rclsid: *const Guid, outer: *mut c_void, ctx: u32, riid: *const Guid, out: *mut *mut c_void) -> i32;
        fn CoTaskMemFree(p: *mut c_void);
        fn CreateStreamOnHGlobal(h: *mut c_void, delete_on_release: i32, out: *mut *mut c_void) -> i32;
    }

    #[link(name = "oleaut32")]
    unsafe extern "system" {
        fn SysAllocString(s: *const u16) -> *mut u16;
        fn SysFreeString(s: *mut u16);
    }

    /// Balances `CoInitializeEx` for this thread. `RPC_E_CHANGED_MODE` means COM
    /// is already up under another model, which WIA can still use.
    struct ComInit {
        owns: bool,
    }

    impl ComInit {
        fn new() -> std::result::Result<Self, i32> {
            let hr = unsafe { CoInitializeEx(ptr::null_mut(), COINIT_MULTITHREADED) };
            if hr == S_OK || hr == S_FALSE {
                Ok(Self { owns: true })
            } else if hr == RPC_E_CHANGED_MODE {
                Ok(Self { owns: false })
            } else {
                Err(hr)
            }
        }
    }

    impl Drop for ComInit {
        fn drop(&mut self) {
            if self.owns {
                unsafe { CoUninitialize() };
            }
        }
    }

    fn hr_text(hr: i32) -> String {
        format!("0x{:08X}", hr as u32)
    }

    unsafe fn vtable_slot(obj: *mut c_void, index: usize) -> *const c_void {
        let table = unsafe { *(obj as *mut *const *const c_void) };
        unsafe { *table.add(index) }
    }

    /// Owns one COM reference and releases it on drop.
    struct ComPtr(*mut c_void);

    unsafe impl Send for ComPtr {}

    impl ComPtr {
        fn new(ptr: *mut c_void) -> Self {
            Self(ptr)
        }

        fn ptr(&self) -> *mut c_void {
            self.0
        }
    }

    impl Drop for ComPtr {
        fn drop(&mut self) {
            if !self.0.is_null() {
                let release: unsafe extern "system" fn(*mut c_void) -> u32 =
                    unsafe { std::mem::transmute(vtable_slot(self.0, 2)) };
                unsafe { release(self.0) };
            }
        }
    }

    fn co_create(clsid: &Guid, iid: &Guid) -> Result<ComPtr> {
        let mut out: *mut c_void = ptr::null_mut();
        let hr = unsafe { CoCreateInstance(clsid, ptr::null_mut(), CLSCTX_ALL, iid, &mut out) };
        if hr < 0 || out.is_null() {
            bail!("WIA device manager is unavailable (CoCreateInstance HRESULT {})", hr_text(hr));
        }
        Ok(ComPtr::new(out))
    }

    fn query_interface(obj: *mut c_void, iid: &Guid) -> Option<ComPtr> {
        let query: unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32 =
            unsafe { std::mem::transmute(vtable_slot(obj, 0)) };
        let mut out: *mut c_void = ptr::null_mut();
        let hr = unsafe { query(obj, iid, &mut out) };
        (hr >= 0 && !out.is_null()).then(|| ComPtr::new(out))
    }

    fn to_bstr(text: &str) -> *mut u16 {
        let wide: Vec<u16> = text.encode_utf16().chain(std::iter::once(0)).collect();
        unsafe { SysAllocString(wide.as_ptr()) }
    }

    unsafe fn wide_to_string(p: *const u16) -> String {
        if p.is_null() {
            return String::new();
        }
        let mut len = 0usize;
        while len < 4096 && unsafe { *p.add(len) } != 0 {
            len += 1;
        }
        String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(p, len) })
    }

    #[repr(C)]
    struct PropSpec {
        kind: u32,
        propid: u32,
    }

    /// `PROPVARIANT` prefix plus the union slots we read; 16 bytes of union on
    /// x64 and 8 on x86, so `usize` keeps the layout right on both.
    #[repr(C)]
    struct PropVariant {
        vt: u16,
        reserved1: u16,
        reserved2: u16,
        reserved3: u16,
        data: [usize; 2],
    }

    fn device_property(storage: *mut c_void, propid: u32) -> String {
        let spec = PropSpec { kind: PRSPEC_PROPID, propid };
        let mut value = PropVariant { vt: 0, reserved1: 0, reserved2: 0, reserved3: 0, data: [0; 2] };
        let read: unsafe extern "system" fn(*mut c_void, u32, *const PropSpec, *mut PropVariant) -> i32 =
            unsafe { std::mem::transmute(vtable_slot(storage, 3)) };
        let hr = unsafe { read(storage, 1, &spec, &mut value) };
        if hr < 0 {
            return String::new();
        }
        unsafe {
            match value.vt {
                VT_BSTR => {
                    let p = value.data[0] as *mut u16;
                    let text = wide_to_string(p);
                    if !p.is_null() {
                        SysFreeString(p);
                    }
                    text
                }
                VT_LPWSTR => {
                    let p = value.data[0] as *mut u16;
                    let text = wide_to_string(p);
                    if !p.is_null() {
                        CoTaskMemFree(p as *mut c_void);
                    }
                    text
                }
                _ => String::new(),
            }
        }
    }

    struct Device {
        name: String,
        id: String,
    }

    fn enumerate_devices(devmgr: *mut c_void, kind: i32) -> Result<Vec<Device>> {
        let mut enumerator: *mut c_void = ptr::null_mut();
        let enumerate: unsafe extern "system" fn(*mut c_void, i32, *mut *mut c_void) -> i32 =
            unsafe { std::mem::transmute(vtable_slot(devmgr, 3)) };
        let hr = unsafe { enumerate(devmgr, kind, &mut enumerator) };
        if hr < 0 || enumerator.is_null() {
            return Ok(Vec::new());
        }
        let enumerator = ComPtr::new(enumerator);
        let next: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void, *mut u32) -> i32 =
            unsafe { std::mem::transmute(vtable_slot(enumerator.ptr(), 3)) };
        let mut devices = Vec::new();
        while devices.len() < 32 {
            let mut storage: *mut c_void = ptr::null_mut();
            let mut fetched = 0u32;
            let hr = unsafe { next(enumerator.ptr(), 1, &mut storage, &mut fetched) };
            if hr != S_OK || fetched == 0 || storage.is_null() {
                break;
            }
            let storage = ComPtr::new(storage);
            devices.push(Device {
                name: device_property(storage.ptr(), WIA_DIP_DEV_NAME),
                id: device_property(storage.ptr(), WIA_DIP_DEV_ID),
            });
        }
        Ok(devices)
    }

    fn list_devices() -> Result<Vec<Device>> {
        let devmgr = co_create(&CLSID_WIA_DEV_MGR2, &IID_IWIA_DEV_MGR2)?;
        let mut devices = enumerate_devices(devmgr.ptr(), WIA_DEVICE_TYPE_IMAGE)?;
        if devices.is_empty() {
            devices = enumerate_devices(devmgr.ptr(), WIA_DEVICE_TYPE_ALL)?;
        }
        Ok(devices)
    }

    #[cfg(test)]
    pub(super) fn device_count_for_test() -> Result<usize> {
        let _com = ComInit::new().map_err(|hr| anyhow!("CoInitializeEx failed (HRESULT {})", hr_text(hr)))?;
        Ok(list_devices()?.len())
    }

    pub(super) fn probe() -> TetherStatus {
        let _com = match ComInit::new() {
            Ok(com) => com,
            Err(_) => return TetherStatus::NoLibrary,
        };
        match list_devices() {
            Ok(devices) if devices.is_empty() => TetherStatus::NoCamera,
            Ok(devices) => {
                let name = devices[0].name.trim();
                let label = if name.is_empty() { "WIA imaging device" } else { name };
                TetherStatus::Available { lib_version: format!("WIA {label}") }
            }
            Err(err) => {
                log::debug!("tether probe: {err:#}");
                TetherStatus::NoCamera
            }
        }
    }

    fn create_device(device_id: &str) -> Result<ComPtr> {
        let devmgr = co_create(&CLSID_WIA_DEV_MGR2, &IID_IWIA_DEV_MGR2)?;
        let id = to_bstr(device_id);
        if id.is_null() {
            bail!("NoCamera: WIA device id could not be allocated");
        }
        let create: unsafe extern "system" fn(*mut c_void, *mut u16, *mut *mut c_void) -> i32 =
            unsafe { std::mem::transmute(vtable_slot(devmgr.ptr(), 4)) };
        let mut root: *mut c_void = ptr::null_mut();
        let hr = unsafe { create(devmgr.ptr(), id, &mut root) };
        unsafe { SysFreeString(id) };
        if hr < 0 || root.is_null() {
            bail!("NoCamera: WIA CreateDevice failed (HRESULT {})", hr_text(hr));
        }
        Ok(ComPtr::new(root))
    }

    pub(super) fn capture() -> Result<CapturedFrame> {
        let _com = ComInit::new().map_err(|hr| anyhow!("NoLibrary: CoInitializeEx failed (HRESULT {})", hr_text(hr)))?;
        let devices = list_devices()?;
        let device = devices.first().ok_or_else(|| anyhow!("NoCamera: no WIA imaging device is connected"))?;
        let root = create_device(&device.id)?;

        let mut child: *mut c_void = ptr::null_mut();
        let command: unsafe extern "system" fn(*mut c_void, i32, *const Guid, *mut *mut c_void) -> i32 =
            unsafe { std::mem::transmute(vtable_slot(root.ptr(), 10)) };
        let hr = unsafe { command(root.ptr(), 0, &WIA_CMD_TAKE_PICTURE, &mut child) };
        if hr < 0 || child.is_null() {
            bail!(
                "Unsupported: this device needs a vendor SDK (WIA_CMD_TAKE_PICTURE failed, HRESULT {})",
                hr_text(hr)
            );
        }
        let child = ComPtr::new(child);
        let jpeg = download(&child)?;
        if !jpeg.starts_with(&[0xFF, 0xD8]) {
            bail!("Unsupported: the WIA driver returned a non-JPEG image");
        }
        let (width, height) = jpeg_dimensions(&jpeg)?;
        Ok(CapturedFrame { jpeg, width, height })
    }

    fn download(item: &ComPtr) -> Result<Vec<u8>> {
        let transfer = query_interface(item.ptr(), &IID_IWIA_TRANSFER)
            .ok_or_else(|| anyhow!("Unsupported: the WIA item does not expose IWiaTransfer"))?;
        let sink = sink_new();
        let download: unsafe extern "system" fn(*mut c_void, i32, *mut c_void) -> i32 =
            unsafe { std::mem::transmute(vtable_slot(transfer.ptr(), 3)) };
        let hr = unsafe { download(transfer.ptr(), 0, sink as *mut c_void) };
        let bytes = unsafe { (*sink).take_bytes() };
        sink_release(sink);
        if hr < 0 {
            bail!("NoCamera: WIA transfer failed (HRESULT {})", hr_text(hr));
        }
        if bytes.is_empty() {
            bail!("NoCamera: the WIA device returned no image data");
        }
        Ok(bytes)
    }

    pub(super) fn preview(max_width: u32) -> Result<crate::engine::RgbaImage> {
        let _com = ComInit::new().map_err(|hr| anyhow!("NoLibrary: CoInitializeEx failed (HRESULT {})", hr_text(hr)))?;
        let devices = list_devices()?;
        let device = devices.first().ok_or_else(|| anyhow!("NoCamera: no WIA imaging device is connected"))?;
        let root = create_device(&device.id)?;

        let mut preview: *mut c_void = ptr::null_mut();
        let get_preview: unsafe extern "system" fn(*mut c_void, i32, *mut *mut c_void) -> i32 =
            unsafe { std::mem::transmute(vtable_slot(root.ptr(), 16)) };
        let hr = unsafe { get_preview(root.ptr(), 0, &mut preview) };
        if hr < 0 || preview.is_null() {
            bail!(
                "Unsupported: this device has no WIA preview component (HRESULT {})",
                hr_text(hr)
            );
        }
        let preview = ComPtr::new(preview);

        let sink = sink_new();
        let new_preview: unsafe extern "system" fn(*mut c_void, i32, *mut c_void, *mut c_void) -> i32 =
            unsafe { std::mem::transmute(vtable_slot(preview.ptr(), 3)) };
        let hr = unsafe { new_preview(preview.ptr(), 0, root.ptr(), sink as *mut c_void) };
        let bytes = unsafe { (*sink).take_bytes() };
        sink_release(sink);
        if hr < 0 {
            bail!("Unsupported: WIA preview failed (HRESULT {})", hr_text(hr));
        }
        if bytes.is_empty() {
            bail!("Unsupported: the WIA preview component returned no image");
        }
        dib_to_rgba(&bytes, max_width)
    }

    /// WIA previews arrive as a packed DIB (`BITMAPINFOHEADER`, bottom-up unless
    /// the height is negative). Only straight 24/32-bit RGB is accepted.
    pub(super) fn dib_to_rgba(bytes: &[u8], max_width: u32) -> Result<crate::engine::RgbaImage> {
        let start = if bytes.starts_with(b"BM") { 14 } else { 0 };
        if bytes.len() < start + 40 {
            bail!("Unsupported: the WIA preview is not a DIB");
        }
        let header = &bytes[start..];
        let header_size = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
        let width = i32::from_le_bytes(header[4..8].try_into().unwrap());
        let height = i32::from_le_bytes(header[8..12].try_into().unwrap());
        let bit_count = u16::from_le_bytes(header[14..16].try_into().unwrap());
        let compression = u32::from_le_bytes(header[16..20].try_into().unwrap());
        if header_size < 40 || width <= 0 || height == 0 || compression != 0 || !matches!(bit_count, 24 | 32) {
            bail!(
                "Unsupported: WIA preview DIB is {}x{} at {} bpp, compression {}",
                width,
                height,
                bit_count,
                compression
            );
        }
        let width = width as u32;
        let rows = height.unsigned_abs();
        let bytes_per_pixel = (bit_count / 8) as usize;
        let stride = ((width as usize * bytes_per_pixel + 3) / 4) * 4;
        let pixels_start = start + header_size;
        let needed = pixels_start + stride * rows as usize;
        if bytes.len() < needed {
            bail!("Unsupported: the WIA preview DIB is truncated");
        }

        let top_down = height < 0;
        let mut rgba = vec![0u8; (width * rows * 4) as usize];
        for y in 0..rows as usize {
            let source_row = if top_down { y } else { rows as usize - 1 - y };
            let row = &bytes[pixels_start + source_row * stride..];
            for x in 0..width as usize {
                let pixel = &row[x * bytes_per_pixel..];
                let out = (y * width as usize + x) * 4;
                rgba[out] = pixel[2];
                rgba[out + 1] = pixel[1];
                rgba[out + 2] = pixel[0];
                rgba[out + 3] = 255;
            }
        }

        if width <= max_width {
            return Ok(crate::engine::RgbaImage::new(width, rows, rgba));
        }
        let scaled_height = (u64::from(rows) * u64::from(max_width) / u64::from(width)).max(1) as u32;
        let image = image::RgbaImage::from_raw(width, rows, rgba)
            .ok_or_else(|| anyhow!("Unsupported: the WIA preview DIB has an invalid size"))?;
        let scaled = image::imageops::resize(&image, max_width, scaled_height, image::imageops::FilterType::Triangle);
        Ok(crate::engine::RgbaImage::new(max_width, scaled_height, scaled.into_raw()))
    }

    #[repr(C)]
    struct CallbackVtbl {
        query_interface: unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
        add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
        release: unsafe extern "system" fn(*mut c_void) -> u32,
        transfer_callback: unsafe extern "system" fn(*mut c_void, i32, *mut c_void) -> i32,
        get_next_stream: unsafe extern "system" fn(*mut c_void, i32, *mut u16, *mut u16, *mut *mut c_void) -> i32,
    }

    static CALLBACK_VTBL: CallbackVtbl = CallbackVtbl {
        query_interface: callback_query_interface,
        add_ref: callback_add_ref,
        release: callback_release,
        transfer_callback: callback_transfer,
        get_next_stream: callback_next_stream,
    };

    /// The IWiaTransferCallback WIA writes the item into. One memory stream is
    /// kept alive by our own reference so it can be read after `Download`.
    #[repr(C)]
    struct TransferSink {
        vtbl: *const CallbackVtbl,
        refs: AtomicUsize,
        stream: Mutex<Option<ComPtr>>,
    }

    fn sink_new() -> *mut TransferSink {
        Box::into_raw(Box::new(TransferSink {
            vtbl: &CALLBACK_VTBL,
            refs: AtomicUsize::new(1),
            stream: Mutex::new(None),
        }))
    }

    fn sink_release(sink: *mut TransferSink) {
        unsafe { callback_release(sink as *mut c_void) };
    }

    impl TransferSink {
        fn take_bytes(&self) -> Vec<u8> {
            let guard = self.stream.lock().unwrap_or_else(|err| err.into_inner());
            match guard.as_ref() {
                Some(stream) => unsafe { read_stream(stream.ptr()) },
                None => Vec::new(),
            }
        }
    }

    unsafe fn read_stream(stream: *mut c_void) -> Vec<u8> {
        let seek: unsafe extern "system" fn(*mut c_void, u64, u32, *mut u64) -> i32 =
            unsafe { std::mem::transmute(vtable_slot(stream, 5)) };
        let mut position = 0u64;
        unsafe { seek(stream, 0, 0, &mut position) };
        let read: unsafe extern "system" fn(*mut c_void, *mut u8, u32, *mut u32) -> i32 =
            unsafe { std::mem::transmute(vtable_slot(stream, 3)) };
        let mut out = Vec::new();
        loop {
            let start = out.len();
            out.resize(start + 64 * 1024, 0);
            let mut got = 0u32;
            let hr = unsafe { read(stream, out.as_mut_ptr().add(start), 64 * 1024, &mut got) };
            out.truncate(start + got as usize);
            if hr < 0 || got == 0 {
                break;
            }
        }
        out
    }

    unsafe extern "system" fn callback_query_interface(this: *mut c_void, iid: *const Guid, out: *mut *mut c_void) -> i32 {
        if out.is_null() || iid.is_null() {
            return E_INVALIDARG;
        }
        let iid = unsafe { *iid };
        if iid == IID_IUNKNOWN || iid == IID_IWIA_TRANSFER_CALLBACK {
            unsafe { *out = this };
            unsafe { callback_add_ref(this) };
            S_OK
        } else {
            unsafe { *out = ptr::null_mut() };
            E_NOINTERFACE
        }
    }

    unsafe extern "system" fn callback_add_ref(this: *mut c_void) -> u32 {
        let sink = unsafe { &*(this as *const TransferSink) };
        (sink.refs.fetch_add(1, Ordering::SeqCst) + 1) as u32
    }

    unsafe extern "system" fn callback_release(this: *mut c_void) -> u32 {
        let sink = this as *mut TransferSink;
        if unsafe { (*sink).refs.fetch_sub(1, Ordering::SeqCst) } == 1 {
            drop(unsafe { Box::from_raw(sink) });
            return 0;
        }
        unsafe { (*sink).refs.load(Ordering::SeqCst) as u32 }
    }

    unsafe extern "system" fn callback_transfer(_this: *mut c_void, _flags: i32, _params: *mut c_void) -> i32 {
        S_OK
    }

    unsafe extern "system" fn callback_next_stream(
        this: *mut c_void,
        _flags: i32,
        _item_name: *mut u16,
        _full_item_name: *mut u16,
        out: *mut *mut c_void,
    ) -> i32 {
        if out.is_null() {
            return E_INVALIDARG;
        }
        let mut stream: *mut c_void = ptr::null_mut();
        let hr = unsafe { CreateStreamOnHGlobal(ptr::null_mut(), 1, &mut stream) };
        if hr < 0 || stream.is_null() {
            unsafe { *out = ptr::null_mut() };
            return hr;
        }
        let add_ref: unsafe extern "system" fn(*mut c_void) -> u32 =
            unsafe { std::mem::transmute(vtable_slot(stream, 1)) };
        unsafe { add_ref(stream) };
        let sink = unsafe { &*(this as *const TransferSink) };
        let mut guard = sink.stream.lock().unwrap_or_else(|err| err.into_inner());
        *guard = Some(ComPtr::new(stream));
        unsafe { *out = stream };
        S_OK
    }

}

#[cfg(not(any(target_os = "linux", windows)))]
mod gp {
    use anyhow::{anyhow, Result};

    pub fn missing_library() -> anyhow::Error {
        anyhow!("NoCamera: libgphoto2 tethering is only implemented on Linux")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_reports_one_of_the_three_states() {
        let status = probe();
        println!("tether probe: {status:?}");
        match status {
            TetherStatus::NoLibrary => {}
            TetherStatus::NoCamera => {}
            TetherStatus::Available { lib_version } => assert!(!lib_version.is_empty()),
        }
    }

    #[test]
    fn capture_without_a_camera_is_a_readable_error() {
        if matches!(probe(), TetherStatus::Available { .. }) {
            return;
        }
        let err = capture_frame().expect_err("capture must fail without a camera").to_string();
        println!("capture error: {err}");
        assert!(err.contains("NoCamera"), "unexpected error: {err}");
    }

    /// Needs a real camera on USB/PTP and `libgphoto2.so.6`; run with
    /// `cargo test -p lumen-engine -- --ignored tether`.
    #[test]
    #[ignore]
    fn real_camera_captures_and_previews() {
        assert!(matches!(probe(), TetherStatus::Available { .. }), "no camera: {:?}", probe());
        let frame = capture_frame().expect("capture");
        assert!(frame.width > 0 && frame.height > 0);
        let preview = preview_frame(1024).expect("preview");
        assert!(preview.width <= 1024 && preview.width > 0);
    }

    /// WIA ships with Windows, so the Windows backend must never degrade to
    /// `NoLibrary`, and repeated probes must not panic on COM teardown.
    #[cfg(windows)]
    #[test]
    fn windows_probe_is_available_and_does_not_panic() {
        let first = probe();
        let second = probe();
        println!("tether probe: {first:?} / {second:?}");
        assert_ne!(first, TetherStatus::NoLibrary, "WIA COM init failed on Windows");
        if let TetherStatus::Available { lib_version } = &first {
            assert!(lib_version.starts_with("WIA ") && lib_version.len() > 4, "unexpected device label: {lib_version}");
        }
    }

    /// With no WIA image device attached, both probe and capture must say
    /// `NoCamera` rather than reporting a driver failure. A real camera makes
    /// this a no-op; hardware paths are covered by `--ignored tether`.
    #[cfg(windows)]
    #[test]
    fn windows_without_a_camera_reports_no_camera() {
        let status = probe();
        println!("tether probe: {status:?}");
        if matches!(status, TetherStatus::Available { .. }) {
            return;
        }
        assert_eq!(status, TetherStatus::NoCamera);
        let err = capture_frame().expect_err("capture must fail without a camera").to_string();
        assert!(err.contains("NoCamera"), "unexpected error: {err}");
    }

    /// Proves the COM class id and vtable slots really reach WIA, so a broken
    /// enumeration cannot hide behind `probe`'s `NoCamera` fallback.
    #[cfg(windows)]
    #[test]
    fn windows_wia_enumeration_reaches_the_device_manager() {
        let count = wia::device_count_for_test().expect("WiaDevMgr2 enumeration");
        println!("WIA imaging devices: {count}");
    }

    /// 2x2 24-bit bottom-up DIB: bottom row blue, top row red.
    #[cfg(windows)]
    #[test]
    fn dib_preview_decodes_and_flips_rows() {
        let mut dib = vec![0u8; 40 + 16];
        dib[0..4].copy_from_slice(&40u32.to_le_bytes());
        dib[4..8].copy_from_slice(&2i32.to_le_bytes());
        dib[8..12].copy_from_slice(&2i32.to_le_bytes());
        dib[12..14].copy_from_slice(&1u16.to_le_bytes());
        dib[14..16].copy_from_slice(&24u16.to_le_bytes());
        dib[20..24].copy_from_slice(&16u32.to_le_bytes());
        for x in 0..2 {
            let offset = 40 + x * 3;
            dib[offset..offset + 3].copy_from_slice(&[255, 0, 0]);
        }
        for x in 0..2 {
            let offset = 40 + 8 + x * 3;
            dib[offset..offset + 3].copy_from_slice(&[0, 0, 255]);
        }

        let image = wia::dib_to_rgba(&dib, 2).expect("decode DIB");
        assert_eq!((image.width, image.height), (2, 2));
        assert_eq!(&image.pixels[0..4], &[255, 0, 0, 255]);
        assert_eq!(&image.pixels[8..12], &[0, 0, 255, 255]);
    }
}
