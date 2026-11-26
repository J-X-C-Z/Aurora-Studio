use super::{Handle, c_char, c_void};
use std::{io, sync::OnceLock};

const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x800;
const OPENCL: &[u16] = &[79, 112, 101, 110, 67, 76, 46, 100, 108, 108, 0];

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryExW(name: *const u16, file: Handle, flags: u32) -> Handle;
    fn GetProcAddress(module: Handle, name: *const c_char) -> *const c_void;
    fn FreeLibrary(module: Handle) -> i32;
}

struct Library(Handle);

impl Drop for Library {
    fn drop(&mut self) {
        unsafe { FreeLibrary(self.0); }
    }
}

type ContextCallback = Option<unsafe extern "system" fn(*const c_char, *const c_void, usize, Handle)>;
type ProgramCallback = Option<unsafe extern "system" fn(Handle, Handle)>;

macro_rules! opencl_api {
    ($($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty;)*) => {
        #[allow(non_snake_case)]
        pub(super) struct Api {
            $(pub(super) $name: unsafe extern "system" fn($($ty),*) -> $ret,)*
        }

        impl Api {
            #[allow(non_snake_case)]
            fn load() -> io::Result<Self> {
                let handle = unsafe { LoadLibraryExW(OPENCL.as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32) };
                if handle.is_null() {
                    return Err(io::Error::other(format!("System32 OpenCL.dll: {}", io::Error::last_os_error())));
                }
                let library = Library(handle);
                $(let $name = unsafe { GetProcAddress(library.0, concat!(stringify!($name), "\0").as_ptr().cast()) };
                  if $name.is_null() {
                      return Err(io::Error::other(concat!("System32 OpenCL.dll missing ", stringify!($name))));
                  })*
                let api = Self {
                    $($name: unsafe { std::mem::transmute::<*const c_void, unsafe extern "system" fn($($ty),*) -> $ret>($name) },)*
                };
                std::mem::forget(library);
                Ok(api)
            }
        }
    };
}

opencl_api! {
    clGetPlatformIDs(count: u32, ids: *mut Handle, actual: *mut u32) -> i32;
    clGetDeviceIDs(platform: Handle, kind: u64, count: u32, ids: *mut Handle, actual: *mut u32) -> i32;
    clGetDeviceInfo(device: Handle, key: u32, size: usize, value: *mut c_void, actual: *mut usize) -> i32;
    clCreateContext(props: *const isize, count: u32, devices: *const Handle, notify: ContextCallback, user: Handle, err: *mut i32) -> Handle;
    clCreateCommandQueue(context: Handle, device: Handle, flags: u64, err: *mut i32) -> Handle;
    clCreateProgramWithSource(context: Handle, count: u32, texts: *const *const c_char, sizes: *const usize, err: *mut i32) -> Handle;
    clBuildProgram(program: Handle, count: u32, devices: *const Handle, options: *const c_char, notify: ProgramCallback, user: Handle) -> i32;
    clCreateKernel(program: Handle, name: *const c_char, err: *mut i32) -> Handle;
    clCreateBuffer(context: Handle, flags: u64, bytes: usize, host: Handle, err: *mut i32) -> Handle;
    clSetKernelArg(kernel: Handle, index: u32, size: usize, arg: *const c_void) -> i32;
    clEnqueueNDRangeKernel(queue: Handle, kernel: Handle, dimensions: u32, offset: *const usize, global: *const usize, local: *const usize, count: u32, wait: *const Handle, event: *mut Handle) -> i32;
    clEnqueueReadBuffer(queue: Handle, source: Handle, blocking: u32, offset: usize, size: usize, output: Handle, count: u32, wait: *const Handle, event: *mut Handle) -> i32;
    clReleaseContext(handle: Handle) -> i32;
    clReleaseCommandQueue(handle: Handle) -> i32;
    clReleaseProgram(handle: Handle) -> i32;
    clReleaseKernel(handle: Handle) -> i32;
    clReleaseMemObject(handle: Handle) -> i32;
}

pub(super) fn bindings() -> io::Result<&'static Api> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();
    use aurora_portrait::native_infer::backend::{BackendKind,BackendRequest};
    if !BackendRequest::from_env().map_err(io::Error::other)?.permits(BackendKind::OpenCl) {
        return Err(io::Error::other("AURORA_GPU_BACKEND excludes OpenCL"));
    }
    API.get_or_init(|| Api::load().map_err(|e|e.to_string()))
        .as_ref().map_err(|message| io::Error::other(message.clone()))
}
