use super::NikonCrop;
use std::{cell::RefCell, collections::HashMap, ffi::{c_char, c_void}, io, ptr::{null, null_mut}};
use super::Stage1Params;

type Handle = *mut c_void;

#[path = "system_opencl.rs"]
mod system_opencl;
use system_opencl::{Api, bindings};


struct Resource(Handle, unsafe extern "system" fn(Handle) -> i32);
impl Drop for Resource {
    fn drop(&mut self) { unsafe { (self.1)(self.0); } }
}

struct Runtime {
    programs: HashMap<&'static str, Resource>,
    queue: Resource,
    context: Resource,
    device: Handle,
}

thread_local! {
    static RUNTIME: RefCell<Option<Runtime>> = const { RefCell::new(None) };
}

impl Runtime {
    fn new(cl: &Api) -> io::Result<Self> {
        let device = nvidia_device(cl)?;
        let mut error = 0;
        let handle = unsafe { (cl.clCreateContext)(null(), 1, &device, None, null_mut(), &mut error) };
        check("context", error)?;
        let context = Resource(handle, cl.clReleaseContext);
        let handle = unsafe { (cl.clCreateCommandQueue)(context.0, device, 0, &mut error) };
        check("queue", error)?;
        let queue = Resource(handle, cl.clReleaseCommandQueue);
        Ok(Self { programs: HashMap::new(), queue, context, device })
    }

    fn program(&mut self, cl: &Api, text: &'static str) -> io::Result<Handle> {
        if let Some(program) = self.programs.get(text) {
            return Ok(program.0);
        }
        let source_code = text.as_ptr().cast::<c_char>();
        let mut error = 0;
        let handle = unsafe {
            (cl.clCreateProgramWithSource)(self.context.0, 1, &source_code, &text.len(), &mut error)
        };
        check("program", error)?;
        let program = Resource(handle, cl.clReleaseProgram);
        check("build", unsafe { (cl.clBuildProgram)(program.0, 1, &self.device, null(), None, null_mut()) })?;
        self.programs.insert(text, program);
        Ok(handle)
    }
}

fn check(stage: &str, status: i32) -> io::Result<()> {
    if status == 0 { Ok(()) } else { Err(io::Error::other(format!("AdaptiveGG OpenCL {stage}: {status}"))) }
}

fn nvidia_device(cl: &Api) -> io::Result<Handle> {
    unsafe {
        let mut count = 0;
        check("platform count", (cl.clGetPlatformIDs)(0, null_mut(), &mut count))?;
        let mut platforms = vec![null_mut(); count as usize];
        check("platform list", (cl.clGetPlatformIDs)(count, platforms.as_mut_ptr(), null_mut()))?;
        for platform in platforms {
            if (cl.clGetDeviceIDs)(platform, 4, 0, null_mut(), &mut count) != 0 || count == 0 { continue; }
            let mut devices = vec![null_mut(); count as usize];
            check("device list", (cl.clGetDeviceIDs)(platform, 4, count, devices.as_mut_ptr(), null_mut()))?;
            for device in devices {
                let mut vendor = 0u32;
                if (cl.clGetDeviceInfo)(device, 0x1001, size_of::<u32>(), (&mut vendor as *mut u32).cast(), null_mut()) == 0
                    && vendor == 0x10de { return Ok(device); }
            }
        }
    }
    Err(io::Error::other("AdaptiveGG OpenCL requires an NVIDIA GPU"))
}

pub(in crate::aurora_raw::z6iii) fn available() -> bool {
    bindings().and_then(nvidia_device).is_ok()
}

fn argument<T>(kernel: Handle, index: u32, value: &T) -> io::Result<()> {
    let cl = bindings()?;
    check("argument", unsafe { (cl.clSetKernelArg)(kernel, index, size_of::<T>(), (value as *const T).cast()) })
}

pub(super) fn prepare(raw: &[u16], width: usize, height: usize, roi: NikonCrop,
                      threshold: f32, black: f32) -> io::Result<Vec<u16>> {
    assert_eq!(raw.len(), width * height);
    assert_eq!((roi.left | roi.top | roi.width) & 1, 0);
    assert!(roi.left > 0 && roi.top > 0 && roi.left + roi.width < width && roi.top + roi.height < height);
    let input: Vec<_> = raw.iter().map(|&value| value << 2).collect();
    let mut output = input.clone();
    let text = program_source(0);
    let roi_words = [roi.left as i32, roi.top as i32, roi.width as i32, roi.height as i32];
    let stride = width as i32;
    let settings = [threshold, threshold * 0.1, 0.25, 0.1, 110000.0, black, 0.0, 0.0];
    let work = [roi.width / 2, roi.height];
    process(&input, &mut output, text, c"adaptive_gg", work, |kernel, source| {
        argument(kernel, 2, &roi_words)?;
        argument(kernel, 3, &stride)?;
        argument(kernel, 4, &stride)?;
        argument(kernel, 5, &settings)?;
        argument(kernel, 6, &source)?;
        argument(kernel, 7, &[0i32; 4])?;
        argument(kernel, 8, &source)?;
        argument(kernel, 9, &[0i32; 4])
    })?;
    Ok(output)
}

pub(super) fn stage1(input: &[u16], width: usize, height: usize, params: Stage1Params) -> io::Result<Vec<[f32; 4]>> {
    assert_eq!(input.len(), width * height);
    assert!(width > 16 && height > 16 && width.is_multiple_of(2));
    let mut output = vec![[0.0; 4]; input.len()];
    let text = program_source(1);
    let roi = [8, 8, width as i32 - 16, height as i32 - 16];
    process(input, &mut output, text, c"bayer_stage1", [(width - 16) / 2, height - 16], |kernel, _| {
        argument(kernel, 2, &roi)?;
        argument(kernel, 3, &(width as i32))?;
        argument(kernel, 4, &(width as i32 * 4))?;
        argument(kernel, 5, &[params.red_gain, params.blue_gain])?;
        argument(kernel, 6, &params.noise_floor)?;
        argument(kernel, 7, &params.black_pivot)
    })?;
    Ok(output)
}

pub(super) fn stage3(input: &[[f32; 3]], width: usize, height: usize, scale: [f32; 3], floor: f32) -> io::Result<Vec<[f32; 4]>> {
    let input = stage3_input(input, width, height, scale);
    let mut diagonal = vec![[0.0; 4]; input.len()];
    let mut output = vec![[0.0; 4]; input.len()];
    let text = program_source(2);
    let setup = |kernel, _| stage3_args(kernel, width, height, floor);
    process(&input, &mut diagonal, text, c"reconstruct_diagonal", [(width - 4) / 2, height - 4], setup)?;
    process(&diagonal, &mut output, text, c"reconstruct_green", [(width - 8) / 2, height - 8], setup)?;
    Ok(output)
}

fn stage3_input(input: &[[f32; 3]], width: usize, height: usize, scale: [f32; 3]) -> Vec<[f32; 4]> {
    assert_eq!(input.len(), width * height);
    assert!(width > 12 && height > 12 && width.is_multiple_of(2));
    input.iter().map(|pixel| [pixel[0] * scale[0], pixel[1] * scale[1], pixel[2] * scale[2], 0.0]).collect()
}

fn stage3_args(kernel: Handle, width: usize, height: usize, floor: f32) -> io::Result<()> {
    argument(kernel, 2, &(width as i32))?;
    argument(kernel, 3, &(height as i32))?;
    argument(kernel, 4, &floor)
}

pub(super) fn stage3_gg(input: &[[f32; 3]], width: usize, height: usize,
                        scale: [f32; 3], threshold: f32, noise: f32) -> io::Result<Vec<[f32; 3]>> {
    let input = stage3_input(input, width, height, scale);
    let work = [width - 12, height - 12];
    let mut output = vec![[0.0; 4]; work[0] * work[1]];
    let [red, green, blue] = scale.map(f32::recip);
    with_runtime(|cl, runtime| {
        let stage3 = runtime.program(cl, program_source(2))?;
        let gg = runtime.program(cl, program_source(3))?;
        let bytes = size_of_val(input.as_slice());
        let source = buffer(cl, runtime, 36, bytes, input.as_ptr().cast_mut().cast())?;
        let diagonal = buffer(cl, runtime, 1, bytes, null_mut())?;
        let green_plane = buffer(cl, runtime, 1, bytes, null_mut())?;
        let result = buffer(cl, runtime, 1, size_of_val(output.as_slice()), null_mut())?;
        for (entry, from, to, work) in [
            (c"reconstruct_diagonal", &source, &diagonal, [(width - 4) / 2, height - 4]),
            (c"reconstruct_green", &diagonal, &green_plane, [(width - 8) / 2, height - 8]),
        ] {
            let kernel = kernel(cl, stage3, entry, from, to)?;
            stage3_args(kernel.0, width, height, threshold)?;
            enqueue(cl, runtime, &kernel, work)?;
        }
        let kernel = kernel(cl, gg, c"gg_compensate", &green_plane, &result)?;
        argument(kernel.0, 2, &(width as i32))?;
        argument(kernel.0, 3, &6_i32)?;
        argument(kernel.0, 4, &(work[0] as i32))?;
        argument(kernel.0, 5, &0_i32)?;
        argument(kernel.0, 6, &[red, green, blue, 1.0])?;
        argument(kernel.0, 7, &[noise, threshold])?;
        enqueue(cl, runtime, &kernel, work)?;
        readback(cl, runtime, &result, &mut output)
    })?;
    Ok(output.into_iter().map(|[red, green, blue, _]| [red, green, blue]).collect())
}

pub(crate) fn anti_alias(input: &[[f32; 3]], width: usize, height: usize, params: [f32; 2]) -> io::Result<Vec<[f32; 3]>> {
    assert_eq!(input.len(), width * height);
    assert!(width > 4 && height > 4);
    let input: Vec<_> = input.iter().map(|&[r, g, b]| [r, g, b, 0.0]).collect();
    let work = [width - 4, height - 4];
    let mut output = vec![[0.0; 4]; work[0] * work[1]];
    process(&input, &mut output, program_source(4), c"anti_alias", work, |kernel, _| {
        argument(kernel, 2, &(width as i32))?;
        argument(kernel, 3, &params)
    })?;
    Ok(output.into_iter().map(|[r, g, b, _]| [r, g, b]).collect())
}

fn program_source(index: usize) -> &'static str {
    include_str!("reconstruct.cl").split("// AURORA_PROGRAM\n").nth(index).unwrap()
}

pub(in super::super) fn recover_highlights(pixels: &mut [[f32; 3]], settings: [f32; 16]) -> io::Result<()> {
    if !pixels.iter().any(|pixel| (0..3).any(|channel| pixel[channel] > settings[channel])) { return Ok(()); }
    let input: Vec<_> = pixels.iter().map(|&[red, green, blue]| [red, green, blue, 0.0]).collect();
    let mut output = vec![[0.0; 4]; pixels.len()];
    process(&input, &mut output, include_str!("highlight.cl"), c"highlight", [pixels.len(), 1], |kernel, _| {
        argument(kernel, 2, &settings)
    })?;
    for (pixel, [red, green, blue, _]) in pixels.iter_mut().zip(output) { *pixel = [red, green, blue]; }
    Ok(())
}

pub(in super::super) fn export_color(pixels: &[[f32; 3]], film: &[[f32; 3]], table: &[[u16; 3]]) -> io::Result<Vec<[u16; 3]>> {
    let input: Vec<_> = pixels.iter().map(|&[red, green, blue]| [red, green, blue, 0.0]).collect();
    let mut output = vec![[0; 3]; pixels.len()];
    let cl = bindings()?;
    process_with_resources(&input, &mut output, include_str!("export_color.cl"), c"export_color", [pixels.len(), 1],
        |kernel, _, context| {
            let make_buffer = |bytes, data| {
                let mut error = 0;
                let handle = unsafe { (cl.clCreateBuffer)(context, 36, bytes, data, &mut error) };
                check("color input buffer", error)?;
                Ok::<_, io::Error>(Resource(handle, cl.clReleaseMemObject))
            };
            let nodes = make_buffer(size_of_val(film), film.as_ptr().cast_mut().cast())?;
            let grid = make_buffer(size_of_val(table), table.as_ptr().cast_mut().cast())?;
            argument(kernel, 2, &nodes.0)?;
            argument(kernel, 3, &(film.len() as i32))?;
            argument(kernel, 4, &grid.0)?;
            Ok((nodes, grid))
        })?;
    Ok(output)
}

fn process<S, T>(input: &[S], output: &mut [T], text: &'static str, entry: &std::ffi::CStr,
              work: [usize; 2], setup: impl FnOnce(Handle, Handle) -> io::Result<()>) -> io::Result<()> {
    process_with_resources(input, output, text, entry, work, |kernel, source, _| setup(kernel, source))
}

fn process_with_resources<S, T, R>(input: &[S], output: &mut [T], text: &'static str, entry: &std::ffi::CStr,
              work: [usize; 2], setup: impl FnOnce(Handle, Handle, Handle) -> io::Result<R>) -> io::Result<()> {
    with_runtime(|cl, runtime| {
        let program = runtime.program(cl, text)?;
        let source = buffer(cl, runtime, 36, size_of_val(input), input.as_ptr().cast_mut().cast())?;
        let result = buffer(cl, runtime, 33, size_of_val(output), output.as_mut_ptr().cast())?;
        let kernel = kernel(cl, program, entry, &source, &result)?;
        let _resources = setup(kernel.0, source.0, runtime.context.0)?;
        enqueue(cl, runtime, &kernel, work)?;
        readback(cl, runtime, &result, output)
    })
}

fn with_runtime<T>(run: impl FnOnce(&Api, &mut Runtime) -> io::Result<T>) -> io::Result<T> {
    let cl = bindings()?;
    RUNTIME.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(Runtime::new(cl)?);
        }
        let runtime = slot.as_mut().expect("OpenCL runtime initialized");
        let result = run(cl, runtime);
        if result.is_err() {
            *slot = None;
        }
        result
    })
}

fn buffer(cl: &Api, runtime: &Runtime, flags: u64, bytes: usize, host: Handle) -> io::Result<Resource> {
    let mut error = 0;
    let handle = unsafe { (cl.clCreateBuffer)(runtime.context.0, flags, bytes, host, &mut error) };
    check("buffer", error)?;
    Ok(Resource(handle, cl.clReleaseMemObject))
}

fn kernel(cl: &Api, program: Handle, entry: &std::ffi::CStr, source: &Resource, result: &Resource) -> io::Result<Resource> {
    let mut error = 0;
    let handle = unsafe { (cl.clCreateKernel)(program, entry.as_ptr(), &mut error) };
    check("kernel", error)?;
    let kernel = Resource(handle, cl.clReleaseKernel);
    argument(kernel.0, 0, &source.0)?;
    argument(kernel.0, 1, &result.0)?;
    Ok(kernel)
}

fn enqueue(cl: &Api, runtime: &Runtime, kernel: &Resource, work: [usize; 2]) -> io::Result<()> {
    unsafe {
        check("enqueue", (cl.clEnqueueNDRangeKernel)(runtime.queue.0, kernel.0, 2, null(), work.as_ptr(), null(), 0, null(), null_mut()))
    }
}

fn readback<T>(cl: &Api, runtime: &Runtime, result: &Resource, output: &mut [T]) -> io::Result<()> {
    unsafe {
        check("readback", (cl.clEnqueueReadBuffer)(runtime.queue.0, result.0, 1, 0, size_of_val(output), output.as_mut_ptr().cast(), 0, null(), null_mut()))
    }
}
