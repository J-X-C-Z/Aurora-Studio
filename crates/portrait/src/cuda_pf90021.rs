//! PF90021 `LipStickLutMakeup` on NVIDIA through the shared portrait context.
//!
//! The host decodes the vendor 512x512 atlas into a typed 64^3 float volume.
//! Positive and negative directions select different volumes; neutral bypasses
//! the GPU. A non-neutral launch has two passes: full LUT transform into an
//! RGBA32F intermediate, then the separate masked base composite. Source alpha
//! is copied through both passes.
//!
//! The driver, module cache, PTX compatibility retry and process-wide context
//! live in [`crate::cuda`]. This module only validates same-job inputs, builds
//! the ABI blocks and records both real dispatches.

use crate::cuda::{CudaError, Dispatch};
use crate::{Image, Refusal};

/// Recovered PF90021 LUT helper pass.
pub const LUT_PASS_KERNEL: &str = "pf90021_lut64_pass_kernel";
/// Recovered separate masked base pass.
pub const COMPOSITE_KERNEL: &str = "pf90021_mask_composite_kernel";

pub const LUT_EDGE: usize = 64;
pub const LUT_FLOATS: usize = LUT_EDGE * LUT_EDGE * LUT_EDGE * 3;
pub const ATLAS_EDGE: usize = 512;
pub const ATLAS_RGB_BYTES: usize = ATLAS_EDGE * ATLAS_EDGE * 3;

const BLOCK: (u32, u32) = (16, 16);
const PTX: &str = include_str!("../native/cuda/pf90021_volume_lut.ptx");

/// Kernel ABI block for the LUT helper, matching `Pf90021LutPassParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LutPassParams {
    width: u32,
    height: u32,
    source_stride: u32,
    output_stride: u32,
}

/// Kernel ABI block for the masked base pass, matching
/// `Pf90021CompositeParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CompositeParams {
    width: u32,
    height: u32,
    source_stride: u32,
    effect_stride: u32,
    mask_stride: u32,
    output_stride: u32,
    u_alpha: f32,
    reserved: u32,
}

const _: () = assert!(std::mem::size_of::<LutPassParams>() == 16);
const _: () = assert!(std::mem::size_of::<CompositeParams>() == 32);

impl LutPassParams {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            source_stride: width * 4,
            output_stride: width * 16,
        }
    }
}

impl CompositeParams {
    fn new(width: u32, height: u32, u_alpha: f32) -> Self {
        Self {
            width,
            height,
            source_stride: width * 4,
            effect_stride: width * 16,
            mask_stride: width,
            output_stride: width * 4,
            u_alpha,
            reserved: 0,
        }
    }
}

/// Host-owned runtime scalar product from the static PF90021 contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuntimeScalars {
    pub degree01: f32,
    pub opacity01: f32,
    pub shelter: f32,
    pub region: f32,
}

impl RuntimeScalars {
    pub fn u_alpha(self) -> Result<f32, Refusal> {
        for (name, value) in [
            ("degree01", self.degree01),
            ("opacity01", self.opacity01),
            ("shelter", self.shelter),
            ("region", self.region),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(Refusal::ValueOutOfRange(name));
            }
        }
        let u_alpha = self.degree01 * self.opacity01 * self.shelter * self.region;
        if !u_alpha.is_finite() || !(0.0..=1.0).contains(&u_alpha) {
            return Err(Refusal::ValueOutOfRange("u_alpha"));
        }
        Ok(u_alpha)
    }
}

/// Host-selected PF90021 direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Neutral,
    Positive,
    Negative,
}

/// Logical `volume[z][y][x][rgb]` LUT. The 512x512 atlas is never accepted as a
/// kernel buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeLut64 {
    nodes: Vec<f32>,
}

impl VolumeLut64 {
    pub fn from_nodes(nodes: Vec<f32>) -> Result<Self, Refusal> {
        if nodes.len() != LUT_FLOATS {
            return Err(Refusal::ShapeMismatch("PF90021_LUT64_NODES"));
        }
        if nodes.iter().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("PF90021_LUT64_NODES"));
        }
        Ok(Self { nodes })
    }

    /// Decodes a tight 512x512 RGB8 atlas using
    /// `px=(z&7)*64+x`, `py=(z>>3)*64+y`, top row first.
    pub fn from_atlas_rgb8(atlas: &[u8]) -> Result<Self, Refusal> {
        if atlas.len() != ATLAS_RGB_BYTES {
            return Err(Refusal::ShapeMismatch("PF90021_ATLAS_RGB8"));
        }
        let mut nodes = vec![0.0; LUT_FLOATS];
        for z in 0..LUT_EDGE {
            let tile_x = z & 7;
            let tile_y = z >> 3;
            for y in 0..LUT_EDGE {
                let source_row = tile_y * LUT_EDGE + y;
                for x in 0..LUT_EDGE {
                    let source_column = tile_x * LUT_EDGE + x;
                    let source = (source_row * ATLAS_EDGE + source_column) * 3;
                    let destination = ((z * LUT_EDGE + y) * LUT_EDGE + x) * 3;
                    for channel in 0..3 {
                        nodes[destination + channel] = atlas[source + channel] as f32 / 255.0;
                    }
                }
            }
        }
        Self::from_nodes(nodes)
    }

    pub fn nodes(&self) -> &[f32] {
        &self.nodes
    }
}

/// Same-job inputs for one PF90021 call. The three optional fields are the
/// runtime values the adapter refuses to invent.
pub struct Request<'a> {
    pub source: &'a Image,
    pub direction: Direction,
    pub lut: Option<&'a VolumeLut64>,
    pub mask_r: Option<&'a [u8]>,
    pub runtime: Option<RuntimeScalars>,
}

struct Plan<'a> {
    lut: &'a VolumeLut64,
    mask_r: &'a [u8],
    u_alpha: f32,
}

fn plan<'a>(request: &Request<'a>) -> Result<Option<Plan<'a>>, Refusal> {
    request
        .source
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("source"))?;
    if request.direction == Direction::Neutral {
        return Ok(None);
    }
    let (lut, missing_lut) = match request.direction {
        Direction::Positive => (request.lut, "PF90021_POSITIVE_LUT64"),
        Direction::Negative => (request.lut, "PF90021_NEGATIVE_LUT64"),
        Direction::Neutral => unreachable!(),
    };
    let lut = lut.ok_or(Refusal::MissingInput(missing_lut))?;
    let mask_r = request
        .mask_r
        .ok_or(Refusal::MissingInput("PF90021_MASK_R"))?;
    let runtime = request
        .runtime
        .ok_or(Refusal::MissingInput("PF90021_RUNTIME_SCALARS"))?;
    if mask_r.len() != request.source.pixel_count() {
        return Err(Refusal::ShapeMismatch("PF90021_MASK_R"));
    }
    let u_alpha = runtime.u_alpha()?;
    Ok(Some(Plan {
        lut,
        mask_r,
        u_alpha,
    }))
}

/// Evidence for both kernels in the recovered two-pass order.
#[derive(Clone, Debug, PartialEq)]
pub struct Dispatches {
    pub lut: Dispatch,
    pub composite: Dispatch,
}

/// Result of a call. Neutral is a real no-launch bypass.
#[derive(Clone, Debug, PartialEq)]
pub enum Run {
    Bypassed(Image),
    Executed { output: Image, dispatch: Dispatches },
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn sample_volume(volume: &[f32], rgb: [f32; 3]) -> [f32; 3] {
    let scaled = rgb.map(|value| value.clamp(0.0, 1.0) * 63.0);
    let base = scaled.map(|value| value.floor() as usize);
    let next = base.map(|value| (value + 1).min(LUT_EDGE - 1));
    let frac = scaled.map(|value| value - value.floor());
    let node = |x: usize, y: usize, z: usize, channel: usize| {
        volume[((z * LUT_EDGE + y) * LUT_EDGE + x) * 3 + channel]
    };
    std::array::from_fn(|channel| {
        let c00 = lerp(
            node(base[0], base[1], base[2], channel),
            node(next[0], base[1], base[2], channel),
            frac[0],
        );
        let c01 = lerp(
            node(base[0], next[1], base[2], channel),
            node(next[0], next[1], base[2], channel),
            frac[0],
        );
        let c10 = lerp(
            node(base[0], base[1], next[2], channel),
            node(next[0], base[1], next[2], channel),
            frac[0],
        );
        let c11 = lerp(
            node(base[0], next[1], next[2], channel),
            node(next[0], next[1], next[2], channel),
            frac[0],
        );
        let c0 = lerp(c00, c01, frac[1]);
        let c1 = lerp(c10, c11, frac[1]);
        lerp(c0, c1, frac[2]).clamp(0.0, 1.0)
    })
}

/// CPU oracle for the same two-pass contract used by the CUDA adapter.
pub fn apply_cpu(
    source: &Image,
    volume: &VolumeLut64,
    mask_r: &[u8],
    runtime: RuntimeScalars,
) -> Result<Image, Refusal> {
    source.validate()?;
    if mask_r.len() != source.pixel_count() {
        return Err(Refusal::ShapeMismatch("PF90021_MASK_R"));
    }
    let u_alpha = runtime.u_alpha()?;
    let mut output = source.clone();
    for (pixel, mask) in mask_r.iter().enumerate() {
        let at = pixel * 4;
        let rgb = [
            source.pixels[at] as f32 / 255.0,
            source.pixels[at + 1] as f32 / 255.0,
            source.pixels[at + 2] as f32 / 255.0,
        ];
        let mapped = sample_volume(volume.nodes(), rgb);
        let alpha = u_alpha.clamp(0.0, 1.0) * (*mask as f32 / 255.0);
        for channel in 0..3 {
            let value = lerp(rgb[channel], mapped[channel], alpha).clamp(0.0, 1.0);
            output.pixels[at + channel] = (value * 255.0 + 0.5).floor() as u8;
        }
        output.pixels[at + 3] = source.pixels[at + 3];
    }
    Ok(output)
}

fn as_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

#[cfg(test)]
fn as_u8_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

/// PF90021 session using the crate's process-wide NVIDIA domain.
#[cfg(windows)]
pub struct Pf90021(crate::cuda::Nvidia);

#[cfg(windows)]
impl Pf90021 {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self(crate::cuda::Nvidia::open_ptx(PTX)?))
    }

    pub fn suspend(&mut self) -> Result<(), CudaError> {
        self.0.suspend()
    }

    pub fn resume(&mut self) -> Result<(), CudaError> {
        self.0.resume()
    }

    /// Runs the host-selected direction through both recovered passes.
    pub fn run(&self, request: Request<'_>) -> Result<Run, CudaError> {
        let Some(plan) = plan(&request)? else {
            return Ok(Run::Bypassed(request.source.clone()));
        };

        let source = request.source;
        let pixels = source.pixel_count();
        let grid = (
            source.width.div_ceil(BLOCK.0).max(1),
            source.height.div_ceil(BLOCK.1).max(1),
        );
        let source_bytes = source.pixels.len();
        let volume_bytes = as_bytes(plan.lut.nodes()).len();
        let effect_bytes = pixels * 16;
        let mask_bytes = plan.mask_r.len();
        self.0.with_reusable_buffers(
            "pf90021.volume-composite",
            &[
                source_bytes,
                volume_bytes,
                effect_bytes,
                mask_bytes,
                source_bytes,
            ],
            |buffers| {
                buffers.upload(0, &source.pixels)?;
                buffers.upload(1, as_bytes(plan.lut.nodes()))?;
                buffers.upload(3, plan.mask_r)?;

                let lut_params = LutPassParams::new(source.width, source.height);
                let mut source_arg = buffers.handle(0);
                let mut volume_arg = buffers.handle(1);
                let mut effect_arg = buffers.handle(2);
                let mut lut_params_arg = lut_params;
                let mut lut_args: [*mut std::ffi::c_void; 4] = [
                    &mut source_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut volume_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut effect_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut lut_params_arg as *mut LutPassParams as *mut std::ffi::c_void,
                ];
                let lut_dispatch = buffers.launch(LUT_PASS_KERNEL, grid, BLOCK, &mut lut_args)?;

                let composite_params =
                    CompositeParams::new(source.width, source.height, plan.u_alpha);
                let mut mask_arg = buffers.handle(3);
                let mut output_arg = buffers.handle(4);
                let mut composite_params_arg = composite_params;
                let mut composite_args: [*mut std::ffi::c_void; 5] = [
                    &mut source_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut effect_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut mask_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut output_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut composite_params_arg as *mut CompositeParams as *mut std::ffi::c_void,
                ];
                let composite_dispatch =
                    buffers.launch(COMPOSITE_KERNEL, grid, BLOCK, &mut composite_args)?;

                let mut output = vec![0u8; source_bytes];
                buffers.download(4, &mut output)?;
                buffers.mark_consumed();
                Ok(Run::Executed {
                    output: Image {
                        width: source.width,
                        height: source.height,
                        pixels: output,
                    },
                    dispatch: Dispatches {
                        lut: lut_dispatch,
                        composite: composite_dispatch,
                    },
                })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set_node(volume: &mut [f32], x: usize, y: usize, z: usize, rgb: [f32; 3]) {
        let at = ((z * LUT_EDGE + y) * LUT_EDGE + x) * 3;
        volume[at..at + 3].copy_from_slice(&rgb);
    }

    fn set_tap(volume: &mut [f32], source: [u8; 3], mapped: [f32; 3]) {
        let scaled = [
            source[0] as f32 / 255.0 * 63.0,
            source[1] as f32 / 255.0 * 63.0,
            source[2] as f32 / 255.0 * 63.0,
        ];
        let base = scaled.map(|value| value.floor() as usize);
        let next = base.map(|value| (value + 1).min(LUT_EDGE - 1));
        for z in [base[2], next[2]] {
            for y in [base[1], next[1]] {
                for x in [base[0], next[0]] {
                    set_node(volume, x, y, z, mapped);
                }
            }
        }
    }

    fn fixture_volumes() -> (VolumeLut64, VolumeLut64) {
        let mut positive = vec![0.5; LUT_FLOATS];
        let mut negative = vec![0.5; LUT_FLOATS];
        set_tap(
            &mut positive,
            [128, 64, 200],
            [0.575_574_9, 0.232_341_42, 0.782_929_7],
        );
        set_tap(
            &mut positive,
            [32, 160, 96],
            [0.289_457_92, 0.608_304_5, 0.363_091_17],
        );
        set_tap(&mut positive, [255, 255, 255], [1.0, 1.0, 1.0]);
        set_tap(
            &mut negative,
            [128, 64, 200],
            [0.542_995_04, 0.251_949_28, 0.782_659],
        );
        (
            VolumeLut64::from_nodes(positive).unwrap(),
            VolumeLut64::from_nodes(negative).unwrap(),
        )
    }

    #[test]
    fn abi_blocks_match_the_c_header() {
        assert_eq!(std::mem::size_of::<LutPassParams>(), 16);
        assert_eq!(std::mem::size_of::<CompositeParams>(), 32);
        assert_eq!(as_u8_bytes(&LutPassParams::new(3, 2)).len(), 16);
        assert_eq!(as_u8_bytes(&CompositeParams::new(3, 2, 0.25)).len(), 32);
    }

    #[test]
    fn atlas_decode_uses_the_recovered_tile_origin() {
        let mut atlas = vec![0u8; ATLAS_RGB_BYTES];
        let (x, y, z) = (11, 7, 9);
        let at = (((z >> 3) * LUT_EDGE + y) * ATLAS_EDGE + (z & 7) * LUT_EDGE + x) * 3;
        atlas[at..at + 3].copy_from_slice(&[10, 20, 30]);
        let volume = VolumeLut64::from_atlas_rgb8(&atlas).unwrap();
        let node = ((z * LUT_EDGE + y) * LUT_EDGE + x) * 3;
        assert_eq!(
            &volume.nodes()[node..node + 3],
            &[10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0]
        );
    }

    #[test]
    fn missing_runtime_inputs_are_typed_refusals() {
        let source = Image::solid(1, 1, [128, 64, 200, 190]);
        let volume = VolumeLut64::from_nodes(vec![0.5; LUT_FLOATS]).unwrap();
        let mask = [128u8];
        let runtime = RuntimeScalars {
            degree01: 0.5,
            opacity01: 0.8,
            shelter: 1.0,
            region: 1.0,
        };
        let request = |lut, mask_r, runtime| Request {
            source: &source,
            direction: Direction::Positive,
            lut,
            mask_r,
            runtime,
        };
        assert_eq!(
            plan(&request(None, Some(&mask), Some(runtime))).err(),
            Some(Refusal::MissingInput("PF90021_POSITIVE_LUT64"))
        );
        assert_eq!(
            plan(&request(Some(&volume), None, Some(runtime))).err(),
            Some(Refusal::MissingInput("PF90021_MASK_R"))
        );
        assert_eq!(
            plan(&request(Some(&volume), Some(&mask), None)).err(),
            Some(Refusal::MissingInput("PF90021_RUNTIME_SCALARS"))
        );
        assert!(
            plan(&Request {
                source: &source,
                direction: Direction::Neutral,
                lut: None,
                mask_r: None,
                runtime: None,
            })
            .unwrap()
            .is_none()
        );

        assert_eq!(
            plan(&Request {
                source: &source,
                direction: Direction::Positive,
                lut: Some(&volume),
                mask_r: Some(&[]),
                runtime: Some(runtime),
            })
            .err(),
            Some(Refusal::ShapeMismatch("PF90021_MASK_R"))
        );
        assert_eq!(
            plan(&Request {
                source: &source,
                direction: Direction::Positive,
                lut: Some(&volume),
                mask_r: Some(&mask),
                runtime: Some(RuntimeScalars {
                    degree01: f32::NAN,
                    ..runtime
                }),
            })
            .err(),
            Some(Refusal::ValueOutOfRange("degree01"))
        );
    }

    #[test]
    fn fixture_vectors_match_the_cpu_reference() {
        let (positive, negative) = fixture_volumes();
        let runtime = RuntimeScalars {
            degree01: 0.5,
            opacity01: 0.800_000_011_920_929,
            shelter: 1.0,
            region: 1.0,
        };
        assert_eq!(runtime.u_alpha().unwrap().to_bits(), 0x3ecc_cccd);
        let cases = [
            ([128, 64, 200, 190], 128, &positive, [132, 63, 200, 190]),
            ([128, 64, 200, 190], 128, &negative, [130, 64, 200, 190]),
            ([32, 160, 96, 255], 224, &positive, [47, 158, 95, 255]),
            ([128, 64, 200, 190], 0, &positive, [128, 64, 200, 190]),
            ([255, 255, 255, 255], 255, &positive, [255, 255, 255, 255]),
        ];
        for (rgba, mask, volume, expected) in cases {
            let source = Image::solid(1, 1, rgba);
            let actual = apply_cpu(&source, volume, &[mask], runtime).unwrap();
            assert_eq!(actual.pixels, expected);
        }
    }

    #[cfg(windows)]
    fn gpu() -> Option<Pf90021> {
        match Pf90021::open() {
            Ok(gpu) => Some(gpu),
            Err(error) => {
                if std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_some() {
                    panic!("CUDA required but unavailable: {error:?}");
                }
                eprintln!("SKIPPED (no NVIDIA execution path): {error:?}");
                None
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn real_gpu_matches_cpu_for_volume_lut_vectors() {
        let Some(gpu) = gpu() else { return };
        let (positive, negative) = fixture_volumes();
        let runtime = RuntimeScalars {
            degree01: 0.5,
            opacity01: 0.800_000_011_920_929,
            shelter: 1.0,
            region: 1.0,
        };
        let cases = [
            (
                Direction::Positive,
                &positive,
                [128, 64, 200, 190],
                128,
                [132, 63, 200, 190],
            ),
            (
                Direction::Negative,
                &negative,
                [128, 64, 200, 190],
                128,
                [130, 64, 200, 190],
            ),
            (
                Direction::Positive,
                &positive,
                [32, 160, 96, 255],
                224,
                [47, 158, 95, 255],
            ),
            (
                Direction::Positive,
                &positive,
                [128, 64, 200, 190],
                0,
                [128, 64, 200, 190],
            ),
            (
                Direction::Positive,
                &positive,
                [255, 255, 255, 255],
                255,
                [255, 255, 255, 255],
            ),
        ];
        for (direction, volume, rgba, mask, expected) in cases {
            let source = Image::solid(1, 1, rgba);
            let cpu = apply_cpu(&source, volume, &[mask], runtime).unwrap();
            let run = gpu
                .run(Request {
                    source: &source,
                    direction,
                    lut: Some(volume),
                    mask_r: Some(&[mask]),
                    runtime: Some(runtime),
                })
                .expect("real two-pass dispatch");
            let Run::Executed { output, dispatch } = run else {
                panic!("non-neutral input bypassed");
            };
            assert_eq!(dispatch.lut.kernel, LUT_PASS_KERNEL);
            assert_eq!(dispatch.composite.kernel, COMPOSITE_KERNEL);
            assert_eq!(
                output.pixels, cpu.pixels,
                "{direction:?} {rgba:?} mask={mask}"
            );
            assert_eq!(output.pixels, expected);
        }
    }

    #[cfg(windows)]
    #[test]
    fn real_gpu_matches_cpu_on_trilinear_gradient() {
        let _cache_guard = crate::cuda::test_exclusive_cache_guard();
        let Some(gpu) = gpu() else { return };
        let mut nodes = vec![0.0f32; LUT_FLOATS];
        for z in 0..LUT_EDGE {
            for y in 0..LUT_EDGE {
                for x in 0..LUT_EDGE {
                    set_node(
                        &mut nodes,
                        x,
                        y,
                        z,
                        [
                            (x + y) as f32 / 126.0,
                            (y + z) as f32 / 126.0,
                            (x + z) as f32 / 126.0,
                        ],
                    );
                }
            }
        }
        let volume = VolumeLut64::from_nodes(nodes).unwrap();
        let (width, height) = (64u32, 64u32);
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.extend_from_slice(&[
                    (x * 4) as u8,
                    (y * 4) as u8,
                    ((x + y) * 2) as u8,
                    (255 - x) as u8,
                ]);
            }
        }
        let source = Image::new(width, height, pixels).unwrap();
        let mask_r: Vec<u8> = (0..source.pixel_count())
            .map(|pixel| (pixel * 37 % 256) as u8)
            .collect();
        let runtime = RuntimeScalars {
            degree01: 0.73,
            opacity01: 0.91,
            shelter: 0.83,
            region: 0.67,
        };
        let cpu = apply_cpu(&source, &volume, &mask_r, runtime).unwrap();
        let run = gpu
            .run(Request {
                source: &source,
                direction: Direction::Positive,
                lut: Some(&volume),
                mask_r: Some(&mask_r),
                runtime: Some(runtime),
            })
            .expect("real two-pass dispatch");
        let Run::Executed { output, dispatch } = run else {
            panic!("non-neutral input bypassed");
        };
        assert_eq!(dispatch.lut.kernel, LUT_PASS_KERNEL);
        assert_eq!(dispatch.composite.kernel, COMPOSITE_KERNEL);
        assert_eq!(output.pixels, cpu.pixels);
        for (pixel, (actual, expected)) in output
            .pixels
            .chunks_exact(4)
            .zip(cpu.pixels.chunks_exact(4))
            .enumerate()
        {
            assert_eq!(actual[3], expected[3], "alpha changed at pixel {pixel}");
            assert_eq!(actual, expected, "pixel {pixel}");
        }

        let before = crate::cuda::driver_metrics();
        let second = gpu
            .run(Request {
                source: &source,
                direction: Direction::Positive,
                lut: Some(&volume),
                mask_r: Some(&mask_r),
                runtime: Some(runtime),
            })
            .expect("warm two-pass dispatch");
        let after = crate::cuda::driver_metrics();
        let Run::Executed { output: warm, .. } = second else {
            panic!("warm non-neutral input bypassed");
        };
        assert_eq!(warm.pixels, cpu.pixels, "warm CPU byte mismatch");
        assert_eq!(
            after.device_allocations - before.device_allocations,
            0,
            "warm PF90021 allocations"
        );
        assert_eq!(
            after.context_syncs - before.context_syncs,
            0,
            "readback owns the PF90021 sync boundary"
        );
        eprintln!(
            "N35_PF90021_WARM alloc=0 sync=0 bytes={} dispatch=2 device={}",
            warm.pixels.len(),
            dispatch.composite.device
        );
    }

    #[cfg(windows)]
    #[test]
    fn neutral_and_missing_inputs_never_launch() {
        let Some(gpu) = gpu() else { return };
        let source = Image::solid(2, 1, [1, 2, 3, 4]);
        let run = gpu
            .run(Request {
                source: &source,
                direction: Direction::Neutral,
                lut: None,
                mask_r: None,
                runtime: None,
            })
            .unwrap();
        assert_eq!(run, Run::Bypassed(source.clone()));
        assert_eq!(
            gpu.run(Request {
                source: &source,
                direction: Direction::Positive,
                lut: None,
                mask_r: None,
                runtime: None,
            })
            .err(),
            Some(CudaError::Refused(Refusal::MissingInput(
                "PF90021_POSITIVE_LUT64"
            )))
        );
    }
}
