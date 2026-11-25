//! PF90004 fixed LUT/gamma/matrix chain on NVIDIA.
//!
//! The CPU oracle remains `kernels::radiance::apply`. This adapter reuses its
//! parameter compiler and palette compiler, then runs the two native-order
//! passes in one dispatch while preserving the intermediate RGBA8 store.
//!
//! Missing parsing/face planes or assets are refused by the stage before this
//! adapter is opened. Shape, range and finite checks are repeated here.

use crate::cuda::{CudaError, Dispatch};
use crate::kernels::radiance::{
    self, FixedAssets, GAMMA_FLOATS, LUT_A_ROWS, LUT_B_ROWS, MATRIX_FLOATS,
};
use crate::{Image, Refusal};

pub const KERNEL: &str = "pf90004_radiance_kernel";
const BLOCK: (u32, u32) = (16, 16);
const PTX: &str = include_str!("../native/cuda/pf90004_radiance.ptx");

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub mask_row_bytes: u32,
    pub parsing_row_bytes: u32,
    pub face_row_bytes: u32,
    pub use_brightness: u32,
    pub use_contrast: u32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 32);

#[derive(Clone, Copy)]
pub struct Request<'a> {
    pub source: &'a Image,
    pub skin_r: &'a [u8],
    pub parsing_rgba: &'a [u8],
    pub face_rgba: &'a [u8],
    pub ui: f32,
    pub value_mean: f32,
    pub assets: &'a FixedAssets<'a>,
}

struct Plan {
    degree: i32,
    brightness: Option<Vec<u8>>,
    contrast: Option<Vec<u8>>,
    hue_maps: Vec<f32>,
}

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

fn check(request: &Request<'_>) -> Result<Plan, Refusal> {
    let source = request.source;
    source
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("source"))?;
    let pixels = source.pixel_count();
    if request.skin_r.len() != pixels {
        return Err(Refusal::ShapeMismatch("skin_r"));
    }
    if request.parsing_rgba.len() != pixels.saturating_mul(4) {
        return Err(Refusal::ShapeMismatch("parsing_rgba"));
    }
    if request.face_rgba.len() != pixels.saturating_mul(4) {
        return Err(Refusal::ShapeMismatch("face_rgba"));
    }
    if request.assets.lut_a.len() != LUT_A_ROWS * 256 {
        return Err(Refusal::ShapeMismatch("lut_a"));
    }
    if request.assets.lut_b.len() != LUT_B_ROWS * 256 {
        return Err(Refusal::ShapeMismatch("lut_b"));
    }
    if request.assets.gamma.len() != GAMMA_FLOATS {
        return Err(Refusal::ShapeMismatch("gamma"));
    }
    if request.assets.matrices.len() != MATRIX_FLOATS {
        return Err(Refusal::ShapeMismatch("matrices"));
    }
    if request
        .assets
        .gamma
        .iter()
        .chain(request.assets.matrices)
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("assets"));
    }
    if !request.value_mean.is_finite() || !(0.0..=255.0).contains(&request.value_mean) {
        return Err(Refusal::ValueOutOfRange("value_mean"));
    }

    let degree = radiance::ui_to_native_degree(request.ui)?;
    if degree == 0 {
        return Ok(Plan {
            degree,
            brightness: None,
            contrast: None,
            hue_maps: Vec::new(),
        });
    }
    let parameters = radiance::compile_parameters(degree, request.value_mean)?;
    let (brightness_degree, contrast_degree) = radiance::lut_degrees(parameters[0], parameters[1]);
    Ok(Plan {
        degree,
        brightness: radiance::select_lut_row(request.assets.lut_a, -150, brightness_degree)?,
        contrast: radiance::select_lut_row(request.assets.lut_b, -50, contrast_degree)?,
        hue_maps: radiance::hue_maps_for(&parameters),
    })
}

#[cfg(windows)]
pub struct Radiance(crate::cuda::Nvidia);

#[cfg(windows)]
impl Radiance {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self(crate::cuda::Nvidia::open_ptx(PTX)?))
    }

    pub fn apply(&self, request: Request<'_>) -> Result<(Image, Option<Dispatch>), CudaError> {
        let plan = check(&request)?;
        if plan.degree == 0 {
            return Ok((request.source.clone(), None));
        }

        let identity: Vec<u8> = (0..=255).collect();
        let brightness = plan.brightness.as_deref().unwrap_or(&identity);
        let contrast = plan.contrast.as_deref().unwrap_or(&identity);
        let uniforms = Uniforms {
            width: request.source.width,
            height: request.source.height,
            stride: request.source.width * 4,
            mask_row_bytes: request.source.width,
            parsing_row_bytes: request.source.width * 4,
            face_row_bytes: request.source.width * 4,
            use_brightness: u32::from(plan.brightness.is_some()),
            use_contrast: u32::from(plan.contrast.is_some()),
        };
        let grid = (
            request.source.width.div_ceil(BLOCK.0).max(1),
            request.source.height.div_ceil(BLOCK.1).max(1),
        );
        let (pixels, dispatch) = self.0.run_u8(
            KERNEL,
            grid,
            &[
                &request.source.pixels,
                request.skin_r,
                request.parsing_rgba,
                request.face_rgba,
                brightness,
                contrast,
                f32_bytes(&plan.hue_maps),
                f32_bytes(request.assets.gamma),
                f32_bytes(request.assets.matrices),
            ],
            as_bytes(&uniforms),
        )?;
        let image = Image::new(request.source.width, request.source.height, pixels)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, Some(dispatch)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::radiance::{self, MaskPlanes};

    fn assets() -> FixedAssets<'static> {
        use std::sync::OnceLock;

        /// `include_bytes!` has alignment 1, so casting its pointer to `f32` is
        /// undefined behaviour (and aborts under the alignment check on
        /// aarch64). Decode into an aligned, cached buffer instead; the assets
        /// are native-endian `f32` arrays.
        fn floats(bytes: &'static [u8], slot: &'static OnceLock<Vec<f32>>) -> &'static [f32] {
            slot.get_or_init(|| {
                bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            })
        }

        static GAMMA: OnceLock<Vec<f32>> = OnceLock::new();
        static MATRICES: OnceLock<Vec<f32>> = OnceLock::new();
        FixedAssets {
            lut_a: include_bytes!("../../../assets/portrait/radiance/native-lut-a.bin"),
            lut_b: include_bytes!("../../../assets/portrait/radiance/native-lut-b.bin"),
            gamma: floats(
                include_bytes!("../../../assets/portrait/radiance/native-gamma-rgba32f.bin"),
                &GAMMA,
            ),
            matrices: floats(
                include_bytes!("../../../assets/portrait/radiance/native-color-matrices.bin"),
                &MATRICES,
            ),
        }
    }

    fn fixture(width: u32, height: u32) -> (Image, Vec<u8>, Vec<u8>, Vec<u8>) {
        let pixels = width as usize * height as usize;
        let mut source = Vec::with_capacity(pixels * 4);
        let mut skin = Vec::with_capacity(pixels);
        let mut parsing = Vec::with_capacity(pixels * 4);
        let mut face = Vec::with_capacity(pixels * 4);
        for index in 0..pixels {
            let x = index % width as usize;
            let y = index / width as usize;
            source.extend_from_slice(&[
                (70 + x * 7 + y * 3) as u8,
                (95 + x * 5 + y * 9) as u8,
                (120 + x * 11 + y * 2) as u8,
                (31 + index * 17) as u8,
            ]);
            skin.push((40 + (x * 19 + y * 23) % 216) as u8);
            parsing.extend_from_slice(&[
                (index * 13) as u8,
                (32 + index * 11) as u8,
                (64 + index * 5) as u8,
                (16 + index * 7) as u8,
            ]);
            face.extend_from_slice(&[
                (index * 3) as u8,
                (48 + index * 9) as u8,
                (index * 21) as u8,
                (80 + index * 5) as u8,
            ]);
        }
        (
            Image::new(width, height, source).unwrap(),
            skin,
            parsing,
            face,
        )
    }

    fn oracle(
        source: &Image,
        skin: &[u8],
        parsing: &[u8],
        face: &[u8],
        ui: f32,
        value_mean: f32,
        assets: &FixedAssets<'_>,
    ) -> Image {
        radiance::apply(
            source,
            &MaskPlanes {
                skin,
                parsing_rgba: parsing,
                face_rgba: face,
            },
            ui,
            value_mean,
            assets,
        )
        .unwrap()
        .image
    }

    #[test]
    fn abi_and_fixed_asset_shapes_are_exact() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 32);
        assert_eq!(std::mem::align_of::<Uniforms>(), 4);
        assert!(PTX.contains(&format!(".entry {KERNEL}(")));
        assert!(
            check(&Request {
                source: &Image::solid(1, 1, [1, 2, 3, 4]),
                skin_r: &[255],
                parsing_rgba: &[0; 4],
                face_rgba: &[0; 4],
                ui: 50.0,
                value_mean: 100.0,
                assets: &assets(),
            })
            .is_ok()
        );
    }

    #[test]
    fn missing_or_wrong_size_inputs_are_typed_refusals() {
        let source = Image::solid(2, 2, [10, 20, 30, 40]);
        let all_assets = assets();
        let base = Request {
            source: &source,
            skin_r: &[255; 4],
            parsing_rgba: &[0; 16],
            face_rgba: &[0; 16],
            ui: 50.0,
            value_mean: 100.0,
            assets: &all_assets,
        };
        assert_eq!(
            check(&Request {
                parsing_rgba: &[0; 15],
                ..base
            })
            .err()
            .unwrap(),
            Refusal::ShapeMismatch("parsing_rgba")
        );
        let base = Request {
            source: &source,
            skin_r: &[255; 4],
            parsing_rgba: &[0; 16],
            face_rgba: &[0; 16],
            ui: 50.0,
            value_mean: 100.0,
            assets: &all_assets,
        };
        assert_eq!(
            check(&Request {
                face_rgba: &[0; 15],
                ..base
            })
            .err()
            .unwrap(),
            Refusal::ShapeMismatch("face_rgba")
        );
    }

    #[test]
    fn value_mean_domain_and_zero_degree_rounding_are_checked() {
        let source = Image::solid(1, 1, [1, 2, 3, 4]);
        let all_assets = assets();
        let base = Request {
            source: &source,
            skin_r: &[255],
            parsing_rgba: &[0; 4],
            face_rgba: &[0; 4],
            ui: 50.0,
            value_mean: 100.0,
            assets: &all_assets,
        };
        for value_mean in [255.0001, f32::NAN, f32::INFINITY, -0.001] {
            assert_eq!(
                check(&Request {
                    value_mean,
                    ..base
                })
                .err()
                .unwrap(),
                Refusal::ValueOutOfRange("value_mean")
            );
        }
        let plan = check(&Request {
            ui: 0.001,
            ..base
        })
        .unwrap();
        assert_eq!(plan.degree, 0);
        assert!(plan.hue_maps.is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn real_cuda_matches_the_independent_cpu_oracle() {
        let (source, skin, parsing, face) = fixture(35, 27);
        let fixed = assets();
        let gpu = match Radiance::open() {
            Ok(gpu) => gpu,
            Err(error) => {
                if std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_some() {
                    panic!("CUDA is required but unavailable: {error:?}");
                }
                eprintln!("CUDA unavailable; parity test skipped: {error:?}");
                return;
            }
        };
        for (ui, value_mean) in [
            (1.0f32, 0.0f32),
            (17.0, 64.5),
            (47.0, 143.25),
            (99.999, 255.0),
            (100.0, 255.0),
        ] {
            let expected = oracle(&source, &skin, &parsing, &face, ui, value_mean, &fixed);
            let (actual, dispatch) = gpu
                .apply(Request {
                    source: &source,
                    skin_r: &skin,
                    parsing_rgba: &parsing,
                    face_rgba: &face,
                    ui,
                    value_mean,
                    assets: &fixed,
                })
                .unwrap();
            let dispatch = dispatch.expect("non-neutral PF90004 dispatch");
            assert_eq!(dispatch.kernel, KERNEL);
            assert_eq!(actual.pixels, expected.pixels, "ui={ui}");
            eprintln!(
                "PF90004 ui={ui} valueMean={value_mean} bytes={} mismatch=0 device={} cc={}.{} driver={} kernel={}",
                actual.pixels.len(),
                dispatch.device,
                dispatch.compute_cap.0,
                dispatch.compute_cap.1,
                dispatch.driver_version,
                dispatch.kernel,
            );
        }
        let zero_skin = vec![0u8; source.pixel_count()];
        let zero_rgba = vec![0u8; source.pixel_count() * 4];
        let expected = oracle(
            &source,
            &zero_skin,
            &zero_rgba,
            &zero_rgba,
            47.0,
            143.25,
            &fixed,
        );
        let (actual, dispatch) = gpu
            .apply(Request {
                source: &source,
                skin_r: &zero_skin,
                parsing_rgba: &zero_rgba,
                face_rgba: &zero_rgba,
                ui: 47.0,
                value_mean: 143.25,
                assets: &fixed,
            })
            .unwrap();
        assert_eq!(actual.pixels, expected.pixels);
        assert!(dispatch.is_some());
        let (neutral, dispatch) = gpu
            .apply(Request {
                source: &source,
                skin_r: &skin,
                parsing_rgba: &parsing,
                face_rgba: &face,
                ui: 0.0,
                value_mean: 143.25,
                assets: &fixed,
            })
            .unwrap();
        assert_eq!(neutral.pixels, source.pixels);
        assert!(dispatch.is_none());
    }

    #[cfg(windows)]
    #[test]
    fn real_cuda_matches_the_cpu_oracle_across_rgb8_domain() {
        let width = 256u32;
        let height = 4u32;
        let pixels = (width * height) as usize;
        let mut source = Vec::with_capacity(pixels * 4);
        for y in 0..height {
            for x in 0..width {
                let value = x as u8;
                let inverse = !value;
                let rgba = match y {
                    0 => [value, value, value, 255],
                    1 => [value, inverse, 64, 192],
                    2 => [0, value, inverse, 96],
                    _ => [inverse, 0, value, 32],
                };
                source.extend_from_slice(&rgba);
            }
        }
        let source = Image::new(width, height, source).unwrap();
        let skin: Vec<u8> = (0..pixels).map(|i| (i * 37 % 256) as u8).collect();
        let parsing: Vec<u8> = (0..pixels * 4)
            .map(|i| ((i * 17 + 3) % 256) as u8)
            .collect();
        let face: Vec<u8> = (0..pixels * 4)
            .map(|i| ((i * 29 + 11) % 256) as u8)
            .collect();
        let fixed = assets();
        let gpu = match Radiance::open() {
            Ok(gpu) => gpu,
            Err(error) => {
                if std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_some() {
                    panic!("CUDA is required but unavailable: {error:?}");
                }
                eprintln!("CUDA unavailable; RGB8 domain test skipped: {error:?}");
                return;
            }
        };
        let expected = oracle(
            &source,
            &skin,
            &parsing,
            &face,
            47.0,
            143.25,
            &fixed,
        );
        let (actual, dispatch) = gpu
            .apply(Request {
                source: &source,
                skin_r: &skin,
                parsing_rgba: &parsing,
                face_rgba: &face,
                ui: 47.0,
                value_mean: 143.25,
                assets: &fixed,
            })
            .unwrap();
        assert!(dispatch.is_some());
        assert_eq!(actual.pixels, expected.pixels);
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "GPU timing; run with --ignored and PORTRAIT_CUDA_REQUIRED=1"]
    fn pf90004_adapter_timing_p50_p95() {
        fn percentile(values: &mut [u128], percent: usize) -> u128 {
            values.sort_unstable();
            let index = ((values.len() - 1) * percent).div_ceil(100);
            values[index]
        }

        let (source, skin, parsing, face) = fixture(256, 192);
        let fixed = assets();
        let gpu = Radiance::open().expect("CUDA required for timing");
        for _ in 0..4 {
            let _ = gpu
                .apply(Request {
                    source: &source,
                    skin_r: &skin,
                    parsing_rgba: &parsing,
                    face_rgba: &face,
                    ui: 47.0,
                    value_mean: 143.25,
                    assets: &fixed,
                })
                .unwrap();
        }

        let mut gpu_us = Vec::with_capacity(31);
        for _ in 0..31 {
            let start = std::time::Instant::now();
            let _ = gpu
                .apply(Request {
                    source: &source,
                    skin_r: &skin,
                    parsing_rgba: &parsing,
                    face_rgba: &face,
                    ui: 47.0,
                    value_mean: 143.25,
                    assets: &fixed,
                })
                .unwrap();
            gpu_us.push(start.elapsed().as_micros());
        }

        let mut cpu_us = Vec::with_capacity(11);
        for _ in 0..11 {
            let start = std::time::Instant::now();
            let _ = oracle(&source, &skin, &parsing, &face, 47.0, 143.25, &fixed);
            cpu_us.push(start.elapsed().as_micros());
        }
        eprintln!(
            "PF90004_TIMING width={} height={} gpu_p50_us={} gpu_p95_us={} cpu_p50_us={} cpu_p95_us={}",
            source.width,
            source.height,
            percentile(&mut gpu_us, 50),
            percentile(&mut gpu_us, 95),
            percentile(&mut cpu_us, 50),
            percentile(&mut cpu_us, 95),
        );
    }
}
