//! PF30030 hand blue veins on NVIDIA.
//!
//! The adapter consumes only same-job producer outputs: a hand-vein mask and
//! the model-authored correction RGB image. It never segments a hand or invents
//! correction bytes. `ui` is a `0..100` enable gate only; the recovered client
//! pixel path does not multiply by `ui / 100`.
//!
//! The CPU reference performs the recovered separable box feather. CUDA then
//! runs the integer pixel law inside the explicit ROI:
//! `out = clamp_u8(src + round_away((corr - 80) * mask / 255))`.

use crate::kernels::hair_body;
use crate::{Image, Refusal};

/// Recovered kernel entry point.
pub const KERNEL: &str = "pf30030_blue_vein_kernel";
/// The vendor correction image's neutral byte.
pub const CORRECTION_NEUTRAL: i32 = 80;
const BLOCK: u32 = 16;

const PTX: &str = include_str!("../native/cuda/pf30030_blue_vein.ptx");

/// 32-byte kernel argument block, matching `Pf30030BlueVeinUniforms`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub roi_x: u32,
    pub roi_y: u32,
    pub roi_width: u32,
    pub roi_height: u32,
    pub reserved: u32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 32);

/// Explicit same-job inputs. `None` means the external producer did not supply
/// the surface; an enabled effect refuses it instead of synthesising a mask or
/// correction image.
#[derive(Clone, Copy, Debug)]
pub struct Inputs<'a> {
    pub ui: i32,
    pub vein_mask: Option<&'a [u8]>,
    pub correction_rgb: Option<&'a [u8]>,
    pub feather_radius: u32,
    pub roi: (u32, u32, u32, u32),
}

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn resolve<'a>(source: &Image, inputs: &Inputs<'a>) -> Result<(&'a [u8], &'a [u8]), Refusal> {
    source.validate()?;
    let count = source.pixel_count();
    if !(0..=100).contains(&inputs.ui) {
        return Err(Refusal::ValueOutOfRange("PF30030_UI"));
    }
    let vein_mask = inputs
        .vein_mask
        .ok_or(Refusal::MissingInput("PF30030_HAND_VEIN_MASK"))?;
    if vein_mask.len() != count {
        return Err(Refusal::ShapeMismatch("PF30030_HAND_VEIN_MASK"));
    }
    let correction_rgb = inputs
        .correction_rgb
        .ok_or(Refusal::MissingInput("PF30030_MODEL_CORRECTION_RGB"))?;
    if correction_rgb.len() != count * 3 {
        return Err(Refusal::ShapeMismatch("PF30030_MODEL_CORRECTION_RGB"));
    }
    if !hair_body::roi_valid(inputs.roi, source.width, source.height) {
        return Err(Refusal::ShapeMismatch("PF30030_ROI"));
    }
    Ok((vein_mask, correction_rgb))
}

/// CPU oracle for the same explicit input contract as the CUDA adapter.
pub fn reference(source: &Image, inputs: &Inputs<'_>) -> Result<Image, Refusal> {
    let (vein_mask, correction_rgb) = resolve(source, inputs)?;
    hair_body::hand_blue_vein(
        source,
        inputs.ui,
        vein_mask,
        correction_rgb,
        inputs.feather_radius,
        inputs.roi,
    )
}

#[cfg(windows)]
pub struct HandBlueVein {
    gpu: crate::cuda::Nvidia,
}

#[cfg(windows)]
impl HandBlueVein {
    pub fn open() -> Result<Self, crate::cuda::CudaError> {
        Ok(Self {
            gpu: crate::cuda::Nvidia::open_ptx(PTX)?,
        })
    }

    /// CUDA twin of [`reference`]. The mask feather remains the recovered host
    /// stage; only the ROI pixel correction executes on the GPU.
    pub fn apply(
        &self,
        source: &Image,
        inputs: &Inputs<'_>,
    ) -> Result<(Image, Vec<crate::cuda::Dispatch>), crate::cuda::CudaError> {
        let (vein_mask, correction_rgb) =
            resolve(source, inputs).map_err(crate::cuda::CudaError::Refused)?;
        if inputs.ui == 0 {
            return Ok((source.clone(), Vec::new()));
        }

        let feathered = hair_body::feather_box(
            vein_mask,
            source.width as usize,
            source.height as usize,
            inputs.feather_radius,
        );
        let (roi_x, roi_y, roi_width, roi_height) = inputs.roi;
        let uniforms = Uniforms {
            width: source.width,
            height: source.height,
            stride: source.width * 4,
            roi_x,
            roi_y,
            roi_width,
            roi_height,
            reserved: 0,
        };
        let grid = (
            roi_width.div_ceil(BLOCK).max(1),
            roi_height.div_ceil(BLOCK).max(1),
        );
        let (pixels, dispatch) = self.gpu.run_u8_with_init(
            KERNEL,
            grid,
            &[&source.pixels, &feathered, correction_rgb],
            as_bytes(&uniforms),
            Some(0),
        )?;
        let image = Image::new(source.width, source.height, pixels).ok_or(
            crate::cuda::CudaError::Refused(Refusal::ShapeMismatch("PF30030_OUTPUT")),
        )?;
        Ok((image, vec![dispatch]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Image, Vec<u8>, Vec<u8>) {
        const WIDTH: u32 = 9;
        const HEIGHT: u32 = 7;
        let mut source = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
        let mut mask = Vec::with_capacity((WIDTH * HEIGHT) as usize);
        let mut correction = Vec::with_capacity((WIDTH * HEIGHT * 3) as usize);
        let mask_sequence = [0, 1, 31, 64, 96, 127, 128, 129, 191, 223, 255];
        for index in 0u32..WIDTH * HEIGHT {
            let i = index as usize;
            source.extend_from_slice(&[
                (i * 17 + 3) as u8,
                (i * 43 + 11) as u8,
                (i * 97 + 29) as u8,
                (i * 13 + 7) as u8,
            ]);
            mask.push(mask_sequence[i % mask_sequence.len()]);
            for channel in 0..3 {
                let delta = ((i * 37 + channel * 53) % 201) as i32 - 100;
                correction.push((CORRECTION_NEUTRAL + delta) as u8);
            }
        }
        (Image::new(WIDTH, HEIGHT, source).unwrap(), mask, correction)
    }

    #[test]
    fn kernel_block_and_ptx_abi_match_the_header() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 32);
        assert_eq!(std::mem::align_of::<Uniforms>(), 4);
        let signature = PTX
            .split(&format!(".entry {KERNEL}("))
            .nth(1)
            .expect("PTX kernel entry")
            .split(')')
            .next()
            .unwrap();
        let params: Vec<_> = signature
            .lines()
            .filter(|line| line.contains(".param"))
            .collect();
        assert_eq!(params.len(), 5);
        assert!(
            params[4].contains(".b8") && params[4].contains("[32]"),
            "run_u8 passes the uniform block by value"
        );
    }

    #[test]
    fn missing_producers_are_typed_refusals() {
        let source = Image::solid(2, 2, [10, 20, 30, 40]);
        let mask = [255u8; 4];
        let correction = [80u8; 12];

        let missing_mask = Inputs {
            ui: 100,
            vein_mask: None,
            correction_rgb: Some(&correction),
            feather_radius: 0,
            roi: (0, 0, 2, 2),
        };
        assert_eq!(
            reference(&source, &missing_mask),
            Err(Refusal::MissingInput("PF30030_HAND_VEIN_MASK"))
        );

        let missing_correction = Inputs {
            ui: 100,
            vein_mask: Some(&mask),
            correction_rgb: None,
            feather_radius: 0,
            roi: (0, 0, 2, 2),
        };
        assert_eq!(
            reference(&source, &missing_correction),
            Err(Refusal::MissingInput("PF30030_MODEL_CORRECTION_RGB"))
        );
    }

    #[test]
    fn ui_is_only_an_enable_gate() {
        let source = Image::solid(1, 1, [10, 20, 30, 77]);
        let mask = [128u8];
        let correction = [81u8, 79, 80];
        let mut one = Inputs {
            ui: 1,
            vein_mask: Some(&mask),
            correction_rgb: Some(&correction),
            feather_radius: 0,
            roi: (0, 0, 1, 1),
        };
        let at_one = reference(&source, &one).unwrap();
        one.ui = 100;
        assert_eq!(reference(&source, &one).unwrap(), at_one);
        assert_eq!(at_one.pixel(0, 0), [11, 19, 30, 77]);
    }

    #[test]
    fn rounding_roi_and_alpha_follow_the_cpu_oracle() {
        let source = Image::new(
            3,
            1,
            vec![10, 20, 30, 77, 200, 201, 202, 88, 40, 50, 60, 99],
        )
        .unwrap();
        let mask = [128, 255, 255];
        let correction = [81, 79, 80, 120, 40, 80, 0, 0, 0];
        let inputs = Inputs {
            ui: 50,
            vein_mask: Some(&mask),
            correction_rgb: Some(&correction),
            feather_radius: 0,
            roi: (0, 0, 2, 1),
        };
        let output = reference(&source, &inputs).unwrap();
        assert_eq!(output.pixel(0, 0), [11, 19, 30, 77]);
        assert_eq!(output.pixel(1, 0), [240, 161, 202, 88]);
        assert_eq!(output.pixel(2, 0), source.pixel(2, 0));
    }

    #[cfg(windows)]
    #[test]
    fn real_cuda_matches_cpu_fixture_and_writes_evidence() {
        let (source, mask, correction) = fixture();
        let inputs = Inputs {
            ui: 100,
            vein_mask: Some(&mask),
            correction_rgb: Some(&correction),
            feather_radius: 2,
            roi: (1, 1, 7, 5),
        };
        let cpu = reference(&source, &inputs).unwrap();
        let gpu = HandBlueVein::open().unwrap();
        let (actual, dispatches) = gpu.apply(&source, &inputs).unwrap();
        assert_eq!(actual, cpu);
        assert_eq!(dispatches.len(), 1);
        let dispatch = &dispatches[0];
        assert_eq!(dispatch.kernel, KERNEL);

        let mut alpha_mismatches = 0;
        let mut outside_roi_mismatches = 0;
        let (roi_x, roi_y, roi_width, roi_height) = inputs.roi;
        for y in 0..source.height {
            for x in 0..source.width {
                let pixel = (y * source.width + x) as usize;
                if actual.pixels[pixel * 4 + 3] != source.pixels[pixel * 4 + 3] {
                    alpha_mismatches += 1;
                }
                let inside =
                    x >= roi_x && x < roi_x + roi_width && y >= roi_y && y < roi_y + roi_height;
                if !inside
                    && actual.pixels[pixel * 4..pixel * 4 + 4]
                        != source.pixels[pixel * 4..pixel * 4 + 4]
                {
                    outside_roi_mismatches += 1;
                }
            }
        }
        assert_eq!(alpha_mismatches, 0);
        assert_eq!(outside_roi_mismatches, 0);

        if let Ok(dir) = std::env::var("PF30030_FIXTURE_DIR") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("source.bgra"), &source.pixels).unwrap();
            std::fs::write(dir.join("vein_mask.r8"), &mask).unwrap();
            std::fs::write(dir.join("correction.rgb8"), &correction).unwrap();
            std::fs::write(dir.join("cpu.bgra"), &cpu.pixels).unwrap();
            std::fs::write(dir.join("gpu.bgra"), &actual.pixels).unwrap();
            std::fs::write(
                dir.join("manifest.json"),
                format!(
                    concat!(
                        "{{\n",
                        "  \"schema\": \"aurora/pf30030-gpu-fixture/1\",\n",
                        "  \"width\": {},\n",
                        "  \"height\": {},\n",
                        "  \"source\": \"source.bgra\",\n",
                        "  \"vein_mask\": \"vein_mask.r8\",\n",
                        "  \"correction_rgb\": \"correction.rgb8\",\n",
                        "  \"cpu\": \"cpu.bgra\",\n",
                        "  \"gpu\": \"gpu.bgra\",\n",
                        "  \"ui\": {},\n",
                        "  \"feather_radius\": {},\n",
                        "  \"correction_neutral\": {},\n",
                        "  \"roi\": [{}, {}, {}, {}],\n",
                        "  \"rounding\": \"nearest, ties away from zero\"\n",
                        "}}\n"
                    ),
                    source.width,
                    source.height,
                    inputs.ui,
                    inputs.feather_radius,
                    CORRECTION_NEUTRAL,
                    roi_x,
                    roi_y,
                    roi_width,
                    roi_height
                ),
            )
            .unwrap();
            std::fs::write(
                dir.join("run-summary.json"),
                format!(
                    concat!(
                        "{{\n",
                        "  \"status\": \"PASS\",\n",
                        "  \"device\": \"{}\",\n",
                        "  \"compute_capability\": \"{}.{}\",\n",
                        "  \"driver_api\": {},\n",
                        "  \"kernel\": \"{}\",\n",
                        "  \"grid\": [{}, {}],\n",
                        "  \"block\": [{}, {}],\n",
                        "  \"ptx_bytes\": {},\n",
                        "  \"ptx_isa\": \"{}.{}\",\n",
                        "  \"byte_mismatches\": 0,\n",
                        "  \"alpha_mismatches\": 0,\n",
                        "  \"outside_roi_mismatches\": 0\n",
                        "}}\n"
                    ),
                    dispatch.device,
                    dispatch.compute_cap.0,
                    dispatch.compute_cap.1,
                    dispatch.driver_version,
                    dispatch.kernel,
                    dispatch.grid.0,
                    dispatch.grid.1,
                    dispatch.block.0,
                    dispatch.block.1,
                    dispatch.ptx_bytes,
                    dispatch.ptx_isa.0,
                    dispatch.ptx_isa.1
                ),
            )
            .unwrap();
        }
    }
}
