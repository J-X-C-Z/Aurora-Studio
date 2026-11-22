//! PF30025 tooth repair on NVIDIA.
//!
//! Same input and byte contract as `kernels::face_detail::tooth_repair_same_job`:
//! repair output `780929437` is packed BGRA, output `3204548346` is the f32 gate,
//! only the ROI is consumed, source alpha is retained, and the final byte uses
//! vendor `cvtt` truncation toward zero before clamping.

use crate::kernels::face_detail::Rect;
use crate::{Image, Refusal};

#[cfg(windows)]
use crate::cuda::{CudaError, Dispatch};

pub const KERNEL: &str = "pf30025_tooth_repair_kernel";
const BLOCK: u32 = 16;
const PTX: &str = include_str!("../native/cuda/pf30025_tooth_repair.ptx");

pub struct Job<'a> {
    pub source: &'a Image,
    /// Model output `780929437`, per-face slot `+0x228`, packed BGRA8.
    pub repair_780929437_bgra: &'a [u8],
    /// Model output `3204548346`, per-face slot `+0x280`, one f32 per pixel.
    pub gate_3204548346: &'a [f32],
    pub roi: Rect,
    /// UI percent. The CPU entry multiplies degree01 by 100 before this divide.
    pub ui: f32,
}

/// 32-byte argument block, matching `native/cuda/pf30025_tooth_repair.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub degree: f32,
    pub roi_x: u32,
    pub roi_y: u32,
    pub roi_w: u32,
    pub roi_h: u32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 32);

fn bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

/// Validates before any upload and returns the CPU's normalized degree.
pub fn validate_job(job: &Job<'_>) -> Result<f32, Refusal> {
    job.source
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("source"))?;
    let pixels = job.source.pixel_count();
    let bytes = pixels
        .checked_mul(4)
        .ok_or(Refusal::ShapeMismatch("pf30025 repair_bgra"))?;
    if job.repair_780929437_bgra.is_empty() {
        return Err(Refusal::MissingInput("pf30025 repair_bgra"));
    }
    if job.repair_780929437_bgra.len() != bytes {
        return Err(Refusal::ShapeMismatch("pf30025 repair_bgra"));
    }
    if job.gate_3204548346.is_empty() {
        return Err(Refusal::MissingInput("PF30025_3204548346"));
    }
    // Keep the adapter aligned with the CPU's contiguous-plane limitation.
    if job.gate_3204548346.len() != pixels {
        return Err(Refusal::ShapeMismatch("PF30025_3204548346"));
    }
    if !job.roi.inside(job.source.width, job.source.height) {
        return Err(Refusal::ValueOutOfRange("roi"));
    }
    if !job.ui.is_finite() || !(0.0..=100.0).contains(&job.ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    for y in job.roi.y..job.roi.y + job.roi.height {
        for x in job.roi.x..job.roi.x + job.roi.width {
            let value =
                job.gate_3204548346[y as usize * job.source.width as usize + x as usize];
            if !value.is_finite() {
                return Err(Refusal::ValueOutOfRange("PF30025_3204548346"));
            }
        }
    }
    Ok(job.ui * 0.01)
}

#[cfg(windows)]
pub struct ToothRepair {
    gpu: crate::cuda::Nvidia,
}

#[cfg(windows)]
impl ToothRepair {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self {
            gpu: crate::cuda::Nvidia::open_ptx(PTX)?,
        })
    }

    pub fn apply(&self, job: &Job<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
        let degree = validate_job(job).map_err(CudaError::Refused)?;
        let uniforms = Uniforms {
            width: job.source.width,
            height: job.source.height,
            stride: job.source.width * 4,
            degree,
            roi_x: job.roi.x as u32,
            roi_y: job.roi.y as u32,
            roi_w: job.roi.width as u32,
            roi_h: job.roi.height as u32,
        };
        let grid = (
            (job.roi.width as u32).div_ceil(BLOCK).max(1),
            (job.roi.height as u32).div_ceil(BLOCK).max(1),
        );
        let (pixels, dispatch) = self.gpu.run_u8_with_init(
            KERNEL,
            grid,
            &[
                &job.source.pixels,
                job.repair_780929437_bgra,
                f32_bytes(job.gate_3204548346),
            ],
            bytes(&uniforms),
            Some(0),
        )?;
        let image = Image::new(job.source.width, job.source.height, pixels)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, vec![dispatch]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> Image {
        Image::new(
            2,
            2,
            vec![
                10, 20, 30, 77, 40, 50, 60, 88, 70, 80, 90, 99, 100, 110, 120, 111,
            ],
        )
        .unwrap()
    }

    fn job<'a>(source: &'a Image, repair: &'a [u8], gate: &'a [f32]) -> Job<'a> {
        Job {
            source,
            repair_780929437_bgra: repair,
            gate_3204548346: gate,
            roi: Rect {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
            ui: 80.0,
        }
    }

    #[test]
    fn kernel_block_matches_the_c_header() {
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
        assert!(params[4].contains(".b8") && params[4].contains("[32]"));
    }

    #[test]
    fn entry_checks_match_the_cpu_contract() {
        let source = source();
        let repair = vec![0u8; 16];
        let gate = [0.25; 4];
        assert_eq!(
            validate_job(&job(&source, &repair, &gate)),
            Ok(80.0f32 * 0.01)
        );

        let mut wrong = job(&source, &repair, &gate);
        wrong.repair_780929437_bgra = &[];
        assert_eq!(
            validate_job(&wrong),
            Err(Refusal::MissingInput("pf30025 repair_bgra"))
        );

        let wrong = job(&source, &repair[..12], &gate);
        assert_eq!(
            validate_job(&wrong),
            Err(Refusal::ShapeMismatch("pf30025 repair_bgra"))
        );

        let wrong = job(&source, &repair, &[]);
        assert_eq!(
            validate_job(&wrong),
            Err(Refusal::MissingInput("PF30025_3204548346"))
        );

        let wrong = job(&source, &repair, &gate[..3]);
        assert_eq!(
            validate_job(&wrong),
            Err(Refusal::ShapeMismatch("PF30025_3204548346"))
        );

        let mut wrong = job(&source, &repair, &gate);
        wrong.roi.x = 1;
        assert_eq!(validate_job(&wrong), Err(Refusal::ValueOutOfRange("roi")));

        let wrong = job(&source, &repair, &[0.25, 0.5, f32::NAN, 0.0]);
        assert_eq!(
            validate_job(&wrong),
            Err(Refusal::ValueOutOfRange("PF30025_3204548346"))
        );
        assert!(validate_job(&job(&source, &repair, &[2.0; 4])).is_ok());

        let mut wrong = job(&source, &repair, &gate);
        wrong.ui = 100.5;
        assert_eq!(validate_job(&wrong), Err(Refusal::ValueOutOfRange("ui")));
    }
}
