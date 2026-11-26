//! Device-side viewer scope counting: the CUDA twin of [`crate::scopes::viewer_maps`].
//!
//! `native/cuda/scope_gpu.cu` reproduces the CPU reference bin for bin — same
//! stride, same BT.709 luma order, same half-away-from-zero rounding, same
//! `cvt.rmi`-friendly form — so a frame counted here and a frame counted by
//! `viewer_maps` land in identical bins. That is the whole contract: these maps
//! are only used when the pixels they were counted from are the pixels the
//! preview shows.
//!
//! The module loads into the same process CUDA domain as `develop_gpu`, so a
//! caller that already holds the developed frame in VRAM launches these kernels
//! on the develop stream and reads the final quantized pixels straigh from the
//! device buffer it just wrote. No full-image H2D, no second context, no extra
//! stream and no barrier of its own.
//!
//! Only the zone the caller asks for is cleared and returned: the kernel adds
//! into all three maps (it is the CPU twin, not a configurable operator), but
//! `ScopeMaps::from_zone` keeps just the selected map and reports the others as
//! not covered, so the panel never carries two maps of zeros.
//!
//! `AURORA_SCOPE_GPU=0` / `false` disables the path; a missing variable means
//! "try on Windows", like the other optional CUDA stages. Any failure leaves
//! the caller counting on the CPU.

use std::ffi::c_void;

use aurora_portrait::cuda::{CudaError, Nvidia, Stream};

/// One thread per sampled pixel: 32x8 keeps the density-map atomics warp-local.
const BLOCK: (u32, u32) = (32, 8);

const PTX: &str = include_str!("../../native/cuda/scope_gpu.ptx");

/// Successful device scope passes. The parity tests assert it moved, so a silent
/// CPU fallback can never be reported as a GPU pass.
#[cfg(test)]
pub(crate) static GPU_SCOPES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `0` / `false` disables the path; a missing variable means "try on Windows".
pub(crate) fn enabled() -> bool {
    match std::env::var_os("AURORA_SCOPE_GPU") {
        Some(value) => {
            let value = value.to_string_lossy();
            !(value == "0" || value.eq_ignore_ascii_case("false"))
        }
        None => cfg!(windows),
    }
}

/// The scope kernels, loaded next to the develop module in the process domain.
pub(crate) struct ScopeGpu {
    gpu: Nvidia,
}

impl ScopeGpu {
    pub(crate) fn open() -> Result<Self, CudaError> {
        Ok(Self { gpu: Nvidia::open_ptx(PTX)? })
    }

    /// Clears `clear_words` counters at word `clear_from` of `maps`, then counts
    /// the `w`×`h` frame at `pixels` into the three density maps.
    ///
    /// `pixels` is the device address of the frame the preview will show, in the
    /// byte order `bgra` selects. Both launches go on `stream`; the caller syncs
    /// once for the frame, the scopes and the pixel copy together.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn count(
        &self,
        stream: &Stream<'_>,
        pixels: u64,
        maps: u64,
        w: usize,
        h: usize,
        clear_from: usize,
        clear_words: usize,
        bgra: bool,
    ) -> Result<(), CudaError> {
        if w == 0 || h == 0 || clear_words == 0 {
            return Ok(());
        }
        // Only the zone the caller reads back has to start from zero: the kernel
        // adds into all three maps, but the other two never leave the device.
        let mut dst = maps + (clear_from * 4) as u64;
        let mut n = clear_words as i32;
        self.launch_1d(stream, "zero_u32_kernel", clear_words, [ptr(&mut dst), ptr(&mut n)])?;

        let step = crate::scopes::viewer_step(w, h);
        let mut p = pixels;
        let mut m = maps;
        let mut w_i = w as i32;
        let mut h_i = h as i32;
        let mut step_i = step as i32;
        let mut rshift = if bgra { 16i32 } else { 0i32 };
        let sampled_w = w.div_ceil(step);
        let sampled_h = h.div_ceil(step);
        self.gpu.launch_raw_stream(
            "scope_maps_kernel",
            (
                sampled_w.div_ceil(BLOCK.0 as usize) as u32,
                sampled_h.div_ceil(BLOCK.1 as usize) as u32,
            ),
            BLOCK,
            &mut [
                ptr(&mut p),
                ptr(&mut m),
                ptr(&mut w_i),
                ptr(&mut h_i),
                ptr(&mut step_i),
                ptr(&mut rshift),
            ],
            stream,
        )?;
        #[cfg(test)]
        GPU_SCOPES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn launch_1d<const N: usize>(
        &self,
        stream: &Stream<'_>,
        kernel: &'static str,
        n: usize,
        mut args: [*mut c_void; N],
    ) -> Result<(), CudaError> {
        self.gpu.launch_raw_stream(
            kernel,
            ((n as u32).div_ceil(256), 1),
            (256, 1),
            &mut args,
            stream,
        )?;
        Ok(())
    }

    /// Counts a host frame through an upload and reads the whole layout back.
    ///
    /// Test and benchmark entry only: the preview path counts the pixels the
    /// renderer already left in VRAM, so it never uploads an image for scopes.
    #[cfg(test)]
    pub(crate) fn count_host(
        &self,
        pixels: &[u8],
        w: usize,
        h: usize,
        bgra: bool,
        wants: crate::scopes::ScopeWants,
    ) -> Result<crate::scopes::ScopeMaps, CudaError> {
        use crate::scopes::{viewer_step, ScopeMaps, VIEWER_WORDS};

        let dev_px = self.gpu.alloc_buffer(pixels.len())?;
        let dev_maps = self.gpu.alloc_buffer(VIEWER_WORDS * 4)?;
        let stream = self.gpu.create_stream()?;
        // Pageable H2D on the legacy stream can finish staging before DMA is
        // visible on a non-blocking stream. Put upload, counter and readback on
        // the same stream, keep both pinned allocations alive through the fence.
        let mut input = self.gpu.alloc_pinned(pixels.len())?;
        input.copy_from(pixels);
        self.gpu.upload_async(&dev_px, input.as_slice(), &stream)?;
        let (base, words) = wants.zone();
        let mut output = self.gpu.alloc_pinned(words * 4)?;
        self.count(&stream, dev_px.handle(), dev_maps.handle(), w, h, base, words, bgra)?;
        self.gpu.download_async_at(&dev_maps, base * 4, output.as_mut_slice(), &stream)?;
        self.gpu.stream_sync(&stream)?;
        let counts: Vec<u32> = output.as_slice()
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let step = viewer_step(w, h);
        let sampled = (w.div_ceil(step) * h.div_ceil(step)) as u32;
        Ok(ScopeMaps::from_zone(base, &counts, sampled, wants))
    }
}

fn ptr<T>(value: &mut T) -> *mut c_void {
    value as *mut T as *mut c_void
}

/// [`crate::scopes::DeviceScopeCounter`] over CUDA: the maps stay in the develop
/// pool's VRAM, the two launches go on the develop stream, and the caller's own
/// fence covers them.
impl crate::scopes::DeviceScopeCounter for ScopeGpu {
    type Submission = Stream<'static>;
    fn backend(&self) -> &'static str {
        "cuda"
    }

    fn count(
        &self,
        frame: crate::scopes::DeviceScopeFrame<'_,Self::Submission>,
        maps: u64,
        wants: crate::scopes::ScopeWants,
    ) -> Result<(), crate::scopes::ScopeCountError> {
        use crate::scopes::ScopeCountError;
        let (base, words) = wants.zone();
        if words == 0 {
            return Ok(());
        }
        if frame.width == 0 || frame.height == 0 {
            return Err(ScopeCountError::Refused("empty frame".into()));
        }
        self.count(frame.stream, frame.pixels, maps, frame.width, frame.height, base, words, frame.bgra)
            .map_err(|error| ScopeCountError::Refused(format!("{error:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scopes::{viewer_maps, ScopeMaps, ScopeWants, VIEWER_LEVELS, VIEWER_VECTOR};

    /// Photo-ish frame in BGRA: gradient plus a little noise, so no bin is
    /// trivially zero and the vectorscope really moves.
    fn frame(w: usize, h: usize) -> Vec<u8> {
        let mut out = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let base = 18.0 + 205.0 * (0.35 * x as f32 / w as f32 + 0.5 * y as f32 / h as f32);
                let noise = ((x * 37 + y * 11) % 13) as f32 - 6.0;
                let grey = (base + noise).clamp(0.0, 255.0);
                let r = (grey + 40.0 * x as f32 / w as f32).min(255.0) as u8;
                let b = (grey - 25.0 * x as f32 / w as f32).max(0.0) as u8;
                let o = (y * w + x) * 4;
                out[o] = b;
                out[o + 1] = grey as u8;
                out[o + 2] = r;
                out[o + 3] = 255;
            }
        }
        out
    }

    fn swap_red_blue(px: &[u8]) -> Vec<u8> {
        let mut out = px.to_vec();
        for p in out.chunks_exact_mut(4) {
            p.swap(0, 2);
        }
        out
    }

    fn device_or_skip(what: &str) -> Option<ScopeGpu> {
        if !enabled() {
            eprintln!("skipping {what}: AURORA_SCOPE_GPU off");
            return None;
        }
        match ScopeGpu::open() {
            Ok(gpu) => Some(gpu),
            Err(err) => {
                eprintln!("skipping {what}: no CUDA device ({err:?})");
                None
            }
        }
    }

    /// Describes where two maps disagree, without dumping 107 520 numbers.
    fn diff_summary(gpu: &[u32], cpu: &[u32]) -> String {
        crate::scopes::bins_differ(gpu, cpu).unwrap_or_else(|| "no differing bins".to_string())
    }

    /// The GPU bins and the CPU reference bins have to be equal *everywhere*,
    /// for both display orders and every map, at preview-ish sizes.
    #[test]
    fn device_bins_match_the_cpu_reference_on_a_real_card() {
        let Some(gpu) = device_or_skip("device_bins_match_the_cpu_reference_on_a_real_card") else {
            return;
        };
        let before = GPU_SCOPES.load(std::sync::atomic::Ordering::Relaxed);
        for (w, h) in [(2000usize, 1333usize), (3000, 2000), (613, 409)] {
            let bgra = frame(w, h);
            let rgba = swap_red_blue(&bgra);
            for (pixels, order, is_bgra) in [(&bgra, "bgra", true), (&rgba, "rgba", false)] {
                let cpu = viewer_maps(pixels, w, h, is_bgra, ScopeWants::ALL);
                let naive = crate::scopes::naive_viewer_maps(pixels, w, h, is_bgra);
                assert!(
                    cpu.words() == &naive[..],
                    "{w}x{h} {order}: viewer_maps disagrees with the global-stride reference: {}",
                    diff_summary(cpu.words(), &naive)
                );
                let gpu_maps = gpu
                    .count_host(pixels, w, h, is_bgra, ScopeWants::ALL)
                    .expect("device scope pass");
                assert!(
                    gpu_maps.words() == &naive[..],
                    "{w}x{h} {order}: {}",
                    diff_summary(gpu_maps.words(), &naive)
                );
                assert_eq!(gpu_maps.sampled(), cpu.sampled(), "{w}x{h} {order}: sample count");
            }
        }
        assert!(
            GPU_SCOPES.load(std::sync::atomic::Ordering::Relaxed) > before,
            "no device scope pass ran; this test must not pass on the CPU"
        );
    }

    /// One selected kind comes back with its own map filled and only its own
    /// zone filled: the panel never carries the other two maps.
    #[test]
    fn one_selected_kind_returns_only_its_zone() {
        let Some(gpu) = device_or_skip("one_selected_kind_returns_only_its_zone") else {
            return;
        };
        let (w, h) = (1600usize, 1067usize);
        let pixels = frame(w, h);
        let cpu = viewer_maps(&pixels, w, h, true, ScopeWants::ALL);
        for wants in [ScopeWants::WAVEFORM, ScopeWants::PARADE, ScopeWants::VECTOR] {
            let maps = gpu.count_host(&pixels, w, h, true, wants).expect("device scope pass");
            assert_eq!(maps.wants(), wants);
            let (base, words) = wants.zone();
            assert!(
                maps.words()[base..base + words] == cpu.words()[base..base + words],
                "{wants:?}: selected zone differs: {}",
                diff_summary(&maps.words()[base..base + words], &cpu.words()[base..base + words])
            );
            // Everything outside the selected zone stays zero, so a caller that
            // asks for a different kind can tell the map does not answer for it.
            let outside_zero = maps.words()[..base].iter().all(|c| *c == 0)
                && maps.words()[base + words..].iter().all(|c| *c == 0);
            assert!(outside_zero, "{wants:?}: unfilled zone must stay zero");
        }
    }

    /// The bin geometry the panel draws from survives the round trip: flat white
    /// lights the top row, flat black the bottom row, and neutral grey the
    /// vectorscope centre.
    #[test]
    fn device_bin_geometry_matches_the_panel_edges() {
        let Some(gpu) = device_or_skip("device_bin_geometry_matches_the_panel_edges") else {
            return;
        };
        let (w, h) = (640usize, 427usize);
        for (value, row) in [(255u8, 0usize), (0, VIEWER_LEVELS - 1)] {
            let px: Vec<u8> = std::iter::repeat([value, value, value, 255u8])
                .take(w * h)
                .flatten()
                .collect();
            let maps = gpu.count_host(&px, w, h, true, ScopeWants::WAVEFORM).expect("device pass");
            for column in [0usize, 1, 255] {
                assert!(maps.wave_at(row, column) > 0, "value {value}: row {row} column {column}");
            }
        }
        let grey: Vec<u8> = std::iter::repeat([128u8, 128, 128, 255])
            .take(w * h)
            .flatten()
            .collect();
        let maps = gpu.count_host(&grey, w, h, true, ScopeWants::VECTOR).expect("device pass");
        let centre = VIEWER_VECTOR / 2;
        assert_eq!(maps.vector_at(centre, centre), maps.vector_peak());
    }

    /// `ScopeMaps::from_zone` is the contract between the device return and the
    /// drawing code: counts land at their zone, the other maps stay empty.
    #[test]
    fn from_zone_places_counts_and_marks_the_covered_map() {
        let (base, words) = ScopeWants::PARADE.zone();
        let counts: Vec<u32> = (0..words as u32).collect();
        let maps = ScopeMaps::from_zone(base, &counts, 7, ScopeWants::PARADE);
        assert_eq!(maps.wants(), ScopeWants::PARADE);
        assert_eq!(maps.sampled(), 7);
        assert_eq!(maps.parade_bins(), &counts[..]);
        assert!(maps.wave_bins().iter().all(|c| *c == 0));
        assert!(maps.vector_bins().iter().all(|c| *c == 0));
    }
}
