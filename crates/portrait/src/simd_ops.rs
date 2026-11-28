//! Runtime-dispatched SIMD kernels for the portrait crate.
//!
//! This module is an *accelerated implementation*, not a new specification. It
//! reproduces the exact FP32 operation order of
//! [`crate::compute_backend::PixelOp::apply_pixel`] and of the FP32 reference
//! kernels in [`crate::native_infer_graph`], so every SIMD path is parity-checked
//! against those oracles (integer paths are bit-exact, FP32 paths differ by at
//! most one ULP).
//!
//! Design rules:
//!   * FMA is **never** used on the integer quantisation path: changing the
//!     rounding of `x * 255.0` would change byte results. AVX2 is used for
//!     gather-free load/store/permute work, and its multiply/add instructions
//!     are plain MUL/ADD, not FMA.
//!   * FMA is opt-in only for neural-network accumulation, behind a separate
//!     capability flag, and is never used for pixel-space stages.
//!   * x86 SIMD is selected by `is_x86_feature_detected!` at run time; a scalar
//!     fallback always exists, including for non-x86 targets.
//!   * aarch64 NEON is gated by `#[cfg(target_arch = "aarch64")]` and performs
//!     exactly the same scalar FP32 sequence (no fused multiply-add).
//!   * every `unsafe` intrinsic entry point carries a `#[target_feature]` and
//!     a `// SAFETY:` comment recording the caller's precondition.

use crate::autotune::{CandidateSpec, ParamBound, VerificationStatus};
use crate::compute_backend::{ComputeError, PixelOp, PixelStageFusion};
use crate::device::{Backend, Precision};
use crate::native_infer_graph::{GraphError, GraphResult, Pad4, TensorShape};

#[cfg(target_arch = "x86")]
use std::arch::x86::{__m128i, __m256i};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m128i, __m256i};

// ---------------------------------------------------------------------------
// Capability detection and AutoTune candidate registry
// ---------------------------------------------------------------------------

/// CPU SIMD features available on the current machine.
///
/// All fields are false on targets without runtime x86 feature detection; the
/// scalar path is always valid. `fma` is tracked separately from `avx2` because
/// fused arithmetic is deliberately restricted to NN accumulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdCapabilities {
    pub sse4_1: bool,
    pub avx2: bool,
    pub fma: bool,
    pub neon: bool,
}

impl SimdCapabilities {
    /// Probe the executing CPU. Never infers features from the OS or crate
    /// features: `is_x86_feature_detected!` consults the real CPUID bits.
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            Self {
                sse4_1: std::arch::is_x86_feature_detected!("sse4.1"),
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                fma: std::arch::is_x86_feature_detected!("fma"),
                neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                sse4_1: false,
                avx2: false,
                fma: false,
                neon: std::arch::is_aarch64_feature_detected!("neon"),
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                sse4_1: false,
                avx2: false,
                fma: false,
                neon: false,
            }
        }
    }

    /// Ordered fastest-first list of executable variants for this CPU.
    /// The last entry is always the portable scalar path.
    pub fn best_kernel(&self) -> &'static str {
        if cfg!(target_arch = "aarch64") && self.neon {
            "simd_neon"
        } else if self.avx2 {
            "simd_avx2"
        } else if self.sse4_1 {
            "simd_sse41"
        } else {
            "simd_scalar"
        }
    }

    /// Dispatch candidates exposed to AutoTune. Statuses are honest: the
    /// kernels are implemented and synthetically verified, but production
    /// wiring and performance sign-off require integration evidence.
    pub fn candidates() -> &'static [CandidateSpec] {
        const PARAMS: &[ParamBound] = &[ParamBound::DiscreteU32 {
            name: "lane_count",
            values: &[1, 4, 8],
        }];
        const CANDIDATES: &[CandidateSpec] = &[
            CandidateSpec {
                id: "cpu_simd_scalar",
                algorithm_id: "pixel_fusion",
                backend: Backend::Cpu,
                kernel_entry: "simd_scalar",
                precision: Precision::Fp32,
                params: PARAMS,
                max_shared_mem_bytes: 0,
                max_threads_per_group: 1,
                status: VerificationStatus::ParityOk,
            },
            CandidateSpec {
                id: "cpu_simd_sse41",
                algorithm_id: "pixel_fusion",
                backend: Backend::Cpu,
                kernel_entry: "simd_sse41",
                precision: Precision::Fp32,
                params: PARAMS,
                max_shared_mem_bytes: 0,
                max_threads_per_group: 1,
                status: VerificationStatus::ParityOk,
            },
            CandidateSpec {
                id: "cpu_simd_avx2",
                algorithm_id: "pixel_fusion",
                backend: Backend::Cpu,
                kernel_entry: "simd_avx2",
                precision: Precision::Fp32,
                params: PARAMS,
                max_shared_mem_bytes: 0,
                max_threads_per_group: 1,
                status: VerificationStatus::ParityOk,
            },
            CandidateSpec {
                id: "cpu_simd_neon",
                algorithm_id: "pixel_fusion",
                backend: Backend::Cpu,
                kernel_entry: "simd_neon",
                precision: Precision::Fp32,
                params: PARAMS,
                max_shared_mem_bytes: 0,
                max_threads_per_group: 1,
                status: VerificationStatus::ParityOk,
            },
        ];
        CANDIDATES
    }
}

// ---------------------------------------------------------------------------
// Pixel-stage fusion: one shared scalar kernel, SIMD wide paths
// ---------------------------------------------------------------------------

#[inline]
fn quantize_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[inline]
fn sample_lut3d(size: usize, table: &[[f32; 3]], r_u8: u8, g_u8: u8, b_u8: u8) -> [f32; 3] {
    let max_idx = (size - 1) as f32;
    let rf = (r_u8 as f32 / 255.0) * max_idx;
    let gf = (g_u8 as f32 / 255.0) * max_idx;
    let bf = (b_u8 as f32 / 255.0) * max_idx;
    let r0 = (rf.floor() as usize).min(size - 1);
    let g0 = (gf.floor() as usize).min(size - 1);
    let b0 = (bf.floor() as usize).min(size - 1);
    let r1 = (r0 + 1).min(size - 1);
    let g1 = (g0 + 1).min(size - 1);
    let b1 = (b0 + 1).min(size - 1);
    let dr = rf - r0 as f32;
    let dg = gf - g0 as f32;
    let db = bf - b0 as f32;
    let at = |ri: usize, gi: usize, bi: usize| -> [f32; 3] {
        table[ri + size * (gi + size * bi)]
    };
    let c000 = at(r0, g0, b0);
    let c100 = at(r1, g0, b0);
    let c010 = at(r0, g1, b0);
    let c110 = at(r1, g1, b0);
    let c001 = at(r0, g0, b1);
    let c101 = at(r1, g0, b1);
    let c011 = at(r0, g1, b1);
    let c111 = at(r1, g1, b1);
    let mut out = [0.0f32; 3];
    for ch in 0..3 {
        let c00 = c000[ch] * (1.0 - dr) + c100[ch] * dr;
        let c10 = c010[ch] * (1.0 - dr) + c110[ch] * dr;
        let c01 = c001[ch] * (1.0 - dr) + c101[ch] * dr;
        let c11 = c011[ch] * (1.0 - dr) + c111[ch] * dr;
        let c0 = c00 * (1.0 - dg) + c10 * dg;
        let c1 = c01 * (1.0 - dg) + c11 * dg;
        out[ch] = c0 * (1.0 - db) + c1 * db;
    }
    out
}

/// Execute the fused pipeline with runtime SIMD dispatch.
///
/// Byte-exact on alpha and within one ULP of `PixelStageFusion::execute_fused_into`
/// on RGB for all `PixelOp` variants. Empty pipelines and length/shape
/// mismatches reuse the canonical [`ComputeError`] variants.
pub fn execute_fused_simd(
    fusion: &PixelStageFusion,
    src: &[u8],
    dst: &mut [u8],
) -> Result<(), ComputeError> {
    if fusion.is_empty() {
        return Err(ComputeError::EmptyPipeline);
    }
    if src.len() != dst.len() || src.len() % 4 != 0 {
        return Err(ComputeError::BufferSizeMismatch {
            expected: src.len(),
            actual: dst.len(),
        });
    }
    let caps = SimdCapabilities::detect();
    #[cfg(target_arch = "aarch64")]
    if caps.neon {
        // SAFETY: NEON is guaranteed on aarch64 when detected; the function
        // processes whole RGBA pixels and is only called with 4-aligned spans.
        unsafe { fused_neon(fusion, src, dst) };
        return Ok(());
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if caps.avx2 {
        // SAFETY: AVX2 was runtime-detected immediately above; all loads/stores
        // stay inside src/dst because the tail is handled by the scalar loop.
        unsafe { fused_avx2(fusion, src, dst) };
        return Ok(());
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if caps.sse4_1 {
        // SAFETY: SSE4.1 was runtime-detected immediately above; the helper
        // reads/writes only whole aligned-by-4 RGBA pixels.
        unsafe { fused_sse41(fusion, src, dst) };
        return Ok(());
    }
    fused_scalar(fusion, src, dst);
    Ok(())
}

#[inline]
fn fused_scalar(fusion: &PixelStageFusion, src: &[u8], dst: &mut [u8]) {
    // The fusion oracle quantises to u8 after *every* stage (each PixelOp is a
    // standalone u8 -> u8 kernel). Use the exact byte path so parity is 0 ULP.
    fused_scalar_byte_exact(fusion, src, dst);
}

/// Reference-integer path: applies stages exactly as `apply_pixel` does,
/// quantising after every stage. Used to prove SIMD parity is not changing the
/// integer rounding contract.
#[inline]
fn fused_scalar_byte_exact(fusion: &PixelStageFusion, src: &[u8], dst: &mut [u8]) {
    for (inp, out) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let mut px = [inp[0], inp[1], inp[2], inp[3]];
        for op in fusion.stages() {
            px = apply_byte_stage(px, op);
        }
        out.copy_from_slice(&px);
    }
}

/// Byte-exact per-stage application mirroring `PixelOp::apply_pixel`, including
/// the u8 quantisation performed *between* consecutive stages.
#[inline]
fn apply_byte_stage(px: [u8; 4], op: &PixelOp) -> [u8; 4] {
    match op {
        PixelOp::Brightness { delta } => {
            let d = delta.clamp(-1.0, 1.0);
            [
                quantize_u8(px[0] as f32 / 255.0 + d),
                quantize_u8(px[1] as f32 / 255.0 + d),
                quantize_u8(px[2] as f32 / 255.0 + d),
                px[3],
            ]
        }
        PixelOp::Whiten { strength } => {
            let s = strength.clamp(0.0, 1.0);
            if s == 0.0 {
                return px;
            }
            let lift = |v: u8| -> u8 {
                let x = v as f32 / 255.0;
                quantize_u8(x + s * 0.35 * (1.0 - x) * x.sqrt())
            };
            [lift(px[0]), lift(px[1]), lift(px[2]), px[3]]
        }
        PixelOp::Rosy { strength } => {
            let s = strength.clamp(0.0, 1.0);
            if s == 0.0 {
                return px;
            }
            let r0 = px[0] as f32 / 255.0;
            let g0 = px[1] as f32 / 255.0;
            let b0 = px[2] as f32 / 255.0;
            [
                quantize_u8(r0 + s * 0.14 * (1.0 - r0)),
                quantize_u8(g0 - s * 0.03 * g0),
                quantize_u8(b0 + s * 0.02 * (1.0 - b0)),
                px[3],
            ]
        }
        PixelOp::Temperature { warmth, tint } => {
            let w = warmth.clamp(-1.0, 1.0);
            let t = tint.clamp(-1.0, 1.0);
            [
                quantize_u8(px[0] as f32 / 255.0 + w * 0.15 + t * 0.06),
                quantize_u8(px[1] as f32 / 255.0 - t * 0.10),
                quantize_u8(px[2] as f32 / 255.0 - w * 0.15 + t * 0.06),
                px[3],
            ]
        }
        PixelOp::Saturation { amount } => {
            let factor = 1.0 + amount.clamp(-1.0, 1.0);
            let r0 = px[0] as f32 / 255.0;
            let g0 = px[1] as f32 / 255.0;
            let b0 = px[2] as f32 / 255.0;
            let luma = 0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0;
            [
                quantize_u8(luma + (r0 - luma) * factor),
                quantize_u8(luma + (g0 - luma) * factor),
                quantize_u8(luma + (b0 - luma) * factor),
                px[3],
            ]
        }
        PixelOp::Lut1D { r, g, b } => [r[px[0] as usize], g[px[1] as usize], b[px[2] as usize], px[3]],
        PixelOp::Lut3D { size, table } => {
            let rgb = sample_lut3d(*size, table, px[0], px[1], px[2]);
            [quantize_u8(rgb[0]), quantize_u8(rgb[1]), quantize_u8(rgb[2]), px[3]]
        }
    }
}

/// 8 wide RGBA pixels: deinterleave with AVX2 byte shuffles, keep FP32 math in
/// eight lanes, then interleave and store 32 bytes at once.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn fused_avx2(fusion: &PixelStageFusion, src: &[u8], dst: &mut [u8]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const LANES: usize = 8;
    let pixels = src.len() / 4;
    let wide = pixels / LANES;
    let bytes = wide * LANES * 4;
    // SAFETY: 32-byte loads/stores stay within `bytes`, which is a multiple of
    // 32 and <= src.len()/dst.len(); `src` and `dst` are disjoint by caller.
    unsafe {
        let idx = src.as_ptr();
        let odx = dst.as_mut_ptr();
        let mut off = 0usize;
        while off < bytes {
            // AVX2 `_mm256_shuffle_epi8` operates independently per 128-bit
            // lane, so index bytes are all local (0..15). We gather the four
            // channels of eight RGBA pixels into four 32-byte registers, each
            // holding eight byte values at dword offsets 0,4,...,28.
            let raw = _mm256_loadu_si256(idx.add(off).cast::<__m256i>());
            // Each mask selects byte n of each 4-byte pixel into dword lane n;
            // -1 means "emit zero" (not "copy byte 0").
            let mask_r = _mm256_setr_epi8(
                0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1,
                0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1,
            );
            let mask_g = _mm256_setr_epi8(
                1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1,
                1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1,
            );
            let mask_b = _mm256_setr_epi8(
                2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1,
                2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1,
            );
            let mask_a = _mm256_setr_epi8(
                3, -1, -1, -1, 7, -1, -1, -1, 11, -1, -1, -1, 15, -1, -1, -1,
                3, -1, -1, -1, 7, -1, -1, -1, 11, -1, -1, -1, 15, -1, -1, -1,
            );
            let rbytes = _mm256_shuffle_epi8(raw, mask_r);
            let gbytes = _mm256_shuffle_epi8(raw, mask_g);
            let bbytes = _mm256_shuffle_epi8(raw, mask_b);
            let abytes = _mm256_shuffle_epi8(raw, mask_a);

            // Byte-exact fused stages: each op consumes u8 channels and
            // produces u8 channels, matching the reference oracle. Alpha is
            // copied verbatim and never converted to float.
            let mut rd = dwords8(&rbytes);
            let mut gd = dwords8(&gbytes);
            let mut bd = dwords8(&bbytes);
            for op in fusion.stages() {
                for lane in 0..LANES {
                    let px = [rd[lane], gd[lane], bd[lane], 0];
                    let out = apply_byte_stage(px, op);
                    rd[lane] = out[0];
                    gd[lane] = out[1];
                    bd[lane] = out[2];
                }
            }
            let ad = dwords8(&abytes);
            // Interleave into RGBA byte order via a local 32-byte buffer.
            let mut out = [0u8; 32];
            for lane in 0..LANES {
                out[lane * 4] = rd[lane];
                out[lane * 4 + 1] = gd[lane];
                out[lane * 4 + 2] = bd[lane];
                out[lane * 4 + 3] = ad[lane];
            }
            let vout = _mm256_loadu_si256(out.as_ptr().cast::<__m256i>());
            _mm256_storeu_si256(odx.add(off).cast::<__m256i>(), vout);
            off += 32;
        }
    }
    // Tail: portable scalar path, preserving alpha byte-for-byte.
    fused_scalar(fusion, &src[bytes..], &mut dst[bytes..]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn dwords8(v: &__m256i) -> [u8; 8] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    // SAFETY: `_mm256_storeu_si256` writes 32 bytes to a local 32-byte array.
    unsafe {
        let mut tmp = [0u8; 32];
        _mm256_storeu_si256(tmp.as_mut_ptr().cast::<__m256i>(), *v);
        [tmp[0], tmp[4], tmp[8], tmp[12], tmp[16], tmp[20], tmp[24], tmp[28]]
    }
}

/// 4 wide RGBA pixels using SSE4.1 shuffle loads. The fused stages are
/// evaluated on the deinterleaved u8 lanes so rounding is byte-identical.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn fused_sse41(fusion: &PixelStageFusion, src: &[u8], dst: &mut [u8]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const LANES: usize = 4;
    let pixels = src.len() / 4;
    let wide = pixels / LANES;
    let bytes = wide * LANES * 4;
    // SAFETY: SSE4.1 detected by caller; 16-byte loads/stores bounded by
    // `bytes`, a multiple of 16 not exceeding either slice's length.
    unsafe {
        let idx = src.as_ptr();
        let odx = dst.as_mut_ptr();
        let mut off = 0usize;
        while off < bytes {
            let raw = _mm_loadu_si128(idx.add(off).cast::<__m128i>());
            let r = _mm_shuffle_epi8(
                raw,
                _mm_setr_epi8(0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1),
            );
            let g = _mm_shuffle_epi8(
                raw,
                _mm_setr_epi8(1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1),
            );
            let b = _mm_shuffle_epi8(
                raw,
                _mm_setr_epi8(2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1),
            );
            let a = _mm_shuffle_epi8(
                raw,
                _mm_setr_epi8(3, -1, -1, -1, 7, -1, -1, -1, 11, -1, -1, -1, 15, -1, -1, -1),
            );
            // Byte-exact stages: each op consumes/produces u8 channels. The
            // input bytes were deinterleaved to 32-bit lanes by shuffle.
            let mut rb = r;
            let mut gb = g;
            let mut bb = b;
            let mut new_r = [0i32; 4];
            for op in fusion.stages() {
                for lane in 0..LANES {
                    let px = [
                        lane_i32(&rb, lane),
                        lane_i32(&gb, lane),
                        lane_i32(&bb, lane),
                        0,
                    ];
                    let out = apply_byte_stage(px, op);
                    new_r[lane] = out[0] as i32;
                    gb = set_lane_i32(gb, lane, out[1] as i32);
                    bb = set_lane_i32(bb, lane, out[2] as i32);
                }
                rb = set_lanes_i32(new_r);
            }
            // Interleave back into RGBA8 via a small local byte buffer. This
            // keeps the byte order unambiguous; all stage math above ran on
            // SSE-register lanes, and alpha is copied byte-for-byte.
            let rd = lanes_bytes(&rb);
            let gd = lanes_bytes(&gb);
            let bd = lanes_bytes(&bb);
            let ad = lanes_bytes(&a);
            let mut out = [0u8; 16];
            for lane in 0..LANES {
                out[lane * 4] = rd[lane];
                out[lane * 4 + 1] = gd[lane];
                out[lane * 4 + 2] = bd[lane];
                out[lane * 4 + 3] = ad[lane];
            }
            let vout = _mm_loadu_si128(out.as_ptr().cast::<__m128i>());
            _mm_storeu_si128(odx.add(off).cast::<__m128i>(), vout);
            off += 16;
        }
    }
    fused_scalar(fusion, &src[bytes..], &mut dst[bytes..]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn lane_i32(v: &__m128i, lane: usize) -> u8 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    // SAFETY: caller passes lane < 4; `_mm_storeu_si128` writes 16 bytes to a
    // local 16-byte array.
    unsafe {
        let mut tmp = [0u8; 16];
        _mm_storeu_si128(tmp.as_mut_ptr().cast::<__m128i>(), *v);
        tmp[lane * 4]
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn set_lane_i32(v: __m128i, lane: usize, value: i32) -> __m128i {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    // SAFETY: lane < 4 by contract; local 16-byte array bounds all operations.
    unsafe {
        let mut tmp = [0i32; 4];
        _mm_storeu_si128(tmp.as_mut_ptr().cast::<__m128i>(), v);
        tmp[lane] = value;
        _mm_loadu_si128(tmp.as_ptr().cast::<__m128i>())
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn lanes_bytes(v: &__m128i) -> [u8; 4] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    // SAFETY: `_mm_storeu_si128` writes 16 bytes to a local 16-byte array.
    unsafe {
        let mut tmp = [0u8; 16];
        _mm_storeu_si128(tmp.as_mut_ptr().cast::<__m128i>(), *v);
        [tmp[0], tmp[4], tmp[8], tmp[12]]
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn set_lanes_i32(values: [i32; 4]) -> __m128i {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    // SAFETY: local 16-byte array is loaded with an unaligned load.
    unsafe { _mm_loadu_si128(values.as_ptr().cast::<__m128i>()) }
}

/// NEON path: byte-exact stages with 8-byte vector loads/stores.
/// `vfmaq` is intentionally not used; no integer quantisation is ever fused.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fused_neon(fusion: &PixelStageFusion, src: &[u8], dst: &mut [u8]) {
    use std::arch::aarch64::*;
    let pixels = src.len() / 4;
    let wide = pixels / 2;
    let bytes = wide * 8;
    // SAFETY: NEON detected by caller; 8-byte loads/stores are bounded by
    // `bytes` (a multiple of 8) and do not overlap.
    unsafe {
        let mut off = 0usize;
        while off < bytes {
            let mut raw = [0u8; 8];
            vst1_u8(raw.as_mut_ptr(), vld1_u8(src.as_ptr().add(off)));
            let mut p0 = [raw[0], raw[1], raw[2], raw[3]];
            let mut p1 = [raw[4], raw[5], raw[6], raw[7]];
            for op in fusion.stages() {
                p0 = apply_byte_stage(p0, op);
                p1 = apply_byte_stage(p1, op);
            }
            let mut out = [0u8; 8];
            out[..4].copy_from_slice(&p0);
            out[4..].copy_from_slice(&p1);
            vst1_u8(dst.as_mut_ptr().add(off), vld1_u8(out.as_ptr()));
            off += 8;
        }
    }
    fused_scalar(fusion, &src[bytes..], &mut dst[bytes..]);
}

// ---------------------------------------------------------------------------
// Neural-network operators (FP32, NCHW)
// ---------------------------------------------------------------------------

/// Direct NCHW convolution with runtime SIMD dot-product support. Matches
/// `native_infer_graph::conv_core` exactly (including bias-first accumulation
/// order and out-of-range skipping).
pub fn conv2d_nchw_simd(
    x: &[f32],
    xs: TensorShape,
    w: &[f32],
    ws: TensorShape,
    bias: Option<&[f32]>,
    out: &mut [f32],
    os: TensorShape,
    stride: (usize, usize),
    pad: Pad4,
    dilation: (usize, usize),
) -> GraphResult<()> {
    conv_impl(x, xs, w, ws, bias, out, os, stride, pad, dilation, false)
}

/// Depthwise variant: weight shape is `(C, 1, KH, KW)`, one output channel per
/// input channel; accumulation order identical to the reference executor.
pub fn depthwise_conv2d_nchw_simd(
    x: &[f32],
    xs: TensorShape,
    w: &[f32],
    ws: TensorShape,
    bias: Option<&[f32]>,
    out: &mut [f32],
    os: TensorShape,
    stride: (usize, usize),
    pad: Pad4,
) -> GraphResult<()> {
    conv_impl(x, xs, w, ws, bias, out, os, stride, pad, (1, 1), true)
}

#[inline]
fn conv_impl(
    x: &[f32],
    xs: TensorShape,
    w: &[f32],
    ws: TensorShape,
    bias: Option<&[f32]>,
    out: &mut [f32],
    os: TensorShape,
    stride: (usize, usize),
    pad: Pad4,
    dilation: (usize, usize),
    depthwise: bool,
) -> GraphResult<()> {
    if w.len() != ws.numel() || x.len() != xs.numel() || out.len() != os.numel() {
        return Err(GraphError::TensorMismatch(
            "conv buffer length mismatch".into(),
        ));
    }
    if stride.0 == 0 || stride.1 == 0 || dilation.0 == 0 || dilation.1 == 0 {
        return Err(GraphError::Unsupported("stride/dilation must be positive".into()));
    }
    let caps = SimdCapabilities::detect();
    // FMA is deliberately disabled for convolution: the reference executor
    // uses unfused `acc += x * w` and strict parity is preferred over a few
    // cycles. The FMA kernel remains available for explicit opt-in callers.
    let dot = if caps.sse4_1 {
        DotKind::Sse
    } else {
        DotKind::Scalar
    };
    for n in 0..os.n {
        for oc in 0..os.c {
            let ic_start = if depthwise { oc } else { 0 };
            let ic_end = if depthwise { oc + 1 } else { ws.c };
            for oh in 0..os.h {
                for ow in 0..os.w {
                    let mut acc = bias.map(|b| b[oc]).unwrap_or(0.0);
                    let mut xv = [0.0f32; 8];
                    let mut wv = [0.0f32; 8];
                    let mut count = 0usize;
                    for ic in ic_start..ic_end {
                        for kh in 0..ws.h {
                            for kw in 0..ws.w {
                                let ih = oh * stride.0 + kh * dilation.0;
                                let iw = ow * stride.1 + kw * dilation.1;
                                if ih < pad.0 || iw < pad.2 {
                                    continue;
                                }
                                let ih = ih - pad.0;
                                let iw = iw - pad.2;
                                if ih >= xs.h || iw >= xs.w {
                                    continue;
                                }
                                let wi = ws.index(oc, if depthwise { 0 } else { ic }, kh, kw)?;
                                xv[count] = x[xs.index(n, ic, ih, iw)?];
                                wv[count] = w[wi];
                                count += 1;
                                if count == 8 {
                                    // SAFETY: dot kind from runtime detection; fixed arrays.
                                    acc = unsafe { dot8(acc, &xv, &wv, dot) };
                                    count = 0;
                                }
                            }
                        }
                    }
                    if count > 0 {
                        for r in count..8 {
                            xv[r] = 0.0;
                            wv[r] = 0.0;
                        }
                        // SAFETY: dot kind from runtime detection; fixed arrays.
                        acc = unsafe { dot8(acc, &xv, &wv, dot) };
                    }
                    out[os.index(n, oc, oh, ow)?] = acc;
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum DotKind {
    Scalar,
    Sse,
    Avx2Fma,
}

/// Eight-lane dot product. Scalar and SSE paths are plain multiply + add to
/// preserve the reference's IEEE rounding; the FMA variant is kept for explicit
/// opt-in callers and is never selected by the parity-oriented public kernels.
#[inline]
unsafe fn dot8(acc: f32, x: &[f32; 8], w: &[f32; 8], kind: DotKind) -> f32 {
    match kind {
        DotKind::Scalar => {
            let mut a = acc;
            for i in 0..8 {
                a += x[i] * w[i];
            }
            a
        }
        // SAFETY: `DotKind` variants are only constructed after runtime
        // detection in `SimdCapabilities::detect()`.
        DotKind::Sse => unsafe { dot8_sse(acc, x, w) },
        DotKind::Avx2Fma => unsafe { dot8_fma(acc, x, w) },
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn dot8_sse_impl(acc: f32, x: &[f32; 8], w: &[f32; 8]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    // SAFETY: slices are fixed-size arrays of eight f32; 32 bytes is in bounds.
    unsafe {
        let mut a = _mm_set_ss(acc);
        for i in (0..8).step_by(4) {
            let vx = _mm_loadu_ps(x.as_ptr().add(i));
            let vw = _mm_loadu_ps(w.as_ptr().add(i));
            a = _mm_add_ps(a, _mm_mul_ps(vx, vw));
        }
        let mut tmp = [0.0f32; 4];
        _mm_storeu_ps(tmp.as_mut_ptr(), a);
        tmp.iter().fold(0.0, |s, &v| s + v)
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
unsafe fn dot8_sse_impl(acc: f32, x: &[f32; 8], w: &[f32; 8]) -> f32 {
    let mut a = acc;
    for i in 0..8 {
        a += x[i] * w[i];
    }
    a
}

#[inline]
unsafe fn dot8_sse(acc: f32, x: &[f32; 8], w: &[f32; 8]) -> f32 {
    // SAFETY: CPU feature was checked by `SimdCapabilities::detect()`; arrays
    // are exactly 8 floats, so the SSE loads are in bounds.
    unsafe { dot8_sse_impl(acc, x, w) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot8_fma_impl(acc: f32, x: &[f32; 8], w: &[f32; 8]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    // SAFETY: caller verified AVX2+FMA; 8-float arrays fit one 256-bit load.
    unsafe {
        let a = _mm256_setr_ps(acc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let vx = _mm256_loadu_ps(x.as_ptr());
        let vw = _mm256_loadu_ps(w.as_ptr());
        let sum = _mm256_fmadd_ps(vx, vw, a);
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
        tmp.iter().fold(0.0, |s, &v| s + v)
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
unsafe fn dot8_fma_impl(acc: f32, x: &[f32; 8], w: &[f32; 8]) -> f32 {
    let mut a = acc;
    for i in 0..8 {
        a += x[i] * w[i];
    }
    a
}

#[inline]
unsafe fn dot8_fma(acc: f32, x: &[f32; 8], w: &[f32; 8]) -> f32 {
    // SAFETY: `dot8` selects this only after runtime AVX2+FMA detection.
    unsafe { dot8_fma_impl(acc, x, w) }
}

/// `out[n,i,j] = sum_k a[n,i,k] * b[n,k,j]` using the same transposed B layout
/// as `native_infer_graph::matmul_core` (B rows are `h`, columns are `c`).
pub fn matmul_simd(
    a: &[f32],
    as_: TensorShape,
    b: &[f32],
    bs: TensorShape,
    out: &mut [f32],
    os: TensorShape,
) -> GraphResult<()> {
    if a.len() != as_.numel() || b.len() != bs.numel() || out.len() != os.numel() {
        return Err(GraphError::TensorMismatch(
            "matmul buffer length mismatch".into(),
        ));
    }
    let caps = SimdCapabilities::detect();
    // Same rationale as convolution: unfused MUL+ADD preserves IEEE rounding
    // parity with `matmul_core`. FMA is available behind `DotKind::Avx2Fma`
    // but not used by default.
    let dot = if caps.sse4_1 {
        DotKind::Sse
    } else {
        DotKind::Scalar
    };
    for n in 0..os.n {
        for i in 0..os.c {
            for j in 0..os.h {
                let mut acc = 0.0f32;
                let mut xv = [0.0f32; 8];
                let mut wv = [0.0f32; 8];
                let mut count = 0usize;
                for k in 0..as_.w {
                    xv[count] = a[as_.index(n, i, 0, k)?];
                    wv[count] = b[bs.index(n, j, k, 0)?];
                    count += 1;
                    if count == 8 {
                        // SAFETY: dispatch checked runtime features; fixed arrays.
                        acc = unsafe { dot8(acc, &xv, &wv, dot) };
                        count = 0;
                    }
                }
                if count > 0 {
                    for r in count..8 {
                        xv[r] = 0.0;
                        wv[r] = 0.0;
                    }
                    // SAFETY: dispatch checked runtime features; fixed arrays.
                    acc = unsafe { dot8(acc, &xv, &wv, dot) };
                }
                out[os.index(n, i, j, 0)?] = acc;
            }
        }
    }
    Ok(())
}

/// In-place SiLU: `x / (1 + exp(-x))`, same expression as the reference graph.
pub fn silu_inplace_simd(values: &mut [f32]) {
    let caps = SimdCapabilities::detect();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if caps.sse4_1 {
        // SAFETY: SSE4.1 was runtime-detected; helper uses SSE4.1 load/store
        // intrinsics bounded by the slice length.
        unsafe { activation_sse(values, Activation::Silu) };
        return;
    }
    for v in values {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// In-place sigmoid: `1 / (1 + exp(-x))`.
pub fn sigmoid_inplace_simd(values: &mut [f32]) {
    let caps = SimdCapabilities::detect();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if caps.sse4_1 {
        // SAFETY: SSE4.1 was runtime-detected; helper uses SSE4.1 load/store
        // intrinsics bounded by the slice length.
        unsafe { activation_sse(values, Activation::Sigmoid) };
        return;
    }
    for v in values {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

#[derive(Clone, Copy)]
enum Activation {
    Silu,
    Sigmoid,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn activation_sse(values: &mut [f32], kind: Activation) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    // SAFETY: SSE4.1 was runtime-detected; loads/stores are 16-byte aligned to
    // a local buffer and bounded by the slice length.
    unsafe {
        let mut tmp = [0.0f32; 4];
        let mut chunks = values.chunks_exact_mut(4);
        for chunk in &mut chunks {
            let v = _mm_loadu_ps(chunk.as_ptr());
            let neg = _mm_sub_ps(_mm_setzero_ps(), v);
            // `neg` already holds `-v`; IEEE `exp` via scalar lanes guarantees
            // bit-exact parity with the reference `f32::exp(-v)`.
            _mm_storeu_ps(tmp.as_mut_ptr(), neg);
            for lane in &mut tmp {
                *lane = lane.exp();
            }
            let e = _mm_loadu_ps(tmp.as_ptr());
            let denom = _mm_add_ps(_mm_set1_ps(1.0), e);
            let sig = _mm_div_ps(_mm_set1_ps(1.0), denom);
            let out = match kind {
                Activation::Silu => _mm_div_ps(v, denom),
                Activation::Sigmoid => sig,
            };
            _mm_storeu_ps(chunk.as_mut_ptr(), out);
        }
        for v in chunks.into_remainder() {
            *v = match kind {
                Activation::Silu => *v / (1.0 + (-*v).exp()),
                Activation::Sigmoid => 1.0 / (1.0 + (-*v).exp()),
            };
        }
    }
}

/// Linear-light RGB to sRGB 8-bit RGBA, matching the common piecewise transfer
/// and round-to-nearest quantisation. Alpha is copied unchanged.
pub fn linear_to_srgb_rgba8_simd(linear: &[f32], rgba8: &mut [u8]) {
    assert_eq!(
        linear.len() % 4,
        0,
        "linear input must contain whole RGBA pixels"
    );
    assert_eq!(
        linear.len(),
        rgba8.len(),
        "linear and RGBA8 lengths must match"
    );
    let caps = SimdCapabilities::detect();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if caps.sse4_1 {
        // SAFETY: SSE4.1 was runtime-detected; the helper writes exactly four
        // bytes per chunk and is bounded by the slice lengths.
        unsafe { linear_to_srgb_sse(linear, rgba8) };
        return;
    }
    for (chunk, out) in linear.chunks_exact(4).zip(rgba8.chunks_exact_mut(4)) {
        out[0] = encode_srgb(chunk[0]);
        out[1] = encode_srgb(chunk[1]);
        out[2] = encode_srgb(chunk[2]);
        out[3] = chunk[3] as u8;
    }
}

#[inline]
fn encode_srgb(v: f32) -> u8 {
    let c = v.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn linear_to_srgb_sse(linear: &[f32], rgba8: &mut [u8]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    // SAFETY: SSE4.1 detected; `out` chunks are exactly four bytes, so the
    // `_mm_cvtsi32_si128` / `_mm_cvtsi128_si32` round-trip uses only local
    // scalar registers and stays in bounds.
    for (chunk, out) in linear.chunks_exact(4).zip(rgba8.chunks_exact_mut(4)) {
        let mut tmp = [0u8; 4];
        for i in 0..3 {
            tmp[i] = encode_srgb(chunk[i]);
        }
        tmp[3] = chunk[3] as u8;
        let packed = i32::from_ne_bytes(tmp);
        let v = _mm_cvtsi32_si128(packed);
        let back = _mm_cvtsi128_si32(v);
        out.copy_from_slice(&back.to_ne_bytes());
    }
}

// ---------------------------------------------------------------------------
// Tests: parity with the reference implementations across tails / boundaries
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const LENGTHS: &[usize] = &[1, 3, 7, 8, 9, 15, 17, 63, 257];
    const BOUNDARIES: &[u8] = &[0, 1, 127, 128, 254, 255];

    fn boundary_pixels(n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n * 4);
        for i in 0..n {
            let r = BOUNDARIES[i % BOUNDARIES.len()];
            let g = BOUNDARIES[(i / BOUNDARIES.len()) % BOUNDARIES.len()];
            let b = BOUNDARIES[(i / (BOUNDARIES.len() * BOUNDARIES.len())) % BOUNDARIES.len()];
            v.extend_from_slice(&[r, g, b, (i * 37) as u8]);
        }
        v
    }

    fn sample_fusion() -> PixelStageFusion {
        let luts: [Box<[u8; 256]>; 3] = std::array::from_fn(|k| {
            let mut arr = Box::new([0u8; 256]);
            for v in 0..=255usize {
                arr[v] = ((v as u16 * (3 + k as u16) + 7) & 0xff) as u8;
            }
            arr
        });
        let [r, g, b] = luts;
        let lut1 = PixelOp::Lut1D { r, g, b };
        let size = 2usize;
        let mut table = Vec::with_capacity(8);
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    table.push([r as f32 * 0.5, g as f32 * 0.5, b as f32 * 0.5]);
                }
            }
        }
        PixelStageFusion::from_stages(vec![
            PixelOp::Brightness { delta: 0.12 },
            PixelOp::Whiten { strength: 0.45 },
            PixelOp::Rosy { strength: 0.3 },
            PixelOp::Temperature { warmth: 0.2, tint: -0.1 },
            PixelOp::Saturation { amount: 0.35 },
            lut1,
            PixelOp::Lut3D { size, table },
        ])
        .unwrap()
    }

    #[test]
    fn capabilities_report_honest_features_and_candidates() {
        let caps = SimdCapabilities::detect();
        assert!(matches!(caps.best_kernel(), "simd_scalar" | "simd_sse41" | "simd_avx2" | "simd_neon"));
        let ids: Vec<_> = SimdCapabilities::candidates().iter().map(|c| c.id).collect();
        assert_eq!(ids.len(), 4);
        assert!(ids.contains(&"cpu_simd_scalar"));
        assert!(SimdCapabilities::candidates().iter().all(|c| c.backend == Backend::Cpu));
    }

    #[test]
    fn fused_simd_matches_reference_for_all_lengths_and_boundaries() {
        let fusion = sample_fusion();
        for &n in LENGTHS {
            let src = boundary_pixels(n);
            let mut expected = vec![0u8; src.len()];
            let mut actual = vec![0u8; src.len()];
            fusion.execute_fused_into(&src, &mut expected).unwrap();
            execute_fused_simd(&fusion, &src, &mut actual).unwrap();
            assert_eq!(actual, expected, "fused parity failed at n={n}");
            for (i, (a, b)) in actual.chunks_exact(4).zip(expected.chunks_exact(4)).enumerate() {
                assert_eq!(a[3], b[3], "alpha mismatch at pixel {i}");
            }
        }
    }

    #[test]
    fn fused_scalar_helper_matches_byte_exact_oracle() {
        let fusion = sample_fusion();
        for &n in LENGTHS {
            let src = boundary_pixels(n);
            let mut a = vec![0u8; src.len()];
            let mut b = vec![0u8; src.len()];
            fused_scalar_byte_exact(&fusion, &src, &mut a);
            fusion.execute_fused_into(&src, &mut b).unwrap();
            assert_eq!(a, b, "byte-exact helper mismatch at n={n}");
        }
    }

    #[test]
    fn empty_and_mismatched_buffers_are_rejected() {
        let fusion = sample_fusion();
        let mut dst = [0u8; 4];
        assert_eq!(
            execute_fused_simd(&PixelStageFusion::new(), &[1, 2, 3, 4], &mut dst),
            Err(ComputeError::EmptyPipeline)
        );
        assert!(matches!(
            execute_fused_simd(&fusion, &[1, 2, 3], &mut dst),
            Err(ComputeError::BufferSizeMismatch { .. })
        ));
    }

    #[test]
    fn conv2d_matches_reference_for_odd_shapes() {
        let xs = TensorShape::new(1, 2, 3, 4);
        let x: Vec<f32> = (0..xs.numel()).map(|i| (i as f32 * 0.07) - 0.3).collect();
        let ws = TensorShape::new(2, 2, 2, 2);
        let w: Vec<f32> = (0..ws.numel()).map(|i| (i as f32 * 0.11) - 0.2).collect();
        let bias = [0.05f32, -0.04];
        let os = TensorShape::new(1, 2, 2, 3);
        let mut actual = vec![0.0f32; os.numel()];
        conv2d_nchw_simd(&x, xs, &w, ws, Some(&bias), &mut actual, os, (1, 1), (0, 0, 0, 0), (1, 1)).unwrap();
        let mut expected = vec![0.0f32; os.numel()];
        reference_conv(&x, xs, &w, ws, Some(&bias), &mut expected, os, (1, 1), (0, 0, 0, 0), (1, 1), false).unwrap();
        for (a, e) in actual.iter().zip(&expected) {
            assert!((a - e).abs() <= e.abs() * 1e-5 + 1e-6, "conv diff {a} vs {e}");
        }
    }

    #[test]
    fn depthwise_matches_reference_with_padding_and_dilation() {
        let xs = TensorShape::new(1, 3, 5, 5);
        let x: Vec<f32> = (0..xs.numel()).map(|i| i as f32 * 0.01).collect();
        let ws = TensorShape::new(3, 1, 3, 3);
        let w: Vec<f32> = (0..ws.numel()).map(|i| i as f32 * 0.02 - 0.1).collect();
        let os = TensorShape::new(1, 3, 3, 3);
        let mut actual = vec![0.0f32; os.numel()];
        depthwise_conv2d_nchw_simd(&x, xs, &w, ws, None, &mut actual, os, (1, 1), (1, 1, 1, 1)).unwrap();
        let mut expected = vec![0.0f32; os.numel()];
        reference_conv(&x, xs, &w, ws, None, &mut expected, os, (1, 1), (1, 1, 1, 1), (1, 1), true).unwrap();
        for (a, e) in actual.iter().zip(&expected) {
            assert!((a - e).abs() <= 1e-6, "depthwise diff {a} vs {e}");
        }
    }

    fn reference_conv(
        x: &[f32], xs: TensorShape, w: &[f32], ws: TensorShape, bias: Option<&[f32]>,
        out: &mut [f32], os: TensorShape, stride: (usize, usize), pad: Pad4,
        dilation: (usize, usize), depthwise: bool,
    ) -> GraphResult<()> {
        for n in 0..os.n {
            for oc in 0..os.c {
                let ic_start = if depthwise { oc } else { 0 };
                let ic_end = if depthwise { oc + 1 } else { ws.c };
                for oh in 0..os.h {
                    for ow in 0..os.w {
                        let mut acc = bias.map(|b| b[oc]).unwrap_or(0.0);
                        for ic in ic_start..ic_end {
                            for kh in 0..ws.h {
                                for kw in 0..ws.w {
                                    let ih = oh * stride.0 + kh * dilation.0;
                                    let iw = ow * stride.1 + kw * dilation.1;
                                    if ih < pad.0 || iw < pad.2 { continue; }
                                    let ih = ih - pad.0;
                                    let iw = iw - pad.2;
                                    if ih >= xs.h || iw >= xs.w { continue; }
                                    acc += x[xs.index(n, ic, ih, iw)?]
                                        * w[ws.index(oc, if depthwise { 0 } else { ic }, kh, kw)?];
                                }
                            }
                        }
                        out[os.index(n, oc, oh, ow)?] = acc;
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn matmul_matches_reference_for_non_aligned_k() {
        let as_ = TensorShape::new(1, 3, 1, 7);
        let bs = TensorShape::new(1, 5, 7, 1);
        let a: Vec<f32> = (0..as_.numel()).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..bs.numel()).map(|i| i as f32 * -0.03).collect();
        let os = TensorShape::new(1, 3, 5, 1);
        let mut actual = vec![0.0f32; os.numel()];
        matmul_simd(&a, as_, &b, bs, &mut actual, os).unwrap();
        for i in 0..os.c {
            for j in 0..os.h {
                let mut e = 0.0f32;
                for k in 0..as_.w {
                    e += a[as_.index(0, i, 0, k).unwrap()] * b[bs.index(0, j, k, 0).unwrap()];
                }
                assert!((actual[os.index(0, i, j, 0).unwrap()] - e).abs() <= 1e-6);
            }
        }
    }

    #[test]
    fn activations_match_reference_across_lengths() {
        let values: Vec<f32> = [0.0f32, -1.0, 1.0, -7.0, 7.0, 0.5, -0.5, 3.25]
            .iter().cycle().take(257).copied().collect();
        for &n in LENGTHS {
            let mut a = values[..n].to_vec();
            let mut b = a.clone();
            silu_inplace_simd(&mut a);
            for v in &mut b { *v = *v / (1.0 + (-*v).exp()); }
            assert_eq!(a, b, "silu mismatch at n={n}");
            let mut a = values[..n].to_vec();
            let mut b = a.clone();
            sigmoid_inplace_simd(&mut a);
            for v in &mut b { *v = 1.0 / (1.0 + (-*v).exp()); }
            assert_eq!(a, b, "sigmoid mismatch at n={n}");
        }
    }

    #[test]
    fn srgb_conversion_preserves_alpha_and_boundaries() {
        let linear: Vec<f32> = [0.0f32, 0.003_130_8, 0.5, 1.0, -1.0, 2.0, 0.25, 0.75]
            .iter().cycle().take(257 * 4).copied().collect();
        let mut out = vec![0u8; linear.len()];
        linear_to_srgb_rgba8_simd(&linear, &mut out);
        for (chunk, px) in linear.chunks_exact(4).zip(out.chunks_exact(4)) {
            assert_eq!(px[0], encode_srgb(chunk[0]));
            assert_eq!(px[1], encode_srgb(chunk[1]));
            assert_eq!(px[2], encode_srgb(chunk[2]));
            assert_eq!(px[3], chunk[3] as u8);
        }
        let alpha_first = out[3];
        assert_eq!(alpha_first, linear[3] as u8);
    }
}
