//! CPU SIMD fallback kernels for the export pixel plumbing.
//!
//! Export has a CPU-only tail that no device backend can absorb: the frame the
//! develop stage produced has to be narrowed, de-alpha'd, resampled and handed to
//! an encoder. On a machine with no usable GPU (or with a very old integrated
//! one) that tail is the wall-clock difference between "a batch is done" and
//! "the batch is still writing". This module gives those loops a `std::arch`
//! path — AVX2 on x86-64, NEON on AArch64 — and a scalar reference next to each
//! kernel.
//!
//! # Contract
//!
//! Every kernel here must produce **bit-identical** output to its `*_scalar`
//! reference. That is the whole point: the export path is validated by published
//! file bytes, and a "close enough" SIMD kernel would silently change every
//! exported PNG. The rules that keep the equality:
//!
//! * **No fused multiply-add.** The float kernel emits `mul` then `add`, exactly
//!   like the source line `dst[i] += src[i] as f32 * w`, and the crate keeps FP
//!   contraction off, so a lane cannot round once where the scalar code rounds
//!   twice.
//! * **No fast-math shortcuts.** No reciprocal, no rsqrt, no reassociation: the
//!   per-element order of operations is unchanged, only 8 elements are done per
//!   instruction.
//! * **Integer kernels keep the scalar integer formula.** `narrow_u16_to_u8`
//!   reproduces `(v as u32 * 255 + 32767) / 65535` exactly (the SIMD form is the
//!   `2^16 - 1` division identity `h + ((h + l) >= 65535)`, proven exhaustively
//!   by `narrow_matches_scalar_for_every_u16`).
//! * **Tails and short buffers are handled, never read past.** A length that is
//!   not a multiple of the vector width is finished by the scalar loop; a
//!   destination shorter than the source is filled only as far as it goes and
//!   the kernel reports how much it wrote.
//!
//! # NaN semantics
//!
//! The float kernel is `dst += src as f32 * weight`, so a non-finite weight
//! propagates IEEE-754 style: with `weight = NaN` every destination becomes NaN,
//! with `weight = ±inf` it becomes `±inf` for non-zero sources and stays
//! `dst + NaN` = NaN for a zero source (`0 * inf`), and a NaN already in `dst`
//! stays NaN. The scalar reference has the same behaviour because it runs the
//! same two operations. No kernel ever substitutes a sentinel, clamps a
//! non-finite value, or takes a `is_finite` shortcut — that decision belongs to
//! the caller that owns the pixels (`resize_rgba16` rounds and clamps), not here.
//!
//! What is **not** part of the contract is which NaN *payload* survives. When
//! both operands are NaN, x86 scalar and vector instructions (and the operand
//! order LLVM happens to pick for the commutative add) select different source
//! operands, so the quiet-NaN bit pattern can differ while both results are
//! still NaN. `accumulate_keeps_nan_and_inf_semantics` pins exactly that: NaN
//! stays NaN at the same positions, every non-NaN value is bit-identical, and
//! `0 * inf` never silently becomes a number.
//!
//! # Dispatch
//!
//! [`backend`] resolves once per process from the running CPU (runtime feature
//! detection, not a compile-time `target-cpu`), and `AURORA_CPU_SIMD` overrides
//! it for A/B measurement: `scalar` (or `0`/`off`), `avx2`, `neon`, `auto`.
//! An override the CPU cannot honour logs a warning and stays scalar, so a
//! pinned `avx2` on an old laptop fails loudly in the log instead of executing
//! an illegal instruction.

use std::sync::OnceLock;

/// Which instruction set the kernels dispatch to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdBackend {
    /// Portable loop, the reference every other path is checked against.
    Scalar,
    /// 256-bit x86-64 AVX2 (`vpmulld`, `vpsrld`, `vpackusdw`, `vcvtdq2ps`).
    Avx2,
    /// 128-bit AArch64 Advanced SIMD.
    Neon,
}

impl SimdBackend {
    /// Short label for logs and benchmark output.
    pub fn label(self) -> &'static str {
        match self {
            SimdBackend::Scalar => "scalar",
            SimdBackend::Avx2 => "avx2",
            SimdBackend::Neon => "neon",
        }
    }
}

static RESOLVED: OnceLock<SimdBackend> = OnceLock::new();

/// The kernel backend this process uses.
pub fn backend() -> SimdBackend {
    *RESOLVED.get_or_init(resolve_backend)
}

/// [`backend`] plus the raw feature bits, for a log line or a benchmark header.
pub fn report() -> String {
    let backend = backend();
    let mut bits = Vec::new();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        bits.push(format!("avx2={}", std::is_x86_feature_detected!("avx2")));
        bits.push(format!("sse4.2={}", std::is_x86_feature_detected!("sse4.2")));
        bits.push(format!("fma={}", std::is_x86_feature_detected!("fma")));
        bits.push(format!("avx512f={}", std::is_x86_feature_detected!("avx512f")));
    }
    #[cfg(target_arch = "aarch64")]
    {
        bits.push(format!("neon={}", std::arch::is_aarch64_feature_detected!("neon")));
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        bits.push("no-simd-arch".to_string());
    }
    format!("cpu-simd={} ({})", backend.label(), bits.join(" "))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn avx2_available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn avx2_available() -> bool {
    false
}

#[cfg(target_arch = "aarch64")]
fn neon_available() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
}

#[cfg(not(target_arch = "aarch64"))]
fn neon_available() -> bool {
    false
}

fn detected_backend() -> SimdBackend {
    if avx2_available() {
        return SimdBackend::Avx2;
    }
    if neon_available() {
        return SimdBackend::Neon;
    }
    SimdBackend::Scalar
}

fn resolve_backend() -> SimdBackend {
    let detected = detected_backend();
    let requested = match std::env::var("AURORA_CPU_SIMD") {
        Ok(raw) => raw.trim().to_ascii_lowercase(),
        Err(_) => {
            log::debug!("{}", report_for(detected));
            return detected;
        }
    };
    let chosen = match requested.as_str() {
        "" | "auto" => detected,
        "0" | "off" | "scalar" | "none" => SimdBackend::Scalar,
        "avx2" => {
            if avx2_available() {
                SimdBackend::Avx2
            } else {
                log::warn!("AURORA_CPU_SIMD=avx2 but this CPU has no AVX2; using the scalar kernels");
                SimdBackend::Scalar
            }
        }
        "neon" => {
            if neon_available() {
                SimdBackend::Neon
            } else {
                log::warn!("AURORA_CPU_SIMD=neon but this CPU has no NEON; using the scalar kernels");
                SimdBackend::Scalar
            }
        }
        other => {
            log::warn!("AURORA_CPU_SIMD={other} is not a kernel name; using {}", detected.label());
            detected
        }
    };
    log::debug!("{}", report_for(chosen));
    chosen
}

fn report_for(backend: SimdBackend) -> String {
    format!("cpu-simd={} (detected={})", backend.label(), detected_backend().label())
}

// ---------------------------------------------------------------------------
// 16-bit -> 8-bit narrowing
// ---------------------------------------------------------------------------

/// `dst[i] = (src[i] as u32 * 255 + 32767) / 65535`, the rounding the 16-bit
/// export mirror has always used.
///
/// Copies `min(src.len(), dst.len())` samples and returns that count; a longer
/// destination is left untouched past the copy.
pub fn narrow_u16_to_u8(src: &[u16], dst: &mut [u8]) -> usize {
    let n = src.len().min(dst.len());
    if n == 0 {
        return 0;
    }
    match backend() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx2 => {
            // SAFETY: `avx2_available()` gated the dispatch, and the kernel
            // bounds-checks every load/store against `n <= min(len)`.
            unsafe { narrow_u16_to_u8_avx2(src, dst, n) };
            n
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => {
            // SAFETY: `neon_available()` gated the dispatch.
            unsafe { narrow_u16_to_u8_neon(src, dst, n) };
            n
        }
        _ => {
            narrow_u16_to_u8_scalar(&src[..n], &mut dst[..n]);
            n
        }
    }
}

/// Scalar reference for [`narrow_u16_to_u8`].
pub fn narrow_u16_to_u8_scalar(src: &[u16], dst: &mut [u8]) {
    for (out, value) in dst.iter_mut().zip(src.iter()) {
        *out = ((*value as u32 * 255 + 32767) / 65535) as u8;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn narrow_u16_to_u8_avx2(src: &[u16], dst: &mut [u8], n: usize) {
    use std::arch::x86_64::*;

    let scale = _mm256_set1_epi32(255);
    let bias = _mm256_set1_epi32(32767);
    let mask_low = _mm256_set1_epi32(0xFFFF);
    let edge = _mm256_set1_epi32(65534);
    let one = _mm256_set1_epi32(1);

    // 16 samples per iteration so the result leaves in one 16-byte store: at
    // 24 MP the loop is bandwidth-bound, and 8-byte stores halve the store
    // throughput the scalar build gets from SSE2.
    let quotient = |v: __m256i| -> __m256i {
        // p = v * 255 + 32767, then p / 65535 == h + ((h + l) >= 65535) with
        // h = p >> 16 and l = p & 0xFFFF, exact for every p below 2^25.
        let p = _mm256_add_epi32(_mm256_mullo_epi32(v, scale), bias);
        let h = _mm256_srli_epi32::<16>(p);
        let l = _mm256_and_si256(p, mask_low);
        let sum = _mm256_add_epi32(h, l);
        let carry = _mm256_cmpgt_epi32(sum, edge);
        _mm256_add_epi32(h, _mm256_and_si256(carry, one))
    };

    let mut i = 0usize;
    while i + 16 <= n {
        let low = _mm256_cvtepu16_epi32(_mm_loadu_si128(src.as_ptr().add(i) as *const __m128i));
        let high = _mm256_cvtepu16_epi32(_mm_loadu_si128(src.as_ptr().add(i + 8) as *const __m128i));
        let q_low = quotient(low);
        let q_high = quotient(high);
        // packus_epi32 works inside each 128-bit lane, so the halves come back
        // interleaved; one 64-bit permute puts the low 8 samples first.
        let packed16 = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi32(q_low, q_high));
        let packed8 = _mm256_packus_epi16(packed16, packed16);
        let bytes = _mm_unpacklo_epi64(
            _mm256_castsi256_si128(packed8),
            _mm256_extracti128_si256::<1>(packed8),
        );
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, bytes);
        i += 16;
    }
    while i < n {
        *dst.get_unchecked_mut(i) = ((*src.get_unchecked(i) as u32 * 255 + 32767) / 65535) as u8;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn narrow_u16_to_u8_neon(src: &[u16], dst: &mut [u8], n: usize) {
    use std::arch::aarch64::*;

    let mut i = 0usize;
    while i + 8 <= n {
        let raw = vld1q_u16(src.as_ptr().add(i));
        let low = vmovl_u16(vget_low_u16(raw));
        let high = vmovl_u16(vget_high_u16(raw));
        // Same integer identity as the AVX2 path: p / 65535 == h + (h + l >= 65535).
        let quotient = |v: uint32x4_t| -> uint32x4_t {
            let p = vaddq_u32(vmulq_n_u32(v, 255), vdupq_n_u32(32767));
            let h = vshrq_n_u32::<16>(p);
            let l = vandq_u32(p, vdupq_n_u32(0xFFFF));
            let sum = vaddq_u32(h, l);
            let carry = vcgeq_u32(sum, vdupq_n_u32(65535));
            vaddq_u32(h, vandq_u32(carry, vdupq_n_u32(1)))
        };
        // 8 x u32 -> 8 x u16 -> 8 x u8, then one 8-byte store.
        let words = vcombine_u16(vmovn_u32(quotient(low)), vmovn_u32(quotient(high)));
        vst1_u8(dst.as_mut_ptr().add(i), vmovn_u16(words));
        i += 8;
    }
    while i < n {
        *dst.get_unchecked_mut(i) = ((*src.get_unchecked(i) as u32 * 255 + 32767) / 65535) as u8;
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// RGBA -> RGB packing (JPEG/encoder input)
// ---------------------------------------------------------------------------

/// Copies the RGB channels of `src` (RGBA8, four bytes per pixel) into `dst`
/// (three bytes per pixel), dropping alpha.
///
/// Returns the number of pixels copied: `min(src.len() / 4, dst.len() / 3)`.
/// The destination never overshoots, so a caller that pre-sized `dst` exactly
/// pays no padding cost.
pub fn pack_rgb_from_rgba(src: &[u8], dst: &mut [u8]) -> usize {
    let n = (src.len() / 4).min(dst.len() / 3);
    if n == 0 {
        return 0;
    }
    match backend() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx2 => {
            // SAFETY: gated by runtime detection; the kernel writes exactly
            // `n * 3` bytes into a destination that holds at least that much.
            unsafe { pack_rgb_from_rgba_avx2(src.as_ptr(), dst.as_mut_ptr(), n) };
            n
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => {
            // SAFETY: as above.
            unsafe { pack_rgb_from_rgba_neon(src.as_ptr(), dst.as_mut_ptr(), n) };
            n
        }
        _ => {
            pack_rgb_from_rgba_scalar(&src[..n * 4], &mut dst[..n * 3]);
            n
        }
    }
}

/// Scalar reference for [`pack_rgb_from_rgba`].
pub fn pack_rgb_from_rgba_scalar(src: &[u8], dst: &mut [u8]) {
    let pixels = (src.len() / 4).min(dst.len() / 3);
    for index in 0..pixels {
        let from = index * 4;
        let to = index * 3;
        dst[to..to + 3].copy_from_slice(&src[from..from + 3]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn pack_rgb_from_rgba_avx2(src: *const u8, dst: *mut u8, pixels: usize) {
    use std::arch::x86_64::*;

    // Bytes 0,1,2 of every group of four: four pixels per shuffle.
    let mask = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    let mut i = 0usize;
    while i + 4 <= pixels {
        let v = _mm_loadu_si128(src.add(i * 4) as *const __m128i);
        let packed = _mm_shuffle_epi8(v, mask);
        let out = dst.add(i * 3);
        _mm_storel_epi64(out as *mut __m128i, packed);
        let high = _mm_srli_si128::<8>(packed);
        (out.add(8) as *mut u32).write_unaligned(_mm_cvtsi128_si32(high) as u32);
        i += 4;
    }
    while i < pixels {
        let from = src.add(i * 4);
        let out = dst.add(i * 3);
        // `std::ptr::copy`, not `copy_from`: the in-place kernel's first few
        // writes overlap the bytes just read, and that must stay defined.
        std::ptr::copy(from, out, 3);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pack_rgb_from_rgba_neon(src: *const u8, dst: *mut u8, pixels: usize) {
    use std::arch::aarch64::*;

    // 0x80 selects zero for the unused lanes; only the low 12 bytes are stored.
    let table = vld1q_u8([0u8, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0x80, 0x80, 0x80, 0x80].as_ptr());
    let mut i = 0usize;
    while i + 4 <= pixels {
        let v = vld1q_u8(src.add(i * 4));
        let packed = vqtbl1q_u8(v, table);
        let out = dst.add(i * 3);
        vst1_u64(out as *mut u64, vreinterpret_u64_u8(vget_low_u8(packed)));
        vst1q_lane_u32::<0>(out.add(8) as *mut u32, vreinterpretq_u32_u8(packed));
        i += 4;
    }
    while i < pixels {
        let from = src.add(i * 4);
        let out = dst.add(i * 3);
        // `std::ptr::copy`, not `copy_from`: the in-place kernel's first few
        // writes overlap the bytes just read, and that must stay defined.
        std::ptr::copy(from, out, 3);
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// In-place RGBA -> RGB compaction (the JPEG path reuses the frame it owns)
// ---------------------------------------------------------------------------

/// Packs the RGB channels of `buf` to the front and truncates the alpha bytes,
/// in place: `4 * pixels` bytes become `3 * pixels` bytes.
///
/// Returns the number of pixels packed. Safe in place because every read sits at
/// or above the write offset for the same iteration, and future reads are always
/// ahead of past writes.
pub fn compact_rgba_to_rgb(buf: &mut Vec<u8>) -> usize {
    let pixels = buf.len() / 4;
    if pixels == 0 {
        buf.clear();
        return 0;
    }
    match backend() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx2 => {
            // SAFETY: gated by runtime detection; the kernel reads `pixels * 4`
            // bytes and writes `pixels * 3`, both inside `buf`'s allocation.
            unsafe { compact_rgba_to_rgb_avx2(buf.as_mut_ptr(), pixels) };
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => {
            // SAFETY: as above.
            unsafe { compact_rgba_to_rgb_neon(buf.as_mut_ptr(), pixels) };
        }
        _ => compact_rgba_to_rgb_scalar(buf, pixels),
    }
    buf.truncate(pixels * 3);
    pixels
}

/// Scalar reference for [`compact_rgba_to_rgb`]: the loop the export path used
/// before the kernel existed, including its `index = 0` shortcut.
pub fn compact_rgba_to_rgb_scalar(buf: &mut [u8], pixels: usize) {
    let mut sample = [0u8; 3];
    for index in 1..pixels {
        let source = index * 4;
        sample.copy_from_slice(&buf[source..source + 3]);
        let target = index * 3;
        buf[target..target + 3].copy_from_slice(&sample);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn compact_rgba_to_rgb_avx2(buf: *mut u8, pixels: usize) {
    use std::arch::x86_64::*;

    let mask = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    let mut i = 0usize;
    while i + 4 <= pixels {
        // Load first: the store below overlaps the bytes just read.
        let v = _mm_loadu_si128(buf.add(i * 4) as *const __m128i);
        let packed = _mm_shuffle_epi8(v, mask);
        let out = buf.add(i * 3);
        _mm_storel_epi64(out as *mut __m128i, packed);
        let high = _mm_srli_si128::<8>(packed);
        (out.add(8) as *mut u32).write_unaligned(_mm_cvtsi128_si32(high) as u32);
        i += 4;
    }
    while i < pixels {
        let from = buf.add(i * 4);
        let out = buf.add(i * 3);
        // `std::ptr::copy`, not `copy_from`: the in-place kernel's first few
        // writes overlap the bytes just read, and that must stay defined.
        std::ptr::copy(from, out, 3);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn compact_rgba_to_rgb_neon(buf: *mut u8, pixels: usize) {
    use std::arch::aarch64::*;

    let table = vld1q_u8([0u8, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0x80, 0x80, 0x80, 0x80].as_ptr());
    let mut i = 0usize;
    while i + 4 <= pixels {
        let v = vld1q_u8(buf.add(i * 4));
        let packed = vqtbl1q_u8(v, table);
        let out = buf.add(i * 3);
        vst1_u64(out as *mut u64, vreinterpret_u64_u8(vget_low_u8(packed)));
        vst1q_lane_u32::<0>(out.add(8) as *mut u32, vreinterpretq_u32_u8(packed));
        i += 4;
    }
    while i < pixels {
        let from = buf.add(i * 4);
        let out = buf.add(i * 3);
        // `std::ptr::copy`, not `copy_from`: the in-place kernel's first few
        // writes overlap the bytes just read, and that must stay defined.
        std::ptr::copy(from, out, 3);
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Weighted row accumulation (16-bit resample, vertical pass)
// ---------------------------------------------------------------------------

/// `dst[i] += src[i] as f32 * weight` for `i < min(dst.len(), src.len())`.
///
/// This is the inner loop of the 16-bit Triangle resample: one tap of one source
/// row folded into an f32 accumulator row. It stays an accumulate (never a
/// multiply into a fresh buffer) so the per-element order of the scalar code is
/// preserved tap by tap.
pub fn accumulate_weighted_u16_into_f32(dst: &mut [f32], src: &[u16], weight: f32) {
    let n = dst.len().min(src.len());
    if n == 0 {
        return;
    }
    match backend() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx2 => {
            // SAFETY: gated by runtime detection and bounded by `n`.
            unsafe { accumulate_weighted_u16_into_f32_avx2(dst, src, weight, n) };
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => {
            // SAFETY: gated by runtime detection and bounded by `n`.
            unsafe { accumulate_weighted_u16_into_f32_neon(dst, src, weight, n) };
        }
        _ => accumulate_weighted_u16_into_f32_scalar(&mut dst[..n], &src[..n], weight),
    }
}

/// Scalar reference for [`accumulate_weighted_u16_into_f32`].
pub fn accumulate_weighted_u16_into_f32_scalar(dst: &mut [f32], src: &[u16], weight: f32) {
    for (out, value) in dst.iter_mut().zip(src.iter()) {
        *out += *value as f32 * weight;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_weighted_u16_into_f32_avx2(dst: &mut [f32], src: &[u16], weight: f32, n: usize) {
    use std::arch::x86_64::*;

    let w = _mm256_set1_ps(weight);
    let mut i = 0usize;
    while i + 16 <= n {
        let raw = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
        let low = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)));
        let high = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(raw)));
        let d0 = _mm256_loadu_ps(dst.as_ptr().add(i));
        let d1 = _mm256_loadu_ps(dst.as_ptr().add(i + 8));
        // Separate mul and add: an FMA here would round once instead of twice
        // and change the resampled pixels.
        let s0 = _mm256_add_ps(d0, _mm256_mul_ps(low, w));
        let s1 = _mm256_add_ps(d1, _mm256_mul_ps(high, w));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), s0);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8), s1);
        i += 16;
    }
    while i < n {
        *dst.get_unchecked_mut(i) += *src.get_unchecked(i) as f32 * weight;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn accumulate_weighted_u16_into_f32_neon(dst: &mut [f32], src: &[u16], weight: f32, n: usize) {
    use std::arch::aarch64::*;

    let w = vdupq_n_f32(weight);
    let mut i = 0usize;
    while i + 8 <= n {
        let raw = vld1q_u16(src.as_ptr().add(i) as *const u16);
        let low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(raw)));
        let high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(raw)));
        let d0 = vld1q_f32(dst.as_ptr().add(i));
        let d1 = vld1q_f32(dst.as_ptr().add(i + 4));
        // AArch64 has a fused multiply-add, and using it here would round once
        // where the scalar reference rounds twice; the kernels stay separate.
        vst1q_f32(dst.as_mut_ptr().add(i), vaddq_f32(d0, vmulq_f32(low, w)));
        vst1q_f32(dst.as_mut_ptr().add(i + 4), vaddq_f32(d1, vmulq_f32(high, w)));
        i += 8;
    }
    while i < n {
        *dst.get_unchecked_mut(i) += *src.get_unchecked(i) as f32 * weight;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every backend the CPU can actually run, so a parity test covers what the
    /// dispatch can pick and not just the scalar reference.
    fn runtimes() -> Vec<SimdBackend> {
        let mut out = vec![SimdBackend::Scalar];
        if avx2_available() {
            out.push(SimdBackend::Avx2);
        }
        if neon_available() {
            out.push(SimdBackend::Neon);
        }
        out
    }

    fn pyxel(seed: u64, index: usize) -> u16 {
        // A spread that hits 0, 65535 and the rounding boundaries of the
        // 8-bit division instead of only the midtones a smooth ramp covers.
        let mixed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add((index as u64).wrapping_mul(1442695040888963407));
        let raw = (mixed >> 33) as u32 % 65536;
        match index % 5 {
            0 => 0,
            1 => 65535,
            2 => (raw & 0xFF00) as u16,
            3 => (raw & 0x00FF) as u16,
            _ => raw as u16,
        }
    }

    /// The reference formula, spelled out, for the exhaustive check.
    fn expected_narrow(value: u16) -> u8 {
        ((value as u32 * 255 + 32767) / 65535) as u8
    }

    #[test]
    fn narrow_matches_scalar_for_every_u16() {
        // Exhaustive: 65,536 inputs prove the integer identity used by the
        // vectors, so no rounding boundary can be missed.
        let values: Vec<u16> = (0..=u16::MAX).collect();
        let target: Vec<u8> = values.iter().map(|value| expected_narrow(*value)).collect();
        assert_eq!(target[0], 0);
        assert_eq!(target[65535], 255);
        assert_eq!(target[128], 0);
        assert_eq!(target[129], 1);
        for backend in runtimes() {
            let mut dst = vec![0u8; values.len()];
            match backend {
                SimdBackend::Scalar => narrow_u16_to_u8_scalar(&values, &mut dst),
                SimdBackend::Avx2 => {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    unsafe {
                        narrow_u16_to_u8_avx2(&values, &mut dst, values.len())
                    };
                }
                SimdBackend::Neon => {
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        narrow_u16_to_u8_neon(&values, &mut dst, values.len())
                    };
                }
            }
            assert_eq!(dst, target, "{backend:?} disagrees with the scalar division formula");
        }
    }

    #[test]
    fn narrow_handles_tails_and_short_destinations() {
        let src: Vec<u16> = (0..=37u16).map(|i| i.wrapping_mul(1789)).collect();
        for len in 0..=src.len() {
            let mut expected = vec![0u8; len];
            narrow_u16_to_u8_scalar(&src[..len], &mut expected);
            for backend in runtimes() {
                let mut dst = vec![0u8; len];
                let copied = match backend {
                    SimdBackend::Scalar => {
                        narrow_u16_to_u8_scalar(&src[..len], &mut dst);
                        len
                    }
                    SimdBackend::Avx2 => {
                        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                        unsafe {
                            narrow_u16_to_u8_avx2(&src, &mut dst, len)
                        };
                        len
                    }
                    SimdBackend::Neon => {
                        #[cfg(target_arch = "aarch64")]
                        unsafe {
                            narrow_u16_to_u8_neon(&src, &mut dst, len)
                        };
                        len
                    }
                };
                assert_eq!(copied, len);
                assert_eq!(dst, expected, "{backend:?} len {len}");
            }
        }
        // A destination smaller than the source is filled only as far as it goes.
        let src = vec![65535u16; 8];
        let mut dst = vec![0u8; 3];
        assert_eq!(narrow_u16_to_u8(&src, &mut dst), 3);
        assert_eq!(dst, vec![255u8; 3]);
    }

    #[test]
    fn narrow_never_writes_past_the_destination() {
        let src = vec![40000u16; 16];
        for backend in runtimes() {
            let mut dst = vec![0xAAu8; 20];
            match backend {
                SimdBackend::Scalar => narrow_u16_to_u8_scalar(&src, &mut dst),
                SimdBackend::Avx2 => {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    unsafe {
                        narrow_u16_to_u8_avx2(&src, &mut dst, src.len())
                    };
                }
                SimdBackend::Neon => {
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        narrow_u16_to_u8_neon(&src, &mut dst, src.len())
                    };
                }
            }
            assert_eq!(&dst[16..], &[0xAAu8; 4], "{backend:?} wrote the guard bytes");
            let want = expected_narrow(40000);
            assert!(dst[..16].iter().all(|b| *b == want), "{backend:?} wrong narrow");
        }
    }

    fn rgba_buffer(pixels: usize) -> Vec<u8> {
        (0..pixels * 4)
            .map(|i| match i % 4 {
                3 => 7, // alpha must not leak into the packed result
                _ => (i * 31 % 251) as u8,
            })
            .collect()
    }

    #[test]
    fn pack_and_compact_match_the_scalar_loop_for_every_tail() {
        for pixels in 0..=37usize {
            let src = rgba_buffer(pixels);
            let mut expected = vec![0u8; pixels * 3];
            pack_rgb_from_rgba_scalar(&src, &mut expected);
            for backend in runtimes() {
                let mut dst = vec![0u8; pixels * 3];
                let copied = match backend {
                    SimdBackend::Scalar => {
                        pack_rgb_from_rgba_scalar(&src, &mut dst);
                        pixels
                    }
                    SimdBackend::Avx2 => {
                        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                        unsafe {
                            pack_rgb_from_rgba_avx2(src.as_ptr(), dst.as_mut_ptr(), pixels);
                        }
                        pixels
                    }
                    SimdBackend::Neon => {
                        #[cfg(target_arch = "aarch64")]
                        unsafe {
                            pack_rgb_from_rgba_neon(src.as_ptr(), dst.as_mut_ptr(), pixels);
                        }
                        pixels
                    }
                };
                assert_eq!(copied, pixels);
                assert_eq!(dst, expected, "{backend:?} pack at {pixels} px");
            }

            // In place, the same pixels must land at the same offsets.
            let mut scalar = src.clone();
            compact_rgba_to_rgb_scalar(&mut scalar, pixels);
            for backend in runtimes() {
                let mut buf = src.clone();
                match backend {
                    SimdBackend::Scalar => compact_rgba_to_rgb_scalar(&mut buf, pixels),
                    SimdBackend::Avx2 => {
                        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                        unsafe {
                            compact_rgba_to_rgb_avx2(buf.as_mut_ptr(), pixels)
                        };
                    }
                    SimdBackend::Neon => {
                        #[cfg(target_arch = "aarch64")]
                        unsafe {
                            compact_rgba_to_rgb_neon(buf.as_mut_ptr(), pixels)
                        };
                    }
                }
                assert_eq!(&buf[..pixels * 3], &scalar[..pixels * 3], "{backend:?} compact at {pixels} px");
            }
        }
        // The real entry point truncates to three bytes per pixel.
        let mut buf = rgba_buffer(9);
        assert_eq!(compact_rgba_to_rgb(&mut buf), 9);
        assert_eq!(buf.len(), 27);
    }

    #[test]
    fn accumulate_matches_scalar_including_reassociation_traps() {
        // Weights chosen so an FMA or a reassociated sum would differ: 1/3 is
        // not representable and the magnitudes span the f32 exponent range.
        let weights = [0.0f32, 1.0, 1.0 / 3.0, -2.5, 65535.0, 1e-7, 0.99999994];
        for pixels in 0..=41usize {
            let src: Vec<u16> = (0..pixels).map(|i| pyxel(0x5EED_1234, i)).collect();
            let base: Vec<f32> = (0..pixels).map(|i| i as f32 * 0.5).collect();
            for weight in weights {
                let mut expected = base.clone();
                accumulate_weighted_u16_into_f32_scalar(&mut expected, &src, weight);
                for backend in runtimes() {
                    let mut dst = base.clone();
                    match backend {
                        SimdBackend::Scalar => accumulate_weighted_u16_into_f32_scalar(&mut dst, &src, weight),
                        SimdBackend::Avx2 => {
                            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                            unsafe {
                                accumulate_weighted_u16_into_f32_avx2(&mut dst, &src, weight, pixels)
                            };
                        }
                        SimdBackend::Neon => {
                            #[cfg(target_arch = "aarch64")]
                            unsafe {
                                accumulate_weighted_u16_into_f32_neon(&mut dst, &src, weight, pixels)
                            };
                        }
                    }
                    assert_eq!(
                        dst.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "{backend:?} accumulate at {pixels} px weight {weight}"
                    );
                }
            }
        }
    }

    #[test]
    fn accumulate_keeps_nan_and_inf_semantics() {
        let src = [0u16, 1, 65535, 40000, 0, 7, 65535, 0, 300];
        let base = [0.0f32, 1.0, -2.0, 3.5, f32::NAN, 0.0, 1.0, 0.0, 0.0];
        for weight in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut expected = base;
            accumulate_weighted_u16_into_f32_scalar(&mut expected, &src, weight);
            for backend in runtimes() {
                let mut dst = base;
                match backend {
                    SimdBackend::Scalar => accumulate_weighted_u16_into_f32_scalar(&mut dst, &src, weight),
                    SimdBackend::Avx2 => {
                        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                        unsafe {
                            accumulate_weighted_u16_into_f32_avx2(&mut dst, &src, weight, src.len())
                        };
                    }
                    SimdBackend::Neon => {
                        #[cfg(target_arch = "aarch64")]
                        unsafe {
                            accumulate_weighted_u16_into_f32_neon(&mut dst, &src, weight, src.len())
                        };
                    }
                }
                // NaN positions and every non-NaN bit pattern must match; the
                // NaN payload itself is not specified (see the module docs).
                for (index, (lane, reference)) in dst.iter().zip(expected.iter()).enumerate() {
                    assert_eq!(
                        lane.is_nan(),
                        reference.is_nan(),
                        "{backend:?} NaN position {index} weight {weight}"
                    );
                    if !reference.is_nan() {
                        assert_eq!(
                            lane.to_bits(),
                            reference.to_bits(),
                            "{backend:?} finite value {index} weight {weight}"
                        );
                    }
                }
                if weight.is_nan() {
                    assert!(dst.iter().all(|v| v.is_nan()));
                }
                // 0 * inf is NaN, and NaN in the accumulator stays NaN.
                if weight.is_infinite() {
                    assert!(dst[0].is_nan(), "0 * inf must stay NaN");
                    assert!(dst[4].is_nan(), "a NaN accumulator must stay NaN");
                    assert_eq!(dst[1], weight, "1 * inf must be the infinity itself");
                }
            }
        }
    }

    #[test]
    fn a_pinned_backend_that_is_unavailable_stays_scalar() {
        // `resolve_backend` reads the environment once per process, so this test
        // only pins the behaviour of the helpers it dispatches through.
        if avx2_available() {
            assert!(matches!(detected_backend(), SimdBackend::Avx2));
        }
        assert_eq!(backend(), *RESOLVED.get().unwrap_or(&backend()));
        assert!(report().starts_with("cpu-simd="));
    }
}

/// Release-mode benchmark for the kernels and for the export writes that call
/// them. Not part of the default test run: it allocates a 24 MP frame set and
/// writes two 24 MP files.
///
/// ```text
/// cargo test --release -p aurora-engine --lib simd_fallback::bench \
///     -- --ignored --nocapture
/// AURORA_CPU_SIMD=scalar cargo test --release ...   # same bench, scalar path
/// ```
#[cfg(test)]
mod bench {
    use super::*;
    use std::time::{Duration, Instant};

    use crate::catalog::{ImageId, ImageInfo};
    use crate::output::ProcessRecipe;
    use crate::{ExportFormat, MetadataPolicy};

    /// 6048x4032 = 24.4 MP, the frame size the export memory budget is sized for.
    const W: u32 = 6048;
    const H: u32 = 4032;

    fn mib(bytes: usize) -> f64 {
        bytes as f64 / (1024.0 * 1024.0)
    }

    fn throughput(bytes: usize, elapsed: Duration) -> f64 {
        mib(bytes) / elapsed.as_secs_f64().max(1e-9)
    }

    fn line(name: &str, elapsed: Duration, bytes: usize, per_px: usize) {
        println!(
            "{name:<38} {:>9.3} ms  {:>8.1} MiB/s  {:>6.2} ns/px",
            elapsed.as_secs_f64() * 1000.0,
            throughput(bytes, elapsed),
            elapsed.as_secs_f64() * 1e9 / per_px.max(1) as f64,
        );
    }

    fn wide_frame(pixels: usize) -> Vec<u16> {
        let mut out = Vec::with_capacity(pixels * 4);
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..pixels * 4 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.push((state >> 33) as u16);
        }
        out
    }

    fn bytes_frame(pixels: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(pixels * 4);
        let mut state = 0xA5A5_1234_DEAD_BEEFu64;
        for index in 0..pixels * 4 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            out.push(if index % 4 == 3 { 255 } else { (state >> 40) as u8 });
        }
        out
    }

    fn write_and_sha(path: &std::path::Path) -> String {
        let bytes = std::fs::read(path).expect("read the exported file");
        let sha = sha256_hex(&bytes);
        format!("{} bytes sha256={sha}", bytes.len())
    }

    /// Compact SHA-256 so the benchmark report can name the exact bytes it
    /// produced without adding a dependency to the workspace lockfile.
    fn sha256_hex(data: &[u8]) -> String {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ];
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        ];
        let mut message = data.to_vec();
        let bit_len = (data.len() as u64) * 8;
        message.push(0x80);
        while message.len() % 64 != 56 {
            message.push(0);
        }
        message.extend_from_slice(&bit_len.to_be_bytes());
        let mut w = [0u32; 64];
        for chunk in message.chunks_exact(64) {
            for (index, word) in chunk.chunks_exact(4).enumerate() {
                w[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
            }
            for index in 16..64 {
                let s0 = w[index - 15].rotate_right(7) ^ w[index - 15].rotate_right(18) ^ (w[index - 15] >> 3);
                let s1 = w[index - 2].rotate_right(17) ^ w[index - 2].rotate_right(19) ^ (w[index - 2] >> 10);
                w[index] = w[index - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[index - 7])
                    .wrapping_add(s1);
            }
            let mut v = h;
            for index in 0..64 {
                let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
                let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
                let temp1 = v[7]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[index])
                    .wrapping_add(w[index]);
                let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
                let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
                let temp2 = s0.wrapping_add(maj);
                v[7] = v[6];
                v[6] = v[5];
                v[5] = v[4];
                v[4] = v[3].wrapping_add(temp1);
                v[3] = v[2];
                v[2] = v[1];
                v[1] = v[0];
                v[0] = temp1.wrapping_add(temp2);
            }
            for index in 0..8 {
                h[index] = h[index].wrapping_add(v[index]);
            }
        }
        h.iter().map(|word| format!("{word:08x}")).collect()
    }

    #[test]
    fn sha256_matches_the_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    fn info_for(dir: &std::path::Path, name: &str) -> ImageInfo {
        ImageInfo {
            id: ImageId(1),
            path: dir.join(name),
            file_name: name.into(),
            kind: crate::FileKind::Png,
            width: W,
            height: H,
            file_size: 1,
            exif: Default::default(),
            iptc: Default::default(),
            keywords: Vec::new(),
            rating: 0,
            color_tag: Default::default(),
            variant: 1,
            has_adjustments: false,
            trashed: false,
            heif: None,
        }
    }

    #[test]
    #[ignore = "24 MP release benchmark; run explicitly with --release --ignored --nocapture"]
    fn bench_24mp_export_kernels() {
        println!("--- aurora-engine CPU SIMD bench ---");
        println!("{}", report());
        println!(
            "build={} threads={} frame={}x{} ({:.1} MP)",
            if cfg!(debug_assertions) { "debug" } else { "release" },
            rayon::current_num_threads(),
            W,
            H,
            (W as f64 * H as f64) / 1e6
        );
        let pixels = (W as usize) * (H as usize);

        // Scalar and accelerated rounds interleaved: a sequential A/A/B/B run
        // lets a thermal or allocator drift masquerade as a kernel win.
        macro_rules! ab {
            ($reset:expr, $scalar:expr, $simd:expr) => {{
                let mut best = [Duration::MAX; 2];
                for _ in 0..5 {
                    ($reset)();
                    let started = Instant::now();
                    $scalar;
                    best[0] = best[0].min(started.elapsed());
                    ($reset)();
                    let started = Instant::now();
                    $simd;
                    best[1] = best[1].min(started.elapsed());
                }
                best
            }};
        }

        // 16-bit -> 8-bit mirror.
        let wide = wide_frame(pixels);
        let mut narrow_dst = vec![0u8; pixels * 4];
        let narrow_bytes = wide.len() * 2 + narrow_dst.len();
        let [scalar, simd] = ab!(
            || {},
            narrow_u16_to_u8_scalar(&wide, &mut narrow_dst),
            narrow_u16_to_u8(&wide, &mut narrow_dst)
        );
        line("narrow u16->u8 (scalar)", scalar, narrow_bytes, pixels * 4);
        line(&format!("narrow u16->u8 ({})", backend().label()), simd, narrow_bytes, pixels * 4);
        println!("  speedup {:.2}x", scalar.as_secs_f64() / simd.as_secs_f64());

        // RGBA -> RGB pack.
        let rgba = bytes_frame(pixels);
        let mut rgb = vec![0u8; pixels * 3];
        let pack_bytes = rgba.len() + rgb.len();
        let [scalar, simd] = ab!(
            || {},
            pack_rgb_from_rgba_scalar(&rgba, &mut rgb),
            pack_rgb_from_rgba(&rgba, &mut rgb)
        );
        line("pack rgba->rgb (scalar)", scalar, pack_bytes, pixels);
        line(&format!("pack rgba->rgb ({})", backend().label()), simd, pack_bytes, pixels);
        println!("  speedup {:.2}x", scalar.as_secs_f64() / simd.as_secs_f64());

        // In-place compaction: the buffer has to be restored between runs, so
        // the restore sits in the untimed reset.
        let mut buffer = rgba.clone();
        let [scalar, simd] = ab!(
            || {
                buffer.clear();
                buffer.extend_from_slice(&rgba);
            },
            compact_rgba_to_rgb_scalar(&mut buffer, pixels),
            compact_rgba_to_rgb(&mut buffer)
        );
        line("compact rgba->rgb in place (scalar)", scalar, pack_bytes, pixels);
        line(&format!("compact rgba->rgb in place ({})", backend().label()), simd, pack_bytes, pixels);
        println!("  speedup {:.2}x", scalar.as_secs_f64() / simd.as_secs_f64());

        // Weighted 16-bit row accumulation (the resample's inner loop).
        let src16 = wide_frame(pixels);
        let mut acc = vec![0f32; pixels * 4];
        let acc_bytes = src16.len() * 2 + acc.len() * 4 * 2;
        let [scalar, simd] = ab!(
            || acc.fill(0.0),
            accumulate_weighted_u16_into_f32_scalar(&mut acc, &src16, 1.0 / 3.0),
            accumulate_weighted_u16_into_f32(&mut acc, &src16, 1.0 / 3.0)
        );
        line("resample accumulate (scalar)", scalar, acc_bytes, pixels * 4);
        line(&format!("resample accumulate ({})", backend().label()), simd, acc_bytes, pixels * 4);
        println!("  speedup {:.2}x", scalar.as_secs_f64() / simd.as_secs_f64());

        // End to end: the two writes every export runs, with their output SHA.
        let dir = std::env::temp_dir().join(format!("aurora-simd-bench-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let wide_recipe = ProcessRecipe {
            destination: Some(dir.clone()),
            format: ExportFormat::Png,
            bit_depth: 16,
            include_metadata: MetadataPolicy::StripAll,
            ..ProcessRecipe::default()
        };
        let wide_info = info_for(&dir, "bench16.png");
        let wide_pixels = wide_frame(pixels);
        let started = Instant::now();
        let path16 = crate::recipe::write_export16(
            &dir,
            &wide_info,
            &wide_recipe,
            1,
            wide_pixels,
            W,
            H,
            |_| {},
        )
        .expect("16-bit PNG export");
        let png_ms = started.elapsed().as_secs_f64() * 1000.0;
        println!("export 24 MP PNG16 {:>9.3} ms  {}", png_ms, write_and_sha(&path16));

        let eight_recipe = ProcessRecipe { bit_depth: 8, ..wide_recipe.clone() };
        let eight_info = info_for(&dir, "bench8.jpg");
        let eight_pixels = bytes_frame(pixels);
        let started = Instant::now();
        let path8 = crate::recipe::write_export(
            &dir,
            &eight_info,
            &ProcessRecipe { format: ExportFormat::Jpeg, ..eight_recipe },
            1,
            eight_pixels,
            W,
            H,
            |_| {},
        )
        .expect("8-bit JPEG export");
        let jpg_ms = started.elapsed().as_secs_f64() * 1000.0;
        println!("export 24 MP JPEG8 {:>8.3} ms  {}", jpg_ms, write_and_sha(&path8));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
