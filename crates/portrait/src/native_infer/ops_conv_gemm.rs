//! Native CUDA kernels and Rust dispatch bindings for core linear algebra &
//! convolution operators (`Conv`, `ConvTranspose`, `MatMul`, `Gemm`).
//!
//! All launch entry points accept raw `u64` CUDA device pointers so they compose
//! directly with `arena.rs` static byte offsets (`arena_base_ptr + offset`) with
//! zero intermediate allocations.

use crate::cuda::{CudaError, Dispatch};
use crate::native_infer::graph::{ConvAttrs, ConvTransposeAttrs, GemmAttrs};
use crate::Refusal;

#[cfg(windows)]
use crate::cuda::{Nvidia, Stream};
#[cfg(windows)]
use std::ffi::c_void;

/// Embedded PTX compiled from `native/cuda/infer_conv_gemm.cu`.
pub const PTX: &str = include_str!("../../native/cuda/infer_conv_gemm.ptx");

pub const KERNEL_IM2COL_NCHW: &str = "im2col_nchw_f32";
pub const KERNEL_SGEMM_TILED: &str = "sgemm_tiled_f32";
pub const KERNEL_ADD_BIAS_NCHW: &str = "add_bias_nchw_f32";
pub const KERNEL_DEPTHWISE_CONV2D: &str = "depthwise_conv2d_nchw_f32";
pub const KERNEL_CONV_TRANSPOSE2D: &str = "conv_transpose2d_nchw_f32";
pub const KERNEL_BATCHED_MATMUL: &str = "batched_matmul_f32";

const THREADS_1D: u32 = 256;
const GEMM_TILE: u32 = 32;
const GEMM_BLOCK: (u32, u32) = (16, 16);

/// Optional fused epilogue activation supported by Conv / GEMM kernels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum FusedActivation {
    #[default]
    None = 0,
    Relu = 1,
    Silu = 2,
}

/// Bias broadcasting mode for `sgemm_tiled_f32`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum GemmBiasMode {
    /// No external bias pointer; if `beta != 0`, accumulates `beta * C[row, col]`.
    #[default]
    None = 0,
    /// `beta * bias[row]` (Conv NCHW per-channel bias or Gemm `[M, 1]`).
    RowBroadcast = 1,
    /// `beta * bias[col]` (Gemm `[N]` or `[1, N]`).
    ColBroadcast = 2,
    /// `beta * bias[0]` (Gemm scalar `[1]` or `[]`).
    ScalarBroadcast = 3,
    /// `beta * bias[row * N + col]` (Gemm non-in-place `[M, N]`).
    Full2D = 4,
}

/// Concrete 2D NCHW convolution geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dParams {
    pub batch: usize,
    pub c_in: usize,
    pub h_in: usize,
    pub w_in: usize,
    pub c_out: usize,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub dh: usize,
    pub dw: usize,
    pub pad_t: usize,
    pub pad_l: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub group: usize,
    pub activation: FusedActivation,
}

impl Conv2dParams {
    /// Build concrete Conv2D parameters from NCHW input/weight shapes and ONNX `ConvAttrs`.
    pub fn from_attrs(
        x_shape: &[usize],
        w_shape: &[usize],
        attrs: &ConvAttrs,
        activation: FusedActivation,
    ) -> Result<Self, CudaError> {
        if x_shape.len() != 4 || w_shape.len() != 4 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("conv2d rank")));
        }
        let (batch, c_in, h_in, w_in) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
        let c_out = w_shape[0];
        let kh = attrs.kernel_shape.as_ref().and_then(|v| v.first().copied()).unwrap_or(w_shape[2]);
        let kw = attrs.kernel_shape.as_ref().and_then(|v| v.get(1).copied()).unwrap_or(w_shape[3]);
        let sh = attrs.strides.first().copied().unwrap_or(1);
        let sw = attrs.strides.get(1).copied().unwrap_or(1);
        let dh = attrs.dilations.first().copied().unwrap_or(1);
        let dw = attrs.dilations.get(1).copied().unwrap_or(1);
        let pad_t = attrs.pads.first().copied().unwrap_or(0);
        let pad_l = attrs.pads.get(1).copied().unwrap_or(0);
        let pad_b = attrs.pads.get(2).copied().unwrap_or(pad_t);
        let pad_r = attrs.pads.get(3).copied().unwrap_or(pad_l);
        let group = attrs.group;

        if group == 0
            || c_in == 0
            || c_out == 0
            || kh == 0
            || kw == 0
            || sh == 0
            || sw == 0
            || dh == 0
            || dw == 0
            || c_in % group != 0
            || c_out % group != 0
            || w_shape[1] != c_in / group
        {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("conv2d attrs")));
        }

        let eff_kh = dh * (kh - 1) + 1;
        let eff_kw = dw * (kw - 1) + 1;
        if h_in + pad_t + pad_b < eff_kh || w_in + pad_l + pad_r < eff_kw {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("conv2d spatial")));
        }
        let out_h = (h_in + pad_t + pad_b - eff_kh) / sh + 1;
        let out_w = (w_in + pad_l + pad_r - eff_kw) / sw + 1;

        Ok(Self {
            batch,
            c_in,
            h_in,
            w_in,
            c_out,
            kh,
            kw,
            sh,
            sw,
            dh,
            dw,
            pad_t,
            pad_l,
            out_h,
            out_w,
            group,
            activation,
        })
    }

    /// True if this convolution can run on the zero-workspace depthwise kernel.
    #[inline]
    pub fn is_depthwise(&self) -> bool {
        self.group > 1 && self.group == self.c_in && self.c_in == self.c_out
    }

    /// True if this convolution is a direct 1x1 pointwise conv requiring no im2col.
    #[inline]
    pub fn is_direct_1x1(&self) -> bool {
        self.kh == 1
            && self.kw == 1
            && self.sh == 1
            && self.sw == 1
            && self.pad_t == 0
            && self.pad_l == 0
            && self.out_h == self.h_in
            && self.out_w == self.w_in
    }

    /// Scratch workspace bytes required by [`ConvGemmKernels::conv2d_nchw`].
    /// Returns `0` for depthwise and direct 1x1 convolutions.
    pub fn im2col_workspace_bytes(&self) -> usize {
        if self.is_depthwise() || self.is_direct_1x1() {
            return 0;
        }
        self.c_in
            .saturating_mul(self.kh)
            .saturating_mul(self.kw)
            .saturating_mul(self.out_h)
            .saturating_mul(self.out_w)
            .saturating_mul(std::mem::size_of::<f32>())
    }
}

/// Concrete 2D NCHW transposed convolution geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvTranspose2dParams {
    pub batch: usize,
    pub c_in: usize,
    pub h_in: usize,
    pub w_in: usize,
    pub c_out: usize,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub dh: usize,
    pub dw: usize,
    pub pad_t: usize,
    pub pad_l: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub group: usize,
    pub activation: FusedActivation,
}

impl ConvTranspose2dParams {
    /// Build concrete ConvTranspose2D parameters from NCHW input/weight shapes and ONNX `ConvTransposeAttrs`.
    pub fn from_attrs(
        x_shape: &[usize],
        w_shape: &[usize],
        attrs: &ConvTransposeAttrs,
        activation: FusedActivation,
    ) -> Result<Self, CudaError> {
        if x_shape.len() != 4 || w_shape.len() != 4 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch(
                "conv_transpose2d rank",
            )));
        }
        let (batch, c_in, h_in, w_in) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
        let group = attrs.group;
        let c_out = w_shape[1] * group;
        let kh = attrs.kernel_shape.as_ref().and_then(|v| v.first().copied()).unwrap_or(w_shape[2]);
        let kw = attrs.kernel_shape.as_ref().and_then(|v| v.get(1).copied()).unwrap_or(w_shape[3]);
        let sh = attrs.strides.first().copied().unwrap_or(1);
        let sw = attrs.strides.get(1).copied().unwrap_or(1);
        let dh = attrs.dilations.first().copied().unwrap_or(1);
        let dw = attrs.dilations.get(1).copied().unwrap_or(1);
        let pad_t = attrs.pads.first().copied().unwrap_or(0);
        let pad_l = attrs.pads.get(1).copied().unwrap_or(0);
        let pad_b = attrs.pads.get(2).copied().unwrap_or(pad_t);
        let pad_r = attrs.pads.get(3).copied().unwrap_or(pad_l);
        let oph = attrs.output_padding.first().copied().unwrap_or(0);
        let opw = attrs.output_padding.get(1).copied().unwrap_or(0);

        if group == 0
            || c_in == 0
            || c_out == 0
            || kh == 0
            || kw == 0
            || sh == 0
            || sw == 0
            || dh == 0
            || dw == 0
            || w_shape[0] != c_in
            || c_in % group != 0
            || c_out % group != 0
            || h_in == 0
            || w_in == 0
        {
            return Err(CudaError::Refused(Refusal::ShapeMismatch(
                "conv_transpose2d attrs",
            )));
        }

        let raw_h = sh * (h_in - 1) + oph + dh * (kh - 1) + 1;
        let raw_w = sw * (w_in - 1) + opw + dw * (kw - 1) + 1;
        if raw_h <= pad_t + pad_b || raw_w <= pad_l + pad_r {
            return Err(CudaError::Refused(Refusal::ShapeMismatch(
                "conv_transpose2d spatial",
            )));
        }
        let out_h = raw_h - pad_t - pad_b;
        let out_w = raw_w - pad_l - pad_r;

        Ok(Self {
            batch,
            c_in,
            h_in,
            w_in,
            c_out,
            kh,
            kw,
            sh,
            sw,
            dh,
            dw,
            pad_t,
            pad_l,
            out_h,
            out_w,
            group,
            activation,
        })
    }
}

/// Broadcast metadata for up to 4 batch dimensions (2D..=6D `MatMul`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchedMatMulPlan {
    m: u32,
    n: u32,
    k: u32,
    dims: [u32; 4],
    a_strides: [u64; 4],
    b_strides: [u64; 4],
    batch_total: u32,
}

fn plan_batched_matmul(a_shape: &[usize], b_shape: &[usize]) -> Result<BatchedMatMulPlan, CudaError> {
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(CudaError::Refused(Refusal::ShapeMismatch("matmul rank < 2")));
    }
    let a_batch = &a_shape[..a_shape.len() - 2];
    let b_batch = &b_shape[..b_shape.len() - 2];
    let max_b_rank = a_batch.len().max(b_batch.len());
    if max_b_rank > 4 {
        return Err(CudaError::Refused(Refusal::ShapeMismatch(
            "matmul batch rank > 4",
        )));
    }

    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let kb = b_shape[b_shape.len() - 2];
    let n = b_shape[b_shape.len() - 1];
    if k != kb || m == 0 || n == 0 || k == 0 {
        return Err(CudaError::Refused(Refusal::ShapeMismatch("matmul dims")));
    }

    // Compute element strides for A's and B's batch dimensions (right-aligned to 4 slots).
    let mut a_raw_strides = [0u64; 4];
    let mut acc_a = (m as u64) * (k as u64);
    for (i, &dim) in a_batch.iter().rev().enumerate() {
        a_raw_strides[3 - i] = acc_a;
        acc_a *= dim as u64;
    }

    let mut b_raw_strides = [0u64; 4];
    let mut acc_b = (k as u64) * (n as u64);
    for (i, &dim) in b_batch.iter().rev().enumerate() {
        b_raw_strides[3 - i] = acc_b;
        acc_b *= dim as u64;
    }

    let mut dims = [1u32; 4];
    let mut a_strides = [0u64; 4];
    let mut b_strides = [0u64; 4];
    let mut batch_total: u64 = 1;

    for slot in 0..4 {
        let rev_idx = 3 - slot;
        let da = if rev_idx < a_batch.len() {
            a_batch[a_batch.len() - 1 - rev_idx]
        } else {
            1
        };
        let db = if rev_idx < b_batch.len() {
            b_batch[b_batch.len() - 1 - rev_idx]
        } else {
            1
        };
        let d_out = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return Err(CudaError::Refused(Refusal::ShapeMismatch(
                "matmul broadcast",
            )));
        };
        if d_out == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("matmul zero dim")));
        }
        dims[slot] = d_out as u32;
        a_strides[slot] = if da == 1 { 0 } else { a_raw_strides[slot] };
        b_strides[slot] = if db == 1 { 0 } else { b_raw_strides[slot] };
        batch_total = batch_total
            .checked_mul(d_out as u64)
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("matmul batch")))?;
    }

    let batch_total_u32 = u32::try_from(batch_total)
        .map_err(|_| CudaError::Refused(Refusal::ValueOutOfRange("matmul batch")))?;

    Ok(BatchedMatMulPlan {
        m: to_u32(m)?,
        n: to_u32(n)?,
        k: to_u32(k)?,
        dims,
        a_strides,
        b_strides,
        batch_total: batch_total_u32,
    })
}

fn resolve_gemm_bias_mode(
    c_in_ptr: Option<u64>,
    c_in_shape: Option<&[usize]>,
    m: usize,
    n: usize,
) -> Result<(Option<u64>, GemmBiasMode), CudaError> {
    let Some(ptr) = c_in_ptr else {
        return Ok((None, GemmBiasMode::None));
    };
    let default_shape = [n];
    let shape = c_in_shape.unwrap_or(&default_shape);
    let mode = match shape {
        [] | [1] | [1, 1] => GemmBiasMode::ScalarBroadcast,
        [d] if *d == n => GemmBiasMode::ColBroadcast,
        [1, d] if *d == n => GemmBiasMode::ColBroadcast,
        [d, 1] if *d == m => GemmBiasMode::RowBroadcast,
        [dm, dn] if *dm == m && *dn == n => GemmBiasMode::Full2D,
        _ => {
            return Err(CudaError::Refused(Refusal::ShapeMismatch(
                "gemm bias broadcast",
            )));
        }
    };
    Ok((Some(ptr), mode))
}

#[inline]
fn to_u32(v: usize) -> Result<u32, CudaError> {
    u32::try_from(v).map_err(|_| CudaError::Refused(Refusal::ValueOutOfRange("u32 dim")))
}

#[inline]
fn grid_1d(total: usize) -> Result<(u32, u32), CudaError> {
    let blocks = total.div_ceil(THREADS_1D as usize);
    let blocks_u32 =
        u32::try_from(blocks).map_err(|_| CudaError::Refused(Refusal::ValueOutOfRange("grid_1d")))?;
    Ok((blocks_u32.max(1), 1))
}

// ===========================================================================
// Windows CUDA Driver Dispatch Implementation
// ===========================================================================

#[cfg(windows)]
pub struct ConvGemmKernels {
    nvidia: Nvidia,
}

#[cfg(windows)]
impl ConvGemmKernels {
    /// Load and JIT-compile `infer_conv_gemm.ptx` on the shared process CUDA context.
    pub fn open() -> Result<Self, CudaError> {
        let nvidia = Nvidia::open_ptx(PTX)?;
        Ok(Self { nvidia })
    }

    /// Access the underlying `Nvidia` handle for buffer allocation, copies, or streams.
    pub fn nvidia(&self) -> &Nvidia {
        &self.nvidia
    }

    fn launch(
        &self,
        kernel: &'static str,
        grid: (u32, u32),
        block: (u32, u32),
        params: &mut [*mut c_void],
        stream: Option<&Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        match stream {
            Some(s) => self.nvidia.launch_raw_stream(kernel, grid, block, params, s),
            None => self.nvidia.launch_raw(kernel, grid, block, params),
        }
    }

    /// Unfold one NCHW tensor `[c_in, h_in, w_in]` at `x_ptr` into `[c_in * kh * kw, out_h * out_w]` at `col_ptr`.
    #[allow(clippy::too_many_arguments)]
    pub fn im2col_nchw(
        &self,
        x_ptr: u64,
        col_ptr: u64,
        c_in: usize,
        h_in: usize,
        w_in: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        dh: usize,
        dw: usize,
        pad_t: usize,
        pad_l: usize,
        out_h: usize,
        out_w: usize,
        stream: Option<&Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        let total = c_in
            .checked_mul(kh)
            .and_then(|v| v.checked_mul(kw))
            .and_then(|v| v.checked_mul(out_h))
            .and_then(|v| v.checked_mul(out_w))
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("im2col elems")))?;
        if total == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("im2col empty")));
        }

        let mut p_x = x_ptr;
        let mut p_col = col_ptr;
        let mut p_cin = to_u32(c_in)?;
        let mut p_hin = to_u32(h_in)?;
        let mut p_win = to_u32(w_in)?;
        let mut p_kh = to_u32(kh)?;
        let mut p_kw = to_u32(kw)?;
        let mut p_sh = to_u32(sh)?;
        let mut p_sw = to_u32(sw)?;
        let mut p_dh = to_u32(dh)?;
        let mut p_dw = to_u32(dw)?;
        let mut p_pt = to_u32(pad_t)?;
        let mut p_pl = to_u32(pad_l)?;
        let mut p_oh = to_u32(out_h)?;
        let mut p_ow = to_u32(out_w)?;

        let mut args: [*mut c_void; 15] = [
            &mut p_x as *mut u64 as *mut c_void,
            &mut p_col as *mut u64 as *mut c_void,
            &mut p_cin as *mut u32 as *mut c_void,
            &mut p_hin as *mut u32 as *mut c_void,
            &mut p_win as *mut u32 as *mut c_void,
            &mut p_kh as *mut u32 as *mut c_void,
            &mut p_kw as *mut u32 as *mut c_void,
            &mut p_sh as *mut u32 as *mut c_void,
            &mut p_sw as *mut u32 as *mut c_void,
            &mut p_dh as *mut u32 as *mut c_void,
            &mut p_dw as *mut u32 as *mut c_void,
            &mut p_pt as *mut u32 as *mut c_void,
            &mut p_pl as *mut u32 as *mut c_void,
            &mut p_oh as *mut u32 as *mut c_void,
            &mut p_ow as *mut u32 as *mut c_void,
        ];

        self.launch(
            KERNEL_IM2COL_NCHW,
            grid_1d(total)?,
            (THREADS_1D, 1),
            &mut args,
            stream,
        )
    }

    /// Strided-batched 32x32 shared-memory tiled SGEMM with optional fused bias and activation.
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_strided_batched(
        &self,
        a_ptr: u64,
        b_ptr: u64,
        c_ptr: u64,
        bias_ptr: Option<u64>,
        bias_mode: GemmBiasMode,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        beta: f32,
        batch_count: usize,
        stride_a: u64,
        stride_b: u64,
        stride_c: u64,
        stride_bias: u64,
        activation: FusedActivation,
        stream: Option<&Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        if m == 0 || n == 0 || k == 0 || batch_count == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("sgemm zero dim")));
        }
        let m_u32 = to_u32(m)?;
        let n_u32 = to_u32(n)?;
        let k_u32 = to_u32(k)?;
        let batch_u32 = to_u32(batch_count)?;

        let tiles_m = m_u32.div_ceil(GEMM_TILE);
        let tiles_n = n_u32.div_ceil(GEMM_TILE);
        let total_tiles = (batch_u32 as u64)
            .checked_mul(tiles_m as u64)
            .and_then(|v| v.checked_mul(tiles_n as u64))
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("sgemm grid")))?;
        let total_tiles_u32 = u32::try_from(total_tiles)
            .map_err(|_| CudaError::Refused(Refusal::ValueOutOfRange("sgemm grid")))?;

        let mut p_a = a_ptr;
        let mut p_b = b_ptr;
        let mut p_c = c_ptr;
        let mut p_bias = bias_ptr.unwrap_or(0);
        let mut p_m = m_u32;
        let mut p_n = n_u32;
        let mut p_k = k_u32;
        let mut p_ta: u32 = u32::from(trans_a);
        let mut p_tb: u32 = u32::from(trans_b);
        let mut p_alpha = alpha;
        let mut p_beta = beta;
        let mut p_bias_mode = bias_mode as u32;
        let mut p_act = activation as u32;
        let mut p_batch = batch_u32;
        let mut p_sa = stride_a;
        let mut p_sb = stride_b;
        let mut p_sc = stride_c;
        let mut p_sbias = stride_bias;

        let mut args: [*mut c_void; 18] = [
            &mut p_a as *mut u64 as *mut c_void,
            &mut p_b as *mut u64 as *mut c_void,
            &mut p_c as *mut u64 as *mut c_void,
            &mut p_bias as *mut u64 as *mut c_void,
            &mut p_m as *mut u32 as *mut c_void,
            &mut p_n as *mut u32 as *mut c_void,
            &mut p_k as *mut u32 as *mut c_void,
            &mut p_ta as *mut u32 as *mut c_void,
            &mut p_tb as *mut u32 as *mut c_void,
            &mut p_alpha as *mut f32 as *mut c_void,
            &mut p_beta as *mut f32 as *mut c_void,
            &mut p_bias_mode as *mut u32 as *mut c_void,
            &mut p_act as *mut u32 as *mut c_void,
            &mut p_batch as *mut u32 as *mut c_void,
            &mut p_sa as *mut u64 as *mut c_void,
            &mut p_sb as *mut u64 as *mut c_void,
            &mut p_sc as *mut u64 as *mut c_void,
            &mut p_sbias as *mut u64 as *mut c_void,
        ];

        self.launch(
            KERNEL_SGEMM_TILED,
            (total_tiles_u32, 1),
            GEMM_BLOCK,
            &mut args,
            stream,
        )
    }

    /// Standalone in-place NCHW channel bias add + optional fused activation.
    pub fn add_bias_nchw(
        &self,
        y_ptr: u64,
        bias_ptr: Option<u64>,
        batch: usize,
        channels: usize,
        spatial: usize,
        activation: FusedActivation,
        stream: Option<&Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        let total = batch
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(spatial))
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("add_bias elems")))?;
        if total == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("add_bias empty")));
        }

        let mut p_y = y_ptr;
        let mut p_bias = bias_ptr.unwrap_or(0);
        let mut p_batch = to_u32(batch)?;
        let mut p_channels = to_u32(channels)?;
        let mut p_spatial = to_u32(spatial)?;
        let mut p_act = activation as u32;

        let mut args: [*mut c_void; 6] = [
            &mut p_y as *mut u64 as *mut c_void,
            &mut p_bias as *mut u64 as *mut c_void,
            &mut p_batch as *mut u32 as *mut c_void,
            &mut p_channels as *mut u32 as *mut c_void,
            &mut p_spatial as *mut u32 as *mut c_void,
            &mut p_act as *mut u32 as *mut c_void,
        ];

        self.launch(
            KERNEL_ADD_BIAS_NCHW,
            grid_1d(total)?,
            (THREADS_1D, 1),
            &mut args,
            stream,
        )
    }

    /// Direct zero-workspace Depthwise Conv2D (`group == c_in == c_out`).
    pub fn depthwise_conv2d_nchw(
        &self,
        x_ptr: u64,
        w_ptr: u64,
        bias_ptr: Option<u64>,
        y_ptr: u64,
        params: &Conv2dParams,
        stream: Option<&Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        if params.group != params.c_in || params.c_in != params.c_out {
            return Err(CudaError::Refused(Refusal::ShapeMismatch(
                "depthwise group mismatch",
            )));
        }
        let total = params
            .batch
            .checked_mul(params.c_out)
            .and_then(|v| v.checked_mul(params.out_h))
            .and_then(|v| v.checked_mul(params.out_w))
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("dwconv elems")))?;
        if total == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("dwconv empty")));
        }

        let mut p_x = x_ptr;
        let mut p_w = w_ptr;
        let mut p_bias = bias_ptr.unwrap_or(0);
        let mut p_y = y_ptr;
        let mut p_n = to_u32(params.batch)?;
        let mut p_c = to_u32(params.c_in)?;
        let mut p_hin = to_u32(params.h_in)?;
        let mut p_win = to_u32(params.w_in)?;
        let mut p_kh = to_u32(params.kh)?;
        let mut p_kw = to_u32(params.kw)?;
        let mut p_sh = to_u32(params.sh)?;
        let mut p_sw = to_u32(params.sw)?;
        let mut p_dh = to_u32(params.dh)?;
        let mut p_dw = to_u32(params.dw)?;
        let mut p_pt = to_u32(params.pad_t)?;
        let mut p_pl = to_u32(params.pad_l)?;
        let mut p_oh = to_u32(params.out_h)?;
        let mut p_ow = to_u32(params.out_w)?;
        let mut p_act = params.activation as u32;

        let mut args: [*mut c_void; 19] = [
            &mut p_x as *mut u64 as *mut c_void,
            &mut p_w as *mut u64 as *mut c_void,
            &mut p_bias as *mut u64 as *mut c_void,
            &mut p_y as *mut u64 as *mut c_void,
            &mut p_n as *mut u32 as *mut c_void,
            &mut p_c as *mut u32 as *mut c_void,
            &mut p_hin as *mut u32 as *mut c_void,
            &mut p_win as *mut u32 as *mut c_void,
            &mut p_kh as *mut u32 as *mut c_void,
            &mut p_kw as *mut u32 as *mut c_void,
            &mut p_sh as *mut u32 as *mut c_void,
            &mut p_sw as *mut u32 as *mut c_void,
            &mut p_dh as *mut u32 as *mut c_void,
            &mut p_dw as *mut u32 as *mut c_void,
            &mut p_pt as *mut u32 as *mut c_void,
            &mut p_pl as *mut u32 as *mut c_void,
            &mut p_oh as *mut u32 as *mut c_void,
            &mut p_ow as *mut u32 as *mut c_void,
            &mut p_act as *mut u32 as *mut c_void,
        ];

        self.launch(
            KERNEL_DEPTHWISE_CONV2D,
            grid_1d(total)?,
            (THREADS_1D, 1),
            &mut args,
            stream,
        )
    }

    /// Unified NCHW Conv2D dispatcher supporting standard, 1x1 pointwise, dilated,
    /// grouped, and depthwise convolutions with optional fused bias and activation.
    ///
    /// `workspace_ptr` may be `0` when `params.im2col_workspace_bytes() == 0`
    /// (depthwise or direct 1x1 convolutions); otherwise it must point to at least
    /// `params.im2col_workspace_bytes()` bytes of device memory.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_nchw(
        &self,
        x_ptr: u64,
        w_ptr: u64,
        bias_ptr: Option<u64>,
        y_ptr: u64,
        workspace_ptr: u64,
        params: &Conv2dParams,
        stream: Option<&Stream<'_>>,
    ) -> Result<(), CudaError> {
        if params.is_depthwise() {
            self.depthwise_conv2d_nchw(x_ptr, w_ptr, bias_ptr, y_ptr, params, stream)?;
            return Ok(());
        }

        let group = params.group;
        let c_in_g = params.c_in / group;
        let c_out_g = params.c_out / group;
        let k_dim = c_in_g * params.kh * params.kw;
        let out_hw = params.out_h * params.out_w;
        let in_sample_bytes = (params.c_in * params.h_in * params.w_in * 4) as u64;
        let out_sample_bytes = (params.c_out * out_hw * 4) as u64;

        let stride_a = (c_out_g * k_dim) as u64;
        let stride_b = (k_dim * out_hw) as u64;
        let stride_c = (c_out_g * out_hw) as u64;
        let stride_bias = c_out_g as u64;

        let (bias_mode, beta) = if bias_ptr.is_some() {
            (GemmBiasMode::RowBroadcast, 1.0f32)
        } else {
            (GemmBiasMode::None, 0.0f32)
        };

        let direct_1x1 = params.is_direct_1x1();
        if !direct_1x1 && workspace_ptr == 0 {
            return Err(CudaError::Refused(Refusal::MissingInput("im2col workspace")));
        }

        for n_i in 0..params.batch {
            let x_n = x_ptr + (n_i as u64) * in_sample_bytes;
            let y_n = y_ptr + (n_i as u64) * out_sample_bytes;

            let b_mat_ptr = if direct_1x1 {
                x_n
            } else {
                self.im2col_nchw(
                    x_n,
                    workspace_ptr,
                    params.c_in,
                    params.h_in,
                    params.w_in,
                    params.kh,
                    params.kw,
                    params.sh,
                    params.sw,
                    params.dh,
                    params.dw,
                    params.pad_t,
                    params.pad_l,
                    params.out_h,
                    params.out_w,
                    stream,
                )?;
                workspace_ptr
            };

            self.sgemm_strided_batched(
                w_ptr,
                b_mat_ptr,
                y_n,
                bias_ptr,
                bias_mode,
                c_out_g,
                out_hw,
                k_dim,
                false,
                false,
                1.0,
                beta,
                group,
                stride_a,
                stride_b,
                stride_c,
                stride_bias,
                params.activation,
                stream,
            )?;
        }

        Ok(())
    }

    /// Gather-based NCHW ConvTranspose2D (`k=2x2 s=2x2 p=0` and general parameters).
    pub fn conv_transpose2d_nchw(
        &self,
        x_ptr: u64,
        w_ptr: u64,
        bias_ptr: Option<u64>,
        y_ptr: u64,
        params: &ConvTranspose2dParams,
        stream: Option<&Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        let total = params
            .batch
            .checked_mul(params.c_out)
            .and_then(|v| v.checked_mul(params.out_h))
            .and_then(|v| v.checked_mul(params.out_w))
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("convt elems")))?;
        if total == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("convt empty")));
        }

        let mut p_x = x_ptr;
        let mut p_w = w_ptr;
        let mut p_bias = bias_ptr.unwrap_or(0);
        let mut p_y = y_ptr;
        let mut p_n = to_u32(params.batch)?;
        let mut p_cin = to_u32(params.c_in)?;
        let mut p_hin = to_u32(params.h_in)?;
        let mut p_win = to_u32(params.w_in)?;
        let mut p_cout = to_u32(params.c_out)?;
        let mut p_kh = to_u32(params.kh)?;
        let mut p_kw = to_u32(params.kw)?;
        let mut p_sh = to_u32(params.sh)?;
        let mut p_sw = to_u32(params.sw)?;
        let mut p_dh = to_u32(params.dh)?;
        let mut p_dw = to_u32(params.dw)?;
        let mut p_pt = to_u32(params.pad_t)?;
        let mut p_pl = to_u32(params.pad_l)?;
        let mut p_oh = to_u32(params.out_h)?;
        let mut p_ow = to_u32(params.out_w)?;
        let mut p_group = to_u32(params.group)?;
        let mut p_act = params.activation as u32;

        let mut args: [*mut c_void; 21] = [
            &mut p_x as *mut u64 as *mut c_void,
            &mut p_w as *mut u64 as *mut c_void,
            &mut p_bias as *mut u64 as *mut c_void,
            &mut p_y as *mut u64 as *mut c_void,
            &mut p_n as *mut u32 as *mut c_void,
            &mut p_cin as *mut u32 as *mut c_void,
            &mut p_hin as *mut u32 as *mut c_void,
            &mut p_win as *mut u32 as *mut c_void,
            &mut p_cout as *mut u32 as *mut c_void,
            &mut p_kh as *mut u32 as *mut c_void,
            &mut p_kw as *mut u32 as *mut c_void,
            &mut p_sh as *mut u32 as *mut c_void,
            &mut p_sw as *mut u32 as *mut c_void,
            &mut p_dh as *mut u32 as *mut c_void,
            &mut p_dw as *mut u32 as *mut c_void,
            &mut p_pt as *mut u32 as *mut c_void,
            &mut p_pl as *mut u32 as *mut c_void,
            &mut p_oh as *mut u32 as *mut c_void,
            &mut p_ow as *mut u32 as *mut c_void,
            &mut p_group as *mut u32 as *mut c_void,
            &mut p_act as *mut u32 as *mut c_void,
        ];

        self.launch(
            KERNEL_CONV_TRANSPOSE2D,
            grid_1d(total)?,
            (THREADS_1D, 1),
            &mut args,
            stream,
        )
    }

    /// ONNX `Gemm`: `Y = alpha * op(A) * op(B) + beta * C` with unidirectional C broadcasting.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        a_ptr: u64,
        b_ptr: u64,
        c_in_ptr: Option<u64>,
        c_in_shape: Option<&[usize]>,
        y_ptr: u64,
        a_shape: &[usize],
        b_shape: &[usize],
        attrs: &GemmAttrs,
        stream: Option<&Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("gemm rank")));
        }
        let m = if attrs.trans_a { a_shape[1] } else { a_shape[0] };
        let k_a = if attrs.trans_a { a_shape[0] } else { a_shape[1] };
        let k_b = if attrs.trans_b { b_shape[1] } else { b_shape[0] };
        let n = if attrs.trans_b { b_shape[0] } else { b_shape[1] };
        if k_a != k_b || m == 0 || n == 0 || k_a == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("gemm dims")));
        }

        let (bias_ptr, bias_mode) = resolve_gemm_bias_mode(c_in_ptr, c_in_shape, m, n)?;
        let beta = if bias_ptr.is_some() { attrs.beta } else { 0.0 };

        self.sgemm_strided_batched(
            a_ptr,
            b_ptr,
            y_ptr,
            bias_ptr,
            bias_mode,
            m,
            n,
            k_a,
            attrs.trans_a,
            attrs.trans_b,
            attrs.alpha,
            beta,
            1,
            0,
            0,
            0,
            0,
            FusedActivation::None,
            stream,
        )
    }

    /// ONNX `MatMul`: supports 2D, 3D, 4D, 5D, and 6D tensors with NumPy batch broadcasting.
    pub fn batched_matmul(
        &self,
        a_ptr: u64,
        b_ptr: u64,
        c_ptr: u64,
        a_shape: &[usize],
        b_shape: &[usize],
        stream: Option<&Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        let plan = plan_batched_matmul(a_shape, b_shape)?;

        let tiles_m = plan.m.div_ceil(GEMM_TILE);
        let tiles_n = plan.n.div_ceil(GEMM_TILE);
        let total_tiles = (plan.batch_total as u64)
            .checked_mul(tiles_m as u64)
            .and_then(|v| v.checked_mul(tiles_n as u64))
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("matmul grid")))?;
        let total_tiles_u32 = u32::try_from(total_tiles)
            .map_err(|_| CudaError::Refused(Refusal::ValueOutOfRange("matmul grid")))?;

        let mut p_a = a_ptr;
        let mut p_b = b_ptr;
        let mut p_c = c_ptr;
        let mut p_m = plan.m;
        let mut p_n = plan.n;
        let mut p_k = plan.k;
        let mut p_d0 = plan.dims[0];
        let mut p_d1 = plan.dims[1];
        let mut p_d2 = plan.dims[2];
        let mut p_d3 = plan.dims[3];
        let mut p_as0 = plan.a_strides[0];
        let mut p_as1 = plan.a_strides[1];
        let mut p_as2 = plan.a_strides[2];
        let mut p_as3 = plan.a_strides[3];
        let mut p_bs0 = plan.b_strides[0];
        let mut p_bs1 = plan.b_strides[1];
        let mut p_bs2 = plan.b_strides[2];
        let mut p_bs3 = plan.b_strides[3];

        let mut args: [*mut c_void; 18] = [
            &mut p_a as *mut u64 as *mut c_void,
            &mut p_b as *mut u64 as *mut c_void,
            &mut p_c as *mut u64 as *mut c_void,
            &mut p_m as *mut u32 as *mut c_void,
            &mut p_n as *mut u32 as *mut c_void,
            &mut p_k as *mut u32 as *mut c_void,
            &mut p_d0 as *mut u32 as *mut c_void,
            &mut p_d1 as *mut u32 as *mut c_void,
            &mut p_d2 as *mut u32 as *mut c_void,
            &mut p_d3 as *mut u32 as *mut c_void,
            &mut p_as0 as *mut u64 as *mut c_void,
            &mut p_as1 as *mut u64 as *mut c_void,
            &mut p_as2 as *mut u64 as *mut c_void,
            &mut p_as3 as *mut u64 as *mut c_void,
            &mut p_bs0 as *mut u64 as *mut c_void,
            &mut p_bs1 as *mut u64 as *mut c_void,
            &mut p_bs2 as *mut u64 as *mut c_void,
            &mut p_bs3 as *mut u64 as *mut c_void,
        ];

        self.launch(
            KERNEL_BATCHED_MATMUL,
            (total_tiles_u32, 1),
            GEMM_BLOCK,
            &mut args,
            stream,
        )
    }
}

// ===========================================================================
// Non-Windows Stub
// ===========================================================================

#[cfg(not(windows))]
pub struct ConvGemmKernels;

#[cfg(not(windows))]
impl ConvGemmKernels {
    pub fn open() -> Result<Self, CudaError> {
        Err(CudaError::Unavailable("cuda driver only supported on windows"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn im2col_nchw(
        &self,
        _x_ptr: u64,
        _col_ptr: u64,
        _c_in: usize,
        _h_in: usize,
        _w_in: usize,
        _kh: usize,
        _kw: usize,
        _sh: usize,
        _sw: usize,
        _dh: usize,
        _dw: usize,
        _pad_t: usize,
        _pad_l: usize,
        _out_h: usize,
        _out_w: usize,
        _stream: Option<&crate::cuda::Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("cuda driver only supported on windows"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_strided_batched(
        &self,
        _a_ptr: u64,
        _b_ptr: u64,
        _c_ptr: u64,
        _bias_ptr: Option<u64>,
        _bias_mode: GemmBiasMode,
        _m: usize,
        _n: usize,
        _k: usize,
        _trans_a: bool,
        _trans_b: bool,
        _alpha: f32,
        _beta: f32,
        _batch_count: usize,
        _stride_a: u64,
        _stride_b: u64,
        _stride_c: u64,
        _stride_bias: u64,
        _activation: FusedActivation,
        _stream: Option<&crate::cuda::Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("cuda driver only supported on windows"))
    }

    pub fn add_bias_nchw(
        &self,
        _y_ptr: u64,
        _bias_ptr: Option<u64>,
        _batch: usize,
        _channels: usize,
        _spatial: usize,
        _activation: FusedActivation,
        _stream: Option<&crate::cuda::Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("cuda driver only supported on windows"))
    }

    pub fn depthwise_conv2d_nchw(
        &self,
        _x_ptr: u64,
        _w_ptr: u64,
        _bias_ptr: Option<u64>,
        _y_ptr: u64,
        _params: &Conv2dParams,
        _stream: Option<&crate::cuda::Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("cuda driver only supported on windows"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_nchw(
        &self,
        _x_ptr: u64,
        _w_ptr: u64,
        _bias_ptr: Option<u64>,
        _y_ptr: u64,
        _workspace_ptr: u64,
        _params: &Conv2dParams,
        _stream: Option<&crate::cuda::Stream<'_>>,
    ) -> Result<(), CudaError> {
        Err(CudaError::Unavailable("cuda driver only supported on windows"))
    }

    pub fn conv_transpose2d_nchw(
        &self,
        _x_ptr: u64,
        _w_ptr: u64,
        _bias_ptr: Option<u64>,
        _y_ptr: u64,
        _params: &ConvTranspose2dParams,
        _stream: Option<&crate::cuda::Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("cuda driver only supported on windows"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        _a_ptr: u64,
        _b_ptr: u64,
        _c_in_ptr: Option<u64>,
        _c_in_shape: Option<&[usize]>,
        _y_ptr: u64,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _attrs: &GemmAttrs,
        _stream: Option<&crate::cuda::Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("cuda driver only supported on windows"))
    }

    pub fn batched_matmul(
        &self,
        _a_ptr: u64,
        _b_ptr: u64,
        _c_ptr: u64,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _stream: Option<&crate::cuda::Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("cuda driver only supported on windows"))
    }
}

// ===========================================================================
// Self-contained CPU Reference & CUDA Parity Tests (< 1e-4)
// ===========================================================================

#[cfg(test)]
#[allow(dead_code)]
mod tests {
    use super::*;

    #[test]
    fn params_and_broadcast_planning() {
        // Depthwise 3x3 -> zero workspace
        let dw_attrs = ConvAttrs {
            kernel_shape: Some(vec![3, 3]),
            strides: vec![1, 1],
            pads: vec![1, 1, 1, 1],
            dilations: vec![1, 1],
            group: 32,
        };
        let dw = Conv2dParams::from_attrs(
            &[1, 32, 64, 64],
            &[32, 1, 3, 3],
            &dw_attrs,
            FusedActivation::None,
        )
        .unwrap();
        assert!(dw.is_depthwise());
        assert_eq!(dw.im2col_workspace_bytes(), 0);

        // Direct 1x1 -> zero workspace
        let pw_attrs = ConvAttrs {
            kernel_shape: None,
            strides: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            dilations: vec![1, 1],
            group: 1,
        };
        let pw = Conv2dParams::from_attrs(
            &[1, 64, 32, 32],
            &[128, 64, 1, 1],
            &pw_attrs,
            FusedActivation::Relu,
        )
        .unwrap();
        assert!(pw.is_direct_1x1());
        assert_eq!(pw.im2col_workspace_bytes(), 0);

        // 4D MatMul broadcast: [2, 1, 16, 32] x [1, 4, 32, 8] -> dims [1, 1, 2, 4]
        let plan = plan_batched_matmul(&[2, 1, 16, 32], &[1, 4, 32, 8]).unwrap();
        assert_eq!(plan.dims, [1, 1, 2, 4]);
        assert_eq!(plan.batch_total, 8);
        assert_eq!(plan.a_strides, [0, 0, 16 * 32, 0]);
        assert_eq!(plan.b_strides, [0, 0, 0, 32 * 8]);
    }

    fn det_vec(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(747796405).wrapping_add(2891336453);
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = ((s >> 8) as f32) / ((1u32 << 24) as f32);
                (u - 0.5) * 2.0
            })
            .collect()
    }

    fn apply_act_cpu(v: f32, act: FusedActivation) -> f32 {
        match act {
            FusedActivation::None => v,
            FusedActivation::Relu => v.max(0.0),
            FusedActivation::Silu => v / (1.0 + (-v).exp()),
        }
    }

    fn cpu_conv2d_nchw(
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        p: &Conv2dParams,
    ) -> Vec<f32> {
        let c_in_g = p.c_in / p.group;
        let c_out_g = p.c_out / p.group;
        let mut y = vec![0.0f32; p.batch * p.c_out * p.out_h * p.out_w];

        for ni in 0..p.batch {
            for co in 0..p.c_out {
                let g = co / c_out_g;
                let ci_start = g * c_in_g;
                for oh in 0..p.out_h {
                    for ow in 0..p.out_w {
                        let mut acc = bias.map_or(0.0, |b| b[co]);
                        for ci_i in 0..c_in_g {
                            let ci = ci_start + ci_i;
                            for kh_i in 0..p.kh {
                                let ih = (oh * p.sh) as isize - p.pad_t as isize
                                    + (kh_i * p.dh) as isize;
                                if ih < 0 || ih >= p.h_in as isize {
                                    continue;
                                }
                                for kw_i in 0..p.kw {
                                    let iw = (ow * p.sw) as isize - p.pad_l as isize
                                        + (kw_i * p.dw) as isize;
                                    if iw < 0 || iw >= p.w_in as isize {
                                        continue;
                                    }
                                    let x_idx = ((ni * p.c_in + ci) * p.h_in + ih as usize)
                                        * p.w_in
                                        + iw as usize;
                                    let w_idx = ((co * c_in_g + ci_i) * p.kh + kh_i) * p.kw + kw_i;
                                    acc += x[x_idx] * w[w_idx];
                                }
                            }
                        }
                        let y_idx = ((ni * p.c_out + co) * p.out_h + oh) * p.out_w + ow;
                        y[y_idx] = apply_act_cpu(acc, p.activation);
                    }
                }
            }
        }
        y
    }

    fn cpu_conv_transpose2d_nchw(
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        p: &ConvTranspose2dParams,
    ) -> Vec<f32> {
        let c_in_g = p.c_in / p.group;
        let c_out_g = p.c_out / p.group;
        let mut y = vec![0.0f32; p.batch * p.c_out * p.out_h * p.out_w];

        for ni in 0..p.batch {
            for co in 0..p.c_out {
                let g = co / c_out_g;
                let co_g = co % c_out_g;
                let ci_start = g * c_in_g;
                for oh in 0..p.out_h {
                    for ow in 0..p.out_w {
                        let mut acc = bias.map_or(0.0, |b| b[co]);
                        for kh_i in 0..p.kh {
                            let h_num = oh as isize + p.pad_t as isize - (kh_i * p.dh) as isize;
                            if h_num < 0 || h_num % (p.sh as isize) != 0 {
                                continue;
                            }
                            let ih = (h_num / p.sh as isize) as usize;
                            if ih >= p.h_in {
                                continue;
                            }
                            for kw_i in 0..p.kw {
                                let w_num =
                                    ow as isize + p.pad_l as isize - (kw_i * p.dw) as isize;
                                if w_num < 0 || w_num % (p.sw as isize) != 0 {
                                    continue;
                                }
                                let iw = (w_num / p.sw as isize) as usize;
                                if iw >= p.w_in {
                                    continue;
                                }
                                for ci_i in 0..c_in_g {
                                    let ci = ci_start + ci_i;
                                    let x_idx = ((ni * p.c_in + ci) * p.h_in + ih) * p.w_in + iw;
                                    let w_idx =
                                        ((ci * c_out_g + co_g) * p.kh + kh_i) * p.kw + kw_i;
                                    acc += x[x_idx] * w[w_idx];
                                }
                            }
                        }
                        let y_idx = ((ni * p.c_out + co) * p.out_h + oh) * p.out_w + ow;
                        y[y_idx] = apply_act_cpu(acc, p.activation);
                    }
                }
            }
        }
        y
    }

    fn cpu_gemm(
        a: &[f32],
        b: &[f32],
        c: Option<&[f32]>,
        c_shape: Option<&[usize]>,
        a_shape: &[usize],
        b_shape: &[usize],
        attrs: &GemmAttrs,
    ) -> Vec<f32> {
        let m = if attrs.trans_a { a_shape[1] } else { a_shape[0] };
        let k = if attrs.trans_a { a_shape[0] } else { a_shape[1] };
        let n = if attrs.trans_b { b_shape[0] } else { b_shape[1] };
        let (_, mode) = resolve_gemm_bias_mode(c.map(|_| 1), c_shape, m, n).unwrap();

        let mut out = vec![0.0f32; m * n];
        for r in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let av = if attrs.trans_a {
                        a[kk * m + r]
                    } else {
                        a[r * k + kk]
                    };
                    let bv = if attrs.trans_b {
                        b[col * k + kk]
                    } else {
                        b[kk * n + col]
                    };
                    acc += av * bv;
                }
                let mut val = attrs.alpha * acc;
                if let Some(cb) = c {
                    let b_term = match mode {
                        GemmBiasMode::None => 0.0,
                        GemmBiasMode::RowBroadcast => cb[r],
                        GemmBiasMode::ColBroadcast => cb[col],
                        GemmBiasMode::ScalarBroadcast => cb[0],
                        GemmBiasMode::Full2D => cb[r * n + col],
                    };
                    val += attrs.beta * b_term;
                }
                out[r * n + col] = val;
            }
        }
        out
    }

    fn cpu_batched_matmul(a: &[f32], b: &[f32], a_shape: &[usize], b_shape: &[usize]) -> Vec<f32> {
        let plan = plan_batched_matmul(a_shape, b_shape).unwrap();
        let m = plan.m as usize;
        let n = plan.n as usize;
        let k = plan.k as usize;
        let mut out = vec![0.0f32; plan.batch_total as usize * m * n];

        for b_idx in 0..plan.batch_total {
            let mut rem = b_idx;
            let i3 = (rem % plan.dims[3]) as usize;
            rem /= plan.dims[3];
            let i2 = (rem % plan.dims[2]) as usize;
            rem /= plan.dims[2];
            let i1 = (rem % plan.dims[1]) as usize;
            let i0 = (rem / plan.dims[1]) as usize;

            let off_a = i0 * plan.a_strides[0] as usize
                + i1 * plan.a_strides[1] as usize
                + i2 * plan.a_strides[2] as usize
                + i3 * plan.a_strides[3] as usize;
            let off_b = i0 * plan.b_strides[0] as usize
                + i1 * plan.b_strides[1] as usize
                + i2 * plan.b_strides[2] as usize
                + i3 * plan.b_strides[3] as usize;
            let off_c = b_idx as usize * m * n;

            for r in 0..m {
                for c_i in 0..n {
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        acc += a[off_a + r * k + kk] * b[off_b + kk * n + c_i];
                    }
                    out[off_c + r * n + c_i] = acc;
                }
            }
        }
        out
    }

    fn f32_as_bytes(slice: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) }
    }

    fn f32_as_bytes_mut(slice: &mut [f32]) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u8, slice.len() * 4) }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[cfg(windows)]
    fn run_gpu_conv2d(
        kernels: &ConvGemmKernels,
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        p: &Conv2dParams,
    ) -> Vec<f32> {
        let nv = kernels.nvidia();
        let d_x = nv.alloc_buffer(x.len() * 4).unwrap();
        let d_w = nv.alloc_buffer(w.len() * 4).unwrap();
        let d_b = bias.map(|b| {
            let buf = nv.alloc_buffer(b.len() * 4).unwrap();
            nv.upload_buffer(&buf, f32_as_bytes(b)).unwrap();
            buf
        });
        let out_len = p.batch * p.c_out * p.out_h * p.out_w;
        let d_y = nv.alloc_buffer(out_len * 4).unwrap();
        let ws_bytes = p.im2col_workspace_bytes();
        let d_ws = if ws_bytes > 0 {
            Some(nv.alloc_buffer(ws_bytes).unwrap())
        } else {
            None
        };

        nv.upload_buffer(&d_x, f32_as_bytes(x)).unwrap();
        nv.upload_buffer(&d_w, f32_as_bytes(w)).unwrap();

        kernels
            .conv2d_nchw(
                d_x.handle(),
                d_w.handle(),
                d_b.as_ref().map(|b| b.handle()),
                d_y.handle(),
                d_ws.as_ref().map_or(0, |b| b.handle()),
                p,
                None,
            )
            .unwrap();

        let mut out = vec![0.0f32; out_len];
        nv.download_buffer(&d_y, f32_as_bytes_mut(&mut out)).unwrap();
        out
    }

    #[cfg(windows)]
    #[test]
    fn cuda_conv2d_standard_3x3_1x1_and_dilated_parity() {
        let Ok(kernels) = ConvGemmKernels::open() else {
            eprintln!("skipping CUDA test: no NVIDIA driver available");
            return;
        };

        // 1) Standard 3x3 stride=1 pad=1 + bias + SiLU
        let attrs_3x3 = ConvAttrs {
            kernel_shape: Some(vec![3, 3]),
            strides: vec![1, 1],
            pads: vec![1, 1, 1, 1],
            dilations: vec![1, 1],
            group: 1,
        };
        let p_3x3 = Conv2dParams::from_attrs(
            &[2, 7, 19, 23],
            &[13, 7, 3, 3],
            &attrs_3x3,
            FusedActivation::Silu,
        )
        .unwrap();
        let x = det_vec(2 * 7 * 19 * 23, 11);
        let w = det_vec(13 * 7 * 3 * 3, 12);
        let b = det_vec(13, 13);
        let gpu = run_gpu_conv2d(&kernels, &x, &w, Some(&b), &p_3x3);
        let cpu = cpu_conv2d_nchw(&x, &w, Some(&b), &p_3x3);
        let diff = max_abs_diff(&gpu, &cpu);
        assert!(diff < 1e-4, "conv3x3 max_abs_diff={diff}");

        // 2) Direct 1x1 pointwise conv (zero workspace) + ReLU
        let attrs_1x1 = ConvAttrs {
            kernel_shape: Some(vec![1, 1]),
            strides: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            dilations: vec![1, 1],
            group: 1,
        };
        let p_1x1 = Conv2dParams::from_attrs(
            &[1, 35, 17, 19],
            &[41, 35, 1, 1],
            &attrs_1x1,
            FusedActivation::Relu,
        )
        .unwrap();
        assert_eq!(p_1x1.im2col_workspace_bytes(), 0);
        let x = det_vec(35 * 17 * 19, 21);
        let w = det_vec(41 * 35, 22);
        let b = det_vec(41, 23);
        let gpu = run_gpu_conv2d(&kernels, &x, &w, Some(&b), &p_1x1);
        let cpu = cpu_conv2d_nchw(&x, &w, Some(&b), &p_1x1);
        let diff = max_abs_diff(&gpu, &cpu);
        assert!(diff < 1e-4, "conv1x1 max_abs_diff={diff}");

        // 3) Dilated 3x3 dilation=2 asymmetric pad + stride=2
        let attrs_dil = ConvAttrs {
            kernel_shape: Some(vec![3, 3]),
            strides: vec![2, 2],
            pads: vec![2, 1, 2, 2],
            dilations: vec![2, 2],
            group: 1,
        };
        let p_dil = Conv2dParams::from_attrs(
            &[1, 5, 21, 18],
            &[9, 5, 3, 3],
            &attrs_dil,
            FusedActivation::None,
        )
        .unwrap();
        let x = det_vec(5 * 21 * 18, 31);
        let w = det_vec(9 * 5 * 3 * 3, 32);
        let gpu = run_gpu_conv2d(&kernels, &x, &w, None, &p_dil);
        let cpu = cpu_conv2d_nchw(&x, &w, None, &p_dil);
        let diff = max_abs_diff(&gpu, &cpu);
        assert!(diff < 1e-4, "conv_dilated max_abs_diff={diff}");
    }

    #[cfg(windows)]
    #[test]
    fn cuda_depthwise_and_grouped_conv2d_parity() {
        let Ok(kernels) = ConvGemmKernels::open() else {
            eprintln!("skipping CUDA test: no NVIDIA driver available");
            return;
        };

        // 1) Depthwise 5x5 stride=2 pad=2 (zero workspace)
        let attrs_dw = ConvAttrs {
            kernel_shape: Some(vec![5, 5]),
            strides: vec![2, 2],
            pads: vec![2, 2, 2, 2],
            dilations: vec![1, 1],
            group: 16,
        };
        let p_dw = Conv2dParams::from_attrs(
            &[2, 16, 25, 29],
            &[16, 1, 5, 5],
            &attrs_dw,
            FusedActivation::None,
        )
        .unwrap();
        assert!(p_dw.is_depthwise());
        assert_eq!(p_dw.im2col_workspace_bytes(), 0);
        let x = det_vec(2 * 16 * 25 * 29, 41);
        let w = det_vec(16 * 5 * 5, 42);
        let b = det_vec(16, 43);
        let gpu = run_gpu_conv2d(&kernels, &x, &w, Some(&b), &p_dw);
        let cpu = cpu_conv2d_nchw(&x, &w, Some(&b), &p_dw);
        let diff = max_abs_diff(&gpu, &cpu);
        assert!(diff < 1e-4, "depthwise max_abs_diff={diff}");

        // 2) Grouped Conv (group=4, C_in=12, C_out=20, 3x3)
        let attrs_grp = ConvAttrs {
            kernel_shape: Some(vec![3, 3]),
            strides: vec![1, 1],
            pads: vec![1, 1, 1, 1],
            dilations: vec![1, 1],
            group: 4,
        };
        let p_grp = Conv2dParams::from_attrs(
            &[2, 12, 15, 17],
            &[20, 3, 3, 3],
            &attrs_grp,
            FusedActivation::Relu,
        )
        .unwrap();
        let x = det_vec(2 * 12 * 15 * 17, 51);
        let w = det_vec(20 * 3 * 3 * 3, 52);
        let b = det_vec(20, 53);
        let gpu = run_gpu_conv2d(&kernels, &x, &w, Some(&b), &p_grp);
        let cpu = cpu_conv2d_nchw(&x, &w, Some(&b), &p_grp);
        let diff = max_abs_diff(&gpu, &cpu);
        assert!(diff < 1e-4, "grouped conv max_abs_diff={diff}");
    }

    #[cfg(windows)]
    #[test]
    fn cuda_conv_transpose2d_2x2_s2_parity() {
        let Ok(kernels) = ConvGemmKernels::open() else {
            eprintln!("skipping CUDA test: no NVIDIA driver available");
            return;
        };

        let attrs = ConvTransposeAttrs {
            kernel_shape: Some(vec![2, 2]),
            strides: vec![2, 2],
            pads: vec![0, 0, 0, 0],
            dilations: vec![1, 1],
            group: 1,
            output_padding: vec![0, 0],
        };
        let p = ConvTranspose2dParams::from_attrs(
            &[2, 12, 16, 16],
            &[12, 8, 2, 2],
            &attrs,
            FusedActivation::None,
        )
        .unwrap();
        assert_eq!((p.out_h, p.out_w), (32, 32));

        let x = det_vec(2 * 12 * 16 * 16, 61);
        let w = det_vec(12 * 8 * 2 * 2, 62);
        let b = det_vec(8, 63);

        let nv = kernels.nvidia();
        let d_x = nv.alloc_buffer(x.len() * 4).unwrap();
        let d_w = nv.alloc_buffer(w.len() * 4).unwrap();
        let d_b = nv.alloc_buffer(b.len() * 4).unwrap();
        let out_len = p.batch * p.c_out * p.out_h * p.out_w;
        let d_y = nv.alloc_buffer(out_len * 4).unwrap();
        nv.upload_buffer(&d_x, f32_as_bytes(&x)).unwrap();
        nv.upload_buffer(&d_w, f32_as_bytes(&w)).unwrap();
        nv.upload_buffer(&d_b, f32_as_bytes(&b)).unwrap();

        kernels
            .conv_transpose2d_nchw(
                d_x.handle(),
                d_w.handle(),
                Some(d_b.handle()),
                d_y.handle(),
                &p,
                None,
            )
            .unwrap();

        let mut gpu = vec![0.0f32; out_len];
        nv.download_buffer(&d_y, f32_as_bytes_mut(&mut gpu)).unwrap();
        let cpu = cpu_conv_transpose2d_nchw(&x, &w, Some(&b), &p);
        let diff = max_abs_diff(&gpu, &cpu);
        assert!(diff < 1e-4, "conv_transpose 2x2 s=2 max_abs_diff={diff}");
    }

    #[cfg(windows)]
    #[test]
    fn cuda_batched_matmul_3d_4d_and_gemm_transb_parity() {
        let Ok(kernels) = ConvGemmKernels::open() else {
            eprintln!("skipping CUDA test: no NVIDIA driver available");
            return;
        };
        let nv = kernels.nvidia();

        // 1) 3D Batched MatMul with broadcast: [3, 37, 41] x [1, 41, 35] -> [3, 37, 35]
        {
            let a_shape = [3, 37, 41];
            let b_shape = [1, 41, 35];
            let a = det_vec(3 * 37 * 41, 71);
            let b = det_vec(41 * 35, 72);
            let out_len = 3 * 37 * 35;
            let d_a = nv.alloc_buffer(a.len() * 4).unwrap();
            let d_b = nv.alloc_buffer(b.len() * 4).unwrap();
            let d_c = nv.alloc_buffer(out_len * 4).unwrap();
            nv.upload_buffer(&d_a, f32_as_bytes(&a)).unwrap();
            nv.upload_buffer(&d_b, f32_as_bytes(&b)).unwrap();

            kernels
                .batched_matmul(d_a.handle(), d_b.handle(), d_c.handle(), &a_shape, &b_shape, None)
                .unwrap();

            let mut gpu = vec![0.0f32; out_len];
            nv.download_buffer(&d_c, f32_as_bytes_mut(&mut gpu)).unwrap();
            let cpu = cpu_batched_matmul(&a, &b, &a_shape, &b_shape);
            let diff = max_abs_diff(&gpu, &cpu);
            assert!(diff < 1e-4, "matmul 3D broadcast max_abs_diff={diff}");
        }

        // 2) 4D Multi-Head Attention MatMul: [2, 6, 49, 64] x [2, 6, 64, 49] -> [2, 6, 49, 49]
        {
            let a_shape = [2, 6, 49, 64];
            let b_shape = [2, 6, 64, 49];
            let a = det_vec(2 * 6 * 49 * 64, 81);
            let b = det_vec(2 * 6 * 64 * 49, 82);
            let out_len = 2 * 6 * 49 * 49;
            let d_a = nv.alloc_buffer(a.len() * 4).unwrap();
            let d_b = nv.alloc_buffer(b.len() * 4).unwrap();
            let d_c = nv.alloc_buffer(out_len * 4).unwrap();
            nv.upload_buffer(&d_a, f32_as_bytes(&a)).unwrap();
            nv.upload_buffer(&d_b, f32_as_bytes(&b)).unwrap();

            kernels
                .batched_matmul(d_a.handle(), d_b.handle(), d_c.handle(), &a_shape, &b_shape, None)
                .unwrap();

            let mut gpu = vec![0.0f32; out_len];
            nv.download_buffer(&d_c, f32_as_bytes_mut(&mut gpu)).unwrap();
            let cpu = cpu_batched_matmul(&a, &b, &a_shape, &b_shape);
            let diff = max_abs_diff(&gpu, &cpu);
            assert!(diff < 1e-4, "matmul 4D max_abs_diff={diff}");
        }

        // 3) ONNX Gemm with transB=1, alpha=1.0, beta=1.0, 1D bias [N]
        {
            let a_shape = [35, 73];
            let b_shape = [68, 73]; // trans_b=true -> [73, 68]
            let c_shape = [68];
            let attrs = GemmAttrs {
                alpha: 1.0,
                beta: 1.0,
                trans_a: false,
                trans_b: true,
            };
            let a = det_vec(35 * 73, 91);
            let b = det_vec(68 * 73, 92);
            let c = det_vec(68, 93);
            let out_len = 35 * 68;

            let d_a = nv.alloc_buffer(a.len() * 4).unwrap();
            let d_b = nv.alloc_buffer(b.len() * 4).unwrap();
            let d_c = nv.alloc_buffer(c.len() * 4).unwrap();
            let d_y = nv.alloc_buffer(out_len * 4).unwrap();
            nv.upload_buffer(&d_a, f32_as_bytes(&a)).unwrap();
            nv.upload_buffer(&d_b, f32_as_bytes(&b)).unwrap();
            nv.upload_buffer(&d_c, f32_as_bytes(&c)).unwrap();

            kernels
                .gemm(
                    d_a.handle(),
                    d_b.handle(),
                    Some(d_c.handle()),
                    Some(&c_shape),
                    d_y.handle(),
                    &a_shape,
                    &b_shape,
                    &attrs,
                    None,
                )
                .unwrap();

            let mut gpu = vec![0.0f32; out_len];
            nv.download_buffer(&d_y, f32_as_bytes_mut(&mut gpu)).unwrap();
            let cpu = cpu_gemm(&a, &b, Some(&c), Some(&c_shape), &a_shape, &b_shape, &attrs);
            let diff = max_abs_diff(&gpu, &cpu);
            assert!(diff < 1e-4, "gemm transB=1 max_abs_diff={diff}");
        }
    }
}
