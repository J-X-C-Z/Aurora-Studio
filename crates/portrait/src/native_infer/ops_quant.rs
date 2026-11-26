//! CUDA UInt8 dynamic quantization operator bindings for Aurora Native Inference.
//!
//! Covers the 3 quantized ONNX operators used exclusively by `eSamEncoderPreUInt8.onnx`:
//! - `DynamicQuantizeLinear` (`dynamic_quantize_linear_minmax_f32` + `dynamic_quantize_linear_apply_f32_to_u8`)
//! - `MatMulInteger` (`matmul_integer_u8_to_i32`, 2D/3D/4D batched U8 × U8 → I32)
//! - `ConvInteger` (`conv_integer_u8_to_i32`, 2D NCHW U8 × U8 → I32)
//!
//! All GPU launch methods accept raw CUDA device pointers (`u64`) so callers
//! (`NativeCudaSession` / VRAM arena) can dispatch directly without temporary copies.

use crate::cuda::{CudaError, Dispatch, Stream};
#[cfg(windows)]
use crate::cuda::Nvidia;
#[cfg(windows)]
use std::ffi::c_void;

/// Embedded PTX compiled from `native/cuda/infer_quant_int8.cu` (`-arch=compute_75`).
pub const QUANT_INT8_PTX: &str = include_str!("../../native/cuda/infer_quant_int8.ptx");

pub const KERNEL_DQL_MINMAX: &str = "dynamic_quantize_linear_minmax_f32";
pub const KERNEL_DQL_APPLY: &str = "dynamic_quantize_linear_apply_f32_to_u8";
pub const KERNEL_MATMUL_INT_U8: &str = "matmul_integer_u8_to_i32";
pub const KERNEL_CONV_INT_U8: &str = "conv_integer_u8_to_i32";

/// Optional zero-point device pointer mode for quantized integer kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZeroPointArg {
    /// Omitted (implicitly 0).
    None,
    /// Single scalar `u8` at device address.
    Scalar(u64),
    /// 1D per-row (`[M]` for A), per-column (`[N]` for B), or per-channel (`[C_out]` for W).
    PerAxis(u64),
    /// 2D batched per-row (`[batch, M]`) or batched per-column (`[batch, N]`).
    BatchedPerAxis(u64),
}

impl ZeroPointArg {
    #[inline]
    pub const fn ptr_and_mode(self) -> (u64, i32) {
        match self {
            Self::None => (0, 0),
            Self::Scalar(p) => (p, 1),
            Self::PerAxis(p) => (p, 2),
            Self::BatchedPerAxis(p) => (p, 3),
        }
    }
}

/// Launch parameters for `matmul_integer_u8_to_i32` (2D / 3D / 4D batched).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatMulIntegerParams {
    pub a_ptr: u64,
    pub b_ptr: u64,
    pub y_ptr: u64,
    pub a_zero_point: ZeroPointArg,
    pub b_zero_point: ZeroPointArg,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// Outer batch dimension (`1` for 2D/3D).
    pub d0: usize,
    /// Inner batch dimension (`1` for 2D, `B` for 3D).
    pub d1: usize,
    /// Element stride in `A` along `d0` (0 if broadcast).
    pub a_stride0: i64,
    /// Element stride in `A` along `d1` (0 if broadcast).
    pub a_stride1: i64,
    /// Element stride in `B` along `d0` (0 if broadcast).
    pub b_stride0: i64,
    /// Element stride in `B` along `d1` (0 if broadcast).
    pub b_stride1: i64,
}

impl MatMulIntegerParams {
    /// Convenience builder for standard 2D `A[M, K] × B[K, N] -> Y[M, N]`.
    pub fn new_2d(
        a_ptr: u64,
        b_ptr: u64,
        y_ptr: u64,
        m: usize,
        n: usize,
        k: usize,
        a_zero_point: ZeroPointArg,
        b_zero_point: ZeroPointArg,
    ) -> Self {
        Self {
            a_ptr,
            b_ptr,
            y_ptr,
            a_zero_point,
            b_zero_point,
            m,
            n,
            k,
            d0: 1,
            d1: 1,
            a_stride0: 0,
            a_stride1: 0,
            b_stride0: 0,
            b_stride1: 0,
        }
    }

    /// Convenience builder for 3D/4D batched `MatMulInteger` with NumPy broadcast
    /// over leading batch dimensions `(a_batch, b_batch)` (length 0..=2).
    pub fn new_batched(
        a_ptr: u64,
        b_ptr: u64,
        y_ptr: u64,
        a_batch: &[usize],
        b_batch: &[usize],
        m: usize,
        n: usize,
        k: usize,
        a_zero_point: ZeroPointArg,
        b_zero_point: ZeroPointArg,
    ) -> Result<Self, CudaError> {
        let (a0, a1) = match a_batch {
            [] => (1, 1),
            [x] => (1, *x),
            [x, y] => (*x, *y),
            _ => return Err(CudaError::Unavailable("MatMulInteger supports up to 4D")),
        };
        let (b0, b1) = match b_batch {
            [] => (1, 1),
            [x] => (1, *x),
            [x, y] => (*x, *y),
            _ => return Err(CudaError::Unavailable("MatMulInteger supports up to 4D")),
        };
        let d0 = if a0 == b0 {
            a0
        } else if a0 == 1 {
            b0
        } else if b0 == 1 {
            a0
        } else {
            return Err(CudaError::Unavailable("MatMulInteger batch d0 broadcast mismatch"));
        };
        let d1 = if a1 == b1 {
            a1
        } else if a1 == 1 {
            b1
        } else if b1 == 1 {
            a1
        } else {
            return Err(CudaError::Unavailable("MatMulInteger batch d1 broadcast mismatch"));
        };

        let mk = (m * k) as i64;
        let kn = (k * n) as i64;
        let a_stride1 = if a1 == 1 { 0 } else { mk };
        let a_stride0 = if a0 == 1 { 0 } else { (a1 as i64) * mk };
        let b_stride1 = if b1 == 1 { 0 } else { kn };
        let b_stride0 = if b0 == 1 { 0 } else { (b1 as i64) * kn };

        Ok(Self {
            a_ptr,
            b_ptr,
            y_ptr,
            a_zero_point,
            b_zero_point,
            m,
            n,
            k,
            d0,
            d1,
            a_stride0,
            a_stride1,
            b_stride0,
            b_stride1,
        })
    }
}

/// Launch parameters for `conv_integer_u8_to_i32` (NCHW 2D convolution).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvIntegerParams {
    pub x_ptr: u64,
    pub w_ptr: u64,
    pub y_ptr: u64,
    pub x_zero_point: ZeroPointArg,
    pub w_zero_point: ZeroPointArg,
    pub n: usize,
    pub c_in: usize,
    pub h_in: usize,
    pub w_in: usize,
    pub c_out: usize,
    pub k_h: usize,
    pub k_w: usize,
    pub h_out: usize,
    pub w_out: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub group: usize,
}

// ============================================================================
// Windows CUDA implementation
// ============================================================================

#[cfg(windows)]
pub struct QuantInt8Kernels {
    gpu: Nvidia,
}

#[cfg(windows)]
impl QuantInt8Kernels {
    /// Load and JIT-compile the embedded `infer_quant_int8.ptx` module on the shared CUDA context.
    pub fn open() -> Result<Self, CudaError> {
        let gpu = Nvidia::open_ptx(QUANT_INT8_PTX)?;
        Ok(Self { gpu })
    }

    /// Borrow the underlying [`Nvidia`] driver handle (useful for buffer allocation/transfer in tests).
    pub fn nvidia(&self) -> &Nvidia {
        &self.gpu
    }

    /// Execute ONNX `DynamicQuantizeLinear` on the default stream:
    /// `x_ptr` (`[n]` f32) -> `y_ptr` (`[n]` u8), `y_scale_ptr` (`[1]` f32), `y_zero_point_ptr` (`[1]` u8).
    pub fn dynamic_quantize_linear(
        &self,
        x_ptr: u64,
        n: usize,
        y_ptr: u64,
        y_scale_ptr: u64,
        y_zero_point_ptr: u64,
    ) -> Result<(Dispatch, Dispatch), CudaError> {
        self.dynamic_quantize_linear_impl(x_ptr, n, y_ptr, y_scale_ptr, y_zero_point_ptr, None)
    }

    /// Execute ONNX `DynamicQuantizeLinear` on an explicit CUDA stream.
    pub fn dynamic_quantize_linear_on_stream(
        &self,
        x_ptr: u64,
        n: usize,
        y_ptr: u64,
        y_scale_ptr: u64,
        y_zero_point_ptr: u64,
        stream: &Stream<'_>,
    ) -> Result<(Dispatch, Dispatch), CudaError> {
        self.dynamic_quantize_linear_impl(
            x_ptr,
            n,
            y_ptr,
            y_scale_ptr,
            y_zero_point_ptr,
            Some(stream),
        )
    }

    fn dynamic_quantize_linear_impl(
        &self,
        x_ptr: u64,
        n: usize,
        y_ptr: u64,
        y_scale_ptr: u64,
        y_zero_point_ptr: u64,
        stream: Option<&Stream<'_>>,
    ) -> Result<(Dispatch, Dispatch), CudaError> {
        let mut x_arg = x_ptr;
        let mut n_arg = n as i32;
        let mut scale_arg = y_scale_ptr;
        let mut zp_arg = y_zero_point_ptr;
        let mut minmax_params: [*mut c_void; 4] = [
            &mut x_arg as *mut u64 as *mut c_void,
            &mut n_arg as *mut i32 as *mut c_void,
            &mut scale_arg as *mut u64 as *mut c_void,
            &mut zp_arg as *mut u64 as *mut c_void,
        ];
        let d1 = match stream {
            Some(s) => self.gpu.launch_raw_stream(
                KERNEL_DQL_MINMAX,
                (1, 1),
                (256, 1),
                &mut minmax_params,
                s,
            )?,
            None => self
                .gpu
                .launch_raw(KERNEL_DQL_MINMAX, (1, 1), (256, 1), &mut minmax_params)?,
        };

        let blocks = ((n as u32).saturating_add(255) / 256).clamp(1, 65535);
        let mut y_arg = y_ptr;
        let mut apply_params: [*mut c_void; 5] = [
            &mut x_arg as *mut u64 as *mut c_void,
            &mut n_arg as *mut i32 as *mut c_void,
            &mut scale_arg as *mut u64 as *mut c_void,
            &mut zp_arg as *mut u64 as *mut c_void,
            &mut y_arg as *mut u64 as *mut c_void,
        ];
        let d2 = match stream {
            Some(s) => self.gpu.launch_raw_stream(
                KERNEL_DQL_APPLY,
                (blocks, 1),
                (256, 1),
                &mut apply_params,
                s,
            )?,
            None => self
                .gpu
                .launch_raw(KERNEL_DQL_APPLY, (blocks, 1), (256, 1), &mut apply_params)?,
        };

        Ok((d1, d2))
    }

    /// Execute ONNX `MatMulInteger` on the default stream.
    pub fn matmul_integer(&self, p: &MatMulIntegerParams) -> Result<Dispatch, CudaError> {
        self.matmul_integer_impl(p, None)
    }

    /// Execute ONNX `MatMulInteger` on an explicit CUDA stream.
    pub fn matmul_integer_on_stream(
        &self,
        p: &MatMulIntegerParams,
        stream: &Stream<'_>,
    ) -> Result<Dispatch, CudaError> {
        self.matmul_integer_impl(p, Some(stream))
    }

    fn matmul_integer_impl(
        &self,
        p: &MatMulIntegerParams,
        stream: Option<&Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        let batch = (p.d0 * p.d1) as u32;
        if p.m == 0 || p.n == 0 || batch == 0 {
            return Err(CudaError::Unavailable("MatMulInteger empty output"));
        }
        let tiles_m = ((p.m + 15) / 16) as u32;
        let tiles_n = ((p.n + 15) / 16) as u32;
        let grid = (tiles_m.saturating_mul(tiles_n).max(1), batch);
        let block = (16, 16);

        let (mut a_zp_ptr, mut a_zp_mode) = p.a_zero_point.ptr_and_mode();
        let (mut b_zp_ptr, mut b_zp_mode) = p.b_zero_point.ptr_and_mode();
        let mut a_ptr = p.a_ptr;
        let mut b_ptr = p.b_ptr;
        let mut y_ptr = p.y_ptr;
        let mut m = p.m as i32;
        let mut n = p.n as i32;
        let mut k = p.k as i32;
        let mut d0 = p.d0 as i32;
        let mut d1 = p.d1 as i32;
        let mut a_s0 = p.a_stride0;
        let mut a_s1 = p.a_stride1;
        let mut b_s0 = p.b_stride0;
        let mut b_s1 = p.b_stride1;

        let mut params: [*mut c_void; 16] = [
            &mut a_ptr as *mut u64 as *mut c_void,
            &mut b_ptr as *mut u64 as *mut c_void,
            &mut a_zp_ptr as *mut u64 as *mut c_void,
            &mut b_zp_ptr as *mut u64 as *mut c_void,
            &mut y_ptr as *mut u64 as *mut c_void,
            &mut m as *mut i32 as *mut c_void,
            &mut n as *mut i32 as *mut c_void,
            &mut k as *mut i32 as *mut c_void,
            &mut d0 as *mut i32 as *mut c_void,
            &mut d1 as *mut i32 as *mut c_void,
            &mut a_s0 as *mut i64 as *mut c_void,
            &mut a_s1 as *mut i64 as *mut c_void,
            &mut b_s0 as *mut i64 as *mut c_void,
            &mut b_s1 as *mut i64 as *mut c_void,
            &mut a_zp_mode as *mut i32 as *mut c_void,
            &mut b_zp_mode as *mut i32 as *mut c_void,
        ];

        match stream {
            Some(s) => self
                .gpu
                .launch_raw_stream(KERNEL_MATMUL_INT_U8, grid, block, &mut params, s),
            None => self
                .gpu
                .launch_raw(KERNEL_MATMUL_INT_U8, grid, block, &mut params),
        }
    }

    /// Execute ONNX `ConvInteger` on the default stream.
    pub fn conv_integer(&self, p: &ConvIntegerParams) -> Result<Dispatch, CudaError> {
        self.conv_integer_impl(p, None)
    }

    /// Execute ONNX `ConvInteger` on an explicit CUDA stream.
    pub fn conv_integer_on_stream(
        &self,
        p: &ConvIntegerParams,
        stream: &Stream<'_>,
    ) -> Result<Dispatch, CudaError> {
        self.conv_integer_impl(p, Some(stream))
    }

    fn conv_integer_impl(
        &self,
        p: &ConvIntegerParams,
        stream: Option<&Stream<'_>>,
    ) -> Result<Dispatch, CudaError> {
        let hw_out = p.h_out * p.w_out;
        let noc = p.n * p.c_out;
        if hw_out == 0 || noc == 0 || p.group == 0 {
            return Err(CudaError::Unavailable("ConvInteger empty output or group=0"));
        }
        let grid_x = ((hw_out + 255) / 256) as u32;
        let grid_y = noc as u32;
        let grid = (grid_x.max(1), grid_y);
        let block = (256, 1);

        let (mut x_zp_ptr, mut x_zp_mode) = p.x_zero_point.ptr_and_mode();
        let (mut w_zp_ptr, mut w_zp_mode) = p.w_zero_point.ptr_and_mode();
        let mut x_ptr = p.x_ptr;
        let mut w_ptr = p.w_ptr;
        let mut y_ptr = p.y_ptr;
        let mut batch_n = p.n as i32;
        let mut c_in = p.c_in as i32;
        let mut h_in = p.h_in as i32;
        let mut w_in = p.w_in as i32;
        let mut c_out = p.c_out as i32;
        let mut k_h = p.k_h as i32;
        let mut k_w = p.k_w as i32;
        let mut h_out = p.h_out as i32;
        let mut w_out = p.w_out as i32;
        let mut stride_h = p.stride_h as i32;
        let mut stride_w = p.stride_w as i32;
        let mut pad_top = p.pad_top as i32;
        let mut pad_left = p.pad_left as i32;
        let mut dilation_h = p.dilation_h as i32;
        let mut dilation_w = p.dilation_w as i32;
        let mut group = p.group as i32;

        let mut params: [*mut c_void; 23] = [
            &mut x_ptr as *mut u64 as *mut c_void,
            &mut w_ptr as *mut u64 as *mut c_void,
            &mut x_zp_ptr as *mut u64 as *mut c_void,
            &mut w_zp_ptr as *mut u64 as *mut c_void,
            &mut y_ptr as *mut u64 as *mut c_void,
            &mut batch_n as *mut i32 as *mut c_void,
            &mut c_in as *mut i32 as *mut c_void,
            &mut h_in as *mut i32 as *mut c_void,
            &mut w_in as *mut i32 as *mut c_void,
            &mut c_out as *mut i32 as *mut c_void,
            &mut k_h as *mut i32 as *mut c_void,
            &mut k_w as *mut i32 as *mut c_void,
            &mut h_out as *mut i32 as *mut c_void,
            &mut w_out as *mut i32 as *mut c_void,
            &mut stride_h as *mut i32 as *mut c_void,
            &mut stride_w as *mut i32 as *mut c_void,
            &mut pad_top as *mut i32 as *mut c_void,
            &mut pad_left as *mut i32 as *mut c_void,
            &mut dilation_h as *mut i32 as *mut c_void,
            &mut dilation_w as *mut i32 as *mut c_void,
            &mut group as *mut i32 as *mut c_void,
            &mut x_zp_mode as *mut i32 as *mut c_void,
            &mut w_zp_mode as *mut i32 as *mut c_void,
        ];

        match stream {
            Some(s) => self
                .gpu
                .launch_raw_stream(KERNEL_CONV_INT_U8, grid, block, &mut params, s),
            None => self
                .gpu
                .launch_raw(KERNEL_CONV_INT_U8, grid, block, &mut params),
        }
    }
}

// ============================================================================
// Non-Windows stub
// ============================================================================

#[cfg(not(windows))]
pub struct QuantInt8Kernels {
    _private: (),
}

#[cfg(not(windows))]
impl QuantInt8Kernels {
    fn unavailable() -> CudaError {
        CudaError::Unavailable("CUDA quant int8 kernels (non-Windows build)")
    }

    pub fn open() -> Result<Self, CudaError> {
        Err(Self::unavailable())
    }

    pub fn dynamic_quantize_linear(
        &self,
        _x_ptr: u64,
        _n: usize,
        _y_ptr: u64,
        _y_scale_ptr: u64,
        _y_zero_point_ptr: u64,
    ) -> Result<(Dispatch, Dispatch), CudaError> {
        Err(Self::unavailable())
    }

    pub fn dynamic_quantize_linear_on_stream(
        &self,
        _x_ptr: u64,
        _n: usize,
        _y_ptr: u64,
        _y_scale_ptr: u64,
        _y_zero_point_ptr: u64,
        _stream: &Stream<'_>,
    ) -> Result<(Dispatch, Dispatch), CudaError> {
        Err(Self::unavailable())
    }

    pub fn matmul_integer(&self, _p: &MatMulIntegerParams) -> Result<Dispatch, CudaError> {
        Err(Self::unavailable())
    }

    pub fn matmul_integer_on_stream(
        &self,
        _p: &MatMulIntegerParams,
        _stream: &Stream<'_>,
    ) -> Result<Dispatch, CudaError> {
        Err(Self::unavailable())
    }

    pub fn conv_integer(&self, _p: &ConvIntegerParams) -> Result<Dispatch, CudaError> {
        Err(Self::unavailable())
    }

    pub fn conv_integer_on_stream(
        &self,
        _p: &ConvIntegerParams,
        _stream: &Stream<'_>,
    ) -> Result<Dispatch, CudaError> {
        Err(Self::unavailable())
    }
}

// ============================================================================
// Pure-Rust CPU reference implementations (for verification & fallback)
// ============================================================================

/// CPU reference for ONNX `DynamicQuantizeLinear`.
pub fn cpu_dynamic_quantize_linear(x: &[f32]) -> (Vec<u8>, f32, u8) {
    let mut min_val = 0.0f32;
    let mut max_val = 0.0f32;
    for &v in x {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }
    let scale = (max_val - min_val) / 255.0f32;
    let zp = if scale > 0.0f32 {
        (-min_val / scale).round_ties_even().clamp(0.0, 255.0) as u8
    } else {
        0u8
    };
    let zp_f = zp as f32;
    let y = if scale > 0.0f32 {
        x.iter()
            .map(|&v| ((v / scale).round_ties_even() + zp_f).clamp(0.0, 255.0) as u8)
            .collect()
    } else {
        vec![zp; x.len()]
    };
    (y, scale, zp)
}

/// CPU reference for ONNX `MatMulInteger` (2D / 3D / 4D batched).
#[allow(clippy::too_many_arguments)]
pub fn cpu_matmul_integer(
    a: &[u8],
    b: &[u8],
    a_zp: Option<&[u8]>,
    a_zp_mode: i32,
    b_zp: Option<&[u8]>,
    b_zp_mode: i32,
    m: usize,
    n: usize,
    k: usize,
    d0: usize,
    d1: usize,
    a_stride0: i64,
    a_stride1: i64,
    b_stride0: i64,
    b_stride1: i64,
) -> Vec<i32> {
    let batch = d0 * d1;
    let mut y = vec![0i32; batch * m * n];
    let get_zp = |zp: Option<&[u8]>, mode: i32, b_idx: usize, axis_idx: usize, axis_len: usize| -> i32 {
        let Some(slice) = zp else { return 0 };
        match mode {
            1 => slice[0] as i32,
            2 => slice[axis_idx] as i32,
            3 => slice[b_idx * axis_len + axis_idx] as i32,
            _ => 0,
        }
    };
    for b_idx in 0..batch {
        let b0 = b_idx / d1;
        let b1 = b_idx % d1;
        let a_base = (b0 as i64 * a_stride0 + b1 as i64 * a_stride1) as usize;
        let b_base = (b0 as i64 * b_stride0 + b1 as i64 * b_stride1) as usize;
        let y_base = b_idx * m * n;
        for i in 0..m {
            let za = get_zp(a_zp, a_zp_mode, b_idx, i, m);
            for j in 0..n {
                let zb = get_zp(b_zp, b_zp_mode, b_idx, j, n);
                let mut acc = 0i32;
                for kk in 0..k {
                    let av = a[a_base + i * k + kk] as i32 - za;
                    let bv = b[b_base + kk * n + j] as i32 - zb;
                    acc += av * bv;
                }
                y[y_base + i * n + j] = acc;
            }
        }
    }
    y
}

/// CPU reference for ONNX `ConvInteger` (NCHW 2D).
#[allow(clippy::too_many_arguments)]
pub fn cpu_conv_integer(
    x: &[u8],
    w: &[u8],
    x_zp: Option<u8>,
    w_zp: Option<&[u8]>,
    batch_n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    c_out: usize,
    k_h: usize,
    k_w: usize,
    h_out: usize,
    w_out: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    dilation_h: usize,
    dilation_w: usize,
    group: usize,
) -> Vec<i32> {
    let mut y = vec![0i32; batch_n * c_out * h_out * w_out];
    let c_in_g = c_in / group;
    let c_out_g = c_out / group;
    let zx = x_zp.unwrap_or(0) as i32;

    for n in 0..batch_n {
        for oc in 0..c_out {
            let g = oc / c_out_g;
            let ic_start = g * c_in_g;
            let zw = match w_zp {
                None => 0,
                Some(s) if s.len() == 1 => s[0] as i32,
                Some(s) => s[oc] as i32,
            };
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = 0i32;
                    for ic in 0..c_in_g {
                        for kh in 0..k_h {
                            let ih = (oh * stride_h + kh * dilation_h) as isize - pad_top as isize;
                            if ih < 0 || ih >= h_in as isize {
                                continue;
                            }
                            for kw in 0..k_w {
                                let iw =
                                    (ow * stride_w + kw * dilation_w) as isize - pad_left as isize;
                                if iw < 0 || iw >= w_in as isize {
                                    continue;
                                }
                                let x_idx = (((n * c_in + ic_start + ic) * h_in) + ih as usize)
                                    * w_in
                                    + iw as usize;
                                let w_idx = (((oc * c_in_g + ic) * k_h) + kh) * k_w + kw;
                                acc += (x[x_idx] as i32 - zx) * (w[w_idx] as i32 - zw);
                            }
                        }
                    }
                    let y_idx = (((n * c_out + oc) * h_out) + oh) * w_out + ow;
                    y[y_idx] = acc;
                }
            }
        }
    }
    y
}

// ============================================================================
// Tests: CPU reference + real CUDA GPU exact parity
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_f32(len: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
        let mut s = seed;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 40) as f32) / ((1u32 << 24) as f32);
            out.push(lo + u * (hi - lo));
        }
        out
    }

    fn lcg_u8(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            out.push((s >> 56) as u8);
        }
        out
    }

    #[test]
    fn ptx_image_contains_all_four_entry_points() {
        assert!(QUANT_INT8_PTX.contains(KERNEL_DQL_MINMAX));
        assert!(QUANT_INT8_PTX.contains(KERNEL_DQL_APPLY));
        assert!(QUANT_INT8_PTX.contains(KERNEL_MATMUL_INT_U8));
        assert!(QUANT_INT8_PTX.contains(KERNEL_CONV_INT_U8));
    }

    #[test]
    fn cpu_dql_edge_cases() {
        // Mixed positive/negative
        let x = vec![-2.0f32, -1.0, 0.0, 1.0, 3.0];
        let (y, scale, zp) = cpu_dynamic_quantize_linear(&x);
        assert!((scale - 5.0 / 255.0).abs() < 1e-6);
        assert_eq!(zp, 102);
        assert_eq!(y, vec![0, 51, 102, 153, 255]);

        // All zeros
        let (y0, s0, z0) = cpu_dynamic_quantize_linear(&[0.0, 0.0, 0.0]);
        assert_eq!(s0, 0.0);
        assert_eq!(z0, 0);
        assert_eq!(y0, vec![0, 0, 0]);
    }

    #[cfg(windows)]
    fn as_bytes_f32(s: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) }
    }

    #[cfg(windows)]
    fn as_bytes_mut_i32(s: &mut [i32]) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, s.len() * 4) }
    }

    #[cfg(windows)]
    #[test]
    fn gpu_dynamic_quantize_linear_matches_cpu_exact() {
        let Ok(kernels) = QuantInt8Kernels::open() else {
            return; // No NVIDIA GPU on host
        };
        let gpu = kernels.nvidia();

        let cases: &[(&str, Vec<f32>)] = &[
            ("mixed_4096", lcg_f32(4096, 101, -3.5, 5.25)),
            ("all_positive", lcg_f32(1024, 202, 1.0, 10.0)),
            ("all_negative", lcg_f32(777, 303, -8.0, -0.5)),
            ("all_zero", vec![0.0; 512]),
            ("non_multiple_of_256", lcg_f32(1337, 404, -1.25, 1.25)),
        ];

        for (name, x) in cases {
            let (ref_y, ref_scale, ref_zp) = cpu_dynamic_quantize_linear(x);
            let d_x = gpu.alloc_buffer(x.len() * 4).unwrap();
            let d_y = gpu.alloc_buffer(x.len()).unwrap();
            let d_scale = gpu.alloc_buffer(4).unwrap();
            let d_zp = gpu.alloc_buffer(1).unwrap();

            gpu.upload_buffer(&d_x, as_bytes_f32(x)).unwrap();
            kernels
                .dynamic_quantize_linear(
                    d_x.handle(),
                    x.len(),
                    d_y.handle(),
                    d_scale.handle(),
                    d_zp.handle(),
                )
                .unwrap();

            let mut got_y = vec![0u8; x.len()];
            let mut scale_bytes = [0u8; 4];
            let mut zp_bytes = [0u8; 1];
            gpu.download_buffer(&d_y, &mut got_y).unwrap();
            gpu.download_buffer(&d_scale, &mut scale_bytes).unwrap();
            gpu.download_buffer(&d_zp, &mut zp_bytes).unwrap();
            let got_scale = f32::from_le_bytes(scale_bytes);
            let got_zp = zp_bytes[0];

            assert!(
                (got_scale - ref_scale).abs() < 1e-6,
                "{name}: scale mismatch gpu={got_scale} cpu={ref_scale}"
            );
            assert_eq!(got_zp, ref_zp, "{name}: zero_point mismatch");
            assert_eq!(got_y, ref_y, "{name}: quantized u8 mismatch");
        }
    }

    #[cfg(windows)]
    #[test]
    fn gpu_matmul_integer_matches_cpu_exact() {
        let Ok(kernels) = QuantInt8Kernels::open() else {
            return;
        };
        let gpu = kernels.nvidia();

        // Case 1: 2D non-aligned dims with scalar a_zp and per-column b_zp
        {
            let (m, n, k) = (19, 37, 53);
            let a = lcg_u8(m * k, 11);
            let b = lcg_u8(k * n, 22);
            let a_zp = vec![127u8];
            let b_zp = lcg_u8(n, 33);

            let ref_y = cpu_matmul_integer(
                &a,
                &b,
                Some(&a_zp),
                1,
                Some(&b_zp),
                2,
                m,
                n,
                k,
                1,
                1,
                0,
                0,
                0,
                0,
            );

            let d_a = gpu.alloc_buffer(a.len()).unwrap();
            let d_b = gpu.alloc_buffer(b.len()).unwrap();
            let d_azp = gpu.alloc_buffer(a_zp.len()).unwrap();
            let d_bzp = gpu.alloc_buffer(b_zp.len()).unwrap();
            let d_y = gpu.alloc_buffer(ref_y.len() * 4).unwrap();
            gpu.upload_buffer(&d_a, &a).unwrap();
            gpu.upload_buffer(&d_b, &b).unwrap();
            gpu.upload_buffer(&d_azp, &a_zp).unwrap();
            gpu.upload_buffer(&d_bzp, &b_zp).unwrap();

            let params = MatMulIntegerParams::new_2d(
                d_a.handle(),
                d_b.handle(),
                d_y.handle(),
                m,
                n,
                k,
                ZeroPointArg::Scalar(d_azp.handle()),
                ZeroPointArg::PerAxis(d_bzp.handle()),
            );
            kernels.matmul_integer(&params).unwrap();

            let mut got_y = vec![0i32; ref_y.len()];
            gpu.download_buffer(&d_y, as_bytes_mut_i32(&mut got_y)).unwrap();
            assert_eq!(got_y, ref_y, "2D MatMulInteger mismatch");
        }

        // Case 2: 4D batched [2, 3, 17, 29] x [1, 3, 29, 21] with scalar zero points
        {
            let (m, n, k) = (17, 21, 29);
            let a = lcg_u8(2 * 3 * m * k, 44);
            let b = lcg_u8(1 * 3 * k * n, 55);
            let a_zp = vec![100u8];
            let b_zp = vec![140u8];

            let d_a = gpu.alloc_buffer(a.len()).unwrap();
            let d_b = gpu.alloc_buffer(b.len()).unwrap();
            let d_azp = gpu.alloc_buffer(1).unwrap();
            let d_bzp = gpu.alloc_buffer(1).unwrap();
            let d_y = gpu.alloc_buffer(2 * 3 * m * n * 4).unwrap();
            gpu.upload_buffer(&d_a, &a).unwrap();
            gpu.upload_buffer(&d_b, &b).unwrap();
            gpu.upload_buffer(&d_azp, &a_zp).unwrap();
            gpu.upload_buffer(&d_bzp, &b_zp).unwrap();

            let params = MatMulIntegerParams::new_batched(
                d_a.handle(),
                d_b.handle(),
                d_y.handle(),
                &[2, 3],
                &[1, 3],
                m,
                n,
                k,
                ZeroPointArg::Scalar(d_azp.handle()),
                ZeroPointArg::Scalar(d_bzp.handle()),
            )
            .unwrap();
            let ref_y = cpu_matmul_integer(
                &a,
                &b,
                Some(&a_zp),
                1,
                Some(&b_zp),
                1,
                m,
                n,
                k,
                params.d0,
                params.d1,
                params.a_stride0,
                params.a_stride1,
                params.b_stride0,
                params.b_stride1,
            );

            kernels.matmul_integer(&params).unwrap();
            let mut got_y = vec![0i32; ref_y.len()];
            gpu.download_buffer(&d_y, as_bytes_mut_i32(&mut got_y)).unwrap();
            assert_eq!(got_y, ref_y, "4D Batched MatMulInteger mismatch");
        }
    }

    #[cfg(windows)]
    #[test]
    fn gpu_conv_integer_matches_cpu_exact() {
        let Ok(kernels) = QuantInt8Kernels::open() else {
            return;
        };
        let gpu = kernels.nvidia();

        // Cover all 3 eSamEncoderPreUInt8 shapes + grouped/dilated general case:
        // (name, N, C_in, H_in, W_in, C_out, K, stride, pad, dilation, group, per_channel_w_zp)
        let configs = [
            ("esam_k16_s16_p0", 1, 3, 32, 32, 8, 16, 16, 0, 1, 1, true),
            ("esam_k1_s1_p0", 1, 16, 8, 8, 12, 1, 1, 0, 1, 1, true),
            ("esam_k3_s1_p1", 1, 8, 10, 10, 6, 3, 1, 1, 1, 1, false),
            ("general_group2_dil2", 2, 4, 9, 9, 6, 3, 1, 2, 2, 2, true),
        ];

        for (idx, &(name, n, c_in, h_in, w_in, c_out, k, s, p, dil, group, per_ch)) in
            configs.iter().enumerate()
        {
            let h_out = (h_in + 2 * p - dil * (k - 1) - 1) / s + 1;
            let w_out = (w_in + 2 * p - dil * (k - 1) - 1) / s + 1;
            let x = lcg_u8(n * c_in * h_in * w_in, 1000 + idx as u64);
            let w = lcg_u8(c_out * (c_in / group) * k * k, 2000 + idx as u64);
            let x_zp = vec![113u8];
            let w_zp = if per_ch {
                lcg_u8(c_out, 3000 + idx as u64)
            } else {
                vec![128u8]
            };

            let ref_y = cpu_conv_integer(
                &x,
                &w,
                Some(x_zp[0]),
                Some(&w_zp),
                n,
                c_in,
                h_in,
                w_in,
                c_out,
                k,
                k,
                h_out,
                w_out,
                s,
                s,
                p,
                p,
                dil,
                dil,
                group,
            );

            let d_x = gpu.alloc_buffer(x.len()).unwrap();
            let d_w = gpu.alloc_buffer(w.len()).unwrap();
            let d_xzp = gpu.alloc_buffer(1).unwrap();
            let d_wzp = gpu.alloc_buffer(w_zp.len()).unwrap();
            let d_y = gpu.alloc_buffer(ref_y.len() * 4).unwrap();
            gpu.upload_buffer(&d_x, &x).unwrap();
            gpu.upload_buffer(&d_w, &w).unwrap();
            gpu.upload_buffer(&d_xzp, &x_zp).unwrap();
            gpu.upload_buffer(&d_wzp, &w_zp).unwrap();

            let params = ConvIntegerParams {
                x_ptr: d_x.handle(),
                w_ptr: d_w.handle(),
                y_ptr: d_y.handle(),
                x_zero_point: ZeroPointArg::Scalar(d_xzp.handle()),
                w_zero_point: if per_ch {
                    ZeroPointArg::PerAxis(d_wzp.handle())
                } else {
                    ZeroPointArg::Scalar(d_wzp.handle())
                },
                n,
                c_in,
                h_in,
                w_in,
                c_out,
                k_h: k,
                k_w: k,
                h_out,
                w_out,
                stride_h: s,
                stride_w: s,
                pad_top: p,
                pad_left: p,
                dilation_h: dil,
                dilation_w: dil,
                group,
            };
            kernels.conv_integer(&params).unwrap();

            let mut got_y = vec![0i32; ref_y.len()];
            gpu.download_buffer(&d_y, as_bytes_mut_i32(&mut got_y)).unwrap();
            assert_eq!(got_y, ref_y, "{name}: ConvInteger mismatch");
        }
    }
}
