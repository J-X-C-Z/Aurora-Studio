//! Native CUDA elementwise, activation, comparison, type-cast, Softmax, and
//! normalization kernels for the Aurora Native Inference Engine.
//!
//! Zero external DLL dependencies: embeds PTX compiled from
//! `native/cuda/infer_elem_norm.cu` and dispatches via `crate::cuda::Nvidia`
//! (dynamic `nvcuda.dll` driver loader). All launch methods accept raw `u64`
//! device pointers so they operate directly on static byte offsets inside the
//! single VRAM arena planned by `arena.rs`.

use crate::cuda::{CudaError, Dispatch};
use crate::Refusal;

#[cfg(windows)]
use crate::cuda::Nvidia;
#[cfg(windows)]
use std::ffi::c_void;

/// Embedded PTX compiled from `native/cuda/infer_elem_norm.cu`.
pub const PTX: &str = include_str!("../../native/cuda/infer_elem_norm.ptx");

/// Kernel entry-point names in [`PTX`].
pub const KERNEL_BINARY_BROADCAST_F32: &str = "binary_broadcast_f32";
pub const KERNEL_COMPARE_BROADCAST_F32: &str = "compare_broadcast_f32";
pub const KERNEL_AND_BROADCAST_U8: &str = "and_broadcast_u8";
pub const KERNEL_WHERE_BROADCAST_F32: &str = "where_broadcast_f32";
pub const KERNEL_UNARY_F32: &str = "unary_f32";
pub const KERNEL_PRELU_BROADCAST_F32: &str = "prelu_broadcast_f32";
pub const KERNEL_CAST_F16_TO_F32: &str = "cast_f16_to_f32";
pub const KERNEL_CAST_F32_TO_F16: &str = "cast_f32_to_f16";
pub const KERNEL_CAST_I64_TO_F32: &str = "cast_i64_to_f32";
pub const KERNEL_CAST_F32_TO_I64: &str = "cast_f32_to_i64";
pub const KERNEL_SOFTMAX_AXIS_F32: &str = "softmax_axis_f32";
pub const KERNEL_LAYER_NORM_LAST_F32: &str = "layer_norm_last_f32";
pub const KERNEL_INSTANCE_NORM_NCHW_F32: &str = "instance_norm_nchw_f32";
pub const KERNEL_BATCH_NORM_FUSED_NCHW_F32: &str = "batch_norm_fused_nchw_f32";

const BLOCK_1D: u32 = 256;

// ---------------------------------------------------------------------------
// Operator Enums (discriminants match `infer_elem_norm.cu` switch tables)
// ---------------------------------------------------------------------------

/// Elementwise binary arithmetic operators (`binary_broadcast_f32`).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Pow = 4,
}

/// Elementwise comparison operators (`compare_broadcast_f32`, outputs `u8` 0/1).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareOp {
    Equal = 0,
    Greater = 1,
    GreaterOrEqual = 2,
    Less = 3,
}

/// Elementwise unary activation and math operators (`unary_f32`).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Relu = 0,
    Sigmoid = 1,
    SiLU = 2,
    Gelu = 3,
    Erf = 4,
    Sqrt = 5,
    Reciprocal = 6,
    Abs = 7,
    Sin = 8,
    Cos = 9,
    HardSwish = 10,
    HardSigmoid = 11,
    LeakyRelu = 12,
    Clip = 13,
}

// ---------------------------------------------------------------------------
// <= 5D NumPy Broadcast Descriptors (ABI-identical to `infer_elem_norm.cu`)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Broadcast5D {
    pub total: u32,
    pub ndim: u32,
    pub out_shape: [u32; 5],
    pub a_strides: [u32; 5],
    pub b_strides: [u32; 5],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Broadcast5DTernary {
    pub total: u32,
    pub ndim: u32,
    pub out_shape: [u32; 5],
    pub c_strides: [u32; 5],
    pub x_strides: [u32; 5],
    pub y_strides: [u32; 5],
}

fn to_u32(v: usize, label: &'static str) -> Result<u32, CudaError> {
    u32::try_from(v).map_err(|_| CudaError::Refused(Refusal::ValueOutOfRange(label)))
}

fn compute_strides_padded(
    shape: &[usize],
    out_shape: &[usize],
) -> Result<[u32; 5], CudaError> {
    let ndim = out_shape.len();
    let pad = ndim
        .checked_sub(shape.len())
        .ok_or(CudaError::Refused(Refusal::ShapeMismatch("broadcast rank")))?;
    let mut padded = [1usize; 5];
    for (i, &d) in shape.iter().enumerate() {
        padded[pad + i] = d;
    }

    // Standard contiguous row-major strides for `shape`, then zero out broadcast axes.
    let mut raw_strides = [0usize; 5];
    let mut acc = 1usize;
    for d in (0..ndim).rev() {
        raw_strides[d] = acc;
        acc = acc
            .checked_mul(padded[d])
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("broadcast stride")))?;
    }

    let mut out = [0u32; 5];
    for d in 0..ndim {
        let s_dim = padded[d];
        let o_dim = out_shape[d];
        if s_dim == o_dim {
            out[d] = if o_dim == 1 {
                0
            } else {
                to_u32(raw_strides[d], "broadcast stride")?
            };
        } else if s_dim == 1 {
            out[d] = 0;
        } else {
            return Err(CudaError::Refused(Refusal::ShapeMismatch(
                "incompatible broadcast dims",
            )));
        }
    }
    Ok(out)
}

impl Broadcast5D {
    /// Computes the NumPy <= 5D broadcast descriptor for `(a_shape, b_shape)`.
    pub fn new(a_shape: &[usize], b_shape: &[usize]) -> Result<Self, CudaError> {
        let ndim = a_shape.len().max(b_shape.len());
        if ndim > 5 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch(
                "broadcast rank > 5",
            )));
        }
        let mut out_dims = vec![1usize; ndim];
        for i in 0..ndim {
            let da = if i + a_shape.len() >= ndim {
                a_shape[i + a_shape.len() - ndim]
            } else {
                1
            };
            let db = if i + b_shape.len() >= ndim {
                b_shape[i + b_shape.len() - ndim]
            } else {
                1
            };
            if da == db {
                out_dims[i] = da;
            } else if da == 1 {
                out_dims[i] = db;
            } else if db == 1 {
                out_dims[i] = da;
            } else {
                return Err(CudaError::Refused(Refusal::ShapeMismatch(
                    "incompatible broadcast dims",
                )));
            }
        }
        let total = out_dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("broadcast total")))?;
        if total == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch(
                "empty broadcast tensor",
            )));
        }

        let mut out_shape = [1u32; 5];
        for (i, &d) in out_dims.iter().enumerate() {
            out_shape[i] = to_u32(d, "broadcast dim")?;
        }
        let a_strides = compute_strides_padded(a_shape, &out_dims)?;
        let b_strides = compute_strides_padded(b_shape, &out_dims)?;

        Ok(Self {
            total: to_u32(total, "broadcast total")?,
            ndim: ndim as u32,
            out_shape,
            a_strides,
            b_strides,
        })
    }
}

impl Broadcast5DTernary {
    /// Computes the NumPy <= 5D broadcast descriptor for `(cond, x, y)`.
    pub fn new(
        c_shape: &[usize],
        x_shape: &[usize],
        y_shape: &[usize],
    ) -> Result<Self, CudaError> {
        let ndim = c_shape.len().max(x_shape.len()).max(y_shape.len());
        if ndim > 5 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch(
                "ternary broadcast rank > 5",
            )));
        }
        let mut out_dims = vec![1usize; ndim];
        for i in 0..ndim {
            let dc = if i + c_shape.len() >= ndim {
                c_shape[i + c_shape.len() - ndim]
            } else {
                1
            };
            let dx = if i + x_shape.len() >= ndim {
                x_shape[i + x_shape.len() - ndim]
            } else {
                1
            };
            let dy = if i + y_shape.len() >= ndim {
                y_shape[i + y_shape.len() - ndim]
            } else {
                1
            };
            let mut m = 1usize;
            for d in [dc, dx, dy] {
                if d == 1 {
                    continue;
                }
                if m == 1 || m == d {
                    m = d;
                } else {
                    return Err(CudaError::Refused(Refusal::ShapeMismatch(
                        "incompatible ternary broadcast dims",
                    )));
                }
            }
            out_dims[i] = m;
        }
        let total = out_dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange(
                "ternary broadcast total",
            )))?;
        if total == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch(
                "empty ternary broadcast tensor",
            )));
        }

        let mut out_shape = [1u32; 5];
        for (i, &d) in out_dims.iter().enumerate() {
            out_shape[i] = to_u32(d, "ternary broadcast dim")?;
        }
        let c_strides = compute_strides_padded(c_shape, &out_dims)?;
        let x_strides = compute_strides_padded(x_shape, &out_dims)?;
        let y_strides = compute_strides_padded(y_shape, &out_dims)?;

        Ok(Self {
            total: to_u32(total, "ternary broadcast total")?,
            ndim: ndim as u32,
            out_shape,
            c_strides,
            x_strides,
            y_strides,
        })
    }
}

// ---------------------------------------------------------------------------
// Windows CUDA Kernel Dispatcher
// ---------------------------------------------------------------------------

#[cfg(windows)]
pub struct ElemNormKernels {
    gpu: Nvidia,
}

#[cfg(windows)]
impl ElemNormKernels {
    /// Loads (or reuses cached) `infer_elem_norm.ptx` on the shared CUDA context.
    pub fn open() -> Result<Self, CudaError> {
        let gpu = Nvidia::open_ptx(PTX)?;
        Ok(Self { gpu })
    }

    /// Borrows the underlying [`Nvidia`] handle for buffer management or streams.
    pub fn nvidia(&self) -> &Nvidia {
        &self.gpu
    }

    /// Generic <= 5D NumPy-broadcast binary FP32 op (`Add/Sub/Mul/Div/Pow`).
    pub fn binary_broadcast_f32(
        &self,
        op: BinaryOp,
        a_ptr: u64,
        a_shape: &[usize],
        b_ptr: u64,
        b_shape: &[usize],
        out_ptr: u64,
    ) -> Result<Dispatch, CudaError> {
        let mut bc = Broadcast5D::new(a_shape, b_shape)?;
        let grid = (bc.total.div_ceil(BLOCK_1D).max(1), 1);
        let mut a = a_ptr;
        let mut b = b_ptr;
        let mut out = out_ptr;
        let mut op_code = op as u32;
        let mut params: [*mut c_void; 5] = [
            &mut a as *mut u64 as *mut c_void,
            &mut b as *mut u64 as *mut c_void,
            &mut out as *mut u64 as *mut c_void,
            &mut bc as *mut Broadcast5D as *mut c_void,
            &mut op_code as *mut u32 as *mut c_void,
        ];
        self.gpu
            .launch_raw(KERNEL_BINARY_BROADCAST_F32, grid, (BLOCK_1D, 1), &mut params)
    }

    /// Generic <= 5D NumPy-broadcast FP32 comparison outputting `u8` (0 or 1).
    pub fn compare_broadcast_f32(
        &self,
        op: CompareOp,
        a_ptr: u64,
        a_shape: &[usize],
        b_ptr: u64,
        b_shape: &[usize],
        out_ptr: u64,
    ) -> Result<Dispatch, CudaError> {
        let mut bc = Broadcast5D::new(a_shape, b_shape)?;
        let grid = (bc.total.div_ceil(BLOCK_1D).max(1), 1);
        let mut a = a_ptr;
        let mut b = b_ptr;
        let mut out = out_ptr;
        let mut op_code = op as u32;
        let mut params: [*mut c_void; 5] = [
            &mut a as *mut u64 as *mut c_void,
            &mut b as *mut u64 as *mut c_void,
            &mut out as *mut u64 as *mut c_void,
            &mut bc as *mut Broadcast5D as *mut c_void,
            &mut op_code as *mut u32 as *mut c_void,
        ];
        self.gpu.launch_raw(
            KERNEL_COMPARE_BROADCAST_F32,
            grid,
            (BLOCK_1D, 1),
            &mut params,
        )
    }

    /// Generic <= 5D NumPy-broadcast logical AND over `u8` boolean tensors.
    pub fn and_broadcast_u8(
        &self,
        a_ptr: u64,
        a_shape: &[usize],
        b_ptr: u64,
        b_shape: &[usize],
        out_ptr: u64,
    ) -> Result<Dispatch, CudaError> {
        let mut bc = Broadcast5D::new(a_shape, b_shape)?;
        let grid = (bc.total.div_ceil(BLOCK_1D).max(1), 1);
        let mut a = a_ptr;
        let mut b = b_ptr;
        let mut out = out_ptr;
        let mut params: [*mut c_void; 4] = [
            &mut a as *mut u64 as *mut c_void,
            &mut b as *mut u64 as *mut c_void,
            &mut out as *mut u64 as *mut c_void,
            &mut bc as *mut Broadcast5D as *mut c_void,
        ];
        self.gpu
            .launch_raw(KERNEL_AND_BROADCAST_U8, grid, (BLOCK_1D, 1), &mut params)
    }

    /// Generic <= 5D NumPy-broadcast `Where(cond: u8, x: f32, y: f32) -> f32`.
    #[allow(clippy::too_many_arguments)]
    pub fn where_broadcast_f32(
        &self,
        cond_ptr: u64,
        cond_shape: &[usize],
        x_ptr: u64,
        x_shape: &[usize],
        y_ptr: u64,
        y_shape: &[usize],
        out_ptr: u64,
    ) -> Result<Dispatch, CudaError> {
        let mut bc = Broadcast5DTernary::new(cond_shape, x_shape, y_shape)?;
        let grid = (bc.total.div_ceil(BLOCK_1D).max(1), 1);
        let mut c = cond_ptr;
        let mut x = x_ptr;
        let mut y = y_ptr;
        let mut out = out_ptr;
        let mut params: [*mut c_void; 5] = [
            &mut c as *mut u64 as *mut c_void,
            &mut x as *mut u64 as *mut c_void,
            &mut y as *mut u64 as *mut c_void,
            &mut out as *mut u64 as *mut c_void,
            &mut bc as *mut Broadcast5DTernary as *mut c_void,
        ];
        self.gpu
            .launch_raw(KERNEL_WHERE_BROADCAST_F32, grid, (BLOCK_1D, 1), &mut params)
    }

    /// Unary activation or math op on `total` contiguous `f32` elements.
    ///
    /// Parameter usage:
    /// - `HardSigmoid`: `p0 = alpha`, `p1 = beta`
    /// - `LeakyRelu`: `p0 = alpha` (`p1` ignored)
    /// - `Clip`: `p0 = lo`, `p1 = hi`
    /// - All other ops ignore `p0` and `p1`.
    pub fn unary_f32(
        &self,
        op: UnaryOp,
        x_ptr: u64,
        out_ptr: u64,
        total: usize,
        p0: f32,
        p1: f32,
    ) -> Result<Dispatch, CudaError> {
        if total == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("unary total")));
        }
        let mut total_u32 = to_u32(total, "unary total")?;
        let grid = (total_u32.div_ceil(BLOCK_1D).max(1), 1);
        let mut x = x_ptr;
        let mut out = out_ptr;
        let mut op_code = op as u32;
        let mut a0 = p0;
        let mut a1 = p1;
        let mut params: [*mut c_void; 6] = [
            &mut x as *mut u64 as *mut c_void,
            &mut out as *mut u64 as *mut c_void,
            &mut total_u32 as *mut u32 as *mut c_void,
            &mut op_code as *mut u32 as *mut c_void,
            &mut a0 as *mut f32 as *mut c_void,
            &mut a1 as *mut f32 as *mut c_void,
        ];
        self.gpu
            .launch_raw(KERNEL_UNARY_F32, grid, (BLOCK_1D, 1), &mut params)
    }

    /// PRelu with unidirectional <= 5D broadcast of `slope` to `x_shape`.
    pub fn prelu_broadcast_f32(
        &self,
        x_ptr: u64,
        x_shape: &[usize],
        slope_ptr: u64,
        slope_shape: &[usize],
        out_ptr: u64,
    ) -> Result<Dispatch, CudaError> {
        let mut bc = Broadcast5D::new(x_shape, slope_shape)?;
        let grid = (bc.total.div_ceil(BLOCK_1D).max(1), 1);
        let mut x = x_ptr;
        let mut s = slope_ptr;
        let mut out = out_ptr;
        let mut params: [*mut c_void; 4] = [
            &mut x as *mut u64 as *mut c_void,
            &mut s as *mut u64 as *mut c_void,
            &mut out as *mut u64 as *mut c_void,
            &mut bc as *mut Broadcast5D as *mut c_void,
        ];
        self.gpu
            .launch_raw(KERNEL_PRELU_BROADCAST_F32, grid, (BLOCK_1D, 1), &mut params)
    }

    /// Cast `total` IEEE-754 FP16 (`u16`) elements to FP32 (`f32`).
    pub fn cast_f16_to_f32(
        &self,
        src_ptr: u64,
        dst_ptr: u64,
        total: usize,
    ) -> Result<Dispatch, CudaError> {
        self.launch_cast(KERNEL_CAST_F16_TO_F32, src_ptr, dst_ptr, total)
    }

    /// Cast `total` FP32 (`f32`) elements to IEEE-754 FP16 (`u16`, round-to-nearest-even).
    pub fn cast_f32_to_f16(
        &self,
        src_ptr: u64,
        dst_ptr: u64,
        total: usize,
    ) -> Result<Dispatch, CudaError> {
        self.launch_cast(KERNEL_CAST_F32_TO_F16, src_ptr, dst_ptr, total)
    }

    /// Cast `total` `i64` elements to FP32 (`f32`).
    pub fn cast_i64_to_f32(
        &self,
        src_ptr: u64,
        dst_ptr: u64,
        total: usize,
    ) -> Result<Dispatch, CudaError> {
        self.launch_cast(KERNEL_CAST_I64_TO_F32, src_ptr, dst_ptr, total)
    }

    /// Cast `total` FP32 (`f32`) elements to `i64` (truncate toward zero).
    pub fn cast_f32_to_i64(
        &self,
        src_ptr: u64,
        dst_ptr: u64,
        total: usize,
    ) -> Result<Dispatch, CudaError> {
        self.launch_cast(KERNEL_CAST_F32_TO_I64, src_ptr, dst_ptr, total)
    }

    fn launch_cast(
        &self,
        kernel: &'static str,
        src_ptr: u64,
        dst_ptr: u64,
        total: usize,
    ) -> Result<Dispatch, CudaError> {
        if total == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("cast total")));
        }
        let mut total_u32 = to_u32(total, "cast total")?;
        let grid = (total_u32.div_ceil(BLOCK_1D).max(1), 1);
        let mut src = src_ptr;
        let mut dst = dst_ptr;
        let mut params: [*mut c_void; 3] = [
            &mut src as *mut u64 as *mut c_void,
            &mut dst as *mut u64 as *mut c_void,
            &mut total_u32 as *mut u32 as *mut c_void,
        ];
        self.gpu.launch_raw(kernel, grid, (BLOCK_1D, 1), &mut params)
    }

    /// Numerically stable Softmax over arbitrary axis reshaped as `[outer, axis_dim, inner]`.
    pub fn softmax_axis_f32(
        &self,
        x_ptr: u64,
        out_ptr: u64,
        outer: usize,
        axis_dim: usize,
        inner: usize,
    ) -> Result<Dispatch, CudaError> {
        if outer == 0 || axis_dim == 0 || inner == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("softmax dims")));
        }
        let rows = outer
            .checked_mul(inner)
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("softmax rows")))?;
        let rows_u32 = to_u32(rows, "softmax rows")?;
        let mut outer_u32 = to_u32(outer, "softmax outer")?;
        let mut axis_u32 = to_u32(axis_dim, "softmax axis_dim")?;
        let mut inner_u32 = to_u32(inner, "softmax inner")?;
        let mut x = x_ptr;
        let mut out = out_ptr;
        let mut params: [*mut c_void; 5] = [
            &mut x as *mut u64 as *mut c_void,
            &mut out as *mut u64 as *mut c_void,
            &mut outer_u32 as *mut u32 as *mut c_void,
            &mut axis_u32 as *mut u32 as *mut c_void,
            &mut inner_u32 as *mut u32 as *mut c_void,
        ];
        self.gpu
            .launch_raw(KERNEL_SOFTMAX_AXIS_F32, (rows_u32, 1), (BLOCK_1D, 1), &mut params)
    }

    /// LayerNormalization on `axis = -1` for input shaped `[rows, norm_dim]`.
    /// `scale_ptr` and `bias_ptr` are optional (`None` passes null device pointer).
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm_last_f32(
        &self,
        x_ptr: u64,
        scale_ptr: Option<u64>,
        bias_ptr: Option<u64>,
        out_ptr: u64,
        rows: usize,
        norm_dim: usize,
        eps: f32,
    ) -> Result<Dispatch, CudaError> {
        if rows == 0 || norm_dim == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("layer_norm dims")));
        }
        let mut rows_u32 = to_u32(rows, "layer_norm rows")?;
        let mut dim_u32 = to_u32(norm_dim, "layer_norm norm_dim")?;
        let mut x = x_ptr;
        let mut scale = scale_ptr.unwrap_or(0);
        let mut bias = bias_ptr.unwrap_or(0);
        let mut out = out_ptr;
        let mut eps_val = eps;
        let mut params: [*mut c_void; 7] = [
            &mut x as *mut u64 as *mut c_void,
            &mut scale as *mut u64 as *mut c_void,
            &mut bias as *mut u64 as *mut c_void,
            &mut out as *mut u64 as *mut c_void,
            &mut rows_u32 as *mut u32 as *mut c_void,
            &mut dim_u32 as *mut u32 as *mut c_void,
            &mut eps_val as *mut f32 as *mut c_void,
        ];
        self.gpu.launch_raw(
            KERNEL_LAYER_NORM_LAST_F32,
            (rows_u32, 1),
            (BLOCK_1D, 1),
            &mut params,
        )
    }

    /// InstanceNormalization on NCHW `[n, c, h, w]`.
    #[allow(clippy::too_many_arguments)]
    pub fn instance_norm_nchw_f32(
        &self,
        x_ptr: u64,
        scale_ptr: Option<u64>,
        bias_ptr: Option<u64>,
        out_ptr: u64,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        eps: f32,
    ) -> Result<Dispatch, CudaError> {
        if n == 0 || c == 0 || h == 0 || w == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch(
                "instance_norm dims",
            )));
        }
        let hw = h
            .checked_mul(w)
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("instance_norm hw")))?;
        let slices = n
            .checked_mul(c)
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange(
                "instance_norm slices",
            )))?;
        let slices_u32 = to_u32(slices, "instance_norm slices")?;
        let mut n_u32 = to_u32(n, "instance_norm n")?;
        let mut c_u32 = to_u32(c, "instance_norm c")?;
        let mut hw_u32 = to_u32(hw, "instance_norm hw")?;
        let mut x = x_ptr;
        let mut scale = scale_ptr.unwrap_or(0);
        let mut bias = bias_ptr.unwrap_or(0);
        let mut out = out_ptr;
        let mut eps_val = eps;
        let mut params: [*mut c_void; 8] = [
            &mut x as *mut u64 as *mut c_void,
            &mut scale as *mut u64 as *mut c_void,
            &mut bias as *mut u64 as *mut c_void,
            &mut out as *mut u64 as *mut c_void,
            &mut n_u32 as *mut u32 as *mut c_void,
            &mut c_u32 as *mut u32 as *mut c_void,
            &mut hw_u32 as *mut u32 as *mut c_void,
            &mut eps_val as *mut f32 as *mut c_void,
        ];
        self.gpu.launch_raw(
            KERNEL_INSTANCE_NORM_NCHW_F32,
            (slices_u32, 1),
            (BLOCK_1D, 1),
            &mut params,
        )
    }

    /// Pre-fused BatchNormalization on NCHW `[n, c, h, w]` consuming
    /// `(scale_eff, bias_eff)` from `bn_fold.rs`: `y = x * scale_eff[c] + bias_eff[c]`.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm_fused_nchw_f32(
        &self,
        x_ptr: u64,
        scale_eff_ptr: u64,
        bias_eff_ptr: u64,
        out_ptr: u64,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<Dispatch, CudaError> {
        if n == 0 || c == 0 || h == 0 || w == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("batch_norm dims")));
        }
        let hw = h
            .checked_mul(w)
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("batch_norm hw")))?;
        let total = n
            .checked_mul(c)
            .and_then(|nc| nc.checked_mul(hw))
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange(
                "batch_norm total",
            )))?;
        let total_u32 = to_u32(total, "batch_norm total")?;
        let mut n_u32 = to_u32(n, "batch_norm n")?;
        let mut c_u32 = to_u32(c, "batch_norm c")?;
        let mut hw_u32 = to_u32(hw, "batch_norm hw")?;
        let grid = (total_u32.div_ceil(BLOCK_1D).max(1), 1);
        let mut x = x_ptr;
        let mut scale_eff = scale_eff_ptr;
        let mut bias_eff = bias_eff_ptr;
        let mut out = out_ptr;
        let mut params: [*mut c_void; 7] = [
            &mut x as *mut u64 as *mut c_void,
            &mut scale_eff as *mut u64 as *mut c_void,
            &mut bias_eff as *mut u64 as *mut c_void,
            &mut out as *mut u64 as *mut c_void,
            &mut n_u32 as *mut u32 as *mut c_void,
            &mut c_u32 as *mut u32 as *mut c_void,
            &mut hw_u32 as *mut u32 as *mut c_void,
        ];
        self.gpu.launch_raw(
            KERNEL_BATCH_NORM_FUSED_NCHW_F32,
            grid,
            (BLOCK_1D, 1),
            &mut params,
        )
    }
}

// ---------------------------------------------------------------------------
// Non-Windows Stub
// ---------------------------------------------------------------------------

#[cfg(not(windows))]
pub struct ElemNormKernels;

#[cfg(not(windows))]
impl ElemNormKernels {
    pub fn open() -> Result<Self, CudaError> {
        Err(CudaError::Unavailable("CUDA driver requires Windows"))
    }

    pub fn binary_broadcast_f32(
        &self,
        _op: BinaryOp,
        _a_ptr: u64,
        _a_shape: &[usize],
        _b_ptr: u64,
        _b_shape: &[usize],
        _out_ptr: u64,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("CUDA driver requires Windows"))
    }

    pub fn compare_broadcast_f32(
        &self,
        _op: CompareOp,
        _a_ptr: u64,
        _a_shape: &[usize],
        _b_ptr: u64,
        _b_shape: &[usize],
        _out_ptr: u64,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("CUDA driver requires Windows"))
    }

    pub fn and_broadcast_u8(
        &self,
        _a_ptr: u64,
        _a_shape: &[usize],
        _b_ptr: u64,
        _b_shape: &[usize],
        _out_ptr: u64,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("CUDA driver requires Windows"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn where_broadcast_f32(
        &self,
        _cond_ptr: u64,
        _cond_shape: &[usize],
        _x_ptr: u64,
        _x_shape: &[usize],
        _y_ptr: u64,
        _y_shape: &[usize],
        _out_ptr: u64,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("CUDA driver requires Windows"))
    }

    pub fn unary_f32(
        &self,
        _op: UnaryOp,
        _x_ptr: u64,
        _out_ptr: u64,
        _total: usize,
        _p0: f32,
        _p1: f32,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("CUDA driver requires Windows"))
    }

    pub fn prelu_broadcast_f32(
        &self,
        _x_ptr: u64,
        _x_shape: &[usize],
        _slope_ptr: u64,
        _slope_shape: &[usize],
        _out_ptr: u64,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("CUDA driver requires Windows"))
    }

    pub fn cast_f16_to_f32(
        &self,
        _src_ptr: u64,
        _dst_ptr: u64,
        _total: usize,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("CUDA driver requires Windows"))
    }

    pub fn cast_f32_to_f16(
        &self,
        _src_ptr: u64,
        _dst_ptr: u64,
        _total: usize,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("CUDA driver requires Windows"))
    }

    pub fn cast_i64_to_f32(
        &self,
        _src_ptr: u64,
        _dst_ptr: u64,
        _total: usize,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("CUDA driver requires Windows"))
    }

    pub fn cast_f32_to_i64(
        &self,
        _src_ptr: u64,
        _dst_ptr: u64,
        _total: usize,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("CUDA driver requires Windows"))
    }

    pub fn softmax_axis_f32(
        &self,
        _x_ptr: u64,
        _out_ptr: u64,
        _outer: usize,
        _axis_dim: usize,
        _inner: usize,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("CUDA driver requires Windows"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm_last_f32(
        &self,
        _x_ptr: u64,
        _scale_ptr: Option<u64>,
        _bias_ptr: Option<u64>,
        _out_ptr: u64,
        _rows: usize,
        _norm_dim: usize,
        _eps: f32,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("CUDA driver requires Windows"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn instance_norm_nchw_f32(
        &self,
        _x_ptr: u64,
        _scale_ptr: Option<u64>,
        _bias_ptr: Option<u64>,
        _out_ptr: u64,
        _n: usize,
        _c: usize,
        _h: usize,
        _w: usize,
        _eps: f32,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("CUDA driver requires Windows"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm_fused_nchw_f32(
        &self,
        _x_ptr: u64,
        _scale_eff_ptr: u64,
        _bias_eff_ptr: u64,
        _out_ptr: u64,
        _n: usize,
        _c: usize,
        _h: usize,
        _w: usize,
    ) -> Result<Dispatch, CudaError> {
        Err(CudaError::Unavailable("CUDA driver requires Windows"))
    }
}

// ---------------------------------------------------------------------------
// Unit Tests: CPU Reference vs Native CUDA Kernel Parity (< 1e-5)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    use crate::native_infer::onnx_proto::{f16_to_f32, f32_to_f16};

    /// Abramowitz & Stegun 7.1.26 erf(x) in f64 (max error <= 1.5e-7).
    fn erf_ref(x: f64) -> f64 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let ax = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * ax);
        let poly = t
            * (0.254829592
                + t * (-0.284496736
                    + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        sign * (1.0 - poly * (-ax * ax).exp())
    }

    #[test]
    fn broadcast_5d_layout_and_validation() {
        let bc = Broadcast5D::new(&[2, 3, 1, 5], &[1, 3, 4, 1]).expect("valid 4D broadcast");
        assert_eq!(bc.ndim, 4);
        assert_eq!(bc.total, 2 * 3 * 4 * 5);
        assert_eq!(&bc.out_shape[..4], &[2, 3, 4, 5]);
        assert_eq!(&bc.a_strides[..4], &[15, 5, 0, 1]);
        assert_eq!(&bc.b_strides[..4], &[0, 4, 1, 0]);

        // Scalar broadcast
        let sc = Broadcast5D::new(&[2, 4], &[]).expect("scalar broadcast");
        assert_eq!(sc.ndim, 2);
        assert_eq!(sc.total, 8);
        assert_eq!(&sc.b_strides[..2], &[0, 0]);

        // Incompatible dims rejected
        assert!(Broadcast5D::new(&[2, 3], &[2, 4]).is_err());
        // Rank > 5 rejected
        assert!(Broadcast5D::new(&[1, 1, 1, 1, 1, 1], &[1]).is_err());
    }

    #[cfg(windows)]
    fn f32_as_bytes(slice: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
    }

    #[cfg(windows)]
    fn f32_as_bytes_mut(slice: &mut [f32]) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                slice.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(slice),
            )
        }
    }

    #[cfg(windows)]
    fn u16_as_bytes(slice: &[u16]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
    }

    #[cfg(windows)]
    fn u16_as_bytes_mut(slice: &mut [u16]) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                slice.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(slice),
            )
        }
    }

    #[cfg(windows)]
    fn i64_as_bytes(slice: &[i64]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
    }

    #[cfg(windows)]
    fn i64_as_bytes_mut(slice: &mut [i64]) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                slice.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(slice),
            )
        }
    }

    #[cfg(windows)]
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[cfg(windows)]
    #[test]
    fn gpu_binary_compare_and_where_4d_broadcast_parity() {
        let Ok(k) = ElemNormKernels::open() else {
            return;
        };
        let gpu = k.nvidia();

        let a_shape = [2usize, 3, 1, 5];
        let b_shape = [1usize, 3, 4, 1];
        let bc = Broadcast5D::new(&a_shape, &b_shape).unwrap();
        let out_len = bc.total as usize;

        // Positive bases for Pow safety.
        let a_host: Vec<f32> = (0..30).map(|i| 0.5 + (i as f32) * 0.13).collect();
        let b_host: Vec<f32> = (0..12).map(|i| 0.4 + (i as f32) * 0.17).collect();

        let d_a = gpu.alloc_buffer(a_host.len() * 4).unwrap();
        let d_b = gpu.alloc_buffer(b_host.len() * 4).unwrap();
        let d_out = gpu.alloc_buffer(out_len * 4).unwrap();
        gpu.upload_buffer(&d_a, f32_as_bytes(&a_host)).unwrap();
        gpu.upload_buffer(&d_b, f32_as_bytes(&b_host)).unwrap();

        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Pow,
        ] {
            k.binary_broadcast_f32(op, d_a.handle(), &a_shape, d_b.handle(), &b_shape, d_out.handle())
                .unwrap();
            let mut got = vec![0.0f32; out_len];
            gpu.download_buffer(&d_out, f32_as_bytes_mut(&mut got)).unwrap();

            let mut expected = vec![0.0f32; out_len];
            for idx in 0..out_len {
                let mut rem = idx as u32;
                let mut oa = 0usize;
                let mut ob = 0usize;
                for d in (0..bc.ndim as usize).rev() {
                    let dim = bc.out_shape[d];
                    let c = rem % dim;
                    rem /= dim;
                    oa += (c * bc.a_strides[d]) as usize;
                    ob += (c * bc.b_strides[d]) as usize;
                }
                expected[idx] = match op {
                    BinaryOp::Add => a_host[oa] + b_host[ob],
                    BinaryOp::Sub => a_host[oa] - b_host[ob],
                    BinaryOp::Mul => a_host[oa] * b_host[ob],
                    BinaryOp::Div => a_host[oa] / b_host[ob],
                    BinaryOp::Pow => a_host[oa].powf(b_host[ob]),
                };
            }
            let diff = max_abs_diff(&got, &expected);
            assert!(diff < 1e-5, "binary {op:?} max_abs_diff={diff}");
        }

        // Compare + And + Where
        let d_cmp1 = gpu.alloc_buffer(out_len).unwrap();
        let d_cmp2 = gpu.alloc_buffer(out_len).unwrap();
        let d_and = gpu.alloc_buffer(out_len).unwrap();

        for cmp_op in [
            CompareOp::Equal,
            CompareOp::Greater,
            CompareOp::GreaterOrEqual,
            CompareOp::Less,
        ] {
            k.compare_broadcast_f32(
                cmp_op,
                d_a.handle(),
                &a_shape,
                d_b.handle(),
                &b_shape,
                d_cmp1.handle(),
            )
            .unwrap();
            let mut got_u8 = vec![0u8; out_len];
            gpu.download_buffer(&d_cmp1, &mut got_u8).unwrap();

            for idx in 0..out_len {
                let mut rem = idx as u32;
                let mut oa = 0usize;
                let mut ob = 0usize;
                for d in (0..bc.ndim as usize).rev() {
                    let dim = bc.out_shape[d];
                    let c = rem % dim;
                    rem /= dim;
                    oa += (c * bc.a_strides[d]) as usize;
                    ob += (c * bc.b_strides[d]) as usize;
                }
                let exp = match cmp_op {
                    CompareOp::Equal => a_host[oa] == b_host[ob],
                    CompareOp::Greater => a_host[oa] > b_host[ob],
                    CompareOp::GreaterOrEqual => a_host[oa] >= b_host[ob],
                    CompareOp::Less => a_host[oa] < b_host[ob],
                };
                assert_eq!(got_u8[idx], u8::from(exp), "compare {cmp_op:?} at {idx}");
            }
        }

        // d_cmp1 holds Less, compute GreaterOrEqual in d_cmp2, then And & Where.
        k.compare_broadcast_f32(
            CompareOp::Greater,
            d_a.handle(),
            &a_shape,
            d_b.handle(),
            &b_shape,
            d_cmp1.handle(),
        )
        .unwrap();
        k.compare_broadcast_f32(
            CompareOp::GreaterOrEqual,
            d_a.handle(),
            &a_shape,
            d_b.handle(),
            &b_shape,
            d_cmp2.handle(),
        )
        .unwrap();
        let out_shape = [2usize, 3, 4, 5];
        k.and_broadcast_u8(
            d_cmp1.handle(),
            &out_shape,
            d_cmp2.handle(),
            &out_shape,
            d_and.handle(),
        )
        .unwrap();
        k.where_broadcast_f32(
            d_and.handle(),
            &out_shape,
            d_a.handle(),
            &a_shape,
            d_b.handle(),
            &b_shape,
            d_out.handle(),
        )
        .unwrap();
        let mut cond_host = vec![0u8; out_len];
        let mut where_got = vec![0.0f32; out_len];
        gpu.download_buffer(&d_and, &mut cond_host).unwrap();
        gpu.download_buffer(&d_out, f32_as_bytes_mut(&mut where_got)).unwrap();
        for idx in 0..out_len {
            let mut rem = idx as u32;
            let mut oa = 0usize;
            let mut ob = 0usize;
            for d in (0..bc.ndim as usize).rev() {
                let dim = bc.out_shape[d];
                let c = rem % dim;
                rem /= dim;
                oa += (c * bc.a_strides[d]) as usize;
                ob += (c * bc.b_strides[d]) as usize;
            }
            let exp = if cond_host[idx] != 0 { a_host[oa] } else { b_host[ob] };
            assert!((where_got[idx] - exp).abs() < 1e-6);
        }
    }

    #[cfg(windows)]
    #[test]
    fn gpu_unary_and_prelu_parity() {
        let Ok(k) = ElemNormKernels::open() else {
            return;
        };
        let gpu = k.nvidia();

        let n = 513usize;
        let x_signed: Vec<f32> = (0..n).map(|i| -3.5 + 7.0 * (i as f32) / (n as f32)).collect();
        let x_pos: Vec<f32> = (0..n).map(|i| 0.1 + 5.0 * (i as f32) / (n as f32)).collect();

        let d_xs = gpu.alloc_buffer(n * 4).unwrap();
        let d_xp = gpu.alloc_buffer(n * 4).unwrap();
        let d_out = gpu.alloc_buffer(n * 4).unwrap();
        gpu.upload_buffer(&d_xs, f32_as_bytes(&x_signed)).unwrap();
        gpu.upload_buffer(&d_xp, f32_as_bytes(&x_pos)).unwrap();

        let cases: &[(UnaryOp, bool, f32, f32)] = &[
            (UnaryOp::Relu, false, 0.0, 0.0),
            (UnaryOp::Sigmoid, false, 0.0, 0.0),
            (UnaryOp::SiLU, false, 0.0, 0.0),
            (UnaryOp::Gelu, false, 0.0, 0.0),
            (UnaryOp::Erf, false, 0.0, 0.0),
            (UnaryOp::Sqrt, true, 0.0, 0.0),
            (UnaryOp::Reciprocal, true, 0.0, 0.0),
            (UnaryOp::Abs, false, 0.0, 0.0),
            (UnaryOp::Sin, false, 0.0, 0.0),
            (UnaryOp::Cos, false, 0.0, 0.0),
            (UnaryOp::HardSwish, false, 0.0, 0.0),
            (UnaryOp::HardSigmoid, false, 1.0 / 6.0, 0.5),
            (UnaryOp::LeakyRelu, false, 0.15, 0.0),
            (UnaryOp::Clip, false, -1.25, 2.0),
        ];

        for &(op, use_pos, p0, p1) in cases {
            let (src_buf, src_host) = if use_pos { (&d_xp, &x_pos) } else { (&d_xs, &x_signed) };
            k.unary_f32(op, src_buf.handle(), d_out.handle(), n, p0, p1)
                .unwrap();
            let mut got = vec![0.0f32; n];
            gpu.download_buffer(&d_out, f32_as_bytes_mut(&mut got)).unwrap();

            let expected: Vec<f32> = src_host
                .iter()
                .map(|&v| match op {
                    UnaryOp::Relu => v.max(0.0),
                    UnaryOp::Sigmoid => 1.0 / (1.0 + (-v).exp()),
                    UnaryOp::SiLU => v / (1.0 + (-v).exp()),
                    UnaryOp::Gelu => {
                        let e = erf_ref((v as f64) * std::f64::consts::FRAC_1_SQRT_2) as f32;
                        0.5 * v * (1.0 + e)
                    }
                    UnaryOp::Erf => erf_ref(v as f64) as f32,
                    UnaryOp::Sqrt => v.sqrt(),
                    UnaryOp::Reciprocal => 1.0 / v,
                    UnaryOp::Abs => v.abs(),
                    UnaryOp::Sin => v.sin(),
                    UnaryOp::Cos => v.cos(),
                    UnaryOp::HardSwish => v * (v / 6.0 + 0.5).clamp(0.0, 1.0),
                    UnaryOp::HardSigmoid => (p0 * v + p1).clamp(0.0, 1.0),
                    UnaryOp::LeakyRelu => if v >= 0.0 { v } else { p0 * v },
                    UnaryOp::Clip => v.clamp(p0, p1),
                })
                .collect();

            let diff = max_abs_diff(&got, &expected);
            assert!(diff < 1e-5, "unary {op:?} max_abs_diff={diff}");
        }

        // PRelu with 4D channel-broadcast slope [1, 4, 1, 1]
        let (bn, bc_ch, bh, bw) = (2usize, 4usize, 5usize, 7usize);
        let total = bn * bc_ch * bh * bw;
        let x4: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.31).sin() * 2.0).collect();
        let slope = vec![0.05f32, 0.25, -0.1, 0.4];
        let d_x4 = gpu.alloc_buffer(total * 4).unwrap();
        let d_s = gpu.alloc_buffer(bc_ch * 4).unwrap();
        let d_y4 = gpu.alloc_buffer(total * 4).unwrap();
        gpu.upload_buffer(&d_x4, f32_as_bytes(&x4)).unwrap();
        gpu.upload_buffer(&d_s, f32_as_bytes(&slope)).unwrap();

        k.prelu_broadcast_f32(
            d_x4.handle(),
            &[bn, bc_ch, bh, bw],
            d_s.handle(),
            &[1, bc_ch, 1, 1],
            d_y4.handle(),
        )
        .unwrap();
        let mut got_p = vec![0.0f32; total];
        gpu.download_buffer(&d_y4, f32_as_bytes_mut(&mut got_p)).unwrap();
        let exp_p: Vec<f32> = x4
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let ch = (i / (bh * bw)) % bc_ch;
                if v >= 0.0 { v } else { v * slope[ch] }
            })
            .collect();
        let diff = max_abs_diff(&got_p, &exp_p);
        assert!(diff < 1e-5, "prelu max_abs_diff={diff}");
    }

    #[cfg(windows)]
    #[test]
    fn gpu_cast_f16_and_i64_parity() {
        let Ok(k) = ElemNormKernels::open() else {
            return;
        };
        let gpu = k.nvidia();

        let vals_f32: Vec<f32> = vec![
            0.0, -0.0, 1.0, -1.0, 0.5, -2.75, 3.140625, 1e-3, -123.5, 65504.0,
        ];
        let n = vals_f32.len();
        let d_f32 = gpu.alloc_buffer(n * 4).unwrap();
        let d_f16 = gpu.alloc_buffer(n * 2).unwrap();
        let d_f32_back = gpu.alloc_buffer(n * 4).unwrap();
        gpu.upload_buffer(&d_f32, f32_as_bytes(&vals_f32)).unwrap();

        k.cast_f32_to_f16(d_f32.handle(), d_f16.handle(), n).unwrap();
        k.cast_f16_to_f32(d_f16.handle(), d_f32_back.handle(), n).unwrap();

        let mut got_f16 = vec![0u16; n];
        let mut got_f32 = vec![0.0f32; n];
        gpu.download_buffer(&d_f16, u16_as_bytes_mut(&mut got_f16)).unwrap();
        gpu.download_buffer(&d_f32_back, f32_as_bytes_mut(&mut got_f32))
            .unwrap();

        for i in 0..n {
            let exp_h = f32_to_f16(vals_f32[i]);
            let exp_f = f16_to_f32(exp_h);
            assert_eq!(got_f16[i], exp_h, "f32->f16 mismatch at {i}");
            assert!((got_f32[i] - exp_f).abs() < 1e-5, "f16->f32 mismatch at {i}");
        }

        // i64 <-> f32
        let vals_i64: Vec<i64> = vec![-100_000, -7, 0, 1, 42, 1_000_000];
        let m = vals_i64.len();
        let d_i64 = gpu.alloc_buffer(m * 8).unwrap();
        let d_f = gpu.alloc_buffer(m * 4).unwrap();
        let d_i64_back = gpu.alloc_buffer(m * 8).unwrap();
        gpu.upload_buffer(&d_i64, i64_as_bytes(&vals_i64)).unwrap();

        k.cast_i64_to_f32(d_i64.handle(), d_f.handle(), m).unwrap();
        k.cast_f32_to_i64(d_f.handle(), d_i64_back.handle(), m).unwrap();

        let mut got_f = vec![0.0f32; m];
        let mut got_i64 = vec![0i64; m];
        gpu.download_buffer(&d_f, f32_as_bytes_mut(&mut got_f)).unwrap();
        gpu.download_buffer(&d_i64_back, i64_as_bytes_mut(&mut got_i64))
            .unwrap();
        for i in 0..m {
            assert!((got_f[i] - vals_i64[i] as f32).abs() < 1e-5);
            assert_eq!(got_i64[i], vals_i64[i]);
        }
    }

    #[cfg(windows)]
    #[test]
    fn gpu_softmax_layernorm_instancenorm_fused_bn_parity() {
        let Ok(k) = ElemNormKernels::open() else {
            return;
        };
        let gpu = k.nvidia();

        // 1. Softmax on arbitrary axis [outer=3, axis_dim=31, inner=5] with large shift
        {
            let (outer, axis_dim, inner) = (3usize, 31usize, 5usize);
            let total = outer * axis_dim * inner;
            let x: Vec<f32> = (0..total)
                .map(|i| 100.0 + ((i as f32) * 0.43).sin() * 4.0)
                .collect();
            let d_x = gpu.alloc_buffer(total * 4).unwrap();
            let d_y = gpu.alloc_buffer(total * 4).unwrap();
            gpu.upload_buffer(&d_x, f32_as_bytes(&x)).unwrap();
            k.softmax_axis_f32(d_x.handle(), d_y.handle(), outer, axis_dim, inner)
                .unwrap();
            let mut got = vec![0.0f32; total];
            gpu.download_buffer(&d_y, f32_as_bytes_mut(&mut got)).unwrap();

            let mut expected = vec![0.0f32; total];
            for o in 0..outer {
                for i in 0..inner {
                    let base = o * axis_dim * inner + i;
                    let mut m = f32::NEG_INFINITY;
                    for a in 0..axis_dim {
                        m = m.max(x[base + a * inner]);
                    }
                    let mut s = 0.0f64;
                    for a in 0..axis_dim {
                        s += ((x[base + a * inner] - m) as f64).exp();
                    }
                    for a in 0..axis_dim {
                        expected[base + a * inner] =
                            (((x[base + a * inner] - m) as f64).exp() / s) as f32;
                    }
                }
            }
            let diff = max_abs_diff(&got, &expected);
            assert!(diff < 1e-5, "softmax max_abs_diff={diff}");
        }

        // 2. LayerNorm [rows=5, norm_dim=384] with and without bias
        {
            let (rows, dim) = (5usize, 384usize);
            let eps = 1e-6f32;
            let x: Vec<f32> = (0..rows * dim)
                .map(|i| ((i as f32) * 0.19).cos() * 3.0 + 1.5)
                .collect();
            let scale: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32) * 0.003).collect();
            let bias: Vec<f32> = (0..dim).map(|i| -0.2 + (i as f32) * 0.001).collect();

            let d_x = gpu.alloc_buffer(x.len() * 4).unwrap();
            let d_s = gpu.alloc_buffer(dim * 4).unwrap();
            let d_b = gpu.alloc_buffer(dim * 4).unwrap();
            let d_y = gpu.alloc_buffer(x.len() * 4).unwrap();
            gpu.upload_buffer(&d_x, f32_as_bytes(&x)).unwrap();
            gpu.upload_buffer(&d_s, f32_as_bytes(&scale)).unwrap();
            gpu.upload_buffer(&d_b, f32_as_bytes(&bias)).unwrap();

            for with_bias in [true, false] {
                k.layer_norm_last_f32(
                    d_x.handle(),
                    Some(d_s.handle()),
                    with_bias.then_some(d_b.handle()),
                    d_y.handle(),
                    rows,
                    dim,
                    eps,
                )
                .unwrap();
                let mut got = vec![0.0f32; rows * dim];
                gpu.download_buffer(&d_y, f32_as_bytes_mut(&mut got)).unwrap();

                let mut expected = vec![0.0f32; rows * dim];
                for r in 0..rows {
                    let row = &x[r * dim..(r + 1) * dim];
                    let mean = row.iter().map(|&v| v as f64).sum::<f64>() / dim as f64;
                    let var = row
                        .iter()
                        .map(|&v| {
                            let d = v as f64 - mean;
                            d * d
                        })
                        .sum::<f64>()
                        / dim as f64;
                    let inv_std = 1.0 / (var as f32 + eps).sqrt();
                    for c in 0..dim {
                        let mut y = (row[c] - mean as f32) * inv_std * scale[c];
                        if with_bias {
                            y += bias[c];
                        }
                        expected[r * dim + c] = y;
                    }
                }
                let diff = max_abs_diff(&got, &expected);
                assert!(diff < 1e-5, "layer_norm(bias={with_bias}) max_abs_diff={diff}");
            }
        }

        // 3. InstanceNorm NCHW [2, 6, 13, 17]
        {
            let (n, c, h, w) = (2usize, 6usize, 13usize, 17usize);
            let hw = h * w;
            let eps = 1e-5f32;
            let x: Vec<f32> = (0..n * c * hw)
                .map(|i| ((i as f32) * 0.27).sin() * 2.5 - 0.7)
                .collect();
            let scale: Vec<f32> = (0..c).map(|i| 0.8 + 0.15 * i as f32).collect();
            let bias: Vec<f32> = (0..c).map(|i| -0.3 + 0.1 * i as f32).collect();

            let d_x = gpu.alloc_buffer(x.len() * 4).unwrap();
            let d_s = gpu.alloc_buffer(c * 4).unwrap();
            let d_b = gpu.alloc_buffer(c * 4).unwrap();
            let d_y = gpu.alloc_buffer(x.len() * 4).unwrap();
            gpu.upload_buffer(&d_x, f32_as_bytes(&x)).unwrap();
            gpu.upload_buffer(&d_s, f32_as_bytes(&scale)).unwrap();
            gpu.upload_buffer(&d_b, f32_as_bytes(&bias)).unwrap();

            k.instance_norm_nchw_f32(
                d_x.handle(),
                Some(d_s.handle()),
                Some(d_b.handle()),
                d_y.handle(),
                n,
                c,
                h,
                w,
                eps,
            )
            .unwrap();
            let mut got = vec![0.0f32; x.len()];
            gpu.download_buffer(&d_y, f32_as_bytes_mut(&mut got)).unwrap();

            let mut expected = vec![0.0f32; x.len()];
            for ni in 0..n {
                for ci in 0..c {
                    let base = (ni * c + ci) * hw;
                    let slice = &x[base..base + hw];
                    let mean = slice.iter().map(|&v| v as f64).sum::<f64>() / hw as f64;
                    let var = slice
                        .iter()
                        .map(|&v| {
                            let d = v as f64 - mean;
                            d * d
                        })
                        .sum::<f64>()
                        / hw as f64;
                    let inv_std = 1.0 / (var as f32 + eps).sqrt();
                    for p in 0..hw {
                        expected[base + p] =
                            (slice[p] - mean as f32) * inv_std * scale[ci] + bias[ci];
                    }
                }
            }
            let diff = max_abs_diff(&got, &expected);
            assert!(diff < 1e-5, "instance_norm max_abs_diff={diff}");
        }

        // 4. BatchNorm Fused NCHW [2, 8, 11, 15]
        {
            let (n, c, h, w) = (2usize, 8usize, 11usize, 15usize);
            let hw = h * w;
            let x: Vec<f32> = (0..n * c * hw)
                .map(|i| ((i as f32) * 0.37).cos() * 3.0)
                .collect();
            let scale_eff: Vec<f32> = (0..c).map(|i| 0.7 + 0.2 * i as f32).collect();
            let bias_eff: Vec<f32> = (0..c).map(|i| -0.4 + 0.12 * i as f32).collect();

            let d_x = gpu.alloc_buffer(x.len() * 4).unwrap();
            let d_s = gpu.alloc_buffer(c * 4).unwrap();
            let d_b = gpu.alloc_buffer(c * 4).unwrap();
            let d_y = gpu.alloc_buffer(x.len() * 4).unwrap();
            gpu.upload_buffer(&d_x, f32_as_bytes(&x)).unwrap();
            gpu.upload_buffer(&d_s, f32_as_bytes(&scale_eff)).unwrap();
            gpu.upload_buffer(&d_b, f32_as_bytes(&bias_eff)).unwrap();

            k.batch_norm_fused_nchw_f32(
                d_x.handle(),
                d_s.handle(),
                d_b.handle(),
                d_y.handle(),
                n,
                c,
                h,
                w,
            )
            .unwrap();
            let mut got = vec![0.0f32; x.len()];
            gpu.download_buffer(&d_y, f32_as_bytes_mut(&mut got)).unwrap();

            let mut expected = vec![0.0f32; x.len()];
            for ni in 0..n {
                for ci in 0..c {
                    let base = (ni * c + ci) * hw;
                    for p in 0..hw {
                        expected[base + p] = x[base + p] * scale_eff[ci] + bias_eff[ci];
                    }
                }
            }
            let diff = max_abs_diff(&got, &expected);
            assert!(diff < 1e-5, "batch_norm_fused max_abs_diff={diff}");
        }
    }
}
