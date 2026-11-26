//! Native CUDA spatial & tensor operator kernels for Aurora Native Inference Engine.
//!
//! Covers ONNX pooling, reduction, resize (3 coordinate modes), constant pad,
//! 2D–5D transpose, concat, slice, gather, scatter_nd, expand, and tile.
//!
//! All dispatch entry points take raw CUDA device addresses (`u64`), making them
//! directly compatible with static VRAM offsets planned by `arena.rs` and
//! zero-dynamic-allocation execution.

#[cfg(windows)]
use std::ffi::c_void;

use crate::cuda::CudaError;
#[cfg(windows)]
use crate::cuda::Nvidia;
use crate::native_infer::graph::{CoordTransformMode, NearestMode, ResizeMode};
use crate::Refusal;

pub const PTX: &str = include_str!("../../native/cuda/infer_spatial_tensor.ptx");

pub const KERNEL_MAXPOOL2D: &str = "maxpool2d_nchw_f32";
pub const KERNEL_AVGPOOL2D: &str = "avgpool2d_nchw_f32";
pub const KERNEL_GLOBAL_AVGPOOL: &str = "global_avgpool_nchw_f32";
pub const KERNEL_REDUCE_MEAN: &str = "reduce_mean_f32";
pub const KERNEL_RESIZE: &str = "resize_nchw_f32";
pub const KERNEL_PAD_CONSTANT: &str = "pad_constant_nchw_f32";
pub const KERNEL_PAD_CONSTANT_4D: &str = "pad_constant_4d_f32";
pub const KERNEL_TRANSPOSE_ND: &str = "transpose_nd_f32";
pub const KERNEL_CONCAT_AXIS: &str = "concat_axis_f32";
pub const KERNEL_SLICE_ND: &str = "slice_nd_f32";
pub const KERNEL_GATHER_AXIS: &str = "gather_axis_f32";
pub const KERNEL_COPY: &str = "copy_f32";
pub const KERNEL_COPY_BYTES: &str = "copy_bytes_u8";
pub const KERNEL_SCATTER_ND: &str = "scatter_nd_f32";
pub const KERNEL_EXPAND_ND: &str = "expand_nd_f32";
pub const KERNEL_TILE_ND: &str = "tile_nd_f32";

#[cfg(windows)]
const BLOCK_1D: u32 = 256;

/// Supported ONNX `Resize` coordinate / interpolation modes audited across all models.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeKernelMode {
    /// `mode=linear`, `coordinate_transformation_mode=half_pixel`
    LinearHalfPixel = 0,
    /// `mode=nearest`, `coordinate_transformation_mode=asymmetric`, `nearest_mode=floor`
    NearestAsymmetricFloor = 1,
    /// `mode=nearest`, `coordinate_transformation_mode=tf_half_pixel_for_nn`, `nearest_mode=floor`
    NearestTfHalfPixelForNnFloor = 2,
}

impl ResizeKernelMode {
    pub fn from_attrs(
        mode: ResizeMode,
        coord_mode: CoordTransformMode,
        nearest_mode: NearestMode,
    ) -> Result<Self, CudaError> {
        match (mode, coord_mode, nearest_mode) {
            (ResizeMode::Linear, CoordTransformMode::HalfPixel, _) => Ok(Self::LinearHalfPixel),
            (ResizeMode::Nearest, CoordTransformMode::Asymmetric, NearestMode::Floor) => {
                Ok(Self::NearestAsymmetricFloor)
            }
            (ResizeMode::Nearest, CoordTransformMode::TfHalfPixelForNn, NearestMode::Floor) => {
                Ok(Self::NearestTfHalfPixelForNnFloor)
            }
            _ => Err(CudaError::Refused(Refusal::NotImplemented(0))),
        }
    }
}

/// Computes ONNX 2D pooling output spatial size for `ceil_mode = false / true`.
pub fn pool2d_out_dim(
    in_dim: usize,
    kernel: usize,
    stride: usize,
    pad_begin: usize,
    pad_end: usize,
    dilation: usize,
    ceil_mode: bool,
) -> Result<usize, CudaError> {
    if kernel == 0 || stride == 0 || dilation == 0 {
        return Err(CudaError::Refused(Refusal::ValueOutOfRange("pool2d_params")));
    }
    let eff_k = dilation * (kernel - 1) + 1;
    let padded = in_dim + pad_begin + pad_end;
    if padded < eff_k {
        return Err(CudaError::Refused(Refusal::ShapeMismatch("pool2d_out_dim")));
    }
    let num = padded - eff_k;
    let out = if ceil_mode {
        (num + stride - 1) / stride + 1
    } else {
        num / stride + 1
    };
    Ok(out)
}

#[cfg(windows)]
#[inline]
fn grid_1d(total: usize) -> (u32, u32) {
    let blocks = ((total as u64) + (BLOCK_1D as u64) - 1) / (BLOCK_1D as u64);
    (blocks.max(1) as u32, 1)
}

#[cfg(windows)]
#[inline]
fn to_i32(val: usize, label: &'static str) -> Result<i32, CudaError> {
    i32::try_from(val).map_err(|_| CudaError::Refused(Refusal::ValueOutOfRange(label)))
}

#[cfg(windows)]
#[inline]
fn to_i32_signed(val: i64, label: &'static str) -> Result<i32, CudaError> {
    i32::try_from(val).map_err(|_| CudaError::Refused(Refusal::ValueOutOfRange(label)))
}

#[cfg(windows)]
fn contiguous_strides(shape: &[usize]) -> Result<Vec<usize>, CudaError> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1]
            .checked_mul(shape[i + 1])
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("tensor_stride")))?;
    }
    Ok(strides)
}

#[cfg(windows)]
fn shape_numel(shape: &[usize]) -> Result<usize, CudaError> {
    shape.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d)
            .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("tensor_numel")))
    })
}

#[cfg(windows)]
fn pad5_dims(shape: &[usize], label: &'static str) -> Result<[i32; 5], CudaError> {
    if shape.len() > 5 {
        return Err(CudaError::Refused(Refusal::ShapeMismatch(label)));
    }
    let mut out = [1i32; 5];
    let offset = 5 - shape.len();
    for (i, &d) in shape.iter().enumerate() {
        out[offset + i] = to_i32(d, label)?;
    }
    Ok(out)
}

// ============================================================================
// Windows CUDA Driver Dispatch Implementation
// ============================================================================

#[cfg(windows)]
pub struct SpatialTensorKernels {
    nvidia: Nvidia,
}

#[cfg(windows)]
impl SpatialTensorKernels {
    /// Loads `infer_spatial_tensor.ptx` into the process-wide CUDA driver domain.
    pub fn open() -> Result<Self, CudaError> {
        let nvidia = Nvidia::open_ptx(PTX)?;
        Ok(Self { nvidia })
    }

    /// Borrow the underlying [`Nvidia`] driver handle (e.g. for buffer allocation or sync).
    pub fn nvidia(&self) -> &Nvidia {
        &self.nvidia
    }

    #[allow(clippy::too_many_arguments)]
    pub fn maxpool2d_nchw_f32(
        &self,
        input: u64,
        output: u64,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        out_h: usize,
        out_w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        pad_t: usize,
        pad_l: usize,
        dh: usize,
        dw: usize,
    ) -> Result<(), CudaError> {
        let total = shape_numel(&[n, c, out_h, out_w])?;
        if total == 0 {
            return Ok(());
        }
        let mut p_in = input;
        let mut p_out = output;
        let mut p_n = to_i32(n, "n")?;
        let mut p_c = to_i32(c, "c")?;
        let mut p_h = to_i32(h, "h")?;
        let mut p_w = to_i32(w, "w")?;
        let mut p_oh = to_i32(out_h, "out_h")?;
        let mut p_ow = to_i32(out_w, "out_w")?;
        let mut p_kh = to_i32(kh, "kh")?;
        let mut p_kw = to_i32(kw, "kw")?;
        let mut p_sh = to_i32(sh, "sh")?;
        let mut p_sw = to_i32(sw, "sw")?;
        let mut p_pt = to_i32(pad_t, "pad_t")?;
        let mut p_pl = to_i32(pad_l, "pad_l")?;
        let mut p_dh = to_i32(dh, "dh")?;
        let mut p_dw = to_i32(dw, "dw")?;

        let mut params: [*mut c_void; 16] = [
            (&mut p_in as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut p_n as *mut i32).cast(),
            (&mut p_c as *mut i32).cast(),
            (&mut p_h as *mut i32).cast(),
            (&mut p_w as *mut i32).cast(),
            (&mut p_oh as *mut i32).cast(),
            (&mut p_ow as *mut i32).cast(),
            (&mut p_kh as *mut i32).cast(),
            (&mut p_kw as *mut i32).cast(),
            (&mut p_sh as *mut i32).cast(),
            (&mut p_sw as *mut i32).cast(),
            (&mut p_pt as *mut i32).cast(),
            (&mut p_pl as *mut i32).cast(),
            (&mut p_dh as *mut i32).cast(),
            (&mut p_dw as *mut i32).cast(),
        ];
        self.nvidia
            .launch_raw(KERNEL_MAXPOOL2D, grid_1d(total), (BLOCK_1D, 1), &mut params)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn avgpool2d_nchw_f32(
        &self,
        input: u64,
        output: u64,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        out_h: usize,
        out_w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        pad_t: usize,
        pad_l: usize,
        count_include_pad: bool,
    ) -> Result<(), CudaError> {
        let total = shape_numel(&[n, c, out_h, out_w])?;
        if total == 0 {
            return Ok(());
        }
        let mut p_in = input;
        let mut p_out = output;
        let mut p_n = to_i32(n, "n")?;
        let mut p_c = to_i32(c, "c")?;
        let mut p_h = to_i32(h, "h")?;
        let mut p_w = to_i32(w, "w")?;
        let mut p_oh = to_i32(out_h, "out_h")?;
        let mut p_ow = to_i32(out_w, "out_w")?;
        let mut p_kh = to_i32(kh, "kh")?;
        let mut p_kw = to_i32(kw, "kw")?;
        let mut p_sh = to_i32(sh, "sh")?;
        let mut p_sw = to_i32(sw, "sw")?;
        let mut p_pt = to_i32(pad_t, "pad_t")?;
        let mut p_pl = to_i32(pad_l, "pad_l")?;
        let mut p_cip: i32 = if count_include_pad { 1 } else { 0 };

        let mut params: [*mut c_void; 15] = [
            (&mut p_in as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut p_n as *mut i32).cast(),
            (&mut p_c as *mut i32).cast(),
            (&mut p_h as *mut i32).cast(),
            (&mut p_w as *mut i32).cast(),
            (&mut p_oh as *mut i32).cast(),
            (&mut p_ow as *mut i32).cast(),
            (&mut p_kh as *mut i32).cast(),
            (&mut p_kw as *mut i32).cast(),
            (&mut p_sh as *mut i32).cast(),
            (&mut p_sw as *mut i32).cast(),
            (&mut p_pt as *mut i32).cast(),
            (&mut p_pl as *mut i32).cast(),
            (&mut p_cip as *mut i32).cast(),
        ];
        self.nvidia
            .launch_raw(KERNEL_AVGPOOL2D, grid_1d(total), (BLOCK_1D, 1), &mut params)?;
        Ok(())
    }

    pub fn global_avgpool_nchw_f32(
        &self,
        input: u64,
        output: u64,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<(), CudaError> {
        let nc = shape_numel(&[n, c])?;
        let spatial = shape_numel(&[h, w])?;
        if nc == 0 {
            return Ok(());
        }
        let mut p_in = input;
        let mut p_out = output;
        let mut p_nc = to_i32(nc, "nc")?;
        let mut p_spatial = to_i32(spatial, "spatial")?;

        let mut params: [*mut c_void; 4] = [
            (&mut p_in as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut p_nc as *mut i32).cast(),
            (&mut p_spatial as *mut i32).cast(),
        ];
        // Shared memory in kernel is fixed at 256 floats: blockDim.x must be 256.
        self.nvidia
            .launch_raw(KERNEL_GLOBAL_AVGPOOL, (nc as u32, 1), (256, 1), &mut params)?;
        Ok(())
    }

    /// Reduces `input` (rank ≤ 5) along the specified `axes` (supports negative axes).
    /// Empty `axes` reduces all axes (ONNX `noop_with_empty_axes=0` semantics).
    pub fn reduce_mean_f32(
        &self,
        input: u64,
        output: u64,
        in_shape: &[usize],
        axes: &[i64],
    ) -> Result<(), CudaError> {
        if in_shape.len() > 5 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("reduce_mean_rank")));
        }
        let rank = in_shape.len();
        let offset = 5 - rank;
        let mut mask: i32 = 0;
        if axes.is_empty() {
            for i in 0..rank {
                mask |= 1 << (offset + i);
            }
        } else {
            for &ax in axes {
                let norm = if ax < 0 { ax + rank as i64 } else { ax };
                if norm < 0 || norm >= rank as i64 {
                    return Err(CudaError::Refused(Refusal::ValueOutOfRange("reduce_mean_axis")));
                }
                mask |= 1 << (offset + norm as usize);
            }
        }

        let dims = pad5_dims(in_shape, "reduce_mean_shape")?;
        let mut out_total: usize = 1;
        for i in 0..5 {
            if (mask & (1 << i)) == 0 {
                out_total = out_total
                    .checked_mul(dims[i] as usize)
                    .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("reduce_mean_out")))?;
            }
        }
        if out_total == 0 {
            return Ok(());
        }

        let mut p_in = input;
        let mut p_out = output;
        let mut d0 = dims[0];
        let mut d1 = dims[1];
        let mut d2 = dims[2];
        let mut d3 = dims[3];
        let mut d4 = dims[4];
        let mut p_mask = mask;
        let mut p_total = to_i32(out_total, "out_total")?;

        let mut params: [*mut c_void; 9] = [
            (&mut p_in as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut d0 as *mut i32).cast(),
            (&mut d1 as *mut i32).cast(),
            (&mut d2 as *mut i32).cast(),
            (&mut d3 as *mut i32).cast(),
            (&mut d4 as *mut i32).cast(),
            (&mut p_mask as *mut i32).cast(),
            (&mut p_total as *mut i32).cast(),
        ];
        self.nvidia
            .launch_raw(KERNEL_REDUCE_MEAN, grid_1d(out_total), (BLOCK_1D, 1), &mut params)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn resize_nchw_f32(
        &self,
        input: u64,
        output: u64,
        n: usize,
        c: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        mode: ResizeKernelMode,
    ) -> Result<(), CudaError> {
        let total = shape_numel(&[n, c, out_h, out_w])?;
        if total == 0 {
            return Ok(());
        }
        if in_h == 0 || in_w == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("resize_in_spatial")));
        }
        let mut p_in = input;
        let mut p_out = output;
        let mut p_n = to_i32(n, "n")?;
        let mut p_c = to_i32(c, "c")?;
        let mut p_ih = to_i32(in_h, "in_h")?;
        let mut p_iw = to_i32(in_w, "in_w")?;
        let mut p_oh = to_i32(out_h, "out_h")?;
        let mut p_ow = to_i32(out_w, "out_w")?;
        let mut p_mode = mode as i32;

        let mut params: [*mut c_void; 9] = [
            (&mut p_in as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut p_n as *mut i32).cast(),
            (&mut p_c as *mut i32).cast(),
            (&mut p_ih as *mut i32).cast(),
            (&mut p_iw as *mut i32).cast(),
            (&mut p_oh as *mut i32).cast(),
            (&mut p_ow as *mut i32).cast(),
            (&mut p_mode as *mut i32).cast(),
        ];
        self.nvidia
            .launch_raw(KERNEL_RESIZE, grid_1d(total), (BLOCK_1D, 1), &mut params)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn pad_constant_nchw_f32(
        &self,
        input: u64,
        output: u64,
        n: usize,
        c: usize,
        in_h: usize,
        in_w: usize,
        pad_t: usize,
        pad_b: usize,
        pad_l: usize,
        pad_r: usize,
        const_val: f32,
    ) -> Result<(), CudaError> {
        let out_h = in_h + pad_t + pad_b;
        let out_w = in_w + pad_l + pad_r;
        let total = shape_numel(&[n, c, out_h, out_w])?;
        if total == 0 {
            return Ok(());
        }
        let mut p_in = input;
        let mut p_out = output;
        let mut p_n = to_i32(n, "n")?;
        let mut p_c = to_i32(c, "c")?;
        let mut p_ih = to_i32(in_h, "in_h")?;
        let mut p_iw = to_i32(in_w, "in_w")?;
        let mut p_pt = to_i32(pad_t, "pad_t")?;
        let mut p_pb = to_i32(pad_b, "pad_b")?;
        let mut p_pl = to_i32(pad_l, "pad_l")?;
        let mut p_pr = to_i32(pad_r, "pad_r")?;
        let mut p_val = const_val;

        let mut params: [*mut c_void; 11] = [
            (&mut p_in as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut p_n as *mut i32).cast(),
            (&mut p_c as *mut i32).cast(),
            (&mut p_ih as *mut i32).cast(),
            (&mut p_iw as *mut i32).cast(),
            (&mut p_pt as *mut i32).cast(),
            (&mut p_pb as *mut i32).cast(),
            (&mut p_pl as *mut i32).cast(),
            (&mut p_pr as *mut i32).cast(),
            (&mut p_val as *mut f32).cast(),
        ];
        self.nvidia
            .launch_raw(KERNEL_PAD_CONSTANT, grid_1d(total), (BLOCK_1D, 1), &mut params)?;
        Ok(())
    }

    /// General 4D NCHW constant padding supporting padding along all 4 dimensions (N, C, H, W).
    #[allow(clippy::too_many_arguments)]
    pub fn pad_constant_4d_f32(
        &self,
        input: u64,
        output: u64,
        in_shape: [usize; 4],
        pad_begin: [usize; 4],
        pad_end: [usize; 4],
        const_val: f32,
    ) -> Result<(), CudaError> {
        let out_n = in_shape[0] + pad_begin[0] + pad_end[0];
        let out_c = in_shape[1] + pad_begin[1] + pad_end[1];
        let out_h = in_shape[2] + pad_begin[2] + pad_end[2];
        let out_w = in_shape[3] + pad_begin[3] + pad_end[3];
        let total = shape_numel(&[out_n, out_c, out_h, out_w])?;
        if total == 0 {
            return Ok(());
        }
        let mut p_in = input;
        let mut p_out = output;
        let mut p_in_n = to_i32(in_shape[0], "in_n")?;
        let mut p_in_c = to_i32(in_shape[1], "in_c")?;
        let mut p_in_h = to_i32(in_shape[2], "in_h")?;
        let mut p_in_w = to_i32(in_shape[3], "in_w")?;
        let mut p_out_n = to_i32(out_n, "out_n")?;
        let mut p_out_c = to_i32(out_c, "out_c")?;
        let mut p_out_h = to_i32(out_h, "out_h")?;
        let mut p_out_w = to_i32(out_w, "out_w")?;
        let mut p_pn0 = to_i32(pad_begin[0], "pad_n0")?;
        let mut p_pc0 = to_i32(pad_begin[1], "pad_c0")?;
        let mut p_ph0 = to_i32(pad_begin[2], "pad_h0")?;
        let mut p_pw0 = to_i32(pad_begin[3], "pad_w0")?;
        let mut p_val = const_val;

        let mut params: [*mut c_void; 15] = [
            (&mut p_in as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut p_in_n as *mut i32).cast(),
            (&mut p_in_c as *mut i32).cast(),
            (&mut p_in_h as *mut i32).cast(),
            (&mut p_in_w as *mut i32).cast(),
            (&mut p_out_n as *mut i32).cast(),
            (&mut p_out_c as *mut i32).cast(),
            (&mut p_out_h as *mut i32).cast(),
            (&mut p_out_w as *mut i32).cast(),
            (&mut p_pn0 as *mut i32).cast(),
            (&mut p_pc0 as *mut i32).cast(),
            (&mut p_ph0 as *mut i32).cast(),
            (&mut p_pw0 as *mut i32).cast(),
            (&mut p_val as *mut f32).cast(),
        ];
        self.nvidia
            .launch_raw(KERNEL_PAD_CONSTANT_4D, grid_1d(total), (BLOCK_1D, 1), &mut params)?;
        Ok(())
    }

    /// General 1D–5D transpose with arbitrary permutation `perm`.
    pub fn transpose_nd_f32(
        &self,
        input: u64,
        output: u64,
        in_shape: &[usize],
        perm: &[usize],
    ) -> Result<(), CudaError> {
        let rank = in_shape.len();
        if rank == 0 || rank > 5 || perm.len() != rank {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("transpose_nd")));
        }
        let mut seen = [false; 5];
        for &p in perm {
            if p >= rank || seen[p] {
                return Err(CudaError::Refused(Refusal::ValueOutOfRange("transpose_perm")));
            }
            seen[p] = true;
        }
        let total = shape_numel(in_shape)?;
        if total == 0 {
            return Ok(());
        }

        let in_strides = contiguous_strides(in_shape)?;
        let mut out_d = [1i32; 5];
        let mut mapped_s = [0i32; 5];
        let offset = 5 - rank;
        for (i, &p) in perm.iter().enumerate() {
            out_d[offset + i] = to_i32(in_shape[p], "transpose_out_d")?;
            mapped_s[offset + i] = to_i32(in_strides[p], "transpose_in_s")?;
        }

        let mut p_in = input;
        let mut p_out = output;
        let mut p_total = to_i32(total, "total")?;

        let mut params: [*mut c_void; 13] = [
            (&mut p_in as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut out_d[0] as *mut i32).cast(),
            (&mut out_d[1] as *mut i32).cast(),
            (&mut out_d[2] as *mut i32).cast(),
            (&mut out_d[3] as *mut i32).cast(),
            (&mut out_d[4] as *mut i32).cast(),
            (&mut mapped_s[0] as *mut i32).cast(),
            (&mut mapped_s[1] as *mut i32).cast(),
            (&mut mapped_s[2] as *mut i32).cast(),
            (&mut mapped_s[3] as *mut i32).cast(),
            (&mut mapped_s[4] as *mut i32).cast(),
            (&mut p_total as *mut i32).cast(),
        ];
        self.nvidia
            .launch_raw(KERNEL_TRANSPOSE_ND, grid_1d(total), (BLOCK_1D, 1), &mut params)?;
        Ok(())
    }

    /// Copies one source tensor segment into `dst` at `dst_axis_offset` along the concat axis.
    #[allow(clippy::too_many_arguments)]
    pub fn concat_segment_f32(
        &self,
        src: u64,
        dst: u64,
        outer: usize,
        src_axis_dim: usize,
        dst_axis_dim: usize,
        dst_axis_offset: usize,
        inner: usize,
    ) -> Result<(), CudaError> {
        let total = shape_numel(&[outer, src_axis_dim, inner])?;
        if total == 0 {
            return Ok(());
        }
        if dst_axis_offset + src_axis_dim > dst_axis_dim {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("concat_segment")));
        }
        let mut p_src = src;
        let mut p_dst = dst;
        let mut p_outer = to_i32(outer, "outer")?;
        let mut p_src_ax = to_i32(src_axis_dim, "src_axis_dim")?;
        let mut p_dst_ax = to_i32(dst_axis_dim, "dst_axis_dim")?;
        let mut p_dst_off = to_i32(dst_axis_offset, "dst_axis_offset")?;
        let mut p_inner = to_i32(inner, "inner")?;

        let mut params: [*mut c_void; 7] = [
            (&mut p_src as *mut u64).cast(),
            (&mut p_dst as *mut u64).cast(),
            (&mut p_outer as *mut i32).cast(),
            (&mut p_src_ax as *mut i32).cast(),
            (&mut p_dst_ax as *mut i32).cast(),
            (&mut p_dst_off as *mut i32).cast(),
            (&mut p_inner as *mut i32).cast(),
        ];
        self.nvidia
            .launch_raw(KERNEL_CONCAT_AXIS, grid_1d(total), (BLOCK_1D, 1), &mut params)?;
        Ok(())
    }

    /// Concatenates multiple `(src_ptr, src_axis_dim)` segments into `dst` (`outer × sum_axis × inner`).
    pub fn concat_axis_f32(
        &self,
        inputs: &[(u64, usize)],
        dst: u64,
        outer: usize,
        inner: usize,
    ) -> Result<(), CudaError> {
        let dst_axis_dim: usize = inputs.iter().map(|&(_, d)| d).sum();
        let mut offset = 0usize;
        for &(src, src_axis_dim) in inputs {
            self.concat_segment_f32(src, dst, outer, src_axis_dim, dst_axis_dim, offset, inner)?;
            offset += src_axis_dim;
        }
        Ok(())
    }

    /// General ≤5D strided slice supporting positive or negative `steps`.
    /// `starts` and `steps` must already be normalized to valid indices in `in_shape`.
    pub fn slice_nd_f32(
        &self,
        input: u64,
        output: u64,
        in_shape: &[usize],
        out_shape: &[usize],
        starts: &[i64],
        steps: &[i64],
    ) -> Result<(), CudaError> {
        let rank = in_shape.len();
        if rank == 0
            || rank > 5
            || out_shape.len() != rank
            || starts.len() != rank
            || steps.len() != rank
        {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("slice_nd")));
        }
        let total = shape_numel(out_shape)?;
        if total == 0 {
            return Ok(());
        }
        let in_strides = contiguous_strides(in_shape)?;
        let offset = 5 - rank;
        let mut out_d = [1i32; 5];
        let mut in_s = [0i32; 5];
        let mut st = [0i32; 5];
        let mut sp = [1i32; 5];
        for i in 0..rank {
            out_d[offset + i] = to_i32(out_shape[i], "slice_out_d")?;
            in_s[offset + i] = to_i32(in_strides[i], "slice_in_s")?;
            st[offset + i] = to_i32_signed(starts[i], "slice_start")?;
            sp[offset + i] = to_i32_signed(steps[i], "slice_step")?;
        }

        let mut p_in = input;
        let mut p_out = output;
        let mut p_total = to_i32(total, "total")?;

        let mut params: [*mut c_void; 23] = [
            (&mut p_in as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut out_d[0] as *mut i32).cast(),
            (&mut out_d[1] as *mut i32).cast(),
            (&mut out_d[2] as *mut i32).cast(),
            (&mut out_d[3] as *mut i32).cast(),
            (&mut out_d[4] as *mut i32).cast(),
            (&mut in_s[0] as *mut i32).cast(),
            (&mut in_s[1] as *mut i32).cast(),
            (&mut in_s[2] as *mut i32).cast(),
            (&mut in_s[3] as *mut i32).cast(),
            (&mut in_s[4] as *mut i32).cast(),
            (&mut st[0] as *mut i32).cast(),
            (&mut st[1] as *mut i32).cast(),
            (&mut st[2] as *mut i32).cast(),
            (&mut st[3] as *mut i32).cast(),
            (&mut st[4] as *mut i32).cast(),
            (&mut sp[0] as *mut i32).cast(),
            (&mut sp[1] as *mut i32).cast(),
            (&mut sp[2] as *mut i32).cast(),
            (&mut sp[3] as *mut i32).cast(),
            (&mut sp[4] as *mut i32).cast(),
            (&mut p_total as *mut i32).cast(),
        ];
        self.nvidia
            .launch_raw(KERNEL_SLICE_ND, grid_1d(total), (BLOCK_1D, 1), &mut params)?;
        Ok(())
    }

    /// Gathers slices along an axis using device-resident `i64` indices (handles negative indices).
    #[allow(clippy::too_many_arguments)]
    pub fn gather_axis_f32(
        &self,
        data: u64,
        indices_i64: u64,
        output: u64,
        outer: usize,
        axis_dim: usize,
        num_indices: usize,
        inner: usize,
    ) -> Result<(), CudaError> {
        let total = shape_numel(&[outer, num_indices, inner])?;
        if total == 0 {
            return Ok(());
        }
        if axis_dim == 0 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("gather_axis_dim")));
        }
        let mut p_data = data;
        let mut p_idx = indices_i64;
        let mut p_out = output;
        let mut p_outer = to_i32(outer, "outer")?;
        let mut p_ax = to_i32(axis_dim, "axis_dim")?;
        let mut p_num = to_i32(num_indices, "num_indices")?;
        let mut p_inner = to_i32(inner, "inner")?;

        let mut params: [*mut c_void; 7] = [
            (&mut p_data as *mut u64).cast(),
            (&mut p_idx as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut p_outer as *mut i32).cast(),
            (&mut p_ax as *mut i32).cast(),
            (&mut p_num as *mut i32).cast(),
            (&mut p_inner as *mut i32).cast(),
        ];
        self.nvidia
            .launch_raw(KERNEL_GATHER_AXIS, grid_1d(total), (BLOCK_1D, 1), &mut params)?;
        Ok(())
    }

    /// Device-to-device `f32` copy kernel.
    pub fn copy_f32(&self, src: u64, dst: u64, count: usize) -> Result<(), CudaError> {
        if count == 0 || src == dst {
            return Ok(());
        }
        let mut p_src = src;
        let mut p_dst = dst;
        let mut p_total = to_i32(count, "copy_count")?;
        let mut params: [*mut c_void; 3] = [
            (&mut p_src as *mut u64).cast(),
            (&mut p_dst as *mut u64).cast(),
            (&mut p_total as *mut i32).cast(),
        ];
        self.nvidia
            .launch_raw(KERNEL_COPY, grid_1d(count), (BLOCK_1D, 1), &mut params)?;
        Ok(())
    }

    /// Device-to-device exact byte copy kernel (safe for all dtypes including U8/F16/I64).
    pub fn copy_bytes_u8(&self, src: u64, dst: u64, total_bytes: usize) -> Result<(), CudaError> {
        if total_bytes == 0 || src == dst {
            return Ok(());
        }
        let mut p_src = src;
        let mut p_dst = dst;
        let mut p_total = to_i32(total_bytes, "copy_bytes")?;
        let mut params: [*mut c_void; 3] = [
            (&mut p_src as *mut u64).cast(),
            (&mut p_dst as *mut u64).cast(),
            (&mut p_total as *mut i32).cast(),
        ];
        self.nvidia
            .launch_raw(KERNEL_COPY_BYTES, grid_1d(total_bytes), (BLOCK_1D, 1), &mut params)?;
        Ok(())
    }

    /// ONNX `ScatterND` (`reduction=none`).
    ///
    /// If `output != data`, copies `data` to `output` first via `copy_f32`, then
    /// scatters `updates` into `output` using `indices_i64` (`[num_updates, index_depth_k]`).
    #[allow(clippy::too_many_arguments)]
    pub fn scatter_nd_f32(
        &self,
        data: u64,
        indices_i64: u64,
        updates: u64,
        output: u64,
        data_shape: &[usize],
        num_updates: usize,
        index_depth_k: usize,
    ) -> Result<(), CudaError> {
        let rank = data_shape.len();
        if rank == 0 || rank > 5 || index_depth_k == 0 || index_depth_k > rank {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("scatter_nd")));
        }
        let data_numel = shape_numel(data_shape)?;
        if data != output {
            self.copy_f32(data, output, data_numel)?;
        }
        let slice_size = shape_numel(&data_shape[index_depth_k..])?;
        let total = shape_numel(&[num_updates, slice_size])?;
        if total == 0 {
            return Ok(());
        }
        let strides = contiguous_strides(data_shape)?;
        let mut d = [1i32; 5];
        let mut s = [0i32; 5];
        for i in 0..rank {
            d[i] = to_i32(data_shape[i], "scatter_dim")?;
            s[i] = to_i32(strides[i], "scatter_stride")?;
        }

        let mut p_idx = indices_i64;
        let mut p_upd = updates;
        let mut p_out = output;
        let mut p_nu = to_i32(num_updates, "num_updates")?;
        let mut p_k = to_i32(index_depth_k, "index_depth_k")?;
        let mut p_ss = to_i32(slice_size, "slice_size")?;

        let mut params: [*mut c_void; 16] = [
            (&mut p_idx as *mut u64).cast(),
            (&mut p_upd as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut p_nu as *mut i32).cast(),
            (&mut p_k as *mut i32).cast(),
            (&mut p_ss as *mut i32).cast(),
            (&mut d[0] as *mut i32).cast(),
            (&mut d[1] as *mut i32).cast(),
            (&mut d[2] as *mut i32).cast(),
            (&mut d[3] as *mut i32).cast(),
            (&mut d[4] as *mut i32).cast(),
            (&mut s[0] as *mut i32).cast(),
            (&mut s[1] as *mut i32).cast(),
            (&mut s[2] as *mut i32).cast(),
            (&mut s[3] as *mut i32).cast(),
            (&mut s[4] as *mut i32).cast(),
        ];
        self.nvidia
            .launch_raw(KERNEL_SCATTER_ND, grid_1d(total), (BLOCK_1D, 1), &mut params)?;
        Ok(())
    }

    /// NumPy-style broadcast expand for rank ≤ 5.
    pub fn expand_nd_f32(
        &self,
        input: u64,
        output: u64,
        in_shape: &[usize],
        out_shape: &[usize],
    ) -> Result<(), CudaError> {
        if in_shape.len() > 5 || out_shape.len() > 5 || out_shape.len() < in_shape.len() {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("expand_nd_rank")));
        }
        let total = shape_numel(out_shape)?;
        if total == 0 {
            return Ok(());
        }
        let in_strides = contiguous_strides(in_shape)?;
        let out_rank = out_shape.len();
        let in_rank = in_shape.len();
        let out_offset = 5 - out_rank;
        let rank_diff = out_rank - in_rank;

        let mut out_d = [1i32; 5];
        let mut bcast_s = [0i32; 5];
        for i in 0..out_rank {
            out_d[out_offset + i] = to_i32(out_shape[i], "expand_out_d")?;
            if i >= rank_diff {
                let in_i = i - rank_diff;
                let in_dim = in_shape[in_i];
                if in_dim == out_shape[i] {
                    bcast_s[out_offset + i] = to_i32(in_strides[in_i], "expand_in_s")?;
                } else if in_dim == 1 {
                    bcast_s[out_offset + i] = 0;
                } else {
                    return Err(CudaError::Refused(Refusal::ShapeMismatch("expand_broadcast")));
                }
            }
        }

        let mut p_in = input;
        let mut p_out = output;
        let mut p_total = to_i32(total, "total")?;

        let mut params: [*mut c_void; 13] = [
            (&mut p_in as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut out_d[0] as *mut i32).cast(),
            (&mut out_d[1] as *mut i32).cast(),
            (&mut out_d[2] as *mut i32).cast(),
            (&mut out_d[3] as *mut i32).cast(),
            (&mut out_d[4] as *mut i32).cast(),
            (&mut bcast_s[0] as *mut i32).cast(),
            (&mut bcast_s[1] as *mut i32).cast(),
            (&mut bcast_s[2] as *mut i32).cast(),
            (&mut bcast_s[3] as *mut i32).cast(),
            (&mut bcast_s[4] as *mut i32).cast(),
            (&mut p_total as *mut i32).cast(),
        ];
        self.nvidia
            .launch_raw(KERNEL_EXPAND_ND, grid_1d(total), (BLOCK_1D, 1), &mut params)?;
        Ok(())
    }

    /// Periodic tensor tiling for rank ≤ 5.
    pub fn tile_nd_f32(
        &self,
        input: u64,
        output: u64,
        in_shape: &[usize],
        repeats: &[usize],
    ) -> Result<(), CudaError> {
        let rank = in_shape.len();
        if rank == 0 || rank > 5 || repeats.len() != rank {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("tile_nd")));
        }
        let mut out_shape = Vec::with_capacity(rank);
        for i in 0..rank {
            out_shape.push(
                in_shape[i]
                    .checked_mul(repeats[i])
                    .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("tile_out_dim")))?,
            );
        }
        let total = shape_numel(&out_shape)?;
        if total == 0 {
            return Ok(());
        }
        let in_strides = contiguous_strides(in_shape)?;
        let offset = 5 - rank;
        let mut out_d = [1i32; 5];
        let mut in_d = [1i32; 5];
        let mut in_s = [0i32; 5];
        for i in 0..rank {
            out_d[offset + i] = to_i32(out_shape[i], "tile_out_d")?;
            in_d[offset + i] = to_i32(in_shape[i], "tile_in_d")?;
            in_s[offset + i] = to_i32(in_strides[i], "tile_in_s")?;
        }

        let mut p_in = input;
        let mut p_out = output;
        let mut p_total = to_i32(total, "total")?;

        let mut params: [*mut c_void; 18] = [
            (&mut p_in as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut out_d[0] as *mut i32).cast(),
            (&mut out_d[1] as *mut i32).cast(),
            (&mut out_d[2] as *mut i32).cast(),
            (&mut out_d[3] as *mut i32).cast(),
            (&mut out_d[4] as *mut i32).cast(),
            (&mut in_d[0] as *mut i32).cast(),
            (&mut in_d[1] as *mut i32).cast(),
            (&mut in_d[2] as *mut i32).cast(),
            (&mut in_d[3] as *mut i32).cast(),
            (&mut in_d[4] as *mut i32).cast(),
            (&mut in_s[0] as *mut i32).cast(),
            (&mut in_s[1] as *mut i32).cast(),
            (&mut in_s[2] as *mut i32).cast(),
            (&mut in_s[3] as *mut i32).cast(),
            (&mut in_s[4] as *mut i32).cast(),
            (&mut p_total as *mut i32).cast(),
        ];
        self.nvidia
            .launch_raw(KERNEL_TILE_ND, grid_1d(total), (BLOCK_1D, 1), &mut params)?;
        Ok(())
    }
}

// ============================================================================
// Non-Windows Stub
// ============================================================================

#[cfg(not(windows))]
pub struct SpatialTensorKernels {
    _private: (),
}

#[cfg(not(windows))]
impl SpatialTensorKernels {
    fn unavailable() -> CudaError {
        CudaError::Unavailable("CUDA driver (non-Windows build)")
    }

    pub fn open() -> Result<Self, CudaError> {
        Err(Self::unavailable())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn maxpool2d_nchw_f32(
        &self,
        _input: u64,
        _output: u64,
        _n: usize,
        _c: usize,
        _h: usize,
        _w: usize,
        _out_h: usize,
        _out_w: usize,
        _kh: usize,
        _kw: usize,
        _sh: usize,
        _sw: usize,
        _pad_t: usize,
        _pad_l: usize,
        _dh: usize,
        _dw: usize,
    ) -> Result<(), CudaError> {
        Err(Self::unavailable())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn avgpool2d_nchw_f32(
        &self,
        _input: u64,
        _output: u64,
        _n: usize,
        _c: usize,
        _h: usize,
        _w: usize,
        _out_h: usize,
        _out_w: usize,
        _kh: usize,
        _kw: usize,
        _sh: usize,
        _sw: usize,
        _pad_t: usize,
        _pad_l: usize,
        _count_include_pad: bool,
    ) -> Result<(), CudaError> {
        Err(Self::unavailable())
    }

    pub fn global_avgpool_nchw_f32(
        &self,
        _input: u64,
        _output: u64,
        _n: usize,
        _c: usize,
        _h: usize,
        _w: usize,
    ) -> Result<(), CudaError> {
        Err(Self::unavailable())
    }

    pub fn reduce_mean_f32(
        &self,
        _input: u64,
        _output: u64,
        _in_shape: &[usize],
        _axes: &[i64],
    ) -> Result<(), CudaError> {
        Err(Self::unavailable())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn resize_nchw_f32(
        &self,
        _input: u64,
        _output: u64,
        _n: usize,
        _c: usize,
        _in_h: usize,
        _in_w: usize,
        _out_h: usize,
        _out_w: usize,
        _mode: ResizeKernelMode,
    ) -> Result<(), CudaError> {
        Err(Self::unavailable())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn pad_constant_nchw_f32(
        &self,
        _input: u64,
        _output: u64,
        _n: usize,
        _c: usize,
        _in_h: usize,
        _in_w: usize,
        _pad_t: usize,
        _pad_b: usize,
        _pad_l: usize,
        _pad_r: usize,
        _const_val: f32,
    ) -> Result<(), CudaError> {
        Err(Self::unavailable())
    }

    pub fn transpose_nd_f32(
        &self,
        _input: u64,
        _output: u64,
        _in_shape: &[usize],
        _perm: &[usize],
    ) -> Result<(), CudaError> {
        Err(Self::unavailable())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn concat_segment_f32(
        &self,
        _src: u64,
        _dst: u64,
        _outer: usize,
        _src_axis_dim: usize,
        _dst_axis_dim: usize,
        _dst_axis_offset: usize,
        _inner: usize,
    ) -> Result<(), CudaError> {
        Err(Self::unavailable())
    }

    pub fn concat_axis_f32(
        &self,
        _inputs: &[(u64, usize)],
        _dst: u64,
        _outer: usize,
        _inner: usize,
    ) -> Result<(), CudaError> {
        Err(Self::unavailable())
    }

    pub fn slice_nd_f32(
        &self,
        _input: u64,
        _output: u64,
        _in_shape: &[usize],
        _out_shape: &[usize],
        _starts: &[i64],
        _steps: &[i64],
    ) -> Result<(), CudaError> {
        Err(Self::unavailable())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gather_axis_f32(
        &self,
        _data: u64,
        _indices_i64: u64,
        _output: u64,
        _outer: usize,
        _axis_dim: usize,
        _num_indices: usize,
        _inner: usize,
    ) -> Result<(), CudaError> {
        Err(Self::unavailable())
    }

    pub fn copy_f32(&self, _src: u64, _dst: u64, _count: usize) -> Result<(), CudaError> {
        Err(Self::unavailable())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn scatter_nd_f32(
        &self,
        _data: u64,
        _indices_i64: u64,
        _updates: u64,
        _output: u64,
        _data_shape: &[usize],
        _num_updates: usize,
        _index_depth_k: usize,
    ) -> Result<(), CudaError> {
        Err(Self::unavailable())
    }

    pub fn expand_nd_f32(
        &self,
        _input: u64,
        _output: u64,
        _in_shape: &[usize],
        _out_shape: &[usize],
    ) -> Result<(), CudaError> {
        Err(Self::unavailable())
    }

    pub fn tile_nd_f32(
        &self,
        _input: u64,
        _output: u64,
        _in_shape: &[usize],
        _repeats: &[usize],
    ) -> Result<(), CudaError> {
        Err(Self::unavailable())
    }
}

// ============================================================================
// Unit Tests: CPU Reference vs Live CUDA Execution (max_abs_diff < 1e-5)
// ============================================================================

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    fn f32_to_bytes(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn i64_to_bytes(vals: &[i64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 8);
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn make_data(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.37 + seed).sin() * 3.5) - 0.8)
            .collect()
    }

    fn assert_close(got: &[f32], expected: &[f32], tol: f32, tag: &str) {
        assert_eq!(got.len(), expected.len(), "{tag}: length mismatch");
        let mut max_diff = 0.0f32;
        let mut worst_idx = 0usize;
        for (i, (&a, &b)) in got.iter().zip(expected.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
                worst_idx = i;
            }
        }
        assert!(
            max_diff < tol,
            "{tag}: max_abs_diff={max_diff} >= {tol} at idx={worst_idx} (got={}, expected={})",
            got[worst_idx],
            expected[worst_idx]
        );
    }

    fn open_or_skip() -> Option<SpatialTensorKernels> {
        match SpatialTensorKernels::open() {
            Ok(k) => Some(k),
            Err(CudaError::Unavailable(_)) => None,
            Err(e) => panic!("unexpected CUDA error: {e:?}"),
        }
    }

    #[test]
    fn test_maxpool2d_ceil_mode_and_negative_values() {
        let Some(k) = open_or_skip() else { return };
        let (n, c, h, w) = (2, 3, 5, 7);
        let (kh, kw, sh, sw, pad_t, pad_l, pad_b, pad_r) = (2, 2, 2, 2, 0, 0, 0, 0);
        let out_h = pool2d_out_dim(h, kh, sh, pad_t, pad_b, 1, true).unwrap();
        let out_w = pool2d_out_dim(w, kw, sw, pad_l, pad_r, 1, true).unwrap();
        assert_eq!((out_h, out_w), (3, 4)); // ceil_mode=1 on 5x7 with k=2,s=2

        // All negative values to verify padded region is never compared as 0.0.
        let input: Vec<f32> = make_data(n * c * h * w, 1.0)
            .into_iter()
            .map(|v| -v.abs() - 0.5)
            .collect();

        let mut expected = vec![0.0f32; n * c * out_h * out_w];
        for bi in 0..n {
            for ci in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut best = f32::NEG_INFINITY;
                        for r in 0..kh {
                            let ih = (oh * sh + r) as isize - pad_t as isize;
                            if ih < 0 || ih >= h as isize {
                                continue;
                            }
                            for s in 0..kw {
                                let iw = (ow * sw + s) as isize - pad_l as isize;
                                if iw < 0 || iw >= w as isize {
                                    continue;
                                }
                                let v = input[((bi * c + ci) * h + ih as usize) * w + iw as usize];
                                if v > best {
                                    best = v;
                                }
                            }
                        }
                        expected[((bi * c + ci) * out_h + oh) * out_w + ow] = best;
                    }
                }
            }
        }

        let nv = k.nvidia();
        let d_in = nv.alloc_buffer(input.len() * 4).unwrap();
        let d_out = nv.alloc_buffer(expected.len() * 4).unwrap();
        nv.upload_buffer(&d_in, &f32_to_bytes(&input)).unwrap();
        k.maxpool2d_nchw_f32(
            d_in.handle(),
            d_out.handle(),
            n,
            c,
            h,
            w,
            out_h,
            out_w,
            kh,
            kw,
            sh,
            sw,
            pad_t,
            pad_l,
            1,
            1,
        )
        .unwrap();
        let mut raw = vec![0u8; expected.len() * 4];
        nv.download_buffer(&d_out, &mut raw).unwrap();
        assert_close(&bytes_to_f32(&raw), &expected, 1e-5, "maxpool2d_ceil1");
    }

    #[test]
    fn test_avgpool2d_include_and_exclude_pad() {
        let Some(k) = open_or_skip() else { return };
        let (n, c, h, w) = (1, 4, 6, 7);
        let (kh, kw, sh, sw, pad_t, pad_l, pad_b, pad_r) = (3, 3, 2, 2, 1, 1, 1, 1);
        let out_h = pool2d_out_dim(h, kh, sh, pad_t, pad_b, 1, false).unwrap();
        let out_w = pool2d_out_dim(w, kw, sw, pad_l, pad_r, 1, false).unwrap();
        let input = make_data(n * c * h * w, 2.0);

        let nv = k.nvidia();
        let d_in = nv.alloc_buffer(input.len() * 4).unwrap();
        let d_out = nv.alloc_buffer(n * c * out_h * out_w * 4).unwrap();
        nv.upload_buffer(&d_in, &f32_to_bytes(&input)).unwrap();

        for include_pad in [false, true] {
            let mut expected = vec![0.0f32; n * c * out_h * out_w];
            for bi in 0..n {
                for ci in 0..c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = 0.0f32;
                            let mut cnt = 0usize;
                            for r in 0..kh {
                                let ih = (oh * sh + r) as isize - pad_t as isize;
                                if ih < 0 || ih >= h as isize {
                                    continue;
                                }
                                for s in 0..kw {
                                    let iw = (ow * sw + s) as isize - pad_l as isize;
                                    if iw < 0 || iw >= w as isize {
                                        continue;
                                    }
                                    sum += input[((bi * c + ci) * h + ih as usize) * w + iw as usize];
                                    cnt += 1;
                                }
                            }
                            let denom = if include_pad { kh * kw } else { cnt };
                            expected[((bi * c + ci) * out_h + oh) * out_w + ow] = sum / denom as f32;
                        }
                    }
                }
            }

            k.avgpool2d_nchw_f32(
                d_in.handle(),
                d_out.handle(),
                n,
                c,
                h,
                w,
                out_h,
                out_w,
                kh,
                kw,
                sh,
                sw,
                pad_t,
                pad_l,
                include_pad,
            )
            .unwrap();
            let mut raw = vec![0u8; expected.len() * 4];
            nv.download_buffer(&d_out, &mut raw).unwrap();
            assert_close(
                &bytes_to_f32(&raw),
                &expected,
                1e-5,
                if include_pad { "avgpool_inc_pad" } else { "avgpool_exc_pad" },
            );
        }
    }

    #[test]
    fn test_global_avgpool_and_reduce_mean() {
        let Some(k) = open_or_skip() else { return };
        let (n, c, h, w) = (2, 5, 19, 23);
        let input = make_data(n * c * h * w, 3.0);
        let nv = k.nvidia();
        let d_in = nv.alloc_buffer(input.len() * 4).unwrap();
        nv.upload_buffer(&d_in, &f32_to_bytes(&input)).unwrap();

        // 1. GlobalAveragePool
        let mut expected_gap = vec![0.0f32; n * c];
        let spatial = h * w;
        for nc in 0..(n * c) {
            let sum: f64 = input[nc * spatial..(nc + 1) * spatial]
                .iter()
                .map(|&v| v as f64)
                .sum();
            expected_gap[nc] = (sum / spatial as f64) as f32;
        }
        let d_gap = nv.alloc_buffer(expected_gap.len() * 4).unwrap();
        k.global_avgpool_nchw_f32(d_in.handle(), d_gap.handle(), n, c, h, w)
            .unwrap();
        let mut raw_gap = vec![0u8; expected_gap.len() * 4];
        nv.download_buffer(&d_gap, &mut raw_gap).unwrap();
        assert_close(&bytes_to_f32(&raw_gap), &expected_gap, 1e-5, "global_avgpool");

        // 2. ReduceMean over H, W (axes=[2, 3]) -> matches GlobalAveragePool
        let d_rm_hw = nv.alloc_buffer(expected_gap.len() * 4).unwrap();
        k.reduce_mean_f32(d_in.handle(), d_rm_hw.handle(), &[n, c, h, w], &[2, 3])
            .unwrap();
        let mut raw_rm_hw = vec![0u8; expected_gap.len() * 4];
        nv.download_buffer(&d_rm_hw, &mut raw_rm_hw).unwrap();
        assert_close(&bytes_to_f32(&raw_rm_hw), &expected_gap, 1e-5, "reduce_mean_hw");

        // 3. ReduceMean over C (axes=[1])
        let mut expected_c = vec![0.0f32; n * h * w];
        for bi in 0..n {
            for hi in 0..h {
                for wi in 0..w {
                    let mut sum = 0.0f64;
                    for ci in 0..c {
                        sum += input[((bi * c + ci) * h + hi) * w + wi] as f64;
                    }
                    expected_c[(bi * h + hi) * w + wi] = (sum / c as f64) as f32;
                }
            }
        }
        let d_rm_c = nv.alloc_buffer(expected_c.len() * 4).unwrap();
        k.reduce_mean_f32(d_in.handle(), d_rm_c.handle(), &[n, c, h, w], &[1])
            .unwrap();
        let mut raw_rm_c = vec![0u8; expected_c.len() * 4];
        nv.download_buffer(&d_rm_c, &mut raw_rm_c).unwrap();
        assert_close(&bytes_to_f32(&raw_rm_c), &expected_c, 1e-5, "reduce_mean_c");
    }

    #[test]
    fn test_resize_all_three_onnx_modes() {
        let Some(k) = open_or_skip() else { return };
        let (n, c, in_h, in_w) = (1, 3, 5, 6);
        let (out_h, out_w) = (11, 9);
        let input = make_data(n * c * in_h * in_w, 4.0);
        let nv = k.nvidia();
        let d_in = nv.alloc_buffer(input.len() * 4).unwrap();
        let d_out = nv.alloc_buffer(n * c * out_h * out_w * 4).unwrap();
        nv.upload_buffer(&d_in, &f32_to_bytes(&input)).unwrap();

        let scale_h = out_h as f64 / in_h as f64;
        let scale_w = out_w as f64 / in_w as f64;

        for mode in [
            ResizeKernelMode::LinearHalfPixel,
            ResizeKernelMode::NearestAsymmetricFloor,
            ResizeKernelMode::NearestTfHalfPixelForNnFloor,
        ] {
            let mut expected = vec![0.0f32; n * c * out_h * out_w];
            for nc in 0..(n * c) {
                let plane = &input[nc * in_h * in_w..(nc + 1) * in_h * in_w];
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let val = match mode {
                            ResizeKernelMode::LinearHalfPixel => {
                                let fy = (((oh as f64 + 0.5) / scale_h) - 0.5)
                                    .clamp(0.0, (in_h - 1) as f64);
                                let fx = (((ow as f64 + 0.5) / scale_w) - 0.5)
                                    .clamp(0.0, (in_w - 1) as f64);
                                let y0 = fy.floor() as usize;
                                let x0 = fx.floor() as usize;
                                let y1 = (y0 + 1).min(in_h - 1);
                                let x1 = (x0 + 1).min(in_w - 1);
                                let wy = (fy - y0 as f64) as f32;
                                let wx = (fx - x0 as f64) as f32;
                                let v00 = plane[y0 * in_w + x0];
                                let v01 = plane[y0 * in_w + x1];
                                let v10 = plane[y1 * in_w + x0];
                                let v11 = plane[y1 * in_w + x1];
                                let top = v00 + wx * (v01 - v00);
                                let bot = v10 + wx * (v11 - v10);
                                top + wy * (bot - top)
                            }
                            ResizeKernelMode::NearestAsymmetricFloor => {
                                let iy = ((oh as f64 / scale_h).floor() as isize)
                                    .clamp(0, (in_h - 1) as isize) as usize;
                                let ix = ((ow as f64 / scale_w).floor() as isize)
                                    .clamp(0, (in_w - 1) as isize) as usize;
                                plane[iy * in_w + ix]
                            }
                            ResizeKernelMode::NearestTfHalfPixelForNnFloor => {
                                let iy = (((oh as f64 + 0.5) / scale_h).floor() as isize)
                                    .clamp(0, (in_h - 1) as isize) as usize;
                                let ix = (((ow as f64 + 0.5) / scale_w).floor() as isize)
                                    .clamp(0, (in_w - 1) as isize) as usize;
                                plane[iy * in_w + ix]
                            }
                        };
                        expected[(nc * out_h + oh) * out_w + ow] = val;
                    }
                }
            }

            k.resize_nchw_f32(
                d_in.handle(),
                d_out.handle(),
                n,
                c,
                in_h,
                in_w,
                out_h,
                out_w,
                mode,
            )
            .unwrap();
            let mut raw = vec![0u8; expected.len() * 4];
            nv.download_buffer(&d_out, &mut raw).unwrap();
            assert_close(&bytes_to_f32(&raw), &expected, 1e-5, &format!("resize_{mode:?}"));
        }
    }

    #[test]
    fn test_pad_constant_nchw() {
        let Some(k) = open_or_skip() else { return };
        let (n, c, in_h, in_w) = (2, 2, 4, 5);
        let (pt, pb, pl, pr) = (1, 2, 3, 1);
        let const_val = -7.25f32;
        let out_h = in_h + pt + pb;
        let out_w = in_w + pl + pr;
        let input = make_data(n * c * in_h * in_w, 5.0);

        let mut expected = vec![const_val; n * c * out_h * out_w];
        for nc in 0..(n * c) {
            for ih in 0..in_h {
                for iw in 0..in_w {
                    expected[(nc * out_h + (ih + pt)) * out_w + (iw + pl)] =
                        input[(nc * in_h + ih) * in_w + iw];
                }
            }
        }

        let nv = k.nvidia();
        let d_in = nv.alloc_buffer(input.len() * 4).unwrap();
        let d_out = nv.alloc_buffer(expected.len() * 4).unwrap();
        nv.upload_buffer(&d_in, &f32_to_bytes(&input)).unwrap();
        k.pad_constant_nchw_f32(
            d_in.handle(),
            d_out.handle(),
            n,
            c,
            in_h,
            in_w,
            pt,
            pb,
            pl,
            pr,
            const_val,
        )
        .unwrap();
        let mut raw = vec![0u8; expected.len() * 4];
        nv.download_buffer(&d_out, &mut raw).unwrap();
        assert_close(&bytes_to_f32(&raw), &expected, 1e-5, "pad_constant");
    }

    #[test]
    fn test_transpose_5d_and_4d() {
        let Some(k) = open_or_skip() else { return };
        // 5D perm [2, 0, 3, 1, 4] (used 60x in ViT/SAM QKV split)
        let in_shape = [2, 3, 4, 5, 2];
        let perm = [2, 0, 3, 1, 4];
        let total = shape_numel(&in_shape).unwrap();
        let input = make_data(total, 6.0);
        let in_s = contiguous_strides(&in_shape).unwrap();
        let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();
        let out_s = contiguous_strides(&out_shape).unwrap();

        let mut expected = vec![0.0f32; total];
        for i0 in 0..in_shape[0] {
            for i1 in 0..in_shape[1] {
                for i2 in 0..in_shape[2] {
                    for i3 in 0..in_shape[3] {
                        for i4 in 0..in_shape[4] {
                            let coords = [i0, i1, i2, i3, i4];
                            let src_i = i0 * in_s[0]
                                + i1 * in_s[1]
                                + i2 * in_s[2]
                                + i3 * in_s[3]
                                + i4 * in_s[4];
                            let dst_i = coords[perm[0]] * out_s[0]
                                + coords[perm[1]] * out_s[1]
                                + coords[perm[2]] * out_s[2]
                                + coords[perm[3]] * out_s[3]
                                + coords[perm[4]] * out_s[4];
                            expected[dst_i] = input[src_i];
                        }
                    }
                }
            }
        }

        let nv = k.nvidia();
        let d_in = nv.alloc_buffer(total * 4).unwrap();
        let d_out = nv.alloc_buffer(total * 4).unwrap();
        nv.upload_buffer(&d_in, &f32_to_bytes(&input)).unwrap();
        k.transpose_nd_f32(d_in.handle(), d_out.handle(), &in_shape, &perm)
            .unwrap();
        let mut raw = vec![0u8; total * 4];
        nv.download_buffer(&d_out, &mut raw).unwrap();
        assert_close(&bytes_to_f32(&raw), &expected, 1e-5, "transpose_5d");
    }

    #[test]
    fn test_concat_slice_gather_scatter_expand_tile() {
        let Some(k) = open_or_skip() else { return };
        let nv = k.nvidia();

        // --- 1. Concat along axis=1: [2, 3, 4] + [2, 5, 4] -> [2, 8, 4] ---
        let a = make_data(2 * 3 * 4, 7.0);
        let b = make_data(2 * 5 * 4, 8.0);
        let mut exp_cat = vec![0.0f32; 2 * 8 * 4];
        for o in 0..2 {
            for ax in 0..3 {
                for i in 0..4 {
                    exp_cat[(o * 8 + ax) * 4 + i] = a[(o * 3 + ax) * 4 + i];
                }
            }
            for ax in 0..5 {
                for i in 0..4 {
                    exp_cat[(o * 8 + (3 + ax)) * 4 + i] = b[(o * 5 + ax) * 4 + i];
                }
            }
        }
        let d_a = nv.alloc_buffer(a.len() * 4).unwrap();
        let d_b = nv.alloc_buffer(b.len() * 4).unwrap();
        let d_cat = nv.alloc_buffer(exp_cat.len() * 4).unwrap();
        nv.upload_buffer(&d_a, &f32_to_bytes(&a)).unwrap();
        nv.upload_buffer(&d_b, &f32_to_bytes(&b)).unwrap();
        k.concat_axis_f32(&[(d_a.handle(), 3), (d_b.handle(), 5)], d_cat.handle(), 2, 4)
            .unwrap();
        let mut raw_cat = vec![0u8; exp_cat.len() * 4];
        nv.download_buffer(&d_cat, &mut raw_cat).unwrap();
        assert_close(&bytes_to_f32(&raw_cat), &exp_cat, 1e-5, "concat_axis");

        // --- 2. Slice ND with positive & negative steps: [2, 8, 4] -> [2, 3, 2] ---
        let sl_in_shape = [2, 8, 4];
        let sl_out_shape = [2, 3, 2];
        let starts = [0i64, 6, 1];
        let steps = [1i64, -2, 2];
        let mut exp_sl = vec![0.0f32; 2 * 3 * 2];
        for o0 in 0..2 {
            for o1 in 0..3 {
                for o2 in 0..2 {
                    let i0 = (starts[0] + o0 as i64 * steps[0]) as usize;
                    let i1 = (starts[1] + o1 as i64 * steps[1]) as usize;
                    let i2 = (starts[2] + o2 as i64 * steps[2]) as usize;
                    exp_sl[(o0 * 3 + o1) * 2 + o2] = exp_cat[(i0 * 8 + i1) * 4 + i2];
                }
            }
        }
        let d_sl = nv.alloc_buffer(exp_sl.len() * 4).unwrap();
        k.slice_nd_f32(
            d_cat.handle(),
            d_sl.handle(),
            &sl_in_shape,
            &sl_out_shape,
            &starts,
            &steps,
        )
        .unwrap();
        let mut raw_sl = vec![0u8; exp_sl.len() * 4];
        nv.download_buffer(&d_sl, &mut raw_sl).unwrap();
        assert_close(&bytes_to_f32(&raw_sl), &exp_sl, 1e-5, "slice_nd_neg_step");

        // --- 3. Gather axis=1 with negative indices ---
        let g_indices = vec![0i64, -1, 3, -8];
        let mut exp_g = vec![0.0f32; 2 * 4 * 4];
        for o in 0..2 {
            for (k_i, &raw_idx) in g_indices.iter().enumerate() {
                let norm = (if raw_idx < 0 { raw_idx + 8 } else { raw_idx }) as usize;
                for i in 0..4 {
                    exp_g[(o * 4 + k_i) * 4 + i] = exp_cat[(o * 8 + norm) * 4 + i];
                }
            }
        }
        let d_g_idx = nv.alloc_buffer(g_indices.len() * 8).unwrap();
        let d_g_out = nv.alloc_buffer(exp_g.len() * 4).unwrap();
        nv.upload_buffer(&d_g_idx, &i64_to_bytes(&g_indices)).unwrap();
        k.gather_axis_f32(d_cat.handle(), d_g_idx.handle(), d_g_out.handle(), 2, 8, 4, 4)
            .unwrap();
        let mut raw_g = vec![0u8; exp_g.len() * 4];
        nv.download_buffer(&d_g_out, &mut raw_g).unwrap();
        assert_close(&bytes_to_f32(&raw_g), &exp_g, 1e-5, "gather_axis_neg");

        // --- 4. ScatterND (k=2 on 3D [2, 8, 4], slice_size=4, with negative index) ---
        let sc_indices = vec![0i64, 2, 1, -1, 0, 5]; // 3 updates of depth 2
        let sc_updates = make_data(3 * 4, 9.0);
        let mut exp_sc = exp_cat.clone();
        for u in 0..3 {
            let mut i0 = sc_indices[u * 2];
            let mut i1 = sc_indices[u * 2 + 1];
            if i0 < 0 { i0 += 2; }
            if i1 < 0 { i1 += 8; }
            let base = (i0 as usize * 8 + i1 as usize) * 4;
            for s in 0..4 {
                exp_sc[base + s] = sc_updates[u * 4 + s];
            }
        }
        let d_sc_idx = nv.alloc_buffer(sc_indices.len() * 8).unwrap();
        let d_sc_upd = nv.alloc_buffer(sc_updates.len() * 4).unwrap();
        let d_sc_out = nv.alloc_buffer(exp_sc.len() * 4).unwrap();
        nv.upload_buffer(&d_sc_idx, &i64_to_bytes(&sc_indices)).unwrap();
        nv.upload_buffer(&d_sc_upd, &f32_to_bytes(&sc_updates)).unwrap();
        k.scatter_nd_f32(
            d_cat.handle(),
            d_sc_idx.handle(),
            d_sc_upd.handle(),
            d_sc_out.handle(),
            &[2, 8, 4],
            3,
            2,
        )
        .unwrap();
        let mut raw_sc = vec![0u8; exp_sc.len() * 4];
        nv.download_buffer(&d_sc_out, &mut raw_sc).unwrap();
        assert_close(&bytes_to_f32(&raw_sc), &exp_sc, 1e-5, "scatter_nd");

        // --- 5. Expand ND: [1, 3, 1] -> [2, 3, 4] ---
        let ex_in = vec![1.5f32, -2.5, 4.25];
        let mut exp_ex = vec![0.0f32; 2 * 3 * 4];
        for o0 in 0..2 {
            for o1 in 0..3 {
                for o2 in 0..4 {
                    exp_ex[(o0 * 3 + o1) * 4 + o2] = ex_in[o1];
                }
            }
        }
        let d_ex_in = nv.alloc_buffer(ex_in.len() * 4).unwrap();
        let d_ex_out = nv.alloc_buffer(exp_ex.len() * 4).unwrap();
        nv.upload_buffer(&d_ex_in, &f32_to_bytes(&ex_in)).unwrap();
        k.expand_nd_f32(d_ex_in.handle(), d_ex_out.handle(), &[1, 3, 1], &[2, 3, 4])
            .unwrap();
        let mut raw_ex = vec![0u8; exp_ex.len() * 4];
        nv.download_buffer(&d_ex_out, &mut raw_ex).unwrap();
        assert_close(&bytes_to_f32(&raw_ex), &exp_ex, 1e-5, "expand_nd");

        // --- 6. Tile ND: [2, 3] with repeats [2, 3] -> [4, 9] ---
        let tl_in = make_data(6, 10.0);
        let mut exp_tl = vec![0.0f32; 4 * 9];
        for r in 0..4 {
            for c in 0..9 {
                exp_tl[r * 9 + c] = tl_in[(r % 2) * 3 + (c % 3)];
            }
        }
        let d_tl_in = nv.alloc_buffer(tl_in.len() * 4).unwrap();
        let d_tl_out = nv.alloc_buffer(exp_tl.len() * 4).unwrap();
        nv.upload_buffer(&d_tl_in, &f32_to_bytes(&tl_in)).unwrap();
        k.tile_nd_f32(d_tl_in.handle(), d_tl_out.handle(), &[2, 3], &[2, 3])
            .unwrap();
        let mut raw_tl = vec![0u8; exp_tl.len() * 4];
        nv.download_buffer(&d_tl_out, &mut raw_tl).unwrap();
        assert_close(&bytes_to_f32(&raw_tl), &exp_tl, 1e-5, "tile_nd");
    }
}
