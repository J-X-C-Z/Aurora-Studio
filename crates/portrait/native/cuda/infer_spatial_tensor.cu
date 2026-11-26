// Aurora Native Inference Engine — Spatial & Tensor Operator CUDA Kernels
// Compiled to PTX with:
//   nvcc --ptx -O3 --fmad=false --prec-div=true --prec-sqrt=true -arch=compute_75 -Xcompiler /wd4819
//
// Zero external DLL / header dependencies. All kernels take raw device pointers
// compatible with static VRAM arena offsets (`u64` CUdeviceptr).

// ============================================================================
// 1. Pooling & Reduction
// ============================================================================

extern "C" __global__ void maxpool2d_nchw_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int H,
    int W,
    int out_h,
    int out_w,
    int kh,
    int kw,
    int sh,
    int sw,
    int pad_t,
    int pad_l,
    int dh,
    int dw
) {
    int total = N * C * out_h * out_w;
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;

    int ow = idx % out_w;
    int tmp = idx / out_w;
    int oh = tmp % out_h;
    int nc = tmp / out_h;

    int h_start = oh * sh - pad_t;
    int w_start = ow * sw - pad_l;
    const float* in_plane = input + (size_t)nc * (size_t)(H * W);

    float best = -3.402823466e+38f;
    int has_val = 0;
    for (int r = 0; r < kh; ++r) {
        int ih = h_start + r * dh;
        if (ih < 0 || ih >= H) continue;
        int row_off = ih * W;
        for (int s = 0; s < kw; ++s) {
            int iw = w_start + s * dw;
            if (iw < 0 || iw >= W) continue;
            float v = in_plane[row_off + iw];
            if (!has_val || v > best) {
                best = v;
                has_val = 1;
            }
        }
    }
    output[idx] = best;
}

extern "C" __global__ void avgpool2d_nchw_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int H,
    int W,
    int out_h,
    int out_w,
    int kh,
    int kw,
    int sh,
    int sw,
    int pad_t,
    int pad_l,
    int count_include_pad
) {
    int total = N * C * out_h * out_w;
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;

    int ow = idx % out_w;
    int tmp = idx / out_w;
    int oh = tmp % out_h;
    int nc = tmp / out_h;

    int h_start = oh * sh - pad_t;
    int w_start = ow * sw - pad_l;
    const float* in_plane = input + (size_t)nc * (size_t)(H * W);

    float sum = 0.0f;
    int valid_count = 0;
    for (int r = 0; r < kh; ++r) {
        int ih = h_start + r;
        if (ih < 0 || ih >= H) continue;
        int row_off = ih * W;
        for (int s = 0; s < kw; ++s) {
            int iw = w_start + s;
            if (iw < 0 || iw >= W) continue;
            sum += in_plane[row_off + iw];
            valid_count += 1;
        }
    }

    int denom = (count_include_pad != 0) ? (kh * kw) : valid_count;
    output[idx] = (denom > 0) ? (sum / (float)denom) : 0.0f;
}

extern "C" __global__ void global_avgpool_nchw_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int NC,
    int spatial
) {
    int nc = (int)blockIdx.x;
    if (nc >= NC) return;

    __shared__ float sdata[256];
    int tid = (int)threadIdx.x;
    const float* in_plane = input + (size_t)nc * (size_t)spatial;

    float sum = 0.0f;
    for (int i = tid; i < spatial; i += (int)blockDim.x) {
        sum += in_plane[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int stride = (int)(blockDim.x >> 1); stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[nc] = (spatial > 0) ? (sdata[0] / (float)spatial) : 0.0f;
    }
}

// reduction: 0 = mean, 1 = sum, 2 = max
extern "C" __global__ void reduce_axis_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_d0,
    int in_d1,
    int in_d2,
    int in_d3,
    int in_d4,
    int reduce_mask,
    int out_total,
    int reduction
) {
    int out_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (out_idx >= out_total) return;

    int out_d0 = ((reduce_mask & 1) != 0) ? 1 : in_d0;
    int out_d1 = ((reduce_mask & 2) != 0) ? 1 : in_d1;
    int out_d2 = ((reduce_mask & 4) != 0) ? 1 : in_d2;
    int out_d3 = ((reduce_mask & 8) != 0) ? 1 : in_d3;
    int out_d4 = ((reduce_mask & 16) != 0) ? 1 : in_d4;

    int red_d0 = ((reduce_mask & 1) != 0) ? in_d0 : 1;
    int red_d1 = ((reduce_mask & 2) != 0) ? in_d1 : 1;
    int red_d2 = ((reduce_mask & 4) != 0) ? in_d2 : 1;
    int red_d3 = ((reduce_mask & 8) != 0) ? in_d3 : 1;
    int red_d4 = ((reduce_mask & 16) != 0) ? in_d4 : 1;

    int tmp = out_idx;
    int b4 = tmp % out_d4; tmp /= out_d4;
    int b3 = tmp % out_d3; tmp /= out_d3;
    int b2 = tmp % out_d2; tmp /= out_d2;
    int b1 = tmp % out_d1; tmp /= out_d1;
    int b0 = tmp % out_d0;

    int s4 = 1;
    int s3 = in_d4;
    int s2 = s3 * in_d3;
    int s1 = s2 * in_d2;
    int s0 = s1 * in_d1;

    double sum = 0.0;
    float maximum = __int_as_float(0xff800000u);
    for (int i0 = 0; i0 < red_d0; ++i0) {
        int o0 = (b0 + i0) * s0;
        for (int i1 = 0; i1 < red_d1; ++i1) {
            int o1 = o0 + (b1 + i1) * s1;
            for (int i2 = 0; i2 < red_d2; ++i2) {
                int o2 = o1 + (b2 + i2) * s2;
                for (int i3 = 0; i3 < red_d3; ++i3) {
                    int o3 = o2 + (b3 + i3) * s3;
                    for (int i4 = 0; i4 < red_d4; ++i4) {
                        float value = input[o3 + (b4 + i4) * s4];
                        if (reduction == 2) {
                            if (isnan(value)) {
                                output[out_idx] = value;
                                return;
                            }
                            maximum = fmaxf(maximum, value);
                        } else {
                            sum += (double)value;
                        }
                    }
                }
            }
        }
    }

    int red_total = red_d0 * red_d1 * red_d2 * red_d3 * red_d4;
    if (reduction == 2) {
        output[out_idx] = maximum;
    } else if (reduction == 1) {
        output[out_idx] = (float)sum;
    } else {
        output[out_idx] = red_total > 0 ? (float)(sum / (double)red_total) : 0.0f;
    }
}

// ============================================================================
// 2. Resize & Pad (Strict ONNX coordinate transformation modes)
// ============================================================================

// mode = 0: linear + half_pixel
// mode = 1: nearest + asymmetric + floor
// mode = 2: nearest + tf_half_pixel_for_nn + floor
extern "C" __global__ void resize_nchw_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int mode
) {
    int total = N * C * out_h * out_w;
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;

    int ow = idx % out_w;
    int tmp = idx / out_w;
    int oh = tmp % out_h;
    int nc = tmp / out_h;

    const float* in_plane = input + (size_t)nc * (size_t)(in_h * in_w);
    double scale_h = (double)out_h / (double)in_h;
    double scale_w = (double)out_w / (double)in_w;

    if (mode == 0 || mode == 3) {
        double fy;
        double fx;
        if (mode == 3) {
            // ONNX align_corners maps the first and last output centers to the input endpoints.
            fy = out_h > 1 ? (double)oh * (in_h - 1) / (out_h - 1) : 0.0;
            fx = out_w > 1 ? (double)ow * (in_w - 1) / (out_w - 1) : 0.0;
        } else {
            fy = ((double)oh + 0.5) / scale_h - 0.5;
            fx = ((double)ow + 0.5) / scale_w - 0.5;
        }

        if (fy < 0.0) fy = 0.0;
        if (fy > (double)(in_h - 1)) fy = (double)(in_h - 1);
        if (fx < 0.0) fx = 0.0;
        if (fx > (double)(in_w - 1)) fx = (double)(in_w - 1);

        int y0 = (int)floor(fy);
        int x0 = (int)floor(fx);
        int y1 = (y0 + 1 < in_h) ? (y0 + 1) : (in_h - 1);
        int x1 = (x0 + 1 < in_w) ? (x0 + 1) : (in_w - 1);

        float wy = (float)(fy - (double)y0);
        float wx = (float)(fx - (double)x0);

        float v00 = in_plane[y0 * in_w + x0];
        float v01 = in_plane[y0 * in_w + x1];
        float v10 = in_plane[y1 * in_w + x0];
        float v11 = in_plane[y1 * in_w + x1];

        float top = v00 + wx * (v01 - v00);
        float bot = v10 + wx * (v11 - v10);
        output[idx] = top + wy * (bot - top);
    } else if (mode == 1) {
        // ONNX nearest + asymmetric + floor:
        // x_orig = x_resized / scale
        double fy = (double)oh / scale_h;
        double fx = (double)ow / scale_w;

        int iy = (int)floor(fy);
        int ix = (int)floor(fx);
        if (iy < 0) iy = 0;
        if (iy >= in_h) iy = in_h - 1;
        if (ix < 0) ix = 0;
        if (ix >= in_w) ix = in_w - 1;

        output[idx] = in_plane[iy * in_w + ix];
    } else {
        // mode == 2: ONNX nearest + tf_half_pixel_for_nn + floor:
        // x_orig = (x_resized + 0.5) / scale
        double fy = ((double)oh + 0.5) / scale_h;
        double fx = ((double)ow + 0.5) / scale_w;

        int iy = (int)floor(fy);
        int ix = (int)floor(fx);
        if (iy < 0) iy = 0;
        if (iy >= in_h) iy = in_h - 1;
        if (ix < 0) ix = 0;
        if (ix >= in_w) ix = in_w - 1;

        output[idx] = in_plane[iy * in_w + ix];
    }
}

extern "C" __global__ void pad_constant_nchw_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int in_h,
    int in_w,
    int pad_t,
    int pad_b,
    int pad_l,
    int pad_r,
    float const_val
) {
    int out_h = in_h + pad_t + pad_b;
    int out_w = in_w + pad_l + pad_r;
    int total = N * C * out_h * out_w;
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;

    int ow = idx % out_w;
    int tmp = idx / out_w;
    int oh = tmp % out_h;
    int nc = tmp / out_h;

    int ih = oh - pad_t;
    int iw = ow - pad_l;
    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
        output[idx] = input[(size_t)nc * (size_t)(in_h * in_w) + (size_t)(ih * in_w + iw)];
    } else {
        output[idx] = const_val;
    }
}

extern "C" __global__ void pad_constant_4d_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_n,
    int in_c,
    int in_h,
    int in_w,
    int out_n,
    int out_c,
    int out_h,
    int out_w,
    int pad_n0,
    int pad_c0,
    int pad_h0,
    int pad_w0,
    float const_val
) {
    int total = out_n * out_c * out_h * out_w;
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;

    int ow = idx % out_w;
    int tmp = idx / out_w;
    int oh = tmp % out_h;
    tmp /= out_h;
    int oc = tmp % out_c;
    int on = tmp / out_c;

    int in_ni = on - pad_n0;
    int in_ci = oc - pad_c0;
    int in_hi = oh - pad_h0;
    int in_wi = ow - pad_w0;

    if (in_ni >= 0 && in_ni < in_n &&
        in_ci >= 0 && in_ci < in_c &&
        in_hi >= 0 && in_hi < in_h &&
        in_wi >= 0 && in_wi < in_w) {
        size_t src_idx = (((size_t)in_ni * (size_t)in_c + (size_t)in_ci) * (size_t)in_h + (size_t)in_hi) * (size_t)in_w + (size_t)in_wi;
        output[idx] = input[src_idx];
    } else {
        output[idx] = const_val;
    }
}

// ============================================================================
// 3. Tensor Layout Rearrangement & Indexing
// ============================================================================

extern "C" __global__ void transpose_nd_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int out_d0,
    int out_d1,
    int out_d2,
    int out_d3,
    int out_d4,
    int in_s0,
    int in_s1,
    int in_s2,
    int in_s3,
    int in_s4,
    int total
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;

    int tmp = idx;
    int o4 = tmp % out_d4; tmp /= out_d4;
    int o3 = tmp % out_d3; tmp /= out_d3;
    int o2 = tmp % out_d2; tmp /= out_d2;
    int o1 = tmp % out_d1; tmp /= out_d1;
    int o0 = tmp % out_d0;

    int in_idx = o0 * in_s0 + o1 * in_s1 + o2 * in_s2 + o3 * in_s3 + o4 * in_s4;
    output[idx] = input[in_idx];
}

extern "C" __global__ void concat_axis_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int outer,
    int src_axis_dim,
    int dst_axis_dim,
    int dst_axis_offset,
    int inner
) {
    int total = outer * src_axis_dim * inner;
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;

    int i = idx % inner;
    int tmp = idx / inner;
    int a = tmp % src_axis_dim;
    int o = tmp / src_axis_dim;

    size_t dst_idx = ((size_t)o * (size_t)dst_axis_dim + (size_t)(dst_axis_offset + a)) * (size_t)inner + (size_t)i;
    dst[dst_idx] = src[idx];
}

extern "C" __global__ void slice_nd_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int out_d0,
    int out_d1,
    int out_d2,
    int out_d3,
    int out_d4,
    int in_s0,
    int in_s1,
    int in_s2,
    int in_s3,
    int in_s4,
    int start0,
    int start1,
    int start2,
    int start3,
    int start4,
    int step0,
    int step1,
    int step2,
    int step3,
    int step4,
    int total
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;

    int tmp = idx;
    int o4 = tmp % out_d4; tmp /= out_d4;
    int o3 = tmp % out_d3; tmp /= out_d3;
    int o2 = tmp % out_d2; tmp /= out_d2;
    int o1 = tmp % out_d1; tmp /= out_d1;
    int o0 = tmp % out_d0;

    int in_idx = (start0 + o0 * step0) * in_s0
               + (start1 + o1 * step1) * in_s1
               + (start2 + o2 * step2) * in_s2
               + (start3 + o3 * step3) * in_s3
               + (start4 + o4 * step4) * in_s4;
    output[idx] = input[in_idx];
}

extern "C" __global__ void gather_axis_f32(
    const float* __restrict__ data,
    const long long* __restrict__ indices,
    float* __restrict__ output,
    int outer,
    int axis_dim,
    int num_indices,
    int inner
) {
    int total = outer * num_indices * inner;
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;

    int i = idx % inner;
    int tmp = idx / inner;
    int k = tmp % num_indices;
    int o = tmp / num_indices;

    long long raw = indices[k];
    if (raw < 0) raw += (long long)axis_dim;
    if (raw < 0) raw = 0;
    if (raw >= (long long)axis_dim) raw = (long long)(axis_dim - 1);

    size_t src_idx = ((size_t)o * (size_t)axis_dim + (size_t)raw) * (size_t)inner + (size_t)i;
    output[idx] = data[src_idx];
}

extern "C" __global__ void copy_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int total
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;
    dst[idx] = src[idx];
}

extern "C" __global__ void copy_bytes_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int total_bytes
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_bytes) return;
    dst[idx] = src[idx];
}

extern "C" __global__ void scatter_nd_f32(
    const long long* __restrict__ indices,
    const float* __restrict__ updates,
    float* __restrict__ output,
    int num_updates,
    int k,
    int slice_size,
    int d0,
    int d1,
    int d2,
    int d3,
    int d4,
    int s0,
    int s1,
    int s2,
    int s3,
    int s4
) {
    int total = num_updates * slice_size;
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= total) return;

    int u = tid / slice_size;
    int s = tid % slice_size;

    int dims[5] = { d0, d1, d2, d3, d4 };
    int strides[5] = { s0, s1, s2, s3, s4 };

    size_t dst_base = 0;
    const long long* idx_ptr = indices + (size_t)u * (size_t)k;
    for (int j = 0; j < k; ++j) {
        long long v = idx_ptr[j];
        if (v < 0) v += (long long)dims[j];
        if (v < 0) v = 0;
        if (v >= (long long)dims[j]) v = (long long)(dims[j] - 1);
        dst_base += (size_t)v * (size_t)strides[j];
    }

    output[dst_base + (size_t)s] = updates[tid];
}

extern "C" __global__ void expand_nd_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int out_d0,
    int out_d1,
    int out_d2,
    int out_d3,
    int out_d4,
    int in_s0,
    int in_s1,
    int in_s2,
    int in_s3,
    int in_s4,
    int total
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;

    int tmp = idx;
    int o4 = tmp % out_d4; tmp /= out_d4;
    int o3 = tmp % out_d3; tmp /= out_d3;
    int o2 = tmp % out_d2; tmp /= out_d2;
    int o1 = tmp % out_d1; tmp /= out_d1;
    int o0 = tmp % out_d0;

    int in_idx = o0 * in_s0 + o1 * in_s1 + o2 * in_s2 + o3 * in_s3 + o4 * in_s4;
    output[idx] = input[in_idx];
}

extern "C" __global__ void tile_nd_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int out_d0,
    int out_d1,
    int out_d2,
    int out_d3,
    int out_d4,
    int in_d0,
    int in_d1,
    int in_d2,
    int in_d3,
    int in_d4,
    int in_s0,
    int in_s1,
    int in_s2,
    int in_s3,
    int in_s4,
    int total
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;

    int tmp = idx;
    int o4 = tmp % out_d4; tmp /= out_d4;
    int o3 = tmp % out_d3; tmp /= out_d3;
    int o2 = tmp % out_d2; tmp /= out_d2;
    int o1 = tmp % out_d1; tmp /= out_d1;
    int o0 = tmp % out_d0;

    int in_idx = (o0 % in_d0) * in_s0
               + (o1 % in_d1) * in_s1
               + (o2 % in_d2) * in_s2
               + (o3 % in_d3) * in_s3
               + (o4 % in_d4) * in_s4;
    output[idx] = input[in_idx];
}
