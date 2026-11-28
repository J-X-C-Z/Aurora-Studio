// Aurora Studio — cross-platform detail (clarity/sharpen) + histogram (WGSL).
//
// Twin of `detail_histogram_cpu_ref` in `crates/engine/src/gpu_crossplatform.rs`
// and of the CPU detail stage in `mock/detail.rs` / the CUDA twins in
// `native/cuda/detail_gpu.cu`.
//
// Three entry points, one shader module:
//   blur_h        — RGBA8 -> luma plane, 5-tap separable gaussian, rows
//   blur_v        — luma plane -> low-pass plane, 5-tap separable gaussian, columns
//   sharpen_hist  — luma - low-pass high-frequency boost -> RGBA8, and the
//                   256-bin R/G/B/luma histogram reduced through workgroup
//                   atomics
//
// The gaussian is [1 4 6 4 1] / 16 with clamp-to-edge borders, exactly the
// weights the CPU twin accumulates, in the same order.
//
// Histogram layout in the 1024-entry u32 buffer:
//   [0, 256)    red bin of the *output* pixel
//   [256, 512)  green
//   [512, 768)  blue
//   [768, 1024) luma, truncated to a bin the way the CUDA kernel does
// The buffer must arrive zeroed; the caller clears it before every dispatch.

struct DetailParams {
    size: vec2<u32>,
    amount: f32,   // sharpen gain; 0 disables the boost (identity)
    _pad: f32,
};

@group(0) @binding(0) var<uniform> detail_params: DetailParams;
@group(0) @binding(1) var<storage, read> detail_src: array<u32>;
@group(0) @binding(2) var<storage, read_write> luma_h: array<f32>;
@group(0) @binding(3) var<storage, read_write> luma_v: array<f32>;
@group(0) @binding(4) var<storage, read_write> detail_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> hist: array<atomic<u32>>;

// 1024 bins x 4 B = 4 KiB, well inside the default 16 KiB workgroup budget.
var<workgroup> bins: array<atomic<u32>, 1024>;

const WORKGROUP_THREADS = 256u;
const BINS = 1024u;

fn clamp_index(i: i32, n: i32) -> u32 {
    return u32(clamp(i, 0, n - 1));
}

fn unpack_r(px: u32) -> u32 { return px & 0xffu; }
fn unpack_g(px: u32) -> u32 { return (px >> 8u) & 0xffu; }
fn unpack_b(px: u32) -> u32 { return (px >> 16u) & 0xffu; }
fn unpack_a(px: u32) -> u32 { return (px >> 24u) & 0xffu; }

fn luma_of(px: u32) -> f32 {
    return 0.2126 * f32(unpack_r(px))
        + 0.7152 * f32(unpack_g(px))
        + 0.0722 * f32(unpack_b(px));
}

@compute @workgroup_size(16, 16)
fn blur_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = i32(detail_params.size.x);
    let h = i32(detail_params.size.y);
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= w || y >= h) {
        return;
    }
    let row = u32(y) * u32(w);
    var acc = 0.0;
    acc = acc + 1.0 * luma_of(detail_src[row + clamp_index(x - 2, w)]);
    acc = acc + 4.0 * luma_of(detail_src[row + clamp_index(x - 1, w)]);
    acc = acc + 6.0 * luma_of(detail_src[row + clamp_index(x, w)]);
    acc = acc + 4.0 * luma_of(detail_src[row + clamp_index(x + 1, w)]);
    acc = acc + 1.0 * luma_of(detail_src[row + clamp_index(x + 2, w)]);
    luma_h[row + u32(x)] = acc * 0.0625;
}

@compute @workgroup_size(16, 16)
fn blur_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = i32(detail_params.size.x);
    let h = i32(detail_params.size.y);
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= w || y >= h) {
        return;
    }
    let column = u32(x);
    let stride = u32(w);
    var acc = 0.0;
    acc = acc + 1.0 * luma_h[u32(clamp_index(y - 2, h)) * stride + column];
    acc = acc + 4.0 * luma_h[u32(clamp_index(y - 1, h)) * stride + column];
    acc = acc + 6.0 * luma_h[u32(clamp_index(y, h)) * stride + column];
    acc = acc + 4.0 * luma_h[u32(clamp_index(y + 1, h)) * stride + column];
    acc = acc + 1.0 * luma_h[u32(clamp_index(y + 2, h)) * stride + column];
    luma_v[u32(y) * stride + column] = acc * 0.0625;
}

fn quantize_u8(v: f32) -> u32 {
    return u32(clamp(round(v), 0.0, 255.0));
}

// Flat 1-D dispatch: index = workgroup_id.x * 256 + local_invocation_index, so
// one workgroup of 16x16 threads covers 256 consecutive pixels.
@compute @workgroup_size(16, 16)
fn sharpen_hist(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    // Every thread must reach both barriers, so the pixel work is guarded
    // rather than returned over.
    for (var i = lid; i < BINS; i = i + WORKGROUP_THREADS) {
        atomicStore(&bins[i], 0u);
    }
    workgroupBarrier();

    let index = wid.x * WORKGROUP_THREADS + lid;
    let count = arrayLength(&detail_out);
    if (index < count) {
        let px = detail_src[index];
        let current = luma_of(px);
        let goal_luma = clamp(current + (current - luma_v[index]) * detail_params.amount, 0.0, 255.0);
        var r = unpack_r(px);
        var g = unpack_g(px);
        var b = unpack_b(px);
        // Same dead band as `mock/detail.rs`: a change under 1/20 of a code
        // value is not written back.
        if (abs(goal_luma - current) >= 0.05) {
            var f = 1.0;
            if (current > 1.0) {
                f = goal_luma / current;
            }
            r = quantize_u8(f32(r) * f);
            g = quantize_u8(f32(g) * f);
            b = quantize_u8(f32(b) * f);
        }
        detail_out[index] = r | (g << 8u) | (b << 16u) | (unpack_a(px) << 24u);

        atomicAdd(&bins[r], 1u);
        atomicAdd(&bins[256u + g], 1u);
        atomicAdd(&bins[512u + b], 1u);
        let luma_bin = u32(clamp(0.2126 * f32(r) + 0.7152 * f32(g) + 0.0722 * f32(b), 0.0, 255.0));
        atomicAdd(&bins[768u + luma_bin], 1u);
    }

    workgroupBarrier();
    for (var j = lid; j < BINS; j = j + WORKGROUP_THREADS) {
        let value = atomicLoad(&bins[j]);
        if (value != 0u) {
            atomicAdd(&hist[j], value);
        }
    }
}
