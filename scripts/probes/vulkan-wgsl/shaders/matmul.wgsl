// Small FP32 weighted matmul kernel: C = A * B + bias
//   A: [m, k]  row-major f32
//   B: [k, n]  row-major f32
//   C: [m, n]  row-major f32
//   bias: [n]  f32
// Accumulation order is i = 0 .. k-1, identically to the CPU reference, so the
// expected agreement is exact-modulo-FMA and far inside the 1e-4 contract.

struct F32Buf {
    w: array<f32>,
};

struct U32Buf {
    w: array<u32>,
};

@group(0) @binding(0) var<storage, read>       a    : F32Buf;
@group(0) @binding(1) var<storage, read>       b    : F32Buf;
@group(0) @binding(2) var<storage, read_write> c    : F32Buf;
@group(0) @binding(3) var<storage, read>       dims : U32Buf;  // [m, k, n]
@group(0) @binding(4) var<storage, read>       bias : F32Buf;

@compute @workgroup_size(8, 8, 1)
fn matmul_bias(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = dims.w[0];
    let k = dims.w[1];
    let n = dims.w[2];
    let row = gid.x;
    let col = gid.y;
    if (row >= m || col >= n) { return; }

    var acc: f32 = bias.w[col];
    for (var i: u32 = 0u; i < k; i = i + 1u) {
        acc = acc + a.w[row * k + i] * b.w[i * n + col];
    }
    c.w[row * n + col] = acc;
}
