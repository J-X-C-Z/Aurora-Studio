// PF21105 AI body skin unify (kernels/face_detail.rs body_skin_unify).
// One dispatch = one sequential pass; host chains passes.
// t = clamp(mask/255 * (ui/100), 0, 1)
// out.rgb = trunc(current*(1-t) + effect*t); out.a = current.a
struct Params {
    width: u32,
    height: u32,
    degree01: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> current: array<u32>;  // RGBA8 packed
@group(0) @binding(2) var<storage, read> mask: array<u32>;     // one byte/pixel
@group(0) @binding(3) var<storage, read> effect: array<u32>;   // RGBA8 packed
@group(0) @binding(4) var<storage, read_write> out: array<u32>;

fn pack_rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

fn trunc_byte(value: f32) -> u32 {
    return clamp(u32(value), 0u, 255u);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let pixel = gid.y * params.width + gid.x;
    let cur = current[pixel];
    let t = clamp(f32(mask[pixel] & 255u) * (1.0 / 255.0) * params.degree01, 0.0, 1.0);
    if (t <= 0.0) {
        out[pixel] = cur;
        return;
    }
    let eff = effect[pixel];
    let cr = f32(cur & 255u);
    let cg = f32((cur >> 8u) & 255u);
    let cb = f32((cur >> 16u) & 255u);
    let er = f32(eff & 255u);
    let eg = f32((eff >> 8u) & 255u);
    let eb = f32((eff >> 16u) & 255u);
    let r = trunc_byte(cr * (1.0 - t) + er * t);
    let g = trunc_byte(cg * (1.0 - t) + eg * t);
    let b = trunc_byte(cb * (1.0 - t) + eb * t);
    out[pixel] = pack_rgba(r, g, b, (cur >> 24u) & 255u);
}
