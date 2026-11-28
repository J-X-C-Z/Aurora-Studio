// PF100004 redness / red-vein removal (kernels/face_detail.rs red_vein).
// scale = clamp(strength,0,1); ui<=0 -> scale=0.
// alpha = model_alpha/255 * scale; out.rgb = round_ties_even(src*(1-a)+eff*a)
// Quantization is round-half-to-even on the [0,1] normalized value * 255.
struct Params {
    width: u32,
    height: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;          // RGBA8 packed
@group(0) @binding(2) var<storage, read> effect_rgb: array<u32>;   // 3 bytes/pixel packed RGB
@group(0) @binding(3) var<storage, read> alpha_plane: array<u32>;  // one byte/pixel
@group(0) @binding(4) var<storage, read_write> out: array<u32>;

fn pack_rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

fn round_even_byte(value: f32) -> u32 {
    let scaled = clamp(value, 0.0, 1.0) * 255.0;
    let floor_v = floor(scaled);
    let frac = scaled - floor_v;
    var rounded = floor_v;
    if (frac > 0.5) {
        rounded = floor_v + 1.0;
    } else if (frac == 0.5) {
        if ((u32(floor_v) & 1u) != 0u) { rounded = floor_v + 1.0; }
    }
    return clamp(u32(rounded), 0u, 255u);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let pixel = gid.y * params.width + gid.x;
    let packed = src[pixel];
    let alpha = f32(alpha_plane[pixel] & 255u) * (1.0 / 255.0) * params.scale;
    if (alpha <= 0.0) {
        out[pixel] = packed;
        return;
    }
    let eff = effect_rgb[pixel];
    let sr = f32(packed & 255u) * (1.0 / 255.0);
    let sg = f32((packed >> 8u) & 255u) * (1.0 / 255.0);
    let sb = f32((packed >> 16u) & 255u) * (1.0 / 255.0);
    let er = f32(eff & 255u) * (1.0 / 255.0);
    let eg = f32((eff >> 8u) & 255u) * (1.0 / 255.0);
    let eb = f32((eff >> 16u) & 255u) * (1.0 / 255.0);
    let r = round_even_byte(sr * (1.0 - alpha) + er * alpha);
    let g = round_even_byte(sg * (1.0 - alpha) + eg * alpha);
    let b = round_even_byte(sb * (1.0 - alpha) + eb * alpha);
    out[pixel] = pack_rgba(r, g, b, (packed >> 24u) & 255u);
}
