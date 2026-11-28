// PF90085 face skin-tone unify — reduced recovered pixel tail
// (kernels/tone_unify.rs pf90085_blend). The host resolves the producer degree
// (piecewise metric remap or ui/100) and binds a constant target; the optional
// support-weighted fallback target is computed by the host pass.
struct Params {
    width: u32,
    height: u32,
    degree: f32,
    flags: u32,          // bit0 skin mask, bit1 parsing ABC, bit2 protect,
                         // bit3 model weight, bit4 per-pixel model target,
                         // bit5 constant target (else fallback target buffer)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    target_r: f32,
    target_g: f32,
    target_b: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;            // RGBA8 packed
@group(0) @binding(2) var<storage, read> face_mask_g: array<u32>;
@group(0) @binding(3) var<storage, read> skin_mask: array<u32>;
@group(0) @binding(4) var<storage, read> parsing_a: array<u32>;
@group(0) @binding(5) var<storage, read> parsing_g: array<u32>;
@group(0) @binding(6) var<storage, read> parsing_b: array<u32>;
@group(0) @binding(7) var<storage, read> protect: array<u32>;
@group(0) @binding(8) var<storage, read> model_weight: array<u32>;
@group(0) @binding(9) var<storage, read> model_target: array<u32>;    // 3 bytes/pixel packed RGB
@group(0) @binding(10) var<storage, read> fallback_target: array<f32>; // 3 f32 [0,1]
@group(0) @binding(11) var<storage, read_write> out: array<u32>;     // RGBA8 packed

const FACE_PARSING_WEIGHT: f32 = 0.39;
const MASK_GATE: f32 = 0.001;

fn unpack_byte(v: u32) -> f32 { return f32(v & 255u) * (1.0 / 255.0); }

fn quantise_even(value: f32) -> u32 {
    // round_ties_even(clamp(value,0,1)*255)
    let scaled = clamp(value, 0.0, 1.0) * 255.0;
    let floor_v = floor(scaled);
    let frac = scaled - floor_v;
    var rounded = floor_v;
    if (frac > 0.5) {
        rounded = floor_v + 1.0;
    } else if (frac == 0.5) {
        let even = (u32(floor_v) & 1u) == 0u;
        if (!even) { rounded = floor_v + 1.0; }
    }
    return clamp(u32(rounded), 0u, 255u);
}

fn rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let pixel = gid.y * params.width + gid.x;
    let src_packed = src[pixel];
    if (params.degree <= MASK_GATE) {
        out[pixel] = src_packed;
        return;
    }
    let face = unpack_byte(face_mask_g[pixel]);
    let skin = select(0.0, unpack_byte(skin_mask[pixel]), (params.flags & 1u) != 0u);
    var parsing = face * FACE_PARSING_WEIGHT;
    if ((params.flags & 2u) != 0u) {
        parsing = (unpack_byte(parsing_a[pixel]) + unpack_byte(parsing_g[pixel]) +
                   unpack_byte(parsing_b[pixel]) + face) * FACE_PARSING_WEIGHT;
    }
    var support = max(skin, parsing);
    if ((params.flags & 4u) != 0u) {
        support = support * (1.0 - unpack_byte(protect[pixel]));
    }
    if (support <= MASK_GATE) {
        out[pixel] = src_packed;
        return;
    }
    var mix_amount = support * params.degree;
    if ((params.flags & 8u) != 0u) {
        mix_amount = mix_amount * unpack_byte(model_weight[pixel]);
    }
    if (mix_amount <= 0.0) {
        out[pixel] = src_packed;
        return;
    }
    let alpha = clamp(mix_amount, 0.0, 1.0);
    var target = vec3<f32>(params.target_r, params.target_g, params.target_b);
    if ((params.flags & 16u) != 0u) {
        let mp = model_target[pixel];
        target = vec3<f32>(f32(mp & 255u), f32((mp >> 8u) & 255u), f32((mp >> 16u) & 255u)) * (1.0 / 255.0);
    } else if ((params.flags & 32u) == 0u) {
        target = vec3<f32>(fallback_target[pixel * 3u], fallback_target[pixel * 3u + 1u], fallback_target[pixel * 3u + 2u]);
    }
    let sr = unpack_byte(src_packed);
    let sg = unpack_byte(src_packed >> 8u);
    let sb = unpack_byte(src_packed >> 16u);
    let r = quantise_even(sr * (1.0 - alpha) + target.x * alpha);
    let g = quantise_even(sg * (1.0 - alpha) + target.y * alpha);
    let b = quantise_even(sb * (1.0 - alpha) + target.z * alpha);
    out[pixel] = rgba(r, g, b, (src_packed >> 24u) & 255u);
}
