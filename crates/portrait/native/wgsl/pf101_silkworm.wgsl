// PF101 silkworm — recovered BlendImageLayerWithMask.frag pixel law
// (kernels/pf101_silkworm.rs apply_image_layer -> makeup_material dual_rows).
// The host resolves degree01 = clamp(w0*w1*(adjust_gate?0.7:1), 0, 1) and binds
// mask-UV / material-UV planes. Material mix_color is [0,0,0,255] and
// blend_alpha=false, so factor = clamp(material.a * degree01 * mask.r, 0, 1).
// FilterType 42 (default) = SoftLight; 41 Multiply, 43 Lighten, 44 Screen.
struct Params {
    width: u32,
    height: u32,
    degree01: f32,
    filter_type: u32,     // 41/42/43/44
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;          // RGBA8 packed
@group(0) @binding(2) var<storage, read> material: array<u32>;     // RGBA8 packed, sampled by UV
@group(0) @binding(3) var<storage, read> mask: array<u32>;         // RGBA8 packed, sampled by UV
@group(0) @binding(4) var<storage, read> mask_uv: array<vec2<f32>>;     // one per output pixel
@group(0) @binding(5) var<storage, read> material_uv: array<vec2<f32>>; // one per output pixel
@group(0) @binding(6) var<storage, read_write> out: array<u32>;

const INV255: f32 = 0.0039215686274509803;

fn pack_rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

fn sample_rgba(w: u32, h: u32, data: ptr<storage, array<u32>, read>, u: f32, v: f32) -> vec4<f32> {
    // makeup_material::sample_rgba: x = u*w - 0.5, clamp, then lerp tx then ty.
    let px = clamp(u * f32(w) - 0.5, 0.0, f32(w - 1u));
    let py = clamp(v * f32(h) - 0.5, 0.0, f32(h - 1u));
    let x0 = u32(floor(px));
    let y0 = u32(floor(py));
    let x1 = min(x0 + 1u, w - 1u);
    let y1 = min(y0 + 1u, h - 1u);
    let tx = px - f32(x0);
    let ty = py - f32(y0);
    let a = data[y0 * w + x0];
    let b = data[y0 * w + x1];
    let c = data[y1 * w + x0];
    let d = data[y1 * w + x1];
    var res = vec4<f32>(0.0);
    for (var ch = 0u; ch < 4u; ch = ch + 1u) {
        let shift = ch * 8u;
        let av = f32((a >> shift) & 255u);
        let bv = f32((b >> shift) & 255u);
        let cv = f32((c >> shift) & 255u);
        let dv = f32((d >> shift) & 255u);
        let top = av * (1.0 - tx) + bv * tx;
        let bottom = cv * (1.0 - tx) + dv * tx;
        res[ch] = (top * (1.0 - ty) + bottom * ty) * INV255;
    }
    return res;
}

fn soft_light(base: f32, blend: f32) -> f32 {
    if (blend < 0.5) {
        return 2.0 * base * blend + base * base * (1.0 - 2.0 * blend);
    }
    return sqrt(base) * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend);
}

fn blend_channel(base: f32, blend: f32) -> f32 {
    if (params.filter_type == 41u) { return base * blend; }               // Multiply
    if (params.filter_type == 43u) { return max(base, blend); }           // Lighten
    if (params.filter_type == 44u) { return 1.0 - (1.0 - base) * (1.0 - blend); } // Screen
    return soft_light(base, blend);                                        // SoftLight (42)
}

fn byte_even(value: f32) -> u32 {
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
    // Material/mask planes are sampled by explicit UVs. The host binds them at
    // the same extent as the source for this source-level candidate.
    let mat_sample = sample_rgba(params.width, params.height, &material,
                                 material_uv[pixel].x, material_uv[pixel].y);
    let mask_sample = sample_rgba(params.width, params.height, &mask,
                                  mask_uv[pixel].x, mask_uv[pixel].y);
    let factor = clamp(mat_sample.w * params.degree01 * mask_sample.x, 0.0, 1.0);
    if (factor <= 0.01) {
        out[pixel] = packed;
        return;
    }
    let base = vec3<f32>(f32(packed & 255u), f32((packed >> 8u) & 255u),
                         f32((packed >> 16u) & 255u)) * INV255;
    let blended = vec3<f32>(blend_channel(base.x, mat_sample.x),
                            blend_channel(base.y, mat_sample.y),
                            blend_channel(base.z, mat_sample.z));
    let r = byte_even(base.x * (1.0 - factor) + blended.x * factor);
    let g = byte_even(base.y * (1.0 - factor) + blended.y * factor);
    let b = byte_even(base.z * (1.0 - factor) + blended.z * factor);
    out[pixel] = pack_rgba(r, g, b, (packed >> 24u) & 255u);
}
