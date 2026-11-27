// PF30009 hair colour unify (kernels/hair_body.rs hair_color_unify).
// engine=(ui+100)/200, signed=1-2*engine, amount=|signed|, gate 0.001.
// q=hair_mask*amount; out.rgb=round_away(src + q*(effect-src)); alpha preserved.
struct Params {
    width: u32,
    height: u32,
    amount: f32,       // host-resolved |1 - 2*(ui+100)/200|
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;          // RGBA8 packed
@group(0) @binding(2) var<storage, read> effect_rgb: array<u32>;   // 3 bytes/pixel packed RGB
@group(0) @binding(3) var<storage, read> hair_mask: array<f32>;    // one f32/pixel [0,1]
@group(0) @binding(4) var<storage, read_write> out: array<u32>;

fn pack_rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

fn round_away_byte(value: f32) -> u32 {
    // MathF.Round(value, MidpointRounding.AwayFromZero) on 0..255.
    let rounded = round(value); // WGSL round = round-to-nearest, ties away from zero
    return clamp(u32(rounded), 0u, 255u);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let pixel = gid.y * params.width + gid.x;
    let packed = src[pixel];
    let q = hair_mask[pixel] * params.amount;
    let eff = effect_rgb[pixel];
    let sr = f32(packed & 255u);
    let sg = f32((packed >> 8u) & 255u);
    let sb = f32((packed >> 16u) & 255u);
    let er = f32(eff & 255u);
    let eg = f32((eff >> 8u) & 255u);
    let eb = f32((eff >> 16u) & 255u);
    let r = round_away_byte(sr + q * (er - sr));
    let g = round_away_byte(sg + q * (eg - sg));
    let b = round_away_byte(sb + q * (eb - sb));
    out[pixel] = pack_rgba(r, g, b, (packed >> 24u) & 255u);
}
