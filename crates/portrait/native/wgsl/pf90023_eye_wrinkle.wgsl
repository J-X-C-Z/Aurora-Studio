// PF90023 eye wrinkle — FacialSmoothProcessV2 composite
// (kernels/facial_smooth.rs composite_unchecked / skin_adjust.rs wrinkle_apply).
// GPU inputs are f32 RGBA normalized [0,1]; the host supplies source/base/
// residual/mask/skin as separate planes. PF90023 native degree = ui/100.
// Binary32 mul-then-add order is kept unfused; WGSL does not contract FMA.
struct Params {
    width: u32,
    height: u32,
    degree: f32,
    gate: f32,
    legal_alpha: f32,
    dark_circle_alpha: f32,
    diff_factor: f32,
    diff_factor2: f32,
    color_threshold_factor: f32,
    mix_alpha: f32,
    diff_min_threshold: f32,
    diff_max_threshold: f32,
    high_freq_min_threshold: f32,
    high_freq_max_threshold: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<vec4<f32>>;   // RGBA f32
@group(0) @binding(2) var<storage, read> base: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> residual: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> mask_rgba: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> skin_mask: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<vec4<f32>>;

fn wr_mix(x: f32, y: f32, a: f32) -> f32 {
    return x * (1.0 - a) + y * a;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let pixel = gid.y * params.width + gid.x;
    let src = source[pixel];
    if (params.degree <= params.gate) {
        out[pixel] = src;
        return;
    }
    let alpha = max(max(params.dark_circle_alpha * mask_rgba[pixel].x,
                        params.legal_alpha * mask_rgba[pixel].z), 0.0) * skin_mask[pixel];
    if (alpha <= 0.0) {
        out[pixel] = src;
        return;
    }
    let b = base[pixel];
    let r = residual[pixel];
    var result = src;
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        let high = src[ch] - b[ch];
        let diff = clamp((r[ch] - b[ch]) * params.diff_factor + r[ch] * params.diff_factor2,
                         params.diff_min_threshold, params.diff_max_threshold);
        let threshold = wr_mix(r[ch] * params.color_threshold_factor, r[ch], params.mix_alpha);
        let upper = max(threshold, src[ch] - high);
        let reference = clamp(src[ch] + diff - high, 0.0, upper) + high;
        let delta = clamp(wr_mix(src[ch], reference, alpha) - src[ch],
                          params.high_freq_min_threshold, params.high_freq_max_threshold);
        result[ch] = src[ch] + delta;
    }
    out[pixel] = vec4<f32>(result.x, result.y, result.z, src.w);
}
