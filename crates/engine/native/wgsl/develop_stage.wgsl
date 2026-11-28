// Aurora Studio — cross-platform develop stage (WGSL).
//
// Twin of `develop_cpu_ref` in `crates/engine/src/gpu_crossplatform.rs`.
// One pass over the scene-linear frame does the whole point chain:
//
//   white balance gains -> exposure -> highlight/shadow tone toe
//   -> contrast -> saturation -> linear-to-sRGB -> RGBA8
//
// The toning terms are the ones the CPU renderer already uses, so a frame
// developed here and a frame developed by `processing::tone` agree on shape:
//   * highlight/shadow use the kernel-exact `CImgOpLocalToneMappingV8` toe
//     (`processing::tone::hdr_toe`, kernel `z2714f67010`), applied as one
//     luma-derived gain so the stage stays channel-neutral.
//   * contrast is `processing::color::contrast`, fed the already-scaled
//     `contrast / 50 * 0.8` amount.
//   * saturation is the BT.709 luma-preserving form.
//
// Alpha is not read by any colour term: `src.w` is quantised and stored
// unchanged. Inputs must be finite; the caller validates its parameters.

struct DevelopParams {
    wb: vec4<f32>,       // rgb white-balance gains, .w unused
    exposure: f32,       // linear multiplier, i.e. 2^EV
    highlights: f32,     // -1..1, the slider value divided by 100
    shadows: f32,        // -1..1
    contrast: f32,       // -1..1, already scaled by /50*0.8
    saturation: f32,     // 1.0 = neutral
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> develop_params: DevelopParams;
@group(0) @binding(1) var<storage, read> develop_src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> develop_out: array<u32>;

const HDR_TOE_K = 1.0;

fn luma709(c: vec3<f32>) -> f32 {
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

fn linear_to_srgb(v: f32) -> f32 {
    if (v <= 0.0031308) {
        return v * 12.92;
    }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

fn contrast_curve(g: f32, amount: f32) -> f32 {
    if (abs(amount) < 1e-4) {
        return g;
    }
    let k = 1.0 + amount;
    if (k >= 1.0) {
        let x = (g - 0.5) * k;
        return 0.5 + x / (1.0 + abs(x) * (k - 1.0) * 0.8);
    }
    return 0.5 + (g - 0.5) * k;
}

// `CImgOpLocalToneMappingV8`'s hard toe, kernel `z2714f67010`.
fn hdr_toe(x: f32, k: f32) -> f32 {
    let u = x * (2.0 * 0.9);
    let v = abs(u) + 1.0;
    var r = (u * v) / (v + u * u) * 0.5 + 0.5;
    r = max(r, u - r + 1.0) * (1.0 / (2.0 * k));
    let s = r * r;
    let num = (1.0 + r + s) * (r + s * s);
    return num / (1.0 + num) * k;
}

// One gain per pixel from luma, so the panel is channel-neutral.
fn hdr_gain(c: vec3<f32>) -> f32 {
    if (develop_params.highlights == 0.0 && develop_params.shadows == 0.0) {
        return 1.0;
    }
    let x = luma709(c);
    var b = 1.0;
    if (develop_params.highlights != 0.0) {
        let t = hdr_toe(x, HDR_TOE_K);
        var exponent = t;
        if (develop_params.highlights < 0.0) {
            exponent = -t;
        }
        let m = abs(develop_params.highlights);
        b = exp2(exponent) * m + (1.0 - m);
    }
    var a = 1.0;
    if (develop_params.shadows != 0.0) {
        let t = hdr_toe(-x, HDR_TOE_K);
        var exponent = t;
        if (develop_params.shadows < 0.0) {
            exponent = -t;
        }
        let m = abs(develop_params.shadows);
        a = exp2(exponent) * m + (1.0 - m);
    }
    return a / b;
}

// clamp(v, 0, 1) * 255 + 0.5, truncated — the byte rule the CPU reference and
// the CUDA develop kernel both use.
fn quantize_u8(v: f32) -> u32 {
    return u32(floor(clamp(v, 0.0, 1.0) * 255.0 + 0.5));
}

@compute @workgroup_size(256)
fn develop_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= arrayLength(&develop_out)) {
        return;
    }

    let sample = develop_src[index];
    let exposure = develop_params.exposure;
    var c = vec3<f32>(
        sample.x * develop_params.wb.x,
        sample.y * develop_params.wb.y,
        sample.z * develop_params.wb.z,
    ) * exposure;

    c = c * hdr_gain(c);
    c = vec3<f32>(
        contrast_curve(c.x, develop_params.contrast),
        contrast_curve(c.y, develop_params.contrast),
        contrast_curve(c.z, develop_params.contrast),
    );

    let l = luma709(c);
    let s = develop_params.saturation;
    c = vec3<f32>(l + (c.x - l) * s, l + (c.y - l) * s, l + (c.z - l) * s);

    let r = quantize_u8(linear_to_srgb(c.x));
    let g = quantize_u8(linear_to_srgb(c.y));
    let b = quantize_u8(linear_to_srgb(c.z));
    let a = quantize_u8(sample.w);
    develop_out[index] = r | (g << 8u) | (b << 16u) | (a << 24u);
}
