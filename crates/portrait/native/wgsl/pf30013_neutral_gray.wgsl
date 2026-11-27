// PF30013 neutral gray skin flattening (cuda_neutral_gray.cu skin_grey_kernel).
// weight = clamp(mask/255 * alpha, 0, 1); skip if weight <= 0.001.
// HSL(original.H, original.S, current.L) -> projected RGB, then
// out = round(current + (projected - current)*weight), alpha preserved.
// Binary32 add/mul order is kept unfused; host treats ROI as a crop.
struct Params {
    width: u32,
    height: u32,
    alpha: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> original: array<u32>; // RGBA8 packed
@group(0) @binding(2) var<storage, read> current: array<u32>;  // RGBA8 packed
@group(0) @binding(3) var<storage, read> mask: array<u32>;     // one byte/pixel
@group(0) @binding(4) var<storage, read_write> out: array<u32>;

const HSL_EPSILON: f32 = 0.001;
const INV255: f32 = 0.0039215686274509803;

fn pack_rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

struct Hsl { h: f32, s: f32, l: f32 };

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> Hsl {
    let base_max = select(g, b, g < b);
    let maximum = max(r, base_max);
    let minimum = min(r, min(g, b));
    let chroma = maximum - minimum;
    let w = min(r, base_max);
    let offset = select(
        select(-1.0, 0.0, !(g < b)),
        select(2.0 / 3.0, -1.0 / 3.0, g < b),
        r < base_max);
    let h = abs((w - minimum) / (6.0 * chroma + HSL_EPSILON) + offset);
    let l = maximum - chroma * 0.5;
    let s = chroma / (1.0 - abs(l * 2.0 - 1.0) + HSL_EPSILON);
    return Hsl(h, s, l);
}

fn hsl_axis(value: f32, chroma: f32, l: f32) -> f32 {
    return clamp((value - 0.5) * chroma + l, 0.0, 1.0);
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    let x = h * 6.0;
    let chroma = (1.0 - abs(l * 2.0 - 1.0)) * s;
    return vec3<f32>(
        hsl_axis(clamp(abs(x - 3.0) - 1.0, 0.0, 1.0), chroma, l),
        hsl_axis(clamp(2.0 - abs(x - 2.0), 0.0, 1.0), chroma, l),
        hsl_axis(clamp(2.0 - abs(x - 4.0), 0.0, 1.0), chroma, l));
}

fn round_byte(value: f32) -> u32 {
    return clamp(u32(round(clamp(value, 0.0, 1.0) * 255.0)), 0u, 255u);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let pixel = gid.y * params.width + gid.x;
    let cur = current[pixel];
    let weight = clamp(f32(mask[pixel] & 255u) * INV255 * params.alpha, 0.0, 1.0);
    if (weight <= HSL_EPSILON) {
        out[pixel] = cur;
        return;
    }
    let orig = original[pixel];
    let os = rgb_to_hsl(f32(orig & 255u) * INV255, f32((orig >> 8u) & 255u) * INV255,
                        f32((orig >> 16u) & 255u) * INV255);
    let cs = rgb_to_hsl(f32(cur & 255u) * INV255, f32((cur >> 8u) & 255u) * INV255,
                        f32((cur >> 16u) & 255u) * INV255);
    let projected = hsl_to_rgb(os.h, os.s, cs.l);
    let now = vec3<f32>(f32(cur & 255u), f32((cur >> 8u) & 255u), f32((cur >> 16u) & 255u)) * INV255;
    let mixed = now + (projected - now) * weight;
    out[pixel] = pack_rgba(round_byte(mixed.x), round_byte(mixed.y), round_byte(mixed.z),
                           (cur >> 24u) & 255u);
}
