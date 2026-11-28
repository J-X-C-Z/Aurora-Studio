// PF0 spotless — recovered SpotlessProcessNormal fragment.
// Source/blur are f32 RGBA/RGB normalized [0,1]; face/skin masks are f32.
// Quantization and BGRA swap, if any, belong to the host.
struct Params {
    width: u32,
    height: u32,
    dist_factor: f32,
    dist_threshold: f32,
    dark_diff_min_threshold: f32,
    dark_diff_max_threshold: f32,
    white_diff_factor: f32,
    degree: f32,
    flags: u32,          // bit0: explicit skin mask present
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;       // RGBA, stride width*4
@group(0) @binding(2) var<storage, read> blur: array<f32>;      // RGB,  stride width*3
@group(0) @binding(3) var<storage, read> face_mask: array<f32>; // one f32/pixel
@group(0) @binding(4) var<storage, read> skin_mask: array<f32>; // one f32/pixel (or zeros)
@group(0) @binding(5) var<storage, read_write> out: array<f32>; // RGBA

fn sp_step(edge: f32, x: f32) -> f32 {
    if (x < edge) { return 0.0; }
    return 1.0;
}

fn sp_mix(x: f32, y: f32, a: f32) -> f32 {
    // GLSL mix: x*(1-a) + y*a; multiplication before addition.
    return x * (1.0 - a) + y * a;
}

fn sp_length3(vx: f32, vy: f32, vz: f32) -> f32 {
    return sqrt(vx * vx + vy * vy + vz * vz);
}

fn sp_rgb2hsv_hue_v(r: f32, g: f32, b: f32) -> vec2<f32> {
    // SpotlessProcessNormal.frag rgb2hsv (saturation forced to zero).
    let s = sp_step(b, g);
    let px = sp_mix(b, g, s);
    let py = sp_mix(g, b, s);
    let pz = sp_mix(-1.0, 0.0, s);
    let pw = sp_mix(2.0 / 3.0, -1.0 / 3.0, s);
    let t = sp_step(px, r);
    let qx = sp_mix(px, r, t);
    let qy = py;
    let qz = sp_mix(pw, pz, t);
    let qw = sp_mix(r, px, t);
    let d = qx - min(qw, qy);
    let hue = abs(qz + (qw - qy) / (6.0 * d + 1.0e-10));
    return vec2<f32>(hue, qx); // (hue, value)
}

fn sp_skin_value(hue: f32, value: f32) -> f32 {
    var skin = 1.0;
    if ((0.18 <= hue && hue <= 0.89) || value <= 0.2) {
        skin = 0.0;
    } else if (0.16 < hue && hue < 0.18) {
        skin = min(skin, (0.18 - hue) * 50.0);
    } else if (0.89 < hue && hue < 0.91) {
        skin = min(skin, 1.0 - (0.91 - hue) * 50.0);
    } else if (0.2 < value && hue < 0.3) {
        skin = min(skin, 1.0 - (0.3 - value) * 10.0);
    }
    return skin;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let pixel = gid.y * params.width + gid.x;
    let sidx = pixel * 4u;
    let bidx = pixel * 3u;
    let sr = src[sidx];
    let sg = src[sidx + 1u];
    let sb = src[sidx + 2u];
    let sa = src[sidx + 3u];
    var rr = sr;
    var rg = sg;
    var rb = sb;
    let fm = face_mask[pixel];
    if (fm > 0.0) {
        var skin = skin_mask[pixel];
        if ((params.flags & 1u) == 0u) {
            let hv = sp_rgb2hsv_hue_v(sr, sg, sb);
            skin = sp_skin_value(hv.x, hv.y);
        }
        let factor = clamp(fm * skin * params.degree, 0.0, 1.0);
        if (factor > 0.0) {
            let br = blur[bidx];
            let bg = blur[bidx + 1u];
            let bb = blur[bidx + 2u];
            let dark_x = max(br - sr, 0.0);
            let dark_y = max(bg - sg, 0.0);
            let dark_z = max(bb - sb, 0.0);
            let dist = sp_length3(dark_x, dark_y, dark_z) * params.dist_factor;
            if (dist > params.dist_threshold) {
                let dr = clamp(br - sr, params.dark_diff_min_threshold, params.dark_diff_max_threshold);
                let dg = clamp(bg - sg, params.dark_diff_min_threshold, params.dark_diff_max_threshold);
                let db = clamp(bb - sb, params.dark_diff_min_threshold, params.dark_diff_max_threshold);
                rr = sp_mix(sr, min(sr + dr, 1.0), factor);
                rg = sp_mix(sg, min(sg + dg, 1.0), factor);
                rb = sp_mix(sb, min(sb + db, 1.0), factor);
            } else {
                let light_x = max(sr - br, 0.0);
                let light_y = max(sg - bg, 0.0);
                let light_z = max(sb - bb, 0.0);
                let inv_dist = sp_length3(light_x, light_y, light_z) * params.dist_factor;
                if (inv_dist > params.dist_threshold) {
                    rr = sp_mix(sr, br + (sr - br) * params.white_diff_factor, factor);
                    rg = sp_mix(sg, bg + (sg - bg) * params.white_diff_factor, factor);
                    rb = sp_mix(sb, bb + (sb - bb) * params.white_diff_factor, factor);
                }
            }
        }
    }
    out[sidx] = rr;
    out[sidx + 1u] = rg;
    out[sidx + 2u] = rb;
    out[sidx + 3u] = sa;
}
