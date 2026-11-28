// PF232 skull top / skullEnhance — recovered 15-point radial warp
// (kernels/face_detail.rs skull_top producer + portrait_face_stage.rs
// render_skull pixel stage). The host resolves the 15 source/warped contour
// pairs from the UI and landmark packet; this shader consumes them directly.
const CONTOUR_POINTS: u32 = 15u;

struct Params {
    width: u32,
    height: u32,
    roi_x: u32,
    roi_y: u32,
    roi_width: u32,
    roi_height: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<u32>;        // RGBA8 packed
@group(0) @binding(2) var<storage, read> anchor: vec2<f32>;
@group(0) @binding(3) var<storage, read> contour_source: array<vec2<f32>>; // 15 points
@group(0) @binding(4) var<storage, read> contour_warped: array<vec2<f32>>; // 15 points
@group(0) @binding(5) var<storage, read_write> out: array<u32>;

fn sample_bilinear(x: f32, y: f32) -> u32 {
    let fx = clamp(x, 0.0, f32(params.width - 1u));
    let fy = clamp(y, 0.0, f32(params.height - 1u));
    let x0 = u32(floor(fx));
    let y0 = u32(floor(fy));
    let x1 = min(x0 + 1u, params.width - 1u);
    let y1 = min(y0 + 1u, params.height - 1u);
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    let a = source[y0 * params.width + x0];
    let b = source[y0 * params.width + x1];
    let c = source[y1 * params.width + x0];
    let d = source[y1 * params.width + x1];
    var rgb = vec3<f32>(0.0);
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        let shift = ch * 8u;
        let av = f32((a >> shift) & 255u);
        let bv = f32((b >> shift) & 255u);
        let cv = f32((c >> shift) & 255u);
        let dv = f32((d >> shift) & 255u);
        let top = av * (1.0 - tx) + bv * tx;
        let bottom = cv * (1.0 - tx) + dv * tx;
        let value = clamp(u32(round(top * (1.0 - ty) + bottom * ty)), 0u, 255u);
        rgb[ch] = f32(value);
    }
    // Alpha is copied unfiltered from the top-left tap.
    let alpha = (a >> 24u) & 255u;
    return (u32(rgb.x) & 255u) | ((u32(rgb.y) & 255u) << 8u) |
           ((u32(rgb.z) & 255u) << 16u) | (alpha << 24u);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let pixel = gid.y * params.width + gid.x;
    var result = source[pixel];
    if (gid.x < params.roi_x || gid.x >= params.roi_x + params.roi_width ||
        gid.y < params.roi_y || gid.y >= params.roi_y + params.roi_height) {
        out[pixel] = result;
        return;
    }
    let px = f32(gid.x) + 0.5;
    let py = f32(gid.y) + 0.5;
    let dx = px - anchor.x;
    let dy = py - anchor.y;
    let radius = sqrt(dx * dx + dy * dy);
    if (radius < 1.0e-6) {
        out[pixel] = result;
        return;
    }
    let dir = vec2<f32>(dx / radius, dy / radius);
    var best = 0u;
    var best_dot = -3.4028235e38;
    for (var i = 0u; i < CONTOUR_POINTS; i = i + 1u) {
        let original = contour_source[i];
        let warped = contour_warped[i];
        let vx = (warped.x + original.x) * 0.5 - anchor.x;
        let vy = (warped.y + original.y) * 0.5 - anchor.y;
        let norm = sqrt(vx * vx + vy * vy);
        var dot = -3.4028235e38;
        if (norm > 1.0e-6) {
            dot = dir.x * vx / norm + dir.y * vy / norm;
        }
        if (dot > best_dot) {
            best_dot = dot;
            best = i;
        }
    }
    let warped_point = contour_warped[best];
    let source_point = contour_source[best];
    let warped_radius = max((warped_point.x - anchor.x) * dir.x + (warped_point.y - anchor.y) * dir.y, 1.0e-4);
    let source_radius = max((source_point.x - anchor.x) * dir.x + (source_point.y - anchor.y) * dir.y, 1.0e-4);
    let mapped_radius = radius * source_radius / warped_radius;
    result = sample_bilinear(anchor.x + dir.x * mapped_radius, anchor.y + dir.y * mapped_radius);
    out[pixel] = result;
}
