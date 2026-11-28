// PF247 V-face — recovered 0x48 V2FuseOffset producer at the pinned 280x280
// boundary (cuda_vface.rs fuse_offset_cpu). Left-positive/left-negative atlas
// rows share PF247; the host selects the atlas rectangle and supplies its
// packed RGBA8 pixels. The output is the encoded 280x280 float4 offset field:
// [4*d0+, 4*d0-, 4*d1+, 4*d1-], where
// d0 = (base.r-base.g)*0.25 + (atlas.r-atlas.g)*0.25*degree
// d1 = (base.b-base.a)*0.25 + (atlas.b-atlas.a)*0.25*degree.
// This shader does not claim the full V2Warp final geometry path.
const OFFSET_EDGE: u32 = 280u;

struct Params {
    width: u32,             // 280, fixed by the recovered producer
    height: u32,            // 280
    roi_x: u32,
    roi_y: u32,
    roi_width: u32,
    roi_height: u32,
    atlas_width: u32,
    atlas_height: u32,
    degree: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> base: array<u32>;    // 280x280 RGBA8 packed
@group(0) @binding(2) var<storage, read> atlas: array<u32>;   // RGBA8 packed, host-provided
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>; // 280x280 float4

fn sample_rgba8(w: u32, h: u32, data: ptr<storage, array<u32>, read>, u: f32, v: f32) -> vec4<f32> {
    let fx = clamp(floor(u * f32(w)), 0.0, f32(w - 1u));
    let fy = clamp(floor(v * f32(h)), 0.0, f32(h - 1u));
    let ix = u32(fx);
    let iy = u32(fy);
    let packed = data[iy * w + ix];
    return vec4<f32>(f32(packed & 255u), f32((packed >> 8u) & 255u),
                     f32((packed >> 16u) & 255u), f32((packed >> 24u) & 255u)) * (1.0 / 255.0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    // Outside the atlas ROI the recovered producer leaves the field at zero.
    if (gid.x < params.roi_x || gid.x >= params.roi_x + params.roi_width ||
        gid.y < params.roi_y || gid.y >= params.roi_y + params.roi_height) {
        out[gid.y * params.width + gid.x] = vec4<f32>(0.0);
        return;
    }
    let source_uv = vec2<f32>((f32(gid.x) + 0.5) / f32(OFFSET_EDGE),
                              (f32(gid.y) + 0.5) / f32(OFFSET_EDGE));
    let atlas_uv = vec2<f32>(
        f32(gid.x - params.roi_x) / f32(params.roi_width) + 0.5 / f32(params.roi_width),
        f32(gid.y - params.roi_y) / f32(params.roi_height) + 0.5 / f32(params.roi_height));
    let src = sample_rgba8(params.width, params.height, &base, source_uv.x, source_uv.y);
    let roi = sample_rgba8(params.atlas_width, params.atlas_height, &atlas, atlas_uv.x, atlas_uv.y);
    let d0 = (src.x - src.y) * 0.25 + (roi.x - roi.y) * 0.25 * params.degree;
    let d1 = (src.z - src.w) * 0.25 + (roi.z - roi.w) * 0.25 * params.degree;
    out[gid.y * params.width + gid.x] = vec4<f32>(
        max(d0, 0.0) * 4.0, max(-d0, 0.0) * 4.0,
        max(d1, 0.0) * 4.0, max(-d1, 0.0) * 4.0);
}
