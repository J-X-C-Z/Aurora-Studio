// PF90086 body skin-tone unify (kernels/makeup.rs body_tone_unify).
// value = src + 255*(w_m*m*(src/255) + w_o*o), clamp to byte via f32->u8
// truncation. Alpha is preserved. Host supplies M/O planes as f32 triples.
struct Params {
    width: u32,
    height: u32,
    w_m: f32,
    w_o: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;          // RGBA8 packed
@group(0) @binding(2) var<storage, read> m_plane: array<f32>;      // 3 floats/pixel
@group(0) @binding(3) var<storage, read> o_plane: array<f32>;      // 3 floats/pixel
@group(0) @binding(4) var<storage, read_write> out: array<u32>;    // RGBA8 packed

fn pack_rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

fn trunc_byte(value: f32) -> u32 {
    return clamp(u32(value), 0u, 255u); // negative f32 -> u32 is 0 in WGSL saturation
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let pixel = gid.y * params.width + gid.x;
    let packed = src[pixel];
    let base = pixel * 3u;
    var rgb = vec3<f32>(
        f32(packed & 255u),
        f32((packed >> 8u) & 255u),
        f32((packed >> 16u) & 255u));
    let m = vec3<f32>(m_plane[base], m_plane[base + 1u], m_plane[base + 2u]);
    let o = vec3<f32>(o_plane[base], o_plane[base + 1u], o_plane[base + 2u]);
    let value = rgb + 255.0 * (params.w_m * m * (rgb * (1.0 / 255.0)) + params.w_o * o);
    let r = trunc_byte(value.x);
    let g = trunc_byte(value.y);
    let b = trunc_byte(value.z);
    out[pixel] = pack_rgba(r, g, b, (packed >> 24u) & 255u);
}
