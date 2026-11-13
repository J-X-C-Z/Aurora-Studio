#include <metal_stdlib>
using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

// Metal candidate for the proven NVIDIA/Rust SkinAdjust LUT law.
// Source is RGBA8, atlas is packed 512x512x3 RGB8, store is 12-bit floor.
// This is a candidate file only; no production host path is wired yet.

struct SkinLutParams
{
    uint width;
    uint height;
    uint stride;
    uint mask_row_bytes;
    float mix;
    float gate;
    uint lut_width;
    uint lut_height;
};
static_assert(sizeof(SkinLutParams) == 32, "SkinLutParams must stay 32 bytes");

constant int SL_EDGE = 512;

static inline int sl_clampi(int value, int low, int high)
{
    return min(max(value, low), high);
}

static inline int sl_lut_byte(device const uchar* lut, uint x, uint y, uint channel)
{
    return int(lut[(y * 512u + x) * 3u + channel]);
}

static inline float sl_tap(
    device const uchar* lut,
    uint channel,
    float px,
    float py)
{
    float u = px * 512.0f - 0.5f;
    float v = py * 512.0f - 0.5f;
    float i0f = floor(u);
    float j0f = floor(v);
    int i0 = int(i0f);
    int j0 = int(j0f);
    int kx = int(floor((u - i0f) * 256.0f + 0.5f));
    int ky = int(floor((v - j0f) * 256.0f + 0.5f));
    int t = (kx * ky + 128) >> 8;
    int w00 = 256 - kx - ky + t;
    int w10 = kx - t;
    int w01 = ky - t;
    int w11 = t;
    uint x0 = uint(sl_clampi(i0, 0, 511));
    uint x1 = uint(sl_clampi(i0 + 1, 0, 511));
    uint y0 = uint(sl_clampi(j0, 0, 511));
    uint y1 = uint(sl_clampi(j0 + 1, 0, 511));
    int q = sl_lut_byte(lut, x0, y0, channel) * w00
          + sl_lut_byte(lut, x1, y0, channel) * w10
          + sl_lut_byte(lut, x0, y1, channel) * w01
          + sl_lut_byte(lut, x1, y1, channel) * w11;
    q = q * 257;
    return float((q + 128) >> 8) / 65535.0f;
}

// Blue selects the 64x64 tile; red/green select the position inside it.
static inline float sl_lut(
    device const uchar* lut,
    float r,
    float g,
    float b,
    uint channel)
{
    float blue = b * 63.0f;
    float bf = floor(blue);
    float q1y = floor(bf * 0.125f);
    float q1x = bf - q1y * 8.0f;
    float bc = ceil(blue);
    float q2y = floor(bc * 0.125f);
    float q2x = bc - q2y * 8.0f;
    const float k = 0.0009765625f;
    float span = as_type<float>(0x3dfc0001u);
    volatile float span_r = span * r;
    volatile float span_g = span * g;
    float x1 = span_r + (q1x * 0.125f + k);
    float y1 = span_g + (q1y * 0.125f + k);
    float x2 = (q2x * 0.125f + span_r) + k;
    float y2 = (q2y * 0.125f + span_g) + k;
    float low = sl_tap(lut, channel, x1, y1);
    float high = sl_tap(lut, channel, x2, y2);
    volatile float blue_delta = (blue - bf) * (high - low);
    return low + blue_delta;
}

static inline uchar sl_quantize(float value)
{
    int value12 = int(floor(value * 4096.0f));
    int q = (value12 * 255 + 2047) >> 12;
    return uchar(sl_clampi(q, 0, 255));
}

kernel void skin_lut_fixed(
    device const uchar* source [[buffer(0)]],
    device const uchar* mask [[buffer(1)]],
    device const uchar* lut [[buffer(2)]],
    device uchar* out [[buffer(3)]],
    constant SkinLutParams& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height)
        return;
    uint source_index = gid.y * p.stride + gid.x * 4u;
    uint mask_index = gid.y * p.mask_row_bytes + gid.x;
    uchar source_r = source[source_index + 0u];
    uchar source_g = source[source_index + 1u];
    uchar source_b = source[source_index + 2u];
    uchar source_a = source[source_index + 3u];
    out[source_index + 0u] = source_r;
    out[source_index + 1u] = source_g;
    out[source_index + 2u] = source_b;
    out[source_index + 3u] = source_a;

    float weight = (float(mask[mask_index]) / 255.0f) * p.mix;
    if (!(weight > p.gate))
        return;

    float r = float(source_r) / 255.0f;
    float g = float(source_g) / 255.0f;
    float b = float(source_b) / 255.0f;
    float lut_r = sl_lut(lut, r, g, b, 0u);
    float lut_g = sl_lut(lut, r, g, b, 1u);
    float lut_b = sl_lut(lut, r, g, b, 2u);
    volatile float delta_r = weight * (lut_r - r);
    volatile float delta_g = weight * (lut_g - g);
    volatile float delta_b = weight * (lut_b - b);
    float blended_r = r + delta_r;
    float blended_g = g + delta_g;
    float blended_b = b + delta_b;

    out[source_index + 0u] = sl_quantize(blended_r);
    out[source_index + 1u] = sl_quantize(blended_g);
    out[source_index + 2u] = sl_quantize(blended_b);
    out[source_index + 3u] = source_a;
}
