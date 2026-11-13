#include <metal_stdlib>
using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

// Fallback for MSL compilers without atomic_fetch_min_explicit /
// atomic_fetch_max_explicit. Same MeshRasterParams ABI (36 B), same raster
// output, no atomics and no scratch buffers.
//
// One thread per destination pixel. Triangles are resolved serially in EBO
// order per pixel, so First/Last/Reject are deterministic without atomics.
// Buffers:
//   0 dest float2
//   1 source_uv float2
//   2 ebo uint16
//   3 source RGBA8
//   4 out RGBA8 (prefilled with source)
//   5 params (36 B)

struct MeshRasterParams
{
    int width;
    int height;
    int vertex_count;
    int triangle_count;
    int wrap;
    int winding;
    int cull;
    int edge;
    int overlap;
};
static_assert(sizeof(MeshRasterParams) == 36, "MeshRasterParams must stay 36 bytes");

constant int MS_WRAP_REPEAT = 1;
constant int MS_WINDING_POSITIVE_IS_FRONT = 1;
constant int MS_CULL_FRONT = 2;
constant int MS_CULL_BACK = 3;
constant int MS_EDGE_INCLUSIVE = 2;
constant int MS_OVERLAP_REJECT = 1;
constant int MS_OVERLAP_LAST = 3;

constant float MS_DEGENERATE_AREA2 = 1e-12f;

struct MSVec2
{
    float x;
    float y;
};

struct MSWeights
{
    float b0;
    float b1;
    float b2;
    bool ok;
};

struct MSAxis
{
    int i0;
    int i1;
    float frac;
};

static inline float ms_add(float a, float b) { return a + b; }
static inline float ms_mul(float a, float b) { return a * b; }

static inline MSVec2 ms_sub(MSVec2 a, MSVec2 b)
{
    return MSVec2{ ms_add(a.x, -b.x), ms_add(a.y, -b.y) };
}

static inline float ms_cross(MSVec2 a, MSVec2 b)
{
    return ms_add(ms_mul(a.x, b.y), -ms_mul(a.y, b.x));
}

static inline MSVec2 ms_read_vec2(device const float* values, ushort index)
{
    uint offset = uint(index) * 2u;
    return MSVec2{ values[offset], values[offset + 1u] };
}

static inline bool ms_is_front(float area2, int winding)
{
    bool positive = area2 > 0.0f;
    return winding == MS_WINDING_POSITIVE_IS_FRONT ? positive : !positive;
}

static inline MSWeights ms_weights(
    MSVec2 p0,
    MSVec2 p1,
    MSVec2 p2,
    float area2,
    float sx,
    float sy,
    int edge)
{
    MSVec2 s = MSVec2{ sx, sy };
    float w0 = ms_cross(ms_sub(p1, s), ms_sub(p2, s)) / area2;
    float w1 = ms_cross(ms_sub(p2, s), ms_sub(p0, s)) / area2;
    float w2 = ms_cross(ms_sub(p0, s), ms_sub(p1, s)) / area2;
    bool inside = edge == MS_EDGE_INCLUSIVE
        ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
        : (w0 > 0.0f && w1 > 0.0f && w2 > 0.0f);
    if (!inside)
        return MSWeights{ 0.0f, 0.0f, 0.0f, false };
    w0 = fmax(w0, 0.0f);
    w1 = fmax(w1, 0.0f);
    w2 = fmax(w2, 0.0f);
    float sum = ms_add(ms_add(w0, w1), w2);
    if (!(sum > 0.0f))
        return MSWeights{ 0.0f, 0.0f, 0.0f, false };
    return MSWeights{ w0 / sum, w1 / sum, w2 / sum, true };
}

static inline MSAxis ms_axis(float pos, int size, int wrap)
{
    if (wrap == MS_WRAP_REPEAT)
    {
        float floor_pos = floor(pos);
        int raw = int(floor_pos);
        int wrapped = ((raw % size) + size) % size;
        return MSAxis{ wrapped, (wrapped + 1) % size, ms_add(pos, -floor_pos) };
    }
    float limit = float(size - 1);
    float clamped = fmin(fmax(pos, 0.0f), limit);
    float floor_pos = floor(clamped);
    int i = int(floor_pos);
    int i1 = i + 1 < size ? i + 1 : size - 1;
    return MSAxis{ i, i1, ms_add(clamped, -floor_pos) };
}

static inline float ms_texel(
    device const uchar* source,
    int width,
    int x,
    int y,
    int channel)
{
    ulong offset = (ulong(y) * ulong(width) + ulong(x)) * 4ul + ulong(channel);
    return float(source[offset]) / 255.0f;
}

static inline float ms_lerp(float a, float b, float t)
{
    return ms_add(a, ms_mul(ms_add(b, -a), t));
}

static inline uchar ms_quantize(float value)
{
    float q = rint(ms_mul(value, 255.0f));
    q = fmin(fmax(q, 0.0f), 255.0f);
    return uchar(q);
}

static inline void ms_sample(
    device const uchar* source,
    int width,
    int height,
    float u,
    float v,
    int wrap,
    device uchar* out)
{
    MSAxis ax = ms_axis(ms_add(ms_mul(u, float(width)), -0.5f), width, wrap);
    MSAxis ay = ms_axis(ms_add(ms_mul(v, float(height)), -0.5f), height, wrap);
    for (int c = 0; c < 4; ++c)
    {
        float top = ms_lerp(
            ms_texel(source, width, ax.i0, ay.i0, c),
            ms_texel(source, width, ax.i1, ay.i0, c),
            ax.frac);
        float bottom = ms_lerp(
            ms_texel(source, width, ax.i0, ay.i1, c),
            ms_texel(source, width, ax.i1, ay.i1, c),
            ax.frac);
        out[c] = ms_quantize(ms_lerp(top, bottom, ay.frac));
    }
}

kernel void mesh_raster_serial(
    device const float* dest [[buffer(0)]],
    device const float* source_uv [[buffer(1)]],
    device const ushort* ebo [[buffer(2)]],
    device const uchar* source [[buffer(3)]],
    device uchar* out [[buffer(4)]],
    constant MeshRasterParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint pixel = gid;
    uint pixel_count = uint(p.width) * uint(p.height);
    if (pixel >= pixel_count)
        return;

    int px = int(pixel % uint(p.width));
    int py = int(pixel / uint(p.width));
    float sx = ms_add(float(px), 0.5f);
    float sy = ms_add(float(py), 0.5f);

    int chosen = -1;
    int covered = 0;
    for (int triangle = 0; triangle < p.triangle_count; ++triangle)
    {
        int base = triangle * 3;
        ushort i0 = ebo[base];
        ushort i1 = ebo[base + 1];
        ushort i2 = ebo[base + 2];
        if (i0 == i1 || i1 == i2 || i0 == i2)
            continue;

        MSVec2 p0 = ms_read_vec2(dest, i0);
        MSVec2 p1 = ms_read_vec2(dest, i1);
        MSVec2 p2 = ms_read_vec2(dest, i2);
        if (!isfinite(p0.x) || !isfinite(p0.y) || !isfinite(p1.x) || !isfinite(p1.y)
            || !isfinite(p2.x) || !isfinite(p2.y))
            continue;

        float area2 = ms_cross(ms_sub(p1, p0), ms_sub(p2, p0));
        if (fabs(area2) <= MS_DEGENERATE_AREA2)
            continue;

        bool front = ms_is_front(area2, p.winding);
        if (p.cull == MS_CULL_FRONT && front)
            continue;
        if (p.cull == MS_CULL_BACK && !front)
            continue;

        MSWeights w = ms_weights(p0, p1, p2, area2, sx, sy, p.edge);
        if (!w.ok)
            continue;

        covered += 1;
        if (covered == 1)
            chosen = triangle;
        else if (p.overlap == MS_OVERLAP_LAST)
            chosen = triangle;
        else if (p.overlap == MS_OVERLAP_REJECT)
            return; // multi-face pixel stays untouched
    }

    if (chosen < 0)
        return;

    int base = chosen * 3;
    ushort i0 = ebo[base];
    ushort i1 = ebo[base + 1];
    ushort i2 = ebo[base + 2];
    MSVec2 p0 = ms_read_vec2(dest, i0);
    MSVec2 p1 = ms_read_vec2(dest, i1);
    MSVec2 p2 = ms_read_vec2(dest, i2);
    float area2 = ms_cross(ms_sub(p1, p0), ms_sub(p2, p0));
    MSWeights w = ms_weights(p0, p1, p2, area2, sx, sy, p.edge);
    if (!w.ok)
        return;

    MSVec2 uv0 = ms_read_vec2(source_uv, i0);
    MSVec2 uv1 = ms_read_vec2(source_uv, i1);
    MSVec2 uv2 = ms_read_vec2(source_uv, i2);
    float u = ms_add(
        ms_add(ms_mul(w.b0, uv0.x), ms_mul(w.b1, uv1.x)),
        ms_mul(w.b2, uv2.x));
    float v = ms_add(
        ms_add(ms_mul(w.b0, uv0.y), ms_mul(w.b1, uv1.y)),
        ms_mul(w.b2, uv2.y));
    ms_sample(source, p.width, p.height, u, v, p.wrap, out + pixel * 4u);
}
