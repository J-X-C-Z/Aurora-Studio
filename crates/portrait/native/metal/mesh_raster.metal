#include <metal_stdlib>
using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

// Apple Metal port of native/cuda/mesh_raster.cu and src/raster.rs.
// Two passes:
//   mesh_raster_select  one thread per triangle, bbox scan, overlap resolve
//   mesh_raster_emit    one thread per pixel, re-evaluate, bilinear sample
//
// dest/source_uv are interleaved float2, ebo is uint16, source/out are
// straight-alpha RGBA8. Uncovered pixels keep the caller-prefilled out bytes.

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

constant int MR_WRAP_REPEAT = 1;
constant int MR_WINDING_POSITIVE_IS_FRONT = 1;
constant int MR_CULL_FRONT = 2;
constant int MR_CULL_BACK = 3;
constant int MR_EDGE_INCLUSIVE = 2;
constant int MR_OVERLAP_REJECT = 1;
constant int MR_OVERLAP_FIRST = 2;
constant int MR_OVERLAP_LAST = 3;

constant float MR_DEGENERATE_AREA2 = 1e-12f;
constant uint MR_UNCOVERED = 0xFFFFFFFFu;

struct MRVec2
{
    float x;
    float y;
};

struct MRWeights
{
    float b0;
    float b1;
    float b2;
    bool ok;
};

struct MRPerspectiveUV
{
    float u;
    float v;
    float depth;
    bool ok;
};

struct MRAxis
{
    int i0;
    int i1;
    float frac;
};

struct MRBounds
{
    int x0;
    int x1;
    int y0;
    int y1;
};

static inline float mr_add(float a, float b) { return a + b; }
static inline float mr_mul(float a, float b) { return a * b; }

static inline MRVec2 mr_sub(MRVec2 a, MRVec2 b)
{
    return MRVec2{ mr_add(a.x, -b.x), mr_add(a.y, -b.y) };
}

static inline float mr_cross(MRVec2 a, MRVec2 b)
{
    return mr_add(mr_mul(a.x, b.y), -mr_mul(a.y, b.x));
}

static inline MRVec2 mr_read_vec2(device const float* values, ushort index)
{
    uint offset = uint(index) * 2u;
    return MRVec2{ values[offset], values[offset + 1u] };
}

// Perspective-correct UV for normalized barycentric weights.
// vertex_perspective is [depth, reciprocal_w] per vertex.
static inline MRPerspectiveUV mr_perspective_uv(
    MRVec2 uv0,
    MRVec2 uv1,
    MRVec2 uv2,
    MRVec2 vp0,
    MRVec2 vp1,
    MRVec2 vp2,
    MRWeights w)
{
    float q0 = mr_mul(w.b0, vp0.y);
    float q1 = mr_mul(w.b1, vp1.y);
    float q2 = mr_mul(w.b2, vp2.y);
    float inv_w = mr_add(mr_add(q0, q1), q2);
    if (!(inv_w > 0.0f) || !isfinite(inv_w))
        return MRPerspectiveUV{ 0.0f, 0.0f, 0.0f, false };

    float u = mr_add(
        mr_add(mr_mul(q0, uv0.x), mr_mul(q1, uv1.x)),
        mr_mul(q2, uv2.x)) / inv_w;
    float v = mr_add(
        mr_add(mr_mul(q0, uv0.y), mr_mul(q1, uv1.y)),
        mr_mul(q2, uv2.y)) / inv_w;
    float depth = mr_add(
        mr_add(mr_mul(q0, vp0.x), mr_mul(q1, vp1.x)),
        mr_mul(q2, vp2.x)) / inv_w;
    return MRPerspectiveUV{ u, v, depth, true };
}

static inline bool mr_is_front(float area2, int winding)
{
    bool positive = area2 > 0.0f;
    return winding == MR_WINDING_POSITIVE_IS_FRONT ? positive : !positive;
}

static inline MRWeights mr_weights(
    MRVec2 p0,
    MRVec2 p1,
    MRVec2 p2,
    float area2,
    float sx,
    float sy,
    int edge)
{
    MRVec2 s = MRVec2{ sx, sy };
    float w0 = mr_cross(mr_sub(p1, s), mr_sub(p2, s)) / area2;
    float w1 = mr_cross(mr_sub(p2, s), mr_sub(p0, s)) / area2;
    float w2 = mr_cross(mr_sub(p0, s), mr_sub(p1, s)) / area2;
    bool inside = edge == MR_EDGE_INCLUSIVE
        ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
        : (w0 > 0.0f && w1 > 0.0f && w2 > 0.0f);
    if (!inside)
        return MRWeights{ 0.0f, 0.0f, 0.0f, false };
    w0 = fmax(w0, 0.0f);
    w1 = fmax(w1, 0.0f);
    w2 = fmax(w2, 0.0f);
    float sum = mr_add(mr_add(w0, w1), w2);
    if (!(sum > 0.0f))
        return MRWeights{ 0.0f, 0.0f, 0.0f, false };
    return MRWeights{ w0 / sum, w1 / sum, w2 / sum, true };
}

// p = uv * size - 0.5 texel-center rule; clamp or repeat per axis.
static inline MRAxis mr_axis(float pos, int size, int wrap)
{
    if (wrap == MR_WRAP_REPEAT)
    {
        float floor_pos = floor(pos);
        int raw = int(floor_pos);
        int wrapped = ((raw % size) + size) % size;
        return MRAxis{ wrapped, (wrapped + 1) % size, mr_add(pos, -floor_pos) };
    }
    float limit = float(size - 1);
    float clamped = fmin(fmax(pos, 0.0f), limit);
    float floor_pos = floor(clamped);
    int i = int(floor_pos);
    int i1 = i + 1 < size ? i + 1 : size - 1;
    return MRAxis{ i, i1, mr_add(clamped, -floor_pos) };
}

static inline float mr_texel(
    device const uchar* source,
    int width,
    int x,
    int y,
    int channel)
{
    ulong offset =
        (ulong(y) * ulong(width) + ulong(x)) * 4ul + ulong(channel);
    return float(source[offset]) / 255.0f;
}

static inline float mr_lerp(float a, float b, float t)
{
    return mr_add(a, mr_mul(mr_add(b, -a), t));
}

static inline uchar mr_quantize(float value)
{
    float q = rint(mr_mul(value, 255.0f));
    q = fmin(fmax(q, 0.0f), 255.0f);
    return uchar(q);
}

static inline void mr_sample(
    device const uchar* source,
    int width,
    int height,
    float u,
    float v,
    int wrap,
    device uchar* out)
{
    MRAxis ax = mr_axis(mr_add(mr_mul(u, float(width)), -0.5f), width, wrap);
    MRAxis ay = mr_axis(mr_add(mr_mul(v, float(height)), -0.5f), height, wrap);
    for (int c = 0; c < 4; ++c)
    {
        float top = mr_lerp(
            mr_texel(source, width, ax.i0, ay.i0, c),
            mr_texel(source, width, ax.i1, ay.i0, c),
            ax.frac);
        float bottom = mr_lerp(
            mr_texel(source, width, ax.i0, ay.i1, c),
            mr_texel(source, width, ax.i1, ay.i1, c),
            ax.frac);
        out[c] = mr_quantize(mr_lerp(top, bottom, ay.frac));
    }
}

static inline MRBounds mr_bounds(
    MRVec2 p0,
    MRVec2 p1,
    MRVec2 p2,
    int width,
    int height)
{
    float min_x = fmin(p0.x, fmin(p1.x, p2.x));
    float max_x = fmax(p0.x, fmax(p1.x, p2.x));
    float min_y = fmin(p0.y, fmin(p1.y, p2.y));
    float max_y = fmax(p0.y, fmax(p1.y, p2.y));
    float limit_x = float(width - 1);
    float limit_y = float(height - 1);
    return MRBounds{
        int(ceil(fmin(fmax(mr_add(min_x, -0.5f), 0.0f), limit_x))),
        int(floor(fmin(fmax(mr_add(max_x, -0.5f), 0.0f), limit_x))),
        int(ceil(fmin(fmax(mr_add(min_y, -0.5f), 0.0f), limit_y))),
        int(floor(fmin(fmax(mr_add(max_y, -0.5f), 0.0f), limit_y)))
    };
}

kernel void mesh_raster_select(
    device const float* dest [[buffer(0)]],
    device const ushort* ebo [[buffer(1)]],
    constant MeshRasterParams& p [[buffer(2)]],
    device atomic_uint* selected [[buffer(3)]],
    device atomic_uint* coverage [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int triangle = int(gid);
    if (triangle >= p.triangle_count)
        return;

    int base = triangle * 3;
    ushort i0 = ebo[base];
    ushort i1 = ebo[base + 1];
    ushort i2 = ebo[base + 2];
    if (i0 == i1 || i1 == i2 || i0 == i2)
        return;

    MRVec2 p0 = mr_read_vec2(dest, i0);
    MRVec2 p1 = mr_read_vec2(dest, i1);
    MRVec2 p2 = mr_read_vec2(dest, i2);
    if (!isfinite(p0.x) || !isfinite(p0.y) || !isfinite(p1.x) || !isfinite(p1.y)
        || !isfinite(p2.x) || !isfinite(p2.y))
        return;

    float area2 = mr_cross(mr_sub(p1, p0), mr_sub(p2, p0));
    if (fabs(area2) <= MR_DEGENERATE_AREA2)
        return;

    bool front = mr_is_front(area2, p.winding);
    if (p.cull == MR_CULL_FRONT && front)
        return;
    if (p.cull == MR_CULL_BACK && !front)
        return;

    MRBounds bounds = mr_bounds(p0, p1, p2, p.width, p.height);
    if (bounds.x0 > bounds.x1 || bounds.y0 > bounds.y1)
        return;

    for (int py = bounds.y0; py <= bounds.y1; ++py)
    {
        float sy = mr_add(float(py), 0.5f);
        for (int px = bounds.x0; px <= bounds.x1; ++px)
        {
            MRWeights w = mr_weights(
                p0, p1, p2, area2, mr_add(float(px), 0.5f), sy, p.edge);
            if (!w.ok)
                continue;

            uint pixel = uint(py) * uint(p.width) + uint(px);
            atomic_fetch_add_explicit(
                &coverage[pixel], 1u, memory_order_relaxed);
            // Every writer uses the same atomic RMW. A first-writer store can
            // race with a later fetch_min/max and overwrite the winner.
            if (p.overlap == MR_OVERLAP_LAST)
            {
                atomic_fetch_max_explicit(
                    &selected[pixel], uint(triangle), memory_order_relaxed);
            }
            else
            {
                atomic_fetch_min_explicit(
                    &selected[pixel], uint(triangle), memory_order_relaxed);
            }
        }
    }
}

kernel void mesh_raster_emit(
    device const float* dest [[buffer(0)]],
    device const float* source_uv [[buffer(1)]],
    device const ushort* ebo [[buffer(2)]],
    device const uchar* source [[buffer(3)]],
    device uchar* out [[buffer(4)]],
    constant MeshRasterParams& p [[buffer(5)]],
    device const uint* selected [[buffer(6)]],
    device const uint* coverage [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint pixel = gid;
    uint pixel_count = uint(p.width) * uint(p.height);
    if (pixel >= pixel_count)
        return;

    if (coverage[pixel] == 0u)
        return;
    uint triangle = selected[pixel];
    if (triangle == MR_UNCOVERED)
        return;
    if (coverage[pixel] > 1u && p.overlap == MR_OVERLAP_REJECT)
        return;

    int px = int(pixel % uint(p.width));
    int py = int(pixel / uint(p.width));
    int base = int(triangle) * 3;
    ushort i0 = ebo[base];
    ushort i1 = ebo[base + 1];
    ushort i2 = ebo[base + 2];

    MRVec2 p0 = mr_read_vec2(dest, i0);
    MRVec2 p1 = mr_read_vec2(dest, i1);
    MRVec2 p2 = mr_read_vec2(dest, i2);
    float area2 = mr_cross(mr_sub(p1, p0), mr_sub(p2, p0));
    if (fabs(area2) <= MR_DEGENERATE_AREA2)
        return;

    MRWeights w = mr_weights(
        p0, p1, p2, area2, mr_add(float(px), 0.5f), mr_add(float(py), 0.5f), p.edge);
    if (!w.ok)
        return;

    MRVec2 uv0 = mr_read_vec2(source_uv, i0);
    MRVec2 uv1 = mr_read_vec2(source_uv, i1);
    MRVec2 uv2 = mr_read_vec2(source_uv, i2);
    float u = mr_add(
        mr_add(mr_mul(w.b0, uv0.x), mr_mul(w.b1, uv1.x)),
        mr_mul(w.b2, uv2.x));
    float v = mr_add(
        mr_add(mr_mul(w.b0, uv0.y), mr_mul(w.b1, uv1.y)),
        mr_mul(w.b2, uv2.y));
    mr_sample(source, p.width, p.height, u, v, p.wrap, out + pixel * 4u);
}

// Additive perspective path. The existing affine emit kernel is unchanged;
// buffer 8 carries [depth, reciprocal_w] per vertex. This kernel is not yet
// selected by MetalContext in metal.rs.
kernel void mesh_raster_emit_perspective(
    device const float* dest [[buffer(0)]],
    device const float* source_uv [[buffer(1)]],
    device const ushort* ebo [[buffer(2)]],
    device const uchar* source [[buffer(3)]],
    device uchar* out [[buffer(4)]],
    constant MeshRasterParams& p [[buffer(5)]],
    device const uint* selected [[buffer(6)]],
    device const uint* coverage [[buffer(7)]],
    device const float* vertex_perspective [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    uint pixel = gid;
    uint pixel_count = uint(p.width) * uint(p.height);
    if (pixel >= pixel_count)
        return;

    if (coverage[pixel] == 0u)
        return;
    uint triangle = selected[pixel];
    if (triangle == MR_UNCOVERED)
        return;
    if (coverage[pixel] > 1u && p.overlap == MR_OVERLAP_REJECT)
        return;

    int px = int(pixel % uint(p.width));
    int py = int(pixel / uint(p.width));
    int base = int(triangle) * 3;
    ushort i0 = ebo[base];
    ushort i1 = ebo[base + 1];
    ushort i2 = ebo[base + 2];

    MRVec2 p0 = mr_read_vec2(dest, i0);
    MRVec2 p1 = mr_read_vec2(dest, i1);
    MRVec2 p2 = mr_read_vec2(dest, i2);
    float area2 = mr_cross(mr_sub(p1, p0), mr_sub(p2, p0));
    if (fabs(area2) <= MR_DEGENERATE_AREA2)
        return;

    MRWeights w = mr_weights(
        p0, p1, p2, area2, mr_add(float(px), 0.5f), mr_add(float(py), 0.5f), p.edge);
    if (!w.ok)
        return;

    MRVec2 uv0 = mr_read_vec2(source_uv, i0);
    MRVec2 uv1 = mr_read_vec2(source_uv, i1);
    MRVec2 uv2 = mr_read_vec2(source_uv, i2);
    MRVec2 vp0 = mr_read_vec2(vertex_perspective, i0);
    MRVec2 vp1 = mr_read_vec2(vertex_perspective, i1);
    MRVec2 vp2 = mr_read_vec2(vertex_perspective, i2);
    MRPerspectiveUV puv = mr_perspective_uv(
        uv0, uv1, uv2, vp0, vp1, vp2, w);
    if (!puv.ok)
        return;
    // puv.depth is carried for a future depth attachment; this candidate
    // does not perform a depth test.
    mr_sample(source, p.width, p.height, puv.u, puv.v, p.wrap, out + pixel * 4u);
}

// Serial perspective candidate for the fallback topology. It lives in this
// source bundle so mesh_raster_serial.metal keeps its existing single-kernel
// contract; MetalContext in metal.rs does not select it yet.
kernel void mesh_raster_serial_perspective(
    device const float* dest [[buffer(0)]],
    device const float* source_uv [[buffer(1)]],
    device const ushort* ebo [[buffer(2)]],
    device const uchar* source [[buffer(3)]],
    device uchar* out [[buffer(4)]],
    constant MeshRasterParams& p [[buffer(5)]],
    device const float* vertex_perspective [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint pixel = gid;
    uint pixel_count = uint(p.width) * uint(p.height);
    if (pixel >= pixel_count)
        return;

    int px = int(pixel % uint(p.width));
    int py = int(pixel / uint(p.width));
    float sx = mr_add(float(px), 0.5f);
    float sy = mr_add(float(py), 0.5f);

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

        MRVec2 p0 = mr_read_vec2(dest, i0);
        MRVec2 p1 = mr_read_vec2(dest, i1);
        MRVec2 p2 = mr_read_vec2(dest, i2);
        if (!isfinite(p0.x) || !isfinite(p0.y) || !isfinite(p1.x) || !isfinite(p1.y)
            || !isfinite(p2.x) || !isfinite(p2.y))
            continue;

        float area2 = mr_cross(mr_sub(p1, p0), mr_sub(p2, p0));
        if (fabs(area2) <= MR_DEGENERATE_AREA2)
            continue;

        bool front = mr_is_front(area2, p.winding);
        if (p.cull == MR_CULL_FRONT && front)
            continue;
        if (p.cull == MR_CULL_BACK && !front)
            continue;

        MRWeights w = mr_weights(p0, p1, p2, area2, sx, sy, p.edge);
        if (!w.ok)
            continue;

        covered += 1;
        if (covered == 1)
            chosen = triangle;
        else if (p.overlap == MR_OVERLAP_LAST)
            chosen = triangle;
        else if (p.overlap == MR_OVERLAP_REJECT)
            return; // multi-face pixel stays untouched
    }

    if (chosen < 0)
        return;

    int base = chosen * 3;
    ushort i0 = ebo[base];
    ushort i1 = ebo[base + 1];
    ushort i2 = ebo[base + 2];
    MRVec2 p0 = mr_read_vec2(dest, i0);
    MRVec2 p1 = mr_read_vec2(dest, i1);
    MRVec2 p2 = mr_read_vec2(dest, i2);
    float area2 = mr_cross(mr_sub(p1, p0), mr_sub(p2, p0));
    MRWeights w = mr_weights(p0, p1, p2, area2, sx, sy, p.edge);
    if (!w.ok)
        return;

    MRVec2 uv0 = mr_read_vec2(source_uv, i0);
    MRVec2 uv1 = mr_read_vec2(source_uv, i1);
    MRVec2 uv2 = mr_read_vec2(source_uv, i2);
    MRVec2 vp0 = mr_read_vec2(vertex_perspective, i0);
    MRVec2 vp1 = mr_read_vec2(vertex_perspective, i1);
    MRVec2 vp2 = mr_read_vec2(vertex_perspective, i2);
    MRPerspectiveUV puv = mr_perspective_uv(
        uv0, uv1, uv2, vp0, vp1, vp2, w);
    if (!puv.ok)
        return;
    // puv.depth is carried for a future depth attachment; this candidate
    // does not perform a depth test.
    mr_sample(source, p.width, p.height, puv.u, puv.v, p.wrap, out + pixel * 4u);
}
