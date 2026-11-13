#include <metal_stdlib>
using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

// PF100007 direct pixel displacement. Mirrors kernels::smile_warp::
// pf100007_apply: source = p + degree*(lip/255)*(mouth/255)*(dx,dy),
// bilinear BGRA, truncating store, source alpha preserved.

struct SmileParams
{
    uint width;
    uint height;
    uint roi_x;
    uint roi_y;
    uint roi_width;
    uint roi_height;
    float degree;
    uint flags;
};
static_assert(sizeof(SmileParams) == 32, "SmileParams must stay 32 bytes");

constant float SMILE_GATE = 0.001f;

static inline float sm_clamp(float v, float lo, float hi)
{
    return fmin(fmax(v, lo), hi);
}

static inline void sm_store_trunc(device uchar* out, uint offset, float value)
{
    out[offset] = uchar(sm_clamp(value, 0.0f, 255.0f));
}

kernel void pf100007_smile(
    device const uchar* source [[buffer(0)]],
    device const uchar* lipMask [[buffer(1)]],
    device const uchar* mouthMask [[buffer(2)]],
    device const float* modelDx [[buffer(3)]],
    device const float* modelDy [[buffer(4)]],
    device uchar* out [[buffer(5)]],
    constant SmileParams& p [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height)
        return;
    if (gid.x < p.roi_x || gid.y < p.roi_y
        || gid.x >= p.roi_x + p.roi_width || gid.y >= p.roi_y + p.roi_height)
        return;

    uint index = gid.y * p.width + gid.x;
    float support = (float(lipMask[index]) / 255.0f)
                  * (float(mouthMask[index]) / 255.0f);
    float weight = p.degree * support;
    if (support <= SMILE_GATE || weight <= 0.0f)
        return;

    volatile float weight_dx = weight * modelDx[index];
    volatile float weight_dy = weight * modelDy[index];
    float sx = float(gid.x) + weight_dx;
    float sy = float(gid.y) + weight_dy;
    float cx = sm_clamp(sx, 0.0f, float(p.width - 1));
    float cy = sm_clamp(sy, 0.0f, float(p.height - 1));
    uint x0 = uint(cx);
    uint y0 = uint(cy);
    uint x1 = min(x0 + 1, p.width - 1);
    uint y1 = min(y0 + 1, p.height - 1);
    float fx = cx - float(x0);
    float fy = cy - float(y0);
    uint row0 = y0 * p.width;
    uint row1 = y1 * p.width;
    uint offset = index * 4u;
    for (uint c = 0; c < 3u; ++c)
    {
        float p00 = float(source[(row0 + x0) * 4u + c]);
        float p10 = float(source[(row0 + x1) * 4u + c]);
        float p01 = float(source[(row1 + x0) * 4u + c]);
        float p11 = float(source[(row1 + x1) * 4u + c]);
        volatile float top_delta = (p10 - p00) * fx;
        volatile float bottom_delta = (p11 - p01) * fx;
        float top = p00 + top_delta;
        float bottom = p01 + bottom_delta;
        volatile float vertical_delta = (bottom - top) * fy;
        sm_store_trunc(out, offset + c, top + vertical_delta);
    }
    out[offset + 3u] = source[offset + 3u];
}
