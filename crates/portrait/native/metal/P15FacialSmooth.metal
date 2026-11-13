#include <metal_stdlib>
using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

// Rows 36-38 share the recovered FacialSmoothProcessV2 local composite.
// Buffers are float32 [0,1] RGBA; stride is in pixels, not bytes.
// The Apple host owns RGBA8 conversion (* 255.0f, rint, clamp and packing).
struct P15FacialSmoothUniforms
{
    uint width;
    uint height;
    uint stride;
    uint pf;
    float pfValue;
    float gate;
    float legalAlpha;
    float darkCircleAlpha;
    float diffFactor;
    float diffFactor2;
    float colorThresholdFactor;
    float mixAlpha;
    float diffMinThreshold;
    float diffMaxThreshold;
    float highFrequencyMinThreshold;
    float highFrequencyMaxThreshold;
    float surfaceRadius;
    float surfaceThreshold;
    float gaussOffset;
    uint roiSize;
    uint algoType;
    uint boxBlurStage;
};
static_assert(sizeof(P15FacialSmoothUniforms) == 88,
              "P15FacialSmoothUniforms must stay 88 bytes");

struct P15NeckTailUniforms
{
    uint count;
    uint auxPlaneCount;
};
static_assert(sizeof(P15NeckTailUniforms) == 8,
              "P15NeckTailUniforms must stay 8 bytes");

static inline float p15_mul(float a, float b) { return a * b; }
static inline float p15_add(float a, float b) { return a + b; }
static inline float p15_sub(float a, float b) { return a - b; }
static inline float p15_div(float a, float b) { return a / b; }

static inline float p15_max(float a, float b) { return b > a ? b : a; }
static inline float p15_min(float a, float b) { return b < a ? b : a; }

static inline float p15_clamp(float value, float minimum, float maximum)
{
    return p15_min(p15_max(value, minimum), maximum);
}

// GLSL mix(x,y,a) is x*(1-a)+y*a; a fused operation would change rounding.
static inline float p15_mix(float x, float y, float amount)
{
    return p15_add(p15_mul(x, p15_sub(1.0f, amount)), p15_mul(y, amount));
}

static inline float4 p15_load(device const float* rgba, uint offset)
{
    return float4(rgba[offset], rgba[offset + 1], rgba[offset + 2], rgba[offset + 3]);
}

static inline float4 p15_bilinear(
    device const float* rgba,
    P15FacialSmoothUniforms u,
    float2 uv)
{
    float2 position = uv * float2(u.width, u.height) - 0.5f;
    float2 origin = floor(position);
    float2 fraction = position - origin;
    int x0 = clamp(int(origin.x), 0, int(u.width) - 1);
    int x1 = clamp(int(origin.x) + 1, 0, int(u.width) - 1);
    int y0 = clamp(int(origin.y), 0, int(u.height) - 1);
    int y1 = clamp(int(origin.y) + 1, 0, int(u.height) - 1);
    uint row0 = uint(y0) * u.stride;
    uint row1 = uint(y1) * u.stride;
    uint a = (row0 + uint(x0)) * 4u;
    uint b = (row0 + uint(x1)) * 4u;
    uint c = (row1 + uint(x0)) * 4u;
    uint d = (row1 + uint(x1)) * 4u;
    float4 top = float4(
        p15_mix(rgba[a], rgba[b], fraction.x),
        p15_mix(rgba[a + 1], rgba[b + 1], fraction.x),
        p15_mix(rgba[a + 2], rgba[b + 2], fraction.x),
        p15_mix(rgba[a + 3], rgba[b + 3], fraction.x));
    float4 bottom = float4(
        p15_mix(rgba[c], rgba[d], fraction.x),
        p15_mix(rgba[c + 1], rgba[d + 1], fraction.x),
        p15_mix(rgba[c + 2], rgba[d + 2], fraction.x),
        p15_mix(rgba[c + 3], rgba[d + 3], fraction.x));
    return float4(
        p15_mix(top.x, bottom.x, fraction.y),
        p15_mix(top.y, bottom.y, fraction.y),
        p15_mix(top.z, bottom.z, fraction.y),
        p15_mix(top.w, bottom.w, fraction.y));
}

static inline float p15_smoothstep(float edge0, float edge1, float value)
{
    float t = p15_clamp(p15_div(p15_sub(value, edge0), p15_sub(edge1, edge0)),
                        0.0f, 1.0f);
    return p15_mul(p15_mul(t, t), p15_sub(3.0f, p15_mul(2.0f, t)));
}

// SurfaceBlurBasic samples -1,+1,-2,+2,-3,+3,-4,+4 and preserves source alpha.
static inline float4 p15_surface_sample(
    device const float* source,
    P15FacialSmoothUniforms u,
    float2 center,
    float2 step)
{
    float4 centerRgba = p15_bilinear(source, u, center);
    float3 centerColor = centerRgba.xyz;
    float sumR = centerColor.x;
    float sumG = centerColor.y;
    float sumB = centerColor.z;
    float totalWeight = 1.0f;
    const int distances[8] = {-1, 1, -2, 2, -3, 3, -4, 4};
    for (uint i = 0; i < 8; ++i)
    {
        float4 sample = p15_bilinear(
            source, u, center + step * float(distances[i]));
        float dr = p15_sub(sample.x, centerColor.x);
        float dg = p15_sub(sample.y, centerColor.y);
        float db = p15_sub(sample.z, centerColor.z);
        float squared = p15_add(p15_add(p15_mul(dr, dr), p15_mul(dg, dg)),
                                p15_mul(db, db));
        float difference = sqrt(squared);
        float weight = 1.0f - p15_smoothstep(
            0.0f, p15_div(u.surfaceThreshold, 255.0f), difference);
        sumR = p15_add(sumR, p15_mul(sample.x, weight));
        sumG = p15_add(sumG, p15_mul(sample.y, weight));
        sumB = p15_add(sumB, p15_mul(sample.z, weight));
        totalWeight = p15_add(totalWeight, weight);
    }
    return float4(p15_div(sumR, totalWeight),
                  p15_div(sumG, totalWeight),
                  p15_div(sumB, totalWeight),
                  centerRgba.w);
}

// 1DGaussBlur multiplies all four channels; unlike SurfaceBlur/BoxBlur it does
// not copy the center alpha byte into the output.
static inline float4 p15_gauss_sample(
    device const float* source,
    P15FacialSmoothUniforms u,
    float2 center,
    float2 step)
{
    const int distances[8] = {-4, -3, -2, -1, 1, 2, 3, 4};
    const float weights[8] = {0.05f, 0.09f, 0.12f, 0.15f,
                              0.15f, 0.12f, 0.09f, 0.05f};
    float4 sum = p15_bilinear(source, u, center) * 0.18f;
    for (uint i = 0; i < 8; ++i)
    {
        float4 sample = p15_bilinear(
            source, u, center + step * float(distances[i]));
        sum += sample * weights[i];
    }
    return sum;
}

// FacialSmoothBoxBlur averages nine bilinear samples and preserves source alpha.
static inline float4 p15_box_sample(
    device const float* source,
    P15FacialSmoothUniforms u,
    float2 center,
    float2 step)
{
    float3 sum = float3(0.0f);
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i)
            sum += p15_bilinear(
                source, u, center + step * float2(i, j)).xyz;
    return float4(sum / 9.0f,
                  p15_bilinear(source, u, center).w);
}

static inline float p15_composite_channel(
    float sourceValue,
    float baseValue,
    float residual,
    float alpha,
    constant P15FacialSmoothUniforms& u)
{
    float high = p15_sub(sourceValue, baseValue);
    float diffLeft = p15_mul(p15_sub(residual, baseValue), u.diffFactor);
    float diffRight = p15_mul(residual, u.diffFactor2);
    float diff = p15_clamp(p15_add(diffLeft, diffRight),
                           u.diffMinThreshold, u.diffMaxThreshold);
    float referenceThreshold = p15_mix(
        p15_mul(residual, u.colorThresholdFactor),
        residual,
        u.mixAlpha);
    float maximum = p15_max(referenceThreshold, p15_sub(sourceValue, high));
    float beforeClamp = p15_sub(
        p15_add(sourceValue, diff), high);
    float reference = p15_add(
        p15_clamp(beforeClamp, 0.0f, maximum), high);
    float mixed = p15_mix(sourceValue, reference, alpha);
    float highFrequency = p15_clamp(
        p15_sub(mixed, sourceValue),
        u.highFrequencyMinThreshold,
        u.highFrequencyMaxThreshold);
    return p15_add(sourceValue, highFrequency);
}

kernel void p15_facial_smooth_composite(
    device const float* sourceRgba [[buffer(0)]],
    device const float* maskRgba [[buffer(1)]],
    device const float* skinMaskR [[buffer(2)]],
    device const float* baseRgba [[buffer(3)]],
    device const float* residualBaseRgba [[buffer(4)]],
    device float* outRgba [[buffer(5)]],
    constant P15FacialSmoothUniforms& uniforms [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) return;
    uint pixel = gid.y * uniforms.stride + gid.x;
    uint offset = pixel * 4u;
    float4 source = p15_load(sourceRgba, offset);
    float4 mask = p15_load(maskRgba, offset);
    float4 baseColor = p15_load(baseRgba, offset);
    float4 residualBase = p15_load(residualBaseRgba, offset);

    float alpha = p15_mul(
        p15_max(
            p15_max(p15_mul(uniforms.darkCircleAlpha, mask.x),
                    p15_mul(uniforms.legalAlpha, mask.z)),
            0.0f),
        skinMaskR[pixel]);
    float4 result = source;
    if (alpha > 0.0f)
    {
        result.x = p15_composite_channel(
            source.x, baseColor.x, residualBase.x, alpha, uniforms);
        result.y = p15_composite_channel(
            source.y, baseColor.y, residualBase.y, alpha, uniforms);
        result.z = p15_composite_channel(
            source.z, baseColor.z, residualBase.z, alpha, uniforms);
    }
    result.w = source.w;
    outRgba[offset] = result.x;
    outRgba[offset + 1] = result.y;
    outRgba[offset + 2] = result.z;
    outRgba[offset + 3] = result.w;
}

kernel void p15_surface_blur_pass(
    device const float* sourceRgba [[buffer(0)]],
    device float* outRgba [[buffer(1)]],
    constant P15FacialSmoothUniforms& uniforms [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) return;
    float2 center = (float2(gid) + 0.5f) / float2(uniforms.width, uniforms.height);
    // FacialSmoothBoxBlur uses one-pixel shifts in both directions in one pass.
    float2 step = float2(p15_div(1.0f, float(uniforms.width)),
                         p15_div(1.0f, float(uniforms.height)));
    float4 result = p15_box_sample(sourceRgba, uniforms, center, step);
    uint offset = (gid.y * uniforms.stride + gid.x) * 4u;
    outRgba[offset] = result.x;
    outRgba[offset + 1] = result.y;
    outRgba[offset + 2] = result.z;
    outRgba[offset + 3] = result.w;
}

kernel void p15_gauss_pass(
    device const float* sourceRgba [[buffer(0)]],
    device float* outRgba [[buffer(1)]],
    constant P15FacialSmoothUniforms& uniforms [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) return;
    float2 center = (float2(gid) + 0.5f) / float2(uniforms.width, uniforms.height);
    float2 step;
    if (uniforms.boxBlurStage == 1u)
        step = float2(p15_div(uniforms.gaussOffset, float(uniforms.width)), 0.0f);
    else
        step = float2(0.0f, p15_div(uniforms.gaussOffset, float(uniforms.height)));
    float4 result = p15_gauss_sample(sourceRgba, uniforms, center, step);
    uint offset = (gid.y * uniforms.stride + gid.x) * 4u;
    outRgba[offset] = result.x;
    outRgba[offset + 1] = result.y;
    outRgba[offset + 2] = result.z;
    outRgba[offset + 3] = result.w;
}

kernel void p15_surface_pass(
    device const float* sourceRgba [[buffer(0)]],
    device float* outRgba [[buffer(1)]],
    constant P15FacialSmoothUniforms& uniforms [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) return;
    float2 center = (float2(gid) + 0.5f) / float2(uniforms.width, uniforms.height);
    float2 step;
    if (uniforms.boxBlurStage == 1u)
        step = float2(p15_div(uniforms.surfaceRadius, float(uniforms.width)), 0.0f);
    else
        step = float2(0.0f, p15_div(uniforms.surfaceRadius, float(uniforms.height)));
    float4 result = p15_surface_sample(sourceRgba, uniforms, center, step);
    uint offset = (gid.y * uniforms.stride + gid.x) * 4u;
    outRgba[offset] = result.x;
    outRgba[offset + 1] = result.y;
    outRgba[offset + 2] = result.z;
    outRgba[offset + 3] = result.w;
}

// Row 35: recovered model tails and 64x64 max-pool. The post-0x5B7B390
// vendor blend is deliberately absent; no alternative composite is emitted.
kernel void p15_neck_repair_tails(
    device const float* rawBlend [[buffer(0)]],
    device const float* rawColor [[buffer(1)]],
    device const float* rawAux [[buffer(2)]],
    device float* blendTail [[buffer(3)]],
    device float* colorTail [[buffer(4)]],
    device float* auxTail [[buffer(5)]],
    constant P15NeckTailUniforms& uniforms [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uniforms.count) return;
    float blend = rawBlend[gid];
    blendTail[gid] = p15_add(
        p15_mul(0.5f, p15_clamp(blend, -1.0f, 1.0f)), 0.5f);
    for (uint channel = 0; channel < 3u; ++channel)
    {
        uint offset = channel * uniforms.count + gid;
        colorTail[offset] = p15_add(
            p15_mul(127.5f, rawColor[offset]), 127.5f);
    }
    if (gid < uniforms.auxPlaneCount)
    {
        for (uint channel = 0; channel < 3u; ++channel)
        {
            uint offset = channel * uniforms.auxPlaneCount + gid;
            auxTail[offset] = p15_mul(127.5f, rawAux[offset]);
        }
    }
}

kernel void p15_neck_max_pool(
    device const float* blendTail [[buffer(0)]],
    device float* coarse8x10 [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= 10u || gid.y >= 8u) return;
    float maximum = -INFINITY;
    for (uint y = gid.y * 64u; y < (gid.y + 1u) * 64u; ++y)
        for (uint x = gid.x * 64u; x < (gid.x + 1u) * 64u; ++x)
            maximum = p15_max(maximum, blendTail[y * 640u + x]);
    coarse8x10[gid.y * 10u + gid.x] = maximum;
}
