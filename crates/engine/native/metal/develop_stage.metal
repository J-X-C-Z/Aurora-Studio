//
//  develop_stage.metal
//  Aurora Studio — cross-platform develop stage (Apple Metal MSL).
//
//  Twin of `crates/engine/native/wgsl/develop_stage.wgsl` and of
//  `develop_cpu_ref` in `crates/engine/src/gpu_crossplatform.rs`.
//
//  White balance gains -> exposure -> highlight/shadow toe -> contrast
//  -> saturation -> linear-to-sRGB -> RGBA8, in one pass. Alpha is quantised
//  and passed through; no colour term reads it.
//
//  Compiled on macOS only. Nothing on Windows or Linux builds or runs this
//  file; it ships as source for the Metal lane and stays IMPLEMENTED, not
//  COMPILED, until a device test has run.
//
//  `DevelopParams` is byte-identical to the WGSL struct (48 bytes), so a host
//  can share one parameter buffer between the wgpu and Metal paths.
//

#include <metal_stdlib>
using namespace metal;

struct DevelopParams {
    float4 wb;          // rgb white-balance gains, .w unused
    float exposure;     // linear multiplier, i.e. 2^EV
    float highlights;   // -1..1
    float shadows;      // -1..1
    float contrast;     // -1..1, already scaled by /50*0.8
    float saturation;   // 1.0 = neutral
    float pad0;
    float pad1;
    float pad2;
};

constant float HDR_TOE_K = 1.0f;

static inline float luma709(float3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

static inline float linear_to_srgb(float v) {
    if (v <= 0.0031308f) {
        return v * 12.92f;
    }
    return 1.055f * pow(v, 1.0f / 2.4f) - 0.055f;
}

static inline float contrast_curve(float g, float amount) {
    if (fabs(amount) < 1e-4f) {
        return g;
    }
    float k = 1.0f + amount;
    if (k >= 1.0f) {
        float x = (g - 0.5f) * k;
        return 0.5f + x / (1.0f + fabs(x) * (k - 1.0f) * 0.8f);
    }
    return 0.5f + (g - 0.5f) * k;
}

// `CImgOpLocalToneMappingV8`'s hard toe, kernel `z2714f67010`.
static inline float hdr_toe(float x, float k) {
    float u = x * (2.0f * 0.9f);
    float v = fabs(u) + 1.0f;
    float r = (u * v) / (v + u * u) * 0.5f + 0.5f;
    r = fmax(r, u - r + 1.0f) * (1.0f / (2.0f * k));
    float s = r * r;
    float num = (1.0f + r + s) * (r + s * s);
    return num / (1.0f + num) * k;
}

static inline float hdr_gain(float3 c, constant DevelopParams &params) {
    if (params.highlights == 0.0f && params.shadows == 0.0f) {
        return 1.0f;
    }
    float x = luma709(c);
    float b = 1.0f;
    if (params.highlights != 0.0f) {
        float t = hdr_toe(x, HDR_TOE_K);
        float exponent = params.highlights < 0.0f ? -t : t;
        float m = fabs(params.highlights);
        b = exp2(exponent) * m + (1.0f - m);
    }
    float a = 1.0f;
    if (params.shadows != 0.0f) {
        float t = hdr_toe(-x, HDR_TOE_K);
        float exponent = params.shadows < 0.0f ? -t : t;
        float m = fabs(params.shadows);
        a = exp2(exponent) * m + (1.0f - m);
    }
    return a / b;
}

static inline uint quantize_u8(float v) {
    return (uint)floor(clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
}

kernel void develop_stage(device const float4 *src [[buffer(0)]],
                          device uint *dst [[buffer(1)]],
                          constant DevelopParams &params [[buffer(2)]],
                          constant uint &count [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }
    float4 sample = src[gid];
    float3 c = float3(sample.x * params.wb.x,
                      sample.y * params.wb.y,
                      sample.z * params.wb.z) * params.exposure;

    c = c * hdr_gain(c, params);
    c = float3(contrast_curve(c.x, params.contrast),
               contrast_curve(c.y, params.contrast),
               contrast_curve(c.z, params.contrast));

    float l = luma709(c);
    float s = params.saturation;
    c = float3(l + (c.x - l) * s, l + (c.y - l) * s, l + (c.z - l) * s);

    uint r = quantize_u8(linear_to_srgb(c.x));
    uint g = quantize_u8(linear_to_srgb(c.y));
    uint b = quantize_u8(linear_to_srgb(c.z));
    uint a = quantize_u8(sample.w);
    dst[gid] = r | (g << 8u) | (b << 16u) | (a << 24u);
}
