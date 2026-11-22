static inline float recover_channel(float value, float2 neighbors,
                                    float threshold, float strength)
{
    if (!(value > threshold)) {
        return value;
    }
    const float excess = (value - threshold) * (strength / threshold);
    const float balance = min(neighbors.x, neighbors.y) / value;
    const float weight = min(excess * balance, 1.0f);
    const float mean = (neighbors.x + neighbors.y) * 0.5f;
    return mean * weight + value * (1.0f - weight);
}

static inline float normalized_excess(float value, float threshold, float span)
{
    return value > threshold ? (value - threshold) * (1.0f / span) : 0.0f;
}

__kernel void highlight(__global const float4 *src, __global float4 *dst,
                        float16 params)
{
    const size_t index = get_global_id(0);
    const float4 pixel = src[index];
    const float3 threshold = params.s012;
    const float strength = params.s3;
    float3 recovered = (float3)(
        recover_channel(pixel.x, pixel.yz, threshold.x, strength),
        recover_channel(pixel.y, pixel.xz, threshold.y, strength),
        recover_channel(pixel.z, pixel.yx, threshold.z, strength));

    if (any(recovered > threshold)) {
        const float3 span = params.s456;
        const float red_excess = normalized_excess(recovered.x, threshold.x, span.x);
        const float green_excess = normalized_excess(recovered.y, threshold.y, span.y);
        const float blue_excess = normalized_excess(recovered.z, threshold.z, span.z);
        const float3 neutral = recovered * params.s789;
        const float floor_level = min(min(neutral.x, neutral.y), neutral.z) * params.sa;
        const float fade = ((red_excess + green_excess) + blue_excess) * floor_level;
        const float weight = min(1.0f, fade * fade);
        const float luma = (recovered.x * 0.299f + recovered.y * 0.587f) + recovered.z * 0.114f;
        recovered = recovered * (1.0f - weight) + luma * weight;
    }

    dst[index] = (float4)(recovered, pixel.w);
}
