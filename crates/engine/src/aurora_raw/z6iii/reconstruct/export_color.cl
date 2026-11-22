#pragma OPENCL FP_CONTRACT OFF

static float auto_curve(float sample, __constant const float *nodes, int count) {
    int right = 1;
    while (right + 1 < count && sample >= nodes[3 * right]) {
        ++right;
    }
    const float3 first = vload3(right - 1, nodes);
    const float3 last = vload3(right, nodes);
    const float width = last.x - first.x;
    if (fabs(width) < 0.0001f) {
        return 0.5f * fma(sample - first.x, first.z + last.z, first.y + last.y);
    }
    const float position = (sample - first.x) / width;
    const float prior = position - 1.f;
    const float value = fma(position * position * (last.y - first.y), fma(-2.f, position, 3.f), first.y);
    const float tangent = fma(first.z, prior, last.z * position);
    return fma(width, position * tangent * prior, value);
}

static float maximum(float3 rgb) { return fmax(rgb.x, fmax(rgb.y, rgb.z)); }
static float minimum(float3 rgb) { return fmin(rgb.x, fmin(rgb.y, rgb.z)); }

static float luminance(float3 rgb) {
    return fabs(fma(rgb.z, 0.114f, fma(rgb.x, 0.299f, rgb.y * 0.587f))) + 1.e-10f;
}

static float encode_mix(float sample) {
    const float slope = 128.f / 127.f;
    return sample > 1.f / 4096.f ? fma(native_sqrt(sample), slope, 1.f - slope) : sample * (32.f * slope);
}

static float decode_mix(float sample) {
    const float slope = 128.f / 127.f;
    if (sample <= slope / 128.f) return sample * (127.f / 4096.f);
    const float root = (sample - (1.f - slope)) * (127.f / 128.f);
    return root * root;
}

static float3 couple(float3 source, float3 film) {
    source = (float3)(encode_mix(source.x), encode_mix(source.y), encode_mix(source.z));
    float3 target = (float3)(encode_mix(film.x), encode_mix(film.y), encode_mix(film.z));
    float light = luminance(target);
    if (maximum(target) > 1.f) {
        const float clipped = luminance(fmin(target, 1.f));
        target *= clipped / light;
        light = clipped;
    }
    source *= light / luminance(source);
    const float strength = fma((maximum(target) - minimum(target) + 1.e-6f)
                             / (maximum(source) - minimum(source) + 1.e-6f), 0.5f, 0.5f);
    float3 result = fma(source - light, (float3)(strength), (float3)(light));
    const float distance = maximum(result) - light;
    if (distance > 0.f && 1.f - light < 2.f * distance) {
        const float ratio = (1.f - light) / distance;
        const float edge = fmax(1.f - fabs(ratio - 1.f), 0.f);
        const float smooth = fma(edge * edge * edge, -1.f / 6.f, fmin(ratio, 1.f));
        result = fma(result - light, (float3)(smooth), (float3)(light));
    }
    const float3 decoded = (float3)(decode_mix(result.x), decode_mix(result.y), decode_mix(result.z));
    return fma(film, (float3)(0.35f), decoded * (1.f - 0.35f));
}

static float encode_camera(float sample) {
    return sample > 0.01947f ? fma(native_sqrt(sample), 1.075f, 1.f - 1.075f) : 3.8521f * fmax(sample, 0.f);
}

static float3 camera_rgb(float3 raw, __constant const float *nodes, int count) {
    const float3 norm = fmax(raw / 65535.f, 0.f);
    const float3 input = fmax(fma((float3)(norm.z), (float3)(-.01f, -.01f, 1.1f),
        fma((float3)(norm.x), (float3)(1.08f, -.03f, -.03f), norm.y * (float3)(-.07f, 1.04f, -.07f))), 0.f);
    const float3 film = (float3)(auto_curve(input.x, nodes, count), auto_curve(input.y, nodes, count),
                                  auto_curve(input.z, nodes, count));
    const float3 result = couple(input, film);
    return (float3)(encode_camera(result.x), encode_camera(result.y), encode_camera(result.z));
}

static int table_index(int3 grid) { return (grid.x * 49 + grid.y) * 49 + grid.z; }

static float3 interpolate(float3 rgb, __global const ushort *table) {
    const float peak = fmax(1.f, maximum(rgb));
    const float3 position = clamp(rgb * (1.f / peak), 0.f, 1.f) * 48.f;
    const int3 low = convert_int3(position);
    const int3 high = min(low + 1, 48);
    const float3 fraction = position - convert_float3(low);
    int largest = fraction.x > fraction.y ? 0 : 1;
    if (fraction.z > fraction[largest]) largest = 2;
    int smallest = fraction.y < fraction.x ? 1 : 0;
    if (fraction.z < fraction[smallest]) smallest = 2;
    const float middle = fmin(fmax(fraction.x, fraction.y), fmax(fmin(fraction.x, fraction.y), fraction.z));
    int3 corner1 = low;
    corner1[largest] = high[largest];
    int3 corner2 = high;
    corner2[smallest] = low[smallest];
    const float3 base = convert_float3(vload3(table_index(low), table));
    const float3 first = convert_float3(vload3(table_index(corner1), table));
    const float3 second = convert_float3(vload3(table_index(corner2), table));
    const float3 last = convert_float3(vload3(table_index(high), table));
    const float largest_weight = fraction[largest];
    const float smallest_weight = fraction[smallest];
    const float3 subtotal = fma(base, (float3)(1.f - largest_weight), first * (largest_weight - middle));
    const float3 code = fma(last, (float3)(smallest_weight), fma(second, (float3)(middle - smallest_weight), subtotal));
    const float3 normalized = (code * peak) * (1.f / 65535.f);
    return normalized * 65535.f;
}

__kernel void export_color(__global const float4 *source, __global ushort *destination,
                           __constant const float *nodes, int count, __global const ushort *table) {
    const size_t index = get_global_id(0);
    const float3 result = interpolate(camera_rgb(source[index].xyz, nodes, count), table);
    vstore3(convert_ushort3_sat_rtz(result), index, destination);
}
