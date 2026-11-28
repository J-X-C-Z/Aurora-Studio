// Aurora native inference — Direct3D 12 compute kernels for AMD / Intel GPUs.
//
// One root layout serves every entry point (see `crates/portrait/src/dx12.rs`):
//   b0        : 32 packed constants in a 256-byte-aligned upload CBV, `uint P(0..31)` (float parameters are bit-cast)
//   t0..t7    : read-only byte-address buffers
//   u0..u7    : read/write byte-address buffers
//
// Tensors are 32-bit slots. FP16 inputs are unpacked to FP32 by the session
// before a kernel sees them, and `Bool` tensors are carried as 0.0/1.0 floats,
// so arithmetic below is uniform. Index math is 32-bit (all shapes fit in u32).
//
// Thread groups are 64 threads (a multiple of both RDNA Wave32 and GCN/Vega
// Wave64) unless a kernel needs a 2-D tile. Reductions use `groupshared`
// ladder reduction rather than wave intrinsics so they behave identically on
// Wave32 and Wave64 hardware.

cbuffer Params : register(b0)
{
    uint4 packed_params[8];
};
// Root DWORDs are packed; scalar HLSL cbuffer arrays have a 16-byte stride.
#define P(i) (packed_params[(uint)(i) >> 2u][(uint)(i) & 3u])

ByteAddressBuffer s0 : register(t0);
ByteAddressBuffer s1 : register(t1);
ByteAddressBuffer s2 : register(t2);
ByteAddressBuffer s3 : register(t3);
ByteAddressBuffer s4 : register(t4);
ByteAddressBuffer s5 : register(t5);
ByteAddressBuffer s6 : register(t6);
ByteAddressBuffer s7 : register(t7);

RWByteAddressBuffer u0 : register(u0);
RWByteAddressBuffer u1 : register(u1);
RWByteAddressBuffer u2 : register(u2);
RWByteAddressBuffer u3 : register(u3);
RWByteAddressBuffer u4 : register(u4);
RWByteAddressBuffer u5 : register(u5);
RWByteAddressBuffer u6 : register(u6);
RWByteAddressBuffer u7 : register(u7);

#define NT 64

float f32_at(ByteAddressBuffer buf, uint index)
{
    return asfloat(buf.Load(index * 4));
}

// The final CBV DWORD is the first logical group in a split 1D dispatch.
uint group_x(uint3 gid) { return P(31) + gid.x; }
uint linear_id(uint3 gid, uint3 tid)
{
    return group_x(gid) * NT + tid.x;
}

// ---------------------------------------------------------------- reductions

groupshared float g_reduce[NT];

float group_max(float value, uint lane)
{
    g_reduce[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = NT / 2; stride > 0; stride >>= 1)
    {
        if (lane < stride)
        {
            g_reduce[lane] = max(g_reduce[lane], g_reduce[lane + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float result = g_reduce[0];
    GroupMemoryBarrierWithGroupSync();
    return result;
}

float group_sum(float value, uint lane)
{
    g_reduce[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = NT / 2; stride > 0; stride >>= 1)
    {
        if (lane < stride)
        {
            g_reduce[lane] += g_reduce[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float result = g_reduce[0];
    GroupMemoryBarrierWithGroupSync();
    return result;
}

// ------------------------------------------------------------------ math bits

float sigmoid_f(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

// Numerical-Recipes erfc, |relative error| < 1.2e-7; erf = 1 - erfc.
float erfc_f(float x)
{
    float t = 1.0 / (1.0 + 0.5 * abs(x));
    float poly = t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
                 t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 +
                 t * 0.17087277))))))));
    float ty = t * exp(-x * x - 1.26551223 + poly);
    return x >= 0.0 ? ty : 2.0 - ty;
}

float erf_f(float x)
{
    return 1.0 - erfc_f(x);
}

// IEEE round-to-nearest-even float32 -> float16 bits. The `f32tof16` intrinsic
// truncates the mantissa on some AMD drivers, which would make `Cast(to=Float16)`
// disagree with ORT/CUDA by one ULP, so the conversion is done explicitly here.
uint f32_to_f16_rtne(float f)
{
    uint bits = asuint(f);
    uint sign = (bits >> 16) & 0x8000u;
    uint value_bits = bits & 0x7FFFFFFFu;
    uint half_bits = 0u;
    if (value_bits >= 0x477FF000u) // 65520 ties to infinity; preserve NaN classification.
    {
        half_bits = value_bits > 0x7F800000u ? 0x7E00u : 0x7C00u;
    }
    else if (value_bits >= 0x38800000u)
    {
        half_bits = (((value_bits >> 23) - 112u) << 10) | ((value_bits >> 13) & 0x3FFu);
        uint rem = value_bits & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (half_bits & 1u) != 0u))
            half_bits += 1u;
    }
    else if (value_bits >= 0x33000000u) // exactly half a minimum subnormal rounds to even zero
    {
        uint mantissa = (value_bits & 0x007FFFFFu) | 0x00800000u;
        uint shift = 126u - (value_bits >> 23);
        half_bits = mantissa >> shift;
        uint rem = mantissa & ((1u << shift) - 1u);
        uint halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (half_bits & 1u) != 0u))
            half_bits += 1u;
    }
    return sign | half_bits;
}

// Shader Model 5 pow is log-based and does not define negative bases. ONNX
// Pow follows real exponentiation, including integer exponents of negative bases.
float real_pow(float base, float exponent)
{
    uint ab = asuint(base) & 0x7fffffffu;
    uint eb = asuint(exponent) & 0x7fffffffu;
    float magnitude = asfloat(ab);
    const uint inf = 0x7f800000u;
    uint result_bits = 0x7fc00000u; // NaN, including a negative base with nonintegral exponent
    if (eb == 0u || asuint(base) == 0x3f800000u) {
        result_bits = 0x3f800000u;
    } else if (eb == inf && ab <= inf) {
        result_bits = magnitude == 1.0 ? 0x3f800000u : (((magnitude > 1.0) == (exponent > 0.0)) ? inf : 0u);
    } else if (ab <= inf && eb < inf) {
        bool integer_power = trunc(exponent) == exponent;
        // From 2^24 onward, all representable binary32 integers are even.
        bool odd = integer_power && abs(exponent) < 16777216.0 && (((uint)abs(exponent) & 1u) != 0u);
        uint sign_bit = odd ? asuint(base) & 0x80000000u : 0u;
        if (ab == 0u || ab == inf) {
            result_bits = sign_bit | (((ab == inf) == (exponent > 0.0)) ? inf : 0u);
        } else if (base >= 0.0 || integer_power) {
            float result = exponent == 2.0 ? base * base : pow(abs(base), exponent);
            result_bits = asuint(result) | sign_bit;
        }
    }
    return asfloat(result_bits);
}

// -------------------------------------------------------------- element-wise

// Binary opcodes (`P(20)`).
#define BIN_ADD 0
#define BIN_SUB 1
#define BIN_MUL 2
#define BIN_DIV 3
#define BIN_POW 4
#define BIN_PRELU 5
#define BIN_WHERE 6
#define BIN_EQUAL 7
#define BIN_GREATER 8
#define BIN_LESS 9
#define BIN_GREATER_EQUAL 10
#define BIN_AND 11
#define BIN_MIN 12
#define BIN_MAX 13

// P(0) = element count, P(1) = rank, P(2..7) = output dims,
// P(8..13) = left operand strides (elements), P(14..19) = right operand strides,
// P(20) = opcode, P(21) = unused scalar slot.
[numthreads(NT, 1, 1)]
void cs_binary(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint idx = linear_id(gid, tid);
    uint count = P(0);
    if (idx >= count)
    {
        return;
    }
    uint rank = P(1);
    uint rem = idx;
    uint off_a = 0;
    uint off_b = 0;
    [loop]
    for (int d = (int)rank - 1; d >= 0; --d)
    {
        uint dim = P(2 + d);
        uint coord = rem % dim;
        rem /= dim;
        off_a += coord * P(8 + d);
        off_b += coord * P(14 + d);
    }
    float a = f32_at(s0, off_a);
    float b = f32_at(s1, off_b);
    float r;
    uint op = P(20);
    if (op == BIN_ADD) { r = a + b; }
    else if (op == BIN_SUB) { r = a - b; }
    else if (op == BIN_MUL) { r = a * b; }
    else if (op == BIN_DIV) { r = a / b; }
    else if (op == BIN_POW) { r = real_pow(a, b); }
    else if (op == BIN_PRELU) { r = a >= 0.0 ? a : a * b; }
    else if (op == BIN_EQUAL) { r = a == b ? 1.0 : 0.0; }
    else if (op == BIN_GREATER) { r = a > b ? 1.0 : 0.0; }
    else if (op == BIN_LESS) { r = a < b ? 1.0 : 0.0; }
    else if (op == BIN_GREATER_EQUAL) { r = a >= b ? 1.0 : 0.0; }
    else if (op == BIN_AND) { r = (a != 0.0 && b != 0.0) ? 1.0 : 0.0; }
    else if (op == BIN_MIN) { r = min(a, b); }
    else { r = max(a, b); }
    u0.Store(idx * 4, asuint(r));
}

// Three-input Where: condition (s0), then-branch (s1), else-branch (s2) with the
// same broadcast strides as `cs_binary` and `P(7)` as the third operand's stride base.
// P(0) = count, P(1) = rank, P(2..7) dims, strides at 8, 14 and (22..27).
[numthreads(NT, 1, 1)]
void cs_where(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint idx = linear_id(gid, tid);
    uint count = P(0);
    if (idx >= count)
    {
        return;
    }
    uint rank = P(1);
    uint rem = idx;
    uint off_c = 0;
    uint off_t = 0;
    uint off_e = 0;
    [loop]
    for (int d = (int)rank - 1; d >= 0; --d)
    {
        uint dim = P(2 + d);
        uint coord = rem % dim;
        rem /= dim;
        off_c += coord * P(8 + d);
        off_t += coord * P(14 + d);
        off_e += coord * P(22 + d);
    }
    float r = f32_at(s0, off_c) != 0.0 ? f32_at(s1, off_t) : f32_at(s2, off_e);
    u0.Store(idx * 4, asuint(r));
}

// Unary opcodes (`P(20)`), scalar slots `P(21)`, `P(22)`.
#define UN_IDENTITY 0
#define UN_SQRT 1
#define UN_RECIP 2
#define UN_ABS 3
#define UN_SIN 4
#define UN_COS 5
#define UN_RELU 6
#define UN_SIGMOID 7
#define UN_ERF 8
#define UN_LEAKY 9
#define UN_HARD_SIGMOID 10
#define UN_HARD_SWISH 11
#define UN_CLIP 12
#define UN_NEG 13
#define UN_EXP 14
#define UN_TANH 15
#define UN_FLOOR 16
#define UN_CEIL 17
#define UN_LOG 18

// P(0) = element count, P(20) = opcode, P(21..22) = float scalars (bit-cast).
[numthreads(NT, 1, 1)]
void cs_unary(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint idx = linear_id(gid, tid);
    uint count = P(0);
    if (idx >= count)
    {
        return;
    }
    float x = f32_at(s0, idx);
    uint op = P(20);
    float alpha = asfloat(P(21));
    float beta = asfloat(P(22));
    float r;
    if (op == UN_SQRT) { r = sqrt(x); }
    else if (op == UN_RECIP) { r = 1.0 / x; }
    else if (op == UN_ABS) { r = abs(x); }
    else if (op == UN_SIN) { r = sin(x); }
    else if (op == UN_COS) { r = cos(x); }
    else if (op == UN_RELU) { r = max(x, 0.0); }
    else if (op == UN_SIGMOID) { r = sigmoid_f(x); }
    else if (op == UN_ERF) { r = erf_f(x); }
    else if (op == UN_LEAKY) { r = x >= 0.0 ? x : x * alpha; }
    else if (op == UN_HARD_SIGMOID) { r = min(max(x * alpha + beta, 0.0), 1.0); }
    else if (op == UN_HARD_SWISH) { r = x * min(max(x + 3.0, 0.0), 6.0) / 6.0; }
    else if (op == UN_CLIP) { r = min(max(x, alpha), beta); }
    else if (op == UN_NEG) { r = -x; }
    else if (op == UN_EXP) { r = exp(x); }
    else if (op == UN_TANH) { r = tanh(x); }
    else if (op == UN_FLOOR) { r = floor(x); }
    else if (op == UN_CEIL) { r = ceil(x); }
    else if (op == UN_LOG) { r = log(x); }
    else { r = x; }
    u0.Store(idx * 4, asuint(r));
}

// Scalar right operand (Sqrt/Reciprocal-free forms such as `x * 0.5`, Clip bounds
// already covered by cs_unary). P(0) = count, P(20) = opcode, P(21) = scalar.
[numthreads(NT, 1, 1)]
void cs_scalar(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint idx = linear_id(gid, tid);
    uint count = P(0);
    if (idx >= count)
    {
        return;
    }
    float x = f32_at(s0, idx);
    float k = asfloat(P(21));
    uint op = P(20);
    float r;
    if (op == BIN_ADD) { r = x + k; }
    else if (op == BIN_SUB) { r = x - k; }
    else if (op == BIN_MUL) { r = x * k; }
    else if (op == BIN_DIV) { r = x / k; }
    else if (op == BIN_POW) { r = real_pow(x, k); }
    else if (op == BIN_MIN) { r = min(x, k); }
    else { r = max(x, k); }
    u0.Store(idx * 4, asuint(r));
}

// ------------------------------------------------------------------- cast/type

// P(0) = count, P(20) = target: 0 = float32 (identity), 1 = float16 (round-trip
// through IEEE half), 2 = int8, 3 = uint8, 4 = bool, 5 = int16, 6 = uint16.
// All outputs are numeric f32 slots, including narrow integers. ONNX Cast
// truncates finite in-range floats toward zero; it is not QuantizeLinear.
[numthreads(NT, 1, 1)]
void cs_cast(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint idx = linear_id(gid, tid);
    uint count = P(0);
    if (idx >= count)
    {
        return;
    }
    float x = f32_at(s0, idx);
    uint mode = P(20);
    if (mode == 1)
    {
        u0.Store(idx * 4, asuint(f16tof32(f32_to_f16_rtne(x))));
    }
    else if (mode == 2 || mode == 3 || mode == 5 || mode == 6)
    {
        // Preserve low integer bits when narrowing, then sign-extend if needed.
        // A following float operator reads numeric 255.0, never the bit pattern 0xff.
        uint width = mode >= 5 ? 16u : 8u;
        uint mask = (1u << width) - 1u;
        uint lo = asuint((int)trunc(x)) & mask;
        bool signed_target = mode == 2 || mode == 5;
        int v = asint(signed_target && (lo & (1u << (width - 1u))) != 0 ? lo | ~mask : lo);
        u0.Store(idx * 4, asuint((float)v));
    }
    else if (mode == 4)
    {
        u0.Store(idx * 4, asuint(x != 0.0 ? 1.0 : 0.0));
    }
    else
    {
        u0.Store(idx * 4, asuint(x));
    }
}

// -------------------------------------------------------------- softmax/normal

// Softmax over the last axis. One 64-thread group per row.
// P(0) = rows, P(1) = row length (elements), P(2) = input row base offset.
[numthreads(NT, 1, 1)]
void cs_softmax_last(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uint lane : SV_GroupIndex)
{
    uint rows = P(0);
    uint len = P(1);
    uint row = group_x(gid);
    if (row >= rows)
    {
        return;
    }
    uint base = P(2) + row * len;
    float local_max = -3.402823466e+38;
    for (uint i = lane; i < len; i += NT)
    {
        local_max = max(local_max, f32_at(s0, base + i));
    }
    float row_max = group_max(local_max, lane);
    float local_sum = 0.0;
    for (uint j = lane; j < len; j += NT)
    {
        local_sum += exp(f32_at(s0, base + j) - row_max);
    }
    float row_sum = group_sum(local_sum, lane);
    float inv = 1.0 / row_sum;
    for (uint q = lane; q < len; q += NT)
    {
        u0.Store((base + q) * 4, asuint(exp(f32_at(s0, base + q) - row_max) * inv));
    }
}

// Row-wise LayerNormalization over the last axis. One group per row.
// P(0) = rows, P(1) = row length, P(2) = epsilon bits, P(3) = has gamma/beta,
// P(4) = gamma/beta element count (0 = no affine).
[numthreads(NT, 1, 1)]
void cs_layer_norm(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uint lane : SV_GroupIndex)
{
    uint rows = P(0);
    uint len = P(1);
    float eps = asfloat(P(2));
    uint affine = P(3);
    uint row = group_x(gid);
    if (row >= rows)
    {
        return;
    }
    uint base = row * len;
    float local_sum = 0.0;
    for (uint i = lane; i < len; i += NT)
    {
        local_sum += f32_at(s0, base + i);
    }
    float mean = group_sum(local_sum, lane) / (float)len;
    float local_var = 0.0;
    for (uint j = lane; j < len; j += NT)
    {
        float d = f32_at(s0, base + j) - mean;
        local_var += d * d;
    }
    float var = group_sum(local_var, lane) / (float)len;
    float inv = rsqrt(var + eps);
    for (uint q = lane; q < len; q += NT)
    {
        float y = (f32_at(s0, base + q) - mean) * inv;
        if (affine != 0)
        {
            y = y * f32_at(s1, q) + f32_at(s2, q);
        }
        u0.Store((base + q) * 4, asuint(y));
    }
}

// Per-channel affine (BatchNormFused): y = x * scale[c] + bias[c] for NCHW data.
// P(0) = element count, P(1) = spatial size per channel (H*W), P(2) = channel count.
[numthreads(NT, 1, 1)]
void cs_affine_channel(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint idx = linear_id(gid, tid);
    uint count = P(0);
    if (idx >= count)
    {
        return;
    }
    uint sp = P(1);
    uint channels = P(2);
    uint c = (idx / sp) % channels;
    float y = f32_at(s0, idx) * f32_at(s1, c) + f32_at(s2, c);
    u0.Store(idx * 4, asuint(y));
}

// InstanceNormalization: per (n, c) plane mean/variance, then per-channel
// gamma/beta. One group per plane, so the whole plane is normalized in-place in
// registers across a two-pass strided loop.
// P(0) = plane count, P(1) = plane length, P(2) = epsilon bits, P(3) = channel
// count, P(4) = has gamma/beta.
[numthreads(NT, 1, 1)]
void cs_instance_norm(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uint lane : SV_GroupIndex)
{
    uint planes = P(0);
    uint len = P(1);
    float eps = asfloat(P(2));
    uint channels = P(3);
    uint affine = P(4);
    uint plane = group_x(gid);
    if (plane >= planes)
    {
        return;
    }
    uint base = plane * len;
    float local_sum = 0.0;
    for (uint i = lane; i < len; i += NT)
    {
        local_sum += f32_at(s0, base + i);
    }
    float mean = group_sum(local_sum, lane) / (float)len;
    float local_var = 0.0;
    for (uint j = lane; j < len; j += NT)
    {
        float d = f32_at(s0, base + j) - mean;
        local_var += d * d;
    }
    float var = group_sum(local_var, lane) / (float)len;
    float inv = rsqrt(var + eps);
    uint channel = plane % channels;
    float scale = affine != 0 ? f32_at(s1, channel) : 1.0;
    float shift = affine != 0 ? f32_at(s2, channel) : 0.0;
    for (uint q = lane; q < len; q += NT)
    {
        float y = (f32_at(s0, base + q) - mean) * inv * scale + shift;
        u0.Store((base + q) * 4, asuint(y));
    }
}

// ------------------------------------------------------------------ reductions

// Partial sums for a global ReduceMean. Every 64-thread group reduces NT * 4
// elements into one float written at `u0[group_x(gid)]`.
// P(0) = element count, P(1) = source base offset.
[numthreads(NT, 1, 1)]
void cs_reduce_partial(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uint lane : SV_GroupIndex)
{
    uint count = P(0);
    uint base = P(1);
    float acc = 0.0;
    uint start = group_x(gid) * NT * 4 + lane;
    [unroll]
    for (uint t = 0; t < 4; ++t)
    {
        uint i = start + t * NT;
        if (i < count)
        {
            acc += f32_at(s0, base + i);
        }
    }
    float total = group_sum(acc, lane);
    if (lane == 0)
    {
        u0.Store(group_x(gid) * 4, asuint(total));
    }
}

// Finishes a global ReduceMean: sums `P(0)` partials and divides by `P(1)`.
[numthreads(NT, 1, 1)]
void cs_reduce_finish(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uint lane : SV_GroupIndex)
{
    uint partials = P(0);
    float inv = 1.0 / asfloat(P(1));
    float acc = 0.0;
    for (uint i = lane; i < partials; i += NT)
    {
        acc += f32_at(s0, i);
    }
    float total = group_sum(acc, lane);
    if (lane == 0)
    {
        u0.Store(0, asuint(total * inv));
    }
}

// GlobalAveragePool: one group per (n, c) plane reduced to its mean.
// P(0) = plane count, P(1) = plane length.
[numthreads(NT, 1, 1)]
void cs_global_avg_pool(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uint lane : SV_GroupIndex)
{
    uint planes = P(0);
    uint len = P(1);
    uint plane = group_x(gid);
    if (plane >= planes)
    {
        return;
    }
    uint base = plane * len;
    float acc = 0.0;
    for (uint i = lane; i < len; i += NT)
    {
        acc += f32_at(s0, base + i);
    }
    float total = group_sum(acc, lane);
    if (lane == 0)
    {
        u0.Store(plane * 4, asuint(total / (float)len));
    }
}

// Row-wise ReduceMean over the last axis: one group per row.
// P(0) = rows, P(1) = row length.
[numthreads(NT, 1, 1)]
void cs_reduce_rows(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uint lane : SV_GroupIndex)
{
    uint rows = P(0);
    uint len = P(1);
    uint row = group_x(gid);
    if (row >= rows)
    {
        return;
    }
    uint base = row * len;
    float acc = 0.0;
    for (uint i = lane; i < len; i += NT)
    {
        acc += f32_at(s0, base + i);
    }
    float total = group_sum(acc, lane);
    if (lane == 0)
    {
        u0.Store(row * 4, asuint(total / (float)len));
    }
}

// ----------------------------------------------------------------------- GEMM

// 16x16 tile, one output element per thread, shared-memory tiles for both
// operands with a +1 pad on the column to avoid LDS bank conflicts on RDNA.
#define GEMM_TILE 16

groupshared float g_tile_a[GEMM_TILE][GEMM_TILE + 1];
groupshared float g_tile_b[GEMM_TILE][GEMM_TILE + 1];

// Row-major C[M,N] = alpha * op(A)[M,K] * op(B)[K,N] + beta * C.
// P(0)=M P(1)=K P(2)=N P(3)=batch P(4)=lda P(5)=ldb P(6)=ldc
// P(7)=transA P(8)=transB P(9)=aBatch P(10)=bBatch P(11)=cBatch
// P(12)=alpha bits P(13)=beta bits P(14)=hasC
[numthreads(GEMM_TILE, GEMM_TILE, 1)]
void cs_matmul(uint3 gid : SV_GroupID, uint3 group_thread : SV_GroupThreadID)
{
    uint m = P(0);
    uint k = P(1);
    uint n = P(2);
    uint lda = P(4);
    uint ldb = P(5);
    uint ldc = P(6);
    uint trans_a = P(7);
    uint trans_b = P(8);
    uint batch = P(3);
    uint batch_index = gid.z;
    if (batch_index >= batch)
    {
        return;
    }
    uint a_base = batch_index * P(9);
    uint b_base = batch_index * P(10);
    uint c_base = batch_index * P(11);
    uint row = gid.y * GEMM_TILE + group_thread.y;
    uint col = gid.x * GEMM_TILE + group_thread.x;
    float acc = 0.0;
    uint tiles = (k + GEMM_TILE - 1) / GEMM_TILE;
    for (uint t = 0; t < tiles; ++t)
    {
        uint a_col = t * GEMM_TILE + group_thread.x;
        uint a_row = gid.y * GEMM_TILE + group_thread.y;
        float a_val = 0.0;
        if (a_row < m && a_col < k)
        {
            a_val = trans_a != 0 ? f32_at(s0, a_base + a_col * lda + a_row)
                                 : f32_at(s0, a_base + a_row * lda + a_col);
        }
        g_tile_a[group_thread.y][group_thread.x] = a_val;

        uint b_row = t * GEMM_TILE + group_thread.y;
        uint b_col = gid.x * GEMM_TILE + group_thread.x;
        float b_val = 0.0;
        if (b_row < k && b_col < n)
        {
            b_val = trans_b != 0 ? f32_at(s1, b_base + b_col * ldb + b_row)
                                 : f32_at(s1, b_base + b_row * ldb + b_col);
        }
        g_tile_b[group_thread.y][group_thread.x] = b_val;
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint i = 0; i < GEMM_TILE; ++i)
        {
            acc += g_tile_a[group_thread.y][i] * g_tile_b[i][group_thread.x];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < m && col < n)
    {
        float alpha = asfloat(P(12));
        float beta = asfloat(P(13));
        float out_val = alpha * acc;
        if (P(14) != 0)
        {
            out_val += beta * f32_at(s2, c_base + row * ldc + col);
        }
        u0.Store((c_base + row * ldc + col) * 4, asuint(out_val));
    }
}

// ------------------------------------------------------------------ convolution

// Direct NCHW convolution, one thread per output element. Correct for arbitrary
// kernel/stride/pad/dilation/groups; the tiled/GEMM path is a later optimization.
// P(0)=total P(1)=n P(2)=cin P(3)=h P(4)=w P(5)=cout P(6)=kh P(7)=kw
// P(8)=sh P(9)=sw P(10)=ph P(11)=pw P(12)=dh P(13)=dw P(14)=groups
// P(15)=cin_per_group P(16)=h_out P(17)=w_out P(18)=x_base P(19)=w_base
// P(20)=bias_base P(21)=has_bias P(22)=activation (0 none, 1 relu, 2 silu)
[numthreads(NT, 1, 1)]
void cs_conv2d(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint idx = linear_id(gid, tid);
    if (idx >= P(0))
    {
        return;
    }
    uint w_out = P(17);
    uint h_out = P(16);
    uint cout = P(5);
    uint ow = idx % w_out;
    uint rem = idx / w_out;
    uint oh = rem % h_out;
    rem /= h_out;
    uint oc = rem % cout;
    uint batch = rem / cout;

    uint cin_per_group = P(15);
    uint groups = P(14);
    uint group = cout > 0 ? oc / (cout / groups) : 0;
    uint ic_start = group * cin_per_group;

    uint kh = P(6);
    uint kw = P(7);
    uint sh = P(8);
    uint sw = P(9);
    int ph = (int)P(10);
    int pw = (int)P(11);
    uint dh = P(12);
    uint dw = P(13);
    uint h = P(3);
    uint w = P(4);
    uint cin = P(2);

    float acc = P(21) != 0 ? f32_at(s2, P(20) + oc) : 0.0;
    uint weight_plane = cin_per_group * kh * kw;
    uint weight_channel_base = oc * weight_plane;
    for (uint ic = 0; ic < cin_per_group; ++ic)
    {
        uint src_c = ic_start + ic;
        if (src_c >= cin)
        {
            break;
        }
        uint x_plane = (batch * cin + src_c) * h * w;
        uint w_plane = P(19) + weight_channel_base + ic * kh * kw;
        for (uint ky = 0; ky < kh; ++ky)
        {
            int iy = (int)(oh * sh) - ph + (int)(ky * dh);
            if (iy < 0 || iy >= (int)h)
            {
                continue;
            }
            for (uint kx = 0; kx < kw; ++kx)
            {
                int ix = (int)(ow * sw) - pw + (int)(kx * dw);
                if (ix < 0 || ix >= (int)w)
                {
                    continue;
                }
                acc += f32_at(s0, P(18) + x_plane + (uint)iy * w + (uint)ix)
                     * f32_at(s1, w_plane + ky * kw + kx);
            }
        }
    }
    uint act = P(22);
    if (act == 1)
    {
        acc = max(acc, 0.0);
    }
    else if (act == 2)
    {
        acc = acc * sigmoid_f(acc);
    }
    u0.Store(idx * 4, asuint(acc));
}

// MaxPool / AveragePool, NCHW, one thread per output element. Count-include-pad
// is off, so the average divides by the number of in-bounds samples.
// P(0)=total P(1)=channels P(2)=h P(3)=w P(4)=kh P(5)=kw P(6)=sh P(7)=sw
// P(8)=ph P(9)=pw P(10)=h_out P(11)=w_out P(12)=is_average P(13)=x_base
[numthreads(NT, 1, 1)]
void cs_pool(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint idx = linear_id(gid, tid);
    if (idx >= P(0))
    {
        return;
    }
    uint w_out = P(11);
    uint h_out = P(10);
    uint channels = P(1);
    uint ow = idx % w_out;
    uint rem = idx / w_out;
    uint oh = rem % h_out;
    rem /= h_out;
    uint c = rem % channels;
    uint batch = rem / channels;

    uint kh = P(4);
    uint kw = P(5);
    uint sh = P(6);
    uint sw = P(7);
    int ph = (int)P(8);
    int pw = (int)P(9);
    uint h = P(2);
    uint w = P(3);
    uint plane = (batch * channels + c) * h * w;

    float acc = P(12) != 0 ? 0.0 : -3.402823466e+38;
    uint seen = 0;
    for (uint ky = 0; ky < kh; ++ky)
    {
        int iy = (int)(oh * sh) - ph + (int)ky;
        if (iy < 0 || iy >= (int)h)
        {
            continue;
        }
        for (uint kx = 0; kx < kw; ++kx)
        {
            int ix = (int)(ow * sw) - pw + (int)kx;
            if (ix < 0 || ix >= (int)w)
            {
                continue;
            }
            float v = f32_at(s0, P(13) + plane + (uint)iy * w + (uint)ix);
            acc = P(12) != 0 ? acc + v : max(acc, v);
            seen += 1;
        }
    }
    if (P(12) != 0)
    {
        acc = seen > 0 ? acc / (float)seen : 0.0;
    }
    u0.Store(idx * 4, asuint(acc));
}

// -------------------------------------------------------------- data movement

// General permutation copy. P(0)=count P(1)=rank P(2..7)=output dims
// P(8..13)=input strides (elements) P(14)=input base offset.
[numthreads(NT, 1, 1)]
void cs_transpose(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint idx = linear_id(gid, tid);
    if (idx >= P(0))
    {
        return;
    }
    uint rank = P(1);
    uint rem = idx;
    uint src = P(14);
    [loop]
    for (int d = (int)rank - 1; d >= 0; --d)
    {
        uint dim = P(2 + d);
        uint coord = rem % dim;
        rem /= dim;
        src += coord * P(8 + d);
    }
    u0.Store(idx * 4, s0.Load(src * 4));
}

// Concat: copies one input into its slot of the output along `axis`.
// P(0)=input element count P(1)=output axis length P(2)=axis start offset
// P(3)=input axis length P(4)=inner size P(5)=source base offset
// P(6)=destination base offset
[numthreads(NT, 1, 1)]
void cs_concat_copy(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint idx = linear_id(gid, tid);
    uint count = P(0);
    if (idx >= count)
    {
        return;
    }
    uint axis_out = P(1);
    uint start = P(2);
    uint axis_in = P(3);
    uint inner = P(4);
    uint outer_span = axis_in * inner;
    uint outer = idx / outer_span;
    uint rest = idx - outer * outer_span;
    uint a = rest / inner;
    uint i = rest - a * inner;
    uint dst = P(6) + (outer * axis_out + start + a) * inner + i;
    u0.Store(dst * 4, s0.Load((P(5) + idx) * 4));
}

// Static Pad, `constant` mode only: the interior copies the source, the border
// reads the fill value. A negative begin/end crop is the same arithmetic.
// P(0)=output element count P(1)=rank P(2..7)=output dims P(8..13)=input dims
// P(14..19)=input strides (elements) P(20..25)=begin pads (signed)
// P(26)=fill value bits.
[numthreads(NT, 1, 1)]
void cs_pad(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint idx = linear_id(gid, tid);
    if (idx >= P(0))
    {
        return;
    }
    uint rem = idx;
    int src = 0;
    bool interior = true;
    [loop]
    for (int d = (int)P(1) - 1; d >= 0; --d)
    {
        uint odim = P(2 + d);
        uint coord = rem % odim;
        rem /= odim;
        int idim = (int)P(8 + d);
        int s = (int)coord - asint(P(20 + d));
        if (s < 0 || s >= idim)
        {
            interior = false;
        }
        src += s * (int)P(14 + d);
    }
    float v = asfloat(P(26));
    if (interior)
    {
        // The index is only formed when it is inside the source, so the load
        // below can never be out of range.
        v = asfloat(s0.Load((uint)src * 4));
    }
    u0.Store(idx * 4, asuint(v));
}

// Static Slice: `output[coord] = input[start + coord * step]` per axis, with
// ONNX-normalised starts (negative starts already shifted, negative steps
// already oriented).
// P(0)=output element count P(1)=rank P(2..7)=output dims
// P(8..13)=input strides (elements) P(14..19)=starts (signed)
// P(20..25)=steps (signed, never 0).
[numthreads(NT, 1, 1)]
void cs_slice(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint idx = linear_id(gid, tid);
    if (idx >= P(0))
    {
        return;
    }
    uint rem = idx;
    int src = 0;
    [loop]
    for (int d = (int)P(1) - 1; d >= 0; --d)
    {
        uint odim = P(2 + d);
        uint coord = rem % odim;
        rem /= odim;
        int pos = asint(P(14 + d)) + (int)coord * asint(P(20 + d));
        src += pos * (int)P(8 + d);
    }
    u0.Store(idx * 4, s0.Load((uint)src * 4));
}

// Plain float copy between two buffers (Reshape with a real repack, Expand, Tile
// row repeat). P(0)=count P(1)=source base P(2)=destination base.
[numthreads(NT, 1, 1)]
void cs_copy(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint idx = linear_id(gid, tid);
    if (idx >= P(0))
    {
        return;
    }
    u0.Store((P(2) + idx) * 4, s0.Load((P(1) + idx) * 4));
}

// Expand/Tile along the last axis: repeats a source row of `P(1)` elements
// `P(2)` times into a destination row of `P(1) * P(2)` elements.
// P(0)=output element count P(1)=axis length P(3)=output axis length
[numthreads(NT, 1, 1)]
void cs_repeat_axis(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint idx = linear_id(gid, tid);
    if (idx >= P(0))
    {
        return;
    }
    uint axis_in = P(1);
    uint axis_out = P(3);
    uint outer = idx / axis_out;
    uint within = idx - outer * axis_out;
    uint src = outer * axis_in + (within % axis_in);
    u0.Store(idx * 4, s0.Load(src * 4));
}
