#include <metal_stdlib>
using namespace metal;

struct CopyParams {
    uint rank;
    uint out_elems;
    uint elem_bytes;
    uint edge;
    int  dst_base;
    int  src_base;
    float pad_value;
    uint out_shape[8];
    int  dst_stride[8];
    int  mul[8];
    int  add[8];
    int  shift[8];
    int  in_dim[8];
    uint use_mod[8];
};

struct FillParams { uint n; uint elem_bytes; int dst_base; uint bits; };

struct UnaryParams {
    uint n; uint op; float alpha; float beta; float lo; float hi; int x_base; int y_base;
};

struct BinaryParams {
    uint rank; uint n; uint op; uint unused;
    int a_base; int b_base; int y_base;
    uint out_shape[8]; uint a_mul[8]; uint b_mul[8];
};

struct WhereParams {
    uint rank; uint n; uint unused0; uint unused1;
    int c_base; int a_base; int b_base; int y_base;
    uint out_shape[8]; uint c_mul[8]; uint a_mul[8]; uint b_mul[8];
};

struct CastParams { uint n; uint from; uint to; uint unused; int x_base; int y_base; };

struct ChannelAffineParams {
    uint n; uint inner; uint channels; uint unused;
    int x_base; int y_base; int scale_base; int bias_base;
};

struct GemmParams {
    uint m; uint n; uint k;
    int lda; int ldb; int ldc;
    int a_off; int b_off; int c_off;
    float alpha; float beta;
    uint trans_a; uint trans_b; uint has_bias; uint act;
};

struct ConvParams {
    uint n; uint h; uint w; uint c; uint oh; uint ow; uint oc;
    uint kh; uint kw; uint sh; uint sw; uint ph; uint pw; uint dh; uint dw;
    uint groups; uint act;
    int x_off; int w_off; int b_off; int y_off;
};

struct PoolParams {
    uint n; uint c; uint ih; uint iw; uint oh; uint ow;
    uint kh; uint kw; uint sh; uint sw; uint ph; uint pw; uint dh; uint dw;
    uint mode; uint count_include_pad;
    int x_off; int y_off;
};

struct RowParams {
    uint outer; uint r; uint inner; uint mode; uint channels;
    uint y_stride_outer; uint y_stride_inner; uint unused;
    int x_off; int y_off; int g_off; int b_off;
    float eps;
};

struct ResizeParams {
    uint n; uint c; uint ih; uint iw; uint oh; uint ow;
    uint mode; uint coord_mode; uint nearest_mode;
    float rh; float rw; int x_off; int y_off;
};

struct MatMulIntParams {
    uint m; uint n; uint k;
    int a_off; int b_off; int c_off; int azp_off; int bzp_off;
    uint a_zp; uint b_zp; uint use_buf_zp; uint b_signed;
};

struct QuantizeParams { uint n; int x_off; int y_off; int scale_off; int zp_off; };

struct QuantFinalizeParams { int minmax_off; int scale_off; int zp_off; float eps; };

struct ScatterParams {
    uint num_updates; uint k; uint slice_size; uint out_elems; uint rank;
    int idx_off; int upd_off; int out_off;
    int strides[8];
};

struct GatherParams {
    uint outer; uint axis_len; uint inner; uint idx_len; uint out_elems; uint idx_i64;
    uint unused0; uint unused1;
    int data_off; int idx_off; int y_off;
};

struct MinMaxParams { uint n; int x_off; };

// --- helpers ---------------------------------------------------------------

static inline int reflect_position(int pos, int limit) {
    if (limit <= 1) { return 0; }
    int period = 2 * (limit - 1);
    int m = pos % period;
    if (m < 0) { m += period; }
    return (m < limit) ? m : (period - m);
}

static inline int nearest_round(float value, uint mode) {
    switch (mode) {
        case 1: { float f = floor(value); return (int)((value - f > 0.5f) ? f + 1.0f : f); }
        case 2: { float f = floor(value); return (int)((value - f >= 0.5f) ? f + 1.0f : f); }
        case 3: { return (int)ceil(value); }
        default: { return (int)floor(value); }
    }
}

static inline float load_typed(device const uchar* base, int index, uint code) {
    switch (code) {
        case 0: return *(device const float*)(base + index * 4);
        case 1: return (float)*(device const half*)(base + index * 2);
        case 2: return (float)*(device const int*)(base + index * 4);
        case 3: return (float)*(device const long*)(base + index * 8);
        case 4: return (float)*(device const uchar*)(base + index);
        case 5: return (float)*(device const char*)(base + index);
        default: return (*(device const uchar*)(base + index) != 0) ? 1.0f : 0.0f;
    }
}

static inline void store_typed(device uchar* base, int index, uint code, float value) {
    switch (code) {
        case 0: *(device float*)(base + index * 4) = value; break;
        case 1: *(device half*)(base + index * 2) = (half)value; break;
        case 2: *(device int*)(base + index * 4) = (int)value; break;
        case 3: *(device long*)(base + index * 8) = (long)value; break;
        case 4: *(device uchar*)(base + index) = (uchar)clamp(value, 0.0f, 255.0f); break;
        case 5: *(device char*)(base + index) = (char)clamp(value, -128.0f, 127.0f); break;
        default: *(device uchar*)(base + index) = (value != 0.0f) ? 1 : 0; break;
    }
}

// --- kernels ---------------------------------------------------------------

kernel void k_copy(device uchar* dst [[buffer(0)]],
                   device const uchar* src [[buffer(1)]],
                   constant CopyParams& p [[buffer(9)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid >= p.out_elems) { return; }
    uint rem = gid;
    int src_index = p.src_base;
    int dst_index = p.dst_base;
    bool in_range = true;
    for (int d = (int)p.rank - 1; d >= 0; --d) {
        uint dim = p.out_shape[d];
        uint idx = (dim > 0) ? (rem % dim) : 0u;
        if (dim > 0) { rem /= dim; }
        dst_index += (int)idx * p.dst_stride[d];
        int pos = (p.use_mod[d] > 0) ? (int)(idx % p.use_mod[d]) : (int)idx;
        pos += p.shift[d];
        int limit = p.in_dim[d];
        if (pos < 0 || pos >= limit) {
            if (p.edge == 1) { pos = clamp(pos, 0, max(limit - 1, 0)); }
            else if (p.edge == 2) { pos = reflect_position(pos, limit); }
            else { in_range = false; }
        }
        src_index += pos * p.mul[d] + p.add[d];
    }
    if (!in_range) {
        if (p.elem_bytes == 4) { *(device float*)(dst + dst_index * 4) = p.pad_value; }
        else { dst[dst_index] = (uchar)0; }
        return;
    }
    if (p.elem_bytes == 4) { *(device float*)(dst + dst_index * 4) = *(device const float*)(src + src_index * 4); }
    else if (p.elem_bytes == 2) { *(device half*)(dst + dst_index * 2) = *(device const half*)(src + src_index * 2); }
    else { dst[dst_index] = src[src_index]; }
}

kernel void k_fill(device uchar* dst [[buffer(0)]],
                   constant FillParams& p [[buffer(9)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) { return; }
    int index = p.dst_base + (int)gid;
    if (p.elem_bytes == 4) { *(device float*)(dst + index * 4) = as_type<float>(p.bits); }
    else if (p.elem_bytes == 2) { *(device half*)(dst + index * 2) = as_type<half>((ushort)p.bits); }
    else { dst[index] = (uchar)p.bits; }
}

kernel void k_unary(device const float* x [[buffer(0)]],
                    device float* y [[buffer(1)]],
                    constant UnaryParams& p [[buffer(9)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) { return; }
    float v = x[p.x_base + (int)gid];
    float r = v;
    switch (p.op) {
        case 0: r = max(v, 0.0f); break;
        case 1: r = 1.0f / (1.0f + exp(-v)); break;
        case 2: r = sqrt(v); break;
        case 3: r = fabs(v); break;
        case 4: r = sin(v); break;
        case 5: r = cos(v); break;
        case 6: r = 1.0f / v; break;
        case 7: r = erf(v); break;
        case 8: r = (v >= 0.0f) ? v : v * p.alpha; break;
        case 9: r = v * clamp(v + 3.0f, 0.0f, 6.0f) / 6.0f; break;
        case 10: r = clamp(p.alpha * v + p.beta, 0.0f, 1.0f); break;
        default: r = clamp(v, p.lo, p.hi); break;
    }
    y[p.y_base + (int)gid] = r;
}

static inline void flat_index(uint gid, constant BinaryParams& p, thread uint& a, thread uint& b) {
    uint rem = gid;
    a = 0u; b = 0u;
    for (int d = (int)p.rank - 1; d >= 0; --d) {
        uint dim = p.out_shape[d];
        uint idx = (dim > 0) ? (rem % dim) : 0u;
        if (dim > 0) { rem /= dim; }
        a += idx * p.a_mul[d];
        b += idx * p.b_mul[d];
    }
}

kernel void k_binary(device const float* a [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     device float* y [[buffer(2)]],
                     constant BinaryParams& p [[buffer(9)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) { return; }
    uint ai = 0u; uint bi = 0u;
    flat_index(gid, p, ai, bi);
    float av = a[p.a_base + (int)ai];
    float bv = b[p.b_base + (int)bi];
    float r = av;
    switch (p.op) {
        case 0: r = av + bv; break;
        case 1: r = av - bv; break;
        case 2: r = av * bv; break;
        case 3: r = av / bv; break;
        case 4: r = pow(av, bv); break;
        default: r = (av >= 0.0f) ? av : av * bv; break;
    }
    y[p.y_base + (int)gid] = r;
}

kernel void k_compare(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device uchar* y [[buffer(2)]],
                      constant BinaryParams& p [[buffer(9)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) { return; }
    uint ai = 0u; uint bi = 0u;
    flat_index(gid, p, ai, bi);
    float av = a[p.a_base + (int)ai];
    float bv = b[p.b_base + (int)bi];
    bool r = false;
    switch (p.op) {
        case 0: r = (av == bv); break;
        case 1: r = (av > bv); break;
        case 2: r = (av >= bv); break;
        default: r = (av < bv); break;
    }
    y[p.y_base + (int)gid] = r ? 1 : 0;
}

kernel void k_and(device const uchar* a [[buffer(0)]],
                  device const uchar* b [[buffer(1)]],
                  device uchar* y [[buffer(2)]],
                  constant BinaryParams& p [[buffer(9)]],
                  uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) { return; }
    uint ai = 0u; uint bi = 0u;
    flat_index(gid, p, ai, bi);
    bool av = a[p.a_base + (int)ai] != 0;
    bool bv = b[p.b_base + (int)bi] != 0;
    y[p.y_base + (int)gid] = (av && bv) ? 1 : 0;
}

kernel void k_where(device const uchar* c [[buffer(0)]],
                    device const float* a [[buffer(1)]],
                    device const float* b [[buffer(2)]],
                    device float* y [[buffer(3)]],
                    constant WhereParams& p [[buffer(9)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) { return; }
    uint rem = gid;
    uint ci = 0u; uint ai = 0u; uint bi = 0u;
    for (int d = (int)p.rank - 1; d >= 0; --d) {
        uint dim = p.out_shape[d];
        uint idx = (dim > 0) ? (rem % dim) : 0u;
        if (dim > 0) { rem /= dim; }
        ci += idx * p.c_mul[d];
        ai += idx * p.a_mul[d];
        bi += idx * p.b_mul[d];
    }
    bool cond = c[p.c_base + (int)ci] != 0;
    y[p.y_base + (int)gid] = cond ? a[p.a_base + (int)ai] : b[p.b_base + (int)bi];
}

kernel void k_cast(device const uchar* x [[buffer(0)]],
                   device uchar* y [[buffer(1)]],
                   constant CastParams& p [[buffer(9)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) { return; }
    float v = load_typed(x, p.x_base + (int)gid, p.from);
    store_typed(y, p.y_base + (int)gid, p.to, v);
}

kernel void k_channel_affine(device const float* x [[buffer(0)]],
                             device const float* scale [[buffer(1)]],
                             device const float* bias [[buffer(2)]],
                             device float* y [[buffer(3)]],
                             constant ChannelAffineParams& p [[buffer(9)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) { return; }
    uint c = ((uint)gid / p.inner) % p.channels;
    y[p.y_base + (int)gid] = x[p.x_base + (int)gid] * scale[p.scale_base + (int)c]
                           + bias[p.bias_base + (int)c];
}

static inline float gemm_load_a(device const float* a, constant GemmParams& p, uint row, uint col) {
    int index = p.a_off + (p.trans_a != 0 ? (int)(col * p.lda + row) : (int)(row * p.lda + col));
    return a[index];
}

static inline float gemm_load_b(device const float* b, constant GemmParams& p, uint row, uint col) {
    int index = p.b_off + (p.trans_b != 0 ? (int)(col * p.ldb + row) : (int)(row * p.ldb + col));
    return b[index];
}

kernel void k_gemm(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device const float* bias [[buffer(2)]],
                   device float* c [[buffer(3)]],
                   constant GemmParams& p [[buffer(9)]],
                   uint2 gid [[thread_position_in_grid]],
                   uint2 tid [[thread_position_in_threadgroup]]) {
    const uint TILE = 16u;
    threadgroup float tile_a[16][17];
    threadgroup float tile_b[16][17];
    uint row = gid.y;
    uint col = gid.x;
    bool active = (row < p.m) && (col < p.n);
    float acc = 0.0f;
    for (uint k0 = 0u; k0 < p.k; k0 += TILE) {
        uint a_col = k0 + tid.x;
        uint a_row = row;
        tile_a[tid.y][tid.x] = (a_row < p.m && a_col < p.k) ? gemm_load_a(a, p, a_row, a_col) : 0.0f;
        uint b_row = k0 + tid.y;
        uint b_col = col;
        tile_b[tid.y][tid.x] = (b_row < p.k && b_col < p.n) ? gemm_load_b(b, p, b_row, b_col) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < TILE; ++kk) {
            acc += tile_a[tid.y][kk] * tile_b[kk][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!active) { return; }
    float value = p.alpha * acc;
    if (p.has_bias != 0) { value += bias[p.c_off + (int)col]; }
    if (p.beta != 0.0f) { value += p.beta * c[p.c_off + (int)(row * p.ldc + col)]; }
    if (p.act == 1) { value = max(value, 0.0f); }
    else if (p.act == 2) { value = value / (1.0f + exp(-value)); }
    c[p.c_off + (int)(row * p.ldc + col)] = value;
}

static inline float apply_activation(float v, uint act) {
    if (act == 1) { return max(v, 0.0f); }
    if (act == 2) { return v / (1.0f + exp(-v)); }
    return v;
}

kernel void k_conv2d(device const float* x [[buffer(0)]],
                     device const float* w [[buffer(1)]],
                     device const float* bias [[buffer(2)]],
                     device float* y [[buffer(3)]],
                     constant ConvParams& p [[buffer(9)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint ow = gid.x; uint oh = gid.y; uint oz = gid.z;
    if (ow >= p.ow || oh >= p.oh || oz >= p.oc * p.n) { return; }
    uint batch = oz / p.oc;
    uint oc = oz % p.oc;
    uint groups = max(p.groups, 1u);
    uint oc_per_group = p.oc / groups;
    uint ic_per_group = p.c / groups;
    uint group = oc / oc_per_group;
    float acc = 0.0f;
    for (uint ci = 0u; ci < ic_per_group; ++ci) {
        uint c = group * ic_per_group + ci;
        for (uint ky = 0u; ky < p.kh; ++ky) {
            int ih = (int)(oh * p.sh) - (int)p.ph + (int)(ky * p.dh);
            if (ih < 0 || ih >= (int)p.h) { continue; }
            for (uint kx = 0u; kx < p.kw; ++kx) {
                int iw = (int)(ow * p.sw) - (int)p.pw + (int)(kx * p.dw);
                if (iw < 0 || iw >= (int)p.w) { continue; }
                float xv = x[p.x_off + (int)(((batch * p.c + c) * p.h + (uint)ih) * p.w + (uint)iw)];
                float wv = w[p.w_off + (int)(((oc * ic_per_group + ci) * p.kh + ky) * p.kw + kx)];
                acc += xv * wv;
            }
        }
    }
    acc += bias[p.b_off + (int)oc];
    acc = apply_activation(acc, p.act);
    y[p.y_off + (int)(((batch * p.oc + oc) * p.oh + oh) * p.ow + ow)] = acc;
}

kernel void k_conv_transpose2d(device const float* x [[buffer(0)]],
                               device const float* w [[buffer(1)]],
                               device const float* bias [[buffer(2)]],
                               device float* y [[buffer(3)]],
                               constant ConvParams& p [[buffer(9)]],
                               uint3 gid [[thread_position_in_grid]]) {
    uint ow = gid.x; uint oh = gid.y; uint oz = gid.z;
    if (ow >= p.ow || oh >= p.oh || oz >= p.oc * p.n) { return; }
    uint batch = oz / p.oc;
    uint oc = oz % p.oc;
    uint groups = max(p.groups, 1u);
    uint oc_per_group = p.oc / groups;
    uint ic_per_group = max(p.c / groups, 1u);
    uint group = oc / oc_per_group;
    float acc = 0.0f;
    for (uint gi = 0u; gi < ic_per_group; ++gi) {
        uint ci = group * ic_per_group + gi;
        for (uint ky = 0u; ky < p.kh; ++ky) {
            int t = (int)(oh + p.ph) - (int)(ky * p.dh);
            if (t < 0 || (t % (int)p.sh) != 0) { continue; }
            int ih = t / (int)p.sh;
            if (ih < 0 || ih >= (int)p.h) { continue; }
            for (uint kx = 0u; kx < p.kw; ++kx) {
                int s = (int)(ow + p.pw) - (int)(kx * p.dw);
                if (s < 0 || (s % (int)p.sw) != 0) { continue; }
                int iw = s / (int)p.sw;
                if (iw < 0 || iw >= (int)p.w) { continue; }
                float xv = x[p.x_off + (int)(((batch * p.c + ci) * p.h + (uint)ih) * p.w + (uint)iw)];
                float wv = w[p.w_off + (int)(((ci * oc_per_group + oc) * p.kh + ky) * p.kw + kx)];
                acc += xv * wv;
            }
        }
    }
    acc += bias[p.b_off + (int)oc];
    acc = apply_activation(acc, p.act);
    y[p.y_off + (int)(((batch * p.oc + oc) * p.oh + oh) * p.ow + ow)] = acc;
}

kernel void k_pool2d(device const float* x [[buffer(0)]],
                     device float* y [[buffer(1)]],
                     constant PoolParams& p [[buffer(9)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint ow = gid.x; uint oh = gid.y; uint plane = gid.z;
    if (ow >= p.ow || oh >= p.oh || plane >= p.c * p.n) { return; }
    float acc = (p.mode == 0u) ? -INFINITY : 0.0f;
    float count = 0.0f;
    for (uint ky = 0u; ky < p.kh; ++ky) {
        int ih = (int)(oh * p.sh) - (int)p.ph + (int)(ky * p.dh);
        if (ih < 0 || ih >= (int)p.ih) { continue; }
        for (uint kx = 0u; kx < p.kw; ++kx) {
            int iw = (int)(ow * p.sw) - (int)p.pw + (int)(kx * p.dw);
            if (iw < 0 || iw >= (int)p.iw) { continue; }
            float v = x[p.x_off + (int)(((plane * p.ih + (uint)ih) * p.iw + (uint)iw))];
            if (p.mode == 0u) { acc = max(acc, v); } else { acc += v; count += 1.0f; }
        }
    }
    if (p.mode != 0u) {
        if (p.count_include_pad != 0) { acc /= (float)(p.kh * p.kw); }
        else { acc = (count > 0.0f) ? (acc / count) : 0.0f; }
    }
    y[p.y_off + (int)(((plane * p.oh + oh) * p.ow + ow))] = acc;
}

static inline float row_weight(device const float* table, int base, uint index, bool present) {
    return present ? table[base + (int)index] : 1.0f;
}

kernel void k_rows(device const float* x [[buffer(0)]],
                   device const float* gamma [[buffer(1)]],
                   device const float* beta [[buffer(2)]],
                   device float* y [[buffer(3)]],
                   constant RowParams& p [[buffer(9)]],
                   uint gid [[thread_position_in_grid]]) {
    uint rows = p.outer * p.inner;
    if (gid >= rows) { return; }
    uint o = gid / p.inner;
    uint i = gid % p.inner;
    uint base = (uint)p.x_off + (o * p.r) * p.inner + i;
    float scale_value = 1.0f;
    float bias_value = 0.0f;
    if (p.mode == 1u) { scale_value = 0.0f; /* per-element tables below */ }
    if (p.mode == 2u) {
        uint c = o % p.channels;
        scale_value = row_weight(gamma, p.g_off, c, p.g_off >= 0);
        bias_value = row_weight(beta, p.b_off, c, p.b_off >= 0);
    }
    if (p.mode == 0u) {
        float maximum = -INFINITY;
        for (uint r = 0u; r < p.r; ++r) { maximum = max(maximum, x[base + r * p.inner]); }
        float sum = 0.0f;
        for (uint r = 0u; r < p.r; ++r) { sum += exp(x[base + r * p.inner] - maximum); }
        float inv = 1.0f / (sum > 0.0f ? sum : 1.0f);
        for (uint r = 0u; r < p.r; ++r) {
            uint index = (uint)p.y_off + o * p.y_stride_outer + r * p.y_stride_inner + i;
            y[index] = exp(x[base + r * p.inner] - maximum) * inv;
        }
        return;
    }
    if (p.mode == 1u) {
        float sum = 0.0f;
        for (uint r = 0u; r < p.r; ++r) { sum += x[base + r * p.inner]; }
        float mean = sum / (float)p.r;
        float sq = 0.0f;
        for (uint r = 0u; r < p.r; ++r) {
            float d = x[base + r * p.inner] - mean;
            sq += d * d;
        }
        float inv = rsqrt(sq / (float)p.r + p.eps);
        for (uint r = 0u; r < p.r; ++r) {
            uint index = (uint)p.y_off + o * p.y_stride_outer + r * p.y_stride_inner + i;
            float normalised = (x[base + r * p.inner] - mean) * inv;
            float g = (p.g_off >= 0) ? gamma[p.g_off + (int)(r * p.inner + i)] : 1.0f;
            float b = (p.b_off >= 0) ? beta[p.b_off + (int)(r * p.inner + i)] : 0.0f;
            y[index] = normalised * g + b;
        }
        return;
    }
    if (p.mode == 2u) {
        float sum = 0.0f;
        for (uint r = 0u; r < p.r; ++r) { sum += x[base + r * p.inner]; }
        float mean = sum / (float)p.r;
        float sq = 0.0f;
        for (uint r = 0u; r < p.r; ++r) {
            float d = x[base + r * p.inner] - mean;
            sq += d * d;
        }
        float inv = rsqrt(sq / (float)p.r + p.eps);
        for (uint r = 0u; r < p.r; ++r) {
            uint index = (uint)p.y_off + o * p.y_stride_outer + r * p.y_stride_inner + i;
            y[index] = (x[base + r * p.inner] - mean) * inv * scale_value + bias_value;
        }
        return;
    }
    float sum = 0.0f;
    for (uint r = 0u; r < p.r; ++r) { sum += x[base + r * p.inner]; }
    y[(uint)p.y_off + o * p.y_stride_outer + i] = sum / (float)p.r;
}

static inline float resize_coord(int dst, int in_size, int out_size, uint coord_mode, float scale) {
    float src = 0.0f;
    switch (coord_mode) {
        case 1: src = (float)dst / scale; break;
        case 2: src = (out_size > 1) ? ((float)dst * (float)(in_size - 1) / (float)(out_size - 1)) : 0.0f; break;
        case 3: src = ((float)dst + 0.5f) / scale; break;
        case 4: src = (out_size > 1) ? (((float)dst + 0.5f) / scale - 0.5f) : 0.0f; break;
        default: src = ((float)dst + 0.5f) / scale - 0.5f; break;
    }
    return src;
}

kernel void k_resize(device const float* x [[buffer(0)]],
                     device float* y [[buffer(1)]],
                     constant ResizeParams& p [[buffer(9)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint ow = gid.x; uint oh = gid.y; uint plane = gid.z;
    if (ow >= p.ow || oh >= p.oh || plane >= p.c * p.n) { return; }
    uint spatial_in = p.ih * p.iw;
    uint plane_base = (uint)p.x_off + plane * spatial_in;
    uint out_index = (uint)p.y_off + (plane * p.oh + oh) * p.ow + ow;
    if (p.mode == 0u) {
        float sy = resize_coord((int)oh, (int)p.ih, (int)p.oh, p.coord_mode, p.rh);
        float sx = resize_coord((int)ow, (int)p.iw, (int)p.ow, p.coord_mode, p.rw);
        int iy = clamp(nearest_round(sy, p.nearest_mode), 0, (int)p.ih - 1);
        int ix = clamp(nearest_round(sx, p.nearest_mode), 0, (int)p.iw - 1);
        y[out_index] = x[plane_base + (uint)(iy * (int)p.iw + ix)];
        return;
    }
    float sy = resize_coord((int)oh, (int)p.ih, (int)p.oh, p.coord_mode, p.rh);
    float sx = resize_coord((int)ow, (int)p.iw, (int)p.ow, p.coord_mode, p.rw);
    int y0 = (int)floor(sy);
    int x0 = (int)floor(sx);
    float fy = (p.coord_mode == 2u) ? 0.0f : (sy - (float)y0);
    float fx = (p.coord_mode == 2u) ? 0.0f : (sx - (float)x0);
    if (p.coord_mode == 4u) { fy = (p.oh > 1) ? (sy - (float)y0) : 0.0f; fx = (p.ow > 1) ? (sx - (float)x0) : 0.0f; }
    int y1 = min(y0 + 1, (int)p.ih - 1);
    int x1 = min(x0 + 1, (int)p.iw - 1);
    y0 = clamp(y0, 0, (int)p.ih - 1);
    x0 = clamp(x0, 0, (int)p.iw - 1);
    float v00 = x[plane_base + (uint)(y0 * (int)p.iw + x0)];
    float v01 = x[plane_base + (uint)(y0 * (int)p.iw + x1)];
    float v10 = x[plane_base + (uint)(y1 * (int)p.iw + x0)];
    float v11 = x[plane_base + (uint)(y1 * (int)p.iw + x1)];
    y[out_index] = mix(mix(v00, v01, fx), mix(v10, v11, fx), fy);
}

kernel void k_matmul_int(device const uchar* a [[buffer(0)]],
                         device const uchar* b [[buffer(1)]],
                         device int* c [[buffer(2)]],
                         device const uchar* azp [[buffer(3)]],
                         device const uchar* bzp [[buffer(4)]],
                         constant MatMulIntParams& p [[buffer(9)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= p.m || col >= p.n) { return; }
    int zero_a = (p.use_buf_zp != 0) ? (int)azp[p.azp_off] : (int)p.a_zp;
    int zero_b = (p.use_buf_zp != 0) ? (int)bzp[p.bzp_off] : (int)p.b_zp;
    int acc = 0;
    for (uint k = 0u; k < p.k; ++k) {
        int av = (int)a[p.a_off + (int)(row * p.k + k)] - zero_a;
        int bv = (p.b_signed != 0)
                     ? (int)((char)b[p.b_off + (int)(k * p.n + col)]) - zero_b
                     : (int)b[p.b_off + (int)(k * p.n + col)] - zero_b;
        acc += av * bv;
    }
    c[p.c_off + (int)(row * p.n + col)] = acc;
}

kernel void k_quantize(device const float* x [[buffer(0)]],
                       device uchar* y [[buffer(1)]],
                       device const float* scale [[buffer(2)]],
                       device const uchar* zp [[buffer(3)]],
                       constant QuantizeParams& p [[buffer(9)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) { return; }
    float s = scale[p.scale_off];
    if (s == 0.0f) { s = 1.0f; }
    float v = round(x[p.x_off + (int)gid] / s) + (float)zp[p.zp_off];
    y[p.y_off + (int)gid] = (uchar)clamp(v, 0.0f, 255.0f);
}

kernel void k_quant_finalize(device const float* minmax [[buffer(0)]],
                             device float* scale [[buffer(1)]],
                             device uchar* zp [[buffer(2)]],
                             constant QuantFinalizeParams& p [[buffer(9)]]) {
    float lo = minmax[p.minmax_off];
    float hi = minmax[p.minmax_off + 1];
    float s = (hi - lo) / 255.0f;
    if (!(s > p.eps)) { s = 1.0f; }
    float zero = clamp(round(-lo / s), 0.0f, 255.0f);
    scale[p.scale_off] = s;
    zp[p.zp_off] = (uchar)zero;
}

kernel void k_minmax(device const float* x [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     constant MinMaxParams& p [[buffer(9)]]) {
    float lo = INFINITY;
    float hi = -INFINITY;
    for (uint i = 0u; i < p.n; ++i) {
        float v = x[p.x_off + (int)i];
        lo = min(lo, v);
        hi = max(hi, v);
    }
    if (p.n == 0u) { lo = 0.0f; hi = 0.0f; }
    out[0] = lo;
    out[1] = hi;
}

kernel void k_scatter(device const long* indices [[buffer(0)]],
                      device const float* updates [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant ScatterParams& p [[buffer(9)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= p.num_updates) { return; }
    int base = p.out_off;
    for (uint d = 0u; d < p.k; ++d) {
        long raw = indices[p.idx_off + (int)(gid * p.k + d)];
        int coord = (int)raw;
        if (coord < 0) { continue; }
        base += coord * p.strides[d];
    }
    for (uint s = 0u; s < p.slice_size; ++s) {
        out[base + (int)s] = updates[p.upd_off + (int)(gid * p.slice_size + s)];
    }
}

kernel void k_gather(device const float* data [[buffer(0)]],
                     device const uchar* indices [[buffer(1)]],
                     device float* y [[buffer(2)]],
                     constant GatherParams& p [[buffer(9)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= p.out_elems) { return; }
    uint i = gid % p.inner;
    uint rest = gid / p.inner;
    uint j = rest % p.idx_len;
    uint o = rest / p.idx_len;
    long raw = (p.idx_i64 != 0)
                   ? *(device const long*)(indices + (p.idx_off + (int)j) * 8)
                   : (long)*(device const int*)(indices + (p.idx_off + (int)j) * 4);
    long axis = raw;
    if (axis < 0) { axis += (long)p.axis_len; }
    if (axis < 0) { axis = 0; }
    if (axis > (long)p.axis_len - 1) { axis = (long)p.axis_len - 1; }
    y[p.y_off + (int)gid] = data[p.data_off + (int)((o * p.axis_len + (uint)axis) * p.inner + i)];
}
