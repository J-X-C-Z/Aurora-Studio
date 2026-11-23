// C1 blemish gain-map consumer on CUDA.
//
// One thread per output pixel. The 450-long-side RGBA32F map is sampled
// bilinearly with the exact expression the CPU lane uses, then the amount,
// confidence curve and protection mask are applied per channel.
//
// Every arithmetic step goes through the __f*_rn intrinsics and the image is
// compiled with -fmad=false, so the result is the same sequence of IEEE-754
// single-precision roundings as the Rust scalar oracle. In particular the
// compiler may not contract a*b+c into an FMA or turn /65535.0f into a
// multiply by a reciprocal.

struct BlemishParams {
    unsigned int width;
    unsigned int height;
    unsigned int map_width;
    unsigned int map_height;
    float sx;
    float sy;
    float amount_scale;
    float curve_t;
    unsigned int has_curve;
    unsigned int has_mask;
};

__device__ __forceinline__ float lerp_rn(float a, float b, float t) {
    return __fadd_rn(a, __fmul_rn(__fsub_rn(b, a), t));
}

__device__ __forceinline__ void sample_axis(
    unsigned int index,
    float scale,
    unsigned int limit,
    unsigned int* i0_out,
    unsigned int* i1_out,
    float* t_out) {
    float f = __fsub_rn(__fmul_rn(__fadd_rn((float)index, 0.5f), scale), 0.5f);
    f = fminf(fmaxf(f, 0.0f), (float)(limit - 1u));
    unsigned int i0 = (unsigned int)floorf(f);
    unsigned int i1 = i0 + 1u < limit ? i0 + 1u : limit - 1u;
    *i0_out = i0;
    *i1_out = i1;
    *t_out = __fsub_rn(f, (float)i0);
}

__device__ __forceinline__ float weight_from(
    float conf,
    const BlemishParams& p,
    const unsigned char* mask,
    unsigned int pixel) {
    float weight = __fmul_rn(conf, p.amount_scale);
    weight = fminf(fmaxf(weight, 0.0f), 1.0f);
    if (p.has_curve != 0u) {
        weight = __fadd_rn(__fmul_rn(weight, __fsub_rn(1.0f, p.curve_t)), p.curve_t);
    }
    if (p.has_mask != 0u) {
        float keep = __fdiv_rn((float)(255u - (unsigned int)mask[pixel]), 255.0f);
        weight = __fmul_rn(weight, keep);
    }
    return weight;
}

extern "C" __global__ void blemish_gainmap_16u_kernel(
    const unsigned short* __restrict__ src,
    const float4* __restrict__ map,
    const unsigned char* __restrict__ mask,
    unsigned short* __restrict__ out,
    BlemishParams p) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) {
        return;
    }

    unsigned int x0, x1, y0, y1;
    float tx, ty;
    sample_axis(x, p.sx, p.map_width, &x0, &x1, &tx);
    sample_axis(y, p.sy, p.map_height, &y0, &y1, &ty);

    unsigned int row0 = y0 * p.map_width;
    unsigned int row1 = y1 * p.map_width;
    float4 a = map[row0 + x0];
    float4 b = map[row0 + x1];
    float4 c = map[row1 + x0];
    float4 d = map[row1 + x1];

    float top_r = lerp_rn(a.x, b.x, tx);
    float top_g = lerp_rn(a.y, b.y, tx);
    float top_b = lerp_rn(a.z, b.z, tx);
    float top_c = lerp_rn(a.w, b.w, tx);
    float bot_r = lerp_rn(c.x, d.x, tx);
    float bot_g = lerp_rn(c.y, d.y, tx);
    float bot_b = lerp_rn(c.z, d.z, tx);
    float bot_c = lerp_rn(c.w, d.w, tx);

    float gain_r = lerp_rn(top_r, bot_r, ty);
    float gain_g = lerp_rn(top_g, bot_g, ty);
    float gain_b = lerp_rn(top_b, bot_b, ty);
    float conf = lerp_rn(top_c, bot_c, ty);

    unsigned int pixel = y * p.width + x;
    float weight = weight_from(conf, p, mask, pixel);
    float inv = __fsub_rn(1.0f, weight);

    unsigned int i = pixel * 3u;
    const unsigned short* s = src + i;
    unsigned short* o = out + i;

    float sr = __fdiv_rn((float)s[0], 65535.0f);
    float sg = __fdiv_rn((float)s[1], 65535.0f);
    float sb = __fdiv_rn((float)s[2], 65535.0f);

    float vr = __fmul_rn(sr, __fadd_rn(__fmul_rn(gain_r, weight), inv));
    float vg = __fmul_rn(sg, __fadd_rn(__fmul_rn(gain_g, weight), inv));
    float vb = __fmul_rn(sb, __fadd_rn(__fmul_rn(gain_b, weight), inv));

    vr = fminf(fmaxf(vr, 0.0f), 1.0f);
    vg = fminf(fmaxf(vg, 0.0f), 1.0f);
    vb = fminf(fmaxf(vb, 0.0f), 1.0f);

    o[0] = (unsigned short)(__fadd_rn(__fmul_rn(vr, 65535.0f), 0.5f));
    o[1] = (unsigned short)(__fadd_rn(__fmul_rn(vg, 65535.0f), 0.5f));
    o[2] = (unsigned short)(__fadd_rn(__fmul_rn(vb, 65535.0f), 0.5f));
}

extern "C" __global__ void blemish_gainmap_8u_kernel(
    const unsigned char* __restrict__ src,
    const float4* __restrict__ map,
    const unsigned char* __restrict__ mask,
    unsigned char* __restrict__ out,
    BlemishParams p) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) {
        return;
    }

    unsigned int x0, x1, y0, y1;
    float tx, ty;
    sample_axis(x, p.sx, p.map_width, &x0, &x1, &tx);
    sample_axis(y, p.sy, p.map_height, &y0, &y1, &ty);

    unsigned int row0 = y0 * p.map_width;
    unsigned int row1 = y1 * p.map_width;
    float4 a = map[row0 + x0];
    float4 b = map[row0 + x1];
    float4 c = map[row1 + x0];
    float4 d = map[row1 + x1];

    float gain_r = lerp_rn(lerp_rn(a.x, b.x, tx), lerp_rn(c.x, d.x, tx), ty);
    float gain_g = lerp_rn(lerp_rn(a.y, b.y, tx), lerp_rn(c.y, d.y, tx), ty);
    float gain_b = lerp_rn(lerp_rn(a.z, b.z, tx), lerp_rn(c.z, d.z, tx), ty);
    float conf = lerp_rn(lerp_rn(a.w, b.w, tx), lerp_rn(c.w, d.w, tx), ty);

    unsigned int pixel = y * p.width + x;
    float weight = weight_from(conf, p, mask, pixel);
    float inv = __fsub_rn(1.0f, weight);

    unsigned int i = pixel * 3u;
    const unsigned char* s = src + i;
    unsigned char* o = out + i;

    float sr = __fdiv_rn((float)s[0], 255.0f);
    float sg = __fdiv_rn((float)s[1], 255.0f);
    float sb = __fdiv_rn((float)s[2], 255.0f);

    float vr = __fmul_rn(sr, __fadd_rn(__fmul_rn(gain_r, weight), inv));
    float vg = __fmul_rn(sg, __fadd_rn(__fmul_rn(gain_g, weight), inv));
    float vb = __fmul_rn(sb, __fadd_rn(__fmul_rn(gain_b, weight), inv));

    vr = fminf(fmaxf(vr, 0.0f), 1.0f);
    vg = fminf(fmaxf(vg, 0.0f), 1.0f);
    vb = fminf(fmaxf(vb, 0.0f), 1.0f);

    o[0] = (unsigned char)(__fadd_rn(__fmul_rn(vr, 255.0f), 0.5f));
    o[1] = (unsigned char)(__fadd_rn(__fmul_rn(vg, 255.0f), 0.5f));
    o[2] = (unsigned char)(__fadd_rn(__fmul_rn(vb, 255.0f), 0.5f));
}
