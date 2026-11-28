// PF30009 hair colour unify (kernels/hair_body.rs hair_color_unify).
// Metal mirror of native/wgsl/pf30009_hair_color_unify.wgsl.
#include <metal_stdlib>
using namespace metal;

// Keep binary32 mul-then-add order; fastMathEnabled=NO alone does not
// guarantee that the compiler refuses FMA contraction.
#pragma clang fp contract(off)

struct HairColorUnifyParams
{
    uint width;
    uint height;
    float amount;      // host-resolved |1 - 2*(ui+100)/200|
    uint reserved0;
    uint reserved1;
    uint reserved2;
};
static_assert(sizeof(HairColorUnifyParams) == 24, "HairColorUnifyParams must stay 24 bytes");

static inline uint pack_rgba(uint r, uint g, uint b, uint a)
{
    return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

static inline uint round_away_byte(float value)
{
    // MidpointRounding.AwayFromZero on non-negative 0..255 range.
    float rounded = floor(value + 0.5f);
    return clamp(uint(rounded), 0u, 255u);
}

kernel void pf30009_hair_color_unify(
    constant HairColorUnifyParams& p [[buffer(0)]],
    device const uint* src [[buffer(1)]],
    device const uint* effect_rgb [[buffer(2)]],
    device const float* hair_mask [[buffer(3)]],
    device uint* out [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    uint pixel = gid.y * p.width + gid.x;
    uint packed = src[pixel];
    float q = hair_mask[pixel] * p.amount;
    uint eff = effect_rgb[pixel];
    float sr = float(packed & 255u), sg = float((packed >> 8u) & 255u), sb = float((packed >> 16u) & 255u);
    float er = float(eff & 255u), eg = float((eff >> 8u) & 255u), eb = float((eff >> 16u) & 255u);
    uint r = round_away_byte(sr + q * (er - sr));
    uint g = round_away_byte(sg + q * (eg - sg));
    uint b = round_away_byte(sb + q * (eb - sb));
    out[pixel] = pack_rgba(r, g, b, (packed >> 24u) & 255u);
}
