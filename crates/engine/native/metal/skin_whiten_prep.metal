// Engine Metal prep kernel for the first continuous chain (M2).
//
// The chain is: 3x3 box blur (this file) -> o3_masked_lut_whiten.
// The blur exists so the batch has two real kernels with a data dependency
// through a shared buffer; it is not a proposed product effect. Its CPU twin
// lives in `crates/engine/src/mock/skin_whiten_metal.rs` and uses the same
// integer arithmetic, so parity is exact for the blur stage.
#include <metal_stdlib>

using namespace metal;

// Keep integer operands integral; the pragma also documents that no
// floating-point contraction happens in this file.
#pragma clang fp contract(off)

struct SkinWhitenBlurParams {
    uint width;
    uint height;
    uint reserved0;
    uint reserved1;
};
static_assert(sizeof(SkinWhitenBlurParams) == 16, "blur params ABI must stay 16 bytes");

// BGRA8 3x3 box blur with clamped edges. Every channel, including alpha, is
// blurred with (sum + 4) / 9 so the CPU reference can match bit-for-bit.
kernel void skin_whiten_box_blur(
    device const uchar *source_bgra [[buffer(0)]],
    device uchar *destination_bgra [[buffer(1)]],
    constant SkinWhitenBlurParams &p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    uint index = (gid.y * p.width + gid.x) * 4u;
    int last_x = (int)p.width - 1;
    int last_y = (int)p.height - 1;

    for (uint c = 0u; c < 4u; ++c) {
        uint sum = 0u;
        for (int dy = -1; dy <= 1; ++dy) {
            int sy = clamp((int)gid.y + dy, 0, last_y);
            uint row = (uint)sy * p.width;
            for (int dx = -1; dx <= 1; ++dx) {
                int sx = clamp((int)gid.x + dx, 0, last_x);
                sum += (uint)source_bgra[(row + (uint)sx) * 4u + c];
            }
        }
        destination_bgra[index + c] = (uchar)((sum + 4u) / 9u);
    }
}
