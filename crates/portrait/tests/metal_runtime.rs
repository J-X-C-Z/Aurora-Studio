//! macOS-only real-device checks for the Metal host.
//!
//! Every test is `#[ignore]`, so a default `cargo test` only compiles them and
//! never records a pass. Run on macOS with:
//!
//! ```text
//! cargo test -p aurora-pixcake --test metal_runtime -- --ignored --test-threads=1
//! ```
//!
//! On a non-macOS host an explicit `--ignored` run panics instead of skipping,
//! so a missing Apple GPU can never be mistaken for a pass.

use aurora_pixcake::kernels::facial_smooth::{composite_pixel, Uniforms as CpuUniforms};
use aurora_pixcake::metal::{
    BlurKernel, MetalContext, MetalRefusal, ProductUniforms, ProjectionInput, SmoothInput,
    WhitenInput, LUT_FLOATS,
};

fn require_macos() {
    if !cfg!(target_os = "macos") {
        panic!("requires macOS and a Metal device; non-macOS is not a pass");
    }
}

fn product() -> ProductUniforms {
    ProductUniforms {
        legal_alpha: 0.9,
        dark_circle_alpha: 0.7,
        diff_factor: 1.1,
        diff_factor2: 0.15,
        color_threshold_factor: 0.2,
        mix_alpha: 0.35,
        diff_min_threshold: -1.0,
        diff_max_threshold: 1.0,
        high_freq_min_threshold: -0.4,
        high_freq_max_threshold: 0.4,
    }
}

fn cpu_product() -> CpuUniforms {
    let product = product();
    CpuUniforms {
        legal_alpha: product.legal_alpha,
        dark_circle_alpha: product.dark_circle_alpha,
        diff_factor: product.diff_factor,
        diff_factor2: product.diff_factor2,
        color_threshold_factor: product.color_threshold_factor,
        mix_alpha: product.mix_alpha,
        diff_min_threshold: product.diff_min_threshold,
        diff_max_threshold: product.diff_max_threshold,
        high_freq_min_threshold: product.high_freq_min_threshold,
        high_freq_max_threshold: product.high_freq_max_threshold,
    }
}

#[test]
#[ignore = "requires macOS Metal device"]
fn p15_composite_matches_cpu_float_golden_on_non_square_roi() {
    require_macos();
    assert!(aurora_pixcake::metal::available());

    // Five by three ROI with a padded stride: exercises a non-square dispatch.
    let width = 5u32;
    let height = 3u32;
    let stride = 6u32;
    let floats = (stride * height * 4) as usize;
    let mut source = vec![0.0f32; floats];
    let mut base = vec![0.0f32; floats];
    let mut residual = vec![0.0f32; floats];
    let mut mask = vec![0.0f32; floats];
    let mut skin = vec![0.0f32; (stride * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let pixel = (y * stride + x) as usize;
            let i = pixel * 4;
            source[i] = (x as f32 + 1.0) / 6.0;
            source[i + 1] = (y as f32 + 1.0) / 4.0;
            source[i + 2] = ((x + y) as f32) / 8.0;
            source[i + 3] = ((x * 3 + y) as f32) / 20.0 + 0.1;
            base[i] = source[i] * 0.9;
            base[i + 1] = source[i + 1] * 0.9;
            base[i + 2] = source[i + 2] * 0.9;
            base[i + 3] = source[i + 3];
            residual[i] = source[i] * 0.95;
            residual[i + 1] = source[i + 1] * 0.95;
            residual[i + 2] = source[i + 2] * 0.95;
            residual[i + 3] = source[i + 3];
            mask[i] = 0.8;
            mask[i + 1] = 0.0;
            mask[i + 2] = 0.4;
            mask[i + 3] = 0.0;
            skin[pixel] = 0.9;
        }
    }

    let input = SmoothInput {
        source: &source,
        mask_rgba: &mask,
        skin_mask: &skin,
        base: &base,
        residual: &residual,
        width,
        height,
        stride,
        pf: 90023,
        pf_value: 1.0,
        ticket: 0x0200_0001,
        uniforms: product(),
    };
    let context = MetalContext::new().expect("MetalContext");
    let actual = context.facial_smooth(&input).expect("p15 dispatch");
    assert_eq!(actual.len(), floats);

    let mut max_diff = 0.0f32;
    let golden_uniforms = cpu_product();
    for y in 0..height as usize {
        for x in 0..width as usize {
            let pixel = y * stride as usize + x;
            let i = pixel * 4;
            let golden = composite_pixel(
                [source[i], source[i + 1], source[i + 2], source[i + 3]],
                [base[i], base[i + 1], base[i + 2], base[i + 3]],
                [
                    residual[i],
                    residual[i + 1],
                    residual[i + 2],
                    residual[i + 3],
                ],
                mask[i],
                mask[i + 2],
                skin[pixel],
                golden_uniforms,
            )
            .expect("CPU golden");
            for channel in 0..3 {
                let diff = (actual[i + channel] - golden[channel]).abs();
                max_diff = max_diff.max(diff);
                assert!(
                    diff <= 1e-5,
                    "pixel ({x},{y}) channel {channel}: gpu={} cpu={} diff={}",
                    actual[i + channel],
                    golden[channel],
                    diff
                );
            }
            assert_eq!(actual[i + 3].to_bits(), source[i + 3].to_bits());
        }
    }
    eprintln!("p15 max abs diff = {max_diff:e}");
}

#[test]
#[ignore = "requires macOS Metal device"]
fn p15_inactive_gate_refuses_before_dispatch() {
    require_macos();
    let source = [0.2f32, 0.2, 0.2, 0.75];
    let mask = [1.0f32, 0.0, 1.0, 0.0];
    let skin = [1.0f32];
    let input = SmoothInput {
        source: &source,
        mask_rgba: &mask,
        skin_mask: &skin,
        base: &source,
        residual: &source,
        width: 1,
        height: 1,
        stride: 1,
        pf: 90023,
        pf_value: 0.001,
        ticket: 1,
        uniforms: product(),
    };
    let context = MetalContext::new().expect("MetalContext");
    assert_eq!(context.facial_smooth(&input), Err(MetalRefusal::Inactive));
    let blurred = context
        .blur(BlurKernel::Box, &source, 1, 1, 1, 0, 1, product())
        .expect("1x1 box blur");
    for channel in 0..3 {
        assert!((blurred[channel] - source[channel]).abs() <= 1e-6);
    }
    assert_eq!(blurred[3].to_bits(), source[3].to_bits());
}

#[test]
#[ignore = "requires macOS Metal device"]
fn u8_whiten_preserves_alpha_and_honours_gate() {
    require_macos();
    let context = MetalContext::new().expect("MetalContext");

    // A 3x2 non-square surface with distinct alpha bytes.
    let width = 3u32;
    let height = 2u32;
    let mut source = Vec::new();
    for i in 0..(width * height) as u8 {
        source.extend_from_slice(&[10 + i, 20 + i, 30 + i, 40 + i]);
    }
    let mask = vec![255u8; (width * height) as usize];
    let mut lut = vec![0.0f32; LUT_FLOATS];
    for texel in lut.chunks_exact_mut(4) {
        texel[0] = 1.0;
        texel[1] = 1.0;
        texel[2] = 1.0;
        texel[3] = 1.0;
    }
    let base = WhitenInput {
        bgra: &source,
        width,
        height,
        stride_bytes: width * 4,
        mask: &mask,
        mask_x: 0,
        mask_y: 0,
        mask_width: width,
        mask_height: height,
        lut: &lut,
        alpha: 1.0,
        flags: 0,
        ticket: 7,
    };

    let white = context.masked_whiten(&base).expect("whiten alpha=1");
    for (index, pixel) in white.chunks_exact(4).enumerate() {
        assert_eq!(&pixel[..3], &[255, 255, 255], "pixel {index} not whitened");
        assert_eq!(pixel[3], source[index * 4 + 3], "alpha changed");
    }

    for gate in [0.0f32, 0.0005, 0.001] {
        let input = WhitenInput {
            alpha: gate,
            ..base
        };
        let output = context.masked_whiten(&input).expect("gated whiten");
        assert_eq!(
            output, source,
            "gate {gate} must leave the surface unchanged"
        );
    }
}

#[test]
#[ignore = "requires macOS Metal device"]
fn mask_projection_shape_refusal_and_coverage_max() {
    require_macos();
    let context = MetalContext::new().expect("MetalContext");
    let source_mask = [1u8, 2, 3, 4];
    let base = ProjectionInput {
        source_mask: &source_mask,
        source_mask_width: 2,
        source_mask_height: 2,
        surface_width: 4,
        surface_height: 4,
        destination_x: 0,
        destination_y: 0,
        destination_width: 4,
        destination_height: 4,
        source_rect_x: 0,
        source_rect_y: 0,
        source_rect_width: 2,
        source_rect_height: 2,
        domain_x0: 0,
        domain_y0: 0,
        roi_width: 2,
        roi_height: 2,
        ticket: 11,
    };
    let projected = context.project_mask(&base).expect("projection");
    let expected = [1u8, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4];
    assert_eq!(projected, expected);

    let bad_rect = ProjectionInput {
        destination_width: 5,
        ..base
    };
    assert_eq!(
        context.project_mask(&bad_rect),
        Err(MetalRefusal::ShapeMismatch("projection_rect"))
    );
    let short = ProjectionInput {
        source_mask: &source_mask[..3],
        ..base
    };
    assert_eq!(
        context.project_mask(&short),
        Err(MetalRefusal::ShapeMismatch("source_mask"))
    );
}
