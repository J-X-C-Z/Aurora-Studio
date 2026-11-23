use std::collections::BTreeSet;

use aurora_portrait::cuda::{CudaError, Dispatch};
use aurora_portrait::effect_router::{
    adapter, route, AdapterMeta, Backend, BackendUsed, Beard, BeardInput, CudaEffect, EyeBag,
    EyeBagInput, Gloss, GlossInput, MakeupInput, MakeupLayer, MeshAttribute, MeshAttributeInput,
    MoistureBody, MoistureBodyInput, MoistureFace, MoistureFaceInput, P15FacialSmooth, P15Input,
    RouteError, RouteNote, SkinAdjustLut, SkinLutInput, SkinTint, SkinTintInput, SmileInput,
    SmileMaskInput, SmilePf100007, SmilePf8033, TeethBlend, TeethBlendInput, TeethInput,
    TeethWhiten, TextureBlend, TextureBlendInput, UvMap, UvMapInput, ADAPTERS,
};
use aurora_portrait::kernels::beard;
use aurora_portrait::kernels::facial_smooth;
use aurora_portrait::kernels::makeup_material::{BlendMode, ColorMaterialParams};
use aurora_portrait::kernels::mesh_attribute;
use aurora_portrait::kernels::moisture::MaskChannel;
use aurora_portrait::kernels::smile_warp;
use aurora_portrait::kernels::teeth_blend;
use aurora_portrait::kernels::texture_mask_blend::{self, NormalBlend};
use aurora_portrait::kernels::uvmap_color::{self, Channel};
use aurora_portrait::{Image, Refusal};

fn dispatch() -> Dispatch {
    Dispatch {
        kernel: "test_kernel",
        grid: (1, 1),
        block: (1, 1),
        ptx_bytes: 1,
        ptx_isa: (9, 0),
        device: "test".to_string(),
        compute_cap: (8, 9),
        driver_version: 1,
    }
}

struct FailCuda;

impl CudaEffect for FailCuda {
    const NAME: &'static str = "test_fail_cuda";
    const PFS: &'static [u32] = &[];
    const KERNELS: &'static [&'static str] = &["test_kernel"];
    const CPU_ENTRY: &'static str = "test_cpu";

    type Input<'a> = ();
    type Output = u8;

    fn cpu(_: &()) -> Result<u8, Refusal> {
        Ok(7)
    }

    fn cuda(_: &()) -> Result<(u8, Vec<Dispatch>), CudaError> {
        Err(CudaError::Unavailable("test driver"))
    }
}

struct EmptyDispatch;

impl CudaEffect for EmptyDispatch {
    const NAME: &'static str = "test_empty_dispatch";
    const PFS: &'static [u32] = &[];
    const KERNELS: &'static [&'static str] = &["test_kernel"];
    const CPU_ENTRY: &'static str = "test_cpu";

    type Input<'a> = ();
    type Output = u8;

    fn cpu(_: &()) -> Result<u8, Refusal> {
        Ok(7)
    }

    fn cuda(_: &()) -> Result<(u8, Vec<Dispatch>), CudaError> {
        Ok((7, Vec::new()))
    }
}

struct GoodCuda;

impl CudaEffect for GoodCuda {
    const NAME: &'static str = "test_good_cuda";
    const PFS: &'static [u32] = &[];
    const KERNELS: &'static [&'static str] = &["test_kernel"];
    const CPU_ENTRY: &'static str = "test_cpu";

    type Input<'a> = ();
    type Output = u8;

    fn cpu(_: &()) -> Result<u8, Refusal> {
        Ok(7)
    }

    fn cuda(_: &()) -> Result<(u8, Vec<Dispatch>), CudaError> {
        Ok((7, vec![dispatch()]))
    }
}

#[test]
fn registry_lists_sixteen_audited_effects() {
    assert_eq!(ADAPTERS.len(), 16);
    let names: BTreeSet<&str> = ADAPTERS.iter().map(|meta| meta.name).collect();
    assert_eq!(names.len(), 16, "adapter names must be unique");
    for &AdapterMeta {
        name,
        pfs,
        kernels,
        cpu_entry,
        ..
    } in ADAPTERS
    {
        assert!(!pfs.is_empty(), "{name} has no PF");
        assert!(!kernels.is_empty(), "{name} has no kernel");
        assert!(!cpu_entry.is_empty(), "{name} has no CPU entry");
    }
    let expected: &[(&str, &[u32])] = &[
        ("teeth_whiten", &[108]),
        ("uvmap_type143", &[407]),
        ("gloss_pass", &[8068]),
        ("skin_tint_pf21015", &[21015]),
        ("p15_facial_smooth", &[90023, 90024, 21024]),
        ("skin_adjust_lut", &[90005, 90007]),
        ("beard_blend", &[21026]),
        ("moisture_face", &[90002]),
        ("moisture_body", &[90010]),
        ("makeup_layer", &[90033]),
        ("teeth_blend", &[30026]),
        ("texture_blend", &[30009]),
        ("smile_pf100007", &[100007]),
        ("smile_pf8033", &[8033]),
        ("mesh_attribute", &[8033]),
        ("eye_bag_compose", &[90116]),
    ];
    for (name, pfs) in expected {
        let meta = adapter(name).unwrap_or_else(|| panic!("missing adapter {name}"));
        assert_eq!(meta.pfs, *pfs, "{name} PF set changed");
    }
    assert!(adapter("beard_blend").is_some());
    assert!(adapter("not_an_adapter").is_none());
}

#[test]
fn cpu_policy_never_reports_gpu() {
    let routed = route::<GoodCuda>(&(), Backend::Cpu).unwrap();
    assert_eq!(routed.backend, BackendUsed::Cpu);
    assert_eq!(routed.output, 7);
    assert!(routed.dispatches.is_empty());
    assert!(routed.fallback.is_none());
}

#[test]
fn cuda_policy_propagates_unavailability() {
    assert_eq!(
        route::<FailCuda>(&(), Backend::Cuda),
        Err(RouteError::Cuda(CudaError::Unavailable("test driver")))
    );
}

#[test]
fn auto_policy_records_the_exact_cuda_error() {
    let routed = route::<FailCuda>(&(), Backend::Auto).unwrap();
    assert_eq!(routed.backend, BackendUsed::Cpu);
    assert_eq!(routed.output, 7);
    assert_eq!(
        routed.fallback,
        Some(RouteNote::Cuda(CudaError::Unavailable("test driver")))
    );
}

#[test]
fn a_cuda_result_without_a_dispatch_is_not_evidence() {
    assert_eq!(
        route::<EmptyDispatch>(&(), Backend::Cuda),
        Err(RouteError::MissingDispatchEvidence("test_empty_dispatch"))
    );
    let routed = route::<EmptyDispatch>(&(), Backend::Auto).unwrap();
    assert_eq!(routed.backend, BackendUsed::Cpu);
    assert_eq!(
        routed.fallback,
        Some(RouteNote::MissingDispatchEvidence("test_empty_dispatch"))
    );
}

#[test]
fn cuda_success_keeps_the_dispatch() {
    let routed = route::<GoodCuda>(&(), Backend::Cuda).unwrap();
    assert_eq!(routed.backend, BackendUsed::Cuda);
    assert_eq!(routed.dispatches.len(), 1);
    assert_eq!(routed.dispatches[0].kernel, "test_kernel");
    assert!(routed.fallback.is_none());
}

#[test]
fn real_cpu_fallbacks_call_the_recovered_kernel() {
    let source = Image::solid(8, 8, [10, 20, 30, 255]);
    let repair = Image::solid(8, 8, [200, 100, 50, 255]);
    let mask_a = vec![200u8; 64];
    let mask_b = vec![200u8; 64];
    let beard_input = BeardInput {
        source: &source,
        repair: &repair,
        mask_a: &mask_a,
        mask_b: &mask_b,
        strength: 1.0,
    };
    assert_eq!(
        route::<Beard>(&beard_input, Backend::Cpu).unwrap().output,
        beard::apply(&source, &repair, &mask_a, &mask_b, 1.0).unwrap()
    );

    let original = Image::solid(6, 5, [80, 90, 100, 255]);
    let effect = Image::solid(6, 5, [30, 60, 200, 255]);
    let mask = vec![128u8; 30];
    let blend_input = TeethBlendInput {
        original: &original,
        effect: &effect,
        mask: &mask,
    };
    assert_eq!(
        route::<TeethBlend>(&blend_input, Backend::Cpu)
            .unwrap()
            .output,
        teeth_blend::apply(&original, &effect, &mask).unwrap()
    );

    let texture_effect = Image::solid(3, 2, [30, 60, 200, 255]);
    let texture_input = TextureBlendInput {
        original: &original,
        effect: &texture_effect,
        mask: &mask,
        effect_uv: None,
        params: NormalBlend::PF30009,
    };
    assert_eq!(
        route::<TextureBlend>(&texture_input, Backend::Cpu)
            .unwrap()
            .output,
        texture_mask_blend::apply(
            texture_input.original,
            texture_input.effect,
            &mask,
            None,
            NormalBlend::PF30009
        )
        .unwrap()
    );

    let gloss_source = Image::solid(5, 4, [120, 90, 60, 255]);
    let gloss_mask = vec![160u8; 20];
    let gloss_extra = vec![255u8; 20];
    let gloss_input = GlossInput {
        source: &gloss_source,
        masks_r: [&gloss_mask, &gloss_mask, &gloss_mask],
        extra_masks_r: [&gloss_extra, &gloss_extra, &gloss_extra],
        native_percent: 60.0,
    };
    assert_eq!(
        route::<Gloss>(&gloss_input, Backend::Cpu).unwrap().output,
        aurora_portrait::kernels::skin_models::gloss_apply(
            &gloss_source,
            [&gloss_mask, &gloss_mask, &gloss_mask],
            [&gloss_extra, &gloss_extra, &gloss_extra],
            60.0
        )
        .unwrap()
    );
}

#[test]
fn all_sixteen_cpu_lanes_accept_a_minimal_same_job_frame() {
    let image = Image::solid(2, 2, [80, 120, 160, 255]);
    let masks = [255u8; 4];
    let lut = vec![128u8; 512 * 512 * 3];
    let lut2 = vec![64u8; 512 * 512 * 3];

    let teeth = TeethInput {
        source: &image,
        mask_g: &masks,
        whiten_lut: &lut,
        bright_lut: &lut2,
        common_ui: 0.0,
        bright_ui: 0.0,
        deyellow_ui: 0.0,
    };
    assert_eq!(
        route::<TeethWhiten>(&teeth, Backend::Cpu).unwrap().output,
        image
    );

    let tint = SkinTintInput {
        source: &image,
        mask_r: &masks,
        tint_green: &lut,
        tint_magenta: &lut2,
        ui: 0.0,
    };
    assert_eq!(
        route::<SkinTint>(&tint, Backend::Cpu).unwrap().output,
        image
    );

    let skin_lut = SkinLutInput {
        source: &image,
        mask_r: &masks,
        lut_rgb512: &lut,
        mix: 0.0,
        pf: 90005,
    };
    assert_eq!(
        route::<SkinAdjustLut>(&skin_lut, Backend::Cpu)
            .unwrap()
            .output,
        image
    );

    let p15 = P15Input {
        source: &image,
        base: &image,
        residual: &image,
        mask_r: &masks,
        mask_b: &masks,
        skin_mask: &masks,
        pf: 90024,
        pf_value: 0.0,
        uniforms: facial_smooth::Uniforms {
            legal_alpha: 0.0,
            dark_circle_alpha: 0.0,
            diff_factor: 0.0,
            diff_factor2: 0.0,
            color_threshold_factor: 0.0,
            mix_alpha: 0.0,
            diff_min_threshold: 0.0,
            diff_max_threshold: 0.0,
            high_freq_min_threshold: 0.0,
            high_freq_max_threshold: 0.0,
        },
    };
    assert_eq!(
        route::<P15FacialSmooth>(&p15, Backend::Cpu).unwrap().output,
        image
    );

    let moisture_face = MoistureFaceInput {
        source: &image,
        mask_rgba: &[255u8; 16],
        blur_y: &masks,
        ui: 0.0,
        pivot: 0.0,
        gain: 1.0,
        channel: MaskChannel::R,
        skin_map_width: 1,
        skin_map_height: 1,
        skin_map: &[0.5],
        hardlight: &[0.0, 1.0],
        beauty: &[0.0, 1.0],
    };
    assert_eq!(
        route::<MoistureFace>(&moisture_face, Backend::Cpu)
            .unwrap()
            .output,
        image
    );

    let weight = [0.0f32; 4];
    let moisture_body = MoistureBodyInput {
        source: &image,
        weight_skin: &weight,
        weight_dog: &weight,
        beauty: &[0.0, 1.0],
        degree: 0.0,
        roi: (0, 0, 2, 2),
    };
    assert_eq!(
        route::<MoistureBody>(&moisture_body, Backend::Cpu)
            .unwrap()
            .output,
        image
    );

    let uv = [[0.0f32; 2]; 4];
    let makeup = MakeupInput {
        source: &image,
        material: &image,
        owner_mask: &image,
        mask_uv: &uv,
        material_uv: &uv,
        params: ColorMaterialParams {
            mix_color: [255, 0, 0, 255],
            opacity_percent: 0.0,
            degree01: 0.0,
            blend_mode: BlendMode::Normal,
            mask_channel: aurora_portrait::kernels::makeup_material::MaskChannel::R,
            blend_alpha: false,
        },
    };
    assert_eq!(
        route::<MakeupLayer>(&makeup, Backend::Cpu).unwrap().output,
        image
    );

    let eye_bag = EyeBagInput {
        source: &image,
        color3: &[0.0f32; 12],
        response: &[0.0f32; 4],
        degree: 0.0,
    };
    assert_eq!(
        route::<EyeBag>(&eye_bag, Backend::Cpu).unwrap().output,
        image
    );

    let frame = vec![77u8; 16];
    let smile = SmileInput {
        source_bgra: &frame,
        width: 2,
        height: 2,
        degree: 0.0,
        lip_mask: &masks,
        mouth_mask: &masks,
        model_dx: &[0.0f32; 4],
        model_dy: &[0.0f32; 4],
        roi: smile_warp::Rect {
            x: 0,
            y: 0,
            width: 2,
            height: 2,
        },
    };
    assert_eq!(
        route::<SmilePf100007>(&smile, Backend::Cpu).unwrap().output,
        frame
    );
}

#[test]
fn uv_and_mesh_fallbacks_match_their_cpu_entries() {
    let source = Image::solid(6, 4, [44, 55, 66, 255]);
    let field = vec![[0.0f32; 4]; 24];
    let uv_input = UvMapInput {
        source: &source,
        field: &field,
        channel: Channel::Rg,
        catmull: None,
    };
    assert_eq!(
        route::<UvMap>(&uv_input, Backend::Cpu).unwrap().output,
        source
    );
    let zero_field = Image::solid(6, 4, [0, 0, 0, 0]);
    assert_eq!(
        uvmap_color::apply(&source, &zero_field, Channel::Rg, None).unwrap(),
        source
    );

    let dest = [[0.0f32, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
    let attr = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let faces = [0u32, 1, 2, 0, 2, 3];
    let mesh_input = MeshAttributeInput {
        width: 4,
        height: 4,
        dest: &dest,
        attr: &attr,
        faces: &faces,
    };
    assert_eq!(
        route::<MeshAttribute>(&mesh_input, Backend::Cpu)
            .unwrap()
            .output,
        mesh_attribute::rasterize(4, 4, &dest, &attr, &faces).unwrap()
    );
}

#[test]
fn smile_fallback_maps_its_private_refusal_type() {
    let frame = vec![12u8; 8 * 8 * 4];
    let dx = vec![0.0f32; 64];
    let dy = vec![0.0f32; 64];
    let mask = vec![255u8; 64];
    let input = SmileMaskInput {
        source_bgra: &frame,
        width: 8,
        height: 8,
        dx: &dx,
        dy: &dy,
        mask: &mask,
        degree01: 0.0,
    };
    assert_eq!(
        route::<SmilePf8033>(&input, Backend::Cpu).unwrap().output,
        smile_warp::pf8033_dense_warp(&frame, 8, 8, &dx, &dy, &mask, 0.0).unwrap()
    );

    let short = SmileMaskInput {
        mask: &mask[..63],
        ..input
    };
    assert_eq!(
        route::<SmilePf8033>(&short, Backend::Cpu),
        Err(RouteError::Cpu(Refusal::MissingInput("pf8033 dx/dy/mask")))
    );
}

#[test]
fn real_cuda_dispatch_is_reported_or_typed() {
    let original = Image::solid(4, 4, [90, 100, 110, 255]);
    let effect = Image::solid(4, 4, [40, 70, 210, 255]);
    let mask = vec![200u8; 16];
    let input = TeethBlendInput {
        original: &original,
        effect: &effect,
        mask: &mask,
    };

    let cpu = route::<TeethBlend>(&input, Backend::Cpu).unwrap();
    match route::<TeethBlend>(&input, Backend::Cuda) {
        Ok(gpu) => {
            assert_eq!(gpu.backend, BackendUsed::Cuda);
            assert!(!gpu.dispatches.is_empty());
            assert!(gpu.fallback.is_none());
            assert_eq!(gpu.output, cpu.output);
            println!(
                "CUDA kernel={} grid={:?} device={}",
                gpu.dispatches[0].kernel, gpu.dispatches[0].grid, gpu.dispatches[0].device
            );
        }
        Err(RouteError::Cuda(CudaError::Unavailable(reason))) => {
            assert!(
                std::env::var_os("PIXCAKE_CUDA_REQUIRED").is_none(),
                "CUDA required but unavailable: {reason}"
            );
            let auto = route::<TeethBlend>(&input, Backend::Auto).unwrap();
            assert_eq!(auto.backend, BackendUsed::Cpu);
            assert_eq!(
                auto.fallback,
                Some(RouteNote::Cuda(CudaError::Unavailable(reason)))
            );
        }
        Err(other) => panic!("NVIDIA path present but broken: {other:?}"),
    }
}
