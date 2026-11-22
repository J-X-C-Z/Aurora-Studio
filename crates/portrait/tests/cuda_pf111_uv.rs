// Synthetic PF111 typed-raster parity against the existing mesh-attribute PTX.
// This does not claim a vendor EBO, vendor UV value, or original-output parity.

#[cfg(windows)]
mod pf111_gpu {
    use aurora_portrait::cuda::CudaError;
    use aurora_portrait::cuda_mesh_attribute::{RESOLVE_KERNEL, SELECT_KERNEL};
    use aurora_portrait::cuda_pf111_uv::Pf111UvRaster;
    use aurora_portrait::kernels::pf111_uv::{
        EyePupilPosition49, EyeSide, NormalEyeEbo, NormalEyeUv49, PF111_VERTEX_COUNT,
        Pf111RasterInputs, RasterExtent, rasterize_pixels,
    };

    fn gpu() -> Option<Pf111UvRaster> {
        match Pf111UvRaster::open() {
            Ok(gpu) => Some(gpu),
            Err(CudaError::Unavailable(reason)) => {
                eprintln!("SKIP_NO_NVIDIA_DRIVER: {reason}");
                assert!(
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                    "CUDA required but unavailable: {reason}"
                );
                None
            }
            Err(broken) => panic!("NVIDIA path present but broken: {broken:?}"),
        }
    }

    #[test]
    fn synthetic_triangle_matches_the_cpu_bit_for_bit() {
        let Some(gpu) = gpu() else { return };
        let mut positions = vec![[0.0; 2]; PF111_VERTEX_COUNT];
        let mut uv = vec![[0.0; 2]; PF111_VERTEX_COUNT];
        positions[0] = [0.0, 0.0];
        positions[1] = [4.0, 0.0];
        positions[2] = [0.0, 4.0];
        uv[0] = [0.0, 0.0];
        uv[1] = [1.0, 0.0];
        uv[2] = [0.0, 1.0];
        let ebo = [0u16, 1, 2];
        let inputs = Pf111RasterInputs {
            side: EyeSide::Right,
            positions: Some(EyePupilPosition49::new(&positions).unwrap()),
            normal_eye_uv: Some(NormalEyeUv49::new(&uv).unwrap()),
            extent: Some(RasterExtent::new(4, 4).unwrap()),
            normal_eye_ebo: Some(NormalEyeEbo::new(&ebo).unwrap()),
        };

        let expected = rasterize_pixels(inputs).unwrap();
        let (actual, dispatches) = gpu.rasterize(inputs).expect("real GPU dispatch");
        assert_eq!(dispatches.len(), 2);
        assert_eq!(dispatches[0].kernel, SELECT_KERNEL);
        assert_eq!(dispatches[1].kernel, RESOLVE_KERNEL);
        assert_eq!(actual.side(), EyeSide::Right);
        assert!(actual.mask_uv().is_none());
        assert_eq!(actual.writes(), expected.writes());
        for (index, (left, right)) in expected
            .material_uv()
            .iter()
            .zip(actual.material_uv())
            .enumerate()
        {
            assert_eq!(
                left[0].to_bits(),
                right[0].to_bits(),
                "material u bits differ at pixel {index}"
            );
            assert_eq!(
                left[1].to_bits(),
                right[1].to_bits(),
                "material v bits differ at pixel {index}"
            );
        }
    }
}
