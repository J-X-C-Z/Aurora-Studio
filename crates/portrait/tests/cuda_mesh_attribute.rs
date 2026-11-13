#[cfg(feature = "external-fixtures")]
#[path = "support/mod.rs"]
mod external_fixtures;

// PF8033 attribute rasterizer on the GPU, bit-for-bit against the CPU.
//
// Last-writer order is reproduced by "highest covering triangle index wins",
// which is what the CPU's sequential overwrite leaves behind.

#[cfg(windows)]
mod attribute_gpu {
    use aurora_pixcake::cuda::CudaError;
    use aurora_pixcake::cuda_mesh_attribute::{MeshAttribute, RESOLVE_KERNEL, SELECT_KERNEL};
    use aurora_pixcake::kernels::mesh_attribute;
    use aurora_pixcake::Refusal;


    #[cfg(feature = "external-fixtures")]


    fn decode_faces() -> Vec<u32> {
        let bytes = super::external_fixtures::read("smile/triangles.u32le");
        assert_eq!(bytes.len(), 51228);
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[cfg(feature = "external-fixtures")]

    fn decode_mean() -> Vec<[f32; 2]> {
        let bytes = super::external_fixtures::read("smile/mean.f32le");
        assert_eq!(bytes.len(), 26472);
        bytes.chunks_exact(12)
            .map(|c| {
                [
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                    f32::from_le_bytes([c[4], c[5], c[6], c[7]]),
                ]
            })
            .collect()
    }

    fn gpu() -> Option<MeshAttribute> {
        match MeshAttribute::open() {
            Ok(gpu) => Some(gpu),
            Err(CudaError::Unavailable(reason)) => {
                eprintln!("SKIP_NO_NVIDIA_DRIVER: {reason}");
                assert!(
                    std::env::var_os("PIXCAKE_CUDA_REQUIRED").is_none(),
                    "CUDA required but unavailable: {reason}"
                );
                None
            }
            Err(broken) => panic!("NVIDIA path present but broken: {broken:?}"),
        }
    }

    /// Bit-for-bit comparison of both planes.
    fn assert_bit_identical(
        gpu: &MeshAttribute,
        width: u32,
        height: u32,
        dest: &[[f32; 2]],
        attr: &[[f32; 2]],
        faces: &[u32],
        label: &str,
    ) {
        let expected = mesh_attribute::rasterize(width, height, dest, attr, faces).unwrap();
        let (actual, dispatches) = gpu.rasterize(width, height, dest, attr, faces).expect("real GPU dispatch");
        assert_eq!(dispatches.len(), 2);
        assert_eq!(dispatches[0].kernel, SELECT_KERNEL);
        assert_eq!(dispatches[1].kernel, RESOLVE_KERNEL);
        assert_eq!(actual.writes.len(), expected.writes.len(), "{label}: write plane length");
        for (index, (left, right)) in expected.writes.iter().zip(&actual.writes).enumerate() {
            assert_eq!(left, right, "{label}: writes at pixel {index}");
        }
        for (index, (left, right)) in expected.values.iter().zip(&actual.values).enumerate() {
            assert_eq!(
                left[0].to_bits(),
                right[0].to_bits(),
                "{label}: dx bits at pixel {index} (cpu {left:?} gpu {right:?})"
            );
            assert_eq!(left[1].to_bits(), right[1].to_bits(), "{label}: dy bits at pixel {index}");
        }
    }

    #[test]
    #[cfg(feature = "external-fixtures")]
    fn real_mesh_matches_the_cpu_on_96x64() {
        let Some(gpu) = gpu() else { return };
        let mean = decode_mean();
        let faces = decode_faces();
        assert_eq!(mean.len(), 2206);
        assert_eq!(faces.len(), 4269 * 3);
        // Same explicit screen placement as the CPU test.
        let dest: Vec<[f32; 2]> = mean
            .iter()
            .map(|point| [(point[0] + 80.0) * 0.5, (80.0 - point[1]) * 0.3])
            .collect();
        let attr: Vec<[f32; 2]> = (0..mean.len())
            .map(|index| [index as f32 * 0.125, index as f32 * -0.0625])
            .collect();
        assert_bit_identical(&gpu, 96, 64, &dest, &attr, &faces, "96x64");
    }

    #[test]
    #[cfg(feature = "external-fixtures")]
    fn real_mesh_matches_the_cpu_on_512x512() {
        let Some(gpu) = gpu() else { return };
        let mean = decode_mean();
        let faces = decode_faces();
        let dest: Vec<[f32; 2]> = mean
            .iter()
            .map(|point| [(point[0] + 80.0) * 2.6, (80.0 - point[1]) * 1.6])
            .collect();
        let attr: Vec<[f32; 2]> = mean.iter().map(|point| [point[0] * 0.01, point[1] * -0.02]).collect();
        assert_bit_identical(&gpu, 512, 512, &dest, &attr, &faces, "512x512");
    }

    #[test]
    fn overlapping_triangles_keep_the_last_writer() {
        let Some(gpu) = gpu() else { return };
        // The same triangle twice: identical geometry, different attribute.
        let dest = [
            [0.0f32, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ];
        let first = [[1.0f32, 1.0], [1.0, 1.0], [1.0, 1.0]];
        let second = [[2.0f32, 2.0], [2.0, 2.0], [2.0, 2.0]];
        // The second copy of the same face wins everywhere.
        let mut attr = Vec::new();
        attr.extend_from_slice(&first);
        attr.extend_from_slice(&second);
        assert_bit_identical(&gpu, 10, 10, &dest, &attr, &[0, 1, 2, 3, 4, 5], "overlap");

        let (out, _) = gpu.rasterize(10, 10, &dest, &attr, &[0, 1, 2, 3, 4, 5]).unwrap();
        assert!(out.writes.iter().all(|writes| *writes <= 2));
        assert!(out.writes.iter().any(|writes| *writes == 2), "fixture must overlap");
        for (value, writes) in out.values.iter().zip(&out.writes) {
            if *writes > 0 {
                // Renormalised barycentrics need not sum to exactly 1.
                assert!(
                    (value[0] - 2.0).abs() < 1e-6 && (value[1] - 2.0).abs() < 1e-6,
                    "the later triangle must win, got {value:?}"
                );
            }
        }
    }

    #[test]
    fn uncovered_pixels_stay_zero_and_refusals_match() {
        let Some(gpu) = gpu() else { return };
        let dest = [[0.0f32, 0.0], [3.0, 0.0], [0.0, 3.0]];
        let attr = [[1.0f32, 1.0]; 3];
        let (out, _) = gpu.rasterize(16, 16, &dest, &attr, &[0, 1, 2]).unwrap();
        assert!(out.writes.iter().any(|writes| *writes == 0));
        for (value, writes) in out.values.iter().zip(&out.writes) {
            if *writes == 0 {
                assert_eq!(*value, [0.0, 0.0]);
            }
        }
        assert_eq!(
            gpu.rasterize(0, 4, &dest, &attr, &[0, 1, 2]).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("mesh attribute size")))
        );
        assert_eq!(
            gpu.rasterize(4, 4, &dest, &attr, &[0, 1]).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("mesh attribute faces")))
        );
        assert_eq!(
            gpu.rasterize(4, 4, &dest, &attr, &[0, 1, 9]).err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("mesh attribute face index")))
        );
    }
}
