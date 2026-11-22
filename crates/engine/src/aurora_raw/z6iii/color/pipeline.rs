//! Narrow CPU color chain: camera code values -> legacy LUT/matrix -> working RGB.
//! All inputs are explicit; no white balance or production defaults are applied here.

use super::camera_clut::CameraClut;
use super::icc_source::CameraPcs;
use super::legacy_lut::{build_legacy_lut_from_fcrv_with_encoding, Encoding, LegacyLutParams};
use super::working_rgb_table;
use super::{legacy_color_management_rgb16, normalize_rgb16};

pub struct LegacyColorChain {
    matrix: [[f32; 3]; 3],
    exposure_scale: f32,
    lut: Box<[u16; 65536]>,
    green_blue: Option<[Box<[u16; 65536]>; 2]>,
    sizes: [usize; 3],
    nodes: Vec<[u16; 3]>,
}

impl LegacyColorChain {
    /// `fcrv` supplies the film response, `camera_profile` the ProStandard camera ICC,
    /// `photo_profile` the ID14 Photo RGB profile (None selects ID10 BT.2020).
    pub fn new(
        matrix: [[f32; 3]; 3],
        exposure_scale: f32,
        tone: LegacyLutParams,
        fcrv: &[u8],
        camera_profile: &[u8],
        photo_profile: Option<&[u8]>,
        points: u8,
    ) -> Option<Self> {
        Self::new_with_encoding(matrix, exposure_scale, tone, fcrv, camera_profile, photo_profile, points, Encoding::PowerLaw)
    }

    pub fn new_with_encoding(
        matrix: [[f32; 3]; 3],
        exposure_scale: f32,
        tone: LegacyLutParams,
        fcrv: &[u8],
        camera_profile: &[u8],
        photo_profile: Option<&[u8]>,
        points: u8,
        encoding: Encoding,
    ) -> Option<Self> {
        Self::with_tables(
            matrix,
            exposure_scale,
            build_legacy_lut_from_fcrv_with_encoding(tone, fcrv, encoding)?,
            None,
            camera_profile,
            photo_profile,
            points,
        )
    }

    pub fn new_per_channel(
        matrix: [[f32; 3]; 3],
        exposure_scale: f32,
        [red, green, blue]: [LegacyLutParams; 3],
        fcrv: &[u8],
        camera_profile: &[u8],
        photo_profile: Option<&[u8]>,
        points: u8,
    ) -> Option<Self> {
        Self::new_per_channel_with_encoding(matrix, exposure_scale, [red, green, blue], fcrv, camera_profile, photo_profile, points, Encoding::PowerLaw)
    }

    pub fn new_per_channel_with_encoding(
        matrix: [[f32; 3]; 3],
        exposure_scale: f32,
        [red, green, blue]: [LegacyLutParams; 3],
        fcrv: &[u8],
        camera_profile: &[u8],
        photo_profile: Option<&[u8]>,
        points: u8,
        encoding: Encoding,
    ) -> Option<Self> {
        let lut = build_legacy_lut_from_fcrv_with_encoding(red, fcrv, encoding)?;
        let green_blue = [
            build_legacy_lut_from_fcrv_with_encoding(green, fcrv, encoding)?,
            build_legacy_lut_from_fcrv_with_encoding(blue, fcrv, encoding)?,
        ];
        Self::with_tables(matrix, exposure_scale, lut, Some(green_blue), camera_profile, photo_profile, points)
    }

    fn with_tables(
        matrix: [[f32; 3]; 3],
        exposure_scale: f32,
        lut: Box<[u16; 65536]>,
        green_blue: Option<[Box<[u16; 65536]>; 2]>,
        camera_profile: &[u8],
        photo_profile: Option<&[u8]>,
        points: u8,
    ) -> Option<Self> {
        if !matrix.iter().flatten().all(|value| value.is_finite())
            || !exposure_scale.is_finite()
            || exposure_scale <= 0.0
        {
            return None;
        }
        let camera = CameraPcs::from_prostandard(camera_profile)?;
        let table = working_rgb_table(&camera, photo_profile, points)?;
        let sizes = [usize::from(points); 3];
        let nodes: Vec<[u16; 3]> = table
            .chunks_exact(3)
            .map(|node| [node[0], node[1], node[2]])
            .collect();
        CameraClut::new(sizes, &nodes)?;
        Some(Self { matrix, exposure_scale, lut, green_blue, sizes, nodes })
    }

    /// Legacy RGB16 before the camera CLUT.
    /// `rgb` is finite camera code units after white balance, before the legacy matrix.
    pub fn apply_legacy(&self, rgb: [f32; 3]) -> Option<[u16; 3]> {
        if !rgb.into_iter().all(f32::is_finite) {
            return None;
        }
        let tables = match &self.green_blue {
            Some([green, blue]) => [self.lut.as_ref(), green.as_ref(), blue.as_ref()],
            None => [self.lut.as_ref(); 3],
        };
        Some(legacy_color_management_rgb16(
            rgb,
            &self.matrix,
            self.exposure_scale,
            Some(tables),
        ))
    }

    /// Camera CLUT tail for an already-computed legacy RGB16.
    pub fn legacy_to_working(
        &self,
        legacy: [u16; 3],
    ) -> Option<([u16; 3], [f32; 3], [f32; 3])> {
        let camera = normalize_rgb16(legacy);
        let working = self.camera_to_working(camera)?;
        Some((legacy, camera, working))
    }

    pub(crate) fn camera_to_working(&self, camera: [f32; 3]) -> Option<[f32; 3]> {
        CameraClut::new(self.sizes, &self.nodes)?.sample_type1(camera)
    }

    /// Returns legacy RGB16, normalized camera RGB, and encoded working RGB.
    pub fn apply(&self, rgb: [f32; 3]) -> Option<([u16; 3], [f32; 3], [f32; 3])> {
        let legacy = self.apply_legacy(rgb)?;
        self.legacy_to_working(legacy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone() -> LegacyLutParams {
        LegacyLutParams {
            pre_scale: 1.0,
            curve_black: 0.0,
            curve_white: 1.0,
            film_black: 0.0,
            film_white: 1.0,
            post_gain: 1.0,
            post_offset: 0.0,
        }
    }

    #[test]
    fn new_rejects_invalid_inputs() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(LegacyColorChain::new([[f32::NAN; 3]; 3], 1.0, tone(), &[], &[], None, 33).is_none());
        assert!(LegacyColorChain::new(identity, 0.0, tone(), &[], &[], None, 33).is_none());
        assert!(LegacyColorChain::new(identity, f32::INFINITY, tone(), &[], &[], None, 33).is_none());
        assert!(LegacyColorChain::new(identity, 1.0, tone(), &[], &[], None, 33).is_none());
    }

    #[test]
    fn synthetic_identity_chain_stages_in_order() {
        let lut: Box<[u16; 65536]> =
            (0..=65535_u32).map(|code| code as u16).collect::<Vec<_>>().try_into().unwrap();
        let nodes: Vec<[u16; 3]> = (0..8_u16)
            .map(|index| std::array::from_fn(|channel| ((index >> (2 - channel)) & 1) * 65535))
            .collect();
        let chain = LegacyColorChain {
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            exposure_scale: 1.0,
            lut,
            green_blue: None,
            sizes: [2, 2, 2],
            nodes,
        };
        assert_eq!(chain.apply([65535.0, 0.0, 65535.75]), Some(([65535, 0, 65535], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0])));
        let (legacy, camera, working) = chain.apply([32767.0, 0.0, 65535.0]).unwrap();
        assert_eq!(legacy, [32767, 0, 65535]);
        assert_eq!(camera.map(f32::to_bits), [0x3eff_ff00, 0, 0x3f80_0000]);
        assert!((0.49..0.51).contains(&working[0]) && working[1] == 0.0 && working[2] > 0.999);
        assert_eq!(chain.apply([f32::NAN, 0.0, 0.0]), None);
    }
}
