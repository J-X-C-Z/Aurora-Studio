//! Static `PoseRefine` / `SubPoseRefine` vendor control tables.
//!
//! Every number below is copied from decoded vendor `effect.json` tables. This
//! module is deliberately data-only and pure: it neither reads files at runtime
//! nor invents a warp formula. The authoritative row-per-field table, including
//! the full PF406 radial factor sets and shoulder-scale duplicates, is
//! `BODY-GEOMETRY-PARAMS.tsv`.
//!
//! `ControlContent` can carry either three nodes (`params: [0, 0.5, 1]`) or two
//! nodes (`params: [0, 1]`). `ControlCurve::remap` is a piecewise-linear map in
//! *parameter* space; the caller still decides what the remapped value drives.
//! No demographic factor is applied here because the decoded values are 1.0 for
//! the only decoded factor-bearing rows.

use crate::Refusal;

/// One `(ControlContent.RenderParamRanges, ControlContent.RenderDegreeRanges)`
/// pair. Curves with three nodes use node 0/1/2; two-node curves use 0/1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlCurve {
    /// `RenderParamRanges`.
    pub params: [f32; 3],
    /// `RenderDegreeRanges`, in the same node order.
    pub degrees: [f32; 3],
    /// Number of valid nodes; 2 or 3.
    pub nodes: usize,
    /// `ControlContent.DefaultDegree`, expressed in the same *parameter* space
    /// as `params` (the vendor UI value, not the remapped degree).
    pub default_param: f32,
}

impl ControlCurve {
    /// Piecewise-linear vendor remap. Inputs outside the table are clamped to
    /// the endpoint value; non-finite input is refused.
    pub fn remap(&self, param: f32) -> Result<f32, Refusal> {
        if !param.is_finite() {
            return Err(Refusal::ValueOutOfRange("body control param"));
        }
        let (left, right) = match self.nodes {
            2 => (0, 1),
            3 if param <= self.params[1] => (0, 1),
            3 => (1, 2),
            _ => return Err(Refusal::ValueOutOfRange("body control nodes")),
        };
        let p0 = self.params[left];
        let p1 = self.params[right];
        if p1 <= p0 {
            return Err(Refusal::ValueOutOfRange("body control params"));
        }
        let t = ((param - p0) / (p1 - p0)).clamp(0.0, 1.0);
        Ok(self.degrees[left] + (self.degrees[right] - self.degrees[left]) * t)
    }

    /// `remap(default_param)` with the same contract.
    pub fn default_degree(&self) -> Result<f32, Refusal> {
        self.remap(self.default_param)
    }
}

/// The seven `PointFactor` / `OutCircleFactor` / `InnerCircleFactor` /
/// `EarFactor` values in one PF406 radial parameter set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialFactors {
    pub out_circle: f32,
    pub inner_circle: f32,
    pub ear: f32,
    pub points: [f32; 7],
}

/// PF406 `HeadZoom` factors. Both `LeftSideFaceParam` and
/// `RightSideFaceParam` are retained even though the decoded
/// `LeftSideFaceParam` is the only one named in the alignment gap report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadZoomParams {
    pub expand: f32,
    pub face_dist_threshold: f32,
    pub normal: RadialFactors,
    pub left_side_face: RadialFactors,
    pub right_side_face: RadialFactors,
    pub negative_normal: RadialFactors,
    pub negative_left_side_face: RadialFactors,
    pub negative_right_side_face: RadialFactors,
    /// `DegreeRange`, i.e. the UI slider domain; `EnableDegreeRemap` is true.
    pub ui_range: [f32; 2],
    /// `DefaultDegree` in the UI domain.
    pub default_ui: f32,
    /// UI-domain remap to `ParamRange` `params`.
    pub param_range: [f32; 3],
    /// UI-domain remap to `EffectRange` degrees.
    pub effect_range: [f32; 3],
}

/// PF407 `LengthenLegRefine` segment weights. No Rust warp law consumes these
/// yet; they are carried so a future column producer cannot silently invent
/// a 1.0/1.0 split.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LegFactors {
    pub thigh: f32,
    pub shank: f32,
    pub use_ba_channel: bool,
}

pub const LEG_FACTORS: LegFactors = LegFactors {
    thigh: 0.35,
    shank: 1.0,
    use_ba_channel: true,
};

/// `ShoulderWidth` has two `EffectName` entries with the same PFs 417/418 but
/// different `DegreesScale`/`RadiusScale` and opposite degree ranges. They
/// cannot be collapsed into one table row without losing vendor information.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShoulderWidthParams {
    pub degrees_scale: [f32; 3],
    pub radius_scale: [f32; 3],
    pub narrow: ControlCurve,
    pub widen: ControlCurve,
}

pub const SHOULDER_WIDTH_NARROW: ShoulderWidthParams = ShoulderWidthParams {
    degrees_scale: [0.6299999952316284, 0.5600000023841858, 0.5],
    radius_scale: [1.2000000476837158, 1.2000000476837158, 1.2000000476837158],
    narrow: ControlCurve {
        params: [0.0, 0.5, 1.0],
        degrees: [-0.4000000059604645, 0.0, 0.0],
        nodes: 3,
        default_param: 0.5,
    },
    widen: ControlCurve {
        params: [0.0, 0.5, 1.0],
        degrees: [0.0, 0.0, 0.4000000059604645],
        nodes: 3,
        default_param: 0.5,
    },
};

pub const SHOULDER_WIDTH_WIDEN: ShoulderWidthParams = ShoulderWidthParams {
    degrees_scale: [0.3400000035762787, 0.6000000238418579, 0.44999998807907104],
    radius_scale: [1.1799999475479126, 1.3200000524520874, 0.9399999976158142],
    narrow: SHOULDER_WIDTH_NARROW.narrow,
    widen: SHOULDER_WIDTH_NARROW.widen,
};

/// NeckLength (PF410) `CircleFactor` / `InnerCircleFactor` / `EarFactor` /
/// seven `PointFactor` values from the decoded vendor config.
pub const NECK_LENGTH_RADIAL: RadialFactors = RadialFactors {
    out_circle: 1.5,
    inner_circle: 1.3700000047683716,
    ear: 1.7599999904632568,
    points: [
        1.7999999523162842,
        1.7999999523162842,
        2.0999999046325684,
        2.5199999809265137,
        2.0999999046325684,
        1.7999999523162842,
        1.7999999523162842,
    ],
};

pub const HEAD_ZOOM: HeadZoomParams = HeadZoomParams {
    expand: 1.1699999570846558,
    face_dist_threshold: 0.06019999831914902,
    normal: RadialFactors {
        out_circle: 1.0099999904632568,
        inner_circle: 1.3600000143051147,
        ear: 0.800000011920929,
        points: [
            1.2000000476837158,
            1.600000023841858,
            1.7899999618530273,
            1.7899999618530273,
            1.7899999618530273,
            1.600000023841858,
            1.2000000476837158,
        ],
    },
    left_side_face: RadialFactors {
        out_circle: 1.5,
        inner_circle: 1.5099999904632568,
        ear: 1.1100000143051147,
        points: [
            1.350000023841858,
            1.1799999475479126,
            0.8999999761581421,
            0.5699999928474426,
            0.5899999737739563,
            0.6499999761581421,
            1.0499999523162842,
        ],
    },
    right_side_face: RadialFactors {
        out_circle: 1.8300000429153442,
        inner_circle: 1.75,
        ear: 1.1799999475479126,
        points: [
            0.949999988079071,
            0.7599999904632568,
            0.8999999761581421,
            0.5899999737739563,
            1.0700000524520874,
            1.2400000095367432,
            1.350000023841858,
        ],
    },
    negative_normal: RadialFactors {
        out_circle: 1.100000023841858,
        inner_circle: 1.0299999713897705,
        ear: 0.8399999737739563,
        points: [
            0.699999988079071,
            1.600000023841858,
            1.75,
            1.75,
            1.75,
            1.600000023841858,
            0.699999988079071,
        ],
    },
    negative_left_side_face: RadialFactors {
        out_circle: 1.590000033378601,
        inner_circle: 1.25,
        ear: 1.2300000190734863,
        points: [
            1.600000023841858,
            1.7000000476837158,
            1.350000023841858,
            1.1799999475479126,
            1.409999966621399,
            1.5499999523162842,
            1.4700000286102295,
        ],
    },
    negative_right_side_face: RadialFactors {
        out_circle: 1.6200000047683716,
        inner_circle: 1.2400000095367432,
        ear: 1.2300000190734863,
        points: [
            1.4700000286102295,
            1.5499999523162842,
            1.409999966621399,
            1.1799999475479126,
            1.409999966621399,
            1.5499999523162842,
            1.4700000286102295,
        ],
    },
    ui_range: [0.0, 100.0],
    default_ui: 50.0,
    param_range: [0.0, 0.5, 1.0],
    effect_range: [-0.800000011920929, 0.0, 0.800000011920929],
};

const fn three(params: [f32; 3], degrees: [f32; 3], default_param: f32) -> ControlCurve {
    ControlCurve { params, degrees, nodes: 3, default_param }
}

const fn two(degrees: [f32; 2], default_param: f32) -> ControlCurve {
    ControlCurve {
        params: [0.0, 1.0, 1.0],
        degrees: [degrees[0], degrees[1], 0.0],
        nodes: 2,
        default_param,
    }
}

/// Vendor degree curve for one ParamFlag, or `None` when the PF is not
/// present in the two decoded PoseRefine configs with a curve. PF406 has no
/// `ControlContent`; its top-level `ParamRange` -> `EffectRange` is returned
/// after normalizing `DefaultDegree=50` from `DegreeRange=[0,100]` to 0.5.
///
/// Duplicate `ShoulderWidth` entries both target 417/418; the first (narrow)
/// entry wins here and the widen entry is available as
/// [`SHOULDER_WIDTH_WIDEN`].
pub fn control_curve(pf: u32) -> Option<ControlCurve> {
    Some(match pf {
        400 => three([0.0, 0.5, 1.0], [-0.5, 0.0, 0.5], 0.5),
        401 => two([0.0, 0.6], 0.0),
        403 => three([0.0, 0.5, 1.0], [-0.5, 0.0, 0.5], 0.5),
        404 => two([0.0, 0.5], 0.0),
        405 => three([0.0, 0.5, 1.0], [-2.0, 0.0, 2.0], 0.5),
        // PF406 has no ControlContent; its top-level ParamRange -> EffectRange
        // is the same three-node curve after the 0..100 UI normalization.
        406 => three(HEAD_ZOOM.param_range, HEAD_ZOOM.effect_range, 0.5),
        407 => two([1.0, 1.255], 0.0),
        408 => two([0.0, 1.0], 0.0),
        410 => three([0.0, 0.5, 1.0], [-0.5, 0.0, 1.0], 0.5),
        411 | 412 | 413 | 414 => two([0.0, 0.55], 0.0),
        415 | 416 => two([-1.0, 1.0], 0.5),
        417 | 418 => SHOULDER_WIDTH_NARROW.narrow,
        30074 | 30075 => two([-1.0, 1.0], 0.5),
        30094 => two([0.0, 1.25], 1.0),
        30095 => two([0.0, 1.0], 1.0),
        38556 => two([0.0, 1.0], 0.0),
        _ => return None,
    })
}

/// `maxHeadSize` from the decoded effect containing this PF. PF406 is the
/// `HeadZoom` effect and has no `maxHeadSize` field; it uses
/// `face_dist_threshold` instead.
pub fn max_head_size(pf: u32) -> Option<f32> {
    Some(match pf {
        400 | 401 | 403 | 404 | 410 | 411 | 412 | 413 | 414 | 415 | 416 => 0.36,
        405 | 407 | 408 | 30094 | 30095 => 0.36,
        38556 => 0.360000014305115,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curves_match_decoded_endpoints_and_defaults() {
        for pf in [400, 401, 403, 404, 405, 407, 408, 410, 415, 416, 38556] {
            let curve = control_curve(pf).unwrap();
            assert!(curve.default_degree().is_ok(), "PF{pf}");
            assert_eq!(curve.remap(curve.params[0]), Ok(curve.degrees[0]), "PF{pf}");
            assert_eq!(curve.remap(curve.params[curve.nodes - 1]), Ok(curve.degrees[curve.nodes - 1]), "PF{pf}");
        }
        assert_eq!(control_curve(400).unwrap().remap(0.5), Ok(0.0));
        assert_eq!(control_curve(400).unwrap().remap(0.25), Ok(-0.25));
        assert_eq!(control_curve(405).unwrap().remap(0.25), Ok(-1.0));
        assert_eq!(control_curve(407).unwrap().remap(0.0), Ok(1.0));
        assert_eq!(control_curve(407).unwrap().remap(1.0), Ok(1.255));
        assert_eq!(control_curve(407).unwrap().default_param, 0.0);
        assert_eq!(control_curve(30094).unwrap().remap(1.0), Ok(1.25));
        assert_eq!(control_curve(30094).unwrap().default_degree(), Ok(1.25));
        assert_eq!(control_curve(30095).unwrap().default_degree(), Ok(1.0));
        assert_eq!(control_curve(410).unwrap().remap(1.0), Ok(1.0));
        assert_eq!(control_curve(411).unwrap().remap(1.0), Ok(0.55));
        assert_eq!(control_curve(413).unwrap().default_degree(), Ok(0.0));
        assert_eq!(control_curve(416).unwrap().remap(0.0), Ok(-1.0));
        assert_eq!(control_curve(38556).unwrap().remap(0.0), Ok(0.0));
        assert_eq!(control_curve(406).unwrap().remap(0.5), Ok(0.0));
        assert_eq!(control_curve(406).unwrap().remap(1.0), Ok(0.800000011920929));
        assert_eq!(control_curve(90041), None);
    }

    #[test]
    fn remap_clamps_and_refuses_non_finite() {
        let curve = control_curve(400).unwrap();
        assert_eq!(curve.remap(-1.0), Ok(-0.5));
        assert_eq!(curve.remap(2.0), Ok(0.5));
        assert_eq!(curve.remap(f32::NAN), Err(Refusal::ValueOutOfRange("body control param")));
    }

    #[test]
    fn radial_and_leg_factors_keep_vendor_numbers() {
        assert_eq!(HEAD_ZOOM.expand, 1.1699999570846558);
        assert_eq!(HEAD_ZOOM.face_dist_threshold, 0.06019999831914902);
        assert_eq!(HEAD_ZOOM.normal.points[0], 1.2000000476837158);
        assert_eq!(HEAD_ZOOM.normal.points[6], 1.2000000476837158);
        assert_eq!(HEAD_ZOOM.normal.points[3], 1.7899999618530273);
        assert_eq!(HEAD_ZOOM.left_side_face.points[3], 0.5699999928474426);
        assert_eq!(HEAD_ZOOM.right_side_face.out_circle, 1.8300000429153442);
        assert_eq!(HEAD_ZOOM.negative_normal.points[0], 0.699999988079071);
        assert_eq!(HEAD_ZOOM.ui_range, [0.0, 100.0]);
        assert_eq!(HEAD_ZOOM.default_ui, 50.0);
        assert_eq!(NECK_LENGTH_RADIAL.points[3], 2.5199999809265137);
        assert_eq!(LEG_FACTORS.thigh, 0.35);
        assert_eq!(LEG_FACTORS.shank, 1.0);
        assert!(LEG_FACTORS.use_ba_channel);
        assert_eq!(max_head_size(38556), Some(0.360000014305115));
        assert_eq!(max_head_size(406), None);
    }

    #[test]
    fn shoulder_width_entries_are_not_collapsed() {
        assert_eq!(SHOULDER_WIDTH_NARROW.narrow.degrees, [-0.4000000059604645, 0.0, 0.0]);
        assert_eq!(SHOULDER_WIDTH_WIDEN.widen.degrees, [0.0, 0.0, 0.4000000059604645]);
        assert_eq!(SHOULDER_WIDTH_NARROW.radius_scale[0], 1.2000000476837158);
        assert_eq!(SHOULDER_WIDTH_WIDEN.radius_scale[1], 1.3200000524520874);
        assert_eq!(control_curve(417), Some(SHOULDER_WIDTH_NARROW.narrow));
    }
}
