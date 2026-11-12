// Source: PixCake 0004 preset ParamFlags + shipped vendor UI key registry
// Source: decrypted effect.json configs (ParamFlag) and preset string-enum slots
// Confidence: mapping proves product-layer semantics only, NOT the response curve.

use crate::adjust::{AdjustmentSet, ToneChannel};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    AiRetouch,
    Background,
    BodyFaceSculpt,
    Clothing,
    ColorGrade,
    CoreRetouch,
    Eye,
    Geometry,
    Hair,
    LightShadow,
    Makeup,
    MakeupRemoval,
    MaterialStyle,
    SkinBodyV2,
    SkinRetouch,
    Teeth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExposureField {
    Exposure,
    Contrast,
    Brightness,
    Saturation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HdrField {
    Highlights,
    Shadows,
    Whites,
    Blacks,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LevelsField {
    InputBlack,
    InputWhite,
    Gamma,
    OutputBlack,
    OutputWhite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HslField {
    Hue,
    Saturation,
    Lightness,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SharpeningField {
    Amount,
    Radius,
    Threshold,
    HaloSuppression,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NoiseField {
    Luminance,
    Details,
    Color,
    SinglePixel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LensField {
    Distortion,
    LightFalloff,
    SharpnessFalloff,
    PurpleFringing,
    ShiftX,
    ShiftY,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GeometryField {
    Rotation,
    FlipHorizontal,
    FlipVertical,
    CropPositionOrSize,
    Keystone,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GrainField {
    Impact,
    Granularity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    Exposure { field: ExposureField },
    Hdr(HdrField),
    WhiteBalance { kelvin: bool },
    Levels { channel: ToneChannel, field: LevelsField },
    Curve { channel: ToneChannel },
    ColorEditor { sector: usize, field: HslField },
    Sharpening(SharpeningField),
    Noise(NoiseField),
    Lens(LensField),
    Geometry(GeometryField),
    Vignette,
    Grain(GrainField),
    Style(&'static str),
    /// A material/asset path selector: stored, but not a numeric adjustment.
    Asset(&'static str),
    Effect(&'static str),
    Unmapped(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct C1Param {
    pub id: i32,
    pub key: &'static str,
    pub family: Family,
    pub target: Target,
}

pub static C1_PARAMS: &[C1Param] = &[
    C1Param { id: 0, key: "spotless", family: Family::SkinRetouch, target: Target::Effect("product-layer SkinRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 1, key: "spotless_acne", family: Family::SkinRetouch, target: Target::Effect("product-layer SkinRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 2, key: "spotless_spot", family: Family::SkinRetouch, target: Target::Effect("product-layer SkinRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 3, key: "spotless_texture", family: Family::SkinRetouch, target: Target::Effect("product-layer SkinRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 5, key: "lightenPouch", family: Family::CoreRetouch, target: Target::Effect("product-layer CoreRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8, key: "shadowStereo", family: Family::Makeup, target: Target::Exposure { field: ExposureField::Contrast } },
    C1Param { id: 9, key: "highlight", family: Family::Makeup, target: Target::Hdr(HdrField::Highlights) },
    C1Param { id: 100, key: "eyebrowDegree_left", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 101, key: "silkwormDegree", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 103, key: "eyelinerDegree_upper", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 105, key: "eyelashDegree_upper", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 106, key: "blusherDegree", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 107, key: "lipstickDegree", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 108, key: "teethWhitening", family: Family::Teeth, target: Target::Effect("product-layer Teeth control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 109, key: "teethWhitening_brightness", family: Family::Teeth, target: Target::Effect("product-layer Teeth control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 110, key: "teethWhitening_deYellow", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 111, key: "eyePupilDegree", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 114, key: "", family: Family::Makeup, target: Target::Effect("effect-config ParamFlag in SymEyeLashColorMakeup; no UI semantic key") },
    C1Param { id: 115, key: "faceDecorationDegree", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 116, key: "contourDegree", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 117, key: "eyebrowDegree_right", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 118, key: "eyelashDegree_lower", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 119, key: "", family: Family::Makeup, target: Target::Effect("effect-config ParamFlag in SymEyeLashColorMakeup; no UI semantic key") },
    C1Param { id: 120, key: "silkwormVertical_left", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 121, key: "silkwormVertical_right", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 122, key: "silkwormHori_left", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 123, key: "silkwormHori_right", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 124, key: "silkwormSize_left", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 125, key: "silkwormSize_right", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 200, key: "", family: Family::BodyFaceSculpt, target: Target::Effect("effect-config ParamFlag in FacialRefine; no UI semantic key") },
    C1Param { id: 201, key: "", family: Family::Eye, target: Target::Effect("effect-config ParamFlag in EyeRefine; no UI semantic key") },
    C1Param { id: 202, key: "hairLine", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 203, key: "noseSize", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 204, key: "", family: Family::BodyFaceSculpt, target: Target::Effect("effect-config ParamFlag in FacialRefine; no UI semantic key") },
    C1Param { id: 205, key: "chinLength", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 206, key: "", family: Family::Eye, target: Target::Effect("effect-config ParamFlag in EyeRefine; no UI semantic key") },
    C1Param { id: 207, key: "", family: Family::Eye, target: Target::Effect("effect-config ParamFlag in EyeRefine; no UI semantic key") },
    C1Param { id: 208, key: "", family: Family::Eye, target: Target::Effect("effect-config ParamFlag in EyeRefine; no UI semantic key") },
    C1Param { id: 209, key: "noseHigher", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 210, key: "noseAlar", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 211, key: "noseBridge", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 212, key: "noseTip", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 213, key: "mouthSize", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 214, key: "mouthHigher", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 215, key: "upperLip", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 216, key: "lowerLip", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 217, key: "", family: Family::BodyFaceSculpt, target: Target::Effect("effect-config ParamFlag in FacialRefine; no UI semantic key") },
    C1Param { id: 218, key: "", family: Family::BodyFaceSculpt, target: Target::Effect("effect-config ParamFlag in FacialRefine; no UI semantic key") },
    C1Param { id: 219, key: "", family: Family::BodyFaceSculpt, target: Target::Effect("effect-config ParamFlag in FacialRefine; no UI semantic key") },
    C1Param { id: 220, key: "", family: Family::BodyFaceSculpt, target: Target::Effect("effect-config ParamFlag in FacialRefine; no UI semantic key") },
    C1Param { id: 221, key: "", family: Family::BodyFaceSculpt, target: Target::Effect("effect-config ParamFlag in FacialRefine; no UI semantic key") },
    C1Param { id: 222, key: "headNarrow", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 223, key: "", family: Family::BodyFaceSculpt, target: Target::Effect("effect-config ParamFlag in FacialRefine; no UI semantic key") },
    C1Param { id: 224, key: "mouthWidth", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 225, key: "", family: Family::Eye, target: Target::Effect("effect-config ParamFlag in EyeRefine; no UI semantic key") },
    C1Param { id: 226, key: "", family: Family::Eye, target: Target::Effect("effect-config ParamFlag in EyeRefine; no UI semantic key") },
    C1Param { id: 227, key: "", family: Family::BodyFaceSculpt, target: Target::Effect("effect-config ParamFlag in FacialRefine; no UI semantic key") },
    C1Param { id: 228, key: "", family: Family::BodyFaceSculpt, target: Target::Effect("effect-config ParamFlag in FacialRefine; no UI semantic key") },
    C1Param { id: 230, key: "smallFace", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 232, key: "skullEnhance", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 233, key: "mLip", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 234, key: "philtrum", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 235, key: "atrium", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 236, key: "lowerCourt", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 237, key: "chinIn", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 238, key: "mouthRotate", family: Family::BodyFaceSculpt, target: Target::Geometry(GeometryField::Rotation) },
    C1Param { id: 240, key: "gumlineAdjust_left", family: Family::Teeth, target: Target::Effect("product-layer Teeth control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 241, key: "gumlineAdjust_right", family: Family::Teeth, target: Target::Effect("product-layer Teeth control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 242, key: "sharpChin_left", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 243, key: "sharpChin_right", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 244, key: "mouthMoveHor", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 245, key: "noseMoveHor", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 247, key: "vFace_left", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 248, key: "vFace_right", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 249, key: "faceShapePreset_degree", family: Family::CoreRetouch, target: Target::Effect("product-layer CoreRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 400, key: "slimArm", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 401, key: "slimWaist", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 403, key: "enhanceBreast", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 404, key: "niceButtock", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 405, key: "slimBody", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 406, key: "shrinkHead", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 407, key: "legLengthen", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 408, key: "heighten", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 410, key: "neckLength", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 411, key: "rightShoulder_left", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 412, key: "rightShoulder_right", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 413, key: "swanNeck_left", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 414, key: "swanNeck_right", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 415, key: "upBodyRatio", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 416, key: "waistLengthen", family: Family::CoreRetouch, target: Target::Effect("product-layer CoreRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 417, key: "shoulderWidth_left", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 418, key: "shoulderWidth_right", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8001, key: "eyebrowThickness_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8002, key: "eyebrowThickness_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8003, key: "eyebrowPosition_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8004, key: "eyebrowPosition_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8005, key: "eyebrowDistance_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8006, key: "eyebrowDistance_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8007, key: "eyebrowAngle_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8008, key: "eyebrowAngle_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8009, key: "eyebrowShape_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8010, key: "eyebrowShape_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8011, key: "eyeZoom_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8012, key: "eyeZoom_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8013, key: "eyeHigher_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8014, key: "eyeHigher_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8015, key: "eyeDistance_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8016, key: "eyeDistance_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8017, key: "eyeAngle_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8018, key: "eyeAngle_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8019, key: "eyeWidth_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8020, key: "eyeWidth_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8021, key: "eyePosition_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8022, key: "eyePosition_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8023, key: "jaw_left", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8024, key: "jaw_right", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8025, key: "faceLift_left", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8026, key: "faceLift_right", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8027, key: "cheekBone_left", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8028, key: "cheekBone_right", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8029, key: "temple_left", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8030, key: "temple_right", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8031, key: "neckSlim_left", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8032, key: "neckSlim_right", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8033, key: "smileMouthCorner", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8034, key: "brightEye_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8035, key: "brightEye_white_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8036, key: "brightEye_pupil_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8037, key: "brightEye_highlight_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8038, key: "brightEye_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8039, key: "brightEye_white_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8040, key: "brightEye_pupil_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8041, key: "brightEye_highlight_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8042, key: "eyePupil_left", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8043, key: "eyePupilAngle_left", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8044, key: "eyePupilHorMove_left", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8045, key: "eyePupilVerMove_left", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8046, key: "eyePupil_right", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8047, key: "eyePupilAngle_right", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8048, key: "eyePupilHorMove_right", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8049, key: "eyePupilVerMove_right", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8050, key: "eyeSize_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8051, key: "eyeSize_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8052, key: "eyePupilMirror_left", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8053, key: "eyePupilMirror_right", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8057, key: "eyeMakeupSaturation_left", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8058, key: "eyeMakeupLuma_left", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8059, key: "eyeMakeupStereo_left", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8061, key: "eyeMakeupSaturation_right", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8062, key: "eyeMakeupLuma_right", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8063, key: "eyeMakeupStereo_right", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8065, key: "lipMakeupSaturation", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8066, key: "lipMakeupLuma", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8067, key: "lipMakeupStereo", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8068, key: "skinGloss", family: Family::SkinRetouch, target: Target::Effect("product-layer SkinRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 8069, key: "skinTexture", family: Family::Makeup, target: Target::Noise(NoiseField::Luminance) },
    C1Param { id: 8070, key: "skinClarity", family: Family::SkinRetouch, target: Target::Sharpening(SharpeningField::Amount) },
    C1Param { id: 21002, key: "tianchongMode", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21012, key: "quxiaci_dimian", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21013, key: "quxiaci_beijing", family: Family::Background, target: Target::Effect("product-layer Background control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21014, key: "kouxiangbaoliu", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21015, key: "skinColorHue", family: Family::SkinRetouch, target: Target::ColorEditor { sector: 0, field: HslField::Hue } },
    C1Param { id: 21016, key: "hairVolume_left", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21017, key: "hairVolume_right", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21020, key: "aiRemoveTattoo", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21022, key: "removeStubble_cleanShave", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21023, key: "removeStubble_stubble", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21024, key: "removeGlabellaLines", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21025, key: "removeMarionetteLines", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21026, key: "removeStubble", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21027, key: "yinyingButouming", family: Family::LightShadow, target: Target::Effect("product-layer LightShadow control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21028, key: "yinyingXuhua", family: Family::LightShadow, target: Target::Effect("product-layer LightShadow control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21029, key: "jianbian", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21030, key: "xuhuaFanwei", family: Family::SkinBodyV2, target: Target::Noise(NoiseField::Luminance) },
    C1Param { id: 21031, key: "yinyingBlendMode", family: Family::LightShadow, target: Target::Effect("product-layer LightShadow control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21032, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 21033, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 21034, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 21035, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 21036, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 21037, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 21038, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 21039, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 21040, key: "yinyingSucaiPath", family: Family::LightShadow, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 21042, key: "aiYinyingSwitch", family: Family::LightShadow, target: Target::Effect("product-layer LightShadow control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21043, key: "aiYinyingChengdu", family: Family::LightShadow, target: Target::Effect("product-layer LightShadow control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21044, key: "guangyingjunyun_chengdu", family: Family::LightShadow, target: Target::Sharpening(SharpeningField::Amount) },
    C1Param { id: 21047, key: "aiYinyingBlendMode", family: Family::LightShadow, target: Target::Effect("product-layer LightShadow control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21048, key: "guangyingjunyun_mingdu", family: Family::LightShadow, target: Target::Exposure { field: ExposureField::Brightness } },
    C1Param { id: 21049, key: "removeTearTrough_left", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21050, key: "removeTearTrough_right", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21052, key: "guangyingjunyun", family: Family::LightShadow, target: Target::Effect("product-layer LightShadow control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21054, key: "redEyeRemoval", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21055, key: "CLOTHES_BLEMISH_REMOVAL", family: Family::Clothing, target: Target::Effect("product-layer Clothing control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21056, key: "CLOTHES_EDGE_SMOOTH_PROTECT_THING", family: Family::Clothing, target: Target::Effect("product-layer Clothing control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21069, key: "removeCheekWrinkles", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21070, key: "removeNasalWrinkles", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21071, key: "removePerioralWrinkles", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21075, key: "removeSpill", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21076, key: "smileDegree_closeMouth", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21077, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 21078, key: "", family: Family::Eye, target: Target::Effect("effect-config ParamFlag in AnimalBrightEyeNormalWithPupil; no UI semantic key") },
    C1Param { id: 21079, key: "", family: Family::Eye, target: Target::Effect("effect-config ParamFlag in AnimalBrightEyeNormalWithPupil; no UI semantic key") },
    C1Param { id: 21080, key: "", family: Family::Eye, target: Target::Effect("effect-config ParamFlag in AnimalBrightEyeNormalWithPupil; no UI semantic key") },
    C1Param { id: 21081, key: "", family: Family::Eye, target: Target::Effect("effect-config ParamFlag in AnimalBrightEyeNormalWithPupil; no UI semantic key") },
    C1Param { id: 21082, key: "", family: Family::Makeup, target: Target::Effect("effect-config ParamFlag in AnimalEyePupilMakeupAdjustMaterial; no UI semantic key") },
    C1Param { id: 21083, key: "", family: Family::Makeup, target: Target::Effect("effect-config ParamFlag in AnimalEyePupilMakeupAdjustMaterial; no UI semantic key") },
    C1Param { id: 21084, key: "", family: Family::Makeup, target: Target::Effect("effect-config ParamFlag in AnimalEyePupilMakeupAdjustMaterial; no UI semantic key") },
    C1Param { id: 21085, key: "", family: Family::Makeup, target: Target::Effect("effect-config ParamFlag in AnimalEyePupilMakeupAdjustMaterial; no UI semantic key") },
    C1Param { id: 21086, key: "", family: Family::Makeup, target: Target::Effect("effect-config ParamFlag in AnimalEyePupilMakeupAdjustMaterial; no UI semantic key") },
    C1Param { id: 21087, key: "", family: Family::Makeup, target: Target::Effect("effect-config ParamFlag in AnimalEyePupilMakeupAdjustMaterial; no UI semantic key") },
    C1Param { id: 21088, key: "", family: Family::Makeup, target: Target::Effect("effect-config ParamFlag in AnimalEyePupilMakeupAdjustMaterial; no UI semantic key") },
    C1Param { id: 21089, key: "", family: Family::Makeup, target: Target::Effect("effect-config ParamFlag in AnimalEyePupilMakeupAdjustMaterial; no UI semantic key") },
    C1Param { id: 21090, key: "", family: Family::Makeup, target: Target::Effect("effect-config ParamFlag in AnimalEyePupilMakeupAdjustMaterial; no UI semantic key") },
    C1Param { id: 21091, key: "", family: Family::Makeup, target: Target::Effect("effect-config ParamFlag in AnimalEyePupilMakeupAdjustMaterial; no UI semantic key") },
    C1Param { id: 21092, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 21093, key: "irisEditingSwitch", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21094, key: "irisEditingDirectSwitch", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21095, key: "irisEditingLeftHorizontal", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21096, key: "irisEditingLeftVertical", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21097, key: "irisEditingLeftScale", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21098, key: "irisEditingRightHorizontal", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21099, key: "irisEditingRightVertical", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21100, key: "irisEditingRightScale", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21104, key: "removeNoseFlaw", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 21105, key: "skinUnifyColorPicker", family: Family::SkinRetouch, target: Target::ColorEditor { sector: 0, field: HslField::Hue } },
    C1Param { id: 21106, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 21107, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 21109, key: "", family: Family::Makeup, target: Target::Effect("effect-config ParamFlag in AnimalEyePupilMakeupAdjustMaterial; no UI semantic key") },
    C1Param { id: 21110, key: "", family: Family::Makeup, target: Target::Effect("effect-config ParamFlag in AnimalEyePupilMakeupAdjustMaterial; no UI semantic key") },
    C1Param { id: 21135, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 21136, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 21138, key: "CLOTHES_SWEAT_STAIN_REMOVAL", family: Family::Clothing, target: Target::Effect("product-layer Clothing control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30003, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 30004, key: "blusherColor", family: Family::ColorGrade, target: Target::ColorEditor { sector: 0, field: HslField::Hue } },
    C1Param { id: 30005, key: "eyelashColor", family: Family::Eye, target: Target::ColorEditor { sector: 0, field: HslField::Hue } },
    C1Param { id: 30006, key: "eyebrowColor", family: Family::Eye, target: Target::ColorEditor { sector: 0, field: HslField::Hue } },
    C1Param { id: 30007, key: "eyelinerDegree_lower", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30008, key: "eyelinerColor", family: Family::Eye, target: Target::ColorEditor { sector: 0, field: HslField::Hue } },
    C1Param { id: 30009, key: "hairColorUnify", family: Family::Hair, target: Target::ColorEditor { sector: 0, field: HslField::Hue } },
    C1Param { id: 30010, key: "faceShaping3D_upDown", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30011, key: "faceShaping3D_tilt", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30012, key: "faceShaping3D_turnAround", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30013, key: "neutralGrayFlattening", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30014, key: "neutralGrayStereo", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30015, key: "neutralGrayStereo_facial", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30016, key: "neutralGrayStereo_contour", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30017, key: "poreEnhance", family: Family::SkinRetouch, target: Target::Sharpening(SharpeningField::Amount) },
    C1Param { id: 30018, key: "chunsebgquchuanbang", family: Family::Background, target: Target::Effect("product-layer Background control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30021, key: "freqSplit_face", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30022, key: "freqSplit_body", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30023, key: "removeDoubleEyelids_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30025, key: "teethBlemishRemoval", family: Family::Teeth, target: Target::Effect("product-layer Teeth control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30026, key: "teethAlignment", family: Family::Teeth, target: Target::Effect("product-layer Teeth control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30027, key: "teethBlemishRemoval_edge", family: Family::Teeth, target: Target::Effect("product-layer Teeth control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30028, key: "faceShapePreset_path", family: Family::SkinBodyV2, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 30029, key: "aiBodyReshape_bodySmooth", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30030, key: "removeVein", family: Family::SkinRetouch, target: Target::Effect("product-layer SkinRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30031, key: "freqSplit_face_high", family: Family::SkinBodyV2, target: Target::Sharpening(SharpeningField::Amount) },
    C1Param { id: 30032, key: "freqSplit_face_low", family: Family::SkinBodyV2, target: Target::Noise(NoiseField::Luminance) },
    C1Param { id: 30033, key: "freqSplit_body_high", family: Family::SkinBodyV2, target: Target::Sharpening(SharpeningField::Amount) },
    C1Param { id: 30034, key: "freqSplit_body_low", family: Family::SkinBodyV2, target: Target::Noise(NoiseField::Luminance) },
    C1Param { id: 30039, key: "removeDoubleEyelids_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30040, key: "lipstickColor", family: Family::ColorGrade, target: Target::ColorEditor { sector: 0, field: HslField::Hue } },
    C1Param { id: 30059, key: "skinToneUnifyMulti", family: Family::SkinRetouch, target: Target::Effect("product-layer SkinRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30071, key: "eyeInnerCorner_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30072, key: "eyeOuterCorner_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30073, key: "eyelidUnder_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30074, key: "slimLeg_thigh", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30075, key: "slimLeg_calf", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30076, key: "smileMaterialPath", family: Family::SkinBodyV2, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 30077, key: "eyeInnerCorner_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30078, key: "eyeOuterCorner_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30079, key: "eyelidUnder_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30080, key: "foreheadWidth_left", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30081, key: "foreheadWidth_right", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30082, key: "foreheadHeight_left", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30083, key: "foreheadHeight_right", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30092, key: "faceWidth_left", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30093, key: "faceWidth_right", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30094, key: "legLengthen_thigh", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30095, key: "legLengthen_calf", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30096, key: "sediao", family: Family::ColorGrade, target: Target::WhiteBalance { kelvin: false } },
    C1Param { id: 30187, key: "handNeutralGray", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30188, key: "removeDoubleChin_degree", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30189, key: "removeDoubleChin_shadow", family: Family::LightShadow, target: Target::Effect("product-layer LightShadow control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30190, key: "handReshape", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30191, key: "foreheadPlump", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30192, key: "foreheadPlump_flatten", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30193, key: "foreheadPlump_brighten", family: Family::SkinBodyV2, target: Target::ColorEditor { sector: 0, field: HslField::Lightness } },
    C1Param { id: 30194, key: "midFacePlump", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30195, key: "midFacePlump_flatten", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30196, key: "midFacePlump_brighten", family: Family::SkinBodyV2, target: Target::ColorEditor { sector: 0, field: HslField::Lightness } },
    C1Param { id: 30197, key: "cheekPlump", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30198, key: "nasalBasePlump", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30199, key: "nasalBasePlump_flatten", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30200, key: "nasalBasePlump_brighten", family: Family::SkinBodyV2, target: Target::ColorEditor { sector: 0, field: HslField::Lightness } },
    C1Param { id: 30201, key: "eyeSocketPlump_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30202, key: "eyeSocketPlump_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30203, key: "geisonPlump", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30204, key: "chinPlump", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30205, key: "chinPlump_flatten", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30206, key: "chinPlump_brighten", family: Family::SkinBodyV2, target: Target::ColorEditor { sector: 0, field: HslField::Lightness } },
    C1Param { id: 30207, key: "mouthAngulusPlump", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30208, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 30209, key: "jawlineEnhance", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30210, key: "fullFacePlump", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30258, key: "facialLiftEyeLiquifyDegree", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 30259, key: "removeSweat", family: Family::Clothing, target: Target::Effect("product-layer Clothing control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 31561, key: "facialLiftMouthDegree", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 31655, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 31933, key: "facialLiftMidFaceDegree", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 32035, key: "mouthJawReshapeContour", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 32729, key: "facialLiftMidFaceSmoothDegree", family: Family::SkinBodyV2, target: Target::Noise(NoiseField::Luminance) },
    C1Param { id: 33070, key: "facialLiftMouthSmoothDegree", family: Family::SkinBodyV2, target: Target::Noise(NoiseField::Luminance) },
    C1Param { id: 33149, key: "facialLiftMidFaceLiquifyDegree", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 34241, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 34583, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 34678, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 34722, key: "facialLiftEyeDegree", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 34794, key: "facialLiftContourDegree", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 35040, key: "shuipingFanzhuan", family: Family::Geometry, target: Target::Geometry(GeometryField::FlipHorizontal) },
    C1Param { id: 35470, key: "facialLiftEyeSmoothDegree", family: Family::Eye, target: Target::Noise(NoiseField::Luminance) },
    C1Param { id: 35846, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 36671, key: "mouthJawReshape", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 36965, key: "tiankongJianbian", family: Family::Background, target: Target::Effect("product-layer Background control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 37595, key: "mouthJawReshapeShadow", family: Family::LightShadow, target: Target::Effect("product-layer LightShadow control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 37926, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 38556, key: "increaseBreast", family: Family::BodyFaceSculpt, target: Target::Effect("product-layer BodyFaceSculpt control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 39129, key: "", family: Family::SkinBodyV2, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 39753, key: "facialLiftMouthLiquifyDegree", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 39798, key: "xuanzhuanJiaodu", family: Family::SkinBodyV2, target: Target::Geometry(GeometryField::Rotation) },
    C1Param { id: 39920, key: "facialLiftTotalDegree", family: Family::SkinBodyV2, target: Target::Effect("product-layer SkinBodyV2 control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 40100, key: "makeupRemovalEnable", family: Family::MakeupRemoval, target: Target::Effect("product-layer MakeupRemoval control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 40101, key: "makeupRemovalBlusher", family: Family::MakeupRemoval, target: Target::Effect("product-layer MakeupRemoval control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 40102, key: "makeupRemovalSilkworm", family: Family::MakeupRemoval, target: Target::Effect("product-layer MakeupRemoval control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 40103, key: "makeupRemovalEyepupil", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 40104, key: "makeupRemovalEyelash", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 40105, key: "makeupRemovalEyebrow", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 40106, key: "makeupRemovalEyeliner", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 40107, key: "makeupRemovalEyeMakeup", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 40108, key: "makeupRemovalLipstick", family: Family::MakeupRemoval, target: Target::Effect("product-layer MakeupRemoval control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 40109, key: "makeupRemovalContourPositive", family: Family::MakeupRemoval, target: Target::Effect("product-layer MakeupRemoval control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 40110, key: "makeupRemovalContourNegative", family: Family::MakeupRemoval, target: Target::Effect("product-layer MakeupRemoval control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 40111, key: "makeupRemovalFacialDecoration", family: Family::MakeupRemoval, target: Target::Effect("product-layer MakeupRemoval control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 40112, key: "makeupRemovalContourDegree", family: Family::MakeupRemoval, target: Target::Effect("product-layer MakeupRemoval control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 40113, key: "makeupRemovalEyeDegree", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 40114, key: "makeupRemovalFullFaceDegree", family: Family::MakeupRemoval, target: Target::Effect("product-layer MakeupRemoval control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 40115, key: "makeupRemovalFullFaceEnable", family: Family::MakeupRemoval, target: Target::Effect("product-layer MakeupRemoval control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 41922, key: "eyebrowTailLength_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 44124, key: "eyebrowHeadLength_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 44297, key: "removeBlueVeins", family: Family::SkinRetouch, target: Target::Effect("product-layer SkinRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 44634, key: "eyeholeCornerSwitch", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 48019, key: "eyebrowHeadLength_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 48605, key: "", family: Family::BodyFaceSculpt, target: Target::Effect("effect-config ParamFlag in FacialRefine; no UI semantic key") },
    C1Param { id: 49221, key: "eyebrowTailLength_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 80001, key: "skinToneSelect_path", family: Family::SkinRetouch, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 80002, key: "skinToneSelect_degree", family: Family::SkinRetouch, target: Target::Effect("product-layer SkinRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 80003, key: "skinToneWarmth", family: Family::SkinRetouch, target: Target::Effect("product-layer SkinRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90001, key: "textureSmoothSkin", family: Family::SkinRetouch, target: Target::Sharpening(SharpeningField::Amount) },
    C1Param { id: 90002, key: "moistureSmoothSkin", family: Family::SkinRetouch, target: Target::Noise(NoiseField::Luminance) },
    C1Param { id: 90003, key: "skinColorUniform", family: Family::SkinRetouch, target: Target::ColorEditor { sector: 0, field: HslField::Hue } },
    C1Param { id: 90004, key: "skinRadiance", family: Family::SkinRetouch, target: Target::Effect("product-layer SkinRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90005, key: "skinWhitening", family: Family::SkinRetouch, target: Target::ColorEditor { sector: 0, field: HslField::Lightness } },
    C1Param { id: 90006, key: "beijingzengqiang", family: Family::Background, target: Target::Hdr(HdrField::Shadows) },
    C1Param { id: 90007, key: "skinRosy", family: Family::SkinRetouch, target: Target::ColorEditor { sector: 0, field: HslField::Lightness } },
    C1Param { id: 90008, key: "removeStretchMark", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90010, key: "moistureSmoothSkin_body", family: Family::SkinRetouch, target: Target::Noise(NoiseField::Luminance) },
    C1Param { id: 90013, key: "removeNevus", family: Family::SkinRetouch, target: Target::Effect("product-layer SkinRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90015, key: "removeNeckLine", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90019, key: "bodySpotless", family: Family::SkinRetouch, target: Target::Effect("product-layer SkinRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90020, key: "eyeMakeupEnhance_left", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90021, key: "lipMakeupEnhance", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90023, key: "removeEyeLines", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90024, key: "removeForeheadLines", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90032, key: "eyeMakeupDegree", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90033, key: "suitMakeupDegree", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90034, key: "removeDoubleChin", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90035, key: "chunsebgquxiaci", family: Family::Background, target: Target::Effect("product-layer Background control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90037, key: "removeStretchMark_line", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90038, key: "qusecaiduanceng", family: Family::MaterialStyle, target: Target::ColorEditor { sector: 0, field: HslField::Saturation } },
    C1Param { id: 90040, key: "teethRepair", family: Family::Teeth, target: Target::Effect("product-layer Teeth control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90041, key: "aiBodyReshape", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90042, key: "bianyuantiaozheng", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90043, key: "", family: Family::MaterialStyle, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 90044, key: "", family: Family::MaterialStyle, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 90046, key: "symmetryReshapeSwitch", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90047, key: "symmetryReshapeFace", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90048, key: "symmetryReshapeBody", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90049, key: "bianyuanGuodu", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90050, key: "sewen", family: Family::ColorGrade, target: Target::WhiteBalance { kelvin: false } },
    C1Param { id: 90051, key: "baohedu", family: Family::ColorGrade, target: Target::Exposure { field: ExposureField::Saturation } },
    C1Param { id: 90052, key: "liangdu", family: Family::ColorGrade, target: Target::Exposure { field: ExposureField::Brightness } },
    C1Param { id: 90053, key: "xuhua", family: Family::MaterialStyle, target: Target::Noise(NoiseField::Luminance) },
    C1Param { id: 90054, key: "jingwuSecai", family: Family::MaterialStyle, target: Target::ColorEditor { sector: 0, field: HslField::Saturation } },
    C1Param { id: 90055, key: "renwuSecai", family: Family::MaterialStyle, target: Target::ColorEditor { sector: 0, field: HslField::Saturation } },
    C1Param { id: 90056, key: "jingwuFanshe", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90057, key: "jingwuShuixiaoXuhua", family: Family::MaterialStyle, target: Target::Noise(NoiseField::Luminance) },
    C1Param { id: 90059, key: "", family: Family::MaterialStyle, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 90060, key: "butouming", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90061, key: "quxiaci_chengdu", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90080, key: "removeOilShine", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90085, key: "skinToneUnifyFace", family: Family::SkinRetouch, target: Target::Effect("product-layer SkinRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90086, key: "skinToneUnifyBody", family: Family::SkinRetouch, target: Target::Effect("product-layer SkinRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90087, key: "hairPartFill", family: Family::Hair, target: Target::Effect("product-layer Hair control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90088, key: "removeStrayHair", family: Family::Hair, target: Target::Effect("product-layer Hair control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90089, key: "bodySpotless_baby", family: Family::SkinRetouch, target: Target::Effect("product-layer SkinRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90090, key: "suitMakeupPath", family: Family::MaterialStyle, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 90091, key: "blusherPath", family: Family::MaterialStyle, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 90092, key: "eyebrowPath", family: Family::Eye, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 90093, key: "eyelashPath", family: Family::Eye, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 90094, key: "eyeMakeupPath", family: Family::Eye, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 90095, key: "lipstickPath", family: Family::MaterialStyle, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 90097, key: "eyePupilPath", family: Family::Eye, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 90099, key: "bgReplacePath", family: Family::Background, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 90100, key: "skyReplacePath", family: Family::Background, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 90110, key: "", family: Family::MaterialStyle, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 90111, key: "chuizhiWeizhi", family: Family::MaterialStyle, target: Target::Geometry(GeometryField::CropPositionOrSize) },
    C1Param { id: 90112, key: "shuipingWeizhi", family: Family::MaterialStyle, target: Target::Geometry(GeometryField::CropPositionOrSize) },
    C1Param { id: 90113, key: "removeStrayHair_person", family: Family::Hair, target: Target::Effect("product-layer Hair control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90114, key: "removeStrayHair_bg", family: Family::Hair, target: Target::Effect("product-layer Hair control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90115, key: "aiFacialSmooth", family: Family::MaterialStyle, target: Target::Noise(NoiseField::Luminance) },
    C1Param { id: 90116, key: "removeEyeBag", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90117, key: "removeEyeBag_keepSilkworm", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90118, key: "lipLineCorrection", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90119, key: "eyeMakeupEnhance_right", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90130, key: "glassesGlareRemoval", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90131, key: "removeWhiteHair", family: Family::Hair, target: Target::Effect("product-layer Hair control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90132, key: "hairColorSelect_path", family: Family::Hair, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 90133, key: "daxiao", family: Family::MaterialStyle, target: Target::Geometry(GeometryField::CropPositionOrSize) },
    C1Param { id: 90134, key: "chuizhiWeizhi", family: Family::MaterialStyle, target: Target::Geometry(GeometryField::CropPositionOrSize) },
    C1Param { id: 90135, key: "shuipingWeizhi", family: Family::MaterialStyle, target: Target::Geometry(GeometryField::CropPositionOrSize) },
    C1Param { id: 90136, key: "butouming", family: Family::MaterialStyle, target: Target::Effect("product-layer MaterialStyle control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90137, key: "hairColorSelect_degree", family: Family::Hair, target: Target::ColorEditor { sector: 0, field: HslField::Hue } },
    C1Param { id: 90139, key: "hairSmooth", family: Family::Hair, target: Target::Effect("product-layer Hair control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90140, key: "hairSmooth_strand", family: Family::Hair, target: Target::Effect("product-layer Hair control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90141, key: "CLOTHES_EDGE_SMOOTH", family: Family::Clothing, target: Target::Effect("product-layer Clothing control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90142, key: "eyelidsMaterialPath", family: Family::Eye, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 90143, key: "doubleEyelidsDegree_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90144, key: "doubleEyelidsDegree_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90145, key: "doubleEyelidsLength_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90146, key: "doubleEyelidsLength_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90147, key: "doubleEyelidsHeight_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90148, key: "doubleEyelidsHeight_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90149, key: "doubleEyelidsHeight_lateral_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90150, key: "eyebrowEnhance_left", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90151, key: "eyebrowEnhance_right", family: Family::Makeup, target: Target::Effect("product-layer Makeup control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90153, key: "doubleEyelidsHeight_lateral_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90154, key: "doubleEyelidsHeight_medial_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90155, key: "doubleEyelidsHeight_medial_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90156, key: "doubleEyelidsLength_lateral_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90157, key: "doubleEyelidsLength_lateral_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90158, key: "doubleEyelidsLength_medial_left", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90159, key: "doubleEyelidsLength_medial_right", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90160, key: "hairGloss", family: Family::Hair, target: Target::Effect("product-layer Hair control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90170, key: "eyeSymmetry", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90171, key: "eyeSymmetry_horizontal", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90172, key: "eyeSymmetry_size", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90173, key: "eyeSymmetry_size_combobox", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 90200, key: "bgColor", family: Family::Background, target: Target::Hdr(HdrField::Shadows) },
    C1Param { id: 90201, key: "bgBlendMode", family: Family::Background, target: Target::Effect("product-layer Background control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 100001, key: "", family: Family::AiRetouch, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 100002, key: "", family: Family::AiRetouch, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 100003, key: "", family: Family::AiRetouch, target: Target::Unmapped("absent from effect configs, vendor UI registry and preset string enums") },
    C1Param { id: 100004, key: "redBloodRemoval", family: Family::AiRetouch, target: Target::Effect("product-layer AiRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 100005, key: "eyeWhitePurify", family: Family::Eye, target: Target::Effect("product-layer Eye control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 100007, key: "smileDegree_openMouth", family: Family::AiRetouch, target: Target::Effect("product-layer AiRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 100008, key: "repairBreast", family: Family::AiRetouch, target: Target::Effect("product-layer AiRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 100010, key: "makeupProtection", family: Family::AiRetouch, target: Target::Effect("product-layer AiRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 100011, key: "arcProtection", family: Family::AiRetouch, target: Target::Effect("product-layer AiRetouch control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 100013, key: "REMOVE_CLOTHES_CREASE_TOP", family: Family::Clothing, target: Target::Effect("product-layer Clothing control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 100014, key: "REMOVE_CLOTHES_CREASE_TOP_COARSE", family: Family::Clothing, target: Target::Effect("product-layer Clothing control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 100015, key: "REMOVE_CLOTHES_CREASE_TOP_DETAIL", family: Family::Clothing, target: Target::Effect("product-layer Clothing control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 100016, key: "REMOVE_CLOTHES_CREASE_BOTTOM", family: Family::Clothing, target: Target::Effect("product-layer Clothing control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 100017, key: "REMOVE_CLOTHES_CREASE_BOTTOM_COARSE", family: Family::Clothing, target: Target::Effect("product-layer Clothing control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 100018, key: "REMOVE_CLOTHES_CREASE_BOTTOM_DETAIL", family: Family::Clothing, target: Target::Effect("product-layer Clothing control; AdjustmentSet has no matching field (semantic only, response curve unverified)") },
    C1Param { id: 900109, key: "eyeLightPath", family: Family::Eye, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 900110, key: "faceDecorationPath", family: Family::AiRetouch, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 900111, key: "contourPath", family: Family::AiRetouch, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 900112, key: "silkwormPath", family: Family::AiRetouch, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
    C1Param { id: 900113, key: "", family: Family::AiRetouch, target: Target::Effect("effect-config ParamFlag in built-in pipeline; no UI semantic key") },
    C1Param { id: 900114, key: "eyelinerPath", family: Family::Eye, target: Target::Asset("material/style asset selector (se); not a numeric adjustment") },
];

fn norm(value: f32, low: f32, high: f32) -> f32 {
    low + value.clamp(0.0, 1.0) * (high - low)
}

pub fn apply_params(set: &mut AdjustmentSet, params: &[(i32, f32)]) {
    for (id, value) in params {
        let value = *value;
        let Some(param) = C1_PARAMS.iter().find(|param| param.id == *id) else { continue };
        match param.target {
            Target::Exposure { field } => match field {
                ExposureField::Exposure => set.exposure.exposure = norm(value, -4.0, 4.0),
                ExposureField::Contrast => set.exposure.contrast = norm(value, -50.0, 50.0),
                ExposureField::Brightness => set.exposure.brightness = norm(value, -50.0, 50.0),
                ExposureField::Saturation => set.exposure.saturation = norm(value, -100.0, 100.0),
            },
            Target::Hdr(field) => {
                let slot = match field {
                    HdrField::Highlights => &mut set.hdr.highlights,
                    HdrField::Shadows => &mut set.hdr.shadows,
                    HdrField::Whites => &mut set.hdr.whites,
                    HdrField::Blacks => &mut set.hdr.blacks,
                };
                *slot = norm(value, -100.0, 100.0);
            }
            Target::WhiteBalance { kelvin } => {
                if kelvin {
                    set.white_balance.kelvin = norm(value, 2000.0, 50000.0);
                } else {
                    set.white_balance.tint = norm(value, -50.0, 50.0);
                }
            }
            Target::Levels { channel, field } => {
                let levels = set.levels.channel_mut(channel);
                match field {
                    LevelsField::InputBlack => levels.input_black = value.clamp(0.0, 1.0),
                    LevelsField::InputWhite => levels.input_white = value.clamp(0.0, 1.0),
                    LevelsField::Gamma => levels.gamma = norm(value, 0.1, 10.0),
                    LevelsField::OutputBlack => levels.output_black = value.clamp(0.0, 1.0),
                    LevelsField::OutputWhite => levels.output_white = value.clamp(0.0, 1.0),
                }
            }
            Target::Curve { channel } => {
                set.curves.channel_mut(channel).points[1].y = value.clamp(0.0, 1.0);
            }
            Target::ColorEditor { sector, field } => {
                if sector < set.color_editor.basic.len() {
                    let shift = &mut set.color_editor.basic[sector];
                    match field {
                        HslField::Hue => shift.hue = norm(value, -30.0, 30.0),
                        HslField::Saturation => shift.saturation = norm(value, -100.0, 100.0),
                        HslField::Lightness => shift.lightness = norm(value, -100.0, 100.0),
                    }
                }
            }
            Target::Sharpening(field) => {
                match field {
                    SharpeningField::Amount => set.sharpening.amount = norm(value, 0.0, 1000.0),
                    SharpeningField::Radius => set.sharpening.radius = norm(value, 0.2, 2.5),
                    SharpeningField::Threshold => set.sharpening.threshold = norm(value, 0.0, 10.0),
                    SharpeningField::HaloSuppression => set.sharpening.halo_suppression = norm(value, 0.0, 100.0),
                }
            }
            Target::Noise(field) => {
                let slot = match field {
                    NoiseField::Luminance => &mut set.noise_reduction.luminance,
                    NoiseField::Details => &mut set.noise_reduction.details,
                    NoiseField::Color => &mut set.noise_reduction.color,
                    NoiseField::SinglePixel => &mut set.noise_reduction.single_pixel,
                };
                *slot = norm(value, 0.0, 100.0);
            }
            Target::Lens(field) => {
                match field {
                    LensField::Distortion => set.lens.distortion = norm(value, 0.0, 100.0),
                    LensField::LightFalloff => set.lens.light_falloff = norm(value, 0.0, 100.0),
                    LensField::SharpnessFalloff => set.lens.sharpness_falloff = norm(value, 0.0, 100.0),
                    LensField::PurpleFringing => set.lens.purple_fringing = norm(value, 0.0, 100.0),
                    LensField::ShiftX => set.lens.shift_x = norm(value, -100.0, 100.0),
                    LensField::ShiftY => set.lens.shift_y = norm(value, -100.0, 100.0),
                }
            }
            Target::Geometry(field) => match field {
                GeometryField::Rotation => set.geometry.rotation = norm(value, -45.0, 45.0),
                GeometryField::FlipHorizontal => set.geometry.flip_horizontal = value >= 0.5,
                GeometryField::FlipVertical => set.geometry.flip_vertical = value >= 0.5,
                GeometryField::CropPositionOrSize | GeometryField::Keystone => {
                    set.geometry.crop.x = value.clamp(0.0, 1.0);
                    set.geometry.crop.width = (1.0 - value.clamp(0.0, 1.0)).max(0.0);
                }
            },
            Target::Vignette => set.vignetting.amount = norm(value, -4.0, 4.0),
            Target::Grain(field) => {
                match field {
                    GrainField::Impact => set.film_grain.impact = norm(value, 0.0, 100.0),
                    GrainField::Granularity => set.film_grain.granularity = norm(value, 0.0, 100.0),
                }
            }
            Target::Style(name) => set.style = Some(name.to_owned()),
            Target::Asset(_) => {}
            Target::Effect(_) => {}
            Target::Unmapped(_) => {}
        }
    }
}

pub fn coverage() -> (usize, usize) {
    (C1_PARAMS.iter().filter(|param| !matches!(param.target, Target::Unmapped(_))).count(), C1_PARAMS.len())
}

pub fn unmapped() -> Vec<i32> {
    C1_PARAMS.iter().filter(|param| matches!(param.target, Target::Unmapped(_))).map(|param| param.id).collect()
}

/// Whether `apply_params` reaches a real `AdjustmentSet` field for this id.
/// `Effect`, `Asset` and `Unmapped` only carry product-layer semantics.
pub fn writes_field(id: i32) -> bool {
    C1_PARAMS
        .iter()
        .find(|param| param.id == id)
        .is_some_and(|param| !matches!(param.target, Target::Effect(_) | Target::Unmapped(_) | Target::Asset(_)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_matches_source() {
        assert_eq!(coverage(), (452, 482));
        assert_eq!(unmapped().len(), 30);
        assert_eq!(C1_PARAMS.windows(2).filter(|pair| pair[0].id >= pair[1].id).count(), 0);
    }

    #[test]
    fn implementation_flag_matches_target_kind() {
        assert!(writes_field(90052));
        assert!(!writes_field(0));
        assert!(!writes_field(30028));
        assert!(!writes_field(21032));
        assert!(!writes_field(-1));
    }

    #[test]
    fn brightness_applies_and_clamps() {
        let mut set = AdjustmentSet::default();
        apply_params(&mut set, &[(90052, 1.5), (90052, -0.5)]);
        assert_eq!(set.exposure.brightness, -50.0);
        apply_params(&mut set, &[(90052, -0.5)]);
        assert_eq!(set.exposure.brightness, -50.0);
        apply_params(&mut set, &[(90052, 0.75)]);
        assert_eq!(set.exposure.brightness, 25.0);
    }

    #[test]
    fn unknown_id_leaves_state_untouched() {
        let mut set = AdjustmentSet::default();
        let before = set.clone();
        apply_params(&mut set, &[(i32::MAX, 1.0), (-7, 0.5), (3002, 1.0)]);
        assert_eq!(set, before);
    }
}
