//! Build Face2D draw bytes and material-UV bytes for the B3 makeup rows from
//! the same-job 194-point landmark array, without an external 3DMM.
//!
//! `points194` is `primary142 ++ secondary52` (R76 s01). The primary-142 roles
//! below come from the EXE's own static template (RVA `0x91bcd30`,
//! `primary142_template.json`), cross-checked in
//! `04_reference/legacy-root/逆向工程/像素蛋糕/analysis/v2-plane-semantics-v1`:
//!
//! ```text
//! 0..=8      left brow ring        9..=17     right brow ring
//! 18..=25    left eye lid ring     26..=35    left pupil/iris
//! 36..=43    right eye lid ring    44..=53    right pupil/iris
//! 54..=72    nose bridge & wings   73..=94    outer lip ring (22 pts)
//! 95..=106   inner lip ring (12)   107..=108  chin midline
//! 109..=141  jaw/chin U-contour
//! ```
//!
//! Whole-face `[95, 40, 901, 1081]` templates (`Blusher`, `Contour`,
//! `FacialDecoration`, `FaceStereo`) use a piecewise-affine triangular warp from
//! the canonical 142-point + derived forehead template to `points194[0..142]` so
//! cheek/contour textures follow head yaw and roll without leaking across the
//! nose bridge, eye sockets, or lips.

use std::sync::Arc;

use aurora_portrait::Refusal;

use crate::portrait_makeup_bundle::{
    B3_MAKEUP_ROWS, MakeupRowProfile, OwnedFace2dDrawBytes, OwnedFace2dUvBytes,
};

const POINTS194: usize = 194;
const PRIMARY142: usize = 142;
const FOREHEAD_ANCHORS: usize = 7;
const FACE_VERTS: usize = PRIMARY142 + FOREHEAD_ANCHORS;
const VERTEX_STRIDE: usize = 8;

pub(crate) const LANDMARKS_INVALID: &str = "makeup mesh2d landmarks";
pub(crate) const FRAME_INVALID: &str = "makeup mesh2d frame size";
pub(crate) const REGION_UNMAPPED: &str = "makeup mesh2d region";

const BROW_LEFT: [usize; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
const BROW_RIGHT: [usize; 9] = [9, 10, 11, 12, 13, 14, 15, 16, 17];
const EYE_LEFT_OUTLINE: [usize; 8] = [18, 19, 20, 21, 22, 23, 24, 25];
const EYE_RIGHT_OUTLINE: [usize; 8] = [36, 37, 38, 39, 40, 41, 42, 43];
const BROW_LEFT_REF: [usize; 5] = [0, 8, 7, 6, 5];
const BROW_RIGHT_REF: [usize; 5] = [9, 17, 16, 15, 14];
const IRIS_LEFT: [usize; 8] = [27, 28, 29, 30, 31, 32, 33, 34];
const IRIS_RIGHT: [usize; 8] = [45, 46, 47, 48, 49, 50, 51, 52];

// Verified against constants.json["template"]:
// 73 = left mouth corner, 78 = Cupid's bow center, 83 = right mouth corner, 89 = bottom center.
// 95 = left inner mouth corner, 98 = upper inner center, 101 = right inner corner, 104 = lower inner center.
const LIP_OUTER: [usize; 22] = [
    73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
];
const LIP_INNER: [usize; 12] = [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106];

/// Canonical `[95, 40, 901, 1081]` UV coordinates for `primary142` (`0..142`)
/// plus 7 derived forehead/temple anchors (`142..149`), aligned to the vendor
/// Makeups standard canvas (`X = 644.5 * (tx - 0.4945) + 503.5`, `Y = 642.0 * ty + 440.0`).
const FACE_TEMPLATE_149_UV: [[f32; 2]; FACE_VERTS] = [
    [0.179909, 0.416191], [0.219987, 0.382899], [0.272695, 0.376785], [0.326024, 0.382120],
    [0.375735, 0.394818], [0.369118, 0.421135], [0.322411, 0.412598], [0.273316, 0.407346],
    [0.224981, 0.409300], [0.729891, 0.416043], [0.688967, 0.382450], [0.635428, 0.376007],
    [0.581343, 0.381379], [0.531033, 0.394221], [0.537821, 0.420775], [0.585100, 0.411937],
    [0.634797, 0.406570], [0.683949, 0.408829], [0.233225, 0.473379], [0.263265, 0.457523],
    [0.301349, 0.452513], [0.338969, 0.460991], [0.366638, 0.481502], [0.332226, 0.485865],
    [0.296528, 0.488759], [0.262119, 0.484423], [0.299042, 0.470252], [0.268421, 0.469481],
    [0.276677, 0.451731], [0.297763, 0.443981], [0.319183, 0.451147], [0.328183, 0.468636],
    [0.319899, 0.486312], [0.298833, 0.494034], [0.277466, 0.486902], [0.298295, 0.469055],
    [0.675824, 0.473860], [0.645672, 0.457750], [0.607466, 0.452578], [0.569805, 0.461115],
    [0.542062, 0.481664], [0.576514, 0.486002], [0.612224, 0.488946], [0.646744, 0.484740],
    [0.610146, 0.470557], [0.641125, 0.469703], [0.632895, 0.451891], [0.611775, 0.444080],
    [0.590289, 0.451232], [0.581222, 0.468753], [0.589481, 0.486487], [0.610572, 0.494266],
    [0.632011, 0.487154], [0.611179, 0.469223], [0.453649, 0.454887], [0.453431, 0.507549],
    [0.453085, 0.559918], [0.452796, 0.612852], [0.453352, 0.666137], [0.404676, 0.471336],
    [0.397444, 0.509274], [0.390213, 0.547211], [0.382981, 0.585149], [0.352305, 0.625231],
    [0.375212, 0.651499], [0.402733, 0.645494], [0.503313, 0.471344], [0.510345, 0.509300],
    [0.517377, 0.547256], [0.524409, 0.585212], [0.555462, 0.625582], [0.532192, 0.651846],
    [0.504154, 0.645662], [0.317102, 0.743111], [0.337675, 0.733525], [0.361817, 0.726032],
    [0.388545, 0.719878], [0.417718, 0.716024], [0.453126, 0.721743], [0.488658, 0.715989],
    [0.518246, 0.719808], [0.545542, 0.725881], [0.570450, 0.733325], [0.591813, 0.743022],
    [0.575177, 0.760156], [0.558541, 0.777291], [0.535692, 0.789754], [0.512843, 0.802217],
    [0.482992, 0.806483], [0.453140, 0.810748], [0.423576, 0.806440], [0.394012, 0.802132],
    [0.371676, 0.789631], [0.349341, 0.777129], [0.333221, 0.760120], [0.333771, 0.744908],
    [0.371001, 0.742341], [0.417277, 0.742718], [0.453200, 0.745474], [0.489558, 0.742669],
    [0.536857, 0.742213], [0.575035, 0.744796], [0.539138, 0.758324], [0.497505, 0.767122],
    [0.453284, 0.771338], [0.409638, 0.767076], [0.368825, 0.758323], [0.453527, 0.859813],
    [0.453915, 0.908878], [0.100241, 0.464592], [0.099661, 0.504934], [0.099926, 0.545027],
    [0.101873, 0.585141], [0.106439, 0.625219], [0.113708, 0.665034], [0.124034, 0.704302],
    [0.138730, 0.742533], [0.158897, 0.778860], [0.184363, 0.812193], [0.214199, 0.842689],
    [0.247726, 0.870450], [0.283685, 0.895753], [0.320971, 0.919426], [0.360537, 0.939975],
    [0.405349, 0.953512], [0.454302, 0.957943], [0.503476, 0.953753], [0.548884, 0.940610],
    [0.589298, 0.920530], [0.627559, 0.897369], [0.664419, 0.872397], [0.698609, 0.844846],
    [0.729051, 0.814488], [0.754920, 0.781144], [0.775248, 0.744781], [0.790092, 0.706405],
    [0.800591, 0.666959], [0.807947, 0.626956], [0.812646, 0.586645], [0.814726, 0.546343],
    [0.814978, 0.506024], [0.813890, 0.465460], [0.117908, 0.365817], [0.179921, 0.297101],
    [0.272710, 0.229113], [0.453400, 0.232558], [0.635443, 0.228336], [0.729903, 0.296953],
    [0.795874, 0.366641],
];

/// Constrained anatomical triangulation over `FACE_TEMPLATE_149_UV` (CCW in Y-down,
/// leaving eye openings `18..=25` / `36..=43` and lip region `73..=94` open, and
/// keeping nose midline/walls separated from cheeks).
const FACE_MESH_TRIANGLES: [[u16; 3]; 161] = [
    [145, 144, 146], [145, 3, 144], [143, 1, 142], [62, 63, 24], [73, 115, 63],
    [145, 4, 3], [13, 4, 145], [147, 11, 146], [12, 145, 146], [12, 13, 145],
    [11, 12, 146], [2, 1, 143], [3, 2, 144], [2, 143, 144], [1, 0, 142],
    [63, 25, 24], [108, 126, 125], [124, 108, 125], [108, 124, 123], [85, 131, 130],
    [63, 114, 113], [115, 114, 63], [73, 118, 117], [118, 73, 119], [116, 73, 117],
    [73, 116, 115], [10, 147, 148], [10, 11, 147], [0, 109, 142], [25, 112, 111],
    [112, 63, 113], [112, 25, 63], [18, 111, 110], [18, 25, 111], [109, 18, 110],
    [18, 109, 0], [93, 120, 119], [121, 93, 92], [120, 93, 121], [127, 126, 108],
    [127, 107, 128], [107, 127, 108], [107, 108, 123], [122, 107, 123], [129, 85, 130],
    [69, 72, 57], [72, 58, 57], [132, 83, 133], [83, 132, 131], [84, 131, 85],
    [84, 83, 131], [4, 6, 3], [141, 9, 148], [9, 10, 148], [73, 94, 119],
    [94, 93, 119], [91, 107, 122], [91, 121, 92], [91, 122, 121], [129, 86, 85],
    [70, 72, 69], [70, 136, 83], [136, 70, 137], [42, 70, 69], [70, 138, 137],
    [58, 79, 78], [72, 79, 58], [77, 58, 78], [64, 77, 76], [136, 135, 83],
    [5, 6, 4], [6, 5, 21], [61, 62, 24], [23, 61, 24], [61, 23, 60],
    [16, 12, 11], [10, 16, 11], [139, 36, 140], [36, 141, 140], [36, 9, 141],
    [8, 18, 0], [8, 19, 18], [8, 0, 1], [7, 2, 3], [6, 7, 3],
    [8, 7, 19], [2, 7, 1], [7, 8, 1], [20, 6, 21], [20, 7, 6],
    [7, 20, 19], [107, 87, 128], [87, 129, 128], [87, 86, 129], [70, 71, 72],
    [71, 79, 72], [77, 65, 58], [65, 77, 64], [58, 65, 57], [65, 62, 57],
    [62, 65, 63], [65, 64, 63], [75, 64, 76], [83, 134, 133], [135, 134, 83],
    [62, 56, 57], [61, 56, 62], [56, 69, 57], [16, 15, 12], [15, 16, 38],
    [12, 15, 13], [43, 70, 42], [70, 43, 138], [138, 43, 139], [43, 36, 139],
    [36, 17, 9], [9, 17, 10], [17, 16, 10], [91, 90, 107], [90, 89, 107],
    [88, 87, 107], [89, 88, 107], [5, 22, 21], [59, 22, 5], [23, 22, 60],
    [22, 59, 60], [59, 55, 60], [55, 61, 60], [55, 56, 61], [56, 68, 69],
    [68, 42, 69], [68, 55, 67], [55, 68, 56], [15, 14, 13], [68, 41, 42],
    [40, 41, 67], [41, 68, 67], [17, 37, 16], [37, 17, 36], [16, 37, 38],
    [82, 81, 71], [82, 70, 83], [82, 71, 70], [71, 80, 79], [81, 80, 71],
    [54, 55, 59], [14, 54, 13], [54, 4, 13], [54, 5, 4], [54, 59, 5],
    [39, 15, 38], [39, 14, 15], [14, 39, 40], [75, 74, 64], [74, 73, 63],
    [64, 74, 63], [66, 14, 40], [66, 54, 14], [66, 40, 67], [55, 66, 67],
    [54, 66, 55],
];

/// Canonical `[350, 800, 300, 120]` UV coordinates for `LIP_OUTER` (`73..=94`) and
/// `LIP_INNER` (`95..=106`), matching `Lipstick/MULS*/Satin/staintexture.png`.
const LIP_TEMPLATE_34_UV: [[f32; 2]; 34] = [
    [0.102362, 0.197507], [0.164151, 0.111172], [0.236658, 0.043667], [0.316931, 0.000000],
    [0.404547, 0.000000], [0.510891, 0.004878], [0.617607, 0.000000], [0.706469, 0.000000],
    [0.788446, 0.042297], [0.863253, 0.109359], [0.927411, 0.196702], [0.877447, 0.351053],
    [0.827484, 0.505404], [0.758858, 0.617684], [0.690233, 0.729964], [0.600581, 0.768393],
    [0.510934, 0.806818], [0.422143, 0.768008], [0.333352, 0.729200], [0.266270, 0.616582],
    [0.199189, 0.503966], [0.150775, 0.350763], [0.152427, 0.213690], [0.264241, 0.190567],
    [0.403224, 0.193962], [0.511112, 0.218785], [0.620309, 0.193518], [0.762364, 0.189448],
    [0.877026, 0.212706], [0.769216, 0.334572], [0.644183, 0.413830], [0.511369, 0.451806],
    [0.380283, 0.413405], [0.257707, 0.334570],
];

/// Canonical `[200, 400, 300, 300]` UV coordinates for `eye_mask_left`
/// (`[outer, u1, u2, u3, inner, l3, l2, l1]`).
const EYE_RING_UV: [[f32; 2]; 8] = [
    [0.325, 0.503], [0.420, 0.453], [0.517, 0.435], [0.613, 0.457],
    [0.710, 0.513], [0.613, 0.567], [0.517, 0.583], [0.420, 0.560],
];

#[derive(Clone, Debug)]
pub(crate) struct RegionMesh {
    pub positions: Vec<[f32; 2]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u16>,
}

/// Build 2D draw bytes for one makeup row from 194 landmarks in source pixels.
pub(crate) fn build_face2d_draw_bytes(
    points194: &[[f32; 2]],
    frame_size: [u32; 2],
    profile: &MakeupRowProfile,
) -> Result<OwnedFace2dDrawBytes, Refusal> {
    let mesh = region_mesh(points194, frame_size, profile)?;
    let mut index_bytes = Vec::with_capacity(mesh.indices.len() * 2);
    for index in &mesh.indices {
        index_bytes.extend_from_slice(&index.to_le_bytes());
    }
    Ok(OwnedFace2dDrawBytes {
        vertices: encode_pairs(&mesh.positions),
        vertex_stride: VERTEX_STRIDE,
        position_offset: 0,
        indices_u16: Arc::from(index_bytes.into_boxed_slice()),
        draw_first_index: 0,
        draw_index_count: mesh.indices.len(),
    })
}

/// Build a material-UV plane matching [`build_face2d_draw_bytes`] vertex for
/// vertex. Every UV coordinate lies in `[0, 1]`.
pub(crate) fn build_region_uv(
    points194: &[[f32; 2]],
    frame_size: [u32; 2],
    profile: &MakeupRowProfile,
) -> Result<OwnedFace2dUvBytes, Refusal> {
    let mesh = region_mesh(points194, frame_size, profile)?;
    Ok(OwnedFace2dUvBytes {
        vertices: encode_pairs(&mesh.uvs),
        vertex_stride: VERTEX_STRIDE,
        offset: 0,
    })
}

pub(crate) fn region_mesh(
    points194: &[[f32; 2]],
    frame_size: [u32; 2],
    profile: &MakeupRowProfile,
) -> Result<RegionMesh, Refusal> {
    validate_contract(points194, frame_size)?;
    if !B3_MAKEUP_ROWS.contains(&profile.pf) {
        return Err(Refusal::UnknownEffect(profile.pf));
    }

    let semantic = profile.mask_semantic;
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    if semantic.contains("EyeBrow") {
        let (brow, left_anchor, right_anchor) = if profile.pf == 117 {
            (&BROW_RIGHT, 13usize, 9usize)
        } else {
            (&BROW_LEFT, 0usize, 4usize)
        };
        push_oriented_quad(
            points194,
            brow,
            left_anchor,
            right_anchor,
            frame_size,
            0.15,
            0.45,
            &mut positions,
            &mut uvs,
            &mut indices,
        )?;
    } else if semantic.contains("EyePupil") {
        push_iris_mesh(
            points194,
            26,
            &IRIS_LEFT,
            &EYE_LEFT_OUTLINE,
            frame_size,
            &mut positions,
            &mut uvs,
            &mut indices,
        )?;
        push_iris_mesh(
            points194,
            44,
            &IRIS_RIGHT,
            &EYE_RIGHT_OUTLINE,
            frame_size,
            &mut positions,
            &mut uvs,
            &mut indices,
        )?;
    } else if semantic.contains("EyeLash") || semantic.contains("EyeMask") {
        push_curved_eye_pair(
            points194,
            profile.pf,
            frame_size,
            &mut positions,
            &mut uvs,
            &mut indices,
        )?;
    } else if semantic.contains("Lip") {
        push_lip_annulus(points194, frame_size, &mut positions, &mut uvs, &mut indices)?;
    } else if semantic.contains("FaceStereo")
        || semantic.contains("Blusher")
        || semantic.contains("blusher")
    {
        push_face_template_mesh(points194, frame_size, &mut positions, &mut uvs, &mut indices)?;
    } else if semantic.contains("face-parsing") || semantic.contains("MASK_RGBA") {
        let all: Vec<usize> = (0..POINTS194).collect();
        push_hull_aabb(points194, &all, &mut positions, &mut uvs, &mut indices)?;
    } else {
        return Err(Refusal::MissingInput(REGION_UNMAPPED));
    }

    if indices.is_empty() || positions.len() != uvs.len() {
        return Err(Refusal::ShapeMismatch(REGION_UNMAPPED));
    }
    Ok(RegionMesh {
        positions,
        uvs,
        indices,
    })
}

fn validate_contract(points194: &[[f32; 2]], frame_size: [u32; 2]) -> Result<(), Refusal> {
    if points194.len() != POINTS194 {
        return Err(Refusal::ShapeMismatch(LANDMARKS_INVALID));
    }
    if frame_size[0] == 0 || frame_size[1] == 0 {
        return Err(Refusal::ShapeMismatch(FRAME_INVALID));
    }
    if points194.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange(LANDMARKS_INVALID));
    }
    Ok(())
}

fn clamp_ray_to_frame(
    anchor: [f32; 2],
    target: [f32; 2],
    anchor_uv: [f32; 2],
    target_uv: [f32; 2],
    frame_size: [u32; 2],
) -> ([f32; 2], [f32; 2]) {
    let fw = frame_size[0] as f32;
    let fh = frame_size[1] as f32;
    let a = [anchor[0].clamp(0.0, fw), anchor[1].clamp(0.0, fh)];
    let dx = target[0] - a[0];
    let dy = target[1] - a[1];
    let mut t = 1.0f32;
    if dx < -1e-6 {
        t = t.min(-a[0] / dx);
    } else if dx > 1e-6 {
        t = t.min((fw - a[0]) / dx);
    }
    if dy < -1e-6 {
        t = t.min(-a[1] / dy);
    } else if dy > 1e-6 {
        t = t.min((fh - a[1]) / dy);
    }
    t = t.clamp(0.0, 1.0);
    let pos = [
        (a[0] + t * dx).clamp(0.0, fw),
        (a[1] + t * dy).clamp(0.0, fh),
    ];
    let uv = [
        (anchor_uv[0] + t * (target_uv[0] - anchor_uv[0])).clamp(0.0, 1.0),
        (anchor_uv[1] + t * (target_uv[1] - anchor_uv[1])).clamp(0.0, 1.0),
    ];
    (pos, uv)
}

/// 142-point + derived-forehead piecewise-affine face mesh mapped to the
/// canonical `[95, 40, 901, 1081]` template. Back-facing / folded triangles
/// under steep yaw profiles are culled so the far cheek never overlaps onto the
/// nose bridge or visible cheek.
fn push_face_template_mesh(
    points194: &[[f32; 2]],
    frame_size: [u32; 2],
    vertices: &mut Vec<[f32; 2]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u16>,
) -> Result<(), Refusal> {
    let fw = frame_size[0] as f32;
    let fh = frame_size[1] as f32;
    let mut face_pts = Vec::with_capacity(FACE_VERTS);
    let mut face_uvs = Vec::with_capacity(FACE_VERTS);
    for i in 0..PRIMARY142 {
        let p = points194[i];
        face_pts.push([p[0].clamp(0.0, fw), p[1].clamp(0.0, fh)]);
        face_uvs.push(FACE_TEMPLATE_149_UV[i]);
    }

    let eye_mid = midpoint(points194[22], points194[40]);
    let up = [eye_mid[0] - points194[125][0], eye_mid[1] - points194[125][1]];
    let brow_c = midpoint(points194[4], points194[13]);
    let brow_c_uv = midpoint(FACE_TEMPLATE_149_UV[4], FACE_TEMPLATE_149_UV[13]);
    let forehead_specs: [(usize, [f32; 2], [f32; 2], f32, f32); FOREHEAD_ANCHORS] = [
        (109, points194[109], FACE_TEMPLATE_149_UV[109], 0.20, 0.05),
        (0, points194[0], FACE_TEMPLATE_149_UV[0], 0.25, 0.0),
        (2, points194[2], FACE_TEMPLATE_149_UV[2], 0.31, 0.0),
        (4, brow_c, brow_c_uv, 0.34, 0.0),
        (11, points194[11], FACE_TEMPLATE_149_UV[11], 0.31, 0.0),
        (9, points194[9], FACE_TEMPLATE_149_UV[9], 0.25, 0.0),
        (141, points194[141], FACE_TEMPLATE_149_UV[141], 0.20, 0.05),
    ];
    for (idx, (_, anchor, anchor_uv, up_k, inward_k)) in forehead_specs.iter().enumerate() {
        let target = [
            anchor[0] + up_k * up[0] + inward_k * (brow_c[0] - anchor[0]),
            anchor[1] + up_k * up[1] + inward_k * (brow_c[1] - anchor[1]),
        ];
        let target_uv = FACE_TEMPLATE_149_UV[PRIMARY142 + idx];
        let (pos, uv) = clamp_ray_to_frame(*anchor, target, *anchor_uv, target_uv, frame_size);
        face_pts.push(pos);
        face_uvs.push(uv);
    }

    if vertices.len() + FACE_VERTS > u16::MAX as usize {
        return Err(Refusal::ValueOutOfRange(REGION_UNMAPPED));
    }
    let base = vertices.len() as u16;
    vertices.extend_from_slice(&face_pts);
    uvs.extend_from_slice(&face_uvs);

    let face_det = cross(face_pts[109], face_pts[141], face_pts[125]);
    let face_sign = if face_det.abs() > 1e-3 {
        face_det.signum()
    } else {
        1.0
    };

    let start_len = indices.len();
    for &[a, b, c] in &FACE_MESH_TRIANGLES {
        let cr = cross(
            face_pts[a as usize],
            face_pts[b as usize],
            face_pts[c as usize],
        );
        if cr * face_sign > 1e-4 {
            indices.extend_from_slice(&[base + a, base + b, base + c]);
        }
    }

    if indices.len() == start_len {
        for &[a, b, c] in &FACE_MESH_TRIANGLES {
            let cr = cross(
                face_pts[a as usize],
                face_pts[b as usize],
                face_pts[c as usize],
            );
            if cr.abs() > 1e-5 {
                indices.extend_from_slice(&[base + a, base + b, base + c]);
            }
        }
    }

    if indices.len() == start_len {
        vertices.truncate(base as usize);
        uvs.truncate(base as usize);
        let all: Vec<usize> = (0..PRIMARY142).collect();
        return push_hull_aabb(points194, &all, vertices, uvs, indices);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct EyeFrame {
    ex: [f32; 2],
    up: [f32; 2],
    ew: f32,
    upper: [[f32; 2]; 5],
    lower: [[f32; 2]; 5],
    brow: [[f32; 2]; 5],
}

fn build_eye_frame(
    points194: &[[f32; 2]],
    eye_outline: &[usize; 8],
    brow_ref: &[usize; 5],
) -> EyeFrame {
    let outer = points194[eye_outline[0]];
    let inner = points194[eye_outline[4]];
    let dx = inner[0] - outer[0];
    let dy = inner[1] - outer[1];
    let ew_raw = dx.hypot(dy);
    let (ex, ew) = if ew_raw > 1e-3 && ew_raw.is_finite() {
        ([dx / ew_raw, dy / ew_raw], ew_raw)
    } else {
        ([1.0, 0.0], 2.0)
    };
    let perp = [-ex[1], ex[0]];
    let e_mid = midpoint(outer, inner);
    let mut b_mid = [0.0f32; 2];
    for &idx in brow_ref {
        b_mid[0] += points194[idx][0];
        b_mid[1] += points194[idx][1];
    }
    b_mid[0] /= brow_ref.len() as f32;
    b_mid[1] /= brow_ref.len() as f32;
    let to_brow = [b_mid[0] - e_mid[0], b_mid[1] - e_mid[1]];
    let up = if perp[0] * to_brow[0] + perp[1] * to_brow[1] >= 0.0 {
        perp
    } else {
        [-perp[0], -perp[1]]
    };
    EyeFrame {
        ex,
        up,
        ew,
        upper: [
            points194[eye_outline[0]],
            points194[eye_outline[1]],
            points194[eye_outline[2]],
            points194[eye_outline[3]],
            points194[eye_outline[4]],
        ],
        lower: [
            points194[eye_outline[0]],
            points194[eye_outline[7]],
            points194[eye_outline[6]],
            points194[eye_outline[5]],
            points194[eye_outline[4]],
        ],
        brow: brow_ref.map(|i| points194[i]),
    }
}

fn push_curved_eye_pair(
    points194: &[[f32; 2]],
    pf: u32,
    frame_size: [u32; 2],
    vertices: &mut Vec<[f32; 2]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u16>,
) -> Result<(), Refusal> {
    for (eye_outline, brow_ref) in [
        (&EYE_LEFT_OUTLINE, &BROW_LEFT_REF),
        (&EYE_RIGHT_OUTLINE, &BROW_RIGHT_REF),
    ] {
        let frame = build_eye_frame(points194, eye_outline, brow_ref);
        match pf {
            103 | 30007 | 105 => {
                push_upper_eye_ribbon(&frame, pf == 105, frame_size, vertices, uvs, indices)?;
            }
            118 => {
                push_lower_eye_ribbon(&frame, frame_size, vertices, uvs, indices)?;
            }
            90032 => {
                push_eyeshadow_annulus(&frame, frame_size, vertices, uvs, indices)?;
            }
            _ => return Err(Refusal::MissingInput(REGION_UNMAPPED)),
        }
    }
    Ok(())
}

fn clamp_pos(p: [f32; 2], frame_size: [u32; 2]) -> [f32; 2] {
    [
        p[0].clamp(0.0, frame_size[0] as f32),
        p[1].clamp(0.0, frame_size[1] as f32),
    ]
}

fn uv_step(base: [f32; 2], dex: f32, dup: f32) -> [f32; 2] {
    // 1.0 eye width in image ≈ 0.385 in canonical [200, 400, 300, 300] UV.
    const UV_SCALE: f32 = 0.385;
    [
        (base[0] + dex * UV_SCALE).clamp(0.0, 1.0),
        (base[1] - dup * UV_SCALE).clamp(0.0, 1.0),
    ]
}

fn push_upper_eye_ribbon(
    f: &EyeFrame,
    is_eyelash: bool,
    frame_size: [u32; 2],
    vertices: &mut Vec<[f32; 2]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u16>,
) -> Result<(), Refusal> {
    let wing_out = if is_eyelash { 0.34 } else { 0.48 };
    let wing_lift = if is_eyelash { 0.02 } else { 0.04 };
    let top_frac = if is_eyelash { 0.34 } else { 0.17 };
    let mid_frac = top_frac * 0.45;
    let bot_frac = if is_eyelash { -0.04 } else { -0.05 };

    let outer_wing = [
        f.upper[0][0] - f.ex[0] * wing_out * f.ew + f.up[0] * wing_lift * f.ew,
        f.upper[0][1] - f.ex[1] * wing_out * f.ew + f.up[1] * wing_lift * f.ew,
    ];
    let outer_wing_uv = uv_step(EYE_RING_UV[0], -wing_out, wing_lift);

    let inner_wing = [
        f.upper[4][0] + f.ex[0] * 0.14 * f.ew,
        f.upper[4][1] + f.ex[1] * 0.14 * f.ew,
    ];
    let inner_wing_uv = uv_step(EYE_RING_UV[4], 0.14, 0.0);

    let stations = [
        (outer_wing, outer_wing_uv, f.brow[0], 0.65),
        (f.upper[0], EYE_RING_UV[0], f.brow[0], 0.90),
        (f.upper[1], EYE_RING_UV[1], f.brow[1], 1.00),
        (f.upper[2], EYE_RING_UV[2], f.brow[2], 1.00),
        (f.upper[3], EYE_RING_UV[3], f.brow[3], 1.00),
        (f.upper[4], EYE_RING_UV[4], f.brow[4], 0.85),
        (inner_wing, inner_wing_uv, f.brow[4], 0.50),
    ];

    if vertices.len() + stations.len() * 3 > u16::MAX as usize {
        return Err(Refusal::ValueOutOfRange(REGION_UNMAPPED));
    }
    let base = vertices.len() as u16;
    for &(pt, uv0, brow_pt, taper) in &stations {
        let brow_gap = ((brow_pt[0] - pt[0]) * f.up[0] + (brow_pt[1] - pt[1]) * f.up[1])
            .clamp(0.08 * f.ew, 0.60 * f.ew);
        let h_top = (top_frac * taper * f.ew).min(brow_gap * 0.90);
        let h_mid = (mid_frac * taper * f.ew).min(h_top * 0.65);
        let h_bot = bot_frac * taper * f.ew;
        for h in [h_bot, h_mid, h_top] {
            let p = [pt[0] + f.up[0] * h, pt[1] + f.up[1] * h];
            vertices.push(clamp_pos(p, frame_size));
            uvs.push(uv_step(uv0, 0.0, h / f.ew));
        }
    }
    for s in 0..(stations.len() as u16 - 1) {
        for r in 0..2u16 {
            let i0 = base + s * 3 + r;
            let i1 = base + (s + 1) * 3 + r;
            let i2 = base + (s + 1) * 3 + (r + 1);
            let i3 = base + s * 3 + (r + 1);
            indices.extend_from_slice(&[i0, i1, i2, i0, i2, i3]);
        }
    }
    Ok(())
}

fn push_lower_eye_ribbon(
    f: &EyeFrame,
    frame_size: [u32; 2],
    vertices: &mut Vec<[f32; 2]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u16>,
) -> Result<(), Refusal> {
    let outer_wing = [
        f.lower[0][0] - f.ex[0] * 0.20 * f.ew,
        f.lower[0][1] - f.ex[1] * 0.20 * f.ew,
    ];
    let outer_uv = uv_step(EYE_RING_UV[0], -0.20, 0.0);
    let inner_wing = [
        f.lower[4][0] + f.ex[0] * 0.08 * f.ew,
        f.lower[4][1] + f.ex[1] * 0.08 * f.ew,
    ];
    let inner_uv = uv_step(EYE_RING_UV[4], 0.08, 0.0);

    let stations = [
        (outer_wing, outer_uv, 0.65f32),
        (f.lower[0], EYE_RING_UV[0], 0.90),
        (f.lower[1], EYE_RING_UV[7], 1.00),
        (f.lower[2], EYE_RING_UV[6], 1.00),
        (f.lower[3], EYE_RING_UV[5], 0.95),
        (f.lower[4], EYE_RING_UV[4], 0.70),
        (inner_wing, inner_uv, 0.45),
    ];

    if vertices.len() + stations.len() * 2 > u16::MAX as usize {
        return Err(Refusal::ValueOutOfRange(REGION_UNMAPPED));
    }
    let base = vertices.len() as u16;
    for &(pt, uv0, taper) in &stations {
        for h in [0.03 * taper * f.ew, -0.34 * taper * f.ew] {
            let p = [pt[0] + f.up[0] * h, pt[1] + f.up[1] * h];
            vertices.push(clamp_pos(p, frame_size));
            uvs.push(uv_step(uv0, 0.0, h / f.ew));
        }
    }
    for s in 0..(stations.len() as u16 - 1) {
        let i0 = base + s * 2;
        let i1 = base + (s + 1) * 2;
        let i2 = base + (s + 1) * 2 + 1;
        let i3 = base + s * 2 + 1;
        indices.extend_from_slice(&[i0, i1, i2, i0, i2, i3]);
    }
    Ok(())
}

fn push_eyeshadow_annulus(
    f: &EyeFrame,
    frame_size: [u32; 2],
    vertices: &mut Vec<[f32; 2]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u16>,
) -> Result<(), Refusal> {
    if vertices.len() + 16 > u16::MAX as usize {
        return Err(Refusal::ValueOutOfRange(REGION_UNMAPPED));
    }
    let base = vertices.len() as u16;

    // Inner loop (0..8) closely tracks the closed eye opening [18..=25] / [36..=43],
    // leaving the eyeball center unpainted.
    let inner_pts = [
        f.upper[0], f.upper[1], f.upper[2], f.upper[3],
        f.upper[4], f.lower[3], f.lower[2], f.lower[1],
    ];
    for i in 0..8 {
        vertices.push(clamp_pos(inner_pts[i], frame_size));
        uvs.push(EYE_RING_UV[i]);
    }

    // Outer loop (8..16) spans outer wing, under-brow upper socket, inner corner,
    // and lower lash shadow with independent left/right [0, 1]^2 UVs.
    let upper_outer = |lid: [f32; 2], brow: [f32; 2]| -> [f32; 2] {
        let gap = ((brow[0] - lid[0]) * f.up[0] + (brow[1] - lid[1]) * f.up[1])
            .clamp(0.18 * f.ew, 0.80 * f.ew);
        let h = gap.max(0.45 * f.ew) * 0.90;
        [lid[0] + f.up[0] * h, lid[1] + f.up[1] * h]
    };
    let lower_outer = |lid: [f32; 2], h_frac: f32| -> [f32; 2] {
        [
            lid[0] - f.up[0] * h_frac * f.ew,
            lid[1] - f.up[1] * h_frac * f.ew,
        ]
    };
    let outer_pts = [
        [
            f.upper[0][0] - f.ex[0] * 0.80 * f.ew + f.up[0] * 0.06 * f.ew,
            f.upper[0][1] - f.ex[1] * 0.80 * f.ew + f.up[1] * 0.06 * f.ew,
        ],
        upper_outer(f.upper[1], f.brow[1]),
        upper_outer(f.upper[2], f.brow[2]),
        upper_outer(f.upper[3], f.brow[3]),
        [
            f.upper[4][0] + f.ex[0] * 0.24 * f.ew,
            f.upper[4][1] + f.ex[1] * 0.24 * f.ew,
        ],
        lower_outer(f.lower[3], 0.48),
        lower_outer(f.lower[2], 0.54),
        lower_outer(f.lower[1], 0.52),
    ];
    let outer_uvs: [[f32; 2]; 8] = [
        [0.02, 0.50], [0.22, 0.14], [0.50, 0.10], [0.74, 0.14],
        [0.82, 0.52], [0.72, 0.76], [0.50, 0.79], [0.22, 0.76],
    ];
    for i in 0..8 {
        vertices.push(clamp_pos(outer_pts[i], frame_size));
        uvs.push(outer_uvs[i]);
    }

    for i in 0..8u16 {
        let next = (i + 1) % 8;
        let in0 = base + i;
        let in1 = base + next;
        let out0 = base + 8 + i;
        let out1 = base + 8 + next;
        indices.extend_from_slice(&[in0, out0, out1, in0, out1, in1]);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_iris_mesh(
    points194: &[[f32; 2]],
    center_idx: usize,
    ring: &[usize; 8],
    eye_outline: &[usize; 8],
    frame_size: [u32; 2],
    vertices: &mut Vec<[f32; 2]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u16>,
) -> Result<(), Refusal> {
    if vertices.len() + 9 > u16::MAX as usize {
        return Err(Refusal::ValueOutOfRange(REGION_UNMAPPED));
    }
    let fw = frame_size[0] as f32;
    let fh = frame_size[1] as f32;

    let mut ring_pts = [points194[ring[0]]; 8];
    let mut ring_mean = [0.0f32; 2];
    for (k, &idx) in ring.iter().enumerate() {
        ring_pts[k] = points194[idx];
        ring_mean[0] += ring_pts[k][0];
        ring_mean[1] += ring_pts[k][1];
    }
    ring_mean[0] /= 8.0;
    ring_mean[1] /= 8.0;

    let p_center = points194[center_idx];
    let (mut eye_min_x, mut eye_max_x) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut eye_min_y, mut eye_max_y) = (f32::INFINITY, f32::NEG_INFINITY);
    for &idx in eye_outline {
        eye_min_x = eye_min_x.min(points194[idx][0]);
        eye_max_x = eye_max_x.max(points194[idx][0]);
        eye_min_y = eye_min_y.min(points194[idx][1]);
        eye_max_y = eye_max_y.max(points194[idx][1]);
    }
    let center = if p_center[0] >= eye_min_x
        && p_center[0] <= eye_max_x
        && p_center[1] >= eye_min_y
        && p_center[1] <= eye_max_y
    {
        p_center
    } else {
        ring_mean
    };

    let mut avg_r = 0.0f32;
    for pt in &ring_pts {
        avg_r += (pt[0] - center[0]).hypot(pt[1] - center[1]);
    }
    let radius = (avg_r / 8.0).max(1.0);
    // mt.png irises sit inside normalized radius ~0.41 around center (0.5, 0.5).
    const IRIS_UV_RADIUS: f32 = 0.41;

    // Optional eyelid aperture Y-bounds along center's vertical axis when eye landmarks form a true eye box.
    let upper_mid = points194[eye_outline[2]];
    let lower_mid = points194[eye_outline[6]];
    let can_clamp_lids = center[0] > eye_min_x
        && center[0] < eye_max_x
        && upper_mid[1] < lower_mid[1]
        && (eye_max_x - eye_min_x) > 4.0;

    let base = vertices.len() as u16;
    vertices.push([center[0].clamp(0.0, fw), center[1].clamp(0.0, fh)]);
    uvs.push([0.5, 0.5]);

    for &pt in &ring_pts {
        let mut vx = pt[0];
        let mut vy = pt[1];
        if can_clamp_lids {
            let y_top = upper_mid[1].min(points194[eye_outline[1]][1]).min(points194[eye_outline[3]][1]);
            let y_bot = lower_mid[1].max(points194[eye_outline[7]][1]).max(points194[eye_outline[5]][1]);
            let dy = vy - center[1];
            let mut t = 1.0f32;
            if dy < -1e-4 && vy < y_top {
                t = ((y_top - center[1]) / dy).clamp(0.1, 1.0);
            } else if dy > 1e-4 && vy > y_bot {
                t = ((y_bot - center[1]) / dy).clamp(0.1, 1.0);
            }
            vx = center[0] + t * (pt[0] - center[0]);
            vy = center[1] + t * (pt[1] - center[1]);
        }
        let u = (0.5 + IRIS_UV_RADIUS * (vx - center[0]) / radius).clamp(0.0, 1.0);
        let v = (0.5 + IRIS_UV_RADIUS * (vy - center[1]) / radius).clamp(0.0, 1.0);
        vertices.push([vx.clamp(0.0, fw), vy.clamp(0.0, fh)]);
        uvs.push([u, v]);
    }

    for k in 0..8u16 {
        let next = (k + 1) % 8;
        indices.extend_from_slice(&[base, base + 1 + k, base + 1 + next]);
    }
    Ok(())
}

/// Emits a non-overlapping lip annulus between `LIP_OUTER` (`73..=94`) and
/// `LIP_INNER` (`95..=106`), leaving the inner mouth/teeth opening unrasterized.
fn push_lip_annulus(
    points194: &[[f32; 2]],
    frame_size: [u32; 2],
    vertices: &mut Vec<[f32; 2]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u16>,
) -> Result<(), Refusal> {
    let total = LIP_OUTER.len() + LIP_INNER.len();
    if vertices.len() + total > u16::MAX as usize {
        return Err(Refusal::ValueOutOfRange(REGION_UNMAPPED));
    }
    let fw = frame_size[0] as f32;
    let fh = frame_size[1] as f32;
    let base = vertices.len() as u16;
    for (slot, &landmark) in LIP_OUTER.iter().chain(&LIP_INNER).enumerate() {
        let p = points194[landmark];
        vertices.push([p[0].clamp(0.0, fw), p[1].clamp(0.0, fh)]);
        uvs.push(LIP_TEMPLATE_34_UV[slot]);
    }

    let push_strip = |outer: &[u16], inner: &[u16], indices: &mut Vec<u16>| {
        let mut i = 0usize;
        let mut j = 0usize;
        let no = outer.len();
        let ni = inner.len();
        while i < no - 1 || j < ni - 1 {
            let t_o = if i < no - 1 {
                (i + 1) as f32 / (no - 1) as f32
            } else {
                2.0
            };
            let t_i = if j < ni - 1 {
                (j + 1) as f32 / (ni - 1) as f32
            } else {
                2.0
            };
            if t_o <= t_i {
                indices.extend_from_slice(&[base + outer[i], base + outer[i + 1], base + inner[j]]);
                i += 1;
            } else {
                indices.extend_from_slice(&[base + outer[i], base + inner[j + 1], base + inner[j]]);
                j += 1;
            }
        }
    };

    // Upper lip: outer 73..=83 (local 0..=10) <-> inner 95..=101 (local 22..=28)
    let upper_outer: [u16; 11] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let upper_inner: [u16; 7] = [22, 23, 24, 25, 26, 27, 28];
    push_strip(&upper_outer, &upper_inner, indices);

    // Lower lip: outer 83..=94,73 (local 10..=21,0) <-> inner 101..=106,95 (local 28..=33,22)
    let lower_outer: [u16; 13] = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 0];
    let lower_inner: [u16; 7] = [28, 29, 30, 31, 32, 33, 22];
    push_strip(&lower_outer, &lower_inner, indices);

    Ok(())
}

/// Append one padded oriented quad (4 vertices, 2 triangles) aligned to the
/// `left_anchor -> right_anchor` axis so brow makeup textures rotate with head
/// roll and brow slope.
#[allow(clippy::too_many_arguments)]
fn push_oriented_quad(
    points194: &[[f32; 2]],
    landmarks: &[usize],
    left_anchor: usize,
    right_anchor: usize,
    frame_size: [u32; 2],
    pad_u_frac: f32,
    pad_v_frac: f32,
    vertices: &mut Vec<[f32; 2]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u16>,
) -> Result<(), Refusal> {
    if landmarks.is_empty() {
        return Err(Refusal::ShapeMismatch(REGION_UNMAPPED));
    }
    let p_l = *points194
        .get(left_anchor)
        .ok_or(Refusal::ValueOutOfRange(LANDMARKS_INVALID))?;
    let p_r = *points194
        .get(right_anchor)
        .ok_or(Refusal::ValueOutOfRange(LANDMARKS_INVALID))?;
    let dx = p_r[0] - p_l[0];
    let dy = p_r[1] - p_l[1];
    let len = dx.hypot(dy);
    let (tx, ty) = if len > 1e-3 && len.is_finite() {
        let mut ux = dx / len;
        let mut uy = dy / len;
        if ux < 0.0 {
            ux = -ux;
            uy = -uy;
        }
        (ux, uy)
    } else {
        (1.0, 0.0)
    };
    let (nx, ny) = (-ty, tx);

    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    for &landmark in landmarks {
        let p = *points194
            .get(landmark)
            .ok_or(Refusal::ValueOutOfRange(LANDMARKS_INVALID))?;
        cx += p[0];
        cy += p[1];
    }
    cx /= landmarks.len() as f32;
    cy /= landmarks.len() as f32;

    let mut s_min = f32::INFINITY;
    let mut s_max = f32::NEG_INFINITY;
    let mut d_min = f32::INFINITY;
    let mut d_max = f32::NEG_INFINITY;
    for &landmark in landmarks {
        let p = points194[landmark];
        let rx = p[0] - cx;
        let ry = p[1] - cy;
        let s = rx * tx + ry * ty;
        let d = rx * nx + ry * ny;
        s_min = s_min.min(s);
        s_max = s_max.max(s);
        d_min = d_min.min(d);
        d_max = d_max.max(d);
    }

    let w = (s_max - s_min).max(2.0);
    let h = (d_max - d_min).max(2.0);
    let s0 = s_min - w * pad_u_frac;
    let s1 = s_max + w * pad_u_frac;
    let d0 = d_min - h * pad_v_frac;
    let d1 = d_max + h * pad_v_frac;
    let fw = frame_size[0] as f32;
    let fh = frame_size[1] as f32;
    let corner = |s: f32, d: f32| {
        [
            (cx + s * tx + d * nx).clamp(0.0, fw),
            (cy + s * ty + d * ny).clamp(0.0, fh),
        ]
    };
    if vertices.len() + 4 > u16::MAX as usize {
        return Err(Refusal::ValueOutOfRange(REGION_UNMAPPED));
    }
    let base = vertices.len() as u16;
    vertices.extend_from_slice(&[corner(s0, d0), corner(s1, d0), corner(s1, d1), corner(s0, d1)]);
    uvs.extend_from_slice(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    Ok(())
}

/// Append the convex perimeter of a landmark subset and fan it with AABB UVs.
fn push_hull_aabb(
    points194: &[[f32; 2]],
    landmarks: &[usize],
    vertices: &mut Vec<[f32; 2]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u16>,
) -> Result<(), Refusal> {
    if landmarks.len() < 3 {
        return Err(Refusal::ShapeMismatch(REGION_UNMAPPED));
    }
    let selected = landmarks
        .iter()
        .map(|&landmark| {
            points194
                .get(landmark)
                .copied()
                .ok_or(Refusal::ValueOutOfRange(LANDMARKS_INVALID))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let hull = convex_hull(&selected);
    if hull.len() < 3 {
        return Err(Refusal::ShapeMismatch(REGION_UNMAPPED));
    }
    if vertices.len() + hull.len() > u16::MAX as usize {
        return Err(Refusal::ValueOutOfRange(REGION_UNMAPPED));
    }
    let mut minimum = [f32::INFINITY; 2];
    let mut maximum = [f32::NEG_INFINITY; 2];
    for &index in &hull {
        let p = selected[index];
        minimum[0] = minimum[0].min(p[0]);
        minimum[1] = minimum[1].min(p[1]);
        maximum[0] = maximum[0].max(p[0]);
        maximum[1] = maximum[1].max(p[1]);
    }
    let span = [maximum[0] - minimum[0], maximum[1] - minimum[1]];
    let base = vertices.len() as u16;
    for &index in &hull {
        let p = selected[index];
        vertices.push(p);
        uvs.push([
            normalize_uv(p[0], minimum[0], span[0]),
            normalize_uv(p[1], minimum[1], span[1]),
        ]);
    }
    let count = hull.len() as u16;
    for step in 1..count - 1 {
        indices.extend_from_slice(&[base, base + step, base + step + 1]);
    }
    Ok(())
}

/// Andrew monotone chain, counter-clockwise, deterministic tie-break by index.
fn convex_hull(points: &[[f32; 2]]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&a, &b| {
        points[a][0]
            .total_cmp(&points[b][0])
            .then_with(|| points[a][1].total_cmp(&points[b][1]))
            .then_with(|| a.cmp(&b))
    });

    let mut lower = Vec::with_capacity(order.len());
    for &index in &order {
        while lower.len() >= 2
            && cross(
                points[lower[lower.len() - 2]],
                points[lower[lower.len() - 1]],
                points[index],
            ) <= 0.0
        {
            lower.pop();
        }
        lower.push(index);
    }

    let mut upper = Vec::with_capacity(order.len());
    for &index in order.iter().rev() {
        while upper.len() >= 2
            && cross(
                points[upper[upper.len() - 2]],
                points[upper[upper.len() - 1]],
                points[index],
            ) <= 0.0
        {
            upper.pop();
        }
        upper.push(index);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn midpoint(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

fn cross(origin: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f64 {
    ((a[0] - origin[0]) as f64) * ((b[1] - origin[1]) as f64)
        - ((a[1] - origin[1]) as f64) * ((b[0] - origin[0]) as f64)
}

fn normalize_uv(value: f32, minimum: f32, span: f32) -> f32 {
    if span > 0.0 {
        ((value - minimum) / span).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

fn encode_pairs(pairs: &[[f32; 2]]) -> Arc<[u8]> {
    let mut bytes = Vec::with_capacity(pairs.len() * VERTEX_STRIDE);
    for point in pairs {
        bytes.extend_from_slice(&point[0].to_le_bytes());
        bytes.extend_from_slice(&point[1].to_le_bytes());
    }
    Arc::from(bytes.into_boxed_slice())
}

#[cfg(test)]
pub(crate) fn posed_real_face194_for_tests(
    yaw_deg: f32,
    roll_deg: f32,
    frame: [u32; 2],
) -> Vec<[f32; 2]> {
    let cx_uv = 0.4535f32;
    let cy_uv = 0.6200f32;
    let scale_x = 901.0 * 0.45;
    let scale_y = 1081.0 * 0.45;
    let yaw = yaw_deg.to_radians();
    let roll = roll_deg.to_radians();
    let fc = [frame[0] as f32 * 0.5, frame[1] as f32 * 0.5];

    let mut pts = Vec::with_capacity(POINTS194);
    for i in 0..PRIMARY142 {
        let [u, v] = FACE_TEMPLATE_149_UV[i];
        let dx = (u - cx_uv) * scale_x;
        let dy = (v - cy_uv) * scale_y;
        let z = match i {
            54..=58 => (i - 54) as f32 * 12.0 + 30.0,
            59..=72 => 25.0,
            73..=106 => 30.0,
            109..=141 => -40.0,
            _ => 15.0,
        };
        let xr = dx * yaw.cos() + z * yaw.sin();
        let yr = dy;
        let xrr = xr * roll.cos() - yr * roll.sin() + fc[0];
        let yrr = xr * roll.sin() + yr * roll.cos() + fc[1];
        pts.push([xrr, yrr]);
    }
    while pts.len() < POINTS194 {
        pts.push(fc);
    }
    pts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portrait_makeup_bundle::{row_profile, row_profile_for_effect};

    fn synthetic194() -> Vec<[f32; 2]> {
        let tau = std::f32::consts::TAU;
        (0..POINTS194)
            .map(|index| {
                let angle = index as f32 / POINTS194 as f32 * tau;
                let radius = 180.0 + (index % 7) as f32 * 3.0;
                [
                    256.0 + radius * angle.cos(),
                    256.0 + radius * angle.sin(),
                ]
            })
            .collect()
    }

    fn posed_real_face194(yaw_deg: f32, roll_deg: f32, frame: [u32; 2]) -> Vec<[f32; 2]> {
        posed_real_face194_for_tests(yaw_deg, roll_deg, frame)
    }

    fn u16s(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect()
    }

    fn assert_stream(points: &[[f32; 2]], profile: &MakeupRowProfile) {
        let draw = build_face2d_draw_bytes(points, [512, 512], profile).unwrap();
        assert_eq!(draw.vertex_stride, 8);
        assert_eq!(draw.position_offset, 0);
        assert_eq!(draw.draw_first_index, 0);
        assert!(!draw.vertices.is_empty());
        assert_eq!(draw.vertices.len() % 8, 0);
        let vertex_count = draw.vertices.len() / 8;
        for point in draw.vertices.chunks_exact(8) {
            let x = f32::from_le_bytes(point[0..4].try_into().unwrap());
            let y = f32::from_le_bytes(point[4..8].try_into().unwrap());
            assert!(x.is_finite() && y.is_finite());
        }

        let indices = u16s(&draw.indices_u16);
        assert!(!indices.is_empty());
        assert_eq!(draw.draw_index_count, indices.len());
        assert_eq!(draw.draw_index_count % 3, 0);
        assert!(indices.iter().all(|&index| (index as usize) < vertex_count));

        let uv = build_region_uv(points, [512, 512], profile).unwrap();
        assert_eq!(uv.vertex_stride, 8);
        assert_eq!(uv.offset, 0);
        assert_eq!(uv.vertices.len(), draw.vertices.len());
        for pair in uv.vertices.chunks_exact(8) {
            let u = f32::from_le_bytes(pair[0..4].try_into().unwrap());
            let v = f32::from_le_bytes(pair[4..8].try_into().unwrap());
            assert!((0.0..=1.0).contains(&u));
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn every_b3_row_builds_a_non_empty_stream() {
        let points = synthetic194();
        for pf in B3_MAKEUP_ROWS {
            assert_stream(&points, &row_profile(pf).unwrap());
        }
        assert_stream(
            &points,
            &row_profile_for_effect(115, "BlusherImageMakeup").unwrap(),
        );
        assert_stream(&points, &row_profile_for_effect(107, "LipColor").unwrap());
        assert_stream(
            &points,
            &row_profile_for_effect(107, "LipStickImageMakeup").unwrap(),
        );
    }

    #[test]
    fn ring_topology_is_stable() {
        let points = synthetic194();

        let brow = build_face2d_draw_bytes(&points, [512, 512], &row_profile(100).unwrap())
            .unwrap();
        assert_eq!(brow.vertices.len() / 8, 4);
        assert_eq!(brow.draw_index_count, 2 * 3);

        let eye = build_face2d_draw_bytes(&points, [512, 512], &row_profile(103).unwrap())
            .unwrap();
        assert_eq!(eye.vertices.len() / 8, 42);
        assert_eq!(eye.draw_index_count, 48 * 3);

        let lips = build_face2d_draw_bytes(&points, [512, 512], &row_profile(107).unwrap())
            .unwrap();
        assert_eq!(lips.vertices.len() / 8, 34);
        assert_eq!(lips.draw_index_count, 34 * 3);
    }

    #[test]
    fn profile_face_blusher_mesh_culls_folded_triangles_and_preserves_nose_uv() {
        let points = posed_real_face194(55.0, 20.0, [1000, 1000]);
        let mesh = region_mesh(&points, [1000, 1000], &row_profile(106).unwrap()).unwrap();
        // Every emitted triangle has consistent positive CCW area (folded far-side triangles culled).
        for tri in mesh.indices.chunks_exact(3) {
            let cr = cross(
                mesh.positions[tri[0] as usize],
                mesh.positions[tri[1] as usize],
                mesh.positions[tri[2] as usize],
            );
            assert!(cr > 0.0, "folded triangle survived culling: area={cr}");
        }
        // Nose bridge landmark 56 must retain canonical midline UV (~0.453), never cheek UV (~0.25/0.68).
        let u56 = mesh.uvs[56][0];
        assert!(
            (u56 - FACE_TEMPLATE_149_UV[56][0]).abs() < 1e-5,
            "nose bridge UV drifted under 55-deg yaw: {u56}"
        );
    }

    #[test]
    fn eyeshadow_and_iris_give_each_eye_independent_uvs() {
        let points = posed_real_face194(0.0, 15.0, [1000, 1000]);
        let eyeshadow =
            region_mesh(&points, [1000, 1000], &row_profile(90032).unwrap()).unwrap();
        // Left eye occupies vertices 0..16, right eye occupies 16..32; each spans [0..0.9] independently.
        let left_u_max = eyeshadow.uvs[..16]
            .iter()
            .map(|uv| uv[0])
            .fold(0.0f32, f32::max);
        let right_u_min = eyeshadow.uvs[16..]
            .iter()
            .map(|uv| uv[0])
            .fold(1.0f32, f32::min);
        assert!(left_u_max > 0.7 && right_u_min < 0.2);

        let pupil = region_mesh(&points, [1000, 1000], &row_profile(111).unwrap()).unwrap();
        assert!((pupil.uvs[0][0] - 0.5).abs() < 1e-5 && (pupil.uvs[9][0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn lip_annulus_leaves_mouth_cavity_open_and_tracks_canonical_uv() {
        let points = posed_real_face194(15.0, -10.0, [1000, 1000]);
        let mesh = region_mesh(&points, [1000, 1000], &row_profile(107).unwrap()).unwrap();
        // Mouth interior centroid (average of inner top 98 and inner bottom 104) must not fall inside any triangle.
        let inner_center = midpoint(points[98], points[104]);
        for tri in mesh.indices.chunks_exact(3) {
            let p0 = mesh.positions[tri[0] as usize];
            let p1 = mesh.positions[tri[1] as usize];
            let p2 = mesh.positions[tri[2] as usize];
            let area = cross(p0, p1, p2);
            assert!(area > 0.0, "lip annulus triangle inverted: {tri:?}");
            let w0 = cross(p1, p2, inner_center) / area;
            let w1 = cross(p2, p0, inner_center) / area;
            let w2 = cross(p0, p1, inner_center) / area;
            assert!(
                !(w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0),
                "lip triangle covers inner mouth cavity!"
            );
        }
    }

    #[test]
    fn bad_inputs_are_refused_by_name() {
        let points = synthetic194();
        let brow = row_profile(100).unwrap();

        assert_eq!(
            build_face2d_draw_bytes(&points[..193], [512, 512], &brow),
            Err(Refusal::ShapeMismatch(LANDMARKS_INVALID))
        );

        let mut non_finite = points.clone();
        non_finite[7][0] = f32::NAN;
        assert_eq!(
            build_face2d_draw_bytes(&non_finite, [512, 512], &brow),
            Err(Refusal::ValueOutOfRange(LANDMARKS_INVALID))
        );

        assert_eq!(
            build_region_uv(&points, [0, 512], &row_profile(8).unwrap()),
            Err(Refusal::ShapeMismatch(FRAME_INVALID))
        );

        let mut unknown = row_profile(8).unwrap();
        unknown.mask_semantic = "unmapped semantic";
        assert_eq!(
            build_face2d_draw_bytes(&points, [512, 512], &unknown),
            Err(Refusal::MissingInput(REGION_UNMAPPED))
        );
    }
}
