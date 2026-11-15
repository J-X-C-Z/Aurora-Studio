//! One module per recovered effect family.
//!
//! Families export `META` and typed entry points. Shared producer-only stages
//! have no effect entry. This is the single registry integration point.

pub mod body_geometry;
pub mod blemish;
pub mod blemish_blend;
pub mod beard_transform;
pub mod beard;
pub mod double_chin;
pub mod close_smile;
pub mod detail_fusion;
pub mod eye_geometry;
pub mod eye_bag;
pub mod eye_type1010;
pub mod eye_type1010_core;
pub mod eye_type1010_live;
pub mod face_geometry;
pub mod face_detail;
pub mod facial_smooth;
pub mod face_pose;
pub mod forehead_common;
pub mod hair_body;
pub mod hair_color;
pub mod hair_material;
pub mod leg_refine_uv;
pub mod leg_refine;
pub mod makeup;
pub mod makeup_material;
pub mod mesh_attribute;
pub mod material_defines;
pub mod mask_compose;
pub mod neck_repair;
pub mod neck_paste;
pub mod skin_adjust;
pub mod skin_color;
pub mod sticker;
pub mod sweat;
pub mod skin_models;
pub mod texture_material;
pub mod texture_mask_blend;
pub mod teeth_gray;
pub mod teeth_blend;
pub mod teeth_alignment;
pub mod suit_makeup;
pub mod tone_unify;
pub mod plump;
pub mod radiance;
pub mod moisture;
pub mod smile_warp;
pub mod pore;
pub mod uvmap_color;

use crate::EffectMeta;

fn registry_impl() -> Vec<EffectMeta> {
    let mut all = Vec::new();
    all.extend_from_slice(facial_smooth::META);
    all.extend_from_slice(face_geometry::META);
    all.extend_from_slice(face_detail::META);
    all.extend_from_slice(eye_geometry::META);
    all.extend_from_slice(body_geometry::META);
    all.extend_from_slice(hair_body::META);
    all.extend_from_slice(blemish::META);
    all.extend_from_slice(makeup::META);
    all.extend_from_slice(skin_models::META);
    all.extend_from_slice(skin_adjust::META);
    all.extend_from_slice(skin_adjust::REFUSED);
    all.extend_from_slice(skin_color::META);
    all.extend_from_slice(leg_refine::META);
    all.extend_from_slice(tone_unify::META);
    all.extend_from_slice(radiance::META);
    all.extend_from_slice(moisture::META);
    all.extend_from_slice(smile_warp::META);
    all.extend_from_slice(teeth_alignment::META);
    all.extend_from_slice(beard::META);
    all.extend_from_slice(close_smile::META);
    all.extend_from_slice(double_chin::META);
    all.extend_from_slice(sticker::META);
    all.extend_from_slice(pore::META);
    all.extend_from_slice(neck_paste::META);
    all.extend_from_slice(sweat::META);
    all
}

/// Single registry integration point. Refused rows that share a producer
/// ParamFlag with an unresolved final path are listed here only.
pub fn registry() -> Vec<EffectMeta> {
    let mut all = registry_impl();
    all.sort_by_key(|row| row.pf);
    let duplicates = all.windows(2).filter(|pair| pair[0].pf == pair[1].pf).count();
    assert_eq!(duplicates, 0, "registry has duplicate ParamFlag rows");
    all
}
