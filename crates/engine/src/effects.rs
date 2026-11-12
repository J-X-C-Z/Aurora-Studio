//! Region effect data contract — how the 378 C1 params with no AdjustmentSet field are stored.
//! Semantics come from the PixCake UI key registry (crate::pixcake_map::C1_PARAMS): every entry below is
//! named after a real key there. The response curve of each effect is NOT verified.

use crate::pixcake_map::{Family, Target};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegionKind {
    Face,
    LeftEye,
    RightEye,
    Mouth,
    Teeth,
    Hair,
    Skin,
    Clothing,
    Background,
    FullBody,
    Custom,
}

/// Where an effect acts. `rect` is `[x, y, w, h]`, normalized and directed in the oriented,
/// uncropped image — the same frame as [`crate::Spot`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionRef {
    pub kind: RegionKind,
    pub rect: [f32; 4],
}

impl RegionRef {
    pub const fn whole(kind: RegionKind) -> Self {
        Self { kind, rect: [0.0, 0.0, 1.0, 1.0] }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionEffect {
    pub id: u64,
    pub region: RegionRef,
    pub effect: EffectKind,
    pub strength: f32,
    pub options: EffectOptions,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EffectOptions {
    pub seed: u32,
    pub invert: bool,
    pub color: Option<[f32; 3]>,
    pub invert_mask: bool,
}

/// One entry per `Target::Effect` family in `pixcake_map`; an effect keeps the exact C1 key it came
/// from, so a region effect maps back to its preset slot without a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EffectKind {
    /// BodyFaceSculpt: point-based liquify of face and body proportions.
    Sculpt { param: &'static str },
    /// Eye: eye, iris, eyelid and eye-bag work.
    Eye { param: &'static str },
    /// Makeup: eyebrow, eyeliner, eyeshadow, lipstick, contour, pupil.
    Makeup { param: &'static str },
    /// SkinBodyV2: skin smoothing, wrinkle and stubble removal, plumping.
    SkinBody { param: &'static str },
    /// SkinRetouch: blemish removal, skin tone, gloss.
    SkinRetouch { param: &'static str },
    /// MaterialStyle: named look packs applied to face, body or whole image.
    MaterialStyle { param: &'static str },
    /// Clothing: creases, stains, edge cleanup on garments.
    Clothing { param: &'static str },
    /// MakeupRemoval: removes existing makeup before re-applying.
    MakeupRemoval { param: &'static str },
    /// LightShadow: relighting on the face and neck, shadow blending.
    LightShadow { param: &'static str },
    /// Teeth: whitening, gum line, blemish, alignment.
    Teeth { param: &'static str },
    /// Hair: fill, stray hair, white hair, smooth, gloss.
    Hair { param: &'static str },
    /// Background: background descent, sky gradient, blend mode.
    Background { param: &'static str },
    /// CoreRetouch: pouch, waist, face-shape preset.
    CoreRetouch { param: &'static str },
    /// AiRetouch: breast repair, mouth opening, makeup and arc protection.
    AiRetouch { param: &'static str },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EffectFamily {
    Sculpt,
    Eye,
    Makeup,
    SkinBody,
    SkinRetouch,
    MaterialStyle,
    Clothing,
    MakeupRemoval,
    LightShadow,
    Teeth,
    Hair,
    Background,
    CoreRetouch,
    AiRetouch,
}

impl EffectKind {
    pub fn family(self) -> EffectFamily {
        match self {
            Self::Sculpt { .. } => EffectFamily::Sculpt,
            Self::Eye { .. } => EffectFamily::Eye,
            Self::Makeup { .. } => EffectFamily::Makeup,
            Self::SkinBody { .. } => EffectFamily::SkinBody,
            Self::SkinRetouch { .. } => EffectFamily::SkinRetouch,
            Self::MaterialStyle { .. } => EffectFamily::MaterialStyle,
            Self::Clothing { .. } => EffectFamily::Clothing,
            Self::MakeupRemoval { .. } => EffectFamily::MakeupRemoval,
            Self::LightShadow { .. } => EffectFamily::LightShadow,
            Self::Teeth { .. } => EffectFamily::Teeth,
            Self::Hair { .. } => EffectFamily::Hair,
            Self::Background { .. } => EffectFamily::Background,
            Self::CoreRetouch { .. } => EffectFamily::CoreRetouch,
            Self::AiRetouch { .. } => EffectFamily::AiRetouch,
        }
    }

    /// The pixcake_map family this effect belongs to, so mapping stays provable against the table.
    pub fn c1_family(self) -> Family {
        match self {
            Self::Sculpt { .. } => Family::BodyFaceSculpt,
            Self::Eye { .. } => Family::Eye,
            Self::Makeup { .. } => Family::Makeup,
            Self::SkinBody { .. } => Family::SkinBodyV2,
            Self::SkinRetouch { .. } => Family::SkinRetouch,
            Self::MaterialStyle { .. } => Family::MaterialStyle,
            Self::Clothing { .. } => Family::Clothing,
            Self::MakeupRemoval { .. } => Family::MakeupRemoval,
            Self::LightShadow { .. } => Family::LightShadow,
            Self::Teeth { .. } => Family::Teeth,
            Self::Hair { .. } => Family::Hair,
            Self::Background { .. } => Family::Background,
            Self::CoreRetouch { .. } => Family::CoreRetouch,
            Self::AiRetouch { .. } => Family::AiRetouch,
        }
    }

    pub fn param(self) -> &'static str {
        match self {
            Self::Sculpt { param }
            | Self::Eye { param }
            | Self::Makeup { param }
            | Self::SkinBody { param }
            | Self::SkinRetouch { param }
            | Self::MaterialStyle { param }
            | Self::Clothing { param }
            | Self::MakeupRemoval { param }
            | Self::LightShadow { param }
            | Self::Teeth { param }
            | Self::Hair { param }
            | Self::Background { param }
            | Self::CoreRetouch { param }
            | Self::AiRetouch { param } => param,
        }
    }
}

impl EffectFamily {
    pub const ALL: [Self; 14] = [
        Self::Sculpt,
        Self::Eye,
        Self::Makeup,
        Self::SkinBody,
        Self::SkinRetouch,
        Self::MaterialStyle,
        Self::Clothing,
        Self::MakeupRemoval,
        Self::LightShadow,
        Self::Teeth,
        Self::Hair,
        Self::Background,
        Self::CoreRetouch,
        Self::AiRetouch,
    ];

    /// The pixcake_map family this family is spelled with, for audit against the table.
    pub fn c1_family(self) -> Family {
        match self {
            Self::Sculpt => Family::BodyFaceSculpt,
            Self::Eye => Family::Eye,
            Self::Makeup => Family::Makeup,
            Self::SkinBody => Family::SkinBodyV2,
            Self::SkinRetouch => Family::SkinRetouch,
            Self::MaterialStyle => Family::MaterialStyle,
            Self::Clothing => Family::Clothing,
            Self::MakeupRemoval => Family::MakeupRemoval,
            Self::LightShadow => Family::LightShadow,
            Self::Teeth => Family::Teeth,
            Self::Hair => Family::Hair,
            Self::Background => Family::Background,
            Self::CoreRetouch => Family::CoreRetouch,
            Self::AiRetouch => Family::AiRetouch,
        }
    }

    /// The default region of a family; sided and lip keys narrow it in [`to_region_effects`].
    pub fn region(self) -> RegionKind {
        match self {
            Self::Sculpt => RegionKind::Face,
            Self::Eye | Self::Makeup => RegionKind::Face,
            Self::SkinBody | Self::SkinRetouch => RegionKind::Skin,
            Self::MaterialStyle => RegionKind::FullBody,
            Self::Clothing => RegionKind::Clothing,
            Self::MakeupRemoval => RegionKind::Face,
            Self::LightShadow => RegionKind::Face,
            Self::Teeth => RegionKind::Teeth,
            Self::Hair => RegionKind::Hair,
            Self::Background => RegionKind::Background,
            Self::CoreRetouch => RegionKind::Face,
            Self::AiRetouch => RegionKind::FullBody,
        }
    }
}

/// Reverse lookup over `crate::pixcake_map::C1_PARAMS`; `None` for ids that adjust the image globally,
/// are unmapped, or do not exist.
pub fn from_c1(id: i32) -> Option<EffectKind> {
    let param = crate::pixcake_map::C1_PARAMS.iter().find(|param| param.id == id)?;
    if !matches!(param.target, Target::Effect(_)) {
        return None;
    }
    let key = param.key;
    Some(match param.family {
        Family::BodyFaceSculpt => EffectKind::Sculpt { param: key },
        Family::Eye => EffectKind::Eye { param: key },
        Family::Makeup => EffectKind::Makeup { param: key },
        Family::SkinBodyV2 => EffectKind::SkinBody { param: key },
        Family::SkinRetouch => EffectKind::SkinRetouch { param: key },
        Family::MaterialStyle => EffectKind::MaterialStyle { param: key },
        Family::Clothing => EffectKind::Clothing { param: key },
        Family::MakeupRemoval => EffectKind::MakeupRemoval { param: key },
        Family::LightShadow => EffectKind::LightShadow { param: key },
        Family::Teeth => EffectKind::Teeth { param: key },
        Family::Hair => EffectKind::Hair { param: key },
        Family::Background => EffectKind::Background { param: key },
        Family::CoreRetouch => EffectKind::CoreRetouch { param: key },
        Family::AiRetouch => EffectKind::AiRetouch { param: key },
        Family::Geometry | Family::ColorGrade => return None,
    })
}

pub fn families() -> &'static [EffectFamily] {
    &EffectFamily::ALL
}

/// How many `pixcake_map` params a family carries.
pub fn count(family: EffectFamily) -> usize {
    crate::pixcake_map::C1_PARAMS.iter().filter(|param| from_c1(param.id).is_some_and(|kind| kind.family() == family)).count()
}

/// Turns one C1 batch into region effects. Params for the same region and family merge into a
/// single effect (last value wins) keyed by the family; the first id of the group is kept.
/// C1 batches carry no geometry, so `rect` is the full frame until the UI supplies a box.
pub fn to_region_effects(params: &[(i32, f32)]) -> Vec<RegionEffect> {
    let mut out: Vec<RegionEffect> = Vec::new();
    for &(id, value) in params {
        let Some(effect) = from_c1(id) else { continue };
        let family = effect.family();
        let region = RegionRef::whole(region_of(effect));
        let strength = value.clamp(0.0, 1.0);
        let existing = out
            .iter_mut()
            .find(|slot| slot.effect.family() == family && slot.region.kind == region.kind);
        // shortcut: one slot per region+family, so a family's sub-controls collapse to the last
        // one. Upgrade to per-param slots when a family needs more than one live control.
        match existing {
            Some(slot) => {
                slot.strength = strength;
                slot.effect = effect;
            }
            None => out.push(RegionEffect {
                id: id as u64,
                region,
                effect,
                strength,
                options: EffectOptions::default(),
            }),
        }
    }
    out
}

/// Sided keys act on one eye; lip keys act on the mouth.
fn region_of(effect: EffectKind) -> RegionKind {
    let key = effect.param();
    match effect.family() {
        EffectFamily::Eye | EffectFamily::Makeup => {
            if key.contains("left") {
                RegionKind::LeftEye
            } else if key.contains("right") {
                RegionKind::RightEye
            } else if key.contains("lip") {
                RegionKind::Mouth
            } else {
                RegionKind::Face
            }
        }
        family => family.region(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_effect_param_maps() {
        let effects: Vec<_> = crate::pixcake_map::C1_PARAMS
            .iter()
            .filter(|param| matches!(param.target, Target::Effect(_)))
            .collect();
        assert_eq!(effects.len(), 378);
        for param in effects {
            let kind = from_c1(param.id).expect(param.key);
            assert_eq!(kind.param(), param.key);
            assert_eq!(kind.c1_family(), param.family);
        }
        assert_eq!(crate::pixcake_map::coverage(), (452, 482));
    }

    #[test]
    fn non_effect_ids_are_rejected() {
        assert_eq!(from_c1(90052), None);
        assert_eq!(from_c1(-7), None);
        assert_eq!(from_c1(i32::MAX), None);
    }

    #[test]
    fn family_counts_sum_to_effect_total() {
        let total: usize = families().iter().map(|&family| count(family)).sum();
        assert_eq!(total, 378);
        assert_eq!(count(EffectFamily::Eye), 97);
        assert_eq!(count(EffectFamily::Sculpt), 80);
        assert_eq!(count(EffectFamily::Makeup), 58);
        assert_eq!(count(EffectFamily::SkinBody), 49);
        assert_eq!(count(EffectFamily::SkinRetouch), 16);
        assert_eq!(count(EffectFamily::MaterialStyle), 18);
        assert_eq!(count(EffectFamily::Clothing), 11);
        assert_eq!(count(EffectFamily::MakeupRemoval), 10);
        assert_eq!(count(EffectFamily::LightShadow), 9);
        assert_eq!(count(EffectFamily::Teeth), 8);
        assert_eq!(count(EffectFamily::Hair), 8);
        assert_eq!(count(EffectFamily::Background), 5);
        assert_eq!(count(EffectFamily::AiRetouch), 6);
    }

    #[test]
    fn same_family_merges() {
        let effects = to_region_effects(&[(0, 0.3), (1, 0.6), (5, 0.9), (0, 0.1)]);
        assert_eq!(effects.len(), 2);
        assert_eq!(effects[0].region.kind, RegionKind::Skin);
        assert_eq!(effects[0].effect, EffectKind::SkinRetouch { param: "spotless" });
        assert!((effects[0].strength - 0.1).abs() < 1e-6);
        assert_eq!(effects[0].id, 0);
        assert_eq!(effects[1].effect, EffectKind::CoreRetouch { param: "lightenPouch" });
    }

    #[test]
    fn strength_clamps() {
        let effects = to_region_effects(&[(100, 2.5), (5, -1.0)]);
        assert_eq!(effects[0].strength, 1.0);
        assert_eq!(effects[1].strength, 0.0);
    }

    #[test]
    fn sided_and_lip_keys_narrow_the_region() {
        let eyebrow = crate::pixcake_map::C1_PARAMS.iter().find(|param| param.key == "eyebrowDegree_left").unwrap();
        let blusher = crate::pixcake_map::C1_PARAMS.iter().find(|param| param.key == "blusherDegree").unwrap();
        let lip = crate::pixcake_map::C1_PARAMS.iter().find(|param| param.key == "lipstickDegree").unwrap();
        assert_eq!(to_region_effects(&[(eyebrow.id, 0.5)])[0].region.kind, RegionKind::LeftEye);
        assert_eq!(to_region_effects(&[(blusher.id, 0.5)])[0].region.kind, RegionKind::Face);
        assert_eq!(to_region_effects(&[(lip.id, 0.5)])[0].region.kind, RegionKind::Mouth);
    }
}
