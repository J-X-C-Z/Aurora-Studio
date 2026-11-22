//! PF90088 HairFly selector -> registered model/session-slot producer.
//!
//! Static evidence from `pix-worker.exe`
//! (sha256 `509D22938469906D353D44C355F4AA1E445F81EA56CA288A540AE5553CC96B84`):
//!
//! - model registration table `0x14930F530`:
//!   `1655581403, 3378116080, 2127140479, 4083080210, 195031232`.
//! - selector `0x147FC1CA0` reads output `1644271730` (`[1,2]`) and writes
//!   the binary route bit to `+0x12`.
//! - large-model runner `0x147FC22B0` chooses one of four handles:
//!   `+0x38`, `+0x40`, `+0x48`, `+0x50` from the `+0x11` gate and the
//!   selector route bit in `+0x10`.
//! - native job `0x147B4F620` reads PF90088, PF90113 and PF90114 from the
//!   same per-face node `+0x20`, then writes the two products to native
//!   parameter index 3.
//!
//! This producer refuses missing or ambiguous model/area inputs. It never
//! substitutes a default model, a default area, or a mask-derived area.

/// Packaged selector model id from the five-entry HairFly registration table.
pub const SELECTOR_MODEL_HASH_ID: u32 = 1_655_581_403;
/// Sole selector output; runtime shape is `[1, 2]` float32.
pub const SELECTOR_OUTPUT_NAME: u32 = 1_644_271_730;

pub const PF90088_HASH_ID: u32 = 90_088;
pub const PF90113_HASH_ID: u32 = 90_113;
pub const PF90114_HASH_ID: u32 = 90_114;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HairFlyModelSlot {
    Large0,
    Large1,
    Large2,
    Large3,
}

impl HairFlyModelSlot {
    pub const fn index(self) -> u8 {
        match self {
            Self::Large0 => 0,
            Self::Large1 => 1,
            Self::Large2 => 2,
            Self::Large3 => 3,
        }
    }

    pub const fn handle_offset(self) -> u16 {
        match self {
            Self::Large0 => 0x38,
            Self::Large1 => 0x40,
            Self::Large2 => 0x48,
            Self::Large3 => 0x50,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HairFlyRegisteredModel {
    pub hash_id: u32,
    pub slot: HairFlyModelSlot,
    pub width: u16,
    pub height: u16,
}

/// Large-model registration order after the selector entry.
///
/// `width`/`height` come from the four real forward results in s885. The
/// slot-to-handle offsets come from the four branches in `0x147FC22B0`.
pub const HAIRFLY_REGISTERED_MODELS: [HairFlyRegisteredModel; 4] = [
    HairFlyRegisteredModel {
        hash_id: 3_378_116_080,
        slot: HairFlyModelSlot::Large0,
        width: 512,
        height: 768,
    },
    HairFlyRegisteredModel {
        hash_id: 2_127_140_479,
        slot: HairFlyModelSlot::Large1,
        width: 512,
        height: 768,
    },
    HairFlyRegisteredModel {
        hash_id: 4_083_080_210,
        slot: HairFlyModelSlot::Large2,
        width: 384,
        height: 512,
    },
    HairFlyRegisteredModel {
        hash_id: 195_031_232,
        slot: HairFlyModelSlot::Large3,
        width: 384,
        height: 512,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SameJobFace {
    pub job_id: u64,
    pub face_index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairFlySelectorReceipt {
    pub same_job: SameJobFace,
    pub model_hash_id: u32,
    pub output_name: u32,
    /// The two real selector logits, not an area value and not a mask ratio.
    pub logits: [f32; 2],
    /// Worker runner `+0x11`: the second branch bit used with `+0x10`.
    ///
    /// The static code proves this bit selects the 512/384 group. The exact
    /// source predicate is a separate same-job capture and is not defaulted.
    pub aux_gate_high: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairFlyNativeFaceReceipt {
    pub same_job: SameJobFace,
    /// PF90088 face node `+0x20`.
    pub pf90088: f32,
    /// PF90113 face node `+0x20`.
    pub pf90113: f32,
    /// PF90114 face node `+0x20`.
    pub pf90114: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairFlyNativeIndex3 {
    pub degree: f32,
    pub index3_background: f32,
    pub index3_person: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HairFlyProducerRefusal {
    SelectorModelMismatch,
    SelectorOutputMismatch,
    SelectorNonFiniteLogit,
    SelectorNotSameJob,
    NativeNonFinite(&'static str),
    NativeOutOfRange(&'static str),
    JobMismatch,
    FaceMismatch,
    SelectorArgmaxOutOfRange,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairFlyResolvedRoute {
    pub model: HairFlyRegisteredModel,
    pub selector_argmax: u8,
    pub native_index3: HairFlyNativeIndex3,
}

/// Mirrors `0x147FC1FE4..0x147FC1FFB`: index 1 only when `logits[1]` is
/// strictly greater; equal logits resolve to index 0.
pub fn selector_argmax(logits: [f32; 2]) -> Result<u8, HairFlyProducerRefusal> {
    if !logits.iter().all(|value| value.is_finite()) {
        return Err(HairFlyProducerRefusal::SelectorNonFiniteLogit);
    }
    Ok(u8::from(logits[1] > logits[0]))
}

/// Exact four-branch mapping from `0x147FC22B0`.
pub fn model_for_selector(
    selector_argmax: u8,
    aux_gate_high: bool,
) -> Option<HairFlyRegisteredModel> {
    let slot = match (aux_gate_high, selector_argmax) {
        (true, 0) => HairFlyModelSlot::Large0,
        (true, 1) => HairFlyModelSlot::Large1,
        (false, 0) => HairFlyModelSlot::Large2,
        (false, 1) => HairFlyModelSlot::Large3,
        _ => return None,
    };
    Some(HAIRFLY_REGISTERED_MODELS[usize::from(slot.index())])
}

fn check_native_value(field: &'static str, value: f32) -> Result<(), HairFlyProducerRefusal> {
    if !value.is_finite() {
        return Err(HairFlyProducerRefusal::NativeNonFinite(field));
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(HairFlyProducerRefusal::NativeOutOfRange(field));
    }
    Ok(())
}

/// Resolve one same-job HairFly producer route.
///
/// The selector and native receipts must belong to the same job and face.
/// The returned model is one of the four registered large models; the
/// returned native values are the raw PF values plus the two products the
/// native job writes to parameter index 3.
pub fn resolve_hairfly_producer(
    selector: &HairFlySelectorReceipt,
    native: &HairFlyNativeFaceReceipt,
) -> Result<HairFlyResolvedRoute, HairFlyProducerRefusal> {
    if selector.model_hash_id != SELECTOR_MODEL_HASH_ID {
        return Err(HairFlyProducerRefusal::SelectorModelMismatch);
    }
    if selector.output_name != SELECTOR_OUTPUT_NAME {
        return Err(HairFlyProducerRefusal::SelectorOutputMismatch);
    }
    if selector.same_job.job_id == 0 {
        return Err(HairFlyProducerRefusal::SelectorNotSameJob);
    }

    let selector_argmax = selector_argmax(selector.logits)?;

    if selector.same_job.job_id != native.same_job.job_id {
        return Err(HairFlyProducerRefusal::JobMismatch);
    }
    if selector.same_job.face_index != native.same_job.face_index {
        return Err(HairFlyProducerRefusal::FaceMismatch);
    }

    check_native_value("PF90088_NATIVE_DEGREE", native.pf90088)?;
    check_native_value("PF90113_NATIVE_PERSON_AREA", native.pf90113)?;
    check_native_value("PF90114_NATIVE_BACKGROUND_AREA", native.pf90114)?;

    let model = model_for_selector(selector_argmax, selector.aux_gate_high)
        .ok_or(HairFlyProducerRefusal::SelectorArgmaxOutOfRange)?;

    Ok(HairFlyResolvedRoute {
        model,
        selector_argmax,
        native_index3: HairFlyNativeIndex3 {
            degree: native.pf90088,
            index3_background: native.pf90088 * native.pf90114,
            index3_person: native.pf90088 * native.pf90113,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selector(argmax: u8, aux_gate_high: bool) -> HairFlySelectorReceipt {
        let logits = if argmax == 0 { [2.0, 1.0] } else { [1.0, 2.0] };
        HairFlySelectorReceipt {
            same_job: SameJobFace {
                job_id: 7,
                face_index: 0,
            },
            model_hash_id: SELECTOR_MODEL_HASH_ID,
            output_name: SELECTOR_OUTPUT_NAME,
            logits,
            aux_gate_high,
        }
    }

    fn native() -> HairFlyNativeFaceReceipt {
        HairFlyNativeFaceReceipt {
            same_job: SameJobFace {
                job_id: 7,
                face_index: 0,
            },
            pf90088: 0.5,
            pf90113: 0.25,
            pf90114: 0.75,
        }
    }

    #[test]
    fn registration_table_matches_static_order() {
        assert_eq!(
            HAIRFLY_REGISTERED_MODELS.map(|model| model.hash_id),
            [3_378_116_080, 2_127_140_479, 4_083_080_210, 195_031_232]
        );
        assert_eq!(
            HAIRFLY_REGISTERED_MODELS.map(|model| model.slot.handle_offset()),
            [0x38, 0x40, 0x48, 0x50]
        );
    }

    #[test]
    fn argmax_ties_follow_worker_strict_greater() {
        assert_eq!(selector_argmax([1.0, 1.0]), Ok(0));
        assert_eq!(selector_argmax([1.0, 2.0]), Ok(1));
        assert_eq!(
            selector_argmax([f32::NAN, 2.0]),
            Err(HairFlyProducerRefusal::SelectorNonFiniteLogit)
        );
    }

    #[test]
    fn four_slot_mapping_is_exact() {
        assert_eq!(model_for_selector(0, true).unwrap().hash_id, 3_378_116_080);
        assert_eq!(model_for_selector(1, true).unwrap().hash_id, 2_127_140_479);
        assert_eq!(model_for_selector(0, false).unwrap().hash_id, 4_083_080_210);
        assert_eq!(model_for_selector(1, false).unwrap().hash_id, 195_031_232);
        assert_eq!(model_for_selector(2, true), None);
    }

    #[test]
    fn producer_requires_same_job_and_face() {
        let mut other_job = native();
        other_job.same_job.job_id = 8;
        assert_eq!(
            resolve_hairfly_producer(&selector(0, true), &other_job),
            Err(HairFlyProducerRefusal::JobMismatch)
        );

        let mut other_face = native();
        other_face.same_job.face_index = 1;
        assert_eq!(
            resolve_hairfly_producer(&selector(0, true), &other_face),
            Err(HairFlyProducerRefusal::FaceMismatch)
        );
    }

    #[test]
    fn native_products_match_index3() {
        let route = resolve_hairfly_producer(&selector(1, false), &native()).unwrap();
        assert_eq!(route.model.hash_id, 195_031_232);
        assert_eq!(route.native_index3.degree, 0.5);
        assert_eq!(route.native_index3.index3_background, 0.375);
        assert_eq!(route.native_index3.index3_person, 0.125);
    }

    #[test]
    fn missing_area_is_refused_not_defaulted() {
        let mut bad = native();
        bad.pf90113 = f32::NAN;
        assert_eq!(
            resolve_hairfly_producer(&selector(0, true), &bad),
            Err(HairFlyProducerRefusal::NativeNonFinite(
                "PF90113_NATIVE_PERSON_AREA"
            ))
        );

        let mut out = native();
        out.pf90114 = 1.5;
        assert_eq!(
            resolve_hairfly_producer(&selector(0, true), &out),
            Err(HairFlyProducerRefusal::NativeOutOfRange(
                "PF90114_NATIVE_BACKGROUND_AREA"
            ))
        );
    }
}
