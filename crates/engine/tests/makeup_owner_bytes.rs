use std::collections::BTreeSet;

use aurora_engine::portrait_makeup_bundle::{
    B3_MAKEUP_ROWS, SUIT_MATERIAL_MASK_SLOTS, allowed_source_effects,
};
use serde_json::Value;

const SOURCE_MAP: &str = include_str!("fixtures/makeup_owner_sources/static-source-map.json");

fn source_map() -> Value {
    serde_json::from_str(SOURCE_MAP).expect("checked-in static source map")
}

#[test]
fn static_sources_cover_all_b3_rows_and_ui_selectors() {
    let map = source_map();
    let rows = map["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 14);

    let pfs = rows
        .iter()
        .map(|row| row["pf"].as_u64().unwrap() as u32)
        .collect::<Vec<_>>();
    assert_eq!(pfs, B3_MAKEUP_ROWS);

    for row in rows {
        let pf = row["pf"].as_u64().unwrap() as u32;
        let canonical = row["effect_name"].as_str().unwrap();
        let allowed = allowed_source_effects(pf).unwrap();
        assert!(allowed.contains(&canonical), "PF{pf} canonical effect");

        let observed = row["observed_variants"]
            .as_array()
            .unwrap()
            .iter()
            .map(|variant| variant["effect_name"].as_str().unwrap())
            .collect::<BTreeSet<_>>();
        for effect in observed {
            assert!(allowed.contains(&effect), "PF{pf} effect {effect}");
        }

        assert_eq!(row["same_job_owner_mask"], "missing", "PF{pf}");
        assert_eq!(row["same_job_vbo"], "missing", "PF{pf}");
        assert_eq!(row["same_job_ebo"], "missing", "PF{pf}");
        assert_eq!(row["same_job_mask_uv"], "missing", "PF{pf}");
        assert_eq!(row["same_job_material_uv"], "missing", "PF{pf}");
    }

    assert_eq!(map["ui_keys"]["blusherDegree"], 106);
    assert_eq!(map["ui_keys"]["eyelinerDegree_upper"], 103);
    assert_eq!(map["ui_keys"]["eyelinerDegree_lower"], 30007);
    assert_eq!(map["ui_keys"]["eyePupilDegree"], 111);
    assert_eq!(map["ui_keys"]["eyeMakeupDegree"], 90032);
}

#[test]
fn variant_splits_are_not_collapsed_to_one_row_kind() {
    let map = source_map();
    let variants = |pf: u64| {
        map["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["pf"] == pf)
            .unwrap()["observed_variants"]
            .as_array()
            .unwrap()
            .iter()
            .map(|variant| variant["effect_name"].as_str().unwrap())
            .collect::<BTreeSet<_>>()
    };

    assert_eq!(
        variants(107),
        BTreeSet::from(["LipColor", "LipStickImageMakeup"])
    );
    assert_eq!(
        variants(115),
        BTreeSet::from(["3DMMFaceSticker", "BlusherImageMakeup"])
    );
}

#[test]
fn twelve_suit_materials_match_effect_json_and_real_files() {
    let map = source_map();
    let materials = map["suit_materials"].as_array().unwrap();
    assert_eq!(materials.len(), SUIT_MATERIAL_MASK_SLOTS.len());

    for (record, slot) in materials.iter().zip(SUIT_MATERIAL_MASK_SLOTS) {
        assert_eq!(record["order"].as_u64().unwrap() as u8, slot.order);
        assert_eq!(record["group"].as_str().unwrap(), slot.group);
        assert_eq!(record["effect"].as_str().unwrap(), slot.effect);

        let material = &record["material"];
        assert_eq!(material["exists"], true, "slot {}", slot.order);
        assert_eq!(material["format"], "png", "slot {}", slot.order);
        assert!(
            material["width"].as_u64().unwrap() > 0,
            "slot {}",
            slot.order
        );
        assert!(
            material["height"].as_u64().unwrap() > 0,
            "slot {}",
            slot.order
        );
        assert!(
            slot.material.ends_with(material["path"].as_str().unwrap()),
            "slot {} path {}",
            slot.order,
            material["path"]
        );
        assert_eq!(
            record["mask_fields"][slot.mask_field].as_u64().unwrap() as u32,
            slot.mask_value,
            "slot {}",
            slot.order
        );
        assert_eq!(
            material["sha256"].as_str().unwrap().len(),
            64,
            "slot {}",
            slot.order
        );
    }

    assert_eq!(SUIT_MATERIAL_MASK_SLOTS[7].group, "Eyeliner");
    assert_eq!(SUIT_MATERIAL_MASK_SLOTS[8].group, "Eyeliner");
}

#[test]
fn no_local_same_job_bytes_are_promoted_from_candidates() {
    let map = source_map();
    let bytes = &map["same_job_bytes"];
    assert_eq!(bytes["accepted_files"].as_array().unwrap().len(), 0);
    assert_eq!(bytes["owner_mask"], "0/13");
    assert_eq!(bytes["vbo"], "0/14");
    assert_eq!(bytes["ebo"], "0/14");
    assert_eq!(bytes["mask_uv"], "0/13");
    assert_eq!(bytes["material_uv"], "0/7");
    assert!(!bytes["rejected_candidates"].as_array().unwrap().is_empty());
}
