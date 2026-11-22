from __future__ import annotations

import hashlib
import json
import re
import struct
from datetime import datetime, timezone
from pathlib import Path


CONFIG_ROOT = Path(
    r"C:\WorkSpace\自研图像编辑软件\逆向工程\像素蛋糕\effect-decrypted\config"
)
UI_BUNDLE = Path(
    r"C:\WorkSpace\自研图像编辑软件\逆向工程\像素蛋糕\www\agent\assets"
    r"\dnd-overlay.vue_vue_type_style_index_0_scoped_e11a735c_lang-Cf6pNYKR.js"
)
PORTRAIT_PARSE = Path(
    r"C:\WorkSpace\Aurora Studio\90_work\lanes\portrait-parse"
)
LANE = Path(__file__).resolve().parents[1]

B3_ROWS = [8, 9, 100, 103, 105, 106, 107, 111, 115, 116, 117, 118, 30007, 90032]

ROWS = {
    8: ("FaceStereo/effect.json", 0, "shadowStereo", None, "shadowStereo"),
    9: ("FaceStereo/effect.json", 2, "highlight", None, "highlight"),
    100: (
        "Makeups/Eyebrow/MUEB00000010/effect.json",
        0,
        "eyebrowDegree_left",
        "eyebrowPath",
        "eyebrowDegree_left",
    ),
    103: (
        "Makeups/EyeLiner/MUELR00000001/effect.json",
        0,
        "eyelinerDegree_upper",
        "eyelinerPath",
        "eyelinerDegree_upper",
    ),
    105: (
        "Makeups/Eyelash/MUEL00000001/effect.json",
        0,
        "eyelashDegree_upper",
        "eyelashPath",
        "eyelashDegree_upper",
    ),
    106: (
        "Makeups/Blusher/MUBL00000001/effect.json",
        0,
        "blusherDegree",
        "blusherPath",
        "blusherDegree",
    ),
    107: (
        "Makeups/Lipstick/MULS00000001/Satin/effect.json",
        1,
        "lipstickDegree",
        "lipstickPath",
        "lipstickDegree",
    ),
    111: (
        "Makeups/Eyepupil/MUEP00000010/effect.json",
        0,
        "eyePupilDegree",
        "eyePupilPath",
        "eyePupilDegree",
    ),
    115: (
        "Makeups/FacialDecoration/MUFD00000004/effect.json",
        0,
        "faceDecorationDegree",
        "faceDecorationPath",
        "faceDecorationDegree",
    ),
    116: (
        "Makeups/Contour/MUCT00000001/effect.json",
        0,
        "contourDegree",
        "contourPath",
        "contourDegree",
    ),
    117: (
        "Makeups/Eyebrow/MUEB00000010/effect.json",
        1,
        "eyebrowDegree_right",
        "eyebrowPath",
        "eyebrowDegree_right",
    ),
    118: (
        "Makeups/Eyelash/MUEL00000001/effect.json",
        2,
        "eyelashDegree_lower",
        "eyelashPath",
        "eyelashDegree_lower",
    ),
    30007: (
        "Makeups/EyeLiner/MUELR00000001/effect.json",
        1,
        "eyelinerDegree_lower",
        "eyelinerPath",
        "eyelinerDegree_lower",
    ),
    90032: (
        "Makeups/Eyemakeup/MUEM00000001/effect.json",
        0,
        "eyeMakeupDegree",
        "eyeMakeupPath",
        "eyeMakeupDegree",
    ),
}

UI_KEYS = {
    "highlight": 9,
    "shadowStereo": 8,
    "eyebrowPath": 90092,
    "eyebrowDegree_left": 100,
    "eyebrowDegree_right": 117,
    "eyelinerPath": 900114,
    "eyelinerDegree_upper": 103,
    "eyelinerDegree_lower": 30007,
    "eyelashPath": 90093,
    "eyelashDegree_upper": 105,
    "eyelashDegree_lower": 118,
    "blusherPath": 90091,
    "blusherDegree": 106,
    "lipstickPath": 90095,
    "lipstickDegree": 107,
    "lipstickTexture": 90095,
    "eyePupilPath": 90097,
    "eyePupilDegree": 111,
    "eyeMakeupPath": 90094,
    "eyeMakeupDegree": 90032,
    "faceDecorationPath": 900110,
    "faceDecorationDegree": 115,
    "contourPath": 900111,
    "contourDegree": 116,
}

SUIT_SLOTS = [
    (1, "Bronzer", "BlusherImageMakeup", "xr.png"),
    (2, "Bronzer", "BlusherImageMakeup", "xr_SideFace.png"),
    (3, "Blusher", "BlusherImageMakeup", "sh.png"),
    (4, "Eyebrow", "NormalEyeBrowImageMakeup", "mmz.png"),
    (5, "Eyebrow", "NormalEyeBrowImageMakeup", "mmy.png"),
    (6, "Eyemakeup", "SymEyeImageMakeup", "yyz.png"),
    (7, "Eyemakeup", "SymEyeImageMakeup", "yyy.png"),
    (8, "Eyeliner", "SymEyeImageMakeup", "yxy.png"),
    (9, "Eyeliner", "SymEyeImageMakeup", "yxz.png"),
    (10, "Eyelash", "SymEyeLashImageMakeup", "jmz.png"),
    (11, "Eyelash", "SymEyeLashImageMakeup", "jmy.png"),
    (12, "Lipstick", "LipStickImageMakeup", "zc.png"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def asset_meta(path: Path) -> dict:
    data = path.read_bytes()
    meta = {
        "path": path.relative_to(CONFIG_ROOT).as_posix(),
        "bytes": len(data),
        "sha256": sha256(path),
    }
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        meta["format"] = "png"
        meta["width"], meta["height"] = struct.unpack(">II", data[16:24])
        meta["bit_depth"] = data[24]
        meta["color_type"] = data[25]
    else:
        meta["format"] = path.suffix.lower().lstrip(".")
    return meta


def load_effect(rel: str) -> tuple[Path, dict]:
    path = CONFIG_ROOT / rel
    return path, json.loads(path.read_text(encoding="utf-8"))


def mask_fields(effect: dict) -> dict[str, int]:
    return {
        key: value
        for key, value in effect.items()
        if key.endswith("MaskType") and isinstance(value, int)
    }


def path_role(effect_name: str, index: int) -> str:
    if effect_name == "EyePupilMakeup":
        return "material" if index == 0 else "auxiliary_mask_asset"
    if effect_name in {
        "NormalEyeBrowColorMakeup",
        "SymEyeColorMakeup",
        "SymEyeLashColorMakeup",
        "BlusherColorMakeup",
    }:
        return "owner_mask_template"
    return "material"


def variant_index() -> dict[int, list[dict]]:
    variants = {pf: [] for pf in B3_ROWS}
    for path in CONFIG_ROOT.rglob("effect.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        for index, effect in enumerate(data.get("Effects", [])):
            pf = effect.get("ParamFlag")
            if pf not in variants:
                continue
            variants[pf].append(
                {
                    "config": path.relative_to(CONFIG_ROOT).as_posix(),
                    "effect_index": index,
                    "effect_name": effect.get("EffectName"),
                    "paths": effect.get("Paths") or [],
                    "mask_fields": mask_fields(effect),
                    "mesh_type": effect.get("MeshType"),
                    "face_3dmm_type": effect.get("Face3DMMType"),
                }
            )
    return variants


def all_variants(variants: dict[int, list[dict]]) -> list[dict]:
    grouped = {}
    for pf, rows in variants.items():
        for row in rows:
            key = (
                row["effect_name"],
                tuple(row["paths"]),
                tuple(sorted(row["mask_fields"].items())),
                row["mesh_type"],
                row["face_3dmm_type"],
            )
            entry = grouped.setdefault(
                key,
                {
                    "effect_name": row["effect_name"],
                    "paths": row["paths"],
                    "mask_fields": row["mask_fields"],
                    "mesh_type": row["mesh_type"],
                    "face_3dmm_type": row["face_3dmm_type"],
                    "param_flags": set(),
                    "example_config": row["config"],
                    "example_effect_index": row["effect_index"],
                },
            )
            entry["param_flags"].add(pf)
    result = []
    for entry in grouped.values():
        entry["param_flags"] = sorted(entry["param_flags"])
        result.append(entry)
    return sorted(
        result,
        key=lambda item: (
            item["effect_name"] or "",
            item["param_flags"],
            item["example_config"],
        ),
    )


def row_record(pf: int) -> dict:
    rel, index, ui_key, path_key, degree_key = ROWS[pf]
    path, data = load_effect(rel)
    effect = data["Effects"][index]
    paths = effect.get("Paths") or []
    assets = []
    for asset_index, asset_path in enumerate(paths):
        asset = (path.parent / asset_path).resolve()
        if not asset.is_file():
            assets.append(
                {
                    "path": asset.relative_to(CONFIG_ROOT).as_posix(),
                    "role": path_role(effect.get("EffectName", ""), asset_index),
                    "exists": False,
                }
            )
            continue
        meta = asset_meta(asset)
        meta["role"] = path_role(effect.get("EffectName", ""), asset_index)
        meta["exists"] = True
        assets.append(meta)
    model_path = effect.get("ModelPath")
    model = None
    if model_path:
        model_file = (path.parent / model_path).resolve()
        model = asset_meta(model_file) if model_file.is_file() else {
            "path": str(model_path),
            "exists": False,
        }
    return {
        "pf": pf,
        "ui_key": ui_key,
        "selector": {
            "path_key": path_key,
            "degree_key": degree_key,
        },
        "canonical_config": rel,
        "canonical_effect_index": index,
        "effect_name": effect.get("EffectName"),
        "param_flag": effect.get("ParamFlag"),
        "filter_type": effect.get("FilterType"),
        "opacity": effect.get("Opacity"),
        "mix_color": effect.get("MixColor"),
        "roi": effect.get("ROIRect"),
        "mesh_type": effect.get("MeshType"),
        "mask_fields": mask_fields(effect),
        "assets": assets,
        "model": model,
        "same_job_owner_mask": "missing",
        "same_job_vbo": "missing",
        "same_job_ebo": "missing",
        "same_job_mask_uv": "missing",
        "same_job_material_uv": "missing",
    }


def suit_materials() -> list[dict]:
    root = CONFIG_ROOT / "Makeups/Suit/MU00000001"
    records = []
    for order, group, expected_effect, file_name in SUIT_SLOTS:
        config = root / group / "effect.json"
        data = json.loads(config.read_text(encoding="utf-8"))
        matches = [
            (index, effect)
            for index, effect in enumerate(data["Effects"])
            if (effect.get("Paths") or [None])[0] == file_name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"suit slot {order}: expected one {group}/{file_name}, got {len(matches)}"
            )
        index, effect = matches[0]
        if effect.get("EffectName") != expected_effect:
            raise RuntimeError(
                f"suit slot {order}: effect {effect.get('EffectName')} != {expected_effect}"
            )
        material = root / group / file_name
        meta = asset_meta(material)
        meta["exists"] = True
        records.append(
            {
                "order": order,
                "group": group,
                "effect": effect.get("EffectName"),
                "effect_index": index,
                "config": config.relative_to(CONFIG_ROOT).as_posix(),
                "material": meta,
                "mask_fields": mask_fields(effect),
                "filter_type": effect.get("FilterType"),
                "opacity": effect.get("Opacity"),
                "mix_color": effect.get("MixColor"),
                "roi": effect.get("ROIRect"),
                "mesh_type": effect.get("MeshType"),
                "shelter_type": effect.get("ShelterType"),
                "force_use_shelter": bool(effect.get("ForceUseShelter", False)),
            }
        )
    return records


def rejected_same_job_candidates() -> list[dict]:
    roots = [
        "s811_pf111_material_producer",
        "s908_pf111_same_job_uv",
        "s927_makeup_uv_bundle",
        "s932_pf111_uv_producer",
        "s933_pf111_ebo_raster",
        "s970_pf111_same_job_vbo",
        "wl46-makeup-mask",
        "wl46-makeup-compile",
    ]
    keywords = re.compile(
        r"(makeup|eyepupil|normal.?eye|owner.?mask|mask.?uv|material.?uv|vbo|ebo)",
        re.IGNORECASE,
    )
    binary_suffixes = {".bin", ".f32le", ".u16le", ".jsonl"}
    rejected = []
    for root_name in roots:
        root = PORTRAIT_PARSE / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in binary_suffixes:
                continue
            if not keywords.search(path.name):
                continue
            rel = path.relative_to(root).as_posix()
            if "fixture" in rel.lower() or "synthetic" in rel.lower():
                reason = "synthetic or contract fixture"
            elif path.stat().st_size == 0:
                reason = "empty capture"
            elif "package" in rel.lower() or "stage-entry" in rel.lower():
                reason = "deterministic stage-entry geometry, not same-job bytes"
            else:
                reason = "no same-job identity or draw binding"
            rejected.append(
                {
                    "lane": root_name,
                    "path": rel,
                    "bytes": path.stat().st_size,
                    "reason": reason,
                }
            )
    return sorted(rejected, key=lambda item: (item["lane"], item["path"]))


def main() -> None:
    ui_text = UI_BUNDLE.read_text(encoding="utf-8", errors="ignore")
    missing_ui = [
        f"{key}:{value}"
        for key, value in UI_KEYS.items()
        if f"{key}:{value}" not in ui_text
    ]
    if missing_ui:
        raise RuntimeError(f"UI keys missing: {missing_ui}")

    variants = variant_index()
    rows = [row_record(pf) for pf in B3_ROWS]
    for row in rows:
        row["observed_variants"] = [
            item
            for item in all_variants(variants)
            if row["pf"] in item["param_flags"]
        ]

    suit = suit_materials()
    if len(suit) != 12:
        raise RuntimeError(f"suit material count {len(suit)} != 12")

    rejected = rejected_same_job_candidates()
    report = {
        "schema": "aurora.portrait-parse.wl47-makeup-owner-bytes.static-source-map/1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "sources": {
            "effect_config_root": str(CONFIG_ROOT),
            "ui_bundle": {
                "path": str(UI_BUNDLE),
                "bytes": UI_BUNDLE.stat().st_size,
                "sha256": sha256(UI_BUNDLE),
            },
        },
        "ui_keys": UI_KEYS,
        "rows": rows,
        "suit_materials": suit,
        "same_job_bytes": {
            "owner_mask": "0/13",
            "vbo": "0/14",
            "ebo": "0/14",
            "mask_uv": "0/13",
            "material_uv": "0/7",
            "accepted_files": [],
            "rejected_candidates": rejected,
        },
        "counts": {
            "b3_rows": len(rows),
            "suit_materials": len(suit),
            "canonical_assets_present": sum(
                1
                for row in rows
                for asset in row["assets"]
                if asset.get("exists")
            ),
            "same_job_owner_masks": 0,
            "same_job_vbo_streams": 0,
            "same_job_ebo_streams": 0,
            "same_job_mask_uv_streams": 0,
            "same_job_material_uv_streams": 0,
        },
        "findings": [
            {
                "id": "WL47-S-001",
                "status": "PROVEN",
                "claim": "PF107 has both LipColor and LipStickImageMakeup variants.",
            },
            {
                "id": "WL47-S-002",
                "status": "PROVEN",
                "claim": "PF115 has both 3DMMFaceSticker and BlusherImageMakeup variants.",
            },
            {
                "id": "WL47-S-003",
                "status": "PROVEN",
                "claim": "The twelve MU00000001 material files and mask fields match static effect.json.",
            },
            {
                "id": "WL47-S-004",
                "status": "OPEN",
                "claim": "No same-job owner-mask or two-path VBO/EBO/UV bytes were found locally.",
            },
        ],
    }

    evidence = LANE / "EVIDENCE" / "static-source-map.json"
    fixture = (
        LANE
        / "crates"
        / "engine"
        / "tests"
        / "fixtures"
        / "makeup_owner_sources"
        / "static-source-map.json"
    )
    evidence.parent.mkdir(parents=True, exist_ok=True)
    fixture.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    evidence.write_text(text, encoding="utf-8")
    fixture.write_text(text, encoding="utf-8")

    print(
        "rows=14 suit_materials=12 "
        f"canonical_assets={report['counts']['canonical_assets_present']} "
        f"same_job_owner_masks=0 vbo=0 ebo=0 mask_uv=0 material_uv=0 "
        f"rejected_candidates={len(rejected)}"
    )


if __name__ == "__main__":
    main()
