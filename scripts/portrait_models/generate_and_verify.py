#!/usr/bin/env python3
"""Deterministic generator and SHA-256 verifier for 7 portrait prior/LUT asset categories.

Uses only the Python 3 standard library. Generates mathematical color-science
LUTs and parametric geometric priors ONLY; neural model slots are exported as
explicit contracts with `requires_real_dataset_training: true` and never
fabricate model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Tuple


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def u8(x: float) -> int:
    return int(round(clamp01(x) * 255.0))


def encode_png_rgba(width: int, height: int, rgba: bytes) -> bytes:
    assert len(rgba) == width * height * 4

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw_rows = bytearray()
    stride = width * 4
    for y in range(height):
        raw_rows.append(0)  # filter type 0
        raw_rows.extend(rgba[y * stride : (y + 1) * stride])

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(raw_rows), level=6)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", idat)
        + chunk(b"IEND", b"")
    )


def build_hald_lut_512(transform_rgb) -> bytes:
    """Builds a 512x512 RGBA8 3D LUT atlas (64x64x64 grid, 8x8 tiles of 64x64)."""
    width = 512
    height = 512
    buf = bytearray(width * height * 4)
    for b in range(64):
        tile_x = (b % 8) * 64
        tile_y = (b // 8) * 64
        fb = b / 63.0
        for g in range(64):
            fg = g / 63.0
            y = tile_y + g
            for r in range(64):
                fr = r / 63.0
                x = tile_x + r
                nr, ng, nb = transform_rgb(fr, fg, fb)
                idx = (y * width + x) * 4
                buf[idx] = u8(nr)
                buf[idx + 1] = u8(ng)
                buf[idx + 2] = u8(nb)
                buf[idx + 3] = 255
    return encode_png_rgba(width, height, bytes(buf))


def generate_skin_adjust(root: Path) -> List[Dict]:
    d = root / "skin-adjust"
    d.mkdir(parents=True, exist_ok=True)
    specs = [
        ("whiten.png", lambda r, g, b: (r ** 0.86, g ** 0.88, b ** 0.85)),
        ("ruddy_new.png", lambda r, g, b: (clamp01(r * 1.08 + 0.03 * (1 - r)), g * 0.96, b * 0.95)),
        ("TintGreen.png", lambda r, g, b: (r * 0.94, clamp01(g * 1.06), b * 0.96)),
        ("TintMagenta.png", lambda r, g, b: (clamp01(r * 1.05), g * 0.93, clamp01(b * 1.05))),
        ("cold.png", lambda r, g, b: (r * 0.92, g * 0.99, clamp01(b * 1.08 + 0.02 * (1 - b)))),
        ("warm.png", lambda r, g, b: (clamp01(r * 1.08 + 0.02 * (1 - r)), clamp01(g * 1.02), b * 0.90)),
    ]
    entries = []
    for name, fn in specs:
        data = build_hald_lut_512(fn)
        p = d / name
        p.write_bytes(data)
        entries.append(asset_entry("skin_adjust", f"skin-adjust/{name}", data, "prior_lut", False))
    return entries


def generate_radiance(root: Path) -> List[Dict]:
    d = root / "radiance"
    d.mkdir(parents=True, exist_ok=True)
    entries = []

    # 300 * 64 * 4 bytes = 76800 B highlight-lift radiance prior
    lut_a = bytearray()
    for row in range(300):
        t = row / 299.0
        for col in range(64):
            u = col / 63.0
            lift = clamp01(u + 0.08 * t * math.sin(math.pi * u))
            v = u8(lift)
            lut_a.extend((v, v, v, 255))
    (d / "prior-lut-a.bin").write_bytes(lut_a)
    entries.append(asset_entry("radiance", "radiance/prior-lut-a.bin", bytes(lut_a), "prior_lut", False))

    # 150 * 64 * 4 bytes = 38400 B contrast-softening prior
    lut_b = bytearray()
    for row in range(150):
        t = row / 149.0
        for col in range(64):
            u = col / 63.0
            s = u * u * (3.0 - 2.0 * u)
            v = u8(u * (1.0 - 0.15 * t) + s * (0.15 * t))
            lut_b.extend((v, v, v, 255))
    (d / "prior-lut-b.bin").write_bytes(lut_b)
    entries.append(asset_entry("radiance", "radiance/prior-lut-b.bin", bytes(lut_b), "prior_lut", False))

    # 4096 * 4 * f32le = 65536 B sRGB-to-linear gamma table
    gamma = bytearray()
    for i in range(4096):
        x = i / 4095.0
        lin = x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
        gamma.extend(struct.pack("<ffff", lin, lin, lin, 1.0))
    (d / "prior-gamma-rgba32f.bin").write_bytes(gamma)
    entries.append(asset_entry("radiance", "radiance/prior-gamma-rgba32f.bin", bytes(gamma), "prior_lut", False))

    # 2 * 3 * 3 * f32le = 72 B identity Rec.709 color matrices
    mats = struct.pack("<18f", 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1)
    (d / "prior-color-matrices.bin").write_bytes(mats)
    entries.append(asset_entry("radiance", "radiance/prior-color-matrices.bin", mats, "prior_lut", False))
    return entries


def generate_teeth(root: Path) -> List[Dict]:
    d = root / "teeth"
    d.mkdir(parents=True, exist_ok=True)
    entries = []

    def whiten_rgb(r: float, g: float, b: float) -> Tuple[float, float, float]:
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        # Desaturate yellow cast and gently lift luminance
        return (
            clamp01(r * 0.78 + luma * 0.22 + 0.03 * (1 - luma)),
            clamp01(g * 0.80 + luma * 0.20 + 0.04 * (1 - luma)),
            clamp01(b * 0.72 + luma * 0.28 + 0.07 * (1 - luma)),
        )

    def bright_rgb(r: float, g: float, b: float) -> Tuple[float, float, float]:
        return (r ** 0.82, g ** 0.82, b ** 0.80)

    for name, fn in [("Whiten.png", whiten_rgb), ("Bright.png", bright_rgb)]:
        data = build_hald_lut_512(fn)
        (d / name).write_bytes(data)
        entries.append(asset_entry("teeth", f"teeth/{name}", data, "prior_lut", False))

    # 111x43 elliptical dental-arch geometric prior mask
    w, h = 111, 43
    mask = bytearray(w * h * 4)
    for y in range(h):
        ny = (y - (h - 1) * 0.5) / ((h - 1) * 0.5)
        for x in range(w):
            nx = (x - (w - 1) * 0.5) / ((w - 1) * 0.5)
            r2 = nx * nx + ny * ny
            val = u8(max(0.0, 1.0 - r2))
            idx = (y * w + x) * 4
            mask[idx : idx + 4] = bytes((val, val, val, 255))
    mask_png = encode_png_rgba(w, h, bytes(mask))
    (d / "toothMask.png").write_bytes(mask_png)
    entries.append(asset_entry("teeth", "teeth/toothMask.png", mask_png, "prior_geometry", True))
    return entries


def generate_lip_luts(root: Path) -> List[Dict]:
    d = root / "lip_luts"
    d.mkdir(parents=True, exist_ok=True)
    w, h = 256, 2

    blend = bytearray(w * h * 4)
    level = bytearray(w * h * 4)
    for x in range(w):
        t = x / 255.0
        smooth = t * t * (3.0 - 2.0 * t)
        # Row 0: R = coverage blend weight; Row 1: G,B = gloss/specular weight
        i0 = x * 4
        blend[i0 : i0 + 4] = bytes((u8(smooth), 0, 0, 255))
        i1 = (w + x) * 4
        gloss = clamp01(0.5 * smooth + 0.5 * (t ** 1.5))
        blend[i1 : i1 + 4] = bytes((0, u8(gloss), u8(gloss), 255))
        # Level weight: row 0 R = luminance preservation weight
        lvl = clamp01(0.25 + 0.75 * math.sin(math.pi * t))
        level[i0 : i0 + 4] = bytes((u8(lvl), 0, 0, 255))
        level[i1 : i1 + 4] = bytes((0, 0, 0, 255))

    b_png = encode_png_rgba(w, h, bytes(blend))
    l_png = encode_png_rgba(w, h, bytes(level))
    (d / "blend_weight_lut.png").write_bytes(b_png)
    (d / "level_weight_lut.png").write_bytes(l_png)
    return [
        asset_entry("lip_luts", "lip_luts/blend_weight_lut.png", b_png, "prior_lut", False),
        asset_entry("lip_luts", "lip_luts/level_weight_lut.png", l_png, "prior_lut", False),
    ]


def generate_smile_priors(root: Path) -> List[Dict]:
    d = root / "smile"
    d.mkdir(parents=True, exist_ok=True)
    # 64-vertex parametric mouth-arch grid (4 rows x 16 cols)
    rows, cols = 4, 16
    mean = bytearray()
    normal = bytearray()
    side_l = bytearray()
    side_r = bytearray()
    for r in range(rows):
        v = r / (rows - 1) - 0.5
        for c in range(cols):
            u = c / (cols - 1) * 2.0 - 1.0
            mean.extend(struct.pack("<fff", u, v * 0.25, 0.0))
            normal.extend(struct.pack("<fff", 0.0, 0.0, 1.0))
            lift = max(0.0, u * u) * 0.15
            side_l.extend(struct.pack("<fff", -0.05 if u < 0 else 0.0, lift if u < 0 else 0.0, 0.0))
            side_r.extend(struct.pack("<fff", 0.05 if u > 0 else 0.0, lift if u > 0 else 0.0, 0.0))

    tris = bytearray()
    for r in range(rows - 1):
        for c in range(cols - 1):
            a = r * cols + c
            b = a + 1
            c0 = (r + 1) * cols + c
            d0 = c0 + 1
            tris.extend(struct.pack("<6I", a, b, c0, b, d0, c0))

    files = [
        ("mean.f32le", bytes(mean)),
        ("normal.f32le", bytes(normal)),
        ("side_left.f32le", bytes(side_l)),
        ("side_right.f32le", bytes(side_r)),
        ("triangles.u32le", bytes(tris)),
    ]
    entries = []
    for name, payload in files:
        (d / name).write_bytes(payload)
        entries.append(asset_entry("smile", f"smile/{name}", payload, "prior_geometry", True))
    return entries


def generate_face_mesh_priors(root: Path) -> List[Dict]:
    d = root / "face_mesh_priors"
    d.mkdir(parents=True, exist_ok=True)
    specs = [
        ("face_reshape_parts.f32le", 0.08),
        ("face_width_parts.f32le", 0.10),
        ("forehead_parts.f32le", 0.06),
    ]
    entries = []
    for name, amp in specs:
        buf = bytearray()
        for i in range(64):
            theta = (i / 64.0) * 2.0 * math.pi
            buf.extend(struct.pack("<ff", amp * math.cos(theta), amp * math.sin(theta)))
        payload = bytes(buf)
        (d / name).write_bytes(payload)
        entries.append(asset_entry("face_mesh_priors", f"face_mesh_priors/{name}", payload, "prior_geometry", True))
    return entries


def generate_neural_slots_contract(root: Path) -> List[Dict]:
    d = root / "neural_weight_slots"
    d.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": "aurora-neural-weight-slots/1",
        "note": "Contracts only. No synthetic or fabricated model weights are included.",
        "slots": [
            {"id": "face_detection", "format": "onnx", "input_shape": [1, 3, 256, 256], "requires_real_dataset_training": True},
            {"id": "face_mesh", "format": "onnx", "input_shape": [1, 3, 192, 192], "requires_real_dataset_training": True},
            {"id": "face_parsing_v2", "format": "onnx", "input_shape": [1, 3, 512, 512], "requires_real_dataset_training": True},
            {"id": "blemish_pf90116", "format": "onnx", "input_shape": [1, 3, 512, 512], "requires_real_dataset_training": True},
            {"id": "eye_bag_pf90116", "format": "onnx", "input_shape": [1, 3, 256, 256], "requires_real_dataset_training": True},
            {"id": "body_pose", "format": "onnx", "input_shape": [1, 3, 256, 192], "requires_real_dataset_training": True},
            {"id": "portrait_matting", "format": "onnx", "input_shape": [1, 3, 512, 512], "requires_real_dataset_training": True},
        ],
    }
    payload = (json.dumps(contract, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (d / "neural_slots_contract.json").write_bytes(payload)
    return [
        asset_entry(
            "neural_weight_slots",
            "neural_weight_slots/neural_slots_contract.json",
            payload,
            "neural_weight_slot",
            True,
        )
    ]


def asset_entry(category: str, rel_path: str, payload: bytes, asset_kind: str, needs_training: bool) -> Dict:
    return {
        "category": category,
        "file": rel_path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
        "asset_kind": asset_kind,
        "requires_real_dataset_training": needs_training,
    }


def generate_all(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: List[Dict] = []
    entries.extend(generate_skin_adjust(out_dir))
    entries.extend(generate_radiance(out_dir))
    entries.extend(generate_teeth(out_dir))
    entries.extend(generate_lip_luts(out_dir))
    entries.extend(generate_smile_priors(out_dir))
    entries.extend(generate_face_mesh_priors(out_dir))
    entries.extend(generate_neural_slots_contract(out_dir))

    manifest = {
        "schema": "aurora-portrait-priors-manifest/1",
        "categories": 7,
        "assets": entries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    sums = "\n".join(f"{e['sha256'].lower()}  {e['file']}" for e in entries) + "\n"
    (out_dir / "SHA256SUMS").write_text(sums, encoding="utf-8")
    return manifest_path


def verify_all(out_dir: Path) -> None:
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["assets"]:
        p = out_dir / entry["file"]
        if not p.is_file():
            failures.append(f"missing: {entry['file']}")
            continue
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest().upper()
        if len(data) != entry["bytes"] or digest != entry["sha256"]:
            failures.append(f"mismatch: {entry['file']}")
    if failures:
        raise SystemExit("verification failed: " + ", ".join(failures))
    print(f"verified {len(manifest['assets'])} portrait prior/LUT assets across {manifest['categories']} categories")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate & verify 7 portrait prior/LUT asset categories.")
    parser.add_argument("--out", type=Path, help="Output directory to generate assets into")
    parser.add_argument("--verify", type=Path, help="Directory containing manifest.json to verify")
    args = parser.parse_args()
    if args.out:
        manifest = generate_all(args.out)
        print(f"generated manifest: {manifest}")
    if args.verify:
        verify_all(args.verify)
    if not args.out and not args.verify:
        parser.print_help()


if __name__ == "__main__":
    main()
