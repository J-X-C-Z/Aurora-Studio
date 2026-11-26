"""Generate the two lip LUT atlases and their manifest.

Standard library only. Writes, into `--output` (default: this package root):

    assets/LipsLUT.png      512x512 RGBA8 atlas, 64^3 volume, positive row
    assets/LipDodgeLUT.png  512x512 RGB8  atlas, 64^3 volume, negative row
    MANIFEST.json           bytes, SHA-256 of the file and of the pixels, the
                            format contract, the curve parameters, the
                            per-asset constraint results and the sample table

Nothing outside this package is read, no environment variable is consulted and
there is no network access, so a copy of this directory can regenerate the
assets from scratch:

    python tools/generate_luts.py --output /tmp/luts

The shipped `assets/` files were produced by this script; `tools/verify_luts.py`
re-derives every number in the manifest from the files themselves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lip_curves
import lut_atlas
import png_codec

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGE = lut_atlas.LUT_EDGE

SAMPLE_INPUTS = [
    (0, 0, 0),
    (255, 255, 255),
    (128, 128, 128),
    (77, 26, 32),
    (153, 51, 64),
    (200, 60, 70),
    (230, 120, 130),
    (63, 0, 0),
    (0, 0, 63),
    (0, 63, 0),
]

ASSETS = [
    {
        "name": "LipsLUT",
        "file": "assets/LipsLUT.png",
        "direction": "positive",
        "channels": 4,
        "colour_type": 6,
        "transform": lip_curves.lips_transform,
        "gain": lip_curves.lips_gain,
        "curve": {
            "id": "lips-luma-preserving-chroma-boost",
            "formula": "out = Y + gain * (c - Y), gain = min(1 + 0.30 * bump(Y), cube room)",
            "luma": "Rec.709 (0.2126, 0.7152, 0.0722)",
            "window": "bump(y) = 4y(1-y)",
            "target_luma": "Y, unchanged",
            "fixed_points": "(0,0,0)->(0,0,0); (1,1,1)->(1,1,1); the neutral axis is the identity",
        },
    },
    {
        "name": "LipDodgeLUT",
        "file": "assets/LipDodgeLUT.png",
        "direction": "negative",
        "channels": 3,
        "colour_type": 2,
        "transform": lip_curves.lip_dodge_transform,
        "gain": lip_curves.lip_dodge_gain,
        "curve": {
            "id": "lip-dodge-desaturate-and-lift",
            "formula": ("out = Yd + gain * (c - Y), gain = min(1 - 0.35 * bump(Y), cube room), "
                        "Yd = Y + 0.05 * bump(Y)"),
            "luma": "Rec.709 (0.2126, 0.7152, 0.0722)",
            "window": "bump(y) = 4y(1-y)",
            "target_luma": "Yd = Y + 0.05 * bump(Y): mid-luma is lifted, both ends stay fixed",
            "fixed_points": ("(0,0,0)->(0,0,0); (1,1,1)->(1,1,1); the neutral axis stays "
                             "neutral (all three channels equal) but is lifted"),
        },
    },
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_volume(spec) -> tuple[bytearray, int]:
    """Evaluate the curve over the 64^3 grid in (z, y, x) order.

    Returns the volume and how many nodes had their chroma gain limited by the
    RGB cube rather than by the curve.
    """
    volume = bytearray(EDGE ** 3 * 3)
    bound_active = 0
    for z in range(EDGE):
        b = lip_curves.node_value(z)
        for y in range(EDGE):
            g = lip_curves.node_value(y)
            for x in range(EDGE):
                r = lip_curves.node_value(x)
                rgb = (r, g, b)
                values = spec["transform"](rgb)
                offset = lut_atlas.volume_index(x, y, z)
                for channel in range(3):
                    volume[offset + channel] = lip_curves.to_byte(values[channel])
                used, wanted = spec["gain"](rgb)
                if used < wanted - 1e-12:
                    bound_active += 1
    return volume, bound_active


def _axis_min_steps(volume: bytes, dx: int, dy: int, dz: int) -> list[int]:
    """Per-channel minimum step along one axis direction."""
    minimum = [255, 255, 255]
    for z in range(EDGE - dz):
        for y in range(EDGE - dy):
            for x in range(EDGE - dx):
                low = lut_atlas.volume_index(x, y, z)
                high = lut_atlas.volume_index(x + dx, y + dy, z + dz)
                for channel in range(3):
                    step = volume[high + channel] - volume[low + channel]
                    if step < minimum[channel]:
                        minimum[channel] = step
    return minimum


def monotonicity(volume: bytes) -> dict:
    """Each channel is non-decreasing along its own axis; the rest is reported.

    The volume axis order is (z, y, x) with x<-red, y<-green, z<-blue, so the
    meaningful pairs are red along +x, green along +y and blue along +z. The
    other six channel/axis pairs are measured and reported, not claimed.
    """
    axes = {
        "x_red_axis": _axis_min_steps(volume, 1, 0, 0),
        "y_green_axis": _axis_min_steps(volume, 0, 1, 0),
        "z_blue_axis": _axis_min_steps(volume, 0, 0, 1),
    }
    return {
        "own_axis_pairs": ["red/+x", "green/+y", "blue/+z"],
        "min_step_own_axis": [axes["x_red_axis"][0], axes["y_green_axis"][1],
                              axes["z_blue_axis"][2]],
        "min_step_every_channel_axis_pair": axes,
        "note": ("the two rows are different claims: only the own-axis triple is required to be "
                 "non-decreasing; the remaining six pairs are informational"),
    }


def check_asset(spec, volume: bytearray, bound_active: int) -> dict:
    report = {}
    black = list(volume[lut_atlas.volume_index(0, 0, 0):lut_atlas.volume_index(0, 0, 0) + 3])
    white = list(volume[lut_atlas.volume_index(EDGE - 1, EDGE - 1, EDGE - 1):
                        lut_atlas.volume_index(EDGE - 1, EDGE - 1, EDGE - 1) + 3])
    if black != [0, 0, 0] or white != [255, 255, 255]:
        raise SystemExit(f"{spec['name']}: corners are {black} / {white}")
    report["corners"] = {"black": black, "white": white}

    diagonal = []
    for index in range(EDGE):
        offset = lut_atlas.volume_index(index, index, index)
        r, g, b = volume[offset], volume[offset + 1], volume[offset + 2]
        if not (r == g == b):
            raise SystemExit(f"{spec['name']}: neutral axis is not neutral at node {index}")
        diagonal.append(r)
    report["neutral_axis"] = {
        "exactly_neutral_on_the_diagonal": True,
        "stored_value": diagonal,
    }

    mono = monotonicity(volume)
    if min(mono["min_step_own_axis"]) < 0:
        raise SystemExit(f"{spec['name']}: own-axis monotonicity violated: "
                         f"{mono['min_step_own_axis']}")
    report["monotonicity"] = mono

    identity = [round(index / 63.0 * 255.0) for index in range(EDGE)]
    worst = 0
    total = 0
    changed = 0
    at_0 = 0
    at_255 = 0
    for z in range(EDGE):
        for y in range(EDGE):
            for x in range(EDGE):
                offset = lut_atlas.volume_index(x, y, z)
                for channel, expected in enumerate((identity[x], identity[y], identity[z])):
                    stored = volume[offset + channel]
                    delta = abs(stored - expected)
                    total += delta
                    if delta:
                        changed += 1
                    if delta > worst:
                        worst = delta
                    if stored == 0:
                        at_0 += 1
                    elif stored == 255:
                        at_255 += 1
    entries = EDGE ** 3 * 3
    if worst < 8:
        raise SystemExit(f"{spec['name']}: deviation from identity {worst} is below 8/255")
    report["vs_identity"] = {
        "max_abs_delta_bytes": worst,
        "mean_abs_delta_bytes": round(total / entries, 6),
        "changed_entries": changed,
        "changed_fraction": round(changed / entries, 6),
    }
    report["rails"] = {
        "entries_at_0": at_0,
        "entries_at_255": at_255,
        "nodes_limited_by_the_cube_bound": bound_active,
        "note": ("a stored 0 or 255 means the true value was within half a byte of that rail; "
                 "the curves never leave [0,1] because the gain is bounded to the cube"),
    }
    return report


def sample_table(atlas: bytes, channels: int) -> list:
    rows = []
    for (r, g, b) in SAMPLE_INPUTS:
        reference = lut_atlas.sample_atlas_f64(atlas, channels, r / 255.0, g / 255.0, b / 255.0)
        rows.append({
            "in": [r, g, b],
            "out": [lut_atlas.lip_to_byte(value) for value in reference],
            "reference_float": [round(float(value), 8) for value in reference],
        })
    return rows


def build(output_root: str) -> dict:
    os.makedirs(os.path.join(output_root, "assets"), exist_ok=True)
    manifest = {
        "package": "portrait-lip-luts",
        "about": ("Two original 64^3 lip colour LUT atlases for the PF90021 lip makeup "
                  "enhance slot, generated by the curves in tools/lip_curves.py."),
        "generated_by": "tools/generate_luts.py",
        "dependencies": "Python 3.9+ standard library only",
        "format": {
            "atlas": "PNG, 8-bit, non-interlaced, 512x512, 8x8 tiles of 64x64",
            "tile_index": "z = 8*t + s, px = (z & 7)*64 + x, py = (z >> 3)*64 + y, top row first",
            "axes": "red -> +x inside a tile, green -> +y, blue selects the tile",
            "channels": ("the positive row is RGBA8; its alpha byte is written 255 and is inert "
                         "because the loader path calls to_rgb8(). The negative row is RGB8."),
            "byte_counts": ("a 512x512 RGBA8 atlas is 1048576 raw bytes; a 512x512 RGB8 atlas is "
                            "786432 raw bytes; the decoded 64^3 RGB volume is also 786432 bytes"),
            "cpu_oracle_input": "RGB8, exactly 512*512*3 bytes",
        },
        "provenance": {
            "origin": ("Authored for this project from the analytic curves in this package. No "
                       "vendor LUT, model, photograph or third-party file was read, sampled or "
                       "traced to produce them, and no neural weights are involved."),
            "license": "No separate licence is added; this contribution does not change repository licensing.",
            "verified_against": ("the atlas layout implemented by "
                                 "crates/portrait/src/cuda_pf90021.rs "
                                 "(VolumeLut64::from_atlas_rgb8) and by "
                                 "crates/portrait/src/kernels/face_detail.rs (lut3d_sample)"),
            "scope": ("Format and pixels only. Believing these files are read by a running "
                      "product build requires that build's own resolver to point at them; see "
                      "the end of README.md for what exists on which branch."),
        },
        "assets": [],
    }
    for spec in ASSETS:
        volume, bound_active = build_volume(spec)
        checks = check_asset(spec, volume, bound_active)
        atlas = lut_atlas.volume_to_atlas(bytes(volume), spec["channels"])
        path = os.path.join(output_root, *spec["file"].split("/"))
        size = png_codec.write_png(path, lut_atlas.ATLAS_EDGE, lut_atlas.ATLAS_EDGE,
                                   spec["channels"], atlas)
        width, height, channels, reread = png_codec.read_png(path)
        if (width, height, channels) != (512, 512, spec["channels"]) or reread != atlas:
            raise SystemExit(f"{spec['file']}: PNG round trip failed")
        if bytes(lut_atlas.atlas_to_volume(reread, channels)) != bytes(volume):
            raise SystemExit(f"{spec['file']}: atlas does not decode back to the volume")
        manifest["assets"].append({
            "name": spec["name"],
            "file": spec["file"],
            "direction": spec["direction"],
            "png": {
                "width": 512, "height": 512, "bit_depth": 8,
                "colour_type": spec["colour_type"], "channels": spec["channels"],
                "interlace": 0, "row_filter": "0 (None) on every row",
                "bytes": size,
                "sha256": sha256(open(path, "rb").read()),
            },
            "pixels": {
                "raw_atlas_bytes": len(atlas),
                "raw_atlas_sha256": sha256(atlas),
                "decoded_volume_bytes": len(volume),
                "decoded_volume_sha256": sha256(bytes(volume)),
            },
            "curve": spec["curve"],
            "checks": checks,
            "samples": sample_table(atlas, spec["channels"]),
        })
    path = os.path.join(output_root, "MANIFEST.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the two lip LUT atlases.")
    parser.add_argument("--output", default=PACKAGE_ROOT,
                        help="directory to write assets/ and MANIFEST.json into "
                             "(default: this package)")
    args = parser.parse_args()
    manifest = build(args.output)
    for asset in manifest["assets"]:
        print(f"{asset['file']}  bytes={asset['png']['bytes']}  sha256={asset['png']['sha256']}")
        print(f"  pixels={asset['pixels']['raw_atlas_sha256']}  "
              f"max_delta={asset['checks']['vs_identity']['max_abs_delta_bytes']}/255  "
              f"changed={asset['checks']['vs_identity']['changed_fraction']:.3f}")
        print(f"  sample {asset['samples'][4]['in']} -> {asset['samples'][4]['out']}")
    print(f"manifest={os.path.join(args.output, 'MANIFEST.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
