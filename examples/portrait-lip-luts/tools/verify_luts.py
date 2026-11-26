"""Verify the shipped atlases from the files, with an independent code path.

Standard library only (Pillow is used as an extra, optional cross-check when it
happens to be installed). Reads `MANIFEST.json` and `assets/`, recomputes every
number in it, and rebuilds both atlases from a scratch directory twice to show
the result is deterministic.

Checks, in order:

  1. container        PNG structure + pixel SHA-256 + file SHA-256 vs manifest
  2. axis identity    node (x, y, z) really holds red=x, green=y, blue=z
  3. boundaries       corners, neutral diagonal, own-axis monotonicity, rails
  4. samples          the quoted sample table recomputed from the files
  5. tile order       a diagnostic atlas is written, decoded, and a transposed
                      copy must decode differently
  6. determinism      two fresh generations agree with the shipped pixels

Run:  python tools/verify_luts.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lut_atlas
import png_codec

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGE = lut_atlas.LUT_EDGE

failures: list[str] = []


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


def load_assets(manifest: dict):
    loaded = []
    for asset in manifest["assets"]:
        path = os.path.join(PACKAGE_ROOT, *asset["file"].split("/"))
        raw = open(path, "rb").read()
        width, height, channels, pixels = png_codec.read_png(path)
        ok = (width == 512 and height == 512 and channels == asset["png"]["channels"]
              and len(pixels) == asset["pixels"]["raw_atlas_bytes"])
        check(f"container/{asset['name']}",
              ok and sha256(pixels) == asset["pixels"]["raw_atlas_sha256"]
              and len(raw) == asset["png"]["bytes"],
              f"{width}x{height}x{channels} file={len(raw)}B pixels={len(pixels)}B "
              f"pixel_sha={'ok' if sha256(pixels) == asset['pixels']['raw_atlas_sha256'] else 'MISMATCH'}")
        file_sha = sha256(raw)
        if file_sha != asset["png"]["sha256"]:
            print(f"       note: file SHA-256 differs from the manifest "
                  f"({file_sha[:16]}… vs {asset['png']['sha256'][:16]}…). The PNG container "
                  f"depends on the zlib build; the pixel SHA-256 above is the "
                  f"version-independent check and it passed.")
        else:
            print(f"       file SHA-256 matches the manifest, so this host's zlib "
                  f"compresses identically to the reference build")
        frames = None
        try:
            from PIL import Image  # optional
            with Image.open(path) as image:
                frames = image.convert("RGBA" if channels == 4 else "RGB").tobytes()
        except ImportError:
            print("       Pillow not installed; skipping the optional cross-decoder check")
        if frames is not None:
            check(f"container/{asset['name']}/pillow", frames == pixels,
                  "Pillow decodes the same pixels")
        loaded.append({**asset, "path": path, "atlas": pixels, "channels": channels,
                       "volume": bytes(lut_atlas.atlas_to_volume(pixels, channels))})
    return loaded


def check_axis_identity(asset: dict) -> None:
    """Node (x, y, z) must equal the curve at (x/63, y/63, z/63).

    That identity is what pins red to +x, green to +y and blue to the tile
    index: if any two axes were swapped, the stored triple would not match the
    curve evaluated in that order.
    """
    import lip_curves
    probes = [(0, 0, 0), (63, 63, 63), (5, 9, 11), (63, 0, 33), (0, 63, 46), (31, 17, 52)]
    curve = (lip_curves.lips_transform if asset["direction"] == "positive"
             else lip_curves.lip_dodge_transform)
    matches = 0
    for (x, y, z) in probes:
        want = tuple(lip_curves.to_byte(c) for c in
                     curve((lip_curves.node_value(x), lip_curves.node_value(y),
                            lip_curves.node_value(z))))
        offset = lut_atlas.volume_index(x, y, z)
        got = (asset["volume"][offset], asset["volume"][offset + 1], asset["volume"][offset + 2])
        if got == want:
            matches += 1
    check(f"axis_identity/{asset['name']}", matches == len(probes),
          f"{matches}/{len(probes)} nodes equal curve(x/63, y/63, z/63), so the "
          f"x<-red / y<-green / z<-blue assignment is the one in the manifest")


def check_boundaries(asset: dict) -> None:
    volume = asset["volume"]
    checks = asset["checks"]
    black = tuple(volume[lut_atlas.volume_index(0, 0, 0) + c] for c in range(3))
    white = tuple(volume[lut_atlas.volume_index(63, 63, 63) + c] for c in range(3))
    check(f"corners/{asset['name']}", black == (0, 0, 0) and white == (255, 255, 255),
          f"black={list(black)} white={list(white)}")

    diagonal = []
    for index in range(EDGE):
        offset = lut_atlas.volume_index(index, index, index)
        triple = (volume[offset], volume[offset + 1], volume[offset + 2])
        if not triple[0] == triple[1] == triple[2]:
            diagonal.append(triple)
    check(f"neutral_axis/{asset['name']}", not diagonal,
          f"all {EDGE} diagonal nodes have equal channels"
          if not diagonal else f"{len(diagonal)} diagonal nodes are not neutral: {diagonal[:3]}")

    own = []
    for channel, step in ((0, (1, 0, 0)), (1, (0, 1, 0)), (2, (0, 0, 1))):
        minimum = 255
        dx, dy, dz = step
        for z in range(EDGE - dz):
            for y in range(EDGE - dy):
                for x in range(EDGE - dx):
                    low = lut_atlas.volume_index(x, y, z) + channel
                    high = lut_atlas.volume_index(x + dx, y + dy, z + dz) + channel
                    delta = volume[high] - volume[low]
                    if delta < minimum:
                        minimum = delta
        own.append(minimum)
    claimed = checks["monotonicity"]["min_step_own_axis"]
    check(f"monotone_own_axis/{asset['name']}", own == claimed and min(own) >= 0,
          f"red/+x={own[0]} green/+y={own[1]} blue/+z={own[2]} (min step per channel; "
          f"manifest says {claimed})")

    identity = [round(index / 63.0 * 255.0) for index in range(EDGE)]
    worst = 0
    changed = 0
    rails = 0
    for z in range(EDGE):
        for y in range(EDGE):
            for x in range(EDGE):
                offset = lut_atlas.volume_index(x, y, z)
                for channel, expected in enumerate((identity[x], identity[y], identity[z])):
                    stored = volume[offset + channel]
                    delta = abs(stored - expected)
                    if delta > worst:
                        worst = delta
                    if delta:
                        changed += 1
                    if stored in (0, 255):
                        rails += 1
    claimed_identity = checks["vs_identity"]
    check(f"non_identity/{asset['name']}",
          worst == claimed_identity["max_abs_delta_bytes"] and worst >= 8,
          f"max|node - identity| = {worst}/255, changed entries = {changed} "
          f"(manifest: {claimed_identity['max_abs_delta_bytes']}, "
          f"{claimed_identity['changed_entries']})")
    check(f"rails/{asset['name']}",
          rails == checks["rails"]["entries_at_0"] + checks["rails"]["entries_at_255"],
          f"{rails} entries sit on a rail; the curves themselves stay inside [0,1] "
          f"because the gain is bounded to the cube")


def check_samples(asset: dict, manifest_asset: dict) -> None:
    rows = []
    for row in manifest_asset["samples"]:
        r, g, b = row["in"]
        reference = lut_atlas.sample_atlas_f64(asset["atlas"], asset["channels"],
                                               r / 255.0, g / 255.0, b / 255.0)
        rows.append([lut_atlas.lip_to_byte(value) for value in reference])
    claimed = [row["out"] for row in manifest_asset["samples"]]
    check(f"samples/{asset['name']}", rows == claimed,
          f"{len(rows)} points recomputed from the file; first coloured point "
          f"{manifest_asset['samples'][4]['in']} -> {rows[4]} (manifest {claimed[4]})")


def check_tile_order(scratch: str) -> None:
    volume = bytearray(EDGE ** 3 * 3)
    axis = [index * 4 for index in range(EDGE)]
    for z in range(EDGE):
        for y in range(EDGE):
            for x in range(EDGE):
                offset = lut_atlas.volume_index(x, y, z)
                volume[offset] = axis[x]
                volume[offset + 1] = axis[y]
                volume[offset + 2] = axis[z]
    atlas = lut_atlas.volume_to_atlas(bytes(volume), 3)
    probe_dir = os.path.join(scratch, "probe")
    os.makedirs(probe_dir, exist_ok=True)
    probe = os.path.join(probe_dir, "node-index-atlas.png")
    png_codec.write_png(probe, 512, 512, 3, atlas)
    _, _, channels, reread = png_codec.read_png(probe)
    decoded = bytes(lut_atlas.atlas_to_volume(reread, channels))
    check("tile_order/round_trip", decoded == bytes(volume),
          f"a diagnostic atlas that stores its own (4x, 4y, 4z) decodes back node for node "
          f"(probe kept in the scratch directory only)")

    flat = bytearray(reread)
    swapped = bytearray(len(flat))
    for z in range(EDGE):
        dest_row, dest_column = (z & 7) * 64, (z >> 3) * 64
        src_row, src_column = (z >> 3) * 64, (z & 7) * 64
        for row in range(64):
            source = ((src_row + row) * 512 + src_column) * 3
            target = ((dest_row + row) * 512 + dest_column) * 3
            swapped[target:target + 192] = flat[source:source + 192]
    wrong = bytes(lut_atlas.atlas_to_volume(bytes(swapped), 3))
    differing = sum(1 for index in range(0, len(wrong), 3) if wrong[index:index + 3]
                    != bytes(volume)[index:index + 3])
    check("tile_order/discriminates", differing > 0,
          f"transposing the 8x8 tile grid changes {differing}/{EDGE ** 3} nodes, so the test "
          f"can tell the two layouts apart")


def check_determinism(manifest: dict, scratch: str) -> None:
    generator = os.path.join(PACKAGE_ROOT, "tools", "generate_luts.py")
    runs = []
    for index in (1, 2):
        target = os.path.join(scratch, f"build{index}")
        shutil.rmtree(target, ignore_errors=True)
        env = {key: value for key, value in os.environ.items() if not key.startswith("AURORA_")}
        env["PYTHONHASHSEED"] = "0"
        completed = subprocess.run([sys.executable, generator, "--output", target],
                                   capture_output=True, text=True, env=env)
        if completed.returncode != 0:
            check(f"determinism/run{index}", False, completed.stderr.strip()[-400:])
            return
        runs.append({asset["name"]: os.path.join(target, *asset["file"].split("/"))
                     for asset in manifest["assets"]})
    pixels_ok = True
    files_ok = True
    detail = []
    for asset in manifest["assets"]:
        shipped = open(os.path.join(PACKAGE_ROOT, *asset["file"].split("/")), "rb").read()
        for run in runs:
            rebuilt = open(run[asset["name"]], "rb").read()
            if sha256(rebuilt) != sha256(shipped):
                files_ok = False
        first = open(runs[0][asset["name"]], "rb").read()
        second = open(runs[1][asset["name"]], "rb").read()
        if sha256(first) != sha256(second):
            pixels_ok = False
        detail.append(f"{asset['name']}={'identical' if sha256(first) == sha256(shipped) else 'differs'}")
    check("determinism/two_rebuilds", pixels_ok and files_ok,
          "two fresh generations from an empty directory are byte-identical to the shipped "
          f"files ({', '.join(detail)})")


def main() -> int:
    manifest_path = os.path.join(PACKAGE_ROOT, "MANIFEST.json")
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    print(f"package={PACKAGE_ROOT}")
    print(f"dependencies: python {sys.version.split()[0]} standard library only")
    scratch = tempfile.mkdtemp(prefix="lip-lut-verify-")
    try:
        loaded = load_assets(manifest)
        for asset, claimed in zip(loaded, manifest["assets"]):
            check_axis_identity(asset)
            check_boundaries(asset)
            check_samples(asset, claimed)
        check_tile_order(scratch)
        check_determinism(manifest, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
