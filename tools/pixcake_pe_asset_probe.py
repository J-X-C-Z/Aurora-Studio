#!/usr/bin/env python3
"""Read-only catalog for PixCake PE/portrait texture assets.

The tool catalogs the installed Resources tree, Qt RCC v3 packages, the
pixcook tar payload, independent PAK files, and PE overlay data. It never
attaches to a process, writes into the installation, or treats synthetic
planes as vendor assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


NAMESPACE = "assets/PE"
ROLE_PATTERNS = {
    "beauty": ("beauty",),
    "hardlight": ("hardlight", "hard_light"),
    "skin_map": ("skin_map", "skin-map", "skinmap"),
    "blur_alpha": ("blur_alpha", "blur-alpha", "bluralpha"),
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".jxl", ".webp"}
PAK_SUFFIXES = {".pak", ".pck"}
PE_OVERLAY_NAMES = {b"assets/PE", b"assets\\PE", b"RetouchMoisturizing"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def role_matches(path: str) -> list[str]:
    lowered = path.lower().replace("\\", "/")
    return [
        role
        for role, needles in ROLE_PATTERNS.items()
        if any(needle in lowered for needle in needles)
    ]


def catalog_tree(resources: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    namespace_matches: list[str] = []
    pak_files: list[dict[str, Any]] = []
    for path in sorted(resources.rglob("*")):
        if not path.is_file():
            continue
        rel = rel_posix(path, resources)
        lowered = f"/{rel.lower()}/"
        if f"/{NAMESPACE.lower()}/" in lowered:
            namespace_matches.append(rel)
        roles = role_matches(rel)
        if roles or path.suffix.lower() in IMAGE_SUFFIXES | PAK_SUFFIXES:
            receipt = {
                "path": rel,
                "size": path.stat().st_size,
                "roles": roles,
            }
            entries.append(receipt)
        if path.suffix.lower() in PAK_SUFFIXES:
            pak_files.append(
                {
                    "path": rel,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "files_considered": sum(1 for path in resources.rglob("*") if path.is_file()),
        "namespace_matches": namespace_matches,
        "candidate_files": entries,
        "independent_paks": pak_files,
    }


def catalog_pixcook(path: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    namespace_matches: list[str] = []
    samplers: dict[str, list[dict[str, Any]]] = {}
    roots: dict[str, int] = {}
    with path.open("rb") as stream:
        offset = 0
        index = 0
        while True:
            header = stream.read(512)
            if len(header) < 512 or not header[:100].strip(b"\0"):
                break
            name = header[:100].split(b"\0", 1)[0].decode("utf-8", "replace")
            if not name:
                raise ValueError(f"{path}: entry {index} has an empty tar name")
            try:
                size = int(header[124:136].split(b"\0", 1)[0].strip() or b"0", 8)
            except ValueError as error:
                raise ValueError(f"{path}: entry {index} has a bad tar size") from error
            data_offset = offset + 512
            data = stream.read(size)
            padding = (-size) % 512
            if padding:
                stream.read(padding)
            lowered = f"/{name.lower().replace(chr(92), '/')}/"
            if f"/{NAMESPACE.lower()}/" in lowered:
                namespace_matches.append(name)
            root = name.split("/", 1)[0] if "/" in name else "<root>"
            roots[root] = roots.get(root, 0) + 1
            roles = role_matches(name)
            if roles:
                entries.append(
                    {
                        "index": index,
                        "path": name,
                        "size": size,
                        "data_offset": data_offset,
                        "sha256": sha256_bytes(data),
                        "roles": roles,
                    }
                )
            if name.endswith((".frag", ".vert")):
                text = data.decode("ascii", "ignore")
                for sampler in (
                    "s_beauty_map_texture",
                    "s_hardlight_map_texture",
                    "s_skin_map_texture",
                    "s_blur_texture",
                ):
                    if sampler in text:
                        samplers.setdefault(sampler, []).append(
                            {
                                "path": name,
                                "size": size,
                                "data_offset": data_offset,
                                "sha256": sha256_bytes(data),
                            }
                        )
            index += 1
            offset = data_offset + size + padding
    return {
        "path": path.name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "entries": index,
        "roots": roots,
        "namespace_matches": namespace_matches,
        "candidate_entries": entries,
        "samplers": samplers,
    }


def rcc_name(data: bytes, names_offset: int, rel: int) -> str:
    offset = names_offset + rel
    length, _hash = struct.unpack_from(">HI", data, offset)
    return data[offset + 6 : offset + 6 + length * 2].decode("utf-16-be")


def catalog_rcc(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    magic, version, tree_offset, data_offset, names_offset = struct.unpack_from(
        ">4sIIII", data, 0
    )
    if magic != b"qres" or version != 3:
        raise ValueError(f"{path}: unsupported RCC {magic!r} version {version}")
    entries: list[dict[str, Any]] = []
    namespace_matches: list[str] = []
    seen: set[int] = set()

    def walk(node_index: int, parent: str) -> None:
        if node_index in seen:
            raise ValueError(f"{path}: cyclic RCC node {node_index}")
        seen.add(node_index)
        node = tree_offset + node_index * 22
        name_offset, flags = struct.unpack_from(">IH", data, node)
        name = rcc_name(data, names_offset, name_offset)
        current = parent + "/" + name if name and node_index else parent
        lowered = f"/{current.lower()}/"
        if f"/{NAMESPACE.lower()}/" in lowered:
            namespace_matches.append(current)
        if flags & 2:
            count, first = struct.unpack_from(">II", data, node + 6)
            for child in range(count):
                walk(first + child, current)
            return
        _country, _language, payload_offset = struct.unpack_from(">HHI", data, node + 6)
        roles = role_matches(current)
        if roles:
            entries.append(
                {
                    "path": current,
                    "data_offset": payload_offset,
                    "roles": roles,
                }
            )

    walk(0, "")
    return {
        "path": path.name,
        "size": len(data),
        "sha256": sha256_bytes(data),
        "entries": len(seen),
        "data_offset": data_offset,
        "namespace_matches": namespace_matches,
        "candidate_entries": entries,
    }


def pe_overlay(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    section_count = struct.unpack_from("<H", data, pe_offset + 6)[0]
    optional_size = struct.unpack_from("<H", data, pe_offset + 20)[0]
    section_table = pe_offset + 24 + optional_size
    mapped_end = section_table + section_count * 40
    for index in range(section_count):
        node = section_table + index * 40
        raw_size, raw_offset = struct.unpack_from("<II", data, node + 16)
        mapped_end = max(mapped_end, raw_offset + raw_size)
    overlay = data[mapped_end:]
    return {
        "path": path.name,
        "size": len(data),
        "sha256": sha256_bytes(data),
        "overlay_offset": mapped_end,
        "overlay_size": len(overlay),
        "overlay_sha256": sha256_bytes(overlay),
        "namespace_matches": [
            needle.decode("ascii") for needle in PE_OVERLAY_NAMES if needle in overlay
        ],
        "overlay_head_hex": overlay[:32].hex(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--install-root", default=r"C:\Program Files\PixCake\pixcake_2", type=Path
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.install_root.resolve()
    resources = root / "Resources"
    if not resources.is_dir():
        raise SystemExit(f"missing Resources directory: {resources}")

    tree = catalog_tree(resources)
    pixcook_path = resources / "pixcook"
    pixcook = catalog_pixcook(pixcook_path) if pixcook_path.is_file() else None
    rccs = [
        catalog_rcc(path)
        for path in sorted(resources.glob("*.rcc"))
        if path.is_file()
    ]
    overlays = [
        pe_overlay(path)
        for path in (root / "pixcake.exe", root / "pix-worker.exe")
        if path.is_file()
    ]

    sampler_paths = sorted(
        {
            item["path"]
            for rcc in rccs
            for item in rcc["candidate_entries"]
        }
        | {
            item["path"]
            for item in (pixcook["candidate_entries"] if pixcook else [])
        }
    )
    namespace_hits = (
        tree["namespace_matches"]
        + (pixcook["namespace_matches"] if pixcook else [])
        + [hit for rcc in rccs for hit in rcc["namespace_matches"]]
        + [hit for overlay in overlays for hit in overlay["namespace_matches"]]
    )
    catalog = {
        "schema": "aurora/pixcake-pe-assets/1",
        "namespace": NAMESPACE,
        "install_root": str(root),
        "status": "READY" if namespace_hits else "BLOCKED_NO_ASSETS_PE_NAMESPACE",
        "namespace_matches": namespace_hits,
        "resources_tree": tree,
        "pixcook": pixcook,
        "rcc_packages": rccs,
        "pe_overlays": overlays,
        "shader_samplers": pixcook["samplers"] if pixcook else {},
        "candidate_paths": sampler_paths,
        "moisture_bindings": {
            role: {
                "status": "MISSING_VENDOR_ASSET",
                "named_refusal": f'MissingInput("PF90002_{role.upper()}")',
            }
            for role in ROLE_PATTERNS
        },
        "blocker": None
        if namespace_hits
        else {
            "kind": "namespace",
            "exact": NAMESPACE,
            "detail": (
                "No assets/PE path exists in the public Resources tree, "
                "pixcook tar names, Qt RCC names, independent PAK names, "
                "or PE overlay bytes."
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(catalog["status"])
    return 0 if namespace_hits else 2


if __name__ == "__main__":
    raise SystemExit(main())
