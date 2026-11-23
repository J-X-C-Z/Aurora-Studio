#!/usr/bin/env python3
"""Recover the PF90008 StretchV3 processor/model registration from pixcake.exe."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import pefile


DEFAULT_EXE = Path(r"C:\Program Files\PixCake\pixcake_2\pixcake.exe")
DEFAULT_RESOURCES = DEFAULT_EXE.parent / "Resources"

IMAGE_BASE = 0x140000000
PROCESSOR_KEY = 0x1E9DD78
EFFECT_INIT = 0x147CEAB50
PROCESSOR_LOOKUP = 0x147CEACB0
MODEL_INIT = 0x144F1B6F0
CALLBACK_BUILDER = 0x1448525E7
EFFECT_VTABLE = 0x1495BF0F8
MODEL_VTABLE = 0x1491C8908
PIPELINE_VTABLE = 0x1495BF448
PARAMS_VTABLE = 0x1495BF340
ENCODED_RESOURCES = (0x100015F98, 0x100015FB5)
CALLBACK_KEYS = (
    0x1E9DA58,
    0x1E9DE40,
    PROCESSOR_KEY,
    0x1E9FD80,
    0x1EA2A6C,
    0x1EA2AD0,
)


class PeView:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.pe = pefile.PE(str(path), fast_load=True)
        self.base = self.pe.OPTIONAL_HEADER.ImageBase
        if self.base != IMAGE_BASE:
            raise ValueError(f"unexpected image base {self.base:#x}")
        self.sections = list(self.pe.sections)

    def offset(self, va: int) -> int:
        for section in self.sections:
            start = self.base + section.VirtualAddress
            if start <= va < start + section.SizeOfRawData:
                return section.PointerToRawData + va - start
        raise ValueError(f"VA not backed by raw data: {va:#x}")

    def dword(self, va: int) -> int:
        with self.path.open("rb") as file:
            file.seek(self.offset(va))
            return struct.unpack("<I", file.read(4))[0]

    def qword(self, va: int) -> int:
        with self.path.open("rb") as file:
            file.seek(self.offset(va))
            return struct.unpack("<Q", file.read(8))[0]

    def bytes(self, va: int, size: int) -> bytes:
        with self.path.open("rb") as file:
            file.seek(self.offset(va))
            return file.read(size)

    def c_string(self, va: int) -> str:
        data = self.bytes(va, 256)
        return data.split(b"\0", 1)[0].decode("ascii", "replace")

    def rtti(self, vtable: int) -> dict[str, object]:
        locator = self.qword(vtable - 8)
        signature = self.dword(locator)
        if signature != 1:
            raise ValueError(f"bad RTTI locator at {locator:#x}")
        type_rva = self.dword(locator + 12)
        class_rva = self.dword(locator + 16)
        type_descriptor = self.base + type_rva
        return {
            "vtable": f"0x{vtable:X}",
            "locator": f"0x{locator:X}",
            "type_descriptor": f"0x{type_descriptor:X}",
            "class_descriptor": f"0x{self.base + class_rva:X}",
            "name": self.c_string(type_descriptor + 16),
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_contains(path: Path, needle: bytes, label: str, start: int = 0, size: int | None = None) -> None:
    with path.open("rb") as file:
        file.seek(start)
        data = file.read(size) if size is not None else file.read()
    if needle not in data:
        raise ValueError(f"missing byte signature: {label}")


def scan_terms(path: Path, terms: tuple[bytes, ...]) -> dict[str, int]:
    counts = {term.decode("ascii"): 0 for term in terms}
    overlap = max(map(len, terms)) - 1
    carry = b""
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            data = carry + chunk
            for term in terms:
                counts[term.decode("ascii")] += data.count(term)
            carry = data[-overlap:] if overlap else b""
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--resources", type=Path, default=DEFAULT_RESOURCES)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    view = PeView(args.exe)
    effect_init = view.bytes(EFFECT_INIT, 0x100)
    lookup = view.bytes(PROCESSOR_LOOKUP, 6)
    model_init = view.bytes(MODEL_INIT, 0x3C0)
    callback_window = view.bytes(CALLBACK_BUILDER, 0x100)

    if b"\x48\xC7\x80\xB0\0\0\0\x78\xDD\xE9\x01" not in effect_init:
        raise ValueError("EffectStretchMark key store signature changed")
    if lookup != b"\x41\xB8\x78\xDD\xE9\x01":
        raise ValueError("processor lookup signature changed")
    if view.bytes(0x1497DC3A9, 17) != b"StretchV3Pipeline":
        raise ValueError("pipeline name association changed")
    if b"\x48\xC7\x47\x58\x78\xDD\xE9\x01" not in model_init:
        raise ValueError("model key store signature changed")
    for resource in ENCODED_RESOURCES:
        if struct.pack("<Q", resource) not in model_init:
            raise ValueError(f"missing model resource {resource:#x}")
    for key in CALLBACK_KEYS:
        if struct.pack("<I", key) not in callback_window:
            raise ValueError(f"missing callback key {key:#x}")

    effect_rtti = view.rtti(EFFECT_VTABLE)
    model_rtti = view.rtti(MODEL_VTABLE)
    pipeline_rtti = view.rtti(PIPELINE_VTABLE)
    params_rtti = view.rtti(PARAMS_VTABLE)
    if "EffectStretchMark@PixCook" not in effect_rtti["name"]:
        raise ValueError("EffectStretchMark RTTI mismatch")
    if "EffectStretchMarkModel@PixCook" not in model_rtti["name"]:
        raise ValueError("EffectStretchMarkModel RTTI mismatch")
    if "StretchV3Pipeline@TSAI" not in pipeline_rtti["name"]:
        raise ValueError("StretchV3Pipeline RTTI mismatch")
    if "StretchV3Params@TSAI" not in params_rtti["name"]:
        raise ValueError("StretchV3Params RTTI mismatch")

    effect_json = args.resources / "effect" / "config" / "StretchMark" / "effect.json"
    pixcook = args.resources / "pixcook"
    terms = (b"StretchV3", b"StretchMark", b"EffectStretchMark", b"15f98", b"15fb5")
    payload = {
        "effect_json": {
            "path": str(effect_json),
            "size": effect_json.stat().st_size,
            "sha256": sha256(effect_json),
            "plaintext": effect_json.read_bytes().decode("ascii", "backslashreplace")
            if all(32 <= byte < 127 for byte in effect_json.read_bytes())
            else None,
        },
        "pixcook": {
            "path": str(pixcook),
            "size": pixcook.stat().st_size,
            "plaintext_term_hits": scan_terms(pixcook, terms),
        },
    }
    payload["status"] = "blocked"
    payload["blocker"] = (
        "No plaintext StretchV3 tensor/weights or model graph exists in Resources; "
        "the only named config is a 119-byte obfuscated leaf and pixcook has none "
        "of the StretchV3/StretchMark/resource-ID strings."
    )

    result = {
        "schema": "aurora.portrait.stretchv3.registry.r26.v1",
        "binary": {
            "path": str(args.exe),
            "size": args.exe.stat().st_size,
            "timestamp": view.pe.FILE_HEADER.TimeDateStamp,
            "sha256": sha256(args.exe),
        },
        "processor": {
            "lookup_key": f"0x{PROCESSOR_KEY:X}",
            "init": f"0x{EFFECT_INIT:X}",
            "lookup_call": f"0x{PROCESSOR_LOOKUP:X}",
            "pipeline_name": "StretchV3Pipeline",
            "pipeline_name_va": "0x1497DC3A9",
            **effect_rtti,
        },
        "model": {
            "init": f"0x{MODEL_INIT:X}",
            **model_rtti,
            "resource_ids_low32": [f"0x{resource & 0xFFFFFFFF:X}" for resource in ENCODED_RESOURCES],
            "encoded_registration_words": [f"0x{resource:X}" for resource in ENCODED_RESOURCES],
            "variants": ["0x0", "0xFFFFFFFE"],
            "semantics": "Exact registration words; high32/variant labels remain unresolved",
        },
        "rtti": {
            "StretchV3Params": params_rtti,
            "StretchV3Pipeline": pipeline_rtti,
        },
        "callback_table": {
            "builder": f"0x{CALLBACK_BUILDER:X}",
            "keys": [f"0x{key:X}" for key in CALLBACK_KEYS],
            "ownership_proven_for": f"0x{PROCESSOR_KEY:X}",
        },
        "payload": payload,
        "cpu": {
            "status": "blocked",
            "blocker": payload["blocker"],
        },
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
