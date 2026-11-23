#!/usr/bin/env python3
"""Read-only Capture One Mat/draw-buffer capture over ReadProcessMemory.

The process handle requests only query and VM_READ access. This tool never
injects, suspends, debugs, or writes target memory.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
from typing import Any


SCHEMA = "aurora.c1-readprocess-capture/1"
FIXED_PE_SHA256 = "a84519da489bc52e4e914af7880ede43e3ce58992657723d02197161832464c7"
PRODUCER_RVA = 0x823A3B0
CONSUMER_RVA = 0x82AA2E0
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_VM_READ = 0x0010
LIST_MODULES_ALL = 0x03
MEM_COMMIT = 0x1000
MEM_PRIVATE = 0x20000
PAGE_GUARD = 0x100
PAGE_NOACCESS = 0x01
READABLE_WRITABLE = {0x04, 0x08, 0x40, 0x80}
USER_MAX = 0x00007FFFFFFEFFFF
IMAGE_SCN_MEM_EXECUTE = 0x20000000


class CaptureError(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass
class Request:
    width: int
    height: int
    roi: tuple[int, int, int, int]
    dtype: str
    channels: int
    expected_stride: int | None
    max_capture_bytes: int
    stable_reads: int

    @property
    def element_size(self) -> int:
        return (1 if self.dtype == "u8" else 4) * self.channels


def fail(code: str, detail: str) -> None:
    raise CaptureError(code, detail)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hexptr(value: int | None) -> str | None:
    return None if value is None else f"0x{value:X}"


if os.name != "nt":
    fail("WINDOWS_ONLY", "ReadProcessMemory capture requires Windows")

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)
K32 = ctypes.WinDLL("kernel32", use_last_error=True)
HANDLE = wintypes.HANDLE
DWORD = wintypes.DWORD
HMODULE = wintypes.HMODULE
LPVOID = ctypes.c_void_p
SIZE_T = ctypes.c_size_t


class MODULEINFO(ctypes.Structure):
    _fields_ = [
        ("lpBaseOfDll", LPVOID),
        ("SizeOfImage", DWORD),
        ("EntryPoint", LPVOID),
    ]


k32.OpenProcess.argtypes = [DWORD, wintypes.BOOL, DWORD]
k32.OpenProcess.restype = HANDLE
k32.CloseHandle.argtypes = [HANDLE]
k32.ReadProcessMemory.argtypes = [HANDLE, LPVOID, LPVOID, SIZE_T, ctypes.POINTER(SIZE_T)]
k32.ReadProcessMemory.restype = wintypes.BOOL
k32.VirtualQueryEx.argtypes = [HANDLE, LPVOID, LPVOID, SIZE_T]
k32.VirtualQueryEx.restype = SIZE_T
k32.QueryFullProcessImageNameW.argtypes = [HANDLE, DWORD, wintypes.LPWSTR, ctypes.POINTER(DWORD)]
k32.QueryFullProcessImageNameW.restype = wintypes.BOOL
psapi.EnumProcesses.argtypes = [ctypes.POINTER(DWORD), DWORD, ctypes.POINTER(DWORD)]
psapi.EnumProcesses.restype = wintypes.BOOL
psapi.EnumProcessModulesEx.argtypes = [
    HANDLE, ctypes.POINTER(HMODULE), DWORD, ctypes.POINTER(DWORD), DWORD
]
psapi.EnumProcessModulesEx.restype = wintypes.BOOL
psapi.GetModuleFileNameExW.argtypes = [HANDLE, HMODULE, wintypes.LPWSTR, DWORD]
psapi.GetModuleFileNameExW.restype = DWORD
psapi.GetModuleInformation.argtypes = [HANDLE, HMODULE, ctypes.POINTER(MODULEINFO), DWORD]
psapi.GetModuleInformation.restype = wintypes.BOOL


def open_readonly(pid: int) -> HANDLE:
    handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, pid)
    if not handle:
        fail("BLOCKED_PROCESS_OPEN", f"OpenProcess(pid={pid}) failed winerr={ctypes.get_last_error()}")
    return handle


def process_path(handle: HANDLE) -> str | None:
    size = DWORD(32768)
    buffer = ctypes.create_unicode_buffer(size.value)
    if not k32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
        return None
    return buffer.value


def list_c1_processes() -> list[dict[str, Any]]:
    pids = (DWORD * 4096)()
    needed = DWORD()
    if not psapi.EnumProcesses(pids, ctypes.sizeof(pids), ctypes.byref(needed)):
        fail("BLOCKED_PROCESS_ENUM", f"EnumProcesses failed winerr={ctypes.get_last_error()}")
    found: list[dict[str, Any]] = []
    for pid in pids[: needed.value // ctypes.sizeof(DWORD)]:
        if not pid or pid == os.getpid():
            continue
        handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            continue
        try:
            path = process_path(handle)
            name = Path(path).name.casefold() if path else ""
            if name in {"captureone.exe", "capture one.exe", "portrait.exe"}:
                found.append({"pid": int(pid), "name": name, "path": path})
        finally:
            k32.CloseHandle(handle)
    return found


def read_remote(handle: HANDLE, address: int, size: int) -> bytes:
    if size < 0 or not 0x10000 <= address <= USER_MAX or address + size > USER_MAX + 1:
        fail("BLOCKED_INVALID_ADDRESS", f"invalid read range {hexptr(address)}+{size}")
    output = (ctypes.c_ubyte * size)()
    got = SIZE_T()
    if not k32.ReadProcessMemory(handle, address, output, size, ctypes.byref(got)):
        fail(
            "BLOCKED_MEMORY_UNREADABLE",
            f"ReadProcessMemory {hexptr(address)}+{size} failed winerr={ctypes.get_last_error()}",
        )
    if got.value != size:
        fail("BLOCKED_SHORT_READ", f"RPM {hexptr(address)}+{size} returned {got.value}")
    return bytes(output)


def memory_info(handle: HANDLE, address: int) -> dict[str, int] | None:
    raw = (ctypes.c_ubyte * 48)()
    if not k32.VirtualQueryEx(handle, address, raw, len(raw)):
        return None
    base, alloc = struct.unpack_from("<QQ", raw, 0)
    region_size = struct.unpack_from("<Q", raw, 24)[0]
    state, protect, mem_type = struct.unpack_from("<III", raw, 32)
    return {
        "base": base,
        "allocation": alloc,
        "size": region_size,
        "state": state,
        "protect": protect,
        "type": mem_type,
    }


def readable(handle: HANDLE, address: int, size: int) -> bool:
    end = address + size
    cursor = address
    while cursor < end:
        info = memory_info(handle, cursor)
        if not info or info["state"] != MEM_COMMIT:
            return False
        protect = info["protect"]
        if protect & (PAGE_GUARD | PAGE_NOACCESS) or (protect & 0xFF) not in {
            0x02, 0x04, 0x08, 0x20, 0x40, 0x80
        }:
            return False
        cursor = min(end, info["base"] + info["size"])
    return True


def list_modules(handle: HANDLE, exe_path: str | None) -> list[dict[str, Any]]:
    modules = (HMODULE * 4096)()
    needed = DWORD()
    if not psapi.EnumProcessModulesEx(
        handle, modules, ctypes.sizeof(modules), ctypes.byref(needed), LIST_MODULES_ALL
    ):
        fail("BLOCKED_MODULE_ENUM", f"EnumProcessModulesEx failed winerr={ctypes.get_last_error()}")
    result = []
    for module in modules[: needed.value // ctypes.sizeof(HMODULE)]:
        info = MODULEINFO()
        if not psapi.GetModuleInformation(handle, module, ctypes.byref(info), ctypes.sizeof(info)):
            continue
        name_buf = ctypes.create_unicode_buffer(32768)
        psapi.GetModuleFileNameExW(handle, module, name_buf, len(name_buf))
        path = name_buf.value or (exe_path if info.lpBaseOfDll else None)
        result.append(
            {
                "name": Path(path).name if path else "",
                "path": path,
                "base": int(info.lpBaseOfDll or 0),
                "size": int(info.SizeOfImage),
                "entry": int(info.EntryPoint or 0),
            }
        )
    return result


def parse_pe(headers: bytes) -> dict[str, Any]:
    if len(headers) < 0x200 or headers[:2] != b"MZ":
        fail("BLOCKED_PE_INVALID", "missing MZ header")
    pe = struct.unpack_from("<I", headers, 0x3C)[0]
    if pe + 24 > len(headers) or headers[pe : pe + 4] != b"PE\0\0":
        fail("BLOCKED_PE_INVALID", "missing PE signature")
    machine, count = struct.unpack_from("<HH", headers, pe + 4)
    optional_size = struct.unpack_from("<H", headers, pe + 20)[0]
    optional = pe + 24
    if machine != 0x8664 or optional + 112 > len(headers) or struct.unpack_from("<H", headers, optional)[0] != 0x20B:
        fail("BLOCKED_PE_INVALID", "module is not PE32+ AMD64")
    image_size = struct.unpack_from("<I", headers, optional + 56)[0]
    size_headers = struct.unpack_from("<I", headers, optional + 60)[0]
    sections = []
    table = optional + optional_size
    for index in range(count):
        offset = table + index * 40
        if offset + 40 > len(headers):
            fail("BLOCKED_PE_INVALID", "section table outside readable headers")
        virtual_size, virtual_address, raw_size = struct.unpack_from("<III", headers, offset + 8)
        characteristics = struct.unpack_from("<I", headers, offset + 36)[0]
        sections.append(
            {
                "name": headers[offset : offset + 8].rstrip(b"\0").decode("ascii", "replace"),
                "rva": virtual_address,
                "span": virtual_size or raw_size,
                "executable": bool(characteristics & IMAGE_SCN_MEM_EXECUTE),
            }
        )
    return {"image_size": image_size, "size_headers": size_headers, "sections": sections}


def rva_section(pe: dict[str, Any], rva: int) -> dict[str, Any] | None:
    for section in pe["sections"]:
        if section["rva"] <= rva < section["rva"] + section["span"]:
            return section
    return None


def find_module(handle: HANDLE, pid: int, name_pattern: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = process_path(handle)
    modules = list_modules(handle, path)
    matches = [m for m in modules if name_pattern.casefold() in m["name"].casefold()]
    if not matches:
        fail("BLOCKED_C1_MODULE_NOT_FOUND", f"no module matched {name_pattern!r}")
    if len(matches) != 1:
        fail("BLOCKED_C1_MODULE_AMBIGUOUS", json.dumps([m["name"] for m in matches]))
    module = matches[0]
    headers = read_remote(handle, module["base"], 0x1000)
    pe = parse_pe(headers)
    if pe["image_size"] != module["size"]:
        fail("BLOCKED_PE_INVALID", "SizeOfImage differs from module size")
    for label, rva in (("producer", PRODUCER_RVA), ("consumer", CONSUMER_RVA)):
        section = rva_section(pe, rva)
        if not section or not section["executable"]:
            fail("BLOCKED_RVA_NOT_EXECUTABLE", f"{label} RVA {hexptr(rva)} is not in an executable section")
        read_remote(handle, module["base"] + rva, 24)
    disk_hash = None
    if module["path"] and Path(module["path"]).is_file():
        digest = hashlib.sha256()
        with open(module["path"], "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        disk_hash = digest.hexdigest()
    return module, {"pe": pe, "disk_sha256": disk_hash, "process_path": path}


def candidate(
    kind: str, location: int, data: int, stride: int, confidence: int
) -> dict[str, Any]:
    return {
        "kind": kind,
        "structure": hexptr(location),
        "data": hexptr(data),
        "stride": stride,
        "confidence": confidence,
        "_data": data,
    }


def parse_mat(handle: HANDLE, blob: bytes, offset: int, location: int, request: Request) -> dict[str, Any] | None:
    if offset < 0 or offset + 96 > len(blob):
        return None
    flags, dims, rows, cols = struct.unpack_from("<iiii", blob, offset)
    if dims != 2 or rows != request.height or cols != request.width:
        return None
    channels = ((flags >> 3) & 63) + 1
    depth = flags & 7
    expected_depth = 0 if request.dtype == "u8" else 5
    if channels != request.channels or depth != expected_depth:
        return None
    data = struct.unpack_from("<Q", blob, offset + 16)[0]
    stride = struct.unpack_from("<Q", blob, offset + 80)[0]
    if not stride:
        step_ptr = struct.unpack_from("<Q", blob, offset + 72)[0]
        if 0x10000 <= step_ptr <= USER_MAX and readable(handle, step_ptr, 8):
            stride = struct.unpack("<Q", read_remote(handle, step_ptr, 8))[0]
    if stride < request.width * request.element_size or stride > 1 << 30:
        return None
    if request.expected_stride is not None and stride != request.expected_stride:
        return None
    return candidate("cv-mat-header", location, data, stride, 90)


def parse_draw(blob: bytes, pos: int, request: Request) -> list[dict[str, Any]]:
    row_bytes = request.width * request.element_size
    found = []
    # Explicit layouts reduce false positives. Each tuple is (data delta, stride delta, size).
    forms = (
        (-8, 8, 4),
        (-8, 8, 8),
        (8, 16, 8),
        (16, 8, 4),
        (16, 8, 8),
    )
    for data_delta, stride_delta, stride_size in forms:
        data_at = pos + data_delta
        stride_at = pos + stride_delta
        if data_at < 0 or stride_at < 0 or stride_at + stride_size > len(blob):
            continue
        data = struct.unpack_from("<Q", blob, data_at)[0]
        stride = struct.unpack_from("<I" if stride_size == 4 else "<Q", blob, stride_at)[0]
        if stride < row_bytes or stride > 1 << 30:
            continue
        if request.expected_stride is not None and stride != request.expected_stride:
            continue
        found.append(candidate("draw-buffer-struct", pos, data, stride, 55))
    return found


def scan_candidates(handle: HANDLE, request: Request, max_scan_bytes: int) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    scanned = 0
    regions = 0
    cursor = 0x10000
    needles = {struct.pack("<II", request.width, request.height), struct.pack("<II", request.height, request.width)}
    while cursor <= USER_MAX and scanned < max_scan_bytes:
        info = memory_info(handle, cursor)
        if not info:
            break
        if info["size"] == 0:
            break
        next_cursor = info["base"] + info["size"]
        protect = info["protect"]
        if (
            info["state"] == MEM_COMMIT
            and info["type"] == MEM_PRIVATE
            and not protect & (PAGE_GUARD | PAGE_NOACCESS)
            and (protect & 0xFF) in READABLE_WRITABLE
        ):
            regions += 1
            offset = 0
            while offset < info["size"] and scanned < max_scan_bytes:
                chunk_size = min(4 << 20, info["size"] - offset, max_scan_bytes - scanned)
                if chunk_size < 128:
                    break
                chunk_addr = info["base"] + offset
                try:
                    blob = read_remote(handle, chunk_addr, chunk_size)
                except CaptureError:
                    break
                seen: set[int] = set()
                for needle in needles:
                    position = blob.find(needle)
                    while position >= 0:
                        if (position, needle[0]) in seen:
                            position = blob.find(needle, position + 1)
                            continue
                        seen.add((position, needle[0]))
                        mat = parse_mat(
                            handle,
                            blob,
                            position - 8,
                            chunk_addr + position - 8,
                            request,
                        )
                        if mat:
                            candidates.append(mat)
                        for draw in parse_draw(blob, position, request):
                            draw["structure"] = hexptr(chunk_addr + position)
                            candidates.append(draw)
                        position = blob.find(needle, position + 1)
                scanned += chunk_size
                if chunk_size < 4 << 20:
                    break
                offset += chunk_size - 128
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    row_bytes = request.width * request.element_size
    for item in candidates:
        span = (request.height - 1) * item["stride"] + row_bytes
        item["span_readable"] = readable(handle, item["_data"], span)
    valid = [item for item in candidates if item["span_readable"] and item["_data"]]
    unique: dict[tuple[int, int], dict[str, Any]] = {}
    for item in valid:
        unique.setdefault((item["_data"], item["stride"]), item)
    return {
        "bytes_scanned": scanned,
        "regions_scanned": regions,
        "candidate_count": len(candidates),
        "valid_unique_count": len(unique),
        "candidates": list(unique.values())[:32],
        "limit_reached": scanned >= max_scan_bytes,
    }


def capture(handle: HANDLE, item: dict[str, Any], request: Request) -> dict[str, Any]:
    x, y, width, height = request.roi
    row_bytes = width * request.element_size
    total = row_bytes * height
    if total > request.max_capture_bytes:
        fail("BLOCKED_CAPTURE_LIMIT", f"ROI needs {total} bytes > {request.max_capture_bytes}")
    digest = hashlib.sha256()
    spans = []
    for row in range(height):
        address = item["_data"] + (y + row) * item["stride"] + x * request.element_size
        if not readable(handle, address, row_bytes):
            fail("BLOCKED_ROI_UNREADABLE", f"ROI row {row} at {hexptr(address)} is not fully readable")
        first = read_remote(handle, address, row_bytes)
        for _ in range(request.stable_reads - 1):
            if read_remote(handle, address, row_bytes) != first:
                fail("BLOCKED_UNSTABLE_READ", f"ROI row {row} changed between reads")
        digest.update(first)
        spans.append({"address": hexptr(address), "bytes": row_bytes})
    return {
        "sha256": digest.hexdigest(),
        "canonical_bytes": total,
        "row_count": height,
        "row_bytes": row_bytes,
        "stride": item["stride"],
        "roi": list(request.roi),
        "row_spans": spans[:8],
        "row_spans_truncated": len(spans) > 8,
    }


def redact(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in item.items() if key != "_data"} for item in items]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_roi(value: str) -> tuple[int, int, int, int]:
    try:
        roi = tuple(int(part, 0) for part in value.split(","))
    except ValueError:
        fail("INVALID_ROI", "--roi must be x,y,width,height")
    if len(roi) != 4 or roi[2] <= 0 or roi[3] <= 0:
        fail("INVALID_ROI", "--roi must be x,y,width,height")
    return roi  # type: ignore[return-value]


def run_capture(args: argparse.Namespace) -> dict[str, Any]:
    request = Request(
        args.width,
        args.height,
        parse_roi(args.roi),
        args.dtype,
        args.channels,
        args.expected_stride,
        args.max_capture_bytes,
        args.stable_reads,
    )
    if request.roi[0] + request.roi[2] > request.width or request.roi[1] + request.roi[3] > request.height:
        fail("INVALID_ROI", "ROI leaves the declared frame")
    processes = list_c1_processes()
    if args.pid is not None:
        processes = [item for item in processes if item["pid"] == args.pid]
    if not processes:
        fail("BLOCKED_NO_CAPTURE_ONE_PROCESS", "no running CaptureOne.exe/portrait.exe target")
    if len(processes) != 1:
        fail("BLOCKED_CAPTURE_ONE_PROCESS_AMBIGUOUS", json.dumps([item["pid"] for item in processes]))
    target = processes[0]
    handle = open_readonly(target["pid"])
    try:
        module, evidence = find_module(handle, target["pid"], args.module)
        expected = None if args.allow_unpinned_pe else args.expected_pe_sha256.lower()
        if expected and evidence["disk_sha256"] != expected:
            fail(
                "BLOCKED_FIXED_PE_SHA_MISMATCH",
                f"module SHA {evidence['disk_sha256']} != {expected}; RVAs are not authoritative",
            )
        scan = scan_candidates(handle, request, args.max_scan_bytes)
        if scan["valid_unique_count"] == 0:
            code = "BLOCKED_SCAN_LIMIT_NO_VALID_POINTER_CANDIDATE" if scan["limit_reached"] else "BLOCKED_NO_VALID_POINTER_CANDIDATE"
            fail(code, "no readable, shape/stride-consistent Mat or draw-buffer pointer structure")
        captures = [capture(handle, item, request) for item in scan["candidates"]]
        hashes = sorted({item["sha256"] for item in captures})
        if len(hashes) != 1:
            fail("BLOCKED_AMBIGUOUS_POINTER_CANDIDATES", f"valid candidates produced {len(hashes)} distinct ROI hashes")
        result = {
            "schema": SCHEMA,
            "status": "CAPTURED_READ_ONLY_UNVERIFIED",
            "blocker": None,
            "authorization": {
                "access": "PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ",
                "target_writes": False,
                "remote_threads": False,
                "debugger_attach": False,
                "process_suspend": False,
            },
            "target": target,
            "module": {
                "name": module["name"],
                "path": module["path"],
                "base": hexptr(module["base"]),
                "size": module["size"],
                "disk_sha256": evidence["disk_sha256"],
                "pe_sha_pin": expected,
                "producer_rva": hexptr(PRODUCER_RVA),
                "consumer_rva": hexptr(CONSUMER_RVA),
            },
            "request": {
                "width": request.width,
                "height": request.height,
                "roi": list(request.roi),
                "dtype": request.dtype,
                "channels": request.channels,
                "expected_stride": request.expected_stride,
                "stable_reads": request.stable_reads,
            },
            "scan": {**scan, "candidates": redact(scan["candidates"])},
            "captures": captures,
            "executable_pf_delta": 0,
            "semantic_receipt": "NOT_PROVEN: same-job typecode, output4/local-Mat identity and draw warp are absent",
        }
        return result
    finally:
        k32.CloseHandle(handle)


def blocked_manifest(error: CaptureError) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": error.code,
        "blocker": {"code": error.code, "detail": error.detail},
        "authorization": {
            "access": "PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ",
            "target_writes": False,
            "remote_threads": False,
            "debugger_attach": False,
            "process_suspend": False,
        },
        "executable_pf_delta": 0,
        "capture_performed": False,
    }


def selftest() -> None:
    pixels = bytearray(b"ABCDEF..........GHIJKL..........")
    pixel_buffer = (ctypes.c_ubyte * len(pixels)).from_buffer(pixels)
    data_address = ctypes.addressof(pixel_buffer)
    header = (ctypes.c_ubyte * 96)()
    header_address = ctypes.addressof(header)
    struct.pack_into("<iiii", header, 0, 16, 2, 2, 3)
    struct.pack_into("<Q", header, 16, data_address)
    struct.pack_into("<Q", header, 32, data_address)
    struct.pack_into("<Q", header, 40, data_address + len(pixels))
    struct.pack_into("<Q", header, 48, data_address + len(pixels))
    struct.pack_into("<Q", header, 72, header_address + 80)
    struct.pack_into("<Q", header, 80, 16)
    handle = open_readonly(os.getpid())
    request = Request(3, 2, (1, 0, 2, 2), "u8", 3, 16, 1024, 2)
    try:
        item = parse_mat(handle, bytes(header), 0, header_address, request)
        if not item:
            fail("SELFTEST_FAIL", "synthetic cv::Mat header was rejected")
        receipt = capture(handle, item, request)
        expected = sha256(pixels[3:9] + pixels[19:25])
        if receipt["sha256"] != expected:
            fail("SELFTEST_FAIL", "ROI SHA differs from canonical expected bytes")
        pe = parse_pe(Path(sys.executable).read_bytes()[:0x1000])
        if pe["image_size"] <= 0 or not pe["sections"]:
            fail("SELFTEST_FAIL", "local PE parser returned empty sections")
    finally:
        k32.CloseHandle(handle)
    print("SELFTEST=PASS READ_PROCESS_MEMORY=READ_ONLY MAT_RVA_PE=OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("selftest")
    cap = sub.add_parser("capture")
    cap.add_argument("--pid", type=int)
    cap.add_argument("--width", type=int, required=True)
    cap.add_argument("--height", type=int, required=True)
    cap.add_argument("--roi", required=True, help="x,y,width,height")
    cap.add_argument("--dtype", choices=("u8", "f32le"), default="u8")
    cap.add_argument("--channels", type=int, choices=range(1, 5), default=3)
    cap.add_argument("--expected-stride", type=int)
    cap.add_argument("--module", default="portrait.exe")
    cap.add_argument("--expected-pe-sha256", default=FIXED_PE_SHA256)
    cap.add_argument("--allow-unpinned-pe", action="store_true")
    cap.add_argument("--max-scan-bytes", type=int, default=256 << 20)
    cap.add_argument("--max-capture-bytes", type=int, default=64 << 20)
    cap.add_argument("--stable-reads", type=int, choices=(1, 2), default=2)
    cap.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "selftest":
        selftest()
        return 0
    if args.command == "doctor":
        processes = list_c1_processes()
        print(
            json.dumps(
                {
                    "status": "READY_READ_ONLY",
                    "readprocessmemory": True,
                    "target_writes": False,
                    "c1_processes": processes,
                },
                indent=2,
            )
        )
        return 0
    try:
        result = run_capture(args)
        write_json(args.out, result)
        print(json.dumps({"status": result["status"], "out": str(args.out)}))
        return 0
    except CaptureError as error:
        write_json(args.out, blocked_manifest(error))
        print(json.dumps({"status": error.code, "detail": error.detail, "out": str(args.out)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
