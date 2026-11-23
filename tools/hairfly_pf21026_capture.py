#!/usr/bin/env python3
"""Read-only verifier and ingress for a same-job HairFly/PF21026 capture.

The tool never attaches to, injects into, or modifies Capture One. A debugger
or vendor export must produce the raw files described by the manifest first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any


SCHEMA = "aurora.hairfly-pf21026.capture.v1"
FIXED_PE_SHA256 = "a84519da489bc52e4e914af7880ede43e3ce58992657723d02197161832464c7"
PRODUCER_RVA = 0x823A3B0
CONSUMER_RVA = 0x82AA2E0
OUTPUT3_SLOT = 0x350
OUTPUT4_SLOT = 0x4B0
NATIVE_MAT_OFFSET = 0x4A8
MASK_B_TENSOR = "2405607714"
CHANNELS = {"ch0": "Ch0", "ch1": "Ch1", "ch2": "Ch2"}


class CaptureError(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def fail(code: str, detail: str) -> None:
    raise CaptureError(code, detail)


def require_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        fail("INVALID_MANIFEST", f"{name} must be an integer >= {minimum}")
    return value


def require_hex32(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        fail("INVALID_MANIFEST", f"{name} must be 64 lowercase hex characters")
    try:
        bytes.fromhex(value)
    except ValueError:
        fail("INVALID_MANIFEST", f"{name} must be hexadecimal")
    if value.lower() != value:
        fail("INVALID_MANIFEST", f"{name} must use lowercase hex")
    return value


def require_roi(value: Any, width: int, height: int, name: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        fail("INVALID_ROI", f"{name} must be [x,y,width,height]")
    x, y, w, h = (require_int(item, f"{name}[{i}]") for i, item in enumerate(value))
    if w == 0 or h == 0 or x + w > width or y + h > height:
        fail("INVALID_ROI", f"{name} is empty or outside the frame")
    return [x, y, w, h]


def artifact_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        fail("INVALID_MANIFEST", f"{name} must be a relative artifact path")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        fail("INVALID_MANIFEST", f"{name} escapes the capture bundle")
    if not path.is_file():
        fail("MISSING_ARTIFACT", f"{name} does not exist: {value}")
    return path


def read_artifact(root: Path, spec: Any, name: str) -> tuple[Path, bytes]:
    if not isinstance(spec, dict):
        fail("INVALID_MANIFEST", f"{name} must be an object")
    path = artifact_path(root, spec.get("file"), f"{name}.file")
    return path, path.read_bytes()


def rows_canonical(
    data: bytes,
    rows: int,
    row_bytes: int,
    row_stride: int,
    channel_stride: int | None = None,
) -> bytes:
    if row_bytes <= 0 or row_stride < row_bytes:
        fail("INVALID_STRIDE", "row_stride_bytes is smaller than one canonical row")
    stride = channel_stride if channel_stride is not None else row_stride * rows
    if stride < row_stride * rows:
        fail("INVALID_STRIDE", "channel_stride_bytes is smaller than the strided rows")
    if len(data) < stride:
        fail("SHORT_ARTIFACT", f"artifact needs {stride} bytes, has {len(data)}")
    out = bytearray(row_bytes * rows)
    for row in range(rows):
        start = row * row_stride
        out[row * row_bytes : (row + 1) * row_bytes] = data[start : start + row_bytes]
    return bytes(out)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def f32_le_values(data: bytes, name: str) -> list[float]:
    if len(data) % 4:
        fail("INVALID_ARTIFACT", f"{name} is not a whole number of f32 values")
    values = list(struct.unpack(f"<{len(data) // 4}f", data))
    if any(value != value or abs(value) == float("inf") for value in values):
        fail("NON_FINITE_MASK", f"{name} contains NaN or infinity")
    return values


def verify(bundle: Path, output: Path | None) -> dict[str, Any]:
    manifest_path = bundle / "capture.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail("MISSING_MANIFEST", str(manifest_path))
    except json.JSONDecodeError as error:
        fail("INVALID_MANIFEST", str(error))
    if manifest.get("schema") != SCHEMA:
        fail("INVALID_MANIFEST", f"schema must be {SCHEMA}")

    same_job = manifest.get("same_job")
    if not isinstance(same_job, dict):
        fail("INVALID_MANIFEST", "same_job must be an object")
    job_id = require_int(same_job.get("job_id"), "same_job.job_id", 1)
    face_index = require_int(same_job.get("face_index"), "same_job.face_index")
    producer_job = require_int(same_job.get("producer_job_id"), "same_job.producer_job_id", 1)
    consumer_job = require_int(same_job.get("consumer_job_id"), "same_job.consumer_job_id", 1)
    if job_id != producer_job or job_id != consumer_job:
        fail("JOB_MISMATCH", "job/face capture is not from one same-job frame")

    capture = manifest.get("capture")
    if not isinstance(capture, dict):
        fail("INVALID_MANIFEST", "capture must be an object")
    source_sha256 = require_hex32(capture.get("source_sha256"), "capture.source_sha256")
    model_hash_id = require_int(capture.get("model_hash_id"), "capture.model_hash_id", 1)
    model_width = require_int(capture.get("model_width"), "capture.model_width", 1)
    model_height = require_int(capture.get("model_height"), "capture.model_height", 1)
    frame_width = require_int(capture.get("frame_width"), "capture.frame_width", 1)
    frame_height = require_int(capture.get("frame_height"), "capture.frame_height", 1)
    channel_key = capture.get("mask_b_channel")
    if channel_key not in CHANNELS:
        fail("MISSING_CHANNEL_CHOICE", "capture.mask_b_channel must be ch0, ch1, or ch2")
    channel = CHANNELS[channel_key]
    repair_output = capture.get("repair_output")
    if repair_output not in {"output3", "output4"}:
        fail("INVALID_MANIFEST", "capture.repair_output must be output3 or output4")

    draw = manifest.get("draw_buffer")
    if not isinstance(draw, dict):
        fail("PF90088_ROI_WARP_DRAW_BUFFER", "draw_buffer is absent")
    model_roi = require_roi(draw.get("model_roi"), model_width, model_height, "draw_buffer.model_roi")
    frame_roi = require_roi(draw.get("frame_roi"), frame_width, frame_height, "draw_buffer.frame_roi")
    warp = draw.get("warp")
    if not isinstance(warp, list) or len(warp) != 2:
        fail("PF90088_ROI_WARP_MATRIX", "draw_buffer.warp must be 2x3")
    clean_warp: list[list[float]] = []
    for row, values in enumerate(warp):
        if not isinstance(values, list) or len(values) != 3:
            fail("PF90088_ROI_WARP_MATRIX", f"draw_buffer.warp[{row}] must have 3 values")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            fail("PF90088_ROI_WARP_MATRIX", "warp values must be numeric")
        converted = [float(v) for v in values]
        if any(v != v or abs(v) == float("inf") for v in converted):
            fail("PF90088_ROI_WARP_MATRIX", "warp contains NaN or infinity")
        clean_warp.append(converted)
    vertex_count = require_int(draw.get("vertex_count"), "draw_buffer.vertex_count", 1)
    vertex_path, vertex_data = read_artifact(bundle, draw.get("vertex_buffer"), "draw_buffer.vertex_buffer")
    if not vertex_data:
        fail("PF90088_ROI_WARP_DRAW_BUFFER", "vertex buffer is empty")

    mask_spec = manifest.get("mask_b")
    if not isinstance(mask_spec, dict):
        fail("PF21026_MASK_B_CHANNEL_DECODE", "mask_b is absent")
    if mask_spec.get("layout") != "chw" or mask_spec.get("dtype") != "f32le":
        fail("PF21026_MASK_B_CHANNEL_DECODE", "mask_b must be f32le CHW")
    if mask_spec.get("shape") != [3, model_height, model_width]:
        fail("PF21026_MASK_B_CHANNEL_DECODE", "mask_b shape does not match the registered model")
    mask_path, mask_data = read_artifact(bundle, mask_spec, "mask_b")
    mask_row_stride = require_int(mask_spec.get("row_stride_bytes"), "mask_b.row_stride_bytes")
    mask_channel_stride = require_int(
        mask_spec.get("channel_stride_bytes"), "mask_b.channel_stride_bytes"
    )
    channel_index = int(channel_key[-1])
    plane_bytes = rows_canonical(
        mask_data[channel_index * mask_channel_stride :],
        model_height,
        model_width * 4,
        mask_row_stride,
        mask_channel_stride,
    )
    f32_le_values(plane_bytes, f"mask_b.{channel_key}")
    planes = []
    for index in range(3):
        plane = rows_canonical(
            mask_data[index * mask_channel_stride :],
            model_height,
            model_width * 4,
            mask_row_stride,
            mask_channel_stride,
        )
        f32_le_values(plane, f"mask_b.ch{index}")
        planes.append(plane)
    decoded_mask = b"".join(planes)

    repair_spec = manifest.get("repair_output")
    if not isinstance(repair_spec, dict):
        fail("PF21026_REPAIR_OUTPUT_MISSING", "repair_output is absent")
    if repair_spec.get("layout") != "hwc" or repair_spec.get("dtype") != "u8":
        fail("INVALID_ARTIFACT", "repair_output must be u8 HWC")
    if repair_spec.get("shape") != [frame_height, frame_width, 3]:
        fail("SHAPE_MISMATCH", "repair_output shape does not match the frame")
    repair_path, repair_data = read_artifact(bundle, repair_spec, "repair_output")
    repair_row_stride = require_int(
        repair_spec.get("row_stride_bytes"), "repair_output.row_stride_bytes"
    )
    repair_rgb = rows_canonical(
        repair_data, frame_height, frame_width * 3, repair_row_stride
    )

    mat_spec = manifest.get("local_mat")
    if not isinstance(mat_spec, dict):
        fail("PF21026_REPAIR_NATIVE_MAT_CONSUMER", "local_mat is absent")
    if mat_spec.get("layout") != "hwc" or mat_spec.get("dtype") != "u8":
        fail("INVALID_ARTIFACT", "local_mat must be u8 HWC")
    if mat_spec.get("shape") != [frame_height, frame_width, 3]:
        fail("SHAPE_MISMATCH", "local_mat shape does not match the frame")
    mat_path, mat_data = read_artifact(bundle, mat_spec, "local_mat")
    mat_row_stride = require_int(mat_spec.get("row_stride_bytes"), "local_mat.row_stride_bytes")
    mat_rgb = rows_canonical(mat_data, frame_height, frame_width * 3, mat_row_stride)
    if mat_rgb != repair_rgb:
        fail(
            "PF21026_REPAIR_NATIVE_MAT_RECEIPT",
            "local Mat pixels differ from the selected model repair output",
        )

    evidence = manifest.get("edge_receipt")
    if not isinstance(evidence, dict):
        fail("PF21026_REPAIR_NATIVE_MAT_EDGE", "edge_receipt is absent")
    if require_int(evidence.get("producer_rva"), "edge_receipt.producer_rva") != PRODUCER_RVA:
        fail("PF21026_REPAIR_WRITER_RECEIPT", "producer RVA mismatch")
    if require_int(evidence.get("consumer_rva"), "edge_receipt.consumer_rva") != CONSUMER_RVA:
        fail("PF21026_REPAIR_NATIVE_MAT_EDGE", "consumer RVA mismatch")
    if require_int(
        evidence.get("native_mat_offset"), "edge_receipt.native_mat_offset"
    ) != NATIVE_MAT_OFFSET:
        fail("PF21026_REPAIR_NATIVE_MAT_EDGE", "native Mat offset mismatch")
    expected_slot = OUTPUT4_SLOT if repair_output == "output4" else OUTPUT3_SLOT
    if require_int(
        evidence.get("producer_lookup_slot"), "edge_receipt.producer_lookup_slot"
    ) != expected_slot:
        fail("PF21026_REPAIR_LOOKUP_SLOT_RECEIPT", "producer lookup slot mismatch")
    if repair_output == "output3":
        fail(
            "PF21026_REPAIR_OUTPUT3_NATIVE_U8_CONSUMER",
            "G6 found no native-u8 conversion for output3",
        )

    raw_mask_spec = manifest.get("raw_mask")
    typed_blocker = None
    raw_mask_sha256 = None
    if raw_mask_spec is None:
        typed_blocker = {
            "code": "PF90088_RAW_MASK_POSTPROCESS_RECEIPT",
            "detail": "native border-feather/min/ROI-warp/u8 raw mask was not captured",
        }
    else:
        if not isinstance(raw_mask_spec, dict):
            fail("INVALID_MANIFEST", "raw_mask must be an object")
        if raw_mask_spec.get("layout") != "hw" or raw_mask_spec.get("dtype") != "u8":
            fail("INVALID_ARTIFACT", "raw_mask must be u8 HW")
        if raw_mask_spec.get("shape") != [frame_height, frame_width]:
            fail("SHAPE_MISMATCH", "raw_mask shape does not match the frame")
        _, raw_data = read_artifact(bundle, raw_mask_spec, "raw_mask")
        raw_stride = require_int(raw_mask_spec.get("row_stride_bytes"), "raw_mask.row_stride_bytes")
        raw_mask = rows_canonical(raw_data, frame_height, frame_width, raw_stride)
        raw_mask_sha256 = sha256(raw_mask)

    native_mat_sha256 = sha256(mat_rgb)
    result: dict[str, Any] = {
        "schema": "aurora.hairfly-pf21026.receipts.v1",
        "status": "CAPTURE_VERIFIED_TYPED_READY" if typed_blocker is None else "CAPTURE_VERIFIED_TYPED_BLOCKED",
        "source": {
            "capture_manifest_sha256": sha256(manifest_path.read_bytes()),
            "source_sha256": source_sha256,
            "files": {
                "mask_b": str(mask_path.relative_to(bundle)),
                "repair_output": str(repair_path.relative_to(bundle)),
                "local_mat": str(mat_path.relative_to(bundle)),
                "vertex_buffer": str(vertex_path.relative_to(bundle)),
            },
        },
        "same_job": {"job_id": job_id, "face_index": face_index},
        "hairfly": {
            "tensor": MASK_B_TENSOR,
            "shape": [1, 3, model_height, model_width],
            "channel": channel,
            "mask_b_plane_sha256": sha256(plane_bytes),
            "mask_b_decoded_sha256": sha256(decoded_mask),
            "raw_mask_sha256": raw_mask_sha256,
            "draw_buffer": {
                "model_roi": model_roi,
                "frame_roi": frame_roi,
                "warp": clean_warp,
                "vertex_count": vertex_count,
                "vertex_buffer_sha256": sha256(vertex_data),
                "vertex_stride_bytes": require_int(
                    draw.get("vertex_stride_bytes"), "draw_buffer.vertex_stride_bytes"
                ),
                "roi_stride_bytes": require_int(
                    draw.get("roi_stride_bytes"), "draw_buffer.roi_stride_bytes"
                ),
            },
            "strides": {
                "mask_row_stride_bytes": mask_row_stride,
                "mask_channel_stride_bytes": mask_channel_stride,
            },
        },
        "pf21026": {
            "tensor": "2892147817",
            "repair_output": "Output4",
            "native_mat_sha256": native_mat_sha256,
            "producer_rva": f"0x{PRODUCER_RVA:X}",
            "producer_lookup_slot": f"0x{OUTPUT4_SLOT:X}",
            "consumer_rva": f"0x{CONSUMER_RVA:X}",
            "native_mat_offset": f"0x{NATIVE_MAT_OFFSET:X}",
            "repair_source_sha256": sha256(repair_rgb),
            "strides": {
                "repair_row_stride_bytes": repair_row_stride,
                "native_mat_row_stride_bytes": mat_row_stride,
            },
        },
        "typed_blocker": typed_blocker,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def debugger_doctor(binary: Path | None, cdb: Path | None) -> dict[str, Any]:
    known = [
        cdb,
        Path(os.environ.get("ProgramFiles", "")) / "Windows Kits/10/Debuggers/x64/cdb.exe",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Windows Kits/10/Debuggers/x64/cdb.exe",
    ]
    found = next((path for path in known if path and path.is_file()), None)
    result: dict[str, Any] = {
        "cdb": str(found) if found else None,
        "debugger_channel": "AVAILABLE" if found else "ABSENT",
    }
    if binary is not None:
        if not binary.is_file():
            fail("MISSING_BINARY", str(binary))
        digest = sha256(binary.read_bytes())
        result["binary_sha256"] = digest
        result["binary_matches_g6_pe"] = digest == FIXED_PE_SHA256
    requirements = {
        "module_name": None,
        "producer_post_prologue_pc_rva": None,
        "consumer_local_mat_pc_rva": None,
        "job_id_storage": None,
        "draw_buffer_storage": None,
        "mask_b_data_pointer_and_shape": None,
        "native_mat_data_pointer_and_shape": None,
    }
    result["unresolved_capture_inputs"] = requirements
    result["blocker"] = (
        "NO_READ_ONLY_CAPTURE_CHANNEL: no cdb/windbg transport and no exact post-prologue "
        "PCs or runtime locations for job, draw buffer, mask tensor, and Mat header"
    )
    return result


def make_selftest_bundle(root: Path) -> Path:
    model_w, model_h, frame_w, frame_h = 7, 5, 9, 6
    mask_rows = []
    for channel in range(3):
        data = bytearray()
        for y in range(model_h):
            for x in range(model_w):
                data += struct.pack("<f", (channel + 1) * 0.1 + (x + y) / 1000)
        mask_rows.append(bytes(data))
    mask_stride = model_w * 4
    (root / "mask.f32le").write_bytes(b"".join(mask_rows))
    repair = bytearray()
    for y in range(frame_h):
        for x in range(frame_w):
            repair += bytes((x, y, (x + y) & 255))
    (root / "repair.u8").write_bytes(repair)
    (root / "mat.u8").write_bytes(repair)
    (root / "raw.u8").write_bytes(bytes(range(frame_w * frame_h)))
    (root / "vertices.bin").write_bytes(struct.pack("<4f", 0, 0, 1, 1))
    manifest = {
        "schema": SCHEMA,
        "same_job": {
            "job_id": 7,
            "face_index": 0,
            "producer_job_id": 7,
            "consumer_job_id": 7,
        },
        "capture": {
            "source_sha256": "00" * 32,
            "model_hash_id": 3378116080,
            "model_width": model_w,
            "model_height": model_h,
            "frame_width": frame_w,
            "frame_height": frame_h,
            "mask_b_channel": "ch1",
            "repair_output": "output4",
        },
        "draw_buffer": {
            "model_roi": [0, 0, model_w, model_h],
            "frame_roi": [1, 1, model_w, model_h],
            "warp": [[1, 0, 1], [0, 1, 1]],
            "vertex_count": 4,
            "vertex_stride_bytes": 16,
            "roi_stride_bytes": frame_w * 3,
            "vertex_buffer": {"file": "vertices.bin"},
        },
        "mask_b": {
            "file": "mask.f32le",
            "dtype": "f32le",
            "layout": "chw",
            "shape": [3, model_h, model_w],
            "row_stride_bytes": mask_stride,
            "channel_stride_bytes": mask_stride * model_h,
        },
        "repair_output": {
            "file": "repair.u8",
            "dtype": "u8",
            "layout": "hwc",
            "shape": [frame_h, frame_w, 3],
            "row_stride_bytes": frame_w * 3,
        },
        "local_mat": {
            "file": "mat.u8",
            "dtype": "u8",
            "layout": "hwc",
            "shape": [frame_h, frame_w, 3],
            "row_stride_bytes": frame_w * 3,
        },
        "edge_receipt": {
            "producer_rva": PRODUCER_RVA,
            "producer_lookup_slot": OUTPUT4_SLOT,
            "consumer_rva": CONSUMER_RVA,
            "native_mat_offset": NATIVE_MAT_OFFSET,
        },
        "raw_mask": {
            "file": "raw.u8",
            "dtype": "u8",
            "layout": "hw",
            "shape": [frame_h, frame_w],
            "row_stride_bytes": frame_w,
        },
    }
    (root / "capture.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def selftest() -> None:
    with tempfile.TemporaryDirectory(prefix="hairfly-capture-") as temp:
        root = Path(temp)
        make_selftest_bundle(root)
        result = verify(root, None)
        assert result["status"] == "CAPTURE_VERIFIED_TYPED_READY"
        assert result["hairfly"]["channel"] == "Ch1"
        assert result["pf21026"]["native_mat_sha256"] == result["pf21026"]["repair_source_sha256"]
        manifest_path = root / "capture.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["raw_mask"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = verify(root, None)
        assert result["status"] == "CAPTURE_VERIFIED_TYPED_BLOCKED"
        assert result["typed_blocker"]["code"] == "PF90088_RAW_MASK_POSTPROCESS_RECEIPT"
    print("SELFTEST=PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify_parser = sub.add_parser("verify", help="validate a bundle and emit typed receipts")
    verify_parser.add_argument("--bundle", type=Path, required=True)
    verify_parser.add_argument("--out", type=Path)
    doctor_parser = sub.add_parser("doctor", help="inspect the read-only debugger channel")
    doctor_parser.add_argument("--binary", type=Path)
    doctor_parser.add_argument("--cdb", type=Path)
    sub.add_parser("selftest", help="run an in-memory synthetic contract test")
    args = parser.parse_args()
    try:
        if args.command == "selftest":
            selftest()
            return 0
        if args.command == "doctor":
            print(json.dumps(debugger_doctor(args.binary, args.cdb), indent=2))
            return 0
        result = verify(args.bundle.resolve(), args.out)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "CAPTURE_VERIFIED_TYPED_READY" else 2
    except CaptureError as error:
        print(
            json.dumps(
                {"status": "REFUSED", "code": error.code, "detail": error.detail},
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
