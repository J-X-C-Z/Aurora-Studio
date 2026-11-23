"""Validate PF90116 model 814444604 with ONNX Runtime on CPU.

The harness verifies the sealed ONNX artifact and receipt, the Rust discovery
contract, the real input/golden evidence, and both formal outputs. Geometry is
checked as a separate same-job requirement after inference succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


MODEL_ID = 814_444_604
MODEL_SHA256 = "82b5e7575606430f899866601592c50de7b8756387df0e13654cf5e1fffb7312"
NET_SHA256 = "036768d2d3a388380d58a66c274835eba4b4e4f2b06f12f340db4031cf6a9def"
INPUT_NAME = "t39"
INPUT_CONTRACT_NAME = "781141184"
RESPONSE_NAME = "3204548346"
COLOR_NAME = "780929437"
INPUT_SHAPE = [1, 3, 384, 512]
RESPONSE_SHAPE = [1, 1, 384, 512]
COLOR_SHAPE = [1, 3, 384, 512]
SIDECAR_SCHEMA = "aurora/pf90116-onnx-artifact/v1"
GEOMETRY_SCHEMA = "aurora/pf90116-runtime-geometry/v1"
CANONICAL_MODEL = Path(
    r"C:\WorkSpace\Aurora Studio\90_work\build-cache\pf90116\814444604.onnx"
)
RUST_SOURCE = Path("crates/engine/src/portrait_eye_bag_stage.rs")


class Refusal(RuntimeError):
    def __init__(self, code: str, detail: str, stage: str = "preflight") -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.stage = stage


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path | None, code: str, label: str) -> Path:
    if path is None or not path.is_file():
        raise Refusal(code, f"{label} file is missing: {path}")
    return path.resolve()


def load_json(path: Path, code: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Refusal(code, f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise Refusal(code, f"{label} root must be an object")
    return value


def parse_receipt(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise Refusal("PF90116_ARTIFACT_RECEIPT", f"unreadable receipt: {error}") from error
    for line in lines:
        fields = line.split()
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            raise Refusal("PF90116_ARTIFACT_RECEIPT", f"invalid receipt line: {line!r}")
        entries[fields[1]] = fields[0]
    if not entries:
        raise Refusal("PF90116_ARTIFACT_RECEIPT", "receipt is empty")
    return entries


def verify_receipt(artifact: Path, sidecar: Path, receipt: Path) -> dict[str, Any]:
    expected = parse_receipt(receipt)
    actual = {
        artifact.name: sha256_file(artifact),
        sidecar.name: sha256_file(sidecar),
    }
    for name, digest in actual.items():
        if expected.get(name) != digest:
            raise Refusal(
                "PF90116_ARTIFACT_RECEIPT",
                f"{name} receipt mismatch: expected {expected.get(name)}, got {digest}",
            )
    return {"receipt": str(receipt), "entries": actual}


def rust_string(source: str, name: str) -> str:
    match = re.search(
        rf'pub const {re.escape(name)}: &str\s*=\s*r?"([^"]+)";',
        source,
    )
    if match is None:
        raise Refusal("PF90116_RUST_CONTRACT", f"cannot read Rust constant {name}")
    return match.group(1)


def rust_usize_array(source: str, name: str) -> list[int]:
    match = re.search(
        rf"pub const {re.escape(name)}: \[usize; \d+\]\s*=\s*\[([^\]]+)\];",
        source,
    )
    if match is None:
        raise Refusal("PF90116_RUST_CONTRACT", f"cannot read Rust constant {name}")
    return [int(item.strip()) for item in match.group(1).split(",") if item.strip()]


def rust_sha256(source: str, name: str) -> str:
    match = re.search(
        rf"pub const {re.escape(name)}: \[u8; 32\]\s*=\s*\[(.*?)\];",
        source,
        re.DOTALL,
    )
    if match is None:
        raise Refusal("PF90116_RUST_CONTRACT", f"cannot read Rust constant {name}")
    octets = re.findall(r"0x([0-9a-fA-F]{2})", match.group(1))
    if len(octets) != 32:
        raise Refusal("PF90116_RUST_CONTRACT", f"{name} does not contain 32 octets")
    return bytes(int(octet, 16) for octet in octets).hex()


def verify_rust_contract(source_path: Path, sidecar: dict[str, Any]) -> dict[str, Any]:
    path = require_file(source_path, "PF90116_RUST_CONTRACT", "Rust contract source")
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise Refusal("PF90116_RUST_CONTRACT", f"cannot read {path}: {error}") from error

    expected_path = Path(rust_string(source, "PF90116_DEFAULT_MODEL_PATH"))
    artifact_path = Path(str(sidecar.get("artifact_path", "")))
    checks = {
        "default_model_path": expected_path,
        "sidecar_artifact_path": artifact_path,
        "model_env": rust_string(source, "PF90116_MODEL_PATH_ENV"),
        "graph_input": rust_string(source, "PF90116_GRAPH_INPUT_NAME"),
        "contract_input": rust_string(source, "PF90116_INPUT_NAME"),
        "response": rust_string(source, "PF90116_RESPONSE_NAME"),
        "color": rust_string(source, "PF90116_COLOR_NAME"),
        "input_shape": rust_usize_array(source, "PF90116_INPUT_SHAPE"),
        "response_shape": rust_usize_array(source, "PF90116_RESPONSE_SHAPE"),
        "color_shape": rust_usize_array(source, "PF90116_COLOR_SHAPE"),
        "payload_sha256": rust_sha256(source, "PF90116_MODEL_PAYLOAD_SHA256"),
        "net_sha256": rust_sha256(source, "PF90116_MODEL_NET_SHA256"),
    }
    if expected_path != artifact_path:
        raise Refusal(
            "PF90116_MODEL_DISCOVERY",
            f"Rust default {expected_path} != sidecar artifact {artifact_path}",
        )
    if checks["graph_input"] != INPUT_NAME or checks["contract_input"] != INPUT_CONTRACT_NAME:
        raise Refusal("PF90116_MODEL_DISCOVERY", "Rust input names drifted")
    if checks["input_shape"] != INPUT_SHAPE:
        raise Refusal("PF90116_MODEL_DISCOVERY", "Rust input shape drifted")
    if checks["response"] != RESPONSE_NAME or checks["response_shape"] != RESPONSE_SHAPE:
        raise Refusal("PF90116_MODEL_DISCOVERY", "Rust response contract drifted")
    if checks["color"] != COLOR_NAME or checks["color_shape"] != COLOR_SHAPE:
        raise Refusal("PF90116_MODEL_DISCOVERY", "Rust color contract drifted")
    if checks["payload_sha256"] != sidecar.get("onnx_sha256"):
        raise Refusal("PF90116_MODEL_DISCOVERY", "Rust payload SHA differs from sidecar")
    source_net = sidecar.get("source", {}).get("private_fb_sha256")
    if checks["net_sha256"] != source_net:
        raise Refusal("PF90116_MODEL_DISCOVERY", "Rust net SHA differs from sidecar")
    required = [
        "std::env::var_os(PF90116_MODEL_PATH_ENV)",
        "canonical_path.is_file()",
        'MissingInput("PF90116_MODEL_PATH")',
    ]
    if any(fragment not in source for fragment in required):
        raise Refusal("PF90116_MODEL_DISCOVERY", "Rust env/canonical discovery branch drifted")
    return {key: value for key, value in checks.items()}


def verify_sidecar(sidecar: dict[str, Any], artifact: Path) -> None:
    if sidecar.get("schema") != SIDECAR_SCHEMA:
        raise Refusal("PF90116_ARTIFACT_RECEIPT", "unexpected sidecar schema")
    if sidecar.get("model_id") != MODEL_ID:
        raise Refusal("PF90116_ARTIFACT_RECEIPT", "unexpected model id")
    if sidecar.get("onnx_sha256") != MODEL_SHA256:
        raise Refusal("PF90116_ARTIFACT_RECEIPT", "unexpected sealed artifact SHA")
    if sha256_file(artifact) != MODEL_SHA256:
        raise Refusal("PF90116_MODEL_PAYLOAD_SHA256", "artifact payload SHA mismatch")
    if sidecar.get("bytes") != artifact.stat().st_size:
        raise Refusal("PF90116_ARTIFACT_RECEIPT", "artifact byte count mismatch")
    if sidecar.get("source", {}).get("private_fb_sha256") != NET_SHA256:
        raise Refusal("PF90116_ARTIFACT_RECEIPT", "private-net SHA mismatch")

    inputs = sidecar.get("inputs")
    outputs = sidecar.get("outputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise Refusal("PF90116_MODEL_INPUT_CONTRACT", "sidecar must declare one input")
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise Refusal("PF90116_MODEL_OUTPUT_CONTRACT", "sidecar must declare two outputs")
    input_meta = inputs[0]
    if (
        input_meta.get("name") != INPUT_NAME
        or input_meta.get("contract_name") != INPUT_CONTRACT_NAME
        or input_meta.get("dtype") != "float32"
        or input_meta.get("shape") != INPUT_SHAPE
    ):
        raise Refusal("PF90116_MODEL_INPUT_CONTRACT", "sidecar input contract drifted")
    by_name = {item.get("name"): item for item in outputs}
    if set(by_name) != {RESPONSE_NAME, COLOR_NAME}:
        raise Refusal("PF90116_MODEL_OUTPUT_CONTRACT", "sidecar output names drifted")
    for name, shape in ((RESPONSE_NAME, RESPONSE_SHAPE), (COLOR_NAME, COLOR_SHAPE)):
        item = by_name[name]
        if item.get("dtype") != "float32" or item.get("shape") != shape:
            raise Refusal("PF90116_MODEL_OUTPUT_CONTRACT", f"sidecar {name} contract drifted")


def load_input(path: Path, expected_sha256: str) -> np.ndarray:
    if sha256_file(path) != expected_sha256:
        raise Refusal("PF90116_INPUT_SHA256", "real model input SHA mismatch")
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise Refusal("PF90116_INPUT", f"cannot load real model input: {error}") from error
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise Refusal("PF90116_INPUT_DTYPE", "real input must be a float32 ndarray")
    if list(value.shape) != INPUT_SHAPE:
        raise Refusal(
            "PF90116_INPUT_SHAPE",
            f"real input shape {list(value.shape)} != {INPUT_SHAPE}",
        )
    if not np.isfinite(value).all():
        raise Refusal("PF90116_INPUT_VALUES", "real input contains non-finite values")
    return np.ascontiguousarray(value)


def load_golden(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if sha256_file(path) != expected_sha256:
        raise Refusal("PF90116_GOLDEN_SHA256", "golden archive SHA mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            golden = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise Refusal("PF90116_GOLDEN", f"cannot load golden archive: {error}") from error
    if set(golden) != {RESPONSE_NAME, COLOR_NAME}:
        raise Refusal("PF90116_GOLDEN", "golden output names drifted")
    for name, shape in ((RESPONSE_NAME, RESPONSE_SHAPE), (COLOR_NAME, COLOR_SHAPE)):
        if golden[name].dtype != np.float32 or list(golden[name].shape) != shape:
            raise Refusal("PF90116_GOLDEN", f"golden {name} dtype/shape drifted")
    return golden


def verify_session_contract(session: ort.InferenceSession) -> dict[str, Any]:
    providers = session.get_providers()
    if providers != ["CPUExecutionProvider"]:
        raise Refusal("PF90116_ORT_PROVIDER", f"CPU-only session required, got {providers}")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise Refusal("PF90116_MODEL_INPUT_CONTRACT", "ORT session must expose one input")
    if len(outputs) != 2:
        raise Refusal("PF90116_MODEL_OUTPUT_CONTRACT", "ORT session must expose two outputs")
    item = inputs[0]
    if item.name != INPUT_NAME or item.type != "tensor(float)" or item.shape != INPUT_SHAPE:
        raise Refusal("PF90116_MODEL_INPUT_CONTRACT", "ORT input metadata drifted")
    output_meta = {item.name: (item.type, item.shape) for item in outputs}
    expected = {
        RESPONSE_NAME: ("tensor(float)", RESPONSE_SHAPE),
        COLOR_NAME: ("tensor(float)", COLOR_SHAPE),
    }
    if output_meta != expected:
        raise Refusal("PF90116_MODEL_OUTPUT_CONTRACT", "ORT output metadata drifted")
    return {
        "providers": providers,
        "input": {"name": item.name, "type": item.type, "shape": item.shape},
        "outputs": {
            name: {"type": kind, "shape": shape}
            for name, (kind, shape) in output_meta.items()
        },
    }


def verify_geometry(path: Path | None) -> dict[str, Any]:
    path = require_file(path, "PF90116_GEOMETRY", "same-job geometry")
    geometry = load_json(path, "PF90116_GEOMETRY", "same-job geometry")
    required = {
        "schema",
        "job_ticket",
        "vendor_job_id",
        "face_record_id",
        "landmark_receipt",
        "degree",
        "frame_to_model",
        "model_roi",
        "work_width",
        "work_height",
        "work_to_destination",
        "destination_roi",
    }
    if set(geometry) != required or geometry.get("schema") != GEOMETRY_SCHEMA:
        raise Refusal("PF90116_GEOMETRY_FIELDS", "geometry schema or fields drifted")
    for name in ("job_ticket", "vendor_job_id", "face_record_id", "landmark_receipt"):
        if not isinstance(geometry[name], int) or geometry[name] <= 0:
            raise Refusal("PF90116_GEOMETRY_IDENTITY", f"{name} must be positive")
    degree = geometry["degree"]
    if not isinstance(degree, (int, float)) or not math.isfinite(degree) or degree <= 0.001:
        raise Refusal("PF90116_ACTIVE_DEGREE", "degree must be above 0.001")
    for name in ("frame_to_model", "work_to_destination"):
        value = geometry[name]
        if not isinstance(value, list) or len(value) != 6:
            raise Refusal("PF90116_GEOMETRY_FIELDS", f"{name} must contain six values")
        if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value):
            raise Refusal("PF90116_GEOMETRY_FIELDS", f"{name} contains invalid values")
    for name in ("model_roi", "destination_roi"):
        value = geometry[name]
        if not isinstance(value, list) or len(value) != 4:
            raise Refusal("PF90116_GEOMETRY_FIELDS", f"{name} must contain four values")
        if not all(isinstance(item, int) for item in value) or value[2] <= 0 or value[3] <= 0:
            raise Refusal("PF90116_GEOMETRY_FIELDS", f"{name} is invalid")
    if not isinstance(geometry["work_width"], int) or geometry["work_width"] <= 0:
        raise Refusal("PF90116_GEOMETRY_FIELDS", "work_width must be positive")
    if not isinstance(geometry["work_height"], int) or geometry["work_height"] <= 0:
        raise Refusal("PF90116_GEOMETRY_FIELDS", "work_height must be positive")
    return {"path": str(path), "schema": GEOMETRY_SCHEMA, "accepted": True}


def run_model(
    artifact: Path,
    input_value: np.ndarray,
    golden: dict[str, np.ndarray],
    sidecar: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        session = ort.InferenceSession(
            str(artifact),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as error:
        raise Refusal("PF90116_MODEL_SESSION", f"CPU ORT session failed: {error}") from error
    contract = verify_session_contract(session)
    try:
        values = session.run([RESPONSE_NAME, COLOR_NAME], {INPUT_NAME: input_value})
    except Exception as error:
        raise Refusal("PF90116_MODEL_INFERENCE", f"CPU ORT inference failed: {error}") from error

    expected_verification = sidecar.get("verification", {}).get("outputs", {})
    actual: dict[str, Any] = {}
    exact = True
    for name, value in zip((RESPONSE_NAME, COLOR_NAME), values):
        array = np.asarray(value)
        expected = golden[name]
        same = bool(np.array_equal(array, expected))
        exact = exact and same
        digest = sha256_bytes(array.tobytes(order="C"))
        expected_meta = expected_verification.get(name, {})
        if expected_meta.get("sha256") != digest:
            raise Refusal("PF90116_OUTPUT_SHA256", f"{name} output SHA mismatch")
        for field, reducer in (
            ("min", np.min),
            ("max", np.max),
            ("mean", np.mean),
        ):
            observed = float(reducer(array))
            wanted = expected_meta.get(field)
            if not isinstance(wanted, (int, float)) or not math.isclose(
                observed, wanted, rel_tol=0.0, abs_tol=1e-12
            ):
                raise Refusal("PF90116_OUTPUT_VALUES", f"{name} {field} drifted")
        actual[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": digest,
            "golden_bit_exact": same,
            "golden_max_abs": float(np.abs(array - expected).max()),
        }
    if sidecar.get("verification", {}).get("golden_bit_exact") is not True or not exact:
        raise Refusal("PF90116_GOLDEN_PARITY", "CPU outputs are not bit-exact to golden")
    return contract, {
        "backend": f"onnxruntime {ort.__version__}/CPUExecutionProvider",
        "golden_bit_exact": exact,
        "outputs": actual,
    }


def resolve_paths(args: argparse.Namespace) -> dict[str, Any]:
    artifact = args.artifact or Path(os.environ.get("AURORA_PF90116_MODEL", CANONICAL_MODEL))
    artifact = require_file(artifact, "PF90116_MODEL_FILE", "ONNX artifact")
    sidecar = require_file(
        args.sidecar or artifact.with_name(artifact.name + ".meta.json"),
        "PF90116_ARTIFACT_RECEIPT",
        "artifact sidecar",
    )
    receipt = require_file(
        args.receipt or artifact.with_name(artifact.name + ".sha256"),
        "PF90116_ARTIFACT_RECEIPT",
        "artifact receipt",
    )
    sidecar_data = load_json(sidecar, "PF90116_ARTIFACT_RECEIPT", "artifact sidecar")
    source = sidecar_data.get("source", {})
    data_root = Path(str(source.get("artifact_root", "")))
    input_path = args.input or Path(
        os.environ.get("AURORA_PF90116_INPUT", data_root / "assets/input_real_384x512.npy")
    )
    golden_path = args.golden or Path(
        os.environ.get("AURORA_PF90116_GOLDEN", data_root / "assets/forward_outputs.npz")
    )
    geometry_path = args.geometry or (
        Path(os.environ["AURORA_PF90116_GEOMETRY"])
        if os.environ.get("AURORA_PF90116_GEOMETRY")
        else None
    )
    return {
        "artifact": artifact,
        "sidecar": sidecar,
        "sidecar_data": sidecar_data,
        "receipt": receipt,
        "input": input_path,
        "golden": golden_path,
        "geometry": geometry_path,
        "rust_source": args.rust_source.resolve(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--rust-source", type=Path, default=RUST_SOURCE)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "schema": "aurora/pf90116-runtime-harness/v1",
        "model_id": MODEL_ID,
        "status": "FAIL",
    }
    try:
        paths = resolve_paths(args)
        sidecar = paths["sidecar_data"]
        result["paths"] = {key: str(value) for key, value in paths.items() if isinstance(value, Path)}
        result["receipt"] = verify_receipt(paths["artifact"], paths["sidecar"], paths["receipt"])
        verify_sidecar(sidecar, paths["artifact"])
        result["rust_discovery"] = verify_rust_contract(paths["rust_source"], sidecar)
        source = sidecar.get("source", {})
        input_path = require_file(paths["input"], "PF90116_INPUT", "real model input")
        golden_path = require_file(paths["golden"], "PF90116_GOLDEN", "golden archive")
        input_value = load_input(input_path, str(source.get("input_sha256", "")))
        golden = load_golden(golden_path, str(source.get("golden_sha256", "")))
        contract, runtime = run_model(paths["artifact"], input_value, golden, sidecar)
        result["artifact"] = {
            "sha256": MODEL_SHA256,
            "bytes": paths["artifact"].stat().st_size,
            "sidecar_schema": sidecar["schema"],
        }
        result["input"] = {"sha256": sha256_file(input_path), "shape": INPUT_SHAPE}
        result["session"] = contract
        result["runtime"] = runtime
        result["geometry"] = verify_geometry(paths["geometry"])
        result["status"] = "PASS"
        exit_code = 0
    except Refusal as refusal:
        result["refusal"] = {
            "code": refusal.code,
            "detail": refusal.detail,
            "stage": refusal.stage,
        }
        exit_code = 3 if refusal.code == "PF90116_GEOMETRY" else 2
    except Exception as error:
        result["refusal"] = {
            "code": "PF90116_HARNESS_INTERNAL",
            "detail": str(error),
            "stage": "internal",
        }
        exit_code = 4

    payload = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    print(payload)
    if args.json_out is not None:
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
