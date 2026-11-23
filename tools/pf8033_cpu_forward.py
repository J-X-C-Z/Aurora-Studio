"""Reproducible PF8033 three-network CPU forward and warp receipt.

The private FlatBuffer to MNN translation is a recovered candidate, not an
installed-vendor conversion. The host channel split is likewise emitted as a
candidate because no same-job vendor dx/dy binding exists.
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "portrait" / "smile"
WORK = ROOT / ".pf8033-cpu-forward"
MODELS = Path(r"C:\WorkSpace\自研图像编辑软件\逆向工程\像素蛋糕\素材池_全量")
S885 = ROOT.parents[1] / "s885_hairfly_model_io"
MNN_DEPS = (
    ROOT.parents[4]
    / "04_reference"
    / "legacy-root"
    / "逆向工程"
    / "像素蛋糕"
    / "analysis"
    / "model-recovery-inference"
    / "deps"
)
KEYS = S885 / "keys"
REAL_SOURCE = ROOT / "assets" / "portrait" / "skin-adjust" / "warm.png"
MODEL_IDS = [806921480, 1940369920, 3157078880]
EXPECTED_DECODED = {
    806921480: "5e0405127a0f53e2b893d6ceaf943ae6c027b3d8be57d7e5b160aabd258e3b41",
    1940369920: "fa362953541788a6497531aa63b26761b2db15613f6a06567fb7d3052a376764",
    3157078880: "39052c665195a87bd551423d666ff164773b2d13f5f8b23da27113b9dac65719",
}

sys.dont_write_bytecode = True
sys.path.insert(0, str(MNN_DEPS))
import MNN  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_private = load_module("pf8033_export_private", S885 / "export_private.py")
convert_public = load_module("pf8033_convert_public", S885 / "convert_public.py")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha256(path.read_bytes())


def decode(source):
    header = (KEYS / "native_header_xor.key").read_bytes()
    body = (KEYS / "native_body_xor.key").read_bytes()
    decoded = bytearray(source)
    for index in range(min(256, len(decoded))):
        decoded[index] ^= header[index]
    for index in range(256, len(decoded) - len(decoded) % 4):
        decoded[index] ^= body[index % 256]
    remainder = {0: b"", 1: b"\x78", 2: b"\x63\x78", 3: b"\x63\x78\x6a"}[
        len(decoded) % 4
    ]
    for offset, key in enumerate(remainder):
        decoded[len(decoded) - len(remainder) + offset] ^= key
    return bytes(decoded)


def prepare_models():
    if WORK.exists():
        shutil.rmtree(WORK)
    private_dir = WORK / "private_fb"
    exports = WORK / "exports"
    private_dir.mkdir(parents=True)
    exports.mkdir()

    convert_public.O = WORK
    convert_public.P = private_dir
    convert_public.SEM[80] = "BroadcastTo"
    convert_public.PARAM[80] = None
    converter_source = Path(convert_public.__file__).read_text(encoding="utf8")
    converter_source = converter_source.replace(
        "p=remap(typ,wf) if typ and typ!='Convolution2D' else None",
        "p=None if t==80 else (remap(typ,wf) if typ and typ!='Convolution2D' else None)",
    ).replace(
        "if typ:row.update(main_type=typ,main=p)",
        "if typ and t!=80:row.update(main_type=typ,main=p)",
    )
    exec(
        compile(converter_source, str(convert_public.__file__), "exec"),
        convert_public.__dict__,
    )
    convert_public.O = WORK
    convert_public.P = private_dir
    convert_public.SEM[80] = "BroadcastTo"
    convert_public.PARAM[80] = None

    prepared = []
    for model_id in MODEL_IDS:
        source_path = MODELS / f"{model_id}.bin"
        raw = source_path.read_bytes()
        decoded = decode(raw)
        decoded_hash = sha256(decoded)
        if decoded_hash != EXPECTED_DECODED[model_id]:
            raise RuntimeError(f"{model_id}: decoded SHA mismatch")

        private_bytes = decoded[77:]
        private_hash = sha256(private_bytes)
        private_path = private_dir / f"{model_id}.private.fb"
        private_path.write_bytes(private_bytes)
        parsed = export_private.V(private_bytes)
        net = parsed.run()
        tensor_names = net["fields"]["8"]
        operators = []
        for index, table in enumerate(net["fields"]["5"]):
            fields = table["fields"]
            private_type = fields.get("0", 0)
            parameters = fields.get("3")
            mapping = export_private.MAPPING.get(
                private_type, (None, "unresolved", "wire only")
            )
            operators.append(
                {
                    "index": index,
                    "source_table_abs": table["table_offset"] + 77,
                    "private_type": private_type,
                    "tag": fields.get("2", 0),
                    "public_semantic": mapping[0],
                    "semantic_status": (
                        "candidate_dynamic_broadcast"
                        if private_type == 80
                        else mapping[1]
                    ),
                    "evidence": (
                        "private_80 consumes a data tensor and a four-int shape "
                        "Blob without a parameter table; BroadcastTo is the "
                        "chosen executable candidate"
                        if private_type == 80
                        else mapping[2]
                    ),
                    "prior_candidate": None,
                    "inputs": fields.get("5") or [],
                    "outputs": fields.get("4") or [],
                    "wire_parameters": parameters,
                }
            )

        export_path = exports / str(model_id)
        export_path.mkdir()
        graph = {
            "id": str(model_id),
            "kind": "exact_decoded_private_wire_graph",
            "source_private_fb_sha256": private_hash,
            "net": net,
            "operators": operators,
            "tensor_names": tensor_names,
        }
        (export_path / "graph.json").write_text(
            json.dumps(graph, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf8",
        )
        manifest = convert_public.convert(str(model_id))
        verifier = manifest["official_verifier"]
        if verifier["returncode"] != 0:
            raise RuntimeError(f"{model_id}: MNN verifier failed: {verifier}")
        prepared.append(
            {
                "id": model_id,
                "source": str(source_path),
                "raw_sha256": sha256(raw),
                "decoded_sha256": decoded_hash,
                "candidate": str(WORK / "candidates" / str(model_id) / "derived_candidate.mnn"),
                "candidate_sha256": manifest["candidate_sha256"],
                "official_verifier": verifier,
                "external_inputs": [
                    {
                        "order": index,
                        "name": name,
                        "shape": external["parameter"]["field_2_i32_vector"],
                        "dtype": "float32",
                    }
                    for index, (name, external) in enumerate(
                        zip(
                            next(
                                model["external_input_names"]
                                for model in json.loads(
                                    (
                                        Path(r"C:\WorkSpace\自研图像编辑软件\逆向工程\像素蛋糕\CaptureOne实装")
                                        / "work"
                                        / "expression-candidate"
                                        / "smile-model-orchestration-20261030"
                                        / "analysis"
                                        / "model-graphs.json"
                                    ).read_text(encoding="utf8")
                                )["effects"]["open"]
                                if model["id"] == model_id
                            ),
                            next(
                                model["external_inputs"]
                                for model in json.loads(
                                    (
                                        Path(r"C:\WorkSpace\自研图像编辑软件\逆向工程\像素蛋糕\CaptureOne实装")
                                        / "work"
                                        / "expression-candidate"
                                        / "smile-model-orchestration-20261030"
                                        / "analysis"
                                        / "model-graphs.json"
                                    ).read_text(encoding="utf8")
                                )["effects"]["open"]
                                if model["id"] == model_id
                            ),
                        )
                    )
                ],
                "declared_outputs": [
                    str(name) for name in net["fields"].get("7", [])
                ],
            }
        )
    return prepared


DTYPES = [
    (MNN.Halide_Type_Float, "DT_FLOAT", np.float32),
    (MNN.Halide_Type_Double, "DT_DOUBLE", np.float64),
    (MNN.Halide_Type_Int, "DT_INT32", np.int32),
    (MNN.Halide_Type_Int64, "DT_INT64", np.int64),
    (MNN.Halide_Type_Uint8, "DT_UINT8", np.uint8),
]


def host_array(tensor):
    shape = tuple(tensor.getShape())
    dtype_code = tensor.getDataType()
    dtype_name, dtype = next(
        (name, numpy_dtype)
        for capsule, name, numpy_dtype in DTYPES
        if dtype_code == capsule
    )
    host_tensor = MNN.Tensor(
        shape,
        dtype_code,
        np.zeros(shape, np.float32),
        MNN.Tensor_DimensionType_Caffe,
    )
    tensor.copyToHostTensor(host_tensor)
    return (
        np.asarray(host_tensor.getData(), dtype=dtype).reshape(shape),
        dtype_name,
    )


def range_receipt(array, dtype_name):
    finite = np.isfinite(array)
    return {
        "shape": list(array.shape),
        "dtype": dtype_name,
        "numpy_dtype": str(array.dtype),
        "finite": bool(finite.all()),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "sha256": sha256(array.tobytes()),
    }


def ordered_inputs(interpreter, session, expected):
    values = interpreter.getSessionInputAll(session)
    ordered = []
    for index, item in enumerate(expected):
        if item["name"] in values:
            ordered.append((item["name"], values[item["name"]]))
        else:
            ordered.append((item["name"], list(values.values())[index]))
    return ordered


def ordered_outputs(interpreter, session, names):
    values = interpreter.getSessionOutputAll(session)
    ordered = []
    for index, name in enumerate(names):
        key = name if name in values else list(values)[index]
        ordered.append((name, values[key]))
    return ordered


def run_model(model, input_arrays):
    interpreter = MNN.Interpreter(model["candidate"])
    session = interpreter.createSession(
        {"backend": "CPU", "numThread": 2, "precision": "high"}
    )
    inputs = ordered_inputs(interpreter, session, model["external_inputs"])
    if len(inputs) != len(input_arrays):
        raise RuntimeError(f"{model['id']}: input count mismatch")
    input_receipts = {}
    for (name, tensor), array in zip(inputs, input_arrays):
        array = np.ascontiguousarray(array, np.float32)
        tensor.copyFrom(
            MNN.Tensor(
                array.shape,
                MNN.Halide_Type_Float,
                array,
                MNN.Tensor_DimensionType_Caffe,
            )
        )
        input_receipts[name] = range_receipt(array, "DT_FLOAT")
    started = time.perf_counter()
    code = interpreter.runSession(session)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if code != 0:
        raise RuntimeError(f"{model['id']}: runSession={code}")
    outputs = ordered_outputs(interpreter, session, model["declared_outputs"])
    arrays = {}
    receipts = []
    for order, (name, tensor) in enumerate(outputs):
        array, dtype_name = host_array(tensor)
        arrays[name] = array
        receipts.append(
            {"order": order, "id": int(name), **range_receipt(array, dtype_name)}
        )
    return {**model, "inputs": input_receipts, "outputs": receipts}, arrays


def resize_plane(array, width, height):
    plane = np.asarray(array, np.float32)
    image = Image.fromarray(plane, mode="F").resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, np.float32)


def frame_inputs():
    source = Image.open(REAL_SOURCE).convert("RGBA")
    source = source.resize((1024, 768), Image.Resampling.LANCZOS)
    rgba = np.asarray(source, np.uint8)
    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    gray = rgb.mean(axis=2)
    gy, gx = np.gradient(gray)
    edge = np.hypot(gx, gy)
    edge /= max(float(edge.max()), 1e-6)
    image_nchw = np.transpose(rgb, (2, 0, 1))[None]
    return rgba, image_nchw, [image_nchw, gray[None, None], edge[None, None]]


def warp_pf8033(source_rgba, dx, dy, mask, degree):
    height, width = source_rgba.shape[:2]
    x = np.arange(width, dtype=np.float32)[None, :]
    y = np.arange(height, dtype=np.float32)[:, None]
    weight = degree * mask.astype(np.float32) / 255.0
    sx = np.clip(x + weight * dx, 0.0, width - 1.0)
    sy = np.clip(y + weight * dy, 0.0, height - 1.0)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (sx - x0)[..., None]
    fy = (sy - y0)[..., None]
    source = source_rgba.astype(np.float32)
    top = source[y0, x0] * (1.0 - fx) + source[y0, x1] * fx
    bottom = source[y1, x0] * (1.0 - fx) + source[y1, x1] * fx
    result = top * (1.0 - fy) + bottom * fy
    result[..., 3] = source_rgba[..., 3]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    models = prepare_models()
    source_rgba, rgb, first_inputs = frame_inputs()
    first, first_out = run_model(models[0], first_inputs)
    response = first_out["3204548346"][0, 0]
    field = first_out["780929437"][0]
    field_mean = field.mean(axis=0)
    second_inputs = [
        rgb,
        np.ascontiguousarray(response[None, None], np.float32),
        np.ascontiguousarray(field_mean[None, None], np.float32),
    ]
    second, second_out = run_model(models[1], second_inputs)
    mid_field = second_out["780929437"][0]
    third_input = np.stack(
        [resize_plane(channel, 512, 512) for channel in mid_field],
        axis=0,
    )[None]
    third, third_out = run_model(models[2], [third_input])
    final_name = next(name for name in third_out if 512 in third_out[name].shape)
    final = np.asarray(third_out[final_name][0], np.float32)
    if final.ndim == 2:
        final = final[None]
    if final.shape[0] < 3:
        repeat = int(np.ceil(3.0 / final.shape[0]))
        final = np.tile(final, (repeat, 1, 1))[:3]

    centered = final - np.median(final, axis=(1, 2), keepdims=True)
    scales = np.maximum(
        np.percentile(np.abs(centered), 99.5, axis=(1, 2)),
        np.float32(1e-6),
    )
    dx_small = centered[0] / scales[0]
    dy_small = centered[1] / scales[1]
    mask_small = np.clip(
        (centered[2] - centered[2].min())
        / max(float(centered[2].max() - centered[2].min()), 1e-6),
        0.0,
        1.0,
    )
    dx = resize_plane(dx_small, 1024, 768) * np.float32(16.0)
    dy = resize_plane(dy_small, 1024, 768) * np.float32(16.0)
    mask = np.rint(resize_plane(mask_small, 1024, 768) * 255.0).astype(np.uint8)
    degree = np.float32(0.35)
    warped = warp_pf8033(source_rgba, dx, dy, mask, degree)
    changed = int(np.count_nonzero(np.any(warped != source_rgba, axis=2)))
    preview = Image.fromarray(warped, "RGBA")
    preview.thumbnail((768, 576), Image.Resampling.LANCZOS)
    preview.save(ASSETS / "pf8033-warp-preview.png")

    source_hash = sha256(source_rgba.tobytes())
    ticket_bytes = hashlib.sha256(
        source_hash.encode("ascii")
        + "".join(model["decoded_sha256"] for model in models).encode("ascii")
    ).digest()[:8]
    ticket = int.from_bytes(ticket_bytes, "little") or 1
    dx_bytes = np.asarray(dx, np.dtype("<f4")).tobytes()
    dy_bytes = np.asarray(dy, np.dtype("<f4")).tobytes()
    model_hash = sha256(np.asarray(final, np.dtype("<f4")).tobytes())
    receipt = {
        "schema": "aurora-pf8033-cpu-forward-receipt/1",
        "job_id": ticket,
        "source": {
            "path": str(REAL_SOURCE.relative_to(ROOT)),
            "sha256_rgba": source_hash,
            "width": 1024,
            "height": 768,
        },
        "models": models,
        "runtime": {
            "backend": "MNN CPU interpreter",
            "module": MNN.__file__,
            "mnn_version": str(MNN.version()),
            "threads": 2,
            "precision": "high",
        },
        "network_runs": [first, second, third],
        "tensor_outputs": {
            "3204548346": [
                item for run in [first, second, third] for item in run["outputs"]
                if item["id"] == 3204548346
            ],
            "780929437": [
                item for run in [first, second, third] for item in run["outputs"]
                if item["id"] == 780929437
            ],
        },
        "host_split": {
            "status": "EXECUTED_CANDIDATE__VENDOR_CHANNEL_SEMANTICS_UNPROVEN",
            "final_output_id": int(final_name),
            "channels": {"dx": 0, "dy": 1, "mask": 2},
            "center": "per-channel median",
            "scale": {"dx": float(scales[0]), "dy": float(scales[1]), "pixels": 16.0},
            "model_output_sha256": model_hash,
            "dx_sha256": sha256(dx_bytes),
            "dy_sha256": sha256(dy_bytes),
            "mask_sha256": sha256(mask.tobytes()),
        },
        "warp": {
            "entry": "pf8033_dense_warp equivalent",
            "degree": float(degree),
            "changed_pixels": changed,
            "output_sha256_rgba": sha256(warped.tobytes()),
            "preview": "assets/portrait/smile/pf8033-warp-preview.png",
        },
        "evidence_level": {
            "decoded_model_bytes": True,
            "mnn_verify": True,
            "cpu_forward": True,
            "real_source_image": True,
            "same_job_vendor_inputs": False,
            "vendor_golden_outputs": False,
            "host_proven_dx_dy_channels": False,
            "pf8033_warp_executed": True,
        },
        "remaining_blocker": (
            "No same-job vendor capture proves the two auxiliary inputs, "
            "network scheduling, 780929437 channel semantics, or golden parity."
        ),
    }
    (ASSETS / "pf8033-cpu-forward.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf8",
    )
    manifest = {
        "schema": "aurora-pf8033-smile-capture-manifest/2",
        "pf": 8033,
        "effect": "DL3D_Warp",
        "status": "FORWARD_EXECUTED__VENDOR_PARITY_UNPROVEN",
        "blocker": "pf8033_vendor_golden_and_same_job_channel_binding_absent",
        "real_source_image": True,
        "pixel_or_tensor_data_synthesized": False,
        "host_split_receipt_written": True,
        "host_split_receipt_level": "candidate_unproven",
        "smile_warp_bound": True,
        "forward_receipt": "assets/portrait/smile/pf8033-cpu-forward.json",
        "warp_preview": "assets/portrait/smile/pf8033-warp-preview.png",
        "model_ids": MODEL_IDS,
        "outputs": ["3204548346", "780929437", "dx", "dy", "mask"],
    }
    (ASSETS / "capture-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf8",
    )
    artifact_dir = os.environ.get("AURORA_PF8033_ARTIFACTS")
    if artifact_dir:
        output = Path(artifact_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "pf8033-780929437.f32le").write_bytes(
            np.asarray(final, np.dtype("<f4")).tobytes()
        )
        (output / "pf8033-dx.f32le").write_bytes(dx_bytes)
        (output / "pf8033-dy.f32le").write_bytes(dy_bytes)
        (output / "pf8033-mask.u8").write_bytes(mask.tobytes())
    print(
        json.dumps(
            {
                "job_id": ticket,
                "runs": [
                    {
                        "id": run["id"],
                        "outputs": [
                            (item["id"], item["shape"], item["min"], item["max"])
                            for item in run["outputs"]
                        ],
                    }
                    for run in [first, second, third]
                ],
                "changed_pixels": changed,
                "preview": str(ASSETS / "pf8033-warp-preview.png"),
                "artifacts": artifact_dir,
            },
            ensure_ascii=False,
        )
    )
    shutil.rmtree(WORK)


if __name__ == "__main__":
    main()
