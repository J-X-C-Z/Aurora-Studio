#!/usr/bin/env python3
"""Rebuild and execute PF90116 model 814444604 from its private flatbuffer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper as H, numpy_helper as N


MODEL_ID = 814_444_604
PRIVATE_FB_SHA256 = "036768d2d3a388380d58a66c274835eba4b4e4f2b06f12f340db4031cf6a9def"
WEIGHTS_SHA256 = "d462556cbe79158651e6f254d2583ff283afcbaa261f0f943743d8623e0cfe26"
INPUT_SHA256 = "a95e042a8df25e4e9e4030c0d4b8a1aa35a9ce625345cadae8d1819290705763"
GOLDEN_SHA256 = "6ae30a3b2ba72bdfe7a35dba18d748077b013ca0e670f640ae6fc2d2a5caab6f"
ONNX_SHA256 = "82b5e7575606430f899866601592c50de7b8756387df0e13654cf5e1fffb7312"
RESPONSE_NAME = "3204548346"
COLOR_NAME = "780929437"
RESPONSE_SHAPE = [1, 1, 384, 512]
COLOR_SHAPE = [1, 3, 384, 512]
INPUT_SHAPE = [1, 3, 384, 512]
DEFAULT_ARTIFACT_ROOT = Path(
    r"C:\WorkSpace\自研图像编辑软件\逆向工程\像素蛋糕\历史成果\c1-pixcake"
    r"\work40\eyebag-next-6869"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pinned_file(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    actual = sha256(data)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}: {actual} != {expected}")
    return data


FMT = {"i": "i", "u": "I", "b": "B", "f": "f", "l": "q"}
LAY = {
    "Net": ["b", "b", "b", "b", "b", "@Op", "i", "@str", "@str", "b", "b", "b", "b", "Extra", "b", "str", "b", "str"],
    "Extra": ["u", "str"],
    "Op": ["i", "str", "b", "PARAM", "[i]", "[i]", "b"],
    "Input": ["i", "b", "[i]"],
    "Conv": ["[f]", "[f]", "[l]", "Quan", "u", "u", "Common"],
    "Common": ["i"] * 8 + ["[i]"] + ["i"] * 3 + ["b"] * 3 + ["[i]", "b"],
    "Quan": ["[b]", "f", "i", "b", "f", "f", "i", "i", "i", "b", "b", "u", "[u]", "[f]"],
    "Blob": ["[i]", "b", "i", "[b]", "[f]", "[l]", "[i]"],
    "T3": ["i"] * 8,
    "T4": ["i", "b", "[i]"],
    "T13": ["i"] * 3 + ["b"] * 3,
    "T14": ["i", "b", "b"],
    "T16": ["i"],
    "T18": ["f", "f", "i", "i", "i", "i", "b", "b", "f", "f", "f", "b", "i", "f"],
    "T20": ["i"] * 6 + ["[i]"] + ["b"] * 5 + ["i"],
    "T21": ["i"] * 3,
    "T23": ["i", "[f]", "[f]"],
    "T24": ["[i]"],
    "T25": ["i", "i"],
    "T26": ["b"],
    "T27": ["i", "[f]"],
    "T29": ["[i]", "b"],
    "T30": ["b", "b"],
    "T39": ["f"],
    "T40": ["i", "i"],
    "T53": ["b", "b", "f", "[i]", "i"],
    "T55": ["i"] * 3,
    "T66": ["f", "f"],
    "T91": ["[i]", "[f]", "f", "[f]", "i"],
}
TAGS = {
    3: "T3",
    4: "T4",
    13: "T13",
    14: "T14",
    16: "T16",
    18: "T18",
    19: "Blob",
    20: "T20",
    21: "T21",
    23: "T23",
    24: "T24",
    25: "T25",
    26: "T26",
    27: "T27",
    29: "T29",
    30: "T30",
    39: "T39",
    40: "T40",
    53: "T53",
    55: "T55",
    62: "Input",
    66: "T66",
    91: "T91",
    93: "Conv",
}


class BadWire(Exception):
    pass


class V:
    def __init__(self, data: bytes):
        self.data = data
        self.visited: dict[int, tuple[str, dict]] = {}
        self.vectors: list[tuple[int, int, str]] = []

    def need(self, position: int, size: int, align: int = 1) -> None:
        if not (0 <= position <= len(self.data) and 0 <= size <= len(self.data) - position):
            raise BadWire(f"bounds:{position}:{size}:{len(self.data)}")
        if position % align:
            raise BadWire(f"alignment:{position}:{align}")

    def value(self, position: int, fmt: str):
        size = struct.calcsize("<" + FMT[fmt])
        self.need(position, size, size)
        return struct.unpack_from("<" + FMT[fmt], self.data, position)[0]

    def pointer(self, position: int) -> int | None:
        offset = self.value(position, "u")
        if not offset:
            return None
        if offset < 4:
            raise BadWire(f"pointer:{offset}")
        self.need(position + offset, 4, 4)
        return position + offset

    def vector(self, position: int, fmt: str):
        count = self.value(position, "u")
        size = struct.calcsize("<" + FMT[fmt])
        self.need(position + 4, count * size, size)
        self.vectors.append((position, count, fmt))
        if count >= 10_000:
            return {"length": count, "data_offset": position + 4}
        return list(struct.unpack_from("<" + str(count) + FMT[fmt], self.data, position + 4))

    def string(self, position: int) -> str:
        count = self.value(position, "u")
        self.need(position + 4, count + 1)
        if self.data[position + 4 + count] != 0:
            raise BadWire("string_without_nul")
        return self.data[position + 4 : position + 4 + count].decode("utf-8")

    def table(self, position: int, name: str, depth: int = 0) -> dict:
        if depth >= 64:
            raise BadWire("table_depth")
        if position in self.visited:
            if self.visited[position][0] != name:
                raise BadWire(f"table_alias:{position}:{name}")
            return self.visited[position][1]
        self.need(position, 4, 4)
        vtable = position - self.value(position, "i")
        self.need(vtable, 4, 2)
        vtable_size, object_size = struct.unpack_from("<HH", self.data, vtable)
        if vtable_size < 4 or vtable_size % 2 or object_size < 4:
            raise BadWire(f"vtable:{vtable_size}:{object_size}")
        self.need(vtable, vtable_size, 2)
        self.need(position, object_size)
        offsets = struct.unpack_from("<" + str((vtable_size - 4) // 2) + "H", self.data, vtable + 4)
        declaration = LAY[name]
        if len(offsets) > len(declaration):
            raise BadWire(f"fields:{name}:{len(offsets)}:{len(declaration)}")
        result = {"table_offset": position, "fields": {}}
        self.visited[position] = (name, result)
        intervals: list[tuple[int, int]] = []
        for index, offset in enumerate(offsets):
            if not offset:
                continue
            field_type = declaration[index]
            width = struct.calcsize("<" + FMT[field_type]) if field_type in FMT else 4
            if not (4 <= offset and offset + width <= object_size):
                raise BadWire(f"field:{name}:{index}:{offset}")
            if any(not (offset + width <= start or offset >= end) for start, end in intervals):
                raise BadWire(f"field_overlap:{name}:{index}")
            intervals.append((offset, offset + width))
            field_position = position + offset
            if field_type in FMT:
                value = self.value(field_position, field_type)
            else:
                destination = self.pointer(field_position)
                if destination is None:
                    value = None
                elif field_type == "str":
                    value = self.string(destination)
                elif field_type.startswith("["):
                    value = self.vector(destination, field_type[1])
                elif field_type.startswith("@"):
                    count = self.value(destination, "u")
                    self.need(destination + 4, count * 4, 4)
                    value = []
                    for item in range(destination + 4, destination + 4 + count * 4, 4):
                        child = self.pointer(item)
                        if child is None:
                            raise BadWire("null_table_entry")
                        value.append(self.string(child) if field_type == "@str" else self.table(child, field_type[1:], depth + 1))
                elif field_type == "PARAM":
                    tag = result["fields"].get("2", 0)
                    if tag not in TAGS:
                        raise BadWire(f"unknown_tag:{tag}")
                    value = self.table(destination, TAGS[tag], depth + 1)
                else:
                    value = self.table(destination, field_type, depth + 1)
            result["fields"][str(index)] = value
        if name == "Op":
            tag = result["fields"].get("2", 0)
            parameter = result["fields"].get("3")
            if (tag == 0) != (parameter is None):
                raise BadWire("op_union_mismatch")
        return result

    def run(self) -> dict:
        root = self.pointer(0)
        if root is None:
            raise BadWire("null_root")
        net = self.table(root, "Net")
        ops = net["fields"]["5"]
        names = net["fields"]["8"]
        outputs = net["fields"]["7"]
        defined: set[int] = set()
        for op in ops:
            fields = op["fields"]
            inputs = fields.get("5") or []
            op_outputs = fields.get("4") or []
            if not op_outputs:
                raise BadWire("op_without_output")
            if any(not 0 <= item < len(names) for item in inputs + op_outputs):
                raise BadWire("tensor_index")
            if not set(inputs) <= defined:
                raise BadWire("use_before_definition")
            if set(op_outputs) & defined:
                raise BadWire("tensor_redefinition")
            defined.update(op_outputs)
        if defined != set(range(len(names))) or any(name not in names for name in outputs):
            raise BadWire("tensor_partition")
        return net


class Reader:
    def __init__(self, raw: bytes):
        self.raw = raw

    def u(self, position: int, fmt: str = "i"):
        return struct.unpack_from("<" + fmt, self.raw, position)[0]

    def field(self, position: int, index: int) -> int | None:
        vtable = position - self.u(position)
        size = self.u(vtable, "H")
        offset = self.u(vtable + 4 + 2 * index, "H") if 4 + 2 * index < size else 0
        return position + offset if offset else None

    def pointer(self, position: int, index: int) -> int | None:
        field_position = self.field(position, index)
        return field_position + self.u(field_position, "I") if field_position is not None else None

    def value(self, position: int, index: int, fmt: str = "i", default=0):
        field_position = self.field(position, index)
        return self.u(field_position, fmt) if field_position is not None else default

    def vector(self, position: int, index: int, dtype: str):
        vector_position = self.pointer(position, index)
        if vector_position is None:
            return None
        return np.frombuffer(
            self.raw,
            dtype=dtype,
            count=self.u(vector_position, "I"),
            offset=vector_position + 4,
        ).copy()


def lower_model(raw: bytes, weights: np.lib.npyio.NpzFile) -> tuple[onnx.ModelProto, dict]:
    net = V(raw).run()
    ops = net["fields"]["5"]
    names = net["fields"]["8"]
    formal = net["fields"]["7"]
    if len(ops) != 73:
        raise ValueError(f"unexpected op count: {len(ops)}")
    expected_types = {"4": 1, "79": 4, "14": 1, "48": 29, "84": 9, "64": 9, "27": 4, "103": 6, "17": 4, "75": 5, "53": 1}
    types = Counter(str(op["fields"]["0"]) for op in ops)
    if dict(types) != expected_types:
        raise ValueError(f"unexpected private op types: {dict(types)}")
    op70 = ops[70]["fields"]
    if op70["0"] != 53 or op70.get("5") != [72] or op70.get("4") != [15] or names[15] != RESPONSE_NAME:
        raise ValueError("op70/tensor15 identity drift")

    reader = Reader(raw)
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    constants: dict[int, np.ndarray] = {}
    shapes: dict[int, tuple[int, ...]] = {}
    dtypes: dict[int, int] = {}
    mapping: dict[int, dict] = {}

    def tensor_name(index: int) -> str:
        return f"t{index}"

    def add_tensor(name: str, array: np.ndarray) -> str:
        initializers.append(N.from_array(np.asarray(array), name))
        return name

    def emit(kind: str, inputs: list[str], output: str | None = None, **attrs) -> str:
        destination = output or tensor_name(op_outputs[0])
        nodes.append(H.make_node(kind, inputs, [destination], name=f"op{op_index:04d}_{len(nodes)}_{kind}", **attrs))
        return destination

    def constant(index: int) -> np.ndarray:
        if index not in constants:
            raise ValueError(f"op{op_index}: tensor {index} is not constant")
        return constants[index]

    def integer_tensor(values, dtype=np.int64) -> str:
        return add_tensor(f"k{op_index}_{len(initializers)}", np.asarray(values, dtype=dtype))

    input_name = ""
    input_shape: tuple[int, ...] = ()
    for op_index, op in enumerate(ops):
        fields = op["fields"]
        op_type = fields["0"]
        parameter = (fields.get("3") or {}).get("table_offset")
        input_ids = fields.get("5", [])
        op_outputs = fields["4"]
        inputs = [tensor_name(index) for index in input_ids]
        input_shapes = [shapes.get(index) for index in input_ids]
        value = None
        output_shape = None
        dtype = dtypes.get(input_ids[0], T.FLOAT) if input_ids else T.FLOAT

        if op_type == 4:
            output_shape = tuple(int(dim) for dim in reader.vector(parameter, 2, "i4"))
            input_name = tensor_name(op_outputs[0])
            input_shape = output_shape
            dtype = T.FLOAT
        elif op_type == 79:
            code = reader.value(parameter, 2)
            output_shape = tuple(int(dim) for dim in reader.vector(parameter, 0, "i4")) if reader.pointer(parameter, 0) is not None else ()
            if code == 3:
                value = reader.vector(parameter, 6, "<i4").astype(np.int64).reshape(output_shape)
                dtype = T.INT64
            else:
                value = reader.vector(parameter, 4, "<f4").astype(np.float32).reshape(output_shape)
                dtype = T.FLOAT
        elif op_type in (48, 88):
            common = reader.pointer(parameter, 6)
            kx, ky, sx, sy, dx, dy, px, py = [reader.value(common, index) for index in range(8)]
            group = reader.value(common, 9)
            output_channels, input_channels = reader.value(common, 10), reader.value(common, 11)
            pad_mode = reader.value(common, 14, "B")
            pads = reader.vector(common, 8, "i4").tolist() if reader.pointer(common, 8) is not None else [py, px] * 2
            weight = weights[f"op{op_index}_weight"].reshape(output_channels, input_channels // group, ky, kx)
            bias = reader.vector(parameter, 0, "<f4")
            attrs = {"strides": [sy, sx], "dilations": [dy, dx], "group": group}
            if pad_mode == 1:
                attrs["auto_pad"] = "SAME_UPPER"
                output_shape = (
                    input_shapes[0][0],
                    output_channels,
                    math.ceil(input_shapes[0][2] / sy),
                    math.ceil(input_shapes[0][3] / sx),
                )
            else:
                if pad_mode == 2:
                    pads = [0, 0, 0, 0]
                attrs["pads"] = pads
                output_shape = (
                    input_shapes[0][0],
                    output_channels,
                    (input_shapes[0][2] + pads[0] + pads[2] - dy * (ky - 1) - 1) // sy + 1,
                    (input_shapes[0][3] + pads[1] + pads[3] - dx * (kx - 1) - 1) // sx + 1,
                )
            relu, relu6 = reader.value(common, 12, "B"), reader.value(common, 13, "B")
            destination = tensor_name(op_outputs[0]) + "_conv" if relu or relu6 else None
            emit("Conv", inputs + [add_tensor(f"w{op_index}", weight), add_tensor(f"b{op_index}", bias)], destination, **attrs)
            if relu:
                nodes.append(H.make_node("Relu", [destination], [tensor_name(op_outputs[0])], name=f"op{op_index:04d}_relu"))
            elif relu6:
                emit(
                    "Clip",
                    [
                        destination,
                        add_tensor(f"lo{op_index}", np.array(0, np.float32)),
                        add_tensor(f"hi{op_index}", np.array(6, np.float32)),
                    ],
                )
        elif op_type == 17:
            common = reader.pointer(parameter, 6)
            kx, ky, sx, sy, dx, dy, px, py = [reader.value(common, index) for index in range(8)]
            group = reader.value(common, 9)
            output_channels, input_channels = reader.value(common, 10), reader.value(common, 11)
            pad_mode = reader.value(common, 14, "B")
            pads = reader.vector(common, 8, "i4").tolist() if reader.pointer(common, 8) is not None else [py, px] * 2
            weight = weights[f"op{op_index}_weight"].reshape(input_channels, output_channels // group, ky, kx)
            bias = reader.vector(parameter, 0, "<f4")
            attrs = {"strides": [sy, sx], "dilations": [dy, dx], "group": group}
            if pad_mode == 1:
                attrs["auto_pad"] = "SAME_UPPER"
            else:
                if pad_mode == 2:
                    pads = [0, 0, 0, 0]
                attrs["pads"] = pads
            output_shape = (
                input_shapes[0][0],
                output_channels,
                (input_shapes[0][2] - 1) * sy + dy * (ky - 1) + 1 - pads[0] - pads[2],
                (input_shapes[0][3] - 1) * sx + dx * (kx - 1) + 1 - pads[1] - pads[3],
            )
            relu = reader.value(common, 12, "B")
            destination = tensor_name(op_outputs[0]) + "_deconv" if relu else None
            emit("ConvTranspose", inputs + [add_tensor(f"w{op_index}", weight), add_tensor(f"b{op_index}", bias)], destination, **attrs)
            if relu:
                nodes.append(H.make_node("Relu", [destination], [tensor_name(op_outputs[0])], name=f"op{op_index:04d}_relu"))
        elif op_type == 84:
            slope = float(reader.value(parameter, 0, "f"))
            output_shape = input_shapes[0]
            emit("Relu" if slope == 0.0 else "LeakyRelu", inputs, alpha=slope if slope else None)
            mapping.setdefault(op_index, {}).update(relu_slope=slope)
        elif op_type == 64:
            activation = reader.value(parameter, 0)
            code = reader.value(parameter, 1)
            kind = {0: "Add", 1: "Sub", 2: "Mul", 7: "Div", 19: "Mod"}[code]
            output_shape = np.broadcast_shapes(*input_shapes)
            if all(index in constants for index in input_ids):
                value = {
                    0: np.add,
                    1: np.subtract,
                    2: np.multiply,
                    7: np.divide,
                    19: np.mod,
                }[code](*[constants[index] for index in input_ids])
                if activation:
                    value = np.maximum(value, 0)
            else:
                destination = tensor_name(op_outputs[0]) + "_binary" if activation else None
                emit(kind, inputs, destination)
                if activation:
                    if activation != 1:
                        raise ValueError(f"op{op_index}: unsupported binary activation {activation}")
                    nodes.append(H.make_node("Relu", [destination], [tensor_name(op_outputs[0])], name=f"op{op_index:04d}_relu"))
            mapping.setdefault(op_index, {}).update(binary_code=code, activation=activation)
        elif op_type == 75:
            axis = reader.value(parameter, 0)
            output_shape = list(input_shapes[0])
            output_shape[axis] = sum(shape[axis] for shape in input_shapes)
            output_shape = tuple(output_shape)
            emit("Concat", inputs, axis=axis)
            mapping.setdefault(op_index, {}).update(axis=axis)
        elif op_type == 27:
            kx, ky, sx, sy, px, py = [reader.value(parameter, index) for index in range(6)]
            average = reader.value(parameter, 7, "B") == 1
            output_shape = (
                input_shapes[0][0],
                input_shapes[0][1],
                (input_shapes[0][2] + 2 * py - ky) // sy + 1,
                (input_shapes[0][3] + 2 * px - kx) // sx + 1,
            )
            emit(
                "AveragePool" if average else "MaxPool",
                inputs,
                kernel_shape=[ky, kx],
                strides=[sy, sx],
                pads=[py, px] * 2,
                ceil_mode=0,
            )
            mapping.setdefault(op_index, {}).update(pool="average" if average else "max")
        elif op_type == 14:
            onnx_form = reader.value(parameter, 7) == 1
            if any(reader.value(parameter, index) != 0 for index in [2, 3, 4, 5, 6]):
                raise ValueError(f"op{op_index}: unsupported slice attrs")
            starts = constant(input_ids[1]).reshape(-1).astype(np.int64)
            ends = constant(input_ids[2]).reshape(-1).astype(np.int64)
            axes = constant(input_ids[3]).reshape(-1).astype(np.int64) if onnx_form else np.arange(len(starts), dtype=np.int64)
            steps = (
                constant(input_ids[4]).reshape(-1).astype(np.int64)
                if onnx_form and len(input_ids) > 4
                else (np.ones(len(starts), dtype=np.int64) if onnx_form else constant(input_ids[3]).reshape(-1).astype(np.int64))
            )
            slices = [slice(None)] * len(input_shapes[0])
            for axis, start, end, step in zip(axes, starts, ends, steps):
                slices[axis] = slice(int(start), int(end), int(step))
            output_shape = np.empty(input_shapes[0], np.uint8)[tuple(slices)].shape
            if input_ids[0] in constants:
                value = constants[input_ids[0]][tuple(slices)]
            else:
                emit("Slice", [inputs[0], integer_tensor(starts), integer_tensor(ends), integer_tensor(axes), integer_tensor(steps)])
        elif op_type == 53:
            output_shape = input_shapes[0]
            emit("Sigmoid", inputs)
            mapping.setdefault(op_index, {}).update(semantic="Sigmoid")
        elif op_type == 103:
            resize_type, coordinate_type = reader.value(parameter, 5), reader.value(parameter, 11)
            mode = {1: "nearest", 2: "linear", 3: "cubic"}[resize_type]
            coordinate = {1: "align_corners", 2: "half_pixel", 4: "asymmetric"}[coordinate_type]
            parameter_values = constant(input_ids[1]).reshape(-1)
            roi = add_tensor(f"roi{op_index}", np.zeros(0, np.float32))
            if parameter_values.dtype.kind in "iu":
                sizes = parameter_values.astype(np.int64)
                if sizes.size != len(input_shapes[0]):
                    raise ValueError(f"op{op_index}: resize rank")
                emit(
                    "Resize",
                    [inputs[0], roi, add_tensor(f"sz{op_index}", sizes)],
                    mode=mode,
                    coordinate_transformation_mode=coordinate,
                    nearest_mode="floor",
                )
                output_shape = tuple(int(size) for size in sizes)
            else:
                scales = parameter_values.astype(np.float32)
                if scales.size != len(input_shapes[0]):
                    raise ValueError(f"op{op_index}: resize rank")
                emit(
                    "Resize",
                    [inputs[0], roi, add_tensor(f"scales{op_index}", scales)],
                    mode=mode,
                    coordinate_transformation_mode=coordinate,
                    nearest_mode="floor",
                )
                output_shape = tuple(int(round(dim * scale)) for dim, scale in zip(input_shapes[0], scales))
            mapping.setdefault(op_index, {}).update(resize_type=resize_type, coordinate_type=coordinate_type, mode=mode, coordinate=coordinate)
        else:
            raise ValueError(f"op{op_index}: unsupported private type {op_type}")

        if value is not None:
            if value.dtype.kind in "iu":
                value = value.astype(np.int64)
                dtype = T.INT64
            else:
                value = value.astype(np.float32)
                dtype = T.FLOAT
            constants[op_outputs[0]] = value
            add_tensor(tensor_name(op_outputs[0]), value)
        if output_shape is None:
            raise ValueError(f"op{op_index}: missing output shape")
        shapes[op_outputs[0]] = tuple(int(dim) for dim in output_shape)
        dtypes[op_outputs[0]] = dtype
        mapping.setdefault(op_index, {}).update(
            private_type=op_type,
            inputs=list(input_ids),
            outputs=list(op_outputs),
            shape=list(shapes[op_outputs[0]]),
            constant=value is not None,
        )

    output_ids = [names.index(name) for name in formal]
    for index, name in zip(output_ids, formal):
        nodes.append(H.make_node("Identity", [tensor_name(index)], [name], name=f"formal_{index}"))
    used = {name for node in nodes for name in node.input}
    initializers = [initializer for initializer in initializers if initializer.name in used]
    graph = H.make_graph(
        nodes,
        "eyebag_814444604",
        [H.make_tensor_value_info(input_name, T.FLOAT, input_shape)],
        [H.make_tensor_value_info(name, dtypes[index], shapes[index]) for index, name in zip(output_ids, formal)],
        initializers,
    )
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", 13)], ir_version=9)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model, {
        "private_type_counts": dict(sorted(types.items(), key=lambda item: int(item[0]))),
        "input_name": input_name,
        "input_shape": list(input_shape),
        "formal_outputs": {name: list(shapes[index]) for index, name in zip(output_ids, formal)},
        "op70": {
            "index": 70,
            "private_type": 53,
            "semantic": "Sigmoid",
            "inputs": [72],
            "outputs": [15],
            "tensor_15_name": names[15],
            "shape": list(shapes[15]),
        },
        "mapping": [mapping[index] for index in sorted(mapping)],
    }


def run_golden(model: onnx.ModelProto, root: Path) -> dict:
    input_data = np.load(root / "assets/input_real_384x512.npy")
    golden_data = np.load(root / "assets/forward_outputs.npz")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, {session.get_inputs()[0].name: input_data})
    result = {}
    exact = True
    for name, output in zip(output_names, outputs):
        golden = golden_data[name]
        difference = np.abs(output - golden)
        same = bool(np.array_equal(output, golden))
        exact = exact and same
        result[name] = {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "min": float(output.min()),
            "max": float(output.max()),
            "mean": float(output.mean()),
            "sha256": sha256(output.tobytes()),
            "golden_max_abs": float(difference.max()),
            "golden_mean_abs": float(difference.mean()),
            "golden_bit_exact": same,
        }
    if set(output_names) != {RESPONSE_NAME, COLOR_NAME}:
        raise ValueError(f"unexpected outputs: {output_names}")
    if result[RESPONSE_NAME]["shape"] != RESPONSE_SHAPE or result[COLOR_NAME]["shape"] != COLOR_SHAPE:
        raise ValueError("formal output shape drift")
    return {
        "backend": f"onnxruntime {ort.__version__}/CPUExecutionProvider",
        "providers": session.get_providers(),
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "graph_optimization": "ORT_DISABLE_ALL",
        "outputs": result,
        "golden_bit_exact": exact,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("AURORA_PF90116_ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT)))
    parser.add_argument("--save-onnx", type=Path)
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    private_fb = pinned_file(root / "assets/814444604.private.fb", PRIVATE_FB_SHA256)
    pinned_file(root / "assets/814444604.native-weights.npz", WEIGHTS_SHA256)
    pinned_file(root / "assets/input_real_384x512.npy", INPUT_SHA256)
    pinned_file(root / "assets/forward_outputs.npz", GOLDEN_SHA256)
    with np.load(root / "assets/814444604.native-weights.npz") as weights:
        model, graph = lower_model(private_fb, weights)
    serialized = model.SerializeToString()
    model_sha = sha256(serialized)
    if model_sha != ONNX_SHA256:
        raise ValueError(f"rebuilt ONNX SHA256 mismatch: {model_sha} != {ONNX_SHA256}")
    if args.save_onnx:
        args.save_onnx.write_bytes(serialized)
    report = {
        "schema": "aurora/pf90116-forward-adapter/v1",
        "model_id": MODEL_ID,
        "artifact_root": str(root),
        "source": {
            "private_fb_sha256": PRIVATE_FB_SHA256,
            "weights_sha256": WEIGHTS_SHA256,
            "input_sha256": INPUT_SHA256,
            "golden_sha256": GOLDEN_SHA256,
            "rebuilt_onnx_sha256": model_sha,
            "onnx_sha256_matches_sealed": model_sha == ONNX_SHA256,
        },
        "graph": graph,
        "conversion": {
            "onnx_opset": 13,
            "onnx_nodes": len(model.graph.node),
            "onnx_initializers": len(model.graph.initializer),
            "checker_full": True,
            "strict_shape_inference": True,
        },
        "preprocessing": {
            "status": "DECLARED_NOT_VENDOR_VERIFIED",
            "definition": "same-job ROI -> bilinear 512x384 -> RGB/255 -> NCHW float32",
        },
        "postprocessing": {
            "status": "NOT_INCLUDED",
            "op70": "Sigmoid produces response 3204548346 in [0,1]",
            "remaining": "threshold/filter/blend/affine paste and final pixel composition are not proven",
        },
        "color_domain": {
            "name": COLOR_NAME,
            "status": "RAW_MODEL_UNITS",
            "not_source_bytes": True,
            "observed_golden_range": [-0.5813234448432922, 1.2192578315734863],
            "transform": "none",
        },
    }
    if not args.skip_forward:
        forward = run_golden(model, root)
        report["forward"] = forward
        if not forward["golden_bit_exact"]:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
