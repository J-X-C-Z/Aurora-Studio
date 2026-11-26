#!/usr/bin/env python3
"""Offline EV6 image parity comparator.

Compares a reference/candidate pair per `PF<number>|<backend>` key, preserves
8/16-bit samples and channel order, reports whole-frame and ROI
inside/outside statistics, and writes bounded 8x nearest-neighbour heatmaps.

Run `python ev6_compare.py --help`; see README.md for the roster format.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

try:  # optional; only needed for 16-bit RGB/RGBA PNG
    import cv2 as _cv2
except Exception:  # pragma: no cover - exercised by the no-cv2 unit test
    _cv2 = None

cv2 = _cv2
CV2_AVAILABLE = cv2 is not None

TOOL_NAME = "ev6-image-compare"
TOOL_VERSION = "0.2.0"
STATUSES = ("pass", "fail", "missing", "not_collected", "unsupported", "error")

BAND_ROWS = 256
# 8x nearest output for a full frame: source_pixels * 64 * 3 bytes. 350k source
# pixels cap the RGB output at roughly 67 MB; larger frames are tiled.
HEATMAP_FULL_SOURCE_PIXELS = 350_000
HEATMAP_TILE_SOURCE = 512
HEATMAP_MAX_TILES = 256

_MODE_CHANNELS = {
    "L": (1, 8),
    "LA": (2, 8),
    "RGB": (3, 8),
    "RGBA": (4, 8),
    "I;16": (1, 16),
    "I;16B": (1, 16),
    "I;16L": (1, 16),
}


class Ev6Error(Exception):
    """Base for explicit, user-facing failures."""


class UnsupportedImage(Ev6Error):
    """The container/bit depth cannot be compared without an implicit conversion."""


class RosterError(Ev6Error):
    """The expected roster is malformed or ambiguous."""


class EmptyRoiError(Ev6Error):
    """The ROI maps to zero pixels."""


class ComparisonError(Ev6Error):
    """The two images cannot be compared without resizing or converting."""


@dataclass(frozen=True)
class ImageMeta:
    path: Path
    width: int
    height: int
    channels: int
    bit_depth: int
    mode: str


@dataclass
class RosterItem:
    item_id: str
    pf: str
    backend: str
    reference: Path
    candidate: Path
    expected: bool
    roi: list[float] | None = None

    @property
    def key(self) -> str:
        return roster_key(self.pf, self.backend)


def roster_key(pf: str, backend: str) -> str:
    if not isinstance(pf, str) or not isinstance(backend, str):
        raise RosterError("pf and backend must both be strings")
    pf_text = pf.strip().upper()
    backend_text = backend.strip().lower()
    if not re.fullmatch(r"PF[0-9]{1,9}", pf_text):
        raise RosterError("pf must be PF followed by 1 to 9 decimal digits")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", backend_text):
        raise RosterError("backend must be a label of 1 to 32 ASCII letters, digits or hyphens")
    return f"{pf_text}|{backend_text}"


# ---------------------------------------------------------------------------
# Container probing / loading
# ---------------------------------------------------------------------------

def _png_header(path: Path) -> tuple[int, int, int, int] | None:
    with path.open("rb") as handle:
        head = handle.read(33)
    if len(head) < 33 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", head[16:24])
    return width, height, head[24], head[25]


def _tiff_bits_per_sample(path: Path) -> int | None:
    try:
        with Image.open(path) as image:
            tag = image.tag_v2.get(258) if hasattr(image, "tag_v2") else None
    except Exception:
        return None
    if tag is None:
        return None
    return int(tag[0]) if isinstance(tag, (tuple, list)) else int(tag)


def probe_image(path: Path) -> ImageMeta:
    """Read dimensions, channels and bit depth *without* decoding/converting."""
    if not path.is_file():
        raise FileNotFoundError(path)
    header = _png_header(path)
    if header is not None:
        width, height, bit_depth, color_type = header
        if color_type == 3:
            raise UnsupportedImage(
                f"{path.name}: palette PNG (colour type 3) is refused; expand it explicitly "
                "outside this tool if a comparison is intended"
            )
        if bit_depth == 16 and color_type in (2, 6):
            if not CV2_AVAILABLE:
                raise UnsupportedImage(
                    f"{path.name}: 16-bit RGB/RGBA PNG needs OpenCV cv2.IMREAD_UNCHANGED; "
                    "OpenCV is not installed, refusing rather than downcasting"
                )
            channels = 3 if color_type == 2 else 4
            return ImageMeta(path, width, height, channels, 16, "cv2:BGR16" if channels == 3 else "cv2:BGRA16")
        if color_type in (0, 4) and bit_depth == 16:
            if color_type == 4:
                raise UnsupportedImage(
                    f"{path.name}: 16-bit gray+alpha PNG has no lossless decoder path; refusing"
                )
            return ImageMeta(path, width, height, 1, 16, "I;16")
        if bit_depth != 8:
            raise UnsupportedImage(f"{path.name}: PNG bit depth {bit_depth} is not supported")
        channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
        return ImageMeta(path, width, height, channels, 8, {1: "L", 3: "RGB", 2: "LA", 4: "RGBA"}[channels])

    bits = _tiff_bits_per_sample(path)
    with Image.open(path) as image:
        mode = image.mode
        width, height = image.size
    if bits is not None and bits > 8 and mode not in ("I;16", "I;16B", "I;16L"):
        raise UnsupportedImage(
            f"{path.name}: {bits}-bit TIFF decoded as mode {mode!r}; refusing an implicit downcast"
        )
    if mode not in _MODE_CHANNELS:
        raise UnsupportedImage(
            f"{path.name}: Pillow mode {mode!r} is not supported; supported modes are "
            f"{sorted(_MODE_CHANNELS)}"
        )
    channels, depth = _MODE_CHANNELS[mode]
    return ImageMeta(path, width, height, channels, depth, mode)


def load_image(path: Path) -> tuple[np.ndarray, ImageMeta]:
    """Load samples preserving bit depth and channels; never resize or convert."""
    meta = probe_image(path)
    if meta.mode.startswith("cv2:"):
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise UnsupportedImage(f"{path.name}: OpenCV could not decode the file")
        if array.dtype != np.uint16:
            raise UnsupportedImage(
                f"{path.name}: OpenCV returned {array.dtype}; expected uint16, refusing"
            )
        if array.ndim != 3 or array.shape[:2] != (meta.height, meta.width) or array.shape[2] != meta.channels:
            raise UnsupportedImage(
                f"{path.name}: OpenCV shape {array.shape} != probed "
                f"{(meta.height, meta.width, meta.channels)}"
            )
        if meta.channels == 3:
            array = np.ascontiguousarray(array[:, :, ::-1])  # BGR -> RGB
        else:
            array = np.ascontiguousarray(array[:, :, [2, 1, 0, 3]])  # BGRA -> RGBA
        return array, meta

    with Image.open(path) as image:
        if image.mode != meta.mode:
            raise UnsupportedImage(f"{path.name}: mode changed while loading ({image.mode!r})")
        array = np.array(image)
    if array.ndim == 2:
        array = array[:, :, None]
    expected = (meta.height, meta.width, meta.channels)
    if array.shape != expected:
        raise UnsupportedImage(f"{path.name}: decoded shape {array.shape} != probed {expected}")
    if array.dtype not in (np.uint8, np.uint16):
        raise UnsupportedImage(f"{path.name}: decoded dtype {array.dtype} is not 8/16-bit")
    expected_dtype = np.uint16 if meta.bit_depth == 16 else np.uint8
    if array.dtype != expected_dtype:
        raise UnsupportedImage(
            f"{path.name}: probed {meta.bit_depth}-bit but decoded {array.dtype}; refusing"
        )
    return array, meta


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------

def load_roster(path: Path) -> list[RosterItem]:
    if not path.is_file():
        raise RosterError(f"roster not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RosterError(f"roster is not valid JSON: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise RosterError("roster must be an object with an 'items' array")
    base = path.parent
    items: list[RosterItem] = []
    for index, raw in enumerate(payload["items"]):
        if not isinstance(raw, dict):
            raise RosterError(f"items[{index}] is not an object")
        for field_name in ("pf", "backend", "reference", "candidate", "expected"):
            if field_name not in raw:
                raise RosterError(f"items[{index}] is missing '{field_name}'")
        if not isinstance(raw["expected"], bool):
            raise RosterError(f"items[{index}].expected must be true or false")
        pf, backend = raw["pf"], raw["backend"]
        key = roster_key(pf, backend)
        item_id = str(raw.get("id") or key)
        roi = raw.get("roi")
        if roi is not None and (not isinstance(roi, list) or len(roi) != 4):
            raise RosterError(f"items[{index}].roi must be [x0, y0, x1, y1] when present")
        items.append(
            RosterItem(
                item_id=item_id,
                pf=str(pf),
                backend=str(backend),
                reference=(base / str(raw["reference"])).resolve(),
                candidate=(base / str(raw["candidate"])).resolve(),
                expected=raw["expected"],
                roi=roi,
            )
        )
    seen_ids: dict[str, int] = {}
    seen_keys: dict[str, int] = {}
    duplicates: list[str] = []
    for index, item in enumerate(items):
        if item.item_id in seen_ids:
            duplicates.append(f"duplicate id {item.item_id!r} at items[{index}]")
        if item.key in seen_keys:
            duplicates.append(f"duplicate key {item.key!r} at items[{index}]")
        seen_ids.setdefault(item.item_id, index)
        seen_keys.setdefault(item.key, index)
    if duplicates:
        raise RosterError("; ".join(duplicates))
    return items


# ---------------------------------------------------------------------------
# ROI and banded statistics
# ---------------------------------------------------------------------------

def parse_roi(roi: Iterable[float] | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Map a normalised ``[x0,y0,x1,y1]`` ROI to a half-open pixel rectangle."""
    if roi is None:
        return None
    values = list(roi)
    if len(values) != 4:
        raise EmptyRoiError("ROI must be [x0, y0, x1, y1]")
    try:
        x0, y0, x1, y1 = (float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise EmptyRoiError(f"ROI values must be numbers: {roi!r}") from error
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise EmptyRoiError("ROI values must be finite")
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise EmptyRoiError(
            f"ROI must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, got {roi!r}"
        )
    left = max(0, int(math.floor(x0 * width)))
    top = max(0, int(math.floor(y0 * height)))
    right = min(width, int(math.ceil(x1 * width)))
    bottom = min(height, int(math.ceil(y1 * height)))
    if right <= left or bottom <= top:
        raise EmptyRoiError(f"ROI {roi!r} maps to zero pixels on {width}x{height}")
    return left, top, right, bottom


def _bin_edges(maximum: int) -> list[int]:
    return [int(round(maximum * index / 16.0)) for index in range(17)]


class _RegionAccumulator:
    """Per-channel integer accumulators; one row band is processed at a time."""

    def __init__(self, channels: int, maximum: int) -> None:
        self.edges = _bin_edges(maximum)
        self.channels = [
            {
                "max_abs": 0,
                "sum_signed": 0,
                "sum_abs": 0,
                "sum_sq": 0,
                "ref_sum": 0,
                "n": 0,
                "hist_abs": [0] * 17,
                "ref_hist": [0] * 16,
            }
            for _ in range(channels)
        ]

    def add(self, diff_band: np.ndarray, ref_band: np.ndarray, mask: np.ndarray | None) -> None:
        for index, state in enumerate(self.channels):
            channel_diff = diff_band[..., index]
            channel_ref = ref_band[..., index]
            if mask is not None:
                channel_diff = channel_diff[mask]
                channel_ref = channel_ref[mask]
            if channel_diff.size == 0:
                continue
            absolute = np.abs(channel_diff)
            state["max_abs"] = max(state["max_abs"], int(absolute.max()))
            state["sum_signed"] += int(channel_diff.sum(dtype=np.int64))
            state["sum_abs"] += int(absolute.sum(dtype=np.int64))
            state["sum_sq"] += int((channel_diff.astype(np.int64) ** 2).sum(dtype=np.int64))
            state["ref_sum"] += int(channel_ref.sum(dtype=np.int64))
            state["n"] += int(channel_diff.size)
            clipped = np.bincount(np.minimum(absolute, 16).ravel(), minlength=17)
            for bucket, count in enumerate(clipped[:17]):
                state["hist_abs"][bucket] += int(count)
            ref_counts, _ = np.histogram(channel_ref, bins=self.edges)
            for bucket, count in enumerate(ref_counts):
                state["ref_hist"][bucket] += int(count)

    def finish(self) -> dict[str, Any]:
        channels = []
        for state in self.channels:
            count = state["n"] or 1
            mean = state["sum_signed"] / count
            variance = max(0.0, state["sum_sq"] / count - mean * mean)
            channels.append(
                {
                    "name": "",
                    "max_abs": state["max_abs"],
                    "mean_signed": mean,
                    "mean_abs": state["sum_abs"] / count,
                    "variance_signed": variance,
                    "reference_mean": state["ref_sum"] / count,
                    "histogram_abs": {
                        (str(bucket) if bucket < 16 else ">=16"): state["hist_abs"][bucket]
                        for bucket in range(17)
                    },
                    "reference_histogram": state["ref_hist"],
                }
            )
        pixels = self.channels[0]["n"] if self.channels else 0
        return {
            "pixels": pixels,
            "max_abs": max((channel["max_abs"] for channel in channels), default=0),
            "mean_abs": (sum(channel["mean_abs"] for channel in channels) / len(channels)) if channels and pixels else 0.0,
            "variance_signed": (sum(channel["variance_signed"] for channel in channels) / len(channels)) if channels and pixels else 0.0,
            "channels": channels,
        }


def compare_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    channel_names: list[str],
    roi: Iterable[float] | None,
    tolerance: int,
    ignore_alpha: bool,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ComparisonError(
            f"size/channel mismatch: reference {reference.shape} vs candidate {candidate.shape}; "
            "no resize or channel conversion is performed"
        )
    if reference.dtype != candidate.dtype:
        raise ComparisonError(
            f"bit-depth mismatch: {reference.dtype} vs {candidate.dtype}; no conversion is performed"
        )
    if ignore_alpha and reference.shape[2] == 4:
        reference = reference[:, :, :3]
        candidate = candidate[:, :, :3]
        channel_names = channel_names[:3]
    if tolerance < 0:
        raise ComparisonError("tolerance must be >= 0")
    maximum = 65535 if reference.dtype == np.uint16 else 255
    height, width = reference.shape[:2]
    roi_pixels = parse_roi(roi, width, height)
    whole = _RegionAccumulator(len(channel_names), maximum)
    inside = _RegionAccumulator(len(channel_names), maximum) if roi_pixels else None
    outside = _RegionAccumulator(len(channel_names), maximum) if roi_pixels else None
    band = np.empty((min(BAND_ROWS, height), width, len(channel_names)), dtype=np.int32)
    for top in range(0, height, BAND_ROWS):
        bottom = min(top + BAND_ROWS, height)
        rows = bottom - top
        np.subtract(candidate[top:bottom], reference[top:bottom], out=band[:rows], dtype=np.int32)
        whole.add(band[:rows], reference[top:bottom], None)
        if roi_pixels:
            left, roi_top, right, roi_bottom = roi_pixels
            inside_band = np.zeros((rows, width), dtype=bool)
            start = max(top, roi_top) - top
            end = min(bottom, roi_bottom) - top
            if end > start:
                inside_band[start:end, left:right] = True
            inside.add(band[:rows], reference[top:bottom], inside_band)
            outside.add(band[:rows], reference[top:bottom], ~inside_band)
    whole_result = whole.finish()
    inside_result = inside.finish() if inside else None
    outside_result = outside.finish() if outside else None
    for index, name in enumerate(channel_names):
        whole_result["channels"][index]["name"] = name
        if inside_result:
            inside_result["channels"][index]["name"] = name
            outside_result["channels"][index]["name"] = name
    max_abs = whole_result["max_abs"]
    return {
        "width": width,
        "height": height,
        "channels": len(channel_names),
        "dtype": str(reference.dtype),
        "tolerance": tolerance,
        "ignore_alpha": ignore_alpha,
        "roi": list(roi) if roi is not None else None,
        "roi_pixels": list(roi_pixels) if roi_pixels is not None else None,
        "bit_exact": max_abs == 0,
        "status": "pass" if max_abs <= tolerance else "fail",
        "whole": whole_result,
        "roi_inside": inside_result,
        "roi_outside": outside_result,
    }


# ---------------------------------------------------------------------------
# Heatmap: 8x nearest, bounded for large frames
# ---------------------------------------------------------------------------

def heatmap_rgb(absdiff: np.ndarray) -> np.ndarray:
    scaled = np.clip(absdiff.astype(np.float64), 0.0, 255.0) / 255.0
    red = np.clip(3.0 * scaled - 1.5, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(3.0 * scaled - 1.5), 0.0, 1.0)
    blue = np.clip(1.5 - 3.0 * scaled, 0.0, 1.0)
    return np.stack([red, green, blue], axis=2)


def nearest_8x(rgb: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(rgb, 8, axis=0), 8, axis=1)


def save_heatmap(absdiff: np.ndarray, path: Path) -> None:
    rgb = nearest_8x(heatmap_rgb(absdiff))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255.0).round().astype(np.uint8), mode="RGB").save(path)


def _tile_rects(left: int, top: int, right: int, bottom: int, tile: int):
    for y in range(top, bottom, tile):
        for x in range(left, right, tile):
            yield x, y, min(x + tile, right), min(y + tile, bottom)


def write_heatmaps(
    reference: np.ndarray,
    candidate: np.ndarray,
    roi: Iterable[float] | None,
    output_dir: Path,
    key: str,
    ignore_alpha: bool,
) -> tuple[list[dict[str, Any]], bool]:
    """Write 8x heatmaps within a bounded pixel/file budget.

    Small frames: one full-frame image. Large frames: a ROI crop when a ROI was
    supplied, otherwise only the 512x512 source tiles that contain a non-zero
    difference. Every file records its source rectangle explicitly.
    """
    if ignore_alpha and reference.shape[2] == 4:
        reference = reference[:, :, :3]
        candidate = candidate[:, :, :3]
    height, width = reference.shape[:2]
    roi_pixels = parse_roi(roi, width, height)
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(key)
    results: list[dict[str, Any]] = []
    truncated = False

    def emit(left: int, top: int, right: int, bottom: int, name: str) -> None:
        nonlocal truncated
        tile_ref = reference[top:bottom, left:right]
        tile_cand = candidate[top:bottom, left:right]
        diff = tile_cand.astype(np.int32) - tile_ref.astype(np.int32)
        absdiff = np.abs(diff).max(axis=2)
        path = heatmap_dir / name
        save_heatmap(absdiff, path)
        results.append({
            "path": path.relative_to(output_dir).as_posix(),
            "x": left,
            "y": top,
            "width": right - left,
            "height": bottom - top,
            "max_abs": int(absdiff.max()) if absdiff.size else 0,
        })

    frame_pixels = width * height
    if frame_pixels <= HEATMAP_FULL_SOURCE_PIXELS:
        emit(0, 0, width, height, f"{slug}.png")
    elif roi_pixels:
        left, top, right, bottom = roi_pixels
        roi_area = (right - left) * (bottom - top)
        if roi_area <= HEATMAP_FULL_SOURCE_PIXELS:
            emit(left, top, right, bottom, f"{slug}_roi_x{left}_y{top}.png")
        else:
            for x, y, x2, y2 in _tile_rects(left, top, right, bottom, HEATMAP_TILE_SOURCE):
                if len(results) >= HEATMAP_MAX_TILES:
                    truncated = True
                    break
                emit(x, y, x2, y2, f"{slug}_roi_x{x}_y{y}.png")
    else:
        for x, y, x2, y2 in _tile_rects(0, 0, width, height, HEATMAP_TILE_SOURCE):
            if len(results) >= HEATMAP_MAX_TILES:
                truncated = True
                break
            tile_diff = candidate[y:y2, x:x2].astype(np.int32) - reference[y:y2, x:x2].astype(np.int32)
            if not np.any(tile_diff):
                continue
            emit(x, y, x2, y2, f"{slug}_x{x}_y{y}.png")
    return results, truncated


def _slug(text: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in text)


# ---------------------------------------------------------------------------
# Run one roster
# ---------------------------------------------------------------------------

def compare_one(item: RosterItem, output_dir: Path, tolerance: int, ignore_alpha: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": item.item_id,
        "key": item.key,
        "pf": item.pf,
        "backend": item.backend,
        "expected": item.expected,
        "reference": str(item.reference),
        "candidate": str(item.candidate),
    }
    missing = [str(path) for path in (item.reference, item.candidate) if not path.is_file()]
    if missing:
        record.update(status="missing", reason=f"missing file(s): {', '.join(missing)}")
        return record
    try:
        reference, reference_meta = load_image(item.reference)
        candidate, candidate_meta = load_image(item.candidate)
        if reference_meta.channels != candidate_meta.channels:
            raise ComparisonError(
                f"channel mismatch: {reference_meta.channels} vs {candidate_meta.channels}; "
                "no channel conversion is performed"
            )
        names = ["R", "G", "B", "A"][: reference_meta.channels]
        if reference_meta.channels == 1:
            names = ["Y"]
        elif reference_meta.channels == 2:
            names = ["Y", "A"]
        result = compare_arrays(reference, candidate, names, item.roi, tolerance, ignore_alpha)
        heatmaps, truncated = write_heatmaps(reference, candidate, item.roi, output_dir, item.key, ignore_alpha)
        record.update(result)
        record["heatmaps"] = heatmaps
        record["heatmap"] = heatmaps[0]["path"] if heatmaps else None
        record["heatmap_truncated"] = truncated
        record["reference_bit_depth"] = reference_meta.bit_depth
        record["candidate_bit_depth"] = candidate_meta.bit_depth
        if record["status"] == "pass":
            record["reason"] = "max_abs <= tolerance"
        else:
            record["reason"] = f"max_abs {record['whole']['max_abs']} > tolerance {tolerance}"
        return record
    except UnsupportedImage as error:
        record.update(status="unsupported", reason=str(error))
        return record
    except (ComparisonError, EmptyRoiError) as error:
        record.update(status="error", reason=str(error))
        return record
    except (OSError, ValueError) as error:
        record.update(status="error", reason=f"image cannot be read: {error}")
        return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [record for record in records if record["expected"]]
    present = [record for record in expected if record["status"] not in ("missing", "not_collected")]
    missing_expected = [record for record in expected if record["status"] == "missing"]
    not_collected = [record for record in records if not record["expected"] and record["status"] == "missing"]
    passed = [record for record in records if record["status"] == "pass"]
    failed = [record for record in records if record["status"] == "fail"]
    unsupported = [record for record in records if record["status"] == "unsupported"]
    errors = [record for record in records if record["status"] == "error"]
    all_expected_pass = bool(expected) and not missing_expected and all(
        record["status"] == "pass" for record in present
    ) and not unsupported and not errors
    return {
        "expected_total": len(expected),
        "expected_collected": len(present),
        "expected_missing": len(missing_expected),
        "not_collected": len(not_collected),
        "pass": len(passed),
        "fail": len(failed),
        "unsupported": len(unsupported),
        "error": len(errors),
        "all_expected_pass": all_expected_pass,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _channel_table(region: dict[str, Any] | None) -> str:
    if not region or not region.get("channels"):
        return "_n/a_"
    rows = ["| ch | max_abs | mean_signed | mean_abs | var |", "|---|---:|---:|---:|---:|"]
    for channel in region["channels"]:
        rows.append(
            "| {name} | {max_abs} | {mean_signed:.4f} | {mean_abs:.4f} | {variance_signed:.4f} |".format(**channel)
        )
    return "\n".join(rows)


def write_report(output_dir: Path, records: list[dict[str, Any]], roster: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    payload = {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "roster": str(roster),
        "key_contract": "PF<number>|<backend>",
        "statuses": list(STATUSES),
        "summary": summary,
        "items": records,
    }
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# EV6 image parity",
        "",
        "Synthetic self-check; **not** a hardware or OEM/vendor parity claim.",
        "",
        "Key contract: `PF<number>|<backend>`. `missing` (not collected) is kept apart from `pass`.",
        "",
        "## Summary",
        "",
        "| expected | collected | missing (expected) | not collected | pass | fail | unsupported | error | all expected pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        f"| {summary['expected_total']} | {summary['expected_collected']} | {summary['expected_missing']} | "
        f"{summary['not_collected']} | {summary['pass']} | {summary['fail']} | {summary['unsupported']} | "
        f"{summary['error']} | {summary['all_expected_pass']} |",
        "",
        "## Items",
        "",
        "| key | expected | status | max_abs | mean_abs | heatmap | reason |",
        "|---|:---:|---|---:|---:|---|---|",
    ]
    for record in records:
        max_abs = record.get("whole", {}).get("max_abs", "-")
        mean_abs = record.get("whole", {}).get("mean_abs", "-")
        mean_text = f"{mean_abs:.4f}" if isinstance(mean_abs, (int, float)) else mean_abs
        heatmaps = record.get("heatmaps") or []
        if heatmaps:
            note = f"{len(heatmaps)} tile(s)" if len(heatmaps) > 1 else "8x"
            link = f"[{note}]({heatmaps[0]['path']})"
            if record.get("heatmap_truncated"):
                link += " (truncated)"
        else:
            link = "-"
        lines.append(
            f"| `{record['key']}` | {record['expected']} | {record['status']} | {max_abs} | {mean_text} | {link} | {record.get('reason', '')} |"
        )
    lines.extend(["", "## Channel detail", ""])
    for record in records:
        if "whole" not in record:
            continue
        lines.extend([f"### `{record['key']}` — whole frame", "", _channel_table(record["whole"]), ""])
        if record.get("roi_inside"):
            lines.extend(["ROI inside:", "", _channel_table(record["roi_inside"]), ""])
        if record.get("roi_outside"):
            lines.extend(["ROI outside:", "", _channel_table(record["roi_outside"]), ""])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    html = [
        "<!doctype html><meta charset='utf-8'><title>EV6 image parity</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px}table{border-collapse:collapse}"
        "td,th{border:1px solid #bbb;padding:4px 8px;font-size:13px}</style>",
        "<h1>EV6 image parity</h1>",
        "<p>Synthetic self-check; not a hardware or OEM/vendor parity claim.</p>",
        "<table><tr><th>key</th><th>expected</th><th>status</th><th>max_abs</th><th>heatmap</th></tr>",
    ]
    for record in records:
        heatmaps = record.get("heatmaps") or []
        image = f"<a href='{heatmaps[0]['path']}'><img src='{heatmaps[0]['path']}' width='320'></a>" if heatmaps else "-"
        html.append(
            f"<tr><td>{record['key']}</td><td>{record['expected']}</td><td>{record['status']}</td>"
            f"<td>{record.get('whole', {}).get('max_abs', '-')}</td><td>{image}</td></tr>"
        )
    html.append("</table>")
    (output_dir / "report.html").write_text("\n".join(html), encoding="utf-8")


def create_output_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise Ev6Error(f"output already exists; choose a new directory: {path}") from error


def run_roster(roster_path: Path, output_dir: Path, tolerance: int, ignore_alpha: bool) -> dict[str, Any]:
    items = load_roster(roster_path)
    create_output_directory(output_dir)
    records = [compare_one(item, output_dir, tolerance, ignore_alpha) for item in items]
    write_report(output_dir, records, roster_path)
    return summarize(records)


# ---------------------------------------------------------------------------
# Synthetic self-test (offline samples)
# ---------------------------------------------------------------------------

def _png16(path: Path, values: np.ndarray, color_type: int) -> None:
    """Write a 16-bit PNG without using the decoder under test."""
    height, width, channels = values.shape
    assert color_type == {3: 2, 4: 6}[channels]
    raw = bytearray()
    for row in values:
        raw.append(0)
        raw.extend(row.astype(">u2").tobytes())

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 16, color_type, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def _png16_rgb(path: Path, values: np.ndarray) -> None:
    _png16(path, values, 2)


def _png16_rgba(path: Path, values: np.ndarray) -> None:
    _png16(path, values, 6)


def _rgb_gradient(width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    red = np.tile(x, (height, 1))
    green = np.tile(y[:, None], (1, width))
    blue = ((red.astype(np.uint16) + green.astype(np.uint16)) // 2).astype(np.uint8)
    return np.stack([red, green, blue], axis=2)


def _rgb16_pattern(width: int, height: int) -> np.ndarray:
    values = np.zeros((height, width, 3), dtype=np.uint16)
    values[..., 0] = 0x1234
    values[..., 1] = 0x2345
    values[..., 2] = 0x3456
    values[0, 0] = [0x0001, 0x8000, 0xFFFF]
    return values


def _rgba16_pattern(width: int, height: int) -> np.ndarray:
    rgb = _rgb16_pattern(width, height)
    alpha = np.full((height, width, 1), 0xFFFE, dtype=np.uint16)
    return np.concatenate([rgb, alpha], axis=2)


def make_samples(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    ref_dir = root / "ref"
    cand_dir = root / "cand"
    ref_dir.mkdir(exist_ok=True)
    cand_dir.mkdir(exist_ok=True)
    base = _rgb_gradient(64, 48)

    def save(directory: Path, name: str, array: np.ndarray) -> None:
        Image.fromarray(array).save(directory / name)

    save(ref_dir, "pass.png", base)
    save(cand_dir, "pass.png", base.copy())

    off_by_one = base.copy()
    off_by_one[10, 12, 0] = np.uint8(min(255, int(off_by_one[10, 12, 0]) + 1))
    save(ref_dir, "off_by_one.png", base)
    save(cand_dir, "off_by_one.png", off_by_one)

    rgba = np.dstack([base, np.full(base.shape[:2], 255, dtype=np.uint8)])
    rgba_candidate = rgba.copy()
    rgba_candidate[5, 5, 3] = 254
    save(ref_dir, "alpha.png", rgba)
    save(cand_dir, "alpha.png", rgba_candidate)

    save(ref_dir, "missing.png", base)
    save(ref_dir, "size.png", base)
    save(cand_dir, "size.png", base[:24, :32])

    roi_ref = base.copy()
    roi_cand = base.copy()
    roi_cand[12:36, 16:48, 1] = np.uint8(min(255, int(roi_cand[12:36, 16:48, 1].max()) + 1))
    save(ref_dir, "roi.png", roi_ref)
    save(cand_dir, "roi.png", roi_cand)

    gray = np.arange(32 * 24, dtype=np.uint16).reshape(24, 32) * 17
    Image.fromarray(gray).save(ref_dir / "gray16.png")
    Image.fromarray(gray.copy()).save(cand_dir / "gray16.png")

    rgb16 = _rgb16_pattern(16, 12)
    _png16_rgb(ref_dir / "rgb16.png", rgb16)
    _png16_rgb(cand_dir / "rgb16.png", rgb16.copy())

    rgba16 = _rgba16_pattern(16, 12)
    rgba16_candidate = rgba16.copy()
    rgba16_candidate[2, 3, 3] = 0x0001
    _png16_rgba(ref_dir / "rgba16.png", rgba16)
    _png16_rgba(cand_dir / "rgba16.png", rgba16_candidate)

    # Deliberate 8-bit vs 16-bit pair: must be an explicit bit-depth error.
    save(ref_dir, "mismatch8.png", base)
    _png16_rgb(cand_dir / "mismatch16.png", _rgb16_pattern(64, 48))

    roster = {
        "items": [
            {"id": "pass", "pf": "PF90002", "backend": "cuda", "reference": "ref/pass.png", "candidate": "cand/pass.png", "expected": True},
            {"id": "off-by-one", "pf": "PF90002", "backend": "metal", "reference": "ref/off_by_one.png", "candidate": "cand/off_by_one.png", "expected": True},
            {"id": "alpha", "pf": "PF108", "backend": "cuda", "reference": "ref/alpha.png", "candidate": "cand/alpha.png", "expected": True},
            {"id": "missing", "pf": "PF90085", "backend": "wgsl", "reference": "ref/missing.png", "candidate": "cand/missing.png", "expected": True},
            {"id": "size", "pf": "PF247", "backend": "cuda", "reference": "ref/size.png", "candidate": "cand/size.png", "expected": True},
            {"id": "roi", "pf": "PF90086", "backend": "cuda", "reference": "ref/roi.png", "candidate": "cand/roi.png", "expected": True, "roi": [0.25, 0.25, 0.75, 0.75]},
            {"id": "gray16", "pf": "PF90085", "backend": "cpu", "reference": "ref/gray16.png", "candidate": "cand/gray16.png", "expected": True},
            {"id": "rgb16", "pf": "PF0", "backend": "cuda", "reference": "ref/rgb16.png", "candidate": "cand/rgb16.png", "expected": True},
            {"id": "rgba16-alpha", "pf": "PF5", "backend": "cuda", "reference": "ref/rgba16.png", "candidate": "cand/rgba16.png", "expected": True},
            {"id": "mismatch-8-16", "pf": "PF101", "backend": "cuda", "reference": "ref/mismatch8.png", "candidate": "cand/mismatch16.png", "expected": True},
            {"id": "not-collected", "pf": "PF102", "backend": "metal", "reference": "ref/not_collected.png", "candidate": "cand/not_collected.png", "expected": False},
        ]
    }
    roster_path = root / "roster.json"
    roster_path.write_text(json.dumps(roster, indent=2), encoding="utf-8")
    return roster_path


def selftest_expected() -> dict[str, Any]:
    if CV2_AVAILABLE:
        return {
            "expected_total": 10, "expected_collected": 9, "expected_missing": 1,
            "not_collected": 1, "pass": 3, "fail": 4, "unsupported": 0, "error": 2,
            "all_expected_pass": False,
        }
    return {
        "expected_total": 10, "expected_collected": 9, "expected_missing": 1,
        "not_collected": 1, "pass": 2, "fail": 3, "unsupported": 3, "error": 1,
        "all_expected_pass": False,
    }


def selftest(root: Path) -> dict[str, Any]:
    target = root / "ev6-selftest"
    create_output_directory(target)
    roster = make_samples(target)
    summary = run_roster(roster, target / "report", tolerance=0, ignore_alpha=False)
    expected = selftest_expected()
    ok = all(summary.get(key) == value for key, value in expected.items())
    summary["selftest_ok"] = ok
    summary["expected_sample_outcome"] = dict(expected)
    print(json.dumps({"samples": str(target), "summary": summary}, indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="compare a roster")
    run_parser.add_argument("--roster", required=True, type=Path)
    run_parser.add_argument("--out", required=True, type=Path)
    run_parser.add_argument("--tolerance", type=int, default=0)
    run_parser.add_argument("--ignore-alpha", action="store_true", help="explicitly compare RGB only (RGBA pairs only)")
    selftest_parser = sub.add_parser("selftest", help="generate offline samples and run them")
    selftest_parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "selftest":
            summary = selftest(args.out)
            return 0 if summary.get("selftest_ok") else 1
        summary = run_roster(args.roster, args.out, args.tolerance, args.ignore_alpha)
        print(json.dumps(summary, indent=2))
        return 0 if summary["all_expected_pass"] else 1
    except RosterError as error:
        print(f"roster error: {error}", file=sys.stderr)
        return 2
    except (Ev6Error, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
