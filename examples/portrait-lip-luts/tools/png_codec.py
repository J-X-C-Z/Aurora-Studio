"""Minimal, dependency-free PNG writer/reader for 8-bit RGB/RGBA atlases.

Written for this asset lane so that every byte of the shipped LUT atlases is
produced by code that can be read here, and so that "the file is a valid PNG
with these pixels" can be checked without trusting the encoder that wrote it.

Encoder: 8-bit, colour type 2 (RGB) or 6 (RGBA), no interlace, filter type 0 on
every row, one IDAT. Deterministic: same pixels + same zlib build => same bytes.

Decoder: supports bit depth 8, colour types 0/2/4/6, filter types 0-4, and
multiple IDAT chunks. Anything else is refused instead of guessed.
"""

from __future__ import annotations

import struct
import zlib

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_COLOUR_CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


class PngError(ValueError):
    pass


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_png(path, width: int, height: int, channels: int, pixels: bytes) -> int:
    """Write an 8-bit PNG; returns the file size in bytes."""
    if channels not in (3, 4):
        raise PngError("only 3 or 4 channels are written here")
    expected = width * height * channels
    if len(pixels) != expected:
        raise PngError(f"pixel buffer is {len(pixels)} bytes, expected {expected}")
    colour_type = 2 if channels == 3 else 6
    stride = width * channels
    raw = bytearray()
    for row in range(height):
        raw.append(0)  # filter: None
        start = row * stride
        raw += pixels[start:start + stride]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    body = (
        PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _chunk(b"IEND", b"")
    )
    with open(path, "wb") as handle:
        handle.write(body)
    return len(body)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def read_png(path):
    """Returns (width, height, channels, pixels) for an 8-bit PNG."""
    data = open(path, "rb").read()
    if not data.startswith(PNG_SIGNATURE):
        raise PngError("missing PNG signature")
    pos = len(PNG_SIGNATURE)
    header = None
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != crc:
            raise PngError(f"bad CRC on {kind!r}")
        if kind == b"IHDR":
            width, height, depth, colour, comp, filt, interlace = struct.unpack(">IIBBBBB", payload)
            if depth != 8 or comp != 0 or filt != 0 or interlace != 0:
                raise PngError("only 8-bit non-interlaced PNG is decoded here")
            if colour not in _COLOUR_CHANNELS:
                raise PngError(f"unsupported colour type {colour}")
            header = (width, height, _COLOUR_CHANNELS[colour])
        elif kind == b"IDAT":
            idat += payload
        elif kind == b"IEND":
            break
        pos += 12 + length
    if header is None:
        raise PngError("no IHDR")
    width, height, channels = header
    stride = width * channels
    raw = zlib.decompress(bytes(idat))
    if len(raw) != height * (stride + 1):
        raise PngError("decompressed size does not match the header")
    out = bytearray(height * stride)
    prev = bytearray(stride)
    pos = 0
    for row in range(height):
        filter_type = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        pos += stride
        if filter_type == 0:
            pass
        elif filter_type == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif filter_type == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif filter_type == 3:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                upleft = prev[i - channels] if i >= channels else 0
                line[i] = (line[i] + _paeth(left, prev[i], upleft)) & 0xFF
        else:
            raise PngError(f"unknown filter type {filter_type}")
        out[row * stride:(row + 1) * stride] = line
        prev = line
    return width, height, channels, bytes(out)
