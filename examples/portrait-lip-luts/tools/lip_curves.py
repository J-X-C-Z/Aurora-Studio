"""The two lip colour curves, as plain Python (standard library only).

Both curves are analytic, parameterised only by node coordinates, and are the
only source of the shipped atlas bytes. Nothing is read from disk, no vendor
atlas or image is sampled, and there is no randomness.

Design constraints (all asserted by `generate_luts.py`):

  1. Fixed corners: (0,0,0) -> (0,0,0) and (1,1,1) -> (1,1,1).
  2. Neutral axis: every node with r = g = b stays neutral.
  3. Monotone: each stored channel is non-decreasing along its own axis.
  4. No clipping: the chroma gain is limited to the exact room the RGB cube has,
     so no value leaves [0,1] before quantisation.
  5. Non-identity: the change is visible in the midtones and measured per asset.

Luma is Rec.709. `bump(y) = 4y(1-y)` is a smooth window that is 0 at both ends
and 1 at mid-luma, so every effect fades out exactly where constraints 1 and 2
have to hold.
"""

from __future__ import annotations

LUMA_R = 0.2126
LUMA_G = 0.7152
LUMA_B = 0.0722

#: positive direction: strengthen the lip colour that is already there
LIPS_SATURATION_GAIN = 0.30
#: negative direction: pull the lip back toward its natural tone ...
LIP_DODGE_DESATURATION = 0.35
#: ... and lift it slightly, which is what "dodge" means here
LIP_DODGE_LIFT = 0.05
#: headroom kept from 0.0 and 1.0. Zero: the gain is limited to the exact room
#: the cube has, so the transform never leaves [0,1].
CLIP_MARGIN = 0.0

_EPSILON = 1e-12


def luma(r: float, g: float, b: float) -> float:
    """Rec.709 luma, summed left to right."""
    return LUMA_R * r + LUMA_G * g + LUMA_B * b


def bump(y: float) -> float:
    return 4.0 * y * (1.0 - y)


def bounded_gain(wanted: float, pivot: float, target: float,
                 rgb: tuple[float, float, float]) -> float:
    """Largest gain <= `wanted` keeping every channel inside [margin, 1-margin].

    The output is `target + gain * (channel - pivot)`, so the room on each
    channel is `(1 - margin - target) / delta` above the pivot and
    `(target - margin) / -delta` below it. A negative gain is refused: these
    curves may only scale chroma, never invert it.
    """
    room = float("inf")
    for channel in rgb:
        delta = channel - pivot
        if delta > _EPSILON:
            candidate = (1.0 - CLIP_MARGIN - target) / delta
        elif delta < -_EPSILON:
            candidate = (target - CLIP_MARGIN) / -delta
        else:
            continue
        if candidate < room:
            room = candidate
    gain = wanted if wanted < room else room
    return gain if gain > 0.0 else 0.0


def lips_gain(rgb: tuple[float, float, float]) -> tuple[float, float]:
    """(used, wanted) chroma gain of the positive curve."""
    y = luma(*rgb)
    wanted = 1.0 + LIPS_SATURATION_GAIN * bump(y)
    return bounded_gain(wanted, y, y, rgb), wanted


def lips_transform(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Positive `LipsLUT`: luma-preserving chroma boost, fading out at the ends."""
    y = luma(*rgb)
    gain, _ = lips_gain(rgb)
    return tuple(y + gain * (channel - y) for channel in rgb)


def lip_dodge_gain(rgb: tuple[float, float, float]) -> tuple[float, float]:
    """(used, wanted) chroma gain of the negative curve."""
    y = luma(*rgb)
    wanted = 1.0 - LIP_DODGE_DESATURATION * bump(y)
    target = y + LIP_DODGE_LIFT * bump(y)
    return bounded_gain(wanted, y, target, rgb), wanted


def lip_dodge_transform(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Negative `LipDodgeLUT`: luma-dependent desaturation plus a small lift."""
    y = luma(*rgb)
    window = bump(y)
    gain, _ = lip_dodge_gain(rgb)
    target = y + LIP_DODGE_LIFT * window
    return tuple(target + gain * (channel - y) for channel in rgb)


def node_value(index: int) -> float:
    """Node coordinate for grid index 0..63."""
    return index / 63.0


def to_byte(value: float) -> int:
    """Quantise to 8-bit, ties to even (Python's `round` does that)."""
    scaled = round(value * 255.0)
    if scaled < 0:
        return 0
    if scaled > 255:
        return 255
    return int(scaled)
