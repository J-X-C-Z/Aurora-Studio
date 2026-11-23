# PF90008 StretchV3 registry recovery (R26)

## Conclusion

`0x1E9DD78` is the processor lookup key shared by
`PixCook::EffectStretchMark` and `PixCook::EffectStretchMarkModel`. The effect
instance names the pipeline `StretchV3Pipeline`; the model class registers the
low-32 resource IDs `0x15F98` and `0x15FB5`.

## Binary anchors

- Image base: `0x140000000`; timestamp: `1791552666`.
- Effect constructor: `0x147CEAB50`.
  - Stores vtable `0x1495BF0F8` (`PixCook::EffectStretchMark`).
  - Stores `StretchV3Pipeline` at object `+0x58`.
  - Stores key `0x1E9DD78` at object `+0xB0`.
- Processor lookup: `0x147CEACB0`, where `r8d = 0x1E9DD78` is passed to
  `0x144FC8CD0`.
- Model constructor: `0x144F1B6F0`.
  - Stores vtable `0x1491C8908` (`PixCook::EffectStretchMarkModel`).
  - Stores the same key at `+0x58`.
  - Registers `0x100015F98` and `0x100015FB5` with variants `0` and
    `0xFFFFFFFE`.
- Callback-table builder: `0x1448525E7`; it registers
  `0x1E9DA58, 0x1E9DE40, 0x1E9DD78, 0x1E9FD80, 0x1EA2A6C, 0x1EA2AD0`.
  Ownership is proven only for `0x1E9DD78`.
- RTTI confirms the class chain
  `EffectStretchMark@PixCook -> StretchV3Pipeline@TSAI`, with
  `EffectStretchMarkModel@PixCook` and `StretchV3Params@TSAI`.

## Payload and CPU status

The installed `effect/config/StretchMark/effect.json` is only 119 bytes and
does not expose readable model references. The `Resources/pixcook` container is
2,071,019,532 bytes and contains none of `StretchV3`, `StretchMark`,
`EffectStretchMark`, `15f98`, or `15fb5` as plaintext. The `.rcc` archives also
contain none of those strings.

Exact blocker: no original StretchV3 tensor/weights or model graph exists in a
readable form. Mapping the key to a model class is complete, but CPU model
forward cannot be attached without inventing weights or an unsupported affine
and blend law. The existing Aurora CPU boundary must continue to refuse.
