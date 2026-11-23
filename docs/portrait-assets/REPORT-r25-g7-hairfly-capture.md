# R25 G7 HairFly / PF21026 same-frame capture

Status: `BLOCKED_NO_READ_ONLY_CAPTURE_CHANNEL`

Baseline worktree: `6b76bd7` (`eececfd` + G6). Changes are limited to this
worktree. No main-tree write, Cargo command, GPU run, `target/` use,
repository-wide scan, Capture One injection, or Capture One modification was
performed.

## Result

G6's fixed PE evidence was used as the only address authority:

- fixed PE SHA-256:
  `a84519da489bc52e4e914af7880ede43e3ce58992657723d02197161832464c7`;
- producer `0x823A3B0`: output 3 in lookup slot `0x350`, output 4 in `0x4B0`;
- consumer `0x82AA2E0`: its own local Mat at `[rbp+0x4A8]`;
- output 3 has no located native-u8 consumer conversion, so only output 4 can
  reach the current typed path.

No same-job live capture is available. `tools/hairfly_pf21026_capture.py
doctor` reports `debugger_channel=ABSENT`: neither `cdb.exe` nor
`windbg.exe` exists on `PATH` or the standard Windows SDK debugger location.
G6 also stops at function RVAs. It does not provide post-prologue PCs,
the job-id location, draw-buffer storage, the `2405607714` data pointer/shape,
or the local Mat header/data pointer. Attaching or guessing those locations
would violate the read-only requirement and could pause or disturb Capture One.

## Capture ingress

`tools/hairfly_pf21026_capture.py` is a read-only verifier for a debugger or
vendor-export bundle. It does not attach to Capture One.

```powershell
python tools/hairfly_pf21026_capture.py doctor
python tools/hairfly_pf21026_capture.py verify --bundle <bundle> --out <typed-receipts.json>
python tools/hairfly_pf21026_capture.py selftest
```

The manifest requires explicit job/face IDs from producer and consumer, source
SHA, model ID and dimensions, explicit `ch0|ch1|ch2`, output 4, model/frame
ROIs, affine warp, vertex count/stride/SHA, and row/channel strides for the
three-channel f32 mask, repair output, local Mat, and optional raw mask.
Canonical row stripping makes SHA independent of padding while preserving each
reported stride. The tool refuses non-finite masks, hash gaps, ROI overflow,
wrong RVAs/slots, and any byte difference between output 4 and the local Mat.

On a complete bundle it emits field-compatible values for G6's
`HairFlyMaskDecodeReceipt` (`2405607714` plane/raw-mask SHA, draw buffer) and
`Pf21026RepairConsumerReceipt` (job, output 4, RVAs, slot `0x4B0`, offset
`0x4A8`, Mat SHA). No incomplete capture is connected to the product typed
path. If the native border-feather/min/ROI-warp/u8 raw mask is absent, the
result is `CAPTURE_VERIFIED_TYPED_BLOCKED` with G6's exact
`PF90088_RAW_MASK_POSTPROCESS_RECEIPT`; output 3 remains
`PF21026_REPAIR_OUTPUT3_NATIVE_U8_CONSUMER`.

## Exact blocker

Obtain one of:

1. a read-only debugger transport with exact post-prologue producer/consumer
   PCs and runtime locations for job, draw buffer, `2405607714` pointer/shape,
   and the local Mat header/data; or
2. a vendor same-job export containing the bundle fields above.

Then run `verify`. A successful `CAPTURE_VERIFIED_TYPED_READY` bundle can feed
the existing G6 receipts without defaulting a channel or repair output.

## Verification

`python -m py_compile tools/hairfly_pf21026_capture.py`,
`python tools/hairfly_pf21026_capture.py selftest` (`SELFTEST=PASS`),
`doctor`, and `git diff --check` passed. Compile, GPU, and pixel parity are not
claimed. Machine-readable status is in
`docs/portrait-assets/r25-g7-hairfly-capture-blocker.json`.
