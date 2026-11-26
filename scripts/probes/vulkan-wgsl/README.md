# WGSL + Vulkan compute probe

A standalone Rust fixture that takes two small WGSL compute kernels, compiles them to
SPIR-V with [`naga`](https://crates.io/crates/naga) (WGSL stays the single source of truth,
no `glslang`/`dxc` at build time) and submits them to real Vulkan hardware through
[`ash`](https://crates.io/crates/ash). It exists to make two claims checkable on any machine
with a GPU instead of asserted in prose:

| # | Claim | Criterion |
|---|---|---|
| 1 | Two integer RGBA8 stages (A: Q8 per-channel gain + signed offset, B: Q8 contrast about mid-grey 128), each quantized back to 8 bit, fused into **one dispatch** | byte-identical to a CPU reference that runs the stages serially, compared per channel R/G/B/A |
| 2 | The fusion is *provable*, not just equal | the fused path's intermediate buffer is pre-filled with a poison pattern and must still hold it afterwards; the serial path's intermediate must equal the reference stage-A output |
| 3 | A small FP32 weighted matmul, `C = A·B + bias` | `max_abs < 1e-4` against an FP32 CPU reference, plus an FMA reference that isolates driver contraction from a wrong kernel |

This is a **fixture**, not a backend. See [What this is not](#what-this-is-not) before quoting
any number it prints.

## Layout

```
scripts/probes/vulkan-wgsl/
├── Cargo.toml            # own workspace root; ash + naga pinned, Cargo.lock committed
├── Cargo.lock
├── shaders/fusion.wgsl   # entry points: op_a, op_b, op_ab
├── shaders/matmul.wgsl   # entry point:  matmul_bias
└── src/
    ├── main.rs           # device selection, case matrix, JSON reports
    ├── vk.rs             # minimal Vulkan compute wrapper + counted submit/dispatch/readback events
    ├── shader.rs         # WGSL -> SPIR-V through naga
    └── refs.rs           # CPU references (integer byte-exact + f32/fma/f64 matmul) and their tests
```

## Build and run from the repository root

```bash
cargo build --manifest-path scripts/probes/vulkan-wgsl/Cargo.toml
cargo run   --manifest-path scripts/probes/vulkan-wgsl/Cargo.toml -- target/probe-logs run1
```

The first argument is the output directory (`./logs` if omitted), the second is a tag written
into `summary.json`.

`Cargo.toml` declares an empty `[workspace]` table on purpose: this crate sits under
`scripts/`, outside the repository workspace members, so it has to be its own workspace root.
Without that table `cargo` stops with *"current package believes it's in a workspace when it's
not"*. Consequence worth knowing: the repository workspace build never compiles this crate, and
this crate never inherits the workspace's profiles or lints.

## Requirements

- A Vulkan 1.2+ loader and driver with at least one compute queue (`vulkan-1.dll` on Windows,
  `libvulkan.so` on Linux). Devices below 1.2 are still enumerated, but the
  `Vulkan12Features` query is skipped for them.
- Stable Rust (edition 2021). No C toolchain, no shader compiler, no SDK.
- No validation layer is required. The probe does not install one; correctness rests on the
  CPU-reference comparison. If you do have one enabled, expect it to add its own output.

## Environment variables

| Variable | Effect |
|---|---|
| `C_PROBE_VENDORS=0x1002,0x10de` | Measure only these PCI vendor ids. A listed vendor that is absent becomes a recorded failure instead of a silent skip — use this to reproduce a specific hardware matrix. |
| `C_PROBE_ONLY_INVENTORY=1` | Enumerate and print devices, measure nothing. |

Default (no variables): every non-software device the loader reports is measured. Software
adapters (`deviceType == CPU`, e.g. lavapipe) are always enumerated, printed and rejected as
measurement targets — they are never counted as hardware.

## Tests

```bash
cargo test --manifest-path scripts/probes/vulkan-wgsl/Cargo.toml
```

The unit tests are CPU-only and need no GPU: they run the full case matrix through the
reference implementation and assert that the fused and serial references agree, that 0 %
strength is a strict identity, that fully transparent pixels are never touched, that
quantization rounds half-up and clamps at both ends, that the comparison helper catches a
one-LSB difference, and that the f32/FMA/f64 matmul references sit inside the 1e-4 contract.
The hardware half of the claim — GPU output equals those references — can only be checked by
running the binary on a device.

## Reading a run

Exit codes: `0` pass, `1` failure or nothing measured, `2` shader compile error, `3` Vulkan
unavailable (a `partial-*.json` with the error is written).

Artifacts land in the output directory: `device-<vendor>-<device>.json` per measured device,
`inventory.json` for every enumerated device, and `summary.json` for the run.

Submit/dispatch/readback counts are counted events, not inferences from the call graph: a fused
case reports `submits=1 dispatches=1`, a serial case `submits=1 dispatches=2`, and each case
performs exactly one readback (plus one verification read of the intermediate image, outside the
measured window). A case "passes" only if the GPU bytes equal the CPU bytes.

Case matrix per device: 6 sizes (`37x23`, `1x1`, `3x5`, `63x65`, `127x31`, `64x64`) × 3 strength
levels (0 / 50 / 100 %) × 2 modes (fused / serial) = 36 fusion cases, plus 2 matmul shapes
(`16x32x8`, `64x128x32`).

## What this is not

- **Not a production compute backend and not part of the rendering path.** It never links
  `wgpu`; it talks to Vulkan directly, on purpose, so that submit/dispatch/readback accounting
  and memory residency are visible. Do not treat the presence of WGSL here as evidence about
  how the shipping GPU path behaves.
- **Not a benchmark.** `elapsed_us` is one case's begin→submit→fence→readback wall clock on a
  warm device. It is not P50/P95, not a frame budget, and not comparable across APIs or devices.
- **No fusion performance target is measured.** Claim 1 is bit-exact equivalence and dispatch
  accounting, not milliseconds. Any latency budget needs a separate, properly instrumented
  measurement.
- **No external inputs.** Test images and weights are generated in-process from a deterministic
  integer generator, so a run needs no model weights, no image assets and no third-party
  runtime (there is no ONNX Runtime, no vendor SDK, no downloaded data anywhere in this crate).
- **Not a generality claim.** No tiling, no FP16/INT8, no async queues, no descriptor indexing,
  no pipeline cache. Buffers here are kilobytes.

## Reference run

One Windows 11 machine, 2026-11-26, two non-software Vulkan devices (one integrated AMD GPU,
vendor `0x1002`; one discrete NVIDIA GPU, vendor `0x10de`), invoked with
`C_PROBE_VENDORS=0x1002,0x10de`:

```
fusion cases: 72/72 byte-exact vs CPU serial reference (155940 pixels compared per channel R,G,B,A)
fused dispatch accounting: 36 dispatches over 36 fused cases; serial: 72 over 36 cases (2x)
fused intermediate buffer untouched (poison intact): 36/36
CPU fused == CPU serial reference (both baselines agree): 72/72
matmul cases inside 1e-4: 4/4
verdict: PASS
```

Matmul detail, identical on both devices:

```
16x32x8    max_abs(f32 ref)=4.768e-7  max_abs(fma ref)=0.000e0  max_abs(f64 ref)=7.281e-7
64x128x32  max_abs(f32 ref)=2.861e-6  max_abs(fma ref)=0.000e0  max_abs(f64 ref)=4.290e-6
```

Exact agreement with the FMA reference is the useful part: it attributes the `4.8e-7 / 2.9e-6`
gap against the plain f32 reference to the driver contracting `mul+add` into `fma`, rather than
to a kernel that computes the wrong thing. These numbers describe that one run on those two
devices — a different machine will report a different case count, and the pass criteria are
ratios, not these totals.
