# Aurora portrait effect backend (`aurora-portrait`)

`aurora-portrait` contains recovered portrait effect math with explicit inputs.
It has no GPUI dependency. The workspace crates `aurora-engine`
(`crates/engine`) and `portrait-app` (`crates/app`) consume it directly without
duplicating the kernels.

```rust
use aurora_portrait::{
    Image,
    kernels::mask_compose::{apply, MaskComposeMode},
};

let image = Image::solid(1, 1, [100, 150, 200, 77]);
let output = apply(&image, &[128], MaskComposeMode::MaskOnly).unwrap();
assert_eq!(output.pixel(0, 0), [50, 75, 100, 77]);
```

Call a recovered family through its typed module API. `catalog()` lists the
registered parameter flags and required inputs; a registry entry is not an
automatic model, landmark detector, material loader, or UI binding.

`catalog_json()` (or `cargo run -p aurora-portrait --example catalog_json`)
exports the same registry as JSON for a host that configures effects from data.

## Initial recovered stages

- `mask_compose`: shared `0x5B8CEA0` byte law, RGB truncation and original alpha.
  Mask-only zero produces zero RGB; alternate zero preserves RGB.
- `facial_smooth`: local `FacialSmoothProcessV2` for forehead, glabella, and eye
  wrinkle paths (rows 36-38), with ten explicit uniforms, source/base/residual
  images, and three masks. `composite_pixel` preserves float output for GPU
  comparison; `apply` quantizes to RGBA8. Neither supplies the upstream models.
- `neck_repair`: PF90015 model tails and exact coarse max-pool verification.
  The original final blend after `0x5B7B390` is still unresolved. The earlier
  C# source-over alternative is not registered as a recovered vendor effect.

Malformed image lengths, missing masks and invalid uniforms produce typed
errors. Public image fields are validated at kernel boundaries. Test uniforms
are fixtures, not recovered product defaults.

## Validation

```text
cargo test -p aurora-portrait --locked -- --test-threads=1
cargo check --workspace --all-targets --locked
```

On a Windows machine with NVIDIA hardware, require actual CUDA execution when
running the embedded GPU checks rather than accepting an unavailable-driver
skip:

```powershell
$env:PORTRAIT_CUDA_REQUIRED = "1"
cargo test -p aurora-portrait --locked cuda:: -- --test-threads=1
```

CPU tests establish Rust behavior against the supplied recovered fixtures.
Shader source, successful compilation, actual GPU execution, and comparison
against the original program's output are separate evidence levels. In
particular, Windows CPU tests do not establish Apple Metal execution.

No frontend controls or Capture One installation are part of this backend.

## Eye-bag final crop composition

`kernels::eye_bag::compose_field_image_pf90116` accepts an RGBA8 source,
an aligned interleaved float RGB color target, a response plane and degree.
It quantizes the response to a saturated byte with nearest-even rounding,
mixes the float target with the source, then truncates and clamps RGB to bytes.
Source alpha is retained. Degrees at or below `0.001` keep the source.
`BlendMode::Vendor` also reaches this stage through the existing RGB/image APIs;
the old `ExplicitLinear` candidate remains available explicitly.

On Windows, `cuda_eye_bag::EyeBag::compose_field` uses the same float color
input contract and is checked against the CPU stage. `compose` is a convenience
entry for an already quantized color image; it cannot retain sub-byte model
colors. Neither entry synthesizes model output or face ROI alignment. PF5
dark-circle composition remains unresolved and its `Vendor` mode still refuses.

## GPU and geometry adapters

- `cuda::Nvidia`: dynamically loads the Windows CUDA driver. P15 composite,
  surface blur, Gaussian blur and box blur use the bundled PTX and explicit
  float planes. A returned dispatch describes the actual device and loaded
  PTX ISA. The current P15 image has been exercised on RTX 4070 Ti; the PTX
  header compatibility workaround is not a recompilation or a general promise
  that a newer PTX image works on an older driver.
  GPU work is serialised by a process-wide lock: parallel test threads creating
  several contexts crashed the driver with `STATUS_ACCESS_VIOLATION`. This is a
  deliberate one-job-at-a-time limit, not a thread-safety claim.
  Verified on RTX 4070 Ti, each byte-identical to its Rust oracle: P15
  composite (float and RGBA8, non-square and padded), skin whiten/ruddy on the
  bundled atlases, PF8068 gloss, PF108-110 teeth whitening on the bundled
  TeethWhitenV3 pair, the mesh select/emit pair, the 13 fragment vectors, the
  PF90002/90010 moisture pair, PF407 colour/leg paths, and the PF100007/PF8033
  dense smile warps. `cuda_smile::Smile::apply_pf8033` accepts explicit dense
  pixel displacements and a mask. Its shared kernel preserves the CPU
  truncating byte store and source alpha.
- `metal::MetalContext`: macOS runtime Metal compilation and synchronous
  dispatch for the supplied P15, whitening and PF100007 smile sources. Smile
  compilation is cached on its first use through `MetalContext::smile`.
  Non-macOS execution
  returns `HostUnavailable`. Windows type checks do not establish Apple ABI,
  Metal compilation, or M-series GPU correctness.
- `device::select`: chooses from explicit probe results and per-effect
  readiness. An explicit CUDA/Metal request fails when unavailable; it does
  not report CPU fallback as GPU execution.
- `cuda_texture_blend::Blend::apply`: explicit NVIDIA execution of
  `kernels::texture_mask_blend::apply`, including a separate effect UV stream
  and original alpha. A missing CUDA driver is an error, never an implicit
  CPU execution. Its correctness checks use the production CPU implementation.
- `raster::rasterize`: mesh execution with explicit geometry, source UV, EBO,
  sampler and face-culling/overlap rules. The original native and C# cull labels
  disagree in an older lane; this module defines its own rules explicitly
  rather than inferring an uncaptured original GL state.
- `leg_refine`: PF407's two-stage path. The strip offset is kept in float32;
  an RGBA8 carrier clamps the signed `src_uv - dst_uv` to zero and silently
  makes the effect an identity, so the field never crosses that boundary. The
  type143 colour pass writes the sampled texel including alpha.

The family modules also include geometry-only and parameter-only stages.
`Access::Cpu` describes a tested CPU stage, which can produce geometry rather
than an image. `Access::GpuPlanned` denotes an incomplete final path, and
`Access::Refused` preserves a known unresolved effect. Count these separately
when reporting readiness; counting `ParamFlag` entries is not counting effects.

## Local assets

Private third-party LUTs, meshes, and test fixtures are not stored in Git; see
[`../../private-assets.md`](../../private-assets.md) for layout and checksum
verification. When `private-assets/` is absent, tests that require those files
are marked `ignored` (`needs private-assets/`).

`private-assets/portrait/skin-adjust/` carries the two 512x512 PNG LUT atlases
named by the `SkinAdjust_PixCook` config (`WhitenPath`, `RuddyPath`) and the
`SkinChooses` curve subset. They are material inputs, not a skin mask, face ROI
or UI-to-native mapping.

The gloss values in that config are captured native percent (neutral 50), not
the signed UI (neutral 0). No conversion between the two is fabricated.

[`tests/cuda_lut.rs`](tests/cuda_lut.rs) runs the skin LUT atlases through the
NVIDIA stage and compares with the CPU oracle byte for byte. Its mask is
synthetic, so it proves the material path, not a portrait result.
[`tests/metal_asset.rs`](tests/metal_asset.rs) carries the same atlas as RGBA
float32 for the macOS run and asserts the LUT channel order offline.

`private-assets/portrait/teeth/` carries the `TeethWhitenV3` pair (`Whiten.png`,
`Bright.png`) and the static 111x43 `toothMask.png`.
`skin-adjust/curves.tsv` holds the 27 generated 74-sample HSLTuner curves
derived from the nine `SkinChooses` control sets. Verified:
`HairColorUnify/effect.json` has no parameter table and no asset references, so
PF30009 keeps its own explicit curves and must not default to these SkinAdjust
presets. Only the shader shape (`HSLTuner.frag` with three 74-sample `GL_R32F`
curves) is shared.

`private-assets/portrait/radiance/` carries the PF90004 tables (300/150 LUT
rows, 16384 gamma floats, 18 matrix floats) with a SHA256 manifest.

`private-assets/portrait/smile/` carries the four PF8033 meshes as little-endian
float32 triples, their common 4269-triangle topology and the config. The caller
supplies the same-job projection matrix and selects the reshape explicitly.
`face_pose::compose` converts the same-job face3d pose `p7`, the crop-to-image
affine and the image size into that projection; its native SSE fixture matches
within `1e-6`. `mesh_attribute::rasterize` then interpolates the per-vertex
displacement over the real triangles into a per-pixel field, with an explicit
screen domain and last-writer coverage. The field-to-warp degree composition and
handedness stay caller decisions because the vendor convention is not proven.

`kernels::texture_mask_blend` is the recovered `BlendMode_Texture_Normal`
stage. Original and mask share the geometry UV; the effect texture keeps its
own UV stream. The output is the shader's encoded RGBA8 mix with the original
alpha; the host framebuffer blend is a separate caller step.

`kernels::teeth_blend` is the PF30026 final mask blend from `pix-worker.exe`:
`m = mask * 0x3B807BBB`, a `cvtt` store and original alpha. Its opaque model
output and dense warp remain caller inputs.

`makeup_material::lipstick_image_layer` binds `LipStickImageMakeup` with the
recovered `LipMaskType` table (`3 -> MASK_R`) and `FilterType` table
(`41 -> Multiply`, `42 -> SoftLight`). Material and owner mask take separate
UV streams; the host combines degree, opacity, shelter and region once.

`makeup_material::{color_makeup, image_makeup}` serve the eye and brow rows
whose cell class, mask field and `FilterType` are pinned (PF100/117 RB,
PF103/30007 RG, PF90032 RGB, PF106/116 RGBA). `image_makeup_unmasked` is the
`EyePupilMaskType=0` path, where both mask branches are skipped. The
`*_dual_parallel` form splits the frame into row bands; output is byte
identical to the serial call and from 65,536 pixels it is 5.9-7.8x faster on a
20-thread machine.

`kernels::suit_makeup::compose` runs the PF90033 preset as an ordered pass
list: each pass applies `degree * opacity * shelter * region * material.a *
mask` exactly once, `> 0.01` gates the blend, and a `LipColor` pass is refused
instead of being blended as Normal. `compose_gpu` is the NVIDIA twin; a colour
pass becomes a 1x1 material so both kinds share the image-layer kernel.

`kernels::beard::apply` is PF21026 stubble removal: `w = min(mask_a, mask_b)`,
an AddEx with `gamma = 0`, a strict `strength > 0.005` gate, and a `0.75`
strength scale under a 300 px long edge. The UI-to-native strength mapping is
not recovered, so the caller passes it. `cuda_beard` and the Apple MSL
candidate share the same host-side gate and scale.

`kernels::teeth_alignment` holds PF30026's fixed 1024x768 target quad, the
2x3 affine (`Fit::FirstThree` or `Fit::LeastSquares`, no default) and the
A/B edge gate (`A+B > 100`, `ratio > 0.14`). The mask the worker reads at
`record+0xA8` has no static producer, so callers supply it.
