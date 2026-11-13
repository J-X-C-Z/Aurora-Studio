# Aurora portrait stages

`aurora-pixcake` 0.1.0, MIT, Rust edition 2021. A dependency-free Rust library
of explicit-input CPU pixel/geometry stages and CUDA/Metal runtime adapters.
The crate inherits version/license from the release workspace. No model,
LUT/atlas, private topology, real photograph or original binary is bundled.
All original library modules and shader source dependencies are retained.
A callable stage is not an automatic portrait pipeline or evidence of full
original-algorithm parity. GPU availability and missing-input refusals remain
explicit; Metal runtime tests require macOS and a Metal device.

## API changes in this public snapshot

- `kernels::forehead_common::Inputs` adds `topology: &[u16]`: exactly 705
  indices, each below 136. Empty input returns `Refusal::MissingTopology`;
  wrong length or invalid indices return `Refusal::BadIndexTable`.
  No type1012 topology is embedded or synthesized at runtime.
- `kernels::skin_models::gloss_apply_with_luts` and
  `cuda_gloss::Gloss::apply_with_luts` add a final argument
  `Option<[&[u8; 256]; 3]>` in positive1, positive2, negative order.
  Callers supply their own authorized tables. `None` returns
  `Refusal::MissingInput("gloss_luts")` for active work. Legacy `gloss_apply`
  and `Gloss::apply` retain their signatures and delegate with `None`.
  Native percent 50 remains an identity after all existing input checks;
  CUDA returns no dispatch. No UI-to-native conversion is invented.
- CUDA gloss takes one additional device LUT pointer per pass. Its PTX was
  rebuilt from the curated source with `nvcc --ptx -arch=sm_89 --fmad=false`;
  no LUT constant-memory payload remains. The 32-byte uniform block is unchanged.
- `texture_surface_blur` (PF8069 negative UI) remains available with its original
  signature and no external assets. Image/mask/finite/range checks are retained.
- Existing `Image`, facial smoothing, mask compose, neck paste and other typed
  APIs remain available. Catalog gloss input metadata now names caller tables.

## Build and tests

Run from the release workspace:

```text
cargo check -p aurora-pixcake --all-targets --all-features --locked
cargo test -p aurora-pixcake --locked -- --test-threads=1
```

Set `CARGO_TARGET_DIR` to the project-root-relative
`90_work/build-cache/portrait-publish` when working in the release worktree.
`PIXCAKE_CUDA_REQUIRED=1` makes a missing CUDA device fail tests rather than skip.
See VALIDATION.md for the actual execution results and limitations.

## Opt-in external tests

`external-fixtures` only compiles local fixture-dependent tests. It does not
package, download, embed or authorize any resource. Set
`PORTRAIT_EXTERNAL_FIXTURES` to an explicitly supplied local fixture directory.
Missing files fail enabled tests; they are never silently accepted. Assets are
read at runtime, so even an all-features compile needs no private files.

Expected layout under that directory:

- `smile/triangles.u32le`, `smile/mean.f32le`;
- `teeth/whiten.rgb`, `teeth/bright.rgb`, `teeth/whiten_lut.f32`,
  `teeth/blend_fixture.bin`;
- `eye/positive_asym.part8.rawA.bin`, `eye/positive_asym.part8.rawB.bin`,
  `eye/positive_asym.part9.rawA.bin`, `eye/positive_asym.part9.rawB.bin`;
- `pf21015/TintGreen.rgb8`, `pf21015/TintMagenta.rgb8`;
- `metal_skin/` and `metal_smile/` indexes and their referenced local files.

None of these files or directories ships here. All in-memory synthetic tests
remain enabled by default; teeth-blend shape/alpha tests now generate their
own small inputs. External hashes and oracle assertions are retained behind
the feature. Synthetic gloss test curves exist only inside test code, never
as production replacements for missing tables.
