# Aurora Studio 全面能力与性能审查（实施中）

- Repository: J-X-C-Z/Aurora-Studio
- Audit baseline: `99ec192e17cbe7b58ccf701530dd3790c40a92bf`
- Working branch: `perf/full-audit-2026-11-24`
- First implementation commit: `b1b0b549c561e33ef014df6cfcd33b366e84934f`
- Scope: whole-repository tree inventory (872 blobs, 106 trees, not truncated), manifest/README inspection, plus deep source-path review of RAW, FileEngine, Source/Mip, SourceSample, Preview, GPU Develop, CUDA runtime, ONNX/AI, portrait, layers, batch/export. This is not a claim that every line of all 872 files has been manually read; full line-by-line review and Windows hardware validation remain outstanding.

## Capability map

| Area | Existing capability observed | Performance/architecture focus |
|---|---|---|
| Desktop app | Rust + GPUI app, state-driven preview | startup critical path, async task prioritization |
| RAW | LibRaw plus pure-Rust NEF/ARW/DNG/CR2/CR3 front-end and fallback | decode working-set admission, reuse decoded bytes, decoder parity |
| Source | F32 RGB Source and mip pyramid | total derived-memory accounting, bounded eviction |
| Preview | latest-value scheduling, display conversion instrumentation, tile sampling | cross-image tile cache, stage cache, generation safety |
| GPU | CUDA NR/Clarity/Pixel/Develop and shared runtime wrapper | true device-wide budget, fewer full-frame conversions and transfers |
| AI / Portrait | ONNX Runtime, face input/session caches, dust inference | model/session single-flight, content fingerprint reuse, inference cache |
| Layers / graph | local masks/layers and DAG graph support | reachable-node pruning, dirty propagation, mask reuse |
| Export / batch | staged atomic output and sequential batch | bounded global export scheduler, RAM/VRAM admission |

## Implemented in this branch

### Multi-key SourceSample Tile LRU

File: `crates/engine/src/mock/process.rs`

Previous design retained a single `TileStore`. Switching to a different image/geometry/region key replaced the entire store, so returning to a recently edited image lost its cached samples.

Changes:
- Replace the single optional store with a bounded multi-key LRU.
- Retain up to 8 sampling keys.
- Keep a single global byte budget (default 192 MiB, existing environment override preserved).
- Promote keys and tile coordinates on hits.
- Evict least-recently-used tiles across keys when over budget.
- Add unit tests for cross-key reuse, key-count eviction, and global byte budget.

The sampling formula, tile size, cache key dependencies, and output pixel math are unchanged. This is a cache-policy change only.

## Remaining high-priority work

1. GPU Develop preflight must happen before allocating Sample/Display/Gain/IO full-frame buffers. Current code path can materialize multiple full-frame arrays before a later GPU-capacity refusal.
2. GPU Develop input currently converts SourceSample to Sample and then to an interleaved four-float IO vector; avoid redundant full-frame materialization and add a tile/streaming input path.
3. DecodeGate estimates must account for raw sensor dimensions and actual decoder transient working sets, including large JPEG/TIFF/PNG, not just retained proxy dimensions or file category.
4. SourceCache byte accounting must include derived portrait/AI/mask buffers and rescan invalidation.
5. Export jobs need a global RAM/VRAM-aware admission scheduler, not only per-batch serialization.
6. GPU pools remain per-stage; the shared wrapper is not yet a device-wide memory manager or a cross-stage resident-frame pipeline.
7. Portrait input clone/hash and producer/result cache behavior need revalidation after recent PF changes.
8. Dust inference result caching and inference preprocessing must be validated under parameter drags and geometry changes.
9. ONNX mask/session caches should use bounded LRU/byte budgets and avoid long global locks during model initialization.
10. Add end-to-end performance telemetry for decode, source construction, sampling, GPU transfer/kernel, detail, portrait, overlays, scopes, display, and export.

## Validation status

- Source patch committed to branch: yes.
- Unit tests added: 3.
- cargo fmt: not run (no local checkout/build environment in this chat).
- cargo check/test: not run.
- Windows Release / CUDA / ONNX hardware benchmark: not run.
- No measured speedup is claimed. Expected benefit is improved cache reuse when alternating among recently used images with the same sampling configuration, bounded by the existing byte budget.

## Suggested next implementation sequence

P0:
- GPU Develop allocation preflight and buffer reuse.
- Correct decode working-set estimation for RAW proxy and large raster images.
- SourceCache complete accounting and rescan invalidation.

P1:
- Global export admission scheduler.
- GPU device-wide memory budget and cross-stage buffer reuse.
- Portrait producer/result cache LRU and snapshot fingerprint reuse.
- Dust detection cache keyed by source/geometry/model/config.

P2:
- Incremental stage cache and DAG dirty propagation.
- ONNX mask cache LRU.
- Startup/library scan progressive loading and per-stage startup telemetry.

## Review links

- Main baseline: https://github.com/J-X-C-Z/Aurora-Studio/tree/99ec192e17cbe7b58ccf701530dd3790c40a92bf
- Working branch: https://github.com/J-X-C-Z/Aurora-Studio/tree/perf/full-audit-2026-11-24
- First implementation commit: https://github.com/J-X-C-Z/Aurora-Studio/commit/b1b0b549c561e33ef014df6cfcd33b366e84934f


## Follow-up implementation (branch continuation)

### GPU Develop preflight moved ahead of gain/output allocation

File: `crates/engine/src/mock/process.rs`

- Split the former `gpu_develop_context()` into an admission/prewarm step (`gpu_develop_preflight()`) and later gain-plane preparation.
- The admission decision now runs immediately after the Render Plan is built, before optional Vignette/LCC GPU gain-plane generation and before allocating RGBA8/RGBA16 output buffers.
- Gain-plane construction happens only after GPU Develop has passed the device/budget/crossover checks.
- Pixel math, sample centers, output quantization, CPU fallback, and the CPU Reference path are unchanged.

This specifically removes the previous ordering where a frame could spend time constructing GPU gain planes and output buffers before the GPU Develop admission decision. It does not yet provide a whole-render memory planner: Clarity Guide/Plan preparation and the separate Pixel GPU path still need independent accounting.

### Batch registry bounded retention

File: `crates/engine/src/batch.rs`

- Completed Batch states previously remained in a process-global HashMap indefinitely, including per-image failure strings.
- The registry now retains at most 128 completed batches, evicting the oldest completion first while never evicting active batches.
- Added a unit test covering oldest-completed eviction and active-batch retention.
- Callers should stop polling after `finished == true`; older completed progress may no longer be available after the retention window is exceeded.

### Validation status of follow-up changes

- Source changes committed to the isolated branch: yes.
- New batch-registry unit test added: yes.
- GPU preflight refactor: static call-order review completed.
- `cargo fmt`, `cargo check`, `cargo test`, Windows Release build and RTX benchmark: not run in this environment. No measured speedup is claimed.

## Follow-up implementation: bounded three-stage batch export pipeline

### Where a batch export really runs

- GPUI batch entry: `crates/app/src/state/mod.rs` `AppState::process` calls
  `ImageEngine::start_export` once per selected image x enabled recipe (on the UI
  thread) and `poll_jobs` polls `job_status` every 120 ms. `crates/engine/src/batch.rs`
  is a separate sequential driver and is not on the app path.
- Admission and memory budget: `ExportScheduler` in `crates/engine/src/file_engine.rs`
  (slot limit, byte budget derived from installed/available RAM, FIFO ticket queue,
  `acquire_cancellable`).
- Codec and publishing: `ExportService::start_export` in
  `crates/engine/src/file_engine_services.rs`, writing through
  `crate::output::StagedExport` and publishing under `publish_gate`.

### What changed

Files: `crates/engine/src/export/pipeline.rs` (new),
`crates/engine/src/export/stages.rs` (new), `crates/engine/src/export/mod.rs`,
`crates/engine/src/file_engine_services.rs` (export worker body only),
`crates/engine/src/file_engine.rs` (`estimated_export_bytes` visibility).

- The standard export chain is split into three stages with a single
  implementation: `stages::acquire_source` (the deferred full-resolution decode),
  `stages::render_frame` (`process::render` / `render16`, same arguments),
  `stages::encode_and_publish` (`recipe::write_export(16)` into
  `output::StagedExport` plus `stages::publish_staged`, which keeps the
  `gate -> maps -> registry` order, the cancel check, the supersede check and the
  output map).
- `pipeline::Pipeline` runs those stages on one render lane (GPU) and two encode
  lanes connected by `BoundedQueue`s capped by item count **and** bytes (4 items /
  1 GiB per stage by default). A full queue blocks its producer (backpressure); an
  empty queue always admits, so a frame larger than the whole cap still progresses
  instead of deadlocking. The byte cap bounds what is *queued*; a frame already
  held by a stage is charged by the export scheduler's admission budget, which is
  the same accounting a serial batch had.
- Per-item cancel tokens are polled at every stage boundary and wake a blocked
  producer within 5 ms. A cancelled or failed item never reaches the encode lane,
  so no staging folder and no partial file is created for it.
- The Z6III exact export stays fused and inline (RAW scope untouched) and publishes
  through the same `publish_staged`.
- Switches: `AURORA_EXPORT_PIPELINE=serial|0` runs the same stage functions inline
  (A/B and the byte-identity regression), `AURORA_EXPORT_RENDER_LANES`,
  `AURORA_EXPORT_ENCODE_LANES` set the lane counts.

Byte identity is structural: serial and pipelined modes call the same three
functions, so `export::stages::tests::pipelined_and_serial_exports_publish_identical_bytes`
compares published file names and file bytes rather than pixels. The regression
generates its own 48x32 gradient PNGs with the `image` crate; no external fixture.

### Known limits of this change

- Native RAW full-size decode and the Z6III exact path still run where they ran
  before (submitting thread / inline). Stage 1 carries the deferred
  full-resolution raster decode, not the RAW decoder; changing the RAW path was out
  of scope.
- The pipeline does not replace `ExportScheduler`; a job still holds one admission
  permit for its whole chain, so peak memory stays what the budget already allowed.
  What changes is that a job no longer serialises its own decode/render/encode.
- Renders are serialised on one lane. `mock::process::render` is already
  rayon-parallel per frame and two concurrent GPU develops contend for one device,
  so this is the intended shape; `AURORA_EXPORT_RENDER_LANES=2` restores the old
  two-renders-at-once behaviour for an A/B comparison.
- No Windows Release benchmark numbers are claimed here; the numbers must come from
  the owner machine (see `CONTRIBUTING.md`). `export::stages::pipeline_stats()`
  exposes per-stage item counts, queue peak bytes and blocked-push counts for that
  measurement.

### Validation status of the export pipeline

Run by this lane in a detached worktree of the same baseline commit, with an
isolated `CARGO_TARGET_DIR` so the shared build directory stayed untouched:

```powershell
cargo check -p aurora-engine --lib --tests --locked
cargo check -p aurora-engine --lib --features z6iii-raw-export --locked
cargo test  -p aurora-engine --lib --locked export:: -- --test-threads=1
cargo test  -p aurora-engine --locked --test export_faults_bo4 -- --test-threads=1
```

- `export::` unit tests: 26 passed, 0 failed (10 of them new: 6 pipeline
  executor, 4 stage/engine regressions).
- `export_faults_bo4`: 4 passed, 0 failed (cancel keeps existing bytes, duplicate
  names, read-only store, failed writes).
- Both feature configurations type-check; no new warnings in the touched files.

Indicative A/B on 8 synthetic 1600x1200 PNGs, dev profile, one machine, CPU develop
path: inline 3942 ms -> pipelined 2247 ms (1.75x). Peak queued bytes 46 MB / 11 MB,
no blocked pushes at that frame size. This is a shape check, not the Windows
Release 24 MP RAW number the owner machine still has to produce.

Not verified here: `cargo test --workspace`, the Windows Release build, `cargo build
-p portrait-app --release`, and any real-camera RAW batch. The app crate was not
compiled by this lane because other lanes had in-progress edits in the same tree.
