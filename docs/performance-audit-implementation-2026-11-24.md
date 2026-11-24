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
