# `aurora-engine` 公共面评审（G12）

- 日期：2026-11-26
- worktree：`J-X-C-Z/架构`（macOS arm64），HEAD `8819dc4` + 并发未提交改动
  （`mod layer_stage;`、`mock/process.rs`、`processing/color.rs` 等由 G2 layer lane 编辑）。
- 口径：以当前源码与 `cargo` 实际输出为准。外部调用点只统计本仓库：
  `crates/app/src`、`crates/app/tests`（不存在）、`crates/engine/tests`、
  `crates/engine/examples`、`crates/portrait/{src,tests,examples}`。
  `90_work/lanes/` 等主树外 lane 不在本 worktree 内，未计入调用点（见“风险”）。
- 结论性命令：`cargo check --workspace --all-targets --locked`（exit 0）、
  `cargo test -p aurora-engine --test architecture_rules --locked`（11/0）、
  `cargo test -p aurora-engine --lib ...`（1911/29/94/105，失败集不变）、
  `cargo test -p portrait-app --lib --locked`（99/1/5，既有 EIP fixture）。

## 1. 范围与方法

审计对象是 `crates/engine/src/lib.rs` 的 crate root 出口以及
`crates/engine/src/engine.rs` 的 `ImageEngine` 方法：

1. 解析 root 的 `pub use`（含多行 brace 列表与 glob）和 `pub mod`；
2. 对 glob（`pub use module::*`）枚举模块内 `pub` item；
3. 逐名在外部目录做整词 grep（`\bName\b`），并区分“模块路径调用”
   （`aurora_engine::module::Name`）与“root 调用”（`aurora_engine::Name` 或
   `use aurora_engine::*` 后的裸名）；
4. 对候选同时检查 engine 内部是否通过 root 路径使用：`crate::Name`、
   `use crate::{... Name ...}`、以及 `mock/{mod,process,develop_gpu}.rs` 的
   `use crate::*;`。内部路径判断最终以编译器为准：每次改动后跑
   `cargo check --workspace --all-targets --locked`。

说明：整词 grep 对 `Source`、`render`、`open`、`enabled`、`decode` 等常见词有
误报（app 自有同名符号/方法、字符串字面量）。这类条目以编译器结果定案，不把
误报计为调用点。

## 2. 总览数字

| 面 | 数量 |
|---|---|
| root `pub use` 语句（改动前） | 60 |
| root 显式导出名（改动前） | 350 |
| root glob 模块（含 `portrait` 内 2 个） | 30，枚举 `pub` item 约 796 |
| `pub mod` / `pub(crate) mod` / `mod` | 54 / 2 / 61 |
| `ImageEngine` trait 方法 | 81 |
| 本轮判定为内部、已收窄的显式名 | 256 |
| 本轮收窄的模块/glob | 1 个模块 re-export、1 个 `pub mod`、3 个 root glob |
| 保留的显式领域类型（root） | 约 90（见 §5） |

审计总量约 1280 个符号（350 显式名 + 约 796 glob item + 54 模块 + 81 trait 方法）。

## 3. 分类结论

- **稳定领域 API（keep）**：`ImageEngine`/`FileEngine`/`MockEngine`、
  `AdjustmentSet` 及其子结构、`PreviewRequest`/`PreviewFrame`/`RenderQuality`、
  document/contracts、目录/集合/元数据、导出/作业、打印、合并、ICC/lens/style
  resources、C1 catalog 与 sidecar 类型。这些即使当前 app 不按名引用，也由
  `ImageEngine` 方法签名或文档契约承载，必须保持 `pub`。
- **内部 producer/vendor/CUDA/portrait-stage/render-plan（收窄）**：
  `portrait_stage::*` 的 producer 输入/回执/模型常量、`portrait_historical_tone_provider`、
  `body_shader_abi`、`wrinkle_legacy`、`aurora_portrait::cuda_pf90021`、
  `mock::process` 渲染帧类型、`render_plan`、以及三个 root glob
  （`blemish_gainmap`、`portrait_makeup_bundle`、`portrait_makeup_provider`）。
- **保留但对齐 owner**：`c1_model`/`c1_masks`/`c1_styles`/`lensdb`/`recipe`/`raw`
  等 glob 里的零调用名。它们不是 producer/vendor 内部词表，可能是主树外
  `c1-parse` lane 的接口；本轮不动，记录为后续可选收窄。

## 4. 已应用的收窄

“零调用证据”= 在 §1 的五个外部目录中没有任何**代码**引用该名字（root 或模块路径）；
命中仅为字符串字面量或 app 自有同名符号的，按零调用处理。所有改动后
`cargo check --workspace --all-targets --locked` 通过。

| # | 条目 | 原位置 | 外部调用点 | 处理 | 风险 | 应用命令 |
|---|---|---|---|---|---|---|
| A1 | `Pf90021Direction`、`Pf90021RuntimeScalars`、`VolumeLut64` | `lib.rs` `pub use aurora_portrait::cuda_pf90021::{...}` | none（app/tests/examples/portrait 均无） | drop root re-export | 低；CUDA 类型仍可从 `aurora_portrait::cuda_pf90021` 取 | 删除该 `pub use` 块 |
| A2 | `portrait_historical_tone_provider::{build_*, HistoricalTone*Payload/RowData/RowJob/Provider*, HISTORICAL_TONE_PROVIDER_PFS}`（17） | `lib.rs` `pub use portrait_historical_tone_provider::{...}` | none | drop root re-export | 低；模块私有，主树无消费者 | 删除该 `pub use` 块 |
| A3 | `body_shader_abi::{decode/encode_body_geometry, BodyKeypointSet, BodySixPointShaderInput, EncodedBodyGeometry, BODY_*}`（11） | `lib.rs` 同名 `pub use` | none | `pub(crate) use` | 低；GPU shader ABI，非公共契约 | `pub use` → `#[allow(dead_code, unused_imports)] pub(crate) use` |
| A4 | `portrait_stage` 历史肤色后端选择（`apply_historical_tone_row`、`gpu_kernel`、`gpu_lut_reason`、`plan_historical_*`、`Historical*Gpu*`、`GPU_PROVEN_*` 等，15） | `lib.rs` 同名 `pub use` | none（`gpu_kernel` 仅命中 app 的字符串 `"gpu_kernel"`） | `pub(crate) use` | 低；producer 后端内部 | 同上 |
| A5 | `portrait_stage` producer 输入/回执/模型常量（164） | `lib.rs` 大 `pub use portrait_stage::{...}` | none | `pub(crate) use` | 低；内部使用已由 `crate::FaceInputBundle`、`crate::FaceInputSession`、`crate::C1FaceDetection`、`crate::FaceStageTransform` 及模块路径覆盖 | 拆分：`pub use` 保留 11 个测试驱动的名字，其余 `pub(crate) use` |
| A6 | `wrinkle_legacy::{WRINKLE_*, Wrinkle*, faceparsing_v2_wrinkle_masks_refusal, validate_wrinkle_uniforms}`（23） | `lib.rs` 同名 `pub use` | none | `pub(crate) use` | 低；legacy producer 词表 | 同 A3 |
| A7 | `portrait_stage` glare/skin-repair producer 内部（18，`GlareFinalWritebackOutput`、`GlareInputs`、`SpotlessInputs` 等） | `lib.rs` 同名 `pub use` | none | `pub(crate) use` | 低 | 拆分：`pub use` 保留 21 个测试驱动的名字，其余 `pub(crate) use` |
| A8 | `portrait_face_input_stage`（模块 re-export） | `lib.rs` `pub use portrait_stage::portrait_face_input_stage;` | none | `pub(crate) use` | 低；内部调用 `crates/engine/src/mock/process.rs:1179/1180/1186/1196/2519`、`portrait_eye_bag_inputs.rs:61/308`、`portrait_makeup_wiring.rs:38` | `pub use` → `pub(crate) use` |
| A9 | `blemish_gainmap::*` root glob | `lib.rs` `pub use blemish_gainmap::*;` | 仅模块路径：`crates/engine/examples/blemish_gpu_bench.rs:10`、`blemish_env_off.rs:3` | drop root glob（保留 `pub mod blemish_gainmap;`） | 低；模块路径不变 | 删除 root glob 行 |
| A10 | `portrait_makeup_bundle::*` root glob | `lib.rs` 同名 glob | 仅模块路径：`crates/engine/tests/makeup_owner_bytes.rs:3` | drop root glob（保留 `pub mod`） | 低 | 删除 root glob 行 |
| A11 | `portrait_makeup_provider::*` root glob | `lib.rs` 同名 glob | none | drop root glob（保留 `pub mod`） | 低；模块仍 `pub` | 删除 root glob 行 |
| A12 | `mock::process::{render, render16, Rendered, Rendered16, Source}`（5） | `lib.rs` `#[doc(hidden)] pub use` | none 按名（`Source`/`render` 的 app 命中是 app 自有 `Source` 与 `.render()` 方法） | `pub(crate) use` | 低；内部路径 `pano.rs:17`、`mock/mod.rs:604`、`mock/process.rs` 已覆盖 | `#[doc(hidden)] pub use` → `#[allow(dead_code, unused_imports)] pub(crate) use` |
| A13 | `render_plan`（`pub mod`） | `lib.rs` `pub mod render_plan;` | none；仅自身 14 个 `#[cfg(test)]` | `pub(crate) mod` + `#[allow(dead_code)]` | 低；尚未接入默认渲染路径 | `pub mod` → `#[allow(dead_code)] pub(crate) mod` |

A5/A7 中保留 root `pub use` 的名字（共 32）是 engine 自身集成测试驱动的：

- `crates/engine/tests/pf30025_live_bind.rs:5-11`：`Pf30025Inputs`、
  `Pf30025LiveModelOutputs`、`Pf30025ProducerInputs`、`Pf30025SameJobReceipt`、
  `Pf30025SourceSurface`、`PortraitBackend`、`PortraitJobInputs`、
  `PortraitJobReceipt`、`PortraitMasks`、`ProducerInputs`、
  `apply_portrait_effects_with_job`。
- `crates/engine/tests/wl35_glare_bind.rs:1-6`、`wl41_glare_residual.rs:1-4`、
  `wl46_glare_producer.rs:1-6`：`Glare*Inputs/Receipts`、`PF90130_*`、
  `Pf90130AlignedGeometry`、`Pf90130GlareProducer`、`apply_glare_*`、
  `apply_pf90130_producer_stage`、`bridge_glare_planes`。
- `PortraitImage`/`PortraitRefusal`（`pub use aurora_portrait::{Image as ..., Refusal as ...}`）
  同样由这些测试使用，保持 root `pub`。

## 5. 保留的稳定领域 API（代表调用点）

| 面 | 代表条目 | 外部调用点（file:line） |
|---|---|---|
| trait 与引擎 | `ImageEngine`、`FileEngine`、`MockEngine` | `crates/app/src/state/source.rs:7`；`crates/engine/tests/preview_contract.rs:5`；`crates/app/src/state/mod.rs:2890` |
| 编辑模型 | `AdjustmentSet`、`LocalAdjustments`、`Curve`、`ColorWheel`、`NormRect` | `crates/app/src/state/binding.rs:6/50-90`；`crates/app/src/widgets/curve.rs:11`；`crates/app/src/widgets/wheel.rs:11`；`crates/engine/tests/preview_contract.rs:5` |
| 预览契约 | `PreviewRequest`、`PreviewFrame`、`RenderQuality`、`RgbaImage`、`Histogram` | `crates/app/src/state/preview.rs:13`；`crates/app/src/state/scopes.rs:8`；`crates/app/src/ui/print.rs:17`；`crates/engine/tests/preview_contract.rs:5` |
| 目录/库 | `ImageId`、`ImageInfo`、`Collection`、`CollectionKind`、`UserCollection`、`ColorTag`、`SmartRule` | `crates/app/src/state/library.rs:14`；`crates/app/src/ui/collections.rs:16`；`crates/app/src/state/arrange.rs:9` |
| 导出/作业 | `ExportFormat`、`ExportJob`、`JobStatus`、`ProcessRecipe`、`MetadataPolicy` | `crates/app/src/ui/export.rs:19`；`crates/app/src/state/tasks.rs:7`；`crates/engine/tests/export_faults_bo4.rs:12-15` |
| 打印 | `PrintRequest`、`PrintLayout`、`PrintPreview`（同时在 `ImageEngine` 签名中） | `crates/app/src/ui/print.rs:16-17` |
| 合并/软打样 | `HdrOptions`、`PanoramaOptions`、`Projection`、`DeghostMode`、`Merge*`、`SoftProofSettings` | `crates/app/src/planned.rs:55`；`engine.rs` 方法签名 |
| 采集/联机 | `CameraStatus`、`CameraSettings`、`TetherStatus`、`CaptureDevice` | `crates/app/src/state/device.rs:7`；`crates/app/src/tools/capture.rs:11`；`crates/app/src/ui/live_view.rs:16` |
| C1/catalog/sidecar | `C1Catalog`、`C1Import`、`export_session`、`c1_cos`/`c1_writeback` 模块 | `crates/app/src/state/mod.rs:2141-2158/2939-2940`；`crates/engine/tests/export_faults_bo4.rs:183-185` |
| resources | `StyleInfo`、`ParamMeta`、`ParamOverrides`、`MetaSource`、`icc`/`print`/`aimask`/`raw_cache` 模块 | `crates/app/src/planned.rs:12`；`crates/app/src/widgets/param_c1.rs:14/41`；`crates/app/src/lib.rs:116/122/134`；`crates/app/src/ui/print.rs:16`；`crates/engine/tests/quality_bench.rs:478-488` |
| 节点/图层 | `NodeGraph`、`Layer`、`Mask`、`MaskOp`、`GraphNode`、`LayerKind` | `crates/app/src/state/nodes.rs:8`；`crates/app/src/tools/nodes.rs:20` |
| 文档/契约 | `document::DocumentRevision`、`contracts` | `crates/app/src/state/mod.rs:2328/2533`；`lib.rs` `pub mod contracts` |

（`crates/app/src/**` 中大量文件使用 `use aurora_engine::*`，例如
`state/mod.rs:33`、`ui/viewer/mod.rs:13`、`ui/viewer/overlays.rs:8`、
`tools/{exposure,color,output,details,library,layers,lens}.rs`、`state/render.rs:10`、
`state/collections.rs:8`；上表只列显式命名点。）

## 6. 延后项（需 owner / 主树外 lane 确认）

| 条目 | 现状 | 外部调用点 | 建议 | 延后原因 |
|---|---|---|---|---|
| `pub mod processing` | `lib.rs`；注释写明“with compatibility re-exports” | 本仓库 none | 候选 `pub(crate) mod` | 兼容 re-export 可能服务主树外 lane；需 owner 确认后再收 |
| `c1_model::*` 中 39 个零调用名、`c1_masks` 8 个、`c1_styles` 9 个 | 私有模块 + root glob | 本仓库 none | 候选 `pub(crate)` / 拆 glob | c1-parse lane 的接口面，主树外使用未验证 |
| `lensdb::*`、`layer_graph::*`、`matchlook::*`、`recipe::*` 的零调用名 | 私有模块 + root glob | 本仓库 none（`recipe` 全 0） | 候选 `pub(crate)` | 领域/output 语义，是否外部 API 需 owner 裁决 |
| `raw`/`aurora_raw`/`camera_params`/`c1_*` 等 `pub mod`（本仓库 0 直接路径调用） | `pub mod` | none 直接；app 经 root glob 用条目 | 候选 `pub(crate) mod` | 主树外 RAW/C1 lane 可能依赖；未验证 |
| `pub use aurora_raw::z6iii` | feature `z6iii-raw-export` | none | keep | 仅 feature 构建；`aurora_raw` 本身 `pub`，收窄收益低 |
| `C1Collection`/`C1DocumentType`/`C1Image`/`C1Keyword`/`C1Variant`/`WalReplay`、`resolve_database`、`C1_COLLECTION_ID_BASE`、`SessionExport` | root `pub use` | none | keep | 稳定 C1 领域 API；c1-parse lane 接口 |
| `exif::read_exif`、`camera_defaults::overrides_for`、`c1_banding::BandingRange`、`closedeyes::{detect_eyes, EyeBox}`、`filmcurve::{curve_dir, FilmCurve}`、`dust_patch::{DustDetector, DustError}`、`dust_lcc::{DustLccError, LccLoadError}`、`merge_async::{MergeErrorCode, MergeStage}`、`rawinfo::RawInfoBlock`、`store::StoreError` | root `pub use` | none | keep | 领域值/错误类型；root glob 或 trait 相邻，收窄收益低 |
| glob 内其余零调用名（如 `adjust` 57、`raw` 51、`output` 23） | 私有模块 + root glob；内部可能 `crate::Name` | app 经 `use aurora_engine::*` 可能按名使用 | 不批量收窄 | 常见词误报多，需逐个编译验证；不属于本轮 producer/vendor 范围 |

## 7. `ImageEngine` 领域方法分类（81 个方法全部保留）

`ImageEngine` 是 app 唯一入口（architecture rule 1），不做删除或降级；只记录语义分区：

| 分区 | 方法 |
|---|---|
| 身份/能力 | `name`、`capabilities`、`portrait_capabilities`、`portrait_capability_registry_tsv` |
| 目录/集合/元数据 | `images`、`import`、`collections`、`library_root`、`collection_path`、`user_collections`、`manage_files`、`collection_images`、`save_user_collections`、`set_trashed`、`update_info`、`update_info_batch`、`layer_summaries`、`retouch_masks` |
| 编辑持久化 | `load_adjustments`、`save_adjustments`、`save_adjustments_batch`、`save_adjustments_batch_report`、`save_adjustments_queued`、`flush_pending_saves`、`pending_save_status`、`new_variant` |
| C1 sidecar / 镜头标定 | `save_c1_settings`、`save_c1_catalog`、`load_lcc_calibration`、`set_lcc_calibration`、`lcc_calibration` |
| 渲染 | `thumbnail`、`thumbnail_profiled`、`render_preview`、`render_preview_fast`、`sample_color` |
| 分析 | `auto_adjust`、`pick_white_balance`、`match_look`、`compute_ai_mask`、`eye_report`、`auto_rotate` |
| 资源 | `icc_profiles`、`lens_profiles`、`styles`、`film_curves`、`import_style`、`style_for_image`、`param_metadata`、`param_overrides` |
| 输出/作业 | `start_export`、`job_status`、`cancel_job`、`source_status` |
| 合并（同步/异步） | `merge_hdr_frames`、`merge_hdr_with_options`、`merge_panorama_frames`、`merge_panorama_with_options`、`merge_hdr`、`merge_panorama`、`merge_progress`、`cancel_merge`、`merge_result` |
| 软打样/打印 | `set_soft_proof_profile`、`soft_proof_profile`、`soft_proof_preview`、`soft_proof_gamut_warning`、`print_layout`、`print_preview`、`print_to_pdf`、`print_to_printer` |
| 采集/联机 | `capture_devices`、`capture_start`、`capture_stop`、`capture_arrived`、`tether_status`、`camera_status`、`camera_settings`、`capture`、`live_view_frame`、`live_view_focus` |

有默认实现、引擎可不实现的方法（`*_with_options`、`save_adjustments_*`、`merge_*`、
`print_*`、`capture_*`、`live_view_*`、`soft_proof_*` 等）仍是**公共领域契约**：
它们出现在 trait 签名里，因此其参数/返回类型（`MergeRequest`、`MergeOutcome`、
`PrintLayout`、`SoftProofSettings`、`CaptureDevice` 等）必须保持 `pub`，不能因
“app 未按名引用”而降为 `pub(crate)`。

## 8. 应用顺序

1. **先跑基线**：`cargo check --workspace --all-targets --locked`、
   architecture 11/0、engine lib 与 app lib 计数。
2. **零内部依赖的 root re-export**（A1、A2、A9–A11）：直接删除或拆掉 root glob，
   模块路径保持可达。
3. **`pub mod` / 模块 re-export**（A8、A13）：改 `pub(crate)`，先确认内部 `crate::`
   路径存在（`FaceInputBundle` 等）或只有自测（`render_plan`）。
4. **纯 producer 词表**（A3、A4、A6）：整组 `pub(crate) use`，加
   `#[allow(dead_code, unused_imports)]`。
5. **混合大组**（A5、A7）：先 grep 出测试驱动的名字保留 `pub use`，其余
   `pub(crate) use`；拆分后用编译器确认无遗漏。
6. **渲染器内部帧类型**（A12）：`pub(crate) use`。
7. 每批后跑 `cargo check --workspace --all-targets --locked`；最后跑三套测试。

## 9. 验证结果（本 worktree 实测）

```text
cargo check --workspace --all-targets --locked
→ exit 0（Finished dev profile）

cargo test -p aurora-engine --test architecture_rules --locked
→ 11 passed; 0 failed; 0 ignored

cargo test -p aurora-engine --lib --locked -- --test-threads=1 \
  $(scripts/testdata-skips.txt 的 106 条 --skip)
→ 1911 passed; 29 failed; 94 ignored; 0 measured; 105 filtered out
  （与基线完全一致；失败集为既有外部 fixture / real-catalog 项，
  含 deliberate-not-skipped 的 portrait_historical_tone_provider 两项）

cargo test -p portrait-app --lib --locked
→ 99 passed; 1 failed; 5 ignored
  （既有 EIP fixture：state::import_edits_tests::eip_variants_fill_the_preview_edits_immediately）
```

注意：`pub(crate)` 收窄会让原先仅靠“公共可达性”存活的 producer item 变成
dead code，`aurora-engine` lib warning 由 223 增至 282。这是**可达性事实**，不是
功能变化；`cargo check` 仍 exit 0。若后续决定这些代码应长期保留，建议在该
producer 模块声明处加定向 `#[allow(dead_code)]`，而不是把 root re-export 加回来。

## 10. 风险与后续

- **主树外 lane**：`AGENTS.md` 说 `portrait-parse`、`c1-parse` 在
  `90_work/lanes/` 改副本并按 patch 合并。本轮只验证了主树；若这些 lane 用
  `aurora_engine::PortraitJobInputs` 等被保留名之外的 producer 名，需要统筹在
  合并 patch 时改为 `aurora_engine::portrait_stage::...` 之外的路径——但
  `portrait_stage` 是私有模块，主树外无法访问。若确有主树外生产者依赖，必须由
  owner 决定是恢复 root re-export 还是提供新的公开 producer 契约。**本轮未验证
  主树外依赖**。
- **Windows/feature**：`z6iii-raw-export` 与 CUDA/DirectML 路径未在本机验证；
  本轮改动只涉及可见性，不改变 `cfg` 结构。
- **dead code**：见 §9；建议后续由 owner 决定“接线 / 删除 / 定向 allow”。
- **未收窄的零调用领域名**：见 §6；建议在确认主树外 lane 后分批处理。
