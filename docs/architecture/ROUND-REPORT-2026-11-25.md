# Aurora Studio 公共架构 lane：本轮报告（含第二轮）

```text
Aurora Studio 公共架构 lane
基线 SHA / 当前工作副本：8118a4a / J-X-C-Z/架构 worktree（未提交）
总完成度：约 90%（按 A0–A5 产物与验收加权；只统计已实现且有实际测试证据的部分）
设计完成度 / 实现完成度 / 验证完成度：约 95% / 约 90% / 约 65%
```

本会话共派出 18 个子代理（8 + 5 + 5；最后一个 app lane 由统筹接管完成），产物已集成到工作副本；未 commit、未 push、未发布。所有结论以 macOS arm64 本机实际命令为准，Windows/Metal 未验证。

## 当前已验证状态（2026-11-25，文档同步轮）

本节于 2026-11-25 在 `J-X-C-Z/架构` worktree 上按当前源码重新核对。架构 lane 的基线是 `8118a4a`；同步过程中并发 agent 已把工作落到 commit `f999acd`（`refactor(architecture): split engine contracts/services/jobs and app state groups`），工作树另有未提交的 `processing/mod.rs`/`mock/process.rs` 改动。**本轮只重跑了架构规则测试**；全量套件数字引用本副本已记录的实际运行基线（早于 `f999acd`），未在 `f999acd` 后重跑。

### 已重跑的命令

```text
$ cargo test -p aurora-engine --test architecture_rules --locked
running 11 tests
test arch_rule1_app_talks_only_through_image_engine ... ok
test arch_rule2_engine_has_no_ui_dependency ... ok
test arch_rule3_portrait_is_independent_of_engine ... ok
test arch_rule4_contracts_are_backend_neutral ... ok
test arch_rule5_cuda_ptx_includes_are_platform_gated ... ok
test arch_rule6_render_plan_and_contracts_do_not_import_process ... ok
test arch_rule7_no_dependency_cycle_among_workspace_packages ... ok
test arch_rule8_processing_does_not_import_mock_or_gpu_runtime ... ok
test arch_rule9_lifecycle_and_save_queue_stay_standalone ... ok
test arch_rule10_file_engine_services_borrow_the_engine ... ok
test arch_rule11_app_does_not_reference_engine_services ... ok

test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.10s
```

**当前最终结果是 11 passed / 0 failed。** 同步过程中出现过一次短暂回归：`f999acd` 曾把 `Plan` 等计划类型放进 `processing/plan.rs`，规则 8 的 standalone word `Plan` 立即命中（当时实测 10 passed / 1 failed）。随后的并发修复把同一组类型移到 `crates/engine/src/render_plan_types.rs`（`lib.rs:260` `pub(crate) mod render_plan_types;`），没有再削弱规则 8；重跑即 11 passed。记录这段是为了说明门禁确实会拦住“把渲染计划词汇塞进 `processing`”，而不是把中间失败抹掉。该命令同时是 `.github/workflows/ci.yml` 的 `Architecture rules` 步骤（`windows` job，位于 `Check all targets` 与 `Run synthetic tests` 之间，`--locked -- --test-threads=1`）。

### 全量套件基线（本工作副本，既有记录，未在本轮重跑）

| 套件 | 结果 | 既有失败（不是通过项） |
|---|---|---|
| `aurora-engine --lib` + `scripts/testdata-skips.txt` | 1893 passed / 29 failed / 94 ignored / 105 filtered | 29 个失败是既有环境/资产问题：缺外部 fixture 15、macOS `/var`→`/private/var` 规范化 7、CUDA 在 macOS 不可用 3、共享 `params.json`/目录 2、`portrait_historical_tone_provider` code/test 不匹配 2 |
| `portrait-app --lib` | 96 passed / 1 failed / 5 ignored | `state::import_edits_tests::eip_variants_fill_the_preview_edits_immediately`，缺外部 EIP fixture |
| `aurora-portrait --lib` | 675 passed / 2 failed | 两个缺真实资产：`kernels::face_width_414`、`plump_session::tests::three_real_assets_match_payload_and_net_hashes` |
| `architecture_rules` | **11 passed / 0 failed**（最终工作树，2026-11-25 重跑） | 无。`f999acd` 中间态曾因 `processing/plan.rs` 的 `Plan` 失败（10/1），并发修复移到 `render_plan_types.rs` 后恢复 11/0 |

29 + 1 + 2 个失败均为既有失败，不写成本轮通过，也不表示本轮回归。全量套件数字来自 `f999acd` 之前的工作树，未在提交后重跑，不能据此推断提交后的通过数；架构规则是唯一在本轮按最终树重跑的门禁。

### 当前源码确认已落地

- engine 模块：`contracts/`（`compute`/`frame`/`identity`）、`document.rs`、`render_plan.rs`、`render_plan_types.rs`、`compute_bridge.rs`、`save_queue.rs`、`save_report.rs`、`job_lifecycle.rs`、`file_engine_services.rs`、`processing/`（`clarity`、`color`、`encoding`、`frame_map`、`geometry`、`grain`、`spots`、`statistics`、`tone`）均存在。`Plan`/`LayerPlan`/`MaskGrid`/`BandingPlan`/`PixelGains` 已从 `mock/process.rs` 移到 `render_plan_types.rs:24/268/354/359/392`（`lib.rs:260` 声明 `pub(crate) mod render_plan_types;`；写入时已重新核对）。
- `file_engine_services.rs` 10 个借用式服务：`CatalogService`、`SourceService`、`RenderService`、`InferenceService`、`CaptureService`、`ExportService`、`MergeService`、`PersistenceService`、`ImportService`、`LibraryService`；每个都是 `struct ...Service<'a> { engine: &'a FileEngine }`。
- app 子状态：`state/tasks.rs` `TaskCenter`、`state/device.rs` `DeviceState`、`state/preview.rs` `PreviewState`、`state/editor.rs` `EditorState`、`state/settings_state.rs` `SettingsState`、`state/library.rs` `LibraryState`；`AppState` 以 `tasks`/`device`/`preview_state`/`editor`/`prefs`/`library` 持有（`state/mod.rs:508-543`，`LibraryState` 在 `:513`）。`ViewPrefs` 与 `PanelPrefs` 在 `SettingsState` 内，路径为 `state.prefs.view` / `state.prefs.panels`。`filter`/`sort`/`sort_desc`/`nodes` 仍在 `AppState`。
- `mock/geometry.rs` → `processing/frame_map.rs`；`mock/mod.rs:85` 用 `pub(crate) use crate::processing::frame_map as geometry` 保持旧路径。
- 平台加载：`aimask/ort.rs`、`jxl.rs`、`heif.rs`、`heif_write.rs`、`raw.rs` 均在 POSIX 用 `dlopen`/`dlsym`，Windows 用 `LoadLibraryW`/`GetProcAddress`；非 Windows CUDA 由 `crates/portrait/src/cuda.rs:3495` 的 `mod no_driver` 提供同形不可用面。
- Z6III 中性算法：`aurora_raw/z6iii/mod.rs:7-13` 只对 `desktop`/OpenCL/CLI 加 Windows+feature 门控，`color`/`geometry16`/`reconstruct`/`render` 中性模块跨平台编译。
- Keystone Auto：`keystone.rs:8-10` 说明改为对真实 `KeystoneMap` 数值反解，`solve_axis_inverts_the_real_map` 覆盖反解。
- 保存队列已接线（源码优先于 `save_queue.rs:67` 的旧注释）：`file_engine.rs:2141/2285` 启动队列，`:3051` `save_adjustments_queued`、`:3072` `flush_saves`、`:3077` `save_queue_status`；app 在 `state/mod.rs:1219/1266` 与 `lib.rs:72` 提交并在 Quit 时有界 flush。

### 仍未完成

`Plan`/`LayerPlan`/`MaskGrid`/`BandingPlan`/`PixelGains` 已移到 `render_plan_types.rs:24/268/354/359/392`，架构规则 8 不再失败。`render_sized` 仍在 `mock/process.rs:1764`；`layer_plans`/`cached_layer_weights` 在 `mock/process.rs:937/959`，`apply_layers` 在 `:3479`；`mock/retouch.rs`（674 行）仍是未挂载孤儿。`LibraryState` 已落地（`state/library.rs:22`，`AppState.library` 在 `state/mod.rs:513`），但 `filter`/`sort`/`sort_desc`/`nodes` 仍在 `AppState`。`Geometry.distortion`/`optical_center` 与 `LensCorrection.uniform_light_grid`/`distortion` 仍在持久化 `AdjustmentSet`（`adjust.rs:1578/1590/1865/1868`），`RuntimeInputs` 只有 `producer_inputs`（`document.rs:207`）。Windows 原生 Release、DirectML/CUDA/OpenCL/Metal 真机、ONNX Runtime 存在性、P50/P95 性能与外部 fixture 均未验证。许可证 MIT vs Proprietary 需仓库所有者裁决。逐项证据、验收标准与下一步见 [REMAINING-GAPS.md](REMAINING-GAPS.md)。

## 本轮已实现

### A0 基线与平台可验证性

1. 非 Windows 构建/测试基线修复：`no_driver` CUDA 适配面、macOS ORT dlopen、`raw::LoadPath`、jxl/heif 显式拒绝、`heif.rs` 的 `kernel32` 门控、Windows CUDA 示例门控、`cuda_radiance` 对齐 UB 修复。此前 `cargo check` 失败、`cargo test` 无法链接；现在 `cargo check --workspace --all-targets --locked` 与 engine 测试二进制在 macOS 原生通过。
2. `ARCHITECTURE-BASELINE/OWNERSHIP/ADR/MIGRATION-MAP/VALIDATION/CHECKS` + 三份子域审计（boundary/frontend/registry）。

### A1 正确性与文档/运行时

3. **保存错误传播**（A-storage）：`SaveReport`（Saved/Failed/Superseded + index_error）、`SaveOrder` 同序列约束内存与磁盘、`ImageEngine::save_adjustments_batch_report`、app 显示逐文档结果。
4. **文档/运行时分离两步完成**（A-document/A-doc2）：`portrait_producer_inputs` 已从 `AdjustmentSet` 删除，由 `document::RuntimeInputs` 唯一持有并经 `RenderRequest.runtime` 传入；`document_fingerprint`、`DocumentSnapshot`、身份 newtype；`RuntimeInputs::fingerprint`。
5. **未知顶层存档字段 round-trip**：`AdjustmentSet::extensions`（`#[serde(flatten)]`），adjust/store 两级测试；旧 flat `schema_version` 不会被误当扩展。

### A2 公共计算契约

6. `contracts::{frame,compute,identity}`：PixelFormat/PlaneLayout/FrameDescriptor（checked 尺寸）、CoordinateSpace/ColorSpace/AlphaMode、BackendPolicy/ExecutionPlan/DispatchReceipt/FallbackReason、身份 newtype。19 个测试。

### A3 渲染与人像模块化

7. `render_plan`：复用 `NodeGraph::plan()` 的顺序/剪枝/警告，新增直接依赖、脏传播、后端等价缓存 key、执行计数；14 个测试。
8. `effect_descriptor`（aurora-portrait）：稳定 EffectId/PF/参数/输入依赖/三维能力；6 个已证据效果；12 个测试。
9. **机械迁移**：`mock::grain` → `processing::grain`（17 测试，diff 仅 5 行文档）；`clarity`/`spots` → `processing`（spots 16 测试；`crop_for_aspect` 等纯几何抽到 `processing/geometry.rs`，`mock::geometry` re-export，`mock::geometry` 33 测试）。架构规则 8 禁止 `processing` 出现任何 `mock::` 引用。

### A3/A4 质量与任务边界

10. **导出禁止放大 proxy**：>4096 raster 在导出 worker 内（scheduler 准入后）解码全分辨率；失败/超内存上限让 job 明确失败；≤4096 保持原路径；4 个测试（4500×3000 合成、≤proxy、失败不发布、内存上限）。
11. **Keystone 正确性修复**：未知焦距回退 `c=1`（C1 `UpdateGeo` 语义），4 个旧 `0.35·tilt` 断言按独立 f64 重建更新；`mock::geometry` 33 passed。
12. `save_queue`：有界单 writer、per-path revision、coalescing、flush/close、prune、O(1) status；11 个测试。**已接入 FileEngine 和 app**：`StoreSaveWriter`、`save_adjustments_queued`、`flush_saves`、`save_queue_status`（7 个测试）；`ImageEngine` 暴露队列方法，app 提交流程走队列，Quit 时 `flush_and_quit` 有界 flush；无队列引擎回退同步 report。
13. `job_lifecycle`：JobScope/JobId/JobState/CancelToken/JobError/Reservation/DispatchEvidence/JobRegistry；15 个测试。**已替换导出 gate**（`State.jobs`/`cancelled` 删除，`job_status`/`cancel_job` 保持，新增 `export_job_snapshot`）；**合并任务**改用 `JobRegistry` + `CancelToken`（`MergeProgress`/结果保留）；**解码 leader** 与**预览调用**留下生命周期记录（`decode_job_snapshot`/`preview_job_snapshot`，预览终态修剪）。AI 推理/缩略图专属记录未接入。
15. **第三轮补充**：21 个 color/tone 原语迁移到 `processing::color`（`mock::process` 122 tests、字节对拍）；Auto Keystone 删除 `INSET_PER_UNIT=0.35`，改为对真实 `KeystoneMap` 数值反解（`keystone` 34 tests）。
16. **第四轮**：`processing::tone`（HDR/亮度对比/dehaze 9 项）与 `processing::statistics`（luma/mean3x3/box blur/histogram 6 项）迁移，字节对拍通过；`MockProcess` 122 tests；合并/解码/预览接入 `job_lifecycle`（合并 11 tests、解码/预览记录测试通过）。
20. **AppState EditorState**：`edits`/active layer/node/clipboard/spot anchor 收进 `state/editor.rs`；undo/redo/gesture 行为不变（app 96 passed）。
24. **FrameMap 迁移**：`mock::geometry` → `processing::frame_map`，兼容别名保留；33 测试通过，为 `Plan`/`LayerPlan` 迁移铺路。
21. **Z6III 平台分层（第一刀）**：`z6iii` 中性算法（color/geometry16/CPU reconstruct/render helpers）在非 Windows 编译并测试（aurora_raw 83 passed）；desktop/OpenCL/CLI 仍 gate。**22. 剩余纯色彩 helper**：6 个 color/editor/range 函数迁入 processing::color；**23. App SettingsState**：settings/panels/engine_name/capabilities 拆出（view 仍留）。
19. **AppState TaskCenter**：`jobs`/`next_job`/`ai_busy`/`analysing` 收进 `state/tasks.rs`；新增 10 个手势/撤销/保存/队列分支测试（app 85 passed）。
17. **运行时字段身份**：`Geometry`/`LensCorrection` 手写 `PartialEq`，排除 `distortion`/`optical_center`/`uniform_light_grid`，文档相等性与持久化字段一致；2 个测试。
18. **FileEngine 服务边界**：`file_engine_services.rs` 的 `CatalogService`/`SourceService`/`RenderService`/`ExportService`/`MergeService` 借用引擎；catalog/source/preview/thumbnail、导出编排（start_export/job_status/cancel_job/export_job_snapshot）与 HDR/全景合并实现已移入，trait 方法一行委托；无行为变化，全量失败集不变。Persistence/Inference/Capture/Import 均已拆出；补齐 Thumbnail/Inference/Import 的 JobScope 记录。app 侧新增 `TaskCenter` 与手势/撤销/保存不变量测试（app 85 passed）。
14. 架构规则（第一轮 9 条，现为 11 条；新增 services 借用、app 不引用 services，processing 规则扩展）。

## 实际修改文件与公共接口

- 新增：`contracts/*`、`document.rs`、`render_plan.rs`、`save_report.rs`、`save_queue.rs`、`job_lifecycle.rs`、`processing/{grain,clarity,spots,geometry}.rs`、`portrait/effect_descriptor.rs`、`tests/architecture_rules.rs`、`scripts/architecture/check.sh`、`docs/architecture/*`。
- 修改：`engine.rs`（trait report）、`file_engine.rs`（SaveReport/SaveOrder/导出全分辨率/内存上限）、`store.rs`、`adjust.rs`、`preview_cache.rs`、`face_parsing.rs`、`mock/process.rs`（仅 producer 行）、`mock/mod.rs`、`mock/geometry.rs`、`lib.rs`、`heif.rs`、`raw.rs`、`aimask/ort.rs`、`portrait_stage.rs`（类型注解）、`portrait/src/{cuda,cuda_v2fuseoffset,lib}.rs`、`cuda_radiance.rs`、`n35_reuse_acceptance.rs`、`app/src/state/mod.rs`、`app/src/tools/portrait.rs`（测试修正）。
- 新公共 API：`SaveReport`/`DocumentSaveResult`/`SaveOutcome`、`save_adjustments_batch_report`、`contracts::*`、`document::*`、`render_plan::*`、`SaveQueue`/`SaveWriter`/`FlushReport`、`job_lifecycle::*`、`effect_descriptor::*`。

## 保留的旧 API、存档和画质约束

- `AdjustmentSet` 持久化字段与 `.portrait` schema version 1 不变；未知字段 round-trip。
- `ImageEngine` 原方法签名不变；同步保存仍是生产路径。
- `NodeGraph` 拓扑/渲染顺序、C1 参数、色彩链、PF 语义、CUDA PTX/MSL、模型资产未改。
- `processing` 迁移为文件移动 + re-export；现有 byte-parity/流程测试通过。
- Keystone 修复会改变无焦距数据文档的渲染结果，这是 C1 语义修正，不是性能优化；真实照片未对拍。

## Windows Agent / Mac Agent 协调与待交接

- 未改 `mock/gpu_runtime.rs`、`mock/detail*`、`mock/clarity_gpu.rs`、`mock/develop_gpu.rs`、`mock/nr_gpu.rs` 的实现。
- `no_driver` 是可替换 adapter 面；Metal 应实现同一 `contracts` 词汇。macOS 的 jxl/heif/ORT 仍是显式拒绝/可选 dlopen，需 Mac Agent 替换。
- 非 Windows C1 链会跳过 CA/高光恢复/抗锯齿/finish，**不等于像素等价**；需要真机/对拍。
- `contracts`/`render_plan`/`save_queue`/`job_lifecycle` 均未接入性能 Agent 的调度/驻留实现；接入前需评审 key、预算与状态语义。
- `keystone.rs` 的 `INSET_PER_UNIT=0.35` 与当前渲染器不一致，需其负责人重推导。
- 未访问另一台机器，未声称检查其进程/构建锁。

## 测试命令和实际结果

- `cargo check --workspace --all-targets --locked`：**通过**（macOS arm64）。
- `cargo metadata --locked --no-deps --format-version 1`：通过。
- 新增模块：contracts 19、document 11、render_plan 14、save_report/save 7、save_queue 11、job_lifecycle 15、effect_descriptor 12、architecture_rules 11、processing grain 17 / spots 16 / clarity 0+1 ignored(bench)、mock::geometry 33、导出 4、存档扩展 4、app save_notice 3。
- `aurora-engine --lib` + `scripts/testdata-skips.txt`：**1893 passed / 29 failed / 94 ignored / 105 filtered**。29 个失败为缺外部资产/ICC/C1（15）、macOS `/var` 规范化（7）、CUDA 不可用（3）、共享 params.json（2）、historical-tone 既有 code/test 不匹配（2）。**0 个 keystone/HEIF/JXL 失败**。
- `aurora-portrait --lib`：**675 passed / 2 failed**（缺真实资产）。
- `portrait-app --lib`：**96 passed / 1 failed / 5 ignored**（缺 EIP fixture/python）。
- `sh scripts/architecture/check.sh`：11 passed（本轮已重跑 `cargo test -p aurora-engine --test architecture_rules --locked` 确认）。

## 风险、失败项、未验证项

1. 保存已不走 UI 线程写文件：app 提交走队列，Quit 时有界 flush；剩余限制是队列 Drop 的 2s 排空与无 fsync（不声称断电持久性），以及 `export_job_ids`/`save_infos` 不自动修剪。
2. `job_lifecycle` 已替换导出 gate 并接入合并/解码/预览记录；AI 推理与缩略图专属记录未接入；`export_job_ids`/`export_outputs`、合并 records、解码 records 未自动修剪（预览记录已修剪）。
3. 导出 native RAW 取源仍在 `start_export` 且未计入 scheduler 预算；EIP 内含 RAW 且无 native 时显式拒绝。
4. 非 Windows C1 像素路径缺失 CA/高光/AA/finish；平台库名候选与本机 Homebrew 不匹配。
5. 29 个默认失败仍需 fixture 或平台修复；`scripts/testdata-skips.txt` 未覆盖全部。
6. ~~`keystone.rs` 的 auto-keystone 换算仍按 0.35~~：已改为对真实 `KeystoneMap` 数值反解，`keystone` 34 tests pass。
7. 许可证 MIT vs AGENTS Proprietary，仅报告未改。
8. Windows 原生 Release/测试、macOS Metal、真实模型、端到端画质/性能、`>4096` 大图真机内存均未验证。
9. `processing/clarity.rs` 的测试读取 `crate::render`/`crate::Source`（root re-export），是测试专用依赖；生产模块不依赖渲染器。
10. `processing/color.rs` 把 `linear_to_srgb`/`white_balance_gains` 以 `pub(crate) use` 保留在 `mock::process` 路径；这是兼容 shim，后续调用方应改到 `processing::color`。

## 下一批最小可交付改动

1. 把 `SaveQueue` 接入 `FileEngine`（`Arc<Store>`/`SaveWriter` 适配 + `save_adjustments_queued`），app 关窗有界 flush，保留同步兼容路径。
2. 把 `job_lifecycle` 适配到 export/preview/jobs，先并行记录再替换旧字段。
3. 已修正运行时字段的**身份语义**（不参与相等性）；下一步才是把它们真正搬到 `RuntimeInputs`，并让渲染路径显式接收。
4. `mock/process.rs` 纯数学块 → `processing::{color,tone,statistics,encoding}` 已完成；剩余 `render_sized` 内部、layer/retouch 与 `Plan`/`Source` 边界。
5. native RAW 取源移入 worker 并计入 scheduler；`keystone.rs` 0.35 换算重推导。
6. Windows 真机验证构建/GPU/导出；Mac Agent 替换 jxl/heif/ORT 并验证 Metal；补外部 fixture 或显式 `#[ignore]`。
7. 许可证与 `00_admin` ownership 文档由仓库所有者裁决/补齐。
