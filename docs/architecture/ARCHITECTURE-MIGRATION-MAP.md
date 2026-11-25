# 迁移映射与补丁顺序

原则：每个 patch 独立可回滚；不混合机械移动、算法变更和性能优化；移动前先有兼容 re-export 与对拍/测试；不整体移动活跃文件。

## 1. 已完成 / 本轮 patch

### P0-0 非 Windows 构建基线（本 lane，纯平台门控）

| 旧 | 新 | 兼容性 | 回滚 |
|---|---|---|---|
| `aurora_portrait::cuda` 仅 Windows driver | 增加 `#[cfg(not(windows))] no_driver` 同形适配面，操作返回 `Unavailable` | Windows API 不变；非 Windows 从“不能编译”变为“可编译并回退” | 删除 `no_driver` 与对应 `pub use` |
| `ort.rs` 仅 Windows/Linux 加载 | Linux `dlopen` 分支扩展到 macOS + dylib 候选 | Windows DirectML 路径不变 | 收回 cfg 分支 |
| `raw.rs::start` 固定 `Vec<u16>` | `LoadPath` 别名 | Win32 宽字符路径不变 | 删除别名 |
| `jxl` / `heif` 仅 Win32 DLL | 非 Windows 同形拒绝模块（`lib.rs` 内） | Windows 不变；非 Windows 明确报错 | 删除 stub |
| `heif.rs` 无条件 `#[link(name="kernel32")]` | 模块整体 `#![cfg(windows)]` | 这是 macOS 测试无法链接的根因 | 恢复模块 |
| `portrait_stage.rs` 两处 `None` 推断 | 显式 `Option<()>` / `Option<String>` | 无行为变化 | 删除注解 |
| `n35_reuse_acceptance` 非 Windows 引用 CUDA 类型 | `#[cfg(windows)] mod win_example` + 非 Windows 说明 | Windows 基准行为不变 | 恢复文件 |
| `cuda_radiance` 测试对未对齐字节做 `&[f32]` 转换 | 解码到对齐缓存 | 仅测试；修复 aarch64 abort | 恢复测试 |

### P0-1 保存错误传播

| 旧 | 新 | 兼容性 |
|---|---|---|
| `persist_batch -> ()`，失败只 warn | `persist_batch -> SaveReport`（Saved/Failed/Superseded + index_error） | 私有函数，无外部 API 变化 |
| `save_adjustments_batch -> anyhow::Result<()>` 总是 Ok | 有失败时返回 Err（仍保留内存中的最新编辑） | trait 签名不变 |
| UI 只显示通用错误 | 新增 `ImageEngine::save_adjustments_batch_report`（默认逐文档，FileEngine 覆盖为单次 catalog 提交）；app 显示 saved/failed/superseded 与索引失败 | 纯新增默认方法，旧调用不变 |

后续（未实现）：`save_adjustments_batch_report` 提升为 trait 默认方法；有界单 writer 队列；per-document 串行化；关闭窗口有界 flush。

### P0-2 文档 / 运行时分离（第一步）

| 旧 | 新 | 兼容性 |
|---|---|---|
| `AdjustmentSet` 派生 `PartialEq`，producer 输入参与比较 | 手写 `PartialEq` 只比较持久化字段；producer 字段已彻底移出文档 | 存档格式不变；撤销/`is_default()` 语义修正 |
| 无文档身份 | `document_fingerprint`（仅持久化字段）、`RuntimeInputs::fingerprint`（仅 producer）、`DocumentSnapshot`、身份 newtype | 新增 API，旧调用不变 |

第二步（已完成）：`portrait_producer_inputs` 已从 `AdjustmentSet` 删除，由 `document::RuntimeInputs` 唯一持有；`RenderRequest.runtime` 携带；`preview_cache` 的 key 明确为文档-only，未来预览携带运行时输入时必须把 `RuntimeInputs::fingerprint` 混入 `PreviewKey`。

### P1-1 公共计算契约

新增 `engine::contracts::{frame,compute,identity}`。旧代码尚未使用；迁移路径：

1. 性能 Agent 评审 `BackendPolicy`/`FallbackReason` 与现有 `mock::gpu_runtime::Failure` 的映射。
2. Mac Agent 用 `BackendCapability{Metal}` 实现同一 `select_backend`。
3. `FrameDescriptor` 先用于跨平台 adapter 的边界校验，再替换 `FrameHandle` 的内部描述；不一次性改 `RgbaImage`/`render16` ABI。

### P1-2 渲染图编译

新增 `engine::render_plan`，只读 `layers::NodeGraph`。迁移路径：

1. 先用测试证明可达性/顺序/脏传播/缓存 key 语义（本轮）。
2. 在 `mock/process.rs` 旁路接入：编译计划只用于决定缓存命中，不改变混合顺序；与性能 Agent 协调后再让执行器消费计划。
3. 预览与导出共用同一编译结果后，才删除旧的临时缓存判断。

### P1-3 人像 descriptor

新增 `aurora_portrait::effect_descriptor`。迁移路径：

1. 为已有证据的少数效果建 descriptor（本轮）。
2. `capability_registry::rows()` 与 app 的 `contract.rs`/`evidence.rs` 改为从 descriptor 派生，保留旧 TSV/JSON 作为证据。
3. 路由仍由 `effect_router` 决定；descriptor 不执行效果。

### A3 第一刀：`mock::grain` → `processing::grain`

- 旧：`crates/engine/src/mock/grain.rs`，`mock` 内部 `pub(crate) mod grain`。
- 新：`crates/engine/src/processing/grain.rs`，`processing` 公开；`mock` 用
  `pub(crate) use crate::processing::grain` 保持 `super::grain` 路径兼容。
- 语义：只增加模块文档，算法/常量/测试逐字节不变（`git diff` 仅路径与文档）。
- 验证：`processing` 不依赖 mock/GPU（架构规则 8），grain 与 render byte-parity 测试。
- 回滚：把文件移回并与 `mock/mod.rs` 的 `mod` 声明一起恢复。
- 后续：`clarity.rs`、`spots.rs` 同法迁移，再拆 `process.rs` 的纯数学与 `render_sized`。

### A8 有界单 writer 保存队列（引擎侧）

- 新增 `crates/engine/src/save_queue.rs`：`SaveWriter`（write/commit_index）、有界 `SaveQueue`、
  每路径严格递增 `DocumentRevision`、`SubmitOutcome`、`FlushReport`/`PathState`、
  超时 flush、close/drop 排空。
- **已接线**（2026-11-25 源码确认）：`FileEngine` 启动队列并在 app 提交/Quit flush 时使用；同步 `save_adjustments_batch_report` 保留为兼容路径。详见 §1b 与 ADR-013。
- 下一步：用 `Store` 实现 `SaveWriter`，在 FileEngine 暴露 `save_adjustments_queued`
  （返回 Accepted/Superseded/Full）与 `flush`，app 关窗调用有界 flush 并显示
  Pending/Saved/Failed；同步 API 保留为兼容路径。
- 验证：队列 10 个测试（含 writer 阻塞时 submit/flush 不阻塞、失败/索引失败、旧版本不覆盖）。

### A14 统一任务生命周期（词汇层）

- 新增 `crates/engine/src/job_lifecycle.rs`：`JobScope`/`JobId`、`JobState`、`CancelToken`、
  `JobError`、`ResourceReservation`、`DispatchEvidence`（复用 `contracts` 回执）、
  `JobRegistry`（提交/转移/取消/代次/快照/修剪）。
- **导出 gate 已替换**：`FileEngine` 的 `State.jobs`/`cancelled` 已删除，导出任务使用 `JobRegistry` + `CancelToken`；`job_status`/`cancel_job` 行为保持，新增 `export_job_snapshot`。
- 未迁移：merge/preview/decode gate。
- 验证：`job_lifecycle` 15、导出/job 相关测试通过；全量失败集不变。

### 第三轮进行中

- **SaveQueue → FileEngine**：`StoreSaveWriter`（Store clone + `Arc<Mutex<State>>`）、
  `save_adjustments_queued`、`flush_saves`、`save_queue_status`；同步路径保持。
- ✅ **process 纯数学（第一片）**：21 个 color/tone 原语迁移到 `processing::color`；`mock::process` 保留私有 use + `pub(crate) use` 兼容路径；`mock::process` 122 tests、字节对拍通过。
- **Auto Keystone**：移除 `INSET_PER_UNIT=0.35`，用 `KeystoneMap` 数值反解 detected angle
  到 Keystone 单位，保证与 `a4b9d27` 的 `UpdateGeo` 基一致。

### 第五轮进行中：FileEngine 服务边界

- ✅ 新 `file_engine_services.rs`：`CatalogService`/`SourceService`/`RenderService`/`ExportService`/`MergeService`
  借用 `&FileEngine`；catalog/source/preview/thumbnail、导出编排、HDR/全景合并实现已移入；trait 方法一行委托。
- 约束已满足：单飞、解码 gate、预算、预览缓存/generation、job 记录、export staging/publish、merge 结果不变；全量失败集不变。
- ✅ Persistence/Inference/Capture/Import/Library 已拆出（`LibraryService` 使用 `library_ops` 的 `pub(super)` helper）；10 个服务均为借用式，规则 10 通过。

## 1b. 2026-11-25 文档同步确认的实际落点

以下按当前源码核对，不是计划；未完成的兄弟项仍列在 [REMAINING-GAPS.md](REMAINING-GAPS.md)。

| 旧 / 目标 | 当前落点 | 兼容方式 | 证据 |
|---|---|---|---|
| `mock/geometry.rs` 的 `FrameMap`/`KeystoneMap`/`frame_to_source` | `processing/frame_map.rs`（纯 crop helper 在 `processing/geometry.rs`，由 `frame_map` re-export） | `mock/mod.rs:85` `pub(crate) use crate::processing::frame_map as geometry` | 规则 8 通过；`frame_map` 33 tests |
| `ViewPrefs` 暂留 `AppState` | `state/settings_state.rs` 的 `SettingsState { settings, panels, view, engine_name, capabilities }` | 调用点改 `state.prefs.view` / `state.prefs.panels`，不保留全局 Deref | `settings_state.rs:12-20`；`state/mod.rs:697` |
| AppState 任务/设备/预览/编辑/设置字段 | `state/tasks.rs` `TaskCenter`、`state/device.rs` `DeviceState`、`state/preview.rs` `PreviewState`、`state/editor.rs` `EditorState`、`state/settings_state.rs` `SettingsState` | `AppState` 以 `tasks`/`device`/`preview_state`/`editor`/`prefs` 组合；named ops 仍留 `AppState` | `state/mod.rs:508-543` |
| AppState 目录/选择/排序状态 | `state/library.rs` `LibraryState`（`images`/`selection`/`current`/`recent_imports`/`recent_captures`/`trash`/`active_collection`/`active_collection_name`/`annotations`/`custom_order`/`stacks`/`open_stacks`） | `AppState.library`；`filter`/`sort`/`sort_desc`/`nodes` 仍在 `AppState` | `library.rs:22`；`state/mod.rs:513` |
| `mock/process.rs` 的 `Plan`/`LayerPlan`/`MaskGrid`/`BandingPlan`/`PixelGains` | `render_plan_types.rs:24/268/354/359/392`（`lib.rs:260` `pub(crate) mod render_plan_types;`） | `mock/process.rs` 用 `use` 保持原名 | 已落地；中途放进 `processing/plan.rs` 曾使规则 8 失败（10/1），移到 `render_plan_types.rs` 后恢复 11/0 |
| `FileEngine` 单体方法 | `file_engine_services.rs` 的 10 个借用式服务：`CatalogService`/`SourceService`/`RenderService`/`InferenceService`/`CaptureService`/`ExportService`/`MergeService`/`PersistenceService`/`ImportService`/`LibraryService` | 每个 `struct ...Service<'a> { engine: &'a FileEngine }`；`ImageEngine`/inherent 方法保留一行委托 | 规则 10 通过；`file_engine_services.rs:34/115/306/491/787/900/1190/1629/1849/2155` |
| Win32-only JXL/HEIF/ORT/libraw 加载 | POSIX `dlopen`/`dlsym` + Windows `LoadLibraryW`/`GetProcAddress` | 候选路径复用 Homebrew 版本化 dylib；`AURORA_*_LIB` 优先 | `jxl.rs:208-214`、`heif.rs:4-5/96-98`、`heif_write.rs:80-83`、`raw.rs:1286-1292`、`aimask/ort.rs:106-107` |
| 非 Windows CUDA | `crates/portrait/src/cuda.rs:3495` `#[cfg(not(windows))] mod no_driver`，操作返回 `CudaError::Unavailable` | 与 Windows driver 同形；engine GPU 阶段保持编译并走 CPU 回退 | ADR-006；规则 5 例外 |
| `aurora_raw::z6iii` 整体 Windows+feature gate | 中性 `color`/`geometry16`/`reconstruct`/`render` helper 跨平台编译；`desktop`/OpenCL/CLI 仍 gate | `z6iii/mod.rs:7-13` 只给 Windows 分支加 cfg | ADR-024；`aurora_raw` 83 passed |
| `keystone.rs` 的 `INSET_PER_UNIT=0.35` 固定换算 | 对真实 `KeystoneMap` 数值反解 `tan(θ)=C(aspect)·v` | 检测算法与签名不变；`solve_axis_inverts_the_real_map` | ADR-011；`keystone.rs:8-10`、`:580` |
| `save_queue` 作为独立未接线模块 | 已接线：`file_engine.rs:2141/2285` 启动，`:3051` `save_adjustments_queued`、`:3072` `flush_saves`、`:3077` `save_queue_status`；app `state/mod.rs:1219/1266`、`lib.rs:72` | 同步 `save_adjustments_batch_report` 仍是兼容路径 | ADR-013；`save_queue.rs:67` 注释现已写明这是 production save path（2026-11-25 核对，旧 “not wired” 注释已修） |

## 2. 下一批最小可交付（按依赖顺序）

| 序号 | 任务 | 依赖 | 风险 |
|---|---|---|---|
| 1 | `save_adjustments_batch_report` 提升为 trait 方法 + app 显示逐文档状态 | A-storage（本轮） | 低 |
| 2 | ~~未知顶层存档字段 round-trip~~ 已完成：`AdjustmentSet::extensions` flatten + adjust/store 测试 | ADR-003 | 低 |
| 3 | `write_atomic` 临时文件名加 PID/序号；可选 fsync/目录同步 | A-storage | 中：Windows rename 语义 |
| 4 | ~~把 `portrait_producer_inputs` 移出 `AdjustmentSet`~~ 已完成：`RuntimeInputs` 唯一持有，`RenderRequest.runtime` | ADR-003 第二步 | 低（无生产调用者） |
| 5 | `contracts` 与 `gpu_runtime` 的 Frame/账本映射 | 性能 Agent 评审 | 中：不能改变显存计费 |
| 6 | `render_plan` 旁路接入预览缓存 | 性能 Agent 排期 | 中高：缓存失效错误会改变画质 |
| 7 | `aurora_raw` 的 Z6III 算法/平台分层 | Mac Agent 交接 | 高：需真实 NEF 对拍 |
| 8 | `mock` 正式渲染迁出到 `render/processing` | 性能 Agent + 机械移动 | 高：必须先冻结活跃写入 |

## 3. 明确不做（本阶段）

- 不新建 11 个 crate；不改 package 名。
- 不重写 CUDA PTX / Metal MSL / C1 颜色链 / PF 语义。
- 不引入 async runtime、内部 HTTP/RPC、DI 框架。
- 不添加全仓 `cargo fmt --check`。
- 不修改许可证；不提交/推送/发布。
- 不把预览降分辨率或降低模型输入来“优化”。

## 4. 回滚方式

每个 patch 都是独立文件集合：

- 平台门控：删除 `no_driver`/`LoadPath`/jxl stub 与对应 cfg，恢复原文件。
- 保存报告：`persist_batch` 可恢复 `()` 并保留 warn（但不建议）。
- 文档语义：恢复派生 `PartialEq` 即可（不推荐，会带回身份 bug）。
- 新模块：删除 `lib.rs` 的模块声明即可完全退出，不影响旧路径。

## 5. 与另外两个 Agent 的交接点

- 性能 Agent：`gpu_runtime`/`detail_gpu`/`clarity_gpu`/`develop_gpu` 的调度与账本实现保持其所有权；本 lane 只新增非 Windows 适配面和契约类型。若其 patch 与 `cuda.rs` 冲突，保留双方非冲突实现，语义冲突交其核对。
- Mac Agent：`no_driver` 是可替换的 adapter 面；Metal 应实现同一 `contracts` 词汇，而不是修改 `mock/process.rs` 的算法。jxl/ORT 的 macOS stub 可由其替换为真实 `libjxl`/Core ML 适配。
