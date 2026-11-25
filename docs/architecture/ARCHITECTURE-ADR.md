# 架构决策记录（ADR）

状态：A0/A1 第一轮。每条决策都标注“已实现 / 仅有设计 / 未验证”。

## ADR-001：模块化单体，暂不增加 crate 数量

- 决策：保留 `aurora-engine`、`aurora-portrait`、`portrait-app` 三个 package；领域边界先用 engine 内部模块 + `pub(crate)`/显式 re-export 表达。只有独立构建、稳定 API、依赖隔离或测试收益被证明时才抽 crate。
- 理由：当前首要问题是 `mock`/`file_engine`/`portrait_stage` 的职责混杂和反向依赖，不是 crate 数量。拆 crate 会增加编译/feature/交接成本，且会与性能/Mac Agent 的活跃文件冲突。
- 备选：立即拆 `aurora-core`/`aurora-compute`/`aurora-render` 等 11 个 crate。成本高、收益未证明，放弃。
- 状态：已实现（保留三 crate）。

## ADR-002：新公共契约放在 `engine::contracts`，零上层依赖

- 决策：`FrameDescriptor`、`PixelFormat`、`PlaneLayout`、`ColorSpace`、`CoordinateSpace`、`AlphaMode`、`BackendPolicy`、`ExecutionPlan`、`DispatchReceipt`、`FallbackReason`、身份 newtype 放在 `crates/engine/src/contracts/`。该模块只依赖 std + serde，不依赖 `mock`/`raw`/`aurora_portrait`/平台 FFI。
- 理由：Windows CUDA 与 macOS Metal 需要一个共同词汇表；把契约放在会被两边都依赖的 engine 内，避免循环依赖，也避免现在新增 crate。
- 约束：不把张量推理和显示互操作塞进同一个万能 Backend trait；AI provider 与 DisplayInterop 保持独立契约（后续 ADR）。
- 状态：已实现（见 `contracts/` 与架构检查规则 4）。

## ADR-003：文档与运行时分离采用两步 adapter，不重写存档

- 决策：第一步保持 `.portrait` schema（version 1）和 `AdjustmentSet` 字段不变，只修正“运行时输入参与文档身份”的错误：手写 `PartialEq` 忽略 `portrait_producer_inputs`；新增 `document_fingerprint`（仅持久化字段）与 `RuntimeInputs::fingerprint`（仅 producer）；新增 `DocumentSnapshot`（`Arc` 共享）和独立身份 newtype。第二步（已实现）把 producer 输入彻底移出 `AdjustmentSet`，由 `RenderRequest.runtime`/`RuntimeInputs` 唯一持有。
- 理由：直接改字段会同时影响 app、preview cache、mock render、face parsing 和所有存档测试；与性能 Agent 的活跃路径冲突。先修语义、后移数据，可独立回滚。
- 兼容：旧文件可读；未知顶层字段通过 `#[serde(flatten)] extensions` 原样 round-trip（A1 补充实现），已知字段不会重复进入扩展区。
- 状态：两步均已实现；`document` 11 个测试、`preview_cache` 9 个测试通过。`Geometry`/`LensCorrection` 的运行时字段（`distortion`/`optical_center`/`uniform_light_grid`）已从文档相等性排除（手写 `PartialEq`）；把它们移到 `RuntimeInputs` 仍需渲染路径改造。

## ADR-004：保存错误契约先行，异步写队列后置

- 决策：先把 `persist_batch` 的静默失败改成结构化 `SaveReport`（Saved/Failed/Superseded + index_error），保持 `save_adjustments_batch` 的 trait 签名；再用序号防止旧写覆盖新写。有界异步写服务是下一步，不在错误契约修好前引入队列。
- 理由：队列会掩盖原有可靠性问题；先让 UI 能显示“部分失败/索引失败/已排队”才有意义。
- 约束：锁内只做短状态更新；磁盘 I/O、索引写入在锁外；同一文档严格排序，不同文档不丢弃；`write_atomic` 的单文件原子语义保留。
- 状态：已实现（A-storage）；异步队列未实现。

## ADR-005：渲染图编译与像素执行分离

- 决策：`layers::NodeGraph` 仍是用户编辑拓扑；新增 `render_plan` 只做编译：输出可达性、确定性拓扑序、剪枝报告、依赖循环、脏节点传播、按 source/节点版本/参数子集/ROI/色彩空间/精度/算法版本/backend 等价标签生成缓存 key。像素执行、GPU 驻留和融合仍由现有渲染器/性能 Agent 负责。
- 理由：把“哪些节点该重算”变成可单测的纯逻辑，避免为了性能改变真实混合顺序；也让预览与导出未来共用同一编译逻辑。
- 状态：编译层已实现；尚未接入 `mock/process.rs`（刻意不做，避免与性能 Agent 冲突）。

## ADR-006：平台 driver 用“不可用适配面”而不是整模块 cfg 掉

- 决策：`aurora_portrait::cuda` 在非 Windows 提供与 Windows driver 同形的 `no_driver` 类型面，所有操作返回 `CudaError::Unavailable`；engine 的 GPU 阶段因此保持编译并走既有 CPU 回退，而不是把 `gpu_runtime`/`pixel_gpu`/`blemish_gainmap_cuda` 整块 cfg 掉。
- 理由：整块 cfg 掉会迫使调用点分散加 cfg，且丢失 VRAM 账本/回执等平台无关逻辑；不可用适配面保留了唯一契约，并为 Metal 适配器留下同形接口。
- 边界：这不是第二套实现；不分配、不启动内核。Strict-GPU 必须在计划层拒绝，而不是静默 CPU。
- 状态：已实现（非 Windows 构建通过）；Metal 未实现。

## ADR-007：人像效果用 descriptor/registry 统一，能力分三维

- 决策：`aurora_portrait::effect_descriptor` 定义稳定 `EffectId`、vendor PF 别名、参数规格、输入依赖、CPU reference、后端集合；能力拆成 **静态实现（是否编译）/ 证据（是否有对拍）/ 运行时可用（驱动/模型/资产/输入）** 三维，禁止压成一个 bool。先为已有证据的效果建立 descriptor，不迁移全部 PF。
- 理由：当前 `capability_registry`、`effect_router`、app 的 `contract.rs`/`evidence.rs`/`tools/portrait.rs` 各维护一份状态；三维分离能避免“文档写了等于现在能跑”。
- 状态：descriptor/registry 已新增；尚未替换现有路由与 UI。

## ADR-008：架构规则用普通 Rust 集成测试执行

- 决策：`crates/engine/tests/architecture_rules.rs` 读取源码树，断言 app 不触达私有引擎/GPU、engine 不依赖 app/gpui、portrait 不依赖 engine、contracts 不反向依赖、PTX include 有明确 allowlist、三包无依赖环。配套 `scripts/architecture/check.sh`。
- 理由：无需新框架、无需 CI 权限、可在任何平台运行；比行数阈值更针对实际契约。
- 约束：不增加全仓 `cargo fmt --check`；许可证不一致只报告不自动失败。
- 状态：已实现。

## ADR-009：不引入 async runtime、微服务、内部 RPC 或 DI 框架

- 决策：继续用 Rust 类型、作用域明确的线程池、有界队列、`Arc` 快照。渲染用不可变文档快照；pending 只保留最新值。
- 理由：符合仓库现有 GPUI/rayon 架构和性能约束。
- 状态：已遵守。

## 未决问题

1. 许可证：`Cargo.toml`/`LICENSE` 是 MIT，`AGENTS.md` 要求 Proprietary。**需仓库所有者裁决**，本 lane 不修改。
2. 未知顶层存档字段的 round-trip 丢失；需要 `store` 层保留原始 JSON map 或在 schema 中加入 `extensions`。
3. `aurora_raw ↔ mock` 循环与 `recipe -> file_engine` 反向依赖；需要先与性能 Agent 协商渲染服务归属，再做机械迁移。
4. `ImageEngine` 95 个方法（含测试名误计，实际约 70 个领域方法）如何拆成 Library/Source/Edit/Render/Inference/Export/Persistence/Capture 服务；需要一次接口评审。

## ADR-010：身份与缓存词汇不重复定义

- 决策：`SourceId`/`DocumentRevision`/`RenderGeneration` 只在 `contracts::identity` 定义，`document` 通过 `pub use` 复用；`render_plan` 的 `CacheColorSpace`/`CachePrecision` 明确命名为“缓存输出编码/存储精度”，与 `contracts::ColorSpace`（含 camera-native/ICC）和 `contracts::Precision`（f32/f16/mixed 计算精度）区分。
- 理由：跨 lane 集成时发现同一 crate 出现两套同名身份类型和两套 ColorSpace/Precision；保留两套会让缓存 key 与公共契约漂移。区分命名后，两者可以各自演进，但不会误用。
- 状态：已实现（`document.rs` 复用 contracts 身份类型；`render_plan` 枚举重命名，14 个测试通过）。

## ADR-011：Keystone 按 C1 `UpdateGeo` 语义修正未知焦距回退

- 决策：`mock::geometry::KeystoneMap` 的 `c` 在 `focal_length <= 0` 时回退 **1**（归一化分支），只有 `flags` bit0 才走 `c = 0`；4 个旧 `0.35·tilt` 梯形断言按独立重建的 `UpdateGeo` 数值更新。
- 证据：`adjust.rs` 的 `Keystone::tilt_scale`/`focal_length` 文档、`c1_keystone_basis` 的 `c>0` 分支、C1 disassembly 注释；A-geometry lane 在仓库外用 f64 重建公式并复现了旧二进制输出，再核对新常数。
- 影响：无焦距数据的 keystone 文档不再落入忽略焦距的非归一化分支，渲染结果会变化；这是修正而非性能/画质优化。
- 后续：`keystone.rs` 的 `INSET_PER_UNIT=0.35` 已删除；Auto Keystone 改为对 `KeystoneMap` 的 `tan(θ)=C(aspect)·v` 数值反解（4:3 垂直 C≈0.004），检测算法与签名不变。
- 状态：已实现，`mock::geometry` 33 tests pass；真实照片/Windows 真机未验证。

## ADR-012：导出禁止放大 proxy，全分辨率解码在 worker 内完成

- 决策：导出请求的长边 > `SOURCE_LONG_SIDE`（4096）且没有 native RAW 时，`start_export` 只准备 `FullResExportRequest`（path/lens/尺寸）；worker 在 `ExportScheduler` 准入后解码原图全分辨率，再做尺寸下限校验与内存上限校验。解码失败、返回过小或超上限都让 `JobStatus::Failed` 并发布空；≤4096 保持原预览源路径。
- 理由：仓库不变量禁止“缩放后上采样冒充优化”。把解码放在 worker 内避免 UI 线程多秒阻塞；`FullResExportRequest` 的估算按 24 B/px，上限取 min(总 RAM/4, 可用/2)，未知时 2 GiB。
- 限制：native RAW 取源仍在 `start_export` 且未计入 scheduler 预算；EIP 内含 RAW、无 native 解码时显式拒绝；HEIF 依赖 libheif/`image` 可读；内存估算用扫描尺寸，若 entry 尺寸为 0 则上限检查可能不足。
- 状态：已实现；4 个合成测试（全分辨率、≤proxy、失败不发布、内存上限）通过；Windows 真机/大图实测未验证。

## ADR-013：保留同步保存，新增有界单 writer 队列作为下一步

- 决策：`save_queue` 独立实现（`SaveWriter`/有界 `SaveQueue`/per-path revision/`FlushReport`），不接线 `FileEngine`；同步 `save_adjustments_batch_report` 仍是生产路径。队列的 newest-revision 映射按路径常驻以保证“旧不覆盖新”，outcome 明细可由调用方 `prune_saved` 限制；`status()` O(1)，逐路径表用 `documents()`。
- 理由：先把错误契约与排序修好，再引入后台写；避免队列掩盖数据丢失。接线需要 `Store` 共享与 catalog upsert 适配，属于下一步。
- 状态：队列 11 个测试通过；已通过 `StoreSaveWriter` 接入 `FileEngine`（7 个测试）；`ImageEngine` 暴露 `save_adjustments_queued`/`flush_pending_saves`/`pending_save_status`，app 的提交走队列并在 Quit 时有界 flush（`flush_and_quit`）；无队列引擎仍走同步 report。

## ADR-014：任务生命周期先做词汇层，不替换现有 gate

- 决策：`job_lifecycle` 提供 `JobScope/JobId/JobState/CancelToken/JobError/Reservation/DispatchEvidence/JobRegistry`，复用 `contracts` 回执；不替换 `jobs`/`cancelled`/`MergeRegistry`/decode gate/export gate。
- 理由：现有 gate 有真实并发语义，直接替换风险高；先统一 id/状态/取消/代次，让 UI 与证据层有共同语言，再逐个适配。
- 状态：15 个测试通过；已接入导出任务（`export_jobs`/`job_status`/`cancel_job`/`export_job_snapshot`）、合并任务（`MergeRegistry` 使用 `JobRegistry` + `CancelToken`）、解码 leader（`decode_jobs`）与预览调用（`preview_jobs`，终态定期修剪）。AI 推理/模型加载与缩略图专属记录尚未接入；合并任务重复 cancel 现在返回 false（“新请求”语义），首次/终态行为不变。

## ADR-015：`processing` 只接收纯阶段，`mock` 保留兼容 re-export

- 决策：`grain`、`clarity`、`spots` 已迁到 `processing`；`spots` 依赖的 `crop_for_aspect`/`clean_rect`/`oriented_size` 等纯几何已抽到 `processing/geometry.rs`，`mock::geometry` 反向 re-export。`mock/mod.rs` 用 `pub(crate) use crate::processing::{...}` 保持旧路径。架构规则 8 禁止 `processing` 出现任何 `mock::` 引用（含相对路径），规则 9 禁止协议模块反向依赖具体实现。
- 理由：机械迁移先于 `process.rs` 内部拆分；每一步只移动文件/路径并保持现有 byte-parity 测试。
- 状态：grain 17、spots 16、`mock::geometry` 33 测试通过；`mock::process` 的默认帧字节测试通过；下一步是 `process.rs` 纯数学块。

## ADR-016：FileEngine 用组合式服务拆边界，不先拆 trait

- 决策：新增 `file_engine_services.rs`，`CatalogService<'a>`、`SourceService<'a>`、`RenderService<'a>` 借用 `&FileEngine`，拥有各自领域的实现；`ImageEngine`/inherent 方法保留为一行委托。服务不拥有状态、不引入第二把锁；`FileEngine` 仍实现全部 trait。
- 已搬入：catalog 查询、source 获取（单飞/解码 gate/缓存准入/来源原因）、render（preview/thumbnail/gamut）、export（`start_export`/`job_status`/`cancel_job`/`export_job_snapshot`）、merge（`merge_*`/`start_merge`/`merge_*_impl`）、persistence（`StoreSaveWriter`/`SaveOrder`/`persist*`/queue/flush）。
- 已搬入：InferenceService（compute_ai_mask/ai_input/eye_report + Inference job 记录）、CaptureService（tether/camera/capture/live view）、RenderService thumbnail 的 Thumbnail job 记录。
- 已搬入：ImportService（import/rescan + Import job 记录；`FileEngine::open` 使用不记 job 的 `rescan_inner`）。
- 已搬入：LibraryService（library_ops 的 move/copy/locate/synchronize/trash）。FileEngine 服务域全部拆出。
- 理由：先有可审查的组合边界，等接口稳定再考虑 trait；避免 `Arc<Mutex<FileEngine>>` 式服务。
- 状态：`cargo check --workspace --all-targets` 通过；全量 engine 1893/29 失败集不变（既有失败，见 VALIDATION §3）；export/merge/source/render/save 相关测试通过。10 个服务（含 `LibraryService`）均为 `engine: &'a FileEngine`，规则 10/11 通过。

## ADR-017：AppState 先拆 TaskCenter，并补手势/保存不变量测试

- 决策：把 `jobs`/`next_job`/`ai_busy`/`analysing` 收进 `state/tasks.rs` 的 `TaskCenter`，AppState 只协调；不引入 Entity 化或事件总线。手势方法拆出 `*_inner(Option<&mut Context>)`，测试可在无 GPUI 窗口时走同一保存/撤销路径。
- 理由：先拆更新频繁且有明确边界的任务状态；Preview/History/Device/Settings 仍留在 AppState，避免一次性大改。
- 测试：live gesture 1 undo/1 save、edit/undo/redo、批量编辑 1 次批量保存、queued/sync 分支、TaskCenter id/顺序/flags；app 85 passed / 1 known fixture failure。
- 状态：已实现；Preview/History/Device/Capture/Settings 待后续。

## ADR-018：平面帧契约与 GPU 运行时桥接

- 决策：`contracts::frame` 增加 `PlaneSpec`/`PlanarLayout`，`FrameDescriptor.planes: Option<PlanarLayout>`（serde 向后兼容）；新增 `compute_bridge` 把 `mock::gpu_runtime` 的 `FrameFormat`/`Failure`/`Launch` 映射为 `FrameDescriptor`/`FallbackReason`/`DispatchReceipt`。
- 理由：让公共契约表达真实的多平面 GPU 帧与回退原因，给 CUDA/Metal adapter 一个共同词汇；桥接是只读、`pub(crate)`、未接线。
- 限制：`F32Planes` 只描述逻辑全分辨率三平面，不声称与运行时 grid+2 plane 的 extents/embedded cells 字节一致；`Counters`/VRAM ledger 尚未映射。
- 状态：27 个 contracts 测试、8 个 bridge 测试通过。

## ADR-019：App DeviceState 与 DisplayTarget

- 决策：`camera`/`camera_settings`/`tether`/`tether_probing`/`window_hwnd`/`scale_factor` 收进 `state/device.rs`；显示探测改为接收 `DisplayTarget { Window(isize), None }`，Windows `win_icm` 仍用同一句柄，非 Windows 仍 no-op。
- 理由：隔离唯一的平台句柄耦合，为后续 Metal/macOS 显示 profile adapter 留接口；TaskCenter 之后第二个 AppState 子状态。
- 状态：app 96 passed / 1 known fixture failure；Preview/Editor 已拆，Catalog/Settings 仍未拆。

## ADR-020：架构规则扩到 11 条并进入 Windows CI

- 决策：新增规则 10（services 只借用、不拥有 FileEngine）、规则 11（app 不引用 services）；规则 8 扩展为 production 代码禁止 `Plan`/`Source`/`FrameMap`/`render_plan`；CI 在 `cargo check` 后运行 `architecture_rules`。
- 理由：把服务边界变成可失败的契约；不引入 fmt 门禁。
- 状态：最终工作树 11 条规则通过；YAML 保持原 job/权限。同步中间态 `f999acd` 曾把计划类型移入 `processing/plan.rs`，规则 8 因 `struct Plan`（`:359`）失败（10 passed / 1 failed）；随后类型移到 `crates/engine/src/render_plan_types.rs`，规则恢复 11/0，未削弱规则。

## ADR-021：HEIF 读取器在 POSIX/macOS 使用 libheif（dlopen）

- 决策：`heif.rs` 不再整体 `#![cfg(windows)]`；加载层分平台（Windows `LoadLibraryW`/`GetProcAddress`/`FreeLibrary`，POSIX `dlopen`/`dlsym`/`dlclose`），候选复用 writer 的 `candidate_paths`（`AURORA_HEIF_LIB` 独占，Homebrew 前缀 + 版本化 dylib）。解码/EXIF/XMP/颜色/错误文本不变。
- 理由：writer 已能在 macOS 写 HEIC，reader 仍是拒绝 stub，导致 3 个默认读取测试失败；真实 libheif 1.23.7 可用。
- 状态：macOS HEIF 测试 27 passed（含 3 个回归）；全量失败集 29 不变；Windows/Linux 未执行。

## ADR-022：App EditorState 提取

- 决策：`edits`/`active_layer`/`active_node`/`next_layer`/`spot_anchor`/`clipboard`/`clip_groups` 收进 `state/editor.rs` 的 `EditorState`；`Edit` 一并移入并 re-export，gesture/undo/redo 逻辑保留在 AppState（与 GPUI/render/engine 耦合），行为不变。
- 状态：app 96 passed / 1 known fixture failure；`state::render` 19、commit invariants 6、task center 4 全通过。剩余 Catalog/Settings/History helpers。

## ADR-023：JPEG XL 读取器在 POSIX/macOS 使用 libjxl（dlopen）

- 决策：`jxl.rs` 平台化加载（Windows `LoadLibraryW`，POSIX `dlopen`），候选复用 Homebrew `libjxl` 版本化 dylib，`AURORA_JXL_LIB` 优先；解码/颜色/尺寸逻辑不变。
- 证据：`cjxl` 生成的合成 JXL 经 `decode_cop` 回读通过（`cjxl_synthetic_roundtrip`）；3 个 jxl 测试通过。
- 状态：macOS 可用；Windows 路径未在本机执行。

## ADR-024：Z6III 中性算法与平台适配分层

- 决策：`aurora_raw::z6iii` 不再整体由 `cfg(all(windows, feature))` 关闭；color、geometry16、CPU reconstruct、render 线性 helper 在所有平台编译，`desktop`/OpenCL loader/CLI 保持 Windows/product gate。OpenCL stage-3 快路径仅 Windows，其他平台走既有 CPU `c1_stage3_planar`。
- 理由：平台限制下沉到真正的 Win32/OpenCL 边界；macOS 可编译并测试中性算法（aurora_raw 83 passed），Windows 代码路径仅为加法 cfg，未执行验证。
- 状态：macOS 编译/测试；Windows 未验证。

## ADR-025：App SettingsState

- 决策：`settings`/`panels`/`engine_name`/`capabilities` 收进 `state/settings_state.rs`；`tools/portrait.rs` 的 `s.panels` 改为 `s.prefs.panels`，不保留全局 Deref 兼容。
- 后续（2026-11-25 源码确认）：`ViewPrefs` 也已进入 `SettingsState`，路径为 `state.prefs.view`；`PanelPrefs` 为 `state.prefs.panels`。`SettingsState` 字段见 `settings_state.rs:12-20`，调用点如 `state/mod.rs:697`。原“view 暂留 AppState”的描述已过期。
- 状态：app 96 passed / 1 failed / 5 ignored（同一 EIP fixture 失败，不是本轮通过）。`LibraryState` 已落地（`state/library.rs:22`，`AppState.library` 在 `state/mod.rs:513`）；`filter`/`sort`/`sort_desc`/`nodes` 仍留 `AppState`（见 REMAINING-GAPS G13）。

## ADR-026：FrameMap/KeystoneMap 迁入 processing::frame_map

- 决策：`mock::geometry` 的 FrameMap/KeystoneMap/frame_to_source 与既有 `processing::geometry` 纯 crop helper 合并到 `processing::frame_map`；`mock/mod.rs` 用 `pub(crate) use crate::processing::frame_map as geometry` 保持所有旧路径。架构规则 8 相应不再把 `FrameMap` 视为 mock 耦合。
- 理由：FrameMap 是纯值几何，是 `Plan`/`LayerPlan` 迁移的前置；移动后 `processing` 内自洽，mock 只做兼容别名。
- 状态：33 个几何测试（新路径）通过；全量 1893/29 失败集不变。
- 后续（2026-11-25 源码确认）：`Plan`/`LayerPlan`/`MaskGrid`/`BandingPlan`/`PixelGains` 已移到 `crates/engine/src/render_plan_types.rs:24/268/354/359/392`（`lib.rs:260` `pub(crate) mod render_plan_types;`），`mock/process.rs` 通过 `use` 保持原名。中途曾短暂放进 `processing/plan.rs` 并触发规则 8；移到 `render_plan_types.rs` 后恢复 11/0。`render_sized`（`mock/process.rs:1764`）、`apply_layers`（`:3479`）、`mock/retouch.rs` 仍未迁出。

## ADR-027：AppState 五个子状态分组完成

- 决策：`AppState` 以组合字段持有 `tasks: TaskCenter`、`device: DeviceState`、`preview_state: PreviewState`、`editor: EditorState`、`prefs: SettingsState`（`state/mod.rs:508-543`）。每个子状态是普通数据持有者，不引入 Entity 化、事件总线或第二套模型；gesture/undo/redo、保存、渲染决策仍留在 `AppState`，因为它们同时触达 GPUI/render/engine。
- 理由：先拆有明确边界且更新频繁的状态，减少 `mod.rs` 的阅读与冲突面；保持 UI 仍只 observe `AppState`。
- 结果：Tasks/Device/Preview/Editor/Settings/Library 已拆。`LibraryState` 在 `state/library.rs:22`，字段为 `images`/`selection`/`current`/`recent_imports`/`recent_captures`/`trash`/`active_collection`/`active_collection_name`/`annotations`/`custom_order`/`stacks`/`open_stacks`，`AppState.library` 在 `state/mod.rs:513`。`ViewPrefs`/`PanelPrefs` 在 `SettingsState` 内（见 ADR-025 更新）。history helper、`filter`/`sort`/`sort_desc`/`nodes` 仍未拆。
- 状态：app 96 passed / 1 failed / 5 ignored（既有 EIP fixture 失败，不是本轮通过）；`state::render` 19、commit invariants 6、task center 4 通过。

## ADR-028：平台加载、Z6III 中性算法与 Keystone 反解的现状确认

- 平台加载：`aimask/ort.rs`、`jxl.rs`、`heif.rs`、`heif_write.rs`、`raw.rs` 采用双平台加载（Windows `LoadLibraryW`/`GetProcAddress`；POSIX `dlopen`/`dlsym`）。非 Windows CUDA 仍是 `crates/portrait/src/cuda.rs:3495` 的 `no_driver` 同形不可用面，不分配、不启动内核。对应 ADR-006/021/023。
- Z6III：`aurora_raw/z6iii/mod.rs:7-13` 只对 `desktop`/OpenCL/CLI 加 Windows+feature 门控；`color`/`geometry16`/`reconstruct`/`render` 中性模块跨平台编译。对应 ADR-024。
- Keystone：`keystone.rs:8-10` 明确不再使用固定 inset，改为对真实 `KeystoneMap` 数值反解；`solve_axis_inverts_the_real_map`（`keystone.rs:580`）覆盖反解。对应 ADR-011。
- 状态：以上仅在 macOS arm64 上完成编译/测试（`heif` 27、`jxl` 3、`aurora_raw` 83、`keystone` 34）。Windows 原生、真实 ONNX Runtime、OpenCL/CUDA 真机与真实照片对拍均**未验证**。
