# 验证记录（A0–A3 第一轮）

基线 SHA：`8118a4a`。所有结果来自本机 macOS 27.2.1 / Apple Silicon arm64 的实际执行。
未执行的项目明确标注，不推断为通过。

## 0. 状态口径

每条结论只有两种状态，禁止把 Unverified 写成 Verified：

- **Verified locally（macOS arm64）**：本节写出的命令确实在本机执行过，并记录了实际输出；行内注明命令。
- **Unverified**：没有在本机执行，或本机不存在该平台/资产。包括但不限于：Windows 原生 Release、P50/P95 性能、DirectML/CUDA/OpenCL/Metal 真机、外部 fixture 测试、ONNX Runtime 是否存在。

2026-11-25 文档同步轮只重跑了 `cargo test -p aurora-engine --test architecture_rules --locked`，最终结果 **11 passed / 0 failed**。同步中间态 `f999acd` 曾因 `processing/plan.rs:359` 的 `Plan` 出现 10 passed / 1 failed；并发修复把类型移到 `render_plan_types.rs` 后恢复 11/0。其余既有结果保留原记录，未重跑，也不升级为“本轮验证”。

| 项目 | 状态 | 依据 |
|---|---|---|
| macOS arm64 编译 / 原生链接 | Verified locally | §1、§3 的命令与输出 |
| 架构规则（最终工作树） | Verified locally | 2026-11-25 重跑：11 passed / 0 failed，见 §2；`f999acd` 中间态曾 10/1，随后修复 |
| engine / app / portrait 全量套件数字 | Verified locally（既有运行，早于 `f999acd`，本轮未重跑） | §3；失败逐项列出，不当作通过 |
| Windows 原生 Release 构建与测试 | Unverified | 本机无 Windows |
| DirectML / ONNX Runtime GPU | Unverified | 本机未安装 ORT；`AURORA_AI_PROVIDER` 真机路径未执行 |
| CUDA 细节阶段真机执行 | Unverified | macOS `no_driver` 返回 `Unavailable` |
| OpenCL（Z6III stage-3）真机 | Unverified | Windows-only 快路径未执行 |
| macOS Metal 后端 | Unverified（不存在） | 尚无 Metal 实现，只有 CPU 回退 |
| P50/P95、冷暖帧、VRAM/RSS | Unverified | 无性能基准产物 |
| 外部 fixture（`--features external-fixtures`） | Unverified | `crates/engine/testdata/` 为空 |
| 真实照片 / 画质对拍 | Unverified | 无合法资产与对拍记录 |

## 1. 构建

| 命令 | 结果 | 状态 | 说明 |
|---|---|---|---|
| `cargo metadata --locked --no-deps --format-version 1` | 通过 | Verified locally | 3 个 package |
| `cargo check -p aurora-portrait --lib --locked` | 通过 | Verified locally | 修复 `cuda.rs` 门控 + `no_driver` + `V2Warp` 导入后 |
| `cargo check -p aurora-engine --lib --locked` | 通过 | Verified locally | 修复 ORT/macOS、`raw::LoadPath`、jxl/heif stub、`portrait_stage` 推断后 |
| `cargo check --workspace --all-targets --locked` | 通过 | Verified locally | 修复 `n35_reuse_acceptance` 的 Windows-only CUDA 引用后；含测试与示例 |
| `cargo test -p aurora-engine --lib --no-run --locked` | 通过并链接成功 | Verified locally | 修复 `heif.rs` 无条件 `#[link(name="kernel32")]` 后，macOS 可原生运行测试 |
| `cargo build --release --locked` | **未执行** | Unverified | 本轮不抢占重型构建；无 Windows 真机 |
| Windows 原生 Release | **未执行 / 未验证** | Unverified | 本机无 Windows |
| macOS 原生 Release + Metal | **未执行 / 未实现** | Unverified | Metal 后端尚不存在；仅完成编译与 CPU 回退 |

## 2. 新增模块测试（全部实际执行；状态：Verified locally）

| 范围 | 命令（均带 `--test-threads=1`） | 结果 |
|---|---|---|
| 公共契约 | `cargo test -p aurora-engine --lib contracts::` | **19 passed, 0 failed** |
| 文档/运行时 | `cargo test -p aurora-engine --lib document::tests::` / `preview_cache` / `face_parsing` | **11 / 9 / 7 passed**（face_parsing 2 ignored） |
| 渲染图编译 | `cargo test -p aurora-engine --lib render_plan` | **14 passed, 0 failed** |
| 保存报告（纯类型） | `cargo test -p aurora-engine --lib save_report` | **6 passed, 0 failed** |
| 保存失败注入 + 顺序 | `... batch_report_separates_saved_and_failed_documents` / `a_stale_sequence_does_not_overwrite_newer_bytes` / `a_stale_report_does_not_restore_older_in_memory_edits` / `save_adjustments_reports_an_unwritable_adjust_directory` | 各 **1 passed** |
| App 保存提示 | `cargo test -p portrait-app --lib save_notice` | **6 passed, 0 failed** |
| 未知顶层字段 round-trip | `cargo test -p aurora-engine --lib unknown_field_tests` / `unknown_top_level_keys_survive` | **2 + 1 passed** |
| 人像 descriptor | `cargo test -p aurora-portrait --lib effect_descriptor` | **12 passed, 0 failed** |
| 架构规则 | `cargo test -p aurora-engine --test architecture_rules --locked` | **11 passed / 0 failed**（最终工作树，2026-11-25 重跑）。中间态 `f999acd` 曾因 `processing/plan.rs:359` 的 `Plan` 为 10/1；并发修复移到 `crates/engine/src/render_plan_types.rs` 后恢复 11/0，未削弱规则 8 |
| 迁移：grain | `cargo test -p aurora-engine --lib processing::grain` / `mock::detail` / `default_render_bytes_are_the_pre_patch_bytes` | **17 / 73 / 1 passed**；`git diff -M` similarity 98%，仅 5 行文档 |
| 架构规则脚本 | `sh scripts/architecture/check.sh` | 通过（11 条规则） |
| 任务生命周期 | `cargo test -p aurora-engine --lib job_lifecycle` | **15 passed, 0 failed** |
| 导出任务 registry | `cargo test -p aurora-engine --lib "export"/"job"`（A14b 报告） | 导出/job 相关通过；旧 `State.jobs`/`cancelled` 已删除 |
| 合并/解码/预览 lifecycle | `merge_async` 11 passed；`decode_jobs_record_success_and_failure` / `preview_jobs_record_terminal_calls_and_stay_bounded` 各 1 passed | 合并使用 `JobRegistry`，解码 leader/预览调用留下终态记录；预览记录 ≤64 |
| 保存队列 | `cargo test -p aurora-engine --lib save_queue` | **11 passed, 0 failed**（含 writer 阻塞时 status/flush 不阻塞、prune 保留排序） |
| FileEngine 队列接线 | `cargo test -p aurora-engine --lib queued_save_tests` | **7 passed, 0 failed**（round-trip+reopen、逐文档失败、Full 重试、revision 顺序、未知 image、同步路径不入队、阻塞时 status/flush） |
| App 队列保存 | `cargo test -p portrait-app --lib save_notice` | **6 passed, 0 failed**（accepted 静默、Full/Failed 提示、quit flush 报告） |
| HEIF 读取器（macOS） | `cargo test -p aurora-engine --lib heif` | **27 passed / 1 ignored**（真实 libheif 1.23.7；3 个回归修复） |
| 迁移：color | `cargo test -p aurora-engine --lib mock::process` | **122 passed / 10 ignored**；21 个原语迁移，函数体对比一致 |
| 迁移：tone/statistics | 同上 + HDR/dehaze/histogram 对拍 | tone 9 项、statistics 6 项迁移；`mock::process` 122 passed；`parallel_histogram...` 等通过 |
| 运行时字段身份 | `cargo test -p aurora-engine --lib runtime_identity_tests` | **3 passed**；LCC grid / distortion / optical_center 不再影响文档相等性 |
| 运行时字段迁移（G3） | `cargo test -p aurora-engine --lib document:: adjust:: processing::frame_map preview_cache`；`mock::process::tests` | 字段已从 `AdjustmentSet` 删除，由 `document::LensRuntime` 持有并经 `render_with_runtime`/`sample_point` 显式传入；`PreviewKey::with_runtime`/`dust_frame_key` 混入指纹；document 11、frame_map 33、preview_cache 9、process 116 passed；adjust 仅既有 NEF fixture 1 failed |
| App 状态分组（G5/G13） | `cargo test -p portrait-app --lib` | **99 passed / 1 failed / 5 ignored**（既有 EIP fixture）；history/undo 纯状态转移 + `LibraryState` 浏览器字段 |
| 原子写临时名（G10） | `cargo test -p aurora-engine --lib store::tests` | **34 passed / 1 ignored**；唯一临时名、8 线程同目标并发、失败清理、durable 路径 |
| FileEngine 服务 | 全量 suite + source/preview/export/merge/save/inference/capture/import/library 定向测试 | Catalog/Source/Render/Export/Merge/Persistence/Inference/Capture/Import/Library 全部拆出；**1911 passed / 29 failed**（2026-11-26 全量），失败集不变 |
| 迁移：frame_map | `cargo test -p aurora-engine --lib frame_map` | **33 passed**；`mock::geometry` 兼容别名；规则 8 通过 |
| Z6III 平台分层 | `cargo test -p aurora-engine --lib aurora_raw` | **83 passed / 13 ignored**（中性算法在 macOS 编译/测试；Windows/OpenCL 仍 gate） |
| 迁移：color helpers | `mock::process` + `color_editor_*` | 6 个纯色彩 helper 迁移，字节一致 |
| Thumbnail/Inference job | `thumbnail_jobs_record_...` / `compute_ai_mask_failure_...` / `eye_report_records_...` | 通过；终态记录有界 |
| Auto Keystone 反解 | `cargo test -p aurora-engine --lib keystone` | **34 passed**；删除 0.35 常数，按真实 `KeystoneMap` 解 |
| 导出全分辨率 | `... export_above_the_proxy...` / `... export_at_or_below...` / `... full_resolution_decode_failure...` / `... full_res_export_memory_guard...` | 各 **1 passed**（4500×3000 PNG：全分辨率输出、≤proxy 不额外解码、失败不发布、内存上限拒绝） |
| 迁移：clarity/spots | `cargo test -p aurora-engine --lib processing::clarity` / `processing::spots` | **0 passed / 1 ignored（benchmark）**、**16 passed**；`mock::geometry` 33 passed |
| 保存文档扩展 | `unknown_field_tests` / `unknown_top_level_keys_survive` / `legacy_flat_schema_version...` | 3 passed |

新增测试合计约 110 个（不含既有测试）。

## 3. 既有测试套件（本机、原生链接）

### 3.1 `aurora-engine --lib`（状态：Verified locally，2026-11-26 重跑）

- 不加过滤：**1749 passed / 136 failed / 83 ignored**。
- 应用 `scripts/testdata-skips.txt` 的 106 个过滤器：**1911 passed / 29 failed / 94 ignored / 105 filtered**（2026-11-26；较 1893 增加 18 个新测试，失败集不变）。
- 剩余 **29** 个失败按原因分类（均为本机环境或既有问题，不在本轮改动路径；表内合计 15+7+3+0+2+2=29）：

| 类别 | 数量 | 例子 | 是否本轮引入 |
|---|---:|---|---|
| 缺外部 fixture（`crates/engine/testdata/` 为空、ICC/C1/NEF 资产不在仓库） | 15 | `c1_cos::a_real_sidecar_round_trips_through_save`、`camera_profile::real_icm_parses_as_mft2_lut`、`recipe::the_icc_profile_is_embedded_in_the_file` | 否 |
| macOS `/var` → `/private/var` 规范化导致相对路径变绝对 | 7 | `file_engine::library_ops::move_folder_preserves_relative_paths` | 否（保存错误契约让它在更早处显式失败，而不是静默） |
| CUDA 在 macOS 不可用 | 3 | `mock::pixel_gpu::grain_matches_the_cpu_stage` | 否（`no_driver` 返回 Unavailable 是预期） |
| Keystone/透视断言失败 | **0**（原 4） | 根因 `a4b9d27` 的 `UpdateGeo` 基替换 + 未知焦距 `c` 回退错误；已修复并更新 4 个测试，`mock::geometry` **33 passed** | 本轮修复（有独立 f64 重建验证） |
| 共享 `params.json`/目录可用性等 | 2 | `param_contract::loads_all_82_contracts` | 否 |
| `portrait_historical_tone_provider` 两项 | 2 | skip 文件明确标注“真实 code/test 不匹配，必须修”，本轮未触碰 | 否（已知既有缺陷） |

注意：`scripts/testdata-skips.txt` 的注释称其覆盖 108 个失败，但当前树仍有 33 个未列出；该清单属于另一 lane 的资产，本轮未修改。

### 3.2 `aurora-portrait --lib`（状态：Verified locally，既有运行；本轮未重跑）

- **675 passed / 2 failed / 0 ignored**。
- 两个失败都缺外部真实资产：`kernels::face_width_414::tests::fixed_fixture_matches_recovered_414_bytes`、`plump_session::tests::three_real_assets_match_payload_and_net_hashes`（后者硬编码了 `<本机 Windows fixture 路径>` 的 Windows 路径）。
- 修复了一个真实的跨平台测试 UB：`cuda_radiance` 测试把 `include_bytes!`（对齐 1）直接转成 `&[f32]`，在 aarch64 上触发 `from_raw_parts` 对齐检查并 abort。改为解码到对齐缓存后，整个 lib 套件才能继续执行。

### 3.3 `portrait-app --lib`（状态：Verified locally，既有运行；本轮未重跑）

- **96 passed / 1 failed / 5 ignored**。
- `state::import_edits_tests::eip_variants_fill_the_preview_edits_immediately`：缺 EIP fixture。
- `tools::portrait::every_live_skin_and_teeth_effect_has_one_section`：原失败是测试自身漏算（`listed=30`，`live` 只串了 SKIN+TEETH=27；`SKIN_SECTIONS` 的 Eyes & Hair 3 项来自 `WIRED_EYES_EFFECTS`/`WIRED_HAIR_EFFECTS`）。已修正测试的 live 集合，30==30 通过；没有改 UI 列表。

## 4. 画质与性能证据（状态：Unverified）

- **没有**做像素对拍、PSNR/SSIM、冷暖帧、P50/P95、VRAM/RSS 或导出分辨率测试。
- 本轮所有代码改动都应是行为保持或纯新增；唯一语义修正是“文档相等性忽略 producer 输入”和“保存失败显式返回”，均有单元测试。
- 渲染图/契约/descriptor 尚未接入旧渲染路径，因此不改变当前像素输出。
- **未验证**：Windows 真机、macOS Metal、真实模型推理、>4096 导出、GPU 回退的实际像素一致性。

## 5. 本轮发现的既有质量/正确性阻塞（需负责人裁决）

1. ~~导出用 4096 proxy 放大冒充全分辨率~~：**已修复（ADR-012）**。>4096 raster 且无 native RAW 时，导出 worker 在 scheduler 准入后解码全分辨率，失败/超内存上限显式失败；≤4096 保持原路径。仍有残留：native RAW 取源仍在 `start_export` 且未计入 scheduler 预算，EIP 内含 RAW 且无 native 解码时显式拒绝。
2. ~~保存仍在 UI 线程同步写 N 个文件 + catalog 索引，无关闭窗口 flush~~：**已接线有界队列（ADR-013）**。`file_engine.rs:2141/2285` 启动 `SaveQueue`，app 在 `state/mod.rs:1219/1266` 提交并在 Quit 时有界 flush；同步 report 仍是兼容路径。残留：不做 `fsync`、批量非事务、Drop 排空 2s、`export_job_ids`/`save_infos` 不自动修剪。
3. ~~Keystone 纯数学测试 4 个失败~~：**已修复**。根因是 `a4b9d27` 的 `UpdateGeo` 基替换 + 未知焦距回退错误；已按独立 f64 重建更新 4 个断言，`mock::geometry` 33 passed，`keystone` 34 passed（§3.1）。
4. ~~UI wired 效果 30 个 vs section 27 个~~：**已修复**。是测试自身只串了 SKIN+TEETH；加入 EYES/HAIR 后 30==30，未改 UI 列表（§3.3）。
5. **许可证不一致**：`Cargo.toml:9`/`LICENSE:1` = MIT，`AGENTS.md:23` 要求 Proprietary。按约束未修改，需仓库所有者裁决（见 REMAINING-GAPS.md）。
6. ~~未知顶层存档字段 round-trip 丢失~~：已修复（`AdjustmentSet::extensions` flatten）。
7. **`crates/engine/testdata/` 为空**：默认测试在干净 checkout 上有 136 个失败；CI 的 skip 清单不完整。

## 6. 未执行 / 未验证清单

- `cargo test --workspace --locked`（全量，无过滤器）：未作为通过项记录；分段结果见 §3。
- Windows 原生 Release 构建与测试、DirectML/ORT GPU 路径、CUDA 细节阶段真机执行。
- ONNX Runtime 是否存在并可由 POSIX `dlopen` 加载：`aimask/ort.rs` 有 macOS 加载代码，但本机未安装/未运行，状态 Unverified。
- OpenCL（Z6III stage-3）真机、macOS Metal 后端（不存在）、Core ML、ARM64 原生依赖打包。
- 外部 fixture 测试（`--features external-fixtures`）：无合法本机资产；engine 29 / app 1 / portrait 2 个既有失败因此未消除。
- 端到端画质对拍、P50/P95 性能基准、冷暖帧、VRAM/RSS、UI 延迟/重绘计数。
- 另一台机器/其他 Agent 的进程、构建锁与未提交改动：无法读取，未声称检查。

## 7. 结论

- 非 Windows 构建从小红变为可编译、可链接、可跑测试；这是本轮最直接的可验证收益。
- 六个代码 lane 的实际产物（契约、文档/运行时、保存报告、图编译、效果 descriptor、架构规则）均通过各自测试，且未接入旧渲染路径，不改变当前像素输出。
- 这套改动**不能**被表述为“架构重构完成”：`Plan`/`LayerPlan`/`MaskGrid`/`BandingPlan`/`PixelGains` 已移到 `render_plan_types.rs`，`render_sized`、`apply_layers` 与 `mock/retouch.rs` 仍在 `mock/` 范围；`LibraryState` 已落地但 `filter`/`sort`/`nodes` 仍在 `AppState`；运行时几何字段仍在 `AdjustmentSet`；Windows/Metal/性能/外部 fixture 未验证。当前逐项状态与接手顺序见 [REMAINING-GAPS.md](REMAINING-GAPS.md)。
- “服务边界未拆”“导出放大问题仍在”“存档扩展字段仍丢失”是本记录早先版本的结论；当前源码中服务边界已拆（10 个借用式服务）、导出全分辨率已实现（ADR-012）、未知顶层字段已 round-trip。不要把旧结论当现状。
