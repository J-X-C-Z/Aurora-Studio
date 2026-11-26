# 剩余重构缺口与接手清单（2026-11-25）

本文是**可分配的任务清单**，不是进度宣传。`file:line` 是 2026-11-25 在当前 worktree
（`J-X-C-Z/架构`，HEAD `f999acd`，另有并发未提交改动）上的观测快照。行号会漂移，
判断以文件与标识符为准；接手的 lane 应先重新 grep 再动手。

状态口径与 `ARCHITECTURE-VALIDATION.md` 相同：

- **源码缺口**：能从当前源码直接证明未迁移/未接线。
- **未验证**：本机没有该平台、资产或基准，不能声称通过。
- **需裁决**：需要仓库所有者决定，文档 lane 不自行修改。

## 最终门禁快照

2026-11-25 在最终工作树重跑：

```text
$ cargo test -p aurora-engine --test architecture_rules --locked
running 11 tests
test arch_rule10_file_engine_services_borrow_the_engine ... ok
test arch_rule7_no_dependency_cycle_among_workspace_packages ... ok
test arch_rule4_contracts_are_backend_neutral ... ok
test arch_rule6_render_plan_and_contracts_do_not_import_process ... ok
test arch_rule9_lifecycle_and_save_queue_stay_standalone ... ok
test arch_rule8_processing_does_not_import_mock_or_gpu_runtime ... ok
test arch_rule11_app_does_not_reference_engine_services ... ok
test arch_rule1_app_talks_only_through_image_engine ... ok
test arch_rule3_portrait_is_independent_of_engine ... ok
test arch_rule2_engine_has_no_ui_dependency ... ok
test arch_rule5_cuda_ptx_includes_are_platform_gated ... ok

test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.10s
```

历史说明：中间态 `f999acd` 曾把计划类型放进 `processing/plan.rs`，规则 8 的
standalone word `Plan` 命中（实测 10 passed / 1 failed）。随后同一组类型移到
`crates/engine/src/render_plan_types.rs`，规则恢复 11/0；**没有削弱规则 8**。

最新全量（2026-11-26，应用 `scripts/testdata-skips.txt` 的 106 个过滤器）：engine
**1911 passed / 29 failed / 94 ignored / 105 filtered**（失败集与既有 29 项一致）；app
**99 passed / 1 failed / 5 ignored**（既有 EIP fixture）。portrait 套件本轮未重跑。

## 已关闭（本轮并发/同步落地，不再作为任务）

| ID | 事项 | 当前证据 | 观察结果 |
|---|---|---|---|
| G1 | 计划类型移出 `mock/process.rs` | `Plan`/`LayerPlan`/`MaskGrid`/`BandingPlan`/`PixelGains` 在 `crates/engine/src/render_plan_types.rs:24/268/354/359/392`；`lib.rs:260` `pub(crate) mod render_plan_types;`；`mock/process.rs` 用 `use` 保持原名 | 规则 8 11/0；计划类型已在渲染计划层，不在 `processing`。验证：`cargo test -p aurora-engine --test architecture_rules --locked`。建议 lane：`crates/engine` |
| G4 | `LibraryState` 提取 | `crates/app/src/state/library.rs`；`AppState.library` 在 `state/mod.rs`；字段 `images`/`selection`/`current`/`recent_imports`/`recent_captures`/`trash`/`active_collection`/`active_collection_name`/`annotations`/`custom_order`/`stacks`/`open_stacks` | app suite 99 passed / 1 failed / 5 ignored（同一 EIP fixture 失败）；`filter`/`sort`/`sort_desc`/`nodes` 已随后由 G13 并入 |
| G11 | `save_queue` 头注释与源码不一致 | `save_queue.rs:67` 现写 “this queue is the production save path”；`file_engine.rs:2141/3051/3072/3077` 接线；app `state/mod.rs:1203/1252`、`lib.rs:72` | 已修复，不再是缺口。验证：`cargo test -p aurora-engine --lib save_queue --locked` 与 `queued_save_tests`。建议 lane：`crates/engine` storage |
| G5 | history/undo 纯状态转移 | `crates/app/src/state/editor.rs` 新增 `can_undo`/`can_redo`/`history_entries`/`undo_step`/`redo_step`/`jump_history_step`；`AppState` 保留同名薄协调层（`state/mod.rs`），顺序仍为 end_gesture → 状态转移 → after_commit → render/notify | 行为不变；`commit_invariant` 6、`state::editor::tests` 3 通过；app 99/1/5。建议 lane：`crates/app` |
| G13 | `filter`/`sort`/`sort_desc`/`nodes` 分组 | 已移入 `crates/app/src/state/library.rs`（类型/可见性/default 不变）；只改写真正指向 `AppState` 的接收者，未动 `NodeGraph.nodes` 等同名无关字段 | workspace check exit 0；app 99/1/5；`state::render` 19 passed / 5 ignored。建议 lane：`crates/app` |
| G10 | `write_atomic` 临时名与持久性 | `crates/engine/src/store.rs`：`.<file>.<pid>-<seq>.tmp` + `create_new` 重试，失败删除临时文件；新增 `write_atomic_durable`（Unix `sync_all` + 目录同步，best-effort，尚未接线）；批量保存仍明确非事务 | 新增 4 个测试（唯一临时名、8 线程同目标并发整块落盘、失败清理、durable 路径）；G10 当次 engine 1897/29/94/105（基线 1893 + 4）；最终全量见门禁快照；失败集不变，规则 11/0。Windows rename 与断电持久性未验证 |

| G3 | 运行时几何字段迁入 `RuntimeInputs` | `document::LensRuntime` 持有 `distortion`/`optical_center`/`uniform_light_grid`；`LensCorrection`/`Geometry` 已删除这些字段；`FileEngine::runtime_inputs` 按图提供；`render_with_runtime`/`render16_with_runtime`/`render_at_with_runtime`/`sample_point` 显式接收；`PreviewKey::with_runtime` 与 `dust_frame_key` 混入镜头指纹 | engine **1911 passed / 29 failed / 94 ignored / 105 filtered**（失败集不变）；app 99/1/5；规则 11/0；`mock::process::tests` 116 passed。旧 `.portrait` 仍可读（字段原本 `serde(skip)`）。建议 lane：`crates/engine` |
## 待接手项

全局约束（每项都适用）：

- 遵循 `AGENTS.md`：`c1-parse`、`portrait-parse`、`frontend-opus` 不在主树直接写，
  按 patch 交给统筹；`backend-opt` 的 GPU/性能改动也走 patch。
- 不改 `LICENSE`、不切换许可证；未获明确授权不提交/推送/发布（本清单本身由统筹
  lane 推送）；不动 `crates/engine/testdata/` 与用户照片/catalog/sidecar。
- 不通过删除或放宽测试来“修复”；`app` 只通过 `aurora_engine::ImageEngine` 调用引擎。
- 每项完成后必须附上实际命令输出；没有命令证据的项标成未验证。

### G2 — 把 `render_sized`、layer 合成与 retouch 决策迁出 `mock`

- 进展（2026-11-26，最终）：
  - **已迁出且已接线**：`clean_region` → `processing/sizing.rs`；`luma_range_weight` →
    `processing/layer_weight.rs`；`apply_layers`/`apply_layer_color_row`/
    `static_mask_can_apply`/`layer_weight`/`layer_pixel`/`local`、
    `cached_layer_weights`、`layer_mask_weight` → `crates/engine/src/layer_stage.rs`
    （`layer_stage` 已不依赖 `mock::process`）；`retouch_plane`/`union_plane`/
    `apply_retouch` → `processing/retouch_stage.rs` 并由渲染器调用。
  - `color_balance`、`black_white`、`bw_sector_y`、`bw_split_tone`、`bw_weights` →
    `processing/color.rs`；`mock/process.rs` 通过 `pub(crate) use` 保持测试与调用点不变。
  - 行为未改：`mock::process::tests` 116 passed；全量 engine 1911/29/94/105（失败集不变）；
    规则 11/0。GPU 路由与混合顺序未动。
  - **仍在 `mock/process.rs`**：`render_sized`、`layer_plans`、`mask_grid`/
    `cached_mask_grids`（都直接持有 `Source`/GPU 句柄，需先定义渲染层边界）。
  - **待 owner 裁决**：`crates/engine/src/mock/retouch.rs`（674 行，仍未被
    `mock/mod.rs` 声明）是删除还是接线。
- 目标：纯 sizing/layer 权重/retouch 逻辑不再定义在 `mock/process.rs` 或孤儿文件里。
- 触碰点（2026-11-26 快照，行号会漂移，先 grep）：`render_sized`、`layer_plans`、
  `mask_grid`/`cached_mask_grids` 仍在 `mock/process.rs`；layer 合成在
  `crates/engine/src/layer_stage.rs`；retouch 在
  `crates/engine/src/processing/retouch_stage.rs`；孤儿文件
  `crates/engine/src/mock/retouch.rs`（674 行，未被 `mock/mod.rs` 声明）。
- 验收：`render_sized`/layer 权重逻辑位于非 `mock` 模块并通过默认帧字节对拍；
  `retouch.rs` 要么被明确接线并有测试，要么作为 dead code 删除（先由 owner 裁决）。
- 验证：`cargo test -p aurora-engine --lib mock::process::tests --locked`；
  `cargo test -p aurora-engine --lib processing:: --locked`；
  默认帧字节测试（`default_render_bytes_are_the_pre_patch_bytes`）；
  `cargo test -p aurora-engine --test architecture_rules --locked`。
- 约束：先补 parity 测试再移动；不改混合顺序/像素；不把活跃 GPU 文件当机械搬迁对象；
  不要把 `Plan`/`Source` 词汇放进 `processing/`（规则 8）。
- 建议 lane：`crates/engine`（渲染/性能），与 backend-opt 协调。

### G6 — Windows 原生、GPU 真机与 ONNX Runtime 未验证

- 目标：给出 Windows 原生 Release 与 GPU/ORT 真机证据，或明确保持未验证。
- 证据缺口：本机 macOS arm64；`crates/engine/src/aimask/ort.rs:106-107` 有 POSIX
  `dlopen`/`dlsym`，但本机未安装/未运行 ORT；CUDA 在非 Windows 是
  `crates/portrait/src/cuda.rs:3495` 的 `no_driver`，操作返回 `Unavailable`。
- 验收：Windows 上 `cargo check --workspace --all-targets --locked` 与 engine 测试通过；
  DirectML/CUDA 细节阶段有设备名、驱动版本与实际输出；ORT 存在性单独记录。
- 验证：Windows 原生执行 `cargo check --workspace --all-targets --locked`、
  `cargo test -p aurora-engine --test architecture_rules --locked`、
  `cargo test -p aurora-engine --locked -- --test-threads=1`；GPU 路径附日志。
- 约束：无 Windows/设备时只能写 Unverified，不能推断通过。
- 建议 lane：infra / Windows 真机（blocked on hardware）。

### G7 — Metal 与 P50/P95 性能未实现/未测量

- 目标：Metal adapter 存在，且有冷暖帧、P50/P95、画质差异与比例测试。
- 证据缺口：无 Metal 实现；无基准产物（`ARCHITECTURE-VALIDATION.md` §4）。
- 验收：同一视口拖动/静止同分辨率同像素质量；记录冷/暖帧、P50/P95、VRAM/RSS；
  画质差异有指标。
- 验证：Windows/macOS 原生 Release 基准脚本 + 画质对拍；命令与原始输出入库。
- 约束：禁止缩放后上采样冒充优化；没有真机证据就标 Unverified。
- 建议 lane：infra / 性能 + Mac adapter（blocked on Metal）。

### G8 — 外部 fixture 测试未消除

- 目标：默认 CI 只跑 synthetic；缺资产项显式 `#[ignore = "external fixture"]` 或有可授权 fixture。
- 证据：engine 29 failed / app 1 failed / portrait 2 failed，均为既有失败
  （`ARCHITECTURE-VALIDATION.md` §3）；`scripts/testdata-skips.txt` 与失败集不完全一致。
- 验收：默认套件不再有“缺资产”失败；外部测试单独 feature 运行并记录。
- 验证：`cargo test -p aurora-engine --locked -- --test-threads=1`；
  `cargo test -p aurora-engine --features external-fixtures --locked -- --test-threads=1`（本机资产合法时）。
- 约束：不改 `crates/engine/testdata/` 外部资产、不把资产提交 Git；不把失败写成通过。
- 建议 lane：infra / owner（资产授权）。

### G12 — 公共面仍暴露内部类型（评审完成，零风险部分已落地）

- 目标：公共 API 与内部实现分离，trait 领域方法可评审。
- 评审记录：**`docs/architecture/PUBLIC-SURFACE-REVIEW.md`**（2026-11-26）。
  审计范围：60 条 root `pub use`/350 个显式名、30 个 root glob（约 796 item）、
  54 个 `pub mod`、`ImageEngine` 的 81 个方法；逐名 grep `crates/app/src`、
  `crates/engine/{tests,examples}`、`crates/portrait/{src,tests,examples}`，
  并以 `cargo check --workspace --all-targets` 定案常见词误报。
- 已应用（256 个显式名 + 1 个模块 re-export + 1 个 `pub mod` + 3 个 root glob）：
  - `portrait_stage` producer/历史色调/glare 内部（164+15+18 名）→ `pub(crate) use`；
    仅保留 engine 测试驱动的 32 个名字为 root `pub use`（pf30025/glare 测试）。
  - `body_shader_abi`(11)、`wrinkle_legacy`(23)、`mock::process` 帧类型(5)
    → `pub(crate) use`；`portrait_face_input_stage` 模块 re-export → `pub(crate)`。
  - `aurora_portrait::cuda_pf90021`(3)、`portrait_historical_tone_provider`(17)
    → 删除 root re-export（模块/上游路径仍在）。
  - `pub mod render_plan` → `#[allow(dead_code)] pub(crate) mod`（仅自测）。
  - `blemish_gainmap`/`portrait_makeup_bundle`/`portrait_makeup_provider`
    的 root glob 删除；三个模块仍 `pub`，调用点均走模块路径
    （`crates/engine/examples/blemish_gpu_bench.rs:10`、`blemish_env_off.rs:3`、
    `crates/engine/tests/makeup_owner_bytes.rs:3`）。
- 延后（见评审 §6）：`pub mod processing`、`c1_model`/`c1_masks`/`c1_styles`/
  `lensdb`/`layer_graph`/`matchlook`/`recipe` 的零调用名、`raw`/`aurora_raw`/
  `camera_params` 等 `pub mod`、`z6iii`、C1 catalog/sidecar 领域类型。原因：
  属稳定领域或主树外 c1-parse/portrait lane 的未验证接口，需 owner 裁决；
  本轮未验证主树外依赖。
- 验证（本 worktree）：`cargo check --workspace --all-targets --locked` exit 0；
  architecture_rules **11 passed**；engine lib **1911 passed / 29 failed / 94 ignored /
  105 filtered**（失败集不变）；app lib **99 passed / 1 failed / 5 ignored**
  （既有 EIP fixture）。`pub(crate)` 使 producer item 变为 dead code，engine lib
  warning 由 223 增至 282（可达性事实，非功能回归）。
- 约束：不新增 crate；不改 `LICENSE`；不恢复已收窄的 root re-export，除非 owner
  确认主树外生产者在用。
- 建议 lane：`crates/engine` + owner 评审。

### G9 — 许可证 MIT vs Proprietary（需裁决）

- 目标：由仓库所有者决定，不自行修改。
- 证据：`Cargo.toml:9` `license = "MIT"`；`LICENSE:1` MIT；`AGENTS.md:23` 要求 `Proprietary`。
- 验收：所有者记录裁决；若改许可证，单独 patch 并说明影响。
- 验证：无需命令；以 owner 书面决定为准。
- 约束：**本清单不授权任何 lane 改 `LICENSE` 或许可证字段。**
- 建议 lane：owner（blocked）。

## 接手顺序

按风险/依赖排序：

0. **确认架构规则保持 11/0**：并发 agent 已把 `Plan` 类型从 `processing/plan.rs`
   移到 `render_plan_types.rs`；先跑 `cargo test -p aurora-engine --test
   architecture_rules --locked` 确认，再开始其他项。不要在 `processing` 里重新放入
   `Plan`/`render_plan` 词汇。
1. **G3（运行时几何字段）**：语义最接近“文档/运行时”边界，已有相等性排除与 2 个
   测试作前提；先做收益最高，但必须先让渲染路径显式接收运行时输入。
2. **G2（`render_sized`/layer/retouch）**：机械迁移依赖 G3 的边界稳定；retouch
   先做 dead/alive 裁决。
3. **G5 + G13（app history / 库排序字段）**：已完成（app 99/1/5）。
4. **G12（公共面收窄）**：小而独立；G10（storage 临时名）已完成，仅 Windows
   rename 真机核对未验证。
5. **G6/G7/G9：blocked / unverifiable / needs-owner**。G6/G7 缺 Windows 真机、GPU 与
   Metal；G9 必须 owner 裁决。没有证据前只能保持 Unverified，不能写成通过。
6. **G8**：与 G6 一并由 infra/owner 处理；默认 CI 保持 synthetic，外部 fixture
   单独运行。
