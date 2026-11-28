# 剩余重构缺口与接手清单（2026-11-25）

> 本文来自分支 `J-X-C-Z/架构` 的提交 `d46a8dc`，该提交的代码部分已被主干后续重构取代，
> 只有本文档被单独带入主干。条目中的行号以当时为准，接手前请对照当前源码重新确认。

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

## 已关闭（本轮并发/同步落地，不再作为任务）

| ID | 事项 | 当前证据 | 观察结果 |
|---|---|---|---|
| G1 | 计划类型移出 `mock/process.rs` | `Plan`/`LayerPlan`/`MaskGrid`/`BandingPlan`/`PixelGains` 在 `crates/engine/src/render_plan_types.rs:24/268/354/359/392`；`lib.rs:260` `pub(crate) mod render_plan_types;`；`mock/process.rs` 用 `use` 保持原名 | 规则 8 11/0；计划类型已在渲染计划层，不在 `processing`。验证：`cargo test -p aurora-engine --test architecture_rules --locked`。建议 lane：`crates/engine` |
| G4 | `LibraryState` 提取 | `crates/app/src/state/library.rs:22`；`AppState.library` 在 `state/mod.rs:513`；字段 `images`/`selection`/`current`/`recent_imports`/`recent_captures`/`trash`/`active_collection`/`active_collection_name`/`annotations`/`custom_order`/`stacks`/`open_stacks` | app suite 96 passed / 1 failed / 5 ignored（同一 EIP fixture 失败）；`filter`/`sort`/`sort_desc`/`nodes` 仍在 `AppState`（见 G13） |
| G11 | `save_queue` 头注释与源码不一致 | `save_queue.rs:67` 现写 “this queue is the production save path”；`file_engine.rs:2141/3051/3072/3077` 接线；app `state/mod.rs:1203/1252`、`lib.rs:72` | 已修复，不再是缺口。验证：`cargo test -p aurora-engine --lib save_queue --locked` 与 `queued_save_tests`。建议 lane：`crates/engine` storage |

## 待接手项

全局约束（每项都适用）：

- 遵循 `AGENTS.md`：`c1-parse`、`portrait-parse`、`frontend-opus` 不在主树直接写，
  按 patch 交给统筹；`backend-opt` 的 GPU/性能改动也走 patch。
- 不改 `LICENSE`、不切换许可证；未获明确授权不提交/推送/发布（本清单本身由统筹
  lane 推送）；不动 `crates/engine/testdata/` 与用户照片/catalog/sidecar。
- 不通过删除或放宽测试来“修复”；`app` 只通过 `aurora_engine::ImageEngine` 调用引擎。
- 每项完成后必须附上实际命令输出；没有命令证据的项标成未验证。

### G2 — 把 `render_sized`、layer 合成与 retouch 决策迁出 `mock`

- 目标：纯 sizing/layer 权重/retouch 逻辑不再定义在 `mock/process.rs` 或孤儿文件里。
- 触碰点（2026-11-25 快照）：`render_sized` `crates/engine/src/mock/process.rs:1764`；
  `layer_plans` `:937`；`cached_layer_weights` `:959`；`apply_layers` `:3479`；
  `crates/engine/src/mock/retouch.rs`（674 行，`mock/mod.rs` 未声明 `mod retouch`）。
- 验收：`render_sized`/layer 权重逻辑位于非 `mock` 模块并通过默认帧字节对拍；
  `retouch.rs` 要么被明确接线并有测试，要么作为 dead code 删除（先由 owner 裁决）。
- 验证：`cargo test -p aurora-engine --lib mock::process --locked`；
  `cargo test -p aurora-engine --lib processing:: --locked`；
  默认帧字节测试（`default_render_bytes_are_the_pre_patch_bytes`）；
  `cargo test -p aurora-engine --test architecture_rules --locked`。
- 约束：先补 parity 测试再移动；不改混合顺序/像素；不把活跃 GPU 文件当机械搬迁对象。
- 建议 lane：`crates/engine`（渲染/性能），与 backend-opt 协调。

### G3 — 把运行时几何字段移出持久化 `AdjustmentSet`

- 目标：`distortion`/`optical_center`/`uniform_light_grid` 只存在于运行时输入，不进入存档身份与相等性。
- 触碰点：`LensCorrection.distortion` `crates/engine/src/adjust.rs:1578`；
  `uniform_light_grid` `:1590`；`Geometry.distortion` `:1865`；`Geometry.optical_center`
  `:1868`；`PartialEq` 已排除（`:75-80`、`:1628`、`:1871`）；`RuntimeInputs` 目前只有
  `producer_inputs`（`crates/engine/src/document.rs:206-207`）。
- 验收：字段从 `AdjustmentSet` 删除并由 `RuntimeInputs` 唯一持有；`RenderRequest.runtime`
  消费；预览缓存 key 混入 `RuntimeInputs::fingerprint` 或来源身份；存档 schema 仍可读。
- 验证：`cargo test -p aurora-engine --lib runtime_identity_tests --locked`；
  `cargo test -p aurora-engine --lib document:: --locked`；
  `cargo test -p aurora-engine --lib preview_cache --locked`；
  `cargo test -p aurora-engine --lib adjust:: --locked`。
- 约束：先确认渲染路径显式接收运行时输入，再删字段；旧 `.portrait` 文件必须可读。
- 建议 lane：`crates/engine`（document/render）。

### G5 — 把 history/undo 命名操作移出 `AppState`（并处理 G13 的库字段）

- 目标：undo/redo/history 的纯状态转移有独立归属；`AppState` 只协调渲染/保存回调。
- 触碰点：`can_undo` `crates/app/src/state/mod.rs:1044`；`history` `:1052`；
  `undo` `:1262`；`undo_inner` `:1267`；`redo` `:1282`；`redo_inner` `:1287`；
  `jump_history` `:1323`；undo/redo 栈在 `crates/app/src/state/editor.rs:15-24`。
- 验收：命名操作委托给独立 helper；gesture 仍是一步 undo/一次保存；行为与现在一致。
- 验证：`cargo test -p portrait-app --lib commit_invariant --locked`；
  `cargo test -p portrait-app --lib task_center --locked`；
  `cargo test -p portrait-app --lib state::render --locked`；
  `cargo test -p portrait-app --lib --locked`（记录既有 EIP fixture 失败）。
- 约束：改 UI 行为需单独 patch；app 只经 `ImageEngine`。
- 建议 lane：`crates/app`（frontend patch）。

### G13 — `AppState` 的 `filter`/`sort`/`sort_desc`/`nodes` 仍未分组

- 目标：浏览器过滤/排序/collection 树归入 `LibraryState` 或独立 `BrowserState`。
- 触碰点：`crates/app/src/state/mod.rs:514-517`；`LibraryState` 在
  `crates/app/src/state/library.rs:22`。
- 验收：字段迁移后 UI 读取路径不变，selection/current 不变量测试通过。
- 验证：`cargo test -p portrait-app --lib --locked`；`state::render`、commit invariants。
- 建议 lane：`crates/app`（frontend patch）。

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

### G10 — `write_atomic` 不是事务

- 目标：同目标并发写不再共用固定临时名；可选持久性增强。
- 触碰点：`crates/engine/src/store.rs:599-605`（固定 `<file>.tmp`、`fs::write`、`fs::rename`；
  无 `fsync`/目录同步）。
- 验收：临时名含 PID/序号；同目录并发写不互相覆盖；单文件 rename 原子语义保留；
  批量仍明确“非事务”。
- 验证：`cargo test -p aurora-engine --lib store --locked`；新增并发同目标写测试。
- 约束：Windows rename 语义需真机核对；不声称断电持久性除非有 fsync 证据。
- 建议 lane：`crates/engine`（storage）。

### G12 — 公共面仍暴露内部类型

- 目标：公共 API 与内部实现分离，trait 领域方法可评审。
- 触碰点：`crates/engine/src/lib.rs` 约 56 处 `pub use`；`crates/engine/src/engine.rs`
  约 97 个 `fn` 行；`portrait_stage`/vendor/CUDA 相关类型仍被 root re-export。
- 验收：内部类型改 `pub(crate)`；稳定边界才 re-export；`ImageEngine` 方法清单有评审记录。
- 验证：`cargo check --workspace --all-targets --locked`；
  `cargo test -p aurora-engine --test architecture_rules --locked`；app 编译/测试。
- 约束：先确认没有 app 调用点依赖被收窄的 re-export；不新增 crate。
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
3. **G5 + G13（app history / 库排序字段）**：纯 app 侧，行为保持，风险低；放在
   引擎边界稳定之后，避免 app patch 与引擎 patch 交叉。
4. **G10、G12（storage 临时名、公共面收窄）**：小而独立，可并行；G10 需 Windows
   rename 真机核对。
5. **G6/G7/G9：blocked / unverifiable / needs-owner**。G6/G7 缺 Windows 真机、GPU 与
   Metal；G9 必须 owner 裁决。没有证据前只能保持 Unverified，不能写成通过。
6. **G8**：与 G6 一并由 infra/owner 处理；默认 CI 保持 synthetic，外部 fixture
   单独运行。
