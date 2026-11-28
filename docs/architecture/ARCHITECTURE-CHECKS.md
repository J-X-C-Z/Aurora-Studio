# 架构检查（可执行规则）

状态：A-QA 第二轮（服务/组合边界 + CI 接入）。检查基于当前源码树（baseline HEAD
`8118a4a` 之上的工作树），以 `crates/engine/tests/architecture_rules.rs` 为准；
文档中的行号是撰写时的快照，规则本身按“文件 + 词”计数或按精确集合断言，不依赖行号。

## 这是什么，不是什么

- 是：11 条轻量、可证伪的静态检查，读取工作树源码并断言依赖/分层/组合边界规则。
- 是：一个独立集成测试，无新增 crate，只用 `std`，可在任意平台运行。
- 是：CI 的一步（见下方“CI 接入”），不依赖外部 fixture，不增加工具链。
- 不是：`cargo fmt --check` 门禁。仓库大量使用压缩长行，本检查不检查格式。
- 不是：完整的 `cfg` 求值、编译器或依赖解析器。规则 5 的“平台门控”是启发式
  检查，权威约束是其中那份精确文件清单；规则 8 的 `#[cfg(test)]` 跳过也是
  粗粒度花括号计数，不是词法器。
- 不是：Windows 原生运行时验证。这里的结论都来自源码/清单文本。

## 如何运行

```sh
sh scripts/architecture/check.sh
```

等价于：

```sh
cargo test -p aurora-engine --test architecture_rules --locked -- --test-threads=1
```

当前实际输出（macOS，11 条全部通过）：

```text
running 11 tests
test arch_rule10_file_engine_services_borrow_the_engine ... ok
test arch_rule11_app_does_not_reference_engine_services ... ok
test arch_rule1_app_talks_only_through_image_engine ... ok
test arch_rule2_engine_has_no_ui_dependency ... ok
test arch_rule3_portrait_is_independent_of_engine ... ok
test arch_rule4_contracts_are_backend_neutral ... ok
test arch_rule5_cuda_ptx_includes_are_platform_gated ... ok
test arch_rule6_render_plan_and_contracts_do_not_import_process ... ok
test arch_rule7_no_dependency_cycle_among_workspace_packages ... ok
test arch_rule8_processing_does_not_import_mock_or_gpu_runtime ... ok
test arch_rule9_lifecycle_and_save_queue_stay_standalone ... ok

test result: ok. 11 passed; 0 failed
```

测试名统一带 `arch_rule` 前缀，仅便于日志过滤；不是 harness 要求。数字编号跨轮
累加：规则 1–9 是 A-QA 第一轮，规则 10–11 是本轮新增，规则 8 本轮扩展。

## 规则一览

| # | 规则 | 防止什么 | 当前状态 |
|---|------|----------|----------|
| 1 | `crates/app/src` 只通过 `aurora_engine::ImageEngine` / 公共类型调用引擎 | UI 绕过 trait 直接抓 `FileEngine`、mock 管线、GPU 运行时或 CUDA/Metal 适配器 | 5 处已知例外（见下） |
| 2 | `crates/engine` 不依赖 `portrait-app` / `gpui` / `gpui-kit` | 引擎反向依赖 UI，破坏 headless/CI | 通过 |
| 3 | `crates/portrait` 不依赖 `aurora-engine` | 建立 `engine -> portrait` 的反向边，形成循环 | 通过 |
| 4 | `engine::contracts` 不引用 `mock` / `raw` / `aurora_portrait` / 平台 FFI / 原生 PTX-MSL | 后端中立契约被实现细节污染 | 通过 |
| 5 | 内嵌 CUDA PTX 的源码文件必须平台门控或有无 Windows 适配面 | 新增 GPU 文件悄悄破坏非 Windows 构建 | 精确清单 38 个文件 + 1 条例外（见下） |
| 6 | `render_plan` / `contracts` 不引用 `super::process` | 编译计划/契约耦合到 mock 像素管线 | 通过 |
| 7 | 三个 workspace package 之间无依赖环 | 依赖图成环，无法独立构建 | 通过 |
| 8 | 生产代码 `processing/` 不出现 `mock::`、`super::process`、`gpu_runtime`、`render_plan`、`RenderPlan`、`FrameMap`、`Plan`、`Source` | 迁移后的纯阶段偷偷依赖旧渲染器、编译计划或 `Source` 输入 | 通过（测试代码可建 `Source` fixture，见下） |
| 9 | `job_lifecycle.rs` / `save_queue.rs` 不依赖 `mock`/`file_engine`/`portrait_stage` | 协议层反向依赖具体实现，无法单测 | 通过 |
| 10 | `file_engine_services.rs` 只借用 `&FileEngine`，不拥有引擎 | 服务持有第二个引擎/第二把锁，或把方法重新塞回 `impl FileEngine` | 通过（无例外） |
| 11 | `crates/app/src` 不引用 `file_engine_services` | UI 依赖引擎内部组合方式，绕过 `ImageEngine` | 通过（无例外） |

### 规则 1：app 只走 `ImageEngine`

扫描 `crates/app/src` 的非整行注释代码，禁止词：

`FileEngine`、`aurora_engine::mock`、`gpu_runtime`、`portrait_stage::`、
`aurora_portrait`、`cuda::`、`metal::`。

`CUDA` / `Metal` 出现在用户可见文案或注释里（如 `contract.rs`、`ui/prefs.rs`、
`tools/portrait.rs`）不算违规；检查匹配的是代码中的模块/适配器形状。

**当前已知例外（生产代码，TODO）**——`FileEngine` 的具体构造：

- `crates/app/src/state/source.rs:7` `use aurora_engine::{FileEngine, ImageEngine, MockEngine};`
- `crates/app/src/state/source.rs:51` `FileEngine::open_catalog(path)?`（按扩展名分派 C1 / `.lrcat`）
- `crates/app/src/state/source.rs:53` `FileEngine::open(path)?`
- `crates/app/src/state/source.rs:117` `FileEngine::create_session(path, name)?`
- `crates/app/src/state/source.rs:122` `FileEngine::create_catalog(path, name)?`

这五处构造后立即擦除为 `Arc<dyn ImageEngine>`，后续处理仍走 trait；但按规则字面
仍是直接引用。TODO：在 `aurora_engine` 内提供工厂（例如 `open_source(path) ->
Box<dyn ImageEngine>`），把具体类型收回引擎。

**当前已知例外（仅测试）**：

- `crates/app/src/state/mod.rs:2986` 与 `:2993`，位于
  `#[cfg(test)] mod import_edits_tests`；测试需要真实 `FileEngine` 落盘。

例外按“文件 + 词 + 出现次数”钉住（`source.rs` 5 次、`state/mod.rs` 2 次），
而不是按行号。原因：app crate 由其他 lane 并行编辑，行号会漂移；但**新增**一次
`FileEngine` 会改变计数并立即失败。文档记录当前行号供人工审阅。

### 规则 2：engine 不依赖 UI

同时检查：

- `crates/engine/Cargo.toml` 的依赖键（含 `[dev-dependencies]`、
  `[build-dependencies]`、`[target.*.dependencies]`）不得出现
  `portrait-app` / `portrait_app` / `gpui` / `gpui-kit` / `gpui_kit`。
- `crates/engine/src` 的导入形状：`use gpui`、`gpui::`、`gpui_kit`、
  `extern crate gpui`、`use portrait_app`、`portrait_app::` 等。

`crates/engine/src/blemish_gainmap.rs` 里有 Windows 测试 fixture 路径字符串包含
`gpui` 字样，但那不是导入，因此不会被误报。

### 规则 3：portrait 不依赖 engine

`crates/portrait/Cargo.toml` 当前 `[dependencies]` 为空；源码中没有
`use aurora_engine` / `aurora_engine::`。`effect_router.rs:3`（`//!`）与
`cuda.rs:290`（`///`）只在注释里提到该名字，检查跳过整行注释。

### 规则 4：contracts 后端中立

扫描 `crates/engine/src/contracts/`，禁止：`crate::mock`、`crate::raw`、
`aurora_portrait`、`std::os::windows`，以及引用 `native/`、`.ptx`、`.metal`、
`.msl` 的 `include_str!` / `include_bytes!`。当前该目录只有 `mod.rs` 文档与类型
（`compute.rs`、`frame.rs`、`identity.rs`），全部通过。对应 ADR-002。

### 规则 5：CUDA PTX 平台门控

发现方式：递归扫描 `crates/engine/src` 与 `crates/portrait/src` 的 `.rs`，找出
`include_str!` / `include_bytes!` 参数含 `native/cuda` 或 `.ptx` 的文件。

精确清单（38 个，测试常量 `CUDA_PTX_FILES`）：

- `crates/engine/src`（6）：`blemish_gainmap_cuda.rs`、
  `mock/{clarity_gpu,detail_gpu,develop_gpu,nr_gpu,pixel_gpu}.rs`
- `crates/portrait/src`（32）：`cuda.rs` 及 `cuda_*.rs`
  （`cuda_beard`、`cuda_blue_vein`、`cuda_eye_bag`、`cuda_foot_blue_vein`、
  `cuda_gloss`、`cuda_hair_detail`、`cuda_hair_seam`、`cuda_hand_neutral_gray`、
  `cuda_hlf`、`cuda_lip_line`、`cuda_lut`、`cuda_makeup`、`cuda_mesh`、
  `cuda_mesh_attribute`、`cuda_model_preprocess`、`cuda_moisture`、
  `cuda_neutral_gray`、`cuda_oil_shine_blend`、`cuda_pf30025`、`cuda_pf8050`、
  `cuda_pf90021`、`cuda_radiance`、`cuda_skin_models`、`cuda_smile`、
  `cuda_stray_hair`、`cuda_teeth`、`cuda_teeth_blend`、`cuda_texture_blend`、
  `cuda_uvmap`、`cuda_v2fuseoffset`、`cuda_v2warp`）。

集合必须**精确相等**：新增或移除都会失败，强制有意识地更新清单。

门控判定（启发式）：文件满足其一即可——

1. 父模块声明前有 `#[cfg(windows)]`（`mock/clarity_gpu`、`detail_gpu`、
   `develop_gpu`、`nr_gpu` 由 `mock/mod.rs` 门控）；
2. 文件内含 `#[cfg(windows)]` 或 `cfg(target_os = "windows")`
   （`mock/pixel_gpu` 与全部 portrait `cuda*`）；
3. 在显式非 Windows 适配例外清单里。

**当前例外**：

- `crates/engine/src/blemish_gainmap_cuda.rs`：文件与模块声明都没有 Windows
  cfg；它无条件引用 `aurora_portrait::cuda`，而后者在
  `crates/portrait/src/cuda.rs` 提供 `cfg(not(windows))` 的 `Nvidia` 等桩
  （约 `cuda.rs:3495` 起，`Nvidia` 桩在 3506 附近）。这是“显式非 Windows
  适配面”，记录在测试常量 `NON_WINDOWS_ADAPTER_EXCEPTIONS` 中。若删除该桩，
  规则 5 和真实非 Windows 构建都会失败。

### 规则 6：render_plan / contracts 不碰 mock process

在 `render_plan.rs` 与 `contracts/` 中禁止 `super::process`、
`crate::mock::process`。当前两者都没有。对应 ADR-005（编译层刻意不接入
`mock/process.rs`）。

### 规则 7：三个 package 无依赖环

解析三份 `Cargo.toml` 的依赖表（含 target 条件依赖），仅保留彼此的内部包名，
做 DFS 环检测。当前边恰好是：

```text
aurora-engine  -> aurora-portrait
portrait-app   -> aurora-engine
```

`aurora-engine -> aurora-portrait` 是 Windows/macOS 共用的效果后端；反向边不存在。
测试还断言内部边集合精确等于上面两条，防止依赖被悄悄丢弃。

### 规则 8：`processing` 不依赖 mock / GPU 运行时 / 渲染计划 / `Source`

分两档扫描 `crates/engine/src/processing/`：

1. **所有代码**（含测试）禁止 `mock::`（含 `crate::mock` 与
   `super::super::mock` 等相对路径）、`super::process`、`gpu_runtime`。从 `mock`
   迁出的纯阶段必须保持只依赖 adjustment 数据；`mock` 可以用
   `pub(crate) use crate::processing::...` 反向兼容，但迁移后的文件不能反过来
   引用 mock。第一个迁移对象是 `processing::grain`。
2. **仅生产代码**（跳过 `#[cfg(test)]` 项）禁止 `render_plan`、`RenderPlan`、
   `FrameMap`，以及按标识符边界匹配的 `Plan`、`Source`。这些是拆分后的渲染计划
   与旧渲染器输入；纯阶段不应触碰。

标识符边界避免把 `Plane`（清晰度平面）误判为 `Plan`。`#[cfg(test)]` 跳过是
粗粒度花括号计数：只跳过被该属性门控的模块/函数/单行项。

**当前例外（仅测试）**：`crates/engine/src/processing/clarity.rs` 的
`#[cfg(test)] mod tests` 里有一个 `#[ignore]` 基准，构建 `crate::Source::from_rgba`
并调用 `crate::render` 来对比新旧清晰度耗时（生产阶段本身不引用 renderer）。
规则 8 的第 2 档跳过整个 `#[cfg(test)]` 项，因此该基准不算违规；生产代码里
新增同类引用仍会失败。

### 规则 9：job_lifecycle / save_queue 保持独立

`job_lifecycle.rs` 与 `save_queue.rs` 不得出现 `crate::mock`、`crate::file_engine`、
`crate::portrait_stage`。它们是可注入的协议（writer / registry），必须能脱离具体
存储与渲染实现单独测试；接入 FileEngine 由调用方完成，而不是协议反向依赖。

### 规则 10：服务借用引擎，不拥有引擎

扫描 `crates/engine/src/file_engine_services.rs`，禁止：

- `Arc<Mutex<FileEngine>>`、`Arc<FileEngine>`、`Box<FileEngine>`、`Rc<FileEngine>`；
- `impl FileEngine`（含 `impl FileEngine for` 与 `impl FileEngine {`）；
- 同时含 `static` 与 `FileEngine` 的行（全局引擎）；
- 同时含 `engine:` 与 `FileEngine` 但不含 `&` 的行（`engine: FileEngine` 字段）。

当前该文件的 10 个服务（`CatalogService`、`SourceService`、`RenderService`、
`InferenceService`、`CaptureService`、`ExportService`、`MergeService`、
`PersistenceService`、`ImportService`、`LibraryService`）都声明
`engine: &'a FileEngine`，无例外。`use super::library_ops::{...}` 只借用库操作
辅助函数（`pub(super) fn`），不拥有引擎。

### 规则 11：app 不引用引擎内部服务

扫描 `crates/app/src`，禁止出现 `file_engine_services`。该模块是引擎内部组合细节；
UI 只调用 `ImageEngine`。具体 `FileEngine` 构造仍由规则 1 的计数例外钉住。
当前无引用，无例外。

## 未实现的规则：锁不跨 `fs::` I/O（有意跳过）

本轮候选规则之一是“`library_ops.rs` / `file_engine_services.rs` 不得持有
`state.lock()` 跨越 `fs::` 调用”。静态文本检查无法可靠判定这一点：

- 锁可能是 `let st = engine.state.lock();`，`fs::` 在另一个函数或闭包里；
- `drop(st)`、作用域结束、`return` 都决定锁的生命周期；
- 把“同一 `fn` 体内同时出现 `state.lock()` 与 `fs::`”当作违规会误报大量
  先取锁、克隆、释放、再写盘的正常代码；
- 反之，任何不解析控制流的近似都会漏报真正的“持锁写盘”。

因此**不做**这条检查，也不把它写成“已通过”。当前依靠代码审阅：
`file_engine_services.rs` 的持久化路径（`StoreSaveWriter`、`persist_batch_at`）
刻意在 `save_adjustments` 前后短暂取锁，注释记录了锁顺序。若将来要自动化，
应使用带控制流的 lint / clippy 自定义检查，而不是子串匹配。

## CI 接入

`.github/workflows/ci.yml` 的 `windows` job 在 `Check all targets` 之后、`Run
synthetic tests` 之前新增一步：

```yaml
      - name: Architecture rules
        # Source-tree layering checks (crates/engine/tests/architecture_rules.rs).
        # No external fixtures and no extra toolchain: runs on the same stable
        # Rust installed above.
        run: cargo test -p aurora-engine --test architecture_rules --locked -- --test-threads=1
```

- 复用 `dtolnay/rust-toolchain@stable`（与 `Check all targets` 同一工具链）。
- `--locked` 与现有步骤一致。
- 该集成测试只读源码树，不需要 `testdata/` 或 `external-fixtures`。
- 未改动 workflow 的输入、权限或任何 secret；`schedule` 全量审计 job 不含此步
  （架构检查是每次 push/PR 的常规门禁）。

## 许可证不一致（仅报告，不自动强制）

- `Cargo.toml:9` 写的是 `license = "MIT"`（`workspace.package.license`）。
- `AGENTS.md:23` 要求“许可证保持 `Proprietary`。不要添加或切换到开源许可证。”

这两个来源当前互相矛盾。本检查**不**断言许可证字段，也不自动改写：
许可证变更需要仓库所有者裁决。这里只把不一致作为已知项报告出来，避免它被
当成“已通过检查”。

## 维护约定

- 新增 GPU/PTX 文件：更新 `CUDA_PTX_FILES`，并说明其非 Windows 路径。
- 新增 app 直接引用引擎具体类型：规则 1 会因计数变化失败；要么修实现，要么
  显式新增例外并在此文档登记 file:line 与 TODO。
- 新增 app 对 `file_engine_services` 的引用：规则 11 失败；改为走 `ImageEngine`。
- 新增服务：保持 `struct ...Service<'a>` + `engine: &'a FileEngine`；规则 10
  会拦住所有权形状。
- 新增 `processing` 阶段：生产代码不要出现 `mock::`、`gpu_runtime`、
  `render_plan`、`RenderPlan`、`FrameMap`、`Plan`、`Source`；测试里需要 fixture
  时放进 `#[cfg(test)]` 项。
- 不要给本检查加入 `cargo fmt --check`；格式化由人工/独立流程处理。
