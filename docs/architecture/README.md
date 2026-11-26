# 公共架构 lane 文档索引

首次接手先读[代码架构导览](CODEBASE-TOUR.zh-CN.md)，新一轮人像工作的验证边界见[人像 147 交接](PORTRAIT147-HANDOFF.zh-CN.md)。

下表是此前模块化单体的架构决策与迁移记录。它解释设计意图；哪些路径实际接入，仍要以目标分支源码和运行证据为准。

| 文档 | 内容 | 状态 |
|---|---|---|
| [ARCHITECTURE-BASELINE.md](ARCHITECTURE-BASELINE.md) | 基线 SHA、环境、规模、实际依赖、已确认问题 | A0 完成 |
| [ARCHITECTURE-OWNERSHIP.md](ARCHITECTURE-OWNERSHIP.md) | 三 Agent 边界、本轮文件所有权、交接与冲突 | A0 完成 |
| [ARCHITECTURE-ADR.md](ARCHITECTURE-ADR.md) | 9 条决策、备选方案、未决问题 | A0/A1 |
| [ARCHITECTURE-MIGRATION-MAP.md](ARCHITECTURE-MIGRATION-MAP.md) | 旧→新路径、patch 顺序、回滚、下一批任务 | A0/A1 |
| [ARCHITECTURE-VALIDATION.md](ARCHITECTURE-VALIDATION.md) | 每条实际命令、平台、结果、未验证项 | 随执行更新 |
| [ARCHITECTURE-CHECKS.md](ARCHITECTURE-CHECKS.md) | 可执行的依赖/平台规则与例外 | A-QA |
| [A-REGISTRY-AUDIT.md](A-REGISTRY-AUDIT.md) | 人像能力/证据状态重复点与统一接口 | A-registry |
| [A-BOUNDARY-AUDIT.md](A-BOUNDARY-AUDIT.md) | RAW/平台 cfg、native FFI、mock 渲染迁移清单 | A-boundary |
| [A-FRONTEND-AUDIT.md](A-FRONTEND-AUDIT.md) | AppState、命令/手势、导出与任务边界 | A-frontend |
| [ROUND-REPORT-2026-11-25.md](ROUND-REPORT-2026-11-25.md) | 本轮报告 + 2026-11-25 当前已验证状态与套件基线 | 随源码更新 |
| [人像 147 交接](PORTRAIT147-HANDOFF.zh-CN.md) | 当前可交付成果、指定基线、草稿和接手顺序 | 本次交接 |

代码落点（有的已接入应用，有的仅提供契约；调用链见导览）：

- `crates/engine/src/contracts/` — Frame/坐标/后端/回执/身份契约（A-core）。
- `crates/app/src/state/editor.rs` — EditorState/Edit/history 字段（gesture 逻辑仍在 AppState）。
- `crates/app/src/state/tasks.rs` — AppState 的 TaskCenter（jobs/ids/busy flags）与手势不变量测试。
- `crates/app/src/state/device.rs` / `preview.rs` / `settings_state.rs` — DeviceState、PreviewState、SettingsState（`settings`/`panels`/`view`/`engine_name`/`capabilities`）。
- `crates/engine/src/file_engine_services.rs` — FileEngine 的 10 个借用式服务：Catalog/Source/Render/Inference/Capture/Export/Merge/Persistence/Import/Library（trait 方法保留为委托）。
- `crates/app/src/state/library.rs` — `LibraryState` 的库状态与相关数据；交互入口仍由 `AppState` 编排。
- `crates/engine/src/document.rs` — 文档 vs 运行时输入、指纹、快照与身份 newtype（A-document）；`Geometry`/`LensCorrection` 的运行时字段已从相等性排除。
- `crates/engine/src/render_plan.rs` — 节点图编译、脏传播、缓存 key（A-graph）。
- `crates/engine/src/save_report.rs` — 批量保存逐文档结果（A-storage）。
- `crates/engine/src/save_queue.rs` — 有界单 writer 持久化队列；已通过 `FileEngine` 与 `ImageEngine` 接入，app 提交与 Quit flush 已切换。
- `crates/engine/src/jxl.rs` / `heif.rs` — POSIX dlopen 与 Windows DLL 双平台加载。
- `crates/engine/src/compute_bridge.rs` — gpu_runtime → contracts 的只读桥接。
- `crates/app/src/capability.rs` — 三维能力视图（静态/证据/运行时）。
- `crates/engine/src/job_lifecycle.rs` — 统一任务 id/状态/取消/回执协议；已接入导出任务（merge/preview 尚未迁移）。
- `crates/engine/src/processing/` — 从 `mock` 机械迁出的纯阶段：`grain`、`clarity`、`spots`、`color`、`tone`、`statistics`、`encoding`、`frame_map` 与共享 `geometry`；`mock` 保留 re-export。
- `crates/portrait/src/effect_descriptor.rs` — 效果 descriptor/registry 与三维能力状态（A-registry）。
- `crates/engine/tests/architecture_rules.rs` — 架构规则测试（A-QA）。

阅读顺序：BASELINE → ADR → MIGRATION-MAP → VALIDATION；具体子域看对应 AUDIT。
