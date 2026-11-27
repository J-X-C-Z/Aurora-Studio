# 公共架构 lane：文件所有权与交接（A0）

基线 SHA：`8118a4a`。本文件描述**当前这一轮**的实际写入边界，不替代仓库所有者未提供的 `00_admin/OWNERSHIP-2026-11-14.md`。

## 人像模块 A（2026-11-27，独立于下述 A0 历史分工）

| 范围 | 唯一写入方 |
|---|---|
| 注册表、PF 分组/证据与参数单位 | 人像注册表维护者 |
| `portrait_stage` 及渲染入口的牙齿调度 | 人像 stage 维护者；其它 stage 修改由统筹在其交接后落地 |
| `ImageEngine` 与 `FileEngine` 的逐图探测 | 引擎探测维护者 |
| 面板、应用状态和异步失效 | 人像面板维护者 |
| 款式缩略图、色板及整妆选择组件 | 妆容选择器维护者 |
| 整妆配方读取、原子替换、打包颜色、丰盈旧配方兼容、模块声明与最终集成 | 人像模块统筹 |

不改动 RAW、显示后端、CPU/GPU 细节阶段和现有 Metal 适配；不在本模块替换日用安装。

## 1. 可用信息与缺口

- `AGENTS.md` 声明分工文件在 `../../00_admin/OWNERSHIP-2026-11-14.md`；该路径在本机不存在，无法读取。
- `AGENTS.md` 明确：c1-parse、portrait-parse、frontend-opus 不直接写主树；backend-opt 的 GPU/性能改动走 patch；`aimask/*`、`mock/detail*`、`portrait_stage.rs` 已有落地功能不得覆盖（ORT DirectML、CUDA 细节阶段）。
- 同步任务：`Aurora-Studio-Hybrid-Compute-Optimization-Prompt.md`（性能 Agent）、`Aurora-Studio-macOS-Native-Metal-Adaptation-Prompt.md`（Mac Agent）。
- 本机可见 worktree：
  - 主树 `<main worktree: 架构整理>`，分支 `架构整理`，SHA `7b6229d`。
  - `Mac-原生适配` worktree，分支同名，SHA `9bf908e`；`git diff main...` 为空（尚未落地提交）。
  - 本副本 `<worktree: J-X-C-Z/架构>`，分支 `J-X-C-Z/架构`。
- 未能访问另一台机器：不声称检查过其进程、构建锁或未提交改动。

## 2. 本轮写入所有权（子 Agent 互不重叠）

| 子任务 | 允许写入 | 明确禁止 |
|---|---|---|
| A-core（契约） | `crates/engine/src/contracts/{mod,frame,compute,identity}.rs` | 驱动、CUDA/Metal 实现、Cargo manifests |
| A-document（文档/运行时） | `crates/engine/src/document.rs`、`crates/engine/src/adjust.rs` | store/file_engine、前端控件、人像 producer |
| A-storage（持久化） | `crates/engine/src/save_report.rs`、`store.rs`、`file_engine.rs`（仅 save/persist 函数）、`crates/app/src/state/mod.rs`（仅 after_commit_many） | 人像、GPU 内核、`engine.rs`（trait 改动交统筹） |
| A-graph（渲染图） | `crates/engine/src/render_plan.rs` | `layers.rs`、`layer_graph.rs`、`mock/process.rs`、性能 Agent 调度实现 |
| A-registry（人像注册） | `crates/portrait/src/effect_descriptor.rs`、`docs/architecture/A-REGISTRY-AUDIT.md` | 未交接的 PF 实现、`capability_registry.rs`、`effect_router.rs` |
| A-QA（架构检查） | `crates/engine/tests/architecture_rules.rs`、`scripts/architecture/check.sh`、`docs/architecture/ARCHITECTURE-CHECKS.md` | 生产源码、Cargo.toml |
| 统筹（本 Agent） | `lib.rs` 模块声明、`docs/architecture/*`、跨 lane 集成 patch | — |

## 模块 B：同帧美体生成与消费

独立于目录模块，从 `portrait/zero-ort-baseline-20261126` 的 `536fdcf` 开发；不覆盖 C4 的推理后端或 RAW/细节路径。

| 写入者 | 写入面 |
|---|---|
| 姿态 ABI | `crates/engine/src/{aimask/body_pose.rs,body_shader_abi.rs}`、必要的 `crates/portrait/src/kernels/body_69slot_bridge.rs` 适配 |
| 解析与实例 | `crates/engine/src/{portrait_body_inputs.rs,aimask/mod.rs}` |
| 网格 / 稠密场 | `crates/engine/src/portrait_body_mesh.rs` / `crates/engine/src/portrait_body_fields.rs`，各自单写 |
| 验收 | `crates/engine/src/portrait_body_live_tests.rs` |
| 统筹 | `crates/engine/src/{portrait_body_producer.rs,portrait_body_stage.rs,portrait_body_warp.rs,portrait_stage.rs,mock/process.rs}`，以及 `crates/portrait/src/kernels/{body_geometry.rs,body_geometry_params.rs,hair_body.rs}` 的采样坐标和曲线复用；预览/宽位共同消费 |

## 3. 三个 Agent 的边界（按任务书 §1.1，待仓库所有者确认）

| 责任域 | 公共架构 Agent（本 lane） | 性能 Agent | Mac Agent |
|---|---|---|---|
| 公共类型/依赖边界 | 设计、兼容适配、集成 | 评审、提需求 | 评审、提需求 |
| CPU/GPU 调度、FrameHandle、资源账本 | 契约与迁移约束 | 主责现有运行时与高性能实现 | Metal/统一内存 adapter |
| RenderGraph | 图语义、节点/缓存契约 | 增量执行、驻留、优化 | Metal 后端接入 |
| FileEngine/持久化 | 服务拆分、正确性修复 | 性能需求 | 平台兼容验证 |
| 人像 | 效果/能力注册与 producer 契约 | 现有 CUDA/人像性能 | 现有 Metal/产品接入 |
| RAW | decoder/reconstructor 边界 | CPU/CUDA/OpenCL 优化 | ARM64/Metal/原生依赖 |
| GPUI | 状态边界与服务设计 | Windows 显示/性能 | macOS 显示/平台功能 |
| 构建与测试 | 公共 CI 契约与集成检查 | Windows 真机验证 | macOS ARM64 真机验证 |

未完成交接的文件保持原负责人所有权。本轮没有移动 `mock/process.rs`、`mock/gpu_runtime.rs`、`portrait_stage.rs`、`file_engine.rs` 的主体代码。

## 4. 本轮已触碰的敏感文件与理由

| 文件 | 改动 | 理由 | 对性能/Mac Agent 的影响 |
|---|---|---|---|
| `crates/portrait/src/cuda.rs` | Windows-only re-export 加门控；新增 `#[cfg(not(windows))] no_driver` 适配面 | 非 Windows 构建基线失败；GPU 阶段需要可编译的“不可用”回退 | Windows 行为不变；Mac Agent 可在此实现 Metal 适配或替换 |
| `crates/portrait/src/cuda_v2fuseoffset.rs` | `V2Warp` 导入加 `#[cfg(windows)]` | 非 Windows 编译错误；CPU 参考路径不需要 GPU 类型 | 无 |
| `crates/engine/src/aimask/ort.rs` | Linux `dlopen` 分支扩展到 macOS；增加 dylib 候选 | 非 Windows 构建失败 | 未改 DirectML/CUDA provider 逻辑；Mac Agent 可替换为 Core ML |
| `crates/engine/src/raw.rs` | `LoadPath` 类型别名 | 非 Windows 编译错误 | 无算法变化 |
| `crates/engine/src/jxl.rs` / `lib.rs` | 非 Windows 显式拒绝模块 | 非 Windows 构建失败 | Mac Agent 可替换为真实 libjxl 适配 |
| `crates/engine/src/portrait_stage.rs` | 两处 `Option` 显式类型 | 非 Windows 编译错误 | 无语义变化；未触碰 PF 路由 |
| `crates/engine/src/lib.rs` | 新增 `contracts`/`document`/`render_plan`/`save_report` 模块声明；`heif` 的 Windows/非 Windows 模块切换 | 新模块入口 + 非 Windows 链接基线 | 纯新增/门控，Windows 不变 |
| `crates/engine/src/heif.rs` | 顶部 `#![cfg(windows)]` | 无条件的 `#[link(name="kernel32")]` 使 macOS 测试无法链接 | Windows 不变；非 Windows 走显式拒绝 stub |
| `crates/engine/src/engine.rs` | 新增 trait 方法 `save_adjustments_batch_report`（默认实现） | 让 app 能看到逐文档保存结果 | 纯新增默认方法，旧调用不变 |
| `crates/engine/src/file_engine.rs` | 新增 trait 覆盖，委托给 A-storage 的 inherent report 方法 | 避免 `Arc<dyn ImageEngine>` 退回 N 次单写 | 与 A-storage 同一 patch |
| `crates/portrait/examples/n35_reuse_acceptance.rs` | 主体包进 `#[cfg(windows)] mod win_example`，非 Windows 打印说明 | 示例引用 Windows-only CUDA 类型 | Windows 行为不变 |
| `crates/portrait/src/cuda_radiance.rs` | 测试改为解码到对齐缓存 | `include_bytes!` → `&[f32]` 在 aarch64 是 UB 并 abort | 仅测试；算法不变 |

**协调要求**：任何 Agent 若要在上述文件继续修改，先合并本 patch；不得用整文件 ours/theirs 覆盖。若性能 Agent 正在改 `gpu_runtime`/`detail*`，本 lane 的改动只是新增非 Windows adapter，不与其冲突。

## 5. 冲突与交接状态

| 接口 | 状态 | 下一步 |
|---|---|---|
| `ImageEngine::save_adjustments_batch` | 保持签名；A-storage 在 FileEngine 内新增 report 方法 | 统筹决定是否把 `save_adjustments_batch_report` 提升为 trait 默认方法 |
| `AdjustmentSet` serde 形状 | 不变；producer 字段已移出，`RuntimeInputs` 唯一持有 | 下一阶段迁移其余 `#[serde(skip)]` 运行时字段 |
| `contracts::*` | 新增、未被现有代码使用 | 性能/Mac Agent 评审后，把 `mock::gpu_runtime` 的 FrameHandle/账本映射到该契约 |
| `render_plan::*` | 新增、只读 `layers::NodeGraph` | 性能 Agent 接入增量执行前先评审 key 语义 |
| `effect_descriptor::*` | 新增；未替换 `capability_registry` | 人像负责人确认 descriptor 字段与证据等级 |
| 许可证 | **不一致，仅报告** | 仓库所有者裁决（MIT vs AGENTS 的 Proprietary） |

## 6. 不可越过的红线（本轮遵守）

- 未 commit、未 push、未建远程、未发布。
- 未改许可证、未改 `crates/engine/testdata/`、未改用户图片/catalog/sidecar。
- 未降低预览分辨率/精度，未用假蒙版，未改 PF/色彩算法。
- 所有结论标注“已运行 / 未运行 / 未验证”。
