# Aurora Studio 仓库维护规范（给 Agent）

本文件约束所有在本仓库工作的自动化 Agent，人类贡献者另见
[CONTRIBUTING.md](CONTRIBUTING.md)。两者冲突时以更严格的一条为准。

## 0. 事实来源

- 结论以**当前源码和实际运行的命令**为准。历史报告、记忆、旧截图、其他 Agent 的
  交接说明只能当线索，引用前对照源码重新确认。
- 本仓库是**公开仓库**。写进 Git 的任何内容，包括提交信息和注释，都按"全世界可见"处理。

## 1. 不能进 Git 的内容

| 类别 | 放哪里 |
|---|---|
| 从第三方软件提取或解码的资产（LUT、网格、预设、图集、模型、着色器数据） | `private-assets/portrait/` |
| Capture One 提取数据（相机参数表、会话模板、ICC、曲线、风格） | `private-assets/c1/` |
| 依赖上述数据、或从第三方进程抓取的测试 fixture | `private-assets/fixtures/` |
| 逆向记录、采集/解包脚本、带厂商安装路径或接口信息的报告 | `private-assets/research/` |
| Capture One 样本目录与 RAW 样张 | `crates/engine/testdata/`（已忽略） |
| 个人照片、EXIF 序列号/GPS、本机用户名与绝对路径、客户文件名、凭据与 token | 任何地方都不写 |
| 构建产物、日志、dump、release 包 | 不提交 |

规则：

- `private-assets/` 的布局、构建开关和新增流程见 [private-assets.md](private-assets.md)。
  代码只能通过 `aurora_portrait::private_asset!` / `private_asset_str!` /
  `private_assets::path` 引用私有数据，缺失时返回 `Refusal::MissingInput` 或错误，
  **不 panic、不使用假数据、不悄悄跳过**。
- 依赖私有数据的测试加
  `#[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]`。
- 测试里需要本机文件时，路径从环境变量读，不写死 `C:\Users\<名字>\...` 一类默认值。
- 发现已在 Git 中的上述内容：停止扩散，报告给仓库所有者；不要自行改写 Git 历史或强推。
- 许可证暂不处理：不要修改 `LICENSE` 或 `Cargo.toml` 的 `license` 字段。

## 2. 文件放在哪

| 路径 | 内容 |
|---|---|
| `crates/app` | GPUI 界面（包名 `portrait-app`，二进制 `aurora-studio`）。只通过 `aurora_engine::ImageEngine` 调用处理逻辑 |
| `crates/engine` | 处理引擎 `aurora-engine` |
| `crates/portrait` | 人像效果 kernel `aurora-portrait`，不依赖 GPUI |
| `crates/*/tests/fixtures/` | 只放自己生成的合成 fixture，并在 fixture 旁写明生成方式 |
| `build-support/` | 多个 `build.rs` 共用的构建逻辑 |
| `scripts/` | 构建、打包、CI 辅助脚本 |
| `docs/` | 文档，入口 [docs/README.md](docs/README.md)；按主题放子目录 |
| 仓库根目录 | 只放 README、CONTRIBUTING、AGENTS、CHANGELOG、LICENSE、ASSET-LICENSES、private-assets.md 与工作区配置。**不放报告和临时文件** |

不要新建 `work/`、`tmp/`、`REPORT-*.md` 这类目录或文件。一次性的调研结果写进
`docs/` 的对应主题，涉及私有数据时放 `private-assets/research/`。

## 3. 分支与提交

- 未获仓库所有者明确授权，**不提交、不推送**，不创建远程仓库、release 或上传产物。
  授权只对当次请求有效。
- 从最新的 `origin/main` 拉分支，命名 `<类型>/<简短描述>`，如 `fix/export-overwrite`、
  `feat/portrait-teeth`、`chore/repo-tidy`。不在 `main` 上直接改。
- 一个提交只做一件事。提交信息用 `类型(范围): 摘要`，类型取
  `feat`/`fix`/`perf`/`refactor`/`test`/`docs`/`chore`/`ci`；正文写清改了什么、为什么、
  跑了哪些命令、哪些没验证。
- 不改写已推送的历史，不 `push --force`，不在共享工作树执行 `reset --hard`、`clean -fd`。
  需要暂存时用带唯一标签的 `git stash push -m`，或临时 WIP 提交。
- 分支合并后由合并者删除远程分支。未合并的分支只有确认已被取代后才删，删除前在说明里
  记下分支名和最后的提交号。

## 4. 改代码

- 改动保持在任务范围内：不顺手修无关警告，不重排压缩长行，不为格式化改动稳定模块。
  不要给仓库加全量 `cargo fmt --check`。
- 平台相关代码用 `cfg(windows)` / `cfg(target_os = "macos")` 隔离，另一平台必须仍能编译。
- 以下已落地的路径，改动 `aimask/*`、`mock/detail*`、`portrait_stage.rs` 前先确认不覆盖：
  `aimask/ort.rs` 的 ONNX Runtime provider 选择（`AURORA_AI_PROVIDER`、`AURORA_ORT_PROVIDERS`），
  `mock/detail_gpu.rs` 与 `native/cuda/detail_gpu.ptx` 的 CUDA 细节阶段（`AURORA_DETAIL_GPU`）。
- 原图、catalog/session、sidecar 与导出目标只读，或先写临时文件再原子替换；
  不删除、不截断用户文件。
- 多个 Agent 并行时，同一文件同一时间只由一个 Agent 修改。当前分工记录在
  [docs/architecture/ARCHITECTURE-OWNERSHIP.md](docs/architecture/ARCHITECTURE-OWNERSHIP.md)
  和 [docs/macos-adaptation/OWNERSHIP.md](docs/macos-adaptation/OWNERSHIP.md)；跨边界改动先在那里登记。

## 5. 验证

提交前至少运行：

```powershell
cargo metadata --locked --no-deps --format-version 1
cargo check --workspace --all-targets --locked
cargo test --workspace --locked -- --test-threads=1
cargo build -p portrait-app --release --locked
```

- 动到私有资产引用时，有 `private-assets/` 和无 `private-assets/`（设置
  `AURORA_PRIVATE_ASSETS=` 为空）两种情况都要 `cargo check --workspace --all-targets --locked`。
- 默认测试必须使用合成数据，能在 GitHub CI 上跑。需要本机资产的测试放在
  `external-fixtures` feature 后或用上面的 `ignore` 标记：
  `cargo test -p aurora-engine --features external-fixtures --locked -- --test-threads=1`。
- 引擎与人像 crate 依赖 Windows 专有代码。非 Windows 主机可用
  `--target x86_64-pc-windows-gnu` 做类型检查；这不等于 Windows 原生验证，汇报时要写明。
- 没有实际运行的检查一律写"未验证"，不能从 DLL 存在、编译通过或本地单次结果推断通过。

## 6. 性能与画质

- 同一视口的拖动和静止预览必须同分辨率、同像素质量，禁止缩放后上采样冒充优化。
- 允许当前在途帧完成；pending 只保留最新值。旧图片和过期 generation 不得覆盖当前图片
  或更晚已展示的帧，最终帧必须使用最新值。
- 性能结论必须基于 Windows 原生 Release，附冷暖帧、P50/P95、画质差异和比例测试，
  细则见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 7. 文档

- 只用仓库内的相对链接。本机路径和其他工作区路径（如 `90_work/`、`06_reports/`、
  `../../0x_*`）对其他开发者不存在，不要写。
- 状态报告写明依据的提交号与实际命令；新增文档后更新 [docs/README.md](docs/README.md)。
- 移动或删除文件时，同步修正所有引用（代码里的 `include_bytes!`、文档链接、CI、脚本）。

## 8. 交接

每次交付（提交说明、PR 描述或对所有者的汇报）都包含：

1. 改了什么、为什么；用户可见的变化。
2. 实际运行的命令和结果，失败项原样贴出。
3. 未验证的部分与原因（平台、资产、真机、CI）。
4. 风险、已知限制，以及需要所有者决定的事项。
