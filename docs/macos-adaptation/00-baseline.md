# M0 — Mac 真机基线与适配清单

> 状态：进行中。所有数字来自本机实际命令，未运行的项一律标“未验证”。

## 1. 环境（2026-11-25）

| 项 | 值 |
|---|---|
| 机器 | MacBook Air（Mac14,2），Apple M2，8 核 CPU（4P+4E），10 核 GPU |
| 统一内存 | 16 GB |
| macOS | 27.2.1（build 26B131） |
| 显示 | 内置 Liquid Retina 2560×1664，主屏，Metal 4 |
| CPU 架构 | arm64（原生） |
| Rust | rustc 1.98.1 / cargo 1.98.1 |
| 已装 target | aarch64-apple-darwin、aarch64-unknown-linux-gnu、x86_64-*（host 为 arm64） |
| Xcode | /Applications/Xcode.app/Contents/Developer |
| Metal 工具链 | 可用（Metal.xctoolchain，Xcode 27.2） |
| macOS SDK | MacOSX27.2.sdk |
| 仓库 HEAD | `9bf908e53fd029c12f4f454546c61064f9ef9010`（J-X-C-Z/Mac-原生适配 worktree） |
| 主树 | 另一 worktree（main）@ `7b6229d`，会话期间前进到 `8118a4a` |

序列号、硬件 UUID、UDID 已脱敏，不写入仓库。

### Metal 设备实测（`MTLCreateSystemDefaultDevice`，2026-11-25）

| 项 | 值 |
|---|---|
| 设备 | Apple M2 |
| 统一内存 | `hasUnifiedMemory = true` |
| `recommendedMaxWorkingSetSize` | 12124 MiB（**不是额外显存**，是共享物理内存的建议工作集） |
| `maxBufferLength` | 9093 MiB |
| `maxThreadsPerThreadgroup` | 1024×1024×1024 |
| `maxThreadgroupMemoryLength` | 32768 B |
| GPU family | Apple7 / Apple8 支持 |

现有 MSL 可用 Xcode 工具链离线编译：`crates/portrait/native/metal/*.metal` 共 6 个文件
全部通过 `xcrun -sdk macosx metal -c`（6/6，0 失败）。这只证明 shader 语法与工具链可用，
**不证明**产品路径会执行这些 kernel。

## 2. 基线与冲突检查

- 本 worktree 独立于主树，分支 `J-X-C-Z/Mac-原生适配`，开工时 `git status` 干净。
- 未发现其他 Mac/Metal/platform 分支。
- `Cargo.toml` 中 workspace `license = "MIT"`，而 `AGENTS.md` 要求保持 `Proprietary`。
  **冲突未裁决**，本 lane 不修改许可证，交由仓库所有者确认。

### 首次编译基线（`cargo check --workspace --all-targets --locked --target aarch64-apple-darwin`）

首次结果：**失败**，`aurora-portrait` 4 个错误（E0432）：

- `crates/portrait/src/cuda.rs` 无条件 `pub use driver::{HostRegistration, PinnedBuffer, Stream}`，
  而 `mod driver` 仅在 `#[cfg(windows)]` 下存在。
- `crates/portrait/src/cuda_v2fuseoffset.rs` 无条件导入仅 Windows 存在的 `V2Warp`。

已在本轮修复（最小 cfg 收口，不改语义）：

- 三条 `pub use` 加上 `#[cfg(windows)]`。
- `V2Warp` 导入拆到 `#[cfg(windows)]`。

修复后剩余的 `aurora-engine` 错误（19 个）见 `01-blockers.md`，由 Mac-A 负责。

## 3. 能力状态图例

每项标记：`未实现` / `仅可编译` / `库级测试通过` / `产品路径接入` / `Mac 真机验证`。

| 能力 | 当前状态 | 备注 |
|---|---|---|
| ARM64 `cargo check` | 未实现（首次失败） | 正在修 |
| ARM64 Release 启动 | 未实现 | |
| RAW（NEF HE/HE★/无损） | 未审计 | LibRaw FFI 仅有 Windows 装载路径 |
| JPEG/PNG/TIFF | 未审计 | `image` crate 跨平台 |
| HEIF/JXL | 未审计 | `jxl.rs` 硬依赖 Windows 装载器 |
| 调色（Develop） | 未审计 | CPU 路径跨平台；CUDA 仅 Windows |
| 人像 Metal | 未审计 | 已有 host + MSL，产品接入未核实 |
| AI 蒙版 / ORT | 未审计 | provider 仅 CUDA/DirectML/CPU |
| Core ML / ANE | 未实现 | |
| 图层 | 未审计 | |
| 导出 | 未审计 | |
| GPUI 原生体验 | 未审计 | |

## 3. ARM64 Release 首次产物（2026-11-25）

```
target/aarch64-apple-darwin/release/aurora-studio
  Mach-O 64-bit executable arm64；lipo -archs → arm64
  otool -L：22 个依赖，全部 /usr/lib 或 /System/Library，无第三方 dylib
```

冒烟运行：`aurora-studio --gpu-selfcheck` → `GPU selfcheck failed: ONNX Runtime is not available`
（真实原因、非崩溃；本机确实没有 ORT）。

GUI 启动冒烟：直接运行 Release 二进制，进程存活 8 秒且无 panic/崩溃（窗口由 GPUI macOS
后端创建，未做视觉/交互检查），随后主动结束。日志仅有一条如实的 ORT 不可用 warning。
**窗口内容、菜单、快捷键、IME、Retina 均未验证。**

## 4. 复现命令

```sh
cargo metadata --locked --no-deps --format-version 1
cargo check --workspace --all-targets --locked --target aarch64-apple-darwin
cargo test --workspace --locked --target aarch64-apple-darwin -- --test-threads=1
cargo build -p portrait-app --release --locked --target aarch64-apple-darwin
```

应用 package 名为 `portrait-app`，二进制为 `aurora-studio`。
