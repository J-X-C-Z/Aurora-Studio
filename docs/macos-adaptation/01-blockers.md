# M0/M1 — macOS ARM64 编译阻塞清单

> **状态（2026-11-25）：已全部解决。** `cargo check --workspace --all-targets --locked
> --target aarch64-apple-darwin` 为 0 error，Release 构建与 `.app` 启动通过。
> 本文保留为“当时失败点 + 处理原则”的记录，实施结果见 `02-build-report.md`。

命令：`cargo check --workspace --all-targets --locked --target aarch64-apple-darwin`
基线 commit：`9bf908e`。首次失败后已修复 portrait 的 4 个 E0432（见 `00-baseline.md`）。
以下是剩余 19 个 `aurora-engine` 错误，按负责 Agent 分组。

## Mac-A（构建 / FFI）

### `crates/engine/src/mock/gpu_runtime.rs`
- E0432 `gpu_runtime.rs:21`：`use aurora_portrait::cuda::{CudaError, DeviceBuffer, Nvidia, Stream}`
  —— 这些类型只在 `#[cfg(windows)] mod driver` 下导出。
- E0425:985 `HostRegistration`、E0425:991 `PinnedBuffer` 同上。
- 处理原则：整个模块只在 Windows 有 GPU 实现。macOS 应 `#[cfg(windows)]` 挂载，
  并提供与 `clarity_gpu`/`nr_gpu` 同风格的 `#[cfg(not(windows))]` 桩，
  或让所有调用点在非 Windows 下走 CPU。**不得静默返回成功冒充 GPU。**

### `crates/engine/src/mock/pixel_gpu.rs`
- E0432:15 `DeviceBuffer`。同上。

### `crates/engine/src/blemish_gainmap_cuda.rs`
- E0432:13 `Nvidia, PinnedBuffer`；E0425:49 `DeviceBuffer`。
- 该模块是 blemish gain-map 的 CUDA twin；macOS 无 CUDA，需 `#[cfg(windows)]`
  或用 Metal 重写（后者属 Mac-C 范围，先保证 CPU oracle 可用）。

### `crates/engine/src/mock/mod.rs`
- `pub(crate) mod gpu_runtime;`、`pub(crate) mod pixel_gpu;` 目前无条件挂载。
- 已有 `clarity_gpu`/`nr_gpu` 的 `#[cfg(not(windows))]` 桩可作为模板。
- 检查 `detail_gpu` / `develop_gpu` 在非 Windows 的引用点是否都已 cfg 收口。

### `crates/engine/src/jxl.rs`
- E0433:2 `use std::os::windows::ffi::OsStrExt`；E0599:420 `encode_wide`。
- 文件通过 `LoadLibraryW` 动态装载 `libjxl`。macOS 需 `dlopen`/`dlsym` 分支
  （与 `raw.rs`、`ort.rs` 统一方案），或先在此平台明确返回
  `Unsupported` 的真实原因。**不得 silent 输出替代格式。**

### `crates/engine/src/raw.rs`
- E0308 ×5（1453/1481/1597/1680/1774）：libraw 的 `open`/`open_w` 与路径类型
  在非 Windows 下不匹配（`Vec<u16>` vs `CString`，`*const i8` vs `*const u16`）。
- 处理原则：macOS 走 `libraw_open_file`（`*const c_char`，UTF-8），
  保持现有 Windows 路径不变；动态库名/搜索路径按平台区分，
  优先 `@rpath`/bundle `Contents/Frameworks`，不写死 Homebrew 绝对路径。

### `crates/engine/src/aimask/ort.rs`
- E0425:707 `load_library`、E0425:716/719 `get_base`、E0308:799 `symbol` 返回类型。
- 现有代码只有 `#[cfg(target_os = "linux")]`（dlsym）与 `#[cfg(windows)]`
  （LoadLibrary/GetProcAddress）分支；macOS 需要 `dlopen`/`dlsym` 分支。
- 不要臆造 Core ML EP 符号；macOS 先提供 ARM64 CPU baseline。

### `crates/engine/src/portrait_stage.rs`
- E0282 ×2（4340/5297）：`let mut stepped = None;` /
  `let mut cuda_hlf_auto_fallback = None;` 在 macOS 下后续赋值都被
  `#[cfg(windows)]` 剔除，导致类型无法推断。补显式类型注解即可。

## 验收

1. `cargo check --workspace --all-targets --locked --target aarch64-apple-darwin` 通过。
2. `cargo test --workspace --locked --target aarch64-apple-darwin -- --test-threads=1` 通过
   （synthetic；外部资产测试保持 skip）。
3. `cargo build -p portrait-app --release --locked --target aarch64-apple-darwin` 通过。
4. 所有非 Windows 分支必须返回真实原因或走 CPU oracle，不得空实现返回成功。
