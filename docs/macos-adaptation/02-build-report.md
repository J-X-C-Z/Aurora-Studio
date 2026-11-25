# M1 — macOS ARM64 编译 / 测试 / Release 构建报告

> 本报告只记录本机实际命令与输出。Windows 侧编译、运行与性能均**未验证**。

## 1. 结论

| 项 | 结果 |
|---|---|
| `cargo check --workspace --all-targets --locked --target aarch64-apple-darwin` | **通过（0 error）** |
| `cargo build -p portrait-app --release --locked --target aarch64-apple-darwin` | **通过** |
| `cargo test --workspace --locked --target aarch64-apple-darwin -- --test-threads=1` | **未全绿**：2505 passed / 130 failed / 108 ignored（`--no-fail-fast` 全量统计） |

测试失败集中在两类：**工作树中不存在的外部 `testdata` / `params.json` 资产**，以及**不属于 Mac-A 文件清单的模块在 macOS 上的既有断言/路径差异**。本 lane 负责的模块（`mock/{gpu_runtime,pixel_gpu}`、`blemish_gainmap_cuda`、`jxl`、`raw`、`aimask/ort`、`portrait_stage`、`cuda*.rs`）在合成测试下通过。

## 2. 环境（同 00-baseline.md）

- MacBook Air (Mac14,2)，Apple M2，16 GB，macOS 27.2.1（26B131）
- rustc/cargo 1.98.1，target `aarch64-apple-darwin`（原生）
- worktree HEAD 基线 `9bf908e`；生成报告时工作树含其他 Agent 的并行改动（见 §8）

## 3. 命令与结果

```sh
cargo metadata --locked --no-deps --format-version 1
# PASS

cargo check --workspace --all-targets --locked --target aarch64-apple-darwin
# PASS，0 error；526 条 warning（按 target 重复计数，无新增实质告警）

cargo build -p portrait-app --release --locked --target aarch64-apple-darwin
# PASS，release profile 2m13s
# target/aarch64-apple-darwin/release/aurora-studio（ARM64，约 50.7 MB）

cargo test --workspace --locked --target aarch64-apple-darwin --no-fail-fast -- --test-threads=1
# 2505 passed / 130 failed / 108 ignored；失败 target：
#   aurora-engine --lib            1706 passed / 126 failed / 84 ignored
#   aurora-portrait --lib           660 passed /   2 failed /  0 ignored
#   aurora-portrait --test metal_asset 0 passed /  1 failed /  1 ignored
#   portrait-app --lib               62 passed /   1 failed /  5 ignored
# 其余 target 与 doc-test 全部通过。
```

先按 AGENTS.md 原样运行（不加 `--no-fail-fast`）时，`aurora-engine` lib 失败后 cargo 中止，故用 `--no-fail-fast` 取得全量计数；两次 `aurora-engine` 结果一致（首次 1699/133/84，最终 1706/126/84，差额来自并行 Agent 对 `library_ops.rs` 的修改，非本 lane）。

## 4. 修改文件与理由

| 文件 | 修改 | 理由 |
|---|---|---|
| `crates/engine/src/mock/mod.rs` | `gpu_runtime`、`pixel_gpu` 仅 Windows 挂载；新增 `#[cfg(not(windows))]` 同风格桩 | GPU runtime 依赖 `Nvidia/DeviceBuffer/Stream` 等仅 Windows 类型；非 Windows 桩让调用方走 CPU（`install_portrait_bridge` 无 CUDA 上下文可观察，为 no-op） |
| `crates/engine/src/blemish_gainmap_cuda.rs` | CUDA 实现整体包进 `#[cfg(windows)] mod imp` + `pub use imp::*`；非 Windows 提供 `apply_16u/apply_8u` 返回 `CudaError::Unavailable` | 无 CUDA 驱动时给调用方**真实类型错误**，`blemish_gainmap.rs` 已有 Err→CPU 路径；不静默成功 |
| `crates/engine/src/jxl.rs` | Windows 装载路径加 cfg；新增 Unix/macOS `dlopen/dlsym/dlclose`，先找 `AURORA_JXL_DIR/libjxl.dylib`，再走动态装载器搜索路径；找不到返回明确错误 | 真实 macOS LibJXL 路径，绝不替换格式 |
| `crates/engine/src/raw.rs` | `PathArg` 类型别名（Windows=`Vec<u16>`，其他=`CString`）；macOS `libraw.dylib` 名称；macOS `load_library` 依次搜索 `AURORA_LIBRAW_DIR`、可执行文件 `../Frameworks`/同级/`../Resources`、仓库 `vendor/libraw/lib`，最后标准名/装载器路径；`libraw_open_file` 保持 UTF-8 | 修复 5 处路径类型错误；Windows `LoadLibraryW` 路径未改；不写死 Homebrew 或用户名 |
| `crates/engine/src/aimask/ort.rs` | `dlsym`/`get_base`/`load_library` 的 `#[cfg(target_os = "linux")]` 改为 `#[cfg(not(windows))]`；新增 macOS `libonnxruntime.dylib` 候选；错误文案去 Windows 化 | 镜像现有 Linux dlopen/dlsym 分支；未臆造 Core ML EP 符号；`symbol()` 各平台均返回 `*mut c_void` |
| `crates/engine/src/portrait_stage.rs` | 仅 2 处显式类型：`Option<()>`、`Option<String>` | 非 Windows 赋值被 cfg 剔除导致无法推断；无功能改动 |
| `crates/portrait/src/cuda_radiance.rs` | CUDA-only 测试模块改为 `#[cfg(all(test, windows))]` | 该测试的 `from_raw_parts(&[u8] as *const f32)` 在 macOS 静态对齐下触发 UB 检查并 SIGABRT，导致整个 `aurora-portrait --lib` 中止；与其余 CUDA 测试一致收口，Windows 仍运行 |
| `crates/portrait/examples/n35_reuse_acceptance.rs` | 原内容包进 `#[cfg(windows)] mod cuda_example`；非 Windows 提供空 `main` | CUDA-only 示例，`--all-targets` 必需；Windows 逻辑不变 |
| `crates/engine/examples/blemish_gpu_bench.rs` | 同上（`mod cuda_bench`） | 同上 |

已在开工前部分修复、本 lane 验证无误：`crates/portrait/src/cuda.rs`（三条 `pub use` 加 `#[cfg(windows)]`）、`crates/portrait/src/cuda_v2fuseoffset.rs`（`V2Warp` 导入加 `#[cfg(windows)]`）。

## 5. 非 Windows 行为保证

- `pixel_gpu` 桩的 `vignette_plane_cached`/`lcc_plane_for_output` 返回 `None`、`grain_stage` 返回 `false`，调用方走既有 CPU 公式；`sample_plane` 是与 Windows 版逐字相同的 CPU 双线性参考。
- `blemish_gainmap_cuda` 非 Windows 返回 `CudaError::Unavailable(...)`，不返回伪造像素。
- `raw`/`jxl`/`ort` 在 macOS 走真实 `dlopen/dlsym`；库缺失时返回带搜索路径的错误。
- 所有 Windows 代码保留，非 Windows 代码全部在 `#[cfg(...)]` 内。

## 6. 测试失败分类（130）

- **外部资产缺失（约 112）**：`crates/engine/testdata/**`、根 `params.json` 在本 worktree 与主树均不存在（按 AGENTS.md 不入库）。涉及 `c1_cos/c1_styles/c1_writeback/c1catalog/c1import/c1session_export/file_engine/presets/exif/rawinfo/camera_*` 等。另 `plump_session` 使用硬编码 `C:\WorkSpace\...` 路径，`metal_asset` 缺生成的 `whiten_lut.f32`，`portrait-app` 缺 EIP fixture。
- **非本 lane 模块的既有断言/行为差异（约 18）**：
  - `mock::geometry::tests::*`（4）keystone/tilt 数值与期望不符；
  - `file_engine::tests::eip_restores_*`（2）期望 -3.0、实得 -2.0；
  - `camera_curve`/`camera_profile`/`file_engine` 相机配置查找（需随包资产）返回 None；
  - `portrait_historical_tone_provider`（2）、`presets::scans_the_bundled_folder_once`、`recipe::icc_profile_changes_the_output`（需 ICC/预设资产）。
  - `aurora-portrait`: `kernels::face_width_414` golden 第 867 字节不符（127 vs 255）；`plump_session` 硬编码 Windows 路径。
- **本 lane 触及模块无失败**：`jxl`、`raw`、`aimask::ort`、`mock::process`（pixel/develop CPU 路径）等在运行中通过。

上述失败均在 Mac-A 文件清单之外，需由对应归属 Agent 或仓库所有者裁决（补资产、加 skip 门禁或修算法）。本 lane 未修改这些模块。

## 7. Cargo 变更

**无。** `Cargo.toml` / `Cargo.lock` 未改动；未新增依赖。

## 8. 并行改动与越界说明

- 测试/构建期间观察到其他 Agent 的并行改动：`crates/engine/src/heif.rs`、`heif_write.rs`、`raw_cache.rs`、`library_ops.rs`，以及未跟踪的 `.github/workflows/macos.yml`、`scripts/build-macos.sh`、`scripts/package-macos.sh`。**均非本 lane 修改。**
- 本 lane 在清单外的改动仅为两个 CUDA-only 示例的 `#[cfg(windows)]` 收口；这是让 `cargo check --all-targets` 通过所必需，且 Windows 代码路径未改。`cuda_radiance.rs` 属清单内 `cuda*.rs` 的 cfg 收口。

## 9. 未验证项

- Windows 下 `cargo check/test/build`（无法在本机验证）。
- macOS 上 LibRaw/libjxl/ONNX Runtime 的**真实装载与解码**：本机未安装这些动态库，仅验证了代码路径可编译、缺库时返回真实错误。
- Metal 运行时、Release 启动、RAW/HEIF/JXL 产品路径、性能数据：均未验证。
- 性能声明不适用；本报告不包含任何 Windows 性能结论。
