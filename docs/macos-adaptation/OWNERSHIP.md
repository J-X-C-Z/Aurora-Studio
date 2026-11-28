# Mac lane 文件所有权

原则：同一文件只有一个 Agent 主责；跨边界改动先在本文登记，再动代码。
公共接口（FrameHandle、DAG、ComputeScheduler、算法语义、基础 cache）不重复造。

| Agent | 主责文件 | 只读 / 协调 |
|---|---|---|
| 主控（我） | `docs/macos-adaptation/**`、跨 Agent 整合、最终验证 | 全部 |
| Mac-A 构建/FFI | `crates/engine/src/mock/{gpu_runtime,pixel_gpu}.rs`、`mock/mod.rs`、`blemish_gainmap_cuda.rs`、`jxl.rs`、`raw.rs`、`aimask/ort.rs`、`portrait_stage.rs`（仅 cfg 收口）、`crates/portrait/src/cuda*.rs`（cfg 收口） | Cargo manifests、CI |
| Mac-B Metal runtime | 新增 `crates/engine/src/metal/*`、`metal_runtime.rs`；`crates/engine/src/lib.rs` 仅加 `mod` 声明 | 公共 Frame/调度接口 |
| Mac-C MSL kernels | 新增 `crates/engine/src/mock/*_metal.rs`、`crates/engine/native/metal/*.metal` | `mock/mod.rs`（需报备）、Develop 数学 |
| Mac-D RAW/色彩 | `raw.rs` 的 macOS 解码适配（与 Mac-A 交接）、`aurora_raw/**`、`raw_cache.rs`、`icc.rs` 的 macOS 部分 | 通用解码/缓存语义 |
| Mac-E AI/ORT | `aimask/**` 的 provider 与 macOS 装载（与 Mac-A 交接 `ort.rs`） | SessionKey、模型协议 |
| Mac-F 人像 | `crates/portrait/src/metal*.rs`、`effect_router.rs`、`portrait_*stage*.rs` 的人像 Metal 接入 | 人像数学语义 |
| Mac-G App/打包 | `crates/app/**`、`scripts/build-macos.sh`、`scripts/package-macos.sh`、`.github/workflows/macos.yml` | GPUI 公共交互 |
| Mac-H 性能/画质 | `crates/app/src/perf.rs`、`engine/examples/quality_bench.rs`、各 bench test | 与 Windows Agent 统一 workload |

## 冲突规则

- Mac-A 先完成编译基线，其他 Agent 的编译验证排在它之后。
- `raw.rs`：Mac-A 修编译，随后所有权移交 Mac-D；同一时间只允许一方修改。
- `aimask/ort.rs`：Mac-A 修装载，随后移交 Mac-E。
- `portrait_stage.rs`：Mac-A 只加类型注解；功能改动归 Mac-F。
- `mock/mod.rs`：Mac-A 收口 GPU 模块；Mac-C 新增 Metal 模块时先在本表登记。
- 所有新增 macOS 代码必须在 Windows 上仍能编译（`cfg(target_os = "macos")` 隔离）。
- 登记：主控在整合轮中给 `icc.rs` 增加了 macOS ColorSync 目录搜索（`system_profile_dirs`，`cfg(target_os = "macos")`），并让 `recipe::resolve_profile` 的描述匹配也扫描系统目录；后续 macOS 色彩改动仍归 Mac-D。

## 禁止

- 不改许可证；不提交/推送；不写死 `/opt/homebrew`、用户名、盘符。
- 不用假数据、空实现返回成功、缩放上采样冒充优化。
- 不在活跃主树执行 reset/clean/强制切换。
