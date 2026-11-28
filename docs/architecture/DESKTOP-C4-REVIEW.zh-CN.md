# Windows 桌面性能候选 C4 · 审阅说明

**审阅目标：`perf/raw-first-load`，不是 `main`；不是人像 147 或零 ORT 交付。** 本分支的代码来源是为 Windows 原生验证冻结的候选，父提交为 `4a05bff13f2a9055a46e521686fe7a3b3d833e50`。源快照在加本说明前为 834 个文件；交接时按路径和字节逐一核对，全部匹配。这套证据只适用于该候选，不能外推到其他分支。

## 为什么值得看

| 主题 | 优先读 | 现有改进／边界 |
|---|---|---|
| RAW 首载和传感器输出 | [aurora_raw](../../crates/engine/src/aurora_raw/mod.rs)、[预取](../../crates/engine/src/aurora_raw/prefetch.rs)、[预热](../../crates/engine/src/aurora_raw/prewarm.rs) | 将嵌入预览与传感器输出、预取与预热分开；保留 CPU、分段/融合 CUDA 的核验入口。能显示预览不代表已完成 RAW 解码。不同机型和压缩方式需要单测。 |
| 交互预览 | [App 调度](../../crates/app/src/state/render.rs)、[GPU Develop](../../crates/engine/src/mock/develop_gpu.rs)、[示波器](../../crates/engine/src/scopes.rs) | 复用高质量像素链和结果元数据，避免对比页面把调度或示波器开销误记为引擎耗时。旧图片/旧 generation 不得覆盖当前帧。 |
| 导出 | [执行器](../../crates/engine/src/export/pipeline.rs)、[共享阶段](../../crates/engine/src/export/stages.rs) | 通用源获取、渲染、编码发布拆成有界阶段；串行和并行使用同一像素逻辑；临时输出成功后发布。不等同于所有 RAW 派生路径已改成同一调度。 |
| 后端路由 / 对照 | [后端契约](../../crates/portrait/src/native_infer/backend.rs)、[运行时](../../crates/engine/src/aimask/ort.rs) | 记录请求、实际设备和可解释的失败；强制测试不许用自动回退替代目标设备。CUDA/DX12 等差异项照实报。ORT CPU 参考仍可独立配置，有一个未达原精度的模型版本只走可选参考后端。 |
| 可搬迁的 Windows 构建 | [构建指南](../WINDOWS-QUICKSTART.zh-CN.md)、[原生构建脚本](../../scripts/build-windows-release.ps1) | GNU Rust/MinGW 下用 SDK `fxc.exe` 预编译 GPUI 着色器，避开启动时只在构建机存在的 Cargo 源目录。评审分支只提交源码和必须的文本 PTX，不提交 exe、DLL、SDK 或安装包。 |

## 已核验 / 未核验

| 范围 | 观察 |
|---|---|
| 冻结源码，Windows 原生 | locked 元数据、workspace all-targets 检查、默认 workspace 套件（3505 通过、0 失败、378 忽略）、自包含 Windows Release 打包均通过。文本 HLSL 的 fxc 编译为 22/22；AMD Radeon 610M 上明确选择 DX12 的设备测试 12/12。 |
| 预览实际像素 | 合成图的 Draft/Final 完整画质回归通过；在一张真实 RAW 上两档调整，各对比 2137×1425（12,180,900 通道样本），显示字节和示波器桶均相同。原图不随提交传输。 |
| 同系列早期 RAW 正确性取证 | 在 C4 前身的同一 Z8 样片中，CPU、分段 CUDA、融合 CUDA 比较 181,764,096 个 u16 样本，零差；OpenCL 未通过同一门槛。这项逐样本证据与 C4 的 Draft/Final 验证是两次试验。 |
| 真实模型强制设备回归 | 本地受控 ORT CPU 为参考：RTX CUDA **15/18**（2 缺文件、1 超容差，自动产品回退不是 GPU 通过）；Radeon DX12 **4/18**（2 缺文件、12 未支持）。两次严格套件均非零退出。这里的 18 是该套件分母，不是所有效果或全部发布模型。 |
| 真实交互 | 1 次隔离桌面回放捕获 119 个不同 Draft 和 1 个 Final。Draft 起点→Viewer 的 P50/P95 为约 52.17/64.31 ms；帧回调不是屏幕物理呈现。未达到 `<8 ms`，也未做干净机器部署、完整覆盖的 24 MP 批量压力或五平台实测。 |

当前分支只作审阅快照。后续 DX12 `ReduceMean`/池化/Resize 的隔离修订没有混进来；Mac Metal 和 OpenCL 路由源码有实现或限制，不从 Windows 的成功推断其他硬件可用。本分支暂不改许可证；上游既有测试资产的来源与再分发结论请维护者在生成任何发布包前按 [资产与许可清单](../../ASSET-LICENSES.md)单独确认。本次没有新增或改动供应商测试资产、模型或私有参考数据。

## 复核入口

从候选分支仓库根运行。下面默认项无需下载第三方参考文件；`fxc.exe` 需来自本机 SDK，构建脚本不安装或发布应用。

```powershell
cargo metadata --locked --no-deps --format-version 1
cargo check --workspace --all-targets --locked
cargo test --workspace --no-fail-fast --locked -- --test-threads=1
./scripts/build-windows-release.ps1 -FxcPath $env:GPUI_FXC_PATH
cargo test --release --locked -p aurora-portrait --lib native_infer::dx12_session::tests -- --test-threads=1 --ignored
```

下面仅在已有合规取得的本地素材及目标设备时运行：用 `AURORA_TEST_RAW` 指定读取的原图；模型测试用 `AURORA_MODELS_DIR` 和 `AURORA_ORT_LIB` 指定本地输入/参考，设置 `AURORA_GPU_BACKEND=dx12` 或 `cuda` 分开验证，文件缺席不计通过。

```powershell
cargo test --release --locked -p aurora-engine --features z6iii-raw-export --test quality_parity real_raw_scoped_draft_matches_final_display_pixels -- --test-threads=1 --ignored
cargo test --release --locked -p aurora-engine --lib aimask::ort::tests::native_infer_vs_ort_parity_all_shipped_models -- --test-threads=1 --ignored --nocapture
```
