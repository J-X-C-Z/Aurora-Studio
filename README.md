# Aurora Studio

Rust / GPUI 桌面照片编辑器。此版整合 Levin 最新前端、Windows 主后端与人像算法接入，程序名为 `aurora-studio`，crate 统一为 `aurora-*` / `portrait-app`。

## Windows 运行

从 Releases 下载 Windows x64 包并解压，启动 `aurora-studio.exe`。支持文件夹、Capture One Catalog/Session 只读导入，以及不需要照片的演示模式。

```powershell
.\aurora-studio.exe
.\aurora-studio.exe 'D:\Photos'
```

**本地构建、校验安装、非破坏性编辑和批量导出：**见 [Windows 桌面使用与架构](docs/WINDOWS-QUICKSTART.zh-CN.md)。Windows 使用 DirectX 显示；通用 RAW 可使用开源 LibRaw 0.22.3（ABI 25）。AI 工具需对应模型，受支持的原生 NVIDIA 路径使用内嵌 PTX，个别模型版本需要另行配置 CPU 参考运行库。标准导出和编辑不依赖商业模型。发布物不含用户照片、Capture One 模型或第三方闭源推理 DLL。

## 构建与测试

需要 Rust stable（edition 2024）、Windows C++ 工具链与 Windows SDK；当前发布由 Windows 原生构建。macOS 适配见 [docs/macos-adaptation](docs/macos-adaptation/)。

```powershell
cargo metadata --locked --no-deps --format-version 1
cargo check --workspace --all-targets --locked
cargo test --workspace --locked -- --test-threads=1
cargo build -p portrait-app --release --locked
```

默认测试使用合成数据。私有参考测试与数据仅保存在本机 `crates/engine/local-fixtures/`、`crates/engine/testdata/`，不随源码或 Cargo 包发布。持有相应本机文件时，可运行 `cargo test -p aurora-engine --features external-fixtures --locked -- --test-threads=1`。缺少这些文件时会失败的测试列在 [scripts/testdata-skips.txt](scripts/testdata-skips.txt)。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `crates/app` | `portrait-app`：GPUI 界面，二进制名 `aurora-studio`。只通过 `aurora_engine::ImageEngine` 调用处理逻辑 |
| `crates/engine` | `aurora-engine`：解码、渲染管线、目录/会话导入、导出、AI 蒙版、CUDA/Metal 后端 |
| `crates/portrait` | `aurora-portrait`：人像效果 kernel，无 GPUI 依赖，见 [crates/portrait/README.md](crates/portrait/README.md) |
| `private-assets/`（不在 Git 中） | 第三方/Capture One 提取的资产与相关 fixture，本机存在时编译期嵌入，见 [private-assets.md](private-assets.md) |
| `build-support` | engine 与 portrait 的 `build.rs` 共用的私有资产探测 |
| `docs` | 架构、前端契约、平台与研究记录，入口是 [docs/README.md](docs/README.md) |
| `scripts` | 构建、打包、CI 辅助与截图脚本 |
| `.github` | CI、Issue 与 PR 模板 |

## 本版行为

- 接入节点编辑器、挑片、快速页签、示波器、第二查看器、欢迎页与触控布局。
- 保留 Windows 的 RAW、色阶/曲线、颜色编辑器、蒙版持久化、图层合成与 WIA 后端。
- 拖动与静止预览使用同一视口分辨率及像素质量；在途渲染完成后只处理最新待渲染值。
- 自动旋转使用完整 0/90/180/270 度；支持本机胶片曲线列表及当前会话的 C1 风格导入。

## GPU 适配

- **CUDA 细节阶段**：`AURORA_DETAIL_GPU=1` 启用亮度降噪、颜色降噪与锐化 CUDA 路径；与 CPU 阶段逐字节一致，本机 RTX 4070 Ti / 610.88 通过。CUDA 不可用时自动回退 CPU。
- **ONNX Runtime provider**：默认 **CUDA → DirectML → CPU**，交互优先低延迟。TensorRT 只在两种情况下置顶：`AURORA_TRT_CACHE_DIR` 指向的目录已有 `.engine/.profile/.timing` 缓存，或显式设置 `AURORA_ORT_PROVIDERS=tensorrt,cuda,cpu`。`AURORA_ORT_PROVIDERS` 可覆盖任意顺序，未导出的 provider 会被记录并跳过。TRT 冷构建实测 40–190 s/模型（2026-11-23，RTX 4070 Ti），所以不做默认首选。
- **FP16 精度**：实测 TRT FP16 对 subject/skin 的 mask IoU 0.998–0.9998，但原始概率最大差 6–8%。FP16 只建议用于 0.5 阈值掩膜；**不要用 FP16 原始概率做羽化 alpha**。blemish（最大差 5e-4）与 face parsing（8e-3，IoU 0.99969）可以直接用 FP16。
- **人像 CUDA kernel**：`aurora-portrait` 已有 LUT、mesh、teeth、smile、gloss、moisture、makeup 等 CUDA 路径，本机 16 个 CUDA 测试文件、84 个测试通过；尚未接入应用的人像预览主路径。
- **Metal**：保留为 macOS 真机目标，本 Windows 版本未验证，不宣称可用。
- 本构建没有 TensorRT 缓存预热/指纹匹配策略；自动模式是 provider 偏好顺序，不是“TensorRT 已加速”。CUDA/TensorRT provider 需要本机 ONNX Runtime 自己导出对应符号。TRT 的 engine cache 路径目前由部署脚本/外部工具准备，legacy append 还不能把 cache 目录传给 EP（V2 options 未接线，见 [docs/windows-gpu.md](docs/windows-gpu.md)）。

## 已知限制

节点图的并行/图层混合器仍按串行列表渲染，导出仍包含未连接节点。HDR/全景合并、AI 降噪、镜头模糊、ICC 软打样、取消导出及 HEIF/JXL/DNG/PSD 原生导出尚未接通，界面会标注状态。Windows 联机能力取决于 WIA 相机驱动；尚未完成真实相机验收。

人像 132 个已注册 CPU kernel 中，当前 8 个具备真实同 job 输入并会出像素；其余等待 194 点 landmark、逐效果 model output、material/UV 或 ROI/统计量。人像面板会显示缺输入，不用假蒙版或拟合参数硬跑。

本机 HEIF 导出测试因缺少带 HEVC encoder 的 `libheif.dll` 失败；该用例不计入本版通过项。

版本记录见 [CHANGELOG.md](CHANGELOG.md)。开发边界与验收规范见 [CONTRIBUTING.md](CONTRIBUTING.md)、[AGENTS.md](AGENTS.md)、[前端契约](docs/frontend-contract/engine-requests.md)。上文的历史实测不是当前分支或所有设备的验收结论。

## 架构与接手

公共架构 lane 的基线、ADR、模块迁移映射、文件所有权与验证记录见
[`docs/architecture/`](docs/architecture/README.md)，未完成项与接手清单见
[REMAINING-GAPS.md](docs/architecture/REMAINING-GAPS.md)。Windows 原生 Release、macOS Metal、真实模型与性能结论仍以实际运行验证为准。

- [文档索引](docs/README.md)：运行、开发契约、架构决策和工具入口。
- [代码架构导览](docs/architecture/CODEBASE-TOUR.zh-CN.md)：从打开原片到预览、保存与导出的真实调用路径。
- [人像 147 交接](docs/architecture/PORTRAIT147-HANDOFF.zh-CN.md)：已验证的独立成果、尚未接入的功能与基线阻塞。
