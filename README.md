# Aurora Studio

Rust / GPUI 桌面照片编辑器。此版整合 Levin 最新前端、Windows 主后端与人像算法接入，程序名为 `aurora-studio`，crate 统一为 `aurora-*` / `portrait-app`。

## Windows 运行

从 Releases 下载 Windows x64 包并解压，启动 `aurora-studio.exe`。支持文件夹、Capture One Catalog/Session 只读导入，以及不需要照片的演示模式。

```powershell
.\aurora-studio.exe
.\aurora-studio.exe 'D:\Photos'
```

Windows 使用 DirectX 显示。RAW 16-bit 解码需要 LibRaw 0.23；AI 蒙版、闭眼检测、自动旋转需要用户本机合法安装的模型与 ONNX Runtime。发布物不含照片、EXIF 样本、Capture One 模型、ICC、风格或校准表。

## 构建与测试

需要 Rust stable、Windows C++ 工具链与 Windows SDK；当前发布由 Windows 原生构建。

```powershell
$env:CARGO_TARGET_DIR = Join-Path (Get-Location) '90_work/build-cache/windows'
cargo metadata --locked --no-deps --format-version 1
cargo check --workspace --all-targets --locked
cargo test --workspace --locked -- --test-threads=1
cargo build -p portrait-app --release --locked
```

默认测试使用合成数据。私有参考测试与数据仅保存在本机 `crates/engine/local-fixtures/`、`testdata/`，不随源码或 Cargo 包发布。持有相应本机文件时，可运行 `cargo test -p aurora-engine --features external-fixtures --locked -- --test-threads=1`。

## 本版行为

- 接入节点编辑器、挑片、快速页签、示波器、第二查看器、欢迎页与触控布局。
- 保留 Windows 的 RAW、色阶/曲线、颜色编辑器、蒙版持久化、图层合成与 WIA 后端。
- 拖动与静止预览使用同一视口分辨率及像素质量；在途渲染完成后只处理最新待渲染值。
- 自动旋转使用完整 0/90/180/270 度；支持本机胶片曲线列表及当前会话的 C1 风格导入。

## GPU 适配

- **CUDA 细节与几何阶段**：`AURORA_DETAIL_GPU=1` 启用亮度降噪、颜色降噪、摩尔纹、锐化，以及通用镜头畸变/CA/暗角/紫边。整段驻留一次显存，通用路径与 CPU 逐字节对拍；带实测 lens profile 表时回退 CPU。CUDA 不可用时自动回退 CPU。
- **ONNX Runtime provider**：按运行时真实导出的符号自动选择 TensorRT → CUDA → DirectML → CPU；`AURORA_AI_PROVIDER` 可固定 provider。桌面窗口打开后会用低优先级线程预建可用模型 session，`AURORA_AI_PREWARM=0` 可关闭。
- **AI 冷启动诊断**：`AURORA_ORT_CACHE_DIR` 可保存按运行时、provider 和模型版本指纹命名的优化图；`AURORA_ORT_PROFILE_DIR` 可输出 ORT 节点 trace，`AURORA_ORT_OPT_LEVEL` 可覆盖图优化等级。
- **人像 CUDA kernel**：`aurora-portrait` 已有 LUT、mesh、teeth、smile、gloss、moisture、makeup 等 CUDA 路径，本机 16 个 CUDA 测试文件、84 个测试通过；尚未接入应用的人像预览主路径。
- **Metal**：保留为 macOS 真机目标，本 Windows 版本未验证，不宣称可用。
- 本构建没有 TensorRT 缓存预热/指纹匹配策略；自动模式是 provider 偏好顺序，不是“TensorRT 已加速”。CUDA/TensorRT provider 需要本机 ONNX Runtime 自己导出对应符号。

## 已知限制

节点图的并行/图层混合器仍按串行列表渲染，导出仍包含未连接节点。HDR/全景合并、AI 降噪、镜头模糊、ICC 软打样、取消导出及 HEIF/JXL/DNG/PSD 原生导出尚未接通，界面会标注状态。crop/rotate/keystone 尚未并入整帧 develop GPU 链；带 lens profile 表的几何路径仍走 CPU。Windows 联机能力取决于 WIA 相机驱动；尚未完成真实相机验收。

人像 132 个已注册 CPU kernel 中，当前 8 个具备真实同 job 输入并会出像素；其余等待 194 点 landmark、逐效果 model output、material/UV 或 ROI/统计量。人像面板会显示缺输入，不用假蒙版或拟合参数硬跑。

本机 HEIF 导出测试因缺少带 HEVC encoder 的 `libheif.dll` 失败；该用例不计入本版通过项。

完整发布说明见 [`../../05_deliverables/RELEASE-NOTES-2026-11-14.md`](../../05_deliverables/RELEASE-NOTES-2026-11-14.md)。开发边界与验收规范见 [CONTRIBUTING.md](CONTRIBUTING.md)、[AGENTS.md](AGENTS.md)、[前端契约](docs/frontend-contract/engine-requests.md)。
