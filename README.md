# Aurora Studio

Rust / GPUI 桌面照片编辑器。此版整合 Levin 最新前端与 Windows 主后端，程序名为 `aurora-studio`，crate 沿用 `lumen-*`。

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
cargo build -p lumen-app --release --locked
```

默认测试使用合成数据。私有参考测试与数据仅保存在本机 `crates/engine/local-fixtures/`、`testdata/`，不随源码或 Cargo 包发布。持有相应本机文件时，可运行 `cargo test -p lumen-engine --features external-fixtures --locked -- --test-threads=1`。

## 本版行为

- 接入节点编辑器、挑片、快速页签、示波器、第二查看器、欢迎页与触控布局。
- 保留 Windows 的 RAW、色阶/曲线、颜色编辑器、蒙版持久化、图层合成与 WIA 后端。
- 拖动与静止预览使用同一视口分辨率及像素质量；在途渲染完成后只处理最新待渲染值。
- 自动旋转使用完整 0/90/180/270 度；支持本机胶片曲线列表及当前会话的 C1 风格导入。

## 已知限制

节点图的并行/图层混合器仍按串行列表渲染，导出仍包含未连接节点。HDR/全景合并、AI 降噪、镜头模糊、ICC 软打样、取消导出及 HEIF/JXL/DNG/PSD 原生导出尚未接通，界面会标注状态。Windows 联机能力取决于 WIA 相机驱动；尚未完成真实相机验收。

GPU 界面显示不代表图像处理或 AI 推理已使用 CUDA/TensorRT。本版不宣称新的 GPU 推理加速或原厂算法一致性。

开发边界与验收规范见 [CONTRIBUTING.md](CONTRIBUTING.md)、[AGENTS.md](AGENTS.md)、[前端契约](docs/frontend-contract/engine-requests.md)。

## 许可

项目代码采用 [MIT](LICENSE)；第三方依赖适用各自许可。本机 Capture One 资产与照片不包含在 MIT 授权或发布包内。

图库与文件管理的操作说明、目录格式升级和功能边界见 [docs/library-management.md](docs/library-management.md)。
