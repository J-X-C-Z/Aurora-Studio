# macOS 能力矩阵（持续更新）

状态定义：`未实现` / `仅可编译` / `库级测试通过` / `产品路径接入` / `Mac 真机验证`。
证据来自本机实际命令与源码；没有证据的一律写“未验证”。

> 本表第一版依据 `20-raw-audit.md`、`30-ai-audit.md` 与 M0 基线；Metal/人像/App 部分待
> `10-metal-audit.md`、`40-portrait-audit.md`、`50-app-audit.md` 合并后补齐。

## 构建与运行

| 项 | 状态 | 证据 / 阻塞 |
|---|---|---|
| ARM64 `cargo metadata` | 通过 | 本机已跑 |
| ARM64 `cargo check --workspace --all-targets` | 通过（M1） | Mac-A 复跑 0 错 |
| ARM64 测试 | 部分 | 未过滤：2505 passed / 130 failed / 108 ignored；按仓库跳过表：1697 passed / 30 failed（已归因，见 `03-test-baseline.md`） |
| ARM64 Release 构建 | 通过（M1） | 2m13s，arm64 Mach-O，仅系统 dylib |
| Windows 目标编译 | 部分可验证 | `aurora-portrait` Windows check 通过；`aurora-engine` 因 `libsqlite3-sys` 需要 MSVC C 工具链，本机无法交叉验证 |
| `.app` 启动 | Mac 真机验证 | `open dist/Aurora Studio.app` exit 0；PID 88716 存活，`lsappinfo` 前台，日志文件创建；AppleScript quit exit 0；从 `/tmp` 直接运行二进制也正常（`50-app-audit.md` §11.3） |
| Finder 文档关联（声明 vs 生效） | 已声明并注册，导入效果未验证 | Info.plist 有 `CFBundleDocumentTypes`（Viewer/Alternate，RAW + jpg/png/tiff/webp/heic/heif）；`lsregister -dump` 见两项 claim；双击/Open With 的实际导入未做 GUI 验证（§11.1、§11.4） |
| GPUI macOS 渲染 | 上游支持 | 依赖含 `gpui-pre-apple`（Metal renderer）；本机进程可正常启动，未做画面/像素验证 |

## RAW / 色彩

| 能力 | 状态 | 证据 / 阻塞 |
|---|---|---|
| Z6III NEF HE（纯 Rust） | **产品路径接入（路由级）** | `decode16_traced` 在无 LibRaw 时用 `rust_raw::decode_bytes` 喂 C1；macOS 标记 `z6iii_linear`。**但 Z6III pre-matrix/film/CLUT 颜色阶段仍仅 Windows**，输出不与 Windows 颜色一致；无真实样本，像素未验证 |
| Z6III HE★ | 未验证 | 无样本；profile 白名单可能拒绝 |
| 无损 / 未压缩 NEF | 仅可编译 | `rust_raw` 已实现但未接产品路径 |
| DNG / ARW / CR2 / CR3 / RAF / ORF / RW2 | 仅可编译 | 同上；真实文件未验证 |
| LibRaw fallback | 仅可编译 | macOS `dlopen` + `.dylib` 已加；本机无 LibRaw 运行库 |
| Z6III OpenCL 重建 | 未实现（macOS） | 仅 Windows；CPU oracle 存在但两套实现不一致 |
| Highlight / AntiColorAliasing | 未实现（macOS） | 只有 OpenCL，无 CPU oracle |
| JPEG / PNG / TIFF / WebP | 产品路径接入 | `image` crate，跨平台 |
| HEIF 解码 | 仅可编译 | dlopen + `.dylib` 已加；本机有 `/opt/homebrew/lib/libheif.1.23.7.dylib`，未跑解码 |
| JXL | 仅可编译 | dlopen 已加；本机无 libjxl |
| 缓存目录 | 已改 | macOS `~/Library/Caches/Aurora Studio/raw` |
| 缓存键平台/后端分离 | 未实现 | 键不含 OS/arch/backend/precision |

## AI / 推理

| 能力 | 状态 | 证据 / 阻塞 |
|---|---|---|
| ORT macOS CPU baseline | **Mac 真机验证（装载/providers）** | 用本机合法安装的 Capture One 自带 `libonnxruntime.1.20.1.dylib`（arm64）实载：`v1.20.1 api20`、`providers: ["CPU"]`、缺失模型如实报 MISSING；未运行真实推理 |
| CUDA / DirectML / TensorRT | 不适用（macOS） | 保留 Windows 语义 |
| Core ML EP | 仅可编译（plan 级） | 该 dylib 实际导出 CoreML 符号；代码按符号启用 `CoreMl -> CPU`，**未打开 session、未执行、ANE 归属未确认** |
| ANE | 未验证 | 无 Core ML 路径 |
| 模型发现（macOS） | 显式配置 | 不自动读取 Capture One 资产；需 `AURORA_AIMODELS` / `AURORA_C1_DIR`（已用本机 C1 模型目录验证过路径存在，但未自动启用） |
| 缺模型时启动 | 已实现 | 后台预热失败只 warn，不阻塞启动 |

## Metal / 人像 / 显示

| 能力 | 状态 | 证据 / 阻塞 |
|---|---|---|
| 现有 MSL 工具链编译 | 通过 | 6/6 |
| 人像 Metal host（库级） | **库级测试通过** | M2 真机运行 `metal_runtime 4/4`、`metal_mesh_runtime 12/12`、`metal_smile 1/1`、`metal_skin_assets 1/1`，共 18 项 pass（`--ignored --test-threads=1`） |
| 人像 Metal 产品路径（皮肤 LUT） | **产品路径接入** | PF90005/PF90007：24 组用例与 CPU 逐字节一致，backend=Metal，dispatch=1；真实照片未验证 |
| 人像 Metal 产品路径（P15/网格/微笑） | 未实现 | P15 需 ROI/HLF 输入；网格需新 MSL；21024 明确拒绝 |
| 引擎连续 GPU 子图（库级） | **库级测试通过** | `11-metal-runtime.md`：1 CB / 2 kernel / 1 wait / 1 readback，CPU 对拍四通道误差 0 |
| 引擎连续 GPU 子图（产品路径） | 未实现 | 尚未接入 `mock/process.rs` |
| GPUI 显示互操作 | 未实现 | 待 `50-app-audit.md` |
| 内存指标（macOS） | 未实现 | `perf.rs` 仅 Windows |

## 性能（合成、库级；详见 `70-perf-baseline.md`）

| 场景 | Metal P50 | CPU P50 | 倍率 | 画质 |
|---|---|---|---|---|
| 1920×1080，degree 0.5 | 4.87 ms | 113.3 ms | 23.3× | 逐通道差 0 |
| 3840×2160，degree 0.5 | 13.24 ms | 518.0 ms | 39.1× | 逐通道差 0 |
| 37×23，degree 0.5 | 0.31 ms | 0.042 ms | **CPU 快 7×** | 逐通道差 0 |

首次调用（含两条 MSL 运行时编译）= 247.8 ms，属冷启动成本，打包需预编译 metallib。

## 打包

| 项 | 状态 |
|---|---|
| `scripts/build-macos.sh` | **已实跑**：exit 0，`cargo build -p portrait-app --release --locked --target aarch64-apple-darwin` 2m17s；Info.plist `plutil -lint` OK；arm64 Mach-O（`50-app-audit.md` §11.2） |
| `scripts/package-macos.sh` | **已实跑**：exit 0，ad-hoc 签名，`codesign --verify` 通过；`otool -L` 21 项全为系统库；zip 17 MB（sha256 随归档时间戳变化，见 §11.2） |
| macOS CI | 待首次 GitHub runner 实跑（本机结果不能替代 CI） |
| 代码签名 / 公证 | **仅 ad-hoc（本机验证）**；未使用 Developer ID，未公证，Gatekeeper 跨机行为未验证 |
