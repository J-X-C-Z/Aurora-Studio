# Mac lane 工作计划（按任务书里程碑）

本文件是主控的调度表。审计结论见 10/20/30/40/50 号文档；实际状态以代码与命令为准。

## 阶段与依赖

```
M0 基线 ──┬─> M1 ARM64 CPU 可启动 ──┬─> M2 Metal 基础 ──┬─> M3 RAW/人像/AI ──> M4 性能/显示 ──> M5 打包
          │                          │                   │
          └─ 审计（并行，只读）      └─ Mac-B 设计       └─ Mac-D/E/F
```

- M1 是所有写代码工作的前置：编译不绿，其他 Agent 的验证没有意义。
- 审计（M0）与 M1 编译修复并行，审计不占用构建锁。

## 工作包

| 包 | 负责 | 产出 | 依赖 | 状态 |
|---|---|---|---|---|
| WP-1 编译基线 | Mac-A | 19 个错误清零；check/test/release 通过 | 无 | **完成** |
| WP-2 Metal runtime 设计 | Mac-B | 10-metal-audit.md + 最小 API 方案 | WP-1（验证） | **完成** |
| WP-3 RAW/色彩审计 | Mac-D | 20-raw-audit.md | 无 | **完成** |
| WP-4 AI/ORT 审计 | Mac-E | 30-ai-audit.md | 无 | **完成** |
| WP-5 人像能力矩阵 | Mac-F | 40-portrait-audit.md | 无 | **完成** |
| WP-6 App/打包 | Mac-G | 50-app-audit.md、脚本、macOS CI、`.app` 真机启动 | WP-1 | **完成** |
| WP-7 性能/画质基线 | Mac-H | 冷/暖帧、P50/P95、footprint、画质对拍 | WP-1, WP-2 | 进行中 |
| WP-8 Metal runtime（库级） | Mac-B | 2-kernel 连续链，1 CB/1 wait/1 readback，误差 0 | WP-2 | **完成（库级）** |
| WP-8b Metal 产品渲染链 | Mac-C（未启动） | Develop/Clarity 接入 `process.rs` | WP-8 | 未开始 |
| WP-9 RAW macOS 解码 | Mac-D | 纯 Rust 进入产品 16-bit 路由 + Z6III provenance | WP-1, WP-3 | **完成（路由级）** |
| WP-10 AI CPU baseline + CoreML | Mac-E | macOS 装载/候选/CoreML 探测 | WP-1, WP-4 | 进行中 |
| WP-11 人像 Metal 产品接入 | Mac-F | PF90005/PF90007 真机字节一致 | WP-1, WP-5 | **完成（皮肤 LUT）** |

## RAW 交接（Mac-D 审计已完成 → 20-raw-audit.md）

`raw.rs` 先由 Mac-A 修编译（仅平台类型/库名），编译绿后所有权移交 Mac-D，按
20-raw-audit.md 的 P0 顺序执行：

1. P0-1 平台路径载体 + macOS `libraw.dylib` 搜索（bundle/`@loader_path`，不写死 Homebrew）。
2. P0-2 `decode16_traced` 在 LibRaw 缺失时接 `rust_raw::decode_bytes`（用已读入 bytes，不重复读盘）。
3. P0-3 macOS 正确识别 Z6III → `z6iii_linear`，修正 `StageShape` 与缓存键。
4. P0-4 Z6III CPU 重建/颜色阶段在 macOS 编译（只把 OpenCL/desktop/cli 留 Windows）。
5. P0-5 统一两套 CPU 重建实现；P0-6 补 highlight/AA CPU oracle 或明确声明未执行。
6. P0-7 HEIF/JXL dlopen；P0-8 macOS 缓存与色彩资产目录；P0-9 合成测试。

硬约束：§8 的 11 条不变量（源文件只读、CFA/WB/色彩顺序、缓存键 revision、provenance）。
真机样本缺失时保持“未验证”。

## AI 交接（Mac-E 审计已完成 → 30-ai-audit.md）

`aimask/ort.rs` 的 macOS 装载三件套（`load_library` / `get_base` / `Runtime::symbol`）由 Mac-A
先修；随后移交 Mac-E 执行：

1. macOS 库候选：`AURORA_ORT_LIB` → `Contents/Frameworks` → cwd 相对 `libonnxruntime*.dylib`；
   不写死 Homebrew。
2. 缓存目录 `~/Library/Caches/Aurora Studio/onnxruntime`；错误文案平台无关。
3. `Provider::CoreMl` 只在**实测该 dylib 导出** `OrtSessionOptionsAppendExecutionProvider_CoreML`
   后启用；Windows provider 顺序不变。
4. 证据：`--gpu-selfcheck` 输出 + session `provider=CPU` + 一次真实/synthetic 推理才算通过。

**本机现状：没有任何 ONNX Runtime 库、没有模型文件** → AI 与 Core ML 全部保持“未验证”。

## 下一波（审计完成后立即派发）

- **Mac-H（性能/画质）**：
  - `crates/app/src/perf.rs` 的内存统计目前只有 Windows（`K32GetProcessMemoryInfo`）。
    增加 macOS `mach_task_basic_info` / `task_vm_info` 分支，保持返回类型不变。
  - 用 `engine/examples/quality_bench.rs` 与 `portrait/tests/s568_bench.rs` 跑合成基线，
    记录 P50/P95、footprint、缓存冷/暖；输出 JSON/CSV。
  - 建立 CPU↔Metal 画质对拍模板（max abs error、逐通道、PSNR/SSIM、mask IoU/Dice）。
- **Mac-C（MSL kernel）**：等 Mac-B 的 runtime 方案落地后，先做 MSL-1
  （WB/曝光/矩阵/曲线/LUT/Levels/HSL），每个 kernel 必须与 `mock/process.rs` 的
  CPU oracle 逐像素对拍，再谈性能。
- **Mac-B（Metal runtime）**：按 10-metal-audit.md 的最小 API 建
  `crates/engine/src/metal_runtime.rs`：device/queue/pipeline cache/buffer-texture pool/
  residency + generation；先只接一个真实效果验证端到端。

## 并发上限

- 本机 8 核 / 16 GB。写代码 Agent ≤ 2 个同时运行，且不允许两个同时跑 `cargo build --release`。
- 只读审计可 4–5 个并行；禁止它们运行 cargo 构建。
- 重 GPU benchmark 单独执行，避免干扰测量。

## 每轮汇报格式

使用任务书第 15 节模板，写入 `docs/macos-adaptation/REPORT-<date>.md`。
百分比只按已验收项计算，不按代码量估算。

## 待仓库所有者裁决

1. `Cargo.toml` 的 `license = "MIT"` 与 `AGENTS.md` 要求 `Proprietary` 冲突。
2. AGENTS.md 引用的 `../../00_admin/OWNERSHIP-2026-11-14.md` 在本机不存在，lane 分工以本目录 `OWNERSHIP.md` 为准。
3. 是否提供 Z6III NEF 合法样本、ONNX 模型与 LibRaw macOS 运行库；缺失项将保持“未验证”。
