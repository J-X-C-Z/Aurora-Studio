# M0 — AI / ONNX Runtime 审计（Mac lane, Mac-E）

> 只读审计。结论以当前 worktree 源码与本机实际命令为准；未运行/无本地资产的项目一律标“未验证”。
> 基线 commit：`9bf908e`（本 worktree `HEAD`）。范围：`crates/engine/src/aimask/**`、
> `face_parsing.rs`、`closedeyes.rs`、`autorotate.rs`、`ai_denoise.rs`、`dustdetect.rs`、
> `portrait_plump_onnx.rs`、`portrait_vendor_models.rs` 及所有经 `Runtime` 打开 ORT session 的模块。
>
> **状态说明（2026-11-25）**：审计期间 Mac-A 已在工作树（未提交 `git diff`）补齐 `ort.rs`
> 的 macOS 装载分支。本文 §0/§1/§4 已按该新状态复核并注明“已补/部分补”；
> 其余结论不受影响。若 `ort.rs` 继续变动，以最新源码为准。

## 0. 结论摘要

1. **没有 Rust 侧 ORT 依赖**：`Cargo.toml` / `Cargo.lock` 中没有 `ort`、`onnxruntime` 或
   provider crate；运行时通过 `dlopen`/`dlsym` 手工加载 C API（`aimask/ort.rs:105-109`）。
   因此 macOS 适配是**装载层 + provider 选择层**问题，不涉及链接/lockfile 变更。
2. **macOS 装载分支已在审计期间由 Mac-A 补上（工作树未提交）**：`get_base` 与
   `Runtime::symbol` 已从 `#[cfg(target_os="linux")]` 扩为 `#[cfg(not(windows))]`
   （`ort.rs:710-712`, `ort.rs:803-806`），`load_library` 新增 `#[cfg(not(windows))]`
   `dlopen` 实现（`ort.rs:1671-1680`）。审计复核：ABI/调用约定正确，`01-blockers.md`
   原先的 `ort.rs:707/716/719/799` 应可消除；但**尚未编译验证**（构建锁在其他 Agent），
   且库候选与 bundle 搜索仍不完整（见下条）。
3. **库候选的 macOS 命名不完整**：`library_candidates_from` 已加
   `#[cfg(target_os="macos")]` 分支（`ort.rs:1572-1590`），但列的是
   `libonnxruntime.dylib`、`libonnxruntime.23.dylib`、`libonnxruntime.1.dylib`、
   `libonnxruntime.dylib.23`。官方 macOS 命名通常是
   `libonnxruntime.1.23.2.dylib` + 符号链接 `libonnxruntime.1.23.dylib` /
   `libonnxruntime.1.dylib` / `libonnxruntime.dylib`；**缺少 `libonnxruntime.1.23.dylib`**，
   且没有 `.app/Contents/Frameworks` 搜索。当前只有 `AURORA_ORT_LIB`（`ort.rs:1553`）
   能可靠指定。
4. **本机没有任何 ONNX Runtime 运行库**：`/opt/homebrew/opt/onnxruntime` 不存在、
   `brew list onnxruntime` 报无此 keg、`/opt/homebrew` `/usr/local` `~/Library`
   `~/WorkSpace` `~/Downloads` 下未找到 `libonnxruntime*`、Python `import onnxruntime` 失败。
   因此 **CoreML EP 的运行时符号支持在本机未验证**，本文只给出探测流程，不声称结论。
5. **Core ML EP 未接入代码**：`Provider` 枚举只有 TensorRT/CUDA/DirectML/CPU
   （`ort.rs:148-154`），`provider_symbol` / `append_symbol_candidates`
   （`ort.rs:185-208`）、`providers_from`（`ort.rs:218`）、`provider_from_name`
   （`ort.rs:237`）、provider 日志数组（`ort.rs:278-283`）都没有 CoreML。
6. **CPU-only baseline 可行**：`Cpu` 永远是最后一个 provider（`ort.rs:218-228`,
   `ort.rs:263`）；`open_cpu_session` 直接只开 CPU（`ort.rs:1137-1146`）；
   `Runtime::get()` 失败只 warn 并返回 `None`（`ort.rs:688-699`）。
   缺少模型/运行库不会阻止应用启动（`crates/app/src/lib.rs:85-122`）。
7. **证据字段已有 provider trace 与 fallback reason**，但没有 IoU/Dice、landmark 偏移、
   连续 alpha 误差等画质指标的运行时记录（见 §8）。

---

## 1. 运行时装载现状（`crates/engine/src/aimask/ort.rs`）

### 1.1 API 版本契约

| 项 | 值 | 位置 |
|---|---|---|
| 头文件/索引来源 | ORT **1.23.2** header，`ORT_API_VERSION 23` | `ort.rs:4-7`, `ort.rs:29` |
| 请求的最高 API | 23 | `ort.rs:29` |
| 接受的旧 API 下限 | 18 | `ort.rs:31` |
| Table 版本选择 | 先问库自身的 minor，再向下降级到 18 | `ort.rs:728-746` |
| 输入 dtype | float32 / float16 | `ort.rs:23-26` |

代码用的 `OrtApi` 索引是固定数组下标（`ort.rs:69-103`），**没有**任何
`SessionOptionsAppendExecutionProvider*` 项。provider 追加走的是**动态库单独导出的自由符号**，
不是 `OrtApi` 表。因此 CoreML 是否可用取决于 dylib 是否导出对应符号，而不是 API table 版本。

### 1.2 Provider 枚举与选择顺序

- 枚举：`TensorRt | Cuda | DirectMl | Cpu`（`ort.rs:148-154`）。
- 默认顺序：CUDA → DirectML → CPU；TensorRT 默认不置顶（`ort.rs:214-228`）。
  TensorRT 仅在 `AURORA_TRT_CACHE_DIR` 已有 engine/profile/timing，或显式要求时置顶
  （`ort.rs:230-235`, `ort.rs:350-361`, `ort.rs:948-956`）。
- `AURORA_ORT_PROVIDERS`（逗号分隔）优先于 `AURORA_AI_PROVIDER`（`ort.rs:247-265`,
  `ort.rs:918-947`）；CPU 永远补在末尾。
- `AURORA_AI_PROVIDER` 只允许把选择收窄到“已探测且确实导出”的 provider；不可用时退回
  `[Cpu]`，**不会换到另一个 GPU**（`ort.rs:363-384`）。
- Provider plan 每进程计算一次并缓存（`ort.rs:895-902`）；实际执行 provider 由
  `Session::provider()` 报告（`ort.rs:1967-1969`），不能把 plan 第一项当作已执行。

### 1.3 macOS 缺口与必须做的事

| # | 现状（审计时工作树） | 位置 | macOS 必须做 / 复核结论 |
|---|---|---|---|
| 1 | **已补**：`load_library` 有 `#[cfg(not(windows))]` 的 `dlopen` 实现，失败用 `dlerror()` | `ort.rs:1671-1680` | 复核通过；`dlopen/dlsym/dlerror` 声明无条件存在（`ort.rs:105-109`）。建议把 `RTLD_NOW` 改用 `libc::RTLD_NOW`。 |
| 2 | **已补**：`get_base` 用 `#[cfg(not(windows))]` + `dlsym` | `ort.rs:710-712` | 复核通过：`OrtGetApiBase` 是 C ABI，macOS 用 `extern "C"` 正确。尚未编译验证。 |
| 3 | **已补**：`Runtime::symbol` 用 `#[cfg(not(windows))]` + `dlsym` | `ort.rs:803-806` | 复核通过；CoreML/TensorRT 等 EP 探测会用到。 |
| 4 | **部分补**：macOS 候选已加，但命名不完整、无 bundle 搜索 | `ort.rs:1572-1590` | 补 `libonnxruntime.1.23.dylib` / `libonnxruntime.1.23.2.dylib`、`.app/Contents/Frameworks/`、`AURORA_ORT_DIR` 下带版本名；删除不标准的 `libonnxruntime.23.dylib` / `libonnxruntime.dylib.23`（或保留但不要作为唯一希望）。**不得写死 `/opt/homebrew` 或用户名**。 |
| 5 | **已补**：错误文案改为平台无关 | `ort.rs:1639-1641` | 复核通过。 |
| 6 | `resolve_library` 不搜索 bundle；macOS 也没有用 `project_roots` | `ort.rs:1629-1648` + `project_roots` `ort.rs:1607-1627` | `.app` 的 exe 在 `Contents/MacOS/`，`Contents/Frameworks/` 是其兄弟目录；显式加入该候选（Windows 分支才遍历 `roots`）。 |
| 7 | 优化缓存目录用 XDG/`~/.cache` | `ort.rs:552-575` | macOS 遵循平台目录：`~/Library/Caches/Aurora Studio/onnxruntime`（或 `NSSearchPath`/现有跨平台目录库）；保留 `AURORA_ORT_CACHE_DIR`。 |
| 8 | Provider plan 只探测 3 个 GPU + CPU | `ort.rs:908-912` | macOS 上探测 CoreML；不改变 Windows 顺序。 |
| 9 | `intra_op_threads` 取 `available_parallelism().min(8)` | `ort.rs:398-400` | M2 8 核含 4E；8 intra-op 线程可能过订阅。至少做成可观测，后续按性能核心数调。 |
| 10 | `provider_blocker` 仅处理 Windows CUDA | `ort.rs:1002-1009` | CoreML 若需要前置条件（符号、OS 版本、session option）应在此或 session 打开失败路径给出可读原因。 |
| 11 | 无签名/打包约束 | — | `Contents/Frameworks` 内 dylib 需 arm64 slice、`@rpath`/`@loader_path`、ad-hoc 签名；用 `otool -L`/`lipo -archs` 核验。 |

### 1.4 模型查找与缓存（非升级风险，但影响 Mac 可用性）

- 模型搜索：`AURORA_AIMODELS` → 相对子目录 → `c1_roots()`（`aimask/mod.rs:190-234`）。
- `c1_roots()` 只读 `AURORA_C1_DIR` 与 Windows `ProgramFiles`（`aimask/mod.rs:179-188`）。
  **macOS 上不会自动发现 Capture One 的 AIModels**；要么用户设 `AURORA_AIMODELS`，要么补
  `/Applications/Capture One.app/Contents/...` 候选（body pose 已单独列了该路径，
  `body_pose.rs:40`）。
- 优化图缓存：`AURORA_DISABLE_ORT_CACHE` / `AURORA_ORT_CACHE_DIR` / 平台默认
  （`ort.rs:552-575`），文件名 `{stem}-{SessionKey::short()}.onnx`（`ort.rs:581-584`）。
- Profiling：`AURORA_ORT_PROFILE_DIR` 生成 `<stem>-<provider>-<pid>` 前缀
  （`ort.rs:1212-1226`），`Session::end_profiling` 取回 JSON（`ort.rs:1280-1298`）；
  `aimask/mod.rs` 的 ignored 测试可汇总每个节点的 `provider`（`aimask/mod.rs:1307-1348`）。
  这是 macOS 上证明“节点实际归属”的现成手段。

---

## 2. Provider 能力矩阵

“代码支持”= 枚举/探测/追加逻辑存在；“运行时支持”= ORT 该发行版是否可能带该 EP；
“本机已验证”= 本 Mac 上是否实际探测/执行过（本次审计只做了文件与命令检查，未运行推理）。

| Provider | 代码支持 | 运行时支持（普通发行版） | 本机（M2 macOS 27.2.1）已验证 | 依据 |
|---|---|---|---|---|
| CUDA | 是：枚举、符号探测、追加、失败原因 | Windows/Linux + NVIDIA CUDA 构建；macOS 无 CUDA | **否**（无 ORT、无 NVIDIA） | `ort.rs:149-154,185-208,975-1000`；`docs/windows-gpu.md` 仅 Windows |
| DirectML | 是：同上 | Windows 专用（DML） | **否** | `ort.rs:149-154,205`；README 默认顺序说明 |
| TensorRT | 是：大小写双拼写探测、缓存判断 | Windows/Linux + NVIDIA TensorRT 构建 | **否** | `ort.rs:187,198-203,308-361` |
| CPU | 是：永远最后、`open_cpu_session` | 任意 ORT 构建 | **否/未验证**：本机没有 ORT；装载分支已补但未编译/未运行 | `ort.rs:218-228,1137-1146`；§0.4 |
| CoreML | **否**：枚举/符号/名称/日志均缺 | 取决于 ORT 是否以 CoreML EP 构建；**本机无 dylib，未验证** | **否** | 见 §5 |

> 运行时支持一列只陈述“该平台通常存在这种构建”，**不是**本机通过证据。按 AGENTS.md：
> 无真机运行/无 profile 节点归属时不得写“已通过”。

---

## 3. 模型清单、磁盘位置、启动依赖

所有模型均为**可选**：缺失时对应功能返回 typed error/refusal，应用继续启动。

| 模型文件 | 模块 | 期望磁盘位置 | 启动必需？ | CPU-only 路径 | 依据 |
|---|---|---|---|---|---|
| `subjectMasking.onnx` | `aimask/mod.rs` | `AIModels/Common/SubjectMasking/`；或 `AURORA_AIMODELS` 根/子目录；或 C1 根 | 否 | 是：`open_session` 在 macOS plan=[Cpu]；失败再 `open_cpu_session` 重试 | `aimask/mod.rs:190-272,565-587,650-684` |
| `skinMasking.onnx` | 同上 | `AIModels/Common/PeopleMasking/` | 否 | 同上 | 同上 |
| `subjectMaskingFP16.onnx` / `skinMaskingFP16.onnx` | 同上 | 仅当首个 provider 是 DirectML，`AIModels/DirectML/` | 否 | 不选 FP16 | `aimask/mod.rs:209-219,236-251` |
| `bodyPose.onnx` / `body_pose.onnx` / `bodyKeypoints.onnx` / `humanPose.onnx` / `pose.onnx` | `aimask/body_pose.rs` | 10 个目录（含 `/Applications/Capture One.app/Contents/Resources/AIModels/Common/BodyPose`、相对目录）；**必须**有同名 `.sha256` sidecar | 否 | 是：`open_session` 自动含 CPU；缺模型 `MissingInput` | `body_pose.rs:27-46,121-180,244-269` |
| `EyeDetector.onnx`, `DenseNet121_FP32.onnx` | `closedeyes.rs` | `AIModels/Common/ClosedEyesDetection/`（经 `model_path`） | 否 | 是：`open_session` | `closedeyes.rs:55-70,159-286` |
| `autoRotation.onnx` | `autorotate.rs` | `AIModels/Common/Rotation/` | 否 | 是：`open_session` | `autorotate.rs:24,63-88,241-243` |
| `dust.onnx` | `dustdetect.rs` | `AURORA_DUST_MODEL` → `AURORA_AIMODELS` → `AIModels/Common/DustRemoval/`；**需 `AURORA_DUST_ONNX=1`** | 否 | 是：`open_session` | `dustdetect.rs:32-43,221-264` |
| `denoise.onnx` / `denoiseOracle.onnx` / `denoiseFP16.onnx` | `ai_denoise.rs` | `AURORA_AI_DENOISE_MODEL` 或 **仅 Windows** `C:\Program Files\...` / `C:\ProgramData\...` | 否 | 本文件只做探测/分块/16-bit 转换，**不直接开 ORT session**；推理闭包由调用方提供 | `ai_denoise.rs:23-46,78-148` |
| `fbe.onnx` / `2102828093.onnx` | `face_parsing.rs` | `AURORA_FACE_PARSING_MODEL` → `AURORA_FACE_PARSING_MODEL_ROOT` / `AURORA_AIMODELS` 根、`models/`、`assets/models/` | 否 | 是：`open_session` 自动含 CPU；CUDA mask 预处理仅 `#[cfg(windows)]` | `face_parsing.rs:26-35,389-429,466-496,414-417` |
| FacePlump `<model-id>.onnx`（HighRes4/HighRes3/LowRes3） | `portrait_plump_onnx.rs` | exe/cwd 的 `models/faceplump`、`assets/portrait/faceplump`；**另有两条 `C:\WorkSpace\...` 私有绝对路径**；需 `.sha256` sidecar | 否 | 是：严格 `open_session_on([Cuda])` 失败后 `open_cpu_session` | `portrait_plump_onnx.rs:111-183,339-395` |
| `814444604.onnx` (PF90116) | `portrait_eye_bag_stage.rs` | `AURORA_PF90116_MODEL`，否则默认 **Windows 私有路径** `C:\WorkSpace\...\build-cache\pf90116\814444604.onnx` | 否 | 是：`open_session` 非 CPU 时再建 CPU twin | `portrait_eye_bag_stage.rs:27-31,322-360` |
| `blemishRemoval.onnx` | `blemish_gainmap.rs` | `AURORA_BLEMISH_MODEL`，否则默认 **Windows** `C:\Program Files\...\BlemishRemoval\blemishRemoval.onnx` | 否 | 是：`open_session` | `blemish_gainmap.rs:64-72,316-320` |
| `beautyMarks.onnx` | `beautymarks.rs` | `AURORA_BEAUTYMARKS_MODEL` → `AURORA_AIMODELS` → C1 根 | 否 | 是：`open_session` | `beautymarks.rs:240-265,224-226` |
| `MatchLookColor/Bw/Brightness/Exp.onnx` | `matchlook.rs` | 经 `model_path`（`AIModels/Common/MatchLook`） | 否 | 是：`open_session` | `matchlook.rs:488-494,533-534,663-717` |
| `similarity.onnx`, `faceDetection.onnx`, `faceMesh.onnx`, `eSam*` | `model_candidates` 列表/multi-class consumers | `AURORA_AIMODELS` / C1 `AIModels/Common/*` | 否 | 是 | `aimask/mod.rs:190-207` |
| 私有 `*.bin` 人像资产 | `portrait_vendor_models.rs` | 资产目录 `<model-id>.bin` | 否 | **不接 ORT**；全部 `runtime_ready()==false` | `portrait_vendor_models.rs:48-149,156-161` |

### 3.1 启动是否会被 AI 阻塞？

不会。`prewarm_in_background()` 在独立线程里运行（`crates/app/src/lib.rs:85-122`）：

- 只有显式 `AURORA_GPU_SELFCHECK` 才跑 `gpu_selfcheck()`（`lib.rs:95-100`）；
- `prewarm_models()` 失败只 `log::warn!("AI session prewarm skipped")`（`lib.rs:114-120`）；
- `prewarm_models()` 自身对每个模型先 `model_available`，缺失就跳过（`aimask/mod.rs:291-319`）；
- `Runtime::get()` 失败返回 `None`（`ort.rs:688-699`），所有 AI 调用点都以 `context(...)`/refusal 处理；
- 唯一会非零退出的是显式 `--gpu-selfcheck`（`crates/app/src/main.rs:20-28`）。

### 3.2 CPU-only baseline 是否存在？

存在。装载分支已补；当前缺口是库候选命名不完整、无真实 dylib、未编译验证：

- `providers_from(false,false) == [Cpu]`（`ort.rs:2197-2206` 测试）。
- `open_session_on` 对未导出的 provider 记录失败并继续，CPU 最终打开（`ort.rs:1054-1135`）。
- `open_cpu_session` 只请求 CPU（`ort.rs:1137-1146`）。
- 用户可用 `AURORA_AI_PROVIDER=cpu` 强制（`ort.rs:363-369`, `apply_override` `373-384`）。
- 前提：补齐 §1.3 余项（库候选/bundle 搜索）并提供 arm64 ORT CPU dylib；装载分支已补。

---

## 4. macOS CPU baseline 实施计划（最小 patch，与 Windows 行为隔离）

目标：`aarch64-apple-darwin` 上能 `dlopen` 一个合法 arm64 `libonnxruntime.dylib`，探测出
`exported=[CPU]`、`selected=[CPU]`，开 CPU session，跑通 subject/skin 的 synthetic/模型路径；
Windows provider 顺序与行为不变。

1. **装载三件套**（`ort.rs`）：Mac-A 已把 `load_library`、`get_base`、`Runtime::symbol`
   补为 `#[cfg(not(windows))]`（`ort.rs:710-712`, `803-806`, `1671-1680`）。审计复核通过，
   待 `cargo check --target aarch64-apple-darwin` 确认。建议把 `RTLD_NOW=2`（`ort.rs:28`）
   改用 `libc::RTLD_NOW` 以免平台常量漂移。
2. **库候选**（`ort.rs:1548-1600`）：已有 `#[cfg(target_os="macos")]` 分支，但需补齐
   `libonnxruntime.1.23.dylib` / `libonnxruntime.1.23.2.dylib`、bundle
   `Contents/Frameworks`、`AURORA_ORT_DIR` 下的带版本名；不写死 Homebrew/用户名，
   优先 `AURORA_ORT_LIB`。
3. **错误文案**（`ort.rs:1639-1641`）：平台无关。
4. **缓存目录**（`ort.rs:552-575`）：macOS 用 `~/Library/Caches/Aurora Studio/onnxruntime`。
5. **模型发现**（`aimask/mod.rs:179-188`）：增加 macOS Capture One 资源根作为候选（可选，
   但至少文档写明用 `AURORA_AIMODELS`）。
6. **打包**（Mac-G 协作）：dylib 放 `Contents/Frameworks`，`lipo -archs` 必须含 arm64，
   `otool -L` 确认依赖可解析，ad-hoc 签名；不得依赖开发机绝对路径。
7. **验证命令**（授权后）：
   ```sh
   AURORA_ORT_LIB=/abs/path/libonnxruntime.dylib \
   AURORA_AI_PROVIDER=cpu \
   cargo run -p portrait-app --release -- --gpu-selfcheck
   ```
   期望：`runtime : ... v1.x apiNN`、`providers: ["CPU"]`、session `provider=CPU`；
   模型缺失时打印 `MISSING` 而不是 panic。

> 验收口径：只有真实 `--gpu-selfcheck` 输出 + session `provider=CPU` + 一次真实 mask（或
> synthetic 模型）才算“库级/产品路径通过”。`cargo check` 通过不等于 AI 可用。

---

## 5. Core ML EP 计划

### 5.1 代码层改动（不影响 Windows）

1. `Provider` 加 `CoreMl`（`ort.rs:148-154`）。
2. `provider_symbol` / `append_symbol_candidates` 加候选
   `OrtSessionOptionsAppendExecutionProvider_CoreML`（`ort.rs:185-208`）。
   大小写规则沿用现有 `export_name_matches`（`ort.rs:210-212`），但 macOS `dlsym` 区分大小写，
   建议同时探测 `CoreML` / `CoreMl` 两种拼写。
3. `provider_from_name` 接受 `coreml`（及可选 `coreml-ane`）（`ort.rs:237-245`）。
4. `providers_from` 增加平台参数或新增 `providers_from_macos(has_coreml)`：macOS 顺序
   **CoreML → CPU**；Windows 保持 CUDA → DirectML → CPU（`ort.rs:214-228`）。
   `AURORA_ORT_PROVIDERS=coreml,cpu` 可覆盖。
5. `log_provider_plan` 的固定数组加入 CoreML（`ort.rs:278-283`）。
6. `append_provider` 复用的 trampoline 是 `(options, c_int) -> Status`（`ort.rs:983-985`）。
   CoreML 追加函数的第二参数通常是 EP flags（`uint32_t`），形状相同；但**必须**从同一
   dylib 的 header/版本确认参数类型与 `0` 的语义，不能仅凭形状臆断。
7. 若目标 ORT 不导出自由符号，则退而使用 `OrtApi` 的通用
   `SessionOptionsAppendExecutionProvider`（provider 名 + key/value options）。该函数在当前
   代码的固定索引表中**没有**对应项（`ort.rs:69-103`），需要按实际 header 增加索引并做
   版本门槛；在未确认前不要接线。
8. `provider_blocker`（`ort.rs:1002-1009`）为 CoreML 增加可读失败原因（符号缺失、session
   创建失败、模型 op 不支持等）；失败仍走 CPU fallback。
9. SessionKey / 缓存键见 §6。
10. 测试：`provider_export_names_match_the_windows_distributions`（`ort.rs:2234-2249`）与
    `providers_prefer_gpu_and_always_end_with_cpu`（`ort.rs:2197-2206`）需加 macOS/CoreML 用例；
    Windows 断言保持原样。

### 5.2 运行时前置条件（全部需实测）

1. 该 dylib 是 **arm64 macOS** 且以 CoreML EP 构建（`--use_coreml` 或发行版默认包含）。
2. 导出符号 `OrtSessionOptionsAppendExecutionProvider_CoreML`（或通用 API 可用）。
3. dylib 的递归依赖（`libonnxruntime_providers_shared` 等）可解析（`otool -L`）。
4. 目标 macOS 部署版本满足 CoreML EP 要求；ad-hoc 签名下 `dlopen` 成功。
5. 模型算子/shape 被 CoreML EP 支持；不支持节点会在同一 session 内回退 CPU
   （因此 `provider=CoreML` ≠ 全图/ANE 执行）。
6. 若用 `AURORA_AI_PROVIDER=coreml` 强制而 CoreML 未导出，现有逻辑退回 `[Cpu]`
   （`ort.rs:373-384`），这是期望行为，不得静默换成别的 GPU。

### 5.3 符号探测流程（本机现在无法执行：没有 dylib）

先获得一个合法的 arm64 ORT dylib（发行版/用户提供，遵守许可），然后只读探测：

```sh
ORT=/abs/path/libonnxruntime.1.23.2.dylib

# 1) 架构与依赖
file "$ORT"
lipo -archs "$ORT"                 # 期望包含 arm64
otool -L "$ORT"

# 2) 导出的 EP 追加符号（关键）
nm -gU -arch arm64 "$ORT" | grep -i 'OrtSessionOptionsAppendExecutionProvider'
nm -gU -arch arm64 "$ORT" | grep -i 'CoreML'

# 3) 字符串层确认 EP 名称/编译痕迹
strings -a "$ORT" | grep -iE 'CoreMLExecutionProvider|AppendExecutionProvider_CoreML'

# 4) 若使用 Python 发行包（先确认许可与授权）
python3 -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
```

也可用最小 C 探针（只 `dlopen`/`dlsym`，不跑模型）：

```c
/* probe_coreml.c */
#include <dlfcn.h>
#include <stdio.h>
int main(int argc, char **argv) {
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { puts(dlerror()); return 1; }
    void *p = dlsym(h, "OrtSessionOptionsAppendExecutionProvider_CoreML");
    printf("append_coreml=%p\n", p);
    return p ? 0 : 2;
}
```
```sh
cc probe_coreml.c -o probe_coreml
./probe_coreml "$ORT"
```

**报告格式**：必须写清探测了哪个绝对路径、文件哈希/版本、`nm` 原始输出、`get_available_providers()`
原文。只有符号存在 + session 实际打开 + ORT profile 节点显示 `CoreMLExecutionProvider`，
才可写“Core ML EP 已执行”；ANE 归属仍写“未确认”。

### 5.4 Core ML 的精度/证据要求

- 只有在 §8 的连续 alpha/IoU/Dice/landmark 对比完成后，才能把 CoreML 与 CPU 视为等价。
- CoreML EP 可能内部 FP16；FP16 原始概率做羽化前必须验证（README 与 `aimask/mod.rs:243-245`
  明确说不要用 FP16 原始概率做羽化 alpha）。当前代码没有强制这一约束（见 §8.4）。
- 小模型/小 ROI 可能 CPU 更快；按真实冷暖测量决定，不按模型名机械走 CoreML。

---

## 6. SessionKey 与缓存键

现有字段（`ort.rs:508-516`）：

| 字段 | 含义 | 来源 |
|---|---|---|
| `model_hash` | 模型内容 FNV/CRC 指纹 | `model_fingerprint` `ort.rs:438-502` |
| `model_len` | 文件长度 | 同上 |
| `ort_version` | `OrtApiBase::get_version` 字符串 | `ort.rs:720-721`；字段 `ort.rs:138`；写入 `ort.rs:770` |
| `ort_api` | 实际请求到的 API 表版本 | `ort.rs:737-747` |
| `providers` | provider label 列表，如 `CUDA,CPU` | `ort.rs:794` |
| `options` | `provider_options_fingerprint()`：opt/threads/tf32/trt_dir/trt_present | `ort.rs:404-413` |

`canonical()` 格式：`v1|model=..|len=..|ort=..|api=..|eps=..|opts=..`（`ort.rs:527-533`）。
`for_provider()` 只把 `providers` 收窄为单个 label，用于 per-EP 缓存（`ort.rs:521-525`）。

### 6.1 macOS / CoreML 必须新增

| 需新增 | 原因 | 建议 |
|---|---|---|
| `platform` + `arch` | 同一 `ort_version`/provider 名在不同 OS/CPU 上可能产生不同优化图/数值 | `macos/arm64` 等，写入 canonical 并把 prefix 提到 `v2` |
| 运行时 dylib 身份 | 同名/同版本的不同构建（Win GPU vs Mac CoreML）不可共享 key | 绝对路径或文件长度+首尾 hash（可复用 `model_fingerprint` 风格） |
| provider 配置，而非仅 label | CoreML flags / compute units / CUDA device id 等会改变执行 | `opts` 内加 `coreml_flags=..;units=..`；或用 V2 provider options |
| precision | FP32 vs FP16 结果不等价 | 输入/输出 dtype 或模型变体写入 key |
| 输入 shape | 动态轴模型同一文件可有多种张量契约 | 实际 run shape（当前 shape 在 `run_typed` 才给，`ort.rs:1316-1365`） |
| 设备指纹 | SoC/GPU/ANE 能力影响可用 EP 与数值 | `hw.model`/chip 名（脱敏序列号），macOS 版本 |
| OS 版本 | CoreML EP 可用性随系统版本变化 | 至少 `macos_major` |

注意：`optimized_model_cache_path_in` 只按 `key.short()` 命名（`ort.rs:581-584`）；
只要上述字段进入 canonical，缓存自然隔离。**不得**让 Windows 与 macOS 的优化图共用同一 key。

---

## 7. Provider trace / fallback 字段现状

已有（可复用为 Mac 证据）：

- `Session::provider()`：实际打开成功的 EP（`ort.rs:1967-1969`）。
- `Session::provider_trace()` → `(requested_providers_string, Option<fallback_reason>)`
  （`ort.rs:1973-1982`）；fallback 在 `open_session_uncached_on` 记录前序失败原因
  （`ort.rs:1097-1114`）。
- `SessionStatus`：`key/provider/requested/fallback/input/outputs/cold_ms/warmup_ms/stages_us/measured_runs`
  （`aimask/mod.rs:55-91`）；`format_session_status` 可直接打印（`aimask/mod.rs:100-116`）。
- `MaskTiming`：`model/provider/session_key/load/cold/preprocess/infer/post/output_dtype`
  （`aimask/mod.rs:432-443`, 记录于 `459-477`）。
- `SessionTiming`：`cold_us/warmup_us/prepared_us/infer_us/post_us/runs`（`ort.rs:1898-1930`）。
- `AURORA_ORT_PROFILE_DIR` + `end_profiling` 可产出每节点 provider（`ort.rs:1212-1226,1280-1298`）。

**没有**：节点级 provider 默认不采集（仅手动 profile）；没有 `fallback_reason` 的标准化枚举，
目前是拼接好的自由文本。

---

## 8. 精度/证据要求 vs 代码现状

| 要求 | 当前代码是否记录 | 依据 |
|---|---|---|
| 连续 alpha（非阈值） | **是（产品路径）**：subject/skin 输出 f32 概率，范围校验 `[-0.02,1.02]`，双线性重采样；`foreground_fraction` 的 `>0.5` 只用于日志/测试 | `aimask/mod.rs:488-517,844-900,147-152` |
| 阈值化 mask | AI 路径本身不做二值化；下游是否阈值需按调用方审计（本次未覆盖） | 同上 |
| FP16/FP32 dtype | **是**：`MaskTiming.output_dtype`、`SessionStatus.input/outputs` dtype | `aimask/mod.rs:433,56-62`；`ort.rs:1998-2030` |
| provider 实际执行 | **是**：`Session::provider()` + trace | `ort.rs:1967-1982` |
| fallback_reason | **是**（自由文本） | `ort.rs:1097-1114,1978-1982` |
| ORT 节点归属 | **手动**：需 `AURORA_ORT_PROFILE_DIR`，默认不采 | `ort.rs:1212-1226`；`aimask/mod.rs:1307-1387` |
| Mask IoU / Dice | **否**：运行时与测试均未计算；README/注释只有历史手工数据 | `README.md:41`；`aimask/mod.rs:243-245`；全仓 grep 仅 dust 的 NMS IoU（`dustdetect.rs:29`） |
| 连续 alpha 误差（max abs/分位） | **否** | — |
| Landmark 偏移 | **否**：body pose 只返回 keypoints，不记录误差 | `body_pose.rs:55-80,198-241` |
| 输入 shape 记录 | **部分**：metadata 记录模型静态 dims；实际 run shape 未写入 key | `ort.rs:1998-2009,1316-1365` |
| 平台/设备指纹 | **否** | `ort.rs:508-516` |

### 8.4 一个需要后续确认的精度矛盾

- `fp16_sibling` 注释与 README 明确：“FP16 只用于 0.5 阈值掩膜；不要用 FP16 原始概率做羽化 alpha”
  （`aimask/mod.rs:243-245`, `README.md:41`）。
- 但 `run_on_image` 把模型输出（可能是 FP16 转回 f32）直接交给 `resample` 并作为连续 mask 使用
  （`aimask/mod.rs:650-697,844-900`），**没有**强制阈值化或拒绝 FP16 概率。
- macOS CPU baseline 不触发 DirectML FP16 兄弟模型（`directml_first()` `aimask/mod.rs:236-240`），
  所以 baseline 不受影响；但若 CoreML/其他 EP 走 FP16，需要按 §5.4 增加门禁与对拍。

---

## 9. 未验证清单（明确、不可推断为通过）

1. **本机没有任何 ONNX Runtime 库**：CPU baseline 在 macOS 上**未运行**。装载分支已由
   Mac-A 在工作树补上，但**尚未编译验证**，也没有真实 dylib 做过 `dlopen`。
2. **CoreML EP 运行时支持未验证**：没有 dylib 可 `nm`/`dlsym`；不知道目标发行版是否导出
   `OrtSessionOptionsAppendExecutionProvider_CoreML`，也不知道其 flags 语义。
3. **CoreML 的实际执行/ANE 归属未验证**：即使未来 symbol 存在，也需 ORT profile/Instruments
   才能说“已执行”；ANE 归属默认写“未确认”。
4. **任何真实模型推理未运行**：`subjectMasking.onnx`、`skinMasking.onnx`、body pose、
   face parsing、plump、dust 等在本机均无模型文件与合法资产。
5. **跨 provider 画质等价未验证**：没有 IoU/Dice、连续 alpha 误差、landmark 偏移数据。
6. **macOS 模型自动发现未验证**：`c1_roots()` 不含 macOS Capture One 路径；只能靠
   `AURORA_AIMODELS`。
7. **macOS bundle 内 dylib 装载/签名未验证**：`Contents/Frameworks`、`@rpath`、ad-hoc 签名。
8. **`ort.rs` 是否为最新 provider 语义未验证**：代码注释指向 ORT 1.23.2，但 workspace 未锁
   具体运行库；实际部署版本需在探测时记录。
9. **`intra_op_threads` 在 8 核 M2 上的最优值未验证**（当前 `available_parallelism().min(8)`）。
10. **`ai_denoise` 是否在任何产品路径实际接 ORT 未在本审计内确认**；该文件只有探测/预处理。

---

## 10. 复现/探测命令汇总

```sh
# 本机 ORT 存在性（当前结果：均无）
ls /opt/homebrew/opt/onnxruntime 2>&1
brew list onnxruntime 2>&1
find /opt/homebrew /usr/local ~/Library -maxdepth 5 -name 'libonnxruntime*' 2>/dev/null
python3 -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())" 2>&1

# 编译验证（不在此审计中执行，构建锁归其他 Agent）
# cargo check --workspace --all-targets --locked --target aarch64-apple-darwin
# 审计开始时 01-blockers 记录的 ort.rs:707/716/719/799 已由 Mac-A 在本轮补上；需重跑确认

# 未来获得 dylib 后的只读符号探测：见 §5.3
```

## 11. 与其他 Agent 的接口边界

- `aimask/ort.rs`：按 `OWNERSHIP.md`，Mac-A 先修编译装载，随后移交 Mac-E；本审计只给方案，
  不改代码。
- SessionKey 与 provider 协议属于公共接口，CoreML 改动需与主控/Windows Agent 对齐，
  保证 Windows provider 顺序与行为不变（任务书 §6.1、§8）。
- 模型资产（C1 私有模型、FacePlump、PF90116、私有 `.bin`）不得进入仓库或 release；
  缺失时保持“未验证”，由仓库所有者裁决授权（AGENTS.md §资产）。

---

## 12. 实施记录与主控决策修正（Mac-E + Mac lead）

- 实现已完成：macOS dylib 候选、Frameworks 搜索、`~/Library/Caches/Aurora Studio/onnxruntime`、
  `Provider::CoreMl`（按导出符号可选）、SessionKey v2（platform/runtime/precision/eps）、
  `aimask/mod.rs` 的 macOS Capture One AIModels 根。测试：`aimask` 44 passed / 0 failed。
- **主控修正（D8）**：本机 Capture One 自带的 `libonnxruntime.1.20.1.dylib` 确实导出 CoreML，
  但默认 provider 仍改为 **CPU only**；Core ML 仅在显式
  `AURORA_ORT_PROVIDERS=coreml,cpu` / `AURORA_AI_PROVIDER=coreml` 时启用。
  原因：Core ML EP 可能选择 FP16 计算，而本树没有 CPU↔CoreML 的连续 alpha/IoU/Dice 对拍；
  默认切换属于“用画质换性能”，不符合任务书 §6.2 与 README 的 FP16 约束。
- 真实装载证据（只读，未复制资产）：`AURORA_ORT_LIB="…/Capture One.app/Contents/Frameworks/libonnxruntime.1.20.1.dylib" \
  aurora-studio --gpu-selfcheck` → `v1.20.1 api20`、`providers: ["CPU"]`、缺失模型如实 MISSING。
- 仍未验证：Core ML session 实际执行、每节点 ANE/GPU 归属、任意模型推理与画质对拍。

### 12.1 修正：不自动使用 Capture One 的 macOS 模型

本机确实存在
`/Applications/Capture One.app/Contents/Frameworks/ImageProcessing.framework/Resources/AIModels`，
但把它加入 `c1_roots()` 会让产品与默认测试**隐式**读取 Capture One 的授权资产，并使
`closedeyes::tests::flat_image_has_no_eyes` 在无 ORT 时进入 session 打开路径而失败。
因此已撤回自动发现：macOS 只接受显式 `AURORA_AIMODELS` / `AURORA_C1_DIR`。
带该 root 的最终测试运行暴露了这个问题，撤回后 `closedeyes` 3/3、`aimask` 44/44 通过。
