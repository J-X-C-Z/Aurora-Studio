# A-Boundary 审计：平台/cfg、原生 FFI、mock 渲染抽取

基线：`8118a4a`（`docs/architecture/ARCHITECTURE-BASELINE.md` 记录同一 SHA）。
审计对象：`crates/engine/src/aurora_raw/`、`raw.rs`、`mock/*_gpu.rs`、`aimask/`、
`crates/portrait/src/cuda*.rs`、`jxl.rs`、`heif*.rs`、`mock/process.rs` 及各 CPU 阶段。
方法：只读静态阅读 + `cargo metadata --locked --no-deps --format-version 1`。
本 lane **未运行** `cargo check/test/build`（其他 Agent 正在编译），因此除 metadata 外，
所有编译/链接/测试结论都只到“静态可判”的强度；无法由本机证明的项集中在文末“未验证”。

本工作树当前有未提交改动（Mac 平台隔离 + A0/A-core + 保存错误传播）：
`crates/engine/src/{lib,raw,aimask/ort,portrait_stage,file_engine,store,adjust}.rs`、
`crates/portrait/src/{cuda,cuda_v2fuseoffset,lib}.rs`、以及新增
`contracts/ document.rs render_plan.rs save_report.rs docs/architecture scripts/architecture`。
本审计把“已落地的平台隔离”与“仍存在的边界问题”分开描述。

---

## 0. 结论摘要

1. `z6iii` 整个子树在 `aurora_raw/mod.rs:12-13` 被
   `#[cfg(all(windows, feature = "z6iii-raw-export"))]` 关闭，但其中只有
   `system_opencl.rs` / `c1_adaptive_opencl.rs` 真正依赖 Win32；颜色链、CPU 重建、
   16-bit 几何、渲染辅助函数都是平台中立的（§1.2）。
2. 同名算法在两处实现：`aurora_raw/*.rs`（通用 C1 链，跨平台）与
   `aurora_raw/z6iii/reconstruct/*.rs`（Z6III/OpenCL 对拍版）。4 个文件字节相同，
   4 个语义或浮点顺序不同，不能直接合并（§1.3）。
3. 平台门控改变**像素输出**而不只是可构建性：非 Windows 下 C1 CPU 链跳过
   CA、highlight recovery、anti-alias 与 linear finish（`aurora_raw/mod.rs:126-127,
   159-166`），交互路径的 `c1_color` 也整段不存在（`mock/process.rs:2018-2046`）。
4. 原生库全部运行时加载：libraw（`dlopen`/`LoadLibraryW`）、ORT（同）、CUDA
   （`LoadLibraryA("nvcuda.dll")`）、OpenCL（`System32\OpenCL.dll`）、JXL/HEIF
   （Win32 DLL）。非 Windows 的 fallback 已存在，但**库名候选与本机 Homebrew 库名不匹配**：
   libraw 找 `.so.23`、本机是 `libraw.25.dylib`；heif_write 找 `.so.1`、本机是
   `libheif.1.23.7.dylib`；JXL/HEIF 读取器在非 Windows 直接拒绝；ORT 本机未安装（§1.4）。
5. 静态 `#[link(...)]` 站点全部在 `cfg(windows)`/`#![cfg(windows)]` 内，未发现
   无门控的 Windows 库静态链接；但 `cargo check` 不链接，链接级结论未验证（§1.5）。
6. 正式渲染仍在 `mock/process.rs`：9,405 行、127 个顶层函数，`render_sized`
   占 918 行（2429-3346），同时触碰 tile 缓存、f32 帧池、GPU、人像 producer、
   spot/layer/clarity（§2.1）。
7. 最小可机械搬迁的三个模块：`mock/grain.rs`（1-549）、`mock/clarity.rs`（1-625）、
   `mock/spots.rs`（1-1295）。它们只依赖 `AdjustmentSet` 子结构/`Geometry`，
   不碰 `Source`/`Plan`/进程缓存/GPU/文件 IO（§2.3）。
8. `mock/retouch.rs`（674 行）**没有被 `mock/mod.rs` 挂载**，是未编译的孤儿文件；
   不能算迁移候选（§2.3）。
9. 依赖边：`FileEngine -> mock::process/geometry/lens_profile` 与
   `aurora_raw -> mock` 是生产边；`recipe -> file_engine`、`recipe -> mock` 只在
   `#[cfg(test)]` 内（`recipe.rs:368/2033` 之后），`mock -> aurora_raw` 只在
   `cfg(all(windows, feature))` 下成立（§2.5）。
10. A3 应先做“模块搬移 + 兼容 re-export”的纯重构，最后才拆 `Source/Plan/render_sized`；
    对拍只接受逐字节/`to_bits()` 相等，不接受容差；本机无外部 NEF/RAW fixture（§3）。

---

## 1. 平台 / feature 门控现实

### 1.1 门控清单

计数（当前工作树，文本统计）：

| 范围 | `.rs` 文件 | 行数 | `cfg(test)` | `cfg(windows)` | `cfg(not(windows))` |
|---|---:|---:|---:|---:|---:|
| `crates/engine/src` | 230 | 180,716 | 250 | 174 | 40 |
| `crates/portrait/src` | — | 65,646 | 122 | 162 | 19 |

`crates/engine/src` 中 `all(windows, feature = "z6iii-raw-export")` 出现 42 次；
连同 app/recipe/doc 的文本出现共 48 次。`cfg(target_os = "windows")` 16 次，
engine 内无 `cfg(target_os = "macos")`（Metal 只在 portrait）。

逐文件/门控：

| 文件 | 门控 | 行 | 作用 |
|---|---|---|---|
| `aurora_raw/mod.rs` | `all(windows, feature)` | 12 | 整个 `pub mod z6iii` 关闭 |
| 同上 | 同上 | 99/102 | OpenCL 线性帧调用 vs `(None, NotZ6iii)` |
| 同上 | 同上 | 126/159 | X-Trans 与 Bayer C1 的线性 finish |
| 同上 | 同上 | 220 | `finish_linear_stages` 定义 |
| `z6iii/cli.rs` | `cfg(windows)` / `not(windows)` | 1 / 29 | 导出 CLI 实体 / 报错 stub |
| `z6iii/color_trace_cli.rs` | 同上 | 14 / 109-117 | 同上 |
| `z6iii/reconstruct/system_opencl.rs` | 父模块门控 | 7-9 | `kernel32` + `LoadLibraryExW` |
| `raw.rs` | `all(windows, feature)` | 829/834, 850/861, 1191 | Z6III 尺寸/route 探测 |
| `raw.rs` | `target_os = "windows"` / `not` | 1265-1280, 1298, 1364, 1387-1453, 1485-1487, 1601-1603, 1682-1686, 1778-1780 | libraw 加载、宽字符路径、`_wfile` |
| `mock/mod.rs` | `cfg(windows)` | 14-19 | `detail_gpu`/`clarity_gpu`/`develop_gpu` 只在 Windows 编译 |
| 同上 | `cfg(not(windows))` | 20-62, 65-86 | `clarity_gpu`/`nr_gpu` 的 `false` stub |
| `mock/pixel_gpu.rs` | 文件内 `cfg(windows)` | 1143/1190/1247 | 仅 GPU 段/基准测试；CPU 入口跨平台 |
| `mock/gpu_runtime.rs` | 文件内 `cfg(windows)`/`not` | 1088-1104 | `transfer_totals`：Windows 走 driver 指标，其他返回 0 |
| `mock/develop_gpu.rs` | 文件内 `all(windows, feature)` | 381-389 | C1 配色表上传（仅 Z6III feature） |
| `aimask/ort.rs` | `cfg(windows)` / `not` | 106-117, 568, 710-713, 803-807, 1429, 1668-1679, 1717 | `LoadLibraryExW` vs `dlopen`/`dlsym` |
| `aimask/ort.rs` | `cfg(windows)` | 34-48, 761-814, 876, 1003, 1424-1459, 1521, 1647 | CUDA runtime 预载、PE 导出表、DML/TRT 符号 |
| `aimask/mod.rs` | `cfg(windows)` / `not` | 413/426 | 线程优先级（`SetThreadPriority`） |
| `lib.rs` | `cfg(windows)` / `not` | 67 / 71 | `jxl` 实体 / 显式拒绝 stub |
| `heif.rs` | 文件级 `#![cfg(windows)]` | 6 | 整个 HEIF 读取器 |
| `lib.rs` | `cfg(not(windows))` | 65-105 | HEIF 读取拒绝 stub |
| `heif_write.rs` | `cfg(windows)` / `not` | 52-100 | `LoadLibraryW` vs `dlopen` |
| `portrait/src/cuda.rs` | `cfg(windows)` | 409 | `mod driver`（`LoadLibraryA("nvcuda.dll")`） |
| 同上 | `cfg(not(windows))` | 3496-3717 | `mod no_driver` 同形 adapter，全部 `Unavailable` |
| `portrait/src/metal.rs` | `cfg(target_os="macos")` / `not` | 1796-1837 | macOS `dlopen` / 非 macOS `None` |
| `portrait/src/cuda_*.rs` | 文件内 `cfg(windows)` | 每文件 2-8 处 | GPU 入口；CPU 参考/常量保持跨平台 |

### 1.2 `z6iii` 门控：哪些算法模块被平台限制误伤

`aurora_raw/mod.rs:12-13`：

```rust
#[cfg(all(windows, feature = "z6iii-raw-export"))]
pub mod z6iii;
```

被一并关闭的模块（`z6iii/mod.rs:3-7,10-13` 及其子树）：

| 模块 | 是否平台中立 | 证据 |
|---|---|---|
| `z6iii/color/*`（13 文件） | **是** | `color/mod.rs:1` `#![forbid(unsafe_code)]`；纯 ICU/CLUT/film 数学；无 `std::os` |
| `z6iii/reconstruct/c1_nef_he.rs` | **是** | 纯 TIFF MakerNote 解析 |
| `z6iii/reconstruct/c1_nikon_he.rs` | **是** | 纯 JPEG-XS/VCGH 系数逆变换 |
| `z6iii/reconstruct/c1_stage1_*`/`c1_stage2_basic.rs` | **是** | 纯 f32/u16 标量数学 |
| `z6iii/reconstruct/c1_stage3_params.rs` | **是** | 纯参数/噪声公式 |
| `z6iii/reconstruct/stage3/*` | **是** | `stage3/mod.rs:1`、`rb.rs:1` `#![forbid(unsafe_code)]` |
| `z6iii/reconstruct/nikon_he_bitstream.rs` | **是** | 纯位流解码 |
| `z6iii/reconstruct/c1_adaptive_gg.rs` | **是** | 无 Win32 引用 |
| `z6iii/geometry16.rs` | **是** | 纯 u16 重采样（用 `mock::geometry::FrameMap`） |
| `z6iii/render.rs` 的 `apply_linear_chromatic_aberration`、`apply_linear_color_noise_2012`、`recover_and_antialias_frame`、`apply_linear_luma_noise_2018` | **是** | 纯 CPU f32/u16；仅 `linear_rgba16`/`neutral` 调 OpenCL |
| `z6iii/mod.rs`、`desktop.rs` | 基本中立 | `desktop.rs` 无 Win32 API，仅默认路径 `C:\Program Files\...`（`desktop.rs:101-102`） |
| `z6iii/cli.rs`、`color_trace_cli.rs` | 否 | `cfg(windows)` `main`；非 Windows 是报错 stub |
| `reconstruct/system_opencl.rs` + `c1_adaptive_opencl.rs` | 否 | Win32 `LoadLibraryExW`，且只选 NVIDIA（`c1_adaptive_opencl.rs:70-79`，vendor `0x10de`） |

结论：

- 平台限制没有下沉到 FFI 层；它包住了整条 Z6III 解码、重建、颜色和 16-bit
  几何算法。可跨平台的算法已经存在，但只以“通用 C1 链”
  `aurora_raw/c1_raw_he.rs`（`mod.rs:9`，无门控）的形式跨平台可用，而那份实现
  与 Z6III 对拍版不一致（§1.3）。
- **同一算法因平台产生不同像素**：非 Windows 下通用 C1 链在
  `run_c1_stage123_tiles_f32` 之后直接 `pack_rgba16`，跳过
  `apply_linear_chromatic_aberration`、`recover_and_antialias_frame`、
  `finish_linear_stages`（`mod.rs:157-169`；Windows 分支 159-166）。
  这不是性能差异，而是输出差异。
- `raw.rs:826-865` 在非 Windows 把 `z6iii_feature=false`、
  `opencl=NotCompiled`，所以 route 指纹与缓存键也随平台变化（`DecodeRoute`）。

### 1.3 同名模块对拍（`aurora_raw/` vs `aurora_raw/z6iii/reconstruct/`）

MD5（前 `aurora_raw/`，后 `z6iii/reconstruct/`）：

| 文件 | 判定 | 具体差异 |
|---|---|---|
| `c1_nikon_he.rs` | **逐字节相同** (`840d169e…`) | 无 |
| `c1_stage1_params.rs` | **逐字节相同** (`f11215…`) | 无 |
| `nikon_he_bitstream.rs` | **逐字节相同** (`84f769…`) | 无 |
| `stage1_reference.rs` | **逐字节相同** (`22e40b…`) | 无 |
| `c1_nef_he.rs` | 近似相同 | reconstruct 多 `is_z6iii_nef`（`reconstruct/c1_nef_he.rs:121-133`），其余同 |
| `c1_stage1_direction.rs` | **语义不同** | reconstruct 新增 `center_prediction`，参考中心用完整 f32 gain + `floor`；root 用 Q10 整数预测并预折叠 black（`c1_stage1_direction.rs:3-30` vs `reconstruct/…:3-30`） |
| `c1_stage1_green.rs` | **语义不同** | reconstruct 增加 `add_product`，且水平分支 `if direction == 1 && sign == 1 { (integer+far_delta)*gain }`；root 统一 `integer + far_delta*gain`（root:48 vs recon:52-58） |
| `c1_stage2_basic.rs` | **语义不同** | 斜向权重 `oblique_third = f32::from_bits(0x3eaa_aaaa)` vs `one_third`；梯度最小者选 `position` vs `rev().find`（平局方向不同）（root:49-58 vs recon:50-61） |
| `c1_stage3_params.rs` | 结构重构，公式相同 | 提取 `offset_for_noise` 并把同一公式内联改为调用（recon:13-20,36） |
| `c1_rb_core.rs` ↔ `stage3/rb.rs` | **浮点顺序不同** | `+` 结合顺序与 `enabled` 求和顺序不同（root:36-50 vs recon:37-53）；公式相同，舍入可能不同 |
| `c1_stage3_finish.rs` ↔ `stage3/finish.rs` | **逐字节相同** (`47852174…`) | 无 |
| `c1_stage3_planar.rs` ↔ `stage3/mod.rs` | **不同** | root 的 `planar` 私有；recon 的 `planar`/`finish_planar` 公开，新增 `Finish` 枚举与 `ggblur` 变体（recon:8-17,103-120） |

输入/输出签名：

- `c1_nikon_he`、`c1_stage1_params`、`nikon_he_bitstream` 完全同签名。
- `c1_stage1_direction` 五个 `pub fn`（`direction_costs`、`early_direction`、
  `low_curvature_direction`、`extended_costs`、`extended_direction`）签名相同，
  差异全在内部实现。
- `c1_stage1_green::stage1_rgb`、`c1_stage2_basic::refine_green_basic` 签名相同。
- `NikonHeCropFrame`、`NikonCrop`、`NikonBlackLevels` 同形；reconstruct 额外有
  `with_adaptive_gg_variant2`、`with_control_bayer_halo`、`with_stage1_opencl`。

常量 / 颜色语义差异（不是同名文件，但属于两条 Z6III 路径）：

| 项 | 通用 C1 链 | Z6III 对拍链 |
|---|---|---|
| Stage3 offset | 固定 `128.0`（`aurora_raw/mod.rs:404`） | `Z6iiiStage3::offset_for_noise(noise, (50,0.0))`（`z6iii/render.rs:19`） |
| WB 归一 | `nikon_as_shot_rgb` 原值（`aurora_raw/mod.rs:270`） | 先 `prepare_z6iii_global_white_balance`，再按 0.001 `round_ties_even`（`z6iii/render.rs:20-22`） |
| 噪声表 | `lookup_camera_noise` → `camera_params`（`aurora_raw/mod.rs:172-186`） | `z6iii_noise_for_iso`（`z6iii/render.rs:18`） |
| 输出后处理 | 非 Windows 跳过；Windows 才跑 CA/highlight/AA/finish | 一直跑（`render_linear_tiles`） |
| 颜色 | 通用 C1 + 可选 `Z6iiiColorStage`（仅交互、仅 Windows feature） | `Z6iiiColorStage` 的 pre-matrix/49³ CLUT（`color/mod.rs:26-34,43-59`） |

**抽取共享实现之前必须做的事**：

1. 先确定唯一浮点运算顺序（`c1_stage1_direction`/`green`/`stage2_basic`/`rb`），
   并为两套输入各留 golden oracle；目前无法证明哪套与厂商一致。
2. 统一常量来源（offset 128 vs noise 派生；WB 三位小数舍入），否则“合并”会改变
   至少一条路径的像素。
3. 把 `OpenCL` 后端隔离成 `#[cfg(windows)]` 适配器，让 CPU 重建/颜色/几何模块
   可在任何平台编译（这是 cfg 重构，不是算法改动）。
4. 去掉算法签名对 `crate::mock::lens_profile::LensWarp` 的依赖（§2.5），否则
   `aurora_raw` 无法脱离 `mock`。
5. 只有外部 NEF + 参考 TIFF 对拍通过后，才能删掉任一副本。

厂商来源：两份头注释都写 “C1v2012”（`c1_stage1_direction.rs:1`、
`c1_stage2_basic.rs:1`）；`reconstruct/mod.rs:1` 明确“parent gates Windows and
z6iii-raw-export”；`color/*` 注释引用安装的 ICC/FCRV。上述“与厂商一致”均属注释
声明，本机无法验证（§4）。

### 1.4 原生 / FFI 面清单

| 面 | 加载方式 | cfg 门控 | 非 Windows fallback（现状） | 未验证 |
|---|---|---|---|---|
| LibRaw | 运行时 `dlopen`（POSIX）/ `LoadLibraryW` + `SetDllDirectoryW`（Win）；`OnceLock<Option<Api>>` | `target_os = "windows"` 分支（`raw.rs:1265-1280, 1342-1450`） | 库名 `libraw.so.23`/`libraw.so`（`raw.rs:1264-1268`）；无库则 8-bit `decode` 回退内嵌预览（`raw.rs:123-137`），16-bit 先走纯 Rust `rust_raw`（`raw.rs:974-999`） | 无库时 8-bit 传感器路径不可用；本机 `libraw.25.dylib` 与候选名不匹配 |
| ONNX Runtime | `LoadLibraryExW`（Win）/ `dlopen`（Linux/macOS） | `cfg(windows)` vs `cfg(any(linux,macos))`（`aimask/ort.rs:1650-1676`） | macOS 候选 `libonnxruntime*.dylib` + Homebrew 路径（`ort.rs:1568-1592`）；无库 `Runtime::get` 失败，调用方回退/报错 | 本机 `/opt/homebrew/lib` 无 onnxruntime；AI mask/body pose 运行时不可用 |
| CUDA driver | `LoadLibraryA("nvcuda.dll")` + `GetProcAddress`（`portrait/src/cuda.rs:480-545`） | `#[cfg(windows)] mod driver`（`cuda.rs:409`） | `#[cfg(not(windows))] mod no_driver`（`cuda.rs:3496-3717`）返回 `CudaError::Unavailable` | 非 Windows 从不加载；CPU 回退是否覆盖所有调用点未逐点验证 |
| CUDA PTX | `include_str!("…ptx")`，38 个文件 | 各文件内/模块 `cfg(windows)`；`architecture_rules.rs:439-481` 有精确 allowlist | 非 Windows 仍把 PTX 编进二进制，但模块不编译；`blemish_gainmap_cuda.rs` 通过 `no_driver` 保持可编译（allowlist 例外） | PTX 与驱动的 JIT/ISA 行为未验证 |
| OpenCL（Z6III） | `LoadLibraryExW("OpenCL.dll", LOAD_LIBRARY_SEARCH_SYSTEM32)`（`system_opencl.rs:7-9,35-45`） | 父级 `all(windows, feature)` | 无；非 Windows `OpenClRoute::NotCompiled`（`raw.rs:861-864`） | 只接受 NVIDIA vendor（`c1_adaptive_opencl.rs:74-79`）；AMD/Intel 被跳过 |
| JPEG XL | `LoadLibraryW`（`jxl.rs:140-146`） | `lib.rs:67` 整模块 Windows | `lib.rs:71-87` 显式 `bail!` / `dimensions -> None` | 本机有 `libjxl.dylib` 但非 Windows 构建直接拒绝 |
| HEIF/HEIC 读 | `LoadLibraryW`（`heif.rs:116-140`） | 文件级 `#![cfg(windows)]`（`heif.rs:6`） | `lib.rs:65-105` 拒绝 stub | 同上；本机有 `libheif.dylib` 但读取器拒绝 |
| HEIF/HEIC 写 | `LoadLibraryW` / `dlopen`（`heif_write.rs:52-100`） | 两条 cfg | 候选仅 `libheif.so.1`/`libheif.so`（`heif_write.rs:124-132`）；`AURORA_HEIF_LIB` 可指定 | 本机 `.dylib` 不匹配；HEVC 编码器是否存在未验证 |
| SQLite | `rusqlite` + `bundled`（`crates/engine/Cargo.toml:36`） | 无 cfg；静态编译 C SQLite | 同一代码路径 | 并发/锁行为未测试 |

补充事实：

- `crates/engine/Cargo.toml:8-10` 只有两个 feature：`z6iii-raw-export`、`external-fixtures`；
  没有 `rb_golden`、`metal`、`cuda` 之类的后端 feature。
- `c1_rb_core.rs:93` 的 `#[cfg(all(test, rb_golden))] mod golden_tests;` 引用了一个
  **未声明的 cfg 与不存在的文件**；该 golden 测试永远不会编译/运行。
- `mock/clarity.rs:517` 的注释建议用 `--features z6iii-raw-export` 跑基准，但该
  feature 在非 Windows 不等于 `windows` cfg，见 §1.6。

### 1.5 Windows-only 库是否在非 Windows 上被链接

静态 `#[link(name = ...)]` 全量站点与门控：

| 站点 | 门控 | 结论 |
|---|---|---|
| `crates/app/src/perf.rs:132` | `#[cfg(target_os="windows")] mod memory`（113） | 非 Windows 不生成 |
| `crates/app/src/state/settings.rs:202,212` | `#[cfg(windows)]` 模块（138） | 不生成 |
| `crates/portrait/src/cuda.rs:480` | `#[cfg(windows)] mod driver`（409） | 不生成 |
| `crates/engine/src/file_engine.rs:70` | `#[cfg(windows)] fn system_memory_bytes`（55） | 不生成 |
| `crates/engine/src/library_ops.rs:189,574` | `#[cfg(windows)]` fn/模块 | 不生成 |
| `crates/engine/src/tether.rs:541,550` | `#[cfg(windows)] mod wia`（470） | 不生成 |
| `crates/engine/src/print.rs:125,139` | `#[cfg(windows)] mod win`（67） | 不生成 |
| `crates/engine/src/raw_cache.rs:1565` | `#[cfg(windows)] fn free_disk_bytes` | 不生成 |
| `crates/engine/src/heif.rs:106,117,662` | 文件级 `#![cfg(windows)]`（6） | 不生成 |
| `aurora_raw/z6iii/reconstruct/system_opencl.rs:7` | 父级 `all(windows, feature)`（`aurora_raw/mod.rs:12`） | 不生成 |

结论：**未发现无门控的 Windows 库静态链接**。两点保留：

1. 本 lane 未跑链接（`cargo check` 只检查、不链接；`cargo build/test` 未运行）。
2. `tests/architecture_rules.rs:17-26` 的头注释仍声称“非 Windows 链接失败，原因是
   其他 engine 模块带 `#[link(kernel32)]`”。按当前源码，heif 已改为文件级
   `#![cfg(windows)]`，其余全部在 cfg 内；该注释可能已过时。需要一次
   `cargo test --no-run` 或 `cargo build` 才能判定。

### 1.6 三层混淆：能力 / 后端 / fixture

| 层 | 现有表达 | 证据 | 问题 |
|---|---|---|---|
| 算法能力 | `AURORA_C1_RAW`（运行时开关，默认开） | `raw.rs:876-878`；`aurora_raw/mod.rs:5-7` | 平台无关的 C1 算法同时受 env 与 Windows 门控约束 |
| 后端实现 | `all(windows, feature = "z6iii-raw-export")` | `aurora_raw/mod.rs:12` 等 42 处 | 把 OpenCL 后端、CPU 参考、颜色链、导出产品路径绑成一个开关 |
| 产品导出 | `z6iii-raw-export` feature + 两个 bin | `Cargo.toml:8,12-20`；app 仅在 `cfg(windows)` 打开（`app/Cargo.toml:31-32`） | 同一个 feature 同时决定“能否构建导出 CLI”“是否编译 OpenCL”“是否应用 C1 颜色” |
| 外部 fixture | `external-fixtures` | `tests/quality_bench.rs:442,457`；`examples/quality_bench.rs:18-91`；`README-quality-bench.md:7,16-29` | 纯测试层，不改变库代码；但 `crates/engine/testdata/` 在本机不存在 |

风险：

- `#[cfg(feature = "z6iii-raw-export")]`（**不带 windows**）出现在
  `mock/process.rs:9232,9245,9249`；在非 Windows 上手动打开该 feature 会进入
  与 `windows` 不一致的分支。当前这些分支只碰 ICC profile，未必编译失败，但语义
  已按 feature 而非平台分叉。
- engine 的两个 bin 用 `required-features`，非 Windows 下能构建出只报错的
  `main`（`cli.rs:29-32`），因此“feature 打开”不等于“后端可用”。
- 目标结构应当是：`algorithm capability`（跨平台，无 cfg）→ `backend adapter`
  （windows/cuda、macos/metal；各自 cfg）→ `product flag`（导出/CLI）。当前三者
  是同一个 `cfg(all(windows, feature))`。

---

## 2. mock 渲染抽取清单（P0 §5）

### 2.1 `mock/process.rs` 函数级清单与分类

规模：9,405 行，127 个顶层函数。分类键：
**P** 纯像素数学；**S** 只依赖 `Source`/`Plan`/`AdjustmentSet`；
**C** 触进程级缓存；**G** 触 GPU；**R** 触人像 producer；**IO** 文件 IO。

| 行范围 | 函数（代表） | 类 | 说明 |
|---|---|---|---|
| 43-79 | `srgb_to_linear`、`linear_to_srgb`、`GammaLut` | P | `GammaLut` 用不可变 `OnceLock` 表；无渲染状态 |
| 80-227 | `source_sample_at` | S | 采样；无缓存 |
| 228-464 | `sample_key`、`tile_budget_bytes`、`tile_store`、`tile_lookup`、`tile_insert`、`fill_tile`、`tile_for`、`TileCache`/`TileStore` | C | 进程级 192 MiB tile 缓存，最多 8 个 key（`process.rs:117-147`） |
| 465-481 | `f32_frame_pool`、`take_f32_frame`、`give_f32_frame` | C | 进程级 256 MiB / 单条 128 MiB f32 帧池 |
| 784-1056 | `catmull_rom`、`kelvin_rgb`、`white_balance_gains`、`render_wb_gains`、`luma`、`sstep`、`vignette_gain`、`vignette_center`、`hue_rgb`、`rgb_to_hsv`、`hsv_to_rgb`、`wheel_offset`、`contrast` | P/S | 只吃 `AdjustmentSet` 子结构/标量 |
| 1057-1282 | `grid_dims`、`MaskGrid::build/sample`、`cached_mask_grids`、`shape_value`、`box_blur_grid`、`guided_filter_grid` | C/S | `cached_mask_grids` 走 `aimask::lookup` 全局掩码缓存 |
| 1283-1394 | `oriented_to_source`、`ai_input_rgba`、`luma_guide`、`luma_range_weight`、`color_range_weight` | S | `ai_input_rgba` 读 `Source` |
| 1395-1612 | `layer_has_spatial`、`layer_plan`、`layer_plans`、`cached_layer_weights` | C/S | `cached_layer_weights` 有 128 MiB 静态缓存（`process.rs:1550-1551`） |
| 1613-1623 | `lut_of`、`lut_get` | P | 纯 |
| 1624-1823 | `portrait_job_id`、`needs_stage_*`、`stage_face_bundle`、`stage_bundle_cached` | R | 调 `portrait_stage`/`portrait_face_input_stage`/`portrait_makeup_bundle` |
| 1824-1951 | `render`、`render16`、`geometry_for`、`clean_region`、`preview_size`、`render_at` | S | 管线入口；`render_sized` 的薄包装 |
| 1952-2065 | `upsample`、`banding_plan`、`build_plan` | P/S | `build_plan` 构造 `Plan` |
| 2066-2345 | `log_develop_gpu_fallback`、`develop_gpu_gain_plane`、`gpu_develop_preflight`、`encode_gpu_develop_bands` | G | 全部 `#[cfg(windows)]`（2066/2079/2147/2297 前的属性） |
| 2346-2428 | `encode_io_rows`、`develop_bands` | P/S | CPU 行量化/逐带循环 |
| 2429-3346 | `render_sized` | C+G+R+S | 918 行；取样缓存、GPU develop、clarity GPU、spots、layers、retouch、人像 producer 全在这里 |
| 3347-3655 | `precurve_line_with`、`encode_f32_frame`、`merge_wide_delta`、`develop_line*`、`develop_at`、`prepared_sample`、`encode_srgb`、`linear_edits`、`output_banding`、`develop`、`develop_pre_curve` | P/S | 逐像素开发；部分吃 `&Plan` |
| 3656-4396 | `develop_display`、HDR、dehaze、color balance/editor、B&W、`local`、layer 权重 | P/S | 纯数学 + `AdjustmentSet` 子结构/`LayerPlan` |
| 4397-4834 | `retouch_plane`、`apply_retouch`、`apply_layers`、`static_mask_can_apply`、`apply_layer_color_row` | S | 吃整帧；调 `retouch_c1`/`beautymarks` |
| 4835-4911 | `luma_plane`、`mean3x3`、`box_blur3/h/v` | P | 纯 |
| 4912-4968 | `histogram` | P | 纯 |
| 4969-4997 | `sample_point`、`render_single` | S | 采样器 |
| 4998+ | `#[cfg(test)]` | — | 大量对拍/基准，部分需 NEF |

文件 IO：生产代码**没有**文件读写；唯一 env 读取是 `AURORA_SAMPLE_TILE_BUDGET_MB`
（248-259）；`std::fs` 只出现在测试（5428、9283）。这是把它迁出 `mock` 的有利条件。

进程级状态（迁移时必须显式带走或注入）：

| 状态 | 行 | 上限 | 影响 |
|---|---|---|---|
| `TileCache` | 117-227, 260-464 | 192 MiB / 8 keys | 跨帧采样复用；key 含 `Source::snapshot_id` 与 geometry |
| f32 帧池 | 465-496 | 256 MiB / 单条 128 MiB | 仅 `use_f32_detail` 路径使用（2709/2854） |
| layer 权重缓存 | 1535-1612 | 128 MiB / 单条 32 MiB | 只对无 luma/color range 的 layer 生效 |
| `GammaLut` | 55-79 | 单张 8192 f32 表 | 不可变 |
| `MaskGrid` 缓存 | 1071-1137 | 由 `aimask` 管理 | AI 掩码平面 |

### 2.2 detail / clarity / grain / spots

| 文件 | 纯/近纯范围 | 触及其他 | 说明 |
|---|---|---|---|
| `detail.rs`（4,493） | 41-76 `post_process_core_f32`；77-101 支持；102-467 f32 阶段；673-817 亮度/模糊；818-1320 8-bit 阶段；1321-2218 镜头/衍射/CA；2219 `hash_noise` | 8/10 的 GPU stub；468-546 入口；547-672 `post_process_impl` 调 `detail_gpu` | CPU 参考与 GPU 对拍同文件；无缓存/IO/人像 |
| `clarity.rs`（625） | 45-625 全部 CPU 函数（`clarity_kernel`…`legacy_apply`） | 599-606 测试构造 `mock::process::Source` | 只依赖 `Clarity`/`ClarityMethod`；GPU 在独立 `clarity_gpu.rs` |
| `grain.rs`（549） | 28-549 全部 | 无 | 只依赖 `FilmGrain`；整数 hash + erf/模糊；确定性 |
| `spots.rs`（1,295） | 34-1295 全部 | 无 | 依赖 `AdjustmentSet`、`Geometry`、`Spot`；纯 RGBA 操作 |
| `retouch.rs`（674） | — | — | **未被 `mock/mod.rs` 挂载**；全仓无 `retouch::` 引用，是孤儿文件 |

### 2.3 可机械搬迁的最小模块集

判定标准：不含 `Source`/`Plan`、不读写进程级缓存、不触 GPU/人像、不做 IO；
只依赖值类型或 `AdjustmentSet` 子结构；移动后可用 `pub(crate) use` 兼容旧路径。

**第一批（建议直接做）**：

| 顺序 | 模块 | 行范围 | 为什么 |
|---|---|---|---|
| 1 | `mock/grain.rs` → `engine::processing::grain` | 1-549 | 生产阶段（`detail.rs:666`、`pixel_gpu.rs:971` 调用）；唯一输入 `&FilmGrain`；无全局状态；确定性 hash 使逐字节对拍容易 |
| 2 | `mock/clarity.rs` → `engine::processing::clarity` | 1-625（测试段 599+ 留在原处或随迁） | CPU 内核完整独立；只依赖 `Clarity`/`ClarityMethod`；`clarity_gpu.rs` 不搬，调用点不变 |
| 3 | `mock/spots.rs` → `engine::processing::spots` | 1-1295 | 生产阶段（`process.rs:3026,3036,4593,4616`）；纯 RGBA + `Geometry`/`Spot`；无缓存 |

搬迁方式：新文件内容原样复制；在 `mock/mod.rs` 写
`pub(crate) use crate::processing::grain;`（或在 `mock` 内保留同名 re-export），
调用点一行不改。输出逐字节相同是**构造性**保证（同一函数体），不是重新实现。

**第二波（需要先解依赖）**：

- `mock/geometry.rs`（1-1285）：`FrameMap` 被 `file_engine`、`aurora_raw`、
  `portrait_face_input_stage` 直接使用；且 `crop_for_aspect` 等含 C1 语义。
  可搬，但要先把 `FrameMap` 变成 `engine::geometry` 的公共值类型。
- `mock/lens_profile.rs`（1-436）：被 `aurora_raw` 生产依赖（§2.5）；搬它等于
  同时切断 `aurora_raw -> mock` 的 `LensWarp` 边，价值高但需同步改
  `aurora_raw/mod.rs`、`z6iii/{mod,render,geometry16}.rs`。
- `mock/scenes.rs`（1-374）：零 `crate::` 依赖，机械上最容易；但它是演示图像生成器，
  不是正式渲染阶段，放 `engine::render::scenes` 或测试支持更合适。
- `mock/retouch.rs`：孤儿文件；先由所有者决定删除还是恢复挂载，不能当作“可迁移资产”。

**第三波（`process.rs` 内部纯数学）**：

- 43-79（色彩转换/LUT）、784-1056（WB/色相/对比）、1613-1623（curve LUT）、
  3741-4396（tone/HDR/dehaze/color editor/B&W/layer 像素）可分成
  `processing::{color,tone,layer}`；但当前大量函数吃 `&Plan`/`&LayerPlan`，
  需要先做 `Plan` 拆分（§3 步骤 A3-5）。
- `histogram`（4912-4968）、`luma_plane`/box blur（4835-4911）无依赖，可随第三波
  一起走。

### 2.4 暂不可移动（活跃所有权）

| 文件 | 原因 | 依据 |
|---|---|---|
| `mock/gpu_runtime.rs` | 性能 Agent 的 CUDA 运行时/预算/回执 | `gpu_runtime.rs:1-10`；AGENTS.md 的“GPU 与性能改动走 patch” |
| `mock/detail_gpu.rs` | 性能 Agent 的 CUDA detail | `detail_gpu.rs:1-10`；AGENTS.md 明列 |
| `mock/clarity_gpu.rs` | 性能 Agent 的 CUDA clarity | `mock/mod.rs:16-19`；同类 |
| `mock/develop_gpu.rs` | 性能 Agent 的 CUDA develop；含 Z6III C1 上传（381-389） | 同上 |
| `mock/nr_gpu.rs` | 性能 Agent 的 CUDA NR | 同上 |
| `mock/pixel_gpu.rs` | CUDA vignette/LCC/grain 平面 | `pixel_gpu.rs:1-20` |
| `portrait/src/cuda.rs`、`cuda_*.rs`、`metal.rs` | Mac/性能 Agent 的平台适配 | `cuda.rs:409/3496`；`metal.rs:1796-1837` |
| `engine/src/aimask/ort.rs` | Mac Agent 刚改 macOS `dlopen`/候选；DirectML 已落地 | 工作树 diff；AGENTS.md |
| `engine/src/raw.rs` 的 libraw loader | Mac Agent 刚改 `LoadPath` | 工作树 diff（`raw.rs:1440-1458`） |
| `engine/src/heif.rs`、`heif_write.rs`、`jxl.rs` | 平台 adapter | 文件级 cfg/`dlopen` |
| `mock/process.rs` 的 GPU/portrait 段 | 与上述 GPU、人像 producer 同文件 | 2066-2345、3049-3339 |
| `file_engine.rs` 保存/导出段 | 本工作树正在改（`SaveOrder`/`SaveReport`） | diff；`file_engine.rs:1611+` |

### 2.5 实际依赖边（file:line）

`FileEngine -> mock`（生产）：

| 位置 | 内容 |
|---|---|
| `file_engine.rs:20` | `use crate::mock::process::{self, Source};` |
| `file_engine.rs:667` | `mock::geometry::FrameMap` |
| `file_engine.rs:725` | `mock::process::oriented_to_source` |
| `file_engine.rs:2558,2708` | `mock::lens_profile::LensShot::from_exif` |
| `file_engine.rs:4546` | `mock::process::ai_input_rgba` |
| `file_engine.rs:4679` | `mock::geometry::FrameMap` |
| `file_engine.rs:3228,4232,4717,4739` | `process::render_at/render/render16` |

`aurora_raw -> mock`（生产，无门控）：

| 位置 | 内容 |
|---|---|
| `aurora_raw/mod.rs:94-96` | `mock::lens_profile::{LensShot, auto_warp}` |
| `aurora_raw/mod.rs:227` | `LensWarp` 参数 |
| `aurora_raw/z6iii/mod.rs:76,96` | `LensWarp` 参数 |
| `aurora_raw/z6iii/render.rs:13,80,259,305,310,659` | `LensWarp`/`sample` |
| `aurora_raw/z6iii/geometry16.rs:85` | `mock::geometry::FrameMap` |
| `aurora_raw/z6iii/color/output.rs:236` | `mock::process::linear_to_srgb` |

`mock -> aurora_raw`（仅 `cfg(all(windows, feature))`）：

| 位置 | 内容 |
|---|---|
| `mock/process.rs:1494,2020-2032` | `crate::z6iii::Z6iiiColorStage`（`crate::z6iii` 在 `lib.rs:263-264` 重新导出） |
| `mock/develop_gpu.rs:381-389` | `C1ColorParams::from_stage` |

因此 `aurora_raw <-> mock` 循环只在 Windows + feature 构建成立；非 Windows 只有
`aurora_raw -> mock` 单向。

`raw <-> aurora_raw`（生产，双向）：

- `raw.rs:231` 调 `aurora_raw::decode_he_rgba16_with`；`raw.rs:981` 调
  `aurora_raw::rust_raw::decode`；`raw.rs:2686` 调 `c1_raw_he::decode_frame_cropped`。
- `aurora_raw/mod.rs:87,90,93` 调 `crate::raw::decode_bayer16` /
  `exif_orientation_bytes` / `RawStageTimings`。

`recipe -> file_engine` / `recipe -> mock`（**仅测试**）：

- `recipe.rs:368`、`recipe.rs:2033` 开始两个 `#[cfg(test)] mod tests`；
  `recipe.rs:2979,3045,3146`（`file_engine::embedded_icc`）与
  `recipe.rs:3987-4001`（`mock::process`）都在测试模块内。
- 反向 `file_engine -> recipe` 是生产：`file_engine.rs:4681,4724,4747,5294`。

其他对 `mock` 的生产依赖（P0 影响面，按文件）：

| 文件:行 | 依赖 |
|---|---|
| `raw_cache.rs:19` | `use crate::mock::process::Source;`（另见 142,220,1195,1207） |
| `hdr.rs:15` | `mock::process::linear_to_srgb`；102,135,185,248,264 用 `Plane` |
| `pano.rs:16` | `mock::process::{linear_to_srgb, Plane}` |
| `merge_async.rs:17` | `mock::process::Source` |
| `matchlook.rs:829` | `mock::process::white_balance_gains` |
| `portrait_stage.rs:4633` | `mock::gpu_runtime::install_portrait_bridge` |
| `portrait_face_input_stage.rs:2158,2182` | `mock::geometry::FrameMap` |
| `lib.rs:373` | `pub use mock::process::{render, render16, Rendered, Rendered16, Source};` |

仅测试引用：`c1_spots.rs:567,659`、`c1catalog.rs:1703`、`c1_styles.rs:1160+`、
`recipe.rs:3987+`、`mock/{detail,nr_gpu,process}.rs` 的 NEF 测试。

### 2.6 抽取时的行为约束

- 缓存 key 依赖 `Source::snapshot_id`（`process.rs:228-247, 499-506`），
  搬移不得改变 id 生成顺序；`f32_frame_pool` 的容量上限（256/128 MiB）与
  tile 预算（192 MiB）是产品行为，不能在搬移中“顺手优化”。
- `render_sized` 的阶段顺序是产品语义：develop → clarity/detail → spots →
  retouch → layers → portrait（`process.rs:3026-3339`）。任何重排都算算法改动。
- 拖动/静止同分辨率要求见 AGENTS.md；`render_at(fast)` 会半分辨率上采样
  （`process.rs:1922-1951`），迁移时不得把它与正式路径混用。

---

## 3. A3 推荐 patch 顺序（迁移 + 回滚 + 验证）

### 3.1 对拍/oracle 策略

| 类型 | 规则 |
|---|---|
| 确定性 CPU 像素 | **逐字节**相等：`Vec<u8>`/`Vec<u16>` 直接 `assert_eq!`；f32 中间值用 `.to_bits()` |
| 浮点重排 | 不允许任何容差；如果必须重排，单独 patch + 单独评审（本审计不建议在 A3 做） |
| GPU 路径 | 需要 Windows + NVIDIA 真机参考帧；本机不可得 → **未验证** |
| RAW/Z6III | 需要外部 NEF + 参考 TIFF；本机 `crates/engine/testdata/` 不存在、`vendor/` 不存在、未设 `AURORA_QUALITY_FIXTURES` → **未验证** |
| 合成输入 | 用 `Source::from_rgba/from_rgba16` 的固定渐变/棋盘；固定 `AdjustmentSet`；固定尺寸 |
| 架构边界 | `cargo test -p aurora-engine --test architecture_rules`（当前 9 条规则）或 `scripts/architecture/check.sh` |

注意：全量 `cargo test --workspace` 在当前副本不是可靠 gate。多个**未加 `#[ignore]`**
的测试直接读取不存在的 `testdata/`，例如 `adjust.rs:1947`（`nef-exif-head.bin`）、
`camera_curve.rs:137,145+`、`rawinfo.rs:851,867`、`c1_eip.rs:541`（外部逆向资产）、
`file_engine.rs:5099,5784`。A3 应使用**定向测试**并在报告中记录范围。

### 3.2 Patch 步骤

| 步骤 | 内容 | 回滚 | 验证 |
|---|---|---|---|
| A3-0 | 冻结 `mock/*_gpu.rs`、`portrait/src/cuda*`、`metal.rs`、`aimask/ort.rs`、`raw.rs` loader、`file_engine` 保存段的所有权；记录基线 `cargo metadata` | 无代码变更 | metadata 通过；确认 perf/Mac patch 未覆盖 |
| A3-1 | 新建 `engine::processing`（或 `engine::render`）骨架 + `mod.rs`，只写兼容 `pub(crate) use mock::…`，不移动代码 | 删除新模块与 lib 声明 | `cargo check --workspace --all-targets --locked`（由集成者执行）；`architecture_rules` 规则 2/6/7 |
| A3-2 ✅ | 搬 `grain.rs` 到 `processing/grain.rs`；`mock` 保留 re-export（本轮已落地，diff 仅路径/文档） | `git revert`（纯文件移动 + one-line re-export） | `detail.rs:666` 调用点未变；grain/render 现有 byte-parity 测试 |
| A3-3 ✅ | 搬 `clarity.rs` 到 `processing/clarity.rs`；`clarity_gpu.rs` 与原调用点通过 `mock` re-export 不动 | 同上 | `process` 的 clarity 测试：`clarity_and_structure_are_different_operators`、`zero_clarity_is_byte_identical`、`clarity_methods_are_pairwise_distinct`、`precurve_clarity_matches_the_full_frame_pass`（均为合成输入） |
| A3-4 ✅ | 搬 `spots.rs` 到 `processing/spots.rs`；其 `crop_for_aspect` 依赖被抽到 `processing/geometry.rs`，`mock::geometry` 保留 re-export | 同上 | `spots.rs` 测试 + `c1_spots.rs` 引用测试；`process.rs:3026/3036` 输出逐字节不变 |
| A3-5 ◐ | 已迁移：43-79、784-1056、1613-1623 → `processing::color`（21 项）；HDR/tone/dehaze → `processing::tone`（9 项）；luma/box blur/histogram → `processing::statistics`（6 项）。`mock::process` 122 tests、字节对拍通过；剩余 `render_sized` 内部/layer/retouch/`Plan` 边界 | 按块 revert；每块一个 commit | `default_curve_and_target_columns_are_byte_identical`、`zero_*_is_byte_identical`、`bw_*`、`hdr_*`、`dehaze_*`、`color_*` 等现有测试；wrapper 与旧实现的合成对拍 |
| A3-6 | **最后**才拆 `Source`/`Plan`/`render_sized` 到 `engine::render`；先引入 `render::Source` 新名并 re-export，再搬 `render_sized` | 分两步 revert；保留旧 `mock::process::render*` wrapper | 需要逐字节预览/导出对拍；无 NEF 时用合成 RGBA16 + 全量 `AdjustmentSet` 组合；GPU/人像分支保持未迁移直到有真机 |
| A3-7 | **独立轨道**：把 `z6iii` 的 cfg 拆成算法（跨平台）+ OpenCL 适配器（windows）+ 导出产品（feature） | 单独 revert；不改算法 | 先在非 Windows 上编译 CPU 子集；Windows 上确认 OpenCL re-export 与 route 指纹不变；外部 NEF 对拍仍属未验证 |

不建议在本轮做：删除任一同名 Z6III 副本、修改浮点结合顺序、把 GPU 模块搬进
`processing`、给 `grain` 之类模块“顺手”改缓存/并行策略。

### 3.3 实际命令（分级）

已执行（本 lane）：

```sh
cargo metadata --locked --no-deps --format-version 1   # exit 0；3 packages；features 见 §1.4
```

交给集成者、按顺序执行：

```sh
cargo check --workspace --all-targets --locked
cargo test -p aurora-engine --lib --locked -- --test-threads=1
cargo test -p aurora-engine --test architecture_rules --locked -- --test-threads=1
```

不要作为 A3 gate：`cargo test --workspace`（缺失 `testdata/` 的非 ignore 测试会失败）、
`cargo test --features external-fixtures`（无 fixture）、GPU 基准（无 Windows/NVIDIA）。

### 3.4 回滚与风险

- 每一步都是“文件移动 + re-export”；`git revert` 即可恢复，不涉及存档 schema、
  缓存 key 或导出格式。
- 风险最高的两步是 A3-5（`Plan` 边界）与 A3-6（`render_sized`）；A3-2/3/4 的风险
  主要是**并发编辑冲突**，不是行为变化。
- 迁移后若 `mock::process::Source` 的公开 re-export 位置改变（`lib.rs:373`），
  app 与外部 API 会受影响；A3 期间必须保持 `aurora_engine::Source` 路径不变。
- 缓存 key 不含源码文件名/`file!()`（全仓无 `file!()`/`module_path!()`），
  所以搬文件不会改变缓存身份；但不要改 `Source::next_snapshot_id` 或 route 指纹。

---

## 4. 未验证（明确列出）

1. 任何 `cargo check/test/build` 结果：本 lane 未运行；`cargo metadata` 通过不等于
   编译/链接通过。
2. 非 Windows 的链接结果：静态 `#[link]` 均已 cfg 门控，但未实际链接；
   `architecture_rules.rs:17-26` 的旧注释可能已过时。
3. Windows 原生 Release 性能/画质：无 Windows 机器、无冷暖帧/P50/P95 数据。
4. CUDA/OpenCL/DirectML/Metal 的任何实际执行：本机无 `nvcuda.dll`、无
   Windows `OpenCL.dll`、无 ONNX Runtime；Metal 代码未在真机跑过。
5. RAW/Z6III 的逐字节一致性：本机无 NEF/RAW fixture，`crates/engine/testdata/`
   不存在，未启用 `external-fixtures`；`aurora_raw/mod.rs:3-4` 的“byte-identical
   to bundled LibRaw”是注释声明，未复验。
6. 同名模块哪一版与厂商 C1 一致：没有参考 TIFF/差异图；只能确认两版实现不同。
7. 本机 Homebrew 的 `libraw.25.dylib`/`libheif.1.23.7.dylib`/`libjxl.dylib` 是否
   能被现有候选名加载：按源码候选名判断为**不能**（§1.4），但未运行加载实验。
8. `rb_golden` golden 测试：cfg 未声明、文件不存在，无法运行。
9. SQLite 在批量保存/并发写下的行为：未测试（另见 A0 基线 §4.4-4.5）。
10. 许可证：根 `Cargo.toml:9`/`LICENSE` 为 MIT，而 AGENTS.md 要求 Proprietary；
    按任务只报告不改（A0 基线 §4.7 已记录）。
