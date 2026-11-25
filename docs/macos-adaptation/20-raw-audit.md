# M0/RW — macOS RAW / 色彩只读审计

> 状态：审计完成（只读）。基线 commit `9bf908e`，worktree 为 `Mac-原生适配`。
> 本文件只做源码级结论；**没有运行 cargo check/build，也没有任何 Z6III 真机样本**。
> 所有“未验证”均按 AGENTS.md 第 52 行处理：无 CI、外部 fixture 或真机证据不得推断为通过。

## 0. 方法与证据边界

- 只读了源码与 cfg；`cargo metadata --locked --no-deps` 已运行，确认 package/feature/bin。
- 未运行 `cargo check` / `cargo build`：另一 Agent 持有构建锁。编译错误引用 `01-blockers.md`，不是本轮复现。
- `crates/engine/testdata/` 在本 worktree **不存在**（`.gitignore:7`），仓库内也找不到任何 `.NEF/.DNG/.ARW/.CR3/.RAF/.ORF/.RW2` 样本。因此所有真实样本行为一律标“未验证”。
- 真机字面量（Mac14,2 / M2 / macOS 27.2.1）来自 `00-baseline.md`，不是本审计实测。

能力标记沿用 `00-baseline.md`：`未实现` / `仅可编译` / `库级测试通过` / `产品路径接入` / `Mac 真机验证`。

---

## 1. 解码路由总览（以当前源码为准）

### 1.1 三条实际存在的路由

**A. 纯 Rust 原生前端（native）**

- 入口：`raw::decode_bayer16` → `decode_bayer16_with`（`crates/engine/src/raw.rs:964-1004`）。
- 先跑 `crate::aurora_raw::rust_raw::decode(path)`（`raw.rs:985`），失败才回落 LibRaw。
- `AURORA_USE_LIBRAW=1` 强制 LibRaw；`AURORA_NO_LIBRAW=1` 禁止回落（`raw.rs:986-1002`）。
- `rust_raw` 是无条件编译的（`crates/engine/src/aurora_raw/mod.rs:12`），内部按容器分派：
  CR3 / RAF / ORF / RW2 / TIFF（DNG、NEF、ARW、CR2）（`aurora_raw/rust_raw/mod.rs:70-110`）。
- **重要：产品 16-bit 路径不走它。** 见 1.2。

**B. LibRaw 动态装载（fallback / 8-bit 主路径）**

- 模块：`raw.rs` 的 `mod libraw`（`raw.rs:1258`）。
- 动态库名：Windows 只有 `libraw.dll`；非 Windows 只有 `libraw.so.23` / `libraw.so`（`raw.rs:1265-1268`）——**没有 macOS `.dylib`，也没有 Homebrew / bundle 搜索**。
- 装载：Windows `LoadLibraryW` + `SetDllDirectoryW`（`raw.rs:1387-1422`）；非 Windows `dlopen`（`raw.rs:1424-1427`）。
- 打开文件：Windows `libraw_open_wfile`；非 Windows `libraw_open_file`，路径从 `to_string_lossy()` 转 `CString`（`raw.rs:1440-1454`）。
- Bayer 读取直接按 **x64 结构体硬编码偏移** 读 `libraw_data_t`（`raw.rs:1688-1731` / `raw.rs:1796-1887`），并用 C getter 交叉校验 raw_width/raw_height（`raw.rs:1804-1814`）。arm64 布局未验证。
- 8-bit 主路径 `raw::decode` 只调用 LibRaw，失败后回落到相机内嵌 JPEG 预览（`raw.rs:123-141`）。

**C. Z6III OpenCL 重建（Windows-only）**

- 整个模块被 `#[cfg(all(windows, feature = "z6iii-raw-export"))]` 排除在非 Windows 之外（`aurora_raw/mod.rs:13-14`）。
- 产品调用点在 `decode_he_rgba16_with`：非 Windows 分支直接把结果置空并返回 `OpenClRoute::NotZ6iii`（`aurora_raw/mod.rs:99-105`）。
- OpenCL 宿主：`system_opencl.rs` 用 `#[link(name = "kernel32")]` + `LoadLibraryExW("OpenCL.dll")`（`z6iii/reconstruct/system_opencl.rs:7-22`），并且只挑 NVIDIA 设备（`z6iii/reconstruct/c1_adaptive_opencl.rs:62-80`）。
- 因此 **macOS 上这条路由不存在**；C1 的 OpenCL `.cl` 内核（`reconstruct.cl` 342 行、`highlight.cl` 45 行、`export_color.cl` 112 行）可以作为 Metal 的数学参考，但不是可运行代码。

### 1.2 产品实际调用链

```
FileEngine decode
  └─ raw16_source(path)                       file_engine.rs:645-651
       └─ raw_cache::get_or_decode_source16   raw_cache.rs:128
            └─ raw::decode16_traced           raw_cache.rs:116-125 → raw.rs:204
                 ├─ LibRaw Session::open/unpack/bayer   raw.rs:214-229
                 └─ aurora_raw::decode_he_rgba16_with   raw.rs:231
                      ├─ Z6III OpenCL (仅 Windows+feature)  aurora_raw/mod.rs:99-112
                      ├─ X-Trans（必须有 libraw_frame）       aurora_raw/mod.rs:121-133
                      └─ 2x2 Bayer（必须有 libraw_frame）     aurora_raw/mod.rs:138-171
  └─ 若 16-bit 返回 None：raw::decode（LibRaw 8-bit）→ 内嵌 JPEG 预览
```

关键事实：`decode16_traced` 里的 `bayer` **只来自 LibRaw Session**（`raw.rs:214-229`），没有调用 `decode_bayer16`/`rust_raw`。因此：

- 没有 LibRaw 时，只有 **Z6III HE NEF** 能继续：`nikon_c1_stage_input(bytes, None)` 会走纯 Rust `c1_raw_he::decode_frame_cropped`（`aurora_raw/mod.rs:258-280`）。
- 没有 LibRaw 时，DNG / ARW / CR2 / CR3 / ORF / RW2 / RAF / 普通 NEF 都拿不到 `BayerFrame`，`generic_bayer_c1_stage_input` 返回 `None`（`aurora_raw/mod.rs:284-316`），最终 8-bit LibRaw 也失败。
- 即：**native 前端已经写好，但没有接入产品 16-bit 路由**。这是 macOS 最小修复点的核心。

---

## 2. 格式 × 解码路由矩阵

状态以 **当前 HEAD 的 macOS 目标** 为准。当前 `raw.rs` 有编译错误（`01-blockers.md:36-41`），所以 RAW 整体是“未实现（编译阻塞）”；下表描述“修好编译后、不装 LibRaw”的预期状态，并明确标注证据等级。

| 格式 / 变体 | 纯 Rust 路由 | LibRaw 路由 | macOS 现状（源码级） | 阻塞 / 备注 |
|---|---|---|---|---|
| Z6III NEF，HE（14-bit，6064×4040） | `c1_raw_he` HE 位流（`aurora_raw/c1_raw_he.rs:364-...`），`rust_raw/nef.rs:49-55` | 可用但非必需 | **产品路径接入（预计）**：仅依赖纯 Rust + CPU C1，无需 LibRaw | `raw.rs` 编译阻塞；z6iii 颜色阶段在 macOS 未编译；头文件/profile 校验极严（见 §3） |
| Z6III NEF，HE★ | 同一 `find_he_strip`，随后 profile 校验 | 同上 | **未验证**；源码倾向拒绝 | `c1_raw_he.rs:372-392` 要求精确 `CONTACT_INTOPIX_`、41 字节 picture、固定 PIH/CDT。没有 HE★ 样本，不能声称支持 |
| NEF 无损（compression 34713，经典 Huffman） | `rust_raw/nef.rs:243-341` | 可用 | **仅可编译/未验证**；产品未接 native | `decode16_traced` 不调用 native |
| NEF 有损（compression 34892） | 明确 `Unsupported`（`rust_raw/nef.rs:343-353`） | 需要 | **未实现（无 LibRaw 时）** | 只能 LibRaw |
| NEF 未压缩（compression 1） | `rust_raw/nef.rs:40-48` | 可用 | **仅可编译** | 同上 |
| DNG（1 / 32773 / 8 deflate / 7 lossless JPEG） | `rust_raw/dng.rs:199-246` | 可用 | **仅可编译** | 同上 |
| ARW（ARW2 32767、未压缩） | `rust_raw/arw.rs:42-69` | 可用 | **仅可编译** | ARW6/ARW1 明确 Unsupported（`arw.rs:61-63`） |
| CR2（lossless JPEG） | `rust_raw/cr2.rs:17-23` | 可用 | **仅可编译** | 只支持 compression 7 |
| CR3 | `rust_raw/cr3.rs:63` | 可用 | **仅可编译** | 未验证具体机型/压缩 |
| RAF（X-Trans，未压缩 / 14-bit packed） | `rust_raw/raf.rs:43-90` | 可用 | **仅可编译** | 仅部分 packing；未知布局返回 Unsupported |
| ORF | `rust_raw/orf.rs:79-144` | 可用 | **仅可编译** | 部分 14-bit 需要 MakerNote 参数，缺失即 Unsupported（`orf.rs:123-144`） |
| RW2 | `rust_raw/rw2.rs:116` | 可用 | **仅可编译** | 12/14-bit block packing |
| PEF / SRW / 3FR / FFF / MOS / IIQ | 无 | 需要 | **未实现（无 LibRaw 时）** | `is_raw` 认扩展名（`raw.rs:1146-1165`），但没有 native 解码器 |
| JPEG / PNG / TIFF / WebP | `image` crate | — | **产品路径接入（跨平台）** | `file_engine.rs:2600`；engine 依赖见 `crates/engine/Cargo.toml:27` |
| HEIF / HEIC | 无 | — | **未实现** | `heif.rs` 只有 Windows 装载器；`#[link(name="kernel32")]`（`heif.rs:106,117`）。`cargo check` 可能过，但最终链接缺 `-lkernel32` |
| JXL / C1 `.cop` | 无 | — | **未实现（编译阻塞）** | `jxl.rs:3` 无条件 `std::os::windows::ffi::OsStrExt`；`LoadLibraryW`（`jxl.rs:424-427`） |
| `.eip`（ZIP 包内 RAW） | 解包后交给上面的路由 | — | **未实现（随 RAW）** | `file_engine.rs:2560-2578` |
| 相机内嵌 JPEG 预览 | 纯 Rust JPEG 提取 | — | **产品路径接入（跨平台）** | `raw.rs:35-65` / `raw.rs:1146-1180` |

**测试资产约定**（已核实）：

- `.gitignore:7` 忽略 `/crates/engine/testdata/`；`.gitignore:13-14` 忽略 `*.NEF/*.nef`；`.gitignore:64` 忽略 `/vendor/`。
- `scripts/testdata-skips.txt` 存在，106 条非注释项，用于 CI 跳过依赖本地 testdata 的测试。
- 30 个测试引用 `AURORA_TEST_RAW`；它们在没有样本时不会给出任何通过证据。

---

## 3. Z6III 专项：哪些仅 OpenCL、哪些有 CPU oracle、能否移植 Metal

### 3.1 仅 OpenCL（没有 CPU 等价实现）

| 阶段 | 位置 | 说明 |
|---|---|---|
| Stage 1（Green 重建） | `z6iii/reconstruct/c1_adaptive_opencl.rs:116` | `with_stage1_opencl` 强制走 OpenCL（`render.rs:39`） |
| Highlight recovery | `c1_adaptive_opencl.rs:208`；`.cl` 在 `highlight.cl` | **仓库内没有 CPU 版本**；`recover_and_antialias_frame` 失败时静默跳过（`render.rs:196-206`） |
| AntiColorAliasing | `c1_adaptive_opencl.rs:191`；`reconstruct.cl` | **仓库内没有 CPU 版本**；失败时静默跳过（`render.rs:200-206`） |
| Export color | `c1_adaptive_opencl.rs:219`；`export_color.cl` | OpenCL 用于 neutral 导出；产品色彩另有 CPU `OutputClut`（`color/output.rs:216-256`） |
| Stage 3 GG blur | `c1_adaptive_opencl.rs:156` | 与 `stage3/ggblur.rs` 的 CPU 版并行存在 |

### 3.2 有 CPU oracle（纯 Rust，可跨平台）

- **C1 Stage 1/2/3 的 CPU 实现**：`z6iii/reconstruct/c1_stage1_green.rs`、`c1_stage1_direction.rs`、`c1_stage2_basic.rs`、`stage3/{mod,rb,finish,ggblur}.rs`，均为纯 Rust、无 FFI（`stage3/mod.rs:1` 明确 `#![forbid(unsafe_code)]`）。
- **可移植副本**：`aurora_raw/c1_*.rs`（`c1_raw_he.rs` 通过 `#[path]` 引入一套同名文件），已经无条件编译并在 macOS 可用（`aurora_raw/c1_raw_he.rs:3-23`）。
- **色彩阶段**：`z6iii/color/**` 全是纯 Rust，`#![forbid(unsafe_code)]`（`color/mod.rs:1`）。
- **线性 CA / 线性色噪**：`render.rs:251-...` 与 `render.rs:406` 是纯 Rust，只是被 Windows 模块门一起挡掉。
- **几何**：`z6iii/geometry16.rs` 纯 Rust。
- **Bayer→RGB 主循环**：`aurora_raw/mod.rs:369-422` 用 rayon 按 128×128 tile 调 `c1_raw_he::reconstruct_planar_active_mirror_roi`，是纯 CPU 产品路径。

### 3.3 CPU oracle 与 Windows 参考实现并不一致（必须处理）

两套同名文件不是简单复制：

| 文件 | 差异 | 影响 |
|---|---|---|
| `aurora_raw/c1_stage2_basic.rs:55-60` vs `z6iii/reconstruct/c1_stage2_basic.rs:51-64` | 斜向候选用 `one_third`（`0x3eaa_aaab`）vs `oblique_third`（`0x3eaa_aaaa`）；选中项取 `rev().find()`（最后一个）vs `position()`（第一个） | 逐像素不同 |
| `aurora_raw/c1_stage1_green.rs:48-71` vs `z6iii/.../c1_stage1_green.rs:52-83` | 浮点乘加顺序/融合不同（`add_product`） | 低位与高 ISO 不同 |
| `aurora_raw/c1_stage1_direction.rs:3-22` vs `z6iii/.../c1_stage1_direction.rs:3-22` | 中心预测用 `predicted(center)`（带 gain）vs `center_prediction`（全 gain + bias） | 方向选择/预测值不同 |
| `aurora_raw/c1_stage3_params.rs` vs `z6iii/.../c1_stage3_params.rs` | 后者拆出 `offset_for_noise`，公式看起来等价 | 需数值对拍确认 |

`c1_stage1_params.rs`、`c1_nikon_he.rs`、`nikon_he_bitstream.rs` 完全一致（diff 0 行）。

结论：**不能说“macOS 的纯 Rust C1 链就是 Z6III OpenCL 的 CPU oracle”**。在把 Metal 结果与 CPU 对拍之前，先要选定唯一参考实现。`z6iii/reconstruct/*` 的注释明确说它匹配 OpenCL（例如 `c1_stage2_basic.rs:50` “Ordinary OpenCL stage2 uses a shorter literal”），所以它更像是 Windows 参考；`aurora_raw/*` 是较早/不同的一版。

### 3.4 Z6III 重建数学对 Metal 的可移植性

- 数学本身可移植：所有输入/输出都是 `u16`/`f32`/`[f32;3]`/`[f32;4]`，无平台类型；OpenCL C 内核（`reconstruct.cl`、`highlight.cl`、`export_color.cl`）可直接翻译成 MSL。
- 但有三处硬约束必须先解决：
  1. `z6iii` 模块整体被 Windows 门挡住（`aurora_raw/mod.rs:13-14`），CPU 部分也要挪到跨平台 cfg。
  2. `c1_adaptive_opencl.rs` / `system_opencl.rs` 只能在 Windows 编译，需要拆成 `cfg(windows)` 的 OpenCL 后端 + 跨平台 CPU 后端。
  3. **Highlight recovery 与 AntiColorAliasing 没有 CPU oracle**；Metal 移植时只能以 `.cl` 为参考，必须先补 CPU 参考实现才能做逐像素对拍（任务书 §4.4/§5.3 要求）。

### 3.5 Z6III 路由判定的 macOS 语义缺口

- `is_z6iii_nef` 只存在于 Windows 模块（`z6iii/reconstruct/c1_nef_he.rs:121-133`），顶层 `aurora_raw/c1_nef_he.rs` 没有。
- `decode_route` 在非 Windows 恒设 `z6iii_nef=false`、`z6iii_feature=false`、`opencl=NotCompiled`（`raw.rs:826-866`）。
- `decode_he_rgba16_with` 非 Windows 返回 `OpenClRoute::NotZ6iii`（`aurora_raw/mod.rs:103-105`），`cpu_fallback(NotZ6iii)` → `AuroraRoute::Cpu`（`aurora_raw/mod.rs:51-58`）。
- 后果：即使 macOS 用 CPU C1 正确解了 Z6III HE，provenance 仍是 `z6iii_linear=false`、`StageShape::AuroraGeneric`，**不会启用 Z6III pre-matrix**（`raw.rs:701-704`、`raw.rs:403-410`），也不会走 Z6III 颜色阶段。
- 这不是“少一个优化”，而是**色彩正确性缺口**：C1 重建出的相机线性 RGB 需要 Z6III pre-matrix + CLUT，否则与 Windows 产品输出不同。

---

## 4. 色彩管线盘点

### 4.1 白平衡

| 项 | 位置 | 平台 |
|---|---|---|
| Nikon MakerNote as-shot RGB（tag 0x000c，四个有理数，要求两个 G 相等） | `aurora_raw/c1_nef_he.rs:178-205` | 纯 Rust，跨平台 |
| Z6III 全局 WB 归一化 | `z6iii/color/mod.rs:205` | 纯 Rust，但被 Windows 模块门挡住 |
| 通用 Bayer 的 `cam_mul` | `raw.rs:1860-1863`（LibRaw 偏移） / `generic_bayer_c1_stage_input`（`aurora_raw/mod.rs:346-363`） | 需要 LibRaw 或 native BayerFrame |
| gains ↔ Kelvin/Tint | `wb_gains.rs`（纯 Rust）、`c1_model.rs` 的 `gains_to_kelvin_tint` 系列 | 跨平台 |
| C1 Z6III 采样表（861 点，x/v 网格） | `c1_wb_oracle.rs:1-11` + 常量表 | 纯数据，跨平台；**只用于显示，渲染直接用 gains**（`c1_wb_oracle.rs` 头注释） |

注意：不要把旧样本的 4779.55 K 等值硬编码；任务书 §5.6 明确要求以真实样本 + 独立参考为准。

### 4.2 相机矩阵 / ICC / film curve

| 项 | 位置 | 平台 / 资产依赖 |
|---|---|---|
| Z6III pre-matrix 常量 | `z6iii/color/mod.rs:26-32` | 纯 Rust，跨平台 |
| Camera→working CLUT（49³） | `z6iii/color/output.rs:216-262`（CPU `from_camera49` / `sample_srgb`） | 需要相机 ICC 文件 |
| ICC 解析 | `icc.rs:430-658`（`parse_icc`、`parse_mft2`） | 跨平台 |
| ICC 目录发现 | `icc.rs:671-708` | **Windows 路径**：`C:\Program Files\Capture One\Capture One\Color Profiles`、`C:\Windows\System32\spool\drivers\color`；否则 `testdata` |
| film curve `.fcrv` 目录 | `filmcurve.rs:139-149` | **Windows 路径**：`C:\Program Files\Capture One\Capture One\Film Curves`；可用 `AURORA_FILM_CURVES` 覆盖 |
| EXIF model → 默认曲线 | `camera_curve.rs:36-52` | 依赖 `filmcurve::curve_dir` |
| C1 `coreConfig.cdx` | `camera_curve.rs:66-75`；`camera_profile.rs:166-175` | **Windows 路径**：`C:\Program Files\Capture One\Capture One\coreConfig.cdx`；可用 `AURORA_CORECONFIG` |
| presets/styles 根目录 | `presets.rs:579-585` | **Windows 路径**：`C:\Program Files\Capture One\Capture One`；可用 `AURORA_C1_DIR` |
| Z6III 专用导出资产 | `z6iii/desktop.rs:41-70,101-102` | **Windows 路径 + `ImgCoreDll.dll`**，读取 DLL 内固定偏移（40_613_632 等）的 banding 表 |
| BandingSuppression 表 | `c1_banding.rs:139-160` | 纯 Rust，但数据取自 **`ImgCoreDll.dll`**；macOS 找不到即返回 `None`，banding 阶段关闭 |
| 相机噪声参数包 | `camera_params.rs:21-24`（`include_str!` 内嵌 JSON） | 跨平台，无需安装目录 |

macOS 需要的等价物（**不要写死**，必须先核实本机 Capture One 实际布局）：

- Capture One.app 资源根（可能是 `/Applications/Capture One.app/Contents/...`，未验证）；通过 `AURORA_ICC_DIR`、`AURORA_FILM_CURVES`、`AURORA_CORECONFIG`、`AURORA_C1_DIR`、`AURORA_Z6III_C1_DIR` 提供覆盖。
- `ImgCoreDll.dll` 是 Windows PE，**macOS 不存在等价文件**。banding 表 / Z6III neutral 导出的私有资产要么由仓库所有者提供合法 mac 版本，要么明确标记“该能力在 macOS 不可用”。这些资产属于 AGENTS.md 禁止提交的私有模型/曲线/ICC 范畴，不得进入仓库。

### 4.3 CLI / desktop helper 的 Windows 门槛（已核实）

- `z6iii/cli.rs:1` 是 `#[cfg(windows)]`；非 Windows `main` 直接 `bail!("... requires Windows and an OpenCL GPU")`（`cli.rs:29-33`）。
- `z6iii/color_trace_cli.rs:14` 同样 `#[cfg(windows)]`，非 Windows 报错（`:109-115`）。
- 两个 bin 在 `crates/engine/Cargo.toml:12-20` 需要 `z6iii-raw-export` feature。
- `z6iii/desktop.rs` 本身没有 cfg，但只从 Windows feature 模块调用；其中 `Assets::load` 固定读 `ImgCoreDll.dll`（`desktop.rs:61-63`）与 `C:\Program Files\...` 默认根（`desktop.rs:101-102`）。

### 4.4 产品色彩阶段的平台门槛

- `mock/process.rs:2018-2036`：`c1_color: Option<Z6iiiColorStage>` 整个字段和构造都在 `#[cfg(all(windows, feature = "z6iii-raw-export"))]` 内。
- `mock/develop_gpu.rs:381-382`：`from_stage` 同样只 Windows。
- 因此 **macOS 即使解出 Z6III 线性 RGB，也不会进入 Z6III 色彩阶段**（pre-matrix / film / camera CLUT），会走通用 develop。

---

## 5. RAW 缓存有效性

### 5.1 键的实际组成

缓存文件名 API：`cache_name16_for_identity`（`raw_cache.rs:315-341`）：

```
RAW16_MAGIC, path, identity.len, identity.mtime_nanos,
identity.content, identity.full, long_side, route
```

`DecodeRoute`（`raw.rs:799-814`）包含：

- `pipeline_fingerprint`：由 `DecodeStageRevisions` 的 7 个 stage（demosaic / highlight_recovery / chromatic_aberration / noise_model / z6iii_reconstruction / orientation_active_area / output_packing）按 route shape 组合（`raw.rs:327-347`、`raw.rs:373-419`）。
- `c1_raw_enabled`（`AURORA_C1_RAW`）
- `z6iii_nef`
- `z6iii_feature`（`cfg!(all(windows, feature="z6iii-raw-export"))`，`raw.rs:828`）
- `libraw_available`
- `opencl`（`NotCompiled` / `Available` / `Unavailable`，`raw.rs:816-824`）
- `demosaic`

持久化头另有 `route_digest`（`route_digest`，`raw_cache.rs:428-432`）与 `pipeline fingerprint`，读取时逐一比对（`raw_cache.rs:1096-1104`）。格式版本由 `RAW16_MAGIC = b"AUR16B9\0"`（`raw_cache.rs:28`）承担。

### 5.2 明确**没有**进入键的内容

- **OS / 架构**：没有 `target_os`、`arch`、`endianness` 字段。大端读取有转换（`raw_cache.rs:1158-1161`），但键不变。
- **实际后端（CPU vs 未来 Metal）**：`DecodeBackend` 只写进 header provenance（`raw.rs:707-715`、`raw_cache.rs:1285`），**不在 route/key 里**。命中时只校验 `provenance.consistent_with(route)`：`is_current` + `z6iii_linear` 与 `z6iii_nef` 的关系 + `CameraLinear16` 与 `c1_raw_enabled` 的关系（`raw.rs:783-796`）。
- **精度**：body 固定 RGB16（3×u16），`pixel_space` 只写 header；没有“CPU f32 vs Metal f16 中间精度”字段。

### 5.3 结论

- 算法版本：**有**（stage revisions + magic）。
- 后端 / 平台：**只有间接能力位**（`z6iii_feature`、`libraw_available`、`opencl`）。macOS 上这些位与 Windows 不同，所以大多数跨平台文件不会碰撞；但**同一平台内 CPU 与未来 Metal 会碰撞**。
- 因此新增 Metal RAW 后端时，必须二选一：
  1. 把后端/精度加入 `DecodeRoute` 并提升 `RAW16_MAGIC`；或
  2. 先做 CPU↔Metal 等价性证明，再共享缓存。任务书 §5.7 也是这个要求。

---

## 6. macOS 专项风险

### 6.1 缓存目录

`default_raw16_dir` 非 Windows 只认 `XDG_CACHE_HOME` 或 `~/.cache/aurora-studio/raw`（`raw_cache.rs:85-104`）。macOS 约定是 `~/Library/Caches/...`；应增加 `#[cfg(target_os = "macos")]` 分支，保留 `AURORA_RAW_CACHE_DIR` / `AURORA_DISABLE_RAW_CACHE` 覆盖。

### 6.2 路径处理

- 好的一面：`is_raw` 只对扩展名做 `to_ascii_lowercase`，不碰路径主体（`raw.rs:1146-1150`）；`c1import::store_key` 把 `\`→`/`、`:`→`_`，Mac 路径可用（`c1import.rs:197-204`）；`library_ops` 只在 `cfg!(windows)` 时对 key 小写（`library_ops.rs:41`）。
- 风险点：LibRaw 非 Windows 打开路径用 `path.to_string_lossy()` → `CString`（`raw.rs:1447-1448`）。非 UTF-8 文件名会被有损替换。**优先走 `libraw_open_buffer`**（`raw.rs:1573-1591`）可以完全避开路径编码问题；产品 16-bit 路由已经优先 buffer。
- 不要引入 `to_lowercase()` 做路径比较；不要拼接 `\` 分隔符。

### 6.3 Capture One catalog / session 布局

- 只按扩展名解析：`.cocatalogdb` / `.cosessiondb`，支持 bundle 目录、`.cos` session 文件夹、包含数据库的父目录（`c1import.rs:133-197`）。
- 测试里已有 Mac 路径 fixture（`/Users/<user>/Pictures/Shoot/...`，`c1import.rs:405-424`），说明文本路径可按 Mac 形式存储。
- 只读保证：`c1catalog.rs:3-10,87` 明确 read-only 打开 SQLite；`c1import.rs:4` 明确不写数据库；`library_ops.rs:237` 拒绝 C1 catalog 的文件系统操作；`xmp.rs:125` 拒绝写入 C1 bundle。
- 仍需真机核实：macOS Capture One 的 `.cos` 内部结构、SQLite 版本、WAL 行为；没有 Mac 样本，标“未验证”。

### 6.4 动态库 / 链接

- `heif.rs` 无 cfg 使用 `#[link(name = "kernel32")]`（`heif.rs:106,117`）。`cargo check` 可能通过，但 `cargo build` 在 macOS 会因 `-lkernel32` 链接失败。**这是 `01-blockers.md` 未列出的额外阻塞**（静态结论，未运行链接验证）。
- `jxl.rs` 无条件 `std::os::windows::ffi::OsStrExt`（`jxl.rs:3`），`cargo check` 已失败（`01-blockers.md:30-34`）。
- `heif_write.rs` 已有正确的 `cfg(windows)` / `cfg(not(windows))` 模板（`heif_write.rs:52-121`），但非 Windows 候选库只有 `libheif.so.1` / `libheif.so`（`heif_write.rs:124-134`），**缺 `.dylib`**。
- LibRaw 非 Windows 候选只有 `.so`（`raw.rs:1265-1268`），**缺 `.dylib`**，且没有 `@rpath` / `Contents/Frameworks` 搜索。

### 6.5 只读源文件

- RAW 通道：`std::fs::read` / mmap 只读；缓存写入独立目录；未见回写源文件。
- C1 通道：如上，read-only 有显式保证。
- 导出目标：`z6iii/cli.rs` 用 `create_new(true)` 拒绝覆盖（`cli.rs:20-21`）；产品导出走 recipe，不在此审计范围。

---

## 7. 优先工作项（Z6III NEF 优先，按最小正确改动排序）

> 所有权：`raw.rs` 当前归 Mac-A 修编译，随后移交 Mac-D（`OWNERSHIP.md:21`）。下列 1、2 项涉及 `raw.rs`，需先与 Mac-A 交接。

### P0-1 修 `raw.rs` 的 LibRaw 平台类型（编译前置）

- 现状：`start()` 无条件返回 `(*mut c_void, &'static Api, Vec<u16>)`（`raw.rs:1440`），非 Windows 分支实际是 `CString`（`raw.rs:1447-1448`），导致 1453/1481/1597/1680/1774 五处 E0308。
- 最小改动：把路径载体拆成平台无关的枚举（或 `cfg` 返回类型），例如 `enum LibRawPath { Wide(Vec<u16>), Utf8(CString) }`；仅 5 个调用点按平台取指针。不要改 Windows 行为。
- 同时补 `LIBRARY_NAMES` 的 macOS 项（`libraw.dylib` / `libraw.23.dylib`），并加 `AURORA_LIBRAW_DIR` 之外的 `@loader_path` / `Contents/Frameworks` 搜索；**不要写死 `/opt/homebrew`**（`OWNERSHIP.md:29`）。

### P0-2 把 native `rust_raw` 接入产品 16-bit 路由（最小产品收益）

- 现状：`decode16_traced` 的 `bayer` 只来自 LibRaw（`raw.rs:214-229`），native 前端只在 `decode_bayer16` 用（`raw.rs:985`）。
- 最小改动：在 `session`/`bayer` 为 `None` 时，用已读入的 `ctx.bytes` 调 `crate::aurora_raw::rust_raw::decode_bytes(&ctx.bytes)`（`rust_raw/mod.rs:80`），把结果作为 `decode_he_rgba16_with` 的 `libraw_frame`。这样：
  - Z6III HE 走纯 Rust C1；
  - DNG/ARW/CR2/CR3/ORF/RW2/RAF/经典 NEF 也能在没有 LibRaw 时进入 C1 链。
- 不要改变 LibRaw 可用时的优先级；不要重复读文件（用 `bytes`，不用 `path`）。
- 需要对应更新 route/provenance（见 P0-3）。

### P0-3 让 macOS 正确识别 Z6III 并标记 `z6iii_linear`

- 现状：`is_z6iii_nef` 只在 Windows 模块（`z6iii/reconstruct/c1_nef_he.rs:121-133`）；非 Windows `z6iii_nef=false`、`NotZ6iii`、`AuroraRoute::Cpu`（`raw.rs:826-847`、`aurora_raw/mod.rs:103-105`）。
- 最小改动：
  1. 把纯 Tiff 的 `is_z6iii_nef` 移到顶层 `aurora_raw/c1_nef_he.rs`（可复用 `Tiff`）；
  2. 非 Windows 的 `decode_he_rgba16_with` 在识别到 Z6III HE 时返回 `OpenClRoute::Unavailable`（或新增 `NotCompiled`），让 `cpu_fallback` 得到 `Z6iiiCpuOpenClUnavailable` → `z6iii_linear=true`（`raw.rs:633-655`）；
  3. `decode_route` 的 `z6iii_feature` 改为“CPU 重建已编译”，使 `StageShape::AuroraZ6iii` 生效（`raw.rs:403-410`）。
- 这会同时修正缓存键的 stage shape，避免 Z6III 缓存与通用 Bayer 缓存混用。

### P0-4 让 Z6III CPU 重建 + 颜色阶段在 macOS 编译

- 最小改动：
  1. 把 `z6iii` 模块的 cfg 从 `all(windows, feature=...)` 改为 `feature=...`（或新增独立 `z6iii-cpu` 子模块），只把 `c1_adaptive_opencl` / `system_opencl` / `desktop` / `cli` 留在 Windows；
  2. `color/**` 与 `reconstruct/stage3`、`c1_stage1_*`、`c1_stage2_basic`、`c1_adaptive_gg` 保持跨平台；
  3. `mock/process.rs:2018-2036` 的 `c1_color` 构造改为跨平台 cfg（`develop_gpu::from_stage` 仍留 Windows）。
- 不要用空实现冒充 OpenCL：macOS 上没有 OpenCL stage1/highlight/AA 时，必须明确走 CPU 或声明缺失。

### P0-5 统一两套 CPU 重建实现（对拍前必须做）

- 现状：`aurora_raw/c1_stage1_green.rs`、`c1_stage1_direction.rs`、`c1_stage2_basic.rs`、`c1_stage3_params.rs` 与 `z6iii/reconstruct/*` 有语义差异（§3.3）。
- 最小改动：以 Windows 参考（`z6iii/reconstruct/*`，注释明确对齐 OpenCL）为准，把顶层副本改成同一实现；或反过来，但必须在真实样本上选定并记录依据。改动后提升 `DecodeStageRevisions::demosaic` / `z6iii_reconstruction`（`raw.rs:339-347`）。

### P0-6 补 Highlight / AntiColorAliasing 的 CPU oracle（或明确关闭）

- 现状：只有 OpenCL（`c1_adaptive_opencl.rs:191,208`），没有 CPU 版。
- 最小正确步骤：先把 `.cl` 的数学落到 CPU 参考实现并用合成/真实样本对拍；在此之前，macOS 明确标记这两个阶段“未执行”，不得声称与 Windows 画质一致。

### P0-7 修 HEIF/JXL 的平台装载

- `heif.rs`：把 `LoadLibraryW` 分支用 `#[cfg(windows)]` 包住，非 Windows 用 `dlopen`（照抄 `heif_write.rs:52-121` 模板），候选加 `.dylib`。
- `jxl.rs`：去掉无条件 `std::os::windows::ffi`；非 Windows `dlopen` + `libjxl.dylib`。
- `heif_write.rs` 候选加 `libheif.dylib`（`heif_write.rs:124-134`）。
- 缺库时返回真实原因（`Unsupported` + 路径），不得 silent 输出替代格式。

### P0-8 macOS 缓存目录与色彩资产发现

- `raw_cache::default_raw16_dir` 增加 `~/Library/Caches/Aurora Studio/raw` 分支（`raw_cache.rs:85-104`）。
- 为 `icc.rs`、`filmcurve.rs`、`camera_curve.rs`、`camera_profile.rs`、`presets.rs`、`c1_banding.rs` 增加 macOS 查找顺序：先 `AURORA_*` 覆盖，再 Capture One.app 资源目录（**先核实实际路径**），最后明确失败。不要把 Windows 路径当作唯一默认。
- `ImgCoreDll.dll` 相关（banding 表、Z6III neutral 导出）在 macOS 没有等价物：必须由仓库所有者提供合法 mac 资产，或明确标记能力缺失；不得提交私有资产。

### P0-9 合成测试 + 文档

- 现有 `generic_bayer_aligns_all_cfa_phases_through_c1_stage123`（`aurora_raw/mod.rs:506-551`）可在 macOS 跑，作为 C1 CPU 链的合成回归。
- 新增：无 LibRaw 时 `decode16_traced` 走 native 的路由测试（用合成 DNG / 合成 NEF，不依赖真实样本）；缓存键在 `z6iii_nef` 翻转时必须变化。
- 真实 Z6III HE/HE★ 的画质与色彩只能等合法样本，保持“未验证”。

---

## 8. 必须保持的 RAW / 色彩 / 缓存不变量

1. **源文件只读**：解码、缓存、导出不得回写原图；缓存只写 `AURORA_RAW_CACHE_DIR` / 平台缓存目录；C1 数据库只读（`c1catalog.rs:3-10`）。
2. **CFA phase / 黑白电平 / orientation / active crop**：C1 链必须用 `NikonCrop` + `cfa_phase()`（`aurora_raw/c1_raw_he.rs:95-178`）与 `orient_rgba16`（`aurora_raw/mod.rs:437-460`）；不要用缩放后上采样冒充。
3. **Z6III 几何固定**：重建要求 `(6064,4040)` HE strip（`c1_raw_he.rs:368-370`）且 active crop `(6048,4032)`、`left/top >= 2`（`z6iii/render.rs:41`）；OpenCL 与 CPU 都必须一致。
4. **Z6III profile 白名单**：`CONTACT_INTOPIX_\xef\xc0` + 固定 PIH/CDT（`c1_raw_he.rs:372-392`）。不要为了“支持 HE★”放宽到未验证 profile。
5. **WB**：Z6III 使用 MakerNote 0x000c 的四个有理数且两个 G 必须相等（`c1_nef_he.rs:178-205`）；通用 Bayer 使用 `cam_mul` 归一化（`aurora_raw/mod.rs:346-363`）。不要用 C1 UI 的 Kelvin/Tint 反推渲染 gains（`c1_wb_oracle.rs` 只是显示表）。
6. **色彩阶段顺序**：pre-matrix → film（cubic Hermite）→ highlight/hue coupling → camera transfer → 49³ Camera→working CLUT（`color/mod.rs:90-145`，`color/output.rs:216-256`）。顺序不可交换。
7. **Z6III pre-matrix 只在 `z6iii_linear && CameraLinear16` 时启用**（`raw.rs:701-704`）。通用 Bayer / X-Trans / LibRaw / 预览不得启用。
8. **缓存键语义**：任何改变像素的 stage 改动必须提升对应 `DecodeStageRevisions` 字段（`raw.rs:327-347`）；任何持久化布局改动必须提升 `RAW16_MAGIC`（`raw_cache.rs:28`）；CPU/Metal 若不共享语义，必须进键或先做等价性证明。
9. **Provenance 路由**：`DecodeProvenance.z6iii_linear` 必须与实际解码路由一致；缓存命中时必须通过 `consistent_with`（`raw.rs:783-796`）。
10. **不静默降级**：LibRaw 缺失、OpenCL 缺失、HE 变体不支持时，返回真实原因（`Unsupported` / `Unavailable` / `Failed`），不得空实现返回成功，也不得用 8-bit 预览冒充 16-bit sensor decode。
11. **测试资产**：`crates/engine/testdata/` 与 `vendor/` 永不提交；真实照片/EXIF/ICC/fcrv 不进仓库（AGENTS.md:54-58）。

---

## 9. 未验证声明（必须显式保留）

以下结论**没有**真机或样本证据，不能写成“通过”：

1. 本机 **没有 Z6III NEF 样本**；HE 与 HE★ 的实际解码成功、像素、色彩、直方图全部未验证。
2. 本轮**未运行** `cargo check/build/test`；编译状态引用 `01-blockers.md`，不是复现。
3. macOS arm64 上 LibRaw 的 **动态库 ABI / `libraw_data_t` 偏移**（`raw.rs:1688-1731`）未验证；只有 x64 生成的 offsets 注释。
4. 纯 Rust 原生前端对 DNG/ARW/CR2/CR3/RAF/ORF/RW2 的**真实文件**行为未验证；现有测试多为合成容器。
5. `heif.rs` 的 `-lkernel32` 链接失败是**静态推断**（`heif.rs:106,117`），未实际执行链接。
6. macOS Capture One 的 profile / film curve / coreConfig / session 布局**未核实**；本文只给出“必须核实 + `AURORA_*` 覆盖”的要求。
7. Metal 对 Z6III 重建/色彩的等价性**未做任何数值对拍**；在统一 CPU oracle（P0-5）与补齐 highlight/AA CPU 参考（P0-6）之前，不能声称可移植即等价。
8. 性能（冷/暖帧、P50/P95、内存）不在本审计范围，未测量。

---

## 10. 一句话结论

macOS RAW 的真正阻塞不是“LibRaw 没有 Mac 版”这一点，而是三层叠加：
**(a)** `raw.rs` 的 LibRaw 路径类型/库名只按 Windows 写死，导致整个 RAW 模块在 macOS 不编译；
**(b)** 已存在的纯 Rust 原生前端没有接入产品 16-bit 路由，且 Z6III 在非 Windows 被误判为“非 Z6III”，丢掉了 pre-matrix 与色彩阶段；
**(c)** Z6III 的 OpenCL 专用阶段（stage1/highlight/AA/export）与颜色资产路径被 Windows 门整体挡住，而仓库内两套 CPU 重建实现并不一致。

最小、最有价值的第一步：修 `raw.rs` 平台类型 + 把 `rust_raw::decode_bytes` 接进 `decode16_traced`，并让 macOS 正确标记 `z6iii_linear`，使 **Z6III HE NEF 在不依赖 LibRaw 的情况下走纯 Rust CPU 链路**；随后再处理颜色阶段与缓存键。以上第一步的像素正确性仍需真实样本验证。

---

## 11. 实施记录 — 缓存目录与 HEIF 装载（Mac-D）

改动仅三处；未触碰 Mac-A 在途文件（`raw.rs`、`jxl.rs`、`aimask/ort.rs`、`portrait_stage.rs`、`mock/{gpu_runtime,pixel_gpu,mod}.rs`、`blemish_gainmap_cuda.rs`）。未改 `RAW16_MAGIC`、缓存键语义、`Cargo.toml`/`Cargo.lock`，未提交。

### 11.1 `crates/engine/src/raw_cache.rs`
- `default_raw16_dir`（`raw_cache.rs:85`）新增 `#[cfg(target_os = "macos")]` 分支（`raw_cache.rs:98-108`）：返回 `$HOME/Library/Caches/Aurora Studio/raw`。
- 既有覆盖仍优先：`AURORA_DISABLE_RAW_CACHE`、`AURORA_RAW_CACHE_DIR`（`raw_cache.rs:86-91`，未改）。
- Windows 分支保持原样（`raw_cache.rs:92-97`）；Linux/其他非 Windows 的 `XDG_CACHE_HOME` → `~/.cache/aurora-studio/raw` 改为 `#[cfg(all(not(windows), not(macos)))]`（`raw_cache.rs:109-115`），逻辑未变。

### 11.2 `crates/engine/src/heif.rs`
- 原先无条件的 `#[link(name = "kernel32")]`（旧 `heif.rs:106,117`）已收口：Windows 的 `LoadLibraryW`/`GetProcAddress`/`FreeLibrary` 现位于 `#[cfg(windows)]` extern 块（`heif.rs:111-117`）。
- 新增 `#[cfg(not(windows))]` 的 `dlopen`/`dlsym`/`dlclose`（`heif.rs:119-124`），与 `heif_write.rs:58-62` 同形；`Drop` 按平台调用 `FreeLibrary` 或 `dlclose`（`heif.rs:203-216`）。
- 新增非 Windows 候选与搜索（`heif.rs:129-163`）：`AURORA_HEIF_LIB` → 可执行文件相对的 `Contents/Frameworks` 与二进制同目录（`current_exe()` 计算，作为 `@loader_path` 的等价物，无硬编码）→ 裸名交给系统加载器。macOS 候选为 `libheif.dylib`、`libheif.1.dylib`；其他非 Windows 仍为 `libheif.so.1`、`libheif.so`。
- 符号解析统一走 `resolve`（`heif.rs:165-180`）；Windows 诊断文本保持 `libheif.dll 缺少导出 …`（`LIB_HEIF`，`heif.rs:105-109`），非 Windows 为 `libheif 缺少导出 …`。
- 缺库仍走真实错误路径：`open_first_available` 汇总已尝试路径后 `bail!`（`heif.rs:189-201`），无静默回退、无格式替代。
- Windows 专属测试 `missing_dll_returns_clear_error`（自身声明 `#[link(name="kernel32")]`）加 `#[cfg(windows)]`（`heif.rs:763-765`），避免 macOS 测试目标残留 kernel32 链接。

### 11.3 `crates/engine/src/heif_write.rs`
- 非 Windows 候选增加 macOS 分支（`heif_write.rs:128-133`）：`libheif.dylib`、`libheif.1.dylib`；Windows `.dll` 与其他非 Windows `.so` 不变。未重构该文件。

### 11.4 编译与库存在性（实测）
- `cargo check -p aurora-engine --target aarch64-apple-darwin`：**通过**，exit 0，0 error（197 warnings，均为既有 dead_code / FFI-safe 警告）。Mac-A 在途文件当前也通过，无需忽略的错误。
- `cargo check -p aurora-engine --tests --target aarch64-apple-darwin`：**通过**，exit 0。
- 本机确有 libheif：`/opt/homebrew/lib/libheif.dylib`、`libheif.1.dylib`、`libheif.1.23.7.dylib`（symlink 指向 Cellar 1.23.7）。裸名 `libheif.dylib` 能否被 dlopen 找到取决于进程默认搜索路径是否含 `/opt/homebrew/lib`；本次未加载实际库。

### 11.5 仍未验证
- **未做运行时装载/解码测试**：没有用本机 libheif 实际解码或写入 HEIC（需合成或合法样本；本轮只做 check）。
- 非 Windows `open_first_available` 只尝试首个可 `dlopen` 的候选，不预检导出；首个能加载但缺导出的库会让 `symbol!` 明确报错而不回退到后续候选（与旧 Windows 单库语义一致，弱于 `heif_write::load` 的导出探测）。
- bundle 相对搜索只覆盖 `Contents/MacOS/../Frameworks` 与二进制同目录；未验证真实 `.app` 打包布局与代码签名后的 `dlopen` 行为。
- Windows 分支只做源码级保持，未在 Windows 上编译/运行。
- 依指示未运行 `cargo build --release`；链接阶段未验证（但 `heif.rs` 已不再对非 Windows 引入 `-lkernel32`）。
