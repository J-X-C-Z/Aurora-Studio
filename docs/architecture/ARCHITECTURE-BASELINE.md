# Aurora Studio 公共架构基线（A0）

基线时间：2026-11-25
仓库工作副本：`<worktree: J-X-C-Z/架构>`（独立 worktree，分支 `J-X-C-Z/架构`）
基线 SHA：`8118a4ae2ef6c3c440908b99f0f18cd044ea990c`
主分支：`main` = 同一 SHA（`origin/main` 相同）
上一参考 commit：`9bf908e53fd029c12f4f454546c61064f9ef9010`（本副本已包含其后一次 UI 合并）
记录方式：只以本副本实际源码和实际命令为准；旧报告、记忆和截图只作线索。

## 1. 环境与命令结果

| 项目 | 值 |
|---|---|
| 主机 | macOS 27.2.1 / Apple Silicon arm64 |
| Rust | rustc 1.98.1 (48a229cea 2026-09-01) / cargo 1.98.1 |
| workspace | `crates/engine` (aurora-engine)、`crates/app` (portrait-app)、`crates/portrait` (aurora-portrait) |
| `cargo metadata --locked --no-deps` | 通过（3 个 package，edition 2024） |
| `cargo check --workspace --all-targets --locked` | **基线失败，已由本 lane 的最小平台门控 patch 修复后通过**（见 §2） |
| `cargo test` | 本轮尚未完成全量执行，见 `ARCHITECTURE-VALIDATION.md` |
| 外部 fixture | `crates/engine/testdata/` 为空；`vendor/` 不存在；未运行 `external-fixtures` |

Git 状态：开工时工作树干净。本 lane 的所有改动都是未提交的工作副本改动，未 commit、未 push。

## 2. 开工即发现：非 Windows 构建基线不成立

静态审计假设“编译通过”是基线，但本副本在 macOS 上 `cargo check` 失败，共 23 个错误，分布在：

- `crates/portrait/src/cuda.rs:3472-3474`：`HostRegistration`/`PinnedBuffer`/`Stream` 无条件 `pub use driver::*`，而 `driver` 是 `#[cfg(windows)]`。
- `crates/portrait/src/cuda_v2fuseoffset.rs:10`：无条件导入 Windows-only 的 `V2Warp`（CPU 参考路径其实不需要它）。
- `crates/engine/src/mock/gpu_runtime.rs`、`mock/pixel_gpu.rs`、`blemish_gainmap_cuda.rs`：在非 Windows 上引用 `Nvidia`/`DeviceBuffer`/`PinnedBuffer`/`Stream`，但 crate 没有非 Windows 适配面。
- `crates/engine/src/aimask/ort.rs`：`load_library`/`get_base`/`dlsym` 只有 Windows 与 Linux 分支，macOS 缺失；`.dylib` 候选缺失。
- `crates/engine/src/raw.rs:1440`：`start()` 返回类型固定 `Vec<u16>`，非 Windows 分支却构造 `CString`。
- `crates/engine/src/jxl.rs`：整模块是 Win32 DLL 加载器（`LoadLibraryW`/`GetProcAddress`），非 Windows 无定义。
- `crates/engine/src/portrait_stage.rs:4340/5297`：变量只在 `#[cfg(windows)]` 分支赋值，非 Windows 无法推断类型。

处理（最小、平台隔离，Windows 代码零语义变化）：

1. `aurora_portrait::cuda` 增加 `#[cfg(not(windows))] mod no_driver`，提供与 Windows driver 同形的 `Nvidia`/`DeviceBuffer`/`PinnedBuffer`/`HostRegistration`/`Stream`/`BufferSession` 类型面；每个操作返回 `CudaError::Unavailable`。这不是第二套实现，而是平台 adapter：不分配、不启动内核。
2. `ort.rs` 把 Linux 的 `dlopen` 分支扩展到 macOS，并增加 `libonnxruntime*.dylib` 与 Homebrew 路径候选。
3. `raw.rs` 用 `LoadPath` 类型别名区分 Win32 宽字符与 POSIX 路径。
4. `jxl.rs` 在非 Windows 提供显式拒绝（`bail!("JPEG XL decoding is not available on this platform")`），不静默把文件当坏图。
5. `portrait_stage.rs` 给两处 `None` 加显式类型。
6. 修复 `cuda.rs` 的 Windows-only re-export 门控与 `cuda_v2fuseoffset` 的 `V2Warp` 导入门控。

结果：`cargo check -p aurora-portrait --lib`、`cargo check -p aurora-engine --lib`、`cargo check --workspace --all-targets --locked` 在 macOS arm64 上全部通过（仅有既有 warning）。
**这不等于 Windows 构建通过，也不等于 macOS 真机功能可用**：CUDA 路径在 macOS 只是被编译为“不可用并回退”；Metal 尚未实现。

## 3. 规模与热点（生产代码与测试未分开统计，行数本身不是缺陷）

全仓 `crates/engine/src`：223 个 `.rs`，约 177,339 行。最大的模块/文件：

| 模块/文件 | 行数 | 说明 |
|---|---:|---|
| `mock/`（16 文件） | 27,776 | 正式渲染 + 场景/GPU 阶段 + 缓存/采样混杂 |
| `aurora_raw/`（66 文件） | 16,524 | RAW 解码/重建/C1 颜色链 |
| `portrait_stage.rs` | 14,549 | 人像 producer、PF 路由、CPU/GPU、证据 |
| `file_engine.rs` | 7,137 | catalog/source/预览/保存/导出/多类锁 |
| `aimask/`（3 文件） | 4,614 | AI 掩码 + ORT |
| `recipe.rs` | 4,032 | 导出规则、缩放、编码、metadata |
| `portrait_face_input_stage.rs` | 3,993 | 人像输入生产 |
| `c1session_export.rs` | 3,632 | C1 Session 导出 |
| `raw.rs` | 3,222 | RAW 路由、LibRaw、provenance |
| `adjust.rs` | 2,321 | 持久化参数 + 运行时 producer 输入 |
| `engine.rs` | 1,064 | `ImageEngine` 门面（约 70 个领域方法） |
| `app/src/state/mod.rs` | 2,361 | 大范围 AppState + UI + 同步保存 |

平台/风险分布（`crates/engine/src` 文本统计，含测试）：

- `#[cfg(...)]` 约 528 处：`test` 226、`windows` 153、`not(windows)` 39、`all(windows, feature="z6iii-raw-export")` 34。
- `static` 项约 228 个；`Mutex`/`RwLock`/`Condvar` 提及约 239 次；`unsafe` 提及约 716 次。
- 这些数字只用于定位审计范围，不作为质量结论。

## 4. 模块依赖现状（实际导入统计）

按顶层模块 `crate::`/`super::` 导入计数，最显著的耦合：

```text
mock            -> mock/dust_lcc/geometry/detail/gpu_runtime/c1_banding/portrait_stage/icc/raw/engine/recipe/preview_cache ...
file_engine     -> aimask/raw/c1_cos/output/comask/dust_lcc/mock/c1import/recipe/store/tether ...
raw             <-> aurora_raw        （raw -> aurora_raw，aurora_raw -> raw）
mock            <-> raw / aurora_raw  （aurora_raw -> mock；mock -> raw、mock -> z6iii）
recipe          -> file_engine / mock （导出层反向依赖引擎与 mock）
face_parsing    <-> portrait_stage    （两者互相导入）
c1_model        <-> c1_spots
c1_styles       <-> mock
```

已确认的实际问题（不是行号推断）：

1. **正式渲染实现住在 `mock/`**：`FileEngine` 直接调用 `crate::mock`（8 处），`raw_cache`/`hdr`/`dust_lcc`/`c1_styles` 等也依赖 `mock`。`mock` 同时被 `aurora_raw` 依赖，形成 `aurora_raw ↔ mock` 循环。这是 P0 §5 的直接证据。
2. **导出层反向依赖**：`recipe -> file_engine`、`recipe -> mock`，使 `ExportService` 无法独立测试。
3. ~~`AdjustmentSet` 同时是持久化文档和运行时上下文~~：**已修复**。producer 字段已从 `AdjustmentSet` 删除，`document::RuntimeInputs` 唯一持有；`PartialEq`/`is_default()` 只反映持久化文档。
4. ~~批量保存静默失败~~：**已修复**。`persist_batch` 返回 `SaveReport`；索引失败单独报告；`save_adjustments_batch` 有失败时返回 Err；UI 显示 saved/failed/skipped。`SaveOrder` 用同一序列约束内存与磁盘，旧写不覆盖新写。
5. **原子写不是事务（未修）**：`store.rs` 的 `write_atomic` 仍用固定 `<file>.tmp` 名、`fs::write` + `rename`，没有 fsync/目录同步；同一目标并发写会争用同一个临时文件。单文件原子写有效，多文件批量仍非事务。
6. ~~未知顶层字段不 round-trip~~：**已修复**。`AdjustmentSet` 增加 `#[serde(flatten)] extensions: BTreeMap<String, Value>`，未知顶层键在 load→save 中原样保留；有 adjust 与 store 两级测试。
7. **许可证不一致**：根 `Cargo.toml` `[workspace.package] license = "MIT"`、`LICENSE` 文件是 MIT，而 `AGENTS.md` 要求 `Proprietary`。按任务约束**只报告，不修改**。
8. **`aurora_raw` 的 Z6III 路径整体被 `#[cfg(all(windows, feature="z6iii-raw-export"))]` 覆盖**（`aurora_raw/mod.rs:12`），包括可跨平台的解码/重建算法；平台限制没有下沉到 FFI/OpenCL 层。

## 5. 公共接口现状

- `ImageEngine` trait（`engine.rs`）约 70 个领域方法：library/catalog、import、source、preview、portrait、capture、export、print、job、C1、soft proof、merge 等。app 通过它调用，符合边界要求；但实现侧没有服务分层。
- `lib.rs` 大量 `pub use`：把 `portrait_stage`、`blemish_gainmap`、`aurora_portrait` 类型（含 vendor/CUDA 相关类型）直接暴露为 `aurora_engine::*`。公共面与内部实现未分离。
- 存档兼容：`.portrait` 调整文件 schema version 1（`store.rs:29`），读取时拒绝更高版本；未知顶层字段不保留（见 §4.6）。
- app 的 `contract.rs` / `evidence.rs` / `tools/portrait.rs` 与 `aurora-portrait::capability_registry`、`effect_router` 各自维护效果/能力状态，见 `A-REGISTRY-AUDIT.md`（由 A-registry lane 生成）。

## 6. 协作现状

- 本副本是独立 worktree；主树在 `<main worktree: 架构整理>`（分支 `架构整理`，SHA `7b6229d`），另有 `Mac-原生适配` worktree（分支同名，SHA `9bf908e`，`main..` 无提交差异）。
- `AGENTS.md` 指向的 `../../00_admin/OWNERSHIP-2026-11-14.md` 在本机不存在（`<workspace 00_admin 目录>` 不存在）。所有权按 `AGENTS.md` 文字和实际进程状态记录，见 `ARCHITECTURE-OWNERSHIP.md`。
- 未能读取另一台机器的进程/构建锁；不声称检查过远端进程。

## 7. 本基线后的第一批改动

见 `ARCHITECTURE-MIGRATION-MAP.md` 与 `ARCHITECTURE-VALIDATION.md`。原则：先修正确性与可验证性（平台构建、保存错误传播、文档/运行时分离），再新增公共契约与图语义，最后才迁移渲染实现；不先整体移动活跃文件。

## 8. 执行中补充确认的发现（A1–A3）

1. **非 Windows 测试此前根本无法链接**：`heif.rs` 的 `#[link(name = "kernel32")]` 无条件生效，`cargo check` 通过但 `cargo test` 链接失败。已把 `heif` 模块整体门控到 Windows，并在 `lib.rs` 提供同形非 Windows 拒绝 stub；macOS 现在可以原生链接并运行 engine 测试。其余 `kernel32/gdi32/ole32/shell32/winspool` 的 `#[link]` 均已在 `#[cfg(windows)]` 内，逐个核对过。
2. **`crates/portrait/examples/n35_reuse_acceptance.rs` 是 Windows CUDA 基准**，非 Windows 引用被 cfg 掉的 CUDA 类型。已把示例主体包进 `#[cfg(windows)] mod win_example`，非 Windows 只打印一行说明；Windows 行为不变。
3. **`cuda_radiance` 测试存在真实 UB**：把对齐 1 的 `include_bytes!` 直接转成 `&[f32]`，在 aarch64 上触发 `from_raw_parts` 对齐检查并 abort。已改为解码到对齐缓存；这是测试 bug，不是算法变化。
4. ~~导出全分辨率源只覆盖 RAW/EIP~~：**已修复**。非 RAW 且请求长边 > proxy 时，`start_export` 只准备廉价状态，导出 worker 在 scheduler 准入后解码原图全分辨率；解码失败/超内存上限则让 job 明确失败，不再放大 proxy。≤4096 走原预览源路径。RAW 无 native 解码时显式拒绝（不放大）。
5. **既有默认测试失败（非本轮引入）**：
   - `mock::geometry` 的 4 个 keystone/透视断言失败：**已定位并修复**。根因是 `a4b9d27` 用 C1 `CResampler::UpdateGeo` 精确基替换了旧 `0.35·tilt` 梯形，但 4 个测试仍编码旧形状；同时该 commit 的 `c = fLen>0?fLen:1` 注释被实现成未知焦距 `c=0`。现在未知焦距回退 `c=1`，测试按独立 f64 重建的 `UpdateGeo` 数值更新（`mock::geometry` 33 passed）。**遗留已修**：`keystone.rs` 的 `INSET_PER_UNIT=0.35` 已删除，Auto Keystone 改为对真实 `KeystoneMap` 数值反解；`keystone` 34 tests pass。
   - `tools::portrait::every_live_skin_and_teeth_effect_has_one_section`：实测是**测试自身漏算**——`SKIN_SECTIONS` 含 Eyes & Hair 3 项，而测试的 `live` 只串了 SKIN+TEETH；已把 EYES/HAIR 加入后 30==30。不是 UI 列表漂移。
   - 其余失败几乎都是缺 `crates/engine/testdata/` 外部资产、macOS `/var` 规范化、CUDA 不可用。详见 `ARCHITECTURE-VALIDATION.md` §3。
6. **`ImageEngine` 新增 `save_adjustments_batch_report`**（默认逐文档、FileEngine 覆盖为单次 catalog 提交）。app 现在能显示 saved/failed/superseded 与索引失败，而不再只看到一个 `Result`。
7. **`scripts/testdata-skips.txt` 与当前树的失败集不一致**：注释称 108 项，实测仍有 33 项未覆盖。该文件属另一 lane，未修改。
8. **平台门控改变像素而不只是可构建性**：非 Windows 下 C1 CPU 链跳过 CA、高光恢复、抗锯齿与 linear finish（`aurora_raw/mod.rs:126-127,159-166`），交互路径的 `c1_color` 整段不存在（`mock/process.rs:2018-2046`）。这意味着“macOS 能编译并回退”不等于“与 Windows 像素一致”，需要真机/对拍才能宣称等价。
9. ~~非 Windows 原生库名候选与本机 Homebrew 不匹配~~：**已修复**。libraw 候选含 `libraw.25.dylib` + Homebrew 前缀；HEIF writer/reader 共用候选并使用 libheif dlopen，macOS 读取已工作。JXL 仍为显式拒绝；ORT 本机未安装。
10. **同源码树存在两套 C1 算法**：`aurora_raw/*.rs` 与 `aurora_raw/z6iii/reconstruct/*.rs` 中 5 个文件字节相同，另有 5 个在浮点顺序/舍入/Stage3 offset 上不同；未经对拍不能合并（见 `A-BOUNDARY-AUDIT.md` §1.3）。
11. **`mock/retouch.rs` 未被 `mock/mod.rs` 挂载且无调用者**：是未编译的孤儿文件；不能作为“已实现”计入，也不应在本轮删除。
