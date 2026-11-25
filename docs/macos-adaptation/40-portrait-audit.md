# M3 — 人像 Metal 产品路径审计（Mac-F）

> 审计类型：**只读**。本文件不改源码、不跑 `cargo build/check/test`（构建锁由 Mac-A 持有），
> 结论全部来自源码阅读与本机只读命令。基线 commit：`9bf908e`（见 `00-baseline.md`）。
> 行号取自本次读取时的当前工作树；Mac-A 正在改 `portrait_stage.rs`/`cuda.rs` 的 cfg 收口，
> 后续行号可能平移，定位时以函数/符号名为主、行号为辅。
>
> 判定原则：**注册 ≠ 完成，adapter 存在 ≠ 已接线**。任何“能跑/已接入/已对拍”的结论都必须
> 有对应的产品调用点和真实执行证据；没有证据一律写“未验证”。

---

## 0. 结论摘要（先看这段）

1. **现有 Metal 代码没有被任何产品代码调用。**
   全仓库 `aurora_portrait::metal` 的引用只有 `crates/portrait/tests/metal_*.rs` 和
   `crates/portrait/examples/backend_probe.rs:34`。`crates/engine` / `crates/app` 零引用。
   `effect_router::Backend` 只有 `Cpu | Cuda | Auto`（`crates/portrait/src/effect_router.rs:33-38`），
   `PortraitBackend` 只有 `Cpu | Cuda | Skipped`（`crates/engine/src/portrait_stage.rs:439-444`）。
   **今天不存在“Metal 结果”，也不存在能把结果标成 Metal 的字段。**

2. **Metal 的现有真机测试全部是 `#[ignore]`，且本机未运行。**
   `metal_runtime.rs`、`metal_mesh_runtime.rs`（设备项）、`metal_smile.rs`、`metal_skin_assets.rs`
   的 GPU 测试都需要 `--ignored` 并显式 panic on non-macOS；没有 macOS GPU CI，也没有 dispatch
   trace / 输出图。因此全部为 **未验证**。

3. **有一个默认测试在 macOS 上会因缺资产失败。**
   `crates/portrait/tests/metal_asset.rs:109-131` 在 **macOS 上不是 ignore**，它读
   `<repo>/work/backend-r11/o2e/whiten_lut.f32`（`:9-12, :21-28`）。该目录不存在（本机只有
   `work/backend-r11/o1i`）。这是测试资产阻塞，不是源码错误，但会破坏 Mac-A 的
   `cargo test --workspace` 验收口径。

4. **人像产品路径的 P15 行存在跨后端不一致。**
   `PF90023/PF90024`：CPU 走 `skin_adjust::wrinkle_apply`（`portrait_stage.rs:6904-6906 → 7031`），
   Windows GPU 为 opt-in（`AURORA_PORTRAIT_P15_CUDA`，`:7713-7715, :7836-7841`）。
   `PF21024`：CPU 被 `skin_adjust::wrinkle_degree` 拒绝（`kernels/skin_adjust.rs:257-269` 不含 21024），
   但 Windows 的 historical-skin GPU 分支会走到 `WrinkleGpu::apply_historical_same_job`
   （`portrait_stage.rs:7865-7913`，`cuda_wrinkle.rs:281-308` 接受 21024）。
   **Metal 绝不能按 CUDA 行为接线 21024。**

5. **最高价值且风险最低的首个 Metal 产品效果是 PF90005/PF90007（皮肤 LUT）。**
   理由：UI 默认可达；输入只需一张 skin mask + 仓库内置真实 LUT（skin mask 仍依赖人脸解析
   模型，当前缺模型见 §8.2；但库级可用合成 mask 验证）；CPU oracle 就是产品
   `makeup::skin_adjust_lut`；Metal 的 `skin_lut_fixed` 已在真实 atlas 上逐字节对拍（库级）；
   而且它能把缺失的“Metal backend/dispatch 证据”管线一次打通。详见 §7。

---

## 1. 判定口径

### 1.1 状态等级（本审计使用）

| 等级 | 含义 |
|---|---|
| A 产品+真机 | 从 `ImageEngine`/`portrait_stage` 可达，且在 Mac 真机有 dispatch/像素证据 |
| B 产品+合成 | 产品调用点存在，但只用合成输入验证过，未在真机/真照片验证 |
| C 库级对拍 | host/kernel 存在且有 CPU 对拍，但产品不调用 |
| D 仅编译/ABI | 只有 ABI/注册/编译检查，没有像素对拍 |
| E 未实现 | 没有 host 或没有 kernel |

本仓库当前 **没有任何 A**；C 仅限 `#[ignore]` 的库级测试（且本机未跑）。

### 1.2 五个“现有 Metal 实现”的定位

| 效果 | Metal 入口 | 性质 |
|---|---|---|
| P15 facial smooth | `MetalContext::facial_smooth` `metal.rs:728`；`facial_smooth_rgba8` `:755`；`blur` `:761` | 复刻 `FacialSmoothProcessV2` 复合；返回未量化 f32 |
| O3B masked whiten + mask projection | `masked_whiten` `metal.rs:823`；`project_mask` `:875` | 来自 `o3b-…` 实验分支的 float-LUT/BGRA 路径，**不对应产品 PF** |
| Skin LUT（定点） | `skin_lut_fixed` `metal.rs:1253` | 复刻产品 `makeup::skin_adjust_lut`（RGB8 512² atlas、12-bit floor） |
| Mesh raster | `mesh_raster_serial` `:895`；`mesh_raster` `:941`；`mesh_raster_perspective` `:1035` | 仅 affine buffer 候选；**显式拒绝 vendor PF8025/PF8033** |
| Smile warp | `smile` `:1201` | 仅 `pf100007_smile`（lip×mouth 支撑），**没有 PF8033 mask-only 变体** |

---

## 2. 全量能力矩阵（效果 / PF）

列含义：
- **UI 可达**：`crates/app/src/tools/portrait.rs` 里存在真实滑块（不是 catalog 里的条目）。
- **真实 producer/输入**：产品代码能构造 same-job 输入，且所需资产存在。
- **CPU ref**：产品 CPU lane 调用了可对拍的 CPU kernel。
- **CUDA**：Windows adapter 存在**且**产品调用。
- **Metal**：Metal host/kernel 存在**且**有库级测试（≠ 产品调用）。
- **产品调用**：engine/app 是否调用 Metal。
- **证据**：fixture / 对拍等级。

> 所有 “Metal 产品调用” 都是 **否**；下表不再逐行重复“无人调用”的长句。

| # | 效果 / PF | UI 可达 | 真实 producer/输入 | CPU ref | CUDA | Metal | 产品调用 | fixture / 对拍证据 | 等级 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **PF90023 眼纹**（P15 composite） | 是 `portrait.rs:52` | 是*（`attach_wrinkle_from_extra` `portrait_face_input_stage.rs:1935-2124`，P15 rows `:2081-2122`） | 是（`skin_adjust::wrinkle_apply` `kernels/skin_adjust.rs:273`） | 是*（opt-in `portrait_stage.rs:7836-7841`，或 historical-skin GPU `:7865-7913`） | 是（`metal.rs:728`） | 否 | `metal_runtime.rs:58` f32 对 `composite_pixel`，容差 1e-5；**本机未跑** | C |
| 2 | **PF90024 抬头纹**（P15 composite） | 是 `portrait.rs:53` | 同 1 | 是 | 是*（同 1） | 是 | 否 | 同 1；注意 UI→native 是 `ui` 而非 `ui/100`（`skin_adjust.rs:264-268`） | C |
| 3 | **PF21024 眉间纹**（P15 composite） | 是 `portrait.rs:54` | 是（producer 会写 Glabella） | **否**（`wrinkle_degree` 拒绝 21024 `skin_adjust.rs:258`） | **不一致**（historical GPU 接受，`cuda_wrinkle.rs:295`） | host 的 `SUPPORTED_PF` 含 21024（`metal.rs:36`） | 否 | 无 CPU oracle；`resolve_p15` 明确 `MissingInput`（`cuda_wrinkle.rs:164-167`） | E（产品） |
| 4 | **PF90005 美白**（SkinAdjust LUT） | 是 `portrait.rs:38` | 是（skin mask + `include_bytes!` `assets/portrait/skin-adjust/whiten.png`，`portrait_stage.rs:9212-9224`） | 是（`makeup::skin_adjust_lut` `kernels/makeup.rs:411`） | 是（`CachedSkinLut` `portrait_stage.rs:8284-8368`） | 是（`metal.rs:1253`） | 否 | **真实 atlas 逐字节**：`metal_skin_assets.rs:218-239`（CPU fixture）+ `:256-291`（GPU ignore）；fixture atlas 与产品 PNG 解码后 0 字节差（见 §8.1） | C |
| 5 | **PF90007 红润**（SkinAdjust LUT） | 是 `portrait.rs:39` | 是（`ruddy_new.png` `:9226-9238`） | 是 | 是 | 是 | 否 | 同 4（ruddy atlas） | C |
| 6 | **PF21015 肤色色相**（同 kernel 家族） | 是 `portrait.rs:36` | 是（TintGreen/TintMagenta PNG `portrait_stage.rs:9184-9210`） | 是（`skin_adjust_lut`，signed mix `portrait_stage.rs:9282`） | 是（`cuda_lut::tint`） | **primitive 可表达，但无 PF 绑定/验证** | 否 | 无 Metal tint 专用测试 | D |
| 7 | **PF30096 色温** | 是 `portrait.rs:37` | 是（cold/warm PNG `portrait_stage.rs:9240-9266`） | 是（`face_detail::skin_temperature`） | 是（`cuda_lut::temperature`） | 否（无 MSL） | 否 | — | E |
| 8 | **O3B masked whiten / coverage-max projection** | 否（无 PF、无滑块） | **否**（实验分支；float RGBA atlas，非产品资产） | 名义上有 `skin_color::masked_lut_blend` `kernels/skin_color.rs:886`，但**全仓库无调用/无测试/无 PF 行** | 否 | 是（`metal.rs:823, 875`） | 否 | `metal_runtime.rs:186-238`、`:242-286`（合成）；`metal_asset.rs:135` 需缺失的 float LUT | C（不可绑定） |
| 9 | **PF100007 张嘴笑** | **否**（app 无 `pc:100007`） | 是（`ProducerInputs.smile`，`portrait_stage.rs:542`；需要 780929437 模型输出） | 是（`smile_warp::pf100007_apply`，`apply_smile_cpu:7369-7398`） | 是（`same_job` `:8602-8638`） | 是（`metal.rs:1201`） | 否 | `metal_smile.rs:243` GPU ignore；golden pack 是 synthetic（manifest `oracle=pf100007_apply`） | C |
| 10 | **PF8033 微笑嘴角** | 是 `portrait.rs:101` | 是（`FaceGeometry::Smile` + host split receipt） | 是（`pf8033_dense_warp` `portrait_face_stage.rs:1089-1103`） | 是（`cuda_smile::apply_pf8033` `:1107-1142`） | **否**（`smile_warp.metal` 只有 PF100007 变体） | 否 | — | E（Metal） |
| 11 | **PF202 发际线**（mesh fragment） | 是 `portrait.rs:99` | 是（`FaceGeometry::HairLine`） | 是（`rasterize_with_fragment`，`portrait_face_stage.rs:589-598`） | 是（`MeshRaster`，`:560-608`） | **不匹配**（Metal 只有 `raster::rasterize`，无 fragment 状态） | 否 | `metal_mesh_runtime.rs:228` 对 `cpu_golden`（`raster::rasterize`），非 fragment | D |
| 12 | **眼部/面部网格行**（PF8013/14/15/16/17/18/19/20/21/22/23/24/27/28/29/30、30073/79） | 是 `portrait.rs:102-113` | 是（`FaceGeometry::Eye`） | 是（`rasterize_with_fragment` `portrait_face_stage.rs:398-413`） | 是（`apply_eye` `:416-485`） | **不匹配**（同 11） | 否 | 同 11（合成顶点/EBO） | D |
| 13 | **PF242/243/247/248 等 V2 造型** | 是 `portrait.rs:84-89` | 是（`FaceGeometry::V2Positions`） | 是（`cuda_v2warp::apply_cpu` `:525-530`） | 是（`cuda_v2warp::V2Warp` `:487-524`） | 否（无 MSL） | 否 | — | E |
| 14 | **PF232 颅顶** | 是 `portrait.rs:100` | 是（`FaceGeometry::Skull`） | 是（`render_skull`） | 否（径向 warp） | 否 | 否 | — | E |
| 15 | **PF8050/PF8051 六阶段几何** | 否（无独立滑块；body/face 内部） | 是（`cuda_pf8050::Inputs`） | 是（`apply_cpu` `portrait_stage.rs:5568`） | 是（`Pf8050Stage` `:5544`） | 否 | 否 | — | E |
| 16 | **PF407（router 的 UvMap）** | 407 实际是 legLengthen（`portrait_map.rs:211`） | 否（router 的 `field/channel/catmull` producer 缺失） | 是（`uvmap_color::apply`，但非此 PF） | router 有，产品不绑定 | 否 | 否 | `portrait_stage.rs:8225-8230` 明确 BLOCKED | **误绑定风险** |
| 17 | **PF30009（router 的 TextureBlend）** | 30009 = hairColorUnify（`portrait_map.rs:369`） | 否（HSL LUT / hair mask producer 未接） | 是（`texture_mask_blend` primitive） | router 有，产品不绑定 | 否 | 否 | `portrait_stage.rs:8246-8250` 明确 BLOCKED | **误绑定风险** |
| 18 | **PF8033（router 的 MeshAttribute）** | 同 10 | 否（缺 attribute pair producer） | 是（`mesh_attribute::rasterize`） | router 有，产品不绑定 | 否（vendor contract 拒绝） | 否 | `portrait_stage.rs:8251-8255`；`metal_mesh.rs:352-365` | **误绑定风险** |
| 19 | 其余 `effect_router` adapter（Teeth/Gloss/Beard/Moisture/Makeup/TeethBlend/EyeBag 等） | 多数有 | 各自不同 | 有 | 有（Windows） | **无 Metal kernel** | 否 | — | E（Metal） |

\* P15 的“真实 producer”依赖人脸模型（见 §8.2）；在缺模型时 CPU 与 Metal 都会 skip，这不是
Metal 独有的阻塞。

### 2.1 同一效果内部的“代码存在”与“已接线”差异

| 事实 | 证据 |
|---|---|
| `effect_router` 自称“产品调度在这里”，但源码注释明确写“not called from product code yet” | `effect_router.rs:1-15` |
| Windows 产品确实经 `cuda_router` 调 router | `portrait_stage.rs:8192-8201, 5377-5381` |
| `device::select` 已有 `Backend::Metal`，但全仓库无生产调用 | `device.rs:7-12, 60-90`；仅 `device.rs` 自身测试引用 |
| Metal 的 `available()`/`framework_available()` 只是设备/框架探测 | `metal.rs:670-697` |
| `SUPPORTED_PF` 只有 P15 三行 | `metal.rs:36` |

---

## 3. 五个效果的输入、ABI 与 CPU oracle（逐项）

### 3.1 P15 facial smooth（PF90023 / PF90024；21024 未解）

- **输入**（`metal.rs:329-345`）：`source`、`base`、`residual`（f32 RGBA，`stride` 按像素）、
  `mask_rgba`（f32 RGBA，只读 `.x`/`.z`）、`skin_mask`（f32 R）、10 个产品 uniform、
  `pf`、`pf_value`、`ticket`（必须非 0）。
- **ABI**：`P15Uniforms` 88B（`metal.rs:105-130`）；MSL 同名结构 `P15FacialSmooth.metal:11-37`。
- **gating**：`SUPPORTED_PF = [90023,90024,21024]`（`metal.rs:36`）；
  `pf_value` 必须 `> 0.001` 且有限（`metal.rs:1500-1507`）。
- **数学**：`p15_composite_channel`（`P15FacialSmooth.metal:187-214`）与
  `facial_smooth::composite_unchecked`（`kernels/facial_smooth.rs:147-181`）、
  `skin_adjust::wrinkle_apply`（`kernels/skin_adjust.rs:304-341`）一致；alpha 从源逐位恢复
  （`metal.rs:1510-1514`）。
- **CPU oracle**：
  - 浮点：`kernels::facial_smooth::composite_pixel`（`facial_smooth.rs:106`），Metal 测试用它
    （`metal_runtime.rs:120-144`）。
  - 产品字节：`skin_adjust::wrinkle_apply`（`skin_adjust.rs:273`）。
- **关键差异**：`facial_smooth_rgba8` 用 half-away 取整（`metal.rs:1458-1463`、测试 `:2533-2539`），
  而产品 P15 GPU/CPU 用 `StoreMode::TruncU8`（`skin_adjust.rs:128-150`；`portrait_stage.rs:7862, 8712`）。
  **接产品时必须用 TruncU8，不能直接用 `facial_smooth_rgba8`。**
- **ROI**：产品只写 ROI 内像素，ROI 外和 alpha 从源复制（`cuda_wrinkle.rs:264-273`）。
  Metal kernel 没有全局 ROI 偏移参数，host 必须自行 crop/uncrop。

### 3.2 O3B masked whiten + coverage-max projection（无产品 PF）

- **输入**（`metal.rs:434-451`）：BGRA8 源、mask 矩形（`mask_x/y/width/height`）、
  512×512 RGBA f32 atlas、alpha、flags、ticket。
- **ABI**：`WhitenParams` 48B（`metal.rs:133-150`）、`ProjectionParams` 80B（`:153-178`）。
- **语义**：mask rect 局部坐标；`weight = mask/255`、`alpha = weight*p.alpha`，`alpha > 0.001`
  才写；输出 BGRA，alpha 从源恢复；量化是 round-half-even（`o3b_masked_lut_whiten.metal:185-249`）。
  coverage-max projection 要求整数 ROI 放置（`o3b_masked_lut_whiten.metal:52-87`）。
- **CPU oracle**：`kernels::skin_color::masked_lut_blend`（`kernels/skin_color.rs:886-915`）数学等价
  （`clamp_byte` 用 `round_ties_even`，`:141-149`），但**全仓库无调用方、无测试、无 PF 行**。
- **结论**：这是实验分支产物，**不能按名字绑到 PF90005**：产品 90005 是 RGBA8/RGB8 定点 LUT，
  这里却是 BGRA/float4 atlas，且量化/通道序不同。

### 3.3 Skin LUT fixed（PF90005 / PF90007；21015 同族）

- **输入**（`metal.rs:1253-1259`）：RGBA8 `source`、u8 `mask_r`、`lut_rgb`（512×512×3）、`mix`。
- **ABI**：`SkinLutParams` 32B（`metal.rs:655-668`）。
- **数学**：8×8 tile 64×64 的 3D LUT，整数双线性 + 12-bit floor store；`weight=(mask/255)*mix`，
  `weight > 0.001` 才写，alpha 复制（`skin_lut_fixed.metal:97-146`）。
- **CPU oracle**：`makeup::skin_adjust_lut`（`kernels/makeup.rs:411-449`）；这就是产品
  `apply_skin_whiten/ruddy` 调用的函数（`portrait_stage.rs:9296, 9327`）。
- **证据**：`metal_skin_assets.rs:218-239` 在 Windows/CI 上校验 fixture hash 且 CPU oracle 命中
  `expected.rgba8`；`:256-291` 是 macOS-only GPU 逐字节对拍（ignore）。
- **fixture 真实性**：本机用 `sips` 把 `assets/portrait/skin-adjust/{whiten,ruddy_new}.png`
  转 24-bit BMP 后与 `tests/fixtures/metal_skin/atlases/{whiten,ruddy}.rgb8` 逐字节比较，
  **diff_bytes = 0**（512×512×3）。因此这两个 fixture 就是产品 LUT 材质，不是随机图。

### 3.4 Mesh raster

- **输入**（`metal_mesh.rs:378-389`）：`dest: &[[f32;2]]`、`source_uv`、`ebo: &[u16]`、
  `source: Image`(RGBA8)、`Rules`（winding/cull/edge/overlap）、`Sampler`（ClampToEdge/Repeat）。
- **CPU oracle**：`MeshRasterInput::cpu_golden` → `raster::rasterize`（`metal_mesh.rs:453-466`）。
- **与产品不一致的关键点**：产品网格走 `raster::rasterize_with_fragment`，额外带
  `Edge/Defines/Clamp/Tolerance::RESHAPE`（`portrait_face_stage.rs:398-413, 560-608`）。
  Metal 输入 ABI **没有**这些 fragment 状态，`cpu_golden` 也不是 fragment 版本。
  → **不能宣称对产品 eye/PF202 路径等价**。
- **vendor contract**：`MeshContract::VendorForwardMesh(Pf8025|Pf8033)` 在 dispatch 前返回
  `UnsupportedMeshContract`（`metal_mesh.rs:352-365`；host `metal.rs:896-897, 942-945, 1039-1040`）。
  capability 明确 `has_mtl_texture=false / has_mtl_sampler_state=false / has_fbo_render_target=false`
  （`metal_mesh.rs:303-332`）。
- **覆盖语义**：uncovered 像素保留源字节；`Overlap::Last` 用 0 预填 selected
  （`metal.rs:650-653`；测试 `:2521-2530`）。

### 3.5 Smile warp

- **输入**（`metal_smile.rs:29-40`）：源字节（通道 0/1/2 + alpha 3，通道序无关）、宽高、
  **post-gate** degree、lip mask、mouth mask、model dx/dy、ROI。
- **ABI**：`SmileParams` 32B（`metal_smile.rs:13-27`）。
- **数学**：`support=(lip/255)*(mouth/255)`，`weight=degree*support`，`support<=0.001` 跳过；
  字节空间双线性、截断存储、alpha 复制、只写 ROI（`smile_warp.metal:37-89`）。
- **CPU oracle**：`smile_warp::pf100007_apply`（`kernels/smile_warp.rs`），产品经
  `apply_smile_cpu`（`portrait_stage.rs:7369-7398`）。
- **不覆盖 PF8033**：PF8033 是 mask-only dense warp（`pf8033_dense_warp`），支撑项不是 lip×mouth；
  MSL 里只有 `pf100007_smile`（`metal.rs:646`）。**不得把 PF100007 kernel 绑到 8033。**
- **产品前置**：`pf100007_gate(degree*100, material)`（`portrait_stage.rs:7375-7376`）必须在
  host 侧先算，再把它的 `(active, degree)` 传给 Metal。

---

## 4. Dispatches / backend_used / StrictGpu / MissingInput / Unsupported

### 4.1 现状

| 概念 | 现有实现 | 证据 |
|---|---|---|
| `Dispatches` | **CUDA 专用**结构（kernel/grid/block/ptx_bytes/ptx_isa/device/compute_cap/driver_version） | `crates/portrait/src/cuda.rs:167-180` |
| `backend_used` | 只有 `Cpu/Cuda/Skipped` | `portrait_stage.rs:439-444` |
| 空 dispatch 规则 | CUDA 返回空 dispatch 不算证据，CPU 接手并记录原因 | `portrait_stage.rs:5385-5413` |
| `CudaError` | `Unavailable / Driver / Refused(Refusal)` | `cuda.rs:235-244` |
| Metal 拒绝 | `MissingInput / ShapeMismatch / ValueOutOfRange / UnsupportedPf / UnsupportedMeshContract / Inactive / MissingSameJobIdentity / HostUnavailable / CompileFailed / AllocationFailed / CommandFailed` | `metal.rs:311-325` |
| StrictGpu / PreferGpu / CpuOnly / Auto | **未实现**；只有零散 env：`AURORA_PORTRAIT_RENDER_BACKEND=cpu`、`AURORA_PORTRAIT_FORCE_CPU`、`AURORA_PORTRAIT_P15_CUDA` | `portrait_stage.rs:7661-7677, 7713-7715` |
| 用户可见 backend 证据 | `PortraitRunReport` 回传到 `Rendered.portrait_report`，但 app 未展示 | `mock/process.rs:1812, 3314-3330`；`app/src` 无 `portrait_report` 引用 |

### 4.2 Metal 结果被标成 Metal 的必要条件（缺一不可）

1. **枚举与字段**：新增 `PortraitBackend::Metal`（或等价），并同步所有 exhaustive match
   （例：`crates/engine/src/portrait_hlf_wrinkle_stage_tests.rs:123-146`）。现在是 `Cpu|Cuda|Skipped`。
2. **独立证据类型**：`Dispatch` 的 PTX/ISA/compute_cap 字段对 Metal 无意义。必须新增
   Metal dispatch receipt（至少：pipeline 名、grid/threadgroup、device 名、command buffer
   完成状态、读回字节数），或把 `PortraitEffectRun.dispatches` 改为后端枚举。
   由于 `PortraitEffectRun` 是 `lib.rs:161-162` 的公开导出，这是跨 lane 接口变更，需主控协调。
3. **真实完成**：`dispatch_pipeline` 在 `commit + waitUntilCompleted` 后检查
   `status == 4 && error == null`（`metal.rs:1434-1441`）；只有通过且完成读回的结果才能算执行。
   `available()` / `MetalContext::new()` / pipeline 编译成功都**不是**执行证据。
4. **非空 dispatch**：沿用 `portrait_stage.rs:5385` 的规则：dispatch 列表为空 → 不得标 GPU。
5. **不把 CPU 结果标成 Metal**：`StrictGpu` 语义下必须返回 typed error；今天的代码没有该模式，
   不得用 `Backend::Auto` 的 CPU fallback 冒充。
6. **错误必须落盘**：Metal 拒绝要写进 `fallback_reason`（对齐 `portrait_stage.rs:5406-5412`），
   并区分 `MissingInput`（缺同 job 数据）与 `Unsupported`（该效果无 Metal kernel）。
7. **same-job identity**：P15 的 `ticket == 0` 必须 `MissingSameJobIdentity`（`metal.rs:383-385`），
   不能拿测试常量或当前时间伪造。产品侧用 `inputs.job_id()`；`MasksOnly` 返回 `None`
   （`portrait_stage.rs:3708-3716`），此时应 skip 而不是伪造。
8. **线程模型**：`MetalContext` 不 `Send`，由 thread-local 复用（`metal.rs:599-602, 1482-1498`）。
   产品循环在 worker 线程调用即可，但不得跨线程共享，也不得为消除编译错误加
   `unsafe impl Send/Sync`（现有 `unsafe impl Send/Sync for Lib` 只针对 dlopen 句柄/函数指针，
   `metal.rs:1716-1718`）。

---

## 5. Metal 分支的精确接入点（file:line）

> Mac-A 只会在 `portrait_stage.rs:4340`、`:5297` 补类型注解；下面的方案**不改**这两处，
> 也不改任何 `#[cfg(windows)]` CUDA 分支。

### 5.1 主调度循环（所有皮肤/人像效果共用）

- 入口：`apply_portrait_effects_with_report_and_inputs_policy` `portrait_stage.rs:4678`。
- 每效果循环：`:4770`（`while let Some((pf, degree)) = effects.next()`）。
- 现有 CUDA 选择：`:5377-5381`（`#[cfg(windows)] let cuda_result = try_cuda_effect(...).or_else(try_cuda_same_job_effect)`）。
- 现有 CUDA 证据门：`:5383-5414`。
- CPU 回退：`:5415-5425`（`apply_cpu_effect`）。
- **Metal 插入点**：在 `:5414` 之后、`:5415` 之前，新增独立的 `#[cfg(target_os = "macos")]`
  block（独立变量名，避免跨 cfg 类型推断问题），调用新的
  `try_metal_effect(&current, pf, degree, inputs, skin_mask.as_deref())`；
  成功且 dispatch 非空 → 写 `current`、设 `run.backend`、`run.executed = true`、`continue`；
  否则把拒绝写进 `fallback_reason` 后落到现有 CPU 路径。
- **不要**把 Metal 塞进 `try_cuda_effect` / `try_cuda_same_job_effect`（它们整体 `#[cfg(windows)]`，
  且 `CudaError` 类型带 CUDA 语义）。

### 5.2 Skin LUT（PF90005/90007）

- 产品 CPU：`:6924-6925` → `apply_skin_whiten` `:9285-9297` / `apply_skin_ruddy` `:9316-9328`。
- 产品 CUDA：`:9099` → `cuda_router::skin_effect` `:8378-8440` → `CachedSkinLut` `:8284-8368`。
- LUT 资产：`whiten_lut()` `:9212-9224`、`ruddy_lut()` `:9226-9238`（`include_bytes!` 真实 PNG）。
- Mask：`:4764` `inputs.masks.mask_u8(MaskKind::Skin, …)`；`mask_u8` 输出与当前帧同尺寸
  （`portrait_stage.rs:260-272`），可直接作为 Metal 的 `mask_r`。
- **Metal 调用**：`execute_skin_lut_fixed(current, mask, lut, degree.clamp(0.0,1.0))`
  （`metal.rs:1473-1480`），由 thread-local context 复用 `skin_lut_fixed` pipeline。

### 5.3 P15（PF90023/90024）

- 产品 CPU：`:6904-6906` → `apply_wrinkle_cpu` `:7031-7075`。
- 产品输入解析：`inputs.resolve_wrinkle_inputs(current)`（与 `:7837` 相同）。
- 产品 CUDA（两条路）：
  - opt-in P15：`:7836-7841` → `cuda_router::p15` `:8718-8736` → `P15Wrinkle` `:8678-8715`。
  - historical-skin GPU：`:7865-7913`（21024/90023/90024）。
- **Metal 适配步骤**（在 `try_metal_effect` 内）：
  1. `p15_handles(pf)`（`:7707-7709`）只允许 90023/90024；**显式拒绝 21024**。
  2. `cuda_wrinkle::resolve_p15` 得到 `(degree, roi, uniforms)`（`cuda_wrinkle.rs:163-188`）。
  3. 按 `roi` crop `source/base/residual/mask_rgba/skin_mask`；u8→f32（`/255`）；mask 取
     `.r=.z`（对应 Metal 的 `.x/.z`）。
  4. 组 `SmoothInput { pf, pf_value: resolved.degree, ticket: inputs.job_id().ok_or(MissingSameJobIdentity)?, uniforms, … }`。
  5. `execute_facial_smooth`（f32），把结果按 **`StoreMode::TruncU8`** 写回 ROI；ROI 外和
     alpha 从源复制。
- **P15 的 `blur` 不是产品所需**：产品的 `base/residual` 由 HLF 生产
  （`portrait_face_input_stage.rs:1981-2000`），Metal 的 surface/gauss/box pass 不参与产品。

### 5.4 Smile（PF100007）

- 产品 CPU：`:7370-7398`；CUDA：`same_job` `:8602-8638`。
- **Metal 插入**：同一 `try_metal_effect`；先 `pf100007_gate(degree*100, material)`，
  再 `MetalContext::smile(SmileInput { .. })`（`metal.rs:1201-1232`）。
- **PF8033 不在此处**：8033 在 `portrait_face_stage.rs:1067-1148`，且 Metal 无 mask-only kernel。

### 5.5 Mesh

- `portrait_face_stage.rs`：`apply_eye` `:416-485`、`render_mesh` `:533-624`、
  PF8033 `:1067-1148`。
- 这些函数的 `#[cfg(not(windows))]` CPU arm（`:479-484, :609-623, :1143-1147`）就是 macOS 现在的
  唯一路径。
- **不要在该处直接塞 `metal_mesh::MeshRasterInput`**：产品需要 `rasterize_with_fragment` 的
  Edge/Defines/Clamp/Tolerance；现有 Metal ABI 无这些字段，硬接会改变像素语义。

### 5.6 effect_router（lib 级，非必须）

`effect_router::Backend`（`effect_router.rs:33-38`）、`BackendUsed`（`:41-45`）、
`CudaEffect`（`:81-94`，签名返回 `CudaError`/`Dispatch`）都没有 Metal 概念。
它目前不是产品调用点（`effect_router.rs:1-15`），因此 **首个 Metal 产品接入不要绕经 router**；
否则需要同时改 trait、证据类型和 16 个 adapter。

### 5.7 不可改动清单

| 不可变项 | 证据 |
|---|---|
| P15 复合数学、`fastMathEnabled:NO` + `#pragma clang fp contract(off)` | `metal.rs:1640-1642`；`P15FacialSmooth.metal:5-6` |
| P15 alpha 逐位恢复 | `metal.rs:1510-1514, 748, 869` |
| P15 ROI 只写内部、外部与 alpha 复制 | `cuda_wrinkle.rs:264-273`；`StoreMode::TruncU8` `skin_adjust.rs:128-150` |
| P15 gate `> 0.001`、`ticket != 0` | `metal.rs:383-385, 1500-1507` |
| Skin LUT 12-bit floor、gate、alpha 复制 | `skin_lut_fixed.metal:97-146`；`makeup.rs:411-449` |
| Skin LUT 资产必须是解码后的真实 RGB8 PNG | `portrait_stage.rs:9170-9182, 9212-9238` |
| Mesh rules/overlap/uncovered prefill、vendor 拒绝 | `metal_mesh.rs:303-365, 403-466`；`metal.rs:650-653` |
| Smile 支撑 gate、双线性、截断、alpha 复制、ROI | `smile_warp.metal:53-88` |
| Renderer receipt 校验（不能因换后端而跳过） | `portrait_stage.rs:4692-4704`；`PortraitFrameReceipt` `:1395-1413` |
| PF8033 host-split receipt（job/tensor/hash 绑定） | `portrait_face_stage.rs:1076-1086` |
| `MetalContext` 线程绑定、不共享 | `metal.rs:599-602, 1482-1498` |

---

## 6. PF 编号风险（名称相近 ≠ 契约相同）

| PF | 当前权威定义 | 冲突/相似项 | 风险与要求 |
|---|---|---|---|
| **407** | `legLengthen`（`portrait_map.rs:211`）；`kernels::leg_refine::META` `leg_refine.rs:11-16` | `effect_router::UvMap` 的 `PFS=[407]`、`uvmap_type143`（`effect_router.rs:286-290`） | 产品已 BLOCKED：router 是通用 UV 颜色采样，缺 `field_xyzw/channel/catmull` producer（`portrait_stage.rs:8225-8230`）。**不得把 Metal UV kernel 绑到 407。** |
| **90023** | `removeEyeLines`；P15 三行之一 | 与 90024/21024 共用 composite | `native = ui/100`（`skin_adjust.rs:264-268`）；Metal `pf_value` 必须用 resolved degree |
| **90024** | `removeForeheadLines`；P15 | 同上 | `native = ui`（未证 `/100`，`skin_adjust.rs:253-255`）；不要套用 90023 的缩放 |
| **21024** | `removeGlabellaLines`；P15 | 同 composite | `mask.r/mask.b` 贡献律未恢复：`resolve_p15` 返回 `MissingInput`（`cuda_wrinkle.rs:164-167`），CPU `wrinkle_degree` 直接 `UnknownEffect`（`skin_adjust.rs:257-259`）。**Metal `SUPPORTED_PF` 含 21024 只是库级 gate，不得据此接入产品。** |
| **30009** | `hairColorUnify`（`portrait_map.rs:369`；`hair_body::META` `hair_body.rs:31-37`；`hair_color.rs:39-48`） | `effect_router::TextureBlend` `PFS=[30009]`（`effect_router.rs:816-819`） | 产品 BLOCKED（`:8246-8250`）。`NormalBlend::PF30009` 是 vendor 混合参数常量（`texture_mask_blend.rs:12-33`），不等于“PF30009 效果已完成”。**不要把 Metal texture blend 直接绑 30009。** |
| **8033** | `smileMouthCorner`；`smile_warp::META` `:649-657`；输入含 mean/reshape mesh、projection、dense dx/dy/mask | 两个 router adapter 抢 8033：`smile_pf8033`（`:934-937`）与 `mesh_attribute`（`:1029-1032`） | 产品只绑定 smile 的 dense warp；`mesh_attribute` BLOCKED（`:8251-8255`）。Metal 明确拒绝 vendor forward mesh（`metal_mesh.rs:352-365`），且没有 mask-only smile kernel。**8033 只能继续走 CPU/CUDA。** |
| 90005/90007 | `skinWhitening` / `skinRosy`（`makeup::META` `:531-532`） | O3B `masked_whiten` 名字也像“美白” | 产品 90005/90007 是 fixed RGB8 LUT；O3B 是 float RGBA + BGRA + half-even。**不得按名字互绑。** |

补充：`crates/portrait/tests/effect_router.rs:115-128` 把这些 router PF 断言成“expected”，
这只锁定 router 的库级常量，**不代表产品 PF 语义**。改 Metal 时必须以
`portrait_stage.rs` / `capability_registry` 为准。

另：UI 侧的 `docs/frontend-contract/portrait-capability-registry-20261124.tsv` 与
`crates/portrait/assets/portrait/capability-registry-20261124.tsv` **不是逐字节相同**
（90021/90087 的 key 文本不同）。两处都可当“注册表”，但不能同时当作唯一真相；本审计以
crate 内嵌那份 + `kernels::META` 为准。

---

## 7. 建议的 Metal 迁移顺序（每个效果都要有 CPU parity 测试）

> 顺序原则：先选 **UI 可达 + producer 简单 + CPU oracle 精确 + 真实资产已存在** 的效果，
> 先把“Metal 证据/回退管线”打通，再上 ROI/模型复杂的效果。**不要按“kernel 文件已经写好”排序。**

### P0-1 皮肤 LUT：PF90005 / PF90007（首个产品接入）

**为什么是它**：UI 默认滑块（`portrait.rs:38-39`）；输入只有 skin mask + 编译进二进制的真实
LUT（mask 由人脸解析产生；库级对拍可用合成 mask，真实照片需 §8.2 的模型）；CPU oracle 就是
产品函数；Metal 定点 kernel 在真实 atlas 上已逐字节对拍（库级）；实现是单 pass、无 ROI、
不消费模型 tensor。

**必须新增的产品证据管线**（一次建设，后续效果复用）：
1. `PortraitBackend::Metal` + Metal dispatch receipt + `run.backend` 规则。
2. `try_metal_effect` 的 `90005|90007` 分支。
3. 失败原因写入 `fallback_reason`；`StrictGpu`（若同批引入）失败即报错。

**CPU parity 测试（按顺序做）**：
1. **库级字节对拍**（macOS 真机）：`execute_skin_lut_fixed` vs `makeup::skin_adjust_lut`，
   用 `tests/fixtures/metal_skin` 的真实 atlas；覆盖 mask ∈ {0,1,64,128,200,255}、
   mix ∈ {0,0.25,0.5,1.0}、尺寸 {1×1, 17×13, 64×37, 3840×2160 裁剪}；断言 **所有 RGBA 字节相等**
   （alpha 必须等于源）。这是把现有 `metal_skin_assets.rs:256` 从 ignore 变成可记录的真机证据。
2. **引擎级产品对拍**：用合成 `PortraitMasks{skin: Some(...)}` + `pc:90005`/`pc:90007`，
   在 `apply_portrait_effects_with_job` 上分别跑 CPU 与 Metal，断言整帧字节相等、report 的
   backend 分别为 Cpu/Metal、dispatches 非空。
3. 0 / 中间 / 最大强度；mask 全 0（必须 identity）；重复拖动同参数（确定性）；
   CPU→Metal→CPU 切换无脏帧。
4. 真实照片验收：需要 §8.2 的模型 + 合法人像样本；在此之前标“未验证”。

### P0-2 P15：PF90023 / PF90024（第二个）

**为什么第二**：感知收益更高，但需要 ROI、10 个 uniform、face landmarks/parsing/HLF producer，
且库级只证到 1e-5（不等于 TruncU8 下的字节相等）。

**CPU parity 测试**：
1. 浮点：`execute_facial_smooth` vs `kernels::facial_smooth::composite_pixel`
   （复用 `metal_runtime.rs:58`），覆盖非方形 + 带 padding stride。
2. **产品字节**：把 Metal f32 结果按 `StoreMode::TruncU8` 写回 ROI，和
   `skin_adjust::wrinkle_apply` 逐字节比较；覆盖 ROI 在左上/右下/1×1、ROI 外像素、alpha。
   若出现 1 LSB 差异，必须定位到具体运算顺序并修正，不能放宽容差。
3. gating：`pf_value = ACTIVE_GATE` 必须 `Inactive` 不 dispatch；`ticket = 0` 必须
   `MissingSameJobIdentity`；`pf=21024` 必须拒绝（产品级）。
4. 回归：先证明“Metal 关闭时 CPU 字节完全不变”（默认路径不受影响）。

### P1-3 Smile：PF100007（第三个，但先补 UI）

现状没有 `pc:100007` 滑块（`portrait.rs` 全文无 100007），只能靠 sidecar/导入触发。
若产品确认需要，再先加控件，再接 Metal。**CPU parity**：`metal_smile.rs` 的 4 个 golden case
（已经是 `pf100007_apply` 生成）+ 0/中间/最大 degree + ROI 边界 + `support<=0.001` 不变；
注意 degree 必须先过 `pf100007_gate`。

### P2-4 Mesh：产品 fragment 网格（需要新 MSL，不是“接一下”）

现有 Metal 只能做 affine buffer 候选，且 `cpu_golden` 是 `raster::rasterize`，产品是
`rasterize_with_fragment`。**在补齐 Edge/Defines/Clamp/Tolerance 的 MSL + ABI + CPU 对拍之前，
不要接 eye/PF202/PF8033。** 对拍目标必须是 `raster::rasterize_with_fragment`，不是
`raster::rasterize`。PF8033 的 vendor contract 继续保持 typed refusal。

### P3-5 O3B float whiten / projection（除非出现真实 float-LUT 效果，否则不做）

无 PF、无产品 producer、CPU oracle 无调用无测试。第一优先级是把它明确标注为
“实验/未绑定”，避免被误当成 PF90005。

---

## 8. 证据与阻塞

### 8.1 已核实的证据（本机只读）

| 项 | 结果 | 方法/证据 |
|---|---|---|
| Metal host 覆盖 5 类 kernel | 是 | `metal.rs` 入口清单 §1.2 |
| Metal 产品调用 | **0** | `grep -rn 'aurora_portrait::metal' crates/engine crates/app` 无命中 |
| P15 f32 对拍 | 测试存在、未运行 | `metal_runtime.rs:58-150`（ignore） |
| Skin LUT 真实 atlas 对拍 | 测试存在、未运行 | `metal_skin_assets.rs:256-291`（ignore） |
| skin fixture == 产品 PNG | **是（0 字节差）** | `sips`→BMP 解码 vs `.rgb8`，512×512×3 |
| Smile golden pack | 4 例，CPU 自洽 | `metal_smile.rs:193-240`；manifest oracle=`pf100007_apply` |
| Mesh GPU 对拍 | 对 `raster::rasterize`，非产品 fragment | `metal_mesh_runtime.rs:228-607` |
| 默认 macOS 测试的缺资产风险 | **存在** | `metal_asset.rs:109-131` 读缺失的 `work/backend-r11/o2e/whiten_lut.f32` |
| 仓库内 ONNX 模型 | **0 个** | `find . -name '*.onnx' …` 无结果 |
| macOS GPU CI / dispatch trace | **无** | 无相关 workflow；所有 Metal 真机测试 ignore |

### 8.2 缺失资产 / 模型 / fixture（阻塞产品路径验证）

1. **`work/backend-r11/o2e/whiten_lut.f32`（以及 `decode_lut.py`）缺失**
   → `metal_asset.rs` 的 macOS 默认测试无法运行；O3B float-LUT 真机证据缺失。
   本机只有 `work/backend-r11/o1i`。
2. **ONNX Runtime macOS dylib 缺失**（见 `30-ai-audit.md`）→ 所有依赖人脸模型的 producer 不可用。
3. **人脸模型文件缺失**（`portrait_face_input_stage.rs:254-275, 359-390`）：
   `faceDetection.onnx`、`faceMesh.onnx`、`1063025060.onnx`、`323773030.onnx`、
   `2722162333/model.onnx`、`fbe.onnx` / `2102828093.onnx`（sha256 常量在 `:21-33`）。
   没有它们：skin mask、landmarks、P15 producer、PF8033、PF100007 全部无法在产品路径产生真实输入。
4. **PF100007 模型输出 `780929437` + host split spec + Grinning material**
   （`portrait_stage.rs:846-940`；`SmileHostSplitSpec`）→ 无真实笑纹输入。
5. **PF8033 模型产物 / host split 回执**（`PF8033_MODEL_ROOT_ENV`、`PF8033_*`）
   （`portrait_stage.rs:94-104`）→ 只能跑 CPU forward oracle，不能做真实产品对拍。
6. **合法真实人像照片**（不得入库）→ 0/中间/最大强度、不同肤色、多人脸、旋转裁剪、
   暗部高光等验收无法完成；只能合成。
7. **产品级 Metal 对拍 fixture**：目前没有“引擎 apply 函数 + fixed Metal backend”的 golden；
   需要新建 synthetic golden（可由 CPU 结果生成，不入库外部资产）。
8. **Mesh fragment 状态的 MSL/ABI/fixture**：Edge/Defines/Clamp/Tolerance 无 Metal 实现，
   也无对应 golden。
9. **P15 真机字节 golden**：现有只有 1e-5 f32 golden，没有 TruncU8/ROI 的产品字节 golden。

### 8.3 无法验证的项（明确写成未验证）

- 任何 Metal kernel 在本机 M2 上的真实 dispatch、GPU time、输出像素。
- 任何 PF 的 Mac 真机产品结果（缺 ORT/模型/照片）。
- Metal 与 CPU 的产品级字节等价（除库级 skin LUT fixture 的测试设计外，均未运行）。
- 性能（冷/暖、P50/P95、内存）——全部未测，不能引用 Windows 数字。
- StrictGpu/Auto 行为——代码不存在。

---

## 9. 给实现者的最小行动清单（按依赖）

1. **（协调）** 扩展证据类型：`PortraitBackend::Metal` + Metal dispatch receipt；
   `PortraitEffectRun` 是公开接口，需主控/Windows Agent 同意。
2. **（Mac-F）** 新建 `crates/engine/src/portrait_metal_stage.rs`（人像 `portrait_*stage*.rs` 属本 lane），
   先只实现 `90005|90007`；`try_metal_effect` 只依赖 `aurora_portrait::metal` 的公开入口。
3. **（协调 Mac-A）** 在 `portrait_stage.rs:5414/5415` 之间加**一个** `#[cfg(target_os = "macos")]`
   调用 block；不改 `:4340`、`:5297` 和任何 `cfg(windows)` 分支。`lib.rs` 的 `mod` 声明与
   Mac-B 协调。
4. **（Mac-F）** 在 macOS 真机运行 `cargo test -p aurora-portrait --test metal_skin_assets -- --ignored`
   （需 Mac-A 先让构建绿），记录 `-- --nocapture` 的逐字节结果；把命令、时间、device 写进证据。
5. **（Mac-F）** 再做 P15；严格用 `resolve_p15` + `TruncU8` + ROI crop/uncrop + job ticket。
6. **（Mac-F）** 补 synthetic 引擎级 golden（CPU 生成），覆盖 0/中间/最大、mask 全 0、
   CPU↔Metal 切换；真实照片项保持“未验证”，直到 §8.2 的模型/照片到位。

---

## 10. 参考索引（本审计引用的关键位置）

- 现有 Metal host：`crates/portrait/src/metal.rs`（入口 `:728, :755, :761, :823, :875, :895, :941, :1035, :1201, :1253`；
  ABI `:105-178, :655-668`；拒绝 `:311-325`；gate `:1500-1507`；dispatch `:1394-1443`；thread-local `:1482-1498`）。
- Mesh 契约：`crates/portrait/src/metal_mesh.rs:257-365, 378-466`。
- Smile 契约：`crates/portrait/src/metal_smile.rs:13-103`；MSL `native/metal/smile_warp.metal`。
- MSL：`native/metal/{P15FacialSmooth,o3b_masked_lut_whiten,skin_lut_fixed,mesh_raster,mesh_raster_serial,smile_warp}.metal`。
- 产品调度：`crates/engine/src/portrait_stage.rs:439-464, 3762-3800, 4453-4464, 4678, 4770, 5377-5425, 6887-6925,
  7031-7075, 7369-7398, 7707-7715, 7791-7913, 8198-8256, 8284-8440, 8512-8638, 8649-8736, 9035-9102, 9170-9328`。
- 人像 producer：`crates/engine/src/portrait_face_input_stage.rs:1935-2124`（P15 rows `:2081-2122`）。
- 网格产品路径：`crates/engine/src/portrait_face_stage.rs:416-485, 533-624, 1067-1148`。
- 渲染接入：`crates/engine/src/mock/process.rs:3182-3330`。
- UI：`crates/app/src/tools/portrait.rs:35-60, 101`；`crates/app/src/contract.rs:514-522, 566`。
- 证据类型：`crates/portrait/src/cuda.rs:167-180, 235-244`；`crates/portrait/src/device.rs:7-12, 60-90`。
- 注册表：`crates/portrait/src/capability_registry.rs`；`crates/portrait/assets/portrait/capability-registry-20261124.tsv`。
- 测试：`crates/portrait/tests/{metal_runtime,metal_mesh_runtime,metal_smile,metal_skin_assets,metal_asset}.rs`。

---

## 11. 实现记录（Mac-F，M3 P0-1：PF90005/PF90007 Metal 产品接线）

> 状态：**已在 M2（aarch64-apple-darwin）本机实现并验证**。合成输入下 Metal 与产品
> CPU 逐字节相等（`max_byte_diff = 0`）。真实照片/模型、Windows 原生回归与性能均
> **未验证**。未提交、未推送。

### 11.1 改动文件

| 文件 | 改动 |
|---|---|
| `crates/engine/src/portrait_stage.rs` | `PortraitBackend::Metal`；`MetalDispatch` 证据类型；`PortraitEffectRun.metal_dispatches`；macOS `try_metal_effect` + 主循环插入点；所有 `PortraitBackend` 穷尽 match 同步 |
| `crates/engine/tests/portrait_metal_skin.rs`（新增） | 引擎级 CPU/Metal 字节对拍、dispatch 证据、确定性、全 0 mask、缺 mask、强制 CPU 回归、21024 负例 |
| `crates/engine/src/portrait_hlf_wrinkle_stage_tests.rs:153-157` | 只补一个 `PortraitBackend::Metal` 分支，因新增枚举变体后该穷尽 match 必须同步；不涉及 CUDA 行为 |
| `docs/macos-adaptation/40-portrait-audit.md` | 本节 |

`try_cuda_effect`、`try_cuda_same_job_effect`、任何 `#[cfg(windows)]` CUDA 生产分支、
Mac-A 的两处类型注解均未改动。

### 11.2 证据类型（对应 §4.2）

- `PortraitBackend::Metal`（`portrait_stage.rs:444`），与 `Cpu/Cuda/Skipped` 并列。
- `MetalDispatch`（`:456-468`）：`kernel: &'static str`、`grid: (u32,u32)`（threadgroup 数）、
  `threads: (u32,u32)`、`dispatch_count: u32`。**不伪造** CUDA 专用字段
  （`ptx_bytes/ptx_isa/compute_cap/driver_version`）。
- `PortraitEffectRun.metal_dispatches: Vec<MetalDispatch>`（`:482`）。CUDA `dispatches` 仍只
  承载 CUDA 证据；两者不混用。
- 只有 `Ok(next)` 且 `metal_dispatches` 非空才写 `run.backend = Metal`、`executed = true`。
  dispatch 为空时写 `fallback_reason` 并走原 CPU 路径（与 `:5385` 的 CUDA 规则一致）。
- receipt 只在 `execute_skin_lut_fixed` 返回后构造；该函数内部 `dispatch_pipeline` 已确认
  `MTLCommandBufferStatusCompleted` 且 command buffer 无 error（`metal.rs:1434-1441`），
  因此非空 receipt 即“真实完成”证据，而不是“设备可用”。

### 11.3 调度接入点（对应 §5.1）

- 插入位置：`portrait_stage.rs:5532-5561`，即在 `#[cfg(windows)]` CUDA 证据门之后、
  `apply_cpu_effect` 之前，独立 `#[cfg(target_os = "macos")]` block，独立变量名
  `metal_result/metal_dispatches`，不跨 cfg 共享类型推断。
- 成功：写回 `current`、`backend=Metal`、`metal_dispatches=receipt`、`executed=true`、`continue`。
- 拒绝/失败：把**真实原因**追加进 `fallback_reason`，继续执行原 CPU 路径。
- 线程模型：只调用 `aurora_portrait::metal::execute_skin_lut_fixed`，复用其 thread-local
  `MetalContext`（`metal.rs:1482-1498`），不共享、不加 `Send/Sync`。

### 11.4 路由范围与拒绝（对应 §5.2/§5.7、§6）

`try_metal_effect`（`portrait_stage.rs:4735`）只在 `pf ∈ {90005, 90007}` 时返回
`Some(...)`；其余一律 `None`，即**不会被标记为 Metal**：

- **PF21024（P15）显式不接**：不在 match 分支内；负例测试证明报告不会出现 Metal。
- mesh（eye/PF202/PF8033 等）与 smile（PF100007）不在该桥内；它们继续走既有 CPU/CUDA。
- 90005/90007 使用产品同一份 `whiten_lut()`/`ruddy_lut()`（`portrait_stage.rs:9365`、
  `:9379`，`include_bytes!` 真实 PNG 解码）。
- mask 来自 `inputs.masks.mask_u8(MaskKind::Skin, w, h, region)`（与当前帧同尺寸）。
- degree 与 CPU 完全相同的 `degree.clamp(0.0, 1.0)`（`apply_skin_whiten/ruddy`）。
- 形状不符、mask 缺失、LUT 缺失、强制 CPU、host 返回 typed refusal 都写进
  `fallback_reason` 后落到 CPU。
- 复用既有 `AURORA_PORTRAIT_RENDER_BACKEND=cpu` / `AURORA_PORTRAIT_FORCE_CPU`
  （D5：不新增全局开关），仅作为 dev/AB 的“禁用 GPU”。

### 11.5 StrictGpu

本代码库**没有** `StrictGpu`/`PreferGpu` 产品设置；只有测试约定 `PORTRAIT_CUDA_REQUIRED`
与上述 dev/AB env。因此没有“Metal 失败必须报错”的模式需要实现；失败一律以
`fallback_reason` 记录后使用 CPU。若后续引入 StrictGpu，必须在 `try_metal_effect` 的
`Err` 分支改成返回 typed error，不能静默 CPU。

### 11.6 本机验证（2026-11-25，M2）

```
cargo check --workspace --all-targets --locked --target aarch64-apple-darwin
# Finished (无 error)

cargo test -p aurora-engine --locked --target aarch64-apple-darwin \
  --test portrait_metal_skin -- --test-threads=1 --nocapture
# running 5 tests ... test result: ok. 5 passed; 0 failed; 0 ignored
```

对拍覆盖：PF90005/PF90007 × 尺寸 {1×1, 17×13, 64×37, 37×64} × degree {0, 0.5, 1.0}
= **24 组**。每组：

- 引擎强制 CPU 输出 == 产品 `makeup::skin_adjust_lut` 输出；
- 引擎 Metal 输出 == 同一 CPU oracle，**`max_byte_diff=0`**（逐字节相等，alpha 保持源值）；
- 报告 `backend=Metal`，`metal_dispatches[0] = { kernel: "skin_lut_fixed",
  grid: (ceil(w/16), ceil(h/16)), threads: (16,16), dispatch_count: 1 }`；
- 第二次相同运行字节与 receipt 完全一致（确定性）。

其余断言：全 0 mask 为 identity；缺 mask → `Skipped` 且字节不变；两个强制 CPU 开关都
得到 CPU 字节并在 `fallback_reason` 中写明开关名；PF21024 不得出现 Metal。

既有 Metal host 真机回归（未改 `crates/portrait/**`）：

```
cargo test -p aurora-engine --test metal_runtime_macos        # 2 passed
cargo test -p aurora-portrait --test metal_runtime \
  --test metal_skin_assets --test metal_mesh_runtime --test metal_smile -- --ignored
# 12 + 4 + 1 + 1 = 18 passed; 0 failed
```

### 11.7 仍属未验证

1. **真实照片/真实 skin mask**：缺 ONNX Runtime dylib 与人脸模型（§8.2），只能用合成 mask。
2. **Windows 原生编译/运行**：本机只跑了 aarch64；新增代码均在 `cfg(target_os="macos")`
   或与平台无关的类型/match 中，但未在 Windows 实机验证。
3. **性能**：无冷暖帧/P50/P95；本记录只证明正确性与 dispatch 存在。
4. **`PortraitEffectRun` 是公开导出**：新增 `metal_dispatches` 字段是对外接口增量，
   `MetalDispatch` 类型目前未从 `lib.rs` 重新导出；跨 lane 消费者若需命名该类型，需协调
   在 `lib.rs` 增补 re-export。本轮未改 `lib.rs`。
5. `METAL` 真机 CI / artifact 证据仍无；以上均为本机命令输出。
