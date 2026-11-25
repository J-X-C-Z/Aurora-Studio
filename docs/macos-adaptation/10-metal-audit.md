# M2 — Metal 运行时与 Frame/驻留审计

> 只读审计。未修改任何源码：本文只新增本文件。
> 未运行 `cargo build/check/test`（另一个 Agent 持有构建锁），仅做静态阅读，
> 少数事实用 `objc2-metal` / `gpui-pre` 的随源码交叉核对。所有未运行的项标“未验证”。
>
> 审计基线：worktree `Mac-原生适配`，HEAD `9bf908e`。审计期间 Mac-A 正在修改
> `mock/mod.rs`、`gpu_runtime.rs`、`pixel_gpu.rs`、`portrait_stage.rs`、`ort.rs` 等
> （`git status` 可见未提交改动）；本文第 2 节按**当前工作树**记录，合并前需复核。

## 0. 结论摘要

1. **现有 Metal host 只服务测试/示例，没有产品路径。** `native/metal/*.metal` 源文件
   共 15 个 kernel 入口（host 按使用路径懒编译其中一部分），产品侧
   （`aurora-engine` 的 `ImageEngine` / `portrait_stage`）没有任何调用点；
   唯一调用方是 `crates/portrait/tests/*` 与 `examples/backend_probe.rs`。
2. **同步模型是“一 kernel 一 command buffer 一 `waitUntilCompleted`”**，所有资源都是
   `MTLBuffer`（`newBufferWith* options:0`，即 Shared），没有 `MTLTexture`、sampler、
   fence/event、in-flight ring 或命令批处理。它满足逐 kernel 对拍，但不是可用的连续子图。
3. **FFI 所有权有真实缺陷**：`MTLCreateSystemDefaultDevice` 是 Create 规则（返回 +1），
   host 又用 `Objc::retain` 加一次引用再在 Drop 里 release 一次，净泄漏一个 device 引用；
   `available()` 每次探测都会泄漏一次。另有 `device_pointer()` 裸指针安全函数、
   依赖 `waitUntilCompleted` 才成立的 command buffer/buffer 生命周期。
4. **引擎共享 GPU 抽象是 CUDA 形状的，而且在当前工作树里被 `#[cfg(windows)]` 掉了。**
   `mock/gpu_runtime.rs` 直接 `use aurora_portrait::cuda::{...}`；Mac-A 已把它改成
   Windows 模块 + 非 Windows 空桩。Metal 后端要复用 Slot/预算/Failure/Counters/FrameHandle，
   必须先把它拆成与后端无关的账本/池 + 后端实现，而不是新建第二套调度器。
5. **Frame 边界仍是纯 CPU**：`RgbaImage.pixels: Arc<Vec<u8>>` → app 侧 RGBA→BGRA 复制 →
   `RenderImage`（CPU bytes）→ GPUI 首次绘制时上传到 sprite atlas。
   GPUI 0.3.11 没有 RGB 外部纹理 API；`paint_surface` 存在，但只接受
   `420YpCbCr8BiPlanarFullRange`（NV12），不能无损承载照片预览。今天可行的低复制路径是
   “GPU 写 Shared BGRA buffer → 一次读回 → GPUI 上传”，**不是零复制**。

---

## 1. 现有 Metal host 清单

### 1.1 Host 结构与可用性探测

`crates/portrait/src/metal.rs` 是手写 Objective-C 消息发送 + `dlopen` 的 host：

- 框架只从绝对路径加载：`/usr/lib/libobjc.A.dylib`、Foundation、CoreGraphics、Metal
  （`metal.rs:1799-1835`）；`libobjc.A.dylib`、`Foundation`、`CoreGraphics` 缺失即失败。
- MSL 以 `include_str!` 内嵌并在运行时 `newLibraryWithSource:options:error:` 编译
  （`metal.rs:29-31, 1629-1685`），`setFastMathEnabled:NO`（`:1640-1642`）；
  没有预编译 `.metallib`。
- `framework_available()`（`metal.rs:670-673`）：仅 `cfg!(target_os="macos") && lib().is_some()`，
  不证明有 device。
- `available()`（`metal.rs:675-697`）：macOS 上再调用 `MTLCreateSystemDefaultDevice`；
  非空才 true。设备对象 retain 后立即 drop，不保留长生命周期对象。非 macOS 恒 false。
  **未在本 M2 实跑验证**；按代码应返回 true。
- `MetalContext`（`metal.rs:599-618`）：device + **一个** command queue + eager 编译的
  P15/whiten pipeline 列表 + mesh/perspective/smile/skin 的 `OnceCell` 懒编译。
  文档明确它 **不是 `Send`**，必须同线程创建/使用/销毁。
- 便捷入口 `execute_*` 使用 `thread_local! SHARED_METAL_CONTEXT`
  （`metal.rs:1482-1498`）：每个线程各自持有 device/queue/pipeline。
- P15 的 6 个 entry 在 `MetalContext::new` 时全部编译（`metal.rs:630-637, 709`），
  但 neck 两个 kernel 没有任何 host 方法，属于“编译了但从不 dispatch”。

### 1.2 入口点表

约定：
- 绑定列括号内为 uniform 结构体字节数；“ABI”列给出 buffer 数量/类型。
- “同步”列默认是 `dispatch_pipeline` 的 `commandBuffer → computeCommandEncoder →
  dispatchThreadgroups → endEncoding → commit → waitUntilCompleted → status/error`
  （`metal.rs:1394-1443`）。
- “产品接入”= 可从 `ImageEngine`/`portrait_stage` 到达。
- “CPU reference”= 仓库内有可对拍的 CPU 实现（不是测试内嵌 oracle）。

| 文件 | kernel | 绑定 / uniform | ABI / 精度 | dispatch grid | 同步 | 产品接入 | CPU reference |
|---|---|---|---|---|---|---|---|
| `native/metal/P15FacialSmooth.metal:216` | `p15_facial_smooth_composite` | 6 device buffer + `P15Uniforms` 88B (`buffer 6`) | 5×f32 RGBA + f32 R + out f32 RGBA；FP32 | 16×16 tg，`ceil(w/16)×ceil(h/16)` | 单 cb + wait | 否（仅测试） | 是：`kernels::facial_smooth::composite_pixel` (`facial_smooth.rs:106`) |
| 同上 `:257` | `p15_surface_blur_pass` | src/out + `P15Uniforms` 88B | f32 RGBA；FP32 | 16×16 | 单 cb + wait | 否（`MetalContext::blur`，测试） | 否（`skin_models::texture_surface_blur` 是另一套 PF8069 法则） |
| 同上 `:276` | `p15_gauss_pass` | src/out + 88B | f32 RGBA；FP32 | 16×16 | 单 cb + wait | 否（同上） | 否 |
| 同上 `:297` | `p15_surface_pass` | src/out + 88B | f32 RGBA；FP32 | 16×16 | 单 cb + wait | 否（同上） | 否 |
| 同上 `:320` | `p15_neck_repair_tails` | 6 device buffer + `P15NeckTailUniforms` 8B | f32 planes；FP32 | 1D（host 未实现） | 无 dispatch | **否，host 从不调用** | 否 |
| 同上 `:350` | `p15_neck_max_pool` | blendTail/coarse + 无 uniform | f32；常量 10×8/640 | 常量网格（host 未实现） | 无 dispatch | **否，host 从不调用** | 否 |
| `native/metal/o3b_masked_lut_whiten.metal:93` | `o3_coverage_max_mask_projection` | u8 src / u8 dst + `O3MaskProjectionParams` 80B | u8 mask；整数投影 + FP32 辅助 | 16×16，按 destination | 单 cb + wait | 否（测试） | 否（仅测试内嵌 expected，`tests/metal_runtime.rs:270`） |
| 同上 `:205` | `o3_masked_lut_whiten` | BGRA8 src/dst + u8 mask + `float4` LUT atlas + `O3MaskedWhitenParams` 48B | u8 BGRA；FP32 采样 + half-even 量化 | 16×16 | 单 cb + wait | 否（测试） | 是：`kernels::skin_color::masked_lut_blend` (`skin_color.rs:886`) |
| `native/metal/mesh_raster.metal:249` | `mesh_raster_select` | dest f32 / ebo u16 / params / atomic u32 selected / atomic u32 coverage + `MeshRasterParams` 36B | f32 顶点 + u16 + u32；FP32 | 1D，64 threads，`ceil(triangles/64)` | 单 cb + wait（两遍=2 cb） | 否（测试） | 是：`raster::rasterize`（`metal_mesh.rs:454-468`） |
| 同上 `:318` | `mesh_raster_emit` | dest/uv/ebo/src u8/out u8/params/selected/coverage | 同上 | 1D，64，`ceil(pixels/64)` | 单 cb + wait | 否（测试） | 同上 |
| 同上 `:376` | `mesh_raster_emit_perspective` | emit 的 8 个 buffer + `float2` `[depth, reciprocal_w]` | FP32 | 1D，64 | 单 cb + wait | 否（测试；shader 注释“尚未被选中”已过时，host `:1035-1128` 会 dispatch） | 部分：`raster` + `perspective_uv` (`metal_mesh.rs:525`) |
| 同上 `:438` | `mesh_raster_serial_perspective` | 7 buffer（含 perspective） | FP32 | 1D，64 | 单 cb + wait | 否（perspective compile-fail fallback，测试） | 部分 |
| `native/metal/mesh_raster_serial.metal:184` | `mesh_raster_serial` | dest/uv/ebo/src/out/params | FP32 + u8 | 1D，64 | 单 cb + wait | 否（`mesh_raster_serial`，测试） | 是 |
| `native/metal/smile_warp.metal:37` | `pf100007_smile` | BGRA8 src/lip/mouth + f32 dx/dy + out + `SmileParams` 32B | u8 BGRA，双线性 + 截断存储；FP32 | 16×16 | 单 cb + wait | 否（测试） | 是：`kernels::smile_warp::pf100007_apply` (`smile_warp.rs:235`) |
| `native/metal/skin_lut_fixed.metal:104` | `skin_lut_fixed` | RGBA8 src + u8 mask + RGB8 LUT + out + `SkinLutParams` 32B | u8；12-bit floor / 定标整数混合 | 16×16 | 单 cb + wait | 否（测试；文件头“candidate only”已过时，host 已接） | 是：`kernels::makeup::skin_adjust_lut` (`makeup.rs:411`) |

补充事实：

- 所有 MSL 都 `#pragma clang fp contract(off)`，partial 运算用 `volatile` 阻止 FMA 合并
  （如 `P15FacialSmooth.metal:5-6`、`smile_warp.metal:5-6`、`skin_lut_fixed.metal:42-47`）。
- 所有 kernel 都是 `device ... [[buffer(n)]]`；**没有任何 `texture2d`/sampler**。
  `metal.rs:308-309` 与 `metal_mesh.rs:253` 的 `TEXTURE_BINDINGS` 都是空数组。
- P15 composite 返回**未量化 f32 RGBA**，alpha 由 host 从源逐位恢复
  （`metal.rs:725-750, 1510-1514`）；RGBA8 打包用 half-away（`:1458-1463`）。
- O3B whiten 的 alpha 也从源恢复（`metal.rs:868-870`），LUT 是 512×512 `float4`（4 MiB）。
- Mesh host 的 out buffer 先以 source bytes 填充，未覆盖像素保留原图
  （`metal.rs:967-968`）；`selected_fill` 按 overlap 规则预填 0 或 0xFF（`:650-653`）。

### 1.3 Buffer/纹理布局与 ABI

| 结构体 | 字节 | 位置（host assert / shader assert） |
|---|---|---|
| `P15Uniforms` / `P15FacialSmoothUniforms` | 88 | `metal.rs:130` / `P15FacialSmooth.metal:36` |
| `P15NeckTailUniforms` | 8 | 仅 shader `P15FacialSmooth.metal:44`（host 无结构） |
| `WhitenParams` / `O3MaskedWhitenParams` | 48 | `metal.rs:150` / `o3b_masked_lut_whiten.metal:50` |
| `ProjectionParams` / `O3MaskProjectionParams` | 80 | `metal.rs:178` / `o3b_masked_lut_whiten.metal:34` |
| `MeshRasterRules` | 16 | `metal_mesh.rs:95` |
| `MeshRasterParams` | 36 | `metal_mesh.rs:96` / `mesh_raster.metal:28` |
| `SmileParams` | 32 | `metal_smile.rs:27` / `smile_warp.metal:23` |
| `SkinLutParams` | 32 | `metal.rs:668` / `skin_lut_fixed.metal:23` |

缓冲区：全部由 `newBufferWithBytes:length:options:0` 或
`newBufferWithLength:options:0` 分配（`metal.rs:1348-1382`）。options=0 即
`MTLResourceOptions` 默认（`StorageModeShared` + 默认 cache）；读回直接读
`contents`（`metal.rs:1605-1627`），只有在 `waitUntilCompleted` 之后才安全。
没有 `StorageModePrivate`、没有 staging blit、没有 `MTLTexture`、没有 sampler。
本审计未找到任何代码读取 `recommendedMaxWorkingSetSize`。

### 1.4 Dispatch grid 与线程组

- 2D kernel：固定 `16×16` threadgroup，`groups = ceil(w/16), ceil(h/16)`
  （`metal.rs:1574-1586`）。
- 1D kernel（mesh/serial）：固定 64 threads，`groups = ceil(count/64)`
  （`metal.rs:1588-1603`）。
- **没有**查询 `threadExecutionWidth` / `maxTotalThreadsPerThreadgroup`；
  **没有** threadgroup memory、simdgroup、tile+halo。
- 越界保护在 shader 内做（`gid >= width/height`），因此 grid 向上取整是安全的。

### 1.5 同步模型

- `dispatch_pipeline` 每次调用创建一个 command buffer、一个 compute encoder、
  一次 dispatch、`commit`、`waitUntilCompleted`、检查 `status != 4` 与 `error`
  （`metal.rs:1394-1443`）。
- mesh 两遍（select + emit）是**两个 command buffer、两次 wait**，源码注释以这一点
  作为跨 pass 可见性的保证（`metal.rs:935-940`）。
- 没有 `MTLSharedEvent`/fence、没有 `addCompletedHandler`、没有多帧 in-flight。
- host 在本函数返回前就 drop command buffer/encoder（`Objc` Drop，`:1763-1769`）；
  正确性完全依赖已经 wait 完成。
- GPUI 的 renderer 是另一套：一个 queue、一个 command buffer 画整帧，
  `presentDrawable` 后 `commit`，不在 CPU 等待；只有 `render_to_image` 才
  `waitUntilCompleted`（`gpui-pre-apple-0.3.11/src/metal_renderer.rs:397, 595-646, 724-789`）。

### 1.6 产品接入与 CPU reference

- 全仓库搜索 `aurora_portrait::metal` 的结果只有 `crates/portrait/tests/*` 与
  `crates/portrait/examples/backend_probe.rs`。`portrait_stage.rs` 只走
  `effect_router`（CPU/CUDA/Auto），`effect_router` 的 `Backend` 枚举没有 Metal
  （`effect_router.rs:34-42`）。
- `aurora_portrait::device::select` 虽然已有 `Backend::Metal` 与 `DeviceProbe`
  （`device.rs:19-25, 57-70`），但全仓库没有生产调用方，只有单元测试
  （`rg 'device::select'` 仅命中 `device.rs` 自身）。
- 因此“人像 Metal 产品接入”当前状态是 **未接入**；不是“部分接入”。

---

## 2. 引擎 GPU 阶段与共享抽象

### 2.1 阶段现状（当前工作树）

| 阶段 | 文件 | 开关 / 默认 | 运行时 | 数据域 / 语义 | 非 macOS 现状 |
|---|---|---|---|---|---|
| Develop（点链） | `mock/develop_gpu.rs` | `AURORA_DEVELOP_GPU`；默认 `cfg!(windows)`（`:1496-1503`） | **绕过共享 `Gpu`**：自有 `Box<Nvidia>` + `SendBuffer`（`:1341-1371`），只复用 `VramLease` | 已采样 linear RGB → `linear_edits` → C1 camera colour 或 sRGB encode → `develop_display` → banding → gamma；纯 point op | `mock/mod.rs:92-93` 仅 Windows 挂载；所有调用点在 `process.rs` 以 `#[cfg(windows)]` 收口 |
| Clarity precurve / structure | `mock/clarity_gpu.rs` | `AURORA_CLARITY_GPU`；默认 true（`:252-257`） | 共享 `Gpu` + `FrameHandle` + `Slot::{Frame,Scratch}`（`:25, 92-160`） | precurve 在 linear/pre-display；structure 在 quantized u8/u16 | `mock/mod.rs:94-136` 空桩，全部返回 false |
| NR（luma/colour） | `mock/nr_gpu.rs` | `AURORA_NR_GPU`；默认非 test 为 true（`:95-100`） | 共享 `Gpu::with_stream()` + `Slot::{Frame,Aux0}`（`:62-66`） | `[[f32;3]]` display 域；luma/colour 两段 | `mock/mod.rs:139-160` 空桩 |
| Detail（legacy u8） | `mock/detail_gpu.rs` | `AURORA_DETAIL_GPU=1` 才开；默认 false（`:64-74`） | 共享 `Gpu` + `Slot::Frame`（`:35-50`） | 8-bit u8：旧版 luma NR / 5×5 colour / USM / lens | `mock/mod.rs:88-89` 仅 Windows |
| Pixel（vignette/LCC/grain） | `mock/pixel_gpu.rs` | `AURORA_PIXEL_GPU` + 每算子开关（`:30-62`） | 共享 `Gpu` + `Slot::{Frame,Scratch,Aux0,Aux2}` + static buffer（`:150-180`） | gain plane / u8 grain | `mock/mod.rs:21-87` 空桩 |
| Blemish gain-map | `blemish_gainmap_cuda.rs` | `AURORA_BLEMISH_GPU=0` 关闭 | 直接 `Nvidia`/`PinnedBuffer`（`:13-21`） | 8u/16u gain-map 采样 | 整个实现包在 `#[cfg(windows)] mod imp`（`:13`）；`lib.rs:101` 无条件声明模块 |

### 2.2 可复用的共享抽象（Metal 后端应插在这里）

`crates/engine/src/mock/gpu_runtime.rs` 已经定义了一套与阶段无关的契约：

- **Slot / 增长式缓冲**：`Slot::{Frame,Scratch,Aux0,Aux1,Aux2}`，只增不减
  （`:26-47`）；`Gpu::reserve` 处理增长、回滚、预算拒绝（`:775-811`）。
- **预算账本**：`VramLedger` / `VramLease`（`:288-443, 567-621`），进程级单一预算
  （默认 2560 MiB，`:228`），支持 evictable 与 reclaim 回调（`:455-482`），
  按 stage 优先级驱逐。
- **调度/传输计数**：`Counters { uploads, downloads, h2d_bytes, d2h_bytes, launches,
  syncs, fused_frames, skipped_* }`（`:195-211`），`launches: Vec<Launch>`（`:186-193`）。
- **失败原因**：`Failure` 七类（`:50-81`），`note_fallback` 与
  `FallbackReport`（`:126-178`）；每个 stage 必须记录“GPU 拒绝”而不是静默成功。
- **驻留帧**：`FrameLayout { width,height,format,bytes,generation }`（`:623-641`）
  与 `FrameHandle`（`:643-691`），后者已按 stage 账本计费。
- **异步/锁页**：`with_stream()`、`upload_async/download_async/sync`（`:754-767, 944-979`）、
  `register_host`/`pinned`（`:981-993`）。
- **设备丢失**：`is_device_lost` / `mark_lost` / `reopen`（`:107-124, 1036-1066`）。
- **人像桥**：`PortraitObserver` 把 portrait CUDA 事件接入同一套计数
  （`:1131-1231`）。

Metal 后端应复用这些**语义**（Slot、账本、Failure、Counters、FrameHandle），
但不能直接把 `Gpu` 当后端无关类型：它内部字段是 `Nvidia`/`Stream`/`DeviceBuffer`
（`:705-718`），`gpu_runtime.rs:21` 顶层 `use aurora_portrait::cuda::{...}`。
当前工作树里 `mock/mod.rs:12-20` 已把它变成 `#[cfg(windows)]` + 空桩，
所以 **macOS 上现在连这套抽象都没有编译**。

### 2.3 不能直接复用/需要拆分的点

1. `Gpu` 是具体类型而非 trait：`open/open_sync` 直接 `Nvidia::open_ptx`（`:723-746`），
   `alloc` 直接 `alloc_buffer`（`:813-820`）。要么引入 `enum Backend`，要么引入
   最小 trait（`alloc/free/upload/download/launch/sync`），二者都能保住现有调用点。
2. `develop_gpu` 完全绕过 `Gpu`（`:41, 1341-1371`），只复用 `VramLease`。
   即使把 `gpu_runtime` 变成通用后端，develop 仍需单独迁移；Metal develop 必须
   先决定是补上共享 Slot/FrameHandle，还是继续自带 pool（不建议后者）。
3. `FrameLayout.generation` 只是字段：clarity 构造时固定写 0
   （`clarity_gpu.rs:131, 196`），全仓库没有比较/失效逻辑。Metal 驻留必须真正实现
   “generation 不匹配即重建”，否则就是虚假的驻留契约。
4. `FrameFormat` 只有 `F32Rgb12` / `F32Planes`（`:635-641`），没有 BGRA8/RGBA8/RGBA16、
   没有 stride/ROI/halo/色彩空间/位深字段。spec 4.2 要求的 Frame/Residency 元数据
   目前大半不存在。
5. 预算是固定常量 2560 MiB（`:221-228`），不感知统一内存总占用；Metal 侧若再建
   独立预算就会违反“不能把 RAM 和 VRAM 各自全部分配”（spec 4.2）。

---

## 3. engine ↔ app 的 Frame 边界

### 3.1 类型与复制路径

1. 引擎输出：`RgbaImage { width, height, pixels: Arc<Vec<u8>> }`，8-bit straight-alpha
   RGBA、行主序、无 padding（`crates/engine/src/engine.rs:21-27`）。
   `PreviewFrame` 只是 `image + histogram + generation + elapsed_ms + degraded +
   gamut_warning`（`engine.rs:129-141`）。
2. `MockEngine::preview_frame` 把 `process::render_at` 的 `out.rgba` 放进
   `Arc<Vec<u8>>` 并写入 preview cache（`mock/mod.rs:298-306`）；命中缓存时
   `CachedFrame::into_image` 共享同一个 Arc（`:282-287`）。
3. app 在 **background executor** 上调用 `engine.render_preview_fast`
   （`crates/app/src/state/render.rs:973-1003`）。
4. `into_bgra_pixels`（`render.rs:73-105`）：
   - `Arc::try_unwrap` 成功（缩略图等独占）→ 原地 `swap(0,2)`；
   - 失败（preview cache 还持有 Arc）→ **新分配一个 BGRA `Vec<u8>`**，逐像素读写一次。
   之后 `image::RgbaImage::from_raw` + `RenderImage::new(vec![image::Frame::new(buffer)])`
   （`:111-118`）。
5. GPUI `RenderImage` 内部就是 `SmallVec<[image::Frame; 1]>` 的 **CPU BGRA bytes**
   （`gpui-pre-0.3.11/src/assets.rs:42-49, 59-76`）。首次 `paint_image` 时，
   `data.as_bytes(frame_index)` 作为 `Cow::Borrowed` 交给 sprite atlas
   （`gpui-pre-0.3.11/src/window.rs:5280-5310`），再上传到 **BGRA8Unorm** 纹理
   （`gpui-pre-apple-0.3.11/src/metal_atlas.rs:148-184`，Apple GPU 用 Shared，
   否则 Managed）。
6. 结论：**帧从引擎到屏幕至少 1 次 RGBA→BGRA 全帧复制 + 1 次 CPU→GPU atlas 上传**；
   没有任何 MTLBuffer/MTLTexture 跨 crate 传递。

### 3.2 GPUI 0.3.11 的 Metal 能力（已核对 vendored 源）

- macOS 后端就是原生 Metal：`MetalRenderer` 持有 `Retained<MTLDevice>`、
  `Retained<MTLCommandQueue>`、`CAMetalLayer`（`gpui-pre-apple-0.3.11/src/metal_renderer.rs:128-157`）。
  - layer：`BGRA8Unorm`、可选 Display P3 colorspace、`maximumDrawableCount=3`、
    opaque（`:194-225`）。
  - device 选择：`MTLCopyAllDevices` 里按 `(isRemovable, !isLowPower)` 取最小，
    失败才回退 `MTLCreateSystemDefaultDevice`（`:244-258`）。M2 只有一个 GPU，
    两者等价；**API 上不保证与引擎自行 `MTLCreateSystemDefaultDevice` 的指针相同**。
  - queue：一个 `newCommandQueue`（`:397`）。
  - 绘制：`nextDrawable` → `render_frame` → `presentDrawable` + `commit`，
    **不等 CPU**（`:595-646`）；只有离屏 `render_to_image` 才
    `waitUntilCompleted`（`:724-789`）。
  - 存储模式：render target 用 Private/Memoryless（`:540-565`）；
    headless/离屏用 Shared/Managed（`:764-768`）；instance buffer 按
    `hasUnifiedMemory()` 选 Shared 或 Managed（`:301-306, 321-325`）。
- 外部纹理 API：
  - **有**：`Window::paint_surface(bounds, CFRetained<CVPixelBuffer>)`
    （`gpui-pre-0.3.11/src/window.rs:5382-5399`）与 `Surface` element
    （`elements/surface.rs:11-27, 60-110`）。
  - **但**Metal renderer 强制要求
    `kCVPixelFormatType_420YpCbCr8BiPlanarFullRange`（NV12），并把两平面
    Y/CbCr 作为 `R8Unorm`/`RG8Unorm` 纹理在 fragment shader 里转 RGB
    （`gpui-pre-apple-0.3.11/src/metal_renderer.rs:1411-1469`）。
    所以它只适合视频帧，**不能无损承载 RGBA/BGRA 照片预览**。
  - **没有**任意 `MTLTexture`/`IOSurface`/`RenderImage` GPU-handle 注入 API。
    `RenderImage` 只有 CPU `image::Frame`（assets.rs:42-49）。
- renderer 是私有的：`MacPlatform` 只存 `renderer::Context`（`gpui-pre-macos-0.3.11/src/platform.rs:164-170`），
  window 内的 `renderer` 字段私有（`window.rs:661, 1010`）；
  `MetalRenderer` 的公开方法只有 `layer/layer_ptr/sprite_atlas/render_*`
  （`metal_renderer.rs:446-457, 710-815`），**没有 `device()`**。
  macOS 的 `Window::gpu_specs()` 直接返回 `None`（`gpui-pre-macos-0.3.11/src/window.rs:2172-2174`），
  因此 `app/src/perf.rs:201` 在 macOS 上记录不到 GPU 名称。
- 色彩边界：layer 与 atlas 都是 `BGRA8Unorm`（非 sRGB 格式），默认
  `WindowOptions.use_display_p3 = false`（`gpui-pre-0.3.11/src/platform.rs:2818`）；
  app 的 workspace/live_view 只透传 `TitleBar::window_options()`，该外部实现是否
  打开 P3 **未验证**。引擎已在 CPU 完成 ICC/proof/display profile
  （`engine.rs:149-190`），GPUI 对 atlas 字节不做 sRGB 解码或矩阵转换。
  因此现状是“引擎输出什么 BGRA，GPUI 就按 layer colorspace 解释”；若未来启用
  P3 layer，必须确认值域解释，避免双重或错误转换。
- 免费得到的东西：Metal 窗口呈现、CAMetalLayer 尺寸/缩放、P3 colorspace 设置、
  drawable 环（3）、display link 驱动、sprite atlas 的 GPU 纹理与上传、
  private/memoryless render target、headless 渲染器（仅 `bench-support`/`test-support`
  feature 下可用，`metal_renderer.rs:1889-1903`；本 app 未开启，`crates/app/Cargo.toml`）。

### 3.3 零复制/低复制路径需要什么

- 需要引擎 Frame 能携带 GPU 资源：`MetalFrame { buffer/texture, width, height,
  stride, format, color_space, bit_depth, roi, halo, generation, lease }`，
  而不是只有 `Arc<Vec<u8>>`。
- 需要一个能接受 GPU 纹理的显示 sink。当前只有三条路：
  1. **改 GPUI**：新增 BGRA8/IOSurface 外部纹理 API（最小 patch：一个
     `paint_image_from_surface` 或在 `RenderImage` 里允许 `MTLTexture`）。需要与主
     Agent 协调，超出 Mac-B 所有权。
  2. **走 `paint_surface`**：受 NV12 限制，色度 4:2:0 会破坏照片预览，不可接受。
  3. **bounded staging（今天唯一可落地）**：GPU 直接写 BGRA8 到 Shared `MTLBuffer`，
     `read_into` 一次读回到 `Vec<u8>`，构造 `RenderImage`；GPUI 再上传 atlas。
     这去掉 RGBA→BGRA 交换，但不减少 GPU→CPU→GPU 的两次复制，**不得称零复制**。
- 同设备要求：引擎自建 device 与 GPUI device 必须是同一个；M2 实际只有一个 GPU，
  但 GPUI 不暴露 device，只能靠“系统默认设备”约定，不能做指针相等性校验。
  Intel 双 GPU 机器上 GPUI 可能选低功耗 GPU 而系统默认是另一个，必须在代码里
  探测并拒绝跨设备共享（本机 M2 不触发，标为未验证）。
- 还需要 in-flight ring / completion fence：一旦不逐 kernel `waitUntilCompleted`，
  在 GPU 完成前把 buffer 交给 CPU/GPUI 会读到旧帧或释放中的资源。

---

## 4. 形成一个连续 GPU 子图还缺什么

### 4.1 当前真实的顺序与色彩域（不可按示意重排）

`process.rs::render_sized` 的预览路径（与 `detail.rs` 合并）实际顺序：

1. **采样/几何**：`source_sample_at` / `BandSampler` 在 CPU 完成（`process.rs:2617-2627`）。
   区域、缩放、bicubic/bilinear、lens warp 都在这一步。
2. **Pixel gain plane**：vignette/LCC 在 CPU 侧准备（`process.rs:2523-2556`，
   GPU 版仅 Windows），随后作为 gain 参与 develop 的 `linear_edits`。
3. **Develop 点链**：`develop_pre_curve`（`linear_edits` 含 gain → `encode_srgb` 或
   C1 `develop_srgb`）→ clarity precurve（可选）→ `develop_display`
   （tone/levels/curves/saturation/colour balance/BW）→ `output_banding` → gamma
   （`process.rs:3440-3560, 3639-3700`）。layers 不在这里。
4. **Clarity precurve** 插在 pre-curve 与 display 之间（`process.rs:2704-2705`；
   `clarity_gpu` 的 `apply_precurve`，`:511-521`）。
5. **Display encode**：`develop_display` 输出 display-referred 值，随后由调用方
   clamp 到 [0,1]；u8 路径在 `develop_line_with` 里量化（`process.rs:3510-3523`），
   f32 路径保留 f32（`:2705` 处才 clamp）。**不要在 develop 内部提前 clamp**。
6. **Structure**（clarity 的第二部分）在 develop 之后、detail 之前作用于 u8/u16
   （`process.rs:2875-2900`）。
7. **Detail**（display 域）：
   - f32 路径：single-pixel → luma/colour NR → USM
     （`detail.rs:41-74`）；
   - u8 路径：single-pixel/luma NR → colour NR → moire → diffraction → USM →
     lens/dust/movement/falloff → grain（`detail.rs:547-667`）。
8. **Spots / retouch**：在 detail 之后、layers 之前（`process.rs:3005-3050`）。
9. **Layers/Composite**：`apply_layers` 在最后（`process.rs:3047`）；不是“develop 之后、
   detail 之前”。`develop_gpu` 明确拒绝有 layers 的帧
   （`develop_gpu.rs:1841-1843`）。
10. **Portrait**：8-bit 阶段，CPU/ORT（`process.rs:3050-...`）。
11. **Histogram**：CPU（`process.rs:4912`）；**ICC/proof/display profile**：CPU
    （`engine.rs:149-190`）。
12. app RGBA→BGRA、GPUI atlas 上传（见第 3 节）。

### 4.2 Gap 清单

| # | 缺口 | 证据 | 影响 |
|---|---|---|---|
| G1 | 共享运行时在 macOS 不存在 | `mock/mod.rs:12-20`（Mac-A 未提交改动） | Metal 无法复用 Slot/预算/Failure/Counters |
| G2 | `Gpu` 绑死 CUDA 类型 | `gpu_runtime.rs:21, 705-718` | 需要后端抽象/枚举 |
| G3 | Develop 绕过共享运行时 | `develop_gpu.rs:41, 1341-1371` | 即使有通用后端，develop 仍要迁移 |
| G4 | 无 Metal develop 点链（WB/曝光/矩阵/曲线/LUT/Levels/HSL/BW/banding/gamma） | 只有 CUDA PTX `native/cuda/develop_gpu.ptx` | 连续子图的第一段缺失 |
| G5 | 无 Metal 采样/几何/lens warp | 采样在 CPU `process.rs:2617-2627` | 无法做到“源→输出”全驻留；只能从已采样 plane 开始 |
| G6 | 无 Metal clarity precurve/structure | `clarity_gpu` CUDA-only；非 Windows 空桩 | 子图中段断裂 |
| G7 | 无 Metal NR/USM（f32/u8） | `nr_gpu` CUDA-only；`detail_gpu` CUDA-only | 子图后段断裂 |
| G8 | 无 Metal moire/diffraction/lens/dust/grain | `detail.rs:621-667` 全 CPU/Windows CUDA | 只能显示域前半段驻留 |
| G9 | 无 Metal composite（layers/masks）/spots/retouch | `process.rs:3005-3047`；`develop_gpu.rs:1841-1843` | “Composite→Display”不能进同一子图 |
| G10 | 无 Metal histogram/ICC/proof | `process.rs:4912`；`engine.rs:149-190` | 辅助分析仍 CPU，不能阻塞主链 |
| G11 | 无 command 批处理/fence/in-flight | `metal.rs:1394-1443` | 每 kernel 一次同步，无法形成子图 |
| G12 | 无 texture/sampler/private 中间资源 | `metal.rs:1352-1381`；`TEXTURE_BINDINGS` 空 | 无法做硬件采样/私有中间纹理/blit |
| G13 | FrameHandle 的 generation 未实现 | `gpu_runtime.rs:632`；`clarity_gpu.rs:131,196` | 过期驻留可能覆盖新帧 |
| G14 | Frame 元数据不足（stride/format/ROI/halo/colorspace/bit depth） | `gpu_runtime.rs:623-641`；`engine.rs:21-27` | 无法正确表示 tile/宽色域/16-bit |
| G15 | 无 Metal 端到端/性能证据 | `app/src/perf.rs:84-97, 172-180` | macOS 上 GPU 阶段无计时、RAM 统计为 None |
| G16 | 产品选择层没有 Metal | `effect_router.rs:34-42`；`device.rs` 仅测试 | 即使做了 runtime 也没有真实调用路径 |
| G17 | “统一内存池”只覆盖 CUDA stage slots | `VramLedger` 只按 stage tag 记账（`gpu_runtime.rs:279-443`）；GPUI atlas、ORT/portrait、导出、CPU source/mip 均未计入 | 直接相加 RAM 与 GPU 预算会超卖；Metal 预算必须并入同一个账本 |

### 4.3 目标拓扑（遵序，不投机重排）

今天可成立的第一条“连续”子图（只覆盖 f32 路径、无 layers/spots/portrait）：

```
CPU: source sampling + vignette/LCC gain plane（保持现状）
  └─ upload 一次 f32 linear RGBA
GPU: develop point chain → clarity precurve → develop_display → NR → USM
  └─ 一次 download（RGBA8 或 BGRA8）
CPU: structure（现有后续阶段，暂不移植）→ spots/layers/portrait → histogram/ICC
```

这条链与 `process.rs` 的 `gpu_two_pass` 分支（`:2603-2681`）数学顺序一致：
precurve 必须在 display 之前，NR/USM 在 display 之后。**不得**把 structure 提到
develop 之前，也不得把 layers 从最后移进子图。

之后的扩展顺序（按依赖）：
1. 把 sampling/geometry 移植到 MSL（需与 `source_sample_at` 对拍坐标/边界/ROI/halo）。
2. 把 structure 与 8-bit detail（moire/diffraction/lens/grain）移植，使 u8 路径也能驻留。
3. 再考虑 layers/masks 栅格化与 alpha blend；这要求先移植 mask 权重与 `local()`。
4. Display 侧的 GPU→GPUI 交接只有在 GPUI 增加 BGRA 外部纹理后才可能零复制。

---

## 5. 建议的 Metal 运行时模块布局

所有权：按 `OWNERSHIP.md`，Mac-B 新增 `crates/engine/src/metal/*` 与
`crates/engine/src/metal_runtime.rs`；`crates/engine/src/lib.rs` 只加 `mod` 声明。
`mock/mod.rs` 属 Mac-A，改动前报备；`crates/portrait/src/metal*.rs` 属 Mac-F。

### 5.1 文件

```
crates/engine/src/metal_runtime.rs        // 唯一公开入口（cfg(target_os="macos")）
crates/engine/src/metal/mod.rs
crates/engine/src/metal/ffi.rs            // Objc/autorelease/MTL 绑定 + 错误映射
crates/engine/src/metal/device.rs         // device + 单 queue + 能力探测
crates/engine/src/metal/library.rs        // LibraryCache + PipelineCache
crates/engine/src/metal/pool.rs           // BufferPool / TexturePool / size classes
crates/engine/src/metal/residency.rs      // MetalFrame / generation / lease
crates/engine/src/metal/encode.rs         // CommandBatch：一个 cb 编多个 kernel
crates/engine/src/metal/workspace.rs      // 每帧 scratch ping-pong + in-flight ring
crates/engine/native/metal/<stage>.metal  // Mac-C 负责
crates/engine/src/mock/<stage>_metal.rs   // Mac-C 负责
```

不要在 `metal/` 里复制以下已有物：`VramLease/VramLedger`、`Failure`、
`Counters/Launch`、`FrameLayout`、`FallbackReport`。它们只在 `gpu_runtime` 中保留一份。

### 5.2 最小公开 API（建议）

```rust
// metal_runtime.rs
pub struct MetalRuntime;                 // device + queue + caches + pools
pub struct MetalWorkspace;               // one in-flight frame
pub struct MetalFrame;                   // committed, awaiting CPU readback
pub struct MetalProbe { pub name: String, pub unified: bool,
                        pub recommended_working_set: u64, pub max_buffer_len: u64 }
pub struct PipelineKey(&'static str, &'static str); // (library, entry)
pub struct Grid { pub groups: [u32;3], pub threads: [u32;3] }
pub enum Binding<'a> { Buffer(&'a MetalBuffer), Uniform(&'a [u8]) }

impl MetalRuntime {
    pub fn probe() -> Option<MetalProbe>;
    pub fn shared() -> Result<&'static MetalRuntime, Failure>; // OnceLock, 单 queue
    pub fn begin(&self, desc: FrameDesc) -> Result<MetalWorkspace, Failure>;
    pub fn counters(&self) -> Counters;      // 复用 gpu_runtime::Counters
    pub fn reclaim_idle(&self) -> usize;     // 复用 VramLease 的 reclaim hook
}
impl MetalWorkspace {
    pub fn dispatch(&mut self, key: PipelineKey,
                    bindings: &[Binding<'_>], grid: Grid) -> Result<(), Failure>;
    pub fn commit(self) -> Result<MetalFrame, Failure>;      // commit，不 wait
}
impl MetalFrame {
    pub fn generation(&self) -> u64;
    pub fn read_into(&self, dst: &mut [u8]) -> Result<(), Failure>; // 唯一 wait 点
}
```

内部（不公开）：
- `LibraryCache`：`(msl_source_hash, fast_math=false)` → `MTLLibrary`。
- `PipelineCache`：`(library, entry, constants)` → `MTLComputePipelineState`。
  首次创建后记录 `threadExecutionWidth` / `maxTotalThreadsPerThreadgroup`，
  grid 由调用方或小策略选择，**不硬编码 16×16**。
- `BufferPool`：按 size class 复用；上传/uniform/readback 用 `StorageModeShared`；
  多阶段中间结果用 `StorageModePrivate` + blit/staging；每个 buffer 附带
  `VramLease`（唯一预算来源）。
- `Workspace`：2–3 帧 in-flight ring，`addCompletedHandler` 或 `MTLSharedEvent`
  回收到 pool；只有 `read_into`/display 边界才等待。
- `CommandBatch`：一个 `MTLCommandBuffer` 顺序编多个 compute encoder；
  相邻 kernel 之间不 `waitUntilCompleted`。

### 5.3 与现有抽象的挂接点

- `gpu_runtime::Gpu` → 改成 `Gpu { backend: Backend }`，其中
  `enum Backend { Cuda(CudaGpu), Metal(MetalGpu) }`；对外方法签名不变
  （`reserve/upload/download/launch/sync/frame`），这样 clarity/nr/detail/pixel
  的调用点可以逐步切换，不需要一次性重写。
- `FrameHandle` → 内部 buffer 改为 `enum ResidentBuffer { Cuda(DeviceBuffer),
  Metal(MetalBuffer) }`；`FrameLayout.format` 增加 `Bgra8/Rgba8/Rgba16`，
  并真正比较/推进 `generation`。
- `VramLease` 不变；macOS 预算不新增第二个池，而是从统一内存账本划一段给 GPU
  （不要 `RAM + recommendedMaxWorkingSetSize` 相加）。
- `Failure` 沿用现有七类，Metal 错误映射到 `Unavailable/Driver/Invalid/Refused`，
  `note_fallback` 的 stage tag 保持 `develop/clarity/nr/detail/pixel`，
  这样 `FallbackReport` 的语义在 Windows/macOS 一致。
- `device::DeviceProbe`：由 `MetalRuntime::probe()` 填充
  `Backend::Metal { driver_ready, kernel_compiled, fp32: true, fp16: false,
  max_width/height }`。这只是一次探测，不是调度。
- 产品调度：`process.rs` 的 stage 顺序与 `app/src/state/render.rs` 的
  latest-value/in-flight/generation 逻辑**保持原样**；Metal runtime 只负责
  资源与编码，**不新建第二个 scheduler**，也不引入 DAG。
- 依赖选择（需主 Agent 决策）：
  - 方案 A（推荐）：`[target.'cfg(target_os="macos")'.dependencies]`
    引入 `objc2` + `objc2-metal` 0.3.2 + `objc2-foundation`。它们已在
    `Cargo.lock`（GPUI 使用），所有权用 `Retained`，可避免第 6 节的
    Create-rule 泄漏。代价是 engine 新平台依赖。
  - 方案 B：把 `crates/portrait/src/metal.rs` 的 `Lib/Objc/Pool` 提升为
    共享 `metal/ffi.rs`（并修泄漏），portrait host 之后迁移到新 runtime。
    代价是跨 crate 重构 + Mac-F 协调。
  - 无论哪种，长期只保留一套 host；portrait 的测试 host 可以在迁移期保留，
    但不得再新增第二套。

---

## 6. FFI / 所有权风险（带 file:line）

| 级别 | 风险 | 位置 | 说明 / 建议 |
|---|---|---|---|
| 高 | `MTLCreateSystemDefaultDevice` 被 `Objc::retain` 后 Drop 只 release 一次 → 净泄漏 +1 | `metal.rs:686-696, 704-705, 1753-1768` | 该函数是 Create 规则，返回 +1（`objc2-metal-0.3.2/src/generated/MTLDevice.rs:47-55` 用 `Retained::from_raw` 表达）。应改用 `Objc::adopt`（拥有 +1），而不是 retain。`available()` 每次探测也泄漏。 |
| 高 | `device_pointer()` 是安全函数却返回裸 `*mut c_void`，无生命周期/所有权说明 | `metal.rs:1445-1447` | 调用方可随意 retain/release 或跨线程使用。建议删除或改成 `unsafe` + 明确“borrow for duration of context”。 |
| 高 | 逐 kernel `waitUntilCompleted` 是 buffer/command buffer 生命周期的前提 | `metal.rs:1394-1443` | 一旦改成批处理/异步，`Objc` 在 commit 后立即 drop encoder/cb；buffer 也可能被 pool 复用。必须加 completion handler/in-flight ring 后再放开 wait。 |
| 中 | 每 kernel 独立 command buffer，mesh 两遍两次 wait；无 fence/event | `metal.rs:1400-1443, 935-940` | 直接阻碍连续子图；性能上每次 dispatch 都有 CPU/GPU 往返。 |
| 中 | `unsafe impl Send/Sync for Lib`（含裸 `send` 代码指针） | `metal.rs:1706-1718` | 当前成立（不可变函数指针 + 从不 dlclose）。若未来支持运行时切换/卸载框架，即不成立；需注释约束。 |
| 中 | `MetalContext` 线程绑定，但 `execute_*` 每线程各建 device/queue/pipeline | `metal.rs:599-602, 1482-1498` | 引擎在 background executor/rayon 线程渲染（`app/src/state/render.rs:973-1003`），可能产生多个 device 与重复编译。建议全局单 runtime + `Retained`（Send+Sync），而不是 thread_local。 |
| 中 | 选择器/ABI 全靠 `transmute`，没有类型检查 | `metal.rs:1842-1974` | 错误签名是 UB。`send_void_bool` 用 `i8` 传 BOOL（`:1882-1885`），arm64 上 BOOL 为 bool，实践可用但属平台假设。优先用 objc2-metal 的类型化绑定。 |
| 中 | 所有 buffer `options:0`（Shared），读回直接读 `contents` | `metal.rs:1348-1382, 1605-1627` | 仅对 Shared + 已完成 GPU 写入安全。Private 中间资源必须 blit 到 staging；新代码不要复用 `read_bytes` 读 Private。 |
| 中 | `as_bytes<T>` 把任意 `T` 当 `&[u8]` | `metal.rs:1562-1564` | 依赖所有 uniform 都是 `repr(C)`（当前如此）。新结构体必须加 `repr(C)` 与 size assert。 |
| 低 | autorelease pool 每次调用新建；adopted/retained 对象可跨 pool 存活 | `metal.rs:730, 783, 856, 1771-1789` | 当前正确。批处理路径应显式每帧/每批 push/pop，避免长帧累积 autoreleased 对象。 |
| 低 | `error` 对象是 +0 autoreleased，只在 pool 内读取 | `metal.rs:1436-1440, 1687-1704` | 当前正确；不要把它存到 pool 外。 |
| 低 | MSL fast-math 必须保持关闭 | `metal.rs:1640-1642`；各 `.metal:5-6` | 任何新 pipeline cache 若忘记 `setFastMathEnabled:false` 会改变 FP32 舍入；应把该选项固化在 library key 里。 |
| 引擎侧 | `unsafe impl Send for FrameHandle/SlotBuffer/SendStream` 依赖 stage `Mutex` 与 driver 表生命周期 | `gpu_runtime.rs:213-217, 653-655, 700-701` | Metal 用 `Retained<MTLBuffer>` 时天然 Send/Sync；若写裸 FFI wrapper，必须给出同样的生命周期证明，不能只加 `unsafe impl Send`。 |
| 引擎侧 | `DeviceBuffer<'_>` → `'static` 的 transmute | `gpu_runtime.rs:664-666, 815-817, 1081-1083` | Metal 类比：MTLBuffer 会 retain device，风险较低；但 pool/queue 必须先于 buffer 存活。 |
| 引擎侧 | `generation` 不校验 | `gpu_runtime.rs:632`；`clarity_gpu.rs:131, 196` | 旧帧驻留在 resize/切图后可能被复用；Metal residency 必须比较并推进。 |

---

## 7. 本 M2 今天能验证什么

> 本节是“可运行的计划”，不是已执行结果。审计未运行 cargo。

### 7.1 可直接运行（需要 Mac-A 的编译基线已修好）

```sh
cargo test -p aurora-portrait \
  --test metal_runtime --test metal_mesh_runtime \
  --test metal_smile --test metal_skin_assets \
  -- --ignored --test-threads=1
```

注意：
- 必须用 `--test <binary>`；`cargo test ... metal_` 是按**测试函数名**过滤，
  这些测试函数名并不以 `metal_` 开头，可能匹配到 0 个。
- 所有 GPU 测试都是 `#[ignore]`，默认只编译不跑（`tests/metal_runtime.rs:1-8` 等）。
- `require_macos()` 只检查 `target_os`，不检查 `available()`；没有 device 时
  会 panic/失败，不会跳过（`metal_runtime.rs:19-23`）。
- `metal_runtime` 第一个测试显式 `assert!(available())`（`:60`）。

覆盖范围：
| 测试 | 覆盖 | 依赖 |
|---|---|---|
| `metal_runtime` | P15 composite vs `composite_pixel`（非方 ROI、宽 stride）、1×1 box blur、O3B whiten 几何/gate、mask projection expected plane | 合成输入，无外部文件 |
| `metal_mesh_runtime` | affine select/emit、overlap/cull、perspective、serial fallback、readback contract、输入拒绝 | 合成输入 + `raster` CPU golden |
| `metal_smile` | 4 个 golden case GPU vs CPU `pf100007_apply` | `tests/fixtures/metal_smile/`（32 个文件，已 tracked） |
| `metal_skin_assets` | 2 个 atlas（whiten/ruddy）、多个 case 的 GPU vs `makeup` | `tests/fixtures/metal_skin/`（33 个文件，已 tracked） |

**当前会被卡住的一项**：`tests/metal_asset.rs` 的
`lut_asset_is_rgba_float_and_o3b_abi_is_rgba` 在 macOS 上默认**不 ignore**
（`#[cfg_attr(not(target_os="macos"), ignore)]`，`:109-111`），会 `expect` 读取
`work/backend-r11/o2e/whiten_lut.f32`（`:9-13, 21-28`）。该文件在本 worktree
**不存在**（`ls` 失败）。因此默认
`cargo test -p aurora-portrait` 在本机会失败；`--test` 列表里应暂时排除
`metal_asset`，或由资产所有者补齐/改为合成 LUT。

另外未验证：这些 MSL 在 Xcode 27.2 / Metal 4 工具链下是否全部编译通过；
`available()` 在本 M2 是否 true。均需实跑一次才能标“通过”。

### 7.2 需要新 fixture / 新测试

- **Develop 点链**：需要 deterministic 合成线性 RGB（含 <0、>1、NaN 边界）+
  CPU `develop_display` 对拍；LUT/curve 由 Rust CPU 表生成，不用私有资产。
- **Clarity precurve/structure**：现有 `clarity_gpu` 的 Windows 测试是资产/对拍参考，
  需把输入改成合成 grid，并定义 P50/最大误差。
- **NR/USM**：需要合成噪声/边缘图 + 现有 CPU `detail` 阶段；注意现有 CUDA PTX 的
  luma/colour 法则与 C1 oracle 不完全相同（`detail.rs:500-545`），Metal 必须以
  **当前 CPU reference** 为准，不能移植过时的 8-bit 近似。
- **Wide/16-bit**：`rgba16` 路径目前只在 CPU（`process.rs:2574`），需新增
  RGBA16 对拍与格式测试。
- **ROI/halo/tile**：每个新 kernel 都要非方、padded stride、ROI 越界测试
  （现有 P15/whiten 测试是模板）。
- **显示 boundary**：GPUI headless renderer 只在 `bench-support`/`test-support`
  feature 下存在（`metal_renderer.rs:1889-1903`），本 app 未启用；需要单独 harness
  或给 app 加 dev-only feature。
- **性能**：冷/暖帧、P50/P95、GPU time（`MTLCommandBuffer.gpuStartTime/gpuEndTime`）、
  CPU wait、上传/下载字节。当前 `app/src/perf.rs` 的 `gpu_*` 字段全 null
  （`:84-97`），macOS RAM 统计也是 None（`:172-180`）。

### 7.3 fixture 来源风险（unknown，交所有者裁决）

`tests/fixtures/metal_smile/manifest.json:4-8` 与
`tests/fixtures/metal_skin/manifest.json:3-5` 声明的 oracle 路径是
`crates/pixcake/src/kernels/...`，本仓库不存在该路径（crate 已改名/合并）。
`metal_skin` 含 `whiten.rgb8`、`ruddy.rgb8` 两个 512×512×3 atlas。按 `AGENTS.md`
“第三方模型/预设/资产不得进入仓库”的规则，这些图集的来源需要所有者确认；
本审计不判定合规性，也不修改它们。

---

## 8. Apple Silicon 事实在本实现的落点

- **统一内存 ≠ 零复制**：`newBufferWithBytes` 仍会把 host 数据复制进 MTLBuffer
  （`metal.rs:1352-1360`）。`StorageModeShared` 只是让 CPU 能直接看到同一块内存，
  不消除 `Vec<u8>` → MTLBuffer 的拷贝。
- **Shared vs Private**：上传、uniform、最终 readback 用 Shared；
  多阶段中间纹理/大 scratch 用 Private + blit/staging（spec 4.2）。当前 host
  全部 Shared、零 texture，属于“能跑但最简单”的形态。
- **一个 queue**：GPUI 自己已有一个 queue（`metal_renderer.rs:397`）；
  引擎 runtime 应只建一个自己的 queue，不按 kernel/线程复制。两者（GPUI vs engine）
  在 M2 上是不同 MTLCommandQueue 指向同一 device；跨 queue 资源交接需要
  `MTLSharedEvent` 或 wait，不能假设自动同步。
- **不要每 kernel `waitUntilCompleted`**：当前 host 正好相反
  （`metal.rs:1435`）；这是 M2 最需要改的行为。等待应只发生在
  `read_into`、display handoff、或下一帧需要上一帧 scratch 时。
- **`recommendedMaxWorkingSetSize` 不是额外 VRAM**：它是 GPU 工作集的建议上限；
  统一内存下不能把它与进程 RAM 相加后各自分配。当前仓库没有任何读取；
  新 runtime 应把它作为**软上限**并入既有 `VramLedger`，而不是新预算。
- **FP16**：现有 kernel 全 FP32；spec 要求只有逐算法对拍且有收益才启用。
  不应在 runtime 层全局强制 half。

---

## 9. 待所有者/主 Agent 决策

1. **依赖**：engine 引入 `objc2-metal`（推荐，所有权正确）还是提升 portrait 的
   手写 FFI 为共享模块？后者需要 Mac-F 协调并先修 `MTLDevice` 泄漏。
2. **`gpu_runtime` 归属**：它是 Mac-A 文件；把它改为后端无关需要 Mac-A 配合。
   Mac-B 不应直接改。
3. **GPUI 外部纹理**：是否接受为 BGRA8/IOSurface 增加最小 patch？不 patch 就只能
   bounded staging（不是零复制）。这是显示侧唯一能突破 2 次拷贝的路径。
4. **资产**：`metal_skin`/`metal_smile` fixture 与 `whiten_lut.f32` 的来源/合规性。
5. **首个落地 kernel**：建议 `o3_masked_lut_whiten`（有产品 CPU oracle
   `skin_color::masked_lut_blend`、自包含 fixture、u8 输入输出），用它验证
   device/queue/pipeline/pool/readback；随后再接 develop 点链。
