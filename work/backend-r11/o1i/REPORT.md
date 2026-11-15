# O1i PF108-110 牙齿美白真 NVIDIA 执行

## 结论

`PASS_REAL_GPU_TEETH`。whiten→bright 两段在真实 RTX 4070 Ti 上跑通，与 `face_detail::teeth_whitening` **逐字节相等**：1x1 / 17x13 / 32x24 × 5 组 UI，`differing = 0`，alpha 逐像素等于源。512x512 会话路径中位 **0.739 ms** vs CPU **63.9 ms**。

## 实现

- 一个 kernel 内完成两段：先 whiten 表、再 bright 表，中间值以 `store_unorm8` 落盘后再进入下一段（与 CPU 一致，不是 f32 直连）。
- `mask.g` 门：`mask = clamp01(mask_g/255)`，`weight = store_unorm8(mask*dz)/255`、`store_unorm8(mask*ea)/255`；两者都 `<= 0.001` 的像素直接复制源。
- `u_degree = 1`：采样端的混合是 `src*(1-degree) + sampled*degree`，degree 取每像素权重，等价于 shader 里 `u_degree(=1) * maskWeight`。
- `store_unorm8` 不是四舍五入，而是**分数对 256 项阈值表比较**（`fraction >= TEETH_STORE_THRESHOLDS[index]`）。阈值表按 CPU 源码逐位生成，在 GPU 上用 `double` 比较，避免 f32 舍入翻转。
- 32 字节 ABI（`width/height/stride/mask_row_bytes/dz/ea/reserved0/reserved1`），常量断言钉住；`teeth_lut.h` 有同名 `static_assert`。
- 复用 `DeviceSet` + `launch_raw`，无自建 driver；`GPU_LOCK` 未动。

## 资产

`assets/portrait/teeth/{Whiten,Bright}.png` 用与 `processing::portrait_assets::load_lut_png` 完全相同的代码路径解码（`PngDecoder` → `DynamicImage::from_decoder` → `into_rgb8`），导出到 `o1i/whiten.rgb`、`o1i/bright.rgb`（各 786,432 B），测试用 `include_bytes!` 读取。生成器在 `o1i/decode/`。

## 真 GPU 覆盖

| 用例 | 内容 |
| --- | --- |
| UI 组合 | (100,100,100)、(50,70,60)、(100,0,100)、(100,100,0)、(30,30,30) |
| 几何 | 1x1、17x13、32x24 |
| 单段路径 | `deyellow=0`（只有 bright）、`bright=0`（只有 whiten） |
| 门限 | `common=0` → 0 次 launch 且恒等；`mask=0` → 1 次 launch 但逐字节恒等；`mask=1` → 权重 1/255 仍越过门限，结果与 CPU 一致 |
| alpha | 每个用例逐像素断言等于源 |
| 拒绝 | mask 长度、whiten 长度、`common_ui` NaN、`bright_ui=101` |

## 验证

- `work/backend-r11/o1i/check.rs`（挂真实源码 + 测试文件）：4 个 `teeth_gpu::*` 全绿。
- 回归：`PIXCAKE_CUDA_REQUIRED=1 cargo test -p aurora-portrait` 的 lib 273 + fragment 2(1 ignored) + gloss 5 + lut 4 + mesh 7 + roi 7 全绿；`cargo build` 无警告。

## 计时（本路）

512x512、100/100/100、grid 32x32、block 16x16：cold 0.9 ms、中位 **0.739 ms**（20 次），CPU oracle 63.9 ms。会话内 **5 次分配**（源/掩码/两张 LUT/输出），每次 apply 0 次分配——两张 786KB LUT 只上传一次。

## 需要主控做的

`crates/portrait/src/lib.rs` 加 `pub mod cuda_teeth;`。**在此之前 `cargo test` 会因 `tests/cuda_teeth.rs` 找不到模块而红**（本轮用 o1i harness 验证）。O1h/O1g 的测试不受影响。

## 边界

无 UI、无 C1、无安装、无生产写入。未接真实牙齿 ROI/掩码来源；测试掩码为合成数据。
