# 画质基准 harness（quality_bench）

`tests/support/quality_metrics.rs` 提供共享指标实现，`tests/quality_bench.rs`
是默认运行的合成自测，`examples/quality_bench.rs` 是读取外部 manifest 的 CLI。

默认 `cargo test` 只跑合成数据，不读外部文件、不联网。真实 RAW / 真实照片检查
位于 `external-fixtures` feature 与 `#[ignore]` 之后，外部样本不进仓库。

## 运行

```powershell
# 合成自测（默认，必须全绿）
cargo test -p aurora-engine --test quality_bench --locked -- --test-threads=1

# 确认外部 feature 可以编译
cargo check -p aurora-engine --features external-fixtures --tests --locked
cargo check -p aurora-engine --features external-fixtures --example quality_bench --locked

# 外部 CLI
cargo run -p aurora-engine --features external-fixtures --example quality_bench -- `
  --manifest C:\fixtures\quality\manifest.json --out C:\fixtures\quality\report.json

# 真实 RAW 冷解码/缓存一致性（需要 manifest 里至少一个 raw 字段）
$env:AURORA_QUALITY_FIXTURES = "C:\fixtures\quality\manifest.json"
cargo test -p aurora-engine --features external-fixtures --locked `
  --test quality_bench -- --ignored --test-threads=1
```

未启用 `external-fixtures` 时，示例仍可编译，但会打印提示并以退出码 2 结束。

## Manifest 格式

顶层字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `samples` | 数组 | 必填，逐样本配置 |
| `colorchecker_threshold` | number | 可选，ΔE2000 阈值，默认 `2.0` |
| `false_color_chroma` | number | 可选，false color chroma 阈值，默认 `8.0` |

每个 sample 的字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `name` | string | 必填，报告里的样本名 |
| `image` | string | 必填，用于 ColorChecker / SNR / MTF / false color 的图像绝对路径（PNG/JPEG/TIFF/WebP） |
| `reference` | string | 可选，CPU 参考图绝对路径 |
| `candidate` | string | 可选，OpenCL/候选图绝对路径；与 `reference` 必须同时出现 |
| `colorchecker` | object | 可选，见下 |
| `shadow_roi` | `[x, y, w, h]` | 可选，暗部 SNR 区域（像素坐标） |
| `edge_roi` | `[x, y, w, h]` | 可选，MTF50 斜边区域（像素坐标） |
| `neutral_roi` | `[x, y, w, h]` | 可选，false color 中性区（像素坐标） |
| `raw` | string | 可选，真实 RAW 绝对路径，用于冷解码一致性测试 |
| `raw_long_side` | number | 可选，传给 `raw_cache::get_or_decode` 的长边，默认 `2560` |

`colorchecker` 对象：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `patches` | `[[x, y], ...]` | 必填，每个 patch 的中心像素坐标 |
| `expected` | `[[L, a, b], ...]` | 可选，与 `patches` 等长的期望 Lab(D65) |
| `patch_radius` | number | 可选，中位数采样方块的半边长，默认 `4` 像素 |

示例（占位绝对路径）：

```json
{
  "colorchecker_threshold": 2.0,
  "false_color_chroma": 8.0,
  "samples": [
    {
      "name": "colorchecker-daylight",
      "image": "C:\\fixtures\\quality\\colorchecker.png",
      "colorchecker": {
        "patches": [[410, 300], [510, 300], [610, 300]],
        "expected": [[96.539, -0.425, 1.186], [81.257, -0.638, -0.335], [66.766, -0.734, -0.504]],
        "patch_radius": 6
      },
      "neutral_roi": [300, 700, 120, 120]
    },
    {
      "name": "shadow-noise",
      "image": "C:\\fixtures\\quality\\shadow.png",
      "shadow_roi": [512, 512, 256, 256]
    },
    {
      "name": "mtf-slanted-edge",
      "image": "C:\\fixtures\\quality\\edge.png",
      "edge_roi": [200, 200, 128, 128]
    },
    {
      "name": "opencl-vs-cpu",
      "image": "C:\\fixtures\\quality\\candidate16.png",
      "reference": "C:\\fixtures\\quality\\reference16.png",
      "candidate": "C:\\fixtures\\quality\\candidate16.png"
    },
    {
      "name": "raw-cache",
      "image": "C:\\fixtures\\quality\\raw-preview.png",
      "raw": "C:\\fixtures\\quality\\capture.NEF",
      "raw_long_side": 2560
    }
  ]
}
```

ROI 必须完全位于图像内，越界会直接报错而不是静默裁剪，避免把坐标写错当成通过。

## 指标定义与阈值

1. **ColorChecker ΔE2000**
   - sRGB(EOTF) → 线性 RGB → XYZ(D65, IEC 61966-2-1 矩阵) → Lab（白点
     `X=0.95047, Y=1.0, Z=1.08883`）。
   - 每个 patch 在中心 ±`patch_radius` 的方块内按通道取中位数，再转 Lab。
   - CIEDE2000 取 `kL=kC=kH=1`；Sharma 标准测试对 (2.0425 / 2.8615 / 3.4412 /
     1.0000 等) 在合成测试中断言。
   - 输出每个 patch 的 ΔE、平均 ΔE、最大 ΔE；`expected` 缺省时只报告测量值。
   - CLI 的通过阈值取 `colorchecker_threshold`，默认 2.0。

2. **暗部 ROI SNR**
   - 每像素先对 sRGB 通道做 EOTF 线性化，再算 Rec.709 亮度
     `Y = 0.2126R + 0.7152G + 0.0722B`。
   - `SNR = 20 * log10(mean / std)`，`std` 为总体标准差。
   - ROI 完全平坦（`std = 0`）时 SNR 记为 `null`。
   - 合成自测：线性均值 0.18、σ=0.02，σ 相对误差 < 5%。

3. **MTF50（ISO 12233 风格斜边）**
   - 在 ROI 内用中心差分求梯度，累加结构张量
     `[[Σgx², Σgxgy], [Σgxgy, Σgy²]]`，最大特征向量方向作为边缘法向。
   - 像素投影到法向轴，按 `0.25 px` 分箱并求均值得 ESF；中间空洞线性插值。
   - ESF 中心差分得 LSF，以 LSF 峰值为中心加 Hamming 窗。
   - 朴素 DFT 求单边幅值谱并归一化到 MTF(0)=1，线性插值找首次 0.5 交点。
   - 频率单位 cycles/px；分箱 0.25 px 对应 Nyquist 2 c/px。
   - 合成自测：高斯模糊直边 σ=1.0 px 理论 MTF50≈0.1874 c/px，σ=0.5 px
     理论≈0.3748 c/px，容差 ±12%。

4. **False color 比例**
   - 每像素转 Lab(D65)，`chroma = sqrt(a² + b²)`。
   - 输出 ROI 内 `chroma > threshold` 的像素比例，以及 P95 / P99 / max。
   - 纯灰（R=G=B，任意亮度）的 a、b 必须为 0，合成自测要求比例严格为 0。

5. **CPU vs OpenCL 差异**
   - `reference` 与 `candidate` 必须同尺寸。同 bit depth 直接比较；RGBA8 与
     RGBA16 混用时把 8-bit 侧乘 257 升到 16-bit 再比较。
   - 绝对差统一换算到 8-bit level（0 ..= 255）后输出 max / mean / P99，以及
     严格 `> 1/255`、`> 2/255` 的通道样本比例。P99 由 0.25 level 直方图得到，
     内存不随图幅增长。
   - 支持 RGBA8 与 RGBA16（PNG 16-bit 会被 `image` 解码为 16-bit）。

6. **RAW 冷解码 vs 缓存命中**
   - 使用公开 API `aurora_engine::raw_cache::get_or_decode(path, dir, long_side)`。
   - 同一目录第一次调用是冷解码并写缓存，第二次是缓存命中；断言两次
     `RgbaImage.pixels` 完全相等（当前公开类型暴露 `pixels`，没有 `as_raw()`）。
   - 再用另一个空缓存目录做第二次冷解码，断言两次冷解码逐像素一致。
   - 该测试需要 manifest 中至少一个 `raw` 字段，并设置
     `AURORA_QUALITY_FIXTURES`；属于 `external-fixtures` + `#[ignore]`。

## 外部样本约定

- 样本、manifest、报告都不进仓库；`.gitignore` 已排除 `testdata/`。
- manifest 只写本机绝对路径，便于在不同机器上复用同一份样本目录。
- 未提供真实样本时不得把外部测试描述为“已通过”。
