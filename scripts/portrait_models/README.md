# Portrait Models & Prior/LUT Asset Pipeline (`scripts/portrait_models`)

本目录提供人像管线 **7 类基础先验/LUT 资产** 的可重复确定性生成、导出与 SHA-256 校验工程化骨架。

## 设计原则

1. **零伪造权重**：本框架仅生成基于色彩科学公式与参数化几何公式的**数学先验（Mathematical Priors）**，绝不生成随机或伪造的神经网络权重。
2. **明确资产边界**：在 `manifest.json` 中对每类资产显式标注 `asset_kind` 与 `requires_real_dataset_training`：
   - `prior_lut`：确定性色彩/曲线查找表（可直接作为无外部依赖的基础先验）。
   - `prior_geometry`：参数化 2D/3D 几何先验（可用作管线冒烟与基础形变，生产级效果需真实 3DMM / 人像关键点数据集拟合替换）。
   - `neural_weight_slot`：神经网络权重占位契约（**不含权重文件**，必须由真实人像数据集训练得到的 `.onnx` 权重替换）。
3. **可重复与可校验**：同一版本脚本在任意平台输出逐字节一致的资产，并生成 `manifest.json` 与 `SHA256SUMS` 供 CI 与运行时校验。

## 7 类资产清单

| # | 类别 ID | 输出文件 | 资产性质 (`asset_kind`) | 是否需真实人像数据集训练替换 | 说明 |
|---|---|---|---|---|---|
| 1 | `skin_adjust` | `whiten.png`, `ruddy_new.png`, `TintGreen.png`, `TintMagenta.png`, `cold.png`, `warm.png` | `prior_lut` | 否（可后续用专业调色数据集精调） | 512×512 Hald 3D LUT（64³），对应 `crates/portrait/src/kernels/skin_adjust.rs` |
| 2 | `radiance` | `prior-lut-a.bin`, `prior-lut-b.bin`, `prior-gamma-rgba32f.bin`, `prior-color-matrices.bin` | `prior_lut` | 否 | 透亮高光压缩曲线、sRGB/Linear Gamma LUT 与 Rec.709 矩阵 |
| 3 | `teeth` | `Whiten.png`, `Bright.png`, `toothMask.png` | `prior_lut` / `prior_geometry` | `toothMask` 运行时由分割模型替代 | 512×512 去黄美白/提亮 3D LUT 与 111×43 椭圆齿弓先验掩膜 |
| 4 | `lip_luts` | `blend_weight_lut.png`, `level_weight_lut.png` | `prior_lut` | 否 | 256×2 RGBA8 唇色混合权重与明度保持曲线，对齐 `lip_color.rs` 契约 |
| 5 | `smile` | `mean.f32le`, `normal.f32le`, `side_left.f32le`, `side_right.f32le`, `triangles.u32le` | `prior_geometry` | **是（需真实 3DMM 数据集拟合替换）** | 参数化微笑嘴角形变基向量与三角拓扑 |
| 6 | `face_mesh_priors` | `face_reshape_parts.f32le`, `face_width_parts.f32le`, `forehead_parts.f32le` | `prior_geometry` | **是（需真实人像网格数据集校准）** | 瘦脸/脸宽/发际线控制点位移先验 |
| 7 | `neural_weight_slots` | `neural_slots_contract.json` | `neural_weight_slot` | **是（无权重，必须训练/提供真实 ONNX）** | 人脸检测/网格/分割/祛痘/去眼袋/姿态模型输入输出与 SHA-256 校验契约 |

## 使用与测试方法

### 方式 A：Rust 原生工具（无额外依赖，Windows/macOS/Linux 通用）

```powershell
# 1. 编译（产物放入 build-cache，不污染仓库）
rustc --edition 2021 -O scripts/portrait_models/portrait_priors.rs -o "C:\WorkSpace\Aurora Studio\90_work\build-cache\portrait_priors.exe"

# 2. 运行内置单元测试（含 SHA-256 标准向量、PNG/BIN 确定性、篡改检测）
& "C:\WorkSpace\Aurora Studio\90_work\build-cache\portrait_priors.exe" --test

# 3. 生成 7 类资产到指定输出目录
& "C:\WorkSpace\Aurora Studio\90_work\build-cache\portrait_priors.exe" --out "C:\WorkSpace\Aurora Studio\90_work\build-cache\portrait-priors-out"

# 4. 校验已生成资产的字节数与 SHA-256
& "C:\WorkSpace\Aurora Studio\90_work\build-cache\portrait_priors.exe" --verify "C:\WorkSpace\Aurora Studio\90_work\build-cache\portrait-priors-out"
```

### 方式 B：Python 3 脚本（仅使用标准库，零第三方依赖）

```bash
python3 scripts/portrait_models/generate_and_verify.py --out ./build/portrait-priors-out
python3 scripts/portrait_models/generate_and_verify.py --verify ./build/portrait-priors-out
```
