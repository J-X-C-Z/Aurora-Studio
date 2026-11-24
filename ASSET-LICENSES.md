# 资产与许可清单（草案）

> 状态：**草案，未生效**。许可证方案仍待用户决策（`REVIEW-TRIAGE-20261124.md` 的 D-1），
> 本文件不代表最终许可声明，也未改动 `Cargo.toml`、`LICENSE`、`AGENTS.md`、
> `CONTRIBUTING.md`、`CHANGELOG.md`。C1 相关资产已合并
> `06_reports/C1-ASSET-INVENTORY-20261124.tsv`（81 行，2026-11-24 13:46）。
>
> 起草：integration 会话，2026-11-24。依据：审查条目 F-04、C1-1、IN-1/IN-2 派单。

## 1. 当前仓库声明（事实，未改动）

| 位置 | 现行内容 |
|---|---|
| `Cargo.toml` `[workspace.package]` | `license = "MIT"` |
| `LICENSE` | MIT 全文，`Copyright (c) 2026 J-X-C-Z` |
| `AGENTS.md:23` | “许可证保持 `Proprietary`。不要添加或切换到开源许可证。” |
| `CONTRIBUTING.md:8` | “许可证保持 `Proprietary`。贡献不代表获得再分发权，不得擅自改成开源许可。” |
| `CHANGELOG.md:22` | “Workspace license metadata matches the MIT `LICENSE`.” |
| `cargo metadata`（三个 package） | `aurora-engine 0.1.0`、`aurora-portrait 0.1.0`、`portrait-app 0.1.0` 均为 `license = MIT` |

结论：仓库内 **MIT 声明与 Proprietary 要求并存**，属审查条目 F-04，需 D-1 拍板后统一。

## 2. 自研代码

`crates/engine/**`、`crates/app/**`、`crates/portrait/**` 为自研 Rust 源码。版权与许可
以 D-1 选定的方案为准；本文件不预设结论。

## 3. 第三方代码依赖

- 依赖清单以 `Cargo.lock` 为准；`cargo metadata --locked --no-deps` 可复现三个本地 package。
- `cargo tree -d` 显示的重复版本（节选）：`base64 0.22.1/0.23.1`、`usvg 0.45.1/0.46.0`、
  `bitflags 1.3.2/2.x` 等，来源为 GPUI 组件链与自身依赖，不是资产许可问题。
- 后续由 IN-2 的 CI 补强加入 `cargo deny`（licenses/bans）与 `cargo audit`，输出到
  `06_reports/`，本草案不代替机器检查。

## 4. 运行时资产（随二进制编译或随程序加载）

### 4.1 `assets/portrait/`

| 资产 | 字节 | SHA256 | 来源 | 运行期必需 |
|---|---:|---|---|---|
| `radiance/native-lut-a.bin` | 76800 | `48E82EC7…6BB5D1` | 像素蛋糕 SkinRadiance PF90004 解密配置 | 是（`cuda_radiance.rs` `include_bytes!`） |
| `radiance/native-lut-b.bin` | 38400 | `B804EFCF…38AF1D` | 同上 | 是 |
| `radiance/native-gamma-rgba32f.bin` | 65536 | `91462755…C503BA` | 同上 | 是 |
| `radiance/native-color-matrices.bin` | 72 | `6436DE49…801D65` | 同上 | 是 |
| `skin-adjust/whiten.png` 等 6 张 LUT | 97292–173386 | 见 `skin-adjust/manifest.json` | 像素蛋糕 SkinAdjust 解密配置 | 是 |
| `smile/mean.f32le`、`triangles.u32le` 等 5 件 | 26472–51228 | 见 `smile/manifest.json` | 像素蛋糕 SmileMouth3D 解密配置；源路径写在 manifest 的 `source` 字段 | 是 |
| `teeth/Whiten.png`、`Bright.png` | 99654/196345 | 见 `teeth/manifest.json` | 像素蛋糕 TeethWhitenV3 解密配置 | 是（`face_parsing.rs` `include_bytes!`） |
| `suit/presets.json` | 329062 | `55B098E4…75677` | 像素蛋糕 suit 解密预设（24 presets / 355 passes） | 是（`portrait_makeup_bundle.rs` `include_str!`） |

### 4.2 `crates/portrait/assets/portrait/`

与根 `assets/portrait/` 高度重复（`smile`、`radiance`、`skin-adjust`、`teeth`、`suit`
按字节一致），另有根目录没有的 `face-reshape/`、`face-width/`、`forehead/` 三组
`parts.f32le`。`crates/engine/src/portrait_stage.rs` 与 `face_parsing.rs` 通过
`include_bytes!("../../portrait/assets/...")` 直接引用。**两份都是编译输入，不能只留一份。**

### 4.3 `crates/engine/data/`（C1 原厂提取，编译内嵌）

| 资产 | 字节 | SHA256 | 来源 | 运行期必需 |
|---|---:|---|---|---|
| `camera_params.json` | 134036 | `e8b7c11f…5365f3` | C1 16.8.6.3784 `ImgCoreDll.dll` + `coreConfig.cdx` 静态提取（23 台完整包） | 是（相机默认/CCM/噪声/曲线绑定） |
| `camera_params_all.json` | 2515158 | `8f1e70f1…c357cc` | 同上，714 台全量 | **否**（当前仅 `for_any_camera`/诊断用，仍占二进制体积） |

来源 DLL 指纹 `CA02B6AC…44A4`，与本机安装一致。

### 4.4 `crates/engine/assets/`

| 资产 | 字节 | SHA256 | 来源 | 运行期必需 |
|---|---:|---|---|---|
| `c1-session-template.cosessiondb` | 233472 | `e477ba98…26aa86` | Capture One 会话模板（待 C1-1 清单确认权利） | 是（`c1session_export.rs` `include_bytes!`） |
| `portrait/vface/*.f32le`、`*.u16le` | 2104–4208 | 见 §5 哈希附注 | 像素蛋糕 FaceSurgery V2 解密资产（`docs/portrait-assets/r26-pe-assets-catalog.json` 记录 `effect/config/FaceSurgery/V2/face/Grid_2005.png`） | 是 |
| `portrait/vface/atlas/Grid_2005.rgba8` 等 5 张 | 8448–174420 | 见 §5 哈希附注 | 同上 | 是 |

`crates/app/assets/aurora-theme.json`（3252 B）为自研界面主题，随包分发。

### 4.5 C1 相关清单（已合并 `06_reports/C1-ASSET-INVENTORY-20261124.tsv`）

该 TSV 共 81 行，分四类：

| 类别 | 行数 | 内容 | 再分发 |
|---|---:|---|---|
| A 编译内嵌数据 | 3 | `camera_params.json`、`camera_params_all.json`、`c1-session-template.cosessiondb` | 待用户确认 |
| B 源码常量 | 10 | `c1_model.rs`、`param_meta.rs`、`camera_defaults.rs`、`camera_curve.rs`、Z6III 预矩阵/噪声表、`c1_banding.rs`、`c1_wb_oracle.rs`、`styles.rs` | 多数待确认；`c1_wb_oracle.rs`/`styles.rs` 为 MIT 自研 |
| C 本地测试样本 | 61 | `crates/engine/testdata/**`（`crates/engine/testdata` 被 `.gitignore` 忽略，未随包） | 待用户确认；仅 `cargo test` |
| D 运行时外部依赖 | 7 | C1 安装目录的 `coreConfig.cdx`、`lensDatabase.cdx`、`ImgCoreDll.dll`、`Film Curves`、`Color Profiles`、`Styles`、`Presets60` | **不可再分发**（未获授权） |

要点：

- `camera_params_all.json`（2.5 MB）不是运行期必需，却以 `include_str!` 进入 `aurora-engine`。
  若 D-1 选择收紧，应改为 feature/构建期裁剪，而不是继续内嵌。改代码属 backend-opt。
- D 类 7 项目前只在用户本机读取，不在主树；发布包不得包含。
- C 类共 61 个本地样本（含 `.icm`、`.fcrv`、`.costyle`、`.copreset`、`c1-sample.cocatalogdb`、
  `nef-exif-head.bin`），全部被 git 忽略；其中 `nef-exif-head.bin` 可能含相机序列号，注意隐私。
- 只有 `TestHalfDown.fcrv`、`vignette-1ev-1500x1000.lcc` 标注为“自行构造”，可再分发。

## 5. 测试资产

| 位置 | 文件数 | 性质 | 权利状态 |
|---|---:|---|---|
| `crates/portrait/tests/fixtures/**` | 117（3.5 MB） | 合成/解码 fixture，含 `metal_skin`、`metal_smile` 的期望值 | 自研 fixture 与像素蛋糕解密 LUT 的派生数据混合；逐项来源见各 `manifest.json` / `SHA256SUMS` |
| `crates/engine/testdata/**` | 见 CI 规则 | 本地 C1 样本（CI 明确禁止入库） | 不得进入 release 包 |

`crates/portrait/tests/fixtures/pf21015/README.md` 明确写明 atlas 是测试 fixture，
运行期不需要；但 atlas 字节与 `assets/portrait/skin-adjust/TintGreen.png` 等相同，
权利随上游资产。

## 6. 研究材料与工具（不应进入 release 包）

| 位置 | 文件数 | 分类 | 说明 |
|---|---:|---|---|
| `crates/portrait/work/**` | 15 | research（**当前被 git 跟踪**） | backend-r11/o1i 基准、PTX 构建脚本、`bright.rgb`/`whiten.rgb` 中间帧 |
| `work/**`（仓库根） | 10 | research（**当前被 git 跟踪**） | 与 `crates/portrait/work/backend-r11/o1i` 重复 |
| `tools/body_pose_capture/**` | 12 | tool | PowerShell 抓取/分类脚本与 manifest（含厂商路径信息） |
| `docs/portrait-assets/**` | 9 | research | 资产获取记录、blocker、r26 catalog |
| `engine/**`（仓库根，未跟踪） | 23 | 历史副本 | 旧 `lumen-engine` crate 副本，工作区未跟踪，不参与 workspace |

## 7. 待决事项

1. **D-1 许可证**：MIT / Proprietary / 自有代码 Proprietary + 子资产单独授权，三选一。
2. **像素蛋糕派生资产**：当前运行时资产大量来自厂商解密配置。再分发权未确认，
   在 D-1 与法务判断前只能作为内部构建输入，不能对外发布。
3. **C1 资产**：见 §4.5。编译内嵌的 `camera_params*.json` 与
   `c1-session-template.cosessiondb` 权利待确认；C1 安装目录 7 类运行期依赖不得再分发。
4. **`vendor/libraw`**：本地 RAW 解码运行库，未纳入 git；发布包需单独说明来源与 LGPL/CDDL 义务。
5. **两份 `assets/portrait` 与两份 `work`**：去重、去跟踪属 IN-2 迁移建议，本轮不搬文件。

## 8. 哈希附注（vface 与 C1 模板）

```text
e477ba981a2b26219b1d58d8a10c2296b80d4683d9bed94f5a5247bb4026aa86  crates/engine/assets/c1-session-template.cosessiondb
5f102c6057311f67648174aa60f67971637e17a811c5d697bb74ec12ec72f30b  crates/engine/assets/portrait/vface/face_coeff526.f32le
707746096ccccaf8df2234daf6605cb6e335bf56e656051f7078e64bc4eb9fb8  crates/engine/assets/portrait/vface/topology2982.u16le
0f3eb71525714617c16d2d855d77fadc89d9290a9da12c63a97089e1866cc422  crates/engine/assets/portrait/vface/triangle_weights994.f32le
24734f9ca9aba22c86e571e085dc46122200f0d55e8951eada118d7a2f1c613d  crates/engine/assets/portrait/vface/uv526.f32le
8051d0ebb85b8007ac4190be28e9dcbdb671a480f589ff36d92a8a0aa1cf8548  crates/engine/assets/portrait/vface/atlas/Grid_2005.rgba8
ebeb1e9c9f610cbf131afb533a48530dd86eee5a080d62c3f7263d6b47ca7a0d  crates/engine/assets/portrait/vface/atlas/Grid_2010_Negative.rgba8
624fcc9067829e63c408d8ec1a28fc4f9c43efdeaba0b6580f82492f4ba877b6  crates/engine/assets/portrait/vface/atlas/Grid_2010_Positive.rgba8
67784eb87d05899b5dfc9032eca30be6221a661c92c1a9f673b9dc99dbc85cfa  crates/engine/assets/portrait/vface/atlas/Grid_2011_Negative.rgba8
a661ba7c7c47c83e0915c8176dde6f92bafcbbfddd1a6978991b949444380c4f  crates/engine/assets/portrait/vface/atlas/Grid_2011_Positive.rgba8
```
