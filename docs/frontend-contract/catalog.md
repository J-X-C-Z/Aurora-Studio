# Lumen Studio 效果目录 — 给 Opus 的前端生成依据

生成时间 2026-11-11，基于 `cargo test` 通过的 281 个测试的代码现状。

## 文件说明
| 文件 | 内容 |
|---|---|
| `params.json` | C1 原生 72 个滑块参数（min/max/step/default） |
| `portrait-algorithms.json` | 人像算法参数：285 个已识别参数的定义、分类、完成状态 |
| `catalog.md` | 本文件：状态汇总 + 前端使用说明 |

## C1 原生功能分类

### 基础调整（ready — 能调、能实时预览、能导出）
| 组 | 滑块数 | 说明 |
|---|---|---|
| Exposure | 4 | 曝光/对比/亮度/饱和 |
| WhiteBalance | 2 | 黑体近似 |
| HighDynamicRange | 4 | 高光/阴影/白场/黑场 |
| Levels | 5 | RGB + 分通道 |
| Clarity | 2 | 清晰度/结构 |
| Dehaze | 1 | 去雾 |
| BlackWhite | 10 | 六滤镜 + 分离色调 |
| Sharpening | 4 | 已接线 |
| Noise | 5 | 亮度降噪 |
| FilmGrain | 2 | 颗粒 |
| Vignetting | 1 | 暗角 |
| Rotation | 1 | 旋转 |

### 光学与几何（partial — 能用但有已知缺口）
| 组 | 滑块数 | 缺口 |
|---|---|---|
| ColorEditor | 4 | 18 字段只用 5 个 |
| Moire | 2 | 3×3 低通，无频域 |
| LensCorrection | 3 | 畸变/色差方向是假设 |
| PurpleFringing | 1 | 同上 |
| Keystone | 4 | 方向假设，未对拍 |
| LocalAdjustments | 1 | Copy 形状 0 权重 |

### 非滑块功能
| 功能 | 状态 | 接口 |
|---|---|---|
| 曲线（5 通道） | ready | `curves` |
| 色彩平衡四向轮 | ready | `color_balance` |
| 裁剪/翻转/比例 | ready | `geometry` |
| 图层与蒙版 | ready | `layers` |
| 污点修复 | partial | `spots` |
| RAW 解码（Linux） | ready | libraw |
| 胶片曲线（4166 条） | ready | `base.curve` |
| 预设/样式（127 个） | ready | `FileEngine::styles()` |
| C1 Catalog 导入 | partial | 集合+WAL |
| 导出 JPEG/TIFF/PNG | ready | 配方/ICC/EXIF/水印 |
| 导出 JXL/HEIF/DNG/PSD | missing | 回退 JPEG |
| 元数据策略 | partial | 两档 |
| AI 蒙版 | missing | 后端未接 |
| 联机拍摄 | missing | 后端未接 |
| Session .cos / XMP 写回 / .P1X | missing | 后端未接 |

## 人像算法分类（portrait-algorithms.json）

285 个参数按 ID 段分类，来源：产品 UI 注册表 + 效果配置。完成状态分三档：
- **已实现**：后端算法已接入渲染管线，前端可正常做交互
- **已解析未接入**：参数定义和效果映射完整，后端算法还没写
- **无法解析**：只有低置信度家族推断，不建议做 UI

| ID 段 | 分类 | 参数数 | 状态 |
|---|---|---|---|
| 0–999 | 瑕疵去除 | 14 | 已解析未接入 |
| 21000–21999 | 面部重塑 | 58 | 已解析未接入 |
| 30000–39999 | 皮肤/妆容/身体 | 78 | 已解析未接入 |
| 40000–49999 | 妆容细节 | 18 | 已解析未接入 |
| 80000–89999 | 智能识别 | 3 | 已解析未接入 |
| 90000–90999 | 素材/样式/天空替换 | 93 | 已解析未接入 |
| 100000+ | 素材定位/边缘调整 | 15 | 已解析未接入 |
| 900000+ | 局部去瑕疵 | 6 | 已解析未接入 |

其中 7 个参数被多个效果共用（`shared_effect_param`），30 个无法解析（仅家族假设）。
具体逐条状态、置信度和效果映射读 `portrait-algorithms.json`。

## 前端做 UI 的规则
- **C1 原生 ready**：正常做交互，参数读 `params.json`。
- **C1 原生 partial**：做 UI，注明有缺口。
- **C1 原生 missing**：灰显或不做，等后端。
- **人像算法"已解析未接入"**：参数和映射都在 `portrait-algorithms.json`，UI 可以画，但不要承诺精度；算法后端还没实现。
- **人像算法"无法解析"**：不要做 UI。
EOF
