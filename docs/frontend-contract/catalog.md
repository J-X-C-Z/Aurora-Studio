# Lumen Studio 效果目录 — 给 Opus 的前端生成依据

生成时间 2026-11-11，基于 `cargo test` 通过的 281 个测试的代码现状。

2026-11-24 更新：人像效果以 `portrait-capability-registry-20261124.tsv` 的七级证据为准；
本目录内的字节副本与 `03_algorithms/reports/PORTRAIT-CAPABILITY-REGISTRY-20261124.tsv`
一致，SHA256 `d1b92c6f0b1d15b2807f917c1298c5cbac4f72449c38e9f4e6af92a28fed85c4`（86,739 字节，2026-11-25 同步），
共 147 个 effect、6 项 test_gap。七级依次为 ev1 内核存在、ev2 CPU 参考、ev3 GPU 内核、
ev4 同 job 输入、ev5 应用主预览接入、ev6 原厂同图对拍、ev7 真实照片验收。
只有 ev5 有证据才允许在界面标为可用；ev6/ev7 当前都是 0 项有证据。

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
| LocalAdjustments | 1 | Copy 形状读取源图层蒙版（源图层停用或排在后面也生效，循环引用按空处理） |

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
| 导出 PSD | ready | 合并图层 RGB，8/16 位，嵌入 ICC；无图层、无 EXIF |
| 导出 DNG | ready | 渲染后 RGB 解码为线性 sRGB 的 LinearRaw DNG（ColorMatrix1 = XYZ D65→线性 sRGB），固定 16 位，不套输出 ICC，无 EXIF；未经 Adobe 验证器检验 |
| 导出 HEIF | partial | 依赖运行时 libheif |
| 导出 JXL | missing | 返回 UnsupportedFormat，不回退 JPEG |
| Lightroom .xmp 预设导入 | ready | 只读，映射为 AdjustmentSet（曝光/WB/HSL/曲线等） |
| Lightroom .lrcat 导入 | partial | 只读打开；评级/色标/Develop 设置按文件路径套用到照片所在文件夹库；集合已可读取但未接入界面 |
| 元数据策略 | partial | 两档 |
| AI 蒙版 | missing | 后端未接 |
| 联机拍摄 | missing | 后端未接 |
| Session .cos 只读导入 | partial | 见 C1 Catalog |
| XMP 写回 / .P1X | missing | 后端未接 |

## 人像算法分类（portrait-algorithms.json）

285 个参数按 ID 段分类，来源：产品 UI 注册表 + 效果配置。旧的“已实现 / 已解析未接入 /
无法解析”三档已停用：参数解析程度不等于后端能力，完成度改由七级证据判断。
- **可用**：达到 ev5（应用主预览接入）；ev6/ev7 未验证仍须显式说明
- **已登记未接入主预览**：ev1..ev4 有证据但 ev5 无证据，只能显示“未接入主预览”
- **能力表无对应项**：不推断后端支持，保持中性

已登记但无参数控件的效果（例如 PF30096 色调、PF90021 唇妆增强、PF44297 祛青筋）
在“人像”页单独列出并附七级证据，避免把“没有控件”误读成“能力不存在”。

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
- **人像效果**：以 `portrait-capability-registry-20261124.tsv` 的七级证据为准，未达 ev5 不得标为可用；悬停显示七级逐项状态。
- **人像参数**：解析出来只表示可保存，不表示预览/导出受支持；能力表无对应项时保持中性。
- **Metal 后端**：仅为 roadmap（未真机验证）；当前 GPU 加速是 Windows CUDA/DirectML，普通调色仍走 CPU。
EOF
