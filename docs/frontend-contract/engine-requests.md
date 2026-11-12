# 前端 → 引擎：待办与约定（2026-11-11，2026-11-12 更新）

前端（`crates/app`）这一轮把引擎已有、但界面没接上的能力接了进来。下面是需要引擎侧（Codex）处理的部分，按优先级排列。

## 对照 win-main 的结论（2026-11-12，前端已通过 SSH 只读读取 `02_engine/win-main`）

前端已经直接读取了 Windows 上的 `C:\WorkSpace\Aurora Studio\02_engine\win-main`，**之前请求的推送到 levin `~/incoming3/` 不再需要**。对照结果如下：

- `ImageEngine` trait（`engine.rs`）和 levin 完全一致。
- **`win-main` 里没有节点图类型**（`NodeGraph` 等，以及 `AdjustmentSet::nodes`，见第 0 条）。合并时这两处必须保留，否则 levin 前端会编译失败（12 处报错）。**请不要再整体覆盖 levin 的 `crates/engine/src`**。
- 第 1 条 `auto_rotate` 的 270→14 bug 仍然存在（`file_engine.rs`：`Ok(quadrant.degrees() as u8)`）。
- 新模块 `heif.rs` / `heif_write.rs` **还没有接入**：`ExportFormat::is_supported()` 里没有 HEIF，`FileEngine` 导入时也没用 `heif::decode`。接入后前端会把 HEIF 放进导出格式列表。
- `c1_styles.rs` 已导出 `load_style`，但引擎里没有入口。希望由 `ImageEngine` 提供"导入 `.costyle`/`.copreset` → `StyleInfo`"的方法，前端再加"导入 C1 风格"。
- `batch.rs`（`Batch::start/progress/cancel`）是基于 `ImageEngine` 实现的，前端可以直接拿来替换批处理页现在的队列。确认它会保留，前端就开始接。
- `scopes.rs`（波形图、分量、矢量示波器）是对 RGBA 的纯函数。前端打算用预览帧的像素来调用它，做查看器示波器。如果你更希望由引擎在 `PreviewFrame` 里直接返回示波器数据，请说明。
- 导出元数据策略已扩到 7 档，并且都已实现。前端下一轮会把导出页从 2 档改成 7 档。
- AI 蒙版：新增了头发和天空（天空是颜色估算）。人物和衣物仍然复用主体模型，见第 3 条。
- 曲线改成了单调的分段线性。前端曲线控件用的是 `Curve::eval` 画线，合并后会自动一致。前端也会禁止把点拖到比前一个点更低。
- 第 2 条能力上报：看到有 `win_capability_map` 子任务在做，接口形状请参照第 2 条。

## 前端先行：界面已做好，等引擎补的接口（2026-11-12）

前端已经按下面的接口先把界面做好了。前端这一侧的调用都集中在 `crates/app/src/planned.rs`，每个函数的文档注释就是要加的签名；在引擎补上之前，它们统一返回"引擎尚未提供此功能"。
**引擎加好任意一个之后，告诉前端；前端只改 `planned.rs` 里对应的函数体，并把 `*_READY` 改为 true，界面其他部分不用动。**

| # | 功能 | 前端入口 | 请在引擎里加的接口 |
|---|---|---|---|
| A | 胶片曲线列表 | 色彩 → 基础特性 → 胶片曲线 | `fn film_curves(&self) -> Vec<String>`（默认返回空列表），即 `base.film_curve` 可接受的 `.fcrv` 名称 |
| B | 导入 Capture One 风格 | 调整 → 风格与预设 → "导入 Capture One 风格…"（可多选文件） | `fn import_style(&self, path: &Path) -> anyhow::Result<StyleInfo>`，导入后 `styles()` 里要能看到。可以直接复用 win-main 的 `c1_styles::load_style` |
| C | 取消导出 | 批处理页，排队中或进行中的任务右侧 × | `fn cancel_job(&self, job_id: u64) -> bool`；取消后 `job_status` 返回 `Failed { message: "cancelled" }` |
| D | 原图降级提示 | 查看器右下角的黄色提示 | `fn source_status(&self, image: ImageId) -> Option<String>`：比如"libraw 解不了，显示的是机内嵌入预览"。一切正常时返回 `None`（第 4 条的另一种做法） |
| E | HDR 合并 | 图像 → 合并为 HDR…（Ctrl+H） | `fn merge_hdr(&self, images: &[ImageId], options: &HdrOptions) -> anyhow::Result<ImageInfo>`；`HdrOptions { align: bool, deghost: u8 /*0 关 1 低 2 中 3 高*/, auto_tone: bool }`。输出为线性 DNG 或 16 位 TIFF，放在第一帧旁边并加入图库 |
| F | 全景合并 | 图像 → 合并为全景…（Ctrl+M） | `fn merge_panorama(&self, images: &[ImageId], options: &PanoramaOptions) -> anyhow::Result<ImageInfo>`；`PanoramaOptions { projection: Spherical/Cylindrical/Perspective, boundary_warp: f32 /*0..=100*/, auto_crop: bool }` |
| G | 软打样 | 视图 → 软打样（Ctrl+Alt+P）；使用所选处理配方的 ICC | `PreviewRequest::proof_profile: Option<String>`：预览先转到这个输出配置文件，再转回显示，让超出色域的颜色呈现打印后的样子 |
| H | AI 降噪、条纹抑制 | 细节 → 降噪 | 前端已经把值写进 `extra["AIDenoiseAmount"]`、`extra["BandingReductionAmount"]`（C1 原生键，0..=100）。请在渲染里读取它们 |
| I | 镜头模糊 | 细节 → 镜头模糊 | 前端写入 `extra` 的键：`aurora:LensBlurAmount`（0..=100）、`aurora:LensBlurFocusNear`（0..=100，默认 0）、`aurora:LensBlurFocusFar`（0..=100，默认 100）、`aurora:LensBlurBoost`（0..=100）、`aurora:LensBlurBokeh`（0 圆形 / 1 泡泡 / 2 五叶 / 3 环形 / 4 猫眼）。需要景深估计加上按深度的模糊 |
| J | HEIF / JXL / DNG / PSD 导出 | 输出页 → 格式（标注了"未接入，导出会失败"） | 把编码器接上，并更新 `ExportFormat::is_supported()`。前端标注会自动消失 |

另外，示波器（曝光页 → 示波器：波形、分量、矢量）已经做好，**现在就能用**。它由前端根据预览像素计算，和曝光警告的做法一样（`state/scopes.rs`）。如果你希望改用 win-main 的 `scopes.rs`，请在 `PreviewFrame` 里返回示波器数据，前端再切过去。

## 前端第二轮已接入的 levin 引擎能力（2026-11-12），以及还缺的接口

已接入（全部是 levin 引擎已经渲染或导出、但之前没有界面的字段）：

- 导出元数据：7 档全部开放（`MetadataPolicy::is_implemented()` 为 true 的才列出）。
- 镜头：`LensCorrection::use_profile` 开关（"使用镜头库校准"）。
- 基础特性：`BaseCharacteristics::film_curve` 可以查看，并可关闭（设为 `None`）。
- 色彩编辑器（基本模式）：所选色区的 `ColorBand` 全部字段，包括色相和饱和度范围、平滑度、3 项均匀度、反转；高级和肤色范围新增 `min_saturation`。
- 节点蒙版：`Mask::color_range`（中心、范围、平滑度、最低饱和度）。

需要引擎补的：

1. **胶片曲线列表**：`ImageEngine` 没有列出可用 `.fcrv` 的方法，所以前端只能"保留当前曲线"或"关闭"。希望加上 `fn film_curves(&self) -> Vec<String>`（默认返回空列表），前端会做成下拉选择。
2. **`ProcessRecipe::open_with` 没有生效**：导出代码里没有用到这个字段，所以前端没有做"导出后打开"。如果要支持，请在导出完成后执行它，或者把这个字段删掉。
3. `contract.rs` 的 `COLOR_EDITOR` 状态还写着"18 个范围字段只用到 5 个"。现在 `ColorBand` 全部字段都已经有界面了，请确认引擎实际渲染了哪些字段，前端再更新这个状态说明。

## 0. 同步会覆盖前端补进引擎的内容（请先看）

20:49 那次从 `~/incoming2` 整体覆盖 `crates/engine/src` 后，节点图的纯数据类型丢失，前端编译失败（12 处报错）。
为了让前端能编译，前端**只补回了纯数据**，没有改动任何渲染逻辑：

- `layers.rs` 末尾：`NodeKind` / `GraphNode` / `NodeLink` / `NodeGraph`，以及 `NodeGraph::{INPUT, OUTPUT, node, source, reaches, render_order}`，外加 3 个测试（`layers::node_tests`）。
- `adjust.rs`：`AdjustmentSet::nodes: Option<NodeGraph>`（序列化时如果是 `None` 就省略；旧数据仍能正常读取）。

**请把这两处合并进你的源码树。** 否则下次同步还会把它们删掉。语义见 `node-graph.md`。

`node-graph.md` 第 1 条写的"`layer_plans` 按 `render_order()` 筛选图层"在当前代码里**已经不存在**（也是被同步覆盖掉的）。
现在的情况是：预览、缩略图、取色由前端先做 `flattened()` 再发给引擎，所以结果是对的；但**导出和批处理用的是存储的调整，没连上的节点也会被渲染**。请恢复这条规则：

```rust
// mock/process.rs, layer_plans
let order = adj.nodes.as_ref().map(|g| g.render_order());
// Some(order) → 只取 order 中的图层并按 order 排序；None → 按 adj.layers 原顺序
```

## 1. Bug：`auto_rotate` 把 270° 截断成 14

`fn auto_rotate(&self, ..) -> Result<u8>` 返回的是角度，而 `quadrant.degrees()` 的类型是 `u16`，所以 `270 as u8 == 14`。
前端目前把 14 当作 270 处理（`state/analysis.rs::quarter_turns`）。

请改成返回顺时针的四分之一圈数 `0..=3`（和 `Geometry::orientation` 的单位一致），或者改为返回 `u16`。改完告诉前端，前端会删掉对 14 的特判。

## 2. 能力上报（替代前端写死的状态）

现在前端是靠 `contract.rs` 里手写的常量来决定哪些功能可用，和引擎的实际能力已经对不上了（AI 蒙版、联机拍摄都曾被误标为"后端未接"）。
请在 `ImageEngine` 中加一个带默认实现的方法：

```rust
fn capabilities(&self) -> EngineCapabilities { EngineCapabilities::default() }

#[derive(Clone, Debug, Default)]
pub struct EngineCapabilities {
    pub ai_mask_kinds: Vec<AiMaskKind>, // 真正有模型的种类
    pub eye_report: bool,
    pub auto_rotate: bool,
    pub match_look: bool,
    pub tether: bool,                   // 本平台是否支持联机拍摄（与是否连着相机无关）
    pub export_formats: Vec<ExportFormat>,
}
```

对应检查要轻量（只看模型文件和运行库在不在，不要真的跑推理）。接好后，前端会根据它来灰显菜单项和 AI 蒙版种类。

## 3. AI 蒙版

- `Person` 和 `Clothing` 目前直接复用主体模型的结果，用户会以为得到的是"人物"或"衣物"蒙版。请改成和 `Sky` 等一样返回错误，或者接入真正的模型。
- 前端已改为：菜单始终可用，失败时显示原因，并在本次会话里把该种类标成"不可用"。
- 模型来自 Capture One 的安装目录（`subjectMasking.onnx` 等）。按照 AGENTS.md 的资产规定，发布前需要仓库所有者确认授权。

## 4. RAW 解码降级要让前端知道

libraw 失败，或者解码结果和机内预览不一致时，引擎会静默改用嵌入的 8 位 JPEG 预览，只写一条 `warn` 日志。
请在 `PreviewFrame`（或 `ImageInfo`）中加一个字段：

```rust
pub degraded: Option<String>, // 例如 "嵌入预览（libraw 无法解码 Nikon HE）"
```

前端会在查看器和缩略图上加标记。

## 5. 错误信息可本地化

前端现在会把引擎的错误原文显示给用户（预览渲染失败、缩略图失败、AI 蒙版、闭眼检查、自动旋转、风格匹配），而这些原文全是英文。
请对常见失败给出稳定的错误类型（参照已有的 `UnsupportedFormat` 做法，比如"模型缺失""运行库缺失""不支持的文件"），前端按类型翻译；其他错误继续显示原文。

## 6. 联机拍摄

- `FileEngine::camera_status()` 永远返回 `Disconnected`，`camera_settings()` 永远返回 `None`。前端已改为按 `tether_status()` 来启用拍摄功能，但相机型号、电量、存储卡和设置都显示不出来。
- `live_view_focus` 目前什么也不做。
- 只支持 Linux（libgphoto2）。Windows 需要另外的后端，前端的状态说明里已经注明了这一点。

## 7. Match Look

前端把结果里的以下字段叠加到目标图上：`white_balance`（整个结构）、`exposure.exposure`、`exposure.saturation`、`levels.rgb`。其余字段保持目标图原值，每张图记为一步撤销。
如果引擎以后会在结果里返回更多字段，请在 `MatchLook` 的文档里写明"哪些字段属于这次匹配的结果"，前端照着改。

## 8. 节点图混合器（原 node-graph.md 第 2、3 条，仍然有效）

并行混合器：`out = I + Σ m_i·(B_i − I)`；图层混合器：`out = B_0；然后对每个 i，out = lerp(out, B_i, m_i)`。
做完之后告诉前端，把 `NODE_GRAPH` 的状态改为 Ready。

## 9. 显示器色彩管理（前端已就绪，等引擎）

前端已完成：首选项 → 外观 → 色彩管理 → "显示器配置文件"。会列出系统里的 ICC 配置文件（Linux 为 `~/.local/share/icc` 和 `/usr/share/color/icc`，Windows 为 `System32\spool\drivers\color`，macOS 为 ColorSync 目录），用户的选择保存在 `Settings::display_profile: Option<PathBuf>`。界面上明确标注了"引擎尚未转换，目前按 sRGB 显示"（`contract::DISPLAY_PROFILE`）。

需要引擎做的：

```rust
// PreviewRequest 新增（serde 缺省为 None = 按 sRGB 输出，与现在一致）
pub display_profile: Option<std::path::PathBuf>,
```

- 预览、缩略图、第二查看器都按这个配置文件输出。直方图、取色、曝光警告仍然以工作空间的值为准，不受显示转换影响。
- 配置文件读取失败时退回 sRGB，并在 `PreviewFrame` 里给出原因（可以和第 4 条的 `degraded` 字段共用）。
- 做完后告诉前端，前端负责把设置传进请求，并把状态改为 Ready。

## 10. 清理

- `crates/engine/src/file_engine.rs.rs`：旧副本，没有被任何模块引用。
- 根目录的 `engine/`：不在 workspace 里，是否删除由你判断。

## 本轮前端已完成（供参考）

| 能力 | 入口 |
|---|---|
| 联机拍摄按 `tether_status()` 启用；可"重新检测"；分别说明"未找到库"和"未连接相机" | 拍摄页、工具栏、Ctrl+K |
| AI 蒙版菜单启用；失败时显示原因并标记"不可用" | 节点页 |
| 闭眼检查（选中的图；拍摄后自动检查）；缩略图上显示闭眼标记 | 图像 → 检查闭眼 |
| 自动旋转（选中的图，每张记一步撤销） | 图像 → 自动旋转 |
| 风格匹配（参考图为对比变体或当前图） | 调整 → 匹配参考风格 |
| 预览渲染失败时，在查看器左上角显示原因 | 查看器 |
| 缩略图生成失败时显示"无法读取"占位 | 浏览器 |
| 人像页在接入前折叠为一张说明卡；新建工作区时默认隐藏该页 | 人像页 |
| 鼠标悬停在风格或预设上时临时预览效果（不提交） | 调整页 |
| 挑片模式（全屏，1–5 评级，X 拒绝，自动前进，仅未评级，Z 键 100%） | 视图 → 挑片模式（Ctrl+Alt+C） |
| 快速页签（可以从任意面板标题的右键菜单添加或移除，并调整顺序）；页签可以拖动排序 | 工具栏左侧页签 |
| 显示器配置文件设置（等引擎，见第 9 条） | 首选项 → 外观 |
| 导出元数据 7 档 | 输出页 → 元数据 |
| 镜头库校准开关 | 镜头页 |
| 胶片曲线查看与关闭 | 色彩页 → 基础特性 |
| 色带范围、平滑度、均匀度、反转；高级范围的最低饱和度 | 色彩页 → 色彩编辑器 |
| 蒙版颜色范围 | 节点页 |
| 示波器：波形、RGB 分量、矢量（可用） | 曝光页 / 快速页签 |
| 前端先行 A–J：胶片曲线列表、导入 C1 风格、取消导出、原图降级提示、HDR/全景合并、软打样、AI 降噪、条纹抑制、镜头模糊、更多导出格式 | 见"前端先行"一节 |
