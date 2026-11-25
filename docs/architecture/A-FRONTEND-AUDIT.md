# A-FRONTEND-AUDIT — AppState、命令/手势与导出/任务边界

- Lane: A-frontend（只读审计）
- Baseline: `8118a4a`（本 worktree `J-X-C-Z/架构`）
- 写权限: 仅本文件。本轮未改 Rust 源码、`Cargo.toml`、既有文档，未 commit/push。
- 方法: 逐文件读源码 + `rg` 定位；未运行 `cargo check/test`（其他 lane 在编译），未做 UI 计时或重绘计数。
- 标注: **[R]** 直接读到的源码事实；**[I]** 由代码路径推断（未运行验证）；**[U]** 无法在本机证明，明确列为未验证。

> 本 lane 没有测量 UI 延迟、帧时、重绘次数或拖动掉帧。下文所有“同步/阻塞”“可能卡顿”均为源码路径判断，不是实测数值。

---

## 0. 结论摘要（按影响排序）

| # | 发现 | 证据 | 级别 |
|---|---|---|---|
| F1 | 每次手势提交都在 UI 线程同步落盘：`end_gesture → after_commit → save_adjustments_batch → persist_batch`，含 N 个调整文件写 + `catalog.json` 重写。批量编辑 N 张也是一次同步写入。**后续状态：已修复**——app 提交走 `save_adjustments_queued`（无队列引擎回退同步 report），Quit 时 `flush_and_quit` 有界 flush 并提示未完成项。 | `crates/app/src/state/mod.rs:1159,1206`; `crates/engine/src/file_engine.rs` | [R] |
| F2 | 导出在**进入导出准入之前**、在**调用线程（UI）上**获取源：`native_raw_source(job.image, None)` 可能同步解码并阻塞 `decode_gates.heavy` Condvar；随后才 `std::thread::spawn` + `ExportScheduler::acquire_cancellable`。 | `crates/engine/src/file_engine.rs:4665-4669,4681-4689`; gate `:1333-1365` | [R] |
| F3 | 全分辨率导出源只有 `native_raw_source`（仅 RAW/EIP）；**所有非 RAW 源**以及 native RAW 解码失败时都走 `self.source()` 的 4096 proxy。**后续状态：已修复**——请求长边 > proxy 时导出 worker 在 scheduler 准入后解码原图全分辨率；失败/超内存上限让 job 明确失败，绝不放大 proxy；有 4500×3000 合成测试与内存上限测试。≤4096 仍走预览源。 | `crates/engine/src/file_engine.rs`（`StandardExportSource`/`FullResExportRequest`） | [R] |
| F4 | 保存在非“提交事件”上也会发生：`set_current`（换图）、`undo`、`redo`、`edit()` 都先调 `end_gesture`；应用退出路径没有显式 flush（`Quit` 直接 `cx.quit()`），进行中的手势/未提交改动无落盘保证。 | `crates/app/src/state/mod.rs:820,1167-1169,1229,1242`; `crates/app/src/lib.rs:63` | [R]（退出 flush 为 [I]） |
| F5 | App 看不到 A-storage 新增的 `SaveReport` 明细：trait `save_adjustments_batch` 把报告压成 `into_error()`；`after_commit_many` 只拿一个 `Result`。`Superseded`/`index_error` 无法分开展示。**后续状态：已修复**——新增 `ImageEngine::save_adjustments_batch_report`，app 显示 saved/failed/skipped + 索引失败，并有 3 个测试。 | `crates/engine/src/file_engine.rs`; `crates/engine/src/save_report.rs`; `crates/app/src/state/mod.rs` | [R] |
| F6 | 错误分类不完整：预览失败是原始字符串；AI 掩码失败只记住 `AiMaskKind` 集合并拼字符串；HDR/全景的结构化 `MergeErrorCode` 在 `planned` 里被压成 `anyhow` 字符串。 | `crates/app/src/state/render.rs:1078-1084`; `crates/app/src/state/mod.rs:1715-1719,1744`; `crates/app/src/planned.rs:77,86` | [R] |
| F7 | 能力/效果清单与引擎不一致：`contract::AI_MASK` 文案只说 Subject/背景/皮肤，`FileEngine::capabilities` 报 7 种；证据表 ev5 仅 8 项，App 的 `WIRED_*` 列表共 81 项，`effect_descriptor::seed_descriptors` 只覆盖其中 6 项。 | `crates/app/src/contract.rs:62`; `crates/engine/src/file_engine.rs:4837-4852`; `crates/app/src/evidence.rs:251-253`; `crates/app/src/tools/portrait.rs:73-180`; `crates/portrait/src/effect_descriptor.rs:554-705` | [R] |
| F8 | Windows 句柄耦合集中在显示配置探测：`window_hwnd: isize`、`detected_display_profile(hwnd)`、`win_icm`（GetDC/EnumDisplayMonitors）。预览/缩略图请求把它当普通输入，没有平台抽象。 | `crates/app/src/state/mod.rs:539`; `crates/app/src/state/render.rs:854,1171`; `crates/app/src/state/settings.rs:137-166,180-370`; `crates/app/src/ui/viewer/mod.rs:1218-1240` | [R] |
| F9 | 优先级并不独立：预览单飞并合并最新请求；before 帧在 after 完成后串行发第二次全量渲染；scope 与 overlays 在同一个后台任务里 `rayon::join`；缩略图仅在“无预览在飞”时运行；导出任务走独立 `ExportScheduler` + 共享 decode gate，不与预览共享内存预算。 | `crates/app/src/state/render.rs:274-331,923-1135,1146-1209`; `crates/engine/src/file_engine.rs:1438-1500` | [R] |
| F10 | 隐私：导出日志已做文件名/sanitize，但应用与引擎在 warn/info 仍有绝对路径（目录）与层名输出。 | `crates/app/src/lib.rs:49`; `crates/engine/src/file_engine.rs:590,792,994-997,3475,3683`; sanitizer `crates/engine/src/output.rs:830-860` | [R] |

---

## 1. AppState 责任清单

`pub struct AppState` 起始 `crates/app/src/state/mod.rs:508`，字段 509–619。按“职责”而非“可访问性”分组：

### 1.1 引擎与库（Library）

| 字段 | 行 | 职责 | 备注 |
|---|---:|---|---|
| `engine: Arc<dyn ImageEngine>` | 509 | 唯一处理入口 | 边界符合 AGENTS：app 只经 trait 调用 |
| `images`, `nodes`, `trash` | 510-514 | 目录/集合/回收站 | 全量内存镜像 |
| `active_collection`, `active_collection_name` | 515,550 | 当前集合 | |
| `filter`, `sort`, `sort_desc`, `custom_order`, `stacks`, `open_stacks` | 516-518,610-613 | 过滤/排序/栈 | `visible_images()` 每次全量重算（`mod.rs:760`） |
| `selection`, `current` | 519-520 | 选择与“当前编辑图” | `current` 同时驱动预览、编辑、AI、历史 |
| `recent_imports`, `recent_captures` | 512-513 | 会话内最近项 | |
| `annotations`, `eyes`, `sampled_color`, `picker_point` | 553,585,564-565 | 会话派生数据 | `eyes` 仅本会话 |
| `styles`, `recipes`, `selected_recipe` | 554,559-560 | 用户资产/导出配方 | 配方是 UI 侧默认值（`new()` 硬编码 3 条，`mod.rs:632-653`） |

### 1.2 编辑器 / 文档（Editor/Document）

| 字段 | 行 | 职责 | 备注 |
|---|---:|---|---|
| `edits: HashMap<ImageId, Edit>` | 521 | 每图 `AdjustmentSet` + undo/redo + gesture | `Edit` 定义 `mod.rs:250-256` |
| `active_layer: Option<u64>` | 523 | 当前层；`None` 背景 | |
| `active_node: Option<u64>` | 525 | 节点图混音/输出选择 | 与 `active_layer` 语义分开 |
| `tool_tab`, `cursor`, `spot_anchor` | 526-530 | 工具上下文 | |
| `clipboard`, `clip_groups` | 547-549 | 调整拷贝/粘贴 | |
| `hover_look` | 575 | 风格悬停预览（不落盘） | |

`Edit`（`mod.rs:250-256`）：`adj`、`undo: Vec<(label, AdjustmentSet)>`、`redo`、`gesture: Option<(label, before)>`。**历史与文档状态耦合在同一个结构里**，是本 lane 建议最先拆出的纯数据单元。

### 1.3 视口 / 预览（Viewport/Preview）

| 字段 | 行 | 职责 | 备注 |
|---|---:|---|---|
| `preview: Preview` | 541 | 当前/前后帧、直方图、generation、pending、scope、error | 定义 `mod.rs:318-347` |
| `thumbs`, `thumb_queue`, `thumb_rev`, `thumb_task`, `thumb_failed` | 542-546,589 | 缩略图缓存与单队列 | `thumb_task: Option<Task<()>>` 是 GPUI 任务句柄 |
| `zoom`, `viewport`, `viewport_origin`, `scale_factor` | 533-540 | 缩放/布局输入 | `viewport` 为设备像素 |
| `window_hwnd` | 539 | 显示器配置探测句柄 | 唯一平台句柄（见 §4） |
| `source_note` | 592 | 当前图非全数据原因 | 每次预览刷新 |
| `annotations` | 553 | 批注帧坐标 | |

`Preview` 关键私有字段（`mod.rs:318-347`）：`generation: u64`、`in_flight`、`pending: Option<(RenderQuality, Instant)>`、`before_dirty`、`rendering`、`draft`、`scope`、`error: Option<(ImageId, SharedString)>`、`last_request/last_overlay/last_scope`（用于不重渲的 overlay 复用）。

### 1.4 历史（History）

没有独立字段；历史在 `Edit.undo/redo`（`mod.rs:252-253`）。全局 `can_undo/can_redo/history/jump_history` 都通过 `current` 取对应 `Edit`（`mod.rs:1099-1118,1268-1291`）。

### 1.5 任务（Tasks）

| 字段 | 行 | 职责 | 备注 |
|---|---:|---|---|
| `jobs: Vec<JobView>`, `next_job` | 561-562 | 导出队列的 UI 镜像 | `poll_jobs` 每 120 ms 拉取（`mod.rs:1891-1912`） |
| `ai_busy` | 572 | AI 掩码单飞标志 | 每图一次，`mod.rs:1693-1698` |
| `analysing` | 587 | 眼睛/旋转/Match Look 分析单飞 | `state/analysis.rs:54-61` |
| `capturing` | 551 | 拍摄单飞 | `mod.rs:1534-1543` |
| `tether_probing` | 570 | 探针在飞 | |
| `c1_save_busy`, `c1_save_status`, `c1_save_warnings` | 577-581 | C1 保存结果 | |
| `thumb_task` | 546 | 后台缩略图任务 | |
| `notice` | 571 | 短消息 | 所有失败最终多落在这里 |

### 1.6 设备 / 能力（Device/Capabilities）

| 字段 | 行 | 备注 |
|---|---:|---|
| `camera`, `camera_settings`, `tether` | 566-569 | 启动后探针；`tether` 优先于 `camera`（注释 `mod.rs:666-668`） |
| `ai_unavailable: HashSet<AiMaskKind>` | 583 | 只记“哪种”不可用，不记原因 |
| `icc_profiles`, `icc_group`, `icc_query` | 555-557 | ICC 列表一次性从引擎取 |
| `lens_profiles` | 558 | `FileEngine::lens_profiles` 当前返回空（`file_engine.rs:4626`） |
| `capabilities: EngineCapabilities` | 596 | 启动时读一次（`planned.rs:22`） |
| `engine_name` | 594 | |
| `source`, `welcome`, `opening`, `open_error` | 598-605 | 打开流程 |

### 1.7 设置 / UI 偏好（Settings）

`settings: Settings`（593）、`view: ViewPrefs`（531）、`panels: PanelPrefs`（532）、`tool_tab`（526）、`slideshow`（614）、`cull`（615）、`prev_cursor/cursor_changed`（618-619）。`Settings` 自己有 `save()`（多处直接调用，例如 `mod.rs:1307,1322,1330`）。

### 1.8 需要跨子树共享、暂不宜拆的字段

`current`、`selection`、`edits`、`settings` 被 Library、Editor、Preview、Tasks 同时读；任何按字段切 Entity 都会带来跨 Entity 读锁。建议先拆**纯结构 + 服务**，再把 GPUI `Entity` 用在异步完成方（Preview/Jobs），不要先拆库。

---

## 2. 编辑命令 / 手势生命周期

### 2.1 函数逐个

| 函数 | 行 | 行为 | 撤销步 | 保存 | 渲染 |
|---|---:|---|---|---|---|
| `edit_live(label, f)` | 1123 | 拖动中调用；首次调用记录 `gesture=(label, before)`，之后只改 `adj` | 否 | 否 | `Draft` |
| `apply_gesture_edit` | 1129 | `gesture.is_none()` 时快照 | — | — | — |
| `end_gesture` | 1140 | 取走 gesture；`before != adj` 时 push undo（1154），清 redo，超 `undo_levels` 丢最旧，`after_commit`（1159） | 一次 | 同步一次 | `Final`（无论是否变化，1160） |
| `edit(label, f)` | 1166 | `end_gesture(); apply; end_gesture()` | 一次 | 同步 | Final |
| `edit_selected(label, f)` | 1173 | 对 `selection` 每张直接 push undo（1183），一次 `after_commit_many`（1188） | 每图一次 | 批量一次 | Final |
| `after_commit` / `after_commit_many` | 1194/1198 | `save_adjustments_batch`、更新 `has_adjustments`、`queue_thumbs` | — | 同步 | — |
| `undo` | 1228 | `end_gesture` → pop undo → swap → push redo → `after_commit` → Final | 回退一步 | 同步 | Final |
| `redo` | 1241 | `end_gesture` → pop redo → swap → push undo（1247） | 前进一步 | 同步 | Final |
| `jump_history` | 1268 | `end_gesture` → 连续 pop/push 到目标长度 → `after_commit` 一次 | 多步 | 同步一次 | Final |
| `apply_loaded_style` | 1369 | 异步风格加载回来后，逐图 push undo（1395），一次批量保存（1400） | 每图一次 | 批量一次 | Final |
| `set_current` | 818 | 换图前 `end_gesture`（820），清预览/采样，`request_render(Final)`（833） | 可能提交 | 可能同步 | Final |

`end_gesture` 的调用者（除内部）：`widgets/slider.rs:293`（slider on_commit）、`tools/exposure.rs:250,306`、`tools/color.rs:301`、`tools/nodes.rs:261`、`ui/viewer/mod.rs:491`、`ui/workspace.rs:792`、`ui/mobile.rs:601`、`perf.rs:213`（录制回放）。这说明“提交”由控件释放事件驱动，而不是文档层的显式事务。

### 2.2 不变量（源码可证）

1. **一次拖动 = 一次 undo + 一次保存**，前提是 `end_gesture` 只被调用一次且期间 `current` 不变。[R] `mod.rs:1132-1135,1152-1159`
2. **无变化不产生 undo**：`before != edit.adj` 才 push，但**仍会**发 Final 渲染（1160）。[R]
3. **批量编辑每图一个 undo、整批一次保存**。[R] `mod.rs:1183-1188`
4. **undo/redo 先提交未完成手势**，因此可能在一次用户操作里产生两次保存（手势 + 回退）。[R] `mod.rs:1229,1242`
5. **保存失败不回滚内存调整**：`after_commit_many` 只 `notify`，`Edit.adj` 保持新值、undo 栈已前进；没有重试队列。[R] `mod.rs:1206-1213`
6. **每次提交都重建缩略图**（`queue_thumbs(ids)`，1222），N 张批量编辑会入队 N 个缩略图任务。[R]

### 2.3 边界问题

- **非提交边界保存**：换图（`set_current:820`）、undo/redo、离散 `edit()` 都会提前结束手势并落盘。语义上可接受，但“保存时机”取决于 UI 事件而非显式 commit，难以在文档层测试。[R]
- **退出无 flush**：`lib.rs:63` 的 `Quit` 只有 `cx.quit()`；没有 `end_gesture`/保存调用。若用户在拖动中退出且 GPUI 不补发 mouse-up，最后一次手势不会落盘。[I]，未运行验证。
- **SaveReport 未接通**（与 A-storage 协调，勿由本 lane 改）：`after_commit_many` 调 trait 方法，返回 `into_error()` 汇总（`file_engine.rs:4107-4112`），`Superseded` 与 `index_error` 不能分开呈现；批量部分失败只有“N of M + first failure + catalog index”一段文本。[R]
- **同步 I/O**：`persist_batch` 在调用线程写 N 个调整文件并 `store.save_index`（`file_engine.rs:3279-3327`），调用链在 UI `Context` 内；结构上会阻塞 UI，但未测时延。[R]/[I]
- **保存顺序**：`SaveOrder` 用每路径递增序号拒绝旧写（`file_engine.rs:1618-1642`），因此旧的 `end_gesture` 不会覆盖新的；这是正确的 stale-write guard。[R]

---

## 3. 预览 / generation

### 3.1 请求与安装

| 机制 | 位置 | 行为 |
|---|---|---|
| `request_render(quality)` | `render.rs:923` | 设 `preview.draft`，进 `enqueue_render` |
| `enqueue_render` | `render.rs:928` | `accept_request` 合并；生成 `generation += 1`；算 before 请求；`cx.spawn` 后台渲染 |
| `accept_request` | `render.rs:274-287` | 有 in-flight → 覆盖 `pending` 为最新，返回 `Coalesced`；否则 `Started` |
| `finish_request` | `render.rs:296-300` | 清 in_flight，取走 pending，递归 `enqueue_render` |
| `may_install` | `render.rs:302-315` | 只有 `current == image_id` 且 **已显示帧的 stamp.id 不大于新 generation** 才安装 |
| `needs_before`/`wants_before` | `render.rs:200-220,317-324` | compare 开且 (dirty 或 图/geometry/region 变) 才渲 before；只在 after 完成后、仍在当前图时安装 |
| `refresh_overlays_if_unchanged` | `render.rs:882-900` | 输入相同且无 scope 时只重建 warning/focus，不重渲染 |
| `thumbs_allowed` | `render.rs:325-330` | **无预览 in_flight 且队列非空** 才允许缩略图 |
| `queue_thumbs`/`start_queued_thumbs` | `render.rs:1146-1208` | 单任务循环；`thumb_rev` 变了或预览在飞就退出 |
| `refresh_thumbs` | `render.rs:1213-1222` | 清缓存、`thumb_rev += 1`、重建 |

**stale 丢弃**：after 帧靠 `may_install`；缩略图靠 `thumb_rev` 相等才插入（`render.rs:1183,1203`）。旧 generation 不会覆盖当前图或更晚显示的帧（与 AGENTS 性能条款一致）。[R]

**漏洞点**：`may_install` 只比较 `stamp.id > generation` 且要求同图；如果旧帧的 generation **大于**新帧（乱序完成），会被正确拒绝；若旧帧 generation 更小但当前帧不是同图，也会被拒绝。逻辑成立。[R]

### 3.2 Draft / Final 选择点

| 场景 | 质量 | 行 |
|---|---|---|
| 启动 / 换图 / 工具切换 | Final | `mod.rs:744,833,1770` |
| 控件拖动 `edit_live` | Draft | `mod.rs:1125` |
| 手势结束 / 离散编辑 / undo-redo / 批量 | Final | `mod.rs:1160,1190,1236,1249,1402` |
| 平移 `pan_by` | Draft | `mod.rs:1798` |
| 缩放 `set_zoom` / 视口变化 | Final | `mod.rs:1787,1809` |
| before 帧 | Final | `render.rs:951` |

引擎侧 `preview_frame`（`file_engine.rs:3187-3197`）明确：Draft 与 Final 输出同尺寸；Draft 用 half-size `fast` 渲染再上采样，避免释放时跳变。这与 AGENTS“同分辨率、同像素质量”一致。[R]

### 3.3 优先级是否独立

| 工作 | 通道 | 是否有独立优先级 | 证据 |
|---|---|---|---|
| 主预览 | 单飞 + pending 最新 | 是（最新覆盖） | `render.rs:274-300` |
| before-after | 主预览任务完成后串行再发一次全量 | 否（排在 after 之后） | `render.rs:1094-1134` |
| scope | 与 overlays 同一 `rayon::join` | 否（同一帧任务内） | `render.rs:997-1012` |
| 缩略图 | 仅预览空闲时；decode gate `Thumbnail` 优先级 | 部分（引擎 gate 有，但 App 层被预览完全饿死） | `render.rs:325-330,1151-1208`; `file_engine.rs:1230-1245` |
| 导出 | 独立 `ExportScheduler`（槽位+内存预算）+ 共享 decode gate `Interactive` | 是，但与预览不共享预算 | `file_engine.rs:1438-1500,4684-4689` |

---

## 4. 依赖 Windows HWND 的业务状态

| 位置 | 行 | 用途 | 跨平台适配需要 |
|---|---:|---|---|
| `AppState.window_hwnd: isize` | `state/mod.rs:539` | 保存主窗口原生句柄 | 抽象为 `DisplayTarget`（monitor id / output）而非裸句柄 |
| 赋值 | `ui/viewer/mod.rs:1218-1240` | `RawWindowHandle::Win32(w).hwnd`；非 Windows 恒 0 | 用 `raw-window-handle` 的 `AppKit`/`Xlib`/`Wayland` 分支，或 GPUI 暴露的 `window.display_id()` |
| `detected_display_profile(hwnd)` | `state/settings.rs:137-147` | Windows 读 ICC；非 Windows 恒 `None` | macOS 用 `CGDisplayCopyColorSpace`/ColorSync；Linux 用 colord/X11 `_ICC_PROFILE` |
| `detected_display_profile_status` | `state/settings.rs:156-166` | 偏好页状态 | 同上 |
| `win_icm`（GetDC/GetWindowRect/EnumDisplayMonitors） | `state/settings.rs:180-370` | 找窗口下显示器 | 平台模块；接口保持 `(Option<PathBuf>, fallback)` |
| 预览请求读句柄 | `state/render.rs:854` | 每个 `PreviewRequest.display_profile` | 改为 `AppState.display_profile()` 服务，句柄不进入请求构造 |
| 缩略图请求读句柄 | `state/render.rs:1171` | 同上 | 同上 |
| 偏好页 | `ui/prefs.rs:359` | 显示配置状态 | 同上 |
| 缓存失效 | `state/mod.rs:1808` | 视口变化时 `invalidate_display_profile_cache()` | 平台无关缓存键（monitor id） |

**结论**：HWND 只影响“显示器 ICC 探测”，没有进入渲染协议；适配面小。需要把 `isize` 换成枚举 + trait 方法，并让 `PreviewRequest.display_profile` 的构造只依赖已解析的 `Option<PathBuf>`（请求本身已如此）。[R]

---

## 5. 硬编码能力 / 效果清单

### 5.1 `crates/app/src/contract.rs`

| 项 | 行 | 内容 | 是否应查引擎 |
|---|---:|---|---|
| `Status` 常量（`COLOR_EDITOR`…`METAL`） | 50-130 | 24 条人工维护的功能状态文案 | 部分可查 `EngineCapabilities`；多数是“算法近似程度”，引擎今天不报告 |
| `export_formats` 回退 | 85-91 | 引擎报空时用静态 `ExportFormat::ALL.filter(is_supported)` | 是；`is_supported()` 是编译期能力，不是运行期 |
| `format_label` | 99-102 | 未接入仍列出并标“导出会失败” | 与 `format_ready` 一致，可保留 |
| `BANDS: [Band; 8]` | 249 | PF id 段硬编码 | 应由 `effect_descriptor`/registry 生成 |
| `LABELS` | 378 | UI 键→中英标签 | 人像注册表也有 label；重复来源 |
| `UNRESOLVED` | 170-176 | 嵌入 `docs/frontend-contract/portrait-algorithms.json` | 静态快照；不查引擎 |
| `AI_MASK` 文案 | 62 | “仅 Subject、背景、皮肤” | **已与引擎不一致**：`file_engine.rs:4837-4852` 报 Subject/Person/Clothing/Skin/Hair/Sky/Background |
| `TETHER` 文案 | 63 | “设置与对焦尚未读取” | 与引擎实现需人工核对，未见运行期查询 |

### 5.2 `crates/app/src/evidence.rs`

| 项 | 行 | 内容 | 问题 |
|---|---:|---|---|
| `REGISTRY` 嵌入 TSV | 17 | `docs/frontend-contract/portrait-capability-registry-20261124.tsv` | 日期/字节数/SHA 硬编码（`REGISTRY_BYTES=76393`，`REGISTRY_SHA256`） |
| `LEVELS` / `MAIN_PREVIEW=4` | 25-38 | 七级证据、ev5 门槛 | 策略硬编码：只有 ev5 才算“可用” |
| `catalog_key` 特例 | 93-99 | PF30096 → `"sediao"` | 单点硬编码映射 |
| `Availability` | 157-176 | ev5 → Preview | 不查引擎运行期；`Unregistered` 不等于不可用 |
| 测试计数 | 251-253 | 147 effects，8 preview-connected | 与 `WIRED_*` 81 项不一致（见下） |

### 5.3 `crates/app/src/tools/portrait.rs`

| 列表 | 行 | 数量 | 用途 |
|---|---:|---:|---|
| `RETOUCH_GROUPS` | 30-35 | 4 | 面板分组 |
| `SKIN_SECTIONS` | 38-45 | 6 段 / 30 keys | 面板分组 |
| `FACE_PAIRS` | 48-69 | 20 | 左右成对命名 |
| `WIRED_SKIN_EFFECTS` | 73-98 | 24 | slider + “是否已接入” |
| `WIRED_EYES_EFFECTS` | 101-104 | 2 | |
| `WIRED_HAIR_EFFECTS` | 107-109 | 1 | |
| `WIRED_TEETH_EFFECTS` | 112-116 | 3 | |
| `WIRED_FACE_GEOMETRY_EFFECTS` | 120-161 | 40 | |
| `WIRED_FACE_SINGLE_EFFECTS` | 163-167 | 3 | |
| `WIRED_PLUMP_EFFECTS` | 171-179 | 8 | |
| `WIRED_TOTAL` | 182-189 | **81** | 由上面相加 |
| `has_face_reshape` / `has_live_portrait_effect` | 190-210 | — | 驱动预览“人像已应用”标志与 region 扩边（`render.rs:838,957`；测试另见 `render.rs:1426`） |

### 5.4 与 `aurora_portrait::effect_descriptor` 的交叉核对

- `seed_descriptors()`（`crates/portrait/src/effect_descriptor.rs:554-705`）当前播种 6 个：`skin.gloss`(8068)、`skin.texture`(8069)、`skin.clarity`(8070)、`skin.color_hue`(21015)、`skin.temperature`(30096)、`skin.whitening`(90005)。
- 证据表 ev5 实际有 8 个 effect：8068/8069/8070/21015/30096/**90004**/**90007**（`docs/frontend-contract/portrait-capability-registry-20261124.tsv`，`main_preview` 非“未验证”）。
- App 的 `WIRED_*` 有 81 个 key，远超 ev5 的 8 个。
- **无法今天验证的项**：
  1. 75 个 `WIRED_*` key 没有任何 `EffectDescriptor`（无法回答 static/evidence/runtime 三维状态）。
  2. 90004（skin radiance）、90007（skin rosy）有 ev5 证据但没有 descriptor。
  3. `has_live_portrait_effect` 只看 `extra` 是否有非中性值，不看引擎能力/模型是否存在；因此“已应用”标志可能在人像后端缺失时为真。
  4. 其余 147 条 registry 行没有对应 descriptor。
- 建议（不改本 lane）：UI 的每个 slider 状态查 `EffectRegistry::get(EffectId)` + `capability()`，缺 descriptor 时显示“未登记”而不是“已接入”；保留 `WIRED_*` 仅作面板分组数据。

---

## 6. 导出流水线与资源准入

### 6.1 `FileEngine::start_export`（`file_engine.rs:4629-4794`）实际顺序

| 步骤 | 行 | 线程 | 说明 |
|---|---:|---|---|
| 1. 格式/位深校验 | 4630-4631 | UI | `ensure_supported` / `supports_bit_depth` |
| 2. 取 ImageInfo/调整快照 | 4633-4643 | UI | 锁 state 一次 |
| 3. 目标目录解析 + staging 清扫 | 4644-4649 | UI | `resolve_output_dir`、`sweep_stale_staging` |
| 4. Z6III 精确路径（feature 门控） | 4650-4660 | UI | 失败 warn 后回退标准引擎 |
| 5. **取源：native RAW（可能解码/阻塞）→ 否则 proxy** | 4665-4669 | **UI** | `native_raw_source` 内部 `decode_gates.heavy.acquire`（Condvar） |
| 6. 计算输出尺寸 | 4670-4679 | UI | `FrameMap::frame_size` → `recipe::output_size`；**把 `recipe.scale` 改成 `Fixed100`** |
| 7. 估算内存 + 插入 `Queued` | 4681-4682 | UI | |
| 8. `std::thread::spawn` | 4683 | 新线程 | |
| 9. `ExportScheduler::acquire_cancellable` | 4686 | 工作线程 | 槽位 + 内存预算；可取消 |
| 10. 标 `Running{0.05}` | 4689-4694 | 工作线程 | |
| 11. staging 文件 + 渲染（`process::render` / `render16`） | 4701-4757 | 工作线程 | 全尺寸渲染 |
| 12. 写导出：resize → sharpen → ICC → watermark → encode → metadata | 4702-4731 | 工作线程 | `recipe::write_export*` |
| 13. `publish_gate` + 最终 cancel 检查 + `staged.publish` | 4762-4767 | 工作线程 | 原子改名 |
| 14. 写终态 | 4769-4787 | 工作线程 | 取消后不覆盖 |

### 6.2 `recipe.rs` 内部顺序

`write_export`（`recipe.rs:1422`）/`write_export16`（1438）→ `write_export_pixels`（1648）：

`checked_len` → `create_dir_all` →（16-bit 先收窄 8-bit）→ `output_size` → 重采样（16-bit `resize_rgba16` 并行 1680-1690；8-bit `image::imageops::resize`）→ 输出锐化 → ICC（`icc::convert_*`）→ watermark → 读 ICC bytes → EXIF/XMP 策略 → 编码 → 返回路径。metadata 在编码阶段嵌入，不在渲染前。[R]

### 6.3 资源准入是否覆盖 native RAW 分配？

**否。** [R]

- `ExportScheduler`（`file_engine.rs:1438-1500`）只对**输出尺寸估算**（`estimated_export_bytes`，`bit_depth` 24/36 B/px，`file_engine.rs:1417-1425`）收费，且在 spawn 之后。
- `native_raw_source`（`file_engine.rs:2695-2782`）在 **spawn 之前**在调用线程分配并缓存一整帧 native RAW；其准入是 `decode_gates.heavy`（与预览共享，单位基于 `estimated_decode_bytes`），不是导出预算。
- 结果：导出内存峰值 = native 源（UI 线程，先发生）+ 导出调度器允许的渲染/编码缓冲；两个预算互不知情。`ExportScheduler` 的 `active_slots == 0` 特例（`file_engine.rs:1478-1481`）还允许单任务超预算。
- native RAW 失败时回退 proxy（§0 F3），不改变 `JobStatus`。

**补充（重要）**：`self.source()` 对所有非 native 路径都做 `fit_long_side(..., SOURCE_LONG_SIDE=4096)`（`file_engine.rs:472-481,2627-2631`）；`raw16_source` 也以 `get_or_decode_source16(path, SOURCE_LONG_SIDE)` 取 4096 proxy（`file_engine.rs:649`）。而 `start_export` 的 `source_size` 在回退分支用 `(info.width, info.height)`（完整尺寸，`file_engine.rs:4669`），`process::render` 按 `max_size` 定输出长边（`mock/process.rs:1889-1893`）。因此：
1. 非 RAW（JPEG/TIFF/PNG/HEIF）在超过 4096 长边时，导出会把 4096 proxy 放大到 `info.width/height` 再写文件；
2. RAW native 解码失败时同样放大。
两者都不会在 `JobStatus` 或 UI 中体现。该结论由代码路径推出；未做同图对拍，故画质差异标记 [U]。

### 6.4 发布原子性

`StagedExport::new` 在目标卷建 staging；写完后在 `publish_gate` 下做最终 cancel 检查再 `publish`（改名）。`cancel_job` 取同一 gate，保证“取消返回 true ⇒ 未发布；发布先到 ⇒ cancel 返回 false”。[R] `file_engine.rs:4701-4707,4762-4767,4800-4814`

---

## 7. 任务 / gate / registry 清单

| 机制 | 位置 | 生命周期 | 取消令牌 | 排序/generation | 旧结果能否覆盖新结果 |
|---|---|---|---|---|---|
| `Preview` 单飞 | `app/state/render.rs:274-331` | 每次请求；pending 只留最新 | 无显式取消；靠 generation + `may_install` | `generation: u64` 单调；`stamp.id` 比较 | 否：同图旧 id 被拒；异图被拒 [R] |
| `thumb_queue`/`thumb_task`/`thumb_rev` | `render.rs:1146-1222` | 单队列、单任务 | `thumb_rev` 变更即退出循环 | rev 相等才写缓存 | 否 [R] |
| `jobs: Vec<JobView>` + `poll_jobs` | `app/state/mod.rs:561,1891-1912` | 每次导出；UI 120 ms 轮询 | `planned::cancel_job` → 引擎 | `job_id` 键 | 否：按 id 取状态 [R] |
| 引擎 `jobs: HashMap<u64, JobStatus>` | `file_engine.rs:1598` | 引擎生命周期，永不清理 | `cancelled: HashSet<u64>`（1599-1600） | 按 `job_id` | 否：单 worker 同 id；cancel 后 worker 不再写（`4770`） [R] |
| `cancelled` 集合 | `file_engine.rs:1600` | 进程内永久 | — | — | 决定 worker 最终是否写终态 [R] |
| `ExportScheduler` | `file_engine.rs:1438-1510` | 全局 `OnceLock` | `acquire_cancellable` 的闭包 | FIFO ticket + 内存预算 | 不影响结果内容 |
| `DecodeGate` heavy/light | `file_engine.rs:1225-1405` | 引擎生命周期 | RAII permit（drop 释放） | 优先级 Interactive < Thumbnail，同优先 FIFO | 不适用 |
| `SourceKey` flight | `file_engine.rs:1530-1610` | 一次解码；leader 结束移除 | `FlightLeader::Drop` 发布错误 | 每 key 单飞 | 不适用 |
| `MergeRegistry` | `engine/merge_async.rs:174-211` | 进程内 HashMap，不清理 | `AtomicBool cancel` | `MergeJobId` 自增 | `result` 只写一次（`MergeJob::finish`） [R] |
| `SaveOrder` | `file_engine.rs:1618-1642,1649` | 引擎生命周期 | — | 每路径序号；旧 seq 被拒 | **否：旧写被标记 Superseded** [R] |
| `source_cache`/`native_raw` LRU | `file_engine.rs:1590-1597,2695-2782` | 引擎生命周期 | — | tick/大小预算 | 不适用 |

**结论**：预览、缩略图、保存、导出各自都有 stale guard；目前未发现“旧结果覆盖新结果”的直接路径。风险集中在**导出源回退的静默降质**与**UI 同步 I/O**，而不是竞态。

---

## 8. 边界错误处理

### 8.1 隐藏失败的位置

| 位置 | 行 | 行为 | 风险 |
|---|---:|---|---|
| `after_commit_many` | `app/state/mod.rs:1206-1213` | 失败只 `notify` 一次；内存编辑保留、undo 已前进、不重试 | 用户可能在没有落盘的情况下继续编辑 |
| `preview` 后台错误 | `app/state/render.rs:1081-1085` | `log::error!`；仅当前图设置 `preview.error` | 非当前图失败完全静默 |
| `source_status` | `app/state/render.rs:1045` | warn + 归类为 3 种提示 | 归类靠字符串包含 |
| `thumbnail` 失败 | `app/state/render.rs:1194-1196` | `thumb_failed` 集合 + error log | UI 只显示占位，无原因 |
| `job_status` 未知 id | `crates/engine/src/file_engine.rs:4796-4798` | 返回 `Failed{"unknown job"}` | 无法区分“未知/已清理”与真正失败 |
| `native_raw_source` 失败 | `file_engine.rs:2759` | warn 后返回 `None`；导出回退 proxy | 静默降质（F3） |
| `persist_batch` 单文件失败 | `file_engine.rs:3302-3310` | warn + 收集到 SaveReport | App 只看到汇总文本 |
| AI 掩码失败 | `app/state/mod.rs:1715-1719,1744` | `ai_unavailable.insert(kind)` + “kind: err” 字符串 | 不区分无模型/无运行时/资源不足/平台不支持 |
| 眼睛检查失败 | `app/state/analysis.rs:121-130` | 只保留 first error 文本 | 同上 |
| `planned::merge_*` | `app/planned.rs:77-87` | `MergeError` 被 `?` 转字符串 | 结构化 `MergeErrorCode` 丢失 |
| `import` 失败 | `app/state/mod.rs:1026-1032` | `ImportErrorKind` 分类 | 已做得较好；作为对照 |
| `ExportErrorKind` | `app/tools/output.rs:375-379` | Cancelled/Unsupported/Io/Other | 已做得较好；但无“资源不足/设备丢失/内部不变量” |

### 8.2 UI 能否区分七类错误？

| 类别 | 导出 | 预览 | AI 掩码 | 导入 | 合并 |
|---|---|---|---|---|---|
| 用户输入错误 | 部分（format/bit depth） | 否 | 部分（无 face 输入） | 是 | 部分 |
| 不支持的模型/平台 | 部分（Unsupported） | 否 | **否**（只有 kind 集合） | 是（NoLibRaw/Unsupported） | 否 |
| 临时资源不足 | **否**（落入 Other） | 否 | 否 | 是（TooLarge） | 否 |
| I/O | 是（Io） | 否 | 否 | 是 | 否（字符串） |
| 设备丢失 | 否 | 否 | 否 | 否 | 否 |
| 已取消 | 是（Cancelled） | 否 | 否 | 否 | 引擎有 code，App 丢失 |
| 内部不变量 | 否 | 否 | 否 | 否 | 否 |

**建议**：在引擎边界返回结构化 `PreviewErrorKind` / `AiMaskErrorKind` / 复用 `MergeErrorCode`；`ExportErrorKind` 增加 `Resource`、`DeviceLost`、`Invariant`。不要靠 `message.contains(...)` 分类（`output.rs:120-140`）。

### 8.3 隐私

| 位置 | 行 | 输出 | 判定 |
|---|---:|---|---|
| `crates/app/src/lib.rs` | 49 | `log::error!("cannot open {}: {err}", path.display())` | **绝对路径**（用户选择的目录/目录名） |
| `crates/engine/src/file_engine.rs` | 590,604 | `EIP {}: ... {name}`, `package.display()` | **绝对路径** |
| 同上 | 792,994,997,1027,3475,3683 | `path.display()` | **绝对路径** |
| 同上 | 735,766,788,846 | C1 layer `name` | 用户层名（可能含人名/描述） |
| 同上 | 2120,2308,3301,3827 | `rel_path`（库内相对路径） | 可接受（不含根目录） |
| 同上 | 4778 | `export job {id} wrote {file_name}` | 已只输出文件名 |
| 同上 | 4782-4785 | `info.file_name` + `sanitize_log_text(err)` | 文件名 + 已脱敏错误 |
| `crates/engine/src/output.rs` | 830-860 | `log_name` / `sanitize_log_text` | 正确：去目录、掩码绝对路径 |
| `crates/app/src/state/settings.rs` | 690 | settings 路径 | 应用配置路径，风险低 |
| `crates/app/src/state/mod.rs` | 1026 | `import: {err:#}` | 错误串可能含绝对路径（取决于引擎） |

**结论**：导出路径已做隐私处理；C1/EIP/打开目录路径仍会进入 warn/info。EXIF 序列号/GPS 未在 app 层记录；引擎侧未见主动打印 GPS。未做日志文件全量扫描（[U]）。

---

## 9. 推荐的抽取顺序、最小补丁与测试

### 9.1 子状态抽取顺序（按“能证明减少更新”排序）

| 顺序 | 子状态 | 内容 | 包装 | 为什么先做 | 风险 |
|---:|---|---|---|---|---|
| 1 | `PreviewState` | `preview`, `thumbs`, `thumb_queue`, `thumb_rev`, `thumb_task`, `thumb_failed`, `source_note` | 普通 struct + 现有 `render.rs` helper；暂不做 Entity | 已经是文件级封装；异步完成方单一；已有 ordering tests | 低 |
| 2 | `JobService` | `jobs`, `next_job`, `poll_jobs` | `Entity<JobService>` + `observe` 只在状态变化时通知队列面板 | 有后台轮询；减少整树 notify 的收益可测（但本 lane 未测） | 中 |
| 3 | `EditHistory` | `Edit.undo/redo/gesture` | 纯 struct（无 GPUI） | 同步逻辑，不需要 Entity；便于文档层单测 | 低 |
| 4 | `CapabilityView` | `capabilities`, `ai_unavailable`, `engine_name`, ICC/lens 列表 | 启动读取 + 显式刷新 | 用引擎/descriptor 替换硬编码清单 | 低 |
| 5 | `ViewportDevice` | `zoom`, `viewport`, `viewport_origin`, `scale_factor`, `window_hwnd` | struct + `PlatformDisplay` trait | 隔离 HWND；跨平台必需 | 中 |
| 6 | `EditorDocument` | `edits`, `active_layer`, `active_node`, `current`（current 先留 AppState） | 最后再拆 | `current` 被多方读；过早拆会引入跨 Entity 锁 | 高 |

**GPUI Entity/observe 的使用原则**：只在“异步完成后局部重绘”能替代整树 `cx.notify()` 时使用。当前 `AppState` 的所有 `cx.notify()` 都会通知整树（`mod.rs:1059-1064` 等），但本 lane **没有测量** Entity 化前后的重绘次数，因此“减少更新”是设计假设，不是已证结论。

### 9.2 最小安全补丁（不含行为变更）

| 补丁 | 内容 | 文件（本 lane 不得改，交相应 owner） |
|---|---|---|
| P1 | 把 `Preview`/缩略图字段搬进 `PreviewState`，字段仍属 `AppState` | `crates/app/src/state/{mod,render}.rs` |
| P2 | `after_commit_many` 改用 `save_adjustments_batch_report`，把 `saved/failed/superseded/index_error` 分开展示 | 与 A-storage 协调（`mod.rs:1198`） |
| P3 | 导出取源移出 UI 线程：全分辨率 raster 解码已在 worker 内（`StandardExportSource::FullRes`）；native RAW 取源仍在 `start_export` 且未计入 native 预算，待后续 | `crates/engine/src/file_engine.rs` |
| P4 | ✅ 已实现：>4096 栅格解码全分辨率或明确失败（`JobStatus::Failed`），≤4096 不变；内存上限拒绝；测试覆盖 | `file_engine.rs` |
| P5 | 能力清单改查 `EffectRegistry` + `EngineCapabilities`；`AI_MASK` 文案由 `capabilities.ai_mask_kinds` 生成 | `contract.rs`, `tools/portrait.rs` |
| P6 | `window_hwnd: isize` → `DisplayTarget` + `PlatformDisplay` trait | `state/{mod,settings}.rs`, `ui/viewer/mod.rs` |
| P7 | 错误枚举化：`PreviewErrorKind`、`AiMaskErrorKind`、`MergeErrorCode` 透传到 App | `engine.rs`, `planned.rs`, `state/*` |

### 9.3 命令/手势不变量（应写成测试）

1. 一次 `edit_live`×N + `end_gesture` ⇒ `undo.len()` +1，`save_adjustments_batch` 调用 1 次。
2. 拖动回到原值 ⇒ undo 不增，但仍发 Final。
3. `edit_selected` N 张 ⇒ 每张 undo+1，保存调用 1 次、参数长度 N。
4. `set_current` 在拖动中 ⇒ 旧图先提交一次保存，新图预览 generation 递增。
5. `undo`/`redo` 在拖动中 ⇒ 先提交手势，再回退/前进（可能 2 次保存）。
6. 批量保存部分失败 ⇒ 内存调整保留；UI 能看到 failed/superseded/index 三类。
7. 取消导出 ⇒ 最终状态 `Failed{"cancelled"}`，不产生输出文件。
8. 关闭窗口 ⇒ 进行中的手势被提交或明确丢弃（需定义并测试）。

### 9.4 测试清单（建议落在 `crates/app` 的单测/集成；多数可脱离 UI）

| 场景 | 断言 | 现有基础 |
|---|---|---|
| slider drag | 1 undo / 1 save / Draft→Final | 可仿 `perf.rs:207-214` 的录制 |
| undo | 步数、adj 回退、保存调用 | `mod.rs` 单测风格 |
| batch edit | 每图 undo、单次批量保存、N 个缩略图入队 | 无 |
| image switch | 先提交旧图、清 preview、Final | `set_current` |
| zoom | `set_zoom` Final；`pan_by` Draft；同尺寸输出 | `render.rs:3187` 注释 |
| before/after | before 只在 compare 且 after 完成后安装；关 compare 丢弃 | `render.rs:200-220,317-324`，已有 ordering tests 1281-1722 |
| task cancel | 取消后 worker 不覆盖状态；staging 无残留 | `file_engine.rs:6638,6533` |
| >4096 栅格导出 | 输出像素来自源分辨率，不是放大的 4096 proxy；或返回明确错误 | **缺失，需新增** |
| save failure | 内存保留、通知一次、报告可区分 | `save_report_tests` `file_engine.rs:7225` |
| window close | 手势 flush 或明确丢弃 | **缺失，需新增** |
| display profile | 请求携带解析后的 profile；缓存失效后重建 | `settings.rs:390` 有 rect 测试 |

现有可复用测试：`crates/app/src/state/render.rs:1281-1722`（accept/may_install/place_frame 的排序测试，明确不覆盖 GPUI upload），`crates/engine/src/file_engine.rs:6745-6830`（decode gate 优先级/预算），`crates/engine/src/file_engine.rs:7225+`（SaveReport），`crates/engine/src/file_engine.rs:6533-6666`（staging/取消/不覆盖/隐私）。

---

## 10. 明确未验证（不得当作通过）

- **[U] UI 延迟、帧时、重绘次数**：本 lane 未测量，未运行应用，未采集 perf 记录。任何“Entity 化减少更新/更顺滑”的结论都未验证。
- **[U] 退出时手势是否落盘**：未运行 GPUI 关闭流程；源码无显式 flush。
- **[U] 导出 proxy 回退的实际画质差异**：由 `geometry_for` 的 `max_size` 推得会放大（非 RAW 与 native RAW 失败两条路径）；未做同图对拍，未测分辨率。
- **[U] 各错误类别在真机上的发生率**：未统计、未复现，不声称比例。
- **[U] macOS/Linux 显示 ICC 解析**：本机 macOS 未运行探测；`detected_display_profile` 非 Windows 恒 `None`（`settings.rs:143-147`）。
- **[U] `effect_descriptor` 播种是否覆盖全部 81 个 `WIRED_*`**：读源码为 6 个；未运行 registry 测试。
- **[U] 编译器/测试状态**：本轮未运行 `cargo check/test`；不能推断通过。
- **[U] Windows 真机导出并发/内存预算**：未在 Windows 原生 Release 验证。
