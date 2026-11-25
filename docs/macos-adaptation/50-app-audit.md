# M0/M1 — App / GPUI / 打包审计（Mac-G）

> 只读审计，基线 commit `9bf908e`（本 worktree）。
> 审计期间未运行 `cargo build/check/test`（构建锁由 Mac-A 持有），
> 因此“能否启动”的结论来自源码、Cargo.lock 与已解析的依赖源码，不是运行证据。
> 新增文件：`scripts/build-macos.sh`、`scripts/package-macos.sh`、
> `.github/workflows/macos.yml`、本文。审计轮次未修改任何既有源码。
>
> **第二轮（§10 实施记录）**：经授权写入 `crates/app/**`，并按统筹要求在
> `.github/workflows/macos.yml` 修正测试门禁与 GPU 目标；引擎/人像源码与
> Cargo manifest 仍未改。

## 0. 结论摘要（Top 8）

| # | 问题 | 位置 |
|---|---|---|
| 1 | **基线提交下无法启动**：workspace 在 aarch64-apple-darwin 上因 `aurora-engine` 的 Windows 专有装载路径/E0432 等 19 个错误无法完成编译，`portrait-app` 因此没有可链接的 Release 二进制。审计期间 `git status` 显示 Mac-A 正在修改这些 engine 文件（未提交），本审计未运行构建，不能判断是否已经修绿。 | `docs/macos-adaptation/01-blockers.md:7-52`；典型点 `crates/engine/src/mock/gpu_runtime.rs:21`、`crates/engine/src/jxl.rs:2`、`crates/engine/src/raw.rs:1453` |
| 2 | **快捷键全部是 `ctrl-*`，不是 `cmd-*`**；GPUI 的 `platform` 修饰键在 macOS 是 Command。默认 Cmd+Q/Cmd+Z/Cmd+A 等不会触发应用动作；原生菜单也没有 About/Hide/Quit/Services/Window 应用菜单。 | `crates/app/src/actions.rs:144-208`、`crates/app/src/actions.rs:287-450`、`crates/app/src/actions.rs:265` |
| 3 | **macOS 显示器 ICC 自动检测缺失**：`detected_display_profile*` 仅 Windows 实现，macOS 恒返回 `None`，预览按 sRGB 处理。 | `crates/app/src/state/settings.rs:130-160`；调用点 `crates/app/src/state/render.rs:854`、`:1171`；`crates/app/src/ui/prefs.rs:359` |
| 4 | **平台目录不符合 macOS 规则**：设置走 `~/.config`，缓存走 `~/.cache`，日志只写 stderr（Finder 启动后不可见）。 | `crates/app/src/state/settings.rs:622-638`、`crates/engine/src/raw_cache.rs:85-114`、`crates/engine/src/aimask/ort.rs:553-578`、`crates/app/src/lib.rs:144-166` |
| 5 | **Capture One 资产只找 Windows 安装目录**（ICC/胶片曲线/coreConfig/AIModels/lens DB/presets）。macOS 上这些能力会以“模型/配置不可用”降级，而不是读取 Mac 版 C1 数据。 | `crates/engine/src/icc.rs:675,703`、`filmcurve.rs:141-148`、`camera_curve.rs:69-77`、`ai_denoise.rs:29-31`、`aimask/body_pose.rs:35-44`、`lensdb.rs:380-386`、`presets.rs:579-584` |
| 6 | **Finder 打开/拖入未接线**：入口只解析 argv；GPUI 已提供 `on_open_urls`/文件拖放事件，但应用没有注册，Info.plist 也没有文档类型。 | `crates/app/src/main.rs:11-19`、`crates/app/src/lib.rs:32-55`；GPUI 侧 `gpui-pre-macos-0.3.11/src/platform.rs:136,859` |
| 7 | **Retina/DPI 数学基本正确但语义标注错误、跨屏未验证**：viewport 实际是 logical points，注释写成 device pixels；`RenderImage.scale_factor` 固定 1.0，靠元素显式尺寸回推。 | `crates/app/src/state/mod.rs:527`、`crates/app/src/state/render.rs:802-826`、`crates/app/src/state/render.rs:115`、`crates/app/src/ui/viewer/mod.rs:656-688` |
| 8 | **显示交接是 CPU BGRA → GPUI 内部 MTLTexture，无外部纹理 API**。不能声称零拷贝；真正的 MTLTexture/IOSurface 交接需要改 GPUI 公共接口。 | `crates/app/src/state/render.rs:73-118`；`gpui-pre-apple-0.3.11/src/metal_atlas.rs:163-180`；`gpui-pre-0.3.11/src/assets.rs:43-57` |

补充：`build.rs` 与 `windows_subsystem` 在 macOS **安全/无副作用**（见 §5）；
`license = "MIT"` 与 `AGENTS.md` 的 `Proprietary` 冲突仍未裁决，本 lane 未改。

---

## 1. 启动、窗口、菜单、快捷键、全屏、IME

### 1.1 入口与 GPUI 后端

- 二进制入口 `crates/app/src/main.rs:6-30`：解析 `--touch`、`--gpu-selfcheck` 和路径参数，
  然后 `portrait_app::run(gpui_kit::application(), …)`（`main.rs:29`）。
- `gpui_kit::application()` 在桌面平台直接选 `gpui_platform::application()`
  （`gpui-kit-0.9.0/src/lib.rs` 的 `platform` re-export / `pub use platform::application`），
  macOS 目标由 `gpui-pre-macos` + `gpui-pre-apple` 实现。
- `gpui-pre-macos` 是完整的 AppKit host：
  - 应用类/委托、激活策略：`gpui-pre-macos-0.3.11/src/platform.rs:67-150`、`:546-658`
  - `NSMenu`/`NSMenuItem` 原生菜单栏：`platform.rs:272-470`、`:1142`
  - `NSOpenPanel`/`NSSavePanel`：`platform.rs`（导入见 `NSOpenPanel`）
  - 原生剪贴板：`src/pasteboard.rs`
  - 输入法：注册 `NSTextInputClient` 与 `insertText:`/`setMarkedText:` 等
    （`src/window.rs:211-239`），因此**平台层具备 IME 能力**。
  - 全屏：`windowWillEnterFullScreen:` / `windowDidExitFullScreen:` 等
    （`src/window.rs:434-447`、`:900-920`）
  - Retina/display link/Metal renderer：`gpui-pre-apple`（Metal 渲染见 §4）

**判断**：GPUI 层不是“Windows-only”。一旦 Mac-A 把引擎编译修绿，
应用具备在 macOS 打开原生窗口的技术路径；**但当前没有可运行产物**。

### 1.2 窗口与标题栏

- 主窗口：`crates/app/src/ui/workspace.rs:835-851`，1600×1000 居中，最小 1100×700，
  使用 `TitleBar::window_options()`（透明标题栏、`app_owns_titlebar_drag: true`）。
- 标题栏由 `gpui-component` 自绘：macOS 留出 80 px 左内边距给红绿灯
  （`gpui-component-0.9.0/src/title_bar.rs:16-20`），并且不画 Windows 式控制按钮
  （`title_bar.rs:256-262`）。应用自己的标题栏在 `workspace.rs:591-651`，
  把应用内菜单条塞进左侧。
- 应用在启动时 `cx.activate(true)`（`crates/app/src/lib.rs:66`）。

风险（未验证）：应用内菜单条与原生菜单栏并存，macOS 上可能出现“双菜单”；
自绘标题栏与交通灯/全屏过渡的实际对齐需真机截图验证。

### 1.3 菜单与 Cmd 快捷键（重点问题）

- 快捷键表：`crates/app/src/actions.rs:133-236`。默认值全部是 Capture One Windows 布局：
  `ctrl-q`、`ctrl-shift-i`、`ctrl-d`、`ctrl-z`、`ctrl-a`、`f11` 等。
- GPUI 的修饰键语义（`gpui-pre-macos-0.3.11/src/events.rs:94-108`）：
  `modifiers.platform` = Command，`modifiers.control` = Control。
  `KeyBinding::load("ctrl-q")` 绑定的是 **Control+Q**；macOS 用户期望 **Cmd+Q**。
- 因此：默认配置下 **Cmd+Q 不退出**、Cmd+Z 不撤销、Cmd+A 不全选（应用动作）；
  Control+Q 才会命中。
- 原生菜单项的关键等价键由 keymap 反查（`gpui-pre-macos .../platform.rs:360-425`）：
  由于默认绑定是 `control`，菜单会显示 ⌃Q，而不是 ⌘Q。
- 菜单列表 `actions.rs:287-450` 以 **File** 开头，没有 macOS 应用菜单
  （About/Services/Hide/Hide Others/Show All/Quit）。`create_menu_bar`
  （`gpui-pre-macos .../platform.rs:272-306`）不会自动插入；结果 macOS 会把
  第一个菜单放进应用菜单位，标题显示 “File”，标准项缺失。
- `display_keys` 把 `cmd|super|win` 一律显示成 **“Win”**（`actions.rs:255-266`），
  在 macOS 快捷键编辑器里会出现 “Win+Q” 这类错误标签。
- 用户可在偏好设置里录制 Cmd 组合（`crates/app/src/ui/prefs.rs:115-143` 会写入
  `cmd`），绑定本身能用；问题是**默认值和显示标签**没有平台化。

若只做最小修复，建议（不在本次写入范围）：
1. 新增平台默认绑定表：macOS 用 `cmd-*`，Windows/Linux 保持 `ctrl-*`；
2. 菜单列表按平台插入应用菜单（About/Preferences/Quit）与 Window 菜单；
3. `display_keys` 在 macOS 输出 `⌘`。
4. 不要简单把所有 `ctrl` 全局替换为 `cmd`：`ctrl-alt-*`、`alt-*` 等
   Capture One 快捷键在 Mac 上需要逐条确认，避免与 macOS 系统快捷键冲突。

### 1.4 全屏

- 只有 F11 绑定与 View 菜单项：`actions.rs:206`、`actions.rs:423`；
  处理函数 `ui/workspace.rs:441` 调 `window.toggle_fullscreen()`。
- GPUI macOS 支持原生全屏（`gpui-pre-macos .../window.rs:434-447`），
  绿色交通灯也应可用；但 F11 不是 macOS 惯例，Ctrl+Cmd+F 没有绑定。
  实际绿色按钮/F11/退出全屏的状态一致性**未验证**。

### 1.5 输入法与键盘上下文

- 应用用 `KeyBindingContextPredicate::parse("Workspace && !Input")`
  （`actions.rs:241`）让快捷键在文本输入获得焦点时让路，这是 IME 友好的。
- 平台实现 `NSTextInputClient`（`gpui-pre-macos .../window.rs:211-239`），
  中文/日文 marked text 有实现。
- 但**没有在真机验证**：中文输入法组字、候选窗定位、文件名输入、面板内快捷键冲突。
  CI 的 headless 测试不能证明这一点。

### 1.6 Finder 打开与拖放

- `main.rs:11-19` 只读取进程 argv。macOS 双击关联文件走的是
  `application:openURLs:`，不是 argv。
- GPUI 提供 `Platform::on_open_urls`（`gpui-pre-macos .../platform.rs:859,1484`），
  `gpui-pre` 的 `App` 层有对应入口；**应用没有注册**，`crates/app/src/lib.rs:32-55`
  也没有处理。
- 应用内的 `.on_drop`（`ui/browser.rs:195`、`tools/library.rs:80`、
  `ui/inspector.rs:135`）只接收应用自定义 drag payload，不是 Finder 的外部文件。
  GPUI macOS 支持外部拖放事件（`gpui-pre-macos .../window.rs:473-497`），
  但应用没有把 `FileDropEvent`/`ExternalPaths` 接到 import。
- 结论：**“双击图片用 Aurora 打开”和“从 Finder 拖照片进窗口”目前都未实现**。
  另外 Info.plist 需要 `CFBundleDocumentTypes`/`UTImportedTypeDeclarations`
  才能成为关联应用；本次生成的开发 bundle 有意没有声明（应用尚不能处理事件）。

---

## 2. 平台目录（settings / cache / logs / catalog）

| 用途 | 当前 macOS 实际位置 | 证据 | macOS 正确位置 |
|---|---|---|---|
| 设置/工作区/最近项目 | `$AURORA_SETTINGS`，否则 `$XDG_CONFIG_HOME/aurora-studio`，否则 `~/.config/aurora-studio/settings.json` | `crates/app/src/state/settings.rs:622-638`；最近项目也在 settings JSON 内 `:649-655` | `~/Library/Application Support/Aurora Studio/settings.json` |
| “旧名迁移” | `legacy_path()` 与 `path()` 是同一个目录，迁移为空操作 | `settings.rs:641-643` 与 `:622-627` | 迁移旧名时应指向旧的 `~/Library/Application Support/Aurora Studio` |
| RAW 16-bit 缓存 | `$AURORA_RAW_CACHE_DIR`，否则 `$XDG_CACHE_HOME/aurora-studio/raw`，否则 `~/.cache/aurora-studio/raw` | `crates/engine/src/raw_cache.rs:85-114` | `~/Library/Caches/Aurora Studio/raw` |
| ONNX Runtime 优化图缓存 | 同上，`.../aurora-studio/onnxruntime` | `crates/engine/src/aimask/ort.rs:553-578` | `~/Library/Caches/Aurora Studio/onnxruntime` |
| 日志 | 只有 stderr；Finder 双击启动后通常不可见 | `crates/app/src/lib.rs:144-166` | `~/Library/Logs/Aurora Studio/` 或 `os_log`（`Logger`/`OSLog`） |
| 导出临时文件 | `std::env::temp_dir()` | `crates/app/src/ui/export.rs:493` | 可接受（`NSTemporaryDirectory` 语义）；原子替换逻辑与平台无关 |
| C1 recipes | 只读 `%LOCALAPPDATA%\CaptureOne\Recipes165`，macOS 无 fallback | `crates/engine/src/recipe.rs:2179-2185` | `~/Library/Application Support/Capture One/Recipes165` 或 `/Library/Application Support/Capture One/...`（需真机确认） |
| 导入默认目录 | `USERPROFILE` → `HOME`，再找 `Pictures` / `图片` | `crates/app/src/ui/import.rs:523-529` | `~/Pictures` 可用；`HOME` fallback 已存在，不需要改 |

补充：

- 应用不把 catalog/session、sidecar、导出目标写进 `.app`；`Source::Catalog/Folder`
  只是用户选定路径（`crates/app/src/state/source.rs:11-59`）。这是正确的边界。
- 不要用 `dirs`/`directories` 新依赖也可以实现：用 `std::env::var_os("HOME")`
  拼 `Library/...` 即可；但更稳妥的是使用 `NSSearchPathForDirectoriesInDomains`
  （Foundation）或 `NSFileManager.URLsForDirectory:inDomains:`，以正确处理沙盒/本地化。
  现有 `std::fs` 路径拼接可以工作，但不是 Apple 推荐 API。

---

## 3. 色彩与显示

### 3.1 已存在的 macOS 分支

- `display_profiles()` 已有 macOS 分支：`/Library/ColorSync/Profiles` 和
  `$HOME/Library/ColorSync/Profiles`（`crates/app/src/state/settings.rs:101-126`），
  扩展名过滤 `.icc/.icm`，按文件名排序去重。这个列表可用于手动选择 profile。
- 但**自动检测当前显示器 profile** 的函数是 Windows-only：
  - `detected_display_profile(hwnd)`：非 Windows 直接 `None`
    （`settings.rs:130-140`）。
  - `detected_display_profile_status(hwnd)`：非 Windows 返回 `{None,false}`
    （`settings.rs:149-160`）。
  - `invalidate_display_profile_cache()`：非 Windows 空操作（`settings.rs:162-166`）。
- 窗口句柄在 macOS 永远是 0：`ui/viewer/mod.rs:1213-1226` 只在 `cfg(windows)`
  读 `RawWindowHandle::Win32`，`#[cfg(not(windows))] let hwnd = 0;`。
- 调用链：
  - 预览：`state/render.rs:854` 用 `settings.display_profile` 或
    `detected_display_profile(window_hwnd)` 填 `PreviewRequest.display_profile`。
  - 缩略图：`state/render.rs:1171` 同样。
  - 偏好设置显示检测结果：`ui/prefs.rs:359`。
- 结果：macOS 上自动检测永远失败，若用户不手动选 profile，引擎按 sRGB 预览；
  这与显示器是否 P3/广色域无关，只是“没有做显示 profile 转换”。

### 3.2 macOS 应如何检测（建议，未实现）

不新增 crate 的前提下，最干净的方案是在 `state/settings.rs` 的 `win_icm`
旁边加一个平台 FFI 模块：

1. `CGMainDisplayID()` / `CGDisplayCopyDisplayMode` 确定当前窗口所在显示器；
   多显示器需要窗口坐标 → `CGGetDisplaysWithPoint` 或在 GPUI 层拿到 `display_id`。
2. `ColorSyncProfileCreateWithDisplayID(display_id, &error)` 得到
   `ColorSyncProfileRef`；
3. `ColorSyncProfileGetURL(profile, &error)` 得到 `CFURLRef`，再
   `CFURLCopyFileSystemPath` → `PathBuf`；
4. 失败时回退到 `CGDisplayCopyColorSpace` + ColorSync 描述，再回退到
   `~/Library/ColorSync/Profiles` 的默认 profile；
5. 缓存需要按 `CGDirectDisplayID` 和显示器配置变更（`CGDisplayRegisterReconfigurationCallback`）
   失效，而不是只在窗口 resize 时失效。

注意：`ColorSyncProfileCreateWithDisplayID`/`ColorSyncProfileGetURL` 属于
ColorSync.framework；`CGDisplayCopyColorSpace` 属于 CoreGraphics。两者都是系统
库，不需要 bundled dylib。实现时用 `#[link(name = "ColorSync")]` +
`#[link(name = "CoreGraphics")]` 或直接用 `objc2-core-graphics`（如果 Mac-B
后续引入 objc2 依赖）。**本审计不新增依赖。**

### 3.3 与 GPUI surface 的边界（重要，勿误判）

- GPUI 的 `WindowOptions.use_display_p3` 默认 `false`
  （`gpui-pre-0.3.11/src/platform.rs:2818`）。
- macOS Metal renderer 只有在 `use_display_p3 == true` 时才把
  `CAMetalLayer.colorspace` 设为 Display P3
  （`gpui-pre-apple-0.3.11/src/metal_renderer.rs:201-209`）；
  pixel format 始终是 `BGRA8Unorm`，不是 `BGRA8Unorm_sRGB`。
- 应用没有设置 `use_display_p3`（`ui/workspace.rs:835-838` 只设 bounds/min size/titlebar）。
- 因此当前管线是：引擎按 `display_profile`（macOS 上为 None → sRGB）输出 8-bit BGRA
  → GPUI 上传到无显式 colorspace / sRGB 语义的 layer。
  在 P3 屏上 WindowServer 是否会把它当 sRGB 还是 display-native 解释，
  **本机未做带标签测试图的验证**。不要在未验证前同时做“显示 ICC 转换 + 改 layer 色彩空间”，
  否则可能双重转换。
- 正确顺序应是：先确定 GPUI surface 的实际色彩空间，再决定预览是否转换到该空间。
  这需要 GPUI 的一个最小选项（显式设置 layer colorspace）或平台侧 shim；属于
  Mac-B/主控协调范围。
- **不因为显示器是 P3 就修改导出色彩空间**。导出颜色空间由
  `ProcessRecipe`/`ExportFormat`/ICC 决定，与 `display_profile` 无关；
  `display_profile` 只出现在 `PreviewRequest`（`state/render.rs:854`）。

---

## 4. Retina / DPI 与帧呈现

### 4.1 逻辑点 vs 物理像素

- GPUI 的 `Pixels` 是**逻辑点**（`gpui-pre-0.3.11/src/window.rs:3153` 明确写
  “viewport in window-local logical pixels”）。
- `AppState.viewport` 保存的是 viewer 的 logical size，但注释写成了
  “device pixels”（`state/mod.rs:527`）。这是**注释错误**，不是实现错误。
- 每帧 `on_prepaint_viewport` 从 `bounds.size` 和 `window.scale_factor()`
  更新 viewport（`ui/viewer/mod.rs:1213-1239`），scale factor 变化会触发一次
  `request_render(Final)`（`state/mod.rs:1739-1750`）。
- 预览请求分辨率：
  - `fit_scale = (viewport_logical * scale_factor) / frame_px`
    （`state/render.rs:802-806`）——device px / frame px，数学正确。
  - `visible_region` 同样乘 `scale_factor`（`state/render.rs:819-826`）。
  - `max_size` 上限 8192（`state/render.rs:841-844`），Retina 4K/6K 视口仍有余量。
- 绘制：
  - `target_s = effective_scale() / scale_factor` 得到 logical px / frame px
    （`ui/viewer/mod.rs:656`）。
  - 图片元素用显式 `w/h` 逻辑尺寸 + `object_fit(Fill)`
    （`ui/viewer/mod.rs:680-688`、`:716-720`）。
  - `RenderImage.scale_factor` 始终是 1.0（`state/render.rs:111-118`），
    因为绘制完全由元素尺寸决定；这能工作，但依赖隐式约定。
- 结论：100% 时是“1 图像像素 = 1 物理像素”，fit 时按视口物理尺寸请求；
  没有发现“先缩后上采样”的路径。**但没有真机 Retina/外接屏切换的像素验证**。

### 4.2 风险点

- 注释错误会误导后续优化：`viewport` 不是 device px。
- 跨屏移动（不同 scale factor）会重新请求 Final，但显示器颜色 profile
  不会随之失效（macOS 上 `invalidate_display_profile_cache` 是空操作）。
- 100% 缩放时 `effective_scale = scale_factor`，在 2x 屏上请求 2× 图像像素；
  这是正确的 1:1 物理像素语义，但需要确认用户预期“100%”是逻辑 100% 还是物理 1:1。
- 大图 + 2x + 8192 上限：极端情况会被 clamp，需记录是否影响画质（当前未验证）。

### 4.3 Metal / GPUI 纹理交接

- `RenderImage` 是纯 CPU 数据：`data: SmallVec<[image::Frame; 1]>`
  （`gpui-pre-0.3.11/src/assets.rs:43-57`），接口是 `as_bytes()`。
- `into_bgra_pixels` 做 RGBA→BGRA（`state/render.rs:73-105`），
  然后 `RenderImage::new(vec![image::Frame::new(...)])`（`:115`）。
- GPUI Apple 后端为每个 image 分配 `MTLTextureDescriptor`，格式 `BGRA8Unorm`，
  storage mode 在支持时用 shared，然后 `upload()` 复制像素
  （`gpui-pre-apple-0.3.11/src/metal_atlas.rs:134-180`）。
- **没有“导入外部 MTLTexture/IOSurface”的公共 API**（审计了
  `RenderImage`、`RenderImageParams`、metal atlas、macOS window）。
- 因此：
  - 当前是“CPU BGRA buffer → GPU texture 复制”，不是零拷贝；
  - 要做到引擎 Metal 直接出纹理再由 GPUI 显示，需要给 GPUI 增加外部纹理
    handle/生命周期接口（in-flight 完成、resize、旧帧释放），属于公共接口变更；
  - 在接口存在前，合理的中间方案是有界 staging + 复用 BGRA buffer +
    异步 readback，减少同步等待，但**不要声称零拷贝**；
  - 保留 CPU-only 渲染入口。
- Histogram/waveform/parade/vectorscope/警告/focus mask 目前在 CPU 侧与图像并行
  （`state/render.rs:989-998`，`rayon::join`），符合“辅助分析不阻塞主预览”的方向；
  但它们仍与 generation 绑定，需继续保证旧帧不回写。

---

## 5. 打包前置条件

### 5.1 名称与产物

- package：`portrait-app`（`crates/app/Cargo.toml:2`）。
- binary：`aurora-studio`（`crates/app/Cargo.toml:12-14`），路径
  `target/<target>/release/aurora-studio`。
- `dist/` 已被 `.gitignore` 忽略（`.gitignore` 的 `/dist/`）。

### 5.2 build.rs 与 `windows_subsystem` 安全性

- `crates/app/build.rs:1-6` 无条件调用 `embed_resource::compile("app.rc", NONE)`。
- `embed-resource 3.0.12` 在非 Windows 目标的 `ResourceCompiler::probe()` 返回
  空错误串，`compile_impl` 返回 `CompilationResult::NotWindows`，
  外层 `try_compile_impl!` 直接返回该值；随后 `.manifest_optional()` 对
  `NotWindows` 返回 `Ok(())`。所以 `.expect("embed app.rc …")` **不会 panic**。
  证据：`embed-resource-3.0.12/src/lib.rs:567-585,330-360,379-386`。
- `#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]`
  （`crates/app/src/main.rs:1`）：该属性只在 Windows 链接时生效，macOS 上被接受且无影响；
  不会隐藏 stdout/stderr 之外的行为差异。真正的 macOS 问题是应用自身日志只写 stderr（§2）。

### 5.3 资源与动态库

- 运行时资源已嵌入二进制：
  - 主题 `include_str!("../assets/aurora-theme.json")`（`crates/app/src/theme.rs:129`）；
  - 引擎相机参数/人像数据 `include_str!`/`include_bytes!`
    （如 `crates/engine/src/camera_params.rs:21-24`、
    `crates/engine/src/portrait_v2_mesh.rs:19-32`）；
  - GPUI Kit/Ely 资源通过 `rust-embed`（`gpui-kit-assets-0.9.0/Cargo.toml:98-103`）。
- 所以 `.app/Contents/Resources` 目前只需放 LICENSE/第三方许可，不需要资源树；
  本次脚本会复制 `LICENSE` 和 `ASSET-LICENSES.md`。
- 当前 macOS 构建没有链接第三方 dylib：仓库内 `#[link(name = …)]` 与
  `LoadLibrary`/`kernel32` 都在 `cfg(windows)` 下（如 `perf.rs:113-140`、
  `settings.rs:195-210`、`engine` 的 `library_ops/file_engine/tether/print/heif`）。
  libraw/ONNX Runtime/libheif/libjxl 是**运行时 dlopen/动态装载**（且 macOS 分支尚在阻塞清单）。
- 若 Mac-A/D/E 后续启用 macOS dlopen，发布包必须：
  - 把 dylib 放进 `Contents/Frameworks`，install name 用 `@rpath/...`；
  - 在构建时设置 `-Wl,-rpath,@executable_path/../Frameworks`（或等价）；
  - 不写死 `/opt/homebrew`、`/usr/local`、用户名；
  - 由 `scripts/package-macos.sh` 的 `otool -L` 检查兜底（非系统且未打包 → 失败）。

### 5.4 现有约定与 CI

- `README.md` 只有 Windows 运行/构建说明；没有 macOS 章节。
- `scripts/` 已有 bash 脚本（`shot.sh`、`tour.sh`）和 PowerShell 脚本；
  新增 bash 与现有习惯一致。
- 仓库没有 Info.plist、图标或 `.entitlements`；本次生成的 Info.plist 是本地开发配置。
- `.github/workflows/ci.yml` 的 jobs：windows（`:23-26`）、supply-chain（`:93`）、
  sbom（`:119`）、nightly-audit（`:155`）。**未改动该文件。**
  新增 `.github/workflows/macos.yml` 独立运行，不影响 Windows job。
- 新增 workflow 使用 `macos-14`，记录 `uname -m`/`runner.arch`；
  若 runner 不是 arm64，则 check 仍交叉、测试跑宿主架构、release 仍构建
  `aarch64-apple-darwin`，并在 summary 里明确标注。
- Metal GPU 测试全部是 `#[ignore = "requires macOS Metal device"]`
  （如 `crates/portrait/tests/metal_runtime.rs:57`、
  `metal_mesh_runtime.rs:227`、`metal_smile.rs:243`、`metal_asset.rs:134`、
  `metal_skin_assets.rs:256`）。默认 `cargo test` 把它们计为 ignored，不是 pass；
  workflow 只有在 `MTLCreateSystemDefaultDevice()` 非 nil 时才用 `--ignored` 真跑，
  否则输出跳过并明确“不计为通过”。

### 5.5 签名与分发边界

- 本 lane 不持有 Developer ID，未生成证书，未 notarize。
- `scripts/build-macos.sh` 产出未签名 `.app`，bundle id 固定为
  `local.aurora-studio.dev`，并在脚本与 Info.plist 注释里标明“本地开发”。
- `scripts/package-macos.sh` 只做 ad-hoc `codesign -s -`、`codesign --verify`
  和 zip + SHA-256；不会运行 `spctl --assess`（它必然拒绝未公证包），
  也不会声称 Gatekeeper 通过。
- 正式分发仍需：Developer ID Application 证书、`codesign --options runtime`
  （如需 hardened runtime）、`notarytool` 提交与 `stapler`，以及用户的明确授权。
  这些全部**未做**。

---

## 6. `MACOSX_DEPLOYMENT_TARGET` 选择依据

选择：**12.0**（`scripts/build-macos.sh` 默认值；允许用户覆盖为更高版本，
低于 12.0 直接失败）。

依据：

1. `gpui-pre-apple 0.3.11` 的 `build.rs` 在 macOS 上把
   `src/shaders.metal` 编译为 metallib 时硬编码
   `-mmacosx-version-min=12.0`（`gpui-pre-apple-0.3.11/build.rs` 的
   `compile_metal_shaders`）。这是整个依赖栈实际声明的下限；
   把 app 的 deployment target 设得更低不会让 metallib 支持更低系统。
2. `gpui-pre-macos` 对更新的 API 做了运行时判断，而不是编译期要求：
   - URL scheme 注册要求 12.0（`platform.rs:812-818`）；
   - 窗口恢复 `NSWindowRestorationOptions` 仅 15.0+（`window.rs:344-360`、`:1608-1625`）；
   - 全屏标题栏透明处理 15.3+（`window.rs:3080-3120`）；
   - 菜单自动键等价本地化 12.0+（`platform.rs:406`）。
3. Rust 对 `aarch64-apple-darwin` 的默认最低版本是 11.0，但这不是本仓库依赖栈的
   真实下限；以 12.0 为准更保守且可解释。
4. 未使用任何确认需要 13/14/15 的编译期 API；不声称支持 11.0。

> 若未来把 gpui-pre 替换为支持更低 metallib 版本的实现，才可重新评估 11.0。
> 若某些依赖要求 13+，必须提供证据后上调，不能凭“新系统更省事”默认拉高。

---

## 7. 明确未验证项

> 本节记录审计当时的状态。其中“构建命令本身未运行”“没有可运行 `.app`”
> 已由 §11 的真机构建与启动记录覆盖；其余条目仍然有效，除非 §11 另有说明。

以下项目 **没有** 运行证据，不得写成通过：

- **Finder 双击启动**：尚未构建出可运行的 `.app`；没有 `open "dist/Aurora Studio.app"`
  的成功记录。
- **干净机器启动**：没有在未安装 Xcode/Homebrew/命令行工具的 Mac 用户环境测试；
  动态库搜索路径与资源路径未验证。
- **Notarization / Gatekeeper**：完全未做，也不声称。
- **菜单/快捷键真机行为**：Cmd 键、IME 组字、复制粘贴、Undo/Redo、
  Ctrl+Cmd+F、绿色全屏按钮、Dock/窗口恢复均未在真机逐项验证。
- **Retina / 外接显示器 / 不同缩放**：没有像素级对拍或截图证据；
  P3 显示器上的色彩空间解释未验证。
- **Metal GPU 测试**：CI 上是否有 Metal 设备未知；本地未运行；
  无设备时必须跳过且不计 pass。
- **文件对话框 / 外置盘断开 / 权限错误**：未测试。
- **Finder 拖入照片、Open With、文档类型**：代码层面未接线（见 §1.6），
  因此不是“未验证”，而是“未实现”。
- **macOS RAW/HEIF/JXL/ORT 装载**：属于 Mac-A/D/E；本审计只看到当前的 Windows-only
  装载代码与阻塞清单，未验证任何 macOS 解码路径。
- **构建命令本身**：`cargo check/test/build --target aarch64-apple-darwin`
  在审计时未运行（构建锁）。基线失败清单见 `01-blockers.md`；审计期间
  `git status` 显示 Mac-A 已改动多个 engine 文件（未提交），因此**不能把
  “19 个错误”当作当前状态**，只能当作基线状态。
- 新增脚本/workflow 的语法已本地校验（`bash -n`、YAML parse），并用一个
  stub `cargo` 走通了 `.app` 组装、`plutil`、`lipo`、`otool`、ad-hoc `codesign`
  和 zip 流程；**没有执行真实构建**，也没有验证过真实 `aurora-studio` 二进制。

---

## 8. 建议的最小后续工作（按依赖顺序）

1. **Mac-A**：清零 `01-blockers.md` 的 19 个 engine 错误；之后运行
   `cargo check/test/build --target aarch64-apple-darwin`，并执行
   `scripts/build-macos.sh` + `scripts/package-macos.sh`。
2. **Mac-G（下一轮，需授权写源码）**：
   - 平台化默认快捷键（macOS `cmd-*`）与菜单（应用菜单 + Window 菜单）；
   - `display_keys` 的 macOS 标签；
   - `state/settings.rs` 增加 ColorSync/CGDisplay 检测；
   - 平台目录改为 Application Support/Caches/Logs（保留环境变量覆盖）；
   - 接 `on_open_urls` 与 GPUI 外部文件拖放；
   - Finder 启动/Retina/IME 的真机验证清单。
3. **主控/Mac-B**：决定 GPUI surface 色彩空间与外部纹理接口；在此之前不宣称
   P3 正确、不宣称零拷贝、不宣称 Metal 预览已产品化。
4. **构建脚本/CI**：等编译绿后第一次真实跑通 `macos.yml`，记录 runner 架构与
   实际耗时；Metal 步骤在无设备时保持 skip。

## 9. 参考

- `docs/macos-adaptation/00-baseline.md`、`01-blockers.md`、`OWNERSHIP.md`、`05-plan.md`
- `AGENTS.md`（验证命令、Proprietary、外部资产边界）
- 依赖源码（cargo registry）：
  `gpui-kit-0.9.0`、`gpui-component-0.9.0`、`gpui-pre-0.3.11`、
  `gpui-pre-macos-0.3.11`、`gpui-pre-apple-0.3.11`、`embed-resource-3.0.12`

---

## 10. 实施记录（Mac-G 第二轮，app 层）

范围：只改 `crates/app/**` 和统筹明确要求的 `.github/workflows/macos.yml`。
未改 `crates/engine/**`、`crates/portrait/**`、`Cargo.toml`/`Cargo.lock`
（本轮 `git status` 中的 engine/portrait/lock 改动来自其他 lane）。许可证未动。

### 10.1 平台化快捷键与显示标签

| 位置 | 改动 | 依据 |
|---|---|---|
| `crates/app/src/actions.rs:155-221` | 默认 `ctrl-*` 改为 `secondary-*`。GPUI 的 `secondary` 解析为 macOS 的 Command、Windows/Linux 的 Control | `gpui-pre-0.3.11/src/platform/keystroke.rs:158-168`；`KeyBinding::load` 走 `parse_with_os`（`src/keymap/binding.rs:51-67`） |
| `actions.rs:182-183` | Merge HDR / Merge Panorama 保留 `ctrl-h` / `ctrl-m`，在 macOS 上让出 Cmd+H（Hide）与 Cmd+M（Minimize） | macOS 系统菜单约定 |
| `actions.rs:218` | 全屏：macOS 默认 `ctrl-cmd-f`，Windows/Linux 仍为 `f11` | macOS 全屏惯例 |
| `actions.rs:248-283` | `bind_keys` 在 macOS 追加 `cmd-h`/`cmd-alt-h`/`cmd-m`（Hide / Hide Others / Minimize 的菜单键等价） | 菜单键等价由 keymap 反查 |
| `actions.rs:286-304` | `display_keys`：`cmd|super|win` 与 `secondary` 在 macOS 显示 `⌘`，不再显示 “Win”；非 macOS 保持 “Win”/“Ctrl” | 审计 §1.3 |

Windows 保持：`secondary` 在非 macOS 解析为 `control`，所以 Windows 键位仍是 Ctrl；
`display_keys` 仍输出 Ctrl；没有新增菜单。

### 10.2 macOS 应用菜单与 Window 菜单

| 位置 | 改动 |
|---|---|
| `actions.rs` 的 `actions!` 列表 | 新增 `Hide`、`HideOthers`、`ShowAll`、`Minimize`、`ZoomWindow` |
| `actions.rs:480-502` | `macos_app_menu()`：About / Preferences / Services / Hide / Hide Others / Show All / Quit |
| `actions.rs:504-517` | `macos_window_menu()`：标题必须是字面量 `Window`（GPUI 据此调用 `setWindowsMenu:`），含 Minimize / Zoom / Full Screen |
| `actions.rs:519-541` | `install_menus` 仅在 macOS 把应用菜单放最前、Window 菜单插在 Help 前；窗口内 `AppMenuBar` 仍用原 File…Help 列表 |
| `crates/app/src/lib.rs:157-160` | 全局处理器：Hide→`cx.hide()`、HideOthers→`cx.hide_other_apps()`、ShowAll→`cx.unhide_other_apps()` |
| `crates/app/src/ui/workspace.rs:261-262` | Minimize→`window.minimize_window()`、Zoom→`window.zoom_window()` |

依据：`gpui-pre-macos-0.3.11/src/platform.rs:270-304` 不自动插入应用菜单，且
`menu_config.name == "Window"` 才注册 Window 菜单；`:324-468` 支持
`SystemMenuType::Services`，键等价从 keymap 取。

### 10.3 显示器 ICC（macOS ColorSync）

`crates/app/src/state/settings.rs:132-181`（调用入口）与 `:466-604`（`mac_display`）。

- `status()`：`CGMainDisplayID()` → `ColorSyncProfileCreateWithDisplayID` →
  `ColorSyncProfileGetURL` → `CFURLCopyFileSystemPath`，得到主显示器 ICC 路径；
  任一步失败返回 `None`。5 秒 TTL 缓存，`invalidate_display_profile_cache()`
  在 macOS 清缓存。
- FFI 签名已逐项对照 macOS SDK：
  `ColorSyncProfile.h:221` 的 `ColorSyncProfileCreateWithDisplayID(uint32_t)`
  **只有一个参数**（初稿多写的 error 参数已删除）；
  `CFURL.h:22,222`、`CFString.h:113,294,355`。符号在
  `ColorSync.tbd`/`CoreGraphics.tbd`/`CoreFoundation.tbd` 中确认存在。
- **限制**：只取主显示器，不是窗口所在显示器——GPUI 没有公开 display id。
  `detected_display_profile_status` 在 macOS 返回 `fallback = true`，偏好设置
  显示“回退主屏”。
- 不因 P3/广色域修改导出色彩空间；`display_profile` 仍只进入 `PreviewRequest`。
  审计 §3.3 的“显示 ICC + layer 色彩空间可能双重转换”风险不变。

### 10.4 平台目录

| 位置 | 改动 |
|---|---|
| `settings.rs:790-804` | `config_dir` 新增 macOS 分支 `~/Library/Application Support/Aurora Studio`；Windows（APPDATA）与 Linux（XDG_CONFIG_HOME/~/.config）不变；`AURORA_SETTINGS` 仍最优先 |
| `settings.rs:806-819` | macOS 的 `legacy_path` 指向旧位置 `~/.config/aurora-studio/settings.json`，首次启动可迁移；其他平台保持原定义 |
| `crates/app/src/lib.rs:252-323` | macOS 日志同时 append 到 `~/Library/Logs/Aurora Studio/aurora-studio.log`（逐行 flush），并保留 stderr；Windows/Linux 仍只有 stderr |

缓存目录（`~/Library/Caches/...`）由引擎解析（`crates/engine/src/raw_cache.rs`、
`crates/engine/src/aimask/ort.rs`），属于引擎 lane，本轮未改；需 Mac-A/Mac-D
在 engine 内落地。审计 §2 的缓存行仍适用。

### 10.5 Finder 打开与外部拖放

| 位置 | 改动 |
|---|---|
| `lib.rs:32-79` | `OpenUrlSink`、`file_url_to_path`（file URL / host / 百分号解码）、`percent_decode` |
| `lib.rs:82-95` | `import_opened`：`welcome = false` 后 `state.import(paths)` |
| `lib.rs:125-146` | `Application::on_open_urls` 在 `run` 之前注册；启动期 URL 先入队 |
| `lib.rs:163-174` | state 就绪后写入 `AsyncApp`/`Entity<AppState>` 并导入队列 |
| `ui/workspace.rs:834-849` | 根元素 `.on_drop::<ExternalPaths>`：Finder 拖入的文件直接导入 |
| `lib.rs:157-160` | macOS 应用菜单动作的全局处理器 |

依据：`gpui-pre-0.3.11/src/app.rs:357-365`（`on_open_urls`）、
`platform.rs` 的 `Platform::on_open_urls`；
`gpui-pre-macos-0.3.11/src/platform.rs:1484-1508`（`application:openURLs:`）；
`gpui-pre-0.3.11/src/window.rs:5899-5912`（`FileDropEvent::Entered` 转成
`ExternalPaths` 内部 drag）；`elements/div.rs:2964-2995`（按 TypeId 匹配 drop）。

仍未做：`scripts/build-macos.sh` 的 Info.plist 没有 `CFBundleDocumentTypes`，
所以“设为默认打开应用 / Finder 双击关联”仍需后续在脚本里声明。本轮按边界
未改脚本；`open -a … file` 与 Open With 已可通过 openURLs 到达应用。
（`CFBundleDocumentTypes` 已在 §11 补上。）

### 10.6 .github/workflows/macos.yml（统筹要求）

- 测试门禁改为读取 `scripts/testdata-skips.txt`：每个非注释行作为显式
  `--skip <name>` 传入，并打印应用的过滤器数量；列表为空直接失败。
  （`:78-108`）
- GPU 步骤移除 `--test metal_asset`：其两个测试读取本地专用
  `work/backend-r11/o2e/whiten_lut.f32`，干净 checkout 上没有，会假失败。
  保留 `metal_runtime`、`metal_mesh_runtime`、`metal_smile`、`metal_skin_assets`
  （本机 18/18 ignored 测试通过）。（`:126-139`）
- 删除“预期一直 red 直到 engine 编译修复”的过时头部注释，改为当前状态
  （ARM64 check 通过、`.app` 可组装）。（`:1-7`）

### 10.7 验证证据与未验证项

实际运行：

- `cargo check -p portrait-app --target aarch64-apple-darwin` → `Finished`，
  0 error；`portrait-app` 的 11 个警告均为既有 unused-import/dead-code，
  改动文件没有新增警告。
- `cargo check -p portrait-app --target aarch64-apple-darwin --all-targets`
  → `Finished`，0 error。
  （首次运行曾撞上其他 lane 正在编辑的 `crates/engine/src/metal/pool.rs`
  E0594/E0382；对方保存后重跑即为绿，非本轮改动。）
- FFI 签名/符号对照 `xcrun --show-sdk-path` 的 SDK 头与 `.tbd` 逐项确认。
- workflow：`ruby -ryaml` 解析通过；测试步骤 shell `bash -n` 通过；用系统
  bash 3.2 实跑过滤器提取，得到 106 个 `--skip`（212 个参数）。

**只有编译证据、没有真机运行证据**：

- 菜单实际渲染、Cmd 快捷键命中，Cmd+H/Cmd+M 与 Merge 的让路、Cmd+,/Cmd+Q
  的 AppKit 行为；
- `on_open_urls` 启动期队列与运行期导入、Open With / `open -a`；
- Finder 拖放的实际命中（drop 依赖根元素在 MouseUp 时 hovered）；
- ColorSync 返回路径是否可读、外接屏/合盖/热插拔后的刷新；多显示器仍取主屏；
- 日志文件创建与 Finder 启动下的可见性；
- 中文 IME 组字、菜单键等价本地化；
- Windows 目标没有在本机交叉编译（无 Windows target/SDK）；Windows 分支仅靠
  cfg 保持原路径，未做运行验证。

---

## 11. 打包与启动实施记录（Mac-G 第三轮，2026-11-25）

本轮只改 `scripts/build-macos.sh` 与本目录文档；未改 engine/portrait 源码、
`Cargo.toml`/`Cargo.lock`，未提交/推送，许可证未动。以下命令全部在本机
（macOS 27.2.1 / arm64，`Mac14,2`）实际运行。

### 11.1 Info.plist：Finder 文档关联

`scripts/build-macos.sh` 生成的 Info.plist 新增 `CFBundleDocumentTypes` 两项：

| CFBundleTypeName | CFBundleTypeRole | LSHandlerRank | CFBundleTypeExtensions |
|---|---|---|---|
| Camera RAW image | Viewer | Alternate | iiq, nef, cr3, cr2, arw, dng, raf, orf, rw2, pef, srw, 3fr, fff, mos |
| Image | Viewer | Alternate | jpg, jpeg, png, tif, tiff, webp, heic, heif |

- 扩展名来源：RAW 列表逐项对应 `crates/engine/src/raw.rs:1146-1167` 的
  `is_raw`；普通图像列表对应 `crates/engine/src/file_engine.rs:3401-3410`
  的 `is_supported`（`jpg/jpeg/png/tif/tiff/webp/heic/heif`）。导入对话框
  `crates/app/src/ui/import.rs:28-29` 的 16 项是本表的子集，无额外格式。
- `Viewer` + `Alternate`：只进入 Open With / 候选列表，**不声明默认处理器**。
  没有 `LSSupportsOpeningDocumentsInPlace`、没有 `UTImportedTypeDeclarations`
  （这些类型由系统/相机厂商声明，应用不拥有）。
- **未添加 `CFBundleURLTypes`**：应用没有注册自定义 URL scheme。GPUI 的
  `Platform::on_open_urls` 只负责接收 AppKit 送来的 `file://` URL
  （`crates/app/src/lib.rs:125-146`），不是 scheme 注册。凭空声明一个
  `aurora://` 只会制造不可用入口。
- 脚本在组装后执行 `plutil -lint`，本机输出：
  `dist/Aurora Studio.app/Contents/Info.plist: OK`。

LaunchServices 实际注册证据（`lsregister -f` 后 `lsregister -dump`）：

```
claim id:   Camera RAW image
rank:       Alternate
roles:      Viewer
bindings:   .iiq, .nef, .cr3, .cr2, .arw, .dng, .raf, .orf, .rw2, .pef, .srw, .3fr, .fff, .mos
claim id:   Image
rank:       Alternate
roles:      Viewer
bindings:   .jpg, .jpeg, .png, .tif, .tiff, .webp, .heic, .heif
```

注册结果里同时出现 `claimed UTIs: public.jpeg, public.heic, com.canon.cr2-raw-image, …`，
这是系统按扩展名/类型推导出的结果，不代表应用声明了这些 UTI 的所有权。

### 11.2 构建与打包

```sh
./scripts/build-macos.sh        # exit 0
./scripts/package-macos.sh      # exit 0
```

`build-macos.sh` 关键输出：

```
  host architecture : arm64 (build output is always aarch64-apple-darwin)
  deployment target : 12.0
  version           : 0.1.0
  bundle id         : local.aurora-studio.dev (local development only)
    Finished `release` profile [optimized] target(s) in 2m 17s
== assemble …/dist/Aurora Studio.app ==
…/Contents/Info.plist: OK
  lipo -archs : arm64
…/Contents/MacOS/aurora-studio: Mach-O 64-bit executable arm64
```

`package-macos.sh` 关键输出：

```
  version : 0.1.0
  bundle  : local.aurora-studio.dev
  minimum : 12.0
…/Contents/MacOS/aurora-studio: Mach-O 64-bit executable arm64
  lipo -archs : arm64
```

`otool -L` 共 21 条，全部是 `/System/Library/...` 或 `/usr/lib/...`
（AppKit、Metal、ColorSync、CoreGraphics、CoreVideo、UniformTypeIdentifiers、
libobjc、libSystem 等），没有任何 `@rpath`/Homebrew/开发者路径依赖，脚本的
非系统依赖检查通过。

签名（ad-hoc，非公证）：

```
codesign --force --sign - --timestamp=none "dist/Aurora Studio.app"
codesign --verify --strict --verbose=2 "dist/Aurora Studio.app"
→ valid on disk
→ satisfies its Designated Requirement
→ Signature : adhoc
→ TeamIdentifier=not set
```

产物：`dist/Aurora Studio-0.1.0-arm64.zip`（17 MB），可执行文件 48 MB。
`codesign --verify` 退出码 0。首次打包生成的 zip sha256 为
`345961edd5631c96c4a2a177c9d102f5a2051ab19d3c72ed8374fccb1e21899d`；为捕获
退出码重跑一次后变为 `0575248954559e63fd5d0136010287b5f4f36f46b0a2403abc2eec969c89c17a`
——zip 内含归档时间戳，**同一二进制重复打包哈希会变**，不能把 sha256 当作
构建可复现性的证据；sidecar `.sha256` 始终对应当前 zip。

### 11.3 启动冒烟

1. `open "dist/Aurora Studio.app"` → 退出码 0。进程 PID 88716（PPID 1，
   07:56:54 启动），`lsappinfo` 显示 `bundleID="local.aurora-studio.dev"`、
   状态 `(in front)`。
2. 日志文件首次创建：`~/Library/Logs/Aurora Studio/aurora-studio.log`
   （07:57，252 字节），内容一行：

   ```
   [WARN] aurora_engine::aimask::ort: ONNX Runtime unavailable: ONNX Runtime not found; set AURORA_ORT_LIB to an absolute ONNX Runtime library path (looked in libonnxruntime.dylib, libonnxruntime.23.dylib, libonnxruntime.1.dylib, libonnxruntime.dylib.23)
   ```

   `log show --last 2m --predicate 'process == "aurora-studio"'` 也有该进程的
   AppKit/LaunchServices/XPC 记录，证明它完成了正常 AppKit 启动路径。
3. 进程存活约 22 秒后，`osascript -e 'tell application "Aurora Studio" to quit'`
   退出码 0，`pgrep -x aurora-studio` 随后返回无进程。
4. `~/Library/Logs/DiagnosticReports/` 里**没有** `aurora-studio` 的崩溃报告。
   该目录存在其他 lane 的测试二进制崩溃（`aurora_portrait-<hash>`、
   `aurora_engine-<hash>`，时间 05:59–07:45），与本次启动无关，未处理。
5. cwd 独立性：`cd /tmp` 后直接运行
   `"/…/dist/Aurora Studio.app/Contents/MacOS/aurora-studio"`，PID 90760
   存活（`ps` 可查），SIGTERM 后 `wait` 返回 143（正常被终止），无
   panic/崩溃。启动不依赖仓库工作目录。

### 11.4 仍未验证 / 边界

- **Finder 双击 / Open With 的实际导入效果**：Info.plist 声明与 LaunchServices
  注册已核实，但没有做 GUI 双击操作，也没有验证 `application:openURLs:`
  到达后文件是否成功进入导入流程。不写成“Finder 关联已验证可用”。
- **干净机器**：未在无 Xcode/Homebrew 的 Mac 上运行；dylib 搜索路径未验证。
- **Gatekeeper / 公证**：未做，ad-hoc 签名在别的机器上仍可能被拦截。
- **菜单、快捷键、IME、Retina/外接屏、拖放**：同 §10.7，仍未做真机交互验证。
- **Intel / universal**：本机只产出 arm64 slice，未验证 x86_64。
- 本轮没有截屏或桌面录制。
