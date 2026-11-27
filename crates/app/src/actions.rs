//! Commands, their default key bindings (Capture One's Windows layout where one
//! exists) and the application menu.

use gpui_kit::component::{GlobalState, menu::AppMenuBar};
use gpui_kit::{Action, App, Entity, KeyBinding, Menu, MenuItem, actions};
use serde::Deserialize;

use crate::i18n::t;

actions!(
    portrait,
    [
        Quit,
        SaveUserStyle,
        AddSerialNode,
        AddParallelNode,
        AddLayerNode,
        AddOutsideNode,
        ToggleNode,
        DeleteNode,
        TidyNodes,
        ToggleNodeDock,
        ToggleOverlay,
        ChooseOverlay,
        Slideshow,
        CullMode,
        SecondViewer,
        StackSelected,
        UnstackSelected,
        AutoStack,
        ToggleStacksOpen,
        Import,
        OpenFolder,
        OpenCatalog,
        NewSession,
        SyncXmp,
        ExportSelected,
        OpenExportDialog,
        Undo,
        Redo,
        CopyAdjustments,
        ApplyAdjustments,
        ResetAdjustments,
        AutoAdjust,
        AutoRotate,
        CheckClosedEyes,
        MergeHdr,
        MergePanorama,
        MatchLook,
        NewVariant,
        CloneVariant,
        SelectAll,
        SelectCurrentOnly,
        NextImage,
        PrevImage,
        ZoomFit,
        Zoom100,
        ZoomIn,
        ZoomOut,
        ToggleBrowser,
        ToggleGridView,
        ToggleInspector,
        ToggleCompare,
        ToggleExposureWarning,
        ToggleProof,
        ToggleSoftProof,
        ToggleGrid,
        ToggleFullscreen,
        ToggleMaskOverlay,
        RotateLeft,
        RotateRight,
        FlipHorizontal,
        NewLayer,
        ShowShortcuts,
        ShowPreferences,
        ShowAbout,
        CommandPalette,
        ToggleLanguage,
        MoveToTrash,
        Capture,
        LiveView,
        ClearAnnotations,
        ToggleMultiView,
        SetCompareVariant,
        ClearCompareVariant,
        Print,
        SaveWorkspace,
        ToggleInspectorSide,
        // macOS application menu. Harmless (and unreachable) on other platforms.
        Hide,
        HideOthers,
        ShowAll,
        Minimize,
        ZoomWindow,
    ]
);

#[derive(Action, Clone, PartialEq, Deserialize)]
#[action(namespace = portrait, no_json)]
pub struct SetRating(pub i8);

#[derive(Action, Clone, PartialEq, Deserialize)]
#[action(namespace = portrait, no_json)]
pub struct SetColorTag(pub u8);

#[derive(Action, Clone, PartialEq, Deserialize)]
#[action(namespace = portrait, no_json)]
pub struct SelectToolTab(pub usize);

/// Ctrl+1…9: the n-th tab as the tab bar shows it (order and hidden tabs applied).
#[derive(Action, Clone, PartialEq, Deserialize)]
#[action(namespace = portrait, no_json)]
pub struct SelectNthTab(pub usize);

#[derive(Action, Clone, PartialEq, Deserialize)]
#[action(namespace = portrait, no_json)]
pub struct SelectCursor(pub u8);

#[derive(Action, Clone, PartialEq, Deserialize)]
#[action(namespace = portrait, no_json)]
pub struct SetWorkspace(pub usize);

/// Opens the `n`-th entry of File → Open Recent.
#[derive(Action, Clone, PartialEq, Deserialize)]
#[action(namespace = portrait, no_json)]
pub struct OpenRecent(pub usize);

/// Key context of the workspace.
pub const CONTEXT: &str = "Workspace";

/// One rebindable command: a stable id for the settings file, a label, its
/// group in the shortcut editor, Capture One's default keys, and the action.
pub struct Shortcut {
    pub id: String,
    pub label: String,
    pub group: &'static str,
    pub defaults: Vec<&'static str>,
    pub action: Box<dyn Action>,
    /// Active everywhere, not only in the workspace outside text fields.
    pub global: bool,
}

/// Every command with a shortcut, in the order the editor lists them.
pub fn shortcuts() -> Vec<Shortcut> {
    let mut out = Vec::new();
    let mut add = |id: &str, en: &'static str, zh: &'static str, group: &'static str, keys: &[&'static str], action: Box<dyn Action>| {
        out.push(Shortcut { id: id.to_string(), label: t(en, zh).to_string(), group, defaults: keys.to_vec(), action, global: id == "quit" });
    };
    let file = t("File", "文件");
    let edit = t("Edit", "编辑");
    let image = t("Image", "图像");
    let view = t("View", "视图");
    let tools = t("Tools", "工具");
    let rate = t("Rating & Tags", "评级与色标");
    // `secondary` is GPUI's platform-aware primary modifier: Command on macOS,
    // Control on Windows/Linux. Windows bindings therefore stay Ctrl+…; on macOS
    // they become Cmd+…. Two Capture One bindings intentionally keep `ctrl`:
    // Cmd+H hides the app and Cmd+M minimizes the window on macOS, so Merge HDR
    // and Merge Panorama stay on Control+H / Control+M there.
    add("quit", "Exit", "退出", file, &["secondary-q"], Box::new(Quit));
    add("import", "Import images", "导入图像", file, &["secondary-shift-i"], Box::new(Import));
    add("export-selected", "Process selected", "处理所选", file, &["secondary-d"], Box::new(ExportSelected));
    add("export", "Export…", "导出…", file, &["secondary-shift-d"], Box::new(OpenExportDialog));
    add("print", "Print…", "打印…", file, &["secondary-p"], Box::new(Print));
    add("prefs", "Preferences", "首选项", file, &["secondary-,"], Box::new(ShowPreferences));
    add("capture", "Capture", "拍摄", file, &["secondary-k"], Box::new(Capture));
    add("live-view", "Live View", "实时取景", file, &["secondary-alt-v"], Box::new(LiveView));
    add("undo", "Undo", "撤销", edit, &["secondary-z"], Box::new(Undo));
    add("redo", "Redo", "重做", edit, &["secondary-y", "secondary-shift-z"], Box::new(Redo));
    add("copy-adj", "Copy adjustments", "拷贝调整", edit, &["secondary-shift-c"], Box::new(CopyAdjustments));
    add("apply-adj", "Apply adjustments", "应用调整", edit, &["secondary-shift-v"], Box::new(ApplyAdjustments));
    add("reset-adj", "Reset adjustments", "复位调整", edit, &["secondary-r"], Box::new(ResetAdjustments));
    add("auto-adj", "Auto adjust", "自动调整", edit, &["secondary-l"], Box::new(AutoAdjust));
    add("match-look", "Match look to reference", "匹配参考风格", edit, &[], Box::new(MatchLook));
    add("select-all", "Select all", "全选", edit, &["secondary-a"], Box::new(SelectAll));
    add("select-current", "Select only the current image", "只选当前图像", edit, &["escape"], Box::new(SelectCurrentOnly));
    add("palette", "Command palette", "命令面板", edit, &["secondary-shift-p"], Box::new(CommandPalette));
    add("new-variant", "New variant", "新建变体", image, &["f2"], Box::new(NewVariant));
    add("clone-variant", "Clone variant", "克隆变体", image, &["f3"], Box::new(CloneVariant));
    add("compare-variant", "Set compare variant", "设为对比变体", image, &["shift-enter"], Box::new(SetCompareVariant));
    add("clear-compare", "Clear compare variant", "取消对比变体", image, &["shift-escape"], Box::new(ClearCompareVariant));
    add("next", "Next image", "下一张", image, &["right"], Box::new(NextImage));
    add("prev", "Previous image", "上一张", image, &["left"], Box::new(PrevImage));
    add("rotate-left", "Rotate left", "向左旋转", image, &["secondary-alt-l"], Box::new(RotateLeft));
    add("auto-rotate", "Auto rotate", "自动旋转", image, &[], Box::new(AutoRotate));
    add("closed-eyes", "Check closed eyes", "检查闭眼", image, &[], Box::new(CheckClosedEyes));
    add("merge-hdr", "Merge to HDR", "合并为 HDR", image, &["ctrl-h"], Box::new(MergeHdr));
    add("merge-pano", "Merge to panorama", "合并为全景", image, &["ctrl-m"], Box::new(MergePanorama));
    add("rotate-right", "Rotate right", "向右旋转", image, &["secondary-alt-r"], Box::new(RotateRight));
    add("trash", "Move to trash", "移到废纸篓", image, &["delete"], Box::new(MoveToTrash));
    add("new-layer", "New adjustment layer", "新建调整图层", image, &["secondary-shift-l"], Box::new(NewLayer));
    add("mask", "Display mask", "显示蒙版", image, &["m"], Box::new(ToggleMaskOverlay));
    let nodes = t("Nodes", "节点");
    add("node-serial", "Add serial node", "添加串行节点", nodes, &["alt-s"], Box::new(AddSerialNode));
    add("node-parallel", "Add parallel node", "添加并行节点", nodes, &["alt-shift-p"], Box::new(AddParallelNode));
    add("node-layer", "Add layer node", "添加图层节点", nodes, &["alt-l"], Box::new(AddLayerNode));
    add("node-outside", "Add outside node", "添加外部节点", nodes, &["alt-shift-o"], Box::new(AddOutsideNode));
    add("node-toggle", "Enable / disable node", "启用 / 停用节点", nodes, &["secondary-shift-e"], Box::new(ToggleNode));
    add("node-dock", "Dock node editor", "停靠节点编辑器", nodes, &["secondary-shift-n"], Box::new(ToggleNodeDock));
    add("zoom-fit", "Zoom to fit", "适合窗口", view, &["secondary-0"], Box::new(ZoomFit));
    add("zoom-100", "Zoom 100%", "100% 缩放", view, &["secondary-alt-0"], Box::new(Zoom100));
    add("zoom-in", "Zoom in", "放大", view, &["secondary-="], Box::new(ZoomIn));
    add("zoom-out", "Zoom out", "缩小", view, &["secondary--"], Box::new(ZoomOut));
    add("compare", "Before / after", "前后对比", view, &["y"], Box::new(ToggleCompare));
    add("multi", "Multiple view", "多图查看", view, &["n"], Box::new(ToggleMultiView));
    add("overlay", "Show / hide overlay", "显示/隐藏叠加图", view, &["alt-o"], Box::new(ToggleOverlay));
    add("slideshow", "Slideshow", "幻灯片放映", view, &["secondary-alt-s"], Box::new(Slideshow));
    add("cull", "Cull", "挑片模式", view, &["secondary-alt-c"], Box::new(CullMode));
    add("second-viewer", "Second viewer window", "第二查看器窗口", view, &["secondary-shift-v"], Box::new(SecondViewer));
    add("stack", "Stack selected", "堆叠所选", image, &["secondary-alt-k"], Box::new(StackSelected));
    add("unstack", "Unstack", "取消堆叠", image, &["secondary-alt-shift-k"], Box::new(UnstackSelected));
    add("stacks-open", "Expand / collapse stacks", "展开/折叠堆叠", image, &["shift-s"], Box::new(ToggleStacksOpen));
    add("save-style", "Save user style", "存储用户风格", edit, &["secondary-alt-y"], Box::new(SaveUserStyle));
    add("warning", "Exposure warning", "曝光警告", view, &["alt-e"], Box::new(ToggleExposureWarning));
    add("proof", "Proof margin", "校样边距", view, &["alt-p"], Box::new(ToggleProof));
    add("soft-proof", "Soft proof", "软打样", view, &["secondary-alt-p"], Box::new(ToggleSoftProof));
    add("grid", "Grid", "网格", view, &["alt-g"], Box::new(ToggleGrid));
    add("browser", "Show / hide browser", "显示/隐藏浏览器", view, &["secondary-b"], Box::new(ToggleBrowser));
    add("grid-view", "Grid view", "网格视图", view, &["secondary-g"], Box::new(ToggleGridView));
    add("inspector", "Show / hide tools", "显示/隐藏工具", view, &["secondary-t"], Box::new(ToggleInspector));
    add("inspector-side", "Tools on the other side", "工具栏换边", view, &["secondary-shift-t"], Box::new(ToggleInspectorSide));
    // macOS's full-screen chord is Control+Command+F; Windows keeps F11.
    let fullscreen: &[&'static str] = if cfg!(target_os = "macos") { &["ctrl-cmd-f"] } else { &["f11"] };
    add("fullscreen", "Full screen", "全屏", view, fullscreen, Box::new(ToggleFullscreen));
    add("shortcuts", "Keyboard shortcuts", "键盘快捷键", view, &["secondary-/"], Box::new(ShowShortcuts));
    add("language", "Switch language", "切换语言", view, &["secondary-shift-u"], Box::new(ToggleLanguage));
    for (i, k) in ["secondary-1", "secondary-2", "secondary-3", "secondary-4", "secondary-5", "secondary-6", "secondary-7", "secondary-8", "secondary-9"].iter().enumerate() {
        out.push(Shortcut { id: format!("tab-{i}"), label: format!("{} {}", t("Tool tab", "工具页签"), i + 1), group: tools, defaults: vec![k], action: Box::new(SelectNthTab(i)), global: false });
    }
    let cursor_keys = ["v", "h", "z", "c", "r", "k", "q", "b", "e", "l", "t", "i", "w", "a", "o"];
    for (i, k) in cursor_keys.iter().enumerate() {
        let Some(tool) = crate::ui::toolbar::CURSOR_TOOLS.get(i) else { continue };
        out.push(Shortcut { id: format!("cursor-{i}"), label: format!("{} · {}", t("Cursor", "光标"), tool.label()), group: tools, defaults: vec![k], action: Box::new(SelectCursor(i as u8)), global: false });
    }
    out.push(Shortcut { id: "reject".into(), label: t("Reject", "拒绝").into(), group: rate, defaults: vec!["-"], action: Box::new(SetRating(-1)), global: false });
    for r in 0..=5 {
        out.push(Shortcut { id: format!("rating-{r}"), label: format!("{} {r}", t("Rating", "评级")), group: rate, defaults: vec![["0", "1", "2", "3", "4", "5"][r]], action: Box::new(SetRating(r as i8)), global: false });
    }
    for i in 0..8 {
        out.push(Shortcut { id: format!("tag-{i}"), label: format!("{} {i}", t("Color tag", "色标")), group: rate, defaults: vec![["alt-0", "alt-1", "alt-2", "alt-3", "alt-4", "alt-5", "alt-6", "alt-7"][i]], action: Box::new(SetColorTag(i as u8)), global: false });
    }
    out
}

/// The keys `shortcut` is bound to after the user's overrides.
pub fn keys_for(shortcut: &Shortcut, overrides: &std::collections::BTreeMap<String, Vec<String>>) -> Vec<String> {
    overrides.get(&shortcut.id).cloned().unwrap_or_else(|| shortcut.defaults.iter().map(|k| k.to_string()).collect())
}

/// (Re)installs Aurora's key bindings with `overrides`, keeping every binding
/// the component libraries registered.
pub fn bind_keys(cx: &mut App, overrides: &std::collections::BTreeMap<String, Vec<String>>) {
    let others: Vec<KeyBinding> = cx.key_bindings().borrow().bindings().filter(|b| !b.action().name().starts_with("portrait::")).cloned().collect();
    cx.clear_key_bindings();
    cx.bind_keys(others);
    // Workspace shortcuts stand aside while a text input has focus, so typing a
    // keyword never changes a rating or the cursor tool.
    let predicate = gpui_kit::KeyBindingContextPredicate::parse("Workspace && !Input").ok().map(std::rc::Rc::new);
    let mut keys = Vec::new();
    for s in shortcuts() {
        for k in keys_for(&s, overrides) {
            match KeyBinding::load(&k, s.action.boxed_clone(), if s.global { None } else { predicate.clone() }, false, None, cx) {
                Ok(b) => keys.push(b),
                Err(e) => log::warn!("shortcut {}: {k}: {e:?}", s.id),
            }
        }
    }
    cx.bind_keys(keys);
    // Standard macOS application-menu chords. They are not part of the user
    // rebindable list; Merge HDR / Merge Panorama keep Control+H / Control+M so
    // Cmd+H and Cmd+M stay free for AppKit.
    #[cfg(target_os = "macos")]
    {
        let extras: [(&str, Box<dyn Action>); 3] = [
            ("cmd-h", Box::new(Hide)),
            ("cmd-alt-h", Box::new(HideOthers)),
            ("cmd-m", Box::new(Minimize)),
        ];
        let mut bindings = Vec::new();
        for (source, action) in extras {
            match KeyBinding::load(source, action, None, false, None, cx) {
                Ok(binding) => bindings.push(binding),
                Err(error) => log::warn!("platform shortcut {source}: {error:?}"),
            }
        }
        cx.bind_keys(bindings);
    }
}

/// `secondary-shift-k` → `Ctrl+Shift+K` (`⌘+Shift+K` on macOS).
pub fn display_keys(keys: &str) -> String {
    let mac = cfg!(target_os = "macos");
    keys.split(' ')
        .map(|stroke| {
            stroke
                .split('-')
                .filter(|p| !p.is_empty())
                .map(|p| match p {
                    "ctrl" => "Ctrl".to_string(),
                    "alt" => "Alt".to_string(),
                    "shift" => "Shift".to_string(),
                    // `cmd`/`super`/`win` all parse to GPUI's `platform` modifier,
                    // which is Command on macOS — never label it "Win" there.
                    "cmd" | "super" | "win" => if mac { "⌘".to_string() } else { "Win".to_string() },
                    "secondary" => if mac { "⌘".to_string() } else { "Ctrl".to_string() },
                    "left" => "←".to_string(),
                    "right" => "→".to_string(),
                    "up" => "↑".to_string(),
                    "down" => "↓".to_string(),
                    "enter" => "Enter".to_string(),
                    "escape" => "Esc".to_string(),
                    "delete" => "Delete".to_string(),
                    "space" => "Space".to_string(),
                    other if other.len() == 1 => other.to_uppercase(),
                    other => {
                        let mut c = other.chars();
                        c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
                    }
                })
                .collect::<Vec<_>>()
                .join("+")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn menus(workspaces: &[String], recent: &[String], welcome: bool) -> Vec<Menu> {
    let mut all = all_menus(workspaces, recent);
    if welcome {
        // Nothing is open on the welcome page, so the image menus would only hold dead items.
        let image_menus = [t("Camera", "相机"), t("Image", "图像"), t("Adjustments", "调整"), t("Nodes", "节点")];
        all.retain(|m| !image_menus.iter().any(|n| m.name.as_ref() == *n));
    }
    all
}

fn all_menus(workspaces: &[String], recent: &[String]) -> Vec<Menu> {
    let mut recent_items: Vec<MenuItem> = recent.iter().enumerate().map(|(i, name)| MenuItem::action(name.clone(), OpenRecent(i))).collect();
    if recent_items.is_empty() {
        recent_items.push(MenuItem::action(t("No Recent Items", "无最近项目"), OpenRecent(usize::MAX)).disabled(true));
    }
    let mut workspace_items: Vec<MenuItem> = workspaces.iter().enumerate().map(|(i, name)| MenuItem::action(name.clone(), SetWorkspace(i))).collect();
    workspace_items.push(MenuItem::separator());
    workspace_items.push(MenuItem::action(t("Save Workspace…", "存储工作区…"), SaveWorkspace));
    workspace_items.push(MenuItem::action(t("Tools on the Other Side", "工具栏换边"), ToggleInspectorSide));
    vec![
        Menu {
            name: t("File", "文件").into(),
            items: vec![
                MenuItem::action(t("Import Images…", "导入图像…"), Import),
                MenuItem::separator(),
                MenuItem::action(t("Open Folder…", "打开文件夹…"), OpenFolder),
                MenuItem::action(t("Open Capture One Catalog… (partial)", "打开 Capture One 目录…（部分）"), OpenCatalog),
                MenuItem::submenu(Menu { name: t("Open Recent", "打开最近").into(), items: recent_items, disabled: false }),
                MenuItem::action(t("New Session… (unavailable)", "新建会话…（后端未接）"), NewSession),
                MenuItem::action(t("Sync XMP Metadata (unavailable)", "同步 XMP 元数据（后端未接）"), SyncXmp),
                MenuItem::separator(),
                MenuItem::action(t("Export Selected", "导出所选"), ExportSelected),
                MenuItem::action(t("Export…", "导出…"), OpenExportDialog),
                MenuItem::action(t("Print…", "打印…"), Print),
                MenuItem::separator(),
                MenuItem::action(t("Preferences…", "首选项…"), ShowPreferences),
                MenuItem::separator(),
                MenuItem::action(t("Exit", "退出"), Quit),
            ],
            disabled: false,
        },
        Menu {
            name: t("Edit", "编辑").into(),
            items: vec![
                MenuItem::action(t("Undo", "撤销"), Undo),
                MenuItem::action(t("Redo", "重做"), Redo),
                MenuItem::separator(),
                MenuItem::action(t("Select All", "全选"), SelectAll),
                MenuItem::action(t("Select Current Only", "只选当前图像"), SelectCurrentOnly),
                MenuItem::separator(),
                MenuItem::action(t("Command Palette…", "命令面板…"), CommandPalette),
            ],
            disabled: false,
        },
        Menu {
            name: t("Camera", "相机").into(),
            items: vec![
                MenuItem::action(t("Capture", "拍摄"), Capture),
                MenuItem::action(t("Live View", "实时取景"), LiveView),
            ],
            disabled: false,
        },
        Menu {
            name: t("Image", "图像").into(),
            items: vec![
                MenuItem::action(t("Move to Trash", "移到废纸篓"), MoveToTrash),
                MenuItem::action(t("Clear Annotations", "清除标注"), ClearAnnotations),
                MenuItem::separator(),
                MenuItem::action(t("New Variant", "新建变体"), NewVariant),
                MenuItem::action(t("Clone Variant", "克隆变体"), CloneVariant),
                MenuItem::separator(),
                MenuItem::action(t("Stack Selected", "堆叠所选"), StackSelected),
                MenuItem::action(t("Unstack", "取消堆叠"), UnstackSelected),
                MenuItem::action(t("Auto Stack by Capture Time", "按拍摄时间自动堆叠"), AutoStack),
                MenuItem::action(t("Expand / Collapse Stacks", "展开/折叠堆叠"), ToggleStacksOpen),
                MenuItem::separator(),
                MenuItem::action(t("Rotate Left", "向左旋转"), RotateLeft),
                MenuItem::action(t("Rotate Right", "向右旋转"), RotateRight),
                MenuItem::action(t("Flip Horizontal", "水平翻转"), FlipHorizontal),
                MenuItem::action(t("Auto Rotate", "自动旋转"), AutoRotate),
                MenuItem::separator(),
                MenuItem::action(t("Check Closed Eyes", "检查闭眼"), CheckClosedEyes),
                MenuItem::separator(),
                MenuItem::action(t("Merge to HDR…", "合并为 HDR…"), MergeHdr),
                MenuItem::action(t("Merge to Panorama…", "合并为全景…"), MergePanorama),
            ],
            disabled: false,
        },
        Menu {
            name: t("Adjustments", "调整").into(),
            items: vec![
                MenuItem::action(t("Auto Adjust", "自动调整"), AutoAdjust),
                MenuItem::action(t("Match Look to Reference", "匹配参考风格"), MatchLook),
                MenuItem::action(t("Reset", "复位"), ResetAdjustments),
                MenuItem::separator(),
                MenuItem::action(t("Copy Adjustments", "拷贝调整"), CopyAdjustments),
                MenuItem::action(t("Apply Adjustments", "应用调整"), ApplyAdjustments),
                MenuItem::separator(),
                MenuItem::action(t("Save User Style…", "存储用户风格…"), SaveUserStyle),
            ],
            disabled: false,
        },
        Menu {
            name: t("Nodes", "节点").into(),
            items: vec![
                MenuItem::action(t("Add Serial Node", "添加串行节点"), AddSerialNode),
                MenuItem::action(t("Add Parallel Node", "添加并行节点"), AddParallelNode),
                MenuItem::action(t("Add Layer Node", "添加图层节点"), AddLayerNode),
                MenuItem::action(t("Add Outside Node", "添加外部节点"), AddOutsideNode),
                MenuItem::separator(),
                MenuItem::action(t("Enable / Disable Node", "启用 / 停用节点"), ToggleNode),
                MenuItem::action(t("Delete Node", "删除节点"), DeleteNode),
                MenuItem::action(t("Tidy Nodes", "整理节点"), TidyNodes),
                MenuItem::action(t("Dock Node Editor", "停靠节点编辑器"), ToggleNodeDock),
                MenuItem::separator(),
                MenuItem::action(t("New Adjustment Layer", "新建调整图层"), NewLayer),
                MenuItem::action(t("Display Mask", "显示蒙版"), ToggleMaskOverlay),
            ],
            disabled: false,
        },
        Menu {
            name: t("View", "视图").into(),
            items: vec![
                MenuItem::action(t("Zoom to Fit", "适合窗口"), ZoomFit),
                MenuItem::action(t("Zoom 100%", "100% 缩放"), Zoom100),
                MenuItem::action(t("Zoom In", "放大"), ZoomIn),
                MenuItem::action(t("Zoom Out", "缩小"), ZoomOut),
                MenuItem::separator(),
                MenuItem::action(t("Before / After", "前后对比"), ToggleCompare),
                MenuItem::action(t("Multiple View", "多图查看"), ToggleMultiView),
                MenuItem::action(t("Set Compare Variant", "设为对比变体"), SetCompareVariant),
                MenuItem::action(t("Clear Compare Variant", "取消对比变体"), ClearCompareVariant),
                MenuItem::action(t("Exposure Warning", "曝光警告"), ToggleExposureWarning),
                MenuItem::action(t("Proof Margin", "校样边距"), ToggleProof),
                MenuItem::action(t("Soft Proof", "软打样"), ToggleSoftProof),
                MenuItem::action(t("Grid", "网格"), ToggleGrid),
                MenuItem::action(t("Overlay", "叠加图"), ToggleOverlay),
                MenuItem::action(t("Choose Overlay Image…", "选择叠加图…"), ChooseOverlay),
                MenuItem::separator(),
                MenuItem::action(t("Slideshow", "幻灯片放映"), Slideshow),
                MenuItem::action(t("Cull", "挑片模式"), CullMode),
                MenuItem::action(t("Second Viewer Window", "第二查看器窗口"), SecondViewer),
                MenuItem::separator(),
                MenuItem::action(t("Show/Hide Browser", "显示/隐藏浏览器"), ToggleBrowser),
                MenuItem::action(t("Grid View", "网格视图"), ToggleGridView),
                MenuItem::action(t("Show/Hide Tools", "显示/隐藏工具"), ToggleInspector),
                MenuItem::action(t("Full Screen", "全屏"), ToggleFullscreen),
                MenuItem::separator(),
                MenuItem::submenu(Menu { name: t("Workspace", "工作区").into(), items: workspace_items, disabled: false }),
            ],
            disabled: false,
        },
        Menu {
            name: t("Help", "帮助").into(),
            items: vec![
                MenuItem::action(t("Keyboard Shortcuts", "键盘快捷键"), ShowShortcuts),
                MenuItem::action(t("Switch Language (English / 中文)", "切换语言（English / 中文）"), ToggleLanguage),
                MenuItem::separator(),
                MenuItem::action(t("About Aurora Studio", "关于 Aurora Studio"), ShowAbout),
            ],
            disabled: false,
        },
    ]
}

/// The macOS application menu AppKit expects: About, Preferences, Services,
/// Hide / Hide Others / Show All and Quit. Key equivalents come from the
/// keymap (`cmd-h`, `cmd-alt-h`, `secondary-q`, …).
#[cfg(target_os = "macos")]
fn macos_app_menu() -> Menu {
    Menu {
        name: "Aurora Studio".into(),
        items: vec![
            MenuItem::action(t("About Aurora Studio", "关于 Aurora Studio"), ShowAbout),
            MenuItem::separator(),
            MenuItem::action(t("Preferences…", "首选项…"), ShowPreferences),
            MenuItem::separator(),
            MenuItem::os_submenu("Services", gpui_kit::SystemMenuType::Services),
            MenuItem::separator(),
            MenuItem::action(t("Hide Aurora Studio", "隐藏 Aurora Studio"), Hide),
            MenuItem::action(t("Hide Others", "隐藏其他"), HideOthers),
            MenuItem::action(t("Show All", "全部显示"), ShowAll),
            MenuItem::separator(),
            MenuItem::action(t("Quit Aurora Studio", "退出 Aurora Studio"), Quit),
        ],
        disabled: false,
    }
}

/// The macOS Window menu. GPUI only registers it with AppKit (`setWindowsMenu:`)
/// when the title is exactly `Window`, so this title is intentionally not localized.
#[cfg(target_os = "macos")]
fn macos_window_menu() -> Menu {
    Menu {
        name: "Window".into(),
        items: vec![
            MenuItem::action(t("Minimize", "最小化"), Minimize),
            MenuItem::action(t("Zoom", "缩放"), ZoomWindow),
            MenuItem::separator(),
            MenuItem::action(t("Full Screen", "全屏"), ToggleFullscreen),
        ],
        disabled: false,
    }
}

/// Installs the menus for both the native menu bar and the in-window [`AppMenuBar`].
pub fn install_menus(menu_bar: &Entity<AppMenuBar>, workspaces: &[String], recent: &[String], welcome: bool, cx: &mut App) {
    #[cfg(target_os = "macos")]
    {
        // Prepend the application menu and insert the Window menu before Help,
        // matching AppKit's usual order. The in-window bar below keeps the plain
        // Windows/Linux menu list.
        let mut native = Vec::with_capacity(10);
        native.push(macos_app_menu());
        let mut rest = menus(workspaces, recent, welcome);
        let help = rest.pop();
        native.extend(rest);
        native.push(macos_window_menu());
        native.extend(help);
        cx.set_menus(native);
    }
    #[cfg(not(target_os = "macos"))]
    cx.set_menus(menus(workspaces, recent, welcome));
    let owned = menus(workspaces, recent, welcome).into_iter().map(|m| m.owned()).collect();
    GlobalState::global_mut(cx).set_app_menus(owned);
    menu_bar.update(cx, |bar, cx| bar.reload(cx));
}
