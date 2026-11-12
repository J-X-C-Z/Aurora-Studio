//! Command palette (Ctrl+Shift+P): every command and tool tab, searchable.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, WindowExt as _,
    command::{Command, CommandItem, CommandState},
};
use gpui_kit::{Action, AppContext as _, Entity, Focusable as _, ParentElement, Styled, Window, px};

use crate::actions::*;
use crate::i18n::t;
use crate::state::{AppState, ToolTab};

fn items() -> Vec<CommandItem> {
    let cmd = |label: &str, icon: IconName, action: Box<dyn Action>| CommandItem::new().label(label.to_string()).icon(Icon::new(icon)).action(action);
    let mut out = vec![
        cmd(t("Import images…", "导入图像…"), IconName::ImageDown, Box::new(Import)),
        cmd(t("Capture", "拍摄"), IconName::Aperture, Box::new(Capture)),
        cmd(t("Live View", "实时取景"), IconName::Video, Box::new(LiveView)),
        cmd(t("Move to trash", "移到废纸篓"), IconName::Trash, Box::new(MoveToTrash)),
        cmd(t("Clear annotations", "清除标注"), IconName::PenLine, Box::new(ClearAnnotations)),
        cmd(t("Process selected", "处理所选"), IconName::Upload, Box::new(ExportSelected)),
        cmd(t("Undo", "撤销"), IconName::Undo2, Box::new(Undo)),
        cmd(t("Redo", "重做"), IconName::Redo2, Box::new(Redo)),
        cmd(t("Auto adjust", "自动调整"), IconName::WandSparkles, Box::new(AutoAdjust)),
        cmd(t("Reset adjustments", "复位调整"), IconName::RotateCcw, Box::new(ResetAdjustments)),
        cmd(t("Copy adjustments", "拷贝调整"), IconName::Copy, Box::new(CopyAdjustments)),
        cmd(t("Apply adjustments", "应用调整"), IconName::ClipboardPaste, Box::new(ApplyAdjustments)),
        cmd(t("New variant", "新建变体"), IconName::CopyPlus, Box::new(NewVariant)),
        cmd(t("Clone variant", "克隆变体"), IconName::Copy, Box::new(CloneVariant)),
        cmd(t("New adjustment layer", "新建调整图层"), IconName::Layers, Box::new(NewLayer)),
        cmd(t("Before / after", "前后对比"), IconName::Columns2, Box::new(ToggleCompare)),
        cmd(t("Zoom to fit", "适合窗口"), IconName::Maximize, Box::new(ZoomFit)),
        cmd(t("Zoom 100%", "100% 缩放"), IconName::ZoomIn, Box::new(Zoom100)),
        cmd(t("Grid view", "网格视图"), IconName::LayoutGrid, Box::new(ToggleGridView)),
        cmd(t("Rotate left", "向左旋转"), IconName::RotateCcwSquare, Box::new(RotateLeft)),
        cmd(t("Rotate right", "向右旋转"), IconName::RotateCwSquare, Box::new(RotateRight)),
        cmd(t("Keyboard shortcuts", "键盘快捷键"), IconName::Keyboard, Box::new(ShowShortcuts)),
        cmd(t("Preferences", "首选项"), IconName::Settings, Box::new(ShowPreferences)),
        cmd(t("Switch language", "切换语言"), IconName::Languages, Box::new(ToggleLanguage)),
    ];
    for (i, tab) in ToolTab::ALL.iter().enumerate() {
        out.push(
            CommandItem::new()
                .label(format!("{} · {}", t("Tools", "工具"), tab.label()))
                .icon(Icon::new(tab.icon()))
                .action(Box::new(SelectToolTab(i))),
        );
    }
    out
}

pub fn open(_state: &Entity<AppState>, window: &mut Window, cx: &mut gpui_kit::App) {
    let palette = cx.new(|cx| CommandState::new(window, cx));
    let first = std::rc::Rc::new(std::cell::Cell::new(true));
    window.open_dialog(cx, move |dialog, _, _| {
        let palette = palette.clone();
        let first = first.clone();
        dialog.close_button(false).p_0().w(px(560.0)).content(move |content, window, cx| {
            if first.replace(false) {
                let palette = palette.clone();
                window.defer(cx, move |window, cx| palette.read(cx).focus_handle(cx).focus(window, cx));
            }
            content.child(
                Command::new(&palette)
                    .bordered(false)
                    .placeholder(t("Type a command…", "输入命令…"))
                    .max_h(px(420.0))
                    .items(items())
                    .on_confirm(|_, window, cx| window.close_dialog(cx)),
            )
        })
    });
}
