//! Interface language. Every visible string is written once in English and once
//! in Simplified Chinese at its call site: `t("Exposure", "曝光")`.

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Locale {
    En,
    ZhCn,
}

static LOCALE: AtomicU8 = AtomicU8::new(1);

pub fn locale() -> Locale {
    match LOCALE.load(Ordering::Relaxed) {
        0 => Locale::En,
        _ => Locale::ZhCn,
    }
}

pub fn set_locale(locale: Locale) {
    LOCALE.store(
        match locale {
            Locale::En => 0,
            Locale::ZhCn => 1,
        },
        Ordering::Relaxed,
    );
}

/// The string for the current language.
pub fn t(en: &'static str, zh: &'static str) -> &'static str {
    match locale() {
        Locale::En => en,
        Locale::ZhCn => zh,
    }
}

/// Display name for names the engine supplies in English (built-in collections
/// and style groups). Anything else, such as the user's own names, is unchanged.
pub fn builtin(name: &str) -> String {
    if locale() != Locale::ZhCn {
        return name.to_string();
    }
    match name {
        "All Images" => "所有图像",
        "Recent Imports" => "最近导入",
        "Recent Captures" => "最近拍摄",
        "Trash" => "废纸篓",
        "Film" => "胶片",
        "Black & White" => "黑白",
        "Cinematic" => "电影感",
        "Landscape" => "风光",
        "Portrait" => "人像",
        other => other,
    }
    .to_string()
}
