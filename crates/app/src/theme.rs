//! Aurora's dark workspace palette, projected onto both GPUI Kit and Ely so their
//! components read as one system.

use gpui_kit::component::{Theme as KitTheme, ThemeMode, ThemeRegistry};
use gpui_kit::{App, Global, Hsla, SharedString, px, rgb};

/// Semantic colors for everything Aurora paints itself.
#[derive(Clone, Debug)]
pub struct Palette {
    /// Window chrome: title bar, tool strip, status bar.
    pub chrome: Hsla,
    /// Inspector and browser surfaces.
    pub panel: Hsla,
    /// Tool panel headers and raised rows.
    pub raised: Hsla,
    /// Inputs, tracks, wells.
    pub well: Hsla,
    pub hover: Hsla,
    pub pressed: Hsla,
    /// Hairline between regions.
    pub divider: Hsla,
    /// Light edge on raised surfaces.
    pub edge: Hsla,
    pub text: Hsla,
    pub text_muted: Hsla,
    pub text_subtle: Hsla,
    pub text_disabled: Hsla,
    pub accent: Hsla,
    pub accent_soft: Hsla,
    pub on_accent: Hsla,
    pub warning: Hsla,
    pub danger: Hsla,
    pub success: Hsla,
    pub selection: Hsla,
    pub thumb_knob: Hsla,
}

pub fn hex(c: u32) -> Hsla {
    rgb(c).into()
}

pub fn hexa(c: u32, alpha: f32) -> Hsla {
    let mut h: Hsla = rgb(c).into();
    h.a = alpha;
    h
}

impl Palette {
    pub fn dark(accent: u32) -> Self {
        Self {
            chrome: hex(0x202022),
            panel: hex(0x262628),
            raised: hex(0x2d2d30),
            well: hex(0x1b1b1c),
            hover: hex(0x343438),
            pressed: hex(0x3d3d42),
            divider: hex(0x141415),
            edge: hexa(0xffffff, 0.045),
            text: hex(0xdcdcde),
            text_muted: hex(0x9c9ca2),
            text_subtle: hex(0x7e7e85),
            text_disabled: hex(0x58585e),
            accent: hex(accent),
            accent_soft: hexa(accent, 0.18),
            on_accent: hex(0xffffff),
            warning: hex(0xe8b45e),
            danger: hex(0xe5605a),
            success: hex(0x6bbc89),
            selection: hexa(accent, 0.32),
            thumb_knob: hex(0xd8d8dc),
        }
    }
}

/// App-wide visual settings, read with [`pal`].
pub struct AuroraTheme {
    pub palette: Palette,
    pub accent: u32,
}

impl Global for AuroraTheme {}

pub const ACCENTS: [(u32, &str); 6] = [
    (0x4c8dff, "Blue"),
    (0xf08a3c, "Amber"),
    (0x3fb68b, "Jade"),
    (0xd05f9a, "Rose"),
    (0x8f7cf2, "Violet"),
    (0xb8b8bd, "Graphite"),
];

static TOUCH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the interface is laid out for fingers (the phone and tablet shell).
pub fn touch() -> bool {
    TOUCH.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_touch(on: bool) {
    TOUCH.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn pal(cx: &App) -> &Palette {
    &cx.global::<AuroraTheme>().palette
}

/// Row heights and paddings shared by the inspector widgets.
pub mod metrics {
    use super::touch;

    /// Height of one inspector row: a finger needs more than a pointer does.
    pub fn row() -> f32 {
        if touch() { 40.0 } else { ROW }
    }

    pub fn panel_header() -> f32 {
        if touch() { 40.0 } else { PANEL_HEADER }
    }

    pub const TOOLBAR: f32 = 44.0;
    pub const STATUS_BAR: f32 = 24.0;
    pub const TOOL_STRIP: f32 = 44.0;
    pub const PANEL_HEADER: f32 = 28.0;
    pub const ROW: f32 = 24.0;
    pub const LABEL_W: f32 = 84.0;
    pub const VALUE_W: f32 = 46.0;
}

const KIT_THEME: &str = include_str!("../assets/aurora-theme.json");

pub fn init(cx: &mut App) {
    let accent = ACCENTS[0].0;
    cx.set_global(AuroraTheme { palette: Palette::dark(accent), accent });
    if let Err(err) = ThemeRegistry::global_mut(cx).load_themes_from_str(KIT_THEME) {
        log::error!("theme: Aurora Dark failed to load: {err}");
    }
    apply(cx);
}

/// Pushes the current palette into GPUI Kit and Ely.
pub fn apply(cx: &mut App) {
    let accent = cx.global::<AuroraTheme>().accent;
    let config = ThemeRegistry::global(cx).themes().get(&SharedString::from("Aurora Dark")).cloned();
    if let Some(config) = config {
        KitTheme::update(cx, |theme| {
            theme.apply_config(&config);
            theme.font_size = px(13.0);
            theme.radius = px(4.0);
            theme.radius_lg = px(6.0);
        });
    }
    KitTheme::change(ThemeMode::Dark, None, cx);
    KitTheme::update(cx, |theme| {
        let a: Hsla = rgb(accent).into();
        theme.primary = a;
        theme.ring = a;
        theme.slider_bar = a;
        theme.selection = {
            let mut s = a;
            s.a = 0.32;
            s
        };
    });

    let p = cx.global::<AuroraTheme>().palette.clone();
    let mut ely = ely_core::theme::Palette::dark(false);
    ely.bg = p.panel;
    ely.surface = p.raised;
    ely.sunken = p.well;
    ely.overlay = p.raised;
    ely.hover = p.hover;
    ely.active = p.pressed;
    ely.border = hex(0x3a3a3e);
    ely.fg = p.text;
    ely.fg_muted = p.text_muted;
    ely.fg_subtle = p.text_subtle;
    ely.accent = p.accent;
    ely.on_accent = p.on_accent;
    ely.focus = p.accent;
    ely.selection = p.selection;
    ely_core::theme::Theme::set_palette(ely_core::theme::Mode::Dark, Some(ely), cx);
    ely_core::theme::Theme::set_mode_now(ely_core::theme::Mode::Dark, cx);
}

pub fn set_accent(accent: u32, cx: &mut App) {
    cx.set_global(AuroraTheme { palette: Palette::dark(accent), accent });
    apply(cx);
    cx.refresh_windows();
}

/// sRGB color of a hue at full saturation, for color wheels and tags.
pub fn hue_color(hue: f32, sat: f32, light: f32) -> Hsla {
    gpui_kit::hsla(hue.rem_euclid(360.0) / 360.0, sat, light, 1.0)
}
