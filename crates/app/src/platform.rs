//! Platform conventions: which modifier is the command key, how shortcuts are
//! written, and where the menu bar lives.
//!
//! macOS builds follow Mac conventions (⌘ instead of Ctrl, the system menu
//! bar, ⇧⌘I-style key labels). `AURORA_PLATFORM=mac` turns the same
//! conventions on in any desktop build, so the Mac interface can be previewed
//! and screenshotted on Linux or Windows.

use std::sync::OnceLock;

pub fn is_mac() -> bool {
    static MAC: OnceLock<bool> = OnceLock::new();
    *MAC.get_or_init(|| {
        cfg!(target_os = "macos") || std::env::var("AURORA_PLATFORM").is_ok_and(|v| v.eq_ignore_ascii_case("mac") || v.eq_ignore_ascii_case("macos"))
    })
}

/// Default shortcuts are written the Windows way (`ctrl-shift-i`); on a Mac the
/// command key takes Ctrl's place, and keys macOS reserves move aside.
pub fn native_keys(keys: &str) -> String {
    if !is_mac() {
        return keys.to_string();
    }
    keys.split(' ').map(native_stroke).collect::<Vec<_>>().join(" ")
}

fn native_stroke(stroke: &str) -> String {
    // macOS owns these: ⌘H hides the app, ⌘M minimizes, F11 shows the desktop.
    match stroke {
        "ctrl-h" => return "cmd-shift-h".into(),
        "ctrl-m" => return "cmd-shift-m".into(),
        "f11" => return "ctrl-cmd-f".into(),
        "ctrl-y" => return "cmd-shift-z".into(),
        "delete" => return "cmd-backspace".into(),
        _ => {}
    }
    match stroke.strip_prefix("ctrl-") {
        Some(rest) => format!("cmd-{rest}"),
        None => stroke.to_string(),
    }
}

/// Splits `ctrl-shift--` into its modifiers and key, keeping `-` as a key.
fn parse(stroke: &str) -> (Vec<&str>, &str) {
    let mut mods = Vec::new();
    let mut rest = stroke;
    loop {
        let Some((head, tail)) = rest.split_once('-') else { break };
        if tail.is_empty() || !matches!(head, "ctrl" | "alt" | "shift" | "cmd" | "super" | "win" | "fn" | "secondary") {
            break;
        }
        mods.push(head);
        rest = tail;
    }
    (mods, rest)
}

fn key_name(key: &str) -> String {
    let mac = is_mac();
    match key {
        "left" => "←".into(),
        "right" => "→".into(),
        "up" => "↑".into(),
        "down" => "↓".into(),
        "enter" if mac => "↩".into(),
        "enter" => "Enter".into(),
        "escape" if mac => "⎋".into(),
        "escape" => "Esc".into(),
        "backspace" if mac => "⌫".into(),
        "delete" if mac => "⌦".into(),
        "delete" => "Delete".into(),
        "space" => "Space".into(),
        "tab" if mac => "⇥".into(),
        other if other.chars().count() == 1 => other.to_uppercase(),
        other => {
            let mut c = other.chars();
            c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
        }
    }
}

/// One keystroke for people: `Ctrl+Shift+I` on Windows and Linux, `⇧⌘I` on a Mac.
fn display_stroke(stroke: &str) -> String {
    let (mods, key) = parse(stroke);
    let has = |m: &str| mods.contains(&m);
    let key = key_name(key);
    if is_mac() {
        // Apple's order: Control, Option, Shift, Command.
        let mut s = String::new();
        if has("ctrl") {
            s.push('⌃');
        }
        if has("alt") {
            s.push('⌥');
        }
        if has("shift") {
            s.push('⇧');
        }
        if has("cmd") || has("super") || has("secondary") {
            s.push('⌘');
        }
        s + &key
    } else {
        let mut parts: Vec<String> = Vec::new();
        if has("ctrl") || has("secondary") {
            parts.push("Ctrl".into());
        }
        if has("alt") {
            parts.push("Alt".into());
        }
        if has("shift") {
            parts.push("Shift".into());
        }
        if has("cmd") || has("super") || has("win") {
            parts.push("Win".into());
        }
        parts.push(key);
        parts.join("+")
    }
}

/// Shortcuts as written in bindings (`ctrl-k ctrl-s`, alternatives joined by
/// ` / `) for people.
pub fn display(keys: &str) -> String {
    keys.split(" / ")
        .map(|alt| alt.split(' ').filter(|s| !s.is_empty()).map(display_stroke).collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join(" / ")
}

/// A default shortcut as this platform binds and shows it: `keys("ctrl-shift-i")`
/// is `Ctrl+Shift+I` on Windows and `⇧⌘I` on a Mac.
pub fn keys(default: &str) -> String {
    display(&native_keys(default))
}

/// The Alt key's name in sentences ("Alt-click", "⌥-click").
pub fn alt() -> &'static str {
    if is_mac() { "⌥" } else { "Alt" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strokes_parse_with_minus_as_a_key() {
        assert_eq!(parse("ctrl--"), (vec!["ctrl"], "-"));
        assert_eq!(parse("ctrl-shift-i"), (vec!["ctrl", "shift"], "i"));
        assert_eq!(parse("-"), (vec![], "-"));
        assert_eq!(parse("f11"), (vec![], "f11"));
    }

    #[test]
    fn mac_strokes_swap_ctrl_for_command_and_avoid_system_keys() {
        assert_eq!(native_stroke("ctrl-alt-p"), "cmd-alt-p");
        assert_eq!(native_stroke("ctrl-h"), "cmd-shift-h");
        assert_eq!(native_stroke("alt-e"), "alt-e");
        assert_eq!(native_stroke("ctrl--"), "cmd--");
    }
}
