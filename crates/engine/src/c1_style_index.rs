use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use crate::engine::StyleInfo;

/// Every Capture One style and preset installed under the app directory, as
/// engine styles. `LUMEN_C1_STYLES` names a directory whose direct children are
/// the install's style roots, so tests can point at synthetic packages.
pub fn installed_styles() -> Vec<StyleInfo> {
    installed_styles_from(&roots())
}

pub(crate) fn installed_styles_from(dirs: &[PathBuf]) -> Vec<StyleInfo> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for dir in dirs {
        for path in package_paths(dir) {
            let group = path
                .parent()
                .and_then(|p| p.strip_prefix(dir).ok())
                .map(relative_name)
                .unwrap_or_default();
            match crate::c1_styles::load_style(&path) {
                Ok(style) if seen.insert(style.name.clone()) => out.push(StyleInfo {
                    name: style.name,
                    group,
                    adjustments: style.adjustments,
                }),
                Ok(_) => {}
                Err(err) => log::warn!("c1 style {}: {err:#}", path.display()),
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.group.cmp(&b.group)));
    out
}

fn roots() -> Vec<PathBuf> {
    if let Some(dir) = std::env::var_os("LUMEN_C1_STYLES") {
        return vec![PathBuf::from(dir)];
    }
    let install = crate::presets::c1_dir();
    vec![install.join("Styles"), install.join("Presets60")]
}

fn package_paths(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect(root, &mut out);
    out
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("costyle" | "copreset")
        ) {
            out.push(path);
        }
    }
}

fn relative_name(path: &Path) -> String {
    path.components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join(" / ")
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    fn sl(name: &str, exposure: f32) -> String {
        format!(
            r#"<?xml version="1.0" encoding="utf-8"?>
<SL Engine="600"><E K="Name" V="{name}"/><E K="Exposure" V="{exposure}"/></SL>"#
        )
    }

    fn temp_root(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("lumen-style-index-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn real_install_has_styles() {
        let _serial = crate::filmcurve::test_lock();
        if !roots().first().is_some_and(|path| path.is_dir()) {
            return;
        }
        assert!(!installed_styles().is_empty());
    }

    #[test]
    fn lumen_c1_styles_overrides_the_install_directory() {
        let _serial = crate::filmcurve::test_lock();
        let root = temp_root("override");
        fs::write(root.join("Lane Only.costyle"), sl("Lane Only", 0.3)).unwrap();
        let previous = std::env::var_os("LUMEN_C1_STYLES");
        unsafe { std::env::set_var("LUMEN_C1_STYLES", &root) };
        let styles = installed_styles();
        match previous {
            Some(v) => unsafe { std::env::set_var("LUMEN_C1_STYLES", v) },
            None => unsafe { std::env::remove_var("LUMEN_C1_STYLES") },
        }
        assert_eq!(styles.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["Lane Only"]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bad_package_is_skipped() {
        let root = temp_root("bad");
        fs::create_dir_all(root.join("Group")).unwrap();
        fs::write(root.join("Group/Broken.costyle"), b"not a style").unwrap();
        fs::write(root.join("Group/Good.costyle"), sl("Good", 0.5)).unwrap();
        let styles = installed_styles_from(&[root.clone()]);
        assert_eq!(styles.len(), 1);
        assert_eq!(styles[0].name, "Good");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sorted_by_name() {
        let root = temp_root("sorted");
        fs::write(root.join("Zeta.costyle"), sl("Zeta", 1.0)).unwrap();
        fs::write(root.join("Alpha.copreset"), sl("Alpha", 0.0)).unwrap();
        let styles = installed_styles_from(&[root.clone()]);
        let names: Vec<_> = styles.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["Alpha", "Zeta"]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn override_dir_maps_groups() {
        let root = temp_root("groups");
        fs::create_dir_all(root.join("Film")).unwrap();
        fs::write(root.join("Film/One.costyle"), sl("One", 0.1)).unwrap();
        fs::write(root.join("One.costyle"), sl("Top", 0.2)).unwrap();
        let styles = installed_styles_from(&[root.clone()]);
        assert_eq!(
            styles.iter().find(|s| s.name == "One").unwrap().group,
            "Film"
        );
        assert_eq!(styles.iter().find(|s| s.name == "Top").unwrap().group, "");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn duplicate_names_keep_first_in_walk_order() {
        let root = temp_root("dupes");
        fs::write(root.join("First.costyle"), sl("Same", 0.0)).unwrap();
        fs::write(root.join("Second.costyle"), sl("Same", 1.0)).unwrap();
        let styles = installed_styles_from(&[root.clone()]);
        assert_eq!(styles.len(), 1);
        assert_eq!(styles[0].adjustments.exposure.exposure, 0.0);
        fs::remove_dir_all(root).unwrap();
    }
}
