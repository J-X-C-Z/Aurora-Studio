//! Lightweight, dependency-free architecture checks for the Aurora Studio workspace.
//!
//! These tests read the source tree from [`CARGO_MANIFEST_DIR`] (the `crates/engine`
//! directory) and assert a handful of dependency/layering rules. They are meant to
//! be falsifiable substring/manifest checks, not a compiler or a `cargo fmt` gate.
//!
//! Design notes:
//! * Only `std` is used, so the test adds no crates and runs on every platform.
//! * Full-line comments (`//`, `//!`, `///`, `/*`, `* `) are skipped. Trailing
//!   comments are *not* stripped; a forbidden token in a trailing comment would be
//!   reported, which is acceptable for a coarse guard.
//! * Where the current tree legitimately violates a rule (mostly historical
//!   `FileEngine` construction in the app), the exact `file:line` occurrences are
//!   listed as explicit exceptions. A stale exception is itself a failure, so the
//!   list forces an intentional review whenever the file moves.
//! * Files that include CUDA PTX are pinned to an exact set; adding or removing one
//!   fails the test until the allowlist is updated deliberately.
//! * A proposed "no lock held across `fs::` I/O" rule is intentionally not
//!   implemented: a substring check cannot see lock lifetime (drop/scope/early
//!   return), so it would both miss real cases and flag normal short-lock code.
//!   `docs/architecture/ARCHITECTURE-CHECKS.md` records the reasoning instead of
//!   claiming a pass. It would need a control-flow-aware lint, not this harness.
//!
//! Run `sh scripts/architecture/check.sh`, equivalent to:
//! `cargo test -p aurora-engine --test architecture_rules --locked -- --test-threads=1`
//!
//! The `arch_rule` name prefix is not required by the harness; it only makes the
//! suite easy to isolate in test logs.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Workspace root, derived from the compiling crate's manifest directory.
fn workspace_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .expect("crates/engine must live two levels below the workspace root")
        .to_path_buf()
}

/// Path relative to the workspace root, using `/` separators on every platform.
fn rel(root: &Path, path: &Path) -> String {
    let stripped = path.strip_prefix(root).unwrap_or(path);
    stripped
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

fn walk_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            walk_rs(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Yield `(1-based line number, line)` for non-full-line-comment lines.
fn code_lines(src: &str) -> impl Iterator<Item = (usize, &str)> {
    src.lines().enumerate().filter_map(|(i, line)| {
        let t = line.trim_start();
        let is_comment = t.starts_with("//")
            || t.starts_with("/*")
            || t == "*"
            || t.starts_with("* ")
            || t.starts_with("*/");
        (!is_comment).then_some((i + 1, line))
    })
}

/// `(relative file, line, needle)` for every forbidden substring occurrence.
fn scan_file(root: &Path, path: &Path, needles: &[&str]) -> Vec<(String, usize, String)> {
    let src = match fs::read_to_string(path) {
        Ok(src) => src,
        Err(_) => return Vec::new(),
    };
    let mut hits = Vec::new();
    for (line_no, line) in code_lines(&src) {
        for needle in needles {
            if line.contains(needle) {
                hits.push((rel(root, path), line_no, (*needle).to_string()));
            }
        }
    }
    hits
}

fn scan_dir(root: &Path, dir_rel: &str, needles: &[&str]) -> Vec<(String, usize, String)> {
    let mut files = Vec::new();
    walk_rs(&root.join(dir_rel), &mut files);
    let mut hits = Vec::new();
    for file in &files {
        hits.extend(scan_file(root, file, needles));
    }
    hits
}

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// True when `word` occurs in `line` as a whole identifier. Used for short
/// tokens such as `Plan`/`Source`: plain `contains` would also match `Plane`
/// or `SourceMap`. Multi-character tokens such as `render_plan` stay
/// substring checks.
fn contains_word(line: &str, word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    let mut search_from = 0usize;
    loop {
        let Some(offset) = line[search_from..].find(word) else {
            return false;
        };
        let start = search_from + offset;
        let end = start + word.len();
        let boundary = |c: Option<char>| c.map_or(true, |c| !is_ident_char(c));
        if boundary(line[..start].chars().next_back()) && boundary(line[end..].chars().next()) {
            return true;
        }
        search_from = end;
        if search_from >= line.len() {
            return false;
        }
    }
}

fn is_full_line_comment(trimmed: &str) -> bool {
    trimmed.starts_with("//")
        || trimmed.starts_with("/*")
        || trimmed == "*"
        || trimmed.starts_with("* ")
        || trimmed.starts_with("*/")
}

fn is_cfg_test_attr(trimmed: &str) -> bool {
    trimmed.starts_with("#[cfg(test)]")
        || trimmed.starts_with("#[cfg(all(test")
        || trimmed.starts_with("#[cfg(any(test")
}

/// Code lines with items gated by `#[cfg(test)]` removed.
///
/// The production-processing rule must not fail merely because a test module
/// builds a `crate::Source` fixture. Only the gated item is skipped: a braced
/// block is skipped to its matching close, and a single-line item (`use x;`)
/// ends at its semicolon. Brace counting is coarse (strings are not lexed),
/// which is acceptable for an attribute that in this tree always precedes a
/// module or function.
fn code_lines_excluding_cfg_test(src: &str) -> Vec<(usize, &str)> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let trimmed = lines[i].trim_start();
        if is_cfg_test_attr(trimmed) {
            i += 1;
            while i < lines.len() && lines[i].trim_start().starts_with("#[") {
                i += 1;
            }
            if i >= lines.len() {
                break;
            }
            let mut depth = 0i32;
            let mut saw_brace = false;
            while i < lines.len() {
                let line = lines[i];
                for ch in line.chars() {
                    match ch {
                        '{' => {
                            depth += 1;
                            saw_brace = true;
                        }
                        '}' => depth -= 1,
                        _ => {}
                    }
                }
                i += 1;
                if saw_brace && depth <= 0 {
                    break;
                }
                if !saw_brace && line.trim_end().ends_with(';') {
                    break;
                }
            }
            continue;
        }
        if !is_full_line_comment(trimmed) {
            out.push((i + 1, lines[i]));
        }
        i += 1;
    }
    out
}

/// Like [`scan_dir`], but skips `#[cfg(test)]` items and supports
/// identifier-boundary words in addition to plain substrings.
fn scan_dir_non_test(
    root: &Path,
    dir_rel: &str,
    substrings: &[&str],
    words: &[&str],
) -> Vec<(String, usize, String)> {
    let mut files = Vec::new();
    walk_rs(&root.join(dir_rel), &mut files);
    let mut hits = Vec::new();
    for file in &files {
        let src = match fs::read_to_string(file) {
            Ok(src) => src,
            Err(_) => continue,
        };
        for (line_no, line) in code_lines_excluding_cfg_test(&src) {
            for needle in substrings {
                if line.contains(needle) {
                    hits.push((rel(root, file), line_no, (*needle).to_string()));
                }
            }
            for word in words {
                if contains_word(line, word) {
                    hits.push((rel(root, file), line_no, (*word).to_string()));
                }
            }
        }
    }
    hits
}

/// Parse `[package] name` from a Cargo.toml.
fn package_name(text: &str) -> Option<String> {
    let mut in_package = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with('[') && line.ends_with(']') {
            in_package = line == "[package]";
            continue;
        }
        if in_package {
            if let Some(rest) = line.strip_prefix("name") {
                if let Some(eq) = rest.find('=') {
                    return Some(rest[eq + 1..].trim().trim_matches('"').to_string());
                }
            }
        }
    }
    None
}

/// Dependency table keys from `[dependencies]`, `[dev-dependencies]`,
/// `[build-dependencies]` and `[target.<cfg>.dependencies]`.
fn manifest_dep_keys(path: &Path) -> BTreeSet<String> {
    let text = fs::read_to_string(path).unwrap_or_default();
    let mut section = String::new();
    let mut keys = BTreeSet::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = line.to_string();
            continue;
        }
        if !section.ends_with("dependencies]") {
            continue;
        }
        if let Some(eq) = line.find('=') {
            let lhs = line[..eq].trim();
            let key = lhs.split('.').next().unwrap_or(lhs).trim();
            if !key.is_empty() {
                keys.insert(key.to_string());
            }
        }
    }
    keys
}

/// True when `<stem>` is declared in its parent module behind `#[cfg(windows)]`.
fn is_module_gated(crate_src: &Path, file: &Path) -> bool {
    let stem = match file.file_stem().and_then(|s| s.to_str()) {
        Some(stem) => stem,
        None => return false,
    };
    let parent = file.parent().unwrap_or(crate_src);
    let parent_mod = parent.join("mod.rs");
    let module_file = if parent_mod.exists() {
        parent_mod
    } else {
        // Top-level modules are declared in the crate root.
        crate_src.join("lib.rs")
    };
    let text = match fs::read_to_string(&module_file) {
        Ok(text) => text,
        Err(_) => return false,
    };
    let lines: Vec<&str> = text.lines().collect();
    let needle = format!("mod {stem};");
    for (i, line) in lines.iter().enumerate() {
        if line.trim().ends_with(&needle) {
            for previous in lines[..i].iter().rev() {
                let p = previous.trim();
                if p.is_empty() {
                    continue;
                }
                return p.contains("#[cfg(windows)]")
                    || p.contains("cfg(target_os = \"windows\")");
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Rule 1: app talks to the engine only through `ImageEngine` and public types.
// ---------------------------------------------------------------------------

#[test]
fn arch_rule1_app_talks_only_through_image_engine() {
    const NEEDLES: &[&str] = &[
        "FileEngine",
        "aurora_engine::mock",
        "gpu_runtime",
        "portrait_stage::",
        "aurora_portrait",
        "cuda::",
        "metal::",
    ];

    // Known violations in the current tree. Exceptions are pinned by file, token
    // and occurrence count (rather than by line) because the app crate is edited
    // concurrently by other lanes; a shifted line should not red the guard, but an
    // added occurrence must. The current lines are captured in
    // docs/architecture/ARCHITECTURE-CHECKS.md.
    //
    // * source.rs constructs concrete `FileEngine` values and immediately erases
    //   them to `Arc<dyn ImageEngine>`; processing calls still go through the trait.
    // * state/mod.rs usages are inside `#[cfg(test)] mod import_edits_tests`.
    const EXCEPTIONS: &[(&str, &str, usize)] = &[
        ("crates/app/src/state/source.rs", "FileEngine", 3),
        ("crates/app/src/state/mod.rs", "FileEngine", 2),
    ];

    let root = workspace_root();
    let hits = scan_dir(&root, "crates/app/src", NEEDLES);

    // Group hits as (file, needle) -> line numbers.
    let mut actual: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    for (file, line, needle) in hits {
        actual.entry((file, needle)).or_default().push(line);
    }

    let expected: BTreeSet<(String, String)> = EXCEPTIONS
        .iter()
        .map(|(file, needle, _)| (file.to_string(), needle.to_string()))
        .collect();

    let mut problems = Vec::new();
    for ((file, needle), lines) in &actual {
        if !expected.contains(&(file.clone(), needle.clone())) {
            let where_ = lines
                .iter()
                .map(|l| format!("{file}:{l}"))
                .collect::<Vec<_>>()
                .join(", ");
            problems.push(format!(
                "{where_}: `{needle}` is forbidden in crates/app/src; the app must call \
                 aurora_engine::ImageEngine and public types only"
            ));
        }
    }
    for (file, needle, count) in EXCEPTIONS {
        let lines = actual
            .get(&(file.to_string(), needle.to_string()))
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if lines.len() != *count {
            let where_ = if lines.is_empty() {
                "none".to_string()
            } else {
                lines
                    .iter()
                    .map(|l| format!("{file}:{l}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            problems.push(format!(
                "{file}: expected {count} documented `{needle}` exception(s), found {} ({where_}); \
                 update the exception list deliberately",
                lines.len()
            ));
        }
    }

    assert!(
        problems.is_empty(),
        "rule 1 failed:\n{}",
        problems.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Rule 2: engine has no UI dependency (portrait-app / gpui / gpui-kit).
// ---------------------------------------------------------------------------

#[test]
fn arch_rule2_engine_has_no_ui_dependency() {
    const FORBIDDEN_MANIFEST: &[&str] = &[
        "portrait-app",
        "portrait_app",
        "gpui",
        "gpui-kit",
        "gpui_kit",
    ];
    // Import-shaped needles, so a Windows fixture path that merely contains the
    // letters "gpui" is not mistaken for a dependency edge.
    const FORBIDDEN_IMPORTS: &[&str] = &[
        "use gpui",
        "gpui::",
        "gpui_kit",
        "extern crate gpui",
        "use portrait_app",
        "portrait_app::",
        "extern crate portrait_app",
    ];

    let root = workspace_root();
    let manifest = root.join("crates/engine/Cargo.toml");
    let deps = manifest_dep_keys(&manifest);
    let forbidden: BTreeSet<String> = FORBIDDEN_MANIFEST.iter().map(|s| s.to_string()).collect();
    let manifest_hits: Vec<String> = deps
        .intersection(&forbidden)
        .map(|dep| {
            format!(
                "crates/engine/Cargo.toml: dependency `{dep}` is forbidden in aurora-engine"
            )
        })
        .collect();

    let import_hits: Vec<String> = scan_dir(&root, "crates/engine/src", FORBIDDEN_IMPORTS)
        .into_iter()
        .map(|(file, line, needle)| {
            format!("{file}:{line}: `{needle}` imports UI code into aurora-engine")
        })
        .collect();

    assert!(
        manifest_hits.is_empty() && import_hits.is_empty(),
        "rule 2 failed:\n{}\n{}",
        manifest_hits.join("\n"),
        import_hits.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Rule 3: portrait has no dependency on aurora-engine (no reverse edge).
// ---------------------------------------------------------------------------

#[test]
fn arch_rule3_portrait_is_independent_of_engine() {
    const FORBIDDEN_MANIFEST: &[&str] = &["aurora-engine", "aurora_engine"];
    const FORBIDDEN_IMPORTS: &[&str] = &[
        "use aurora_engine",
        "aurora_engine::",
        "extern crate aurora_engine",
    ];

    let root = workspace_root();
    let manifest = root.join("crates/portrait/Cargo.toml");
    let deps = manifest_dep_keys(&manifest);
    let forbidden: BTreeSet<String> = FORBIDDEN_MANIFEST.iter().map(|s| s.to_string()).collect();
    let manifest_hits: Vec<String> = deps
        .intersection(&forbidden)
        .map(|dep| {
            format!(
                "crates/portrait/Cargo.toml: dependency `{dep}` would create a reverse edge \
                 aurora-portrait -> aurora-engine"
            )
        })
        .collect();

    let import_hits: Vec<String> = scan_dir(&root, "crates/portrait/src", FORBIDDEN_IMPORTS)
        .into_iter()
        .map(|(file, line, needle)| {
            format!("{file}:{line}: `{needle}` makes aurora-portrait depend on aurora-engine")
        })
        .collect();

    assert!(
        manifest_hits.is_empty() && import_hits.is_empty(),
        "rule 3 failed:\n{}\n{}",
        manifest_hits.join("\n"),
        import_hits.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Rule 4: contracts/ is the backend-neutral vocabulary.
// ---------------------------------------------------------------------------

#[test]
fn arch_rule4_contracts_are_backend_neutral() {
    const FORBIDDEN: &[&str] = &[
        "crate::mock",
        "crate::raw",
        "aurora_portrait",
        "std::os::windows",
    ];

    let root = workspace_root();
    let contracts = root.join("crates/engine/src/contracts");

    let mut files = Vec::new();
    walk_rs(&contracts, &mut files);

    let mut hits: Vec<String> = scan_dir(
        &root,
        "crates/engine/src/contracts",
        FORBIDDEN,
    )
    .into_iter()
    .map(|(file, line, needle)| {
        format!("{file}:{line}: `{needle}` is forbidden in engine::contracts")
    })
    .collect();

    // `include_str!`/`include_bytes!` of native GPU programs is also forbidden.
    for file in &files {
        let src = match fs::read_to_string(file) {
            Ok(src) => src,
            Err(_) => continue,
        };
        for (line_no, line) in code_lines(&src) {
            let embeds = line.contains("include_str!") || line.contains("include_bytes!");
            let native = line.contains("native/")
                || line.contains(".ptx")
                || line.contains(".metal")
                || line.contains(".msl");
            if embeds && native {
                hits.push(format!(
                    "{}:{line_no}: contracts must not embed a native GPU program",
                    rel(&root, file)
                ));
            }
        }
    }

    assert!(
        hits.is_empty(),
        "rule 4 failed (contracts must not reference mock/raw/aurora_portrait/platform FFI):\n{}",
        hits.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Rule 5: every CUDA-PTX-including source file is platform-gated.
// ---------------------------------------------------------------------------

/// Exact set of `crates/*/src` files that embed CUDA PTX. Adding a new backend
/// file requires updating this list on purpose.
const CUDA_PTX_FILES: &[&str] = &[
    "crates/engine/src/blemish_gainmap_cuda.rs",
    "crates/engine/src/mock/clarity_gpu.rs",
    "crates/engine/src/mock/detail_gpu.rs",
    "crates/engine/src/mock/develop_gpu.rs",
    "crates/engine/src/mock/nr_gpu.rs",
    "crates/engine/src/mock/pixel_gpu.rs",
    "crates/portrait/src/cuda.rs",
    "crates/portrait/src/cuda_beard.rs",
    "crates/portrait/src/cuda_blue_vein.rs",
    "crates/portrait/src/cuda_eye_bag.rs",
    "crates/portrait/src/cuda_foot_blue_vein.rs",
    "crates/portrait/src/cuda_gloss.rs",
    "crates/portrait/src/cuda_hair_detail.rs",
    "crates/portrait/src/cuda_hair_seam.rs",
    "crates/portrait/src/cuda_hand_neutral_gray.rs",
    "crates/portrait/src/cuda_hlf.rs",
    "crates/portrait/src/cuda_lip_line.rs",
    "crates/portrait/src/cuda_lut.rs",
    "crates/portrait/src/cuda_makeup.rs",
    "crates/portrait/src/cuda_mesh.rs",
    "crates/portrait/src/cuda_mesh_attribute.rs",
    "crates/portrait/src/cuda_model_preprocess.rs",
    "crates/portrait/src/cuda_moisture.rs",
    "crates/portrait/src/cuda_neutral_gray.rs",
    "crates/portrait/src/cuda_oil_shine_blend.rs",
    "crates/portrait/src/cuda_pf30025.rs",
    "crates/portrait/src/cuda_pf8050.rs",
    "crates/portrait/src/cuda_pf90021.rs",
    "crates/portrait/src/cuda_radiance.rs",
    "crates/portrait/src/cuda_skin_models.rs",
    "crates/portrait/src/cuda_smile.rs",
    "crates/portrait/src/cuda_stray_hair.rs",
    "crates/portrait/src/cuda_teeth.rs",
    "crates/portrait/src/cuda_teeth_blend.rs",
    "crates/portrait/src/cuda_texture_blend.rs",
    "crates/portrait/src/cuda_uvmap.rs",
    "crates/portrait/src/cuda_v2fuseoffset.rs",
    "crates/portrait/src/cuda_v2warp.rs",
];

/// Files that embed PTX but are neither module-gated nor contain an in-file
/// `#[cfg(windows)]`. They stay cross-platform through an explicit adapter stub
/// elsewhere; the reason is recorded here.
const NON_WINDOWS_ADAPTER_EXCEPTIONS: &[(&str, &str)] = &[(
    "crates/engine/src/blemish_gainmap_cuda.rs",
    "delegates to aurora_portrait::cuda, whose non-Windows impl is the stub at \
     crates/portrait/src/cuda.rs (cfg(not(windows)) Nvidia, ~line 3506)",
)];

#[test]
fn arch_rule5_cuda_ptx_includes_are_platform_gated() {
    let root = workspace_root();
    let mut found: BTreeMap<String, PathBuf> = BTreeMap::new();

    for crate_src in ["crates/engine/src", "crates/portrait/src"] {
        let crate_src_path = root.join(crate_src);
        let mut files = Vec::new();
        walk_rs(&crate_src_path, &mut files);
        for file in files {
            let src = match fs::read_to_string(&file) {
                Ok(src) => src,
                Err(_) => continue,
            };
            let embeds_ptx = code_lines(&src).any(|(_, line)| {
                (line.contains("include_str!") || line.contains("include_bytes!"))
                    && (line.contains("native/cuda") || line.contains(".ptx"))
            });
            if embeds_ptx {
                found.insert(rel(&root, &file), file);
            }
        }
    }

    let expected: BTreeSet<String> = CUDA_PTX_FILES.iter().map(|s| s.to_string()).collect();
    let actual: BTreeSet<String> = found.keys().cloned().collect();

    let added: Vec<String> = actual.difference(&expected).cloned().collect();
    let removed: Vec<String> = expected.difference(&actual).cloned().collect();

    let mut problems = Vec::new();
    for file in &added {
        problems.push(format!(
            "{file}: newly embeds CUDA PTX; add it to CUDA_PTX_FILES and document its \
             non-Windows path deliberately"
        ));
    }
    for file in &removed {
        problems.push(format!(
            "{file}: listed in CUDA_PTX_FILES but no longer embeds CUDA PTX; remove the \
             stale entry"
        ));
    }

    // Gating: module-level `#[cfg(windows)]`, an in-file cfg split, or a documented
    // explicit non-Windows adapter.
    for (file_rel, path) in &found {
        let crate_src = if file_rel.starts_with("crates/engine/") {
            root.join("crates/engine/src")
        } else {
            root.join("crates/portrait/src")
        };
        let src = fs::read_to_string(path).unwrap_or_default();
        let in_file_gate = src.contains("#[cfg(windows)]")
            || src.contains("cfg(target_os = \"windows\")");
        let module_gate = is_module_gated(&crate_src, path);
        let documented = NON_WINDOWS_ADAPTER_EXCEPTIONS
            .iter()
            .any(|(p, _)| p == file_rel);
        if !(in_file_gate || module_gate || documented) {
            problems.push(format!(
                "{file_rel}: embeds CUDA PTX but has no #[cfg(windows)] module gate, no \
                 in-file cfg(windows), and no documented non-Windows adapter"
            ));
        }
    }

    // A documented exception that no longer embeds PTX is stale.
    for (file, _) in NON_WINDOWS_ADAPTER_EXCEPTIONS {
        if !found.contains_key(*file) {
            problems.push(format!(
                "{file}: stale non-Windows adapter exception; no longer embeds CUDA PTX"
            ));
        }
    }

    assert!(
        problems.is_empty(),
        "rule 5 failed:\n{}",
        problems.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Rule 6: render_plan / contracts never reach into the mock process pipeline.
// ---------------------------------------------------------------------------

#[test]
fn arch_rule6_render_plan_and_contracts_do_not_import_process() {
    const FORBIDDEN: &[&str] = &["super::process", "crate::mock::process"];

    let root = workspace_root();
    let mut hits: Vec<String> = scan_dir(&root, "crates/engine/src/contracts", FORBIDDEN)
        .into_iter()
        .map(|(file, line, needle)| {
            format!("{file}:{line}: `{needle}` couples a contract to the mock process pipeline")
        })
        .collect();
    hits.extend(
        scan_file(&root, &root.join("crates/engine/src/render_plan.rs"), FORBIDDEN)
            .into_iter()
            .map(|(file, line, needle)| {
                format!(
                    "{file}:{line}: `{needle}` couples render_plan to the mock process pipeline"
                )
            }),
    );

    assert!(
        hits.is_empty(),
        "rule 6 failed (compiled plans/contracts must stay independent of mock::process):\n{}",
        hits.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Rule 8: modules migrated into `processing` never import the mock pipeline.
// ---------------------------------------------------------------------------

#[test]
fn arch_rule8_processing_does_not_import_mock_or_gpu_runtime() {
    const FORBIDDEN: &[&str] = &[
        // Any reference to the mock module, including a relative one such as
        // `super::super::mock::geometry`; a migrated stage must not depend on
        // where it used to live.
        "mock::",
        "super::process",
        "gpu_runtime",
    ];
    // Render-plan/source vocabulary introduced by the composition split. A
    // production stage must stay a pure pixel function: it must not reach for
    // the compiled plan (`render_plan` / `RenderPlan`) or the renderer's input
    // (`Source`). `FrameMap` now lives in `processing::frame_map`, so it is a
    // local geometry value and no longer forbidden. These are only checked
    // outside `#[cfg(test)]`: the clarity benchmark legitimately builds a
    // `crate::Source` fixture to measure the old renderer.
    const FORBIDDEN_NON_TEST_SUBSTR: &[&str] = &["render_plan", "RenderPlan"];
    const FORBIDDEN_NON_TEST_WORDS: &[&str] = &["Plan", "Source"];

    let root = workspace_root();
    let mut hits: Vec<String> = scan_dir(&root, "crates/engine/src/processing", FORBIDDEN)
        .into_iter()
        .map(|(file, line, needle)| {
            format!("{file}:{line}: `{needle}` makes a migrated stage depend on mock/GPU runtime")
        })
        .collect();
    hits.extend(
        scan_dir_non_test(
            &root,
            "crates/engine/src/processing",
            FORBIDDEN_NON_TEST_SUBSTR,
            FORBIDDEN_NON_TEST_WORDS,
        )
        .into_iter()
        .map(|(file, line, needle)| {
            format!(
                "{file}:{line}: `{needle}` couples a production processing stage to the render plan/source"
            )
        }),
    );

    assert!(
        hits.is_empty(),
        "rule 8 failed (processing must stay independent of mock::process, gpu_runtime, \
         render_plan and Source):\n{}",
        hits.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Rule 9: standalone lifecycle/queue modules stay independent of the engine's
// concrete storage and render implementations.
// ---------------------------------------------------------------------------

#[test]
fn arch_rule9_lifecycle_and_save_queue_stay_standalone() {
    const FORBIDDEN: &[&str] = &[
        "crate::mock",
        "crate::file_engine",
        "crate::portrait_stage",
    ];

    let root = workspace_root();
    let mut hits: Vec<String> = Vec::new();
    for file in ["crates/engine/src/job_lifecycle.rs", "crates/engine/src/save_queue.rs"] {
        hits.extend(
            scan_file(&root, &root.join(file), FORBIDDEN)
                .into_iter()
                .map(|(file, line, needle)| {
                    format!("{file}:{line}: `{needle}` couples a standalone protocol to a concrete impl")
                }),
        );
    }

    assert!(
        hits.is_empty(),
        "rule 9 failed (lifecycle/queue protocols must be implementable without the engine):\n{}",
        hits.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Rule 7: no dependency cycle among the three workspace packages.
// ---------------------------------------------------------------------------

#[test]
fn arch_rule7_no_dependency_cycle_among_workspace_packages() {
    const MANIFESTS: &[&str] = &[
        "crates/engine/Cargo.toml",
        "crates/app/Cargo.toml",
        "crates/portrait/Cargo.toml",
    ];

    let root = workspace_root();

    // First pass: collect package names so manifest keys can be matched against
    // real workspace packages (not external crates with similar names).
    let mut names = BTreeSet::new();
    let mut by_manifest: Vec<(String, PathBuf)> = Vec::new();
    for manifest_rel in MANIFESTS {
        let path = root.join(manifest_rel);
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let name = package_name(&text)
            .unwrap_or_else(|| panic!("no [package] name in {}", path.display()));
        names.insert(name.clone());
        by_manifest.push((name, path));
    }

    // Second pass: build the internal dependency graph.
    let mut graph: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (name, path) in &by_manifest {
        let deps = manifest_dep_keys(path);
        let internal: BTreeSet<String> = deps.intersection(&names).cloned().collect();
        graph.entry(name.clone()).or_default().extend(internal);
    }

    // DFS cycle detection with a visiting stack.
    fn visit(
        node: &str,
        graph: &BTreeMap<String, BTreeSet<String>>,
        color: &mut BTreeMap<String, u8>,
        stack: &mut Vec<String>,
        cycles: &mut Vec<String>,
    ) {
        color.insert(node.to_string(), 1);
        stack.push(node.to_string());
        if let Some(edges) = graph.get(node) {
            for dep in edges {
                match color.get(dep).copied().unwrap_or(0) {
                    0 => visit(dep, graph, color, stack, cycles),
                    1 => {
                        let start = stack.iter().position(|n| n == dep).unwrap_or(0);
                        let mut cycle = stack[start..].to_vec();
                        cycle.push(dep.clone());
                        cycles.push(cycle.join(" -> "));
                    }
                    _ => {}
                }
            }
        }
        stack.pop();
        color.insert(node.to_string(), 2);
    }

    let mut color: BTreeMap<String, u8> = BTreeMap::new();
    let mut cycles = Vec::new();
    for name in graph.keys() {
        if color.get(name).copied().unwrap_or(0) == 0 {
            let mut stack = Vec::new();
            visit(name, &graph, &mut color, &mut stack, &mut cycles);
        }
    }

    assert!(
        cycles.is_empty(),
        "rule 7 failed: workspace dependency cycle(s):\n{}",
        cycles.join("\n")
    );

    // Sanity: the three packages and their internal edges are exactly the ones the
    // architecture expects. This catches a silently dropped dependency too.
    let mut edges = Vec::new();
    for (name, deps) in &graph {
        for dep in deps {
            edges.push(format!("{name} -> {dep}"));
        }
    }
    edges.sort();
    assert_eq!(
        edges,
        vec![
            "aurora-engine -> aurora-portrait".to_string(),
            "portrait-app -> aurora-engine".to_string(),
        ],
        "rule 7 failed: unexpected internal dependency edges (got {edges:?})"
    );
}

// ---------------------------------------------------------------------------
// Rule 10: composition services borrow the engine; they never own it.
// ---------------------------------------------------------------------------

#[test]
fn arch_rule10_file_engine_services_borrow_the_engine() {
    // `file_engine_services.rs` holds one service per domain (`CatalogService`,
    // `SourceService`, ...). Each is a `struct ...Service<'a>` with
    // `engine: &'a FileEngine`; the engine keeps the single state lock, cache
    // and store. A service that owned an engine (or a second `Arc<Mutex<_>>`)
    // would reintroduce the second source of truth this split removes, and an
    // `impl FileEngine` here would put the methods back on the god object.
    const FORBIDDEN: &[&str] = &[
        "Arc<Mutex<FileEngine>>",
        "Arc<FileEngine>",
        "Box<FileEngine>",
        "Rc<FileEngine>",
        "impl FileEngine",
    ];

    let root = workspace_root();
    let path = root.join("crates/engine/src/file_engine_services.rs");
    let src = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let mut problems: Vec<String> = scan_file(&root, &path, FORBIDDEN)
        .into_iter()
        .map(|(file, line, needle)| {
            format!(
                "{file}:{line}: `{needle}` gives a service ownership of the engine; services \
                 must borrow `&FileEngine`"
            )
        })
        .collect();

    // A global engine and an owned `engine: FileEngine` field are the same
    // violation expressed in two other shapes.
    for (line_no, line) in code_lines(&src) {
        if line.contains("static") && line.contains("FileEngine") {
            problems.push(format!(
                "{}:{line_no}: a `static`/global `FileEngine` is forbidden; services borrow an \
                 instance",
                rel(&root, &path)
            ));
        }
        if line.contains("engine:") && line.contains("FileEngine") && !line.contains('&') {
            problems.push(format!(
                "{}:{line_no}: a service must declare `engine: &FileEngine`, not own it",
                rel(&root, &path)
            ));
        }
    }

    assert!(
        problems.is_empty(),
        "rule 10 failed (services must borrow, not own, the engine):\n{}",
        problems.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Rule 11: the app does not reference engine-internal composition services.
// ---------------------------------------------------------------------------

#[test]
fn arch_rule11_app_does_not_reference_engine_services() {
    // `file_engine_services` is an engine-internal composition detail. The app
    // keeps calling `ImageEngine`; a direct reference would let UI code depend
    // on how the engine is assembled. (Concrete `FileEngine` construction in
    // the app is already pinned by rule 1's counted exceptions.)
    const FORBIDDEN: &[&str] = &["file_engine_services"];

    let root = workspace_root();
    let hits: Vec<String> = scan_dir(&root, "crates/app/src", FORBIDDEN)
        .into_iter()
        .map(|(file, line, needle)| {
            format!(
                "{file}:{line}: `{needle}` is engine-internal; the app must keep using \
                 `aurora_engine::ImageEngine`"
            )
        })
        .collect();

    assert!(
        hits.is_empty(),
        "rule 11 failed (the app must not reference engine-internal services):\n{}",
        hits.join("\n")
    );
}
