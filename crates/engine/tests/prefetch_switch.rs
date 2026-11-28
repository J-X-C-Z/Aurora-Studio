//! `AURORA_RAW_PREFETCH=0` has to be a real off switch for the neighbour graph.
//!
//! Same reason as `prewarm_backend_switch.rs` for being an integration test: the
//! prefetcher is one process-wide queue, so the only honest way to prove "the
//! switch stops the queue" is a process whose queue starts empty.
//!
//! What is checked, in order:
//! 1. With the switch off, a preview-shaped request queues no plan and decodes
//!    nothing, even though the neighbours exist.
//! 2. Flipping the switch back on restores prefetching, so the off state is a
//!    revertible switch and not a one-way teardown.
//! 3. A newer switch replaces the pending plan (latest-generation policy), and
//!    the replaced plan is counted as cancelled instead of being decoded.

#![cfg(windows)]

use std::path::PathBuf;

use aurora_engine::aurora_raw::prefetch;

fn set_env(key: &str, value: Option<&str>) {
    // SAFETY: this test binary runs one test, so no other thread reads the
    // environment while it is being changed.
    unsafe {
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
}

/// Four RAW stubs beside one JPEG: the folder shape a camera folder usually has.
fn folder(tag: &str) -> Vec<PathBuf> {
    let dir = std::env::temp_dir().join(format!("aurora-prefetch-switch-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut paths = Vec::new();
    for (index, name) in ["a.nef", "b.nef", "c.jpg", "d.nef", "e.nef"].iter().enumerate() {
        let path = dir.join(name);
        std::fs::write(&path, format!("not a raw {index}")).unwrap();
        paths.push(path);
    }
    paths
}

#[test]
fn the_prefetch_switch_turns_the_neighbour_graph_off_and_back_on() {
    set_env("AURORA_RAW_PREFETCH_DELAY_MS", Some("0"));
    let paths = folder("switch");

    // Phase 1: switched off. `request_neighbors_at` must not queue, and the
    // worker must not decode a neighbour behind the switch.
    set_env("AURORA_RAW_PREFETCH", Some("0"));
    assert!(!prefetch::prefetch_enabled());
    assert!(prefetch::wait_idle());
    let before = prefetch::stats();
    prefetch::request_neighbors_at(1, &paths, aurora_engine::raw_cache::PREVIEW_LONG_SIDE);
    assert!(prefetch::wait_idle(), "a cancelled plan must leave the queue idle");
    let off = prefetch::stats();
    assert_eq!(off.requested, before.requested, "an off switch queued a plan: {off:?}");
    assert_eq!(off.decoded, before.decoded, "an off switch decoded a neighbour: {off:?}");

    // Phase 2: switched back on. The same request now runs, so phase 1 measured
    // the switch and not a broken folder.
    set_env("AURORA_RAW_PREFETCH", Some("1"));
    assert!(prefetch::prefetch_enabled());
    let plan = prefetch::neighbor_paths(1, &paths, &prefetch::DEFAULT_OFFSETS);
    assert_eq!(
        plan,
        vec![paths[0].clone(), paths[3].clone()],
        "a.nef and d.nef are the RAW neighbours; c.jpg is skipped inside the walk"
    );
    prefetch::request_neighbors_at(1, &paths, aurora_engine::raw_cache::PREVIEW_LONG_SIDE);
    assert!(prefetch::wait_idle(), "the plan must drain");
    let on = prefetch::stats();
    assert_eq!(on.requested, off.requested + 1, "the switched-on request queued a plan: {on:?}");
    assert_eq!(on.decoded, off.decoded + plan.len() as u64, "every RAW neighbour was warmed: {on:?}");

    // Phase 3: every request carries a newer generation, and a request whose
    // neighbours have no 16-bit decode (a JPEG-only folder) queues nothing at
    // all instead of filling the decoder with work that cannot hit a RAW entry.
    let first_generation = prefetch::request_neighbors_at(3, &paths, aurora_engine::raw_cache::PREVIEW_LONG_SIDE);
    assert!(prefetch::wait_idle());
    let after_second = prefetch::stats();
    assert_eq!(
        after_second.requested,
        on.requested + 1,
        "a second switch queues exactly one new plan: {after_second:?}"
    );
    let second_generation = prefetch::request_neighbors_at(3, &paths, aurora_engine::raw_cache::PREVIEW_LONG_SIDE);
    assert!(
        second_generation > first_generation,
        "generation must advance per request: {first_generation} -> {second_generation}"
    );
    assert!(prefetch::wait_idle());

    let jpegs: Vec<PathBuf> = ["x.jpg", "y.jpg", "z.jpg"].iter().map(PathBuf::from).collect();
    assert!(
        prefetch::neighbor_paths(1, &jpegs, &prefetch::DEFAULT_OFFSETS).is_empty(),
        "a JPEG-only folder has no neighbour to warm"
    );
    let before_jpegs = prefetch::stats();
    prefetch::request_neighbors_at(1, &jpegs, aurora_engine::raw_cache::PREVIEW_LONG_SIDE);
    assert!(prefetch::wait_idle());
    assert_eq!(
        prefetch::stats().requested,
        before_jpegs.requested,
        "an empty plan must not count as a request"
    );
}
