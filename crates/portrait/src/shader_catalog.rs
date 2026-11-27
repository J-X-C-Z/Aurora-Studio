//! Static catalogue of the first-batch portrait compute shaders.
//!
//! This module embeds all 26 shader sources (13 WGSL for WGPU targets and 13
//! Metal for Apple Silicon) at compile time via `include_str!`. It provides
//! source-level validation only: non-empty sources, correct entry points,
//! a uniform parameter struct and an explicit grid-boundary guard. It does
//! **not** compile, run or prove numerical parity with the CPU kernels; the
//! shaders are candidates with no production host dispatch wired yet.

/// One embedded shader pair for an effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderEntry {
    /// Vendor `ParamFlag` this shader pair implements.
    pub pf: u32,
    /// Stable effect name used as the Metal kernel and file stem.
    pub name: &'static str,
    /// WGSL source (`@compute` entry point `main`).
    pub wgsl: &'static str,
    /// Metal source (`kernel void <name>`).
    pub metal: &'static str,
}

/// The 13 first-batch effects, in ascending PF order.
pub static CATALOG: &[ShaderEntry] = &[
    ShaderEntry {
        pf: 0,
        name: "pf0_spotless",
        wgsl: include_str!("../native/wgsl/pf0_spotless.wgsl"),
        metal: include_str!("../native/metal/pf0_spotless.metal"),
    },
    ShaderEntry {
        pf: 101,
        name: "pf101_silkworm",
        wgsl: include_str!("../native/wgsl/pf101_silkworm.wgsl"),
        metal: include_str!("../native/metal/pf101_silkworm.metal"),
    },
    ShaderEntry {
        pf: 232,
        name: "pf232_skull_top",
        wgsl: include_str!("../native/wgsl/pf232_skull_top.wgsl"),
        metal: include_str!("../native/metal/pf232_skull_top.metal"),
    },
    ShaderEntry {
        pf: 247,
        name: "pf247_vface",
        wgsl: include_str!("../native/wgsl/pf247_vface.wgsl"),
        metal: include_str!("../native/metal/pf247_vface.metal"),
    },
    ShaderEntry {
        pf: 90023,
        name: "pf90023_eye_wrinkle",
        wgsl: include_str!("../native/wgsl/pf90023_eye_wrinkle.wgsl"),
        metal: include_str!("../native/metal/pf90023_eye_wrinkle.metal"),
    },
    ShaderEntry {
        pf: 90024,
        name: "pf90024_forehead_wrinkle",
        wgsl: include_str!("../native/wgsl/pf90024_forehead_wrinkle.wgsl"),
        metal: include_str!("../native/metal/pf90024_forehead_wrinkle.metal"),
    },
    ShaderEntry {
        pf: 90085,
        name: "pf90085_face_skin_unify",
        wgsl: include_str!("../native/wgsl/pf90085_face_skin_unify.wgsl"),
        metal: include_str!("../native/metal/pf90085_face_skin_unify.metal"),
    },
    ShaderEntry {
        pf: 90086,
        name: "pf90086_body_skin_unify",
        wgsl: include_str!("../native/wgsl/pf90086_body_skin_unify.wgsl"),
        metal: include_str!("../native/metal/pf90086_body_skin_unify.metal"),
    },
    ShaderEntry {
        pf: 100004,
        name: "pf100004_redness_remove",
        wgsl: include_str!("../native/wgsl/pf100004_redness_remove.wgsl"),
        metal: include_str!("../native/metal/pf100004_redness_remove.metal"),
    },
    ShaderEntry {
        pf: 21024,
        name: "pf21024_glabella_wrinkle",
        wgsl: include_str!("../native/wgsl/pf21024_glabella_wrinkle.wgsl"),
        metal: include_str!("../native/metal/pf21024_glabella_wrinkle.metal"),
    },
    ShaderEntry {
        pf: 21105,
        name: "pf21105_ai_body_skin_unify",
        wgsl: include_str!("../native/wgsl/pf21105_ai_body_skin_unify.wgsl"),
        metal: include_str!("../native/metal/pf21105_ai_body_skin_unify.metal"),
    },
    ShaderEntry {
        pf: 30009,
        name: "pf30009_hair_color_unify",
        wgsl: include_str!("../native/wgsl/pf30009_hair_color_unify.wgsl"),
        metal: include_str!("../native/metal/pf30009_hair_color_unify.metal"),
    },
    ShaderEntry {
        pf: 30013,
        name: "pf30013_neutral_gray",
        wgsl: include_str!("../native/wgsl/pf30013_neutral_gray.wgsl"),
        metal: include_str!("../native/metal/pf30013_neutral_gray.metal"),
    },
];

/// Static validation result for one shader pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Validation {
    pub pf: u32,
    pub name: &'static str,
    pub wgsl_bytes: usize,
    pub metal_bytes: usize,
}

/// Validate one embedded pair without compiling either language.
///
/// Checks performed:
/// - both sources are non-empty;
/// - WGSL has `@compute`, `@workgroup_size(16, 16, 1)`, a `struct Params`
///   (or another uniform struct bound through `var<uniform>`), `fn main` with
///   `@builtin(global_invocation_id)`, and an explicit
///   `gid.x >= params.width || gid.y >= params.height` guard;
/// - Metal has `#include <metal_stdlib>`, `kernel void <name>`, a
///   `static_assert(sizeof(` uniform-ABI assertion, and the same boundary
///   guard using the uniform variable.
pub fn validate_entry(entry: &ShaderEntry) -> Result<Validation, &'static str> {
    if entry.wgsl.is_empty() {
        return Err("empty WGSL source");
    }
    if entry.metal.is_empty() {
        return Err("empty Metal source");
    }
    let wgsl = entry.wgsl;
    if !wgsl.contains("@compute") {
        return Err("WGSL missing @compute");
    }
    if !wgsl.contains("@workgroup_size(16, 16, 1)") {
        return Err("WGSL missing workgroup_size(16, 16, 1)");
    }
    if !wgsl.contains("var<uniform>") {
        return Err("WGSL missing uniform binding");
    }
    if !wgsl.contains("fn main(") || !wgsl.contains("@builtin(global_invocation_id)") {
        return Err("WGSL missing main entry point");
    }
    if !wgsl.contains("gid.x >= params.width || gid.y >= params.height") {
        return Err("WGSL missing boundary guard");
    }
    let metal = entry.metal;
    if !metal.contains("#include <metal_stdlib>") {
        return Err("Metal missing metal_stdlib include");
    }
    let kernel_signature = format!("kernel void {}(", entry.name);
    if !metal.contains(&kernel_signature) {
        return Err("Metal missing matching kernel entry point");
    }
    if !metal.contains("static_assert(sizeof(") {
        return Err("Metal missing uniform ABI assertion");
    }
    if !metal.contains("gid.x >= p.width || gid.y >= p.height") {
        return Err("Metal missing boundary guard");
    }
    Ok(Validation {
        pf: entry.pf,
        name: entry.name,
        wgsl_bytes: wgsl.len(),
        metal_bytes: metal.len(),
    })
}

/// Validate all embedded entries. Returns one result per entry.
pub fn validate_all() -> Result<Vec<Validation>, (&'static str, &'static str)> {
    CATALOG
        .iter()
        .map(|entry| validate_entry(entry).map_err(|reason| (entry.name, reason)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_thirteen_unique_effect_pairs() {
        assert_eq!(CATALOG.len(), 13);
        let mut names = std::collections::HashSet::new();
        let mut pfs = std::collections::HashSet::new();
        for entry in CATALOG {
            assert!(names.insert(entry.name), "duplicate shader name {}", entry.name);
            assert!(pfs.insert(entry.pf), "duplicate pf {}", entry.pf);
        }
    }

    #[test]
    fn all_shaders_pass_static_validation() {
        let results = validate_all().expect("all embedded shaders must validate");
        assert_eq!(results.len(), 13);
        for result in results {
            assert!(result.wgsl_bytes > 0 && result.metal_bytes > 0);
        }
    }

    #[test]
    fn uniform_struct_and_boundary_guard_present_in_every_wgsl() {
        for entry in CATALOG {
            assert!(entry.wgsl.contains("struct Params {") || entry.wgsl.contains("struct "),
                    "{} missing uniform struct", entry.name);
            assert!(entry.wgsl.contains("return;"), "{} missing guard return", entry.name);
        }
    }
}
