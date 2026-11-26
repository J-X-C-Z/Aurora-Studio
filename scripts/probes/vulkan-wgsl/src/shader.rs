//! WGSL -> SPIR-V compilation through `naga` (the compiler wgpu itself uses), so
//! the shaders under `shaders/` stay WGSL source of truth while the probe talks
//! to Vulkan directly for exact submit/dispatch accounting.

use naga::back::spv;
use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};

pub struct Spv {
    pub words: Vec<u32>,
    pub bytes: usize,
    pub entry_points: Vec<String>,
    pub storage_bindings: usize,
}

pub fn compile(name: &str, source: &str) -> Result<Spv, String> {
    let module = wgsl::parse_str(source)
        .map_err(|e| format!("{name}: WGSL parse error\n{}", e.emit_to_string(source)))?;

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
    let info = validator
        .validate(&module)
        .map_err(|e| format!("{name}: WGSL validation error: {e:?}"))?;

    let options = spv::Options::default();
    let words = spv::write_vec(&module, &info, &options, None)
        .map_err(|e| format!("{name}: SPIR-V emit error: {e:?}"))?;

    let entry_points: Vec<String> = module.entry_points.iter().map(|e| e.name.clone()).collect();
    let storage_bindings = module
        .global_variables
        .iter()
        .filter(|(_, v)| matches!(v.space, naga::AddressSpace::Storage { .. }))
        .count();

    if words.first().copied() != Some(0x0723_0203) {
        return Err(format!("{name}: emitted SPIR-V has a bad magic word"));
    }

    Ok(Spv {
        bytes: words.len() * 4,
        words,
        entry_points,
        storage_bindings,
    })
}
