//! Share immutable PSOs and their compatible root signature, never command allocators,
//! mapped parameters, fences or a model's buffers. Windows reuses its ID3D12Device
//! for the same live adapter; different devices must never exchange pipeline objects.
use super::{build_root_signature, Dx12Error, Kernel};
use std::{collections::HashMap, sync::{Arc, Mutex, OnceLock}};
use windows::Win32::Graphics::Direct3D12::{ID3D12Device, ID3D12RootSignature};

pub(super) struct Domain {
    device: ID3D12Device,
    pub(super) root: ID3D12RootSignature,
    compiled: Mutex<HashMap<(&'static str, &'static str), Kernel>>,
}

pub(super) fn for_device(device: &ID3D12Device) -> Result<Arc<Domain>, Dx12Error> {
    static DEVICES: OnceLock<Mutex<Vec<Arc<Domain>>>> = OnceLock::new();
    let mut all = DEVICES.get_or_init(Mutex::default).lock()
        .map_err(|_| Dx12Error::Runtime("pipeline cache lock poisoned".into()))?;
    all.retain(|d| unsafe { d.device.GetDeviceRemovedReason() }.is_ok());
    if let Some(found) = all.iter().find(|d| &d.device == device) { return Ok(found.clone()); }
    let domain = Arc::new(Domain { device: device.clone(), root: build_root_signature(device)?, compiled: Mutex::default() });
    // Dropping the registry's reference cannot invalidate a running session's Arc.
    if all.len() >= 8 { all.remove(0); }
    all.push(domain.clone());
    Ok(domain)
}

impl Domain {
    /// `true` means this invocation created the retained PSO. Do not retain an error:
    /// a transient driver or compiler failure must be retryable.
    pub(super) fn get_or_build(&self, source: &'static str, entry: &'static str,
        build: impl FnOnce() -> Result<Kernel, Dx12Error>) -> Result<(Kernel, bool), Dx12Error> {
        let mut map = self.compiled.lock().map_err(|_| Dx12Error::Runtime("shader cache lock poisoned".into()))?;
        if let Some(kernel) = map.get(&(source, entry)) { return Ok((kernel.clone(), false)); }
        let kernel = build()?;
        map.insert((source, entry), kernel.clone());
        Ok((kernel, true))
    }
}
