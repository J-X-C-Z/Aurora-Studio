//! Direct3D 12 compute device layer for AMD / Intel GPUs on Windows.
//!
//! Every symbol comes from DLLs that ship with Windows 10/11: `d3d12.dll`,
//! `dxgi.dll` and `d3dcompiler_47.dll`. No third-party binary is loaded and no
//! runtime is redistributed alongside this crate.
//!
//! The adapter is chosen by explicit `VendorId`: AMD (`0x1002`) or Intel
//! (`0x8086`). NVIDIA (`0x10DE`) and the Microsoft WARP software rasterizer
//! (`0x1414`) are never selected, so a machine that only has an NVIDIA GPU
//! reports "no eligible adapter" instead of quietly running on something else.
//! `D3D12_FEATURE_DATA_ARCHITECTURE1` is queried on the selected device and its
//! `UMA` flag is reported, because an APU/iGPU shares system memory with the
//! CPU while a discrete card does not.
//!
//! Submission is explicit: work is recorded into one compute command list, then
//! `submit()` closes it, executes it on one queue, signals a fence and waits for
//! that fence value. Readback copies into a READBACK-heap staging buffer, maps
//! it, and returns bytes only after the fence has passed — so a value read here
//! is a value the GPU actually produced.
//!
//! Nothing in this module claims a dispatch happened. [`Dx12Device::info`]
//! reports the adapter that was opened, and every kernel carries the entry point
//! name it was compiled from, so a caller can state exactly what ran.

mod cache;
use std::collections::{HashMap,HashSet};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::core::Interface;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::Fxc::{D3DCOMPILE_OPTIMIZATION_LEVEL3, D3DCompile};
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_11_0, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
    D3D12_COMMAND_QUEUE_PRIORITY_NORMAL, D3D12_COMPUTE_PIPELINE_STATE_DESC,
    D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_CACHED_PIPELINE_STATE, D3D12_FEATURE_DATA_ARCHITECTURE1,
    D3D12_FEATURE_ARCHITECTURE1, D3D12_FENCE_FLAGS, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
    D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_TYPE_UPLOAD,
    D3D12_MEMORY_POOL_UNKNOWN, D3D12_PIPELINE_STATE_FLAG_NONE, D3D12_RANGE,
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_UAV, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATES,
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ,
    D3D12_RESOURCE_UAV_BARRIER, D3D12_ROOT_DESCRIPTOR, D3D12_ROOT_PARAMETER,
    D3D12_ROOT_PARAMETER_0, D3D12_ROOT_PARAMETER_TYPE_SRV,
    D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_SIGNATURE_DESC, D3D12_ROOT_SIGNATURE_FLAG_NONE,
    D3D12_SHADER_BYTECODE, D3D12_SHADER_VISIBILITY_ALL,
    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12SerializeRootSignature, D3D_ROOT_SIGNATURE_VERSION_1,
    D3D12CreateDevice, ID3D12CommandAllocator, ID3D12CommandQueue, ID3D12Device,
    ID3D12Fence, ID3D12GraphicsCommandList, ID3D12PipelineState, ID3D12Resource,
    ID3D12RootSignature, ID3D12CommandList,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_ADAPTER_DESC1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS,
    IDXGIAdapter1, IDXGIFactory4,
};


/// Adapter vendor ids this backend is allowed to select.
pub const VENDOR_AMD: u32 = 0x1002;
pub const VENDOR_INTEL: u32 = 0x8086;
/// Vendor ids that must never be selected by this backend.
pub const VENDOR_NVIDIA: u32 = 0x10DE;
pub const VENDOR_MICROSOFT_WARP: u32 = 0x1414;

/// Number of 32-bit root constants every kernel receives (`cbuffer Params : register(b0)`).
pub const PARAM_SLOTS: usize = 32;
/// Root descriptor slots for byte-address buffers: `t0..t7` read, `u0..u7` write.
pub const SRV_SLOTS: usize = 8;
pub const UAV_SLOTS: usize = 8;

pub type Params = [u32; PARAM_SLOTS];

#[derive(Debug)]
pub enum Dx12Error {
    /// No `d3d12.dll`/`dxgi.dll`/`d3dcompiler_47.dll`, or no AMD/Intel adapter.
    Unavailable(String),
    /// A D3D12/DXGI call returned a failing HRESULT.
    Call(&'static str, windows::core::Error),
    /// `D3DCompile` rejected the HLSL source.
    Shader { entry: String, log: String },
    /// The recorded work failed to execute or produced an unexpected state.
    Runtime(String),
}

impl fmt::Display for Dx12Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(why) => write!(f, "dx12 unavailable: {why}"),
            Self::Call(what, err) => write!(f, "dx12 {what} failed: {err}"),
            Self::Shader { entry, log } => write!(f, "dx12 shader `{entry}` failed to compile: {log}"),
            Self::Runtime(why) => write!(f, "dx12 runtime error: {why}"),
        }
    }
}

impl std::error::Error for Dx12Error {}

fn call<T>(what: &'static str, r: windows::core::Result<T>) -> Result<T, Dx12Error> {
    r.map_err(|e| Dx12Error::Call(what, e))
}

/// One DXGI adapter as the backend sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterInfo {
    pub index: u32,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub dedicated_video_memory: u64,
    pub shared_system_memory: u64,
    /// `D3D12_FEATURE_DATA_ARCHITECTURE1::UMA`: unified memory with the CPU.
    pub uma: bool,
    pub cache_coherent_uma: bool,
    /// `false` for the WARP software rasterizer.
    pub hardware: bool,
    /// This backend may select it (AMD or Intel, hardware, D3D12-capable).
    pub eligible: bool,
    /// Software or third-party adapter: listed for the record, never selected.
    pub excluded_reason: Option<&'static str>,
}

impl AdapterInfo {
    pub fn vendor_label(&self) -> &'static str {
        match self.vendor_id {
            VENDOR_AMD => "AMD",
            VENDOR_INTEL => "Intel",
            VENDOR_NVIDIA => "NVIDIA",
            VENDOR_MICROSOFT_WARP => "Microsoft (WARP)",
            _ => "unknown",
        }
    }
}

/// Which adapter to open. Selection is always restricted to AMD/Intel hardware.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdapterRequest {
    /// DXGI adapter index to use; must still be eligible.
    pub index: Option<u32>,
    /// `"amd"` or `"intel"`.
    pub vendor: Option<String>,
}

impl AdapterRequest {
    /// Reads `AURORA_DX12_ADAPTER` (`amd` / `intel` / an adapter index).
    pub fn from_env() -> Self {
        match std::env::var("AURORA_DX12_ADAPTER") {
            Ok(raw) => Self::parse(&raw),
            Err(_) => Self::default(),
        }
    }

    pub fn parse(raw: &str) -> Self {
        let t = raw.trim().to_ascii_lowercase();
        if t.is_empty() {
            return Self::default();
        }
        if let Ok(index) = t.parse::<u32>() {
            return Self { index: Some(index), vendor: None };
        }
        Self { index: None, vendor: Some(t) }
    }
}

fn adapter_name(desc: &DXGI_ADAPTER_DESC1) -> String {
    let end = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
    String::from_utf16_lossy(&desc.Description[..end])
}

/// Opens a D3D12 device on `adapter`, returning whether D3D12 accepted it.
fn d3d12_capable(adapter: &IDXGIAdapter1) -> Option<ID3D12Device> {
    let mut device: Option<ID3D12Device> = None;
    // SAFETY: `device` is a local out-parameter; the call either fills it or fails.
    unsafe { D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, &mut device) }.ok()?;
    device
}

fn architecture(device: &ID3D12Device) -> (bool, bool) {
    let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE1::default();
    let size = std::mem::size_of::<D3D12_FEATURE_DATA_ARCHITECTURE1>() as u32;
    let ok = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_ARCHITECTURE1,
            (&raw mut arch).cast::<c_void>(),
            size,
        )
    }
    .is_ok();
    if ok {
        (arch.UMA.as_bool(), arch.CacheCoherentUMA.as_bool())
    } else {
        (false, false)
    }
}

/// Enumerates every DXGI adapter, marking which ones this backend may select.
///
/// Listing does not open a long-lived device: `D3D12CreateDevice` is called only
/// to prove capability and the result is dropped immediately.
pub fn enumerate_adapters() -> Result<Vec<AdapterInfo>, Dx12Error> {
    let factory: IDXGIFactory4 = call("CreateDXGIFactory2", unsafe {
        CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))
    })?;
    let mut out = Vec::new();
    let mut index = 0u32;
    loop {
        let adapter = match unsafe { factory.EnumAdapters1(index) } {
            Ok(a) => a,
            Err(_) => break,
        };
        let desc = call("IDXGIAdapter1::GetDesc1", unsafe { adapter.GetDesc1() })?;
        let software = desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0;
        let device = if software { None } else { d3d12_capable(&adapter) };
        let (uma, cache_coherent_uma) = match &device {
            Some(d) => architecture(d),
            None => (false, false),
        };
        let excluded_reason = if software {
            Some("software rasterizer (WARP)")
        } else if device.is_none() {
            Some("no D3D12 device")
        } else if desc.VendorId == VENDOR_NVIDIA {
            Some("NVIDIA adapter: this backend drives AMD/Intel only")
        } else if desc.VendorId == VENDOR_MICROSOFT_WARP {
            Some("Microsoft adapter")
        } else if desc.VendorId != VENDOR_AMD && desc.VendorId != VENDOR_INTEL {
            Some("vendor is neither AMD nor Intel")
        } else {
            None
        };
        out.push(AdapterInfo {
            index,
            name: adapter_name(&desc),
            vendor_id: desc.VendorId,
            device_id: desc.DeviceId,
            dedicated_video_memory: desc.DedicatedVideoMemory as u64,
            shared_system_memory: desc.SharedSystemMemory as u64,
            uma,
            cache_coherent_uma,
            hardware: !software,
            eligible: excluded_reason.is_none(),
            excluded_reason,
        });
        index += 1;
    }
    Ok(out)
}

/// A GPU buffer this backend owns.
pub struct GpuBuffer {
    pub name: String,
    pub size: u64,
    pub id: u64,
    resource: ID3D12Resource,
}

impl GpuBuffer {
    /// GPU virtual address for a root descriptor.
    pub fn gpu_va(&self) -> u64 {
        unsafe { self.resource.GetGPUVirtualAddress() }
    }
}

/// A compiled compute shader ready for dispatch.
#[derive(Clone)]
pub struct Kernel {
    pub entry: String,
    pub bytecode_len: usize,
    pub pipeline: ID3D12PipelineState,
}

/// One buffer view bound to a root descriptor slot.
#[derive(Clone, Copy)]
pub struct BoundBuffer<'a> {
    pub buffer: &'a GpuBuffer,
    /// Byte offset of element 0 inside `buffer` (arena slices).
    pub byte_offset: u64,
}

impl BoundBuffer<'_> {
    /// GPU virtual address the root descriptor points at.
    pub fn gpu_va(&self) -> u64 {
        self.buffer.gpu_va() + self.byte_offset
    }
}

/// Root-descriptor binding for one dispatch.
#[derive(Default)]
pub struct Bindings<'a> {
    pub srvs: Vec<(u32, BoundBuffer<'a>)>,
    pub uavs: Vec<(u32, BoundBuffer<'a>)>,
}

/// An open D3D12 compute device: one queue, one allocator, one command list.
///
/// `Send`/`Sync` are implemented because every method takes `&mut self`: callers
/// serialize access (the session wraps this in a `Mutex`), which is the
/// threading rule D3D12 command lists require.
pub struct Dx12Device {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    fence_value: u64,
    root: ID3D12RootSignature,
    cache: std::sync::Arc<cache::Domain>,
    info: AdapterInfo,
    list_open: bool,
    /// Buffers written since the last UAV barrier; read-before-write needs one.
    written: HashSet<u64>,
    states: HashMap<u64,D3D12_RESOURCE_STATES>,
    param_pages: Vec<(ID3D12Resource,*mut u8)>,
    param_cursor: usize,
    /// UPLOAD staging resources kept alive until the next submit completes.
    staging: Vec<ID3D12Resource>,
    /// Total bytes of DEFAULT-heap buffers allocated by this device.
    allocated_bytes: u64,
    next_id: u64,
    dispatches: u64,
}

// SAFETY: all access is `&mut self`, and the session serializes it behind a Mutex.
unsafe impl Send for Dx12Device {}
unsafe impl Sync for Dx12Device {}

impl Dx12Device {
    /// Opens the AMD/Intel adapter described by `request` and builds the shared
    /// root signature every kernel uses.
    pub fn open(request: &AdapterRequest) -> Result<(Self, AdapterInfo), Dx12Error> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let adapters = enumerate_adapters()?;
        let chosen = select_adapter(&adapters, request)?;
        let factory: IDXGIFactory4 = call("CreateDXGIFactory2", unsafe {
            CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))
        })?;
        let adapter = call(
            "IDXGIFactory4::EnumAdapters1",
            unsafe { factory.EnumAdapters1(chosen.index) },
        )?;
        let device = d3d12_capable(&adapter).ok_or_else(|| {
            Dx12Error::Unavailable(format!(
                "adapter {} ({}) refused a D3D12 device",
                chosen.index, chosen.name
            ))
        })?;
        let (uma, cache_coherent_uma) = architecture(&device);
        let info = AdapterInfo { uma, cache_coherent_uma, ..chosen.clone() };

        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        };
        let queue: ID3D12CommandQueue =
            call("ID3D12Device::CreateCommandQueue", unsafe {
                device.CreateCommandQueue(&queue_desc)
            })?;
        let allocator: ID3D12CommandAllocator =
            call("ID3D12Device::CreateCommandAllocator", unsafe {
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)
            })?;
        let list: ID3D12GraphicsCommandList = call(
            "ID3D12Device::CreateCommandList",
            unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &allocator, None) },
        )?;
        let fence: ID3D12Fence = call("ID3D12Device::CreateFence", unsafe {
            device.CreateFence(0, D3D12_FENCE_FLAGS(0))
        })?;
        let cache=cache::for_device(&device)?;
        let root=cache.root.clone();

        let mut me = Self {
            device,
            queue,
            allocator,
            list,
            fence,
            fence_value: 0,
            root,
            cache,
            info: info.clone(),
            list_open: true,
            written: HashSet::new(),
            states: HashMap::new(),
            param_pages:Vec::new(), param_cursor:0,
            staging: Vec::new(),
            allocated_bytes: 0,
            next_id: NEXT.fetch_add(1, Ordering::Relaxed) << 32,
            dispatches: 0,
        };
        // Reject an unexpected device-removal before the first dispatch.
        let _ = me.device_removed_reason()?;
        Ok((me, info))
    }

    pub fn info(&self) -> &AdapterInfo {
        &self.info
    }

    /// Bytes of DEFAULT-heap GPU memory this device has allocated.
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// Number of compute dispatches recorded by this device.
    pub fn dispatch_count(&self) -> u64 {
        self.dispatches
    }

    fn device_removed_reason(&self) -> Result<(), Dx12Error> {
        unsafe { self.device.GetDeviceRemovedReason() }
            .map_err(|e| Dx12Error::Call("ID3D12Device::GetDeviceRemovedReason", e))
    }

    fn ensure_open(&mut self) -> Result<(), Dx12Error> {
        if !self.list_open {
            call("ID3D12CommandAllocator::Reset", unsafe { self.allocator.Reset() })?;
            call("ID3D12GraphicsCommandList::Reset", unsafe {
                self.list.Reset(&self.allocator, None)
            })?;
            self.list_open = true;
        }
        Ok(())
    }

    /// The shipping kernels have static source; reuse the exact PSO on this adapter.
    pub fn compile_cached(&self,source:&'static str,entry:&'static str)->Result<(Kernel,bool),Dx12Error>{
        self.cache.get_or_build(source,entry,||self.compile(source,entry))
    }

    /// Compiles one HLSL compute entry point from `source`.
    pub fn compile(&self, source: &str, entry: &str) -> Result<Kernel, Dx12Error> {
        let entry_c = std::ffi::CString::new(entry)
            .map_err(|_| Dx12Error::Runtime(format!("entry point `{entry}` has an interior NUL")))?;
        let target = c"cs_5_1";
        let mut blob: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: source pointer/length describe a live `str`; out-params are locals.
        let hr = unsafe {
            D3DCompile(
                source.as_ptr().cast::<c_void>(),
                source.len(),
                windows::core::PCSTR(c"aurora_infer_dx12".as_ptr().cast()),
                None,
                None,
                windows::core::PCSTR(entry_c.as_ptr().cast()),
                windows::core::PCSTR(target.as_ptr().cast()),
                D3DCOMPILE_OPTIMIZATION_LEVEL3,
                0,
                &mut blob,
                Some(&mut errors),
            )
        };
        if let Err(e) = hr {
            let log = errors
                .as_ref()
                .map(|b| blob_text(b))
                .unwrap_or_else(|| e.message().to_string());
            return Err(Dx12Error::Shader { entry: entry.to_string(), log });
        }
        let blob = blob.ok_or_else(|| Dx12Error::Runtime("D3DCompile returned no bytecode".into()))?;
        let bytes = unsafe {
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
        };
        let (code_ptr, code_len) =
            unsafe { (blob.GetBufferPointer(), blob.GetBufferSize()) };
        let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: the root signature outlives the pipeline state in `self`.
            pRootSignature: std::mem::ManuallyDrop::new(Some(self.root.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: code_ptr,
                BytecodeLength: code_len,
            },
            NodeMask: 0,
            CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
            Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
        };
        let pipeline: ID3D12PipelineState =
            call("ID3D12Device::CreateComputePipelineState", unsafe {
                let made=self.device.CreateComputePipelineState(&desc);std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);made
            })?;
        Ok(Kernel { entry: entry.to_string(), bytecode_len: bytes.len(), pipeline })
    }

    /// Allocates a DEFAULT-heap UAV buffer of at least `bytes`.
    pub fn alloc(&mut self, name: impl Into<String>, bytes: u64) -> Result<GpuBuffer, Dx12Error> {
        let size = bytes.max(4).next_multiple_of(256);
        let resource = self.new_resource(
            D3D12_HEAP_TYPE_DEFAULT,
            size,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            "ID3D12Device::CreateCommittedResource(default)",
        )?;
        self.allocated_bytes += size;
        let id = self.next_id;
        self.next_id += 1;
        Ok(GpuBuffer { name: name.into(), size, id, resource })
    }

    fn new_resource(
        &self,
        heap: windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE,
        size: u64,
        state: D3D12_RESOURCE_STATES,
        flags: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAGS,
        what: &'static str,
    ) -> Result<ID3D12Resource, Dx12Error> {
        let props = D3D12_HEAP_PROPERTIES {
            Type: heap,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: if heap == D3D12_HEAP_TYPE_DEFAULT { flags } else { D3D12_RESOURCE_FLAG_NONE },
        };
        let mut resource: Option<ID3D12Resource> = None;
        call(what, unsafe {
            self.device.CreateCommittedResource(
                &props,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut resource,
            )
        })?;
        resource.ok_or_else(|| Dx12Error::Runtime(format!("{what}: no resource returned")))
    }

    // DEFAULT buffers may decay to COMMON only after ExecuteCommandLists completes.
    // A UAV barrier orders UAV writes, but cannot transition COPY_DEST -> SRV -> UAV.
    fn write_constants(&mut self,p:&Params)->Result<u64,Dx12Error>{
        const STRIDE:usize=256; const PAGE:usize=65536;
        let page=self.param_cursor/(PAGE/STRIDE);let offset=(self.param_cursor%(PAGE/STRIDE))*STRIDE;
        if page==self.param_pages.len(){
            let upload=self.new_resource(D3D12_HEAP_TYPE_UPLOAD,PAGE as u64,D3D12_RESOURCE_STATE_GENERIC_READ,D3D12_RESOURCE_FLAG_NONE,"allocate parameter upload page")?;
            let mut mapped=std::ptr::null_mut();let range=D3D12_RANGE{Begin:0,End:0};
            call("map parameter upload page",unsafe{ upload.Map(0,Some(&range),Some(&mut mapped)) })?;
            if mapped.is_null(){return Err(Dx12Error::Runtime("null parameter upload mapping".into()));}
            self.param_pages.push((upload,mapped.cast()));
        }
        let (resource,bytes)=&self.param_pages[page];
        unsafe{std::ptr::copy_nonoverlapping(p.as_ptr().cast::<u8>(),bytes.add(offset),std::mem::size_of_val(p));}
        self.param_cursor+=1; Ok(unsafe{resource.GetGPUVirtualAddress()}+offset as u64)
    }

    fn transition(&mut self, buf: &GpuBuffer, after: D3D12_RESOURCE_STATES) {
        use windows::Win32::Graphics::Direct3D12::*;
        let before = self.states.get(&buf.id).copied().unwrap_or(D3D12_RESOURCE_STATE_COMMON);
        if before == after {return;}
        let mut barrier=D3D12_RESOURCE_BARRIER { Type:D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,Flags:D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous:D3D12_RESOURCE_BARRIER_0{Transition:std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER{
                pResource:std::mem::ManuallyDrop::new(Some(buf.resource.clone())),Subresource:D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore:before, StateAfter:after })}};
        unsafe {self.list.ResourceBarrier(std::slice::from_ref(&barrier));std::mem::ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);}
        self.states.insert(buf.id,after);self.written.remove(&buf.id);
    }

    fn uav_barrier(&mut self, buffers: &[&GpuBuffer]) -> Result<(), Dx12Error> {
        if buffers.is_empty() {
            return Ok(());
        }
        self.ensure_open()?;
        let mut barriers: Vec<D3D12_RESOURCE_BARRIER> = buffers
            .iter()
            .map(|b| D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(Some(b.resource.clone())),
                    }),
                },
            })
            .collect();
        unsafe { self.list.ResourceBarrier(&barriers); for b in &mut barriers {std::mem::ManuallyDrop::drop(&mut (*b.Anonymous.UAV).pResource);} };
        Ok(())
    }

    /// Copies `data` into `buf` through a UPLOAD staging buffer.
    pub fn upload(&mut self, buf: &GpuBuffer, data: &[u8]) -> Result<(), Dx12Error> {
        if data.len() as u64 > buf.size {
            return Err(Dx12Error::Runtime(format!(
                "upload of {} bytes exceeds buffer `{}` of {} bytes",
                data.len(),
                buf.name,
                buf.size
            )));
        }
        if data.is_empty() {
            return Ok(());
        }
        self.ensure_open()?;
        self.transition(buf, D3D12_RESOURCE_STATE_COPY_DEST);
        let size = (data.len() as u64).next_multiple_of(256);
        let staging = self.new_resource(
            D3D12_HEAP_TYPE_UPLOAD,
            size,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
            "ID3D12Device::CreateCommittedResource(upload)",
        )?;
        let range = D3D12_RANGE { Begin: 0, End: 0 };
        let mut ptr: *mut c_void = std::ptr::null_mut();
        call("ID3D12Resource::Map(upload)", unsafe {
            staging.Map(0, Some(&range), Some(&mut ptr as *mut *mut c_void))
        })?;
        if ptr.is_null() {
            return Err(Dx12Error::Runtime("Map(upload) returned null".into()));
        }
        // SAFETY: the mapped region is `size >= data.len()` bytes of write-combined memory.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast::<u8>(), data.len()) };
        unsafe { staging.Unmap(0, None) };
        unsafe {
            self.list
                .CopyBufferRegion(&buf.resource, 0, &staging, 0, data.len() as u64)
        };
        self.staging.push(staging);
        Ok(())
    }

    /// Runs every recorded command and blocks until the fence passes.
    pub fn submit(&mut self) -> Result<(), Dx12Error> {
        if !self.list_open {
            return Ok(());
        }
        call("ID3D12GraphicsCommandList::Close", unsafe { self.list.Close() })?;
        self.list_open = false;
        let list: ID3D12CommandList = self.list.cast().map_err(|e| Dx12Error::Call("cast", e))?;
        unsafe { self.queue.ExecuteCommandLists(&[Some(list)]) };
        self.fence_value += 1;
        let value = self.fence_value;
        call("ID3D12CommandQueue::Signal", unsafe {
            self.queue.Signal(&self.fence, value)
        })?;
        let done = unsafe { self.fence.GetCompletedValue() };
        if done < value {
            // A fence wait without an event handle: poll the completed value. The
            // kernel batches issued here are milliseconds long, and polling keeps
            // this layer free of the event/security API surface.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            loop {
                let done = unsafe { self.fence.GetCompletedValue() };
                if done >= value {
                    break;
                }
                let _ = self.device_removed_reason()?;
                if std::time::Instant::now() > deadline {
                    return Err(Dx12Error::Runtime(format!(
                        "fence value {value} not reached within 30s (last completed {done})"
                    )));
                }
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        }
        self.staging.clear();
        self.param_cursor=0; // only recycle after a completed fence
        self.states.clear(); // all buffers decay to COMMON after this completed ExecuteCommandLists
        self.written.clear();
        self.device_removed_reason()
    }

    /// Reads `len` bytes back from the start of `buf` after the GPU has finished.
    pub fn read(&mut self, buf: &GpuBuffer, len: usize) -> Result<Vec<u8>, Dx12Error> {
        if len as u64 > buf.size {
            return Err(Dx12Error::Runtime(format!(
                "readback of {len} bytes exceeds buffer `{}` of {} bytes",
                buf.name, buf.size
            )));
        }
        self.ensure_open()?;
        self.transition(buf, windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_SOURCE);
        let size = (len as u64).max(4).next_multiple_of(256);
        let staging = self.new_resource(
            D3D12_HEAP_TYPE_READBACK,
            size,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_FLAG_NONE,
            "ID3D12Device::CreateCommittedResource(readback)",
        )?;
        unsafe {
            self.list
                .CopyBufferRegion(&staging, 0, &buf.resource, 0, len as u64)
        };
        self.submit()?;
        let range = D3D12_RANGE { Begin: 0, End: len };
        let mut ptr: *mut c_void = std::ptr::null_mut();
        call("ID3D12Resource::Map(readback)", unsafe {
            staging.Map(0, Some(&range), Some(&mut ptr as *mut *mut c_void))
        })?;
        if ptr.is_null() {
            return Err(Dx12Error::Runtime("Map(readback) returned null".into()));
        }
        // SAFETY: `len` bytes were copied by the GPU into this READBACK resource.
        let out = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len) }.to_vec();
        let written_range = D3D12_RANGE { Begin: 0, End: 0 };
        unsafe { staging.Unmap(0, Some(&written_range)) };
        Ok(out)
    }

    /// Records one dispatch: binds the shared root signature, `params`, the
    /// buffers in `bindings`, then dispatches `groups`.
    pub fn dispatch(
        &mut self,
        kernel: &Kernel,
        params: &Params,
        bindings: &Bindings<'_>,
        groups: (u32, u32, u32),
    ) -> Result<(), Dx12Error> {
        if [groups.0,groups.1,groups.2].into_iter().any(|n|n>65535) {
            return Err(Dx12Error::Runtime(format!("D3D12 dispatch {groups:?} exceeds the per-axis limit")));
        }
        self.ensure_open()?;
        use windows::Win32::Graphics::Direct3D12::{D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
        for (_, binding) in &bindings.srvs {
            if bindings.uavs.iter().any(|(_,u)|u.buffer.id==binding.buffer.id) {return Err(Dx12Error::Runtime("same buffer bound for SRV read and UAV write".into()));}
            self.transition(binding.buffer,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        for (_, binding) in &bindings.uavs {self.transition(binding.buffer,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);}
        // A buffer written earlier in this list and read now needs a UAV barrier.
        let mut hazards: Vec<&GpuBuffer> = Vec::new();
        for (_, bound) in bindings.srvs.iter().chain(bindings.uavs.iter()) {
            if self.written.contains(&bound.buffer.id)
                && !hazards.iter().any(|b| b.id == bound.buffer.id)
            {
                hazards.push(bound.buffer);
            }
        }
        self.uav_barrier(&hazards)?;
        let param_va=self.write_constants(params)?;
        unsafe {
            self.list.SetComputeRootSignature(&self.root);
            self.list.SetPipelineState(&kernel.pipeline);
            self.list.SetComputeRootConstantBufferView(0, param_va);
            for (slot, bound) in &bindings.srvs {
                let idx = 1 + *slot as usize;
                self.list.SetComputeRootShaderResourceView(idx as u32, bound.gpu_va());
            }
            for (slot, bound) in &bindings.uavs {
                let idx = 1 + SRV_SLOTS + *slot as usize;
                self.list.SetComputeRootUnorderedAccessView(idx as u32, bound.gpu_va());
            }
            self.list.Dispatch(groups.0, groups.1, groups.2);
        }
        for (_, bound) in &bindings.uavs {
            self.written.insert(bound.buffer.id);
        }
        self.dispatches += 1;
        Ok(())
    }
}

impl Drop for Dx12Device {
    fn drop(&mut self) {
        let _ = self.submit();
        for (p,_) in &self.param_pages {unsafe{p.Unmap(0,None);}}
    }
}

fn blob_text(blob: &ID3DBlob) -> String {
    let bytes = unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    };
    String::from_utf8_lossy(bytes).into_owned()
}

/// Root layout shared by every kernel in `native/hlsl/infer_core.hlsl`:
/// `b0` = a root CBV pointing to 32 packed DWORDs, `t0..t7` = read buffers, `u0..u7` = read/write buffers.
fn build_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, Dx12Error> {
    let mut params = Vec::with_capacity(1 + SRV_SLOTS + UAV_SLOTS);
    params.push(D3D12_ROOT_PARAMETER {
        ParameterType: windows::Win32::Graphics::Direct3D12::D3D12_ROOT_PARAMETER_TYPE_CBV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {Descriptor: D3D12_ROOT_DESCRIPTOR {ShaderRegister:0,RegisterSpace:0}},
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    });
    for slot in 0..SRV_SLOTS {
        params.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: slot as u32,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
    }
    for slot in 0..UAV_SLOTS {
        params.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: slot as u32,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
    }
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    let mut blob: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    call("D3D12SerializeRootSignature", unsafe {
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, Some(&mut errors))
    })
    .map_err(|e| match errors {
        Some(log) => Dx12Error::Runtime(format!("root signature rejected: {}", blob_text(&log))),
        None => e,
    })?;
    let blob = blob.ok_or_else(|| Dx12Error::Runtime("root signature blob missing".into()))?;
    let bytes = unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    };
    call("ID3D12Device::CreateRootSignature", unsafe {
        device.CreateRootSignature(0, bytes)
    })
}

/// Discover only an eligible hardware adapter matching the override; cold inventory, not execution.
pub fn probe_adapter(request:&AdapterRequest)->Result<AdapterInfo,Dx12Error>{
    select_adapter(&enumerate_adapters()?,request)
}

fn select_adapter(adapters: &[AdapterInfo], request: &AdapterRequest) -> Result<AdapterInfo, Dx12Error> {
    let eligible: Vec<&AdapterInfo> = adapters.iter().filter(|a| a.eligible).collect();
    if eligible.is_empty() {
        let seen: Vec<String> = adapters
            .iter()
            .map(|a| {
                format!(
                    "#{} {} [{} vendor=0x{:04X} uma={} {}]",
                    a.index,
                    a.name,
                    a.vendor_label(),
                    a.vendor_id,
                    a.uma,
                    a.excluded_reason.unwrap_or("eligible")
                )
            })
            .collect();
        return Err(Dx12Error::Unavailable(format!(
            "no AMD/Intel D3D12 adapter; enumerated: [{}]",
            seen.join(", ")
        )));
    }
    if let Some(index) = request.index {
        return eligible.iter().find(|a| a.index == index).map(|a| (*a).clone()).ok_or_else(|| {
            Dx12Error::Unavailable(format!(
                "AURORA_DX12_ADAPTER={index} is not an eligible AMD/Intel adapter (eligible: {:?})",
                eligible.iter().map(|a| a.index).collect::<Vec<_>>()
            ))
        });
    }
    if let Some(vendor) = &request.vendor {
        let want = match vendor.as_str() {
            "amd" => VENDOR_AMD,
            "intel" => VENDOR_INTEL,
            other => {
                return Err(Dx12Error::Unavailable(format!(
                    "AURORA_DX12_ADAPTER={other:?} must be `amd`, `intel` or an adapter index"
                )));
            }
        };
        return eligible.iter().find(|a| a.vendor_id == want).map(|a| (*a).clone()).ok_or_else(|| {
            Dx12Error::Unavailable(format!("no eligible {vendor} adapter"))
        });
    }
    // Default: most dedicated video memory, then lowest index (deterministic).
    let mut best = eligible[0];
    for a in eligible.iter().skip(1) {
        if (a.dedicated_video_memory, std::cmp::Reverse(a.index))
            > (best.dedicated_video_memory, std::cmp::Reverse(best.index))
        {
            best = a;
        }
    }
    Ok(best.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_parsing_never_accepts_nvidia() {
        assert_eq!(AdapterRequest::parse("amd").vendor.as_deref(), Some("amd"));
        assert_eq!(AdapterRequest::parse(" INTEL ").vendor.as_deref(), Some("intel"));
        assert_eq!(AdapterRequest::parse("1").index, Some(1));
        assert_eq!(AdapterRequest::parse(""), AdapterRequest { index: None, vendor: None });
        // A vendor name that is not AMD/Intel can only fail later, never select.
        assert_eq!(AdapterRequest::parse("nvidia").vendor.as_deref(), Some("nvidia"));
    }

    const PROBE_HLSL: &str = r#"
cbuffer Params : register(b0) { uint4 packed[8]; };
#define ARG(i) (packed[(i)>>2][(i)&3])
ByteAddressBuffer s0 : register(t0);
RWByteAddressBuffer u0 : register(u0);

[numthreads(64,1,1)]
void cs_probe_params(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint i = gid.x * 64 + tid.x;
    if (i < 32) { u0.Store(i * 4, asuint(asfloat(ARG(i)))); }
}

[numthreads(64,1,1)]
void cs_probe_copy(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint i = gid.x * 64 + tid.x;
    if (i < ARG(0)) { u0.Store(i * 4, s0.Load(i * 4)); }
}
"#;

    /// Proves root constants and root SRV/UAV descriptors reach the shader.
    #[test]
    #[ignore = "needs a Windows host with an AMD or Intel D3D12 adapter"]
    fn root_constants_and_buffer_bindings_reach_the_shader() {
        let (mut device, info) = Dx12Device::open(&AdapterRequest::parse("amd")).expect("open");
        let params_kernel = device.compile(PROBE_HLSL, "cs_probe_params").expect("compile probe");
        let copy_kernel = device.compile(PROBE_HLSL, "cs_probe_copy").expect("compile copy");

        let dst = device.alloc("dst", 32 * 4).expect("alloc dst");
        let mut params: Params = [0; PARAM_SLOTS];
        for (i, slot) in params.iter_mut().enumerate() {
            *slot = (i as f32).to_bits();
        }
        let bindings = Bindings { srvs: vec![], uavs: vec![(0, BoundBuffer { buffer: &dst, byte_offset: 0 })] };
        device.dispatch(&params_kernel, &params, &bindings, (1, 1, 1)).expect("dispatch params");
        let bytes = device.read(&dst, 32 * 4).expect("read params");
        let values: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        eprintln!("[probe] adapter={:?} params[0..8]={:?}", info.name, &values[..8]);
        assert_eq!(values, (0..32).map(|i| i as f32).collect::<Vec<_>>(), "root constants did not arrive");

        let src = device.alloc("src", 16 * 4).expect("alloc src");
        let payload: Vec<f32> = (0..16).map(|i| i as f32 * 3.0 + 1.0).collect();
        let raw: Vec<u8> = payload.iter().flat_map(|v| v.to_le_bytes()).collect();
        device.upload(&src, &raw).expect("upload");
        let dst2 = device.alloc("dst2", 16 * 4).expect("alloc dst2");
        let mut p2: Params = [0; PARAM_SLOTS];
        p2[0] = 16;
        let bindings = Bindings {
            srvs: vec![(0, BoundBuffer { buffer: &src, byte_offset: 0 })],
            uavs: vec![(0, BoundBuffer { buffer: &dst2, byte_offset: 0 })],
        };
        device.dispatch(&copy_kernel, &p2, &bindings, (1, 1, 1)).expect("dispatch copy");
        let bytes = device.read(&dst2, 16 * 4).expect("read copy");
        let values: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        eprintln!("[probe] copy={values:?}");
        assert_eq!(values, payload, "SRV->UAV copy mismatch");
    }

    /// Enumerates this machine's adapters and opens the AMD/Intel one for real.
    ///
    /// `--ignored` because it needs a Windows 10/11 host with an AMD or Intel GPU.
    #[test]
    #[ignore = "needs a Windows host with an AMD or Intel D3D12 adapter"]
    fn enumerate_and_open_amd_adapter() {
        let adapters = enumerate_adapters().expect("DXGI enumeration");
        for a in &adapters {
            eprintln!(
                "[adapter] #{} {:?} vendor=0x{:04X} ({}) device=0x{:04X} uma={} coherent={} vram={}MiB shared={}MiB eligible={} ({})",
                a.index,
                a.name,
                a.vendor_id,
                a.vendor_label(),
                a.device_id,
                a.uma,
                a.cache_coherent_uma,
                a.dedicated_video_memory / 1048576,
                a.shared_system_memory / 1048576,
                a.eligible,
                a.excluded_reason.unwrap_or("selected-eligible")
            );
        }
        assert!(
            adapters.iter().any(|a| a.eligible),
            "expected an AMD or Intel D3D12 adapter"
        );
        assert!(
            adapters
                .iter()
                .filter(|a| a.vendor_id == VENDOR_NVIDIA)
                .all(|a| !a.eligible),
            "NVIDIA adapters must never be eligible"
        );
        let (device, info) = Dx12Device::open(&AdapterRequest::default()).expect("open AMD/Intel device");
        eprintln!(
            "[device] opened #{} {:?} ({}) uma={}",
            info.index,
            info.name,
            info.vendor_label(),
            info.uma
        );
        assert!(matches!(info.vendor_id, VENDOR_AMD | VENDOR_INTEL));
        assert_eq!(device.info().vendor_id, info.vendor_id);
    }
}
