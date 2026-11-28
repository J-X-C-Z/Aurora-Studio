//! Stable C ABI for mobile hosts (iOS/iPadOS, Android, HarmonyOS NEXT).
//!
//! The ABI is intentionally small and platform-neutral: callers own all pixel
//! memory, receive opaque integer handles, and never need to link Rust std
//! types. Every `extern "C"` entry point is wrapped in `catch_unwind`; a Rust
//! panic is converted into `AURORA_STATUS_PANIC` and never crosses the FFI
//! boundary.
//!
//! # Status of the bridge
//!
//! This is a designed + synthetic-tested bridge. No GPU, NPU, photo, or
//! vendor-parity claim is made here. The only executable pixel path is the
//! byte-exact CPU reference from [`crate::compute_backend`], which is also the
//! host-testable fallback for Strict-mode refusals.
//!
//! # Wiring note
//!
//! The owning agent declares `pub mod mobile_ffi;` in `crates/portrait/src/lib.rs`.
//! This file does not edit that file or `Cargo.toml`.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::autotune::{ExecutionMode, ResourceBudget};
use crate::compute_backend::{PixelOp, PixelStageFusion};

// ---------------------------------------------------------------------------
// Status / enum codes (mirrored exactly in native/include/aurora_portrait_ffi.h)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum AuroraStatus {
    Ok = 0,
    NullPointer = 1,
    InvalidHandle = 2,
    InvalidArgument = 3,
    BufferTooSmall = 4,
    StrictRejected = 5,
    Cancelled = 6,
    ModeUnsupported = 7,
    Panic = 8,
    Internal = 9,
    BudgetExhausted = 10,
}

impl AuroraStatus {
    pub const fn code(self) -> i32 {
        self as i32
    }
}

pub const AURORA_MODE_SESSION_DEFAULT: i32 = -1;
pub const AURORA_MODE_STRICT: i32 = 0;
pub const AURORA_MODE_AUTO_FALLBACK: i32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum AuroraThermalState {
    Nominal = 0,
    Fair = 1,
    Serious = 2,
    Critical = 3,
}

impl AuroraThermalState {
    pub const fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Nominal),
            1 => Some(Self::Fair),
            2 => Some(Self::Serious),
            3 => Some(Self::Critical),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::Fair => "fair",
            Self::Serious => "serious",
            Self::Critical => "critical",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum AuroraMemoryPressure {
    Normal = 0,
    Warning = 1,
    Critical = 2,
}

impl AuroraMemoryPressure {
    pub const fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Normal),
            1 => Some(Self::Warning),
            2 => Some(Self::Critical),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Warning => "warning",
            Self::Critical => "critical",
        }
    }
}

/// Pixel stage tags. Alpha is preserved by every stage, as in
/// [`crate::compute_backend::PixelOp`].
pub const AURORA_STAGE_BRIGHTNESS: u32 = 0;
pub const AURORA_STAGE_WHITEN: u32 = 1;
pub const AURORA_STAGE_ROSY: u32 = 2;
pub const AURORA_STAGE_TEMPERATURE: u32 = 3;
pub const AURORA_STAGE_SATURATION: u32 = 4;
pub const AURORA_STAGE_LUT1D: u32 = 5;
pub const AURORA_STAGE_LUT3D: u32 = 6;

pub const AURORA_ABI_VERSION: u32 = 1;
pub const AURORA_MAX_STAGES: u32 = 64;

// ---------------------------------------------------------------------------
// repr(C) data structures
// ---------------------------------------------------------------------------

/// One fused pixel-stage descriptor. `lut_ptr` borrows caller memory only for
/// the duration of the call; Rust never retains it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AuroraPixelStage {
    pub kind: u32,
    pub a: f32,
    pub b: f32,
    pub lut_ptr: *const u8,
    pub lut_len: u32,
    pub lut_size: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AuroraFusionRequest {
    pub width: u32,
    pub height: u32,
    pub stage_count: u32,
    pub stages: *const AuroraPixelStage,
    /// 0 means no cancellation token.
    pub cancel_token: u64,
    /// `AURORA_MODE_*`; use `AURORA_MODE_SESSION_DEFAULT` for the session mode.
    pub mode: i32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AuroraBudget {
    pub max_vram_bytes: u64,
    pub max_staging_bytes: u64,
    pub max_shared_mem_bytes: u32,
    pub max_threads_per_group: u32,
    pub max_concurrent_dispatches: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AuroraReceipt {
    pub fused_stages: u32,
    pub pixel_count: u64,
    pub memory_passes: u32,
    /// 1 when the bounded-staging tiled path was used.
    pub tiled_fallback: u32,
    /// Backend id; 0 is the CPU reference path.
    pub backend: u32,
    pub status_code: i32,
    pub reserved: u32,
}

// ---------------------------------------------------------------------------
// Session / token registry
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct CancelState {
    cancelled: AtomicBool,
}

impl CancelState {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

#[derive(Debug)]
struct SessionState {
    mode: ExecutionMode,
    thermal: AuroraThermalState,
    pressure: AuroraMemoryPressure,
    budget: AuroraBudget,
    last_receipt: AuroraReceipt,
}

#[derive(Debug)]
struct Session {
    base: ResourceBudget,
    state: Mutex<SessionState>,
}

struct Registry {
    next_handle: u64,
    next_token: u64,
    sessions: HashMap<u64, Arc<Session>>,
    tokens: HashMap<u64, Arc<CancelState>>,
}

impl Default for Registry {
    fn default() -> Self {
        Self {
            next_handle: 1,
            next_token: 1,
            sessions: HashMap::new(),
            tokens: HashMap::new(),
        }
    }
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Registry::default()))
}

fn get_session(handle: u64) -> Result<Arc<Session>, AuroraStatus> {
    let guard = registry().lock().map_err(|_| AuroraStatus::Internal)?;
    guard
        .sessions
        .get(&handle)
        .cloned()
        .ok_or(AuroraStatus::InvalidHandle)
}

fn get_token(token: u64) -> Result<Arc<CancelState>, AuroraStatus> {
    let guard = registry().lock().map_err(|_| AuroraStatus::Internal)?;
    guard
        .tokens
        .get(&token)
        .cloned()
        .ok_or(AuroraStatus::InvalidHandle)
}

/// Generic panic firewall: every exported function supplies its own neutral
/// value so no `Err(Box<dyn Any>)` and no unwind ever reaches the caller.
fn guard<T, F: FnOnce() -> T>(f: F, on_panic: T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(_) => on_panic,
    }
}

#[cfg(test)]
static FORCE_PANIC: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
fn boundary_probe() {
    if FORCE_PANIC.load(Ordering::SeqCst) {
        panic!("forced panic inside FFI boundary probe");
    }
}

#[cfg(not(test))]
fn boundary_probe() {}

// ---------------------------------------------------------------------------
// Budget policy
// ---------------------------------------------------------------------------

fn recompute_budget(
    base: &ResourceBudget,
    thermal: AuroraThermalState,
    pressure: AuroraMemoryPressure,
) -> AuroraBudget {
    let (vram_pct, staging_pct) = match pressure {
        AuroraMemoryPressure::Normal => (100u64, 100u64),
        AuroraMemoryPressure::Warning => (35, 25),
        AuroraMemoryPressure::Critical => (10, 5),
    };
    let (threads, dispatches, shared) = match thermal {
        AuroraThermalState::Nominal => (256u32, u32::MAX, 16 * 1024u32),
        AuroraThermalState::Fair => (128, 2, 16 * 1024),
        AuroraThermalState::Serious => (64, 1, 8 * 1024),
        AuroraThermalState::Critical => (32, 1, 4 * 1024),
    };
    AuroraBudget {
        max_vram_bytes: base.max_vram_bytes * vram_pct / 100,
        max_staging_bytes: base.max_staging_bytes * staging_pct / 100,
        max_shared_mem_bytes: base.max_shared_mem_bytes.min(shared),
        max_threads_per_group: base.max_threads_per_group.min(threads),
        max_concurrent_dispatches: base.max_concurrent_dispatches.min(dispatches.max(1)),
        reserved: 0,
    }
}

fn parse_mode(value: i32) -> Option<ExecutionMode> {
    match value {
        AURORA_MODE_STRICT => Some(ExecutionMode::Strict),
        AURORA_MODE_AUTO_FALLBACK => Some(ExecutionMode::AutoFallback),
        _ => None,
    }
}

fn mode_str(mode: ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Strict => "strict",
        ExecutionMode::AutoFallback => "auto_fallback",
    }
}

fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn capability_json(state: &SessionState) -> String {
    let b = state.budget;
    format!(
        concat!(
            "{{\"abi\":\"aurora-portrait-ffi/1\",\"version\":{},",
            "\"mode\":\"{}\",\"thermal\":\"{}\",\"memory_pressure\":\"{}\",",
            "\"budget\":{{\"max_vram_bytes\":{},\"max_staging_bytes\":{},",
            "\"max_shared_mem_bytes\":{},\"max_threads_per_group\":{},",
            "\"max_concurrent_dispatches\":{}}},",
            "\"features\":[\"rgba8_fusion\",\"cancel_token\",\"strict_mode\",",
            "\"thermal_budget\",\"memory_pressure_budget\"],",
            "\"pixel_status\":\"DESIGNED\",\"name\":\"{}\"}}"
        ),
        AURORA_ABI_VERSION,
        mode_str(state.mode),
        state.thermal.as_str(),
        state.pressure.as_str(),
        b.max_vram_bytes,
        b.max_staging_bytes,
        b.max_shared_mem_bytes,
        b.max_threads_per_group,
        b.max_concurrent_dispatches,
        json_escape("Aurora Portrait mobile bridge"),
    )
}

// ---------------------------------------------------------------------------
// Stage decoding and pixel execution
// ---------------------------------------------------------------------------

fn decode_stage(raw: &AuroraPixelStage) -> Result<PixelOp, AuroraStatus> {
    let invalid = Err(AuroraStatus::InvalidArgument);
    match raw.kind {
        AURORA_STAGE_BRIGHTNESS => Ok(PixelOp::Brightness { delta: raw.a }),
        AURORA_STAGE_WHITEN => Ok(PixelOp::Whiten { strength: raw.a }),
        AURORA_STAGE_ROSY => Ok(PixelOp::Rosy { strength: raw.a }),
        AURORA_STAGE_TEMPERATURE => Ok(PixelOp::Temperature {
            warmth: raw.a,
            tint: raw.b,
        }),
        AURORA_STAGE_SATURATION => Ok(PixelOp::Saturation { amount: raw.a }),
        AURORA_STAGE_LUT1D => {
            if raw.lut_ptr.is_null() || raw.lut_len < 768 {
                return invalid;
            }
            let bytes = unsafe { std::slice::from_raw_parts(raw.lut_ptr, 768) };
            let mut r = Box::new([0u8; 256]);
            let mut g = Box::new([0u8; 256]);
            let mut b = Box::new([0u8; 256]);
            r.copy_from_slice(&bytes[0..256]);
            g.copy_from_slice(&bytes[256..512]);
            b.copy_from_slice(&bytes[512..768]);
            Ok(PixelOp::Lut1D { r, g, b })
        }
        AURORA_STAGE_LUT3D => {
            let size = raw.lut_size as usize;
            if size < 2 || raw.lut_ptr.is_null() {
                return invalid;
            }
            let cells = size
                .checked_mul(size)
                .and_then(|v| v.checked_mul(size))
                .ok_or(AuroraStatus::InvalidArgument)?;
            let byte_len = cells
                .checked_mul(12)
                .ok_or(AuroraStatus::InvalidArgument)?;
            if raw.lut_len as usize != byte_len {
                return invalid;
            }
            let bytes = unsafe { std::slice::from_raw_parts(raw.lut_ptr, byte_len) };
            let mut table = Vec::with_capacity(cells);
            for chunk in bytes.chunks_exact(12) {
                let f = |at: usize| -> f32 {
                    f32::from_le_bytes([chunk[at], chunk[at + 1], chunk[at + 2], chunk[at + 3]])
                };
                table.push([f(0), f(4), f(8)]);
            }
            Ok(PixelOp::Lut3D { size, table })
        }
        _ => invalid,
    }
}

const CANCEL_CHECK_PIXELS: usize = 4096;

/// Byte-exact single-pass traversal. Cancellation is sampled at a bounded
/// cadence so a large frame can be stopped without waiting for completion.
fn run_fused(
    fusion: &PixelStageFusion,
    src: &[u8],
    dst: &mut [u8],
    cancel: Option<&CancelState>,
) -> Result<(), AuroraStatus> {
    if src.len() != dst.len() || src.len() % 4 != 0 {
        return Err(AuroraStatus::InvalidArgument);
    }
    let mut until_check = 0usize;
    for (in_px, out_px) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        if until_check == 0 {
            if cancel.is_some_and(CancelState::is_cancelled) {
                return Err(AuroraStatus::Cancelled);
            }
            until_check = CANCEL_CHECK_PIXELS;
        }
        until_check -= 1;
        let mut px = [in_px[0], in_px[1], in_px[2], in_px[3]];
        for stage in fusion.stages() {
            px = stage.apply_pixel(px);
        }
        out_px.copy_from_slice(&px);
    }
    Ok(())
}

fn execute_rgba8(
    req: &AuroraFusionRequest,
    fusion: &PixelStageFusion,
    src: &[u8],
    dst: &mut [u8],
    cancel: Option<&CancelState>,
    budget: &AuroraBudget,
    mode: ExecutionMode,
    tiled: &mut bool,
) -> Result<(), AuroraStatus> {
    let full_staging = src
        .len()
        .checked_mul(2)
        .ok_or(AuroraStatus::BudgetExhausted)? as u64;
    if full_staging <= budget.max_staging_bytes {
        return run_fused(fusion, src, dst, cancel);
    }

    // Strict mode must fail explicitly rather than silently degrading.
    if mode == ExecutionMode::Strict {
        return Err(AuroraStatus::StrictRejected);
    }

    // AutoFallback: bounded-staging tiled traversal. Source and destination
    // remain caller-owned; only one row pair is treated as working staging.
    *tiled = true;
    let row_bytes = req.width as usize * 4;
    let row_pair = row_bytes
        .checked_mul(2)
        .ok_or(AuroraStatus::BudgetExhausted)?;
    let max_rows = (budget.max_staging_bytes as usize / row_pair).max(1);
    let height = req.height as usize;
    let mut y = 0usize;
    while y < height {
        let rows = max_rows.min(height - y);
        let offset = y * row_bytes;
        let len = rows * row_bytes;
        run_fused(
            fusion,
            &src[offset..offset + len],
            &mut dst[offset..offset + len],
            cancel,
        )?;
        y += rows;
    }
    Ok(())
}

fn store_receipt(session: &Session, receipt: AuroraReceipt) {
    if let Ok(mut state) = session.state.lock() {
        state.last_receipt = receipt;
    }
}

// ---------------------------------------------------------------------------
// Exported C ABI
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn aurora_session_create(mode: c_int, out_handle: *mut u64) -> i32 {
    guard(
        || {
            if out_handle.is_null() {
                return AuroraStatus::NullPointer.code();
            }
            let Some(mode) = parse_mode(mode) else {
                return AuroraStatus::InvalidArgument.code();
            };
            let base = ResourceBudget::conservative_mobile();
            let initial = recompute_budget(
                &base,
                AuroraThermalState::Nominal,
                AuroraMemoryPressure::Normal,
            );
            let session = Arc::new(Session {
                base,
                state: Mutex::new(SessionState {
                    mode,
                    thermal: AuroraThermalState::Nominal,
                    pressure: AuroraMemoryPressure::Normal,
                    budget: initial,
                    last_receipt: AuroraReceipt::default(),
                }),
            });
            let mut reg = match registry().lock() {
                Ok(guard) => guard,
                Err(_) => return AuroraStatus::Internal.code(),
            };
            let handle = reg.next_handle;
            reg.next_handle = reg.next_handle.wrapping_add(1);
            reg.sessions.insert(handle, session);
            unsafe { *out_handle = handle };
            AuroraStatus::Ok.code()
        },
        AuroraStatus::Panic.code(),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn aurora_session_destroy(handle: u64) -> i32 {
    guard(
        || {
            let mut reg = match registry().lock() {
                Ok(guard) => guard,
                Err(_) => return AuroraStatus::Internal.code(),
            };
            if reg.sessions.remove(&handle).is_some() {
                AuroraStatus::Ok.code()
            } else {
                AuroraStatus::InvalidHandle.code()
            }
        },
        AuroraStatus::Panic.code(),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn aurora_session_set_mode(handle: u64, mode: c_int) -> i32 {
    guard(
        || {
            let Some(mode) = parse_mode(mode) else {
                return AuroraStatus::InvalidArgument.code();
            };
            let session = match get_session(handle) {
                Ok(s) => s,
                Err(code) => return code.code(),
            };
            match session.state.lock() {
                Ok(mut state) => {
                    state.mode = mode;
                    AuroraStatus::Ok.code()
                }
                Err(_) => AuroraStatus::Internal.code(),
            }
        },
        AuroraStatus::Panic.code(),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn aurora_cancel_token_create(out_token: *mut u64) -> i32 {
    guard(
        || {
            if out_token.is_null() {
                return AuroraStatus::NullPointer.code();
            }
            let token = Arc::new(CancelState {
                cancelled: AtomicBool::new(false),
            });
            let mut reg = match registry().lock() {
                Ok(guard) => guard,
                Err(_) => return AuroraStatus::Internal.code(),
            };
            let id = reg.next_token;
            reg.next_token = reg.next_token.wrapping_add(1);
            reg.tokens.insert(id, token);
            unsafe { *out_token = id };
            AuroraStatus::Ok.code()
        },
        AuroraStatus::Panic.code(),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn aurora_cancel_token_cancel(token: u64) -> i32 {
    guard(
        || match get_token(token) {
            Ok(state) => {
                state.cancelled.store(true, Ordering::SeqCst);
                AuroraStatus::Ok.code()
            }
            Err(code) => code.code(),
        },
        AuroraStatus::Panic.code(),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn aurora_cancel_token_is_cancelled(token: u64, out_flag: *mut u32) -> i32 {
    guard(
        || {
            if out_flag.is_null() {
                return AuroraStatus::NullPointer.code();
            }
            match get_token(token) {
                Ok(state) => {
                    unsafe { *out_flag = u32::from(state.is_cancelled()) };
                    AuroraStatus::Ok.code()
                }
                Err(code) => code.code(),
            }
        },
        AuroraStatus::Panic.code(),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn aurora_cancel_token_destroy(token: u64) -> i32 {
    guard(
        || {
            let mut reg = match registry().lock() {
                Ok(guard) => guard,
                Err(_) => return AuroraStatus::Internal.code(),
            };
            if reg.tokens.remove(&token).is_some() {
                AuroraStatus::Ok.code()
            } else {
                AuroraStatus::InvalidHandle.code()
            }
        },
        AuroraStatus::Panic.code(),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn aurora_notify_thermal_state(handle: u64, state: c_int) -> i32 {
    guard(
        || {
            let Some(thermal) = AuroraThermalState::from_i32(state) else {
                return AuroraStatus::InvalidArgument.code();
            };
            let session = match get_session(handle) {
                Ok(s) => s,
                Err(code) => return code.code(),
            };
            match session.state.lock() {
                Ok(mut inner) => {
                    inner.thermal = thermal;
                    inner.budget = recompute_budget(&session.base, thermal, inner.pressure);
                    AuroraStatus::Ok.code()
                }
                Err(_) => AuroraStatus::Internal.code(),
            }
        },
        AuroraStatus::Panic.code(),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn aurora_notify_memory_pressure(handle: u64, pressure: c_int) -> i32 {
    guard(
        || {
            let Some(pressure) = AuroraMemoryPressure::from_i32(pressure) else {
                return AuroraStatus::InvalidArgument.code();
            };
            let session = match get_session(handle) {
                Ok(s) => s,
                Err(code) => return code.code(),
            };
            match session.state.lock() {
                Ok(mut inner) => {
                    inner.pressure = pressure;
                    inner.budget = recompute_budget(&session.base, inner.thermal, pressure);
                    AuroraStatus::Ok.code()
                }
                Err(_) => AuroraStatus::Internal.code(),
            }
        },
        AuroraStatus::Panic.code(),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn aurora_budget_snapshot(handle: u64, out_budget: *mut AuroraBudget) -> i32 {
    guard(
        || {
            let session = match get_session(handle) {
                Ok(s) => s,
                Err(code) => return code.code(),
            };
            if out_budget.is_null() {
                return AuroraStatus::NullPointer.code();
            }
            match session.state.lock() {
                Ok(inner) => {
                    unsafe { *out_budget = inner.budget };
                    AuroraStatus::Ok.code()
                }
                Err(_) => AuroraStatus::Internal.code(),
            }
        },
        AuroraStatus::Panic.code(),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn aurora_last_receipt(handle: u64, out_receipt: *mut AuroraReceipt) -> i32 {
    guard(
        || {
            if out_receipt.is_null() {
                return AuroraStatus::NullPointer.code();
            }
            let session = match get_session(handle) {
                Ok(s) => s,
                Err(code) => return code.code(),
            };
            match session.state.lock() {
                Ok(inner) => {
                    unsafe { *out_receipt = inner.last_receipt };
                    AuroraStatus::Ok.code()
                }
                Err(_) => AuroraStatus::Internal.code(),
            }
        },
        AuroraStatus::Panic.code(),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn aurora_capabilities_json(
    handle: u64,
    out_json: *mut c_char,
    capacity: usize,
    out_len: *mut usize,
) -> i32 {
    guard(
        || {
            let session = match get_session(handle) {
                Ok(s) => s,
                Err(code) => return code.code(),
            };
            let json = match session.state.lock() {
                Ok(inner) => capability_json(&inner),
                Err(_) => return AuroraStatus::Internal.code(),
            };
            let needed = json.len().saturating_add(1);
            if !out_len.is_null() {
                unsafe { *out_len = needed };
            }
            if out_json.is_null() || capacity < needed {
                return AuroraStatus::BufferTooSmall.code();
            }
            unsafe {
                ptr::copy_nonoverlapping(json.as_ptr() as *const c_char, out_json, json.len());
                *out_json.add(json.len()) = 0;
            }
            AuroraStatus::Ok.code()
        },
        AuroraStatus::Panic.code(),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn aurora_process_rgba8(
    handle: u64,
    request: *const AuroraFusionRequest,
    src_rgba8: *const u8,
    dst_rgba8: *mut u8,
) -> i32 {
    guard(
        || {
            boundary_probe();

            if request.is_null() || src_rgba8.is_null() || dst_rgba8.is_null() {
                return AuroraStatus::NullPointer.code();
            }
            let req = unsafe { *request };
            if req.width == 0 || req.height == 0 || req.stage_count > AURORA_MAX_STAGES {
                return AuroraStatus::InvalidArgument.code();
            }
            if req.stage_count > 0 && req.stages.is_null() {
                return AuroraStatus::NullPointer.code();
            }
            let pixel_count = match (req.width as usize).checked_mul(req.height as usize) {
                Some(v) => v,
                None => return AuroraStatus::InvalidArgument.code(),
            };
            let byte_len = match pixel_count.checked_mul(4) {
                Some(v) => v,
                None => return AuroraStatus::InvalidArgument.code(),
            };

            let session = match get_session(handle) {
                Ok(s) => s,
                Err(code) => return code.code(),
            };
            let (mode, budget) = match session.state.lock() {
                Ok(inner) => {
                    let mode = if req.mode == AURORA_MODE_SESSION_DEFAULT {
                        inner.mode
                    } else {
                        match parse_mode(req.mode) {
                            Some(m) => m,
                            None => return AuroraStatus::ModeUnsupported.code(),
                        }
                    };
                    (mode, inner.budget)
                }
                Err(_) => return AuroraStatus::Internal.code(),
            };

            let cancel = if req.cancel_token == 0 {
                None
            } else {
                match get_token(req.cancel_token) {
                    Ok(token) => Some(token),
                    Err(code) => return code.code(),
                }
            };

            let mut ops = Vec::with_capacity(req.stage_count as usize);
            for index in 0..req.stage_count as usize {
                let raw = unsafe { *req.stages.add(index) };
                match decode_stage(&raw) {
                    Ok(op) => ops.push(op),
                    Err(code) => return code.code(),
                }
            }
            let fusion = match PixelStageFusion::from_stages(ops) {
                Ok(fusion) => fusion,
                Err(_) => return AuroraStatus::InvalidArgument.code(),
            };

            let src = unsafe { std::slice::from_raw_parts(src_rgba8, byte_len) };
            let dst = unsafe { std::slice::from_raw_parts_mut(dst_rgba8, byte_len) };

            let mut tiled = false;
            let outcome = execute_rgba8(
                &req,
                &fusion,
                src,
                dst,
                cancel.as_deref(),
                &budget,
                mode,
                &mut tiled,
            );
            let mut receipt = AuroraReceipt {
                fused_stages: fusion.len() as u32,
                pixel_count: pixel_count as u64,
                memory_passes: 1,
                tiled_fallback: u32::from(tiled),
                backend: 0,
                status_code: AuroraStatus::Ok.code(),
                reserved: 0,
            };
            match outcome {
                Ok(()) => {
                    store_receipt(&session, receipt);
                    AuroraStatus::Ok.code()
                }
                Err(code) => {
                    receipt.status_code = code.code();
                    store_receipt(&session, receipt);
                    code.code()
                }
            }
        },
        AuroraStatus::Panic.code(),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn aurora_status_name(code: c_int) -> *const c_char {
    guard(
        || {
            let name: &'static [u8] = match code {
                c if c == AuroraStatus::Ok.code() => b"OK\0",
                c if c == AuroraStatus::NullPointer.code() => b"NULL_POINTER\0",
                c if c == AuroraStatus::InvalidHandle.code() => b"INVALID_HANDLE\0",
                c if c == AuroraStatus::InvalidArgument.code() => b"INVALID_ARGUMENT\0",
                c if c == AuroraStatus::BufferTooSmall.code() => b"BUFFER_TOO_SMALL\0",
                c if c == AuroraStatus::StrictRejected.code() => b"STRICT_REJECTED\0",
                c if c == AuroraStatus::Cancelled.code() => b"CANCELLED\0",
                c if c == AuroraStatus::ModeUnsupported.code() => b"MODE_UNSUPPORTED\0",
                c if c == AuroraStatus::Panic.code() => b"PANIC\0",
                c if c == AuroraStatus::Internal.code() => b"INTERNAL\0",
                c if c == AuroraStatus::BudgetExhausted.code() => b"BUDGET_EXHAUSTED\0",
                _ => b"UNKNOWN\0",
            };
            name.as_ptr() as *const c_char
        },
        b"UNKNOWN\0".as_ptr() as *const c_char,
    )
}

// ---------------------------------------------------------------------------
// Host verification (synthetic only; no real device is probed)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn brightness(delta: f32) -> AuroraPixelStage {
        AuroraPixelStage {
            kind: AURORA_STAGE_BRIGHTNESS,
            a: delta,
            b: 0.0,
            lut_ptr: ptr::null(),
            lut_len: 0,
            lut_size: 0,
            reserved: 0,
        }
    }

    fn request(width: u32, height: u32, stage: *const AuroraPixelStage) -> AuroraFusionRequest {
        AuroraFusionRequest {
            width,
            height,
            stage_count: 1,
            stages: stage,
            cancel_token: 0,
            mode: AURORA_MODE_SESSION_DEFAULT,
            reserved: 0,
        }
    }

    fn budget(handle: u64) -> AuroraBudget {
        let mut out = AuroraBudget::default();
        assert_eq!(
            aurora_budget_snapshot(handle, &mut out),
            AuroraStatus::Ok.code()
        );
        out
    }

    #[test]
    fn create_destroy_and_invalid_handles() {
        let mut handle = 0u64;
        assert_eq!(
            aurora_session_create(AURORA_MODE_STRICT, ptr::null_mut()),
            AuroraStatus::NullPointer.code()
        );
        assert_eq!(
            aurora_session_create(99, &mut handle),
            AuroraStatus::InvalidArgument.code()
        );
        assert_eq!(
            aurora_session_create(AURORA_MODE_STRICT, &mut handle),
            AuroraStatus::Ok.code()
        );
        assert_ne!(handle, 0);
        assert_eq!(aurora_session_destroy(handle), AuroraStatus::Ok.code());
        assert_eq!(
            aurora_session_destroy(handle),
            AuroraStatus::InvalidHandle.code()
        );
        assert_eq!(
            aurora_budget_snapshot(handle, ptr::null_mut()),
            AuroraStatus::InvalidHandle.code()
        );
        assert_eq!(
            aurora_session_destroy(u64::MAX),
            AuroraStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn strict_mode_rejects_while_auto_fallback_uses_bounded_tiling() {
        let mut strict = 0u64;
        let mut auto = 0u64;
        assert_eq!(
            aurora_session_create(AURORA_MODE_STRICT, &mut strict),
            AuroraStatus::Ok.code()
        );
        assert_eq!(
            aurora_session_create(AURORA_MODE_AUTO_FALLBACK, &mut auto),
            AuroraStatus::Ok.code()
        );

        let nominal = budget(strict);
        assert_eq!(
            aurora_notify_memory_pressure(strict, AuroraMemoryPressure::Critical as i32),
            AuroraStatus::Ok.code()
        );
        let tight = budget(strict);
        assert!(tight.max_staging_bytes < nominal.max_staging_bytes);
        assert_eq!(
            aurora_notify_thermal_state(strict, AuroraThermalState::Critical as i32),
            AuroraStatus::Ok.code()
        );
        let cooled = budget(strict);
        assert!(cooled.max_threads_per_group < nominal.max_threads_per_group);
        assert_eq!(
            aurora_notify_memory_pressure(strict, AuroraMemoryPressure::Normal as i32),
            AuroraStatus::Ok.code()
        );
        assert_eq!(budget(strict).max_staging_bytes, nominal.max_staging_bytes);

        // 1024x1024 RGBA8 needs 8 MiB of full-frame staging; the critical
        // pressure cap is 5% of the 64 MiB mobile budget (3.2 MiB).
        assert_eq!(
            aurora_notify_memory_pressure(strict, AuroraMemoryPressure::Critical as i32),
            AuroraStatus::Ok.code()
        );
        assert_eq!(
            aurora_notify_memory_pressure(auto, AuroraMemoryPressure::Critical as i32),
            AuroraStatus::Ok.code()
        );

        let stage = brightness(0.05);
        let req = request(1024, 1024, &stage);
        let src = vec![64u8; 1024 * 1024 * 4];
        let mut strict_dst = vec![0u8; src.len()];
        let mut auto_dst = vec![0u8; src.len()];

        assert_eq!(
            aurora_process_rgba8(strict, &req, src.as_ptr(), strict_dst.as_mut_ptr()),
            AuroraStatus::StrictRejected.code()
        );
        assert_eq!(
            aurora_process_rgba8(auto, &req, src.as_ptr(), auto_dst.as_mut_ptr()),
            AuroraStatus::Ok.code()
        );

        let mut receipt = AuroraReceipt::default();
        assert_eq!(
            aurora_last_receipt(auto, &mut receipt),
            AuroraStatus::Ok.code()
        );
        assert_eq!(receipt.tiled_fallback, 1);
        assert_eq!(receipt.pixel_count, 1024 * 1024);
        assert_eq!(receipt.fused_stages, 1);
        assert_eq!(receipt.memory_passes, 1);

        let mut strict_receipt = AuroraReceipt::default();
        assert_eq!(
            aurora_last_receipt(strict, &mut strict_receipt),
            AuroraStatus::Ok.code()
        );
        assert_eq!(strict_receipt.status_code, AuroraStatus::StrictRejected.code());

        aurora_session_destroy(strict);
        aurora_session_destroy(auto);
    }

    #[test]
    fn cancel_token_stops_dispatch_and_reports_state() {
        let mut session = 0u64;
        let mut token = 0u64;
        assert_eq!(
            aurora_session_create(AURORA_MODE_AUTO_FALLBACK, &mut session),
            AuroraStatus::Ok.code()
        );
        assert_eq!(aurora_cancel_token_create(&mut token), AuroraStatus::Ok.code());
        assert_eq!(aurora_cancel_token_cancel(token), AuroraStatus::Ok.code());
        assert_eq!(aurora_cancel_token_cancel(token), AuroraStatus::Ok.code());
        let mut flag = 0u32;
        assert_eq!(
            aurora_cancel_token_is_cancelled(token, &mut flag),
            AuroraStatus::Ok.code()
        );
        assert_eq!(flag, 1);
        assert_eq!(
            aurora_cancel_token_is_cancelled(u64::MAX, &mut flag),
            AuroraStatus::InvalidHandle.code()
        );

        let stage = brightness(0.1);
        let mut req = request(16, 16, &stage);
        req.cancel_token = token;
        let src = vec![10u8; 16 * 16 * 4];
        let mut dst = vec![0u8; src.len()];
        assert_eq!(
            aurora_process_rgba8(session, &req, src.as_ptr(), dst.as_mut_ptr()),
            AuroraStatus::Cancelled.code()
        );
        assert!(dst.iter().all(|&v| v == 0), "cancelled dispatch must not write pixels");

        req.cancel_token = u64::MAX;
        assert_eq!(
            aurora_process_rgba8(session, &req, src.as_ptr(), dst.as_mut_ptr()),
            AuroraStatus::InvalidHandle.code()
        );

        assert_eq!(aurora_cancel_token_destroy(token), AuroraStatus::Ok.code());
        assert_eq!(
            aurora_cancel_token_destroy(token),
            AuroraStatus::InvalidHandle.code()
        );
        aurora_session_destroy(session);
    }

    #[test]
    fn concurrent_calls_share_handles_safely() {
        let mut session = 0u64;
        assert_eq!(
            aurora_session_create(AURORA_MODE_AUTO_FALLBACK, &mut session),
            AuroraStatus::Ok.code()
        );

        let threads: Vec<_> = (0..8u8)
            .map(|seed| {
                let handle = session;
                thread::spawn(move || {
                    let stage = brightness(seed as f32 / 100.0);
                    let req = request(32, 32, &stage);
                    let src = vec![seed; 32 * 32 * 4];
                    let mut dst = vec![0u8; src.len()];
                    let code = aurora_process_rgba8(handle, &req, src.as_ptr(), dst.as_mut_ptr());
                    assert_eq!(code, AuroraStatus::Ok.code());
                    let op = PixelOp::Brightness {
                        delta: seed as f32 / 100.0,
                    };
                    let expected = op.apply_pixel([seed, seed, seed, seed]);
                    assert_eq!(&dst[0..4], &expected);
                })
            })
            .collect();
        for worker in threads {
            worker.join().expect("worker thread must not panic");
        }
        aurora_session_destroy(session);
    }

    #[test]
    fn panic_never_unwinds_across_the_abi() {
        let mut session = 0u64;
        assert_eq!(
            aurora_session_create(AURORA_MODE_STRICT, &mut session),
            AuroraStatus::Ok.code()
        );
        let stage = brightness(0.0);
        let req = request(4, 4, &stage);
        let src = vec![0u8; 64];
        let mut dst = vec![0u8; 64];

        FORCE_PANIC.store(true, Ordering::SeqCst);
        let code = aurora_process_rgba8(session, &req, src.as_ptr(), dst.as_mut_ptr());
        FORCE_PANIC.store(false, Ordering::SeqCst);
        assert_eq!(code, AuroraStatus::Panic.code());
        // The session registry remains usable after a contained panic.
        assert_eq!(
            aurora_process_rgba8(session, &req, src.as_ptr(), dst.as_mut_ptr()),
            AuroraStatus::Ok.code()
        );
        aurora_session_destroy(session);
    }

    #[test]
    fn capability_json_reports_budget_and_states() {
        let mut session = 0u64;
        assert_eq!(
            aurora_session_create(AURORA_MODE_STRICT, &mut session),
            AuroraStatus::Ok.code()
        );
        let mut len = 0usize;
        assert_eq!(
            aurora_capabilities_json(session, ptr::null_mut(), 0, &mut len),
            AuroraStatus::BufferTooSmall.code()
        );
        assert!(len > 64);
        let mut buffer = vec![0u8; len];
        assert_eq!(
            aurora_capabilities_json(
                session,
                buffer.as_mut_ptr() as *mut c_char,
                buffer.len(),
                &mut len
            ),
            AuroraStatus::Ok.code()
        );
        let text = std::ffi::CStr::from_bytes_with_nul(&buffer)
            .expect("NUL-terminated capability JSON")
            .to_str()
            .expect("ASCII JSON")
            .to_string();
        assert!(text.contains("\"abi\":\"aurora-portrait-ffi/1\""));
        assert!(text.contains("\"mode\":\"strict\""));
        assert!(text.contains("\"thermal\":\"nominal\""));
        assert!(text.contains("\"max_staging_bytes\":"));
        assert!(text.contains("\"pixel_status\":\"DESIGNED\""));

        assert_eq!(
            aurora_capabilities_json(u64::MAX, ptr::null_mut(), 0, &mut len),
            AuroraStatus::InvalidHandle.code()
        );
        aurora_session_destroy(session);
    }

    #[test]
    fn lut_stage_decoding_matches_reference_pipeline() {
        let mut session = 0u64;
        assert_eq!(
            aurora_session_create(AURORA_MODE_STRICT, &mut session),
            AuroraStatus::Ok.code()
        );
        let mut lut = vec![0u8; 768];
        for (index, byte) in lut.iter_mut().enumerate() {
            *byte = (index % 256) as u8;
        }
        let stage = AuroraPixelStage {
            kind: AURORA_STAGE_LUT1D,
            a: 0.0,
            b: 0.0,
            lut_ptr: lut.as_ptr(),
            lut_len: lut.len() as u32,
            lut_size: 0,
            reserved: 0,
        };
        let req = request(4, 4, &stage);
        let src = vec![17u8; 64];
        let mut dst = vec![0u8; 64];
        assert_eq!(
            aurora_process_rgba8(session, &req, src.as_ptr(), dst.as_mut_ptr()),
            AuroraStatus::Ok.code()
        );
        let mut g = Box::new([0u8; 256]);
        g.copy_from_slice(&lut[256..512]);
        let fusion = PixelStageFusion::from_stages(vec![PixelOp::Lut1D {
            r: Box::new({
                let mut r = [0u8; 256];
                r.copy_from_slice(&lut[0..256]);
                r
            }),
            g,
            b: Box::new({
                let mut b = [0u8; 256];
                b.copy_from_slice(&lut[512..768]);
                b
            }),
        }])
        .unwrap();
        let expected = fusion.execute_fused(&crate::Image::new(4, 4, src.clone()).unwrap()).unwrap();
        assert_eq!(dst, expected.pixels);

        // Truncated LUT must be rejected, not read out of bounds.
        let mut bad = stage;
        bad.lut_len = 767;
        let req = request(4, 4, &bad);
        assert_eq!(
            aurora_process_rgba8(session, &req, src.as_ptr(), dst.as_mut_ptr()),
            AuroraStatus::InvalidArgument.code()
        );
        aurora_session_destroy(session);
    }
}
