//! Per-photo portrait capability evidence for [`crate::FileEngine`].
//!
//! [`crate::panel_commands::portrait_capabilities`] is a *static* catalogue
//! bridge: it says which `pc:<pf>` rows the dispatcher has a route for. It can
//! never say that this machine, at this moment, with this photo, can really run
//! a row. This module answers that second question and keeps the three axes of
//! [`crate::portrait_registry`] strictly apart:
//!
//! * static support comes from the registry (route + compiled kernels),
//! * historical evidence stays historical and never unlocks anything,
//! * runtime availability is derived from *observed* facts only:
//!   - the inference runtime and the canonical model files resolve at all,
//!   - the photo's own decoded frame was fed to the same shared face-input
//!     producer and skin-mask segmenter the renderer uses, and what came back,
//!   - per-row assets (2D makeup materials, the PF8033 smile artifacts) exist.
//!
//! Nothing here fabricates readiness. A row only reports
//! [`RuntimeAvailability::Available`] when every model, asset and same-frame
//! input its registry descriptor declares has been proven for *this* image. Any
//! other outcome names the missing axis: `MissingModel`, `MissingAsset`,
//! `MissingInput`, `BackendUnavailable` or `StoredOnly`.
//!
//! ## What is deliberately not done
//!
//! * No effect kernel is run. The probe feeds the frame to the two shared
//!   producers every portrait render starts with anyway (face input, skin
//!   mask) — that materialises the same-frame inputs the panel is asking
//!   about, and its result is cached by those producers for the next render.
//! * No decoding. A capability query never turns into an image load: when the
//!   photo is not in the decoded-source cache there is no frame evidence and
//!   the rows stay `MissingInput`.
//! * No cross-image reuse. Evidence is keyed by
//!   [`FrameIdentity`] (path + length + mtime + the cache's content token), so
//!   a second image never inherits the first one's result and a replaced file
//!   invalidates its own entry.
//! * No marker files, no temp directories, no environment switches standing in
//!   for a detection.
//!
//! ## Known limitations
//!
//! * `TeethMask` and `EyeMask` are only produced inside the render path
//!   (`attach_pf108_inputs`, the PF101/PF100004 producers). Until a row that
//!   consumes them has actually run, this probe reports `MissingInput` for
//!   them; `C1FaceParsing::teeth_mask` still refuses with a named error, so the
//!   parsing bundle alone is not evidence.
//! * `PF8`/`PF9` are declared as 2D makeup rows. Nothing refuses them on
//!   principle any more: their owner mask and UVs are assembled by the row's own
//!   producer in [`crate::portrait_probe_inputs::produce_row`], and until that ran
//!   for this frame they are `PendingSameFrame`, not "never".
//! * A row's own verdict decides the row: `Proven` is the only thing that can
//!   make it available, a produced-and-missing row says which input is missing,
//!   and a cheap answer stays `PendingSameFrame`.
//! * The first call per frame may open the ONNX sessions and run the shared
//!   producers; callers must be on a background thread (the interface requests
//!   the snapshot off the UI thread).
//!
//! ## Dependencies this probe can prove
//!
//! | Dependency | Evidence |
//! |---|---|
//! | `Landmarks194` | the shared face-input producer returned a finite 194-point set for this frame |
//! | `FaceParsingV2` | the same bundle carried a readable canonical parsing class plane |
//! | `HumanParsingMask` | `skinMasking.onnx` is on disk and its planes carry a target on this frame |
//! | `SkinMask` / `HairMask` | this frame's own `PortraitMasks` planes carry a target |
//! | `MakeupBundle` | 2D rows: the frame's 194 landmarks exist and the row needs no material PNG; material rows: `makeup_presets_for_pf` found the PNG |
//! | `SmileWarpAsset` | `Pf8033CpuForwardReceipt::discover()` accepts the installed artifacts |
//!
//! Every other dependency the registry can declare — `TeethMask`, `EyeMask`,
//! `BodyPose`, the vendor repair / eye-bag / eye-white / breast / teeth-repair /
//! glare / double-chin / generic-skin models, and the
//! `MoistureCurve`/`FaceStereo`/`LipsLut` asset sets — has no availability
//! accessor in this engine yet. A row that declares one therefore reports
//! `MissingModel` / `MissingAsset` / `MissingInput` instead of optimism. Adding
//! one means adding a single evidence entry next to [`probe_host`] /
//! [`probe_frame`]; it needs a cheap presence check for that dependency, not a
//! capability table.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use aurora_portrait::Refusal;

use crate::portrait_probe_inputs::{
    PfInputVerdict, TEETH_FAMILY_PFS, probed_pfs, produce_row, prove_pf_inputs_with_face_mask,
};
use crate::portrait_registry::{
    InputDependency, PortraitCapabilitySnapshot, RuntimeAvailability, RuntimeProbeContext,
    capability_snapshot, descriptor_by_pf, portrait_input_key,
};
use crate::{AdjustmentSet, ImageId, PortraitImage as Image, Source};

/// How long host-level facts (runtime, model files, assets) are trusted before
/// they are re-read. Installing models or makeup assets while the app runs
/// shows up within this window; a capability query never blocks on a re-read
/// more than once per window.
pub(crate) const HOST_FACTS_TTL: Duration = Duration::from_secs(30);

/// A mask plane whose peak stays below this carries no target. The planes are
/// softmax probabilities, so 0.02 is well above numeric noise and far below any
/// usable skin/hair region.
const MASK_SIGNAL: f32 = 0.02;

/// Ticket range for the probe's own face-input calls. The value only has to be
/// non-zero and distinct from render tickets; the producers re-stamp a cached
/// bundle for whichever job asks next.
const PROBE_TICKET_BASE: u64 = 0x50ff_0000_0000_0000;

/// Capture One's human-parsing segmenter; the same model `people_masks` runs.
const HUMAN_PARSING_MODEL: &str = "skinMasking.onnx";

fn next_ticket() -> u64 {
    static CLOCK: AtomicU64 = AtomicU64::new(0);
    PROBE_TICKET_BASE + CLOCK.fetch_add(1, Ordering::Relaxed)
}

/// Identity of the decoded frame the per-photo evidence belongs to.
///
/// This mirrors the decoded-source cache's own identity (path plus the length,
/// mtime and content token it verifies on every hit), so two images can never
/// share evidence and a file replaced on disk invalidates its own entry.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FrameIdentity {
    pub path: PathBuf,
    pub len: u64,
    pub mtime_nanos: u64,
    /// Content token taken by the decoded-source cache, when one was recorded.
    pub token: Option<u64>,
}

/// Which inference route serves model-backed rows on this host.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum InferenceKind {
    /// No model runtime resolved at all.
    #[default]
    None,
    /// The compiled native CUDA lane, with a real device behind it.
    NativeCuda,
    /// Native library with a per-model device decision. Not an execution receipt.
    NativePerModel,
    /// A real ONNX Runtime library. Never reported as native, even when it is
    /// used as a compatibility path next to a native build.
    OrtCompat,
}

/// What this machine can supply, independent of any photo.
///
/// Everything here is read from already-existing entry points (runtime
/// discovery, canonical model paths, the makeup/smile asset roots); nothing is
/// inferred from a file that merely exists somewhere unrelated.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct HostFacts {
    /// Inference route this host would serve, decided by [`probe_host`].
    pub inference: InferenceKind,
    /// A CUDA device answered the driver probe. On its own this is not a runtime.
    pub native_cuda: bool,
    /// Resolved inference runtime, e.g. an ONNX Runtime library or the
    /// `"<aurora-native-cuda>"` zero-DLL route.
    pub inference_runtime: Option<String>,
    /// Why no inference runtime resolved.
    pub runtime_error: Option<String>,
    /// Canonical face landmark/mesh model set, named by its mesh file.
    pub landmarks_model: Option<String>,
    /// Why the landmark model set does not resolve.
    pub landmarks_refusal: Option<String>,
    /// Canonical `FaceParsingV2` model, named when it resolves.
    pub parsing_model: Option<String>,
    /// Why `FaceParsingV2` does not resolve.
    pub parsing_refusal: Option<String>,
    /// The human-parsing segmenter (`skinMasking.onnx`, Capture One's
    /// `AIModels/Common/PeopleMasking`) is on disk.
    pub human_parsing_model: bool,
    /// The PF8033 cross-forward artifacts (dx/dy/mask) resolve.
    pub smile_artifact: bool,
    pub plump_models: bool,
    /// Decrypted makeup material root, when one is installed.
    pub makeup_asset_root: Option<PathBuf>,
    /// Per-`pf` material presence for the makeup rows that need a PNG.
    pub makeup_material: BTreeMap<u32, bool>,
    /// Monotonic id of this fact set. Frame evidence is only reused while the
    /// host facts it was computed under are unchanged, so installing a model or
    /// an asset invalidates the negative results that were waiting for it.
    pub epoch: u64,
}

impl HostFacts {
    /// Whether any model-backed row can run at all.
    pub(crate) fn backend_ready(&self) -> bool {
        self.inference != InferenceKind::None
    }
}

/// What one photo's own frame proved.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FrameFacts {
    /// Same-frame inputs the shared producers actually returned.
    pub proven_inputs: BTreeSet<InputDependency>,
    /// A model forward for this frame returned usable output. Combined with the
    /// runtime identity, this is what lets a native claim be more than a string.
    pub model_forward_ok: bool,
    /// Why an input is not proven. Reported through the engine log; the
    /// snapshot's own vocabulary is the `required` list.
    pub notes: BTreeMap<InputDependency, String>,
    /// True when the failure was a load/runtime problem a later call may fix,
    /// in which case the result is not cached.
    pub retryable: bool,
}

/// Per-row asset class for a makeup row's `MakeupBundle` dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MakeupAsset {
    /// The row needs no material PNG (owner mask rasterized from landmarks).
    NoAssetNeeded,
    /// A material PNG for this row is installed.
    AssetPresent,
    /// No material PNG for this row is installed.
    AssetMissing,
    /// Not a row this probe models; keep the registry's own answer.
    Undecided,
}

// ---------------------------------------------------------------------------
// Caches
// ---------------------------------------------------------------------------

fn host_cache() -> &'static Mutex<Option<(Instant, Arc<HostFacts>)>> {
    static CACHE: OnceLock<Mutex<Option<(Instant, Arc<HostFacts>)>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

/// Host facts, re-read at most once per [`HOST_FACTS_TTL`].
pub(crate) fn host_facts() -> Arc<HostFacts> {
    host_facts_after(false)
}

/// Host facts with the TTL bypassed. `detail=true` capability queries use this:
/// an explicit refresh must see a model or asset installed after the last window.
pub(crate) fn host_facts_fresh() -> Arc<HostFacts> {
    host_facts_after(true)
}

fn host_facts_after(refresh: bool) -> Arc<HostFacts> {
    {
        let cache = host_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((at, facts)) = cache.as_ref() {
            if !refresh && at.elapsed() < HOST_FACTS_TTL {
                return Arc::clone(facts);
            }
        }
    }
    // Probe outside the lock: `from_environment` stats model files and the
    // parsing model check may hash once, which must not block another caller.
    let probe = probe_host();
    let facts = {
        let mut cache = host_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Unchanged facts keep their epoch, so an unrelated TTL refresh does not
        // throw away per-frame evidence that is still true.
        let epoch = match cache.as_ref() {
            Some((_, previous)) if same_host_facts(previous, &probe) => previous.epoch,
            _ => next_host_epoch(),
        };
        let facts = Arc::new(HostFacts { epoch, ..probe });
        log::debug!(
            "portrait host facts #{}: runtime={:?} landmarks={:?} parsing={:?} smile={} makeup_root={:?}",
            facts.epoch,
            facts.inference,
            facts.landmarks_model,
            facts.parsing_model,
            facts.smile_artifact,
            facts.makeup_asset_root.is_some()
        );
        *cache = Some((Instant::now(), Arc::clone(&facts)));
        facts
    };
    facts
}

/// Field-wise comparison that ignores [`HostFacts::epoch`]: a refresh that saw
/// the same machine must not invalidate per-frame evidence.
fn same_host_facts(a: &HostFacts, b: &HostFacts) -> bool {
    a.inference == b.inference
        && a.native_cuda == b.native_cuda
        && a.inference_runtime == b.inference_runtime
        && a.runtime_error == b.runtime_error
        && a.landmarks_model == b.landmarks_model
        && a.landmarks_refusal == b.landmarks_refusal
        && a.parsing_model == b.parsing_model
        && a.parsing_refusal == b.parsing_refusal
        && a.plump_models == b.plump_models
        && a.smile_artifact == b.smile_artifact
        && a.human_parsing_model == b.human_parsing_model
        && a.makeup_asset_root == b.makeup_asset_root
        && a.makeup_material == b.makeup_material
}

fn next_host_epoch() -> u64 {
    static CLOCK: AtomicU64 = AtomicU64::new(1);
    CLOCK.fetch_add(1, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Host probing
// ---------------------------------------------------------------------------

/// The synthetic library path the zero-DLL native route reports. A real
/// `onnxruntime.dll` path never equals it, which is how "native" stays provable.
const NATIVE_CUDA_MARKER: &str = "<aurora-native-cuda>";

/// A real CUDA device answered the driver. This is a driver call, not a load.
fn native_cuda_device() -> bool {
    #[cfg(windows)]
    {
        aurora_portrait::cuda::driver_memory_info().is_ok()
    }
    #[cfg(not(windows))]
    {
        false
    }
}

fn metal_device_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        aurora_portrait::metal::available()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// What the process was asked to render through. Parsed from the same names the
/// render path reads, so the snapshot can never disagree with the route it
/// describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeviceRequest {
    Auto,
    Cpu,
    Cuda,
    Metal,
    Wgpu,
    Vulkan,
    Dx12,
    /// A named backend this build has no portrait lane for.
    Unsupported(&'static str),
}

impl DeviceRequest {
    fn parse(value: &str) -> Self {
        let value = value.trim().to_ascii_lowercase();
        match value.as_str() {
            "" | "auto" | "default" => Self::Auto,
            "cpu" => Self::Cpu,
            "cuda" | "native" => Self::Cuda,
            "metal" => Self::Metal,
            "wgpu" => Self::Wgpu,
            "vulkan" | "vk" => Self::Vulkan,
            "dx12" | "d3d12" => Self::Dx12,
            "dx11" | "d3d11" => Self::Unsupported("dx11"),
            "opencl" => Self::Unsupported("opencl"),
            "rocm" => Self::Unsupported("rocm"),
            "npu" => Self::Unsupported("npu"),
            _ => Self::Unsupported("unknown"),
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Wgpu => "wgpu",
            Self::Vulkan => "vulkan",
            Self::Dx12 => "dx12",
            Self::Unsupported(name) => name,
        }
    }
}

/// Pure request resolution, so the precedence is testable without touching the
/// environment: `AURORA_PORTRAIT_RENDER_BACKEND` wins, then `AURORA_GPU_BACKEND`,
/// and `PORTRAIT_CUDA_REQUIRED` forces CUDA over both.
pub(crate) fn device_request_from(
    portrait_backend: Option<&str>,
    gpu_backend: Option<&str>,
    cuda_required: bool,
) -> DeviceRequest {
    if cuda_required {
        return DeviceRequest::Cuda;
    }
    match portrait_backend.map(DeviceRequest::parse) {
        Some(DeviceRequest::Auto) | None => {}
        Some(request) => return request,
    }
    gpu_backend.map_or(DeviceRequest::Auto, DeviceRequest::parse)
}

pub(crate) fn device_request() -> DeviceRequest {
    let portrait = std::env::var("AURORA_PORTRAIT_RENDER_BACKEND").ok();
    let gpu = std::env::var("AURORA_GPU_BACKEND").ok();
    let cuda_required = std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_some_and(|value| {
        let value = value.to_string_lossy().trim().to_ascii_lowercase();
        !value.is_empty() && value != "0" && value != "false"
    });
    device_request_from(portrait.as_deref(), gpu.as_deref(), cuda_required)
}

/// What the requested device means for this host.
///
/// `cuda`/`metal` describe the **render** lane (the compiled native kernels and a
/// device that answers), not the inference provider: a compatibility ONNX Runtime
/// may feed tensors while the shading runs on real CUDA kernels, and that is not a
/// zero-ORT claim. The provider actually resolved is recorded separately in
/// [`HostFacts::inference`]; only a strict zero-ORT switch would make a
/// compatibility runtime insufficient, and this build has none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DevicePlan {
    /// CPU fallback is disallowed.
    pub strict_gpu: bool,
    /// The native CUDA render lane is usable.
    pub cuda: bool,
    /// The Metal render lane is usable.
    pub metal: bool,
    /// The request cannot be served here; every model-backed row must say so.
    pub refusal: Option<&'static str>,
}

impl DevicePlan {
    pub(crate) fn cpu_only() -> Self {
        Self {
            strict_gpu: false,
            cuda: false,
            metal: false,
            refusal: None,
        }
    }
}

/// Cross-check the request against the host: an explicitly requested backend is
/// only honoured when this build really has that lane and a device answers. A
/// local CUDA device never satisfies a WGPU/Vulkan request, and a CPU request is
/// never reported as GPU.
pub(crate) fn resolve_device(request: DeviceRequest, host: &HostFacts) -> DevicePlan {
    match request {
        DeviceRequest::Cpu => DevicePlan::cpu_only(),
        DeviceRequest::Auto => DevicePlan {
            strict_gpu: false,
            // The render lane only: a real device is enough, whichever provider
            // happened to be loaded for inference.
            cuda: host.native_cuda,
            metal: metal_device_available(),
            refusal: None,
        },
        DeviceRequest::Cuda => {
            if host.native_cuda {
                DevicePlan {
                    strict_gpu: true,
                    cuda: true,
                    metal: false,
                    refusal: None,
                }
            } else {
                DevicePlan {
                    strict_gpu: true,
                    cuda: false,
                    metal: false,
                    refusal: Some(
                        "CUDA was requested but no native CUDA device resolved on this host",
                    ),
                }
            }
        }
        DeviceRequest::Metal => {
            if metal_device_available() {
                DevicePlan {
                    strict_gpu: true,
                    cuda: false,
                    metal: true,
                    refusal: None,
                }
            } else {
                DevicePlan {
                    strict_gpu: true,
                    cuda: false,
                    metal: false,
                    refusal: Some("Metal was requested but no Metal device resolved on this host"),
                }
            }
        }
        DeviceRequest::Wgpu => DevicePlan {
            strict_gpu: true,
            cuda: false,
            metal: false,
            refusal: Some("wgpu was requested but this build has no wgpu portrait lane"),
        },
        DeviceRequest::Vulkan => DevicePlan {
            strict_gpu: true,
            cuda: false,
            metal: false,
            refusal: Some("Vulkan was requested but this build has no Vulkan portrait lane"),
        },
        DeviceRequest::Dx12 => DevicePlan {
            strict_gpu: true,
            cuda: false,
            metal: false,
            refusal: Some("dx12 was requested but this build has no dx12 portrait lane"),
        },
        DeviceRequest::Unsupported(name) => DevicePlan {
            strict_gpu: true,
            cuda: false,
            metal: false,
            // Named at compile time so the reason is a `'static` string.
            refusal: Some(match name {
                "dx11" => "dx11 was requested but this build has no dx11 portrait lane",
                "opencl" => "opencl was requested but this build has no opencl portrait lane",
                "rocm" => "rocm was requested but this build has no rocm portrait lane",
                "npu" => "npu was requested but this build has no npu portrait lane",
                _ => "an unknown backend was requested; no portrait lane can serve it",
            }),
        },
    }
}

/// Log-safe frame name: the photo's own id never reaches the log, and neither does
/// its absolute path.
pub(crate) fn frame_label(identity: &FrameIdentity) -> String {
    format!(
        "len={} mtime={} token={:?}",
        identity.len, identity.mtime_nanos, identity.token
    )
}

fn probe_host() -> HostFacts {
    let mut facts = HostFacts::default();
    facts.native_cuda = native_cuda_device();
    // Record what the runtime really resolved to. Nothing is inferred from the
    // environment: a native claim needs the zero-DLL marker *and* a device behind
    // it, a real ONNX Runtime library stays a compatibility runtime, and the marker
    // without a device is reported as no runtime rather than as native.
    match crate::aimask::runtime_info() {
        Some((path, version, api)) => {
            if path == NATIVE_CUDA_MARKER {
                if facts.native_cuda {
                    facts.inference = InferenceKind::NativeCuda;
                } else {
                    facts.runtime_error = Some(
                        "the native route reported itself without a CUDA device; not claiming a runtime"
                            .to_owned(),
                    );
                }
            } else if path == "<aurora-native>" {
                facts.inference = InferenceKind::NativePerModel;
            } else {
                facts.inference = InferenceKind::OrtCompat;
            }
            facts.inference_runtime = Some(format!("{path} ({version}, api {api})"));
        }
        None => {
            facts.runtime_error = Some(
                "no ONNX Runtime library and no native inference route on this host".to_owned(),
            );
        }
    }
    match crate::FaceInputModelPaths::from_environment(false) {
        Ok(paths)
            if [
                &paths.detection,
                &paths.mesh,
                &paths.coarse,
                &paths.dense,
                &paths.extra52,
                &paths.landmark_constants,
            ]
            .into_iter()
            .all(|path| path.is_file()) =>
        {
            facts.landmarks_model = Some("face detector, mesh and landmark set".to_owned());
        }
        Ok(_) => {
            facts.landmarks_refusal =
                Some("missing files in face detector/mesh/landmark set".to_owned())
        }
        Err(refusal) => facts.landmarks_refusal = Some(refusal_reason(&refusal)),
    }
    match crate::FaceInputModelPaths::from_environment(true) {
        Ok(_) => facts.parsing_model = Some("FaceParsingV2".to_owned()),
        Err(refusal) => facts.parsing_refusal = Some(refusal_reason(&refusal)),
    }
    facts.plump_models = crate::portrait_stage::discover_plump_onnx_models().is_ok();
    facts.smile_artifact = crate::Pf8033CpuForwardReceipt::discover().is_ok();
    facts.makeup_asset_root = crate::portrait_makeup_wiring::makeup_asset_root();
    facts.makeup_material = makeup_material_presence();
    // The human-parsing segmenter is the same model the skin/hair/face/body
    // planes come from; `model_available` is a file check, never a load.
    facts.human_parsing_model = crate::aimask::model_available(HUMAN_PARSING_MODEL);
    facts
}

/// The makeup rows whose `MakeupBundle` needs a decrypted material PNG.
///
/// Derived from the owning module instead of a duplicated table where possible:
/// `makeup_rows_2d_available()` lists the rows that run from landmarks alone,
/// and every other declared 2D row needs a material.
fn makeup_material_rows() -> impl Iterator<Item = u32> {
    crate::portrait_makeup_wiring::makeup_rows_2d_declared()
        .iter()
        .copied()
        .filter(|pf| !crate::portrait_makeup_wiring::makeup_rows_2d_available().contains(pf))
}

/// Per-row material presence, checked the way that row's own loader resolves it.
///
/// `makeup_presets_for_pf` is only one of two contracts: PF8/PF9 have no presets
/// directory at all and take their material from the static `FaceStereo` set, so an
/// empty preset list says nothing about them. Rows with no accurate check are simply
/// absent from the map, which leaves the registry's own answer instead of a guess.
fn makeup_material_presence() -> BTreeMap<u32, bool> {
    use crate::portrait_makeup_wiring::{makeup_asset_root, row_material_selection};
    let root = makeup_asset_root();
    makeup_material_rows()
        .map(|pf| {
            (
                pf,
                row_material_selection(root.as_deref(), pf, None)
                    .ok()
                    .flatten()
                    .is_some(),
            )
        })
        .collect()
}

fn refusal_reason(refusal: &Refusal) -> String {
    match refusal {
        Refusal::MissingInput(name) => format!("missing input: {name}"),
        Refusal::ShapeMismatch(name) => format!("shape mismatch: {name}"),
        Refusal::ValueOutOfRange(name) => format!("value out of range: {name}"),
        Refusal::UnknownEffect(pf) => format!("unknown effect: PF{pf}"),
        Refusal::NotImplemented(pf) => format!("not implemented: PF{pf}"),
    }
}

/// A load/runtime failure that a later call may fix, as opposed to a
/// deterministic per-frame refusal. Mirrors the face producer's own retryable
/// classification: runtime and model-load errors are not cached.
fn transient_refusal(refusal: &Refusal) -> bool {
    let Refusal::MissingInput(name) = refusal else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    lower.contains("runtime") || lower.contains("onnx") || lower.contains("model file")
}

// ---------------------------------------------------------------------------
// Frame probing
// ---------------------------------------------------------------------------

fn record_mask(facts: &mut FrameFacts, dep: InputDependency, plane: Option<&[f32]>) {
    match plane {
        Some(plane)
            if plane
                .iter()
                .any(|value| value.is_finite() && *value > MASK_SIGNAL) =>
        {
            facts.proven_inputs.insert(dep);
        }
        Some(_) => {
            facts.notes.insert(
                dep,
                format!(
                    "{} plane carries no target (peak <= {MASK_SIGNAL})",
                    dep.as_str()
                ),
            );
        }
        None => {
            facts.notes.insert(
                dep,
                format!(
                    "{} plane absent: skinMasking.onnx produced nothing",
                    dep.as_str()
                ),
            );
        }
    }
}

fn note_model_inputs(facts: &mut FrameFacts, reason: &str) {
    for dep in [
        InputDependency::Landmarks194,
        InputDependency::FaceParsingV2,
        InputDependency::TeethMask,
        InputDependency::EyeMask,
    ] {
        facts.notes.entry(dep).or_insert_with(|| reason.to_owned());
    }
}

/// Frame-level facts and whether this frame can be produced at all. Needs no model
/// forward: it reads the cached planes and the host gates.
fn probe_frame_level(host: &HostFacts, source: &Source) -> (FrameFacts, bool) {
    let mut facts = FrameFacts::default();
    let masks = source.portrait_masks();
    record_mask(&mut facts, InputDependency::SkinMask, masks.skin.as_deref());
    record_mask(&mut facts, InputDependency::HairMask, masks.hair.as_deref());
    // The face plane belongs to the same segmenter as the skin class, but a hair
    // plane is a different class: `HumanParsingMask` is only claimed when the
    // segmenter model is installed *and* the skin class carried a target.
    if host.human_parsing_model && facts.proven_inputs.contains(&InputDependency::SkinMask) {
        facts
            .proven_inputs
            .insert(InputDependency::HumanParsingMask);
    } else if host.human_parsing_model {
        facts.notes.insert(
            InputDependency::HumanParsingMask,
            "no skin class carried a target; a hair plane is not a parsing result".to_owned(),
        );
    }

    if !host.backend_ready() {
        let why = host
            .runtime_error
            .clone()
            .unwrap_or_else(|| "inference runtime unavailable".to_owned());
        note_model_inputs(&mut facts, &why);
        facts.retryable = true;
        return (facts, false);
    }
    if host.landmarks_model.is_none() {
        let why = host
            .landmarks_refusal
            .clone()
            .unwrap_or_else(|| "face landmark model set missing".to_owned());
        note_model_inputs(&mut facts, &why);
        return (facts, false);
    }
    (facts, true)
}

/// Facts that only a real bundle can give.
fn extract_bundle_facts(
    facts: &mut FrameFacts,
    bundle: &crate::portrait_face_input_stage::FaceInputBundle,
) {
    let landmarks_ok = bundle.points194.len() == 194
        && bundle
            .points194
            .iter()
            .flatten()
            .all(|value| value.is_finite());
    if landmarks_ok {
        facts.proven_inputs.insert(InputDependency::Landmarks194);
        facts.model_forward_ok = true;
    } else {
        facts.notes.insert(
            InputDependency::Landmarks194,
            "face bundle carries no usable 194 landmarks".to_owned(),
        );
    }
    match bundle
        .model_inputs
        .as_ref()
        .and_then(|inputs| inputs.parsing.as_ref())
    {
        Some(parsing) => {
            let readable =
                (0..usize::from(parsing.channels)).any(|class| parsing.class(class).is_some());
            if readable {
                facts.proven_inputs.insert(InputDependency::FaceParsingV2);
            } else {
                facts.notes.insert(
                    InputDependency::FaceParsingV2,
                    "face parsing bundle carries no readable class plane".to_owned(),
                );
            }
            facts
                .notes
                .entry(InputDependency::TeethMask)
                .or_insert_with(|| {
                    "FaceParsingV2 has no class -> teeth mapping; the teeth mask is a \
                 separate producer the render path attaches per row"
                        .to_owned()
                });
        }
        None => {
            facts.notes.insert(
                InputDependency::FaceParsingV2,
                "face bundle carries no FaceParsingV2 planes".to_owned(),
            );
        }
    }
    facts
        .notes
        .entry(InputDependency::EyeMask)
        .or_insert_with(|| {
            "no same-frame eye input producer (PF101 / PF100004) has run for this photo".to_owned()
        });
}

// ---------------------------------------------------------------------------
// Snapshot assembly
// ---------------------------------------------------------------------------

/// Whether the unified dispatcher has a route for `pf` at all. A row with no
/// route can only ever be stored, whatever is installed.
///
/// PF109/PF110 are the teeth family's second and third controls and are served by
/// the PF108 stage, so they are judged by that route instead of by a lookup of
/// their own number.
fn staged(pf: u32) -> bool {
    let routed = match pf {
        109 | 110 => 108,
        other => other,
    };
    aurora_portrait::lookup(routed).is_some_and(|meta| meta.access == aurora_portrait::Access::Cpu)
        && crate::portrait_stage::is_bridged_effect(routed)
}

fn makeup_asset(pf: u32, host: &HostFacts) -> MakeupAsset {
    // No row is refused here on principle. PF8/PF9 included: whether their owner
    // mask and UV can be assembled is decided by the row's own producer in
    // `produce_row`, and the cheap tier reports `PendingSameFrame` until it ran.
    if crate::portrait_makeup_wiring::makeup_rows_2d_available().contains(&pf) {
        return MakeupAsset::NoAssetNeeded;
    }
    // PF90021 enhances the bundle's own lip plane and reads no material PNG.
    if pf == 90021 {
        return MakeupAsset::NoAssetNeeded;
    }
    match host.makeup_material.get(&pf) {
        Some(true) => MakeupAsset::AssetPresent,
        Some(false) => MakeupAsset::AssetMissing,
        None => MakeupAsset::Undecided,
    }
}

/// Build the snapshot for one photo from facts that were *observed*, never from
/// `RuntimeProbeContext::fully_ready()`.
///
/// This is the form without row-level evidence; [`assemble_with_rows`] folds in
/// what the row producers themselves returned.
pub(crate) fn assemble(host: &HostFacts, frame: Option<&FrameFacts>) -> PortraitCapabilitySnapshot {
    assemble_with_rows(host, frame, None)
}

pub(crate) fn assemble_with_rows(
    host: &HostFacts,
    frame: Option<&FrameFacts>,
    rows: Option<&BTreeMap<u32, PfInputVerdict>>,
) -> PortraitCapabilitySnapshot {
    assemble_for(host, frame, rows, resolve_device(device_request(), host))
}

/// The assembly itself, with the resolved device passed in so the request/host
/// cross-check is decided once and can be tested without touching the environment.
pub(crate) fn assemble_for(
    host: &HostFacts,
    frame: Option<&FrameFacts>,
    rows: Option<&BTreeMap<u32, PfInputVerdict>>,
    plan: DevicePlan,
) -> PortraitCapabilitySnapshot {
    let mut ctx = RuntimeProbeContext {
        // The CPU reference kernels are compiled into this binary, but they still
        // need the models that feed them, so a host without a runtime reports the
        // backend gap instead of pretending those rows could run.
        cpu_available: host.backend_ready(),
        // Only the device the request actually resolves to is offered to the
        // registry; a local CUDA card never stands in for a wgpu/Vulkan request.
        cuda_available: plan.cuda,
        metal_available: plan.metal,
        strict_gpu: plan.strict_gpu,
        ..RuntimeProbeContext::default()
    };
    if host.backend_ready() {
        if host.plump_models {
            ctx.available_models
                .insert(InputDependency::FacePlumpResidual);
        }
        if host.landmarks_model.is_some() {
            ctx.available_models.insert(InputDependency::Landmarks194);
        }
        if host.parsing_model.is_some() {
            ctx.available_models.insert(InputDependency::FaceParsingV2);
        }
        if host.human_parsing_model {
            ctx.available_models
                .insert(InputDependency::HumanParsingMask);
        }
    }
    if let Some(frame) = frame {
        ctx.available_inputs = frame.proven_inputs.clone();
    }
    // Host-level asset gates. Per-row material presence is refined below, so a
    // row that needs a PNG the host does not ship is still named as an asset
    // gap even when the root exists.
    if host.makeup_asset_root.is_some() {
        ctx.available_assets.insert(InputDependency::MakeupBundle);
    }
    if host.smile_artifact {
        ctx.available_assets.insert(InputDependency::SmileWarpAsset);
    }

    let mut snapshot = capability_snapshot(&ctx);
    let backend_reason = host
        .runtime_error
        .clone()
        .unwrap_or_else(|| "no inference backend is available on this host".to_owned());

    for entry in snapshot.effects.iter_mut() {
        let Some(descriptor) = descriptor_by_pf(entry.pf) else {
            continue;
        };
        if !staged(entry.pf) {
            // No dispatcher route: the parameter is persisted and nothing else.
            entry.availability = RuntimeAvailability::StoredOnly;
            continue;
        }
        if let Some(reason) = plan.refusal {
            // The requested device cannot serve this build. A local card of another
            // kind does not stand in for it, and CPU is not a silent substitute.
            entry.availability = RuntimeAvailability::BackendUnavailable {
                reason: reason.to_owned(),
            };
            continue;
        }
        if !host.backend_ready() {
            entry.availability = RuntimeAvailability::BackendUnavailable {
                reason: backend_reason.clone(),
            };
            continue;
        }

        if let Some(asset) = crate::portrait_stage::missing_tone_table(entry.pf) {
            // The stage itself refuses without this table; a mask cannot replace it.
            entry.availability = RuntimeAvailability::MissingAsset {
                required: asset_deps(descriptor),
            };
            log::debug!(
                "PF{} needs a stage asset this build does not ship: {asset}",
                entry.pf
            );
            continue;
        }

        let mut row_ctx = ctx.clone();
        let mut refined = false;
        if let Some(ready) = crate::portrait_stage::face_part_table_ready(entry.pf) {
            if ready {
                row_ctx
                    .available_assets
                    .insert(InputDependency::FacePartTable);
            }
            refined = true;
        }
        if descriptor
            .dependencies
            .contains(&InputDependency::SkinToneLut)
        {
            row_ctx
                .available_assets
                .insert(InputDependency::SkinToneLut);
            refined = true;
        }
        if descriptor
            .dependencies
            .contains(&InputDependency::MakeupBundle)
        {
            refined = true;
            match makeup_asset(descriptor.id.pf, host) {
                MakeupAsset::AssetMissing => {
                    entry.availability = RuntimeAvailability::MissingAsset {
                        required: vec![InputDependency::MakeupBundle],
                    };
                    continue;
                }
                MakeupAsset::NoAssetNeeded | MakeupAsset::AssetPresent => {
                    row_ctx
                        .available_assets
                        .insert(InputDependency::MakeupBundle);
                    if matches!(entry.pf, 8 | 9)
                        && host.makeup_material.get(&entry.pf) == Some(&true)
                    {
                        row_ctx
                            .available_assets
                            .insert(InputDependency::FaceStereoAsset);
                    }
                }
                MakeupAsset::Undecided => {}
            }
        }
        if descriptor
            .dependencies
            .contains(&InputDependency::SmileWarpAsset)
        {
            refined = true;
            if host.smile_artifact {
                row_ctx
                    .available_assets
                    .insert(InputDependency::SmileWarpAsset);
            } else {
                entry.availability = RuntimeAvailability::MissingAsset {
                    required: vec![InputDependency::SmileWarpAsset],
                };
                continue;
            }
        }
        if refined {
            entry.availability = descriptor.evaluate_availability(&row_ctx);
        }
        if host.plump_models
            && frame.is_some_and(|f| {
                f.proven_inputs.contains(&InputDependency::Landmarks194)
                    && !f.notes.contains_key(&InputDependency::FacePlumpResidual)
            })
            && matches!(&entry.availability, RuntimeAvailability::MissingInput { required } if required.as_slice() == [InputDependency::FacePlumpResidual])
        {
            entry.availability = RuntimeAvailability::PendingSameFrame {
                required: vec![InputDependency::FacePlumpResidual],
            };
        }
    }

    // Row-level evidence is applied last: the producers are the only thing that
    // can tell "the inputs are there" from "the inputs are missing", and a row
    // they have not produced yet is pending, not available.
    if let Some(rows) = rows {
        for entry in snapshot.effects.iter_mut() {
            let Some(verdict) = rows.get(&entry.pf) else {
                continue;
            };
            let Some(descriptor) = descriptor_by_pf(entry.pf) else {
                continue;
            };
            apply_row_verdict(entry, descriptor, verdict);
        }
    }

    // A probed row with no verdict in this call is not evidence of anything. With a
    // frame it is pending measurement, without one the input is missing; this covers
    // a refused cache, a partial row map and an absent one alike, so a row cannot stay
    // green just because some other row was measured.
    {
        let probed = probed_pfs();
        for entry in snapshot.effects.iter_mut() {
            if !probed.contains(&entry.pf) || !entry.availability.is_available() {
                continue;
            }
            if rows.is_some_and(|rows| rows.contains_key(&entry.pf)) {
                continue;
            }
            let Some(descriptor) = descriptor_by_pf(entry.pf) else {
                continue;
            };
            entry.availability = if frame.is_some() {
                RuntimeAvailability::PendingSameFrame {
                    required: same_frame_deps(descriptor),
                }
            } else {
                RuntimeAvailability::MissingInput {
                    required: same_frame_deps(descriptor),
                }
            };
        }
    }

    snapshot.available_effects = snapshot
        .effects
        .iter()
        .filter(|entry| entry.availability.is_available())
        .count();
    snapshot
}

/// Fold one produced row verdict into the registry's own answer.
///
/// A host-level gap (no runtime, no model, no asset root) outranks the row: the
/// row cannot run whatever its inputs do, so those verdicts are left alone.
fn apply_row_verdict(
    entry: &mut crate::portrait_registry::EffectCapabilityEntry,
    descriptor: &crate::portrait_registry::EffectDescriptor,
    verdict: &PfInputVerdict,
) {
    let hosts_says_blocked = matches!(
        entry.availability,
        RuntimeAvailability::BackendUnavailable { .. }
            | RuntimeAvailability::MissingModel { .. }
            | RuntimeAvailability::MissingAsset { .. }
            | RuntimeAvailability::StoredOnly
    );
    match verdict {
        PfInputVerdict::Proven(_) => {
            if !hosts_says_blocked {
                entry.availability = RuntimeAvailability::Available;
            }
        }
        PfInputVerdict::ReadyNotProduced(_) => {
            // Nothing is missing yet, but nothing was produced either: pending,
            // never green — and never over a host that cannot run the row at all.
            if !hosts_says_blocked {
                entry.availability = RuntimeAvailability::PendingSameFrame {
                    required: same_frame_deps(descriptor),
                };
            }
        }
        PfInputVerdict::MissingInput(_) => {
            if !hosts_says_blocked {
                entry.availability = RuntimeAvailability::MissingInput {
                    required: same_frame_deps(descriptor),
                };
            }
        }
        PfInputVerdict::MissingAsset(_) => {
            if !hosts_says_blocked {
                entry.availability = RuntimeAvailability::MissingAsset {
                    required: asset_deps(descriptor),
                };
            }
        }
    }
}

/// The per-photo inputs this row waits for, taken from the registry's own list.
fn same_frame_deps(
    descriptor: &crate::portrait_registry::EffectDescriptor,
) -> Vec<InputDependency> {
    let deps: Vec<InputDependency> = descriptor
        .dependencies
        .iter()
        .copied()
        .filter(|dep| dep.is_per_job_input())
        .collect();
    if deps.is_empty() {
        vec![InputDependency::Landmarks194]
    } else {
        deps
    }
}

fn asset_deps(descriptor: &crate::portrait_registry::EffectDescriptor) -> Vec<InputDependency> {
    let deps: Vec<InputDependency> = descriptor
        .dependencies
        .iter()
        .copied()
        .filter(|dep| dep.is_asset_backed())
        .collect();
    if deps.is_empty() {
        vec![InputDependency::MakeupBundle]
    } else {
        deps
    }
}

// ---------------------------------------------------------------------------
// Row-level evidence
// ---------------------------------------------------------------------------

/// Rows whose conclusions are remembered. Each entry holds only the rows one
/// query asked about, never a whole-frame table.
pub(crate) const ROW_EVIDENCE_CAPACITY: usize = 8;

/// Everything a row verdict is bound to besides the row itself.
///
/// The document part is the registry's own [`portrait_input_key`], the same key the
/// availability layer uses: a legal intensity that does not create or remove an
/// input leaves it unchanged (dragging a slider must not grey a row), while the
/// choices that do — style, colour, phase, validity — change it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RowKey {
    host_epoch: u64,
    input_key: u64,
}

/// Frame facts and row evidence from one producer run. Never stitch together
/// two independently cached frames or two different full-face levels.
#[derive(Debug, Clone, Default)]
struct Measurement {
    frame: Arc<FrameFacts>,
    rows: Arc<BTreeMap<u32, PfInputVerdict>>,
}

struct RowEntry {
    identity: FrameIdentity,
    key: RowKey,
    detail: bool,
    measurement: Measurement,
}

/// Bounded row-evidence table with tier precedence: a detail entry always wins
/// over a cheap one for the same key, and a later cheap result never replaces an
/// earlier detail result.
#[derive(Default)]
struct RowCache {
    entries: Vec<RowEntry>,
}

impl RowCache {
    fn lookup(&self, identity: &FrameIdentity, key: RowKey, detail: bool) -> Option<Measurement> {
        // A detail query is an explicit refresh: it re-reads the selected material
        // and re-produces every row, so it never reuses earlier evidence — not even
        // evidence from an earlier detail run of the same file and document state.
        if detail {
            return None;
        }
        self.entries
            .iter()
            .filter(|entry| &entry.identity == identity && entry.key == key)
            .max_by_key(|entry| entry.detail)
            .map(|entry| entry.measurement.clone())
    }

    /// A detail result replaces the previous detail result for its key, downgrades
    /// included; a cheap result is dropped when the key already has detail evidence.
    fn remember(
        &mut self,
        identity: FrameIdentity,
        key: RowKey,
        detail: bool,
        measurement: Measurement,
    ) {
        if !detail
            && self
                .entries
                .iter()
                .any(|entry| entry.identity == identity && entry.key == key && entry.detail)
        {
            return;
        }
        self.entries.retain(|entry| {
            !(entry.identity == identity && entry.key == key && entry.detail == detail)
        });
        self.entries.push(RowEntry {
            identity,
            key,
            detail,
            measurement,
        });
        while self.entries.len() > ROW_EVIDENCE_CAPACITY {
            self.entries.remove(0);
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

fn row_cache() -> &'static Mutex<RowCache> {
    static CACHE: OnceLock<Mutex<RowCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(RowCache::default()))
}
/// The render path's own same-frame face plane, resampled to the frame size that
/// the row producers expect. A plane of the wrong length is dropped by them.
fn face_plane(source: &Source, frame: &Image) -> Option<Vec<u8>> {
    let masks = source.portrait_masks();
    if masks.width == 0 || masks.height == 0 {
        return None;
    }
    masks.mask_u8(
        crate::portrait_stage::MaskKind::Face,
        frame.width,
        frame.height,
        crate::NormRect::FULL,
    )
}

/// Produce this frame's face-input bundle once and lend it to `f`.
///
/// The refusal is returned instead of swallowed: callers report the reason the
/// producer gave, and one call serves both the frame facts and the row tiers.
fn with_frame_bundle<R>(
    host: &HostFacts,
    source: &Source,
    f: impl FnOnce(&crate::portrait_face_input_stage::FaceInputBundle, &Image, Option<&[u8]>) -> R,
) -> Result<R, Refusal> {
    let frame = source.face_input_image();
    let session = crate::FaceInputSession::open_from_env(host.parsing_model.is_some())?;
    let bundle = session.produce(next_ticket(), &frame)?;
    let plane = face_plane(source, &frame);
    Ok(f(&bundle, &frame, plane.as_deref()))
}

/// Extract every row this call reports from one already-produced bundle.
///
/// The cheap tier covers everything checkable without rasterizing an owner mask;
/// `detail` then upgrades each non-teeth row through its own producer and re-reads
/// the material the document selects, which is what makes a changed material on the
/// same file visible.
fn rows_from_bundle(
    bundle: &crate::portrait_face_input_stage::FaceInputBundle,
    frame: &Image,
    plane: Option<&[u8]>,
    adjustments: &AdjustmentSet,
    detail: bool,
) -> BTreeMap<u32, PfInputVerdict> {
    let mut rows = BTreeMap::new();
    let cheap = prove_pf_inputs_with_face_mask(bundle, frame, plane, Some(&adjustments.extra));
    for row in cheap.rows {
        rows.insert(row.pf, row.verdict);
    }
    if detail {
        for pf in probed_pfs() {
            if TEETH_FAMILY_PFS.contains(&pf) {
                continue;
            }
            // One uniform call for every row, PF101 included: the producer reads
            // the document's own controls (style, custom colour), so this module
            // has no per-row convention of its own to apply.
            let row = produce_row(bundle, frame, pf, plane, Some(&adjustments.extra));
            rows.insert(row.pf, row.verdict);
        }
    }
    rows
}

/// Frame facts and row evidence for one document state, from **one** produced
/// bundle, each cached under its own key.
fn capability_evidence(
    host: &HostFacts,
    identity: &FrameIdentity,
    source: &Source,
    adjustments: &AdjustmentSet,
    detail: bool,
) -> (
    Option<Arc<FrameFacts>>,
    Option<Arc<BTreeMap<u32, PfInputVerdict>>>,
) {
    let key = RowKey {
        host_epoch: host.epoch,
        input_key: portrait_input_key(&adjustments.extra),
    };
    let cached_rows = row_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .lookup(identity, key, detail);
    if let Some(measured) = cached_rows {
        return (Some(measured.frame), Some(measured.rows));
    }

    let (level, can_produce) = probe_frame_level(host, source);
    if !can_produce {
        return (Some(Arc::new(level)), None);
    }
    let produced = with_frame_bundle(host, source, |bundle, frame, plane| {
        let mut facts = level.clone();
        extract_bundle_facts(&mut facts, bundle);
        if detail && host.plump_models {
            // One genuine residual producer serves all eight selectors on this
            // frame. A zero UI value need not disable an otherwise usable row.
            let mut controls = BTreeMap::new();
            for pf in crate::PLUMP_PFS {
                let key = format!("pc:{pf}");
                let v = adjustments
                    .extra
                    .get(&key)
                    .copied()
                    .filter(|v| *v != 0.0)
                    .unwrap_or(1.0);
                controls.insert(key, v);
            }
            let mut inputs = crate::ProducerInputs::default();
            match bundle.attach_plump_from_extra(frame, &controls, &mut inputs) {
                Ok(()) if inputs.plump.is_some() => {
                    facts
                        .proven_inputs
                        .insert(InputDependency::FacePlumpResidual);
                }
                Err(why) => {
                    facts
                        .notes
                        .insert(InputDependency::FacePlumpResidual, refusal_reason(&why));
                }
                _ => {}
            }
        }
        let rows = rows_from_bundle(bundle, frame, plane, adjustments, detail);
        (facts, rows)
    });
    match produced {
        Ok((facts, rows)) => {
            let facts = Arc::new(facts);
            let rows = Arc::new(rows);
            log::debug!(
                "portrait row evidence for {}: {} rows, detail={detail}, digest={:016x}",
                frame_label(identity),
                rows.len(),
                key.input_key
            );
            row_cache()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remember(
                    identity.clone(),
                    key,
                    detail,
                    Measurement {
                        frame: Arc::clone(&facts),
                        rows: Arc::clone(&rows),
                    },
                );
            (Some(facts), Some(rows))
        }
        Err(refusal) => {
            let mut facts = level;
            note_model_inputs(&mut facts, &refusal_reason(&refusal));
            facts.retryable = transient_refusal(&refusal);
            (Some(Arc::new(facts)), None)
        }
    }
}

/// Entry point used by [`crate::FileEngine`].
///
/// `identity`/`source` are the photo's own decoded frame when the engine still
/// holds it; `None` means the frame is not decoded, so no same-frame input can be
/// claimed and every input-dependent row stays `MissingInput`.
///
/// `detail` is the explicit refresh: it re-reads the host (a model installed a
/// second ago is seen), re-produces every row for this document state, and its
/// findings replace earlier ones for the same key — including a `Proven` that a
/// later run shows is missing. A cheap query never overwrites detail evidence.
pub(crate) fn capability_for(
    image: ImageId,
    identity: Option<&FrameIdentity>,
    source: Option<&Source>,
    adjustments: &AdjustmentSet,
    detail: bool,
) -> PortraitCapabilitySnapshot {
    let host = if detail {
        host_facts_fresh()
    } else {
        host_facts()
    };
    let (facts, rows) = match (identity, source) {
        (Some(identity), Some(source)) => {
            capability_evidence(&host, identity, source, adjustments, detail)
        }
        _ => (None, None),
    };
    let snapshot = assemble_with_rows(&host, facts.as_deref(), rows.as_deref());
    log::debug!(
        "portrait capability for {image:?}: {}/{} available, rows={}, detail={detail}",
        snapshot.available_effects,
        snapshot.total_effects,
        rows.as_ref().map_or(0, |rows| rows.len())
    );
    snapshot
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portrait_probe_inputs::{PfInputProof, RowReadiness};

    fn host(runtime: bool, landmarks: bool, parsing: bool) -> HostFacts {
        HostFacts {
            inference: if runtime {
                InferenceKind::OrtCompat
            } else {
                InferenceKind::None
            },
            inference_runtime: runtime.then(|| "test runtime".to_owned()),
            landmarks_model: landmarks.then(|| "faceMesh.onnx".to_owned()),
            parsing_model: parsing.then(|| "FaceParsingV2".to_owned()),
            human_parsing_model: runtime && landmarks,
            ..HostFacts::default()
        }
    }

    /// Every same-frame input this probe can prove today.
    fn provable_frame() -> FrameFacts {
        frame(&[
            InputDependency::Landmarks194,
            InputDependency::FaceParsingV2,
            InputDependency::SkinMask,
            InputDependency::HairMask,
            InputDependency::HumanParsingMask,
        ])
    }

    fn frame(inputs: &[InputDependency]) -> FrameFacts {
        FrameFacts {
            proven_inputs: inputs.iter().copied().collect(),
            ..FrameFacts::default()
        }
    }

    fn identity(name: &str) -> FrameIdentity {
        FrameIdentity {
            path: PathBuf::from(name),
            len: 1,
            mtime_nanos: 1,
            token: None,
        }
    }

    #[test]
    fn bounded_cache_binds_both_halves_to_the_same_frame_document_and_host() {
        let mut cache = RowCache::default();
        let key = RowKey {
            host_epoch: 4,
            input_key: 0,
        };
        for n in 0..ROW_EVIDENCE_CAPACITY + 1 {
            cache.remember(identity(&n.to_string()), key, true, Measurement::default());
        }
        let id = identity(&ROW_EVIDENCE_CAPACITY.to_string());
        assert_eq!(cache.len(), ROW_EVIDENCE_CAPACITY);
        assert!(cache.lookup(&identity("0"), key, false).is_none());
        assert!(cache.lookup(&id, key, false).is_some());
        let mut replaced = id.clone();
        replaced.mtime_nanos += 1;
        assert!(cache.lookup(&replaced, key, false).is_none());
        assert!(
            cache
                .lookup(
                    &id,
                    RowKey {
                        host_epoch: 5,
                        ..key
                    },
                    false
                )
                .is_none()
        );
        let key2 = RowKey {
            input_key: 1,
            ..key
        };
        let f = Arc::new(frame(&[InputDependency::FacePlumpResidual]));
        cache.remember(
            id.clone(),
            key2,
            true,
            Measurement {
                frame: Arc::clone(&f),
                ..Measurement::default()
            },
        );
        assert!(
            !cache
                .lookup(&id, key, false)
                .unwrap()
                .frame
                .proven_inputs
                .contains(&InputDependency::FacePlumpResidual)
        );
        assert!(Arc::ptr_eq(
            &cache.lookup(&id, key2, false).unwrap().frame,
            &f
        ));
    }

    #[test]
    fn assembly_without_a_backend_names_the_backend_and_unlocks_nothing() {
        let snapshot = assemble(&host(false, true, true), None);
        assert_eq!(snapshot.available_effects, 0);
        for entry in &snapshot.effects {
            assert!(
                !entry.availability.is_available(),
                "PF{} must not be available without an inference backend",
                entry.pf
            );
            assert!(
                matches!(
                    entry.availability,
                    RuntimeAvailability::BackendUnavailable { .. }
                        | RuntimeAvailability::StoredOnly
                ),
                "PF{} reported {:?}",
                entry.pf,
                entry.availability
            );
        }
    }

    #[test]
    fn models_alone_never_unlock_a_row() {
        let snapshot = assemble(&host(true, true, true), None);
        assert_eq!(
            snapshot.available_effects, 0,
            "installed models are not same-frame evidence"
        );
        let whitening = snapshot.by_pf(90005).expect("PF90005 is registered");
        assert!(
            !whitening.availability.is_available(),
            "got {:?}",
            whitening.availability
        );
        assert!(
            !matches!(
                whitening.availability,
                RuntimeAvailability::BackendUnavailable { .. }
            ),
            "the backend is available on this host"
        );
    }

    #[test]
    fn missing_parsing_model_names_the_model_rather_than_the_input() {
        let snapshot = assemble(&host(true, true, false), None);
        let teeth = snapshot.by_pf(108).expect("PF108 is registered");
        match &teeth.availability {
            RuntimeAvailability::MissingModel { required } => assert!(
                required.contains(&InputDependency::FaceParsingV2),
                "a missing parsing model must be named: {required:?}"
            ),
            other => panic!("a missing parsing model must be named as a model, got {other:?}"),
        }
    }

    #[test]
    fn only_rows_whose_declared_inputs_are_proven_unlock() {
        let snapshot = assemble(&host(true, true, true), Some(&provable_frame()));

        let geometry = snapshot.by_pf(242).expect("PF242 is registered");
        assert!(
            geometry.availability.is_available()
                == crate::portrait_stage::face_part_table_ready(242).unwrap(),
            "PF242 needs its decoded face-part tables, got {:?}",
            geometry.availability
        );
        let whitening = snapshot.by_pf(90005).expect("PF90005 is registered");
        if aurora_portrait::private_asset!("portrait/skin-adjust/whiten.png").is_some() {
            assert!(
                whitening.availability.is_available(),
                "skin mask + parsing unlock skin tone when the table ships, got {:?}",
                whitening.availability
            );
        } else {
            // A build without the table must report the asset gap: the stage itself
            // refuses this row, so a mask can never make it green.
            assert!(
                matches!(
                    whitening.availability,
                    RuntimeAvailability::MissingAsset { .. }
                ),
                "no whiten LUT shipped, so the row is an asset gap, got {:?}",
                whitening.availability
            );
        }
        assert!(
            snapshot.available_effects >= 30,
            "the proven set must unlock the whole landmark-driven family, got {}",
            snapshot.available_effects
        );

        // Rows whose same-frame plane this probe cannot produce stay honest
        // instead of inheriting the category's readiness.
        for pf in [108u32, 101] {
            let entry = snapshot.by_pf(pf).expect("registered row");
            assert!(
                !entry.availability.is_available(),
                "PF{pf} needs a same-frame plane this probe does not produce, got {:?}",
                entry.availability
            );
        }

        // A different photo's evidence must not leak into this one.
        let empty = assemble(&host(true, true, true), None);
        assert_eq!(empty.available_effects, 0);
        assert!(snapshot.available_effects < snapshot.total_effects);
    }

    #[test]
    fn assetless_makeup_rows_unlock_and_material_rows_do_not() {
        let mut facts = host(true, true, true);
        facts.makeup_material.insert(107, false);
        facts.makeup_material.insert(100, false);
        // Both rows are measured here, so the test is about the host's asset class
        // and not about row evidence.
        let evidence = rows(&[(100, proven_pf101(1)), (107, proven_pf101(1))]);
        let snapshot = assemble_with_rows(&facts, Some(&provable_frame()), Some(&evidence));

        let brow = snapshot.by_pf(100).expect("PF100 is registered");
        assert!(
            brow.availability.is_available(),
            "an owner mask rasterized from landmarks needs no PNG, got {:?}",
            brow.availability
        );
        let lipstick = snapshot.by_pf(107).expect("PF107 is registered");
        assert!(
            matches!(
                lipstick.availability,
                RuntimeAvailability::MissingAsset { .. }
            ),
            "a material row without its PNG is an asset gap even when its inputs were produced, got {:?}",
            lipstick.availability
        );
        let stereo = snapshot.by_pf(8).expect("PF8 is registered");
        assert!(
            !stereo.availability.is_available(),
            "PF8 has no assembler for its owner mask, got {:?}",
            stereo.availability
        );
    }

    #[test]
    fn smile_row_needs_the_warp_artifact_and_the_frame_landmarks() {
        let without = assemble(&host(true, true, true), Some(&provable_frame()));
        let smile = without.by_pf(8033).expect("PF8033 is registered");
        assert!(
            matches!(
                smile.availability,
                RuntimeAvailability::MissingAsset { .. } | RuntimeAvailability::MissingModel { .. }
            ),
            "no smile artifact is installed, got {:?}",
            smile.availability
        );

        let mut facts = host(true, true, true);
        facts.smile_artifact = true;
        let with_artifact = assemble(&facts, Some(&provable_frame()));
        let smile = with_artifact.by_pf(8033).expect("PF8033 is registered");
        assert!(
            smile.availability.is_available(),
            "artifact + 194 landmarks must unlock the row, got {:?}",
            smile.availability
        );
    }

    #[test]
    fn mock_engine_snapshot_never_claims_fully_ready() {
        use crate::ImageEngine;
        let engine = crate::MockEngine::new();
        let snapshot = engine.portrait_capability_snapshot(ImageId(7));
        assert_eq!(
            snapshot.total_effects,
            crate::portrait_registry::descriptors().len()
        );
        assert_eq!(snapshot.available_effects, 0);
        assert!(
            snapshot
                .effects
                .iter()
                .all(|entry| !entry.availability.is_available())
        );
        assert!(snapshot.effects.iter().all(|entry| matches!(
            entry.availability,
            RuntimeAvailability::BackendUnavailable { .. } | RuntimeAvailability::StoredOnly
        )));
    }

    #[test]
    fn file_engine_snapshot_is_per_image_and_starts_without_frame_evidence() {
        use crate::ImageEngine;

        let dir = std::env::temp_dir().join(format!(
            "aurora-portrait-probe-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["a.png", "b.png"] {
            image::RgbaImage::from_pixel(32, 24, image::Rgba([20, 40, 60, 255]))
                .save(dir.join(name))
                .unwrap();
        }
        let engine = crate::FileEngine::open(&dir).expect("open fixture library");
        let images: Vec<ImageId> = engine.images().iter().map(|info| info.id).collect();
        assert_eq!(images.len(), 2);

        let first = engine.portrait_capability_snapshot(images[0]);
        let second = engine.portrait_capability_snapshot(images[1]);
        assert_eq!(
            first.total_effects,
            crate::portrait_registry::descriptors().len()
        );
        assert_eq!(
            second.total_effects, first.total_effects,
            "both photos answer for the same registry"
        );
        // Nothing has decoded these images, so neither may claim a row: an
        // unproven same-frame input is never replaced by the other image's.
        assert_eq!(first.available_effects, 0);
        assert_eq!(second.available_effects, 0);
        assert_eq!(first, second, "no evidence may cross images");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // Row-level evidence
    // -----------------------------------------------------------------------

    fn proven_pf101(style: usize) -> PfInputVerdict {
        PfInputVerdict::Proven(PfInputProof::Pf101 {
            style_index: style,
            presets_on_disk: 1,
            accepted_by: "test",
            material_accepted: true,
            face_mask_accepted: true,
            band_uv_accepted: true,
        })
    }

    fn pending() -> PfInputVerdict {
        PfInputVerdict::ReadyNotProduced(RowReadiness {
            mask_semantic: "test",
            mask_channel: None,
            requires_owner_mask: true,
            requires_material: false,
            geometry_ok: true,
            material: None,
        })
    }

    fn rows(entries: &[(u32, PfInputVerdict)]) -> BTreeMap<u32, PfInputVerdict> {
        entries.iter().cloned().collect()
    }

    #[test]
    fn a_pending_row_is_pending_and_never_available() {
        let mut facts = host(true, true, true);
        facts.makeup_asset_root = Some(PathBuf::from("root"));
        let evidence = rows(&[(107, pending())]);
        let snapshot = assemble_with_rows(&facts, Some(&provable_frame()), Some(&evidence));
        let lipstick = snapshot.by_pf(107).expect("PF107 is registered");
        match &lipstick.availability {
            RuntimeAvailability::PendingSameFrame { required } => {
                assert!(
                    required.contains(&InputDependency::Landmarks194)
                        || required.contains(&InputDependency::FaceParsingV2),
                    "a pending row must name the input it waits for: {required:?}"
                );
            }
            other => panic!("a cheap-only row must be pending, got {other:?}"),
        }
        assert!(!lipstick.availability.is_available());
    }

    #[test]
    fn a_proven_row_needs_the_host_runtime_to_unlock() {
        let evidence = rows(&[(107, proven_pf101(1))]);
        let ready = {
            let mut facts = host(true, true, true);
            facts.makeup_asset_root = Some(PathBuf::from("root"));
            assemble_with_rows(&facts, Some(&provable_frame()), Some(&evidence))
        };
        let lipstick = ready.by_pf(107).expect("PF107 is registered");
        assert!(
            lipstick.availability.is_available(),
            "produced inputs on a ready host are available, got {:?}",
            lipstick.availability
        );

        // Same evidence, no runtime: the host gap outranks the row verdict.
        let without_runtime = assemble_with_rows(&host(false, true, true), None, Some(&evidence));
        for (pf, verdict) in &evidence {
            let entry = without_runtime.by_pf(*pf).expect("registered row");
            assert!(
                !entry.availability.is_available(),
                "PF{pf} claimed {verdict:?} on a host with no runtime"
            );
        }
    }

    #[test]
    fn detail_evidence_replaces_proven_and_cheap_never_overwrites_detail() {
        let identity = identity("frame.png");
        let key = RowKey {
            host_epoch: 3,
            input_key: 7,
        };
        let mut cache = RowCache::default();
        let detail_proven = Measurement {
            rows: Arc::new(rows(&[(107, proven_pf101(1))])),
            ..Measurement::default()
        };
        cache.remember(identity.clone(), key, true, detail_proven.clone());

        // A detail query never reuses earlier evidence; a cheap one may read it.
        assert!(
            cache.lookup(&identity, key, true).is_none(),
            "an explicit refresh must re-read the material and re-produce every row"
        );
        assert!(cache.lookup(&identity, key, false).is_some());

        // A later cheap result may not replace the detail evidence.
        cache.remember(
            identity.clone(),
            key,
            false,
            Measurement {
                rows: Arc::new(rows(&[(107, pending())])),
                ..Measurement::default()
            },
        );
        let hit = cache
            .lookup(&identity, key, false)
            .expect("detail evidence");
        assert!(
            hit.rows.get(&107).is_some_and(PfInputVerdict::is_proven),
            "a cheap refresh overwrote detail evidence: {hit:?}"
        );

        // A later detail run that finds the row missing does replace it.
        let missing = Measurement {
            rows: Arc::new(rows(&[(107, PfInputVerdict::MissingInput("no material"))])),
            ..Measurement::default()
        };
        cache.remember(identity.clone(), key, true, missing);
        let hit = cache
            .lookup(&identity, key, false)
            .expect("newer detail evidence");
        assert!(
            !hit.rows.get(&107).is_some_and(PfInputVerdict::is_proven),
            "detail must be able to downgrade a stale positive: {hit:?}"
        );
        assert_eq!(cache.len(), 1, "one entry per key and tier");
    }

    #[test]
    fn device_requests_follow_the_documented_precedence() {
        use DeviceRequest::*;
        // The portrait path's own name wins over the generic one.
        assert_eq!(device_request_from(Some("wgpu"), Some("cuda"), false), Wgpu);
        assert_eq!(device_request_from(Some("CPU"), Some("cuda"), false), Cpu);
        // `auto` on the portrait path falls through to the generic name.
        assert_eq!(
            device_request_from(Some("auto"), Some("vulkan"), false),
            Vulkan
        );
        assert_eq!(device_request_from(None, Some("cuda"), false), Cuda);
        // `PORTRAIT_CUDA_REQUIRED` outranks both.
        assert_eq!(device_request_from(Some("cpu"), Some("wgpu"), true), Cuda);
        assert_eq!(
            device_request_from(Some("npu"), None, false),
            Unsupported("npu")
        );
        assert_eq!(device_request_from(Some(""), None, false), Auto);
    }

    #[test]
    fn a_wgpu_request_is_never_satisfied_by_a_local_cuda_card() {
        let mut facts = host(true, true, true);
        facts.native_cuda = true;
        facts.inference = InferenceKind::NativeCuda;

        let plan = resolve_device(DeviceRequest::Wgpu, &facts);
        assert!(plan.refusal.is_some(), "wgpu has no lane in this build");
        assert!(
            !plan.cuda,
            "a CUDA card must not stand in for a wgpu request"
        );
        assert!(plan.strict_gpu);

        let snapshot = assemble_for(&facts, Some(&provable_frame()), None, plan);
        assert_eq!(
            snapshot.available_effects, 0,
            "a refused device request may not leave rows green"
        );
        for entry in &snapshot.effects {
            assert!(
                matches!(
                    entry.availability,
                    RuntimeAvailability::BackendUnavailable { .. }
                        | RuntimeAvailability::StoredOnly
                ),
                "PF{} reported {:?} for a refused wgpu request",
                entry.pf,
                entry.availability
            );
        }

        // The same host with a cuda request is served for real.
        let cuda = resolve_device(DeviceRequest::Cuda, &facts);
        assert!(cuda.cuda && cuda.refusal.is_none());
        assert!(assemble_for(&facts, Some(&provable_frame()), None, cuda).available_effects > 0);

        // A compatibility runtime feeding tensors does not stop the CUDA render
        // lane from being usable — that is not a zero-ORT claim either way.
        let compat = {
            let mut facts = facts.clone();
            facts.inference = InferenceKind::OrtCompat;
            facts
        };
        let plan = resolve_device(DeviceRequest::Cuda, &compat);
        assert!(
            plan.cuda && plan.refusal.is_none(),
            "ORT-compat inference must not block real CUDA shading"
        );

        // And a CPU request never reports the card.
        let cpu = resolve_device(DeviceRequest::Cpu, &facts);
        assert!(!cpu.cuda && !cpu.metal && !cpu.strict_gpu);
    }

    #[test]
    fn an_unmeasured_probed_row_is_pending_even_when_other_rows_were_measured() {
        let mut facts = host(true, true, true);
        facts.makeup_asset_root = Some(PathBuf::from("root"));
        let frame = provable_frame();

        // Realistic fixture: brows and liners can build owner masks without
        // a PNG. Before their own producer runs they must stay pending.
        let unmeasured = [100, 103, 117, 30007];
        let baseline = assemble_with_rows(&facts, Some(&frame), None);
        for pf in unmeasured {
            assert!(
                matches!(
                    baseline.by_pf(pf).unwrap().availability,
                    RuntimeAvailability::PendingSameFrame { .. }
                ),
                "PF{pf}"
            );
        }

        let evidence = rows(&[(107, proven_pf101(1))]);
        let snapshot = assemble_with_rows(&facts, Some(&frame), Some(&evidence));
        for pf in unmeasured {
            let entry = snapshot.by_pf(pf).expect("probed row");
            assert!(
                matches!(
                    entry.availability,
                    RuntimeAvailability::PendingSameFrame { .. }
                ),
                "PF{pf} stayed {:?} without a verdict for this call",
                entry.availability
            );
        }
        // The row that *was* measured keeps its own answer.
        let measured = snapshot.by_pf(107).expect("PF107 is registered");
        assert!(measured.availability.is_available());
    }

    #[test]
    fn a_cuda_request_without_a_device_blocks_every_row() {
        let facts = host(true, true, true); // ORT only, no native device
        let plan = resolve_device(DeviceRequest::Cuda, &facts);
        assert!(plan.refusal.is_some());
        let snapshot = assemble_for(&facts, Some(&provable_frame()), None, plan);
        assert_eq!(snapshot.available_effects, 0);
    }

    #[test]
    fn the_teeth_family_second_and_third_controls_use_the_pf108_route() {
        assert_eq!(
            staged(109),
            staged(108),
            "PF109 is served by the PF108 stage"
        );
        assert_eq!(
            staged(110),
            staged(108),
            "PF110 is served by the PF108 stage"
        );
        let snapshot = assemble(&host(true, true, true), Some(&provable_frame()));
        let rows: Vec<_> = [108u32, 109, 110]
            .iter()
            .map(|pf| snapshot.by_pf(*pf).expect("registered teeth row"))
            .collect();
        for entry in &rows {
            assert_ne!(
                entry.availability,
                RuntimeAvailability::StoredOnly,
                "PF{} must not be stored-only: the PF108 stage serves it",
                entry.pf
            );
        }
        assert_eq!(
            rows[0].availability.reason_code(),
            rows[1].availability.reason_code(),
            "the family shares one attachment, so one reason code"
        );
        assert_eq!(
            rows[0].availability.reason_code(),
            rows[2].availability.reason_code()
        );
    }

    #[test]
    fn pending_never_hides_a_missing_backend_or_model() {
        let blocked = assemble_for(
            &host(false, true, true),
            None,
            Some(&rows(&[(107, pending()), (108, pending())])),
            resolve_device(DeviceRequest::Auto, &host(false, true, true)),
        );
        for pf in [107u32, 108] {
            let entry = blocked.by_pf(pf).expect("registered row");
            assert!(
                matches!(
                    entry.availability,
                    RuntimeAvailability::BackendUnavailable { .. }
                        | RuntimeAvailability::MissingModel { .. }
                ),
                "PF{pf} let a pending row hide a host gap: {:?}",
                entry.availability
            );
        }
    }

    #[test]
    fn the_row_key_tracks_input_shaping_controls_not_plain_intensities() {
        let key_of = |extra: &[(&str, f32)]| {
            let mut adjustments = AdjustmentSet::default();
            for (name, value) in extra {
                adjustments.extra.insert((*name).to_owned(), *value);
            }
            portrait_input_key(&adjustments.extra)
        };

        // A legal strength is the same row with the same inputs: dragging a slider
        // must not invalidate the evidence and grey the row.
        assert_eq!(
            key_of(&[("pc:107", 0.2)]),
            key_of(&[("pc:107", 0.9)]),
            "a plain intensity does not create or remove an input"
        );
        assert_eq!(
            key_of(&[("pc:107", 0.9), ("exposure", -1.5)]),
            key_of(&[("pc:107", 0.2), ("exposure", 2.0)]),
            "a tone control is not an input either"
        );

        // The choices that do select another material still change the key.
        assert_ne!(
            key_of(&[("pc:107:style", 2.0)]),
            key_of(&[("pc:107:style", 3.0)]),
            "the style selects a different asset"
        );
        assert_ne!(
            key_of(&[("pc:107:rgb", 0.2)]),
            key_of(&[("pc:107:rgb", 0.6)]),
            "the colour is part of the row's inputs"
        );
    }

    #[test]
    fn a_second_image_never_inherits_row_evidence() {
        let first = identity("first.png");
        let mut second = identity("second.png");
        second.token = Some(9);
        let key = RowKey {
            host_epoch: 1,
            input_key: 0,
        };
        let mut cache = RowCache::default();
        cache.remember(
            first.clone(),
            key,
            true,
            Measurement {
                rows: Arc::new(rows(&[(107, proven_pf101(1))])),
                ..Measurement::default()
            },
        );

        // The first image has evidence for this key: a cheap query reads it.
        assert!(
            cache.lookup(&first, key, false).is_some(),
            "the measured image keeps its own cheap hit"
        );
        // A detail query is an explicit refresh, so it never reuses it.
        assert!(
            cache.lookup(&first, key, true).is_none(),
            "detail must re-measure instead of reusing evidence"
        );
        // The second image never inherits anything, on either tier.
        assert!(cache.lookup(&second, key, false).is_none());
        assert!(cache.lookup(&second, key, true).is_none());

        // A different document state on the same image is a different key too.
        let other_document = RowKey {
            host_epoch: 1,
            input_key: 1,
        };
        assert!(cache.lookup(&first, other_document, false).is_none());
    }
}
