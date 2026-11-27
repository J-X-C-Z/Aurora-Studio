//! # aurora-engine
//!
//! The contract between the Aurora Studio interface and its image-processing engine.
//!
//! * [`AdjustmentSet`] is the full, serializable state of one variant's edits.
//! * [`ImageEngine`] is what an engine implements; the interface calls nothing else.
//! * [`FileEngine`] reads a real folder of images and persists edits in `.portrait`.
//! * [`MockEngine`] is a reference implementation with procedural images, kept for
//!   interface work when no folder is given.
//!
//! Coordinates: masks, spots and picker points are normalized `0.0 ..= 1.0` in the
//! *oriented, uncropped* image (after quarter turns and flips, before crop and fine
//! rotation), so they stay attached to content when the crop changes. Preview
//! regions are normalized in the *developed frame* (after crop and rotation).

pub mod aurora_store;
mod adjust;
mod c1catalog;
mod c1import;
mod c1_masks;
pub mod c1_banding;
pub mod c1_cos;
pub mod c1_eip;
pub mod c1_writeback;
pub mod c1_spots;
pub mod dust_lcc;
pub mod dust_patch;
pub mod retouch_c1;
mod c1session_export;
mod sqlite_write;
mod c1_style_index;
mod c1_styles;
pub(crate) mod c1_model;
pub mod c1_nef_lensdata;
mod c1_wb_oracle;
pub mod c1_retouch;
pub mod c1_keystone_basis;
pub mod c1_ca_six_curves;
pub mod c1_luma_noise;
pub mod c1_usm_linear;
pub mod c1_clarity;
pub mod hairfly_selector;
pub mod comask;
mod camera_defaults;
pub mod camera_params;
mod ca_fix;
mod batch;
mod catalog;
mod closedeyes;
/// Stable pixel/compute contracts shared by CPU, CUDA and Metal adapters.
pub mod contracts;
/// Translation from the in-crate GPU runtime vocabulary to `contracts`.
mod compute_bridge;
/// Persistent edit document versus per-job runtime inputs.
pub mod document;
mod engine;
/// Unified job lifecycle: scoped id, state, cancellation and receipts.
pub mod job_lifecycle;
/// Stages migrated out of `mock` (with compatibility re-exports).
pub mod processing;
/// Production render pipeline facade and `RenderPlan` execution bridge.
pub mod render;
/// Layered export pipeline (`ExportPlan`, `Encoder`, `MetadataWriter`, `AtomicPublisher`).
pub mod export;
/// CPU SIMD kernels (AVX2/NEON) for the export pixel plumbing, with the scalar
/// reference each kernel is checked against.
pub mod simd_fallback;
/// Unified portrait effect descriptor and capability registry.
pub mod portrait_registry;
mod exif;
mod file_engine;
pub mod param_contract;
pub mod auto_exposure;
pub mod fcrv_index;
mod filmcurve;
pub mod face_parsing;
mod heif;
mod heif_write;
mod highlight;
pub mod hdr;
pub mod pano;
pub mod print;
pub mod jxl;
mod iptc;
pub mod lcp;
mod layers;
mod lens_correct;
mod layer_graph;
mod lensdb;
mod matchlook;
mod merge_async;
mod merge_options;
mod panel_commands;
mod mock;

// Metal compute runtime (Mac lane, M2). Declarations only: nothing in the
// engine pipeline calls this yet; product wiring is a follow-up owned by Mac-C.
// The module tree is `cfg(target_os = "macos")` so Windows builds are unchanged.
#[cfg(target_os = "macos")]
pub mod metal_runtime;
#[cfg(target_os = "macos")]
pub(crate) mod metal;
#[cfg(target_os = "macos")]
#[path = "mock/skin_whiten_metal.rs"]
pub mod skin_whiten_metal;

// Roadmap M4 (section 8.2): pure CPU/GPU cost model, EWMA/quantile calibration,
// hysteresis and failure classification. Model-only: no scheduling, no env, no IO;
// nothing in the pipeline calls this yet.
pub mod compute_cost;

pub fn prewarm_gpu() {
    #[cfg(windows)]
    if aurora_portrait::native_infer::backend::BackendRequest::from_env().is_ok_and(|r|r.permits(aurora_portrait::native_infer::backend::BackendKind::Cuda)){mock::prewarm_gpu();}
}

pub mod mask_store;
mod noise_defaults;
mod output;
mod body_shader_abi;
mod portrait_stage;
pub mod portrait_model_output_provider;
pub mod portrait_vendor_models;
pub mod portrait_vendor_config;
mod wrinkle_legacy;
pub mod portrait_makeup_bundle;
pub mod portrait_makeup_suit_inputs;
mod portrait_historical_tone_provider;
pub mod portrait_makeup_provider;
mod portrait_eye_bag_inputs;
mod portrait_makeup_mesh2d;
mod portrait_makeup_wiring;
mod portrait_makeup_recipes;
mod portrait_look_layer;
mod portrait_look_catchlight;
mod portrait_probe_inputs;
pub use portrait_makeup_recipes::{MakeupSuit, apply_makeup_suit, makeup_suits};
pub use portrait_makeup_wiring::{MakeupPresetInfo, makeup_custom_color_supported, makeup_palette_for_pf, makeup_presets_for_pf};
mod portrait_moisture_inputs;
pub mod portrait_pf115_pf107;
mod portrait_pf90021_input;
pub use portrait_stage::portrait_face_input_stage;
pub mod blemish_gainmap;
pub mod blemish_gainmap_cuda;
pub use blemish_gainmap::*;
pub use aurora_portrait::cuda_pf90021::{
    Direction as Pf90021Direction, RuntimeScalars as Pf90021RuntimeScalars, VolumeLut64,
};
pub use aurora_portrait::{Image as PortraitImage, Refusal as PortraitRefusal};
pub use portrait_historical_tone_provider::{
    build_complete_historical_tone_inputs, build_historical_tone_inputs,
    HistoricalToneBreastRepairPayload, HistoricalToneCollarbonePayload,
    HistoricalToneEyeRedVeinsPayload, HistoricalToneFlatGrayPayload,
    HistoricalToneGray3dPayload, HistoricalToneHairColorPayload,
    HistoricalToneHandGrayPayload, HistoricalToneHandShapePayload,
    HistoricalToneHandVeinsPayload, HistoricalToneMultiTonePayload,
    HistoricalToneProviderPayload, HistoricalToneProviderRow, HistoricalToneRowData,
    HistoricalToneRowJob, HISTORICAL_TONE_PROVIDER_PFS,
};
pub use body_shader_abi::{
    decode_body_geometry, encode_body_geometry, BodyKeypointSet, BodySixPointShaderInput,
    EncodedBodyGeometry, BODY_EBO_INDEX_BYTES, BODY_LANDMARK_COUNT, BODY_SIX_POINT_SLOTS,
    BODY_VBO_DESTINATION_XY_OFFSET, BODY_VBO_SOURCE_UV_OFFSET, BODY_VBO_STRIDE_BYTES,
};
pub use portrait_stage::{
    apply_historical_tone_row, gpu_kernel, gpu_lut_reason, plan_historical_skin_backend,
    plan_historical_tone_backend, HistoricalSkinGpuPlan, HistoricalSkinPath,
    HistoricalToneFixture, HistoricalToneGpu, HistoricalToneGpuPlan, HistoricalTonePath,
    HistoricalToneRowOutcome, GPU_MATERIALIZED_FIXTURE_ENV,
    GPU_PROVEN_HISTORICAL_SKIN_PFS, GPU_PROVEN_HISTORICAL_TONE_PFS,
};
pub(crate) use portrait_stage::sha256_32;
pub use portrait_stage::{
    BodyGeometryInput, BodyGeometryJob, BodyGeometryProducerInputs, BodyLandmarkDomain,
    BodyVendorParams,
    BodyLandmarkEffect, BodyLandmarkFrame, BodyLandmarkRef, BodyShiftInput, FaceGeometry,
    FaceGeometryDraw,
    FaceGeometryInputs, FaceGeometryProviderInputs, FaceGeometryProviderMesh,
    FaceGeometryProviderMeshReceipt, FaceGeometryProviderReceipt, FaceGeometryProviderRoute,
    FaceGeometryProviderRow, HistoricalSkinData, HistoricalSkinExternalInputs,
    HistoricalSkinExternalRow, HistoricalSkinImmediateInputs, HistoricalSkinImmediateRow,
    HistoricalSkinInput, HistoricalSkinInputs, HistoricalSkinPass, HistoricalSkinWrinkle,
    HistoricalToneData, HistoricalToneFace,
    HistoricalToneInput, HistoricalToneInputs,
    HlfBodyAdjustLutProducerInputs, Pf8050AttachReceipt, Pf8050FboReceipt, Pf8050GeometryInputs,
    Pf8050StageInputs, Pf8050VendorRuntime, Pf30025Inputs, Pf30025LiveModelOutputs,
    Pf30025ProducerInputs, Pf30025SameJobReceipt, Pf30025SourceSurface, Pf30026EdgeOutput,
    Pf30026GateBinding, Pf8069PositiveInputs, Pf90004ProducerInputs, Pf90021DirectionReceipt, Pf90021Inputs,
    Pf90021MaskSource, Pf90021ProducerInputs, Pf90021RuntimeReceipt, Pf90021SameJobBinding,
    FaceLandmarkProducerInputs, FACE_LANDMARK_52_MODEL_SHA256, FACE_LANDMARK_52_RESOURCE,
    FaceInputBundle, FaceInputModelPaths, FaceInputProducer, C1FaceDetection,
    C1FaceMesh, C1FaceParsing, C1_FACE_DETECTION_MODEL_SHA256, C1_FACE_DETECTION_MODEL_ID,
    C1_FACE_MESH_MODEL_SHA256, C1_FACE_MESH_MODEL_ID, C1_FACE_PARSING_MODEL_SHA256,
    C1_FACE_PARSING_MODEL_ID, FACE_LANDMARK_1063025060_MODEL_SHA256,
    FACE_LANDMARK_323773030_MODEL_SHA256, FACE_LANDMARK_CONSTANTS_SHA256,
    FaceInputFileStamp, FaceInputResultKey, FaceInputSession, FaceInputSessionKey,
    FaceModelIdentity, FaceNormalization, FaceStageTransform, FACE_STAGE_TRANSFORM_REFUSAL,
    EyeBagFrameProducer, Pf90116ColorDomain, Pf90116FrameGeometry,
    Pf90116FrameBinding, Pf90116FrameReceipt, Pf90116InputOrder, Pf90116TensorReceipts,
    PF90116_ACTIVE_GATE, PF90116_COLOR_NAME, PF90116_COLOR_SHAPE, PF90116_INPUT_MEAN,
    PF90116_INPUT_NAME, PF90116_INPUT_SCALE, PF90116_INPUT_SHAPE, PF90116_MODEL_HEIGHT,
    PF90116_MODEL_ID, PF90116_MODEL_NET_SHA256, PF90116_MODEL_PAYLOAD_SHA256,
    PF90116_MODEL_WIDTH, PF90116_RESPONSE_NAME, PF90116_RESPONSE_SHAPE, MakeupInput,
    MetalDispatch, PortraitBackend, PortraitEffectRun, PortraitRuntimeStatus, PortraitRunReport,
    PortraitSkipReason, bridged_effects, is_bridged_effect, supported_effects,
    PortraitFrameReceipt, PortraitGeometryRun, PortraitJobInputs, PortraitJobReceipt,
    PortraitMasks, ProducerInputs, PLUMP_PFS,
    PlumpCompositeReceipt, PlumpFinalComposite, PlumpProducerInputs, SmileAxisDecode, SmileAxisSign,
    SmileChannelSplit, SmileFieldChannel, SmileHostSplitSpec,
    SmileProducerInputs, SmileRgbReconstructionField, SmileSigmoidEdgeResponse,
    BeardInputs, HairGlossInputs, HairSmoothInputs, MoistureBodyInputs, MoistureFaceInputs,
    RepairModelPlanes, RepairOutput, RepairOutputReceipt, RepairPlaneSha256,
    RepairProducerInputs, SixPointWarp, SlimProducerInputs,
    StrayHairInputs, StrayHairProducerInputs, SuitPassInput, SuitPassKindInput,
    TeethAlignmentProducerInputs, TeethProducerInputs,
    apply_pf8050_pf8051_geometry,
    apply_pf8050_pf8051_stage,
    apply_pf8033_cpu_forward,
    build_historical_skin_inputs,
    apply_portrait_effects_with_job,
    apply_portrait_effects_with_report_and_inputs,
    build_face_geometry_inputs,
    build_pf8033_cpu_forward_row,
    full_face_descriptor_from_store, full_face_descriptor_from_ui,
    landmark_receipt_hashes,
    load_pf8033_cpu_forward_artifacts,
    mesh_receipt,
    pf8033_cpu_forward_cpu_oracle,
    provider_route,
    PROVIDER_ROUTED_PFS,
    Pf8033CpuForwardArtifacts,
    Pf8033CpuForwardReceipt,
    Pf8033CpuForwardRun,
    Pf8033HostSplitReceiptData,
    Pf8033ModelReceipt,
    Pf8033NetworkReceipt,
    Pf8033TensorReceipt,
    Pf8033WarpReceipt,
    PF8033_CPU_FORWARD_ARTIFACTS_ENV,
    PF8033_CPU_FORWARD_RECEIPT_ENV,
    PF8033_DX_FILE,
    PF8033_DY_FILE,
    PF8033_FINAL_MODEL_ID,
    PF8033_FINAL_TENSOR_FILE,
    PF8033_FINAL_TENSOR_ID,
    PF8033_FRAME_HEIGHT,
    PF8033_FRAME_WIDTH,
    PF8033_MASK_FILE,
    PF8033_MODEL_HEIGHT,
    PF8033_MODEL_IDS,
    PF8033_MODEL_ROOT_ENV,
    PF8033_MODEL_WIDTH,
};
pub use wrinkle_legacy::{
    WRINKLE_FACEPARSING_MASK_MAPPING_REFUSAL, WRINKLE_FACE_ROI_WRITER_RVA,
    WRINKLE_LANDMARK_194_COUNT, WRINKLE_LANDMARK_56_COUNT,
    WRINKLE_LEGACY_SKIN_MASK_FILTER_SLOT, WRINKLE_LEGACY_SKIN_MASK_SERVICE_OFFSET,
    WRINKLE_LEGACY_SKIN_MASK_TEXTURE_INDEX, WRINKLE_MASK_ROI_SIDE,
    WRINKLE_SHADER_UPLOAD_REFUSAL, WRINKLE_V1_PARAM_SLOTS, WRINKLE_V1_SHADER_SLOTS,
    WrinkleFaceRoiContract, WrinkleLegacySkinMaskProducer, WrinkleLegacySkinMaskTexture,
    WrinkleMaskMiniImage, WrinkleMaskMiniImageProducer, WrinkleProducerInputs, WrinkleSameJobReceipt,
    WrinkleShaderSlot, WrinkleV1ParamSlot, WrinkleV1ParamsResultReceipt,
    faceparsing_v2_wrinkle_masks_refusal, validate_wrinkle_uniforms,
};
pub use portrait_stage::{
    GlareChainFinalWritebackInputs, GlareFinalWritebackInputs, GlareFinalWritebackOutput,
    GlareInputs, GlarePlanarBundle, GlarePlanarInputs, GlarePlanarReceipts, GlareWritebackInputs,
    GlareWritebackOutput, PF90130_A_PLANAR_BYTES, PF90130_B_PLANAR_BYTES, PF90130_EFFECT_TENSOR_ID,
    PF90130_INPUT_TENSOR_ID, PF90130_MASK_TENSOR_ID, PF90130_MODEL_ID, PF90130_NORMALIZE_SCALE,
    PF90130_NORMALIZE_SUB, PF90130_PIXELS, PF90130_PLANE_HEIGHT, PF90130_PLANE_WIDTH,
    Pf101ProducerInputs, Pf90130AlignedGeometry, Pf90130GlareProducer, Pf90130ProducerInputs,
    Pf90130SameJobReceipt, Pf90130StageRun,
    SkinRepairData, SkinRepairOutput, SkinRepairProducerInputs, SpotlessGainmapJob, SpotlessInputs,
    apply_glare_chain_final_writeback, apply_glare_final_writeback, apply_glare_stage_writeback,
    apply_pf90130_producer_stage, bridge_glare_planes, run_skin_repair,
    resolve_spotless_gainmap_amount, run_spotless_gainmap_16u,
};
pub use portrait_makeup_bundle::*;
pub use portrait_makeup_provider::*;
mod param_meta;
mod preview_cache;
mod presets;
pub mod raw;
pub mod aurora_raw;
#[cfg(all(windows, feature = "z6iii-raw-export"))]
pub use aurora_raw::z6iii;
pub mod raw_cache;
mod rawmatch;
mod rawinfo;
mod wb_gains;
mod recipe;
/// Compiled execution view of a saved node graph.
pub mod render_plan;
mod save_queue;
pub use save_queue::*;
mod save_report;
pub use save_report::*;
mod scopes;
mod store;
mod styles;
mod testdata;
mod tether;
mod xmp;
pub mod camera_curve;
pub mod icc;
pub mod camera_profile;
pub mod ai_denoise;
pub mod aimask;
pub mod dustdetect;
pub mod beautymarks;
pub mod autorotate;
pub mod keystone;

// Unified compute capability/policy layer (roadmap M1). Pure addition: no
// product call site consults it yet; see the module docs for the phase-2 plan.
pub mod compute;

// Vendor (像素蛋糕) preset parameters, injected into this machine's Capture One
// install by third-party DLLs. Deliberately not re-exported at the crate root:
// these 482 keys are NOT Capture One's own adjustment model. Capture One's real
// per-variant parameter set is the 163-column `ZVARIANTLAYER` table.
mod effects;
mod portrait_map;
pub mod portrait {
    pub use crate::effects::*;
    pub use crate::portrait_map::*;
}

pub use aurora_store::{
    is_aurora_catalog_path, is_aurora_session_path, AuroraDb, DiskThumbCache, FavoriteFolder,
    LibraryKind, LibraryProfile, SessionFolderRole, AURORA_CATALOG_EXT, AURORA_META_DIR,
    AURORA_SESSION_EXT,
};
pub use adjust::*;
pub use c1catalog::{
    C1Catalog, C1Collection, C1DocumentType, C1Image, C1Keyword, C1Variant, WalReplay,
};
pub use c1import::{resolve_database, C1Import, C1_COLLECTION_ID_BASE};
pub use c1session_export::{export_session, SessionExport};
pub use c1_masks::*;
pub use c1_style_index::*;
pub use c1_styles::*;
pub use c1_model::*;
pub use camera_defaults::overrides_for;
pub use c1_banding::BandingRange;
pub use batch::*;
pub use catalog::*;
pub use exif::read_exif;
pub use closedeyes::{any_eye_closed, detect_eyes, EyeBox, EyeReport};
pub use engine::*;
pub use file_engine::{
    supports_photo_path, FileEngine, ImportErrorKind, QueuedDocumentResult, QueuedSaveOutcome, QueuedSaveSummary,
    SourceStatusKind,
};
pub use fcrv_index::*;
pub use filmcurve::{curve_dir, parse_film_curves, FilmCurve};
pub use lcp::*;
pub use layers::*;
pub use lens_correct::*;
pub use dust_patch::{DustDetector, DustError, DustPatch};
pub use dust_lcc::{DustLccError, LccLoadError};
pub use layer_graph::*;
pub use lensdb::*;
pub use matchlook::*;
pub use merge_async::{
    MergeError, MergeErrorCode, MergeJobId, MergeKind, MergeOutcome, MergeProgress, MergeRequest,
    MergeStage, SoftProofSettings,
};
pub use merge_options::*;
pub use mock::MockEngine;
pub use output::*;
pub use panel_commands::{
    CaptureDevice, PortraitCapability, PrintLayout, PrintPreview, PrintRequest,
};
#[doc(hidden)]
pub use mock::process::{render, render16, Rendered, Rendered16, Source};
pub use param_meta::*;
pub use presets::*;
pub use raw::*;
pub use rawinfo::{parse_info_block, read_nikon_info_block, CameraDefaults, RawInfoBlock};
pub use recipe::*;
pub use scopes::*;
pub use store::{Catalog, Entry, Store, StoreError, write_atomic};
pub use styles::*;
pub use tether::*;
pub use xmp::XmpMeta;

