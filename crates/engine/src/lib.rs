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

mod adjust;
mod c1catalog;
mod c1import;
mod c1_masks;
pub mod c1_cos;
pub mod c1_eip;
pub mod c1_spots;
pub mod dust_lcc;
pub mod dust_patch;
pub mod retouch_c1;
mod c1session_export;
mod sqlite_write;
mod c1_style_index;
mod c1_styles;
mod c1_model;
pub mod c1_nef_lensdata;
mod c1_wb_oracle;
pub mod c1_retouch;
pub mod hairfly_selector;
pub mod comask;
mod camera_defaults;
pub mod camera_params;
mod ca_fix;
mod batch;
mod catalog;
mod closedeyes;
mod engine;
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
mod layers;
mod lens_correct;
mod layer_graph;
mod lensdb;
mod matchlook;
mod merge_options;
mod mock;

pub fn prewarm_gpu() {
    #[cfg(windows)]
    mock::prewarm_gpu();
}

pub mod mask_store;
mod noise_defaults;
mod output;
mod portrait_stage;
mod wrinkle_legacy;
pub mod portrait_makeup_bundle;
pub use aurora_portrait::cuda_pf90021::{
    Direction as Pf90021Direction, RuntimeScalars as Pf90021RuntimeScalars, VolumeLut64,
};
pub use aurora_portrait::{Image as PortraitImage, Refusal as PortraitRefusal};
pub use portrait_stage::{
    BodyGeometryInput, BodyGeometryJob, BodyShiftInput, FaceGeometry, FaceGeometryDraw,
    FaceGeometryInputs, HistoricalSkinData, HistoricalSkinInput, HistoricalSkinInputs,
    HistoricalSkinPass, HistoricalSkinWrinkle, HistoricalToneData, HistoricalToneFace,
    HistoricalToneInput, HistoricalToneInputs,
    HlfBodyAdjustLutProducerInputs, Pf8050AttachReceipt, Pf8050FboReceipt, Pf8050GeometryInputs,
    Pf8050StageInputs, Pf8050VendorRuntime, Pf30025Inputs, Pf30025LiveModelOutputs,
    Pf30025ProducerInputs, Pf30025SameJobReceipt, Pf30025SourceSurface, Pf30026EdgeOutput,
    Pf30026GateBinding, Pf90004ProducerInputs, Pf90021DirectionReceipt, Pf90021Inputs,
    Pf90021MaskSource, Pf90021ProducerInputs, Pf90021RuntimeReceipt, Pf90021SameJobBinding,
    MakeupInput, PortraitBackend, PortraitEffectRun, PortraitFrameReceipt, PortraitGeometryRun,
    PortraitJobInputs, PortraitJobReceipt, PortraitMasks, ProducerInputs, PLUMP_PFS,
    PlumpCompositeReceipt, PlumpFinalComposite, PlumpProducerInputs, SmileAxisDecode, SmileAxisSign,
    SmileChannelSplit, SmileFieldChannel, SmileHostSplitSpec,
    SmileProducerInputs, SmileRgbReconstructionField, SmileSigmoidEdgeResponse,
    RepairOutput, RepairProducerInputs, SixPointWarp, SlimProducerInputs,
    StrayHairProducerInputs, SuitPassInput, SuitPassKindInput, TeethAlignmentProducerInputs,
    TeethProducerInputs,
    apply_pf8050_pf8051_geometry,
    apply_pf8050_pf8051_stage,
    apply_portrait_effects_with_job,
    apply_portrait_effects_with_report_and_inputs,
    full_face_descriptor_from_ui,
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
    Pf90130AlignedGeometry, Pf90130GlareProducer, Pf90130SameJobReceipt, Pf90130StageRun,
    SkinRepairData, SkinRepairOutput, SkinRepairProducerInputs, SpotlessInputs,
    apply_glare_chain_final_writeback, apply_glare_final_writeback, apply_glare_stage_writeback,
    apply_pf90130_producer_stage, bridge_glare_planes, run_skin_repair,
};
pub use portrait_makeup_bundle::*;
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
mod scopes;
mod store;
mod styles;
mod tether;
mod xmp;
pub mod camera_curve;
pub mod icc;
pub mod camera_profile;
pub mod aimask;
pub mod dustdetect;
pub mod autorotate;
pub mod keystone;

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
pub use batch::*;
pub use catalog::*;
pub use exif::read_exif;
pub use closedeyes::{any_eye_closed, detect_eyes, EyeBox, EyeReport};
pub use engine::*;
pub use file_engine::{FileEngine, ImportErrorKind, SourceStatusKind};
pub use fcrv_index::*;
pub use filmcurve::{curve_dir, parse_film_curves, FilmCurve};
pub use layers::*;
pub use lens_correct::*;
pub use dust_patch::{DustDetector, DustError, DustPatch};
pub use dust_lcc::DustLccError;
pub use layer_graph::*;
pub use lensdb::*;
pub use matchlook::*;
pub use merge_options::*;
pub use mock::MockEngine;
pub use output::*;
#[doc(hidden)]
pub use mock::process::{render, render16, Rendered, Rendered16, Source};
pub use param_meta::*;
pub use presets::*;
pub use raw::*;
pub use rawinfo::{parse_info_block, read_nikon_info_block, CameraDefaults, RawInfoBlock};
pub use recipe::*;
pub use scopes::*;
pub use store::{Catalog, Entry, Store, StoreError};
pub use styles::*;
pub use tether::*;
pub use xmp::XmpMeta;

