//! # lumen-engine
//!
//! The contract between the Lumen Studio interface and its image-processing engine.
//!
//! * [`AdjustmentSet`] is the full, serializable state of one variant's edits.
//! * [`ImageEngine`] is what an engine implements; the interface calls nothing else.
//! * [`FileEngine`] reads a real folder of images and persists edits in `.lumen`.
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
mod c1_style_index;
mod c1_styles;
mod c1_model;
mod camera_defaults;
mod ca_fix;
mod batch;
mod catalog;
mod closedeyes;
mod engine;
mod exif;
mod file_engine;
pub mod fcrv_index;
mod filmcurve;
mod heif;
mod heif_write;
mod highlight;
mod iptc;
mod layers;
mod lens_correct;
mod layer_graph;
mod lensdb;
mod matchlook;
mod mock;
pub mod mask_store;
mod noise_defaults;
mod output;
mod param_meta;
mod preview_cache;
mod presets;
pub mod raw;
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

// PixCake (像素蛋糕) preset parameters, injected into this machine's Capture One
// install by third-party DLLs. Deliberately not re-exported at the crate root:
// these 482 keys are NOT Capture One's own adjustment model. Capture One's real
// per-variant parameter set is the 163-column `ZVARIANTLAYER` table.
mod effects;
mod pixcake_map;
pub mod pixcake {
    pub use crate::effects::*;
    pub use crate::pixcake_map::*;
}

pub use adjust::*;
pub use c1catalog::{
    C1Catalog, C1Collection, C1DocumentType, C1Image, C1Keyword, C1Variant, WalReplay,
};
pub use c1import::{resolve_database, C1Import, C1_COLLECTION_ID_BASE};
pub use c1_masks::*;
pub use c1_style_index::*;
pub use c1_styles::*;
pub use c1_model::*;
pub use camera_defaults::overrides_for;
pub use batch::*;
pub use catalog::*;
pub use closedeyes::{any_eye_closed, detect_eyes, EyeBox, EyeReport};
pub use engine::*;
pub use file_engine::FileEngine;
pub use fcrv_index::*;
pub use filmcurve::{curve_dir, parse_film_curves, FilmCurve};
pub use layers::*;
pub use lens_correct::*;
pub use layer_graph::*;
pub use lensdb::*;
pub use matchlook::*;
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

