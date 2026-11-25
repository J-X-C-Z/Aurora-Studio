//! Reusable pixel-processing modules.
//!
//! Stages that are pure or depend only on adjustment data live here, outside
//! `mock`. `mock` keeps compatibility re-exports so existing call sites keep
//! compiling while the production renderer is migrated off the mock module in
//! small, byte-identical steps.

pub mod clarity;
pub use clarity::*;
pub mod color;
pub use color::*;
pub mod encoding;
pub mod frame_map;
pub mod geometry;
pub use geometry::*;
pub mod grain;
pub use grain::*;
pub(crate) mod layer_weight;
pub(crate) mod retouch_stage;
pub(crate) mod sizing;
pub mod spots;
pub use spots::*;
pub mod statistics;
pub mod tone;
