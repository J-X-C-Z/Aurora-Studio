//! Aurora's own controls, painted to match the dark workspace.

pub mod controls;
pub mod curve;
pub mod histogram;
pub mod levels;
pub mod paint;
pub mod panel;
pub mod param_c1;
pub mod param_portrait;
pub mod slider;
pub mod surface;
pub mod text;
pub mod wheel;

pub use controls::{IconBtn, Segmented, select, switch_row};
pub use curve::CurveEditor;
pub use histogram::{HistogramMode, HistogramView};
pub use levels::LevelsView;
pub use panel::{HeaderAction, ToolPanel, caption, row};
pub use param_c1::param_c1;
pub use param_portrait::param_portrait;
pub use slider::{ParamSlider, Scale, Track, param};
pub use text::{field_state, text_field};
pub use wheel::ColorWheelView;
