//! Runs `tests/cuda_teeth.rs` before `lib.rs` mounts `cuda_teeth`.
//!
//!   rustc --test --edition 2021 -O work\backend-r11\o1i\check.rs -o work\backend-r11\o1i\check.exe

extern crate self as aurora_portrait;

#[path = "../../../crates/portrait/src/lib.rs"]
mod pc;

pub use pc::*;

#[path = "../../../crates/portrait/src/cuda_teeth.rs"]
mod cuda_teeth;

include!("../../../crates/portrait/tests/cuda_teeth.rs");
