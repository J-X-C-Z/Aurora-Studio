//! PF90021 `LipStickLutMakeup` mask consumer: recovered `LipMaskType` table and
//! channel sampling.
//!
//! Evidence
//! --------
//! * `pixcake.exe` (SHA-256
//!   `A84519DA489BC52E4E914AF7880EDE43E3CE58992657723D02197161832464C7`):
//!   `LipMaskType` string at RVA `0x97D9A11`; the constructor `FUN_14614B040`
//!   (RVA `0x614B040`) reads `effect+0x288` (= `LipMaskType`), indexes
//!   `0x92CBB18[value-1]` for `value <= 17`, and stores the result into
//!   `filter+0x2B4` through `FUN_146124370`. Table bytes: 17 little-endian
//!   `u32` `[0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13]`; out of range -> `-1`.
//! * The selected channel is then declared as a `MASK_*` macro. `MASK_R` for
//!   table value 0 is the binding already closed by the s943/s951 lane
//!   (`filter+0x108` bound at unit `0x84C2`, uniform `h`, fragment samples `.r`).
//!   Decoding values 1..13 through the vendor `GetMaskWeight` `#elif` order in
//!   `glsl/Sticker/StickerWithMask.frag` is `INFERRED`; only value 0 (`MASK_R`)
//!   is directly pinned.
//!
//! Scope
//! -----
//! This is a CPU reference consumer, not a producer. It takes one same-frame,
//! already-parsed mask plane with explicit width/height/channel semantics and
//! reproduces the proven shader law: select the `LipMaskType` channel
//! combination and `clamp(sum)` it to `[0,1]` (the `GetMaskWeight` body).
//!
//! Feather and threshold are **not recovered** for PF90021. Neither
//! `LipColorNormal.frag` / `LipMultiColors.frag` / `StickerWithMask.frag`
//! contains a `feather`/radius uniform nor a `smoothstep` on the mask; the only
//! mask gates found are `mask_color.r > 0.0`, `factor > 0.01` and
//! `u_lip_color.a > 0.001`, and texture filtering is `GL_LINEAR` /
//! `GL_CLAMP_TO_EDGE` (see the caller). These two values are therefore explicit
//! caller inputs. A missing value is a named [`Refusal::MissingInput`]; this
//! module never substitutes a default and never invents a feather law.

use crate::kernels::makeup_material::MaskChannel;
use crate::Refusal;

/// `pixcake.exe` RVA of the recovered `LipMaskType` lookup table (17 x u32).
pub const LIP_MASK_TYPE_TABLE_RVA: usize = 0x92CBB18;

/// Recovered table bytes indexed as `table[lip_mask_type - 1]`, for
/// `lip_mask_type` in `1..=17`. `0` and `> 17` are out of range and map to
/// `-1` in the vendor constructor.
pub const LIP_MASK_TYPE_TABLE: [u8; 17] = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];

/// `LipMaskType` outside `1..=17` (vendor default `-1`).
pub const LIP_MASK_TYPE: &str = "PF90021_LIP_MASK_TYPE";
/// The mask plane has zero dimensions, an unsupported channel count or a data
/// length that does not match `width * height * channels * element_size`.
pub const LIP_MASK_PLANE: &str = "PF90021_MASK_PLANE";
/// The plane does not carry every channel the selected combination reads.
pub const LIP_MASK_PLANE_CHANNELS: &str = "PF90021_MASK_PLANE_CHANNELS";
/// The PF90021 feather/soft-edge parameter was not supplied.
pub const LIP_MASK_FEATHER: &str = "PF90021_FEATHER";
/// The PF90021 mask threshold parameter was not supplied.
pub const LIP_MASK_THRESHOLD: &str = "PF90021_MASK_THRESHOLD";

/// Table-value -> channel decode. Table values `0..=13` are PROVEN bytes; the
/// meaning of `0` as `MASK_R` is PROVEN. The 14-entry order below mirrors the
/// existing main-tree `makeup_material::lipstick_mask_channel`. The vendor
/// `GetMaskWeight` also has an `MASK_RBA` branch, so this branch list excludes
/// `Rba`; whether the vendor table tails at `GBA` or `RGBA` is `OPEN`
/// (see REPORT). Only the single-channel prefix and `0 -> R` are used as proven.
const CHANNEL_ORDER: [MaskChannel; 14] = [
    MaskChannel::R,
    MaskChannel::G,
    MaskChannel::B,
    MaskChannel::A,
    MaskChannel::RG,
    MaskChannel::RB,
    MaskChannel::RA,
    MaskChannel::GB,
    MaskChannel::GA,
    MaskChannel::BA,
    MaskChannel::Rgb,
    MaskChannel::Rga,
    MaskChannel::Gba,
    MaskChannel::Rgba,
];

/// Maps a vendor `LipMaskType` to the mask channel combination sampled by the
/// fragment stage.
///
/// Only values with recovered table bytes are admitted. `0` and values above
/// `17` are the vendor's `-1` default and are refused as
/// [`Refusal::ValueOutOfRange`].
pub fn lip_mask_channel(lip_mask_type: u32) -> Result<MaskChannel, Refusal> {
    let index = lip_mask_type
        .checked_sub(1)
        .filter(|index| *index < LIP_MASK_TYPE_TABLE.len() as u32)
        .ok_or(Refusal::ValueOutOfRange(LIP_MASK_TYPE))? as usize;
    let table_value = LIP_MASK_TYPE_TABLE[index] as usize;
    CHANNEL_ORDER
        .get(table_value)
        .copied()
        .ok_or(Refusal::ValueOutOfRange(LIP_MASK_TYPE))
}

/// One same-frame parsed mask plane.
///
/// Channels are in `R, G, B, A` order; `1..=4` channels are accepted. `u8`
/// values are normalized by `/ 255` (the vendor texture is `GL_RGBA8` and the
/// shader reads the normalized sampled value).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaskPlaneData<'a> {
    U8(&'a [u8]),
    F32(&'a [f32]),
}

/// A parsed mask plane plus its shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskPlane<'a> {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub data: MaskPlaneData<'a>,
}

impl<'a> MaskPlane<'a> {
    pub fn u8(width: u32, height: u32, channels: u8, data: &'a [u8]) -> Self {
        Self { width, height, channels, data: MaskPlaneData::U8(data) }
    }

    pub fn f32(width: u32, height: u32, channels: u8, data: &'a [f32]) -> Self {
        Self { width: width, height: height, channels, data: MaskPlaneData::F32(data) }
    }

    fn validate(&self) -> Result<(), Refusal> {
        if self.width == 0 || self.height == 0 || !(1..=4).contains(&self.channels) {
            return Err(Refusal::ShapeMismatch(LIP_MASK_PLANE));
        }
        let pixels = self.width as usize * self.height as usize * self.channels as usize;
        let len = match self.data {
            MaskPlaneData::U8(data) => data.len(),
            MaskPlaneData::F32(data) => data.len(),
        };
        if len != pixels {
            return Err(Refusal::ShapeMismatch(LIP_MASK_PLANE));
        }
        if let MaskPlaneData::F32(data) = self.data {
            if data.iter().any(|value| !value.is_finite()) {
                return Err(Refusal::ValueOutOfRange(LIP_MASK_PLANE));
            }
        }
        Ok(())
    }

    fn require_channels(&self, channel: MaskChannel) -> Result<(), Refusal> {
        let needed = highest_channel(channel) + 1;
        if self.channels as usize >= needed {
            Ok(())
        } else {
            Err(Refusal::ShapeMismatch(LIP_MASK_PLANE_CHANNELS))
        }
    }

    fn pixel(&self, index: usize) -> [f32; 4] {
        let channels = self.channels as usize;
        let mut out = [0.0f32; 4];
        for (channel, slot) in out.iter_mut().take(channels).enumerate() {
            let offset = index * channels + channel;
            *slot = match self.data {
                MaskPlaneData::U8(data) => data[offset] as f32 / 255.0,
                MaskPlaneData::F32(data) => data[offset],
            };
        }
        out
    }
}

fn highest_channel(channel: MaskChannel) -> usize {
    match channel {
        MaskChannel::R => 0,
        MaskChannel::G | MaskChannel::RG => 1,
        MaskChannel::B | MaskChannel::RB | MaskChannel::GB | MaskChannel::Rgb => 2,
        MaskChannel::A
        | MaskChannel::RA
        | MaskChannel::GA
        | MaskChannel::BA
        | MaskChannel::Rga
        | MaskChannel::Rba
        | MaskChannel::Gba
        | MaskChannel::Rgba => 3,
    }
}

/// `GetMaskWeight` body from the vendor fragment shaders: sum the selected
/// channels and clamp to `[0,1]`.
fn combine_channels(pixel: [f32; 4], channel: MaskChannel) -> f32 {
    let value = match channel {
        MaskChannel::R => pixel[0],
        MaskChannel::G => pixel[1],
        MaskChannel::B => pixel[2],
        MaskChannel::A => pixel[3],
        MaskChannel::RG => pixel[0] + pixel[1],
        MaskChannel::RB => pixel[0] + pixel[2],
        MaskChannel::RA => pixel[0] + pixel[3],
        MaskChannel::GB => pixel[1] + pixel[2],
        MaskChannel::GA => pixel[1] + pixel[3],
        MaskChannel::BA => pixel[2] + pixel[3],
        MaskChannel::Rgb => pixel[0] + pixel[1] + pixel[2],
        MaskChannel::Rga => pixel[0] + pixel[1] + pixel[3],
        MaskChannel::Rba => pixel[0] + pixel[2] + pixel[3],
        MaskChannel::Gba => pixel[1] + pixel[2] + pixel[3],
        MaskChannel::Rgba => pixel[0] + pixel[1] + pixel[2] + pixel[3],
    };
    value.clamp(0.0, 1.0)
}

/// Samples the recovered mask weight plane (one `f32` per pixel, top row
/// first).
///
/// This applies only the proven law: `LipMaskType` -> channel combination ->
/// `clamp(sum)`. No feather, threshold, UV warp or ROI is applied here; those
/// are not part of the recovered expression.
pub fn sample_lip_mask(plane: &MaskPlane, lip_mask_type: u32) -> Result<Vec<f32>, Refusal> {
    let channel = lip_mask_channel(lip_mask_type)?;
    plane.validate()?;
    plane.require_channels(channel)?;
    let count = plane.width as usize * plane.height as usize;
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        out.push(combine_channels(plane.pixel(index), channel));
    }
    Ok(out)
}

/// Explicit caller-owned runtime values for a PF90021 application.
///
/// Both fields are `Option` on purpose: the vendor feather/threshold law is
/// `NOT_FOUND` in the recovered shader and host, so there is no default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LipMaskRuntime {
    pub feather: Option<f32>,
    pub threshold: Option<f32>,
}

/// The sampled mask plane plus the caller's declared runtime values.
#[derive(Clone, Debug, PartialEq)]
pub struct LipMaskWeights {
    pub channel: MaskChannel,
    pub weights: Vec<f32>,
    pub feather: f32,
    pub threshold: f32,
}

/// Samples the mask and requires the caller's feather and threshold.
///
/// Missing fields are named refusals ([`LIP_MASK_FEATHER`],
/// [`LIP_MASK_THRESHOLD`]). Present fields are validated but **not applied**:
/// the vendor feather convolution and the threshold comparison are not
/// recovered for PF90021, so this kernel returns the proven channel-combined
/// plane and carries the values so downstream code cannot silently default
/// them. A caller that needs the unrecovered law must implement it explicitly.
pub fn lip_mask_weights(
    plane: &MaskPlane,
    lip_mask_type: u32,
    runtime: LipMaskRuntime,
) -> Result<LipMaskWeights, Refusal> {
    let channel = lip_mask_channel(lip_mask_type)?;
    let feather = runtime.feather.ok_or(Refusal::MissingInput(LIP_MASK_FEATHER))?;
    if !feather.is_finite() || feather < 0.0 {
        return Err(Refusal::ValueOutOfRange(LIP_MASK_FEATHER));
    }
    let threshold = runtime.threshold.ok_or(Refusal::MissingInput(LIP_MASK_THRESHOLD))?;
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(Refusal::ValueOutOfRange(LIP_MASK_THRESHOLD));
    }
    let weights = sample_lip_mask(plane, lip_mask_type)?;
    Ok(LipMaskWeights { channel, weights, feather, threshold })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba_plane() -> [u8; 8] {
        // Two pixels: [10,20,30,40] and [200,100,50,255].
        [10, 20, 30, 40, 200, 100, 50, 255]
    }

    #[test]
    fn lip_mask_type_3_selects_mask_r() {
        assert_eq!(lip_mask_channel(3), Ok(MaskChannel::R));
        // The recovered table maps 1..=4 to the same value 0.
        for lip_mask_type in 1..=4 {
            assert_eq!(lip_mask_channel(lip_mask_type), Ok(MaskChannel::R));
        }
        assert_eq!(lip_mask_channel(5), Ok(MaskChannel::G));
        assert_eq!(lip_mask_channel(8), Ok(MaskChannel::RG));
        assert_eq!(lip_mask_channel(17), Ok(MaskChannel::Rgba));
    }

    #[test]
    fn lip_mask_type_out_of_range_refused() {
        assert_eq!(
            lip_mask_channel(0),
            Err(Refusal::ValueOutOfRange(LIP_MASK_TYPE))
        );
        assert_eq!(
            lip_mask_channel(18),
            Err(Refusal::ValueOutOfRange(LIP_MASK_TYPE))
        );
        assert_eq!(
            lip_mask_channel(u32::MAX),
            Err(Refusal::ValueOutOfRange(LIP_MASK_TYPE))
        );
    }

    #[test]
    fn samples_the_selected_channel() {
        let data = rgba_plane();
        let plane = MaskPlane::u8(2, 1, 4, &data);
        let weights = sample_lip_mask(&plane, 3).unwrap();
        assert_eq!(weights, vec![10.0 / 255.0, 200.0 / 255.0]);
    }

    #[test]
    fn samples_a_channel_sum() {
        let data = rgba_plane();
        let plane = MaskPlane::u8(2, 1, 4, &data);
        let weights = sample_lip_mask(&plane, 8).unwrap();
        assert!((weights[0] - 30.0 / 255.0).abs() < 1e-6);
        assert_eq!(weights[1], 1.0); // 200 + 100 saturates at 1.0
    }

    #[test]
    fn f32_plane_is_normalized_by_the_caller() {
        let data = [0.25f32, 0.0, 0.0, 1.0];
        let plane = MaskPlane::f32(1, 1, 4, &data);
        assert_eq!(sample_lip_mask(&plane, 3).unwrap(), vec![0.25f32]);
    }

    #[test]
    fn length_mismatch_refused() {
        let data = [1u8, 2, 3];
        let plane = MaskPlane::u8(2, 1, 4, &data);
        assert_eq!(
            sample_lip_mask(&plane, 3),
            Err(Refusal::ShapeMismatch(LIP_MASK_PLANE))
        );
    }

    #[test]
    fn channel_mismatch_refused() {
        let data = [7u8, 9];
        let plane = MaskPlane::u8(2, 1, 1, &data);
        // Type 7 reads the A channel, which a one-channel plane does not carry.
        assert_eq!(
            sample_lip_mask(&plane, 7),
            Err(Refusal::ShapeMismatch(LIP_MASK_PLANE_CHANNELS))
        );
    }

    #[test]
    fn non_finite_float_refused() {
        let data = [f32::NAN, 0.0, 0.0, 1.0];
        let plane = MaskPlane::f32(1, 1, 4, &data);
        assert_eq!(
            sample_lip_mask(&plane, 3),
            Err(Refusal::ValueOutOfRange(LIP_MASK_PLANE))
        );
    }

    #[test]
    fn missing_feather_refused() {
        let data = rgba_plane();
        let plane = MaskPlane::u8(2, 1, 4, &data);
        let runtime = LipMaskRuntime { feather: None, threshold: Some(0.0) };
        assert_eq!(
            lip_mask_weights(&plane, 3, runtime),
            Err(Refusal::MissingInput(LIP_MASK_FEATHER))
        );
    }

    #[test]
    fn missing_threshold_refused() {
        let data = rgba_plane();
        let plane = MaskPlane::u8(2, 1, 4, &data);
        let runtime = LipMaskRuntime { feather: Some(0.0), threshold: None };
        assert_eq!(
            lip_mask_weights(&plane, 3, runtime),
            Err(Refusal::MissingInput(LIP_MASK_THRESHOLD))
        );
    }

    #[test]
    fn explicit_runtime_returns_proven_weights() {
        let data = rgba_plane();
        let plane = MaskPlane::u8(2, 1, 4, &data);
        let runtime = LipMaskRuntime { feather: Some(1.5), threshold: Some(0.25) };
        let out = lip_mask_weights(&plane, 3, runtime).unwrap();
        assert_eq!(out.channel, MaskChannel::R);
        assert_eq!(out.weights, vec![10.0 / 255.0, 200.0 / 255.0]);
        assert_eq!(out.feather, 1.5);
        assert_eq!(out.threshold, 0.25);
    }

    #[test]
    fn invalid_runtime_refused() {
        let data = rgba_plane();
        let plane = MaskPlane::u8(2, 1, 4, &data);
        assert_eq!(
            lip_mask_weights(
                &plane,
                3,
                LipMaskRuntime { feather: Some(-1.0), threshold: Some(0.0) }
            ),
            Err(Refusal::ValueOutOfRange(LIP_MASK_FEATHER))
        );
        assert_eq!(
            lip_mask_weights(
                &plane,
                3,
                LipMaskRuntime { feather: Some(0.0), threshold: Some(1.5) }
            ),
            Err(Refusal::ValueOutOfRange(LIP_MASK_THRESHOLD))
        );
    }
}
