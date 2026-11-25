//! Same-frame producer inputs for the long-tail portrait rows.
//!
//! This module is the engine-side `ProducerInputs` wire for the rows whose
//! producer math is recovered but whose data has no place in the existing
//! bundle:
//!
//! * PF90034 / PF30188 / PF30189 double chin — per-face binary32 weights
//!   `W_geometry = f32(PF30188 * PF90034)`, `W_shadow = f32(PF90034 * PF30189)`,
//!   gate `1e-4` (task05 FINDINGS §4.2/§4.3, worker `0x7AEEA97`/`0x7AEEA9C`).
//!   The canonical weights and gate live in
//!   [`aurora_portrait::kernels::double_chin`]; this module carries the
//!   same-frame per-face packet and delegates the arithmetic there.
//! * PF90085 skin-tone unify face — per-face degree
//!   `piecewiseRemap(v,[0,4.5,5,8,10],[0,0,0.1,1,1]) * PF90085`, gate `0.001`
//!   (task06 FINDINGS §4.3, `.rdata` `0x930C7B0`/`0x930C79C`). The remap and
//!   the degree law are canonical in [`aurora_portrait::kernels::tone_unify`].
//! * PF44124 / PF48019 / PF49221 / PF41922 brow head/tail length — the vendor
//!   `FaceSurgery/V2/effect.json` two-part control tables and the recovered
//!   three-node remap (task08 FINDINGS §2/§4.4). The slot law is canonical in
//!   [`aurora_portrait::kernels::eye_geometry::brow_length_slot_degree`].
//!
//! Nothing here synthesises a value: a missing per-face node is the vendor's
//! own `0.0f` (`0x7AEE963..0x7AEE987`), and every other absent input is a
//! typed [`Refusal`]. The pixel consumers live in `aurora_portrait`; this
//! module only carries and validates the inputs they need, so a kernel that is
//! not wired to a stage route yet is reported by name instead of imitated.
//!
//! Still refused by name (no local model / mask binding):
//! * PF90087 `hairPartFill` — `HairSeam: hairMask` binding is NOT-FOUND in the
//!   fixed PE and no local model exists (task06 FINDINGS §2.4). The per-part
//!   degree law `d_i > 0.001` is canonical in
//!   [`aurora_portrait::kernels::face_detail::hair_seam_active_parts`].
//! * PF100005 `eyeWhitePurify` — model `3071265413 -> 1527454531 -> 2294125226`
//!   and its 8-plane float32 output are not available locally; the model-to-
//!   layer RGB/mask binding is open (task07 FINDINGS, face_detail row).
//! * PF30209 `jawlineEnhance` — stage `+0x3A0/+0x3B8` is not forwarded by the
//!   fixed PE and the ordered jawline index table is not a PE constant
//!   (task05 FINDINGS §5); the 33-point window `L[109..=141]` + center
//!   `L[54]` producer is canonical in
//!   [`aurora_portrait::kernels::remaining_special_provider::jawline_window`].

use aurora_portrait::kernels::double_chin;
use aurora_portrait::kernels::eye_geometry;
use aurora_portrait::kernels::tone_unify;
use aurora_portrait::Refusal;

/// Vendor double-chin execution gate; canonical value lives in
/// [`double_chin::WEIGHT_GATE`] (`0x8BFC350`, compared as binary64 at
/// `0x7E47324`/`0x7E47331`).
pub use double_chin::WEIGHT_GATE as DOUBLE_CHIN_WEIGHT_GATE;

/// PF90085 / PF30209-style support gate (`0.001f`, worker `0x8BEE250`).
pub const PF90085_DEGREE_GATE: f32 = 0.001;

/// PF90085 producer metric knots (`.rdata` `0x930C7B0`, task06 §4.3).
/// Canonical in [`tone_unify::PF90085_METRIC_KNOTS`].
pub use tone_unify::PF90085_METRIC_KNOTS;
/// PF90085 producer metric values (`.rdata` `0x930C79C`, task06 §4.3).
/// Canonical in [`tone_unify::PF90085_METRIC_VALUES`].
pub use tone_unify::PF90085_METRIC_VALUES;

/// Brow length `RenderParamRanges` (task08 §2, PROVEN).
pub use eye_geometry::BROW_LENGTH_PARAM_RANGE;
/// Short-part `RenderDegreeRanges`, indexed by `ControlIndex` 0..=3 (task08 §2).
pub use eye_geometry::BROW_LENGTH_SHORT_DEGREES;
/// Long-part `RenderDegreeRanges`, indexed by `ControlIndex` 0..=3 (task08 §2).
pub use eye_geometry::BROW_LENGTH_LONG_DEGREES;
/// Long-part head `DegreeFactor` (task08 §2); every other slot is `1.0`.
pub use eye_geometry::BROW_HEAD_LONG_DEGREE_FACTOR;
/// `BROW_LENGTH_AMOUNT` (`0x9059804 = 0.8`). The kernel keeps its own private
/// copy; the wire exposes the constant for the consumer handoff.
pub const BROW_LENGTH_AMOUNT: f32 = 0.8000000119209290;

/// PF90085 named producer gap: the degree law is recovered, the palette chain
/// is not. The four still-missing pixel inputs are canonical in
/// [`tone_unify::PF90085_MISSING_PRODUCER_INPUTS`]. Consumed by
/// `portrait_face_stage::producer_gap`.
pub const PF90085_PRODUCER_GAP: &str = "PF90085 per-face metric v + FaceParsingV1 skin image + FaceParsingV2 planes + landmarks + ref_img + 74-sample HSL controls + GammaMap + sRGB/ProPhoto matrices (degree law recovered; pixel producer absent)";
/// PF90087 named producer gap: the per-part degree map law is recovered
/// (`face_detail::hair_seam_active_parts`), the mask binding and a local
/// HairSeam model are not.
pub const PF90087_PRODUCER_GAP: &str = "PF90087 HairSeamV0 hairMask channel/size/binding NOT-FOUND in the fixed PE; per-part degree map d_i>0.001 recovered; no local HairSeam model";
/// PF100005 named producer gap.
pub const PF100005_PRODUCER_GAP: &str = "PF100005 eye-white model 3071265413/1527454531/2294125226 and 8-plane float32 output absent locally; 4+4 split and mixer tail known, model-to-layer RGB/mask binding open";
/// PF30209 named producer gap; the `+0x3A0/+0x3B8` non-forwarding statement is
/// canonical in
/// [`aurora_portrait::kernels::remaining_special_provider::PF30209_END_STATIC`]
/// and the recovered window in `jawline_window`.
pub const PF30209_PRODUCER_GAP: &str = "PF30209 stage +0x3A0/+0x3B8 is not forwarded by the fixed PE; ordered jawline index table is not a PE constant; 33-point window L[109..=141] + center L[54] recovered";
/// PF30188/PF30189 named producer gap: the weights ride the canonical
/// [`double_chin`] producer; the model field / shadow composite stays open
/// (mirrors `remaining_special_provider::CONTRACTS` rows 30188/30189).
pub const PF30188_PRODUCER_GAP: &str = "PF30188/PF30189 double-chin weights f32(PF90034*PF30188|PF30189) with 1e-4 gate recovered; same-job per-face packet and the final model field/shadow composite are open";

/// One vendor per-face `std::map<face_index, node(+0x20 float)>` snapshot.
///
/// Key order is irrelevant; values are the raw binary32 the native parameter
/// table held. A face that is absent from the map yields `0.0f`, which is what
/// `0x7AEE963..0x7AEE987` writes for a missing node.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FaceDegreeMap {
    entries: Vec<(u32, f32)>,
}

impl FaceDegreeMap {
    pub fn new(mut entries: Vec<(u32, f32)>) -> Result<Self, Refusal> {
        if entries.iter().any(|(_, value)| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("longtail per-face degree"));
        }
        entries.sort_by_key(|(face, _)| *face);
        if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(Refusal::ShapeMismatch("longtail per-face degree face id"));
        }
        Ok(Self { entries })
    }

    pub fn value(&self, face: u32) -> Option<f32> {
        self.entries
            .iter()
            .find(|(id, _)| *id == face)
            .map(|(_, value)| *value)
    }

    /// Vendor missing-node semantics: an absent face reads `0.0f`.
    pub fn value_or_zero(&self, face: u32) -> f32 {
        self.value(face).unwrap_or(0.0)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[(u32, f32)] {
        &self.entries
    }

    pub fn faces(&self) -> impl Iterator<Item = u32> + '_ {
        self.entries.iter().map(|(face, _)| *face)
    }
}

fn union_faces(maps: &[&FaceDegreeMap]) -> Vec<u32> {
    let mut faces: Vec<u32> = maps.iter().flat_map(|map| map.faces()).collect();
    faces.sort_unstable();
    faces.dedup();
    faces
}

/// One face's recovered double-chin weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoubleChinFaceWeights {
    pub face_id: u32,
    /// `f32(PF30188 * PF90034)`, raw binary32, no clamp or normalization.
    pub geometry: f32,
    /// `f32(PF90034 * PF30189)`, raw binary32, no clamp or normalization.
    pub shadow: f32,
    /// [`double_chin::Weights::is_active`]: `geometry >= 1e-4 || shadow >= 1e-4`
    /// evaluated as binary64 like the native `ucomisd` pair at
    /// `0x7E47324`/`0x7E47331`. The engine packet refuses non-finite products
    /// first, so the kernel's NaN-passes-the-gate machine quirk never reaches
    /// a caller through this wire.
    pub active: bool,
}

/// Same-job PF90034/30188/30189 per-face packet.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DoubleChinDegreeInputs {
    pub ticket: u64,
    pub pf90034: FaceDegreeMap,
    pub pf30188: FaceDegreeMap,
    pub pf30189: FaceDegreeMap,
}

impl DoubleChinDegreeInputs {
    pub fn new(
        ticket: u64,
        pf90034: FaceDegreeMap,
        pf30188: FaceDegreeMap,
        pf30189: FaceDegreeMap,
    ) -> Result<Self, Refusal> {
        if ticket == 0 {
            return Err(Refusal::MissingInput("PF90034_JOB_TICKET"));
        }
        if pf90034.is_empty() && pf30188.is_empty() && pf30189.is_empty() {
            return Err(Refusal::MissingInput("PF90034_PER_FACE_PACKET"));
        }
        Ok(Self {
            ticket,
            pf90034,
            pf30188,
            pf30189,
        })
    }

    /// Strictly per face: no broadcast, no accumulation. An absent multiplier is
    /// the vendor `0.0f`, so the face reads inactive. The arithmetic and the
    /// duplicate/finite checks delegate to the canonical
    /// [`double_chin::per_face_weights`].
    pub fn per_face_weights(&self) -> Result<Vec<DoubleChinFaceWeights>, Refusal> {
        let faces = union_faces(&[&self.pf90034, &self.pf30188, &self.pf30189]);
        if faces.is_empty() {
            return Err(Refusal::MissingInput("PF90034_PER_FACE_PACKET"));
        }
        let degrees: Vec<double_chin::FaceDegrees> = faces
            .iter()
            .map(|face| double_chin::FaceDegrees {
                face_id: *face,
                pf90034: self.pf90034.value(*face),
                pf30188: self.pf30188.value(*face),
                pf30189: self.pf30189.value(*face),
            })
            .collect();
        double_chin::per_face_weights(&degrees)?
            .into_iter()
            .map(|(face_id, weights)| {
                if !weights.geometry.is_finite() || !weights.shadow.is_finite() {
                    return Err(Refusal::ValueOutOfRange("PF90034_PER_FACE_WEIGHT"));
                }
                Ok(DoubleChinFaceWeights {
                    face_id,
                    geometry: weights.geometry,
                    shadow: weights.shadow,
                    active: weights.is_active(),
                })
            })
            .collect()
    }

    /// True when at least one face passes the recovered 1e-4 gate.
    pub fn any_active(&self) -> Result<bool, Refusal> {
        Ok(self
            .per_face_weights()?
            .iter()
            .any(|weights| weights.active))
    }
}

/// Clamped piecewise-linear remap used by the PF90085 producer; delegates to
/// the canonical [`tone_unify::piecewise_remap_clamped`] with the recovered
/// knots.
pub fn pf90085_metric_base(metric: f32) -> Result<f32, Refusal> {
    tone_unify::piecewise_remap_clamped(metric, &PF90085_METRIC_KNOTS, &PF90085_METRIC_VALUES)
}

/// Same-job PF90085 per-face degree packet.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SkinUnifyDegreeInputs {
    pub ticket: u64,
    /// Vendor PF90085 per-face value (`0x15FE5`); missing node reads `0.0f`.
    pub pf90085: FaceDegreeMap,
    /// Per-face producer metric `v`. Its physical meaning is LIVE_ONLY, so a
    /// missing entry is a typed refusal rather than a default.
    pub metric: FaceDegreeMap,
}

impl SkinUnifyDegreeInputs {
    pub fn new(
        ticket: u64,
        pf90085: FaceDegreeMap,
        metric: FaceDegreeMap,
    ) -> Result<Self, Refusal> {
        if ticket == 0 {
            return Err(Refusal::MissingInput("PF90085_JOB_TICKET"));
        }
        if pf90085.is_empty() && metric.is_empty() {
            return Err(Refusal::MissingInput("PF90085_PER_FACE_PACKET"));
        }
        Ok(Self {
            ticket,
            pf90085,
            metric,
        })
    }

    /// `degree = piecewiseRemap(v) * PF90085`, delegated to the canonical
    /// [`tone_unify::pf90085_metric_degree`]; the caller applies the `0.001`
    /// gate with [`Self::active`].
    pub fn degree(&self, face: u32) -> Result<f32, Refusal> {
        let metric = self.metric.value(face);
        let value = self.pf90085.value_or_zero(face);
        tone_unify::pf90085_metric_degree(metric, value)
    }

    pub fn active(&self, face: u32) -> Result<bool, Refusal> {
        Ok(self.degree(face)? > PF90085_DEGREE_GATE)
    }

    /// Resolve every face present in either map. A face that has a PF90085
    /// value but no metric is refused by name instead of being dropped; a
    /// metric face with no PF90085 node reads the vendor `0.0f`.
    pub fn validate(&self) -> Result<Vec<(u32, f32)>, Refusal> {
        let faces = union_faces(&[&self.pf90085, &self.metric]);
        if faces.is_empty() {
            return Err(Refusal::MissingInput("PF90085_FACE_METRIC"));
        }
        faces
            .into_iter()
            .map(|face| Ok((face, self.degree(face)?)))
            .collect()
    }
}

/// Vendor brow part kind (task08 §2); canonical in
/// [`eye_geometry::BrowPartKind`].
pub use eye_geometry::BrowPartKind;

/// One `(part, ControlIndex)` binding; canonical in
/// [`eye_geometry::BrowLengthSlot`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrowLengthSlot {
    pub part: BrowPartKind,
    /// Vendor `ControlIndex`: 0=head-left, 1=head-right, 2=tail-left, 3=tail-right.
    pub control_index: usize,
}

/// The kernel's local refusal enum differs from the crate one; the wire maps
/// every eye-geometry refusal onto `ValueOutOfRange` with the kernel's name,
/// which is all the stage report needs to stay actionable.
fn map_eye_refusal(name: &'static str) -> Refusal {
    Refusal::ValueOutOfRange(name)
}

impl BrowLengthSlot {
    pub fn new(part: BrowPartKind, control_index: usize) -> Result<Self, Refusal> {
        if control_index > 3 {
            return Err(Refusal::ValueOutOfRange("brow_control_index"));
        }
        Ok(Self {
            part,
            control_index,
        })
    }

    pub fn is_head(&self) -> bool {
        self.control_index < 2
    }

    fn kernel_slot(self) -> Result<eye_geometry::BrowLengthSlot, Refusal> {
        eye_geometry::BrowLengthSlot::new(self.part, self.control_index)
            .map_err(|refusal| map_eye_refusal(eye_geometry_refusal_name(refusal)))
    }

    /// Three-node `ControlCurve::remap` over `[0,0.5,1]`; neutral `0.5` is
    /// `0.0`. Delegates to the canonical
    /// [`eye_geometry::brow_length_slot_degree`].
    pub fn degree(&self, ui01: f32) -> Result<f32, Refusal> {
        if !ui01.is_finite() || !(0.0..=1.0).contains(&ui01) {
            return Err(Refusal::ValueOutOfRange("brow_length_ui"));
        }
        eye_geometry::brow_length_slot_degree(self.kernel_slot()?, ui01)
            .map_err(|refusal| map_eye_refusal(eye_geometry_refusal_name(refusal)))
    }
}

fn eye_geometry_refusal_name(refusal: eye_geometry::Refusal) -> &'static str {
    match refusal {
        eye_geometry::Refusal::NonFinite(name)
        | eye_geometry::Refusal::OutOfRange(name)
        | eye_geometry::Refusal::MissingMesh(name)
        | eye_geometry::Refusal::MissingEbo(name)
        | eye_geometry::Refusal::OpenGeometry(name)
        | eye_geometry::Refusal::ZeroQuads(name) => name,
        eye_geometry::Refusal::MissingLandmarks => "brow_length_missing_landmarks",
        eye_geometry::Refusal::MissingGuide => "brow_length_missing_guide",
        eye_geometry::Refusal::MissingFaceFields => "brow_length_missing_face_fields",
        eye_geometry::Refusal::MissingParamList => "brow_length_missing_param_list",
        eye_geometry::Refusal::MissingFactorList => "brow_length_missing_factor_list",
        eye_geometry::Refusal::MissingPreparedMesh => "brow_length_missing_prepared_mesh",
        eye_geometry::Refusal::UnsupportedRoute(_) => "brow_length_unsupported_route",
    }
}

/// Both recovered parts for one control slot.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BrowSlotDegrees {
    pub short: f32,
    pub long: f32,
}

/// Same-frame brow length controls.
///
/// The four entries are vendor `ControlIndex` 0..=3, i.e. PF44124 (head-left),
/// PF48019 (head-right), PF49221 (tail-left), PF41922 (tail-right). The UI
/// value is the engine slider; the short/long part split is derived by the
/// recovered three-node remap, not by a hard-coded sign.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BrowLengthDegreeInputs {
    pub ticket: u64,
    pub ui01: [f32; 4],
}

impl BrowLengthDegreeInputs {
    pub fn new(ticket: u64, ui01: [f32; 4]) -> Result<Self, Refusal> {
        if ticket == 0 {
            return Err(Refusal::MissingInput("BROW_LENGTH_JOB_TICKET"));
        }
        if ui01.iter().any(|value| !value.is_finite() || !(0.0..=1.0).contains(value)) {
            return Err(Refusal::ValueOutOfRange("brow_length_ui"));
        }
        Ok(Self { ticket, ui01 })
    }

    /// Resolve the short and long degree of every control slot.
    pub fn resolved(&self) -> Result<[BrowSlotDegrees; 4], Refusal> {
        let mut out = [BrowSlotDegrees::default(); 4];
        for (index, slot) in out.iter_mut().enumerate() {
            let ui = self.ui01[index];
            slot.short = BrowLengthSlot::new(BrowPartKind::Short, index)?.degree(ui)?;
            slot.long = BrowLengthSlot::new(BrowPartKind::Long, index)?.degree(ui)?;
        }
        Ok(out)
    }

    pub fn is_neutral(&self) -> bool {
        self.ui01 == [0.5; 4]
    }
}

/// The two three-point brow segments from the 194-point landmark packet
/// (`{39,41,43}` and `{22,24,26}`, task08 §4.1). Rendering the 144-vertex
/// consumer (`eye_geometry::brow_length_consumer`) needs them plus the three
/// 24-float coefficient arrays, which are not in the fixed PE yet.
pub const BROW_SEGMENT_A_LANDMARKS: [usize; 3] = [39, 41, 43];
/// Segment B three-point landmark window (`0x707EB54`, task08 §4.1).
pub const BROW_SEGMENT_B_LANDMARKS: [usize; 3] = [22, 24, 26];

pub fn brow_segments(
    points194: &[[f32; 2]],
) -> Result<([[f32; 2]; 3], [[f32; 2]; 3]), Refusal> {
    if points194.len() < 194 {
        return Err(Refusal::ShapeMismatch("brow_length_194_landmarks"));
    }
    let mut segment_a = [[0.0f32; 2]; 3];
    let mut segment_b = [[0.0f32; 2]; 3];
    for (index, landmark) in BROW_SEGMENT_A_LANDMARKS.iter().enumerate() {
        let point = points194[*landmark];
        if !point[0].is_finite() || !point[1].is_finite() {
            return Err(Refusal::ValueOutOfRange("brow_length_landmark"));
        }
        segment_a[index] = point;
    }
    for (index, landmark) in BROW_SEGMENT_B_LANDMARKS.iter().enumerate() {
        let point = points194[*landmark];
        if !point[0].is_finite() || !point[1].is_finite() {
            return Err(Refusal::ValueOutOfRange("brow_length_landmark"));
        }
        segment_b[index] = point;
    }
    Ok((segment_a, segment_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(entries: &[(u32, f32)]) -> FaceDegreeMap {
        FaceDegreeMap::new(entries.to_vec()).unwrap()
    }

    #[test]
    fn per_face_map_rejects_duplicates_and_non_finite_values() {
        assert_eq!(
            FaceDegreeMap::new(vec![(1, 0.5), (1, 0.3)]),
            Err(Refusal::ShapeMismatch("longtail per-face degree face id"))
        );
        assert_eq!(
            FaceDegreeMap::new(vec![(1, f32::NAN)]),
            Err(Refusal::ValueOutOfRange("longtail per-face degree"))
        );
        let sorted = map(&[(2, 0.2), (1, 0.1)]);
        assert_eq!(sorted.value(1), Some(0.1));
        assert_eq!(sorted.value_or_zero(9), 0.0);
    }

    #[test]
    fn double_chin_weights_match_the_recovered_product_and_gate() {
        let inputs = DoubleChinDegreeInputs::new(
            7,
            map(&[(1, 0.5), (2, 0.0)]),
            map(&[(1, 0.4)]),
            map(&[(1, 0.2)]),
        )
        .unwrap();
        let weights = inputs.per_face_weights().unwrap();
        assert_eq!(weights.len(), 2);
        assert_eq!(
            weights[0],
            DoubleChinFaceWeights {
                face_id: 1,
                geometry: 0.2,
                shadow: 0.1,
                active: true,
            }
        );
        // Face 2 has PF90034 = 0.0 and no PF30188/30189 entry -> both 0 -> inactive.
        assert_eq!(weights[1].geometry, 0.0);
        assert!(!weights[1].active);
        assert!(inputs.any_active().unwrap());
    }

    #[test]
    fn double_chin_weights_delegate_to_the_kernel_producer() {
        let inputs = DoubleChinDegreeInputs::new(
            7,
            map(&[(3, 0.7)]),
            map(&[(3, 0.9)]),
            map(&[(3, 0.3)]),
        )
        .unwrap();
        let wire = inputs.per_face_weights().unwrap();
        let kernel = double_chin::per_face_weights(&[double_chin::FaceDegrees {
            face_id: 3,
            pf90034: Some(0.7),
            pf30188: Some(0.9),
            pf30189: Some(0.3),
        }])
        .unwrap();
        assert_eq!(wire[0].geometry, kernel[0].1.geometry);
        assert_eq!(wire[0].shadow, kernel[0].1.shadow);
        assert_eq!(wire[0].active, kernel[0].1.is_active());
    }

    #[test]
    fn double_chin_gate_uses_binary64_and_missing_packet_refuses() {
        // `1e-4f32` widens to `9.9999997e-5` binary64, i.e. BELOW the recovered
        // `1e-4` binary64 gate: exactly the f32 value is inactive. This is the
        // canonical kernel semantics (`double_chin::WEIGHT_GATE`).
        let inputs = DoubleChinDegreeInputs::new(
            3,
            map(&[(1, 1e-4)]),
            map(&[(1, 1.0)]),
            FaceDegreeMap::default(),
        )
        .unwrap();
        assert!(!inputs.any_active().unwrap());
        // One f32 ulp above the gate is active.
        let inputs = DoubleChinDegreeInputs::new(
            3,
            map(&[(1, 1.000_000_2e-4)]),
            map(&[(1, 1.0)]),
            FaceDegreeMap::default(),
        )
        .unwrap();
        assert!(inputs.any_active().unwrap());
        // Both products below 1e-4 -> inactive.
        let inputs = DoubleChinDegreeInputs::new(
            3,
            map(&[(1, 1e-5)]),
            map(&[(1, 1.0)]),
            map(&[(1, 1.0)]),
        )
        .unwrap();
        assert!(!inputs.any_active().unwrap());
        assert_eq!(
            DoubleChinDegreeInputs::new(
                0,
                map(&[(1, 0.5)]),
                FaceDegreeMap::default(),
                FaceDegreeMap::default()
            ),
            Err(Refusal::MissingInput("PF90034_JOB_TICKET"))
        );
        assert_eq!(
            DoubleChinDegreeInputs::new(
                1,
                FaceDegreeMap::default(),
                FaceDegreeMap::default(),
                FaceDegreeMap::default()
            ),
            Err(Refusal::MissingInput("PF90034_PER_FACE_PACKET"))
        );
    }

    #[test]
    fn pf90085_metric_remaps_each_segment_and_clamps() {
        assert_eq!(pf90085_metric_base(4.0).unwrap(), 0.0);
        assert!((pf90085_metric_base(4.75).unwrap() - 0.05).abs() < 1e-6);
        assert!((pf90085_metric_base(6.5).unwrap() - 0.55).abs() < 1e-5);
        assert_eq!(pf90085_metric_base(9.0).unwrap(), 1.0);
        assert_eq!(pf90085_metric_base(-3.0).unwrap(), 0.0);
        assert_eq!(pf90085_metric_base(12.0).unwrap(), 1.0);
        assert_eq!(
            pf90085_metric_base(f32::NAN),
            Err(Refusal::ValueOutOfRange("PF90085_METRIC"))
        );
    }

    #[test]
    fn pf90085_degree_requires_the_face_metric() {
        let inputs =
            SkinUnifyDegreeInputs::new(5, map(&[(1, 0.6)]), map(&[(1, 9.0)])).unwrap();
        assert!((inputs.degree(1).unwrap() - 0.6).abs() < 1e-6);
        assert!(inputs.active(1).unwrap());
        assert_eq!(
            inputs.degree(2),
            Err(Refusal::MissingInput("PF90085_FACE_METRIC"))
        );
        // Missing PF90085 node reads 0.0f -> degree 0, inactive.
        let inputs =
            SkinUnifyDegreeInputs::new(5, FaceDegreeMap::default(), map(&[(1, 9.0)])).unwrap();
        assert_eq!(inputs.degree(1).unwrap(), 0.0);
        assert!(!inputs.active(1).unwrap());
        assert_eq!(
            SkinUnifyDegreeInputs::new(0, map(&[(1, 1.0)]), map(&[(1, 9.0)])),
            Err(Refusal::MissingInput("PF90085_JOB_TICKET"))
        );
    }

    #[test]
    fn brow_slots_match_the_vendor_two_part_tables() {
        let short_head = BrowLengthSlot::new(BrowPartKind::Short, 0).unwrap();
        assert!((short_head.degree(0.0).unwrap() + 1.0).abs() < 1e-6);
        assert_eq!(short_head.degree(0.5).unwrap(), 0.0);
        assert_eq!(short_head.degree(1.0).unwrap(), 0.0);
        let short_tail = BrowLengthSlot::new(BrowPartKind::Short, 2).unwrap();
        assert!((short_tail.degree(0.0).unwrap() + 1.6).abs() < 1e-6);
        let long_head = BrowLengthSlot::new(BrowPartKind::Long, 0).unwrap();
        assert!((long_head.degree(1.0).unwrap() - 1.1).abs() < 1e-5);
        let long_tail = BrowLengthSlot::new(BrowPartKind::Long, 2).unwrap();
        assert!((long_tail.degree(1.0).unwrap() - 1.8).abs() < 1e-6);
        assert_eq!(long_tail.degree(0.5).unwrap(), 0.0);
        assert_eq!(
            BrowLengthSlot::new(BrowPartKind::Short, 4),
            Err(Refusal::ValueOutOfRange("brow_control_index"))
        );
        assert_eq!(
            long_head.degree(1.1),
            Err(Refusal::ValueOutOfRange("brow_length_ui"))
        );
    }

    #[test]
    fn brow_packet_resolves_four_slots_and_validates_ui() {
        let packet = BrowLengthDegreeInputs::new(9, [0.0, 0.0, 1.0, 1.0]).unwrap();
        let slots = packet.resolved().unwrap();
        assert!((slots[0].short + 1.0).abs() < 1e-6);
        assert_eq!(slots[0].long, 0.0);
        assert!((slots[2].long - 1.8).abs() < 1e-6);
        assert_eq!(slots[2].short, 0.0);
        assert!(!packet.is_neutral());
        assert!(BrowLengthDegreeInputs::new(9, [0.5; 4]).unwrap().is_neutral());
        assert_eq!(
            BrowLengthDegreeInputs::new(9, [0.5, 0.5, 0.5, 1.5]),
            Err(Refusal::ValueOutOfRange("brow_length_ui"))
        );
        assert_eq!(
            BrowLengthDegreeInputs::new(0, [0.5; 4]),
            Err(Refusal::MissingInput("BROW_LENGTH_JOB_TICKET"))
        );
    }

    #[test]
    fn brow_segments_pick_the_recovered_landmark_windows() {
        let points: Vec<[f32; 2]> = (0..194).map(|i| [i as f32, i as f32 * 2.0]).collect();
        let (a, b) = brow_segments(&points).unwrap();
        assert_eq!(a, [[39.0, 78.0], [41.0, 82.0], [43.0, 86.0]]);
        assert_eq!(b, [[22.0, 44.0], [24.0, 48.0], [26.0, 52.0]]);
        assert_eq!(
            brow_segments(&points[..193]),
            Err(Refusal::ShapeMismatch("brow_length_194_landmarks"))
        );
    }

    #[test]
    fn skin_unify_validate_requires_every_face_metric() {
        let inputs = SkinUnifyDegreeInputs::new(
            5,
            map(&[(1, 0.6), (2, 0.4)]),
            map(&[(1, 9.0), (2, 5.2)]),
        )
        .unwrap();
        let degrees = inputs.validate().unwrap();
        assert_eq!(degrees.len(), 2);
        assert!((degrees[0].1 - 0.6).abs() < 1e-6);
        // 5.2 -> base 0.1/0.666..*0.4; just check finiteness and positivity.
        assert!(degrees[1].1 > 0.0);
        let missing = SkinUnifyDegreeInputs::new(
            5,
            map(&[(1, 0.6), (2, 0.4)]),
            map(&[(1, 9.0)]),
        )
        .unwrap();
        assert_eq!(
            missing.validate(),
            Err(Refusal::MissingInput("PF90085_FACE_METRIC"))
        );
        let empty = SkinUnifyDegreeInputs::new(
            5,
            map(&[(1, 0.6)]),
            FaceDegreeMap::default(),
        )
        .unwrap();
        assert_eq!(
            empty.validate(),
            Err(Refusal::MissingInput("PF90085_FACE_METRIC"))
        );
        // A metric-only packet reads the vendor 0.0f for the missing PF90085
        // node and still validates (degree 0, inactive).
        let metric_only =
            SkinUnifyDegreeInputs::new(5, FaceDegreeMap::default(), map(&[(1, 9.0)])).unwrap();
        assert_eq!(metric_only.validate().unwrap(), vec![(1, 0.0)]);
    }

    #[test]
    fn producer_gap_names_match_the_wiring_map() {
        assert_eq!(
            super::PF90085_PRODUCER_GAP,
            "PF90085 per-face metric v + FaceParsingV1 skin image + FaceParsingV2 planes + landmarks + ref_img + 74-sample HSL controls + GammaMap + sRGB/ProPhoto matrices (degree law recovered; pixel producer absent)"
        );
        assert_eq!(
            super::PF90087_PRODUCER_GAP,
            "PF90087 HairSeamV0 hairMask channel/size/binding NOT-FOUND in the fixed PE; per-part degree map d_i>0.001 recovered; no local HairSeam model"
        );
        assert_eq!(
            super::PF100005_PRODUCER_GAP,
            "PF100005 eye-white model 3071265413/1527454531/2294125226 and 8-plane float32 output absent locally; 4+4 split and mixer tail known, model-to-layer RGB/mask binding open"
        );
        assert_eq!(
            super::PF30209_PRODUCER_GAP,
            "PF30209 stage +0x3A0/+0x3B8 is not forwarded by the fixed PE; ordered jawline index table is not a PE constant; 33-point window L[109..=141] + center L[54] recovered"
        );
        assert!(super::PF30209_PRODUCER_GAP
            .starts_with(aurora_portrait::kernels::remaining_special_provider::PF30209_END_STATIC));
        assert_eq!(
            super::PF30188_PRODUCER_GAP,
            "PF30188/PF30189 double-chin weights f32(PF90034*PF30188|PF30189) with 1e-4 gate recovered; same-job per-face packet and the final model field/shadow composite are open"
        );
    }
}
