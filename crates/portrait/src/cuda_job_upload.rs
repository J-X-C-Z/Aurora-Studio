//! One shape-addressed upload description for a portrait provider job.
//!
//! Provider payloads arrive in several typed arrays: geometry points, repair
//! planes and historical model outputs. The CUDA boundary only needs one stable
//! description of those payloads, one shape key, one content key per slice and
//! one upload/synchronisation boundary.

use std::fmt;
use std::mem::size_of;

use crate::Refusal;

/// Device slices are separated at the CUDA allocation granularity used by the
/// portrait path. Padding belongs to the arena and is never interpreted.
pub const JOB_SLICE_ALIGN: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum JobBufferRole {
    FaceGeometry = 1,
    BodyGeometry = 2,
    RepairPlane = 3,
    HistoricalTone = 4,
    HistoricalSkin = 5,
}

impl JobBufferRole {
    pub const fn label(self) -> &'static str {
        match self {
            Self::FaceGeometry => "face-geometry",
            Self::BodyGeometry => "body-geometry",
            Self::RepairPlane => "repair-plane",
            Self::HistoricalTone => "historical-tone",
            Self::HistoricalSkin => "historical-skin",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum JobBufferPurpose {
    /// Provider bytes uploaded before the first kernel consumes the arena.
    ReadOnlyInput = 1,
    /// Host bytes needed only to seed a kernel output.
    Seed = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum JobElement {
    U8 = 1,
    U16 = 2,
    U32 = 4,
    F32 = 8,
    F32x2 = 16,
    F32x4 = 32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct JobShapeKey {
    pub lo: u64,
    pub hi: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct JobContentKey {
    pub lo: u64,
    pub hi: u64,
}

/// Typed provider bytes. Implementations emit little-endian scalar bytes so the
/// same description works for `[u8]`, `[u16]` and `[f32]` payloads.
pub trait JobPayload {
    fn element(&self) -> JobElement;
    fn byte_len(&self) -> usize;
    fn for_each_le_bytes(&self, emit: &mut dyn FnMut(&[u8]));

    fn content_key(&self) -> JobContentKey {
        let mut hash = Key128::new(0x5750_392d_4a4f_4221);
        hash.write(&[self.element() as u8]);
        hash.write_usize(self.byte_len());
        self.for_each_le_bytes(&mut |bytes| hash.write(bytes));
        hash.finish()
    }

    fn write_le(&self, dst: &mut [u8]) {
        let mut offset = 0usize;
        self.for_each_le_bytes(&mut |bytes| {
            let end = offset + bytes.len();
            dst[offset..end].copy_from_slice(bytes);
            offset = end;
        });
        assert_eq!(offset, dst.len(), "job payload length changed while writing");
    }
}

macro_rules! impl_primitive_payload {
    ($ty:ty, $element:expr) => {
        impl JobPayload for [$ty] {
            fn element(&self) -> JobElement {
                $element
            }

            fn byte_len(&self) -> usize {
                self.len() * size_of::<$ty>()
            }

            fn for_each_le_bytes(&self, emit: &mut dyn FnMut(&[u8])) {
                let mut bytes = [0u8; size_of::<$ty>()];
                for value in self {
                    bytes.copy_from_slice(&value.to_le_bytes());
                    emit(&bytes);
                }
            }
        }
    };
}

impl_primitive_payload!(u8, JobElement::U8);
impl_primitive_payload!(u16, JobElement::U16);
impl_primitive_payload!(u32, JobElement::U32);
impl_primitive_payload!(f32, JobElement::F32);

/// Owned-length bridges so a caller can hand &[T; N] straight to a descriptor.
/// A &[T; N] only unsizes to &dyn JobPayload in one step, so the array
/// itself must implement the trait; going via the slice would be two unsizings.
macro_rules! impl_array_payload {
    ($ty:ty, $element:expr) => {
        impl<const N: usize> JobPayload for [$ty; N] {
            fn element(&self) -> JobElement {
                $element
            }

            fn byte_len(&self) -> usize {
                <[$ty]>::byte_len(self)
            }

            fn for_each_le_bytes(&self, emit: &mut dyn FnMut(&[u8])) {
                <[$ty]>::for_each_le_bytes(self, emit)
            }
        }
    };
}

impl_array_payload!(u8, JobElement::U8);
impl_array_payload!(u16, JobElement::U16);
impl_array_payload!(u32, JobElement::U32);
impl_array_payload!(f32, JobElement::F32);

impl<const N: usize> JobPayload for [[f32; 2]; N] {
    fn element(&self) -> JobElement {
        JobElement::F32x2
    }

    fn byte_len(&self) -> usize {
        <[[f32; 2]]>::byte_len(self)
    }

    fn for_each_le_bytes(&self, emit: &mut dyn FnMut(&[u8])) {
        <[[f32; 2]]>::for_each_le_bytes(self, emit)
    }
}

impl<const N: usize> JobPayload for [[f32; 4]; N] {
    fn element(&self) -> JobElement {
        JobElement::F32x4
    }

    fn byte_len(&self) -> usize {
        <[[f32; 4]]>::byte_len(self)
    }

    fn for_each_le_bytes(&self, emit: &mut dyn FnMut(&[u8])) {
        <[[f32; 4]]>::for_each_le_bytes(self, emit)
    }
}

impl JobPayload for [[f32; 2]] {
    fn element(&self) -> JobElement {
        JobElement::F32x2
    }

    fn byte_len(&self) -> usize {
        self.len() * size_of::<[f32; 2]>()
    }

    fn for_each_le_bytes(&self, emit: &mut dyn FnMut(&[u8])) {
        for point in self {
            for value in point {
                emit(&value.to_le_bytes());
            }
        }
    }
}

impl JobPayload for [[f32; 4]] {
    fn element(&self) -> JobElement {
        JobElement::F32x4
    }

    fn byte_len(&self) -> usize {
        self.len() * size_of::<[f32; 4]>()
    }

    fn for_each_le_bytes(&self, emit: &mut dyn FnMut(&[u8])) {
        for value in self {
            for lane in value {
                emit(&lane.to_le_bytes());
            }
        }
    }
}

pub struct JobBufferDesc<'a> {
    pub role: JobBufferRole,
    pub ordinal: u16,
    pub purpose: JobBufferPurpose,
    pub payload: &'a dyn JobPayload,
}

impl fmt::Debug for JobBufferDesc<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JobBufferDesc")
            .field("role", &self.role)
            .field("ordinal", &self.ordinal)
            .field("purpose", &self.purpose)
            .field("element", &self.payload.element())
            .field("bytes", &self.payload.byte_len())
            .finish()
    }
}

/// Stable geometry of one CUDA upload arena. Content is deliberately excluded:
/// a shape-key hit means the device layout can be reused, not that its bytes are
/// current.
pub struct JobBufferPlan<'a> {
    job_id: u64,
    frame_size: (u32, u32),
    shape_key: JobShapeKey,
    total_bytes: usize,
    buffers: Vec<JobBufferDesc<'a>>,
    offsets: Vec<usize>,
    content_keys: Vec<JobContentKey>,
}

impl<'a> JobBufferPlan<'a> {
    pub fn new(
        job_id: u64,
        frame_size: (u32, u32),
        mut buffers: Vec<JobBufferDesc<'a>>,
    ) -> Result<Self, Refusal> {
        if job_id == 0 || frame_size.0 == 0 || frame_size.1 == 0 {
            return Err(Refusal::ValueOutOfRange("CUDA job identity"));
        }
        if buffers.is_empty() {
            return Err(Refusal::MissingInput("CUDA job payloads"));
        }
        if buffers.len() > u16::MAX as usize {
            return Err(Refusal::ValueOutOfRange("CUDA job payload count"));
        }
        buffers.sort_by_key(|buffer| (buffer.role, buffer.ordinal, buffer.purpose));
        for pair in buffers.windows(2) {
            if (pair[0].role, pair[0].ordinal) == (pair[1].role, pair[1].ordinal) {
                return Err(Refusal::ValueOutOfRange("duplicate CUDA job payload"));
            }
        }

        let mut offsets = Vec::with_capacity(buffers.len());
        let mut content_keys = Vec::with_capacity(buffers.len());
        let mut cursor = 0usize;
        for buffer in &buffers {
            let bytes = buffer.payload.byte_len();
            if bytes == 0 {
                return Err(Refusal::ShapeMismatch("empty CUDA job payload"));
            }
            cursor = align_up(cursor, JOB_SLICE_ALIGN)
                .ok_or(Refusal::ValueOutOfRange("CUDA job arena"))?;
            offsets.push(cursor);
            cursor = cursor
                .checked_add(bytes)
                .ok_or(Refusal::ValueOutOfRange("CUDA job arena"))?;
            content_keys.push(buffer.payload.content_key());
        }
        let total_bytes = cursor.max(1);
        let shape_key = shape_key(frame_size, &buffers, &offsets, total_bytes);

        Ok(Self {
            job_id,
            frame_size,
            shape_key,
            total_bytes,
            buffers,
            offsets,
            content_keys,
        })
    }

    pub fn job_id(&self) -> u64 {
        self.job_id
    }

    pub fn frame_size(&self) -> (u32, u32) {
        self.frame_size
    }

    pub fn shape_key(&self) -> JobShapeKey {
        self.shape_key
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    pub fn descriptor(&self, index: usize) -> Option<&JobBufferDesc<'a>> {
        self.buffers.get(index)
    }

    pub fn offset(&self, index: usize) -> Option<usize> {
        self.offsets.get(index).copied()
    }

    pub fn byte_len(&self, index: usize) -> Option<usize> {
        self.buffers
            .get(index)
            .map(|buffer| buffer.payload.byte_len())
    }

    pub fn content_key(&self, index: usize) -> Option<JobContentKey> {
        self.content_keys.get(index).copied()
    }

    pub fn write_payload(&self, index: usize, dst: &mut [u8]) -> Result<(), Refusal> {
        let buffer = self
            .buffers
            .get(index)
            .ok_or(Refusal::ShapeMismatch("CUDA job payload index"))?;
        if buffer.payload.byte_len() != dst.len() {
            return Err(Refusal::ShapeMismatch("CUDA job payload length"));
        }
        buffer.payload.write_le(dst);
        Ok(())
    }

    pub fn changed_from(&self, prior: &[JobContentKey]) -> Vec<bool> {
        self.content_keys
            .iter()
            .enumerate()
            .map(|(index, key)| prior.get(index) != Some(key))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JobUploadReceipt {
    pub shape_key: JobShapeKey,
    pub payloads: usize,
    pub uploaded_payloads: usize,
    pub deduped_payloads: usize,
    pub h2d_copies: usize,
    pub h2d_bytes: usize,
    pub explicit_syncs: usize,
    pub arena_bytes: usize,
}

fn align_up(value: usize, align: usize) -> Option<usize> {
    debug_assert!(align.is_power_of_two());
    value.checked_add(align - 1).map(|value| value & !(align - 1))
}

fn shape_key(
    frame_size: (u32, u32),
    buffers: &[JobBufferDesc<'_>],
    offsets: &[usize],
    total_bytes: usize,
) -> JobShapeKey {
    let mut hash = Key128::new(0x5750_392d_5348_4150);
    hash.write_u32(frame_size.0);
    hash.write_u32(frame_size.1);
    hash.write_usize(buffers.len());
    hash.write_usize(total_bytes);
    hash.write_usize(JOB_SLICE_ALIGN);
    for (buffer, offset) in buffers.iter().zip(offsets) {
        hash.write(&[buffer.role as u8, buffer.purpose as u8]);
        hash.write_u16(buffer.ordinal);
        hash.write(&[buffer.payload.element() as u8]);
        hash.write_usize(*offset);
        hash.write_usize(buffer.payload.byte_len());
    }
    hash.finish_shape()
}

struct Key128 {
    lo: u64,
    hi: u64,
}

impl Key128 {
    fn new(seed: u64) -> Self {
        Self {
            lo: 0xcbf2_9ce4_8422_2325 ^ seed,
            hi: 0x9e37_79b9_7f4a_7c15 ^ seed.rotate_left(17),
        }
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.lo ^= u64::from(byte);
            self.lo = self.lo.wrapping_mul(0x0000_0100_0000_01b3);
            self.hi ^= u64::from(byte).wrapping_add(self.lo.rotate_left(13));
            self.hi = self
                .hi
                .wrapping_mul(0x9e37_79b1_85eb_ca87)
                .rotate_left(11);
        }
    }

    fn write_u16(&mut self, value: u16) {
        self.write(&value.to_le_bytes());
    }

    fn write_u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }

    fn write_usize(&mut self, value: usize) {
        self.write(&(value as u64).to_le_bytes());
    }

    fn finish(self) -> JobContentKey {
        JobContentKey {
            lo: self.lo,
            hi: self.hi,
        }
    }

    fn finish_shape(self) -> JobShapeKey {
        JobShapeKey {
            lo: self.lo,
            hi: self.hi,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc<'a>(
        role: JobBufferRole,
        ordinal: u16,
        payload: &'a dyn JobPayload,
    ) -> JobBufferDesc<'a> {
        JobBufferDesc {
            role,
            ordinal,
            purpose: JobBufferPurpose::ReadOnlyInput,
            payload,
        }
    }

    #[test]
    fn shape_reuses_layout_while_content_tracks_bytes() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.0, 4.0];
        let p1 = JobBufferPlan::new(
            7,
            (512, 512),
            vec![desc(JobBufferRole::FaceGeometry, 0, &a)],
        )
        .unwrap();
        let p2 = JobBufferPlan::new(
            7,
            (512, 512),
            vec![desc(JobBufferRole::FaceGeometry, 0, &b)],
        )
        .unwrap();
        assert_eq!(p1.shape_key(), p2.shape_key());
        assert_ne!(p1.content_key(0), p2.content_key(0));
        assert_eq!(p1.offsets[0], 0);
    }

    #[test]
    fn duplicate_role_ordinal_is_refused() {
        let a = [1u8];
        let error = match JobBufferPlan::new(
            7,
            (1, 1),
            vec![
                desc(JobBufferRole::RepairPlane, 2, &a),
                desc(JobBufferRole::RepairPlane, 2, &a),
            ],
        ) {
            Ok(_) => panic!("duplicate payload must be refused"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            Refusal::ValueOutOfRange("duplicate CUDA job payload")
        );
    }

    #[test]
    fn changed_mask_is_per_payload() {
        let a = [1u8, 2];
        let b = [1u8, 2];
        let c = [9u8, 2];
        let first = JobBufferPlan::new(
            7,
            (1, 1),
            vec![
                desc(JobBufferRole::RepairPlane, 0, &a),
                desc(JobBufferRole::HistoricalSkin, 0, &b),
            ],
        )
        .unwrap();
        let keys = [first.content_key(0).unwrap(), first.content_key(1).unwrap()];
        let second = JobBufferPlan::new(
            8,
            (1, 1),
            vec![
                desc(JobBufferRole::RepairPlane, 0, &c),
                desc(JobBufferRole::HistoricalSkin, 0, &b),
            ],
        )
        .unwrap();
        assert_eq!(second.changed_from(&keys), vec![true, false]);
    }
}
