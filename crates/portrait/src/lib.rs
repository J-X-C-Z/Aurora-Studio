//! PixelCake-derived effect backend.
//!
//! Modules expose recovered geometry, parameter and pixel stages, with separate
//! runtime adapters for GPU work. Nothing discovers masks, faces or model
//! outputs: a caller supplies the same-job inputs. A callable stage does not
//! establish that the entire original effect pipeline is reproduced.
//!
//! CPU kernels have no windowing dependency and stay testable headless.

pub mod kernels;
pub mod device;
pub mod cuda;
pub mod cuda_lut;
pub mod cuda_mesh;
pub mod cuda_gloss;
pub mod cuda_teeth;
pub mod cuda_uvmap;
pub mod cuda_moisture;
pub mod cuda_smile;
pub mod cuda_texture_blend;
pub mod cuda_eye_bag;
pub mod cuda_beard;
pub mod cuda_makeup;
pub mod cuda_teeth_blend;
pub mod cuda_mesh_attribute;
pub mod raster;
pub mod raster_fragment;
pub mod metal;
pub mod metal_mesh;
pub mod metal_smile;

/// Packed RGBA8 image, row-major, four bytes per pixel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Image {
    /// Returns `None` when `pixels` is not exactly `width * height * 4` bytes.
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Option<Self> {
        let expected = Self::byte_len(width, height)?;
        (pixels.len() == expected).then_some(Self {
            width,
            height,
            pixels,
        })
    }

    fn byte_len(width: u32, height: u32) -> Option<usize> {
        if width == 0 || height == 0 {
            return None;
        }
        (width as usize)
            .checked_mul(height as usize)?
            .checked_mul(4)
    }

    /// Public fields allow FFI callers to assemble an image, so kernels validate
    /// at entry rather than relying solely on the constructor.
    pub fn validate(&self) -> Result<(), Refusal> {
        if Self::byte_len(self.width, self.height) != Some(self.pixels.len()) {
            return Err(Refusal::ShapeMismatch("image"));
        }
        Ok(())
    }

    pub fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Self {
        let len = Self::byte_len(width, height).expect("nonempty image with valid dimensions");
        Self {
            width,
            height,
            pixels: rgba.iter().copied().cycle().take(len).collect(),
        }
    }

    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }

    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        assert!(
            x < self.width && y < self.height,
            "pixel coordinates outside image"
        );
        let i = (y as usize * self.width as usize + x as usize) * 4;
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }
}

/// Why an effect could not produce pixels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    UnknownEffect(u32),
    MissingInput(&'static str),
    ShapeMismatch(&'static str),
    ValueOutOfRange(&'static str),
    NotImplemented(u32),
}

/// What the backend can actually do for an effect today.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// A CPU geometry or pixel stage exists; this does not imply a full effect.
    Cpu,
    /// Parameter/producer math exists, but the final execution path is still open.
    GpuPlanned,
    /// Recovered only far enough to refuse; no pixel consumer.
    Refused,
}

/// One registry row: what an effect is and what a same-job call must supply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EffectMeta {
    /// Vendor `ParamFlag` this row belongs to.
    pub pf: u32,
    /// Settings key used by presets and the vendor UI registry, empty if unresolved.
    pub key: &'static str,
    pub label: &'static str,
    pub access: Access,
    /// Live inputs the kernel requires; empty means source pixels are enough.
    pub inputs: &'static [&'static str],
}

/// All registry rows, in `ParamFlag` order.
pub fn catalog() -> Vec<EffectMeta> {
    let mut all = kernels::registry();
    all.sort_by_key(|meta| meta.pf);
    all
}

pub fn lookup(pf: u32) -> Option<EffectMeta> {
    catalog().into_iter().find(|meta| meta.pf == pf)
}

/// Machine-readable view of the registry for a host that configures effects
/// from data instead of linking every call site. Rebuilt on demand; the
/// catalog is small.
pub fn catalog_json() -> String {
    fn escape(text: &str) -> String {
        text.chars()
            .flat_map(|c| match c {
                '"' => "\\\"".chars().collect::<Vec<_>>(),
                '\\' => "\\\\".chars().collect(),
                '\n' => "\\n".chars().collect(),
                c => vec![c],
            })
            .collect()
    }
    let mut out = String::from("[");
    for (index, row) in catalog().iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let access = match row.access {
            Access::Cpu => "cpu",
            Access::GpuPlanned => "gpu_planned",
            Access::Refused => "refused",
        };
        let inputs: Vec<String> = row.inputs.iter().map(|i| format!("\"{}\"", escape(i))).collect();
        out.push_str(&format!(
            "{{\"pf\":{},\"key\":\"{}\",\"label\":\"{}\",\"access\":\"{}\",\"inputs\":[{}]}}",
            row.pf,
            escape(row.key),
            escape(row.label),
            access,
            inputs.join(",")
        ));
    }
    out.push(']');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_rejects_wrong_length() {
        assert!(Image::new(2, 2, vec![0; 15]).is_none());
        assert!(Image::new(2, 2, vec![0; 16]).is_some());
        assert!(Image::new(0, 2, vec![]).is_none());
        assert!(Image::new(u32::MAX, u32::MAX, vec![]).is_none());
        let malformed = Image {
            width: 2,
            height: 2,
            pixels: vec![0; 4],
        };
        assert_eq!(malformed.validate(), Err(Refusal::ShapeMismatch("image")));
    }

    #[test]
    fn catalog_is_sorted_and_unique() {
        let rows = catalog();
        for pair in rows.windows(2) {
            assert!(
                pair[0].pf < pair[1].pf,
                "{} then {}",
                pair[0].pf,
                pair[1].pf
            );
        }
        for row in &rows {
            assert_eq!(lookup(row.pf).as_ref(), Some(row));
        }
    }

    #[test]
    fn intermediate_stages_do_not_enable_unresolved_final_effects() {
        // PF21026 moved to `kernels::beard`; PF90015 to `kernels::neck_paste`
        // once the neck colour tail was proven to be final RGB and the paste
        // back was proven to run without blending.
        for pf in [90001] {
            assert_eq!(lookup(pf).unwrap().access, Access::Refused,
                "a callable intermediate stage is not a recovered final effect: PF{pf}");
        }
        for pf in [8068, 8069, 8070] {
            assert_eq!(lookup(pf).unwrap().access, Access::Cpu,
                "the recovered local pixel stage must replace its old parameter-only entry");
        }
    }

    #[test]
    fn callable_rows_name_their_same_job_inputs() {
        for row in catalog() {
            if row.access != Access::Refused {
                assert!(
                    !row.inputs.is_empty(),
                    "PF{} claims a callable stage but lists no inputs",
                    row.pf
                );
            }
        }
    }

    #[test]
    fn catalog_json_lists_every_row_with_its_access() {
        let rows = catalog();
        let json = catalog_json();
        assert!(json.starts_with('[') && json.ends_with(']'));
        assert_eq!(json.matches("\"pf\":").count(), rows.len());
        for row in &rows {
            assert!(json.contains(&format!("\"pf\":{}", row.pf)), "missing PF{}", row.pf);
        }
        assert!(json.contains("\"access\":\"cpu\""));
        assert!(json.contains("\"access\":\"refused\""));
        assert!(!json.contains("\"\"\""));
    }
}

#[cfg(all(test, feature = "external-fixtures"))]
#[path = "../tests/support/mod.rs"]
mod external_fixtures;
