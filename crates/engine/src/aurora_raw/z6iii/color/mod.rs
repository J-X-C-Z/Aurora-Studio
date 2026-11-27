#![forbid(unsafe_code)]

pub mod legacy_lut;
pub mod camera_clut;
pub mod icc_source;
pub mod pipeline;
pub mod coupled;
pub mod exposure;
pub mod z6iii_noise;
pub mod output;
mod icc_dst;
pub mod icc_grid;
mod icc_working;
pub use icc_working::{is_id10_profile, working_rgb_table};
mod legacy_fcrv;
pub use legacy_fcrv::film_settings;

pub(super) fn film_nodes(bytes: &[u8]) -> Option<Vec<[f32; 3]>> {
    let nodes = legacy_fcrv::primary_nodes(bytes)?;
    if nodes.is_empty() {
        return Some(vec![[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]);
    }
    (2..=16).contains(&nodes.len()).then(|| nodes.into_iter().map(|node| node.map(|value| value as f32)).collect())
}

const Z6III_PRE_MATRIX: [[f32; 3]; 3] = [
    [1.08, -0.07, -0.01],
    [-0.03, 1.04, -0.01],
    [-0.03, -0.07, 1.10],
];

const IDENTITY_PRE_MATRIX: [[f32; 3]; 3] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

fn linear_film_nodes() -> Vec<[f32; 3]> {
    vec![[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
}

/// C1 post-demosaic color stage: 3x3 pre-matrix, cubic Hermite film curve,
/// highlight/hue coupling, camera transfer, and 49^3 Camera -> sRGB CLUT.
#[derive(Clone)]
pub struct Z6iiiColorStage {
    pre_matrix: [[f32; 3]; 3],
    clut: std::sync::Arc<output::OutputClut>,
    film: std::sync::Arc<Vec<[f32; 3]>>,
}

impl Z6iiiColorStage {
    pub fn from_bytes(
        camera_icc: &[u8],
        srgb_icc: Option<&[u8]>,
        curve_bytes: Option<&[u8]>,
    ) -> Option<Self> {
        let clut = std::sync::Arc::new(output::OutputClut::from_camera49_srgb(camera_icc, srgb_icc)?);
        let film = std::sync::Arc::new(match curve_bytes {
            Some(bytes) => film_nodes(bytes)?,
            None => linear_film_nodes(),
        });
        Some(Self { pre_matrix: Z6III_PRE_MATRIX, clut, film })
    }

    pub fn from_names(icc_profile: Option<&str>, film_curve: Option<&str>) -> Option<Self> {
        Self::from_names_for_camera(icc_profile, film_curve, true)
    }

    pub fn from_names_for_camera(
        icc_profile: Option<&str>,
        film_curve: Option<&str>,
        use_z6iii_pre_matrix: bool,
    ) -> Option<Self> {
        let pre_matrix = if use_z6iii_pre_matrix {
            Z6III_PRE_MATRIX
        } else {
            IDENTITY_PRE_MATRIX
        };
        Self::from_names_with_pre_matrix(icc_profile, film_curve, pre_matrix)
    }

    pub fn from_names_with_pre_matrix(
        icc_profile: Option<&str>,
        film_curve: Option<&str>,
        pre_matrix: [[f32; 3]; 3],
    ) -> Option<Self> {
        let profile = crate::icc::profile(icc_profile?)?;
        let camera_path = profile.path.clone()?;
        let srgb_path = crate::icc::profile("sRGB Color Space Profile")
            .or_else(|| crate::icc::profile(crate::icc::DEFAULT_PROFILE))
            .and_then(|profile| profile.path.clone());
        let clut = cached_srgb_clut(camera_path, srgb_path)?;
        let film = cached_film_nodes(film_curve)
            .unwrap_or_else(|| std::sync::Arc::new(linear_film_nodes()));
        Some(Self { pre_matrix, clut, film })
    }

    pub fn develop_srgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        self.trace(rgb).output
    }

    /// Read-only accessors for the optional CUDA develop path. Read-only
    /// copies of the same constants the CPU stage evaluates; no pixel math or
    /// public signature changes.
    pub(crate) fn pre_matrix(&self) -> [[f32; 3]; 3] {
        self.pre_matrix
    }

    pub(crate) fn film_nodes(&self) -> &[[f32; 3]] {
        &self.film
    }

    pub(crate) fn clut_nodes(&self) -> &[[u16; 3]] {
        self.clut.nodes()
    }

    /// Intermediate values of the colour stage for one pixel, in chain order.
    ///
    /// `linear` is the post-black/white-balance camera value, `matrix` the
    /// 3x3 pre-matrix, `film` the `.fcrv` node evaluation, `coupled` the
    /// highlight/hue coupling blend, `camera` the camera transfer encode and
    /// `output` the 49^3 Camera -> working-RGB CLUT. Same per-pixel math as
    /// [`Self::encode_camera_rgb`] + [`Self::apply_camera_icc`]; exposed for the
    /// `z6iii-color-trace` diagnostic binary.
    #[doc(hidden)]
    pub fn trace(&self, rgb: [f32; 3]) -> StageTrace {
        let linear = rgb.map(|value| value.max(0.0));
        let matrix = self.pre_matrix.map(|[red, green, blue]| {
            (red * linear[0] + green * linear[1] + blue * linear[2]).max(0.0)
        });
        let film = matrix.map(|value| coupled::film_at(&self.film, value));
        let coupled = coupled::couple_float(matrix, film);
        let camera = coupled.map(coupled::encode_camera);
        let output = self.apply_camera_icc(camera);
        StageTrace { linear, matrix, film, coupled, camera, output }
    }

    pub(crate) fn encode_camera_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        let norm = rgb.map(|value| value.max(0.0));
        let before = self.pre_matrix.map(|[red, green, blue]| {
            (red * norm[0] + green * norm[1] + blue * norm[2]).max(0.0)
        });
        let curved = before.map(|value| coupled::film_at(&self.film, value));
        let coupled = coupled::couple_float(before, curved);
        coupled.map(coupled::encode_camera)
    }

    pub(crate) fn apply_camera_icc(&self, encoded: [f32; 3]) -> [f32; 3] {
        self.clut.sample_srgb(encoded).unwrap_or([0.0; 3])
    }
}

/// One pixel's values at every stage of [`Z6iiiColorStage`].
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct StageTrace {
    pub linear: [f32; 3],
    pub matrix: [f32; 3],
    pub film: [f32; 3],
    pub coupled: [f32; 3],
    pub camera: [f32; 3],
    pub output: [f32; 3],
}

fn cached_srgb_clut(
    camera_path: std::path::PathBuf,
    srgb_path: Option<std::path::PathBuf>,
) -> Option<std::sync::Arc<output::OutputClut>> {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex, OnceLock};

    type ClutCache = Mutex<HashMap<(PathBuf, Option<PathBuf>), Option<Arc<output::OutputClut>>>>;
    static CACHE: OnceLock<ClutCache> = OnceLock::new();

    let key = (camera_path, srgb_path);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return hit.clone();
    }
    let built = std::fs::read(&key.0).ok().and_then(|camera_bytes| {
        let srgb_bytes = key.1.as_deref().and_then(|path| std::fs::read(path).ok());
        output::OutputClut::from_camera49_srgb(&camera_bytes, srgb_bytes.as_deref()).map(Arc::new)
    });
    cache.lock().unwrap_or_else(|e| e.into_inner()).insert(key, built.clone());
    built
}

fn cached_film_nodes(film_curve: Option<&str>) -> Option<std::sync::Arc<Vec<[f32; 3]>>> {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex, OnceLock};

    type FilmCache = Mutex<HashMap<(PathBuf, Option<String>), Option<Arc<Vec<[f32; 3]>>>>>;
    static CACHE: OnceLock<FilmCache> = OnceLock::new();

    let key = (crate::filmcurve::curve_dir(), film_curve.map(str::to_owned));
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return hit.clone();
    }
    let built = match key.1.as_deref() {
        Some(name) => std::fs::read(key.0.join(format!("{name}.fcrv")))
            .ok()
            .and_then(|bytes| film_nodes(&bytes))
            .map(Arc::new),
        None => Some(Arc::new(linear_film_nodes())),
    };
    cache.lock().unwrap_or_else(|e| e.into_inner()).insert(key, built.clone());
    built
}

pub fn prepare_z6iii_global_white_balance(mut channels: [f32; 3]) -> [f32; 3] {
    let minimum = channels.into_iter().fold(f32::INFINITY, f32::min);
    if minimum > 0.0 {
        channels = channels.map(|channel| channel / minimum);
    }
    channels.map(|channel| if channel.is_nan() { 1.0 } else { channel.clamp(0.0001, 1000.0) })
}

pub fn prepare_undo_white_balance(gain: [f32; 3], engine: u32) -> Option<[f32; 3]> {
    if !gain.into_iter().all(|channel| channel.is_finite() && channel > 0.0) {
        return None;
    }
    let common = if engine == 9 {
        gain.into_iter().fold(1.0_f32, f32::max)
    } else {
        1.0
    };
    let effective_gain = gain.map(|channel| channel * common);
    effective_gain.into_iter().all(f32::is_finite).then_some(effective_gain)
}

pub fn undo_white_balance(rgb: [f32; 3], gain: [f32; 3], engine: u32) -> Option<[f32; 3]> {
    let inverse = prepare_undo_white_balance(gain, engine)?.map(|channel| 1.0 / channel);
    let output = std::array::from_fn(|channel| rgb[channel] * inverse[channel]);
    output.into_iter().all(f32::is_finite).then_some(output)
}

/// Finite RGB code values. Models the truncating Convert32fTo16 leaf.
pub fn truncate_to_rgb16(rgb: [f32; 3]) -> [u16; 3] {
    rgb.map(|code| code.clamp(0.0, 65535.0).trunc() as u16)
}

/// Unit-range device RGB after legacy color management, before camera ICC.
pub fn normalize_rgb16(rgb: [u16; 3]) -> [f32; 3] {
    rgb.map(|code| f32::from(code) * f32::from_bits(0x3780_0080))
}

pub fn legacy_color_management_rgb16(
    rgb: [f32; 3],
    matrix: &[[f32; 3]; 3],
    scale: f32,
    tables: Option<[&[u16; 65536]; 3]>,
) -> [u16; 3] {
    let adjusted = matrix.map(|row| (rgb[0] * row[0] + rgb[1] * row[1]) + rgb[2] * row[2]);
    let adjusted = if scale < 1.0 { adjusted.map(|code| code * scale) } else { adjusted };
    let quantized = truncate_to_rgb16(adjusted);
    match tables {
        Some(tables) => std::array::from_fn(|channel| tables[channel][usize::from(quantized[channel])]),
        None => quantized,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_domain_handoff() {
        let undone = undo_white_balance([65535.75, -4.0, 262144.0], [2.0, 0.5, 4.0], 7);
        assert_eq!(undone, Some([32767.875, -8.0, 65536.0]));
        let rgb16 = truncate_to_rgb16(undone.unwrap());
        assert_eq!(rgb16, [32767, 0, 65535]);
        assert_eq!(normalize_rgb16(rgb16).map(f32::to_bits), [0x3eff_ff00, 0, 0x3f80_0000]);
        assert_eq!(undo_white_balance([1.0; 3], [2.0, 0.0, 4.0], 9), None);
        assert_eq!(prepare_undo_white_balance([2.0, 0.5, 4.0], 9), Some([8.0, 2.0, 16.0]));
        assert_eq!(prepare_undo_white_balance([0.25, 0.5, 0.75], 9), Some([0.25, 0.5, 0.75]));
    }

}
