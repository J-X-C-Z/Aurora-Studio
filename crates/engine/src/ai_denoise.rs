//! Capture One AI Denoise (`RemoveNoise` / `Denoise::OnnxProvider`) interface
//! and tiled 16-bit <-> float32 preprocessing/blending pipeline.
//!
//! Matches the reversed `ImgCoreDll!0x1484D10` contract:
//! - normalize 16-bit samples to `[0.0, 1.0]` via `v / 65535.0`;
//! - run overlapping tiles with smooth cosine feathering at tile borders;
//! - clamp and convert back to 16-bit `[0, 65535]`.

use std::path::{Path, PathBuf};

/// Standard tile size and overlap halo for tiled AI denoise inference.
pub const DEFAULT_TILE: usize = 512;
pub const DEFAULT_OVERLAP: usize = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AiDenoiseStatus {
    Ready(PathBuf),
    ModelUnavailable,
}

/// Probes `AURORA_AI_DENOISE_MODEL` and the local Capture One `AIModels` tree
/// for an ONNX denoise model without failing when absent.
pub fn probe_model() -> AiDenoiseStatus {
    if let Some(p) = std::env::var_os("AURORA_AI_DENOISE_MODEL").map(PathBuf::from) {
        if p.is_file() {
            return AiDenoiseStatus::Ready(p);
        }
    }
    for root in crate::bundled_resources::ai_model_roots() {
        for rel in [
            "AIModels/Common/Denoise/denoise.onnx",
            "AIModels/Common/Denoise/denoiseOracle.onnx",
            "AIModels/DirectML/denoiseFP16.onnx",
            "Common/Denoise/denoise.onnx",
            "Common/Denoise/denoiseOracle.onnx",
            "DirectML/denoiseFP16.onnx",
        ] {
            let candidate = root.join(rel);
            if candidate.is_file() {
                return AiDenoiseStatus::Ready(candidate);
            }
        }
    }
    AiDenoiseStatus::ModelUnavailable
}

/// Converts interleaved RGB16 (`0..=65535`) to planar CHW float32 (`0.0..=1.0`)
/// as expected by `Denoise::OnnxProvider::Run`.
pub fn rgb16_to_chw_f32(rgb16: &[u16], width: usize, height: usize) -> Vec<f32> {
    let n = width * height;
    let mut out = vec![0.0f32; n * 3];
    let inv = 1.0 / 65535.0f32;
    for i in 0..n {
        let s = i * 3;
        out[i] = f32::from(rgb16[s]) * inv;
        out[n + i] = f32::from(rgb16[s + 1]) * inv;
        out[2 * n + i] = f32::from(rgb16[s + 2]) * inv;
    }
    out
}

/// Converts planar CHW float32 (`0.0..=1.0`) back to interleaved RGB16.
pub fn chw_f32_to_rgb16(chw: &[f32], width: usize, height: usize) -> Vec<u16> {
    let n = width * height;
    let mut out = vec![0u16; n * 3];
    for i in 0..n {
        let d = i * 3;
        out[d] = (chw[i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        out[d + 1] = (chw[n + i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        out[d + 2] = (chw[2 * n + i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    }
    out
}

/// Runs a caller-supplied tile inference closure `infer(tile_chw, tw, th) -> out_chw`
/// across the full image with feathered overlap blending, ensuring zero tile seams.
pub fn run_tiled_rgb16<F>(
    rgb16: &[u16],
    width: usize,
    height: usize,
    tile_size: usize,
    overlap: usize,
    mut infer: F,
) -> Option<Vec<u16>>
where
    F: FnMut(&[f32], usize, usize) -> Option<Vec<f32>>,
{
    if width == 0 || height == 0 || rgb16.len() != width * height * 3 {
        return None;
    }
    let tile = tile_size.max(16);
    let step = tile.saturating_sub(overlap).max(8);
    let mut accum = vec![[0.0f32; 3]; width * height];
    let mut weights = vec![0.0f32; width * height];

    let mut y0 = 0usize;
    loop {
        let th = tile.min(height - y0);
        let mut x0 = 0usize;
        loop {
            let tw = tile.min(width - x0);
            let mut sub = vec![0u16; tw * th * 3];
            for y in 0..th {
                let src = ((y0 + y) * width + x0) * 3;
                let dst = y * tw * 3;
                sub[dst..dst + tw * 3].copy_from_slice(&rgb16[src..src + tw * 3]);
            }
            let chw_in = rgb16_to_chw_f32(&sub, tw, th);
            let chw_out = infer(&chw_in, tw, th)?;
            if chw_out.len() != tw * th * 3 {
                return None;
            }
            let n = tw * th;
            for y in 0..th {
                let wy = border_weight(y, th, overlap, y0 > 0, y0 + th < height);
                for x in 0..tw {
                    let wx = border_weight(x, tw, overlap, x0 > 0, x0 + tw < width);
                    let w = (wx * wy).max(1e-4);
                    let idx = (y0 + y) * width + (x0 + x);
                    let ti = y * tw + x;
                    accum[idx][0] += chw_out[ti] * w;
                    accum[idx][1] += chw_out[n + ti] * w;
                    accum[idx][2] += chw_out[2 * n + ti] * w;
                    weights[idx] += w;
                }
            }
            if x0 + tw >= width {
                break;
            }
            x0 += step;
        }
        if y0 + th >= height {
            break;
        }
        y0 += step;
    }

    let mut out = vec![0u16; width * height * 3];
    for (i, (acc, &w)) in accum.iter().zip(weights.iter()).enumerate() {
        let inv = if w > 0.0 { 1.0 / w } else { 1.0 };
        let d = i * 3;
        out[d] = ((acc[0] * inv).clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        out[d + 1] = ((acc[1] * inv).clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        out[d + 2] = ((acc[2] * inv).clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    }
    Some(out)
}

#[inline]
fn border_weight(pos: usize, len: usize, overlap: usize, has_prev: bool, has_next: bool) -> f32 {
    if overlap == 0 {
        return 1.0;
    }
    let mut w = 1.0f32;
    if has_prev && pos < overlap {
        w = w.min((pos as f32 + 0.5) / overlap as f32);
    }
    if has_next && pos + overlap >= len {
        w = w.min(((len - 1 - pos) as f32 + 0.5) / overlap as f32);
    }
    w.clamp(0.0, 1.0)
}

/// Checks if a given model path exists and is readable.
pub fn is_model_ready(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_identity_inference_is_exact_without_seams() {
        let (w, h) = (80, 64);
        let mut src = vec![0u16; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                src[i] = ((x * 800) % 65535) as u16;
                src[i + 1] = ((y * 900) % 65535) as u16;
                src[i + 2] = (((x + y) * 400) % 65535) as u16;
            }
        }
        let out = run_tiled_rgb16(&src, w, h, 32, 8, |tile, _tw, _th| Some(tile.to_vec()))
            .expect("tiled identity inference must succeed");
        for (a, b) in src.iter().zip(out.iter()) {
            assert!(a.abs_diff(*b) <= 1, "seam drift detected: {a} vs {b}");
        }
    }
}
