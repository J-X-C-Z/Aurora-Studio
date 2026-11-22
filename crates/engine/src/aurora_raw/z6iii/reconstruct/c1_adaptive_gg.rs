//! AdaptiveGG2012 variant 2. Input samples are right-aligned Nikon HE codes.
use super::{mirror_once, NikonCrop};

fn filtered(center: u16, neighbors: [u16; 4], threshold: f32, black: f32) -> u16 {
    let p = f32::from(center);
    let signal = (p - black).max(0.0);
    if signal >= 4.0 * threshold {
        return center;
    }
    let base = threshold * 0.1 + signal * 0.25;
    let n = neighbors.map(f32::from);
    let w = n.map(|q| 1.0 / (base + (p - q).abs()));
    let sum = n[0] * w[0] + n[1] * w[1] + n[2] * w[2] + n[3] * w[3];
    let mean = sum / (w[0] + w[1] + w[2] + w[3]);
    let dark = mean * 0.5 + p * 0.5;
    if signal < threshold {
        dark as u16
    } else {
        let mix = (signal - threshold) / (3.0 * threshold);
        let blend = dark * (1.0 - mix) + p * mix;
        blend as u16
    }
}

pub(super) fn prepare(raw: &[u16], width: usize, height: usize, crop: NikonCrop,
                      threshold: f32, black: f32) -> Vec<u16> {
    let mut output: Vec<_> = raw.iter().map(|v| v << 2).collect();
    // Match the active-crop mirror used by the subsequent Bayer reconstruction.
    // Precompute the one-pixel halo so tile seams never become image boundaries.
    let columns: Vec<_> = (0..crop.width).map(|x| {
        [-1, 1].map(|d| crop.left + mirror_once(x as i64 + d, crop.width))
    }).collect();
    debug_assert_eq!(height * width, raw.len());
    for y in 0..crop.height {
        let sy = crop.top + y;
        let [up, down] = [-1, 1].map(|d|
            (crop.top + mirror_once(y as i64 + d, crop.height)) * width);
        for x in ((1 ^ ((sy ^ crop.left) & 1))..crop.width).step_by(2) {
            let [left, right] = columns[x];
            let i = sy * width + crop.left + x;
            let neighbors = [raw[up+left], raw[up+right], raw[down+left], raw[down+right]];
            output[i] = filtered(output[i], neighbors.map(|v| v << 2), threshold, black);
        }
    }
    output
}
