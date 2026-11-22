//! Prints Capture One's segmentation models' tensor IO, then runs both on one
//! image and reports timings and mask statistics.
//!
//! ```text
//! LUMEN_AIMODELS=~/agents/evidence \
//!   cargo run -p lumen-engine --bin aimask_probe -- photo.jpg /tmp
//! ```

use std::time::Instant;

use anyhow::Result;

fn write_pgm(path: &str, w: usize, h: usize, data: &[f32]) {
    let mut bytes = format!("P5\n{w} {h}\n255\n").into_bytes();
    bytes.extend(data.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
    std::fs::write(path, bytes).expect("write pgm");
}

fn report(name: &str, w: usize, h: usize, data: &[f32], ms: f32, out: &str) {
    let (mut min, mut max, mut sum, mut over) = (f32::MAX, f32::MIN, 0.0, 0usize);
    for &v in data {
        min = min.min(v);
        max = max.max(v);
        sum += v;
        over += usize::from(v > 0.5);
    }
    let mean = sum / data.len().max(1) as f32;
    println!(
        "{name}: {w}x{h}, {ms:.0} ms, min={min:.3} max={max:.3} mean={mean:.3}, foreground={:.1}%",
        over as f32 / data.len().max(1) as f32 * 100.0
    );
    write_pgm(&format!("{out}/{name}.pgm"), w, h, data);
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "photo.jpg".into());
    let out_dir = args.next().unwrap_or_else(|| "/tmp".into());
    let image = image::open(&path)?.to_rgba8();
    let (w, h) = image.dimensions();
    println!("image {path}: {w}x{h}");

    for model in ["subjectMasking.onnx", "skinMasking.onnx"] {
        println!(
            "{model} provider: {}",
            aurora_engine::aimask::model_provider(model).unwrap_or("unavailable")
        );
        for line in aurora_engine::aimask::model_summary(model)? {
            println!("{line}");
        }
    }

    let rgba = image.as_raw();
    let start = Instant::now();
    let subject = aurora_engine::aimask::subject_mask(rgba, w, h)?;
    let subject_ms = start.elapsed().as_secs_f32() * 1000.0;
    report("subject", w as usize, h as usize, &subject, subject_ms, &out_dir);

    let start = Instant::now();
    let skin = aurora_engine::aimask::skin_mask(rgba, w, h)?;
    let skin_ms = start.elapsed().as_secs_f32() * 1000.0;
    report("skin", w as usize, h as usize, &skin, skin_ms, &out_dir);
    Ok(())
}

