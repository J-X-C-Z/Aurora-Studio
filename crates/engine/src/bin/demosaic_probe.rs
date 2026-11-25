//! Compares libraw demosaic filters on one RAW file with quality numbers.
//!
//! Usage: `cargo run --release -p aurora-engine --bin demosaic_probe [raw] [out_dir]`

use std::path::PathBuf;
use std::time::Instant;

use image::RgbaImage;
use aurora_engine::raw::{decode, set_demosaic};

/// Filters this libraw build actually implements (13+ silently fall back to AHD).
const FILTERS: [(i32, &str); 6] =
    [(0, "linear"), (1, "VNG"), (2, "PPG"), (3, "AHD"), (11, "DCB"), (12, "AFD")];

struct Metrics {
    /// Chroma spread inside the top 5% Laplacian pixels, in code values.
    false_color: f32,
    /// Mean green-channel gradient energy, scaled by 1e3.
    detail: f32,
    /// Mean |R-G| on the top 5% gradient pixels, in code values.
    edge_rg: f32,
    /// Mean |B-G| on the same pixels, in code values.
    edge_bg: f32,
}

fn code(bytes: &[u8], i: usize) -> (f32, f32, f32) {
    let p = &bytes[i * 4..i * 4 + 3];
    (p[0] as f32, p[1] as f32, p[2] as f32)
}

/// |4Y - neighbours|, in code values.
fn laplacian(lum: &[f32], w: usize, i: usize) -> f32 {
    (4.0 * lum[i] - lum[i - 1] - lum[i + 1] - lum[i - w] - lum[i + w]).abs()
}

/// Luma gradient magnitude, in code values.
fn gradient(lum: &[f32], w: usize, i: usize) -> f32 {
    let dx = (lum[i + 1] - lum[i - 1]) * 0.5;
    let dy = (lum[i + w] - lum[i - w]) * 0.5;
    dx.hypot(dy)
}

/// 95th percentile of code-value responses, from a one-code-unit histogram.
fn top_five_percent(mut sample: impl FnMut(usize) -> f32, len: usize) -> f32 {
    let mut hist = vec![0u32; 1 << 16];
    for k in 0..len {
        hist[(sample(k) as usize).min((1 << 16) - 1)] += 1;
    }
    let target = (len as f64 * 0.95) as u32;
    let mut seen = 0u32;
    for (bin, count) in hist.iter().enumerate() {
        seen += count;
        if seen > target {
            return bin as f32;
        }
    }
    (1 << 16) as f32
}

fn analyse(img: &RgbaImage) -> Metrics {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bytes = img.as_raw();
    let lum: Vec<f32> = (0..w * h)
        .map(|i| {
            let (r, g, b) = code(bytes, i);
            0.2126 * r + 0.7152 * g + 0.0722 * b
        })
        .collect();

    let iw = w - 2;
    let interior = iw * (h - 2);
    // Index k walks the interior row by row; rows are contiguous in `lum`.
    let at = |k: usize| (k / iw + 1) * w + (k % iw + 1);
    let lap_thr = top_five_percent(|k| laplacian(&lum, w, at(k)), interior);
    let grad_thr = top_five_percent(|k| gradient(&lum, w, at(k)), interior);

    let mut detail = 0.0f64;
    let (mut sa, mut sa2, mut sb, mut sb2, mut n) = (0.0f64, 0.0, 0.0f64, 0.0, 0usize);
    let (mut sa_edge, mut sb_edge, mut n_edge) = (0.0f64, 0.0f64, 0usize);
    for k in 0..interior {
        let i = at(k);
        let (_, gl, _) = code(bytes, i - 1);
        let (_, gr, _) = code(bytes, i + 1);
        let (_, gd, _) = code(bytes, i - w);
        let (_, gu, _) = code(bytes, i + w);
        let (gx, gy) = ((gr - gl) * 0.5, (gu - gd) * 0.5);
        detail += f64::from(gx * gx + gy * gy);

        let (r, gc, b) = code(bytes, i);
        let (a, ch) = (r - gc, b - gc);
        let midtone = (40.0..=235.0).contains(&lum[i]);
        if midtone && laplacian(&lum, w, i) >= lap_thr {
            sa += f64::from(a);
            sa2 += f64::from(a * a);
            sb += f64::from(ch);
            sb2 += f64::from(ch * ch);
            n += 1;
        }
        if midtone && gradient(&lum, w, i) >= grad_thr {
            sa_edge += f64::from(a.abs());
            sb_edge += f64::from(ch.abs());
            n_edge += 1;
        }
    }
    let var = |sum: f64, sum2: f64, n: usize| {
        let mean = sum / n as f64;
        (sum2 / n as f64 - mean * mean).max(0.0)
    };
    let n = n.max(1);
    let n_edge = n_edge.max(1);
    Metrics {
        false_color: ((var(sa, sa2, n) + var(sb, sb2, n)) / 2.0).sqrt() as f32,
        detail: (detail / interior as f64 * 1e3) as f32,
        edge_rg: (sa_edge / n_edge as f64) as f32,
        edge_bg: (sb_edge / n_edge as f64) as f32,
    }
}

fn main() -> anyhow::Result<()> {
    if std::env::var_os("LUMEN_DEMOSAIC").is_some() {
        anyhow::bail!("unset LUMEN_DEMOSAIC: it overrides set_demosaic and would flatten the comparison");
    }
    let mut args = std::env::args().skip(1);
    let raw = PathBuf::from(
        args.next().unwrap_or_else(|| "/home/jxcz/agents/evidence/_DSC5673.NEF".into()),
    );
    let out = PathBuf::from(args.next().unwrap_or_else(|| "out".into()));
    std::fs::create_dir_all(&out)?;

    let _ = decode(&raw)?; // warm the page cache and libraw's decoder table

    let mut rows = Vec::new();
    for (quality, name) in FILTERS {
        set_demosaic(quality);
        let mut best = f64::MAX;
        let mut img = None;
        for _ in 0..3 {
            let start = Instant::now();
            let decoded = decode(&raw)?;
            best = best.min(start.elapsed().as_secs_f64() * 1e3);
            img = Some(decoded);
        }
        let img = img.unwrap();
        let metrics = analyse(&img);
        image::imageops::crop_imm(&img, img.width() / 3, img.height() / 3, 1000, 700)
            .to_image()
            .save(out.join(format!("crop-{name}.png")))?;
        rows.push((
            quality,
            name,
            best,
            img.width(),
            img.height(),
            metrics,
        ));
        println!(
            "{name:<7} q={quality:<2} {best:8.0} ms  false={:5.2}  detail={:6.2}  edge|R-G|={:5.2}  edge|B-G|={:5.2}",
            rows.last().unwrap().5.false_color,
            rows.last().unwrap().5.detail,
            rows.last().unwrap().5.edge_rg,
            rows.last().unwrap().5.edge_bg
        );
    }

    let mut csv = String::from("filter,quality,ms,width,height,false_color,detail,edge_rg,edge_bg\n");
    for (quality, name, ms, w, h, m) in &rows {
        csv.push_str(&format!(
            "{name},{quality},{ms:.0},{w},{h},{:.4},{:.4},{:.4},{:.4}\n",
            m.false_color, m.detail, m.edge_rg, m.edge_bg
        ));
    }
    std::fs::write(out.join("demosaic.csv"), &csv)?;
    let json = serde_json::json!({
        "raw": raw.display().to_string(),
        "rows": rows.iter().map(|(q, name, ms, w, h, m)| serde_json::json!({
            "filter": name, "quality": q, "ms": ms.round(), "width": w, "height": h,
            "false_color": m.false_color, "detail": m.detail,
            "edge_rg": m.edge_rg, "edge_bg": m.edge_bg,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(out.join("demosaic.json"), serde_json::to_string_pretty(&json)?)?;
    println!("\nwrote {}/demosaic.csv and demosaic.json", out.display());
    Ok(())
}

