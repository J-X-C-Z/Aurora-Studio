//! External quality-bench CLI.
//!
//! Usage (requires the `external-fixtures` feature):
//!
//! ```text
//! cargo run -p aurora-engine --features external-fixtures --example quality_bench -- \
//!     --manifest C:\fixtures\manifest.json [--out C:\fixtures\report.json]
//! ```
//!
//! The manifest references local images by absolute path; see
//! `tests/README-quality-bench.md`. Without the feature this binary compiles
//! but exits non-zero with an explanation, so the default build stays
//! hermetic.

#[path = "../tests/support/quality_metrics.rs"]
mod metrics;

#[cfg(not(feature = "external-fixtures"))]
fn main() {
    eprintln!("quality_bench 是外部 fixture 工具，默认构建不包含 manifest 执行逻辑。");
    eprintln!(
        "请使用: cargo run -p aurora-engine --features external-fixtures --example quality_bench -- --manifest <path> [--out report.json]"
    );
    std::process::exit(2);
}

#[cfg(feature = "external-fixtures")]
fn main() {
    if let Err(err) = run() {
        eprintln!("quality_bench: {err}");
        std::process::exit(1);
    }
}

#[cfg(feature = "external-fixtures")]
#[derive(serde::Serialize, Default)]
struct BenchReport {
    manifest: String,
    samples: Vec<SampleReport>,
    failures: usize,
}

#[cfg(feature = "external-fixtures")]
#[derive(serde::Serialize, Default)]
struct SampleReport {
    name: String,
    image: String,
    colorchecker: Option<metrics::ColorCheckerReport>,
    colorchecker_ok: Option<bool>,
    shadow_snr: Option<metrics::SnrReport>,
    mtf50: Option<metrics::MtfReport>,
    false_color: Option<metrics::FalseColorReport>,
    diff: Option<metrics::DiffStats>,
    raw_cache: Option<RawCacheReport>,
    errors: Vec<String>,
}

#[cfg(feature = "external-fixtures")]
#[derive(serde::Serialize)]
struct RawCacheReport {
    path: String,
    long_side: u32,
    width: u32,
    height: u32,
    cold_equals_hit: bool,
}

#[cfg(feature = "external-fixtures")]
fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let mut manifest_path: Option<String> = None;
    let mut out_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--manifest" => {
                i += 1;
                manifest_path = Some(
                    args.get(i)
                        .ok_or("--manifest 需要一个路径参数")?
                        .clone(),
                );
            }
            "--out" => {
                i += 1;
                out_path = Some(args.get(i).ok_or("--out 需要一个路径参数")?.clone());
            }
            "--help" | "-h" => {
                println!(
                    "用法: quality_bench --manifest <manifest.json> [--out report.json]\n\
                     需要以 --features external-fixtures 构建。"
                );
                return Ok(());
            }
            other => return Err(format!("未知参数: {other}")),
        }
        i += 1;
    }
    let manifest_path = manifest_path.ok_or("缺少 --manifest <manifest.json>")?;
    let manifest = metrics::load_manifest(std::path::Path::new(&manifest_path))?;
    if manifest.samples.is_empty() {
        return Err("manifest 中没有 samples".to_string());
    }

    let cc_threshold = manifest
        .colorchecker_threshold
        .unwrap_or(metrics::DEFAULT_COLORCHECKER_THRESHOLD);
    let fc_threshold = manifest
        .false_color_chroma
        .unwrap_or(metrics::DEFAULT_FALSE_COLOR_CHROMA);

    let mut report = BenchReport {
        manifest: manifest_path.clone(),
        samples: Vec::with_capacity(manifest.samples.len()),
        failures: 0,
    };

    for sample in &manifest.samples {
        println!("\n== {} ==", sample.name);
        println!("  image: {}", sample.image);

        let mut entry = SampleReport {
            name: sample.name.clone(),
            image: sample.image.clone(),
            ..Default::default()
        };

        let loaded = match metrics::LoadedImage::load(std::path::Path::new(&sample.image)) {
            Ok(image) => {
                println!("  size: {}x{}", image.width(), image.height());
                Some(image)
            }
            Err(err) => {
                entry.errors.push(err.clone());
                None
            }
        };
        let rgba8 = loaded.as_ref().map(|image| image.to_rgba8());

        // ColorChecker.
        if let (Some(cfg), Some(img)) = (&sample.colorchecker, rgba8.as_ref()) {
            match metrics::colorchecker(img, cfg) {
                Ok(cc) => {
                    println!(
                        "  ColorChecker: {} patches, mean ΔE2000={}, max ΔE2000={} (threshold {:.2})",
                        cc.patches.len(),
                        fmt_opt(cc.mean_delta_e, 3),
                        fmt_opt(cc.max_delta_e, 3),
                        cc_threshold
                    );
                    for patch in &cc.patches {
                        println!(
                            "    patch {:>2} @({}, {}) median RGB {:?} Lab [{:.2}, {:.2}, {:.2}] ΔE={}",
                            patch.index,
                            patch.center[0],
                            patch.center[1],
                            patch.median_rgb,
                            patch.measured_lab[0],
                            patch.measured_lab[1],
                            patch.measured_lab[2],
                            fmt_opt(patch.delta_e, 3)
                        );
                    }
                    entry.colorchecker_ok = cc.max_delta_e.map(|max| max <= cc_threshold);
                    if entry.colorchecker_ok == Some(false) {
                        println!("    -> 超过阈值 {cc_threshold:.2}");
                    }
                    entry.colorchecker = Some(cc);
                }
                Err(err) => entry.errors.push(format!("colorchecker: {err}")),
            }
        }

        // Shadow SNR.
        if let (Some(roi), Some(img)) = (sample.shadow_roi, rgba8.as_ref()) {
            match metrics::shadow_snr(img, metrics::Roi::from(roi)) {
                Ok(snr) => {
                    println!(
                        "  Shadow SNR: mean={:.6} std={:.6} SNR={} dB ({} px)",
                        snr.mean,
                        snr.std,
                        fmt_opt(snr.snr_db, 2),
                        snr.pixels
                    );
                    entry.shadow_snr = Some(snr);
                }
                Err(err) => entry.errors.push(format!("shadow_snr: {err}")),
            }
        }

        // MTF50.
        if let (Some(roi), Some(img)) = (sample.edge_roi, rgba8.as_ref()) {
            match metrics::mtf50(img, metrics::Roi::from(roi)) {
                Ok(mtf) => {
                    println!(
                        "  MTF50: {:.4} c/px (normal {:.2} deg, LSF {}, bin {:.2} px)",
                        mtf.mtf50_cycles_per_px,
                        mtf.edge_normal_deg,
                        mtf.lsf_len,
                        mtf.esf_step_px
                    );
                    entry.mtf50 = Some(mtf);
                }
                Err(err) => entry.errors.push(format!("mtf50: {err}")),
            }
        }

        // False colour.
        if let (Some(roi), Some(img)) = (sample.neutral_roi, rgba8.as_ref()) {
            match metrics::false_color(img, metrics::Roi::from(roi), fc_threshold) {
                Ok(fc) => {
                    println!(
                        "  False color: ratio={:.4} P95={:.2} P99={:.2} max={:.2} (threshold {:.2})",
                        fc.ratio, fc.p95, fc.p99, fc.max, fc.threshold
                    );
                    entry.false_color = Some(fc);
                }
                Err(err) => entry.errors.push(format!("false_color: {err}")),
            }
        }

        // CPU vs OpenCL (reference vs candidate).
        match (&sample.reference, &sample.candidate) {
            (Some(reference), Some(candidate)) => {
                let reference_img = metrics::LoadedImage::load(std::path::Path::new(reference));
                let candidate_img = metrics::LoadedImage::load(std::path::Path::new(candidate));
                match (reference_img, candidate_img) {
                    (Ok(a), Ok(b)) => match a.diff(&b) {
                        Ok(diff) => {
                            println!(
                                "  CPU/GPU diff: max={:.3} mean={:.4} P99={:.3} >1/255={:.4}% >2/255={:.4}% ({} samples)",
                                diff.max,
                                diff.mean,
                                diff.p99,
                                diff.over_1_255 * 100.0,
                                diff.over_2_255 * 100.0,
                                diff.samples
                            );
                            entry.diff = Some(diff);
                        }
                        Err(err) => entry.errors.push(format!("diff: {err}")),
                    },
                    (Err(err), _) => entry.errors.push(format!("reference: {err}")),
                    (_, Err(err)) => entry.errors.push(format!("candidate: {err}")),
                }
            }
            (Some(_), None) => entry
                .errors
                .push("提供了 reference 但缺少 candidate".to_string()),
            (None, Some(_)) => entry
                .errors
                .push("提供了 candidate 但缺少 reference".to_string()),
            (None, None) => {}
        }

        // RAW cold decode vs cache hit, with a scratch cache directory.
        if let Some(raw) = &sample.raw {
            let long_side = sample.raw_long_side.unwrap_or(2560);
            match raw_cache_check(raw, long_side) {
                Ok(summary) => {
                    println!(
                        "  RAW cache: {}x{} cold==hit={} ({})",
                        summary.width, summary.height, summary.cold_equals_hit, raw
                    );
                    entry.raw_cache = Some(summary);
                }
                Err(err) => entry.errors.push(format!("raw_cache: {err}")),
            }
        }

        for err in &entry.errors {
            eprintln!("  error: {err}");
        }
        let failed = !entry.errors.is_empty() || entry.colorchecker_ok == Some(false);
        if failed {
            report.failures += 1;
        }
        report.samples.push(entry);
    }

    if let Some(out) = out_path {
        let json = serde_json::to_string_pretty(&report)
            .map_err(|err| format!("序列化报告失败: {err}"))?;
        std::fs::write(&out, json)
            .map_err(|err| format!("写入报告 {} 失败: {err}", out))?;
        println!("\nJSON 报告已写入 {out}");
    }

    println!(
        "\n完成: {} 个样本，{} 个失败",
        report.samples.len(),
        report.failures
    );
    if report.failures > 0 {
        return Err(format!("{} 个样本未通过阈值或发生错误", report.failures));
    }
    Ok(())
}

#[cfg(feature = "external-fixtures")]
fn fmt_opt(value: Option<f64>, decimals: usize) -> String {
    match value {
        Some(v) => format!("{:.*}", decimals, v),
        None => "n/a".to_string(),
    }
}

#[cfg(feature = "external-fixtures")]
fn raw_cache_check(raw: &str, long_side: u32) -> Result<RawCacheReport, String> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "aurora-quality-bench-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir)
        .map_err(|err| format!("创建临时缓存目录 {} 失败: {err}", dir.display()))?;

    let path = std::path::Path::new(raw);
    let result = (|| {
        let cold = aurora_engine::raw_cache::get_or_decode(path, &dir, long_side)
            .map_err(|err| format!("冷解码失败: {err}"))?;
        let hit = aurora_engine::raw_cache::get_or_decode(path, &dir, long_side)
            .map_err(|err| format!("缓存命中解码失败: {err}"))?;
        Ok(RawCacheReport {
            path: raw.to_string(),
            long_side,
            width: cold.width,
            height: cold.height,
            cold_equals_hit: cold.pixels == hit.pixels,
        })
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}
