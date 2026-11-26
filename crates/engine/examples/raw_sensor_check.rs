//! Check the sensor output of an explicitly supplied RAW, never search personal files.
//! Optional second argument is a NEW scratch file for independent byte-level A/B comparisons.
use anyhow::{Context, Result, ensure};
use aurora_engine::raw::{DecodeBackend, decode16_traced, quick_embedded_preview};
use std::{fs::OpenOptions, io::{BufWriter, Write}, path::PathBuf, time::Instant};

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let path = PathBuf::from(args.next().context("usage: raw_sensor_check FILE [NEW_RGBA16LE_DUMP]")?);
    let dump = args.next().map(PathBuf::from);
    ensure!(args.next().is_none(), "unexpected extra arguments");
    let start = Instant::now();
    let embedded = quick_embedded_preview(&path, 1200)?;
    let preview_ms = start.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();
    #[cfg(windows)] let before = aurora_portrait::cuda::driver_metrics();
    let (w, h, px, provenance) = decode16_traced(&path)
        .context("no full sensor decode; an embedded JPEG is not a sensor read")?;
    let sensor_ms = start.elapsed().as_secs_f64() * 1000.;
    ensure!(matches!(provenance.backend, DecodeBackend::LibRaw | DecodeBackend::AuroraCpuChain | DecodeBackend::AuroraZ6iiiOpenCl), "unexpected decode backend: {:?}", provenance.backend);
    ensure!(px.len() == w as usize * h as usize * 4, "bad RGBA16 plane");
    let stats=aurora_engine::raw::raw_decode_stats();
    let fused:u64=stats.iter().map(|(_,s)|s.timings.cuda_fused_frames).sum();
    let cpu_post:u64=stats.iter().map(|(_,s)|s.timings.cpu_highlight_aa_frames).sum();
    let opencl_post:u64=stats.iter().map(|(_,s)|s.timings.opencl_highlight_aa_frames).sum();
    #[cfg(windows)] let io = {
        let after=aurora_portrait::cuda::driver_metrics();
        serde_json::json!({"h2d_bytes":after.host_to_device_bytes-before.host_to_device_bytes,
            "d2h_bytes":after.device_to_host_bytes-before.device_to_host_bytes,
            "ptx_module_loads":after.ptx_module_load_calls-before.ptx_module_load_calls})
    };
    #[cfg(not(windows))] let io=serde_json::Value::Null;
    let (mut min, mut max, mut hash) = (u16::MAX, 0_u16, 0xcbf29ce484222325_u64);
    for pixel in px.chunks_exact(4) {
        for &v in &pixel[..3] { min = min.min(v); max = max.max(v); }
        for v in pixel { for b in v.to_le_bytes() { hash = (hash ^ b as u64).wrapping_mul(0x100000001b3); } }
    }
    ensure!(max > min && max > 255, "unexpected sensor dynamic range: {min}..{max}");
    if let Some(file) = dump {
        let mut writer = BufWriter::new(OpenOptions::new().create_new(true).write(true).open(file)?);
        let mut bytes = Vec::with_capacity(65536);
        for block in px.chunks(32768) {
            bytes.clear();
            bytes.extend(block.iter().flat_map(|v| v.to_le_bytes()));
            writer.write_all(&bytes)?;
        }
        writer.flush()?;
    }
    println!("{}", serde_json::json!({"width":w, "height":h, "sensor_ms":sensor_ms,
        "preview_ms":preview_ms, "preview_width":embedded.width(), "preview_height":embedded.height(),
        "cuda_fused_frames":fused,"cpu_highlight_aa_frames":cpu_post,"opencl_highlight_aa_frames":opencl_post,"cuda_io":io,"fnv64":format!("{hash:016x}"), "minimum":min, "maximum":max, "provenance":provenance}));
    Ok(())
}
