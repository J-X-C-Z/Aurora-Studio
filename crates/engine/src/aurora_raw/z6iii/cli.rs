#[cfg(windows)]
fn main() -> anyhow::Result<()> {
    use anyhow::{Context, bail};
    use aurora_engine::z6iii::{ExportAssets, render};
    use std::{fs, io::{BufWriter, Write}, path::PathBuf};

    let args: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let [nef, camera, auto, linear, working, adobe, tables, output] = args.as_slice() else {
        bail!("Usage: z6iii-export <input.nef> <camera.icm> <curve.fcrv> <linear.fcrv> <working.icc> <adobe.icc> <banding-dir> <output.tif>");
    };
    let read = |path: &std::path::Path| fs::read(path).with_context(|| format!("read {}", path.display()));
    let assets = ExportAssets {
        camera_icc: &read(camera)?,
        curve: &read(auto)?,
        linear_curve: &read(linear)?,
        working_icc: &read(working)?,
        adobe_rgb_icc: &read(adobe)?,
        banding_tables: [&read(&tables.join("arg7.u8"))?, &read(&tables.join("arg8.u8"))?, &read(&tables.join("arg9.u32le"))?],
    };
    let frame = render(&read(nef)?, &assets)?;
    let file = fs::OpenOptions::new().write(true).create_new(true).open(output)
        .with_context(|| format!("create {} (will not overwrite)", output.display()))?;
    let mut writer = BufWriter::new(file);
    frame.write_tiff(&mut writer)?;
    writer.flush()?;
    Ok(())
}

#[cfg(not(windows))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("z6iii-export requires Windows and an OpenCL GPU")
}
