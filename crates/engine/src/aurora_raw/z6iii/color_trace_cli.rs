//! `z6iii-color-trace`: per-stage dump of the C1 colour chain.
//!
//! Decodes a Z6III HE NEF to linear camera RGB16, runs every pixel through
//! [`Z6iiiColorStage`] and writes one 16-bit TIFF per stage so the stage that
//! introduces a delta against Capture One can be isolated.
//!
//! Usage:
//!   z6iii-color-trace [--iso=N] <input.nef> <camera.icm> <curve.fcrv> <srgb.icc> <outdir>
//!
//! `--iso=N` enables the output-domain BandingSuppression the product applies
//! after the colour stage. Without it the stage stays off, so the dump is the
//! pre-banding output.

#[cfg(windows)]
fn main() -> anyhow::Result<()> {
    use anyhow::{Context, bail};
    use aurora_engine::z6iii::Z6iiiColorStage;
    use std::{fs, path::{Path, PathBuf}};
    use tiff::encoder::{colortype::RGB16, TiffEncoder};

    let iso = std::env::args()
        .find_map(|arg| arg.strip_prefix("--iso=").and_then(|value| value.parse::<u32>().ok()));
    let args: Vec<_> = std::env::args_os()
        .skip(1)
        .filter(|arg| !arg.to_string_lossy().starts_with("--"))
        .map(PathBuf::from)
        .collect();
    let [nef, camera, curve, srgb, outdir] = args.as_slice() else {
        bail!("Usage: z6iii-color-trace [--iso=N] <input.nef> <camera.icm> <curve.fcrv> <srgb.icc> <outdir>");
    };

    let read = |path: &Path| fs::read(path).with_context(|| format!("read {}", path.display()));
    let stage = Z6iiiColorStage::from_bytes(&read(camera)?, Some(&read(srgb)?), Some(&read(curve)?))
        .context("build Z6iiiColorStage")?;

    let (width, height, linear) = aurora_engine::aurora_raw::decode_he_rgba16(nef)
        .context("Aurora Raw decode (linear camera RGBA16)")?;
    let (width, height) = (width as usize, height as usize);
    anyhow::ensure!(linear.len() == width * height * 4, "unexpected RGBA16 length");

    let banding = iso.and_then(|iso| {
        let model = "NIKON Z 6III";
        let range = aurora_engine::c1_banding::range_for_frame(None, Some(model))?;
        let amount = if range.is_adjustable() {
            range.default
        } else {
            aurora_engine::c1_banding::default_for_iso(model, iso)?
        };
        let params = aurora_engine::c1_banding::params_for_amount(amount);
        let (left, right, packed) = aurora_engine::c1_banding::vendor_buffers()?;
        let tables = aurora_engine::c1_banding::Tables::new([&left[..], &right[..], &packed[..]])?;
        Some((params, tables))
    });
    if banding.is_some() {
        println!("output-domain BandingSuppression enabled for ISO {iso:?}");
    }

    fs::create_dir_all(outdir).with_context(|| format!("create {}", outdir.display()))?;
    let mut stages: [(&str, Vec<u16>); 5] = [
        ("s1_linear", vec![0; width * height * 3]),
        ("s2_matrix", vec![0; width * height * 3]),
        ("s3_film", vec![0; width * height * 3]),
        ("s4_camera", vec![0; width * height * 3]),
        ("s5_srgb", vec![0; width * height * 3]),
    ];

    for (index, pixel) in linear.chunks_exact(4).enumerate() {
        let rgb = [
            f32::from(pixel[0]) / 65535.0,
            f32::from(pixel[1]) / 65535.0,
            f32::from(pixel[2]) / 65535.0,
        ];
        let trace = stage.trace(rgb);
        let mut output = trace.output;
        if let Some((params, tables)) = &banding {
            let mut codes = output.map(|value| (value.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16);
            tables.apply_pixel(&mut codes, (index % width) as i32, (index / width) as i32, *params);
            output = codes.map(|code| f32::from(code) / 65535.0);
        }
        for (slot, values) in [
            (0, trace.linear),
            (1, trace.matrix),
            (2, trace.film),
            (3, trace.camera),
            (4, output),
        ] {
            let dst = index * 3;
            stages[slot].1[dst] = code(values[0]);
            stages[slot].1[dst + 1] = code(values[1]);
            stages[slot].1[dst + 2] = code(values[2]);
        }
    }

    for (name, pixels) in &stages {
        let path = outdir.join(format!("{name}.tif"));
        let file = fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;
        let mut encoder = TiffEncoder::new(std::io::BufWriter::new(file))?;
        let mut image = encoder.new_image::<RGB16>(width as u32, height as u32)?;
        if *name == "s5_srgb" {
            image.encoder().write_tag(tiff::tags::Tag::IccProfile, &read(srgb)?[..])?;
        }
        image.write_data(pixels)?;
        println!("wrote {} ({width}x{height})", path.display());
    }
    Ok(())
}

/// Float stage value -> 16-bit code, the same clamp+truncate the export uses.
#[cfg(windows)]
fn code(value: f32) -> u16 {
    (value.clamp(0.0, 1.0) * 65535.0) as u16
}

#[cfg(not(windows))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("z6iii-color-trace requires Windows and an OpenCL GPU")
}
