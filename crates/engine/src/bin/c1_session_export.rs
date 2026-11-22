use std::{env, path::Path};

use anyhow::{bail, Result};

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let (Some(source), Some(target)) = (args.next(), args.next()) else {
        bail!("usage: c1_session_export <catalog-or-session> <new-session-folder>");
    };
    if args.next().is_some() {
        bail!("usage: c1_session_export <catalog-or-session> <new-session-folder>");
    }
    let import = aurora_engine::C1Import::open(Path::new(&source))?;
    let export = aurora_engine::export_session(&import, Path::new(&target))?;
    println!(
        "{} images, {} variants, {} linked, {} copied, {} masks copied",
        export.images,
        export.variants,
        export.media_linked,
        export.media_copied,
        export.masks_copied
    );
    println!("{}", export.session_dir.display());
    Ok(())
}
