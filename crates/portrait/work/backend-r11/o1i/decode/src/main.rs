//! Decodes the two teeth atlases exactly like `processing::portrait_assets`:
//! `PngDecoder` -> `DynamicImage::from_decoder` -> `into_rgb8`.
//!
//! Writes `whiten.rgb` and `bright.rgb` (512*512*3 bytes each) next to o1i.

use image::{codecs::png::PngDecoder, DynamicImage, ImageDecoder};
use std::io::{BufReader, Seek};
use std::path::Path;

fn decode(path: &Path) -> Vec<u8> {
    let file = std::fs::File::open(path).expect("open png");
    let mut reader = BufReader::new(file);
    reader.rewind().ok();
    let decoder = PngDecoder::new(reader).expect("png decoder");
    let (width, height) = decoder.dimensions();
    assert_eq!((width, height), (512, 512), "{path:?} must be 512x512");
    DynamicImage::from_decoder(decoder)
        .expect("decode")
        .into_rgb8()
        .into_raw()
}

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../..");
    let teeth = root.join("assets/portrait/teeth");
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    for (name, file) in [("whiten.rgb", "Whiten.png"), ("bright.rgb", "Bright.png")] {
        let bytes = decode(&teeth.join(file));
        assert_eq!(bytes.len(), 512 * 512 * 3, "{file} shape");
        std::fs::write(out.join(name), &bytes).expect("write raw");
        println!("{name}: {} bytes", bytes.len());
    }
}
