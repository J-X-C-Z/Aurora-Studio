//! `AURORA_BLEMISH_GPU=0` must return the CPU bytes unchanged.

use aurora_engine::blemish_gainmap::{
    apply_gainmap_16u_gated, GainMap, SpotlessGainmapInputs,
};
use std::path::PathBuf;

fn read_u16(path: &PathBuf) -> Vec<u16> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(2)
        .map(|p| u16::from_le_bytes([p[0], p[1]]))
        .collect()
}

fn main() {
    unsafe {
        std::env::set_var("AURORA_BLEMISH_GPU", "0");
        std::env::set_var("AURORA_BLEMISH_GAINMAP", "1");
    }
    let dir = std::env::args_os()
        .nth(1)
        .or_else(|| std::env::var_os("AURORA_BLEMISH_FIXTURE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("usage: blemish_env_off <fixture_dir> (or set AURORA_BLEMISH_FIXTURE)");
            std::process::exit(2);
        });
    let src = read_u16(&dir.join("s1_linear_9868.u16"));
    let golden = read_u16(&dir.join("engine_9868_blemish.u16"));
    let rgba: Vec<f32> = std::fs::read(dir.join("gainmap_9868_450x300.f32"))
        .unwrap()
        .chunks_exact(4)
        .map(|p| f32::from_le_bytes([p[0], p[1], p[2], p[3]]))
        .collect();
    let map = GainMap {
        width: 450,
        height: 300,
        rgba,
    };
    let input = SpotlessGainmapInputs {
        rgb16: &src,
        width: 6048,
        height: 4032,
        gain_map: &map,
        amount: 0.5,
        protection_mask: None,
    };
    let out = apply_gainmap_16u_gated(&input, true).expect("gated consumer");
    assert_eq!(out, golden, "AURORA_BLEMISH_GPU=0 bytes differ");
    println!(
        "AURORA_BLEMISH_GPU=0: {} bytes, byte-identical to engine_9868_blemish.u16",
        out.len() * 2
    );
}
