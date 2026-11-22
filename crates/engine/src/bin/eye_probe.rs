//! Prints the closed-eye models' tensor IO and runs them on one photo.
//!
//! ```text
//! LUMEN_AIMODELS=~/agents/evidence \
//!   cargo run -p lumen-engine --bin eye_probe -- photo.jpg
//! ```

use anyhow::Result;

fn main() -> Result<()> {
    for model in ["EyeDetector.onnx", "DenseNet121_FP32.onnx"] {
        match aurora_engine::aimask::model_summary(model) {
            Ok(lines) => lines.iter().for_each(|l| println!("{l}")),
            Err(err) => println!("{model}: {err:#}"),
        }
    }
    Ok(())
}

