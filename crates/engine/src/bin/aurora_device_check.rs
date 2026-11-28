//! Non-interactive counterpart to Settings > General > Inference device check.
//! `cargo run --locked -p aurora-engine --bin aurora_device_check -- dx12`
fn main() {
    let request = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("AURORA_GPU_BACKEND").ok())
        .unwrap_or_else(|| "auto".into());
    match aurora_engine::InferenceProbe::run(&request) {
        Ok(r) => println!(
            "{}",
            serde_json::to_string_pretty(&r).expect("finite integer timings")
        ),
        Err(e) => {
            eprintln!(
                "{}",
                serde_json::json!({"requested":request,"executed":null,"error":e})
            );
            std::process::exit(1);
        }
    }
}
