//! Prints the effect registry as JSON for data-driven configuration.
//!
//!   cargo run -p aurora-portrait --example catalog_json --offline

fn main() {
    println!("{}", aurora_portrait::catalog_json());
}
