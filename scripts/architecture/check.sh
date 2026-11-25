#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
cargo test -p aurora-engine --test architecture_rules --locked -- --test-threads=1
