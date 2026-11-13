# Portrait crate validation (2026-11-13)

CARGO_TARGET_DIR = 90_work/build-cache/portrait-publish (project root relative).
PIXCAKE_CUDA_REQUIRED=1; no external-fixtures enabled during runtime tests.

## cargo check -p aurora-pixcake --all-targets --all-features --locked

Exit code: 0

```text
    Checking aurora-pixcake v0.1.0 (crates/portrait)
warning: method `byte` is never used
  --> crates\portrait\tests\cuda_roi.rs:40:8
   |
20 | impl Rng {
   | -------- method in this implementation
...
40 |     fn byte(&mut self) -> u8 {
   |        ^^^^
   |
   = note: `#[warn(dead_code)]` (part of `#[warn(unused)]`) on by default

warning: `aurora-pixcake` (test "cuda_roi") generated 1 warning
    Finished `dev` profile [optimized + debuginfo] target(s) in 1.96s
```

## cargo test -p aurora-pixcake --locked --quiet -- --test-threads=1

Exit code: 0

```text

running 465 tests
....................................................................................... 87/465
....................................................................................... 174/465
....................................................................................... 261/465
....................................................................................... 348/465
....................................................................................... 435/465
..............................
test result: ok. 465 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.41s


running 2 tests
..
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.29s


running 4 tests
....
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.68s


running 3 tests
..i
test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.32s


running 6 tests
......
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.74s


running 3 tests
...
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.52s


running 5 tests
.....
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.82s


running 3 tests
...
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.59s


running 7 tests
.......
test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.14s


running 2 tests
..
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.30s


running 5 tests
.....
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.69s


running 7 tests
.......
test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.41s


running 10 tests
..........
test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.10s


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s


running 5 tests
.....
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.80s


running 4 tests
....
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.69s


running 6 tests
......
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.30s


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s


running 14 tests
iiiiiiiiii.i.i
test result: ok. 2 passed; 0 failed; 12 ignored; 0 measured; 0 filtered out; finished in 0.01s


running 4 tests
iiii
test result: ok. 0 passed; 0 failed; 4 ignored; 0 measured; 0 filtered out; finished in 0.00s


running 1 test
.
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s


running 2 tests
..
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

warning: method `byte` is never used
  --> crates\portrait\tests\cuda_roi.rs:40:8
   |
20 | impl Rng {
   | -------- method in this implementation
...
40 |     fn byte(&mut self) -> u8 {
   |        ^^^^
   |
   = note: `#[warn(dead_code)]` (part of `#[warn(unused)]`) on by default
```
