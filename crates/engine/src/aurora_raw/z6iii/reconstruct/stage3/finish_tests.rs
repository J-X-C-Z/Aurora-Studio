#[test]
#[ignore = "run verify_finish.py to supply the reference"]
fn finish_wrapper_matches() {
    let data = std::fs::read(std::env::var_os("C1_FINISH_ORACLE").unwrap()).unwrap();
    assert_eq!(data.len() % 4, 0);
    let mut words = data.chunks_exact(4).map(|word| u32::from_le_bytes(word.try_into().unwrap()));
    let count = words.next().unwrap();
    let mut pixels = 0;
    let mut differences = 0;
    for case in 0..count {
        let width = words.next().unwrap() as usize;
        let height = words.next().unwrap() as usize;
        let control = words.next().unwrap() as i32;
        let mut floats = (&mut words).map(f32::from_bits);
        let gains = std::array::from_fn(|_| floats.next().unwrap());
        let offset = floats.next().unwrap();
        let noise = floats.next().unwrap();
        let stride = width + 12;
        let [mut red, mut green, mut blue] = std::array::from_fn(|_| vec![0.0; stride * (height + 12)]);
        for row in 4..8 + height {
            for column in 2..10 + width {
                let index = row * stride + column;
                red[index] = floats.next().unwrap();
                green[index] = floats.next().unwrap();
                blue[index] = floats.next().unwrap();
            }
        }
        super::finish::apply([&mut red, &mut green, &mut blue], stride, width, height, gains, offset, noise, control);
        for row in 6..6 + height {
            for column in 6..6 + width {
                let index = row * stride + column;
                for actual in [red[index], green[index], blue[index]] {
                    let expected = floats.next().unwrap();
                    if actual.to_bits() != expected.to_bits() {
                        if differences < 4 {
                            eprintln!("case {case} ({column},{row}) {actual:?} != {expected:?}");
                        }
                        differences += 1;
                    }
                }
                pixels += 1;
            }
        }
    }
    assert!(words.next().is_none());
    assert_eq!(differences, 0, "{pixels} RGB pixels");
    println!("{count} wrapper cases, {pixels} RGB pixels bit-exact");
}
