fn header(bytes: &[u8]) -> Option<(fn([u8; 4]) -> u32, usize)> {
    let read = match bytes.get(..2)? {
        b"II" => u32::from_le_bytes,
        b"MM" => u32::from_be_bytes,
        _ => return None,
    };
    if read(bytes.get(2..6)?.try_into().ok()?) != 6 {
        return None;
    }
    let count = read(bytes.get(6..10)?.try_into().ok()?) as usize;
    if !(2..=4096).contains(&count) {
        return None;
    }
    Some((read, count))
}

pub fn film_settings(bytes: &[u8]) -> Option<(u32, f32)> {
    let (read, count) = header(bytes)?;
    let mut fields = bytes.get(14 + count * 8..)?;
    for _ in 0..2 {
        let length = read(fields.split_off(..4)?.try_into().ok()?) as usize;
        fields = fields.get(length..)?;
    }
    let flags = read(fields.get(..4)?.try_into().ok()?);
    let gain = f32::from_bits(read(fields.get(4..8)?.try_into().ok()?));
    (gain.is_finite() && gain > 0.0).then_some((flags, gain))
}

pub(super) fn primary_nodes(bytes: &[u8]) -> Option<Vec<[f64; 3]>> {
    let (read, count) = header(bytes)?;
    let mut nodes = Vec::with_capacity(count + 2);
    for pair in bytes.get(10..10 + count * 8)?.chunks_exact(8) {
        let x = f64::from(read(pair[..4].try_into().ok()?)) / f64::from(u32::MAX);
        let y = f64::from(read(pair[4..].try_into().ok()?)) / f64::from(u32::MAX);
        nodes.push([x, y, 0.0]);
    }
    if nodes.first()?.get(..2)? != [0.0, 0.0] || nodes.last()?.get(..2)? != [1.0, 1.0] {
        return None;
    }
    let widths: Vec<_> = nodes.windows(2).map(|pair| pair[1][0] - pair[0][0]).collect();
    if widths.iter().any(|width| *width <= 0.0) {
        return None;
    }
    if count == 2 {
        return Some(Vec::new());
    }
    let slopes: Vec<_> = nodes.windows(2).zip(&widths).map(|(pair, width)| (pair[1][1] - pair[0][1]) / width).collect();
    let mut diagonal = vec![1.0; count];
    let mut second = vec![0.0; count];
    for index in 1..count - 1 {
        let left = widths[index - 1];
        let right = widths[index];
        let factor = left / diagonal[index - 1];
        let previous_upper = if index == 1 { 0.0 } else { left };
        diagonal[index] = 2.0 * (left + right) - factor * previous_upper;
        second[index] = 3.0 * (slopes[index] - slopes[index - 1]) - factor * second[index - 1];
    }
    for index in (1..count - 1).rev() {
        second[index] = (second[index] - widths[index] * second[index + 1]) / diagonal[index];
    }
    for index in 0..count - 1 {
        nodes[index][2] = slopes[index] - widths[index] * (2.0 * second[index] + second[index + 1]) / 3.0;
    }
    nodes[count - 1][2] = slopes[count - 2] + widths[count - 2] * second[count - 2] / 3.0;
    nodes.insert(0, nodes[0]);
    nodes.push(nodes[count]);
    Some(nodes)
}
