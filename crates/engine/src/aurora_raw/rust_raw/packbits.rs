//! PackBits (TIFF compression 32773) decoder.

use super::{NativeError, NativeResult};

/// Expands PackBits data until `expected` bytes are produced.
///
/// A run that would overrun `expected` is truncated; that is what LibRaw's
/// strip reader effectively does and it keeps a corrupt tail from allocating.
pub(crate) fn decode(input: &[u8], expected: usize) -> NativeResult<Vec<u8>> {
    let mut out = Vec::with_capacity(expected);
    let mut at = 0usize;
    while at < input.len() && out.len() < expected {
        let header = input[at] as i8;
        at += 1;
        match header {
            0..=127 => {
                let count = header as usize + 1;
                let end = at
                    .checked_add(count)
                    .filter(|&end| end <= input.len())
                    .ok_or_else(|| NativeError::corrupt("truncated PackBits literal"))?;
                let take = count.min(expected - out.len());
                out.extend_from_slice(&input[at..at + take]);
                at = end;
            }
            -127..=-1 => {
                let byte = *input
                    .get(at)
                    .ok_or_else(|| NativeError::corrupt("truncated PackBits run"))?;
                at += 1;
                let count = (1i32 - i32::from(header)) as usize;
                let take = count.min(expected - out.len());
                out.extend(std::iter::repeat_n(byte, take));
            }
            // -128 is a no-op.
            _ => {}
        }
    }
    if out.len() != expected {
        return Err(NativeError::corrupt(format!(
            "PackBits produced {} bytes, expected {expected}",
            out.len()
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_literals_and_runs() {
        // 3 literal bytes, then a four-byte run of 0xaa.
        let input = [2, 1, 2, 3, 0xfd, 0xaa];
        assert_eq!(decode(&input, 7).unwrap(), [1, 2, 3, 0xaa, 0xaa, 0xaa, 0xaa]);
    }

    #[test]
    fn rejects_truncated_input() {
        assert!(decode(&[5, 1, 2], 6).is_err());
    }
}
