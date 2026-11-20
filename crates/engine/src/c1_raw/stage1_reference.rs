//! stage1 producer (DLL 0x1683DB0): predicted value, residual, packed cost.

pub struct Params {
    pub base: i64,
    pub g1_q10: i64,
    pub g2_q10: i64,
}

impl Default for Params {
    fn default() -> Self {
        Self { base: 0, g1_q10: 1024, g2_q10: 768 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub r: usize,
    pub c: usize,
    pub phase: u8,
    pub center: i64,
    pub pred: i64,
    /// Residuals in the order up, down, left, right.
    pub res: [i64; 4],
    /// (|left-right| << 16) | |up-down|
    pub cost: i64,
}

/// pred = base + ((pixel - base) * gain_q10 >> 10), arithmetic shift.
#[inline]
pub fn pred(base: i64, pixel: i64, gain_q10: i64) -> i64 {
    base + (((pixel - base) * gain_q10) >> 10)
}

/// One block per 2x2 centre, two phases: (r,c) with gain1, (r+1,c+1) with gain2.
pub fn stage1(plane: &[u16], w: usize, h: usize, stride: usize, halo: usize, p: &Params) -> Vec<Block> {
    let m = halo.max(2);
    let mut out = Vec::new();
    if w <= 2 * m || h <= 2 * m {
        return out;
    }
    let at = |y: usize, x: usize| plane[y * stride + x] as i64;
    let mut r = m;
    while r < h - m {
        let mut c = m;
        while c < w - m {
            for (phase, rr, cc, gain) in [(1u8, r, c, p.g1_q10), (2u8, r + 1, c + 1, p.g2_q10)] {
                let center = at(rr, cc);
                let pr = pred(p.base, center, gain);
                let res = [
                    at(rr - 2, cc) - pr,
                    at(rr + 2, cc) - pr,
                    at(rr, cc - 2) - pr,
                    at(rr, cc + 2) - pr,
                ];
                let cost = ((res[2] - res[3]).abs() << 16) | (res[0] - res[1]).abs();
                out.push(Block { r: rr, c: cc, phase, center, pred: pr, res, cost });
            }
            c += 2;
        }
        r += 2;
    }
    out
}
