//! Procedural photographs so the interface has believable content without files.

use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Scene {
    AlpineLake,
    Dunes,
    Portrait,
    CityNight,
    Forest,
    Sunset,
    Bokeh,
    SnowField,
    Canyon,
    Lavender,
}

impl Scene {
    pub(crate) const ALL: [Scene; 10] = [
        Scene::AlpineLake,
        Scene::Dunes,
        Scene::Portrait,
        Scene::CityNight,
        Scene::Forest,
        Scene::Sunset,
        Scene::Bokeh,
        Scene::SnowField,
        Scene::Canyon,
        Scene::Lavender,
    ];

    pub(crate) fn title(self) -> &'static str {
        match self {
            Scene::AlpineLake => "Alpine Lake",
            Scene::Dunes => "Dunes",
            Scene::Portrait => "Studio Portrait",
            Scene::CityNight => "City Night",
            Scene::Forest => "Forest Light",
            Scene::Sunset => "Coast Sunset",
            Scene::Bokeh => "Garden Bokeh",
            Scene::SnowField => "Snow Field",
            Scene::Canyon => "Canyon",
            Scene::Lavender => "Lavender Rows",
        }
    }

    pub(crate) fn portrait(self) -> bool {
        matches!(self, Scene::Portrait | Scene::Forest)
    }
}

fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(374_761_393)
        ^ (y as u32).wrapping_mul(668_265_263)
        ^ seed.wrapping_mul(2_147_483_647);
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h & 0x00ff_ffff) as f32 / 0x00ff_ffff as f32
}

fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

pub(crate) fn noise(x: f32, y: f32, seed: u32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let tx = smooth(x - xi as f32);
    let ty = smooth(y - yi as f32);
    let a = hash(xi, yi, seed);
    let b = hash(xi + 1, yi, seed);
    let c = hash(xi, yi + 1, seed);
    let d = hash(xi + 1, yi + 1, seed);
    let top = a + (b - a) * tx;
    let bottom = c + (d - c) * tx;
    top + (bottom - top) * ty
}

pub(crate) fn fbm(x: f32, y: f32, seed: u32, octaves: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    for i in 0..octaves {
        sum += amp * noise(x * freq, y * freq, seed.wrapping_add(i * 31));
        freq *= 2.03;
        amp *= 0.5;
    }
    sum
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn hex(c: u32) -> [f32; 3] {
    [((c >> 16) & 255) as f32 / 255.0, ((c >> 8) & 255) as f32 / 255.0, (c & 255) as f32 / 255.0]
}

fn sstep(a: f32, b: f32, x: f32) -> f32 {
    smooth(((x - a) / (b - a)).clamp(0.0, 1.0))
}

/// Renders `scene` as RGBA8 at the given size.
pub(crate) fn render(scene: Scene, width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut out = vec![0u8; (width * height * 4) as usize];
    let aspect = width as f32 / height as f32;
    out.par_chunks_mut((width * 4) as usize).enumerate().for_each(|(row, line)| {
        let v = row as f32 / height as f32;
        for col in 0..width as usize {
            let u = col as f32 / width as f32;
            let c = shade(scene, u, v, aspect, seed);
            let px = &mut line[col * 4..col * 4 + 4];
            for k in 0..3 {
                px[k] = (c[k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
            px[3] = 255;
        }
    });
    out
}

fn shade(scene: Scene, u: f32, v: f32, aspect: f32, seed: u32) -> [f32; 3] {
    let grain = (hash((u * 4000.0) as i32, (v * 4000.0) as i32, seed) - 0.5) * 0.018;
    let c = match scene {
        Scene::AlpineLake => alpine(u, v, aspect, seed),
        Scene::Dunes => dunes(u, v, seed),
        Scene::Portrait => portrait(u, v, aspect, seed),
        Scene::CityNight => city(u, v, aspect, seed),
        Scene::Forest => forest(u, v, seed),
        Scene::Sunset => sunset(u, v, aspect, seed),
        Scene::Bokeh => bokeh(u, v, aspect, seed),
        Scene::SnowField => snow(u, v, seed),
        Scene::Canyon => canyon(u, v, seed),
        Scene::Lavender => lavender(u, v, aspect, seed),
    };
    [c[0] + grain, c[1] + grain, c[2] + grain]
}

fn ridge(u: f32, seed: u32, base: f32, amp: f32, freq: f32) -> f32 {
    base - amp * fbm(u * freq, 0.5, seed, 5)
}

fn alpine(u: f32, v: f32, aspect: f32, seed: u32) -> [f32; 3] {
    let horizon = 0.58;
    let sky = mix(hex(0x2f5d9e), hex(0xbcd6ee), v / horizon);
    let cloud = sstep(0.55, 0.8, fbm(u * 3.0 * aspect, v * 6.0, seed + 7, 5)) * (1.0 - v / horizon);
    let mut c = mix(sky, hex(0xf4f6f8), cloud * 0.8);
    let far = ridge(u, seed + 1, 0.42, 0.22, 3.0);
    let near = ridge(u, seed + 2, 0.56, 0.16, 5.0);
    if v > far && v < horizon {
        let snow = sstep(0.05, 0.0, v - far - 0.03 * fbm(u * 20.0, v * 20.0, seed, 3));
        let rock = mix(hex(0x5a6578), hex(0x8b93a3), fbm(u * 30.0, v * 30.0, seed + 3, 4));
        c = mix(rock, hex(0xf2f5fa), snow);
    }
    if v > near && v < horizon {
        c = mix(hex(0x1f3b2a), hex(0x34553a), fbm(u * 60.0, v * 60.0, seed + 4, 4));
    }
    if v >= horizon {
        let mirror = horizon - (v - horizon) * 0.9;
        let ripple = (fbm(u * 80.0, v * 400.0, seed + 5, 3) - 0.5) * 0.02;
        // The mirrored row is above the horizon, so this recursion is one level deep.
        let r = alpine(u + ripple, mirror.clamp(0.0, horizon - 1e-3), aspect, seed);
        c = mix(r, hex(0x163047), 0.35 + (v - horizon) * 0.8);
    }
    c
}

fn dunes(u: f32, v: f32, seed: u32) -> [f32; 3] {
    let sky = mix(hex(0x7fa7d6), hex(0xf3d9b1), v / 0.4);
    if v < 0.4 {
        return sky;
    }
    let mut c = hex(0xd99a5b);
    for i in 0..4 {
        let crest = 0.4 + i as f32 * 0.15 + 0.08 * (u * 6.0 + i as f32 * 2.1 + seed as f32).sin();
        if v > crest {
            let lit = ((u * 6.0 + i as f32 * 2.1 + seed as f32).cos() * 0.5 + 0.5).powf(1.5);
            c = mix(hex(0x8a4f2a), hex(0xf0b77a), lit);
            let ripples = ((v * 300.0 + u * 40.0).sin() * 0.5 + 0.5) * 0.06;
            c = [c[0] - ripples, c[1] - ripples, c[2] - ripples];
        }
    }
    c
}

fn portrait(u: f32, v: f32, aspect: f32, seed: u32) -> [f32; 3] {
    let dx = (u - 0.5) * aspect;
    let bg = mix(hex(0x6c7480), hex(0x2a2e35), ((dx * dx + (v - 0.35).powi(2)).sqrt() * 1.6).min(1.0));
    let mut c = bg;
    // Shoulders.
    let shoulder = 0.78 + 0.25 * (dx * 2.2).powi(2);
    if v > shoulder {
        c = mix(hex(0x1f2a44), hex(0x33415f), fbm(u * 40.0, v * 40.0, seed, 3));
    }
    // Neck.
    if dx.abs() < 0.07 && v > 0.55 && v < shoulder + 0.02 {
        c = hex(0xb57a5c);
    }
    // Face.
    let fx = dx / 0.17;
    let fy = (v - 0.42) / 0.21;
    let face = fx * fx + fy * fy;
    if face < 1.0 {
        let light = (1.0 - (fx + 0.6).powi(2) * 0.25).clamp(0.4, 1.0);
        let skin = mix(hex(0x9c6046), hex(0xe9b896), light);
        let cheek = (-((fx.abs() - 0.5).powi(2) + (fy - 0.25).powi(2)) * 12.0).exp() * 0.12;
        c = [skin[0] + cheek, skin[1], skin[2]];
        // Eyes and lips.
        for side in [-0.38f32, 0.38] {
            let ex = (fx - side) / 0.14;
            let ey = (fy + 0.05) / 0.06;
            if ex * ex + ey * ey < 1.0 {
                c = mix(hex(0xf2efe9), hex(0x3b2a1e), sstep(0.5, 0.2, ex * ex + ey * ey));
            }
        }
        let lx = fx / 0.32;
        let ly = (fy - 0.5) / 0.07;
        if lx * lx + ly * ly < 1.0 {
            c = hex(0xb4545a);
        }
    }
    // Hair.
    let hair = (fx * fx * 0.9 + ((v - 0.36) / 0.25).powi(2)) < 1.25 && (fy < -0.45 || fx.abs() > 0.92);
    if hair && face < 1.6 {
        c = mix(hex(0x1b120d), hex(0x4a3020), fbm(u * 120.0, v * 30.0, seed + 9, 4));
    }
    c
}

fn city(u: f32, v: f32, aspect: f32, seed: u32) -> [f32; 3] {
    let mut c = mix(hex(0x070b1e), hex(0x283a6b), v);
    let cols = 26.0;
    let b = (u * cols).floor();
    let top = 0.25 + 0.45 * hash(b as i32, 0, seed);
    if v > top {
        let wx = (u * cols * 6.0).fract();
        let wy = (v * 90.0).fract();
        let lit = hash((u * cols * 6.0) as i32, (v * 90.0) as i32, seed) > 0.55;
        c = hex(0x10131c);
        if wx > 0.25 && wx < 0.75 && wy > 0.3 && wy < 0.8 && lit {
            c = mix(hex(0xffc66e), hex(0x9fd8ff), hash(b as i32, 3, seed));
        }
    }
    for i in 0..9 {
        let lx = (i as f32 + 0.5) / 9.0;
        let d = (((u - lx) * aspect).powi(2) + (v - 0.86).powi(2)).sqrt();
        let glow = (-d * 30.0).exp();
        c = [c[0] + glow * 1.2, c[1] + glow * 0.8, c[2] + glow * 0.4];
    }
    if v > 0.9 {
        c = mix(hex(0x1a1d26), hex(0x3b2e22), (v - 0.9) * 8.0);
    }
    c
}

fn forest(u: f32, v: f32, seed: u32) -> [f32; 3] {
    let mut c = mix(hex(0xc8e3b0), hex(0x355e2b), v);
    for i in 0..14 {
        let x = hash(i, 1, seed);
        let w = 0.01 + 0.04 * hash(i, 2, seed);
        if (u - x).abs() < w {
            c = mix(hex(0x2a1d14), hex(0x5b4330), fbm(u * 200.0, v * 10.0, seed + i as u32, 3));
        }
    }
    let rays = ((u - v * 0.4) * 30.0).sin().max(0.0).powf(8.0) * (1.0 - v) * 0.35;
    let leaves = sstep(0.55, 0.75, fbm(u * 12.0, v * 12.0, seed + 99, 5)) * (1.0 - v * 1.2).max(0.0);
    c = mix(c, hex(0x5f8f3a), leaves);
    if v > 0.82 {
        c = mix(hex(0x3b4e25), hex(0x6b5a35), fbm(u * 50.0, v * 50.0, seed, 4));
    }
    [c[0] + rays, c[1] + rays * 0.95, c[2] + rays * 0.7]
}

fn sunset(u: f32, v: f32, aspect: f32, seed: u32) -> [f32; 3] {
    let horizon = 0.6;
    let mut c = if v < horizon {
        let t = v / horizon;
        mix(mix(hex(0x2b1b4f), hex(0xd2507b), t * 1.4), hex(0xffb35c), (t - 0.6) * 2.5)
    } else {
        mix(hex(0x5b2e4a), hex(0x0f1830), (v - horizon) * 2.5)
    };
    let sun = (((u - 0.62) * aspect).powi(2) + (v - 0.55).powi(2)).sqrt();
    if sun < 0.06 && v < horizon {
        c = hex(0xfff0c8);
    }
    let glow = (-sun * 8.0).exp() * 0.6;
    c = [c[0] + glow, c[1] + glow * 0.6, c[2] + glow * 0.25];
    if v > horizon {
        let streak = ((u - 0.62).abs() < 0.05 + (v - horizon) * 0.2) as i32 as f32
            * sstep(0.4, 0.7, fbm(u * 30.0, v * 300.0, seed, 3));
        c = [c[0] + streak * 0.6, c[1] + streak * 0.4, c[2] + streak * 0.15];
    }
    let cloud = sstep(0.55, 0.75, fbm(u * 4.0, v * 14.0, seed + 5, 5)) * (v < horizon - 0.1) as i32 as f32;
    mix(c, hex(0x40203a), cloud * 0.7)
}

fn bokeh(u: f32, v: f32, aspect: f32, seed: u32) -> [f32; 3] {
    let mut c = mix(hex(0x1d3b1f), hex(0x5f7d2e), fbm(u * 2.0, v * 2.0, seed, 4));
    for i in 0..40 {
        let x = hash(i, 11, seed);
        let y = hash(i, 12, seed);
        let r = 0.03 + 0.07 * hash(i, 13, seed);
        let d = (((u - x) * aspect).powi(2) + (v - y).powi(2)).sqrt();
        if d < r {
            let col = match i % 4 {
                0 => hex(0xffd36b),
                1 => hex(0xff7aa8),
                2 => hex(0xb5f26b),
                _ => hex(0xfff3d6),
            };
            let edge = sstep(r * 0.8, r, d) * 0.3 + 0.35;
            c = mix(c, col, edge);
        }
    }
    let fx = ((u - 0.5) * aspect) / 0.12;
    let fy = (v - 0.55) / 0.18;
    let petal_angle = fy.atan2(fx);
    let petal = 0.75 + 0.25 * (petal_angle * 6.0).cos();
    let r = (fx * fx + fy * fy).sqrt();
    if r < petal {
        c = mix(hex(0xc2185b), hex(0xff8fb8), r / petal);
        if r < 0.22 {
            c = hex(0xf9c74f);
        }
    }
    c
}

fn snow(u: f32, v: f32, seed: u32) -> [f32; 3] {
    let sky = mix(hex(0xaac4dd), hex(0xe9f0f6), v / 0.45);
    if v < 0.45 + 0.03 * fbm(u * 5.0, 0.0, seed, 3) {
        return sky;
    }
    let tone = fbm(u * 6.0, v * 20.0, seed + 2, 5);
    let mut c = mix(hex(0xd5e0ec), hex(0xfbfdff), tone);
    let tree = (u * 40.0).fract() < 0.15 && hash((u * 40.0) as i32, 1, seed) > 0.7 && v < 0.62;
    if tree {
        c = hex(0x26302a);
    }
    c
}

fn canyon(u: f32, v: f32, seed: u32) -> [f32; 3] {
    let sky = mix(hex(0x3d76b8), hex(0xa9c9e8), v / 0.3);
    let left = 0.35 + 0.25 * fbm(v * 4.0, 1.0, seed, 4) - v * 0.1;
    let right = 0.65 - 0.25 * fbm(v * 4.0, 2.0, seed, 4) + v * 0.1;
    if v < 0.3 && u > left && u < right {
        return sky;
    }
    let strata = (v * 60.0 + fbm(u * 8.0, v * 8.0, seed, 4) * 6.0).sin() * 0.5 + 0.5;
    let base = mix(hex(0x8f3b1b), hex(0xe08a4f), strata);
    let lit = if u < 0.5 { 0.9 } else { 0.6 };
    let mut c = [base[0] * lit, base[1] * lit, base[2] * lit];
    if v > 0.85 && u > left - 0.05 && u < right + 0.05 {
        c = mix(hex(0x3a6b5a), hex(0x7aa58d), fbm(u * 40.0, v * 100.0, seed, 3));
    }
    c
}

fn lavender(u: f32, v: f32, aspect: f32, seed: u32) -> [f32; 3] {
    let horizon = 0.38;
    if v < horizon {
        let sky = mix(hex(0x6fa0d8), hex(0xffe0b5), v / horizon);
        let tree = ((u - 0.7) * aspect).abs() < 0.05 * (1.0 - (horizon - v) * 4.0).max(0.0) && v > horizon - 0.15;
        return if tree { hex(0x2c3a1f) } else { sky };
    }
    let depth = (v - horizon) / (1.0 - horizon);
    let row = ((u - 0.5) / (0.05 + depth) * 9.0).sin();
    let flower = mix(hex(0x5b3f8c), hex(0xa889d6), fbm(u * 200.0, v * 200.0, seed, 3));
    let soil = hex(0x4e5b32);
    mix(soil, flower, sstep(-0.2, 0.4, row))
}
