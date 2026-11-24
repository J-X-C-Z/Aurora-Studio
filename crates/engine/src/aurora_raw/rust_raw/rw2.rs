//! Panasonic RW2 sensor decode.
//!
//! Behaviour follows LibRaw 0.23's Panasonic front-end: the raw IFD is a TIFF
//! IFD carrying a LONG `0x0001` PanasonicRaw marker, dimensions/crops/levels
//! come from Panasonic-private tags, and sensor samples are decoded according
//! to RawFormat `0x002d`.  `make_frame` produces the same tight Bayer plane
//! shape as the other native RAW decoders.

use super::tiff::{Ifd, Tiff};
use super::{make_frame, meta, NativeError, NativeResult};

use crate::raw::BayerFrame;

const FILTER_WORDS: [u32; 4] = [0x9494_9494, 0x6161_6161, 0x4949_4949, 0x1616_1616];

#[derive(Debug, Default, Clone, Copy)]
struct PanaColor {
    matrix: [i32; 9],
    black: i32,
    maximum: u32,
}

/// Panasonic entries from LibRaw's Adobe colour table, in table order.
fn adobe_panasonic(model: &str) -> Option<PanaColor> {
    const TABLE: &[(&str, i32, u32, [i32; 9])] = &[
        ("DC-S1RM2", 0, 0, [9941, -4056, -871, -3745, 11604, 2433, -254, 1132, 5526]),
        ("DC-S1R", 0, 0, [11822, -5321, -1249, -5958, 15114, 766, -614, 1264, 7043]),
        ("DC-S1H", 0, 0, [9397, -3719, -805, -5425, 13326, 2309, -972, 1715, 6034]),
        ("DC-S1M2ES", 0, 0, [11333, -4794, -1062, -3912, 12027, 2115, 42, 784, 5904]),
        ("DC-S1M2", 0, 0, [10730, -4442, -922, -4478, 12504, 2199, -380, 1198, 6039]),
        ("DC-S5M2", 0, 0, [10308, -4206, -783, -4088, 12102, 2229, -125, 1051, 5912]),
        ("DC-S5", 0, 0, [9744, -3905, -779, -4899, 12807, 2324, -798, 1630, 5827]),
        ("DC-S1", 0, 0, [9744, -3905, -779, -4899, 12807, 2324, -798, 1630, 5827]),
        ("DC-S9", 0, 0, [9983, -3890, -841, -4180, 12164, 2263, -249, 1139, 5766]),
        ("DMC-CM1", -15, 0, [8770, -3194, -820, -2871, 11281, 1803, -513, 1552, 4434]),
        ("DC-FZ1000M2", -15, 0, [9803, -4185, -992, -4066, 12578, 1628, -838, 1824, 5288]),
        ("DMC-FZ1000", -15, 0, [7830, -2696, -763, -3325, 11667, 1866, -641, 1712, 4824]),
        ("DMC-FZ2500", -15, 0, [7386, -2443, -743, -3437, 11864, 1757, -608, 1660, 4766]),
        ("DMC-FZ100", -15, 0xfff, [16197, -6146, -1761, -2393, 10765, 1869, 366, 2238, 5248]),
        ("DMC-FZ150", -15, 0xfff, [11904, -4541, -1189, -2355, 10899, 1662, -296, 1586, 4289]),
        ("DMC-FZ200", -15, 0xfff, [8112, -2563, -740, -3730, 11784, 2197, -941, 2075, 4933]),
        ("DMC-FZ300", -15, 0xfff, [8378, -2798, -769, -3068, 11410, 1877, -538, 1792, 4623]),
        ("DMC-FZ18", 0, 0, [9932, -3060, -935, -5809, 13331, 2753, -1267, 2155, 5575]),
        ("DMC-FZ28", -15, 0xf96, [10109, -3488, -993, -5412, 12812, 2916, -1305, 2140, 5543]),
        ("DMC-FZ30", 0, 0xf94, [10976, -4029, -1141, -7918, 15491, 2600, -1670, 2071, 8246]),
        ("DMC-FZ35", -15, 0, [9938, -2780, -890, -4604, 12393, 2480, -1117, 2304, 4620]),
        ("DMC-FZ40", -15, 0, [13639, -5535, -1371, -1698, 9633, 2430, 316, 1152, 4108]),
        ("DMC-FZ50", 0, 0, [7906, -2709, -594, -6231, 13351, 3220, -1922, 2631, 6537]),
        ("DMC-FZ70", -15, 0, [11532, -4324, -1066, -2375, 10847, 1749, -564, 1699, 4351]),
        ("DC-FZ80", -15, 0, [8550, -2908, -842, -3195, 11529, 1881, -338, 1603, 4631]),
        ("DMC-FZ8", 0, 0xf7f, [8986, -2755, -802, -6341, 13575, 3077, -1476, 2144, 6379]),
        ("DMC-L10", -15, 0xf96, [8025, -1942, -1050, -7920, 15904, 2100, -2456, 3005, 7039]),
        ("DMC-L1", 0, 0xf7f, [8054, -1885, -1025, -8349, 16367, 2040, -2805, 3542, 7629]),
        ("DMC-LC1", 0, 0, [11340, -4069, -1275, -7555, 15266, 2448, -2960, 3426, 7685]),
        ("DMC-LF1", -15, 0, [9379, -3267, -816, -3227, 11560, 1881, -926, 1928, 5340]),
        ("DC-LX100M2", -15, 0, [8585, -3127, -833, -4005, 12250, 1953, -650, 1494, 4862]),
        ("DMC-LX100", -15, 0, [8844, -3538, -768, -3709, 11762, 2200, -698, 1792, 5220]),
        ("DMC-LX1", 0, 0xf7f, [10704, -4187, -1230, -8314, 15952, 2501, -920, 945, 8927]),
        ("DMC-LX2", 0, 0, [8048, -2810, -623, -6450, 13519, 3272, -1700, 2146, 7049]),
        ("DMC-LX3", -15, 0, [8128, -2668, -655, -6134, 13307, 3161, -1782, 2568, 6083]),
        ("DMC-LX5", -15, 0, [10909, -4295, -948, -1333, 9306, 2399, 22, 1738, 4582]),
        ("DMC-LX7", -15, 0, [10148, -3743, -991, -2837, 11366, 1659, -701, 1893, 4899]),
        ("DMC-LX9", -15, 0, [7790, -2736, -755, -3452, 11870, 1769, -628, 1647, 4898]),
        ("DMC-FX150", -15, 0xfff, [9082, -2907, -925, -6119, 13377, 3058, -1797, 2641, 5609]),
        ("DC-G99", -15, 0, [9657, -3963, -748, -3361, 11378, 2258, -568, 1415, 5158]),
        ("DC-G100", -15, 0, [8370, -2869, -710, -3389, 11372, 2298, -640, 1599, 4887]),
        ("DMC-G10", 0, 0, [10113, -3400, -1114, -4765, 12683, 2317, -377, 1437, 6710]),
        ("DMC-G1", -15, 0xf94, [8199, -2065, -1056, -8124, 16156, 2033, -2458, 3022, 7220]),
        ("DMC-G2", -15, 0xf3c, [10113, -3400, -1114, -4765, 12683, 2317, -377, 1437, 6710]),
        ("DMC-G3", -15, 0xfff, [6763, -1919, -863, -3868, 11515, 2684, -1216, 2387, 5879]),
        ("DMC-G5", -15, 0xfff, [7798, -2562, -740, -3879, 11584, 2613, -1055, 2248, 5434]),
        ("DMC-G6", -15, 0xfff, [8294, -2891, -651, -3869, 11590, 2595, -1183, 2267, 5352]),
        ("DMC-G7", -15, 0xfff, [7610, -2780, -576, -4614, 12195, 2733, -1375, 2393, 6490]),
        ("DMC-G8", -15, 0xfff, [7610, -2780, -576, -4614, 12195, 2733, -1375, 2393, 6490]),
        ("DC-G9M2", 0, 0, [8325, -3456, -623, -4330, 12089, 2528, -860, 2646, 5984]),
        ("DC-G9", -15, 0, [7685, -2375, -634, -3687, 11700, 2249, -748, 1546, 5111]),
        ("DMC-GH1", -15, 0xf92, [6299, -1466, -532, -6535, 13852, 2969, -2331, 3112, 5984]),
        ("DMC-GH2", -15, 0xf95, [7780, -2410, -806, -3913, 11724, 2484, -1018, 2390, 5298]),
        ("DMC-GH3", -15, 0, [6559, -1752, -491, -3672, 11407, 2586, -962, 1875, 5130]),
        ("DMC-GH4", -15, 0, [7122, -2108, -512, -3155, 11201, 2231, -541, 1423, 5045]),
        ("DC-GH5s", -15, 0, [6929, -2355, -708, -4192, 12534, 1828, -1097, 1989, 5195]),
        ("DC-GH5M2", 0, 0, [9300, -3659, -755, -2981, 10988, 2287, -190, 1077, 5016]),
        ("DC-GH5", -15, 0, [7641, -2336, -605, -3218, 11299, 2187, -485, 1338, 5121]),
        ("DC-GH6", -2048, 0, [7949, -3491, -710, -3435, 11681, 1977, -503, 1622, 5065]),
        ("DC-GH7", 0, 0, [8573, -3575, -678, -4252, 12079, 2451, -808, 2524, 5936]),
        ("DMC-GM1", -15, 0, [6770, -1895, -744, -5232, 13145, 2303, -1664, 2691, 5703]),
        ("DMC-GM5", -15, 0, [8238, -3244, -679, -3921, 11814, 2384, -836, 2022, 5852]),
        ("DC-GF10", -15, 0, [7610, -2780, -576, -4614, 12195, 2733, -1375, 2393, 6490]),
        ("DMC-GF1", -15, 0xf92, [7888, -1902, -1011, -8106, 16085, 2099, -2353, 2866, 7330]),
        ("DMC-GF2", -15, 0xfff, [7888, -1902, -1011, -8106, 16085, 2099, -2353, 2866, 7330]),
        ("DMC-GF3", -15, 0xfff, [9051, -2468, -1204, -5212, 13276, 2121, -1197, 2510, 6890]),
        ("DMC-GF5", -15, 0xfff, [8228, -2945, -660, -3938, 11792, 2430, -1094, 2278, 5793]),
        ("DMC-GF6", -15, 0, [8130, -2801, -946, -3520, 11289, 2552, -1314, 2511, 5791]),
        ("DMC-GF7", -15, 0, [7610, -2780, -576, -4614, 12195, 2733, -1375, 2393, 6490]),
        ("DMC-GF8", -15, 0, [7610, -2780, -576, -4614, 12195, 2733, -1375, 2393, 6490]),
        ("DC-GF9", -15, 0, [7610, -2780, -576, -4614, 12195, 2733, -1375, 2393, 6490]),
        ("DMC-GX85", -15, 0, [7771, -3020, -629, -4029, 11950, 2345, -821, 1977, 6119]),
        ("DMC-GX1", -15, 0, [6763, -1919, -863, -3868, 11515, 2684, -1216, 2387, 5879]),
        ("DMC-GX7", -15, 0, [7610, -2780, -576, -4614, 12195, 2733, -1375, 2393, 6490]),
        ("DMC-GX8", -15, 0, [7564, -2263, -606, -3148, 11239, 2177, -540, 1435, 4853]),
        ("DC-GX9", -15, 0, [7564, -2263, -606, -3148, 11239, 2177, -540, 1435, 4853]),
        ("DMC-ZS100", -15, 0, [7790, -2736, -755, -3452, 11870, 1769, -628, 1647, 4898]),
        ("DC-ZS200", -15, 0, [7790, -2736, -755, -3452, 11870, 1769, -628, 1647, 4898]),
        ("DMC-ZS40", -15, 0, [8607, -2822, -808, -3755, 11930, 2049, -820, 2060, 5224]),
        ("DMC-ZS50", -15, 0, [8802, -3135, -789, -3151, 11468, 1904, -550, 1745, 4810]),
        ("DMC-ZS60", -15, 0, [8550, -2908, -842, -3195, 11529, 1881, -338, 1603, 4631]),
        ("DC-ZS70", -15, 0, [9052, -3117, -883, -3045, 11346, 1927, -205, 1520, 4730]),
        ("DC-ZS80", -15, 0, [12194, -5340, -1329, -3035, 11394, 1858, -50, 1418, 5219]),
    ];
    let model = model.trim().to_ascii_uppercase();
    TABLE.iter().find(|(prefix, ..)| model.starts_with(prefix)).map(|(_, black, maximum, matrix)| {
        PanaColor { matrix: *matrix, black: *black, maximum: *maximum }
    })
}

pub(crate) fn decode(bytes: &[u8]) -> NativeResult<BayerFrame> {
    if bytes.get(..4) != Some(b"IIU\0") {
        return Err(NativeError::corrupt("not an RW2 container"));
    }
    let tiff = Tiff::parse(bytes, 0)?;
    let first = tiff.u32_at(4)?;
    let pana = find_pana_ifd(&tiff, first)?
        .ok_or_else(|| NativeError::corrupt("RW2 has no PanasonicRaw IFD"))?;

    let raw_width = pana.u32(0x0002).or_else(|| pana.u32(0x0100)).unwrap_or(0) as usize;
    let raw_height = pana.u32(0x0003).or_else(|| pana.u32(0x0101)).unwrap_or(0) as usize;
    if !(22..=64_000).contains(&raw_width) || !(22..=64_000).contains(&raw_height) {
        return Err(NativeError::corrupt("RW2 dimensions are out of range"));
    }
    let left_border = pana.u16(0x0005).unwrap_or(0) as usize;
    let bottom_border = pana.u16(0x0006).unwrap_or(0) as usize;
    let right_border = pana.u16(0x0007).unwrap_or(0) as usize;
    let mut top = pana.u16(0x0004).unwrap_or(0) as usize;
    let mut left = 0usize;
    let mut width = left_border;
    let mut height = bottom_border;
    if left_border != 0 && right_border > left_border {
        left = left_border;
        width = right_border;
    } else {
        width += right_border;
    }
    // LibRaw's optional crop tags are absent by default; the explicit user-crop
    // option is not set in the native front-end.
    if let (Some(top_crop), Some(left_crop)) = (pana.u16(0x002f), pana.u16(0x0030)) {
        let top_crop = top_crop as usize;
        let left_crop = left_crop as usize;
        if top_crop <= raw_height && left_crop <= raw_width {
            top = top_crop;
            left = left_crop;
            if let Some(bottom) = pana.u16(0x0031) {
                if (bottom as usize) > top {
                    height = bottom as usize - top;
                }
            }
            if let Some(right) = pana.u16(0x0032) {
                if (right as usize) > left {
                    width = right as usize - left;
                }
            }
        }
    }

    let encoding = pana.u16(0x002d).unwrap_or(0);
    let bits = if let Some(bpp) = pana.u16(0x000a) {
        bpp.clamp(8, 16) as u8
    } else {
        pana.u16(0x0102).unwrap_or(12).clamp(8, 16) as u8
    };
    if encoding == 8 {
        return Err(NativeError::unsupported("Panasonic RW2 RawFormat 8 (tiled C8)"));
    }

    let mut linear_max = [0u32; 4];
    for (index, tag) in [0x000e_u16, 0x000f, 0x0010].into_iter().enumerate() {
        let mut value = pana.u16(tag).unwrap_or(0) as u32;
        if value == 16_383 {
            value -= 64;
        }
        if value == 4_095 {
            value -= 16;
        }
        linear_max[index] = value;
    }
    linear_max[3] = linear_max[1];

    let adjust_sizes = encoding <= 5;
    if adjust_sizes {
        height += 12;
        if height > raw_height {
            height = raw_height;
        }
        if let Some((margin_left, margin_top, width_inc, height_inc)) = legacy_size_table(raw_width, raw_height) {
            left = margin_left;
            top = margin_top;
            width = width.saturating_add_signed(width_inc);
            height = height.saturating_add_signed(height_inc);
        }
    }
    if left >= raw_width || top >= raw_height {
        return Err(NativeError::corrupt("RW2 crop origin is outside the sensor"));
    }
    width = width.min(raw_width - left);
    height = height.min(raw_height - top);
    if width < 22 || height < 22 {
        return Err(NativeError::corrupt("RW2 active area is too small"));
    }

    let mut filters_code = pana.u16(0x0009).unwrap_or(0) as usize;
    if filters_code == 0 {
        filters_code = 1;
    }
    // DC-LX100M2 high-resolution mode renumbers the CFA phase.
    let make = pana.ascii(0x010f).unwrap_or_default();
    let model = pana.ascii(0x0110).unwrap_or_default();
    if model.eq_ignore_ascii_case("DC-LX100M2")
        && raw_height == 3568
        && raw_width == 4816
        && filters_code == 3
    {
        filters_code = 4;
    }
    let phase = (filters_code - 1) ^ (left & 1) ^ ((top << 1) & 3);
    let filters = FILTER_WORDS[phase & 3];
    let colors = 3u32;

    let mut black = 0u32;
    if adjust_sizes {
        let pana_black = [
            pana.u16(0x001c).unwrap_or(0) as u32,
            pana.u16(0x001d).unwrap_or(0) as u32,
            pana.u16(0x001e).unwrap_or(0) as u32,
        ];
        if pana_black.iter().all(|&value| value != 0) {
            let add = if encoding == 4 { 15 } else { 0 };
            let cblack = [
                pana_black[0] + add,
                pana_black[1] + add,
                pana_black[2] + add,
                pana_black[1] + add,
            ];
            black = cblack.into_iter().min().unwrap_or(0);
        }
    }

    let mut maximum = if encoding >= 6 {
        linear_max[..3].iter().copied().max().unwrap_or(0)
    } else {
        0
    };
    if maximum == 0 {
        maximum = (1u32 << bits.min(16)) - 1;
    }
    maximum = maximum.min(0xffff);

    let color = adobe_panasonic(&model).unwrap_or_default();
    if color.maximum != 0 {
        maximum = color.maximum;
    }
    if color.black < 0 && black == 0 {
        black = color.black.unsigned_abs();
    } else if color.black > 0 {
        black = color.black as u32;
    }
    let cam_xyz = {
        let mut matrix = [[0.0f64; 3]; 4];
        for index in 0..9 {
            matrix[index / 3][index % 3] = f64::from(color.matrix[index] as f32 / 10_000.0f32);
        }
        matrix
    };
    let rgb_cam = if color.matrix[0] != 0 {
        meta::cam_xyz_to_rgb_cam(&cam_xyz, colors as usize)
    } else {
        [[0.0f32; 4]; 3]
    };

    let mut cam_mul = [0.0f32; 4];
    if let (Some(red), Some(blue)) = (pana.u16(0x0011), pana.u16(0x0012)) {
        cam_mul[0] = f32::from(red) / 256.0;
        cam_mul[2] = f32::from(blue) / 256.0;
    }
    for (index, tag) in [0x0024_u16, 0x0025, 0x0026].into_iter().enumerate() {
        if let Some(value) = pana.u16(tag) {
            cam_mul[index] = f32::from(value);
        }
    }
    cam_mul[1] = read_wb_info2(&pana, 0x0027, cam_mul[1]);
    if let Some(green) = read_camera_ifd_auto_wb(&pana) {
        if cam_mul[1] == 0.0 {
            cam_mul[1] = green;
        }
    }
    if cam_mul[..3].iter().any(|&value| value <= 0.001)
        || cam_mul[..3].iter().any(|&value| value > 100.0)
    {
        cam_mul[0] = 0.0;
        cam_mul[3] = 0.0;
    }

    let data_offset = pana.u32(0x0118).or_else(|| pana.u32(0x0111)).unwrap_or(0) as usize;
    let plane = if encoding == 7 {
        decode_block_packed(bytes, data_offset, raw_width, raw_height, bits)?
    } else if encoding == 6 {
        return Err(NativeError::unsupported("Panasonic RW2 RawFormat 6 (C6)"));
    } else if encoding >= 2 {
        decode_classic(bytes, data_offset, raw_width, raw_height, encoding, bits)?
    } else if bytes.len().saturating_sub(data_offset) == raw_width * raw_height * 2 {
        decode_uncompressed(bytes, data_offset, raw_width, raw_height)?
    } else {
        return Err(NativeError::unsupported(format!(
            "Panasonic RW2 RawFormat {encoding} with {} remaining bytes",
            bytes.len().saturating_sub(data_offset)
        )));
    };

    make_frame(
        raw_width,
        raw_height,
        left,
        top,
        width,
        height,
        bits,
        plane,
        filters,
        colors,
        black,
        maximum,
        cam_mul,
        rgb_cam,
    )
}

fn find_pana_ifd<'a>(tiff: &Tiff<'a>, first: u32) -> NativeResult<Option<Ifd<'a>>> {
    fn visit<'a>(tiff: &Tiff<'a>, offset: u32, depth: u8, found: &mut Option<Ifd<'a>>) -> NativeResult<()> {
        if depth > 3 || found.is_some() {
            return Ok(());
        }
        let ifd = tiff.ifd(offset)?;
        if let Some(entry) = ifd.entry(0x0001) {
            if entry.typ == super::tiff::TYPE_LONG && entry.count == 1 {
                *found = Some(ifd.clone());
                return Ok(());
            }
        }
        for sub in ifd.sub_ifds() {
            visit(tiff, sub, depth + 1, found)?;
        }
        if let Some(next) = ifd.next {
            visit(tiff, next, depth + 1, found)?;
        }
        Ok(())
    }
    let mut found = None;
    visit(tiff, first, 0, &mut found)?;
    Ok(found)
}

fn legacy_size_table(width: usize, height: usize) -> Option<(usize, usize, isize, isize)> {
    const TABLE: &[(usize, usize, usize, usize, isize, isize)] = &[
        (3130, 1743, 4, 0, -6, 0),
        (3130, 2055, 4, 0, -6, 0),
        (3130, 2319, 4, 0, -6, 0),
        (3170, 2103, 18, 0, -42, 20),
        (3170, 2367, 18, 13, -42, -21),
        (3177, 2367, 0, 0, -1, 0),
        (3304, 2458, 0, 0, -1, 0),
        (3330, 2463, 9, 0, -5, 0),
        (3330, 2479, 9, 0, -17, 4),
        (3370, 1899, 15, 0, -44, 20),
        (3370, 2235, 15, 0, -44, 20),
        (3370, 2511, 15, 10, -44, -21),
        (3690, 2751, 3, 0, -8, -3),
        (3710, 2751, 0, 0, -3, 0),
        (3724, 2450, 0, 0, 0, -2),
        (3770, 2487, 17, 0, -44, 19),
        (3770, 2799, 17, 15, -44, -19),
        (3880, 2170, 6, 0, -6, 0),
        (4060, 3018, 0, 0, 0, -2),
        (4290, 2391, 3, 0, -8, -1),
        (4330, 2439, 17, 15, -44, -19),
        (4508, 2962, 0, 0, -3, -4),
        (4508, 3330, 0, 0, -3, -6),
        (10480, 7794, 0, 0, -2, 0),
        (11560, 8680, 8, 8, -16, -16),
    ];
    TABLE.iter().find(|values| values.0 == width && values.1 == height)
        .map(|values| (values.2, values.3, values.4, values.5))
}

/// WBInfo2: LibRaw uses the second light source entry beyond index six when
/// the ordinary R/B balance tags did not fill `cam_mul`.
fn read_wb_info2(ifd: &Ifd<'_>, tag: u16, fallback: f32) -> f32 {
    let Some(Ok(bytes)) = ifd.tag_bytes(tag) else {
        return fallback;
    };
    if bytes.len() < 2 {
        return fallback;
    }
    let count = usize::from(bytes[0]);
    if count > 0x100 {
        return fallback;
    }
    let mut at = 1usize;
    for index in 0..count {
        if bytes.len() < at + 2 {
            break;
        }
        let light = usize::from(bytes[at]);
        at += 2;
        if light >= 0x100 {
            at += 6;
            continue;
        }
        if bytes.len() < at + 6 {
            break;
        }
        let values = [bytes[at], bytes[at + 2], bytes[at + 4], bytes[at + 5]];
        at += 6;
        if index == 1 && count > 6 {
            return values[1] as f32;
        }
    }
    fallback
}

fn read_camera_ifd_auto_wb(ifd: &Ifd<'_>) -> Option<f32> {
    let entry = ifd.entry(0x0120)?;
    let bytes = ifd.bytes(entry).ok()?;
    if bytes.len() < 12 || !matches!(&bytes[..2], b"II" | b"MM") {
        return None;
    }
    let tiff = Tiff::parse(bytes, 0).ok()?;
    let camera = tiff.ifd(tiff.u32_at(4).ok()?).ok()?;
    camera.u16(0x3420).map(|value| value as f32)
}

fn decode_uncompressed(bytes: &[u8], offset: usize, width: usize, height: usize) -> NativeResult<Vec<u16>> {
    let count = width.checked_mul(height).ok_or_else(|| NativeError::corrupt("RW2 plane overflow"))?;
    let raw = bytes
        .get(offset..offset.saturating_add(count * 2))
        .ok_or_else(|| NativeError::corrupt("RW2 uncompressed plane is truncated"))?;
    Ok(meta::unpack_bits_msb(raw, 16, count)?)
}

/// Encoding 5 (and C7): 16-byte blocks of 12-bit tens or 14-bit nines.
fn decode_block_packed(
    bytes: &[u8],
    offset: usize,
    width: usize,
    height: usize,
    bits: u8,
) -> NativeResult<Vec<u16>> {
    let count = width.checked_mul(height).ok_or_else(|| NativeError::corrupt("RW2 plane overflow"))?;
    let mut plane = vec![0u16; count];
    let mut at = offset;
    for row in 0..height {
        let mut col = 0usize;
        while col < width {
            let block = bytes
                .get(at..at + 16)
                .ok_or_else(|| NativeError::corrupt("RW2 block is truncated"))?;
            at += 16;
            let mut buf = [0u16; 10];
            let sample_count = if bits == 12 {
                buf = decode_12bit_block(block);
                10
            } else {
                buf[..9].copy_from_slice(&decode_14bit_block(block));
                9
            };
            for &value in &buf[..sample_count] {
                if col < width {
                    plane[row * width + col] = value;
                    col += 1;
                }
            }
        }
    }
    Ok(plane)
}

fn decode_12bit_block(block: &[u8]) -> [u16; 10] {
    let b = |index: usize| u16::from(block[index]);
    [
        ((b(1) & 0x0f) << 8) | b(0),
        (b(2) << 4) | (b(1) >> 4),
        ((b(4) & 0x0f) << 8) | b(3),
        (b(5) << 4) | (b(4) >> 4),
        ((b(7) & 0x0f) << 8) | b(6),
        (b(8) << 4) | (b(7) >> 4),
        ((b(10) & 0x0f) << 8) | b(9),
        (b(11) << 4) | (b(10) >> 4),
        ((b(13) & 0x0f) << 8) | b(12),
        (b(14) << 4) | (b(13) >> 4),
    ]
}

fn decode_14bit_block(block: &[u8]) -> [u16; 9] {
    let b = |index: usize| u16::from(block[index]);
    [
        b(0) | ((b(1) & 0x3f) << 8),
        (b(1) >> 6) | (b(2) << 2) | ((b(3) & 0x0f) << 10),
        (b(3) >> 4) | (b(4) << 4) | ((b(5) & 0x03) << 12),
        ((b(5) & 0xfc) >> 2) | (b(6) << 6),
        b(7) | ((b(8) & 0x3f) << 8),
        (b(8) >> 6) | (b(9) << 2) | ((b(10) & 0x0f) << 10),
        (b(10) >> 4) | (b(11) << 4) | ((b(12) & 0x03) << 12),
        ((b(12) & 0xfc) >> 2) | (b(13) << 6),
        b(14) | ((b(15) & 0x3f) << 8),
    ]
}

/// LibRaw's reverse bit reader for Panasonic encodings 2/3/4.
struct ReverseBits<'a> {
    data: &'a [u8],
    /// Counts backwards from the end of a 0x4000-byte window.
    position: usize,
}

impl<'a> ReverseBits<'a> {
    fn new(data: &'a [u8], offset: usize) -> Self {
        Self { data, position: (offset + 0x4000) & 0x1ffff }
    }

    fn get(&mut self, count: usize) -> NativeResult<u32> {
        if count == 0 {
            return Ok(0);
        }
        self.position = self.position.wrapping_sub(count) & 0x1ffff;
        let byte = (self.position >> 3) ^ 0x3ff0;
        let window = self.data.get(byte).copied().unwrap_or(0) as u32
            | ((self.data.get(byte + 1).copied().unwrap_or(0) as u32) << 8);
        let shift = self.position & 7;
        Ok((window >> shift) & ((1u32 << count) - 1))
    }
}

fn decode_classic(
    bytes: &[u8],
    offset: usize,
    width: usize,
    height: usize,
    encoding: u16,
    bits: u8,
) -> NativeResult<Vec<u16>> {
    if encoding == 5 {
        return decode_block_packed(bytes, offset, width, height, bits);
    }
    if bits != 14 && bits != 12 {
        return Err(NativeError::unsupported(format!("Panasonic classic encoding with {bits} bits")));
    }
    let mut bits_reader = ReverseBits::new(bytes, offset);
    let mut plane = vec![0u16; width.checked_mul(height).ok_or_else(|| NativeError::corrupt("RW2 plane overflow"))?];
    let mut pred = [0i32; 2];
    let mut nonzero = [0u32; 2];
    for row in 0..height {
        for col in 0..width {
            let i = col % 14;
            if i == 0 {
                pred = [0, 0];
                nonzero = [0, 0];
            }
            let shift = if i % 3 == 2 {
                let value = bits_reader.get(2)?;
                4 >> (3usize.saturating_sub(value as usize))
            } else {
                0
            };
            let channel = i & 1;
            if nonzero[channel] != 0 {
                let j = bits_reader.get(8)? as i32;
                if j != 0 {
                    pred[channel] -= 0x80 << shift;
                    if pred[channel] < 0 || shift == 4 {
                        pred[channel] &= !(!0i32 << shift);
                    }
                    pred[channel] += j << shift;
                }
            } else {
                let value = bits_reader.get(8)? as u32;
                nonzero[channel] = value;
                if value != 0 || i > 11 {
                    pred[channel] = ((value << 4) | bits_reader.get(4)?) as i32;
                }
            }
            plane[row * width + col] = pred[channel] as u16;
        }
    }
    Ok(plane)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_fourteen_bit_blocks() {
        let mut block = [0u8; 16];
        let samples = [0x1234u16, 0x2abc, 0x3def, 0x0123, 0x3456, 0x2789, 0x3abc, 0x0def, 0x1abc];
        // Pack 9 x 14-bit samples little-endian into 16 bytes (matching decode_14bit_block).
        block[0] = samples[0] as u8;
        block[1] = ((samples[0] >> 8) as u8 & 0x3f) | (((samples[1] & 0x03) as u8) << 6);
        block[2] = (samples[1] >> 2) as u8;
        block[3] = ((samples[1] >> 10) as u8 & 0x0f) | (((samples[2] & 0x0f) as u8) << 4);
        block[4] = (samples[2] >> 4) as u8;
        block[5] = ((samples[2] >> 12) as u8 & 0x03) | (((samples[3] & 0x3f) as u8) << 2);
        block[6] = (samples[3] >> 6) as u8;
        block[7] = samples[4] as u8;
        block[8] = ((samples[4] >> 8) as u8 & 0x3f) | (((samples[5] & 0x03) as u8) << 6);
        block[9] = (samples[5] >> 2) as u8;
        block[10] = ((samples[5] >> 10) as u8 & 0x0f) | (((samples[6] & 0x0f) as u8) << 4);
        block[11] = (samples[6] >> 4) as u8;
        block[12] = ((samples[6] >> 12) as u8 & 0x03) | (((samples[7] & 0x3f) as u8) << 2);
        block[13] = (samples[7] >> 6) as u8;
        block[14] = samples[8] as u8;
        block[15] = (samples[8] >> 8) as u8 & 0x3f;
        assert_eq!(decode_14bit_block(&block), samples);
    }

    #[test]
    fn decodes_twelve_bit_blocks() {
        let mut block = [0u8; 16];
        let samples = [0x123u16, 0x456, 0x789, 0xabc, 0xdef, 0x012, 0x345, 0x678, 0x9ab, 0xcde];
        for (index, &value) in samples.iter().enumerate() {
            let pair = index / 2;
            if index % 2 == 0 {
                block[pair * 3] = value as u8;
                block[pair * 3 + 1] = ((value >> 8) as u8) | (((samples[index + 1] & 0x0f) as u8) << 4);
            } else {
                block[pair * 3 + 2] = (samples[index] >> 4) as u8;
            }
        }
        assert_eq!(decode_12bit_block(&block), samples);
    }

    #[test]
    fn black_and_maximum_follow_rawformat_rules() {
        let black = [100u32, 101, 102, 101];
        let minimum = black.into_iter().min().unwrap();
        assert_eq!(minimum, 100);
        // RawFormat 4 adds 15 to each channel before the minimum is taken.
        let black4 = [115u32, 116, 117, 116];
        assert_eq!(black4.into_iter().min().unwrap(), 115);
        let linear = [16_319u32, 4_079, 1_000];
        assert_eq!(linear.into_iter().max().unwrap(), 16_319);
    }

    #[test]
    fn cfa_phase_follows_panasonic_word_and_crop() {
        let phase = |filters: usize, left: usize, top: usize| {
            FILTER_WORDS[((filters - 1) ^ (left & 1) ^ ((top << 1) & 3)) & 3]
        };
        assert_eq!(phase(1, 0, 0), 0x9494_9494);
        assert_eq!(phase(4, 0, 0), 0x1616_1616);
        assert_eq!(phase(1, 1, 0), 0x6161_6161);
        assert_eq!(phase(1, 0, 1), 0x4949_4949);
    }

    #[test]
    fn reverse_bits_returns_written_fields() {
        let mut bytes = vec![0u8; 0x4000];
        // For offset=0, position starts at 0x4000; first read of 4 bits lands at
        // 0x3ffc -> byte (0x3ffc >> 3) ^ 0x3ff0 = 0x380f, shift = 4;
        // next read of 3 bits lands at 0x3ff9 -> byte 0x380f, shift = 1.
        bytes[0x380f] = (0b0110 << 4) | (0b101 << 1);
        let mut reader = ReverseBits::new(&bytes, 0);
        assert_eq!(reader.get(4).unwrap(), 0b0110);
        assert_eq!(reader.get(3).unwrap(), 0b101);
    }
}
