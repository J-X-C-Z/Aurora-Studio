//! 线性 DNG 1.6（TIFF/EP 架构）编码器。
//!
//! 本模块把行优先、交错的 16-bit RGB 样本写成一个**最小但结构完整**的
//! LinearRaw DNG：小端 TIFF 头 + 单个 IFD（标签按 tag 升序）+ 4 字节对齐的
//! 数据区 + 未压缩的像素 strip。不依赖 `tiff` crate，字节全部自行组装。
//!
//! **静态验证，不等于 Adobe DNG 验证器通过/真机对拍。**
//! 下方单元测试只证明字节结构与标签取值正确；未做 Adobe DNG Validator
//! 合规验证，也未做原厂同图对拍或真机/真实照片往返验证。

use anyhow::{anyhow, bail, Result};

// TIFF 6.0 / TIFF-EP field types used here.
const TYPE_BYTE: u16 = 1;
const TYPE_ASCII: u16 = 2;
const TYPE_SHORT: u16 = 3;
const TYPE_LONG: u16 = 4;
const TYPE_RATIONAL: u16 = 5;
const TYPE_SRATIONAL: u16 = 10;

/// 输入图像：行优先、交错 RGB，每个样本 16 bit。
///
/// `pixels_rgb16.len()` 必须严格等于 `width * height * 3`，否则编码返回 Err。
#[derive(Debug, Clone, Copy)]
pub struct DngImage<'a> {
    /// 像素宽度（列数）。
    pub width: u32,
    /// 像素高度（行数）。
    pub height: u32,
    /// 行优先交错 RGB 样本；第 i 个像素占用 `[3*i, 3*i+3)`。
    pub pixels_rgb16: &'a [u16],
    /// 写入 DNG UniqueCameraModel（同时复用为 Model 标签）；必须为 7-bit ASCII 且不含内嵌 NUL。
    pub unique_camera_model: &'a str,
}

/// 一个待写入 IFD 的条目；`bytes` 始终是**小端**排布的值字节。
#[derive(Debug, Clone)]
struct IfdEntry {
    tag: u16,
    ty: u16,
    count: u32,
    bytes: Vec<u8>,
    /// Some(off) 表示值放在 IFD 后的数据区，value/offset 字段写该偏移；
    /// None 表示值（<=4 字节）直接内联在 value/offset 字段中。
    out_of_line_offset: Option<u32>,
}

fn type_size(ty: u16) -> Option<usize> {
    match ty {
        TYPE_BYTE | TYPE_ASCII => Some(1),
        TYPE_SHORT => Some(2),
        TYPE_LONG => Some(4),
        TYPE_RATIONAL | TYPE_SRATIONAL => Some(8),
        _ => None,
    }
}

fn make_entry(tag: u16, ty: u16, bytes: Vec<u8>, count: usize) -> Result<IfdEntry> {
    let sz = type_size(ty).ok_or_else(|| anyhow!("unsupported TIFF type {ty} for tag {tag}"))?;
    let expected = sz
        .checked_mul(count)
        .ok_or_else(|| anyhow!("value size overflow for tag {tag}"))?;
    if bytes.len() != expected {
        bail!(
            "tag {tag}: value byte length {} != type size {sz} * count {count}",
            bytes.len()
        );
    }
    Ok(IfdEntry {
        tag,
        ty,
        count: u32::try_from(count)?,
        bytes,
        out_of_line_offset: None,
    })
}

fn byte_entry(tag: u16, values: &[u8]) -> Result<IfdEntry> {
    make_entry(tag, TYPE_BYTE, values.to_vec(), values.len())
}

fn short_entry(tag: u16, values: &[u16]) -> Result<IfdEntry> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(2));
    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    make_entry(tag, TYPE_SHORT, bytes, values.len())
}

fn long_entry(tag: u16, values: &[u32]) -> Result<IfdEntry> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    make_entry(tag, TYPE_LONG, bytes, values.len())
}

fn rational_entry(tag: u16, values: &[(u32, u32)]) -> Result<IfdEntry> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(8));
    for &(num, den) in values {
        bytes.extend_from_slice(&num.to_le_bytes());
        bytes.extend_from_slice(&den.to_le_bytes());
    }
    make_entry(tag, TYPE_RATIONAL, bytes, values.len())
}

fn srational_entry(tag: u16, values: &[(i32, i32)]) -> Result<IfdEntry> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(8));
    for &(num, den) in values {
        bytes.extend_from_slice(&num.to_le_bytes());
        bytes.extend_from_slice(&den.to_le_bytes());
    }
    make_entry(tag, TYPE_SRATIONAL, bytes, values.len())
}

fn ascii_entry(tag: u16, value: &str) -> Result<IfdEntry> {
    let raw = value.as_bytes();
    if raw.iter().any(|&b| b == 0 || b > 0x7f) {
        bail!("tag {tag}: ASCII value must be 7-bit and contain no interior NUL");
    }
    let mut bytes = Vec::with_capacity(raw.len().saturating_add(1));
    bytes.extend_from_slice(raw);
    bytes.push(0); // TIFF ASCII 以 NUL 结尾，且计入 count。
    let count = bytes.len();
    make_entry(tag, TYPE_ASCII, bytes, count)
}

/// 把 `x` 向上取整到 4 字节边界，使用 checked 加法。
fn align4(x: u32) -> Result<u32> {
    x.checked_add(3)
        .map(|v| v & !3u32)
        .ok_or_else(|| anyhow!("DNG offset alignment overflow"))
}

/// 编码线性 16-bit DNG 1.6 字节流。
///
/// 返回小端 TIFF 字节；任何长度/偏移溢出或非法输入均以 `Err` 返回，不 panic。
pub fn encode_linear_dng16(image: &DngImage<'_>) -> Result<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        bail!("DNG dimensions must both be nonzero");
    }

    // 像素数量严格匹配，避免静默截断。
    let samples = usize::try_from(image.width)?
        .checked_mul(usize::try_from(image.height)?)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| anyhow!("pixel count overflow"))?;
    if image.pixels_rgb16.len() != samples {
        bail!(
            "pixel buffer length mismatch: expected {samples} u16 samples, got {}",
            image.pixels_rgb16.len()
        );
    }
    let strip_bytes = u32::try_from(
        u64::from(image.width)
            .checked_mul(u64::from(image.height))
            .and_then(|v| v.checked_mul(6))
            .ok_or_else(|| anyhow!("strip byte count overflow"))?,
    )?;

    // 预先验证 ASCII 合法性：Model 与 UniqueCameraModel 共用同一个字符串。
    ascii_entry(50708, image.unique_camera_model)?;

    let mut entries = vec![
        long_entry(256, &[image.width])?,                 // ImageWidth
        long_entry(257, &[image.height])?,                // ImageLength
        short_entry(258, &[16, 16, 16])?,                 // BitsPerSample
        short_entry(259, &[1])?,                          // Compression = none
        short_entry(262, &[34892])?,                      // PhotometricInterpretation = LinearRaw
        ascii_entry(271, "Aurora Studio")?,               // Make
        ascii_entry(272, image.unique_camera_model)?,     // Model
        long_entry(273, &[0])?,                           // StripOffsets (patched below)
        short_entry(277, &[3])?,                          // SamplesPerPixel
        long_entry(278, &[image.height])?,                // RowsPerStrip
        long_entry(279, &[strip_bytes])?,                 // StripByteCounts
        short_entry(284, &[1])?,                          // PlanarConfiguration = chunky
        ascii_entry(305, "Aurora Studio Linear DNG Encoder")?, // Software
        byte_entry(50706, &[1, 6, 0, 0])?,                // DNGVersion 1.6.0.0
        byte_entry(50707, &[1, 4, 0, 0])?,                // DNGBackwardVersion 1.4.0.0
        ascii_entry(50708, image.unique_camera_model)?,   // UniqueCameraModel
        long_entry(50714, &[0, 0, 0])?,                   // BlackLevel (3 channels)
        long_entry(50717, &[65535, 65535, 65535])?,       // WhiteLevel (3 channels)
        // ColorMatrix1: 3x3 identity, nine real SRATIONAL values in the data area.
        srational_entry(
            50721,
            &[
                (1, 1), (0, 1), (0, 1),
                (0, 1), (1, 1), (0, 1),
                (0, 1), (0, 1), (1, 1),
            ],
        )?,
        rational_entry(50728, &[(1, 1), (1, 1), (1, 1)])?, // AsShotNeutral
        srational_entry(50730, &[(0, 1)])?,               // BaselineExposure 0 EV
        short_entry(50778, &[21])?,                       // CalibrationIlluminant1 = D65
    ];
    entries.sort_by_key(|e| e.tag); // TIFF 要求 IFD 条目按 tag 升序。

    // 布局：Header(8) -> IFD -> 数据区(4 对齐) -> 像素 strip。
    let ifd_size = u32::try_from(
        usize::try_from(entries.len())?
            .checked_mul(12)
            .and_then(|v| v.checked_add(2 + 4)) // entry count + next-IFD offset
            .ok_or_else(|| anyhow!("IFD size overflow"))?,
    )?;
    let data_start = align4(8u32.checked_add(ifd_size).ok_or_else(|| anyhow!("IFD offset overflow"))?)?;

    let mut cursor = data_start;
    for e in &mut entries {
        if e.bytes.len() > 4 {
            e.out_of_line_offset = Some(cursor);
            cursor = align4(
                cursor
                    .checked_add(u32::try_from(e.bytes.len())?)
                    .ok_or_else(|| anyhow!("data area offset overflow"))?,
            )?;
        }
    }
    let strip_offset = cursor; // cursor 在每次分配后已 4 对齐。

    // 回填 StripOffsets（内联 4 字节 LONG，不改变布局）。
    match entries.iter_mut().find(|e| e.tag == 273) {
        Some(e) => e.bytes = strip_offset.to_le_bytes().to_vec(),
        None => bail!("internal error: StripOffsets entry missing"),
    }

    let total = strip_offset
        .checked_add(strip_bytes)
        .ok_or_else(|| anyhow!("DNG total size overflow"))?;

    let mut out = Vec::with_capacity(usize::try_from(total)?);
    // --- TIFF header ---
    out.extend_from_slice(b"II");
    out.extend_from_slice(&42u16.to_le_bytes());
    out.extend_from_slice(&8u32.to_le_bytes());

    // --- IFD0 ---
    out.extend_from_slice(&u16::try_from(entries.len())?.to_le_bytes());
    for e in &entries {
        out.extend_from_slice(&e.tag.to_le_bytes());
        out.extend_from_slice(&e.ty.to_le_bytes());
        out.extend_from_slice(&e.count.to_le_bytes());
        match e.out_of_line_offset {
            Some(off) => out.extend_from_slice(&off.to_le_bytes()),
            None => {
                let mut field = [0u8; 4];
                match field.get_mut(..e.bytes.len()) {
                    Some(dst) => dst.copy_from_slice(&e.bytes),
                    None => bail!("internal error: inline value longer than 4 bytes"),
                }
                out.extend_from_slice(&field);
            }
        }
    }
    out.extend_from_slice(&0u32.to_le_bytes()); // 无后续 IFD

    // --- 数据区：补齐到 data_start，再按分配顺序写出所有外置值 ---
    let after_ifd = u32::try_from(out.len())?;
    if after_ifd < data_start {
        out.resize(out.len() + usize::try_from(data_start - after_ifd)?, 0);
    }
    for e in &entries {
        if e.out_of_line_offset.is_some() {
            out.extend_from_slice(&e.bytes);
            let rem = e.bytes.len() % 4;
            if rem != 0 {
                out.resize(out.len() + (4 - rem), 0);
            }
        }
    }
    if u32::try_from(out.len())? != strip_offset {
        bail!("internal error: strip offset layout mismatch");
    }

    // --- 像素 strip：u16 小端、行优先交错，长度严格为 strip_bytes ---
    for &sample in image.pixels_rgb16 {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    if out.len() != usize::try_from(total)? {
        bail!("internal error: encoded size mismatch");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    // 独立的极简 TIFF 解析器：只读原始字节，不复用编码器的常量或构造器。
    use std::collections::HashMap;

    const T_BYTE: u16 = 1;
    const T_ASCII: u16 = 2;
    const T_SHORT: u16 = 3;
    const T_LONG: u16 = 4;
    const T_RATIONAL: u16 = 5;
    const T_SRATIONAL: u16 = 10;

    fn local_type_size(ty: u16) -> Option<usize> {
        match ty {
            1 | 2 => Some(1),
            3 => Some(2),
            4 => Some(4),
            5 | 10 => Some(8),
            _ => None,
        }
    }

    #[derive(Debug, Clone)]
    struct RawEntry {
        #[allow(dead_code)] // 保留 tag 便于调试转储；查询通过 HashMap key 完成。
        tag: u16,
        ty: u16,
        count: u32,
        value_field: [u8; 4],
    }

    struct TinyTiff {
        buf: Vec<u8>,
        entries: HashMap<u16, RawEntry>,
        tags_in_order: Vec<u16>,
        ifd_end: usize,
    }

    impl TinyTiff {
        fn parse(buf: &[u8]) -> Result<Self, String> {
            if buf.len() < 8 {
                return Err("buffer shorter than TIFF header".into());
            }
            if &buf[0..2] != b"II" {
                return Err("not little-endian TIFF".into());
            }
            let magic = u16::from_le_bytes([buf[2], buf[3]]);
            if magic != 42 {
                return Err(format!("bad magic {magic}"));
            }
            let ifd_off = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
            if ifd_off + 2 > buf.len() {
                return Err("IFD offset out of range".into());
            }
            let n = u16::from_le_bytes([buf[ifd_off], buf[ifd_off + 1]]) as usize;
            let mut entries = HashMap::new();
            let mut tags_in_order = Vec::with_capacity(n);
            for i in 0..n {
                let p = ifd_off + 2 + i * 12;
                let slice = buf
                    .get(p..p + 12)
                    .ok_or_else(|| format!("IFD entry {i} out of range"))?;
                let tag = u16::from_le_bytes([slice[0], slice[1]]);
                let ty = u16::from_le_bytes([slice[2], slice[3]]);
                let count = u32::from_le_bytes([slice[4], slice[5], slice[6], slice[7]]);
                let mut value_field = [0u8; 4];
                value_field.copy_from_slice(&slice[8..12]);
                entries.insert(
                    tag,
                    RawEntry {
                        tag,
                        ty,
                        count,
                        value_field,
                    },
                );
                tags_in_order.push(tag);
            }
            Ok(Self {
                buf: buf.to_vec(),
                entries,
                tags_in_order,
                ifd_end: ifd_off + 2 + n * 12 + 4,
            })
        }

        fn get(&self, tag: u16) -> &RawEntry {
            self.entries
                .get(&tag)
                .unwrap_or_else(|| panic!("missing TIFF tag {tag}"))
        }

        fn type_and_count(&self, tag: u16) -> (u16, u32) {
            let e = self.get(tag);
            (e.ty, e.count)
        }

        fn value_bytes(&self, tag: u16) -> Vec<u8> {
            let e = self.get(tag);
            let sz = local_type_size(e.ty).expect("unknown TIFF type") * e.count as usize;
            if sz <= 4 {
                e.value_field[..sz].to_vec()
            } else {
                let off = u32::from_le_bytes(e.value_field) as usize;
                assert!(off >= self.ifd_end, "out-of-line value must live after IFD");
                self.buf
                    .get(off..off + sz)
                    .unwrap_or_else(|| panic!("out-of-line value for tag {tag} out of range"))
                    .to_vec()
            }
        }

        fn offset_of(&self, tag: u16) -> Option<usize> {
            let e = self.get(tag);
            let sz = local_type_size(e.ty).unwrap() * e.count as usize;
            if sz > 4 {
                Some(u32::from_le_bytes(e.value_field) as usize)
            } else {
                None
            }
        }

        fn u16s(&self, tag: u16) -> Vec<u16> {
            self.value_bytes(tag)
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        }

        fn u32s(&self, tag: u16) -> Vec<u32> {
            self.value_bytes(tag)
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }

        fn bytes(&self, tag: u16) -> Vec<u8> {
            self.value_bytes(tag)
        }

        fn ascii(&self, tag: u16) -> String {
            let b = self.bytes(tag);
            assert_eq!(*b.last().expect("ASCII must be NUL-terminated"), 0);
            let end = b.iter().position(|&x| x == 0).unwrap();
            String::from_utf8(b[..end].to_vec()).expect("ASCII value")
        }

        fn rationals(&self, tag: u16) -> Vec<(u32, u32)> {
            self.value_bytes(tag)
                .chunks_exact(8)
                .map(|c| {
                    (
                        u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                        u32::from_le_bytes([c[4], c[5], c[6], c[7]]),
                    )
                })
                .collect()
        }

        fn srationals(&self, tag: u16) -> Vec<(i32, i32)> {
            self.value_bytes(tag)
                .chunks_exact(8)
                .map(|c| {
                    (
                        i32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                        i32::from_le_bytes([c[4], c[5], c[6], c[7]]),
                    )
                })
                .collect()
        }
    }

    #[test]
    fn encode_gradient_and_parse_every_required_tag() {
        let (w, h) = (4u32, 3u32);
        let mut pixels: Vec<u16> = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                // 三个通道使用不同线性梯度，样本值互不重复。
                pixels.push(1000 + (x * 137 + y * 29) as u16);
                pixels.push(2000 + (x * 211 + y * 37) as u16);
                pixels.push(3000 + (x * 307 + y * 41) as u16);
            }
        }
        let model = "Aurora Test Camera 1";
        let image = super::DngImage {
            width: w,
            height: h,
            pixels_rgb16: &pixels,
            unique_camera_model: model,
        };
        let encoded = super::encode_linear_dng16(&image).expect("encode must succeed");
        let t = TinyTiff::parse(&encoded).expect("parse must succeed");

        // IFD 标签必须升序。
        let mut sorted = t.tags_in_order.clone();
        sorted.sort_unstable();
        assert_eq!(t.tags_in_order, sorted);

        // 基本几何/采样标签。
        assert_eq!(t.type_and_count(256), (T_LONG, 1));
        assert_eq!(t.u32s(256), vec![4]);
        assert_eq!(t.type_and_count(257), (T_LONG, 1));
        assert_eq!(t.u32s(257), vec![3]);
        assert_eq!(t.type_and_count(258), (T_SHORT, 3));
        assert_eq!(t.u16s(258), vec![16, 16, 16]);
        // BitsPerSample 是 6 字节数组，必须放在数据区而非内联。
        assert!(t.offset_of(258).is_some());
        assert_eq!(t.type_and_count(259), (T_SHORT, 1));
        assert_eq!(t.u16s(259), vec![1]);
        assert_eq!(t.type_and_count(262), (T_SHORT, 1));
        assert_eq!(t.u16s(262), vec![34892]); // LinearRaw
        assert_eq!(t.type_and_count(273), (T_LONG, 1));
        assert_eq!(t.type_and_count(277), (T_SHORT, 1));
        assert_eq!(t.u16s(277), vec![3]);
        assert_eq!(t.type_and_count(278), (T_LONG, 1));
        assert_eq!(t.u32s(278), vec![3]);
        assert_eq!(t.type_and_count(279), (T_LONG, 1));
        assert_eq!(t.u32s(279), vec![72]); // 4*3*3*2
        assert_eq!(t.type_and_count(284), (T_SHORT, 1));
        assert_eq!(t.u16s(284), vec![1]);

        // 可选但推荐的标签也做基本校验。
        assert_eq!(t.ascii(271), "Aurora Studio");
        assert_eq!(t.ascii(272), model);
        assert!(t.ascii(305).starts_with("Aurora Studio"));

        // DNG 核心标签。
        assert_eq!(t.type_and_count(50706), (T_BYTE, 4));
        assert_eq!(t.bytes(50706), vec![1, 6, 0, 0]);
        assert_eq!(t.type_and_count(50707), (T_BYTE, 4));
        assert_eq!(t.bytes(50707), vec![1, 4, 0, 0]);
        assert_eq!(t.type_and_count(50708), (T_ASCII, 21)); // 20 字符 + NUL
        assert_eq!(t.ascii(50708), model);
        assert_eq!(t.type_and_count(50714), (T_LONG, 3));
        assert_eq!(t.u32s(50714), vec![0, 0, 0]);
        assert_eq!(t.type_and_count(50717), (T_LONG, 3));
        assert_eq!(t.u32s(50717), vec![65535, 65535, 65535]);
        assert_eq!(t.type_and_count(50721), (T_SRATIONAL, 9));
        assert_eq!(
            t.srationals(50721),
            vec![
                (1, 1), (0, 1), (0, 1),
                (0, 1), (1, 1), (0, 1),
                (0, 1), (0, 1), (1, 1),
            ]
        );
        assert_eq!(t.type_and_count(50728), (T_RATIONAL, 3));
        assert_eq!(t.rationals(50728), vec![(1, 1), (1, 1), (1, 1)]);
        assert_eq!(t.type_and_count(50730), (T_SRATIONAL, 1));
        assert_eq!(t.srationals(50730), vec![(0, 1)]);
        assert_eq!(t.type_and_count(50778), (T_SHORT, 1));
        assert_eq!(t.u16s(50778), vec![21]); // D65

        // 像素 strip 与输入逐字节一致，0 差异。
        let off = t.u32s(273)[0] as usize;
        let len = t.u32s(279)[0] as usize;
        let strip = encoded
            .get(off..off + len)
            .unwrap_or_else(|| panic!("strip out of range"));
        let mut expected = Vec::with_capacity(len);
        for &p in &pixels {
            expected.extend_from_slice(&p.to_le_bytes());
        }
        assert_eq!(strip, expected.as_slice());
        assert_eq!(off + len, encoded.len()); // strip 位于文件末尾
    }

    #[test]
    fn mismatched_pixel_length_is_error() {
        let pixels = [1u16, 2, 3];
        let image = super::DngImage {
            width: 4,
            height: 3,
            pixels_rgb16: &pixels,
            unique_camera_model: "Bad",
        };
        assert!(super::encode_linear_dng16(&image).is_err());
    }

    #[test]
    fn zero_dimension_and_non_ascii_model_are_errors() {
        let pixels = [0u16; 3];
        let zero = super::DngImage {
            width: 0,
            height: 1,
            pixels_rgb16: &pixels,
            unique_camera_model: "X",
        };
        assert!(super::encode_linear_dng16(&zero).is_err());

        let bad = super::DngImage {
            width: 1,
            height: 1,
            pixels_rgb16: &pixels,
            unique_camera_model: "非 ASCII",
        };
        assert!(super::encode_linear_dng16(&bad).is_err());
    }
}
