use std::{borrow::Cow, collections::BTreeMap, io, str};

const SCHEMAS: &[&str] = &[
    "1680", "1674", "1670", "1665", "1664", "166", "165", "152", "141", "140", "131", "130", "120",
    "110", "100", "91", "90", "82", "81", "80", "70", "50", "",
];
const MAX_MANIFEST_BYTES: usize = 1024 * 1024;
const MAX_DEFLATED_MEMBER_BYTES: usize = 512 * 1024 * 1024;
const ZIP32_MAX: usize = u32::MAX as usize;
const ZIP16_MAX: usize = u16::MAX as usize;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn word(bytes: &[u8], offset: usize) -> usize {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()) as usize
}

fn dword(bytes: &[u8], offset: usize) -> usize {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize
}

fn zip64_word(bytes: &[u8], offset: usize) -> io::Result<usize> {
    let word = bytes.get(offset..offset + 8).ok_or_else(|| invalid("truncated ZIP64 field"))?;
    usize::try_from(u64::from_le_bytes(word.try_into().unwrap()))
        .map_err(|_| invalid("ZIP64 field exceeds platform size"))
}

fn directory_bounds(data: &[u8], end: usize) -> io::Result<(usize, usize, usize)> {
    let footer = &data[end..];
    let mut count = word(footer, 10);
    let mut size = dword(footer, 12);
    let mut start = dword(footer, 16);
    if word(footer, 4) != 0 || word(footer, 6) != 0 || word(footer, 8) != count {
        return Err(invalid("unsupported multi-disk EIP"));
    }
    let mut directory_end = end;
    if let Some(locator) = end.checked_sub(20).and_then(|offset| data.get(offset..end))
        .filter(|locator| locator.starts_with(b"PK\x06\x07"))
    {
        if dword(locator, 4) != 0 || dword(locator, 16) != 1 {
            return Err(invalid("unsupported multi-disk ZIP64 EIP"));
        }
        directory_end = zip64_word(locator, 8)?;
        let record = data.get(directory_end..end - 20)
            .filter(|record| record.len() >= 56 && record.starts_with(b"PK\x06\x06"))
            .ok_or_else(|| invalid("invalid ZIP64 end record"))?;
        if zip64_word(record, 4)?.checked_add(12) != Some(record.len())
            || dword(record, 16) != 0 || dword(record, 20) != 0
        {
            return Err(invalid("unsupported ZIP64 end record"));
        }
        let full_count = zip64_word(record, 32)?;
        let full_size = zip64_word(record, 40)?;
        let full_start = zip64_word(record, 48)?;
        if zip64_word(record, 24)? != full_count
            || (count != ZIP16_MAX && count != full_count)
            || (size != ZIP32_MAX && size != full_size)
            || (start != ZIP32_MAX && start != full_start)
        {
            return Err(invalid("ZIP and ZIP64 directory records differ"));
        }
        (count, size, start) = (full_count, full_size, full_start);
    } else if count == ZIP16_MAX || start == ZIP32_MAX || size == ZIP32_MAX {
        return Err(invalid("ZIP64 locator missing"));
    }
    if start.checked_add(size) != Some(directory_end) || count > size / 46 {
        return Err(invalid("invalid ZIP central directory"));
    }
    Ok((start, directory_end, count))
}

fn resolve_zip64_fields(extra: &[u8], fields: &mut [usize; 3]) -> io::Result<()> {
    let mut tags = extra;
    while tags.len() >= 4 {
        let field_end = 4 + word(tags, 2);
        let value = tags.get(4..field_end).ok_or_else(|| invalid("truncated ZIP extra field"))?;
        if word(tags, 0) == 1 {
            let mut offset = 0;
            for field in fields.iter_mut().filter(|field| **field == ZIP32_MAX) {
                *field = zip64_word(value, offset)?;
                offset += 8;
            }
            return Ok(());
        }
        tags = &tags[field_end..];
    }
    Err(invalid("ZIP64 member extra field missing"))
}

fn inflate(payload: &[u8], size: usize) -> io::Result<Vec<u8>> {
    if size > MAX_DEFLATED_MEMBER_BYTES {
        return Err(invalid("deflated EIP member exceeds size limit"));
    }
    use flate2::{Decompress, FlushDecompress, Status};
    let mut decoder = Decompress::new(false);
    let mut decoded = Vec::new();
    let mut chunk = [0; 32 * 1024];
    loop {
        let input_start = decoder.total_in() as usize;
        let output_start = decoder.total_out() as usize;
        let status = decoder.decompress(&payload[input_start..], &mut chunk, FlushDecompress::None)
            .map_err(|_| invalid("invalid EIP deflate stream"))?;
        let output_end = decoder.total_out() as usize;
        if output_end > size {
            return Err(invalid("EIP deflate length exceeds declared size"));
        }
        decoded.extend_from_slice(&chunk[..output_end - output_start]);
        if status == Status::StreamEnd {
            return if output_end == size && decoder.total_in() as usize == payload.len() {
                Ok(decoded)
            } else {
                Err(invalid("EIP deflate length mismatch"))
            };
        }
        if output_start == output_end && input_start == decoder.total_in() as usize {
            return Err(invalid("incomplete EIP deflate stream"));
        }
    }
}

struct Entry<'a> {
    offset: usize,
    flags: usize,
    method: usize,
    crc: u32,
    stored_size: usize,
    expanded_size: usize,
    content_size: usize,
    extra: &'a [u8],
}

#[derive(Default, Debug)]
pub struct Manifest {
    pub paths: BTreeMap<String, String>,
    pub lccs: Vec<(String, String)>,
    pub iccs: Vec<(String, String)>,
}

pub struct EipPackage<'a> {
    data: &'a [u8],
    entries: BTreeMap<&'a str, Entry<'a>>,
    directory_offset: usize,
    pub manifest_name: String,
    pub manifest: Manifest,
}

impl<'a> EipPackage<'a> {
    pub fn parse(data: &'a [u8]) -> io::Result<Self> {
        let end = (data.len().saturating_sub(65_557)..=data.len().saturating_sub(22))
            .rev()
            .find(|&offset| {
                let footer = &data[offset..];
                footer.len() >= 22
                    && footer.starts_with(b"PK\x05\x06")
                    && footer.len() == 22 + word(footer, 20)
            })
            .ok_or_else(|| invalid("ZIP end record missing"))?;
        let (directory_offset, directory_end, count) = directory_bounds(data, end)?;
        let mut directory = &data[directory_offset..directory_end];
        let mut entries = BTreeMap::new();
        for _ in 0..count {
            if directory.len() < 46 || !directory.starts_with(b"PK\x01\x02") {
                return Err(invalid("invalid ZIP central header"));
            }
            let name_end = 46 + word(directory, 28);
            let extra_end = name_end + word(directory, 30);
            let record_end = extra_end + word(directory, 32);
            if record_end > directory.len() || word(directory, 34) != 0 {
                return Err(invalid("invalid ZIP central record"));
            }
            let name = str::from_utf8(&directory[46..name_end])
                .map_err(|_| invalid("EIP member name is not UTF-8"))?;
            let extra = &directory[name_end..extra_end];
            let mut fields = [dword(directory, 24), dword(directory, 20), dword(directory, 42)];
            if fields.contains(&ZIP32_MAX) {
                resolve_zip64_fields(extra, &mut fields)?;
            }
            let [expanded_size, stored_size, offset] = fields;
            let content_size = match extra {
                [_, _, _, _] if dword(extra, 0) != 0 => dword(extra, 0),
                _ => expanded_size,
            };
            if content_size > expanded_size {
                return Err(invalid("EIP content length exceeds expanded length"));
            }
            let entry = Entry {
                offset,
                flags: word(directory, 8),
                method: word(directory, 10),
                crc: dword(directory, 16) as u32,
                stored_size,
                expanded_size,
                content_size,
                extra,
            };
            if entries.insert(name, entry).is_some() {
                return Err(invalid("duplicate EIP member"));
            }
            directory = &directory[record_end..];
        }
        if !directory.is_empty() {
            return Err(invalid("ZIP central directory length mismatch"));
        }

        let manifest_name = SCHEMAS
            .iter()
            .map(|schema| format!("manifest{schema}.xml"))
            .find(|name| entries.contains_key(name.as_str()))
            .ok_or_else(|| invalid("EIP manifest missing"))?;
        let mut package = Self {
            data,
            entries,
            directory_offset,
            manifest_name,
            manifest: Manifest::default(),
        };
        let mut manifest = package.read_manifest(&package.manifest_name)?;
        if package.manifest_name != "manifest.xml" && package.entries.contains_key("manifest.xml") {
            for (role, name) in package.read_manifest("manifest.xml")?.paths {
                manifest.paths.entry(role).or_insert(name);
            }
        }
        if !manifest.paths.contains_key("RawPath") {
            return Err(invalid("EIP manifest has no RawPath"));
        }
        package.manifest = manifest;
        Ok(package)
    }

    pub fn member(&self, name: &str) -> io::Result<Cow<'a, [u8]>> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| invalid("EIP member missing"))?;
        let allowed_flags = 0x0808 | if entry.method == 8 { 0x0006 } else { 0 };
        if !matches!(entry.method, 0 | 8) || entry.flags & !allowed_flags != 0 {
            return Err(invalid("unsupported EIP ZIP member encoding"));
        }
        let local = self.data[..self.directory_offset]
            .get(entry.offset..)
            .ok_or_else(|| invalid("invalid EIP local offset"))?;
        if local.len() < 30
            || !local.starts_with(b"PK\x03\x04")
            || word(local, 6) != entry.flags
            || word(local, 8) != entry.method
        {
            return Err(invalid("invalid EIP local header"));
        }

        let name_end = 30 + word(local, 26);
        let data_start = name_end + word(local, 28);
        if local.get(30..name_end) != Some(name.as_bytes())
            || (entry.extra.len() == 4 && local.get(name_end..data_start) != Some(entry.extra))
        {
            return Err(invalid("EIP local and central metadata differ"));
        }
        let data_end = data_start
            .checked_add(entry.stored_size)
            .ok_or_else(|| invalid("EIP member size overflow"))?;
        let payload = local
            .get(data_start..data_end)
            .ok_or_else(|| invalid("truncated EIP member"))?;
        let mut decoded = match entry.method {
            0 if entry.stored_size == entry.expanded_size => Cow::Borrowed(payload),
            8 => Cow::Owned(inflate(payload, entry.expanded_size)?),
            _ => return Err(invalid("EIP stored member size mismatch")),
        };
        if crc32fast::hash(&decoded) != entry.crc {
            return Err(invalid("EIP member checksum mismatch"));
        }
        match &mut decoded {
            Cow::Borrowed(bytes) => *bytes = &bytes[..entry.content_size],
            Cow::Owned(bytes) => bytes.truncate(entry.content_size),
        }
        Ok(decoded)
    }

    pub fn expanded_size(&self, name: &str) -> Option<usize> {
        self.entries.get(name).map(|entry| entry.expanded_size)
    }

    fn read_manifest(&self, name: &str) -> io::Result<Manifest> {
        if self.expanded_size(name).unwrap_or_default() > MAX_MANIFEST_BYTES {
            return Err(invalid("EIP manifest exceeds size limit"));
        }
        let bytes = self.member(name)?;
        let xml = str::from_utf8(&bytes).map_err(|_| invalid("EIP manifest is not UTF-8"))?;
        let document = roxmltree::Document::parse(xml)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let root = document.root_element();
        if !root.has_tag_name("Manifest") {
            return Err(invalid("invalid EIP manifest root"));
        }
        let file = root
            .children()
            .find(|child| child.has_tag_name("File"))
            .ok_or_else(|| invalid("EIP manifest has no File"))?;
        let mut manifest = Manifest::default();
        for node in file.children().filter(|node| node.is_element()) {
            let role = node.tag_name().name();
            match role {
                "RawPath" | "SettingsPath" | "ProxyPath" | "CoMaskPath" | "CoHairPath" => {
                    if let Some(path) = node.text().map(str::trim).filter(|path| !path.is_empty()) {
                        manifest.paths.insert(role.into(), path.into());
                    }
                }
                "LCCs" | "ICCs" => {
                    let (tag, id, resources) = if role == "LCCs" {
                        ("LCCPath", "LccID", &mut manifest.lccs)
                    } else {
                        ("ICCPath", "IccID", &mut manifest.iccs)
                    };
                    for resource in node.children().filter(|child| child.has_tag_name(tag)) {
                        if let Some(path) = resource
                            .text()
                            .map(str::trim)
                            .filter(|path| !path.is_empty())
                        {
                            resources
                                .push((resource.attribute(id).unwrap_or("").into(), path.into()));
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, path::Path};

    fn deflated_archive(zip64: bool, corrupt_crc: bool) -> Vec<u8> {
        use std::io::Write;

        let manifest = b"<Manifest><File><RawPath>photo.IIQ</RawPath></File></Manifest>";
        let raw = b"packed media fixture";
        let members: [(&[u8], &[u8]); 2] = [(b"manifest.xml", manifest), (b"photo.IIQ", raw)];
        let mut archive = Vec::new();
        let mut directory = Vec::new();
        for (name, input) in members {
            let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::fast());
            encoder.write_all(input).unwrap();
            let payload = encoder.finish().unwrap();
            let crc = crc32fast::hash(input) ^ u32::from(corrupt_crc && name == b"photo.IIQ");
            let offset = archive.len();
            let mut local = [0u8; 30];
            local[..4].copy_from_slice(b"PK\x03\x04");
            local[4] = if zip64 { 45 } else { 20 };
            local[8] = 8;
            local[14..18].copy_from_slice(&crc.to_le_bytes());
            let stored_size = if zip64 { u32::MAX } else { payload.len() as u32 };
            let expanded_size = if zip64 { u32::MAX } else { input.len() as u32 };
            local[18..22].copy_from_slice(&stored_size.to_le_bytes());
            local[22..26].copy_from_slice(&expanded_size.to_le_bytes());
            local[26..28].copy_from_slice(&(name.len() as u16).to_le_bytes());
            let mut extra = Vec::new();
            if zip64 {
                extra.extend_from_slice(&[1, 0, 16, 0]);
                extra.extend_from_slice(&(input.len() as u64).to_le_bytes());
                extra.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            }
            local[28..30].copy_from_slice(&(extra.len() as u16).to_le_bytes());
            archive.extend_from_slice(&local);
            archive.extend_from_slice(name);
            archive.extend_from_slice(&extra);
            archive.extend_from_slice(&payload);

            let mut central = [0u8; 46];
            central[..4].copy_from_slice(b"PK\x01\x02");
            central[4..6].copy_from_slice(&local[4..6]);
            central[6..16].copy_from_slice(&local[4..14]);
            central[16..30].copy_from_slice(&local[14..28]);
            let legacy_offset = if zip64 { u32::MAX } else { offset as u32 };
            central[42..46].copy_from_slice(&legacy_offset.to_le_bytes());
            if zip64 {
                extra[2] = 24;
                extra.extend_from_slice(&(offset as u64).to_le_bytes());
            }
            central[30..32].copy_from_slice(&(extra.len() as u16).to_le_bytes());
            directory.extend_from_slice(&central);
            directory.extend_from_slice(name);
            directory.extend_from_slice(&extra);
        }
        let start = archive.len();
        archive.extend_from_slice(&directory);
        let mut footer = [0u8; 22];
        footer[..4].copy_from_slice(b"PK\x05\x06");
        if zip64 {
            let record_offset = archive.len();
            let mut record = [0u8; 56];
            record[..4].copy_from_slice(b"PK\x06\x06");
            record[4..12].copy_from_slice(&44u64.to_le_bytes());
            record[12] = 45;
            record[14] = 45;
            record[24..32].copy_from_slice(&2u64.to_le_bytes());
            record[32..40].copy_from_slice(&2u64.to_le_bytes());
            record[40..48].copy_from_slice(&(directory.len() as u64).to_le_bytes());
            record[48..56].copy_from_slice(&(start as u64).to_le_bytes());
            archive.extend_from_slice(&record);
            archive.extend_from_slice(b"PK\x06\x07\0\0\0\0");
            archive.extend_from_slice(&(record_offset as u64).to_le_bytes());
            archive.extend_from_slice(&1u32.to_le_bytes());
            footer[8..20].fill(0xff);
        } else {
            footer[8..10].copy_from_slice(&2u16.to_le_bytes());
            footer[10..12].copy_from_slice(&2u16.to_le_bytes());
            footer[12..16].copy_from_slice(&(directory.len() as u32).to_le_bytes());
            footer[16..20].copy_from_slice(&(start as u32).to_le_bytes());
        }
        archive.extend_from_slice(&footer);
        archive
    }

    #[test]
    fn deflate_and_zip64_read_manifest_and_media_with_crc_validation() {
        for zip64 in [false, true] {
            let bytes = deflated_archive(zip64, false);
            let package = EipPackage::parse(&bytes).unwrap();
            assert_eq!(package.manifest.paths["RawPath"], "photo.IIQ");
            assert_eq!(package.member("photo.IIQ").unwrap().as_ref(), b"packed media fixture");
            let bytes = deflated_archive(zip64, true);
            assert_eq!(
                EipPackage::parse(&bytes).unwrap().member("photo.IIQ").unwrap_err().to_string(),
                "EIP member checksum mismatch"
            );
        }
    }

    #[test]
    fn deflate_rejects_wrong_length_and_trailing_bytes() {
        let mut stream = vec![0x73, 0x04, 0x00];
        assert_eq!(inflate(&stream, 1).unwrap(), b"A");
        assert!(inflate(&stream, 0).is_err());
        assert!(inflate(&stream, 2).is_err());
        assert!(inflate(&stream[..2], 1).is_err());
        assert!(inflate(&stream, MAX_DEFLATED_MEMBER_BYTES + 1).is_err());
        stream.push(0);
        assert!(inflate(&stream, 1).is_err());
    }

    #[test]
    fn original_c1_packages() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../../../04_reference/legacy-root/逆向工程/Capture One/resources/eip_samples",
        );
        let bytes = fs::read(fixtures.join("FULL.EIP")).unwrap();
        let package = EipPackage::parse(&bytes).unwrap();
        assert_eq!(package.manifest_name, "manifest1680.xml");
        let paths = &package.manifest.paths;
        assert_eq!(paths["ProxyPath"], "CaptureOne/Cache/Proxies/0.IIQ.jpg");
        assert_eq!(paths["CoMaskPath"], "CaptureOne/Settings1680/0.IIQ.comask");
        assert_eq!(paths["CoHairPath"], "CaptureOne/Settings1680/0.IIQ.cohair");
        assert_eq!(package.member(&paths["CoMaskPath"]).unwrap().len(), 4);
        assert_eq!(package.manifest.lccs.len(), 2);
        assert_eq!(package.manifest.iccs.len(), 1);

        let bytes = fs::read(fixtures.join("PACKAGE.EIP")).unwrap();
        let package = EipPackage::parse(&bytes).unwrap();
        assert!(!package.manifest.paths.contains_key("CoMaskPath"));
        let settings = &package.manifest.paths["SettingsPath"];
        assert_eq!(package.expanded_size(settings), Some(16_409));
        assert_eq!(package.member(settings).unwrap().len(), 25);
    }

    #[test]
    fn real_mask_padding_is_removed_exactly() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mask = manifest.join(
            "../../../90_work/lanes/c1-parse/win_retouchmask_r27/crates/engine/testdata/comask/_DSC5673.NEF.comask",
        );
        if !mask.is_file() {
            return;
        }
        let archive = std::env::temp_dir().join(format!("c1-eip-{}", std::process::id()));
        fs::create_dir_all(&archive).unwrap();
        let package_path = archive.join("fixture.EIP");
        let raw_path = archive.join("fixture.NEF");
        fs::write(&raw_path, b"fixture only").unwrap();
        let output = std::process::Command::new("python")
            .args(["-B", "-c", "import sys; sys.path.insert(0, sys.argv[1]); from formats.eip import create; create(sys.argv[2], sys.argv[3], comask_path=sys.argv[4])"])
            .arg(manifest.join("../../../../04_reference/legacy-root/逆向工程/Capture One/sdk"))
            .arg(&package_path)
            .arg(&raw_path)
            .arg(&mask)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );

        let bytes = fs::read(package_path).unwrap();
        let _ = fs::remove_dir_all(&archive);
        let package = EipPackage::parse(&bytes).unwrap();
        let name = &package.manifest.paths["CoMaskPath"];
        assert_eq!(name, "CaptureOne/Settings1680/0.NEF.comask");
        let source = fs::read(mask).unwrap();
        let restored = package.member(name).unwrap();
        assert_eq!(restored.len(), 1_270_726);
        assert_eq!(package.expanded_size(name), Some(source.len() + 16_384));
        assert_eq!(restored, source);

        let offset = package.entries[name.as_str()].offset;
        let name = name.clone();
        let payload = offset + 30 + word(&bytes[offset..], 26) + word(&bytes[offset..], 28);
        let mut damaged = bytes;
        damaged[payload + source.len()] ^= 1;
        let package = EipPackage::parse(&damaged).unwrap();
        assert_eq!(
            package.member(&name).unwrap_err().to_string(),
            "EIP member checksum mismatch"
        );
    }
}
