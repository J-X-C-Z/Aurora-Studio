use std::{collections::BTreeMap, io, str};

const SCHEMAS: &[&str] = &[
    "1680", "1674", "1670", "1665", "1664", "166", "165", "152", "141", "140", "131", "130", "120",
    "110", "100", "91", "90", "82", "81", "80", "70", "50", "",
];
const MAX_MANIFEST_BYTES: usize = 1024 * 1024;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn word(bytes: &[u8], offset: usize) -> usize {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()) as usize
}

fn dword(bytes: &[u8], offset: usize) -> usize {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize
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
        let footer = &data[end..];
        let count = word(footer, 10);
        let directory_offset = dword(footer, 16);
        let directory_size = dword(footer, 12);
        // shortcut: C1 samples use single-disk ZIP32, extend when a ZIP64 EIP is available.
        if word(footer, 4) != 0
            || word(footer, 6) != 0
            || word(footer, 8) != count
            || count == u16::MAX as usize
            || directory_offset.checked_add(directory_size) != Some(end)
        {
            return Err(invalid("unsupported ZIP central directory"));
        }

        let mut directory = &data[directory_offset..end];
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
            let expanded_size = dword(directory, 24);
            let extra = &directory[name_end..extra_end];
            let content_size = match extra {
                [_, _, _, _] if dword(extra, 0) != 0 => dword(extra, 0),
                _ => expanded_size,
            };
            if content_size > expanded_size {
                return Err(invalid("EIP content length exceeds expanded length"));
            }
            let entry = Entry {
                offset: dword(directory, 42),
                flags: word(directory, 8),
                method: word(directory, 10),
                crc: dword(directory, 16) as u32,
                stored_size: dword(directory, 20),
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

    pub fn member(&self, name: &str) -> io::Result<&'a [u8]> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| invalid("EIP member missing"))?;
        // shortcut: C1 samples store uncompressed members, extend when a compressed EIP is available.
        if entry.method != 0
            || entry.flags & !(0x0008 | 0x0800) != 0
            || entry.stored_size != entry.expanded_size
        {
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
        if crc32fast::hash(payload) != entry.crc {
            return Err(invalid("EIP member checksum mismatch"));
        }
        Ok(&payload[..entry.content_size])
    }

    pub fn expanded_size(&self, name: &str) -> Option<usize> {
        self.entries.get(name).map(|entry| entry.expanded_size)
    }

    fn read_manifest(&self, name: &str) -> io::Result<Manifest> {
        if self.expanded_size(name).unwrap_or_default() > MAX_MANIFEST_BYTES {
            return Err(invalid("EIP manifest exceeds size limit"));
        }
        let xml =
            str::from_utf8(self.member(name)?).map_err(|_| invalid("EIP manifest is not UTF-8"))?;
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
