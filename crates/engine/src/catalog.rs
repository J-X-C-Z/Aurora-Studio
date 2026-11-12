//! Images, their metadata, and how a catalog organizes them.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Stable identity of one variant in the catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ImageId(pub u64);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorTag {
    #[default]
    None,
    Red,
    Orange,
    Yellow,
    Green,
    Blue,
    Pink,
    Purple,
}

impl ColorTag {
    pub const ALL: [Self; 8] = [
        Self::None,
        Self::Red,
        Self::Orange,
        Self::Yellow,
        Self::Green,
        Self::Blue,
        Self::Pink,
        Self::Purple,
    ];

    /// sRGB hex the interface paints for this tag.
    pub fn rgb(self) -> u32 {
        match self {
            Self::None => 0x000000,
            Self::Red => 0xE5484D,
            Self::Orange => 0xF2994A,
            Self::Yellow => 0xF2C94C,
            Self::Green => 0x46A758,
            Self::Blue => 0x3E8EF7,
            Self::Pink => 0xE93D82,
            Self::Purple => 0x8E4EC6,
        }
    }

    /// The `xmp:Label` name the tag is written as; empty for no tag.
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Red => "Red",
            Self::Orange => "Orange",
            Self::Yellow => "Yellow",
            Self::Green => "Green",
            Self::Blue => "Blue",
            Self::Pink => "Pink",
            Self::Purple => "Purple",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileKind {
    #[default]
    Raw,
    Jpeg,
    Tiff,
    Png,
    Heif,
    Dng,
}

/// Capture metadata read from the file; everything is optional because many files carry little.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Exif {
    pub camera_make: Option<String>,
    pub camera_model: Option<String>,
    pub lens: Option<String>,
    pub focal_length_mm: Option<f32>,
    pub aperture: Option<f32>,
    /// Seconds.
    pub shutter: Option<f32>,
    pub iso: Option<u32>,
    pub exposure_bias: Option<f32>,
    pub flash: Option<bool>,
    /// RFC 3339.
    pub captured_at: Option<String>,
    pub gps: Option<(f64, f64)>,
}

/// Editable descriptive metadata (IPTC core).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Iptc {
    pub title: String,
    pub description: String,
    pub creator: String,
    pub copyright: String,
    pub headline: String,
    pub city: String,
    pub country: String,
    pub job_id: String,
    /// IPTC keywords (2:25); shown in the metadata panel's keyword list.
    pub keywords: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageInfo {
    pub id: ImageId,
    pub path: PathBuf,
    pub file_name: String,
    pub kind: FileKind,
    pub width: u32,
    pub height: u32,
    pub file_size: u64,
    pub exif: Exif,
    pub iptc: Iptc,
    pub keywords: Vec<String>,
    /// 0 ..= 5; -1 means rejected.
    pub rating: i8,
    pub color_tag: ColorTag,
    /// Variant number within its file; 1 is the original.
    pub variant: u32,
    pub has_adjustments: bool,
    /// Soft delete: the file stays on disk, the trash collection lists it.
    pub trashed: bool,
}

impl ImageInfo {
    pub fn aspect(&self) -> f32 {
        self.width as f32 / self.height.max(1) as f32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CollectionKind {
    AllImages,
    RecentImports,
    RecentCaptures,
    Trash,
    Folder,
    Album,
    SmartAlbum,
    Project,
    Group,
}

/// A node in the library's collection tree.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Collection {
    pub id: u64,
    pub name: String,
    pub kind: CollectionKind,
    pub count: usize,
    pub children: Vec<Collection>,
}

/// Condition a smart album matches. The interface evaluates these; the engine
/// only stores them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SmartRule {
    RatingAtLeast(i8),
    ColorTag(ColorTag),
    KeywordContains(String),
    NameContains(String),
    CameraContains(String),
    LensContains(String),
    IsoAtLeast(u32),
    Edited(bool),
    Kind(FileKind),
}

/// An album, project or group the user made in the interface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UserCollection {
    pub id: u64,
    pub parent: Option<u64>,
    pub name: String,
    pub kind: CollectionKind,
    pub images: Vec<ImageId>,
    pub rules: Vec<SmartRule>,
    pub match_all: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_rule_round_trips_json() {
        let rules = vec![
            SmartRule::RatingAtLeast(4),
            SmartRule::ColorTag(ColorTag::Red),
            SmartRule::KeywordContains("bride".into()),
            SmartRule::NameContains("DSC".into()),
            SmartRule::CameraContains("Nikon".into()),
            SmartRule::LensContains("85mm".into()),
            SmartRule::IsoAtLeast(800),
            SmartRule::Edited(true),
            SmartRule::Kind(FileKind::Raw),
        ];
        let json = serde_json::to_string(&rules).unwrap();
        let back: Vec<SmartRule> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, rules);
    }

    #[test]
    fn user_collection_round_trips_json() {
        let album = UserCollection {
            id: 1 << 40 | 7,
            parent: Some(10),
            name: "Picks".into(),
            kind: CollectionKind::SmartAlbum,
            images: vec![ImageId(3), ImageId(9)],
            rules: vec![SmartRule::RatingAtLeast(5), SmartRule::Edited(false)],
            match_all: false,
        };
        let json = serde_json::to_string(&album).unwrap();
        let back: UserCollection = serde_json::from_str(&json).unwrap();
        assert_eq!(back, album);
    }
}
