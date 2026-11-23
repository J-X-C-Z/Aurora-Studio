//! Vendor portrait model identities recovered from the private PixCake payload.
//!
//! This registry is intentionally stricter than filename lookup. An asset is
//! discoverable only when its numeric id and byte length are both known, while
//! runtime binding remains blocked until the stage mapping, container format and
//! tensor names are independently proven.

use std::io;
use std::path::{Path, PathBuf};

const HAIR_SMOOTH_PFS: &[u32] = &[90139, 90140];
const HAIR_GLOSS_PFS: &[u32] = &[90160];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VendorModelBlocker {
    ModelIdentityUnbound,
    StageMappingUnbound,
    ContainerFormatUnbound,
    TensorNamesUnbound,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VendorModelAsset {
    pub pfs: &'static [u32],
    pub stage: &'static str,
    pub model_id: Option<u32>,
    pub bytes: Option<u64>,
    pub sha256: Option<&'static str>,
    pub tensor_names: &'static [&'static str],
    pub blockers: &'static [VendorModelBlocker],
}

impl VendorModelAsset {
    pub fn runtime_ready(&self) -> bool {
        self.blockers.is_empty()
            && self.model_id.is_some()
            && self.sha256.is_some()
            && !self.tensor_names.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveredVendorModel {
    pub asset: VendorModelAsset,
    pub path: PathBuf,
}

pub const PF30026_MODEL: VendorModelAsset = VendorModelAsset {
    pfs: &[30026],
    stage: "teeth edge gate + effect RGB + edge planes A/B",
    model_id: None,
    bytes: None,
    sha256: None,
    tensor_names: &[],
    blockers: &[
        VendorModelBlocker::ModelIdentityUnbound,
        VendorModelBlocker::TensorNamesUnbound,
    ],
};

pub const PF90139_PF90140_MODEL_1416738152: VendorModelAsset = VendorModelAsset {
    pfs: HAIR_SMOOTH_PFS,
    stage: "hair smooth candidate 960x1280",
    model_id: Some(1_416_738_152),
    bytes: Some(552_397),
    sha256: Some("1a9608c0f84e58359d92da7fb358bb2d815e9da43847b907ff57909e879eedfe"),
    tensor_names: &[],
    blockers: &[
        VendorModelBlocker::StageMappingUnbound,
        VendorModelBlocker::ContainerFormatUnbound,
        VendorModelBlocker::TensorNamesUnbound,
    ],
};

pub const PF90139_PF90140_MODEL_3434447882: VendorModelAsset = VendorModelAsset {
    pfs: HAIR_SMOOTH_PFS,
    stage: "hair smooth candidate 768x1024",
    model_id: Some(3_434_447_882),
    bytes: Some(552_357),
    sha256: Some("5b869d37b10cc02e73d9f735367481600c613eed827a8f166a4d72e2a11dcc84"),
    tensor_names: &[],
    blockers: &[
        VendorModelBlocker::StageMappingUnbound,
        VendorModelBlocker::ContainerFormatUnbound,
        VendorModelBlocker::TensorNamesUnbound,
    ],
};

pub const PF90139_PF90140_MODEL_243887137: VendorModelAsset = VendorModelAsset {
    pfs: HAIR_SMOOTH_PFS,
    stage: "hair smooth pipeline auxiliary",
    model_id: Some(243_887_137),
    bytes: Some(686_032),
    sha256: Some("861d95fc8d746b3d1e94d396b430cb36f2ec1a0c6dd5d2e38b3b79bf9631fc6e"),
    tensor_names: &[],
    blockers: &[
        VendorModelBlocker::StageMappingUnbound,
        VendorModelBlocker::ContainerFormatUnbound,
        VendorModelBlocker::TensorNamesUnbound,
    ],
};

pub const PF90160_MODEL: VendorModelAsset = VendorModelAsset {
    pfs: HAIR_GLOSS_PFS,
    stage: "hair gloss 512x512",
    model_id: Some(2_502_924_957),
    bytes: Some(552_337),
    sha256: Some("d0839b3a969a06892bcd00dd897a6089a4428a38eba73fec3c08ad741aa837d0"),
    tensor_names: &[],
    blockers: &[
        VendorModelBlocker::ContainerFormatUnbound,
        VendorModelBlocker::TensorNamesUnbound,
    ],
};

pub const PORTRAIT_VENDOR_MODELS: &[VendorModelAsset] = &[
    PF30026_MODEL,
    PF90139_PF90140_MODEL_1416738152,
    PF90139_PF90140_MODEL_3434447882,
    PF90139_PF90140_MODEL_243887137,
    PF90160_MODEL,
];

/// Discover only exact private-pool basenames (`<model-id>.bin`) and byte lengths.
///
/// The caller still has to verify the recorded SHA-256 before attaching the
/// asset. This avoids recursive scans and never promotes a same-sized file or a
/// filename-only match to model identity.
pub fn discover_in_asset_dir(dir: &Path) -> io::Result<Vec<DiscoveredVendorModel>> {
    let mut found = Vec::new();
    for asset in PORTRAIT_VENDOR_MODELS {
        let (Some(model_id), Some(bytes)) = (asset.model_id, asset.bytes) else {
            continue;
        };
        let path = dir.join(format!("{model_id}.bin"));
        let metadata = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        if metadata.is_file() && metadata.len() == bytes {
            found.push(DiscoveredVendorModel {
                asset: *asset,
                path,
            });
        }
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovered_hair_assets_are_not_runtime_ready() {
        assert_eq!(PF90139_PF90140_MODEL_1416738152.model_id, Some(1_416_738_152));
        assert_eq!(PF90139_PF90140_MODEL_3434447882.model_id, Some(3_434_447_882));
        assert_eq!(PF90160_MODEL.model_id, Some(2_502_924_957));
        assert!(PORTRAIT_VENDOR_MODELS.iter().all(|asset| !asset.runtime_ready()));
    }

    #[test]
    fn teeth_model_has_no_guessed_identity() {
        assert_eq!(PF30026_MODEL.model_id, None);
        assert!(PF30026_MODEL
            .blockers
            .contains(&VendorModelBlocker::ModelIdentityUnbound));
    }
}
