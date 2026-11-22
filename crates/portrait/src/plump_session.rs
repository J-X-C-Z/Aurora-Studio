use std::fs;
use std::path::{Path, PathBuf};

use crate::model_sha256::{hex, sha256};

pub const MODEL_POOL: &str =
    r"C:\WorkSpace\自研图像编辑软件\逆向工程\像素蛋糕\素材池_全量";
pub const DECODER_ROOT: &str =
    r"C:\WorkSpace\Aurora Studio\04_reference\legacy-root\逆向工程\像素蛋糕\analysis\model-recovery-header";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    HighRes4,
    HighRes3,
    LowRes3,
}

impl Route {
    pub const fn model_id(self) -> u32 {
        match self {
            Route::HighRes4 => 597_383_622,
            Route::HighRes3 => 3_861_222_348,
            Route::LowRes3 => 1_139_333_499,
        }
    }

}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelSpec {
    pub route: Route,
    pub model_id: u32,
    pub payload_bytes: usize,
    pub payload_sha256: &'static str,
    pub net_sha256: &'static str,
    pub input_channels: usize,
    pub input_height: u32,
    pub input_width: u32,
    pub input_tensor: &'static str,
    pub output_tensor: &'static str,
}

pub const MODELS: [ModelSpec; 3] = [
    ModelSpec {
        route: Route::HighRes4,
        model_id: 597_383_622,
        payload_bytes: 2_498_909,
        payload_sha256: "4501dea6c75bf8fd3699ed309517f25285e076e2efa5c8d29866ae49e287fc72",
        net_sha256: "572b03d4c59753efbf0eb5e38910a91b797dec880ed15bb14487cb2eee7a70d5",
        input_channels: 4,
        input_height: 768,
        input_width: 768,
        input_tensor: "3071265413",
        output_tensor: "2294125226",
    },
    ModelSpec {
        route: Route::HighRes3,
        model_id: 3_861_222_348,
        payload_bytes: 1_766_645,
        payload_sha256: "b6097fe76c4aca32b8b74ff2f8a599a446d91a563c6d19d378eda05a7b651f20",
        net_sha256: "d25346b2752c8ebcac05c6583e599c7050d0042762a2100733f2cce3a7739461",
        input_channels: 3,
        input_height: 768,
        input_width: 768,
        input_tensor: "1929493859",
        output_tensor: "1132647165",
    },
    ModelSpec {
        route: Route::LowRes3,
        model_id: 1_139_333_499,
        payload_bytes: 2_701_281,
        payload_sha256: "fb194790e10f14b8bfcd75e530fa6471833d00aa2f7ac66d72943a5c192d6981",
        net_sha256: "8400f51cef63b15c6a55109e241676313c4b4408ca109e424433bb49fa6d4998",
        input_channels: 3,
        input_height: 256,
        input_width: 512,
        input_tensor: "3071265413",
        output_tensor: "2294125226",
    },
];

pub fn spec(route: Route) -> &'static ModelSpec {
    MODELS
        .iter()
        .find(|spec| spec.route == route)
        .expect("route table is exhaustive")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootInfo {
    pub root_offset: u32,
    pub vtable_offset: u32,
    pub vtable_bytes: u16,
    pub object_bytes: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionBinding {
    pub model_id: u32,
    pub route: Route,
    pub net_sha256: [u8; 32],
    pub input_shape: [u32; 4],
    pub input_tensor: &'static str,
    pub output_tensor: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SessionStamp {
    pub model_id: u32,
    pub net_sha256: [u8; 32],
    pub input_digest: [u8; 32],
    pub input_shape: [u32; 4],
    pub output_tensor: &'static str,
}

#[derive(Clone, Debug)]
pub struct LoadedSession {
    spec: ModelSpec,
    net: Vec<u8>,
    net_sha256: [u8; 32],
    payload_sha256: [u8; 32],
    root: RootInfo,
}

impl LoadedSession {
    pub fn open(route: Route) -> Result<Self, String> {
        Self::open_at(
            Path::new(MODEL_POOL),
            Path::new(DECODER_ROOT),
            route,
        )
    }

    pub fn open_at(pool: &Path, decoder: &Path, route: Route) -> Result<Self, String> {
        let spec = *spec(route);
        let payload_path = pool.join(format!("{}.bin", spec.model_id));
        let payload = fs::read(&payload_path)
            .map_err(|error| format!("read {}: {error}", payload_path.display()))?;
        if payload.len() != spec.payload_bytes {
            return Err(format!(
                "model {} size {} != {}",
                spec.model_id,
                payload.len(),
                spec.payload_bytes
            ));
        }
        let payload_sha256 = sha256(&payload);
        if hex(&payload_sha256) != spec.payload_sha256 {
            return Err(format!("model {} payload sha256 mismatch", spec.model_id));
        }

        let header_key = read_key(&decoder.join("native_header_xor.key"))?;
        let body_key = read_key(&decoder.join("native_body_xor.key"))?;
        let decoded = decode_payload(&payload, &header_key, &body_key)?;
        if decoded.get(77..).is_none() {
            return Err(format!("model {} decoded payload is short", spec.model_id));
        }
        let net = decoded[77..].to_vec();
        let net_sha256 = sha256(&net);
        if hex(&net_sha256) != spec.net_sha256 {
            return Err(format!("model {} net sha256 mismatch", spec.model_id));
        }
        let root = inspect_root(&net)?;
        Ok(Self {
            spec,
            net,
            net_sha256,
            payload_sha256,
            root,
        })
    }

    pub fn spec(&self) -> ModelSpec {
        self.spec
    }

    pub fn net(&self) -> &[u8] {
        &self.net
    }

    pub fn net_sha256(&self) -> [u8; 32] {
        self.net_sha256
    }

    pub fn record(&self) -> SessionRecord {
        SessionRecord {
            route: self.spec.route,
            model_id: self.spec.model_id,
            payload_bytes: self.spec.payload_bytes,
            payload_sha256: hex(&self.payload_sha256),
            net_bytes: self.net.len(),
            net_sha256: hex(&self.net_sha256),
            input_shape: [
                1,
                self.spec.input_channels as u32,
                self.spec.input_height,
                self.spec.input_width,
            ],
            input_tensor: self.spec.input_tensor,
            output_tensor: self.spec.output_tensor,
            root: self.root,
        }
    }

    pub fn binding(&self) -> SessionBinding {
        SessionBinding {
            model_id: self.spec.model_id,
            route: self.spec.route,
            net_sha256: self.net_sha256,
            input_shape: [
                1,
                self.spec.input_channels as u32,
                self.spec.input_height,
                self.spec.input_width,
            ],
            input_tensor: self.spec.input_tensor,
            output_tensor: self.spec.output_tensor,
        }
    }

    pub fn stamp(&self, input_digest: [u8; 32]) -> SessionStamp {
        SessionStamp {
            model_id: self.spec.model_id,
            net_sha256: self.net_sha256,
            input_digest,
            input_shape: [
                1,
                self.spec.input_channels as u32,
                self.spec.input_height,
                self.spec.input_width,
            ],
            output_tensor: self.spec.output_tensor,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SessionRecord {
    pub route: Route,
    pub model_id: u32,
    pub payload_bytes: usize,
    pub payload_sha256: String,
    pub net_bytes: usize,
    pub net_sha256: String,
    pub input_shape: [u32; 4],
    pub input_tensor: &'static str,
    pub output_tensor: &'static str,
    pub root: RootInfo,
}

fn read_key(path: &PathBuf) -> Result<[u8; 256], String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    bytes
        .try_into()
        .map_err(|bytes: Vec<u8>| format!("{} is {} bytes, expected 256", path.display(), bytes.len()))
}

fn decode_payload(
    payload: &[u8],
    header_key: &[u8; 256],
    body_key: &[u8; 256],
) -> Result<Vec<u8>, String> {
    if payload.len() < 256 {
        return Err("payload is shorter than the native header".into());
    }
    let mut output = payload.to_vec();
    for index in 0..256 {
        output[index] ^= header_key[index];
    }
    let body_end = payload.len() - payload.len() % 4;
    for index in 256..body_end {
        output[index] ^= body_key[index % 256];
    }
    let tail: &[u8] = match payload.len() % 4 {
        0 => &[],
        1 => &[0x78],
        2 => &[0x63, 0x78],
        3 => &[0x63, 0x78, 0x6a],
        _ => unreachable!(),
    };
    for (index, key) in tail.iter().enumerate() {
        let at = payload.len() - tail.len() + index;
        output[at] ^= key;
    }
    Ok(output)
}

fn inspect_root(net: &[u8]) -> Result<RootInfo, String> {
    if net.len() < 8 {
        return Err("missing FlatBuffer root".into());
    }
    let root_offset = u32::from_le_bytes(net[0..4].try_into().unwrap());
    let root = root_offset as usize;
    if root < 4 || root + 4 > net.len() {
        return Err("FlatBuffer root is outside the model".into());
    }
    let signed = i32::from_le_bytes(net[root..root + 4].try_into().unwrap());
    let vtable = root as i64 - signed as i64;
    if vtable < 0 || vtable as usize + 4 > net.len() {
        return Err("FlatBuffer vtable is outside the model".into());
    }
    let vtable = vtable as usize;
    let vtable_bytes = u16::from_le_bytes(net[vtable..vtable + 2].try_into().unwrap());
    let object_bytes = u16::from_le_bytes(net[vtable + 2..vtable + 4].try_into().unwrap());
    if vtable_bytes < 4
        || vtable_bytes % 2 != 0
        || object_bytes < 4
        || vtable + vtable_bytes as usize > net.len()
        || root + object_bytes as usize > net.len()
    {
        return Err("invalid FlatBuffer vtable".into());
    }
    Ok(RootInfo {
        root_offset,
        vtable_offset: vtable as u32,
        vtable_bytes,
        object_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_real_assets_match_payload_and_net_hashes() {
        for route in [Route::HighRes4, Route::HighRes3, Route::LowRes3] {
            let session = LoadedSession::open(route).unwrap();
            let spec = session.spec();
            assert_eq!(session.net_sha256(), {
                let mut out = [0u8; 32];
                for (i, chunk) in spec.net_sha256.as_bytes().chunks_exact(2).enumerate() {
                    out[i] = u8::from_str_radix(std::str::from_utf8(chunk).unwrap(), 16).unwrap();
                }
                out
            });
            assert_eq!(session.record().root.root_offset, 44);
            assert_eq!(session.record().root.vtable_bytes, 40);
            assert_eq!(session.record().root.object_bytes, 40);
        }
    }

    #[test]
    fn model_shapes_keep_the_four_plane_route_explicit() {
        assert_eq!(spec(Route::HighRes4).input_channels, 4);
        assert_eq!(spec(Route::HighRes3).input_channels, 3);
        assert_eq!(spec(Route::LowRes3).input_width, 512);
    }
}

