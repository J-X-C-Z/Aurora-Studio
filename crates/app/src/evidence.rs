//! 人像能力的七级证据状态。
//!
//! 数据来源是 `03_algorithms/reports/PORTRAIT-CAPABILITY-REGISTRY-20261124.tsv`
//! 的逐字节副本 `docs/frontend-contract/portrait-capability-registry-20261124.tsv`
//! （SHA256 `f3986958a2415cd08bdcd3a6a98d5b2da6a23df90bfd7c4d68d0f41b8ccc9bc1`，
//! 76393 字节）。登记不等于可用：七级证据相互独立，低级证据不能推出高级证据，
//! `未验证` 永远不是通过。
//!
//! 前端只在 ev5（应用主预览接入）有证据时才把效果呈现为可用；ev6（原厂同图对拍）
//! 与 ev7（真实照片验收）单独展示，不能被 ev5 代替。

use std::collections::BTreeMap;
use std::sync::LazyLock;

use crate::contract;
use crate::i18n::t;

pub const UNVERIFIED: &str = "未验证";

/// 嵌入的能力表；重新同步时必须与本目录下的 TSV 完全一致。
pub const REGISTRY: &str = include_str!("../../../docs/frontend-contract/portrait-capability-registry-20261124.tsv");
pub const REGISTRY_SOURCE: &str = "03_algorithms/reports/PORTRAIT-CAPABILITY-REGISTRY-20261124.tsv";
pub const REGISTRY_SHA256: &str = "f3986958a2415cd08bdcd3a6a98d5b2da6a23df90bfd7c4d68d0f41b8ccc9bc1";
/// 字节数；随文件同步变化的快速哨兵，防止副本与 SHA 记录脱节。
const REGISTRY_BYTES: usize = 76393;

/// 七级证据，顺序与能力表一致。索引 4 是 ev5（应用主预览接入）。
pub const LEVELS: [&str; 7] = [
    "kernel_exists",
    "cpu_reference",
    "gpu_kernel",
    "same_job_inputs",
    "main_preview",
    "vendor_parity",
    "real_photo_accept",
];

/// 主预览这一级的下标。
pub const MAIN_PREVIEW: usize = 4;
/// 原厂对拍这一级的下标。
pub const VENDOR_PARITY: usize = 5;
/// 真实照片验收这一级的下标。
pub const REAL_PHOTO: usize = 6;

fn level_name(level: usize) -> (&'static str, &'static str) {
    match level {
        0 => ("Kernel exists", "内核存在"),
        1 => ("CPU reference", "CPU 参考实现"),
        2 => ("GPU kernel", "GPU 内核实跑"),
        3 => ("Same-job inputs", "同 job 输入"),
        4 => ("Main preview", "应用主预览接入"),
        5 => ("Vendor parity", "原厂同图对拍"),
        _ => ("Real-photo accept", "真实照片验收"),
    }
}

/// 能力表里的一行 effect。
#[derive(Clone, Copy, Debug)]
pub struct Effect {
    pub pf: u32,
    pub key: &'static str,
    pub label: &'static str,
    pub stage: &'static str,
    pub inputs: &'static str,
    pub evidence: [&'static str; 7],
    pub notes: &'static str,
}

impl Effect {
    /// 该级是否有证据；`未验证` 不算证据。
    pub fn has(&self, level: usize) -> bool {
        self.evidence.get(level).is_some_and(|cell| !cell.trim().is_empty() && *cell != UNVERIFIED)
    }

    /// 已经拿到证据的级数（0..=7）。
    pub fn reached(&self) -> usize {
        (0..LEVELS.len()).filter(|level| self.has(*level)).count()
    }

    /// 是否达到 ev5（应用主预览接入）。这是前端唯一可以呈现“可用”的门槛。
    pub fn preview_connected(&self) -> bool {
        self.has(MAIN_PREVIEW)
    }

    /// 能力表里 key/label 仍是 `未验证` 占位时的代码侧确定键。
    /// 只登记已经核实的项；其余仍按能力表原值处理，不猜测。
    pub fn catalog_key(&self) -> &'static str {
        match self.pf {
            30096 => "sediao",
            _ => self.key,
        }
    }

    /// 能力表里的标签常常是 `未验证` 占位；这时回退到 UI 键表或人类可读键名。
    pub fn display_label(&self) -> String {
        let key = self.catalog_key();
        let (en, zh) = contract::label_for_key(key);
        if self.label.trim().is_empty() || self.label == UNVERIFIED {
            if let (Some(en), Some(zh)) = (en, zh) {
                return t(en, zh).to_string();
            }
            if !key.trim().is_empty() && key != UNVERIFIED {
                return contract::humanized_key(key);
            }
            return format!("PF{}", self.pf);
        }
        if let (Some(en), Some(zh)) = (en, zh) {
            return t(en, zh).to_string();
        }
        self.label.to_string()
    }

    /// 一行简短的状态文字，用于参数行下方。
    pub fn state_line(&self) -> String {
        let reached = self.reached();
        let parity = if self.has(VENDOR_PARITY) { t("evidenced", "有证据") } else { t("unverified", "未验证") };
        let photo = if self.has(REAL_PHOTO) { t("evidenced", "有证据") } else { t("unverified", "未验证") };
        if self.preview_connected() {
            format!(
                "{}/7 · {} · {} {parity} / {} {photo}",
                reached,
                t("main preview wired", "已接入主预览"),
                t("vendor parity", "原厂对拍"),
                t("real photo", "真实照片")
            )
        } else {
            format!("{}/7 · {}", reached, t("not wired into the main preview", "未接入主预览"))
        }
    }

    /// 七级证据的完整悬停说明。
    pub fn tooltip(&self) -> String {
        let key = self.catalog_key();
        let mut out = format!("PF{}", self.pf);
        if !key.trim().is_empty() && key != UNVERIFIED {
            out.push_str(&format!(" · {key}"));
        }
        out.push('\n');
        for level in 0..LEVELS.len() {
            let (en, zh) = level_name(level);
            let mark = if self.has(level) { "●" } else { "○" };
            out.push_str(&format!("{mark} ev{} {} ", level + 1, t(en, zh)));
            if self.has(level) {
                out.push_str(self.evidence[level]);
            } else {
                out.push_str(t("unverified", "未验证"));
            }
            out.push('\n');
        }
        if !self.inputs.trim().is_empty() && self.inputs != UNVERIFIED {
            out.push_str(&format!("{}: {}\n", t("Inputs", "输入"), self.inputs));
        }
        if !self.stage.is_empty() && self.stage != UNVERIFIED {
            out.push_str(&format!("{}: {}\n", t("Stage", "阶段"), self.stage));
        }
        if !self.notes.trim().is_empty() {
            out.push_str(self.notes);
        }
        out.trim_end().to_string()
    }
}

/// 参数/效果对前端的可用性，只由证据决定。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Availability {
    /// 达到 ev5：主预览可呈现为可用（原厂对拍、真实照片仍可能未验证）。
    Preview,
    /// 能力表有登记，但没有主预览证据：只能写“未接入预览”。
    Registered,
    /// 能力表没有这一项：不推断任何能力。
    Unregistered,
}

pub fn availability(pf: i64) -> Availability {
    match for_pf(pf) {
        Some(effect) if effect.preview_connected() => Availability::Preview,
        Some(_) => Availability::Registered,
        None => Availability::Unregistered,
    }
}

fn parse(line: &'static str) -> Option<Effect> {
    if line.is_empty() {
        return None;
    }
    let cells: Vec<&'static str> = line.split('\t').collect();
    if cells.len() != 18 || cells[0] != "effect" {
        return None;
    }
    let pf: u32 = cells[1].parse().ok()?;
    Some(Effect {
        pf,
        key: cells[2],
        label: cells[3],
        stage: cells[5],
        inputs: cells[4],
        evidence: [cells[10], cells[11], cells[12], cells[13], cells[14], cells[15], cells[16]],
        notes: cells[17],
    })
}

static BY_PF: LazyLock<BTreeMap<u32, Effect>> = LazyLock::new(|| {
    // 副本与记录的哈希/字节数必须同步，否则整份证据不可信。
    debug_assert_eq!(REGISTRY.len(), REGISTRY_BYTES, "registry copy is out of sync with REGISTRY_SHA256");
    REGISTRY.lines().filter_map(parse).map(|effect| (effect.pf, effect)).collect()
});

pub fn all() -> impl Iterator<Item = &'static Effect> {
    BY_PF.values()
}

pub fn for_pf(pf: i64) -> Option<&'static Effect> {
    u32::try_from(pf).ok().and_then(|pf| BY_PF.get(&pf))
}

/// 某个 ID 段内已登记、但前端没有参数控件的效果。
pub fn without_control(lo: i64, hi: i64) -> Vec<&'static Effect> {
    let controls: std::collections::BTreeSet<i64> = contract::PORTRAIT.iter().map(|param| param.id).collect();
    all()
        .filter(|effect| {
            let pf = i64::from(effect.pf);
            pf >= lo && pf <= hi && !controls.contains(&pf)
        })
        .collect()
}

/// 某个 ID 段内达到 ev5 的效果数，以及该段登记的效果总数。
pub fn preview_count(lo: i64, hi: i64) -> (usize, usize) {
    all()
        .filter(|effect| {
            let pf = i64::from(effect.pf);
            pf >= lo && pf <= hi
        })
        .fold((0, 0), |(wired, total), effect| (wired + usize::from(effect.preview_connected()), total + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_copy_matches_the_registry_record() {
        assert_eq!(REGISTRY.len(), REGISTRY_BYTES, "registry copy changed; update REGISTRY_SHA256 and this size");
        assert!(REGISTRY.starts_with("kind\tpf\tkey\tlabel\t"));
        for line in REGISTRY.lines() {
            assert_eq!(line.split('\t').count(), 18, "malformed registry line: {line}");
        }
    }

    #[test]
    fn effects_are_unique_and_counted() {
        assert_eq!(all().count(), 147);
        assert_eq!(all().filter(|e| e.preview_connected()).count(), 8);
        assert_eq!(all().filter(|e| e.has(VENDOR_PARITY)).count(), 0);
        assert_eq!(all().filter(|e| e.has(REAL_PHOTO)).count(), 0);
    }

    #[test]
    fn unverified_is_never_evidence() {
        for effect in all() {
            for level in 0..LEVELS.len() {
                assert!(!effect.evidence[level].is_empty());
                assert_eq!(effect.has(level), effect.evidence[level] != UNVERIFIED);
            }
        }
    }

    #[test]
    fn the_three_rv24_effects_are_reported_as_found() {
        let skin_temperature = for_pf(30096).expect("PF30096");
        assert!(skin_temperature.preview_connected());
        assert_eq!(skin_temperature.display_label(), t("Skin temperature", "肤色温度"));

        let lips = for_pf(90021).expect("PF90021");
        assert_eq!(lips.key, "lipMakeupEnhance");
        assert_eq!(lips.label, "Lip makeup enhance");
        assert!(!lips.preview_connected());

        let foot_veins = for_pf(44297).expect("PF44297");
        assert_eq!(foot_veins.label, "Foot blue veins");
        assert_eq!(foot_veins.evidence[2], UNVERIFIED);
        assert!(!foot_veins.preview_connected());
        assert_eq!(availability(44297), Availability::Registered);
        assert_eq!(availability(1), Availability::Unregistered);
    }

    #[test]
    fn registry_only_effects_are_listed_when_a_band_has_no_control() {
        let missing: Vec<u32> = without_control(90000, 90999).iter().map(|e| e.pf).collect();
        assert!(missing.contains(&90021), "PF90021 has no slider and must stay visible");
        assert!(!missing.contains(&30096), "PF30096 has a sediao slider");
    }
}
