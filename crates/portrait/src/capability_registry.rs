//! Portrait capability registry (PP-1).
//!
//! The canonical table is
//! `03_algorithms/reports/PORTRAIT-CAPABILITY-REGISTRY-20261124.tsv`; a
//! byte-identical copy ships at
//! `assets/portrait/capability-registry-20261124.tsv` so the crate stays
//! self-contained and offline-buildable.
//!
//! The seven evidence levels are independent. `未验证` is never a pass, and a
//! lower level never implies a higher one. Registration is not completion: a
//! callable kernel does not prove same-job input wiring, preview integration,
//! vendor parity, or real-photo acceptance.

pub const REGISTRY_SCHEMA: &str = "aurora/portrait-capability-registry/1";
pub const UNVERIFIED: &str = "未验证";
const COLUMN_COUNT: usize = 18;

pub const EVIDENCE_LEVELS: [&str; 7] = [
    "kernel_exists",
    "cpu_reference",
    "gpu_kernel",
    "same_job_inputs",
    "main_preview",
    "vendor_parity",
    "real_photo_accept",
];

pub const REGISTRY_TSV: &str =
    include_str!("../assets/portrait/capability-registry-20261124.tsv");

/// One registry row. Cell text is borrowed from the embedded table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapabilityRow<'a> {
    pub kind: &'a str,
    pub pf_text: &'a str,
    pub key: &'a str,
    pub label: &'a str,
    pub inputs: &'a str,
    pub cpu_backend: &'a str,
    pub gpu_backend: &'a str,
    pub model_assets: &'a str,
    pub roi_support: &'a str,
    pub quality_fixture: &'a str,
    pub evidence: [&'a str; 7],
    pub notes: &'a str,
}

impl CapabilityRow<'_> {
    pub fn pf(&self) -> Option<u32> {
        self.pf_text.parse().ok()
    }

    pub fn is_test_gap(&self) -> bool {
        self.kind == "test_gap"
    }

    /// `level` is an index into [`EVIDENCE_LEVELS`]. A non-`未验证` cell is
    /// evidence; it is still the caller's job to read the cited path.
    pub fn has_evidence(&self, level: usize) -> bool {
        self.evidence
            .get(level)
            .is_some_and(|cell| *cell != UNVERIFIED)
    }
}

fn parse_line(line: &'static str) -> Option<CapabilityRow<'static>> {
    if line.is_empty() {
        return None;
    }
    let cells: Vec<&'static str> = line.split('\t').collect();
    if cells.len() != COLUMN_COUNT || cells[0] == "kind" {
        return None;
    }
    Some(CapabilityRow {
        kind: cells[0],
        pf_text: cells[1],
        key: cells[2],
        label: cells[3],
        inputs: cells[4],
        cpu_backend: cells[5],
        gpu_backend: cells[6],
        model_assets: cells[7],
        roi_support: cells[8],
        quality_fixture: cells[9],
        evidence: [
            cells[10], cells[11], cells[12], cells[13], cells[14], cells[15], cells[16],
        ],
        notes: cells[17],
    })
}

pub fn rows() -> impl Iterator<Item = CapabilityRow<'static>> {
    REGISTRY_TSV.lines().filter_map(parse_line)
}

pub fn effects() -> impl Iterator<Item = CapabilityRow<'static>> {
    rows().filter(|row| !row.is_test_gap())
}

pub fn test_gaps() -> impl Iterator<Item = CapabilityRow<'static>> {
    rows().filter(|row| row.is_test_gap())
}

pub fn for_pf(pf: u32) -> Option<CapabilityRow<'static>> {
    effects().find(|row| row.pf() == Some(pf))
}

fn escape(text: &str) -> String {
    text.chars()
        .flat_map(|c| match c {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect(),
            '\n' => "\\n".chars().collect(),
            '\r' => "\\r".chars().collect(),
            '\t' => "\\t".chars().collect(),
            c => vec![c],
        })
        .collect()
}

/// Machine-readable view for the capability contract. `pf` is a number for
/// effect rows and `null` for test-gap rows; evidence cells keep the cited
/// path so a host can display the level without a second lookup.
pub fn to_json() -> String {
    let mut out = String::from("{\"schema\":\"");
    out.push_str(REGISTRY_SCHEMA);
    out.push_str("\",\"columns\":18,\"rows\":[");
    for (index, row) in rows().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let pf = match row.pf() {
            Some(pf) => pf.to_string(),
            None => "null".to_owned(),
        };
        out.push_str(&format!(
            "{{\"kind\":\"{}\",\"pf\":{},\"key\":\"{}\",\"label\":\"{}\",\"inputs\":\"{}\",\
             \"cpu_backend\":\"{}\",\"gpu_backend\":\"{}\",\"model_assets\":\"{}\",\
             \"roi_support\":\"{}\",\"quality_fixture\":\"{}\",\"evidence\":{{",
            escape(row.kind),
            pf,
            escape(row.key),
            escape(row.label),
            escape(row.inputs),
            escape(row.cpu_backend),
            escape(row.gpu_backend),
            escape(row.model_assets),
            escape(row.roi_support),
            escape(row.quality_fixture)
        ));
        for (level, name) in EVIDENCE_LEVELS.iter().enumerate() {
            if level > 0 {
                out.push(',');
            }
            out.push_str(&format!("\"{name}\":\"{}\"", escape(row.evidence[level])));
        }
        out.push_str(&format!(
            "}},\"notes\":\"{}\"}}",
            escape(row.notes)
        ));
    }
    out.push_str("]}");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog;
    use std::collections::BTreeSet;

    #[test]
    fn header_has_the_contract_columns() {
        let header = REGISTRY_TSV.lines().next().expect("header");
        assert_eq!(
            header,
            "kind\tpf\tkey\tlabel\tinput_requirements\tcpu_backend\tgpu_backend\t\
             model_assets_hash\troi_support\tquality_fixture\tev1_kernel_exists\t\
             ev2_cpu_reference\tev3_gpu_kernel\tev4_same_job_inputs\tev5_main_preview\t\
             ev6_vendor_parity\tev7_real_photo_accept\tevidence_notes"
        );
        assert_eq!(header.split('\t').count(), COLUMN_COUNT);
    }

    #[test]
    fn every_row_keeps_all_seven_evidence_cells() {
        let mut count = 0;
        for line in REGISTRY_TSV.lines() {
            assert_eq!(
                line.split('\t').count(),
                COLUMN_COUNT,
                "malformed registry line: {line}"
            );
            count += 1;
        }
        assert!(count > 1);
        for row in rows() {
            assert_eq!(row.evidence.len(), EVIDENCE_LEVELS.len());
            assert!(!row.label.is_empty() || row.is_test_gap());
        }
    }

    #[test]
    fn every_catalog_effect_has_exactly_one_row() {
        let mut seen = BTreeSet::new();
        for row in effects() {
            let pf = row.pf().expect("effect row without pf");
            assert!(seen.insert(pf), "duplicate PF{pf}");
        }
        for meta in catalog() {
            assert!(
                seen.contains(&meta.pf),
                "catalog PF{} is missing from the capability registry",
                meta.pf
            );
        }
        assert_eq!(seen.len(), catalog().len());
    }

    #[test]
    fn unverified_stays_explicit_and_is_not_a_pass() {
        for row in effects() {
            for level in 0..EVIDENCE_LEVELS.len() {
                assert!(!row.evidence[level].is_empty());
                assert_eq!(
                    row.has_evidence(level),
                    row.evidence[level] != UNVERIFIED,
                    "PF{} level {} lost its explicit unverified marker",
                    row.pf_text,
                    EVIDENCE_LEVELS[level]
                );
            }
            // PP-1 records no vendor parity or real-photo acceptance yet.
            assert!(!row.has_evidence(5));
            assert!(!row.has_evidence(6));
        }
    }

    #[test]
    fn dispatch_test_gaps_are_registered() {
        let gaps: Vec<_> = test_gaps().collect();
        assert_eq!(gaps.len(), 6);
        let keys: Vec<_> = gaps.iter().map(|row| row.key).collect();
        assert_eq!(
            keys,
            vec!["GAP-01", "GAP-02", "GAP-03", "GAP-04", "GAP-05", "GAP-06"]
        );
    }

    #[test]
    fn json_is_parseable_and_names_every_row() {
        let json = to_json();
        let value: serde_json::Value = serde_json::from_str(&json).expect("registry JSON");
        let rows = value["rows"].as_array().expect("rows array");
        assert_eq!(rows.len(), super::rows().count());
        assert!(json.contains("\"pf\":90021"));
        assert!(json.contains("\"real_photo_accept\":\"未验证\""));
        assert!(json.contains("\"kind\":\"test_gap\""));
    }
}
