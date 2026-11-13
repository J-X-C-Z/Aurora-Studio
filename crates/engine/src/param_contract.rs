//! Reads the shared frontend slider contract (`01_frontend/contracts/params.json`).
//!
//! The contract is the frontend's view of the 72 C1 sliders. It is input data,
//! not a second catalog: [`compare`] reports where it disagrees with
//! [`crate::param_meta::catalog`], and [`overrides`] applies it without touching
//! any other [`ParamMeta`] field.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::param_meta::ParamMeta;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct ParamContract {
    #[serde(rename = "key")]
    pub id: String,
    #[serde(rename = "label")]
    pub name: String,
    pub min: f64,
    pub max: f64,
    pub default: f64,
    pub step: f64,
    pub decimals: u8,
    pub unit: String,
}

pub fn load(path: &Path) -> Result<Vec<ParamContract>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read param contract {}", path.display()))?;
    serde_json::from_str(&text)
        .with_context(|| format!("invalid param contract JSON {}", path.display()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffField {
    Min,
    Max,
    Default,
    MissingInCatalog,
    MissingInContract,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ContractDiff {
    pub id: String,
    pub field: DiffField,
    pub catalog: Option<f32>,
    pub contract: Option<f32>,
}

pub fn compare(catalog: &[ParamMeta], contracts: &[ParamContract]) -> Vec<ContractDiff> {
    let by_id: BTreeMap<&str, &ParamMeta> = catalog.iter().map(|m| (m.key, m)).collect();
    let contracts_by_id: BTreeMap<&str, &ParamContract> =
        contracts.iter().map(|c| (c.id.as_str(), c)).collect();
    let mut diffs = Vec::new();

    for (id, contract) in &contracts_by_id {
        let Some(meta) = by_id.get(id) else {
            diffs.push(ContractDiff {
                id: (*id).to_owned(),
                field: DiffField::MissingInCatalog,
                catalog: None,
                contract: Some(contract.default as f32),
            });
            continue;
        };
        for (field, catalog_value, contract_value) in [
            (DiffField::Min, meta.min, contract.min as f32),
            (DiffField::Max, meta.max, contract.max as f32),
            (DiffField::Default, meta.default, contract.default as f32),
        ] {
            if catalog_value != contract_value {
                diffs.push(ContractDiff {
                    id: (*id).to_owned(),
                    field,
                    catalog: Some(catalog_value),
                    contract: Some(contract_value),
                });
            }
        }
    }

    for (id, meta) in &by_id {
        if !contracts_by_id.contains_key(id) {
            diffs.push(ContractDiff {
                id: (*id).to_owned(),
                field: DiffField::MissingInContract,
                catalog: Some(meta.default),
                contract: None,
            });
        }
    }

    diffs.sort_by(|a, b| a.id.cmp(&b.id).then(a.field.cmp(&b.field)));
    diffs
}

pub fn overrides(catalog: &[ParamMeta], contracts: &[ParamContract]) -> Vec<ParamMeta> {
    let by_id: BTreeMap<&str, &ParamContract> =
        contracts.iter().map(|c| (c.id.as_str(), c)).collect();
    catalog
        .iter()
        .map(|meta| {
            let Some(contract) = by_id.get(meta.key) else {
                return *meta;
            };
            if !(contract.min.is_finite()
                && contract.max.is_finite()
                && contract.default.is_finite())
            {
                return *meta;
            }
            ParamMeta {
                min: contract.min as f32,
                max: contract.max as f32,
                default: contract.default as f32,
                ..*meta
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::param_meta::{MetaSource, by_key};

    fn contract_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/frontend-contract/params.json")
    }

    fn contract(id: &str, min: f64, max: f64, default: f64) -> ParamContract {
        ParamContract {
            id: id.to_owned(),
            name: id.to_owned(),
            min,
            max,
            default,
            step: 0.1,
            decimals: 2,
            unit: String::new(),
        }
    }

    fn catalog_meta(id: &'static str, min: f32, max: f32, default: f32) -> ParamMeta {
        ParamMeta {
            key: id,
            label: id,
            label_zh: id,
            min,
            max,
            default,
            unit: "",
            step: 0.1,
            decimals: 2,
            group: "Synthetic",
            source: MetaSource::Hardcoded,
        }
    }

    #[test]
    fn loads_all_72_contracts() {
        let contracts = load(&contract_path()).unwrap();
        assert_eq!(contracts.len(), 72);
        let exposure = contracts.iter().find(|c| c.id == "Exposure").unwrap();
        assert_eq!(exposure.name, "Exposure");
        assert_eq!(
            (exposure.min, exposure.max, exposure.default),
            (-2.5, 2.5, 0.0)
        );
        assert_eq!((exposure.step, exposure.decimals), (0.1, 2));
        assert_eq!(exposure.unit, "EV");
    }

    #[test]
    fn compare_reports_synthetic_differences_deterministically() {
        let catalog = [
            catalog_meta("Alpha", 0.0, 1.0, 0.25),
            catalog_meta("Beta", -1.0, 1.0, 0.0),
            catalog_meta("OnlyCatalog", 0.0, 1.0, 0.5),
        ];
        let contracts = [
            contract("OnlyContract", 0.0, 1.0, 0.0),
            contract("Beta", -2.0, 2.0, 0.5),
            contract("Alpha", 0.0, 1.0, 0.25),
        ];
        let diffs = compare(&catalog, &contracts);
        assert_eq!(
            diffs
                .iter()
                .map(|d| (d.id.as_str(), d.field, d.catalog, d.contract))
                .collect::<Vec<_>>(),
            [
                ("Beta", DiffField::Min, Some(-1.0), Some(-2.0)),
                ("Beta", DiffField::Max, Some(1.0), Some(2.0)),
                ("Beta", DiffField::Default, Some(0.0), Some(0.5)),
                ("OnlyCatalog", DiffField::MissingInContract, Some(0.5), None),
                ("OnlyContract", DiffField::MissingInCatalog, None, Some(0.0)),
            ]
        );
        assert_eq!(compare(&catalog, &contracts), diffs);
    }

    #[test]
    fn diffs_are_sorted_and_report_missing_entries() {
        let meta = *by_key("Exposure").unwrap();
        let diffs = compare(
            &[meta],
            &[
                contract("Zeta", 0.0, 1.0, 0.0),
                contract("Exposure", -2.5, 3.0, 0.1),
            ],
        );
        assert_eq!(
            diffs
                .iter()
                .map(|d| (d.id.as_str(), d.field))
                .collect::<Vec<_>>(),
            [
                ("Exposure", DiffField::Max),
                ("Exposure", DiffField::Default),
                ("Zeta", DiffField::MissingInCatalog),
            ]
        );
        let diffs = compare(&[meta], &[]);
        assert_eq!(diffs[0].field, DiffField::MissingInContract);
        assert_eq!(diffs[0].catalog, Some(meta.default));
    }

    #[test]
    fn bad_json_names_the_file_and_the_cause() {
        let path = std::env::temp_dir().join(format!(
            "lumen-param-contract-bad-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, "{ not json").unwrap();
        let message = format!("{:#}", load(&path).unwrap_err());
        let _ = std::fs::remove_file(&path);
        assert!(message.contains("invalid param contract JSON"), "{message}");
        assert!(message.contains(&path.display().to_string()), "{message}");
        assert!(message.contains("expected"), "{message}");
    }

    #[test]
    fn overrides_keep_unrelated_fields_and_skip_unknown_ids() {
        let exposure = *by_key("Exposure").unwrap();
        let contrast = *by_key("Contrast").unwrap();
        let out = overrides(
            &[exposure, contrast],
            &[
                contract("Exposure", -9.0, 9.0, 1.5),
                contract("Nope", 0.0, 1.0, 0.0),
            ],
        );
        assert_eq!(
            out[0],
            ParamMeta {
                min: -9.0,
                max: 9.0,
                default: 1.5,
                ..exposure
            }
        );
        assert_eq!(out[0].key, exposure.key);
        assert_eq!(out[0].label, exposure.label);
        assert_eq!(out[0].label_zh, exposure.label_zh);
        assert_eq!((out[0].unit, out[0].step, out[0].decimals), ("EV", 0.1, 2));
        assert_eq!(
            (out[0].group, out[0].source),
            ("Exposure", MetaSource::RawImage)
        );
        assert_eq!(out[1], contrast);
    }

    #[test]
    fn overrides_skip_non_finite_contract_values() {
        let exposure = *by_key("Exposure").unwrap();
        let out = overrides(&[exposure], &[contract("Exposure", f64::NAN, 9.0, 1.0)]);
        assert_eq!(out[0], exposure);
    }
}
