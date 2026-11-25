//! Aurora Native Inference Engine — Phase 1 Pure-Rust Foundation.
//!
//! Zero-external-dependency ONNX graph loader, strongly-typed IR, static shape
//! inference, `Conv + BatchNormalization` weight folding, and liveness-based
//! VRAM Memory Arena Planner.
//!
//! Designed to replace `onnxruntime*.dll` (~213 MB) together with Phase 2
//! custom CUDA operator kernels.

pub mod arena;
pub mod bn_fold;
pub mod graph;
pub mod onnx_proto;

use std::collections::HashMap;
use std::fmt;

use arena::{plan_memory_arena, ArenaPlan, CUDA_DEFAULT_ALIGNMENT};
use bn_fold::{fold_batch_norms, BnFoldStats};
use graph::{GraphError, IrGraph};
use onnx_proto::{decode_model, ProtoError};

#[derive(Debug)]
pub enum NativeInferError {
    Proto(ProtoError),
    Graph(GraphError),
}

impl fmt::Display for NativeInferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proto(e) => write!(f, "ONNX proto decode error: {e}"),
            Self::Graph(e) => write!(f, "IR graph error: {e}"),
        }
    }
}

impl std::error::Error for NativeInferError {}

impl From<ProtoError> for NativeInferError {
    fn from(e: ProtoError) -> Self {
        Self::Proto(e)
    }
}

impl From<GraphError> for NativeInferError {
    fn from(e: GraphError) -> Self {
        Self::Graph(e)
    }
}

/// Complete compiled & memory-planned model artifact ready for Phase 2 CUDA execution.
#[derive(Debug)]
pub struct CompiledPlan {
    pub graph: IrGraph,
    pub bn_stats: BnFoldStats,
    pub resolved_shapes: usize,
    pub arena: ArenaPlan,
}

/// Decode an `.onnx` buffer, build the strongly-typed IR, fold BatchNorms,
/// validate topology, infer static shapes, and plan the VRAM arena.
pub fn load_and_plan(
    onnx_bytes: &[u8],
    input_overrides: &HashMap<String, Vec<usize>>,
) -> Result<CompiledPlan, NativeInferError> {
    let model = decode_model(onnx_bytes)?;
    let mut graph = IrGraph::from_model_proto(&model)?;
    graph.validate_topology()?;
    let bn_stats = fold_batch_norms(&mut graph, true);
    graph.validate_topology()?;
    let resolved_shapes = graph.infer_shapes(input_overrides)?;
    let arena = plan_memory_arena(&graph, CUDA_DEFAULT_ALIGNMENT);
    Ok(CompiledPlan {
        graph,
        bn_stats,
        resolved_shapes,
        arena,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_infer::onnx_proto::{f16_to_f32, f32_to_f16, DataType};

    // --- Minimal protobuf wire encoder for synthetic in-memory ONNX tests ---

    fn enc_varint(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                b |= 0x80;
            }
            out.push(b);
            if v == 0 {
                break;
            }
        }
    }

    fn enc_tag(out: &mut Vec<u8>, field: u32, wire: u8) {
        enc_varint(out, ((field as u64) << 3) | (wire as u64));
    }

    fn enc_bytes(out: &mut Vec<u8>, field: u32, data: &[u8]) {
        enc_tag(out, field, 2);
        enc_varint(out, data.len() as u64);
        out.extend_from_slice(data);
    }

    fn enc_str(out: &mut Vec<u8>, field: u32, s: &str) {
        enc_bytes(out, field, s.as_bytes());
    }

    fn enc_value_info(name: &str, dtype: DataType, dims: &[i64]) -> Vec<u8> {
        let mut shape = Vec::new();
        for &d in dims {
            let mut dim_msg = Vec::new();
            enc_tag(&mut dim_msg, 1, 0);
            enc_varint(&mut dim_msg, d as u64);
            enc_bytes(&mut shape, 1, &dim_msg);
        }
        let mut tensor_type = Vec::new();
        enc_tag(&mut tensor_type, 1, 0);
        enc_varint(&mut tensor_type, dtype as u64);
        enc_bytes(&mut tensor_type, 2, &shape);

        let mut type_proto = Vec::new();
        enc_bytes(&mut type_proto, 1, &tensor_type);

        let mut vi = Vec::new();
        enc_str(&mut vi, 1, name);
        enc_bytes(&mut vi, 2, &type_proto);
        vi
    }

    fn enc_tensor_f32_raw(name: &str, dims: &[i64], vals: &[f32]) -> Vec<u8> {
        let mut t = Vec::new();
        for &d in dims {
            enc_tag(&mut t, 1, 0);
            enc_varint(&mut t, d as u64);
        }
        enc_tag(&mut t, 2, 0);
        enc_varint(&mut t, DataType::Float as u64);
        enc_str(&mut t, 8, name);
        let mut raw = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        enc_bytes(&mut t, 9, &raw);
        t
    }

    fn enc_tensor_f32_float_data(name: &str, dims: &[i64], vals: &[f32]) -> Vec<u8> {
        let mut t = Vec::new();
        for &d in dims {
            enc_tag(&mut t, 1, 0);
            enc_varint(&mut t, d as u64);
        }
        enc_tag(&mut t, 2, 0);
        enc_varint(&mut t, DataType::Float as u64);
        for &v in vals {
            enc_tag(&mut t, 4, 5);
            t.extend_from_slice(&v.to_le_bytes());
        }
        enc_str(&mut t, 8, name);
        t
    }

    fn enc_tensor_f16_raw(name: &str, dims: &[i64], vals: &[f32]) -> Vec<u8> {
        let mut t = Vec::new();
        for &d in dims {
            enc_tag(&mut t, 1, 0);
            enc_varint(&mut t, d as u64);
        }
        enc_tag(&mut t, 2, 0);
        enc_varint(&mut t, DataType::Float16 as u64);
        enc_str(&mut t, 8, name);
        let mut raw = Vec::with_capacity(vals.len() * 2);
        for &v in vals {
            raw.extend_from_slice(&f32_to_f16(v).to_le_bytes());
        }
        enc_bytes(&mut t, 9, &raw);
        t
    }

    fn enc_attr_ints(name: &str, vals: &[i64]) -> Vec<u8> {
        let mut a = Vec::new();
        enc_str(&mut a, 1, name);
        for &v in vals {
            enc_tag(&mut a, 8, 0);
            enc_varint(&mut a, v as u64);
        }
        enc_tag(&mut a, 20, 0);
        enc_varint(&mut a, 7); // INTS
        a
    }

    fn enc_attr_float(name: &str, val: f32) -> Vec<u8> {
        let mut a = Vec::new();
        enc_str(&mut a, 1, name);
        enc_tag(&mut a, 2, 5);
        a.extend_from_slice(&val.to_le_bytes());
        enc_tag(&mut a, 20, 0);
        enc_varint(&mut a, 1); // FLOAT
        a
    }

    fn enc_node(
        op_type: &str,
        name: &str,
        inputs: &[&str],
        outputs: &[&str],
        attrs: &[Vec<u8>],
    ) -> Vec<u8> {
        let mut n = Vec::new();
        for i in inputs {
            enc_str(&mut n, 1, i);
        }
        for o in outputs {
            enc_str(&mut n, 2, o);
        }
        enc_str(&mut n, 3, name);
        enc_str(&mut n, 4, op_type);
        for a in attrs {
            enc_bytes(&mut n, 5, a);
        }
        n
    }

    #[test]
    fn f16_roundtrip_accuracy() {
        for &v in &[0.0f32, -0.0, 1.0, -1.0, 0.5, 3.140625, 1e-4, -65504.0] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert!((v - back).abs() <= v.abs() * 1e-3 + 1e-6, "v={v}, back={back}");
        }
    }

    #[test]
    fn synthetic_onnx_end_to_end_decode_fold_infer_and_arena() {
        // Build synthetic ONNX: Input[1,3,16,16] -> Conv1(3->8, 3x3 p=1) -> BN1 -> Relu -> Conv2(8->4, 1x1) -> Output[1,4,16,16]
        let mut g = Vec::new();
        enc_str(&mut g, 2, "synthetic_cnn");

        // Nodes
        let conv1 = enc_node(
            "Conv",
            "conv1",
            &["x", "w1"],
            &["c1"],
            &[
                enc_attr_ints("kernel_shape", &[3, 3]),
                enc_attr_ints("pads", &[1, 1, 1, 1]),
                enc_attr_ints("strides", &[1, 1]),
            ],
        );
        let bn1 = enc_node(
            "BatchNormalization",
            "bn1",
            &["c1", "bn_g", "bn_b", "bn_m", "bn_v"],
            &["b1"],
            &[enc_attr_float("epsilon", 1e-5)],
        );
        let relu1 = enc_node("Relu", "relu1", &["b1"], &["r1"], &[]);
        let conv2 = enc_node(
            "Conv",
            "conv2",
            &["r1", "w2", "b2"],
            &["y"],
            &[enc_attr_ints("kernel_shape", &[1, 1])],
        );
        for n in [conv1, bn1, relu1, conv2] {
            enc_bytes(&mut g, 1, &n);
        }

        // Initializers (mix of raw_data FP32, float_data FP32, and FP16 raw_data)
        enc_bytes(&mut g, 5, &enc_tensor_f32_raw("w1", &[8, 3, 3, 3], &vec![0.1; 8 * 3 * 9]));
        enc_bytes(&mut g, 5, &enc_tensor_f32_float_data("bn_g", &[8], &[1.0; 8]));
        enc_bytes(&mut g, 5, &enc_tensor_f32_float_data("bn_b", &[8], &[0.0; 8]));
        enc_bytes(&mut g, 5, &enc_tensor_f32_raw("bn_m", &[8], &[0.0; 8]));
        enc_bytes(&mut g, 5, &enc_tensor_f32_raw("bn_v", &[8], &[1.0; 8]));
        enc_bytes(&mut g, 5, &enc_tensor_f16_raw("w2", &[4, 8, 1, 1], &vec![0.25; 4 * 8]));
        enc_bytes(&mut g, 5, &enc_tensor_f32_raw("b2", &[4], &[0.0; 4]));

        // Graph IO
        enc_bytes(&mut g, 11, &enc_value_info("x", DataType::Float, &[1, 3, 16, 16]));
        enc_bytes(&mut g, 12, &enc_value_info("y", DataType::Float, &[1, 4, 16, 16]));

        // ModelProto
        let mut model = Vec::new();
        enc_tag(&mut model, 1, 0);
        enc_varint(&mut model, 8);
        enc_bytes(&mut model, 7, &g);

        let compiled = load_and_plan(&model, &HashMap::new()).expect("load_and_plan");
        assert_eq!(compiled.bn_stats.conv_bn_folded, 1);
        assert_eq!(compiled.graph.nodes.len(), 3); // Conv1(folded), Relu1, Conv2
        let y_id = compiled.graph.name_to_id["y"];
        assert_eq!(
            compiled.graph.tensors[y_id.0].static_shape(),
            Some(vec![1, 4, 16, 16])
        );
        compiled.arena.verify_no_overlap().expect("arena valid");
        // Intermediate tensors: b1 [1,8,16,16]=8192B and r1 [1,8,16,16]=8192B (live together at Relu1).
        assert_eq!(compiled.arena.peak_bytes, 2 * 8192);
    }

    #[test]
    #[ignore = "needs deliverables ONNX models on disk"]
    fn load_and_plan_all_shipped_deliverable_models() {
        let root = std::path::Path::new(
            r"C:\WorkSpace\Aurora Studio\05_deliverables\aurora-studio-raw-fast-20261126-191c43b\models",
        );
        assert!(root.exists(), "deliverables models dir not found: {root:?}");

        fn collect(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    collect(&path, out);
                } else if path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                    out.push(path);
                }
            }
        }

        let mut files = Vec::new();
        collect(root, &mut files);
        files.sort();
        assert_eq!(files.len(), 19, "expected 19 shipped ONNX models");

        let mut total_conv_bn_folded = 0usize;
        let mut total_standalone_bn = 0usize;
        for path in &files {
            let bytes = std::fs::read(path).expect("read onnx");
            let mut overrides = HashMap::new();
            let fname = path.file_name().unwrap().to_str().unwrap();
            match fname {
                "DenseNet121_FP32.onnx" | "EyeDetector.onnx" => {
                    overrides.insert("input".into(), vec![1, 3, 64, 64]);
                }
                "faceDetection.onnx" => {
                    overrides.insert("input".into(), vec![1, 3, 640, 640]);
                }
                "autoRotation.onnx" => {
                    overrides.insert("conv2d_input".into(), vec![1, 384, 384, 1]);
                }
                _ => {}
            }
            let plan = load_and_plan(&bytes, &overrides)
                .unwrap_or_else(|e| panic!("failed on {fname}: {e}"));
            plan.arena
                .verify_no_overlap()
                .unwrap_or_else(|e| panic!("arena overlap in {fname}: {e}"));
            total_conv_bn_folded += plan.bn_stats.conv_bn_folded;
            total_standalone_bn += plan.bn_stats.standalone_bn_fused;
            eprintln!(
                "[OK] {:28} nodes={:4} conv_bn={} standalone_bn={:2} resolved={:4}/{} peak_vram={:6.2} MiB (naive={:6.2} MiB, save={:4.1}%)",
                fname,
                plan.graph.nodes.len(),
                plan.bn_stats.conv_bn_folded,
                plan.bn_stats.standalone_bn_fused,
                plan.resolved_shapes,
                plan.graph.tensors.len(),
                plan.arena.peak_bytes as f64 / 1048576.0,
                plan.arena.naive_bytes as f64 / 1048576.0,
                plan.arena.savings_ratio() * 100.0,
            );
        }
        assert_eq!(total_conv_bn_folded, 7);
        assert_eq!(total_standalone_bn, 70);
    }
}
