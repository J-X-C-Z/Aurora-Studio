//! Aurora Native Inference Engine — Phase 1 Pure-Rust Foundation.
//!
//! Zero-external-dependency ONNX graph loader, strongly-typed IR, static shape
//! inference, `Conv + BatchNormalization` weight folding, static graph fusion
//! (`fuse.rs`: GELU / SiLU / `Conv + activation`, alias and dead-node removal),
//! and liveness-based VRAM Memory Arena Planner.
//!
//! Designed to replace `onnxruntime*.dll` (~213 MB) together with Phase 2
//! custom CUDA operator kernels.

pub mod arena;
pub mod backend;
pub mod cpu_session;
pub mod dispatch;
#[cfg(windows)]
pub mod dx12_session;
pub mod metal_session;
pub mod bn_fold;
pub mod cpu_shape_eval;
pub mod cuda_graph;
pub mod fuse;
pub mod graph;
pub mod onnx_proto;
pub mod ops_conv_gemm;
pub mod ops_elem_norm;
pub mod ops_quant;
pub mod ops_spatial_tensor;
pub mod session;
pub mod selfcheck;

use std::collections::HashMap;
use std::fmt;

use arena::{plan_memory_arena, ArenaPlan, CUDA_DEFAULT_ALIGNMENT};
use bn_fold::{fold_batch_norms, BnFoldStats};
use cpu_shape_eval::CpuShapeEvaluator;
use fuse::{finalize_graph, fuse_activations, FuseConfig, FuseStats};
use graph::{GraphError, IrGraph, NodeId, TensorId};
use onnx_proto::{decode_model, ProtoError};
use ops_conv_gemm::FusedActivation;

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
    /// Graph-rewrite statistics (`fuse.rs`); all-zero when fusion is off.
    pub fuse_stats: FuseStats,
    pub resolved_shapes: usize,
    pub arena: ArenaPlan,
    /// `Conv` node id (already re-indexed into `graph`) -> epilogue activation the
    /// convolution kernel must apply to its output. Populated only when
    /// [`FuseLevel::Full`](fuse::FuseLevel::Full) folded `Conv + Relu/SiLU`;
    /// empty otherwise, so the executor keeps using `FusedActivation::None`.
    pub conv_activations: HashMap<NodeId, FusedActivation>,
}

/// Round trips allowed between the per-op shape rules and the CPU shape
/// subgraph evaluator.
///
/// Each round can unlock the next `Reshape` / `Expand` / `Resize` whose size
/// operand only becomes a constant once an earlier chain resolved, so one round
/// is not enough for a ViT that builds its `[batch, tokens, dim]` shape from
/// `Shape -> Gather -> Concat -> Where`.
const SHAPE_RESOLVE_ROUNDS: usize = 4;

/// Folds the shapes the CPU shape-subgraph evaluator can prove back into the IR.
///
/// `infer_shapes` derives each tensor from the op it is produced by, which stops
/// at the first node that needs a *value* rather than a shape (a `Where` between
/// two size constants, a `Slice` of a `Shape`, a `Concat` of the two). Those
/// chains are exactly how the shipped ViT/FPN models build their reshape
/// targets, so half of `subjectMasking` and `MatchLookColor` stayed dynamic:
/// every intermediate then missed the arena and was allocated per run, which
/// also kept CUDA Graph capture off.
///
/// [`CpuShapeEvaluator`] already evaluates those chains symbolically (it is what
/// the executor uses for the same nodes at run time). Running it once per round,
/// writing the resolved shapes back and re-propagating is enough to make those
/// tensors static. Only unknown dimensions are filled in — a declared dim is
/// never overwritten — so this cannot move a shape that the model stated itself.
fn resolve_shapes_from_subgraphs(graph: &mut IrGraph, input_overrides: &HashMap<String, Vec<usize>>) -> usize {
    let mut resolved = 0usize;
    for _ in 0..SHAPE_RESOLVE_ROUNDS {
        let mut evaluator = CpuShapeEvaluator::from_graph(graph, &HashMap::new());
        evaluator.eval_graph(graph);
        // Hand the folded size values to the per-op rules: `Reshape`, `Expand`,
        // `Resize` and `Split` propagate their shape (and dtype) from an integer
        // operand as long as it is a known constant, and the evaluator has just
        // turned those operands into ones.
        let mut seed: HashMap<TensorId, Vec<i64>> = HashMap::new();
        for info in &graph.tensors {
            let Some(tensor) = evaluator.get_tensor(info.id) else {
                continue;
            };
            if !matches!(
                tensor.data.dtype(),
                onnx_proto::DataType::Int64 | onnx_proto::DataType::Int32
            ) {
                continue;
            }
            let values = tensor.to_i64_vec();
            if values.len() <= 64 {
                seed.insert(info.id, values);
            }
        }
        let mut changed = 0usize;
        for info in graph.tensors.iter_mut() {
            if info.is_weight || info.is_graph_input || info.is_graph_output {
                continue;
            }
            if info.shape.as_ref().is_some_and(|dims| dims.iter().all(|d| d.is_some())) {
                continue;
            }
            let Some(shape) = evaluator.get_shape(info.id) else {
                continue;
            };
            if info.static_shape().as_deref() == Some(shape) {
                continue;
            }
            info.shape = Some(shape.iter().map(|&dim| Some(dim)).collect());
            if info.dtype == onnx_proto::DataType::Undefined
                && let Some(tensor) = evaluator.get_tensor(info.id)
            {
                info.dtype = tensor.data.dtype();
            }
            changed += 1;
        }
        if changed == 0 && seed.is_empty() {
            break;
        }
        resolved += changed;
        let _ = graph.infer_shapes_with_constants(input_overrides, &seed);
    }
    resolved
}

/// Decode an `.onnx` buffer, build the strongly-typed IR, fold BatchNorms,
/// apply the graph-optimization passes selected by [`fuse::ENV_FUSE`],
/// validate topology, infer static shapes, and plan the VRAM arena.
pub fn load_and_plan(
    onnx_bytes: &[u8],
    input_overrides: &HashMap<String, Vec<usize>>,
) -> Result<CompiledPlan, NativeInferError> {
    load_and_plan_with_fuse(onnx_bytes, input_overrides, FuseConfig::from_env())
}

/// Keep CUDA-only epilogues out of backends that execute the source IR. The portable
/// graph still folds BatchNorm; an optional CUDA flag cannot delete a Relu on another device.
pub(super) fn load_and_plan_portable(onnx_bytes: &[u8], overrides: &HashMap<String, Vec<usize>>) -> Result<CompiledPlan, NativeInferError> {
    load_and_plan_with_fuse(onnx_bytes, overrides, FuseConfig::OFF)
}

/// [`load_and_plan`] with an explicit fusion configuration (A/B testing, tests).
pub fn load_and_plan_with_fuse(
    onnx_bytes: &[u8],
    input_overrides: &HashMap<String, Vec<usize>>,
    fuse_config: FuseConfig,
) -> Result<CompiledPlan, NativeInferError> {
    let model = decode_model(onnx_bytes)?;
    let mut graph = IrGraph::from_model_proto(&model)?;
    graph.validate_topology()?;
    let bn_stats = fold_batch_norms(&mut graph, true);
    graph.validate_topology()?;
    // Phase 1: activation fusion. Structural only, so it runs on the un-typed IR and
    // the fused graph is what gets shape-inferred and arena-planned below.
    let mut fuse_stats = fuse_activations(&mut graph, fuse_config);
    if !fuse_stats.is_noop() {
        graph.validate_topology()?;
    }
    let mut resolved_shapes = graph.infer_shapes(input_overrides)?;
    // The per-op rules cannot fold size *values*, so resolve the shape subgraphs
    // (`Shape -> Slice -> Concat -> Where -> Reshape/Expand/Resize`) and let the
    // rules propagate the result. Without this the intermediates of every ViT
    // stay dynamic and the arena planner has to skip them.
    resolved_shapes += resolve_shapes_from_subgraphs(&mut graph, input_overrides);
    // Phase 2: aliasing / dead nodes / `Conv` epilogue, which need inferred dtypes.
    let (tail_stats, conv_activations) = finalize_graph(&mut graph, fuse_config);
    if !tail_stats.is_noop() {
        graph.validate_topology()?;
    }
    fuse_stats += tail_stats;
    let arena = plan_memory_arena(&graph, CUDA_DEFAULT_ALIGNMENT);
    Ok(CompiledPlan {
        graph,
        bn_stats,
        fuse_stats,
        resolved_shapes,
        arena,
        conv_activations,
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

    fn check_routed_model(k: backend::BackendKind) {
        let route=dispatch::open(backend::BackendRequest::Only(k),&selfcheck::model(),&HashMap::new()).expect("open actual backend");
        assert_eq!(route.session.kind(),k);assert!(route.previous_attempts.is_empty());
        for (x,wanted) in [([1.,2.,3.,4.,5.,6.], [0.,7.,0.,16.]), ([-2.,0.,-3.,4.,1.,-1.], [1.5,0.,5.5,0.])] {
            let inputs=HashMap::from([("x".to_owned(), (&x[..],vec![2,3]))]);
            let out=route.session.run_f32(&inputs).expect("matrix, broadcast bias, relu");
            assert_eq!(out.len(),1);assert_eq!(out["out"],(wanted.to_vec(),vec![2,2]));
        }
        let bad=[1.;6];let wrong=HashMap::from([("x".to_owned(),(&bad[..],vec![3,2]))]);
        assert!(route.session.run_f32(&wrong).is_err(),"same element count is not the same shape");
    }
    #[test] fn routed_cpu_named_onnx_matches_literal_results() {check_routed_model(backend::BackendKind::Cpu);}
    #[test] #[cfg(windows)] #[ignore="needs a real CUDA device"]
    fn routed_cuda_named_onnx_matches_literal_results() {check_routed_model(backend::BackendKind::Cuda);}
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
        // Intermediate tensors: b1 [1,8,16,16]=8192B (folded Conv1 output) and
        // r1 [1,8,16,16]=8192B (Relu output). They are live together at Relu1, and
        // Relu is an in-place `unary_f32` op, so r1 reuses b1's bytes instead of
        // getting a second buffer: one arena slot instead of two.
        assert_eq!(compiled.arena.peak_bytes, 8192);
        assert_eq!(compiled.arena.inplace_reuses, 1);
    }

    #[test]
    fn synthetic_gelu_chain_fuses_and_removes_its_arena_buffers() {
        // x[1,8,16,16] -> Div(x, sqrt2) -> Erf -> Add(1) -> Mul(x, .) -> Mul(., 0.5) -> y
        // With fusion off: 5 nodes, 4 live intermediates (peak 2 x 8 KiB after reuse).
        // With fusion on:  1 `Gelu` node, no intermediates at all (peak 0).
        let mut g = Vec::new();
        enc_str(&mut g, 2, "synthetic_gelu");
        for node in [
            enc_node("Div", "div_sqrt2", &["x", "c_sqrt2"], &["t1"], &[]),
            enc_node("Erf", "erf", &["t1"], &["t2"], &[]),
            enc_node("Add", "add_one", &["t2", "c_one"], &["t3"], &[]),
            enc_node("Mul", "mul_x", &["x", "t3"], &["t4"], &[]),
            enc_node("Mul", "mul_half", &["t4", "c_half"], &["y"], &[]),
        ] {
            enc_bytes(&mut g, 1, &node);
        }
        enc_bytes(
            &mut g,
            5,
            &enc_tensor_f32_raw("c_sqrt2", &[], &[std::f32::consts::SQRT_2]),
        );
        enc_bytes(&mut g, 5, &enc_tensor_f32_raw("c_one", &[], &[1.0]));
        enc_bytes(&mut g, 5, &enc_tensor_f32_raw("c_half", &[], &[0.5]));
        enc_bytes(&mut g, 11, &enc_value_info("x", DataType::Float, &[1, 8, 16, 16]));
        enc_bytes(&mut g, 12, &enc_value_info("y", DataType::Float, &[1, 8, 16, 16]));

        let mut model = Vec::new();
        enc_tag(&mut model, 1, 0);
        enc_varint(&mut model, 8);
        enc_bytes(&mut model, 7, &g);

        let unfused = load_and_plan_with_fuse(&model, &HashMap::new(), FuseConfig::OFF)
            .expect("unfused plan");
        assert_eq!(unfused.graph.nodes.len(), 5);
        assert!(unfused.fuse_stats.is_noop());
        assert!(unfused.conv_activations.is_empty());
        // 4 intermediates of 8 KiB, ping-ponged into 2 buffers.
        assert_eq!(unfused.arena.peak_bytes, 2 * 8192);
        unfused.arena.verify_no_overlap().expect("unfused arena valid");

        let fused = load_and_plan_with_fuse(&model, &HashMap::new(), FuseConfig::FULL)
            .expect("fused plan");
        assert_eq!(fused.fuse_stats.gelu_fused, 1);
        assert_eq!(fused.graph.nodes.len(), 1);
        assert!(matches!(fused.graph.nodes[0].op, graph::Op::Gelu));
        assert_eq!(fused.arena.peak_bytes, 0);
        assert_eq!(fused.arena.dynamic_skipped, 0);
        assert_eq!(
            fused.graph.tensors[fused.graph.name_to_id["y"].0].static_shape(),
            Some(vec![1, 8, 16, 16])
        );
    }


    #[test]
    #[ignore = "needs the shipped ONNX models: set AURORA_MODELS_DIR"]
    fn every_statically_shaped_activation_of_shipped_models_is_arena_planned() {
        // Regression guard for the two gaps that kept CUDA Graph capture off the
        // ViT / FPN models: shape subgraphs the per-op rules cannot fold
        // (`Shape -> Gather -> Concat -> Where -> Reshape/Expand/Resize`) and FP16
        // activations that the executor stores in FP32. Both used to leave
        // intermediates out of the arena, which forced one `cuMemAlloc` per
        // tensor per call (thousands per inference on `MatchLookColor`) and made
        // the model uncapturable.
        //
        // `subjectMasking.onnx` keeps a known, measured gap: its stage1/stage2
        // pyramid derives the `Resize` sizes of `/stage1/Resize_4` from a
        // `/stage1/Concat_9` whose operands the CPU shape evaluator still cannot
        // fold, so 426 activations stay dynamic. The expectation below is a
        // ceiling on that gap, not an absence of it.
        let Ok(root) = std::env::var("AURORA_MODELS_DIR") else {
            eprintln!("set AURORA_MODELS_DIR to the models directory");
            return;
        };
        let root = std::path::Path::new(&root).to_path_buf();
        // (relative path, input name, input shape, allowed unplanned activations)
        let cases: &[(&str, &str, &[usize], usize)] = &[
            (r"AIModels\Common\SubjectMasking\subjectMasking.onnx", "x", &[1, 3, 320, 320], 426),
            (r"AIModels\Common\MatchLook\MatchLookColor.onnx", "input.1", &[1, 3, 384, 384], 0),
            (r"AIModels\Common\MatchLook\MatchLookBrightness.onnx", "input.1", &[1, 3, 384, 384], 0),
            (r"AIModels\Common\PeopleMasking\skinMasking.onnx", "input_29", &[1, 512, 512, 3], 0),
            (r"AIModels\Common\FaceDetection\faceDetection.onnx", "input", &[1, 3, 320, 320], 0),
            (r"AIModels\Common\SamEncoder\eSamEncoderPreUInt8.onnx", "input_image", &[1024, 1024, 3], 0),
        ];
        let mut checked = 0usize;
        for &(rel, input_name, shape, ceiling) in cases {
            let path = root.join(rel);
            if !path.is_file() {
                eprintln!("[MISSING] {rel}");
                continue;
            }
            let bytes = std::fs::read(&path).expect("read onnx");
            let overrides = HashMap::from([(input_name.to_string(), shape.to_vec())]);
            let plan = load_and_plan(&bytes, &overrides).expect("load_and_plan");
            let mut unplanned = Vec::new();
            for node in &plan.graph.nodes {
                for out in node.outputs.iter().flatten() {
                    let info = &plan.graph.tensors[out.0];
                    if info.is_weight || info.is_graph_input || info.is_graph_output {
                        continue;
                    }
                    // A tensor the shape evaluator still cannot size is a real
                    // per-run allocation, so this is the honest denominator.
                    if info.static_shape().is_none() {
                        unplanned.push(format!("{} (dynamic)", info.name));
                    } else if !plan.arena.allocations.contains_key(out) {
                        unplanned.push(info.name.clone());
                    }
                }
            }
            eprintln!(
                "[PLAN] {rel} nodes={} peak={:.2}MiB dyn_skipped={} unplanned={}/{}",
                plan.graph.nodes.len(),
                plan.arena.peak_bytes as f64 / 1048576.0,
                plan.arena.dynamic_skipped,
                unplanned.len(),
                ceiling,
            );
            assert!(
                unplanned.len() <= ceiling,
                "{rel}: {} activation(s) are not arena-planned, above the recorded ceiling {ceiling}; first: {:?}",
                unplanned.len(),
                &unplanned[..unplanned.len().min(6)],
            );
            plan.arena.verify_no_overlap().expect("arena overlap");
            checked += 1;
        }
        assert!(checked > 0, "no shipped model was checked");
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
