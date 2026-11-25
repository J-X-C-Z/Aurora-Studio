//! Static `Conv + BatchNormalization` weight folding and standalone BN parameter
//! pre-fusion.
//!
//! For any single-consumer `Conv -> BatchNormalization` pair:
//! ```text
//! y = gamma * (conv(x) - mean) / sqrt(var + eps) + beta
//!   = conv_folded(x)
//! ```
//! where for each output channel `c`:
//! - `s[c] = gamma[c] / sqrt(var[c] + eps)`
//! - `W_folded[c, ...] = W[c, ...] * s[c]`
//! - `b_folded[c] = (b[c] - mean[c]) * s[c] + beta[c]`
//!
//! Eliminating the BN node removes both kernel launch overhead and an entire
//! intermediate feature-map round-trip to VRAM.
//!
//! For standalone `BatchNormalization` nodes that cannot be absorbed into an
//! upstream `Conv` (e.g. Pre-Activation DenseNet `Concat/Pool -> BN -> Relu`),
//! [`pre_fuse_standalone_bn`] collapses the 4 parameters `(gamma, beta, mean, var)`
//! into 2 parameters `(scale_eff, bias_eff)` so runtime executes a single channel-wise
//! FMA (`y = x * scale_eff + bias_eff`).

use std::collections::{HashMap, HashSet};

use crate::native_infer::graph::{IrGraph, NodeId, Op, TensorId};
use crate::native_infer::onnx_proto::{f32_to_f16, DataType, TensorProto};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BnFoldStats {
    /// Number of `Conv -> BatchNormalization` pairs folded into `Conv`.
    pub conv_bn_folded: usize,
    /// Number of remaining standalone `BatchNormalization` nodes pre-fused to 2-param FMA.
    pub standalone_bn_fused: usize,
}

/// Fold all eligible `Conv -> BatchNormalization` pairs in `graph`.
/// When `fuse_standalone` is true, also simplifies remaining `BatchNormalization`
/// nodes into [`Op::BatchNormFused`] with precomputed `(scale_eff, bias_eff)`.
pub fn fold_batch_norms(graph: &mut IrGraph, fuse_standalone: bool) -> BnFoldStats {
    let mut stats = BnFoldStats::default();

    // Count how many nodes consume each tensor, and track producer node per tensor.
    let mut consumer_count: HashMap<TensorId, usize> = HashMap::new();
    let mut producer: HashMap<TensorId, usize> = HashMap::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        for inp in node.inputs.iter().flatten() {
            *consumer_count.entry(*inp).or_insert(0) += 1;
        }
        for out in node.outputs.iter().flatten() {
            producer.insert(*out, idx);
        }
    }
    let graph_outputs: HashSet<TensorId> = graph.outputs.iter().copied().collect();

    let mut nodes_to_remove: HashSet<usize> = HashSet::new();
    let mut tensor_redirects: HashMap<TensorId, TensorId> = HashMap::new();

    // Pass 1: Conv -> BatchNormalization absorption.
    for bn_idx in 0..graph.nodes.len() {
        let eps = match &graph.nodes[bn_idx].op {
            Op::BatchNormalization(a) => a.epsilon,
            _ => continue,
        };
        if graph.nodes[bn_idx].inputs.len() < 5 {
            continue;
        }
        let Some(x_id) = graph.nodes[bn_idx].inputs[0] else {
            continue;
        };
        let Some(bn_out_id) = graph.nodes[bn_idx].outputs.first().copied().flatten() else {
            continue;
        };
        // Only fold if Conv output is consumed solely by this BN and is not a graph output.
        if consumer_count.get(&x_id).copied().unwrap_or(0) != 1 || graph_outputs.contains(&x_id) {
            continue;
        }
        let Some(&conv_idx) = producer.get(&x_id) else {
            continue;
        };
        if !matches!(graph.nodes[conv_idx].op, Op::Conv(_)) {
            continue;
        }

        // Gather static weight tensors for Conv (W, optional B) and BN (scale, B, mean, var).
        let (Some(w_id), Some(gamma_id), Some(beta_id), Some(mean_id), Some(var_id)) = (
            graph.nodes[conv_idx].inputs.get(1).copied().flatten(),
            graph.nodes[bn_idx].inputs[1],
            graph.nodes[bn_idx].inputs[2],
            graph.nodes[bn_idx].inputs[3],
            graph.nodes[bn_idx].inputs[4],
        ) else {
            continue;
        };
        let bias_id = graph.nodes[conv_idx].inputs.get(2).copied().flatten();

        // Do not mutate weights shared across multiple Conv nodes.
        if consumer_count.get(&w_id).copied().unwrap_or(0) != 1 {
            continue;
        }
        if let Some(bid) = bias_id {
            if consumer_count.get(&bid).copied().unwrap_or(0) != 1 {
                continue;
            }
        }

        let (Some(w_tp), Some(gamma_tp), Some(beta_tp), Some(mean_tp), Some(var_tp)) = (
            graph.weights.get(&w_id),
            graph.weights.get(&gamma_id),
            graph.weights.get(&beta_id),
            graph.weights.get(&mean_id),
            graph.weights.get(&var_id),
        ) else {
            continue;
        };

        if w_tp.dims.len() < 2 {
            continue;
        }
        let c_out = w_tp.dims[0] as usize;
        let Ok(mut w_f32) = w_tp.to_f32_vec() else {
            continue;
        };
        let (Ok(gamma), Ok(beta), Ok(mean), Ok(var)) = (
            gamma_tp.to_f32_vec(),
            beta_tp.to_f32_vec(),
            mean_tp.to_f32_vec(),
            var_tp.to_f32_vec(),
        ) else {
            continue;
        };
        if gamma.len() != c_out || beta.len() != c_out || mean.len() != c_out || var.len() != c_out
        {
            continue;
        }
        if w_f32.len() % c_out != 0 {
            continue;
        }
        let stride_per_out_ch = w_f32.len() / c_out;

        let old_bias: Vec<f32> = match bias_id.and_then(|id| graph.weights.get(&id)) {
            Some(btp) => match btp.to_f32_vec() {
                Ok(v) if v.len() == c_out => v,
                _ => continue,
            },
            None => vec![0.0; c_out],
        };

        let mut new_bias = vec![0.0f32; c_out];
        for c in 0..c_out {
            let s = gamma[c] / (var[c] + eps).sqrt();
            let base = c * stride_per_out_ch;
            for k in 0..stride_per_out_ch {
                w_f32[base + k] *= s;
            }
            new_bias[c] = (old_bias[c] - mean[c]) * s + beta[c];
        }

        // Write back folded Conv weight in original precision (FP32 or FP16).
        let orig_dt = w_tp.data_type;
        let new_w_tp = encode_f32_as(&w_tp.name, w_tp.dims.clone(), &w_f32, orig_dt);
        graph.weights.insert(w_id, new_w_tp);

        // Write back or create folded Conv bias.
        if let Some(bid) = bias_id {
            let b_dt = graph.weights.get(&bid).map(|t| t.data_type).unwrap_or(orig_dt);
            let b_name = graph.tensors[bid.0].name.clone();
            let new_b_tp = encode_f32_as(&b_name, vec![c_out as i64], &new_bias, b_dt);
            graph.weights.insert(bid, new_b_tp);
        } else {
            let b_name = format!("{}_bn_folded_bias", graph.nodes[conv_idx].name);
            let new_b_tp = encode_f32_as(&b_name, vec![c_out as i64], &new_bias, orig_dt);
            let new_bid = graph.add_weight(new_b_tp);
            if graph.nodes[conv_idx].inputs.len() < 3 {
                graph.nodes[conv_idx].inputs.resize(3, None);
            }
            graph.nodes[conv_idx].inputs[2] = Some(new_bid);
        }

        // Redirect Conv output directly to BN's output tensor ID so downstream
        // consumers and graph.outputs remain unchanged without dangling IDs.
        graph.nodes[conv_idx].outputs[0] = Some(bn_out_id);
        tensor_redirects.insert(x_id, bn_out_id);
        nodes_to_remove.insert(bn_idx);
        stats.conv_bn_folded += 1;
    }

    // Remove folded BN nodes and re-index NodeIds.
    if !nodes_to_remove.is_empty() {
        let mut kept = Vec::with_capacity(graph.nodes.len() - nodes_to_remove.len());
        for (idx, mut node) in graph.nodes.drain(..).enumerate() {
            if nodes_to_remove.contains(&idx) {
                continue;
            }
            for inp in &mut node.inputs {
                if let Some(tid) = inp {
                    if let Some(&new_tid) = tensor_redirects.get(tid) {
                        *tid = new_tid;
                    }
                }
            }
            node.id = NodeId(kept.len());
            kept.push(node);
        }
        graph.nodes = kept;
    }

    // Pass 2: Pre-fuse remaining standalone BatchNormalization into 2-parameter FMA.
    if fuse_standalone {
        for idx in 0..graph.nodes.len() {
            let eps = match &graph.nodes[idx].op {
                Op::BatchNormalization(a) => a.epsilon,
                _ => continue,
            };
            if graph.nodes[idx].inputs.len() < 5 {
                continue;
            }
            let (Some(gamma_id), Some(beta_id), Some(mean_id), Some(var_id)) = (
                graph.nodes[idx].inputs[1],
                graph.nodes[idx].inputs[2],
                graph.nodes[idx].inputs[3],
                graph.nodes[idx].inputs[4],
            ) else {
                continue;
            };
            let (Some(g_tp), Some(b_tp), Some(m_tp), Some(v_tp)) = (
                graph.weights.get(&gamma_id),
                graph.weights.get(&beta_id),
                graph.weights.get(&mean_id),
                graph.weights.get(&var_id),
            ) else {
                continue;
            };
            let (Ok(gamma), Ok(beta), Ok(mean), Ok(var)) = (
                g_tp.to_f32_vec(),
                b_tp.to_f32_vec(),
                m_tp.to_f32_vec(),
                v_tp.to_f32_vec(),
            ) else {
                continue;
            };
            let c = gamma.len();
            if beta.len() != c || mean.len() != c || var.len() != c {
                continue;
            }
            let mut scale_eff = vec![0.0f32; c];
            let mut bias_eff = vec![0.0f32; c];
            for i in 0..c {
                let s = gamma[i] / (var[i] + eps).sqrt();
                scale_eff[i] = s;
                bias_eff[i] = beta[i] - mean[i] * s;
            }
            let dt = g_tp.data_type;
            let node_name = graph.nodes[idx].name.clone();
            let s_id = graph.add_weight(encode_f32_as(
                &format!("{node_name}_scale_eff"),
                vec![c as i64],
                &scale_eff,
                dt,
            ));
            let b_id = graph.add_weight(encode_f32_as(
                &format!("{node_name}_bias_eff"),
                vec![c as i64],
                &bias_eff,
                dt,
            ));
            let x_in = graph.nodes[idx].inputs[0];
            graph.nodes[idx].inputs = vec![x_in, Some(s_id), Some(b_id)];
            graph.nodes[idx].op = Op::BatchNormFused;
            stats.standalone_bn_fused += 1;
        }
    }

    stats
}

fn encode_f32_as(name: &str, dims: Vec<i64>, values: &[f32], dtype: DataType) -> TensorProto {
    match dtype {
        DataType::Float16 => {
            let mut raw = Vec::with_capacity(values.len() * 2);
            for &v in values {
                raw.extend_from_slice(&f32_to_f16(v).to_le_bytes());
            }
            TensorProto {
                name: name.to_owned(),
                dims,
                data_type: DataType::Float16,
                raw_data: raw,
            }
        }
        _ => TensorProto::from_f32(name, dims, values),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_infer::graph::{BatchNormAttrs, ConvAttrs, IrNode};

    /// Reference NCHW 2D convolution on CPU for mathematical equivalence checks.
    fn cpu_conv2d(
        x: &[f32],
        n: usize,
        c_in: usize,
        h: usize,
        w: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
        c_out: usize,
        kh: usize,
        kw: usize,
        pad: usize,
        stride: usize,
        group: usize,
    ) -> Vec<f32> {
        let oh = (h + 2 * pad - kh) / stride + 1;
        let ow = (w + 2 * pad - kw) / stride + 1;
        let c_in_per_g = c_in / group;
        let c_out_per_g = c_out / group;
        let mut out = vec![0.0f32; n * c_out * oh * ow];

        for ni in 0..n {
            for co in 0..c_out {
                let g = co / c_out_per_g;
                let ci_start = g * c_in_per_g;
                let b = bias.map(|bs| bs[co]).unwrap_or(0.0);
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut acc = b;
                        for ci_rel in 0..c_in_per_g {
                            let ci = ci_start + ci_rel;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = oy * stride + ky;
                                    let ix = ox * stride + kx;
                                    if iy >= pad && iy < h + pad && ix >= pad && ix < w + pad {
                                        let sy = iy - pad;
                                        let sx = ix - pad;
                                        let x_val = x[((ni * c_in + ci) * h + sy) * w + sx];
                                        let w_val = weight
                                            [((co * c_in_per_g + ci_rel) * kh + ky) * kw + kx];
                                        acc += x_val * w_val;
                                    }
                                }
                            }
                        }
                        out[((ni * c_out + co) * oh + oy) * ow + ox] = acc;
                    }
                }
            }
        }
        out
    }

    fn cpu_bn(
        x: &[f32],
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        gamma: &[f32],
        beta: &[f32],
        mean: &[f32],
        var: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        for ni in 0..n {
            for ci in 0..c {
                let inv_std = 1.0 / (var[ci] + eps).sqrt();
                for y in 0..h {
                    for xx in 0..w {
                        let idx = ((ni * c + ci) * h + y) * w + xx;
                        out[idx] = gamma[ci] * (x[idx] - mean[ci]) * inv_std + beta[ci];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn conv_bn_fold_is_mathematically_equivalent_standard_and_depthwise() {
        for (c_in, c_out, group, with_bias) in [
            (3usize, 4usize, 1usize, false),
            (3, 4, 1, true),
            (4, 4, 4, false), // depthwise
            (4, 4, 4, true),  // depthwise with bias
        ] {
            let (n, h, w, kh, kw, pad, stride) = (1, 5, 5, 3, 3, 1, 1);
            let x: Vec<f32> = (0..n * c_in * h * w)
                .map(|i| ((i as f32 * 0.37).sin()) * 2.0 - 0.5)
                .collect();
            let c_in_per_g = c_in / group;
            let weight: Vec<f32> = (0..c_out * c_in_per_g * kh * kw)
                .map(|i| ((i as f32 * 0.71).cos()) * 0.6)
                .collect();
            let bias: Vec<f32> = (0..c_out).map(|i| 0.1 * (i as f32) - 0.15).collect();
            let gamma: Vec<f32> = (0..c_out).map(|i| 0.8 + 0.25 * i as f32).collect();
            let beta: Vec<f32> = (0..c_out).map(|i| -0.3 + 0.2 * i as f32).collect();
            let mean: Vec<f32> = (0..c_out).map(|i| 0.15 * i as f32 - 0.2).collect();
            let var: Vec<f32> = (0..c_out).map(|i| 0.25 + 0.3 * i as f32).collect();
            let eps = 1e-5f32;

            // Reference: Conv followed by BN.
            let conv_ref = cpu_conv2d(
                &x,
                n,
                c_in,
                h,
                w,
                &weight,
                with_bias.then_some(bias.as_slice()),
                c_out,
                kh,
                kw,
                pad,
                stride,
                group,
            );
            let expected = cpu_bn(&conv_ref, n, c_out, h, w, &gamma, &beta, &mean, &var, eps);

            // Build IR graph: Input -> Conv -> BN -> Output.
            let mut g = IrGraph {
                name: "conv_bn_test".into(),
                tensors: Vec::new(),
                nodes: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                weights: HashMap::new(),
                name_to_id: HashMap::new(),
            };
            let in_id = g.register_tensor(
                "x".into(),
                DataType::Float,
                Some(vec![Some(n), Some(c_in), Some(h), Some(w)]),
                false,
                true,
                false,
            );
            g.inputs.push(in_id);
            let w_id = g.add_weight(TensorProto::from_f32(
                "conv_w",
                vec![c_out as i64, c_in_per_g as i64, kh as i64, kw as i64],
                &weight,
            ));
            let b_id = with_bias.then(|| {
                g.add_weight(TensorProto::from_f32("conv_b", vec![c_out as i64], &bias))
            });
            let conv_out = g.register_tensor("conv_y".into(), DataType::Float, None, false, false, false);
            let gamma_id = g.add_weight(TensorProto::from_f32("bn_g", vec![c_out as i64], &gamma));
            let beta_id = g.add_weight(TensorProto::from_f32("bn_b", vec![c_out as i64], &beta));
            let mean_id = g.add_weight(TensorProto::from_f32("bn_m", vec![c_out as i64], &mean));
            let var_id = g.add_weight(TensorProto::from_f32("bn_v", vec![c_out as i64], &var));
            let bn_out = g.register_tensor("bn_y".into(), DataType::Float, None, false, false, true);
            g.outputs.push(bn_out);

            let mut conv_inputs = vec![Some(in_id), Some(w_id)];
            if let Some(bid) = b_id {
                conv_inputs.push(Some(bid));
            }
            g.nodes.push(IrNode {
                id: NodeId(0),
                name: "conv0".into(),
                op: Op::Conv(ConvAttrs {
                    kernel_shape: Some(vec![kh, kw]),
                    strides: vec![stride, stride],
                    pads: vec![pad, pad, pad, pad],
                    dilations: vec![1, 1],
                    group,
                }),
                inputs: conv_inputs,
                outputs: vec![Some(conv_out)],
            });
            g.nodes.push(IrNode {
                id: NodeId(1),
                name: "bn0".into(),
                op: Op::BatchNormalization(BatchNormAttrs {
                    epsilon: eps,
                    momentum: 0.9,
                }),
                inputs: vec![
                    Some(conv_out),
                    Some(gamma_id),
                    Some(beta_id),
                    Some(mean_id),
                    Some(var_id),
                ],
                outputs: vec![Some(bn_out)],
            });

            let stats = fold_batch_norms(&mut g, true);
            assert_eq!(stats.conv_bn_folded, 1);
            assert_eq!(stats.standalone_bn_fused, 0);
            assert_eq!(g.nodes.len(), 1);
            g.validate_topology().expect("topology valid after fold");

            // Evaluate folded Conv on CPU and compare against BN(Conv(x)).
            let fw_id = g.nodes[0].inputs[1].unwrap();
            let fb_id = g.nodes[0].inputs[2].unwrap();
            let folded_w = g.weights[&fw_id].to_f32_vec().unwrap();
            let folded_b = g.weights[&fb_id].to_f32_vec().unwrap();
            let actual = cpu_conv2d(
                &x,
                n,
                c_in,
                h,
                w,
                &folded_w,
                Some(&folded_b),
                c_out,
                kh,
                kw,
                pad,
                stride,
                group,
            );

            let max_diff = expected
                .iter()
                .zip(actual.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-5,
                "fold mismatch (group={group}, bias={with_bias}): max_diff={max_diff}"
            );
        }
    }
}
