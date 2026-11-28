//! Static VRAM Memory Arena Planner based on tensor liveness analysis.
//!
//! Computes exact `[first_write, last_read]` intervals for all intermediate
//! activation tensors and assigns non-overlapping byte offsets inside a single
//! contiguous GPU buffer (`peak_bytes`), enabling zero-dynamic-allocation
//! execution and CUDA Graph capture.
//!
//! Two reuse mechanisms keep the arena small:
//!
//! 1. **Liveness recycling** — a tensor whose last reader is behind the current
//!    write returns its block to a coalescing free list.
//! 2. **In-place elementwise reuse** — when an elementwise node of the
//!    `unary_f32` family reads a single-consumer intermediate that dies at that
//!    node, its output is placed on the *same* bytes. The kernel reads `x[i]`
//!    before writing `out[i]` (see `native/cuda/infer_elem_norm.cu`), so this is
//!    bit-exact and turns long `Conv -> Act -> Act` chains into one buffer per
//!    chain instead of two. Chains are allowed (each link is in-place over the
//!    same bytes); the plan records the alias edge so
//!    [`ArenaPlan::verify_no_overlap`] can prove the sharing is deliberate.
//!    Disable with `AURORA_NATIVE_NO_INPLACE=1` for an A/B run.

use std::collections::HashMap;

use crate::native_infer::graph::{IrGraph, Op, TensorId};

/// Alignment in bytes for CUDA global memory allocations (256 B guarantees
/// optimal coalescing and Tensor Core / cuBLAS alignment).
pub const CUDA_DEFAULT_ALIGNMENT: usize = 256;

/// Environment switch that turns off in-place elementwise buffer reuse.
pub const ENV_NO_INPLACE: &str = "AURORA_NATIVE_NO_INPLACE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaAllocation {
    pub tensor_id: TensorId,
    /// Static byte offset from the base pointer of the GPU arena.
    pub offset: usize,
    /// Unaligned logical byte size of the tensor.
    pub size_bytes: usize,
    /// Aligned reservation length inside the arena.
    pub aligned_bytes: usize,
    /// Node index where this tensor is written.
    pub first_write: usize,
    /// Last node index where this tensor is read (inclusive).
    pub last_read: usize,
    /// `Some(src)` when this tensor deliberately shares `src`'s bytes: `src` was a
    /// single-consumer elementwise input that dies at the producing node.
    pub aliased_from: Option<TensorId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArenaPlan {
    pub alignment: usize,
    /// Peak VRAM bytes required for all intermediate activation tensors.
    pub peak_bytes: usize,
    /// Sum of all intermediate activation sizes without memory reuse.
    pub naive_bytes: usize,
    /// Number of intermediate tensors skipped because their shape was still dynamic.
    pub dynamic_skipped: usize,
    /// Number of elementwise outputs placed on their (dead) input's bytes.
    pub inplace_reuses: usize,
    /// Static offset assignment per intermediate `TensorId`.
    pub allocations: HashMap<TensorId, ArenaAllocation>,
}

impl ArenaPlan {
    /// Fraction of VRAM saved by liveness-based buffer reuse (0.0..1.0).
    pub fn savings_ratio(&self) -> f64 {
        if self.naive_bytes == 0 {
            return 0.0;
        }
        1.0 - (self.peak_bytes as f64 / self.naive_bytes as f64)
    }

    /// Follow the `aliased_from` chain to the tensor that owns the bytes.
    fn alias_root(&self, tensor: TensorId) -> TensorId {
        let mut current = tensor;
        // The chain is one link per in-place node; the cap only guards against a
        // malformed plan.
        for _ in 0..=self.allocations.len() {
            match self.allocations.get(&current).and_then(|a| a.aliased_from) {
                Some(next) if next != current => current = next,
                _ => break,
            }
        }
        current
    }

    /// Sanity check: verify that no two tensors with overlapping liveness
    /// intervals `[first_write, last_read]` overlap in `[offset, offset + aligned_bytes)`.
    ///
    /// Deliberate in-place sharing (same alias root, same offset) is allowed and
    /// checked separately, so a real planner bug still fails loudly.
    pub fn verify_no_overlap(&self) -> Result<(), String> {
        let items: Vec<&ArenaAllocation> = self.allocations.values().collect();
        for i in 0..items.len() {
            let a = items[i];
            if a.offset % self.alignment != 0 {
                return Err(format!(
                    "tensor {:?} offset {} not aligned to {}",
                    a.tensor_id, a.offset, self.alignment
                ));
            }
            if a.offset + a.aligned_bytes > self.peak_bytes {
                return Err(format!(
                    "tensor {:?} exceeds peak_bytes {}",
                    a.tensor_id, self.peak_bytes
                ));
            }
            if let Some(src) = a.aliased_from {
                let Some(src_alloc) = self.allocations.get(&src) else {
                    return Err(format!(
                        "tensor {:?} aliases unknown tensor {:?}",
                        a.tensor_id, src
                    ));
                };
                if src_alloc.offset != a.offset || src_alloc.aligned_bytes != a.aligned_bytes {
                    return Err(format!(
                        "tensor {:?} aliases {:?} but offsets differ ({} vs {})",
                        a.tensor_id, src, a.offset, src_alloc.offset
                    ));
                }
            }
            for b in &items[i + 1..] {
                if a.aliased_from == Some(b.tensor_id) || b.aliased_from == Some(a.tensor_id) {
                    continue;
                }
                if self.alias_root(a.tensor_id) == self.alias_root(b.tensor_id) {
                    continue;
                }
                let time_overlaps = a.first_write <= b.last_read && b.first_write <= a.last_read;
                if time_overlaps {
                    let mem_overlaps = a.offset < b.offset + b.aligned_bytes
                        && b.offset < a.offset + a.aligned_bytes;
                    if mem_overlaps {
                        return Err(format!(
                            "arena overlap between {:?} ([{}..{}] @ {}+{}) and {:?} ([{}..{}] @ {}+{})",
                            a.tensor_id,
                            a.first_write,
                            a.last_read,
                            a.offset,
                            a.aligned_bytes,
                            b.tensor_id,
                            b.first_write,
                            b.last_read,
                            b.offset,
                            b.aligned_bytes,
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Planner knobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaOptions {
    /// Alignment in bytes for every arena allocation.
    pub alignment: usize,
    /// Enable in-place reuse for dead single-consumer elementwise inputs.
    pub allow_inplace_reuse: bool,
}

impl ArenaOptions {
    /// Read [`ENV_NO_INPLACE`] (`1` disables the optimisation).
    pub fn from_env(alignment: usize) -> Self {
        Self {
            alignment,
            allow_inplace_reuse: std::env::var(ENV_NO_INPLACE).ok().as_deref() != Some("1"),
        }
    }
}

impl Default for ArenaOptions {
    fn default() -> Self {
        Self::from_env(CUDA_DEFAULT_ALIGNMENT)
    }
}

#[derive(Clone, Copy, Debug)]
struct FreeBlock {
    offset: usize,
    size: usize,
}

/// `true` when the CUDA kernel for `op` reads element `i` of its only input before
/// writing element `i` of its output, which makes an in-place output buffer
/// bit-exact. All of these dispatch to `unary_f32` in `infer_elem_norm.cu`:
///
/// ```text
/// const float v = x[idx];
/// ... compute res from v ...
/// out[idx] = res;
/// ```
///
/// Every other op (`Conv`, `Concat`, `Transpose`, `Softmax`, reductions, binary
/// broadcast) is excluded: those kernels read neighbours, stay resident across
/// several passes, or write a different element than they read.
pub fn allows_inplace_output(op: &Op) -> bool {
    matches!(
        op,
        Op::Relu
            | Op::Sigmoid
            | Op::Gelu
            | Op::SiLU
            | Op::Erf
            | Op::Sqrt
            | Op::Reciprocal
            | Op::Abs
            | Op::Sin
            | Op::Cos
            | Op::HardSwish
            | Op::HardSigmoid { .. }
            | Op::LeakyRelu { .. }
            | Op::Clip
    )
}

/// Element width the executor actually reserves for one element of `dtype` in
/// the scratch arena.
///
/// [`NativeCudaSession`](super::session::NativeCudaSession) keeps every float
/// activation in FP32 storage: FP16 weights are unpacked at upload time and
/// `Cast(to = Float16)` rounds through IEEE FP16 but writes back four bytes, so
/// an activation *declared* `Float16` still needs 4 bytes per element. Sizing
/// the plan from the declared width reserves half the bytes the kernels write,
/// and every FP16 model then falls out of the arena at run time — one
/// `cuMemAlloc` per intermediate, which also keeps CUDA Graph capture off. The
/// declared width stays authoritative for every other type.
pub fn planned_element_bytes(dtype: crate::native_infer::onnx_proto::DataType) -> Option<usize> {
    use crate::native_infer::onnx_proto::DataType;
    let declared = dtype.byte_size()?;
    Some(match dtype {
        DataType::Float16 => declared.max(4),
        other => other.byte_size().unwrap_or(declared),
    })
}

/// Byte footprint of one tensor inside the scratch arena, or `None` while its
/// shape is still dynamic.
pub fn planned_tensor_bytes(info: &crate::native_infer::graph::TensorInfo) -> Option<usize> {
    let shape = info.static_shape()?;
    let elems = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))?;
    elems.checked_mul(planned_element_bytes(info.dtype)?)
}

/// Tensor -> the tensor whose bytes it may reuse, for one graph snapshot.
fn inplace_candidates(
    graph: &IrGraph,
    consumer_count: &HashMap<TensorId, usize>,
) -> HashMap<TensorId, TensorId> {
    let mut aliases: HashMap<TensorId, TensorId> = HashMap::new();
    for node in &graph.nodes {
        if !allows_inplace_output(&node.op) || node.outputs.len() != 1 {
            continue;
        }
        let (Some(out), Some(src)) = (
            node.outputs.first().copied().flatten(),
            node.inputs.first().copied().flatten(),
        ) else {
            continue;
        };
        if out == src {
            continue;
        }
        let out_info = &graph.tensors[out.0];
        let src_info = &graph.tensors[src.0];
        // Both sides must be plain arena tensors: graph inputs/outputs and weights
        // are bound by the caller and never come from the scratch arena.
        let arena_only = |info: &crate::native_infer::graph::TensorInfo| {
            !info.is_weight && !info.is_graph_input && !info.is_graph_output
        };
        if !arena_only(out_info) || !arena_only(src_info) {
            continue;
        }
        // The input must die at this node: one consumer, no alias of its own being
        // reused by a *different* producer, same footprint.
        if consumer_count.get(&src).copied().unwrap_or(0) != 1 {
            continue;
        }
        let (Some(out_size), Some(src_size)) = (
            planned_tensor_bytes(out_info),
            planned_tensor_bytes(src_info),
        ) else {
            continue;
        };
        if out_size == 0 || out_size != src_size {
            continue;
        }
        aliases.insert(out, src);
    }
    aliases
}

/// Build a static memory arena plan for all intermediate tensors in `graph`,
/// with in-place elementwise reuse and [`CUDA_DEFAULT_ALIGNMENT`].
///
/// Graph inputs, graph outputs, and static weights are excluded from the
/// reusable scratch arena (callers bind input/output buffers and persistent
/// weight buffers separately).
pub fn plan_memory_arena(graph: &IrGraph, alignment: usize) -> ArenaPlan {
    plan_memory_arena_with(graph, ArenaOptions::from_env(alignment))
}

/// [`plan_memory_arena`] with explicit options (A/B testing, tests).
pub fn plan_memory_arena_with(graph: &IrGraph, options: ArenaOptions) -> ArenaPlan {
    let alignment = options.alignment.max(1);

    // 1. Compute liveness intervals [first_write, last_read] for every tensor and
    //    how many node inputs read it.
    let mut first_write: HashMap<TensorId, usize> = HashMap::new();
    let mut last_read: HashMap<TensorId, usize> = HashMap::new();
    let mut consumer_count: HashMap<TensorId, usize> = HashMap::new();

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for out in node.outputs.iter().flatten() {
            first_write.entry(*out).or_insert(node_idx);
            last_read.entry(*out).or_insert(node_idx);
        }
        for inp in node.inputs.iter().flatten() {
            last_read
                .entry(*inp)
                .and_modify(|lr| *lr = (*lr).max(node_idx))
                .or_insert(node_idx);
            *consumer_count.entry(*inp).or_insert(0) += 1;
        }
    }

    let aliases = if options.allow_inplace_reuse && !graph.nodes.is_empty() {
        inplace_candidates(graph, &consumer_count)
    } else {
        HashMap::new()
    };

    // 2. Collect intermediate tensors that need arena space.
    struct Pending {
        id: TensorId,
        size_bytes: usize,
        aligned_bytes: usize,
        first_write: usize,
        last_read: usize,
    }

    let mut pending = Vec::new();
    let mut naive_bytes = 0usize;
    let mut dynamic_skipped = 0usize;

    for info in &graph.tensors {
        if info.is_weight || info.is_graph_input || info.is_graph_output {
            continue;
        }
        let Some(&fw) = first_write.get(&info.id) else {
            continue;
        };
        let lr = last_read.get(&info.id).copied().unwrap_or(fw);
        match planned_tensor_bytes(info) {
            Some(0) => continue,
            Some(sz) => {
                let aligned = align_up(sz, alignment);
                naive_bytes = naive_bytes.saturating_add(aligned);
                pending.push(Pending {
                    id: info.id,
                    size_bytes: sz,
                    aligned_bytes: aligned,
                    first_write: fw,
                    last_read: lr,
                });
            }
            None => {
                dynamic_skipped += 1;
            }
        }
    }

    // Sort by (first_write ASC, aligned_bytes DESC) for low fragmentation.
    pending.sort_by(|a, b| {
        a.first_write
            .cmp(&b.first_write)
            .then_with(|| b.aligned_bytes.cmp(&a.aligned_bytes))
    });

    // 3. Linear-scan best-fit allocator with coalescing free list. `active` holds
    //    the tensor ids that own blocks; their (possibly extended) `last_read` and
    //    extent are read from `allocations`, so an in-place alias can prolong the
    //    block it shares.
    let mut active: Vec<TensorId> = Vec::new();
    let mut free_list: Vec<FreeBlock> = Vec::new();
    let mut allocations: HashMap<TensorId, ArenaAllocation> = HashMap::new();
    let mut peak_bytes = 0usize;
    let mut inplace_reuses = 0usize;

    for item in pending {
        // Expire tensors whose last_read < item.first_write.
        // Note: tensors with last_read == item.first_write are STILL live during
        // this node (they are inputs to the node producing `item`), so strictly `<`.
        let mut i = 0;
        while i < active.len() {
            let owner = active[i];
            let expired_last_read = allocations
                .get(&owner)
                .map(|a| a.last_read)
                .unwrap_or(usize::MAX);
            if expired_last_read < item.first_write {
                active.swap_remove(i);
                if let Some(alloc) = allocations.get(&owner) {
                    insert_and_coalesce(
                        &mut free_list,
                        FreeBlock {
                            offset: alloc.offset,
                            size: alloc.aligned_bytes,
                        },
                    );
                }
            } else {
                i += 1;
            }
        }

        // In-place path: reuse the dead input's bytes and extend the lifetime of
        // the whole alias chain so the block cannot be recycled early.
        if let Some(&src) = aliases.get(&item.id) {
            if let Some(src_alloc) = allocations.get(&src).copied() {
                let alloc = ArenaAllocation {
                    tensor_id: item.id,
                    offset: src_alloc.offset,
                    size_bytes: item.size_bytes,
                    aligned_bytes: src_alloc.aligned_bytes,
                    first_write: item.first_write,
                    last_read: item.last_read,
                    aliased_from: Some(src),
                };
                let mut chain = src;
                for _ in 0..=allocations.len() {
                    if let Some(entry) = allocations.get_mut(&chain) {
                        entry.last_read = entry.last_read.max(item.last_read);
                        match entry.aliased_from {
                            Some(next) if next != chain => chain = next,
                            _ => break,
                        }
                    } else {
                        break;
                    }
                }
                allocations.insert(item.id, alloc);
                inplace_reuses += 1;
                continue;
            }
            // Source was not arena-allocated (dynamic shape / skipped): fall through
            // to a normal allocation.
        }

        // Best-fit search in free_list.
        let mut best_idx: Option<usize> = None;
        let mut best_size = usize::MAX;
        for (idx, blk) in free_list.iter().enumerate() {
            if blk.size >= item.aligned_bytes && blk.size < best_size {
                best_size = blk.size;
                best_idx = Some(idx);
            }
        }

        let offset = if let Some(idx) = best_idx {
            let blk = free_list.remove(idx);
            let rem = blk.size - item.aligned_bytes;
            if rem > 0 {
                insert_and_coalesce(
                    &mut free_list,
                    FreeBlock {
                        offset: blk.offset + item.aligned_bytes,
                        size: rem,
                    },
                );
            }
            blk.offset
        } else {
            // Bump-allocate at the end of the arena.
            let off = peak_bytes;
            peak_bytes = peak_bytes.saturating_add(item.aligned_bytes);
            off
        };

        allocations.insert(
            item.id,
            ArenaAllocation {
                tensor_id: item.id,
                offset,
                size_bytes: item.size_bytes,
                aligned_bytes: item.aligned_bytes,
                first_write: item.first_write,
                last_read: item.last_read,
                aliased_from: None,
            },
        );
        active.push(item.id);
    }

    ArenaPlan {
        alignment,
        peak_bytes,
        naive_bytes,
        dynamic_skipped,
        inplace_reuses,
        allocations,
    }
}

fn align_up(n: usize, align: usize) -> usize {
    let rem = n % align;
    if rem == 0 {
        n
    } else {
        n + (align - rem)
    }
}

fn insert_and_coalesce(free_list: &mut Vec<FreeBlock>, block: FreeBlock) {
    free_list.push(block);
    free_list.sort_by_key(|b| b.offset);
    let mut merged: Vec<FreeBlock> = Vec::with_capacity(free_list.len());
    for blk in free_list.drain(..) {
        if let Some(last) = merged.last_mut() {
            if last.offset + last.size == blk.offset {
                last.size += blk.size;
                continue;
            }
        }
        merged.push(blk);
    }
    *free_list = merged;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_infer::graph::{IrNode, NodeId};
    use crate::native_infer::onnx_proto::DataType;

    fn empty_graph(name: &str) -> IrGraph {
        IrGraph {
            name: name.into(),
            tensors: Vec::new(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            weights: HashMap::new(),
            name_to_id: HashMap::new(),
        }
    }

    fn options(allow_inplace_reuse: bool) -> ArenaOptions {
        ArenaOptions {
            alignment: CUDA_DEFAULT_ALIGNMENT,
            allow_inplace_reuse,
        }
    }

    #[test]
    fn arena_reuses_buffers_in_deep_chain_and_never_overlaps() {
        // Chain of 8 Relu nodes on a 1x16x32x32 FP32 tensor (64 KiB each).
        // Without any reuse: 7 intermediate tensors = 448 KiB.
        // Liveness recycling alone: 2 ping-pong buffers = 128 KiB.
        // In-place elementwise reuse: one buffer = 64 KiB, kernel-exact because
        // `unary_f32` reads `x[i]` before writing `out[i]`.
        let mut g = empty_graph("chain");
        let shape = Some(vec![Some(1), Some(16), Some(32), Some(32)]);
        let mut prev =
            g.register_tensor("in".into(), DataType::Float, shape.clone(), false, true, false);
        g.inputs.push(prev);
        for i in 0..8 {
            let is_last = i == 7;
            let out = g.register_tensor(
                format!("t{i}"),
                DataType::Float,
                shape.clone(),
                false,
                false,
                is_last,
            );
            g.nodes.push(IrNode {
                id: NodeId(i),
                name: format!("relu_{i}"),
                op: Op::Relu,
                inputs: vec![Some(prev)],
                outputs: vec![Some(out)],
            });
            prev = out;
        }
        g.outputs.push(prev);

        let recycling_only = plan_memory_arena_with(&g, options(false));
        recycling_only.verify_no_overlap().expect("no overlap");
        assert_eq!(recycling_only.dynamic_skipped, 0);
        assert_eq!(recycling_only.naive_bytes, 7 * 65536);
        assert_eq!(recycling_only.peak_bytes, 2 * 65536);
        assert_eq!(recycling_only.inplace_reuses, 0);

        let inplace = plan_memory_arena_with(&g, options(true));
        inplace.verify_no_overlap().expect("no overlap with aliases");
        assert_eq!(inplace.naive_bytes, 7 * 65536);
        assert_eq!(inplace.peak_bytes, 65536);
        // relu_1 ..= relu_6 reuse their input; relu_7 writes the graph output,
        // which is bound outside the arena.
        assert_eq!(inplace.inplace_reuses, 6);
        assert!(inplace.savings_ratio() > 0.85);
    }

    #[test]
    fn arena_preserves_residual_skip_connection_until_add() {
        // in -> relu0 (t0) -> relu1 (t1) -> relu2 (t2) -> add(t0, t2) -> out
        // t0 must stay alive across relu1 and relu2 until add, so t1 cannot reuse
        // it; t2 may reuse t1 (which dies at relu2).
        let mut g = empty_graph("residual");
        let shape = Some(vec![Some(1), Some(4), Some(16), Some(16)]); // 4 KiB
        let inp = g.register_tensor("in".into(), DataType::Float, shape.clone(), false, true, false);
        g.inputs.push(inp);
        let t0 = g.register_tensor("t0".into(), DataType::Float, shape.clone(), false, false, false);
        let t1 = g.register_tensor("t1".into(), DataType::Float, shape.clone(), false, false, false);
        let t2 = g.register_tensor("t2".into(), DataType::Float, shape.clone(), false, false, false);
        let out = g.register_tensor("out".into(), DataType::Float, shape.clone(), false, false, true);
        g.outputs.push(out);

        g.nodes.push(IrNode { id: NodeId(0), name: "n0".into(), op: Op::Relu, inputs: vec![Some(inp)], outputs: vec![Some(t0)] });
        g.nodes.push(IrNode { id: NodeId(1), name: "n1".into(), op: Op::Relu, inputs: vec![Some(t0)], outputs: vec![Some(t1)] });
        g.nodes.push(IrNode { id: NodeId(2), name: "n2".into(), op: Op::Relu, inputs: vec![Some(t1)], outputs: vec![Some(t2)] });
        g.nodes.push(IrNode { id: NodeId(3), name: "n3".into(), op: Op::Add, inputs: vec![Some(t0), Some(t2)], outputs: vec![Some(out)] });

        let strict = plan_memory_arena_with(&g, options(false));
        strict.verify_no_overlap().expect("residual skip must not overlap");
        // t0, t1, t2 are all simultaneously live at n2.
        assert_eq!(strict.peak_bytes, 3 * 4096);

        let inplace = plan_memory_arena_with(&g, options(true));
        inplace.verify_no_overlap().expect("residual skip must not overlap");
        // t0 is consumed twice (relu1 + add) so t1 keeps its own buffer; t2 reuses
        // t1 because relu2 is its only reader.
        assert_eq!(inplace.inplace_reuses, 1);
        assert_eq!(inplace.peak_bytes, 2 * 4096);
        assert_eq!(
            inplace.allocations[&t2].offset, inplace.allocations[&t1].offset,
            "t2 must share t1's bytes"
        );
    }

    #[test]
    fn binary_and_multi_output_ops_never_reuse_input_buffers() {
        // `Add(t0, t0)` reads its input twice and `Op::Add` is not in the
        // in-place family; nothing may be aliased.
        let mut g = empty_graph("binary");
        let shape = Some(vec![Some(1), Some(4), Some(4), Some(4)]);
        let inp = g.register_tensor("in".into(), DataType::Float, shape.clone(), false, true, false);
        g.inputs.push(inp);
        let t0 = g.register_tensor("t0".into(), DataType::Float, shape.clone(), false, false, false);
        let t1 = g.register_tensor("t1".into(), DataType::Float, shape.clone(), false, false, true);
        g.outputs.push(t1);
        g.nodes.push(IrNode { id: NodeId(0), name: "relu".into(), op: Op::Relu, inputs: vec![Some(inp)], outputs: vec![Some(t0)] });
        g.nodes.push(IrNode { id: NodeId(1), name: "add".into(), op: Op::Add, inputs: vec![Some(t0), Some(t0)], outputs: vec![Some(t1)] });

        let plan = plan_memory_arena_with(&g, options(true));
        assert_eq!(plan.inplace_reuses, 0);
        plan.verify_no_overlap().expect("valid");
    }

    #[test]
    fn forged_alias_is_rejected_by_the_validator() {
        let mut g = empty_graph("forged");
        let shape = Some(vec![Some(1), Some(16), Some(16), Some(16)]);
        let inp = g.register_tensor("in".into(), DataType::Float, shape.clone(), false, true, false);
        g.inputs.push(inp);
        let t0 = g.register_tensor("t0".into(), DataType::Float, shape.clone(), false, false, false);
        let t1 = g.register_tensor("t1".into(), DataType::Float, shape.clone(), false, false, false);
        let out = g.register_tensor("out".into(), DataType::Float, shape.clone(), false, false, true);
        g.outputs.push(out);
        g.nodes.push(IrNode { id: NodeId(0), name: "relu".into(), op: Op::Relu, inputs: vec![Some(inp)], outputs: vec![Some(t0)] });
        g.nodes.push(IrNode { id: NodeId(1), name: "relu2".into(), op: Op::Relu, inputs: vec![Some(t0)], outputs: vec![Some(t1)] });
        g.nodes.push(IrNode { id: NodeId(2), name: "relu3".into(), op: Op::Relu, inputs: vec![Some(t1)], outputs: vec![Some(out)] });

        let mut plan = plan_memory_arena_with(&g, options(false));
        // Pretend the second Relu aliased the first tensor without moving its bytes.
        let mut forged = plan.allocations[&t1];
        forged.aliased_from = Some(t0);
        plan.allocations.insert(t1, forged);
        let err = plan.verify_no_overlap().expect_err("mismatched alias must fail");
        assert!(err.contains("offsets differ"), "unexpected error: {err}");
    }

    #[test]
    fn fp16_activations_are_planned_at_fp32_width() {
        // `NativeCudaSession` keeps every float activation in FP32 storage, so a
        // plan sized from the declared FP16 width halves the reservation and the
        // run then falls out of the arena for every intermediate (which is what
        // used to happen on all four `MatchLook*` models).
        let mut g = empty_graph("fp16");
        let shape = Some(vec![Some(1), Some(64), Some(16), Some(16)]); // 16384 elements
        let inp = g.register_tensor("in".into(), DataType::Float16, shape.clone(), false, true, false);
        g.inputs.push(inp);
        let t0 = g.register_tensor("t0".into(), DataType::Float16, shape.clone(), false, false, false);
        let out = g.register_tensor("out".into(), DataType::Float16, shape.clone(), false, false, true);
        g.outputs.push(out);
        g.nodes.push(IrNode { id: NodeId(0), name: "conv".into(), op: Op::Relu, inputs: vec![Some(inp)], outputs: vec![Some(t0)] });
        g.nodes.push(IrNode { id: NodeId(1), name: "relu".into(), op: Op::Relu, inputs: vec![Some(t0)], outputs: vec![Some(out)] });

        let plan = plan_memory_arena_with(&g, options(true));
        assert_eq!(
            plan.allocations[&t0].size_bytes,
            16384 * 4,
            "an FP16 activation occupies 4 bytes per element in the session"
        );
        assert_eq!(plan.peak_bytes, 16384 * 4);
        plan.verify_no_overlap().expect("valid");
    }

    #[test]
    fn dynamic_shapes_are_skipped_not_silently_sized() {
        let mut g = empty_graph("dynamic");
        let inp = g.register_tensor("in".into(), DataType::Float, None, false, true, false);
        g.inputs.push(inp);
        let t0 = g.register_tensor("t0".into(), DataType::Float, None, false, false, false);
        let out = g.register_tensor("out".into(), DataType::Float, None, false, false, true);
        g.outputs.push(out);
        g.nodes.push(IrNode { id: NodeId(0), name: "relu".into(), op: Op::Relu, inputs: vec![Some(inp)], outputs: vec![Some(t0)] });
        g.nodes.push(IrNode { id: NodeId(1), name: "relu2".into(), op: Op::Relu, inputs: vec![Some(t0)], outputs: vec![Some(out)] });

        let plan = plan_memory_arena_with(&g, options(true));
        assert_eq!(plan.dynamic_skipped, 1);
        assert_eq!(plan.peak_bytes, 0);
        assert_eq!(plan.inplace_reuses, 0);
        plan.verify_no_overlap().expect("valid");
    }
}
