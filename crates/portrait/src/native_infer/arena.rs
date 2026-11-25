//! Static VRAM Memory Arena Planner based on tensor liveness analysis.
//!
//! Computes exact `[first_write, last_read]` intervals for all intermediate
//! activation tensors and assigns non-overlapping byte offsets inside a single
//! contiguous GPU buffer (`peak_bytes`), enabling zero-dynamic-allocation
//! execution and CUDA Graph capture.

use std::collections::HashMap;

use crate::native_infer::graph::{IrGraph, TensorId};

/// Alignment in bytes for CUDA global memory allocations (256 B guarantees
/// optimal coalescing and Tensor Core / cuBLAS alignment).
pub const CUDA_DEFAULT_ALIGNMENT: usize = 256;

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

    /// Sanity check: verify that no two tensors with overlapping liveness
    /// intervals `[first_write, last_read]` overlap in `[offset, offset + aligned_bytes)`.
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
            for b in &items[i + 1..] {
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

#[derive(Clone, Copy, Debug)]
struct FreeBlock {
    offset: usize,
    size: usize,
}

/// Build a static memory arena plan for all intermediate tensors in `graph`.
///
/// Graph inputs, graph outputs, and static weights are excluded from the
/// reusable scratch arena (callers bind input/output buffers and persistent
/// weight buffers separately).
pub fn plan_memory_arena(graph: &IrGraph, alignment: usize) -> ArenaPlan {
    let alignment = alignment.max(1);

    // 1. Compute liveness intervals [first_write, last_read] for every tensor.
    let mut first_write: HashMap<TensorId, usize> = HashMap::new();
    let mut last_read: HashMap<TensorId, usize> = HashMap::new();

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
        }
    }

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
        match info.static_byte_size() {
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

    // 3. Linear-scan best-fit allocator with coalescing free list.
    let mut active: Vec<ArenaAllocation> = Vec::new();
    let mut free_list: Vec<FreeBlock> = Vec::new();
    let mut allocations: HashMap<TensorId, ArenaAllocation> = HashMap::new();
    let mut peak_bytes = 0usize;

    for item in pending {
        // Expire tensors whose last_read < item.first_write.
        // Note: tensors with last_read == item.first_write are STILL live during
        // this node (they are inputs to the node producing `item`), so strictly `<`.
        let mut i = 0;
        while i < active.len() {
            if active[i].last_read < item.first_write {
                let expired = active.swap_remove(i);
                insert_and_coalesce(
                    &mut free_list,
                    FreeBlock {
                        offset: expired.offset,
                        size: expired.aligned_bytes,
                    },
                );
            } else {
                i += 1;
            }
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

        let alloc = ArenaAllocation {
            tensor_id: item.id,
            offset,
            size_bytes: item.size_bytes,
            aligned_bytes: item.aligned_bytes,
            first_write: item.first_write,
            last_read: item.last_read,
        };
        active.push(alloc);
        allocations.insert(item.id, alloc);
    }

    ArenaPlan {
        alignment,
        peak_bytes,
        naive_bytes,
        dynamic_skipped,
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
    use crate::native_infer::graph::{IrNode, NodeId, Op};
    use crate::native_infer::onnx_proto::DataType;

    #[test]
    fn arena_reuses_buffers_in_deep_chain_and_never_overlaps() {
        // Chain of 8 Relu nodes on a 1x16x32x32 FP32 tensor (64 KiB each).
        // Without reuse: 7 intermediate tensors = 448 KiB.
        // With liveness reuse: only 2 ping-pong buffers = 128 KiB (>70% savings).
        let mut g = IrGraph {
            name: "chain".into(),
            tensors: Vec::new(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            weights: HashMap::new(),
            name_to_id: HashMap::new(),
        };
        let shape = Some(vec![Some(1), Some(16), Some(32), Some(32)]);
        let mut prev = g.register_tensor("in".into(), DataType::Float, shape.clone(), false, true, false);
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

        let plan = plan_memory_arena(&g, CUDA_DEFAULT_ALIGNMENT);
        plan.verify_no_overlap().expect("no overlap");
        assert_eq!(plan.dynamic_skipped, 0);
        assert_eq!(plan.naive_bytes, 7 * 65536);
        assert_eq!(plan.peak_bytes, 2 * 65536);
        assert!(plan.savings_ratio() > 0.70);
    }

    #[test]
    fn arena_preserves_residual_skip_connection_until_add() {
        // in -> relu0 (t0) -> relu1 (t1) -> relu2 (t2) -> add(t0, t2) -> out
        // t0 must stay alive across relu1 and relu2 until add!
        let mut g = IrGraph {
            name: "residual".into(),
            tensors: Vec::new(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            weights: HashMap::new(),
            name_to_id: HashMap::new(),
        };
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

        let plan = plan_memory_arena(&g, CUDA_DEFAULT_ALIGNMENT);
        plan.verify_no_overlap().expect("residual skip must not overlap");
        // t0, t1, t2 are all simultaneously live at n2 (t0 until n3, t1 read at n2, t2 written at n2).
        assert_eq!(plan.peak_bytes, 3 * 4096);
    }
}
