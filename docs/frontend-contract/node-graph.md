# 节点化图层 — 前端已完成，需要引擎接入的部分

2026-11-11。前端已把"图层"改为达芬奇式节点编辑器（`crates/app/src/tools/nodes.rs`、
`crates/app/src/state/nodes.rs`）。引擎侧只加了**纯数据**，没有改渲染：

- `crates/engine/src/layers.rs`：`NodeGraph` / `GraphNode` / `NodeKind` / `NodeLink`，
  以及 `NodeGraph::render_order()`（纯函数）。
- `AdjustmentSet::nodes: Option<NodeGraph>`（`serde` 缺省为 `None`，旧数据不受影响）。

## 数据含义

| 概念 | 存储 |
|---|---|
| 校正节点 | `NodeKind::Layer`，`id` 等于 `AdjustmentSet::layers` 中同 id 的 `Layer`；节点的调色和蒙版就是该 Layer 的 `adjustments` / `mask` / `opacity` / `enabled` |
| 输入 | `NodeGraph::INPUT`（`u64::MAX - 1`）= 经过背景（全局）调整之后的图像 |
| 输出 | `NodeGraph::OUTPUT`（`u64::MAX`） |
| 并行混合器 | `NodeKind::Parallel`，多个输入 |
| 图层混合器 | `NodeKind::LayerMixer`，多个输入，**输入编号越大越在上层**（与达芬奇一致） |
| 连线 | `NodeLink { from, to, input }`：`from` 的输出接到 `to` 的第 `input` 个输入 |

- 新建的校正节点默认蒙版为"整幅画面"（一个 `radius = [2, 2]` 的 Radial），即达芬奇
  "节点默认作用于全画面"；画第一个窗口时前端会替换掉它。
- 外部节点（Outside）= 复制所选节点的蒙版并取反（快照，因为 `MaskShape::Copy` 目前权重为 0）。

## 前端已保证的事

1. 每次编辑图后，`layers` 按 `render_order()` 排序（未连接的节点排在最后）。
   因此**只按列表顺序串联渲染**的引擎，对纯串行图结果完全正确。
2. 预览、缩略图、取色发给引擎的 `AdjustmentSet` 已经过 `flattened()`：
   只包含输入→输出路径上的层，按图顺序。

## 需要引擎做的（按优先级）

1. **导出 / 批处理已接入路径过滤**：共享的 `mock::process::layer_plans` 按
   `render_order()` 选择图层；存储调整直接进入引擎时，也会跳过未连接节点。
   预览、导出、缩略图与取色共用此规则。纯串行图与前端 `flattened()` 逐像素一致，
   由 `stored_node_graph_matches_flattened_preview_for_export_and_picker` 验证。
2. **并行混合器**（目前前端提示"按串联渲染"）：所有分支以混合器之前的同一张图 `I` 为输入，
   各分支输出 `B_i`、蒙版 `m_i`（含不透明度），
   `out = I + Σ_i m_i · (B_i − I)`（单个全画面分支时等于该分支）。
3. **图层混合器**：`out = B_0；对 i = 1..n：out = lerp(out, B_i, m_i)`。
4. 完成 2/3 后告诉前端，把 `contract.rs` 里的 `NODE_GRAPH` 状态改为 Ready。
