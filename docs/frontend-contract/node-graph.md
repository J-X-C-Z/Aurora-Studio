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
2. **并行混合器**（已实现，见文末「2026-11-27」）：所有分支以混合器之前的同一张图 `I` 为输入，
   各分支输出 `B_i`、蒙版 `m_i`（含不透明度），
   `out = I + Σ_i m_i · (B_i − I)`（单个全画面分支时等于该分支）。
3. **图层混合器**：`out = B_0；对 i = 1..n：out = lerp(out, B_i, m_i)`。
4. 完成 2/3 后告诉前端，把 `contract.rs` 里的 `NODE_GRAPH` 状态改为 Ready（已实现，状态保留 Partial，见文末）。

## 2026-11-24：引擎侧剪枝与拒绝语义（已实现）

第 1 条现在由引擎自己保证，不再依赖前端先 `flattened()`：

- **`NodeGraph::plan() -> Result<GraphPlan, GraphError>`**
  - `plan.order`：OUTPUT 依赖的节点，拓扑序（每个节点排在其输入之后）。
  - `plan.layers`：其中 `Layer` 节点，即渲染顺序。
  - `plan.pruned`：图中存在、但输出不依赖（或不能追溯到 INPUT）的 `Layer` 节点。**这些层不得渲染。**
  - `plan.warnings`：不拒绝、但需要提示的结构问题（见下）。
- **`NodeGraph::repaired(&[Layer]) -> NodeGraph`**：按行修复图——缺 INPUT/OUTPUT 补上，空图先连 INPUT→OUTPUT，无对应行的 Layer 节点绕过，图里没有的 `Layer` 行按串联补到输出前。**界面与渲染/导出共用这一份修复**。
- **`NodeGraph::inspect(&[Layer])`**：对修复后的图取 `plan()`，再额外报告 `LayerWithoutRow` / `RowWithoutNode`（都只影响提示，不改变渲染结果）。
- **`layered(&AdjustmentSet) -> Result<Vec<&Layer>, GraphError>`**：渲染、导出、批量、缩略图、取色共用的选层入口。没有存图时返回全部行（按存储顺序）；有存图时先 `repaired(&rows)` 再取 `plan.layers`，**顺序以图为准**，所以新建但还没写进图的层仍会渲染。
- **`render_layers(&AdjustmentSet)`**：渲染热路径的包装。图被拒绝时只渲染背景并记一次 `log::error!`，不猜测顺序。

### 拒绝（错误）

| 错误 | 含义 | 前端应做 |
|---|---|---|
| `GraphError::Cycle { nodes }` | OUTPUT 依赖的节点成环，无求值顺序；`nodes` 只含环上节点 | 高亮 `nodes`，提示断开其中一条边；引擎此时只画背景，不会自行选边 |

- 环**只在 OUTPUT 侧**才算错误：与输出无关的环不影响画面，只进 `pruned`。
- 悬空连线（端点不存在）**不是错误**：引擎丢弃该边并给 `DanglingLink` 警告，输出其余部分照常渲染。
- `render_order()` 仍是界面用的“尽力而为”排序（环在闭合处截断），只为编号/整理服务；渲染和导出不得以它为准。

### 警告（不拒绝）

| 警告 | 含义 |
|---|---|
| `MissingInput { node, input }` | mixer 缺某个输入，但更高编号的输入已连 |
| `EmptyMixer { node }` | 渲染路径上的 mixer 没有任何输入 |
| `LayerWithoutRow { id }` | `Layer` 节点没有对应行；修复时被绕过，不参与渲染 |
| `RowWithoutNode { id }` | `Layer` 行在存储的图里没有节点；修复时按串联补到输出前，**仍会渲染与导出**（提示补节点即可） |
| `NoInputNode` | 图里没有输入节点，无法判定「是否接到底图」；输出可达的层都按已连接处理（并保留此警告） |
| `DanglingLink { link }` | 连线端点不存在；已丢弃该边，其余照常渲染 |

mixer 的结构问题（缺输入、空 mixer）仍放在 `warnings` 里，不静默吞掉；渲染语义见下节。

### 前端侧对应变化

- `state::flattened(adj)`：改为调用 `flattened_checked(adj)`；图被拒绝时返回**无层**的调整集（只画背景），并记一次错误。
- `state::flattened_checked(adj) -> Result<AdjustmentSet, GraphError>`：新增，供需要展示错误的面板使用。

## 2026-11-27：混合器渲染（已实现）

`NodeGraph::mix_plan(&rows)` / `mix_plan(&adj)` 把含混合器的图展开为求值步骤（`MixPlan`）；纯串行图返回 `None`，
仍走原有的平铺图层列表，逐字节不变。`render_layers` / `layered` 仍返回平铺列表（引导图、蒙版网格等只需要"有哪些层"）。
CPU 渲染（`mock::process::apply_layers`，预览、导出、缩略图共用）与取色（`sample_point`）按步骤求值，每个节点的输出是一张图：

- **串联段**（`Chain`）：连续的校正节点（中间节点没有分叉）按原来的串行方式一次渲染，与同样的层在串行图里完全一致。
  禁用或不透明度为 0 的层原样透传。
- **并行混合器**：所有分支以共同输入 `I` 为基准——取所有输入共同的最近祖先（单个输入时即该输入本身，没有祖先时取输入图）。
  `out = I + Σ_i (B_i − I)`。分支自己的蒙版与不透明度已包含在 `B_i` 里（`m_i·(G_i − I) = B_i − I`），所以不再重复乘；
  只有一个全画面分支时等于该分支。结果在 8 位域内求和并截断到 0..255。
- **图层混合器**：按输入编号升序，最低输入是底图；对更高的输入 `i`，设其来源链的第一个非校正节点为 `R_i`、
  链上各层的蒙版覆盖为 `c_i = 1 − Π(1 − m_l)`（`m_l` 含不透明度），则
  `out = B_i + (1 − c_i)·(out − R_i)`，等价于 `lerp(out, G_i, c_i)`（`G_i` 为不带蒙版的调色结果）。
  输入链上没有校正层（直接接输入或混合器）时 `c_i = 1`，即不透明覆盖。

已知近似：亮度/颜色范围蒙版的覆盖按 `R_i`（分支输入）的像素计算，而不是链内逐层的中间值；
接入图层混合器的混合器按不透明处理；混合图按 8 位帧逐段量化，不与串行图逐字节一致（只有含混合器的图受影响）。
GPU 路径只负责显影阶段，图层都在 CPU 的 `apply_layers` 里合成，因此没有需要回退的 GPU 分支。
