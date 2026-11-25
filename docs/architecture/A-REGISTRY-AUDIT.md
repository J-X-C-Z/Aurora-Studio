# A-Registry 审计：人像效果的注册 / 能力 / UI 状态

状态：只读审计 + descriptor 种子。基线 `8118a4a`；本文件引用的是审计时工作树
（含其他 lane 的未提交改动）的行号，文件移动后需重新核对。相关决策：ADR-006、
ADR-007；迁移映射：`ARCHITECTURE-MIGRATION-MAP.md` §P1-3。

本审计不修改任何路由、算法、UI 或证据表。它为
`aurora_portrait::effect_descriptor`（本轮新增的种子）确定“单一事实来源”的边界。

---

## 1. 现在有哪些地方各自维护状态

| # | 位置（file:line） | 维护什么 | 与谁重复 | 备注 |
|---|---|---|---|---|
| 1 | `crates/portrait/src/kernels/mod.rs:72-111` | 把所有 `kernels::*::META` 汇总成一个 `Vec<EffectMeta>`；按 PF 排序并断言无重复 PF | 与 #2、#3、#6 的 PF 集合重复 | `registry_impl` 是 crate 内唯一的 kernel 汇总点，但每个 family 仍各自写 `META` |
| 2 | `crates/portrait/src/kernels/*.rs` 的 `pub const META`（约 30 处；如 `skin_models.rs:401`、`radiance.rs:16`、`face_detail.rs:15`、`makeup.rs:521`、`face_geometry.rs:323`、`facial_smooth.rs:9`、`hair_body.rs:18`、`skin_adjust.rs:19`、`moisture.rs:283`、`smile_warp.rs:650`、`body_geometry.rs:27`、`eye_geometry.rs:151`、`skin_color.rs:22`、`tone_unify.rs:24` 等） | 每个效果的 PF、key、label、`Access`、input 名称 | 与 #3 capability 行的 pf/key/label/inputs 重复；与 #5 router 的 PFS/CPU_ENTRY 重复；与 #13 descriptor 将来重复 | `META` 是 crate 内的第二套效果表；新增效果必须改对应 family 文件 |
| 3 | `crates/portrait/src/lib.rs:144-204` | `EffectMeta` 类型；`catalog()`、`lookup(pf)`、`catalog_json()` | 汇总 #2；key/label/inputs 与 #4/#10 重复 | `catalog_json` 是给宿主的机器可读输出，但和 #4 的 `to_json` 是两套 JSON |
| 4 | `crates/portrait/src/capability_registry.rs:14-164` | 7 级证据 `EVIDENCE_LEVELS`；`CapabilityRow`、`rows()`、`effects()`、`for_pf()`、`to_json()` | key/label/inputs/后端文字与 #2/#3 重复；证据解析与 #10 重复 | 数据源是 `crates/portrait/assets/portrait/capability-registry-20261124.tsv`（`REGISTRY_TSV`, line 28） |
| 5 | `crates/portrait/src/effect_router.rs:80-140`（trait 常量）+ adapter 实现 202-1040 | 每个 adapter 的 `NAME`、`PFS`、`KERNELS`、`CPU_ENTRY`、dispatch 证据；`ADAPTERS` 列表 | PF/CPU 入口/后端存在性与 #2/#3/#5 重复 | 路由事实（能否 GPU、是否要求 dispatch）只在这里；但它不是产品 dispatch（产品在 engine） |
| 6 | `crates/engine/src/portrait_stage.rs:3762-3800` | `SUPPORTED_EFFECTS` “能从 frame+mask+bundled LUT 跑”的 PF 列表 | 与 #2 的 `Access::Cpu`、#5 的 adapter 列表、#9 的 live 列表重复 | 新增效果通常要同时改这里 |
| 7 | `crates/engine/src/portrait_stage.rs:3821-3853` | `face_mesh_group(pf)`：哪些 PF 属于同一 mesh 组 | 与 #2 的 key/label、#8 `META` 重复 | 左右配对/组语义只在 engine |
| 8 | `crates/engine/src/portrait_stage.rs:4453-4464` | `supports_effect(pf)`：`SUPPORTED_EFFECTS ∪ B3_MAKEUP_ROWS ∪ 8025/8026 ∪ pair ∪ mesh ∪ body ∪ plump ∪ repair ∪ historical` | 把多个下级列表拼起来；是所有 PF 可执行性的隐含总表 | 这是“效果能否进入 stage”的真正总闸，但不对外导出 |
| 9 | `crates/engine/src/portrait_stage.rs:4711` | `effects.sort_by_key(|&(pf, _)| pf)`：所有效果按 PF 升序执行 | 与 descriptor 的 `order_constraints` 语义重叠（当前只有 PF 序） | 任何显式顺序约束目前无处表达 |
| 10 | `crates/engine/src/portrait_stage.rs:6570-6604`（skin 输入检查在 6585-6604） | 每种效果需要的 input 在运行时逐 PF 检查；缺什么报 `MissingInput` | input 需求与 #2 `EffectMeta::inputs`、#4 TSV `input_requirements` 重复 | 三处文字描述可能不同步 |
| 11 | `crates/engine/src/portrait_stage.rs:6898-6933` | 主 CPU dispatch `match pf`（如 8068/8069/8070/21015/30096/90005/90007） | 与 #5 router、#2 `META` 重复 | 新增效果要在这里加 arm |
| 12 | `crates/engine/src/portrait_stage.rs:9048-9130` | 皮肤效果 CUDA dispatch `match pf`；决定使用 `cuda_router` 或直接调用 `cuda_skin_models`/`cuda_lut` | 后端可用性与 #3 TSV `cpu_backend/gpu_backend`、#5 `ADAPTERS` 重复 | 同一 PF 的 GPU 路径在 engine 和 router 两处描述 |
| 13 | `crates/engine/src/portrait_stage.rs:9268-9391` | `apply_skin_tint/temperature/gloss/texture/clarity` 包装：degree→signed UI、LUT 选择、缺资产时的 refusal | 参数范围/中性值与 #15/#16 重复 | 参数语义散落在包装函数里 |
| 14 | `crates/engine/src/portrait_makeup_bundle.rs:24` | `B3_MAKEUP_ROWS: [u32; 14]` | 与 #2 `makeup::META`、#8 重复 | 14 行 makeup 的同一 PF 集合 |
| 15 | `crates/engine/src/portrait_face_stage.rs:311`、`portrait_body_stage.rs:603`、`portrait_plump_stage.rs:19`、`portrait_repair_stage.rs:693`、`portrait_historical_skin.rs:54-56`、`portrait_historical_tone.rs:225` | 各 stage 的 `pair()` / `supports()` / `PFS` 列表 | 与 #8 重复，且分散在 6 个文件 | 新增一个 PF 可能要在其中多处登记 |
| 16 | `crates/engine/src/portrait_map.rs:122-129` 及行表 | `C1Param`：id、key、`Family`、`Target`（产品语义/AdjustmentSet 映射） | key 与 #2/#3/#10 重复，但 `Target` 语义是独有的 | 这是唯一记录“产品语义”的表；descriptor 不应复制 `Target` |
| 17 | `crates/engine/src/param_meta.rs:78-178` | C1 工具滑块的 min/max/default/unit/step/group | 与 #2 的部分数字、#15 的默认值重复；不是按 PF 组织 | 只覆盖 C1 原生工具，不含全部人像 PF |
| 18 | `crates/app/src/contract.rs:170-178` | `UNRESOLVED`：只能按 id band 推断、无控件的 id | 与 #19 JSON 的 `unresolved` 重复 | 解析 `docs/frontend-contract/portrait-algorithms.json` |
| 19 | `crates/app/src/contract.rs:180-337` | `RawParam`/`PortraitParam`/`PORTRAIT`：从 JSON 得到 id、default、control、key、label、confidence、evidence | default/key/label 与 #2/#3/#20/#21 重复 | UI 参数目录 |
| 20 | `crates/app/src/contract.rs:249-258` | `BANDS`：8 个 id 段及面板标题 | 与 #21 的分组、#16 family 部分重叠 | UI 分组不是效果语义 |
| 21 | `crates/app/src/contract.rs:378-577` | `LABELS`：英文/中文标签 | 与 TSV label、#2 label 重复 | 只手写已核实项；其余回退 humanized key |
| 22 | `crates/app/src/evidence.rs:21-225` | **第二份** capability TSV 解析：`REGISTRY`、`LEVELS`、`Effect`、`BY_PF`、`for_pf`、`availability`、`without_control` | 与 #4 的解析/证据等级重复；`catalog_key` 硬编码 PF30096→`sediao`（87-92） | 数据源是 `docs/frontend-contract/portrait-capability-registry-20261124.tsv`，与 #4 的 assets 副本**不是逐字节相同**（见 §2） |
| 23 | `crates/app/src/tools/portrait.rs:30-45` | `RETOUCH_GROUPS`、`SKIN_SECTIONS`：面板/分组 | 与 #20 `BANDS`、#19 分组重复 | 只影响 UI |
| 24 | `crates/app/src/tools/portrait.rs:48-69` | `FACE_PAIRS`：20 组左右名称，顺序必须与 `WIRED_FACE_GEOMETRY_EFFECTS` 对应 | 与 #2 face_geometry `META` 的左右配对重复 | 有测试保证顺序（line 223-229） |
| 25 | `crates/app/src/tools/portrait.rs:73-188` | `WIRED_SKIN_EFFECTS`、`WIRED_EYES_EFFECTS`、`WIRED_HAIR_EFFECTS`、`WIRED_TEETH_EFFECTS`、`WIRED_FACE_GEOMETRY_EFFECTS`、`WIRED_FACE_SINGLE_EFFECTS`、`WIRED_PLUMP_EFFECTS`、`WIRED_TOTAL`：UI “Live” 集合、默认/中性值、标签 | “哪些效果真正可交互”与 #6/#8 的 SUPPORTED/supports 重复；默认值与 #17/#19 重复 | `has_live_portrait_effect`（200-210）是预览是否需要重算的第二套判定 |
| 26 | `crates/app/src/tools/portrait.rs:254-262` | `is_wired_key()`：某个 `pc:<id>` 是否已有专属面板 | 与 #25 的集合重复 | 避免同一 key 出现两次 |
| 27 | `docs/frontend-contract/portrait-algorithms.json` | 每个 id 的 `default`、`ui_key`、`config_effects`、confidence/evidence | 被 #18/#19/#25 消费；default 数量与 #17/#25 可能不同 | 生成物；不是运行时代码 |
| 28 | `docs/frontend-contract/portrait-capability-registry-20261124.tsv` | 证据表的 docs 副本 | 与 #4 的 assets 副本重复（63/66 行 notes 与 2 行 key/input 不同） | 两个 crate 各读一份 |
| 29 | `crates/portrait/src/effect_descriptor.rs`（本轮） | 稳定 id、PF 别名、参数规格、输入依赖、CPU reference、后端集合、顺序约束、notes | 目前是新增种子，尚未替换 #2/#4/#5/#8/#25 | 纯数据；不执行效果，不依赖 engine |
| 30 | `docs/frontend-contract/catalog.md`、`portrait-evidence.md` | 散文状态说明 | 与 #4 的证据等级描述重复（叙述性） | 仅文档 |

### 1.1 “新增一个效果”今天要改哪里

以已种子的六个皮肤效果为例，新增/修改一个 PF 通常会碰到：

1. family 的 `META`（#2）——PF/key/label/inputs；
2. `capability_registry` 的 TSV 副本（#4，且要同步 docs 副本 #28）；
3. `effect_router` 的 adapter 常量或 `ADAPTERS`（#5）；
4. `portrait_stage` 的 `SUPPORTED_EFFECTS`、`supports_effect`、required-input 分支、CPU/CUDA `match`（#6/#8/#10/#11/#12）；
5. app 的 `WIRED_*`、分组、标签（#23/#24/#25/#26）；
6. 可能还要 `portrait_map` 的 `C1Param`（#16）和 UI JSON（#27）。

这就是 §11 指出的“加一个效果要改多处中央 match/UI 列表”。

---

## 2. 已核实的具体不一致

以下差异均在本机用命令核实；不是推断。

### 2.1 两份 capability TSV 不同

```
docs/frontend-contract/portrait-capability-registry-20261124.tsv
  sha256 f3986958a2415cd08bdcd3a6a98d5b2da6a23df90bfd7c4d68d0f41b8ccc9bc1, 76393 bytes
crates/portrait/assets/portrait/capability-registry-20261124.tsv
  sha256 ce2e8dec8085e2b731fdb54643d5c2b837ac6b3cd9bfdb15e0f1d02e5a1be790, 66399 bytes
```

- 两份都是 154 行（含表头）、153 个数据行；`kind`/`pf` 列完全一致。
- 66 行只差 `evidence_notes`（第 18 列）。
- 2 行的 `key` 或 `input_requirements` 不同：
  - PF90021：assets 为 `PF90021_UI_KEY_NOT_RECOVERED`，docs 为 `lipMakeupEnhance`；
  - PF8/PF9：`input_requirements` 措辞不同（assets 写具体 mask 类型，docs 写
    `face-parsing RGBA mask` / `blusher RGBA mask`）。
- 七个证据列（ev1..ev7）在两份文件中一致。

后果：`aurora-portrait` 的 `for_pf`/`to_json` 与 app 的 `evidence.rs` 看到的是
不同文本；一个 crate 内 `row.key` 可能是占位符，另一个 crate 内已是 UI key。
当前 app 用 `catalog_key()` 硬编码修正 PF30096，而 PF90021 的 key 差异没有统一
修正。

### 2.2 PF30096 的注册信息是占位符

`capability_registry` 行（两份 TSV 都是）：`key=未验证`、`label=未验证`、
`cpu_backend=未验证`、`gpu_backend=未验证`，但七个证据列中有 ev1/ev2/ev4/ev5。
app 的 `evidence.rs:87-92` 用硬编码把 PF30096 映射到 `sediao`。同一个 PF 的
“叫什么、能跑什么后端”在两个 crate 里由不同代码回答。

### 2.3 证据列与代码中的后端不完全一致

- PF8069：`capability_registry` 的 `gpu_backend=探针 CPU`、ev3(`gpu_kernel`)=未验证，
  但仓库里有 `crates/portrait/src/cuda_skin_models.rs` 的 CUDA 路径和
  `portrait_stage.rs:9048+` 的调度。即“适配器存在/已编译”不等于“有 GPU 证据”。
- PF30096：`gpu_backend=未验证`，但 `cuda_lut.rs` 存在且 engine 会尝试 CUDA。
- PF8068/PF8070/PF21015/PF90005 的 ev3 有证据，`gpu_backend=探针实际 CUDA`。

这正是 descriptor 把 `StaticState` / `EvidenceState` / `RuntimeState` 拆开的原因：
不能用一个 bool 同时回答“编译了没有 / 有没有对拍 / 现在能不能跑”。

### 2.4 UI 的 “Live” 集合与 engine 的可执行集合是两套

- app `WIRED_*`（#25）决定哪些控件显示为可交互；
- engine `SUPPORTED_EFFECTS` + `supports_effect`（#6/#8）决定哪些 PF 能进入 stage；
- `capability_registry` ev5 又决定面板文案是否写“已接入主预览”。

三者没有统一的生成源。当前没有发现它们互相矛盾的确定证据（unverified），但存在
漂移风险：例如 `WIRED_SKIN_EFFECTS` 的默认/中性值（8068=0.5、8069=0.5、8070=0.0、
21015=0.5、30096=0.5、90005=0.0）只写在 UI 表里，engine 的包装函数默认值另算
（`portrait_stage.rs:6539` 对 8069 用 `unwrap_or(0.5)`）。

---

## 3. 提议的单一事实来源

保留三类**不同**的信息，不再混写：

| 信息 | 单一来源 | 消费者 |
|---|---|---|
| 效果身份、PF 别名、参数规格、输入依赖、CPU reference、后端集合 | `aurora_portrait::effect_descriptor::{EffectDescriptor, EffectRegistry}` | `capability_registry`（key/inputs 的交叉校验）、`effect_router`（PFS/CPU_ENTRY 校验）、engine dispatch（输入检查、后端选择）、app（控件/分组） |
| 七级证据的**单元格文本** | capability TSV（只保留一份；另一份由构建/测试生成或删除） | descriptor 测试、app evidence 视图 |
| 产品语义（`Family`/`Target`） | `engine::portrait_map::C1_PARAMS` | engine；descriptor 不复制 |
| UI 本地化、面板分组、显示顺序 | app `contract.rs`/`tools/portrait.rs` | 仅 app |

边界规则：

- descriptor **不执行**效果，也不决定路由；它只声明事实。
- `capability_registry::CapabilityRow` 的 key/label/inputs 将来从 descriptor 派生，
  evidence 七列仍来自 TSV（证据是外部事实，不能由代码臆造）。
- `StaticState` 只回答“这个 target 上 adapter 编译了没有”；`EvidenceState` 只回答
  “文档记录到第几级”；`RuntimeState` 由宿主在真正检查驱动/模型/LUT/输入后填写。
  三者不能折叠成一个 `bool`（有测试保证）。

---

## 4. 迁移顺序（每步独立可回滚）

以下顺序与 `ARCHITECTURE-MIGRATION-MAP.md` §P1-3 一致；本轮只完成 step 1。

1. **（本轮）建 descriptor 与种子**：6 个 skin 效果，PF 与 capability 行有测试断言。
   不改任何现有路由/UI。回滚：删除 `lib.rs` 的 `pub mod effect_descriptor;` 即可。
2. **统一 TSV 副本**：确认 63/66 行 notes 与 2 行 key/input 差异哪一份是准；保留
   一份 canonical（建议 assets 副本），另一份改为生成物或删除。需要仓库所有者裁决
   PF90021 的 key 与 PF8/9 的输入措辞。测试：两份若仍存在则 sha256 相同。
3. **descriptor 覆盖全部 ev5 效果，再覆盖全部 147 行**；为每个 descriptor 增加
   `capability_registry` 行断言（PF、key、evidence 等级）。仍不替换调用点。
4. **`capability_registry` 从 descriptor 派生 key/label/inputs**，只保留 evidence
   七列；`lib.rs::catalog/catalog_json` 改为从同一 registry 生成，废弃 family `META`
   的重复字段（先保留 compatibility re-export）。
5. **`effect_router` 交叉校验**：测试断言每个 `AdapterMeta::pfs` 都能在
   descriptor registry 中解析，且 `CPU_ENTRY` 与 descriptor 的 `cpu_reference` 一致；
   路由逻辑不变。
6. **app 改为 descriptor 驱动**：`WIRED_*` 集合与默认值从 descriptor 派生；
   `contract.rs` 只保留本地化/分组；`evidence.rs` 不再解析第二份 TSV。先保持现有
   面板 id 和顺序，避免 UI 回归。
7. **engine 收敛 dispatch**：`SUPPORTED_EFFECTS` + 各 stage 的 `PFS`/`supports`
   改为一张由 descriptor 校验的 PF→handler 表；required-input 检查从 descriptor 的
   `inputs` 生成。此步风险最高，必须在第 6 步稳定并有回归测试后做。

明确不做：不改算法/PTX/MSL/C1 颜色链；不新增 crate；不把证据等级压成 bool；
不删除仍含外部分支/未验证项的 TSV 行。

---

## 5. 种子效果（本轮）

`EffectDescriptor` 种子（`crates/portrait/src/effect_descriptor.rs`,
`seed_descriptors()`）只包含“CPU kernel 真实存在 + capability 行存在 + 参数默认值
有出处”的效果。参数是 engine 消费的存储值 `pc:<pf>`（`0.0..=1.0`），UI 显示为
`0..=100`；默认值来源已写在每条 `notes` 里。

| PF | EffectId | registry key | default | ev1..ev7 证据 | 后端（代码） |
|---|---|---|---|---|---|
| 8068 | `skin.gloss` | `skinGloss` | 0.5 | Y Y Y Y Y - - | Cpu, Cuda（`cuda_gloss`/router `Gloss`） |
| 8069 | `skin.texture` | `skinTexture` | 0.5 | Y Y - Y Y - - | Cpu, Cuda（`cuda_skin_models`，ev3 未验证） |
| 8070 | `skin.clarity` | `skinClarity` | 0.0 | Y Y Y Y Y - - | Cpu, Cuda（`cuda_skin_models`） |
| 21015 | `skin.color_hue` | `skinColorHue` | 0.5 | Y Y Y Y Y - - | Cpu, Cuda（`cuda_lut`/router `SkinTint`） |
| 30096 | `skin.temperature` | 未验证（app 硬编码 `sediao`） | 0.5 | Y Y - Y Y - - | Cpu, Cuda（`cuda_lut`，ev3 未验证） |
| 90005 | `skin.whitening` | `skinWhitening` | 0.0 | Y Y Y Y Y - - | Cpu, Cuda（router `SkinAdjustLut`） |

（`Y` = 该级有证据，`-` = `未验证`。来源：
`crates/portrait/assets/portrait/capability-registry-20261124.tsv`，两份 TSV 的
证据列一致。）

未种子但同样 ev5 的 PF90004（radiance）与 PF90007（rosy）只是本轮数量上限，不含
阻塞性差异；可原样加入。

参数默认值出处：

- `docs/frontend-contract/portrait-algorithms.json`：PF8069=0.5、PF21015=0.5、
  PF30096=0.5、PF90005=0.0；
- `crates/app/src/tools/portrait.rs:73-99`（`WIRED_SKIN_EFFECTS`）对 JSON 未收录的
  PF8068=0.5、PF8070=0.0 提供 live 中性值；
- kernel 入口的范围：`skin_models.rs:95-102`（gloss 0..100 native percent）、
  `skin_models.rs:304-310`（texture UI -100..100）、`skin_models.rs:326-350`
  （clarity degree 0..1）。

---

## 6. 未验证 / 需要裁决

- **许可证**：workspace `Cargo.toml` 为 MIT，`AGENTS.md` 要求 Proprietary；本 lane
  未修改，与 ADR-009 未决问题一致。需仓库所有者裁决。
- **PF90021 key 差异**（§2.1）：哪一份 TSV 是准，未裁决。
- **PF8/PF9 input 措辞差异**（§2.1）：两份都声称是同一效果，未核对源码语义。
- **UI “Live” 集合与 engine 可执行集合是否存在实际矛盾**：本轮只核对了六个种子
  PF 的默认值/范围，未逐项对比全部 147 行，标记为 unverified。
- **非 Windows 构建**：本轮审计时 `cargo test -p aurora-portrait --locked` 在 macOS
  无法完成（Windows-only CUDA example、`cuda_radiance` 未对齐 `from_raw_parts`）。
  统筹随后修复了这两处（example 在非 Windows 变为 no-op；测试解码到对齐缓存），
  修复后 `cargo test -p aurora-portrait --lib` 为 **675 passed / 2 failed**，两个失败
  都是缺真实资产。descriptor 自身 12 个测试通过。Windows 原生结果仍未验证。

---

## 7. 验证记录

本机（macOS，rustc 1.98.1）实际执行：

```
cargo check -p aurora-portrait --locked
  -> Finished（仅既有 warnings）

cargo test -p aurora-portrait --lib --locked effect_descriptor -- --test-threads=1
  -> 12 passed; 0 failed

cargo test -p aurora-portrait --lib --locked -- --test-threads=1
  -> 修复对齐 UB 后：675 passed; 2 failed（缺真实资产）
```

因此：descriptor 代码与其测试已在本机验证；整个 crate 的两个失败是外部资产缺失，
不是 descriptor 回归。Windows 原生结果未验证。
