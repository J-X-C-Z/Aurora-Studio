# 人像能力证据（七级）与前端展示规则

- 日期：2026-11-24
- 会话：frontend-opus（rv24）
- 数据源：`03_algorithms/reports/PORTRAIT-CAPABILITY-REGISTRY-20261124.tsv`
- 前端内嵌副本：`docs/frontend-contract/portrait-capability-registry-20261124.tsv`
  86,739 字节，SHA256 `d1b92c6f0b1d15b2807f917c1298c5cbac4f72449c38e9f4e6af92a28fed85c4`（2026-11-25 与源文件及
  `crates/portrait/assets/portrait/` 副本逐字节一致；`cargo test -p portrait-app --lib evidence::` 会校验）
- 解析与判定代码：`crates/app/src/evidence.rs`

## 1. 七级证据

| 级别 | 列 | 含义 |
|---|---|---|
| ev1 | `ev1_kernel_exists` | 内核/实现存在 |
| ev2 | `ev2_cpu_reference` | CPU 参考实现与模块测试 |
| ev3 | `ev3_gpu_kernel` | GPU 内核实跑（探针） |
| ev4 | `ev4_same_job_inputs` | 同 job 输入真实照片探针 |
| ev5 | `ev5_main_preview` | **应用主预览接入** |
| ev6 | `ev6_vendor_parity` | 原厂同图对拍 |
| ev7 | `ev7_real_photo_accept` | 真实照片验收 |

七级相互独立：低级证据不能推出高级证据，`未验证` 永远不是通过。

## 2. 展示规则：把历史证据和当前可操作性分开

- `ev1…ev7` 是上表日期的记录，不是本机这张照片的输入状态。历史快照中 8 个 effect 有 ev5，ev6/ev7 为零；注册、构建、打开面板或换一张照片都不改变这些数字。
- 人像页的主表是 147 个不同的 `kind=effect` PF；PF109/110 是 PF108 的两个子控，不另算主效果。已登记但缺少后端、模型、素材或本图输入的行仍能找到，要显示具体原因，不能当作执行成功。
- 滑杆、款式、颜色与整妆都按当前图的 `RuntimeAvailability` 决定是否可操作。只有 `Available` 放行编辑；未取得当前图的探测结果、`PendingSameFrame`（初筛通过，等待刷新实测）、`MissingModel`、`MissingAsset`、`MissingInput`、`BackendUnavailable`、`StoredOnly` 都不修改参数。款式、色板和整套可以先检查一个不同的候选；仅在该候选所有相关行实际 `Available` 时才写成一次编辑。换图或文档已变时丢弃旧结果，普通强度拖动不会清空输入结论。
- 历史上有 ev5 不表示本图有牙齿或眼部遮罩；本图可编辑也不表示有原厂同图证据。工具提示与回执分别标出两者。

### 参数与整妆协议

| 参数 | 存储与行为 |
|---|---|
| `pc:90032` | 眼妆强度，0…1。使用 `EyeMaskType=10`，不属于唇彩。 |
| `pc:30210` | 新写入 0、1、2、3、4 整档。读旧文件时只将开区间 `(0,1)` 按 `round(4×v)` 换为 1…4，字面 `1.0` 始终是第 1 档；预览、导出与界面共用该读法。 |
| 面部几何 | 面部网格的 14 个原生百分比保存 −100…100。其余按实际生产器的归一化范围；例如 PF8025/8026 的瘦脸通道只接受 0…1，PF8033 也单向。只在界面绑定处换算百分比。 |
| `pc:108/109/110` | 主强度、亮白、去黄；编辑子控也只请求一次 PF108，不改写用户存储的父值。全部中性时不应生成 pass。 |
| `pc:<pf>:style`、`:color` | 真实扫描目录与引擎推荐色板中的 **1 基**序号。切换回推荐色应清除覆盖它的自定义色。 |
| `pc:<pf>:rgb` | `0xRRGGBB` 整数，可精确存入 `f32`；键不存在才表示“没有自定义色”。`0` 是不透明黑色。PF117 的显式选择优先于从 PF100 继承。 |
| `pc:111:catchlight_{style,amount,rgb}` | 原创真人虹膜光斑，不是新增的官方 PF。1 基真人款号、0…1 独立强度、可选 RGB24；强度为 0 可保留选材而不叠加。宠物款分库。本版此附属层只在 CPU/自动模式执行，指定未实现的 GPU 后端时不得留下半张已改的虹膜。 |
| 原创整妆 | `Makeups/Suit/<id>/suit.json` 的 `aurora.suit.recipe/1` 用 `apply` 写同一组真实单项键。一套一次撤销；不再叠加一个 PF90033 渲染通道。非本套妆的皮肤、脸型、牙齿编辑保留。缺选中材质、序号对应素材已变或存在未绑定的有效子妆，不应只套用一半。 |

## 3. rv24 三项结论

### PF30096 — 补入（sediao / Skin temperature）

- 能力表已有该行，但 `key`/`label`/`input_requirements` 三格是 `未验证` 占位。
- 代码侧有确定元数据：`crates/portrait/src/kernels/face_detail.rs` 的
  `META { pf: 30096, key: "sediao", label: "Skin temperature", inputs: ["skin mask","cold LUT","warm LUT"] }`；
  win-main 已有 UI 滑杆 `("pc:30096", "Skin temperature", "色调")`。
- 前端处理：`sediao` 的参数行改用“Skin temperature / 肤色温度”，并显示证据
  （ev1/ev2/ev4/ev5 有证据 → 主预览可用；ev6/ev7 未验证）。
- 与 win-main 的中文文案“色调”不一致，交由统筹在合并时统一；本会话不改主树。

### PF90021 — 修正 key（lipMakeupEnhance）

- 能力表：`key = lipMakeupEnhance`、`label = Lip makeup enhance`、`stage = cpu_stage`，
  `evidence_notes` 已记录 `CORRECTED 2026-11-24`：解码 `MakeupEnhance/effect.json` 得到
  `EffectName=LipStickLutMakeup`、`Paths=[0000.png, LipsLUT.png]`、负向 `LipDodgeLUT.png`；
  早先“肤色透亮”的读法作废，label 保持 “Lip makeup enhance”。
- 前端原本没有任何 90021 参数行（`portrait-algorithms.json` 未收录），因此“修”的落点是
  把它作为**已登记、无参数控件**的效果显示：键名 `lipMakeupEnhance`，标签“唇妆增强”，
  附七级证据（ev1/ev2/ev4 有，ev3/ev5/ev6/ev7 未验证），不标为可用。

### PF44297 — 评估（祛青筋 / 脚部）

- 能力表：`key = Gn.removeBlueVeins`、`label = Foot blue veins`、`stage = cpu_stage`；
  证据只有 ev1/ev2，ev3/ev4/ev5/ev6/ev7 全部 `未验证`。
- 与 PF30030（`removeBlueVeinsSlider`，Hand blue veins）是两个不同的效果：前者脚部，后者手部。
- 前端原来把 44297（`removeBlueVeins`）和 30030（`removeVein`）都写成“祛青筋”，无法区分部位。
- 处理：44297 → “Foot blue veins / 祛脚部青筋”，30030 → “Hand blue veins / 祛手部青筋”；
  两者都按证据显示“未接入主预览”，不得标为可用。

## 4. 同步与校验

- 重新同步能力表时：覆盖 `docs/frontend-contract/portrait-capability-registry-20261124.tsv`，
  同步更新 `evidence.rs` 里的 `REGISTRY_SHA256` 与 `REGISTRY_BYTES`。
- `cargo test -p portrait-app --lib evidence::` 会校验：每行 18 列、147 个 effect、8 个 ev5、
  ev6/ev7 为 0、PF30096/90021/44297 的结论、以及内嵌副本字节数。
- 历史口径差异：`90_work/lanes/backend-opt/rv24_bo2/PORTRAIT-CONTRACT-UPDATE.md` 记录的
  registry SHA `69ad4457…` / 72,620 字节、原前端副本 `f3986958…` / 76,393 字节均已过时。
  2026-11-25 同步后以磁盘实测 `d1b92c6f…` / 86,739 字节为准（capability-sync 2026-11-25）。
