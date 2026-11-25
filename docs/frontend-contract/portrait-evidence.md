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

## 2. 展示规则（前端硬约束）

1. 只有 **ev5 有证据** 的效果才允许在界面呈现为“可用”。当前 147 个 effect 中只有 8 个
   （PF8068、PF8069、PF8070、PF21015、PF30096、PF90004、PF90005、PF90007）。
2. 有 ev5 也只标“部分”：ev6/ev7 目前 **0 项有证据**，必须显式写“原厂对拍/真实照片未验证”。
3. 有登记但没有 ev5 的效果写“未接入主预览，仅保存”，可用状态为中性/不可用，不得显示成功。
4. 能力表没有对应 PF 的参数：写“后端能力未登记”，不推断任何能力；参数仍可保存，但不承诺预览或导出。
5. 每条参数行悬停显示七级逐项状态与证据路径；面板标题给出本组汇总。
6. 已登记但前端没有参数控件的效果，单独列在同名分组的“已登记但无参数控件的效果”面板，
   不得因为“没有滑块”而从界面消失。

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
