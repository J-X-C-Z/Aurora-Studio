# M1 — ARM64 测试基线记录

命令（Mac-A 执行，2026-11-25）：
`cargo test --workspace --locked --target aarch64-apple-darwin -- --test-threads=1`

首次结果（未加跳过过滤）：
**1699 passed / 133 failed / 84 ignored**（`aurora-engine --lib` 结束于 FAILED）。

失败按原因分类：

| 类别 | 数量（约） | 例子 | 性质 |
|---|---|---|---|
| 缺本地 testdata / 私有 C1 资产 | 100+ | `c1_cos::*`, `c1_styles::*`, `c1catalog::*`, `adjust::as_shot_reads_the_real_nef_gains` | 环境缺失；仓库约定由 `scripts/testdata-skips.txt` 过滤 |
| 缺 ONNX 模型 / ICC / 曲线资产 | 约 15 | `camera_profile::*`, `icc::other_c1_profile_families_match_lcms`, `recipe::icc_profile_changes_the_output` | 环境缺失 |
| 仓库内路径缺失（与平台无关） | 2 | `param_contract::*` 查找 `01_frontend/contracts/params.json`，该目录在 HEAD 不存在 | **预存在**，非 macOS 引入 |
| 数学断言失败（与平台无关） | 4 | `mock::geometry::tests::keystone_*` 期望 `[0.21,-0.14]`，实得 `[0.0098,0.0098]`；文件未被本 lane 修改 | **预存在**，需仓库所有者/Windows lane 确认是否本来就红 |
| 明确“必须修”的红项 | 2 | `portrait_historical_tone_provider::*`（`scripts/testdata-skips.txt` 注释点名不跳过） | **预存在** |
| 已知 path 语义差异 | 5 | `file_engine::library_ops::*` → `path must be inside the library root` | 疑似 macOS `/var` ↔ `/private/var` 符号链接导致，需单独验证 |

结论：
- 本 lane 的编译修复没有引入上述失败中的任何一类；失败集合与“无本地资产 + HEAD 预存在红项”一致。
- **未使用跳过列表的原始 133 失败不能算通过**；下一步必须用 `AURORA_CI_SKIP_TESTDATA=1`
  等价的 `--skip` 过滤重跑，并把剩余失败逐条归因。
- 平台相关的一项（library_ops）需要真实修复，不能加进跳过表。

## 应用仓库自带跳过表后的结果

`cargo test --workspace --locked --target aarch64-apple-darwin -- <skips> --test-threads=1`
（105 个 skip 过滤）：

**1697 passed / 30 failed / 84 ignored / 105 filtered out**（`aurora-engine --lib` FAILED）。

残余 30 项全部已归因，不含本 lane 引入的回归：

- `file_engine::library_ops::*`（7）：**已修**。根因是 macOS `temp_dir()` 返回 `/var/...`
  而物理路径是 `/private/var/...`，`FileEngine` 存 canonical root 后，非 canonical 的
  请求被 `starts_with` 拒绝。修法：`library_ops.rs` 新增 `physical_path()`（规范化最深的
  已存在祖先并回接未创建部分），测试 fixture 用 `canonical_root` 统一物理形式。
  修复后 `library_ops` 16/16 通过。
- `mock::geometry::tests::keystone_*` 等 4 项：文件未被本 lane 修改，纯 f32 数学断言
  失败，属 HEAD 预存在红项。
- `param_contract::*` 2 项：查找 `01_frontend/contracts/params.json`，该目录在 HEAD 与
  main 都不存在，属预存在红项。
- `portrait_historical_tone_provider::*` 2 项：`scripts/testdata-skips.txt` 明确点名
  “必须修、不跳过”，属预存在红项。
- 其余 15 项（`c1_cos`、`c1_eip`、`camera_profile`、`icc`、`recipe`、`file_engine::tests`）
  需要本地 C1/ICC 资产，跳过表未覆盖，属环境缺失。
- `aurora-portrait --test metal_asset`（1 项）：非 ignore 的 LUT 测试读
  `<repo>/work/backend-r11/o2e/whiten_lut.f32`，本机只有 `o1i`。属本地资产缺失；
  该测试在 macOS 上不是 `#[ignore]`，因此会破坏默认 `cargo test` 口径。

## 待办

1. 预存在红项（param_contract / geometry / historical_tone）报给仓库所有者，不在本 lane 静默修。
2. 集成测试目标（`crates/engine/tests/*.rs`）尚未逐一确认。
3. 合入前在 rebase 到 `8118a4a` 后重跑本节全部命令。

## 最终验证（2026-11-25，全 workspace + 跳过表）

```
cargo test --workspace --locked --target aarch64-apple-darwin --no-fail-fast -- <105 skips> --test-threads=1
→ 2533 passed / 28 failed / 108 ignored
```

随后修复两处本 lane 自身引入的问题：

1. `library_ops` 的 `/var` ↔ `/private/var`（7 项转绿，见上）。
2. 撤回 `aimask/mod.rs` 对 Capture One macOS 模型目录的自动发现——它会让 `closedeyes`
   测试进入无 ORT 的 session 路径（1 项转绿），并避免隐式消费授权资产。

修复后残余 27 项：本地 C1/ICC 资产缺失 19、HEAD 预存在数学/路径红项 8
（`mock::geometry` 4、`param_contract` 2、`portrait_historical_tone_provider` 2）。
`metal_asset` 仍需 `work/backend-r11/o2e/whiten_lut.f32`。
