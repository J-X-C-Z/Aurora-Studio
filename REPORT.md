# A1g PF8033 CPU Forward -> Rust Smile Stage

## A1g 结论

可执行 PF 只有 `PF8033`。Rust 现可发现 A1f CPU-forward 回执与三只模型、校验
源图/模型/网络输入输出/`780929437`/dx/dy/mask 的 SHA，把真实外部平面接入现有
`FaceGeometry::Smile`，并强制走 CPU `pf8033_dense_warp`。同一源图、degree 和
artifact 下会复核 `changed_pixels=762262` 与 RGBA 输出 SHA；任一文件缺失、长度错误
或 SHA 不符均按具名 `Refusal` 拒绝。该路径仍是执行过的候选链，不是 vendor parity。

模型 SHA（raw / decoded）：

- `806921480`: `87342bf07c1b32bb872277c2d048e3ced15f4416f800f4c1ee2b3e9cb54e7689` / `5e0405127a0f53e2b893d6ceaf943ae6c027b3d8be57d7e5b160aabd258e3b41`
- `1940369920`: `ed8b7587d71f4c18d37a509829d20b4e9140cb77b4618ee8eb9ffad5fc0c0faf` / `fa362953541788a6497531aa63b26761b2db15613f6a06567fb7d3052a376764`
- `3157078880`: `eac03538db8e2c93e5db9b33157583064943a99a4afacdf393114d48f10c395a` / `39052c665195a87bd551423d666ff164773b2d13f5f8b23da27113b9dac65719`

运行前设置 `AURORA_PF8033_ARTIFACTS`；运行
`python tools/pf8033_cpu_forward.py` 会写入 `pf8033-780929437.f32le`、`pf8033-dx.f32le`、
`pf8033-dy.f32le`、`pf8033-mask.u8`。回执可用 `AURORA_PF8033_FORWARD_RECEIPT` 覆盖；
模型目录可用 `AURORA_PF8033_MODEL_ROOT` 覆盖。阻塞项仍是同 job vendor 调度、
`780929437` 原厂通道语义和 vendor golden；不得据此宣称原厂 parity。
按本轮约束未运行 `cargo`/GPU；验证等级为 `git apply --check` 与静态契约检查。

## A1f PF8033 三网 CPU Forward

## 结论

相对 `eececfda32bcf860385e3f358afb8a31d9de303e`，三只已解码模型现已实际完成
私有 FlatBuffer 结构校验、标准 MNN 候选重编码、官方 `VerifyNetBuffer` 和 MNN CPU
`runSession`。三网均为 `code=0`，输出均为有限 `float32`；真实仓库图像
`assets/portrait/skin-adjust/warm.png` 已转为 `1024x768` 作为 RGB 输入。

本次不是原厂同 job 重放，也没有 vendor golden。辅助两平面由首网真实输出继续驱动
第二网；第三网输入由第二网 `780929437` 缩放得到。该调度是显式候选，不是已证的
vendor work-item 绑定。

## Forward 结果

| model | output tensor | runtime shape | min | max |
|---|---|---|---:|---:|
| `806921480` | `3204548346` | `[1,1,768,1024]` | `2.8025343e-6` | `0.21066721` |
| `806921480` | `780929437` | `[1,3,768,1024]` | `-0.41030294` | `1.4276646` |
| `1940369920` | `3204548346` | `[1,1,768,1024]` | `2.9631431e-4` | `0.44152126` |
| `1940369920` | `780929437` | `[1,3,768,1024]` | `-0.39281163` | `1.1425303` |
| `3157078880` | `780929437` | `[1,3,512,512]` | `-0.21758376` | `1.0510557` |

`3157078880` 的 5 个 `private_80` 均消费一个数据 tensor 与一个四整数 shape
Blob，且没有参数表。候选语义定为动态 `BroadcastTo`；静态证据支持该输入形态，
但没有原厂算子等价证明。

## Receipt 与 Warp

`assets/portrait/smile/pf8033-cpu-forward.json` 保存输入、三网输出、shape、范围、
dtype、SHA-256、MNN 模块、模型容器与 decoded SHA。job ticket 为
`3621424029303566009`。

最终 `780929437` 的通道 `0/1/2` 作为候选 `dx/dy/mask`；每通道中位数居中，
`dx/dy` 按 P99.5 绝对值归一后乘 16 像素，mask 归一到 `u8`，degree 为 `0.35`。
该 host split 已写 receipt 并等价执行
`sx=x+degree*mask/255*dx`、`sy` 同理的双线性 PF8033 warp，改变了
`762262` 个像素。

预览：`assets/portrait/smile/pf8033-warp-preview.png`

## 证据边界

已通过：容器/密钥解码、decoded SHA、完整私有 wire 解析、标准 MNN 候选、三网
官方 verifier、真实图像 CPU forward、输出 receipt、PF8033 warp。

未证明：同 job vendor 输入的语义、三网真实调度、`private_80` 原厂等价、
`780929437 -> dx/dy/mask` 的原厂通道绑定及 vendor golden 数值一致性。manifest
将这些状态明确写成 candidate/unproven，不把预览冒充原厂结果。

复现：

```powershell
python tools/pf8033_cpu_forward.py
```

脚本只在 lane worktree 的临时目录生成 MNN 候选，完成后删除；vendor 模型、密钥、
候选 MNN 和完整 tensor 均不进入补丁。
