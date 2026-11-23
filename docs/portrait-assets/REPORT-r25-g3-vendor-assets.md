# R25 G3 vendor asset acquisition

状态：`MODELS_PARTIALLY_ACQUIRED__NO_SAME_JOB_OUTPUTS__NO_SAFE_G2_WIRING`

基线 `eececfd`，前置 HEAD `8c10fd3`（G2b）。本轮只写当前 worktree/topic，未写主树，未运行 cargo/GPU，未扫描 `target/`。

## 结论

找到并核验 8 个真实私有模型文件，共 41,588,323 B；没有找到可由原厂资源端点直接绑定的 URL。应用内没有恢复出可用的资源 URL 或私网下载端点，能验收的来源是已落地 payload 文件本身。

`PF21026` 与 `PF90088` 共用 `1655581403`、`3378116080`、`2127140479`、`4083080210`、`195031232` 模型族；`PF100007` 使用 `806921480`、`1940369920`、`3157078880`。这些私有 FlatBuffer 已按固定 XOR 包装解码，5 个 HairFly 候选通过私有结构校验、官方 MNN verifier 和最小 CPU forward；`PF100007` 三网也有真实候选 forward 证据。

但这些 forward 使用确定性合成输入或候选裁片，不是原厂同一 render job 的输入输出；不能登记成 vendor fixture，也不能接入 G2 producer。`PF90002`/`PF90010` 仍没有模型 ID、真实 LUT/纹理对象或 sampler 到 payload 的绑定。因此本轮可执行 PF 为 **0**，新增 G2 接线为 **0**。

| PF | acquisition | 可执行 | 阻塞 |
|---|---:|---:|---|
| `90002` | 0 个模型/LUT | 否 | D3/D5 平面没有对象身份或同 job 输出 |
| `90010` | 0 个模型/LUT | 否 | `b_`/`bW` 没有 payload 绑定 |
| `21026` | 5 个模型族候选 | 否 | 模型选择、mask shape 冲突、repair output 3/4 冲突 |
| `90088` | 5 个模型 + selector 候选 | 否 | selector 规则、raw mask、person/background area 未捕获 |
| `100007` | 3 个 SmileV1 模型 | 否 | 输出不是 dx/dy；material UV 与唇/口 mask 缺失 |

## 载荷与解码

真实文件位于 `extract/pixcook/<model-id>`。首字节是加密封装，不是标准 MNN/ONNX magic；原始 16 B 前缀和完整 SHA-256 已写入机器清单。

已核验解码步骤：

1. 用 `native_header_xor.key`（256 B，SHA-256 `ff21780d...151d`）异或前 256 B。
2. 从偏移 256 到最后一个完整 4 B，用 `native_body_xor.key`（256 B，SHA-256 `a47db4d5...c302`）循环异或。
3. 末尾 1/2/3 B 按 `78`、`63 78`、`63 78 6a` 处理。
4. 跳过 77 B 私有 wrapper，解析 FlatBuffer 根节点、算子表和 tensor 表。

关键模型身份：

```text
1655581403  selector        2,410,861 B  6a025f05c26aca02def0d81e2c443d0e3dc01e61f2af00a525aca6d4ca60d14d
3378116080  5-head 512x768  7,318,717 B  7d3f2aec4c4f815d758892f7c5a8ad66fff336c5fe9272a05613e0e5aa52a68b
2127140479  5-head 512x768  7,318,717 B  0e027141a594e4a20e01b3fc61cd02c9b1e57c3160488b294a9746dc659c12c6
4083080210  5-head 384x512  7,208,025 B  8140003eed6e5606742be72b64ef04ad240a6252fc1ecef71bc556ff6ae668fd
195031232  5-head 384x512  7,224,497 B  054ad6613315b13688eeb7b159d8c86a0e54969cf3a9c8d15fd350396367f998
806921480   SmileV1        2,662,989 B  87342bf07c1b32bb872277c2d048e3ced15f4416f800f4c1ee2b3e9cb54e7689
1940369920  SmileV1        2,662,989 B  ed8b7587d71f4c18d37a509829d20b4e9140cb77b4618ee8eb9ffad5fc0c0faf
3157078880  SmileV1        5,024,629 B  eac03538db8e2c93e5db9b33157583064943a99a4afacdf393114d48f10c395a
```

`3378116080` 的五端口为 `158607435 [1,1,512,768]`、`2405607714 [1,1,512,768]`、`3434589800 [1,3,512,768]`、`2977284499 [1,3,512,768]`、`2892147817 [1,3,512,768]`。运行时候选还观察到 `2405607714` 为三通道，与图上单通道冲突，故不能按静态 shape 硬接。

`PF90088` 的 raw mask 只到 `min(mask_a,mask_b)` 语义，后续 feather/ROI/warp/u8 和模型族选择未恢复；person/background area 不在这五个模型输出里。

`PF100007` 的 `3204548346` 是 `[1,1,768,1024]` sigmoid 响应，`780929437` 是三通道 RGB-like 重建，二者都不能冒充 signed `dx/dy`；`3157078880` 已实证为 `ADD(conv_branch, primary_input)`。

## 接入裁决

G2 producer 已有 `beard_inputs`、`stray_hair_inputs`、`smile_inputs_with_assets` 等 typed 入口，但本轮没有真实 same-job tensor、UV 或 mask 可传入。用合成 forward、单位 UV、全/零 mask 或重复模型 SHA 会把“模型文件存在”冒充“产品可执行”，因此保持具名拒绝，不新增接线。

机器可读结果见 `docs/portrait-assets/acquisition-r25-g3-vendor-assets.json`。下一可执行 PF 必须来自原厂同 job 导出或程序自身文档化 capture：携带 job ticket、源帧 SHA、模型 ID、输入/输出 tensor bytes、shape、dtype 和 SHA，并由 G2 provider 一次校验后进入现有 consumer。
