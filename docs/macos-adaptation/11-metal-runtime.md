# M2 — Metal runtime 实施记录与真机证据

> 本文件记录实际代码与命令结果。没有运行的项目一律标“未验证”。

## 1. 新增模块

| 文件 | 职责 |
|---|---|
| `crates/engine/src/metal_runtime.rs` | 公开门面：`MetalRuntime::{shared, probe, probe_info, begin, counters}`、`MetalWorkspace`、`MetalFrame`、`Grid`、`Binding`、`PipelineKey` |
| `crates/engine/src/metal/device.rs` | 进程内唯一 `MTLDevice` + `MTLCommandQueue`（`OnceLock`） |
| `crates/engine/src/metal/library.rs` | `LibraryCache`（key = MSL source hash + fast_math=false）+ `PipelineCache`（library hash + entry）；读取 `threadExecutionWidth` / `maxTotalThreadsPerThreadgroup` |
| `crates/engine/src/metal/pool.rs` | `BufferPool`，size class 复用，`StorageModeShared`，字节计数，`live_bytes` 归零 |
| `crates/engine/src/metal/encode.rs` | `CommandBatch`：一个 command buffer，多个 compute encoder，`commit` 不等待 |
| `crates/engine/src/metal/residency.rs` | `MetalFrame`、generation；`read_into` 是唯一 `waitUntilCompleted` + 唯一 CPU 拷贝 |
| `crates/engine/native/metal/o3b_masked_lut_whiten.metal` | 与 `crates/portrait/native/metal/` 同字节副本（`cmp` 验证） |
| `crates/engine/native/metal/skin_whiten_prep.metal` | 3×3 整数 box blur |
| `crates/engine/src/mock/skin_whiten_metal.rs` | 两 kernel 连续链 + CPU twin |
| `crates/engine/tests/metal_runtime_macos.rs` | 真机探测与 CPU 对拍 |

## 2. 依赖

`crates/engine/Cargo.toml` 增加 `[target.'cfg(target_os = "macos")'.dependencies]`：
`objc2 = "0.6"`、`objc2-metal = "0.3"`、`objc2-foundation = "0.3"`、`block2 = "0.6"`。
`Cargo.lock` +5 行、无新包（版本早已由 `gpui-pre-apple 0.3.11` 解析存在）。

## 3. 真机结果（Apple M2，macOS 27.2.1）

命令：`cargo test -p aurora-engine --locked --target aarch64-apple-darwin metal -- --test-threads=1`

```
test macos::metal_probe_reports_a_real_device ... ok
test macos::metal_two_kernel_chain_matches_cpu_reference ... ok
test result: ok. 2 passed; 0 failed; 0 ignored
```

`--nocapture` 实测：

```
metal probe: name="Apple M2" unified=true recommended_working_set=12713115648 max_buffer_len=9534832640
metal pipelines: blur tew=32 max_total=1024 groups=[2,1,1] threads=[32,32,1]
                 whiten tew=32 max_total=1024 groups=[2,1,1] threads=[32,32,1]
metal parity: 37x23 pixels=851 degree=0.75 per-channel max |gpu-cpu| = B:0 G:0 R:0 A:0
metal counters delta: command_buffers=1 encoders=2 dispatches=2 waits=1 readbacks=1
                      allocations=7 reuses=0 live_bytes_after=0 pooled_bytes_after=4218880
```

- 断言容差：RGB ≤ 1 LSB（fast math 关闭、`#pragma clang fp contract(off)`），alpha 必须完全相等；
  实测四个通道最大误差均为 **0**。
- “连续子图”证据：**1 个 command buffer、2 个 encoder/dispatch、1 次等待、1 次读回**。
- `live_bytes_after = 0`，池内保留 4,218,880 B（6×4 KiB + 4 MiB LUT）。

## 4. 明确未完成

- **未接入产品渲染链**（`mock/process.rs` / `ImageEngine` 未调用）；当前是库级 + 测试路径。
- 未做 in-flight ring / `addCompletedHandler`，多帧持续复用时池要等 `read_into` 才回收。
- MSL 仍运行时编译；预编译 `.metallib` 属打包后续。
- 只覆盖 `MTLBuffer`/Shared；Private 中间纹理与 blit staging 未实现。
- 与 `mock/gpu_runtime` 的 `Failure`/账本统一未做（该模块当前仅 Windows）。
- 旧 `crates/portrait/src/metal.rs` host 未迁移（D2），其 device over-retain 仍需修。
