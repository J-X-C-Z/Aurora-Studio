# 70 — 性能基线：进程内存与 Metal 连续子图微基准

> 状态：已实跑（单进程、单次）。所有数字来自本机实际命令。**这些是库级合成
> 微基准，不是产品预览管线，也不是 Windows 对比。** 未运行的项不写入“通过”。

## 1. 结论先行

- `crates/app/src/perf.rs` 已有 macOS 实现：当前值取 Mach `task_vm_info.phys_footprint`，
  峰值只取 OS 提供的 `ledger_phys_footprint_peak`（rev3）或
  `mach_task_basic_info.resident_size_max`，从不采样或合成峰值；Linux/其他平台仍返回 `None`。
- `crates/engine/examples/metal_chain_bench.rs` 在 Apple M2 上实跑了 3 个尺寸 × 2 个
  degree，每个 case 40 次暖迭代（n = 40，最近秩百分位，因此 P99 = 最大值）。
- 全部 6 个 case 的 GPU 输出与 CPU reference **逐通道最大字节差为 0**（B/G/R/A 全 0）。
- 每个 case 的每轮计数恒为 **1 command buffer / 2 compute encoder / 2 dispatch /
  1 wait / 1 readback**；`live_bytes` 回到 0。
- 这个两 kernel 链在 37×23 上 **CPU 更快**（GPU 约慢 7 倍），在 1920×1080 上 GPU 的
  P50 约为 CPU 的 1/23，在 3840×2160 上约为 CPU 的 1/39～1/52。比例只对本条的
  reference 实现成立，见 §7。

## 2. 环境

| 项 | 值 |
|---|---|
| 仓库 HEAD | `9bf908e53fd029c12f4f454546c61064f9ef9010`（worktree 有未提交改动：本 lane 的 `perf.rs` 修改与本文件/新 example，另有其他 lane 的在途改动） |
| 机器 | MacBook Air（Mac14,2），Apple M2，8 核 CPU（4P+4E），10 核 GPU |
| 统一内存 | 16 GB |
| macOS | 27.2.1（build 26B131） |
| 架构 / target | arm64 / `aarch64-apple-darwin` |
| Rust | rustc 1.98.1（48a229cea 2026-09-01），cargo 1.98.1 |
| Xcode / SDK | Xcode 27.2，MacOSX.sdk（Metal 运行时编译 MSL） |
| 构建 profile | `release`：`lto = "thin"`，`codegen-units = 4`（workspace `[profile.release]`） |
| 显示 | 内置 Liquid Retina 2560×1664；**本基准不涉及显示/合成** |
| 序列号 / UUID | 已脱敏，未记录 |

Metal 设备（同一进程 `MetalRuntime::probe_info()`）：

```
device=Apple M2 unified=true
recommendedMaxWorkingSetSize = 12,713,115,648 B (12,124 MiB)
maxBufferLength              =  9,534,832,640 B (9,093 MiB)
```

## 3. 命令（按实际执行顺序）

```sh
# 1) 编译检查
cargo check -p aurora-engine --example metal_chain_bench --locked --target aarch64-apple-darwin
cargo check -p portrait-app --lib --locked --target aarch64-apple-darwin

# 2) macOS 内存读取的单测（debug 测试进程）
cargo test -p portrait-app --lib --locked --target aarch64-apple-darwin \
    mach_task_info -- --nocapture

# 3) 基准 Release 构建
cargo build -p aurora-engine --release --locked --target aarch64-apple-darwin \
    --example metal_chain_bench

# 4) 单独运行（先确认 pgrep -x rustc 为空，避免编译争用污染测量）
AURORA_METAL_BENCH_JSON=/tmp/metal_chain_bench.json \
    target/aarch64-apple-darwin/release/examples/metal_chain_bench

# 等价的 cargo 形式（本机实际用上面的二进制直跑，避免 cargo 自身开销）：
# cargo run -p aurora-engine --release --locked --target aarch64-apple-darwin \
#     --example metal_chain_bench
```

运行前检查：`pgrep -x rustc` 返回 0 个进程。基准运行期间没有执行其他构建。
只跑了一个进程、一次；没有跨进程重复，也没有 CPU 绑核或关闭后台服务。

## 4. 定义

- **process warmup**：进程内第一次调用，包含两个 MSL library/pipeline 的运行时编译。
  单独报告，不计入各 case 的 “cold”。
- **cold first call（每个 case）**：该尺寸/degree 在已预热进程中的第一次调用。
  若该 size class 尚未分配，则包含首次 buffer 分配；不包含 MSL 编译
  （编译只在 `process_warmup` 里发生一次）。
- **warm sample**：一次完整库调用，即上传、编码、commit、唯一一次
  `waitUntilCompleted`、唯一一次 CPU readback 的总墙钟时间。
- **暖迭代次数**：40（`warm_iterations = 40`，加 1 次 cold，`runs_in_case = 41`）。
- **百分位**：对 40 个样本排序后取最近秩 `index = ceil(p/100 × n) − 1`。
  因此 n = 40 时 P99 就是最大值，P50 是第 20 小。
- **GPU/CPU 对比**：两者输入字节、LUT、degree 完全相同；CPU 是
  `skin_whiten_metal::cpu_reference`（整数 box blur + `masked_lut_blend`）。

## 5. 进程内存实现（TASK 1）

`crates/app/src/perf.rs` 保留 `working_set_bytes()` / `peak_working_set_bytes()`
签名与 `Option<u64>` 返回。macOS 分支：

| 函数 | 取值 | 回退 |
|---|---|---|
| `working_set_bytes` | `task_vm_info.phys_footprint`（rev1，偏移 144 B） | `mach_task_basic_info.resident_size` |
| `peak_working_set_bytes` | `task_vm_info.ledger_phys_footprint_peak`（rev3，偏移 168 B） | `mach_task_basic_info.resident_size_max` |

- 结构体与常量在模块内以 `repr(C)` 声明，字段顺序对照 SDK 的
  `<mach/task_info.h>`；编译期断言校验 `mach_task_basic_info` = 48 B、
  `task_vm_info` 前缀（到 ledger peak）= 176 B。
- `task_info` 的返回计数按 natural_t（4 B）单位检查：只有覆盖 `phys_footprint`
  （≥ 38 单位）或 ledger peak（≥ 44 单位）时才使用对应字段。
- 峰值是内核维护的高水位，不是定时采样；两种回退指标（footprint peak vs
  resident peak）语义不同，代码按原样返回，不互相冒充。
- Windows 分支未改动；Linux/其他平台两个函数仍返回 `None`。
- 单元测试输出（本次）：

```
perf memory: working=3293664 B peak=12173312 B
test perf::tests::mach_task_info_reports_a_footprint_and_peak ... ok
test result: ok. 1 passed; 0 failed; 0 ignored
```

即测试进程当时 phys_footprint ≈ 3.14 MiB，OS 峰值 ≈ 11.61 MiB。

## 6. 实测结果

### 6.1 Markdown 表（程序原样输出）

Device: Apple M2. Warm iterations per case: 40. Percentiles are nearest-rank on the sorted samples (P99 = max at n=40).

| size | degree | GPU cold ms | GPU P50 | GPU P95 | GPU P99 | CPU cold ms | CPU P50 | CPU P95 | CPU P99 | CPU/GPU P50 | max diff B/G/R/A | CB/enc/disp/wait/read per run | allocs | live B | pooled B | footprint before→after MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 37x23 | 0.5 | 0.508 | 0.307 | 0.342 | 0.966 | 0.074 | 0.042 | 0.043 | 0.060 | 0.14x | 0/0/0/0 | 1/2/2/1/1 | 0 | 0 | 4218880 | 13.5→13.7 |
| 37x23 | 1.0 | 0.302 | 0.289 | 0.346 | 0.350 | 0.048 | 0.042 | 0.043 | 0.054 | 0.15x | 0/0/0/0 | 1/2/2/1/1 | 0 | 0 | 4218880 | 13.7→13.7 |
| 1920x1080 | 0.5 | 10.723 | 4.868 | 5.486 | 6.419 | 103.093 | 113.302 | 141.078 | 168.063 | 23.28x | 0/0/0/0 | 1/2/2/1/1 | 4 | 0 | 31481856 | 23.6→81.2 |
| 1920x1080 | 1.0 | 6.976 | 4.695 | 4.814 | 6.456 | 109.811 | 108.978 | 128.163 | 132.731 | 23.21x | 0/0/0/0 | 1/2/2/1/1 | 0 | 0 | 31481856 | 81.2→81.2 |
| 3840x2160 | 0.5 | 25.785 | 13.244 | 18.625 | 19.730 | 500.342 | 518.039 | 1034.806 | 1131.208 | 39.12x | 0/0/0/0 | 1/2/2/1/1 | 3 | 0 | 132145152 | 112.8→335.6 |
| 3840x2160 | 1.0 | 15.830 | 7.969 | 14.259 | 15.923 | 364.813 | 417.852 | 472.893 | 486.276 | 52.43x | 0/0/0/0 | 1/2/2/1/1 | 0 | 0 | 132145152 | 335.6→335.6 |

说明：`allocs` 是该 case 内新建 `MTLBuffer` 对象数；`pooled B` 是基准结束时池内保留字节；
`CPU/GPU P50` = CPU P50 ÷ GPU P50（< 1 表示 CPU 更快）。表内数值对应下方 JSON。

### 6.2 JSON（stdout 逐行，已脱敏；文件经 `AURORA_METAL_BENCH_JSON` 同步写出）

```json
{"arch":"aarch64","degrees":[0.5,1.0],"device_name":"Apple M2","event":"environment","max_buffer_len_bytes":9534832640,"os":"macos","recommended_working_set_bytes":12713115648,"sizes":[[37,23],[1920,1080],[3840,2160]],"unified_memory":true,"warm_iterations":40}
{"buffer_allocations_delta":7,"event":"process_warmup","ms":247.825958,"pipelines_compiled_delta":2}
{"counters_delta_buffer_allocations":0,"counters_delta_buffer_reuses":287,"counters_delta_command_buffers":41,"counters_delta_compute_encoders":82,"counters_delta_dispatches":82,"counters_delta_readbacks":41,"counters_delta_waits":41,"cpu_cold_first_call_ms":0.073667,"cpu_over_gpu_p50_ratio":0.13666744916989448,"cpu_warm_max_ms":0.060166,"cpu_warm_mean_ms":0.042346975,"cpu_warm_min_ms":0.041458999999999996,"cpu_warm_p50_ms":0.041916999999999996,"cpu_warm_p95_ms":0.0425,"cpu_warm_p99_ms":0.060166,"degree":0.5,"event":"case","gpu_cold_first_call_ms":0.508125,"gpu_output_stable":true,"gpu_warm_max_ms":0.966292,"gpu_warm_mean_ms":0.3273052,"gpu_warm_min_ms":0.284833,"gpu_warm_p50_ms":0.306708,"gpu_warm_p95_ms":0.3415,"gpu_warm_p99_ms":0.966292,"height":23,"live_bytes_after":0,"max_abs_diff_a":0,"max_abs_diff_b":0,"max_abs_diff_g":0,"max_abs_diff_r":0,"peak_live_bytes_cumulative":4218880,"per_run_command_buffers":1,"per_run_compute_encoders":2,"per_run_dispatches":2,"per_run_readbacks":1,"per_run_waits":1,"pixels":851,"pooled_bytes_after":4218880,"process_footprint_after_bytes":14352984,"process_footprint_before_bytes":14156376,"process_footprint_delta_bytes":196608,"runs_in_case":41,"warm_iterations":40,"width":37}
{"counters_delta_buffer_allocations":0,"counters_delta_buffer_reuses":287,"counters_delta_command_buffers":41,"counters_delta_compute_encoders":82,"counters_delta_dispatches":82,"counters_delta_readbacks":41,"counters_delta_waits":41,"cpu_cold_first_call_ms":0.047916,"cpu_over_gpu_p50_ratio":0.1452717455344456,"cpu_warm_max_ms":0.053917,"cpu_warm_mean_ms":0.04251042500000001,"cpu_warm_min_ms":0.04125,"cpu_warm_p50_ms":0.041916999999999996,"cpu_warm_p95_ms":0.043125,"cpu_warm_p99_ms":0.053917,"degree":1.0,"event":"case","gpu_cold_first_call_ms":0.30166699999999996,"gpu_output_stable":true,"gpu_warm_max_ms":0.34991599999999995,"gpu_warm_mean_ms":0.2943259749999999,"gpu_warm_min_ms":0.27720799999999995,"gpu_warm_p50_ms":0.28854199999999997,"gpu_warm_p95_ms":0.346333,"gpu_warm_p99_ms":0.34991599999999995,"height":23,"live_bytes_after":0,"max_abs_diff_a":0,"max_abs_diff_b":0,"max_abs_diff_g":0,"max_abs_diff_r":0,"peak_live_bytes_cumulative":4218880,"per_run_command_buffers":1,"per_run_compute_encoders":2,"per_run_dispatches":2,"per_run_readbacks":1,"per_run_waits":1,"pixels":851,"pooled_bytes_after":4218880,"process_footprint_after_bytes":14352984,"process_footprint_before_bytes":14352984,"process_footprint_delta_bytes":0,"runs_in_case":41,"warm_iterations":40,"width":37}
{"counters_delta_buffer_allocations":4,"counters_delta_buffer_reuses":283,"counters_delta_command_buffers":41,"counters_delta_compute_encoders":82,"counters_delta_dispatches":82,"counters_delta_readbacks":41,"counters_delta_waits":41,"cpu_cold_first_call_ms":103.09275,"cpu_over_gpu_p50_ratio":23.277341140215714,"cpu_warm_max_ms":168.063084,"cpu_warm_mean_ms":116.64892300000001,"cpu_warm_min_ms":99.69324999999999,"cpu_warm_p50_ms":113.302458,"cpu_warm_p95_ms":141.07820800000002,"cpu_warm_p99_ms":168.063084,"degree":0.5,"event":"case","gpu_cold_first_call_ms":10.723167,"gpu_output_stable":true,"gpu_warm_max_ms":6.418959,"gpu_warm_mean_ms":4.928374075000001,"gpu_warm_min_ms":4.6927080000000005,"gpu_warm_p50_ms":4.867500000000001,"gpu_warm_p95_ms":5.4861249999999995,"gpu_warm_p99_ms":6.418959,"height":1080,"live_bytes_after":0,"max_abs_diff_a":0,"max_abs_diff_b":0,"max_abs_diff_g":0,"max_abs_diff_r":0,"peak_live_bytes_cumulative":31465472,"per_run_command_buffers":1,"per_run_compute_encoders":2,"per_run_dispatches":2,"per_run_readbacks":1,"per_run_waits":1,"pixels":2073600,"pooled_bytes_after":31481856,"process_footprint_after_bytes":85115552,"process_footprint_before_bytes":24773208,"process_footprint_delta_bytes":60342344,"runs_in_case":41,"warm_iterations":40,"width":1920}
{"counters_delta_buffer_allocations":0,"counters_delta_buffer_reuses":287,"counters_delta_command_buffers":41,"counters_delta_compute_encoders":82,"counters_delta_dispatches":82,"counters_delta_readbacks":41,"counters_delta_waits":41,"cpu_cold_first_call_ms":109.811167,"cpu_over_gpu_p50_ratio":23.212217364679567,"cpu_warm_max_ms":132.731084,"cpu_warm_mean_ms":112.71827389999999,"cpu_warm_min_ms":106.55208300000001,"cpu_warm_p50_ms":108.978459,"cpu_warm_p95_ms":128.16312499999998,"cpu_warm_p99_ms":132.731084,"degree":1.0,"event":"case","gpu_cold_first_call_ms":6.976125,"gpu_output_stable":true,"gpu_warm_max_ms":6.456458,"gpu_warm_mean_ms":4.340723949999999,"gpu_warm_min_ms":2.955833,"gpu_warm_p50_ms":4.694875000000001,"gpu_warm_p95_ms":4.814291,"gpu_warm_p99_ms":6.456458,"height":1080,"live_bytes_after":0,"max_abs_diff_a":0,"max_abs_diff_b":0,"max_abs_diff_g":0,"max_abs_diff_r":0,"peak_live_bytes_cumulative":31465472,"per_run_command_buffers":1,"per_run_compute_encoders":2,"per_run_dispatches":2,"per_run_readbacks":1,"per_run_waits":1,"pixels":2073600,"pooled_bytes_after":31481856,"process_footprint_after_bytes":85099168,"process_footprint_before_bytes":85131936,"process_footprint_delta_bytes":-32768,"runs_in_case":41,"warm_iterations":40,"width":1920}
{"counters_delta_buffer_allocations":3,"counters_delta_buffer_reuses":284,"counters_delta_command_buffers":41,"counters_delta_compute_encoders":82,"counters_delta_dispatches":82,"counters_delta_readbacks":41,"counters_delta_waits":41,"cpu_cold_first_call_ms":500.34175000000005,"cpu_over_gpu_p50_ratio":39.11613444204288,"cpu_warm_max_ms":1131.207917,"cpu_warm_mean_ms":639.906534425,"cpu_warm_min_ms":363.93975,"cpu_warm_p50_ms":518.0394160000001,"cpu_warm_p95_ms":1034.806167,"cpu_warm_p99_ms":1131.207917,"degree":0.5,"event":"case","gpu_cold_first_call_ms":25.785417000000002,"gpu_output_stable":true,"gpu_warm_max_ms":19.730332999999998,"gpu_warm_mean_ms":14.254783199999997,"gpu_warm_min_ms":9.712333,"gpu_warm_p50_ms":13.243625,"gpu_warm_p95_ms":18.625125,"gpu_warm_p99_ms":19.730332999999998,"height":2160,"live_bytes_after":0,"max_abs_diff_a":0,"max_abs_diff_b":0,"max_abs_diff_g":0,"max_abs_diff_r":0,"peak_live_bytes_cumulative":113254400,"per_run_command_buffers":1,"per_run_compute_encoders":2,"per_run_dispatches":2,"per_run_readbacks":1,"per_run_waits":1,"pixels":8294400,"pooled_bytes_after":132145152,"process_footprint_after_bytes":351880032,"process_footprint_before_bytes":118325944,"process_footprint_delta_bytes":233554088,"runs_in_case":41,"warm_iterations":40,"width":3840}
{"counters_delta_buffer_allocations":0,"counters_delta_buffer_reuses":287,"counters_delta_command_buffers":41,"counters_delta_compute_encoders":82,"counters_delta_dispatches":82,"counters_delta_readbacks":41,"counters_delta_waits":41,"cpu_cold_first_call_ms":364.813333,"cpu_over_gpu_p50_ratio":52.433023810270726,"cpu_warm_max_ms":486.275916,"cpu_warm_mean_ms":418.47140509999997,"cpu_warm_min_ms":368.40662499999996,"cpu_warm_p50_ms":417.851875,"cpu_warm_p95_ms":472.89300000000003,"cpu_warm_p99_ms":486.275916,"degree":1.0,"event":"case","gpu_cold_first_call_ms":15.829541999999998,"gpu_output_stable":true,"gpu_warm_max_ms":15.923,"gpu_warm_mean_ms":9.124492749999998,"gpu_warm_min_ms":7.6995,"gpu_warm_p50_ms":7.969250000000001,"gpu_warm_p95_ms":14.259125,"gpu_warm_p99_ms":15.923,"height":2160,"live_bytes_after":0,"max_abs_diff_a":0,"max_abs_diff_b":0,"max_abs_diff_g":0,"max_abs_diff_r":0,"peak_live_bytes_cumulative":113254400,"per_run_command_buffers":1,"per_run_compute_encoders":2,"per_run_dispatches":2,"per_run_readbacks":1,"per_run_waits":1,"pixels":8294400,"pooled_bytes_after":132145152,"process_footprint_after_bytes":351945568,"process_footprint_before_bytes":351929184,"process_footprint_delta_bytes":16384,"runs_in_case":41,"warm_iterations":40,"width":3840}
```

### 6.3 读取要点

- **process warmup = 247.83 ms**，编译 2 个 pipeline。这是进程内第一次 Metal 调用的
  一次性成本；预编译 `.metallib` 尚未实现（见 `11-metal-runtime.md` §4）。
- **配额（batching 契约）**：6 个 case 共 41 runs/ case，合计
  41 CB、82 encoder、82 dispatch、41 wait、41 readback，`per_run_*` 全部为 1/2/2/1/1。
  连续子图确实只用一个 command buffer、一次等待、一次读回。
- **buffer 池**：`live_bytes_after = 0`；`pooled_bytes_after` 随 size class 增长，
  3840×2160 后为 132,145,152 B（126 MiB），远低于池的 256 MiB 上限。
  1080p 与 4K 的首个 degree case 分别新建 4 / 3 个 `MTLBuffer`，第二个 degree
  全部复用（alloc = 0，reuse = 287）。
  “allocated bytes” 由 `peak_live_bytes_cumulative`（字节，进程启动以来峰值）
  与 `counters_delta_buffer_allocations`（新建对象数）表示；运行时的计数器没有
  “累计分配字节”这一项，因此不虚构该字段。
- **画质**：GPU cold 输出与 CPU reference 的 B/G/R/A 最大绝对字节差全部为 0；
  `gpu_output_stable = true` 表示 40 次暖迭代输出逐字节等于 cold 输出。
- **进程 footprint**：case 前后值包含基准自身输入/回读 Vec 和分配器保留，不是
  Metal 池的独立账本。1080p 首次 23.6→81.2 MiB、4K 首次 112.8→335.6 MiB，
  随后同尺寸 case 基本稳定（81.2→81.2、335.6→335.6 MiB）。
- **P99 = max**（n = 40）。4K degree 0.5 的 P99 19.73 ms 与 P95 18.63 ms 接近，
  但 P50 13.24 ms：首次触碰 4K 池与新页导致长尾，见 §7。

## 7. 这些数字能证明什么、不能证明什么

### 能证明

1. 该 2-kernel 链在 Apple M2 上真实执行，且在这个合成输入上与 CPU reference
   逐字节一致（B/G/R/A 最大差 0），两个尺寸档位覆盖到 4K。
2. 运行时契约成立：一个 command buffer 内 2 个 encoder/dispatch，只有 1 次
   `waitUntilCompleted` 和 1 次 readback；每轮 `live_bytes` 归零。
3. 就这条链本身，GPU 的相对收益尺寸相关：37×23 上 dispatch/提交/读回固定开销
   压倒计算，CPU 明显更快；1080p 起 GPU 明显更快。
4. macOS 进程内存接口能返回 OS 维护的当前 footprint 和高水位。

### 不能证明

1. **不是产品预览管线。** 没有 `ImageEngine`、`mock::process`、RAW 解码、
   portrait stage、显示合成或 GPUI。不能据此宣称任何 UI 帧率、拖动延迟或
   “4K 调色 P95 ≤ 33 ms”。
2. **不是 Windows 对比。** 仓库规则要求性能声明基于 Windows 原生 Release；
   本文件没有 Windows 数据，任何跨平台比例都未验证。
3. **不是优化后的 CPU 基线。** CPU reference 是标量整数 box blur +
   `masked_lut_blend` 的教育型 twin；GPU/CPU 比例描述的是“相对这个 oracle”，
   不是相对产品里可能的 SIMD/多线程 CPU 路径。
4. **不覆盖热状态与长时间稳定性。** 单进程、短时、无持续负载、无能耗/热降频
   观测；也没有跨进程重复来估计运行间方差。
5. **4K 两个 degree 不可横向比较快慢。** degree 0.5 是 4K 的第一次接触：
   新建 3 个 buffer、首次触碰约 233 MB 新页，P50/P95 为 13.24/18.63 ms；
   紧随其后的 degree 1.0 复用同一批热 buffer/页，P50/P95 为 7.97/14.26 ms。
   算法工作量相同，差异来自首次触碰、页缓存与调度噪声；不能解读为
   “degree 1.0 更快”。
6. **计数是进程全局累计的。** `peak_live_bytes_cumulative` 是运行时自进程启动
   以来的峰值（4K 时 113,254,400 B），不是单 case 峰值；`counters_delta_*`
   才是 case 内增量。

## 8. 产品级基准还缺什么

- **真实资产**：真实 NEF / 其他已支持 RAW、真实人像 PF 输入与合法 LUT；
  本文件的输入是 xorshift 合成图与 `synthetic_test_lut`。
- **产品路径接入**：`ImageEngine` / 预览管线里的 Metal 子图（当前只在库级
  测试与 example 中调用），以及 GPUI 显示互操作后的端到端计时。
- **UI 指标**：事件线程帧预算、拖动 P50/P95、首张可操作时间；需要在
  `AURORA_PERF_REPLAY` 类 instrumentation 下单独采集，不能用本文件推断。
- **同设备同画质基线**：产品 CPU 路径 vs Metal 路径的真实对比，以及
  Windows 原生 Release 的对照数据。
- **长时与压力**：多帧 ring、缓存命中/失效、内存压力、睡眠唤醒、显示变化、
  持续 10 分钟以上的热降频与泄漏观察。
- **预编译 metallib** 后的冷启动对比（本次 warmup 的 247.83 ms 含运行时 MSL
  编译，不能代表未来的冷启动）。

## 9. 未验证清单

- Linux/其他平台的 `working_set_bytes` / `peak_working_set_bytes` 仍为 `None`
  这一行为经由 `cfg` 保证，但未在本机交叉编译验证。
- 1080p/4K 的 CPU 数值波动较大（4K degree 0.5 的 max 1.13 s vs min 0.364 s），
  未做绑核或更多重复；不应把它当作稳定的 CPU 基线。
- 未在 Intel Mac、Windows、CI runner 上运行本基准。
- 未验证 Metal 与 GPUI 同设备/同 registryID 的显示路径。

---

文件：`crates/app/src/perf.rs`（macOS `task_info`）、
`crates/engine/examples/metal_chain_bench.rs`（本基准）、本文件。
本次未提交、未推送，未改许可证，未新增依赖。
