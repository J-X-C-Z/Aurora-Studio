# Windows GPU 推理配置

本配置只加速现有 ONNX 主体/皮肤蒙版模型。

## 平台状态：Metal 仅为 roadmap

- Metal（macOS GPU）**仅为路线图**：没有 Metal 后端，也没有任何真机执行记录、性能数据或像素对拍；
  界面与文档都不得把它写成已支持或已验证。
- 当前实际可用的 GPU 路径只有 Windows 的 CUDA / TensorRT / DirectML（见下文实测），
  其余调色、细节与人像阶段仍由 CPU 执行。
- 记录口径：没有 Windows 原生 Release 实测、没有 provider profile 节点归属、没有与 CPU 的
  像素差异时，一律写“未验证”，不得由 DLL 存在、`GetAvailableProviders` 结果或安装成功推断通过。
普通曝光、颜色和细节处理仍使用 CPU；DirectX 11 的 GPU 显示不能作为推理加速证据。
本轮另外完成了 CPU 滤镜、重复解码和缩略图队列优化，见
[整体性能记录](performance-2026-11-12.md)。

## 本机已配置

RTX 4070 Ti 12 GB、驱动 610.88、现有 CUDA Toolkit 13.4。
项目专用运行库为 ONNX Runtime GPU 1.31.1、cuDNN 9.28.1.4、TensorRT 10.16.1.11，
CUDA 与 TensorRT 均使用 CUDA 13 发行包。运行库位于本机独立目录，不修改全局驱动、
系统 PATH、注册表或电源设置；前端布局、动画、快捷键和缩略图更新节奏保持原逻辑。

当前 native C API 的 GPU 执行已通过 ONNX profile 中的节点 provider 验证。
测试使用现有本地模型，模型文件、运行库、序列化 engine 和测试照片均不进入源码包。

## 运行

复制并按本机安装位置填写仓库根目录 `gpu-runtime.local.json`：

```json
{
  "ort_library": "D:/AuroraRuntime/ort/onnxruntime.dll",
  "dll_dirs": [
    "D:/AuroraRuntime/ort",
    "D:/AuroraRuntime/cudnn/bin",
    "D:/AuroraRuntime/tensorrt_libs",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/x64"
  ],
  "provider": "auto",
  "device": 0,
  "gpu_memory_mb": 2048,
  "threads": 4,
  "cache_dir": "D:/AuroraCache"
}
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/launch-gpu.ps1
```

启动器默认使用根目录 `aurora-studio.exe`，否则使用 `target/release/aurora-studio.exe`。
`-Executable` 可指定构建位置；其余 `AppArgs` 原样传给应用。设置只写入子进程环境，
不修改父会话。修改配置后重新启动应用。

`auto` 仅在相同指纹的 TensorRT engine 全部存在、且已成功完成实际推理预热时使用
TensorRT；否则优先 CUDA，再回退 CPU。显式 `tensorrt` 模式用于离线构建和预热，
首次构建可能超过一分钟。初始化失败会尝试后续 provider，实际节点归属以 profile 为准。

## 精度、资源与缓存

- CUDA `use_tf32=0`，TensorRT 关闭 FP16/INT8，启动前设置 `NVIDIA_TF32_OVERRIDE=0`。
  使用 FP32；不以减小模型输入、蒙版尺寸或预览分辨率获取性能。
- CUDA 使用 HEURISTIC 卷积算法选择、按需扩展 arena 和默认 stream 拷贝。
  `gpu_memory_mb` 是每个 CUDA session 的 arena 限额，**不是进程总显存上限**。
  TensorRT builder workspace 为该值的一半；其余权重与上下文另有显存开销。
- GPU 会话关闭 intra/inter-op 线程空转，避免空闲工作线程占用界面需要的 CPU。
  保留顺序执行与既有请求合并语义，不更改前端事件与动画。
- 当前输入/输出使用 CPU tensor，地址随调用变化，因此不启用需要稳定 device IO
  binding 的 CUDA Graph。后续改动应先实现对应的内存生命周期再单独验证。
- TensorRT 复用 engine 与 timing cache，开启上下文内存共享、顺序构建，
  不强行复用不匹配的 timing cache。命名空间包含 GPU/驱动、ORT、TensorRT/cuDNN、
  模型路径/大小/修改时间、设备与显存参数；改变构建精度或 provider 参数时必须失效缓存。
- 完成标记记录实际生成的 engine 文件及大小；缺少文件时自动模式选 CUDA。
  缓存由成功推理产生，单纯安装 TensorRT 或创建 session 不标记就绪。
- CUDA module loading 为 LAZY，驱动编译缓存位于配置缓存目录的 `cuda/`，上限 256 MiB。

这些选项依据 [CUDA EP 配置](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)、
[TensorRT EP 配置](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html) 与
[线程管理](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)；
运行库按 [NVIDIA pip 安装方式](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/installing-tensorrt/install-pip.html)
选择匹配的 CUDA 主版本，配置后仍须实测 provider 节点。

## 验证方法

```powershell
cargo build --release --locked -j 1 -p lumen-engine --bin aimask_probe
```

用启动器指定 `-Executable` 为 `aimask_probe.exe`，`-AppArgs` 依次为输入 PNG/JPEG、
输出目录、采样数，配合 `-Wait`。多次采样先预热 3 次；报告包含 session 加载、
首次调用与暖帧 P50/P95。探针使用与应用相同的 Rust C API，不是 Python 推理替代。

设置 `LUMEN_AI_PROFILE` 为输出目录后，每个 session 的首次调用会保存 ONNX profile。
其中 `args.provider` 可证明 CUDA/TensorRT 实际执行了节点。探针还输出原始 `.f32`
蒙版，供独立比较最大/平均绝对误差与阈值分类差异；输出保留本地。

普通 `cargo test` 不依赖 GPU、安装目录或私有模型。硬件验收必须另行记录证据，
不能将 `GetAvailableProviders` 或 DLL 存在等同于启用成功。

<!-- measured-results -->

## 实测结果

2026-11-12 在上述 RTX 4070 Ti 和 1600×1067 测试图上完成。下表的暖态数据为预热
3 次后的 30 帧采样，单位均为 ms。实际 provider 归属来自同一轮 ONNX profile 中可观测
的 kernel 事件；`auto` 与显式 TensorRT 路径的节点归属相同。

| provider | 主体 P50 / P95 | 皮肤 P50 / P95 | profile 中实际 provider |
| --- | ---: | ---: | --- |
| CPU | 395.79 / 735.14 | 312.64 / 1020.56 | 主体 CPU 286；皮肤 CPU 528 |
| CUDA | 44.42 / 58.79 | 102.26 / 115.37 | 主体 CUDA 386；皮肤 CUDA 541 |
| TensorRT | 32.45 / 38.27 | 99.14 / 117.63 | 主体 TRT 1；皮肤 TRT 4 + CUDA 3 |
| auto | 29.19 / 31.63 | 99.27 / 133.82 | 与 TensorRT 相同：主体 TRT 1；皮肤 TRT 4 + CUDA 3 |

主体 `auto` 的 P50 从 CPU 的 395.79 ms 降至 29.19 ms，皮肤从 312.64 ms 降至
99.27 ms。TensorRT 能覆盖主体的大部分计算图；皮肤的 3 个 `Resize` 节点由 TensorRT
回退到 CUDA，因此皮肤不是纯 TensorRT 执行。

以 CPU FP32 结果为基准，CUDA 主体的最大/平均绝对误差为
`2.748e-5 / 1.485e-7`，皮肤为 `2.027e-6 / 6.909e-9`；TensorRT 和 `auto`
主体为 `2.360e-5 / 1.434e-7`，皮肤为 `2.176e-6 / 8.365e-9`。
按 `>= 0.5` 分类，四组结果的主体和皮肤均为 0 个分类变化（各检查 1,707,200 像素）。

`host-optimized` 样本的暖态 P50 为主体 16.15 ms、皮肤 103.52 ms；该样本是在
Windows GUI Release 链接编译的后台负载下采集，与 `auto-final` 属于不同轮测试，
没有隔离出严格的同条件。独立的逐元素比较显示，它与 `auto-final` 的主体、皮肤
`.f32` 输出逐字节相同，SHA-256 分别一致；最大/平均绝对误差和 `0.5` 阈值分类变化
均为 0。主体从 `auto-final` 的 29.19 ms 到 `host-optimized` 的 16.15 ms 仅为
跨轮观察值，既不能严格算作性能提升，也不能把差额全部归因于 host 去重。皮肤同样
不具备严格同条件，不能据此宣称改善。

本轮硬件结果仅覆盖当前机器和测试图片上的主体/皮肤模型，其他共用模型尚未逐个做
性能与像素验收。GPU 当前只加速这些 AI 蒙版推理；普通曝光、颜色和细节调色仍走 CPU。
GitHub 上传及线上 CI 尚未执行。
