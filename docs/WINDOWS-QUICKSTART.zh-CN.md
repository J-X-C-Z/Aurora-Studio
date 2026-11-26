# Aurora Studio · Windows 桌面版

## 实际修一张图

1. 双击 **Aurora Studio**，选 **打开文件夹…**。选中含有照片的文件夹，在底部胶片栏点选原图。也可把照片或文件夹拖到**桌面快捷方式**上；来自不同图库的文件会分别打开。欢迎页的「浏览演示图像」是可选的程序示例，不是自己的图库。
2. 进入调色工具，拖动「曝光」后松手。一次连续拖动对应一次撤销；`Ctrl+Z` 撤销，`Ctrl+Shift+Z` 重做。正常关闭窗口后，调整保存在图库内隐藏的 `.portrait` 中，原照片的字节不变。迁移或备份整套编辑时，把原片与 `.portrait` 一起复制。只读目录先复制到可写的位置再编辑。
3. 在胶片栏选一张或多张。`Ctrl+D` 处理所选照片，默认配方为全尺寸 JPEG，成片在原片旁的 `Output` 文件夹；`Ctrl+Shift+D` 打开导出对话框，可选 TIFF 16 位、更改尺寸和目的地。到「批次」查看每项成功或错误后再关窗口。保留原片，不把成片路径设置为原文件。
4. 重开同一图库，点击之前的照片。「曝光」仍然显示保存的数值。部分 AI 工具需要兼容模型；普通调色、旁车保存和标准导出不以模型为前提。CPU/显卡路径及 RAW 支持因机型、传感器、压缩方式和驱动而异；出现嵌入预览不表示已解码传感器数据。

## 应用如何工作

```text
桌面快捷方式 / 打开方式
  └─ portrait-app（GPUI：欢迎页、图库、选择、撤销、任务与预览）
       └─ ImageEngine（UI 与引擎的接口）
            ├─ FileEngine：真实文件、调整旁车、缓存与导出
            │    ├─ RAW：内嵌 JPEG 首屏 → 传感器解码 → 去马赛克/线性处理
            │    ├─ 预览：调色 → 显示变换 → 查看器、直方图、示波器
            │    ├─ 导出：准备源 → 渲染 → 编码 → 临时文件完成后发布
            │    └─ aurora-portrait：ONNX 图、显存规划、内嵌 PTX / 系统 NVIDIA 驱动
            └─ MockEngine：只有主动选示例时才用程序生成的照片
```

- `crates/app/src/state/source.rs` 和 `ui/welcome.rs`：按文件、图库或目录打开；`state/mod.rs` 与 `ui/workspace.rs`：提交手势、保存与关闭。
- `crates/engine/src/file_engine.rs`、`file_engine_services.rs`、`store.rs`：扫描原片、保存调整、提交任务；`export/`：有界流水线及失败清理。
- `crates/engine/src/aurora_raw/`：传感器、色彩、可选开源 LibRaw 与缓存。`mock/develop_gpu.rs` 的历史目录名不表示仅供演示，它也是实际文件引擎复用的开发/预览路径。
- `crates/portrait/src/native_infer/`：静态形状、融合、arena、执行器和 CUDA Graph。图不适合录制、形状动态或验证失败时保留普通执行。嵌入的 PTX 由驱动加载，最终用户不需要装 CUDA 编译器。

## 开发者复现

使用 Windows Rust GNU 工具链、MinGW 与 Windows SDK 的 `fxc.exe`；仓库根目录执行：

```powershell
./scripts/build-windows-release.ps1 -FxcPath $env:GPUI_FXC_PATH  # 可追加 -LibRawRoot <含 bin/ 与 licenses/ 的目录>
./scripts/install-windows.ps1  # 校验 SHA-256，保留旧安装，创建桌面快捷方式
cargo check --workspace --all-targets --locked
cargo test --locked -p aurora-engine --test desktop_workflow -- --test-threads=1
cargo test --locked -p aurora-engine --test architecture_rules --test architecture_regression
cargo test --locked -p aurora-portrait --lib native_infer:: -- --test-threads=1
```

发布目录为 `target/dist`（可用 `-OutDir` 更改）。标准 RAW 加载器优先支持与 exe 同级的 `libraw/libraw-25.dll`（LibRaw 0.22.3 及其整套开源依赖、许可证），开发者也可设置 `AURORA_LIBRAW_DIR`。不要把商业应用的模型、ORT、cuDNN 或 cuBLAS 拷进发布包。若另行提供用于对照的 CPU 参考库，可设置 `AURORA_MODELS_DIR`、`AURORA_ORT_LIB`、`AURORA_ORT_PROVIDERS=cpu` 后按任务书跑严格对拍。缺文件、缺参考或超误差必须单列，不算通过。

对指定 Z6III HE 黄金原片，设 `AURORA_Z6III_GOLDEN_NEF` 后运行 `cargo test -p aurora-engine --features z6iii-raw-export cuda_vs_opencl_bit_exact_and_bench_real_nef -- --ignored --nocapture --test-threads=1`。只有同一张原片应有 `ef4057f5335a4ba7` 的像素哈希。其他 Z6III HE 使用 `AURORA_TEST_RAW` 做双后端比较，不套用该哈希。公开通用 RAW 可用 `cargo run --release --locked -p aurora-engine --features z6iii-raw-export --example raw_sensor_check -- <原片路径>` 检查传感器输出、位深范围与指纹。

## 运行库与移植边界

- 便携包附带的可替换 RAW 动态库来自 [MSYS2 mingw64](https://repo.msys2.org/mingw/mingw64/) 的 LibRaw 0.22.3-1、GCC/libgomp/libstdc++ 运行库、winpthreads、lcms2、libjpeg-turbo 与 zlib。单独的库及其原许可放在 `libraw/` 和 `libraw/licenses/`；无需设置全局 DLL 搜索路径。[LibRaw 对应源码](https://github.com/LibRaw/LibRaw/tree/0.22.3) · [MSYS2 构建配方](https://github.com/msys2/MINGW-packages)。
- 开发者替换 PTX 时：在具备相应 Windows SDK、Visual Studio 与 NVIDIA Toolkit 的开发命令行，对 `infer_quant_int8.cu` 和 `infer_conv_gemm.cu` 使用 `nvcc -arch=compute_75 --ptx -O3`；`scope_gpu.cu` 还需 `-fmad=false` 以保持逐桶一致。更换产物后重跑真实模型和图像对拍，再生成 release。此次本机三个变更模块用 NVIDIA NVRTC 13.4 的 compute_75 目标生成并在本机驱动 JIT 验证；这不等同于验证了 nvcc 编译，也不是所有老驱动都兼容。
- 卸载应用只需删本机安装文件及「Aurora Studio」快捷方式。旧版本保留在安装器结果的 `Previous` 目录；原片及旁车不要删除。

本仓公开前端合同是 `docs/frontend-contract/params.json` 的 72 个滑块（逐项校验名称、上下限、默认值、步长、小数与单位）。旧研究快照有 82 项，本仓未分发；拿到该文件后可设置 `AURORA_LEGACY_PARAM_CONTRACT` 并用 `--features external-fixtures` 复跑两条原断言，不把 72 项当 82 项。
