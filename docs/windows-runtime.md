# Windows 运行库

GUI 使用系统 DirectX 11；运行包中的第三方库与许可证见 THIRD-PARTY-NOTICES.md。

## RAW

本次运行包的 LibRaw 源码版本为 0.23.0-Snapshot202611，commit `a9a14b1970d188c25dade81f8326ff00a6e33649`。二进制为动态 C API 库，使用 MSYS2 mingw64 工具链构建。源码与上游构建文件位于：

https://github.com/LibRaw/LibRaw/tree/a9a14b1970d188c25dade81f8326ff00a6e33649

将 libraw.dll 和它实际依赖的 libwinpthread-1.dll、zlib1.dll 放在程序同目录的 libraw/ 中；也可用 LUMEN_LIBRAW_DIR 指向自己的运行库目录。发布包应同时保留对应许可和来源说明。旧的 MSYS2 0.22.3 安装脚本不用于本次 0.23 运行包。

缺少 LibRaw 或无法处理某种压缩时，后端可能退回 RAW 内嵌预览；不能把这种结果当作完整传感器解码验收。HE* 压缩 NEF 的完整解码仍有限制。

## 本机可选资产

- 模型和 ONNX Runtime 由用户本机合法安装，发布包不携带。
- LUMEN_C1_DIR 可指定本机安装根；LUMEN_FILM_CURVES 指定胶片曲线目录。
- LUMEN_LENS_DB 用于本机镜头数据库。没有可用校准数据时，不能宣称已完成校准校正。
- AURORA_SETTINGS 可把测试设置隔离到专用 JSON 文件；性能冒烟使用合成图，不触碰用户照片与设置。

图形界面使用 NVIDIA 显卡显示，不等于 AI 推理在 CUDA/TensorRT 执行。本版未添加或验收新的 CUDA/TensorRT 配置。
