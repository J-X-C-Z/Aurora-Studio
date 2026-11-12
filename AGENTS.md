# Aurora Studio 仓库约定

## 目标

保持界面、引擎、测试资产和发布流程可审查。结论以当前源码和实际命令为准；
历史进度文档、记忆和旧截图只能提供线索。

## 边界

- `crates/app` 只通过 `lumen_engine::ImageEngine` 调用处理逻辑。
- 项目代码采用 MIT 许可证；第三方组件与本机私有资产保持各自许可。
- 未获明确授权，不提交、推送、创建远程仓库、发布 release 或上传构建产物。
- 不修改 `crates/engine/testdata/` 中原外部资产，也不把这些资产提交到 Git。
- 不擅自删除或改写用户图片、catalog/session、sidecar 和导出目标。

## 验证

```powershell
cargo metadata --locked --no-deps --format-version 1
cargo check --workspace --all-targets --locked
cargo test --workspace --locked -- --test-threads=1
cargo build -p lumen-app --release --locked
```

默认测试必须 synthetic 且可在 GitHub CI 运行。外部资产测试仅限本机：

```powershell
cargo test -p lumen-engine --features external-fixtures --locked -- --test-threads=1
```

不要给仓库增加全量 `cargo fmt --check` 门禁；现有代码大量采用压缩长行。

## 性能与画质

- 同一视口的拖动和静止预览必须同分辨率、同像素质量，禁止缩放后上采样冒充优化。
- 允许当前在途帧完成；pending 只保留最新值。旧图片和过期 generation 不得覆盖
  当前图片或更晚已展示帧，最终帧必须使用最新值。
- 原图数据和 C1 输入保持只读；编辑、缓存与导出不得回写源文件。
- 性能声明必须基于 Windows 原生 Release，附冷暖帧、P50/P95、画质差异和比例测试。
- 无 CI、外部 fixture 或真机证据时，明确写成未验证，不能推断为通过。

## 资产

个人照片、EXIF 序列号/GPS、客户路径、Capture One 私有模型/预设/ICC/film curve
和第三方模型不得进入仓库或 release。发现此类文件时报告阻塞，由仓库所有者裁决
清理或重新授权；不要自行删除原文件。
