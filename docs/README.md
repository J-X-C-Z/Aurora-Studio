# 文档索引

首次接手先看[代码架构导览](architecture/CODEBASE-TOUR.zh-CN.md)：界面从哪里发请求，预览、RAW、人像、导出和保存分别走哪条路径。历史报告的计数和平台实测只适用于各自分支，引用前先核对提交号。

| 入口 | 用途 |
|---|---|
| [项目 README](../README.md) | 应用概览、启动与开发构建命令 |
| [贡献规范](../CONTRIBUTING.md) | 源图只读、画质与性能约束、测试要求 |
| [架构决策与迁移记录](architecture/README.md) | ADR、契约、边界审计和验证记录 |
| [前端契约](frontend-contract/engine-requests.md) | 通用请求、节点图、目录与人像证据 |
| [Windows 推理](windows-gpu.md) · [macOS 适配](macos-adaptation/05-plan.md) | 各自的设备、工具链与已知边界 |
| [人像 147 交接](architecture/PORTRAIT147-HANDOFF.zh-CN.md) | 哪些可以复用，哪些是草稿，需要哪个上游提交才能继续 |
| [资产与许可清单](../ASSET-LICENSES.md) | 历史来源与待维护者确认项；不是最终许可声明 |

## 独立开发工具

- [EV6 同图对拍](../scripts/ev6/README.md)：像素/ROI/通道误差、热图及合成自测。
- [Vulkan/WGSL 探针](../scripts/probes/vulkan-wgsl/README.md)：分段、融合和 CPU 参考的合成夹具，独立 Cargo workspace。
- [原创唇妆 LUT](../examples/portrait-lip-luts/README.md)：两张可再生成的 PNG 与独立校验，不改变安装后的默认素材。
