# 文档索引

结论以当前源码和实际命令为准。下面的报告多为某一轮工作的快照，行号、计数和
“已验证”状态都可能已经过时，引用前请对照源码重新确认。

## 先读这些

| 文档 | 用途 |
|---|---|
| [../README.md](../README.md) | 项目概览、构建测试命令、仓库结构 |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | 提交门槛、预览与性能规则、原图数据安全、测试范围 |
| [../AGENTS.md](../AGENTS.md) | 自动化 Agent 与 lane 协作约定 |
| [../ASSET-LICENSES.md](../ASSET-LICENSES.md) | 运行时资产来源与许可状态（草案） |
| [architecture/README.md](architecture/README.md) | 模块边界、ADR、迁移映射、架构规则测试 |
| [architecture/REMAINING-GAPS.md](architecture/REMAINING-GAPS.md) | 未完成重构的接手清单 |
| [Windows 桌面性能候选 C4](architecture/DESKTOP-C4-REVIEW.zh-CN.md) | 历史性能分支的代码入口、复现命令与未通过的门槛 |

## 按主题

| 目录 / 文件 | 内容 |
|---|---|
| [frontend-contract/](frontend-contract/engine-requests.md) | 界面与引擎之间的请求契约、节点图、目录、人像参数与证据表 |
| [windows-gpu.md](windows-gpu.md) | Windows 上 CUDA / TensorRT / DirectML 的运行库与实测口径 |
| [macos-adaptation/](macos-adaptation/05-plan.md) | macOS ARM64 / Metal 适配的基线、阻塞项、决策与计划 |
| [../private-assets.md](../private-assets.md) | 不进 Git 的私有资产：目录布局、构建开关、新增流程 |
| [private-assets.sha256](private-assets.sha256) | 私有资产文件清单与校验和 |
| [performance-audit-implementation-2026-11-24.md](performance-audit-implementation-2026-11-24.md) | 整体性能审计与落地记录 |

## 写新文档时

- 放进上面对应的主题目录，不要放在仓库根目录。
- 只写仓库内的相对链接；本机或 lane 工作区路径（如 `90_work/`、`06_reports/`）对其他开发者不存在。
- 写明依据的提交号和实际运行的命令；没运行的检查写成“未验证”。
- 不写个人路径、用户名、相机序列号、客户文件名或凭据。
