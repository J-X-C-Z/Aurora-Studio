# Mac lane 技术决策记录

| # | 决策 | 依据 | 影响 |
|---|---|---|---|
| D1 | 新的 Metal runtime 用 `objc2` 0.6.5 + `objc2-metal` 0.3.2 + `objc2-foundation` 0.3.2，作为 `cfg(target_os = "macos")` 依赖 | 这三个 crate 已由 `gpui-pre-apple 0.3.11` 解析进 `Cargo.lock`，不新增未知依赖；所有权/autorelease 语义正确，避免重写 ObjC msgSend | Windows 构建不受影响；`Cargo.toml` 增 target-specific 段 |
| D2 | `crates/portrait/src/metal*.rs` 现有手写 host 在 M2 保留为**测试/legacy**，不迁移；M3 再评估并入 engine runtime | 其 `MTLCreateSystemDefaultDevice` 多 retain 一次（泄漏），且每 kernel `waitUntilCompleted`，不适合连续子图 | 暂时存在两套 host；禁止在 legacy host 上继续堆产品逻辑。修复泄漏列为 M3 必做 |
| D3 | `MACOSX_DEPLOYMENT_TARGET = 12.0` | `gpui-pre-apple 0.3.11` 的 metallib 硬编码 `-mmacosx-version-min=12.0`；更新 API 均为运行时判断 | 低于 12.0 的构建脚本直接失败 |
| D4 | 显示路径不宣称零拷贝 | GPUI 0.3.11 无 RGB 外部纹理；`paint_surface` 只接受 NV12 | 先做 GPU→Shared→一次读回→GPUI 上传；外部纹理需改公共接口，另立 patch |
| D5 | macOS 不新增全局环境开关；沿用既有 `AURORA_*` 仅作开发/AB | 任务书 §8 | 用户设置走统一配置 |
| D6 | 许可证冲突（`MIT` vs `Proprietary`）**不裁决、不修改** | AGENTS.md / CONTRIBUTING.md | 由仓库所有者决定 |
| D7 | 公共 `gpu_runtime` 的后端无关拆分由 Mac-A 执行；Mac-B 只新增 `metal/*` | 任务书 §4.1，避免第二套调度器 | Mac-B 不直接改 `mock/gpu_runtime.rs` |
| D8 | macOS 默认推理 provider = **CPU**；Core ML 仅在显式 `AURORA_ORT_PROVIDERS=coreml,cpu` 时启用 | 本机 Capture One 的 ORT 1.20.1 确实导出 CoreML，但 EP 可能用 FP16 计算，且没有 CPU↔CoreML 的连续 alpha/IoU 对拍证据；README 明确反对用 FP16 原始概率做羽化 | 代码只按符号“可导出”，不自动选中；CoreML 执行与 ANE 归属仍为未验证 |
