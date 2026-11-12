## 改动

说明用户可见行为、修改原因和范围。

## 验证

- [ ] `cargo check --workspace --all-targets --locked`
- [ ] `cargo test --workspace --locked -- --test-threads=1`（默认关闭 `external-fixtures`）
- [ ] `cargo build -p lumen-app --release --locked`
- [ ] 涉及预览性能时附 Windows 原生 Release 冷暖帧、P50/P95 和画质证据
- [ ] 涉及拖动/静止时确认同分辨率、同像素质量
- [ ] 比例测试覆盖 1:1、4:3、3:2、16:9 和竖幅
- [ ] 原图、catalog/session 和 sidecar 未被修改
- [ ] 未提交个人照片、本地路径、测试资产或第三方私有模型/ICC/曲线

未执行的检查：

风险与已知限制：
