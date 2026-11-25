# 私有资产（不进 Git）

仓库是公开的。第三方软件里提取或解码出来的数据、Capture One 的提取数据、
从第三方进程抓取的测试 fixture，以及相关的逆向记录和工具，都**不放进 Git**，
只放在工作区根目录下被 `.gitignore` 忽略的 `private-assets/`。

## 目录布局

```text
private-assets/
├── portrait/        人像运行时资产（LUT、网格、预设、vface 图集等），编译期嵌入
├── c1/              Capture One 提取数据：camera_params*.json、会话模板
├── fixtures/
│   └── portrait/    依赖上述资产或抓取数据的测试 fixture
└── research/        逆向记录、采集工具与报告；不参与构建，仅供本机查阅
```

除 `research/` 外的完整文件清单与 SHA-256 见
[docs/private-assets.sha256](docs/private-assets.sha256)，可在 `private-assets/` 内校验：

```sh
cd private-assets && sha256sum -c ../docs/private-assets.sha256
```

## 构建如何使用它

`crates/engine` 和 `crates/portrait` 的 `build.rs` 都调用
`build-support/private_assets.rs`：

- 目录存在（默认 `<工作区>/private-assets`，或环境变量 `AURORA_PRIVATE_ASSETS`
  指向的目录）时，设置 `cfg(aurora_private_assets)`，资产在编译期嵌入，行为与之前一致。
  目录存在但缺文件会直接编译失败，错误信息里有缺失的路径。
- 目录不存在时，`aurora_portrait::private_asset!` 返回 `None`。需要这些资产的效果
  返回 `Refusal::MissingInput("private-assets/...")`，C1 会话导出返回错误，
  相机默认参数表为空。不会用假数据代替。
- 依赖这些资产的测试带
  `#[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]`，
  没有资产时显示为 ignored，不会被算作通过。

运行期读取的文件（如 PF8033 回执）用 `aurora_portrait::private_assets::path(...)` 定位，
运行时的 `AURORA_PRIVATE_ASSETS` 优先于编译期目录。

## 新增私有数据时

1. 文件放进 `private-assets/` 对应子目录，不要放进 `assets/`、`tests/fixtures/` 或 `docs/`。
2. 代码里用 `private_asset!` / `private_asset_str!` 引用，处理 `None`（拒绝，而不是 panic）。
3. 测试加上面的 `cfg_attr(... ignore ...)`。
4. 更新 `docs/private-assets.sha256`：

   ```sh
   cd private-assets && find . -type f ! -path './research/*' | sed 's#^\./##' | sort | xargs sha256sum > ../docs/private-assets.sha256
   ```

5. 同时用有资产和无资产（`AURORA_PRIVATE_ASSETS=` 置空）两种方式各跑一次
   `cargo check --workspace --all-targets --locked`。

## 获取

私有资产不对外分发。需要的开发者向仓库所有者索取，自行确认使用权限。
