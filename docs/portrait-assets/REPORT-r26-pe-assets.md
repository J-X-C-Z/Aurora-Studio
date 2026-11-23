# R26 PE assets

状态：`BLOCKED_NO_ASSETS_PE_NAMESPACE`

基线 `eececfd`，工作副本 HEAD `17dc3b8`。本轮只写
`r26-pe-assets/repo`，未写主树，未运行 cargo/GPU，未扫描 `target/`。

## 结论

安装根为 `C:\Program Files\PixCake\pixcake_2`。公开 `Resources` 目录树、
`Resources\pixcook`、`PixCakeMaterial.rcc`、`PixCakeSequenceFrame.rcc`、
独立 Qt PAK 和 `pixcake.exe`/`pix-worker.exe` 的 PE overlay 均没有
`assets/PE` 命名空间。

`pixcook` 是 3,106 项 tar 载荷，根目录只有 `glsl`、`Configure`、
`InnerMaterial` 和无目录条目；其中 8 个
`RetouchMoisturizing` shader 节点提供 sampler 名称，但没有同名纹理文件。
`PixCakeMaterial.rcc` 的 751 项里只有
`/resources/material/skin/skin_texture_01@3x.png` 至 `04@3x.png` 这组
普通皮肤材质，不能据此冒充 `skin_map`、`hardlight`、`beauty` 或
`blur_alpha`。

`pixcake.exe` 的尾随 overlay 为 10,752 B，开头是 Authenticode/PKCS#7
证书结构，不含 `assets/PE`、moisture shader 或纹理 token。

## 精确阻塞

```text
namespace: assets/PE
searched:
  C:\Program Files\PixCake\pixcake_2\Resources\
  C:\Program Files\PixCake\pixcake_2\Resources\pixcook
  C:\Program Files\PixCake\pixcake_2\Resources\PixCakeMaterial.rcc
  C:\Program Files\PixCake\pixcake_2\Resources\PixCakeSequenceFrame.rcc
  C:\Program Files\PixCake\pixcake_2\Resources\*.pak
  C:\Program Files\PixCake\pixcake_2\pixcake.exe:overlay
  C:\Program Files\PixCake\pixcake_2\pix-worker.exe:overlay
result: ABSENT
```

因此没有可登记的 vendor texture SHA，也没有可接给 PF90002/PF90010
producer 的真实 `skin_map`、`hardlight`、`beauty` 或 `blur_alpha`。
现有 typed producer 保持具名拒绝，不接入合成纹理、普通 UI skin texture
或重复 shader 节点。

## 工具

`tools/pixcake_pe_asset_probe.py` 只读解析 public tree、RCC v3、
`pixcook` tar、PAK 和 PE overlay，并输出 catalog。成功发现
`assets/PE` 时会列出候选；失败时返回码 2 并写入精确 namespace blocker。

```powershell
python tools\pixcake_pe_asset_probe.py `
  --install-root "C:\Program Files\PixCake\pixcake_2" `
  --out docs\portrait-assets\r26-pe-assets-catalog.json
```

## 未验证

没有恢复 vendor texture 内容，未连接 moisture producer，未运行像素或
原厂 parity。下一步必须取得包含 `assets/PE` 的安装 payload、更新包或
独立资源包；仅凭当前安装无法越过该命名空间阻塞。
