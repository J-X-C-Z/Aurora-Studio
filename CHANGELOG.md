# Changelog

## 0.1.0-preview.20261114

### GPU

- CUDA detail path (`AURORA_DETAIL_GPU=1`) for luminance NR, colour NR and sharpening; byte-exact against CPU on RTX 4070 Ti / 610.88.
- ONNX Runtime provider selection: TensorRT → CUDA → DirectML → CPU by exported symbols; `AURORA_AI_PROVIDER` pins one provider. Current Capture One ORT 1.22.1 exports DirectML + CPU.
- Portrait CUDA kernels: 16 test files / 84 tests passed with `PIXCAKE_CUDA_REQUIRED=1`; not yet wired into the app's portrait preview path.
- Metal remains an unverified macOS target on this Windows build.

### Portrait

- Eight effects now render with real same-job inputs: PF21015, PF30096, PF90005, PF90007, PF8068, PF8069, PF8070, PF90004.
- People parser returns face skin, body skin and hair in one inference; masks are cached per source image.
- Remaining 124 registered CPU kernels are labelled as missing face geometry, model output, material/UV or ROI/statistics.

### Fixed

- Portrait zero values are no longer treated as "unset"; `0.0` is a valid negative value for PF21015/PF30096/PF8069.
- Skin LUT manifest now includes cold/warm atlases and no longer carries external install provenance.
- Workspace license metadata matches the MIT `LICENSE`.

## 0.1.0-preview.20261112

- Levin frontend snapshot and Windows backend integration.
- Drag preview keeps full viewport quality; latest-value merge during drag.
- Correct 270-degree auto-rotation contract.
- Capture One style import for the current session; local film-curve list.
- LibRaw 0.23 dynamic decode with embedded-preview fallback for unsupported HE* NEF.
