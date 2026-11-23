# B3 body-pose acquisition report

## Result

Delivered an executable, fail-closed acquisition kit under
`tools/body_pose_capture/`. It can discover local pose-named ONNX files,
inventory and extract PixCake's private numeric container entries, download a
manifest-declared resource, verify SHA-256, and write the `.sha256` sidecar
expected by `BodyPoseModel::discover`.

No body-pose model was fabricated or connected to the runtime. No keypoints
were emitted.

## Model identity

| Source | Model ID | File / container path | Closed? |
|---|---|---|---|
| Capture One | none | `C:\Program Files\Capture One\Capture One\AIModels` | No pose ONNX present |
| PixCake local | unresolved | `C:\Program Files\PixCake\pixcake_2\resources\pixcook` | Container format closed; pose ID open |
| PixCake cloud | unresolved | `https://api.pixcakeai.com`, `https://api.pixsoda.com` | Origins observed; concrete model route open |

Known non-target model: PixCake `1527454531` at container offset `355397632`,
`4042709` bytes, is the eye-whiten network and is not body pose.

## Proven acquisition facts

- The PixCake resource container is a ustar archive of `2071019532` bytes with
  `2079` numeric and `1027` named entries on the audited install.
- The first 100 bytes of the first ustar header are XORed with repeating key
  `ec cd ad cb`; later headers are plain ustar.
- PixCake binary strings expose `AVBodyReshapeDetectorV3`,
  `CalculateBodyRoiByAlgorithm`, `KeyPoint`, `keypoints`, `_pose_b1H`,
  `_pose_b1J`, `_pose_w1H`, `pose_points_bucket` and `pose_points_original`.
- Resource-response/request field names observed in the binary include
  `resourceId`, `downloadUrl`, `modelId`, `selectedModelId`,
  `cloudSourceDownloadUrl`, `md5Empty` and `sha1Empty`.
- The existing B2 runtime requires fixed float32 NCHW `[1,3,H,W]` input and a
  static `[1,K,2|3]`-family output. Its SHA sidecar is `<model>.sha256`.

## Unclosed

- No numeric PixCake model ID is proven to produce the native 69-point skeleton.
- No concrete HTTP method, route, query/body schema, authentication or complete
  JSON response has been captured for the model-download service.
- Named `PoseRefine` GLSL resources are rendering kernels, not a keypoint model.
- A PixCake numeric entry remains a private MNN-family asset until separately
  converted and checked; it cannot be renamed to ONNX.
- Runtime acceptance still requires a real model, tensor-contract match and a
  real-image keypoint run.

## Reproduce

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\body_pose_capture\discover-body-pose.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\body_pose_capture\inventory-pixcook.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\body_pose_capture\extract-pixcook.ps1 -Id 231823567
```

`PATCH.diff` is relative to master control `eececfd` and includes prerequisite
B2 commit `e3ef60a` because the assigned HEAD was `eececfd+B2`.
