# B5 PixCake pose static-resource report

## Verdict

No body-pose/body-landmark model was recovered. The PixCake container and
PoseRefine configuration are now statically characterized, but no numeric
resource can be proven to be the 69-point body-pose producer.

No model was connected to the B runtime and no keypoints were produced.

## Closed evidence

- The installed container is
  `C:\Program Files\PixCake\pixcake_2\Resources\pixcook`, 2,071,019,532 bytes.
- It contains 3,106 ustar entries: 2,079 numeric resource IDs and 1,027 named
  files. The first ustar header uses the repeating key `ec cd ad cb`.
- `Resources\effect\config\PoseRefine\effect.json` decodes from that key into
  valid JSON. It declares body warp effects including `NeckLength`,
  `ShoulderWidth`, `ArmRefine`, `WaistRefine`, `SlimLegRefine` and
  `BreastRefine`.
- The decoded payload ends with a 16-byte `*xixt%(g*A$l` effect footer; the
  decoder strips that footer before exposing JSON.
- `Resources\effect\config\SubPoseRefine\effect.json` decodes into `BodyRefine`,
  `LengthenLegRefine` and `HeightenRefine`.
- `pix-worker.exe` contains the runtime symbols `AIBodyReshapeDetectorV3`,
  `AILimbReshapeDetector`, `BodyReshapingV2`, `LimbReshapingParams`,
  `jointPoints`, `SkeletonFeatures`, and the hash lookup error
  `[AIModel] LookupModelPath: model not found, hashId=%u, md5=%s`.

These findings close the effect layer and the hashId lookup mechanism. They do
not close a pose model identity.

## Decoder

`tools/body_pose_capture/decode-pixcook.ps1` supports:

- read-only ustar inventory;
- extraction by numeric resource ID with outer XOR;
- named entry extraction without an assumed transform;
- standalone effect-config decoding;
- `-RequireModelMagic` fail-closed model validation.

It recognizes PNG, JSON, MNN, TensorFlow Lite, ONNX protobuf candidates and
NCNN parameter text. Unknown content is reported as opaque rather than guessed.

## Precise blocker

1. There is no local `AIModels` tree in the audited PixCake installation.
2. The decoded PoseRefine/SubPoseRefine JSON contains parameters only. It has no
   model ID, resource ID, file name, digest, tensor name or tensor shape.
3. Numeric container IDs are not named. Known eye-whiten resource
   `1527454531` is 4,042,709 bytes at offset 355,397,632; even after outer XOR
   it remains opaque and exposes no supported neural-model magic.
4. The known non-target is insufficient to infer that neighboring IDs are
   pose models. Static naming alone would not prove an output keypoint layout.
5. No concrete pose hashId is present in the executable's exposed strings, and
   no static ID-to-model-name table was found in the local SQLite catalog.
6. Consequently the input/output tensor contract, SHA-256 and runtime adapter
   cannot be truthfully produced.

## Required next evidence

One of the following must close the blocker:

- the PixCake model-payload decryption routine or key plus a concrete pose
  `hashId`;
- a trusted app-resource response/HAR that pairs pose `resourceId`/`hashId` with
  `downloadUrl` and digest;
- a local pose model artifact with supported MNN/ONNX/TFLite magic and readable
  tensor metadata.

Acceptance for B runtime then requires: verified SHA-256, exact model input
layout/preprocessing, output shape and keypoint order matching the native
PixCake producer, plus at least one real keypoint output.

## Reproduction

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\body_pose_capture\decode-pixcook.ps1 -Inventory
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\body_pose_capture\decode-pixcook.ps1 -StandaloneFile "C:\Program Files\PixCake\pixcake_2\Resources\effect\config\PoseRefine\effect.json" -OutFile .\pose-refine.json
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\body_pose_capture\decode-pixcook.ps1 -Id 1527454531 -InspectOnly
```
