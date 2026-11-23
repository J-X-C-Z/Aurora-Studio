# B7 body-pose private payload format report

## Verdict

No body-pose or landmark network was recovered. All 2,079 numeric resources were classified; no entry passed MNN, TFLite FlatBuffer, strict ONNX protobuf, generic FlatBuffers, or one-byte/repeating-XOR model checks. `pose_model` remains null and the B runtime remains unconnected.

## Classification

- Formats: outer:png=6, raw:png=10, raw:json=2, raw:weak-onnx-prefix=2, opaque=2059.
- Classes: recognized-non-model=20, high-entropy-opaque=556, binary-opaque=1503.
- Entropy over head/middle/tail samples: min 0.379, median 7.5403, max 7.9936.
- The B6 count of 2,061 unresolved payloads is reconciled as 2,059 opaque plus 2 rejected weak ONNX-prefix payloads (1524722780:554610B@354464768, 3899286051:554612B@1598331392).
- Raw/outer-XOR/single-byte-XOR neural candidates: 0. Pose/keypoint/landmark/joint/skeleton token hits: 0.
- The 30 PoseRefine/AIBodyReshape shaders prove downstream warp math only; they contain no weights, tensor contract, landmark order, or model identity.

## Exact blocker

The private numeric namespace has no supported serialization or proven outer transform, its weak `08 07` prefixes fail ONNX structural validation, and no static record maps any resource to `AIBodyReshapeDetectorV3`, PoseRefine, a tensor header, or the native 69-point order. Inventing a decoder key, model ID, SHA-256, tensor contract, or keypoints would be fabrication.

Acceptance requires a concrete pose artifact or trusted download plus its digest, a supported model transform/format, exact tensor I/O and landmark order, and at least one real keypoint run before `BodyPoseModel` can be connected.

## Reproduce

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\body_pose_capture\classify-pose-payloads.ps1 -OutFile .\pose-payload-classification.json
```

The complete per-resource classification is in `tools/body_pose_capture/pose-payload-classification.manifest.json`.
