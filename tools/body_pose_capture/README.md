# Body-pose model acquisition and capture kit

This directory contains a fail-closed acquisition kit for Aurora's body-pose
runtime. It does not contain vendor model binaries. Every downloaded or
extracted artifact must have an expected SHA-256 before it is accepted.

## Commands

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\body_pose_capture\discover-body-pose.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\body_pose_capture\inventory-pixcook.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\body_pose_capture\extract-pose-model.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\body_pose_capture\extract-pixcook.ps1 -Id 231823567
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\body_pose_capture\fetch-body-pose.ps1 -Manifest .\path\to\verified-manifest.json
```

`discover-body-pose.ps1` searches explicit Capture One and PixCake model roots.
`inventory-pixcook.ps1` walks the PixCake ustar container without extracting its
2 GB payload. `extract-pose-model.ps1` decodes both PoseRefine effect configs,
inventories the 3,106-entry container, records pose shader evidence, scans
payload headers in raw and outer-XOR form, and optionally correlates a supplied
JSON/HAR network payload by `resourceId`/`modelId`/`hashId`.
`extract-pixcook.ps1` extracts one numeric container entry by ID.
`fetch-body-pose.ps1` accepts HTTPS/file URLs only when the manifest provides a
64-hex SHA-256.

The extractor intentionally leaves `pose_model` null unless an artifact has a
verified neural format, SHA-256 and a proven pose tensor/keypoint contract.
`-RequireNeuralCandidate` fails closed when no supported model magic is found.
To correlate a captured private response without asserting model identity:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\body_pose_capture\extract-pose-model.ps1 -NetworkPayloadPath .\capture.har -OutFile .\pose-static.json
```

The fetch manifest is:

```json
{
  "artifacts": [
    {
      "model_id": "required-model-id",
      "file_name": "bodyPose.onnx",
      "url": "https://example.invalid/model",
      "sha256": "64 lowercase or uppercase hex digits",
      "headers": {}
    }
  ]
}
```

The fetcher writes `bodyPose.onnx.sha256`, which is the sidecar format used by
`aurora_engine::aimask::body_pose::discover`. It also writes a capture receipt
containing request URL, header names, response status, response headers, byte
count and measured digest. Header values are deliberately omitted.

## Evidence boundary

The PixCake container starts with a ustar header whose first 100 bytes are XORed
with the repeating key `ec cd ad cb`. This is a container-layout fact, not proof
that a numeric entry is a body-pose model.

Observed client strings include `AVBodyReshapeDetectorV3`,
`CalculateBodyRoiByAlgorithm`, `KeyPoint`, `keypoints`, `_pose_b1H`,
`_pose_b1J`, `_pose_w1H`, `pose_points_bucket` and `pose_points_original`.
Observed resource-response fields include `resourceId`, `downloadUrl`, `modelId`,
`selectedModelId`, `cloudSourceDownloadUrl`, `md5Empty` and `sha1Empty`.
No observed string closes the endpoint route or maps a numeric model ID to the
69-point body-pose output. Do not infer that mapping from a filename alone.

Capture One's local `AIModels` tree has no pose ONNX on the audited machine.
PixCake's `resources\pixcook` is a mixed resource container, not an ONNX model
directory. A candidate is accepted only after its bytes, sidecar digest and
ONNX tensor contract are all verified.
