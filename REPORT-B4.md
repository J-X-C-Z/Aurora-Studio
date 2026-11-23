# B4 PixCake body-pose endpoint report

## Verdict

No validated body-pose HTTP request or model artifact was recovered. PixCake was
not running during the capture window; only Capture One 16.8.6.3784 was live.
Historical PixCake logs contain concrete API URLs, but they omit the HTTP verb,
request headers, request body and complete response. No model ID, download URL
or digest can therefore be connected to the native 69-point body skeleton.

No model was passed to the B runtime.

## Read-only runtime evidence

- Capture One was running as PID 28052. Its TCP snapshot contained only its
  loopback plugin listener on `127.0.0.1:40000`.
- No `pixcake.exe`, `pix-worker.exe` or PixSoda process was present.
- PixCake 10.0.5-156 historical logs were found under
  `%APPDATA%\PixCake-qt_pro\logs\26_11_13-02_13_34-10.0.5-156\`.
- The system proxy pointed to `127.0.0.1:7897`, but no decrypted HTTP trace was
  available. A proxy tunnel alone does not expose HTTPS paths or bodies.
- The old PixCake log observed Qt remote debugging on `127.0.0.1:9004`. That
  remains useful for CEF traffic only; it does not capture native QtNetwork
  requests.
- No token, cookie or authorization value from the historical logs was reused.
  Raw PixCake logs must be treated as credentials-bearing and redacted.

## Captured request metadata

These are historical URLs from `app-client-*.txt`, not reconstructed routes.
The log line records only a `HttpRequestTask` display name ending in `-`; it
does not retain method, headers, body or response.

| Host / path | Evidence | Proven classification |
|---|---|---|
| `https://api.pixcakeai.com/v2/api/app_resource/query` | Two startup requests in the historical client log | Resource catalog query candidate; not proven to serve the pose model |
| `https://api.pixcakeai.com/v1/api/app_resource/group/query` | Historical client log | Resource-group query; not proven to serve the pose model |
| `https://api.pixcakeai.com/v1/api/app/user/info` | Historical client log | Account bootstrap |
| `https://res.pixcakeai.com/resource/app/18/<opaque>.png` | Historical download task | Static banner resource; unrelated concrete asset type |
| `https://download.pixcakeai.com/package/...` | Historical download task | Application/package download |
| `https://api.pixsoda.com` | `pixcake.exe` string | Origin only; no concrete route captured |
| `/download?encrypted_query_param=` | `pixcake.exe` string | Generic download URL fragment; owner and model role unclosed |

The strongest lead is `/v2/api/app_resource/query`, because it is reached during
resource preload. It is still a candidate, not a closed pose endpoint.

## Response and header fields

Executable strings contain the field names `resourceId`, `downloadUrl`,
`modelId`, `selectedModelId`, `cloudSourceDownloadUrl`, `md5Empty` and
`sha1Empty`. One error template also proves the client expects a `downloadUrl`
when `resourceId` is present:

```text
download url empty. resourceId:%1 md5Empty:%2 sha1Empty:%3
```

Header-name strings include `Authorization`, `Bearer`, `Device-Id`,
`User-Agent`, `X-Request-Id` and `X-Request-Timestamp`. This proves names only;
no concrete header set or ordering was captured.

No captured object pairs a concrete `resourceId` with a pose `modelId` and a
`downloadUrl`. No model MD5/SHA-1/SHA-256 is available.

## Pose and encryption boundary

- `AVBodyReshapeDetectorV3`, `CalculateBodyRoiByAlgorithm`, `KeyPoint`,
  `keypoints`, `_pose_b1H`, `_pose_b1J`, `_pose_w1H`,
  `pose_points_bucket` and `pose_points_original` remain symbolic leads.
- PixCake's numeric `pixcook` entries remain a private container format. The
  first ustar header is XORed with `ec cd ad cb`; this does not map an entry to
  body pose.
- Body effect configs are not plain JSON. For example,
  `Resources\effect\config\BodyPortraiture\effect.json` is 123 opaque bytes,
  begins `97 c0 a7 eb cc ef e8 ad...`, and ends with the ASCII marker
  `*xixt%(g*A$l`. A second marker `*xixt%(g*1$0` exists in the executables.
- The exact effect cipher is not closed. AES-family strings exist in the
  executable, but static evidence is insufficient to claim AES-CBC/CFB/CTR.
- The downloaded model's encryption/serialization is also open because no model
  response or artifact was obtained.

## Precise blocker

1. PixCake was not running, so there was no current process to observe.
2. The retained client log records URL display names only; verb, headers, body
   and complete response are absent.
3. A trusted TLS interception proxy was not active, and using the historical
   token would violate the no-auth-bypass boundary.
4. `api.pixsoda.com` has only an origin string, not a route.
5. No artifact exists to hash, decrypt, inspect for ONNX/MNN tensor shape, or
   run against the B runtime.

Use `tools/body_pose_capture/CAPTURE-NETWORK-B4.md` to capture the missing HAR.
Acceptance requires one HAR entry and one downloaded artifact that jointly prove
the pose model ID, method/path, response `downloadUrl`, response digest and
actual tensor contract.

