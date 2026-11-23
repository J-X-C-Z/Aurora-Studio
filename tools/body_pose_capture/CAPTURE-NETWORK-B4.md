# PixCake body-pose network capture template

This procedure records the user's own authenticated PixCake session. It does
not inject into a process, reuse a captured token, bypass authentication or
disable certificate validation in PixCake.

## Safety

- Capture only traffic from the current logged-in PixCake session.
- Do not publish raw HAR, logs, cookies, `Authorization`, query signatures or
  presigned object-storage parameters.
- Replace header values with `<redacted>` before archiving. Retain only header
  names and non-secret schema fields.
- Discard the temporary interception CA and proxy settings after capture.

## HAR procedure

1. Close PixCake. Start a trusted local HTTPS debugging proxy such as HTTP
   Toolkit or Fiddler, and enable only its local capture certificate.
2. Launch `C:\Program Files\PixCake\pixcake_2\pixcake.exe` normally and log in
   through the application. Do not supply a copied token.
3. import a portrait, run body detection, then adjust body reshape, body
   proportions, waist/leg/neck/double-chin controls to force pose/resource
   traffic.
4. Capture `api.pixcakeai.com`, `api.pixsoda.com`, `res.pixcakeai.com`,
   `download.pixcakeai.com` and any CDN host returned by a resource response.
5. Export HAR with response content. If certificate pinning blocks capture,
   retain the exact TLS error and stop; do not bypass pinning.
6. Remove the proxy and certificate. Redact secrets.

Historical route seeds are `POST`-unknown, so do not guess the verb:

```text
/v2/api/app_resource/query
/v1/api/app_resource/group/query
/v1/api/app/user/info
```

## Required fields per candidate entry

Record the following tuple before considering an entry a pose endpoint:

```text
request.method
request.url
request.headers[name]             # values redacted in retained evidence
request.postData.mimeType
request.postData.text             # schema only; redact account/device values
response.status
response.content.mimeType
response body fields:
  resourceId
  modelId / selectedModelId
  downloadUrl / cloudSourceDownloadUrl
  md5 / sha1 / sha256
```

An entry closes the endpoint only if the response or its downloaded manifest
names the pose model and the artifact passes the tensor-contract check.

## Redacted HAR template

```json
{
  "log": {
    "version": "1.2",
    "creator": { "name": "PixCake capture", "version": "1" },
    "entries": [
      {
        "request": {
          "method": "<METHOD>",
          "url": "<URL without presigned secrets>",
          "headers": [
            { "name": "Authorization", "value": "<redacted>" },
            { "name": "Device-Id", "value": "<redacted>" },
            { "name": "Content-Type", "value": "<MIME>" }
          ],
          "postData": {
            "mimeType": "<MIME>",
            "text": "<REQUEST_BODY_SCHEMA_WITH_SECRETS_REDACTED>"
          }
        },
        "response": {
          "status": 0,
          "headers": [],
          "content": {
            "mimeType": "application/json",
            "text": "{\"resourceId\":\"<id>\",\"modelId\":\"<id>\",\"downloadUrl\":\"<url>\"}"
          }
        }
      }
    ]
  }
}
```

## curl replay template

Replay only with an independently obtained current session token and only if
the user explicitly authorizes the request. Never use a token copied from a
log. The method and body must come from the HAR, not from a guess.

```powershell
curl.exe --request "<METHOD_FROM_HAR>" "<URL_FROM_HAR>" `
  --header "Authorization: Bearer $env:PIXCAKE_TOKEN" `
  --header "Device-Id: $env:PIXCAKE_DEVICE_ID" `
  --header "Content-Type: <MIME_FROM_HAR>" `
  --data-binary "@.\request-body.bin" `
  --dump-header .\response-headers.txt `
  --trace-ascii .\request-trace.txt `
  --output .\response-body.bin `
  --fail-with-body --location
```

Model retrieval must validate the response-declared digest before use:

```powershell
curl.exe --location --fail --show-error `
  --dump-header .\model-response-headers.txt `
  --output .\bodyPose.bin "<DOWNLOAD_URL_FROM_RESPONSE>"

$actual = (Get-FileHash .\bodyPose.bin -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne "<SHA256_FROM_MANIFEST>") { throw "model digest mismatch" }
```

Do not rename an unverified MNN-family resource to `.onnx`. The artifact is
accepted only after its real format, fixed float32 NCHW input and static
`[1,K,2|3]`-family output are proven and a real-image keypoint run succeeds.

