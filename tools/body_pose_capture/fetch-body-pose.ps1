[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,
    [string]$OutDirectory = (Join-Path ([IO.Path]::GetTempPath()) "aurora-body-pose-capture\models"),
    [switch]$AllowInsecureHttp
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    throw "Acquisition manifest not found: $Manifest"
}
$spec = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
if (-not $spec.artifacts -or @($spec.artifacts).Count -eq 0) {
    throw "Manifest contains no artifacts."
}

[void](New-Item -ItemType Directory -Force -Path $OutDirectory)
foreach ($artifact in @($spec.artifacts)) {
    $modelId = [string]$artifact.model_id
    $fileName = [string]$artifact.file_name
    $url = [string]$artifact.url
    $expected = ([string]$artifact.sha256).Trim().ToLowerInvariant()

    if (-not $modelId -or $modelId -match "required|unknown|todo") { throw "Manifest model_id is unresolved." }
    if (-not $fileName -or $fileName -ne (Split-Path -Leaf $fileName)) { throw "file_name must be a leaf filename." }
    if ($expected -notmatch "^[0-9a-f]{64}$") { throw "Artifact $modelId has no valid SHA-256." }
    if (-not $url) { throw "Artifact $modelId has no URL." }

    $target = Join-Path $OutDirectory $fileName
    $temp = "$target.part"
    if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Force }

    $requestKind = $null
    $response = $null
    $requestHeaderNames = @()
    try {
        if ($url -match "^https?://") {
            $uri = [Uri]$url
            if ($uri.Scheme -eq "http" -and -not $AllowInsecureHttp) {
                throw "HTTP is disabled; use HTTPS or pass -AllowInsecureHttp explicitly."
            }
            $headers = @{}
            if ($artifact.headers) {
                foreach ($property in $artifact.headers.PSObject.Properties) {
                    $headers[$property.Name] = [string]$property.Value
                }
            }
            $requestHeaderNames = @($headers.Keys | Sort-Object)
            $requestKind = "http"
            $response = Invoke-WebRequest -UseBasicParsing -Uri $url -Method Get -Headers $headers -OutFile $temp -PassThru
        }
        elseif ($url -match "^file://") {
            $requestKind = "file-uri"
            $sourcePath = ([Uri]$url).LocalPath
            Copy-Item -LiteralPath $sourcePath -Destination $temp
        }
        else {
            $requestKind = "local-file"
            if (-not (Test-Path -LiteralPath $url -PathType Leaf)) { throw "Local artifact not found: $url" }
            Copy-Item -LiteralPath $url -Destination $temp
        }

        $actual = (Get-FileHash -LiteralPath $temp -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expected) {
            throw "SHA-256 mismatch for $fileName`: got $actual expected $expected"
        }
        Move-Item -LiteralPath $temp -Destination $target -Force
        "$actual  $fileName" | Set-Content -LiteralPath "$target.sha256" -Encoding ASCII

        $responseHeaders = [ordered]@{}
        $status = $null
        $contentType = $null
        if ($response) {
            $status = [int]$response.StatusCode
            foreach ($key in $response.Headers.Keys) {
                $responseHeaders[[string]$key] = [string]$response.Headers[$key]
            }
            $contentType = [string]$response.Headers["Content-Type"]
        }
        $receipt = [ordered]@{
            schema = "aurora/body-pose-capture/1"
            model_id = $modelId
            file = (Resolve-Path -LiteralPath $target).Path
            bytes = (Get-Item -LiteralPath $target).Length
            sha256 = $actual
            request = [ordered]@{
                kind = $requestKind
                method = if ($response) { "GET" } else { "COPY" }
                url = $url
                header_names = $requestHeaderNames
            }
            response = [ordered]@{
                status = $status
                content_type = $contentType
                headers = $responseHeaders
            }
            tensor_contract_verified = $false
        }
        $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$target.capture.json" -Encoding UTF8
        Write-Host ("captured {0}: {1} bytes sha256={2}" -f $fileName, $receipt.bytes, $actual)
    }
    finally {
        if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Force }
    }
}
