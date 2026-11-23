[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Id,
    [string]$ContainerPath = "$env:ProgramFiles\PixCake\pixcake_2\resources\pixcook",
    [string]$IndexFile = (Join-Path ([IO.Path]::GetTempPath()) "aurora-body-pose-capture\pixcook-index.json"),
    [string]$OutDirectory = (Join-Path ([IO.Path]::GetTempPath()) "aurora-body-pose-capture\models"),
    [string]$ExpectedSha256
)

$ErrorActionPreference = "Stop"
if ($Id -notmatch "^\d+$") { throw "PixCake model ID must be numeric: $Id" }
if (-not (Test-Path -LiteralPath $IndexFile -PathType Leaf)) {
    throw "Index not found. Run inventory-pixcook.ps1 first: $IndexFile"
}
if (-not (Test-Path -LiteralPath $ContainerPath -PathType Leaf)) {
    throw "PixCake container not found: $ContainerPath"
}

$index = Get-Content -LiteralPath $IndexFile -Raw | ConvertFrom-Json
$entry = $index.numeric_entries | Where-Object { $_.name -eq $Id } | Select-Object -First 1
if (-not $entry) { throw "Model/resource ID $Id is not in the container index." }
if ([int64]$entry.bytes -le 0) { throw "Model/resource ID $Id is empty." }

[void](New-Item -ItemType Directory -Force -Path $OutDirectory)
$target = Join-Path $OutDirectory ("{0}.bin" -f $Id)
$source = [IO.File]::OpenRead($ContainerPath)
$destination = [IO.File]::Open($target, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
try {
    [void]$source.Seek([int64]$entry.offset, [IO.SeekOrigin]::Begin)
    $remaining = [int64]$entry.bytes
    $buffer = New-Object byte[] (1MB)
    while ($remaining -gt 0) {
        $want = [int][Math]::Min([int64]$buffer.Length, $remaining)
        $read = $source.Read($buffer, 0, $want)
        if ($read -le 0) { throw "Unexpected EOF while extracting ID $Id." }
        $destination.Write($buffer, 0, $read)
        $remaining -= $read
    }
}
finally {
    $destination.Dispose()
    $source.Dispose()
}

$actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ExpectedSha256) {
    $expected = $ExpectedSha256.Trim().ToLowerInvariant()
    if ($expected -notmatch "^[0-9a-f]{64}$") { throw "ExpectedSha256 must contain 64 hex digits." }
    if ($actual -ne $expected) {
        Remove-Item -LiteralPath $target -Force
        throw "SHA-256 mismatch for ID ${Id}: got $actual expected $expected"
    }
}

$sidecar = "$target.sha256"
"$actual  $(Split-Path -Leaf $target)" | Set-Content -LiteralPath $sidecar -Encoding ASCII
$receipt = [ordered]@{
    schema = "aurora/pixcook-capture/1"
    model_id = $Id
    container = (Resolve-Path -LiteralPath $ContainerPath).Path
    offset = [int64]$entry.offset
    bytes = [int64]$entry.bytes
    file = (Resolve-Path -LiteralPath $target).Path
    sha256 = $actual
    verified_expected_sha256 = [bool]$ExpectedSha256
    tensor_contract_verified = $false
}
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$target.capture.json" -Encoding UTF8
Write-Host ("extracted ID {0} ({1} bytes) sha256={2}" -f $Id, $entry.bytes, $actual)
