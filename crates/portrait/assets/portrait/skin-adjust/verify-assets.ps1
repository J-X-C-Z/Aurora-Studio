$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifest = Get-Content -LiteralPath (Join-Path $root 'manifest.json') -Raw | ConvertFrom-Json
$failures = @()
foreach ($asset in $manifest.assets) {
    $path = Join-Path $root $asset.file
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ($hash -ne $asset.sha256 -or (Get-Item -LiteralPath $path).Length -ne $asset.bytes) {
        $failures += $asset.file
    }
}
foreach ($field in @('skin_choices', 'color_choose')) {
    $entry = $manifest.$field
    if ($entry) {
        $path = Join-Path $root $entry.file
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $entry.sha256) {
            $failures += $entry.file
        }
    }
}
if ($failures.Count -ne 0) { throw ('asset mismatch: ' + ($failures -join ',')) }
Write-Output ('assets OK: ' + $manifest.assets.Count + ' LUTs')