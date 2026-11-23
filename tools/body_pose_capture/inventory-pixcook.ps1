[CmdletBinding()]
param(
    [string]$ContainerPath = "$env:ProgramFiles\PixCake\pixcake_2\resources\pixcook",
    [string]$OutFile = (Join-Path ([IO.Path]::GetTempPath()) "aurora-body-pose-capture\pixcook-index.json"),
    [string]$Filter = "pose|body|skeleton|keypoint|model"
)

$ErrorActionPreference = "Stop"
$key = [byte[]](0xec, 0xcd, 0xad, 0xcb)
$blockSize = 512

if (-not (Test-Path -LiteralPath $ContainerPath -PathType Leaf)) {
    throw "PixCake container not found: $ContainerPath"
}

function ConvertFrom-OctalSize([byte[]]$Header) {
    $text = [Text.Encoding]::ASCII.GetString($Header, 124, 12).Split([char]0)[0].Trim()
    if (-not $text) { return [int64]0 }
    return [Convert]::ToInt64($text, 8)
}

$entries = New-Object System.Collections.Generic.List[object]
$stream = [IO.File]::OpenRead($ContainerPath)
try {
    $buffer = New-Object byte[] $blockSize
    $first = $true
    while ($true) {
        $read = $stream.Read($buffer, 0, $blockSize)
        if ($read -lt $blockSize) { break }

        $header = [byte[]]$buffer.Clone()
        if ($first) {
            for ($index = 0; $index -lt 100; $index++) {
                $header[$index] = $header[$index] -bxor $key[$index % 4]
            }
            $first = $false
        }

        $name = [Text.Encoding]::UTF8.GetString($header, 0, 100).Split([char]0)[0]
        if (-not $name) { break }
        $magic = [Text.Encoding]::ASCII.GetString($header, 257, 8).Split([char]0)[0]
        if ($magic -notmatch "ustar") {
            throw "Entry '$name' is not a ustar header."
        }

        $size = ConvertFrom-OctalSize $header
        $offset = $stream.Position
        $entries.Add([pscustomobject]@{
            name = $name
            offset = $offset
            bytes = $size
            type = [Text.Encoding]::ASCII.GetString($header, 156, 1)
        })

        $padding = ($blockSize - ($size % $blockSize)) % $blockSize
        [void]$stream.Seek($offset + $size + $padding, [IO.SeekOrigin]::Begin)
    }
}
finally {
    $stream.Dispose()
}

$numeric = @($entries | Where-Object { $_.name -match "^\d+$" })
$named = @($entries | Where-Object { $_.name -notmatch "^\d+$" })
$matches = @($entries | Where-Object { $_.name -match $Filter })

$document = [ordered]@{
    schema = "aurora/pixcook-index/1"
    container = (Resolve-Path -LiteralPath $ContainerPath).Path
    container_bytes = (Get-Item -LiteralPath $ContainerPath).Length
    first_header_xor_key = "eccdcadcb"
    entries = $entries.Count
    numeric_entries = $numeric
    named_entries = $named
    name_matches = $matches
    body_pose_model_id = $null
    status = "indexed"
    note = "A name or size match is not model-identity proof. Tensor I/O must be verified before runtime use."
}

$directory = Split-Path -Parent $OutFile
if ($directory) { [void](New-Item -ItemType Directory -Force -Path $directory) }
$document | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutFile -Encoding UTF8
Write-Host ("indexed {0} entries ({1} numeric, {2} named) -> {3}" -f $entries.Count, $numeric.Count, $named.Count, $OutFile)
