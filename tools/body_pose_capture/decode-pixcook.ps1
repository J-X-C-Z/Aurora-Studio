[CmdletBinding()]
param(
    [string]$ContainerPath = "$env:ProgramFiles\PixCake\pixcake_2\Resources\pixcook",
    [string]$Id,
    [string]$Entry,
    [string]$StandaloneFile,
    [string]$OutFile,
    [switch]$Inventory,
    [switch]$XorPayload,
    [switch]$InspectOnly,
    [switch]$RequireModelMagic
)

$ErrorActionPreference = "Stop"
$xorKey = [byte[]](0xec, 0xcd, 0xad, 0xcb)
$effectFooterMarker = [Text.Encoding]::ASCII.GetBytes("*xixt%(g*A$l")
$blockSize = 512

function ConvertFrom-OctalSize([byte[]]$Header) {
    $text = [Text.Encoding]::ASCII.GetString($Header, 124, 12).Split([char]0)[0].Trim()
    if (-not $text) { return [int64]0 }
    return [Convert]::ToInt64($text, 8)
}

function Read-ContainerEntries([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "PixCake container not found: $Path"
    }

    $entries = New-Object System.Collections.Generic.List[object]
    $stream = [IO.File]::OpenRead($Path)
    try {
        $buffer = New-Object byte[] $blockSize
        $first = $true
        while ($true) {
            $read = $stream.Read($buffer, 0, $blockSize)
            if ($read -lt $blockSize) { break }

            $header = [byte[]]$buffer.Clone()
            if ($first) {
                for ($index = 0; $index -lt 100; $index++) {
                    $header[$index] = $header[$index] -bxor $xorKey[$index % 4]
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
                offset = [int64]$offset
                bytes = [int64]$size
                type = [Text.Encoding]::ASCII.GetString($header, 156, 1)
            })

            $padding = ($blockSize - ($size % $blockSize)) % $blockSize
            [void]$stream.Seek($offset + $size + $padding, [IO.SeekOrigin]::Begin)
        }
    }
    finally {
        $stream.Dispose()
    }
    return $entries
}

function Get-DecodedHead([string]$Path, [int64]$Offset, [int64]$Bytes, [bool]$ApplyXor) {
    $want = [int][Math]::Min([int64]4096, $Bytes)
    if ($want -le 0) { return [byte[]]@() }

    $buffer = New-Object byte[] $want
    $stream = [IO.File]::OpenRead($Path)
    try {
        [void]$stream.Seek($Offset, [IO.SeekOrigin]::Begin)
        $read = $stream.Read($buffer, 0, $want)
    }
    finally {
        $stream.Dispose()
    }

    if ($ApplyXor) {
        for ($index = 0; $index -lt $read; $index++) {
            $buffer[$index] = $buffer[$index] -bxor $xorKey[$index % 4]
        }
    }
    if ($read -lt $buffer.Length) {
        [Array]::Resize([ref]$buffer, $read)
    }
    return $buffer
}

function Get-MagicInfo([byte[]]$Bytes) {
    $headLength = [Math]::Min(64, $Bytes.Length)
    $hex = ""
    if ($headLength -gt 0) {
        $hex = (($Bytes[0..($headLength - 1)] | ForEach-Object { $_.ToString("x2") }) -join "")
    }

    $format = "unknown"
    $label = "opaque-or-encrypted"
    if ($Bytes.Length -ge 8 -and $hex.StartsWith("89504e470d0a1a0a")) {
        $format = "png"
        $label = "PNG"
    }
    elseif ($Bytes.Length -ge 2 -and $Bytes[0] -eq 0x7b -and
            @(0x09, 0x0a, 0x0d, 0x20, 0x22, 0x7d) -contains $Bytes[1]) {
        $format = "json"
        $label = "UTF-8 JSON"
    }
    elseif ($Bytes.Length -ge 4 -and
            $Bytes[0] -eq 0x1c -and $Bytes[1] -eq 0 -and $Bytes[2] -eq 0 -and $Bytes[3] -eq 0) {
        $format = "mnn"
        $label = "MNN"
    }
    elseif ($Bytes.Length -ge 8 -and
            $Bytes[4] -eq 0x54 -and $Bytes[5] -eq 0x46 -and $Bytes[6] -eq 0x4c -and $Bytes[7] -eq 0x33) {
        $format = "tflite"
        $label = "TensorFlow Lite"
    }
    elseif ($Bytes.Length -ge 4 -and
            $Bytes[0] -eq 0x08 -and ($Bytes[1] -ge 0x07 -and $Bytes[1] -le 0x0c)) {
        $format = "onnx-protobuf-candidate"
        $label = "ONNX protobuf candidate"
    }
    elseif ($Bytes.Length -ge 7 -and [Text.Encoding]::ASCII.GetString($Bytes, 0, 7) -eq "7767517") {
        $format = "ncnn-param"
        $label = "NCNN parameter text"
    }

    return [pscustomobject]@{
        format = $format
        label = $label
        head_hex = $hex
    }
}

function Test-NeuralMagic([string]$Format) {
    return @("mnn", "tflite", "onnx-protobuf-candidate") -contains $Format
}

function Get-EffectivePayloadBytes([string]$Path, [int64]$Offset, [int64]$Bytes) {
    $footerBytes = 16
    if ($Bytes -lt $footerBytes) { return $Bytes }

    $tail = New-Object byte[] $footerBytes
    $stream = [IO.File]::OpenRead($Path)
    try {
        [void]$stream.Seek($Offset + $Bytes - $footerBytes, [IO.SeekOrigin]::Begin)
        $read = $stream.Read($tail, 0, $footerBytes)
    }
    finally {
        $stream.Dispose()
    }
    if ($read -ne $footerBytes) { return $Bytes }

    for ($index = 0; $index -lt $effectFooterMarker.Length; $index++) {
        if ($tail[4 + $index] -ne $effectFooterMarker[$index]) {
            return $Bytes
        }
    }
    return $Bytes - $footerBytes
}

function Copy-DecodedRange(
    [string]$Path,
    [int64]$Offset,
    [int64]$Bytes,
    [bool]$ApplyXor,
    [string]$DestinationPath
) {
    $directory = Split-Path -Parent $DestinationPath
    if ($directory) {
        [void](New-Item -ItemType Directory -Force -Path $directory)
    }

    $source = [IO.File]::OpenRead($Path)
    $destinationStream = [IO.File]::Open(
        $DestinationPath,
        [IO.FileMode]::Create,
        [IO.FileAccess]::Write,
        [IO.FileShare]::None
    )
    try {
        [void]$source.Seek($Offset, [IO.SeekOrigin]::Begin)
        $remaining = $Bytes
        $position = [int64]0
        $buffer = New-Object byte[] (1MB)
        while ($remaining -gt 0) {
            $want = [int][Math]::Min([int64]$buffer.Length, $remaining)
            $read = $source.Read($buffer, 0, $want)
            if ($read -le 0) { throw "Unexpected EOF at offset $Offset." }
            if ($ApplyXor) {
                for ($index = 0; $index -lt $read; $index++) {
                    $buffer[$index] = $buffer[$index] -bxor $xorKey[($position + $index) % 4]
                }
            }
            $destinationStream.Write($buffer, 0, $read)
            $position += $read
            $remaining -= $read
        }
    }
    finally {
        $destinationStream.Dispose()
        $source.Dispose()
    }
}

if ($Inventory) {
    $entries = Read-ContainerEntries $ContainerPath
    $numeric = @($entries | Where-Object { $_.name -match "^\d+$" })
    $named = @($entries | Where-Object { $_.name -notmatch "^\d+$" })
    $inventory = [ordered]@{
        schema = "aurora/pixcook-decode-inventory/1"
        container = (Resolve-Path -LiteralPath $ContainerPath).Path
        container_bytes = (Get-Item -LiteralPath $ContainerPath).Length
        first_header_xor_key = "eccdcadcb"
        entries = $entries.Count
        numeric_entries = $numeric.Count
        named_entries = $named.Count
        pose_named_entries = @($named | Where-Object {
            $_.name -match "(?i)pose|body|landmark|keypoint|skeleton"
        })
    }
    $inventory | ConvertTo-Json -Depth 6
    return
}

$selected = $null
$sourcePath = $ContainerPath
$offset = [int64]0
$bytes = [int64]0
$applyXor = $false
$entryName = $null

if ($StandaloneFile) {
    if (-not (Test-Path -LiteralPath $StandaloneFile -PathType Leaf)) {
        throw "Standalone file not found: $StandaloneFile"
    }
    $sourcePath = (Resolve-Path -LiteralPath $StandaloneFile).Path
    $bytes = (Get-Item -LiteralPath $sourcePath).Length
    $entryName = (Split-Path -Leaf $sourcePath)
    $applyXor = $true
}
elseif ($Id -or $Entry) {
    $entries = Read-ContainerEntries $ContainerPath
    if ($Id) {
        $selected = $entries | Where-Object { $_.name -eq $Id } | Select-Object -First 1
        if (-not $selected) { throw "Resource ID not found: $Id" }
        $applyXor = $true
    }
    else {
        $selected = $entries | Where-Object { $_.name -eq $Entry } | Select-Object -First 1
        if (-not $selected) { throw "Container entry not found: $Entry" }
        $applyXor = [bool]$XorPayload
    }
    $offset = [int64]$selected.offset
    $bytes = [int64]$selected.bytes
    $entryName = $selected.name
}
else {
    throw "Specify -Inventory, -Id, -Entry, or -StandaloneFile."
}

if ($XorPayload) {
    $applyXor = $true
}

if (-not $InspectOnly -and -not $OutFile) {
    throw "Specify -OutFile or -InspectOnly."
}

$uncutBytes = $bytes
$bytes = Get-EffectivePayloadBytes $sourcePath $offset $bytes
$head = Get-DecodedHead $sourcePath $offset $bytes $applyXor
$magic = Get-MagicInfo $head

if ($RequireModelMagic -and -not (Test-NeuralMagic $magic.format)) {
    throw "No supported neural-model magic after decoding '$entryName': $($magic.format) ($($magic.head_hex))"
}

$writtenFile = $null
$sha256 = $null
if (-not $InspectOnly) {
    Copy-DecodedRange $sourcePath $offset $bytes $applyXor $OutFile
    $writtenFile = (Resolve-Path -LiteralPath $OutFile).Path
    $sha256 = (Get-FileHash -LiteralPath $writtenFile -Algorithm SHA256).Hash.ToLowerInvariant()
}

[ordered]@{
    schema = "aurora/pixcook-decode/1"
    entry = $entryName
    source = $sourcePath
    offset = $offset
    bytes = $bytes
    xor_applied = $applyXor
    footer_stripped = ($uncutBytes -ne $bytes)
    detected_format = $magic.format
    detected_label = $magic.label
    decoded_head_hex = $magic.head_hex
    out_file = $writtenFile
    sha256 = $sha256
    tensor_contract_verified = $false
} | ConvertTo-Json -Depth 5
