[CmdletBinding()]
param(
    [string]$ContainerPath = "$env:ProgramFiles\PixCake\pixcake_2\Resources\pixcook",
    [string]$OutFile = (Join-Path ([IO.Path]::GetTempPath()) "aurora-pose-b7\payload-classification.json"),
    [int]$SampleBytes = 8192
)

$ErrorActionPreference = "Stop"
$blockSize = 512
$outerXorKey = [byte[]](0xec, 0xcd, 0xad, 0xcb)
$poseTokens = @(
    "pose", "keypoint", "landmark", "joint", "skeleton",
    "bodyreshape", "aibody", "detector", "human"
)

Add-Type -TypeDefinition @"
using System;

public static class AuroraPoseScan {
    public static byte[] Xor(byte[] input, int key) {
        byte[] output = new byte[input.Length];
        for (int i = 0; i < input.Length; i++) output[i] = (byte)(input[i] ^ key);
        return output;
    }

    public static byte[] Xor4(byte[] input, byte[] key) {
        byte[] output = new byte[input.Length];
        for (int i = 0; i < input.Length; i++) output[i] = (byte)(input[i] ^ key[i % 4]);
        return output;
    }

    public static double Entropy(byte[] input) {
        if (input.Length == 0) return 0.0;
        int[] counts = new int[256];
        foreach (byte value in input) counts[value]++;
        double entropy = 0.0;
        foreach (int count in counts) {
            if (count == 0) continue;
            double probability = count / (double)input.Length;
            entropy -= probability * Math.Log(probability, 2.0);
        }
        return Math.Round(entropy, 4);
    }

    public static double PrintableRatio(byte[] input) {
        if (input.Length == 0) return 0.0;
        int printable = 0;
        foreach (byte value in input) {
            if ((value >= 0x20 && value <= 0x7e) || value == 0x09 || value == 0x0a || value == 0x0d) printable++;
        }
        return Math.Round(printable / (double)input.Length, 4);
    }
}
"@

function Convert-FromOctalSize([byte[]]$Header) {
    $text = [Text.Encoding]::ASCII.GetString($Header, 124, 12).Split([char]0)[0].Trim()
    if (-not $text) { return [int64]0 }
    return [Convert]::ToInt64($text, 8)
}

function Read-ContainerEntries([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "PixCake container not found: $Path"
    }

    $entries = [Collections.Generic.List[object]]::new()
    $stream = [IO.File]::OpenRead($Path)
    try {
        $buffer = [byte[]]::new($blockSize)
        $first = $true
        while ($true) {
            $read = $stream.Read($buffer, 0, $blockSize)
            if ($read -lt $blockSize) { break }

            $header = [byte[]]$buffer.Clone()
            if ($first) {
                for ($index = 0; $index -lt 100; $index++) {
                    $header[$index] = $header[$index] -bxor $outerXorKey[$index % 4]
                }
                $first = $false
            }

            $name = [Text.Encoding]::UTF8.GetString($header, 0, 100).Split([char]0)[0]
            if (-not $name) { break }
            $magic = [Text.Encoding]::ASCII.GetString($header, 257, 8).Split([char]0)[0]
            if ($magic -notmatch "ustar") { throw "Entry '$name' is not a ustar header." }

            $size = Convert-FromOctalSize $header
            $offset = $stream.Position
            $entries.Add([pscustomobject]@{
                name = $name
                offset = [int64]$offset
                bytes = [int64]$size
                type = [Text.Encoding]::ASCII.GetString($header, 156, 1)
                numeric = [bool]($name -match "^\d+$")
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

function Read-BytesAt([IO.FileStream]$Stream, [int64]$Offset, [int]$Wanted) {
    $bytes = [byte[]]::new($Wanted)
    [void]$Stream.Seek($Offset, [IO.SeekOrigin]::Begin)
    $read = $Stream.Read($bytes, 0, $Wanted)
    if ($read -eq $Wanted) { return $bytes }
    [Array]::Resize([ref]$bytes, $read)
    return $bytes
}

function Get-Hex([byte[]]$Bytes, [int]$Limit = 32) {
    if ($Bytes.Length -eq 0) { return "" }
    $length = [Math]::Min($Limit, $Bytes.Length)
    return (($Bytes[0..($length - 1)] | ForEach-Object { $_.ToString("x2") }) -join "")
}

function Get-Xor([byte[]]$Bytes, [int]$Key) {
    return [AuroraPoseScan]::Xor($Bytes, $Key)
}

function Get-OuterXor([byte[]]$Bytes, [int]$Offset = 0) {
    if ($Offset -eq 0) { return [AuroraPoseScan]::Xor4($Bytes, $outerXorKey) }
    $rotated = [byte[]]($outerXorKey[$Offset..3] + $outerXorKey[0..($Offset - 1)])
    return [AuroraPoseScan]::Xor4($Bytes, $rotated)
}

function Get-Entropy([byte[]]$Bytes) {
    return [AuroraPoseScan]::Entropy($Bytes)
}

function Get-PrintableRatio([byte[]]$Bytes) {
    return [AuroraPoseScan]::PrintableRatio($Bytes)
}

function Read-Varint([byte[]]$Bytes, [ref]$Position, [int]$Limit) {
    [uint64]$value = 0
    $shift = 0
    $positionValue = [int]$Position.Value
    while ($positionValue -lt $Limit -and $shift -le 63) {
        $byte = $Bytes[$positionValue]
        $positionValue++
        $value = $value -bor ([uint64]($byte -band 0x7f) -shl $shift)
        if (($byte -band 0x80) -eq 0) {
            $Position.Value = $positionValue
            return $value
        }
        $shift += 7
    }
    throw "invalid-varint"
}

function Test-OnnxHeader([byte[]]$Bytes) {
    if ($Bytes.Length -lt 4 -or $Bytes[0] -ne 0x08) { return $false }
    $position = [ref]0
    try {
        $tag = Read-Varint $Bytes $position $Bytes.Length
        if (($tag -shr 3) -ne 1 -or ($tag -band 7) -ne 0) { return $false }
        $irVersion = Read-Varint $Bytes $position $Bytes.Length
        if ($irVersion -lt 1 -or $irVersion -gt 32) { return $false }

        $hasProducer = $false
        $hasGraph = $false
        while ($position.Value -lt $Bytes.Length) {
            $fieldStart = $position.Value
            $fieldTag = Read-Varint $Bytes $position $Bytes.Length
            $field = [int]($fieldTag -shr 3)
            $wire = [int]($fieldTag -band 7)
            if ($field -eq 0 -or $field -gt 32) { return $false }
            if ($wire -eq 0) {
                [void](Read-Varint $Bytes $position $Bytes.Length)
            }
            elseif ($wire -eq 1) {
                if ($position.Value + 8 -gt $Bytes.Length) { return $false }
                $position.Value += 8
            }
            elseif ($wire -eq 2) {
                $length = Read-Varint $Bytes $position $Bytes.Length
                if ($length -gt [uint64]($Bytes.Length - $position.Value)) {
                    return [bool]($field -eq 7 -and $length -gt 0 -and $hasProducer)
                }
                if ($field -eq 2) {
                    $producer = [Text.Encoding]::UTF8.GetString(
                        $Bytes, $position.Value, [int]$length
                    )
                    if ($producer -match "^[\x20-\x7e]{1,256}$") { $hasProducer = $true }
                }
                if ($field -eq 7 -and $length -gt 0) { $hasGraph = $true }
                $position.Value += [int]$length
            }
            elseif ($wire -eq 5) {
                if ($position.Value + 4 -gt $Bytes.Length) { return $false }
                $position.Value += 4
            }
            else {
                return $false
            }
            if ($field -eq 7) { break }
            if ($position.Value -le $fieldStart) { return $false }
        }
        return [bool]($hasProducer -and $hasGraph)
    }
    catch {
        return $false
    }
}

function Get-FlatbufferInfo([byte[]]$Bytes) {
    if ($Bytes.Length -lt 12) { return $null }
    $root = [BitConverter]::ToUInt32($Bytes, 0)
    if ($root -lt 4 -or ($root % 4) -ne 0 -or $root + 4 -gt $Bytes.Length) { return $null }
    $soffset = [BitConverter]::ToInt32($Bytes, [int]$root)
    $vtable = [int64]$root - $soffset
    if ($vtable -lt 0 -or $vtable + 4 -gt $Bytes.Length) { return $null }
    $vtableSize = [BitConverter]::ToUInt16($Bytes, [int]$vtable)
    $objectSize = [BitConverter]::ToUInt16($Bytes, [int]($vtable + 2))
    if ($vtableSize -lt 4 -or ($vtableSize % 2) -ne 0 -or
        $vtable + $vtableSize -gt $Bytes.Length -or
        $objectSize -lt 4 -or $root + $objectSize -gt $Bytes.Length) {
        return $null
    }
    return [pscustomobject]@{ root = $root; vtable = $vtable; vtable_size = $vtableSize; object_size = $objectSize }
}

function Get-FormatInfo([byte[]]$Bytes) {
    $head = Get-Hex $Bytes
    if ($Bytes.Length -ge 8 -and (Get-Hex $Bytes 8) -eq "89504e470d0a1a0a") {
        return [pscustomobject]@{ format = "png"; tensor_header = $null; head_hex = $head }
    }
    if ($Bytes.Length -ge 2 -and $Bytes[0] -eq 0x7b -and
        @(0x09, 0x0a, 0x0d, 0x20, 0x22, 0x7d) -contains $Bytes[1]) {
        return [pscustomobject]@{ format = "json"; tensor_header = $null; head_hex = $head }
    }
    if ($Bytes.Length -ge 4 -and $Bytes[0] -eq 0x1c -and $Bytes[1] -eq 0 -and
        $Bytes[2] -eq 0 -and $Bytes[3] -eq 0) {
        return [pscustomobject]@{ format = "mnn"; tensor_header = "MNN magic 1c000000"; head_hex = $head }
    }
    if ($Bytes.Length -ge 8 -and [Text.Encoding]::ASCII.GetString($Bytes, 4, 4) -eq "TFL3") {
        $flatbuffer = Get-FlatbufferInfo $Bytes
        if ($null -ne $flatbuffer) {
            return [pscustomobject]@{ format = "tflite-flatbuffer"; tensor_header = "TFL3 + valid root/vtable"; head_hex = $head }
        }
    }
    if (Test-OnnxHeader $Bytes) {
        return [pscustomobject]@{ format = "onnx-protobuf"; tensor_header = "ir_version + producer + graph"; head_hex = $head }
    }
    $flatbuffer = Get-FlatbufferInfo $Bytes
    if ($null -ne $flatbuffer) {
        return [pscustomobject]@{ format = "flatbuffer-structural"; tensor_header = "valid root/vtable"; head_hex = $head }
    }
    if ($Bytes.Length -ge 7 -and [Text.Encoding]::ASCII.GetString($Bytes, 0, 7) -eq "7767517") {
        return [pscustomobject]@{ format = "ncnn-param"; tensor_header = $null; head_hex = $head }
    }
    if ($Bytes.Length -ge 4 -and $Bytes[0] -eq 0x50 -and $Bytes[1] -eq 0x4b -and
        $Bytes[2] -eq 0x03 -and $Bytes[3] -eq 0x04) {
        return [pscustomobject]@{ format = "zip"; tensor_header = $null; head_hex = $head }
    }
    if ($Bytes.Length -ge 2 -and $Bytes[0] -eq 0x08 -and $Bytes[1] -ge 0x07 -and $Bytes[1] -le 0x0c) {
        return [pscustomobject]@{ format = "weak-onnx-prefix"; tensor_header = $null; head_hex = $head }
    }
    return [pscustomobject]@{ format = "opaque"; tensor_header = $null; head_hex = $head }
}

function Get-SingleByteXorModels([byte[]]$Bytes) {
    $hits = [Collections.Generic.List[object]]::new()
    if ($Bytes.Length -lt 8) { return $hits }
    $candidateKeys = [Collections.Generic.List[int]]::new()
    for ($key = 1; $key -le 255; $key++) {
        if (($Bytes[0] -bxor $key) -eq 0x1c -and
            ($Bytes[1] -bxor $key) -eq 0 -and
            ($Bytes[2] -bxor $key) -eq 0 -and
            ($Bytes[3] -bxor $key) -eq 0) {
            $candidateKeys.Add($key)
        }
    }
    $tfliteKey = $Bytes[4] -bxor 0x54
    if ($tfliteKey -ne 0 -and
        ($Bytes[5] -bxor $tfliteKey) -eq 0x46 -and
        ($Bytes[6] -bxor $tfliteKey) -eq 0x4c -and
        ($Bytes[7] -bxor $tfliteKey) -eq 0x33) {
        $candidateKeys.Add($tfliteKey)
    }
    $onnxKey = $Bytes[0] -bxor 0x08
    if ($onnxKey -ne 0) { $candidateKeys.Add($onnxKey) }
    foreach ($key in ($candidateKeys | Sort-Object -Unique)) {
        $transformed = Get-Xor $Bytes $key
        $info = Get-FormatInfo $transformed
        if (@("mnn", "tflite-flatbuffer", "onnx-protobuf", "flatbuffer-structural") -contains $info.format) {
            $hits.Add([ordered]@{ key_hex = $key.ToString("x2"); format = $info.format; head_hex = $info.head_hex })
        }
    }
    return $hits
}

function Get-PoseTokens([byte[]]$Bytes) {
    $text = [Text.Encoding]::ASCII.GetString($Bytes).ToLowerInvariant()
    return @($poseTokens | Where-Object { $text.Contains($_) } | Sort-Object -Unique)
}

function Get-SampleBytes([IO.FileStream]$Stream, [object]$Entry, [int]$Wanted) {
    $length = [int][Math]::Min([int64]$Wanted, $Entry.bytes)
    if ($length -le 0) { return [byte[]]@() }
    $parts = [Collections.Generic.List[byte]]::new()
    $offsets = @([int64]0)
    if ($Entry.bytes -gt $length) {
        $offsets += [int64][Math]::Max(0, [Math]::Floor(($Entry.bytes - $length) / 2))
        $offsets += [int64]($Entry.bytes - $length)
    }
    foreach ($relative in ($offsets | Select-Object -Unique)) {
        $bytes = Read-BytesAt $Stream ($Entry.offset + $relative) $length
        $parts.AddRange([byte[]]$bytes)
    }
    return $parts.ToArray()
}

if (-not (Test-Path -LiteralPath $ContainerPath -PathType Leaf)) {
    throw "PixCake container not found: $ContainerPath"
}

$entries = Read-ContainerEntries $ContainerPath
$numericEntries = @($entries | Where-Object { $_.numeric -and $_.bytes -gt 0 -and $_.type -eq "0" })
$namedEntries = @($entries | Where-Object { -not $_.numeric })
$poseShaders = @($namedEntries | Where-Object {
    $_.type -eq "0" -and $_.name -match "(?i)(PoseRefine|AIBodyReshape|HumanDetected)" -and
    $_.name -match "\.(frag|vert)$"
})
$rows = [Collections.Generic.List[object]]::new()
$modelCandidates = [Collections.Generic.List[object]]::new()
$poseTokenHits = [Collections.Generic.List[object]]::new()
$weakOnnxPrefixes = [Collections.Generic.List[object]]::new()
$formatCounts = @{}
$classCounts = @{}

$stream = [IO.File]::OpenRead($ContainerPath)
try {
    foreach ($entry in $numericEntries) {
        $headLength = [int][Math]::Min(65536, $entry.bytes)
        $head = Read-BytesAt $stream $entry.offset $headLength
        $outerHead = Get-OuterXor $head
        $sample = Get-SampleBytes $stream $entry $SampleBytes

        $raw = Get-FormatInfo $head
        $outer = Get-FormatInfo $outerHead
        $xorModels = @(Get-SingleByteXorModels $head)
        $rawTokens = @(Get-PoseTokens $sample)
        $outerTokens = @(Get-PoseTokens (Get-OuterXor $sample))
        $tokens = @($rawTokens + $outerTokens | Sort-Object -Unique)
        $entropy = Get-Entropy $sample
        $printable = Get-PrintableRatio $sample

        $modelLike = @(
            $raw.format -in @("mnn", "tflite-flatbuffer", "onnx-protobuf") -or
            $outer.format -in @("mnn", "tflite-flatbuffer", "onnx-protobuf") -or
            $xorModels.Count -gt 0
        ) -contains $true

        if ($raw.format -eq "weak-onnx-prefix" -or $outer.format -eq "weak-onnx-prefix") {
            $weakOnnxPrefixes.Add([ordered]@{
                resource_id = $entry.name
                offset = $entry.offset
                bytes = $entry.bytes
                raw_head_hex = $raw.head_hex
                outer_xor_head_hex = $outer.head_hex
                rejected = "prefix-only; no validated ONNX top-level protobuf"
            })
        }
        if ($modelLike) {
            $candidate = [ordered]@{
                resource_id = $entry.name
                offset = $entry.offset
                bytes = $entry.bytes
                raw_format = $raw.format
                outer_xor_format = $outer.format
                single_byte_xor_models = $xorModels
                head_hex = $raw.head_hex
            }
            $modelCandidates.Add($candidate)
        }
        if ($tokens.Count -gt 0) {
            $poseTokenHits.Add([ordered]@{
                resource_id = $entry.name
                offset = $entry.offset
                bytes = $entry.bytes
                tokens = $tokens
                model_like = $modelLike
            })
        }

        $contentClass = if ($modelLike) {
            "neural-model-candidate"
        }
        elseif ($raw.format -ne "opaque" -or $outer.format -ne "opaque") {
            "recognized-non-model"
        }
        elseif ($entropy -ge 7.75) {
            "high-entropy-opaque"
        }
        elseif ($printable -ge 0.75) {
            "text-or-obfuscated"
        }
        else {
            "binary-opaque"
        }
        $formatKey = if ($raw.format -ne "opaque") { "raw:$($raw.format)" } else { "outer:$($outer.format)" }
        if ($raw.format -eq "opaque" -and $outer.format -eq "opaque") { $formatKey = "opaque" }
        if (-not $formatCounts.ContainsKey($formatKey)) { $formatCounts[$formatKey] = 0 }
        $formatCounts[$formatKey]++
        if (-not $classCounts.ContainsKey($contentClass)) { $classCounts[$contentClass] = 0 }
        $classCounts[$contentClass]++

        $rows.Add([ordered]@{
            resource_id = $entry.name
            offset = $entry.offset
            bytes = $entry.bytes
            class = $contentClass
            raw_format = $raw.format
            outer_xor_format = $outer.format
            outer_xor_head_hex = $outer.head_hex
            entropy = $entropy
            printable_ratio = $printable
            pose_tokens = $tokens
            tensor_header = if ($raw.tensor_header) { $raw.tensor_header } elseif ($outer.tensor_header) { $outer.tensor_header } else { $null }
            single_byte_xor_models = $xorModels
            head_hex = $raw.head_hex
        })
    }
}
finally {
    $stream.Dispose()
}

$verifiedModel = $null
$status = "blocked-no-pose-model"
$poseConclusion = "No numeric entry passed MNN, TFLite, or strict ONNX structure checks, so no payload can be linked to a pose/landmark network."
if ($modelCandidates.Count -gt 0) {
    $poseMatched = @($poseTokenHits | Where-Object { $_.model_like })
    if ($poseMatched.Count -eq 0) {
        $poseConclusion = "Neural-format candidates exist, but none exposes pose/landmark tensor or layer labels; model identity remains unproven."
    }
}

$manifest = [ordered]@{
    schema = "aurora/pixcake-pose-payload-classification/1"
    lane = "B7"
    base_commit = "eececfd"
    prerequisite_head = "adcc32b"
    status = $status
    container = [ordered]@{
        path = (Resolve-Path -LiteralPath $ContainerPath).Path
        bytes = (Get-Item -LiteralPath $ContainerPath).Length
        entries = $entries.Count
        numeric_entries = $numericEntries.Count
        named_entries = $namedEntries.Count
        pose_shader_entries = $poseShaders.Count
    }
    scan = [ordered]@{
        sample_bytes_per_region = $SampleBytes
        regions = "head, middle, tail"
        transforms = @("raw", "outer-xor-eccdcadcb", "single-byte-xor-01..ff")
        format_counts = $formatCounts
        class_counts = $classCounts
        weak_onnx_prefix_candidates = $weakOnnxPrefixes
        neural_model_candidates = $modelCandidates
        pose_token_hits = $poseTokenHits
    }
    evidence = [ordered]@{
        pose_shader_entries = $poseShaders
        conclusion = $poseConclusion
        shader_boundary = "PoseRefine shaders prove downstream displacement math only; they contain no weights, tensor contract, or landmark order."
    }
    pose_model = $verifiedModel
    runtime = [ordered]@{
        model_connected = $false
        integration = "not-attempted-no-verified-model"
        exact_contract = "SHA-256 manifest + supported model format + exact tensor I/O + landmark order + real keypoint output"
    }
    blockers = @(
        "No numeric resource passed strict MNN, TFLite FlatBuffer, or ONNX protobuf header validation.",
        "Weak 08 07 prefixes do not establish ONNX and are retained only as rejected candidates.",
        "No static metadata links any numeric resource to PoseRefine, AIBodyReshapeDetectorV3, or the native 69-point order.",
        "Encrypted or opaque model bytes cannot be named, hashed as a pose model, or loaded by B runtime without a proven transform and identity."
    )
    entries = $rows
}

$directory = Split-Path -Parent $OutFile
if ($directory) { [void](New-Item -ItemType Directory -Force -Path $directory) }
$manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $OutFile -Encoding UTF8
[ordered]@{
    out_file = (Resolve-Path -LiteralPath $OutFile).Path
    status = $status
    numeric_entries = $numericEntries.Count
    opaque_entries = $numericEntries.Count - ($formatCounts.GetEnumerator() | Where-Object { $_.Key -ne "opaque" } | Measure-Object -Property Value -Sum).Sum
    format_counts = $formatCounts
    class_counts = $classCounts
    neural_model_candidates = $modelCandidates.Count
    pose_token_hits = $poseTokenHits.Count
    pose_shaders = $poseShaders.Count
} | ConvertTo-Json -Depth 5
