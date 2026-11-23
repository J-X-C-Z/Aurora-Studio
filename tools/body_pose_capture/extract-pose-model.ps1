[CmdletBinding()]
param(
    [string]$ContainerPath = "$env:ProgramFiles\PixCake\pixcake_2\Resources\pixcook",
    [string]$PoseRefinePath = "$env:ProgramFiles\PixCake\pixcake_2\Resources\effect\config\PoseRefine\effect.json",
    [string]$SubPoseRefinePath = "$env:ProgramFiles\PixCake\pixcake_2\Resources\effect\config\SubPoseRefine\effect.json",
    [string]$NetworkPayloadPath,
    [string]$OutFile = (Join-Path ([IO.Path]::GetTempPath()) "aurora-pose-b6\pose-static.json"),
    [string]$CandidateOutDirectory,
    [switch]$ExtractCandidates,
    [switch]$RequireNeuralCandidate
)

$ErrorActionPreference = "Stop"
$xorKey = [byte[]](0xec, 0xcd, 0xad, 0xcb)
$effectFooterMarker = [Text.Encoding]::ASCII.GetBytes("*xixt%(g*A$l")
$blockSize = 512
$neuralFormats = @("mnn", "tflite", "onnx-protobuf-candidate")

function Get-Sha256([byte[]]$Bytes) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($Bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

function Get-FileSha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Convert-FromOctalSize([byte[]]$Header) {
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

function Get-ContainerHead([IO.FileStream]$Stream, [object]$Entry) {
    $length = [int][Math]::Min([int64]64, $Entry.bytes)
    if ($length -le 0) { return [byte[]]@() }
    $bytes = New-Object byte[] $length
    [void]$Stream.Seek([int64]$Entry.offset, [IO.SeekOrigin]::Begin)
    $read = $Stream.Read($bytes, 0, $length)
    if ($read -eq $length) { return $bytes }
    [Array]::Resize([ref]$bytes, $read)
    return $bytes
}

function Get-MagicInfo([byte[]]$Bytes) {
    $hex = ""
    if ($Bytes.Length -gt 0) {
        $hex = (($Bytes | ForEach-Object { $_.ToString("x2") }) -join "")
    }
    $format = "empty-or-unknown"
    if ($Bytes.Length -ge 8 -and $hex.StartsWith("89504e470d0a1a0a")) {
        $format = "png"
    }
    elseif ($Bytes.Length -ge 2 -and $Bytes[0] -eq 0x7b -and
            @(0x09, 0x0a, 0x0d, 0x20, 0x22, 0x7d) -contains $Bytes[1]) {
        $format = "json"
    }
    elseif ($Bytes.Length -ge 4 -and
            $Bytes[0] -eq 0x1c -and $Bytes[1] -eq 0 -and $Bytes[2] -eq 0 -and $Bytes[3] -eq 0) {
        $format = "mnn"
    }
    elseif ($Bytes.Length -ge 8 -and
            $Bytes[4] -eq 0x54 -and $Bytes[5] -eq 0x46 -and $Bytes[6] -eq 0x4c -and $Bytes[7] -eq 0x33) {
        $format = "tflite"
    }
    elseif ($Bytes.Length -ge 4 -and
            $Bytes[0] -eq 0x08 -and ($Bytes[1] -ge 0x07 -and $Bytes[1] -le 0x0c)) {
        $tag = [int]$Bytes[2]
        $fieldNumber = $tag -shr 3
        $wireType = $tag -band 7
        $printable = 0
        $sampleLength = [Math]::Min(64, $Bytes.Length)
        for ($index = 0; $index -lt $sampleLength; $index++) {
            if ($Bytes[$index] -ge 0x20 -and $Bytes[$index] -le 0x7e) { $printable++ }
        }
        if ($fieldNumber -gt 0 -and $wireType -le 5 -and
            ($printable / [double]$sampleLength) -lt 0.80) {
            $format = "onnx-protobuf-candidate"
        }
    }
    elseif ($Bytes.Length -ge 7 -and [Text.Encoding]::ASCII.GetString($Bytes, 0, 7) -eq "7767517") {
        $format = "ncnn-param"
    }
    elseif ($Bytes.Length -ge 4 -and $Bytes[0] -eq 0x50 -and $Bytes[1] -eq 0x4b -and
            $Bytes[2] -eq 0x03 -and $Bytes[3] -eq 0x04) {
        $format = "zip"
    }
    [pscustomobject]@{ format = $format; head_hex = $hex }
}

function Get-UnclosedObjectCount([string]$Text) {
    $depth = 0
    $inString = $false
    $escaped = $false
    foreach ($character in $Text.ToCharArray()) {
        if ($inString) {
            if ($escaped) {
                $escaped = $false
            }
            elseif ($character -eq '\') {
                $escaped = $true
            }
            elseif ($character -eq '"') {
                $inString = $false
            }
            continue
        }
        if ($character -eq '"') { $inString = $true }
        elseif ($character -eq '{') { $depth++ }
        elseif ($character -eq '}') { $depth-- }
    }
    return [Math]::Max(0, $depth)
}

function Read-DecodedEffect([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Pose effect config not found: $Path"
    }
    $raw = [IO.File]::ReadAllBytes($Path)
    $decodedLength = $raw.Length
    $footerStripped = $false
    if ($raw.Length -ge 16) {
        $markerMatches = $true
        for ($index = 0; $index -lt $effectFooterMarker.Length; $index++) {
            if ($raw[$raw.Length - 16 + 4 + $index] -ne $effectFooterMarker[$index]) {
                $markerMatches = $false
                break
            }
        }
        if ($markerMatches) {
            $decodedLength -= 16
            $footerStripped = $true
        }
    }

    $decoded = New-Object byte[] $decodedLength
    for ($index = 0; $index -lt $decodedLength; $index++) {
        $decoded[$index] = $raw[$index] -bxor $xorKey[$index % 4]
    }

    $text = [Text.Encoding]::UTF8.GetString($decoded).TrimStart([char]0xfeff)
    $repairedBraces = 0
    try {
        $json = $text | ConvertFrom-Json
    }
    catch {
        $repairedBraces = Get-UnclosedObjectCount $text
        if ($repairedBraces -eq 0) {
            throw "Decoded pose effect is not JSON: $Path. $($_.Exception.Message)"
        }
        try {
            $json = ($text + ("}" * $repairedBraces)) | ConvertFrom-Json
        }
        catch {
            throw "Decoded pose effect is not JSON: $Path. $($_.Exception.Message)"
        }
    }
    return [pscustomobject]@{
        json = $json
        raw_bytes = $raw.Length
        decoded_bytes = $decoded.Length
        decoded_sha256 = Get-Sha256 $decoded
        footer_stripped = $footerStripped
        missing_root_braces_repaired = $repairedBraces
    }
}

function Add-ScalarLeaves([object]$Value, [string]$Path, [System.Collections.IDictionary]$Target) {
    if ($null -eq $Value) {
        $Target[$Path] = $null
        return
    }
    if ($Value -is [string] -or $Value -is [ValueType]) {
        $Target[$Path] = $Value
        return
    }
    if ($Value -is [System.Collections.IEnumerable] -and -not ($Value -is [pscustomobject])) {
        $index = 0
        foreach ($item in $Value) {
            Add-ScalarLeaves $item ("{0}[{1}]" -f $Path, $index) $Target
            $index++
        }
        return
    }
    foreach ($property in $Value.PSObject.Properties) {
        if ($property.Name -eq "ControlContent") { continue }
        $childPath = if ($Path) { "$Path.$($property.Name)" } else { $property.Name }
        Add-ScalarLeaves $property.Value $childPath $Target
    }
}

function Find-ReferenceLeaves([object]$Value, [string]$Path, [System.Collections.Generic.List[object]]$Target) {
    if ($null -eq $Value) { return }
    if ($Value -is [string] -or $Value -is [ValueType]) {
        if ($Path -match "(?i)(^|[.\[\]_-])(model|resource|hash|md5|sha1|sha256|download|url|path)([.\[\]_-]|$)") {
            $Target.Add([ordered]@{ path = $Path; value = $Value })
        }
        return
    }
    if ($Value -is [System.Collections.IEnumerable] -and -not ($Value -is [pscustomobject])) {
        $index = 0
        foreach ($item in $Value) {
            Find-ReferenceLeaves $item ("{0}[{1}]" -f $Path, $index) $Target
            $index++
        }
        return
    }
    foreach ($property in $Value.PSObject.Properties) {
        $childPath = if ($Path) { "$Path.$($property.Name)" } else { $property.Name }
        Find-ReferenceLeaves $property.Value $childPath $Target
    }
}

function Convert-EffectConfig([string]$Kind, [string]$Path) {
    $decoded = Read-DecodedEffect $Path
    if ($null -eq $decoded.json.Effects) {
        throw "Pose effect config has no Effects array: $Path"
    }

    $effects = New-Object System.Collections.Generic.List[object]
    foreach ($effect in @($decoded.json.Effects)) {
        $parameters = [ordered]@{}
        foreach ($property in $effect.PSObject.Properties) {
            if ($property.Name -in @("EffectName", "ControlContent")) { continue }
            Add-ScalarLeaves $property.Value $property.Name $parameters
        }
        $controls = New-Object System.Collections.Generic.List[object]
        foreach ($control in @($effect.ControlContent)) {
            if ($null -eq $control) { continue }
            $controls.Add([ordered]@{
                param_flag = if ($null -ne $control.ParamFlag) { [int]$control.ParamFlag } else { $null }
                render_param_ranges = @($control.RenderParamRanges)
                render_degree_ranges = @($control.RenderDegreeRanges)
                default_degree = $control.DefaultDegree
                male_degree_factor = $control.MaleDegreeFactor
                female_degree_factor = $control.FemaleDegreeFactor
                child_degree_factor = $control.ChildDegreeFactor
                older_degree_factor = $control.OlderDegreeFactor
            })
        }
        if ($null -ne $effect.ParamFlag) {
            $controls.Insert(0, [ordered]@{
                param_flag = [int]$effect.ParamFlag
                render_param_ranges = @($effect.ParamRange)
                render_degree_ranges = @($effect.EffectRange)
                default_degree = $effect.DefaultDegree
                male_degree_factor = $effect.MaleDegreeFactor
                female_degree_factor = $effect.FemaleDegreeFactor
                child_degree_factor = $effect.ChildDegreeFactor
                older_degree_factor = $effect.OlderDegreeFactor
            })
        }
        $effects.Add([ordered]@{
            effect_name = [string]$effect.EffectName
            use_ba_channel = [bool]$effect.UseBAChannel
            max_head_size = $effect.maxHeadSize
            param_flags = @($controls | ForEach-Object { $_.param_flag } | Where-Object { $null -ne $_ })
            controls = $controls
            parameters = $parameters
        })
    }

    $references = New-Object System.Collections.Generic.List[object]
    Find-ReferenceLeaves $decoded.json "" $references
    return [ordered]@{
        kind = $Kind
        source_path = (Resolve-Path -LiteralPath $Path).Path
        source_sha256 = Get-FileSha256 $Path
        raw_bytes = $decoded.raw_bytes
        decoded_bytes = $decoded.decoded_bytes
        decoded_sha256 = $decoded.decoded_sha256
        footer_stripped = $decoded.footer_stripped
        missing_root_braces_repaired = $decoded.missing_root_braces_repaired
        effect_count = $effects.Count
        param_flags = @($effects | ForEach-Object { $_.param_flags } | Sort-Object -Unique)
        effects = $effects
        reference_leaves = $references
    }
}

function Get-ObjectRecords([object]$Value) {
    $records = New-Object System.Collections.Generic.List[object]
    $stack = New-Object System.Collections.Stack
    $stack.Push($Value)
    while ($stack.Count -gt 0) {
        $current = $stack.Pop()
        if ($null -eq $current) { continue }
        if ($current -is [System.Collections.IEnumerable] -and -not ($current -is [pscustomobject])) {
            foreach ($item in $current) { $stack.Push($item) }
            continue
        }
        if ($current -is [pscustomobject]) {
            $properties = @($current.PSObject.Properties)
            $matching = @($properties | Where-Object {
                $_.Name -match "(?i)^(resourceId|modelId|selectedModelId|hashId|downloadUrl|cloudSourceDownloadUrl|md5|md5Empty|sha1|sha1Empty|sha256|fileName|resourceName)$"
            })
            if ($matching.Count -gt 0) {
                $record = [ordered]@{}
                foreach ($property in $properties) {
                    if ($property.Value -is [string] -or $property.Value -is [ValueType] -or $null -eq $property.Value) {
                        $record[$property.Name] = $property.Value
                    }
                }
                $records.Add($record)
            }
            foreach ($property in $properties) { $stack.Push($property.Value) }
        }
    }
    return $records
}

function Read-NetworkPayload([string]$Path) {
    if (-not $Path) { return $null }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Network payload not found: $Path"
    }
    $root = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($root.log.entries) {
        $records = New-Object System.Collections.Generic.List[object]
        $index = 0
        foreach ($entry in @($root.log.entries)) {
            $responseJson = $null
            $text = $entry.response.content.text
            if ($text -and $entry.response.content.encoding -ne "base64") {
                try { $responseJson = $text | ConvertFrom-Json } catch { }
            }
            foreach ($record in @(Get-ObjectRecords $responseJson)) {
                $record["payload_record_index"] = $index
                $record["request_method"] = $entry.request.method
                $record["request_url"] = $entry.request.url
                $record["response_status"] = $entry.response.status
                $records.Add($record)
            }
            $index++
        }
        return [ordered]@{
            source_path = (Resolve-Path -LiteralPath $Path).Path
            source_sha256 = Get-FileSha256 $Path
            format = "har"
            records = $records
        }
    }
    return [ordered]@{
        source_path = (Resolve-Path -LiteralPath $Path).Path
        source_sha256 = Get-FileSha256 $Path
        format = "json"
        records = @(Get-ObjectRecords $root)
    }
}

function Copy-ContainerEntry([IO.FileStream]$Source, [object]$Entry, [string]$DestinationPath) {
    [void](New-Item -ItemType Directory -Force -Path (Split-Path -Parent $DestinationPath))
    $destination = [IO.File]::Open($DestinationPath, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        [void]$Source.Seek([int64]$Entry.offset, [IO.SeekOrigin]::Begin)
        $remaining = [int64]$Entry.bytes
        $buffer = New-Object byte[] (1MB)
        while ($remaining -gt 0) {
            $want = [int][Math]::Min([int64]$buffer.Length, $remaining)
            $read = $Source.Read($buffer, 0, $want)
            if ($read -le 0) { throw "Unexpected EOF for container entry $($Entry.name)." }
            $destination.Write($buffer, 0, $read)
            $remaining -= $read
        }
    }
    finally {
        $destination.Dispose()
    }
}

$poseConfig = Convert-EffectConfig "PoseRefine" $PoseRefinePath
$subPoseConfig = Convert-EffectConfig "SubPoseRefine" $SubPoseRefinePath
$network = Read-NetworkPayload $NetworkPayloadPath
$entries = Read-ContainerEntries $ContainerPath
$numericEntries = @($entries | Where-Object { $_.numeric })
$namedEntries = @($entries | Where-Object { -not $_.numeric })
$numericById = @{}
foreach ($entry in $numericEntries) { $numericById[$entry.name] = $entry }

$shaderEntries = @($namedEntries | Where-Object {
    $_.type -eq "0" -and $_.name -match "(?i)(PoseRefine|AIBodyReshape|HumanDetected)" -and $_.name -match "\.(frag|vert)$"
})
$algorithmEvidence = @(
    [ordered]@{
        shader = "glsl/PoseRefine/BodyLiquid.frag"
        evidence = "6-point cosine-falloff displacement with per-point degree and aspect-corrected radius"
    },
    [ordered]@{
        shader = "glsl/PoseRefine/BodyMultipleDegreeLiquid.frag"
        evidence = "25-point displacement with per-point size, signed degree, edge protection and optional protect mask"
    },
    [ordered]@{
        shader = "glsl/PoseRefine/BodyRefine.frag"
        evidence = "mask-weighted UV displacement; ENABLE_UVMAP_BA writes displacement to B/A"
    },
    [ordered]@{
        shader = "glsl/PoseRefine/GlobalStretch.frag"
        evidence = "horizontal remap from normalized source/resolution ranges scaled by mask weight"
    },
    [ordered]@{
        shader = "glsl/AIBodyReshape/AIBodyReshape.frag"
        evidence = "UV shift texture sampling with RGB16 fixed-point or float texture support"
    }
)

$payloadScan = [ordered]@{
    numeric_headers_scanned = 0
    named_headers_scanned = 0
    raw_magic_hits = 0
    outer_xor_magic_hits = 0
    neural_candidates = @()
    opaque_numeric_entries = 0
}
$modelStream = [IO.File]::OpenRead($ContainerPath)
try {
    foreach ($entry in $entries) {
        if ($entry.bytes -le 0 -or $entry.type -ne "0") { continue }
        $head = Get-ContainerHead $modelStream $entry
        if ($head.Length -eq 0) { continue }
        $rawMagic = Get-MagicInfo $head
        $xorBytes = New-Object byte[] $head.Length
        for ($index = 0; $index -lt $head.Length; $index++) {
            $xorBytes[$index] = $head[$index] -bxor $xorKey[$index % 4]
        }
        $xorMagic = Get-MagicInfo $xorBytes
        if ($entry.numeric) {
            $payloadScan.numeric_headers_scanned++
            if ($rawMagic.format -ne "empty-or-unknown") { $payloadScan.raw_magic_hits++ }
            if ($xorMagic.format -ne "empty-or-unknown") { $payloadScan.outer_xor_magic_hits++ }
            if ($rawMagic.format -eq "empty-or-unknown" -and $xorMagic.format -eq "empty-or-unknown") {
                $payloadScan.opaque_numeric_entries++
            }
        }
        else {
            $payloadScan.named_headers_scanned++
        }
        foreach ($candidate in @(
            [pscustomobject]@{ mode = "raw"; magic = $rawMagic },
            [pscustomobject]@{ mode = "outer-xor"; magic = $xorMagic }
        )) {
            if ($neuralFormats -notcontains $candidate.magic.format) { continue }
            $payloadScan.neural_candidates += [ordered]@{
                name = $entry.name
                numeric = $entry.numeric
                offset = $entry.offset
                bytes = $entry.bytes
                decode = $candidate.mode
                format = $candidate.magic.format
                head_hex = $candidate.magic.head_hex
                pose_identity = $false
            }
        }
    }
}
finally {
    $modelStream.Dispose()
}

if ($ExtractCandidates) {
    if (-not $CandidateOutDirectory) { throw "-ExtractCandidates requires -CandidateOutDirectory." }
    [void](New-Item -ItemType Directory -Force -Path $CandidateOutDirectory)
    $extractStream = [IO.File]::OpenRead($ContainerPath)
    try {
        foreach ($candidate in $payloadScan.neural_candidates) {
            $entry = $entries | Where-Object { $_.name -eq $candidate.name } | Select-Object -First 1
            $path = Join-Path $CandidateOutDirectory ("{0}.bin" -f $candidate.name)
            Copy-ContainerEntry $extractStream $entry $path
            $sha = Get-FileSha256 $path
            "$sha  $(Split-Path -Leaf $path)" | Set-Content -LiteralPath "$path.sha256" -Encoding ASCII
            $candidate["out_file"] = (Resolve-Path -LiteralPath $path).Path
            $candidate["sha256"] = $sha
        }
    }
    finally {
        $extractStream.Dispose()
    }
}

$correlations = New-Object System.Collections.Generic.List[object]
if ($network) {
    foreach ($record in $network.records) {
        $ids = @($record["resourceId"], $record["modelId"], $record["selectedModelId"], $record["hashId"]) |
            Where-Object { $null -ne $_ } | ForEach-Object { [string]$_ } | Sort-Object -Unique
        foreach ($id in $ids) {
            $entry = $numericById[$id]
            $correlations.Add([ordered]@{
                id = $id
                container_entry = if ($entry) {
                    [ordered]@{ name = $entry.name; offset = $entry.offset; bytes = $entry.bytes }
                } else { $null }
                download_url = @($record["downloadUrl"], $record["cloudSourceDownloadUrl"]) |
                    Where-Object { $_ } | Select-Object -First 1
                md5 = @($record["md5"], $record["md5Empty"]) | Where-Object { $_ } | Select-Object -First 1
                sha1 = @($record["sha1"], $record["sha1Empty"]) | Where-Object { $_ } | Select-Object -First 1
                sha256 = $record["sha256"]
                model_identity_verified = $false
            })
        }
    }
}

$runtimeSupported = @(400, 401, 404, 405, 406, 407, 408, 410, 415, 416, 38556)
$allParamFlags = @($poseConfig.param_flags + $subPoseConfig.param_flags | Sort-Object -Unique)
$runtimeUnsupported = @($allParamFlags | Where-Object { $runtimeSupported -notcontains [int]$_ })
$blockers = New-Object System.Collections.Generic.List[string]
if ($payloadScan.neural_candidates.Count -eq 0) {
    $blockers.Add("No PixCake container entry exposes supported MNN, TFLite or ONNX protobuf magic in raw or outer-XOR form.")
}
if ($payloadScan.neural_candidates.Count -gt 0) {
    $blockers.Add("Neural-format candidates exist but no static evidence maps any candidate to PoseRefine, SubPoseRefine or the 69-point body producer.")
}
if (-not $network) {
    $blockers.Add("No authenticated resource/model response or HAR was supplied; resourceId/hashId/downloadUrl correlation is unavailable.")
}
elseif ($correlations.Count -eq 0) {
    $blockers.Add("The supplied network payload contains no resourceId/modelId/hashId records correlatable to PixCake container entries.")
}
$blockers.Add("The decoded effect configs contain no modelId, resourceId, digest, tensor name, tensor shape or keypoint order.")
$blockers.Add("The body shaders reveal deformation math, but not the upstream pose weights, exact landmark indices or network preprocessing.")
if ($runtimeUnsupported.Count -gt 0) {
    $blockers.Add("B runtime has no effect implementation for ParamFlags: $($runtimeUnsupported -join ', ').")
}

if ($RequireNeuralCandidate -and $payloadScan.neural_candidates.Count -eq 0) {
    throw "No neural-format candidate was found in the PixCake container."
}

$manifest = [ordered]@{
    schema = "aurora/pixcake-pose-extraction/1"
    lane = "B6"
    base_commit = "eececfd"
    status = if ($correlations.Count -gt 0) { "correlated-but-pose-unproven" } else { "blocked-no-pose-artifact" }
    pose_configs = @($poseConfig, $subPoseConfig)
    container = [ordered]@{
        path = (Resolve-Path -LiteralPath $ContainerPath).Path
        bytes = (Get-Item -LiteralPath $ContainerPath).Length
        entries = $entries.Count
        numeric_entries = $numericEntries.Count
        named_entries = $namedEntries.Count
        pose_shader_entries = $shaderEntries
        algorithm_evidence = $algorithmEvidence
        payload_scan = $payloadScan
    }
    network_payload = $network
    correlations = $correlations
    runtime_bridge = [ordered]@{
        existing_contract = "BodyPoseModel accepts float32 NCHW [1,3,H,W] and 2D/3D keypoint outputs"
        supported_param_flags = $runtimeSupported
        pose_param_flags = $allParamFlags
        unsupported_param_flags = $runtimeUnsupported
        body_algorithm_evidence_only = $true
        model_connected = $false
    }
    pose_model = [ordered]@{
        model_id = $null
        file_name = $null
        resource_id = $null
        sha256 = $null
        input_tensor = $null
        output_tensor = $null
        keypoint_count = $null
        keypoint_order = $null
        runtime_integration = "not-attempted-no-verified-model"
    }
    blockers = $blockers
}

$directory = Split-Path -Parent $OutFile
if ($directory) { [void](New-Item -ItemType Directory -Force -Path $directory) }
$manifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $OutFile -Encoding UTF8
[ordered]@{
    out_file = (Resolve-Path -LiteralPath $OutFile).Path
    status = $manifest.status
    effects = $poseConfig.effect_count + $subPoseConfig.effect_count
    param_flags = $allParamFlags.Count
    shaders = $shaderEntries.Count
    neural_candidates = $payloadScan.neural_candidates.Count
    opaque_numeric_entries = $payloadScan.opaque_numeric_entries
    network_records = if ($network) { $network.records.Count } else { 0 }
    correlations = $correlations.Count
} | ConvertTo-Json -Depth 4
