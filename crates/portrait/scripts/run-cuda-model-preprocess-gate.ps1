param(
    [string]$ExpectedGpu = 'NVIDIA GeForce RTX 4070 Ti',
    [string]$CargoTargetDir = ''
)

$ErrorActionPreference = 'Stop'

$portraitRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $portraitRoot
$repoRoot = Split-Path -Parent $repoRoot
$workspaceRoot = $repoRoot
while ($workspaceRoot -and (Split-Path -Leaf $workspaceRoot) -ne 'Aurora Studio') {
    $parent = Split-Path -Parent $workspaceRoot
    if ($parent -eq $workspaceRoot) {
        throw "Could not locate the Aurora Studio workspace root from $repoRoot"
    }
    $workspaceRoot = $parent
}

if ([string]::IsNullOrWhiteSpace($CargoTargetDir)) {
    $CargoTargetDir = Join-Path $repoRoot 'target\wl46-model-preprocess'
}
$evidenceDir = if ($env:AURORA_EVIDENCE_DIR) {
    $env:AURORA_EVIDENCE_DIR
} else {
    Join-Path $repoRoot 'target\evidence\wl46-model-preprocess'
}
New-Item -ItemType Directory -Force -Path $CargoTargetDir, $evidenceDir | Out-Null
$logPath = Join-Path $evidenceDir 'gpu-rtx4070ti.log'

$gpuNames = @(
    & nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 |
        ForEach-Object { $_.ToString().Trim() }
)
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi failed: $($gpuNames -join ' ')"
}
if (-not ($gpuNames -contains $ExpectedGpu)) {
    throw "Expected GPU '$ExpectedGpu'; found: $($gpuNames -join ', ')"
}

$env:PORTRAIT_CUDA_REQUIRED = '1'
$env:CARGO_TARGET_DIR = $CargoTargetDir
$output = @(
    & cargo test `
        --manifest-path (Join-Path $repoRoot 'Cargo.toml') `
        -p aurora-portrait `
        --release `
        --locked `
        cuda_model_preprocess `
        -- --nocapture --test-threads=1 2>&1
)
$cargoExit = $LASTEXITCODE
$output | Tee-Object -FilePath $logPath
if ($cargoExit -ne 0) {
    throw "cuda_model_preprocess GPU gate failed with exit code $cargoExit"
}

$text = $output -join "`n"
$requiredTests = @(
    'cuda_model_preprocess::tests::gpu_nchw_matches_the_cpu_bit_for_bit',
    'cuda_model_preprocess::tests::gpu_planar_and_quant_match_the_cpu_bit_for_bit'
)
foreach ($testName in $requiredTests) {
    if ($text -notmatch [regex]::Escape("test $testName ... ok")) {
        throw "Required GPU test did not pass: $testName"
    }
}

$nchwCases = ([regex]::Matches($text, 'WL46_NCHW_SHAPE ')).Count
$planarCases = ([regex]::Matches($text, 'WL46_PLANAR_SHAPE ')).Count
$quantCases = ([regex]::Matches($text, 'WL46_QUANT_SHAPE ')).Count
if ($nchwCases -ne 6 -or $planarCases -ne 9 -or $quantCases -ne 9) {
    throw "Shape gate counts differ: nchw=$nchwCases planar=$planarCases quant=$quantCases"
}

$expectedShapePrefixes = @(
    'WL46_NCHW_SHAPE 5x4 stride=5 out=3 ',
    'WL46_NCHW_SHAPE 6x6 stride=9 out=8 ',
    'WL46_NCHW_SHAPE 17x3 stride=17 out=1 ',
    'WL46_PLANAR_SHAPE 4x4 stride=4 ',
    'WL46_PLANAR_SHAPE 5x3 stride=8 ',
    'WL46_PLANAR_SHAPE 1x1 stride=1 ',
    'WL46_QUANT_SHAPE 4x4 stride=4 ',
    'WL46_QUANT_SHAPE 5x3 stride=8 ',
    'WL46_QUANT_SHAPE 1x1 stride=1 '
)
foreach ($prefix in $expectedShapePrefixes) {
    if (-not $text.Contains($prefix)) {
        throw "Required shape gate entry is missing: $prefix"
    }
}

Write-Output "GPU=$($gpuNames -join ', ')"
Write-Output "SHAPE_TUPLES=9 NCHW_CASES=$nchwCases PLANAR_CASES=$planarCases QUANT_CASES=$quantCases"
Write-Output "EVIDENCE=$logPath"
Write-Output 'WL46_GPU_GATE_OK'
