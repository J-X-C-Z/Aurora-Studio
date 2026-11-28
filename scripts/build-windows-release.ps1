<#
.SYNOPSIS
    Builds and stages the native Windows release binary of Aurora Studio (GPUI).

.DESCRIPTION
    Uses the host's GNU toolchain (`stable-x86_64-pc-windows-gnu`) and an isolated
    CARGO_TARGET_DIR, so a release build never fights a concurrent debug/test
    build in the same checkout.

    GPUI's Windows backend has two shader modes:

      * runtime  - `gpui-pre-windows` is compiled with debug-assertions (the
                   workspace profile override) and compiles `src/shaders.hlsl`
                   with `d3dcompiler_47.dll` on startup, reading the shader from
                   the absolute path of the *cargo registry source* baked in at
                   build time. Works on the build machine, breaks once that
                   directory is gone, and needs d3dcompiler_47.dll at runtime.
      * prebuilt - `fxc.exe` (Windows SDK) compiles the shaders during the build
                   and the bytes are linked into the exe. Self-contained; this is
                   what `-FxcPath` selects.

    Resolves FXC from the explicit argument, environment, SDK or PATH. If no FXC
    is found, stops before publishing a machine-dependent executable.

.PARAMETER TargetDir
    CARGO_TARGET_DIR for this release build.

.PARAMETER OutDir
    Publish directory that receives `aurora-studio.exe` and `BUILD-INFO.txt`.

.PARAMETER FxcPath
    Path to a privately extracted `fxc.exe` (Windows SDK). Enables the
    self-contained shader precompile.

.PARAMETER LibRawRoot
    Optional verified open-source runtime bundle, with bin/ and licenses/. Only the
    eight DLLs needed by the C interface are included; no third-party GPU libraries.

.PARAMETER Jobs
    Cargo parallelism; defaults to the number of logical CPUs.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows-release.ps1
#>
[CmdletBinding()]
param(
    [string]$TargetDir = '',
    [string]$OutDir = '',
    [string]$FxcPath = '',
    [string]$LibRawRoot = '',
    [int]$Jobs = 0
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not $TargetDir) { $TargetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $repo 'target' } }
if (-not $OutDir) { $OutDir = Join-Path $repo 'target/dist' }
$logDir = Join-Path (Split-Path $TargetDir -Parent) 'logs'
New-Item -ItemType Directory -Force -Path $TargetDir, $OutDir, $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path $logDir "build-$stamp.log"

Push-Location $repo
try {
    $rustc = (& rustc -vV) -join "`n"
    $hostTriple = ($rustc -split "`n" | Where-Object { $_ -like 'host:*' }) -replace '^host:\s*', ''
    if ($hostTriple -notlike '*windows*') {
        throw "release build must run on Windows (rustc host = $hostTriple)"
    }

    if (-not $FxcPath -and $env:GPUI_FXC_PATH) { $FxcPath = $env:GPUI_FXC_PATH }
    # 1) an installed Windows SDK ("Windows Kits\10\bin\<ver>\x64\fxc.exe")
    if (-not $FxcPath -and ${env:ProgramFiles(x86)}) {
        $root = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits/10/bin/*/x64/fxc.exe'
        $hit = Get-ChildItem $root -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
        if ($hit) { $FxcPath = $hit.FullName }
    }
    # 2) a privately extracted SDK: the official Windows SDK installer can be
    #    laid out offline and unpacked with `msiexec /a` (see HOW-TO-RUN.txt), so
    #    a machine without the SDK installed still gets fxc. Roots are searched
    #    from $env:AURORA_WINSDK_ROOT and ./toolchain.
    if (-not $FxcPath) {
        $privateRoots = @(
            $env:AURORA_WINSDK_ROOT,
            (Join-Path $repo 'toolchain')
        ) | Where-Object { $_ -and (Test-Path $_) }
        foreach ($root in $privateRoots) {
            # PowerShell has no globstar: recurse for fxc.exe, then keep the x64 one.
            $hit = Get-ChildItem -Path $root -Recurse -Filter 'fxc.exe' -File -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -match '\\x64\\fxc\.exe$' } |
                Sort-Object FullName -Descending | Select-Object -First 1
            if ($hit) { $FxcPath = $hit.FullName; break }
        }
    }
    # 3) PATH
    if (-not $FxcPath) {
        $hit = Get-Command fxc.exe -ErrorAction SilentlyContinue
        if ($hit) { $FxcPath = $hit.Source }
    }

    $cargoArgs = @('build', '-p', 'portrait-app', '--release', '--locked', '--message-format=short')
    $shaderMode = 'runtime'
    if ($FxcPath -and (Test-Path $FxcPath)) {
        $env:GPUI_FXC_PATH = $FxcPath
        # Cancel the workspace override that forces runtime HLSL compilation.
        $cargoArgs += @('--config', 'profile.release.package.gpui-pre-windows.debug-assertions=false')
        $shaderMode = "prebuilt ($FxcPath)"
    } else {
        throw 'A redistributable release needs fxc.exe to embed the shaders. Pass -FxcPath or set GPUI_FXC_PATH/AURORA_WINSDK_ROOT to a Windows SDK. For a developer-only local build use cargo build --release.'
    }
    if ($Jobs -gt 0) { $cargoArgs += @('-j', "$Jobs") }

    $env:CARGO_TARGET_DIR = $TargetDir
    "== cargo $($cargoArgs -join ' ') (shaders: $shaderMode) ==" | Tee-Object -FilePath $log
    # cargo writes progress to stderr; under EAP=Stop PowerShell 5.1 turns that
    # into a terminating NativeCommandError, so lower it for the invocation only.
    $previousEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & cargo @cargoArgs 2>&1 | Tee-Object -FilePath $log -Append
    $cargoExit = $LASTEXITCODE
    $ErrorActionPreference = $previousEap
    if ($cargoExit -ne 0) { throw "cargo build failed with exit code $cargoExit (log: $log)" }

    $exe = Join-Path $TargetDir 'release\aurora-studio.exe'
    if (-not (Test-Path $exe)) { throw "missing build artifact: $exe" }
    $staged = Join-Path $OutDir 'aurora-studio.exe'
    Copy-Item $exe $staged -Force

    if ($LibRawRoot) {
        $runtime = Join-Path $OutDir 'libraw'
        $licenses = Join-Path $runtime 'licenses'
        $names = @('libraw-25.dll','libgcc_s_seh-1.dll','libstdc++-6.dll','libgomp-1.dll',
            'libwinpthread-1.dll','liblcms2-2.dll','libjpeg-8.dll','zlib1.dll')
        if (-not (Test-Path (Join-Path $LibRawRoot 'licenses/libraw/LICENSE.LGPL'))) { throw 'Missing LibRaw licenses in the supplied bundle' }
        New-Item -ItemType Directory -Force $runtime,$licenses | Out-Null
        foreach ($name in $names) { Copy-Item -LiteralPath (Join-Path $LibRawRoot "bin/$name") -Destination $runtime -Force }
        Copy-Item (Join-Path $LibRawRoot 'licenses/*') -Destination $licenses -Recurse -Force
    }
    $payload = @((Get-Item $staged))
    if ($LibRawRoot) { $payload += @(Get-ChildItem (Join-Path $OutDir 'libraw') -Recurse -File) }
    $prefix = (Get-Item $OutDir).FullName.TrimEnd('\') + '\'
    $payload | Sort-Object FullName | ForEach-Object {
        (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash + '  ' + $_.FullName.Substring($prefix.Length).Replace('\','/')
    } | Set-Content -Encoding UTF8 (Join-Path $OutDir 'SHA256SUMS.txt')
    $sha = (Get-FileHash $staged -Algorithm SHA256).Hash
    # The script also runs over an exported source tree (`git archive`), where
    # there is no repository: the revision is then simply unknown.
    $gitRev = try { (& git -C $repo rev-parse HEAD 2>$null) } catch { $null }
    if (-not $gitRev) { $gitRev = 'unknown (no git repository)' }
    $gitDirty = try { (& git -C $repo status --porcelain 2>$null | Measure-Object).Count } catch { 0 }
    $sizeMb = [math]::Round((Get-Item $staged).Length / 1MB, 1)
    $objdump = (Get-Command objdump.exe -ErrorAction SilentlyContinue).Source
    $imports = ''
    if ($objdump) {
        $imports = (& $objdump -p $staged | Select-String 'DLL Name:' | ForEach-Object { ($_.Line -split 'DLL Name:\s*')[1] } | Sort-Object -Unique) -join ', '
    }

    $info = @(
        "artifact      : $staged"
        "size          : $sizeMb MB"
        "sha256        : $sha"
        "git revision  : $gitRev (dirty files: $gitDirty)"
        "built at      : $(Get-Date -Format o)"
        "host toolchain: $hostTriple"
        $(($rustc -split "`n" | Where-Object { $_ -match '^(rustc|LLVM)' }) -join '; ')
        "shaders       : $shaderMode"
        "target dir    : $TargetDir"
        "build log     : $log"
        "imports       : $imports"
    )
    $infoPath = Join-Path $OutDir 'BUILD-INFO.txt'
    $info | Set-Content -Path $infoPath -Encoding UTF8

    $info | ForEach-Object { Write-Host $_ }
    Write-Host ''
    Write-Host "staged: $staged"
    Write-Host "report: $infoPath"
}
finally {
    Pop-Location
}
