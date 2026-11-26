<#
.SYNOPSIS
    Launches (and optionally observes) the native Windows release binary.

.DESCRIPTION
    The release exe is a `windows_subsystem = "windows"` process: stderr goes
    nowhere unless the parent redirects the handles. This wrapper redirects
    stdout/stderr to files, records the exit code, and can stop the process after
    a grace period, so a startup failure is a readable log instead of a vanished
    window.

    `-SelfCheck` runs `aurora-studio.exe --gpu-selfcheck`, which initialises the
    engine GPU path and exits *before* any window is created. Safe to run
    unattended.

.PARAMETER Exe
    Path to aurora-studio.exe.

.PARAMETER LogDir
    Directory for stdout/stderr/exit records.

.PARAMETER KillAfterSeconds
    Stop the process after N seconds if it is still running (0 = leave it open).

.PARAMETER SelfCheck
    Run the GPU selfcheck entry point instead of the GUI.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-windows-release.ps1 -SelfCheck

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-windows-release.ps1 -KillAfterSeconds 15
#>
[CmdletBinding()]
param(
    [string]$Exe = (Join-Path (Split-Path $PSScriptRoot -Parent) 'target/dist/aurora-studio.exe'),
    [string]$LogDir = (Join-Path (Split-Path $PSScriptRoot -Parent) 'target/diagnostics'),
    [int]$KillAfterSeconds = 0,
    [switch]$SelfCheck
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Exe)) { throw "missing exe: $Exe" }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$tag = if ($SelfCheck) { 'selfcheck' } else { 'run' }
$out = Join-Path $LogDir "$tag-$stamp.out.log"
$err = Join-Path $LogDir "$tag-$stamp.err.log"

# Keep the app's own logging verbose enough to see where it stops.
$env:RUST_LOG = if ($env:RUST_LOG) { $env:RUST_LOG } else { 'info' }
$env:RUST_BACKTRACE = '1'

if ($SelfCheck) {
    $p = Start-Process -FilePath $Exe -ArgumentList '--gpu-selfcheck' -PassThru -NoNewWindow `
        -RedirectStandardOutput $out -RedirectStandardError $err
} else {
    $p = Start-Process -FilePath $Exe -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
}

if ($SelfCheck) {
    $completed = $p.WaitForExit(120000)
    if (-not $completed) { $p.Kill(); Write-Host 'selfcheck timed out after 120s'; }
} elseif ($KillAfterSeconds -gt 0) {
    $completed = $p.WaitForExit($KillAfterSeconds * 1000)
    if (-not $completed) {
        Write-Host "still running after ${KillAfterSeconds}s - window stayed up; stopping it"
        $p.Kill(); $p.WaitForExit()
    }
}

$p.Refresh()
$exit = if ($p.HasExited) { $p.ExitCode } else { $null }
"exit code : $exit"
"stdout    : $out"
"stderr    : $err"
if ($exit -ne $null -and $exit -ne 0) {
    Write-Host ''
    Write-Host '--- stderr tail ---'
    if (Test-Path $err) { Get-Content $err -Tail 40 }
}
