<#
.SYNOPSIS
    Checks `cargo package --list` for every workspace crate against a publish
    allowlist (IN-2).

.DESCRIPTION
    Cargo decides package contents from the manifest; this script only reports
    and gates them. It never edits Cargo.toml.

    A packaged path is a violation when it
      * lives under a denied research/private directory,
      * has a RAW/private media extension, or
      * is outside the crate's allowlist (Cargo's own bookkeeping files excepted).

    Strict mode (default) exits 1 on the first crate with violations. Warn mode
    prints the same report and exits 0, so a known blocker can be tracked
    without hiding new findings.

.PARAMETER Mode
    strict (default) or warn.

.PARAMETER Package
    One or more of aurora-engine, portrait-app, aurora-portrait (default: all).

.EXAMPLE
    ./scripts/check-package-allowlist.ps1
    ./scripts/check-package-allowlist.ps1 -Mode warn
#>
[CmdletBinding()]
param(
    [ValidateSet('strict', 'warn')]
    [string]$Mode = 'strict',

    [ValidateSet('aurora-engine', 'portrait-app', 'aurora-portrait')]
    [string[]]$Package = @('aurora-engine', 'portrait-app', 'aurora-portrait')
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $repoRoot
try {
    # Cargo always adds these; they are manifest/registry bookkeeping, not
    # shipped content. README.md is auto-included when present.
    $cargoBookkeeping = @(
        'Cargo.toml',
        'Cargo.toml.orig',
        'Cargo.lock',
        '.cargo_vcs_info.json',
        'README.md'
    )

    # RAW camera files and private media must never be packaged.
    $privateMedia = '(?i)\.(nef|dng|cr2|cr3|arw|raf|orf|rw2|pef|srw|heic|heif|eip|psd)$'

    $rules = @{
        'aurora-engine' = [pscustomobject]@{
            AllowDirs  = @('src/', 'native/cuda/', 'native/metal/', 'assets/', 'tests/', 'examples/', 'data/')
            AllowFiles = @()
            DenyDirs   = @('work/', 'tools/', 'docs/', 'testdata/')
        }
        'portrait-app' = [pscustomobject]@{
            AllowDirs  = @('src/', 'assets/')
            AllowFiles = @('build.rs', 'app.rc')
            DenyDirs   = @('work/', 'tools/', 'docs/', 'testdata/')
        }
        'aurora-portrait' = [pscustomobject]@{
            AllowDirs  = @('src/', 'assets/', 'native/', 'tests/')
            AllowFiles = @()
            DenyDirs   = @('work/', 'testdata/')
        }
    }

    $violations = New-Object System.Collections.Generic.List[object]

    foreach ($pkg in $Package) {
        Write-Host "== $pkg =="
        $outFile = [System.IO.Path]::GetTempFileName()
        $errFile = [System.IO.Path]::GetTempFileName()
        try {
            # Start-Process keeps cargo's stderr from being turned into a
            # NativeCommandError under $ErrorActionPreference = 'Stop'.
            $proc = Start-Process -FilePath 'cargo' `
                -ArgumentList @('package', '-p', $pkg, '--list', '--allow-dirty') `
                -RedirectStandardOutput $outFile -RedirectStandardError $errFile `
                -NoNewWindow -Wait -PassThru
            $exitCode = $proc.ExitCode
            $lines = @(Get-Content -Path $outFile)
            $stderr = Get-Content -Path $errFile -Raw -ErrorAction SilentlyContinue
        } finally {
            Remove-Item -Path $outFile, $errFile -Force -ErrorAction SilentlyContinue
        }
        if ($exitCode -ne 0) {
            throw "cargo package --list failed for $pkg (exit $exitCode): $stderr"
        }
        $entries = @($lines | ForEach-Object { "$_".Trim() } | Where-Object { $_ -ne '' })
        $rule = $rules[$pkg]
        $counts = @{ ok = 0; denied = 0; media = 0; outside = 0 }

        foreach ($entry in $entries) {
            $path = $entry -replace '\\', '/'
            if ($cargoBookkeeping -contains $path) {
                $counts.ok++
                continue
            }

            $kind = $null
            $reason = $null

            # Denied directories are crate-root directories only (work/, testdata/,
            # tools/, docs/); source modules such as src/tools/ are legitimate.
            $denied = $false
            foreach ($d in $rule.DenyDirs) {
                if ($path.StartsWith($d, [System.StringComparison]::OrdinalIgnoreCase)) { $denied = $true }
            }

            if ($denied) {
                $kind = 'denied-directory'
                $reason = 'research/private directory must not be packaged'
            } elseif ($path -match $privateMedia) {
                $kind = 'private-media'
                $reason = 'RAW/private media extension must not be packaged'
            } elseif ($rule.AllowFiles -contains $path) {
                $counts.ok++
                continue
            } else {
                $allowed = $false
                foreach ($d in $rule.AllowDirs) {
                    if ($path.StartsWith($d, [System.StringComparison]::Ordinal)) { $allowed = $true; break }
                }
                if (-not $allowed) {
                    $kind = 'outside-allowlist'
                    $reason = 'not in the publish allowlist'
                }
            }

            if ($kind) {
                switch ($kind) {
                    'denied-directory' { $counts.denied++ }
                    'private-media'    { $counts.media++ }
                    default            { $counts.outside++ }
                }
                $violations.Add([pscustomobject]@{
                    Package = $pkg; Kind = $kind; Path = $path; Reason = $reason
                })
            } else {
                $counts.ok++
            }
        }

        Write-Host ("   entries={0} ok={1} denied-directory={2} private-media={3} outside-allowlist={4}" -f `
            $entries.Count, $counts.ok, $counts.denied, $counts.media, $counts.outside)
    }

    if ($violations.Count -gt 0) {
        Write-Host ''
        foreach ($v in $violations) {
            $line = "[{0}] {1}: {2} ({3})" -f $v.Kind, $v.Package, $v.Path, $v.Reason
            if ($env:GITHUB_ACTIONS -eq 'true') {
                Write-Host "::error::$line"
            } else {
                Write-Host "VIOLATION $line"
            }
        }
        Write-Host ''
        Write-Host ("{0} violation(s) found." -f $violations.Count)
        if ($Mode -eq 'strict') {
            Write-Host 'strict mode: failing. Use -Mode warn to report without failing.'
            exit 1
        }
        Write-Host 'warn mode: not failing (known blocker is tracked in the IN-2 report).'
        exit 0
    }

    Write-Host 'All packages are inside their allowlists.'
    exit 0
}
finally {
    Pop-Location
}
