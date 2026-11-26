<#
.SYNOPSIS
  Install a local Aurora release for the current Windows account. No admin rights or PATH edits.
.DESCRIPTION
  Validate the builder's local SHA256SUMS.txt, stage and verify every payload file, then switch
  the app directory and the desktop link. The previous installation stays next to the new one;
  never kill a running editor or touch its settings, catalogs, originals, or other shortcuts.
#>
[CmdletBinding()]
param([string]$PackageDir = '')
$ErrorActionPreference = 'Stop'
if (-not $PackageDir) {
    $PackageDir = if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'SHA256SUMS.txt')) { $PSScriptRoot } else { Join-Path (Split-Path $PSScriptRoot -Parent) 'target/dist' }
}
$pkg = (Get-Item -LiteralPath $PackageDir).FullName
$base = Join-Path $env:LOCALAPPDATA 'Programs'
$install = Join-Path $base 'Aurora Studio'
$exe = Join-Path $install 'aurora-studio.exe'
$link = Join-Path ([Environment]::GetFolderPath('DesktopDirectory')) 'Aurora Studio.lnk'
$id = [guid]::NewGuid().ToString('N').Substring(0,12)
$stage = Join-Path $base ('.Aurora-stage-' + $id)
$previous = Join-Path $base ('Aurora Studio.previous-' + $id)
$pendingLink = Join-Path (Split-Path $link -Parent) ('.Aurora-' + $id + '.lnk')
$backupLink = Join-Path $base ('.Aurora-link-' + $id + '.lnk')
if (Get-Process -Name 'aurora-studio' -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $exe }) {
    throw 'The installed editor is running. Finish your exports and close its window, then retry.'
}
$manifest = Join-Path $pkg 'SHA256SUMS.txt'
if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) { throw 'Missing release manifest. Build using build-windows-release.ps1 first.' }
$expected = @{}
foreach ($line in (Get-Content -LiteralPath $manifest)) {
    if ($line -notmatch '^([0-9A-Fa-f]{64})  (.+)$') { throw "Bad manifest line: $line" }
    $hash = $Matches[1]; $relative = $Matches[2]
    if ($relative -match '(^/|[:\\]|(?:^|/)\.{1,2}(?:/|$))' -or
        $relative -notmatch '^(aurora-studio\.exe|libraw/(libraw-25|libgcc_s_seh-1|libstdc\+\+-6|libgomp-1|libwinpthread-1|liblcms2-2|libjpeg-8|zlib1)\.dll|libraw/licenses/[^:]+)$') { throw "Unexpected payload path: $relative" }
    if ($expected.ContainsKey($relative)) { throw "Duplicate manifest path: $relative" }
    $expected[$relative] = $hash
}
if (-not $expected.ContainsKey('aurora-studio.exe')) { throw 'Manifest has no executable' }
New-Item -ItemType Directory -Force -Path $base | Out-Null
New-Item -ItemType Directory -Path $stage | Out-Null
$movedOld = $false; $movedNew = $false
try {
    foreach ($relative in $expected.Keys) {
        $dest = Join-Path $stage $relative
        New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
        Copy-Item -LiteralPath (Join-Path $pkg $relative) -Destination $dest
        if ((Get-FileHash -LiteralPath $dest -Algorithm SHA256).Hash -ne $expected[$relative]) { throw "Checksum mismatch: $relative" }
    }
    Copy-Item -LiteralPath $manifest -Destination $stage
    $shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($pendingLink)
    $shortcut.TargetPath=$exe; $shortcut.WorkingDirectory=$install; $shortcut.IconLocation="$exe,0"; $shortcut.Description='Aurora Studio · 非破坏性照片编辑'; $shortcut.Save()
    if (Test-Path -LiteralPath $install) { Move-Item -LiteralPath $install -Destination $previous; $movedOld=$true }
    Move-Item -LiteralPath $stage -Destination $install; $movedNew=$true
    if (Test-Path -LiteralPath $link) { [IO.File]::Replace($pendingLink, $link, $backupLink) } else { [IO.File]::Move($pendingLink, $link) }
} catch {
    if ($movedNew) { Move-Item -LiteralPath $install -Destination $stage }
    if ($movedOld) { Move-Item -LiteralPath $previous -Destination $install }
    throw
} finally {
    if (Test-Path -LiteralPath $pendingLink) { Remove-Item -LiteralPath $pendingLink -Force }
    if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
}
[pscustomobject]@{Program=$exe; Desktop=$link; SHA256=$expected['aurora-studio.exe']; VerifiedFiles=$expected.Count; Previous=if($movedOld){$previous}else{$null}; PreviousShortcut=if(Test-Path $backupLink){$backupLink}else{$null}}
