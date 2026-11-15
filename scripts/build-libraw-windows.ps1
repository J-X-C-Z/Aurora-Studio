<#
.SYNOPSIS
Recreates vendor/libraw from the MSYS2 mingw-w64-x86_64-libraw package.

.DESCRIPTION
This build does not compile libraw from source. It installs the official MSYS2
mingw64 binary package and copies the runtime DLLs plus headers into
vendor/libraw, so the app can LoadLibraryW the C API at run time.

Source: https://packages.msys2.org/package/mingw-w64-x86_64-libraw
Version: 0.22.3-1 (libraw-25.dll)
#>

$ErrorActionPreference = 'Stop'
$msys = 'C:\msys64'
$dest = Join-Path $PSScriptRoot '..\vendor\libraw'

& "$msys\usr\bin\bash.exe" -lc 'pacman -S --noconfirm mingw-w64-x86_64-libraw'

New-Item -ItemType Directory -Force `
    "$dest\bin", "$dest\include\libraw" | Out-Null
Copy-Item "$msys\mingw64\bin\libraw-25.dll", `
    "$msys\mingw64\bin\libjpeg-8.dll", "$msys\mingw64\bin\liblcms2-2.dll", `
    "$msys\mingw64\bin\libgcc_s_seh-1.dll", "$msys\mingw64\bin\libgomp-1.dll", `
    "$msys\mingw64\bin\libstdc++-6.dll", "$msys\mingw64\bin\zlib1.dll" `
    -Destination "$dest\bin" -Force
Copy-Item "$msys\mingw64\include\libraw\*" "$dest\include\libraw" -Recurse -Force
