@echo off
rem Build the PF8025/PF8026 V2FuseOffset stage PTX.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "SDK=C:\Program Files (x86)\Windows Kits\10"
set "INCLUDE=%INCLUDE%;%SDK%\Include\10.0.26100.0\ucrt;%SDK%\Include\10.0.26100.0\um;%SDK%\Include\10.0.26100.0\shared"
set "LIB=%LIB%;%SDK%\Lib\10.0.26100.0\ucrt\x64;%SDK%\Lib\10.0.26100.0\um\x64"
cd /d "%~dp0..\..\..\native\cuda"
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe" ^
  -std=c++17 -O3 -fmad=false -gencode arch=compute_89,code=sm_89 -ptx ^
  v2fuseoffset.cu -o v2fuseoffset.ptx
echo nvcc exit=%ERRORLEVEL%
endlocal
