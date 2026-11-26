<#
.SYNOPSIS
    Smoke-tests the native GUI off-screen: the app runs on a private window
    station desktop, so no window ever appears on the interactive desktop.

.DESCRIPTION
    The release exe is `windows_subsystem = "windows"`: a startup panic leaves no
    trace unless stdout/stderr are redirected, and asking a human to watch a
    window is not an automated check. This script:

      1. creates a hidden desktop (`CreateDesktopW`) in the current window station;
      2. starts aurora-studio.exe with STARTUPINFO.lpDesktop set to it and
         stdout/stderr redirected to files;
      3. waits `-Seconds`, then reports whether the process is still alive and
         which windows exist on that desktop (class, title, rectangle);
      4. terminates the process and closes the desktop.

    Nothing is shown on the desktop the user is looking at; this is a headless
    equivalent of "does the GPUI window come up at all".

.PARAMETER Exe
    Path to aurora-studio.exe.

.PARAMETER Seconds
    Observation window before the process is terminated.

.PARAMETER DesktopName
    Name of the private desktop; defaults to a unique name per run.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/smoke-windows-hidden.ps1
#>
[CmdletBinding()]
param(
    [string]$Exe = (Join-Path (Split-Path $PSScriptRoot -Parent) 'target/dist/aurora-studio.exe'),
    [int]$Seconds = 10,
    [string]$DesktopName = ''
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Exe)) { throw "missing exe: $Exe" }
$logDir = (Join-Path (Split-Path $PSScriptRoot -Parent) 'target/diagnostics')
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
if (-not $DesktopName) { $DesktopName = "AuroraSmoke-$stamp" }
$outFile = Join-Path $logDir "smoke-$stamp.out.log"
$errFile = Join-Path $logDir "smoke-$stamp.err.log"

$source = @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

public static class HiddenDesktopSmoke
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct STARTUPINFO
    {
        public int cb;
        public string lpReserved;
        public string lpDesktop;
        public string lpTitle;
        public int dwX, dwY, dwXSize, dwYSize, dwXCountChars, dwYCountChars, dwFillAttribute, dwFlags;
        public short wShowWindow, cbReserved2;
        public IntPtr lpReserved2, hStdInput, hStdOutput, hStdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct PROCESS_INFORMATION
    {
        public IntPtr hProcess, hThread;
        public int dwProcessId, dwThreadId;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SECURITY_ATTRIBUTES
    {
        public int nLength;
        public IntPtr lpSecurityDescriptor;
        public int bInheritHandle;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left, Top, Right, Bottom; }

    [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern IntPtr CreateDesktopW(string name, IntPtr dev, IntPtr devmode, int flags, uint access, IntPtr sa);
    [DllImport("user32.dll", SetLastError = true)]
    static extern bool CloseDesktop(IntPtr h);
    [DllImport("user32.dll", SetLastError = true)]
    static extern bool EnumDesktopWindows(IntPtr h, EnumWindowsProc cb, IntPtr p);
    delegate bool EnumWindowsProc(IntPtr hwnd, IntPtr p);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    static extern int GetClassNameW(IntPtr h, StringBuilder sb, int max);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    static extern int GetWindowTextW(IntPtr h, StringBuilder sb, int max);
    [DllImport("user32.dll")]
    static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")]
    static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern bool CreateProcessW(string app, string cmd, IntPtr pa, IntPtr ta, bool inherit,
        uint flags, IntPtr env, string cwd, ref STARTUPINFO si, out PROCESS_INFORMATION pi);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool TerminateProcess(IntPtr h, uint code);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern uint WaitForSingleObject(IntPtr h, uint ms);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool CloseHandle(IntPtr h);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern IntPtr CreateFileW(string name, uint access, uint share, ref SECURITY_ATTRIBUTES sa,
        uint disposition, uint flags, IntPtr template);

    const uint DESKTOP_ALL = 0x01FF;
    const uint GENERIC_WRITE = 0x40000000;
    const uint FILE_SHARE_WRITE = 2;
    const uint CREATE_ALWAYS = 2;
    const uint FILE_ATTRIBUTE_NORMAL = 0x80;
    const int STARTF_USESTDHANDLES = 0x100;
    const short SW_SHOWNORMAL = 1;
    const uint WAIT_TIMEOUT = 0x102;

    public static string Run(string exe, string desktopName, int seconds, string outPath, string errPath)
    {
        var sb = new StringBuilder();
        IntPtr desk = CreateDesktopW(desktopName, IntPtr.Zero, IntPtr.Zero, 0, DESKTOP_ALL, IntPtr.Zero);
        if (desk == IntPtr.Zero)
            return "FAIL CreateDesktopW error=" + Marshal.GetLastWin32Error();

        var sa = new SECURITY_ATTRIBUTES { nLength = Marshal.SizeOf(typeof(SECURITY_ATTRIBUTES)), bInheritHandle = 1 };
        IntPtr hOut = CreateFileW(outPath, GENERIC_WRITE, FILE_SHARE_WRITE, ref sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, IntPtr.Zero);
        IntPtr hErr = CreateFileW(errPath, GENERIC_WRITE, FILE_SHARE_WRITE, ref sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, IntPtr.Zero);

        var si = new STARTUPINFO();
        si.cb = Marshal.SizeOf(typeof(STARTUPINFO));
        si.lpDesktop = desktopName;
        si.dwFlags = STARTF_USESTDHANDLES;
        si.wShowWindow = SW_SHOWNORMAL;
        si.hStdOutput = hOut;
        si.hStdError = hErr;
        si.hStdInput = IntPtr.Zero;

        PROCESS_INFORMATION pi;
        bool ok = CreateProcessW(exe, null, IntPtr.Zero, IntPtr.Zero, true, 0, IntPtr.Zero,
            System.IO.Path.GetDirectoryName(exe), ref si, out pi);
        if (!ok)
        {
            CloseDesktop(desk);
            return "FAIL CreateProcessW error=" + Marshal.GetLastWin32Error();
        }

        sb.AppendLine("pid=" + pi.dwProcessId);
        uint wait = WaitForSingleObject(pi.hProcess, (uint)(seconds * 1000));
        bool alive = wait == WAIT_TIMEOUT;
        sb.AppendLine("alive_after_" + seconds + "s=" + alive);

        int windows = 0;
        EnumDesktopWindows(desk, (hwnd, p) =>
        {
            var cls = new StringBuilder(256);
            var title = new StringBuilder(512);
            GetClassNameW(hwnd, cls, cls.Capacity);
            GetWindowTextW(hwnd, title, title.Capacity);
            RECT r; GetWindowRect(hwnd, out r);
            if (cls.Length > 0 || title.Length > 0)
            {
                windows++;
                sb.AppendLine(string.Format("window class='{0}' title='{1}' visible={2} rect={3},{4},{5},{6}",
                    cls, title, IsWindowVisible(hwnd), r.Left, r.Top, r.Right, r.Bottom));
            }
            return true;
        }, IntPtr.Zero);
        sb.AppendLine("windows=" + windows);

        if (alive) { TerminateProcess(pi.hProcess, 1); WaitForSingleObject(pi.hProcess, 5000); }
        CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
        if (hOut != IntPtr.Zero) CloseHandle(hOut);
        if (hErr != IntPtr.Zero) CloseHandle(hErr);
        CloseDesktop(desk);
        return sb.ToString();
    }
}
'@

Add-Type -TypeDefinition $source -Language CSharp
$result = [HiddenDesktopSmoke]::Run($Exe, $DesktopName, $Seconds, $outFile, $errFile)
$result

Write-Host ''
Write-Host "stdout: $outFile"
Write-Host "stderr: $errFile"
Start-Sleep -Milliseconds 300   # the child's handles close asynchronously
foreach ($file in @($errFile, $outFile)) {
    try {
        if ((Test-Path $file) -and (Get-Item $file -ErrorAction Stop).Length -gt 0) {
            Write-Host "--- $([IO.Path]::GetFileName($file)) tail ---"
            Get-Content $file -Tail 40 -ErrorAction Stop
        }
    } catch {
        Write-Host "--- $([IO.Path]::GetFileName($file)) unreadable: $($_.Exception.Message)"
    }
}
