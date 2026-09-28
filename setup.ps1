<#
    setup.ps1 - one-time bootstrap for the camera streaming tools.

    Ensures the two external dependencies the stream path needs are present:
      * FFmpeg  (winget: Gyan.FFmpeg)   - record / snap / play
      * VLC     (winget: VideoLAN.VLC)  - GUI playback

    Verifies Python 3 and reports its version. Note that cam_stream.py uses
    only the Python standard library, so no pip packages are required for the
    core stream/view/record flow.

    Safe to re-run: anything already present is detected and skipped.
#>

[CmdletBinding()]
param(
    [switch]$SkipStream,
    [switch]$NoLaunch
)

$ErrorActionPreference = 'Continue'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Step  { param($m) Write-Host "==> $m" -ForegroundColor Cyan }
function Write-Ok    { param($m) Write-Host "    [ok] $m" -ForegroundColor Green }
function Write-Warn2 { param($m) Write-Host "    [!!] $m" -ForegroundColor Yellow }
function Write-Err   { param($m) Write-Host "    [XX] $m" -ForegroundColor Red }

# ---------------------------------------------------------------------------
# PATH refresh. winget installs do not update the PATH of an already-running
# process, so binaries written mid-session stay invisible. Re-read both hives.
# ---------------------------------------------------------------------------
function Update-PathFromRegistry {
    $parts = @()
    foreach ($scope in 'Machine', 'User') {
        $raw = [Environment]::GetEnvironmentVariable('Path', $scope)
        if ($raw) { $parts += $raw -split ';' }
    }
    foreach ($extra in @(
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links",
        "$env:ProgramFiles\VideoLAN\VLC",
        "${env:ProgramFiles(x86)}\VideoLAN\VLC"
    )) {
        if ($extra -and (Test-Path -LiteralPath $extra)) { $parts += $extra }
    }
    $merged = @()
    foreach ($p in $parts) {
        if ($p -and -not ($merged -contains $p)) { $merged += $p }
    }
    $env:Path = ($merged -join ';')
}

# ---------------------------------------------------------------------------
# winget wrapper. Always passes the agreement flags so it never blocks on a
# dialog, which is what makes true one-click operation possible.
# ---------------------------------------------------------------------------
function Test-Winget {
    $null -ne (Get-Command winget -ErrorAction SilentlyContinue)
}

function Install-WingetPackage {
    param(
        [Parameter(Mandatory)][string]$Id,
        [string]$Label = $Id
    )
    $args = @(
        'install', '--id', $Id, '-e',
        '--accept-package-agreements',
        '--accept-source-agreements',
        '--disable-interactivity'
    )
    Write-Host "    installing $Label via winget ($Id) ..." -ForegroundColor DarkGray
    $proc = Start-Process -FilePath 'winget' -ArgumentList $args `
        -NoNewWindow -Wait -PassThru
    if ($proc.ExitCode -eq 0) {
        Write-Ok "$Label installed"
        return $true
    }
    # 0x8A150011 = APPINSTALLER_CLI_ERROR_NO_APPLICABLE_INSTALLER
    Write-Err "winget could not install $Label (exit $($proc.ExitCode))"
    return $false
}

function Resolve-VlcPath {
    $cmd = Get-Command vlc -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @(
        "$env:ProgramFiles\VideoLAN\VLC\vlc.exe",
        "${env:ProgramFiles(x86)}\VideoLAN\VLC\vlc.exe",
        "$env:LOCALAPPDATA\Programs\VideoLAN\VLC\vlc.exe"
    )) {
        if (Test-Path -LiteralPath $p) { return $p }
    }
    return $null
}

function Resolve-FfmpegPath {
    $cmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @(
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\ffmpeg.exe"
    )) {
        if (Test-Path -LiteralPath $p) { return $p }
    }
    return $null
}

# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '  Camera stream bootstrap' -ForegroundColor White
Write-Host '  -----------------------' -ForegroundColor DarkGray
Write-Host ''

Update-PathFromRegistry

# --- Python ---------------------------------------------------------------
Write-Step 'Checking Python'
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue }
if ($py) {
    $ver = (& $py.Source --version 2>&1) -join ' '
    Write-Ok "$ver  ($($py.Source))"
} else {
    Write-Warn2 'Python not found on PATH.'
    if (Test-Winget) {
        Write-Host '    attempting install via winget ...' -ForegroundColor DarkGray
        Install-WingetPackage -Id 'Python.Python.3.13' -Label 'Python 3.13' | Out-Null
        Update-PathFromRegistry
        $py = Get-Command python -ErrorAction SilentlyContinue
        if ($py) { Write-Ok "installed: $(& $py.Source --version 2>&1)" }
        else { Write-Err 'Python still missing. Install from python.org and re-run.' }
    } else {
        Write-Err 'winget unavailable - install Python from python.org, then re-run.'
    }
}
Write-Host '    (cam_stream.py needs no pip packages - standard library only)'

# --- FFmpeg ---------------------------------------------------------------
Write-Step 'Checking FFmpeg'
if (Resolve-FfmpegPath) {
    Write-Ok "present: $(Resolve-FfmpegPath)"
} else {
    if (Test-Winget) {
        Install-WingetPackage -Id 'Gyan.FFmpeg' -Label 'FFmpeg' | Out-Null
        Update-PathFromRegistry
    } else {
        Write-Err 'winget unavailable. Download FFmpeg manually and add it to PATH.'
    }
    if (Resolve-FfmpegPath) { Write-Ok "present: $(Resolve-FfmpegPath)" }
    else { Write-Err 'FFmpeg still missing - record/snap/play will not work.' }
}

# --- VLC ------------------------------------------------------------------
Write-Step 'Checking VLC media player'
$vlc = Resolve-VlcPath
if ($vlc) {
    Write-Ok "present: $vlc"
} else {
    if (Test-Winget) {
        Install-WingetPackage -Id 'VideoLAN.VLC' -Label 'VLC' | Out-Null
        Update-PathFromRegistry
    } else {
        Write-Err 'winget unavailable. Install VLC manually from videolan.org.'
    }
    $vlc = Resolve-VlcPath
    if ($vlc) { Write-Ok "present: $vlc" }
    else { Write-Err 'VLC still missing - the GUI player will not work.' }
}

# --- Config ---------------------------------------------------------------
$envFile = Join-Path $scriptDir 'camera.env'
if (-not (Test-Path -LiteralPath $envFile)) {
    Copy-Item -LiteralPath (Join-Path $scriptDir 'camera.env.example') `
              -Destination $envFile -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $envFile) { Write-Ok 'created camera.env from template' }
}

# --- Summary --------------------------------------------------------------
Write-Host ''
Write-Step 'Summary'
$cam = Join-Path $scriptDir 'cam_stream.py'
if (Test-Path -LiteralPath $cam) { Write-Ok "cam_stream.py present" }
else { Write-Err "cam_stream.py missing from $scriptDir" }

Write-Host ''
Write-Host '  Ready. Next steps:' -ForegroundColor White
Write-Host '    1. Edit camera.env and set CAM_HOST / CAM_USER / CAM_PASS' -ForegroundColor Gray
Write-Host '    2. Double-click start_stream.bat  to watch the live stream' -ForegroundColor Gray
Write-Host '    3. Open a terminal here for snap / record / probe:' -ForegroundColor Gray
Write-Host '         python cam_stream.py snap' -ForegroundColor DarkGray
Write-Host '         python cam_stream.py record 30' -ForegroundColor DarkGray
Write-Host ''
Write-Warn2 'If playback fails with a connection reset, the camera has IP-banned' 
Write-Warn2 'this machine. Wait it out and re-run - that is not a bad password.'
Write-Host ''

if ($SkipStream -or $NoLaunch) { exit 0 }

# --- Launch ---------------------------------------------------------------
if ($vlc) {
    $answer = Read-Host '  Start the stream now? [Y/n]'
    if ($answer -eq '' -or $answer -match '^[Yy]') {
        Start-Process -FilePath $vlc `
            -ArgumentList '--rtsp-tcp', 'rtsp://root:1234567890@117.196.244.183:554/Streaming/Channels/101'
        Write-Ok 'VLC launched'
    }
}
exit 0
