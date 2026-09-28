@echo off
setlocal enabledelayedexpansion
rem ---------------------------------------------------------------
rem  One-click live stream. Double-click this file.
rem
rem  Reads target settings from camera.env next to this script, then
rem  opens the stream in VLC. If VLC is absent it falls back to
rem  ffmpeg/ffplay, which is enough to view but has no GUI controls.
rem ---------------------------------------------------------------

cd /d "%~dp0"

rem ---- defaults, overridden by camera.env if present ----------------
if not defined CAM_HOST  set "CAM_HOST=117.196.244.183"
if not defined CAM_PORT  set "CAM_PORT=554"
if not defined CAM_USER  set "CAM_USER=root"
if not defined CAM_PASS  set "CAM_PASS=1234567890"
if not defined CAM_PATH  set "CAM_PATH=/Streaming/Channels/101"

if exist "%~dp0camera.env" call "%~dp0camera.env"

set "URL=rtsp://%CAM_USER%:%CAM_PASS%@%CAM_HOST%:%CAM_PORT%%CAM_PATH%"

echo.
echo   Target : %CAM_HOST%:%CAM_PORT%%CAM_PATH%
echo   Stream : rtsp://%CAM_USER%:**@%CAM_HOST%:%CAM_PORT%%CAM_PATH%
echo.

rem ---- locate VLC ----------------------------------------------------
rem Sequential checks rather than a for-loop: "%ProgramFiles(x86)%" expands
rem to a value containing literal parentheses, which would terminate a
rem parenthesised block mid-parse.
set "VLC="
if not defined VLC if exist "%ProgramFiles%\VideoLAN\VLC\vlc.exe" set "VLC=%ProgramFiles%\VideoLAN\VLC\vlc.exe"
if not defined VLC if exist "%LOCALAPPDATA%\Programs\VideoLAN\VLC\vlc.exe" set "VLC=%LOCALAPPDATA%\Programs\VideoLAN\VLC\vlc.exe"
if not defined VLC call :try_x86
if not defined VLC for /f "delims=" %%I in ('where vlc 2^>nul') do if not defined VLC set "VLC=%%I"

if defined VLC (
    echo   Launching VLC...
    start "" "%VLC%" --rtsp-tcp "%URL%"
    goto :done
)

rem ---- fall back to ffplay -------------------------------------------
where ffplay >nul 2>&1
if not errorlevel 1 (
    echo   VLC not found - using ffplay instead.
    ffplay -rtsp_transport tcp "%URL%"
    goto :done
)

echo   [XX] Neither VLC nor ffplay found.
echo        Run setup.bat first to install them.
goto :fail

rem ---- resolve the x86 Program Files location -------------------------
rem Lives outside any parenthesised block because the expanded value
rem contains literal parentheses.
:try_x86
set "PF86=%ProgramFiles(x86)%"
if not defined PF86 goto :eof
if not exist "%PF86%\VideoLAN\VLC\vlc.exe" goto :eof
set "VLC=%PF86%\VideoLAN\VLC\vlc.exe"
goto :eof

:done
echo.
echo   Playing. Close the player window to stop.
echo.
rem ping is used instead of `timeout` because timeout refuses redirected input
ping -n 4 127.0.0.1 >nul 2>&1
endlocal
exit /b 0

:fail
echo.
pause
endlocal
exit /b 1
