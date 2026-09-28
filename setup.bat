@echo off
setlocal
rem ---------------------------------------------------------------
rem  One-click bootstrap. Double-click this file.
rem  Installs/verifies Python, FFmpeg and VLC, then offers to play.
rem ---------------------------------------------------------------

cd /d "%~dp0"

echo.
echo   Camera stream setup
echo   ===================
echo.

where powershell >nul 2>&1
if errorlevel 1 (
    echo   [XX] Windows PowerShell not found - cannot continue.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
set RC=%ERRORLEVEL%

echo.
if "%RC%"=="0" (
    echo   Setup finished.
) else (
    echo   Setup finished with exit code %RC%.
)
echo.
pause
endlocal
