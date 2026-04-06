@echo off
REM ---------------------------------------------------------------
REM  Start bundled PulseAudio TCP server for Docker ADAS audio
REM
REM  PulseAudio 1.1 is shipped in dep\pulseaudio-1.1.
REM  This script starts it on port 4713 so the Docker container
REM  can stream audio via PULSE_SERVER=tcp:host.docker.internal:4713
REM
REM  Usage: run from the repo root:  scripts\start_pulseaudio.bat
REM ---------------------------------------------------------------

REM -- Resolve repo root (parent of scripts\) --
set "REPO_ROOT=%~dp0.."
set "PA_DIR=%REPO_ROOT%\dep\pulseaudio-1.1"

REM -- Verify bundled PulseAudio exists --
if not exist "%PA_DIR%\bin\pulseaudio.exe" (
    echo.
    echo [ADAS] ERROR: Bundled PulseAudio not found at:
    echo        %PA_DIR%\bin\pulseaudio.exe
    echo.
    echo   Make sure the dep\pulseaudio-1.1 folder is present in the repo.
    echo.
    pause
    exit /b 1
)

REM -- Check if already running on port 4713 --
netstat -an 2>nul | findstr ":4713" >nul 2>&1
if %ERRORLEVEL%==0 (
    echo [ADAS] PulseAudio is already listening on port 4713.
    echo        Audio should work. You can close this window.
    pause
    exit /b 0
)

echo [ADAS] Starting PulseAudio from dep\pulseaudio-1.1 ...
echo        TCP server on port 4713.  Press Ctrl+C to stop.
echo.

"%PA_DIR%\bin\pulseaudio.exe" --exit-idle-time=-1 -F "%PA_DIR%\etc\pulse\default.pa" --daemonize=no
