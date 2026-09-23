@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\yt-transcribe.exe" (
    echo [setup] First run - creating virtual environment and installing dependencies...
    py -3 -m venv .venv
    if errorlevel 1 goto :fail
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\pip.exe" install -e .
    if errorlevel 1 goto :fail
)

if "%~1"=="" (
    echo [run] Starting local GUI + API server and opening your browser...
    ".venv\Scripts\yt-transcribe.exe" serve --open
) else (
    ".venv\Scripts\yt-transcribe.exe" %*
)
goto :eof

:fail
echo.
echo Setup failed. Make sure Python 3.10+ is installed and on PATH.
pause