@echo off
REM ==============================================================================
REM Local Jarvis - Windows Batch Launcher
REM Activates the Python virtual environment and launches the pipeline orchestrator.
REM
REM Usage:
REM   start_jarvis.bat                 (standard startup)
REM   start_jarvis.bat --no-tray       (headless console mode)
REM   start_jarvis.bat --threshold 0.4 (custom wake threshold)
REM ==============================================================================

setlocal enabledelayedexpansion

REM Resolve project root directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Verify virtual environment exists
if not exist "%SCRIPT_DIR%.venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at %SCRIPT_DIR%.venv
    echo Please create the virtual environment first using:
    echo   python -m venv .venv
    pause
    exit /b 1
)

REM Activate virtual environment
call "%SCRIPT_DIR%.venv\Scripts\activate.bat"

REM Run Local Jarvis orchestrator
python "%SCRIPT_DIR%src\orchestrator.py" %*

exit /b %ERRORLEVEL%
