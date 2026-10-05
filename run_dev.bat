@echo off
REM ==========================================================================
REM  Supertonic TTS - development launcher
REM
REM  Runs the app straight from source using the .venv if it exists,
REM  otherwise the Python on PATH. No packaging, no build step.
REM ==========================================================================
setlocal
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
  where py >nul 2>nul && set "PY=py -3.12"
  if not defined PY where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo.
  echo [ERROR] No Python interpreter found.
  echo         Install Python 3.11 or 3.12 from https://www.python.org/downloads/
  echo         and tick "Add python.exe to PATH" during setup.
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [INFO]  No virtual environment found - using the system Python.
  echo         To create one:  python -m venv .venv
  echo.
)

echo [INFO]  Interpreter: %PY%
echo [INFO]  Starting Supertonic TTS ...
echo.
%PY% -m app.main --verbose
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo [ERROR] Supertonic TTS exited with code %RC%.
  echo         See logs\app.log for details.
  echo.
  pause
)
endlocal
exit /b %RC%
