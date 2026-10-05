@echo off
REM ==========================================================================
REM  Supertonic TTS - build the portable Windows app
REM
REM  Produces  dist\SupertonicTTS\SupertonicTTS.exe  plus models\, output\
REM  and a short usage note. Copy that whole folder to any Windows 11 PC.
REM
REM  Usage:
REM    build_windows.bat              full build into a fresh .venv
REM    build_windows.bat --with-model also copy the model into dist\
REM    build_windows.bat --test       run the unit tests before building
REM ==========================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "WITH_MODEL="
set "RUN_TESTS="
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--with-model" set "WITH_MODEL=1"
if /i "%~1"=="--test" set "RUN_TESTS=1"
shift
goto parse
:parsed

echo.
echo ==========================================================
echo  Supertonic TTS - portable build
echo ==========================================================
echo.

REM ---------------------------------------------------------------- Python ---
set "BOOTSTRAP="
if exist ".venv\Scripts\python.exe" set "BOOTSTRAP=.venv\Scripts\python.exe"
if not defined BOOTSTRAP where py >nul 2>nul && set "BOOTSTRAP=py -3.12"
if not defined BOOTSTRAP where python >nul 2>nul && set "BOOTSTRAP=python"
if not defined BOOTSTRAP (
  echo [ERROR] Python was not found on this machine.
  echo         Install Python 3.11 or 3.12 from https://www.python.org/downloads/
  echo         and tick "Add python.exe to PATH" during setup.
  echo.
  pause
  exit /b 1
)
echo [1/6] Interpreter: %BOOTSTRAP%
%BOOTSTRAP% -c "import sys; assert sys.version_info >= (3,10), 'need Python 3.10+'; print('         Python', '.'.join(map(str, sys.version_info[:3])))"
if errorlevel 1 (
  echo.
  pause
  exit /b 1
)

REM ------------------------------------------------------------- Virtualenv ---
if not exist ".venv\Scripts\python.exe" (
  echo [2/6] Creating virtual environment .venv ...
  %BOOTSTRAP% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Could not create the virtual environment.
    pause
    exit /b 1
  )
) else (
  echo [2/6] Reusing existing .venv
)
set "VENV_PY=.venv\Scripts\python.exe"

REM -------------------------------------------------------------- Upgrading ---
echo [3/6] Installing dependencies ...
"%VENV_PY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
if errorlevel 1 goto :fail
"%VENV_PY%" -m pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 goto :fail

if defined RUN_TESTS (
  echo [4/6] Running tests ...
  "%VENV_PY%" -m unittest discover -s tests -p "test_*.py"
  if errorlevel 1 goto :fail
) else (
  echo [4/6] Skipping tests ^(pass --test to run them^)
)

REM ----------------------------------------------------------------- Build ---
echo [5/6] Building with PyInstaller ...
if exist "build" rmdir /s /q "build"
if exist "dist\SupertonicTTS" rmdir /s /q "dist\SupertonicTTS"
"%VENV_PY%" -m PyInstaller --noconfirm --clean SupertonicTTS.spec
if errorlevel 1 goto :fail

REM ----------------------------------------------------------- Bundle layout ---
echo [6/6] Assembling the portable folder ...
for %%D in ("models" "output" "logs" "voices") do (
  if not exist "dist\SupertonicTTS\%%~D" mkdir "dist\SupertonicTTS\%%~D"
)

if defined WITH_MODEL (
  if exist "models\supertonic\onnx\vocoder.onnx" (
    echo       Copying model files ^(about 400 MB^) ...
    xcopy "models\supertonic" "dist\SupertonicTTS\models\supertonic\" /E /Q /Y >nul
  ) else (
    echo       [WARN] No model found at models\supertonic - run download_model.py first.
  )
)

copy /y "README.txt" "dist\SupertonicTTS\" >nul 2>nul
if not exist "dist\SupertonicTTS\LICENSES" mkdir "dist\SupertonicTTS\LICENSES"
copy /y "models\supertonic\LICENSE" "dist\SupertonicTTS\LICENSES\Supertonic-MODEL-LICENSE.txt" >nul 2>nul
copy /y "LICENSES\README.md" "dist\SupertonicTTS\LICENSES\" >nul 2>nul

echo.
echo ==========================================================
echo  Build finished
echo ==========================================================
echo.
echo   Output:  %CD%\dist\SupertonicTTS\
echo.
dir /b "dist\SupertonicTTS" | findstr /i "exe"
echo.
if defined WITH_MODEL (
  echo   The model IS included - this folder is ready to copy anywhere.
) else (
  echo   The model is NOT included. To ship it with the app:
  echo       build_windows.bat --with-model
  echo   Otherwise copy models\supertonic into the dist folder by hand,
  echo   or run download_model.py on the target PC.
)
echo.
echo   Test it with:  dist\SupertonicTTS\SupertonicTTS.exe
echo.
pause
endlocal
exit /b 0

:fail
echo.
echo [ERROR] The build failed. The message above says why.
echo.
pause
endlocal
exit /b 1
