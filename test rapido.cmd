@echo off
REM ===================================================================
REM  CogniCore - quick test
REM  Trains a 2M parameter model for 250 steps (about 5 minutes) and
REM  shows that the loss drops and the model predicts real bytes.
REM  Not the final model, just a check that everything works.
REM ===================================================================

setlocal
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 goto nopython

py -m pip install --user --quiet numpy >nul 2>&1

echo.
echo   Quick test: 2M params, 250 steps, about 5 minutes.
echo   The first run downloads the dataset.
echo.
py quicktest.py 250
echo.
pause
exit /b 0

:nopython
echo.
echo   Python not found. Install it from python.org
echo   and tick "Add python.exe to PATH".
echo.
pause
exit /b 1
