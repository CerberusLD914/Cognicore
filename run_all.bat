@echo off
REM ===================================================================
REM  CogniCore - full training
REM  Verifies gradients, downloads the dataset, trains a 10M parameter
REM  non-LLM model on CPU, evaluates it and quantises to int8.
REM ===================================================================

setlocal
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 goto nopython

py -m pip install --user --quiet numpy >nul 2>&1

echo.
echo   Training 10M parameters on CPU. This takes a while.
echo   The first run downloads the dataset.
echo.
py run_all.py --params 10000000 --steps 1500 --batch 8 --seq 256
if errorlevel 1 goto failed

echo.
echo   Done. Weights are in checkpoints\
echo   Talk to it with:  chat.cmd
echo.
pause
exit /b 0

:failed
echo.
echo   Training failed. Run  py gradcheck.py  to see what broke.
echo.
pause
exit /b 1

:nopython
echo.
echo   Python not found. Install it from python.org
echo   and tick "Add python.exe to PATH".
echo.
pause
exit /b 1
