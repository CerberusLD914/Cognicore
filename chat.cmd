@echo off
REM ===================================================================
REM  CogniCore - interactive console
REM
REM  Type text and press Enter, the model continues it.
REM
REM  Commands:
REM    :stats             model info
REM    :temp 0.9          sampling temperature
REM    :topk 40           top-k cutoff
REM    :n 400             how many bytes to generate
REM    :seed 3            random seed
REM    :compare some text average surprise per byte of your text
REM    :quant             int8 quantisation report
REM    :save out.py       write the generation to a file
REM    :quit              exit
REM
REM  NOTE: this file is plain ASCII on purpose. cmd.exe reads .cmd files
REM  using the system ANSI codepage, so accented characters in a UTF-8
REM  batch file silently corrupt the parser and break the if-blocks.
REM ===================================================================

setlocal
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 goto nopython

py -m pip install --user --quiet numpy >nul 2>&1

if exist "checkpoints\*.npz" goto run
if exist "checkpoints" goto run

echo.
echo   No trained model found yet.
echo.
echo   Quick test first   : "test rapido.cmd"   (2M params, about 5 min)
echo   Full training      : run_all.bat          (10M params)
echo.
pause
exit /b 1

:run
py chat.py %*
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
