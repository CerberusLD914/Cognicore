@echo off
REM ===================================================================
REM  CogniCore - ENTRENAMIENTO CONTINUO CON LOGS EN VIVO
REM
REM  Corre hasta que tu lo pares con Ctrl+C.
REM  Cada pocos segundos muestra loss, bits/byte, validacion, LR,
REM  gradiente, velocidad y si sigue mejorando o se estanco.
REM
REM  Opciones (editalas aqui o pasalas por consola):
REM    py live.py --params 10000000 --batch 8 --seq 256
REM    py live.py --params 50000000 --lr 0.002
REM
REM  Para parar:  Ctrl+C
REM ===================================================================

setlocal
cd /d "%~dp0"
title CogniCore - entrenamiento continuo

where py >nul 2>&1
if errorlevel 1 goto nopython

py -m pip install --user --quiet numpy >nul 2>&1

echo.
echo  ==============================================================
echo    COGNICORE - entrenamiento continuo
echo  ==============================================================
echo.
echo    El proceso corre hasta que pulses Ctrl+C.
echo    Ctrl+C guarda los pesos y te muestra el resumen final.
echo.
echo    Para un modelo mas grande:   py live.py --params 50000000
echo    Para logar mas seguido:      py live.py --every 10
echo.
echo  --------------------------------------------------------------
echo.

py live.py --params 10000000 --batch 8 --seq 256 --every 25 --eval-every 500

echo.
echo  Entrenamiento detenido. Pesos guardados en  checkpoints\
echo  Para hablar con el modelo:  chat.cmd
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
