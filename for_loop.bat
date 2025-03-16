@echo off
set count= 1
:loop
if %count% leq 7 (
    echo Starting training script %count%/7...
    python train.py
    echo Finished training script %count%/7
    set /a count+=1
    goto loop
)

echo All training scripts have been executed.

@REM 訓練完後即關機
@REM shutdown /s /f /t 60
@REM pause