@echo off
set count= 1
:loop
if %count% leq 15 (
    echo Starting training script %count%/15...
    python train.py
    echo Finished training script %count%/15
    set /a count+=1
    goto loop
)

echo All training scripts have been executed.

@REM 訓練完後即關機
@REM shutdown /s /f /t 60
@REM pause