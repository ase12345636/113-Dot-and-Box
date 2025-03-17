@echo off
set count= 1
:loop
if %count% leq 6 (
    echo Starting training script %count%/6...
    python train.py
    echo Finished training script %count%/6
    set /a count+=1
    goto loop
)

echo All training scripts have been executed.

@REM 訓練完後即關機
shutdown /s /f /t 60
pause