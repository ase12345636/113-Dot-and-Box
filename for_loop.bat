@echo off
set count= 1
:loop
if %count% leq 50 (
    echo Starting training script %count%/50...
    python train.py
    echo Finished training script %count%/50
    set /a count+=1
    goto loop
)

echo All training scripts have been executed.

@REM 訓練完後即關機
shutdown /s /f /t 60
pause