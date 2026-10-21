@echo off
title Acer Aspire A515-45 24/7 Autonomous Bagging Daemon
echo ==============================================================
echo STARTING ACER ASPIRE A515-45 24/7 GBDT BAGGING DAEMON
echo ==============================================================
echo Operating Mode: High-Efficiency CPU Multi-Threading (15W TDP)
echo Checkpoint File: output\laptop_24_7_acer_checkpoint.npz
echo.
if exist laptop_24_7_daemon_acer.py (
    python laptop_24_7_daemon_acer.py
) else if exist scratch\laptop_24_7_daemon_acer.py (
    python scratch\laptop_24_7_daemon_acer.py
) else (
    echo Error: laptop_24_7_daemon_acer.py not found in current folder or scratch folder!
)
pause
