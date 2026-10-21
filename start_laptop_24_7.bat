@echo off
title Acer Aspire A515-45 24/7 Autonomous Bagging Daemon
echo ==============================================================
echo STARTING ACER ASPIRE A515-45 24/7 GBDT BAGGING DAEMON
echo ==============================================================
echo Operating Mode: High-Efficiency CPU Multi-Threading (15W TDP)
echo Checkpoint File: output\laptop_24_7_acer_checkpoint.npz
echo.
python scratch\laptop_24_7_daemon_acer.py
pause
