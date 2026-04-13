@echo off
chcp 65001 >nul
title Lab Panel

cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [HATA] Python bulunamadi.
    pause
    exit /b 1
)

if exist __pycache__ rmdir /s /q __pycache__ >nul 2>&1

python db_repair.py

python lab_panel.py

if errorlevel 1 (
    echo [HATA] Panel kapandi.
    pause
)
