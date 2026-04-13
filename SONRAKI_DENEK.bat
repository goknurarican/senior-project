@echo off
chcp 65001 >nul
title BİTİRMEEG — Lab Panel

:: Scriptin bulunduğu klasöre git (BAT nerede olursa olsun)
cd /d "%~dp0"

:: Python yüklü mü kontrol et
where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [HATA] Python bulunamadi.
    echo  Lutfen bir arastirmaciya bildirin.
    echo.
    pause
    exit /b 1
)

:: Eski bytecode önbelleğini temizle (eski kod çalışmasını engeller)
if exist __pycache__ rmdir /s /q __pycache__ >nul 2>&1

:: Lab panelini aç
python lab_panel.py

:: Anormal çıkış olursa
if errorlevel 1 (
    echo.
    echo  [HATA] Panel beklenmedik sekilde kapandi.
    echo  Lutfen bir arastirmaciya bildirin.
    echo.
    pause
)
