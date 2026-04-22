@echo off
chcp 65001 >nul
title BİTİRMEEG — Lab
cd /d "%~dp0"

:: ─────────────────────────────────────────────
::  Python kontrolü
:: ─────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [HATA] Python bulunamadi.
    echo  Kurmak icin: python.org/downloads
    echo.
    pause
    exit /b 1
)

:: ─────────────────────────────────────────────
::  Sunucu durumu — port 3000 dinleniyor mu?
:: ─────────────────────────────────────────────
netstat -aon 2>nul | findstr ":3000 " | findstr "LISTENING" >nul 2>&1
if errorlevel 1 goto :BASLA

:: ── Sunucu zaten çalışıyor → menü ────────────
echo.
echo  ============================================================
echo   BITİRMEEG  ^|  Sunucu calisiyor
echo  ============================================================
echo.
echo   [1]  Sonraki denege gec  ^(veri paketle^)
echo   [2]  Sistemi yeniden baslat
echo   [3]  Cikis
echo.
choice /c 123 /n /m "  Seciminiz (1/2/3): "
if errorlevel 3 exit /b 0
if errorlevel 2 goto :BASLA
if errorlevel 1 goto :PAKETLE

:: ─────────────────────────────────────────────
:PAKETLE
:: ─────────────────────────────────────────────
echo.
echo  [1/2] Veritabani onarilıyor...
if exist __pycache__ rmdir /s /q __pycache__ >nul 2>&1
python db_repair.py

echo  [2/2] Lab paneli aciliyor...
echo.
python lab_panel.py

if errorlevel 1 (
    echo.
    echo  [HATA] Panel kapandi — hata olustu.
    pause
)
exit /b 0

:: ─────────────────────────────────────────────
:BASLA
:: ─────────────────────────────────────────────
echo.
echo  ============================================================
echo   BITİRMEEG  ^|  Sistem baslatiliyor
echo  ============================================================
echo.

:: Eski servisleri durdur
echo  [1/4] Eski servisler durduruluyor...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":3000 "') do (
    taskkill /PID %%a /F >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5001 "') do (
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 3 /nobreak >nul

:: Build (sadece ilk açılışta veya .next silinmişse)
if not exist .next (
    echo  [2/4] Uygulama derleniyor, lutfen bekleyin...
    call npm run build
    if errorlevel 1 (
        echo.
        echo  [HATA] Derleme basarisiz. npm start penceresi kontrol edilsin.
        pause
        exit /b 1
    )
) else (
    echo  [2/4] Derleme atliyor ^(zaten mevcut^)...
)

:: Trigger server
echo  [3/4] Trigger server baslatiliyor ^(port 5001^)...
start /min "BİTİRMEEG — Trigger" cmd /k python trigger_server.py

:: Next.js
echo  [4/4] Web sunucusu baslatiliyor ^(port 3000^)...
start /min "BİTİRMEEG — Next.js" cmd /k npm start

echo.
echo  Her iki servis hazir olunca tarayici acilacak...
echo  ^(Bu pencere kapanabilir^)
echo.
python wait_and_open.py
exit /b 0
