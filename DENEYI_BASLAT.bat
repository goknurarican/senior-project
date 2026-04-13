@echo off
chcp 65001 >nul
title BİTİRMEEG — Deney Başlatma

cd /d "%~dp0"

echo.
echo  ============================================================
echo   BİTİRMEEG — Gunluk Deney Baslatma
echo  ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo  [HATA] Python bulunamadi.
    pause & exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
    echo  [HATA] npm bulunamadi. Node.js yuklu mu?
    pause & exit /b 1
)

:: .next klasörü yoksa build gerekli
if not exist .next (
    echo  [..] .next klasoru yok, npm run build calistiriliyor...
    npm run build
    if errorlevel 1 (
        echo  [HATA] Build basarisiz.
        pause & exit /b 1
    )
)

:: trigger_server zaten çalışıyor mu?
netstat -ano 2>nul | findstr ":5001 " >nul 2>&1
if not errorlevel 1 (
    echo  [OK]  trigger_server.py zaten calisiyor
) else (
    echo  [..] trigger_server.py baslatiliyor...
    start "Trigger Server" cmd /k "cd /d "%~dp0" && python trigger_server.py"
)

:: Next.js zaten çalışıyor mu?
netstat -ano 2>nul | findstr ":3000 " >nul 2>&1
if not errorlevel 1 (
    echo  [OK]  Next.js zaten calisiyor
) else (
    echo  [..] Next.js baslatiliyor...
    start "Next.js Platform" cmd /k "cd /d "%~dp0" && npm start"
    echo  [..] Next.js baslamasi bekleniyor (15 saniye)...
    timeout /t 15 /nobreak >nul
)

echo.
echo  ============================================================
echo   Kontrol listesi:
echo  ============================================================
echo   [ ] BrainVision Recorder acik ve KAYIT BASLADI mi?
echo   [ ] Gazepoint Control acik mi?
echo   [ ] Trigger Server penceresi hata vermiyor mu?
echo   [ ] Next.js penceresi "Ready" diyor mu?
echo.
echo   Hazirsa ENTER'a bas - tarayici acilacak.
echo.
pause

start http://localhost:3000

echo.
echo  Tarayici acildi. Denegi kayit sayfasina yonlendir.
echo.
echo  DENEK BITINCE:
echo    1. BrainVision ^> Stop Recording
echo    2. SONRAKI_DENEK.bat'i calistir
echo.
pause
