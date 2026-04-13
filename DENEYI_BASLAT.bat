@echo off
chcp 65001 >nul
title BİTİRMEEG — Deney Başlatma

cd /d "%~dp0"

echo.
echo  ============================================================
echo   BİTİRMEEG — Günlük Deney Başlatma
echo  ============================================================
echo.

:: Python kontrolü
where python >nul 2>&1
if errorlevel 1 (
    echo  [HATA] Python bulunamadi. Arastirmaciya bildirin.
    pause & exit /b 1
)

:: experiment.db var mi?
if not exist experiment.db (
    echo  [UYARI] experiment.db yok — ilk deneği kayıt ettirince oluşacak.
) else (
    echo  [OK]  experiment.db mevcut
)

:: trigger_server zaten çalışıyor mu?
netstat -ano | findstr ":5001" >nul 2>&1
if not errorlevel 1 (
    echo  [OK]  trigger_server.py zaten port 5001'de çalışıyor
    goto :browser
)

:: trigger_server.py'yi yeni pencerede başlat
echo  [..] trigger_server.py başlatılıyor...
start "EEG + GazePoint Trigger Server" cmd /k "cd /d "%~dp0" && python trigger_server.py"
echo  [OK]  Trigger server başlatıldı ^(ayrı pencerede^)

:: Sunucunun hazır olması için bekle
timeout /t 3 /nobreak >nul

:browser
:: Platforma erişim
echo.
echo  ============================================================
echo   Kontrol listesi:
echo  ============================================================
echo   [ ] BrainVision Recorder acik ve KAYIT BAŞLADI mi?
echo   [ ] Gazepoint Control acik mi?
echo   [ ] Trigger server penceresi "Listening on 5001" diyor mu?
echo.
echo   Hazirsa ENTER'a bas — tarayici acilacak.
echo   (Hazir degilse bu pencereyi kapat, kontrol et, tekrar ac)
echo.
pause

:: Tarayıcıda platformu aç
start http://localhost:3000

echo.
echo  Tarayici acildi. Denegi kayit sayfasina yonlendir.
echo.
echo  DENEK BİTİNCE:
echo    1. BrainVision ^> Stop Recording
echo    2. SONRAKI_DENEK.bat'i calistir
echo.
