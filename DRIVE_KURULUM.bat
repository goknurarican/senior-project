@echo off
chcp 65001 >nul
title BİTİRMEEG — Google Drive Kurulum

cd /d "%~dp0"

echo.
echo  ============================================================
echo   BİTİRMEEG — Google Drive Kurulum (tek seferlik)
echo  ============================================================
echo.
echo  Bu pencereyi kapatma, Google hesabina giris icin
echo  tarayici acilacak.
echo.

:: credentials.json var mi kontrol et
if not exist credentials.json (
    for %%f in (client_secret_*.json) do (
        echo  [OK] Credentials dosyasi bulundu: %%f
        goto :run
    )
    echo  [HATA] credentials.json bulunamadi!
    echo.
    echo  Yapman gereken:
    echo    1. Arastirmacidan credentials.json dosyasini al
    echo    2. Su klasore kopyala:
    echo       %~dp0
    echo    3. Bu BAT dosyasini tekrar calistir
    echo.
    pause
    exit /b 1
)

:run
python setup_drive.py

if errorlevel 1 (
    echo.
    echo  [HATA] Kurulum basarisiz. Yukaridaki mesaji incele.
    echo.
    pause
    exit /b 1
)

echo.
echo  Kurulum tamamlandi! Bu pencereyi kapatabilirsin.
pause
