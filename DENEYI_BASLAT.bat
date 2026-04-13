@echo off
chcp 65001 >nul
title Deney Baslat
cd /d "%~dp0"

:: Build yoksa once build yap
if not exist .next (
    echo [..] .next klasoru yok, build yapiliyor - birkac dakika surabilir...
    call npm run build
    if errorlevel 1 (
        echo [HATA] Build basarisiz. Tekrar dene.
        pause
        exit /b 1
    )
)

:: Trigger server ve Next.js'i ayri pencerelerde ac
start "Trigger Server" cmd /k "cd /d "%~dp0" && python trigger_server.py"
start "Next.js" cmd /k "cd /d "%~dp0" && npm start"

echo.
echo Trigger Server ve Next.js baslatildi.
echo Next.js penceresinde "Ready on http://localhost:3000" yazinca
echo asagidaki ENTER'a bas.
echo.
pause

start http://localhost:3000
