@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: Port 3000'de eski process varsa öldür
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000 "') do (
    taskkill /PID %%a /F >nul 2>&1
)

:: Port 5001'de eski trigger server varsa öldür
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5001 "') do (
    taskkill /PID %%a /F >nul 2>&1
)

if not exist .next (
    echo Build yapiliyor, lutfen bekleyin...
    call npm run build
    if errorlevel 1 ( echo Build basarisiz. & pause & exit /b 1 )
)

start /min "Trigger Server" cmd /k "cd /d "%~dp0" && python trigger_server.py"
start /min "Next.js" cmd /k "cd /d "%~dp0" && npm start"

python wait_and_open.py
