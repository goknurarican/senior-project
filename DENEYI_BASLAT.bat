@echo off
chcp 65001 >nul
title Deney Baslat
cd /d "%~dp0"

echo Trigger server baslatiliyor...
start "Trigger Server" cmd /k python trigger_server.py

echo Next.js baslatiliyor...
start "Next.js" cmd /k npm start

echo.
echo Her iki pencere de acildi.
echo Next.js penceresinde "Ready" yazinca ENTER'a bas.
echo.
pause

start http://localhost:3000
