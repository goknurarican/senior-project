@echo off
chcp 65001 >nul
title BİTİRMEEG — Lab Panel

cd /d "%~dp0"

:: Python kontrolü
where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [HATA] Python bulunamadi.
    echo  Lutfen bir arastirmaciya bildirin.
    echo.
    pause
    exit /b 1
)

:: Eski bytecode önbelleğini temizle
if exist __pycache__ rmdir /s /q __pycache__ >nul 2>&1

:: Sessions tablosunu otomatik onar (eski NULL user_id kayıtları düzelt)
python -c "
import sqlite3, pathlib
db = pathlib.Path('experiment.db')
if db.exists():
    conn = sqlite3.connect(str(db))
    n = conn.execute('''
        UPDATE sessions SET user_id = (
            SELECT e.user_id FROM events e
            WHERE e.session_id = sessions.id AND e.user_id IS NOT NULL LIMIT 1
        )
        WHERE user_id IS NULL AND EXISTS (
            SELECT 1 FROM events e
            WHERE e.session_id = sessions.id AND e.user_id IS NOT NULL
        )
    ''').rowcount
    conn.commit()
    conn.close()
    if n: print(f'  [OK] {n} oturum user_id onarıldı.')
" 2>nul

:: Lab panelini aç
python lab_panel.py

:: Anormal çıkış
if errorlevel 1 (
    echo.
    echo  [HATA] Panel beklenmedik sekilde kapandi.
    echo  Lutfen bir arastirmaciya bildirin.
    echo.
    pause
)
