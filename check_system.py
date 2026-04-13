#!/usr/bin/env python3
"""
BİTİRMEEG — Sistem Kontrol Scripti
Çalıştır: python check_system.py
Her şeyin çalışıp çalışmadığını gösterir.
"""
import sys, socket, sqlite3, json, subprocess, importlib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH    = SCRIPT_DIR / "experiment.db"
LOG_PATH   = SCRIPT_DIR / "packaged_log.json"
TOKEN_FILE = SCRIPT_DIR / "token.json"
CONFIG_FILE= SCRIPT_DIR / "lab_config.json"

OK   = "[OK]  "
WARN = "[WARN]"
ERR  = "[HATA]"
SEP  = "=" * 56

results = []

def check(label, status, detail="", fix=""):
    mark = OK if status == "ok" else (WARN if status == "warn" else ERR)
    line = f"  {mark}  {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    if fix and status != "ok":
        print(f"         -> {fix}")
    results.append((status, label))

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

# ─────────────────────────────────────────────────────────────────
section("1. PYTHON VE PAKETLER")
# ─────────────────────────────────────────────────────────────────
check("Python sürümü", "ok",
      f"{sys.version.split()[0]} ({sys.executable})")

packages = {
    "flask":           "pip install flask",
    "flask_cors":      "pip install flask-cors",
    "pylsl":           "pip install pylsl",
    "sqlite3":         "(yerleşik)",
    "tkinter":         "(yerleşik)",
}
for pkg, install_cmd in packages.items():
    try:
        importlib.import_module(pkg)
        check(pkg, "ok")
    except ImportError:
        check(pkg, "err", "kurulu değil", install_cmd)

drive_packages = ["google.oauth2", "googleapiclient", "google_auth_oauthlib"]
drive_ok = True
for pkg in drive_packages:
    try:
        importlib.import_module(pkg.split(".")[0])
        check(pkg, "ok")
    except ImportError:
        check(pkg, "warn", "kurulu değil",
              "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        drive_ok = False

# ─────────────────────────────────────────────────────────────────
section("2. DOSYALAR")
# ─────────────────────────────────────────────────────────────────
files = {
    "trigger_server.py": ("err", "Eksik"),
    "lab_panel.py":      ("err", "Eksik"),
    "package_subject.py":("err", "Eksik"),
    "data_logger.py":    ("err", "Eksik"),
    "db_repair.py":      ("err", "Eksik"),
    "lab_config.json":   ("warn","Eksik — Drive backup çalışmaz"),
}
for fname, (sev, msg) in files.items():
    p = SCRIPT_DIR / fname
    check(fname, "ok" if p.exists() else sev,
          "" if p.exists() else msg)

creds = SCRIPT_DIR / "credentials.json"
if not creds.exists():
    cands = list(SCRIPT_DIR.glob("client_secret_*.json"))
    if cands:
        check("credentials (client_secret_*.json)", "ok", cands[0].name)
        creds = cands[0]
    else:
        check("credentials.json", "warn",
              "Yok — Drive backup çalışmaz",
              "Mac'ten USB ile kopyala: credentials.json")
else:
    check("credentials.json", "ok")

check("token.json (Drive auth)", "ok" if TOKEN_FILE.exists() else "warn",
      "" if TOKEN_FILE.exists() else "Yok — DRIVE_KURULUM.bat çalıştır")

check("experiment.db", "ok" if DB_PATH.exists() else "warn",
      str(DB_PATH) if DB_PATH.exists() else "Yok — npm start + denek kaydı gerekli")

# ─────────────────────────────────────────────────────────────────
section("3. VERİTABANI")
# ─────────────────────────────────────────────────────────────────
if DB_PATH.exists():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    try:
        n = conn.execute("SELECT COUNT(*) FROM users WHERE role!='admin'").fetchone()[0]
        check("Denek sayısı (users)", "ok" if n > 0 else "warn",
              f"{n} denek", "Bir deneğin siteden kayıt olması gerekiyor")
    except Exception as e:
        check("users tablosu", "err", str(e))

    try:
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        null  = conn.execute("SELECT COUNT(*) FROM sessions WHERE user_id IS NULL").fetchone()[0]
        if null > 0:
            check("sessions.user_id", "warn",
                  f"{null}/{total} NULL — db_repair.py çalıştırılıyor...",)
            n2 = conn.execute("""
                UPDATE sessions SET user_id = (
                    SELECT e.user_id FROM events e
                    WHERE e.session_id = sessions.id
                      AND e.user_id IS NOT NULL LIMIT 1
                )
                WHERE user_id IS NULL AND EXISTS (
                    SELECT 1 FROM events e
                    WHERE e.session_id = sessions.id
                      AND e.user_id IS NOT NULL
                )
            """).rowcount
            conn.commit()
            if n2:
                print(f"         -> {n2} oturum onarildi")
                check("sessions.user_id (onarım sonrası)", "ok",
                      f"{n2} satır dolduruldu")
        else:
            check("sessions.user_id", "ok", f"{total} oturumun tamamı dolu")
    except Exception as e:
        check("sessions tablosu", "err", str(e))

    try:
        ev = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        ev_uid = conn.execute(
            "SELECT COUNT(*) FROM events WHERE user_id IS NOT NULL").fetchone()[0]
        check("events tablosu", "ok", f"{ev} event, {ev_uid} user_id'li")
    except Exception as e:
        check("events tablosu", "err", str(e))

    try:
        eye = conn.execute("SELECT COUNT(*) FROM eye_data").fetchone()[0]
        check("eye_data (Gazepoint)", "ok" if eye > 0 else "warn",
              f"{eye} göz verisi satırı",
              "trigger_server.py çalışmıyordu veya Gazepoint bağlı değildi")
    except Exception as e:
        check("eye_data tablosu", "warn", str(e))

    conn.close()
else:
    print("  (Veritabanı yok — atlandı)")

# ─────────────────────────────────────────────────────────────────
section("4. ÇALIŞAN SERVİSLER")
# ─────────────────────────────────────────────────────────────────

def port_open(host, port):
    try:
        s = socket.create_connection((host, port), timeout=1)
        s.close()
        return True
    except Exception:
        return False

check("Next.js (port 3000)",
      "ok" if port_open("127.0.0.1", 3000) else "warn",
      "çalışıyor" if port_open("127.0.0.1", 3000) else "çalışmıyor",
      "npm start")

check("trigger_server.py (port 5001)",
      "ok" if port_open("127.0.0.1", 5001) else "warn",
      "çalışıyor" if port_open("127.0.0.1", 5001) else "çalışmıyor",
      "python trigger_server.py")

check("Gazepoint API (port 4242)",
      "ok" if port_open("127.0.0.1", 4242) else "warn",
      "çalışıyor" if port_open("127.0.0.1", 4242) else "çalışmıyor / kapalı",
      "Gazepoint Control'ü aç")

# ─────────────────────────────────────────────────────────────────
section("5. GOOGLE DRIVE")
# ─────────────────────────────────────────────────────────────────
if not drive_ok:
    check("Drive bağlantısı", "warn", "Paketler eksik — önce kur")
elif not TOKEN_FILE.exists():
    check("Drive bağlantısı", "warn", "token.json yok", "DRIVE_KURULUM.bat çalıştır")
elif not creds.exists():
    check("Drive bağlantısı", "warn", "credentials.json yok")
else:
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        creds_obj = Credentials.from_authorized_user_file(
            str(TOKEN_FILE),
            ["https://www.googleapis.com/auth/drive"]
        )
        if creds_obj.valid:
            check("Drive token", "ok", "geçerli")
        elif creds_obj.expired and creds_obj.refresh_token:
            creds_obj.refresh(Request())
            TOKEN_FILE.write_text(creds_obj.to_json(), encoding="utf-8")
            check("Drive token", "ok", "yenilendi")
        else:
            check("Drive token", "warn", "süresi dolmuş",
                  "DRIVE_KURULUM.bat tekrar çalıştır")
    except Exception as e:
        check("Drive token", "warn", str(e))

config = {}
if CONFIG_FILE.exists():
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
bcfg = config.get("backup", {})
check("lab_config.json backup enabled",
      "ok" if bcfg.get("enabled") else "warn",
      str(bcfg.get("enabled", "?")))
check("lab_config.json drive_folder_id",
      "ok" if bcfg.get("drive_folder_id") else "warn",
      bcfg.get("drive_folder_id", "YOK"))

# ─────────────────────────────────────────────────────────────────
section("ÖZET")
# ─────────────────────────────────────────────────────────────────
errors   = [r for r in results if r[0] == "err"]
warnings = [r for r in results if r[0] == "warn"]
oks      = [r for r in results if r[0] == "ok"]

print(f"\n  OK: {len(oks)}   UYARI: {len(warnings)}   HATA: {len(errors)}")

if errors:
    print("\n  Kritik hatalar:")
    for _, label in errors:
        print(f"    - {label}")

if warnings:
    print("\n  Uyarılar (deneyi engellemez ama kontrol et):")
    for _, label in warnings:
        print(f"    - {label}")

if not errors and not warnings:
    print("\n  Her sey hazir! Deneye baslanabilir.")
elif not errors:
    print("\n  Deney yapilabilir. Uyarilari gözden gecir.")
else:
    print("\n  Kritik hatalar giderilmeden deney yapma.")
