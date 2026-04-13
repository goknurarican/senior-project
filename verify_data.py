#!/usr/bin/env python3
"""
BİTİRMEEG — Veri Doğrulama Aracı
===================================
Son deneğin göz verisi, marker ve platform verilerini kontrol eder.

Kullanım:
    python verify_data.py                  ← son deneği otomatik seç
    python verify_data.py --user-id 6      ← belirli bir deneği seç
    python verify_data.py --all            ← tüm denekleri listele
"""

import argparse
import csv
import json
import os
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH    = SCRIPT_DIR / "experiment.db"
SUBJ_DIR   = SCRIPT_DIR / "subjects"

SEP  = "=" * 60
SEP2 = "-" * 60

# ── Renk kodları (Windows terminalde de çalışır) ──────────────────────────────
try:
    import colorama; colorama.init()
    OK   = "\033[92m[OK]  \033[0m"
    WARN = "\033[93m[WARN]\033[0m"
    ERR  = "\033[91m[ERR] \033[0m"
    HDR  = "\033[96m"
    RST  = "\033[0m"
except ImportError:
    OK = "[OK]  "; WARN = "[WARN]"; ERR = "[ERR] "; HDR = ""; RST = ""


def db():
    if not DB_PATH.exists():
        print(f"{ERR} experiment.db bulunamadı: {DB_PATH}")
        sys.exit(1)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_ts(ms):
    if not ms:
        return "?"
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%H:%M:%S")


def list_all_users():
    c = db().cursor()
    c.execute("""
        SELECT u.id, u.name, u.email,
               s.experiment_group AS grp,
               s.id AS session_id,
               s.phase,
               s.created_at
        FROM users u
        LEFT JOIN sessions s ON s.user_id = u.id
        WHERE u.role != 'admin'
        ORDER BY u.id DESC
    """)
    rows = c.fetchall()
    if not rows:
        print("Veritabanında denek yok.")
        return
    print(f"\n{HDR}{SEP}\n  TÜM DENEKLER\n{SEP}{RST}")
    for r in rows:
        packaged = (SUBJ_DIR / f"user_{r['id']:03d}_{(r['name'] or '').replace(' ','_').lower()}_{r['grp']}").exists()
        tag = "✓ paketlendi" if packaged else "○ bekleniyor"
        print(f"  #{r['id']}  {r['name'] or '?':<20} | {r['grp'] or '?':<12} | phase: {r['phase'] or '?':<10} | {tag}")


def get_user(user_id=None):
    c = db().cursor()
    if user_id:
        c.execute("SELECT * FROM users WHERE id=? AND role!='admin'", (user_id,))
    else:
        c.execute("""
            SELECT u.* FROM users u
            JOIN sessions s ON s.user_id = u.id
            WHERE u.role != 'admin'
            ORDER BY s.created_at DESC LIMIT 1
        """)
    row = c.fetchone()
    if not row:
        print(f"{ERR} Kullanıcı bulunamadı (id={user_id})")
        sys.exit(1)
    return dict(row)


def get_session(user_id):
    c = db().cursor()
    c.execute("""
        SELECT * FROM sessions WHERE user_id=?
        ORDER BY created_at DESC LIMIT 1
    """, (user_id,))
    row = c.fetchone()
    return dict(row) if row else {}


# ── Bölüm 1: Platform verisi ──────────────────────────────────────────────────

def check_platform(user_id, session_id):
    print(f"\n{HDR}{SEP}\n  1. PLATFORM VERİSİ\n{SEP}{RST}")
    c = db().cursor()

    c.execute("SELECT COUNT(*) FROM events WHERE user_id=? OR session_id=?", (user_id, session_id))
    total = c.fetchone()[0]
    print(f"  Toplam event     : {total}")

    for etype in ["page_view", "mouse_click", "mouse_trajectory", "SCENARIO_TRIGGERED", "EXPERIMENT_FINISHED"]:
        c.execute("SELECT COUNT(*) FROM events WHERE (user_id=? OR session_id=?) AND event_type=?",
                  (user_id, session_id, etype))
        n = c.fetchone()[0]
        tag = OK if n > 0 else WARN
        print(f"  {tag} {etype:<26}: {n}")

    if total > 0:
        c.execute("""
            SELECT MIN(timestamp), MAX(timestamp) FROM events
            WHERE (user_id=? OR session_id=?) AND timestamp IS NOT NULL
        """, (user_id, session_id))
        mn, mx = c.fetchone()
        if mn and mx:
            dur = (mx - mn) / 1000
            print(f"  Oturum süresi    : {dur:.0f} sn  ({_fmt_ts(mn)} → {_fmt_ts(mx)} UTC)")


# ── Bölüm 2: Marker (senaryo) verisi ─────────────────────────────────────────

def check_markers(session_id):
    print(f"\n{HDR}{SEP}\n  2. MARKER / SENARYO VERİSİ\n{SEP}{RST}")
    c = db().cursor()

    # lsl_events (trigger_server tarafından kaydedilen)
    c.execute("""
        SELECT scenario_name, scenario_type, eeg_marker, wall_time_ms, phase
        FROM lsl_events WHERE session_id=?
        ORDER BY wall_time_ms ASC
    """, (session_id,))
    lsl_rows = c.fetchall()

    if not lsl_rows:
        print(f"  {WARN} lsl_events tablosunda bu session için kayıt yok.")
        print(f"       → trigger_server.py çalışıyor muydu? Senaryo tetiklendi mi?")
    else:
        print(f"  {OK} lsl_events: {len(lsl_rows)} marker kaydı bulundu\n")
        print(f"  {'Zaman (UTC)':<12} {'Senaryo':<30} {'EEG Marker':<12} {'Phase'}")
        print(f"  {SEP2}")
        for r in lsl_rows:
            t = _fmt_ts(r['wall_time_ms'])
            print(f"  {t:<12} {r['scenario_name']:<30} S {r['eeg_marker']:<10} {r['phase'] or '?'}")

    # events tablosundaki SCENARIO_TRIGGERED kayıtları
    c.execute("""
        SELECT event_type, event_data, timestamp FROM events
        WHERE (session_id=?) AND event_type='SCENARIO_TRIGGERED'
        ORDER BY timestamp ASC
    """, (session_id,))
    ev_rows = c.fetchall()
    print(f"\n  SCENARIO_TRIGGERED events (platform tarafı): {len(ev_rows)}")
    for r in ev_rows:
        try:
            d = json.loads(r['event_data'] or '{}')
            print(f"    {_fmt_ts(r['timestamp'])}  {d.get('scenario_name','?')} (type={d.get('scenario_type','?')})")
        except Exception:
            print(f"    {r['event_data']}")

    return len(lsl_rows)


# ── Bölüm 3: Göz verisi ───────────────────────────────────────────────────────

def check_eye(session_id):
    print(f"\n{HDR}{SEP}\n  3. GÖZ VERİSİ (eye_data)\n{SEP}{RST}")
    c = db().cursor()

    c.execute("SELECT COUNT(*) FROM eye_data WHERE session_id=?", (session_id,))
    total = c.fetchone()[0]

    if total == 0:
        print(f"  {ERR} eye_data tablosunda bu session için hiç satır yok.")
        print(f"       → trigger_server.py çalışıyor muydu? Gazepoint bağlı mıydı?")
        return

    print(f"  {OK} Toplam göz örneği : {total}")

    c.execute("""
        SELECT MIN(wall_time_ms), MAX(wall_time_ms),
               AVG(gaze_x), AVG(gaze_y),
               AVG(pupil_left), AVG(pupil_right),
               SUM(CASE WHEN gaze_x != 0 OR gaze_y != 0 THEN 1 ELSE 0 END) as nonzero
        FROM eye_data WHERE session_id=?
    """, (session_id,))
    r = c.fetchone()

    dur = ((r[1] or 0) - (r[0] or 0)) / 1000
    nonzero = r[6] or 0
    pct = 100 * nonzero / total if total else 0

    print(f"  Kayıt süresi       : {dur:.0f} sn  ({_fmt_ts(r[0])} → {_fmt_ts(r[1])} UTC)")
    print(f"  Sıfır olmayan örnek: {nonzero} / {total}  (%{pct:.0f})")

    if pct < 5:
        print(f"  {WARN} Gaze_x/y büyük çoğunluğu sıfır.")
        print(f"       → Kameraya kimse bakmıyorsa bu normaldir.")
        print(f"       → Gerçek deneğe geçmeden önce biri bakarak test edin.")
    else:
        print(f"  {OK} Geçerli göz verisi mevcut.")

    print(f"  Ort. gaze_x        : {r[2]:.4f}")
    print(f"  Ort. gaze_y        : {r[3]:.4f}")
    print(f"  Ort. pupil_left    : {r[4]:.4f}")
    print(f"  Ort. pupil_right   : {r[5]:.4f}")

    # Son 5 örnek
    c.execute("""
        SELECT wall_time_ms, gaze_x, gaze_y, pupil_left, pupil_right
        FROM eye_data WHERE session_id=?
        ORDER BY wall_time_ms DESC LIMIT 5
    """, (session_id,))
    print(f"\n  Son 5 örnek:")
    print(f"  {'Zaman':>10}  {'gaze_x':>8}  {'gaze_y':>8}  {'pupil_L':>8}  {'pupil_R':>8}")
    for row in reversed(c.fetchall()):
        print(f"  {_fmt_ts(row[0]):>10}  {row[1]:>8.4f}  {row[2]:>8.4f}  {row[3]:>8.4f}  {row[4]:>8.4f}")


# ── Bölüm 4: Gazepoint native CSV ────────────────────────────────────────────

def check_gazepoint_csv():
    print(f"\n{HDR}{SEP}\n  4. GAZEPOINT NATIVE CSV\n{SEP}{RST}")

    search_dirs = [
        Path.home() / "Documents" / "Gazepoint",
        Path.home() / "Documents",
        Path("C:/Users/Public/Documents/Gazepoint"),
        Path("C:/ProgramData/Gazepoint"),
    ]

    found = []
    for d in search_dirs:
        if not d.exists():
            continue
        for f in d.glob("*.csv"):
            found.append(f)

    if not found:
        print(f"  {WARN} Hiçbir yerde Gazepoint CSV bulunamadı.")
        print(f"  Aranan klasörler:")
        for d in search_dirs:
            print(f"    {d}  {'[mevcut]' if d.exists() else '[yok]'}")
        print(f"\n  Gazepoint Control > File > Data Directory ile kayıt klasörünü bul.")
        print(f"  Bulunca: python package_subject.py --user-id X --gaze-file \"C:/...csv\"")
        return None

    latest = max(found, key=lambda f: f.stat().st_mtime)
    size_kb = latest.stat().st_size // 1024

    print(f"  {OK} Bulunan dosya : {latest}")
    print(f"  Boyut          : {size_kb} KB")

    # İlk satırı oku
    try:
        with open(latest, encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip()
            second = f.readline().strip()
        print(f"  Başlık sütunları: {header[:120]}")
        print(f"  İlk veri satırı : {second[:120]}")

        cols = [c.strip() for c in header.split(',')]
        has_fpog = any('FPOGX' in c for c in cols)
        has_user = any('USER_DATA' in c for c in cols)
        print(f"\n  FPOGX sütunu    : {OK if has_fpog else WARN + ' YOK'}")
        print(f"  USER_DATA sütunu: {OK if has_user else WARN + ' YOK (senaryolar burada görünür)'}")
    except Exception as e:
        print(f"  {WARN} CSV okunamadı: {e}")

    return latest


# ── Bölüm 5: Paketlenmiş dosyalar ────────────────────────────────────────────

def check_packaged_files(user_id, user_name, grp):
    print(f"\n{HDR}{SEP}\n  5. PAKETLENMİŞ DOSYALAR\n{SEP}{RST}")

    name_slug = (user_name or "").replace(" ", "_").lower()
    folder = SUBJ_DIR / f"user_{user_id:03d}_{name_slug}_{grp}"

    if not folder.exists():
        print(f"  {WARN} Klasör henüz oluşturulmamış: {folder.name}")
        print(f"       SONRAKI_DENEK.bat ile paketi oluştur.")
        return

    print(f"  Klasör: {folder.name}")

    checks = {
        "platform/all_events.csv":      "Platform eventleri",
        "platform/mouse_clicks.csv":    "Mouse tıklamaları",
        "platform/scenario_triggers.csv": "Senaryo tetikleyiciler",
        "platform/session_info.json":   "Session bilgisi",
        "eye/eye_data_db.csv":          "Göz verisi (DB)",
        "eeg/marker_legend.csv":        "EEG marker efsanesi",
    }

    for rel, label in checks.items():
        p = folder / rel
        if p.exists():
            size = p.stat().st_size
            tag = OK if size > 50 else WARN + " (boş?)"
            print(f"  {tag} {label:<30}: {size:>8} bytes  {rel}")
        else:
            print(f"  {ERR} {label:<30}: EKSİK  {rel}")

    # SCENARIO_DATASET
    ds_files = list((folder / "platform").glob("SCENARIO_DATASET_*.csv"))
    if ds_files:
        ds = ds_files[0]
        with open(ds, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        print(f"\n  {OK} SCENARIO_DATASET: {len(rows)-1} satır  →  {ds.name}")
        if len(rows) > 1:
            print(f"       Sütunlar: {', '.join(rows[0])}")
            # marker satırlarını bul
            marker_rows = [r for r in rows[1:] if r and r[1] == 'scenario']
            print(f"       Marker satırları ({len(marker_rows)}):")
            for r in marker_rows:
                print(f"         {r[0]:<30}  eeg_marker={r[-1]}  t={r[3]}")
    else:
        print(f"  {WARN} SCENARIO_DATASET CSV bulunamadı.")

    # EEG
    eeg_dir = folder / "eeg"
    eeg_files = list(eeg_dir.glob("*.vhdr")) + list(eeg_dir.glob("*.eeg"))
    if eeg_files:
        print(f"\n  {OK} EEG dosyaları mevcut: {[f.name for f in eeg_files]}")
    else:
        print(f"\n  {WARN} EEG dosyaları henüz kopyalanmadı (BrainVision bağlandığında eklenecek).")

    # Gazepoint native
    eye_csvs = [f for f in (folder / "eye").glob("*.csv") if "eye_data_db" not in f.name]
    if eye_csvs:
        print(f"  {OK} Gazepoint native CSV: {eye_csvs[0].name}")
    else:
        print(f"  {WARN} Gazepoint native CSV yok (manuel kopyalanması gerekiyor).")


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def verify(user_id=None):
    user    = get_user(user_id)
    uid     = user['id']
    name    = user.get('name') or f"User {uid}"
    session = get_session(uid)
    sid     = session.get('id', '')
    grp     = session.get('experiment_group', '?')
    phase   = session.get('phase', '?')

    print(f"\n{HDR}{SEP}")
    print(f"  DENEK #{uid}  —  {name}")
    print(f"  Grup: {grp}  |  Phase: {phase}")
    print(f"  Session ID: {sid}")
    print(f"{SEP}{RST}")

    check_platform(uid, sid)
    n_markers = check_markers(sid)
    check_eye(sid)
    check_gazepoint_csv()
    check_packaged_files(uid, name, grp)

    # Özet
    print(f"\n{HDR}{SEP}\n  ÖZET\n{SEP}{RST}")
    c = db().cursor()
    c.execute("SELECT COUNT(*) FROM eye_data WHERE session_id=?", (sid,))
    eye_n = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM events WHERE (user_id=? OR session_id=?) AND event_type='SCENARIO_TRIGGERED'",
              (uid, sid))
    scen_n = c.fetchone()[0]

    items = [
        ("Platform eventleri",  True,       "var"),
        ("Göz verisi (DB)",     eye_n > 0,  f"{eye_n} satır"),
        ("Scenario markerlari", n_markers > 0, f"{n_markers} trigger"),
        ("EEG bağlantısı",      False,      "deneyden sonra eklenecek"),
        ("Gazepoint native CSV",False,      "manuel kopyalanacak"),
    ]
    for label, ok, note in items:
        print(f"  {OK if ok else WARN} {label:<28}: {note}")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BİTİRMEEG veri doğrulama")
    p.add_argument("--user-id", type=int, help="Belirli bir deneği kontrol et")
    p.add_argument("--all",     action="store_true", help="Tüm denekleri listele")
    a = p.parse_args()

    if a.all:
        list_all_users()
    elif a.user_id:
        verify(a.user_id)
    else:
        verify()  # son deneği otomatik seç
