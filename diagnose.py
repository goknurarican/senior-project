#!/usr/bin/env python3
"""
BİTİRMEEG — Tanı Scripti
Çalıştır: python diagnose.py
"""
import sqlite3
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH    = SCRIPT_DIR / "experiment.db"
LOG_PATH   = SCRIPT_DIR / "packaged_log.json"

SEP  = "=" * 60
SEP2 = "-" * 60

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def check(label, ok, detail=""):
    mark = "OK" if ok else "HATA"
    print(f"  [{mark}]  {label}" + (f" — {detail}" if detail else ""))

# ── 1. Dosya sistemi ───────────────────────────────────────────────────────────
section("1. DOSYA SİSTEMİ")
check("experiment.db mevcut", DB_PATH.exists(), str(DB_PATH))

if not DB_PATH.exists():
    print(f"""
  experiment.db bulunamadı.
  Beklenen konum: {DB_PATH}

  Olası nedenler:
    - npm start hiç çalıştırılmadı
    - npm start farklı bir klasörden çalıştırıldı
    - Henüz hiçbir denek kayıt olmadı (DB ilk signup'ta oluşur)

  Çözüm:
    1. Proje klasöründen:  npm start
    2. Bir denek siteden kayıt olsun
    3. Bu scripti tekrar çalıştır
""")
    raise SystemExit(1)

check("packaged_log.json mevcut", LOG_PATH.exists(),
      str(LOG_PATH) if LOG_PATH.exists() else "henüz oluşmamış (normal)")

# ── 2. Veritabanı tabloları ───────────────────────────────────────────────────
section("2. VERİTABANI TABLOLARI")
conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row

tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
)]
print(f"  Tablolar: {tables}")

for t in ["users", "sessions", "events"]:
    try:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        check(f"{t} tablosu", True, f"{n} satır")
    except Exception as e:
        check(f"{t} tablosu", False, str(e))

# ── 3. Kullanıcılar ───────────────────────────────────────────────────────────
section("3. KULLANICILAR")
users = list(conn.execute("SELECT id, name, email, role, created_at FROM users ORDER BY id"))
if not users:
    print("  Hiç kullanıcı yok!")
else:
    for u in users:
        tag = " ← DENEK" if u["role"] != "admin" else " (admin)"
        print(f"  id={u['id']:3d}  role={u['role']:<6}  "
              f"name={u['name'] or '—':<20}  email={u['email']}{tag}")

non_admin = [u for u in users if u["role"] != "admin"]
check("En az 1 denek (non-admin) var", len(non_admin) > 0,
      f"{len(non_admin)} denek bulundu")

# ── 4. Oturumlar ──────────────────────────────────────────────────────────────
section("4. OTURUMLAR (sessions)")
sessions = list(conn.execute(
    "SELECT id, user_id, experiment_group, phase, assigned_variant, created_at "
    "FROM sessions ORDER BY created_at DESC LIMIT 10"
))
if not sessions:
    print("  Hiç oturum yok!")
else:
    null_uid = 0
    for s in sessions:
        uid_str = str(s["user_id"]) if s["user_id"] is not None else "NULL ⚠"
        if s["user_id"] is None:
            null_uid += 1
        print(f"  session={s['id'][:14]}...  user_id={uid_str:<6}  "
              f"group={s['experiment_group']:<10}  phase={s['phase']}")
    if null_uid > 0:
        print(f"\n  ⚠ {null_uid} oturumda user_id=NULL — eski kod ile oluşturulmuş.")
        print("    Çözüm: git pull → npm run build → npm start → yeni denek al")

sessions_with_uid = [s for s in sessions if s["user_id"] is not None]
check("En az 1 oturumda user_id dolu", len(sessions_with_uid) > 0,
      f"{len(sessions_with_uid)}/{len(sessions)} oturumda user_id var")

# ── 5. Eventler ───────────────────────────────────────────────────────────────
section("5. EVENTLER (events)")
try:
    ev_total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    ev_with_uid = conn.execute(
        "SELECT COUNT(*) FROM events WHERE user_id IS NOT NULL"
    ).fetchone()[0]
    ev_scenarios = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type='SCENARIO_TRIGGERED'"
    ).fetchone()[0]
    print(f"  Toplam event      : {ev_total}")
    print(f"  user_id dolu      : {ev_with_uid}")
    print(f"  Senaryo tetikleme : {ev_scenarios}")
    check("Eventlerde user_id dolu", ev_with_uid > 0)

    print(f"\n  Son 5 event:")
    for e in conn.execute(
        "SELECT user_id, event_type, session_id, timestamp "
        "FROM events ORDER BY timestamp DESC LIMIT 5"
    ):
        sid = str(e["session_id"] or "")[:14]
        print(f"    user_id={e['user_id']}  type={e['event_type']:<25}  session={sid}...")
except Exception as ex:
    check("Events tablosu okunabilir", False, str(ex))

# ── 6. lab_panel.py simülasyonu ───────────────────────────────────────────────
section("6. LAB PANEL SİMÜLASYONU")
packaged = {}
if LOG_PATH.exists():
    try:
        packaged = json.loads(LOG_PATH.read_text(encoding="utf-8"))
        print(f"  Daha önce paketlenenler: {list(packaged.keys()) or 'yok'}")
    except Exception:
        pass

found = None

# Query 1: sessions JOIN users
try:
    rows = list(conn.execute("""
        SELECT s.user_id, u.name, s.experiment_group AS grp,
               s.id AS session_id, s.created_at
        FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE u.role != 'admin'
        ORDER BY s.created_at DESC
    """))
    print(f"  Sorgu 1 (sessions JOIN users)    : {len(rows)} satır")
    for r in rows:
        if str(r["user_id"]) not in packaged:
            found = dict(r); break
except Exception as e:
    print(f"  Sorgu 1 HATA: {e}")

# Query 2: events JOIN users
if not found:
    try:
        rows = list(conn.execute("""
            SELECT e.user_id, u.name,
                   MAX(e.experiment_group) AS grp,
                   MAX(e.session_id)       AS session_id,
                   MAX(e.timestamp)        AS created_at
            FROM events e
            JOIN users u ON u.id = e.user_id
            WHERE u.role != 'admin'
            GROUP BY e.user_id
            ORDER BY MAX(e.timestamp) DESC
        """))
        print(f"  Sorgu 2 (events JOIN users)      : {len(rows)} satır")
        for r in rows:
            if str(r["user_id"]) not in packaged:
                found = dict(r); break
    except Exception as e:
        print(f"  Sorgu 2 HATA: {e}")

# Query 3: direct users table
if not found:
    try:
        row = conn.execute("""
            SELECT u.id AS user_id, u.name,
                   s.experiment_group AS grp,
                   s.id AS session_id, s.created_at
            FROM users u
            LEFT JOIN sessions s ON s.user_id = u.id
            WHERE u.role != 'admin'
            ORDER BY u.id DESC LIMIT 1
        """).fetchone()
        print(f"  Sorgu 3 (direct users)           : {'1 satır' if row else '0 satır'}")
        if row and str(row["user_id"]) not in packaged:
            found = dict(row)
    except Exception as e:
        print(f"  Sorgu 3 HATA: {e}")

print()
if found:
    check("Lab panel denek bulabilir", True,
          f"User {found['user_id']} — {found.get('name','?')} (grup: {found.get('grp','?')})")
else:
    check("Lab panel denek bulabilir", False)
    print("""
  Lab panel neden bulamıyor olabilir:
    a) Tüm denekler zaten paketlenmiş (packaged_log.json'a bakın)
    b) sessions.user_id NULL (eski kod ile oluşturulmuş — git pull + yeni denek lazım)
    c) events tablosunda da user_id yok
    d) Denek admin hesabıyla giriş yaptı (admin paketlenmez)
""")

conn.close()

# ── 7. Özet ───────────────────────────────────────────────────────────────────
section("7. ÖZET VE ÖNERİLEN ADIMLAR")
if not non_admin:
    print("  → Hiç denek yok. Bir deneğin siteden kayıt olması gerekiyor.")
elif not found:
    print("  → Denek var ama lab panel görmüyor.")
    print("    1. git pull yaptın mı? (sessions.user_id fix)")
    print("    2. npm run build && npm start yaptın mı?")
    print("    3. Eski deneği sil, yeni denek al:")
    print(f"       python -c \"import sqlite3; c=sqlite3.connect('experiment.db'); c.execute('DELETE FROM sessions'); c.execute('DELETE FROM events'); c.commit()\"")
    print("    4. Yeni bir denek kayıt olsun, deneyi yapsın")
    print("    5. Sonra bat dosyasını aç")
else:
    print(f"  → Her şey normal. Lab panel User {found['user_id']} görebilmeli.")
    print("    Hâlâ sorun varsa: python lab_panel.py komutunu doğrudan çalıştır.")
