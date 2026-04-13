#!/usr/bin/env python3
"""
BİTİRMEEG — Tanı + Onarım Scripti
Çalıştır: python diagnose.py
"""
import sqlite3
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH    = SCRIPT_DIR / "experiment.db"
LOG_PATH   = SCRIPT_DIR / "packaged_log.json"

SEP  = "=" * 60

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def check(label, ok, detail=""):
    mark = "OK" if ok else "HATA"
    print(f"  [{mark}]  {label}" + (f" — {detail}" if detail else ""))

# ── 1. Dosya sistemi ──────────────────────────────────────────────────────────
section("1. DOSYA SİSTEMİ")
check("experiment.db mevcut", DB_PATH.exists(), str(DB_PATH))

if not DB_PATH.exists():
    print(f"""
  experiment.db bulunamadı — beklenen: {DB_PATH}

  Çözüm:
    1. Proje klasöründen: npm start
    2. Bir denek siteden kayıt olsun
    3. Bu scripti tekrar çalıştır
""")
    raise SystemExit(1)

check("packaged_log.json", LOG_PATH.exists(),
      "henüz yok (normal)" if not LOG_PATH.exists() else str(LOG_PATH))

# ── 2. Tablolar ───────────────────────────────────────────────────────────────
section("2. VERİTABANI TABLOLARI")
conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row

tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
print(f"  Tablolar: {tables}")
for t in ["users", "sessions", "events"]:
    try:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        check(f"{t}", True, f"{n} satır")
    except Exception as e:
        check(f"{t}", False, str(e))

# ── 3. Kullanıcılar ───────────────────────────────────────────────────────────
section("3. KULLANICILAR")
users = list(conn.execute(
    "SELECT id, name, email, role FROM users ORDER BY id"))
for u in users:
    tag = " ← DENEK" if u["role"] != "admin" else " (admin)"
    print(f"  id={u['id']:3d}  role={u['role']:<6}  "
          f"name={u['name'] or '—':<22}  email={u['email']}{tag}")
non_admin = [u for u in users if u["role"] != "admin"]
check("En az 1 denek var", len(non_admin) > 0, f"{len(non_admin)} denek")

# ── 4. Oturumlar ──────────────────────────────────────────────────────────────
section("4. OTURUMLAR (sessions)")
sessions = list(conn.execute(
    "SELECT id, user_id, experiment_group, phase, created_at "
    "FROM sessions ORDER BY created_at DESC LIMIT 10"))

null_uid = sum(1 for s in sessions if s["user_id"] is None)
for s in sessions:
    sid = (s["id"] or "")[:16]
    uid_str = str(s["user_id"]) if s["user_id"] is not None else "NULL ⚠"
    print(f"  {sid:<18}  user_id={uid_str:<6}  "
          f"group={s['experiment_group'] or '?':<12}  phase={s['phase'] or '?'}")

if null_uid > 0:
    print(f"\n  ⚠  {null_uid} oturumda user_id=NULL (eski kod)")

check("En az 1 oturumda user_id dolu",
      null_uid < len(sessions),
      f"{len(sessions)-null_uid}/{len(sessions)} doldu")

# ── 5. Eventler ───────────────────────────────────────────────────────────────
section("5. EVENTLER")
ev_total    = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
ev_with_uid = conn.execute(
    "SELECT COUNT(*) FROM events WHERE user_id IS NOT NULL").fetchone()[0]
ev_scen     = conn.execute(
    "SELECT COUNT(*) FROM events WHERE event_type='SCENARIO_TRIGGERED'").fetchone()[0]
print(f"  Toplam event      : {ev_total}")
print(f"  user_id dolu      : {ev_with_uid}")
print(f"  Senaryo tetikleme : {ev_scen}")
check("Eventlerde user_id var", ev_with_uid > 0)

print(f"\n  Son 5 event:")
for e in conn.execute(
    "SELECT user_id, event_type, session_id FROM events "
    "ORDER BY timestamp DESC LIMIT 5"):
    sid = str(e["session_id"] or "")[:16]
    print(f"    user_id={e['user_id']}  "
          f"type={str(e['event_type']):<28}  session={sid}...")

# ── 6. OTOMATİK ONARIM ───────────────────────────────────────────────────────
section("6. OTOMATİK ONARIM")
print("  sessions.user_id = NULL olan oturumlar events tablosundan dolduruluyor...")

fixed = conn.execute("""
    UPDATE sessions
    SET user_id = (
        SELECT e.user_id FROM events e
        WHERE e.session_id = sessions.id
          AND e.user_id IS NOT NULL
        LIMIT 1
    )
    WHERE user_id IS NULL
      AND EXISTS (
        SELECT 1 FROM events e
        WHERE e.session_id = sessions.id
          AND e.user_id IS NOT NULL
      )
""").rowcount
conn.commit()

if fixed > 0:
    print(f"  ✔  {fixed} oturum güncellendi.")
else:
    print("  —  Güncellenecek oturum bulunamadı (ya zaten doluydu ya da event'te user_id yok).")

# Onarım sonrası durumu göster
null_after = conn.execute(
    "SELECT COUNT(*) FROM sessions WHERE user_id IS NULL").fetchone()[0]
filled_after = conn.execute(
    "SELECT COUNT(*) FROM sessions WHERE user_id IS NOT NULL").fetchone()[0]
check("Onarım sonrası durum",
      filled_after > 0,
      f"{filled_after} dolu / {null_after} hâlâ NULL")

# ── 7. Lab panel simülasyonu ──────────────────────────────────────────────────
section("7. LAB PANEL SİMÜLASYONU")
packaged = {}
if LOG_PATH.exists():
    try:
        packaged = json.loads(LOG_PATH.read_text(encoding="utf-8"))
        print(f"  Daha önce paketlenenler: {list(packaged.keys()) or 'yok'}")
    except Exception:
        pass

found = None

for qname, sql in [
    ("sessions JOIN users", """
        SELECT s.user_id, u.name, s.experiment_group AS grp,
               s.id AS session_id, s.created_at
        FROM sessions s JOIN users u ON u.id = s.user_id
        WHERE u.role != 'admin'
        ORDER BY s.created_at DESC
    """),
    ("events JOIN users", """
        SELECT e.user_id, u.name,
               MAX(e.experiment_group) AS grp,
               MAX(e.session_id)       AS session_id,
               MAX(e.timestamp)        AS created_at
        FROM events e JOIN users u ON u.id = e.user_id
        WHERE u.role != 'admin'
        GROUP BY e.user_id ORDER BY MAX(e.timestamp) DESC
    """),
    ("direct users table", """
        SELECT u.id AS user_id, u.name,
               s.experiment_group AS grp,
               s.id AS session_id, s.created_at
        FROM users u
        LEFT JOIN sessions s ON s.user_id = u.id
        WHERE u.role != 'admin'
        ORDER BY u.id DESC LIMIT 1
    """),
]:
    if found:
        break
    try:
        rows = list(conn.execute(sql))
        print(f"  Sorgu [{qname}]: {len(rows)} satır")
        for r in rows:
            if str(r["user_id"]) not in packaged:
                found = dict(r)
                break
    except Exception as e:
        print(f"  Sorgu [{qname}] HATA: {e}")

print()
if found:
    check("Lab panel denek bulabilir", True,
          f"User {found['user_id']} — {found.get('name','?')} "
          f"(grup: {found.get('grp') or '?'})")
    print("""
  ✔  Artık lab panel deneği göstermeli.
     Bat dosyasını kapat, tekrar aç.
""")
else:
    check("Lab panel denek bulabilir", False)
    if not non_admin:
        print("  → Hiç denek yok. Bir deneğin siteden kayıt olması gerekiyor.")
    else:
        print("  → Tüm denekler paketlenmiş veya eventlerde user_id yok.")
        print("    Mevcut denek user_id'lerini packaged_log.json'dan sil veya yeni denek al.")

conn.close()
