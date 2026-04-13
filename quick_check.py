"""Hizli durum kontrolu — python quick_check.py"""
import sqlite3, json
from pathlib import Path

D = Path(__file__).resolve().parent
db = D / "experiment.db"
log = D / "packaged_log.json"

print("=== PACKAGED LOG ===")
if log.exists():
    data = json.loads(log.read_text(encoding="utf-8"))
    if data:
        for uid, info in data.items():
            print(f"  user_id={uid}  folder={info.get('folder','?')}")
    else:
        print("  (bos)")
else:
    print("  packaged_log.json yok")

print()
print("=== USERS ===")
conn = sqlite3.connect(str(db))
conn.row_factory = sqlite3.Row
for r in conn.execute("SELECT id,name,email,role FROM users"):
    print(f"  id={r['id']} role={r['role']} name={r['name']} email={r['email']}")

print()
print("=== SESSIONS (son 5) ===")
for r in conn.execute(
    "SELECT id,user_id,experiment_group,phase,created_at FROM sessions ORDER BY created_at DESC LIMIT 5"
):
    print(f"  {str(r['id'] or '')[:14]}  user_id={r['user_id']}  grp={r['experiment_group']}  phase={r['phase']}")

print()
print("=== EYE DATA ===")
cols = [r[1] for r in conn.execute("PRAGMA table_info(eye_data)").fetchall()]
print(f"  Kolonlar: {cols}")
n = conn.execute("SELECT COUNT(*) FROM eye_data").fetchone()[0]
print(f"  Satir sayisi: {n}")

print()
print("=== LAB PANEL QUERY TEST ===")
packaged = {}
if log.exists():
    packaged = json.loads(log.read_text(encoding="utf-8"))

found = None

# Query 1
rows = list(conn.execute("""
    SELECT s.user_id, u.name, s.experiment_group AS grp, s.id AS session_id, s.created_at
    FROM sessions s JOIN users u ON u.id = s.user_id
    WHERE u.role != 'admin' ORDER BY s.created_at DESC
"""))
print(f"  Q1 (sessions JOIN users): {len(rows)} satir")
for r in rows:
    in_pkg = str(r['user_id']) in packaged
    print(f"    user_id={r['user_id']} name={r['name']} packaged={in_pkg}")
    if not found and not in_pkg:
        found = dict(r)

if not found:
    rows2 = list(conn.execute("""
        SELECT e.user_id, u.name,
               MAX(e.experiment_group) AS grp,
               MAX(e.session_id) AS session_id,
               MAX(e.timestamp) AS created_at
        FROM events e JOIN users u ON u.id = e.user_id
        WHERE u.role != 'admin' GROUP BY e.user_id
        ORDER BY MAX(e.timestamp) DESC
    """))
    print(f"  Q2 (events JOIN users): {len(rows2)} satir")
    for r in rows2:
        in_pkg = str(r['user_id']) in packaged
        print(f"    user_id={r['user_id']} name={r['name']} packaged={in_pkg}")
        if not found and not in_pkg:
            found = dict(r)

if not found:
    row3 = conn.execute("""
        SELECT u.id AS user_id, u.name,
               s.experiment_group AS grp, s.id AS session_id, s.created_at
        FROM users u LEFT JOIN sessions s ON s.user_id = u.id
        WHERE u.role != 'admin' ORDER BY u.id DESC LIMIT 1
    """).fetchone()
    print(f"  Q3 (direct users): {dict(row3) if row3 else 'None'}")
    if row3 and str(row3['user_id']) not in packaged:
        found = dict(row3)

print()
print(f"  SONUC: {'BULUNDU -> ' + str(found) if found else 'BULUNAMADI'}")
conn.close()
