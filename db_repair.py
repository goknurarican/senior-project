"""DB onarım scripti — SONRAKI_DENEK.bat tarafından çağrılır."""
import sqlite3, json, pathlib, sys

SCRIPT_DIR       = pathlib.Path(__file__).resolve().parent
db               = SCRIPT_DIR / "experiment.db"
PACKAGED_LOG     = SCRIPT_DIR / "packaged_log.json"

if not db.exists():
    sys.exit(0)

conn = sqlite3.connect(str(db))

# 1) session user_id onarımı
n = conn.execute("""
    UPDATE sessions SET user_id = (
        SELECT e.user_id FROM events e
        WHERE e.session_id = sessions.id AND e.user_id IS NOT NULL LIMIT 1
    )
    WHERE user_id IS NULL AND EXISTS (
        SELECT 1 FROM events e
        WHERE e.session_id = sessions.id AND e.user_id IS NOT NULL
    )
""").rowcount
conn.commit()
if n:
    print(f"  [OK] {n} oturum user_id onarildi.")

# 2) packaged_log.json temizleme — DB silinip yeniden oluşturulduysa
#    eski log kayıtları aynı ID'li yeni katılımcıları engelleyebilir.
if PACKAGED_LOG.exists():
    try:
        log = json.loads(PACKAGED_LOG.read_text(encoding="utf-8"))
        rows = conn.execute("SELECT id FROM users WHERE role != 'admin'").fetchall()
        valid_ids = {str(r[0]) for r in rows}
        stale = [uid for uid in log if uid not in valid_ids]
        if stale:
            for uid in stale:
                del log[uid]
            PACKAGED_LOG.write_text(
                json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  [OK] {len(stale)} eski paket kaydı temizlendi.")
    except Exception as exc:
        print(f"  [WARN] packaged_log temizleme hatasi: {exc}")

conn.close()
