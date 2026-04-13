"""DB onarım scripti — SONRAKI_DENEK.bat tarafından çağrılır."""
import sqlite3, pathlib, sys

db = pathlib.Path(__file__).resolve().parent / "experiment.db"
if not db.exists():
    sys.exit(0)

conn = sqlite3.connect(str(db))
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
conn.close()
if n:
    print(f"  [OK] {n} oturum user_id onarildi.")
