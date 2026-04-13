# ===============================
# GEREKLİ KÜTÜPHANELER
# ===============================
import sqlite3
import json
import threading
import csv
import time
from pathlib import Path

# ── Paths (always relative to THIS file, not the CWD) ────────────────────────
# Using Path(__file__) means the DB is always created next to data_logger.py
# regardless of which directory the user ran Python from.
_SCRIPT_DIR = Path(__file__).resolve().parent
_DB_PATH    = _SCRIPT_DIR / "experiment.db"

# ===============================
# THREAD LOCK
# ===============================
lock = threading.Lock()

# ===============================
# DATABASE BAĞLANTISI
# ===============================
conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
conn.row_factory = sqlite3.Row   # allows column access by name, not just index
cursor = conn.cursor()

# ===============================
# TABLOLAR
# ===============================
cursor.execute("""
CREATE TABLE IF NOT EXISTS lsl_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    scenario_name TEXT,
    scenario_type TEXT,
    experiment_group TEXT,
    phase TEXT,
    page_url TEXT,
    timestamp INTEGER,
    wall_time_ms INTEGER,
    eeg_marker INTEGER,
    payload TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS eye_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    gazepoint_time REAL,
    wall_time_ms INTEGER,
    gaze_x REAL,
    gaze_y REAL,
    pupil_left REAL,
    pupil_right REAL
)
""")

conn.commit()

# Migration: add new columns to tables that may have been created by an older
# version of this file (silently ignored if the column already exists).
for _sql in [
    "ALTER TABLE lsl_events ADD COLUMN wall_time_ms INTEGER",
    "ALTER TABLE eye_data ADD COLUMN gazepoint_time REAL",
    "ALTER TABLE eye_data ADD COLUMN wall_time_ms INTEGER",
]:
    try:
        cursor.execute(_sql)
        conn.commit()
    except Exception:
        pass  # Column already exists


# ===============================
# SCENARIO + EEG MARKER
# ===============================
def save_event(data):
    try:
        with lock:
            # Duplicate guard: same session + scenario_name + timestamp
            cursor.execute(
                """
                SELECT COUNT(*) FROM lsl_events
                WHERE session_id=? AND scenario_name=? AND timestamp=?
                """,
                (data.get("session_id"), data.get("scenario_name"), data.get("timestamp"))
            )
            if cursor.fetchone()[0] > 0:
                print(f"[WARN] save_event: Duplicate skipped — {data.get('scenario_name')} @ {data.get('timestamp')}")
                return

            cursor.execute(
                """
                INSERT INTO lsl_events (
                    session_id,
                    scenario_name,
                    scenario_type,
                    experiment_group,
                    phase,
                    page_url,
                    timestamp,
                    wall_time_ms,
                    eeg_marker,
                    payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.get("session_id"),
                    data.get("scenario_name"),
                    data.get("scenario_type"),
                    data.get("experiment_group"),
                    data.get("phase"),
                    data.get("page_url"),
                    data.get("timestamp"),
                    data.get("wall_time_ms"),
                    data.get("eeg_marker"),
                    json.dumps(data)
                )
            )
            conn.commit()
    except Exception as e:
        print(f"[ERROR] save_event: {e}")


# ===============================
# EYE TRACKING
# ===============================
def save_eye_data(session_id, gazepoint_time, wall_time_ms, gaze_x, gaze_y, pupil_left, pupil_right):
    """
    gazepoint_time : float  — Gazepoint's own TIME field (seconds, relative clock)
    wall_time_ms   : int    — Python time.time()*1000 at sample receipt (Unix ms,
                              same scale as JS scenario timestamps → used for alignment)
    """
    try:
        with lock:
            cursor.execute(
                """
                INSERT INTO eye_data (
                    session_id,
                    gazepoint_time,
                    wall_time_ms,
                    gaze_x,
                    gaze_y,
                    pupil_left,
                    pupil_right
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    gazepoint_time,
                    wall_time_ms,
                    gaze_x,
                    gaze_y,
                    pupil_left,
                    pupil_right
                )
            )
            conn.commit()
    except Exception as e:
        print(f"[ERROR] save_eye_data: {e}")


# ===============================
# SCENARIO BASED EXPORT
# ===============================
def export_scenario_dataset(session_id, window_before=2000, window_after=3000, output_dir=None):
    """
    Produces a scenario-centred dataset CSV.

    All timestamps compared here are in the same unit — Unix milliseconds:
      • lsl_events.wall_time_ms  — recorded by Python at trigger time
      • eye_data.wall_time_ms    — recorded by Python when the sample arrived
      • events.timestamp         — JS performance.now() converted to ms epoch

    Parameters
    ----------
    output_dir : str | Path | None
        Directory where the CSV is written.  Defaults to the current directory.
    """
    output_dir = Path(output_dir) if output_dir else _SCRIPT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"SCENARIO_DATASET_{session_id}_{int(time.time())}.csv"

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "scenario_name",
            "event_source",
            "event_type",
            "timestamp_ms",
            "relative_time_ms",

            "mouse_x",
            "mouse_y",
            "scroll_y",

            "gaze_x",
            "gaze_y",
            "pupil_left",
            "pupil_right",

            "eeg_marker"
        ])

        # Use explicit column names to avoid index-based errors
        cursor.execute("""
            SELECT scenario_name, wall_time_ms, timestamp, eeg_marker
            FROM lsl_events
            WHERE session_id=?
            ORDER BY wall_time_ms ASC
        """, (session_id,))
        scenarios = cursor.fetchall()

        for scenario_row in scenarios:
            scenario_name = scenario_row["scenario_name"]
            eeg_marker    = scenario_row["eeg_marker"]
            # Prefer wall_time_ms (same clock as eye data); fall back to JS timestamp
            scenario_ref_ms = scenario_row["wall_time_ms"] or scenario_row["timestamp"]

            if scenario_ref_ms is None:
                print(f"[WARN] export: No usable timestamp for scenario '{scenario_name}', skipping.")
                continue

            start = scenario_ref_ms - window_before
            end   = scenario_ref_ms + window_after

            # ------------------------------------------------------------------
            # MOUSE events (Next.js `events` table — may not exist on this DB)
            # ------------------------------------------------------------------
            try:
                cursor.execute("""
                    SELECT event_type, event_data, timestamp
                    FROM events
                    WHERE session_id=? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp ASC
                """, (session_id, start, end))

                for row in cursor.fetchall():
                    evt_data = json.loads(row["event_data"]) if row["event_data"] else {}
                    ts = row["timestamp"]
                    writer.writerow([
                        scenario_name, "mouse", row["event_type"],
                        ts, ts - scenario_ref_ms,
                        evt_data.get("x"), evt_data.get("y"), evt_data.get("scrollY"),
                        "", "", "", "",
                        eeg_marker
                    ])
            except Exception as e:
                print(f"[WARN] export: Could not read 'events' table ({e}). Mouse data skipped.")

            # ------------------------------------------------------------------
            # EYE data — use wall_time_ms for comparison (same clock as scenario)
            # ------------------------------------------------------------------
            cursor.execute("""
                SELECT wall_time_ms, gaze_x, gaze_y, pupil_left, pupil_right
                FROM eye_data
                WHERE session_id=? AND wall_time_ms BETWEEN ? AND ?
                ORDER BY wall_time_ms ASC
            """, (session_id, start, end))

            for row in cursor.fetchall():
                ts = row["wall_time_ms"]
                writer.writerow([
                    scenario_name, "eye", "gaze",
                    ts, ts - scenario_ref_ms,
                    "", "", "",
                    row["gaze_x"], row["gaze_y"], row["pupil_left"], row["pupil_right"],
                    eeg_marker
                ])

            # ------------------------------------------------------------------
            # Scenario trigger row itself
            # ------------------------------------------------------------------
            writer.writerow([
                scenario_name, "scenario", "trigger",
                scenario_ref_ms, 0,
                "", "", "", "", "", "", "",
                eeg_marker
            ])

    print(f"[INFO] SCENARIO DATASET READY: {filename}")
    return str(filename)
