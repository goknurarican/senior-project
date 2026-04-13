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
conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=10)
conn.execute("PRAGMA journal_mode=WAL")    # allow concurrent readers + 1 writer
conn.execute("PRAGMA busy_timeout=10000")  # wait up to 10s instead of failing instantly
conn.execute("PRAGMA synchronous=NORMAL")  # faster writes, still safe with WAL
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# ── Eye-data write buffer ─────────────────────────────────────────────────────
# Gazepoint sends ~150 samples/sec. Writing + committing each sample individually
# floods the DB and causes SQLITE_BUSY for every other writer (Next.js).
# We buffer samples and flush in a background thread every 0.5 s (≈75 rows/batch).
_eye_buffer: list = []
_eye_buffer_lock = threading.Lock()
_EYE_BATCH_SIZE = 75   # also flush when this many rows accumulate


def _flush_eye_buffer_locked():
    """Flush _eye_buffer to DB. Caller must hold _eye_buffer_lock."""
    global _eye_buffer
    if not _eye_buffer:
        return
    rows = _eye_buffer
    _eye_buffer = []
    try:
        with lock:
            cursor.executemany(
                """INSERT INTO eye_data
                   (session_id, gazepoint_time, wall_time_ms,
                    gaze_x, gaze_y, pupil_left, pupil_right)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
    except Exception as exc:
        print(f"[ERROR] eye flush: {exc}")


def _eye_flush_thread():
    """Background thread: flush buffered eye samples every 0.5 s."""
    while True:
        time.sleep(0.5)
        with _eye_buffer_lock:
            _flush_eye_buffer_locked()


threading.Thread(target=_eye_flush_thread, daemon=True).start()

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
    Buffer a single eye sample. The background flush thread commits to DB every 0.5 s.
    gazepoint_time : float — Gazepoint TIME field (relative seconds)
    wall_time_ms   : int   — Unix ms at sample receipt (used for cross-stream alignment)
    """
    with _eye_buffer_lock:
        _eye_buffer.append((session_id, gazepoint_time, wall_time_ms,
                            gaze_x, gaze_y, pupil_left, pupil_right))
        if len(_eye_buffer) >= _EYE_BATCH_SIZE:
            _flush_eye_buffer_locked()


# ===============================
# SCENARIO BASED EXPORT
# ===============================
def export_scenario_dataset(session_id, window_before=2000, window_after=3000,
                            output_dir=None, baseline_interval=10000):
    """
    Produces a scenario-centred dataset CSV for ML training.

    Rows cover two kinds of windows, both using the same wall_time_ms clock:

    1. SCENARIO windows  (phase = actual phase name, e.g. "variant_b")
       Centred on each scenario trigger ±window_before/window_after ms.
       label = scenario_type (e.g. "slow_image", "network_jitter", …)

    2. BASELINE windows  (phase = "control")
       Sampled every `baseline_interval` ms from the control phase
       (before the phase_change marker).
       label = "baseline_control"
       Used as the negative class in ML training.

    All timestamps are Unix ms. Columns:
        phase, label, scenario_name,
        event_source, event_type,
        timestamp_ms, relative_time_ms,
        mouse_x, mouse_y, scroll_y,
        gaze_x, gaze_y, pupil_left, pupil_right,
        eeg_marker
    """
    output_dir = Path(output_dir) if output_dir else _SCRIPT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"SCENARIO_DATASET_{session_id}_{int(time.time())}.csv"

    HEADER = [
        "phase", "label", "scenario_name",
        "event_source", "event_type",
        "timestamp_ms", "relative_time_ms",
        "mouse_x", "mouse_y", "scroll_y",
        "gaze_x", "gaze_y", "pupil_left", "pupil_right",
        "eeg_marker",
    ]

    def write_window(writer, ref_ms, phase, label, scenario_name, eeg_marker,
                     start, end):
        """Write mouse + eye rows for one time window."""
        try:
            cursor.execute("""
                SELECT event_type, event_data, timestamp
                FROM events
                WHERE session_id=? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """, (session_id, start, end))
            for row in cursor.fetchall():
                ed = json.loads(row["event_data"]) if row["event_data"] else {}
                ts = row["timestamp"]
                writer.writerow([
                    phase, label, scenario_name,
                    "mouse", row["event_type"],
                    ts, ts - ref_ms,
                    ed.get("x"), ed.get("y"), ed.get("scrollY") or ed.get("sy"),
                    "", "", "", "",
                    eeg_marker,
                ])
        except Exception as e:
            print(f"[WARN] export: events table error ({e})")

        cursor.execute("""
            SELECT wall_time_ms, gaze_x, gaze_y, pupil_left, pupil_right
            FROM eye_data
            WHERE session_id=? AND wall_time_ms BETWEEN ? AND ?
            ORDER BY wall_time_ms ASC
        """, (session_id, start, end))
        for row in cursor.fetchall():
            ts = row["wall_time_ms"]
            writer.writerow([
                phase, label, scenario_name,
                "eye", "gaze",
                ts, ts - ref_ms,
                "", "", "",
                row["gaze_x"], row["gaze_y"],
                row["pupil_left"], row["pupil_right"],
                eeg_marker,
            ])

    # ── Fetch all lsl_events for this session ─────────────────────────────
    cursor.execute("""
        SELECT scenario_name, scenario_type, phase, wall_time_ms, timestamp, eeg_marker
        FROM lsl_events
        WHERE session_id=?
        ORDER BY wall_time_ms ASC
    """, (session_id,))
    all_events = cursor.fetchall()

    # ── Determine phase_change timestamp ──────────────────────────────────
    phase_change_ms = None
    variant_phase   = "variant"
    for ev in all_events:
        if ev["scenario_type"] == "phase_change" and ev["wall_time_ms"]:
            phase_change_ms = ev["wall_time_ms"]
            variant_phase   = ev["phase"] or "variant"
            break

    # ── Scenario start times (for gap exclusion in baseline) ──────────────
    scenario_windows = []
    for ev in all_events:
        ref = ev["wall_time_ms"] or ev["timestamp"]
        if ref and ev["scenario_type"] not in ("phase_change", "experiment_end", None):
            scenario_windows.append((ref - window_before, ref + window_after))

    def overlaps_scenario(t):
        return any(s <= t <= e for s, e in scenario_windows)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        # ── 1. SCENARIO windows ───────────────────────────────────────────
        scenario_count = 0
        for ev in all_events:
            stype = ev["scenario_type"]
            if stype in ("phase_change", "experiment_end", None):
                continue

            ref_ms = ev["wall_time_ms"] or ev["timestamp"]
            if ref_ms is None:
                continue

            phase      = ev["phase"] or variant_phase
            eeg_marker = ev["eeg_marker"]
            start, end = ref_ms - window_before, ref_ms + window_after

            writer.writerow([
                phase, stype, ev["scenario_name"],
                "scenario", "trigger",
                ref_ms, 0,
                "", "", "", "", "", "", "",
                eeg_marker,
            ])
            write_window(writer, ref_ms, phase, stype, ev["scenario_name"],
                         eeg_marker, start, end)
            scenario_count += 1

        # ── 2. BASELINE windows from control phase ────────────────────────
        baseline_count = 0
        if phase_change_ms:
            # Find earliest data point in control phase
            cursor.execute("""
                SELECT MIN(wall_time_ms) FROM eye_data
                WHERE session_id=? AND wall_time_ms < ?
            """, (session_id, phase_change_ms))
            r = cursor.fetchone()
            control_start = r[0] if r and r[0] else None

            if control_start:
                # Sample baseline windows every baseline_interval ms,
                # skip any that overlap a scenario trigger window
                t = control_start + window_before
                while t + window_after <= phase_change_ms:
                    if not overlaps_scenario(t):
                        write_window(writer, t, "control", "baseline_control",
                                     "baseline_control", 0,
                                     t - window_before, t + window_after)
                        # Write anchor row
                        writer.writerow([
                            "control", "baseline_control", "baseline_control",
                            "scenario", "baseline_anchor",
                            t, 0,
                            "", "", "", "", "", "", "", 0,
                        ])
                        baseline_count += 1
                    t += baseline_interval

    print(f"[INFO] SCENARIO DATASET: {filename.name} "
          f"({scenario_count} scenario windows, {baseline_count} baseline windows)")
    return str(filename)
