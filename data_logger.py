# ===============================
# GEREKLİ KÜTÜPHANELER
# ===============================
import sqlite3
import json
import threading
import csv
import time

# ===============================
# THREAD LOCK
# ===============================
lock = threading.Lock()

# ===============================
# DATABASE BAĞLANTISI
# ===============================
conn = sqlite3.connect("experiment.db", check_same_thread=False)
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
    eeg_marker INTEGER,
    payload TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS eye_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    timestamp REAL,
    gaze_x REAL,
    gaze_y REAL,
    pupil_left REAL,
    pupil_right REAL
)
""")

conn.commit()

# ===============================
# SCENARIO + EEG MARKER
# ===============================
def save_event(data):
    try:
        with lock:
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
                    eeg_marker,
                    payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.get("session_id"),
                    data.get("scenario_name"),
                    data.get("scenario_type"),
                    data.get("experiment_group"),
                    data.get("phase"),
                    data.get("page_url"),
                    data.get("timestamp"),
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
def save_eye_data(session_id, timestamp, gaze_x, gaze_y, pupil_left, pupil_right):
    try:
        with lock:
            cursor.execute(
                """
                INSERT INTO eye_data (
                    session_id,
                    timestamp,
                    gaze_x,
                    gaze_y,
                    pupil_left,
                    pupil_right
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    timestamp,
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
# 🔥 SCENARIO BASED EXPORT (ANA)
# ===============================
def export_scenario_dataset(session_id, window_before=2000, window_after=3000):
    """
    Senaryo merkezli dataset üretir (en doğru analiz formatı)
    """

    filename = f"SCENARIO_DATASET_{session_id}_{int(time.time())}.csv"

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # HEADER
        writer.writerow([
            "scenario_name",
            "event_source",
            "event_type",
            "timestamp",
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

        # SCENARIO'LAR
        cursor.execute("SELECT * FROM lsl_events WHERE session_id=?", (session_id,))
        scenarios = cursor.fetchall()

        for s in scenarios:
            scenario_time = s[7]
            scenario_name = s[2]
            eeg_marker = s[8]

            start = scenario_time - window_before
            end = scenario_time + window_after

            # ======================
            # MOUSE (Next.js events tablosu)
            # ======================
            cursor.execute("""
                SELECT event_type, event_data, timestamp
                FROM events
                WHERE session_id=? AND timestamp BETWEEN ? AND ?
            """, (session_id, start, end))

            for row in cursor.fetchall():
                data = json.loads(row[1]) if row[1] else {}

                writer.writerow([
                    scenario_name,
                    "mouse",
                    row[0],
                    row[2],
                    row[2] - scenario_time,

                    data.get("x"),
                    data.get("y"),
                    data.get("scrollY"),

                    "", "", "", "",

                    eeg_marker
                ])

            # ======================
            # EYE
            # ======================
            cursor.execute("""
                SELECT timestamp, gaze_x, gaze_y, pupil_left, pupil_right
                FROM eye_data
                WHERE session_id=? AND timestamp BETWEEN ? AND ?
            """, (session_id, start, end))

            for row in cursor.fetchall():
                writer.writerow([
                    scenario_name,
                    "eye",
                    "gaze",
                    row[0],
                    row[0] - scenario_time,

                    "", "", "",

                    row[1],
                    row[2],
                    row[3],
                    row[4],

                    eeg_marker
                ])

            # ======================
            # SCENARIO TRIGGER
            # ======================
            writer.writerow([
                scenario_name,
                "scenario",
                "trigger",
                scenario_time,
                0,

                "", "", "", "", "", "", "",

                eeg_marker
            ])

    print(f"🔥 SCENARIO DATASET READY: {filename}")