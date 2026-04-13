#!/usr/bin/env python3
"""
BİTİRMEEG - Subject Data Packager
===================================
Deney sonrası tüm verileri user_id bazlı klasörlere otomatik toplar.

LAB SETUP:
  PC1 (Windows): BrainVision Recorder + EEG amp (paralel port)
  PC2 (Windows): Next.js platform + Gazepoint eye tracker + trigger_server.py

Bu script PC2'de (platform bilgisayarı) çalışır.
EEG dosyaları USB ile kopyalanır veya ağ paylaşımından alınır.

Kullanım:
  python package_subject.py --list
  python package_subject.py --user-id 5
  python package_subject.py --user-id 5 --eeg-dir "E:/" --gaze-file "C:/Users/.../default_2024-01-15.csv"
  python package_subject.py --all      ← WARNING: reads file association notes below

OUTPUT per subject:
  subjects/
  └── user_005_john_doe_variant_a/
      ├── eeg/
      │   ├── recording.eeg          ← BrainVision binary (copied from Recorder PC)
      │   ├── recording.vhdr         ← BrainVision header
      │   ├── recording.vmrk         ← BrainVision markers (S 11 = slow_image, etc.)
      │   ├── marker_legend.csv      ← TTL value → scenario name mapping
      │   └── marker_legend.json     ← same, for Python scripts
      ├── eye/
      │   ├── default_2024-01-15.csv ← Gazepoint Control native CSV (for GP Analysis)
      │   └── eye_data_db.csv        ← our DB records filtered by session_id
      ├── platform/
      │   ├── all_events.csv
      │   ├── mouse_trajectories.csv
      │   ├── mouse_clicks.csv
      │   ├── scenario_triggers.csv
      │   ├── page_views.csv
      │   ├── SCENARIO_DATASET_*.csv ← scenario-aligned analysis CSV
      │   └── session_info.json
      └── metadata.json

SEQUENTIAL PARTICIPANTS — IMPORTANT:
  Run package_subject.py IMMEDIATELY after each participant, BEFORE starting
  the next one.  The script picks the most-recently-modified Gazepoint and
  BrainVision files from the search directories.  If you run --all at the
  end of the day you must supply --gaze-file and --eeg-dir explicitly for
  each user, or all users will get the same (last) recording files.
"""

import argparse, csv, json, os, shutil, sqlite3, sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    from data_logger import export_scenario_dataset
    HAS_DATA_LOGGER = True
except Exception as _dl_err:
    print(f"[WARN] data_logger not importable ({_dl_err}); scenario CSV export skipped.")
    HAS_DATA_LOGGER = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

SCRIPT_DIR  = Path(__file__).parent
DB_PATH     = SCRIPT_DIR / "experiment.db"
OUTPUT_DIR  = SCRIPT_DIR / "subjects"

# Gazepoint Control saves to Documents/Gazepoint/ by default.
GAZEPOINT_DIRS = [
    Path.home() / "Documents" / "Gazepoint",
    Path.home() / "Documents",
    SCRIPT_DIR,
]

# BrainVision Recorder common save locations.
# Path.home() resolves to the actual logged-in Windows user (e.g. C:/Users/labpc).
# Hardcoded paths are kept as fallbacks for specific lab machine configs.
EEG_DIRS = [
    Path.home() / "Desktop",              # current Windows user's Desktop
    Path.home() / "Documents",            # current Windows user's Documents
    Path("C:/Users/rdadmin/Desktop"),     # lab-specific fallback
    Path("C:/BrainVision/Data"),
    Path("D:/EEG_Data"),
    Path("E:/"),                          # USB drive common drive letter
]

# Must match SCENARIO_MARKER_MAP in trigger_server.py exactly.
# Used to write the marker legend so BrainVision Analyzer users know
# what S 11, S 12 … mean.
SCENARIO_MARKER_MAP = {
    "slow_image":       11,
    "broken_image":     12,
    "skeleton_prolong": 13,
    "search_irrelevant":14,
    "button_delay":     15,
    "first_click_miss": 16,
    "feedback_late":    17,
    "network_jitter":   18,
    "overlay_blocking": 19,
    "price_change":     20,
    "coupon_min_spend": 21,
    "coupon_expired":   22,
    "facet_reset_once": 23,
    "sort_reset":       24,
}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db():
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found"); sys.exit(1)
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


# ---------------------------------------------------------------------------
# File finders
# ---------------------------------------------------------------------------

def find_gaze_file():
    """
    Locate the most-recently-modified Gazepoint Control CSV.

    Gazepoint Control default filename patterns (in priority order):
        default_YYYY-MM-DD_HH-MM-SS.csv   most common — timestamped default profile
        default.csv                        older Gazepoint versions
        *_YYYY-*.csv                       custom profile with timestamp
        *.csv                              last resort, only within Gazepoint folder

    Stops at the first GAZEPOINT_DIR that has any matching file, so a broad
    pattern like *.csv never escapes the Gazepoint-specific folder.
    """
    patterns = [
        "default_*.csv",   # timestamped default profile (most common)
        "default.csv",     # un-timestamped default
        "*_2*.csv",        # any profile starting with a 20xx year timestamp
        "*.csv",           # anything in the folder (last resort)
    ]
    for d in GAZEPOINT_DIRS:
        if not d.exists():
            continue
        for p in patterns:
            files = [f for f in d.glob(p) if f.is_file()]
            if files:
                return max(files, key=lambda f: f.stat().st_mtime)
        # Only fall through to the next directory if this one had zero CSV files at all
    return None


def find_eeg(eeg_dir=None):
    """
    Locate the most-recently-modified BrainVision triplet (.vhdr/.eeg/.vmrk).
    Returns a list of Path objects for the files that exist.
    """
    dirs = [Path(eeg_dir)] if eeg_dir else EEG_DIRS
    for d in dirs:
        if not d or not d.exists():
            continue
        vhdrs = sorted(d.glob("*.vhdr"), key=os.path.getmtime, reverse=True)
        if vhdrs:
            stem = vhdrs[0].stem
            return [f for f in [d / f"{stem}.vhdr",
                                 d / f"{stem}.eeg",
                                 d / f"{stem}.vmrk"] if f.exists()]
    return []


# ---------------------------------------------------------------------------
# BrainVision marker legend
# ---------------------------------------------------------------------------

def write_marker_legend(eegdir: Path):
    """
    Write marker_legend.csv and marker_legend.json to the EEG folder.

    BrainVision Analyzer shows markers as "S 11", "S 12" etc.
    Without this file the researcher cannot know which scenario each number
    corresponds to.

    CSV format (readable in Excel and BrainVision Analyzer annotation notes):
        eeg_marker_value, bv_label, scenario_name
        11,               S 11,     slow_image
        12,               S 12,     broken_image
        ...
    """
    rows = sorted(SCENARIO_MARKER_MAP.items(), key=lambda x: x[1])

    with open(eegdir / "marker_legend.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eeg_marker_value", "bv_label", "scenario_name"])
        for name, val in rows:
            w.writerow([val, f"S {val}", name])

    json.dump(
        {str(v): k for k, v in SCENARIO_MARKER_MAP.items()},
        open(eegdir / "marker_legend.json", "w", encoding="utf-8"),
        indent=2,
    )
    print(f"    Marker legend: marker_legend.csv + marker_legend.json")


# ---------------------------------------------------------------------------
# Eye DB export
# ---------------------------------------------------------------------------

def export_eye_db(session_id, output_path: Path):
    """
    Export eye_data rows for this session from our SQLite DB to a CSV.

    This is NOT a replacement for the native Gazepoint CSV (which has many
    more columns and is what Gazepoint Analysis software reads).  This file
    is for alignment in our Python analysis pipeline — it uses wall_time_ms
    (Unix ms) so it can be directly compared to scenario timestamps.

    Columns:
        wall_time_ms   — Unix ms when Python received the sample (alignment key)
        gazepoint_time — Gazepoint's own TIME field (relative seconds)
        gaze_x / gaze_y           — fixation POG, 0–1 normalized
        pupil_left / pupil_right  — pupil diameter in mm
    """
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute("""
            SELECT wall_time_ms, gazepoint_time, gaze_x, gaze_y, pupil_left, pupil_right
            FROM eye_data
            WHERE session_id=?
            ORDER BY wall_time_ms ASC
        """, (session_id,))
        rows = c.fetchall()
    except Exception as e:
        print(f"    [WARN] eye_data table not readable ({e}) — skipped.")
        conn.close()
        return 0
    conn.close()

    if not rows:
        print(f"    Eye DB: 0 rows for session {session_id}")
        return 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["wall_time_ms", "gazepoint_time", "gaze_x", "gaze_y",
                    "pupil_left", "pupil_right"])
        for r in rows:
            w.writerow([r["wall_time_ms"], r["gazepoint_time"],
                        r["gaze_x"], r["gaze_y"],
                        r["pupil_left"], r["pupil_right"]])

    print(f"    Eye DB export: {output_path.name} ({len(rows)} rows)")
    return len(rows)


# ---------------------------------------------------------------------------
# Platform data extraction
# ---------------------------------------------------------------------------

def extract_user_data(user_id):
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    user = c.fetchone()
    if not user:
        print(f"  ERROR: User {user_id} not found!")
        conn.close()
        return None
    info = dict(user)

    # Prefer the sessions table; fall back to events
    c.execute("""
        SELECT id, experiment_group, created_at
        FROM sessions WHERE user_id=?
        ORDER BY created_at DESC LIMIT 1
    """, (user_id,))
    sess = c.fetchone()
    if not sess:
        c.execute("""
            SELECT DISTINCT session_id, experiment_group
            FROM events WHERE user_id=?
        """, (user_id,))
        sess = c.fetchone()

    session_id     = sess["id"]          if sess else "unknown"
    group          = sess[1]             if sess else "unknown"
    session_start  = sess["created_at"]  if sess and "created_at" in sess.keys() else None

    c.execute("""
        SELECT id, session_id, user_id, experiment_group,
               event_type, event_data, page_url,
               timestamp, relative_t_ms, created_at
        FROM events
        WHERE user_id=? OR session_id=?
        ORDER BY timestamp ASC
    """, (user_id, session_id))
    events = [dict(r) for r in c.fetchall()]
    conn.close()

    data = {
        "user_id":      user_id,
        "user_info":    info,
        "group":        group,
        "session_id":   session_id,
        "session_start": session_start,
        "all_events":   events,
        "mouse_trajectories": [e for e in events if e["event_type"] == "mouse_trajectory"],
        "mouse_clicks":       [e for e in events if e["event_type"] == "mouse_click"],
        "scenarios":          [e for e in events if e["event_type"] == "SCENARIO_TRIGGERED"],
        "page_views":         [e for e in events if e["event_type"] == "page_view"],
    }
    print(f"  User {user_id} ({info.get('name','N/A')}) | {group} | session: {session_id}")
    print(f"    Events:{len(events)} Mouse:{len(data['mouse_trajectories'])} "
          f"Clicks:{len(data['mouse_clicks'])} Scenarios:{len(data['scenarios'])}")
    return data


def save_csv(rows, path, cols=None):
    if not rows:
        return
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        if cols:
            df = df[[c for c in cols if c in df.columns]]
        df.to_csv(path, index=False)
    else:
        keys = cols or list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})


# ---------------------------------------------------------------------------
# Main packaging function
# ---------------------------------------------------------------------------

def package_subject(user_id, eeg_dir=None, gaze_file=None):
    print(f"\n{'='*60}\n  PACKAGING: User {user_id}\n{'='*60}")
    data = extract_user_data(user_id)
    if not data:
        return None

    name   = (data["user_info"].get("name", "") or "").replace(" ", "_").lower()
    folder = f"user_{user_id:03d}"
    if name and name not in ("n/a", "none", ""):
        folder += f"_{name}"
    folder += f"_{data['group']}"

    sdir = OUTPUT_DIR / folder
    sdir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output folder: {sdir}")

    # ── Platform events ────────────────────────────────────────────────────
    pdir = sdir / "platform"
    pdir.mkdir(exist_ok=True)
    ecols = ["id", "event_type", "event_data", "page_url",
             "timestamp", "relative_t_ms", "created_at"]
    save_csv(data["all_events"],         pdir / "all_events.csv",         ecols)
    save_csv(data["mouse_trajectories"], pdir / "mouse_trajectories.csv", ecols)
    save_csv(data["mouse_clicks"],       pdir / "mouse_clicks.csv",       ecols)
    save_csv(data["scenarios"],          pdir / "scenario_triggers.csv",  ecols)
    save_csv(data["page_views"],         pdir / "page_views.csv",         ecols)
    for k in ["all_events", "mouse_trajectories", "mouse_clicks", "scenarios", "page_views"]:
        if data[k]:
            print(f"    {k}: {len(data[k])} rows")

    meta = {
        "user_id":       user_id,
        "name":          data["user_info"].get("name"),
        "email":         data["user_info"].get("email"),
        "session_id":    data["session_id"],
        "session_start": data["session_start"],
        "group":         data["group"],
        "packaged_at":   datetime.now().isoformat(),
        "counts": {k: len(data[k])
                   for k in ["all_events", "mouse_trajectories",
                              "mouse_clicks", "scenarios", "page_views"]},
    }
    json.dump(meta, open(pdir / "session_info.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # ── Scenario-aligned analysis CSV (from our DB) ────────────────────────
    if HAS_DATA_LOGGER and data["session_id"] not in (None, "unknown"):
        try:
            csv_path = export_scenario_dataset(data["session_id"], output_dir=pdir)
            print(f"    Scenario dataset: {Path(csv_path).name}")
            meta["scenario_csv"] = Path(csv_path).name
        except Exception as e:
            print(f"    [WARN] Scenario CSV export failed: {e}")

    # ── Eye tracker ────────────────────────────────────────────────────────
    edir = sdir / "eye"
    edir.mkdir(exist_ok=True)

    # 1. Native Gazepoint CSV (what Gazepoint Analysis software reads)
    gf = Path(gaze_file) if gaze_file else find_gaze_file()
    if gf and gf.exists():
        shutil.copy2(gf, edir / gf.name)
        print(f"    Eye (Gazepoint native): {gf.name}")
        meta["gaze_native_file"] = gf.name
    else:
        print(f"    Eye (Gazepoint native): NOT FOUND")
        print(f"      -> Copy the Gazepoint CSV manually to {edir}/")
        print(f"      -> Or re-run with: --gaze-file \"path/to/default_YYYY-MM-DD.csv\"")
        (edir / "PLACE_GAZEPOINT_CSV_HERE.txt").write_text(
            "Copy the Gazepoint Control recording CSV here.\n"
            "Default location: Documents/Gazepoint/default_YYYY-MM-DD_HH-MM-SS.csv\n"
        )

    # 2. Eye data from our DB (session-filtered, wall-clock aligned, for Python)
    if data["session_id"] not in (None, "unknown"):
        n = export_eye_db(data["session_id"], edir / "eye_data_db.csv")
        meta["eye_db_rows"] = n

    # ── EEG (BrainVision) ──────────────────────────────────────────────────
    eegdir = sdir / "eeg"
    eegdir.mkdir(exist_ok=True)

    eeg_files = find_eeg(eeg_dir)
    if eeg_files:
        for f in eeg_files:
            shutil.copy2(f, eegdir / f.name)
            print(f"    EEG: {f.name}")
        meta["eeg_files"] = [f.name for f in eeg_files]
    else:
        print(f"    EEG: NOT FOUND")
        print(f"      -> Copy .eeg/.vhdr/.vmrk to {eegdir}/")
        print(f"      -> Or re-run with: --eeg-dir \"E:/BrainVision/\"")
        (eegdir / "PLACE_EEG_FILES_HERE.txt").write_text(
            "Copy BrainVision Recorder files here:\n"
            "  recording.eeg   — raw binary EEG data\n"
            "  recording.vhdr  — header (text)\n"
            "  recording.vmrk  — markers (text); TTL values 11-24 = scenarios\n"
            "See marker_legend.csv for the mapping.\n"
        )

    # Always write the marker legend so the researcher knows S 11 = slow_image etc.
    write_marker_legend(eegdir)

    # ── Final metadata ─────────────────────────────────────────────────────
    meta["eeg_found"]  = len(eeg_files)
    meta["eye_found"]  = gf is not None and gf.exists() if gf else False
    json.dump(meta, open(sdir / "metadata.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print(f"\n  sdp-data-ml/src/config.py hint:")
    print(f'    "user_{user_id:03d}": {{'
          f'"vhdr":"...vhdr","eeg":"...eeg","vmrk":"...vmrk",'
          f'"group":"{data["group"]}","session_id":"{data["session_id"]}"}}')
    print(f"\n  DONE: {sdir}")
    return sdir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_users():
    conn = get_db()
    c = conn.cursor()
    print(f"\n{'='*60}\n  USERS & EXPERIMENT DATA\n{'='*60}")
    c.execute("SELECT id, name, email, role FROM users ORDER BY id")
    for row in c.fetchall():
        uid, name, email, role = row["id"], row["name"], row["email"], row["role"]
        if role == "admin":
            continue
        c.execute("SELECT COUNT(*) FROM events WHERE user_id=?", (uid,))
        ev = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM events WHERE user_id=? AND event_type='mouse_trajectory'", (uid,))
        mouse = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM events WHERE user_id=? AND event_type='SCENARIO_TRIGGERED'", (uid,))
        scen = c.fetchone()[0]
        c.execute("SELECT DISTINCT experiment_group FROM events WHERE user_id=? AND experiment_group IS NOT NULL", (uid,))
        groups = [r[0] for r in c.fetchall()]
        group = groups[0] if groups else "?"
        print(f"\n  User {uid}: {name or 'N/A'} ({email})")
        print(f"    Group: {group} | Events: {ev} | Mouse: {mouse} | Scenarios: {scen}")
    conn.close()


def main():
    p = argparse.ArgumentParser(
        description="BİTİRMEEG Subject Packager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SEQUENTIAL PARTICIPANTS:
  Run this script immediately after each participant, before starting the next.
  The Gazepoint and EEG file finders pick the most recently modified file.
  If you run --all at end-of-day, supply --gaze-file and --eeg-dir explicitly
  for each user, or files will be associated incorrectly.

EXAMPLES:
  # List all participants in the DB
  python package_subject.py --list

  # Package one participant (auto-find Gazepoint + EEG)
  python package_subject.py --user-id 5

  # Package with explicit file paths (recommended for reliability)
  python package_subject.py --user-id 5 \\
      --gaze-file "C:/Users/lab/Documents/Gazepoint/default_2024-01-15_10-30-00.csv" \\
      --eeg-dir "E:/BrainVision/Data"
""",
    )
    p.add_argument("--user-id",   type=int,  help="Package a single participant by DB user id")
    p.add_argument("--all",       action="store_true", help="Package all non-admin participants")
    p.add_argument("--list",      action="store_true", help="List all participants and their data counts")
    p.add_argument("--eeg-dir",   type=str,  help="Directory containing .eeg/.vhdr/.vmrk files")
    p.add_argument("--gaze-file", type=str,  help="Path to specific Gazepoint CSV file")
    p.add_argument("--db",        type=str,  help="Override database path")
    a = p.parse_args()

    global DB_PATH
    if a.db:
        DB_PATH = Path(a.db)

    if a.list:
        list_users()
        return

    if a.all:
        print("\n[WARN] --all mode: Gazepoint/EEG files are matched by most-recent-modified.")
        print("       Run this immediately after EACH participant for correct file association,")
        print("       or supply --gaze-file and --eeg-dir explicitly per user.\n")
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE role!='admin' ORDER BY id")
        ids = [r["id"] for r in c.fetchall()]
        conn.close()
        for uid in ids:
            package_subject(uid, a.eeg_dir, a.gaze_file)
        return

    if a.user_id:
        package_subject(a.user_id, a.eeg_dir, a.gaze_file)
        return

    p.print_help()


if __name__ == "__main__":
    main()
