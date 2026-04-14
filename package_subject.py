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
    # Session / phase markers
    "control_start":  1,
    "variant_start":  2,
    "experiment_end": 99,
    # UX frustration scenarios
    "slow_image":        11,
    "broken_image":      12,
    "skeleton_prolong":  13,
    "search_irrelevant": 14,
    "button_delay":      15,
    "first_click_miss":  16,
    "feedback_late":     17,
    "network_jitter":    18,
    "overlay_blocking":  19,
    "price_change":      20,
    "coupon_min_spend":  21,
    "coupon_expired":    22,
    "facet_reset_once":  23,
    "sort_reset":        24,
    # Task-level user action markers
    "add_to_cart":       30,
    "checkout_start":    31,
    "purchase_complete":  32,
    "search_performed":  33,
    "product_viewed":    34,
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

    DESCRIPTIONS = {
        1:  "Control phase start — participant logged in, eye tracking live",
        2:  "Variant phase start — UX frustration scenarios now active",
        99: "Experiment end — session complete",
        11: "Slow image load (3-5 s delay on product images)",
        12: "Broken image / 404 asset",
        13: "Skeleton prolonged (loading skeleton stays > 3 s)",
        14: "Search results shuffled to irrelevant order",
        15: "Button unresponsive / delayed (4 s)",
        16: "First click misses target (button jumps away)",
        17: "Feedback / alert delivered late (1.2 s delay)",
        18: "Network jitter — all API calls throttled",
        19: "Overlay blocking — modal covers content",
        20: "Price change warning in cart",
        21: "Coupon rejected — minimum spend not met",
        22: "Coupon rejected — code expired",
        23: "Filter / facet reset unexpectedly",
        24: "Sort order reset unexpectedly",
        30: "User added a product to cart",
        31: "User navigated to checkout page",
        32: "Order placed successfully",
        33: "User submitted a search query",
        34: "User opened a product detail page",
    }

    with open(eegdir / "marker_legend.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eeg_marker_value", "bv_label", "scenario_name", "description"])
        for name, val in rows:
            w.writerow([val, f"S {val}", name, DESCRIPTIONS.get(val, "")])

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
    Export eye_data rows for this session from SQLite to CSV.

    Columns:
        wall_time_ms            Unix ms — alignment key across all streams
        gazepoint_time          Gazepoint relative seconds
        gaze_x / gaze_y         Best-available POG, 0–1 normalised
        pupil_left / pupil_right Pupil diameter mm (LPMMV/RPMMV)
        bpogv                   1 = best-POG valid (not a blink / lost-tracking)
        fpogv                   1 = fixation-POG valid
        phase                   "control" or variant name (from phase_change marker)
        active_scenario         scenario_type active at this sample (within 3 s of trigger),
                                "" if no scenario is active
    """
    conn = get_db()
    c = conn.cursor()

    # ── Phase transition time ─────────────────────────────────────────────
    phase_change_ms = None
    phase_name      = "variant"
    try:
        c.execute("""
            SELECT wall_time_ms, phase FROM lsl_events
            WHERE session_id=? AND scenario_type='variant_start'
            ORDER BY wall_time_ms ASC LIMIT 1
        """, (session_id,))
        pc = c.fetchone()
        if pc and pc["wall_time_ms"]:
            phase_change_ms = pc["wall_time_ms"]
            phase_name      = pc["phase"] or "variant"
    except Exception:
        pass

    # ── Scenario trigger windows ──────────────────────────────────────────
    # For each eye sample, we want to know if a scenario was triggered
    # in the [0, SCENARIO_EFFECT_MS] window before this sample.
    SCENARIO_EFFECT_MS = 3000
    scenario_triggers = []
    try:
        c.execute("""
            SELECT scenario_type, wall_time_ms FROM lsl_events
            WHERE session_id=?
              AND scenario_type NOT IN ('variant_start','control_start','experiment_end')
              AND scenario_type IS NOT NULL
              AND wall_time_ms IS NOT NULL
            ORDER BY wall_time_ms ASC
        """, (session_id,))
        scenario_triggers = [(r["scenario_type"], r["wall_time_ms"])
                             for r in c.fetchall()]
    except Exception:
        pass

    def active_scenario_at(t_ms):
        """Return the scenario_type active at t_ms, or '' if none."""
        for stype, trigger_ms in scenario_triggers:
            if trigger_ms <= t_ms <= trigger_ms + SCENARIO_EFFECT_MS:
                return stype
        return ""

    # ── Eye rows ──────────────────────────────────────────────────────────
    try:
        c.execute("""
            SELECT wall_time_ms, gazepoint_time, gaze_x, gaze_y,
                   pupil_left, pupil_right,
                   COALESCE(bpogv, 0) AS bpogv,
                   COALESCE(fpogv, 0) AS fpogv
            FROM eye_data
            WHERE session_id=?
            ORDER BY wall_time_ms ASC
        """, (session_id,))
        rows = c.fetchall()
    except Exception as e:
        print(f"    [WARN] eye_data not readable ({e}) — skipped.")
        conn.close()
        return 0
    conn.close()

    if not rows:
        print(f"    Eye DB: 0 rows for session {session_id}")
        return 0

    control_count = variant_count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["wall_time_ms", "gazepoint_time", "gaze_x", "gaze_y",
                    "pupil_left", "pupil_right", "bpogv", "fpogv",
                    "phase", "active_scenario"])
        for r in rows:
            t = r["wall_time_ms"]
            if phase_change_ms and t:
                phase = "control" if t < phase_change_ms else phase_name
            else:
                phase = "unknown"
            if phase == "control":
                control_count += 1
            else:
                variant_count += 1
            w.writerow([
                t, r["gazepoint_time"],
                r["gaze_x"], r["gaze_y"],
                r["pupil_left"], r["pupil_right"],
                r["bpogv"], r["fpogv"],
                phase, active_scenario_at(t) if t else "",
            ])

    valid_count = sum(1 for r in rows if r["bpogv"])
    print(f"    Eye DB: {output_path.name} — {len(rows)} rows "
          f"[control={control_count}, {phase_name}={variant_count}, "
          f"bpogv_valid={valid_count}/{len(rows)}]")
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
# Mouse trajectory exploder
# ---------------------------------------------------------------------------

def _mouse_batch_features(pts):
    """
    Compute aggregate ML features for one trajectory batch (list of point dicts).

    Returns a dict with:
        n_points        number of samples in batch
        x_flips         horizontal direction reversals (left→right→left counts)
        y_flips         vertical direction reversals
        auc_px          area between actual path and straight first→last line (px)
        idle_ratio      fraction of samples with velocity < 0.05 px/ms
        mean_velocity   mean px/ms
        max_velocity    max px/ms
        path_length_px  total Euclidean distance travelled
    """
    import math as _math
    if len(pts) < 2:
        return {"n_points": len(pts), "x_flips": 0, "y_flips": 0, "auc_px": 0.0,
                "idle_ratio": 0.0, "mean_velocity": 0.0, "max_velocity": 0.0,
                "path_length_px": 0.0}

    xs, ys, vs = [], [], []
    path_len = 0.0
    for p in pts:
        try:
            xs.append(float(p["x"])); ys.append(float(p["y"]))
            v = float(p.get("velocity") or 0)
            vs.append(v)
        except (TypeError, ValueError):
            continue

    if not xs:
        return {"n_points": 0, "x_flips": 0, "y_flips": 0, "auc_px": 0.0,
                "idle_ratio": 0.0, "mean_velocity": 0.0, "max_velocity": 0.0,
                "path_length_px": 0.0}

    # X-Flips and Y-Flips
    x_flips = sum(1 for i in range(1, len(xs)-1)
                  if (xs[i]-xs[i-1]) * (xs[i+1]-xs[i]) < 0)
    y_flips = sum(1 for i in range(1, len(ys)-1)
                  if (ys[i]-ys[i-1]) * (ys[i+1]-ys[i]) < 0)

    # Path length
    for i in range(1, len(xs)):
        path_len += _math.sqrt((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2)

    # AUC — sum of perpendicular distances from each point to first→last line
    x0, y0, x1, y1 = xs[0], ys[0], xs[-1], ys[-1]
    dx, dy = x1 - x0, y1 - y0
    seg_len = _math.sqrt(dx*dx + dy*dy)
    auc = 0.0
    if seg_len > 0:
        for px, py in zip(xs[1:-1], ys[1:-1]):
            auc += abs((px-x0)*dy - (py-y0)*dx) / seg_len
    else:
        for px, py in zip(xs, ys):
            auc += _math.sqrt((px-x0)**2 + (py-y0)**2)

    IDLE_THRESH = 0.05  # px/ms
    idle_count  = sum(1 for v in vs if v < IDLE_THRESH)

    return {
        "n_points":       len(xs),
        "x_flips":        x_flips,
        "y_flips":        y_flips,
        "auc_px":         round(auc, 2),
        "idle_ratio":     round(idle_count / len(vs), 4) if vs else 0.0,
        "mean_velocity":  round(sum(vs)/len(vs), 6) if vs else 0.0,
        "max_velocity":   round(max(vs), 6) if vs else 0.0,
        "path_length_px": round(path_len, 2),
    }


def export_mouse_points(trajectory_events, click_events, output_dir: Path):
    """
    Explode batched mouse trajectory events into ML-ready feature rows.

    Trajectory CSV columns (mouse_trajectory_points.csv):
      wall_time_ms   Unix ms — aligns with eye_data.wall_time_ms and lsl_events.wall_time_ms
      x, y           viewport pixels (clientX/clientY)
      x_norm, y_norm 0–1 normalised (same coordinate space as Gazepoint gaze_x/gaze_y)
      scroll_y       window.scrollY — absolute page position
      velocity       px/ms between this and previous point (erratic movement indicator)
      acceleration   Δvelocity/Δt — sudden stops/bursts
      page_url
      session_id

    Click CSV columns (mouse_clicks_flat.csv):
      wall_time_ms
      x, y, x_norm, y_norm, scroll_y
      button         0=left 1=middle 2=right
      is_rage_click  1 if ≥2 left clicks within 500 ms and 60 px of same spot
      target, class_name
      page_url, session_id
    """
    import json as _json
    import math

    # ── Trajectory points ──────────────────────────────────────────────────
    points = []
    for ev in trajectory_events:
        try:
            ed = _json.loads(ev.get("event_data") or "{}")
            path = ed.get("path", [])
        except Exception:
            continue
        page     = ev.get("page_url", "")
        sid      = ev.get("session_id", "")
        # experiment_group == phase for this event ("control" or "variant_*")
        phase    = ev.get("experiment_group") or "control"
        screen_w = ed.get("screen_w") or None
        screen_h = ed.get("screen_h") or None
        for pt in path:
            points.append({
                "wall_time_ms": pt.get("t", ""),
                "x":            pt.get("x", ""),
                "y":            pt.get("y", ""),
                "x_norm":       round(pt["x"] / screen_w, 6) if screen_w and pt.get("x") != "" else "",
                "y_norm":       round(pt["y"] / screen_h, 6) if screen_h and pt.get("y") != "" else "",
                "scroll_y":     pt.get("sy", ""),
                "velocity":     "",       # filled in next pass
                "acceleration": "",
                "phase":        phase,
                "page_url":     page,
                "session_id":   sid,
            })

    # Compute velocity and acceleration between consecutive points
    for i, pt in enumerate(points):
        if i == 0:
            pt["velocity"] = 0.0
            pt["acceleration"] = 0.0
            continue
        prev = points[i - 1]
        try:
            dx = float(pt["x"]) - float(prev["x"])
            dy = float(pt["y"]) - float(prev["y"])
            dt = float(pt["wall_time_ms"]) - float(prev["wall_time_ms"])
            v  = math.sqrt(dx*dx + dy*dy) / dt if dt > 0 else 0.0
            pt["velocity"] = round(v, 6)
            prev_v = float(prev["velocity"]) if prev["velocity"] != "" else 0.0
            pt["acceleration"] = round((v - prev_v) / dt, 8) if dt > 0 else 0.0
        except (TypeError, ValueError):
            pt["velocity"] = ""
            pt["acceleration"] = ""

    TRAJ_COLS = ["wall_time_ms", "x", "y", "x_norm", "y_norm", "scroll_y",
                 "velocity", "acceleration", "phase", "page_url", "session_id"]
    if points:
        with open(output_dir / "mouse_trajectory_points.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=TRAJ_COLS)
            w.writeheader()
            w.writerows(points)
        print(f"    Mouse trajectory points: {len(points)} rows "
              f"(velocity + norm included)")
    else:
        print(f"    Mouse trajectory points: 0 rows")

    # ── Per-batch trajectory summary (X-Flips, AUC, Idle ratio) ──────────
    summary_rows = []
    for ev in trajectory_events:
        try:
            ed = _json.loads(ev.get("event_data") or "{}")
            path = ed.get("path", [])
        except Exception:
            continue
        if not path:
            continue
        feats = _mouse_batch_features(path)
        sw = ed.get("screen_w") or None
        sh = ed.get("screen_h") or None
        summary_rows.append({
            "wall_time_ms_start": path[0].get("t", ""),
            "wall_time_ms_end":   path[-1].get("t", ""),
            "phase":              ev.get("experiment_group") or "control",
            "page_url":           ev.get("page_url", ""),
            "screen_w":           sw,
            "screen_h":           sh,
            "dpr":                ed.get("dpr", ""),
            **feats,
        })

    if summary_rows:
        SUMM_COLS = ["wall_time_ms_start", "wall_time_ms_end", "phase", "page_url",
                     "screen_w", "screen_h", "dpr",
                     "n_points", "x_flips", "y_flips", "auc_px",
                     "idle_ratio", "mean_velocity", "max_velocity", "path_length_px"]
        with open(output_dir / "mouse_trajectory_summary.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUMM_COLS)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"    Mouse trajectory summary: {len(summary_rows)} batches "
              f"(X-Flips, AUC, Idle included)")

    # ── Clicks flat ────────────────────────────────────────────────────────
    # Rage click: left-button click within 500 ms AND 60 px of a previous click
    RAGE_WINDOW_MS = 500
    RAGE_RADIUS_PX = 60

    raw_clicks = []
    for ev in click_events:
        try:
            ed = _json.loads(ev.get("event_data") or "{}")
        except Exception:
            ed = {}
        try:
            sw = ed.get("screen_w") or None
            sh = ed.get("screen_h") or None
            cx = ed.get("x", "")
            cy = ed.get("y", "")
        except Exception:
            sw = sh = None; cx = cy = ""
        raw_clicks.append({
            "wall_time_ms": ev.get("timestamp", ""),
            "x":            cx,
            "y":            cy,
            "x_norm":       round(float(cx) / sw, 6) if sw and cx != "" else "",
            "y_norm":       round(float(cy) / sh, 6) if sh and cy != "" else "",
            "scroll_y":     ed.get("sy", ""),
            "button":       ed.get("button", 0),
            "is_rage_click": 0,
            "phase":        ev.get("experiment_group") or "control",
            "target":       ed.get("target", ""),
            "class_name":   ed.get("className", ""),
            "page_url":     ev.get("page_url", ""),
            "session_id":   ev.get("session_id", ""),
        })

    for i, ck in enumerate(raw_clicks):
        if ck["button"] != 0:
            continue
        try:
            t1 = float(ck["wall_time_ms"]); x1 = float(ck["x"]); y1 = float(ck["y"])
        except (TypeError, ValueError):
            continue
        for prev in raw_clicks[max(0, i-10):i]:
            if prev["button"] != 0:
                continue
            try:
                t0 = float(prev["wall_time_ms"]); x0 = float(prev["x"]); y0 = float(prev["y"])
            except (TypeError, ValueError):
                continue
            if (t1 - t0) <= RAGE_WINDOW_MS and math.sqrt((x1-x0)**2 + (y1-y0)**2) <= RAGE_RADIUS_PX:
                ck["is_rage_click"] = 1
                break

    CLICK_COLS = ["wall_time_ms", "x", "y", "x_norm", "y_norm", "scroll_y",
                  "button", "is_rage_click", "phase", "target", "class_name",
                  "page_url", "session_id"]
    if raw_clicks:
        with open(output_dir / "mouse_clicks_flat.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CLICK_COLS)
            w.writeheader()
            w.writerows(raw_clicks)
        rage = sum(1 for c in raw_clicks if c["is_rage_click"])
        print(f"    Mouse clicks (flat): {len(raw_clicks)} rows "
              f"({rage} rage clicks detected)")

    # ── Drag events ───────────────────────────────────────────────────────
    drag_events = [e for e in trajectory_events + click_events
                   if e.get("event_type") == "mouse_drag"]
    # Note: drag events are logged via logEvent so they appear in all_events,
    # not in trajectory_events. Re-fetch from the original events list if needed.
    # The caller (package_subject) filters these separately — handled below.

    return len(points), len(raw_clicks)


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

    # Exploded mouse files (one row per point, wall_time_ms aligned)
    export_mouse_points(data["mouse_trajectories"], data["mouse_clicks"], pdir)

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

    # ── EEG markers from lsl_events (session-specific, Drive-friendly) ─────
    # This is the primary record linking BrainVision TTL values to scenario
    # details.  Unlike marker_legend.csv (which is a static mapping table),
    # this file has one row per marker FIRED during this participant's session.
    if data["session_id"] not in (None, "unknown"):
        try:
            c = get_db().cursor()
            c.execute("""
                SELECT wall_time_ms, scenario_name, scenario_type,
                       eeg_marker, phase, page_url
                FROM lsl_events
                WHERE session_id=?
                ORDER BY wall_time_ms ASC
            """, (data["session_id"],))
            lsl_rows = c.fetchall()
            if lsl_rows:
                eeg_markers_path = eegdir / "eeg_markers.csv"
                with open(eeg_markers_path, "w", newline="", encoding="utf-8") as _f:
                    _w = csv.writer(_f)
                    _w.writerow([
                        "wall_time_ms", "bv_label",
                        "eeg_marker", "scenario_name", "scenario_type",
                        "phase", "page_url",
                    ])
                    for r in lsl_rows:
                        _w.writerow([
                            r["wall_time_ms"],
                            f"S {r['eeg_marker']}" if r["eeg_marker"] else "",
                            r["eeg_marker"],
                            r["scenario_name"],
                            r["scenario_type"],
                            r["phase"],
                            r["page_url"],
                        ])
                print(f"    EEG markers CSV: {len(lsl_rows)} rows → eeg_markers.csv")
                meta["eeg_marker_rows"] = len(lsl_rows)
            else:
                print(f"    EEG markers CSV: no lsl_events found for this session")
        except Exception as e:
            print(f"    [WARN] eeg_markers.csv export failed: {e}")

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
