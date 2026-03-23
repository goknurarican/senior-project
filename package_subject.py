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
  python package_subject.py --list                      # Kullanıcıları listele
  python package_subject.py --user-id 5                 # Tek kullanıcı paketle
  python package_subject.py --user-id 5 --eeg-dir "E:/" # EEG dosyalarıyla birlikte
  python package_subject.py --all                       # Herkesi paketle

Çıktı:
  subjects/
  └── user_005_variant_a/
      ├── eeg/            ← .eeg/.vhdr/.vmrk
      ├── eye/            ← gaze_data.csv
      ├── platform/       ← events, mouse, scenarios
      └── metadata.json
"""

import argparse, json, os, shutil, sqlite3, sys
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR / "experiment.db"
OUTPUT_DIR = SCRIPT_DIR / "subjects"

GAZEPOINT_DIRS = [Path.home()/"Documents"/"Gazepoint", Path.home()/"Documents", SCRIPT_DIR]
EEG_DIRS = [Path("C:/Users/rdadmin/Desktop"), Path("C:/BrainVision/Data"), Path("D:/EEG_Data")]


def get_db():
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found"); sys.exit(1)
    return sqlite3.connect(str(DB_PATH))


def list_users():
    conn = get_db(); c = conn.cursor()
    print(f"\n{'='*60}\n  USERS & EXPERIMENT DATA\n{'='*60}")
    c.execute("SELECT id, name, email, role FROM users ORDER BY id")
    for uid, name, email, role in c.fetchall():
        if role == 'admin': continue
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


def extract_user_data(user_id):
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    user = c.fetchone()
    if not user:
        print(f"  ERROR: User {user_id} not found!"); conn.close(); return None
    cols = [d[0] for d in c.description]
    info = dict(zip(cols, user))

    # Session bul
    c.execute("SELECT id, experiment_group FROM sessions WHERE user_id=? ORDER BY created_at DESC LIMIT 1", (user_id,))
    sess = c.fetchone()
    if not sess:
        c.execute("SELECT DISTINCT session_id, experiment_group FROM events WHERE user_id=?", (user_id,))
        sess = c.fetchone()
    session_id = sess[0] if sess else "unknown"
    group = sess[1] if sess else "unknown"

    # Events
    c.execute("""SELECT id,session_id,user_id,experiment_group,event_type,event_data,page_url,timestamp,relative_t_ms,created_at
                 FROM events WHERE user_id=? OR session_id=? ORDER BY timestamp ASC""", (user_id, session_id))
    ecols = [d[0] for d in c.description]
    events = [dict(zip(ecols, r)) for r in c.fetchall()]

    data = {
        "user_id": user_id, "user_info": info, "group": group, "session_id": session_id,
        "all_events": events,
        "mouse_trajectories": [e for e in events if e["event_type"]=="mouse_trajectory"],
        "mouse_clicks": [e for e in events if e["event_type"]=="mouse_click"],
        "scenarios": [e for e in events if e["event_type"]=="SCENARIO_TRIGGERED"],
        "page_views": [e for e in events if e["event_type"]=="page_view"],
    }
    conn.close()
    print(f"  User {user_id} ({info.get('name','N/A')}) | {group}")
    print(f"    Events:{len(events)} Mouse:{len(data['mouse_trajectories'])} Clicks:{len(data['mouse_clicks'])} Scenarios:{len(data['scenarios'])}")
    return data


def save_csv(rows, path, cols=None):
    if not rows: return
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        if cols: df = df[[c for c in cols if c in df.columns]]
        df.to_csv(path, index=False)
    else:
        import csv
        keys = cols or list(rows[0].keys())
        with open(path,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow({k:r.get(k,"") for k in keys})


def find_gaze_file():
    for d in GAZEPOINT_DIRS:
        if not d.exists(): continue
        for p in ["*gaze*.csv","*gaze*.xlsx"]:
            files = sorted(d.glob(p), key=os.path.getmtime, reverse=True)
            if files: return files[0]
    return None


def find_eeg(eeg_dir=None):
    dirs = [eeg_dir] if eeg_dir else EEG_DIRS
    for d in dirs:
        if not d or not d.exists(): continue
        vhdrs = sorted(d.glob("*.vhdr"), key=os.path.getmtime, reverse=True)
        if vhdrs:
            stem = vhdrs[0].stem
            return [f for f in [d/f"{stem}.vhdr", d/f"{stem}.eeg", d/f"{stem}.vmrk"] if f.exists()]
    return []


def package_subject(user_id, eeg_dir=None, gaze_file=None):
    print(f"\n{'='*60}\n  PACKAGING: User {user_id}\n{'='*60}")
    data = extract_user_data(user_id)
    if not data: return

    name = (data["user_info"].get("name","") or "").replace(" ","_").lower()
    folder = f"user_{user_id:03d}"
    if name and name not in ("n/a","none",""): folder += f"_{name}"
    folder += f"_{data['group']}"

    sdir = OUTPUT_DIR / folder; sdir.mkdir(parents=True, exist_ok=True)
    print(f"\n  -> {sdir}")

    # Platform
    pdir = sdir / "platform"; pdir.mkdir(exist_ok=True)
    ecols = ["id","event_type","event_data","page_url","timestamp","relative_t_ms","created_at"]
    save_csv(data["all_events"], pdir/"all_events.csv", ecols)
    save_csv(data["mouse_trajectories"], pdir/"mouse_trajectories.csv", ecols)
    save_csv(data["mouse_clicks"], pdir/"mouse_clicks.csv", ecols)
    save_csv(data["scenarios"], pdir/"scenario_triggers.csv", ecols)
    save_csv(data["page_views"], pdir/"page_views.csv", ecols)
    for k in ["all_events","mouse_trajectories","mouse_clicks","scenarios","page_views"]:
        if data[k]: print(f"    {k}: {len(data[k])} rows")

    # Session info
    meta = {"user_id":user_id, "name":data["user_info"].get("name"),
            "email":data["user_info"].get("email"), "session_id":data["session_id"],
            "group":data["group"], "packaged_at":datetime.now().isoformat(),
            "counts":{k:len(data[k]) for k in ["all_events","mouse_trajectories","mouse_clicks","scenarios","page_views"]}}
    json.dump(meta, open(pdir/"session_info.json","w",encoding="utf-8"), indent=2, ensure_ascii=False)

    # Eye tracker
    edir = sdir / "eye"; edir.mkdir(exist_ok=True)
    gf = gaze_file or find_gaze_file()
    if gf and Path(gf).exists():
        shutil.copy2(gf, edir/Path(gf).name); print(f"    Eye: {Path(gf).name}")
    else:
        print(f"    Eye: NOT FOUND -> copy manually to {edir}/")
        (edir/"PLACE_GAZE_DATA_HERE.txt").write_text("Copy Gazepoint CSV here\n")

    # EEG
    eegdir = sdir / "eeg"; eegdir.mkdir(exist_ok=True)
    eeg_files = find_eeg(Path(eeg_dir) if eeg_dir else None)
    if eeg_files:
        for f in eeg_files: shutil.copy2(f, eegdir/f.name); print(f"    EEG: {f.name}")
    else:
        print(f"    EEG: NOT FOUND -> copy .eeg/.vhdr/.vmrk to {eegdir}/")
        (eegdir/"PLACE_EEG_FILES_HERE.txt").write_text("Copy BrainVision .eeg/.vhdr/.vmrk here\n")

    # Metadata
    meta["eeg_found"] = len(eeg_files); meta["eye_found"] = gf is not None
    json.dump(meta, open(sdir/"metadata.json","w",encoding="utf-8"), indent=2, ensure_ascii=False)

    # Config hint
    print(f"\n  sdp-data-ml/src/config.py'a ekle:")
    print(f'    "user_{user_id:03d}": {{"vhdr":"...vhdr","eeg":"...eeg","vmrk":"...vmrk","group":"{data["group"]}","session_id":"{data["session_id"]}"}}')
    print(f"\n  DONE: {sdir}")
    return sdir


def main():
    p = argparse.ArgumentParser(description="BİTİRMEEG Subject Packager")
    p.add_argument("--user-id", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--eeg-dir", type=str)
    p.add_argument("--gaze-file", type=str)
    p.add_argument("--db", type=str)
    a = p.parse_args()

    global DB_PATH
    if a.db: DB_PATH = Path(a.db)

    if a.list: list_users(); return
    if a.all:
        conn = get_db(); c = conn.cursor()
        c.execute("SELECT id FROM users WHERE role!='admin' ORDER BY id")
        ids = [r[0] for r in c.fetchall()]; conn.close()
        for uid in ids: package_subject(uid, a.eeg_dir, a.gaze_file)
        return
    if a.user_id: package_subject(a.user_id, a.eeg_dir, a.gaze_file); return
    p.print_help()

if __name__ == "__main__":
    main()
