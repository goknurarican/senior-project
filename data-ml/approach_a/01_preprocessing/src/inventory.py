#!/usr/bin/env python3
"""
Bölüm 0 + Bölüm 1: Hazırlık, Klasör Yapısı ve Veri Envanteri Kontrolü
=======================================================================
Kullanım:
    python src/inventory.py            # Tüm subjectler
    python src/inventory.py --id 14   # Tek subject

Çıktılar:
    data/reports/subject_inventory.csv   - cross-subject özet tablosu
    logs/inventory.log                   - detaylı log
"""

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
CONFIG_FILE = ROOT / "config.yaml"


# ─── Config ──────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── 0.4: Reproducibility seed ───────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─── 0.5: Logging ────────────────────────────────────────

def make_logger(name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ─── 0.1: Directory structure ────────────────────────────

def ensure_dirs(cfg: dict, log: logging.Logger) -> None:
    for key, rel in cfg["paths"].items():
        p = ROOT / rel
        p.mkdir(parents=True, exist_ok=True)
        log.debug(f"  dir OK: {p}")
    log.info("0.1  Directory structure verified.")


# ─── Data class ──────────────────────────────────────────

@dataclass
class SubjectStats:
    subject_id:  int
    name:        str
    folder:      str
    group:       str

    # 1.1 File integrity
    missing_files: list = field(default_factory=list)
    empty_files:   list = field(default_factory=list)
    eeg_triple_ok: bool = False

    # 1.2 EEG
    eeg_duration_s:  float = float("nan")
    eeg_n_samples:   int   = 0
    eeg_n_channels:  int   = 0
    eeg_n_markers:   int   = 0

    # 1.2 Eye
    eye_duration_s:  float = float("nan")
    eye_n_samples:   int   = 0
    eye_bpogv_pct:   float = float("nan")

    # 1.2 Mouse
    mouse_duration_s:    float = float("nan")
    mouse_n_events:      int   = 0
    mouse_n_traj_points: int   = 0

    # 1.2 Markers
    marker_total:            int  = 0
    marker_by_type:          dict = field(default_factory=dict)
    n_unique_scenario_types: int  = 0

    # 1.3 Temporal alignment (Unix ms)
    eeg_t0_ms:           Optional[float] = None
    eeg_t1_ms:           Optional[float] = None
    eye_t0_ms:           Optional[float] = None
    eye_t1_ms:           Optional[float] = None
    mouse_t0_ms:         Optional[float] = None
    mouse_t1_ms:         Optional[float] = None
    eeg_eye_drift_s:     float = float("nan")  # seconds EEG lags behind eye
    eeg_eye_overlap_pct: float = float("nan")  # EEG–Eye pairwise (exclusion check)
    mouse_coverage_pct:  float = float("nan")  # mouse coverage within EEG window
    triple_overlap_pct:  float = float("nan")  # EEG+Eye+Mouse (flag if <90%)
    alignment_flag:      bool  = False

    # 1.5 Exclusion
    exclude:         bool = False
    exclude_reasons: list = field(default_factory=list)


# ─── 1.1: File integrity ─────────────────────────────────

_REQUIRED = [
    ("eye/eye_data_db.csv",                   "eye data"),
    ("platform/mouse_trajectory_points.csv",   "mouse trajectories"),
    ("platform/mouse_clicks.csv",              "mouse clicks"),
    ("platform/all_events.csv",                "all events"),
    ("platform/scenario_triggers.csv",         "scenario triggers"),
    ("eeg/eeg_markers.csv",                    "EEG markers CSV"),
]


def check_integrity(subject_dir: Path, s: SubjectStats, log: logging.Logger) -> None:
    missing, empty = [], []

    eeg_files  = list(subject_dir.glob("*.eeg"))
    vhdr_files = list(subject_dir.glob("*.vhdr"))
    vmrk_files = list(subject_dir.glob("*.vmrk"))

    for files, label in [(eeg_files, "*.eeg"), (vhdr_files, "*.vhdr"), (vmrk_files, "*.vmrk")]:
        if not files:
            missing.append(label)
        elif files[0].stat().st_size == 0:
            empty.append(files[0].name)

    s.eeg_triple_ok = not any(f.startswith("*.") for f in missing)

    for rel, label in _REQUIRED:
        p = subject_dir / rel
        if not p.exists():
            missing.append(rel)
            log.warning(f"    MISSING [{label}]: {rel}")
        elif p.stat().st_size == 0:
            empty.append(rel)
            log.warning(f"    EMPTY   [{label}]: {rel}")

    s.missing_files = missing
    s.empty_files   = empty

    status = "OK" if not missing and not empty else f"FAIL (missing={len(missing)}, empty={len(empty)})"
    log.info(f"  1.1  integrity: {status}")


# ─── EEG stats helpers ───────────────────────────────────

def _parse_vhdr(path: Path) -> dict:
    info = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            l = line.strip()
            if l.startswith("NumberOfChannels="):
                info["n_channels"] = int(l.split("=", 1)[1])
            elif l.startswith("SamplingInterval="):
                info["sfreq"] = 1e6 / float(l.split("=", 1)[1])
    return info


# ─── 1.2: Per-subject basic stats ────────────────────────

def compute_eeg_stats(subject_dir: Path, s: SubjectStats, log: logging.Logger) -> None:
    vhdr_files = list(subject_dir.glob("*.vhdr"))
    eeg_files  = list(subject_dir.glob("*.eeg"))
    vmrk_files = list(subject_dir.glob("*.vmrk"))
    markers_csv = subject_dir / "eeg/eeg_markers.csv"

    if not vhdr_files or not eeg_files:
        log.warning("  1.2-EEG  vhdr/eeg missing - skipped")
        return

    try:
        info   = _parse_vhdr(vhdr_files[0])
        n_ch   = info.get("n_channels", 35)
        sfreq  = info.get("sfreq", 500.0)
        n_samp = eeg_files[0].stat().st_size // (n_ch * 4)  # IEEE_FLOAT_32 = 4 bytes
        dur_s  = n_samp / sfreq

        s.eeg_n_channels = n_ch
        s.eeg_n_samples  = int(n_samp)
        s.eeg_duration_s = round(dur_s, 2)
        log.info(f"  1.2-EEG  {n_ch}ch  {dur_s:.1f}s  ({n_samp:,} samples)")
    except Exception as e:
        log.error(f"  1.2-EEG  stats error: {e}")

    if vmrk_files:
        try:
            n = sum(1 for ln in open(vmrk_files[0], encoding="utf-8", errors="ignore")
                    if ln.startswith("Mk"))
            s.eeg_n_markers = n
            log.info(f"  1.2-EEG  vmrk markers: {n}")
        except Exception as e:
            log.error(f"  1.2-EEG  vmrk read error: {e}")

    # Wall-time window from eeg_markers.csv (synchronized timestamps)
    if markers_csv.exists() and markers_csv.stat().st_size > 0:
        try:
            df = pd.read_csv(markers_csv, usecols=["wall_time_ms"]).dropna()
            if len(df) > 0:
                s.eeg_t0_ms = float(df["wall_time_ms"].min())
                s.eeg_t1_ms = float(df["wall_time_ms"].max())
        except Exception as e:
            log.error(f"  1.2-EEG  markers_csv error: {e}")


def compute_eye_stats(subject_dir: Path, s: SubjectStats, log: logging.Logger) -> None:
    p = subject_dir / "eye/eye_data_db.csv"
    if not p.exists():
        return
    try:
        df = pd.read_csv(p)
        s.eye_n_samples = len(df)

        wt = df["wall_time_ms"].dropna() if "wall_time_ms" in df.columns else pd.Series(dtype=float)
        if len(wt) > 1:
            s.eye_t0_ms     = float(wt.min())
            s.eye_t1_ms     = float(wt.max())
            s.eye_duration_s = round((wt.max() - wt.min()) / 1000, 2)

        if "bpogv" in df.columns:
            s.eye_bpogv_pct = round((df["bpogv"] > 0).mean() * 100, 1)

        log.info(f"  1.2-Eye  {s.eye_n_samples:,} samples  {s.eye_duration_s:.1f}s  "
                 f"bpogv={s.eye_bpogv_pct:.1f}%")
    except Exception as e:
        log.error(f"  1.2-Eye  error: {e}")


def compute_mouse_stats(subject_dir: Path, s: SubjectStats, log: logging.Logger) -> None:
    traj  = subject_dir / "platform/mouse_trajectory_points.csv"
    evts  = subject_dir / "platform/all_events.csv"

    if traj.exists():
        try:
            df = pd.read_csv(traj, usecols=["wall_time_ms"])
            s.mouse_n_traj_points = len(df)
            wt = df["wall_time_ms"].dropna()
            if len(wt) > 1:
                s.mouse_t0_ms     = float(wt.min())
                s.mouse_t1_ms     = float(wt.max())
                s.mouse_duration_s = round((wt.max() - wt.min()) / 1000, 2)
        except Exception as e:
            log.error(f"  1.2-Mouse  traj error: {e}")

    if evts.exists():
        try:
            s.mouse_n_events = max(0, sum(1 for _ in open(evts, encoding="utf-8")) - 1)
        except Exception:
            pass

    log.info(f"  1.2-Mouse  {s.mouse_n_traj_points:,} traj pts  "
             f"{s.mouse_n_events:,} events  {s.mouse_duration_s:.1f}s")


def compute_marker_stats(subject_dir: Path, s: SubjectStats, log: logging.Logger) -> None:
    p = subject_dir / "platform/scenario_triggers.csv"
    if not p.exists():
        return
    try:
        df = pd.read_csv(p)
        s.marker_total = len(df)
        counts: dict = {}
        for _, row in df.iterrows():
            try:
                data  = json.loads(row["event_data"].replace('""', '"'))
                stype = data["details"]["type"]
                counts[stype] = counts.get(stype, 0) + 1
            except (json.JSONDecodeError, KeyError):
                pass
        s.marker_by_type          = counts
        s.n_unique_scenario_types = len(counts)
        top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
        log.info(f"  1.2-Markers  {s.marker_total} triggers  {s.n_unique_scenario_types} types  "
                 f"top3={top3}")
    except Exception as e:
        log.error(f"  1.2-Markers  error: {e}")


# ─── 1.3: Temporal alignment ─────────────────────────────
# Strategy:
#   • Eye tracker and EEG both start within seconds of each other → use eye window
#     as the canonical "session window".
#   • Mouse trajectory data often contains pre-session activity (user logged into
#     the web platform days before the lab visit). We therefore filter mouse
#     timestamps to within the eye-tracker window (± 10 min buffer) before
#     computing overlap, to avoid the pre-session history from diluting the metric.

def check_temporal_alignment(
    s: SubjectStats, subject_dir: Path, cfg: dict, log: logging.Logger
) -> None:
    # Eye window must exist to anchor the session
    if s.eye_t0_ms is None or s.eye_t1_ms is None:
        log.warning("  1.3  No eye timestamps → alignment check skipped")
        return

    # EEG markers must also exist
    if s.eeg_t0_ms is None or s.eeg_t1_ms is None:
        log.warning("  1.3  No EEG marker timestamps → alignment check skipped")
        return

    eye_dur_ms = s.eye_t1_ms - s.eye_t0_ms
    if eye_dur_ms <= 0:
        log.warning("  1.3  Eye duration zero → skipped")
        return

    # Buffer: 10 minutes either side of eye window
    buf_ms       = 600_000
    session_t0   = s.eye_t0_ms - buf_ms
    session_t1   = s.eye_t1_ms + buf_ms

    # ── Filter mouse timestamps to session window ──────────
    mouse_t0_sess: Optional[float] = None
    mouse_t1_sess: Optional[float] = None
    traj = subject_dir / "platform/mouse_trajectory_points.csv"
    if traj.exists():
        try:
            df = pd.read_csv(traj, usecols=["wall_time_ms"])
            wt = df["wall_time_ms"].dropna()
            wt_sess = wt[(wt >= session_t0) & (wt <= session_t1)]
            if len(wt_sess) > 0:
                mouse_t0_sess = float(wt_sess.min())
                mouse_t1_sess = float(wt_sess.max())
        except Exception as e:
            log.error(f"  1.3  Mouse filter error: {e}")

    # ── EEG markers window ─────────────────────────────────
    # Extend EEG marker window to full recording duration (file-size derived).
    # The marker CSV only contains discrete events; fill from first marker to
    # first marker + EEG duration so the window represents the full recording.
    eeg_t0 = s.eeg_t0_ms
    eeg_t1 = s.eeg_t0_ms + s.eeg_duration_s * 1000 if s.eeg_duration_s > 0 else s.eeg_t1_ms

    # ── Triple overlap: fraction of EEG window covered by all three ──────
    # Denominator = EEG duration (primary modality, ~16 min recording).
    # Eye may run longer after the EEG ends (calibration tail); mouse may have
    # gaps. We clip all modalities to the EEG window and measure coverage.
    eeg_dur_ms = eeg_t1 - eeg_t0
    if eeg_dur_ms <= 0:
        log.warning("  1.3  EEG duration zero → skipped")
        return

    # Clip each modality to the EEG window
    def window_in_eeg(t0_ms, t1_ms):
        clipped_start = max(t0_ms, eeg_t0)
        clipped_end   = min(t1_ms, eeg_t1)
        return max(0.0, clipped_end - clipped_start)

    eye_in_eeg   = window_in_eeg(s.eye_t0_ms, s.eye_t1_ms)
    mouse_in_eeg = window_in_eeg(mouse_t0_sess, mouse_t1_sess) if mouse_t0_sess else 0.0

    # Triple overlap = region where all three overlap within the EEG window
    all_t0s = [eeg_t0, s.eye_t0_ms, mouse_t0_sess if mouse_t0_sess else eeg_t0]
    all_t1s = [eeg_t1, s.eye_t1_ms, mouse_t1_sess if mouse_t1_sess else eeg_t1]
    triple_start = max(all_t0s)
    triple_end   = min(all_t1s)
    triple_dur   = max(0.0, triple_end - triple_start)
    pct          = triple_dur / eeg_dur_ms * 100

    # Individual coverages
    eye_cov_pct   = eye_in_eeg   / eeg_dur_ms * 100
    mouse_cov_pct = mouse_in_eeg / eeg_dur_ms * 100
    eeg_eye_drift = (eeg_t0 - s.eye_t0_ms) / 1000  # seconds

    s.eeg_eye_drift_s     = round(eeg_eye_drift, 1)
    s.eeg_eye_overlap_pct = round(eye_cov_pct, 1)
    s.mouse_coverage_pct  = round(mouse_cov_pct, 1)
    s.triple_overlap_pct  = round(pct, 1)

    # 1.3 flag: triple < 90%
    triple_threshold = cfg["exclusion"]["min_triple_overlap_pct"]
    s.alignment_flag = pct < triple_threshold

    flag_str = " ⚠ FLAGGED" if s.alignment_flag else ""
    log.info(f"  1.3  EEG-Eye {eye_cov_pct:.1f}%  Mouse {mouse_cov_pct:.1f}%  "
             f"Triple {pct:.1f}%{flag_str}  (drift={eeg_eye_drift:+.1f}s)")
    log.debug(f"       EEG   [{eeg_t0:.0f}  –  {eeg_t1:.0f}]  dur={eeg_dur_ms/1000:.0f}s")
    log.debug(f"       Eye   [{s.eye_t0_ms:.0f}  –  {s.eye_t1_ms:.0f}]")
    if mouse_t0_sess:
        log.debug(f"       Mouse [{mouse_t0_sess:.0f}  –  {mouse_t1_sess:.0f}] (session-filtered)")


# ─── 1.5: Exclusion criteria ─────────────────────────────

def apply_exclusion(s: SubjectStats, cfg: dict, log: logging.Logger) -> None:
    exc     = cfg["exclusion"]
    reasons = []

    # 1.5: EEG–Eye pairwise overlap < 95%
    if not np.isnan(s.eeg_eye_overlap_pct) and s.eeg_eye_overlap_pct < exc["min_eeg_eye_overlap_pct"]:
        reasons.append(
            f"EEG-Eye overlap {s.eeg_eye_overlap_pct:.1f}% < {exc['min_eeg_eye_overlap_pct']}%"
        )

    # 1.5: BPOGV eye validity < 70%
    if not np.isnan(s.eye_bpogv_pct) and s.eye_bpogv_pct < exc["min_eye_validity_pct"]:
        reasons.append(
            f"eye validity {s.eye_bpogv_pct:.1f}% < {exc['min_eye_validity_pct']}%"
        )

    # 1.5: fewer than min unique scenario types
    if s.n_unique_scenario_types < exc["min_scenarios_triggered"]:
        reasons.append(
            f"scenario types {s.n_unique_scenario_types} < {exc['min_scenarios_triggered']}"
        )

    s.exclude         = bool(reasons)
    s.exclude_reasons = reasons

    verdict = "EXCLUDE - " + " | ".join(reasons) if reasons else "INCLUDE"
    log.info(f"  1.5  {verdict}")


# ─── 1.4: Cross-subject summary ──────────────────────────

def save_summary(all_stats: list, cfg: dict, log: logging.Logger) -> None:
    reports_dir = ROOT / cfg["paths"]["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in all_stats:
        rows.append({
            "subject_id":               s.subject_id,
            "name":                     s.name,
            "group":                    s.group,
            "eeg_triple_ok":            s.eeg_triple_ok,
            "n_missing_files":          len(s.missing_files),
            "n_empty_files":            len(s.empty_files),
            "eeg_duration_s":           s.eeg_duration_s,
            "eeg_n_samples":            s.eeg_n_samples,
            "eeg_n_channels":           s.eeg_n_channels,
            "eeg_n_markers_vmrk":       s.eeg_n_markers,
            "eye_duration_s":           s.eye_duration_s,
            "eye_n_samples":            s.eye_n_samples,
            "eye_bpogv_pct":            s.eye_bpogv_pct,
            "mouse_duration_s":         s.mouse_duration_s,
            "mouse_n_traj_points":      s.mouse_n_traj_points,
            "mouse_n_events":           s.mouse_n_events,
            "scenario_trigger_total":   s.marker_total,
            "n_unique_scenario_types":  s.n_unique_scenario_types,
            "eeg_eye_drift_s":          s.eeg_eye_drift_s,
            "eeg_eye_overlap_pct":      s.eeg_eye_overlap_pct,
            "mouse_coverage_pct":       s.mouse_coverage_pct,
            "triple_overlap_pct":       s.triple_overlap_pct,
            "alignment_flag":           s.alignment_flag,
            "exclude":                  s.exclude,
            "exclude_reasons":          "; ".join(s.exclude_reasons),
        })

    df = pd.DataFrame(rows)

    csv_path  = reports_dir / "subject_inventory.csv"
    xlsx_path = reports_dir / "subject_inventory.xlsx"
    df.to_csv(csv_path, index=False)

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Inventory")

            # Per-subject marker breakdown on a second sheet
            detail_rows = []
            for s in all_stats:
                for stype, cnt in s.marker_by_type.items():
                    detail_rows.append({
                        "subject_id": s.subject_id,
                        "name": s.name,
                        "scenario_type": stype,
                        "count": cnt,
                    })
            if detail_rows:
                pd.DataFrame(detail_rows).to_excel(
                    writer, index=False, sheet_name="Marker_Breakdown"
                )
    except Exception as e:
        log.warning(f"Excel export failed: {e}")

    # ─── Console summary ─────────────────────────────────
    sep = "=" * 110
    print(f"\n{sep}")
    print("CROSS-SUBJECT INVENTORY SUMMARY")
    print(sep)

    display_cols = [
        "subject_id", "name", "group",
        "eeg_duration_s", "eye_bpogv_pct",
        "n_unique_scenario_types",
        "eeg_eye_overlap_pct", "mouse_coverage_pct", "triple_overlap_pct",
        "exclude",
    ]
    print(df[display_cols].to_string(index=False))
    print(sep)

    excluded = df[df["exclude"]]
    if not excluded.empty:
        print(f"\nEXCLUDED SUBJECTS ({len(excluded)}/{len(df)}):")
        for _, row in excluded.iterrows():
            print(f"  sub-{row['subject_id']:02d}  {row['name']:<25}  {row['exclude_reasons']}")
    else:
        print("\nAll subjects PASS exclusion criteria.")

    # Flag worst alignment
    flagged = df[df["alignment_flag"]]
    if not flagged.empty:
        print(f"\nALIGNMENT FLAGS ({len(flagged)}):")
        for _, row in flagged.iterrows():
            print(f"  sub-{row['subject_id']:02d}  {row['name']:<25}  overlap={row['triple_overlap_pct']:.1f}%")

    print(f"\nReports saved:")
    print(f"  CSV:  {csv_path}")
    print(f"  XLSX: {xlsx_path}")


# ─── Main ────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SDP Veri Envanteri (Bölüm 0-1)")
    parser.add_argument("--id", type=int, default=None, help="Tek subject ID (örn: 14)")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])

    logs_dir = ROOT / cfg["paths"]["logs"]
    log = make_logger("inventory", logs_dir)

    log.info("=" * 60)
    log.info("SDP Pipeline - Bölüm 0+1: Hazırlık ve Envanter")
    log.info("=" * 60)
    log.info(f"seed={cfg['seed']}")

    ensure_dirs(cfg, log)

    raw_dir   = ROOT / cfg["paths"]["raw"]
    subjects  = cfg["subjects"]

    if args.id is not None:
        subjects = [s for s in subjects if s["id"] == args.id]
        if not subjects:
            log.error(f"Subject ID {args.id} not found in config.yaml")
            sys.exit(1)

    all_stats = []

    for subj in subjects:
        sid          = subj["id"]
        subject_dir  = raw_dir / subj["folder"]

        log.info(f"\n{'─' * 60}")
        log.info(f"sub-{sid:02d}  {subj['name']}  ({subj['group']})")

        if not subject_dir.exists():
            log.error(f"  Folder not found: {subject_dir}")
            continue

        s = SubjectStats(
            subject_id=sid,
            name=subj["name"],
            folder=subj["folder"],
            group=subj["group"],
        )

        check_integrity(subject_dir, s, log)
        compute_eeg_stats(subject_dir, s, log)
        compute_eye_stats(subject_dir, s, log)
        compute_mouse_stats(subject_dir, s, log)
        compute_marker_stats(subject_dir, s, log)
        check_temporal_alignment(s, subject_dir, cfg, log)
        apply_exclusion(s, cfg, log)

        all_stats.append(s)

    if all_stats:
        log.info(f"\n{'─' * 60}")
        log.info("1.4  Generating cross-subject summary...")
        save_summary(all_stats, cfg, log)

    log.info(f"\nDone. {len(all_stats)}/{len(subjects)} subjects processed.")


if __name__ == "__main__":
    main()
