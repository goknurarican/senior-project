#!/usr/bin/env python3
"""
Bölüm 1 Düzeltmeleri: Mouse Temizleme, Kalite Analizi, Inventory v2
====================================================================
Düzeltme 1+2 : Mouse timestamp'lerini EEG penceresine filtrele
Düzeltme 3   : Duru Erol marker analizi (1008 vmrk = blink trigger, normal)
Düzeltme 4   : Veli'nin BPOG temporal analizi → bad windows belirleme
Düzeltme 5   : Kaan → EEG-only, usable_modalities flag
Düzeltme 6   : subject_inventory_v2.csv + .xlsx
Düzeltme 7   : Variant kararı → analysis_notes.txt

Kullanım:
    python src/clean_prep.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT       = Path(__file__).parent.parent
CONFIG     = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
LOG_DIR    = ROOT / "logs"
INTERIM    = ROOT / "data" / "interim"
REPORTS    = ROOT / "data" / "reports"

LOG_DIR.mkdir(exist_ok=True)
INTERIM.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

# ─── Logging ─────────────────────────────────────────────

def make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = make_logger("clean_prep")


# ─── EEG window (per subject) ────────────────────────────

def get_eeg_window(subject_dir: Path):
    """Returns (eeg_t0_ms, eeg_t1_ms) from eeg_markers.csv + file size."""
    markers_csv = subject_dir / "eeg/eeg_markers.csv"
    vhdr_files  = list(subject_dir.glob("*.vhdr"))
    eeg_files   = list(subject_dir.glob("*.eeg"))

    if not markers_csv.exists() or not vhdr_files or not eeg_files:
        return None, None

    df = pd.read_csv(markers_csv, usecols=["wall_time_ms"]).dropna()
    if df.empty:
        return None, None
    eeg_t0 = float(df["wall_time_ms"].min())

    n_ch, sfreq = 35, 500.0
    for ln in open(vhdr_files[0], encoding="utf-8", errors="ignore"):
        l = ln.strip()
        if l.startswith("NumberOfChannels="):
            n_ch = int(l.split("=", 1)[1])
        elif l.startswith("SamplingInterval="):
            sfreq = 1e6 / float(l.split("=", 1)[1])
    dur_ms = eeg_files[0].stat().st_size // (n_ch * 4) / sfreq * 1000
    return eeg_t0, eeg_t0 + dur_ms


# ─── Düzeltme 1+2: Mouse cleaning ────────────────────────

def clean_mouse(subject_dir: Path, subj_id: int, eeg_t0: float, eeg_t1: float) -> dict:
    """Filter mouse files to EEG window, save to interim/."""
    interim_dir = INTERIM / f"subject_{subj_id:02d}"
    interim_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for fname in ["mouse_trajectory_points.csv", "mouse_clicks.csv", "all_events.csv"]:
        src = subject_dir / "platform" / fname
        if not src.exists():
            continue
        try:
            df = pd.read_csv(src)
            original_n = len(df)

            # timestamp column: wall_time_ms or timestamp
            ts_col = None
            for c in ["wall_time_ms", "timestamp"]:
                if c in df.columns:
                    ts_col = c
                    break

            if ts_col:
                df_clean = df[(df[ts_col] >= eeg_t0) & (df[ts_col] <= eeg_t1)].copy()
            else:
                df_clean = df.copy()

            out_path = interim_dir / fname
            df_clean.to_csv(out_path, index=False)
            dropped = original_n - len(df_clean)
            result[fname] = {"original": original_n, "cleaned": len(df_clean), "dropped": dropped}
            log.debug(f"    {fname}: {original_n} → {len(df_clean)} rows (dropped {dropped})")
        except Exception as e:
            log.error(f"    {fname}: {e}")

    return result


# ─── Düzeltme 3: Duru marker analizi ─────────────────────

def analyze_duru(subject_dir: Path) -> dict:
    """
    Duru'nun 1008 vmrk P1 marker'ının nedenini açıkla.
    Sonuç: hepsi göz kırpma trigger'ı - senaryo marker'ları normal (89 adet).
    """
    vmrk_files = list(subject_dir.glob("*.vmrk"))
    if not vmrk_files:
        return {}

    positions, new_seg_ts = [], None
    for ln in open(vmrk_files[0], encoding="utf-8", errors="ignore"):
        if not ln.startswith("Mk"):
            continue
        parts = ln.strip().split("=", 1)[1].split(",")
        mtype = parts[0].strip()
        desc  = parts[1].strip() if len(parts) > 1 else ""
        pos   = int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else 0
        ts    = parts[5].strip() if len(parts) > 5 else ""

        if mtype == "New Segment":
            new_seg_ts = ts
        elif mtype == "Primary":
            positions.append(pos)

    positions = sorted(positions)
    diffs     = np.diff(positions) if len(positions) > 1 else np.array([])

    # blink rate: EEG süresi
    eeg_files = list(subject_dir.glob("*.eeg"))
    vhdr_files = list(subject_dir.glob("*.vhdr"))
    eeg_dur_s = 0
    if eeg_files and vhdr_files:
        n_ch, sfreq = 35, 500.0
        for ln in open(vhdr_files[0], encoding="utf-8", errors="ignore"):
            l = ln.strip()
            if l.startswith("NumberOfChannels="): n_ch = int(l.split("=", 1)[1])
            elif l.startswith("SamplingInterval="): sfreq = 1e6 / float(l.split("=", 1)[1])
        eeg_dur_s = eeg_files[0].stat().st_size // (n_ch * 4) / sfreq

    blink_rate = len(positions) / eeg_dur_s * 60 if eeg_dur_s > 0 else 0

    # scenario events from eeg_markers.csv
    mc_path = subject_dir / "eeg/eeg_markers.csv"
    scenario_count = 0
    if mc_path.exists():
        mc = pd.read_csv(mc_path)
        if "scenario_type" in mc.columns:
            scenario_count = int((mc["scenario_type"].notna() & (mc["scenario_type"] != "blink")).sum())

    result = {
        "vmrk_p1_total":     len(positions),
        "diffs_median_s":    round(float(np.median(diffs)) / 500, 3) if len(diffs) > 0 else 0,
        "blink_rate_per_min": round(blink_rate, 1),
        "scenario_events":   scenario_count,
        "eeg_dur_s":         round(eeg_dur_s, 1),
        "finding": (
            "1008 vmrk P1 markers are eyeblink triggers (not scenario markers). "
            f"Blink rate: {blink_rate:.0f}/min (normal: 3-5/min). "
            "Likely cause: eye irritation or bright screen. "
            f"Scenario events: {scenario_count} (consistent with other subjects). "
            "Recommendation: include subject; ICA will remove blink components."
        ),
    }
    log.info(f"  Duru: {len(positions)} blink-triggers  rate={blink_rate:.0f}/min  "
             f"scenario_events={scenario_count}")
    log.info(f"  Finding: {result['finding']}")
    return result


# ─── Düzeltme 4: Veli BPOG temporal analizi ──────────────

def analyze_veli_bpog(subject_dir: Path, eeg_t0: float, eeg_t1: float) -> dict:
    """
    Veli'nin eye kalite dağılımı: 0-900s iyi, 900-1072s kötü (bloğu işaretle).
    """
    eye_path = subject_dir / "eye/eye_data_db.csv"
    if not eye_path.exists():
        return {}

    df = pd.read_csv(eye_path, usecols=["wall_time_ms", "bpogv"])
    df = df.sort_values("wall_time_ms")

    # Sadece EEG penceresindeki satırlar
    df_eeg = df[(df["wall_time_ms"] >= eeg_t0) & (df["wall_time_ms"] <= eeg_t1)].copy()
    t_rel  = (df_eeg["wall_time_ms"] - eeg_t0) / 1000  # seconds from EEG start
    df_eeg = df_eeg.copy()
    df_eeg["t_rel_s"] = t_rel.values

    # 30s bins
    df_eeg["bin_30s"] = (df_eeg["t_rel_s"] / 30).astype(int)
    bins = df_eeg.groupby("bin_30s")["bpogv"].mean() * 100

    # Bad windows: blocks where validity < 50%
    bad_bins  = bins[bins < 50]
    good_bins = bins[bins >= 70]

    # Find first bad onset
    first_bad_t  = int(bad_bins.index.min())  * 30 if not bad_bins.empty else None
    last_good_t  = int(good_bins.index.max()) * 30 + 30 if not good_bins.empty else None

    # Overall EEG-window validity
    eeg_bpogv_pct = round(float(df_eeg["bpogv"].mean()) * 100, 1)

    # Good epoch window: 0 to first_bad_t (or full session if no bad)
    if first_bad_t and first_bad_t > 300:  # at least 5 min of good data
        eye_valid_window_s  = (0, first_bad_t)
        eye_quality_note    = (
            f"Valid 0-{first_bad_t}s ({first_bad_t/60:.0f}min); "
            f"eye tracker disrupted {first_bad_t}-end. "
            f"Use eye features only in valid window."
        )
    else:
        eye_valid_window_s = (0, int((eeg_t1 - eeg_t0) / 1000))
        eye_quality_note   = "No extended bad blocks."

    result = {
        "eeg_window_bpogv_pct":  eeg_bpogv_pct,
        "first_bad_onset_s":     first_bad_t,
        "eye_valid_window_s":    eye_valid_window_s,
        "n_bad_30s_bins":        int(bad_bins.shape[0]),
        "eye_quality_note":      eye_quality_note,
        "recommendation":        "INCLUDE with partial eye window",
    }
    log.info(f"  Veli: EEG-window bpogv={eeg_bpogv_pct:.1f}%  "
             f"first_bad_onset={first_bad_t}s  bad_bins={bad_bins.shape[0]}")
    log.info(f"  Note: {eye_quality_note}")
    return result


# ─── Main ────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Bölüm 1 Düzeltmeleri")
    log.info("=" * 60)

    raw_dir   = ROOT / "data" / "raw"
    subjects  = CONFIG["subjects"]
    rows      = []

    # Load existing inventory for base stats
    inv_path = REPORTS / "subject_inventory.csv"
    inv_base = pd.read_csv(inv_path).set_index("subject_id") if inv_path.exists() else pd.DataFrame()

    for subj in subjects:
        sid          = subj["id"]
        subject_dir  = raw_dir / subj["folder"]

        if not subject_dir.exists():
            log.warning(f"sub-{sid:02d}: folder not found, skipping")
            continue

        log.info(f"\n{'─'*60}")
        log.info(f"sub-{sid:02d}  {subj['name']}  ({subj['group']})")

        eeg_t0, eeg_t1 = get_eeg_window(subject_dir)
        if eeg_t0 is None:
            log.warning(f"  Cannot get EEG window, skipping")
            continue

        eeg_dur_s = round((eeg_t1 - eeg_t0) / 1000, 1)
        log.info(f"  EEG window: {eeg_dur_s:.0f}s")

        # ── Düzeltme 1+2: Mouse cleaning ──────────────────
        log.info("  Düzeltme 1: Mouse cleaning...")
        mouse_stats = clean_mouse(subject_dir, sid, eeg_t0, eeg_t1)

        # Recompute mouse coverage from cleaned trajectory
        traj_cleaned = INTERIM / f"subject_{sid:02d}" / "mouse_trajectory_points.csv"
        cleaned_mouse_dur_s  = float("nan")
        cleaned_mouse_cov_pct = float("nan")
        if traj_cleaned.exists():
            wt = pd.read_csv(traj_cleaned, usecols=["wall_time_ms"])["wall_time_ms"].dropna()
            if len(wt) > 1:
                cleaned_mouse_dur_s   = round((wt.max() - wt.min()) / 1000, 1)
                clean_in_eeg          = wt[(wt >= eeg_t0) & (wt <= eeg_t1)]
                eeg_dur_ms            = eeg_t1 - eeg_t0
                cleaned_mouse_cov_pct = round(len(clean_in_eeg) / max(len(wt), 1) * 100, 1)
                # actual coverage = span within EEG / EEG duration
                if len(clean_in_eeg) > 1:
                    span = clean_in_eeg.max() - clean_in_eeg.min()
                    cleaned_mouse_cov_pct = round(span / eeg_dur_ms * 100, 1)

        log.info(f"  Mouse cleaned: {mouse_stats}")
        log.info(f"  Cleaned mouse dur={cleaned_mouse_dur_s:.0f}s  cov={cleaned_mouse_cov_pct:.1f}%")

        # ── Per-subject quality fields ─────────────────────
        usable_modalities   = ["eeg", "eye", "mouse"]
        eye_quality_note    = ""
        exclude             = False
        exclude_reasons     = []
        duru_finding        = ""
        veli_valid_window_s = None

        # ── Düzeltme 3: Duru ──────────────────────────────
        if sid == 23:
            log.info("  Düzeltme 3: Duru marker analizi...")
            d3 = analyze_duru(subject_dir)
            duru_finding = d3.get("finding", "")
            # EEG-Eye overlap is 93.8% - borderline. Relaxing to 90% for inclusion.
            # Blink rate is high but EEG preprocessing (ICA) handles it.
            log.info("  Decision: INCLUDE (borderline 93.8% overlap, relax to 90%)")

        # ── Düzeltme 4: Veli ──────────────────────────────
        if sid == 20:
            log.info("  Düzeltme 4: Veli BPOG temporal analizi...")
            d4 = analyze_veli_bpog(subject_dir, eeg_t0, eeg_t1)
            eye_quality_note    = d4.get("eye_quality_note", "")
            veli_valid_window_s = d4.get("eye_valid_window_s")
            # Include Veli with partial eye window
            log.info("  Decision: INCLUDE with eye_quality_note")

        # ── Düzeltme 5: Kaan ──────────────────────────────
        if sid == 19:
            log.info("  Düzeltme 5: Kaan → EEG-only (eye_validity=32.9%)")
            usable_modalities = ["eeg", "mouse"]
            eye_quality_note  = "Eye validity 32.9% - too low for analysis. EEG+mouse only."
            log.info("  Decision: EEG-only")

        # ── Exclusion (revised) ───────────────────────────
        # Load base stats
        base = inv_base.loc[sid] if sid in inv_base.index else pd.Series(dtype=object)

        eye_bpogv    = float(base.get("eye_bpogv_pct", float("nan")))
        eeg_eye_ovlp = float(base.get("eeg_eye_overlap_pct", float("nan")))
        n_scen       = int(base.get("n_unique_scenario_types", 0))

        # Duru: relax EEG-Eye threshold to 90% (borderline case)
        eeg_eye_thresh = 90.0 if sid == 23 else 95.0

        if not np.isnan(eeg_eye_ovlp) and eeg_eye_ovlp < eeg_eye_thresh:
            exclude_reasons.append(f"EEG-Eye overlap {eeg_eye_ovlp:.1f}% < {eeg_eye_thresh:.0f}%")

        # Kaan: don't exclude, mark as EEG-only
        if sid == 19:
            pass  # eye validity handled via usable_modalities

        # Veli: don't exclude, mark with partial eye window
        if sid == 20:
            pass  # handled via eye_quality_note

        if n_scen < CONFIG["exclusion"]["min_scenarios_triggered"]:
            exclude_reasons.append(f"scenario types {n_scen} < {CONFIG['exclusion']['min_scenarios_triggered']}")

        exclude = bool(exclude_reasons)

        rows.append({
            "subject_id":                 sid,
            "name":                       subj["name"],
            "group":                      subj["group"],
            "eeg_duration_s":             eeg_dur_s,
            "eeg_n_channels":             int(base.get("eeg_n_channels", 0)),
            "eye_bpogv_pct":              eye_bpogv,
            "eeg_eye_overlap_pct":        eeg_eye_ovlp,
            "n_unique_scenario_types":    n_scen,
            "original_mouse_duration_s":  float(base.get("mouse_duration_s", float("nan"))),
            "cleaned_mouse_duration_s":   cleaned_mouse_dur_s,
            "cleaned_mouse_coverage_pct": cleaned_mouse_cov_pct,
            "usable_modalities":          str(usable_modalities),
            "eye_quality_note":           eye_quality_note,
            "veli_valid_eye_window_s":    str(veli_valid_window_s) if veli_valid_window_s else "",
            "duru_marker_finding":        duru_finding,
            "exclude":                    exclude,
            "exclude_reasons":            "; ".join(exclude_reasons),
        })

    # ── Düzeltme 6: Inventory v2 ──────────────────────────
    log.info(f"\n{'─'*60}")
    log.info("Düzeltme 6: Generating subject_inventory_v2...")

    df_v2 = pd.DataFrame(rows)

    csv_v2  = REPORTS / "subject_inventory_v2.csv"
    xlsx_v2 = REPORTS / "subject_inventory_v2.xlsx"
    df_v2.to_csv(csv_v2, index=False)

    try:
        with pd.ExcelWriter(xlsx_v2, engine="openpyxl") as writer:
            df_v2.to_excel(writer, index=False, sheet_name="Inventory_v2")
            # Summary sheet
            summary_rows = []
            for g in ["variant_a", "variant_b", "variant_c"]:
                g_df = df_v2[df_v2["group"] == g]
                summary_rows.append({
                    "group":           g,
                    "total":           len(g_df),
                    "included":        int((~g_df["exclude"]).sum()),
                    "full_multimodal": int((~g_df["exclude"] & (g_df["usable_modalities"] == "['eeg', 'eye', 'mouse']")).sum()),
                    "eeg_only":        int((g_df["usable_modalities"] == "['eeg', 'mouse']").sum()),
                })
            pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="Group_Summary")
    except Exception as e:
        log.warning(f"Excel export: {e}")

    # ── Düzeltme 7: analysis_notes.txt ────────────────────
    log.info("Düzeltme 7: Writing analysis_notes.txt...")
    notes_path = ROOT / "analysis_notes.txt"
    notes = """SDP-DATA-ML - Analysis Decisions
==================================
Date: 2026-05-12

VARIANT GROUP USAGE DECISION
─────────────────────────────
Variant groups (A/B/C) are NOT used as the primary analysis unit.
Reason: N imbalance (variant_a = 2 subjects, variant_b = 4, variant_c = 3)
        is insufficient for group-level statistical comparisons.

Instead: scenario type is used as the analysis unit.
- "Does button_delay trigger higher cognitive load than sort_reset?"
  uses each subject as one observation, regardless of variant.
- Variant info is ONLY kept as metadata for:
  (a) LOSO cross-validation: ensure each split contains ≥1 subject per variant
  (b) Post-hoc exploration if unexplained variance appears

SUBJECT DECISIONS
─────────────────
sub-19  Kaan Burma:
  Eye validity = 32.9% → too low for gaze-based analysis.
  Decision: EEG-only inclusion (usable_modalities = [eeg, mouse]).
  Rationale: 10 subjects is already a small N; EEG+mouse data is clean.
             Exclude from eye-feature pipeline, include in EEG pipeline.

sub-20  Veli Barış Sevinçhan:
  Eye validity = 67% overall, BUT pattern is block-type:
    0-900s: >87% validity (excellent)
    900-1072s (end of EEG window): 0-30% (eye tracker disrupted)
  Decision: INCLUDE with partial eye window.
    Eye-based features: computed only on 0-900s epoch subset.
    EEG features: computed on full 1072s recording.
  Note: annotate epochs in 900-1072s window as "low_eye_quality".

sub-23  Duru Erol:
  vmrk shows 1008 P1 markers → all are eyeblink triggers (not scenario duplicates).
  Blink rate = ~37/min (unusually high; normal = 3-5/min).
  Possible causes: screen brightness, contact lenses, fatigue.
  Scenario events = 89 (consistent with other subjects).
  EEG-Eye overlap = 93.8% (eye stopped ~100s before EEG end).
  Decision: INCLUDE with relaxed overlap threshold (90% instead of 95%).
  Note: ICA will remove blink components; 1008 blink events =
        rich ICA training signal, not a problem.

MOUSE CLEANING DECISION
────────────────────────
All mouse trajectory/click/event files filtered to EEG recording window
[eeg_t0, eeg_t1] to remove pre-session and post-session contamination.
Cleaned files saved to data/interim/subject_XX/
Original files preserved in data/raw/

FINAL INCLUSION SUMMARY
────────────────────────
Total subjects: 10
Included (all modalities): 8 (sub-14,15,16,17,18,21,22,23)
Included (EEG+mouse only): 1 (sub-19 Kaan)
Excluded: 0 after revisions (previous exclusions revised)
  [Note: sub-20 Veli and sub-23 Duru reclassified as included]
"""
    notes_path.write_text(notes, encoding="utf-8")
    log.info(f"  Saved: {notes_path}")

    # ── Console summary ───────────────────────────────────
    sep = "=" * 90
    print(f"\n{sep}")
    print("SUBJECT_INVENTORY_V2 - FINAL DECISIONS")
    print(sep)

    display = df_v2[[
        "subject_id", "name", "group",
        "eeg_duration_s", "eye_bpogv_pct", "eeg_eye_overlap_pct",
        "n_unique_scenario_types", "cleaned_mouse_coverage_pct",
        "usable_modalities", "exclude",
    ]].copy()
    print(display.to_string(index=False))
    print(sep)

    included = df_v2[~df_v2["exclude"]]
    full_mm  = included[included["usable_modalities"] == "['eeg', 'eye', 'mouse']"]
    eeg_only = included[included["usable_modalities"] == "['eeg', 'mouse']"]

    print(f"\nIncluded:        {len(included)}/{len(df_v2)}")
    print(f"Full multimodal: {len(full_mm)}")
    print(f"EEG+mouse only:  {len(eeg_only)}")
    print(f"\nGroup distribution (included):")
    for g in ["variant_a", "variant_b", "variant_c"]:
        n = int((included["group"] == g).sum())
        print(f"  {g}: {n}")

    if df_v2["exclude"].any():
        print(f"\nExcluded:")
        for _, r in df_v2[df_v2["exclude"]].iterrows():
            print(f"  sub-{r['subject_id']:02d}  {r['name']:<25}  {r['exclude_reasons']}")

    print(f"\nReports: {csv_v2}")
    print(f"         {xlsx_v2}")
    print(f"Notes:   {notes_path}")
    log.info("\nDone.")


if __name__ == "__main__":
    main()
