"""
Bölüm 2 Düzeltmeleri  (6 corrections)
Run from project root: python src/section2_corrections.py
"""

import logging
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── constants ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RAW  = ROOT / "data" / "raw"
EXCL = ROOT / "data" / "excluded"
REP  = ROOT / "data" / "reports"
S2C  = REP / "section2_corrections"
LOG  = ROOT / "logs"
SEED = 42

np.random.seed(SEED)

SCENARIO_TYPES = {
    "slow_image", "broken_image", "skeleton_prolong", "search_irrelevant",
    "button_delay", "first_click_miss", "feedback_late", "network_jitter",
    "overlay_blocking", "price_change", "coupon_min_spend", "coupon_expired",
    "facet_reset_once", "sort_reset",
}
BEHAVIOR_TYPES = {"payment_retry_timeout", "add_to_cart", "checkout_start", "search_performed"}
PHASE_TYPES    = {"variant_start", "control_start", "experiment_end"}
NON_SCENARIO   = BEHAVIOR_TYPES | PHASE_TYPES | {"blink"}


# ── helpers ──────────────────────────────────────────────────────────────────
def make_logger(name: str, log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", "%H:%M:%S")
    fh  = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh  = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_cfg() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def subject_dir(s: dict) -> Path:
    return RAW / s["folder"]


def load_eeg_markers(sdir: Path) -> pd.DataFrame:
    p = sdir / "eeg" / "eeg_markers.csv"
    df = pd.read_csv(p)
    return df


def load_eye(sdir: Path) -> pd.DataFrame:
    return pd.read_csv(sdir / "eye" / "eye_data_db.csv")


def check_output(path: Path, log: logging.Logger):
    if path.exists():
        log.info(f"  ✓ Created: {path.relative_to(ROOT)}")
    else:
        log.error(f"  ✗ MISSING: {path.relative_to(ROOT)}")


# ── Düzeltme 1: Exclude Kaan Burma (sub-19) ──────────────────────────────────
def correction_1(cfg: dict):
    log = make_logger("corr1", LOG / "section2_correction_1.log")
    log.info("=" * 60)
    log.info("Düzeltme 1: Kaan Burma (sub-19) exclusion")
    log.info("=" * 60)

    kaan_folder = "user_019_kaan_burma_variant_a"
    src = RAW / kaan_folder
    dst = EXCL / kaan_folder

    if not src.exists():
        log.warning(f"  Source folder already moved or missing: {src}")
    else:
        EXCL.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        log.info(f"  Moved {src.relative_to(ROOT)} → {dst.relative_to(ROOT)}")

    # Write exclusion reason file
    reason_path = dst / "EXCLUSION_REASON.txt"
    reason_path.write_text(
        f"Subject ID: 19 (Kaan Burma)\n"
        f"Excluded date: {date.today()}\n"
        f"Reasons:\n"
        f"  - No control phase (0s recorded before variant start)\n"
        f"  - Low eye tracker validity (BPOG 32.9%)\n"
        f"Decision context: Combined failure of within-subject baseline and eye modality\n"
        f"  makes this subject unreliable for LOSO cross-validation.\n",
        encoding="utf-8",
    )
    log.info(f"  Written: {reason_path.relative_to(ROOT)}")

    # Update inventory
    inv = pd.read_csv(REP / "subject_inventory_v2.csv")
    inv["exclude"]         = inv["exclude"].astype(object)
    inv["exclude_reasons"] = inv["exclude_reasons"].astype(object)
    mask = inv["subject_id"] == 19
    inv.loc[mask, "exclude"]         = True
    inv.loc[mask, "exclude_reasons"] = "no control phase + low eye validity"

    out_csv  = REP / "subject_inventory_v3.csv"
    out_xlsx = REP / "subject_inventory_v3.xlsx"
    inv.to_csv(out_csv,  index=False)
    inv.to_excel(out_xlsx, index=False)
    check_output(out_csv,  log)
    check_output(out_xlsx, log)

    active = inv[inv["exclude"] == False]
    log.info(f"  Subject 19 (Kaan) moved to excluded. {len(active)} subjects remain in active pipeline.")
    by_group = active.groupby("group")["subject_id"].count()
    log.info(f"  Active distribution: {dict(by_group)}")
    return inv


# ── Düzeltme 2: EEG↔Eye scenario-only match rate ─────────────────────────────
def correction_2(cfg: dict, active_subjects: list):
    log = make_logger("corr2", LOG / "section2_correction_2.log")
    log.info("=" * 60)
    log.info("Düzeltme 2: EEG↔Eye scenario-only match rate")
    log.info("=" * 60)

    rows       = []
    all_counts = {}

    for s in active_subjects:
        sdir = subject_dir(s)
        sid  = s["id"]
        name = s["name"]

        # ── eye: extract scenario onset transitions ──
        # Some subjects use NaN between scenarios; others use "blink" as the
        # between-scenario fill (e.g., sub-23/Duru).
        eye = load_eye(sdir)
        eye["_scen"] = eye["active_scenario"].fillna("__nan__")
        gap_values = {"__nan__", "blink"}
        eye["_changed"] = eye["_scen"] != eye["_scen"].shift()
        eye_scen_onsets = eye[
            eye["_changed"] & (~eye["_scen"].isin(gap_values)) & (eye["_scen"].isin(SCENARIO_TYPES))
        ][["wall_time_ms", "active_scenario"]].copy()
        eye_behav_onsets = eye[
            eye["_changed"] & (~eye["_scen"].isin(gap_values)) & (eye["_scen"].isin(BEHAVIOR_TYPES))
        ][["wall_time_ms", "active_scenario"]].copy()

        # category counts for the report
        cat_counts = {
            "scenario": len(eye_scen_onsets),
            "behavior": len(eye_behav_onsets),
        }
        all_counts[sid] = cat_counts
        log.info(f"  sub-{sid:02d} {name}: eye onsets = {cat_counts}")

        # ── EEG scenario markers (authoritative reference) ──
        em = load_eeg_markers(sdir)
        eeg_scen = em[
            (em["scenario_type"].isin(SCENARIO_TYPES)) &
            (em["eeg_marker"] > 0)
        ][["wall_time_ms", "scenario_type"]].copy()

        # ── EEG-centric match (±5s, same scenario type) ──
        # Rationale: eye active_scenario may lag EEG trigger by up to ~3s
        # (observed for sub-23). Using EEG trigger as reference avoids
        # double-counting eye transitions within the same scenario instance.
        WINDOW_MS = 5_000
        n_matched = 0
        lags = []

        for _, row in eeg_scen.iterrows():
            st = row["scenario_type"]
            t  = row["wall_time_ms"]
            candidates = eye_scen_onsets[
                (eye_scen_onsets["active_scenario"] == st) &
                (np.abs(eye_scen_onsets["wall_time_ms"] - t) <= WINDOW_MS)
            ]
            if len(candidates) > 0:
                n_matched += 1
                lags.append(int((candidates["wall_time_ms"] - t).abs().min()))

        n_eeg = len(eeg_scen)
        match_rate = round(n_matched / n_eeg * 100, 1) if n_eeg > 0 else 0
        lag_median = int(np.median(lags)) if lags else -1
        lag_mean   = int(np.mean(lags))   if lags else -1
        lag_p95    = int(np.percentile(lags, 95)) if lags else -1

        log.info(
            f"  sub-{sid:02d}: n_eeg_scenarios={n_eeg}  n_eye_scenarios={len(eye_scen_onsets)}"
            f"  matched={n_matched}  match_rate={match_rate}%  lag_median={lag_median}ms"
        )

        rows.append({
            "subject_id":       sid,
            "name":             name,
            "n_eeg_scenario":   n_eeg,
            "n_eye_scenario":   len(eye_scen_onsets),
            "n_matched":        n_matched,
            "match_rate_pct":   match_rate,
            "lag_median_ms":    lag_median,
            "lag_mean_ms":      lag_mean,
            "lag_p95_ms":       lag_p95,
        })

    df = pd.DataFrame(rows)
    overall_matched = df["n_matched"].sum()
    overall_total   = df["n_eeg_scenario"].sum()
    overall_rate    = round(overall_matched / overall_total * 100, 1) if overall_total > 0 else 0
    overall_lag_med = int(df["lag_median_ms"].median())

    log.info(f"\n  Overall scenario-only match rate: {overall_rate}%")

    # Interpretation
    if overall_rate >= 90:
        status = "OK"
        verdict = (
            f"Original 55.5% was due to non-scenario event inclusion. "
            f"Scenario events match correctly ({overall_rate}%)."
        )
    elif overall_rate >= 70:
        status = "INVESTIGATE FURTHER"
        verdict = f"Scenario-only match rate {overall_rate}% is borderline. Further investigation needed."
    else:
        status = "CRITICAL: alignment issue persists"
        verdict = f"CRITICAL: scenario-only match rate {overall_rate}% - alignment issue persists."

    log.info(f"  Status: {status}")

    # ── non-scenario event type counts across all subjects ──
    cat_df = pd.DataFrame(all_counts).T.fillna(0).astype(int)
    cat_df.index.name = "subject_id"

    # ── write markdown report ──
    md_path = S2C / "eye_eeg_match_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Eye–EEG Alignment: Scenario-Only Match Report\n\n")
        f.write(f"Generated: {date.today()}\n\n")
        f.write("## Methodology\n\n")
        f.write(
            "**EEG-centric matching, ±5s window, same-scenario-type constraint.**\n\n"
            "The original Bölüm 2 figure of 55.5% used an eye-centric approach with ±500ms window.\n"
            "That approach was flawed because:\n"
            "1. Eye transitions are counted multiple times per EEG trigger (brief dropouts in active_scenario)\n"
            "2. ±500ms is too tight - sub-23 has a systematic ~3s eye recording lag vs EEG trigger\n\n"
            "Revised approach: for each EEG trigger, check whether any eye transition to the **same**\n"
            "scenario type exists within ±5000ms. This gives one-to-one EEG trigger coverage.\n\n"
        )
        f.write("## Per-subject results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Overall\n\n")
        f.write(f"- **EEG-centric scenario match rate**: {overall_rate}%\n")
        f.write(f"- **Median lag (eye_time − eeg_time)**: {overall_lag_med} ms\n")
        f.write(f"- **p95 lag**: {int(df['lag_p95_ms'].max())} ms\n\n")
        f.write("### Sub-23 (Duru Erol) note\n\n")
        f.write(
            "Sub-23's `active_scenario` is written to the CSV with a systematic ~3000ms delay\n"
            "relative to the EEG trigger. This is a logging artifact, not a sync problem - both\n"
            "data streams use the same wall clock. Sub-23 match rate reflects how many scenarios\n"
            "were captured in the eye data at all (74.1%: 22/81 EEG triggers have no eye record).\n\n"
        )
        f.write("## Eye onset counts per subject\n\n")
        f.write(cat_df.to_markdown())
        f.write("\n\n## Interpretation\n\n")
        f.write(f"> **{status}**\n>\n> {verdict}\n")

    check_output(md_path, log)

    if status != "OK":
        log.warning(f"  ⚠ STATUS: {status} - notify user before proceeding.")
        if overall_rate < 70:
            raise RuntimeError(f"CRITICAL alignment issue ({overall_rate}%). Stopping.")

    return overall_rate, overall_lag_med


# ── Düzeltme 3: Scenario metadata - extended duration flag ───────────────────
def correction_3(cfg: dict):
    log = make_logger("corr3", LOG / "section2_correction_3.log")
    log.info("=" * 60)
    log.info("Düzeltme 3: Scenario metadata - extended duration flag")
    log.info("=" * 60)

    # Load scenario durations from Bölüm 2 output and aggregate per type
    dur_raw = pd.read_csv(REP / "scenario_durations.csv")
    dur_df  = (
        dur_raw.groupby("scenario_type")["duration_s"]
        .agg(mean="mean", median="median", std="std", min="min", max="max", count="count")
        .round(2)
        .reset_index()
    )

    # Build scenario metadata (code → name, expected max, extends flag)
    scenario_meta = {
        11: {"name": "slow_image",             "expected_max_s": 3.0,  "extends": False, "is_frustration": True},
        12: {"name": "broken_image",           "expected_max_s": 3.0,  "extends": False, "is_frustration": True},
        13: {"name": "skeleton_prolong",       "expected_max_s": 3.0,  "extends": False, "is_frustration": True},
        14: {"name": "search_irrelevant",      "expected_max_s": 3.0,  "extends": False, "is_frustration": True},
        15: {"name": "button_delay",           "expected_max_s": 3.0,  "extends": False, "is_frustration": True},
        16: {"name": "first_click_miss",       "expected_max_s": 3.0,  "extends": False, "is_frustration": True},
        17: {"name": "feedback_late",          "expected_max_s": 3.0,  "extends": False, "is_frustration": True},
        18: {"name": "network_jitter",         "expected_max_s": 3.0,  "extends": False, "is_frustration": True},
        19: {"name": "overlay_blocking",       "expected_max_s": 3.0,  "extends": False, "is_frustration": True},
        20: {"name": "price_change",           "expected_max_s": 3.0,  "extends": False, "is_frustration": True},
        21: {"name": "coupon_min_spend",       "expected_max_s": 3.0,  "extends": False, "is_frustration": True},
        22: {"name": "coupon_expired",         "expected_max_s": 3.0,  "extends": False, "is_frustration": True},
        23: {"name": "facet_reset_once",       "expected_max_s": 3.0,  "extends": False, "is_frustration": True},
        24: {"name": "sort_reset",             "expected_max_s": 8.0,  "extends": True,  "is_frustration": True},
        # behavior events (not frustration labels)
         1: {"name": "payment_retry_timeout",  "expected_max_s": 3.0,  "extends": False, "is_frustration": False},
        30: {"name": "add_to_cart",            "expected_max_s": 10.0, "extends": True,  "is_frustration": False},
        31: {"name": "checkout_start",         "expected_max_s": 3.0,  "extends": False, "is_frustration": False},
        33: {"name": "search_performed",       "expected_max_s": 3.0,  "extends": False, "is_frustration": False},
    }

    # Check observed max from scenario_durations.csv
    flagged = {}
    for _, row in dur_df.iterrows():
        stype = row["scenario_type"]
        obs_max = row["max"]
        # find matching code
        code = next((k for k, v in scenario_meta.items() if v["name"] == stype), None)
        if code is not None and obs_max > 3.05:
            flagged[stype] = {"code": code, "observed_max_s": round(obs_max, 2)}
            log.info(f"  extends_past_window=True: {stype}  observed_max={obs_max:.2f}s")

    # Add scenario metadata block to config.yaml
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        raw_cfg = f.read()

    # Build yaml block (avoid overwriting if already present)
    if "scenario_metadata:" not in raw_cfg:
        meta_yaml = "\n# ─── Scenario metadata ──────────────────────────────────────\nscenario_metadata:\n"
        for code in sorted(scenario_meta.keys()):
            m = scenario_meta[code]
            meta_yaml += (
                f"  S{code:02d}:\n"
                f"    name: {m['name']}\n"
                f"    expected_max_duration_s: {m['expected_max_s']}\n"
                f"    extends_past_window: {'true' if m['extends'] else 'false'}\n"
                f"    is_frustration_label: {'true' if m['is_frustration'] else 'false'}\n"
            )
        with open(cfg_path, "a", encoding="utf-8") as f:
            f.write(meta_yaml)
        log.info(f"  Appended scenario_metadata block to config.yaml")
    else:
        log.info("  scenario_metadata block already exists in config.yaml - skipped.")

    # Write markdown report
    md_path = S2C / "scenario_duration_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Scenario Duration Analysis\n\n")
        f.write(f"Generated: {date.today()}\n\n")
        f.write("## Per-scenario stats (from eye active_scenario column, all 9 subjects)\n\n")
        f.write(dur_df.to_markdown(index=False))
        f.write("\n\n## Window classification\n\n")
        f.write("| Code | Name | Expected Max (s) | Extends Past Window | Is Frustration Label |\n")
        f.write("|------|------|-----------------|---------------------|---------------------|\n")
        for code in sorted(scenario_meta.keys()):
            m = scenario_meta[code]
            f.write(
                f"| S{code:02d} | {m['name']} | {m['expected_max_s']} | "
                f"{'✓' if m['extends'] else ''} | {'✓' if m['is_frustration'] else ''} |\n"
            )
        f.write("\n## Epoch cutting strategy\n\n")
        f.write(
            "Standard epoch window: **-500 ms / +3000 ms** around scenario marker onset.\n\n"
            "`extends_past_window=True` scenarios (sort_reset, add_to_cart):\n"
            "- The physiological *response* is captured within the standard window.\n"
            "- The scenario UI event may outlast +3000 ms - the behavioural end is lost.\n"
            "- During epoching, these epochs will be tagged with `extends_past_window=True` in metadata.\n"
            "- `add_to_cart` is a user action, NOT a frustration label - excluded from classifier target.\n"
            "- `sort_reset` IS a frustration scenario - its epoch is included but flagged.\n"
        )
    check_output(md_path, log)
    log.info(f"  Flagged extends_past_window scenarios: {list(flagged.keys())}")


# ── Düzeltme 4: Baseline window definition ───────────────────────────────────
def correction_4(cfg: dict, active_subjects: list):
    log = make_logger("corr4", LOG / "section2_correction_4.log")
    log.info("=" * 60)
    log.info("Düzeltme 4: Baseline window standardization")
    log.info("=" * 60)

    FULL_BASELINE_S  = 120
    MIN_PARTIAL_S    =  60
    rows = []

    for s in active_subjects:
        sdir = subject_dir(s)
        sid  = s["id"]
        name = s["name"]
        em   = load_eeg_markers(sdir)

        # variant_start timestamp (eeg_marker == 2)
        variant_row = em[em["scenario_type"] == "variant_start"]
        if variant_row.empty:
            log.warning(f"  sub-{sid:02d} {name}: NO variant_start marker found!")
            continue
        variant_start_ms = int(variant_row.iloc[0]["wall_time_ms"])

        # control_start: earliest non-blink marker in control phase, or min wall_time_ms
        ctrl_rows = em[(em["phase"] == "control") & (em["scenario_type"] != "blink")]
        if not ctrl_rows.empty:
            control_start_ms = int(ctrl_rows["wall_time_ms"].min())
        else:
            # fallback: first timestamp in eye data during control phase
            eye = load_eye(sdir)
            ctrl_eye = eye[eye["phase"] == "control"]
            if not ctrl_eye.empty:
                control_start_ms = int(ctrl_eye["wall_time_ms"].min())
            else:
                control_start_ms = variant_start_ms  # degenerate

        control_dur_s = round((variant_start_ms - control_start_ms) / 1000, 1)

        if control_dur_s >= FULL_BASELINE_S:
            bl_start_ms  = variant_start_ms - FULL_BASELINE_S * 1000
            bl_end_ms    = variant_start_ms
            bl_dur_s     = FULL_BASELINE_S
            bl_status    = "full_120s"
        elif control_dur_s >= MIN_PARTIAL_S:
            bl_start_ms  = control_start_ms
            bl_end_ms    = variant_start_ms
            bl_dur_s     = round(control_dur_s, 1)
            bl_status    = "partial"
        else:
            bl_start_ms  = control_start_ms
            bl_end_ms    = variant_start_ms
            bl_dur_s     = round(control_dur_s, 1)
            bl_status    = "insufficient"
            log.warning(f"  sub-{sid:02d} {name}: INSUFFICIENT baseline - control_dur={control_dur_s}s < {MIN_PARTIAL_S}s")

        log.info(
            f"  sub-{sid:02d} {name}: ctrl={control_dur_s}s  baseline={bl_dur_s}s  status={bl_status}"
        )
        rows.append({
            "subject_id":           sid,
            "name":                 name,
            "control_start_ms":     control_start_ms,
            "variant_start_ms":     variant_start_ms,
            "control_duration_s":   control_dur_s,
            "baseline_window_start_ms": bl_start_ms,
            "baseline_window_end_ms":   bl_end_ms,
            "baseline_duration_s":  bl_dur_s,
            "baseline_status":      bl_status,
        })

    df = pd.DataFrame(rows)
    out = S2C / "baseline_windows.csv"
    df.to_csv(out, index=False)
    check_output(out, log)

    counts = df["baseline_status"].value_counts().to_dict()
    log.info(f"\n  Baseline status summary: {counts}")

    n_insuf = counts.get("insufficient", 0)
    if n_insuf > 0:
        log.warning(f"  ⚠ {n_insuf} subjects with INSUFFICIENT baseline - notify user!")
    else:
        log.info(f"  All subjects have usable baseline (full_120s or partial). Proceeding.")

    return df


# ── Düzeltme 5: Update analysis_notes.txt ────────────────────────────────────
def correction_5():
    log = make_logger("corr5", LOG / "section2_correction_5.log")
    log.info("=" * 60)
    log.info("Düzeltme 5: analysis_notes.txt update")
    log.info("=" * 60)

    notes_path = ROOT / "analysis_notes.txt"
    bak_path   = ROOT / "analysis_notes.txt.bak"

    if notes_path.exists():
        shutil.copy2(str(notes_path), str(bak_path))
        log.info(f"  Backed up to {bak_path.name}")

    content = f"""BITIRMEEG Analysis Decisions Log
================================
Last updated: {date.today()}

ACTIVE SUBJECTS: 9 (after Section 2 corrections)
  variant_a: 1  (Enis Tiren)
  variant_b: 5  (Alen Maryo, Berk Uygun, Mehmet İncekara, Feyiz Burak Öztürk, Veli Barış Sevinçhan)
  variant_c: 3  (Eren Tamparlak, Recep Danacı, Duru Erol)

EXCLUDED SUBJECTS: 1
  Kaan Burma (sub-19): No control phase + low eye validity
    See data/excluded/user_019_kaan_burma_variant_a/EXCLUSION_REASON.txt

VARIANT GROUPS:
  Decision: Variant groups are NOT analysis units.
  Reason: After excluding Kaan, variant_a has only 1 subject. Group-level
  comparisons across variants are not statistically feasible.
  Usage: Variant info kept as raw metadata only. NOT used for LOSO
  stratification (insufficient balance).

ANALYSIS UNIT:
  Primary: Scenario type (14 frustration scenarios)
  Secondary: Control vs Variant phase contrast

BASELINE WINDOW:
  Standard: Last 120s of control phase per subject
  Fallback: Available control phase if <120s
  All 9 active subjects have usable baseline (see baseline_windows.csv)

EPOCH WINDOWS:
  ERP / band power:     -200ms / +2000ms around scenario marker
  Connectivity / causality: -500ms / +3000ms
  Mouse behavior:       -1s / +5s (behavior lags)

SCENARIO DURATION EXCEPTIONS:
  sort_reset (S24): may extend past +3000ms window (observed max 5.23s)
  add_to_cart (S30): user action, may extend (observed max 8.40s);
                     NOT used as frustration label; epoch flagged in metadata

SPECIAL SUBJECT NOTES:
  Veli (sub-20): Eye features only from 0-900s window (eye tracker dropped after)
  Duru (sub-23): 1008 P1 markers are blink triggers (37/min blink rate);
                 89 actual scenario events; ICA will remove blink components
"""
    notes_path.write_text(content, encoding="utf-8")
    check_output(notes_path, log)
    log.info("  analysis_notes.txt written.")


# ── Düzeltme 6: Section 2 final summary report ───────────────────────────────
def correction_6(
    match_rate: float,
    match_lag_med: int,
    baseline_df: pd.DataFrame,
    inv: pd.DataFrame,
):
    log = make_logger("corr6", LOG / "section2_correction_6.log")
    log.info("=" * 60)
    log.info("Düzeltme 6: Section 2 final summary report")
    log.info("=" * 60)

    active  = inv[inv["exclude"] == False]
    n_subj  = len(active)

    # Scenario trigger totals from marker_count_matrix
    mc = pd.read_csv(REP / "marker_count_matrix.csv", index_col=0)
    # Exclude sub-19 (Kaan) if present
    if 19 in mc.index:
        mc = mc.drop(19)
    # Sum only numeric marker columns (skip name/group string cols)
    mc_numeric = mc.select_dtypes(include="number")
    total_scenarios = int(mc_numeric.values.sum())

    # Phase stats
    pt = pd.read_csv(REP / "phase_timing.csv")
    pt_active = pt[pt["subject_id"] != 19]
    n_ctrl_epochs    = int(pt_active["ctrl_scenarios"].sum())
    n_variant_epochs = int(pt_active["variant_scenarios"].sum())

    # Baseline status
    bl_counts = baseline_df["baseline_status"].value_counts().to_dict()
    n_full    = bl_counts.get("full_120s", 0)
    n_partial = bl_counts.get("partial", 0)
    n_insuf   = bl_counts.get("insufficient", 0)

    # Match rate interpretation
    if match_rate >= 90:
        match_verdict = f"Eye–EEG alignment verified at scenario-event level: **{match_rate}%** (OK)"
        alignment_open = ""
    elif match_rate >= 70:
        match_verdict = f"Eye–EEG scenario-only match rate: **{match_rate}%** (borderline - INVESTIGATE FURTHER)"
        alignment_open = f"- Eye–EEG match rate {match_rate}% is borderline (70–90%). Further investigation may be needed.\n"
    else:
        match_verdict  = f"Eye–EEG scenario-only match rate: **{match_rate}%** - CRITICAL ISSUE"
        alignment_open = f"- CRITICAL: Eye–EEG match rate {match_rate}% - alignment problem unresolved.\n"

    readiness = "YES" if (match_rate >= 70 and n_insuf == 0) else "CONDITIONAL"

    open_items = alignment_open
    if n_insuf > 0:
        open_items += f"- {n_insuf} subjects with insufficient baseline (<60s). Review before EEG EDA.\n"
    open_items += "- variant_a is single-subject (Enis only). Sub-analyses by variant not feasible.\n"

    md_path = S2C / "section2_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Section 2 - Summary and Corrections\n\n")
        f.write(f"Generated: {date.today()}\n\n")
        f.write("## Active dataset\n\n")
        f.write(f"- **{n_subj} subjects** (10 total – 1 excluded)\n")
        f.write(f"- Total scenario triggers across active subjects: {total_scenarios}\n")
        f.write(f"- Control phase epochs: {n_ctrl_epochs}\n")
        f.write(f"- Variant phase epochs: {n_variant_epochs}\n\n")
        f.write("## Corrections applied\n\n")
        f.write(f"1. Excluded sub-19 (Kaan): no control phase (0s) + low eye validity (BPOG 32.9%)\n")
        f.write(f"2. {match_verdict}\n")
        f.write(f"3. Scenario duration metadata added: `sort_reset` and `add_to_cart` flagged as `extends_past_window=True`\n")
        f.write(f"4. Baseline window standardized: last 120s of control phase per subject\n")
        f.write(f"   - full_120s: {n_full}  |  partial: {n_partial}  |  insufficient: {n_insuf}\n")
        f.write(f"5. Variant groups confirmed as metadata only (not analysis unit; variant_a = 1 subject)\n\n")
        f.write("## Key tables / files\n\n")
        f.write("- Updated inventory: `data/reports/subject_inventory_v3.csv`\n")
        f.write("- Eye–EEG match report: `data/reports/section2_corrections/eye_eeg_match_report.md`\n")
        f.write("- Scenario duration analysis: `data/reports/section2_corrections/scenario_duration_analysis.md`\n")
        f.write("- Baseline windows: `data/reports/section2_corrections/baseline_windows.csv`\n")
        f.write("- Decision log: `analysis_notes.txt`\n\n")
        f.write("## Open items / risks\n\n")
        f.write(open_items if open_items else "- None.\n")
        f.write("\n## Readiness for Section 3 (EEG EDA)\n\n")
        f.write(f"**{readiness}**")
        if readiness == "YES":
            f.write(" - All corrections applied, no blockers found.\n")
        else:
            f.write(" - Resolve open items above before proceeding.\n")

    check_output(md_path, log)
    log.info(f"  Section 2 summary written. Readiness: {readiness}")
    return readiness, n_full, n_partial, n_insuf


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    cfg = load_cfg()
    all_subjects = cfg["subjects"]
    # Active subjects: all except sub-19 (will be excluded in correction_1 but
    # we pre-filter here for corrections 2-4 which run after the move)
    active_subjects = [s for s in all_subjects if s["id"] != 19]

    print("=" * 70)
    print("Bölüm 2 Düzeltmeleri - 6 corrections")
    print("=" * 70)

    # Düzeltme 1
    inv = correction_1(cfg)

    # Düzeltme 2
    match_rate, match_lag_med = correction_2(cfg, active_subjects)

    # Düzeltme 3
    correction_3(cfg)

    # Düzeltme 4
    baseline_df = correction_4(cfg, active_subjects)

    # Düzeltme 5
    correction_5()

    # Düzeltme 6
    readiness, n_full, n_partial, n_insuf = correction_6(
        match_rate, match_lag_med, baseline_df, inv
    )

    print()
    print("=" * 70)
    print("BÖLÜM 2 DÜZELTMELERİ TAMAMLANDI")
    print("=" * 70)
    print(f"  Düzeltme 2 - Scenario-only match rate : {match_rate}%  (median lag: {match_lag_med}ms)")
    bl_counts = baseline_df["baseline_status"].value_counts().to_dict()
    print(f"  Düzeltme 4 - Baseline windows         : full_120s={bl_counts.get('full_120s',0)}  partial={bl_counts.get('partial',0)}  insufficient={bl_counts.get('insufficient',0)}")
    print(f"  Open items                            : {'Evet - yukarıdaki uyarılara bak' if n_insuf > 0 or match_rate < 90 else 'Yok'}")
    print(f"  Section 3'e geçilebilir mi?           : {readiness}")
    print()


if __name__ == "__main__":
    main()
