"""
Bölüm 5: Eye Tracking Preprocessing
Run from project root: python src/section5_eye_preprocessing.py

NOTE - Seçenek C applied:
  GP3 HD export does NOT include pupil diameter (pupil_left/right are binary validity
  flags: 0=invalid, 1=valid). PCMP normalization and all pupil-size features are
  excluded from the pipeline. Blink events are derived from validity-flag transitions.

Pipeline per subject:
  5.1  Load + Veli-specific window
  5.2  Mask invalid gaze (bpogv=0, off-screen; pupil flags for blink detection only)
  5.3  Short-gap interpolation (gaze_x, gaze_y; ≤200ms)
  5.4  Blink event detection from validity transitions
  5.5  I-VT fixation/saccade detection
  5.6  Epoch-level eye features (ERP + Causal windows)
  5.7  QC figures
  5.8  Per-subject report
  5.9  Cross-subject summary
"""

import json
import logging
import os
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as ss

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
RAW_DIR  = ROOT / "data" / "raw"
PROC     = ROOT / "data" / "processed"
REP5     = ROOT / "data" / "reports" / "section5_eye"
BW_CSV   = ROOT / "data" / "reports" / "section2_corrections" / "baseline_windows_v2.csv"
LOG_DIR  = ROOT / "logs"
SEED     = 42
REP5.mkdir(parents=True, exist_ok=True)

EYE_SFREQ   = 62.5            # Hz - GP3 HD
EYE_DT_MS   = 1000.0 / EYE_SFREQ  # ~16ms
MAX_GAP_MS  = 200.0           # interpolate gaps ≤ this
IVT_THRESH  = 30.0            # deg/sec - I-VT saccade threshold
MIN_FIX_MS  = 100.0           # minimum fixation duration
# Screen geometry assumption (24" monitor, 60cm viewing distance, 16:9)
SCREEN_DEG_H = 48.3           # horizontal FOV in degrees
SCREEN_DEG_V = 28.1           # vertical FOV in degrees

SUBJECTS = [
    {"id": 14, "name": "Alen Maryo",          "folder": "user_014_alen_maryo_variant_b"},
    {"id": 15, "name": "Eren Tamparlak",       "folder": "user_015_eren_tamparlak_variant_c"},
    {"id": 16, "name": "Berk Uygun",           "folder": "user_016_berk_uygun_variant_b"},
    {"id": 17, "name": "Mehmet İncekara",      "folder": "user_017_mehmet_i̇ncekara_variant_b"},
    {"id": 18, "name": "Feyiz Burak Öztürk",  "folder": "user_018_feyiz_burak_öztürk_variant_b"},
    {"id": 20, "name": "Veli Barış Sevinçhan", "folder": "user_020_veli_barış_sevinçhan_variant_b",
     "eye_window_s": (0, 900)},
    {"id": 21, "name": "Enis Tiren",           "folder": "user_021_enis_tiren_variant_a"},
    {"id": 22, "name": "Recep Danacı",         "folder": "user_022_recep_danacı_variant_c"},
    {"id": 23, "name": "Duru Erol",            "folder": "user_023_duru_erol_variant_c"},
]

# EEG epoch windows (seconds, relative to scenario marker)
ERP_TMIN,    ERP_TMAX    = -0.200,  2.000
CAUSAL_TMIN, CAUSAL_TMAX = -0.500,  3.000

# Scenario codes (same as Section 4)
SCENARIO_CODES = {
    11:"slow_image",12:"broken_image",13:"skeleton_prolong",
    14:"search_irrelevant",15:"button_delay",16:"first_click_miss",
    17:"feedback_late",18:"network_jitter",19:"overlay_blocking",
    20:"price_change",21:"coupon_min_spend",22:"coupon_expired",
    23:"facet_reset_once",24:"sort_reset",
}


# ── helpers ───────────────────────────────────────────────────────────────────
def make_logger(name, log_file):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def load_baseline_windows():
    return pd.read_csv(BW_CSV)


def load_eeg_markers(folder):
    p = RAW_DIR / folder / "eeg" / "eeg_markers.csv"
    em = pd.read_csv(p)
    em.columns = em.columns.str.strip().str.lower()
    return em


# ── 5.1 Load ──────────────────────────────────────────────────────────────────
def step_load(s, log):
    eye_path = RAW_DIR / s["folder"] / "eye" / "eye_data_db.csv"
    df = pd.read_csv(eye_path)
    df.columns = df.columns.str.strip().str.lower()
    n_total = len(df)

    # Validate expected columns
    required = ["wall_time_ms", "gaze_x", "gaze_y", "pupil_left", "pupil_right",
                "bpogv", "fpogv"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    log.info(f"  5.1 Loaded: {n_total} rows")

    # Veli-specific window
    window_applied = False
    if "eye_window_s" in s:
        t0 = df["wall_time_ms"].min()
        w0, w1 = s["eye_window_s"]
        df = df[(df["wall_time_ms"] >= t0 + w0 * 1000) &
                (df["wall_time_ms"] <= t0 + w1 * 1000)].copy()
        log.info(f"  5.1 Window [{w0}-{w1}s] applied: {len(df)}/{n_total} rows kept")
        window_applied = True

    # Sort by time, then deduplicate rows with identical timestamps
    df = df.sort_values("wall_time_ms").reset_index(drop=True)
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset="wall_time_ms").reset_index(drop=True)
    n_dupes = n_before_dedup - len(df)
    if n_dupes > 0:
        pct = n_dupes / n_before_dedup * 100
        log.warning(f"  5.1 Dropped {n_dupes} duplicate timestamps ({pct:.1f}%) - "
                    f"effective sfreq ~{len(df)/((df.wall_time_ms.max()-df.wall_time_ms.min())/1000):.1f} Hz")

    # Sample rate check (after dedup)
    dt_median = df["wall_time_ms"].diff().median()
    sfreq_est = 1000.0 / dt_median if dt_median > 0 else 0
    log.info(f"       est. {sfreq_est:.1f} Hz (median Δt={dt_median:.1f}ms) - "
             f"Duration: {(df.wall_time_ms.max()-df.wall_time_ms.min())/1000:.1f}s")
    if abs(sfreq_est - EYE_SFREQ) > 5:
        log.warning(f"       ⚠ Sample rate {sfreq_est:.1f} Hz differs from expected {EYE_SFREQ} Hz")
    return df, {"n_total": n_total, "window_applied": window_applied,
                "n_after_window": len(df), "n_dupes_dropped": n_dupes}


# ── 5.2 Mask invalid gaze ─────────────────────────────────────────────────────
def step_mask(df, log):
    stats = {}
    gaze_cols = ["gaze_x", "gaze_y"]

    # bpogv == 0
    bpogv_mask = df["bpogv"] == 0
    n_bpogv = int(bpogv_mask.sum())
    df.loc[bpogv_mask, gaze_cols] = np.nan
    stats["bpogv_invalid"] = n_bpogv
    log.info(f"  5.2 Masked bpogv=0: {n_bpogv} rows ({n_bpogv/len(df)*100:.1f}%)")

    # Off-screen gaze
    off_x = (df["gaze_x"] < 0) | (df["gaze_x"] > 1)
    off_y = (df["gaze_y"] < 0) | (df["gaze_y"] > 1)
    off_mask = off_x | off_y
    n_off = int(off_mask.sum())
    df.loc[off_mask, gaze_cols] = np.nan
    stats["offscreen"] = n_off
    log.info(f"  5.2 Masked off-screen: {n_off} rows ({n_off/len(df)*100:.1f}%)")

    # Pupil validity flags are NOT used for masking: GP3 HD can track gaze via
    # corneal reflection even when pupil detection fails (flags add false positives).
    # Pupil flags are used only in step_detect_blinks for blink event detection.
    n_pup = int(((df["pupil_left"] == 0) | (df["pupil_right"] == 0)).sum())
    stats["pupil_invalid"] = n_pup  # logged but not applied as gaze mask

    total_masked = int(df["gaze_x"].isna().sum())
    stats["total_nan_after_mask"] = total_masked
    stats["valid_pct"] = round((len(df) - total_masked) / len(df) * 100, 1)
    log.info(f"  5.2 Total NaN after masking: {total_masked} ({total_masked/len(df)*100:.1f}%) "
             f"→ valid: {stats['valid_pct']}%")
    return df, stats


# ── 5.3 Short-gap interpolation ───────────────────────────────────────────────
def step_interpolate(df, log):
    interp_stats = {"interpolated_gaps": 0, "long_gaps": 0}
    for col in ["gaze_x", "gaze_y"]:
        s = df[col].copy()
        nan_mask = s.isna()
        if not nan_mask.any():
            continue

        # Find contiguous NaN runs
        changes = nan_mask.astype(int).diff().fillna(0)
        starts  = df.index[changes == 1].tolist()
        ends    = df.index[changes == -1].tolist()
        # Edge cases
        if nan_mask.iloc[0]:  starts = [df.index[0]] + starts
        if nan_mask.iloc[-1]: ends.append(df.index[-1] + 1)

        n_interp = 0; n_long = 0
        for st, en in zip(starts, ends):
            t_start = df.loc[st, "wall_time_ms"]
            t_end   = df.loc[en - 1, "wall_time_ms"] if en - 1 in df.index else t_start
            gap_ms  = t_end - t_start
            if gap_ms <= MAX_GAP_MS:
                n_interp += 1
            else:
                n_long += 1

        # Interpolate all at once; then re-mask long gaps
        s_interp = s.interpolate(method="linear", limit_direction="both",
                                  limit=int(MAX_GAP_MS / EYE_DT_MS + 1))
        # Re-apply NaN for long gaps
        for st, en in zip(starts, ends):
            t_start = df.loc[st, "wall_time_ms"]
            t_end   = df.loc[en - 1, "wall_time_ms"] if en - 1 in df.index else t_start
            if t_end - t_start > MAX_GAP_MS:
                s_interp.iloc[st:en] = np.nan

        df[col] = s_interp
        interp_stats["interpolated_gaps"] += n_interp
        interp_stats["long_gaps"] += n_long

    log.info(f"  5.3 Interpolation: {interp_stats['interpolated_gaps']} short gaps filled, "
             f"{interp_stats['long_gaps']} long gaps left as NaN")
    return df, interp_stats


# ── 5.4 Blink event detection ─────────────────────────────────────────────────
def step_detect_blinks(df, log):
    """Detect blinks from validity-flag transitions (Seçenek C)."""
    # Invalid = bpogv==0 OR either pupil invalid
    invalid = (df["bpogv"] == 0) | (df["pupil_left"] == 0) | (df["pupil_right"] == 0)
    df["blink_flag"] = invalid.astype(int)

    # Find contiguous blink periods
    changes  = invalid.astype(int).diff().fillna(0)
    b_starts = df.index[changes == 1].tolist()
    b_ends   = df.index[changes == -1].tolist()
    if invalid.iloc[0]:  b_starts = [df.index[0]] + b_starts
    if invalid.iloc[-1]: b_ends.append(df.index[-1] + 1)

    blink_events = []
    for st, en in zip(b_starts, b_ends):
        t_s = df.loc[st, "wall_time_ms"]
        t_e = df.loc[min(en, df.index[-1]), "wall_time_ms"]
        dur = t_e - t_s
        if dur >= 50:  # ignore <50ms transients
            blink_events.append({"start_ms": t_s, "end_ms": t_e, "duration_ms": dur})

    blink_df = pd.DataFrame(blink_events)
    n_blinks = len(blink_df)
    total_dur_s = (df["wall_time_ms"].max() - df["wall_time_ms"].min()) / 1000
    blink_rate  = n_blinks / total_dur_s * 60 if total_dur_s > 0 else 0
    log.info(f"  5.4 Blink events: {n_blinks}  rate: {blink_rate:.1f}/min")
    return df, blink_df, {"n_blinks": n_blinks, "blink_rate_per_min": round(blink_rate, 1)}


# ── 5.5 I-VT fixation / saccade detection ─────────────────────────────────────
def step_ivt(df, log):
    t   = df["wall_time_ms"].values / 1000.0  # seconds
    gx  = df["gaze_x"].values * SCREEN_DEG_H  # to degrees
    gy  = df["gaze_y"].values * SCREEN_DEG_V

    # Angular velocity
    dt  = np.diff(t)
    dgx = np.diff(gx)
    dgy = np.diff(gy)
    vel = np.sqrt(dgx**2 + dgy**2) / np.where(dt > 0, dt, np.nan)  # deg/sec

    # Pad to same length as df
    vel = np.concatenate([[np.nan], vel])

    # Classification: NaN gaze → unknown; vel > threshold → saccade; else → fixation
    label = np.full(len(df), "unknown", dtype=object)
    valid = ~df["gaze_x"].isna().values
    label[valid & (vel <= IVT_THRESH)] = "fixation"
    label[valid & (vel >  IVT_THRESH)] = "saccade"
    df["ivt_label"] = label

    # Build fixation events (merge consecutive fixation samples)
    fix_events  = []
    sacc_events = []

    def _merge_events(condition_label, min_dur_ms):
        events = []
        in_event = False
        e_start = None; e_indices = []
        for i, lbl in enumerate(label):
            if lbl == condition_label:
                if not in_event:
                    in_event = True; e_start = i; e_indices = [i]
                else:
                    e_indices.append(i)
            else:
                if in_event:
                    dur = df["wall_time_ms"].iloc[e_indices[-1]] - df["wall_time_ms"].iloc[e_start]
                    if dur >= min_dur_ms:
                        events.append(e_indices)
                    in_event = False; e_indices = []
        if in_event and e_indices:
            dur = df["wall_time_ms"].iloc[e_indices[-1]] - df["wall_time_ms"].iloc[e_start]
            if dur >= min_dur_ms:
                events.append(e_indices)
        return events

    fix_groups  = _merge_events("fixation", MIN_FIX_MS)
    sacc_groups = _merge_events("saccade", 0)

    for grp in fix_groups:
        rows = df.iloc[grp]
        fix_events.append({
            "start_ms":    rows["wall_time_ms"].iloc[0],
            "end_ms":      rows["wall_time_ms"].iloc[-1],
            "duration_ms": rows["wall_time_ms"].iloc[-1] - rows["wall_time_ms"].iloc[0],
            "mean_gaze_x": rows["gaze_x"].mean(),
            "mean_gaze_y": rows["gaze_y"].mean(),
        })

    for grp in sacc_groups:
        rows = df.iloc[grp]
        dx   = (rows["gaze_x"].iloc[-1] - rows["gaze_x"].iloc[0]) * SCREEN_DEG_H
        dy   = (rows["gaze_y"].iloc[-1] - rows["gaze_y"].iloc[0]) * SCREEN_DEG_V
        amp  = np.sqrt(dx**2 + dy**2)
        vels = vel[grp]
        peak_v = float(np.nanmax(vels)) if len(vels) > 0 else 0.0
        sacc_events.append({
            "start_ms":      rows["wall_time_ms"].iloc[0],
            "end_ms":        rows["wall_time_ms"].iloc[-1],
            "duration_ms":   rows["wall_time_ms"].iloc[-1] - rows["wall_time_ms"].iloc[0],
            "amplitude_deg": round(amp, 3),
            "peak_vel_dps":  round(peak_v, 1),
        })

    fix_df  = pd.DataFrame(fix_events)
    sacc_df = pd.DataFrame(sacc_events)

    n_fix  = len(fix_df)
    n_sacc = len(sacc_df)
    mean_fix_dur  = fix_df["duration_ms"].mean()  if n_fix  > 0 else 0
    mean_sacc_amp = sacc_df["amplitude_deg"].mean() if n_sacc > 0 else 0

    log.info(f"  5.5 I-VT: fixations={n_fix}  mean_dur={mean_fix_dur:.0f}ms  "
             f"saccades={n_sacc}  mean_amp={mean_sacc_amp:.2f}°")
    return df, fix_df, sacc_df, {
        "n_fixations": n_fix, "mean_fixation_ms": round(mean_fix_dur, 1),
        "n_saccades": n_sacc, "mean_saccade_amp_deg": round(mean_sacc_amp, 3),
    }


# ── 5.6 Epoch-level features ──────────────────────────────────────────────────
def step_epoch_features(df, blink_df, fix_df, sacc_df, s, log):
    em   = load_eeg_markers(s["folder"])
    scen = em[em["eeg_marker"].isin(SCENARIO_CODES.keys())].copy()
    if scen.empty:
        log.warning("  5.6 No scenario events found in eeg_markers!")
        return pd.DataFrame(), pd.DataFrame(), np.array([])

    # EEG t0 alignment: first blink in eeg_markers → first annotation in EEG
    # For eye alignment: use wall_time_ms directly (both in ms epoch time)
    def window_features(t_ms, tmin_s, tmax_s):
        w_start = t_ms + tmin_s * 1000
        w_end   = t_ms + tmax_s * 1000
        win     = df[(df["wall_time_ms"] >= w_start) & (df["wall_time_ms"] < w_end)]
        if len(win) == 0:
            return None

        nan_ratio    = float(win["gaze_x"].isna().mean())
        gaze_disp_x  = float(win["gaze_x"].std(skipna=True)) if win["gaze_x"].notna().sum() > 1 else 0.0
        gaze_disp_y  = float(win["gaze_y"].std(skipna=True)) if win["gaze_y"].notna().sum() > 1 else 0.0
        gaze_disp    = np.sqrt(gaze_disp_x**2 + gaze_disp_y**2)

        # Fixations in window
        if not fix_df.empty:
            fix_in = fix_df[(fix_df["start_ms"] >= w_start) & (fix_df["end_ms"] <= w_end)]
            fix_count    = len(fix_in)
            fix_mean_dur = float(fix_in["duration_ms"].mean()) if fix_count > 0 else np.nan
        else:
            fix_count = 0; fix_mean_dur = np.nan

        # Saccades in window
        if not sacc_df.empty:
            sacc_in = sacc_df[(sacc_df["start_ms"] >= w_start) & (sacc_df["end_ms"] <= w_end)]
            sacc_count    = len(sacc_in)
            sacc_mean_amp = float(sacc_in["amplitude_deg"].mean()) if sacc_count > 0 else np.nan
        else:
            sacc_count = 0; sacc_mean_amp = np.nan

        # Blinks in window
        if not blink_df.empty:
            blink_in = blink_df[(blink_df["start_ms"] >= w_start) & (blink_df["end_ms"] <= w_end)]
            blink_count = len(blink_in)
        else:
            blink_count = 0

        return {
            "fixation_count":       fix_count,
            "fixation_mean_dur_ms": round(fix_mean_dur, 1) if not np.isnan(fix_mean_dur) else np.nan,
            "saccade_count":        sacc_count,
            "saccade_mean_amp_deg": round(sacc_mean_amp, 3) if not np.isnan(sacc_mean_amp) else np.nan,
            "blink_count":          blink_count,
            "gaze_dispersion":      round(gaze_disp, 4),
            "nan_ratio":            round(nan_ratio, 3),
            "n_samples":            len(win),
        }

    erp_rows    = []
    causal_rows = []
    causal_series = []  # for .npy output

    for _, row in scen.iterrows():
        t_ms    = float(row["wall_time_ms"])
        sc_code = int(row["eeg_marker"])
        sc_name = SCENARIO_CODES.get(sc_code, str(sc_code))

        base = {"scenario_type": sc_name, "eeg_marker": sc_code,
                "wall_time_ms": t_ms}

        # ERP window
        feats = window_features(t_ms, ERP_TMIN, ERP_TMAX)
        if feats:
            erp_rows.append({**base, **feats})

        # Causal window
        feats_c = window_features(t_ms, CAUSAL_TMIN, CAUSAL_TMAX)
        if feats_c:
            causal_rows.append({**base, **feats_c})
            # 50 Hz resample of gaze validity for time series output
            w_start = t_ms + CAUSAL_TMIN * 1000
            w_end   = t_ms + CAUSAL_TMAX * 1000
            win_c   = df[(df["wall_time_ms"] >= w_start) & (df["wall_time_ms"] < w_end)]
            n_target = int((CAUSAL_TMAX - CAUSAL_TMIN) * 50)  # 50 Hz target
            ts = np.full(n_target, np.nan)
            if len(win_c) > 1:
                # resample gaze_x (NaN where invalid) to 50 Hz grid
                t_orig   = win_c["wall_time_ms"].values - w_start
                gx_orig  = win_c["gaze_x"].values
                t_target = np.linspace(0, (CAUSAL_TMAX - CAUSAL_TMIN) * 1000, n_target)
                valid_mask = ~np.isnan(gx_orig)
                if valid_mask.sum() >= 2:
                    ts = np.interp(t_target, t_orig[valid_mask], gx_orig[valid_mask])
            causal_series.append(ts)

    erp_df    = pd.DataFrame(erp_rows)
    causal_df = pd.DataFrame(causal_rows)
    ts_array  = np.array(causal_series) if causal_series else np.array([])

    n_erp = len(erp_df); n_caus = len(causal_df)
    nan_erp  = erp_df["nan_ratio"].mean()   if n_erp  > 0 else 0
    nan_caus = causal_df["nan_ratio"].mean() if n_caus > 0 else 0
    high_nan_erp = (erp_df["nan_ratio"] > 0.5).sum() if n_erp > 0 else 0

    log.info(f"  5.6 ERP epochs: {n_erp}  avg NaN={nan_erp:.2f}  high_nan(>50%): {high_nan_erp}")
    log.info(f"       Causal epochs: {n_caus}  avg NaN={nan_caus:.2f}")
    return erp_df, causal_df, ts_array, {
        "n_erp": n_erp, "n_causal": n_caus,
        "mean_nan_erp": round(float(nan_erp), 3),
        "mean_nan_causal": round(float(nan_caus), 3),
        "high_nan_epochs": int(high_nan_erp),
    }


# ── 5.7 QC figures ────────────────────────────────────────────────────────────
def step_qc_figures(df, blink_df, fix_df, s, bw_row, subj_out, log):
    sid  = s["id"]
    name = s["name"]

    t0_ms  = df["wall_time_ms"].min()
    t_sec  = (df["wall_time_ms"] - t0_ms) / 1000.0
    vs_ms  = float(bw_row["variant_start_ms"])
    vs_sec = (vs_ms - t0_ms) / 1000.0

    # 1. Gaze heatmap (control vs variant)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (label_p, t_min, t_max) in zip(axes, [
            ("Control", 0, vs_sec),
            ("Variant", vs_sec, t_sec.max()),
        ]):
            mask = (t_sec >= t_min) & (t_sec < t_max) & df["gaze_x"].notna()
            gx = df.loc[mask, "gaze_x"].values
            gy = df.loc[mask, "gaze_y"].values
            if len(gx) > 10:
                h, xedges, yedges = np.histogram2d(gx, gy, bins=50,
                                                    range=[[0,1],[0,1]])
                ax.imshow(h.T, origin="lower", aspect="auto", cmap="hot",
                           extent=[0,1,0,1])
            ax.set_title(f"{name} - {label_p}")
            ax.set_xlabel("Gaze X"); ax.set_ylabel("Gaze Y")
        fig.suptitle("Gaze Heatmap")
        fig.tight_layout()
        savefig(fig, subj_out / "qc_gaze_heatmap.png")
    except Exception as e:
        log.warning(f"       Heatmap failed: {e}")

    # 2. Blink rate timeline (per-minute bins)
    try:
        total_s = t_sec.max()
        bins    = np.arange(0, total_s + 60, 60)
        blink_rates = []
        for i in range(len(bins) - 1):
            b_start = t0_ms + bins[i] * 1000
            b_end   = t0_ms + bins[i+1] * 1000
            if not blink_df.empty:
                n = ((blink_df["start_ms"] >= b_start) & (blink_df["start_ms"] < b_end)).sum()
            else:
                n = 0
            blink_rates.append(n)
        fig, ax = plt.subplots(figsize=(12, 3))
        bin_mids = (bins[:-1] + bins[1:]) / 2
        ax.bar(bin_mids / 60, blink_rates, width=0.8, color="steelblue", alpha=0.8)
        ax.axvline(vs_sec / 60, color="red", ls="--", label="Variant start")
        ax.set_xlabel("Time (min)"); ax.set_ylabel("Blinks/min")
        ax.set_title(f"{name} - Blink rate timeline")
        ax.legend(fontsize=8)
        fig.tight_layout()
        savefig(fig, subj_out / "qc_blink_rate_timeline.png")
    except Exception as e:
        log.warning(f"       Blink rate timeline failed: {e}")

    # 3. Fixation duration histogram
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        if not fix_df.empty:
            ax.hist(fix_df["duration_ms"], bins=50, color="steelblue", alpha=0.8, edgecolor="white")
            ax.axvline(fix_df["duration_ms"].median(), color="red", ls="--",
                       label=f"Median={fix_df['duration_ms'].median():.0f}ms")
            ax.legend(fontsize=8)
        ax.set_xlabel("Fixation duration (ms)"); ax.set_ylabel("Count")
        ax.set_title(f"{name} - Fixation duration histogram")
        fig.tight_layout()
        savefig(fig, subj_out / "qc_fixation_duration_hist.png")
    except Exception as e:
        log.warning(f"       Fixation histogram failed: {e}")

    # 4. Gaze validity timeline (replaces PCMP - shows blink_flag over time)
    try:
        fig, ax = plt.subplots(figsize=(14, 3))
        valid_pct = (~df["gaze_x"].isna()).rolling(int(EYE_SFREQ * 10), min_periods=1).mean() * 100
        ax.fill_between(t_sec, valid_pct, color="steelblue", alpha=0.6, label="Valid gaze %")
        ax.axvline(vs_sec, color="red", ls="--", lw=1.5, label="Variant start")
        ax.set_ylim(0, 105)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Valid gaze % (10s window)")
        ax.set_title(f"{name} - Gaze validity timeline")
        ax.legend(fontsize=8)
        fig.tight_layout()
        savefig(fig, subj_out / "qc_gaze_validity_timeline.png")
    except Exception as e:
        log.warning(f"       Validity timeline failed: {e}")

    log.info("  5.7 QC figures saved")


# ── 5.8 Per-subject report ────────────────────────────────────────────────────
def write_subject_report(s, subj_out, load_stats, mask_stats, interp_stats,
                          blink_stats, ivt_stats, epoch_stats, bw_row, log):
    sid  = s["id"]
    name = s["name"]

    # EEG epoch count for alignment check
    erp_fif = PROC / f"subject_{sid:02d}" / "epochs_erp-epo.fif"
    eeg_erp_count = "N/A"
    try:
        import mne
        mne.set_log_level("ERROR")
        ep = mne.read_epochs(str(erp_fif), preload=False, verbose=False)
        eeg_erp_count = len(ep)
    except Exception:
        pass

    eye_erp = epoch_stats.get("n_erp", 0)
    match   = "yes" if eeg_erp_count == eye_erp else f"no (EEG={eeg_erp_count}, eye={eye_erp})"

    bw_start = pd.Timestamp(int(bw_row["baseline_window_start_ms"]), unit="ms")
    bw_end   = pd.Timestamp(int(bw_row["baseline_window_end_ms"]),   unit="ms")

    md = [
        f"# Eye Preprocessing Report - Subject {sid:02d} ({name})",
        f"",
        f"Generated: {date.today()}",
        f"",
        f"## Pupil Data Note",
        f"GP3 HD export contains binary validity flags only (0/1), not diameter.",
        f"PCMP normalization excluded. Blink events derived from validity transitions.",
        f"",
        f"## Input",
        f"- File: data/raw/{s['folder']}/eye/eye_data_db.csv",
        f"- Total samples loaded: {load_stats['n_total']}",
        f"- After window filter: {load_stats['n_after_window']}",
        f"- Special window applied: {'Yes (0-900s)' if load_stats['window_applied'] else 'No'}",
        f"",
        f"## Quality Filtering",
        f"- Invalid BPOG samples masked (bpogv=0): {mask_stats['bpogv_invalid']}",
        f"- Off-screen gaze samples masked: {mask_stats['offscreen']}",
        f"- Pupil invalid samples (blink detection only, not masked): {mask_stats['pupil_invalid']}",
        f"- Total NaN after masking: {mask_stats['total_nan_after_mask']} ({100-mask_stats['valid_pct']:.1f}%)",
        f"- Valid samples: {mask_stats['valid_pct']}%",
        f"- Short gaps interpolated (≤200ms): {interp_stats['interpolated_gaps']}",
        f"- Long gaps left as NaN (>200ms): {interp_stats['long_gaps']}",
        f"",
        f"## Blink Detection (validity transitions)",
        f"- Total blink events: {blink_stats['n_blinks']}",
        f"- Blink rate: {blink_stats['blink_rate_per_min']}/min",
        f"",
        f"## Event Detection (I-VT, threshold={IVT_THRESH}°/s)",
        f"- Total fixations: {ivt_stats['n_fixations']}",
        f"- Mean fixation duration: {ivt_stats['mean_fixation_ms']} ms",
        f"- Total saccades: {ivt_stats['n_saccades']}",
        f"- Mean saccade amplitude: {ivt_stats['mean_saccade_amp_deg']}°",
        f"- Screen geometry assumed: {SCREEN_DEG_H}° H × {SCREEN_DEG_V}° V",
        f"",
        f"## Epoch Features",
        f"- ERP epochs with eye features: {epoch_stats.get('n_erp', 0)}",
        f"- EEG ERP epoch count: {eeg_erp_count}",
        f"- Alignment match: {match}",
        f"- Causal epochs with eye features: {epoch_stats.get('n_causal', 0)}",
        f"- Average NaN ratio per ERP epoch: {epoch_stats.get('mean_nan_erp', 0):.3f}",
        f"- Epochs with >50% NaN: {epoch_stats.get('high_nan_epochs', 0)}",
        f"",
        f"## Available Features",
        f"- fixation_count, fixation_mean_dur_ms",
        f"- saccade_count, saccade_mean_amp_deg",
        f"- blink_count, gaze_dispersion, nan_ratio",
        f"",
        f"## Excluded Features (no pupil diameter)",
        f"- pupil_mean_pcmp, pupil_max_pcmp, pupil_change_pcmp",
        f"- pupil_series_50hz",
        f"",
        f"## Notes",
    ]
    if sid == 20:
        md.append("- Veli: eye features restricted to 0-900s window (device reliability)")
    if sid == 23:
        md.append("- Duru: high blink rate expected; check blink_rate_per_min in this report")
    if epoch_stats.get("high_nan_epochs", 0) > 0:
        md.append(f"- {epoch_stats['high_nan_epochs']} ERP epochs have >50% NaN - low quality")

    (subj_out / "eye_preprocessing_report.md").write_text("\n".join(md), encoding="utf-8")
    log.info(f"  5.8 Report saved: eye_preprocessing_report.md")
    return {"match": match, "eeg_erp": eeg_erp_count, "eye_erp": eye_erp}


# ── per-subject orchestrator ──────────────────────────────────────────────────
def process_subject(s, bw_df):
    sid      = s["id"]
    name     = s["name"]
    subj_out = PROC / f"subject_{sid:02d}"
    subj_out.mkdir(parents=True, exist_ok=True)

    log = make_logger(f"s5_{sid}",
                      LOG_DIR / f"section5_eye_preprocessing_subject_{sid:02d}.log")
    log.info("=" * 60)
    log.info(f"Subject {sid:02d}: {name}")
    log.info("=" * 60)
    t_start = time.time()

    bw_row = bw_df[bw_df["subject_id"] == sid].iloc[0]

    df, load_stats = step_load(s, log)
    df, mask_stats = step_mask(df, log)
    df, interp_stats = step_interpolate(df, log)
    df, blink_df, blink_stats = step_detect_blinks(df, log)
    df, fix_df, sacc_df, ivt_stats = step_ivt(df, log)

    # Stop check: >50% NaN after masking
    if mask_stats["valid_pct"] < 50.0:
        log.error(f"  ✗ STOP: valid_pct={mask_stats['valid_pct']}% < 50% - manual review required!")
        return {"id": sid, "name": name, "status": "STOP",
                "reason": f"valid_pct={mask_stats['valid_pct']}%"}

    result = step_epoch_features(df, blink_df, fix_df, sacc_df, s, log)
    if len(result) == 4:
        erp_df, causal_df, ts_array, epoch_stats = result
    else:
        erp_df, causal_df, ts_array = pd.DataFrame(), pd.DataFrame(), np.array([])
        epoch_stats = {"n_erp": 0, "n_causal": 0, "mean_nan_erp": 0,
                       "mean_nan_causal": 0, "high_nan_epochs": 0}

    step_qc_figures(df, blink_df, fix_df, s, bw_row, subj_out, log)

    # Save outputs
    if not fix_df.empty:
        fix_df.to_csv(subj_out / "eye_fixations.csv", index=False)
    if not sacc_df.empty:
        sacc_df.to_csv(subj_out / "eye_saccades.csv", index=False)
    if not erp_df.empty:
        erp_df.to_csv(subj_out / "eye_epoch_features_erp.csv", index=False)
    if not causal_df.empty:
        causal_df.to_csv(subj_out / "eye_epoch_features_causal.csv", index=False)
    if ts_array.size > 0:
        np.save(str(subj_out / "eye_timeseries_causal.npy"), ts_array)

    align = write_subject_report(s, subj_out, load_stats, mask_stats, interp_stats,
                                  blink_stats, ivt_stats, epoch_stats, bw_row, log)

    elapsed = time.time() - t_start
    log.info(f"  ✓ sub-{sid} DONE in {elapsed:.1f}s  valid={mask_stats['valid_pct']}%  "
             f"fixations={ivt_stats['n_fixations']}  blinks={blink_stats['n_blinks']}  "
             f"ERP_epochs={epoch_stats.get('n_erp',0)}")

    return {
        "id": sid, "name": name, "status": "OK",
        "valid_pct":         mask_stats["valid_pct"],
        "n_fixations":       ivt_stats["n_fixations"],
        "mean_fix_ms":       ivt_stats["mean_fixation_ms"],
        "n_saccades":        ivt_stats["n_saccades"],
        "blink_rate":        blink_stats["blink_rate_per_min"],
        "n_erp_epochs":      epoch_stats.get("n_erp", 0),
        "n_causal_epochs":   epoch_stats.get("n_causal", 0),
        "mean_nan_erp":      epoch_stats.get("mean_nan_erp", 0),
        "high_nan_epochs":   epoch_stats.get("high_nan_epochs", 0),
        "eeg_erp_count":     align["eeg_erp"],
        "align_match":       align["match"],
        "elapsed_s":         round(elapsed, 1),
    }


# ── cross-subject summary ─────────────────────────────────────────────────────
def write_summary(results):
    success = [r for r in results if r.get("status") == "OK"]
    failed  = [r for r in results if r.get("status") != "OK"]

    header = ("| Subject | Valid% | Fix Count | Mean Fix (ms) | "
               "Blink/min | ERP epochs | Align |")
    sep    = ("|---------|--------|-----------|---------------|"
               "-----------|------------|-------|")
    rows   = []
    for r in success:
        rows.append(
            f"| {r['id']} {r['name']} | {r['valid_pct']}% | "
            f"{r['n_fixations']} | {r['mean_fix_ms']:.0f} | "
            f"{r['blink_rate']} | {r['n_erp_epochs']} | {r['align_match']} |"
        )

    # Alignment check: mismatch > 5%
    mismatch_flag = []
    for r in success:
        try:
            eeg = int(r["eeg_erp_count"]) if r["eeg_erp_count"] != "N/A" else None
            eye = r["n_erp_epochs"]
            if eeg and eeg > 0 and abs(eeg - eye) / eeg > 0.05:
                mismatch_flag.append(f"sub-{r['id']} {r['name']} (EEG={eeg}, eye={eye})")
        except Exception:
            pass

    md = f"""# Section 5 - Eye Preprocessing Summary

Generated: {date.today()}

## Overview
- {len(success)}/{len(results)} subjects processed successfully
- Special handling: Veli (0-900s window)
- Pipeline: mask → interpolate → blink detection → I-VT → epoch features
- NOTE: Pupil diameter not available (GP3 HD exported binary validity only).
  PCMP features excluded. Blink detection uses validity-flag transitions.

## Per-subject metrics

{header}
{sep}
{chr(10).join(rows)}

## Quality Flags
- Veli (20): eye features restricted to 0-900s (device reliability)
- Duru (23): monitor blink_rate (expected high)
- Pupil features: excluded for all subjects (no diameter data in export)

## Multimodal Alignment Check
{'- All subjects: EEG ↔ Eye epoch counts match (≤5% tolerance)' if not mismatch_flag else chr(10).join(['- MISMATCH: ' + m for m in mismatch_flag])}

## Available Eye Features (Section 6+)
- fixation_count, fixation_mean_dur_ms
- saccade_count, saccade_mean_amp_deg
- blink_count, gaze_dispersion, nan_ratio

## Excluded Features
- pupil_mean_pcmp, pupil_max_pcmp, pupil_change_pcmp (no pupil diameter)

## Readiness for Section 6 (Mouse Preprocessing)
**{'YES' if not failed and not mismatch_flag else 'CHECK_REQUIRED - see flags above'}**
"""
    out = REP5 / "section5_summary.md"
    out.write_text(md, encoding="utf-8")
    print(f"\n  Summary: {out}")
    return mismatch_flag


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "="*60)
    print("Bölüm 5: Eye Tracking Preprocessing - 9 subjects")
    print("="*60)

    bw_df   = load_baseline_windows()
    results = []
    stop_subjects = []

    for s in SUBJECTS:
        print(f"\n{'─'*50}\nStarting sub-{s['id']} {s['name']}\n{'─'*50}")
        try:
            r = process_subject(s, bw_df)
        except Exception as e:
            import traceback
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            r = {"id": s["id"], "name": s["name"], "status": "FAILED", "reason": str(e)}
        results.append(r)

        if r.get("status") == "STOP":
            stop_subjects.append(r)
            print(f"\n  ⚠ STOP condition for sub-{s['id']} - halting for manual review")
            break

    if stop_subjects:
        print(f"\n{'='*60}\nSTOP CONDITIONS TRIGGERED\n{'='*60}")
        for r in stop_subjects:
            print(f"  sub-{r['id']} {r['name']}: {r['reason']}")
        return

    mismatch = write_summary(results)

    # Update analysis_notes.txt
    notes_path = ROOT / "analysis_notes.txt"
    existing   = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    note = (f"\n\n[{date.today()}] Section 5 Eye Preprocessing:\n"
            f"  Pupil diameter not available in GP3 HD export (pupil_left/right are binary "
            f"validity flags 0/1, not mm). PCMP normalization excluded. Blink events derived "
            f"from validity-flag transitions (bpogv=0 OR pupil validity=0).")
    notes_path.write_text(existing + note, encoding="utf-8")

    # Final summary
    success  = [r for r in results if r.get("status") == "OK"]
    failed   = [r for r in results if r.get("status") not in ("OK",)]
    print(f"\n{'='*60}")
    print("BÖLÜM 5 TAMAMLANDI")
    print(f"{'='*60}")
    print(f"\n  Başarılı: {len(success)}/{len(results)}")
    print(f"\n  {'Sub':<6} {'Name':<22} {'Valid%':>7} {'Fix':>6} {'Blk/m':>7} {'ERP':>5} {'Align':>5}")
    print(f"  {'─'*60}")
    for r in results:
        if r.get("status") == "OK":
            alm = "✓" if r["align_match"] == "yes" else "✗"
            print(f"  {r['id']:<6} {r['name']:<22} {r['valid_pct']:>6}% "
                  f"{r['n_fixations']:>6} {r['blink_rate']:>7} {r['n_erp_epochs']:>5} {alm:>5}")
        else:
            print(f"  {r['id']:<6} {r['name']:<22}  {'FAILED/STOP':>30}")

    if mismatch:
        print(f"\n  ⚠ Epoch alignment mismatch (>5%): {mismatch}")
    else:
        print(f"\n  ✓ Epoch alignment: all subjects within 5% tolerance")

    if failed:
        print(f"\n  ⚠ {len(failed)} subjects failed: {[r['name'] for r in failed]}")


if __name__ == "__main__":
    main()
