"""
Section 8a - Feature Extraction for Approach A (LaBraM + Husformer pipeline).

Outputs:
  approach_a/features/all_eeg_embeddings.npy   (656, 200)
  approach_a/features/all_eye_timeseries.npy   (656, 6, 110)
  approach_a/features/all_mouse_timeseries.npy (656, 7, 210)
  approach_a/features/epoch_metadata.csv       (656 rows)

EEG: epochs_erp-epo.fif (32ch, 500 Hz, -0.2..+2.0 s) → LaBraM → 200-dim
Eye: raw eye_data_db.csv → 6-ch × 110 steps (50 Hz, ERP window)
Mouse: mouse_trajectory_fixed.csv + clicks → 7-ch × 210 steps (50 Hz, -200..+4000 ms)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "approach_a" / "src"))

import logging
import numpy as np
import pandas as pd
import mne
import yaml
from labram_wrapper import load_labram_base

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
EYE_TMIN_MS  = -200
EYE_TMAX_MS  = 2000   # ERP window
EYE_STEPS    = 110    # 2200ms / 20ms
EYE_CHANNELS = 6      # gaze_x, gaze_y, fixation_flag, blink_flag, nan_flag, velocity

MOUSE_TMIN_MS  = -200
MOUSE_TMAX_MS  = 4000
MOUSE_STEPS    = 210  # 4200ms / 20ms
MOUSE_CHANNELS = 7    # x_norm, y_norm, velocity, acceleration, is_idle, click_flag, rage_click_flag

DT_MS = 20.0          # 50 Hz bin width

# Screen FOV for gaze velocity (degrees)
FOV_H_DEG = 48.3
FOV_V_DEG = 28.1

SUBJECTS_TO_PROCESS = [14, 15, 16, 17, 18, 20, 21, 22, 23]  # subject 19 excluded (no processed data)

OUT_DIR = ROOT / "approach_a" / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Config helpers ─────────────────────────────────────────────────────────────
def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def subject_info(cfg, sid):
    for s in cfg["subjects"]:
        if s["id"] == sid:
            return s
    raise KeyError(f"Subject {sid} not in config")


def is_frustration(scenario_code: str, cfg) -> int:
    """Return 1 if scenario code is a frustration trigger, else 0."""
    code = scenario_code.lstrip("S").lstrip("0")
    key = f"S{int(code):02d}" if code.isdigit() else scenario_code
    sm = cfg.get("scenario_metadata", {})
    return int(sm.get(key, {}).get("is_frustration_label", False))


# ── EEG embeddings ─────────────────────────────────────────────────────────────
def extract_eeg_embeddings(sid: int, epoch_rows: pd.DataFrame, encoder) -> np.ndarray:
    """
    Load epochs_erp-epo.fif, select rows by eeg_index, run LaBraM.
    Returns (n, 200) float32.
    """
    epo_path = ROOT / "data" / "processed" / f"subject_{sid}" / "epochs_erp-epo.fif"
    epochs = mne.read_epochs(str(epo_path), preload=True, verbose=False)
    data = epochs.get_data()  # (n_total, 32, 1101) at 500 Hz

    indices = epoch_rows["eeg_index"].values
    selected = data[indices]   # (n, 32, 1101)
    embs = encoder.get_embeddings(selected, batch_size=32)
    log.info(f"  EEG sub-{sid}: {len(indices)} embeddings extracted")
    return embs


# ── Eye time series ────────────────────────────────────────────────────────────
def _build_eye_timeseries_one(t0_ms: float, eye_df: pd.DataFrame,
                               fixations: pd.DataFrame) -> np.ndarray:
    """Build (6, 110) eye time series for one epoch centered at t0_ms."""
    t_start = t0_ms + EYE_TMIN_MS
    t_end   = t0_ms + EYE_TMAX_MS
    grid    = np.linspace(t_start, t_end, EYE_STEPS)  # 110 points

    # Slice raw eye data to slightly wider window for interpolation
    margin = 200
    win = eye_df[(eye_df["wall_time_ms"] >= t_start - margin) &
                 (eye_df["wall_time_ms"] <= t_end   + margin)].copy()

    ts = np.zeros((EYE_CHANNELS, EYE_STEPS), dtype=np.float32)

    if len(win) < 2:
        return ts  # all zeros = invalid epoch

    t_raw = win["wall_time_ms"].values
    gx    = np.where(win["bpogv"].values == 1, win["gaze_x"].values, np.nan)
    gy    = np.where(win["bpogv"].values == 1, win["gaze_y"].values, np.nan)
    blink = ((win["pupil_left"].values == 0) & (win["pupil_right"].values == 0)).astype(float)
    nan_f = (win["bpogv"].values == 0).astype(float)

    # Interpolate gaze_x, gaze_y where valid
    valid = ~np.isnan(gx)
    if valid.sum() >= 2:
        ts[0] = np.clip(np.interp(grid, t_raw[valid], gx[valid]), 0.0, 1.0)
        ts[1] = np.clip(np.interp(grid, t_raw[valid], gy[valid]), 0.0, 1.0)
    else:
        ts[0] = 0.5  # center fallback
        ts[1] = 0.5

    # Fixation flag: 1 if grid point falls within a known fixation
    if len(fixations) > 0:
        fix_flag = np.zeros(EYE_STEPS, dtype=np.float32)
        for _, fx in fixations.iterrows():
            fix_flag[(grid >= fx["start_ms"]) & (grid <= fx["end_ms"])] = 1.0
        ts[2] = fix_flag

    # Blink flag (nearest-neighbor onto grid)
    ts[3] = np.interp(grid, t_raw, blink).astype(np.float32)

    # NaN flag
    ts[4] = np.interp(grid, t_raw, nan_f).astype(np.float32)

    # Gaze velocity in deg/s - computed from consecutive valid gaze points
    vel_raw = np.zeros(len(t_raw), dtype=np.float32)
    for i in range(1, len(t_raw)):
        if not (np.isnan(gx[i]) or np.isnan(gx[i - 1])):
            dt_s = (t_raw[i] - t_raw[i - 1]) / 1000.0
            if dt_s > 0:
                dx_deg = (gx[i] - gx[i - 1]) * FOV_H_DEG
                dy_deg = (gy[i] - gy[i - 1]) * FOV_V_DEG
                vel_raw[i] = np.sqrt(dx_deg**2 + dy_deg**2) / dt_s
    # Cap velocity at 1000 deg/s (fast saccades ≤ 700 deg/s; higher values are blink/noise)
    ts[5] = np.clip(np.interp(grid, t_raw, vel_raw), 0, 1000).astype(np.float32)

    return ts


def extract_eye_timeseries(sid: int, epoch_rows: pd.DataFrame,
                            info: dict) -> np.ndarray:
    """Return (n, 6, 110) eye timeseries for usable epochs."""
    raw_path = ROOT / "data" / "raw" / info["folder"] / "eye" / "eye_data_db.csv"
    eye_df   = pd.read_csv(str(raw_path))
    eye_df   = eye_df.sort_values("wall_time_ms").drop_duplicates(subset="wall_time_ms")

    fix_path  = ROOT / "data" / "processed" / f"subject_{sid}" / "eye_fixations.csv"
    fixations = pd.read_csv(str(fix_path)) if fix_path.exists() else pd.DataFrame()

    n = len(epoch_rows)
    out = np.zeros((n, EYE_CHANNELS, EYE_STEPS), dtype=np.float32)
    for i, (_, row) in enumerate(epoch_rows.iterrows()):
        out[i] = _build_eye_timeseries_one(float(row["wall_time_ms"]), eye_df, fixations)

    log.info(f"  Eye sub-{sid}: {n} timeseries built")
    return out


# ── Mouse time series ──────────────────────────────────────────────────────────
def _build_mouse_timeseries_one(t0_ms: float, traj: pd.DataFrame,
                                 clicks: pd.DataFrame) -> np.ndarray:
    """Build (7, 210) mouse time series for one epoch centered at t0_ms."""
    t_start = t0_ms + MOUSE_TMIN_MS
    t_end   = t0_ms + MOUSE_TMAX_MS
    grid    = np.linspace(t_start, t_end, MOUSE_STEPS)

    margin = 300
    win = traj[(traj["wall_time_ms"] >= t_start - margin) &
               (traj["wall_time_ms"] <= t_end   + margin)]

    ts = np.zeros((MOUSE_CHANNELS, MOUSE_STEPS), dtype=np.float32)

    if len(win) < 2:
        return ts

    t_raw  = win["wall_time_ms"].values
    x_n    = win["x_norm"].values.astype(float)
    y_n    = win["y_norm"].values.astype(float)
    vel    = np.nan_to_num(win["velocity_px_s"].values.astype(float),   nan=0.0)
    acc    = np.nan_to_num(win["acceleration_px_s2"].values.astype(float), nan=0.0)
    idle   = win["is_idle"].astype(float).values

    ts[0] = np.clip(np.interp(grid, t_raw, x_n),  0, 1).astype(np.float32)
    ts[1] = np.clip(np.interp(grid, t_raw, y_n),  0, 1).astype(np.float32)
    ts[2] = np.interp(grid, t_raw, vel).astype(np.float32)
    ts[3] = np.interp(grid, t_raw, acc).astype(np.float32)
    ts[4] = np.round(np.interp(grid, t_raw, idle)).astype(np.float32)

    # Click flag - binary pulse in 20ms bins
    if len(clicks) > 0 and "wall_time_ms" in clicks.columns:
        c_times = clicks["wall_time_ms"].values
        for g_idx, g_t in enumerate(grid):
            if np.any((c_times >= g_t - DT_MS / 2) & (c_times < g_t + DT_MS / 2)):
                ts[5, g_idx] = 1.0

        # Rage click flag - use rage_click column if available
        if "is_rage_click" in clicks.columns:
            rc_times = clicks.loc[clicks["is_rage_click"] == True, "wall_time_ms"].values
            for g_idx, g_t in enumerate(grid):
                if np.any((rc_times >= g_t - DT_MS / 2) & (rc_times < g_t + DT_MS / 2)):
                    ts[6, g_idx] = 1.0

    return ts


def extract_mouse_timeseries(sid: int, epoch_rows: pd.DataFrame) -> np.ndarray:
    """Return (n, 7, 210) mouse timeseries for usable epochs."""
    proc_dir = ROOT / "data" / "processed" / f"subject_{sid}"
    traj     = pd.read_csv(str(proc_dir / "mouse_trajectory_fixed.csv"))
    traj     = traj.sort_values("wall_time_ms")

    # Load clicks from interim directory
    interim  = ROOT / "data" / "interim" / f"subject_{sid}"
    clicks   = pd.DataFrame()
    clicks_path = interim / "mouse_clicks.csv"
    if clicks_path.exists():
        clicks = pd.read_csv(str(clicks_path))

    n = len(epoch_rows)
    out = np.zeros((n, MOUSE_CHANNELS, MOUSE_STEPS), dtype=np.float32)
    for i, (_, row) in enumerate(epoch_rows.iterrows()):
        out[i] = _build_mouse_timeseries_one(float(row["wall_time_ms"]), traj, clicks)

    log.info(f"  Mouse sub-{sid}: {n} timeseries built")
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    cfg = load_config()
    log.info("Loading LaBraM encoder...")
    encoder = load_labram_base()

    all_eeg   = []
    all_eye   = []
    all_mouse = []
    all_meta  = []

    global_idx = 0
    for sid in SUBJECTS_TO_PROCESS:
        log.info(f"\n── Subject {sid} ──────────────────────────────────────")
        align_path = ROOT / "data" / "reports" / "section7_sync" / f"alignment_master_subject_{sid}.csv"
        align = pd.read_csv(str(align_path))
        usable = align[align["usable_for_multimodal"] == "yes"].reset_index(drop=True)
        log.info(f"  Usable epochs: {len(usable)}")

        info = subject_info(cfg, sid)

        # EEG
        eeg_embs = extract_eeg_embeddings(sid, usable, encoder)
        all_eeg.append(eeg_embs)

        # Eye
        eye_ts = extract_eye_timeseries(sid, usable, info)
        all_eye.append(eye_ts)

        # Mouse
        mouse_ts = extract_mouse_timeseries(sid, usable)
        all_mouse.append(mouse_ts)

        # Load mouse ERP features to get rage_click_flag per epoch
        mouse_feat_path = ROOT / "data" / "processed" / f"subject_{sid}" / "mouse_epoch_features_erp.csv"
        mouse_feat = pd.read_csv(str(mouse_feat_path)) if mouse_feat_path.exists() else pd.DataFrame()

        # Metadata rows
        for _, row in usable.iterrows():
            eid = row["event_id"]
            label_frust = is_frustration(str(eid), cfg)

            # rage_click_flag: match by wall_time_ms (nearest within 200ms)
            rage = 0
            if not mouse_feat.empty and "rage_click_flag" in mouse_feat.columns:
                diffs = (mouse_feat["wall_time_ms"] - row["wall_time_ms"]).abs()
                best = diffs.idxmin()
                if diffs[best] < 200:
                    rage = int(bool(mouse_feat.loc[best, "rage_click_flag"]))

            all_meta.append({
                "global_idx":    global_idx,
                "subject_id":    sid,
                "epoch_id":      row["epoch_id"],
                "event_id":      eid,
                "scenario_name": row["scenario_name"],
                "phase":         row["phase"],
                "wall_time_ms":  row["wall_time_ms"],
                "eeg_index":     row["eeg_index"],
                "eye_index":     row["eye_index"],
                "mouse_index":   row["mouse_index"],
                "label_frustration": label_frust,
                "label_rage_click":  rage,
            })
            global_idx += 1

    # Stack and save
    eeg_arr   = np.concatenate(all_eeg,   axis=0)  # (N, 200)
    eye_arr   = np.concatenate(all_eye,   axis=0)  # (N, 6, 110)
    mouse_arr = np.concatenate(all_mouse, axis=0)  # (N, 7, 210)
    meta_df   = pd.DataFrame(all_meta)

    log.info(f"\n── Final shapes ──────────────────────────────────────")
    log.info(f"  EEG embeddings:  {eeg_arr.shape}")
    log.info(f"  Eye timeseries:  {eye_arr.shape}")
    log.info(f"  Mouse timeseries:{mouse_arr.shape}")
    log.info(f"  Metadata:        {meta_df.shape}")

    np.save(str(OUT_DIR / "all_eeg_embeddings.npy"),   eeg_arr)
    np.save(str(OUT_DIR / "all_eye_timeseries.npy"),   eye_arr)
    np.save(str(OUT_DIR / "all_mouse_timeseries.npy"), mouse_arr)
    meta_df.to_csv(str(OUT_DIR / "epoch_metadata.csv"), index=False)

    log.info("Saved to approach_a/features/")

    # Sanity assertions
    N = global_idx
    assert eeg_arr.shape   == (N, 200),            f"EEG shape mismatch: {eeg_arr.shape}"
    assert eye_arr.shape   == (N, 6,  110),        f"Eye shape mismatch: {eye_arr.shape}"
    assert mouse_arr.shape == (N, 7,  210),        f"Mouse shape mismatch: {mouse_arr.shape}"
    log.info(f"All shape assertions PASSED  (N={N})")


if __name__ == "__main__":
    main()
