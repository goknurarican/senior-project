"""
Bölüm 4-6 Eklentisi - Control Phase Epoch Extension.

Outputs per subject (data/processed/subject_XX/):
  control_pseudo_markers.csv
  epochs_erp_control-epo.fif
  epochs_causal_control-epo.fif
  eye_epoch_features_control_erp.csv
  eye_epoch_features_control_causal.csv
  eye_timeseries_control_erp.npy      (n, 6, 110)
  mouse_epoch_features_control_erp.csv
  mouse_epoch_features_control_causal.csv
  mouse_timeseries_control_erp.npy    (n, 7, 210)

Outputs global (approach_a/features/):
  subject_XX/eeg_embeddings_control.npy  (n, 200)
  all_eeg_embeddings_v2.npy
  all_eye_timeseries_v2.npy
  all_mouse_timeseries_v2.npy
  labels_v2.csv
  all_eeg_embeddings_v2_metadata.csv

Outputs reports:
  approach_a/reports/control_vs_variant_pca.png
  approach_a/reports/control_extension_report.md
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "approach_a" / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml
from autoreject import AutoReject
from labram_wrapper import load_labram_base
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
SUBJECTS = [14, 15, 16, 17, 18, 20, 21, 22, 23]

SETTLE_S      = 1.0
BUFFER_S      = 3.5
PSEUDO_STEP_S = 5.0

ERP_TMIN   = -0.2
ERP_TMAX   =  2.0
CAUSAL_TMIN = -0.5
CAUSAL_TMAX =  3.0

# Eye timeseries (ERP window: 2200 ms / 20 ms = 110 steps)
EYE_TMIN_MS  = -200
EYE_TMAX_MS  =  2000
EYE_STEPS    = 110
EYE_CHANNELS = 6

# Mouse timeseries (same window as variant: 4200 ms / 20 ms = 210 steps)
MOUSE_TMIN_MS  = -200
MOUSE_TMAX_MS  =  4000
MOUSE_STEPS    = 210
MOUSE_CHANNELS = 7

DT_MS     = 20.0
FOV_H_DEG = 48.3
FOV_V_DEG = 28.1
SCREEN_W  = 1920
SCREEN_H  = 1080

VELI_SID       = 20
VELI_MAX_S     = 900.0

FEAT_DIR = ROOT / "approach_a" / "features"
REP_DIR  = ROOT / "approach_a" / "reports"
REP_DIR.mkdir(parents=True, exist_ok=True)


# ── Config ─────────────────────────────────────────────────────────────────────
def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)

def subject_info(cfg, sid):
    for s in cfg["subjects"]:
        if s["id"] == sid:
            return s
    raise KeyError(sid)


# ── Wall-time offset ───────────────────────────────────────────────────────────
def get_raw_start_wall_ms(sid: int) -> float:
    """wall_time_ms of EEG t=0 derived from epoch_metadata.csv + first variant epoch."""
    meta = pd.read_csv(FEAT_DIR / "epoch_metadata.csv")
    first_wall = float(meta[meta["subject_id"] == sid]["wall_time_ms"].iloc[0])
    epo_path   = ROOT / "data" / "processed" / f"subject_{sid}" / "epochs_erp-epo.fif"
    epochs     = mne.read_epochs(str(epo_path), preload=False, verbose=False)
    sfreq      = epochs.info["sfreq"]
    first_samp = epochs.events[0, 0]
    return first_wall - first_samp / sfreq * 1000


# ── Step 1 ─────────────────────────────────────────────────────────────────────
def step1_pseudo_markers(sid: int, raw_start_wall_ms: float) -> pd.DataFrame:
    proc_dir = ROOT / "data" / "processed" / f"subject_{sid}"
    epochs   = mne.read_epochs(str(proc_dir / "epochs_erp-epo.fif"),
                                preload=False, verbose=False)
    sfreq               = epochs.info["sfreq"]
    first_variant_raw_s = epochs.events[0, 0] / sfreq
    ctrl_start          = SETTLE_S
    ctrl_end            = first_variant_raw_s - BUFFER_S

    if ctrl_end <= ctrl_start + PSEUDO_STEP_S:
        log.warning(f"  sub-{sid}: control phase too short ({ctrl_end:.1f}s)")
        return pd.DataFrame()

    times_s = np.arange(ctrl_start, ctrl_end, PSEUDO_STEP_S)
    df = pd.DataFrame({
        "epoch_id":      np.arange(len(times_s)),
        "event_code":    99,
        "scenario_name": "control_baseline",
        "phase":         "control",
        "raw_time_s":    times_s,
        "raw_sample":    (times_s * sfreq).astype(int),
        "wall_time_ms":  raw_start_wall_ms + times_s * 1000,
    })
    df.to_csv(str(proc_dir / "control_pseudo_markers.csv"), index=False)
    log.info(f"  sub-{sid}: {len(df)} pseudo-markers  ctrl=[{ctrl_start:.1f}, {ctrl_end:.1f}]s")
    return df


# ── Step 2 ─────────────────────────────────────────────────────────────────────
def step2_eeg_epochs(sid: int, markers_df: pd.DataFrame):
    if markers_df.empty:
        return None, None

    proc_dir = ROOT / "data" / "processed" / f"subject_{sid}"
    raw = mne.io.read_raw_fif(str(proc_dir / "raw_clean-raw.fif"),
                               preload=True, verbose=False)

    samples = markers_df["raw_sample"].values.astype(int)
    events  = np.column_stack([
        samples,
        np.zeros(len(samples), int),
        np.full(len(samples), 99, int),
    ])

    def _cut_and_ar(tmin, tmax, baseline, tag):
        epo = mne.Epochs(
            raw, events, event_id={"control_baseline": 99},
            tmin=tmin, tmax=tmax,
            baseline=baseline, preload=True, verbose=False,
        )
        if len(epo) > 1:
            ar = AutoReject(n_interpolate=[1, 2, 4], random_state=42, verbose=False)
            epo, _ = ar.fit_transform(epo, return_log=True)
        log.info(f"  sub-{sid} {tag}: {len(epo)} epochs after AR")
        return epo

    epo_erp    = _cut_and_ar(ERP_TMIN,    ERP_TMAX,    (ERP_TMIN, 0), "ERP")
    epo_causal = _cut_and_ar(CAUSAL_TMIN, CAUSAL_TMAX, None,           "Causal")

    epo_erp.save(str(proc_dir / "epochs_erp_control-epo.fif"),    overwrite=True, verbose=False)
    epo_causal.save(str(proc_dir / "epochs_causal_control-epo.fif"), overwrite=True, verbose=False)
    return epo_erp, epo_causal


def _survived_markers(epochs_erp, raw_start_wall_ms: float) -> pd.DataFrame:
    if epochs_erp is None or len(epochs_erp) == 0:
        return pd.DataFrame()
    sfreq = epochs_erp.info["sfreq"]
    samps = epochs_erp.events[:, 0]
    return pd.DataFrame({
        "epoch_id":      np.arange(len(samps)),
        "event_code":    99,
        "scenario_name": "control_baseline",
        "phase":         "control",
        "raw_sample":    samps,
        "raw_time_s":    samps / sfreq,
        "wall_time_ms":  raw_start_wall_ms + samps / sfreq * 1000,
    })


# ── Step 3 helpers ─────────────────────────────────────────────────────────────
def _eye_scalar(t0_ms, eye_df, fixations, saccades, tmin_ms, tmax_ms):
    t_s = t0_ms + tmin_ms
    t_e = t0_ms + tmax_ms
    win = eye_df[(eye_df["wall_time_ms"] >= t_s) & (eye_df["wall_time_ms"] <= t_e)]
    n   = len(win)

    if n == 0:
        return dict(fixation_count=0, fixation_mean_dur_ms=np.nan,
                    saccade_count=0, saccade_mean_amp_deg=np.nan,
                    blink_count=0, gaze_dispersion=np.nan,
                    nan_ratio=1.0, n_samples=0)

    # fixations
    if len(fixations):
        fw = fixations[(fixations["end_ms"] >= t_s) & (fixations["start_ms"] <= t_e)]
        fc  = len(fw)
        fdm = float(fw["duration_ms"].mean()) if fc else np.nan
    else:
        fc, fdm = 0, np.nan

    # saccades
    if len(saccades):
        sw  = saccades[(saccades["end_ms"] >= t_s) & (saccades["start_ms"] <= t_e)]
        sc  = len(sw)
        sam = float(sw["amplitude_deg"].mean()) if sc else np.nan
    else:
        sc, sam = 0, np.nan

    bv    = ((win["pupil_left"].values == 0) & (win["pupil_right"].values == 0)).astype(int)
    blink = int(np.diff(np.concatenate([[0], bv])).clip(0).sum())

    valid = win[win["bpogv"] == 1]
    disp  = float(np.sqrt(valid["gaze_x"].std()**2 + valid["gaze_y"].std()**2)) if len(valid) >= 2 else np.nan
    nan_r = float((win["bpogv"] == 0).mean())

    return dict(fixation_count=fc, fixation_mean_dur_ms=fdm,
                saccade_count=sc, saccade_mean_amp_deg=sam,
                blink_count=blink, gaze_dispersion=disp,
                nan_ratio=nan_r, n_samples=n)


def _eye_ts(t0_ms, eye_df, fixations, tmin_ms, tmax_ms, n_steps):
    t_s  = t0_ms + tmin_ms
    t_e  = t0_ms + tmax_ms
    grid = np.linspace(t_s, t_e, n_steps)
    m    = 200
    win  = eye_df[(eye_df["wall_time_ms"] >= t_s - m) &
                  (eye_df["wall_time_ms"] <= t_e + m)].copy()
    ts   = np.zeros((EYE_CHANNELS, n_steps), np.float32)
    if len(win) < 2:
        return ts

    t_raw = win["wall_time_ms"].values
    gx    = np.where(win["bpogv"].values == 1, win["gaze_x"].values, np.nan)
    gy    = np.where(win["bpogv"].values == 1, win["gaze_y"].values, np.nan)
    blink = ((win["pupil_left"].values == 0) & (win["pupil_right"].values == 0)).astype(float)
    nan_f = (win["bpogv"].values == 0).astype(float)

    valid = ~np.isnan(gx)
    if valid.sum() >= 2:
        ts[0] = np.clip(np.interp(grid, t_raw[valid], gx[valid]), 0, 1)
        ts[1] = np.clip(np.interp(grid, t_raw[valid], gy[valid]), 0, 1)
    else:
        ts[0] = ts[1] = 0.5

    if len(fixations):
        ff = np.zeros(n_steps, np.float32)
        for _, fx in fixations.iterrows():
            ff[(grid >= fx["start_ms"]) & (grid <= fx["end_ms"])] = 1.0
        ts[2] = ff

    ts[3] = np.interp(grid, t_raw, blink).astype(np.float32)
    ts[4] = np.interp(grid, t_raw, nan_f).astype(np.float32)

    vel = np.zeros(len(t_raw), np.float32)
    for i in range(1, len(t_raw)):
        if not (np.isnan(gx[i]) or np.isnan(gx[i - 1])):
            dt = (t_raw[i] - t_raw[i - 1]) / 1000.0
            if dt > 0:
                vel[i] = np.sqrt(((gx[i] - gx[i-1]) * FOV_H_DEG)**2 +
                                  ((gy[i] - gy[i-1]) * FOV_V_DEG)**2) / dt
    ts[5] = np.clip(np.interp(grid, t_raw, vel), 0, 1000).astype(np.float32)
    return ts


# ── Step 3 ─────────────────────────────────────────────────────────────────────
_EYE_COL_ORDER = ["scenario_type", "eeg_marker", "wall_time_ms",
                   "fixation_count", "fixation_mean_dur_ms",
                   "saccade_count", "saccade_mean_amp_deg",
                   "blink_count", "gaze_dispersion", "nan_ratio", "n_samples"]

def step3_eye_features(sid: int, markers: pd.DataFrame,
                        info: dict, raw_start_wall_ms: float):
    if markers.empty:
        empty = np.zeros((0, EYE_CHANNELS, EYE_STEPS), np.float32)
        return pd.DataFrame(), pd.DataFrame(), empty

    proc_dir = ROOT / "data" / "processed" / f"subject_{sid}"
    eye_df   = (pd.read_csv(ROOT / "data" / "raw" / info["folder"] / "eye" / "eye_data_db.csv")
                  .sort_values("wall_time_ms").drop_duplicates("wall_time_ms"))
    fixations = pd.read_csv(str(proc_dir / "eye_fixations.csv")) \
                if (proc_dir / "eye_fixations.csv").exists() else pd.DataFrame()
    saccades  = pd.read_csv(str(proc_dir / "eye_saccades.csv")) \
                if (proc_dir / "eye_saccades.csv").exists()  else pd.DataFrame()

    veli_max_ms = raw_start_wall_ms + VELI_MAX_S * 1000 if sid == VELI_SID else None
    nan_row     = {c: np.nan for c in _EYE_COL_ORDER}

    rows_erp, rows_cau, ts_list = [], [], []

    for _, row in markers.iterrows():
        t0 = float(row["wall_time_ms"])
        invalid = veli_max_ms is not None and t0 > veli_max_ms

        def base(tmin, tmax):
            if invalid:
                r = dict(nan_row)
            else:
                r = _eye_scalar(t0, eye_df, fixations, saccades, tmin, tmax)
            r.update(scenario_type="control_baseline", eeg_marker=99, wall_time_ms=t0)
            return r

        rows_erp.append(base(EYE_TMIN_MS, EYE_TMAX_MS))
        rows_cau.append(base(int(CAUSAL_TMIN * 1000), int(CAUSAL_TMAX * 1000)))

        if invalid:
            ts_list.append(np.full((EYE_CHANNELS, EYE_STEPS), np.nan, np.float32))
        else:
            ts_list.append(_eye_ts(t0, eye_df, fixations,
                                    EYE_TMIN_MS, EYE_TMAX_MS, EYE_STEPS))

    df_erp = pd.DataFrame(rows_erp)[_EYE_COL_ORDER]
    df_cau = pd.DataFrame(rows_cau)[_EYE_COL_ORDER]
    ts_arr = np.stack(ts_list, axis=0)

    df_erp.to_csv(str(proc_dir / "eye_epoch_features_control_erp.csv"),    index=False)
    df_cau.to_csv(str(proc_dir / "eye_epoch_features_control_causal.csv"), index=False)
    np.save(str(proc_dir / "eye_timeseries_control_erp.npy"), ts_arr)
    log.info(f"  sub-{sid} eye: {len(df_erp)} epochs  (veli_clamp={veli_max_ms is not None})")
    return df_erp, df_cau, ts_arr


# ── Step 4 helpers ─────────────────────────────────────────────────────────────
def _mouse_scalar(t0_ms, traj, clicks_df, tmin_ms, tmax_ms):
    t_s = t0_ms + tmin_ms
    t_e = t0_ms + tmax_ms
    win = traj[(traj["wall_time_ms"] >= t_s) & (traj["wall_time_ms"] <= t_e)]

    nan_keys = ["velocity_mean", "velocity_max", "velocity_std",
                "acceleration_mean", "acceleration_max",
                "path_length_px", "auc_deviation",
                "x_flips", "y_flips", "idle_ratio",
                "click_count", "rage_click_flag", "rage_click_count",
                "right_click_count", "mean_click_interval_ms",
                "scroll_count", "scroll_direction_changes",
                "total_scroll_distance_px", "back_and_forth_score"]
    if len(win) == 0:
        return {k: np.nan for k in nan_keys}

    vel  = np.nan_to_num(win["velocity_px_s"].values.astype(float), nan=0.0)
    acc  = np.nan_to_num(win["acceleration_px_s2"].values.astype(float), nan=0.0)
    x_n  = win["x_norm"].values.astype(float)
    y_n  = win["y_norm"].values.astype(float)
    idle = win["is_idle"].values.astype(float)

    # path length (pixels)
    dx   = np.diff(x_n) * SCREEN_W
    dy   = np.diff(y_n) * SCREEN_H
    path = float(np.sqrt(dx**2 + dy**2).sum()) if len(dx) else 0.0

    # AUC deviation (max perp distance from start→end line)
    if len(x_n) >= 2:
        xp = x_n * SCREEN_W;  yp = y_n * SCREEN_H
        x0, y0, x1, y1 = xp[0], yp[0], xp[-1], yp[-1]
        ll = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        auc = float(np.abs((y1-y0)*xp - (x1-x0)*yp + x1*y0 - y1*x0).max() / ll) if ll > 0 else 0.0
    else:
        auc = 0.0

    xf = int(np.sum(np.diff(np.sign(np.diff(x_n))) != 0)) if len(x_n) > 2 else 0
    yf = int(np.sum(np.diff(np.sign(np.diff(y_n))) != 0)) if len(y_n) > 2 else 0

    # clicks
    click_count = rage_click = rage_cnt = right_cnt = scroll_cnt = scroll_dist = 0
    click_times = []
    t_col = "wall_time_ms" if "wall_time_ms" in clicks_df.columns else (
            "timestamp"    if "timestamp"    in clicks_df.columns else None)
    if t_col and len(clicks_df):
        wc = clicks_df[(clicks_df[t_col] >= t_s) & (clicks_df[t_col] <= t_e)]
        for _, ck in wc.iterrows():
            et = str(ck.get("event_type", ""))
            if "scroll" in et.lower():
                scroll_cnt += 1
                try:
                    ed = json.loads(ck.get("event_data", "{}"))
                    scroll_dist += abs(ed.get("dy", ed.get("deltaY", 0)))
                except Exception:
                    pass
            elif "click" in et.lower():
                click_count += 1
                click_times.append(float(ck[t_col]))
                try:
                    ed = json.loads(ck.get("event_data", "{}"))
                    if ed.get("button", 0) == 2:
                        right_cnt += 1
                except Exception:
                    pass
        if len(click_times) >= 3:
            ca = np.sort(click_times)
            for i in range(len(ca) - 2):
                if ca[i+2] - ca[i] <= 2000:
                    rage_click = 1
                    rage_cnt  += 1

    mean_ici = float(np.diff(sorted(click_times)).mean()) if len(click_times) >= 2 else np.nan

    return dict(
        velocity_mean=float(vel.mean()), velocity_max=float(vel.max()),
        velocity_std=float(vel.std()),   acceleration_mean=float(acc.mean()),
        acceleration_max=float(acc.max()), path_length_px=path,
        auc_deviation=auc, x_flips=xf, y_flips=yf,
        idle_ratio=float(idle.mean()), click_count=click_count,
        rage_click_flag=rage_click, rage_click_count=rage_cnt,
        right_click_count=right_cnt, mean_click_interval_ms=mean_ici,
        scroll_count=scroll_cnt, scroll_direction_changes=0,
        total_scroll_distance_px=scroll_dist,
        back_and_forth_score=float(xf + yf),
    )


def _mouse_ts(t0_ms, traj, clicks_df, n_steps=MOUSE_STEPS):
    t_s  = t0_ms + MOUSE_TMIN_MS
    t_e  = t0_ms + MOUSE_TMAX_MS
    grid = np.linspace(t_s, t_e, n_steps)
    m    = 300
    win  = traj[(traj["wall_time_ms"] >= t_s - m) &
                (traj["wall_time_ms"] <= t_e + m)]
    ts   = np.zeros((MOUSE_CHANNELS, n_steps), np.float32)
    if len(win) < 2:
        return ts

    t_raw = win["wall_time_ms"].values
    ts[0] = np.clip(np.interp(grid, t_raw, win["x_norm"].values.astype(float)), 0, 1)
    ts[1] = np.clip(np.interp(grid, t_raw, win["y_norm"].values.astype(float)), 0, 1)
    ts[2] = np.interp(grid, t_raw,
                      np.nan_to_num(win["velocity_px_s"].values.astype(float), nan=0.0)).astype(np.float32)
    ts[3] = np.interp(grid, t_raw,
                      np.nan_to_num(win["acceleration_px_s2"].values.astype(float), nan=0.0)).astype(np.float32)
    ts[4] = np.round(np.interp(grid, t_raw, win["is_idle"].astype(float).values)).astype(np.float32)

    t_col = "wall_time_ms" if "wall_time_ms" in clicks_df.columns else (
            "timestamp"    if "timestamp"    in clicks_df.columns else None)
    if t_col and len(clicks_df):
        click_mask = clicks_df["event_type"].str.contains("click", na=False)
        c_times = clicks_df.loc[click_mask, t_col].values.astype(float)
        for gi, gt in enumerate(grid):
            if np.any((c_times >= gt - DT_MS/2) & (c_times < gt + DT_MS/2)):
                ts[5, gi] = 1.0
        if len(c_times) >= 3:
            cs = np.sort(c_times)
            rage_mask_arr = np.zeros(len(cs), bool)
            for i in range(len(cs) - 2):
                if cs[i+2] - cs[i] <= 2000:
                    rage_mask_arr[i:i+3] = True
            rt = cs[rage_mask_arr]
            for gi, gt in enumerate(grid):
                if np.any((rt >= gt - DT_MS/2) & (rt < gt + DT_MS/2)):
                    ts[6, gi] = 1.0
    return ts


# ── Step 4 ─────────────────────────────────────────────────────────────────────
_MOUSE_COL_ORDER = [
    "label", "velocity_mean", "velocity_max", "velocity_std",
    "acceleration_mean", "acceleration_max", "path_length_px", "auc_deviation",
    "x_flips", "y_flips", "idle_ratio", "click_count", "rage_click_flag",
    "rage_click_count", "right_click_count", "mean_click_interval_ms",
    "scroll_count", "scroll_direction_changes", "total_scroll_distance_px",
    "back_and_forth_score", "wall_time_ms", "eeg_marker", "scenario_type", "phase",
]

def step4_mouse_features(sid: int, markers: pd.DataFrame):
    if markers.empty:
        empty = np.zeros((0, MOUSE_CHANNELS, MOUSE_STEPS), np.float32)
        return pd.DataFrame(), pd.DataFrame(), empty

    proc_dir = ROOT / "data" / "processed" / f"subject_{sid}"
    traj     = pd.read_csv(str(proc_dir / "mouse_trajectory_fixed.csv")).sort_values("wall_time_ms")
    clk_path = ROOT / "data" / "interim" / f"subject_{sid}" / "mouse_clicks.csv"
    clicks   = pd.read_csv(str(clk_path)) if clk_path.exists() else pd.DataFrame()

    rows_erp, rows_cau, ts_list = [], [], []
    for _, row in markers.iterrows():
        t0 = float(row["wall_time_ms"])

        def make_row(tmin_ms, tmax_ms):
            r = _mouse_scalar(t0, traj, clicks, tmin_ms, tmax_ms)
            r.update(label=0, wall_time_ms=t0, eeg_marker=99,
                     scenario_type="control_baseline", phase="control")
            return r

        rows_erp.append(make_row(int(ERP_TMIN*1000),    int(ERP_TMAX*1000)))
        rows_cau.append(make_row(int(CAUSAL_TMIN*1000), int(CAUSAL_TMAX*1000)))
        ts_list.append(_mouse_ts(t0, traj, clicks))

    df_erp = pd.DataFrame(rows_erp).reindex(columns=_MOUSE_COL_ORDER)
    df_cau = pd.DataFrame(rows_cau).reindex(columns=_MOUSE_COL_ORDER)
    ts_arr = np.stack(ts_list, axis=0)

    df_erp.to_csv(str(proc_dir / "mouse_epoch_features_control_erp.csv"),    index=False)
    df_cau.to_csv(str(proc_dir / "mouse_epoch_features_control_causal.csv"), index=False)
    np.save(str(proc_dir / "mouse_timeseries_control_erp.npy"), ts_arr)
    log.info(f"  sub-{sid} mouse: {len(df_erp)} epochs")
    return df_erp, df_cau, ts_arr


# ── Step 5 ─────────────────────────────────────────────────────────────────────
def step5_labram(sid: int, epochs_erp, encoder) -> np.ndarray:
    if epochs_erp is None or len(epochs_erp) == 0:
        return np.zeros((0, 200), np.float32)
    feat_sub = FEAT_DIR / f"subject_{sid}"
    feat_sub.mkdir(parents=True, exist_ok=True)
    embs = encoder.get_embeddings(epochs_erp.get_data(), batch_size=32)
    np.save(str(feat_sub / "eeg_embeddings_control.npy"), embs)
    log.info(f"  sub-{sid} LaBraM: {embs.shape}")
    return embs


# ── Step 6 ─────────────────────────────────────────────────────────────────────
def step6_merge_v2(ctrl: dict) -> dict:
    """ctrl = {sid: {eeg, eye, mouse, markers}}"""
    v_eeg   = np.load(str(FEAT_DIR / "all_eeg_embeddings.npy"))
    v_eye   = np.load(str(FEAT_DIR / "all_eye_timeseries.npy"))
    v_mouse = np.load(str(FEAT_DIR / "all_mouse_timeseries.npy"))
    v_meta  = pd.read_csv(str(FEAT_DIR / "epoch_metadata.csv"))
    N_var   = len(v_eeg)
    log.info(f"  Variant: {N_var}")

    c_eeg_l, c_eye_l, c_mouse_l, c_meta_rows = [], [], [], []
    global_idx = N_var
    per_sub    = {}

    for sid in SUBJECTS:
        d = ctrl.get(sid, {})
        ea = d.get("eeg",   np.zeros((0, 200), np.float32))
        ey = d.get("eye",   np.zeros((0, EYE_CHANNELS, EYE_STEPS), np.float32))
        mo = d.get("mouse", np.zeros((0, MOUSE_CHANNELS, MOUSE_STEPS), np.float32))
        mk = d.get("markers", pd.DataFrame())
        n  = min(len(ea), len(ey), len(mo), len(mk))
        per_sub[sid] = n
        if n == 0:
            continue
        c_eeg_l.append(ea[:n]); c_eye_l.append(ey[:n]); c_mouse_l.append(mo[:n])
        for i, (_, r) in enumerate(mk.head(n).iterrows()):
            c_meta_rows.append(dict(
                global_idx=global_idx, subject_id=sid, epoch_id=int(r["epoch_id"]),
                event_id=99, scenario_name="control_baseline", phase="control",
                wall_time_ms=float(r["wall_time_ms"]),
                eeg_index=i, eye_index=i, mouse_index=i,
                label_frustration=0, label_rage_click=0, label_class=0,
            ))
            global_idx += 1

    if not c_eeg_l:
        log.error("No control data to merge!")
        return {}

    c_eeg = np.concatenate(c_eeg_l)
    c_eye = np.concatenate(c_eye_l)
    c_mou = np.concatenate(c_mouse_l)
    c_met = pd.DataFrame(c_meta_rows)

    all_eeg   = np.concatenate([v_eeg,   c_eeg])
    all_eye   = np.concatenate([v_eye,   c_eye])
    all_mouse = np.concatenate([v_mouse, c_mou])

    labels = pd.DataFrame({
        "global_idx":  range(len(all_eeg)),
        "subject_id":  list(v_meta["subject_id"]) + list(c_met["subject_id"]),
        "label":       [1] * N_var + [0] * len(c_eeg),
        "phase":       ["variant"] * N_var + ["control"] * len(c_eeg),
        "wall_time_ms": list(v_meta["wall_time_ms"]) + list(c_met["wall_time_ms"]),
    })

    v_meta["label_class"] = 1
    all_meta = pd.concat([v_meta, c_met], ignore_index=True)

    np.save(str(FEAT_DIR / "all_eeg_embeddings_v2.npy"),   all_eeg)
    np.save(str(FEAT_DIR / "all_eye_timeseries_v2.npy"),   all_eye)
    np.save(str(FEAT_DIR / "all_mouse_timeseries_v2.npy"), all_mouse)
    labels.to_csv(str(FEAT_DIR / "labels_v2.csv"), index=False)
    all_meta.to_csv(str(FEAT_DIR / "all_eeg_embeddings_v2_metadata.csv"), index=False)

    N_ctl = len(c_eeg)
    log.info(f"  v2: variant={N_var}, control={N_ctl}, total={N_var+N_ctl}")
    log.info(f"  Shapes: EEG={all_eeg.shape}, Eye={all_eye.shape}, Mouse={all_mouse.shape}")
    return {"n_variant": N_var, "n_control": N_ctl, "n_total": N_var+N_ctl,
            "shapes": {"eeg": all_eeg.shape, "eye": all_eye.shape, "mouse": all_mouse.shape},
            "per_subject": per_sub}


# ── Step 7 ─────────────────────────────────────────────────────────────────────
def step7_report(summary: dict):
    eeg_v2 = np.load(str(FEAT_DIR / "all_eeg_embeddings_v2.npy"))
    labs   = pd.read_csv(str(FEAT_DIR / "labels_v2.csv"))
    y      = labs["label"].values

    pca  = PCA(n_components=2, random_state=42)
    e2d  = pca.fit_transform(eeg_v2)
    evr  = pca.explained_variance_ratio_

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {0: "#4C9BE8", 1: "#E84C4C"}
    for cls, label in [(0, f"Control (n={(y==0).sum()})"), (1, f"Variant (n={(y==1).sum()})")]:
        m = y == cls
        ax1.scatter(e2d[m, 0], e2d[m, 1], c=colors[cls], label=label,
                    alpha=0.45, s=10, edgecolors="none")
    ax1.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax1.set_title("EEG Embeddings PCA (v2)")
    ax1.legend(fontsize=9)

    meta = pd.read_csv(str(FEAT_DIR / "all_eeg_embeddings_v2_metadata.csv"))
    sids = sorted(meta["subject_id"].unique())
    xp   = np.arange(len(sids))
    w    = 0.4
    ctrl_c = meta[meta["label_class"] == 0].groupby("subject_id").size()
    var_c  = meta[meta["label_class"] == 1].groupby("subject_id").size()
    ax2.bar(xp - w/2, [ctrl_c.get(s, 0) for s in sids], w, label="Control", color="#4C9BE8")
    ax2.bar(xp + w/2, [var_c.get(s, 0)  for s in sids], w, label="Variant", color="#E84C4C")
    ax2.set_xticks(xp)
    ax2.set_xticklabels([f"S{s}" for s in sids], rotation=45, ha="right")
    ax2.set_ylabel("Epoch count")
    ax2.set_title("Epochs per subject")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(str(REP_DIR / "control_vs_variant_pca.png"), dpi=150, bbox_inches="tight")
    plt.close()

    n_var = summary["n_variant"]
    n_ctl = summary["n_control"]
    n_tot = summary["n_total"]
    per   = summary.get("per_subject", {})
    per_rows = "\n".join(f"| sub-{s} | {per.get(s, 0)} |" for s in SUBJECTS)

    report = f"""# Control Epoch Extension Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Dataset v2 Summary

| Metric | Value |
|--------|-------|
| Variant epochs (label=1) | {n_var} |
| Control epochs (label=0) | {n_ctl} |
| Total | {n_tot} |
| Control/Variant ratio | {n_ctl/n_var:.2f} |
| EEG shape | {summary["shapes"]["eeg"]} |
| Eye shape | {summary["shapes"]["eye"]} |
| Mouse shape | {summary["shapes"]["mouse"]} |

## Per-Subject Control Epoch Counts

| Subject | Control Epochs |
|---------|----------------|
{per_rows}

## Pipeline Parameters

- Control start: {SETTLE_S}s (settling), end: first variant onset − {BUFFER_S}s
- Pseudo-marker step: {PSEUDO_STEP_S}s (event_code=99)
- ERP window: {ERP_TMIN}s to +{ERP_TMAX}s
- Causal window: {CAUSAL_TMIN}s to +{CAUSAL_TMAX}s
- AutoReject: n_interpolate=[1,2,4], random_state=42
- Veli (sub-20): eye features NaN after {VELI_MAX_S}s wall time
- Duru (sub-23): fixation features low-confidence (25 Hz effective sfreq)

## Output Files

Per subject (`data/processed/subject_XX/`):
  control_pseudo_markers.csv, epochs_erp_control-epo.fif,
  epochs_causal_control-epo.fif, eye_epoch_features_control_*.csv,
  mouse_epoch_features_control_*.csv, eye_timeseries_control_erp.npy,
  mouse_timeseries_control_erp.npy

Global (`approach_a/features/`):
  subject_XX/eeg_embeddings_control.npy, all_eeg_embeddings_v2.npy,
  all_eye_timeseries_v2.npy, all_mouse_timeseries_v2.npy,
  labels_v2.csv, all_eeg_embeddings_v2_metadata.csv
"""
    (REP_DIR / "control_extension_report.md").write_text(report)
    log.info(f"  Report → {REP_DIR / 'control_extension_report.md'}")


# ── Step 8 ─────────────────────────────────────────────────────────────────────
def step8_notes(summary: dict):
    n_var = summary["n_variant"]
    n_ctl = summary["n_control"]
    note = f"""
[{datetime.now().strftime("%Y-%m-%d")}] Bölüm 4-6 Eklentisi - Control Phase Epoch Extension:
  Control pseudo-markers generated (event_code=99, every {PSEUDO_STEP_S}s,
  settling={SETTLE_S}s, pre-variant buffer={BUFFER_S}s).
  ERP ({ERP_TMIN}/{ERP_TMAX}s) and Causal ({CAUSAL_TMIN}/{CAUSAL_TMAX}s) epochs cut + AutoReject(n_interpolate=[1,2,4]).
  Eye and mouse features extracted; LaBraM embeddings generated.
  v2 arrays: variant={n_var} (label=1), control={n_ctl} (label=0), total={n_var+n_ctl}.
  Shapes: EEG={summary["shapes"]["eeg"]}, Eye={summary["shapes"]["eye"]}, Mouse={summary["shapes"]["mouse"]}.
  Per-subject: {summary.get("per_subject", {})}.
  Reports: approach_a/reports/control_extension_report.md + control_vs_variant_pca.png
"""
    with open(str(ROOT / "analysis_notes.txt"), "a", encoding="utf-8") as f:
        f.write(note)
    log.info("  Appended to analysis_notes.txt")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    cfg = load_config()
    log.info("Loading LaBraM encoder...")
    encoder = load_labram_base()

    ctrl = {}

    for sid in SUBJECTS:
        log.info(f"\n{'='*60}")
        log.info(f"Subject {sid}")
        log.info(f"{'='*60}")

        raw_start_wall_ms = get_raw_start_wall_ms(sid)

        log.info(f"[1] Pseudo-markers")
        markers = step1_pseudo_markers(sid, raw_start_wall_ms)
        if markers.empty:
            continue

        log.info(f"[2] EEG epochs")
        epo_erp, epo_causal = step2_eeg_epochs(sid, markers)

        survived = _survived_markers(epo_erp, raw_start_wall_ms)
        if survived.empty:
            survived = markers  # no AR rejection case

        info = subject_info(cfg, sid)

        log.info(f"[3] Eye features")
        _, _, eye_ts = step3_eye_features(sid, survived, info, raw_start_wall_ms)

        log.info(f"[4] Mouse features")
        _, _, mouse_ts = step4_mouse_features(sid, survived)

        log.info(f"[5] LaBraM embeddings")
        eeg_embs = step5_labram(sid, epo_erp, encoder)

        n = min(len(eeg_embs), len(eye_ts), len(mouse_ts), len(survived))
        ctrl[sid] = {
            "eeg":     eeg_embs[:n],
            "eye":     eye_ts[:n],
            "mouse":   mouse_ts[:n],
            "markers": survived.head(n),
        }
        log.info(f"  → {n} usable control epochs for sub-{sid}")

    log.info("\n[6] Merging v2 arrays")
    summary = step6_merge_v2(ctrl)
    if not summary:
        log.error("Merge failed - aborting")
        return

    log.info("\n[7] PCA report")
    step7_report(summary)

    log.info("\n[8] Analysis notes")
    step8_notes(summary)

    log.info("\n✓ Control epoch extension complete.")
    log.info(f"  Variant={summary['n_variant']}, Control={summary['n_control']}, "
             f"Total={summary['n_total']}")


if __name__ == "__main__":
    main()
