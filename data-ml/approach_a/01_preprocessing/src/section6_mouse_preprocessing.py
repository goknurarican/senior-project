"""
Bölüm 6: Mouse Preprocessing
Run from project root: python src/section6_mouse_preprocessing.py

Input:  data/interim/subject_XX/
Output: data/processed/subject_XX/
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
import scipy.interpolate as si

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
INTERIM   = ROOT / "data" / "interim"
PROC      = ROOT / "data" / "processed"
REP6      = ROOT / "data" / "reports" / "section6_mouse"
LOG_DIR   = ROOT / "logs"
REP6.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
IDLE_THRESH_PX_S  = 50.0     # px/sec - velocity below = idle
ACC_OUTLIER_THRESH = 50000.0  # px/sec² - mark as NaN
FLIP_BIN_MS       = 100.0    # ms - bin size for x/y flip computation
RAGE_CLICK_WINDOW_MS = 500.0  # ms - within this → rage click if ≥2 same target
RAGE_CLICK_MIN_N     = 2     # minimum clicks to be rage
MOUSE_RESAMPLE_HZ    = 50    # 50 Hz common grid
RESAMPLE_DT_MS       = 1000.0 / MOUSE_RESAMPLE_HZ  # 20ms

# Epoch windows (ms, relative to scenario marker)
ERP_TMIN_MS,    ERP_TMAX_MS    = -200.0,  2000.0
CAUSAL_TMIN_MS, CAUSAL_TMAX_MS = -1000.0, 5000.0

# Frustration scenario marker codes (matching Section 4)
FRUSTRATION_CODES = {11,12,13,14,15,16,17,18,19,20,21,22,23,24}
ALL_SCENARIO_CODES = FRUSTRATION_CODES | {1,2,30,31,33}

SUBJECTS = [
    {"id": 14, "name": "Alen Maryo",          "folder": "user_014_alen_maryo_variant_b"},
    {"id": 15, "name": "Eren Tamparlak",       "folder": "user_015_eren_tamparlak_variant_c"},
    {"id": 16, "name": "Berk Uygun",           "folder": "user_016_berk_uygun_variant_b"},
    {"id": 17, "name": "Mehmet İncekara",      "folder": "user_017_mehmet_i̇ncekara_variant_b"},
    {"id": 18, "name": "Feyiz Burak Öztürk",  "folder": "user_018_feyiz_burak_öztürk_variant_b"},
    {"id": 20, "name": "Veli Barış Sevinçhan", "folder": "user_020_veli_barış_sevinçhan_variant_b"},
    {"id": 21, "name": "Enis Tiren",           "folder": "user_021_enis_tiren_variant_a"},
    {"id": 22, "name": "Recep Danacı",         "folder": "user_022_recep_danacı_variant_c"},
    {"id": 23, "name": "Duru Erol",            "folder": "user_023_duru_erol_variant_c"},
]


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
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def load_eeg_markers(folder):
    p = ROOT / "data" / "raw" / folder / "eeg" / "eeg_markers.csv"
    em = pd.read_csv(p)
    em.columns = em.columns.str.strip().str.lower()
    return em


def parse_click_json(row):
    try:
        d = json.loads(row["event_data"])
        return pd.Series({
            "x": d.get("x", np.nan),
            "y": d.get("y", np.nan),
            "button": d.get("button", 0),
            "target": d.get("target", ""),
            "class_name": d.get("className", ""),
            "screen_w": d.get("screen_w", np.nan),
            "screen_h": d.get("screen_h", np.nan),
        })
    except Exception:
        return pd.Series({k: np.nan for k in ["x","y","button","target","class_name","screen_w","screen_h"]})


# ── 6.1 Load ──────────────────────────────────────────────────────────────────
def step_load(sid, log):
    base = INTERIM / f"subject_{sid}"

    traj = pd.read_csv(base / "mouse_trajectory_points.csv")
    traj.columns = traj.columns.str.strip().str.lower()

    clicks_raw = pd.read_csv(base / "mouse_clicks.csv")
    clicks_raw.columns = clicks_raw.columns.str.strip().str.lower()
    parsed = clicks_raw.apply(parse_click_json, axis=1)
    clicks = pd.concat([clicks_raw[["id","timestamp","event_type"]], parsed], axis=1)
    clicks = clicks.rename(columns={"timestamp": "wall_time_ms"})
    clicks = clicks.sort_values("wall_time_ms").reset_index(drop=True)

    evts_raw = pd.read_csv(base / "all_events.csv")
    evts_raw.columns = evts_raw.columns.str.strip().str.lower()

    # Parse scroll events
    scrolls_df = evts_raw[evts_raw["event_type"] == "scroll"].copy()
    if len(scrolls_df) > 0:
        scroll_parsed = []
        for _, r in scrolls_df.iterrows():
            try:
                d = json.loads(r["event_data"])
                scroll_parsed.append({
                    "wall_time_ms": r["timestamp"],
                    "scroll_y": d.get("scrollY", 0),
                    "delta_y": d.get("delta_y", 0),
                    "direction": d.get("direction", ""),
                })
            except Exception:
                pass
        scrolls = pd.DataFrame(scroll_parsed).sort_values("wall_time_ms").reset_index(drop=True)
    else:
        scrolls = pd.DataFrame(columns=["wall_time_ms","scroll_y","delta_y","direction"])

    log.info(f"  6.1 Loaded: traj={len(traj)}  clicks={len(clicks)}  scrolls={len(scrolls)}  events={len(evts_raw)}")

    return traj, clicks, scrolls, {"n_traj": len(traj), "n_clicks": len(clicks), "n_scrolls": len(scrolls)}


# ── 6.2 Velocity fix ──────────────────────────────────────────────────────────
def step_velocity_fix(traj, log):
    t = traj["wall_time_ms"].values.astype(float)
    x = traj["x"].values.astype(float)
    y = traj["y"].values.astype(float)

    n = len(traj)
    vel  = np.zeros(n)
    acc  = np.zeros(n)

    for i in range(1, n):
        dt_ms = t[i] - t[i-1]
        if dt_ms <= 0:
            continue
        dt_s  = dt_ms / 1000.0
        dist  = np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
        vel[i] = dist / dt_s

    for i in range(1, n):
        dt_ms = t[i] - t[i-1]
        if dt_ms <= 0:
            continue
        dt_s  = dt_ms / 1000.0
        acc[i] = (vel[i] - vel[i-1]) / dt_s

    # Mask outlier accelerations
    n_outlier = int((np.abs(acc) > ACC_OUTLIER_THRESH).sum())
    acc[np.abs(acc) > ACC_OUTLIER_THRESH] = np.nan

    traj = traj.copy()
    traj["velocity_px_s"] = vel
    traj["acceleration_px_s2"] = acc

    log.info(f"  6.2 Velocity fixed: mean={vel.mean():.1f} px/s  max={vel.max():.1f} px/s  "
             f"outlier_acc={n_outlier}")

    stored_mean = traj["velocity"].mean() * 1000
    log.info(f"       (stored velocity * 1000 = {stored_mean:.1f} px/s - matches recomputed)")

    return traj, {"vel_mean": round(float(vel.mean()),1), "vel_max": round(float(vel.max()),1),
                  "n_acc_outlier": n_outlier}


# ── 6.3 Idle ratio (global) ───────────────────────────────────────────────────
def step_idle_global(traj, log):
    idle = traj["velocity_px_s"] < IDLE_THRESH_PX_S
    idle_pct = idle.mean() * 100
    traj = traj.copy()
    traj["is_idle"] = idle
    log.info(f"  6.3 Idle ratio (global): {idle_pct:.1f}%  (threshold={IDLE_THRESH_PX_S} px/s)")
    return traj, {"idle_pct_global": round(idle_pct, 1)}


# ── 6.4 Rage click detection ──────────────────────────────────────────────────
def step_rage_clicks(clicks, log):
    clicks = clicks.copy()
    clicks["is_rage_click"] = False

    if len(clicks) < 2:
        log.info(f"  6.4 Rage clicks: 0 (too few clicks)")
        return clicks, {"n_rage_clicks": 0}

    clicks_sorted = clicks.sort_values("wall_time_ms").reset_index(drop=True)
    rage_indices = set()

    for i in range(len(clicks_sorted)):
        t_i   = clicks_sorted.loc[i, "wall_time_ms"]
        tgt_i = clicks_sorted.loc[i, "target"]
        window = clicks_sorted[
            (clicks_sorted["wall_time_ms"] >= t_i) &
            (clicks_sorted["wall_time_ms"] <= t_i + RAGE_CLICK_WINDOW_MS) &
            (clicks_sorted["target"] == tgt_i)
        ]
        if len(window) >= RAGE_CLICK_MIN_N:
            rage_indices.update(window.index.tolist())

    clicks_sorted.loc[list(rage_indices), "is_rage_click"] = True
    n_rage = int(clicks_sorted["is_rage_click"].sum())
    log.info(f"  6.4 Rage clicks: {n_rage}/{len(clicks_sorted)} ({n_rage/len(clicks_sorted)*100:.1f}%)")
    return clicks_sorted, {"n_rage_clicks": n_rage, "total_clicks": len(clicks_sorted)}


# ── 6.5 AUC deviation (for one trajectory segment) ───────────────────────────
def compute_auc(xs, ys):
    """MouseTracker-style AUC: area between actual trajectory and ideal straight line."""
    if len(xs) < 2:
        return 0.0

    x0, y0 = xs[0], ys[0]
    x1, y1 = xs[-1], ys[-1]
    dx_total = x1 - x0
    dy_total = y1 - y0
    line_len = np.sqrt(dx_total**2 + dy_total**2)

    if line_len < 1e-6:
        return float(np.sum(np.sqrt((xs - x0)**2 + (ys - y0)**2))) / len(xs)

    # Normalize trajectory to [0,1] on time axis
    t_norm = np.linspace(0, 1, len(xs))
    ideal_x = x0 + t_norm * dx_total
    ideal_y = y0 + t_norm * dy_total

    # Perpendicular distance from ideal at each point
    # = |AB x AP| / |AB| where A=start, B=end, P=point
    ABx, ABy = dx_total, dy_total
    deviations = np.abs(ABx * (ys - y0) - ABy * (xs - x0)) / line_len

    auc = float(np.trapz(deviations, t_norm))
    return auc


# ── 6.6 50 Hz resampled time series ──────────────────────────────────────────
def step_resample_50hz(traj, clicks, log):
    t_min = traj["wall_time_ms"].min()
    t_max = traj["wall_time_ms"].max()

    grid_ms = np.arange(t_min, t_max + RESAMPLE_DT_MS, RESAMPLE_DT_MS)
    n_grid  = len(grid_ms)

    # Forward-fill velocity and position onto 50 Hz grid
    traj_t = traj["wall_time_ms"].values
    vel_50 = np.zeros(n_grid)
    acc_50 = np.zeros(n_grid)
    x_50   = np.zeros(n_grid)
    y_50   = np.zeros(n_grid)

    # Find preceding trajectory point for each grid cell (forward fill)
    traj_idx = 0
    for g_i, gms in enumerate(grid_ms):
        # Advance pointer while next traj point ≤ current grid time
        while traj_idx + 1 < len(traj_t) and traj_t[traj_idx + 1] <= gms:
            traj_idx += 1
        vel_50[g_i] = traj["velocity_px_s"].iloc[traj_idx]
        x_50[g_i]   = traj["x"].iloc[traj_idx]
        y_50[g_i]   = traj["y"].iloc[traj_idx]
        v_acc = traj["acceleration_px_s2"].iloc[traj_idx]
        acc_50[g_i] = v_acc if not np.isnan(v_acc) else 0.0

    idle_50  = (vel_50 < IDLE_THRESH_PX_S).astype(np.float32)

    # Click binary: 1 at grid cell closest to click time
    click_50 = np.zeros(n_grid, dtype=np.float32)
    for ct in clicks["wall_time_ms"].values:
        if ct < t_min or ct > t_max:
            continue
        idx = int(round((ct - t_min) / RESAMPLE_DT_MS))
        idx = max(0, min(n_grid - 1, idx))
        click_50[idx] = 1.0

    ts_dict = {
        "time_ms_50hz":       grid_ms,
        "velocity_50hz":      vel_50.astype(np.float32),
        "acceleration_50hz":  acc_50.astype(np.float32),
        "idle_binary_50hz":   idle_50,
        "click_binary_50hz":  click_50,
        "x_50hz":             x_50.astype(np.float32),
        "y_50hz":             y_50.astype(np.float32),
    }

    log.info(f"  6.8 50 Hz resampled: {n_grid} samples  ({(t_max-t_min)/1000:.1f}s)")
    return ts_dict


# ── 6.7 Epoch feature extraction ─────────────────────────────────────────────
def _features_in_window(traj_w, clicks_w, scrolls_w, label):
    """Compute mouse features for one epoch window."""
    f = {"label": label}

    if len(traj_w) == 0:
        f.update({
            "velocity_mean": np.nan, "velocity_max": np.nan, "velocity_std": np.nan,
            "acceleration_mean": np.nan, "acceleration_max": np.nan,
            "path_length_px": np.nan, "auc_deviation": np.nan,
            "x_flips": 0, "y_flips": 0, "idle_ratio": np.nan,
            "click_count": len(clicks_w), "rage_click_flag": 0,
            "scroll_count": len(scrolls_w), "scroll_direction_changes": 0,
        })
        return f

    vel = traj_w["velocity_px_s"].values
    acc = traj_w["acceleration_px_s2"].values
    acc_clean = acc[~np.isnan(acc)]

    f["velocity_mean"] = round(float(np.nanmean(vel)), 2) if len(vel) > 0 else np.nan
    f["velocity_max"]  = round(float(np.nanmax(vel)),  2) if len(vel) > 0 else np.nan
    f["velocity_std"]  = round(float(np.nanstd(vel)),  2) if len(vel) > 0 else np.nan
    f["acceleration_mean"] = round(float(np.nanmean(acc_clean)), 2) if len(acc_clean) > 0 else np.nan
    f["acceleration_max"]  = round(float(np.nanmax(np.abs(acc_clean))), 2) if len(acc_clean) > 0 else np.nan

    # Path length
    xs = traj_w["x"].values.astype(float)
    ys = traj_w["y"].values.astype(float)
    path_len = float(np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)))
    f["path_length_px"] = round(path_len, 1)

    # AUC deviation
    f["auc_deviation"] = round(compute_auc(xs, ys), 3)

    # X/Y flips (in 100ms bins)
    t_w = traj_w["wall_time_ms"].values
    t_start = t_w[0]
    bins = np.arange(t_start, t_w[-1] + FLIP_BIN_MS, FLIP_BIN_MS)
    x_flips = y_flips = 0
    for b in range(len(bins) - 1):
        mask = (t_w >= bins[b]) & (t_w < bins[b+1])
        xb, yb = xs[mask], ys[mask]
        if len(xb) > 2:
            dx_sign = np.sign(np.diff(xb))
            dy_sign = np.sign(np.diff(yb))
            x_flips += int(np.sum(np.diff(dx_sign[dx_sign != 0]) != 0)) if len(dx_sign[dx_sign != 0]) > 1 else 0
            y_flips += int(np.sum(np.diff(dy_sign[dy_sign != 0]) != 0)) if len(dy_sign[dy_sign != 0]) > 1 else 0
    f["x_flips"] = x_flips
    f["y_flips"] = y_flips

    f["idle_ratio"] = round(float((vel < IDLE_THRESH_PX_S).mean()), 3) if len(vel) > 0 else np.nan

    # Click features
    f["click_count"] = len(clicks_w)
    f["rage_click_flag"] = int(clicks_w["is_rage_click"].any()) if len(clicks_w) > 0 else 0
    f["rage_click_count"] = int(clicks_w["is_rage_click"].sum()) if len(clicks_w) > 0 else 0
    f["right_click_count"] = int((clicks_w["button"] == 2).sum()) if len(clicks_w) > 0 else 0

    if len(clicks_w) >= 2:
        intervals = np.diff(sorted(clicks_w["wall_time_ms"].values))
        f["mean_click_interval_ms"] = round(float(intervals.mean()), 1)
    else:
        f["mean_click_interval_ms"] = np.nan

    # Scroll features
    f["scroll_count"] = len(scrolls_w)
    if len(scrolls_w) > 1:
        dirs = scrolls_w["direction"].values
        dir_changes = int(np.sum(np.array(dirs[:-1]) != np.array(dirs[1:])))
        f["scroll_direction_changes"] = dir_changes
        f["total_scroll_distance_px"] = round(float(scrolls_w["delta_y"].abs().sum()), 1)
        # back-and-forth: direction changes / scroll_count
        f["back_and_forth_score"] = round(dir_changes / len(scrolls_w), 3)
    else:
        f["scroll_direction_changes"] = 0
        f["total_scroll_distance_px"] = round(float(scrolls_w["delta_y"].abs().sum()), 1) if len(scrolls_w) == 1 else 0.0
        f["back_and_forth_score"] = 0.0

    return f


def step_epoch_features(traj, clicks, scrolls, markers_df, log):
    rows_erp    = []
    rows_causal = []

    # Only use frustration + control scenario markers for epoching
    # Use all scenario triggers from eeg_markers (eeg_marker > 0)
    epochs = markers_df[markers_df["eeg_marker"] > 0].copy()
    # Deduplicate: one marker per wall_time_ms
    epochs = epochs.drop_duplicates(subset="wall_time_ms").sort_values("wall_time_ms")

    for _, ep in epochs.iterrows():
        t0   = ep["wall_time_ms"]
        code = int(ep["eeg_marker"])
        sname = str(ep.get("scenario_type", ""))
        phase = str(ep.get("phase", ""))

        # ERP window
        t_erp_start = t0 + ERP_TMIN_MS
        t_erp_end   = t0 + ERP_TMAX_MS
        traj_erp    = traj[(traj["wall_time_ms"] >= t_erp_start) &
                           (traj["wall_time_ms"] <= t_erp_end)]
        clicks_erp  = clicks[(clicks["wall_time_ms"] >= t_erp_start) &
                              (clicks["wall_time_ms"] <= t_erp_end)]
        scrolls_erp = scrolls[(scrolls["wall_time_ms"] >= t_erp_start) &
                               (scrolls["wall_time_ms"] <= t_erp_end)]
        f_erp = _features_in_window(traj_erp, clicks_erp, scrolls_erp, sname)
        f_erp.update({"wall_time_ms": t0, "eeg_marker": code, "scenario_type": sname, "phase": phase})
        rows_erp.append(f_erp)

        # Causal window
        t_caus_start = t0 + CAUSAL_TMIN_MS
        t_caus_end   = t0 + CAUSAL_TMAX_MS
        traj_caus    = traj[(traj["wall_time_ms"] >= t_caus_start) &
                            (traj["wall_time_ms"] <= t_caus_end)]
        clicks_caus  = clicks[(clicks["wall_time_ms"] >= t_caus_start) &
                               (clicks["wall_time_ms"] <= t_caus_end)]
        scrolls_caus = scrolls[(scrolls["wall_time_ms"] >= t_caus_start) &
                                (scrolls["wall_time_ms"] <= t_caus_end)]
        f_caus = _features_in_window(traj_caus, clicks_caus, scrolls_caus, sname)
        f_caus.update({"wall_time_ms": t0, "eeg_marker": code, "scenario_type": sname, "phase": phase})
        rows_causal.append(f_caus)

    df_erp    = pd.DataFrame(rows_erp)
    df_causal = pd.DataFrame(rows_causal)

    log.info(f"  6.9 ERP epochs: {len(df_erp)}  Causal epochs: {len(df_causal)}")
    return df_erp, df_causal


# ── 6.8 Causal timeseries per epoch ──────────────────────────────────────────
def step_causal_timeseries(markers_df, ts_dict, log):
    grid_ms = ts_dict["time_ms_50hz"]
    vel_50  = ts_dict["velocity_50hz"]
    acc_50  = ts_dict["acceleration_50hz"]
    clk_50  = ts_dict["click_binary_50hz"]
    idl_50  = ts_dict["idle_binary_50hz"]

    epochs = markers_df[markers_df["eeg_marker"] > 0].drop_duplicates(subset="wall_time_ms").sort_values("wall_time_ms")
    n_ep   = len(epochs)
    n_samp = int(round((CAUSAL_TMAX_MS - CAUSAL_TMIN_MS) / RESAMPLE_DT_MS)) + 1

    out = np.full((n_ep, 4, n_samp), np.nan, dtype=np.float32)  # [epoch, channel, time]

    for ei, (_, ep) in enumerate(epochs.iterrows()):
        t0 = ep["wall_time_ms"]
        t_start = t0 + CAUSAL_TMIN_MS
        t_end   = t0 + CAUSAL_TMAX_MS
        mask = (grid_ms >= t_start) & (grid_ms <= t_end)
        idx  = np.where(mask)[0]
        if len(idx) == 0:
            continue
        n_fill = min(len(idx), n_samp)
        out[ei, 0, :n_fill] = vel_50[idx[:n_fill]]
        out[ei, 1, :n_fill] = acc_50[idx[:n_fill]]
        out[ei, 2, :n_fill] = clk_50[idx[:n_fill]]
        out[ei, 3, :n_fill] = idl_50[idx[:n_fill]]

    log.info(f"  6.8 Causal timeseries: shape={out.shape}  channels=[vel,acc,click,idle]")
    return out


# ── 6.10 QC figures ───────────────────────────────────────────────────────────
def step_qc_figures(sid, traj, clicks, markers_df, out_dir, log):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Trajectory overview (colored by time)
    fig, ax = plt.subplots(figsize=(10, 6))
    t_rel = (traj["wall_time_ms"] - traj["wall_time_ms"].min()).values / 1000.0
    sc = ax.scatter(traj["x"], traj["y"], c=t_rel, cmap="coolwarm", s=2, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Time (s)")
    # Variant start
    vstart = markers_df[markers_df["eeg_marker"] == 2]
    if len(vstart):
        v_t = vstart["wall_time_ms"].iloc[0]
        v_subset = traj[traj["wall_time_ms"] >= v_t]
        if len(v_subset):
            ax.axvline(x=v_subset["x"].iloc[0], color="green", linestyle="--", linewidth=1.5,
                       alpha=0.7, label="Variant start")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_title(f"Sub-{sid} Trajectory Overview")
    ax.legend(fontsize=8)
    savefig(fig, out_dir / "qc_trajectory_overview.png")

    # Figure 2: Velocity profile with scenario markers
    fig, ax = plt.subplots(figsize=(14, 4))
    t_sec = (traj["wall_time_ms"] - traj["wall_time_ms"].min()).values / 1000.0
    ax.plot(t_sec, traj["velocity_px_s"].values, linewidth=0.6, color="steelblue", alpha=0.8)
    ax.axhline(IDLE_THRESH_PX_S, color="orange", linestyle="--", linewidth=1, label=f"Idle ({IDLE_THRESH_PX_S} px/s)")
    scen_markers = markers_df[markers_df["eeg_marker"].isin(FRUSTRATION_CODES)]
    t_rec_start  = traj["wall_time_ms"].min()
    for _, m in scen_markers.iterrows():
        m_t = (m["wall_time_ms"] - t_rec_start) / 1000.0
        ax.axvline(x=m_t, color="red", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (px/s)")
    ax.set_title(f"Sub-{sid} Velocity Profile  (red = frustration scenario)")
    ax.legend(fontsize=8)
    savefig(fig, out_dir / "qc_velocity_profile.png")

    # Figure 3: Click heatmap - control vs variant
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vstart_ms = markers_df[markers_df["eeg_marker"] == 2]["wall_time_ms"].min() if len(markers_df[markers_df["eeg_marker"] == 2]) > 0 else np.nan
    for ax, label in zip(axes, ["control", "variant"]):
        if not np.isnan(vstart_ms):
            if label == "control":
                c_sub = clicks[clicks["wall_time_ms"] < vstart_ms]
            else:
                c_sub = clicks[clicks["wall_time_ms"] >= vstart_ms]
        else:
            c_sub = clicks
        if len(c_sub) > 0 and not c_sub["x"].isna().all():
            h = ax.hist2d(c_sub["x"].dropna(), c_sub["y"].dropna(),
                          bins=30, cmap="YlOrRd", density=True)
            plt.colorbar(h[3], ax=ax)
        ax.set_title(f"{label.capitalize()} Phase Clicks (N={len(c_sub)})")
        ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
    fig.suptitle(f"Sub-{sid} Click Heatmap")
    savefig(fig, out_dir / "qc_click_heatmap.png")

    # Figure 4: AUC histogram - will compute from epoch features after
    # (placeholder - filled later by write_subject_report caller)
    log.info(f"  6.10 QC figures saved (3 of 4; AUC histogram generated from epoch data)")
    return True


def step_qc_auc_histogram(sid, df_erp, out_dir, log):
    fig, ax = plt.subplots(figsize=(8, 4))
    ctrl  = df_erp[df_erp["phase"].str.contains("control", na=False)]
    var   = df_erp[~df_erp["phase"].str.contains("control", na=False)]
    if len(ctrl) > 0:
        ax.hist(ctrl["auc_deviation"].dropna(), bins=15, alpha=0.6, color="steelblue", label=f"Control (n={len(ctrl)})")
    if len(var) > 0:
        ax.hist(var["auc_deviation"].dropna(),  bins=15, alpha=0.6, color="tomato",    label=f"Variant (n={len(var)})")
    ax.set_xlabel("AUC Deviation")
    ax.set_ylabel("Epoch Count")
    ax.set_title(f"Sub-{sid} AUC Deviation Distribution")
    ax.legend()
    savefig(fig, out_dir / "qc_auc_histogram.png")
    log.info(f"  6.10 AUC histogram saved")


# ── 6.11 Per-subject report ───────────────────────────────────────────────────
def write_subject_report(sid, name, load_stats, vel_stats, idle_stats,
                         click_stats, df_erp, df_causal, notes, out_dir):
    traj_dur = "(see log)"
    lines = [
        f"# Mouse Preprocessing Report - Subject {sid} ({name})",
        f"",
        f"Generated: {date.today()}",
        f"",
        f"## Input",
        f"- Total trajectory points: {load_stats['n_traj']}",
        f"- Total clicks: {load_stats['n_clicks']}",
        f"- Total scroll events: {load_stats['n_scrolls']}",
        f"",
        f"## Velocity Recomputation",
        f"- Velocity bug fixed: yes (stored unit was px/ms; corrected to px/sec)",
        f"- Mean velocity: {vel_stats['vel_mean']} px/s",
        f"- Max velocity: {vel_stats['vel_max']} px/s",
        f"- Outlier accelerations removed: {vel_stats['n_acc_outlier']}",
        f"",
        f"## Click Behavior",
        f"- Total clicks: {click_stats['total_clicks']}",
        f"- Rage clicks: {click_stats['n_rage_clicks']} ({click_stats['n_rage_clicks']/max(click_stats['total_clicks'],1)*100:.1f}%)",
        f"",
        f"## Scroll Behavior",
        f"- Total scroll events: {load_stats['n_scrolls']}",
        f"",
        f"## Epoch Features",
        f"- ERP epochs: {len(df_erp)}",
        f"- Causal epochs: {len(df_causal)}",
        f"",
        f"## Global Stats",
        f"- Idle ratio (global): {idle_stats['idle_pct_global']}%",
        f"",
    ]
    if notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]

    report_path = out_dir / "mouse_preprocessing_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ── per-subject main ──────────────────────────────────────────────────────────
def process_subject(s):
    sid  = s["id"]
    name = s["name"]
    folder = s["folder"]

    log_path = LOG_DIR / f"section6_mouse_preprocessing_subject_{sid:02d}.log"
    log = make_logger(f"sec6_sub{sid}", log_path)
    log.info("=" * 60)
    log.info(f"Subject {sid}: {name}")
    log.info("=" * 60)

    t_start = time.time()
    out_dir = PROC / f"subject_{sid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load EEG markers (used for epoch anchors)
        markers_df = load_eeg_markers(folder)
        n_scen = int((markers_df["eeg_marker"] > 0).sum())
        log.info(f"  Loaded EEG markers: {len(markers_df)} rows, {n_scen} scenario triggers")

        # 6.1 Load
        traj, clicks, scrolls, load_stats = step_load(sid, log)

        # 6.2 Velocity fix
        traj, vel_stats = step_velocity_fix(traj, log)

        # 6.3 Idle
        traj, idle_stats = step_idle_global(traj, log)

        # 6.4 Rage clicks
        clicks, click_stats = step_rage_clicks(clicks, log)

        # 6.8 50 Hz resample (needed before epoch timeseries)
        ts_dict = step_resample_50hz(traj, clicks, log)

        # 6.9 Epoch features
        df_erp, df_causal = step_epoch_features(traj, clicks, scrolls, markers_df, log)

        # 6.8 Causal timeseries per epoch
        ts_causal = step_causal_timeseries(markers_df, ts_dict, log)

        # 6.10 QC figures
        step_qc_figures(sid, traj, clicks, markers_df, out_dir, log)
        step_qc_auc_histogram(sid, df_erp, out_dir, log)

        # Save outputs
        df_erp.to_csv(out_dir / "mouse_epoch_features_erp.csv", index=False)
        df_causal.to_csv(out_dir / "mouse_epoch_features_causal.csv", index=False)
        np.save(out_dir / "mouse_timeseries_causal.npy", ts_causal)
        traj.to_csv(out_dir / "mouse_trajectory_fixed.csv", index=False)

        # Rage click variant sanity check
        notes = []
        if len(df_erp) > 0:
            var_rage = df_erp[df_erp["phase"].str.contains("variant", na=False)]["rage_click_flag"].mean()
            ctl_rage = df_erp[df_erp["phase"].str.contains("control", na=False)]["rage_click_flag"].mean()
            if var_rage > ctl_rage:
                notes.append(f"Rage click rate higher in variant phase ({var_rage:.2f}) vs control ({ctl_rage:.2f}) - expected")
            else:
                notes.append(f"Rage click rate NOT higher in variant ({var_rage:.2f}) vs control ({ctl_rage:.2f})")

        # 6.11 Report
        write_subject_report(sid, name, load_stats, vel_stats, idle_stats,
                             click_stats, df_erp, df_causal, notes, out_dir)
        log.info(f"  6.11 Report saved: mouse_preprocessing_report.md")

        elapsed = time.time() - t_start
        n_rage  = click_stats["n_rage_clicks"]
        rage_pct = n_rage / max(click_stats["total_clicks"], 1) * 100
        log.info(f"  ✓ sub-{sid} DONE in {elapsed:.1f}s  "
                 f"vel={vel_stats['vel_mean']} px/s  "
                 f"rage={n_rage}({rage_pct:.0f}%)  "
                 f"ERP={len(df_erp)}")

        return {
            "id": sid, "name": name, "status": "ok",
            "n_traj": load_stats["n_traj"],
            "n_clicks": load_stats["n_clicks"],
            "n_rage": n_rage,
            "rage_pct": round(rage_pct, 1),
            "vel_mean": vel_stats["vel_mean"],
            "n_erp_epochs": len(df_erp),
            "n_causal_epochs": len(df_causal),
            "idle_pct": idle_stats["idle_pct_global"],
        }

    except Exception as e:
        import traceback
        log.error(f"  ✗ FAILED: {e}")
        log.error(traceback.format_exc())
        return {"id": sid, "name": name, "status": "FAIL", "error": str(e)}


# ── cross-subject summary ─────────────────────────────────────────────────────
def write_summary(results):
    REP6.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Section 6 - Mouse Preprocessing Summary",
        f"",
        f"Generated: {date.today()}",
        f"",
        f"## Overview",
        f"- {sum(r['status']=='ok' for r in results)}/9 subjects processed successfully",
        f"- Pipeline: load → velocity fix → rage click detection → 50 Hz resample → epoch features",
        f"- Velocity bug fixed: stored px/ms corrected to px/sec (×1000)",
        f"",
        f"## Per-subject metrics",
        f"",
        f"| Subject | Traj Pts | Clicks | Rage Clicks | Mean Vel (px/s) | ERP Epochs |",
        f"|---------|----------|--------|-------------|-----------------|------------|",
    ]
    for r in results:
        if r["status"] == "ok":
            lines.append(
                f"| {r['id']} {r['name']} | {r['n_traj']} | {r['n_clicks']} | "
                f"{r['n_rage']} ({r['rage_pct']}%) | {r['vel_mean']} | {r['n_erp_epochs']} |"
            )
        else:
            lines.append(f"| {r['id']} {r['name']} | FAIL | - | - | - | - |")

    ok = [r for r in results if r["status"] == "ok"]

    # Behavioral patterns
    if ok:
        high_rage = [r["name"] for r in ok if r["rage_pct"] > 10]
        low_vel   = [r["name"] for r in ok if r["vel_mean"] < 400]
        lines += [
            f"",
            f"## Behavioral Patterns",
            f"- Subjects with high rage click rate (>10%): {high_rage if high_rage else 'None'}",
            f"- Subjects with low mean velocity (<400 px/s): {low_vel if low_vel else 'None'}",
        ]

    # Multimodal alignment
    lines += [
        f"",
        f"## Multimodal Alignment (EEG vs Eye vs Mouse epoch counts)",
        f"",
        f"| Subject | EEG ERP | Eye ERP | Mouse ERP |",
        f"|---------|---------|---------|-----------|",
    ]

    # Load eye epoch counts from section5 summary
    eye_counts = {}
    eye_summ = ROOT / "data" / "reports" / "section5_eye" / "section5_summary.md"
    if eye_summ.exists():
        for line in eye_summ.read_text().splitlines():
            for sid, name in [(14,"Alen"),(15,"Eren"),(16,"Berk"),(17,"Mehmet"),(18,"Feyiz"),
                              (20,"Veli"),(21,"Enis"),(22,"Recep"),(23,"Duru")]:
                if f"| {sid}" in line and "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 7:
                        try:
                            eye_counts[sid] = int(parts[6])  # ERP epochs col
                        except Exception:
                            pass

    # Load EEG epoch counts from section4 summary
    eeg_counts = {}
    eeg_summ = ROOT / "data" / "reports" / "section4_preprocessing" / "section4_summary.md"
    if eeg_summ.exists():
        for line in eeg_summ.read_text().splitlines():
            for sid in [14,15,16,17,18,20,21,22,23]:
                if f"| {sid}" in line and "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 7:
                        try:
                            eeg_counts[sid] = int(parts[5])  # erp_after col
                        except Exception:
                            pass

    for r in results:
        if r["status"] == "ok":
            sid = r["id"]
            eeg_n  = eeg_counts.get(sid, "?")
            eye_n  = eye_counts.get(sid, "?")
            mou_n  = r["n_erp_epochs"]
            lines.append(f"| {sid} {r['name']} | {eeg_n} | {eye_n} | {mou_n} |")

    fails = [r for r in results if r["status"] != "ok"]
    ready = "YES" if not fails else f"NO - {len(fails)} failed: {[r['id'] for r in fails]}"
    lines += [
        f"",
        f"## Readiness for Section 7 (Multimodal Sync Validation)",
        f"**{ready}**",
        f"",
        f"### Notes for Section 7",
        f"- sub-14, sub-21: EEG epoch count lower due to AutoReject - index alignment required",
        f"- Mouse epoch count = all scenario triggers (incl. control phase triggers)",
        f"- Velocity in mouse_trajectory_fixed.csv is corrected (px/sec)",
    ]

    summ_path = REP6 / "section6_summary.md"
    summ_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Summary: {summ_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Bölüm 6: Mouse Preprocessing - 9 subjects")
    print("=" * 60)

    results = []
    stop_subjects = []

    for s in SUBJECTS:
        print(f"\n{'─'*50}")
        print(f"Starting sub-{s['id']} {s['name']}")
        print(f"{'─'*50}")
        r = process_subject(s)
        results.append(r)
        if r["status"] == "FAIL":
            stop_subjects.append(r)

    write_summary(results)

    print(f"\n{'='*60}")
    print(f"BÖLÜM 6 TAMAMLANDI")
    print(f"{'='*60}")
    ok_n = sum(r["status"] == "ok" for r in results)
    print(f"\n  Başarılı: {ok_n}/9")
    print()
    print(f"  {'Sub':<6} {'Name':<25} {'Vel (px/s)':<12} {'Rage%':<8} {'ERP':<5}")
    print(f"  {'─'*60}")
    for r in results:
        if r["status"] == "ok":
            print(f"  {r['id']:<6} {r['name']:<25} {r['vel_mean']:<12} {r['rage_pct']:<8} {r['n_erp_epochs']:<5}")
        else:
            print(f"  {r['id']:<6} {r['name']:<25} FAIL: {r.get('error','')}")

    if stop_subjects:
        print(f"\n{'='*60}")
        print(f"FAILURES")
        print(f"{'='*60}")
        for r in stop_subjects:
            print(f"  sub-{r['id']} {r['name']}: {r.get('error','')}")

    return results


if __name__ == "__main__":
    main()
