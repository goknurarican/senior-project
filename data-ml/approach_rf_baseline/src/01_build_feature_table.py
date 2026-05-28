from pathlib import Path
import re
import warnings

import yaml
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}


def load_config():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_output_dir(cfg):
    out = Path(cfg["output_dir"])
    if out.is_absolute():
        return out

    # repo root: senior-project/
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / out


def marker_to_int(value):
    """
    Handles values like 11, "11", "S11", "S11_slow_image".
    """
    if pd.isna(value):
        return None

    s = str(value).strip()
    match = re.search(r"(\d+)", s)
    if not match:
        return None
    return int(match.group(1))


def sequence_match(plv_markers, align_df):
    """
    Forward-search marker sequence matching.
    This is used because eye/mouse epoch counts do not necessarily match EEG epoch counts.
    """
    if "event_id" not in align_df.columns:
        raise ValueError("alignment file is missing required column: event_id")

    align_markers = align_df["event_id"].apply(marker_to_int).values

    matches = []
    align_ptr = 0

    for eeg_marker in plv_markers:
        found = False

        while align_ptr < len(align_markers):
            if align_markers[align_ptr] == int(eeg_marker):
                matches.append(align_ptr)
                align_ptr += 1
                found = True
                break
            align_ptr += 1

        if not found:
            matches.append(-1)

    return matches


def band_power_features(epoch_data, sfreq, ch_names):
    """
    epoch_data: shape (n_channels, n_times), EEG in volts.
    Returns summary EEG features for Random Forest.
    """
    nperseg = min(int(sfreq * 1.0), epoch_data.shape[1])
    freqs, psd = welch(epoch_data, fs=sfreq, nperseg=nperseg, axis=1)

    eps = 1e-30
    feats = {}

    linear_band_power = {}

    for band_name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)

        if not mask.any():
            bp = np.zeros(epoch_data.shape[0], dtype=float)
        else:
            bp = psd[:, mask].mean(axis=1)

        linear_band_power[band_name] = bp

        log_bp = np.log10(bp + eps)
        feats[f"eeg_{band_name}_mean"] = float(np.nanmean(log_bp))
        feats[f"eeg_{band_name}_std"] = float(np.nanstd(log_bp))
        feats[f"eeg_{band_name}_max"] = float(np.nanmax(log_bp))
        feats[f"eeg_{band_name}_min"] = float(np.nanmin(log_bp))

    theta = linear_band_power["theta"]
    alpha = linear_band_power["alpha"]
    beta = linear_band_power["beta"]

    theta_mean = np.nanmean(theta)
    alpha_mean = np.nanmean(alpha)
    beta_mean = np.nanmean(beta)

    feats["eeg_engagement_index"] = float(beta_mean / (alpha_mean + theta_mean + eps))
    feats["eeg_theta_beta_ratio"] = float(theta_mean / (beta_mean + eps))
    feats["eeg_alpha_theta_ratio"] = float(alpha_mean / (theta_mean + eps))

    # Frontal Alpha Asymmetry: log(F4 alpha) - log(F3 alpha)
    name_to_idx = {name: i for i, name in enumerate(ch_names)}
    if "F3" in name_to_idx and "F4" in name_to_idx:
        f3_alpha = alpha[name_to_idx["F3"]]
        f4_alpha = alpha[name_to_idx["F4"]]
        feats["eeg_faa_f4_minus_f3"] = float(np.log(f4_alpha + eps) - np.log(f3_alpha + eps))
    else:
        feats["eeg_faa_f4_minus_f3"] = np.nan

    # Frontal theta summary
    frontal_names = ["F3", "F4", "Fz", "Fp1", "Fp2", "FC1", "FC2"]
    frontal_idx = [name_to_idx[c] for c in frontal_names if c in name_to_idx]

    if frontal_idx:
        feats["eeg_frontal_theta_mean"] = float(np.nanmean(theta[frontal_idx]))
        feats["eeg_frontal_alpha_mean"] = float(np.nanmean(alpha[frontal_idx]))
        feats["eeg_frontal_beta_mean"] = float(np.nanmean(beta[frontal_idx]))
    else:
        feats["eeg_frontal_theta_mean"] = np.nan
        feats["eeg_frontal_alpha_mean"] = np.nan
        feats["eeg_frontal_beta_mean"] = np.nan

    return feats


def entropy_from_values(x, bins=10):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 3:
        return 0.0

    if np.nanstd(x) < 1e-12:
        return 0.0

    hist, _ = np.histogram(x, bins=bins)
    p = hist.astype(float) / max(hist.sum(), 1)

    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def eye_features(eye_arr, eye_idx):
    feats = {}

    if eye_idx is None or np.isnan(eye_idx):
        return {
            "eye_available": 0,
            "eye_valid_ratio": 0,
            "eye_mean": np.nan,
            "eye_std": np.nan,
            "eye_range": np.nan,
            "eye_entropy": np.nan,
        }

    eye_idx = int(eye_idx)

    if eye_idx < 0 or eye_idx >= len(eye_arr):
        return {
            "eye_available": 0,
            "eye_valid_ratio": 0,
            "eye_mean": np.nan,
            "eye_std": np.nan,
            "eye_range": np.nan,
            "eye_entropy": np.nan,
        }

    x = np.asarray(eye_arr[eye_idx], dtype=float).reshape(-1)
    finite = np.isfinite(x)

    feats["eye_available"] = 1
    feats["eye_valid_ratio"] = float(finite.mean()) if len(x) else 0.0

    if finite.sum() == 0:
        feats["eye_mean"] = np.nan
        feats["eye_std"] = np.nan
        feats["eye_range"] = np.nan
        feats["eye_entropy"] = np.nan
        return feats

    xf = x[finite]
    feats["eye_mean"] = float(np.nanmean(xf))
    feats["eye_std"] = float(np.nanstd(xf))
    feats["eye_range"] = float(np.nanmax(xf) - np.nanmin(xf))
    feats["eye_entropy"] = entropy_from_values(xf, bins=10)

    return feats


def mouse_features(mouse_arr, mouse_idx):
    if mouse_idx is None or np.isnan(mouse_idx):
        return empty_mouse_features()

    mouse_idx = int(mouse_idx)

    if mouse_idx < 0 or mouse_idx >= len(mouse_arr):
        return empty_mouse_features()

    x = np.asarray(mouse_arr[mouse_idx], dtype=float)

    # Expected shape: (4, 301)
    if x.ndim != 2 or x.shape[0] < 4:
        flat = x.reshape(-1)
        finite = flat[np.isfinite(flat)]
        return {
            "mouse_available": 1,
            "mouse_velocity_mean": float(np.nanmean(finite)) if len(finite) else np.nan,
            "mouse_velocity_max": float(np.nanmax(finite)) if len(finite) else np.nan,
            "mouse_velocity_std": float(np.nanstd(finite)) if len(finite) else np.nan,
            "mouse_acc_abs_mean": np.nan,
            "mouse_acc_abs_max": np.nan,
            "mouse_click_count": np.nan,
            "mouse_click_rate": np.nan,
            "mouse_idle_ratio": np.nan,
            "mouse_path_proxy": np.nan,
        }

    vel = x[0]
    acc = x[1]
    click = x[2]
    idle = x[3]

    vel_f = vel[np.isfinite(vel)]
    acc_f = acc[np.isfinite(acc)]
    click_f = click[np.isfinite(click)]
    idle_f = idle[np.isfinite(idle)]

    dt = 1.0 / 50.0

    return {
        "mouse_available": 1,
        "mouse_velocity_mean": float(np.nanmean(vel_f)) if len(vel_f) else np.nan,
        "mouse_velocity_max": float(np.nanmax(vel_f)) if len(vel_f) else np.nan,
        "mouse_velocity_std": float(np.nanstd(vel_f)) if len(vel_f) else np.nan,
        "mouse_acc_abs_mean": float(np.nanmean(np.abs(acc_f))) if len(acc_f) else np.nan,
        "mouse_acc_abs_max": float(np.nanmax(np.abs(acc_f))) if len(acc_f) else np.nan,
        "mouse_click_count": float(np.nansum(click_f)) if len(click_f) else 0.0,
        "mouse_click_rate": float(np.nanmean(click_f)) if len(click_f) else 0.0,
        "mouse_idle_ratio": float(np.nanmean(idle_f)) if len(idle_f) else np.nan,
        "mouse_path_proxy": float(np.nansum(np.abs(vel_f)) * dt) if len(vel_f) else np.nan,
    }


def empty_mouse_features():
    return {
        "mouse_available": 0,
        "mouse_velocity_mean": np.nan,
        "mouse_velocity_max": np.nan,
        "mouse_velocity_std": np.nan,
        "mouse_acc_abs_mean": np.nan,
        "mouse_acc_abs_max": np.nan,
        "mouse_click_count": np.nan,
        "mouse_click_rate": np.nan,
        "mouse_idle_ratio": np.nan,
        "mouse_path_proxy": np.nan,
    }


def load_labels_if_available(data_root, cfg):
    approach_a_name = cfg["input_dirs"].get("approach_a", "06_for_approach_a")
    labels_path = data_root / approach_a_name / "labels.csv"

    if not labels_path.exists():
        print(f"[WARN] labels.csv not found: {labels_path}")
        return None

    labels = pd.read_csv(labels_path)
    print(f"[OK] labels.csv loaded: {labels_path} rows={len(labels)}")

    if "global_idx" not in labels.columns:
        print("[WARN] labels.csv has no global_idx column.")
        return None

    if "label_rage_click" not in labels.columns:
        print("[WARN] labels.csv has no label_rage_click column.")
        return None

    return labels[["global_idx", "label_rage_click"]].copy()


def main():
    cfg = load_config()

    data_root = Path(cfg["data_root"])
    eeg_dir = data_root / cfg["input_dirs"]["eeg"]
    eye_dir = data_root / cfg["input_dirs"]["eye"]
    mouse_dir = data_root / cfg["input_dirs"]["mouse"]
    align_dir = data_root / cfg["input_dirs"]["alignment"]

    output_dir = resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_df = load_labels_if_available(data_root, cfg)

    rows = []
    global_idx = 0

    print("=" * 80)
    print("BUILD RF FEATURE TABLE")
    print("=" * 80)

    scenario_markers = {int(k): v for k, v in cfg["scenario_markers"].items()}

    for sid in cfg["subjects"]:
        print(f"\n[subject_{sid}]")

        eeg_path = eeg_dir / f"subject_{sid}" / "epochs_causal-epo.fif"
        eye_path = eye_dir / f"subject_{sid}" / "eye_timeseries_causal.npy"
        mouse_path = mouse_dir / f"subject_{sid}" / "mouse_timeseries_causal.npy"
        align_path = align_dir / f"alignment_master_subject_{sid}.csv"

        epochs = mne.read_epochs(eeg_path, preload=True, verbose=False)
        eeg_data = epochs.get_data()
        sfreq = float(epochs.info["sfreq"])
        ch_names = list(epochs.ch_names)
        eeg_markers = epochs.events[:, 2].astype(int)

        eye_arr = np.load(eye_path, mmap_mode="r")
        mouse_arr = np.load(mouse_path, mmap_mode="r")
        align_df = pd.read_csv(align_path)

        matches = sequence_match(eeg_markers, align_df)

        print(f"  EEG epochs : {len(eeg_data)}")
        print(f"  Eye epochs : {len(eye_arr)}")
        print(f"  Mouse eps  : {len(mouse_arr)}")
        print(f"  Align rows : {len(align_df)}")
        print(f"  Matched    : {sum(m >= 0 for m in matches)} / {len(matches)}")

        for local_idx in range(len(eeg_data)):
            marker = int(eeg_markers[local_idx])
            scenario_name = scenario_markers.get(marker, f"marker_{marker}")

            base = {
                "global_idx": global_idx,
                "subject_id": sid,
                "local_epoch_idx": local_idx,
                "eeg_marker": marker,
                "scenario_name": scenario_name,
                "is_scenario_marker": int(marker in scenario_markers),
            }

            align_pos = matches[local_idx]
            if align_pos >= 0:
                ar = align_df.iloc[align_pos]

                base["alignment_row"] = int(align_pos)
                base["wall_time_ms"] = ar.get("wall_time_ms", np.nan)
                base["phase"] = ar.get("phase", np.nan)

                eye_ok = str(ar.get("eye_available", "no")).lower() == "yes"
                mouse_ok = str(ar.get("mouse_available", "no")).lower() == "yes"

                base["eye_epoch_idx"] = ar.get("eye_index", np.nan) if eye_ok else np.nan
                base["mouse_epoch_idx"] = ar.get("mouse_index", np.nan) if mouse_ok else np.nan
                base["alignment_status"] = "matched"
            else:
                base["alignment_row"] = -1
                base["wall_time_ms"] = np.nan
                base["phase"] = np.nan
                base["eye_epoch_idx"] = np.nan
                base["mouse_epoch_idx"] = np.nan
                base["alignment_status"] = "unmatched"

            eeg_feats = band_power_features(eeg_data[local_idx], sfreq, ch_names)
            eye_feats = eye_features(eye_arr, base["eye_epoch_idx"])
            mouse_feats = mouse_features(mouse_arr, base["mouse_epoch_idx"])

            row = {}
            row.update(base)
            row.update(eeg_feats)
            row.update(eye_feats)
            row.update(mouse_feats)

            rows.append(row)
            global_idx += 1

    df = pd.DataFrame(rows)

    if labels_df is not None:
        df = df.merge(labels_df, on="global_idx", how="left")
    else:
        df["label_rage_click"] = np.nan

    # Keep only rows that are scenario epochs and have a label.
    clean = df[
        (df["is_scenario_marker"] == 1) &
        (df["alignment_status"] == "matched") &
        (df["label_rage_click"].notna())
    ].copy()

    all_path = output_dir / "feature_table_all.csv"
    clean_path = output_dir / "feature_table_clean.csv"
    summary_path = output_dir / "feature_table_summary.txt"

    df.to_csv(all_path, index=False)
    clean.to_csv(clean_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("RF FEATURE TABLE SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"All rows: {len(df)}\n")
        f.write(f"Clean rows: {len(clean)}\n")
        f.write(f"Subjects: {sorted(clean['subject_id'].unique().tolist()) if len(clean) else []}\n")
        if "label_rage_click" in clean.columns and len(clean):
            f.write(f"Positive rage_click: {int(clean['label_rage_click'].sum())}\n")
            f.write(f"Positive rate: {clean['label_rage_click'].mean() * 100:.2f}%\n")
        f.write("\nRows per subject:\n")
        if len(clean):
            f.write(str(clean.groupby("subject_id").size()) + "\n")
        f.write("\nLabels per subject:\n")
        if len(clean):
            f.write(str(clean.groupby("subject_id")["label_rage_click"].agg(["count", "sum", "mean"])) + "\n")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved all table  : {all_path}")
    print(f"Saved clean table: {clean_path}")
    print(f"Saved summary    : {summary_path}")
    print(f"All rows         : {len(df)}")
    print(f"Clean rows       : {len(clean)}")

    if len(clean):
        print(f"Positive labels  : {int(clean['label_rage_click'].sum())}")
        print(f"Positive rate    : {clean['label_rage_click'].mean() * 100:.2f}%")
        print("\nPer-subject label summary:")
        print(clean.groupby("subject_id")["label_rage_click"].agg(["count", "sum", "mean"]))


if __name__ == "__main__":
    main()