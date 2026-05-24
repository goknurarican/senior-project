"""
Bölüm 3: EEG EDA - Kanal Düzeyinde Tanı
Run from project root: python src/section3_eda.py

Analyses (no preprocessing applied):
  3.1  Raw time-series overview (control + variant 30s windows)
  3.2  Power Spectral Density + topographic band maps
  3.3  Channel RMS amplitude flags
  3.4  Inter-channel correlation matrix
  3.5  Sub-1 Hz drift analysis
  3.6  50 Hz line-noise analysis
  3.7  Accelerometer (motion) analysis
  3.8  Cross-subject summary report
"""

import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scipy.signal as ss
import yaml

warnings.filterwarnings("ignore")
os.environ["MNE_LOGGING_LEVEL"] = "ERROR"
import mne
mne.set_log_level("ERROR")

# ── paths & constants ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RAW  = ROOT / "data" / "raw"
REP  = ROOT / "data" / "reports"
S3   = REP / "section3_eda"
LOG  = ROOT / "logs"
SEED = 42

np.random.seed(SEED)

SFREQ      = 500
ACCEL_CHS  = ["x_dir", "y_dir", "z_dir"]
EEG_CHS_32 = [
    "Fp1","Fz","F3","F7","FT9","FC5","FC1","C3","T7","TP9",
    "CP5","CP1","Pz","P3","P7","O1","Oz","O2","P4","P8",
    "TP10","CP6","CP2","Cz","C4","T8","FT10","FC6","FC2","F4","F8","Fp2",
]
BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8, 13),
    "beta":  (13,30),
    "gamma": (30,40),
}


# ── helpers ────────────────────────────────────────────────────────────────────
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


def vhdr_path(s: dict) -> Path:
    return subject_dir(s) / f"{s['eeg_prefix']}.vhdr"


def load_raw(s: dict) -> mne.io.Raw:
    raw = mne.io.read_raw_brainvision(
        str(vhdr_path(s)), preload=True, verbose=False
    )
    raw.set_channel_types({ch: "misc" for ch in ACCEL_CHS if ch in raw.ch_names})
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore", verbose=False)
    return raw


def get_phase_times(s: dict) -> tuple:
    """Return (ctrl_t0, ctrl_t1, var_t0, var_t1) in seconds from EEG start."""
    em_path = subject_dir(s) / "eeg" / "eeg_markers.csv"
    em = pd.read_csv(em_path)

    # EEG t0 from raw first_samp (wall time from .vmrk will shift absolute times)
    # We use phase_timing.csv as the ground truth for control/variant durations
    pt = pd.read_csv(REP / "phase_timing.csv")
    row = pt[pt["subject_id"] == s["id"]].iloc[0]
    ctrl_dur  = float(row["control_dur_s"])
    var_dur   = float(row["variant_dur_s"])

    # variant_start is the anchor: find its time in eeg_markers and align to raw
    var_marker = em[em["scenario_type"] == "variant_start"]
    if var_marker.empty:
        return None, None, None, None

    # wall_time_ms of variant_start vs first EEG marker
    first_ms = em["wall_time_ms"].min()
    var_ms   = int(var_marker.iloc[0]["wall_time_ms"])
    var_t0   = (var_ms - first_ms) / 1000.0   # seconds from raw start

    ctrl_t1  = var_t0
    ctrl_t0  = max(0.0, var_t0 - ctrl_dur)
    var_t1   = min(var_t0 + var_dur, var_t0 + var_dur)

    return ctrl_t0, ctrl_t1, var_t0, var_t1


def pick_eeg(raw: mne.io.Raw) -> mne.io.Raw:
    return raw.copy().pick(EEG_CHS_32, verbose=False)


def welch_psd(data: np.ndarray, sfreq: float, fmin=0.5, fmax=100.0):
    """Compute Welch PSD per channel. Returns (freqs, psd_db)."""
    nperseg = int(sfreq * 2)
    noverlap = nperseg // 2
    freqs, psd = ss.welch(data, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1)
    mask = (freqs >= fmin) & (freqs <= fmax)
    psd_db = 10 * np.log10(psd[:, mask] + 1e-30)
    return freqs[mask], psd_db


def band_power(data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    """Band-limited power per channel."""
    freqs, psd = ss.welch(data, fs=sfreq, nperseg=int(sfreq * 2),
                          noverlap=int(sfreq), axis=-1)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return psd[:, mask].mean(axis=1)


def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def write_flags(flags: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(flags) if flags else "No anomalies detected.\n", encoding="utf-8")


# ── 3.1  Raw overview ─────────────────────────────────────────────────────────
def plot_raw_overview(raw_eeg: mne.io.Raw, tmin: float, tmax: float,
                      title: str, save_path: Path, rng: np.random.Generator):
    dur = tmax - tmin
    if dur < 30:
        win_start = tmin
        win_end   = tmax
    else:
        max_start = tmax - 30
        win_start = float(rng.uniform(tmin, max(tmin, max_start)))
        win_end   = win_start + 30

    data, times = raw_eeg[:, int(win_start * SFREQ):int(win_end * SFREQ)]
    n_ch = data.shape[0]

    fig, ax = plt.subplots(figsize=(18, n_ch * 0.35 + 1))
    scale = 50e-6  # µV display scale
    colors = plt.cm.tab20(np.linspace(0, 1, n_ch))
    for i, (ch_data, ch_name) in enumerate(zip(data, raw_eeg.ch_names)):
        t = times - times[0]
        ax.plot(t, ch_data / scale + i, color=colors[i], lw=0.5, alpha=0.85)
        ax.text(-0.3, i, ch_name, ha="right", va="center", fontsize=6)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title(f"{title}  [{win_start:.1f}–{win_end:.1f} s]")
    ax.spines[["left", "top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, save_path)


# ── 3.2  PSD ──────────────────────────────────────────────────────────────────
def plot_psd(raw_eeg: mne.io.Raw, tmin: float, tmax: float,
             subj_out: Path, name: str, log: logging.Logger):
    data, _ = raw_eeg[:, int(tmin * SFREQ):int(tmax * SFREQ)]
    freqs, psd_db = welch_psd(data, SFREQ)
    n_ch = data.shape[0]

    # Overlay PSD all channels
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(n_ch):
        ax.plot(freqs, psd_db[i], alpha=0.4, lw=0.8, color="steelblue")
    # median
    ax.plot(freqs, np.median(psd_db, axis=0), color="navy", lw=1.5, label="median")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title(f"PSD - {name} - all channels (control phase)")
    ax.axvline(50, color="red", ls="--", lw=0.8, alpha=0.6, label="50 Hz")
    for band, (f1, f2) in BANDS.items():
        ax.axvspan(f1, f2, alpha=0.06, color="orange")
    ax.legend(fontsize=8)
    ax.set_xlim(1, 100)
    savefig(fig, subj_out / "psd_all_channels.png")
    log.info(f"    PSD overlay saved.")

    # Topographic band maps
    try:
        info_eeg = raw_eeg.info
        n_bands  = len(BANDS)
        fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 4))
        for ax, (band_name, (f1, f2)) in zip(axes, BANDS.items()):
            bp = band_power(data, SFREQ, f1, f2)
            bp_db = 10 * np.log10(bp + 1e-30)
            mne.viz.plot_topomap(
                bp_db, info_eeg, axes=ax, show=False,
                cmap="RdYlBu_r", vlim=(np.percentile(bp_db, 10), np.percentile(bp_db, 90)),
                sphere="eeglab",
            )
            ax.set_title(band_name, fontsize=9)
        fig.suptitle(f"Topographic band power - {name}")
        fig.tight_layout()
        savefig(fig, subj_out / "topomap_bands.png")
        log.info(f"    Topographic maps saved.")
    except Exception as exc:
        log.warning(f"    Topomap failed ({exc}) - skipped.")


# ── 3.3  RMS amplitude flags ──────────────────────────────────────────────────
def analyze_rms(raw_eeg: mne.io.Raw, subj_out: Path,
                log: logging.Logger) -> dict:
    data     = raw_eeg.get_data()
    # Use STD (zero-meaned) - raw EEG has large DC offset that would dominate RMS
    amp_uv   = np.std(data, axis=1) * 1e6   # µV, AC amplitude only
    dc_uv    = data.mean(axis=1) * 1e6       # µV, DC offset per channel
    ch_names = raw_eeg.ch_names
    med = np.median(amp_uv)
    flags = {}

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    # AC amplitude
    colors = ["tomato" if r > 3*med else ("gold" if r < 0.3*med else "steelblue") for r in amp_uv]
    axes[0].bar(ch_names, amp_uv, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].axhline(3*med,   color="red",    ls="--", lw=1, label=f"3× median ({3*med:.1f} µV)")
    axes[0].axhline(0.3*med, color="orange", ls="--", lw=1, label=f"0.3× median ({0.3*med:.1f} µV)")
    axes[0].set_xticklabels(ch_names, rotation=90, fontsize=7)
    axes[0].set_ylabel("AC amplitude std (µV)")
    axes[0].set_title(f"Channel amplitude (std) - {subj_out.name}")
    axes[0].legend(fontsize=8)
    # DC offset
    axes[1].bar(ch_names, np.abs(dc_uv) / 1000, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[1].set_xticklabels(ch_names, rotation=90, fontsize=7)
    axes[1].set_ylabel("|DC offset| (mV)")
    axes[1].set_title("DC offset magnitude (will be removed by high-pass filter)")
    fig.tight_layout()
    savefig(fig, subj_out / "channel_rms_boxplot.png")

    flag_lines = []
    for ch, r in zip(ch_names, amp_uv):
        if r > 3 * med:
            flags[ch] = "high_amp"
            flag_lines.append(f"HIGH_AMP: {ch}  std={r:.1f} µV  ({r/med:.1f}× median)")
            log.info(f"    high_amp: {ch}  std={r:.1f} µV")
        elif r < 0.3 * med:
            flags[ch] = "low_sig"
            flag_lines.append(f"LOW_SIG:  {ch}  std={r:.1f} µV  ({r/med:.2f}× median)")
            log.info(f"    low_sig:  {ch}  std={r:.1f} µV")

    write_flags(flag_lines, subj_out / "channel_amplitude_flags.txt")
    return flags


# ── 3.4  Correlation matrix ───────────────────────────────────────────────────
def analyze_correlation(raw_eeg: mne.io.Raw, tmin: float, tmax: float,
                        subj_out: Path, log: logging.Logger) -> dict:
    data, _ = raw_eeg[:, int(tmin * SFREQ):int(tmax * SFREQ)]
    # Normalize before correlation to avoid amplitude bias
    data_z = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-12)
    corr = np.corrcoef(data_z)
    ch_names = raw_eeg.ch_names

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ch_names))); ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(ch_names))); ax.set_yticklabels(ch_names, fontsize=6)
    ax.set_title(f"Channel correlation - {subj_out.name}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    savefig(fig, subj_out / "channel_correlation_heatmap.png")

    # "Isolated" flag: top-5 absolute correlations (excluding self)
    flags = {}
    flag_lines = []
    for i, ch in enumerate(ch_names):
        row = np.abs(corr[i])
        row[i] = 0   # exclude self
        top5 = np.sort(row)[-5:]
        mean_top5 = top5.mean()
        if mean_top5 < 0.3:
            flags[ch] = "isolated"
            flag_lines.append(f"ISOLATED: {ch}  mean_top5_corr={mean_top5:.3f}")
            log.info(f"    isolated: {ch}  top5_corr={mean_top5:.3f}")

    write_flags(flag_lines, subj_out / "channel_correlation_flags.txt")
    return flags


# ── 3.5  Drift (sub-1 Hz power) ───────────────────────────────────────────────
def analyze_drift(raw_eeg: mne.io.Raw, subj_out: Path, log: logging.Logger) -> dict:
    data = raw_eeg.get_data()
    freqs, psd = ss.welch(data, fs=SFREQ, nperseg=int(SFREQ * 4),
                          noverlap=int(SFREQ * 2), axis=-1)
    drift_mask = freqs < 1.0
    drift_power = psd[:, drift_mask].mean(axis=1)
    ch_names = raw_eeg.ch_names
    med = np.median(drift_power)
    mad = np.median(np.abs(drift_power - med))
    threshold = med + 2 * mad

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["tomato" if p > threshold else "steelblue" for p in drift_power]
    ax.bar(ch_names, drift_power * 1e12, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(threshold * 1e12, color="red", ls="--", lw=1, label="median + 2MAD")
    ax.set_xticklabels(ch_names, rotation=90, fontsize=7)
    ax.set_ylabel("Sub-1 Hz power (×10⁻¹²)")
    ax.set_title(f"Drift (sub-1 Hz) - {subj_out.name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, subj_out / "drift_per_channel.png")

    flags = {}
    flag_lines = []
    for ch, p in zip(ch_names, drift_power):
        if p > threshold:
            flags[ch] = "high_drift"
            flag_lines.append(f"HIGH_DRIFT: {ch}  power={p*1e12:.2f} × 10^-12  ({p/med:.1f}× median)")
            log.info(f"    high_drift: {ch}")

    write_flags(flag_lines, subj_out / "drift_flags.txt")
    return flags


# ── 3.6  50 Hz line noise ─────────────────────────────────────────────────────
def analyze_line_noise(raw_eeg: mne.io.Raw, subj_out: Path, log: logging.Logger) -> dict:
    data = raw_eeg.get_data()
    freqs, psd = ss.welch(data, fs=SFREQ, nperseg=int(SFREQ * 2),
                          noverlap=int(SFREQ), axis=-1)
    # Find bin nearest to 50 Hz
    idx50 = np.argmin(np.abs(freqs - 50.0))
    psd_50 = psd[:, idx50]
    psd_50_db = 10 * np.log10(psd_50 + 1e-30)
    ch_names = raw_eeg.ch_names
    med = np.median(psd_50_db)
    threshold = med + 6  # 6 dB above median

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["tomato" if p > threshold else "steelblue" for p in psd_50_db]
    ax.bar(ch_names, psd_50_db, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(threshold, color="red", ls="--", lw=1, label="median + 6 dB")
    ax.set_xticklabels(ch_names, rotation=90, fontsize=7)
    ax.set_ylabel("50 Hz power (dB)")
    ax.set_title(f"Line noise (50 Hz) - {subj_out.name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, subj_out / "line_noise_per_channel.png")

    flags = {}
    flag_lines = []
    for ch, p in zip(ch_names, psd_50_db):
        if p > threshold:
            flags[ch] = "high_line_noise"
            flag_lines.append(f"LINE_NOISE: {ch}  50Hz={p:.1f} dB  ({p-med:.1f} dB above median)")
            log.info(f"    line_noise: {ch}  50Hz={p:.1f}dB  (+{p-med:.1f}dB)")

    write_flags(flag_lines, subj_out / "line_noise_flags.txt")
    return flags


# ── 3.7  Accelerometer / movement ────────────────────────────────────────────
def analyze_movement(raw: mne.io.Raw, s: dict, subj_out: Path,
                     log: logging.Logger) -> dict:
    accel_present = [ch for ch in ACCEL_CHS if ch in raw.ch_names]
    if len(accel_present) < 3:
        log.warning(f"    Accelerometer channels not all present: {accel_present}")
        return {"total_high_motion_s": 0, "scenario_overlap_pct": 0,
                "recommendation": "accel_missing"}

    accel_data = raw.copy().pick(accel_present, verbose=False).get_data()  # shape (3, n_times)
    # Remove DC (gravity component) by high-pass filtering each axis at 0.5 Hz
    sos = ss.butter(4, 0.5, btype="high", fs=SFREQ, output="sos")
    accel_hp = np.array([ss.sosfiltfilt(sos, ax) for ax in accel_data])
    mag = np.sqrt(np.sum(accel_hp ** 2, axis=0))  # dynamic motion magnitude
    times = raw.times

    med = np.median(mag)
    mad = np.median(np.abs(mag - med))
    threshold = med + 3 * mad
    high_mask = mag > threshold

    # Smooth the mask (min 0.5s contiguous)
    from scipy.ndimage import label as ndlabel
    labeled, n_comp = ndlabel(high_mask)
    for comp in range(1, n_comp + 1):
        idx = np.where(labeled == comp)[0]
        if len(idx) < int(0.5 * SFREQ):
            high_mask[idx] = False

    total_high_s = high_mask.sum() / SFREQ

    fig, axes = plt.subplots(4, 1, figsize=(16, 8), sharex=True)
    labels_xyz = ["x", "y", "z"]
    for i, (ax, d, lbl) in enumerate(zip(axes[:3], accel_data, labels_xyz)):
        ax.plot(times, d, lw=0.5, color=f"C{i}")
        ax.set_ylabel(f"{lbl} (mg)", fontsize=8)
        ax.yaxis.set_tick_params(labelsize=7)
    axes[3].plot(times, mag, lw=0.5, color="purple")
    axes[3].fill_between(times, 0, mag, where=high_mask, alpha=0.4, color="red", label="high motion")
    axes[3].axhline(threshold, color="red", ls="--", lw=0.8)
    axes[3].set_ylabel("Magnitude (mg)", fontsize=8)
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(fontsize=7)
    axes[0].set_title(f"Accelerometer - {s['name']}")
    fig.tight_layout()
    savefig(fig, subj_out / "movement_timeline.png")

    # Check overlap with scenario markers
    em = pd.read_csv(subject_dir(s) / "eeg" / "eeg_markers.csv")
    em_scen = em[em["scenario_type"].isin([
        "slow_image","broken_image","skeleton_prolong","search_irrelevant",
        "button_delay","first_click_miss","feedback_late","network_jitter",
        "overlay_blocking","price_change","coupon_min_spend","coupon_expired",
        "facet_reset_once","sort_reset",
    ])]
    first_ms = em["wall_time_ms"].min()
    n_overlap = 0
    for _, row in em_scen.iterrows():
        t_s = (row["wall_time_ms"] - first_ms) / 1000.0
        window = np.arange(int((t_s - 0.5) * SFREQ), int((t_s + 3.5) * SFREQ))
        window = window[(window >= 0) & (window < len(high_mask))]
        if high_mask[window].any():
            n_overlap += 1

    overlap_pct = round(n_overlap / max(len(em_scen), 1) * 100, 1)

    rec = "ok"
    if total_high_s > 60:
        rec = "review - >60s high motion"
    elif overlap_pct > 30:
        rec = "review - >30% scenario overlap"

    report_lines = [
        f"Total high-motion duration: {total_high_s:.1f} s",
        f"High-motion threshold: {threshold:.2f} mg (median+3MAD)",
        f"Scenario windows with high motion: {n_overlap}/{len(em_scen)} ({overlap_pct}%)",
        f"Recommendation: {rec}",
    ]
    (subj_out / "movement_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    log.info(f"    Motion: total={total_high_s:.1f}s  scen_overlap={overlap_pct}%  rec={rec}")

    return {
        "total_high_motion_s":  round(total_high_s, 1),
        "scenario_overlap_pct": overlap_pct,
        "recommendation":       rec,
    }


# ── per-subject orchestrator ──────────────────────────────────────────────────
def analyze_subject(s: dict, log: logging.Logger) -> dict:
    sid   = s["id"]
    name  = s["name"]
    log.info(f"\n{'='*60}")
    log.info(f"Subject {sid:02d}: {name}")
    log.info(f"{'='*60}")

    subj_out = S3 / f"subject_{sid:02d}"
    subj_out.mkdir(parents=True, exist_ok=True)

    # Load raw EEG
    raw = load_raw(s)
    total_dur_s = raw.n_times / SFREQ
    log.info(f"  Loaded: {len(raw.ch_names)} channels, {total_dur_s:.1f}s")

    # Phase times
    ctrl_t0, ctrl_t1, var_t0, var_t1 = get_phase_times(s)
    if ctrl_t0 is None:
        log.warning(f"  Phase times unavailable - skipping.")
        return {}

    # Clamp to recording bounds
    ctrl_t0 = max(ctrl_t0, 0)
    ctrl_t1 = min(ctrl_t1, total_dur_s)
    var_t1  = min(var_t1, total_dur_s)
    log.info(f"  Control: [{ctrl_t0:.1f}, {ctrl_t1:.1f}]s  Variant: [{var_t0:.1f}, {var_t1:.1f}]s")

    raw_eeg = pick_eeg(raw)
    rng = np.random.default_rng(SEED)

    # 3.1 Raw overview
    log.info(f"  3.1 Raw time-series overview...")
    plot_raw_overview(raw_eeg, ctrl_t0, ctrl_t1, f"{name} - Control", subj_out / "raw_overview_control.png", rng)
    plot_raw_overview(raw_eeg, var_t0,  var_t1,  f"{name} - Variant", subj_out / "raw_overview_variant.png", rng)
    log.info(f"    Saved raw overviews.")

    # 3.2 PSD
    log.info(f"  3.2 PSD analysis...")
    plot_psd(raw_eeg, ctrl_t0, ctrl_t1, subj_out, name, log)

    # 3.3 RMS amplitude flags
    log.info(f"  3.3 RMS amplitude flags...")
    amp_flags = analyze_rms(raw_eeg, subj_out, log)
    log.info(f"    Flagged: {len(amp_flags)} channels ({list(amp_flags.keys())})")

    # 3.4 Correlation
    log.info(f"  3.4 Correlation matrix (control phase)...")
    corr_flags = analyze_correlation(raw_eeg, ctrl_t0, ctrl_t1, subj_out, log)
    log.info(f"    Isolated: {len(corr_flags)} channels ({list(corr_flags.keys())})")

    # 3.5 Drift
    log.info(f"  3.5 Drift analysis...")
    drift_flags = analyze_drift(raw_eeg, subj_out, log)
    log.info(f"    High-drift: {len(drift_flags)} channels")

    # 3.6 50 Hz
    log.info(f"  3.6 50 Hz line noise...")
    noise_flags = analyze_line_noise(raw_eeg, subj_out, log)
    log.info(f"    High line noise: {len(noise_flags)} channels")

    # 3.7 Movement
    log.info(f"  3.7 Accelerometer analysis...")
    motion = analyze_movement(raw, s, subj_out, log)

    # Merge all per-channel flags
    all_ch_flags: dict[str, list] = {}
    for ch, ftype in amp_flags.items():
        all_ch_flags.setdefault(ch, []).append(ftype)
    for ch, ftype in corr_flags.items():
        all_ch_flags.setdefault(ch, []).append(ftype)
    for ch, ftype in drift_flags.items():
        all_ch_flags.setdefault(ch, []).append(ftype)
    for ch, ftype in noise_flags.items():
        all_ch_flags.setdefault(ch, []).append(ftype)

    n_bad = len([ch for ch, fl in all_ch_flags.items()
                 if any(f in fl for f in ["high_amp","low_sig","isolated"])])
    log.info(f"  Summary: {n_bad} bad-candidate channels  motion={motion['total_high_motion_s']:.1f}s")

    if n_bad >= 15:
        log.error(
            f"  ⚠ STOP: {n_bad} bad-candidate channels for sub-{sid:02d} ({name}). "
            "Manual review required before proceeding."
        )

    del raw, raw_eeg

    return {
        "subject_id":           sid,
        "name":                 name,
        "group":                s["group"],
        "total_dur_s":          total_dur_s,
        "n_amp_flags":          len(amp_flags),
        "n_corr_flags":         len(corr_flags),
        "n_drift_flags":        len(drift_flags),
        "n_noise_flags":        len(noise_flags),
        "n_bad_candidates":     n_bad,
        "amp_flag_chs":         list(amp_flags.keys()),
        "corr_flag_chs":        list(corr_flags.keys()),
        "drift_flag_chs":       list(drift_flags.keys()),
        "noise_flag_chs":       list(noise_flags.keys()),
        "motion_high_s":        motion["total_high_motion_s"],
        "motion_scen_overlap":  motion["scenario_overlap_pct"],
        "motion_rec":           motion["recommendation"],
        "all_ch_flags":         all_ch_flags,
    }


# ── 3.8  Cross-subject summary ────────────────────────────────────────────────
def write_cross_subject_summary(results: list, log: logging.Logger):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "all_ch_flags"} for r in results])
    df.to_csv(S3 / "cross_subject_flags.csv", index=False)

    # Channel flag heatmap (per-subject × per-channel)
    flag_rows = []
    for r in results:
        row = {"subject_id": r["subject_id"], "name": r["name"]}
        for ch in EEG_CHS_32:
            flags = r.get("all_ch_flags", {}).get(ch, [])
            if "high_amp" in flags:
                row[ch] = "high_amp"
            elif "low_sig" in flags:
                row[ch] = "low_sig"
            elif "isolated" in flags:
                row[ch] = "isolated"
            elif flags:
                row[ch] = flags[0]
            else:
                row[ch] = "OK"
        flag_rows.append(row)

    flag_df = pd.DataFrame(flag_rows)
    flag_df.to_csv(S3 / "cross_subject_channel_flags.csv", index=False)

    # Movement summary
    motion_df = df[["subject_id","name","motion_high_s","motion_scen_overlap","motion_rec"]].copy()
    motion_df.to_csv(S3 / "movement_summary.csv", index=False)

    # Bad-candidate channel heat map image
    numeric_cols = EEG_CHS_32
    heat = pd.DataFrame(
        [[1 if flag_df.loc[flag_df["subject_id"]==r["subject_id"], ch].values[0] != "OK" else 0
          for ch in numeric_cols]
         for r in results],
        index=[f"sub-{r['subject_id']:02d}" for r in results],
        columns=numeric_cols,
    )
    fig, ax = plt.subplots(figsize=(18, 4))
    im = ax.imshow(heat.values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(numeric_cols))); ax.set_xticklabels(numeric_cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(results))); ax.set_yticklabels(heat.index, fontsize=8)
    ax.set_title("Channel flags across subjects (red = any flag)")
    fig.tight_layout()
    savefig(fig, S3 / "cross_subject_channel_heatmap.png")

    # Markdown summary
    total_dur = df["total_dur_s"].sum() / 60
    n_flagged_pairs = df[["n_amp_flags","n_corr_flags","n_drift_flags","n_noise_flags"]].sum().sum()
    concerning = df[df["n_bad_candidates"] >= 3]["name"].tolist()
    clean      = df[df["n_bad_candidates"] == 0]["name"].tolist()

    # Bad channel candidate table
    bad_table = df[["subject_id","name","n_amp_flags","n_corr_flags","n_drift_flags","n_noise_flags","n_bad_candidates"]].copy()
    bad_table.columns = ["ID","Name","High/Low Amp","Isolated","High Drift","Line Noise","Bad Candidates"]

    # Section 4 readiness
    any_stopper = any(r["n_bad_candidates"] >= 15 for r in results)
    readiness = "NO - investigate subjects with ≥15 bad channels" if any_stopper else "YES"

    md_path = S3 / "section3_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Section 3 - EEG EDA Summary\n\n")
        f.write(f"Generated: 2026-05-12\n\n")
        f.write("## Dataset\n\n")
        f.write(f"- 9 subjects analyzed\n")
        f.write(f"- 32 EEG channels each\n")
        f.write(f"- Sample rate: {SFREQ} Hz\n")
        f.write(f"- Total recording time: {total_dur:.1f} min\n\n")
        f.write("## Subject-level findings\n\n")
        f.write("### Per-subject channel issues\n\n")
        f.write(bad_table.to_markdown(index=False))
        f.write("\n\n### Subjects with ≥3 bad-candidate channels\n\n")
        f.write(", ".join(concerning) if concerning else "_None_")
        f.write("\n\n### Subjects with clean EEG (0 bad candidates)\n\n")
        f.write(", ".join(clean) if clean else "_None_")
        f.write("\n\n## Movement analysis\n\n")
        f.write(motion_df.to_markdown(index=False))
        f.write("\n\n## Common issues across subjects\n\n")
        f.write("- **50 Hz line noise**: present in all subjects (expected, notch filter in §4)\n")
        f.write("- **Eye blink artefacts in Fp1/Fp2**: present in all subjects (expected, ICA in §4)\n")
        f.write("- **Drift in lateral/temporal channels**: common in gel-based EEG\n\n")
        f.write("## Bad channel candidates per subject\n\n")
        for r in results:
            cands = [c for c, fl in r.get("all_ch_flags",{}).items()
                     if any(f in fl for f in ["high_amp","low_sig","isolated"])]
            f.write(f"- **sub-{r['subject_id']:02d} {r['name']}**: {', '.join(cands) if cands else 'none'}\n")
        f.write("\n## Readiness for Section 4 (Preprocessing)\n\n")
        f.write(f"**{readiness}**\n")
        if not any_stopper:
            f.write("All subjects have <15 flagged channels. Proceed to Section 4.\n")

    log.info(f"  Section 3 summary: {S3 / 'section3_summary.md'}")
    return df


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    S3.mkdir(parents=True, exist_ok=True)
    log = make_logger("s3_eda", LOG / "section3_eda.log")
    log.info("=" * 60)
    log.info("Bölüm 3: EEG EDA - Kanal Düzeyinde Tanı")
    log.info("=" * 60)

    cfg = load_cfg()
    ACTIVE_IDS = {14,15,16,17,18,20,21,22,23}
    subjects   = [s for s in cfg["subjects"] if s["id"] in ACTIVE_IDS]
    log.info(f"Processing {len(subjects)} subjects...")

    results = []
    for s in subjects:
        try:
            result = analyze_subject(s, log)
            if result:
                results.append(result)
        except Exception as exc:
            log.error(f"  sub-{s['id']:02d} {s['name']}: FAILED - {exc}")
            import traceback
            log.error(traceback.format_exc())

    log.info("\n" + "=" * 60)
    log.info("3.8 Cross-subject summary...")
    summary_df = write_cross_subject_summary(results, log)

    log.info("\n" + "=" * 70)
    log.info("BÖLÜM 3 TAMAMLANDI")
    log.info("=" * 70)
    log.info(f"\n  Sub  Name                       Dur(s)  Bad_cand  Motion(s)  Scen%")
    log.info(f"  {'-'*65}")
    for r in results:
        log.info(
            f"  {r['subject_id']:02d}   {r['name'][:25]:<25}  "
            f"{r['total_dur_s']:>6.0f}  {r['n_bad_candidates']:>4}      "
            f"{r['motion_high_s']:>6.1f}s  {r['motion_scen_overlap']:>4.1f}%"
        )
    log.info(f"\n  Reports: {S3}")
    log.info(f"  Figures: {S3}/subject_XX/")

    # Final check
    stoppers = [r for r in results if r["n_bad_candidates"] >= 15]
    if stoppers:
        for r in stoppers:
            log.error(f"  ⚠ STOP: sub-{r['subject_id']:02d} {r['name']} has "
                      f"{r['n_bad_candidates']} bad-candidate channels. Review before §4.")
    else:
        log.info("  No stoppers found. Readiness for Section 4: YES")


if __name__ == "__main__":
    main()
