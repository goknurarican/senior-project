"""
Bölüm 4: EEG Preprocessing
Run from project root: python src/section4_preprocessing.py

Pipeline per subject:
  4.1  Load + channel setup
  4.2  Filter (notch 50/100/150 Hz, bandpass 1-40 Hz)
  4.3  Bad channel interpolation
  4.4  Average reference
  4.5  ICA decomposition (picard/infomax, 20 components)
  4.6  ICLabel component classification
  4.7  ICA apply
  4.8  QC figures
  4.9  Epoch creation (ERP -0.2/+2.0 s, Causal -0.5/+3.0 s)
  4.10 AutoReject
  4.11 Save outputs
  4.12 Per-subject report
"""

import json
import logging
import os
import sys
import time
import traceback
import warnings
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as ss
import yaml

warnings.filterwarnings("ignore")
os.environ["MNE_LOGGING_LEVEL"] = "ERROR"
import mne
mne.set_log_level("ERROR")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC    = ROOT / "data" / "processed"
REP4    = ROOT / "data" / "reports" / "section4_preprocessing"
S3      = ROOT / "data" / "reports" / "section3_eda"
LOG_DIR = ROOT / "logs"
SEED    = 42

np.random.seed(SEED)

SFREQ      = 500
ACCEL_CHS  = ["x_dir", "y_dir", "z_dir"]

# Scenario event mapping (EEG marker code → name)
SCENARIO_CODES = {
    11: "slow_image",       12: "broken_image",     13: "skeleton_prolong",
    14: "search_irrelevant",15: "button_delay",      16: "first_click_miss",
    17: "feedback_late",    18: "network_jitter",    19: "overlay_blocking",
    20: "price_change",     21: "coupon_min_spend",  22: "coupon_expired",
    23: "facet_reset_once", 24: "sort_reset",
}
EVENT_ID = {f"S{code:02d}_{name}": code for code, name in SCENARIO_CODES.items()}

# ICA artifact labels to remove (ICLabel)
REMOVE_LABELS = {"muscle artifact", "eye blink", "heart beat", "line noise", "channel noise"}
PROB_THRESHOLD = 0.8

SUBJECTS = [
    {"id": 14, "name": "Alen Maryo",         "group": "variant_b", "folder": "user_014_alen_maryo_variant_b",         "eeg_prefix": "AlenAlen422"},
    {"id": 15, "name": "Eren Tamparlak",      "group": "variant_c", "folder": "user_015_eren_tamparlak_variant_c",     "eeg_prefix": "ET4272026"},
    {"id": 16, "name": "Berk Uygun",          "group": "variant_b", "folder": "user_016_berk_uygun_variant_b",         "eeg_prefix": "BU427"},
    {"id": 17, "name": "Mehmet İncekara",     "group": "variant_b", "folder": "user_017_mehmet_i̇ncekara_variant_b",    "eeg_prefix": "MI429"},
    {"id": 18, "name": "Feyiz Burak Öztürk", "group": "variant_b", "folder": "user_018_feyiz_burak_öztürk_variant_b", "eeg_prefix": "FB4292026"},
    {"id": 20, "name": "Veli Barış Sevinçhan","group": "variant_b", "folder": "user_020_veli_barış_sevinçhan_variant_b","eeg_prefix": "VB430"},
    {"id": 21, "name": "Enis Tiren",          "group": "variant_a", "folder": "user_021_enis_tiren_variant_a",         "eeg_prefix": "ET542026"},
    {"id": 22, "name": "Recep Danacı",        "group": "variant_c", "folder": "user_022_recep_danacı_variant_c",       "eeg_prefix": "RD552026"},
    {"id": 23, "name": "Duru Erol",           "group": "variant_c", "folder": "user_023_duru_erol_variant_c",          "eeg_prefix": "DE552026"},
]


# ── helpers ───────────────────────────────────────────────────────────────────
def make_logger(name: str, log_file: Path) -> logging.Logger:
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


def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def welch_psd_db(data: np.ndarray, sfreq: float, fmin=0.5, fmax=100.0):
    nperseg = int(sfreq * 2)
    freqs, psd = ss.welch(data, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], 10 * np.log10(psd[:, mask] + 1e-30)


def get_raw_t0_ms(raw: mne.io.Raw, em: pd.DataFrame) -> float:
    """Align eeg_markers wall_time to raw recording start using first blink anchor."""
    first_blink_annot_s = raw.annotations[0]["onset"]
    first_blink_em_ms   = em["wall_time_ms"].min()
    return float(first_blink_em_ms - first_blink_annot_s * 1000)


def build_events(raw: mne.io.Raw, em: pd.DataFrame) -> tuple:
    """Build MNE events array from eeg_markers.csv."""
    raw_t0_ms = get_raw_t0_ms(raw, em)
    scen = em[em["eeg_marker"].isin(SCENARIO_CODES.keys())].copy()
    scen["onset_s"]  = (scen["wall_time_ms"] - raw_t0_ms) / 1000.0
    scen["sample"]   = (scen["onset_s"] * SFREQ).astype(int)
    # Clamp to valid range with 3s margin for causal window
    margin = int(3.5 * SFREQ)
    scen = scen[(scen["sample"] >= 0) & (scen["sample"] < raw.n_times - margin)]
    if scen.empty:
        return np.zeros((0, 3), dtype=int), {}
    events = np.column_stack([
        scen["sample"].values,
        np.zeros(len(scen), dtype=int),
        scen["eeg_marker"].values,
    ]).astype(int)
    return events, EVENT_ID


def load_bad_channels(sid: int) -> list:
    bc_path = S3 / "bad_channels_per_subject.json"
    with open(bc_path, encoding="utf-8") as f:
        bc = json.load(f)
    return bc.get(f"subject_{sid:02d}", {}).get("bad_channels", [])


# ── 4.1 Load ──────────────────────────────────────────────────────────────────
def step_load(s: dict, log: logging.Logger) -> mne.io.Raw:
    vhdr = RAW_DIR / s["folder"] / f"{s['eeg_prefix']}.vhdr"
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
    log.info(f"  4.1 Loaded: {len(raw.ch_names)} ch, {raw.n_times/SFREQ:.1f}s")
    raw.set_channel_types({ch: "misc" for ch in ACCEL_CHS if ch in raw.ch_names})
    raw.drop_channels([ch for ch in ACCEL_CHS if ch in raw.ch_names])
    log.info(f"       After accel drop: {len(raw.ch_names)} ch")
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore", verbose=False)
    return raw


# ── 4.2 Filter ────────────────────────────────────────────────────────────────
def step_filter(raw: mne.io.Raw, log: logging.Logger) -> mne.io.Raw:
    raw.notch_filter(freqs=[50, 100, 150], method="fir", phase="zero",
                     fir_window="hamming", fir_design="firwin", verbose=False)
    raw.filter(l_freq=1.0, h_freq=40.0, method="fir", phase="zero",
               fir_window="hamming", fir_design="firwin", verbose=False)
    log.info("  4.2 Filter: notch 50/100/150 Hz + bandpass 1-40 Hz applied")
    return raw


# ── 4.3 Bad channel interpolation ─────────────────────────────────────────────
def step_interpolate(raw: mne.io.Raw, sid: int, log: logging.Logger) -> mne.io.Raw:
    bads = load_bad_channels(sid)
    if not bads:
        log.info("  4.3 Interpolation: no bad channels - skipped")
        return raw
    raw.info["bads"] = bads
    raw.interpolate_bads(reset_bads=True, mode="accurate", verbose=False)
    log.info(f"  4.3 Interpolated {len(bads)} channels: {bads}")
    return raw


# ── 4.4 Re-reference ──────────────────────────────────────────────────────────
def step_reref(raw: mne.io.Raw, log: logging.Logger) -> mne.io.Raw:
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    log.info("  4.4 Re-reference: average reference applied")
    return raw


# ── 4.5 ICA ───────────────────────────────────────────────────────────────────
def step_ica(raw: mne.io.Raw, subj_out: Path,
             log: logging.Logger) -> tuple[mne.preprocessing.ICA, mne.io.Raw]:
    raw_for_ica = raw.copy()
    # Try picard first (faster); fall back to infomax
    try:
        import picard  # noqa: F401
        ica = mne.preprocessing.ICA(
            n_components=20, method="picard",
            fit_params=dict(ortho=False, extended=True),
            random_state=SEED, max_iter=500,
        )
        method_used = "picard(extended)"
    except ImportError:
        ica = mne.preprocessing.ICA(
            n_components=20, method="infomax",
            fit_params=dict(extended=True),
            random_state=SEED, max_iter=1000,
        )
        method_used = "infomax(extended)"

    t0 = time.time()
    ica.fit(raw_for_ica, verbose=False)
    elapsed = time.time() - t0
    log.info(f"  4.5 ICA ({method_used}): fit in {elapsed:.1f}s")

    ica_path = subj_out / "ica-decomposition.fif"
    ica.save(str(ica_path), overwrite=True, verbose=False)
    log.info(f"       Saved: {ica_path.name}")
    return ica, raw_for_ica


# ── 4.6 ICLabel classification ────────────────────────────────────────────────
def step_iclabel(ica: mne.preprocessing.ICA,
                 raw_for_ica: mne.io.Raw,
                 subj_out: Path,
                 log: logging.Logger) -> tuple[list, dict]:
    from mne_icalabel import label_components
    ic_labels = label_components(raw_for_ica, ica, method="iclabel")
    labels    = ic_labels["labels"]
    probs     = ic_labels["y_pred_proba"]

    # Find components to remove
    exclude_idx = []
    exclude_detail = {}
    label_names = list({l for l in labels} | REMOVE_LABELS)
    label_counts = {lbl: [] for lbl in set(labels)}

    for idx, (lbl, prob_row) in enumerate(zip(labels, probs)):
        label_counts.setdefault(lbl, []).append(idx)
        max_prob = float(prob_row.max())
        if lbl in REMOVE_LABELS and max_prob >= PROB_THRESHOLD:
            exclude_idx.append(idx)
            exclude_detail.setdefault(lbl, []).append(
                {"idx": idx, "prob": round(max_prob, 3)}
            )

    brain_count = sum(1 for l in labels if l == "brain")
    log.info(f"  4.6 ICLabel: total=20  brain={brain_count}  "
             f"remove={len(exclude_idx)} → {list(exclude_detail.keys())}")
    for lbl, items in exclude_detail.items():
        idxs  = [x["idx"]  for x in items]
        pvals = [x["prob"] for x in items]
        log.info(f"       {lbl}: {idxs}  probs={pvals}")

    # Save labels JSON
    labels_out = {
        "labels":  list(labels),
        "excluded_indices": exclude_idx,
        "excluded_detail":  {k: v for k, v in exclude_detail.items()},
        "brain_components_kept": brain_count,
    }
    with open(subj_out / "ica_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_out, f, indent=2)

    return exclude_idx, labels_out


# ── 4.7 Apply ICA ─────────────────────────────────────────────────────────────
def step_apply_ica(raw: mne.io.Raw, ica: mne.preprocessing.ICA,
                   exclude_idx: list, log: logging.Logger) -> mne.io.Raw:
    ica.exclude = exclude_idx
    raw_clean = ica.apply(raw.copy(), verbose=False)
    log.info(f"  4.7 ICA applied: {len(exclude_idx)} components removed")
    return raw_clean


# ── 4.8 QC figures ────────────────────────────────────────────────────────────
def step_qc_figures(raw: mne.io.Raw, raw_clean: mne.io.Raw,
                    ica: mne.preprocessing.ICA,
                    raw_for_ica: mne.io.Raw,
                    em: pd.DataFrame,
                    subj_out: Path, name: str,
                    log: logging.Logger):
    raw_t0_ms = get_raw_t0_ms(raw, em)

    # PSD before/after
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, r, title in zip(axes, [raw, raw_clean], ["Before ICA", "After ICA"]):
        data = r.get_data(picks="eeg")
        freqs, psd_db = welch_psd_db(data, SFREQ)
        for i in range(data.shape[0]):
            ax.plot(freqs, psd_db[i], alpha=0.3, lw=0.7, color="steelblue")
        ax.plot(freqs, np.median(psd_db, axis=0), color="navy", lw=1.5, label="median")
        ax.axvline(50, color="red", ls="--", lw=0.8, alpha=0.6, label="50 Hz")
        ax.set_xscale("log"); ax.set_xlim(1, 100)
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (dB)")
        ax.set_title(f"{name} - {title}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, subj_out / "qc_psd_before_after.png")

    # Time-series before/after (30s control window)
    var_row = em[em["scenario_type"] == "variant_start"]
    if not var_row.empty:
        var_t0_s = (float(var_row.iloc[0]["wall_time_ms"]) - raw_t0_ms) / 1000
        win_start = max(0.0, var_t0_s - 90)  # 90s before variant start
        win_end   = win_start + 30
    else:
        win_start, win_end = 10.0, 40.0
    win_start = min(win_start, raw.times[-1] - 30)
    win_end   = win_start + 30

    s0, s1 = int(win_start * SFREQ), int(win_end * SFREQ)
    s0 = max(0, s0); s1 = min(s1, raw.n_times)
    eeg_chs = mne.pick_types(raw.info, eeg=True)
    data_before = raw.get_data(picks=eeg_chs)[:, s0:s1]
    data_after  = raw_clean.get_data(picks=eeg_chs)[:, s0:s1]
    t = np.arange(s1 - s0) / SFREQ
    n_ch  = data_before.shape[0]
    scale = 50e-6
    fig, axes = plt.subplots(2, 1, figsize=(18, n_ch * 0.35 + 2), sharex=True)
    ch_names = [raw.ch_names[i] for i in eeg_chs]
    for ax, data, ttl in zip(axes, [data_before, data_after], ["Before ICA", "After ICA"]):
        for i, (d, ch) in enumerate(zip(data, ch_names)):
            ax.plot(t, d / scale + i, lw=0.5, alpha=0.8)
            ax.text(-0.3, i, ch, ha="right", va="center", fontsize=5)
        ax.set_title(f"{name} - {ttl}  [{win_start:.0f}–{win_end:.0f}s]")
        ax.set_yticks([]); ax.spines[["left","top","right"]].set_visible(False)
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    savefig(fig, subj_out / "qc_timeseries_before_after.png")

    # ICA removed components topography
    if ica.exclude:
        try:
            n_excl = len(ica.exclude)
            fig, axes = plt.subplots(1, n_excl, figsize=(3 * n_excl + 1, 3))
            if n_excl == 1:
                axes = [axes]
            for ax, idx in zip(axes, ica.exclude):
                mne.viz.plot_topomap(
                    ica.get_components()[:, idx], raw_for_ica.info,
                    axes=ax, show=False, sphere="eeglab",
                )
                ax.set_title(f"IC{idx}", fontsize=9)
            fig.suptitle(f"{name} - Removed ICA components", fontsize=10)
            fig.tight_layout()
            savefig(fig, subj_out / "qc_removed_components.png")
        except Exception as e:
            log.warning(f"       ICA topomap failed: {e}")

    # Topographic band maps (clean, control phase)
    ctrl_end_s = (float(var_row.iloc[0]["wall_time_ms"]) - raw_t0_ms) / 1000 if not var_row.empty else raw.times[-1] / 2
    ctrl_start_s = max(0.0, ctrl_end_s - 120)
    try:
        BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,40)}
        data_ctrl = raw_clean.get_data(picks="eeg", tmin=ctrl_start_s, tmax=ctrl_end_s)
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for ax, (bname, (f1, f2)) in zip(axes, BANDS.items()):
            freqs_w, psd_w = ss.welch(data_ctrl, fs=SFREQ, nperseg=int(SFREQ*2),
                                      noverlap=int(SFREQ), axis=-1)
            bp = psd_w[:, (freqs_w>=f1)&(freqs_w<=f2)].mean(axis=1)
            bp_db = 10*np.log10(bp + 1e-30)
            mne.viz.plot_topomap(bp_db, raw_clean.info, axes=ax, show=False,
                                 cmap="RdYlBu_r",
                                 vlim=(np.percentile(bp_db,10),np.percentile(bp_db,90)),
                                 sphere="eeglab")
            ax.set_title(bname, fontsize=9)
        fig.suptitle(f"{name} - Band power after preprocessing")
        fig.tight_layout()
        savefig(fig, subj_out / "qc_topomap_bands.png")
    except Exception as e:
        log.warning(f"       Topomap bands failed: {e}")

    log.info("  4.8 QC figures saved")


# ── 4.9 Epoch creation ────────────────────────────────────────────────────────
def step_epochs(raw_clean: mne.io.Raw, em: pd.DataFrame,
                log: logging.Logger) -> tuple:
    events, event_id = build_events(raw_clean, em)
    if len(events) == 0:
        log.warning("  4.9 No valid events - empty epochs")
        return None, None

    log.info(f"  4.9 Creating epochs from {len(events)} scenario events")

    # Only keep event_id entries that actually appear in the events array
    present_codes = set(events[:, 2].tolist())
    event_id_present = {k: v for k, v in event_id.items() if v in present_codes}

    # ERP epochs
    epochs_erp = mne.Epochs(
        raw_clean, events, event_id_present,
        tmin=-0.2, tmax=2.0,
        baseline=(-0.2, 0),
        reject=None, preload=True, verbose=False,
    )
    log.info(f"       ERP epochs: {len(epochs_erp)} (before AutoReject)")

    # Causal epochs
    epochs_causal = mne.Epochs(
        raw_clean, events, event_id_present,
        tmin=-0.5, tmax=3.0,
        baseline=None,
        reject=None, preload=True, verbose=False,
    )
    log.info(f"       Causal epochs: {len(epochs_causal)} (before AutoReject)")
    return epochs_erp, epochs_causal


# ── 4.10 AutoReject ───────────────────────────────────────────────────────────
def step_autoreject(epochs_erp, epochs_causal, sid: int,
                    log: logging.Logger) -> tuple:
    from autoreject import AutoReject

    results = {}

    for name_e, epochs in [("ERP", epochs_erp), ("Causal", epochs_causal)]:
        if epochs is None or len(epochs) == 0:
            results[name_e] = {"before": 0, "after": 0, "rejected": 0, "interpolated": 0}
            continue
        n_before = len(epochs)
        # Adaptive CV: min 3, max 10
        n_cv = max(3, min(10, n_before // 5))
        try:
            ar = AutoReject(
                n_interpolate=[1, 2, 4],
                random_state=SEED,
                n_jobs=1,
                cv=n_cv,
                verbose=False,
            )
            epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
            n_after = len(epochs_clean)
            n_rejected = int(reject_log.bad_epochs.sum())
            n_interp = int((reject_log.labels == 1).sum()) if hasattr(reject_log, "labels") else 0
            log.info(f"       {name_e} AutoReject: {n_before}→{n_after}  "
                     f"rejected={n_rejected}  interpolated={n_interp}")
            # Warn if >50% rejected
            if n_before > 0 and (n_rejected / n_before) > 0.5:
                log.warning(f"       ⚠ {name_e}: {n_rejected}/{n_before} = "
                             f"{n_rejected/n_before*100:.0f}% rejected - high rejection rate!")
        except Exception as e:
            log.warning(f"       {name_e} AutoReject failed ({e}) - keeping unrejected epochs")
            epochs_clean = epochs
            n_after = n_before
            n_rejected = 0
            n_interp = 0
        results[name_e] = {
            "before":      n_before,
            "after":       n_after,
            "rejected":    n_rejected,
            "interpolated": n_interp,
            "epochs":      epochs_clean,
        }

    return results["ERP"], results["Causal"]


# ── 4.11 Save outputs ─────────────────────────────────────────────────────────
def step_save(raw_clean: mne.io.Raw, ica, epochs_erp, epochs_causal,
              subj_out: Path, log: logging.Logger):
    raw_clean.save(str(subj_out / "raw_clean-raw.fif"), overwrite=True, verbose=False)
    log.info(f"       raw_clean-raw.fif saved")
    if epochs_erp is not None and hasattr(epochs_erp, "events"):
        epochs_erp.save(str(subj_out / "epochs_erp-epo.fif"), overwrite=True, verbose=False)
        log.info(f"       epochs_erp-epo.fif: {len(epochs_erp)} epochs")
    if epochs_causal is not None and hasattr(epochs_causal, "events"):
        epochs_causal.save(str(subj_out / "epochs_causal-epo.fif"), overwrite=True, verbose=False)
        log.info(f"       epochs_causal-epo.fif: {len(epochs_causal)} epochs")


# ── 4.12 Per-subject report ───────────────────────────────────────────────────
def write_subject_report(s: dict, subj_out: Path,
                         total_dur: float, bads: list,
                         ica_labels: dict,
                         erp_result: dict, causal_result: dict,
                         log: logging.Logger):
    sid  = s["id"]
    name = s["name"]
    excl = ica_labels.get("excluded_detail", {})

    # Compute 50 Hz attenuation
    fif_path = subj_out / "raw_clean-raw.fif"
    noise_note = ""
    try:
        rc = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
        data_c = rc.get_data(picks="eeg")
        freqs_w, psd_w = ss.welch(data_c, fs=SFREQ, nperseg=int(SFREQ*2), noverlap=int(SFREQ), axis=-1)
        idx50 = np.argmin(np.abs(freqs_w - 50))
        idx10 = np.argmin(np.abs(freqs_w - 10))
        noise_db_after = float(np.median(psd_w[:, idx50]))
        alpha_db_after = float(np.median(psd_w[:, idx10]))
        noise_note = f"50 Hz median: {10*np.log10(noise_db_after+1e-30):.1f} dB after preprocessing"
    except Exception:
        pass

    md = [
        f"# Preprocessing Report - Subject {sid:02d} ({name})",
        f"",
        f"Generated: {date.today()}",
        f"",
        f"## Input",
        f"- Duration: {total_dur:.1f} seconds",
        f"- Original channels: 35 (32 EEG + 3 accelerometer; accel dropped)",
        f"",
        f"## Filtering",
        f"- Notch: 50, 100, 150 Hz - applied",
        f"- Bandpass: 1.0–40.0 Hz - applied",
        f"- Method: FIR, zero-phase, Hamming window",
        f"",
        f"## Bad Channel Interpolation",
        f"- Channels: {bads if bads else 'none'}",
        f"- Method: spherical spline",
        f"",
        f"## Re-referencing",
        f"- Average reference applied (after interpolation)",
        f"",
        f"## ICA",
        f"- Components fit: 20",
        f"- Algorithm: picard (extended infomax equivalent)",
        f"- Random seed: 42",
        f"- Brain components kept: {ica_labels.get('brain_components_kept', '?')}",
        f"- Removed components ({len(ica_labels.get('excluded_indices', []))}):",
    ]
    for lbl, items in excl.items():
        idxs  = [x["idx"]  for x in items]
        pvals = [x["prob"] for x in items]
        md.append(f"  - {lbl}: {idxs}  probs={pvals}")
    if not excl:
        md.append("  - (none)")
    md += [
        f"",
        f"## Epoch Counts",
        f"- ERP (before AutoReject):   {erp_result.get('before', 0)}",
        f"- ERP (after AutoReject):    {erp_result.get('after', 0)}  (rejected: {erp_result.get('rejected', 0)})",
        f"- Causal (before AutoReject): {causal_result.get('before', 0)}",
        f"- Causal (after AutoReject):  {causal_result.get('after', 0)}  (rejected: {causal_result.get('rejected', 0)})",
        f"",
        f"## Quality Indicators",
        f"- Pre/post PSD: see qc_psd_before_after.png",
        f"- {noise_note}",
        f"",
        f"## Notes",
    ]
    if sid == 14:
        md.append("- F4 + Fp2 interpolated → FAA values flagged as LOW CONFIDENCE")
    if sid == 17:
        md.append("- F4 interpolated → FAA flagged. Monitor O1/Oz/O2 ICA outcome.")
    if sid == 23:
        md.append("- Duru: 22 scenarios missing from eye data (3s logging lag); EEG-only for those epochs")
    if sid == 20:
        md.append("- Veli: eye features valid only for 0-900s window")

    (subj_out / "preprocessing_report.md").write_text("\n".join(md), encoding="utf-8")
    log.info(f"  4.12 Report saved: preprocessing_report.md")


# ── subject orchestrator ──────────────────────────────────────────────────────
def process_subject(s: dict) -> dict:
    sid     = s["id"]
    name    = s["name"]
    subj_out = PROC / f"subject_{sid:02d}"
    subj_out.mkdir(parents=True, exist_ok=True)

    log = make_logger(f"s4_{sid}", LOG_DIR / f"section4_preprocessing_subject_{sid:02d}.log")
    log.info("=" * 60)
    log.info(f"Subject {sid:02d}: {name}")
    log.info("=" * 60)

    em_path = RAW_DIR / s["folder"] / "eeg" / "eeg_markers.csv"
    em = pd.read_csv(em_path)

    t_start = time.time()
    try:
        raw       = step_load(s, log)
        total_dur = raw.n_times / SFREQ
        raw_orig  = raw.copy()          # keep for QC before/after comparison

        raw       = step_filter(raw, log)
        bads      = load_bad_channels(sid)
        raw       = step_interpolate(raw, sid, log)
        raw       = step_reref(raw, log)

        ica, raw_for_ica = step_ica(raw, subj_out, log)
        exclude_idx, ica_labels = step_iclabel(ica, raw_for_ica, subj_out, log)
        raw_clean = step_apply_ica(raw, ica, exclude_idx, log)

        step_qc_figures(raw_orig, raw_clean, ica, raw_for_ica, em, subj_out, name, log)

        epochs_erp, epochs_causal = step_epochs(raw_clean, em, log)

        erp_res, causal_res = step_autoreject(epochs_erp, epochs_causal, sid, log)

        epochs_erp_final    = erp_res.pop("epochs",    epochs_erp)
        epochs_causal_final = causal_res.pop("epochs", epochs_causal)

        step_save(raw_clean, ica, epochs_erp_final, epochs_causal_final, subj_out, log)
        write_subject_report(s, subj_out, total_dur, bads, ica_labels,
                             erp_res, causal_res, log)

        elapsed = time.time() - t_start
        log.info(f"\n  ✓ Done in {elapsed:.1f}s")

        return {
            "subject_id":          sid,
            "name":                name,
            "group":               s["group"],
            "success":             True,
            "bad_channels":        bads,
            "n_bad_ch":            len(bads),
            "n_ica_removed":       len(exclude_idx),
            "ica_removed_types":   list(ica_labels.get("excluded_detail", {}).keys()),
            "brain_components":    ica_labels.get("brain_components_kept", 0),
            "erp_before":          erp_res.get("before", 0),
            "erp_after":           erp_res.get("after", 0),
            "erp_rejected":        erp_res.get("rejected", 0),
            "causal_before":       causal_res.get("before", 0),
            "causal_after":        causal_res.get("after", 0),
            "causal_rejected":     causal_res.get("rejected", 0),
            "elapsed_s":           round(elapsed, 1),
        }

    except Exception as exc:
        elapsed = time.time() - t_start
        log.error(f"  ✗ FAILED after {elapsed:.1f}s: {exc}")
        log.error(traceback.format_exc())
        return {"subject_id": sid, "name": name, "success": False, "error": str(exc)}


# ── cross-subject summary ─────────────────────────────────────────────────────
def write_section4_summary(results: list):
    REP4.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r for r in results if r.get("success", False)])

    if df.empty:
        print("No successful results to summarize.")
        return

    df.to_csv(REP4 / "section4_metrics.csv", index=False)

    # Compute retention rates
    df["erp_retention_pct"]    = (df["erp_after"]    / df["erp_before"].clip(1) * 100).round(1)
    df["causal_retention_pct"] = (df["causal_after"] / df["causal_before"].clip(1) * 100).round(1)

    avg_ica   = df["n_ica_removed"].mean()
    avg_erp   = df["erp_retention_pct"].mean()
    avg_caus  = df["causal_retention_pct"].mean()

    attention = df[
        (df["n_ica_removed"] >= 10) |
        (df["erp_retention_pct"] < 50) |
        (df["causal_retention_pct"] < 50)
    ]["name"].tolist()

    faa_flags = []
    for _, row in df.iterrows():
        if row["subject_id"] in [14, 17]:
            faa_flags.append(f"sub-{row['subject_id']:02d} {row['name']} (F4 interpolated)")

    md_path = REP4 / "section4_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Section 4 - Preprocessing Summary\n\n")
        f.write(f"Generated: {date.today()}\n\n")
        f.write("## Overview\n\n")
        f.write(f"- {len(df)}/{len(results)} subjects processed successfully\n")
        f.write("- Pipeline: notch → bandpass → bad channel interp → re-reference → ICA → ICLabel → epoch → AutoReject\n\n")
        f.write("## Per-subject metrics\n\n")
        cols = ["subject_id","name","n_bad_ch","n_ica_removed","erp_before","erp_after","erp_rejection_pct",
                "causal_before","causal_after","causal_rejection_pct"]

        tbl = df.copy()
        tbl["erp_rejection_pct"]    = (100 - df["erp_retention_pct"]).round(1)
        tbl["causal_rejection_pct"] = (100 - df["causal_retention_pct"]).round(1)
        f.write(tbl[["subject_id","name","n_bad_ch","n_ica_removed",
                     "erp_before","erp_after","erp_rejection_pct",
                     "causal_before","causal_after","causal_rejection_pct"]].to_markdown(index=False))

        f.write("\n\n## Common patterns\n\n")
        f.write(f"- Average ICA components removed: {avg_ica:.1f}\n")
        from collections import Counter
        all_types = [t for r in results if r.get("success") for t in r.get("ica_removed_types", [])]
        if all_types:
            most_common = Counter(all_types).most_common(1)[0]
            f.write(f"- Most common removed category: {most_common[0]} ({most_common[1]} subjects)\n")
        f.write(f"- Average ERP epoch retention: {avg_erp:.1f}%\n")
        f.write(f"- Average causal epoch retention: {avg_caus:.1f}%\n\n")
        f.write("## Subjects requiring attention\n\n")
        f.write(", ".join(attention) if attention else "_None_")
        f.write("\n\n## Quality flags\n\n")
        f.write("- FAA low confidence: " + (", ".join(faa_flags) if faa_flags else "_none_") + "\n")
        f.write("\n## Readiness for Section 5 (Eye preprocessing)\n\n")
        ready = len(attention) == 0 or all(r.get("erp_retention_pct", 100) >= 50 for r in results if r.get("success"))
        f.write(f"**{'YES' if ready else 'CONDITIONAL'}**\n")

    print(f"\nSection 4 summary: {md_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    main_log = make_logger("s4_main", LOG_DIR / "section4_preprocessing_main.log")
    main_log.info("=" * 70)
    main_log.info("Bölüm 4: EEG Preprocessing - 9 subjects")
    main_log.info("=" * 70)

    results = []
    for s in SUBJECTS:
        main_log.info(f"\n{'━'*50}\nStarting sub-{s['id']:02d} {s['name']}\n{'━'*50}")
        result = process_subject(s)
        results.append(result)
        if result.get("success"):
            main_log.info(
                f"sub-{s['id']:02d}: ICA_removed={result['n_ica_removed']}  "
                f"ERP {result['erp_before']}→{result['erp_after']}  "
                f"Causal {result['causal_before']}→{result['causal_after']}"
            )
        else:
            main_log.error(f"sub-{s['id']:02d}: FAILED - {result.get('error','?')}")

    main_log.info("\n" + "=" * 70)
    main_log.info("BÖLÜM 4 TAMAMLANDI")
    main_log.info("=" * 70)

    success = [r for r in results if r.get("success")]
    main_log.info(f"\n  Başarılı: {len(success)}/{len(results)}")
    main_log.info(f"\n  Sub  Name                       ICA_out  ERP(B→A)  Causal(B→A)  Time")
    main_log.info(f"  {'-'*72}")
    for r in results:
        if r.get("success"):
            main_log.info(
                f"  {r['subject_id']:02d}   {r['name'][:25]:<25}  "
                f"{r['n_ica_removed']:>4}     "
                f"{r['erp_before']:>3}→{r['erp_after']:<3}   "
                f"{r['causal_before']:>3}→{r['causal_after']:<3}   "
                f"{r['elapsed_s']:.0f}s"
            )
        else:
            main_log.error(f"  {r['subject_id']:02d}   {r['name'][:25]:<25}  FAILED")

    write_section4_summary(results)

    # Stop checks
    failed = [r for r in results if not r.get("success")]
    if failed:
        main_log.error(f"\n  ⚠ {len(failed)} subjects failed: {[r['name'] for r in failed]}")

    high_ica = [r for r in success if r.get("n_ica_removed", 0) >= 10]
    if high_ica:
        hi_names = ["{} ({})".format(r['name'], r['n_ica_removed']) for r in high_ica]
        main_log.warning(
            f"\n  ⚠ High ICA removal (≥10 components): {hi_names}"
        )

    low_epoch = [r for r in success if r.get("erp_after", 99) / max(r.get("erp_before", 1), 1) < 0.5]
    if low_epoch:
        main_log.warning(
            f"\n  ⚠ Low ERP epoch retention (<50%): "
            f"{[r['name'] for r in low_epoch]}"
        )


if __name__ == "__main__":
    main()
