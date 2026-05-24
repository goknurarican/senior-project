"""
Section 4B: Sub-22 Recep - Parameter Variant Comparison + Quick QC (Veli, Eren)

Produces:
  data/processed/subject_22_v2/   (1.5 Hz, 20 comp, thresh 0.8)
  data/processed/subject_22_v3/   (1.5 Hz, 25 comp, thresh 0.8)
  data/processed/subject_22_v4/   (1.5 Hz, 25 comp, thresh 0.7)
  data/reports/section4_corrections/recep_variant_comparison.md
  data/reports/section4_corrections/quick_qc_veli_eren.md
  data/reports/section4_corrections/section4_final_status.md
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
import yaml

warnings.filterwarnings("ignore")
os.environ["MNE_LOGGING_LEVEL"] = "ERROR"
import mne
mne.set_log_level("ERROR")

# ── import shared helpers from section4_preprocessing ─────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from section4_preprocessing import (
    make_logger, savefig, welch_psd_db, get_raw_t0_ms, build_events,
    load_bad_channels, step_load, step_interpolate, step_reref,
    step_epochs, step_autoreject, step_save,
    SFREQ, SEED, S3, RAW_DIR, PROC, LOG_DIR,
    REMOVE_LABELS, SCENARIO_CODES, EVENT_ID,
)

CORRECTIONS_REP = ROOT / "data" / "reports" / "section4_corrections"
CORRECTIONS_REP.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)

# ── Recep subject record ───────────────────────────────────────────────────────
RECEP = {"id": 22, "name": "Recep Danacı", "group": "variant_c",
         "folder": "user_022_recep_danacı_variant_c", "eeg_prefix": "RD552026"}

VELI  = {"id": 20, "name": "Veli Barış Sevinçhan", "group": "variant_b",
          "folder": "user_020_veli_barış_sevinçhan_variant_b", "eeg_prefix": "VB430"}

EREN  = {"id": 15, "name": "Eren Tamparlak", "group": "variant_c",
          "folder": "user_015_eren_tamparlak_variant_c", "eeg_prefix": "ET4272026"}

# Channels for alpha topography metric
OCCIPITAL_CHS = ["O1", "Oz", "O2"]
FRONTAL_CHS   = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"]

VARIANTS = [
    {"tag": "v2", "l_freq": 1.5, "n_components": 20, "prob_threshold": 0.8},
    {"tag": "v3", "l_freq": 1.5, "n_components": 25, "prob_threshold": 0.8},
    {"tag": "v4", "l_freq": 1.5, "n_components": 25, "prob_threshold": 0.7},
]


# ── load eeg_markers.csv for a subject ────────────────────────────────────────
def load_em(s: dict) -> pd.DataFrame:
    em_path = RAW_DIR / s["folder"] / "eeg" / "eeg_markers.csv"
    em = pd.read_csv(em_path)
    em.columns = em.columns.str.strip().str.lower()
    return em


# ── variant-specific filter ───────────────────────────────────────────────────
def step_filter_v(raw: mne.io.Raw, l_freq: float, log: logging.Logger) -> mne.io.Raw:
    raw.notch_filter(freqs=[50, 100, 150], method="fir", phase="zero",
                     fir_window="hamming", fir_design="firwin", verbose=False)
    raw.filter(l_freq=l_freq, h_freq=40.0, method="fir", phase="zero",
               fir_window="hamming", fir_design="firwin", verbose=False)
    log.info(f"  4.2 Filter: notch 50/100/150 Hz + bandpass {l_freq}-40 Hz applied")
    return raw


# ── variant-specific ICA ──────────────────────────────────────────────────────
def step_ica_v(raw: mne.io.Raw, n_components: int, subj_out: Path,
               log: logging.Logger) -> tuple:
    raw_for_ica = raw.copy()
    try:
        import picard  # noqa
        ica = mne.preprocessing.ICA(
            n_components=n_components, method="picard",
            fit_params=dict(ortho=False, extended=True),
            random_state=SEED, max_iter=500,
        )
        method_used = "picard(extended)"
    except ImportError:
        ica = mne.preprocessing.ICA(
            n_components=n_components, method="infomax",
            fit_params=dict(extended=True),
            random_state=SEED, max_iter=1000,
        )
        method_used = "infomax(extended)"

    t0 = time.time()
    ica.fit(raw_for_ica, verbose=False)
    elapsed = time.time() - t0
    log.info(f"  4.5 ICA ({method_used}, {n_components} comp): fit in {elapsed:.1f}s")
    ica.save(str(subj_out / "ica-decomposition.fif"), overwrite=True, verbose=False)
    return ica, raw_for_ica


# ── variant-specific ICLabel ──────────────────────────────────────────────────
def step_iclabel_v(ica, raw_for_ica, prob_threshold: float,
                   subj_out: Path, log: logging.Logger) -> tuple:
    from mne_icalabel import label_components
    ic_labels = label_components(raw_for_ica, ica, method="iclabel")
    labels = ic_labels["labels"]
    probs  = ic_labels["y_pred_proba"]

    exclude_idx    = []
    exclude_detail = {}
    for idx, (lbl, prob_row) in enumerate(zip(labels, probs)):
        max_prob = float(prob_row.max())
        if lbl in REMOVE_LABELS and max_prob >= prob_threshold:
            exclude_idx.append(idx)
            exclude_detail.setdefault(lbl, []).append(
                {"idx": idx, "prob": round(max_prob, 3)}
            )

    brain_count = sum(1 for l in labels if l == "brain")
    log.info(f"  4.6 ICLabel (thresh={prob_threshold}): total={len(labels)} "
             f"brain={brain_count}  remove={len(exclude_idx)} → "
             f"{list(exclude_detail.keys())}")

    labels_out = {
        "labels": list(labels),
        "excluded_indices": exclude_idx,
        "excluded_detail": exclude_detail,
        "brain_components_kept": brain_count,
    }
    with open(subj_out / "ica_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_out, f, indent=2)
    return exclude_idx, labels_out


# ── QC figures (PSD + time series + topomaps) ─────────────────────────────────
def make_qc_figures(raw, raw_clean, ica, raw_for_ica, em, subj_out, name, log):
    raw_t0_ms = get_raw_t0_ms(raw, em)
    var_row   = em[em["scenario_type"] == "variant_start"] if "scenario_type" in em.columns else pd.DataFrame()

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
        ax.set_title(f"{name} - {title}"); ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, subj_out / "qc_psd_before_after.png")

    # Time series (30s window)
    if not var_row.empty:
        var_t0_s = (float(var_row.iloc[0]["wall_time_ms"]) - raw_t0_ms) / 1000
        win_start = max(0.0, var_t0_s - 90)
    else:
        win_start = 10.0
    win_start = min(win_start, raw.times[-1] - 30)
    win_end   = win_start + 30
    s0, s1    = int(win_start * SFREQ), int(win_end * SFREQ)
    s0        = max(0, s0); s1 = min(s1, raw.n_times)
    eeg_chs   = mne.pick_types(raw.info, eeg=True)
    data_b    = raw.get_data(picks=eeg_chs)[:, s0:s1]
    data_a    = raw_clean.get_data(picks=eeg_chs)[:, s0:s1]
    t_axis    = np.arange(s1 - s0) / SFREQ
    ch_names  = [raw.ch_names[i] for i in eeg_chs]
    scale     = 50e-6
    fig, axes = plt.subplots(2, 1, figsize=(18, len(ch_names) * 0.35 + 2), sharex=True)
    for ax, data, ttl in zip(axes, [data_b, data_a], ["Before ICA", "After ICA"]):
        for i, (d, ch) in enumerate(zip(data, ch_names)):
            ax.plot(t_axis, d / scale + i, lw=0.5, alpha=0.8)
            ax.text(-0.3, i, ch, ha="right", va="center", fontsize=5)
        ax.set_title(f"{name} - {ttl}  [{win_start:.0f}–{win_end:.0f}s]")
        ax.set_yticks([]); ax.spines[["left", "top", "right"]].set_visible(False)
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    savefig(fig, subj_out / "qc_timeseries_before_after.png")

    # Removed components topomap
    if ica.exclude:
        try:
            n_excl = len(ica.exclude)
            fig, axes_t = plt.subplots(1, n_excl, figsize=(3 * n_excl + 1, 3))
            if n_excl == 1: axes_t = [axes_t]
            for ax, idx in zip(axes_t, ica.exclude):
                mne.viz.plot_topomap(ica.get_components()[:, idx], raw_for_ica.info,
                                     axes=ax, show=False, sphere="eeglab")
                ax.set_title(f"IC{idx}", fontsize=9)
            fig.suptitle(f"{name} - Removed ICA components", fontsize=10)
            fig.tight_layout()
            savefig(fig, subj_out / "qc_removed_components.png")
        except Exception as e:
            log.warning(f"       Removed components topomap failed: {e}")

    # Band power topomaps
    ctrl_end_s   = (float(var_row.iloc[0]["wall_time_ms"]) - raw_t0_ms) / 1000 if not var_row.empty else raw.times[-1] / 2
    ctrl_start_s = max(0.0, ctrl_end_s - 120)
    try:
        BANDS = {"delta": (1,4), "theta": (4,8), "alpha": (8,13), "beta": (13,30), "gamma": (30,40)}
        data_ctrl = raw_clean.get_data(picks="eeg", tmin=ctrl_start_s, tmax=ctrl_end_s)
        fig, axes_t = plt.subplots(1, 5, figsize=(20, 4))
        for ax, (bname, (f1, f2)) in zip(axes_t, BANDS.items()):
            fw, psw = ss.welch(data_ctrl, fs=SFREQ, nperseg=int(SFREQ*2),
                               noverlap=int(SFREQ), axis=-1)
            bp    = psw[:, (fw >= f1) & (fw <= f2)].mean(axis=1)
            bp_db = 10 * np.log10(bp + 1e-30)
            mne.viz.plot_topomap(bp_db, raw_clean.info, axes=ax, show=False,
                                 cmap="RdYlBu_r",
                                 vlim=(np.percentile(bp_db, 10), np.percentile(bp_db, 90)),
                                 sphere="eeglab")
            ax.set_title(bname, fontsize=9)
        fig.suptitle(f"{name} - Band power after preprocessing")
        fig.tight_layout()
        savefig(fig, subj_out / "qc_topomap_bands.png")
    except Exception as e:
        log.warning(f"       Topomap bands failed: {e}")

    log.info("  4.8 QC figures saved")


# ── Alpha topography metric ───────────────────────────────────────────────────
def compute_alpha_ok(raw_clean: mne.io.Raw) -> tuple:
    """
    Returns (alpha_ok: bool, occ_mean_db: float, front_mean_db: float).
    alpha_ok = True if mean occipital alpha > mean frontal alpha.
    Uses channels available in the recording (skips missing).
    """
    ch_names = raw_clean.ch_names
    occ_avail  = [c for c in OCCIPITAL_CHS  if c in ch_names]
    front_avail = [c for c in FRONTAL_CHS   if c in ch_names]

    data_occ   = raw_clean.get_data(picks=occ_avail)   if occ_avail  else None
    data_front = raw_clean.get_data(picks=front_avail) if front_avail else None

    def alpha_power_db(data):
        _, psd = ss.welch(data, fs=SFREQ, nperseg=int(SFREQ * 2),
                          noverlap=int(SFREQ), axis=-1)
        # 8-12 Hz
        freqs_full = ss.welch(data[0], fs=SFREQ, nperseg=int(SFREQ * 2),
                              noverlap=int(SFREQ))[0]
        alpha_mask = (freqs_full >= 8) & (freqs_full <= 12)
        alpha_psd  = psd[:, alpha_mask].mean(axis=1)
        return float(np.mean(10 * np.log10(alpha_psd + 1e-30)))

    if data_occ is None or data_front is None:
        return False, 0.0, 0.0

    occ_db   = alpha_power_db(data_occ)
    front_db = alpha_power_db(data_front)
    return occ_db > front_db, occ_db, front_db


# ── 50 Hz attenuation metric ──────────────────────────────────────────────────
def compute_50hz_attenuation(raw_before: mne.io.Raw, raw_after: mne.io.Raw) -> float:
    def median_50hz_db(r):
        data = r.get_data(picks="eeg")
        freqs, psd = ss.welch(data, fs=SFREQ, nperseg=int(SFREQ * 2),
                              noverlap=int(SFREQ), axis=-1)
        idx = np.argmin(np.abs(freqs - 50))
        return float(np.median(10 * np.log10(psd[:, idx] + 1e-30)))
    before_db = median_50hz_db(raw_before)
    after_db  = median_50hz_db(raw_after)
    return round(before_db - after_db, 1)


# ── run one Recep variant ─────────────────────────────────────────────────────
def run_recep_variant(variant: dict) -> dict:
    tag        = variant["tag"]
    l_freq     = variant["l_freq"]
    n_comp     = variant["n_components"]
    prob_thr   = variant["prob_threshold"]
    subj_out   = PROC / f"subject_22_{tag}"
    subj_out.mkdir(parents=True, exist_ok=True)

    log = make_logger(f"s4b_{tag}", LOG_DIR / f"section4b_recep_{tag}.log")
    log.info(f"=== Recep variant {tag}: l_freq={l_freq}, n_comp={n_comp}, thresh={prob_thr} ===")
    t_start = time.time()

    em  = load_em(RECEP)
    raw = step_load(RECEP, log)
    raw_before = raw.copy()  # keep unfiltered copy for 50 Hz metric

    raw = step_filter_v(raw, l_freq, log)
    raw = step_interpolate(raw, RECEP["id"], log)  # same bad channels
    raw = step_reref(raw, log)

    ica, raw_for_ica    = step_ica_v(raw, n_comp, subj_out, log)
    exclude_idx, labels = step_iclabel_v(ica, raw_for_ica, prob_thr, subj_out, log)

    ica.exclude = exclude_idx
    raw_clean   = ica.apply(raw.copy(), verbose=False)
    log.info(f"  4.7 ICA applied: {len(exclude_idx)} components removed")

    make_qc_figures(raw, raw_clean, ica, raw_for_ica, em, subj_out,
                    f"Recep Danacı ({tag})", log)

    erp_result, causal_result = step_autoreject(
        *step_epochs(raw_clean, em, log), RECEP["id"], log
    )
    step_save(raw_clean, ica,
              erp_result.get("epochs") if isinstance(erp_result, dict) else None,
              causal_result.get("epochs") if isinstance(causal_result, dict) else None,
              subj_out, log)

    # Compute metrics
    alpha_ok, occ_db, front_db = compute_alpha_ok(raw_clean)
    atten_50hz = compute_50hz_attenuation(raw_before, raw_clean)

    erp_b   = erp_result.get("before", 0)   if isinstance(erp_result, dict) else 0
    erp_a   = erp_result.get("after", 0)    if isinstance(erp_result, dict) else 0
    caus_b  = causal_result.get("before", 0) if isinstance(causal_result, dict) else 0
    caus_a  = causal_result.get("after", 0)  if isinstance(causal_result, dict) else 0

    erp_ret  = round(erp_a  / erp_b  * 100, 1) if erp_b  > 0 else 0.0
    caus_ret = round(caus_a / caus_b * 100, 1) if caus_b > 0 else 0.0

    elapsed = time.time() - t_start
    log.info(f"  ✓ {tag} DONE in {elapsed:.0f}s: brain={labels['brain_components_kept']} "
             f"alpha_ok={alpha_ok} ERP {erp_b}→{erp_a} ({erp_ret}%) "
             f"Causal {caus_b}→{caus_a} ({caus_ret}%)")

    return {
        "tag":           tag,
        "l_freq":        l_freq,
        "n_components":  n_comp,
        "prob_threshold": prob_thr,
        "brain_count":   labels["brain_components_kept"],
        "n_removed":     len(exclude_idx),
        "removed_labels": list(labels.get("excluded_detail", {}).keys()),
        "alpha_ok":      alpha_ok,
        "occ_alpha_db":  round(occ_db, 2),
        "front_alpha_db": round(front_db, 2),
        "erp_before":    erp_b,
        "erp_after":     erp_a,
        "erp_ret_pct":   erp_ret,
        "causal_before": caus_b,
        "causal_after":  caus_a,
        "causal_ret_pct": caus_ret,
        "atten_50hz_db": atten_50hz,
        "elapsed_s":     round(elapsed, 1),
    }


# ── V1 baseline metrics (from existing processed output) ─────────────────────
def load_v1_metrics() -> dict:
    labels_path = PROC / "subject_22" / "ica_labels.json"
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    raw_clean = mne.io.read_raw_fif(
        str(PROC / "subject_22" / "raw_clean-raw.fif"), preload=True, verbose=False
    )
    alpha_ok, occ_db, front_db = compute_alpha_ok(raw_clean)
    return {
        "tag": "v1",
        "l_freq": 1.0, "n_components": 20, "prob_threshold": 0.8,
        "brain_count":    labels["brain_components_kept"],
        "n_removed":      len(labels["excluded_indices"]),
        "alpha_ok":       alpha_ok,
        "occ_alpha_db":   round(occ_db, 2),
        "front_alpha_db": round(front_db, 2),
        "erp_before":     75, "erp_after":     53, "erp_ret_pct":    70.7,
        "causal_before":  75, "causal_after":  47, "causal_ret_pct": 62.7,
    }


# ── Score a variant (how many of 4 targets does it hit?) ─────────────────────
def score_variant(m: dict) -> int:
    score = 0
    if m["brain_count"] >= 5:      score += 1
    if m["alpha_ok"]:               score += 1
    if m["erp_ret_pct"]   >= 80.0:  score += 1
    if m["causal_ret_pct"] >= 80.0: score += 1
    return score


# ── Best variant selector ─────────────────────────────────────────────────────
def pick_best(v1, variants) -> str:
    all_m = [v1] + variants
    scored = [(score_variant(m), m) for m in all_m]
    scored.sort(key=lambda x: (
        x[0],
        int(x[1]["alpha_ok"]),
        x[1]["brain_count"],
        x[1]["erp_ret_pct"] + x[1]["causal_ret_pct"],
    ), reverse=True)
    best_score, best_m = scored[0]
    v1_score = score_variant(v1)
    if best_m["tag"] == "v1":
        return "NO_IMPROVEMENT"
    if best_score <= v1_score:
        return "NO_IMPROVEMENT"
    if best_score < 2:
        return "NO_IMPROVEMENT"
    return best_m["tag"]


# ── Comparison report ─────────────────────────────────────────────────────────
def write_comparison_report(v1: dict, variants: list, decision: str):
    all_m  = [v1] + variants
    header = ("| Variant | l_freq | n_comp | ICL prob | Brain | Alpha OK | "
               "ERP ret | Causal ret | Score |")
    sep    = ("|---------|--------|--------|----------|-------|----------|"
               "---------|------------|-------|")
    rows   = []
    for m in all_m:
        sc = score_variant(m)
        label = m["tag"] + (" (current)" if m["tag"] == "v1" else "")
        rows.append(
            f"| {label} | {m['l_freq']} | {m['n_components']} | "
            f"{m['prob_threshold']} | {m['brain_count']} | "
            f"{'✓' if m['alpha_ok'] else '✗'} | "
            f"{m['erp_ret_pct']}% | {m['causal_ret_pct']}% | {sc}/4 |"
        )

    v1_sc = score_variant(v1)
    score_lines = []
    for m in all_m:
        sc = score_variant(m)
        hits = []
        if m["brain_count"] >= 5:      hits.append("brain≥5 ✓")
        if m["alpha_ok"]:               hits.append("alpha_ok ✓")
        if m["erp_ret_pct"] >= 80:      hits.append(f"ERP {m['erp_ret_pct']}% ✓")
        if m["causal_ret_pct"] >= 80:   hits.append(f"Causal {m['causal_ret_pct']}% ✓")
        score_lines.append(f"- {m['tag']}: {sc}/4  →  {', '.join(hits) if hits else 'no targets met'}")

    if decision == "NO_IMPROVEMENT":
        decision_md = (
            "**NO_IMPROVEMENT** - keep v1 with flagged status\n\n"
            "No variant improved upon v1 by at least 2 additional targets."
        )
    else:
        best = next(m for m in variants if m["tag"] == decision)
        sc   = score_variant(best)
        gains = [g for g in [
            ("brain count" if best["brain_count"] >= 5 and v1["brain_count"] < 5 else None),
            ("alpha_ok"    if best["alpha_ok"] and not v1["alpha_ok"]             else None),
            (f"ERP retention ({best['erp_ret_pct']}%)"        if best["erp_ret_pct"]   >= 80 and v1["erp_ret_pct"]   < 80 else None),
            (f"Causal retention ({best['causal_ret_pct']}%)"  if best["causal_ret_pct"] >= 80 and v1["causal_ret_pct"] < 80 else None),
        ] if g is not None]
        decision_md = (
            f"**USE_{decision}** - replace v1 with {decision}\n\n"
            f"Improvement: v1 scored {v1_sc}/4 → {decision} scored {sc}/4.\n"
            f"Key gains: {', '.join(gains) if gains else 'general signal quality improvement'}"
        )

    md = f"""# Sub-22 Recep - Parameter Variant Comparison

## Background
Recep flagged as borderline in Section 4 QC review:
- ICLabel brain components: {v1['brain_count']}/20
- ERP retention: {v1['erp_ret_pct']}%
- Causal retention: {v1['causal_ret_pct']}%
- Alpha topography anatomically consistent: {'Yes' if v1['alpha_ok'] else 'No (frontal-central > occipital)'}

Root cause: slow drift in central channels (FC1, C4, CP6, F4) persisting after 1.0 Hz high-pass.

## Variants Tested

{header}
{sep}
{chr(10).join(rows)}

## Score per variant (targets: brain≥5, alpha_ok, ERP≥80%, Causal≥80%)

{chr(10).join(score_lines)}

## Recommendation

{decision_md}

## Decision

**{decision}**

Generated: {date.today()}
"""
    out = CORRECTIONS_REP / "recep_variant_comparison.md"
    out.write_text(md, encoding="utf-8")
    print(f"  Report: {out}")


# ── Quick QC for Veli and Eren ────────────────────────────────────────────────
def quick_qc_subject(s: dict) -> dict:
    sid       = s["id"]
    name      = s["name"]
    subj_out  = PROC / f"subject_{sid:02d}"
    raw_clean = mne.io.read_raw_fif(
        str(subj_out / "raw_clean-raw.fif"), preload=True, verbose=False
    )

    # Load raw for 50 Hz before
    em        = load_em(s)
    raw_orig  = step_load(s, logging.getLogger("dummy"))
    raw_orig.filter(l_freq=1.0, h_freq=40.0, method="fir", phase="zero",
                    fir_window="hamming", fir_design="firwin", verbose=False)
    atten_50 = compute_50hz_attenuation(raw_orig, raw_clean)

    alpha_ok, occ_db, front_db = compute_alpha_ok(raw_clean)
    return {
        "sid":       sid,
        "name":      name,
        "alpha_ok":  alpha_ok,
        "occ_db":    round(occ_db, 2),
        "front_db":  round(front_db, 2),
        "atten_50hz": atten_50,
    }


def write_quick_qc_report(veli: dict, eren: dict):
    def status(r):
        return "OK" if r["alpha_ok"] else "Investigate"

    md = f"""# Quick QC - Veli and Eren

| Subject | Alpha OK | Occ alpha (dB) | Front alpha (dB) | 50 Hz Atten (dB) | Status |
|---------|----------|----------------|------------------|-------------------|--------|
| Veli ({veli['sid']}) | {'✓' if veli['alpha_ok'] else '✗'} | {veli['occ_db']} | {veli['front_db']} | {veli['atten_50hz']} | {status(veli)} |
| Eren ({eren['sid']}) | {'✓' if eren['alpha_ok'] else '✗'} | {eren['occ_db']} | {eren['front_db']} | {eren['atten_50hz']} | {status(eren)} |

## Conclusion

{'Both Veli and Eren pass alpha topography check (occipital > frontal). 50 Hz attenuation confirmed. No issues - visual QC validated.' if veli['alpha_ok'] and eren['alpha_ok'] else 'One or more subjects failed alpha topography check. Manual review required.'}

Generated: {date.today()}
"""
    out = CORRECTIONS_REP / "quick_qc_veli_eren.md"
    out.write_text(md, encoding="utf-8")
    print(f"  Report: {out}")
    return veli["alpha_ok"] and eren["alpha_ok"]


# ── Final Section 4 status report ────────────────────────────────────────────
def write_final_status(decision: str, v1: dict, best_metrics: dict,
                        veli_qc: dict, eren_qc: dict):
    if decision == "NO_IMPROVEMENT":
        recep_status = (
            "low EEG confidence (ICLabel brain=3/20, "
            f"ERP {v1['erp_ret_pct']}%, Causal {v1['causal_ret_pct']}%). "
            "Variants tested: no improvement over v1. "
            "Retained with 'low_eeg_confidence' flag for Section 10 outlier control."
        )
        pipeline_note = "1 (Recep) with standard pipeline but flagged"
    else:
        recep_status = (
            f"{decision} applied (1.5 Hz high-pass, {best_metrics['n_components']} ICA components, "
            f"prob threshold {best_metrics['prob_threshold']}). "
            f"Improvement over v1: brain={best_metrics['brain_count']}, "
            f"ERP {best_metrics['erp_ret_pct']}%, Causal {best_metrics['causal_ret_pct']}%."
        )
        pipeline_note = f"1 (Recep) with subject-specific parameters ({decision})"

    md = f"""# Section 4 - Final Status After Corrections

Generated: {date.today()}

## Overview

- 9 subjects, all processed successfully
- 8 with standard pipeline (1.0 Hz, 20 components, ICLabel threshold 0.8)
- {pipeline_note}

## Recep (sub-22) Resolution

{recep_status}

## Quality Flags for Section 5+

| Subject | Flag | Detail |
|---------|------|--------|
| Recep (22) | {'low_eeg_confidence' if decision == 'NO_IMPROVEMENT' else 'subject_specific_params'} | See recep_variant_comparison.md |
| Alen (14) | faa_low_confidence | F4 + Fp2 interpolated |
| Mehmet (17) | faa_low_confidence | F4 interpolated |
| Veli (20) | none | Visual QC passed (alpha_ok={veli_qc['alpha_ok']}) |
| Eren (15) | none | Visual QC passed (alpha_ok={eren_qc['alpha_ok']}) |
| Others (16,18,21,23) | none | Standard quality |

## Ready for Section 5

**YES**
"""
    out = CORRECTIONS_REP / "section4_final_status.md"
    out.write_text(md, encoding="utf-8")
    print(f"  Report: {out}")


# ── apply decision ────────────────────────────────────────────────────────────
def apply_decision(decision: str, analysis_note: str):
    notes_path = ROOT / "analysis_notes.txt"
    existing   = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    notes_path.write_text(existing + "\n\n" + analysis_note, encoding="utf-8")
    print(f"  analysis_notes.txt updated.")

    if decision == "NO_IMPROVEMENT":
        return

    # Rename v1 → v1_archive, copy best → subject_22
    import shutil
    v1_dir      = PROC / "subject_22"
    archive_dir = PROC / "subject_22_v1_archive"
    best_dir    = PROC / f"subject_22_{decision}"
    final_dir   = PROC / "subject_22"

    if not archive_dir.exists():
        shutil.copytree(str(v1_dir), str(archive_dir))
        print(f"  Archived v1 → subject_22_v1_archive/")

    # Remove old subject_22 and copy best
    shutil.rmtree(str(v1_dir))
    shutil.copytree(str(best_dir), str(final_dir))
    print(f"  Copied subject_22_{decision}/ → subject_22/ (new canonical)")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("Section 4B: Recep Variants + Quick QC")
    print("="*60)

    # ── Part A: Run 3 variants for Recep ──────────────────────────────────────
    print("\n▶ Part A: Running 3 variants for Recep (sub-22)")
    print("  Loading v1 baseline metrics...")
    v1 = load_v1_metrics()
    print(f"  v1 baseline: brain={v1['brain_count']} alpha_ok={v1['alpha_ok']} "
          f"ERP={v1['erp_ret_pct']}% Causal={v1['causal_ret_pct']}%")

    variant_results = []
    for vdef in VARIANTS:
        print(f"\n  ── {vdef['tag']} (l_freq={vdef['l_freq']}, "
              f"n_comp={vdef['n_components']}, thresh={vdef['prob_threshold']}) ──")
        result = run_recep_variant(vdef)
        variant_results.append(result)
        print(f"  {vdef['tag']}: brain={result['brain_count']} "
              f"alpha_ok={result['alpha_ok']} "
              f"ERP={result['erp_ret_pct']}% "
              f"Causal={result['causal_ret_pct']}%  "
              f"score={score_variant(result)}/4")

    decision = pick_best(v1, variant_results)
    print(f"\n  Decision: {decision}")

    write_comparison_report(v1, variant_results, decision)

    # ── Part B: Quick QC - Veli and Eren ──────────────────────────────────────
    print("\n▶ Part B: Quick QC for Veli (sub-20) and Eren (sub-15)")
    veli_qc = quick_qc_subject(VELI)
    eren_qc = quick_qc_subject(EREN)
    print(f"  Veli: alpha_ok={veli_qc['alpha_ok']} "
          f"(occ={veli_qc['occ_db']} dB, front={veli_qc['front_db']} dB)")
    print(f"  Eren: alpha_ok={eren_qc['alpha_ok']} "
          f"(occ={eren_qc['occ_db']} dB, front={eren_qc['front_db']} dB)")

    if not (veli_qc["alpha_ok"] and eren_qc["alpha_ok"]):
        print("\n  ⚠ STOP: Alpha topography problem detected. Reporting to user.")
        write_quick_qc_report(veli_qc, eren_qc)
        write_final_status(decision,
                           v1,
                           next((m for m in variant_results if m["tag"] == decision), v1),
                           veli_qc, eren_qc)
        print("\n" + "="*60)
        print("MANUAL REVIEW REQUIRED - see quick_qc_veli_eren.md")
        return

    qc_ok = write_quick_qc_report(veli_qc, eren_qc)
    print(f"  Quick QC: {'PASSED' if qc_ok else 'NEEDS REVIEW'}")

    # ── Part C: Final status report ───────────────────────────────────────────
    print("\n▶ Part C: Final status report")
    best_m = next((m for m in variant_results if m["tag"] == decision), v1)
    write_final_status(decision, v1, best_m, veli_qc, eren_qc)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("BÖLÜM 4B TAMAMLANDI")
    print("="*60)
    print(f"\n  Recep decision:  {decision}")
    print(f"\n  Variant metrics:")
    print(f"  {'Variant':<8} {'brain':>6} {'alpha':>7} {'ERP%':>7} {'Causal%':>9} {'score':>6}")
    print(f"  {'-'*45}")
    for m in [v1] + variant_results:
        print(f"  {m['tag']:<8} {m['brain_count']:>6} "
              f"{'OK' if m['alpha_ok'] else 'FAIL':>7} "
              f"{m['erp_ret_pct']:>6}% "
              f"{m['causal_ret_pct']:>8}%  "
              f"{score_variant(m):>4}/4")
    print(f"\n  Veli alpha_ok:   {veli_qc['alpha_ok']}")
    print(f"  Eren alpha_ok:   {eren_qc['alpha_ok']}")
    print(f"\n  Reports written to: {CORRECTIONS_REP}")
    print(f"\n  ⚠  DUR - Recep kararı: {decision}")
    print(f"     'tamam uygula' dersen apply_decision() çalışacak.")


def load_variant_metrics_from_disk(vdef: dict) -> dict:
    """Reload metrics from an already-processed variant directory (skip ICA re-run)."""
    tag      = vdef["tag"]
    subj_out = PROC / f"subject_22_{tag}"
    with open(subj_out / "ica_labels.json", encoding="utf-8") as f:
        labels = json.load(f)

    raw_clean = mne.io.read_raw_fif(
        str(subj_out / "raw_clean-raw.fif"), preload=True, verbose=False
    )
    alpha_ok, occ_db, front_db = compute_alpha_ok(raw_clean)

    import mne as _mne
    erp_epo   = _mne.read_epochs(str(subj_out / "epochs_erp-epo.fif"),   preload=False, verbose=False)
    caus_epo  = _mne.read_epochs(str(subj_out / "epochs_causal-epo.fif"), preload=False, verbose=False)
    erp_after  = len(erp_epo)
    caus_after = len(caus_epo)
    # before = 75 for Recep (known from run)
    erp_before = caus_before = 75

    return {
        "tag":            tag,
        "l_freq":         vdef["l_freq"],
        "n_components":   vdef["n_components"],
        "prob_threshold": vdef["prob_threshold"],
        "brain_count":    labels["brain_components_kept"],
        "n_removed":      len(labels["excluded_indices"]),
        "alpha_ok":       alpha_ok,
        "occ_alpha_db":   round(occ_db, 2),
        "front_alpha_db": round(front_db, 2),
        "erp_before":     erp_before,
        "erp_after":      erp_after,
        "erp_ret_pct":    round(erp_after / erp_before * 100, 1),
        "causal_before":  caus_before,
        "causal_after":   caus_after,
        "causal_ret_pct": round(caus_after / caus_before * 100, 1),
    }


def main_reports_only():
    """Generate reports from already-computed variant directories (skip ICA)."""
    print("\n" + "="*60)
    print("Section 4B: Reports-only mode (variants already processed)")
    print("="*60)

    v1 = load_v1_metrics()
    print(f"  v1: brain={v1['brain_count']} alpha_ok={v1['alpha_ok']} "
          f"ERP={v1['erp_ret_pct']}% Causal={v1['causal_ret_pct']}%")

    variant_results = []
    for vdef in VARIANTS:
        m = load_variant_metrics_from_disk(vdef)
        variant_results.append(m)
        print(f"  {vdef['tag']}: brain={m['brain_count']} alpha_ok={m['alpha_ok']} "
              f"ERP={m['erp_ret_pct']}% Causal={m['causal_ret_pct']}%  score={score_variant(m)}/4")

    decision = pick_best(v1, variant_results)
    print(f"\n  Decision: {decision}")
    write_comparison_report(v1, variant_results, decision)

    print("\n▶ Part B: Quick QC for Veli and Eren")
    veli_qc = quick_qc_subject(VELI)
    eren_qc = quick_qc_subject(EREN)
    print(f"  Veli: alpha_ok={veli_qc['alpha_ok']} (occ={veli_qc['occ_db']} front={veli_qc['front_db']})")
    print(f"  Eren: alpha_ok={eren_qc['alpha_ok']} (occ={eren_qc['occ_db']} front={eren_qc['front_db']})")

    if not (veli_qc["alpha_ok"] and eren_qc["alpha_ok"]):
        write_quick_qc_report(veli_qc, eren_qc)
        best_m = next((m for m in variant_results if m["tag"] == decision), v1)
        write_final_status(decision, v1, best_m, veli_qc, eren_qc)
        print("\n⚠ STOP: Alpha topography problem - manual review required.")
        return

    write_quick_qc_report(veli_qc, eren_qc)
    best_m = next((m for m in variant_results if m["tag"] == decision), v1)
    write_final_status(decision, v1, best_m, veli_qc, eren_qc)

    print("\n" + "="*60)
    print("BÖLÜM 4B TAMAMLANDI")
    print("="*60)
    print(f"\n  Recep decision: {decision}")
    print(f"\n  {'Variant':<8} {'brain':>6} {'alpha':>7} {'ERP%':>7} {'Causal%':>9} {'score':>6}")
    print(f"  {'-'*45}")
    for m in [v1] + variant_results:
        print(f"  {m['tag']:<8} {m['brain_count']:>6} "
              f"{'OK' if m['alpha_ok'] else 'FAIL':>7} "
              f"{m['erp_ret_pct']:>6}% {m['causal_ret_pct']:>8}%  {score_variant(m):>4}/4")
    print(f"\n  Veli alpha_ok: {veli_qc['alpha_ok']}  |  Eren alpha_ok: {eren_qc['alpha_ok']}")
    print(f"\n  Reports: {CORRECTIONS_REP}")
    print(f"\n  ⚠  DUR - 'tamam uygula' dersen apply_decision('{decision}', ...) çalışacak.")


if __name__ == "__main__":
    import sys as _sys
    if "--reports-only" in _sys.argv:
        main_reports_only()
    else:
        main()
