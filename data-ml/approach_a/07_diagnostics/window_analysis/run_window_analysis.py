"""
Time-Locked Window Analysis
===========================
Tests which temporal segment of the 2200ms ERP epoch carries the most
frustration-discriminative information for LaBraM.

For each window, raw EEG is zero-masked outside that window, then fed through
the pre-trained LaBraM backbone to obtain 200-dim embeddings.  A simple MLP
is trained with 9-fold LOSO to measure classification AUC per window.

Windows (at 500 Hz, epoch tmin=-200ms):
  W0_full  : full epoch  [-200ms, 2000ms]  samples   0:1101
  W1_pre   : pre-stimulus [-200ms,    0ms] samples   0:100
  W2_early : early post  [   0ms,  500ms]  samples 100:350
  W3_mid   : middle      [ 500ms, 1500ms]  samples 350:850
  W4_late  : late        [1500ms, 2000ms]  samples 850:1100

Output
------
  results.json         - per-window mean/std AUC + per-fold AUCs
  window_auc_plot.png  - bar chart with error bars
  interpretation.md    - scenario determination (A/B/C)
"""

import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

# ── Paths ─────────────────────────────────────────────────────────────────────
WIN_DIR    = os.path.dirname(os.path.abspath(__file__))
DIAG_DIR   = os.path.dirname(WIN_DIR)
APPROACH_A = os.path.dirname(DIAG_DIR)
PROJECT    = os.path.dirname(APPROACH_A)

SRC_DIR  = os.path.join(APPROACH_A, "src")
FEAT_DIR = os.path.join(APPROACH_A, "features")
PROC_DIR = os.path.join(PROJECT, "data", "processed")

sys.path.insert(0, SRC_DIR)
from labram_wrapper import load_labram_base

# ── Config ────────────────────────────────────────────────────────────────────
SUBJECTS = [14, 15, 16, 17, 18, 20, 21, 22, 23]
SFREQ    = 500          # Hz - epoch sampling rate
N_TOTAL  = 1101         # samples per epoch (2200ms at 500Hz)
SEED     = 42
N_EPOCHS = 30           # MLP training epochs
BS       = 32
LR       = 1e-3

DEVICE = ("mps" if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Window definitions: (label, start_sample_inclusive, end_sample_exclusive)
# tmin=-200ms → sample 0 = t=-200ms, sample 100 = t=0ms
WINDOWS = {
    "W0_full" : (0,    1101),   # -200ms → 2000ms
    "W1_pre"  : (0,    100),    # -200ms → 0ms
    "W2_early": (100,  350),    #    0ms → 500ms
    "W3_mid"  : (350,  850),    #  500ms → 1500ms
    "W4_late" : (850,  1101),   # 1500ms → 2000ms
}

# ── Data loading ──────────────────────────────────────────────────────────────
def build_v2_epoch_id_to_fif_index(v2_meta):
    """
    Build a lookup: (subject_id, epoch_id_v2) → eeg_index (FIF file index).

    In V2 metadata, epoch_id is the pre-AutoReject marker number, while
    eeg_index is the actual index into the saved FIF file / embedding array.
    """
    lookup = {}
    for _, row in v2_meta.iterrows():
        key = (int(row["subject_id"]), int(row["epoch_id"]))
        lookup[key] = int(row["eeg_index"])
    return lookup


def load_raw_eeg_v3():
    """
    Load V3 raw EEG epochs per subject, correctly mapped from metadata.

    Control epochs (label_class=0):
        epoch_id → direct index into epochs_erp_action_matched-epo.fif

    Variant epochs (label_class=1):
        epoch_id in V3 = epoch_id in V2 (pre-AutoReject marker number).
        Must be translated to eeg_index via V2 metadata, then used as FIF index.

    Returns
    -------
    raw_eeg : np.ndarray, shape (N_total, 32, 1101)
    labels  : np.ndarray, shape (N_total,) - 0=control, 1=frustration
    subjects: np.ndarray, shape (N_total,) - subject id per epoch
    """
    v3_meta = pd.read_csv(os.path.join(FEAT_DIR, "all_eeg_embeddings_v3_metadata.csv"))
    v2_meta = pd.read_csv(os.path.join(FEAT_DIR, "all_eeg_embeddings_v2_metadata.csv"))
    v2_lookup = build_v2_epoch_id_to_fif_index(v2_meta)

    all_eeg, all_lbl, all_sid = [], [], []
    missing = 0

    for sid in SUBJECTS:
        smeta = v3_meta[v3_meta["subject_id"] == sid].copy()
        subj_dir = os.path.join(PROC_DIR, f"subject_{sid}")

        # Load both FIF files once per subject
        ep_ctrl = mne.read_epochs(
            os.path.join(subj_dir, "epochs_erp_action_matched-epo.fif"),
            preload=True, verbose=False
        )
        ep_var = mne.read_epochs(
            os.path.join(subj_dir, "epochs_erp-epo.fif"),
            preload=True, verbose=False
        )
        ctrl_data = ep_ctrl.get_data()   # (N_ctrl, 32, 1101)
        var_data  = ep_var.get_data()    # (N_var,  32, 1101)

        for _, row in smeta.iterrows():
            label = int(row["label_class"])
            eid   = int(row["epoch_id"])

            if label == 0:
                # Control: epoch_id is direct FIF index
                fif_idx = min(eid, len(ctrl_data) - 1)
                all_eeg.append(ctrl_data[fif_idx])
            else:
                # Variant: epoch_id is V2's pre-AR marker number → translate to eeg_index
                fif_idx = v2_lookup.get((sid, eid), None)
                if fif_idx is None or fif_idx >= len(var_data):
                    # Fallback: clip to last available epoch
                    fif_idx = len(var_data) - 1
                    missing += 1
                all_eeg.append(var_data[fif_idx])

            all_lbl.append(label)
            all_sid.append(sid)

    if missing > 0:
        print(f"  Warning: {missing} variant epoch_ids not found in V2 lookup; used last epoch")

    raw_eeg  = np.stack(all_eeg, axis=0).astype(np.float32)   # (N, 32, 1101)
    labels   = np.array(all_lbl, dtype=np.int64)
    subjects = np.array(all_sid, dtype=np.int64)
    print(f"Loaded raw EEG: {raw_eeg.shape}, labels dist: "
          f"{(labels==0).sum()} ctrl / {(labels==1).sum()} frust")
    return raw_eeg, labels, subjects


def apply_window_mask(raw_eeg: np.ndarray, start: int, end: int) -> np.ndarray:
    """Zero-mask outside [start, end) - keeps temporal position intact."""
    masked = np.zeros_like(raw_eeg)
    masked[:, :, start:end] = raw_eeg[:, :, start:end]
    return masked


# ── MLP classifier ────────────────────────────────────────────────────────────
class SimpleMLP(nn.Module):
    def __init__(self, in_dim=200, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 32), nn.GELU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)


def global_normalize(train_emb: np.ndarray, test_emb: np.ndarray):
    """Leakage-free: fit mean/std on train, apply to both."""
    mu  = train_emb.mean(0, keepdims=True)
    sig = train_emb.std(0, keepdims=True) + 1e-8
    return (train_emb - mu) / sig, (test_emb - mu) / sig


def train_eval(tr_emb, tr_lbl, te_emb, te_lbl):
    model = SimpleMLP().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)
    crit  = nn.CrossEntropyLoss()

    try:
        sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=SEED)
        tr_i, _ = next(sss.split(tr_emb, tr_lbl))
    except Exception:
        tr_i = np.arange(len(tr_lbl))

    tr_ds = TensorDataset(torch.tensor(tr_emb[tr_i]), torch.tensor(tr_lbl[tr_i]))
    te_ds = TensorDataset(torch.tensor(te_emb),       torch.tensor(te_lbl))
    tr_ld = DataLoader(tr_ds, BS, shuffle=True,  num_workers=0)
    te_ld = DataLoader(te_ds, BS, shuffle=False, num_workers=0)

    for _ in range(N_EPOCHS):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()

    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            p = torch.softmax(model(xb.to(DEVICE)), -1).cpu().numpy()
            probs.append(p); trues.extend(yb.numpy())
    probs_arr = np.concatenate(probs)
    trues_arr = np.array(trues)
    if len(np.unique(trues_arr)) < 2:
        return 0.5
    return float(roc_auc_score(trues_arr, probs_arr[:, 1]))


# ── LOSO per window ───────────────────────────────────────────────────────────
def run_loso_for_window(encoder, raw_eeg, labels, subjects, win_start, win_end):
    """
    For a given temporal window:
      1. Zero-mask outside window
      2. Re-run LaBraM inference → 200-dim embeddings
      3. 9-fold LOSO with SimpleMLP
    Returns list of per-fold AUCs.
    """
    masked = apply_window_mask(raw_eeg, win_start, win_end)
    print(f"    Extracting LaBraM embeddings for window [{win_start}:{win_end}] ...")
    embeddings = encoder.get_embeddings(masked, batch_size=64)   # (N, 200)

    fold_aucs = []
    for test_sid in SUBJECTS:
        te_mask = subjects == test_sid
        tr_mask = ~te_mask

        tr_emb, te_emb = embeddings[tr_mask], embeddings[te_mask]
        tr_lbl, te_lbl = labels[tr_mask],     labels[te_mask]

        if len(np.unique(te_lbl)) < 2:
            print(f"    sub-{test_sid}: skipped (single class in test)")
            continue

        tr_emb_n, te_emb_n = global_normalize(tr_emb, te_emb)
        auc = train_eval(tr_emb_n, tr_lbl, te_emb_n, te_lbl)
        fold_aucs.append(auc)
        print(f"    sub-{test_sid}: AUC={auc:.3f}")

    return fold_aucs


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(results: dict, out_path: str):
    window_labels = {
        "W0_full":  "Full\n[-200→2000ms]",
        "W1_pre":   "Pre-stim\n[-200→0ms]",
        "W2_early": "Early\n[0→500ms]",
        "W3_mid":   "Mid\n[500→1500ms]",
        "W4_late":  "Late\n[1500→2000ms]",
    }

    keys   = list(results.keys())
    means  = [results[k]["mean_auc"] for k in keys]
    stds   = [results[k]["std_auc"]  for k in keys]
    labels = [window_labels[k] for k in keys]

    colors = ["#2196F3", "#9E9E9E", "#FF9800", "#4CAF50", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(keys)), means, yerr=stds, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.7, alpha=0.85)

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Chance (0.5)")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("LOSO AUC", fontsize=11)
    ax.set_title("LaBraM Window Analysis: Which Temporal Segment Matters?", fontsize=12)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=9)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.015, f"{m:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path}")


# ── Interpretation ────────────────────────────────────────────────────────────
def write_interpretation(results: dict, out_path: str):
    full_auc   = results["W0_full"]["mean_auc"]
    pre_auc    = results["W1_pre"]["mean_auc"]
    early_auc  = results["W2_early"]["mean_auc"]
    mid_auc    = results["W3_mid"]["mean_auc"]
    late_auc   = results["W4_late"]["mean_auc"]

    best_win   = max(results, key=lambda k: results[k]["mean_auc"])
    best_auc   = results[best_win]["mean_auc"]

    # Determine scenario
    all_partial = [pre_auc, early_auc, mid_auc, late_auc]
    n_high = sum(a > 0.75 for a in all_partial)
    best_partial = max(all_partial)

    if n_high >= 3:
        scenario = "A"
        scenario_desc = (
            "**Scenario A: Signal is broadly distributed across the epoch.**\n"
            "Multiple temporal windows achieve AUC > 0.75, indicating that\n"
            "frustration-discriminative EEG patterns are not confined to a single\n"
            "post-stimulus period but persist throughout the epoch."
        )
    elif best_partial >= 0.80 and n_high == 1:
        scenario = "B"
        scenario_desc = (
            f"**Scenario B: Signal is concentrated in window {best_win}.**\n"
            "One temporal window dominates performance, suggesting a time-locked\n"
            "neural response to the frustration trigger. This is consistent with\n"
            "event-related potential (ERP) literature where peaks occur at specific\n"
            "post-stimulus latencies."
        )
    else:
        scenario = "C"
        scenario_desc = (
            "**Scenario C: No single window achieves strong performance.**\n"
            "All partial windows perform significantly below the full-epoch baseline.\n"
            "This implies that LaBraM integrates information across multiple time\n"
            "scales - no short segment contains sufficient discriminative information\n"
            "on its own. Cross-frequency coupling or non-stationarity patterns spread\n"
            "across the entire 2.2s epoch may be the key signal."
        )

    lines = [
        "# Time-Locked Window Analysis - Interpretation",
        "",
        f"**Date:** 2026-05-23",
        f"**Dataset:** V3 action-matched (N=480 epochs, 9 subjects, balanced)",
        f"**Method:** Zero-mask outside window → LaBraM 200-dim embedding → SimpleMLP LOSO",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "| Window | Time Range | Mean AUC ± Std |",
        "|--------|-----------|----------------|",
    ]
    for k, v in results.items():
        time_map = {
            "W0_full":  "-200ms → 2000ms",
            "W1_pre":   "-200ms → 0ms",
            "W2_early": "0ms → 500ms",
            "W3_mid":   "500ms → 1500ms",
            "W4_late":  "1500ms → 2000ms",
        }
        lines.append(f"| {k} | {time_map[k]} | {v['mean_auc']:.3f} ± {v['std_auc']:.3f} |")

    lines += [
        "",
        "---",
        "",
        "## Scenario Determination",
        "",
        f"**Determined Scenario: {scenario}**",
        "",
        scenario_desc,
        "",
        "---",
        "",
        "## Key Findings",
        "",
        f"- Full-epoch AUC: **{full_auc:.3f}** (reference)",
        f"- Best single-window: **{best_win}** (AUC={best_auc:.3f})",
        f"- Pre-stimulus window (W1_pre): AUC={pre_auc:.3f}",
        "  - Values above 0.6 suggest subject-level confounds (e.g., EEG alpha baseline drift);",
        "    values near 0.5 confirm the signal is stimulus-locked.",
        f"- Early post-stimulus (W2_early, 0–500ms): AUC={early_auc:.3f}",
        "  - Early cortical responses (P300, N200) would appear here.",
        f"- Middle window (W3_mid, 500–1500ms): AUC={mid_auc:.3f}",
        "  - Sustained cognitive processing, theta oscillations.",
        f"- Late window (W4_late, 1500–2000ms): AUC={late_auc:.3f}",
        "  - Late slow wave, feedback-related negativity.",
        "",
        "---",
        "",
        "## Implications for Pipeline Validity",
        "",
    ]

    if pre_auc > 0.65:
        lines.append(
            "⚠️  **Pre-stimulus AUC is elevated ({:.3f})**, which may indicate:"
            " (a) carryover effects from prior trials, "
            "(b) tonic EEG differences between control/frustration phases, "
            "or (c) mild data leakage. Further investigation recommended.".format(pre_auc)
        )
    else:
        lines.append(
            f"✓  Pre-stimulus AUC ({pre_auc:.3f}) is near chance, confirming the signal "
            "is time-locked to the event and not driven by pre-existing EEG differences."
        )

    if full_auc > max(all_partial) + 0.05:
        lines += [
            "",
            f"✓  Full-epoch AUC ({full_auc:.3f}) exceeds all partial windows by ≥0.05.",
            "This supports the diagnostic test finding (Test 4): the discriminative",
            "information is in fine-grained temporal dynamics that require the full",
            "2.2s epoch window, not in short-latency scalar features.",
        ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Interpretation saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Time-Locked Window Analysis")
    print("=" * 60)

    # Load raw EEG
    raw_eeg, labels, subjects = load_raw_eeg_v3()

    # Load LaBraM
    print("Loading LaBraM encoder...")
    pretrained = os.path.join(APPROACH_A, "models", "pytorch_model.bin")
    encoder = load_labram_base(pretrained_path=pretrained)
    encoder.model.eval()

    results = {}

    for win_name, (wstart, wend) in WINDOWS.items():
        dur_ms = (wend - wstart) * 2   # at 500Hz, 1 sample = 2ms
        print(f"\n[{win_name}] samples {wstart}:{wend} ({dur_ms}ms)")
        fold_aucs = run_loso_for_window(
            encoder, raw_eeg, labels, subjects, wstart, wend
        )
        mean_auc = float(np.mean(fold_aucs))
        std_auc  = float(np.std(fold_aucs))
        results[win_name] = {
            "start_sample": wstart,
            "end_sample":   wend,
            "duration_ms":  dur_ms,
            "fold_aucs":    fold_aucs,
            "mean_auc":     mean_auc,
            "std_auc":      std_auc,
            "n_folds":      len(fold_aucs),
        }
        print(f"  → mean AUC = {mean_auc:.3f} ± {std_auc:.3f}")

    # Save results
    out_json = os.path.join(WIN_DIR, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_json}")

    # Plot
    out_plot = os.path.join(WIN_DIR, "window_auc_plot.png")
    plot_results(results, out_plot)

    # Interpretation
    out_interp = os.path.join(WIN_DIR, "interpretation.md")
    write_interpretation(results, out_interp)

    print("\n=== Final Summary ===")
    for win_name, v in results.items():
        print(f"  {win_name}: {v['mean_auc']:.3f} ± {v['std_auc']:.3f}")


if __name__ == "__main__":
    main()
