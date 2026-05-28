"""
compute_connectivity.py
=======================
stage 1 of approach b. computes per-epoch functional connectivity on the
v3 action-matched dataset (480 epochs, 9 subjects).

primary metric: weighted phase lag index (wpli, vinck et al. 2011). robust
to volume conduction at sensor level.
secondary metric: amplitude envelope correlation (aec). complements phase-
based wpli with amplitude-based dependence.

four bands: theta 4-8, alpha 8-13, beta 13-30, gamma 30-40 hz.
six rois: frontal, frontal_central, central, parietal, occipital, temporal.

per-epoch wpli computed via mne_connectivity.spectral_connectivity_time with
morlet wavelets so that each epoch yields its own connectivity matrix. aec
uses bandpass + hilbert envelope + pearson correlation per epoch.

outputs (approach_b/01_connectivity_extraction/features/connectivity_per_epoch):
  all_wpli_v3.npy            (480, 4, 6, 6)
  all_aec_v3.npy             (480, 4, 6, 6)
  labels_v3.csv              copy of v3 labels with scenario_name
  connectivity_metadata.csv  per-epoch metadata
qc:
  approach_b/01_connectivity_extraction/reports/connectivity_qc.png
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.signal import hilbert
from mne_connectivity import spectral_connectivity_time

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

np.random.seed(42)

#paths
B_DIR    = Path(__file__).resolve().parents[2]
STAGE1   = B_DIR / "01_connectivity_extraction"
FEAT_OUT = STAGE1 / "features" / "connectivity_per_epoch"
REP_OUT  = STAGE1 / "reports"
FEAT_OUT.mkdir(parents=True, exist_ok=True)
REP_OUT.mkdir(parents=True, exist_ok=True)

PROJECT  = B_DIR.parent
PROC_DIR = PROJECT / "data" / "processed"
V3_FEAT  = PROJECT / "approach_a" / "04_v3_labram_action_matched" / "features"
V2_FEAT  = PROJECT / "approach_a" / "03_v2_labram_pseudo_control" / "features"
V6_FEAT  = PROJECT / "approach_a" / "06_v6_multiclass_characterization" / "features"

SUBJECTS = [14, 15, 16, 17, 18, 20, 21, 22, 23]
SFREQ    = 500.0

#channel layout as in approach a v6
CH_NAMES = ['Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','TP9','CP5','CP1',
            'Pz','P3','P7','O1','Oz','O2','P4','P8','TP10','CP6','CP2','Cz',
            'C4','T8','FT10','FC6','FC2','F4','F8','Fp2']

def _ch_idx(names):
    return [CH_NAMES.index(n) for n in names if n in CH_NAMES]

ROI_CHANNELS = {
    "frontal":         _ch_idx(['Fp1','Fp2','F3','Fz','F4','F7','F8']),
    "frontal_central": _ch_idx(['FC1','FC2','FC5','FC6']),
    "central":         _ch_idx(['C3','Cz','C4']),
    "parietal":        _ch_idx(['P3','Pz','P4','P7','P8']),
    "occipital":       _ch_idx(['O1','Oz','O2']),
    "temporal":        _ch_idx(['T7','T8','TP9','TP10']),
}
ROI_NAMES = list(ROI_CHANNELS.keys())
N_ROI     = len(ROI_NAMES)

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
BAND_NAMES = list(BANDS.keys())
N_BAND     = len(BAND_NAMES)

#frequencies for the wavelet decomposition (1 hz steps in low bands, 2 hz in gamma)
FREQS = np.concatenate([
    np.arange(4, 8, 1.0),     #theta
    np.arange(8, 13, 1.0),    #alpha
    np.arange(13, 30, 2.0),   #beta
    np.arange(30, 41, 2.0),   #gamma
]).astype(float)
N_CYCLES = np.maximum(FREQS / 2.0, 3.0)


def load_subject_epochs(sid: int):
    """load action-matched control + variant epochs for a subject.
    returns (ctrl_data, var_data) each (n_ep, 32, n_times)."""
    subj = PROC_DIR / f"subject_{sid}"
    ep_ctrl = mne.read_epochs(str(subj / "epochs_erp_action_matched-epo.fif"),
                              preload=True, verbose=False)
    ep_var  = mne.read_epochs(str(subj / "epochs_erp-epo.fif"),
                              preload=True, verbose=False)
    ctrl = ep_ctrl.get_data(picks=CH_NAMES).astype(np.float32)
    var  = ep_var.get_data(picks=CH_NAMES).astype(np.float32)
    return ctrl, var


def upper_pairs(n_signals: int):
    """return (sources, targets) for upper-triangle channel pairs."""
    src, tgt = np.triu_indices(n_signals, k=1)
    return src.tolist(), tgt.tolist()


def per_epoch_wpli(data: np.ndarray, indices) -> np.ndarray:
    """compute per-epoch wpli for all channel pairs across the four bands.
    data: (n_ep, n_ch, n_times)
    returns: (n_ep, n_band, n_pairs)
    """
    fmin = tuple(b[0] for b in BANDS.values())
    fmax = tuple(b[1] for b in BANDS.values())
    con = spectral_connectivity_time(
        data,
        freqs=FREQS,
        method='wpli',
        average=False,
        indices=indices,
        sfreq=SFREQ,
        mode='cwt_morlet',
        n_cycles=N_CYCLES,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        padding=0,
        n_jobs=-1,
        verbose=False,
    )
    arr = np.asarray(con.get_data())     # (n_ep, n_pairs, n_band)
    arr = np.transpose(arr, (0, 2, 1))   # (n_ep, n_band, n_pairs)
    arr = np.abs(arr).astype(np.float32) #wpli is already in [0,1] but abs keeps the typing tidy
    return arr


def per_epoch_aec(data: np.ndarray, src, tgt) -> np.ndarray:
    """compute per-epoch amplitude envelope correlation for the given
    channel pairs across the four bands.
    data: (n_ep, n_ch, n_times)
    returns: (n_ep, n_band, n_pairs)
    """
    n_ep, n_ch, n_t = data.shape
    out = np.zeros((n_ep, N_BAND, len(src)), dtype=np.float32)
    for b_idx, (b_name, (fmin, fmax)) in enumerate(BANDS.items()):
        #mne.filter expects (..., n_times)
        filt = mne.filter.filter_data(
            data.astype(np.float64), SFREQ, fmin, fmax,
            verbose=False, n_jobs=-1, l_trans_bandwidth='auto',
            h_trans_bandwidth='auto', filter_length='auto',
        ).astype(np.float32)
        env = np.abs(hilbert(filt, axis=-1)).astype(np.float32)
        #pearson correlation per epoch per pair
        env_mean = env.mean(axis=-1, keepdims=True)
        env_dem  = env - env_mean
        env_std  = env_dem.std(axis=-1) + 1e-12
        for p_idx, (i, j) in enumerate(zip(src, tgt)):
            num = (env_dem[:, i] * env_dem[:, j]).mean(axis=-1)
            r   = num / (env_std[:, i] * env_std[:, j])
            out[:, b_idx, p_idx] = r
    return out


def reduce_pairs_to_rois(pair_values: np.ndarray, src, tgt) -> np.ndarray:
    """collapse channel pair values into a symmetric 6x6 roi matrix.
    pair_values: (n_ep, n_band, n_pairs)
    returns: (n_ep, n_band, n_roi, n_roi)
    """
    n_ep, n_band, _ = pair_values.shape
    out = np.zeros((n_ep, n_band, N_ROI, N_ROI), dtype=np.float32)

    #map each channel to its roi index for fast lookup
    ch_to_roi = -np.ones(len(CH_NAMES), dtype=int)
    for r_idx, (roi, idxs) in enumerate(ROI_CHANNELS.items()):
        for ch in idxs:
            ch_to_roi[ch] = r_idx

    src_arr = np.asarray(src, dtype=int)
    tgt_arr = np.asarray(tgt, dtype=int)
    src_roi = ch_to_roi[src_arr]
    tgt_roi = ch_to_roi[tgt_arr]
    valid   = (src_roi >= 0) & (tgt_roi >= 0)

    #for within-roi values we additionally need same-channel diagonal
    #estimates. for connectivity self-pairs are undefined, so within-roi
    #values use only cross-channel pairs inside that roi.
    for a in range(N_ROI):
        for b in range(a, N_ROI):
            if a == b:
                mask = valid & (src_roi == a) & (tgt_roi == a)
            else:
                mask = valid & (
                    ((src_roi == a) & (tgt_roi == b)) |
                    ((src_roi == b) & (tgt_roi == a))
                )
            if mask.sum() == 0:
                continue
            vals = pair_values[:, :, mask].mean(axis=-1)
            out[:, :, a, b] = vals
            out[:, :, b, a] = vals
    return out


def compute_subject_connectivity(sid: int):
    """returns dict with ctrl/var wpli + aec roi matrices for one subject."""
    ctrl, var = load_subject_epochs(sid)
    n_ch = ctrl.shape[1]
    src, tgt = upper_pairs(n_ch)
    indices  = (np.asarray(src, dtype=int), np.asarray(tgt, dtype=int))

    print(f"  sub-{sid}: ctrl {ctrl.shape}, var {var.shape}")

    print(f"  sub-{sid}: wpli (ctrl)")
    w_ctrl = per_epoch_wpli(ctrl, indices)
    print(f"  sub-{sid}: wpli (var)")
    w_var  = per_epoch_wpli(var, indices)

    print(f"  sub-{sid}: aec (ctrl)")
    a_ctrl = per_epoch_aec(ctrl, src, tgt)
    print(f"  sub-{sid}: aec (var)")
    a_var  = per_epoch_aec(var, src, tgt)

    return {
        "wpli_ctrl": reduce_pairs_to_rois(w_ctrl, src, tgt),
        "wpli_var":  reduce_pairs_to_rois(w_var,  src, tgt),
        "aec_ctrl":  reduce_pairs_to_rois(a_ctrl, src, tgt),
        "aec_var":   reduce_pairs_to_rois(a_var,  src, tgt),
    }


def build_v2_lookup(v2_meta: pd.DataFrame) -> dict:
    return {(int(r.subject_id), int(r.epoch_id)): int(r.eeg_index)
            for _, r in v2_meta.iterrows()}


def assemble_dataset():
    """concatenate per-subject connectivity into the (480, 4, 6, 6) arrays
    in v3 metadata order, mirroring the approach used in v6 tfr extraction."""
    v3_meta = pd.read_csv(str(V3_FEAT / "all_eeg_embeddings_v3_metadata.csv"))
    v2_meta = pd.read_csv(str(V2_FEAT / "all_eeg_embeddings_v2_metadata.csv"))
    v3_lab  = pd.read_csv(str(V3_FEAT / "labels_v3.csv"))
    v6_lab  = pd.read_csv(str(V6_FEAT / "labels_v6.csv"))
    v2_lookup = build_v2_lookup(v2_meta)

    n_total = len(v3_meta)
    all_wpli = np.zeros((n_total, N_BAND, N_ROI, N_ROI), dtype=np.float32)
    all_aec  = np.zeros((n_total, N_BAND, N_ROI, N_ROI), dtype=np.float32)
    meta_rows = []

    for sid in SUBJECTS:
        print(f"\n=== subject {sid} ===")
        smeta = v3_meta[v3_meta["subject_id"] == sid].sort_values("global_idx")
        if smeta.empty:
            continue
        d = compute_subject_connectivity(sid)

        for _, row in smeta.iterrows():
            g_idx = int(row["global_idx"])
            label = int(row["label_class"])
            eid   = int(row["epoch_id"])
            scen  = str(row["scenario_name"])

            if label == 0:
                fif_idx = min(eid, d["wpli_ctrl"].shape[0] - 1)
                all_wpli[g_idx] = d["wpli_ctrl"][fif_idx]
                all_aec[g_idx]  = d["aec_ctrl"][fif_idx]
            else:
                fif_idx = v2_lookup.get((sid, eid))
                if fif_idx is None or fif_idx >= d["wpli_var"].shape[0]:
                    fif_idx = d["wpli_var"].shape[0] - 1
                all_wpli[g_idx] = d["wpli_var"][fif_idx]
                all_aec[g_idx]  = d["aec_var"][fif_idx]

            meta_rows.append({
                "global_idx": g_idx,
                "subject_id": sid,
                "epoch_id":   eid,
                "scenario_name": scen,
                "label_binary":  label,
            })

    meta = pd.DataFrame(meta_rows).sort_values("global_idx").reset_index(drop=True)

    #save outputs
    np.save(str(FEAT_OUT / "all_wpli_v3.npy"), all_wpli)
    np.save(str(FEAT_OUT / "all_aec_v3.npy"),  all_aec)
    v3_lab.to_csv(str(FEAT_OUT / "labels_v3.csv"), index=False)
    meta.to_csv(str(FEAT_OUT / "connectivity_metadata.csv"), index=False)

    #also align v6 scenario labels with our metadata for downstream stages
    v6_lab_sorted = v6_lab.sort_values("global_idx").reset_index(drop=True)
    v6_lab_sorted.to_csv(str(FEAT_OUT / "labels_v6.csv"), index=False)

    print(f"\nsaved arrays to {FEAT_OUT}")
    print(f"  all_wpli_v3.npy {all_wpli.shape}")
    print(f"  all_aec_v3.npy  {all_aec.shape}")
    print(f"  labels_v3.csv   {len(v3_lab)} rows")
    print(f"  labels_v6.csv   {len(v6_lab_sorted)} rows")

    return all_wpli, all_aec, meta


def qc_plot(wpli: np.ndarray, aec: np.ndarray):
    """quick sanity check: per-band wpli/aec distribution histograms and
    average roi-roi heatmaps for the dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 4, figsize=(13, 11))
    for b_idx, b_name in enumerate(BAND_NAMES):
        #wpli histogram (off-diagonal only)
        mask_off = ~np.eye(N_ROI, dtype=bool)
        w_vals = wpli[:, b_idx][:, mask_off].ravel()
        a_vals = aec[:, b_idx][:, mask_off].ravel()

        ax = axes[b_idx, 0]
        ax.hist(w_vals, bins=40, color="0.3", edgecolor="black")
        ax.set_title(f"wPLI {b_name}")
        ax.set_xlabel("wPLI")

        ax = axes[b_idx, 1]
        ax.hist(a_vals, bins=40, color="0.3", edgecolor="black")
        ax.set_title(f"AEC {b_name}")
        ax.set_xlabel("AEC")

        #mean matrices
        m_w = wpli[:, b_idx].mean(axis=0)
        m_a = aec[:, b_idx].mean(axis=0)

        ax = axes[b_idx, 2]
        im = ax.imshow(m_w, cmap="gray", vmin=0.0, vmax=max(0.01, m_w.max()))
        ax.set_xticks(range(N_ROI), ROI_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(N_ROI), ROI_NAMES, fontsize=7)
        ax.set_title(f"mean wPLI {b_name}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[b_idx, 3]
        im = ax.imshow(m_a, cmap="gray", vmin=0.0, vmax=max(0.01, m_a.max()))
        ax.set_xticks(range(N_ROI), ROI_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(N_ROI), ROI_NAMES, fontsize=7)
        ax.set_title(f"mean AEC {b_name}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = REP_OUT / "connectivity_qc.png"
    fig.savefig(str(out_path), dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  qc plot -> {out_path}")


def print_sanity(wpli: np.ndarray, aec: np.ndarray):
    """print band-level ranges as a sanity readout."""
    print("\nsanity ranges per band (off-diagonal only):")
    mask_off = ~np.eye(N_ROI, dtype=bool)
    for b_idx, b_name in enumerate(BAND_NAMES):
        w = wpli[:, b_idx][:, mask_off]
        a = aec[:, b_idx][:, mask_off]
        print(f"  {b_name:>5}: wpli mean={w.mean():.3f} sd={w.std():.3f} "
              f"min={w.min():.3f} max={w.max():.3f} | "
              f"aec mean={a.mean():.3f} sd={a.std():.3f} "
              f"min={a.min():.3f} max={a.max():.3f}")


def main():
    wpli, aec, meta = assemble_dataset()
    print_sanity(wpli, aec)
    qc_plot(wpli, aec)
    print("\nstage 1 done.")


if __name__ == "__main__":
    main()
