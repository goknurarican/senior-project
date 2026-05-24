"""
V6 TFR Extraction - Morlet Wavelet ERSP + ROI/Band Aggregation
==============================================================
Processes each subject's epochs in a SINGLE batch call (fast).

For each V3 epoch:
  1. Morlet wavelet TFR (MNE, average=False) -> (n_ep, 32, 30, n_times)
  2. ERSP: per-epoch dB(power) - baseline mean dB  (baseline = own -200 to 0ms)
  3. Spatial ROI averaging -> 6 ROIs
  4. Band aggregation (theta/alpha/beta/gamma) -> 4 bands
  5. FAA dynamic: log(alpha F4) - log(alpha F3) -> 1 extra series
  6. Interpolate to 110 timepoints

Output: features/all_oscillation_v6.npy  (480, 25, 110)
        features/labels_v6.csv
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import mne
from scipy.interpolate import interp1d
warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

V6_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPROACH_A = os.path.dirname(V6_DIR)
PROJECT    = os.path.dirname(APPROACH_A)
FEAT_V3    = os.path.join(APPROACH_A, "features")
FEAT_V6    = os.path.join(V6_DIR, "features")
PROC_DIR   = os.path.join(PROJECT, "data", "processed")

SUBJECTS = [14, 15, 16, 17, 18, 20, 21, 22, 23]

#── Channel layout ─────────────────────────────────────────────────────────
_CH = ['Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','TP9','CP5','CP1',
       'Pz','P3','P7','O1','Oz','O2','P4','P8','TP10','CP6','CP2','Cz',
       'C4','T8','FT10','FC6','FC2','F4','F8','Fp2']

def _idx(names): return [_CH.index(n) for n in names if n in _CH]

ROI_CHANNELS = {
    "frontal":         _idx(['Fp1','Fp2','F3','Fz','F4','F7','F8']),
    "frontal_central": _idx(['FC1','FC2','FC5','FC6']),
    "central":         _idx(['C3','Cz','C4']),
    "parietal":        _idx(['P3','Pz','P4','P7','P8']),
    "occipital":       _idx(['O1','Oz','O2']),
    "temporal":        _idx(['T7','T8','TP9','TP10']),
}
ROI_NAMES  = list(ROI_CHANNELS.keys())
F3_IDX     = _CH.index('F3')
F4_IDX     = _CH.index('F4')

#── Frequency settings ────────────────────────────────────────────────────
FREQS      = np.logspace(np.log10(1), np.log10(40), 30)
N_CYCLES   = np.maximum(FREQS / 2, 1.0)
N_TIMES_OUT = 110

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 40),
}
BAND_NAMES = list(BANDS.keys())   # 4 bands

def band_mask(fmin, fmax): return (FREQS >= fmin) & (FREQS < fmax)

#── Scenario labels ───────────────────────────────────────────────────────
FRUSTRATION_SCENARIOS = sorted([
    'broken_image','button_delay','coupon_expired','coupon_min_spend',
    'facet_reset_once','feedback_late','first_click_miss','network_jitter',
    'overlay_blocking','price_change','search_irrelevant','skeleton_prolong',
    'slow_image','sort_reset'
])
SCENARIO_NAMES = ['control_action_matched'] + FRUSTRATION_SCENARIOS
SCENARIO_TO_LABEL = {s: i for i, s in enumerate(SCENARIO_NAMES)}

#── Core: batch TFR -> ERSP features ────────────────────────────────────
def batch_ersp_features(epochs_data: np.ndarray, sfreq: float = 500.0) -> np.ndarray:
    """
    epochs_data: (N, 32, n_times) float32 at sfreq Hz
    Returns:     (N, 25, 110) ERSP features
    """
    N, n_ch, n_times = epochs_data.shape
    info    = mne.create_info(ch_names=_CH[:n_ch], sfreq=sfreq, ch_types='eeg')
    ep_obj  = mne.EpochsArray(epochs_data, info, tmin=-0.2, verbose=False)

    tfr = mne.time_frequency.tfr_morlet(
        ep_obj, freqs=FREQS, n_cycles=N_CYCLES,
        use_fft=True, return_itc=False, average=False,
        picks='all', verbose=False
    )
    #power: (N, 32, 30, T)
    power    = tfr.data.astype(np.float32)
    tfr_times = tfr.times

    #Baseline per epoch: pre-stimulus window -200ms to 0ms
    bl_mask = (tfr_times >= -0.2) & (tfr_times <= 0.0)
    if bl_mask.sum() < 2:
        bl_mask[:10] = True

    power_db = 10.0 * np.log10(np.maximum(power, 1e-30))    # (N, 32, 30, T)
    bl_mean  = power_db[:, :, :, bl_mask].mean(axis=-1, keepdims=True)   # (N, 32, 30, 1)
    ersp     = power_db - bl_mean                            # (N, 32, 30, T)

    t_out   = np.linspace(-0.2, 2.0, N_TIMES_OUT)
    results = np.zeros((N, 25, N_TIMES_OUT), dtype=np.float32)

    feat_idx = 0
    for roi_name in ROI_NAMES:
        ch_idx  = ROI_CHANNELS[roi_name]
        roi_ersp = ersp[:, ch_idx, :, :].mean(axis=1)   # (N, 30, T)

        for band_name, (fmin, fmax) in BANDS.items():
            bm       = band_mask(fmin, fmax)
            band_ts  = roi_ersp[:, bm, :].mean(axis=1)   # (N, T)

            for i in range(N):
                interp = interp1d(tfr_times, band_ts[i], kind='linear',
                                  fill_value='extrapolate')
                results[i, feat_idx] = interp(t_out)

            feat_idx += 1

    #FAA: log(alpha F4) - log(alpha F3) (uses raw power, not ERSP)
    am = band_mask(8, 13)
    f4 = power[:, F4_IDX, :, :][:, am, :].mean(axis=1)   # (N, T)
    f3 = power[:, F3_IDX, :, :][:, am, :].mean(axis=1)   # (N, T)
    faa = np.log(np.maximum(f4, 1e-30)) - np.log(np.maximum(f3, 1e-30))   # (N, T)
    for i in range(N):
        interp = interp1d(tfr_times, faa[i], kind='linear', fill_value='extrapolate')
        results[i, feat_idx] = interp(t_out)

    return results    # (N, 25, 110)


def build_v2_lookup(v2_meta):
    return {(int(r.subject_id), int(r.epoch_id)): int(r.eeg_index)
            for _, r in v2_meta.iterrows()}


def main():
    v3_meta  = pd.read_csv(os.path.join(FEAT_V3, "all_eeg_embeddings_v3_metadata.csv"))
    v2_meta  = pd.read_csv(os.path.join(FEAT_V3, "all_eeg_embeddings_v2_metadata.csv"))
    v2_lookup = build_v2_lookup(v2_meta)

    #Build the full (480, 25, 110) array in V3 metadata global order
    all_osc  = np.zeros((len(v3_meta), 25, N_TIMES_OUT), dtype=np.float32)
    label_rows = []

    for sid in SUBJECTS:
        smeta   = v3_meta[v3_meta["subject_id"] == sid].copy().reset_index(drop=True)
        print(f"\nsub-{sid}: {len(smeta)} epochs")

        subj_dir = os.path.join(PROC_DIR, f"subject_{sid}")
        ep_ctrl  = mne.read_epochs(
            os.path.join(subj_dir, "epochs_erp_action_matched-epo.fif"),
            preload=True, verbose=False)
        ep_var   = mne.read_epochs(
            os.path.join(subj_dir, "epochs_erp-epo.fif"),
            preload=True, verbose=False)
        ctrl_data = ep_ctrl.get_data().astype(np.float32)
        var_data  = ep_var.get_data().astype(np.float32)

        #Compute TFR in batch for ctrl and var separately
        print(f"  Computing ctrl TFR ({len(ctrl_data)} ep)...")
        ctrl_feat = batch_ersp_features(ctrl_data)

        print(f"  Computing var TFR ({len(var_data)} ep)...")
        var_feat  = batch_ersp_features(var_data)

        #Assign to global array by V3 order
        for _, row in smeta.iterrows():
            g_idx = int(row["global_idx"])
            label = int(row["label_class"])
            eid   = int(row["epoch_id"])
            scen  = str(row["scenario_name"])

            if label == 0:
                fif_idx = min(eid, len(ctrl_feat) - 1)
                all_osc[g_idx] = ctrl_feat[fif_idx]
            else:
                fif_idx = v2_lookup.get((sid, eid), None)
                if fif_idx is None or fif_idx >= len(var_feat):
                    fif_idx = len(var_feat) - 1
                all_osc[g_idx] = var_feat[fif_idx]

            label_rows.append({
                "global_idx": g_idx,
                "subject_id": sid,
                "epoch_id":   eid,
                "scenario_name": scen,
                "label_binary":  label,
                "label_15class": SCENARIO_TO_LABEL.get(scen, 0),
            })

        #Save per-subject intermediate
        per_sub_idx = smeta["global_idx"].values
        np.save(os.path.join(FEAT_V6, "tfr_per_subject", f"sub_{sid}.npy"),
                all_osc[per_sub_idx])
        print(f"  sub-{sid}: saved {len(smeta)} epochs")

    #Sort label_rows by global_idx
    label_rows.sort(key=lambda r: r["global_idx"])

    #Save global arrays
    np.save(os.path.join(FEAT_V6, "all_oscillation_v6.npy"), all_osc)
    labels_df = pd.DataFrame(label_rows)
    labels_df.to_csv(os.path.join(FEAT_V6, "labels_v6.csv"), index=False)

    print(f"\n=== TFR Extraction Complete ===")
    print(f"all_oscillation_v6.npy: {all_osc.shape}")
    print(f"labels_v6.csv: {len(labels_df)} rows")
    print("\nClass distribution (15-class):")
    vc = labels_df.groupby(['label_15class','scenario_name']).size().reset_index(name='n')
    print(vc.to_string(index=False))


if __name__ == "__main__":
    main()
