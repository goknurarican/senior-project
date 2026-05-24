"""
Step 1: Oscillation Time Series Extraction for V5 Pipeline.

For each V3 epoch (via metadata), loads the corresponding FIF file
and computes 6 oscillation time series using Morlet wavelets:
  1. frontal_theta    4-8 Hz,  [Fp1,Fp2,F3,Fz,F4,FC1,FC2]
  2. frontal_alpha    8-13 Hz, same frontal channels
  3. parietal_alpha   8-13 Hz, [P3,Pz,P4,P7,P8]
  4. central_beta    13-30 Hz, [C3,Cz,C4,CP1,CP2]
  5. faa_dynamic      log(alpha_F4) - log(alpha_F3) per timepoint
  6. engagement_index beta / (alpha + theta) per timepoint, frontal

Output: (480, 6, 110) array  →  approach_a/v5_hybrid/features/all_oscillation_v5.npy
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import mne
from scipy.interpolate import interp1d
warnings.filterwarnings("ignore")

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
V5_DIR      = os.path.dirname(SRC_DIR)
APPROACH_A  = os.path.dirname(V5_DIR)
PROJECT_ROOT = os.path.dirname(APPROACH_A)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FEAT_V3  = os.path.join(APPROACH_A, "features")
OUT_DIR  = os.path.join(V5_DIR, "features")
OSC_TS_DIR = os.path.join(OUT_DIR, "oscillation_timeseries")
os.makedirs(OSC_TS_DIR, exist_ok=True)

N_OUT    = 110          # target timepoints (matches eye_ts)
SFREQ    = 500.0
EPS      = 1e-10

# 32-channel layout (fixed order)
ALL_CH = ['Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','TP9',
          'CP5','CP1','Pz','P3','P7','O1','Oz','O2','P4','P8',
          'TP10','CP6','CP2','Cz','C4','T8','FT10','FC6','FC2','F4','F8','Fp2']

CH_IDX = {ch: i for i, ch in enumerate(ALL_CH)}

FRONTAL_IDX  = [CH_IDX[c] for c in ['Fp1','Fp2','F3','Fz','F4','FC1','FC2']]
PARIETAL_IDX = [CH_IDX[c] for c in ['P3','Pz','P4','P7','P8']]
CENTRAL_IDX  = [CH_IDX[c] for c in ['C3','Cz','C4','CP1','CP2']]
F3_IDX       = CH_IDX['F3']
F4_IDX       = CH_IDX['F4']

# Morlet frequencies
FREQS_THETA = np.arange(4,  9,  1.).astype(float)   # [4,5,6,7,8]
FREQS_ALPHA = np.arange(8,  14, 1.).astype(float)   # [8..13]
FREQS_BETA  = np.arange(13, 31, 2.).astype(float)   # [13,15,...,29]
ALL_FREQS   = np.concatenate([FREQS_THETA, FREQS_ALPHA, FREQS_BETA])
N_THETA     = len(FREQS_THETA)
N_ALPHA     = len(FREQS_ALPHA)
N_CYCLES    = 4


def morlet_tfr_batch(data, sfreq=500.0):
    """
    Compute Morlet TFR for a batch of epochs.
    data: (n_epochs, n_ch, n_times)
    Returns: (n_epochs, n_ch, n_freqs, n_times)  - power
    """
    return mne.time_frequency.tfr_array_morlet(
        data, sfreq=sfreq, freqs=ALL_FREQS,
        n_cycles=N_CYCLES, output='power', verbose=False
    )


def epoch_to_osc(tfr_single):
    """
    Convert single-epoch TFR → 6 oscillation time series, interpolated to N_OUT.
    tfr_single: (n_ch, n_freqs, n_times)
    Returns: (6, N_OUT)
    """
    theta_bp = tfr_single[:, :N_THETA,          :].mean(axis=1)  # (n_ch, T)
    alpha_bp = tfr_single[:, N_THETA:N_THETA+N_ALPHA, :].mean(axis=1)
    beta_bp  = tfr_single[:, N_THETA+N_ALPHA:,   :].mean(axis=1)

    log_theta = np.log(np.maximum(theta_bp, EPS))
    log_alpha = np.log(np.maximum(alpha_bp, EPS))
    log_beta  = np.log(np.maximum(beta_bp,  EPS))

    # 1. frontal_theta (log)
    ft = log_theta[FRONTAL_IDX, :].mean(0)
    # 2. frontal_alpha (log)
    fa = log_alpha[FRONTAL_IDX, :].mean(0)
    # 3. parietal_alpha (log)
    pa = log_alpha[PARIETAL_IDX, :].mean(0)
    # 4. central_beta (log)
    cb = log_beta[CENTRAL_IDX, :].mean(0)
    # 5. FAA: log(alpha_F4) - log(alpha_F3)
    faa = log_alpha[F4_IDX, :] - log_alpha[F3_IDX, :]
    # 6. Engagement index: frontal beta / (frontal alpha + frontal theta)
    f_a_raw = alpha_bp[FRONTAL_IDX, :].mean(0)
    f_t_raw = theta_bp[FRONTAL_IDX, :].mean(0)
    f_b_raw = beta_bp[FRONTAL_IDX,  :].mean(0)
    eng = f_b_raw / (f_a_raw + f_t_raw + EPS)

    features = np.stack([ft, fa, pa, cb, faa, eng], axis=0)  # (6, T_orig)

    # Interpolate to N_OUT
    T = features.shape[1]
    t_orig = np.linspace(0., 1., T)
    t_new  = np.linspace(0., 1., N_OUT)
    out = np.zeros((6, N_OUT), dtype=np.float32)
    for i in range(6):
        f_interp = interp1d(t_orig, features[i], kind='linear')
        out[i] = f_interp(t_new).astype(np.float32)
    return out


def process_subject(sid, meta_sub):
    """
    Process all V3 epochs for one subject.
    Returns: (n_epochs_sub, 6, N_OUT)  ordered by global_idx
    """
    subj_dir = os.path.join(DATA_DIR, f"subject_{sid}")

    # Load both epoch files
    ctrl_path    = os.path.join(subj_dir, "epochs_erp_action_matched-epo.fif")
    variant_path = os.path.join(subj_dir, "epochs_erp-epo.fif")

    ctrl_epochs    = mne.read_epochs(ctrl_path,    preload=True, verbose=False)
    variant_epochs = mne.read_epochs(variant_path, preload=True, verbose=False)

    # Ensure channel order matches ALL_CH
    ctrl_epochs    = ctrl_epochs.reorder_channels(ALL_CH)
    variant_epochs = variant_epochs.reorder_channels(ALL_CH)

    ctrl_data    = ctrl_epochs.get_data()    # (N_ctrl, 32, 1101)
    variant_data = variant_epochs.get_data()  # (N_var, 32, 1101)

    # Compute TFR in batches
    ctrl_tfr    = morlet_tfr_batch(ctrl_data)    # (N_ctrl, 32, n_freqs, 1101)
    variant_tfr = morlet_tfr_batch(variant_data) # (N_var,  32, n_freqs, 1101)

    # Build output in metadata order
    n_sub = len(meta_sub)
    out   = np.zeros((n_sub, 6, N_OUT), dtype=np.float32)

    for row_idx, row in enumerate(meta_sub.itertuples()):
        phase    = row.phase        # 'control' or 'variant_X'
        epoch_id = row.epoch_id     # index into the respective file

        if phase == 'control':
            tfr_single = ctrl_tfr[min(epoch_id, len(ctrl_tfr)-1)]
        else:
            tfr_single = variant_tfr[min(epoch_id, len(variant_tfr)-1)]

        out[row_idx] = epoch_to_osc(tfr_single)

    # Save per-subject cache
    np.save(os.path.join(OSC_TS_DIR, f"sub_{sid}_osc.npy"), out)
    return out


def qc_check(arr, name):
    """Basic quality check."""
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    print(f"  {name}: shape={arr.shape}  min={arr.min():.3f}  max={arr.max():.3f}  "
          f"mean={arr.mean():.3f}  NaN={n_nan}  Inf={n_inf}")


def main():
    meta = pd.read_csv(os.path.join(FEAT_V3, "all_eeg_embeddings_v3_metadata.csv"))
    lab  = pd.read_csv(os.path.join(FEAT_V3, "labels_v3.csv"))
    subjects = sorted(meta["subject_id"].unique())

    all_osc = []
    for sid in subjects:
        print(f"Processing sub-{sid} ...")
        meta_sub = meta[meta["subject_id"] == sid].sort_values("global_idx").reset_index(drop=True)
        osc_sub  = process_subject(sid, meta_sub)
        all_osc.append(osc_sub)
        print(f"  → {osc_sub.shape}")

    all_osc_arr = np.concatenate(all_osc, axis=0)  # (480, 6, 110)
    print(f"\nFinal array: {all_osc_arr.shape}")

    # Quality checks per feature
    feat_names = ['frontal_theta', 'frontal_alpha', 'parietal_alpha',
                  'central_beta', 'faa_dynamic', 'engagement_index']
    print("\n=== Quality Checks ===")
    for i, fn in enumerate(feat_names):
        qc_check(all_osc_arr[:, i, :], fn)

    # Replace any NaN/Inf with 0
    n_bad = (~np.isfinite(all_osc_arr)).sum()
    if n_bad > 0:
        print(f"WARNING: replacing {n_bad} NaN/Inf values with 0")
        all_osc_arr = np.nan_to_num(all_osc_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Sanity check: variant vs control distributions
    labels = lab["label"].values
    print("\n=== Sanity Check: Variant vs Control Feature Means ===")
    for i, fn in enumerate(feat_names):
        v_mean = all_osc_arr[labels==1, i, :].mean()
        c_mean = all_osc_arr[labels==0, i, :].mean()
        print(f"  {fn:25s}: variant={v_mean:.3f}  control={c_mean:.3f}  diff={v_mean-c_mean:+.3f}")

    # Save
    out_path = os.path.join(OUT_DIR, "all_oscillation_v5.npy")
    np.save(out_path, all_osc_arr)
    print(f"\nSaved → {out_path}  ({all_osc_arr.nbytes/1e6:.1f} MB)")

    # Quick visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for i, (fn, ax) in enumerate(zip(feat_names, axes.flat)):
            v_ts = all_osc_arr[labels==1, i, :].mean(0)
            c_ts = all_osc_arr[labels==0, i, :].mean(0)
            t = np.linspace(0, 2, N_OUT)
            ax.plot(t, v_ts, 'r-', label='Variant', lw=1.5)
            ax.plot(t, c_ts, 'b-', label='Control', lw=1.5)
            ax.set_title(fn); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)
        fig.suptitle('Oscillation Features: Variant vs Control (grand mean)')
        fig.tight_layout()
        qc_fig = os.path.join(V5_DIR, "reports", "oscillation_qc.png")
        fig.savefig(qc_fig, dpi=120); plt.close(fig)
        print(f"QC plot saved → {qc_fig}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
