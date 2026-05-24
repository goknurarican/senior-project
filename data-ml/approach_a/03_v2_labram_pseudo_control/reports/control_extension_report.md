# Control Epoch Extension Report

Generated: 2026-05-22 18:38

## Dataset v2 Summary

| Metric | Value |
|--------|-------|
| Variant epochs (label=1) | 656 |
| Control epochs (label=0) | 796 |
| Total | 1452 |
| Control/Variant ratio | 1.21 |
| EEG shape | (1452, 200) |
| Eye shape | (1452, 6, 110) |
| Mouse shape | (1452, 7, 210) |

## Per-Subject Control Epoch Counts

| Subject | Control Epochs |
|---------|----------------|
| sub-14 | 72 |
| sub-15 | 94 |
| sub-16 | 85 |
| sub-17 | 87 |
| sub-18 | 96 |
| sub-20 | 102 |
| sub-21 | 89 |
| sub-22 | 79 |
| sub-23 | 92 |

## Pipeline Parameters

- Control start: 1.0s (settling), end: first variant onset − 3.5s
- Pseudo-marker step: 5.0s (event_code=99)
- ERP window: -0.2s to +2.0s
- Causal window: -0.5s to +3.0s
- AutoReject: n_interpolate=[1,2,4], random_state=42
- Veli (sub-20): eye features NaN after 900.0s wall time
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
