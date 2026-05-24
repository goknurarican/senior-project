# 01 Preprocessing

Raw BrainVision EEG (.eeg/.vhdr/.vmrk) and behavioral logs processed to
analysis-ready epochs.

## Pipeline

- MNE-Python 1.x, 500Hz, 32-channel montage
- Bandpass 0.1-100Hz, notch 50Hz
- ICA artifact rejection (eye blink, muscle)
- Epoch extraction: -200ms to 2000ms around scenario triggers
- Action-matched control epoch extraction (S30/S32 markers, 3s gap filter)
- Epoch rejection: peak-to-peak > 150uV

## How to run

```bash
python src/section4_preprocessing.py
python src/section5_eye_preprocessing.py
python src/section6_mouse_preprocessing.py
python src/section7_sync_validation.py
```

Run from project root. Outputs go to data/processed/subject_*/.

## Files

- `src/section2_corrections.py` - manual corrections per subject
- `src/section3_eda.py`, `section3_post_eda.py` - exploratory analysis
- `src/section4_preprocessing.py` - main EEG pipeline
- `src/section4b_recep_variants.py` - variant epoch extraction
- `src/section5_eye_preprocessing.py` - eye tracking
- `src/section6_mouse_preprocessing.py` - mouse events
- `src/section7_sync_validation.py` - cross-modality sync check
- `src/preprocess.py`, `preprocess_eye.py` - shared utilities
- `src/clean_prep.py` - data cleaning decisions
- `src/marker_analysis.py` - EEG marker inspection
- `src/config.py`, `data_stats.py`, `inventory.py` - config and stats

## Outputs

data/processed/subject_*/epochs_erp-epo.fif
data/processed/subject_*/epochs_erp_action_matched-epo.fif
