#Stage 1: Connectivity Feature Extraction

##Inputs

- Per-subject MNE Epochs files under `data/processed/subject_<sid>/`:
  - `epochs_erp_action_matched-epo.fif` for control epochs
  - `epochs_erp-epo.fif` for variant epochs
- V3 metadata `approach_a/04_v3_labram_action_matched/features/all_eeg_embeddings_v3_metadata.csv`
- V2 metadata `approach_a/03_v2_labram_pseudo_control/features/all_eeg_embeddings_v2_metadata.csv`
- V6 labels for the scenario-name mapping
  `approach_a/06_v6_multiclass_characterization/features/labels_v6.csv`

##What it computes

For each of the 480 V3 epochs and for each of four bands (theta, alpha, beta,
gamma):

- weighted phase lag index (wPLI) per channel pair via Morlet wavelets
- amplitude envelope correlation (AEC) per channel pair via Hilbert envelope

Both metrics are then collapsed into 6 ROIs (frontal, frontal-central,
central, parietal, occipital, temporal). Within-ROI values are computed
from cross-channel pairs that lie inside the same ROI; between-ROI values
average all pairs between the two ROIs.

##Outputs

`features/connectivity_per_epoch/`:

- `all_wpli_v3.npy`  shape `(480, 4, 6, 6)` float32
- `all_aec_v3.npy`   shape `(480, 4, 6, 6)` float32
- `labels_v3.csv`    copy of V3 binary labels for sanity matching
- `labels_v6.csv`    scenario-name labels used by Stage 2 and Stage 3
- `connectivity_metadata.csv` per-epoch (subject, epoch, scenario)

`reports/connectivity_qc.png` is a 4 by 4 grid of band-level histograms
and average ROI matrices for both metrics, used to sanity-check the
extraction.

##Run

```
python approach_b/01_connectivity_extraction/src/compute_connectivity.py
```

Estimated runtime: 30 to 45 minutes on an M1 Mac (wPLI is the slow step).
