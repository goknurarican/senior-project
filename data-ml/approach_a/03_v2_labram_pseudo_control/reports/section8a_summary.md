# Section 8a - Approach A Summary

Generated: 2026-05-13

## Overview

Approach A implements a cross-modal transformer (LaBraM + Husformer) for multimodal
frustration classification using EEG, eye-tracking, and mouse behavior.

## Architecture

### LaBraM (EEG backbone)
- Model: Labram(n_chans=32, n_times=400, sfreq=200, n_outputs=2, n_layers=12, emb_size=200)
- Pretrained: braindecode/labram-pretrained (HuggingFace, 22.8 MB)
- Weight loading: 145/221 keys compatible; all 12 transformer block attn/MLP weights loaded
  - Incompatible (randomly initialized): position_embedding, temporal_embedding (32ch vs 128ch pretrained)
- Input: (B, 32, 1101) ERP epoch at 500 Hz → resample to 200 Hz → first 400 samples
- Output: (B, 200) embedding via fc_norm hook

### Husformer (cross-modal fusion)
- Architecture: 3 modality encoders + symmetric cross-modal attention + classifier head
- EEG branch: Linear(200 → 64)
- Eye branch: Conv1d stack (6, 110) → 64
- Mouse branch: Conv1d stack (7, 210) → 64
- Cross-modal attention: each modality queries the other two (MultiheadAttention, 4 heads)
- Classifier: Linear(192, 128) → GELU → Dropout(0.3) → Linear(128, 2)
- **Total parameters: 132,130**

## Feature Shapes

| Feature | Shape | Window | Sampling |
|---------|-------|--------|----------|
| EEG embeddings | (656, 200) | −200ms..+2000ms | LaBraM 200-dim |
| Eye timeseries | (656, 6, 110) | −200ms..+2000ms | 50 Hz |
| Mouse timeseries | (656, 7, 210) | −200ms..+4000ms | 50 Hz |

### Eye channels (6)
0. gaze_x (bpogv-masked, interpolated)
1. gaze_y (bpogv-masked, interpolated)
2. fixation_flag (from eye_fixations.csv)
3. blink_flag (pupil_left=0 AND pupil_right=0)
4. nan_flag (bpogv=0)
5. gaze_velocity (deg/s, computed from adjacent valid samples)

### Mouse channels (7)
0. x_norm (0–1)
1. y_norm (0–1)
2. velocity_px_s
3. acceleration_px_s2
4. is_idle (velocity < 50 px/s)
5. click_flag (binary: any click in 20ms bin)
6. rage_click_flag (binary)

## Per-subject Epoch Counts

| Subject | Epochs |
|---------|--------|
| 14 Alen Maryo | 66 |
| 15 Eren Tamparlak | 68 |
| 16 Berk Uygun | 78 |
| 17 Mehmet İncekara | 72 |
| 18 Feyiz Burak Öztürk | 75 |
| 20 Veli Barış Sevinçhan | 72 |
| 21 Enis Tiren | 69 |
| 22 Recep Danacı | 75 |
| 23 Duru Erol | 81 |
| **Total** | **656** |

## Labels

| Label column | Positive rate | Notes |
|---|---|---|
| `label_frustration` | 100% (656/656) | All epochs are frustration triggers (S11-S24) |
| `label_rage_click` | 4.6% (30/656) | Per-epoch rage click from mouse features |

For LOSO training, use `label_rage_click` (or define new label based on physiological response).
Class weights recommended due to imbalance.

## Files

```
approach_a/
├── features/
│   ├── all_eeg_embeddings.npy    (656, 200)  float32
│   ├── all_eye_timeseries.npy    (656, 6, 110)  float32
│   ├── all_mouse_timeseries.npy  (656, 7, 210)  float32
│   └── epoch_metadata.csv        (656 rows, 12 cols)
├── models/
│   └── pytorch_model.bin         LaBraM-Base pretrained weights (22.8 MB)
├── src/
│   ├── labram_wrapper.py          LabramEncoder with fc_norm hook
│   ├── husformer.py               HusformerBITIRMEEG (132k params)
│   └── extract_features_8a.py    Full extraction pipeline
├── training/
│   └── train_loso.py             LOSO skeleton (9-fold, argparse)
└── reports/
    ├── section8a_summary.md
    └── sanity_check_report.md
```

## Sanity Check

- Test fold: sub-18 (Feyiz), label: rage_click, 5 training epochs
- Forward + backward pass: **PASS**
- Pipeline end-to-end: **PASS**
- Metrics: acc=0.973 (majority-class baseline), F1=0.000 (expected at 5 epochs with 4.6% positive)

## Readiness for Full LOSO
**YES** - All components functional. Before full run:
1. Add class weights to CrossEntropyLoss (pos_weight ≈ 20×)
2. Increase epochs to ≥50 with early stopping
3. Decide final label strategy (rage_click vs. new physiological label)
