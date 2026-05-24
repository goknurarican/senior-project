# Approach A - Pipeline Sanity Check Report

Generated: 2026-05-13

## Configuration
- Test subject: sub-18 (Feyiz Burak Öztürk)
- Train epochs: 5
- Device: CPU
- Label: `label_rage_click`
- Model: HusformerBITIRMEEG (132,130 params)

## Forward Pass
- EEG embeddings shape: (75, 200) ✓
- Eye timeseries shape: (75, 6, 110) ✓
- Mouse timeseries shape: (75, 7, 210) ✓
- Logits shape: (B, 2) ✓

## Data Split
| Set   | N   | Rage-click=0 | Rage-click=1 |
|-------|-----|--------------|--------------|
| Train | 581 | 553          | 28           |
| Test  | 75  | 73           | 2            |

## Sanity Check Results

| Metric   | Value | Note |
|----------|-------|------|
| Accuracy | 0.973 | Majority-class baseline (all-zero prediction) |
| F1       | 0.000 | Expected: model hasn't learned minority class in 5 epochs |
| AUC      | N/A   | Too few positive test samples (2/75) for stable AUC |

## Assessment
**PASS** - The full pipeline ran without errors:
1. LaBraM loaded 145/221 pretrained keys (all 12 transformer blocks)
2. Feature extraction completed for 9/9 subjects (656 total epochs)
3. Forward pass, backward pass, and gradient flow confirmed
4. Metrics are expected given 5 epochs + 4.6% positive class rate

## Notes for Full LOSO Run
- Add `class_weight` to CrossEntropyLoss to handle 4.6% positive rate
- Increase epochs to ≥50 (or use early stopping on val loss)
- Consider augmentation (Gaussian noise on EEG embeddings, flip gaze horizontally)
- `label_frustration` is all-1 (all epochs are frustration scenarios) - use `label_rage_click` or a new label derived from physiological response
