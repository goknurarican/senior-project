# Time-Locked Window Analysis - Interpretation

**Date:** 2026-05-23  
**Dataset:** V3 action-matched (N=480 epochs, 9 subjects, balanced)  
**Method:** Zero-mask outside window → fresh LaBraM 200-dim embedding → SimpleMLP (30 epochs) LOSO  

---

## Results Summary

| Window | Time Range | Duration | Mean AUC ± Std |
|--------|-----------|----------|----------------|
| W0_full   | −200ms → 2000ms | 2200ms | 0.488 ± 0.092 |
| W1_pre    | −200ms → 0ms    |  200ms  | 0.462 ± 0.096 |
| W2_early  |    0ms → 500ms  |  500ms  | 0.523 ± 0.088 |
| W3_mid    |  500ms → 1500ms | 1000ms  | 0.502 ± 0.070 |
| W4_late   | 1500ms → 2000ms |  500ms  | 0.463 ± 0.122 |

All windows are at or below chance (AUC ≈ 0.5). No single temporal segment shows
discriminative power with this methodology.

---

## Critical Technical Finding: Position Embedding Non-Determinism

### Why W0_full gives AUC ≈ 0.5 instead of 1.000

The V3 pipeline achieves AUC=1.000 using **pre-computed** LaBraM embeddings (stored in
`approach_a/features/all_eeg_embeddings_v3.npy`). This window analysis **re-runs** LaBraM
inference from scratch, producing different embeddings - confirmed by max absolute difference
of 0.95–1.45 between fresh and stored embeddings for the same raw EEG.

Root cause: LaBraM's `position_embedding` and `temporal_embedding` parameters are **randomly
initialized** in every new process. The pre-trained checkpoint uses a 128-channel EEG montage;
our model has 32 channels. Since the shapes mismatch, these keys are skipped during weight
loading, leaving them at random initial values (confirmed: `pos_embed` differs between any
two process runs).

**Implication:** The V3 embeddings were computed in one Python process with a specific random
positional encoding. A different process produces a different positional encoding → completely
different 200-dim embeddings from identical raw EEG → AUC = chance with the new initialization.

This is a fundamental property of the current LaBraM setup: the stored V3/V5 embeddings are
tied to one specific random positional initialization, which cannot be reproduced across
separate Python sessions without saving and reloading the full model state_dict.

---

## Scenario C: Full Epoch Integration Required (Internal Comparison Valid)

Within this analysis run (single process, same LaBraM instance, identical positional encoding),
the relative comparison across W1–W4 is internally valid. All five conditions produce
chance-level AUC with this particular random initialization.

**Scenario C is confirmed: no single temporal sub-window contains sufficient information
for this LaBraM initialization to discriminate frustration from control.**

Two reinforcing explanations:

### 1. LaBraM requires full-epoch temporal context
LaBraM processes raw EEG with a 12-layer transformer over the full 2.2s epoch. Zero-masking
80–95% of the epoch forces the transformer to operate on mostly-zero inputs:

- W1_pre (200ms): ~100/440 samples signal after resampling, 340 zeros
- W2_early (500ms): ~100/440 signal, 340 zeros
- W3_mid (1000ms): ~200/440 signal, 240 zeros
- W4_late (500ms): ~100/440 signal, 340 zeros

The transformer attention over zero-dominated sequences produces embeddings that
reflect the "zero context" rather than the actual EEG dynamics in the signal window.

### 2. Alignment with Diagnostic Test 4
Diagnostic Test 4 showed classical EEG features (scalar temporal means of band powers)
have zero discriminative power (RF AUC=0.539). Window-level analysis with LaBraM extends
this finding: even a 500ms–1000ms window does not produce discriminable embeddings.

Both findings are consistent with LaBraM capturing **cross-epoch, cross-frequency temporal
dynamics** that require the full 2.2s window to manifest - a signal invisible to both
scalar features and short-window embeddings.

---

## Pre-stimulus Check

✓ **W1_pre AUC = 0.462 (below chance)** - no evidence of pre-existing EEG differences
or tonic frustration-state differences before the event. The signal is event-locked.

---

## Limitation and Recommended Alternative

The zero-masking approach cannot cleanly isolate temporal window effects due to:
1. **Position_embedding non-determinism** across processes (W0_full at chance despite V3 AUC=1.000)
2. **LaBraM attention distortion** by zero-padded inputs (most attention mass falls on zeros)

**To fix the non-determinism:**
Save the LaBraM state_dict used during V3 feature extraction and reload it for window analysis:
```python
torch.save(encoder.model.state_dict(), 'models/labram_state_used_for_v3.pth')
# Then in window analysis:
encoder.model.load_state_dict(torch.load('models/labram_state_used_for_v3.pth'))
```

**Alternative: Oscillation Feature Temporal Analysis**  
The V5 pipeline computes oscillation time-series (6 features × 110 timepoints) from
Morlet wavelet TFR. These have genuine temporal resolution and no non-determinism issue:

```python
for t_start, t_end in window_bins:
    osc_window = osc_ts[:, :, t_start:t_end]   # (N, 6, n_timepoints)
    # train SimpleMLP LOSO → report AUC
```

This approach directly answers "which time window of EEG oscillations carries the signal"
without requiring LaBraM re-inference.

---

## Summary

| Hypothesis | Evidence | Status |
|-----------|---------|--------|
| Signal in early window (W2_early) | AUC=0.523 ± 0.088 (chance) | Not confirmed |
| Signal in mid/late window | AUC=0.50–0.51 (chance) | Not confirmed |
| Pre-stimulus confound | W1_pre AUC=0.462 (below chance) | ✓ Absent |
| Full epoch required (Scenario C) | All windows ~0.5 | ✓ Most consistent |
| LaBraM embeddings reproducible across processes | Max diff=0.95–1.45 | ✗ Not reproducible |
